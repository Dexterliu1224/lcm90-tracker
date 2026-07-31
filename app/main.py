"""FastAPI 后端：把 TrackingSession 暴露给网页。

路由都很薄 —— 业务全在 core/control.py 的状态机里，
这里只做参数校验和错误翻译（异常 → 用户能看懂的中文提示）。
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import (FileResponse, JSONResponse, RedirectResponse,
                               StreamingResponse)
from pydantic import BaseModel

from core.auth import (DEFAULT_PASSWORD, DEFAULT_USER, AuthError, SessionStore,
                       UserStore)
from core.control import TrackingSession
from core.camera import list_video_devices
from core.recorder import Recorder, RecorderError

log = logging.getLogger("app")

import sys

# 两类路径要分开对待：
#   静态资源（index.html）打包在 exe 里，跟着 __file__（PyInstaller 解压目录）；
#   config.yaml 必须在 exe 旁边，用户改了才生效。
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(_PKG_ROOT, "app", "static")

if getattr(sys, "frozen", False):
    ROOT = os.path.dirname(os.path.abspath(sys.executable))
else:
    ROOT = _PKG_ROOT


def load_cfg() -> Dict[str, Any]:
    with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


_CFG = load_cfg()
session = TrackingSession(_CFG)
app = FastAPI(title="LCM90 视觉跟踪")


#: 运行数据（账号、录像）改放到这个目录。设了它就不碰程序目录 ——
#: 自动化测试靠它隔离，同一台机器跑多个实例时也用得上。
_DATA_OVERRIDE = os.environ.get("LCM90_DATA_DIR", "").strip()


def _under_root(rel: str) -> str:
    """把配置里的相对路径按「程序所在目录」解释。

    打包成 exe 后 cwd 是用户双击时所在的任意目录，靠 cwd 解释相对路径
    会让账号和录像散落到桌面/C:\\Windows\\System32 之类的地方。
    """
    if os.path.isabs(rel):
        return rel
    if _DATA_OVERRIDE:
        return os.path.join(_DATA_OVERRIDE, os.path.basename(rel))
    return os.path.join(ROOT, rel)


_auth_cfg = dict(_CFG.get("auth", {}) or {})
users = UserStore(_under_root(str(_auth_cfg.get("file", "data/users.json"))))
sessions = SessionStore()
SESSION_COOKIE = "lcm90_session"

_rec_cfg = dict(_CFG.get("recording", {}) or {})
recorder = Recorder(_under_root(str(_rec_cfg.get("dir", "data/recordings"))),
                    fps=float(_rec_cfg.get("fps", 15)),
                    max_minutes=float(_rec_cfg.get("max_minutes", 60)))

#: 不需要登录就能访问的路径。**只有**登录页和登录接口 ——
#: /video 是实时画面，/api/* 能操控望远镜，一个都不能漏在外面。
_PUBLIC_PATHS = {"/login", "/api/login", "/favicon.ico"}


@app.middleware("http")
async def _require_login(request: Request, call_next):
    """全局登录闸。用中间件而不是逐个路由加依赖：漏加一个依赖就是一个
    敞开的口子，而中间件是默认关闭、显式放行。"""
    path = request.url.path
    if path in _PUBLIC_PATHS:
        return await call_next(request)

    username = sessions.username_for(request.cookies.get(SESSION_COOKIE))
    if username is None:
        if path.startswith("/api/"):
            # 接口返回 401 让前端自己跳转；返回重定向的话，fetch 会跟过去
            # 拿到一个 HTML，前端解析 JSON 时报一个莫名其妙的语法错。
            return JSONResponse({"detail": "登录已过期，请重新登录。"},
                                status_code=401)
        return RedirectResponse("/login", status_code=302)

    request.state.username = username
    return await call_next(request)


def _current_user(request: Request) -> str:
    # 中间件已经保证走到这里的请求一定带着有效会话
    return str(getattr(request.state, "username", ""))


def _require_admin(request: Request) -> str:
    username = _current_user(request)
    if users.role_of(username) != "admin":
        raise HTTPException(403, "只有管理员能管理账号。")
    return username


# ================================================================ 请求模型
class ConnectReq(BaseModel):
    port: Optional[str] = None
    sim: bool = False


class CameraReq(BaseModel):
    # int = 普通摄像头设备号；"qhy:0" = QHY 相机；None + sim = 仿真
    source: Optional[Any] = None
    sim: bool = False
    preset: str = "eyepiece"


def parse_camera_source(req: "CameraReq") -> Dict[str, Any]:
    """把界面的选择翻译成 build_camera 的配置覆盖。

    界面的选择必须覆盖 config 的 driver —— 相机和基座犯过同一个病：
    config 默认 simulator 时选真摄像头，建出来的还是仿真画面。
    """
    if req.sim or req.source is None:
        return {"driver": "simulator"}
    s = str(req.source).strip()
    if s.lower().startswith("qhy:"):
        return {"driver": "qhy", "index": int(s.split(":", 1)[1] or 0)}
    return {"driver": "opencv", "source": int(s) if s.lstrip("-").isdigit() else s}


class SelectReq(BaseModel):
    box: Tuple[float, float, float, float]     # 归一化 x,y,w,h


class NudgeReq(BaseModel):
    d_az: float
    d_alt: float


class LoginReq(BaseModel):
    username: str
    password: str


class PasswordReq(BaseModel):
    old_password: str
    new_password: str


class NewUserReq(BaseModel):
    username: str
    password: str
    role: str = "user"


class ResetPwdReq(BaseModel):
    password: str


# ================================================================ 页面与视频
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/login")
def login_page():
    return FileResponse(os.path.join(STATIC, "login.html"))


# ================================================================ 登录与账号
@app.post("/api/login")
def api_login(req: LoginReq, response: Response) -> Dict[str, Any]:
    username = (req.username or "").strip()
    if not users.verify(username, req.password or ""):
        # 刻意不说"用户名不存在"还是"密码错了"：那等于帮人枚举账号
        log.warning("登录失败：%s", username or "(空用户名)")
        raise HTTPException(401, "用户名或密码不对。")
    token = sessions.create(username)
    response.set_cookie(
        SESSION_COOKIE, token, httponly=True, samesite="lax",
        # 本机 http 访问，secure 必须为 False，否则浏览器根本不存这个 Cookie
        secure=False, max_age=None)
    log.info("登录成功：%s", username)
    return {"ok": True, "username": username,
            "role": users.role_of(username),
            "is_default_password": users.uses_default_password(username)}


@app.post("/api/logout")
def api_logout(request: Request, response: Response) -> Dict[str, Any]:
    sessions.drop(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True, "message": "已退出登录。"}


@app.get("/api/me")
def api_me(request: Request) -> Dict[str, Any]:
    username = _current_user(request)
    return {"username": username, "role": users.role_of(username),
            "is_default_password": users.uses_default_password(username),
            "default_user": DEFAULT_USER, "default_password": DEFAULT_PASSWORD}


@app.post("/api/password")
def api_password(req: PasswordReq, request: Request,
                 response: Response) -> Dict[str, Any]:
    username = _current_user(request)
    try:
        users.change_password(username, req.old_password, req.new_password)
    except AuthError as exc:
        raise HTTPException(400, str(exc))
    # 改完密码要把自己**其它**会话踢掉（比如别的机器上还开着），
    # 然后立刻给当前浏览器换一张新票，不然改完就被自己踢下线。
    sessions.drop_user(username)
    response.set_cookie(SESSION_COOKIE, sessions.create(username),
                        httponly=True, samesite="lax", secure=False)
    return {"ok": True, "message": "密码已修改。"}


@app.get("/api/users")
def api_users(request: Request) -> Dict[str, Any]:
    _require_admin(request)
    return {"users": users.list_users(), "me": _current_user(request)}


@app.post("/api/users")
def api_add_user(req: NewUserReq, request: Request) -> Dict[str, Any]:
    _require_admin(request)
    try:
        users.add_user((req.username or "").strip(), req.password, req.role)
    except AuthError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True, "message": "已新建账号「%s」。" % req.username}


@app.post("/api/users/{username}/password")
def api_reset_password(username: str, req: ResetPwdReq,
                       request: Request) -> Dict[str, Any]:
    _require_admin(request)
    try:
        users.reset_password(username, req.password)
    except AuthError as exc:
        raise HTTPException(400, str(exc))
    sessions.drop_user(username)     # 被重置的人必须重新登录
    return {"ok": True, "message": "已重置「%s」的密码。" % username}


@app.delete("/api/users/{username}")
def api_delete_user(username: str, request: Request) -> Dict[str, Any]:
    me = _require_admin(request)
    if username == me:
        raise HTTPException(400, "不能删除自己正在用的账号。")
    try:
        users.delete_user(username)
    except AuthError as exc:
        raise HTTPException(400, str(exc))
    sessions.drop_user(username)
    return {"ok": True, "message": "已删除账号「%s」。" % username}


# ================================================================ 录像
@app.post("/api/record/start")
def api_record_start(request: Request) -> Dict[str, Any]:
    try:
        st = recorder.start(session.annotated_frame,
                            name_hint=_current_user(request))
    except RecorderError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True, "message": "开始录像：%s" % st.get("file"), "record": st}


@app.post("/api/record/stop")
def api_record_stop() -> Dict[str, Any]:
    st = recorder.stop()
    if not st.get("file"):
        return {"ok": True, "message": "当前没有在录像。", "record": st}
    return {"ok": True,
            "message": "录像已保存：%s（%.1f MB）" % (st["file"], st["size_mb"]),
            "record": st}


@app.get("/video")
def video():
    def gen():
        while True:
            frame = session.annotated_frame()
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                if ok:
                    data = buf.tobytes()
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(data)).encode()
                           + b"\r\n\r\n" + data + b"\r\n")
            time.sleep(1 / 18)      # 网页 18fps 足够
    return StreamingResponse(gen(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


# ================================================================ 设备
@app.get("/api/ports")
def api_ports() -> Dict[str, Any]:
    ports: List[Dict[str, str]] = []
    try:
        import serial.tools.list_ports as lp
        for p in lp.comports():
            ports.append({"device": p.device, "description": p.description or ""})
    except Exception as exc:
        log.warning("串口扫描失败: %s", exc)
    return {"ports": ports}


@app.get("/api/qhy/diagnose")
def api_qhy_diagnose() -> Dict[str, Any]:
    """QHY 排查专用：把 SDK 加载、扫描、每一台的 ID 全暴露出来。

    用户遇到过「EZCAP 能看到相机、本软件列表为空、也没有任何报错」的情况，
    光看界面无从下手 —— 浏览器打开 /api/qhy/diagnose 就能拿到真实原因。
    """
    import glob
    out: Dict[str, Any] = {}
    out["dll_found"] = (glob.glob("C:/Windows/System32/qhyccd*")
                        + glob.glob("C:/Windows/SysWOW64/qhyccd*"))
    try:
        from core import qhy
        sdk, err = qhy._load_sdk()
        out["sdk_loaded"] = sdk is not None
        out["sdk_error"] = err
        if sdk is not None:
            try:
                n = int(sdk.ScanQHYCCD())
                out["scan_count"] = n
                ids = []
                import ctypes
                for i in range(n):
                    buf = ctypes.create_string_buffer(64)
                    rc = sdk.GetQHYCCDId(ctypes.c_uint32(i), buf)
                    ids.append({"index": i, "rc": rc,
                                "id": buf.value.decode("ascii", "replace")})
                out["cameras"] = ids
            except Exception as exc:
                out["scan_error"] = repr(exc)
        devices, err2 = qhy.list_qhy_cameras()
        out["list_result"] = devices
        out["list_error"] = err2
    except Exception as exc:
        import traceback
        out["exception"] = traceback.format_exc()
    return out


@app.get("/api/cameras")
def api_cameras() -> Dict[str, Any]:
    devices: List[Dict[str, Any]] = []
    hint = ""
    try:
        from core.qhy import list_qhy_cameras
        qhy, err = list_qhy_cameras()
        devices.extend(qhy)                 # QHY 排最前：这才是目镜相机
        if err and "找不到" in err:
            hint = "未检测到 QHY 驱动（要用 QHY 相机需安装 AllInOne 驱动包）"
    except Exception:
        log.exception("QHY 扫描异常")
    for d in list_video_devices():
        devices.append({"value": str(d["index"]), "name": d["name"]})
    return {"devices": devices, "hint": hint}


@app.post("/api/mount/connect")
def api_mount_connect(req: ConnectReq) -> Dict[str, Any]:
    try:
        return session.connect_mount(None if req.sim else (req.port or None))
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/mount/disconnect")
def api_mount_disconnect() -> Dict[str, Any]:
    return session.disconnect_mount()


@app.post("/api/camera/close")
def api_camera_close() -> Dict[str, Any]:
    return session.close_camera()


@app.post("/api/camera/open")
def api_camera_open(req: CameraReq) -> Dict[str, Any]:
    try:
        override = parse_camera_source(req)
        override["preset"] = req.preset
        return session.open_camera(override)
    except Exception as exc:
        raise HTTPException(400, str(exc))


# ================================================================ 流程
@app.post("/api/select")
def api_select(req: SelectReq) -> Dict[str, Any]:
    try:
        return session.select_target(req.box)
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/select/clear")
def api_select_clear() -> Dict[str, Any]:
    session.clear_target()
    return {"ok": True}


@app.post("/api/calibrate")
def api_calibrate() -> Dict[str, Any]:
    try:
        return session.calibrate()
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/track/start")
def api_track_start() -> Dict[str, Any]:
    try:
        return session.start_tracking()
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/track/stop")
def api_track_stop() -> Dict[str, Any]:
    session.stop_tracking()
    return {"ok": True}


@app.post("/api/nudge")
def api_nudge(req: NudgeReq) -> Dict[str, Any]:
    try:
        return session.nudge(req.d_az, req.d_alt)
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/status")
def api_status(request: Request) -> Dict[str, Any]:
    st = session.status()
    st["record"] = recorder.status()
    username = _current_user(request)
    st["user"] = {"username": username, "role": users.role_of(username),
                  "is_default_password": users.uses_default_password(username)}
    return st


@app.on_event("shutdown")
def _shutdown():
    # 先收录像：视频文件没 release 就是个播不了的半截文件，
    # 而跟踪会话那边最坏也只是晚几秒停。
    recorder.shutdown()
    session.shutdown()
