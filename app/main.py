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


import atexit
import threading

#: 置位后所有 /video 流立即收尾。关程序时必须先放掉它们，
#: 否则 uvicorn 的优雅关闭会一直等这些永不结束的响应。
_stream_stop = threading.Event()
_shutdown_done = threading.Event()

_auth_cfg = dict(_CFG.get("auth", {}) or {})
users = UserStore(_under_root(str(_auth_cfg.get("file", "data/users.json"))))
sessions = SessionStore()
SESSION_COOKIE = "lcm90_session"

# ---- 云端同步（离线优先：没配置或连不上时，一切照常本地运行）----
from core.cloud import (CloudAccounts, CloudConfig, CloudError, S3Client,
                        UploadQueue)

CLOUD_FILE = _under_root("data/cloud.json")
cloud_cfg = CloudConfig.load(CLOUD_FILE)


def _cloud_client() -> Optional[S3Client]:
    """按当前配置造一个客户端；没启用/没配好就返回 None（不是报错）。
    上传线程靠这个 None 判断「现在还不能传」，然后安静地等下一轮。"""
    if not cloud_cfg.enabled:
        return None
    try:
        return S3Client(cloud_cfg)
    except CloudError:
        return None


def _device_tag() -> str:
    import socket
    return (cloud_cfg.device_name or socket.gethostname() or "device").strip()


def _cloud_key(kind: str, filename: str) -> str:
    return "%s/%s/%s/%s" % (cloud_cfg.prefix.strip("/"), _device_tag(),
                            kind, filename)


uploads = UploadQueue(_under_root("data/upload-queue.json"), _cloud_client)
cloud_accounts = CloudAccounts(
    _cloud_client, lambda: "%s/_shared/users.json" % cloud_cfg.prefix.strip("/"))

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
def api_reset_password(username: str, req: ResetPwdReq, request: Request,
                       response: Response) -> Dict[str, Any]:
    me = _require_admin(request)
    try:
        users.reset_password(username, req.password)
    except AuthError as exc:
        raise HTTPException(400, str(exc))
    sessions.drop_user(username)     # 被重置的人必须重新登录
    if username == me:
        # 管理员重置的是自己：drop_user 刚把自己的票也作废了，
        # 不补发新票的话，下一拍 poll 就会拿 401 把他弹回登录页 ——
        # 看起来就像「改个密码怎么被踢出去了」。
        response.set_cookie(SESSION_COOKIE, sessions.create(me),
                            httponly=True, samesite="lax", secure=False)
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


# ================================================================ 星点检测
@app.get("/api/sources")
def api_sources() -> Dict[str, Any]:
    """找出画面里的星点，坐标**归一化**返回（前端不用管分辨率）。"""
    frame = session.latest_frame()
    if frame is None:
        return {"ok": False, "message": "现在没有画面。", "sources": []}

    from core.stars import best_calibration_star, detect_sources, focus_metric

    h, w = frame.shape[:2]
    found = detect_sources(frame)
    # 标定一步大约把目标推动画面短边的 12%（见 calibration._STEP_FRAC_OF_FRAME），
    # 选星时要按这个尺度判断「离边缘够不够远」
    step_px = 0.12 * min(h, w)
    best = best_calibration_star(found, (h, w), step_px=step_px)

    def _pack(s, is_best: bool) -> Dict[str, Any]:
        return {"x": s.x / w, "y": s.y / h,
                "r": max(6.0, s.fwhm * 2.0) / w,   # 画圈半径，按短边归一化
                "flux": round(s.flux, 1), "peak": round(s.peak, 1),
                "fwhm": round(s.fwhm, 2), "saturated": s.saturated,
                "score": round(s.score, 3), "best": is_best}

    return {"ok": True, "count": len(found),
            # 画面宽高给前端做 SVG 的 viewBox —— 用正方形 viewBox 配
            # preserveAspectRatio="none" 会把圆圈拉成椭圆
            "w": w, "h": h,
            "sources": [_pack(s, best is not None and s is best) for s in found],
            "focus_fwhm": focus_metric(found),
            "best_hint": None if best else
                         "这一帧里没有适合标定的星（可能都过曝、挤在一起或太靠边）。"}


class StarSelectReq(BaseModel):
    x: float          # 归一化 0..1
    y: float
    r: float = 0.0    # 归一化半径，0 表示让后端按默认框大小取


@app.post("/api/select/star")
def api_select_star(req: StarSelectReq) -> Dict[str, Any]:
    """点选一个星点直接建跟踪框，免去手动拖框。"""
    frame = session.latest_frame()
    if frame is None:
        raise HTTPException(400, "现在没有画面，先开启视频源。")
    h, w = frame.shape[:2]
    # 框要比星点本身大一圈：CSRT 需要一点背景才能建立模板，
    # 框得死贴星点反而锁不住。
    half = max(14.0, float(req.r) * w * 2.5)
    bw, bh = 2 * half / w, 2 * half / h
    x = min(max(req.x - bw / 2, 0.0), 1.0 - bw)
    y = min(max(req.y - bh / 2, 0.0), 1.0 - bh)
    return session.select_target((x, y, bw, bh))


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
    msg = "录像已保存：%s（%.1f MB）" % (st["file"], st["size_mb"])
    # 录完就排队上传。**只是入队**，真正的上传在后台慢慢做，
    # 断网也无所谓 —— 队列落盘，回到有网的地方会自己补传。
    if cloud_cfg.enabled and cloud_cfg.upload_recordings and st.get("path"):
        if uploads.enqueue(st["path"], _cloud_key("recordings", st["file"])):
            msg += "，已排队上传到云端"
    return {"ok": True, "message": msg, "record": st}


# ================================================================ 云端
class CloudCfgReq(BaseModel):
    enabled: Optional[bool] = None
    endpoint: Optional[str] = None
    bucket: Optional[str] = None
    region: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    prefix: Optional[str] = None
    device_name: Optional[str] = None
    upload_recordings: Optional[bool] = None
    upload_calibration: Optional[bool] = None
    sync_accounts: Optional[bool] = None


@app.get("/api/cloud")
def api_cloud_get(request: Request) -> Dict[str, Any]:
    _require_admin(request)
    return {"config": cloud_cfg.masked(), "queue": uploads.status()}


@app.post("/api/cloud")
def api_cloud_set(req: CloudCfgReq, request: Request) -> Dict[str, Any]:
    _require_admin(request)
    for name, val in req.dict(exclude_none=True).items():
        # 密钥留空表示「不改」——界面上显示的是打码版，原样提交回来时
        # 不能把真密钥覆盖成一串星号。
        if name in ("access_key", "secret_key") and (not val or "*" in val):
            continue
        setattr(cloud_cfg, name, val)
    try:
        cloud_cfg.save(CLOUD_FILE)
    except Exception as exc:
        raise HTTPException(400, "保存云端配置失败：%s" % exc)
    if cloud_cfg.enabled:
        uploads.start()
    return {"ok": True, "message": "云端配置已保存。", "config": cloud_cfg.masked()}


@app.post("/api/cloud/test")
def api_cloud_test(request: Request) -> Dict[str, Any]:
    _require_admin(request)
    client = _cloud_client()
    if client is None:
        raise HTTPException(400, "云端未启用或配置不完整。")
    try:
        return {"ok": True, "message": client.ping()}
    except CloudError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/cloud/sync")
def api_cloud_sync(request: Request) -> Dict[str, Any]:
    """立即同步：把待传的重试一遍，并双向同步账号。"""
    _require_admin(request)
    uploads.retry_now()
    notes = []
    if cloud_cfg.sync_accounts:
        try:
            notes.append(_sync_accounts_once())
        except CloudError as exc:
            notes.append("账号同步失败：%s" % exc)
    st = uploads.status()
    msg = "已开始同步：待传 %d 个文件" % st["pending"]
    if notes:
        msg += "。" + "；".join(notes)
    return {"ok": True, "message": msg, "queue": st}


def _sync_accounts_once() -> str:
    """把云端账号拉下来合并，再把合并结果推回去。"""
    remote = cloud_accounts.pull()
    local = users.export_for_sync()
    if remote is None:
        cloud_accounts.push(local)
        return "云端还没有账号文件，已用本地的初始化"
    merged, changed = CloudAccounts.merge(local, remote)
    if changed:
        try:
            users.import_merged(merged)
        except AuthError as exc:
            return "云端账号未应用：%s" % exc
        # 密码可能在别的设备上被改过，本机的旧会话必须作废
        for name in list((merged.get("users") or {})):
            sessions.drop_user(name)
    cloud_accounts.push(users.export_for_sync())
    return "账号已同步" + ("（本地有更新）" if changed else "")


@app.get("/video")
def video():
    def gen():
        # 没画面时也必须周期性 yield 一个空片段。
        # 原来的写法在 frame 为 None 时**永不 yield**，于是：
        #   * 客户端断开探测不到（探测发生在 yield 之后的写操作上），
        #     这条流会一直占着一个同步线程池名额；
        #   * 关程序时 uvicorn 的优雅关闭要等所有响应结束，
        #     这个 while True 永远不结束 → shutdown 事件一句都不跑
        #     → 停基座、收录像全部被跳过，望远镜关窗后还在转。
        idle = 0
        while not _stream_stop.is_set():
            frame = session.annotated_frame()
            if frame is not None:
                idle = 0
                ok, buf = cv2.imencode(".jpg", frame,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                if ok:
                    data = buf.tobytes()
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(data)).encode()
                           + b"\r\n\r\n" + data + b"\r\n")
            else:
                idle += 1
                # 空注释块：合法的 multipart 片段，只为把「客户端还在不在」
                # 这件事问出来。多帧一次即可，别刷屏。
                if idle % 18 == 0:
                    yield b"--frame\r\nContent-Type: text/plain\r\n\r\n\r\n"
                # 相机关掉后没必要再吊着这条连接，前端会自己重新挂
                if idle > 18 * 10:
                    break
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
    st["cloud"] = {"enabled": bool(cloud_cfg.enabled), **uploads.status()}
    username = _current_user(request)
    st["user"] = {"username": username, "role": users.role_of(username),
                  "is_default_password": users.uses_default_password(username)}
    return st


def shutdown_now() -> None:
    """停掉一切。**幂等**，可以被 lifespan 和 serve.py 各调一次。

    为什么不能只挂在 FastAPI 的 shutdown 事件上：那条路要等所有响应结束，
    而 /video 是长连接。一旦它没能结束，shutdown 就永远不执行 ——
    「停基座」是本程序唯一的停电机路径，绝不能吊在一条可能走不完的链上。
    """
    if _shutdown_done.is_set():
        return
    _shutdown_done.set()
    _stream_stop.set()          # 先放掉视频流，否则谁都等不到它
    try:
        uploads.shutdown()      # 停上传线程；没传完的留在队列里，下次接着传
    except Exception:
        log.exception("关闭上传队列失败")
    try:
        recorder.shutdown()     # 先收录像：没 release 就是个播不了的半截文件
    except Exception:
        log.exception("关闭录像失败")
    try:
        session.shutdown()      # 里面会把基座速率清零并断开
    except Exception:
        log.exception("关闭跟踪会话失败")


@app.on_event("startup")
def _startup():
    if not cloud_cfg.enabled:
        return
    uploads.start()
    if cloud_cfg.sync_accounts:
        # 放后台线程：云端连不上时，这一步会卡住整个程序的启动
        def _first_sync():
            try:
                log.info("首次账号同步：%s", _sync_accounts_once())
            except Exception:
                log.warning("首次账号同步失败，先用本地账号", exc_info=True)
        threading.Thread(target=_first_sync, name="cloud-first-sync",
                         daemon=True).start()


@app.on_event("shutdown")
def _shutdown():
    shutdown_now()


# 最后一道保险：进程无论以什么方式退出，望远镜都必须停下来。
# 全仓库以前没有任何 atexit / 信号处理，停电机只有 lifespan 一条路。
atexit.register(shutdown_now)
