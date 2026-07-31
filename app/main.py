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
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from core.control import TrackingSession
from core.camera import list_video_devices

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


session = TrackingSession(load_cfg())
app = FastAPI(title="LCM90 视觉跟踪")


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


# ================================================================ 页面与视频
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


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
def api_status() -> Dict[str, Any]:
    return session.status()


@app.on_event("shutdown")
def _shutdown():
    session.shutdown()
