"""QHY 相机驱动（ctypes 直调官方 SDK）。

QHY 相机不是 UVC 设备：插上电脑既没有 COM 口也没有"摄像头"，
Windows 根本不认识它。唯一的取像通道是 QHY 官方 SDK ——
装完 AllInOne 驱动包后系统里会有 qhyccd.dll（mac 是 libqhyccd.dylib），
本模块用 ctypes 直接调它：扫描 → 打开 → 设参 → live 流取帧。

从第一个项目的驱动移植，适配本项目的 CameraBase / Frame 契约：
  * grab() 失败返回 None（采集线程长期跑，坏帧不许抛异常）
  * Frame.image 是 BGR uint8（黑白相机转三通道，下游不用分情况）
  * SDK 缺失时 list_qhy_cameras() 返回空表并给出原因，绝不抛异常 ——
    没装驱动包的电脑上其余功能必须照常可用
"""
from __future__ import annotations

import ctypes
import logging
import os
import platform
import time
from ctypes import POINTER, byref, c_char_p, c_double, c_int, c_uint8, c_uint32, c_void_p
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.camera import CameraBase, Frame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- SDK 常量
CONTROL_GAIN = 6
CONTROL_OFFSET = 7
CONTROL_EXPOSURE = 8          # 单位：微秒
CONTROL_USBTRAFFIC = 12
CONTROL_TRANSFERBIT = 14
STREAM_LIVE = 1
QHYCCD_SUCCESS = 0


def _library_candidates() -> List[str]:
    """按优先级列出可能的 SDK 库路径。

    Windows 上不能只按名字找：AllInOne 装完后 qhyccd.dll 未必进
    System32 —— 用户实测装完后系统目录里根本没有，DLL 散落在
    「C:\\Program Files\\<各种中文/英文目录>\\qhyccd.dll」。
    所以这里补一遍常见安装位置的扫描，用户不用手动拷贝。
    """
    system = platform.system()
    if system == "Darwin":
        return ["/usr/local/lib/libqhyccd.dylib", "libqhyccd.dylib"]
    if system != "Windows":
        return ["/usr/local/lib/libqhyccd.so", "libqhyccd.so.6", "libqhyccd.so"]

    import glob
    found: List[str] = ["qhyccd.dll"]          # 先试系统搜索路径
    roots = [os.environ.get("ProgramFiles", r"C:\Program Files"),
             os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")]
    seen = set()
    for root in roots:
        if not root:
            continue
        # 装机目录名各式各样（QHYCCD / 态势感知 / SharpCap …），
        # 所以搜两层通配而不是写死目录名
        for pattern in ("*/qhyccd.dll", "*/*/qhyccd.dll"):
            for path in glob.glob(os.path.join(root, pattern)):
                key = path.lower()
                if key in seen:
                    continue
                seen.add(key)
                # QHY 自家目录优先于第三方软件自带的副本（版本更匹配）
                found.insert(1 if "qhy" in key else len(found), path)
    return found


_sdk_cache: Optional[Any] = None
_sdk_error: Optional[str] = None


def _load_sdk():
    """加载并初始化 SDK（进程内只做一次）。失败返回 (None, 原因)。"""
    global _sdk_cache, _sdk_error
    if _sdk_cache is not None:
        return _sdk_cache, None
    if _sdk_error is not None:
        return None, _sdk_error

    path = os.environ.get("QHYCCD_LIBRARY")
    candidates = [path] if path else _library_candidates()
    sdk = None
    last_exc: Optional[Exception] = None
    for cand in candidates:
        try:
            sdk = ctypes.CDLL(cand)
            break
        except OSError as exc:
            last_exc = exc
    if sdk is None:
        _sdk_error = ("找不到 QHY 官方驱动（qhyccd.dll）。请先安装 QHY 的 "
                      "AllInOne 驱动包：https://www.qhyccd.com/download/ ，"
                      "装完重启本程序。已尝试路径：%s（最后错误：%s）"
                      % (candidates, last_exc))
        return None, _sdk_error

    try:
        sdk.OpenQHYCCD.restype = c_void_p
        sdk.OpenQHYCCD.argtypes = [c_char_p]
        sdk.CloseQHYCCD.argtypes = [c_void_p]
        sdk.InitQHYCCD.argtypes = [c_void_p]
        sdk.SetQHYCCDStreamMode.argtypes = [c_void_p, c_uint8]
        sdk.SetQHYCCDParam.argtypes = [c_void_p, c_int, c_double]
        sdk.SetQHYCCDResolution.argtypes = [c_void_p, c_uint32, c_uint32,
                                            c_uint32, c_uint32]
        sdk.SetQHYCCDBinMode.argtypes = [c_void_p, c_uint32, c_uint32]
        sdk.GetQHYCCDMemLength.argtypes = [c_void_p]
        sdk.GetQHYCCDMemLength.restype = c_uint32
        sdk.BeginQHYCCDLive.argtypes = [c_void_p]
        sdk.StopQHYCCDLive.argtypes = [c_void_p]
        sdk.GetQHYCCDLiveFrame.argtypes = [
            c_void_p, POINTER(c_uint32), POINTER(c_uint32),
            POINTER(c_uint32), POINTER(c_uint32), POINTER(c_uint8)]
        sdk.GetQHYCCDId.argtypes = [c_uint32, c_char_p]
        if sdk.InitQHYCCDResource() != QHYCCD_SUCCESS:
            _sdk_error = "QHY SDK 初始化失败（InitQHYCCDResource）"
            return None, _sdk_error
    except Exception as exc:
        _sdk_error = "QHY SDK 接口异常：%s" % exc
        return None, _sdk_error

    _sdk_cache = sdk
    return sdk, None


def list_qhy_cameras() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """扫描在线的 QHY 相机。返回 (设备表, 不可用原因)。

    设备表元素形如 {"value": "qhy:0", "name": "QHY5III200M（QHY 专业相机）"}；
    SDK 缺失或没插相机时表为空，原因放在第二个返回值里给界面显示。
    """
    sdk, err = _load_sdk()
    if sdk is None:
        return [], err
    try:
        count = sdk.ScanQHYCCD()
    except Exception as exc:
        return [], "QHY 扫描失败：%s" % exc
    devices: List[Dict[str, Any]] = []
    for i in range(int(count)):
        cam_id = ctypes.create_string_buffer(64)
        if sdk.GetQHYCCDId(c_uint32(i), cam_id) != QHYCCD_SUCCESS:
            continue
        raw = cam_id.value.decode("ascii", errors="replace")
        # id 形如 "QHY5III200M-6077..."，横杠前是型号
        model = raw.split("-")[0] or raw
        devices.append({"value": "qhy:%d" % i,
                        "name": "%s（QHY 专业相机）" % model})
    return devices, None


class QHYCamera(CameraBase):
    """QHY live 流相机。跟踪要的是连续低延迟出图，不是单帧长曝光。"""

    def __init__(self, index: int = 0, exposure_ms: float = 20.0,
                 gain: float = 180.0, offset: float = 10.0,
                 usb_traffic: float = 30.0, binning: int = 1):
        self._index = int(index)
        self._exposure_ms = float(exposure_ms)
        self._gain = float(gain)
        self._offset = float(offset)
        self._usb_traffic = float(usb_traffic)
        self._binning = max(1, int(binning))
        self._sdk = None
        self._handle: Optional[c_void_p] = None
        self._buffer = None
        self._frame_index = 0
        self._model = "QHY"
        self._size: Tuple[int, int] = (0, 0)

    # ------------------------------------------------------------ 生命周期
    def open(self) -> None:
        sdk, err = _load_sdk()
        if sdk is None:
            raise RuntimeError(err)
        count = sdk.ScanQHYCCD()
        if int(count) <= self._index:
            raise RuntimeError(
                "只扫描到 %d 台 QHY 相机，取不到第 %d 台。"
                "检查 USB 线是否插好（建议直插电脑，不要过集线器）"
                % (count, self._index))
        cam_id = ctypes.create_string_buffer(64)
        if sdk.GetQHYCCDId(c_uint32(self._index), cam_id) != QHYCCD_SUCCESS:
            raise RuntimeError("读取 QHY 相机序列号失败")
        self._model = cam_id.value.decode("ascii", "replace").split("-")[0]

        handle = sdk.OpenQHYCCD(cam_id.value)
        if not handle:
            raise RuntimeError("打开 %s 失败：可能被其它软件（SharpCap 等）占用"
                               % self._model)
        self._handle = c_void_p(handle)
        self._sdk = sdk

        # 顺序不能乱：流模式 → Init → 参数 → bin/ROI（Init 会重置部分参数）
        sdk.SetQHYCCDStreamMode(self._handle, c_uint8(STREAM_LIVE))
        if sdk.InitQHYCCD(self._handle) != QHYCCD_SUCCESS:
            self.close()
            raise RuntimeError("初始化 %s 失败（InitQHYCCD）" % self._model)
        sdk.SetQHYCCDParam(self._handle, CONTROL_TRANSFERBIT, c_double(8.0))
        sdk.SetQHYCCDParam(self._handle, CONTROL_USBTRAFFIC,
                           c_double(self._usb_traffic))
        sdk.SetQHYCCDParam(self._handle, CONTROL_OFFSET, c_double(self._offset))
        sdk.SetQHYCCDParam(self._handle, CONTROL_GAIN, c_double(self._gain))
        sdk.SetQHYCCDParam(self._handle, CONTROL_EXPOSURE,
                           c_double(self._exposure_ms * 1000.0))
        sdk.SetQHYCCDBinMode(self._handle, c_uint32(self._binning),
                             c_uint32(self._binning))
        # (0,0,0,0) 之外传满幅会因型号而异，这里交给 SDK 用当前 bin 的默认满幅
        length = sdk.GetQHYCCDMemLength(self._handle)
        if int(length) <= 0:
            self.close()
            raise RuntimeError("读取 %s 缓冲大小失败" % self._model)
        self._buffer = (c_uint8 * int(length))()
        if sdk.BeginQHYCCDLive(self._handle) != QHYCCD_SUCCESS:
            self.close()
            raise RuntimeError("启动 %s 视频流失败" % self._model)
        time.sleep(0.3)      # 等首帧管线填满

    def close(self) -> None:
        sdk, handle = self._sdk, self._handle
        self._sdk = None
        self._handle = None
        self._buffer = None
        if sdk is not None and handle is not None:
            try:
                sdk.StopQHYCCDLive(handle)
            except Exception:
                pass
            try:
                sdk.CloseQHYCCD(handle)
            except Exception:
                pass
        # 不调 ReleaseQHYCCDResource：SDK 全局资源进程内复用，
        # 释放后再次 Init 在部分版本上会崩

    @property
    def opened(self) -> bool:
        return self._handle is not None

    # ------------------------------------------------------------ 取帧
    def grab(self) -> Optional[Frame]:
        sdk, handle, buf = self._sdk, self._handle, self._buffer
        if sdk is None or handle is None or buf is None:
            return None
        w, h, bpp, ch = c_uint32(), c_uint32(), c_uint32(), c_uint32()
        try:
            rc = sdk.GetQHYCCDLiveFrame(handle, byref(w), byref(h),
                                        byref(bpp), byref(ch), buf)
        except Exception:
            return None
        if rc != QHYCCD_SUCCESS:
            return None          # live 模式帧没就绪会返回非 0，属正常
        width, height = int(w.value), int(h.value)
        channels = max(int(ch.value), 1)
        if width <= 0 or height <= 0:
            return None
        count = width * height * channels
        raw = np.frombuffer(buf, dtype=np.uint8, count=count)
        img = (raw.reshape((height, width, channels)) if channels > 1
               else raw.reshape((height, width)))
        if channels == 1:
            # 契约要求 BGR：黑白（200M）转三通道，下游不用分情况
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self._size = (width, height)
        self._frame_index += 1
        return Frame(image=np.ascontiguousarray(img), ts=time.time(),
                     index=self._frame_index)

    def info(self) -> Dict[str, Any]:
        return {"driver": "qhy", "model": self._model,
                "width": self._size[0], "height": self._size[1],
                "exposure_ms": self._exposure_ms, "gain": self._gain,
                "binning": self._binning}
