"""闭环控制：相机 + 跟踪器 + 标定 + 基座 + PID 串成一个状态机。

线程模型（为什么分两个线程）：
  - 采集线程：取图 + 跟踪。视觉耗时不定（CSRT 一帧可能 20~60ms），
    绝不能让它阻塞控制回路的定时，否则速率下发抖动会直接打到基座上。
  - 控制线程：定频（loop_hz）读取最新跟踪结果、算 PID、下发速率。
    两者通过带锁的「最新帧/最新结果」交接，控制线程永远拿到最新数据，
    宁可丢帧也不排队 —— 排队意味着用旧数据控制，相位滞后会引起振荡。

安全铁律：目标丢失（lost）时**立即把基座速率清零**。
带着最后的速率继续跑，基座会一路转到限位，视场早就没有目标了。
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# 直接以脚本方式运行（python core/control.py）时 sys.path[0] 是 core/，
# 包内导入会失败；把项目根目录补进去，两种运行方式都能工作。
try:
    from core.calibration import Calibration, Calibrator
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.calibration import Calibration, Calibrator

logger = logging.getLogger(__name__)

# 画面文字：OpenCV 的 putText 不支持中文，优先用 Pillow 渲染；
# 没装 Pillow 时退化为英文并提示一次怎么装，绝不为此崩溃。
try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

_STATE_ZH = {
    "idle": "空闲", "selecting": "已选目标", "calibrating": "标定中",
    "tracking": "跟踪中", "lost": "目标丢失", "error": "错误",
}
_STATE_EN = {
    "idle": "idle", "selecting": "selected", "calibrating": "calibrating",
    "tracking": "tracking", "lost": "LOST", "error": "ERROR",
}


class PID:
    """单轴 PID（误差单位=度，输出=度/秒）。四个工程细节缺一不可：

    1. 积分分离：|err|>i_band 完全不积分。捕获阶段误差好几度，
       照常积分几个周期就顶到限幅，之后必然超调且十几秒卸不掉。
    2. 条件积分抗饱和：输出撞限后撤销本次积分，不让它在饱和区增长，
       否则目标反向时基座要先「卸积分」才肯掉头。
    3. 前馈参与饱和判断：前馈是输出的一部分；若只对 PID 分量判饱和，
       抗饱和逻辑形同虚设。
    4. 微分低通：质心有 ±0.5px 抖动，裸微分会放大成高频噪声打到基座上。
    """

    def __init__(self, kp, ki, kd, i_limit=0.5, i_band=0.5, out_limit=2.5):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.i_limit = float(i_limit)
        self.i_band = float(i_band)
        self.out_limit = float(out_limit)
        self._d_lowpass = 0.3   # 微分低通系数：越小越平滑，经验值
        self._integral = 0.0
        self._prev_error = None  # type: Optional[float]
        self._d_state = 0.0
        self._saturated = False

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None
        self._d_state = 0.0
        self._saturated = False

    def _clamp_integral(self, integral: float) -> float:
        if self.ki == 0:
            return 0.0
        bound = self.i_limit / abs(self.ki)
        return max(-bound, min(bound, integral))

    def update(self, err: float, dt: float, feedforward: float = 0.0) -> float:
        """积分分离（|err|>i_band 不积分）+ 条件积分抗饱和 + 前馈。
        前馈必须参与饱和判断，否则会积分饱和。"""
        if dt <= 0:
            return 0.0

        p = self.kp * err

        # 微分先算：撤销积分时它不受影响
        d = 0.0
        if self._prev_error is not None:
            d_raw = (err - self._prev_error) / dt
            self._d_state += self._d_lowpass * (d_raw - self._d_state)
            d = self.kd * self._d_state
        self._prev_error = err

        prev_integral = self._integral
        if abs(err) <= self.i_band:
            self._integral = self._clamp_integral(self._integral + err * dt)
        else:
            # 大误差期间顺带把残留积分收敛回 0，捕获结束不带历史包袱
            self._integral *= 0.95

        out = feedforward + p + self.ki * self._integral + d

        if abs(out) > self.out_limit:
            # 撞限：撤销本次积分，不让它在饱和区继续增长
            self._integral = prev_integral
            out = feedforward + p + self.ki * self._integral + d
            out = max(-self.out_limit, min(self.out_limit, out))
            self._saturated = True
        else:
            self._saturated = False

        return out

    @property
    def saturated(self) -> bool:
        """上一次 update 是否撞上输出上限。UI 用它提示「基座跟不动了」。"""
        return self._saturated


@dataclass
class Telemetry:
    state: str            # idle/selecting/calibrating/tracking/lost/error
    message: str
    az: float
    alt: float
    rate_az: float
    rate_alt: float
    err_px: Optional[float]
    err_arcsec: Optional[float]
    locked: bool
    score: float
    fps: float
    loop_hz: float
    lost_frames: int
    calibrated: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrackingSession:
    """把相机 + 跟踪器 + 标定 + 基座 + PID 串成一个状态机。

    状态迁移（与 CONTRACT.md 一致）：
        idle → selecting → calibrating → selecting → tracking ⇄ lost
    未标定拒绝 start_tracking：符号/转角未知时跟踪会往反方向修，
    目标越修越远，这是最危险的一种「看起来在工作」的故障。
    """

    def __init__(self, cfg: Dict[str, Any]):
        self._cfg = dict(cfg or {})
        ctrl = self._cfg.get("control", {}) or {}
        self._loop_hz = float(ctrl.get("loop_hz", 15))
        self._lock_radius_px = float(ctrl.get("lock_radius_px", 30))
        self._deadband_px = float(ctrl.get("deadband_px", 2))
        self._pid_cfg = dict(ctrl.get("pid", {}) or {})

        self._lock = threading.RLock()          # 状态与设备句柄
        self._frame_lock = threading.Lock()     # 最新帧/最新跟踪结果

        self._state = "idle"
        self._message = "空闲。请先打开相机并框选目标。"

        self._mount = None
        self._camera = None
        self._tracker = None
        # 测试注入口：设为可调用后 select_target 用它创建跟踪器，
        # 否则懒加载 core.tracker（各模块并行开发，导入放到用时再做）。
        self._tracker_factory = None

        self._calibration = None  # type: Optional[Calibration]
        self._calib_progress = 0.0
        self._calib_message = ""
        self._calib_thread = None  # type: Optional[threading.Thread]

        self._last_frame = None    # type: Optional[np.ndarray]
        self._last_result = None
        self._last_result_ts = None
        self._fps = 0.0
        self._loop_hz_meas = 0.0
        self._rate_az = 0.0
        self._rate_alt = 0.0
        self._ff_az = 0.0
        self._ff_alt = 0.0

        # 仰角缓存：真实基座查一次坐标要走串口（几十 ms），
        # 15Hz 控制回路里每拍都查会把回路拖垮，1 秒刷新一次足够
        # （cos(alt) 在 1 秒内的变化对速率的影响可忽略）。
        self._alt_cache = 45.0
        self._alt_cache_ts = 0.0

        self._pause_capture = False   # 标定期间由标定器独占相机取帧
        self._warned_no_pil = False

        max_rate = float((self._cfg.get("mount", {}) or {}).get("max_rate_deg_s", 2.5))
        self._pid_az = self._build_pid(max_rate)
        self._pid_alt = self._build_pid(max_rate)

        self._stop_evt = threading.Event()
        self._cap_thread = threading.Thread(target=self._capture_loop,
                                            name="capture", daemon=True)
        self._ctl_thread = threading.Thread(target=self._control_loop,
                                            name="control", daemon=True)
        self._cap_thread.start()
        self._ctl_thread.start()

        self._try_load_calibration()

    # ---------------- 内部工具 ----------------

    def _build_pid(self, out_limit: float) -> PID:
        c = self._pid_cfg
        return PID(kp=float(c.get("kp", 0.9)), ki=float(c.get("ki", 0.06)),
                   kd=float(c.get("kd", 0.05)),
                   i_limit=float(c.get("i_limit", 0.5)),
                   i_band=float(c.get("i_band", 0.4)),
                   out_limit=out_limit)

    def _calib_file(self) -> str:
        path = str((self._cfg.get("calibration", {}) or {}).get(
            "file", "data/calibration.json") or "")
        if path and not os.path.isabs(path):
            # 相对路径按「程序所在目录」解释，避免依赖启动时的 cwd。
            # 打包成 exe 后 __file__ 指向 PyInstaller 的临时解压目录，
            # 每次运行都不同且会被删掉 —— 标定存进去等于每次重启都丢，
            # 所以 frozen 时必须落在 exe 旁边。
            if getattr(sys, "frozen", False):
                root = os.path.dirname(os.path.abspath(sys.executable))
            else:
                root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(root, path)
        return path

    def _try_load_calibration(self) -> None:
        """启动时恢复上次标定。装机后相机若被拆装过应重新标定，
        所以恢复时明确提醒，而不是默默沿用。"""
        path = self._calib_file()
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                cal = Calibration.from_dict(json.load(fh))
            if cal.ok:
                with self._lock:
                    self._calibration = cal
                logger.info("已从 %s 恢复标定（%.2f\"/px）。"
                            "若相机拆装或旋转过，请重新标定。",
                            path, cal.arcsec_per_px)
        except Exception:
            logger.exception("读取标定文件失败，忽略：%s", path)

    def _save_calibration(self, cal: Calibration) -> None:
        path = self._calib_file()
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(cal.to_dict(), fh, ensure_ascii=False, indent=2)
        except Exception:
            # 存档失败不影响本次会话使用标定，但要留下线索
            logger.exception("保存标定文件失败：%s", path)

    def _get_alt(self) -> float:
        now = time.monotonic()
        mount = self._mount
        if mount is not None and (now - self._alt_cache_ts) > 1.0:
            try:
                self._alt_cache = float(mount.get_altaz()[1])
                self._alt_cache_ts = now
            except Exception:
                # 读失败沿用缓存：cos(alt) 短时间不变，比中断控制强
                logger.exception("读取基座仰角失败，沿用缓存值")
                self._alt_cache_ts = now
        return self._alt_cache

    def _zero_rates(self, reset_pid: bool = True) -> None:
        """速率清零是安全动作，必须尽力送达；失败要记日志并转 error。"""
        self._rate_az = 0.0
        self._rate_alt = 0.0
        self._ff_az = 0.0
        self._ff_alt = 0.0
        if reset_pid:
            self._pid_az.reset()
            self._pid_alt.reset()
        mount = self._mount
        if mount is None:
            return
        try:
            mount.set_rate(0.0, 0.0)
        except Exception:
            logger.exception("下发零速率失败——基座可能仍在移动，请手动急停")

    # ---------------- 设备 ----------------

    def connect_mount(self, port: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if self._mount is not None and getattr(self._mount, "connected", False):
                return {"ok": True, "message": "基座已经连接。",
                        "info": self._safe_mount_info()}
        mcfg = dict(self._cfg.get("mount", {}) or {})
        if port:
            mcfg["port"] = port
        try:
            from core.mount import build_mount  # 懒加载：模块并行开发
            mount = build_mount(mcfg)
            mount.connect()
        except Exception as exc:
            logger.exception("连接基座失败")
            return {"ok": False,
                    "message": "基座连接失败：%s。请检查串口线是否插好、"
                               "端口号是否正确、手控器是否已开机。" % exc}
        with self._lock:
            self._mount = mount
            # 输出限幅跟随真实基座的机械上限，而不是配置里的猜测值
            limit = float(getattr(mount, "max_rate_deg_s", 2.5))
            self._pid_az = self._build_pid(limit)
            self._pid_alt = self._build_pid(limit)
        return {"ok": True, "message": "基座已连接。",
                "info": self._safe_mount_info()}

    def _safe_mount_info(self) -> Dict[str, Any]:
        mount = self._mount
        if mount is None:
            return {}
        try:
            return dict(mount.info())
        except Exception:
            logger.exception("读取基座信息失败")
            return {}

    def disconnect_mount(self) -> None:
        with self._lock:
            if self._state in ("tracking", "lost"):
                self.stop_tracking()
            mount = self._mount
            self._mount = None
        if mount is None:
            return
        try:
            mount.stop()
        except Exception:
            logger.exception("断开前停转基座失败")
        try:
            mount.close()
        except Exception:
            logger.exception("关闭基座连接失败")

    def open_camera(self, source: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ccfg = dict(self._cfg.get("camera", {}) or {})
        if source:
            ccfg.update(source)
        get_pointing = None
        mount = self._mount
        if mount is not None:
            # 仿真相机用它形成真闭环：画面随基座指向变化
            get_pointing = mount.get_altaz
        try:
            from core.camera import build_camera  # 懒加载：模块并行开发
            cam = build_camera(ccfg, get_pointing=get_pointing)
            cam.open()
        except Exception as exc:
            logger.exception("打开相机失败")
            return {"ok": False,
                    "message": "相机打开失败：%s。请检查设备号/连接线，"
                               "或先用设备列表确认相机存在。" % exc}
        with self._lock:
            old = self._camera
            self._camera = cam
        if old is not None:
            try:
                old.close()
            except Exception:
                logger.exception("关闭旧相机失败")
        try:
            info = dict(cam.info())
        except Exception:
            info = {}
        return {"ok": True, "message": "相机已打开。", "info": info}

    def close_camera(self) -> None:
        with self._lock:
            if self._state in ("tracking", "lost"):
                self.stop_tracking()
            # 没有画面就没有目标：一并清掉选择状态，避免悬空的 tracker
            self._tracker = None
            if self._state != "error":
                self._state = "idle"
                self._message = "相机已关闭。"
            cam = self._camera
            self._camera = None
        with self._frame_lock:
            self._last_frame = None
            self._last_result = None
        if cam is None:
            return
        try:
            cam.close()
        except Exception:
            logger.exception("关闭相机失败")

    # ---------------- 流程 ----------------

    def _make_tracker(self):
        if self._tracker_factory is not None:
            return self._tracker_factory()
        from core.tracker import TargetTracker  # 懒加载：模块并行开发
        tcfg = self._cfg.get("tracker", {}) or {}
        return TargetTracker(algo=str(tcfg.get("algo", "csrt")),
                             max_lost=int(tcfg.get("max_lost", 25)),
                             min_score=float(tcfg.get("min_score", 0.30)),
                             search_scale=float(tcfg.get("search_scale", 2.5)))

    def select_target(self, box: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """box 为**归一化**坐标 (x,y,w,h)，取值 0..1 —— 前端不用关心分辨率。"""
        with self._lock:
            if self._state == "calibrating":
                return {"ok": False, "message": "标定进行中，请等标定结束后再框选目标。"}
            if self._state in ("tracking", "lost"):
                # 换目标前先停车：带着旧速率去框新目标是事故源
                self.stop_tracking()

        try:
            x, y, w, h = (float(v) for v in box)
        except Exception:
            return {"ok": False, "message": "框选参数格式不对，应为归一化 (x, y, w, h)。"}
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) or w <= 0 or h <= 0 \
                or x + w > 1.0001 or y + h > 1.0001:
            return {"ok": False, "message": "框选超出画面范围，请重新框选（坐标取 0..1）。"}

        # 等采集线程先出一帧：刚打开相机就框选的情况很常见
        frame = None
        deadline = time.monotonic() + 1.0
        while frame is None and time.monotonic() < deadline:
            with self._frame_lock:
                frame = None if self._last_frame is None else self._last_frame.copy()
            if frame is None:
                time.sleep(0.05)
        if frame is None:
            return {"ok": False, "message": "相机还没有画面，请先打开相机并确认有图像。"}

        fh, fw = frame.shape[:2]
        px_box = (x * fw, y * fh, w * fw, h * fh)

        try:
            tracker = self._make_tracker()
            started = bool(tracker.start(frame, px_box))
        except Exception as exc:
            logger.exception("初始化跟踪器失败")
            return {"ok": False, "message": "跟踪器初始化失败：%s。" % exc}
        if not started:
            return {"ok": False,
                    "message": "无法在该区域建立跟踪，请框选对比度更高、"
                               "边界更清晰的目标。"}

        with self._lock:
            self._tracker = tracker
            self._state = "selecting"
            if self._calibration is not None and self._calibration.ok:
                self._message = "目标已选中（已有标定，可直接开始跟踪）。"
            else:
                self._message = "目标已选中，尚未标定。请先标定再开始跟踪。"
        with self._frame_lock:
            self._last_result = None
        return {"ok": True, "message": self._message, "box_px": px_box}

    def clear_target(self) -> None:
        with self._lock:
            if self._state in ("tracking", "lost"):
                self.stop_tracking()
            tracker = self._tracker
            self._tracker = None
            if self._state != "error":
                self._state = "idle"
                self._message = "已清除目标。"
        with self._frame_lock:
            self._last_result = None
        if tracker is not None:
            try:
                tracker.reset()
            except Exception:
                logger.exception("重置跟踪器失败")

    def calibrate(self) -> Dict[str, Any]:
        """异步启动标定，进度看 status() 的 calib_progress / calib_message。"""
        with self._lock:
            if self._state == "calibrating":
                return {"ok": False, "message": "标定已在进行中。"}
            if self._state in ("tracking", "lost"):
                return {"ok": False, "message": "请先停止跟踪，再进行标定。"}
            mount = self._mount
            camera = self._camera
            tracker = self._tracker
            if mount is None or not getattr(mount, "connected", False):
                return {"ok": False, "message": "基座未连接，无法标定。请先连接基座。"}
            if camera is None or not getattr(camera, "opened", False):
                return {"ok": False, "message": "相机未打开，无法标定。请先打开相机。"}
            if tracker is None or not getattr(tracker, "active", False):
                return {"ok": False, "message": "尚未框选目标，无法标定。请先框选目标。"}
            self._state = "calibrating"
            self._message = "标定中……"
            self._calib_progress = 0.0
            self._calib_message = "准备标定"

        ccfg = self._cfg.get("calibration", {}) or {}
        calibrator = Calibrator(mount, camera, tracker,
                                step_deg=float(ccfg.get("step_deg", 0.30)),
                                settle_s=float(ccfg.get("settle_s", 1.2)),
                                samples=int(ccfg.get("samples", 3)))

        def _worker():
            # 标定器要自己 grab 帧测位移，采集线程同时 grab 会互相抢帧，
            # 导致两边都拿到断续画面 —— 标定期间让采集线程暂停。
            self._pause_capture = True
            # 置标志后必须等采集线程**真正退出本轮**：它可能已经过了标志检查、
            # 正卡在 cam.grab() / tracker.update() 里（native 调用会释放 GIL，
            # VideoCapture.read 和 CSRT 都不是线程安全的）。等一个取帧周期
            # 加余量，让在途的那次调用先落地。
            time.sleep(0.25)
            try:
                result = calibrator.run(progress_cb=self._on_calib_progress)
            finally:
                self._pause_capture = False
            with self._lock:
                if result.ok:
                    self._calibration = result
                    self._calib_progress = 1.0
                    self._message = ("标定完成：%.2f 角秒/像素，相机转角 %.1f°%s。"
                                     % (result.arcsec_per_px, result.angle_deg,
                                        "，画面镜像" if result.flipped else ""))
                    self._calib_message = self._message
                else:
                    self._calib_message = "标定失败：%s" % result.note
                    self._message = self._calib_message
                if self._state == "calibrating":
                    # 无论成败都回到 selecting：目标还在（丢了用户会重框）
                    self._state = "selecting"
            if result.ok:
                self._save_calibration(result)

        th = threading.Thread(target=_worker, name="calibrate", daemon=True)
        with self._lock:
            self._calib_thread = th
        th.start()
        return {"ok": True, "message": "标定已开始，进度请查询 status()。"}

    def _on_calib_progress(self, msg: str, frac: float) -> None:
        with self._lock:
            self._calib_message = msg
            self._calib_progress = float(frac)

    def start_tracking(self) -> Dict[str, Any]:
        with self._lock:
            if self._state == "calibrating":
                return {"ok": False, "message": "标定进行中，结束后才能开始跟踪。"}
            if self._state == "tracking":
                return {"ok": True, "message": "已经在跟踪中。"}
            mount = self._mount
            if mount is None or not getattr(mount, "connected", False):
                return {"ok": False, "message": "基座未连接，无法跟踪。请先连接基座。"}
            if self._camera is None or not getattr(self._camera, "opened", False):
                return {"ok": False, "message": "相机未打开，无法跟踪。请先打开相机。"}
            tracker = self._tracker
            if tracker is None or not getattr(tracker, "active", False):
                return {"ok": False, "message": "尚未框选目标。请先在画面中框选要跟踪的目标。"}
            cal = self._calibration
            if cal is None or not cal.ok:
                # 这是最重要的一道闸：未标定时符号/转角未知，
                # 闭环很可能往反方向修，目标越修越远还看似「在工作」。
                return {"ok": False,
                        "message": "尚未标定，拒绝开始跟踪：像素到角度的方向未知，"
                                   "可能把目标越修越远。请先点击「标定」（约半分钟）。"}
            self._pid_az.reset()
            self._pid_alt.reset()
            self._state = "tracking"
            self._message = "跟踪中。"
        return {"ok": True, "message": "开始跟踪。"}

    def stop_tracking(self) -> None:
        with self._lock:
            if self._state not in ("tracking", "lost"):
                return
            tracker = self._tracker
            still_selected = tracker is not None and getattr(tracker, "active", False)
            self._state = "selecting" if still_selected else "idle"
            self._message = "已停止跟踪。" if still_selected else "已停止跟踪，目标已失效。"
        self._zero_rates()

    def nudge(self, d_az: float, d_alt: float) -> Dict[str, Any]:
        """手动微调。跟踪/标定期间拒绝：闭环和手动指令会互相打架。"""
        with self._lock:
            if self._state in ("tracking", "calibrating"):
                return {"ok": False, "message": "跟踪/标定进行中，不能手动微调。"}
            mount = self._mount
        if mount is None or not getattr(mount, "connected", False):
            return {"ok": False, "message": "基座未连接，无法微调。"}

        def _worker():
            # nudge 阻塞到停稳（最长几十秒），不能占着 API 调用线程
            try:
                ok = mount.nudge(float(d_az), float(d_alt))
                if not ok:
                    logger.warning("手动微调失败（可能超时或触及限位）")
            except Exception:
                logger.exception("手动微调出错")

        threading.Thread(target=_worker, name="nudge", daemon=True).start()
        return {"ok": True, "message": "微调已开始（az %+.3f°, alt %+.3f°）。"
                                       % (float(d_az), float(d_alt))}

    def shutdown(self) -> None:
        logger.info("正在关闭跟踪会话")
        self._stop_evt.set()
        for th in (self._cap_thread, self._ctl_thread):
            try:
                th.join(timeout=2.0)
            except Exception:
                pass
        # 先停基座再关设备：安全动作永远第一优先
        self._zero_rates(reset_pid=False)
        self.disconnect_mount()
        self.close_camera()

    # ---------------- 线程 ----------------

    def _capture_loop(self) -> None:
        last_ts = None
        while not self._stop_evt.is_set():
            cam = self._camera
            if cam is None or not getattr(cam, "opened", False) or self._pause_capture:
                time.sleep(0.05)
                continue
            try:
                frame = cam.grab()
            except Exception:
                logger.exception("相机取帧失败")
                time.sleep(0.2)
                continue
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.monotonic()
            if last_ts is not None and now > last_ts:
                inst = 1.0 / (now - last_ts)
                # 指数平滑：fps 是给人看的，抖动的数字没有意义
                self._fps += 0.1 * (inst - self._fps)
            last_ts = now

            result = None
            tracker = self._tracker
            if tracker is not None and getattr(tracker, "active", False):
                try:
                    result = tracker.update(frame.image)
                except Exception:
                    logger.exception("跟踪器更新失败")

            with self._frame_lock:
                self._last_frame = frame.image
                if result is not None:
                    self._last_result = result
                    # 新鲜度时间戳：控制回路靠它识别「相机断流」。
                    # 没有它，USB 相机被拔掉后 _last_result 永远冻结在
                    # 最后一帧，回路会用陈旧误差持续下发恒定速率，
                    # 基座一路转到限位 —— 而 tracker 因为不再被调用，
                    # 永远没机会报 lost。
                    self._last_result_ts = time.monotonic()

    def _control_loop(self) -> None:
        period = 1.0 / max(1.0, self._loop_hz)
        next_t = time.monotonic()
        prev_t = next_t
        while not self._stop_evt.is_set():
            next_t += period
            now = time.monotonic()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                inst = 1.0 / dt
                self._loop_hz_meas += 0.1 * (inst - self._loop_hz_meas)

            try:
                self._control_step(dt if dt > 0 else period)
            except Exception:
                # 控制回路绝不能死：出错时先保安全（停车）再继续跑
                logger.exception("控制回路异常，基座速率已清零")
                self._zero_rates()
                with self._lock:
                    self._state = "error"
                    self._message = "控制回路出现异常，基座已停转。请查看日志后重试。"

            sleep = next_t - time.monotonic()
            if sleep > 0:
                # 用 Event.wait 而不是 sleep：shutdown 时能立刻醒来
                self._stop_evt.wait(sleep)
            else:
                # 已经落后于时刻表：放弃追帧，重新对表，避免连环补偿
                next_t = time.monotonic()

    def _control_step(self, dt: float) -> None:
        with self._lock:
            state = self._state
            mount = self._mount
            cal = self._calibration
        if state not in ("tracking", "lost"):
            return
        with self._frame_lock:
            res = self._last_result
            frame = self._last_frame
            res_ts = self._last_result_ts
        if res is None or frame is None or mount is None or cal is None:
            return

        # ---- 新鲜度检查：相机断流时绝不能用冻结的旧结果驱动基座 ----
        if res_ts is not None and time.monotonic() - res_ts > 0.5:
            if state != "lost":
                self._zero_rates()
                with self._lock:
                    self._state = "lost"
                    self._message = ("相机没有新画面，基座已停转。"
                                     "检查相机连接后重新开始跟踪。")
                logger.warning("跟踪结果超过 0.5s 未更新（相机断流？），已停车")
            return

        res_state = getattr(res, "state", "lost")
        res_ok = bool(getattr(res, "ok", False))
        center = getattr(res, "center", None)

        # ---- 丢失：立即停车。这是本模块最重要的一行安全逻辑 ----
        if res_state == "lost" or (not res_ok and res_state != "coasting") \
                or center is None:
            if state != "lost":
                self._zero_rates()
                with self._lock:
                    self._state = "lost"
                    self._message = ("目标丢失，基座已停转。"
                                     "请重新框选目标，或等待自动重捕。")
                logger.warning("目标丢失，基座速率已清零")
            return

        # ---- 丢失后恢复 ----
        if state == "lost":
            with self._lock:
                self._state = "tracking"
                self._message = "目标已重捕，继续跟踪。"
            self._pid_az.reset()
            self._pid_alt.reset()
            logger.info("目标重捕成功，恢复跟踪")

        fh, fw = frame.shape[:2]
        ex = float(center[0]) - fw / 2.0
        ey = float(center[1]) - fh / 2.0
        err_px = math.hypot(ex, ey)

        # ---- 死区：残差已小于抖动量级，下发 0 速率防止基座来回蠕动 ----
        if err_px < self._deadband_px:
            self._rate_az = 0.0
            self._rate_alt = 0.0
            try:
                mount.set_rate(0.0, 0.0)
            except Exception:
                logger.exception("死区内下发零速率失败")
            return

        alt = self._get_alt()
        # 误差与前馈都取负号：像素误差是「目标偏离中心多少」，
        # 而基座该走的方向是把这段位移**抵消**掉（标定矩阵 A 的约定是
        # 「基座移动 → 目标像素位移」，所以求逆向修正就是喂 -e、-v）。
        e_az, e_alt = cal.pixels_to_angles(-ex, -ey, alt)
        vx = float(getattr(res, "vx", 0.0))
        vy = float(getattr(res, "vy", 0.0))

        # ---- 前馈 = 目标的**绝对**角速度估计 ----
        # 相机固连在基座上，像素速度只反映「目标相对基座」的残余运动：
        # 跟稳时像素速度≈0，可目标在天上照样跑。只拿像素速度当前馈，
        # 维持转速的活全落在 P 项上，稳态误差 = 目标速度/kp ——
        # 实测就是残差跟着目标速度上下摆、永远收不紧。
        # 正确的估计：绝对速度 = 当前指令速率 + 像素速度折算，
        # 再一阶低通压住测量噪声（vx/vy 本身有质心抖动的高频分量）。
        dv_az, dv_alt = cal.pixels_to_angles(-vx, -vy, alt)
        raw_ff_az = self._rate_az + dv_az
        raw_ff_alt = self._rate_alt + dv_alt
        a = 0.25
        self._ff_az += a * (raw_ff_az - self._ff_az)
        self._ff_alt += a * (raw_ff_alt - self._ff_alt)

        rate_az = self._pid_az.update(e_az, dt, feedforward=self._ff_az)
        rate_alt = self._pid_alt.update(e_alt, dt, feedforward=self._ff_alt)

        try:
            mount.set_rate(rate_az, rate_alt)
            self._rate_az = rate_az
            self._rate_alt = rate_alt
        except Exception:
            logger.exception("下发速率失败，尝试停车")
            self._zero_rates()
            with self._lock:
                self._state = "error"
                self._message = "基座通信失败，已尝试停车。请检查串口连接。"

    # ---------------- 输出 ----------------

    def status(self) -> Dict[str, Any]:
        with self._frame_lock:
            res = self._last_result
        err_px = None
        err_arcsec = None
        score = 0.0
        lost_frames = 0
        locked = False
        if res is not None:
            center = getattr(res, "center", None)
            score = float(getattr(res, "score", 0.0))
            lost_frames = int(getattr(res, "lost_frames", 0))
            frame = self._last_frame
            if center is not None and frame is not None:
                fh, fw = frame.shape[:2]
                err_px = math.hypot(float(center[0]) - fw / 2.0,
                                    float(center[1]) - fh / 2.0)
        with self._lock:
            cal = self._calibration
            calibrated = cal is not None and cal.ok
            state = self._state
            message = self._message
            calib_progress = self._calib_progress
            calib_message = self._calib_message
            mount = self._mount
            camera = self._camera
        if err_px is not None and calibrated:
            err_arcsec = err_px * cal.arcsec_per_px
        if err_px is not None:
            locked = state == "tracking" and err_px <= self._lock_radius_px

        az = 0.0
        alt = 0.0
        if mount is not None and getattr(mount, "connected", False):
            try:
                az, alt = mount.get_altaz()
            except Exception:
                logger.exception("查询基座坐标失败")

        tel = Telemetry(
            state=state, message=message, az=float(az), alt=float(alt),
            rate_az=self._rate_az, rate_alt=self._rate_alt,
            err_px=err_px, err_arcsec=err_arcsec, locked=locked,
            score=score, fps=self._fps, loop_hz=self._loop_hz_meas,
            lost_frames=lost_frames, calibrated=calibrated,
        )
        d = tel.to_dict()
        # 契约要求：标定进度通过 status 暴露给前端轮询
        d["calib_progress"] = calib_progress
        d["calib_message"] = calib_message
        d["mount_connected"] = mount is not None and getattr(mount, "connected", False)
        d["camera_opened"] = camera is not None and getattr(camera, "opened", False)
        d["calibration"] = cal.to_dict() if cal is not None else None
        return d

    # ---- 画面标注 ----

    _font_cache = None

    @classmethod
    def _load_font(cls, size: int = 18):
        if cls._font_cache is not None:
            return cls._font_cache
        for path in _FONT_CANDIDATES:
            if os.path.exists(path):
                try:
                    cls._font_cache = ImageFont.truetype(path, size)
                    return cls._font_cache
                except Exception:
                    continue
        try:
            cls._font_cache = ImageFont.load_default()
        except Exception:
            cls._font_cache = None
        return cls._font_cache

    def _put_text(self, img: np.ndarray, lines, org: Tuple[int, int],
                  color: Tuple[int, int, int]) -> np.ndarray:
        """在画面上写状态文字。OpenCV 不支持中文，优先 Pillow；
        没有 Pillow 时退化为英文并提示一次安装方法（不因此崩溃）。"""
        x, y = org
        if _PIL_OK:
            font = self._load_font()
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            for i, text in enumerate(lines):
                # 先描黑边再写字，保证亮背景上也读得清
                draw.text((x, y + i * 24), text,
                          font=font, fill=(color[2], color[1], color[0]),
                          stroke_width=2, stroke_fill=(0, 0, 0))
            return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        if not self._warned_no_pil:
            self._warned_no_pil = True
            logger.warning("未安装 Pillow，画面文字退化为英文。"
                           "执行 pip install pillow 后即可显示中文。")
        for i, text in enumerate(lines):
            cv2.putText(img, text, (x, y + 16 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (x, y + 16 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        return img

    @staticmethod
    def _dashed_rect(img: np.ndarray, pt1: Tuple[int, int],
                     pt2: Tuple[int, int], color: Tuple[int, int, int],
                     dash: int = 8, thickness: int = 1) -> None:
        """虚线矩形：OpenCV 没有原生虚线，用短线段拼。"""
        x1, y1 = pt1
        x2, y2 = pt2
        for xa in range(min(x1, x2), max(x1, x2), dash * 2):
            cv2.line(img, (xa, y1), (min(xa + dash, max(x1, x2)), y1), color, thickness)
            cv2.line(img, (xa, y2), (min(xa + dash, max(x1, x2)), y2), color, thickness)
        for ya in range(min(y1, y2), max(y1, y2), dash * 2):
            cv2.line(img, (x1, ya), (x1, min(ya + dash, max(y1, y2))), color, thickness)
            cv2.line(img, (x2, ya), (x2, min(ya + dash, max(y1, y2))), color, thickness)

    def annotated_frame(self) -> Optional[np.ndarray]:
        """画上跟踪框、十字丝、锁定环、搜索窗的一帧，供 MJPEG 推流。"""
        with self._frame_lock:
            if self._last_frame is None:
                return None
            img = self._last_frame.copy()
            res = self._last_result
        with self._lock:
            state = self._state
            calibrated = self._calibration is not None and self._calibration.ok
            calib_progress = self._calib_progress
            calib_message = self._calib_message
            arcsec = self._calibration.arcsec_per_px if calibrated else None

        fh, fw = img.shape[:2]
        cx, cy = fw // 2, fh // 2

        # 中心十字丝：留出中心空隙，不遮挡目标本体
        ch_color = (200, 200, 200)
        gap, arm = 8, 28
        cv2.line(img, (cx - gap - arm, cy), (cx - gap, cy), ch_color, 1)
        cv2.line(img, (cx + gap, cy), (cx + gap + arm, cy), ch_color, 1)
        cv2.line(img, (cx, cy - gap - arm), (cx, cy - gap), ch_color, 1)
        cv2.line(img, (cx, cy + gap, ), (cx, cy + gap + arm), ch_color, 1)

        err_px = None
        score = 0.0
        locked = False
        res_state = ""
        if res is not None:
            res_state = str(getattr(res, "state", ""))
            score = float(getattr(res, "score", 0.0))
            center = getattr(res, "center", None)
            box = getattr(res, "box", None)
            if center is not None:
                err_px = math.hypot(float(center[0]) - cx, float(center[1]) - cy)
                locked = state == "tracking" and err_px <= self._lock_radius_px

            if box is not None:
                x, y, w, h = (int(round(float(v))) for v in box)
                if res_state == "lost" or state == "lost":
                    color = (0, 0, 255)      # 红：丢失（画最后已知位置）
                elif locked:
                    color = (0, 220, 0)      # 绿：已锁定
                else:
                    color = (0, 220, 220)    # 黄：跟踪中未入锁定环
                if res_state == "coasting":
                    # 遮挡外推位置不确定，用虚线区别于实测框
                    self._dashed_rect(img, (x, y), (x + w, y + h), (0, 220, 220))
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # 丢失/外推时画搜索窗：让用户知道重捕会在哪个范围里找
            if center is not None and res_state in ("coasting", "lost"):
                scale = float((self._cfg.get("tracker", {}) or {})
                              .get("search_scale", 2.5))
                if box is not None:
                    sw = int(box[2] * scale)
                    sh = int(box[3] * scale)
                    sx = int(center[0] - sw / 2.0)
                    sy = int(center[1] - sh / 2.0)
                    self._dashed_rect(img, (sx, sy), (sx + sw, sy + sh),
                                      (128, 128, 128), dash=5)

        # 锁定环：目标进环即认为「锁定」，环色随锁定状态变化
        ring_color = (0, 220, 0) if locked else (110, 110, 110)
        cv2.circle(img, (cx, cy), int(self._lock_radius_px), ring_color, 1,
                   lineType=cv2.LINE_AA)

        # 左上角状态文字（中文；无 Pillow 时自动退化为英文）
        zh = _PIL_OK
        state_txt = (_STATE_ZH if zh else _STATE_EN).get(state, state)
        lines = []
        if state == "calibrating":
            if zh:
                lines.append("状态：%s %d%%" % (state_txt, int(calib_progress * 100)))
                if calib_message:
                    lines.append(calib_message)
            else:
                lines.append("state: %s %d%%" % (state_txt, int(calib_progress * 100)))
            # 进度条：轮询 status 之外给画面观看者一个直观进度
            bar_w = 220
            cv2.rectangle(img, (12, 78), (12 + bar_w, 90), (80, 80, 80), 1)
            cv2.rectangle(img, (12, 78), (12 + int(bar_w * calib_progress), 90),
                          (0, 220, 220), -1)
        else:
            if zh:
                line = "状态：%s" % state_txt
                if err_px is not None:
                    line += "  误差 %.1f 像素" % err_px
                    if arcsec is not None:
                        line += "（%.0f 角秒）" % (err_px * arcsec)
                lines.append(line)
                extra = "置信 %.2f" % score if res is not None else ""
                extra += "  未标定" if not calibrated else ""
                if extra.strip():
                    lines.append(extra.strip())
            else:
                line = "state: %s" % state_txt
                if err_px is not None:
                    line += "  err %.1fpx" % err_px
                lines.append(line)
                if not calibrated:
                    lines.append("NOT CALIBRATED")
        state_color = {"tracking": (0, 220, 0), "lost": (0, 0, 255),
                       "error": (0, 0, 255)}.get(state, (230, 230, 230))
        img = self._put_text(img, lines, (12, 10), state_color)
        return img


# ======================================================================
# 自检：假 mount/camera/tracker 把状态机走一圈。
# 验证：未标定拒绝 start；标定注入后速率方向正确；死区下发 0；
#       丢失时速率立即清零；重捕后恢复。
# ======================================================================

if __name__ == "__main__":
    import types

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("control.selftest")

    W, H = 640, 480

    class _FakeMount:
        max_rate_deg_s = 2.5

        def __init__(self):
            self.connected = True
            self.rates = []          # 每次 set_rate 的记录
            self.last_rate = (None, None)

        def get_altaz(self):
            return (180.0, 30.0)

        def set_rate(self, az_dps, alt_dps):
            self.last_rate = (az_dps, alt_dps)
            self.rates.append((az_dps, alt_dps))

        def stop(self):
            self.set_rate(0.0, 0.0)

        def nudge(self, d_az, d_alt, timeout_s=25.0):
            return True

        def close(self):
            self.connected = False

        def info(self):
            return {"driver": "fake"}

    class _FakeCamera:
        opened = True

        def __init__(self):
            self._i = 0

        def grab(self):
            self._i += 1
            time.sleep(1.0 / 120.0)  # 模拟相机节流，避免采集线程空转烧 CPU
            return types.SimpleNamespace(
                image=np.zeros((H, W, 3), dtype=np.uint8),
                ts=time.time(), index=self._i)

        def close(self):
            self.opened = False

        def info(self):
            return {"backend": "fake", "width": W, "height": H}

    def _mkres(state, center, ok=True, score=0.9,
               vx=0.0, vy=0.0, lost_frames=0):
        box = None
        if center is not None:
            box = (center[0] - 20, center[1] - 20, 40.0, 40.0)
        return types.SimpleNamespace(ok=ok, box=box, center=center,
                                     score=score, state=state,
                                     lost_frames=lost_frames, vx=vx, vy=vy)

    class _FakeTracker:
        def __init__(self):
            self.active = False
            self.result = _mkres("idle", None, ok=False)

        def start(self, frame, box):
            self.active = True
            return True

        def update(self, frame):
            return self.result

        def reset(self):
            self.active = False

    cfg = {
        "mount": {"max_rate_deg_s": 2.5},
        "camera": {"width": W, "height": H},
        "tracker": {"search_scale": 2.5},
        "control": {"loop_hz": 50,
                    "pid": {"kp": 0.9, "ki": 0.06, "kd": 0.05,
                            "i_limit": 0.5, "i_band": 0.4},
                    "lock_radius_px": 30, "deadband_px": 2},
        "calibration": {"file": ""},   # 自检不落盘
    }

    session = TrackingSession(cfg)
    mount = _FakeMount()
    camera = _FakeCamera()
    tracker = _FakeTracker()

    # 注入假设备：不经过 connect/open（那会去导入尚在开发的兄弟模块）
    session._mount = mount
    session._camera = camera
    session._tracker_factory = lambda: tracker

    # 等采集线程出第一帧
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with session._frame_lock:
            if session._last_frame is not None:
                break
        time.sleep(0.02)
    assert session._last_frame is not None, "采集线程应产出画面"

    # ---- idle → selecting ----
    r = session.select_target((0.45, 0.45, 0.1, 0.1))
    assert r["ok"], r["message"]
    assert session.status()["state"] == "selecting"

    # ---- 未标定：start_tracking 必须拒绝 ----
    r = session.start_tracking()
    assert not r["ok"] and "标定" in r["message"], "未标定应拒绝并说明原因"
    assert session.status()["state"] == "selecting"
    log.info("未标定拒绝启动：%s", r["message"])

    # ---- 注入标定：A = diag(c, c)，c = 6角秒/px（正号约定见注释）----
    c = 6.0 / 3600.0
    session._calibration = Calibration(
        a11=c, a12=0.0, a21=0.0, a22=c, arcsec_per_px=6.0,
        angle_deg=0.0, flipped=False, rms_px=0.3, ok=True)

    r = session.start_tracking()
    assert r["ok"], r["message"]
    assert session.status()["state"] == "tracking"

    # ---- 方向检查：目标在中心右上 (ex=+40, ey=-25) ----
    # A 为正对角阵时，修正 = A@(-e)：az 应为负、alt 应为正。
    tracker.result = _mkres("tracking", (W / 2 + 40.0, H / 2 - 25.0))
    time.sleep(0.4)
    az_r, alt_r = mount.last_rate
    assert az_r is not None and az_r < 0, "az 速率方向应为负，实际 %s" % az_r
    assert alt_r > 0, "alt 速率方向应为正，实际 %s" % alt_r
    log.info("速率方向正确：az=%.4f°/s alt=%.4f°/s", az_r, alt_r)
    st = session.status()
    assert st["err_px"] is not None and abs(st["err_px"] - math.hypot(40, 25)) < 1.0
    assert st["err_arcsec"] is not None

    # ---- 死区：误差 1px < deadband 2px，应下发 0 速率 ----
    tracker.result = _mkres("tracking", (W / 2 + 1.0, H / 2))
    time.sleep(0.3)
    assert mount.last_rate == (0.0, 0.0), \
        "死区内应下发零速率，实际 %s" % (mount.last_rate,)
    log.info("死区防抖生效：残差 1px 时速率 %s", mount.last_rate)

    # ---- 锁定判定：小误差应报告 locked ----
    tracker.result = _mkres("tracking", (W / 2 + 5.0, H / 2 + 5.0))
    time.sleep(0.2)
    assert session.status()["locked"], "误差在锁定环内应报告 locked"

    # ---- 丢失：速率必须立即清零 ----
    tracker.result = _mkres("tracking", (W / 2 + 120.0, H / 2 + 80.0))
    time.sleep(0.3)
    assert mount.last_rate != (0.0, 0.0), "大误差时应有非零速率（前置条件）"
    tracker.result = _mkres("lost", None, ok=False, score=0.0, lost_frames=30)
    deadline = time.monotonic() + 2.0
    while session.status()["state"] != "lost" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert session.status()["state"] == "lost"
    assert mount.last_rate == (0.0, 0.0), "丢失时速率必须清零"
    # 进入 lost 之后不允许再出现任何非零速率（哪怕一拍）
    mount.rates = []
    time.sleep(0.3)
    nonzero = [r for r in mount.rates if r != (0.0, 0.0)]
    assert not nonzero, "丢失后仍下发了非零速率：%s" % nonzero
    log.info("丢失清零验证通过：state=lost，rate=%s", mount.last_rate)

    # ---- 重捕恢复 ----
    tracker.result = _mkres("tracking", (W / 2 + 10.0, H / 2))
    time.sleep(0.3)
    assert session.status()["state"] == "tracking", "重捕后应回到 tracking"

    # ---- coasting 与标注画面 ----
    tracker.result = _mkres("coasting", (W / 2 + 30.0, H / 2 + 10.0),
                            ok=False, score=0.2)
    time.sleep(0.2)
    ann = session.annotated_frame()
    assert ann is not None and ann.shape == (H, W, 3), "标注帧应与原帧同尺寸"
    assert int((ann != 0).sum()) > 0, "标注帧上应画有内容"

    # ---- status 必须带标定进度字段（契约要求）----
    st = session.status()
    assert "calib_progress" in st and "calib_message" in st
    assert st["calibrated"] is True

    # ---- stop_tracking 回 selecting，PID 复位、速率清零 ----
    session.stop_tracking()
    assert session.status()["state"] == "selecting"
    assert mount.last_rate == (0.0, 0.0)

    session.shutdown()
    log.info("control selftest OK")
