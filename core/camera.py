# -*- coding: utf-8 -*-
"""相机模块：OpenCV 实体相机 + 合成星空仿真相机。

SimCamera 是闭环仿真的关键一环：画面内容由 get_pointing() 回调返回的
(az, alt) 决定 —— 基座转动、指向变化，星空与目标就会在画面里反向平移。
这样控制回路（视觉误差 → PID → 基座速率 → 指向变化 → 视觉误差变小）
才能在没有任何硬件的情况下真实成立。
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 契约数据结构
# ---------------------------------------------------------------------------

def _open_capture(source):
    """打开视频源。Windows 上强制走 DirectShow：
    默认的 MSMF 后端枚举/打开 USB 摄像头动辄卡十几秒，
    某些外接摄像头干脆打不开 —— 用户的体感就是「外部摄像头没法用」。"""
    if sys.platform == "win32" and isinstance(source, int):
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


@dataclass

class Frame:
    image: np.ndarray        # BGR uint8
    ts: float
    index: int


class CameraBase:
    """相机抽象基类。子类必须实现全部方法；grab 失败返回 None 而不是抛异常，
    因为采集线程要长期运行，个别坏帧不应打断整个回路。"""

    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def grab(self) -> Optional[Frame]:
        raise NotImplementedError

    @property
    def opened(self) -> bool:
        raise NotImplementedError

    def info(self) -> Dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 工具：角度换算与 stderr 抑制
# ---------------------------------------------------------------------------

def _wrap_deg(d: float) -> float:
    """把角度差折到 [-180, 180)。方位 0/360 跨界时若不折叠，
    目标会被投影到几万像素之外，画面直接空掉。"""
    return (d + 180.0) % 360.0 - 180.0


@contextlib.contextmanager
def _quiet_stderr():
    """临时把 stderr 重定向到 /dev/null（文件描述符层面）。

    为什么必须这么做：macOS 上 AVFoundation 探测不存在的摄像头索引时，
    C++ 层会直接向 fd 2 打印大段错误，Python 层的 sys.stderr 替换拦不住，
    只能在 fd 层面临时换掉。异常时也必须恢复，否则后续日志全部丢失。
    """
    try:
        fd = sys.stderr.fileno()
    except Exception:
        # 某些嵌入环境 stderr 没有真实 fd，此时无从抑制，直接放行
        yield
        return
    saved = os.dup(fd)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            sys.stderr.flush()
            os.dup2(devnull, fd)
        finally:
            os.close(devnull)
        yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved, fd)
        os.close(saved)


@contextlib.contextmanager
def _quiet_opencv_log():
    """临时压低 OpenCV 自身的日志级别（与 stderr 抑制配合使用）。"""
    prev = None
    try:
        prev = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        prev = None
    try:
        yield
    finally:
        if prev is not None:
            try:
                cv2.utils.logging.setLogLevel(prev)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# OpenCV 实体相机
# ---------------------------------------------------------------------------

class OpenCVCamera(CameraBase):
    def __init__(self, source, width: int = 1280, height: int = 720,
                 fps: float = 30.0):
        # source 允许 int（本机设备号）或 str（视频文件 / 网络流地址）；
        # 配置文件里数字常被写成字符串，这里就地纠正，避免打开失败
        if isinstance(source, str) and source.strip().lstrip("-").isdigit():
            source = int(source.strip())
        self._source = source
        self._width = int(width)
        self._height = int(height)
        self._fps = float(fps)
        self._cap: Optional[cv2.VideoCapture] = None
        self._index = 0
        self._fail_count = 0

    def open(self) -> None:
        if self._cap is not None:
            return
        with _quiet_opencv_log():
            cap = _open_capture(self._source)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                "无法打开摄像头（source=%r）。请检查：设备是否被其他程序占用、"
                "索引/地址是否正确；macOS 首次使用需在「系统设置→隐私与安全性→"
                "摄像头」里授权本程序。" % (self._source,))
        # 分辨率/帧率是"请求"不是"保证"，驱动可能给出最接近的档位，
        # 所以设置后要读回真实值，info() 才不会骗人
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self._width
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self._height
        real_fps = cap.get(cv2.CAP_PROP_FPS)
        if real_fps and real_fps > 0:
            self._fps = float(real_fps)
        self._cap = cap
        self._index = 0
        logger.info("摄像头已打开：source=%r %dx%d @%.1ffps",
                    self._source, self._width, self._height, self._fps)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("摄像头已关闭（source=%r）", self._source)

    def grab(self) -> Optional[Frame]:
        if self._cap is None:
            return None
        ok, img = self._cap.read()
        if not ok or img is None:
            # 节流必须用**独立的失败计数**：_index 只在成功时递增，
            # 拿它做节流的话，卡在 index=0 会每帧刷一条、卡在其他值则
            # 一条都不打 —— 相机拔线了用户完全不知道。
            self._fail_count += 1
            if self._fail_count % 30 == 1:
                logger.warning("取帧失败已连续 %d 次（source=%r），检查相机连接",
                               self._fail_count, self._source)
            return None
        self._fail_count = 0
        if img.ndim == 2:
            # 契约规定输出 BGR；灰度相机（部分天文相机模式）在此补齐通道
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        frame = Frame(image=img, ts=time.time(), index=self._index)
        self._index += 1
        return frame

    @property
    def opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def info(self) -> Dict[str, Any]:
        backend = None
        if self._cap is not None:
            try:
                backend = self._cap.getBackendName()
            except Exception:
                backend = None
        return {"backend": backend or "opencv", "width": self._width,
                "height": self._height, "fps": self._fps,
                "source": self._source}


# ---------------------------------------------------------------------------
# 合成仿真相机
# ---------------------------------------------------------------------------

class SimCamera(CameraBase):
    """合成场景，无硬件时演示用。星空 + 一个可被框选的运动目标。
    画面随 get_pointing() 变化，形成真闭环。

    投影模型（与标定模块的数学一致）：
      x = cx + wrap(obj_az - az) * cos(alt) * (3600 / arcsec_per_px)
      y = cy - (obj_alt - alt) * (3600 / arcsec_per_px)
    方位差乘 cos(alt) 是因为像素反映的是天球角距，不是方位轴角度；
    y 取负是因为图像坐标向下增大而仰角向上增大。
    """

    _TILE_DEG = 0.5          # 星表按角度瓦片生成：指向到哪片天区，星就到哪
    _BASE_SEED = 20260730    # 固定种子：同一天区任何时刻看到同一批星

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30.0,
                 arcsec_per_px: float = 6.0,
                 get_pointing: Optional[Callable[[], Tuple[float, float]]] = None,
                 target_rate_deg_s: float = 0.12, target_heading_deg: float = 40.0):
        self._width = int(width)
        self._height = int(height)
        self._fps = max(1.0, float(fps))
        self._arcsec_per_px = float(arcsec_per_px)
        self._get_pointing = get_pointing
        self._target_rate = float(target_rate_deg_s)
        self._target_heading = float(target_heading_deg)

        self._opened = False
        self._index = 0
        self._t0 = 0.0                 # 目标运动的时间起点
        self._target_az0 = 0.0
        self._target_alt0 = 0.0
        self._last_grab_t = 0.0
        # 瓦片星表缓存：避免每帧重建 RandomState；长时间大范围转动时清空防膨胀
        self._tile_cache: Dict[Tuple[int, int], Tuple[np.ndarray, ...]] = {}
        # 帧噪声用独立 RNG（不固定种子）：噪声要每帧不同才像真实传感器
        self._noise_rng = np.random.default_rng()
        # 供自检验证"目标确实在动"：最近一次渲染的目标像素位置
        self._last_target_px: Optional[Tuple[float, float]] = None

    # -- 指向与目标位置 --------------------------------------------------

    def _pointing(self) -> Tuple[float, float]:
        if self._get_pointing is not None:
            try:
                az, alt = self._get_pointing()
                return float(az), float(alt)
            except Exception:
                # 回调坏了不能让采集线程崩掉，退回默认指向并留下线索
                logger.exception("get_pointing 回调异常，改用默认指向")
        return 180.0, 35.0

    #: 目标沿航线往返运动的单程长度（度）。匀速直线一去不回的话，
    #: 用户打开相机几分钟后目标就飞出好几度，怎么找都找不到 ——
    #: 演示场景里目标必须永远「在附近」。三角波往返保持匀速率，
    #: 只在端点瞬间掉头，对跟踪演示几乎无感。
    #: 幅度取 0.45°：加上 0.2~0.3° 的初始偏置后仍在默认视场
    #: （1.2°×0.7°）之内，基座不动时目标也全程可见。
    _TARGET_SPAN_DEG = 0.45

    def _target_altaz(self, t: float) -> Tuple[float, float]:
        """目标从初始点起、沿 heading 方向以恒定速率**往返**运动。
        heading 取罗盘约定：0°=正仰角方向（画面向上），90°=正方位（向右）。"""
        dt = t - self._t0
        span = self._TARGET_SPAN_DEG
        # 正弦往返：位置 = span·(1−cos)/2，平均速率 = target_rate。
        # 不用三角波是因为三角波在端点瞬间反向（无穷大加速度），
        # PID 每 7 秒被踢出一个 70px 的尖峰 —— 真实目标不会这样飞。
        # 正弦的速度连续变化，峰值 = 平均 × π/2，仍远低于基座上限。
        period = 2.0 * span / max(self._target_rate, 1e-9)
        dist = span * 0.5 * (1.0 - math.cos(2.0 * math.pi * dt / period))
        h = math.radians(self._target_heading)
        az = self._target_az0 + dist * math.sin(h)
        alt = self._target_alt0 + dist * math.cos(h)
        return az, alt

    # -- 生命周期 --------------------------------------------------------

    def open(self) -> None:
        if self._opened:
            return
        az, alt = self._pointing()
        # 目标初始位置钉在"打开相机时的指向"附近，保证一打开就能在画面里
        # 找到它（+0.3°/+0.2° 折合约 +147/-120 像素，位于画面右上区域）
        self._target_az0 = az + 0.3
        self._target_alt0 = alt + 0.2
        self._t0 = time.time()
        self._index = 0
        self._last_grab_t = 0.0
        self._opened = True
        logger.info("仿真相机已打开：%dx%d @%.1ffps，%.1f 角秒/像素，"
                    "目标初始 (az=%.3f, alt=%.3f)，速率 %.3f°/s 方向 %.1f°",
                    self._width, self._height, self._fps, self._arcsec_per_px,
                    self._target_az0, self._target_alt0,
                    self._target_rate, self._target_heading)

    def close(self) -> None:
        self._opened = False
        self._tile_cache.clear()

    @property
    def opened(self) -> bool:
        return self._opened

    def info(self) -> Dict[str, Any]:
        return {"backend": "simulator", "width": self._width,
                "height": self._height, "fps": self._fps,
                "source": "synthetic"}

    # -- 星表 ------------------------------------------------------------

    def _tile_stars(self, ix: int, iy: int) -> Tuple[np.ndarray, ...]:
        """按 (方位瓦片, 仰角瓦片) 生成确定性的星点集合。
        种子由瓦片索引推出：指向怎么移动，同一片天区的星都不变。"""
        key = (ix, iy)
        cached = self._tile_cache.get(key)
        if cached is not None:
            return cached
        if len(self._tile_cache) > 4096:
            # 长时间巡天会积累大量瓦片；星表可重算，直接清空最省事
            self._tile_cache.clear()
        # 种子用**折回后**的方位瓦片索引：az=0.05° 和 az=360.05° 是同一片天，
        # 直接用原始 ix 会让基座过北点（方位折回）时整片星空瞬变
        ixw = ix % int(round(360.0 / self._TILE_DEG))
        seed = (self._BASE_SEED ^ (ixw * 73856093) ^ (iy * 19349663)) & 0x7FFFFFFF
        rng = np.random.RandomState(seed)
        n = int(rng.randint(18, 34))
        az = (ix + rng.rand(n)) * self._TILE_DEG
        alt = (iy + rng.rand(n)) * self._TILE_DEG
        # 亮度取幂律：暗星多亮星少，画面层次才像真实星空
        bright = (255.0 * (rng.rand(n) ** 2.2) * 0.85 + 30.0).clip(0, 255)
        radius = rng.choice([0.8, 1.0, 1.3, 1.8, 2.4], size=n,
                            p=[0.35, 0.3, 0.2, 0.1, 0.05])
        result = (az, alt, bright, radius)
        self._tile_cache[key] = result
        return result

    # -- 渲染 ------------------------------------------------------------

    def _render(self, az: float, alt: float, t: float) -> np.ndarray:
        w, h = self._width, self._height
        k = 3600.0 / self._arcsec_per_px           # 像素/度
        cos_alt = max(0.05, math.cos(math.radians(alt)))  # 防天顶除零发散
        cx, cy = w / 2.0, h / 2.0

        # 背景：暗底 + 轻微垂直渐变模拟天光，避免纯黑导致跟踪器无梯度可用
        lum = np.empty((h, w), np.float32)
        lum[:] = np.linspace(16.0, 9.0, h, dtype=np.float32)[:, None]

        # 可见角度范围（多留一圈瓦片，保证边缘的星不会突然消失/出现）
        half_az = (w / 2.0) / (k * cos_alt)
        half_alt = (h / 2.0) / k
        ix0 = math.floor((az - half_az) / self._TILE_DEG) - 1
        ix1 = math.floor((az + half_az) / self._TILE_DEG) + 1
        iy0 = math.floor((alt - half_alt) / self._TILE_DEG) - 1
        iy1 = math.floor((alt + half_alt) / self._TILE_DEG) + 1

        star_layer = np.zeros((h, w), np.float32)
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                s_az, s_alt, s_b, s_r = self._tile_stars(ix, iy)
                for j in range(s_az.shape[0]):
                    x = cx + _wrap_deg(s_az[j] - az) * cos_alt * k
                    y = cy - (s_alt[j] - alt) * k
                    if -4 <= x < w + 4 and -4 <= y < h + 4:
                        # shift=4 亚像素绘制：指向缓慢变化时星点平滑移动，
                        # 否则 1 像素跳变会污染相位相关和标定
                        cv2.circle(star_layer,
                                   (int(round(x * 16)), int(round(y * 16))),
                                   int(round(s_r[j] * 16)),
                                   float(s_b[j]), -1, cv2.LINE_AA, shift=4)
        # 一次高斯模糊近似大气/离焦 PSF，让星点有柔和轮廓
        star_layer = cv2.GaussianBlur(star_layer, (5, 5), 1.1)
        lum += star_layer

        # 传感器噪声：每帧不同，量级要小到不干扰 CSRT，但足以让画面"活"
        lum += self._noise_rng.normal(0.0, 3.0, size=(h, w)).astype(np.float32)
        lum = np.clip(lum, 0, 255)

        # 转 BGR 并加一点冷色调（B 略强于 R），更接近夜空观感
        img = np.empty((h, w, 3), np.uint8)
        img[:, :, 0] = lum.astype(np.uint8)
        img[:, :, 1] = (lum * 0.97).astype(np.uint8)
        img[:, :, 2] = (lum * 0.93).astype(np.uint8)

        # 目标
        t_az, t_alt = self._target_altaz(t)
        tx = cx + _wrap_deg(t_az - az) * cos_alt * k
        ty = cy - (t_alt - alt) * k
        self._last_target_px = (tx, ty)
        if -60 <= tx < w + 60 and -60 <= ty < h + 60:
            # 机头朝向 = 运动方向在图像里的角度（y 轴向下所以取负仰角分量）
            hr = math.radians(self._target_heading)
            ang = math.atan2(-math.cos(hr), math.sin(hr) * cos_alt)
            self._draw_target(img, tx, ty, ang)
        return img

    @staticmethod
    def _rot_pts(pts, ang: float, tx: float, ty: float) -> np.ndarray:
        """局部坐标 → 旋转平移 → shift=4 定点坐标。集中在一处做，
        保证机身/机翼/条纹用同一变换，不会错位。"""
        ca, sa = math.cos(ang), math.sin(ang)
        out = []
        for (px, py) in pts:
            x = tx + px * ca - py * sa
            y = ty + px * sa + py * ca
            out.append((int(round(x * 16)), int(round(y * 16))))
        return np.array(out, dtype=np.int32)

    def _draw_target(self, img: np.ndarray, tx: float, ty: float,
                     ang: float) -> None:
        """画一个几十像素、带纹理的小飞机。刻意加条纹/舷窗/双色机翼：
        CSRT 靠模板里的空间结构工作，一个纯色亮点会让它无从锁定。"""
        shift = 4
        aa = cv2.LINE_AA

        def poly(pts, color):
            cv2.fillPoly(img, [self._rot_pts(pts, ang, tx, ty)], color,
                         lineType=aa, shift=shift)

        # 机翼（后掠，上下对称，颜色比机身深一档形成对比边界）
        wing_c = (150, 165, 185)
        poly([(4, 0), (-6, 17), (-11, 17), (-5, 0)], wing_c)
        poly([(4, 0), (-6, -17), (-11, -17), (-5, 0)], wing_c)
        # 平尾
        tail_c = (140, 155, 175)
        poly([(-13, 0), (-18, 8), (-21, 8), (-17, 0)], tail_c)
        poly([(-13, 0), (-18, -8), (-21, -8), (-17, 0)], tail_c)
        # 机身（亮灰白）与机头（更亮，形成沿机轴的亮度梯度）
        body = self._rot_pts([(19, 0), (14, 3.2), (-15, 3.2), (-20, 1.5),
                              (-20, -1.5), (-15, -3.2), (14, -3.2)], ang, tx, ty)
        cv2.fillPoly(img, [body], (225, 228, 235), lineType=aa, shift=shift)
        nose = self._rot_pts([(19, 0), (13, 2.6), (13, -2.6)], ang, tx, ty)
        cv2.fillPoly(img, [nose], (250, 252, 255), lineType=aa, shift=shift)
        # 橙色装饰条纹：跨越机身的高对比横向结构，是模板里最强的特征
        for sy in (-1.6, 1.6):
            stripe = self._rot_pts([(11, sy - 0.7), (11, sy + 0.7),
                                    (-14, sy + 0.7), (-14, sy - 0.7)],
                                   ang, tx, ty)
            cv2.fillPoly(img, [stripe], (60, 140, 250), lineType=aa, shift=shift)
        # 舷窗：沿机轴的一排暗点，提供周期性纹理
        for wx in (7, 2, -3, -8):
            c = self._rot_pts([(wx, 0)], ang, tx, ty)[0]
            cv2.circle(img, (int(c[0]), int(c[1])), int(0.9 * 16),
                       (70, 75, 90), -1, aa, shift=shift)
        # 尾部红色信标 + 机头白色航行灯
        for (lx, ly, col) in ((-19.5, 0.0, (60, 60, 255)),
                              (19.0, 0.0, (255, 255, 255))):
            c = self._rot_pts([(lx, ly)], ang, tx, ty)[0]
            cv2.circle(img, (int(c[0]), int(c[1])), int(1.1 * 16), col,
                       -1, aa, shift=shift)
        # 只在目标周边做轻微模糊：柔化锯齿边缘又不拖累整帧性能
        x0 = max(0, int(tx) - 40); x1 = min(img.shape[1], int(tx) + 40)
        y0 = max(0, int(ty) - 40); y1 = min(img.shape[0], int(ty) + 40)
        if x1 > x0 and y1 > y0:
            img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (3, 3), 0.7)

    # -- 取帧 ------------------------------------------------------------

    def grab(self) -> Optional[Frame]:
        if not self._opened:
            return None
        # 节流到标称帧率：真实相机 read() 就是阻塞到下一帧，仿真也保持
        # 同样的时序特性，控制回路的 dt 才有意义
        now = time.time()
        wait = self._last_grab_t + 1.0 / self._fps - now
        if wait > 0:
            time.sleep(wait)
        t = time.time()
        self._last_grab_t = t
        az, alt = self._pointing()
        img = self._render(az, alt, t)
        frame = Frame(image=img, ts=t, index=self._index)
        self._index += 1
        return frame


# ---------------------------------------------------------------------------
# 设备探测与工厂
# ---------------------------------------------------------------------------

def list_video_devices(max_index: int = 6) -> List[Dict[str, Any]]:
    """探测可用摄像头，返回 [{"index":int,"name":str,"width":int,"height":int}]。
    探测失败的设备跳过，不要抛异常。"""
    devices: List[Dict[str, Any]] = []
    for i in range(max_index):
        cap = None
        try:
            # 双重抑制：OpenCV 日志走自己的通道，AVFoundation 直接写 fd 2，
            # 两条路都要堵上，否则 macOS 上每个空索引都刷一屏错误
            with _quiet_opencv_log(), _quiet_stderr():
                cap = _open_capture(i)
                if not cap.isOpened():
                    continue
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                try:
                    backend = cap.getBackendName()
                except Exception:
                    backend = "opencv"
            devices.append({"index": i,
                            "name": "摄像头 %d (%s)" % (i, backend),
                            "width": w, "height": h})
        except Exception:
            # 契约要求任何单个设备的探测失败都不影响整体结果
            logger.debug("探测摄像头索引 %d 失败，已跳过", i, exc_info=True)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
    return devices


def build_camera(cfg: Dict[str, Any],
                 get_pointing: Optional[Callable] = None) -> CameraBase:
    """cfg 为 config.yaml 的 camera 段；driver 取值 opencv / simulator。"""
    driver = str(cfg.get("driver", "simulator")).strip().lower()
    width = int(cfg.get("width", 1280))
    height = int(cfg.get("height", 720))
    fps = float(cfg.get("fps", 30.0))
    if driver == "opencv":
        return OpenCVCamera(source=cfg.get("source", 0),
                            width=width, height=height, fps=fps)
    if driver in ("simulator", "sim"):
        return SimCamera(width=width, height=height, fps=fps,
                         arcsec_per_px=float(cfg.get("arcsec_per_px", 6.0)),
                         get_pointing=get_pointing,
                         target_rate_deg_s=float(
                             cfg.get("target_rate_deg_s", 0.12)),
                         target_heading_deg=float(
                             cfg.get("target_heading_deg", 40.0)))
    raise ValueError(
        "未知的相机驱动 %r：config.yaml 的 camera.driver 只能填 "
        "opencv 或 simulator，请修改后重试。" % (driver,))


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def _self_test() -> None:
    log = logging.getLogger("camera.selftest")

    # --- SimCamera：连续取帧，验证序号、非黑、目标移动 ---
    pointing = {"az": 180.0, "alt": 35.0}
    cam = SimCamera(width=1280, height=720, fps=30.0,
                    get_pointing=lambda: (pointing["az"], pointing["alt"]))
    cam.open()
    assert cam.opened, "打开后 opened 应为 True"
    info = cam.info()
    assert set(info) == {"backend", "width", "height", "fps", "source"}, \
        "info() 键与契约不符：%r" % (info,)

    frames: List[Frame] = []
    targets: List[Tuple[float, float]] = []
    for _ in range(30):
        f = cam.grab()
        assert f is not None, "仿真相机 grab 不应返回 None"
        frames.append(f)
        assert cam._last_target_px is not None
        targets.append(cam._last_target_px)

    idx = [f.index for f in frames]
    assert idx == list(range(idx[0], idx[0] + 30)), "帧序号必须逐帧递增: %r" % idx
    mean_lum = float(np.mean(frames[0].image))
    assert mean_lum > 5.0, "画面近乎全黑（均值 %.2f），星空/背景渲染失败" % mean_lum
    assert frames[0].image.dtype == np.uint8 and frames[0].image.shape == (720, 1280, 3), \
        "帧格式必须是 720x1280 BGR uint8"

    # 目标必须真的在移动：既看内部记录的目标像素位移，也看整帧差分
    move_px = math.hypot(targets[-1][0] - targets[0][0],
                         targets[-1][1] - targets[0][1])
    assert move_px > 10.0, "30 帧内目标仅移动 %.1f 像素，漂移逻辑失效" % move_px
    diffs = [float(np.mean(cv2.absdiff(frames[i].image, frames[i - 1].image)))
             for i in range(1, len(frames))]
    assert min(diffs) > 0.1, "相邻帧完全相同，画面没有随时间变化"
    log.info("SimCamera：30 帧序号递增，画面均值 %.1f，目标位移 %.1f px，"
             "帧间平均差 %.2f", mean_lum, move_px, sum(diffs) / len(diffs))

    # --- 闭环性：指向变化时星空整体平移，且平移量与投影模型一致 ---
    f1 = cam.grab()
    pointing["az"] += 0.10
    f2 = cam.grab()
    g1 = cv2.cvtColor(f1.image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(f2.image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    (dx, dy), _resp = cv2.phaseCorrelate(g1, g2)
    expected = 0.10 * math.cos(math.radians(pointing["alt"])) * 3600.0 / 6.0
    assert abs(abs(dx) - expected) < 4.0 and abs(dy) < 4.0, \
        "指向 +0.1°az 后画面平移 (%.1f, %.1f) px，与模型预期 %.1f px 不符" % \
        (dx, dy, expected)
    # 相机跟着目标方向转，画面里的内容应当反向移动（否则闭环符号是反的）
    assert dx < 0, "方位增大时画面应向左移，实测 dx=%.1f" % dx
    log.info("闭环性：指向 +0.10°az → 画面平移 dx=%.2f px（预期 -%.2f）", dx, expected)

    cam.close()
    assert not cam.opened and cam.grab() is None, "close 后 grab 应返回 None"

    # --- build_camera 工厂 ---
    c2 = build_camera({"driver": "simulator", "width": 640, "height": 480,
                       "fps": 20}, get_pointing=None)
    assert isinstance(c2, SimCamera)
    c2.open()
    f = c2.grab()
    assert f is not None and f.image.shape == (480, 640, 3)
    c2.close()
    c3 = build_camera({"driver": "opencv", "source": "0"})
    assert isinstance(c3, OpenCVCamera)
    try:
        build_camera({"driver": "不存在"})
    except ValueError:
        pass
    else:
        raise AssertionError("未知 driver 应抛 ValueError")
    log.info("build_camera：simulator/opencv 分派正常，未知驱动正确报错")

    # --- 设备探测：默认跳过（macOS 会弹摄像头授权框），需要时用环境变量开启 ---
    if os.environ.get("SELFTEST_PROBE") == "1":
        devs = list_video_devices()
        log.info("探测到 %d 个摄像头：%r", len(devs), devs)
    else:
        log.info("跳过实体摄像头探测（设 SELFTEST_PROBE=1 可开启）")

    log.info("camera.py 自检全部通过")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _self_test()
