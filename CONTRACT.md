# 模块接口契约

各模块按此实现，互不依赖对方内部细节。**签名不得改动**，否则集成会崩。
Python 3.9 兼容：文件首行加 `from __future__ import annotations`，
不要用 `X | Y` 类型语法、不要用 `match`。全部注释与用户可见文案用中文。

---

## core/mount.py

```python
class MountError(RuntimeError): ...

class MountBase:
    max_rate_deg_s: float                 # 机械最大转速（度/秒）

    def connect(self) -> None
    def close(self) -> None
    @property
    def connected(self) -> bool
    def get_altaz(self) -> Tuple[float, float]     # (方位 0..360, 仰角 -90..90)
    def set_rate(self, az_dps: float, alt_dps: float) -> None
    def stop(self) -> None
    def is_slewing(self) -> bool
    def nudge(self, d_az: float, d_alt: float, timeout_s: float = 25.0) -> bool
        """相对当前指向移动指定角度（度），阻塞到停稳。标定专用。
           返回是否成功。d_az 是**方位轴**角度，不是天球角距。"""
    def info(self) -> Dict[str, Any]
        """{"driver":str,"version":str,"aligned":bool,"tracking_mode":int,"port":str}
           取不到的字段给 None，不要抛异常。"""

class NexStarMount(MountBase):
    def __init__(self, port: str, baudrate: int = 9600, timeout_s: float = 3.5,
                 max_rate_deg_s: float = 2.5, min_rate_arcsec_s: float = 2.0,
                 alt_limit_min_deg: float = 5.0, alt_limit_max_deg: float = 88.0)

class SimMount(MountBase):
    def __init__(self, max_rate_deg_s: float = 2.5, min_rate_arcsec_s: float = 2.0,
                 accel_deg_s2: float = 6.0, backlash_deg: float = 0.02,
                 alt_limit_min_deg: float = 5.0, alt_limit_max_deg: float = 88.0,
                 start_az: float = 180.0, start_alt: float = 35.0)

def build_mount(cfg: Dict[str, Any]) -> MountBase
    """cfg 为 config.yaml 的 mount 段；driver 取值 nexstar / simulator。"""
```

**NexStar 协议要点**（LCM90 手控器底部 RJ-22，9600 8N1，应答以 `#` 结尾）：
- `z` 高精度地平坐标 → `"XXXXXXXX,YYYYYYYY#"`，值 = 角度/360×2³²
- 变速率：`P`(0x50) + `3,16|17, 6|7, rateHi, rateLo, 0, 0`，
  16=方位轴 17=俯仰轴，6=正向 7=反向，rate = |度/秒|×3600×4
- `b` + `"XXXXXXXX,YYYYYYYY"` 高精度 GoTo；`L` 查询是否在转；`M` 取消
- `K<c>` 回显自检；`V` 版本；`J` 是否已校准；`t` 跟踪模式
- 仰角回报 0..360，>180 表示负仰角

---

## core/camera.py

```python
@dataclass
class Frame:
    image: np.ndarray        # BGR uint8
    ts: float
    index: int

class CameraBase:
    def open(self) -> None
    def close(self) -> None
    def grab(self) -> Optional[Frame]
    @property
    def opened(self) -> bool
    def info(self) -> Dict[str, Any]    # {"backend","width","height","fps","source"}

class OpenCVCamera(CameraBase):
    def __init__(self, source, width: int = 1280, height: int = 720,
                 fps: float = 30.0)

class SimCamera(CameraBase):
    """合成场景，无硬件时演示用。星空 + 一个可被框选的运动目标。
       画面随 get_pointing() 变化，形成真闭环。"""
    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30.0,
                 arcsec_per_px: float = 6.0,
                 get_pointing: Optional[Callable[[], Tuple[float, float]]] = None,
                 target_rate_deg_s: float = 0.12, target_heading_deg: float = 40.0)

def list_video_devices(max_index: int = 6) -> List[Dict[str, Any]]
    """探测可用摄像头，返回 [{"index":int,"name":str,"width":int,"height":int}]。
       探测失败的设备跳过，不要抛异常。"""

def build_camera(cfg: Dict[str, Any],
                 get_pointing: Optional[Callable] = None) -> CameraBase
    """cfg 为 config.yaml 的 camera 段；driver 取值 opencv / simulator。"""
```

---

## core/tracker.py

```python
@dataclass
class TrackResult:
    ok: bool
    box: Optional[Tuple[float, float, float, float]]   # x, y, w, h
    center: Optional[Tuple[float, float]]
    score: float                 # 0..1 置信度
    state: str                   # "idle" | "tracking" | "coasting" | "lost"
    lost_frames: int
    vx: float = 0.0              # 像素速度（px/s），已低通
    vy: float = 0.0

class TargetTracker:
    """跟一个**用户手动框选**的任意目标 —— 不是点源检测。

    核心是相关滤波（CSRT/KCF/MOSSE）。这类跟踪器会缓慢漂移、
    遇到遮挡或快速运动会失败，所以必须有：
      1. 置信度评估（用归一化互相关 NCC 对模板打分）
      2. 失败后在放大的搜索窗内做模板匹配重捕
      3. 连续失败超过 max_lost 帧才判定丢失
    """
    def __init__(self, algo: str = "csrt", max_lost: int = 25,
                 min_score: float = 0.30, search_scale: float = 2.5,
                 template_alpha: float = 0.08)
    def start(self, frame_bgr: np.ndarray,
              box: Tuple[float, float, float, float]) -> bool
    def update(self, frame_bgr: np.ndarray) -> TrackResult
    def reset(self) -> None
    @property
    def active(self) -> bool
    def template(self) -> Optional[np.ndarray]      # 当前模板，UI 缩略图用

def available_algos() -> List[str]
    """当前 OpenCV 实际可用的算法名，按推荐顺序。至少要能返回 ["template"]。"""
```

要求：
- OpenCV 4.5 前后 `TrackerCSRT_create` 在 `cv2` 和 `cv2.legacy` 都试一遍
- 都不可用时降级为纯模板匹配，**不要抛异常**
- 模板做缓慢在线更新（`template_alpha`），但只在高置信度时更新，避免漂移累积
- `update` 每帧必须返回结果，不得抛异常

---

## core/calibration.py

```python
@dataclass
class Calibration:
    a11: float; a12: float; a21: float; a22: float
    arcsec_per_px: float
    angle_deg: float
    flipped: bool
    rms_px: float
    ok: bool
    note: str = ""

    def pixels_to_angles(self, dx: float, dy: float,
                         alt_deg: float) -> Tuple[float, float]
        """像素偏差 → (方位轴角度, 仰角角度)，单位度。
           方位轴分量已经除过 cos(仰角)。"""
    def to_dict(self) -> Dict[str, Any]
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Calibration"

class Calibrator:
    """自动标定：让基座走已知的几步，看目标在画面里往哪挪了多少，
    反解出 2×2 线性映射 A，使 [d_az_axis, d_alt] = A @ [dx, dy]。

    这样**尺度、相机转角、镜像**一次全部拿到，用户一个数都不用填 ——
    手工填参数最常见的翻车就是符号填反，跟踪时目标越修越远。
    导星软件都是这么标定的。
    """
    def __init__(self, mount, camera, tracker, step_deg: float = 0.30,
                 settle_s: float = 1.2, samples: int = 3)
    def run(self, progress_cb: Optional[Callable[[str, float], None]] = None
            ) -> Calibration
        """四步：+方位 / −方位 / +仰角 / −仰角，每步取多帧平均。
           最小二乘解 A。progress_cb(消息, 0..1)。
           失败时返回 ok=False 并在 note 里说明原因，不要抛异常。"""
```

数学：设基座移动 (Δaz_axis, Δalt)，目标在画面里位移 (Δx, Δy)。
天球角距的方位分量是 Δaz_axis·cos(alt)。用四步的 (Δx,Δy)→(Δaz,Δalt)
组成超定方程，最小二乘求 A。`rms_px` 是残差，用来判断标定是否可信。

---

## core/control.py

```python
class PID:
    def __init__(self, kp, ki, kd, i_limit=0.5, i_band=0.5, out_limit=2.5)
    def reset(self) -> None
    def update(self, err: float, dt: float, feedforward: float = 0.0) -> float
        """积分分离（|err|>i_band 不积分）+ 条件积分抗饱和 + 前馈。
           前馈必须参与饱和判断，否则会积分饱和。"""

@dataclass
class Telemetry:
    state: str            # idle/selecting/calibrating/tracking/lost/error
    message: str
    az: float; alt: float
    rate_az: float; rate_alt: float
    err_px: Optional[float]
    err_arcsec: Optional[float]
    locked: bool
    score: float
    fps: float
    loop_hz: float
    lost_frames: int
    calibrated: bool
    def to_dict(self) -> Dict[str, Any]

class TrackingSession:
    """把相机 + 跟踪器 + 标定 + 基座 + PID 串成一个状态机。

    线程：采集线程（取图 + 跟踪）+ 控制线程（定频下发速率）。
    分开是因为两者速率不同，且视觉耗时不该阻塞控制回路的定时。
    """
    def __init__(self, cfg: Dict[str, Any])
    # 设备
    def connect_mount(self, port: Optional[str] = None) -> Dict[str, Any]
    def disconnect_mount(self) -> None
    def open_camera(self, source: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
    def close_camera(self) -> None
    # 流程
    def select_target(self, box: Tuple[float, float, float, float]) -> Dict[str, Any]
        """box 为**归一化**坐标 (x,y,w,h)，取值 0..1 —— 前端不用关心分辨率。"""
    def clear_target(self) -> None
    def calibrate(self) -> Dict[str, Any]        # 异步启动，进度看 status()
    def start_tracking(self) -> Dict[str, Any]
    def stop_tracking(self) -> None
    def nudge(self, d_az: float, d_alt: float) -> Dict[str, Any]   # 手动微调
    def shutdown(self) -> None
    # 输出
    def status(self) -> Dict[str, Any]
    def annotated_frame(self) -> Optional[np.ndarray]
        """画上跟踪框、十字丝、锁定环、搜索窗的一帧，供 MJPEG 推流。"""
```

状态机允许的迁移：
```
idle ──open_camera──> idle(有画面)
     ──select_target──> selecting(已选中，未标定)
     ──calibrate──> calibrating ──> selecting(已标定)
     ──start_tracking──> tracking ⇄ lost ──> idle
```
未标定就 `start_tracking` 要拒绝并给出明确提示（会跟反方向）。

---

## config.yaml 结构

```yaml
mount:
  driver: simulator          # nexstar | simulator
  port: ""                   # 空则由界面扫描选择
  baudrate: 9600
  max_rate_deg_s: 2.5
  min_rate_arcsec_s: 2.0
  alt_limit_min_deg: 5.0
  alt_limit_max_deg: 88.0

camera:
  driver: simulator          # opencv | simulator
  source: 0
  width: 1280
  height: 720
  fps: 30
  preset: eyepiece           # eyepiece | external —— 只影响默认标定步长与提示

optics:
  eyepiece: {focal_length_mm: 660.0, note: "相机直接装在 LCM90 调焦座上"}
  external: {focal_length_mm: 25.0,  note: "并联的广角镜头/监控镜头"}

tracker:
  algo: csrt
  max_lost: 25
  min_score: 0.30
  search_scale: 2.5

control:
  loop_hz: 15
  pid: {kp: 0.9, ki: 0.06, kd: 0.05, i_limit: 0.5, i_band: 0.4}
  lock_radius_px: 30
  deadband_px: 2

calibration:
  step_deg: 0.30
  settle_s: 1.2
  samples: 3
  file: data/calibration.json

server: {host: "127.0.0.1", port: 8300}
```

---

## 通用要求

- 每个模块自带 `if __name__ == "__main__":` 的自检，能单独跑
- 不要 print 调试信息，用 `logging`
- 注释解释**为什么**，不是复述代码在做什么
- 面向用户的报错要说清楚「出了什么问题 + 怎么办」
