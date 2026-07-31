"""自动标定：反解「像素位移 → 基座角度」的 2×2 线性映射。

为什么要自动标定而不是让用户填参数：
手工填焦距/像元尺寸最常见的翻车是**符号或转角填错**，跟踪时目标会被
越修越远，而且用户很难自己诊断。让基座实际走四步、看目标在画面里
挪了多少，尺度、相机转角、镜像一次全部拿到，一个数都不用填。
导星软件（PHD2 等）都是这么做的。

矩阵 A 的约定（与 CONTRACT.md 一致）：
    [d_az_sky, d_alt] = A @ [dx, dy]
其中 d_az_sky = d_az_axis * cos(alt) 是**天球角距**的方位分量。
让 A 吃天球角距而不是方位轴角度，是因为这样 A 才是一个纯「旋转+缩放
（+可能镜像）」矩阵，奇异值直接就是角像元尺度，不随仰角变化；
换仰角跟踪时只需在 pixels_to_angles 里除回 cos(alt) 即可。
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 仰角逼近天顶时 cos(alt)→0，方位轴速率会被除法放大到无穷；
# 限幅到 0.05（约 87° 仰角）后放弃精确补偿，换取数值安全。
_MIN_COS_ALT = 0.05

# 单步像素位移低于此值说明基座根本没把目标推动（步长太小、卡死或
# 视场极大），此时最小二乘解全是噪声，必须判失败而不是硬算。
_MIN_STEP_PX = 1.0

# ---- 步长自适应 ----------------------------------------------------------
# 固定步长（配置里那个 0.30°）是**盲的**：它不知道一度对应多少像素。
# 画面是 1280×720 这种宽高比时，横向能容下的位移竖向容不下，于是
# 「+方位 / −方位」两步过关、走到第一个仰角步就把目标顶出画面上下边缘，
# 表现就是每次都卡在「第 3 步（+仰角）后目标丢失」。
# 对策：第 1、2 步测出像元尺度后，后面的步长按画面尺寸反算，
# 只缩不放（配置值永远是上限，用户调小依然有效）。
_STEP_FRAC_OF_FRAME = 0.12     # 单步位移目标值：画面**短边**的 12%
_STEP_FRAC_OF_MARGIN = 0.55    # 且不超过目标到最近边缘距离的 55%
_STEP_MIN_DEG = 0.005          # 步长下限，再小基座的齿隙就淹没信号了
_STEP_RETRIES = 2              # 单步丢目标时自动退回、减半重试的次数
_EDGE_MARGIN_FRAC = 0.08       # 开标前目标离边缘至少要有短边的 8%


@dataclass
class Calibration:
    """标定结果。a11..a22 即映射矩阵 A（像素 → 天球角度，单位度/像素）。"""

    a11: float
    a12: float
    a21: float
    a22: float
    arcsec_per_px: float
    angle_deg: float
    flipped: bool
    rms_px: float
    ok: bool
    note: str = ""

    def pixels_to_angles(self, dx: float, dy: float,
                         alt_deg: float) -> Tuple[float, float]:
        """像素偏差 → (方位轴角度, 仰角角度)，单位度。

        A 输出的是天球角距，方位轴要多转 1/cos(alt) 倍才能在天球上
        走出同样的角距，所以这里把方位分量除回 cos(alt)。
        """
        d_az_sky = self.a11 * dx + self.a12 * dy
        d_alt = self.a21 * dx + self.a22 * dy
        cos_alt = max(_MIN_COS_ALT, math.cos(math.radians(alt_deg)))
        return (d_az_sky / cos_alt, d_alt)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Calibration":
        # 宽容缺字段：旧版本存档少字段时给出安全默认，而不是直接崩。
        return Calibration(
            a11=float(d.get("a11", 0.0)),
            a12=float(d.get("a12", 0.0)),
            a21=float(d.get("a21", 0.0)),
            a22=float(d.get("a22", 0.0)),
            arcsec_per_px=float(d.get("arcsec_per_px", 0.0)),
            angle_deg=float(d.get("angle_deg", 0.0)),
            flipped=bool(d.get("flipped", False)),
            rms_px=float(d.get("rms_px", 0.0)),
            ok=bool(d.get("ok", False)),
            note=str(d.get("note", "")),
        )


def _fail(note: str) -> Calibration:
    """统一的失败出口：契约要求不抛异常，全部信息进 note。"""
    logger.warning("标定失败：%s", note)
    return Calibration(a11=0.0, a12=0.0, a21=0.0, a22=0.0,
                       arcsec_per_px=0.0, angle_deg=0.0, flipped=False,
                       rms_px=0.0, ok=False, note=note)


class Calibrator:
    """自动标定：让基座走已知的几步，看目标在画面里往哪挪了多少，
    反解出 2×2 线性映射 A，使 [d_az_axis, d_alt] = A @ [dx, dy]。

    这样**尺度、相机转角、镜像**一次全部拿到，用户一个数都不用填 ——
    手工填参数最常见的翻车就是符号填反，跟踪时目标越修越远。
    导星软件都是这么标定的。
    """

    def __init__(self, mount, camera, tracker, step_deg: float = 0.30,
                 settle_s: float = 1.2, samples: int = 3):
        self.mount = mount
        self.camera = camera
        self.tracker = tracker
        self.step_deg = float(step_deg)
        self.settle_s = float(settle_s)
        self.samples = max(1, int(samples))
        # 画面尺寸 (高, 宽)：从实际抓到的帧里取，用于把步长换算成安全位移。
        # 抓到帧之前是 None，此时步长退回配置值。
        self._hw: Optional[Tuple[int, int]] = None

    # ---------------- 内部工具 ----------------

    @staticmethod
    def _report(cb: Optional[Callable[[str, float], None]],
                msg: str, frac: float) -> None:
        # 回调是外部代码，它抛异常不应连累标定本身。
        if cb is None:
            return
        try:
            cb(msg, max(0.0, min(1.0, frac)))
        except Exception:
            logger.exception("标定进度回调抛出异常，已忽略")

    def _measure_center(self) -> Optional[np.ndarray]:
        """取多帧平均的目标中心。多帧平均是为了压掉逐帧的质心抖动
        （典型 ±0.5px），它会直接进入最小二乘的右端项。"""
        m = self._measure_center_v()
        return None if m is None else m[0]

    def _measure_center_v(self) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
        """同 _measure_center，另外返回 (采样中点时刻, 目标像素速度)。

        速度来自跟踪器的低通估计（vx/vy），用于漂移补偿：
        被跟的目标（飞机等）自己也在动，标定测到的位移里混着
        「基座造成的」和「目标自己走的」两部分，后者必须扣掉。
        """
        pts: List[Tuple[float, float]] = []
        vels: List[Tuple[float, float]] = []
        ts: List[float] = []
        attempts = 0
        # 允许少量坏帧/低置信帧重试，但不无限等：相机断流时要能退出。
        max_attempts = self.samples * 4 + 4
        while len(pts) < self.samples and attempts < max_attempts:
            attempts += 1
            frame = self.camera.grab()
            if frame is None:
                time.sleep(0.05)
                continue
            self._note_frame(frame)
            try:
                res = self.tracker.update(frame.image)
            except Exception:
                logger.exception("跟踪器在标定采样时抛出异常")
                return None
            center = getattr(res, "center", None)
            if getattr(res, "ok", False) and center is not None:
                pts.append((float(center[0]), float(center[1])))
                vels.append((float(getattr(res, "vx", 0.0)),
                             float(getattr(res, "vy", 0.0))))
                ts.append(time.monotonic())
            else:
                time.sleep(0.02)
        # 过半样本有效才可信；只凑到一两帧说明目标已经快丢了。
        if len(pts) < (self.samples // 2 + 1):
            return None
        return (np.asarray(pts, dtype=float).mean(axis=0),
                float(np.mean(ts)),
                np.asarray(vels, dtype=float).mean(axis=0))

    def _pump_tracker(self, seconds: float) -> None:
        """持续喂帧给跟踪器一段时间。

        标定最初的版本在 nudge + 停稳期间完全不调 tracker.update()，
        恢复采样时目标已经跳了两三百像素（基座位移 + 目标自漂移），
        超出搜索窗直接判丢失 —— 实测第一步就失败。
        让跟踪器在整个移动过程中连续看到画面，每帧位移只有几个像素，
        锁定自然保持。"""
        t_end = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < t_end:
            frame = self.camera.grab()
            if frame is None:
                time.sleep(0.02)
                continue
            self._note_frame(frame)
            try:
                self.tracker.update(frame.image)
            except Exception:
                logger.exception("跟踪器在标定喂帧时抛出异常")
                return

    def _nudge_tracking(self, d_az: float, d_alt: float) -> Tuple[bool, Optional[Exception]]:
        """在后台执行 nudge（会阻塞数秒），前台持续喂帧保持跟踪连续。"""
        out: Dict[str, Any] = {}

        def _w():
            try:
                out["ok"] = bool(self.mount.nudge(d_az, d_alt))
            except Exception as exc:   # noqa: BLE001 —— 契约不抛异常
                out["exc"] = exc

        th = threading.Thread(target=_w, name="calib-nudge", daemon=True)
        th.start()
        while th.is_alive():
            self._pump_tracker(0.06)
        return bool(out.get("ok", False)), out.get("exc")

    def _note_frame(self, frame: Any) -> None:
        """记下画面尺寸。标定全程只会是同一路相机，取到一次就够。"""
        if self._hw is not None:
            return
        img = getattr(frame, "image", None)
        shape = getattr(img, "shape", None)
        if shape is not None and len(shape) >= 2:
            self._hw = (int(shape[0]), int(shape[1]))

    def _safe_disp_px(self, center: np.ndarray) -> float:
        """这一步允许把目标推多远（像素）。

        两条约束取小者：占画面短边的固定比例（保证任何方向都走得回来），
        以及目标当前到最近边缘距离的一半多一点（目标本来就偏一边时收紧）。
        """
        if self._hw is None:
            return float("inf")
        h, w = self._hw
        short = float(min(h, w))
        margin = float(min(center[0], w - center[0], center[1], h - center[1]))
        return max(8.0, min(_STEP_FRAC_OF_FRAME * short,
                            _STEP_FRAC_OF_MARGIN * margin))

    def _choose_step(self, px_per_deg: Optional[float], center: np.ndarray,
                     alt_now: float, axis_is_alt: bool) -> float:
        """把「想要的像素位移」换算成这一轴该走的角度。

        px_per_deg 是**天球角距**的像元尺度（前面步骤实测得来）。方位轴
        要多转 1/cos(alt) 才能在天球上走出同样角距，所以方位步要除回去。
        没有实测尺度时（第 1 步）只能用配置值。
        结果永远不超过配置的 step_deg：自适应只负责往小调，
        用户手工调小 step_deg 仍然完全有效。
        """
        if px_per_deg is None or px_per_deg <= 1e-9:
            return self.step_deg
        want_px = self._safe_disp_px(center)
        if not math.isfinite(want_px):
            return self.step_deg
        deg = want_px / px_per_deg
        if not axis_is_alt:
            deg /= max(_MIN_COS_ALT, math.cos(math.radians(alt_now)))
        return max(_STEP_MIN_DEG, min(self.step_deg, deg))

    def _get_alt(self, default: float = 45.0) -> float:
        try:
            return float(self.mount.get_altaz()[1])
        except Exception:
            # 读不到仰角不至于中断标定：cos(alt) 只影响方位尺度的
            # 百分之几，用上一次/默认值继续，误差进 rms 由用户判断。
            logger.exception("标定中读取仰角失败，使用默认值 %.1f°", default)
            return default

    def _restore(self, applied_az: float, applied_alt: float) -> None:
        """失败退出前尽量把基座送回起点，免得用户的目标跑出视场。
        尽力而为：这里再失败也只记日志，不影响失败结果的返回。"""
        if abs(applied_az) < 1e-9 and abs(applied_alt) < 1e-9:
            return
        try:
            self.mount.nudge(-applied_az, -applied_alt)
        except Exception:
            logger.exception("标定失败后回退基座位置未成功")

    # ---------------- 主流程 ----------------

    def run(self, progress_cb: Optional[Callable[[str, float], None]] = None
            ) -> Calibration:
        """四步：+方位 / −方位 / +仰角 / −仰角，每步取多帧平均。
        最小二乘解 A。progress_cb(消息, 0..1)。
        失败时返回 ok=False 并在 note 里说明原因，不要抛异常。
        """
        try:
            return self._run_inner(progress_cb)
        except Exception as exc:  # 契约：任何情况都不抛异常
            logger.exception("标定过程出现未预期异常")
            return _fail("标定过程出现未预期异常：%s。请检查设备连接后重试。" % exc)

    def _run_inner(self, progress_cb: Optional[Callable[[str, float], None]]
                   ) -> Calibration:
        self._report(progress_cb, "开始标定：测量基准位置", 0.02)

        m = self._measure_center_v()
        if m is None:
            return _fail("标定开始前无法稳定看到目标。"
                         "请确认已框选目标、画面清晰后重试。")
        c_prev, t_prev, v_prev = m
        drift_note = ""

        # 目标贴着画面边缘时，任何方向的步进都可能把它顶出去，
        # 且这时算出来的安全步长会小到没有信噪比 —— 不如直接说清楚。
        if self._hw is not None:
            h0, w0 = self._hw
            margin0 = float(min(c_prev[0], w0 - c_prev[0],
                                c_prev[1], h0 - c_prev[1]))
            if margin0 < _EDGE_MARGIN_FRAC * min(h0, w0):
                return _fail("目标离画面边缘太近（%.0f 像素），标定一动就会把它推出视场。"
                             "请先手动微调基座，把目标大致移到画面中央再标定。"
                             % margin0)

        # 四步依次是 +az / −az / +alt / −alt：
        # 1) 每个轴正反各走一次，正反样本平均可部分抵消齿隙（backlash）；
        # 2) 每对的两步大小相同方向相反，走完基座回到起点，目标不会越标越偏。
        names = ("+方位", "−方位", "+仰角", "−仰角")
        signs = (1.0, -1.0, 1.0, -1.0)
        is_alt = (False, False, True, True)
        # 每对的第一步定步长，第二步照抄（取反），保证成对抵消。
        pair_step = [self.step_deg, self.step_deg]

        px_rows: List[Tuple[float, float]] = []    # (dx, dy)
        ang_rows: List[Tuple[float, float]] = []   # (d_az_sky, d_alt)
        applied_az = 0.0
        applied_alt = 0.0
        # 天球角距的像元尺度（像素/度），由已走完的步实测得到。
        # 相机的像元尺度两轴通用，所以方位步测出的值可以直接给仰角步定步长
        # —— 这正是「方位两步过关、仰角步却把目标顶出画面」的解药。
        px_per_deg: Optional[float] = None
        shrink_note = ""

        for i in range(4):
            name = names[i]
            frac0 = 0.05 + 0.20 * i

            # 步进前读仰角：alt 步会改变 cos(alt)，逐步读取比只读一次准。
            alt_now = self._get_alt()

            if i % 2 == 0:
                pair_step[i // 2] = self._choose_step(
                    px_per_deg, c_prev, alt_now, is_alt[i])
            step_i = pair_step[i // 2]

            attempt = 0
            while True:
                d_az = 0.0 if is_alt[i] else signs[i] * step_i
                d_alt = signs[i] * step_i if is_alt[i] else 0.0
                self._report(progress_cb, "第 %d/4 步：%s %.3f°" %
                             (i + 1, name, step_i), frac0)

                # nudge 在后台跑、前台持续喂帧：跟踪器全程连续看到画面，
                # 每帧位移只有几个像素，不会因一次性大跳变丢失目标。
                moved, exc = self._nudge_tracking(d_az, d_alt)
                if exc is not None:
                    logger.exception("标定第 %d 步 nudge 抛出异常", i + 1)
                    self._restore(applied_az, applied_alt)
                    return _fail("第 %d 步（%s）基座移动出错：%s。"
                                 "请检查基座连接与仰角限位后重试。"
                                 % (i + 1, name, exc))
                if not moved:
                    self._restore(applied_az, applied_alt)
                    return _fail("第 %d 步（%s）基座移动失败（可能超时或触及限位）。"
                                 "请检查基座状态后重试。" % (i + 1, name))
                applied_az += d_az
                applied_alt += d_alt

                # 停稳等待：基座有减速与残振，立刻采样会把机械瞬态当成位移。
                # 等待期间继续喂帧，跟踪不断线。
                if self.settle_s > 0:
                    self._pump_tracker(self.settle_s)

                m = self._measure_center_v()
                if m is not None:
                    break

                # ---- 目标被推出视场/跟丢：退回这一步，减半再来 ----
                # 直接判失败等于把「换个步长」这件纯机械的事推给用户，
                # 而软件手里有算它所需的全部信息。
                attempt += 1
                if attempt > _STEP_RETRIES:
                    self._restore(applied_az, applied_alt)
                    return _fail("第 %d 步（%s）后目标丢失，缩小步长重试 %d 次仍未找回。"
                                 "常见原因是跟踪本身不稳（目标太暗、对比度低、画面拖影），"
                                 "请重新框选目标后再标定。" % (i + 1, name, _STEP_RETRIES))
                back_ok, _back_exc = self._nudge_tracking(-d_az, -d_alt)
                if back_ok:
                    applied_az -= d_az
                    applied_alt -= d_alt
                step_i = max(_STEP_MIN_DEG, step_i * 0.4)
                pair_step[i // 2] = step_i
                self._report(progress_cb,
                             "第 %d/4 步：目标被推出视场，改用 %.3f° 重试"
                             % (i + 1, step_i), frac0)
                shrink_note = ("；标定过程中自动把步长缩到 %.3f°（配置值 %.2f° 对当前"
                               "视场偏大），如果每次标定都触发，直接把 "
                               "calibration.step_deg 改成这个值更省时间"
                               % (step_i, self.step_deg))
                # 退回后目标未必还在原处，基准要重新测。
                m0 = self._measure_center_v()
                if m0 is None:
                    self._restore(applied_az, applied_alt)
                    return _fail("第 %d 步（%s）后目标丢失，退回原位也没能重新找到。"
                                 "请重新框选目标，并把 calibration.step_deg 调小到 "
                                 "%.3f 左右后重试。" % (i + 1, name, step_i))
                c_prev, t_prev, v_prev = m0

            c_new, t_new, v_new = m

            # ---- 漂移补偿：扣掉目标**自己**在这段时间里走过的路 ----
            # 被跟的目标（飞机等）本身在运动，测到的位移 = 基座造成的
            # + 目标自漂移。用前后两次采样的平均速度 ×时间差近似漂移量。
            # 不扣的话，四步的漂移会系统性地偏置最小二乘解，
            # 标出来的矩阵带着一个假旋转分量。
            dt_step = max(0.0, t_new - t_prev)
            drift = 0.5 * (v_prev + v_new) * dt_step
            d_px = (c_new - c_prev) - drift
            drift_mag = float(np.hypot(drift[0], drift[1]))
            if drift_mag > 0.5 * float(np.hypot(d_px[0], d_px[1])):
                # 漂移比信号还大一半以上：结果仍然给，但要让用户知道
                drift_note = ("；注意：目标自身移动较快（漂移 %.0f 像素/步），"
                              "标定精度受影响，建议尽量在目标较慢时标定" % drift_mag)
            t_prev, v_prev = t_new, v_new
            if float(np.hypot(d_px[0], d_px[1])) < _MIN_STEP_PX:
                self._restore(applied_az, applied_alt)
                return _fail("第 %d 步（%s）目标几乎没有移动（%.2f 像素）。"
                             "步长可能太小或基座未实际转动，"
                             "可调大 calibration.step_deg 后重试。"
                             % (i + 1, name, float(np.hypot(d_px[0], d_px[1]))))

            # 方位轴角度换成天球角距，保证 A 的奇异值就是像元角尺度。
            cos_alt = max(_MIN_COS_ALT, math.cos(math.radians(alt_now)))
            px_rows.append((float(d_px[0]), float(d_px[1])))
            ang_rows.append((d_az * cos_alt, d_alt))

            # 实测像元尺度，供后面的步定步长。相机的像元尺度两轴通用，
            # 所以方位步测出来的值对仰角步同样有效。
            disp_px = float(np.hypot(d_px[0], d_px[1]))
            sky_deg = abs(d_alt) if is_alt[i] else abs(d_az) * cos_alt
            if sky_deg > 1e-9:
                obs = disp_px / sky_deg
                px_per_deg = obs if px_per_deg is None else 0.5 * (px_per_deg + obs)

            c_prev = c_new
            self._report(progress_cb, "第 %d/4 步完成：位移 %.1f 像素" %
                         (i + 1, disp_px), frac0 + 0.18)

        # 四步成对抵消，理论上已回到起点；中途重试改过步长则会有残留，补回去。
        self._restore(applied_az, applied_alt)

        self._report(progress_cb, "最小二乘求解映射矩阵", 0.90)

        P = np.asarray(px_rows, dtype=float)    # 4×2，每行 (dx, dy)
        Q = np.asarray(ang_rows, dtype=float)   # 4×2，每行 (d_az_sky, d_alt)

        # Q = P @ A^T，lstsq 直接解出 A^T（超定：4 个方程解 4 个未知数×?
        # 实际是 4 样本 × 2 维 = 8 个标量方程解 4 个未知数）。
        try:
            At, _residuals, rank, _sv = np.linalg.lstsq(P, Q, rcond=None)
        except np.linalg.LinAlgError as exc:
            return _fail("最小二乘求解失败：%s。请重试标定。" % exc)
        if rank < 2:
            return _fail("四步的像素位移近乎共线，无法解出二维映射。"
                         "请确认基座两根轴都在动，或调大步长后重试。")
        A = At.T

        det = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
        if abs(det) < 1e-16:
            return _fail("解出的映射矩阵奇异（行列式为 0），结果不可用。"
                         "请检查目标跟踪是否稳定后重试。")

        # 奇异值 = A 的两个方向缩放，即角像元尺度（度/像素）。
        # 相机像元基本是方形，两个奇异值应接近，取平均作标称值。
        svals = np.linalg.svd(A, compute_uv=False)
        arcsec_per_px = float(svals.mean() * 3600.0)

        # A ≈ s·R(θ)（flipped 时右乘 diag(1,-1)），两种情况下都有
        # a11 = s·cosθ、a21 = s·sinθ，所以统一用 atan2(a21, a11) 取转角。
        angle_deg = float(math.degrees(math.atan2(A[1, 0], A[0, 0])))
        flipped = det < 0.0

        # 重投影残差：把角度样本经 A⁻¹ 投回像素域，与实测像素位移求 RMS。
        # 在像素域评估是因为用户对「几个像素」有直觉，对「几毫度」没有。
        A_inv = np.linalg.inv(A)
        px_pred = Q @ A_inv.T
        res = P - px_pred
        rms_px = float(math.sqrt(float(np.mean(np.sum(res * res, axis=1)))))

        note = ""
        if svals.min() > 0 and float(svals.max() / svals.min()) > 1.3:
            # 各向异性明显通常意味着某一步受齿隙/抖动污染，提醒但不判死。
            note = ("两轴像素尺度差异较大（%.2f×），标定可能受齿隙或抖动影响，"
                    "建议重新标定确认。" % float(svals.max() / svals.min()))
        if drift_note:
            note = (note + drift_note) if note else drift_note.lstrip("；")
        if shrink_note:
            note = (note + shrink_note) if note else shrink_note.lstrip("；")

        cal = Calibration(
            a11=float(A[0, 0]), a12=float(A[0, 1]),
            a21=float(A[1, 0]), a22=float(A[1, 1]),
            arcsec_per_px=arcsec_per_px,
            angle_deg=angle_deg,
            flipped=flipped,
            rms_px=rms_px,
            ok=True,
            note=note,
        )
        logger.info("标定完成：%.2f 角秒/像素，相机转角 %.1f°，镜像=%s，"
                    "残差 %.2f 像素", arcsec_per_px, angle_deg, flipped, rms_px)
        self._report(progress_cb, "标定完成：%.2f 角秒/像素，残差 %.2f 像素" %
                     (arcsec_per_px, rms_px), 1.0)
        return cal


# ======================================================================
# 自检：用已知真值矩阵造一套假设备，验证 run() 能把矩阵解回来。
# ======================================================================

if __name__ == "__main__":
    import types

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("calibration.selftest")

    rng = np.random.default_rng(42)

    # ---- 真值：6 角秒/像素，相机转 30°，且镜像（det<0）----
    s = 6.0 / 3600.0  # 度/像素
    th = math.radians(30.0)
    R = np.array([[math.cos(th), -math.sin(th)],
                  [math.sin(th), math.cos(th)]])
    F = np.diag([1.0, -1.0])
    A_true = s * (R @ F)          # 像素 → 天球角度
    A_true_inv = np.linalg.inv(A_true)

    ALT = 35.0

    class _FakeMount:
        """只维护「累计天球角位移」：nudge 的方位轴角度按 cos(alt) 折算。"""

        def __init__(self):
            self.sky = np.zeros(2)
            self.fail_nudge = False

        def get_altaz(self):
            return (180.0, ALT)

        def nudge(self, d_az, d_alt, timeout_s=25.0):
            if self.fail_nudge:
                return False
            cos_alt = math.cos(math.radians(ALT))
            self.sky = self.sky + np.array([d_az * cos_alt, d_alt])
            return True

    # 画面尺寸必须是真实的 1280×720：步长自适应要按它算安全位移，
    # 假相机给个 4×4 就测不出「宽方向容得下、高方向容不下」这件事。
    FRAME_W, FRAME_H = 1280, 720

    class _FakeCamera:
        def grab(self):
            return types.SimpleNamespace(
                image=np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8),
                ts=time.time(), index=0)

    class _FakeTracker:
        """目标像素位置 = 基准 + A_true⁻¹ @ 累计天球位移 + 0.5px 高斯噪声。"""

        def __init__(self, mount, lost=False):
            self.mount = mount
            self.lost = lost
            self.c0 = np.array([640.0, 360.0])

        def update(self, image):
            if self.lost:
                return types.SimpleNamespace(ok=False, center=None)
            c = self.c0 + A_true_inv @ self.mount.sky + rng.normal(0.0, 0.5, 2)
            return types.SimpleNamespace(ok=True,
                                         center=(float(c[0]), float(c[1])))

    # ---- 正常路径 ----
    mount = _FakeMount()
    cal = Calibrator(mount, _FakeCamera(), _FakeTracker(mount),
                     step_deg=0.30, settle_s=0.0, samples=3)

    progress_log = []
    result = cal.run(progress_cb=lambda m, f: progress_log.append((m, f)))

    assert result.ok, "正常路径标定应当成功，note=%s" % result.note

    A_est = np.array([[result.a11, result.a12],
                      [result.a21, result.a22]])
    rel_err = float(np.abs(A_est - A_true).max() / np.abs(A_true).max())
    log.info("恢复矩阵最大元素相对误差 %.2f%%", rel_err * 100.0)
    assert rel_err < 0.05, "矩阵元素误差应 <5%%，实际 %.2f%%" % (rel_err * 100)
    assert result.flipped is True, "应检测出镜像（det<0）"
    assert abs(result.angle_deg - 30.0) < 2.0, \
        "相机转角应约 30°，实际 %.2f°" % result.angle_deg
    assert abs(result.arcsec_per_px - 6.0) / 6.0 < 0.05, \
        "像元尺度应约 6″/px，实际 %.3f" % result.arcsec_per_px
    assert result.rms_px < 2.0, "残差应在噪声量级，实际 %.2f px" % result.rms_px
    assert progress_log and abs(progress_log[-1][1] - 1.0) < 1e-9, \
        "进度回调应收尾于 1.0"

    # ---- 反变换一致性：pixels_to_angles 应还原方位轴角度 ----
    # 构造一个纯方位轴位移对应的像素向量，检查除 cos(alt) 是否正确。
    d_ax = 0.1  # 方位轴度
    px = A_true_inv @ np.array([d_ax * math.cos(math.radians(ALT)), 0.0])
    got_az, got_alt = result.pixels_to_angles(px[0], px[1], ALT)
    assert abs(got_az - d_ax) < 0.01 * d_ax + 1e-4, \
        "pixels_to_angles 方位轴分量应除回 cos(alt)"
    assert abs(got_alt) < 1e-3

    # ---- 序列化往返 ----
    back = Calibration.from_dict(result.to_dict())
    assert back.to_dict() == result.to_dict(), "to_dict/from_dict 应无损往返"

    # ---- 失败路径：目标丢失 ----
    mount2 = _FakeMount()
    bad = Calibrator(mount2, _FakeCamera(), _FakeTracker(mount2, lost=True),
                     step_deg=0.30, settle_s=0.0, samples=3).run()
    assert (not bad.ok) and bad.note, "目标丢失应返回 ok=False 且 note 非空"

    # ---- 失败路径：基座步进失败 ----
    mount3 = _FakeMount()
    mount3.fail_nudge = True
    bad2 = Calibrator(mount3, _FakeCamera(), _FakeTracker(mount3),
                      step_deg=0.30, settle_s=0.0, samples=3).run()
    assert (not bad2.ok) and "移动失败" in bad2.note, \
        "步进失败应返回 ok=False 且 note 说明原因"

    # ---- 回归：步长对竖向视场过大时必须自己缩，而不是死在第 3 步 ----
    # 复现现场故障：1280×720 的画面，配置步长在方位（宽 1280）方向撑得住，
    # 到「+仰角」就把目标顶出上下边缘 → 每次都报「第 3 步（+仰角）后目标丢失」。
    class _ClippingTracker(_FakeTracker):
        """出画面就判丢失 —— 真跟踪器的行为。"""

        def update(self, image):
            res = _FakeTracker.update(self, image)
            if not res.ok:
                return res
            x, y = res.center
            if not (0.0 <= x < FRAME_W and 0.0 <= y < FRAME_H):
                return types.SimpleNamespace(ok=False, center=None)
            return res

    mount4 = _FakeMount()
    # 0.9°/步：横向 ≈ 442 像素（1280 宽扛得住），纵向会冲出 720 的画面。
    cal4 = Calibrator(mount4, _FakeCamera(), _ClippingTracker(mount4),
                      step_deg=0.90, settle_s=0.0, samples=3)
    steps4 = []
    r4 = cal4.run(progress_cb=lambda m, f: steps4.append(m))
    assert r4.ok, "步长过大时应自动缩小并标定成功，而不是失败：%s" % r4.note
    assert abs(r4.arcsec_per_px - 6.0) / 6.0 < 0.05, \
        "自适应步长下尺度仍应约 6″/px，实际 %.3f" % r4.arcsec_per_px
    assert any("+仰角" in s for s in steps4), "应当真的走到过仰角步"
    log.info("步长自适应回归通过：配置 0.90°，实际标定成功，尺度 %.2f\"/px",
             r4.arcsec_per_px)

    # ---- 边缘保护：目标贴着画面边时直接说清楚，别一动就丢 ----
    mount5 = _FakeMount()
    tr5 = _FakeTracker(mount5)
    tr5.c0 = np.array([8.0, 360.0])          # 紧贴左边缘
    edge = Calibrator(mount5, _FakeCamera(), tr5,
                      step_deg=0.30, settle_s=0.0, samples=3).run()
    assert (not edge.ok) and "边缘" in edge.note, \
        "目标贴边应提前拒绝并说明原因，实际 note=%s" % edge.note

    log.info("标定自检全部通过：A 最大元素误差 %.2f%%，角度 %.2f°，"
             "尺度 %.3f\"/px，flipped=%s，rms=%.2fpx",
             rel_err * 100.0, result.angle_deg, result.arcsec_per_px,
             result.flipped, result.rms_px)
    log.info("calibration selftest OK")
