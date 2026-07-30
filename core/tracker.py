# -*- coding: utf-8 -*-
"""任意目标视觉跟踪器（用户手动框选，非点源检测）。

核心是 OpenCV 的相关滤波跟踪器（CSRT/KCF/MOSSE）。这类算法的通病是：
缓慢漂移、遮挡即失败、且失败时经常"自称成功"。所以本模块在其外面
包了三层保险：

  1. 置信度评估 —— 每帧用 NCC（归一化互相关）对当前模板打分，
     相关滤波器说"成功"但分数太低时按漂移处理；
  2. 重捕 —— 失败后在放大的搜索窗内做模板匹配，找回目标后
     重建跟踪器；
  3. 惯性外推（coasting）—— 短暂失败时按低通后的像素速度外推
     中心位置，给控制回路一个可用的前馈量，避免基座急停。

OpenCV 不提供任何相关滤波器时自动降级为纯 NCC 模板匹配 + 局部
搜索窗，保证在任何装机环境下都能工作，绝不抛异常。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 策略常量。集中放在这里而不是散在代码里，调参时不用满文件找。
# ---------------------------------------------------------------------------
_LP_ALPHA = 0.4            # 像素速度一阶低通系数：新样本权重，越大越跟手越噪
_COAST_MAX_S = 0.5         # 短暂失败时按速度外推的最长时长（秒），再长就不可信了
_TEMPLATE_UPDATE_MIN = 0.6  # 置信度高于此才在线融合模板 —— 低分帧融合会把背景学进去
_LOCAL_SEARCH_SCALE = 2.0   # template 模式正常逐帧跟踪的局部搜索窗倍数（相对框大小）
_MIN_BOX_PX = 8             # 框边长下限：再小纹理信息不足，NCC 分数完全不可靠

# 契约要求的三种相关滤波算法在不同 OpenCV 版本里的工厂函数名。
# 4.5 前在 cv2 顶层，4.5 后被挪进 cv2.legacy，两个命名空间都要试。
_CV_FACTORY_NAMES = {
    "csrt": "TrackerCSRT_create",
    "kcf": "TrackerKCF_create",
    "mosse": "TrackerMOSSE_create",
}


def _create_cv_tracker(algo: str) -> Optional[Any]:
    """尝试创建指定的相关滤波跟踪器；任何失败都返回 None 而不是抛异常，
    由调用方决定是否降级为纯模板匹配。"""
    name = _CV_FACTORY_NAMES.get(algo)
    if name is None:
        return None
    namespaces = [cv2]
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None:
        namespaces.append(legacy)
    for ns in namespaces:
        factory = getattr(ns, name, None)
        if factory is None:
            continue
        try:
            return factory()
        except Exception:
            # 有些发行版留了名字但构造会崩（缺 contrib 编译），继续试下一个命名空间
            logger.debug("跟踪器 %s 构造失败，换下一个命名空间再试", algo, exc_info=True)
    return None


def available_algos() -> List[str]:
    """当前 OpenCV 实际可用的算法名，按推荐顺序（CSRT 最稳、MOSSE 最快）。
    template 是纯模板匹配保底，永远可用，因此列表至少有一项。"""
    out: List[str] = []
    for algo in ("csrt", "kcf", "mosse"):
        # 真的创建一个实例来验证，而不是只查属性 —— 有属性但构造崩的情况见过
        if _create_cv_tracker(algo) is not None:
            out.append(algo)
    out.append("template")
    return out


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

    状态机：
        idle ──start──> tracking ⇄ coasting ──连续 max_lost 帧失败──> lost
        lost 状态仍持续尝试重捕，找回即恢复 tracking；reset 回 idle。
    """

    def __init__(self, algo: str = "csrt", max_lost: int = 25,
                 min_score: float = 0.30, search_scale: float = 2.5,
                 template_alpha: float = 0.08):
        algo = (algo or "").strip().lower()
        if algo not in ("csrt", "kcf", "mosse", "template"):
            # 配置写错不该让软件跑不起来：退到默认链，不可用时还会再降级
            logger.warning("未知跟踪算法 %r，改用 csrt（不可用时自动降级 template）", algo)
            algo = "csrt"
        self._algo_req = algo          # 用户要求的算法
        self._algo = algo              # 实际在用的算法（可能已降级 template）
        self._max_lost = int(max_lost)
        self._min_score = float(min_score)
        self._search_scale = float(search_scale)
        self._template_alpha = float(template_alpha)
        # 重捕门槛比逐帧门槛更高：重捕是"重新拍板"，拍错了会把跟踪器
        # 锁到假目标上，比多丢几帧的代价大得多
        self._reacquire_score = max(0.5, self._min_score + 0.15)

        self._cv = None                              # 相关滤波器实例
        self._template: Optional[np.ndarray] = None  # 灰度 float32 模板
        self._box: Optional[Tuple[float, float, float, float]] = None
        self._center: Optional[Tuple[float, float]] = None
        self._state = "idle"
        self._started = False
        self._lost = 0
        self._vx = 0.0
        self._vy = 0.0
        self._prev_center: Optional[Tuple[float, float]] = None
        self._prev_ts = 0.0            # 上一次**真实测量**成功的时刻（速度差分用）
        self._last_ok_ts = 0.0         # 上一次成功时刻（coasting 时长判断用）
        self._last_ts = 0.0            # 上一帧 update 时刻（外推步长用）

    # ------------------------------------------------------------------ 对外接口

    def start(self, frame_bgr: np.ndarray,
              box: Tuple[float, float, float, float]) -> bool:
        try:
            gray = self._to_gray(frame_bgr)
            h_img, w_img = gray.shape[:2]
            x = int(round(box[0]))
            y = int(round(box[1]))
            w = int(round(box[2]))
            h = int(round(box[3]))
            # 框可能被用户拖出画面边缘，先裁回来再检查尺寸
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            if w < _MIN_BOX_PX or h < _MIN_BOX_PX:
                logger.error("选框太小（%dx%d px），纹理不足无法跟踪；请框大一点", w, h)
                return False
            template = gray[y:y + h, x:x + w].astype(np.float32)
            if float(template.std()) < 1e-6:
                logger.error("选框内是完全平坦的区域，无法作为跟踪模板；请框住有纹理的目标")
                return False
            if float(template.std()) < 3.0:
                # 不拒绝，但要提醒：弱纹理目标 NCC 区分度差，容易误捕
                logger.warning("选框内纹理很弱（std=%.1f），跟踪可能不稳", template.std())

            self._template = template
            self._box = (float(x), float(y), float(w), float(h))
            self._center = (x + w / 2.0, y + h / 2.0)

            # 创建相关滤波器；创建或初始化失败都静默降级为模板匹配，不抛异常
            self._algo = self._algo_req
            self._cv = None
            if self._algo != "template":
                self._cv = _create_cv_tracker(self._algo)
                if self._cv is None:
                    logger.warning("当前 OpenCV 不提供 %s 跟踪器，已降级为纯模板匹配", self._algo_req)
                    self._algo = "template"
                else:
                    try:
                        self._cv.init(frame_bgr, (x, y, w, h))
                    except Exception:
                        logger.warning("%s 跟踪器初始化失败，已降级为纯模板匹配",
                                       self._algo_req, exc_info=True)
                        self._cv = None
                        self._algo = "template"

            now = time.monotonic()
            self._started = True
            self._state = "tracking"
            self._lost = 0
            self._vx = 0.0
            self._vy = 0.0
            self._prev_center = self._center
            self._prev_ts = now
            self._last_ok_ts = now
            self._last_ts = now
            logger.info("开始跟踪：算法=%s 框=(%d,%d,%d,%d)", self._algo, x, y, w, h)
            return True
        except Exception:
            # start 也不许把异常抛给上层：框选失败让用户重框即可
            logger.exception("目标框选初始化失败")
            self.reset()
            return False

    def update(self, frame_bgr: np.ndarray) -> TrackResult:
        """每帧必须返回结果，绝不抛异常 —— 控制回路靠这个保证不被打断。"""
        try:
            return self._update_inner(frame_bgr)
        except Exception:
            logger.exception("跟踪更新出现内部异常，本帧按失败处理")
            if self._started:
                self._lost += 1
                if self._lost >= self._max_lost:
                    self._state = "lost"
            return self._failure_result(0.0)

    def reset(self) -> None:
        self._cv = None
        self._template = None
        self._box = None
        self._center = None
        self._state = "idle"
        self._started = False
        self._lost = 0
        self._vx = 0.0
        self._vy = 0.0
        self._prev_center = None

    @property
    def active(self) -> bool:
        return self._started

    def template(self) -> Optional[np.ndarray]:
        """当前模板（uint8 灰度），UI 缩略图用；未开始跟踪时返回 None。"""
        if self._template is None:
            return None
        return np.clip(self._template, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ 内部实现

    def _update_inner(self, frame_bgr: np.ndarray) -> TrackResult:
        if not self._started:
            return TrackResult(ok=False, box=None, center=None, score=0.0,
                               state="idle", lost_frames=0)
        gray = self._to_gray(frame_bgr)
        h_img, w_img = gray.shape[:2]
        now = time.monotonic()
        # dt 加下限：连续两帧间隔理论上可能小到计时器分辨率以下，
        # 直接除会把速度打爆
        dt = max(now - self._last_ts, 1e-4)
        self._last_ts = now

        ok = False
        score = 0.0
        box: Optional[Tuple[float, float, float, float]] = None

        # ---- 第一层：主跟踪器 ----
        if self._cv is not None:
            try:
                cv_ok, cv_box = self._cv.update(frame_bgr)
            except Exception:
                # 个别版本的滤波器在目标出画面时会直接抛异常，等同失败
                logger.debug("相关滤波器 update 异常", exc_info=True)
                cv_ok, cv_box = False, None
            if cv_ok and cv_box is not None:
                cand = tuple(float(v) for v in cv_box)
                if self._box_plausible(cand, w_img, h_img):
                    s = self._score_at(gray, cand)
                    # 关键防漂移逻辑：滤波器自称成功但和模板对不上 →
                    # 不信它，落到下面的重捕分支
                    if s >= self._min_score:
                        ok, score, box = True, s, cand
        else:
            # template 模式：在当前中心附近的局部窗内做 NCC 搜索。
            # 窗口比重捕窗小，为的是快、且减少背景里相似纹理的干扰
            s, cand = self._match_in_window(gray, self._center, _LOCAL_SEARCH_SCALE)
            if cand is not None and s >= self._min_score:
                ok, score, box = True, s, cand

        # ---- 第二层：重捕 ----
        if not ok:
            s, cand = self._match_in_window(gray, self._center, self._search_scale)
            if cand is not None and s >= self._reacquire_score:
                ok, score, box = True, s, cand
                # 旧滤波器实例已经跟丢，内部状态不可复用，必须重建
                self._reinit_cv(frame_bgr, cand)
                if self._lost > 0:
                    logger.info("目标重捕成功（丢失 %d 帧后），score=%.2f", self._lost, s)

        if ok and box is not None:
            return self._on_success(gray, box, score, now)
        return self._on_failure(score, now, dt, w_img, h_img)

    def _on_success(self, gray: np.ndarray,
                    box: Tuple[float, float, float, float],
                    score: float, now: float) -> TrackResult:
        cx = box[0] + box[2] / 2.0
        cy = box[1] + box[3] / 2.0
        # 速度用相邻两次**真实测量**差分（跳过 coasting 帧），再一阶低通。
        # 低通是给控制回路前馈用的：生差分噪声太大，直接喂会抖
        if self._prev_center is not None:
            mdt = max(now - self._prev_ts, 1e-4)
            raw_vx = (cx - self._prev_center[0]) / mdt
            raw_vy = (cy - self._prev_center[1]) / mdt
            self._vx = _LP_ALPHA * raw_vx + (1.0 - _LP_ALPHA) * self._vx
            self._vy = _LP_ALPHA * raw_vy + (1.0 - _LP_ALPHA) * self._vy
        self._prev_center = (cx, cy)
        self._prev_ts = now
        self._box = box
        self._center = (cx, cy)
        self._lost = 0
        self._state = "tracking"
        self._last_ok_ts = now

        # 模板在线更新：只在高置信度时小步融合。目的是适应光照/形态的
        # 缓慢变化，同时靠高门槛 + 小 alpha 防止漂移逐帧累积进模板
        if score > _TEMPLATE_UPDATE_MIN and self._template is not None:
            patch = self._extract_patch(gray, box)
            if patch is not None:
                a = self._template_alpha
                self._template = (1.0 - a) * self._template + a * patch

        return TrackResult(ok=True, box=self._box, center=self._center,
                           score=score, state="tracking", lost_frames=0,
                           vx=self._vx, vy=self._vy)

    def _on_failure(self, score: float, now: float, dt: float,
                    w_img: int, h_img: int) -> TrackResult:
        self._lost += 1
        if self._lost >= self._max_lost:
            if self._state != "lost":
                logger.warning("目标连续 %d 帧未命中，判定丢失；请重新框选或等待自动重捕",
                               self._lost)
            self._state = "lost"
            # 丢失后不再报位置：给控制回路错的位置比不给更危险
            return TrackResult(ok=False, box=None, center=None, score=score,
                               state="lost", lost_frames=self._lost,
                               vx=self._vx, vy=self._vy)

        # coasting：距最近一次成功不超过 0.5 秒时按速度外推中心，
        # 让控制回路能继续前馈；超时后冻结位置，只等重捕，
        # 因为长时间外推只会把搜索窗带离目标
        if self._center is not None and self._box is not None:
            if now - self._last_ok_ts <= _COAST_MAX_S:
                cx = self._center[0] + self._vx * dt
                cy = self._center[1] + self._vy * dt
                cx = min(max(cx, 0.0), float(w_img - 1))
                cy = min(max(cy, 0.0), float(h_img - 1))
                bw, bh = self._box[2], self._box[3]
                self._center = (cx, cy)
                self._box = (cx - bw / 2.0, cy - bh / 2.0, bw, bh)
        self._state = "coasting"
        return TrackResult(ok=False, box=self._box, center=self._center,
                           score=score, state="coasting", lost_frames=self._lost,
                           vx=self._vx, vy=self._vy)

    def _failure_result(self, score: float) -> TrackResult:
        """异常兜底路径用：按当前状态拼一个合法结果，保证 update 有返回值。"""
        if not self._started:
            return TrackResult(ok=False, box=None, center=None, score=0.0,
                               state="idle", lost_frames=0)
        if self._state == "lost":
            return TrackResult(ok=False, box=None, center=None, score=score,
                               state="lost", lost_frames=self._lost,
                               vx=self._vx, vy=self._vy)
        return TrackResult(ok=False, box=self._box, center=self._center,
                           score=score, state=self._state, lost_frames=self._lost,
                           vx=self._vx, vy=self._vy)

    def _reinit_cv(self, frame_bgr: np.ndarray,
                   box: Tuple[float, float, float, float]) -> None:
        if self._algo == "template":
            return
        tr = _create_cv_tracker(self._algo)
        if tr is None:
            self._cv = None
            self._algo = "template"
            return
        try:
            tr.init(frame_bgr, tuple(int(round(v)) for v in box))
            self._cv = tr
        except Exception:
            logger.warning("重捕后 %s 跟踪器重建失败，本目标降级为模板匹配",
                           self._algo, exc_info=True)
            self._cv = None
            self._algo = "template"

    # ------------------------------------------------------------------ 匹配工具

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    @staticmethod
    def _box_plausible(box: Tuple[float, float, float, float],
                       w_img: int, h_img: int) -> bool:
        """滤波器偶尔会吐出 0 尺寸或整体在画面外的框，先过滤掉再打分。"""
        x, y, w, h = box
        if w < 4 or h < 4:
            return False
        return (x + w > 0 and y + h > 0 and x < w_img and y < h_img)

    def _extract_patch(self, gray: np.ndarray,
                       box: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """按框取图并统一缩放到模板尺寸（CSRT 的框会缩放，模板尺寸固定）。"""
        if self._template is None:
            return None
        h_img, w_img = gray.shape[:2]
        x0 = max(0, int(round(box[0])))
        y0 = max(0, int(round(box[1])))
        x1 = min(w_img, int(round(box[0] + box[2])))
        y1 = min(h_img, int(round(box[1] + box[3])))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        patch = gray[y0:y1, x0:x1].astype(np.float32)
        th, tw = self._template.shape[:2]
        if patch.shape[0] != th or patch.shape[1] != tw:
            patch = cv2.resize(patch, (tw, th))
        return patch

    def _score_at(self, gray: np.ndarray,
                  box: Tuple[float, float, float, float]) -> float:
        """在指定框位置对当前模板做 NCC 打分（同尺寸对比，结果是标量）。"""
        patch = self._extract_patch(gray, box)
        if patch is None or self._template is None:
            return 0.0
        # 平坦区域上 TM_CCOEFF_NORMED 分母为 0，结果无定义，直接判 0 分
        if float(patch.std()) < 1e-6 or float(self._template.std()) < 1e-6:
            return 0.0
        res = cv2.matchTemplate(patch, self._template, cv2.TM_CCOEFF_NORMED)
        val = float(np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)[0, 0])
        return max(0.0, min(1.0, val))

    def _match_in_window(self, gray: np.ndarray,
                         center: Optional[Tuple[float, float]],
                         scale: float) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        """以 center 为中心、scale 倍模板大小的窗口内做 NCC 搜索，
        返回 (最高分, 对应框)。窗口自动裁到画面内；无法搜索时返回 (0, None)。"""
        if self._template is None or center is None:
            return 0.0, None
        th, tw = self._template.shape[:2]
        h_img, w_img = gray.shape[:2]
        # 窗口至少比模板大 2px，否则 matchTemplate 没有滑动空间
        ww = min(max(int(round(tw * scale)), tw + 2), w_img)
        wh = min(max(int(round(th * scale)), th + 2), h_img)
        if ww < tw or wh < th:
            return 0.0, None
        x0 = int(round(center[0] - ww / 2.0))
        y0 = int(round(center[1] - wh / 2.0))
        x0 = min(max(x0, 0), w_img - ww)
        y0 = min(max(y0, 0), h_img - wh)
        roi = gray[y0:y0 + wh, x0:x0 + ww].astype(np.float32)
        res = cv2.matchTemplate(roi, self._template, cv2.TM_CCOEFF_NORMED)
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        box = (float(x0 + max_loc[0]), float(y0 + max_loc[1]), float(tw), float(th))
        return max(0.0, min(1.0, float(max_val))), box


# ---------------------------------------------------------------------------
# 自检：合成 200 帧（灰底 + 移动纹理方块 + 噪声 + 中途 10 帧大遮挡），
# 验证成功率、遮挡后重捕、末帧中心精度。单独跑：python -m core.tracker
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rng = np.random.default_rng(7)
    W, H, N = 640, 480, 200
    TW = TH = 40
    # 目标纹理：随机噪点轻微平滑，保证 NCC 有足够辨识度又不至于像真噪声
    _tex = rng.integers(30, 220, size=(TH, TW)).astype(np.uint8)
    _tex = cv2.GaussianBlur(_tex, (3, 3), 0)
    _tex_bgr = cv2.cvtColor(_tex, cv2.COLOR_GRAY2BGR)

    X0, Y0 = 80.0, 90.0          # 起始中心
    VX_T, VY_T = 1.9, 1.2        # 真值速度（px/帧）
    OCCL_BEGIN, OCCL_END = 80, 90  # [80, 90) 共 10 帧大遮挡

    def _make_frame(i: int) -> Tuple[np.ndarray, Tuple[float, float]]:
        img = np.full((H, W, 3), 90, dtype=np.uint8)
        cx = X0 + VX_T * i
        cy = Y0 + VY_T * i
        x = int(round(cx - TW / 2.0))
        y = int(round(cy - TH / 2.0))
        img[y:y + TH, x:x + TW] = _tex_bgr
        if OCCL_BEGIN <= i < OCCL_END:
            # 大块遮挡完全盖住目标，模拟云 / 障碍物飘过
            cv2.rectangle(img, (x - 60, y - 60), (x + TW + 60, y + TH + 60),
                          (60, 60, 60), -1)
        noise = rng.normal(0.0, 6.0, size=img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img, (cx, cy)

    logger.info("可用算法：%s", available_algos())
    tracker = TargetTracker(algo="csrt")

    frame0, c0 = _make_frame(0)
    assert tracker.start(frame0, (c0[0] - TW / 2.0, c0[1] - TH / 2.0, TW, TH)), \
        "start 失败：无法初始化跟踪"
    assert tracker.active and tracker.template() is not None

    n_track = 0
    recaptured = False
    occl_states = []
    last_result = None
    last_truth = None
    for i in range(1, N):
        frame, truth = _make_frame(i)
        r = tracker.update(frame)
        if r.state == "tracking":
            n_track += 1
            if i >= OCCL_END:
                recaptured = True
        if OCCL_BEGIN <= i < OCCL_END:
            occl_states.append(r.state)
        last_result = r
        last_truth = truth

    rate = n_track / float(N - 1)
    if last_result is not None and last_result.center is not None:
        err = float(np.hypot(last_result.center[0] - last_truth[0],
                             last_result.center[1] - last_truth[1]))
    else:
        err = float("inf")

    logger.info("成功率 %.1f%%（%d/%d），遮挡期间状态=%s", rate * 100.0,
                n_track, N - 1, sorted(set(occl_states)))
    logger.info("遮挡后重捕=%s，末帧状态=%s score=%.2f 误差=%.2f px 速度=(%.0f, %.0f) px/s",
                recaptured, last_result.state, last_result.score, err,
                last_result.vx, last_result.vy)

    assert rate > 0.90, "成功率不达标：%.3f" % rate
    assert recaptured, "遮挡结束后未能重捕目标"
    assert err < 8.0, "末帧中心误差过大：%.2f px" % err
    # 遮挡期间不应该报 tracking（那就是跟到遮挡物上了）
    assert "tracking" not in occl_states, "遮挡期间误报成功：%s" % occl_states

    # 附带验证 reset 与 idle 行为符合契约
    tracker.reset()
    assert not tracker.active
    idle_r = tracker.update(frame0)
    assert idle_r.state == "idle" and not idle_r.ok

    logger.info("自测通过")
