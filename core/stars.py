# -*- coding: utf-8 -*-
"""星点检测：自动找出画面里的恒星/行星，供「点一下就跟」和标定选星用。

为什么不用 YOLO 这类目标检测网络：
星点是几个像素的点光源，没有形状、没有纹理，正是 CNN 检测器最不擅长的东西；
COCO 预训练模型里也根本没有「恒星」这一类。天文界几十年来的做法
（SExtractor / DAOStarFinder）是背景估计 → 阈值分割 → 连通域 → 亮度加权质心，
在这个问题上又快又准：720p 一帧十几毫秒、亚像素精度、不吃 GPU、
不给安装包增加一个字节，而且每一步都能讲给学生听。

三个关键的工程选择：
  1. 背景用大 sigma 高斯估计再相减。天光不是常数 —— 有月亮、有城市光污染、
     镜筒还会有渐晕，用全局阈值会在画面一侧漏检、另一侧全是伪影。
  2. 噪声用 MAD 而不是标准差。标准差会被星点本身抬高，星越多阈值越高，
     结果是「星空越灿烂，检出越少」这种荒唐行为。MAD 对离群点免疫。
  3. 质心用亮度加权矩，不用连通域的几何中心。前者是亚像素的，
     后者只能到整像素 —— 而标定的全部精度都建立在质心精度上。
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: 背景估计的高斯 sigma（像素）。要显著大于星点尺寸、又小于天光变化的尺度。
_BG_SIGMA = 12.0

#: 检出阈值 = 背景 + 这个倍数 × 噪声。5σ 是天文测光的惯例：
#: 720p 有 92 万像素，3σ 会带来上千个纯噪声的假星点。
_SIGMA_K = 5.0

#: 星点的面积上下限（像素）。低于下限多半是热像素/宇宙线打点，
#: 高于上限是月亮、云缝或者过曝的大光斑 —— 都不适合当跟踪点。
_MIN_AREA = 3
_MAX_AREA = 2000

#: 峰值到这个亮度就认为过曝。过曝星的质心会被削平，
#: 用它标定等于把误差写进标定矩阵。
_SATURATION = 250

#: MAD 换算成高斯标准差的系数
_MAD_TO_SIGMA = 1.4826


@dataclass
class Source:
    """一个检出的光源（恒星/行星/亮的深空天体核心）。"""

    x: float            # 亚像素质心
    y: float
    flux: float         # 扣背景后的总流量（相对亮度）
    peak: float         # 峰值像素值 0..255
    fwhm: float         # 半高全宽（像素），对焦看它
    area: int           # 连通域像素数
    saturated: bool     # 峰值是否顶到饱和
    nearest_px: float   # 到最近邻光源的距离，越大越孤立
    score: float = 0.0  # 作为标定星的适合度 0..1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _estimate_noise(residual: np.ndarray) -> float:
    """用 MAD 估噪声标准差。

    直接取 std 会被星点抬高 —— 星越多阈值越高、检出越少，
    表现为「对着银河反而找不到星」。MAD 只看中位数附近，天然免疫。
    """
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    sigma = mad * _MAD_TO_SIGMA
    # 全黑画面 MAD 会是 0，阈值退化成 0 会把噪点全检出来
    return max(sigma, 0.5)


def _fwhm_from_moments(patch: np.ndarray) -> float:
    """由二阶矩估计 FWHM。

    高斯光斑有 FWHM = 2√(2ln2)·σ ≈ 2.355σ，而 σ² 正是亮度加权的二阶中心矩。
    比拟合高斯快一个数量级，对焦辅助这种实时用途足够。
    """
    total = float(patch.sum())
    if total <= 0:
        return 0.0
    h, w = patch.shape
    ys, xs = np.mgrid[0:h, 0:w]
    cx = float((xs * patch).sum() / total)
    cy = float((ys * patch).sum() / total)
    varx = float((((xs - cx) ** 2) * patch).sum() / total)
    vary = float((((ys - cy) ** 2) * patch).sum() / total)
    sigma = math.sqrt(max(0.0, 0.5 * (varx + vary)))
    return 2.3548 * sigma


def detect_sources(image: np.ndarray, max_sources: int = 60,
                   sigma_k: float = _SIGMA_K) -> List[Source]:
    """找出画面里的光源，按亮度从高到低返回（最多 max_sources 个）。

    image 为 BGR 或灰度 uint8。任何异常都返回空列表 —— 这是个辅助功能，
    绝不能因为它把采集线程搞崩。
    """
    try:
        return _detect_inner(image, max_sources, sigma_k)
    except Exception:
        logger.exception("星点检测失败，返回空结果")
        return []


def _detect_inner(image: np.ndarray, max_sources: int,
                  sigma_k: float) -> List[Source]:
    if image is None or image.size == 0:
        return []
    gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3 else image)
    gray_f = gray.astype(np.float32)

    # ---- 背景与噪声 ----
    background = cv2.GaussianBlur(gray_f, (0, 0), _BG_SIGMA)
    residual = gray_f - background
    sigma = _estimate_noise(residual)
    thresh = sigma_k * sigma

    mask = (residual > thresh).astype(np.uint8)
    if int(mask.sum()) == 0:
        return []
    # 一次开运算去掉孤立的单像素噪点（热像素、宇宙线），
    # 星点因为有 PSF 展宽，至少跨 2~3 个像素，不会被吃掉
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((2, 2), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n_labels <= 1:
        return []

    h, w = gray.shape[:2]
    found: List[Source] = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < _MIN_AREA or area > _MAX_AREA:
            continue
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        # 取略大于连通域的窗口：阈值只截到了光斑的顶部，
        # 质心和 FWHM 要连翼部一起算才准
        pad = 3
        xa, xb = max(0, x0 - pad), min(w, x0 + bw + pad)
        ya, yb = max(0, y0 - pad), min(h, y0 + bh + pad)
        patch = residual[ya:yb, xa:xb].copy()
        patch[patch < 0] = 0.0
        total = float(patch.sum())
        if total <= 0:
            continue

        ys, xs = np.mgrid[ya:yb, xa:xb]
        cx = float((xs * patch).sum() / total)
        cy = float((ys * patch).sum() / total)
        peak = float(gray[ya:yb, xa:xb].max())

        found.append(Source(
            x=cx, y=cy, flux=total, peak=peak,
            fwhm=_fwhm_from_moments(patch), area=area,
            saturated=peak >= _SATURATION, nearest_px=float("inf")))

    found.sort(key=lambda s: s.flux, reverse=True)
    found = found[:max_sources]
    _fill_nearest(found)
    return found


def _fill_nearest(sources: List[Source]) -> None:
    """算每个光源到最近邻的距离。

    孤立性对跟踪和标定都要紧：挨得近的两颗星，跟踪器会在它们之间跳来跳去，
    而标定过程中基座一动，跟踪器很可能锁到旁边那颗上 —— 解出来的矩阵
    带着一个凭空的位移，方向就是这么被解反的。
    """
    n = len(sources)
    if n < 2:
        for s in sources:
            s.nearest_px = float("inf")
        return
    pts = np.array([[s.x, s.y] for s in sources], dtype=np.float64)
    for i, s in enumerate(sources):
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        d[i] = np.inf
        s.nearest_px = float(d.min())


def score_for_calibration(sources: List[Source],
                          frame_shape: Tuple[int, int],
                          step_px: float = 200.0) -> List[Source]:
    """给每个光源打「适不适合用来标定」的分，并写回 score 字段。

    标定要让基座正反各走两步、把目标在画面里推来推去。选错星的代价很高：
    上一次现场标定就是因为选了会移动的目标，解出的矩阵方向是反的，
    而界面上一切正常。这里把「什么样的星才合格」变成软件自己的判断：

      * 不能过曝 —— 削平的峰会让质心失真
      * 要足够亮 —— 信噪比低的话每一步的质心抖动都进最小二乘
      * 要孤立 —— 邻星会把跟踪器勾走
      * 要离边缘够远 —— 标定一推就出视场，那是最常见的失败
      * FWHM 要正常 —— 太小是热像素，太大是失焦或拖影
    """
    h, w = frame_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    max_flux = max((s.flux for s in sources), default=1.0) or 1.0

    for s in sources:
        if s.saturated:
            s.score = 0.0
            continue

        # 亮度：对数压缩，避免最亮那颗一家独大
        bright = math.log1p(s.flux) / math.log1p(max_flux) if max_flux > 0 else 0.0

        # 边距：要装得下一整轮标定步长，否则一推就飞出画面
        margin = min(s.x, w - s.x, s.y, h - s.y)
        room = min(1.0, margin / max(1.0, step_px * 1.2))

        # 孤立：邻星至少要在 3 倍标定步长之外才算真的安全
        if math.isinf(s.nearest_px):
            lonely = 1.0
        else:
            lonely = min(1.0, s.nearest_px / max(1.0, step_px * 0.5))

        # 形状：FWHM 落在 1.5~8 像素算正常
        if s.fwhm <= 0:
            shape = 0.0
        elif s.fwhm < 1.5:
            shape = s.fwhm / 1.5          # 太尖 = 热像素
        elif s.fwhm <= 8.0:
            shape = 1.0
        else:
            shape = max(0.0, 1.0 - (s.fwhm - 8.0) / 12.0)

        # 靠中心一点点加分：四步走完还能回到原位
        centered = 1.0 - min(1.0, math.hypot(s.x - cx, s.y - cy)
                             / math.hypot(cx, cy))

        s.score = float(0.30 * bright + 0.28 * room + 0.22 * lonely
                        + 0.12 * shape + 0.08 * centered)
    return sources


def best_calibration_star(sources: List[Source],
                          frame_shape: Tuple[int, int],
                          step_px: float = 200.0) -> Optional[Source]:
    """挑一颗最适合标定的星；实在没有合格的就返回 None。"""
    if not sources:
        return None
    score_for_calibration(sources, frame_shape, step_px)
    best = max(sources, key=lambda s: s.score)
    # 分太低说明这一帧里就没有像样的候选（全过曝、全挤在一起、全在边上），
    # 与其推荐一颗坏星，不如老实说没有
    return best if best.score >= 0.35 else None


def focus_metric(sources: List[Source]) -> Optional[float]:
    """对焦指标：取较亮那批星的 FWHM 中位数，越小越锐。

    只用亮星是因为暗星的 FWHM 被噪声主导，抖得没法看；取中位数而不是
    平均，是为了不被一颗拖影的星带偏。
    """
    usable = [s for s in sources if not s.saturated and s.fwhm > 0]
    if not usable:
        return None
    usable.sort(key=lambda s: s.flux, reverse=True)
    top = usable[:max(3, len(usable) // 4)]
    return float(np.median([s.fwhm for s in top]))


# ======================================================================
# 自检：合成一张已知星位的星图，验证能不能把它们找回来
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("stars.selftest")

    W, H = 1280, 720
    rng = np.random.default_rng(7)

    def make_sky(stars, bg_gradient=True, noise=3.0):
        """stars = [(x, y, 峰值亮度, sigma), ...]"""
        img = np.zeros((H, W), np.float32)
        if bg_gradient:
            # 城市光污染式的背景梯度：一侧亮一侧暗，考验背景估计
            img += np.linspace(30.0, 8.0, W, dtype=np.float32)[None, :]
            img += np.linspace(6.0, 0.0, H, dtype=np.float32)[:, None]
        for (sx, sy, amp, sg) in stars:
            x0, x1 = max(0, int(sx - 6 * sg)), min(W, int(sx + 6 * sg) + 1)
            y0, y1 = max(0, int(sy - 6 * sg)), min(H, int(sy + 6 * sg) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            ys, xs = np.mgrid[y0:y1, x0:x1]
            img[y0:y1, x0:x1] += amp * np.exp(
                -(((xs - sx) ** 2 + (ys - sy) ** 2) / (2.0 * sg * sg)))
        img += rng.normal(0.0, noise, (H, W)).astype(np.float32)
        return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                            cv2.COLOR_GRAY2BGR)

    # ---- 位置精度：亚像素质心 ----
    truth = [(320.3, 180.7, 180.0, 1.8), (900.6, 300.2, 120.0, 2.0),
             (640.5, 540.4, 200.0, 1.6), (1100.2, 620.9, 90.0, 2.2),
             (200.8, 600.1, 150.0, 1.9)]
    sky = make_sky(truth)
    srcs = detect_sources(sky)
    assert len(srcs) >= len(truth), \
        "5 颗星只找到 %d 个（阈值或背景估计有问题）" % len(srcs)

    errs = []
    for (tx, ty, _amp, _sg) in truth:
        near = min(srcs, key=lambda s: math.hypot(s.x - tx, s.y - ty))
        e = math.hypot(near.x - tx, near.y - ty)
        errs.append(e)
        assert e < 0.5, "星 (%.1f,%.1f) 的质心误差 %.2f px，超过亚像素要求" % (tx, ty, e)
    log.info("质心精度：%d 颗星，最大误差 %.3f px，平均 %.3f px",
             len(truth), max(errs), sum(errs) / len(errs))

    # ---- 背景梯度不能造成一侧漏检 ----
    # 左边背景 30、右边 8，如果用全局阈值，右侧的暗星会全被淹掉
    dim = [(100.0, 360.0, 60.0, 1.8), (1180.0, 360.0, 60.0, 1.8)]
    sky2 = make_sky(dim)
    s2 = detect_sources(sky2)
    left = [s for s in s2 if s.x < W / 2]
    right = [s for s in s2 if s.x > W / 2]
    assert left and right, \
        "背景梯度下漏检：左 %d 个、右 %d 个" % (len(left), len(right))
    log.info("背景梯度：亮侧与暗侧的等亮度星都检出了")

    # ---- 星多不能反而检出变少（MAD 抗污染）----
    many = [(rng.uniform(60, W - 60), rng.uniform(60, H - 60),
             rng.uniform(70, 200), rng.uniform(1.5, 2.4)) for _ in range(120)]
    s3 = detect_sources(make_sky(many), max_sources=200)
    assert len(s3) >= 80, \
        "密集星场只检出 %d 个，噪声估计被星点污染了（该用 MAD）" % len(s3)
    log.info("密集星场：投 120 颗，检出 %d 个", len(s3))

    # ---- 空画面不能凭空产生星点 ----
    blank = make_sky([], bg_gradient=True)
    s4 = detect_sources(blank)
    assert len(s4) <= 3, "空画面检出了 %d 个假星点" % len(s4)
    log.info("空画面：假检出 %d 个（阈值合理）", len(s4))

    # ---- FWHM 反映锐度：失焦的星 FWHM 必须更大 ----
    sharp = detect_sources(make_sky([(640.0, 360.0, 200.0, 1.5)]))
    blur = detect_sources(make_sky([(640.0, 360.0, 200.0, 4.0)]))
    assert sharp and blur
    f_sharp, f_blur = focus_metric(sharp), focus_metric(blur)
    assert f_sharp is not None and f_blur is not None
    assert f_blur > f_sharp * 1.5, \
        "失焦星的 FWHM（%.2f）没有明显大于锐利星（%.2f）" % (f_blur, f_sharp)
    log.info("对焦指标：锐利 %.2f px vs 失焦 %.2f px", f_sharp, f_blur)

    # ---- 选标定星：过曝的、贴边的、挤在一起的都不能选 ----
    candidates = [
        (30.0, 360.0, 200.0, 2.0),      # 贴左边缘 —— 一推就出画面
        (640.0, 360.0, 255.0, 3.0),     # 正中但过曝
        (645.0, 365.0, 250.0, 3.0),     # 紧挨着上面那颗
        (500.0, 300.0, 150.0, 2.0),     # 干净、亮度适中、离边够远 → 应该选它
    ]
    s5 = detect_sources(make_sky(candidates))
    best = best_calibration_star(s5, (H, W), step_px=200.0)
    assert best is not None, "有合格候选却没选出标定星"
    assert math.hypot(best.x - 500.0, best.y - 300.0) < 5.0, \
        "选了 (%.0f,%.0f)，应该选那颗干净的 (500,300)" % (best.x, best.y)
    assert not best.saturated
    log.info("选标定星：正确避开了贴边、过曝和相邻的候选，选中 (%.0f, %.0f)，"
             "得分 %.2f", best.x, best.y, best.score)

    # ---- 全是烂候选时要老实返回 None ----
    bad = [(20.0, 20.0, 255.0, 3.0), (25.0, 25.0, 255.0, 3.0)]
    assert best_calibration_star(detect_sources(make_sky(bad)),
                                 (H, W), step_px=200.0) is None, \
        "没有合格候选时不该硬推荐一颗"
    log.info("无合格候选时正确返回 None")

    # ---- 异常输入不能抛异常 ----
    assert detect_sources(None) == []
    assert detect_sources(np.zeros((0, 0), np.uint8)) == []
    assert detect_sources(np.zeros((H, W), np.uint8)) == []      # 纯黑灰度图

    # ---- 性能：要能在采集线程里以低频跑 ----
    import time
    sky_perf = make_sky(many)
    t0 = time.monotonic()
    for _ in range(10):
        detect_sources(sky_perf)
    dt = (time.monotonic() - t0) / 10.0
    log.info("性能：720p 密集星场单帧 %.1f ms", dt * 1000.0)
    assert dt < 0.20, "单帧 %.0f ms 太慢，会拖累采集线程" % (dt * 1000)

    log.info("stars selftest OK")
