# -*- coding: utf-8 -*-
"""生成应用图标与界面徽标（校徽 + 航天元素）。

    python tools/make_icon.py            # 在项目根目录执行

输入（可选）：项目根目录的 school-logo.png —— 学校圆形校徽。
    有它：徽标 = 深空底 + 星尘 + 轨道环 + 卫星 + 校徽居中。
    没它：退化为纯航天徽标（轨道 + 行星），流程照常能跑。

输出：
    app.ico                      exe / 任务栏图标（16/24/32/48/64/128/256）
    app/static/favicon.ico       浏览器标签 / Edge 应用模式窗口图标
    app/static/app-badge.png     登录页与顶栏的徽标（512px）
    app/static/school-logo.png   校徽原图副本（备用）

小尺寸（16/24/32）不缩放大图，而是单独绘制简化版：校徽缩到 16px 就是
一团绿色噪点，图标设计的铁律是小尺寸要重新画 —— 只保留「绿色圆 +
白色轨道 + 卫星点」这个最强识别特征。
"""
from __future__ import annotations

import math
import os
import sys

from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, "app", "static")

# 校徽的绿（从校徽取的近似主色）与深空底色
GREEN = (30, 122, 80)
GREEN_HI = (76, 175, 118)
SPACE0 = (7, 13, 28)
SPACE1 = (13, 24, 48)
STARDUST = (200, 225, 255)
ORBIT = (240, 250, 245)


def _space_disc(size: int) -> Image.Image:
    """深空圆底：径向渐变 + 少量星尘。所有绘制在 4x 超采样后缩回，抗锯齿。"""
    s = size * 4
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx = cy = s / 2
    r = s / 2
    steps = 48
    for i in range(steps, 0, -1):
        t = i / steps
        col = tuple(int(SPACE0[j] + (SPACE1[j] - SPACE0[j]) * (1 - t))
                    for j in range(3)) + (255,)
        d.ellipse([cx - r * t, cy - r * t, cx + r * t, cy + r * t], fill=col)
    # 星尘：确定性伪随机（不 import random，图标要可复现）
    seed = 20260818
    for i in range(90):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        x = (seed % 1000) / 1000 * s
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        y = (seed % 1000) / 1000 * s
        if math.hypot(x - cx, y - cy) > r * 0.96:
            continue
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        rr = 1 + (seed % 100) / 100 * s / 220
        a = 60 + seed % 140
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=STARDUST + (a,))
    return im.resize((size, size), Image.LANCZOS)


def _sparkle(d: ImageDraw.ImageDraw, x: float, y: float, r: float,
             color=(255, 255, 255, 235)) -> None:
    """四芒星：比圆点更「航天」，也是画面里准星语言的延续。"""
    d.polygon([(x, y - r), (x + r * 0.28, y - r * 0.28), (x + r, y),
               (x + r * 0.28, y + r * 0.28), (x, y + r),
               (x - r * 0.28, y + r * 0.28), (x - r, y),
               (x - r * 0.28, y - r * 0.28)], fill=color)


def _orbit_ring(size: int, front: bool) -> Image.Image:
    """倾斜轨道环。分前后两半：后半压在校徽下、前半盖在校徽上，
    形成「轨道穿过徽章」的立体感 —— 呼应校徽里自带的那道轨道。"""
    s = size * 4
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx = cy = s / 2
    a, b = s * 0.470, s * 0.170          # 长短半轴
    tilt = math.radians(-18)
    w = max(2, int(s * 0.016))
    pts = []
    for deg in range(0, 361, 2):
        t = math.radians(deg)
        ex, ey = a * math.cos(t), b * math.sin(t)
        x = cx + ex * math.cos(tilt) - ey * math.sin(tilt)
        y = cy + ex * math.sin(tilt) + ey * math.cos(tilt)
        pts.append((x, y, math.sin(t)))
    for i in range(len(pts) - 1):
        (x1, y1, s1), (x2, y2, _s2) = pts[i], pts[i + 1]
        is_front = s1 > 0                 # sin>0 的半边在观察者一侧
        if is_front != front:
            continue
        alpha = 235 if front else 130
        d.line([x1, y1, x2, y2], fill=ORBIT[:3] + (alpha,), width=w)
    if front:
        # 卫星：轨道前段上的一个亮点 + 短高光尾
        t = math.radians(35)
        ex, ey = a * math.cos(t), b * math.sin(t)
        x = cx + ex * math.cos(tilt) - ey * math.sin(tilt)
        y = cy + ex * math.sin(tilt) + ey * math.cos(tilt)
        rr = s * 0.030
        glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.ellipse([x - rr * 2.2, y - rr * 2.2, x + rr * 2.2, y + rr * 2.2],
                   fill=GREEN_HI + (110,))
        im.alpha_composite(glow.filter(ImageFilter.GaussianBlur(s * 0.01)))
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=GREEN_HI + (255,))
        d.ellipse([x - rr * 0.45, y - rr * 0.6, x + rr * 0.2, y + rr * 0.05],
                  fill=(255, 255, 255, 220))
    return im.resize((size, size), Image.LANCZOS)


def _circle_crop(im: Image.Image) -> Image.Image:
    """把校徽裁成正圆（本来就是圆形徽章，裁掉白边噪声）。"""
    im = im.convert("RGBA")
    side = min(im.size)
    im = im.crop(((im.width - side) // 2, (im.height - side) // 2,
                  (im.width + side) // 2, (im.height + side) // 2))
    mask = Image.new("L", (side * 4, side * 4), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, side * 4, side * 4], fill=255)
    im.putalpha(mask.resize((side, side), Image.LANCZOS))
    return im


def build_badge(size: int, emblem: Image.Image | None) -> Image.Image:
    im = _space_disc(size)
    im.alpha_composite(_orbit_ring(size, front=False))
    if emblem is not None:
        es = int(size * 0.60)
        em = emblem.resize((es, es), Image.LANCZOS)
        # 白色描边圈让校徽从深底上分离出来
        pad = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        pd = ImageDraw.Draw(pad)
        c = size / 2
        pd.ellipse([c - es / 2 - size * 0.012, c - es / 2 - size * 0.012,
                    c + es / 2 + size * 0.012, c + es / 2 + size * 0.012],
                   fill=(255, 255, 255, 255))
        im.alpha_composite(pad)
        im.alpha_composite(em, (int(c - es / 2), int(c - es / 2)))
    else:
        # 没有校徽：中央画一颗绿色行星
        d = ImageDraw.Draw(im)
        c = size / 2
        pr = size * 0.22
        d.ellipse([c - pr, c - pr, c + pr, c + pr], fill=GREEN + (255,))
        d.ellipse([c - pr * 0.55, c - pr * 0.8, c + pr * 0.1, c - pr * 0.05],
                  fill=GREEN_HI + (160,))
    im.alpha_composite(_orbit_ring(size, front=True))
    # 点缀两颗四芒星
    d = ImageDraw.Draw(im)
    _sparkle(d, size * 0.205, size * 0.185, size * 0.036)
    _sparkle(d, size * 0.83, size * 0.70, size * 0.024, (255, 255, 255, 190))
    # 外圈细描边收边
    d.ellipse([1, 1, size - 2, size - 2], outline=(255, 255, 255, 60),
              width=max(1, size // 170))
    return im


def build_small(size: int) -> Image.Image:
    """16/24/32 专用简化版：绿色圆 + 白轨道 + 卫星点。"""
    s = size * 8
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    m = s * 0.04
    d.ellipse([m, m, s - m, s - m], fill=SPACE1 + (255,))
    c = s / 2
    pr = s * 0.24
    d.ellipse([c - pr, c - pr, c + pr, c + pr], fill=GREEN + (255,))
    # 一道简化轨道（前半）
    a, b = s * 0.42, s * 0.15
    tilt = math.radians(-18)
    w = max(2, int(s * 0.05))
    prev = None
    for deg in range(0, 181, 4):
        t = math.radians(deg)
        ex, ey = a * math.cos(t), b * math.sin(t)
        x = c + ex * math.cos(tilt) - ey * math.sin(tilt)
        y = c + ex * math.sin(tilt) + ey * math.cos(tilt)
        if prev:
            d.line([prev, (x, y)], fill=(255, 255, 255, 255), width=w)
        prev = (x, y)
    rr = s * 0.075
    d.ellipse([c + a * 0.62 - rr, c - b * 1.9 - rr,
               c + a * 0.62 + rr, c - b * 1.9 + rr],
              fill=(255, 255, 255, 255))
    return im.resize((size, size), Image.LANCZOS)


def main() -> int:
    emblem = None
    for cand in (os.path.join(ROOT, "school-logo.png"),
                 os.path.join(STATIC, "school-logo.png")):
        if os.path.exists(cand):
            emblem = _circle_crop(Image.open(cand))
            print("使用校徽：%s" % cand)
            break
    if emblem is None:
        print("未找到 school-logo.png，先用纯航天版徽标（放入校徽后重跑即可）")

    badge = build_badge(512, emblem)
    os.makedirs(STATIC, exist_ok=True)
    badge.save(os.path.join(STATIC, "app-badge.png"))
    if emblem is not None:
        emblem.resize((512, 512), Image.LANCZOS).save(
            os.path.join(STATIC, "school-logo.png"))

    frames = [build_small(16), build_small(24), build_small(32),
              build_badge(48, emblem), build_badge(64, emblem),
              build_badge(128, emblem), build_badge(256, emblem)]
    ico = os.path.join(ROOT, "app.ico")
    # Pillow 的 ICO 写入以第一帧为基准、按 sizes 缩放；要保留手绘的小尺寸，
    # 用 append_images 逐帧给
    frames[-1].save(ico, format="ICO",
                    sizes=[(f.width, f.height) for f in frames],
                    append_images=frames[:-1])
    import shutil
    shutil.copyfile(ico, os.path.join(STATIC, "favicon.ico"))
    print("已生成 app.ico / favicon.ico / app-badge.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
