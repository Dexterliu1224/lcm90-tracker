# -*- coding: utf-8 -*-
"""核心回归测试：接口契约 + 关键数学 + 已修过的 bug 不再回归。"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from core.calibration import Calibration
from core.control import PID
from core.mount import SimMount, build_mount, MountError
from core.camera import SimCamera
from core.tracker import TargetTracker, available_algos


# ---------------------------------------------------------------- 基座
def test_sim_mount_nudge_moves_and_returns():
    m = SimMount()
    m.connect()
    az0, alt0 = m.get_altaz()
    assert m.nudge(0.3, 0.0)
    az1, _ = m.get_altaz()
    assert abs(((az1 - az0 + 180) % 360 - 180) - 0.3) < 0.02
    assert m.nudge(-0.3, 0.0)


def test_mount_info_has_required_keys():
    m = SimMount(); m.connect()
    info = m.info()
    for key in ("driver", "version", "aligned", "tracking_mode", "port"):
        assert key in info


def test_build_mount_rejects_unknown_driver():
    with pytest.raises((ValueError, MountError)):
        build_mount({"driver": "nonsense"})


# ---------------------------------------------------------------- 相机
def test_sim_camera_target_stays_within_span():
    """目标必须往返运动，不能一去不回 —— 否则用户几分钟后就找不到它了。"""
    cam = SimCamera(get_pointing=lambda: (180.0, 35.0))
    cam._t0 = 0.0
    positions = [cam._target_altaz(t) for t in np.linspace(0, 60, 200)]
    azs = [p[0] for p in positions]
    alts = [p[1] for p in positions]
    assert max(azs) - min(azs) < 1.0, "方位漂移超出往返幅度"
    assert max(alts) - min(alts) < 1.0, "仰角漂移超出往返幅度"


def test_sim_camera_motion_is_smooth():
    """正弦往返：端点不能瞬间反向（三角波会把 PID 踢出 70px 尖峰）。"""
    cam = SimCamera(get_pointing=lambda: (180.0, 35.0))
    cam._t0 = 0.0
    ts = np.linspace(0, 15, 1500)
    alts = np.array([cam._target_altaz(t)[1] for t in ts])
    vel = np.diff(alts) / np.diff(ts)
    acc = np.abs(np.diff(vel) / np.diff(ts[:-1]))
    # 三角波在端点的数值加速度是巨大的脉冲；正弦应处处有限
    assert acc.max() < 1.0, "目标运动在端点不平滑"


# ---------------------------------------------------------------- 跟踪器
def test_tracker_degrades_without_exception():
    assert "template" in available_algos()
    tr = TargetTracker(algo="csrt")     # 没有 contrib 时应静默降级
    img = np.random.randint(0, 60, (240, 320, 3), dtype=np.uint8)
    # 目标要带纹理：纯色方块会被「平坦模板」检查正确拒绝
    rng = np.random.RandomState(7)
    img[100:140, 150:190] = rng.randint(120, 255, (40, 40, 3), dtype=np.uint8)
    assert tr.start(img, (150, 100, 40, 40))
    res = tr.update(img)
    assert res.state in ("tracking", "coasting", "lost", "idle")


# ---------------------------------------------------------------- 标定
def test_calibration_pixels_to_angles_divides_cos_alt():
    cal = Calibration(a11=0.001, a12=0.0, a21=0.0, a22=0.001,
                      arcsec_per_px=3.6, angle_deg=0.0, flipped=False,
                      rms_px=0.1, ok=True)
    az_low, _ = cal.pixels_to_angles(100, 0, 0.0)
    az_high, _ = cal.pixels_to_angles(100, 0, 60.0)
    assert az_high == pytest.approx(az_low / math.cos(math.radians(60.0)), rel=1e-6)


def test_calibration_roundtrip_dict():
    cal = Calibration(a11=1e-3, a12=2e-4, a21=-2e-4, a22=1e-3,
                      arcsec_per_px=3.7, angle_deg=11.3, flipped=True,
                      rms_px=0.5, ok=True, note="x")
    cal2 = Calibration.from_dict(cal.to_dict())
    assert cal2.to_dict() == cal.to_dict()


# ---------------------------------------------------------------- PID
def test_pid_feedforward_participates_in_saturation():
    pid = PID(kp=1.0, ki=0.5, kd=0.0, i_limit=0.5, i_band=10.0, out_limit=1.0)
    # 前馈已经顶满输出：积分不得在饱和区继续累积
    for _ in range(50):
        out = pid.update(0.3, 0.1, feedforward=1.0)
    assert out == pytest.approx(1.0)
    assert abs(pid._integral) < 0.4, "积分在饱和区持续增长（抗饱和失效）"


def test_pid_holds_rate_costs_error_without_ff():
    """没有前馈时稳态误差 = 速度/kp —— 这是必须用绝对速度前馈的原因。"""
    pid = PID(kp=0.9, ki=0.0, kd=0.0, out_limit=2.5)
    # 模拟：要维持 0.12°/s 的输出，需要多大的稳态误差
    err = 0.12 / 0.9
    out = pid.update(err, 0.066, feedforward=0.0)
    assert out == pytest.approx(0.12, rel=1e-6)
    # 也就是 0.133° ≈ 480 角秒 ≈ 80 像素（6"/px）—— 不可接受，所以要前馈
    assert err * 3600 / 6.0 > 50
