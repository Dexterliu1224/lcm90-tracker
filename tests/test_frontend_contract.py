"""前端调用方式必须和后端路由声明一致。

背景：前端的 api() 封装只在带 body 时才发 POST，无参调用会退化成 GET。
后端的动作接口全是 POST —— 于是「自动标定」「开始跟踪」「停止」点了全是
405 Method Not Allowed，其中「停止」还把错误吞掉了（安全隐患）。
本测试静态扫描两边的声明，出现不一致直接失败。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _post_routes() -> set:
    src = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    return set(re.findall(r'@app\.post\("([^"]+)"\)', src))


def _get_routes() -> set:
    src = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    return set(re.findall(r'@app\.get\("([^"]+)"\)', src))


def _frontend_calls():
    """返回 [(接口路径, 是否带 body), ...]，只统计 api(...) 直接调用。"""
    src = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    calls = []
    for m in re.finditer(r'\bapi\(\s*"(/api/[^"]+)"\s*(,)?', src):
        calls.append((m.group(1), m.group(2) is not None))
    return calls


def test_post_routes_are_never_called_without_body():
    # 同一路径可以同时挂 GET 和 POST（/api/users 就是：GET 列表、POST 新建）。
    # 这种路径上的无参调用是**故意**的 GET，不算违约；只有「光有 POST」
    # 的路径被无参调用才会吃 405。
    post = _post_routes() - _get_routes()
    bad = [p for p, has_body in _frontend_calls()
           if p in post and not has_body]
    assert not bad, (
        "这些 POST 接口被前端无参调用，实际会发成 GET 并得到 405：%s\n"
        "改用 act(path) 或给 api() 传一个 body（哪怕是 {}）。" % bad)


def test_act_helper_exists():
    src = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert "const act=" in src, "act() 封装被删了？动作接口会退化成 GET"


def test_ui_port_choice_overrides_config_driver():
    """界面选了真串口，就必须建 nexstar 驱动 —— config 默认 simulator 时
    曾出现"连接成功但连的是仿真基座"：遥测在动、真望远镜不动、无报错。"""
    from core.control import TrackingSession
    s = TrackingSession({"mount": {"driver": "simulator"}})
    r = s.connect_mount("/dev/tty.definitely-not-a-real-port")
    assert r["ok"] is False, "连不存在的串口必须失败，而不是悄悄退回仿真"


def test_camera_source_parsing():
    """界面选择 → 相机驱动的翻译：QHY / 普通摄像头 / 仿真三条路必须分明。"""
    from app.main import CameraReq, parse_camera_source
    assert parse_camera_source(CameraReq(sim=True))["driver"] == "simulator"
    assert parse_camera_source(CameraReq(source=None))["driver"] == "simulator"
    q = parse_camera_source(CameraReq(source="qhy:1"))
    assert q == {"driver": "qhy", "index": 1}
    o = parse_camera_source(CameraReq(source="0"))
    assert o == {"driver": "opencv", "source": 0}


def test_qhy_module_degrades_without_sdk():
    """没装 QHY SDK 的机器上，扫描必须安静返回空表 + 原因，绝不抛异常。"""
    from core.qhy import list_qhy_cameras
    devices, err = list_qhy_cameras()
    assert isinstance(devices, list)
    if err is not None:
        assert "qhyccd" in err or "QHY" in err


def test_qhy_scans_common_install_dirs_on_windows(monkeypatch):
    """Windows 上不能只按 dll 名字找：用户实测 AllInOne 装完后
    System32 里没有 qhyccd.dll，它散落在 Program Files 的各种目录下
    （QHYCCD / 态势感知 / SharpCap），只按名字找必然加载失败。"""
    import platform
    from core import qhy
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    cands = qhy._library_candidates()
    assert cands[0] == "qhyccd.dll", "仍应先试系统搜索路径"
    assert all(isinstance(c, str) for c in cands)


def test_reconnect_switches_device():
    """已连接时再点连接必须真的切换设备，不能早退保留旧的 ——
    实测出过：先连真基座、再选仿真点连接，界面显示仿真，
    标定指令却打到真望远镜上（安全事故级）。"""
    from core.control import TrackingSession
    s = TrackingSession({"mount": {"driver": "simulator"}})
    r1 = s.connect_mount(None)                # 仿真
    assert r1["ok"] is True
    first = s._mount
    r2 = s.connect_mount(None)                # 再连一次仿真 → 应换新实例
    assert r2["ok"] is True
    assert s._mount is not first, "早退保留旧设备 = 切换失效"
    # 切到不存在的真串口必须失败，且旧的仿真已被断开（不能留着乱动）
    r3 = s.connect_mount("/dev/tty.not-exist")
    assert r3["ok"] is False
    assert s._mount is None


def test_pure_azimuth_nudge_never_moves_altitude():
    """纯方位 nudge 不得改动仰角。镜筒平放（alt≈0）而下限是 5° 时，
    旧代码会把目标仰角夹到 5°，凭空要求抬高 → 标定第 1 步必然失败。"""
    from core.mount import SimMount
    m = SimMount(alt_limit_min_deg=5.0, alt_limit_max_deg=88.0)
    m.connect()
    m._az, m._alt = 100.0, 0.0        # 平放在桌上，已在下限之外
    assert m.nudge(0.3, 0.0, timeout_s=10) is True, "纯方位移动应当成功"
    assert abs(m.get_altaz()[1] - 0.0) < 0.05, "仰角不该被限位拽上去"
