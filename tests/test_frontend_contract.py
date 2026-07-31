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


def _frontend_calls():
    """返回 [(接口路径, 是否带 body), ...]，只统计 api(...) 直接调用。"""
    src = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    calls = []
    for m in re.finditer(r'\bapi\(\s*"(/api/[^"]+)"\s*(,)?', src):
        calls.append((m.group(1), m.group(2) is not None))
    return calls


def test_post_routes_are_never_called_without_body():
    post = _post_routes()
    bad = [p for p, has_body in _frontend_calls()
           if p in post and not has_body]
    assert not bad, (
        "这些 POST 接口被前端无参调用，实际会发成 GET 并得到 405：%s\n"
        "改用 act(path) 或给 api() 传一个 body（哪怕是 {}）。" % bad)


def test_act_helper_exists():
    src = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert "const act=" in src, "act() 封装被删了？动作接口会退化成 GET"
