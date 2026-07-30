#!/usr/bin/env python3
"""启动 LCM90 视觉跟踪台。

    python serve.py              # 默认 http://127.0.0.1:8300
    python serve.py --port 8400
    python serve.py --verbose    # 排查问题时打开完整日志
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
import webbrowser


def _force_utf8_output() -> None:
    """UTF-8 输出 + 行缓冲。

    Windows 控制台默认用系统 ANSI 编码（英文系统 cp1252 编不了中文），
    第一句中文 print 就会把进程打死；输出被重定向时 Python 还会改成
    块缓冲，窗口长时间一片空白。两个坑一起堵上。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace",
                               line_buffering=True)
        except Exception:
            pass


_force_utf8_output()

if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="打印每一条网页请求，排查问题时才需要")
    args = ap.parse_args()

    import yaml
    # 打包成 exe 后 __file__ 指向临时解压目录；配置必须跟着 exe 走，
    # 用户改了才生效。
    if getattr(sys, "frozen", False):
        root = os.path.dirname(os.path.abspath(sys.executable))
    else:
        root = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(root, "config.yaml"), "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        print("✗ 找不到 config.yaml，它应该和本程序在同一个文件夹里。")
        return 1

    server = cfg.get("server", {}) or {}
    host = args.host or server.get("host", "127.0.0.1")
    port = args.port or int(server.get("port", 8300))
    url = "http://127.0.0.1:%d" % port

    print()
    print("  ┌" + "─" * 50 + "┐")
    print("  │  LCM90 视觉跟踪台" + " " * 32 + "│")
    print("  └" + "─" * 50 + "┘")
    print()
    print("    控制台地址   %s" % url)
    print("    浏览器会自动打开。要退出程序，直接关掉这个窗口。")
    print("    这里平时不会输出东西 —— 出问题时才会打印原因。")
    print()

    if not args.no_browser:
        def _open():
            time.sleep(2.0)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="  %(levelname)s  %(message)s")

    import uvicorn
    try:
        # 直接导入 app 对象（不是字符串路径）—— PyInstaller 的静态分析
        # 看不见字符串里的模块名，打包 exe 会缺模块，启动即退。
        from app.main import app
        uvicorn.run(app, host=host, port=port,
                    log_level="info" if args.verbose else "warning",
                    access_log=args.verbose, use_colors=False)
    except OSError as exc:
        print("\n  ✗ 启动失败：%s" % exc)
        print("    端口 %d 可能被占用，改 config.yaml 里的 port 再试。" % port)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    code = main()
    if code != 0 and getattr(sys, "frozen", False):
        try:
            input("\n按回车键关闭...")
        except (EOFError, OSError):
            pass
    sys.exit(code)
