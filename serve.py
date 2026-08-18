#!/usr/bin/env python3
"""启动 LCM90 视觉跟踪台。

默认是一个**独立的桌面窗口**（不弹浏览器）：
后端 uvicorn 跑在后台线程，界面装在系统自带的 WebView 里。

    python serve.py                 # 桌面窗口
    python serve.py --browser       # 退回浏览器模式（窗口起不来时用）
    python serve.py --no-window     # 只跑服务、不开窗（自动化/远程访问用）
    python serve.py --port 8400
    python serve.py --verbose       # 排查问题时打开完整日志
    python serve.py --check         # 自检：确认 WebView 组件齐全后退出
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from typing import Optional

logger = logging.getLogger("serve")


def _app_root() -> str:
    """程序所在目录。打包成 exe 后 __file__ 指向临时解压目录，
    配置/数据必须跟着 exe 走，用户改了才生效。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _force_utf8_output() -> None:
    """UTF-8 输出 + 行缓冲，并保证 stdout/stderr 一定可写。

    两个坑：
      1. Windows 控制台默认用系统 ANSI 编码（英文系统 cp1252 编不了中文），
         第一句中文 print 就会把进程打死；输出被重定向时还会变成块缓冲。
      2. 打包成**无控制台**的窗口程序后，sys.stdout / sys.stderr 直接是
         None —— 任何一句 print 都会抛 AttributeError，而且因为没有控制台，
         用户只看到程序闪一下就没了，毫无线索。
    所以没有 stdout 时把它接到 exe 旁边的日志文件上。
    """
    log_path = os.path.join(_app_root(), "data", "app.log")
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            try:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                fh = open(log_path, "a", encoding="utf-8", buffering=1)
                setattr(sys, name, fh)
            except Exception:
                # 连日志都写不了也不能崩：接一个黑洞，让 print 静默成功
                try:
                    setattr(sys, name, open(os.devnull, "w"))
                except Exception:
                    pass
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace",
                               line_buffering=True)
        except Exception:
            pass


_force_utf8_output()

if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port(host: str, port: int, tries: int = 20) -> int:
    """配置端口被占就顺延。

    教学现场最常见的情况是「上一个实例还没退干净就又双击了一次」，
    此时直接报错退出对用户毫无意义 —— 他既看不懂端口冲突，
    也不知道去任务管理器杀进程。换个端口继续跑就是了。
    """
    for offset in range(tries):
        if _port_is_free(host, port + offset):
            return port + offset
    return port


def _wait_until_serving(host: str, port: int, timeout_s: float = 30.0) -> bool:
    """等到端口真的能连上再开窗。

    先开窗后启动服务的话，WebView 会抢在服务就绪前加载，
    拿到 ERR_CONNECTION_REFUSED 并停在一个白屏上不再重试。
    """
    deadline = time.monotonic() + timeout_s
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((probe_host, port)) == 0:
                return True
        time.sleep(0.2)
    return False


def _check_webview() -> int:
    """自检 WebView 组件是否齐全。CI 拿它验证打包结果 ——
    GUI 窗口在无头环境里开不出来，但「模块能不能导入」正是
    PyInstaller 最容易漏掉的东西，而这一项测得到。"""
    native = False
    try:
        import webview  # noqa: F401
        print("OK pywebview 可导入")
        native = True
    except Exception as exc:
        print("!  pywebview 不可用：%s" % exc)

    if native and sys.platform == "win32":
        # 光 import webview 不够：真正的失败点在 winforms 后端 import clr，
        # 而 pythonnet 的 Python.Runtime.dll 在 PyInstaller 下经常解析不了。
        # 这里必须把它真的加载一遍，否则 --check 会给出虚假的通过。
        try:
            import clr  # noqa: F401
            print("OK pythonnet/clr 已就绪（原生窗口可用）")
        except Exception as exc:
            native = False
            print("!  pythonnet/clr 加载失败：%s" % exc)

    browser = _find_browser()
    if browser:
        print("OK 应用模式窗口可用：%s" % browser)
    else:
        print("!  没找到 Edge/Chrome，应用模式窗口不可用")

    if not native and not browser:
        print("x 两种独立窗口都用不了，只能退回普通浏览器")
        return 1
    return 0


#: Edge / Chrome 的常见安装位置。用「应用模式」开窗是 pywebview 之外
#: 最可靠的一条路：Windows 一定有 Edge，而且完全不碰 .NET。
_BROWSER_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)


def _find_browser() -> Optional[str]:
    import shutil
    for path in _BROWSER_CANDIDATES:
        if os.path.exists(path):
            return path
    for name in ("msedge", "chrome", "chromium", "brave"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _open_app_window(url: str, profile_dir: str) -> bool:
    """用 Edge/Chrome 的 --app 模式开一个独立窗口。

    看起来就是个桌面程序：没有地址栏、没有标签页、没有书签栏，任务栏里
    是独立图标。相比 pywebview 的好处是**完全不依赖 .NET/pythonnet** ——
    后者在 PyInstaller 下极易打包失败（Python.Runtime.dll 解析不了），
    而 Windows 上 Edge 是必装组件。

    --user-data-dir 必须给一个独立目录：不给的话，如果用户已经开着 Edge，
    新进程会把请求转交给现有实例然后**立刻退出**，我们就会误以为窗口关了。
    """
    exe = _find_browser()
    if not exe:
        return False
    try:
        os.makedirs(profile_dir, exist_ok=True)
    except OSError:
        pass
    args = [exe, "--app=%s" % url,
            "--user-data-dir=%s" % profile_dir,
            "--no-first-run", "--no-default-browser-check",
            "--disable-features=Translate,AutofillServerCommunication",
            "--window-size=1440,920"]
    try:
        import subprocess
        logger.info("用应用模式打开窗口：%s", exe)
        proc = subprocess.Popen(args)
        proc.wait()          # 阻塞到用户关掉窗口
        return True
    except Exception:
        logger.exception("应用模式开窗失败")
        return False


def _msgbox(title: str, text: str) -> None:
    """弹一个系统消息框。打包成无控制台的窗口程序后，print 只会进日志文件，
    而用户正盯着一个什么都没有的屏幕 —— 这时候必须有个能看见的出口。
    用 Windows 自带的 MessageBoxW，零依赖；其它平台退化成打印。"""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, text, title, 0x40)  # MB_ICONINFORMATION
            return
        except Exception:
            logger.debug("弹出消息框失败", exc_info=True)
    print("\n  %s\n  %s" % (title, text))


def _open_window(url: str, on_closed) -> bool:
    """开原生窗口。返回 False 表示开不出来（调用方应退回浏览器）。"""
    try:
        import webview
    except Exception as exc:
        logger.warning("无法加载 WebView 组件：%s", exc)
        print("\n  ! 打不开程序窗口：%s" % exc)
        return False

    try:
        window = webview.create_window(
            "LCM90 视觉跟踪台", url,
            width=1440, height=920, min_size=(1100, 720),
            background_color="#050A14",   # 和页面同底色，避免加载时白屏闪一下
            # 关窗要先确认：跟踪中误点关闭，望远镜会在没人看着的情况下停机
            confirm_close=True,
            text_select=False)
        try:
            window.events.closed += on_closed
        except Exception:
            # 不同版本的事件 API 略有差异；订阅不上不影响主流程，
            # webview.start() 返回后照样会收尾。
            logger.debug("订阅窗口关闭事件失败", exc_info=True)
        webview.start(localization={
            "global.quitConfirmation": "确定要关闭吗？关闭后将自动断开基座与相机。",
            "global.ok": "确定", "global.cancel": "取消",
            "global.quit": "关闭",
        })      # 阻塞，直到用户关掉窗口
        return True
    except Exception as exc:
        logger.exception("创建程序窗口失败")
        print("\n  ! 程序窗口创建失败：%s" % exc)
        if sys.platform == "win32":
            print("    这台电脑可能缺少 Microsoft Edge WebView2 运行时。")
            print("    到 https://go.microsoft.com/fwlink/p/?LinkId=2124703 "
                  "装一次「Evergreen Bootstrapper」即可，装完重开本程序。")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--browser", action="store_true",
                    help="用浏览器打开而不是程序窗口")
    ap.add_argument("--no-window", action="store_true",
                    help="只启动服务，不开窗也不开浏览器")
    ap.add_argument("--no-browser", action="store_true",
                    help="等同 --no-window（保留旧参数名）")
    ap.add_argument("--verbose", action="store_true",
                    help="打印每一条网页请求，排查问题时才需要")
    ap.add_argument("--check", action="store_true",
                    help="自检 WebView 组件后退出")
    args = ap.parse_args()

    if args.check:
        return _check_webview()

    import yaml
    root = _app_root()
    try:
        with open(os.path.join(root, "config.yaml"), "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        print("x 找不到 config.yaml，它应该和本程序在同一个文件夹里。")
        return 1

    server_cfg = cfg.get("server", {}) or {}
    host = args.host or server_cfg.get("host", "127.0.0.1")
    want_port = args.port or int(server_cfg.get("port", 8300))
    port = _pick_port(host, want_port)
    url = "http://127.0.0.1:%d" % port

    headless = args.no_window or args.no_browser
    windowed = not headless and not args.browser

    print()
    print("  +" + "-" * 50 + "+")
    print("  |  LCM90 视觉跟踪台" + " " * 32 + "|")
    print("  +" + "-" * 50 + "+")
    print()
    if port != want_port:
        print("    端口 %d 被占用，已改用 %d。" % (want_port, port))
    print("    控制台地址   %s" % url)
    if windowed:
        print("    程序窗口即将打开。关掉窗口即退出程序。")
    elif args.browser:
        print("    浏览器会自动打开。要退出程序，直接关掉这个窗口。")
    print("    这里平时不会输出东西 —— 出问题时才会打印原因。")
    print()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="  %(levelname)s  %(message)s")

    import uvicorn
    # 直接导入 app 对象（不是字符串路径）—— PyInstaller 的静态分析
    # 看不见字符串里的模块名，打包 exe 会缺模块，启动即退。
    from app.main import app

    config = uvicorn.Config(app, host=host, port=port,
                            log_level="info" if args.verbose else "warning",
                            access_log=args.verbose, use_colors=False)
    server = uvicorn.Server(config)

    if headless:
        try:
            server.run()
        except OSError as exc:
            print("\n  x 启动失败：%s" % exc)
            return 1
        except KeyboardInterrupt:
            pass
        return 0

    # 窗口/浏览器模式：服务放后台线程，主线程留给 GUI
    # （GUI 框架几乎都要求自己待在主线程上）。
    failure: dict = {}

    def _serve():
        try:
            server.run()
        except Exception as exc:      # noqa: BLE001 —— 要带回主线程报给用户
            failure["exc"] = exc
            logger.exception("服务线程异常退出")

    thread = threading.Thread(target=_serve, name="uvicorn", daemon=True)
    thread.start()

    if not _wait_until_serving(host, port):
        exc = failure.get("exc")
        print("\n  x 服务没能起来%s" % ("：%s" % exc if exc else "（超时）"))
        print("    端口 %d 可能被占用，改 config.yaml 里的 port 再试。" % port)
        return 1

    def _on_closed():
        # 窗口一关就让服务退出，FastAPI 的 shutdown 会停录像、停基座。
        server.should_exit = True

    opened = False
    if windowed:
        # 首选 pywebview 的原生窗口；它依赖 .NET/pythonnet，打包后不一定能用。
        opened = _open_window(url, _on_closed)
        if not opened:
            # 次选：Edge/Chrome 的应用模式。外观同样是独立窗口，
            # 而且不碰 .NET，Windows 上几乎不可能失败。
            print("    改用应用模式窗口（Edge/Chrome）。")
            opened = _open_app_window(url, os.path.join(root, "data",
                                                        "window-profile"))
            if opened:
                _on_closed()      # 窗口已关，通知收尾
        if not opened:
            print("    连应用模式也起不来，已改用普通浏览器。")
            # 打包版没有控制台，上面这句只会进 app.log。用户面对的是一个
            # 没有任何界面、也不知道怎么关掉的后台进程 —— 必须弹窗告诉他。
            _msgbox("LCM90 视觉跟踪台",
                    "打不开独立窗口，已改用浏览器显示界面。\n\n"
                    "软件功能不受影响。要退出程序，请关掉浏览器页面后，"
                    "在任务管理器里结束 lcm90-tracker。")

    if not opened:
        try:
            webbrowser.open(url)
        except Exception:
            pass
        print("    要退出程序，直接关掉这个窗口。")
        try:
            while thread.is_alive():
                thread.join(timeout=0.5)
        except KeyboardInterrupt:
            pass

    # ---- 收尾 ----
    # 关键：**先**主动停一切，再去等 uvicorn。
    # 反过来（先 join 再指望 lifespan）是不行的：uvicorn 的优雅关闭要等所有
    # 响应结束，而 /video 是长连接；它没结束，shutdown 事件就一句都不跑，
    # 「停基座」被整个跳过 —— 关了窗望远镜还在转，这是最不能接受的故障。
    try:
        from app.main import shutdown_now
        shutdown_now()          # 幂等，lifespan 那边再调一次也无妨
    except Exception:
        logger.exception("主动收尾失败")
    server.should_exit = True
    thread.join(timeout=10.0)
    if thread.is_alive():
        logger.warning("服务线程没能在 10 秒内退出，但设备已停")
    return 0


if __name__ == "__main__":
    code = main()
    if code != 0 and getattr(sys, "frozen", False) and sys.stdout is not None:
        try:
            input("\n按回车键关闭...")
        except (EOFError, OSError):
            pass
    sys.exit(code)
