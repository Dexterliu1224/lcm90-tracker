# -*- coding: utf-8 -*-
"""录像：把带标注的画面写成视频文件。

设计要点：
  * **独立线程按固定间隔取帧**。VideoWriter 的帧率是写死在文件头里的，
    而采集帧率会随曝光/CPU 波动；如果来一帧写一帧，录出来的视频回放
    时快时慢。定频取帧（拿不到新帧就重复上一帧）保证时间轴是准的。
  * **绝不让录像拖垮控制回路**。取帧走的是 annotated_frame() 的副本，
    写文件在自己的线程里，写失败只停录像，不碰跟踪。
  * **有最长时长上限**。教学现场忘了点停止是常态，几小时下来能把 C 盘
    写满 —— 到点自动停，并在状态里说明原因。
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: 编码优先级：先试 mp4（Windows 双击就能放），不行退 AVI/MJPG。
#: MJPG 文件大得多，但 OpenCV 自带、没有任何编解码器依赖，是最后的保底。
_CODECS: Tuple[Tuple[str, str], ...] = (
    ("mp4v", ".mp4"),
    ("MJPG", ".avi"),
)


class RecorderError(Exception):
    """录像操作失败，message 直接给用户看。"""


class Recorder:
    """把 frame_provider() 返回的画面定频写进视频文件。线程安全。"""

    def __init__(self, outdir: str, fps: float = 15.0,
                 max_minutes: float = 60.0):
        self._outdir = outdir
        self._fps = max(1.0, float(fps))
        self._max_s = max(0.0, float(max_minutes) * 60.0)

        self._lock = threading.RLock()
        self._writer: Optional[cv2.VideoWriter] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        self._path: Optional[str] = None
        self._size: Optional[Tuple[int, int]] = None   # (w, h)
        self._frames = 0
        self._t0 = 0.0
        self._note = ""          # 自动停止的原因，给界面显示

    # ---------------- 查询 ----------------

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._writer is not None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            rec = self._writer is not None
            path = self._path
            frames = self._frames
            seconds = (time.monotonic() - self._t0) if rec else 0.0
            note = self._note
        size = 0
        if path and os.path.exists(path):
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
        return {"recording": rec,
                "file": os.path.basename(path) if path else None,
                "path": path,
                "seconds": round(seconds, 1),
                "frames": frames,
                "size_mb": round(size / 1048576.0, 1),
                "note": note}

    # ---------------- 控制 ----------------

    def start(self, frame_provider: Callable[[], Optional[np.ndarray]],
              name_hint: str = "") -> Dict[str, Any]:
        """开始录像。frame_provider 返回 BGR uint8 帧或 None。"""
        with self._lock:
            if self._writer is not None:
                raise RecorderError("已经在录像中了。")

        frame = frame_provider()
        if frame is None:
            raise RecorderError("现在没有画面，先开启视频源再录像。")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RecorderError("画面格式不是彩色图，无法录像。")
        h, w = frame.shape[:2]

        try:
            os.makedirs(self._outdir, exist_ok=True)
        except OSError as exc:
            raise RecorderError("建不了录像文件夹 %s：%s" % (self._outdir, exc))

        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = "%s-%s" % (stamp, name_hint) if name_hint else stamp
        # 同一秒里连开两次录像会撞名，第二次直接把第一次覆盖掉。
        # 概率不高，但丢的是已经录好的素材，加个序号一劳永逸。
        if any(os.path.exists(os.path.join(self._outdir, base + ext))
               for _, ext in _CODECS):
            for n in range(2, 100):
                alt = "%s_%d" % (base, n)
                if not any(os.path.exists(os.path.join(self._outdir, alt + ext))
                           for _, ext in _CODECS):
                    base = alt
                    break

        writer = None
        path = None
        tried = []
        for fourcc_name, ext in _CODECS:
            candidate = os.path.join(self._outdir, base + ext)
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
                cand_writer = cv2.VideoWriter(candidate, fourcc, self._fps,
                                              (w, h))
            except Exception as exc:
                tried.append("%s(%s)" % (fourcc_name, exc))
                continue
            # isOpened 是唯一可靠的判据：VideoWriter 构造从不抛异常，
            # 编解码器缺失时它安安静静地返回一个写不进去的对象。
            if cand_writer.isOpened():
                writer, path = cand_writer, candidate
                logger.info("录像编码器 %s，输出 %s", fourcc_name, candidate)
                break
            cand_writer.release()
            tried.append(fourcc_name)
            try:
                if os.path.exists(candidate):
                    os.remove(candidate)     # 留下 0 字节的残file很误导人
            except OSError:
                pass

        if writer is None:
            raise RecorderError(
                "系统里没有可用的视频编码器（试过 %s）。"
                "装一个 K-Lite 之类的解码包，或改用别的机器录。"
                % "、".join(tried))

        with self._lock:
            self._writer = writer
            self._path = path
            self._size = (w, h)
            self._frames = 0
            self._t0 = time.monotonic()
            self._note = ""
            self._stop_evt.clear()
            self._thread = threading.Thread(
                target=self._loop, args=(frame_provider,),
                name="recorder", daemon=True)
            self._thread.start()

        return self.status()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if self._writer is None:
                return self.status()
            thread = self._thread
        self._stop_evt.set()
        if thread is not None:
            thread.join(timeout=3.0)
        return self._finalize()

    def _finalize(self) -> Dict[str, Any]:
        """关闭 writer 并返回最终状态。可重复调用。"""
        with self._lock:
            writer = self._writer
            self._writer = None
            self._thread = None
            frames = self._frames
            path = self._path
            note = self._note
        if writer is not None:
            try:
                writer.release()
            except Exception:
                logger.exception("关闭录像文件失败")
        # 一帧没写成的空文件没有意义，删掉免得用户去点一个坏文件
        if path and frames == 0:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
            with self._lock:
                self._path = None
        st = self.status()
        st["note"] = note
        logger.info("录像结束：%s，%d 帧", st.get("file"), frames)
        return st

    def shutdown(self) -> None:
        """程序退出时调用：绝不能让视频文件停在没写完的状态。"""
        try:
            self.stop()
        except Exception:
            logger.exception("关闭录像器失败")

    # ---------------- 线程 ----------------

    def _loop(self, frame_provider: Callable[[], Optional[np.ndarray]]) -> None:
        period = 1.0 / self._fps
        next_t = time.monotonic()
        last: Optional[np.ndarray] = None
        while not self._stop_evt.is_set():
            next_t += period
            try:
                frame = frame_provider()
            except Exception:
                logger.exception("录像取帧失败")
                frame = None

            if frame is not None:
                last = frame
            elif last is None:
                # 一帧都还没拿到，先等着，别往文件里写垃圾
                if self._stop_evt.wait(max(0.0, next_t - time.monotonic())):
                    break
                continue

            img = last
            with self._lock:
                writer = self._writer
                size = self._size
            if writer is None or size is None:
                break
            # 换了相机分辨率就变了，而视频文件的尺寸是固定的：
            # 缩放而不是中断录像，否则换个源就断片。
            if (img.shape[1], img.shape[0]) != size:
                try:
                    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                except Exception:
                    logger.exception("录像缩放失败")
                    continue
            try:
                writer.write(img)
            except Exception:
                logger.exception("写录像帧失败，已停止录像")
                with self._lock:
                    self._note = "写文件出错，录像已中断（磁盘满或文件被占用？）"
                break

            with self._lock:
                self._frames += 1
                elapsed = time.monotonic() - self._t0
            if self._max_s and elapsed >= self._max_s:
                limit = ("%.0f 分钟" % (self._max_s / 60.0)
                         if self._max_s >= 60 else "%.0f 秒" % self._max_s)
                with self._lock:
                    self._note = "已到最长录像时长 %s，自动停止。" % limit
                logger.info("录像到达时长上限，自动停止")
                break

            sleep = next_t - time.monotonic()
            if sleep > 0:
                if self._stop_evt.wait(sleep):
                    break
            else:
                next_t = time.monotonic()   # 落后了就重新对表，别追帧

        # 线程自己退出（写错/到时长）时也要收尾，否则文件一直开着
        if not self._stop_evt.is_set():
            self._finalize()


# ======================================================================
# 自检：录一小段再读回来验证
# ======================================================================

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("recorder.selftest")

    W, H = 320, 240
    outdir = tempfile.mkdtemp()

    counter = {"n": 0}

    def provider():
        counter["n"] += 1
        img = np.zeros((H, W, 3), np.uint8)
        # 每帧画一个移动的方块，回读时能确认不是全黑
        x = (counter["n"] * 7) % (W - 40)
        cv2.rectangle(img, (x, 100), (x + 40, 140), (0, 200, 255), -1)
        return img

    rec = Recorder(outdir, fps=20.0, max_minutes=60.0)
    assert not rec.recording
    st = rec.status()
    assert st["recording"] is False and st["file"] is None

    st = rec.start(provider)
    assert rec.recording, "start 之后应处于录像状态"
    assert st["file"], "状态里应带文件名"
    log.info("开始录像：%s", st["file"])

    try:
        rec.start(provider)
        raise AssertionError("重复 start 应报错")
    except RecorderError as exc:
        assert "已经在录像" in str(exc)

    time.sleep(1.2)
    mid = rec.status()
    assert mid["frames"] > 5, "1.2 秒内应写进若干帧，实际 %d" % mid["frames"]
    assert mid["seconds"] >= 1.0

    final = rec.stop()
    assert not rec.recording, "stop 之后不应还在录像"
    path = final["path"]
    assert path and os.path.exists(path), "录像文件应存在"
    assert final["frames"] > 5
    log.info("停止录像：%d 帧，%.1f MB", final["frames"], final["size_mb"])

    # ---- 回读验证：文件真的能打开、帧数对得上、画面不是全黑 ----
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), "录出来的文件打不开"
    got = 0
    nonblack = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        got += 1
        if float(np.mean(img)) > 1.0:
            nonblack += 1
    cap.release()
    assert got > 5, "回读只拿到 %d 帧" % got
    assert nonblack > 5, "回读的画面全黑，说明写进去的是空帧"
    log.info("回读验证：%d 帧，其中 %d 帧有画面", got, nonblack)

    # ---- 没有画面时 start 应当明确报错 ----
    rec2 = Recorder(outdir, fps=20.0)
    try:
        rec2.start(lambda: None)
        raise AssertionError("没有画面时 start 应报错")
    except RecorderError as exc:
        assert "没有画面" in str(exc)
    log.info("无画面时拒绝录像：正确")

    # ---- 空录像（一帧没写）不该留下坏文件 ----
    before = set(os.listdir(outdir))
    rec3 = Recorder(outdir, fps=20.0)
    rec3.start(provider)
    rec3.stop()
    # 这次一定写进了帧，文件应该留下
    assert len(set(os.listdir(outdir)) - before) == 1

    # ---- 到达时长上限自动停止 ----
    rec4 = Recorder(outdir, fps=30.0, max_minutes=0.02)   # 1.2 秒
    rec4.start(provider)
    deadline = time.monotonic() + 6.0
    while rec4.recording and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not rec4.recording, "到达时长上限后应自动停止"
    assert "自动停止" in rec4.status()["note"], \
        "自动停止要说明原因，实际 %r" % rec4.status()["note"]
    log.info("时长上限自动停止：%s", rec4.status()["note"])

    # ---- 画面尺寸变了不该中断录像 ----
    flip = {"big": False}

    def changing_provider():
        flip["big"] = not flip["big"]
        h, w = (H * 2, W * 2) if flip["big"] else (H, W)
        return np.full((h, w, 3), 60, np.uint8)

    rec5 = Recorder(outdir, fps=20.0)
    rec5.start(changing_provider)
    time.sleep(0.8)
    st5 = rec5.stop()
    assert st5["frames"] > 5, "分辨率变化时录像不该中断，实际只有 %d 帧" % st5["frames"]
    log.info("分辨率切换不中断：%d 帧", st5["frames"])

    log.info("recorder selftest OK")
