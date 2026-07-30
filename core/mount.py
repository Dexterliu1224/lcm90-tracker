"""基座驱动：NexStar 串口协议（LCM90 等）+ 仿真基座。

物理连接：手控器底部 RJ-22 口 → RS-232 → USB 串口线，9600 8N1。
协议要点（Celestron《NexStar Communication Protocol》v1.2）：
  * 所有指令的应答都以 '#' 结尾，读到 '#' 即为一帧结束。
  * 位置用「整圆的分数」编码：32 位精度 8 个 hex 字符。
  * 变速率转动走 'P' 透传通道，速率单位是 角秒/秒 × 4。

本模块自带 nudge()：用「变速率 × 固定时间」实现纯轴相对运动。
标定必须走 nudge 而不是 GoTo —— LCM 的 GoTo 会经过手控器的对准模型，
转出来的不是纯轴角度，标定矩阵会被污染。
"""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Dict, Optional, Tuple

try:
    import serial  # pyserial
except ImportError:  # 没装 pyserial 的机器也要能导入本模块（纯仿真场景）
    serial = None  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- 角度工具
# 契约没有公共 geometry 模块，为保持 mount.py 自包含，这里内联最小工具集。


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def wrap360(deg: float) -> float:
    """归一化到 [0, 360)。"""
    return deg % 360.0


def wrap180(deg: float) -> float:
    """归一化到 [-180, 180)，用于表示角度差。"""
    return (deg + 180.0) % 360.0 - 180.0


def angle_diff(target_deg: float, current_deg: float) -> float:
    """走最短路径的角度差 target - current，结果在 [-180, 180)。"""
    return wrap180(target_deg - current_deg)


# ---------------------------------------------------------------- 协议编码

# 'P' 透传通道的设备号
DEV_AZM = 16   # 方位/赤经 电机
DEV_ALT = 17   # 俯仰/赤纬 电机

# 变速率转动的子指令
MC_SET_POS_GUIDERATE = 6
MC_SET_NEG_GUIDERATE = 7

#: 协议里速率的编码上限（16 位），换算后约 4.55 度/秒
MAX_ENCODED_RATE = 0xFFFF


def deg_to_frac32(deg: float) -> str:
    """角度 → 32 位整圆分数的 8 位大写 hex。"""
    value = int(round(wrap360(deg) / 360.0 * 4294967296.0)) & 0xFFFFFFFF
    return "%08X" % value


def frac32_to_deg(token: str) -> float:
    return int(token, 16) / 4294967296.0 * 360.0


def to_signed_alt(deg: float) -> float:
    """基座回报的仰角是 0..360，>180 表示负仰角。"""
    deg = wrap360(deg)
    return deg - 360.0 if deg > 180.0 else deg


def encode_rate(rate_deg_s: float) -> Tuple[int, int, int]:
    """速率(度/秒) → (子指令, 高字节, 低字节)。

    协议单位是 角秒/秒 的 4 倍：trackRate = |rate| * 3600 * 4。
    """
    sub = MC_SET_POS_GUIDERATE if rate_deg_s >= 0 else MC_SET_NEG_GUIDERATE
    track_rate = int(round(abs(rate_deg_s) * 3600.0 * 4.0))
    track_rate = min(track_rate, MAX_ENCODED_RATE)
    return sub, (track_rate >> 8) & 0xFF, track_rate & 0xFF


# ---------------------------------------------------------------- 抽象基类


class MountError(RuntimeError):
    """基座通信或状态异常。"""


class MountBase:
    """经纬仪基座的最小能力集。

    实现类需保证 get_altaz / set_rate 线程安全（内部加锁），
    因为控制回路和 Web 状态查询会并发调用。
    """

    #: 机械最大转速（度/秒），由配置注入
    max_rate_deg_s: float = 2.5
    #: 低于此速率一律发 0（度/秒）—— 微小速率只会让电机在齿隙里蹭
    min_rate_deg_s: float = 2.0 / 3600.0
    alt_limit_min_deg: float = 5.0
    alt_limit_max_deg: float = 88.0

    # nudge 的收敛判据与节奏。类属性而非参数：契约固定了 nudge 签名，
    # 子类如需调整（例如齿隙更大的基座）改这里即可。
    _nudge_tol_deg: float = 0.01        # 残差小于此值算到位（约 36 角秒）
    _nudge_cruise_deg_s: float = 0.8    # 巡航速率：太快计时误差放大，太慢等待久
    _nudge_min_move_s: float = 0.25     # 单段最短运动时间，避免速率算得过大
    _nudge_settle_s: float = 0.4        # 停止后等电机减速、机械稳定的时间
    _nudge_max_passes: int = 4          # 修正轮数上限（齿隙/计时误差靠补走消化）

    # ------------------------------------------------------------ 待子类实现
    def connect(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    @property
    def connected(self) -> bool:
        raise NotImplementedError

    def get_altaz(self) -> Tuple[float, float]:
        """返回当前 (方位角 0..360, 仰角 -90..90)，单位度。"""
        raise NotImplementedError

    def set_rate(self, az_dps: float, alt_dps: float) -> None:
        """设置两轴转动速率（度/秒，正=方位增加/仰角升高）。"""
        raise NotImplementedError

    def stop(self) -> None:
        """两轴速率归零。"""
        raise NotImplementedError

    def is_slewing(self) -> bool:
        raise NotImplementedError

    # ------------------------------------------------------------ 通用实现
    def nudge(self, d_az: float, d_alt: float, timeout_s: float = 25.0) -> bool:
        """相对当前指向移动指定角度（度），阻塞到停稳。标定专用。

        返回是否成功。d_az 是**方位轴**角度，不是天球角距。

        实现方式是「变速率指令 × 固定时间」（rate*t=角度），走完停住、
        读回实际位置，误差大就再补一小段。不用 GoTo 是因为 LCM 的 GoTo
        有对准模型介入，标定需要的是纯轴运动。
        齿隙、加减速造成的偏差由后续修正轮消化，所以对首段精度要求不高。
        """
        deadline = time.monotonic() + max(0.5, float(timeout_s))
        try:
            az0, alt0 = self.get_altaz()
        except Exception as exc:
            logger.error("nudge 读取起始位置失败：%s", exc)
            return False

        target_az = wrap360(az0 + float(d_az))
        want_alt = alt0 + float(d_alt)
        # 仰角限位保护：夹住后仍然执行（能走多少走多少），
        # 但最终按「原始目标」判定成败，被夹掉的部分自然导致返回 False。
        target_alt = clamp(want_alt, self.alt_limit_min_deg, self.alt_limit_max_deg)
        if abs(target_alt - want_alt) > self._nudge_tol_deg:
            logger.warning(
                "nudge 目标仰角 %.3f° 超出限位 [%.1f, %.1f]，已夹到 %.3f°",
                want_alt, self.alt_limit_min_deg, self.alt_limit_max_deg, target_alt)

        try:
            for i in range(self._nudge_max_passes):
                az, alt = self.get_altaz()
                rem_az = angle_diff(target_az, az)
                rem_alt = target_alt - alt
                if abs(rem_az) <= self._nudge_tol_deg and abs(rem_alt) <= self._nudge_tol_deg:
                    break

                # 两轴同时走、同时到：取共同时长，各轴速率按剩余量分摊
                worst = max(abs(rem_az), abs(rem_alt))
                dur = max(worst / self._nudge_cruise_deg_s, self._nudge_min_move_s)
                budget = deadline - time.monotonic() - self._nudge_settle_s - 0.05
                if budget <= 0.05:
                    logger.warning("nudge 超时（%.1fs），剩余误差 az=%.4f° alt=%.4f°",
                                   timeout_s, rem_az, rem_alt)
                    break
                dur = min(dur, budget)

                rate_az = rem_az / dur
                rate_alt = rem_alt / dur
                # 速率低于驱动的死区会被当 0 发送，轴根本不动 —— 抬到死区之上
                if 0.0 < abs(rate_az) < self.min_rate_deg_s:
                    rate_az = math.copysign(self.min_rate_deg_s, rate_az)
                if 0.0 < abs(rate_alt) < self.min_rate_deg_s:
                    rate_alt = math.copysign(self.min_rate_deg_s, rate_alt)
                if abs(rem_az) <= self._nudge_tol_deg:
                    rate_az = 0.0
                if abs(rem_alt) <= self._nudge_tol_deg:
                    rate_alt = 0.0

                logger.debug("nudge 第 %d 段：rate=(%.4f, %.4f)°/s，t=%.3fs",
                             i + 1, rate_az, rate_alt, dur)
                self.set_rate(rate_az, rate_alt)
                # 分小段睡：仿真基座靠调用推进内部状态，长睡会让它丢失积分；
                # 对真实基座这只是几次无害的位置轮询
                t_end = time.monotonic() + dur
                while True:
                    left = t_end - time.monotonic()
                    if left <= 0:
                        break
                    time.sleep(min(0.05, left))
                    if left > 0.15:
                        self.get_altaz()
                self.stop()
                time.sleep(self._nudge_settle_s)
        except Exception as exc:
            logger.error("nudge 执行失败：%s", exc)
            try:
                self.stop()
            except Exception:
                pass
            return False

        try:
            self.stop()
            az1, alt1 = self.get_altaz()
        except Exception as exc:
            logger.error("nudge 收尾读取位置失败：%s", exc)
            return False

        # 成败按原始请求判定（含被限位夹掉的部分）
        err_az = angle_diff(wrap360(az0 + float(d_az)), az1)
        err_alt = (alt0 + float(d_alt)) - alt1
        ok = abs(err_az) <= self._nudge_tol_deg * 2 and abs(err_alt) <= self._nudge_tol_deg * 2
        logger.info("nudge(%.3f, %.3f) 完成：实际 (%.4f, %.4f)，残差 (%.4f, %.4f)，%s",
                    d_az, d_alt, angle_diff(az1, az0), alt1 - alt0,
                    err_az, err_alt, "成功" if ok else "失败")
        return ok

    def info(self) -> Dict[str, Any]:
        """基座信息。取不到的字段给 None，不要抛异常。"""
        return {"driver": None, "version": None, "aligned": None,
                "tracking_mode": None, "port": None}

    # with 语法糖：异常退出时保证先停轴再断开
    def __enter__(self) -> "MountBase":
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            self.stop()
        finally:
            self.close()


# ---------------------------------------------------------------- 真实基座


class NexStarMount(MountBase):
    """LCM90 等 NexStar 基座的串口驱动。"""

    def __init__(self, port: str, baudrate: int = 9600, timeout_s: float = 3.5,
                 max_rate_deg_s: float = 2.5, min_rate_arcsec_s: float = 2.0,
                 alt_limit_min_deg: float = 5.0, alt_limit_max_deg: float = 88.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self.max_rate_deg_s = max_rate_deg_s
        self.min_rate_deg_s = min_rate_arcsec_s / 3600.0
        self.alt_limit_min_deg = alt_limit_min_deg
        self.alt_limit_max_deg = alt_limit_max_deg

        self._ser: Optional["serial.Serial"] = None
        self._lock = threading.RLock()
        self._last_rate = (0.0, 0.0)

    # ------------------------------------------------------------ 连接
    def connect(self) -> None:
        if serial is None:
            raise MountError("未安装 pyserial，无法连接真实基座：pip install pyserial")
        with self._lock:
            if self._ser is not None and self._ser.is_open:
                return
            try:
                self._ser = serial.Serial(
                    self.port, self.baudrate, timeout=self.timeout_s,
                    write_timeout=self.timeout_s)
            except Exception as exc:
                raise MountError(
                    f"打开串口 {self.port} 失败：{exc}。"
                    f"请确认串口线已插好、没有被其它软件占用") from exc
            # 手控器上电后可能有残留字节，先清干净再握手
            time.sleep(0.2)
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
            if not self.echo_test():
                self.close()
                raise MountError(
                    f"{self.port} 上没有应答的 NexStar 手控器（回显测试失败）。"
                    f"请确认手控器已开机、选对了串口")

    def close(self) -> None:
        with self._lock:
            if self._ser is not None:
                try:
                    if self._ser.is_open:
                        self._ser.close()
                finally:
                    self._ser = None

    @property
    def connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    # ------------------------------------------------------------ 底层收发
    def _command(self, payload: bytes, expect_bytes: int = 0) -> bytes:
        """发一条指令并读回应答（不含结尾的 '#'）。

        expect_bytes 为应答里 '#' 之前的字节数；0 表示读到 '#' 为止。
        """
        with self._lock:
            if self._ser is None or not self._ser.is_open:
                raise MountError("串口未连接，请先连接基座")
            self._ser.reset_input_buffer()
            self._ser.write(payload)
            self._ser.flush()

            if expect_bytes:
                data = self._ser.read(expect_bytes + 1)
                if len(data) != expect_bytes + 1 or not data.endswith(b"#"):
                    raise MountError(f"指令 {payload!r} 应答异常: {data!r}")
                return data[:-1]

            buf = bytearray()
            deadline = time.time() + self.timeout_s
            while time.time() < deadline:
                ch = self._ser.read(1)
                if not ch:
                    break
                if ch == b"#":
                    return bytes(buf)
                buf.extend(ch)
            raise MountError(f"指令 {payload!r} 超时，已收到: {bytes(buf)!r}")

    def _passthrough(self, dev: int, sub: int, d1: int = 0, d2: int = 0,
                     d3: int = 0, resp_len: int = 0) -> bytes:
        """'P' 透传通道：直接和电机控制板对话。"""
        packet = bytes([ord("P"), 3, dev, sub, d1, d2, d3, resp_len])
        return self._command(packet, expect_bytes=resp_len)

    # ------------------------------------------------------------ 查询
    def echo_test(self, char: bytes = b"x") -> bool:
        """'K' 回显，用来确认链路通。"""
        try:
            return self._command(b"K" + char, expect_bytes=1) == char
        except MountError:
            return False

    def get_version(self) -> str:
        raw = self._command(b"V", expect_bytes=2)
        return "%d.%d" % (raw[0], raw[1])

    def is_aligned(self) -> bool:
        return self._command(b"J", expect_bytes=1) == b"\x01"

    def get_altaz(self) -> Tuple[float, float]:
        """'z' 高精度地平坐标查询。"""
        raw = self._command(b"z").decode("ascii")
        az_tok, alt_tok = raw.split(",")
        return wrap360(frac32_to_deg(az_tok)), to_signed_alt(frac32_to_deg(alt_tok))

    def is_slewing(self) -> bool:
        return self._command(b"L") == b"1"

    def get_tracking_mode(self) -> int:
        return self._command(b"t", expect_bytes=1)[0]

    def info(self) -> Dict[str, Any]:
        """逐项查询，任何一项失败都不影响其它项 —— 界面拿到多少显示多少。"""
        result: Dict[str, Any] = {"driver": "nexstar", "version": None,
                                  "aligned": None, "tracking_mode": None,
                                  "port": self.port}
        if not self.connected:
            return result
        for key, fn in (("version", self.get_version),
                        ("aligned", self.is_aligned),
                        ("tracking_mode", self.get_tracking_mode)):
            try:
                result[key] = fn()
            except Exception as exc:
                logger.debug("查询 %s 失败：%s", key, exc)
        return result

    # ------------------------------------------------------------ 动作
    def set_tracking_mode(self, mode: int) -> None:
        self._command(bytes([ord("T"), mode]))

    def goto_altaz(self, az_deg: float, alt_deg: float) -> None:
        """高精度 GoTo。注意：会经过对准模型，标定请用 nudge()。"""
        alt_deg = clamp(alt_deg, self.alt_limit_min_deg, self.alt_limit_max_deg)
        cmd = b"b" + ("%s,%s" % (deg_to_frac32(az_deg),
                                 deg_to_frac32(alt_deg))).encode("ascii")
        self._command(cmd)

    def cancel_goto(self) -> None:
        self._command(b"M")

    def sync_altaz(self, az_deg: float, alt_deg: float) -> None:
        cmd = b"s" + ("%s,%s" % (deg_to_frac32(az_deg),
                                 deg_to_frac32(alt_deg))).encode("ascii")
        self._command(cmd)

    def set_rate(self, az_dps: float, alt_dps: float) -> None:
        """变速率转动 —— 跟踪回路每个周期调用一次。

        小于 min_rate 的速率一律发 0：LCM 这类基座有齿隙，
        持续下发微小速率只会让电机在间隙里来回蹭，反而抖动。
        """
        az_rate = clamp(float(az_dps), -self.max_rate_deg_s, self.max_rate_deg_s)
        alt_rate = clamp(float(alt_dps), -self.max_rate_deg_s, self.max_rate_deg_s)
        if abs(az_rate) < self.min_rate_deg_s:
            az_rate = 0.0
        if abs(alt_rate) < self.min_rate_deg_s:
            alt_rate = 0.0

        sub, hi, lo = encode_rate(az_rate)
        self._passthrough(DEV_AZM, sub, hi, lo)
        sub, hi, lo = encode_rate(alt_rate)
        self._passthrough(DEV_ALT, sub, hi, lo)
        self._last_rate = (az_rate, alt_rate)

    @property
    def last_rate(self) -> Tuple[float, float]:
        return self._last_rate

    def stop(self) -> None:
        # 安全兜底路径：两条停轴指令必须**各自**尝试 —— 若共用一个 try，
        # 方位轴那条因串口偶发超时抛异常时，俯仰轴的停轴指令根本发不出去，
        # 望远镜会带着残余速率继续转。账目也只在真正全停后才清零，
        # 否则上层读 last_rate 会误以为已经停住。
        errs = []
        for dev in (DEV_AZM, DEV_ALT):
            try:
                self._passthrough(dev, MC_SET_POS_GUIDERATE, 0, 0)
            except Exception as exc:
                errs.append(exc)
        if errs:
            raise MountError("停轴指令部分失败（望远镜可能仍在转动）：%s" % errs)
        self._last_rate = (0.0, 0.0)


# ---------------------------------------------------------------- 仿真基座


class SimMount(MountBase):
    """仿真基座：没有硬件时跑通全链路。

    刻意模拟了 LCM90 的三个真实缺陷，这样在办公室调出来的参数
    拿到现场不会完全跑飞：
      1. 速率上限（约 2.5°/s）
      2. 电机加速度有限（不能瞬间换向）
      3. 齿隙（换向时有一段空行程）
    """

    def __init__(self, max_rate_deg_s: float = 2.5, min_rate_arcsec_s: float = 2.0,
                 accel_deg_s2: float = 6.0, backlash_deg: float = 0.02,
                 alt_limit_min_deg: float = 5.0, alt_limit_max_deg: float = 88.0,
                 start_az: float = 180.0, start_alt: float = 35.0):
        self.max_rate_deg_s = max_rate_deg_s
        self.min_rate_deg_s = min_rate_arcsec_s / 3600.0
        self.accel_deg_s2 = accel_deg_s2
        self.backlash_deg = backlash_deg
        self.alt_limit_min_deg = alt_limit_min_deg
        self.alt_limit_max_deg = alt_limit_max_deg

        self._az = wrap360(start_az)
        self._alt = start_alt
        self._cmd_rate = [0.0, 0.0]      # 指令速率
        self._act_rate = [0.0, 0.0]      # 实际速率（受加速度限制）
        self._backlash_state = [0.0, 0.0]
        self._sign = [0, 0]
        self._connected = False
        self._goto_target: Optional[Tuple[float, float]] = None
        self._lock = threading.RLock()
        self._t_last = time.time()

    # ------------------------------------------------------------ 生命周期
    def connect(self) -> None:
        self._connected = True
        self._t_last = time.time()

    def close(self) -> None:
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------ 物理推进
    def _advance(self) -> None:
        """把仿真状态推进到当前时刻。每次读写状态前调用。"""
        now = time.time()
        dt = now - self._t_last
        self._t_last = now
        if dt <= 0 or dt > 1.0:      # 长时间没调用就不做大跨步积分
            dt = min(max(dt, 0.0), 0.05)

        if self._goto_target is not None:
            self._step_goto(dt)
            return

        for axis in (0, 1):
            target = self._cmd_rate[axis]
            actual = self._act_rate[axis]
            # 加速度限制：换向/起停都要经过加减速斜坡
            max_delta = self.accel_deg_s2 * dt
            actual += clamp(target - actual, -max_delta, max_delta)
            self._act_rate[axis] = actual

            step = float(actual) * dt
            # 齿隙：换向后先走空行程，轴不动
            new_sign = 1 if step > 0 else (-1 if step < 0 else 0)
            if new_sign != 0 and new_sign != self._sign[axis] and self._sign[axis] != 0:
                self._backlash_state[axis] = self.backlash_deg
            if new_sign != 0:
                self._sign[axis] = new_sign
            if self._backlash_state[axis] > 0:
                consumed = min(self._backlash_state[axis], abs(step))
                self._backlash_state[axis] -= consumed
                step -= new_sign * consumed

            if axis == 0:
                self._az = wrap360(self._az + step)
            else:
                # 允许略微越过软限位，模拟真机限位不是硬墙
                self._alt = clamp(self._alt + step,
                                  self.alt_limit_min_deg - 5,
                                  self.alt_limit_max_deg + 5)

    def _step_goto(self, dt: float) -> None:
        target_az, target_alt = self._goto_target  # type: ignore[misc]
        d_az = angle_diff(target_az, self._az)
        d_alt = target_alt - self._alt
        step = self.max_rate_deg_s * dt

        done_az = abs(d_az) <= step
        done_alt = abs(d_alt) <= step
        self._az = wrap360(target_az if done_az
                           else self._az + step * (1 if d_az > 0 else -1))
        self._alt = target_alt if done_alt \
            else self._alt + step * (1 if d_alt > 0 else -1)
        if done_az and done_alt:
            self._goto_target = None

    # ------------------------------------------------------------ 接口
    def get_altaz(self) -> Tuple[float, float]:
        with self._lock:
            self._advance()
            return self._az, self._alt

    def goto_altaz(self, az_deg: float, alt_deg: float) -> None:
        with self._lock:
            self._advance()
            self._cmd_rate = [0.0, 0.0]
            self._act_rate = [0.0, 0.0]
            self._goto_target = (wrap360(az_deg),
                                 clamp(alt_deg, self.alt_limit_min_deg,
                                       self.alt_limit_max_deg))

    def is_slewing(self) -> bool:
        with self._lock:
            self._advance()
            return self._goto_target is not None

    def set_rate(self, az_dps: float, alt_dps: float) -> None:
        with self._lock:
            self._advance()
            self._goto_target = None
            # float() 是必需的：上游可能传进 numpy 标量，
            # 会让符号判断变成 np.bool_ 运算而出错
            az = clamp(float(az_dps), -self.max_rate_deg_s, self.max_rate_deg_s)
            alt = clamp(float(alt_dps), -self.max_rate_deg_s, self.max_rate_deg_s)
            if abs(az) < self.min_rate_deg_s:
                az = 0.0
            if abs(alt) < self.min_rate_deg_s:
                alt = 0.0
            self._cmd_rate = [az, alt]

    @property
    def last_rate(self) -> Tuple[float, float]:
        return (self._cmd_rate[0], self._cmd_rate[1])

    def stop(self) -> None:
        with self._lock:
            self._advance()
            self._goto_target = None
            self._cmd_rate = [0.0, 0.0]

    def sync_altaz(self, az_deg: float, alt_deg: float) -> None:
        with self._lock:
            self._advance()
            self._az = wrap360(az_deg)
            self._alt = alt_deg

    def info(self) -> Dict[str, Any]:
        return {"driver": "simulator", "version": "sim-1.0", "aligned": True,
                "tracking_mode": 0, "port": None}


# ---------------------------------------------------------------- 工厂


def build_mount(cfg: Dict[str, Any]) -> MountBase:
    """cfg 为 config.yaml 的 mount 段；driver 取值 nexstar / simulator。"""
    driver = str(cfg.get("driver", "simulator")).strip().lower()

    if driver == "nexstar":
        port = str(cfg.get("port", "") or "").strip()
        if not port:
            raise MountError(
                "nexstar 驱动需要串口号：请在界面扫描并选择串口，"
                "或在 config.yaml 的 mount.port 填写（例如 /dev/tty.usbserial-XXX）")
        return NexStarMount(
            port=port,
            baudrate=int(cfg.get("baudrate", 9600)),
            timeout_s=float(cfg.get("timeout_s", 3.5)),
            max_rate_deg_s=float(cfg.get("max_rate_deg_s", 2.5)),
            min_rate_arcsec_s=float(cfg.get("min_rate_arcsec_s", 2.0)),
            alt_limit_min_deg=float(cfg.get("alt_limit_min_deg", 5.0)),
            alt_limit_max_deg=float(cfg.get("alt_limit_max_deg", 88.0)))

    if driver == "simulator":
        return SimMount(
            max_rate_deg_s=float(cfg.get("max_rate_deg_s", 2.5)),
            min_rate_arcsec_s=float(cfg.get("min_rate_arcsec_s", 2.0)),
            accel_deg_s2=float(cfg.get("accel_deg_s2", 6.0)),
            backlash_deg=float(cfg.get("backlash_deg", 0.02)),
            alt_limit_min_deg=float(cfg.get("alt_limit_min_deg", 5.0)),
            alt_limit_max_deg=float(cfg.get("alt_limit_max_deg", 88.0)),
            start_az=float(cfg.get("start_az", 180.0)),
            start_alt=float(cfg.get("start_alt", 35.0)))

    raise MountError(
        f"未知的基座驱动 '{driver}'：mount.driver 只能是 nexstar 或 simulator")


# ---------------------------------------------------------------- 自检


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    m = SimMount()
    m.connect()
    assert m.connected, "SimMount 连接失败"

    # 1) info 字段齐全
    inf = m.info()
    need = {"driver", "version", "aligned", "tracking_mode", "port"}
    assert need <= set(inf), f"info() 缺字段：{need - set(inf)}"
    logger.info("info() = %s", inf)

    # 2) 正向 nudge：方位增加约 0.3°
    az0, alt0 = m.get_altaz()
    ok = m.nudge(0.3, 0.0)
    az1, alt1 = m.get_altaz()
    moved = angle_diff(az1, az0)
    assert ok, "nudge(+0.3, 0) 报告失败"
    assert abs(moved - 0.3) < 0.03, f"方位实际移动 {moved:.4f}°，与 0.3° 偏差过大"
    assert abs(alt1 - alt0) < 0.02, f"仰角不该动却动了 {alt1 - alt0:.4f}°"

    # 3) 反向 nudge：验证齿隙被修正轮补掉
    ok2 = m.nudge(-0.3, 0.0)
    az2, _ = m.get_altaz()
    moved2 = angle_diff(az2, az1)
    assert ok2, "nudge(-0.3, 0) 报告失败"
    assert abs(moved2 + 0.3) < 0.03, f"反向方位实际移动 {moved2:.4f}°，应约 -0.3°"

    # 4) 两轴同时 nudge
    ok3 = m.nudge(0.2, -0.15)
    az3, alt3 = m.get_altaz()
    assert ok3, "nudge(0.2, -0.15) 报告失败"
    assert abs(angle_diff(az3, az2) - 0.2) < 0.03
    assert abs((alt3 - alt1) + 0.15) < 0.03

    # 5) set_rate / stop 基本行为
    m.set_rate(0.5, 0.0)
    time.sleep(0.3)
    m.get_altaz()          # 推进仿真
    m.stop()
    assert not m.is_slewing()
    assert m.last_rate == (0.0, 0.0)

    # 6) 工厂：仿真 / 未知驱动 / nexstar 缺串口
    m2 = build_mount({"driver": "simulator", "start_alt": 40.0})
    assert isinstance(m2, SimMount) and abs(m2.get_altaz()[1] - 40.0) < 1e-6
    try:
        build_mount({"driver": "什么鬼"})
        raise AssertionError("未知驱动应抛 MountError")
    except MountError:
        pass
    try:
        build_mount({"driver": "nexstar", "port": ""})
        raise AssertionError("nexstar 缺串口应抛 MountError")
    except MountError:
        pass

    # 7) 协议编码自洽性（不需要硬件）
    assert deg_to_frac32(180.0) == "80000000"
    assert abs(frac32_to_deg("80000000") - 180.0) < 1e-6
    assert to_signed_alt(350.0) == -10.0
    assert encode_rate(1.0) == (MC_SET_POS_GUIDERATE, 0x38, 0x40)   # 14400
    assert encode_rate(-1.0)[0] == MC_SET_NEG_GUIDERATE

    m.close()
    assert not m.connected
    logger.info("mount.py 自检全部通过")
