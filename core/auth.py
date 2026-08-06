# -*- coding: utf-8 -*-
"""登录与账号：口令散列存盘 + 内存会话。

为什么不存明文口令：这台电脑通常是教室里公用的，users.json 谁都能打开。
散列用 PBKDF2-HMAC-SHA256 + 每人独立随机盐，标准库就有，不引新依赖。
迭代次数取 20 万 —— 登录一次多花几十毫秒没人察觉，但离线爆破的代价
会被放大同样的倍数。

会话只放在内存里，程序一重启全部失效。这是**有意**的：单机教学软件，
宁可让老师重登一次，也不想把长期有效的凭据落到磁盘上。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_ITERATIONS = 200_000
_SALT_BYTES = 16
_ALGO = "pbkdf2_sha256"

#: 出厂账号。首次启动没有 users.json 时自动建，登录后界面会一直提醒改密码。
DEFAULT_USER = "admin"
DEFAULT_PASSWORD = "admin"

#: 会话有效期。教学一节课 45 分钟，8 小时足够覆盖一整天连续使用。
SESSION_TTL_S = 8 * 3600

#: 用户名规则：字母数字加下划线连字符，2~24 位。限制的是**用户名**不是
#: 口令 —— 用户名会进 JSON 的键、进界面文本，放开了容易出奇怪的问题。
_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{2,24}$")

MIN_PASSWORD_LEN = 4


class AuthError(Exception):
    """账号操作失败，message 直接给用户看。"""


def _hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt,
                             _ITERATIONS)
    return dk.hex()


class UserStore:
    """账号库。所有方法线程安全，落盘为 data/users.json。

    角色只有两种：admin 能管账号，user 只能用软件、改自己的密码。
    刻意不做更细的权限 —— 这是教室里的单机软件，不是多租户系统。
    """

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.RLock()
        self._users: Dict[str, Dict[str, Any]] = {}
        self._deleted: Dict[str, float] = {}     # 账号名 → 删除时刻（墓碑）
        self._load()

    # ---------------- 落盘 ----------------

    def _load(self) -> None:
        if not self._path or not os.path.exists(self._path):
            self._bootstrap()
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            users = data.get("users") or {}
            if not isinstance(users, dict) or not users:
                raise ValueError("users 段为空")
            self._users = users
            self._deleted = dict(data.get("deleted") or {})
        except Exception:
            # 账号文件损坏时**不能**直接放行，也不能把人锁在门外：
            # 重建出厂账号并留下日志，管理员至少还能进得去。
            logger.exception("读取账号文件失败，已重建出厂账号：%s", self._path)
            self._bootstrap()
            return
        # 兜底：文件里一个管理员都没有，等于谁也管不了账号了
        if not any(u.get("role") == "admin" for u in self._users.values()):
            logger.warning("账号文件里没有管理员，已补建 %s", DEFAULT_USER)
            self._users[DEFAULT_USER] = self._make_record(
                DEFAULT_PASSWORD, "admin", is_default=True)
            self._save()

    def _bootstrap(self) -> None:
        self._users = {DEFAULT_USER: self._make_record(
            DEFAULT_PASSWORD, "admin", is_default=True)}
        self._save()
        logger.info("已创建出厂账号 %s / %s，请登录后立即修改密码",
                    DEFAULT_USER, DEFAULT_PASSWORD)

    def _save(self) -> None:
        if not self._path:
            return
        try:
            parent = os.path.dirname(self._path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            # 先写临时文件再原子替换：直接覆写时如果断电/崩溃，
            # 账号文件会变成半截 JSON，下次启动谁都登不进去。
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"version": 1, "users": self._users,
                           "deleted": self._deleted}, fh,
                          ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)
        except Exception:
            logger.exception("保存账号文件失败：%s", self._path)

    @staticmethod
    def _make_record(password: str, role: str,
                     is_default: bool = False) -> Dict[str, Any]:
        salt = secrets.token_bytes(_SALT_BYTES)
        return {"algo": _ALGO, "iterations": _ITERATIONS,
                "salt": salt.hex(), "hash": _hash_password(password, salt),
                "role": role, "is_default_password": bool(is_default),
                # updated_at 是云端多设备合并的依据：同名账号谁的时间戳新用谁的。
                # 少了它，两台设备各改各的时会互相覆盖。
                "created_at": time.time(), "updated_at": time.time()}

    # ---------------- 查询 ----------------

    def exists(self, username: str) -> bool:
        with self._lock:
            return username in self._users

    def role_of(self, username: str) -> str:
        with self._lock:
            rec = self._users.get(username) or {}
            return str(rec.get("role", "user"))

    def uses_default_password(self, username: str) -> bool:
        with self._lock:
            rec = self._users.get(username) or {}
            return bool(rec.get("is_default_password"))

    def list_users(self) -> List[Dict[str, Any]]:
        """给界面看的账号列表，**绝不**包含盐和散列。"""
        with self._lock:
            return sorted(
                ({"username": name,
                  "role": str(rec.get("role", "user")),
                  "is_default_password": bool(rec.get("is_default_password")),
                  "created_at": rec.get("created_at")}
                 for name, rec in self._users.items()),
                key=lambda u: (u["role"] != "admin", u["username"]))

    # ---------------- 认证 ----------------

    def verify(self, username: str, password: str) -> bool:
        with self._lock:
            rec = self._users.get(username)
        if rec is None:
            # 用户不存在时也走一遍散列计算：直接返回会让"用户名是否存在"
            # 从响应时间上泄漏出去。
            _hash_password(password, secrets.token_bytes(_SALT_BYTES))
            return False
        try:
            salt = bytes.fromhex(str(rec.get("salt", "")))
            expect = str(rec.get("hash", ""))
            iterations = int(rec.get("iterations", _ITERATIONS))
        except Exception:
            logger.exception("账号 %s 的记录格式有问题", username)
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt,
                                 iterations)
        # compare_digest 而不是 == ：定时攻击防护，代价为零
        return hmac.compare_digest(dk.hex(), expect)

    # ---------------- 修改 ----------------

    @staticmethod
    def _check_password(password: str) -> None:
        if not isinstance(password, str) or len(password) < MIN_PASSWORD_LEN:
            raise AuthError("新密码至少要 %d 位。" % MIN_PASSWORD_LEN)

    @staticmethod
    def _check_username(username: str) -> None:
        if not isinstance(username, str) or not _NAME_RE.match(username):
            raise AuthError("用户名只能用字母、数字、下划线和减号，长度 2~24 位。")

    def change_password(self, username: str, old: str, new: str) -> None:
        if not self.verify(username, old):
            raise AuthError("原密码不对。")
        self._check_password(new)
        if new == old:
            raise AuthError("新密码不能和原密码一样。")
        with self._lock:
            rec = self._users.get(username)
            if rec is None:
                raise AuthError("账号不存在。")
            salt = secrets.token_bytes(_SALT_BYTES)
            rec.update(algo=_ALGO, iterations=_ITERATIONS, salt=salt.hex(),
                       hash=_hash_password(new, salt),
                       is_default_password=False, updated_at=time.time())
            self._save()
        logger.info("账号 %s 已修改密码", username)

    def add_user(self, username: str, password: str, role: str = "user") -> None:
        self._check_username(username)
        self._check_password(password)
        if role not in ("admin", "user"):
            raise AuthError("角色只能是 admin 或 user。")
        with self._lock:
            if username in self._users:
                raise AuthError("用户名「%s」已经存在。" % username)
            self._users[username] = self._make_record(password, role)
            # 清掉墓碑：删过 teacher 又重建 teacher，墓碑还留着的话，
            # 下次云端同步会把刚建好的账号又删掉。
            self._deleted.pop(username, None)
            self._save()
        logger.info("新建账号 %s（%s）", username, role)

    def reset_password(self, username: str, password: str) -> None:
        """管理员替别人重置密码（不需要原密码）。"""
        self._check_password(password)
        with self._lock:
            rec = self._users.get(username)
            if rec is None:
                raise AuthError("账号「%s」不存在。" % username)
            salt = secrets.token_bytes(_SALT_BYTES)
            rec.update(algo=_ALGO, iterations=_ITERATIONS, salt=salt.hex(),
                       hash=_hash_password(password, salt),
                       is_default_password=False, updated_at=time.time())
            self._save()
        logger.info("管理员重置了账号 %s 的密码", username)

    def export_for_sync(self) -> Dict[str, Any]:
        """导出给云端的完整账号数据（含散列，但本来就不是明文）。"""
        with self._lock:
            return {"version": 1, "users": dict(self._users),
                    "deleted": dict(self._deleted)}

    def import_merged(self, merged: Dict[str, Any]) -> None:
        """用合并结果覆盖本地。调用方负责先做 merge，这里只落地。"""
        users = merged.get("users") or {}
        if not isinstance(users, dict) or not users:
            raise AuthError("合并结果为空，拒绝覆盖本地账号。")
        if not any(u.get("role") == "admin" for u in users.values()):
            # 同步进来一份没有管理员的数据 = 从此没人能管账号
            raise AuthError("合并结果里没有管理员，拒绝应用。")
        with self._lock:
            self._users = dict(users)
            self._deleted = dict(merged.get("deleted") or {})
            self._save()
        logger.info("已应用云端账号合并结果，共 %d 个账号", len(users))

    def delete_user(self, username: str) -> None:
        with self._lock:
            if username not in self._users:
                raise AuthError("账号「%s」不存在。" % username)
            if self._users[username].get("role") == "admin":
                others = [n for n, r in self._users.items()
                          if r.get("role") == "admin" and n != username]
                if not others:
                    # 删掉最后一个管理员 = 从此没人能管账号，且无法恢复
                    raise AuthError("这是最后一个管理员，删了就没人能管账号了。"
                                    "请先新建另一个管理员。")
            del self._users[username]
            # 记一笔墓碑：只是本地删掉的话，下次从云端同步又会被拉回来
            self._deleted[username] = time.time()
            self._save()
        logger.info("删除账号 %s", username)


class SessionStore:
    """内存会话表。token 放在 HttpOnly Cookie 里，重启即全部失效。"""

    def __init__(self, ttl_s: float = SESSION_TTL_S):
        self._ttl = float(ttl_s)
        self._lock = threading.Lock()
        self._sessions: Dict[str, Tuple[str, float]] = {}

    def create(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[token] = (username, time.monotonic() + self._ttl)
            self._sweep_locked()
        return token

    def username_for(self, token: Optional[str]) -> Optional[str]:
        if not token:
            return None
        with self._lock:
            item = self._sessions.get(token)
            if item is None:
                return None
            username, expires = item
            if time.monotonic() > expires:
                del self._sessions[token]
                return None
            return username

    def drop(self, token: Optional[str]) -> None:
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)

    def drop_user(self, username: str) -> None:
        """踢掉某个账号的全部会话 —— 改密码/删账号后必须调用，
        否则旧 Cookie 还能继续用，改密码等于没改。"""
        with self._lock:
            for token in [t for t, (u, _) in self._sessions.items()
                          if u == username]:
                del self._sessions[token]

    def _sweep_locked(self) -> None:
        now = time.monotonic()
        for token in [t for t, (_, exp) in self._sessions.items() if now > exp]:
            del self._sessions[token]


# ======================================================================
# 自检：账号库与会话表走一圈
# ======================================================================

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("auth.selftest")

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "users.json")

    # ---- 首次启动：自动建出厂账号 ----
    store = UserStore(path)
    assert store.exists(DEFAULT_USER), "首次启动应自动建 admin"
    assert store.verify(DEFAULT_USER, DEFAULT_PASSWORD), "出厂密码应可登录"
    assert store.role_of(DEFAULT_USER) == "admin"
    assert store.uses_default_password(DEFAULT_USER), "应标记为仍是出厂密码"
    assert not store.verify(DEFAULT_USER, "wrong"), "错密码必须登录失败"
    log.info("出厂账号：%s / %s", DEFAULT_USER, DEFAULT_PASSWORD)

    # ---- 落盘的必须是散列，绝不能有明文 ----
    raw = open(path, "r", encoding="utf-8").read()
    assert DEFAULT_PASSWORD not in raw.replace('"admin"', ""), \
        "口令明文出现在了账号文件里"
    rec = json.loads(raw)["users"][DEFAULT_USER]
    assert rec["algo"] == _ALGO and len(rec["hash"]) == 64 and len(rec["salt"]) == 32
    log.info("落盘格式正确：%s，%d 次迭代", rec["algo"], rec["iterations"])

    # ---- 改密码 ----
    try:
        store.change_password(DEFAULT_USER, "wrong", "newpass")
        raise AssertionError("原密码错误时不该允许改密码")
    except AuthError as exc:
        assert "原密码" in str(exc)
    try:
        store.change_password(DEFAULT_USER, DEFAULT_PASSWORD, "1")
        raise AssertionError("过短的新密码应被拒绝")
    except AuthError as exc:
        assert "至少" in str(exc)
    store.change_password(DEFAULT_USER, DEFAULT_PASSWORD, "newpass")
    assert store.verify(DEFAULT_USER, "newpass"), "新密码应能登录"
    assert not store.verify(DEFAULT_USER, DEFAULT_PASSWORD), "旧密码必须失效"
    assert not store.uses_default_password(DEFAULT_USER), "改完不该再标记出厂密码"
    log.info("改密码：旧密码已失效")

    # ---- 新建/删除账号 ----
    store.add_user("teacher", "teachpass", "user")
    assert store.verify("teacher", "teachpass")
    assert store.role_of("teacher") == "user"
    for bad in ("a", "有中文", "x" * 25, "a b"):
        try:
            store.add_user(bad, "goodpass")
            raise AssertionError("非法用户名 %r 应被拒绝" % bad)
        except AuthError:
            pass
    try:
        store.add_user("teacher", "another")
        raise AssertionError("重名账号应被拒绝")
    except AuthError as exc:
        assert "已经存在" in str(exc)

    # 最后一个管理员不能删
    try:
        store.delete_user(DEFAULT_USER)
        raise AssertionError("最后一个管理员不该被删掉")
    except AuthError as exc:
        assert "最后一个管理员" in str(exc)
    store.add_user("admin2", "adminpass", "admin")
    store.delete_user("admin2")          # 有两个管理员时可以删
    store.delete_user("teacher")
    assert not store.exists("teacher")
    log.info("账号增删：重名/非法名/删最后一个管理员都已拦下")

    # ---- 管理员重置他人密码 ----
    store.add_user("student", "初始密码", "user")
    store.reset_password("student", "resetpass")
    assert store.verify("student", "resetpass")
    log.info("管理员重置密码正常")

    # ---- 重新加载：改动应当留在盘上 ----
    store2 = UserStore(path)
    assert store2.verify(DEFAULT_USER, "newpass"), "重启后新密码应仍然有效"
    assert store2.exists("student") and not store2.exists("teacher")
    log.info("重新加载账号库一致")

    # ---- 账号文件损坏：重建出厂账号而不是崩溃/放行 ----
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{ 这不是合法 JSON")
    store3 = UserStore(path)
    assert store3.verify(DEFAULT_USER, DEFAULT_PASSWORD), \
        "文件损坏后应重建出厂账号，让管理员还进得去"
    log.info("账号文件损坏可自愈")

    # ---- 会话 ----
    sess = SessionStore(ttl_s=100.0)
    tok = sess.create("admin")
    assert sess.username_for(tok) == "admin"
    assert sess.username_for("不存在的token") is None
    assert sess.username_for(None) is None
    tok2 = sess.create("admin")
    assert tok2 != tok, "每次登录应发不同的 token"
    sess.drop(tok)
    assert sess.username_for(tok) is None
    # 改密码后必须踢掉该用户的全部会话
    sess.drop_user("admin")
    assert sess.username_for(tok2) is None, "drop_user 后旧会话必须失效"
    # 过期
    expired = SessionStore(ttl_s=-1.0)
    assert expired.username_for(expired.create("admin")) is None, "过期会话应失效"
    log.info("会话：签发/校验/登出/踢人/过期 均正确")

    log.info("auth selftest OK")
