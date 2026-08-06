# -*- coding: utf-8 -*-
"""云端同步：把录像/标定自动传到对象存储，并从云端同步账号。

设计的第一原则是**离线优先**。观测现场的网络时有时无，所以：
  * 录像、标定、跟踪本身完全不依赖网络，断网时软件行为与单机版一模一样；
  * 要上传的东西先进本地队列（落盘），有网了后台慢慢传；
  * 队列在重启后仍然有效 —— 今晚在山上录的，明天回学校插上网自动补传；
  * 账号断网时用本地缓存，联网时与云端合并。
云端只是「多一份备份 + 多一个管理入口」，任何时候都不该成为用不了的理由。

为什么自己实现 S3 签名而不用 boto3：
boto3+botocore 打包进 exe 要多带上百 MB（光 endpoint 数据文件就几十 MB），
而我们只需要 PUT/GET/LIST 和分片上传这几个操作。SigV4 签名算法本身只有
几十行标准库代码，换来的是零新依赖、零体积增长。
阿里云 OSS、腾讯云 COS、MinIO、AWS S3 都兼容这套协议。

大文件必须走分片上传：一小时的录像有几百 MB，在时断时续的网络上整体重传
永远也传不完。分片后每片 8MB，已传完的片记在队列里，续传时直接跳过。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: 分片大小。S3 规定除最后一片外每片至少 5MB；8MB 在慢网络上单片
#: 大约几十秒能传完，失败重传的代价可以接受。
PART_SIZE = 8 * 1024 * 1024

#: 小于这个大小的文件直接单次 PUT，不走分片（省两次往返）
SIMPLE_PUT_MAX = 16 * 1024 * 1024

_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class CloudError(Exception):
    """云端操作失败，message 直接给用户看。"""


# ---------------------------------------------------------------- 配置

@dataclass
class CloudConfig:
    """云端配置。**单独存 data/cloud.json**，不放进 config.yaml ——
    里面有密钥，而 config.yaml 是用户会随手发给别人看的文件。"""

    enabled: bool = False
    endpoint: str = ""          # 例：https://oss-cn-beijing.aliyuncs.com
    bucket: str = ""
    region: str = "us-east-1"   # 阿里云/腾讯云填什么都行，签名用得上而已
    access_key: str = ""
    secret_key: str = ""
    prefix: str = "lcm90"       # 所有文件都放在这个前缀下
    device_name: str = ""       # 区分多台设备，留空则用主机名
    upload_recordings: bool = True
    upload_calibration: bool = True
    sync_accounts: bool = False  # 账号是否以云端为准

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def masked(self) -> Dict[str, Any]:
        """给界面看的版本：密钥只留头尾，绝不整条吐出去。"""
        d = self.to_dict()
        sk = d.get("secret_key") or ""
        d["secret_key"] = (sk[:2] + "*" * 6 + sk[-2:]) if len(sk) > 6 else ("*" * len(sk))
        ak = d.get("access_key") or ""
        d["access_key"] = (ak[:4] + "*" * 4 + ak[-2:]) if len(ak) > 8 else ak
        return d

    @staticmethod
    def load(path: str) -> "CloudConfig":
        """读配置。环境变量优先于文件 —— 这样打包/部署时可以注入，
        而不必把密钥写进任何跟着代码走的文件里。"""
        cfg = CloudConfig._from_env()
        if cfg is not None:
            return cfg
        if not path or not os.path.exists(path):
            return CloudConfig()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh) or {}
            cfg = CloudConfig()
            for k, v in d.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            return cfg
        except Exception:
            logger.exception("读取云端配置失败，按未配置处理：%s", path)
            return CloudConfig()

    @staticmethod
    def _from_env() -> Optional["CloudConfig"]:
        ep = os.environ.get("LCM90_CLOUD_ENDPOINT", "").strip()
        if not ep:
            return None
        cfg = CloudConfig(
            enabled=True, endpoint=ep,
            bucket=os.environ.get("LCM90_CLOUD_BUCKET", "").strip(),
            access_key=os.environ.get("LCM90_CLOUD_KEY", "").strip(),
            secret_key=os.environ.get("LCM90_CLOUD_SECRET", "").strip(),
            region=os.environ.get("LCM90_CLOUD_REGION", "us-east-1").strip())
        logger.info("云端配置来自环境变量")
        return cfg

    def save(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)   # 里面有密钥，别让同机其他账户随便读
        except OSError:
            pass

    def check(self) -> None:
        missing = [n for n in ("endpoint", "bucket", "access_key", "secret_key")
                   if not str(getattr(self, n) or "").strip()]
        if missing:
            raise CloudError("云端配置不完整，还缺：%s" % "、".join(missing))
        if not self.endpoint.startswith(("http://", "https://")):
            raise CloudError("endpoint 要以 http:// 或 https:// 开头，"
                             "例如 https://oss-cn-beijing.aliyuncs.com")


# ---------------------------------------------------------------- S3 客户端

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _uriencode(s: str, encode_slash: bool = True) -> str:
    safe = "-_.~" if encode_slash else "-_.~/"
    return urllib.parse.quote(s, safe=safe)


class S3Client:
    """够用就好的 S3 兼容客户端：PUT / GET / DELETE / 分片上传。

    只实现 SigV4 header 签名这一种方式 —— 它被所有主流对象存储支持，
    而且不像预签名 URL 那样会把凭据暴露在链接里。
    """

    def __init__(self, cfg: CloudConfig, timeout: float = 30.0):
        cfg.check()
        self.cfg = cfg
        self.timeout = float(timeout)
        parsed = urllib.parse.urlparse(cfg.endpoint.rstrip("/"))
        self._scheme = parsed.scheme
        self._host = parsed.netloc
        # 路径风格（endpoint/bucket/key）兼容性最好：MinIO、自建网关、
        # 以及一部分企业内网都不支持 bucket 作为子域名。
        self._base_path = "/%s" % cfg.bucket

    # -------- 签名 --------

    def _sign(self, method: str, path: str, query: Dict[str, str],
              payload_sha: str, extra_headers: Optional[Dict[str, str]] = None
              ) -> Dict[str, str]:
        now = time.gmtime()
        amz_date = time.strftime("%Y%m%dT%H%M%SZ", now)
        date_stamp = time.strftime("%Y%m%d", now)

        headers = {"host": self._host,
                   "x-amz-content-sha256": payload_sha,
                   "x-amz-date": amz_date}
        if extra_headers:
            for k, v in extra_headers.items():
                headers[k.lower()] = v

        signed_names = sorted(headers)
        canonical_headers = "".join("%s:%s\n" % (k, str(headers[k]).strip())
                                    for k in signed_names)
        signed_headers = ";".join(signed_names)
        canonical_query = "&".join(
            "%s=%s" % (_uriencode(k), _uriencode(str(query[k])))
            for k in sorted(query))
        canonical_request = "\n".join([
            method, _uriencode(path, encode_slash=False), canonical_query,
            canonical_headers, signed_headers, payload_sha])

        scope = "%s/%s/s3/aws4_request" % (date_stamp, self.cfg.region)
        to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope,
                             _sha256(canonical_request.encode("utf-8"))])
        k = ("AWS4" + self.cfg.secret_key).encode("utf-8")
        for part in (date_stamp, self.cfg.region, "s3", "aws4_request"):
            k = _hmac(k, part)
        signature = hmac.new(k, to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

        headers["Authorization"] = (
            "AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s"
            % (self.cfg.access_key, scope, signed_headers, signature))
        return headers

    def _request(self, method: str, key: str, query: Optional[Dict[str, str]] = None,
                 body: bytes = b"", extra_headers: Optional[Dict[str, str]] = None
                 ) -> Tuple[int, Dict[str, str], bytes]:
        query = query or {}
        path = self._base_path + ("/" + key.lstrip("/") if key else "")
        payload_sha = _sha256(body) if body else _EMPTY_SHA256
        headers = self._sign(method, path, query, payload_sha, extra_headers)

        url = "%s://%s%s" % (self._scheme, self._host, _uriencode(path, False))
        if query:
            url += "?" + "&".join("%s=%s" % (_uriencode(k), _uriencode(str(query[k])))
                                  for k in sorted(query))
        req = urllib.request.Request(url, data=body or None, method=method)
        for k, v in headers.items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as exc:
            detail = b""
            try:
                detail = exc.read()[:600]
            except Exception:
                pass
            raise CloudError("云端返回 %s：%s"
                             % (exc.code, _explain_s3_error(exc.code, detail)))
        except urllib.error.URLError as exc:
            # 断网、DNS 失败、超时都走这里 —— 对调用方来说是「稍后再试」
            raise CloudError("连不上云端：%s" % exc.reason)
        except Exception as exc:      # noqa: BLE001
            raise CloudError("云端请求出错：%s" % exc)

    # -------- 对外操作 --------

    def put_bytes(self, key: str, data: bytes,
                  content_type: str = "application/octet-stream") -> None:
        self._request("PUT", key, body=data,
                      extra_headers={"content-type": content_type})

    def get_bytes(self, key: str) -> Optional[bytes]:
        try:
            _st, _h, body = self._request("GET", key)
            return body
        except CloudError as exc:
            if "404" in str(exc) or "NoSuchKey" in str(exc):
                return None
            raise

    def delete(self, key: str) -> None:
        self._request("DELETE", key)

    def ping(self) -> str:
        """连通性自检：往桶里写一个小文件再删掉。

        只测 GET/LIST 不够 —— 很多子账号是只读的，而我们真正需要的是**写**。
        与其等到第一次上传录像时才发现没权限，不如现在就说清楚。
        """
        key = "%s/_connftest.txt" % self.cfg.prefix.strip("/")
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.put_bytes(key, ("LCM90 连接测试 %s" % stamp).encode("utf-8"),
                       "text/plain; charset=utf-8")
        try:
            self.delete(key)
        except CloudError:
            pass       # 能写就算通过，删不掉不影响使用
        return "连接成功，读写权限正常。"

    # -------- 分片上传 --------

    def create_multipart(self, key: str, content_type: str) -> str:
        _st, _h, body = self._request("POST", key, query={"uploads": ""},
                                      extra_headers={"content-type": content_type})
        root = ET.fromstring(body)
        for el in root.iter():
            if el.tag.endswith("UploadId"):
                return el.text or ""
        raise CloudError("云端没有返回 UploadId，分片上传无法开始")

    def upload_part(self, key: str, upload_id: str, part_no: int,
                    data: bytes) -> str:
        _st, headers, _b = self._request(
            "PUT", key, query={"partNumber": str(part_no), "uploadId": upload_id},
            body=data)
        etag = headers.get("ETag") or headers.get("Etag") or ""
        if not etag:
            raise CloudError("云端没有返回分片的 ETag")
        return etag.strip('"')

    def complete_multipart(self, key: str, upload_id: str,
                           parts: List[Tuple[int, str]]) -> None:
        xml = ["<CompleteMultipartUpload>"]
        for no, etag in sorted(parts):
            xml.append("<Part><PartNumber>%d</PartNumber><ETag>\"%s\"</ETag></Part>"
                       % (no, etag))
        xml.append("</CompleteMultipartUpload>")
        self._request("POST", key, query={"uploadId": upload_id},
                      body="".join(xml).encode("utf-8"),
                      extra_headers={"content-type": "application/xml"})

    def abort_multipart(self, key: str, upload_id: str) -> None:
        try:
            self._request("DELETE", key, query={"uploadId": upload_id})
        except CloudError:
            logger.debug("放弃分片上传失败（可忽略）", exc_info=True)


def _explain_s3_error(code: int, detail: bytes) -> str:
    """把 S3 的错误码翻译成用户能照着做的话。"""
    text = detail.decode("utf-8", "replace")
    hint = {
        400: "请求被拒绝，多半是 region 或 endpoint 填得不对",
        403: "没有权限。检查 AccessKey/SecretKey 是否正确、"
             "以及这个账号有没有往这个桶写文件的权限",
        404: "桶或文件不存在，检查 bucket 名字是否写对",
        503: "云端忙，稍后会自动重试",
    }.get(code, "")
    # S3 的 XML 错误体里 <Code> 最有用，把它挑出来
    m = ""
    try:
        root = ET.fromstring(text)
        for el in root.iter():
            if el.tag.endswith("Code"):
                m = el.text or ""
                break
    except Exception:
        pass
    parts = [x for x in (hint, m) if x]
    return "；".join(parts) if parts else text[:200]


# ---------------------------------------------------------------- 上传队列

@dataclass
class QueueItem:
    path: str                 # 本地文件绝对路径
    key: str                  # 云端对象名
    size: int = 0
    added_at: float = 0.0
    tries: int = 0
    next_try_at: float = 0.0
    upload_id: str = ""       # 分片上传的会话
    parts: List[List[Any]] = field(default_factory=list)   # [[片号, etag], ...]
    done: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UploadQueue:
    """落盘的上传队列 + 后台上传线程。

    重启后队列仍在：山上录的今晚传不完，回到有网的地方接着传。
    """

    #: 退避序列（秒）。断网时不该每秒重试一次把日志刷爆。
    _BACKOFF = (5, 15, 60, 180, 600, 1800)

    def __init__(self, queue_file: str,
                 client_factory: Callable[[], Optional[S3Client]]):
        self._file = queue_file
        self._make_client = client_factory
        self._lock = threading.RLock()
        self._items: List[QueueItem] = []
        self._uploaded: set = set()      # 传成功过的 key，手动重复点不会再传一遍
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_error = ""
        self._last_ok_at = 0.0
        self._uploading = ""       # 正在传的文件名，给界面显示
        self._progress = 0.0
        self._load()

    # -------- 落盘 --------

    def _load(self) -> None:
        if not self._file or not os.path.exists(self._file):
            return
        try:
            with open(self._file, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
            for d in data.get("items", []):
                it = QueueItem(path=d.get("path", ""), key=d.get("key", ""))
                for k, v in d.items():
                    if hasattr(it, k):
                        setattr(it, k, v)
                if not it.done:
                    self._items.append(it)
            self._uploaded = set(data.get("uploaded") or [])
        except Exception:
            logger.exception("读取上传队列失败，从空队列开始：%s", self._file)

    def _save_locked(self) -> None:
        if not self._file:
            return
        try:
            parent = os.path.dirname(self._file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = self._file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"items": [i.to_dict() for i in self._items],
                           "uploaded": sorted(self._uploaded)}, fh,
                          ensure_ascii=False, indent=1)
            os.replace(tmp, self._file)
        except Exception:
            logger.exception("保存上传队列失败")

    # -------- 对外 --------

    def enqueue(self, path: str, key: str) -> bool:
        if not path or not os.path.exists(path):
            return False
        with self._lock:
            if key in self._uploaded:
                return False       # 这个文件已经传上去了
            if any(i.path == path and not i.done for i in self._items):
                return False       # 已经在队列里
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            self._items.append(QueueItem(path=path, key=key, size=size,
                                         added_at=time.time()))
            self._save_locked()
        logger.info("已加入上传队列：%s → %s", os.path.basename(path), key)
        self._wake.set()
        return True

    def is_uploaded(self, key: str) -> bool:
        with self._lock:
            return key in self._uploaded

    def status(self) -> Dict[str, Any]:
        with self._lock:
            pending = [i for i in self._items if not i.done]
            failed = [i for i in pending if i.tries >= len(self._BACKOFF)]
            total_bytes = sum(i.size for i in pending)
            return {
                "pending": len(pending),
                "stuck": len(failed),
                "pending_mb": round(total_bytes / 1048576.0, 1),
                "uploading": self._uploading,
                "progress": round(self._progress, 3),
                "last_error": self._last_error,
                "last_ok_at": self._last_ok_at,
                "running": self._thread is not None and self._thread.is_alive(),
            }

    def retry_now(self) -> None:
        """用户点「立即同步」：把所有退避清零，马上再试一轮。"""
        with self._lock:
            for i in self._items:
                i.next_try_at = 0.0
                if i.tries >= len(self._BACKOFF):
                    i.tries = 0     # 给彻底放弃的也一次机会
            self._save_locked()
        self._wake.set()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="uploader",
                                        daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()
        th = self._thread
        if th is not None:
            th.join(timeout=3.0)

    # -------- 后台线程 --------

    def _loop(self) -> None:
        while not self._stop.is_set():
            item = self._next_due()
            if item is None:
                self._wake.wait(10.0)
                self._wake.clear()
                continue
            try:
                client = self._make_client()
            except Exception as exc:      # noqa: BLE001
                client = None
                self._last_error = str(exc)
            if client is None:
                self._wake.wait(20.0)
                self._wake.clear()
                continue
            self._upload_one(client, item)

    def _next_due(self) -> Optional[QueueItem]:
        now = time.time()
        with self._lock:
            for i in self._items:
                if not i.done and i.next_try_at <= now \
                        and i.tries < len(self._BACKOFF):
                    return i
        return None

    def _upload_one(self, client: S3Client, item: QueueItem) -> None:
        name = os.path.basename(item.path)
        self._uploading = name
        self._progress = 0.0
        try:
            if not os.path.exists(item.path):
                # 文件被用户删了：从队列里去掉，不算失败
                self._finish(item, ok=True, note="源文件已不存在")
                return
            size = os.path.getsize(item.path)
            if size <= SIMPLE_PUT_MAX:
                with open(item.path, "rb") as fh:
                    client.put_bytes(item.key, fh.read(), _guess_type(item.path))
                self._progress = 1.0
            else:
                self._upload_multipart(client, item, size)
            self._finish(item, ok=True)
            logger.info("上传完成：%s", item.key)
        except CloudError as exc:
            self._fail(item, str(exc))
        except Exception as exc:          # noqa: BLE001
            logger.exception("上传出现未预期错误")
            self._fail(item, str(exc))
        finally:
            self._uploading = ""

    def _upload_multipart(self, client: S3Client, item: QueueItem,
                          size: int) -> None:
        if not item.upload_id:
            item.upload_id = client.create_multipart(item.key,
                                                     _guess_type(item.path))
            item.parts = []
            with self._lock:
                self._save_locked()

        done_nos = {int(p[0]) for p in item.parts}
        total_parts = max(1, (size + PART_SIZE - 1) // PART_SIZE)
        with open(item.path, "rb") as fh:
            for no in range(1, total_parts + 1):
                if self._stop.is_set():
                    raise CloudError("程序正在退出，上传稍后继续")
                if no in done_nos:
                    self._progress = no / total_parts
                    continue        # 断点续传：这一片上次已经传完了
                fh.seek((no - 1) * PART_SIZE)
                chunk = fh.read(PART_SIZE)
                if not chunk:
                    break
                etag = client.upload_part(item.key, item.upload_id, no, chunk)
                item.parts.append([no, etag])
                self._progress = no / total_parts
                with self._lock:
                    self._save_locked()   # 每传完一片就记账，断了不白传
        client.complete_multipart(item.key, item.upload_id,
                                  [(int(n), e) for n, e in item.parts])

    def _finish(self, item: QueueItem, ok: bool, note: str = "") -> None:
        with self._lock:
            item.done = True
            item.error = note
            self._items = [i for i in self._items if not i.done]
            self._save_locked()
        if ok:
            with self._lock:
                self._uploaded.add(item.key)
                self._save_locked()
            self._last_ok_at = time.time()
            self._last_error = ""

    def _fail(self, item: QueueItem, err: str) -> None:
        with self._lock:
            item.tries += 1
            item.error = err
            idx = min(item.tries - 1, len(self._BACKOFF) - 1)
            item.next_try_at = time.time() + self._BACKOFF[idx]
            # 分片会话可能已经失效，下次从头再来（已传的片仍记在 parts 里，
            # 若 upload_id 还有效就能续；无效时 complete 会报错并重建）
            self._save_locked()
        self._last_error = err
        logger.warning("上传失败（第 %d 次）：%s —— %s",
                       item.tries, os.path.basename(item.path), err)


def _guess_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {".mp4": "video/mp4", ".avi": "video/x-msvideo",
            ".json": "application/json; charset=utf-8",
            ".log": "text/plain; charset=utf-8",
            ".png": "image/png", ".jpg": "image/jpeg"}.get(
        ext, "application/octet-stream")


# ---------------------------------------------------------------- 账号同步

class CloudAccounts:
    """把账号文件放在云端，多台设备共用一套账号。

    合并规则很简单也很重要：**以 updated_at 大的为准，逐个账号比较**。
    整份覆盖的话，A 机器改了密码、B 机器加了账号，后同步的那台会把另一台
    的改动抹掉。断网时完全用本地文件，不影响登录。
    """

    def __init__(self, client_factory: Callable[[], Optional[S3Client]],
                 remote_key):
        self._make_client = client_factory
        # 允许传函数：用户改了 prefix 之后，路径要跟着变，
        # 构造时固定成字符串的话会一直往旧位置写。
        self._key_src = remote_key

    @property
    def _key(self) -> str:
        return self._key_src() if callable(self._key_src) else str(self._key_src)

    def pull(self) -> Optional[Dict[str, Any]]:
        client = self._make_client()
        if client is None:
            return None
        raw = client.get_bytes(self._key)
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            logger.exception("云端账号文件格式不对，忽略")
            return None

    def push(self, data: Dict[str, Any]) -> None:
        client = self._make_client()
        if client is None:
            raise CloudError("云端未配置或不可用")
        client.put_bytes(self._key,
                         json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
                         "application/json; charset=utf-8")

    @staticmethod
    def merge(local: Dict[str, Any], remote: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """逐账号按 updated_at 取新的。返回 (合并结果, 本地是否有变化)。"""
        lu = dict((local or {}).get("users") or {})
        ru = dict((remote or {}).get("users") or {})
        changed = False
        for name, rec in ru.items():
            cur = lu.get(name)
            if cur is None:
                lu[name] = rec
                changed = True
            else:
                if float(rec.get("updated_at", 0) or 0) > float(cur.get("updated_at", 0) or 0):
                    lu[name] = rec
                    changed = True
        # 云端明确标记删除的账号，本地也删掉
        for name in list((remote or {}).get("deleted") or []):
            if name in lu:
                del lu[name]
                changed = True
        out = dict(local or {})
        out["users"] = lu
        out.setdefault("version", 1)
        return out, changed


# ======================================================================
# 自检：用一个内存版 S3 服务器把整条链路跑通
# ======================================================================

if __name__ == "__main__":
    import http.server
    import socket
    import tempfile

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("cloud.selftest")

    # ---- 1) SigV4 签名对照 AWS 官方测试向量 ----
    # 用官方 aws4_testsuite 的密钥与已知答案验证签名链，
    # 算错的话会在真机上表现为「403 权限不足」，极难排查。
    known_key = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
    k = ("AWS4" + known_key).encode()
    for part in ("20150830", "us-east-1", "iam", "aws4_request"):
        k = _hmac(k, part)
    expect = "c4afb1cc5771d871763a393e44b703571b55cc28424d1a5e86da6ed3c154a4b9"
    assert k.hex() == expect, "SigV4 派生密钥算错了：%s" % k.hex()
    log.info("SigV4 派生密钥与 AWS 官方测试向量一致")

    # ---- 2) 起一个最小 S3 mock ----
    store: Dict[str, bytes] = {}
    uploads: Dict[str, Dict[int, bytes]] = {}
    flaky = {"fail_next": 0}          # 用来模拟网络抖动

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _key(self):
            path = urllib.parse.urlparse(self.path).path
            return path.split("/", 2)[2] if path.count("/") >= 2 else ""

        def _q(self):
            # keep_blank_values 不能省：CreateMultipartUpload 的标志是无值的
            # ?uploads=，默认解析会把它整个丢掉
            return urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query,
                                         keep_blank_values=True)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def _send(self, code, body=b"", headers=None):
            self.send_response(code)
            for k2, v in (headers or {}).items():
                self.send_header(k2, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_PUT(self):
            body = self._body()
            if flaky["fail_next"] > 0:
                flaky["fail_next"] -= 1
                self._send(503, b"<Error><Code>SlowDown</Code></Error>")
                return
            assert self.headers.get("Authorization", "").startswith("AWS4-HMAC-SHA256"), \
                "请求没带 SigV4 签名"
            q = self._q()
            if "partNumber" in q:
                uid = q["uploadId"][0]
                uploads.setdefault(uid, {})[int(q["partNumber"][0])] = body
                self._send(200, headers={"ETag": '"etag%s"' % q["partNumber"][0]})
                return
            store[self._key()] = body
            self._send(200)

        def do_POST(self):
            q = self._q()
            key = self._key()
            if "uploads" in q:
                uid = "u%d" % (len(uploads) + 1)
                uploads[uid] = {}
                self._send(200, ("<InitiateMultipartUploadResult><UploadId>%s"
                                 "</UploadId></InitiateMultipartUploadResult>" % uid
                                 ).encode())
                return
            uid = q["uploadId"][0]
            parts = uploads.pop(uid, {})
            store[key] = b"".join(parts[n] for n in sorted(parts))
            self._send(200, b"<CompleteMultipartUploadResult/>")

        def do_GET(self):
            key = self._key()
            if key in store:
                self._send(200, store[key])
            else:
                self._send(404, b"<Error><Code>NoSuchKey</Code></Error>")

        def do_DELETE(self):
            store.pop(self._key(), None)
            self._send(204)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    cfg = CloudConfig(enabled=True, endpoint="http://127.0.0.1:%d" % port,
                      bucket="testbucket", access_key="AKIATEST",
                      secret_key="secret123", prefix="lcm90")
    client = S3Client(cfg)

    # ---- 3) 连通性自检 ----
    log.info("ping：%s", client.ping())

    # ---- 4) 小文件直传 ----
    tmp = tempfile.mkdtemp()
    small = os.path.join(tmp, "small.json")
    with open(small, "w") as fh:
        fh.write('{"hello":"world"}')
    client.put_bytes("lcm90/small.json", open(small, "rb").read())
    assert store["lcm90/small.json"] == b'{"hello":"world"}'
    assert client.get_bytes("lcm90/small.json") == b'{"hello":"world"}'
    assert client.get_bytes("lcm90/不存在.json") is None, "取不存在的对象应返回 None"
    log.info("小文件 PUT/GET 正常，取不存在的对象返回 None")

    # ---- 5) 大文件分片上传（20MB，会切成 3 片）----
    big = os.path.join(tmp, "big.mp4")
    payload = os.urandom(20 * 1024 * 1024)
    with open(big, "wb") as fh:
        fh.write(payload)

    qfile = os.path.join(tmp, "queue.json")
    q = UploadQueue(qfile, lambda: client)
    q.enqueue(big, "lcm90/videos/big.mp4")
    assert q.status()["pending"] == 1
    assert not q.enqueue(big, "lcm90/videos/big.mp4"), "同一个文件不该重复入队"
    q.start()
    deadline = time.monotonic() + 60
    while q.status()["pending"] > 0 and time.monotonic() < deadline:
        time.sleep(0.2)
    assert q.status()["pending"] == 0, "20MB 文件没能在 60 秒内传完"
    assert store["lcm90/videos/big.mp4"] == payload, "分片重组后的内容与原文件不一致"
    log.info("分片上传通过：20MB 切 3 片，重组后与原文件逐字节一致")

    # ---- 6) 网络抖动：失败后要重试，且不能丢文件 ----
    flaky["fail_next"] = 2
    small2 = os.path.join(tmp, "retry.json")
    with open(small2, "w") as fh:
        fh.write("retry-me")
    q.enqueue(small2, "lcm90/retry.json")
    q.retry_now()
    deadline = time.monotonic() + 40
    while q.status()["pending"] > 0 and time.monotonic() < deadline:
        q.retry_now()          # 模拟用户点「立即同步」，跳过退避等待
        time.sleep(0.5)
    assert store.get("lcm90/retry.json") == b"retry-me", "重试后仍未传上去"
    log.info("网络抖动后自动重试成功（连续失败 2 次后传成）")

    # ---- 7) 队列要能跨重启：写盘后新建一个队列对象应能读回 ----
    q.shutdown()
    ghost = os.path.join(tmp, "ghost.bin")
    with open(ghost, "wb") as fh:
        fh.write(b"x" * 1024)
    q2 = UploadQueue(qfile, lambda: client)
    q2.enqueue(ghost, "lcm90/ghost.bin")
    q3 = UploadQueue(qfile, lambda: client)      # 模拟重启
    assert q3.status()["pending"] == 1, "重启后队列应当还在"
    log.info("队列跨重启保留：待传 %d 个", q3.status()["pending"])

    # ---- 8) 断网时不能崩，只是排队等着 ----
    dead_cfg = CloudConfig(enabled=True, endpoint="http://127.0.0.1:1",
                           bucket="b", access_key="a", secret_key="s")
    try:
        S3Client(dead_cfg, timeout=1.0).ping()
        raise AssertionError("连不上时应该抛 CloudError")
    except CloudError as exc:
        assert "连不上云端" in str(exc), str(exc)
    log.info("断网时给出的是人话提示：%s", "连不上云端…")

    # ---- 9) 配置校验与脱敏 ----
    try:
        CloudConfig(enabled=True, endpoint="oss-cn-beijing.aliyuncs.com",
                    bucket="b", access_key="a", secret_key="s").check()
        raise AssertionError("endpoint 少了 http:// 应该被拒绝")
    except CloudError as exc:
        assert "http://" in str(exc)
    m = CloudConfig(access_key="AKIDabcdefghijk", secret_key="verysecretvalue").masked()
    assert "verysecretvalue" not in json.dumps(m), "脱敏后仍能看到完整密钥"
    assert m["secret_key"].count("*") >= 6
    log.info("配置校验与脱敏正常：%s", m["secret_key"])

    # ---- 10) 账号合并：各改各的，不能互相抹掉 ----
    local = {"users": {"admin": {"hash": "A", "updated_at": 100},
                       "teacher": {"hash": "T", "updated_at": 100}}}
    remote = {"users": {"admin": {"hash": "A2", "updated_at": 200},
                        "student": {"hash": "S", "updated_at": 150}},
              "deleted": ["teacher"]}
    merged, changed = CloudAccounts.merge(local, remote)
    assert changed
    assert merged["users"]["admin"]["hash"] == "A2", "云端更新的密码应生效"
    assert merged["users"]["student"]["hash"] == "S", "云端新增的账号应同步下来"
    assert "teacher" not in merged["users"], "云端删除的账号应同步删除"

    local2 = {"users": {"admin": {"hash": "NEW", "updated_at": 300}}}
    remote2 = {"users": {"admin": {"hash": "OLD", "updated_at": 200}}}
    merged2, changed2 = CloudAccounts.merge(local2, remote2)
    assert merged2["users"]["admin"]["hash"] == "NEW", "本地更新的不该被旧的云端覆盖"
    assert not changed2
    log.info("账号合并按时间戳逐个比较，双向改动都不丢")

    srv.shutdown()
    log.info("cloud selftest OK")
