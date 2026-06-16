"""word_align CUDA sidecar (TODOS #18) — 长驻独立进程, idle TTL 自杀真正释放 VRAM.

为什么要 sidecar (评审定案):
  word_align 的 CUDA ONNX session 一旦加载, 显存被 ORT BFCArena 高水位缓着不还,
  唯进程退出可释放. 把 CUDA word_align 拆到独立长驻进程, 空闲 idle_ttl 后自杀,
  主 ASR 服务长期 baseline 不被偶发的词级时间戳请求顶高.

架构 (锁定决策):
  - A1: 长驻进程 + idle TTL + **Unix domain socket** request/response (stdlib, 零依赖).
    传 audio_path + chunks JSON (不传字节, sidecar 自己读文件).
  - A3: sidecar **只做 CUDA** (单一职责); CPU 兜底留主进程. 超时主进程杀 sidecar 再 CPU.
  - A4: 仅 cuda runtime 启用 (Mac worker 即退 / CPU 不碰显存, 不需要).
  - codex #6: client 是**进程全局单例** (pool>1 也只一个 CUDA session).
  - codex #8: CUDA OOM → sidecar **退休** (退出进程释放 VRAM), 不永久 poison 主进程.
  - codex #9: sidecar **瘦入口** — 只 import qwen3/word_align.py + 极简音频加载,
    不碰 qwen3_transcriber.py (避免拖入 ASR/diarize 机器).
  - codex #15: socket 路径 **per-PID** (避免 PM2 多实例 / 测试 / 重启撞车).

⚠️ preflight 不替代 OOM fallback (codex #11): sidecar 内 CUDA OOM 仍可能发生 (TOCTOU),
  退休 + 主进程 CPU 兜底永远保留.

本模块分三层 (各自单测):
  1. 协议: send_msg / recv_msg (4 字节大端长度前缀 + JSON body).
  2. server: handle_conn (单请求) + serve (bind UDS + accept loop + idle TTL).
  3. client: WordAlignSidecarClient (主进程侧单例 manager, 见同文件下半).
"""
from __future__ import annotations

import json
import os
import socket
import struct
import time
from typing import Any, Callable, List, Optional, Tuple

from loguru import logger

# 4 字节大端无符号长度前缀: UDS 是字节流, 必须自带 framing 防半包/粘包.
_LEN = struct.Struct(">I")


# ════════════════════════════════════════════════════════════════════════
# 1. 协议 (length-prefixed JSON over UDS)
# ════════════════════════════════════════════════════════════════════════
def _encode(obj: dict) -> bytes:
    """obj → length-prefixed JSON 字节 (utf-8)."""
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return _LEN.pack(len(body)) + body


def send_msg(sock: socket.socket, obj: dict) -> None:
    """发一条消息 (length-prefixed JSON). sendall 保证整帧发出."""
    sock.sendall(_encode(obj))


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """阻塞读够 n 字节; 对端 EOF (返回空) → None."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket) -> Optional[dict]:
    """收一条消息; 对端关闭 / 半截 → None (不抛). socket.timeout 由调用方处理."""
    header = _recv_exact(sock, _LEN.size)
    if header is None:
        return None
    (n,) = _LEN.unpack(header)
    body = _recv_exact(sock, n)
    if body is None:
        return None
    return json.loads(body.decode("utf-8"))


# ════════════════════════════════════════════════════════════════════════
# 2. server (sidecar 进程侧)
# ════════════════════════════════════════════════════════════════════════
def handle_conn(
    conn: socket.socket,
    *,
    aligner: Any,
    audio_loader: Callable[[str], Any],
) -> bool:
    """处理一个连接上的单个请求, 回响应. 返回 retire 标志 (True=该退休/退出进程).

    请求形态:
      - {"cmd": "ping"} → {"ok": True, "pong": True} (健康检查, 不调 aligner).
      - {"audio_path", "chunks", "language"} → 对齐, 回 {"ok": True, "words", "stats"}.

    错误:
      - CUDA 资源错误 (is_resource_error) → {"ok": False, "resource_error": True, ...}
        + 返回 retire=True (codex #8: OOM 退休, 主进程会杀掉/不再用本 sidecar).
      - 普通对齐错误 → {"ok": False, "resource_error": False, ...}, retire=False.
    """
    from src.core.qwen3.word_align import is_resource_error

    req = recv_msg(conn)
    if req is None:
        return False
    if req.get("cmd") == "ping":
        send_msg(conn, {"ok": True, "pong": True})
        return False
    try:
        audio = audio_loader(req["audio_path"])
        words, stats = aligner.align_chunks(
            audio, req["chunks"], language=req.get("language")
        )
        send_msg(conn, {"ok": True, "words": words, "stats": stats})
        return False
    except Exception as exc:  # 对齐失败: 资源类退休, 普通类继续服务
        resource = is_resource_error(exc)
        send_msg(
            conn,
            {"ok": False, "error": f"{type(exc).__name__}: {exc}", "resource_error": resource},
        )
        if resource:
            logger.warning(f"sidecar CUDA 资源错误, 退休退出释放 VRAM: {exc}")
        return resource


def serve(
    sock_path: str,
    *,
    aligner_factory: Callable[[], Any],
    audio_loader: Callable[[str], Any],
    idle_ttl_sec: float,
) -> None:
    """bind UDS + accept loop. 空闲 idle_ttl_sec 无连接 → 退出 (释放 VRAM).

    aligner 惰性 build (首个真实对齐请求才 factory()), 无请求时不占显存.
    idle TTL 用 accept() 超时实现 (codex #5): 服务中 socket busy, 仅 idle 等连接
    时才计时, 超时即退. lease-free — 客户端连不上 = 进程已退, 自行 respawn.
    """
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        try:
            os.unlink(sock_path)  # 清残留 socket 文件
        except FileNotFoundError:
            pass
        srv.bind(sock_path)
        srv.listen(8)  # backlog: 容忍 readiness 探测 + 真实请求短暂并排 (单线程顺序处理)
        srv.settimeout(idle_ttl_sec)
        logger.info(f"word_align sidecar 就绪: {sock_path} (idle_ttl={idle_ttl_sec}s)")

        aligner: Optional[Any] = None
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                logger.info(f"word_align sidecar idle {idle_ttl_sec}s 无请求, 退出释放 VRAM")
                return
            with conn:
                # ping 不需要 aligner; 真实对齐请求才 lazy build
                if aligner is None:
                    # peek 不方便, 直接 build 前先看是不是 ping: handle_conn 内部已分流,
                    # 但 build 在 handle_conn 外做才能保证 ping 不触发 build → 传 lazy factory.
                    aligner = _LazyAligner(aligner_factory)
                retire = handle_conn(conn, aligner=aligner, audio_loader=audio_loader)
            if retire:
                logger.warning("word_align sidecar 退休 (CUDA 资源错误), 退出")
                return
    finally:
        srv.close()
        try:
            os.unlink(sock_path)
        except OSError:
            pass


class _LazyAligner:
    """惰性包装: 首个 align_chunks 才真 build CUDA aligner (ping 不触发 build)."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._inner: Optional[Any] = None

    def align_chunks(self, audio, chunks, language=None):
        if self._inner is None:
            self._inner = self._factory()
        return self._inner.align_chunks(audio, chunks, language=language)


# UDS sun_path 上限 (macOS 104 / Linux 108 字节), socket 必须放短路径.
# 固定用 /tmp (macOS /tmp→/private/tmp 仍短; Linux /tmp 短), per-PID 命名避免
# PM2 多实例 / 测试 / 重启撞车 (codex #15). 不用 tempfile.gettempdir() —
# macOS TMPDIR 是 /var/folders/... 深路径, 容易超 104.
_SOCKET_DIR = "/tmp"


def default_socket_path(owner_pid: Optional[int] = None) -> str:
    """主进程的 word_align sidecar UDS 路径 (per-owner-PID, 短路径).

    owner_pid 缺省取当前进程 PID — 主 ASR 服务进程拥有一个 sidecar.
    """
    pid = owner_pid if owner_pid is not None else os.getpid()
    return os.path.join(_SOCKET_DIR, f"funasr_wa_sidecar_{pid}.sock")


def ping(sock_path: str, timeout: float = 0.5) -> bool:
    """对 sidecar 发一次 ping, 收到 pong 返回 True (健康检查). 任何失败 → False."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(timeout)
        s.connect(sock_path)
        send_msg(s, {"cmd": "ping"})
        resp = recv_msg(s)
        return bool(resp and resp.get("pong"))
    except OSError:
        return False
    finally:
        s.close()


def wait_for_socket(sock_path: str, timeout: float = 5.0, interval: float = 0.02) -> bool:
    """轮询等 sidecar 就绪 (ping 通). 完整 ping/pong 握手, 返回时 sidecar 已回到
    accept 等待态 (单线程顺序处理), 避免 readiness 探测占着半个连接. 超时 → False.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(sock_path) and ping(sock_path, timeout=0.5):
            return True
        time.sleep(interval)
    return False
