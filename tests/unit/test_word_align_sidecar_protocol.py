"""Lane 2 (#18) — word_align sidecar UDS 协议 + server loop 单测.

协议: 4 字节大端长度前缀 + JSON body (send_msg / recv_msg), 用 socketpair 验
round-trip / 半包粘包 / 连接关闭. server loop 用 socketpair + 假 aligner 验
正常对齐 / 资源错误退休 (codex #8) / idle TTL 退出 (codex #5 用 accept 超时).

不起真 subprocess / 不加载真 MMS ONNX.
"""
from __future__ import annotations

import os
import socket
import struct
import threading

import pytest

from src.core.qwen3 import word_align_sidecar as sc


# ──────────────────────────────────────────────────────────────────────────
# 协议: send_msg / recv_msg (length-prefixed JSON)
# ──────────────────────────────────────────────────────────────────────────
def test_send_recv_roundtrip():
    a, b = socket.socketpair()
    try:
        sc.send_msg(a, {"hello": "世界", "n": 42})
        got = sc.recv_msg(b)
        assert got == {"hello": "世界", "n": 42}
    finally:
        a.close(); b.close()


def test_recv_msg_handles_split_frames(monkeypatch):
    """body 分多次到达 (粘包/半包) → recv_exact 拼齐."""
    a, b = socket.socketpair()
    try:
        payload = sc._encode({"x": list(range(50))})
        # 手动分两段发: 先 header+部分 body, 再剩余
        a.sendall(payload[:5])
        a.sendall(payload[5:])
        assert sc.recv_msg(b) == {"x": list(range(50))}
    finally:
        a.close(); b.close()


def test_recv_msg_returns_none_on_closed_peer():
    """对端关闭 (EOF) → None, 不抛."""
    a, b = socket.socketpair()
    a.close()
    try:
        assert sc.recv_msg(b) is None
    finally:
        b.close()


# ──────────────────────────────────────────────────────────────────────────
# server loop: serve_once 处理一个请求 (用 socketpair 注入 conn)
# ──────────────────────────────────────────────────────────────────────────
class _FakeAligner:
    def __init__(self, words=None, exc=None):
        self._words = words or []
        self._exc = exc
        self.calls = 0

    def align_chunks(self, audio, chunks, language=None):
        self.calls += 1
        if self._exc:
            raise self._exc
        return self._words, {"total_windows": 1, "failed_windows": 0,
                             "total_words": len(self._words), "failures": []}


def test_handle_conn_aligns_and_responds():
    """正常请求 → 回 ok + words + stats."""
    a, b = socket.socketpair()
    aligner = _FakeAligner(words=[{"text": "你", "start": 0.1, "end": 0.3, "score": -1.0}])
    try:
        sc.send_msg(a, {"audio_path": "/x.wav", "chunks": [{"start": 0, "end": 1, "text": "你"}], "language": "chi"})
        retire = sc.handle_conn(b, aligner=aligner, audio_loader=lambda p: object())
        resp = sc.recv_msg(a)
        assert resp["ok"] is True
        assert resp["words"] == [{"text": "你", "start": 0.1, "end": 0.3, "score": -1.0}]
        assert resp["stats"]["total_words"] == 1
        assert retire is False
        assert aligner.calls == 1
    finally:
        a.close(); b.close()


def test_handle_conn_ping():
    """ping 命令 → pong, 不调 aligner."""
    a, b = socket.socketpair()
    aligner = _FakeAligner()
    try:
        sc.send_msg(a, {"cmd": "ping"})
        retire = sc.handle_conn(b, aligner=aligner, audio_loader=lambda p: object())
        resp = sc.recv_msg(a)
        assert resp == {"ok": True, "pong": True}
        assert retire is False
        assert aligner.calls == 0
    finally:
        a.close(); b.close()


def test_handle_conn_resource_error_signals_retire():
    """CUDA 资源错误 → 回 ok=false+resource_error=true, retire=True (codex #8 退休)."""
    a, b = socket.socketpair()
    aligner = _FakeAligner(exc=RuntimeError("CUBLAS failure 3: the resource allocation failed"))
    try:
        sc.send_msg(a, {"audio_path": "/x.wav", "chunks": [{"start": 0, "end": 1, "text": "你"}], "language": "chi"})
        retire = sc.handle_conn(b, aligner=aligner, audio_loader=lambda p: object())
        resp = sc.recv_msg(a)
        assert resp["ok"] is False
        assert resp["resource_error"] is True
        assert retire is True
    finally:
        a.close(); b.close()


def test_handle_conn_normal_error_no_retire():
    """普通对齐错误 (非资源) → ok=false+resource_error=false, retire=False."""
    a, b = socket.socketpair()
    aligner = _FakeAligner(exc=RuntimeError("weird text"))
    try:
        sc.send_msg(a, {"audio_path": "/x.wav", "chunks": [{"start": 0, "end": 1, "text": "你"}], "language": "chi"})
        retire = sc.handle_conn(b, aligner=aligner, audio_loader=lambda p: object())
        resp = sc.recv_msg(a)
        assert resp["ok"] is False
        assert resp["resource_error"] is False
        assert retire is False
    finally:
        a.close(); b.close()


# ──────────────────────────────────────────────────────────────────────────
# serve(): bind UDS + accept loop + idle TTL 退出
# ──────────────────────────────────────────────────────────────────────────
def _short_sock(name: str) -> str:
    """UDS 短路径 (macOS sun_path ≤104). pytest tmp_path 太深, 用 /tmp."""
    return os.path.join("/tmp", f"funasr_wa_test_{os.getpid()}_{name}.sock")


def test_serve_idle_ttl_exits():
    """无请求, accept 在 idle_ttl 超时 → serve 返回 (进程随之退出释放 VRAM)."""
    sock_path = _short_sock("idle")
    built = {"n": 0}

    def factory():
        built["n"] += 1
        return _FakeAligner()

    # idle_ttl 极短, 无客户端连接 → 立即超时退出
    sc.serve(sock_path, aligner_factory=factory, audio_loader=lambda p: object(), idle_ttl_sec=0.2)
    # 无请求 → aligner 从未 build (lazy)
    assert built["n"] == 0


def test_serve_handles_request_then_idle_exits():
    """serve 在后台线程跑: 连上发一个请求拿到结果, 之后 idle TTL 退出."""
    sock_path = _short_sock("req")
    aligner = _FakeAligner(words=[{"text": "甲", "start": 0.0, "end": 0.2, "score": -1.0}])

    t = threading.Thread(
        target=sc.serve,
        kwargs=dict(sock_path=sock_path, aligner_factory=lambda: aligner,
                    audio_loader=lambda p: object(), idle_ttl_sec=0.5),
        daemon=True,
    )
    t.start()
    # 等 socket 就绪
    assert sc.wait_for_socket(sock_path, timeout=2.0) is True

    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(sock_path)
    sc.send_msg(conn, {"audio_path": "/x.wav", "chunks": [{"start": 0, "end": 1, "text": "甲"}], "language": "chi"})
    resp = sc.recv_msg(conn)
    conn.close()
    assert resp["ok"] is True
    assert resp["words"][0]["text"] == "甲"

    # idle TTL 后 serve 线程退出
    t.join(timeout=3.0)
    assert not t.is_alive()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
