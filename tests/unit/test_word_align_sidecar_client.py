"""Lane 2 (#18) — word_align sidecar 主进程侧 client manager 单测.

client 编排逻辑 (生命周期/降级), 网络层 (request) + 进程 (Popen) + 就绪
(wait_for_socket) 全 mock, 不起真 subprocess:
- align 成功 → (words, stats)
- sidecar 未起 → 先 spawn (Popen + wait_for_socket) 再请求
- 对齐超时 → 杀 sidecar + raise SidecarTimeout (codex #4 不双跑)
- CUDA 资源错误 → 杀 sidecar (退休) + raise SidecarResourceError (codex #8)
- 普通对齐错误 → raise SidecarAlignError (不杀)
- 连接被拒 (sidecar 刚 idle 退出) → respawn 重试一次成功
- spawn 后 socket 未就绪 → SidecarUnavailable
- 进程全局单例 (codex #6): get 两次同一实例, pool>1 也只一个 sidecar
"""
from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest

from src.core.qwen3 import word_align_sidecar as sc


@pytest.fixture(autouse=True)
def _reset_singleton():
    sc.reset_word_align_sidecar_client()
    yield
    sc.reset_word_align_sidecar_client()


def _make_client(**overrides):
    kwargs = dict(
        sock_path="/tmp/funasr_wa_test_client.sock",
        spawn_cmd=["python", "-m", "fake.sidecar"],
        align_timeout_sec=30.0,
        connect_timeout_sec=2.0,
        ready_timeout_sec=5.0,
    )
    kwargs.update(overrides)
    return sc.WordAlignSidecarClient(**kwargs)


def _ok_resp(words):
    return {"ok": True, "words": words,
            "stats": {"total_windows": 1, "failed_windows": 0, "total_words": len(words), "failures": []}}


def test_align_success_spawns_then_requests():
    """首次 align: sidecar 未起 → spawn (Popen + 就绪等待) → 请求 → 返回 words."""
    client = _make_client()
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None  # alive
    with patch.object(sc.subprocess, "Popen", return_value=fake_proc) as popen, \
         patch.object(sc, "wait_for_socket", return_value=True) as ready, \
         patch.object(sc, "request", return_value=_ok_resp([{"text": "你", "start": 0.1, "end": 0.3, "score": -1.0}])) as req:
        words, stats = client.align("/x.wav", [{"start": 0, "end": 1, "text": "你"}], "chi")
    popen.assert_called_once()
    ready.assert_called_once()
    req.assert_called_once()
    assert words == [{"text": "你", "start": 0.1, "end": 0.3, "score": -1.0}]
    assert stats["total_words"] == 1


def test_align_reuses_running_sidecar():
    """sidecar 已活 (ping 通) → 不重复 spawn."""
    client = _make_client()
    fake_proc = MagicMock(); fake_proc.poll.return_value = None
    with patch.object(sc.subprocess, "Popen", return_value=fake_proc) as popen, \
         patch.object(sc, "wait_for_socket", return_value=True), \
         patch.object(sc, "ping", return_value=True), \
         patch.object(sc, "request", return_value=_ok_resp([{"text": "甲", "start": 0, "end": 0.2, "score": -1.0}])):
        client.align("/x.wav", [{"start": 0, "end": 1, "text": "甲"}], "chi")
        client.align("/x.wav", [{"start": 0, "end": 1, "text": "甲"}], "chi")
    popen.assert_called_once()  # 只 spawn 一次


def test_align_timeout_kills_sidecar():
    """对齐超时 → 杀 sidecar + SidecarTimeout (codex #4: 杜绝 CUDA+CPU 双跑)."""
    client = _make_client()
    fake_proc = MagicMock(); fake_proc.poll.return_value = None
    with patch.object(sc.subprocess, "Popen", return_value=fake_proc), \
         patch.object(sc, "wait_for_socket", return_value=True), \
         patch.object(sc, "request", side_effect=socket.timeout("align timed out")):
        with pytest.raises(sc.SidecarTimeout):
            client.align("/x.wav", [{"start": 0, "end": 1, "text": "你"}], "chi")
    fake_proc.kill.assert_called()


def test_align_resource_error_retires_sidecar():
    """CUDA 资源错误 → 杀 sidecar (退休) + SidecarResourceError (codex #8)."""
    client = _make_client()
    fake_proc = MagicMock(); fake_proc.poll.return_value = None
    bad = {"ok": False, "error": "CUBLAS failure 3", "resource_error": True}
    with patch.object(sc.subprocess, "Popen", return_value=fake_proc), \
         patch.object(sc, "wait_for_socket", return_value=True), \
         patch.object(sc, "request", return_value=bad):
        with pytest.raises(sc.SidecarResourceError):
            client.align("/x.wav", [{"start": 0, "end": 1, "text": "你"}], "chi")
    fake_proc.kill.assert_called()


def test_align_normal_error_does_not_kill():
    """普通对齐错误 → SidecarAlignError, 不杀 sidecar (它还健康)."""
    client = _make_client()
    fake_proc = MagicMock(); fake_proc.poll.return_value = None
    bad = {"ok": False, "error": "weird text", "resource_error": False}
    with patch.object(sc.subprocess, "Popen", return_value=fake_proc), \
         patch.object(sc, "wait_for_socket", return_value=True), \
         patch.object(sc, "request", return_value=bad):
        with pytest.raises(sc.SidecarAlignError):
            client.align("/x.wav", [{"start": 0, "end": 1, "text": "你"}], "chi")
    fake_proc.kill.assert_not_called()


def test_align_connection_refused_respawns_and_retries():
    """sidecar 刚 idle 退出 (连接被拒) → respawn 重试一次成功."""
    client = _make_client()
    fake_proc = MagicMock(); fake_proc.poll.return_value = None
    # 第一次 request 连接被拒 (sidecar 退出), 第二次成功
    responses = [ConnectionRefusedError("refused"), _ok_resp([{"text": "乙", "start": 6, "end": 6.3, "score": -1.0}])]

    def _req(*a, **k):
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    with patch.object(sc.subprocess, "Popen", return_value=fake_proc) as popen, \
         patch.object(sc, "wait_for_socket", return_value=True), \
         patch.object(sc, "ping", return_value=True), \
         patch.object(sc, "request", side_effect=_req):
        words, _ = client.align("/x.wav", [{"start": 5, "end": 7, "text": "乙"}], "chi")
    assert words[0]["text"] == "乙"
    assert popen.call_count == 2  # 初次 spawn + respawn


def test_spawn_socket_not_ready_raises_unavailable():
    """spawn 后 socket 超时未就绪 → SidecarUnavailable."""
    client = _make_client()
    fake_proc = MagicMock(); fake_proc.poll.return_value = None
    with patch.object(sc.subprocess, "Popen", return_value=fake_proc), \
         patch.object(sc, "wait_for_socket", return_value=False):
        with pytest.raises(sc.SidecarUnavailable):
            client.align("/x.wav", [{"start": 0, "end": 1, "text": "你"}], "chi")


def test_singleton_is_process_global():
    """get 两次返回同一实例 (codex #6: pool>1 也只一个 sidecar)."""
    with patch.object(sc, "_build_default_client", side_effect=lambda: _make_client()):
        c1 = sc.get_word_align_sidecar_client()
        c2 = sc.get_word_align_sidecar_client()
    assert c1 is c2


def test_shutdown_kills_proc():
    """shutdown → 杀 sidecar 进程 (服务停时清理)."""
    client = _make_client()
    fake_proc = MagicMock(); fake_proc.poll.return_value = None
    with patch.object(sc.subprocess, "Popen", return_value=fake_proc), \
         patch.object(sc, "wait_for_socket", return_value=True), \
         patch.object(sc, "ping", return_value=True), \
         patch.object(sc, "request", return_value=_ok_resp([])):
        client.align("/x.wav", [{"start": 0, "end": 1, "text": "你"}], "chi")
    client.shutdown()
    fake_proc.kill.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
