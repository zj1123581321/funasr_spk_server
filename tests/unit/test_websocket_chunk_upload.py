"""
TODOS #24 分片上传安全网 — Lane C1: `_handle_chunk_upload` 逐帧接收错误分支

交接文档判定「分片逐帧」是事故同型的零覆盖盲区。`test_websocket_finalize_resilience.py`
已覆盖 finalize 阶段(queue_full / 幂等 / 真错误清理 / TTL sweep),但**逐帧接收**
(`_handle_chunk_upload`)的错误分支此前无任何 unit 覆盖。本文件补齐:

1. 缺 task_id / chunk_index        → missing_chunk_data
2. session 不存在                  → session_not_found
3. 重复分片                        → chunk_received(status=duplicate), 不重复落盘
4. 缺 chunk_data                   → missing_chunk_data
5. 分片 hash 校验失败              → chunk_hash_mismatch, 不计入 chunks_received
6. 正常分片                        → chunk_received(status=received) + 写盘 + 计数
7. 最后一片收齐                    → 触发 _finalize_chunked_upload

这些是 characterization 测试(钉现有行为),mock `_send_message` / `_send_error`
与 `_finalize_chunked_upload` 隔离,不连真 server。
"""
import base64
import hashlib
import time
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from src.api.websocket_handler import WebSocketHandler


@pytest.fixture
def handler():
    return WebSocketHandler()


@pytest.fixture
def fake_ws():
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.remote_address = ("127.0.0.1", 1234)
    return ws


def make_uploading_session(handler, tmp_path, *, total_chunks=2, chunk_size=4):
    """构造一个 uploading 状态、尚未收齐的分片 session(temp 文件已存在供 r+b 写入)"""
    temp = tmp_path / "chunk.bin"
    temp.write_bytes(b"")  # r+b 要求文件存在
    task_id = "task-chunk"
    handler.upload_sessions[task_id] = {
        "task_id": task_id, "file_name": "a.wav", "file_size": 64,
        "file_hash": "deadbeef",
        "chunk_size": chunk_size, "total_chunks": total_chunks, "received_chunks": 0,
        "temp_file_path": str(temp), "chunks_received": set(),
        "output_format": "json", "force_refresh": True,
        "connection_id": "c1", "engine": "qwen3", "language": None,
        "diarize": True, "word_align": None,
        "state": "uploading", "created_at": time.time(),
        "finalized_file_path": None,
    }
    return task_id, temp


def chunk_frame(task_id, chunk_index, payload: bytes, *, with_hash=True, bad_hash=False):
    """构造一帧 upload_chunk data"""
    data = {
        "task_id": task_id,
        "chunk_index": chunk_index,
        "chunk_data": base64.b64encode(payload).decode(),
    }
    if with_hash:
        h = hashlib.md5(payload).hexdigest()
        data["chunk_hash"] = "ff" + h[2:] if bad_hash else h
    return data


def err_types(send_err_mock):
    return [c.args[1] for c in send_err_mock.call_args_list if len(c.args) >= 2]


def msg_calls(send_msg_mock):
    return [c for c in send_msg_mock.call_args_list if len(c.args) >= 2]


class TestChunkUploadErrorBranches:
    @pytest.mark.asyncio
    async def test_missing_task_id(self, handler, fake_ws):
        with patch.object(handler, "_send_message", new=AsyncMock()), \
             patch.object(handler, "_send_error", new=AsyncMock()) as se:
            await handler._handle_chunk_upload(fake_ws, "c1", {"chunk_index": 0})
            assert "missing_chunk_data" in err_types(se)

    @pytest.mark.asyncio
    async def test_missing_chunk_index(self, handler, fake_ws):
        with patch.object(handler, "_send_message", new=AsyncMock()), \
             patch.object(handler, "_send_error", new=AsyncMock()) as se:
            await handler._handle_chunk_upload(fake_ws, "c1", {"task_id": "t"})
            assert "missing_chunk_data" in err_types(se)

    @pytest.mark.asyncio
    async def test_session_not_found(self, handler, fake_ws):
        with patch.object(handler, "_send_message", new=AsyncMock()), \
             patch.object(handler, "_send_error", new=AsyncMock()) as se:
            await handler._handle_chunk_upload(
                fake_ws, "c1", {"task_id": "ghost", "chunk_index": 0})
            assert "session_not_found" in err_types(se)

    @pytest.mark.asyncio
    async def test_duplicate_chunk(self, handler, fake_ws, tmp_path):
        task_id, temp = make_uploading_session(handler, tmp_path)
        # 预置已收到 chunk 0
        handler.upload_sessions[task_id]["chunks_received"] = {0}
        handler.upload_sessions[task_id]["received_chunks"] = 1
        with patch.object(handler, "_send_message", new=AsyncMock()) as sm, \
             patch.object(handler, "_send_error", new=AsyncMock()) as se:
            await handler._handle_chunk_upload(
                fake_ws, "c1", chunk_frame(task_id, 0, b"AAAA"))
            # 发 chunk_received(status=duplicate)
            dup = next(c for c in msg_calls(sm) if c.args[1] == "chunk_received")
            assert dup.args[2]["status"] == "duplicate"
            # 计数不变(没重复落盘)
            assert handler.upload_sessions[task_id]["received_chunks"] == 1
            assert not err_types(se)

    @pytest.mark.asyncio
    async def test_missing_chunk_data(self, handler, fake_ws, tmp_path):
        task_id, temp = make_uploading_session(handler, tmp_path)
        with patch.object(handler, "_send_message", new=AsyncMock()), \
             patch.object(handler, "_send_error", new=AsyncMock()) as se:
            # 有 task_id/chunk_index 但无 chunk_data
            await handler._handle_chunk_upload(
                fake_ws, "c1", {"task_id": task_id, "chunk_index": 0})
            assert "missing_chunk_data" in err_types(se)

    @pytest.mark.asyncio
    async def test_hash_mismatch(self, handler, fake_ws, tmp_path):
        task_id, temp = make_uploading_session(handler, tmp_path)
        with patch.object(handler, "_send_message", new=AsyncMock()), \
             patch.object(handler, "_send_error", new=AsyncMock()) as se:
            await handler._handle_chunk_upload(
                fake_ws, "c1", chunk_frame(task_id, 0, b"AAAA", bad_hash=True))
            assert "chunk_hash_mismatch" in err_types(se)
            # 坏分片不计入
            assert 0 not in handler.upload_sessions[task_id]["chunks_received"]
            assert handler.upload_sessions[task_id]["received_chunks"] == 0


class TestChunkUploadHappyPath:
    @pytest.mark.asyncio
    async def test_valid_chunk_recorded_and_written(self, handler, fake_ws, tmp_path):
        task_id, temp = make_uploading_session(handler, tmp_path, total_chunks=2, chunk_size=4)
        with patch.object(handler, "_send_message", new=AsyncMock()) as sm, \
             patch.object(handler, "_send_error", new=AsyncMock()) as se, \
             patch.object(handler, "_finalize_chunked_upload", new=AsyncMock()) as fz:
            await handler._handle_chunk_upload(
                fake_ws, "c1", chunk_frame(task_id, 0, b"AAAA"))
            # chunk_received(status=received)
            rec = next(c for c in msg_calls(sm) if c.args[1] == "chunk_received")
            assert rec.args[2]["status"] == "received"
            # 计入会话
            assert 0 in handler.upload_sessions[task_id]["chunks_received"]
            assert handler.upload_sessions[task_id]["received_chunks"] == 1
            # 落盘到正确 offset
            assert temp.read_bytes()[:4] == b"AAAA"
            # 未收齐 → 不 finalize
            fz.assert_not_awaited()
            assert not err_types(se)

    @pytest.mark.asyncio
    async def test_no_hash_provided_still_accepts(self, handler, fake_ws, tmp_path):
        """chunk_hash 缺省时跳过校验直接接受(协议允许 client 不带 hash)"""
        task_id, temp = make_uploading_session(handler, tmp_path, total_chunks=2)
        with patch.object(handler, "_send_message", new=AsyncMock()) as sm, \
             patch.object(handler, "_send_error", new=AsyncMock()) as se, \
             patch.object(handler, "_finalize_chunked_upload", new=AsyncMock()):
            await handler._handle_chunk_upload(
                fake_ws, "c1", chunk_frame(task_id, 0, b"BBBB", with_hash=False))
            rec = next(c for c in msg_calls(sm) if c.args[1] == "chunk_received")
            assert rec.args[2]["status"] == "received"
            assert not err_types(se)

    @pytest.mark.asyncio
    async def test_final_chunk_triggers_finalize(self, handler, fake_ws, tmp_path):
        task_id, temp = make_uploading_session(handler, tmp_path, total_chunks=1, chunk_size=4)
        with patch.object(handler, "_send_message", new=AsyncMock()), \
             patch.object(handler, "_send_error", new=AsyncMock()), \
             patch.object(handler, "_finalize_chunked_upload", new=AsyncMock()) as fz:
            await handler._handle_chunk_upload(
                fake_ws, "c1", chunk_frame(task_id, 0, b"CCCC"))
            # 收齐 → 触发 finalize
            fz.assert_awaited_once()
