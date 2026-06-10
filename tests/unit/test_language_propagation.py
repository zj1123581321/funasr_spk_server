"""
词级时间戳 — per-request language 字段穿透 schema→websocket→task_manager→transcriber

ISO 码 (chi/eng/jpn/kor...) 由客户端按请求带, 驱动 word_align 语言; config
word_align_language 兜底. 本测试钉死透传链各环节.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.schemas import FileUploadRequest, TranscriptionTask
from src.api.websocket_handler import WebSocketHandler


# ==================== schema ====================


def test_file_upload_request_language_default_none():
    req = FileUploadRequest(file_name="a.wav", file_size=1, file_hash="h")
    assert req.language is None


def test_file_upload_request_language_set():
    req = FileUploadRequest(file_name="a.wav", file_size=1, file_hash="h", language="eng")
    assert req.language == "eng"


def test_transcription_task_language_default_none():
    # 1a 重构后 language 收进嵌套 options (平铺字段已删, 见 test_transcribe_options.py)
    task = TranscriptionTask(
        task_id="t", file_name="a.wav", file_path="", file_size=1, file_hash="h"
    )
    assert task.options.language is None


# ==================== task_manager.create_task ====================


@pytest.mark.asyncio
async def test_create_task_propagates_language():
    from src.core.task_manager import TaskManager

    tm = TaskManager()
    req = FileUploadRequest(
        file_name="a.wav", file_size=1, file_hash="h", language="jpn"
    )
    task = await tm.create_task(req, task_id="t1")
    assert task.options.language == "jpn"


# ==================== websocket chunked session ====================


@pytest.fixture
def fake_ws():
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.remote_address = ("127.0.0.1", 1234)
    return ws


@pytest.mark.asyncio
async def test_chunked_session_captures_language(fake_ws):
    handler = WebSocketHandler()
    data = {
        "file_name": "x.wav",
        "file_size": 10000,
        "file_hash": "h-x",
        "total_chunks": 1,
        "language": "kor",
    }
    await handler._handle_chunked_upload_request(fake_ws, "conn-1", data)
    session = next(iter(handler.upload_sessions.values()))
    assert session.get("language") == "kor"


# ==================== pool transcribe 转发 language ====================


@pytest.mark.asyncio
async def test_inproc_pool_forwards_language():
    from src.core.qwen3_inproc_pool import Qwen3InProcPool

    inner = MagicMock()
    inner.initialize = AsyncMock(return_value=None)
    inner.transcribe = AsyncMock(return_value="R")
    pool = Qwen3InProcPool(pool_size=1, transcriber_factory=lambda: inner)
    await pool.initialize()
    await pool.transcribe(
        audio_path="a.wav", task_id="t", progress_callback=None,
        output_format="json", language="eng",
    )
    _, kwargs = inner.transcribe.call_args
    assert kwargs.get("language") == "eng"
