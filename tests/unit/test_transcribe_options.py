"""TranscribeOptions 收拢 (diarize 开关 E3/D1) — per-request 选项结构整体穿透

1a 纯重构: TranscriptionTask 挂嵌套 options: TranscribeOptions, 删平铺 language,
所有读点 (task_manager / handler session 回填 / cache helpers) 改走 task.options.
行为零变: language 透传链行为与 test_language_propagation 一致.
"""
from __future__ import annotations

from unittest.mock import patch, AsyncMock

import pytest

from src.models.schemas import FileUploadRequest, TranscribeOptions, TranscriptionTask


# ==================== schema ====================


def test_transcribe_options_defaults():
    opts = TranscribeOptions()
    assert opts.language is None


def test_transcription_task_has_nested_options_default():
    task = TranscriptionTask(
        task_id="t", file_name="a.wav", file_path="", file_size=1, file_hash="h"
    )
    assert isinstance(task.options, TranscribeOptions)
    assert task.options.language is None


def test_transcription_task_flat_language_removed():
    """平铺 language 字段删除 — options 是唯一 source of truth (内部模型无兼容约束)"""
    assert "language" not in TranscriptionTask.model_fields


def test_options_serializes_via_model_dump():
    """file-based pool 跨进程边界用 model_dump() 序列化"""
    opts = TranscribeOptions(language="eng")
    d = opts.model_dump()
    assert d["language"] == "eng"
    # 反序列化回 model (worker 端兜底路径)
    assert TranscribeOptions(**d).language == "eng"


# ==================== task_manager.create_task ====================


@pytest.mark.asyncio
async def test_create_task_builds_options_from_request():
    from src.core.task_manager import TaskManager

    tm = TaskManager()
    req = FileUploadRequest(
        file_name="a.wav", file_size=1, file_hash="h", language="jpn"
    )
    task = await tm.create_task(req, task_id="t1")
    assert task.options.language == "jpn"


# ==================== cache helpers 读 task.options ====================


@pytest.mark.asyncio
async def test_cache_lookup_uses_options_language(tmp_path):
    """submit_task 的 cache lookup engine tag 要折 options.language (word_align 开时)"""
    from src.core.task_manager import TaskManager
    from src.core.config import config

    tm = TaskManager()
    req = FileUploadRequest(
        file_name="a.wav", file_size=1, file_hash="h-opt", language="eng",
        engine=config.transcription.default_engine,
    )
    await tm.create_task(req, task_id="t-cache")
    fake_file = tmp_path / "x.wav"
    fake_file.write_bytes(b"\x00" * 10)

    original_wa = config.qwen3.word_align_enabled
    original_engine = config.transcription.default_engine
    config.qwen3.word_align_enabled = True
    try:
        tm.tasks["t-cache"].engine = "qwen3"
        with patch("src.core.task_manager.db_manager") as mock_db:
            mock_db.get_cached_result = AsyncMock(return_value=None)
            await tm.submit_task("t-cache", str(fake_file))
            engine_arg = mock_db.get_cached_result.call_args.kwargs.get("engine")
            assert engine_arg == "qwen3+wa:eng", f"cache tag 应折 options.language: {engine_arg}"
    finally:
        config.qwen3.word_align_enabled = original_wa
        config.transcription.default_engine = original_engine
