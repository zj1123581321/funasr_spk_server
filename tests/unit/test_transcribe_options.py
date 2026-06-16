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
    # word_align 解析后的确定 bool, 默认关 (与 config 兜底默认 False 一致)
    assert opts.word_align is False


# ==================== word_align per-request (2026-06-16 显存落地) ====================


def test_file_upload_request_word_align_defaults_none():
    """请求级 word_align 是 Optional[bool], 不传=None (区分'未指定'与'显式关'),
    老客户端不带此字段行为零变化."""
    req = FileUploadRequest(file_name="a.wav", file_size=1, file_hash="h")
    assert req.word_align is None


def test_resolve_word_align_precedence():
    """优先级单一事实源 (决策 1A): 请求值非 None 胜出, 否则 config 兜底."""
    from src.models.schemas import resolve_word_align

    # 请求未指定 → 跟随 config 兜底
    assert resolve_word_align(None, config_default=False) is False
    assert resolve_word_align(None, config_default=True) is True
    # 请求显式 → 压过 config (显式关也压过 config 开)
    assert resolve_word_align(True, config_default=False) is True
    assert resolve_word_align(False, config_default=True) is False


def test_old_task_file_without_word_align_defaults_false():
    """REGRESSION: file-based pool 老 .task 文件无 word_align 字段, 反序列化走默认 False, 不崩."""
    # 老协议: options dict 无 word_align
    assert TranscribeOptions(**{"language": "eng"}).word_align is False
    # 更老协议: 平铺 language 构造 (worker 兜底路径)
    assert TranscribeOptions(language="eng").word_align is False


def test_options_serializes_word_align_via_model_dump():
    opts = TranscribeOptions(word_align=True)
    d = opts.model_dump()
    assert d["word_align"] is True
    assert TranscribeOptions(**d).word_align is True


# ==================== create_task 解析 effective word_align ====================


@pytest.mark.asyncio
async def test_create_task_resolves_word_align_explicit_true():
    from src.core.task_manager import TaskManager

    tm = TaskManager()
    req = FileUploadRequest(
        file_name="a.wav", file_size=1, file_hash="h-wa1", word_align=True
    )
    task = await tm.create_task(req, task_id="t-wa1")
    assert task.options.word_align is True


@pytest.mark.asyncio
async def test_create_task_resolves_word_align_none_follows_config():
    from src.core.task_manager import TaskManager
    from src.core.config import config

    tm = TaskManager()
    req = FileUploadRequest(file_name="a.wav", file_size=1, file_hash="h-wa2")
    original = config.qwen3.word_align_enabled
    config.qwen3.word_align_enabled = True
    try:
        task = await tm.create_task(req, task_id="t-wa2")
        assert task.options.word_align is True
    finally:
        config.qwen3.word_align_enabled = original


@pytest.mark.asyncio
async def test_create_task_explicit_false_overrides_config_on():
    from src.core.task_manager import TaskManager
    from src.core.config import config

    tm = TaskManager()
    req = FileUploadRequest(
        file_name="a.wav", file_size=1, file_hash="h-wa3", word_align=False
    )
    original = config.qwen3.word_align_enabled
    config.qwen3.word_align_enabled = True
    try:
        task = await tm.create_task(req, task_id="t-wa3")
        assert task.options.word_align is False
    finally:
        config.qwen3.word_align_enabled = original


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
