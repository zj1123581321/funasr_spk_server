"""#3 worker 接线: _process_task 失败路径用 classify_error 替代字符串匹配

- record_error(kind.value): errors_total 按 kind 分(非全归 engine_error)
- should_retry = kind.retryable
- 模型重置 codex #13: 仅 task.engine=="funasr" 才 _try_reset_model(多引擎安全 —
  qwen3 错误判 MODEL_ERROR 不该去 reset 无关的 funasr)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.task_manager import TaskManager
from src.models.schemas import TranscribeOptions, TranscriptionTask


def _task(tmp_path, engine="funasr"):
    f = tmp_path / "x.wav"
    f.write_bytes(b"\x00" * 10)
    return TranscriptionTask(
        task_id="t", file_name="x.wav", file_path=str(f), file_size=10,
        file_hash="h", engine=engine, options=TranscribeOptions(),
    )


async def _run_with_transcribe_error(mgr, exc, tmp_path, engine="funasr"):
    task = _task(tmp_path, engine)
    mgr.tasks["t"] = task
    fake_tr = MagicMock()
    fake_tr.transcribe = AsyncMock(side_effect=exc)
    with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake_tr), \
         patch("src.core.task_manager.db_manager") as mdb, \
         patch("src.core.task_manager.asyncio.sleep", new=AsyncMock()):
        mdb.get_cached_result = AsyncMock(return_value=None)  # 不命中, 走转录
        mdb.save_result = AsyncMock()
        with patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
             patch.object(mgr, "_notify_task_complete", new=AsyncMock()), \
             patch.object(mgr, "_notify_task_failed", new=AsyncMock()), \
             patch.object(mgr, "_maybe_delete_task_file", new=AsyncMock()), \
             patch.object(mgr, "_send_wework_notification", new=AsyncMock()), \
             patch.object(mgr, "_try_reset_model", new=AsyncMock()) as reset:
            await mgr._process_task("t")
    return task, reset


@pytest.mark.asyncio
async def test_non_retryable_input_fails_without_retry(tmp_path):
    mgr = TaskManager()
    task, reset = await _run_with_transcribe_error(mgr, Exception("音频时长过短"), tmp_path)
    assert task.status.value == "failed"  # 不重试直接终态
    assert mgr.get_metrics_snapshot()["errors_total"].get("non_retryable_input") == 1
    reset.assert_not_called()


@pytest.mark.asyncio
async def test_model_error_funasr_retries_and_resets(tmp_path):
    mgr = TaskManager()
    task, reset = await _run_with_transcribe_error(
        mgr, RuntimeError("VAD algorithm failure"), tmp_path, engine="funasr"
    )
    assert task.status.value == "pending"  # 重试重入队
    reset.assert_called_once()             # funasr 才 reset
    assert mgr.get_metrics_snapshot()["errors_total"].get("model_error") == 1


@pytest.mark.asyncio
async def test_model_error_qwen3_retries_without_resetting_funasr(tmp_path):
    """codex #13: qwen3 任务的 MODEL_ERROR 仍重试, 但不去 reset 无关的 funasr"""
    mgr = TaskManager()
    task, reset = await _run_with_transcribe_error(
        mgr, RuntimeError("VAD algorithm failure"), tmp_path, engine="qwen3"
    )
    assert task.status.value == "pending"  # kind.retryable 仍重试
    reset.assert_not_called()              # 但不 reset funasr


@pytest.mark.asyncio
async def test_engine_error_retries(tmp_path):
    mgr = TaskManager()
    task, reset = await _run_with_transcribe_error(mgr, Exception("something weird"), tmp_path)
    assert task.status.value == "pending"
    assert mgr.get_metrics_snapshot()["errors_total"].get("engine_error") == 1
