"""回归: funasr 的 raw_result 是 list，_process_task JSON 分支不能崩

生产事故 (2026-06-16): funasr 长音频 JSON 任务转录成功后, task_manager._process_task
的 `word_align_error_msg = (raw_result.get("word_align") or {}).get("error")` 对
**list 类型的 raw_result** 调 .get() → `'list' object has no attribute 'get'`,
任务被判失败 + 重试耗尽。

根因: word_align 落地时该行假设 raw_result 永远是 dict (qwen3 形态), 但 funasr 的
model.generate() 返回 list (funasr_transcriber.py:364 `return (transcription_result, result)`,
result 是 funasr 原始 list)。

复用 test_task_manager_engine_propagation 的 _process_task mock 框架。
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_funasr_list_raw_result_completes_not_fails(tmp_path):
    """funasr raw_result=list → 结果处理不崩, 任务正常 COMPLETED"""
    from src.core.task_manager import TaskManager
    from src.models.schemas import (
        TranscriptionTask, TaskStatus, TranscriptionResult, TranscriptionSegment,
    )

    mgr = TaskManager()
    task = TranscriptionTask(
        task_id="funasr-list-1",
        file_name="x.wav",
        file_path=str(tmp_path / "x.wav"),
        file_size=100,
        file_hash="h",
        engine="funasr",
        output_format="json",
    )
    mgr.tasks[task.task_id] = task
    (tmp_path / "x.wav").write_bytes(b"\x00" * 100)

    fake_result = TranscriptionResult(
        task_id=task.task_id, file_name="x.wav", file_hash="h", duration=1.0,
        segments=[TranscriptionSegment(start_time=0, end_time=1, text="hi", speaker="Speaker1")],
        speakers=["Speaker1"], processing_time=0.1,
    )
    # funasr model.generate() 真实返回形态: list[dict]
    fake_raw = [{"key": "x", "text": "hi", "sentence_info": []}]
    fake_transcriber = MagicMock()
    fake_transcriber.transcribe = AsyncMock(return_value=(fake_result, fake_raw))

    with patch("src.core.transcriber_dispatch.resolve_transcriber") as mock_resolve, \
         patch("src.core.task_manager.db_manager") as mock_db:
        mock_resolve.return_value = fake_transcriber
        mock_db.save_result = AsyncMock()
        with patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
             patch.object(mgr, "_notify_task_complete", new=AsyncMock()):
            await mgr._process_task(task.task_id)

    # 修复前: 'list' object has no attribute 'get' → FAILED/重试; 修复后: COMPLETED
    assert task.status == TaskStatus.COMPLETED, f"任务不应失败, 实际 status={task.status}, error={task.error}"
    assert task.error is None
    # funasr 无 word_align, metadata.word_align_error 应为 None
    assert (task.result.metadata or {}).get("word_align_error") is None


@pytest.mark.asyncio
async def test_qwen3_dict_raw_result_still_extracts_word_align_error(tmp_path):
    """守护: qwen3 dict raw_result 仍能正常取 word_align error (修复不能误伤 dict 路径)"""
    from src.core.task_manager import TaskManager
    from src.models.schemas import (
        TranscriptionTask, TaskStatus, TranscriptionResult, TranscriptionSegment,
    )

    mgr = TaskManager()
    task = TranscriptionTask(
        task_id="qwen3-dict-1", file_name="x.wav",
        file_path=str(tmp_path / "x.wav"), file_size=100, file_hash="h2",
        engine="qwen3", output_format="json",
    )
    task.options.word_align = True
    mgr.tasks[task.task_id] = task
    (tmp_path / "x.wav").write_bytes(b"\x00" * 100)

    fake_result = TranscriptionResult(
        task_id=task.task_id, file_name="x.wav", file_hash="h2", duration=1.0,
        segments=[TranscriptionSegment(start_time=0, end_time=1, text="hi", speaker="Speaker1")],
        speakers=["Speaker1"], processing_time=0.1,
    )
    # qwen3 raw_result 是 dict, 含 word_align 失败信息
    fake_raw = {"sentence_info": [], "word_align": {"error": "CUDA OOM fallback failed"}}
    fake_transcriber = MagicMock()
    fake_transcriber.transcribe = AsyncMock(return_value=(fake_result, fake_raw))

    with patch("src.core.transcriber_dispatch.resolve_transcriber") as mock_resolve, \
         patch("src.core.task_manager.db_manager") as mock_db:
        mock_resolve.return_value = fake_transcriber
        mock_db.save_result = AsyncMock()
        with patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
             patch.object(mgr, "_notify_task_complete", new=AsyncMock()):
            await mgr._process_task(task.task_id)

    assert task.status == TaskStatus.COMPLETED
    assert (task.result.metadata or {}).get("word_align_error") == "CUDA OOM fallback failed"
