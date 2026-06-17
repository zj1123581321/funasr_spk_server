"""#22 in-flight 去重(修法一: worker 开工前二次查缓存)

错开到达场景: 同 file_hash 的前序任务在本任务排队期间已转完写缓存, _process_task
真转录前再查一次缓存命中即秒返回, 免重复转录(submit 时查那次还是空的)。

codex review 加固点:
- #1 防泄漏: 二次查缓存自身抛错绝不逃逸到 _worker, 降级继续转录(否则任务卡 PENDING
  / 不再入队 / 看门狗不管 → 永久泄漏, 同型 task_create_before_queue_check_leak)。
- #2 progress: 缓存命中终态化也 set progress=100(否则轮询见 completed+progress=0)。
- #3 清 error: 失败重试后第二次缓存命中成功须清 task.error(否则残留旧错误)。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.task_manager import TaskManager
from src.models.schemas import (
    TranscribeOptions,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionTask,
)


def _task(tmp_path, task_id="t", force_refresh=False, word_align=False):
    f = tmp_path / "x.wav"
    f.write_bytes(b"\x00" * 10)
    return TranscriptionTask(
        task_id=task_id, file_name="x.wav", file_path=str(f), file_size=10,
        file_hash="h", engine="funasr", force_refresh=force_refresh,
        options=TranscribeOptions(diarize=True, word_align=word_align),
    )


def _cached():
    return TranscriptionResult(
        task_id="t", file_name="x.wav", file_hash="h", duration=5.0,
        segments=[TranscriptionSegment(start_time=0.0, end_time=5.0, text="你好", speaker="Speaker1")],
        speakers=["Speaker1"], processing_time=1.0,
    )


@pytest.mark.asyncio
async def test_process_cache_hit_skips_transcribe(tmp_path):
    """错开到达: 开工前二次查缓存命中 → 不真转录, 直接 COMPLETED + progress=100"""
    mgr = TaskManager()
    task = _task(tmp_path)
    mgr.tasks["t"] = task
    fake_tr = MagicMock()
    fake_tr.transcribe = AsyncMock()
    with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake_tr), \
         patch("src.core.task_manager.db_manager") as mdb:
        mdb.get_cached_result = AsyncMock(return_value=_cached())
        with patch.object(mgr, "_notify_task_complete", new=AsyncMock()), \
             patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
             patch.object(mgr, "_maybe_delete_task_file", new=AsyncMock()):
            await mgr._process_task("t")
    fake_tr.transcribe.assert_not_called()
    assert task.status.value == "completed"
    assert task.progress == 100              # codex #2
    assert mgr.processing_tasks == 0         # 计数平衡(finally 减回)


@pytest.mark.asyncio
async def test_process_cache_lookup_error_falls_through_to_transcribe(tmp_path):
    """codex #1: 二次查缓存抛错 → 降级继续转录, 不卡 PENDING / 不泄漏"""
    mgr = TaskManager()
    task = _task(tmp_path)
    mgr.tasks["t"] = task
    fake_tr = MagicMock()
    fake_tr.transcribe = AsyncMock(return_value=(_cached(), {"sentence_info": []}))
    with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake_tr), \
         patch("src.core.task_manager.db_manager") as mdb:
        mdb.get_cached_result = AsyncMock(side_effect=RuntimeError("db boom"))
        mdb.save_result = AsyncMock()
        with patch.object(mgr, "_notify_task_complete", new=AsyncMock()), \
             patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
             patch.object(mgr, "_maybe_delete_task_file", new=AsyncMock()), \
             patch.object(mgr, "_send_wework_notification", new=AsyncMock()):
            await mgr._process_task("t")
    fake_tr.transcribe.assert_called_once()  # 降级真转录
    assert task.status.value == "completed"  # 不卡 PENDING


@pytest.mark.asyncio
async def test_force_refresh_skips_second_cache_check(tmp_path):
    """force_refresh=true: 开工前不二次查缓存, 强制重算"""
    mgr = TaskManager()
    task = _task(tmp_path, force_refresh=True)
    mgr.tasks["t"] = task
    fake_tr = MagicMock()
    fake_tr.transcribe = AsyncMock(return_value=(_cached(), {"sentence_info": []}))
    with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake_tr), \
         patch("src.core.task_manager.db_manager") as mdb:
        mdb.get_cached_result = AsyncMock(return_value=_cached())  # 即使有缓存
        mdb.save_result = AsyncMock()
        with patch.object(mgr, "_notify_task_complete", new=AsyncMock()), \
             patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
             patch.object(mgr, "_maybe_delete_task_file", new=AsyncMock()), \
             patch.object(mgr, "_send_wework_notification", new=AsyncMock()):
            await mgr._process_task("t")
    mdb.get_cached_result.assert_not_called()  # force_refresh 跳过二次查
    fake_tr.transcribe.assert_called_once()


@pytest.mark.asyncio
async def test_word_align_demotion_miss_still_transcribes(tmp_path):
    """codex #9: +wa 请求二次查 miss(前序对齐失败降级写裸 tag) → 仍转录, 不误命中"""
    mgr = TaskManager()
    task = _task(tmp_path, word_align=True)
    mgr.tasks["t"] = task
    fake_tr = MagicMock()
    fake_tr.transcribe = AsyncMock(return_value=(_cached(), {"sentence_info": []}))
    with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake_tr), \
         patch("src.core.task_manager.db_manager") as mdb:
        mdb.get_cached_result = AsyncMock(return_value=None)  # +wa 键 miss
        mdb.save_result = AsyncMock()
        with patch.object(mgr, "_notify_task_complete", new=AsyncMock()), \
             patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
             patch.object(mgr, "_maybe_delete_task_file", new=AsyncMock()), \
             patch.object(mgr, "_send_wework_notification", new=AsyncMock()):
            await mgr._process_task("t")
    fake_tr.transcribe.assert_called_once()


@pytest.mark.asyncio
async def test_try_complete_from_cache_clears_stale_error(tmp_path):
    """codex #3: 失败重试残留 error, 第二次缓存命中成功须清 error"""
    mgr = TaskManager()
    task = _task(tmp_path)
    task.error = "上次转录失败"  # 模拟重试残留
    mgr.tasks["t"] = task
    with patch("src.core.task_manager.db_manager") as mdb:
        mdb.get_cached_result = AsyncMock(return_value=_cached())
        with patch.object(mgr, "_notify_task_complete", new=AsyncMock()):
            hit = await mgr._try_complete_from_cache(task)
    assert hit is True
    assert task.error is None                # codex #3
    assert task.progress == 100              # codex #2
    assert task.status.value == "completed"
