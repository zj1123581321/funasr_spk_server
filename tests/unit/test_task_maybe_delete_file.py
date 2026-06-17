"""
P4 F1 — `_maybe_delete_task_file` DRY helper

eng-review F1 定案 A:cancel/complete/failed 三处"无同 hash live 任务则删文件"逻辑
此前各写一份(重复 3 次), 抽统一 helper。本测试钉 helper 行为(去重后 3 处调它):
- delete_after_transcription on + 有 file_path + 无 live 同 hash 任务 → 删, 返回 True
- 有其他 PENDING/PROCESSING 同 hash 任务 → 不删(别人还要用), 返回 False
- delete_after_transcription off → 不删
- file_path 空 → 不删
- 同 hash 但终态(completed)任务不算 live → 照删
"""
from unittest.mock import patch, AsyncMock

import pytest

from src.core.config import config
from src.core.task_manager import TaskManager
from src.models.schemas import TranscriptionTask, TaskStatus


def _task(task_id, file_hash="h", file_path="/tmp/a.wav", status=TaskStatus.COMPLETED):
    t = TranscriptionTask(task_id=task_id, file_name="a.wav", file_path=file_path,
                          file_size=10, file_hash=file_hash, engine="funasr")
    t.status = status
    return t


@pytest.fixture
def tm(monkeypatch):
    monkeypatch.setattr(config.transcription, "delete_after_transcription", True)
    return TaskManager()


class TestMaybeDeleteTaskFile:
    @pytest.mark.asyncio
    async def test_deletes_when_no_live_same_hash(self, tm):
        t = _task("t1")
        tm.tasks["t1"] = t
        with patch("src.utils.file_utils.delete_file", new=AsyncMock()) as df:
            deleted = await tm._maybe_delete_task_file(t)
            assert deleted is True
            df.assert_awaited_once_with("/tmp/a.wav")

    @pytest.mark.asyncio
    async def test_keeps_when_live_same_hash_exists(self, tm):
        done = _task("t1", file_hash="h", status=TaskStatus.COMPLETED)
        live = _task("t2", file_hash="h", status=TaskStatus.PROCESSING)
        tm.tasks["t1"] = done
        tm.tasks["t2"] = live
        with patch("src.utils.file_utils.delete_file", new=AsyncMock()) as df:
            deleted = await tm._maybe_delete_task_file(done)
            assert deleted is False
            df.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_terminal_same_hash_does_not_block_delete(self, tm):
        """另一个同 hash 但终态(completed)任务不算 live, 不阻止删除。"""
        done = _task("t1", file_hash="h", status=TaskStatus.COMPLETED)
        other_done = _task("t2", file_hash="h", status=TaskStatus.FAILED)
        tm.tasks["t1"] = done
        tm.tasks["t2"] = other_done
        with patch("src.utils.file_utils.delete_file", new=AsyncMock()) as df:
            deleted = await tm._maybe_delete_task_file(done)
            assert deleted is True
            df.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_when_delete_after_off(self, tm, monkeypatch):
        monkeypatch.setattr(config.transcription, "delete_after_transcription", False)
        t = _task("t1")
        tm.tasks["t1"] = t
        with patch("src.utils.file_utils.delete_file", new=AsyncMock()) as df:
            deleted = await tm._maybe_delete_task_file(t)
            assert deleted is False
            df.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_no_file_path(self, tm):
        t = _task("t1", file_path="")
        tm.tasks["t1"] = t
        with patch("src.utils.file_utils.delete_file", new=AsyncMock()) as df:
            deleted = await tm._maybe_delete_task_file(t)
            assert deleted is False
            df.assert_not_awaited()
