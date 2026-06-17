"""
P4 A2 — 孤儿上传文件 sweeper(文件系统 sweeper, 非 evict 兜底)

eng-review + codex 二审定案:孤儿文件清理走 upload-dir 文件系统 sweeper, 不锚内存 task
生命周期(扛进程重启 + 删失败下轮重试, 避 worker race)。

`_sweep_orphan_upload_files`:扫 upload_dir, 删 mtime 超宽限期 且 无 live 引用 的文件。
live 引用 = 任一 PENDING/PROCESSING 任务 file_path, 或任一 upload session 的
finalized_file_path。delete_after_transcription off 时整体不跑(尊重"用户要留文件")。
"""
import os
import time
from pathlib import Path

import pytest

from src.core.config import config
from src.core.task_manager import TaskManager
from src.models.schemas import TranscriptionTask, TaskStatus


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    d = tmp_path / "uploads"
    d.mkdir()
    monkeypatch.setattr(config.server, "upload_dir", str(d))
    monkeypatch.setattr(config.transcription, "delete_after_transcription", True)
    monkeypatch.setattr(config.transcription, "orphan_file_grace_seconds", 7200)
    return d


@pytest.fixture
def tm():
    return TaskManager()


@pytest.fixture(autouse=True)
def _clear_ws_sessions(monkeypatch):
    """默认 ws_handler.upload_sessions 为空, 避免脏状态。"""
    from src.api.websocket_handler import ws_handler
    monkeypatch.setattr(ws_handler, "upload_sessions", {})


def _old_file(d: Path, name: str, age_sec: float) -> Path:
    p = d / name
    p.write_bytes(b"x")
    old = time.time() - age_sec
    os.utime(p, (old, old))
    return p


def _task(task_id, file_path, file_hash="h", status=TaskStatus.PROCESSING):
    t = TranscriptionTask(task_id=task_id, file_name="a.wav", file_path=str(file_path),
                          file_size=10, file_hash=file_hash, engine="funasr")
    t.status = status
    return t


class TestOrphanSweep:
    @pytest.mark.asyncio
    async def test_deletes_old_unreferenced_orphan(self, tm, upload_dir):
        p = _old_file(upload_dir, "orphan.wav", age_sec=10000)  # 超 2h 宽限
        n = await tm._sweep_orphan_upload_files()
        assert n == 1
        assert not p.exists()

    @pytest.mark.asyncio
    async def test_keeps_file_referenced_by_live_task(self, tm, upload_dir):
        p = _old_file(upload_dir, "inuse.wav", age_sec=10000)
        tm.tasks["t1"] = _task("t1", p, status=TaskStatus.PROCESSING)
        n = await tm._sweep_orphan_upload_files()
        assert n == 0
        assert p.exists(), "被 PROCESSING 任务引用的文件不该删"

    @pytest.mark.asyncio
    async def test_terminal_task_does_not_protect_file(self, tm, upload_dir):
        """终态任务(completed)不算 live 引用, 其旧文件应被回收。"""
        p = _old_file(upload_dir, "done.wav", age_sec=10000)
        tm.tasks["t1"] = _task("t1", p, status=TaskStatus.COMPLETED)
        n = await tm._sweep_orphan_upload_files()
        assert n == 1
        assert not p.exists()

    @pytest.mark.asyncio
    async def test_keeps_file_referenced_by_upload_session(self, tm, upload_dir, monkeypatch):
        p = _old_file(upload_dir, "session.wav", age_sec=10000)
        from src.api.websocket_handler import ws_handler
        monkeypatch.setattr(ws_handler, "upload_sessions",
                            {"s1": {"finalized_file_path": str(p)}})
        n = await tm._sweep_orphan_upload_files()
        assert n == 0
        assert p.exists(), "被 upload session finalized_file_path 引用的文件不该删"

    @pytest.mark.asyncio
    async def test_keeps_fresh_file_within_grace(self, tm, upload_dir):
        """宽限期内的新文件(可能在途上传)不删。"""
        p = _old_file(upload_dir, "fresh.wav", age_sec=60)  # 远小于 7200
        n = await tm._sweep_orphan_upload_files()
        assert n == 0
        assert p.exists()

    @pytest.mark.asyncio
    async def test_disabled_when_delete_after_off(self, tm, upload_dir, monkeypatch):
        monkeypatch.setattr(config.transcription, "delete_after_transcription", False)
        p = _old_file(upload_dir, "keep.wav", age_sec=10000)
        n = await tm._sweep_orphan_upload_files()
        assert n == 0
        assert p.exists(), "delete_after_transcription off 时 sweeper 整体不跑"

    @pytest.mark.asyncio
    async def test_missing_upload_dir_no_error(self, tm, monkeypatch):
        monkeypatch.setattr(config.server, "upload_dir", "/nonexistent/dir/xyz")
        monkeypatch.setattr(config.transcription, "delete_after_transcription", True)
        monkeypatch.setattr(config.transcription, "orphan_file_grace_seconds", 7200)
        n = await tm._sweep_orphan_upload_files()
        assert n == 0


class TestConfigField:
    def test_orphan_grace_default(self):
        from src.core.config import TranscriptionConfig
        assert TranscriptionConfig().orphan_file_grace_seconds == 7200
