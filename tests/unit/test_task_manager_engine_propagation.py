"""
PR1 — task_manager engine 字段端到端流转测试

验证：
1. create_task 把 request.engine 透传到 task.engine
2. request.engine 为 None 时 task.engine 走 config.default_engine
3. submit_task 做 cache lookup 时把 engine 传给 db_manager
4. _process_task 通过 resolve_transcriber(task.engine) 取 transcriber

通过 mock db_manager / resolve_transcriber 隔离外部依赖。
"""
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from src.core.task_manager import TaskManager
from src.models.schemas import FileUploadRequest


def make_request(engine=None, file_hash="abc123") -> FileUploadRequest:
    return FileUploadRequest(
        file_name="test.wav",
        file_size=1000,
        file_hash=file_hash,
        engine=engine,
    )


class TestCreateTaskEnginePropagation:
    @pytest.mark.asyncio
    async def test_engine_from_request_propagates_to_task(self):
        mgr = TaskManager()
        req = make_request(engine="qwen3")
        task = await mgr.create_task(req, task_id="t1")
        assert task.engine == "qwen3"

    @pytest.mark.asyncio
    async def test_none_engine_falls_back_to_config_default(self, monkeypatch):
        from src.core.config import config
        original = config.transcription.default_engine
        try:
            config.transcription.default_engine = "funasr"
            mgr = TaskManager()
            req = make_request(engine=None)
            task = await mgr.create_task(req, task_id="t2")
            assert task.engine == "funasr"
        finally:
            config.transcription.default_engine = original

    @pytest.mark.asyncio
    async def test_empty_string_engine_falls_back_to_config_default(self):
        from src.core.config import config
        original = config.transcription.default_engine
        try:
            config.transcription.default_engine = "funasr"
            mgr = TaskManager()
            req = make_request(engine="")
            task = await mgr.create_task(req, task_id="t3")
            assert task.engine == "funasr"
        finally:
            config.transcription.default_engine = original

    @pytest.mark.asyncio
    async def test_default_engine_change_takes_effect_for_new_task(self):
        from src.core.config import config
        original = config.transcription.default_engine
        try:
            config.transcription.default_engine = "qwen3"
            mgr = TaskManager()
            req = make_request(engine=None)
            task = await mgr.create_task(req, task_id="t4")
            assert task.engine == "qwen3"
        finally:
            config.transcription.default_engine = original


class TestSubmitTaskCacheLookupEngineAware:
    @pytest.mark.asyncio
    async def test_cache_lookup_includes_engine(self, tmp_path):
        """submit_task 触发 cache lookup 时必须传 engine"""
        mgr = TaskManager()
        req = make_request(engine="qwen3", file_hash="h1")
        await mgr.create_task(req, task_id="ts1")

        fake_file = tmp_path / "x.wav"
        fake_file.write_bytes(b"\x00" * 100)

        with patch("src.core.task_manager.db_manager") as mock_db:
            mock_db.get_cached_result = AsyncMock(return_value=None)
            await mgr.submit_task("ts1", str(fake_file))
            # 验证传 engine
            assert mock_db.get_cached_result.called
            call = mock_db.get_cached_result.call_args
            # 支持位置或关键字传参
            kwargs = call.kwargs
            args = call.args
            engine_arg = kwargs.get("engine") if "engine" in kwargs else (args[2] if len(args) >= 3 else None)
            assert engine_arg == "qwen3", f"engine 应传入 get_cached_result，调用: {call}"


class TestProcessTaskUsesDispatch:
    """验证 _process_task 通过 resolve_transcriber 取 transcriber，而不是写死 get_transcriber"""

    @pytest.mark.asyncio
    async def test_process_task_calls_resolve_transcriber_with_task_engine(self, tmp_path):
        from src.core.task_manager import TaskManager
        from src.models.schemas import TranscriptionTask, TaskStatus

        mgr = TaskManager()
        task = TranscriptionTask(
            task_id="tp1",
            file_name="x.wav",
            file_path=str(tmp_path / "x.wav"),
            file_size=100,
            file_hash="h",
            engine="qwen3",
        )
        mgr.tasks["tp1"] = task

        # 准备假音频文件
        (tmp_path / "x.wav").write_bytes(b"\x00" * 100)

        # mock transcriber：返回 JSON 模式 (TranscriptionResult, raw_result) 元组
        from src.models.schemas import TranscriptionResult, TranscriptionSegment
        fake_result = TranscriptionResult(
            task_id="tp1",
            file_name="x.wav",
            file_hash="h",
            duration=1.0,
            segments=[TranscriptionSegment(start_time=0, end_time=1, text="hi", speaker="Speaker1")],
            speakers=["Speaker1"],
            processing_time=0.1,
        )
        fake_raw = {"sentence_info": []}
        fake_transcriber = MagicMock()
        fake_transcriber.transcribe = AsyncMock(return_value=(fake_result, fake_raw))

        with patch("src.core.transcriber_dispatch.resolve_transcriber") as mock_resolve, \
             patch("src.core.task_manager.db_manager") as mock_db:
            mock_resolve.return_value = fake_transcriber
            mock_db.save_result = AsyncMock()
            # 屏蔽通知/文件删除 等副作用
            with patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
                 patch.object(mgr, "_notify_task_complete", new=AsyncMock()):
                await mgr._process_task(task.task_id)

            assert mock_resolve.called, "_process_task 应调用 resolve_transcriber"
            mock_resolve.assert_called_with("qwen3")


class TestSaveResultPassesEngine:
    @pytest.mark.asyncio
    async def test_save_result_propagates_engine_to_db(self, tmp_path):
        from src.core.task_manager import TaskManager
        from src.models.schemas import TranscriptionTask
        from src.models.schemas import TranscriptionResult, TranscriptionSegment

        mgr = TaskManager()
        task = TranscriptionTask(
            task_id="ts2",
            file_name="x.wav",
            file_path=str(tmp_path / "x.wav"),
            file_size=100,
            file_hash="h2",
            engine="qwen3",
        )
        mgr.tasks["ts2"] = task
        (tmp_path / "x.wav").write_bytes(b"\x00" * 100)

        fake_result = TranscriptionResult(
            task_id="ts2",
            file_name="x.wav",
            file_hash="h2",
            duration=1.0,
            segments=[TranscriptionSegment(start_time=0, end_time=1, text="hi", speaker="Speaker1")],
            speakers=["Speaker1"],
            processing_time=0.1,
        )
        fake_raw = {"sentence_info": []}
        fake_transcriber = MagicMock()
        fake_transcriber.transcribe = AsyncMock(return_value=(fake_result, fake_raw))

        with patch("src.core.transcriber_dispatch.resolve_transcriber") as mock_resolve, \
             patch("src.core.task_manager.db_manager") as mock_db:
            mock_resolve.return_value = fake_transcriber
            mock_db.save_result = AsyncMock()
            with patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
                 patch.object(mgr, "_notify_task_complete", new=AsyncMock()):
                await mgr._process_task(task.task_id)

            assert mock_db.save_result.called
            call = mock_db.save_result.call_args
            kwargs = call.kwargs
            engine_arg = kwargs.get("engine")
            assert engine_arg == "qwen3", f"save_result 应收到 engine 参数，调用: {call}"
