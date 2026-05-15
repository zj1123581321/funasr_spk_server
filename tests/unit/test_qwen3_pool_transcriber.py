"""
Qwen3PoolTranscriber — 主进程 wrapper, 通过 FileBasedProcessPool 调度 worker subprocess

设计:
- wrapper 接口与 Qwen3DiarizeTranscriber.transcribe 鸭子兼容
  * JSON: (TranscriptionResult, raw_result)
  * SRT:  {format, content, file_name, file_hash, duration, processing_time, raw_result}
- 持有 FileBasedProcessPool, worker_entry_script="src/core/qwen3_worker_process.py"
- transcribe(audio_path, task_id, progress_callback, output_format) →
    pool.generate_with_pool(extra_task_fields={"output_format": output_format})

测试覆盖:
1. constructor 默认创建 pool, pool_size 从 config 读, entry 正确
2. constructor 可注入自定义 pool (依赖注入便于测试)
3. JSON 模式 transcribe → 返回 (TranscriptionResult, raw)
4. SRT 模式 transcribe → 返回 SRT dict
5. progress_callback (sync + async) 在 0% / 100% 被调
6. pool 抛错 → wrapper 透传
7. transcribe 把 output_format 通过 extra_task_fields 传给 pool
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.schemas import TranscriptionResult, TranscriptionSegment


# ==================== 辅助 ====================


def _fake_json_result(task_id: str = "t", text: str = "你好世界"):
    """构造 JSON 模式的 worker 返回 (TranscriptionResult, raw)"""
    tres = TranscriptionResult(
        task_id=task_id,
        file_name="a.wav",
        file_hash="h",
        duration=1.0,
        segments=[
            TranscriptionSegment(start_time=0.0, end_time=1.0, text=text, speaker="Speaker1")
        ],
        speakers=["Speaker1"],
        processing_time=0.1,
    )
    return (tres, {"asr_text": text, "engine": "qwen3"})


def _fake_srt_result(text: str = "字幕内容"):
    return {
        "format": "srt",
        "content": f"1\n00:00:00,000 --> 00:00:01,000\nSpeaker1:{text}\n",
        "file_name": "a.wav",
        "file_hash": "h",
        "duration": 1.0,
        "processing_time": 0.1,
        "raw_result": {"asr_text": text, "engine": "qwen3"},
    }


# ==================== Constructor ====================


class TestConstructor:
    """wrapper 持有 pool, entry 正确"""

    def test_default_pool_has_qwen3_entry(self):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        wrapper = Qwen3PoolTranscriber(pool_size=3)
        assert wrapper._pool.worker_entry_script == "src/core/qwen3_worker_process.py"
        assert wrapper._pool.pool_size == 3

    def test_inject_custom_pool(self):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock(name="custom_pool")
        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool)
        assert wrapper._pool is custom_pool


# ==================== transcribe 模式透传 ====================


class TestTranscribeJsonMode:
    """JSON 模式直接返回 (TranscriptionResult, raw)"""

    @pytest.mark.asyncio
    async def test_returns_tuple_for_json_mode(self):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()
        expected = _fake_json_result(task_id="t-1")
        custom_pool.generate_with_pool = AsyncMock(return_value=expected)

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool)

        result = await wrapper.transcribe(
            audio_path="/fake/a.wav", task_id="t-1", output_format="json"
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        tres, raw = result
        assert tres.task_id == "t-1"
        assert raw["engine"] == "qwen3"

    @pytest.mark.asyncio
    async def test_extra_task_fields_includes_output_format_json(self):
        """wrapper 必须把 output_format 透传给 pool 的 extra_task_fields"""
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()
        custom_pool.generate_with_pool = AsyncMock(return_value=_fake_json_result())

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool)

        await wrapper.transcribe(audio_path="/fake/a.wav", task_id="t", output_format="json")

        # 检查 pool 收到了 output_format=json
        call_kwargs = custom_pool.generate_with_pool.call_args.kwargs
        assert call_kwargs.get("extra_task_fields") == {"output_format": "json"}


class TestTranscribeSrtMode:
    """SRT 模式返回 dict"""

    @pytest.mark.asyncio
    async def test_returns_dict_for_srt_mode(self):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()
        custom_pool.generate_with_pool = AsyncMock(return_value=_fake_srt_result(text="对白"))

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool)
        result = await wrapper.transcribe(
            audio_path="/fake/a.wav", task_id="t-srt", output_format="srt"
        )

        assert isinstance(result, dict)
        assert result["format"] == "srt"
        assert "对白" in result["content"]

    @pytest.mark.asyncio
    async def test_extra_task_fields_includes_output_format_srt(self):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()
        custom_pool.generate_with_pool = AsyncMock(return_value=_fake_srt_result())

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool)
        await wrapper.transcribe(audio_path="/fake/a.wav", task_id="t", output_format="srt")

        call_kwargs = custom_pool.generate_with_pool.call_args.kwargs
        assert call_kwargs.get("extra_task_fields") == {"output_format": "srt"}


# ==================== progress_callback ====================


class TestProgressCallback:
    """跨进程进度难传, wrapper 模拟 0% / 100% 两次回调"""

    @pytest.mark.asyncio
    async def test_async_callback_called(self):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()
        custom_pool.generate_with_pool = AsyncMock(return_value=_fake_json_result())

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool)

        calls = []
        async def progress(pct):
            calls.append(pct)

        await wrapper.transcribe(
            audio_path="/fake/a.wav", task_id="t", progress_callback=progress
        )

        assert 0 in calls
        assert 100 in calls

    @pytest.mark.asyncio
    async def test_sync_callback_called(self):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()
        custom_pool.generate_with_pool = AsyncMock(return_value=_fake_json_result())

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool)

        calls = []
        def progress(pct):
            calls.append(pct)

        await wrapper.transcribe(
            audio_path="/fake/a.wav", task_id="t", progress_callback=progress
        )

        assert 0 in calls
        assert 100 in calls

    @pytest.mark.asyncio
    async def test_none_callback_does_not_crash(self):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()
        custom_pool.generate_with_pool = AsyncMock(return_value=_fake_json_result())

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool)
        # 不传 progress_callback (None) 不应崩
        await wrapper.transcribe(audio_path="/fake/a.wav", task_id="t")


# ==================== error propagation ====================


class TestErrorPropagation:
    """pool 抛错 → wrapper 透传"""

    @pytest.mark.asyncio
    async def test_pool_error_propagates(self):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()
        custom_pool.generate_with_pool = AsyncMock(side_effect=RuntimeError("pool 炸了"))

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool)

        with pytest.raises(RuntimeError, match="pool 炸了"):
            await wrapper.transcribe(audio_path="/fake/a.wav", task_id="t")


# ==================== pool extra_task_fields 参数支持 ====================


class TestPoolExtraTaskFields:
    """FileBasedProcessPool.generate_with_pool 必须接受 extra_task_fields 关键字参数"""

    @pytest.mark.asyncio
    async def test_extra_task_fields_merged_into_task_json(self, tmp_path, monkeypatch):
        """extra_task_fields 的字段应该合并到任务 JSON 文件中"""
        import json
        from src.core.file_based_process_pool import FileBasedProcessPool

        monkeypatch.chdir(tmp_path)

        pool = FileBasedProcessPool(pool_size=1)
        pool.is_initialized = True  # 跳过初始化

        # mock _ensure_workers_alive / _spawn_worker / _calculate_timeout
        pool._ensure_workers_alive = AsyncMock()
        pool._spawn_worker = AsyncMock()
        pool._calculate_timeout = MagicMock(return_value=600)

        # mock worker 不存在(_launch 不会真跑)
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        pool.worker_processes = [fake_proc]

        # 准备 fake audio
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")

        # 在 pool 写完 task_file 之后立刻塞一个 fake 结果文件触发返回
        import asyncio as _asyncio
        import pickle as _pickle

        async def watcher():
            # 等待 task 文件出现, 然后回写 pickle 结果
            for _ in range(50):
                task_files = list(pool.task_dir.glob("worker_0_*.task"))
                if task_files:
                    task_file = task_files[0]
                    task_data = json.loads(task_file.read_text())
                    # 暴露 task_data 给主测试线
                    captured["task_data"] = task_data

                    # 写 pickle 结果
                    result_file = task_file.with_suffix(".pkl")
                    with open(result_file, "wb") as f:
                        _pickle.dump(
                            {"task_id": task_data["task_id"], "success": True, "result": "OK"},
                            f,
                        )
                    return
                await _asyncio.sleep(0.05)

        captured = {}
        watcher_task = _asyncio.create_task(watcher())

        try:
            result = await pool.generate_with_pool(
                audio_path=str(audio),
                extra_task_fields={"output_format": "json", "custom_field": 42},
            )
        finally:
            await watcher_task

        assert result == "OK"
        # 检查 task_data 包含 extra 字段
        td = captured["task_data"]
        assert td["output_format"] == "json"
        assert td["custom_field"] == 42
        # 同时也保留 audio_path / task_id (基础字段)
        assert "task_id" in td
        assert "audio_path" in td
