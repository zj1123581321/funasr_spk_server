"""T-B (diarize 开关前置 commit 0) — qwen3 + SRT 缓存命中返回空 bug 复现与修复

现行 bug 链:
1. task_manager._process_task SRT 模式存 TranscriptionResult(segments=[]) 进缓存
2. qwen3 raw_result 无 sentence_info (qwen3_transcriber raw 是 asr_text/turns 结构)
3. database.get_cached_result SRT exact-hit 同引擎走 _generate_srt_from_raw_result
   (funasr 私有 sentence_info 重建路径) → 返回 ""

修法: SRT 模式 engine 返回携真 segments、缓存存真 segments; database SRT 重建
按 raw 结构分流 (sentence_info → funasr raw 路径, 否则 → schema 中立 segments 路径).
同时把 get_cached_result 的 catch-all 换成具名异常 (坏行不静默吞).
"""
from __future__ import annotations

import json
from typing import List
from unittest.mock import patch, AsyncMock, MagicMock

import aiosqlite
import pytest

from src.core.database import DatabaseManager
from src.core.qwen3.asr import ASRResult
from src.models.schemas import TranscriptionResult, TranscriptionSegment


# ==================== 测试桩 ====================

def _fake_asr_result(text: str = "你好啊我是说话人一你好啊我是说话人二", duration: float = 10.0) -> ASRResult:
    return ASRResult(
        text=text,
        items=[],
        chunks=[],
        duration=duration,
        elapsed=1.0,
        rtf=0.1,
        peak_rss_mb=3500.0,
        rss_delta_mb=1500.0,
    )


def _fake_turns() -> List[dict]:
    return [
        {"start": 0.0, "end": 5.0, "speaker": 0},
        {"start": 5.0, "end": 10.0, "speaker": 1},
    ]


@pytest.fixture
def qwen3_transcriber():
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber
    return Qwen3DiarizeTranscriber(
        asr_model_dir="/fake/qwen3-asr",
        segmentation_model="/fake/seg.onnx",
        embedding_model="/fake/emb.onnx",
    )


# ==================== 1. qwen3 SRT 模式返回携真 segments ====================


class TestQwen3SrtCarriesSegments:
    @pytest.mark.asyncio
    async def test_srt_dict_contains_real_segments(self, qwen3_transcriber, tmp_path):
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result()), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=_fake_turns()), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            ret = await qwen3_transcriber.transcribe(
                audio_path=str(audio), task_id="t", output_format="srt",
            )
        assert ret["format"] == "srt"
        segments = ret.get("segments")
        assert segments, "SRT 模式返回 dict 必须携真 segments (T-B 修复 + T-A 投影地基)"
        assert all(isinstance(s, TranscriptionSegment) for s in segments)
        # segments 与 SRT content 同源: 文本拼接覆盖完整 ASR
        full = "".join(s.text for s in segments)
        assert full == "你好啊我是说话人一你好啊我是说话人二"


# ==================== 2. task_manager SRT 模式缓存存真 segments ====================


class TestTaskManagerSrtCachesRealSegments:
    @pytest.mark.asyncio
    async def test_srt_result_saved_with_real_segments(self, tmp_path):
        from src.core.task_manager import TaskManager
        from src.models.schemas import TranscriptionTask

        mgr = TaskManager()
        task = TranscriptionTask(
            task_id="srt1",
            file_name="x.wav",
            file_path=str(tmp_path / "x.wav"),
            file_size=100,
            file_hash="h-srt",
            engine="qwen3",
            output_format="srt",
        )
        mgr.tasks["srt1"] = task
        (tmp_path / "x.wav").write_bytes(b"\x00" * 100)

        real_segments = [
            TranscriptionSegment(start_time=0.0, end_time=5.0, text="你好", speaker="Speaker1"),
            TranscriptionSegment(start_time=5.0, end_time=10.0, text="再见", speaker="Speaker2"),
        ]
        srt_dict = {
            "format": "srt",
            "content": "1\n00:00:00,000 --> 00:00:05,000\nSpeaker1:你好\n",
            "file_name": "x.wav",
            "file_hash": "h-srt",
            "duration": 10.0,
            "processing_time": 0.5,
            "raw_result": {"asr_text": "你好再见", "engine": "qwen3"},
            "segments": real_segments,
        }
        fake_transcriber = MagicMock()
        fake_transcriber.transcribe = AsyncMock(return_value=srt_dict)

        with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake_transcriber), \
             patch("src.core.task_manager.db_manager") as mock_db:
            mock_db.save_result = AsyncMock()
            with patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
                 patch.object(mgr, "_notify_task_complete", new=AsyncMock()):
                await mgr._process_task("srt1")

        assert mock_db.save_result.called
        saved_result = mock_db.save_result.call_args.args[0]
        assert saved_result.segments == real_segments, "SRT 模式缓存必须存真 segments (不再是 [])"
        assert saved_result.speakers == ["Speaker1", "Speaker2"]


# ==================== 3. database SRT 缓存命中 (qwen3 raw 无 sentence_info) ====================


def make_result_with_segments(file_hash: str = "h-db") -> TranscriptionResult:
    return TranscriptionResult(
        task_id="t-db",
        file_name="x.wav",
        file_hash=file_hash,
        duration=10.0,
        segments=[
            TranscriptionSegment(start_time=0.0, end_time=5.0, text="你好", speaker="Speaker1"),
            TranscriptionSegment(start_time=5.0, end_time=10.0, text="再见", speaker="Speaker2"),
        ],
        speakers=["Speaker1", "Speaker2"],
        processing_time=0.5,
    )


class TestDatabaseSrtHitQwen3Raw:
    @pytest.mark.asyncio
    async def test_srt_hit_with_non_funasr_raw_rebuilds_from_segments(self, tmp_path):
        """复现 bug: qwen3 行 raw_result 无 sentence_info, SRT 命中不能返回空"""
        db = DatabaseManager(db_path=str(tmp_path / "t.db"))
        await db.init_db()
        result = make_result_with_segments()
        qwen3_raw = {"asr_text": "你好再见", "turns": [], "engine": "qwen3"}
        await db.save_result(result, raw_result=qwen3_raw, engine="qwen3")

        cached = await db.get_cached_result(result.file_hash, output_format="srt", engine="qwen3")
        assert cached is not None
        assert cached["format"] == "srt"
        assert cached["content"].strip(), "qwen3 SRT 缓存命中不能返回空 content (T-B bug)"
        assert "Speaker1:你好" in cached["content"]

    @pytest.mark.asyncio
    async def test_srt_hit_with_funasr_raw_keeps_raw_path(self, tmp_path):
        """funasr 行 (raw 有 sentence_info) 仍走原 raw 重建路径, 行为不变"""
        db = DatabaseManager(db_path=str(tmp_path / "t.db"))
        await db.init_db()
        result = make_result_with_segments(file_hash="h-fun")
        funasr_raw = [{
            "sentence_info": [
                {"start": 0, "end": 5000, "text": "原始句子", "spk": 0},
            ]
        }]
        await db.save_result(result, raw_result=funasr_raw, engine="funasr")

        cached = await db.get_cached_result("h-fun", output_format="srt", engine="funasr")
        assert cached is not None
        # raw 路径: 文本来自 sentence_info, 不是 result.segments
        assert "Speaker1:原始句子" in cached["content"]


# ==================== 4. 坏缓存行: 具名异常, 不静默吞 ====================


class TestBadCacheRowNamedExceptions:
    @pytest.mark.asyncio
    async def test_corrupt_result_json_treated_as_miss(self, tmp_path):
        """坏行 (result 列非法 JSON) → 当 miss 处理 + 日志, 不抛"""
        db_path = str(tmp_path / "t.db")
        db = DatabaseManager(db_path=db_path)
        await db.init_db()
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute(
                "INSERT INTO transcription_cache (file_hash, file_name, result, engine) "
                "VALUES (?, ?, ?, ?)",
                ("h-bad", "x.wav", "{not-json", "qwen3"),
            )
            await conn.commit()
        cached = await db.get_cached_result("h-bad", output_format="json", engine="qwen3")
        assert cached is None

    @pytest.mark.asyncio
    async def test_unexpected_error_propagates(self, tmp_path):
        """非具名异常 (编程错误) 必须冒泡, 禁止 catch-all 静默吞"""
        db = DatabaseManager(db_path=str(tmp_path / "t.db"))
        await db.init_db()
        result = make_result_with_segments(file_hash="h-prop")
        await db.save_result(result, raw_result=None, engine="qwen3")

        with patch("src.core.database.json.loads", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                await db.get_cached_result("h-prop", output_format="json", engine="qwen3")
