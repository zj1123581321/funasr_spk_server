"""Qwen3DiarizeTranscriber 单元测试 — JSON 模式

mock src.core.qwen3.asr.run_asr / diarize.run_diarization, 验证:
- transcribe() JSON 模式返回 (TranscriptionResult, raw_result) 元组(与 FunASR 同形)
- TranscriptionResult.segments 来自 merge_asr_and_diarize, speaker 重命名为 "Speaker1/2/..."
- TranscriptionResult.speakers 是 sorted unique 列表
- raw_result 是 dict, 含 asr text / turns / 关键元数据(用于缓存)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
from unittest.mock import patch, AsyncMock

import pytest

from src.core.qwen3.asr import ASRResult, WordItem


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


# ==================== fixtures ====================

@pytest.fixture
def transcriber():
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber
    return Qwen3DiarizeTranscriber(
        asr_model_dir="/fake/qwen3-asr",
        segmentation_model="/fake/seg.onnx",
        embedding_model="/fake/emb.onnx",
    )


# ==================== 测试用例 ====================

class TestQwen3DiarizeTranscriberJsonMode:
    """JSON 模式返回 (TranscriptionResult, raw_result) 元组, 与 FunASR 接口对齐"""

    @pytest.mark.asyncio
    async def test_returns_tuple_with_result_and_raw(self, transcriber, tmp_path):
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result()), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=_fake_turns()), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="hash-stub")):
            ret = await transcriber.transcribe(
                audio_path=str(audio),
                task_id="t-1",
                progress_callback=None,
                output_format="json",
            )
        assert isinstance(ret, tuple), "JSON 模式应返回元组"
        assert len(ret) == 2, "元组应是 (TranscriptionResult, raw_result)"

    @pytest.mark.asyncio
    async def test_transcription_result_fields(self, transcriber, tmp_path):
        from src.models.schemas import TranscriptionResult
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result(duration=10.0)), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=_fake_turns()), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="hash-stub")):
            result, _raw = await transcriber.transcribe(
                audio_path=str(audio),
                task_id="t-42",
                output_format="json",
            )
        assert isinstance(result, TranscriptionResult)
        assert result.task_id == "t-42"
        assert result.file_name == "x.wav"
        assert result.file_hash == "hash-stub"
        assert result.duration == 10.0
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_segments_align_with_turns(self, transcriber, tmp_path):
        """每个 turn 对应一个 segment, speaker label 形如 Speaker{i+1}"""
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result()), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=_fake_turns()), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            result, _ = await transcriber.transcribe(
                audio_path=str(audio), task_id="t", output_format="json",
            )
        assert len(result.segments) == 2
        # turn 0 -> Speaker1, turn 1 -> Speaker2 (与 FunASR 命名一致)
        assert result.segments[0].speaker == "Speaker1"
        assert result.segments[1].speaker == "Speaker2"
        # 时间窗
        assert result.segments[0].start_time == 0.0
        assert result.segments[0].end_time == 5.0
        # 文本拼接覆盖完整 ASR
        full = "".join(s.text for s in result.segments)
        assert full == "你好啊我是说话人一你好啊我是说话人二"

    @pytest.mark.asyncio
    async def test_speakers_field_sorted_unique(self, transcriber, tmp_path):
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        turns = [
            {"start": 0.0, "end": 3.0, "speaker": 1},
            {"start": 3.0, "end": 6.0, "speaker": 0},
            {"start": 6.0, "end": 9.0, "speaker": 1},  # 重复
        ]
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result(duration=9.0)), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=turns), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            result, _ = await transcriber.transcribe(
                audio_path=str(audio), task_id="t", output_format="json",
            )
        assert result.speakers == ["Speaker1", "Speaker2"], f"got: {result.speakers}"

    @pytest.mark.asyncio
    async def test_raw_result_carries_asr_text_and_turns(self, transcriber, tmp_path):
        """raw_result 缓存用, 至少要带 asr text + turns 用于格式重转换"""
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        asr_text = "测试文本"
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result(text=asr_text)), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=_fake_turns()), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            _, raw = await transcriber.transcribe(
                audio_path=str(audio), task_id="t", output_format="json",
            )
        assert isinstance(raw, dict)
        assert raw.get("asr_text") == asr_text
        assert raw.get("turns") == _fake_turns()

    @pytest.mark.asyncio
    async def test_filter_spurious_applied_when_audio_duration_known(self, transcriber, tmp_path):
        """duration>0 时, 应启用 filter_spurious_speakers (1% share)"""
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        # 300s 音频, 1% = 3s, < 3s 的 spurious 应被合并
        spurious_turns = [
            {"start": 0.0, "end": 150.0, "speaker": 0},
            {"start": 150.0, "end": 151.5, "speaker": 2},  # 1.5s spurious
            {"start": 151.5, "end": 300.0, "speaker": 1},
        ]
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result(duration=300.0)), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=spurious_turns), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            result, _ = await transcriber.transcribe(
                audio_path=str(audio), task_id="t", output_format="json",
            )
        # spurious speaker 2 应该被合并, 最终只 2 个 speakers
        assert len(result.speakers) == 2, f"spurious 未过滤: {result.speakers}"


class TestQwen3DiarizeTranscriberSrtMode:
    """SRT 模式返回 dict, 与 FunASR SRT 分支保持同形"""

    @pytest.mark.asyncio
    async def test_srt_mode_returns_dict_with_format_key(self, transcriber, tmp_path):
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result()), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=_fake_turns()), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            ret = await transcriber.transcribe(
                audio_path=str(audio),
                task_id="t-srt",
                output_format="srt",
            )
        assert isinstance(ret, dict), "SRT 模式应返回 dict, 不是元组"
        assert ret["format"] == "srt"

    @pytest.mark.asyncio
    async def test_srt_dict_required_fields(self, transcriber, tmp_path):
        audio = tmp_path / "podcast.mp3"
        audio.write_bytes(b"\x00")
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result(duration=10.0)), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=_fake_turns()), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="hash-x")):
            ret = await transcriber.transcribe(
                audio_path=str(audio), task_id="t", output_format="srt",
            )
        # FunASR SRT 分支返回 {format, content, file_name, file_hash, duration, processing_time, raw_result}
        for key in ("format", "content", "file_name", "file_hash", "duration", "processing_time", "raw_result"):
            assert key in ret, f"SRT dict 缺字段: {key}"
        assert ret["file_name"] == "podcast.mp3"
        assert ret["file_hash"] == "hash-x"
        assert ret["duration"] == 10.0

    @pytest.mark.asyncio
    async def test_srt_content_format(self, transcriber, tmp_path):
        """SRT content 是合规字符串 — 序号 / 时间戳 / Speaker{n}:文本"""
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result(text="abcdefghij")), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=_fake_turns()), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            ret = await transcriber.transcribe(
                audio_path=str(audio), task_id="t", output_format="srt",
            )
        content = ret["content"]
        # 序号 + 时间戳格式
        assert "1\n00:00:00,000 --> 00:00:05,000\n" in content, content
        assert "2\n00:00:05,000 --> 00:00:10,000\n" in content, content
        # Speaker 命名 + 与 FunASR 字节级对齐(冒号后无空格, 见 funasr_transcriber.py:516)
        assert "Speaker1:" in content
        assert "Speaker2:" in content
        assert "Speaker1: " not in content, "Qwen3 SRT 应与 FunASR 字节级一致, 冒号后不应有空格"
        assert "Speaker2: " not in content, "Qwen3 SRT 应与 FunASR 字节级一致, 冒号后不应有空格"

    @pytest.mark.asyncio
    async def test_srt_raw_result_contains_asr_data(self, transcriber, tmp_path):
        """SRT 模式的 raw_result 也要含 asr_text + turns(缓存命中后能重转 JSON)"""
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result(text="测试")), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=_fake_turns()), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            ret = await transcriber.transcribe(
                audio_path=str(audio), task_id="t", output_format="srt",
            )
        raw = ret["raw_result"]
        assert raw["asr_text"] == "测试"
        assert raw["turns"] == _fake_turns()
        assert raw["engine"] == "qwen3"


class TestProgressCallback:
    """progress_callback 兼容 sync + async, 异常被吞不影响 transcribe 完成"""

    @pytest.mark.asyncio
    async def test_sync_callback_called_with_100_at_end(self, transcriber, tmp_path):
        from unittest.mock import MagicMock
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        cb = MagicMock()
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result()), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=_fake_turns()), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            await transcriber.transcribe(
                audio_path=str(audio), task_id="t", progress_callback=cb, output_format="json",
            )
        called_pcts = [args[0] for (args, _) in cb.call_args_list]
        assert 100 in called_pcts, f"应至少调 1 次 100%: {called_pcts}"
        # 单调非递减(进度不能倒退)
        assert called_pcts == sorted(called_pcts), f"进度倒退了: {called_pcts}"

    @pytest.mark.asyncio
    async def test_async_callback_awaited(self, transcriber, tmp_path):
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        cb = AsyncMock()
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result()), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=_fake_turns()), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            await transcriber.transcribe(
                audio_path=str(audio), task_id="t", progress_callback=cb, output_format="json",
            )
        assert cb.await_count >= 1, "async callback 至少 await 一次"
        awaited_pcts = [args[0] for (args, _) in cb.call_args_list]
        assert 100 in awaited_pcts

    @pytest.mark.asyncio
    async def test_callback_exception_is_swallowed(self, transcriber, tmp_path):
        """callback 内部抛错不能影响 transcribe 完成(用户端代码 bug 不应让转录失败)"""
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        def cb(_pct):
            raise RuntimeError("bad callback")
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result()), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=_fake_turns()), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            result, _ = await transcriber.transcribe(
                audio_path=str(audio), task_id="t", progress_callback=cb, output_format="json",
            )
        # transcribe 必须正常完成
        assert result.task_id == "t"

    @pytest.mark.asyncio
    async def test_no_callback_works(self, transcriber, tmp_path):
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result()), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=_fake_turns()), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            # progress_callback=None 是默认, 不应抛
            result, _ = await transcriber.transcribe(
                audio_path=str(audio), task_id="t", output_format="json",
            )
        assert result.task_id == "t"


class TestEngineSingletonReuse:
    """asr engine 加载一次后复用, 避免每次 transcribe 都加载 GGUF"""

    @pytest.mark.asyncio
    async def test_engine_built_once_across_two_calls(self, transcriber, tmp_path):
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result()), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=_fake_turns()), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()) as mock_build, \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            await transcriber.transcribe(audio_path=str(audio), task_id="t1", output_format="json")
            await transcriber.transcribe(audio_path=str(audio), task_id="t2", output_format="json")
        assert mock_build.call_count == 1, f"engine 应该只构造 1 次, 实际 {mock_build.call_count}"
