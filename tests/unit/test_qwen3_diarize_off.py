"""diarize 开关 (落地步骤 3) — qwen3 transcribe diarize=False 跳层 + speaker=null

设计 (定案文档 D2/D5/D8 + T-D #3/#4):
- diarize=False: 真跳过 run_diarization_dispatched + speaker 后处理层
  (filter_spurious / cluster_merge / short_segment_guard / relabel), 省算力
- 段直接来自 ASR ~40s chunk (无 chunks 时单段全文兜底); 超长段切分在步骤 4
- 出口: TranscriptionSegment.speaker=None (Optional, null=未区分, 与"真只有
  一人"可区分), speakers=[]; 内部 Segment(speaker:int) 永不为 None
- SRT 不带 SpeakerN: 前缀 (segments_to_srt include_speaker=False)
"""
from __future__ import annotations

from typing import List
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from src.core.qwen3.asr import ASRChunkItem, ASRResult
from src.core.qwen3.merge import Segment, segments_to_srt
from src.models.schemas import TranscribeOptions, TranscriptionSegment


def _fake_asr_result_with_chunks(duration: float = 90.0) -> ASRResult:
    return ASRResult(
        text="第一窗文本第二窗文本",
        items=[],
        chunks=[
            ASRChunkItem(text="第一窗文本", start=0.0, end=40.0, index=0),
            ASRChunkItem(text="第二窗文本", start=40.0, end=90.0, index=1),
        ],
        duration=duration,
        elapsed=1.0, rtf=0.1, peak_rss_mb=0.0, rss_delta_mb=0.0,
    )


def _fake_asr_result_no_chunks(duration: float = 10.0) -> ASRResult:
    return ASRResult(
        text="全文兜底文本", items=[], chunks=[],
        duration=duration, elapsed=1.0, rtf=0.1, peak_rss_mb=0.0, rss_delta_mb=0.0,
    )


@pytest.fixture
def transcriber():
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber
    return Qwen3DiarizeTranscriber(
        asr_model_dir="/fake/qwen3-asr",
        segmentation_model="/fake/seg.onnx",
        embedding_model="/fake/emb.onnx",
    )


# ==================== schema: speaker Optional ====================


def test_transcription_segment_speaker_accepts_none():
    seg = TranscriptionSegment(start_time=0, end_time=1, text="x", speaker=None)
    assert seg.speaker is None


def test_transcription_segment_speaker_null_in_json():
    seg = TranscriptionSegment(start_time=0, end_time=1, text="x", speaker=None)
    assert '"speaker":null' in seg.model_dump_json().replace(" ", "")


# ==================== diarize=False 跳层 ====================


class TestDiarizeOffSkipsLayers:
    @pytest.mark.asyncio
    async def test_diarize_not_called_and_speaker_null(self, transcriber, tmp_path):
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result_with_chunks()), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched") as mock_diar, \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            result, raw = await transcriber.transcribe(
                audio_path=str(audio), task_id="t", output_format="json",
                options=TranscribeOptions(diarize=False),
            )
        assert not mock_diar.called, "diarize=False 必须真跳过 run_diarization_dispatched"
        assert len(result.segments) == 2, "段应来自 ASR chunk 时间窗"
        assert all(s.speaker is None for s in result.segments)
        assert result.speakers == []
        # 文本完整覆盖
        assert "".join(s.text for s in result.segments) == "第一窗文本第二窗文本"
        # chunk 时间窗保留
        assert result.segments[0].start_time == 0.0
        assert result.segments[0].end_time == 40.0

    @pytest.mark.asyncio
    async def test_no_chunks_falls_back_to_single_segment(self, transcriber, tmp_path):
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result_no_chunks()), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched") as mock_diar, \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            result, _ = await transcriber.transcribe(
                audio_path=str(audio), task_id="t", output_format="json",
                options=TranscribeOptions(diarize=False),
            )
        assert not mock_diar.called
        assert len(result.segments) == 1
        assert result.segments[0].text == "全文兜底文本"
        assert result.segments[0].start_time == 0.0
        assert result.segments[0].end_time == 10.0
        assert result.segments[0].speaker is None

    @pytest.mark.asyncio
    async def test_speaker_postprocess_layers_skipped(self, transcriber, tmp_path):
        """speaker 后处理层 (filter_spurious / cluster_merge / short_guard) 不被调用"""
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result_with_chunks()), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched"), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.filter_spurious_speakers") as mock_filter, \
             patch("src.core.qwen3_transcriber.apply_cluster_centroid_merge_to_turns") as mock_cm, \
             patch("src.core.qwen3_transcriber.apply_short_segment_guard_to_segments") as mock_guard, \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            await transcriber.transcribe(
                audio_path=str(audio), task_id="t", output_format="json",
                options=TranscribeOptions(diarize=False),
            )
        assert not mock_filter.called
        assert not mock_cm.called
        assert not mock_guard.called

    @pytest.mark.asyncio
    async def test_diarize_default_path_unchanged(self, transcriber, tmp_path):
        """diarize 缺省 (True): 行为与历史一致, speaker 仍是 SpeakerN"""
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        turns = [{"start": 0.0, "end": 5.0, "speaker": 0}]
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result_no_chunks()), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=turns) as mock_diar, \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            result, _ = await transcriber.transcribe(
                audio_path=str(audio), task_id="t", output_format="json",
            )
        assert mock_diar.called
        assert result.segments[0].speaker == "Speaker1"
        assert result.speakers == ["Speaker1"]


# ==================== SRT 无前缀 ====================


class TestSrtNoPrefix:
    def test_segments_to_srt_include_speaker_false(self):
        segs = [Segment(start=0.0, end=1.0, speaker=0, text="你好")]
        srt = segments_to_srt(segs, include_speaker=False)
        assert "Speaker" not in srt
        assert "你好" in srt

    def test_segments_to_srt_default_keeps_prefix(self):
        segs = [Segment(start=0.0, end=1.0, speaker=0, text="你好")]
        srt = segments_to_srt(segs)
        assert "Speaker1:你好" in srt

    @pytest.mark.asyncio
    async def test_srt_mode_diarize_off_has_no_speaker_prefix(self, transcriber, tmp_path):
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result_with_chunks()), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched"), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            ret = await transcriber.transcribe(
                audio_path=str(audio), task_id="t", output_format="srt",
                options=TranscribeOptions(diarize=False),
            )
        assert ret["format"] == "srt"
        assert "Speaker" not in ret["content"]
        assert "第一窗文本" in ret["content"]
        # SRT 携带的 segments 同样 speaker=None
        assert all(s.speaker is None for s in ret["segments"])


# ==================== word_align 与 diarize 正交 ====================


@pytest.mark.asyncio
async def test_word_align_still_runs_with_diarize_off(tmp_path):
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber

    tx = Qwen3DiarizeTranscriber(
        asr_model_dir="/fake", segmentation_model="/fake/seg.onnx",
        embedding_model="/fake/emb.onnx", word_align_enabled=True,
    )
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"\x00")
    fake_aligner = MagicMock()
    fake_aligner.align_chunks.return_value = (
        [{"text": "第", "start": 0.1, "end": 0.4, "score": -1.0}],
        {"total_windows": 2, "failed_windows": 0, "total_words": 1, "failures": []},
    )
    with patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_result_with_chunks()), \
         patch("src.core.qwen3_transcriber.run_diarization_dispatched") as mock_diar, \
         patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
         patch("src.core.qwen3_transcriber._load_audio_mono_16k", return_value=([0.0], 16000)), \
         patch.object(tx, "_ensure_word_aligner", return_value=fake_aligner), \
         patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
        result, raw = await tx.transcribe(
            audio_path=str(audio), task_id="t", output_format="json",
            options=TranscribeOptions(language="chi", diarize=False, word_align=True),
        )
    assert not mock_diar.called
    all_words = [w for s in result.segments if s.words for w in s.words]
    assert [w.text for w in all_words] == ["第"]
    assert all(s.speaker is None for s in result.segments)


# ==================== database SRT 渲染点容忍 speaker=None ====================


def test_database_segments_to_srt_renders_none_speaker_without_prefix():
    from src.core.database import DatabaseManager

    db = DatabaseManager.__new__(DatabaseManager)
    segs = [TranscriptionSegment(start_time=0, end_time=1, text="你好", speaker=None)]
    srt = db._segments_to_srt(segs)
    assert "None" not in srt
    assert "Speaker" not in srt
    assert "你好" in srt
