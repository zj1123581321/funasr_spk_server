"""测试 transcribe 输出层 Speaker ID 跨 backend 稳定 (按总时长降序 relabel).

业务问题: raw diarize 输出的 cluster int 是 backend-dependent (cuda ort_cuda
出 55/41, Mac sherpa 出 0/3 这种), 输出层 +1 直接转字符串导致客户端看到的
Speaker ID 跨平台不稳定. transcribe 在输出前应该 relabel, 让 Speaker1 始终
是说话最多的人.
"""
from __future__ import annotations

from typing import List
from unittest.mock import patch, AsyncMock

import pytest

from src.core.qwen3.asr import ASRResult


def _fake_asr_result_long(text: str, duration: float) -> ASRResult:
    return ASRResult(
        text=text,
        items=[],
        chunks=[],
        duration=duration,
        elapsed=1.0,
        rtf=0.0125,
        peak_rss_mb=0.0,
        rss_delta_mb=0.0,
    )


@pytest.fixture
def transcriber_no_postprocess():
    """transcriber 实例, 关闭 cluster_merge/short_guard/silence_align 三层,
    隔离 relabel 单点行为, 不依赖 sherpa/ffmpeg 外部 mock."""
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber
    return Qwen3DiarizeTranscriber(
        asr_model_dir="/fake/qwen3-asr",
        segmentation_model="/fake/seg.onnx",
        embedding_model="/fake/emb.onnx",
        cluster_merge_enabled=False,
        short_segment_guard_enabled=False,
        silence_align_enabled=False,
    )


class TestSpeakerIdStabilityByDurationDesc:
    """raw cluster int → Speaker{i+1} 输出, i 应按总时长降序排."""

    @pytest.mark.asyncio
    async def test_speaker1_is_dominant_when_raw_id_is_55(self, transcriber_no_postprocess, tmp_path):
        """模拟 cuda ort_cuda 场景: raw cluster id 是 55 (主) / 41 (次).

        主说话人时长 60s, 次主 20s. relabel 后:
          int 55 (主, 60s) → 0 → Speaker1
          int 41 (次, 20s) → 1 → Speaker2

        关键: Speaker1 必须是主说话人, 而不是简单的 raw_int + 1 = Speaker56.
        """
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        # 4 个 turn: 主 spk=55 占 60s (30+30), 次 spk=41 占 20s (10+10)
        fake_turns: List[dict] = [
            {"start": 0.0, "end": 30.0, "speaker": 55},
            {"start": 30.0, "end": 40.0, "speaker": 41},
            {"start": 40.0, "end": 70.0, "speaker": 55},
            {"start": 70.0, "end": 80.0, "speaker": 41},
        ]
        asr = _fake_asr_result_long(text="aaaabbbbccccdddd", duration=80.0)

        with patch("src.core.qwen3_transcriber.run_asr", return_value=asr), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=fake_turns), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            result, _ = await transcriber_no_postprocess.transcribe(
                audio_path=str(audio), task_id="t-stable", output_format="json",
            )

        # 期望: 主 (55) → Speaker1, 次 (41) → Speaker2, 不论 raw int 多大
        speakers_in_order = [s.speaker for s in result.segments]
        assert speakers_in_order == ["Speaker1", "Speaker2", "Speaker1", "Speaker2"], \
            f"Speaker1 应该是主说话人 (raw int 55), 实际: {speakers_in_order}"

        # 客户端看到的 speakers list 不含 "Speaker56" / "Speaker42" 这种 raw int 残留
        all_speakers = set(s.speaker for s in result.segments)
        assert all_speakers == {"Speaker1", "Speaker2"}
        assert "Speaker56" not in all_speakers
        assert "Speaker42" not in all_speakers

    @pytest.mark.asyncio
    async def test_speaker1_only_when_single_speaker_raw_id_42(self, transcriber_no_postprocess, tmp_path):
        """单人独白 raw int=42 场景, 输出全是 Speaker1, 不是 Speaker43."""
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")
        fake_turns: List[dict] = [
            {"start": 0.0, "end": 30.0, "speaker": 42},
            {"start": 30.0, "end": 60.0, "speaker": 42},
        ]
        asr = _fake_asr_result_long(text="aaaabbbb", duration=60.0)

        with patch("src.core.qwen3_transcriber.run_asr", return_value=asr), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=fake_turns), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")):
            result, _ = await transcriber_no_postprocess.transcribe(
                audio_path=str(audio), task_id="t-single", output_format="json",
            )

        assert all(s.speaker == "Speaker1" for s in result.segments)
        assert "Speaker43" not in set(s.speaker for s in result.segments)
