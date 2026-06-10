"""diarize 开关 (落地步骤 4) — transcriber 薄 wrapper: nospk 分层切段挂进 transcribe

- diarize=False + 段超长 → split_long_segments 接管 (词切→静音切)
- SRT 路径永远走静音 fallback (word_align JSON-only 不变量, T-D #5)
- nospk_split_enabled=False → 不切 (照 apply_silence_align_to_segments 形状)
- ffmpeg 失败 → fallback to input, 不阻塞转录
- diarize=True 路径不受影响 (切段只挂 nospk 分支)
"""
from __future__ import annotations

from unittest.mock import patch, AsyncMock

import pytest

from src.core.qwen3.asr import ASRChunkItem, ASRResult
from src.models.schemas import TranscribeOptions


def _fake_asr_40s_chunk() -> ASRResult:
    return ASRResult(
        text="字" * 40, items=[],
        chunks=[ASRChunkItem(text="字" * 40, start=0.0, end=40.0, index=0)],
        duration=40.0, elapsed=1.0, rtf=0.1, peak_rss_mb=0.0, rss_delta_mb=0.0,
    )


def _regions_with_gaps(*centers, width=0.4, total=40.0):
    regions, cursor = [], 0.0
    for c in sorted(centers):
        regions.append({"start": cursor, "end": c - width / 2})
        cursor = c + width / 2
    regions.append({"start": cursor, "end": total})
    return regions


def _make_tx(**kw):
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber
    defaults = dict(
        asr_model_dir="/fake", segmentation_model="/fake/seg.onnx",
        embedding_model="/fake/emb.onnx",
        silence_align_enabled=False,  # 隔离: 只测 split 层
    )
    defaults.update(kw)
    return Qwen3DiarizeTranscriber(**defaults)


def _patches(regions):
    return [
        patch("src.core.qwen3_transcriber.run_asr", return_value=_fake_asr_40s_chunk()),
        patch("src.core.qwen3_transcriber.run_diarization_dispatched"),
        patch("src.core.qwen3_transcriber.build_engine", return_value=object()),
        patch("src.core.qwen3_transcriber.ffmpeg_speech_regions", return_value=regions),
        patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")),
    ]


@pytest.mark.asyncio
async def test_nospk_long_chunk_split_by_silence(tmp_path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"\x00")
    tx = _make_tx(nospk_split_max_segment_sec=15.0)
    from contextlib import ExitStack
    with ExitStack() as st:
        for p in _patches(_regions_with_gaps(13.0, 26.0)):
            st.enter_context(p)
        result, raw = await tx.transcribe(
            audio_path=str(audio), task_id="t", output_format="json",
            options=TranscribeOptions(diarize=False),
        )
    assert len(result.segments) == 3, "40s chunk 应按静音中点切成 3 片"
    assert result.segments[0].end_time == pytest.approx(13.0, abs=0.01)
    assert all(s.speaker is None for s in result.segments)
    assert "".join(s.text for s in result.segments) == "字" * 40
    # stats 进 raw_result (可观测性)
    assert raw["nospk_split"]["split_segments"] == 1


@pytest.mark.asyncio
async def test_nospk_srt_split_uses_silence_fallback_and_no_prefix(tmp_path):
    """SRT 路径无 words (word_align JSON-only) → 永远静音 fallback"""
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"\x00")
    tx = _make_tx(word_align_enabled=True)  # 即使 word_align 开, SRT 也不挂词
    from contextlib import ExitStack
    with ExitStack() as st:
        for p in _patches(_regions_with_gaps(13.0, 26.0)):
            st.enter_context(p)
        ret = await tx.transcribe(
            audio_path=str(audio), task_id="t", output_format="srt",
            options=TranscribeOptions(diarize=False),
        )
    assert ret["format"] == "srt"
    assert "Speaker" not in ret["content"]
    # 3 片 → 3 个 SRT block 编号
    assert "\n3\n" in "\n" + ret["content"]
    assert ret["raw_result"]["nospk_split"]["silence_split"] == 1
    assert ret["raw_result"]["nospk_split"]["word_split"] == 0


@pytest.mark.asyncio
async def test_nospk_split_disabled_keeps_chunk_segments(tmp_path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"\x00")
    tx = _make_tx(nospk_split_enabled=False)
    from contextlib import ExitStack
    with ExitStack() as st:
        for p in _patches(_regions_with_gaps(13.0, 26.0)):
            st.enter_context(p)
        result, raw = await tx.transcribe(
            audio_path=str(audio), task_id="t", output_format="json",
            options=TranscribeOptions(diarize=False),
        )
    assert len(result.segments) == 1
    assert raw["nospk_split"]["enabled"] is False


@pytest.mark.asyncio
async def test_nospk_split_ffmpeg_failure_falls_back_to_input(tmp_path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"\x00")
    tx = _make_tx()
    from contextlib import ExitStack
    with ExitStack() as st:
        for p in _patches(None):
            st.enter_context(p)
        # 覆盖 ffmpeg patch 为抛错
        st.enter_context(patch(
            "src.core.qwen3_transcriber.ffmpeg_speech_regions",
            side_effect=RuntimeError("ffmpeg boom"),
        ))
        result, raw = await tx.transcribe(
            audio_path=str(audio), task_id="t", output_format="json",
            options=TranscribeOptions(diarize=False),
        )
    assert len(result.segments) == 1, "ffmpeg 失败 → fallback to input"
    assert "error" in raw["nospk_split"]


@pytest.mark.asyncio
async def test_diarize_on_path_not_split(tmp_path):
    """diarize=True 不挂 nospk 切段 (turn 段不受 max_segment_sec 影响)"""
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"\x00")
    tx = _make_tx()
    turns = [{"start": 0.0, "end": 40.0, "speaker": 0}]
    from contextlib import ExitStack
    with ExitStack() as st:
        for p in _patches(_regions_with_gaps(13.0, 26.0)):
            st.enter_context(p)
        st.enter_context(patch(
            "src.core.qwen3_transcriber.run_diarization_dispatched", return_value=turns,
        ))
        result, raw = await tx.transcribe(
            audio_path=str(audio), task_id="t", output_format="json",
        )
    assert len(result.segments) == 1, "diarize=True 的 40s turn 段不被 nospk 切分"
    assert raw["nospk_split"] == {"enabled": False, "skipped": "diarize_on"}
