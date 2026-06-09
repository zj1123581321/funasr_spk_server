"""
词级时间戳 — transcribe pipeline 挂钩 (word_align 层)

word_align 挂在 silence_align 之后、relabel 之前 (在干净段上挂词).
- flag 开: 段挂 words (TranscriptionSegment.words 填充), raw_result 带遥测.
- flag 关: 老路不变, words 恒 None, 不构造 aligner.
- 失败 (aligner build / 对齐抛错): 段照常出 (words=None), 不崩.

全部 mock, 不加载真 MMS ONNX.
"""
from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.core.qwen3.asr import ASRChunkItem, ASRResult


def _asr_with_chunks() -> ASRResult:
    return ASRResult(
        text="你好啊我是甲你好啊我是乙",
        items=[],
        chunks=[
            ASRChunkItem(text="你好啊我是甲", start=0.0, end=5.0, index=0),
            ASRChunkItem(text="你好啊我是乙", start=5.0, end=10.0, index=1),
        ],
        duration=10.0,
        elapsed=1.0,
        rtf=0.1,
        peak_rss_mb=100.0,
        rss_delta_mb=10.0,
    )


def _turns() -> List[dict]:
    return [
        {"start": 0.0, "end": 5.0, "speaker": 0},
        {"start": 5.0, "end": 10.0, "speaker": 1},
    ]


def _make_transcriber(**overrides):
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber

    kwargs = dict(
        asr_model_dir="/fake/asr",
        segmentation_model="/fake/seg.onnx",
        embedding_model="/fake/emb.onnx",
        cluster_merge_enabled=False,  # 简化: 不跑 sherpa extractor
        silence_align_enabled=False,
    )
    kwargs.update(overrides)
    return Qwen3DiarizeTranscriber(**kwargs)


_COMMON_PATCHES = lambda: [
    patch("src.core.qwen3_transcriber.run_asr", return_value=_asr_with_chunks()),
    patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=_turns()),
    patch("src.core.qwen3_transcriber.build_engine", return_value=object()),
    patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")),
    patch("src.core.qwen3_transcriber._load_audio_mono_16k", return_value=(np.zeros(160000, dtype=np.float32), 16000)),
]


@pytest.mark.asyncio
async def test_word_align_off_no_words(tmp_path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"\x00")
    tx = _make_transcriber(word_align_enabled=False)
    with patch.object(tx, "_ensure_word_aligner") as ensure:
        from contextlib import ExitStack
        with ExitStack() as st:
            for p in _COMMON_PATCHES():
                st.enter_context(p)
            result, raw = await tx.transcribe(str(audio), "t", output_format="json")
    # flag 关: 段无词, 不构造 aligner
    assert all(s.words is None for s in result.segments)
    ensure.assert_not_called()
    assert raw["word_align"]["enabled"] is False


@pytest.mark.asyncio
async def test_word_align_on_attaches_words(tmp_path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"\x00")
    tx = _make_transcriber(word_align_enabled=True)

    fake_aligner = MagicMock()
    fake_aligner.align_chunks.return_value = (
        [
            {"text": "你", "start": 0.5, "end": 0.8, "score": -1.0},
            {"text": "乙", "start": 6.0, "end": 6.3, "score": -1.2},
        ],
        {"total_windows": 2, "failed_windows": 0, "total_words": 2, "failures": []},
    )
    from contextlib import ExitStack
    with ExitStack() as st:
        for p in _COMMON_PATCHES():
            st.enter_context(p)
        st.enter_context(patch.object(tx, "_ensure_word_aligner", return_value=fake_aligner))
        result, raw = await tx.transcribe(str(audio), "t", output_format="json", language="chi")

    # 词被挂到对应时间窗的段
    all_words = [w for s in result.segments if s.words for w in s.words]
    assert {w.text for w in all_words} == {"你", "乙"}
    # 遥测进 raw_result
    assert raw["word_align"]["enabled"] is True
    assert raw["word_align"]["total_words"] == 2
    assert raw["word_align"]["language"] == "chi"
    # 语言透传给 aligner
    _, kwargs = fake_aligner.align_chunks.call_args
    assert kwargs.get("language") == "chi"


@pytest.mark.asyncio
async def test_word_align_failure_segments_still_emit(tmp_path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"\x00")
    tx = _make_transcriber(word_align_enabled=True)

    fake_aligner = MagicMock()
    fake_aligner.align_chunks.side_effect = RuntimeError("model load boom")
    from contextlib import ExitStack
    with ExitStack() as st:
        for p in _COMMON_PATCHES():
            st.enter_context(p)
        st.enter_context(patch.object(tx, "_ensure_word_aligner", return_value=fake_aligner))
        result, raw = await tx.transcribe(str(audio), "t", output_format="json")

    # 对齐失败 → 段照常出, 无词
    assert len(result.segments) == 2
    assert all(s.words is None for s in result.segments)
    assert raw["word_align"]["enabled"] is True
    assert "error" in raw["word_align"]


@pytest.mark.asyncio
async def test_word_align_falls_back_to_config_language(tmp_path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"\x00")
    tx = _make_transcriber(word_align_enabled=True, word_align_language="eng")

    fake_aligner = MagicMock()
    fake_aligner.align_chunks.return_value = ([], {"total_windows": 0, "failed_windows": 0, "total_words": 0, "failures": []})
    from contextlib import ExitStack
    with ExitStack() as st:
        for p in _COMMON_PATCHES():
            st.enter_context(p)
        st.enter_context(patch.object(tx, "_ensure_word_aligner", return_value=fake_aligner))
        # 不传 language → 走 config 兜底 "eng"
        result, raw = await tx.transcribe(str(audio), "t", output_format="json")

    _, kwargs = fake_aligner.align_chunks.call_args
    assert kwargs.get("language") == "eng"
    assert raw["word_align"]["language"] == "eng"
