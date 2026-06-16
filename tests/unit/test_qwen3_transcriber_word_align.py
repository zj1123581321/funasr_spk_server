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


@pytest.fixture(autouse=True)
def _reset_poison():
    """每个测试前后复位 pool 级 CUDA word_align poison flag (class attr, 进程共享)."""
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber
    Qwen3DiarizeTranscriber._cuda_word_align_poisoned = False
    yield
    Qwen3DiarizeTranscriber._cuda_word_align_poisoned = False


def _opts(word_align=True, language=None):
    from src.models.schemas import TranscribeOptions
    return TranscribeOptions(word_align=word_align, language=language)


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
        result, raw = await tx.transcribe(
            str(audio), "t", output_format="json",
            options=_opts(word_align=True, language="chi"),
        )

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
        result, raw = await tx.transcribe(
            str(audio), "t", output_format="json", options=_opts(word_align=True),
        )

    # 对齐失败 (非资源错误) → 段照常出, 无词, 不 fallback
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
        result, raw = await tx.transcribe(
            str(audio), "t", output_format="json", options=_opts(word_align=True),
        )

    _, kwargs = fake_aligner.align_chunks.call_args
    assert kwargs.get("language") == "eng"
    assert raw["word_align"]["language"] == "eng"


# ==================== CPU fallback + OOM poison (2026-06-16 显存落地, Lane C-2) ====================


def _cuda_aligner_raising(exc):
    """模拟 primary CUDA aligner: is_cuda=True, align_chunks 抛指定异常."""
    a = MagicMock()
    a.is_cuda = True
    a.effective_provider = "cuda"
    a.align_chunks.side_effect = exc
    return a


def _ok_aligner(words, provider="cpu"):
    a = MagicMock()
    a.is_cuda = provider == "cuda"
    a.effective_provider = provider
    a.align_chunks.return_value = (
        words, {"total_windows": len(words), "failed_windows": 0, "total_words": len(words), "failures": []},
    )
    return a


@pytest.mark.asyncio
async def test_cuda_oom_falls_back_to_cpu_and_poisons(tmp_path):
    """CUDA 资源错误 → poison pool + dispose + 转 CPU 重试成功 → 挂词, provider=cpu."""
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber

    audio = tmp_path / "x.wav"
    audio.write_bytes(b"\x00")
    tx = _make_transcriber(word_align_enabled=True)

    cuda_aligner = _cuda_aligner_raising(RuntimeError("CUBLAS failure 3: the resource allocation failed"))
    cpu_aligner = _ok_aligner([{"text": "你", "start": 0.5, "end": 0.8, "score": -1.0}], provider="cpu")

    from contextlib import ExitStack
    with ExitStack() as st:
        for p in _COMMON_PATCHES():
            st.enter_context(p)
        st.enter_context(patch.object(tx, "_ensure_word_aligner", return_value=cuda_aligner))
        st.enter_context(patch.object(tx, "_ensure_word_aligner_cpu", return_value=cpu_aligner))
        result, raw = await tx.transcribe(
            str(audio), "t", output_format="json", options=_opts(word_align=True, language="chi"),
        )

    # CPU fallback 挂上词
    all_words = [w for s in result.segments if s.words for w in s.words]
    assert {w.text for w in all_words} == {"你"}
    assert raw["word_align"]["enabled"] is True
    assert raw["word_align"]["provider"] == "cpu"
    assert raw["word_align"].get("cuda_oom_fallback") is True
    # pool 级 poison flag 置位 + CUDA aligner 被 dispose
    assert Qwen3DiarizeTranscriber._cuda_word_align_poisoned is True
    cpu_aligner.align_chunks.assert_called_once()


@pytest.mark.asyncio
async def test_poison_persists_next_request_goes_straight_cpu(tmp_path):
    """poison 后, 后续请求直走 CPU, 不再构造/调用 CUDA aligner."""
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber

    audio = tmp_path / "x.wav"
    audio.write_bytes(b"\x00")
    tx = _make_transcriber(word_align_enabled=True)
    # 预先 poison
    Qwen3DiarizeTranscriber._cuda_word_align_poisoned = True

    cpu_aligner = _ok_aligner([{"text": "甲", "start": 0.1, "end": 0.3, "score": -1.0}], provider="cpu")
    ensure_cuda = MagicMock()

    from contextlib import ExitStack
    with ExitStack() as st:
        for p in _COMMON_PATCHES():
            st.enter_context(p)
        st.enter_context(patch.object(tx, "_ensure_word_aligner", ensure_cuda))
        st.enter_context(patch.object(tx, "_ensure_word_aligner_cpu", return_value=cpu_aligner))
        result, raw = await tx.transcribe(
            str(audio), "t", output_format="json", options=_opts(word_align=True, language="chi"),
        )

    # 直走 CPU, 完全不碰 CUDA aligner
    ensure_cuda.assert_not_called()
    cpu_aligner.align_chunks.assert_called_once()
    assert raw["word_align"]["provider"] == "cpu"
    all_words = [w for s in result.segments if s.words for w in s.words]
    assert {w.text for w in all_words} == {"甲"}


@pytest.mark.asyncio
async def test_cuda_oom_then_cpu_also_fails_no_words(tmp_path):
    """CUDA 资源错误 + CPU fallback 也失败 → 段不带词 + error, 整请求不挂."""
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"\x00")
    tx = _make_transcriber(word_align_enabled=True)

    cuda_aligner = _cuda_aligner_raising(RuntimeError("BFCArena Failed to allocate memory"))
    cpu_aligner = MagicMock()
    cpu_aligner.is_cuda = False
    cpu_aligner.effective_provider = "cpu"
    cpu_aligner.align_chunks.side_effect = RuntimeError("cpu boom too")

    from contextlib import ExitStack
    with ExitStack() as st:
        for p in _COMMON_PATCHES():
            st.enter_context(p)
        st.enter_context(patch.object(tx, "_ensure_word_aligner", return_value=cuda_aligner))
        st.enter_context(patch.object(tx, "_ensure_word_aligner_cpu", return_value=cpu_aligner))
        result, raw = await tx.transcribe(
            str(audio), "t", output_format="json", options=_opts(word_align=True),
        )

    assert len(result.segments) == 2
    assert all(s.words is None for s in result.segments)
    assert raw["word_align"]["enabled"] is True
    assert "error" in raw["word_align"]
