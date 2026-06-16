"""Lane 1 (#17) — word_align CUDA preflight gate (加载 CUDA session 前探显存).

决策 P1 + A2: primary 解析成 CUDA 且 free VRAM < 阈值 → 直走 CPU (不等 OOM).
- 显存不足 → CPU, raw 标 preflight_skipped_cuda, CUDA align 不被调.
- 显存够 → 照走 CUDA.
- 探不到 (None) → 不误杀, 照走 CUDA (codex #11, 交给 OOM fallback).
- preflight 关 → 不探, 照走 CUDA.
- primary 非 CUDA (Mac/CPU) → 不探 (省一次 nvidia-smi), 照走 primary.

全部 mock, 不加载真 MMS ONNX / 不跑真 nvidia-smi.
"""
from __future__ import annotations

from contextlib import ExitStack
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
        duration=10.0, elapsed=1.0, rtf=0.1, peak_rss_mb=100.0, rss_delta_mb=10.0,
    )


def _turns() -> List[dict]:
    return [{"start": 0.0, "end": 5.0, "speaker": 0}, {"start": 5.0, "end": 10.0, "speaker": 1}]


_COMMON_PATCHES = lambda: [
    patch("src.core.qwen3_transcriber.run_asr", return_value=_asr_with_chunks()),
    patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=_turns()),
    patch("src.core.qwen3_transcriber.build_engine", return_value=object()),
    patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")),
    patch("src.core.qwen3_transcriber._load_audio_mono_16k",
          return_value=(np.zeros(160000, dtype=np.float32), 16000)),
]


@pytest.fixture(autouse=True)
def _reset_poison():
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber
    Qwen3DiarizeTranscriber._cuda_word_align_poisoned = False
    yield
    Qwen3DiarizeTranscriber._cuda_word_align_poisoned = False


def _make_transcriber(**overrides):
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber
    kwargs = dict(
        asr_model_dir="/fake/asr", segmentation_model="/fake/seg.onnx",
        embedding_model="/fake/emb.onnx", cluster_merge_enabled=False, silence_align_enabled=False,
        word_align_enabled=True,
    )
    kwargs.update(overrides)
    return Qwen3DiarizeTranscriber(**kwargs)


def _opts(word_align=True, language="chi"):
    from src.models.schemas import TranscribeOptions
    return TranscribeOptions(word_align=word_align, language=language)


def _aligner(is_cuda, provider, words):
    a = MagicMock()
    a.is_cuda = is_cuda
    a.effective_provider = provider
    a.align_chunks.return_value = (
        words, {"total_windows": len(words), "failed_windows": 0, "total_words": len(words), "failures": []},
    )
    return a


async def _run_transcribe(tx, tmp_path, free_value, *, patch_probe=True):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"\x00")
    with ExitStack() as st:
        for p in _COMMON_PATCHES():
            st.enter_context(p)
        if patch_probe:
            st.enter_context(patch("src.core.qwen3_transcriber.free_vram_mib", return_value=free_value))
        return await tx.transcribe(str(audio), "t", output_format="json", options=_opts())


@pytest.mark.asyncio
async def test_preflight_insufficient_vram_routes_to_cpu(tmp_path):
    """free < 阈值 → 走 CPU, CUDA align 不被调, raw 标 preflight_skipped_cuda."""
    tx = _make_transcriber(word_align_preflight_free_mib=4608)
    cuda = _aligner(True, "cuda", [{"text": "X", "start": 0.1, "end": 0.2, "score": -1.0}])
    cpu = _aligner(False, "cpu", [{"text": "你", "start": 0.5, "end": 0.8, "score": -1.0}])
    with patch.object(tx, "_ensure_word_aligner", return_value=cuda), \
         patch.object(tx, "_ensure_word_aligner_cpu", return_value=cpu):
        result, raw = await _run_transcribe(tx, tmp_path, free_value=1200)

    cuda.align_chunks.assert_not_called()
    cpu.align_chunks.assert_called_once()
    assert raw["word_align"]["provider"] == "cpu"
    assert raw["word_align"].get("preflight_skipped_cuda") is True
    all_words = [w for s in result.segments if s.words for w in s.words]
    assert {w.text for w in all_words} == {"你"}


@pytest.mark.asyncio
async def test_preflight_enough_vram_uses_cuda(tmp_path):
    """free ≥ 阈值 → 照走 CUDA."""
    tx = _make_transcriber(word_align_preflight_free_mib=4608)
    cuda = _aligner(True, "cuda", [{"text": "甲", "start": 0.1, "end": 0.2, "score": -1.0}])
    cpu = _aligner(False, "cpu", [])
    with patch.object(tx, "_ensure_word_aligner", return_value=cuda), \
         patch.object(tx, "_ensure_word_aligner_cpu", return_value=cpu):
        result, raw = await _run_transcribe(tx, tmp_path, free_value=6000)

    cuda.align_chunks.assert_called_once()
    cpu.align_chunks.assert_not_called()
    assert raw["word_align"]["provider"] == "cuda"


@pytest.mark.asyncio
async def test_preflight_probe_none_does_not_block(tmp_path):
    """探不到 (None) → 不误杀, 照走 CUDA (交给 OOM fallback, codex #11)."""
    tx = _make_transcriber(word_align_preflight_free_mib=4608)
    cuda = _aligner(True, "cuda", [{"text": "甲", "start": 0.1, "end": 0.2, "score": -1.0}])
    cpu = _aligner(False, "cpu", [])
    with patch.object(tx, "_ensure_word_aligner", return_value=cuda), \
         patch.object(tx, "_ensure_word_aligner_cpu", return_value=cpu):
        result, raw = await _run_transcribe(tx, tmp_path, free_value=None)

    cuda.align_chunks.assert_called_once()
    assert raw["word_align"]["provider"] == "cuda"


@pytest.mark.asyncio
async def test_preflight_disabled_uses_cuda_regardless(tmp_path):
    """preflight 关 → 不探显存, 照走 CUDA 即使 free 很低."""
    tx = _make_transcriber(word_align_preflight_enabled=False, word_align_preflight_free_mib=4608)
    cuda = _aligner(True, "cuda", [{"text": "甲", "start": 0.1, "end": 0.2, "score": -1.0}])
    cpu = _aligner(False, "cpu", [])
    with patch.object(tx, "_ensure_word_aligner", return_value=cuda), \
         patch.object(tx, "_ensure_word_aligner_cpu", return_value=cpu):
        # probe 即使返回低值也不该被查 (assert 未调用)
        with patch("src.core.qwen3_transcriber.free_vram_mib", return_value=10) as probe:
            audio = tmp_path / "x.wav"; audio.write_bytes(b"\x00")
            with ExitStack() as st:
                for p in _COMMON_PATCHES():
                    st.enter_context(p)
                result, raw = await tx.transcribe(str(audio), "t", output_format="json", options=_opts())
    cuda.align_chunks.assert_called_once()
    probe.assert_not_called()
    assert raw["word_align"]["provider"] == "cuda"


@pytest.mark.asyncio
async def test_preflight_skipped_when_primary_is_cpu(tmp_path):
    """primary 非 CUDA (Mac/CPU) → 不探显存 (省 nvidia-smi), 照走 primary CPU."""
    tx = _make_transcriber(word_align_preflight_free_mib=4608)
    cpu_primary = _aligner(False, "cpu", [{"text": "甲", "start": 0.1, "end": 0.2, "score": -1.0}])
    with patch.object(tx, "_ensure_word_aligner", return_value=cpu_primary):
        with patch("src.core.qwen3_transcriber.free_vram_mib", return_value=10) as probe:
            audio = tmp_path / "x.wav"; audio.write_bytes(b"\x00")
            with ExitStack() as st:
                for p in _COMMON_PATCHES():
                    st.enter_context(p)
                result, raw = await tx.transcribe(str(audio), "t", output_format="json", options=_opts())
    cpu_primary.align_chunks.assert_called_once()
    probe.assert_not_called()
    assert raw["word_align"]["provider"] == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
