"""Lane 2 (#18) — word_align 集成: cuda runtime 把 CUDA 对齐路由到 sidecar.

决策 A1/A3/A4 + codex #7:
- cuda + sidecar_enabled → client.align (进程内永不建 CUDA session, codex #7 硬切)
- sidecar 超时/资源错误/不可用 → 主进程 CPU 兜底 (A3)
- sidecar 普通对齐错误 → 无词 (sidecar 健康)
- preflight 不足 → 主进程 CPU (连 sidecar 都不拉)
- sidecar 关 → 进程内路径 (Lane 1)
- 非 CUDA (Mac/CPU) → 进程内 CPU primary, 不碰 sidecar (A4)

全 mock, 不起真 sidecar 进程 / 不加载真 ONNX.
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
        text="你好啊我是甲你好啊我是乙", items=[],
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
        word_align_enabled=True, word_align_sidecar_enabled=True,
    )
    kwargs.update(overrides)
    return Qwen3DiarizeTranscriber(**kwargs)


def _opts(word_align=True, language="chi"):
    from src.models.schemas import TranscribeOptions
    return TranscribeOptions(word_align=word_align, language=language)


def _cpu_aligner(words):
    a = MagicMock()
    a.is_cuda = False
    a.effective_provider = "cpu"
    a.align_chunks.return_value = (
        words, {"total_windows": 1, "failed_windows": 0, "total_words": len(words), "failures": []},
    )
    return a


async def _transcribe(tx, tmp_path, *, sidecar_client, primary_is_cuda, free_value=9000):
    audio = tmp_path / "x.wav"; audio.write_bytes(b"\x00")
    with ExitStack() as st:
        for p in _COMMON_PATCHES():
            st.enter_context(p)
        st.enter_context(patch.object(tx, "_word_align_primary_is_cuda", return_value=primary_is_cuda))
        st.enter_context(patch("src.core.qwen3_transcriber.get_word_align_sidecar_client", return_value=sidecar_client))
        st.enter_context(patch("src.core.qwen3_transcriber.free_vram_mib", return_value=free_value))
        return await tx.transcribe(str(audio), "t", output_format="json", options=_opts())


@pytest.mark.asyncio
async def test_cuda_routes_to_sidecar(tmp_path):
    """cuda + sidecar 启用 → client.align 出词; 进程内 CUDA aligner 不被构造 (codex #7)."""
    tx = _make_transcriber()
    client = MagicMock()
    client.align.return_value = (
        [{"text": "你", "start": 0.5, "end": 0.8, "score": -1.0},
         {"text": "乙", "start": 6.0, "end": 6.3, "score": -1.2}],
        {"total_windows": 2, "failed_windows": 0, "total_words": 2, "failures": []},
    )
    with patch.object(tx, "_ensure_word_aligner") as ensure_cuda:
        result, raw = await _transcribe(tx, tmp_path, sidecar_client=client, primary_is_cuda=True)

    # 进程内 CUDA aligner 永不构造 (硬切, codex #7)
    ensure_cuda.assert_not_called()
    client.align.assert_called_once()
    args, _ = client.align.call_args
    assert args[0] == str(tmp_path / "x.wav")        # audio_path 透传
    assert args[2] == "chi"                          # language
    assert [c["text"] for c in args[1]] == ["你好啊我是甲", "你好啊我是乙"]  # chunks 转 dict
    all_words = [w for s in result.segments if s.words for w in s.words]
    assert {w.text for w in all_words} == {"你", "乙"}
    assert raw["word_align"]["provider"] == "cuda_sidecar"
    assert raw["word_align"]["total_words"] == 2


@pytest.mark.asyncio
async def test_sidecar_timeout_falls_back_to_cpu(tmp_path):
    """sidecar 超时 → 主进程 CPU 兜底."""
    from src.core.qwen3.word_align_sidecar import SidecarTimeout
    tx = _make_transcriber()
    client = MagicMock(); client.align.side_effect = SidecarTimeout("超时")
    cpu = _cpu_aligner([{"text": "甲", "start": 0.1, "end": 0.3, "score": -1.0}])
    with patch.object(tx, "_ensure_word_aligner_cpu", return_value=cpu):
        result, raw = await _transcribe(tx, tmp_path, sidecar_client=client, primary_is_cuda=True)
    cpu.align_chunks.assert_called_once()
    assert raw["word_align"]["provider"] == "cpu"
    assert raw["word_align"].get("sidecar_fallback") == "SidecarTimeout"
    all_words = [w for s in result.segments if s.words for w in s.words]
    assert {w.text for w in all_words} == {"甲"}


@pytest.mark.asyncio
async def test_sidecar_resource_error_falls_back_to_cpu(tmp_path):
    """sidecar CUDA 资源错误 (已退休) → 主进程 CPU."""
    from src.core.qwen3.word_align_sidecar import SidecarResourceError
    tx = _make_transcriber()
    client = MagicMock(); client.align.side_effect = SidecarResourceError("CUBLAS")
    cpu = _cpu_aligner([{"text": "乙", "start": 6.0, "end": 6.3, "score": -1.0}])
    with patch.object(tx, "_ensure_word_aligner_cpu", return_value=cpu):
        result, raw = await _transcribe(tx, tmp_path, sidecar_client=client, primary_is_cuda=True)
    cpu.align_chunks.assert_called_once()
    assert raw["word_align"]["provider"] == "cpu"
    assert raw["word_align"].get("sidecar_fallback") == "SidecarResourceError"


@pytest.mark.asyncio
async def test_sidecar_unavailable_falls_back_to_cpu(tmp_path):
    """sidecar 起不来 → 主进程 CPU."""
    from src.core.qwen3.word_align_sidecar import SidecarUnavailable
    tx = _make_transcriber()
    client = MagicMock(); client.align.side_effect = SidecarUnavailable("起不来")
    cpu = _cpu_aligner([{"text": "甲", "start": 0.1, "end": 0.3, "score": -1.0}])
    with patch.object(tx, "_ensure_word_aligner_cpu", return_value=cpu):
        result, raw = await _transcribe(tx, tmp_path, sidecar_client=client, primary_is_cuda=True)
    cpu.align_chunks.assert_called_once()
    assert raw["word_align"]["provider"] == "cpu"


@pytest.mark.asyncio
async def test_sidecar_align_error_no_words(tmp_path):
    """sidecar 普通对齐错误 (非资源) → 无词, sidecar 仍健康 (不 CPU 兜底)."""
    from src.core.qwen3.word_align_sidecar import SidecarAlignError
    tx = _make_transcriber()
    client = MagicMock(); client.align.side_effect = SidecarAlignError("weird")
    cpu = _cpu_aligner([{"text": "X", "start": 0, "end": 1, "score": -1.0}])
    with patch.object(tx, "_ensure_word_aligner_cpu", return_value=cpu):
        result, raw = await _transcribe(tx, tmp_path, sidecar_client=client, primary_is_cuda=True)
    cpu.align_chunks.assert_not_called()  # 普通错误不 CPU 兜底
    assert all(s.words is None for s in result.segments)
    assert "error" in raw["word_align"]


@pytest.mark.asyncio
async def test_preflight_insufficient_skips_sidecar(tmp_path):
    """显存不足 → 主进程 CPU, 连 sidecar 都不拉 (client.align 不调)."""
    tx = _make_transcriber(word_align_preflight_free_mib=4608)
    client = MagicMock()
    cpu = _cpu_aligner([{"text": "甲", "start": 0.1, "end": 0.3, "score": -1.0}])
    with patch.object(tx, "_ensure_word_aligner_cpu", return_value=cpu):
        result, raw = await _transcribe(tx, tmp_path, sidecar_client=client, primary_is_cuda=True, free_value=1000)
    client.align.assert_not_called()
    cpu.align_chunks.assert_called_once()
    assert raw["word_align"].get("preflight_skipped_cuda") is True


@pytest.mark.asyncio
async def test_sidecar_disabled_uses_inprocess(tmp_path):
    """sidecar 关 → 进程内路径 (Lane 1), 不碰 sidecar client."""
    tx = _make_transcriber(word_align_sidecar_enabled=False)
    cuda_inproc = MagicMock()
    cuda_inproc.is_cuda = True
    cuda_inproc.effective_provider = "cuda"
    cuda_inproc.align_chunks.return_value = (
        [{"text": "甲", "start": 0.1, "end": 0.3, "score": -1.0}],
        {"total_windows": 1, "failed_windows": 0, "total_words": 1, "failures": []},
    )
    client = MagicMock()
    audio = tmp_path / "x.wav"; audio.write_bytes(b"\x00")
    with ExitStack() as st:
        for p in _COMMON_PATCHES():
            st.enter_context(p)
        st.enter_context(patch.object(tx, "_ensure_word_aligner", return_value=cuda_inproc))
        st.enter_context(patch("src.core.qwen3_transcriber.get_word_align_sidecar_client", return_value=client))
        st.enter_context(patch("src.core.qwen3_transcriber.free_vram_mib", return_value=9000))
        result, raw = await tx.transcribe(str(audio), "t", output_format="json", options=_opts())
    client.align.assert_not_called()
    cuda_inproc.align_chunks.assert_called_once()  # 进程内 CUDA
    assert raw["word_align"]["provider"] == "cuda"


@pytest.mark.asyncio
async def test_non_cuda_runtime_uses_inprocess_cpu(tmp_path):
    """Mac/CPU (primary 非 CUDA) → 进程内 CPU primary, 不碰 sidecar (A4)."""
    tx = _make_transcriber()
    cpu_primary = _cpu_aligner([{"text": "甲", "start": 0.1, "end": 0.3, "score": -1.0}])
    client = MagicMock()
    audio = tmp_path / "x.wav"; audio.write_bytes(b"\x00")
    with ExitStack() as st:
        for p in _COMMON_PATCHES():
            st.enter_context(p)
        st.enter_context(patch.object(tx, "_word_align_primary_is_cuda", return_value=False))
        st.enter_context(patch.object(tx, "_ensure_word_aligner", return_value=cpu_primary))
        st.enter_context(patch("src.core.qwen3_transcriber.get_word_align_sidecar_client", return_value=client))
        result, raw = await tx.transcribe(str(audio), "t", output_format="json", options=_opts())
    client.align.assert_not_called()
    cpu_primary.align_chunks.assert_called_once()
    assert raw["word_align"]["provider"] == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
