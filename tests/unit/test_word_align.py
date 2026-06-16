"""
词级时间戳 wrapper (src/core/qwen3/word_align.py) 单元测试

不加载真 MMS ONNX / ctc_forced_aligner 重计算, 全部 mock:
- resolve_word_align_providers: auto → runtime 推荐; 显式 cpu/cuda/EP 名
- build_alignment_session: 模型缺失 fail-fast; happy 走 patched onnxruntime
- align_window: patched ctc 五步函数, 验证 happy / 空文本
- align_chunks: 逐 window 喂, 词时间 offset chunk.start; 某 chunk 失败 stats 记录, 不崩
- WordAligner: lazy 单次 build session
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from src.core.qwen3 import word_align


# ==================== resolve_word_align_providers ====================


def test_resolve_auto_uses_runtime_recommend():
    fake_runtime = MagicMock()
    fake_runtime.recommend_word_align_provider.return_value = "CUDAExecutionProvider"
    providers = word_align.resolve_word_align_providers("auto", runtime=fake_runtime)
    assert providers == ["CUDAExecutionProvider"]


def test_resolve_explicit_cpu_alias():
    assert word_align.resolve_word_align_providers("cpu") == ["CPUExecutionProvider"]


def test_resolve_explicit_cuda_alias():
    assert word_align.resolve_word_align_providers("cuda") == ["CUDAExecutionProvider"]


def test_resolve_explicit_ep_name_passthrough():
    assert word_align.resolve_word_align_providers("CPUExecutionProvider") == [
        "CPUExecutionProvider"
    ]


# ==================== build_alignment_session ====================


def test_build_session_missing_model_raises(tmp_path):
    missing = str(tmp_path / "nope.onnx")
    with pytest.raises(FileNotFoundError, match="word_align"):
        word_align.build_alignment_session(missing, ["CPUExecutionProvider"])


def test_build_session_happy(tmp_path):
    model = tmp_path / "mms.onnx"
    model.write_bytes(b"fake")
    fake_ort = types.SimpleNamespace(InferenceSession=MagicMock(return_value="SESSION"))
    with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
        sess = word_align.build_alignment_session(str(model), ["CPUExecutionProvider"])
    assert sess == "SESSION"
    fake_ort.InferenceSession.assert_called_once_with(
        str(model), providers=["CPUExecutionProvider"]
    )


# ==================== align_window ====================


def _patch_ctc(monkeypatch, word_results):
    """patch ctc_forced_aligner 五步函数, 让 align_window 返回 word_results."""
    import ctc_forced_aligner as ctc

    monkeypatch.setattr(ctc, "generate_emissions", lambda *a, **k: ("EMIT", 20))
    monkeypatch.setattr(
        ctc, "preprocess_text", lambda *a, **k: (["t"], ["text"])
    )
    monkeypatch.setattr(ctc, "get_alignments", lambda *a, **k: ("SEG", "SCORES", "blk"))
    monkeypatch.setattr(ctc, "get_spans", lambda *a, **k: ["SPAN"])
    monkeypatch.setattr(ctc, "postprocess_results", lambda *a, **k: word_results)


def test_align_window_happy(monkeypatch):
    _patch_ctc(
        monkeypatch,
        [
            {"text": "你", "start": 0.0, "end": 0.3, "score": -1.0},
            {"text": "好", "start": 0.3, "end": 0.6, "score": -1.2},
        ],
    )
    import numpy as np

    words = word_align.align_window(
        np.zeros(16000, dtype=np.float32),
        "你好",
        session=MagicMock(),
        tokenizer=MagicMock(),
        language="chi",
        batch_size=16,
    )
    assert [w["text"] for w in words] == ["你", "好"]
    assert words[0]["start"] == 0.0


def test_align_window_empty_text_returns_empty():
    import numpy as np

    words = word_align.align_window(
        np.zeros(1600, dtype=np.float32),
        "   ",
        session=MagicMock(),
        tokenizer=MagicMock(),
        language="chi",
        batch_size=16,
    )
    assert words == []


# ==================== align_chunks ====================


class _Chunk:
    def __init__(self, text, start, end):
        self.text = text
        self.start = start
        self.end = end


def test_align_chunks_offsets_by_chunk_start(monkeypatch):
    import numpy as np

    # align_window 被打桩成: 返回相对窗口 0.0-0.5 的一个词
    def fake_align_window(audio, text, **kw):
        return [{"text": text, "start": 0.0, "end": 0.5, "score": -1.0}]

    monkeypatch.setattr(word_align, "align_window", fake_align_window)

    chunks = [_Chunk("甲", 0.0, 10.0), _Chunk("乙", 10.0, 20.0)]
    audio = np.zeros(20 * 16000, dtype=np.float32)
    words, stats = word_align.align_chunks(
        audio, chunks, session=MagicMock(), tokenizer=MagicMock(),
        language="chi", batch_size=16,
    )
    # 第二 chunk 的词被 offset +10.0
    assert words[0]["start"] == 0.0
    assert words[1]["start"] == 10.0
    assert words[1]["end"] == 10.5
    assert stats["total_windows"] == 2
    assert stats["failed_windows"] == 0
    assert stats["total_words"] == 2


def test_align_chunks_one_window_fails_still_returns_others(monkeypatch):
    import numpy as np

    def fake_align_window(audio, text, **kw):
        if text == "炸":
            raise RuntimeError("trellis OOM")
        return [{"text": text, "start": 0.0, "end": 0.5, "score": -1.0}]

    monkeypatch.setattr(word_align, "align_window", fake_align_window)

    chunks = [_Chunk("好", 0.0, 10.0), _Chunk("炸", 10.0, 20.0), _Chunk("末", 20.0, 30.0)]
    audio = np.zeros(30 * 16000, dtype=np.float32)
    words, stats = word_align.align_chunks(
        audio, chunks, session=MagicMock(), tokenizer=MagicMock(),
        language="chi", batch_size=16,
    )
    assert [w["text"] for w in words] == ["好", "末"]
    assert stats["total_windows"] == 3
    assert stats["failed_windows"] == 1
    assert stats["failures"][0]["index"] == 1
    assert "trellis OOM" in stats["failures"][0]["reason"]


# ==================== WordAligner 单例 ====================


def test_word_aligner_lazy_builds_session_once(monkeypatch):
    import numpy as np

    build_calls = {"n": 0}

    def fake_build(model_path, providers):
        build_calls["n"] += 1
        return MagicMock()

    monkeypatch.setattr(word_align, "build_alignment_session", fake_build)
    monkeypatch.setattr(
        word_align, "align_chunks",
        lambda *a, **k: ([], {"total_windows": 0, "failed_windows": 0, "total_words": 0, "failures": []}),
    )
    # tokenizer build 也打桩, 避免真 import
    monkeypatch.setattr(word_align, "_build_tokenizer", lambda: MagicMock())

    aligner = word_align.WordAligner(
        model_path="/tmp/x.onnx", provider="cpu", language="chi", batch_size=16
    )
    audio = np.zeros(1600, dtype=np.float32)
    aligner.align_chunks(audio, [])
    aligner.align_chunks(audio, [])
    assert build_calls["n"] == 1
