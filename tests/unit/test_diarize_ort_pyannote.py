"""ORT 直 wrap diarize backend — pyannote-segmentation-3.0 sliding window 单测.

覆盖 sliding window 的 3 个内核组件:
1. `_iter_audio_chunks` — 任意长度 audio 切成 10s 重叠 chunks (pad 短 audio / 空 audio)
2. `_powerset_to_multilabel` — pyannote powerset class (7 class) → multi-label (3 spk)
3. `run_segmentation_chunks` — 逐 chunk 独立 argmax (sherpa parity: 不跨 chunk 平均
   logits, speaker slot 是 chunk 局部的)

用 mock ORT session, 不真实跑 onnxruntime, 不需要模型文件.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np


# ==================== _iter_audio_chunks ====================


def test_iter_audio_chunks_60s_audio_default_step_yields_51():
    """60s @ 16k = 960000 samples, chunk=160000, step=16000 → (60-10)/1 + 1 = 51 chunks."""
    from src.core.qwen3.diarize_ort import _iter_audio_chunks

    audio = np.zeros(60 * 16000, dtype=np.float32)
    chunks = list(_iter_audio_chunks(audio, chunk_samples=160000, step_samples=16000))
    assert len(chunks) == 51
    for start, chunk in chunks:
        assert chunk.shape == (160000,)


def test_iter_audio_chunks_pads_short_audio_to_chunk_size():
    """3s audio < chunk_samples 应 yield 1 chunk, 后面 pad 0."""
    from src.core.qwen3.diarize_ort import _iter_audio_chunks

    audio = np.ones(3 * 16000, dtype=np.float32)
    chunks = list(_iter_audio_chunks(audio, chunk_samples=160000, step_samples=16000))
    assert len(chunks) == 1
    start, chunk = chunks[0]
    assert start == 0
    assert chunk.shape == (160000,)
    # 前 48000 是原 audio (1.0), 后面 pad 0
    np.testing.assert_array_equal(chunk[:48000], 1.0)
    np.testing.assert_array_equal(chunk[48000:], 0.0)


def test_iter_audio_chunks_empty_audio_yields_nothing():
    from src.core.qwen3.diarize_ort import _iter_audio_chunks

    chunks = list(_iter_audio_chunks(np.array([], dtype=np.float32)))
    assert chunks == []


def test_iter_audio_chunks_last_chunk_padded_when_exceeds_audio():
    """exactly chunk_samples 长 audio → 1 chunk, 不再 advance."""
    from src.core.qwen3.diarize_ort import _iter_audio_chunks

    audio = np.ones(160000, dtype=np.float32)
    chunks = list(_iter_audio_chunks(audio, chunk_samples=160000, step_samples=16000))
    assert len(chunks) == 1


# ==================== _powerset_to_multilabel ====================


def test_powerset_to_multilabel_silence_class_returns_zero_vector():
    """class 0 = silence → (0, 0, 0)."""
    from src.core.qwen3.diarize_ort import _powerset_to_multilabel

    logits = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    out = _powerset_to_multilabel(logits)
    np.testing.assert_array_equal(out, [[0, 0, 0]])


def test_powerset_to_multilabel_single_speaker_classes():
    """class 1/2/3 分别对应 spk0/spk1/spk2 单独活动."""
    from src.core.qwen3.diarize_ort import _powerset_to_multilabel

    cases = [
        ([0, 1, 0, 0, 0, 0, 0], [1, 0, 0]),
        ([0, 0, 1, 0, 0, 0, 0], [0, 1, 0]),
        ([0, 0, 0, 1, 0, 0, 0], [0, 0, 1]),
    ]
    for one_hot, expected in cases:
        out = _powerset_to_multilabel(np.array([one_hot], dtype=np.float32))
        np.testing.assert_array_equal(out, [expected])


def test_powerset_to_multilabel_overlapping_speakers():
    """class 4/5/6 是 2-speaker 同时活动."""
    from src.core.qwen3.diarize_ort import _powerset_to_multilabel

    cases = [
        ([0, 0, 0, 0, 1, 0, 0], [1, 1, 0]),  # spk0+1
        ([0, 0, 0, 0, 0, 1, 0], [1, 0, 1]),  # spk0+2
        ([0, 0, 0, 0, 0, 0, 1], [0, 1, 1]),  # spk1+2
    ]
    for one_hot, expected in cases:
        out = _powerset_to_multilabel(np.array([one_hot], dtype=np.float32))
        np.testing.assert_array_equal(out, [expected])


def test_powerset_to_multilabel_batched_input_preserves_leading_dim():
    """支持 (B, T, 7) 输入 → (B, T, 3) 输出, 不破 batch axis."""
    from src.core.qwen3.diarize_ort import _powerset_to_multilabel

    logits = np.zeros((2, 5, 7), dtype=np.float32)
    logits[..., 0] = 1.0  # 全 silence
    out = _powerset_to_multilabel(logits)
    assert out.shape == (2, 5, 3)
    np.testing.assert_array_equal(out, 0)


# ==================== run_segmentation_chunks (mock ORT) ====================


def _make_mock_session(class_idx: int = 0):
    """构造一个 mock ORT session, 每次 run 返回全 one-hot 在 class_idx 上 (1, 589, 7).

    Class 0 = silence; class 1/2/3 = single speaker; class 4/5/6 = overlapping.
    """
    sess = MagicMock()
    fake_input = MagicMock()
    fake_input.name = "x"
    sess.get_inputs.return_value = [fake_input]

    def fake_run(_outputs, feed):
        # 验证 feed 是 (1, 1, 160000) float32
        x = feed["x"]
        assert x.shape == (1, 1, 160000), f"unexpected feed shape {x.shape}"
        out = np.zeros((1, 589, 7), dtype=np.float32)
        out[..., class_idx] = 1.0
        return [out]

    sess.run.side_effect = fake_run
    return sess


def test_run_segmentation_chunks_all_silence_returns_zero_labels():
    from src.core.qwen3.diarize_ort import run_segmentation_chunks

    sess = _make_mock_session(class_idx=0)
    audio = np.zeros(30 * 16000, dtype=np.float32)
    starts, labels = run_segmentation_chunks(audio, sess)
    # 30s → (30-10)/1 + 1 = 21 chunks
    assert len(starts) == len(labels) == 21
    for label in labels:
        assert label.shape == (589, 3)
        np.testing.assert_array_equal(label, 0)


def test_run_segmentation_chunks_single_speaker_per_chunk_mask():
    """每个 chunk 都 class=1 (slot0) → 每 chunk 输出 (589, [1,0,0])."""
    from src.core.qwen3.diarize_ort import run_segmentation_chunks

    sess = _make_mock_session(class_idx=1)
    audio = np.zeros(15 * 16000, dtype=np.float32)
    starts, labels = run_segmentation_chunks(audio, sess)
    assert starts[0] == 0 and starts[1] == 16000
    for label in labels:
        assert (label[:, 0] == 1).all()
        assert (label[:, 1] == 0).all()
        assert (label[:, 2] == 0).all()


def test_run_segmentation_chunks_keeps_chunks_independent():
    """关键不变量: 各 chunk argmax 互不影响 (绝不跨 chunk 平均 logits).

    chunk0 标 slot0, 其余 chunk 标 slot1 — 两个 chunk 的 label 必须各自干净,
    不出现平均/融合后的混叠.
    """
    from src.core.qwen3.diarize_ort import run_segmentation_chunks

    sess = MagicMock()
    fake_input = MagicMock()
    fake_input.name = "x"
    sess.get_inputs.return_value = [fake_input]
    call = {"i": 0}

    def fake_run(_outputs, feed):
        i = call["i"]
        call["i"] += 1
        out = np.zeros((1, 589, 7), dtype=np.float32)
        out[..., 1 if i == 0 else 2] = 1.0  # chunk0 → slot0, 其余 → slot1
        return [out]

    sess.run.side_effect = fake_run

    audio = np.zeros(12 * 16000, dtype=np.float32)
    _starts, labels = run_segmentation_chunks(audio, sess)
    assert len(labels) >= 2
    np.testing.assert_array_equal(labels[0][:, 0], 1)
    np.testing.assert_array_equal(labels[0][:, 1], 0)
    np.testing.assert_array_equal(labels[1][:, 0], 0)
    np.testing.assert_array_equal(labels[1][:, 1], 1)
