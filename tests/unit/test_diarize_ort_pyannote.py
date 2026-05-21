"""ORT 直 wrap diarize backend — pyannote-segmentation-3.0 sliding window 单测.

覆盖 sliding window pipeline 的 3 个内核组件:
1. `_iter_audio_chunks` — 任意长度 audio 切成 10s 重叠 chunks (pad 短 audio / 空 audio)
2. `_powerset_to_multilabel` — pyannote powerset class (7 class) → multi-label (3 spk)
3. `_aggregate_chunk_outputs` — Whisper-style 加权融合, 重叠帧除以 count

末尾用 mock ORT session 串完整 pipeline, 验证 shape + 全 silence 一致性.
不真实跑 onnxruntime, 不需要模型文件.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


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


# ==================== _aggregate_chunk_outputs ====================


def test_aggregate_chunk_outputs_single_chunk_copies_through():
    from src.core.qwen3.diarize_ort import _aggregate_chunk_outputs

    out = np.ones((589, 7), dtype=np.float32) * 1.5
    agg = _aggregate_chunk_outputs(
        chunk_starts=[0],
        chunk_outputs=[out],
        audio_samples=10 * 16000,
    )
    assert agg.shape[1] == 7
    np.testing.assert_allclose(agg[:589], 1.5)


def test_aggregate_chunk_outputs_overlapping_chunks_averages_overlap_region():
    """两 chunks 半重叠 → 重叠帧应是 (a+b)/2."""
    from src.core.qwen3.diarize_ort import _aggregate_chunk_outputs

    K = 7
    out1 = np.ones((589, K), dtype=np.float32) * 1.0
    out2 = np.ones((589, K), dtype=np.float32) * 3.0
    agg = _aggregate_chunk_outputs(
        chunk_starts=[0, 16000],  # chunk2 start at 1s; overlap 1-10s
        chunk_outputs=[out1, out2],
        audio_samples=11 * 16000,
    )
    # 第 0 帧 (0s 附近) 只 chunk1 → 1.0
    np.testing.assert_allclose(agg[0], 1.0, atol=1e-6)
    # 第 ~300 帧 (~5s) 两 chunk 都覆盖 → (1+3)/2 = 2.0
    mid_frame = 300
    np.testing.assert_allclose(agg[mid_frame], 2.0, atol=1e-6)


def test_aggregate_chunk_outputs_empty_raises():
    from src.core.qwen3.diarize_ort import _aggregate_chunk_outputs

    with pytest.raises(ValueError):
        _aggregate_chunk_outputs(
            chunk_starts=[], chunk_outputs=[], audio_samples=16000
        )


# ==================== pyannote_segmentation_pipeline (mock ORT) ====================


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


def test_pyannote_pipeline_all_silence_returns_zero_activity():
    from src.core.qwen3.diarize_ort import pyannote_segmentation_pipeline

    sess = _make_mock_session(class_idx=0)
    audio = np.zeros(30 * 16000, dtype=np.float32)
    activity = pyannote_segmentation_pipeline(audio, sess)
    assert activity.shape[1] == 3
    np.testing.assert_array_equal(activity, 0)


def test_pyannote_pipeline_single_speaker_returns_correct_mask():
    """每个 chunk 都 class=1 (spk0) → 整段输出 (T, [1,0,0])."""
    from src.core.qwen3.diarize_ort import pyannote_segmentation_pipeline

    sess = _make_mock_session(class_idx=1)
    audio = np.zeros(15 * 16000, dtype=np.float32)
    activity = pyannote_segmentation_pipeline(audio, sess)
    assert activity.shape[1] == 3
    # 全 spk0 → 第 0 列全 1, 其余列全 0
    assert (activity[:, 0] == 1).all()
    assert (activity[:, 1] == 0).all()
    assert (activity[:, 2] == 0).all()
