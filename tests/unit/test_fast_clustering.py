"""Python FastClustering + TitaNet ORT embedding 单测.

FastClustering: sherpa-onnx 兼容 — cosine distance + average linkage agglomerative,
两种 cut 模式: 固定 num_clusters 或 threshold cut.

TitaNet ORT embedding: log-mel (commit 6 实现) + ORT inference + L2 normalize.
mock ORT session, 不真实加载模型.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


# ==================== fast_clustering ====================


def test_fast_clustering_empty_embeddings_returns_empty_labels():
    from src.core.qwen3.diarize_ort import fast_clustering

    labels = fast_clustering(np.zeros((0, 192), dtype=np.float32))
    assert labels.shape == (0,)


def test_fast_clustering_single_embedding_returns_single_zero_label():
    from src.core.qwen3.diarize_ort import fast_clustering

    emb = np.zeros((1, 192), dtype=np.float32)
    emb[0, 0] = 1.0
    labels = fast_clustering(emb)
    assert labels.tolist() == [0]


def test_fast_clustering_two_orthogonal_embeddings_returns_two_distinct_labels():
    """两正交 (cosine dist=1) embeddings → 距离 > threshold → 2 clusters."""
    from src.core.qwen3.diarize_ort import fast_clustering

    emb = np.zeros((2, 4), dtype=np.float32)
    emb[0, 0] = 1.0
    emb[1, 1] = 1.0
    labels = fast_clustering(emb, threshold=0.5)
    assert len(set(labels.tolist())) == 2


def test_fast_clustering_three_close_embeddings_collapse_to_one():
    """三 embedding 几乎同向 (cosine dist ~0) → 1 cluster."""
    from src.core.qwen3.diarize_ort import fast_clustering

    rng = np.random.RandomState(0)
    base = rng.randn(192).astype(np.float32)
    base /= np.linalg.norm(base)
    emb = np.stack([base, base + 0.01 * rng.randn(192), base + 0.01 * rng.randn(192)])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    labels = fast_clustering(emb.astype(np.float32), threshold=0.5)
    assert len(set(labels.tolist())) == 1


def test_fast_clustering_num_clusters_overrides_threshold():
    """显式 num_clusters=2 强制 2 簇, 即使 threshold 会判 4 簇."""
    from src.core.qwen3.diarize_ort import fast_clustering

    rng = np.random.RandomState(1)
    emb = rng.randn(8, 16).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = fast_clustering(emb, num_clusters=2)
    assert len(set(labels.tolist())) == 2


def test_fast_clustering_labels_are_zero_based_contiguous():
    """labels 应该是 0..K-1 连续 int, 不留间隙."""
    from src.core.qwen3.diarize_ort import fast_clustering

    rng = np.random.RandomState(2)
    emb = rng.randn(5, 8).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = fast_clustering(emb, num_clusters=3)
    s = set(labels.tolist())
    assert s == set(range(len(s)))


# ==================== compute_titanet_embedding ====================


def _make_titanet_session(out_dim: int = 192, expects_length: bool = False):
    sess = MagicMock()
    in_audio = MagicMock()
    in_audio.name = "audio_signal"
    inputs = [in_audio]
    if expects_length:
        in_len = MagicMock()
        in_len.name = "length"
        inputs.append(in_len)
    sess.get_inputs.return_value = inputs

    def fake_run(_, feed):
        mel = feed["audio_signal"]
        assert mel.dtype == np.float32
        assert mel.ndim == 3
        assert mel.shape[1] == 80
        # 输出 (1, out_dim) 全 1
        return [np.ones((1, out_dim), dtype=np.float32)]

    sess.run.side_effect = fake_run
    return sess


def test_compute_titanet_embedding_returns_192_dim_normalized():
    from src.core.qwen3.diarize_ort import compute_titanet_embedding

    sess = _make_titanet_session(out_dim=192)
    audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
    emb = compute_titanet_embedding(audio, sess)
    assert emb.shape == (192,)
    assert emb.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(emb), 1.0, atol=1e-5)


def test_compute_titanet_embedding_handles_length_input():
    """sherpa nemo-titanet ONNX 有 2 inputs (mel + length), 我们要喂 length tensor."""
    from src.core.qwen3.diarize_ort import compute_titanet_embedding

    sess = _make_titanet_session(expects_length=True)

    captured: dict[str, np.ndarray] = {}

    def capture_run(_, feed):
        captured.update(feed)
        return [np.ones((1, 192), dtype=np.float32)]

    sess.run.side_effect = capture_run
    audio = np.random.RandomState(1).randn(8000).astype(np.float32) * 0.1
    _ = compute_titanet_embedding(audio, sess)
    # 至少要喂 audio_signal + length
    assert "audio_signal" in captured
    assert "length" in captured
    assert captured["length"].dtype == np.int64
    # length 值 = mel T_mel
    assert captured["length"].shape == (1,)
    assert captured["length"][0] == captured["audio_signal"].shape[-1]


def test_compute_titanet_embedding_zero_audio_returns_zero_safe():
    """全 0 audio: 不 NaN, embedding 也能合理 normalize (或者全 0)."""
    from src.core.qwen3.diarize_ort import compute_titanet_embedding

    sess = _make_titanet_session(out_dim=192)
    audio = np.zeros(16000, dtype=np.float32)
    emb = compute_titanet_embedding(audio, sess)
    assert emb.shape == (192,)
    assert np.isfinite(emb).all()
