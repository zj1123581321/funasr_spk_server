"""
src/core/qwen3/cluster_merge.py 单元测试 (PR3 cluster centroid merge).

严格 TDD: 每个测试先写, 看红, 最小实现, 看绿. 不复用 PoC 单测.

Sherpa embedding extractor 用 mock fixture, 不依赖真模型.
"""
from __future__ import annotations

import math

import numpy as np


class TestCosine:
    """cosine: numpy 向量余弦相似度."""

    def test_cosine_identical_vectors_is_1(self) -> None:
        """同方向同长度向量, cos=1."""
        from src.core.qwen3.cluster_merge import cosine

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert math.isclose(cosine(a, b), 1.0, abs_tol=1e-9)

    def test_cosine_opposite_vectors_is_minus_1(self) -> None:
        """反方向向量, cos=-1."""
        from src.core.qwen3.cluster_merge import cosine

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert math.isclose(cosine(a, b), -1.0, abs_tol=1e-9)

    def test_cosine_orthogonal_vectors_is_0(self) -> None:
        """正交向量, cos=0."""
        from src.core.qwen3.cluster_merge import cosine

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert math.isclose(cosine(a, b), 0.0, abs_tol=1e-9)

    def test_cosine_null_or_zero_input_returns_minus_1(self) -> None:
        """None / 零向量 视为 "完全不相似" 返回 -1.0 (PoC 语义)."""
        from src.core.qwen3.cluster_merge import cosine

        a = np.array([1.0, 0.0])
        zero = np.array([0.0, 0.0])
        assert cosine(None, a) == -1.0
        assert cosine(a, None) == -1.0
        assert cosine(zero, a) == -1.0
        assert cosine(a, zero) == -1.0


def _seg(start: float, end: float) -> dict:
    """测试辅助: build_centroids 只关心 start/end."""
    return {"start": start, "end": end}


class TestBuildCentroids:
    """build_centroids: 用 mock extractor_fn 算各 cluster centroid."""

    def test_single_cluster_returns_normalized_centroid(self) -> None:
        """1 speaker 多段, centroid = normalize(mean(embeddings))."""
        from src.core.qwen3.cluster_merge import build_centroids

        emb_map = {
            (0.0, 2.0): np.array([3.0, 0.0, 0.0]),
            (2.0, 4.0): np.array([1.0, 0.0, 0.0]),
        }

        def extractor_fn(audio, start, end):
            return emb_map.get((start, end))

        audio_16k = np.zeros(64000)  # dummy
        segs_by_spk = {"0": [_seg(0.0, 2.0), _seg(2.0, 4.0)]}
        centroids = build_centroids(extractor_fn, audio_16k, segs_by_spk)

        assert set(centroids.keys()) == {"0"}
        c = centroids["0"]
        # mean = [2.0, 0.0, 0.0], normalize -> [1.0, 0.0, 0.0]
        assert np.allclose(c, np.array([1.0, 0.0, 0.0]))

    def test_multiple_clusters_return_separate_centroids(self) -> None:
        """多 speaker, 每个独立 centroid."""
        from src.core.qwen3.cluster_merge import build_centroids

        emb_map = {
            (0.0, 2.0): np.array([1.0, 0.0]),
            (2.0, 4.0): np.array([0.0, 1.0]),
        }

        def extractor_fn(audio, start, end):
            return emb_map.get((start, end))

        segs_by_spk = {
            "0": [_seg(0.0, 2.0)],
            "1": [_seg(2.0, 4.0)],
        }
        centroids = build_centroids(extractor_fn, np.zeros(64000), segs_by_spk)
        assert set(centroids.keys()) == {"0", "1"}
        assert np.allclose(centroids["0"], np.array([1.0, 0.0]))
        assert np.allclose(centroids["1"], np.array([0.0, 1.0]))

    def test_speaker_with_all_none_embeddings_is_skipped(self) -> None:
        """所有段都拿不到 embedding 的 speaker, 不出现在 centroids 中."""
        from src.core.qwen3.cluster_merge import build_centroids

        def extractor_fn(audio, start, end):
            # speaker "1" 的段总是返回 None (e.g. 段太短)
            if start >= 5.0:
                return None
            return np.array([1.0, 0.0])

        segs_by_spk = {
            "0": [_seg(0.0, 2.0)],
            "1": [_seg(5.0, 5.5), _seg(6.0, 6.3)],  # 都返回 None
        }
        centroids = build_centroids(extractor_fn, np.zeros(64000), segs_by_spk)
        assert set(centroids.keys()) == {"0"}
