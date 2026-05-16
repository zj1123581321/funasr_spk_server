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
