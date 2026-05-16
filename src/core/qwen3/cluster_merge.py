"""
Qwen3-Diarize cluster centroid merge (PR3).

按 TDD 红绿循环逐步实现. Sherpa embedding extractor 在入口函数
apply_cluster_centroid_merge 才需要, 内部函数都是纯 numpy/字典操作.
"""
from __future__ import annotations

import numpy as np


def cosine(a, b) -> float:
    """numpy 向量余弦相似度.

    Args:
        a, b: 同形状一维向量, 或 None / 零向量.

    Returns:
        余弦值 [-1, 1]. None 或 零向量视为"完全不相似" -> -1.0 (PoC 语义).
    """
    if a is None or b is None:
        return -1.0
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return -1.0
    dot = float(np.dot(a, b))
    return dot / (norm_a * norm_b)
