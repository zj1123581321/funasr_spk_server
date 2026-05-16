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


def build_centroids(
    extractor_fn,
    audio_16k: np.ndarray,
    segments_by_spk: dict,
    max_per_speaker: int = 30,
) -> dict:
    """对每个 speaker, 抽取多段 embedding 求 L2 归一化的 centroid.

    Args:
        extractor_fn: callable (audio, start, end) -> np.ndarray or None.
        audio_16k: 16kHz mono numpy array.
        segments_by_spk: dict[str -> list[{start, end}]].
        max_per_speaker: 每 speaker 最多取多少段 (取最长的 N 段).

    Returns:
        dict[str -> np.ndarray (L2 normalized)]. 跳过没法算 embedding 的 speaker.
    """
    centroids: dict = {}
    for sp, segs in segments_by_spk.items():
        chosen = sorted(segs, key=lambda s: float(s["end"]) - float(s["start"]), reverse=True)[:max_per_speaker]
        embs = []
        for s in chosen:
            emb = extractor_fn(audio_16k, float(s["start"]), float(s["end"]))
            if emb is not None:
                embs.append(emb)
        if not embs:
            continue
        c = np.mean(np.stack(embs), axis=0)
        n = float(np.linalg.norm(c))
        centroids[sp] = c / n if n > 0 else c
    return centroids
