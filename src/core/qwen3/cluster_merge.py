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


def merge_main_high_conf(
    centroids: dict,
    shares: dict,
    main_set: list,
    merge_threshold: float = 0.78,
) -> tuple[dict, list, list]:
    """两个 main cluster cos ≥ threshold 合并, dominant (share 大) 吃 recessive.

    多轮直到无 merge.

    Args:
        centroids: dict[sp -> np.ndarray (L2 normalized)].
        shares: dict[sp -> float], 用于决定 dominant.
        main_set: list of speaker keys that are "main".
        merge_threshold: cosine 阈值, 默认 0.78.

    Returns:
        (mapping_updates, remaining_mains, log)
        mapping_updates: dict[recessive_sp -> dominant_sp]
        remaining_mains: 合并后剩余的 main speakers.
        log: list of {action, from, to, sim}.
    """
    mapping_updates: dict = {}
    remaining_mains = list(main_set)
    log: list = []
    while True:
        merged_any = False
        for i, sp_i in enumerate(remaining_mains):
            for sp_j in remaining_mains[i + 1:]:
                sim = cosine(centroids.get(sp_i), centroids.get(sp_j))
                if sim >= merge_threshold:
                    dom = sp_i if shares.get(sp_i, 0) >= shares.get(sp_j, 0) else sp_j
                    rec = sp_j if dom == sp_i else sp_i
                    # 透传: 已被 map 走的 recessive, 要继续 map 到新 dom
                    for k, v in list(mapping_updates.items()):
                        if v == rec:
                            mapping_updates[k] = dom
                    mapping_updates[rec] = dom
                    log.append({
                        "action": "main_merged_high_conf",
                        "from": rec,
                        "to": dom,
                        "sim": round(sim, 4),
                    })
                    remaining_mains.remove(rec)
                    merged_any = True
                    break
            if merged_any:
                break
        if not merged_any:
            break
    return mapping_updates, remaining_mains, log


def merge_minor_to_main(
    centroids: dict,
    shares: dict,
    main_set: list,
    minor_set: list,
    relabel_threshold: float = 0.55,
) -> tuple[dict, list]:
    """每个 minor cluster -> 最近 main (cos ≥ relabel_threshold).

    Args:
        centroids, shares: 同 merge_main_high_conf.
        main_set: 主 cluster keys.
        minor_set: 次 cluster keys (待分配).
        relabel_threshold: cosine 阈值, 默认 0.55.

    Returns:
        (mapping_updates, log)
        mapping_updates: dict[minor_sp -> main_sp].
        log: list of {action, from, to, sim, share}.
    """
    mapping_updates: dict = {}
    log: list = []
    for sp in minor_set:
        if sp not in centroids:
            log.append({"action": "skip_minor_no_centroid", "from": sp})
            continue
        sims = [(m, cosine(centroids[sp], centroids[m])) for m in main_set if m in centroids]
        sims.sort(key=lambda kv: -kv[1])
        best_main, best_sim = sims[0] if sims else (None, -1.0)
        if best_main is not None and best_sim >= relabel_threshold:
            mapping_updates[sp] = best_main
            log.append({
                "action": "minor_to_main",
                "from": sp,
                "to": best_main,
                "sim": round(best_sim, 4),
                "share": round(shares.get(sp, 0.0), 4),
            })
        else:
            log.append({
                "action": "minor_kept_isolated",
                "from": sp,
                "best_main": best_main,
                "best_sim": round(best_sim, 4) if best_main is not None else None,
                "share": round(shares.get(sp, 0.0), 4),
            })
    return mapping_updates, log


def merge_dominant_mode(
    centroids: dict,
    current_shares: dict,
    remaining_mains: list,
    dominant_share: float = 0.6,
    dominant_merge_threshold: float = 0.6,
) -> tuple[dict, list]:
    """dominant 模式: 合并后仍存在 share >= dominant_share 的 cluster,
    用更低阈值 (dominant_merge_threshold) 把其他 main 合到 dominant.

    适合单人/强主导场景, 避免声纹漂移让单人音频聚成多 cluster.

    Args:
        centroids: dict[sp -> np.ndarray (L2 normalized)].
        current_shares: 合并后基于当前 mapping 重算的 share.
        remaining_mains: 上一步剩余的 main speakers.
        dominant_share: 触发 dominant 模式的 share 阈值, 默认 0.6.
        dominant_merge_threshold: dominant 模式下合并阈值, 默认 0.6.

    Returns:
        (mapping_updates, log)
        mapping_updates: dict[other_main_sp -> dominant_sp].
    """
    mapping_updates: dict = {}
    log: list = []
    if not remaining_mains or not current_shares:
        return mapping_updates, log
    dom_sp = max(current_shares.items(), key=lambda kv: kv[1])[0]
    dom_share = current_shares[dom_sp]
    if dom_share < dominant_share:
        return mapping_updates, log
    for sp in list(remaining_mains):
        if sp == dom_sp:
            continue
        sim = cosine(centroids.get(sp), centroids.get(dom_sp))
        if sim >= dominant_merge_threshold:
            mapping_updates[sp] = dom_sp
            log.append({
                "action": "main_merged_dominant_mode",
                "from": sp,
                "to": dom_sp,
                "sim": round(sim, 4),
                "dom_share": round(dom_share, 3),
            })
    return mapping_updates, log
