"""Speaker cluster centroid merge — 修复多人场景过度聚类.

sherpa diarize (cluster_threshold=0.9) 在多人 (3+) 场景容易把同一个人聚成
多个 cluster + 把音乐/笑声等非语音聚成额外 cluster.

本脚本无 reference 依赖,用 speaker embedding centroid 做层级合并:

策略:
  1. 用每个 cluster 内最长的几个 segment 的 audio interval 算 embedding mean
     → cluster centroid
  2. 区分 "主 cluster" (share >= min_main_share, 默认 3%) 和
     "次要 cluster" (share < min_main_share)
  3. 对每个次要 cluster 找 cosine 最相似的主 cluster:
     - similarity >= relabel_threshold (默认 0.55) 就 relabel 到主 cluster
     - 否则保留 (可能是音乐/外部声源)
  4. 主 cluster 之间也尝试两两合并 (similarity >= merge_threshold, 默认 0.75)

输出:同 schema JSON, 含 cluster_merge_log summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sherpa_onnx
from src.core.config import config
from src.core.qwen3.diarize import _load_audio_mono_16k as _sf_load_16k


def _load_audio_mono_16k(audio_path: str) -> tuple[np.ndarray, int]:
    """Load audio to 16kHz mono float32. Tries soundfile first (wav/flac/ogg/mp3),
    falls back to librosa for m4a/aac/anything else.
    """
    try:
        return _sf_load_16k(audio_path)
    except Exception:
        import librosa
        audio, _sr = librosa.load(audio_path, sr=16000, mono=True)
        return audio.astype(np.float32), 16000


def _embedding_for_interval(extractor, audio: np.ndarray, start: float, end: float) -> np.ndarray | None:
    sr = 16000
    a = int(max(0.0, start) * sr)
    b = int(min(len(audio) / sr, end) * sr)
    if b - a < int(0.3 * sr):
        return None
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=sr, waveform=audio[a:b])
    stream.input_finished()
    if not extractor.is_ready(stream):
        return None
    emb = np.asarray(extractor.compute(stream), dtype=np.float32)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def build_extractor():
    q = config.qwen3
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=q.embedding_model,
        num_threads=4,
        provider=q.provider,
        debug=False,
    )
    if not cfg.validate():
        raise RuntimeError("embedding extractor config invalid")
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


def cluster_centroids(
    extractor,
    audio_16k: np.ndarray,
    segments_by_spk: dict[str, list[dict]],
    max_per_speaker: int = 30,
) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    for sp, segs in segments_by_spk.items():
        chosen = sorted(segs, key=lambda s: float(s["end"]) - float(s["start"]), reverse=True)[:max_per_speaker]
        embs = []
        for s in chosen:
            emb = _embedding_for_interval(extractor, audio_16k, float(s["start"]), float(s["end"]))
            if emb is not None:
                embs.append(emb)
        if not embs:
            continue
        c = np.mean(np.stack(embs), axis=0)
        n = np.linalg.norm(c)
        centroids[sp] = c / n if n > 0 else c
    return centroids


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json", type=Path)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--min-main-share", type=float, default=0.03,
                    help="占比 >= 这个值视为主 cluster, 默认 3%")
    ap.add_argument("--relabel-threshold", type=float, default=0.55,
                    help="次要 cluster -> 最近主 cluster 的 cosine 阈值, 否则不合并")
    ap.add_argument("--merge-threshold", type=float, default=0.78,
                    help="主 cluster 之间互相合并的 cosine 阈值 (高确定性)")
    ap.add_argument("--dominant-share", type=float, default=0.6,
                    help="如果最大 cluster 占比 >= 这个值, 启用 dominant 模式")
    ap.add_argument("--dominant-merge-threshold", type=float, default=0.6,
                    help="dominant 模式下, 其他 main 与 dominant 的合并阈值")
    ap.add_argument("--max-per-speaker", type=int, default=30,
                    help="每个 cluster 取最长前 N segments 算 centroid")
    args = ap.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    segments = payload["segments"]
    audio_duration = float(payload.get("summary", {}).get("audio_duration") or max(float(s["end"]) for s in segments))

    # Group by speaker
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for s in segments:
        by_spk[str(s["speaker"])].append(s)
    spk_dur = {sp: sum(float(s["end"]) - float(s["start"]) for s in segs) for sp, segs in by_spk.items()}
    total_dur = sum(spk_dur.values()) or 1.0
    spk_share = {sp: dur / total_dur for sp, dur in spk_dur.items()}

    main_set = sorted([sp for sp, sh in spk_share.items() if sh >= args.min_main_share],
                      key=lambda sp: -spk_share[sp])
    minor_set = [sp for sp in spk_share if sp not in main_set]
    print(f"[init] speakers={len(spk_share)}  main={main_set}  minor={minor_set}")

    if not main_set:
        print("[skip] no main cluster meets threshold")
        return

    # Load audio for embedding
    print("[load] audio for embedding ...")
    audio_16k, _ = _load_audio_mono_16k(str(args.audio))
    extractor = build_extractor()

    # Centroids for all clusters
    print("[centroid] compute per-cluster centroids ...")
    centroids = cluster_centroids(extractor, audio_16k, by_spk, max_per_speaker=args.max_per_speaker)

    # Build the mapping: each cluster -> final cluster
    mapping: dict[str, str] = {sp: sp for sp in by_spk}
    log: list[dict] = []

    # Step 1: minor -> nearest main (if similarity high enough)
    for sp in minor_set:
        if sp not in centroids:
            log.append({"action": "skip_minor_no_centroid", "from": sp})
            continue
        sims = [(main, cosine(centroids[sp], centroids[main])) for main in main_set if main in centroids]
        sims.sort(key=lambda kv: -kv[1])
        best_main, best_sim = sims[0] if sims else (None, -1.0)
        if best_main is not None and best_sim >= args.relabel_threshold:
            mapping[sp] = best_main
            log.append({"action": "minor_to_main", "from": sp, "to": best_main, "sim": round(best_sim, 4),
                        "share": round(spk_share[sp], 4)})
        else:
            log.append({"action": "minor_kept_isolated", "from": sp, "best_main": best_main,
                        "best_sim": round(best_sim, 4) if best_main else None,
                        "share": round(spk_share[sp], 4)})

    # Step 2: main-to-main high-confidence merge (sim >= merge_threshold, 默认 0.78)
    # 这一步是"明显同一人"的合并 — 适用所有场景
    remaining_mains = list(main_set)
    while True:
        merged_any = False
        for i, sp_i in enumerate(remaining_mains):
            for sp_j in remaining_mains[i+1:]:
                sim = cosine(centroids.get(sp_i), centroids.get(sp_j))
                if sim >= args.merge_threshold:
                    dominant = sp_i if spk_share[sp_i] >= spk_share[sp_j] else sp_j
                    recessive = sp_j if dominant == sp_i else sp_i
                    for k, v in list(mapping.items()):
                        if v == recessive:
                            mapping[k] = dominant
                    log.append({"action": "main_merged_high_conf", "from": recessive, "to": dominant,
                                "sim": round(sim, 4)})
                    remaining_mains.remove(recessive)
                    merged_any = True
                    break
            if merged_any:
                break
        if not merged_any:
            break

    # Step 3: dominant adaptive — 如果合并后仍有 dominant cluster (单人/强主导场景),
    # 用更低阈值把剩余 main 合到 dominant. 避免单人音频因声纹漂移聚成多 cluster.
    # 重新计算 share (基于当前 mapping)
    current_dur: dict[str, float] = {}
    for orig_sp, final_sp in mapping.items():
        current_dur[final_sp] = current_dur.get(final_sp, 0.0) + spk_dur.get(orig_sp, 0.0)
    current_total = sum(current_dur.values()) or 1.0
    current_share = {sp: dur / current_total for sp, dur in current_dur.items()}
    if remaining_mains and current_share:
        dom_sp = max(current_share.items(), key=lambda kv: kv[1])[0]
        dom_share = current_share[dom_sp]
        if dom_share >= args.dominant_share:
            # 把其他 main cluster (非 dom_sp) 与 dom_sp 比较
            for sp in list(remaining_mains):
                if sp == dom_sp:
                    continue
                sim = cosine(centroids.get(sp), centroids.get(dom_sp))
                if sim >= args.dominant_merge_threshold:
                    for k, v in list(mapping.items()):
                        if v == sp:
                            mapping[k] = dom_sp
                    log.append({"action": "main_merged_dominant_mode", "from": sp, "to": dom_sp,
                                "sim": round(sim, 4), "dom_share": round(dom_share, 3)})
                    remaining_mains.remove(sp)

    # Apply mapping
    final_segs = []
    for s in segments:
        s2 = dict(s)
        s2["speaker"] = mapping[str(s["speaker"])]
        final_segs.append(s2)

    final_spk = sorted({s["speaker"] for s in final_segs})
    print(f"[final] speakers={len(final_spk)}  set={final_spk}")
    for entry in log:
        print(f"  {entry}")

    # Build output
    new_payload = dict(payload)
    new_payload["segments"] = final_segs
    summary = dict(new_payload.get("summary") or {})
    summary["cluster_merge"] = {
        "source_json": str(args.input_json),
        "min_main_share": args.min_main_share,
        "relabel_threshold": args.relabel_threshold,
        "merge_threshold": args.merge_threshold,
        "input_speakers": list(by_spk.keys()),
        "final_speakers": final_spk,
        "mapping": mapping,
        "log": log,
    }
    new_payload["summary"] = summary

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(new_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {args.out_json}")


if __name__ == "__main__":
    main()
