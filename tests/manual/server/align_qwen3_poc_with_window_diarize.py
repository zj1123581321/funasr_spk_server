"""
Experimental Qwen3-ForcedAligner + per-window diarization re-merge.

Unlike align_qwen3_poc_segments.py, this script can rerun sherpa diarization on
selected 40s windows, then uses Qwen3-ForcedAligner char timestamps to assign
text to those fresh speaker turns.  It is intended to probe whether finer text
timestamps + better turn boundaries can improve speaker accuracy without
rerunning ASR.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import sherpa_onnx

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import config
from src.core.qwen3.diarize import run_diarization
from src.core.qwen3.merge import Segment, segments_to_srt
from tests.manual.server.align_qwen3_poc_segments import load_audio_window, choose_speaker, _speaker_id


def infer_global_roles(segments: list[dict]) -> tuple[str, str]:
    chars = defaultdict(int)
    for s in segments:
        chars[str(s["speaker"])] += len(str(s.get("text", "")))
    ordered = sorted(chars.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[0][0], ordered[-1][0]


def group_segments(segments: list[dict], window_sec: float, start_sec: float | None, end_sec: float | None):
    groups = {}
    for idx, seg in enumerate(segments):
        s = float(seg["start"]); e = float(seg["end"])
        if start_sec is not None and e <= start_sec: continue
        if end_sec is not None and s >= end_sec: continue
        wi = int(math.floor(max(0, s) / window_sec))
        groups.setdefault(wi, []).append({**seg, "_idx": idx})
    return groups


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else -1.0


def _embedding_for_interval(extractor, audio_16k: np.ndarray, start: float, end: float, sr: int = 16000) -> np.ndarray | None:
    s = max(0, int(start * sr)); e = min(len(audio_16k), int(end * sr))
    if e - s < int(0.5 * sr):
        return None
    stream = extractor.create_stream()
    stream.accept_waveform(sr, audio_16k[s:e])
    stream.input_finished()
    if not extractor.is_ready(stream):
        return None
    emb = np.asarray(extractor.compute(stream), dtype=np.float32)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def build_global_centroids(audio_path: Path, source_segments: list[dict], max_segments_per_speaker: int = 80) -> tuple[dict[str, np.ndarray], object, np.ndarray]:
    from src.core.qwen3.diarize import _load_audio_mono_16k

    q = config.qwen3
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=q.embedding_model,
        num_threads=4,
        provider=q.provider,
    )
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    audio_16k, _sr = _load_audio_mono_16k(str(audio_path))

    by_sp: dict[str, list[dict]] = defaultdict(list)
    for seg in source_segments:
        dur = float(seg["end"]) - float(seg["start"])
        # Prefer longer, less boundary-sensitive examples for centroids.
        if dur >= 4.0 and len(str(seg.get("text", ""))) >= 20:
            by_sp[str(seg["speaker"])].append(seg)

    centroids: dict[str, np.ndarray] = {}
    for sp, segs in by_sp.items():
        chosen = sorted(segs, key=lambda s: float(s["end"]) - float(s["start"]), reverse=True)[:max_segments_per_speaker]
        embs = []
        for seg in chosen:
            emb = _embedding_for_interval(extractor, audio_16k, float(seg["start"]), float(seg["end"]))
            if emb is not None:
                embs.append(emb)
        if embs:
            c = np.mean(np.stack(embs), axis=0)
            n = np.linalg.norm(c)
            centroids[sp] = c / n if n > 0 else c
    return centroids, extractor, audio_16k


def map_local_speakers(
    turns: list[dict],
    source_intervals: list[dict],
    dominant_label: str,
    minority_label: str,
    *,
    mapping_strategy: str = "overlap",
    centroids: dict[str, np.ndarray] | None = None,
    extractor=None,
    audio_16k: np.ndarray | None = None,
) -> list[dict]:
    """Map per-window diarization cluster ids back to global Speaker labels.

    Prefer overlap with the existing global speaker intervals when both
    speakers are present.  If the current output collapsed a 40s window to a
    single speaker, keep that speaker for the longest local cluster and map the
    other cluster to the other global speaker; this is the important repair case
    for swallowed short questions.
    """
    totals = defaultdict(float)
    for t in turns:
        totals[int(t["speaker"])] += float(t["end"]) - float(t["start"])
    if not totals:
        return []

    global_labels = sorted({str(s["speaker"]) for s in source_intervals})
    all_labels = [dominant_label, minority_label]
    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    mapping: dict[int, str] = {}

    if mapping_strategy == "centroid" and centroids and extractor is not None and audio_16k is not None:
        local_embs: dict[int, list[np.ndarray]] = defaultdict(list)
        for t in turns:
            dur = float(t["end"]) - float(t["start"])
            if dur < 0.8:
                continue
            emb = _embedding_for_interval(extractor, audio_16k, float(t["start"]), float(t["end"]))
            if emb is not None:
                local_embs[int(t["speaker"])].append(emb)
        scores: dict[int, dict[str, float]] = defaultdict(dict)
        for sp, embs in local_embs.items():
            c = np.mean(np.stack(embs), axis=0)
            n = np.linalg.norm(c)
            c = c / n if n > 0 else c
            for label, gc in centroids.items():
                scores[sp][label] = _cosine(c, gc)
        used: set[str] = set()
        for sp, _dur in ordered:
            ranked = sorted(scores.get(sp, {}).items(), key=lambda kv: kv[1], reverse=True)
            label = None
            for cand, _score in ranked:
                if cand not in used:
                    label = cand
                    break
            if label is None:
                label = next((x for x in [dominant_label, minority_label] if x not in used), dominant_label)
            mapping[sp] = label
            used.add(label)
    elif len(global_labels) == 1:
        primary = global_labels[0]
        secondary = next((x for x in all_labels if x != primary), minority_label)
        mapping[ordered[0][0]] = primary
        for sp, _dur in ordered[1:]:
            mapping[sp] = secondary
    else:
        scores: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for t in turns:
            tsp = int(t["speaker"])
            ts = float(t["start"]); te = float(t["end"])
            for src in source_intervals:
                ov = min(te, float(src["end"])) - max(ts, float(src["start"]))
                if ov > 0:
                    scores[tsp][str(src["speaker"])] += ov
        used: set[str] = set()
        for sp, _dur in ordered:
            ranked = sorted(scores.get(sp, {}).items(), key=lambda kv: kv[1], reverse=True)
            label = None
            for cand, _score in ranked:
                if cand not in used:
                    label = cand
                    break
            if label is None:
                label = next((x for x in all_labels if x not in used), dominant_label)
            mapping[sp] = label
            used.add(label)

    out = []
    for t in turns:
        tsp = int(t["speaker"])
        row = {"start": float(t["start"]), "end": float(t["end"]), "speaker": mapping[tsp], "local_speaker": tsp}
        if mapping_strategy == "centroid" and 'scores' in locals() and scores.get(tsp):
            ranked = sorted(scores[tsp].items(), key=lambda kv: kv[1], reverse=True)
            row["mapping_scores"] = {k: round(float(v), 4) for k, v in ranked}
            row["mapping_margin"] = round(float(ranked[0][1] - ranked[1][1]), 4) if len(ranked) > 1 else None
        out.append(row)
    return out


def diarize_window(
    audio_path: Path,
    start: float,
    end: float,
    tmp_dir: Path,
    num_speakers: int | None = 2,
    min_duration_on: float = 0.2,
    min_duration_off: float = 0.2,
) -> list[dict]:
    audio, sr = load_audio_window(audio_path, start, end)
    wav = tmp_dir / f"win_{start:.0f}_{end:.0f}.wav"
    sf.write(wav, audio, sr)
    q = config.qwen3
    turns = run_diarization(
        str(wav),
        segmentation_model=q.segmentation_model,
        embedding_model=q.embedding_model,
        num_speakers=num_speakers,
        cluster_threshold=q.cluster_threshold,
        num_threads=4,
        provider=q.provider,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
    )
    return [{"start": start + t["start"], "end": start + t["end"], "speaker": int(t["speaker"])} for t in turns]


def flush(out, speaker, parts, start, end, debug=None):
    txt = "".join(parts)
    if speaker and txt and start is not None and end is not None:
        out.append({"start": round(start, 2), "end": round(max(end, start), 2), "speaker": speaker, "text": txt, **({"debug": debug} if debug else {})})


def align_with_turns(model, audio_path: Path, wstart: float, wend: float, source_intervals: list[dict], turns: list[dict], language: str):
    text = "".join(str(s.get("text", "")) for s in source_intervals)
    audio, sr = load_audio_window(audio_path, wstart, wend)
    t0 = time.time(); res = model.align((audio, sr), text, language)[0]; elapsed = time.time() - t0
    out=[]; cur_sp=None; parts=[]; cur_st=None; cur_en=None; low=0
    for item in res.items:
        ist = min(max(wstart + float(item.start_time), wstart), wend)
        ien = min(max(wstart + float(item.end_time), ist), wend)
        sp, conf, _ = choose_speaker(ist, ien, turns)
        if conf < 0.5: low += 1
        if cur_sp is None:
            cur_sp=sp; parts=[item.text]; cur_st=ist; cur_en=ien
        elif sp == cur_sp:
            parts.append(item.text); cur_en=max(cur_en, ien)
        else:
            flush(out, cur_sp, parts, cur_st, cur_en, {"method":"qwen3_fa_window_diarize_join","window_start":wstart,"window_end":wend})
            cur_sp=sp; parts=[item.text]; cur_st=ist; cur_en=ien
    flush(out, cur_sp, parts, cur_st, cur_en, {"method":"qwen3_fa_window_diarize_join","window_start":wstart,"window_end":wend})
    return out, {"items": len(res.items), "elapsed": elapsed, "low_conf_items": low, "turns": turns}


def _speaker_durations(segs: list[dict]) -> dict[str, float]:
    out = defaultdict(float)
    for s in segs:
        out[str(s.get("speaker"))] += max(0.0, float(s["end"]) - float(s["start"]))
    return dict(out)


def _switches(segs: list[dict]) -> int:
    return sum(1 for a, b in zip(segs, segs[1:]) if str(a.get("speaker")) != str(b.get("speaker")))


def quality_gate_accept(source_intervals: list[dict], aligned: list[dict], stats: dict) -> tuple[bool, list[str]]:
    """Reference-free rollback heuristic for refined windows.

    This is deliberately conservative: reject refinements that look unstable
    structurally or whose centroid mapping is weak. It is a manual PoC gate,
    not yet production policy.
    """
    reasons = []
    if not aligned:
        return False, ["empty_aligned_output"]
    items = max(1, int(stats.get("items", 0) or 0))
    low_ratio = float(stats.get("low_conf_items", 0)) / items
    if low_ratio > 0.35:
        reasons.append(f"low_conf_ratio>{low_ratio:.2f}")
    src_switch = _switches(source_intervals); out_switch = _switches(aligned)
    if out_switch > max(8, src_switch + 5):
        reasons.append(f"too_many_switches:{out_switch}>{src_switch}+5")
    if len(aligned) > max(10, len(source_intervals) * 4 + 4):
        reasons.append(f"too_many_output_segments:{len(aligned)}")
    very_short = sum(1 for s in aligned if float(s["end"]) - float(s["start"]) < 0.35)
    if very_short >= 4:
        reasons.append(f"too_many_very_short_segments:{very_short}")
    src_dur = _speaker_durations(source_intervals); out_dur = _speaker_durations(aligned)
    if len(src_dur) >= 2 and len(out_dur) >= 2:
        total = max(1e-6, sum(src_dur.values()))
        labels = set(src_dur) | set(out_dur)
        drift = sum(abs(src_dur.get(k, 0.0) - out_dur.get(k, 0.0)) for k in labels) / total
        if drift > 1.25 and src_switch <= 1:
            reasons.append(f"speaker_duration_drift>{drift:.2f}")
    margins = []
    top_scores = []
    for t in stats.get("turns", []):
        if t.get("mapping_margin") is not None:
            margins.append(float(t["mapping_margin"]))
        scores = list((t.get("mapping_scores") or {}).values())
        if scores:
            top_scores.append(max(scores))
    if top_scores and max(top_scores) < 0.25:
        reasons.append(f"low_centroid_similarity:{max(top_scores):.2f}")
    if margins and max(margins) < 0.03:
        reasons.append(f"low_centroid_margin:{max(margins):.2f}")
    return not reasons, reasons


def write_srt(payload: dict, path: Path):
    segs=[Segment(float(s['start']),float(s['end']),_speaker_id(str(s['speaker'])),str(s.get('text',''))) for s in payload['segments']]
    path.write_text(segments_to_srt(segs), encoding='utf-8')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input_json', type=Path); ap.add_argument('audio', type=Path)
    ap.add_argument('--aligner-model', type=Path, default=PROJECT_ROOT/'models/qwen3_diarize/Qwen3-ForcedAligner-0.6B')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--window-sec', type=float, default=40.0)
    ap.add_argument('--start-sec', type=float, default=None); ap.add_argument('--end-sec', type=float, default=None)
    ap.add_argument('--window-ids', default='', help='Comma-separated absolute window ids to align, e.g. 9,91,138')
    ap.add_argument('--max-windows', type=int, default=0); ap.add_argument('--language', default='Chinese')
    ap.add_argument('--device-map', default='mps'); ap.add_argument('--dtype', choices=['float16','float32','bfloat16'], default='float16')
    ap.add_argument('--mapping-strategy', choices=['overlap','centroid'], default='overlap')
    ap.add_argument('--num-speakers', default='2', help='Local diarization speaker count, integer or "auto"/"none"')
    ap.add_argument('--min-duration-on', type=float, default=0.2)
    ap.add_argument('--min-duration-off', type=float, default=0.2)
    ap.add_argument('--quality-gate', action='store_true', help='Rollback unstable refined windows using no-reference heuristics')
    args=ap.parse_args()
    import torch
    from qwen_asr import Qwen3ForcedAligner
    dtype={'float16':torch.float16,'float32':torch.float32,'bfloat16':torch.bfloat16}[args.dtype]
    payload=json.loads(args.input_json.read_text(encoding='utf-8')); source=payload['segments']
    dominant, minority = infer_global_roles(source)
    groups=group_segments(source,args.window_sec,args.start_sec,args.end_sec)
    if args.window_ids.strip():
        wanted = {int(x) for x in args.window_ids.split(',') if x.strip()}
        groups = {wi: segs for wi, segs in groups.items() if wi in wanted}
    items=sorted(groups.items())
    if args.max_windows: items=items[:args.max_windows]
    print(f'[load] aligner={args.aligner_model} windows={len(items)} dominant={dominant} minority={minority}')
    num_speakers = None if str(args.num_speakers).lower() in {'auto','none','null'} else int(args.num_speakers)
    centroids = extractor = audio_16k = None
    if args.mapping_strategy == 'centroid':
        print('[centroid] build global speaker centroids')
        centroids, extractor, audio_16k = build_global_centroids(args.audio, source)
        print('[centroid] labels', {k: list(map(float, v[:3])) for k, v in centroids.items()})
    model=Qwen3ForcedAligner.from_pretrained(str(args.aligner_model), dtype=dtype, device_map=args.device_map)
    new=[]; aligned_ids=set(); stats=[]
    with tempfile.TemporaryDirectory(prefix='qwen3_fa_diarize_') as td:
        tmp=Path(td)
        for wi, intervals in items:
            wstart=wi*args.window_sec; wend=wstart+args.window_sec; aligned_ids.add(wi)
            print(f'[window] {wi} {wstart:.1f}-{wend:.1f} src_segments={len(intervals)} chars={sum(len(s.get("text","")) for s in intervals)}')
            raw_turns=diarize_window(
                args.audio,wstart,wend,tmp,
                num_speakers=num_speakers,
                min_duration_on=args.min_duration_on,
                min_duration_off=args.min_duration_off,
            )
            mapped=map_local_speakers(
                raw_turns, intervals, dominant, minority,
                mapping_strategy=args.mapping_strategy,
                centroids=centroids,
                extractor=extractor,
                audio_16k=audio_16k,
            )
            aligned, st=align_with_turns(model,args.audio,wstart,wend,intervals,mapped,args.language)
            accepted=True; reject_reasons=[]
            if args.quality_gate:
                accepted, reject_reasons = quality_gate_accept(intervals, aligned, st)
            stats.append({'window':wi,'start':wstart,'end':wend,'source_segments':len(intervals),'output_segments':len(aligned),'accepted':accepted,'reject_reasons':reject_reasons,**st})
            if accepted:
                new.extend(aligned)
            else:
                new.extend([{k: v for k, v in s.items() if k != "_idx"} for s in intervals])
    preserved=[s for s in source if int(math.floor(max(0,float(s['start']))/args.window_sec)) not in aligned_ids]
    out=json.loads(json.dumps(payload,ensure_ascii=False)); out['segments']=sorted(preserved+new,key=lambda s:(float(s['start']),float(s['end'])))
    out.setdefault('summary',{})['forced_alignment_window_diarize']={'strategy':'qwen3_fa_plus_per_window_sherpa_v1','source_json':str(args.input_json),'aligner_model':str(args.aligner_model),'window_sec':args.window_sec,'aligned_windows':len(items),'accepted_windows':sum(1 for s in stats if s.get('accepted', True)),'dominant_speaker':dominant,'minority_speaker':minority,'mapping_strategy':args.mapping_strategy,'num_speakers':num_speakers,'min_duration_on':args.min_duration_on,'min_duration_off':args.min_duration_off,'quality_gate':args.quality_gate,'window_stats':stats,'source_segment_count':len(source),'output_segment_count':len(out['segments'])}
    args.out_dir.mkdir(parents=True, exist_ok=True); jp=args.out_dir/args.input_json.name; sp=jp.with_suffix('.srt')
    jp.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); write_srt(out,sp)
    print('[out] JSON',jp); print('[out] SRT ',sp)
if __name__=='__main__': main()
