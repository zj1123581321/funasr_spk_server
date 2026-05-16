"""
Qwen3 PoC chunk-level mini-diarize refine（不依赖 detector）

设计目标
========
原 align_qwen3_poc_with_window_diarize 是按 fixed 40s grid window 处理:
window 边界与 ASR 真实 chunk 边界往往错位, 容易把一个 chunk 中段切断,
导致 ForcedAligner 跨边界对齐时累积漂移。本脚本改用"重组 chunk"方案:

  1. 从 v? hypothesis JSON 的 segments 反推 ASR chunk 边界
     v7/v8/v9 等 hypothesis 都是 merge_asr_chunks_and_diarize 的产物,
     edge 与 ASR 真实 chunk 边界 (macro_start + k*40s) 一致。
     一个 chunk 内可能被 turn-overlap 切成多段, 把它们合并回 chunk。

  2. 不依赖 detector, 根据 chunk 内部结构判断是否触发 mini-diarize
     - trigger_cross: chunk 内 v7 已经包含 >=2 个 global speaker
       (说明 merge 已切, 但切分点是粗的, ForcedAligner 能给更准时间戳)
     - trigger_silence: chunk 内只有 1 个 global speaker 但 duration >= MIN_SINGLE_DUR
                       且 ffmpeg silencedetect 找到 >=1 个 >= MIN_SILENCE_SEC 的静音
       (audio 里疑似有切换但被吞掉)
     -  其它 chunk 保留 v7 输出, 不动

  3. 触发的 chunk: 走 sherpa mini-diarize(num_speakers=2) + ForcedAligner
     + centroid mapping + quality_gate, 不接受时 rollback 到 v7

复用
====
- build_global_centroids / _embedding_for_interval / map_local_speakers
  / quality_gate_accept / diarize_window / align_with_turns / infer_global_roles
  全部 import 自 align_qwen3_poc_with_window_diarize, 不重复实现。

输出
====
- {out_dir}/{input_json.name}      refined JSON (segments + summary.chunk_refine_v1)
- {out_dir}/{input_json.stem}.srt  对应 SRT
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.qwen3.merge import Segment, segments_to_srt
from tests.manual.server.align_qwen3_poc_segments import _speaker_id
from tests.manual.server.align_qwen3_poc_with_window_diarize import (
    align_with_turns,
    build_global_centroids,
    diarize_window,
    infer_global_roles,
    map_local_speakers,
    quality_gate_accept,
)


CHUNK_SEC = 40.0  # Qwen3-ASR 内部 chunk_size_sec default


# ---------- step 1: reconstruct chunk boundaries ----------

def reconstruct_chunks(segments: list[dict], macro_segments: list[dict], chunk_sec: float = CHUNK_SEC) -> list[dict]:
    """根据 macro_segments + 40s 周期反推每个 ASR chunk 的 [start,end] + 内含 v7 segments.

    返回 list, 每项 {macro_idx, chunk_idx, start, end, items: [{idx, seg}, ...]}
    顺序按 (macro_idx, chunk_idx).
    """
    def find_macro(t: float) -> Optional[dict]:
        for m in macro_segments:
            if m["start"] - 0.5 <= t <= m["end"] + 0.5:
                return m
        return None

    chunk_map: dict[tuple[int, int], dict] = {}
    for i, seg in enumerate(segments):
        m = find_macro(float(seg["start"]) + 1e-3)
        if not m:
            continue
        rel = float(seg["start"]) - float(m["start"])
        ci = int(math.floor(max(0.0, rel) / chunk_sec))
        cstart = float(m["start"]) + ci * chunk_sec
        cend = min(float(m["end"]), cstart + chunk_sec)
        key = (int(m["idx"]), ci)
        if key not in chunk_map:
            chunk_map[key] = {
                "macro_idx": int(m["idx"]),
                "chunk_idx": ci,
                "start": cstart,
                "end": cend,
                "items": [],
            }
        chunk_map[key]["items"].append({"idx": i, "seg": seg})

    return [chunk_map[k] for k in sorted(chunk_map.keys())]


# ---------- step 2: trigger classification ----------

def ffmpeg_silence_in_window(
    audio_path: Path,
    start: float,
    end: float,
    noise_db: str = "-35dB",
    min_silence_sec: float = 0.5,
) -> list[tuple[float, float]]:
    """返回该窗口内 ffmpeg silencedetect 找到的静音段(绝对秒).

    走 -ss / -t 截窗口, silencedetect noise 阈值默认 -35dB.
    任何 ffmpeg 失败都返回 [] (退化为不触发 silence-based refine, 保守).
    """
    dur = max(0.0, end - start)
    if dur < 1.0:
        return []
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-loglevel",
                "info",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{dur:.3f}",
                "-i",
                str(audio_path),
                "-af",
                f"silencedetect=noise={noise_db}:d={min_silence_sec:.3f}",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        return []
    out_lines = (proc.stderr or "").splitlines()
    silences = []
    cur_start = None
    for line in out_lines:
        if "silence_start" in line:
            try:
                tok = line.split("silence_start:")[1].strip().split()[0]
                cur_start = start + float(tok)
            except Exception:
                cur_start = None
        elif "silence_end" in line and cur_start is not None:
            try:
                tok = line.split("silence_end:")[1].strip().split()[0]
                cur_end = start + float(tok)
                silences.append((cur_start, cur_end))
            except Exception:
                pass
            cur_start = None
    # 只保留长度 >= min_silence_sec 的
    silences = [(s, e) for s, e in silences if (e - s) >= min_silence_sec]
    return silences


def classify_chunk(
    ch: dict,
    audio_path: Path,
    *,
    min_single_dur: float = 25.0,
    min_silence_sec: float = 0.5,
    noise_db: str = "-35dB",
) -> dict:
    """对 chunk 打标签: trigger_cross / trigger_silence / skip + reason."""
    items = ch["items"]
    spk = sorted({str(it["seg"]["speaker"]) for it in items})
    dur = ch["end"] - ch["start"]
    label = "skip"
    reason = "single_speaker_short"
    silences: list[tuple[float, float]] = []
    if len(spk) >= 2:
        label = "trigger_cross"
        reason = f"in_chunk_speakers={len(spk)}"
    elif dur >= min_single_dur:
        # 检查 chunk 内是否有显著静音(疑似漏切)
        # 只看 chunk 中段 [start+5, end-5], 边界静音不算
        s_pad = ch["start"] + 5.0
        e_pad = ch["end"] - 5.0
        if e_pad > s_pad + 1.0:
            silences = ffmpeg_silence_in_window(
                audio_path, s_pad, e_pad, noise_db=noise_db, min_silence_sec=min_silence_sec
            )
            if silences:
                label = "trigger_silence"
                reason = f"silences={len(silences)} longest={max((e-s for s,e in silences), default=0):.2f}s"
            else:
                reason = "single_speaker_no_silence"
        else:
            reason = "single_speaker_too_short_for_silence"
    return {
        "label": label,
        "reason": reason,
        "speakers": spk,
        "duration": dur,
        "n_segments": len(items),
        "silences": [(round(s, 2), round(e, 2)) for s, e in silences],
    }


# ---------- step 3: refine + assemble ----------

def refine_chunk(
    chunk: dict,
    audio_path: Path,
    model,
    centroids,
    extractor,
    audio_16k,
    dominant: str,
    minority: str,
    *,
    language: str,
    num_speakers: int | None,
    min_duration_on: float,
    min_duration_off: float,
    quality_gate: bool,
    tmp_dir: Path,
) -> tuple[list[dict], dict]:
    """单 chunk 的 mini-diarize + ForcedAligner + quality_gate."""
    start = chunk["start"]
    end = chunk["end"]
    intervals = [
        {**it["seg"], "_idx": it["idx"]} for it in chunk["items"]
    ]
    t0 = time.time()
    raw_turns = diarize_window(
        audio_path, start, end, tmp_dir,
        num_speakers=num_speakers,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
    )
    dia_elapsed = time.time() - t0
    mapped = map_local_speakers(
        raw_turns, intervals, dominant, minority,
        mapping_strategy="centroid",
        centroids=centroids,
        extractor=extractor,
        audio_16k=audio_16k,
    )
    aligned, st = align_with_turns(model, audio_path, start, end, intervals, mapped, language)
    accepted = True
    reject_reasons: list[str] = []
    if quality_gate:
        accepted, reject_reasons = quality_gate_accept(intervals, aligned, st)
    return (aligned if accepted else [{k: v for k, v in s.items() if k != "_idx"} for s in intervals]), {
        "diarize_elapsed": round(dia_elapsed, 3),
        "align_elapsed": round(st.get("elapsed", 0.0), 3),
        "items": st.get("items", 0),
        "low_conf_items": st.get("low_conf_items", 0),
        "raw_turns": len(raw_turns),
        "mapped_turns": len(mapped),
        "output_segments": len(aligned),
        "accepted": accepted,
        "reject_reasons": reject_reasons,
    }


def write_srt(payload: dict, path: Path) -> None:
    segs = [
        Segment(
            float(s["start"]),
            float(s["end"]),
            _speaker_id(str(s["speaker"])),
            str(s.get("text", "")),
        )
        for s in payload["segments"]
    ]
    path.write_text(segments_to_srt(segs), encoding="utf-8")


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen3 chunk-level mini-diarize refine (detector-free)")
    ap.add_argument("input_json", type=Path, help="Source hypothesis JSON (e.g. v7 output)")
    ap.add_argument("audio", type=Path, help="Original audio path (must match input_json)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--aligner-model",
        type=Path,
        default=PROJECT_ROOT / "models/qwen3_diarize/Qwen3-ForcedAligner-0.6B",
    )
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--device-map", default="mps")
    ap.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
    )
    ap.add_argument("--chunk-sec", type=float, default=CHUNK_SEC)
    # trigger 控制
    ap.add_argument(
        "--triggers",
        default="cross",
        help="Comma-separated triggers: cross,silence,all (default cross)",
    )
    ap.add_argument(
        "--min-single-dur",
        type=float,
        default=25.0,
        help="trigger_silence 仅对 duration>=此值 的单 speaker chunk 检查",
    )
    ap.add_argument("--min-silence-sec", type=float, default=0.5)
    ap.add_argument("--silence-noise", default="-35dB")
    # diarize 控制
    ap.add_argument(
        "--num-speakers",
        default="2",
        help="Local diarization speaker count, integer or 'auto'",
    )
    ap.add_argument("--min-duration-on", type=float, default=0.2)
    ap.add_argument("--min-duration-off", type=float, default=0.2)
    ap.add_argument("--no-quality-gate", action="store_true")
    # 限制范围 (用于 macro-level smoke test)
    ap.add_argument("--start-sec", type=float, default=None)
    ap.add_argument("--end-sec", type=float, default=None)
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--plan-only", action="store_true", help="Only print chunk plan + trigger counts")
    args = ap.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    source: list[dict] = payload["segments"]
    macro_segments = payload.get("summary", {}).get("macro_segments")
    if not macro_segments:
        raise SystemExit("input_json.summary.macro_segments 不存在, 无法反推 chunk 边界")

    dominant, minority = infer_global_roles(source)
    print(f"[load] source segments={len(source)} dominant={dominant} minority={minority}")

    chunks_all = reconstruct_chunks(source, macro_segments, chunk_sec=args.chunk_sec)
    # 范围裁剪
    if args.start_sec is not None:
        chunks_all = [c for c in chunks_all if c["end"] > args.start_sec]
    if args.end_sec is not None:
        chunks_all = [c for c in chunks_all if c["start"] < args.end_sec]
    print(f"[plan] total chunks reconstructed={len(chunks_all)}")

    # 分类 (此处会跑 ffmpeg silencedetect, 单 chunk ~0.3s)
    triggers_wanted = set(t.strip() for t in args.triggers.split(",") if t.strip())
    if "all" in triggers_wanted:
        triggers_wanted = {"cross", "silence"}

    print(f"[classify] triggers={sorted(triggers_wanted)} min_single_dur={args.min_single_dur}")
    t_cls = time.time()
    plan = []
    for ch in chunks_all:
        cls = classify_chunk(
            ch, args.audio,
            min_single_dur=args.min_single_dur,
            min_silence_sec=args.min_silence_sec,
            noise_db=args.silence_noise,
        )
        ch["_cls"] = cls
        ch["_run"] = (
            (cls["label"] == "trigger_cross" and "cross" in triggers_wanted)
            or (cls["label"] == "trigger_silence" and "silence" in triggers_wanted)
        )
        plan.append({"macro": ch["macro_idx"], "chunk": ch["chunk_idx"], "start": ch["start"], "end": ch["end"], **cls, "run": ch["_run"]})
    print(f"[classify] elapsed={time.time()-t_cls:.1f}s")

    counts = {"trigger_cross": 0, "trigger_silence": 0, "skip": 0}
    for ch in chunks_all:
        counts[ch["_cls"]["label"]] = counts.get(ch["_cls"]["label"], 0) + 1
    n_run = sum(1 for ch in chunks_all if ch["_run"])
    print(f"[plan] {counts} runnable={n_run}")

    if args.max_chunks:
        # 按顺序保留前 N 个 runnable, 其它保留 v7
        kept = 0
        for ch in chunks_all:
            if ch["_run"]:
                if kept < args.max_chunks:
                    kept += 1
                else:
                    ch["_run"] = False
        n_run = sum(1 for ch in chunks_all if ch["_run"])
        print(f"[plan] max_chunks={args.max_chunks} -> runnable={n_run}")

    if args.plan_only:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "plan.json").write_text(
            json.dumps({"counts": counts, "runnable": n_run, "plan": plan}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[plan] written {args.out_dir/'plan.json'}")
        return

    # build centroids
    print("[centroid] building global speaker centroids")
    centroids, extractor, audio_16k = build_global_centroids(args.audio, source)
    print(f"[centroid] labels={list(centroids.keys())}")

    # build aligner
    import torch
    from qwen_asr import Qwen3ForcedAligner
    dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"[load] aligner={args.aligner_model} dtype={args.dtype} device={args.device_map}")
    model = Qwen3ForcedAligner.from_pretrained(
        str(args.aligner_model), dtype=dtype, device_map=args.device_map
    )
    num_speakers = None if str(args.num_speakers).lower() in {"auto", "none", "null"} else int(args.num_speakers)

    refined_segments_by_chunk: dict[tuple[int, int], list[dict]] = {}
    refined_chunk_ids: set[tuple[int, int]] = set()
    accepted_chunk_ids: set[tuple[int, int]] = set()
    stats: list[dict] = []
    t_refine = time.time()
    with tempfile.TemporaryDirectory(prefix="qwen3_chunk_refine_") as td:
        tmp = Path(td)
        run_idx = 0
        run_total = sum(1 for ch in chunks_all if ch["_run"])
        for ch in chunks_all:
            if not ch["_run"]:
                continue
            run_idx += 1
            key = (ch["macro_idx"], ch["chunk_idx"])
            refined_chunk_ids.add(key)
            print(
                f"[refine {run_idx}/{run_total}] macro={ch['macro_idx']} chunk={ch['chunk_idx']} "
                f"[{ch['start']:.1f},{ch['end']:.1f}] {ch['_cls']['label']} {ch['_cls']['reason']}"
            )
            try:
                segs, st = refine_chunk(
                    ch, args.audio, model, centroids, extractor, audio_16k,
                    dominant, minority,
                    language=args.language,
                    num_speakers=num_speakers,
                    min_duration_on=args.min_duration_on,
                    min_duration_off=args.min_duration_off,
                    quality_gate=not args.no_quality_gate,
                    tmp_dir=tmp,
                )
            except Exception as exc:
                print(f"[refine] error: {exc!r}; rollback to v7 for this chunk")
                segs = [{k: v for k, v in it["seg"].items()} for it in ch["items"]]
                st = {"accepted": False, "reject_reasons": [f"exception:{type(exc).__name__}"]}
            refined_segments_by_chunk[key] = segs
            if st.get("accepted"):
                accepted_chunk_ids.add(key)
            stats.append({
                "macro_idx": ch["macro_idx"],
                "chunk_idx": ch["chunk_idx"],
                "start": ch["start"],
                "end": ch["end"],
                "trigger": ch["_cls"]["label"],
                "reason": ch["_cls"]["reason"],
                "n_source_segments": ch["_cls"]["n_segments"],
                **{k: v for k, v in st.items() if k != "raw_turns" or v is not None},
            })

    print(
        f"[refine] elapsed={time.time()-t_refine:.1f}s "
        f"refined_chunks={len(refined_chunk_ids)} accepted={len(accepted_chunk_ids)}"
    )

    # rebuild segments: 对每个 chunk, 若 refined 用 refined; 否则保留 v7
    new_segments: list[dict] = []
    for ch in chunks_all:
        key = (ch["macro_idx"], ch["chunk_idx"])
        if key in refined_chunk_ids:
            for s in refined_segments_by_chunk[key]:
                new_segments.append({k: v for k, v in s.items() if k != "_idx"})
        else:
            for it in ch["items"]:
                seg = it["seg"]
                new_segments.append({k: v for k, v in seg.items()})

    new_segments.sort(key=lambda s: (float(s["start"]), float(s["end"])))
    out = json.loads(json.dumps(payload, ensure_ascii=False))
    out["segments"] = new_segments
    out.setdefault("summary", {})["chunk_refine_v1"] = {
        "strategy": "chunk_reconstruct_plus_per_chunk_sherpa_fa_centroid_gate_v1",
        "source_json": str(args.input_json),
        "aligner_model": str(args.aligner_model),
        "chunk_sec": args.chunk_sec,
        "triggers": sorted(triggers_wanted),
        "min_single_dur": args.min_single_dur,
        "min_silence_sec": args.min_silence_sec,
        "silence_noise": args.silence_noise,
        "num_speakers": num_speakers,
        "min_duration_on": args.min_duration_on,
        "min_duration_off": args.min_duration_off,
        "quality_gate": not args.no_quality_gate,
        "dominant_speaker": dominant,
        "minority_speaker": minority,
        "chunk_count_total": len(chunks_all),
        "chunk_counts_by_label": counts,
        "chunks_refined": len(refined_chunk_ids),
        "chunks_accepted": len(accepted_chunk_ids),
        "source_segment_count": len(source),
        "output_segment_count": len(new_segments),
        "chunk_stats": stats,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jp = args.out_dir / args.input_json.name
    sp = jp.with_suffix(".srt")
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    write_srt(out, sp)
    print(f"[out] JSON {jp}")
    print(f"[out] SRT  {sp}")


if __name__ == "__main__":
    main()
