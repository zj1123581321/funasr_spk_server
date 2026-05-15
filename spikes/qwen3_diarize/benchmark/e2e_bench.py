"""
端到端 benchmark — ASR + Diarization + Merger,串行执行(单进程).

并行执行(分进程 ASR vs Diarize)留给下一版,因为:
  - ASR 占 4GB RSS,fork 一个 worker 太重
  - PoC 第一版用单进程,先看端到端总时长 = ASR_time + diarize_time

用法:
    venv/bin/python benchmark/e2e_bench.py <audio> --out-dir output/
"""
import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import asdict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.asr import run_asr
from src.diarize import run_diarization
from src.merge import merge_asr_and_diarize, segments_to_srt, segments_to_rttm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out-dir", default="output", help="输出目录")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--cluster-threshold", type=float, default=0.9,
                    help="diarize 聚类阈值 (双人对话 0.9 经验值)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_name = Path(args.audio).stem

    # === Stage 1: ASR ===
    print(f"[E2E] === Stage 1: ASR ===", file=sys.stderr)
    t0 = time.time()
    asr = run_asr(args.audio, language=args.language)
    t_asr = time.time() - t0
    print(
        f"[E2E] ASR done: text_len={len(asr.text)} dur={asr.duration:.1f}s "
        f"elapsed={asr.elapsed:.1f}s rtf={asr.rtf:.3f}",
        file=sys.stderr,
    )

    # === Stage 2: Diarization ===
    print(f"[E2E] === Stage 2: Diarization ===", file=sys.stderr)
    t1 = time.time()
    turns = run_diarization(
        args.audio,
        num_speakers=None,
        cluster_threshold=args.cluster_threshold,
        provider="coreml",
        num_threads=4,
    )
    t_dia = time.time() - t1
    speakers = sorted({t["speaker"] for t in turns})
    print(
        f"[E2E] Diarize done: turns={len(turns)} speakers={speakers} "
        f"elapsed={t_dia:.1f}s rtf={t_dia/asr.duration:.3f}",
        file=sys.stderr,
    )

    # === Stage 3: Merge ===
    print(f"[E2E] === Stage 3: Merge ===", file=sys.stderr)
    t2 = time.time()
    segments = merge_asr_and_diarize(asr.text, turns)
    t_merge = time.time() - t2
    print(f"[E2E] Merge done: {len(segments)} segments, took {t_merge*1000:.1f}ms", file=sys.stderr)

    # === Write outputs ===
    json_path = out_dir / f"{audio_name}.e2e.json"
    json_path.write_text(json.dumps(
        {
            "audio": args.audio,
            "duration": asr.duration,
            "asr_text": asr.text,
            "asr": {
                "elapsed": asr.elapsed,
                "rtf": asr.rtf,
                "peak_rss_mb": asr.peak_rss_mb,
            },
            "diarize": {
                "elapsed": t_dia,
                "rtf": t_dia / asr.duration,
                "turns": turns,
            },
            "e2e": {
                "elapsed_serial": t_asr + t_dia + t_merge,
                "rtf_serial": (t_asr + t_dia + t_merge) / asr.duration,
                # 如果 ASR 和 diarize 真并行,理论端到端 = max
                "elapsed_parallel_theoretical": max(t_asr, t_dia) + t_merge,
                "rtf_parallel_theoretical": (max(t_asr, t_dia) + t_merge) / asr.duration,
            },
            "segments": [asdict(s) for s in segments],
        },
        ensure_ascii=False, indent=2,
    ))

    srt_path = out_dir / f"{audio_name}.srt"
    srt_path.write_text(segments_to_srt(segments))

    rttm_path = out_dir / f"{audio_name}.rttm"
    rttm_path.write_text(segments_to_rttm(segments, file_id=audio_name))

    # === 汇总 ===
    rtf_serial = (t_asr + t_dia + t_merge) / asr.duration
    rtf_parallel = (max(t_asr, t_dia) + t_merge) / asr.duration
    print(file=sys.stderr)
    print(f"=== E2E Summary ===", file=sys.stderr)
    print(f"  audio_dur     : {asr.duration:.2f}s", file=sys.stderr)
    print(f"  asr_time      : {t_asr:.2f}s  rtf={t_asr/asr.duration:.3f}", file=sys.stderr)
    print(f"  diarize_time  : {t_dia:.2f}s  rtf={t_dia/asr.duration:.3f}", file=sys.stderr)
    print(f"  merge_time    : {t_merge*1000:.1f}ms", file=sys.stderr)
    print(f"  serial RTF    : {rtf_serial:.3f}", file=sys.stderr)
    print(f"  parallel RTF* : {rtf_parallel:.3f}  (* 假设 ASR/diarize 真并行)", file=sys.stderr)
    print(f"  peak_rss      : {asr.peak_rss_mb:.0f} MB", file=sys.stderr)
    print(f"  speakers      : {speakers}  turns={len(turns)}  segments={len(segments)}", file=sys.stderr)
    print(file=sys.stderr)
    print(f"  -> {json_path}", file=sys.stderr)
    print(f"  -> {srt_path}", file=sys.stderr)
    print(f"  -> {rttm_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
