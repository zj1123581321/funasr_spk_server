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
from src.merge import (
    merge_asr_and_diarize,
    segments_to_srt,
    segments_to_rttm,
    filter_spurious_speakers,
)

# diarize 配置预设(按实测组结果命名)
PRESETS = {
    # 普适最优: fp32 seg + NeMo + cpu 8t + 自动聚类 (threshold=0.9)
    # 实测 2 人/4 人均稳健;后处理过滤 <1% 时长的 spurious cluster
    # 这是不知道说话人数的生产推荐
    "auto": dict(
        segmentation_model="models/sherpa/pyannote-segmentation-3.0/model.onnx",
        embedding_model="models/sherpa/nemo-titanet-small/embedding.onnx",
        provider="cpu",
        num_threads=8,
        num_speakers=None,   # 自动聚类
    ),
    # 已知 2 人最优: int8 seg + NeMo + cpu 8t + 锁 2 spk,RTF 0.1435,RSS 604MB
    "D": dict(
        segmentation_model="models/sherpa/pyannote-segmentation-3.0/model.int8.onnx",
        embedding_model="models/sherpa/nemo-titanet-small/embedding.onnx",
        provider="cpu",
        num_threads=8,
        num_speakers=2,
    ),
    # 质量稳: fp32 seg + NeMo + coreml 4t + 自动聚类,RTF 0.1645,2 spk 自动正确
    "C": dict(
        segmentation_model="models/sherpa/pyannote-segmentation-3.0/model.onnx",
        embedding_model="models/sherpa/nemo-titanet-small/embedding.onnx",
        provider="coreml",
        num_threads=4,
        num_speakers=None,
    ),
    # 原方案: fp32 + 3D-Speaker + coreml,RTF 0.465 (慢 3x)
    "baseline": dict(
        segmentation_model="models/sherpa/pyannote-segmentation-3.0/model.onnx",
        embedding_model="models/sherpa/3dspeaker-eres2net/embedding.onnx",
        provider="coreml",
        num_threads=4,
        num_speakers=None,
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out-dir", default="output", help="输出目录")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--cluster-threshold", type=float, default=0.9,
                    help="diarize 聚类阈值 (双人对话 0.9 经验值)")
    ap.add_argument("--preset", default="auto", choices=list(PRESETS.keys()),
                    help="diarize 预设 (auto=未知人数推荐, D=已知2人, C=2人稳健, baseline=原方案)")
    ap.add_argument("--filter-spurious", action="store_true", default=True,
                    help="过滤总时长<2s 的'假说话人'(短噪声 turn 合并到邻居)")
    ap.add_argument("--no-filter-spurious", dest="filter_spurious", action="store_false")
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
    preset_cfg = PRESETS[args.preset]
    print(f"[E2E] === Stage 2: Diarization (preset={args.preset}) ===", file=sys.stderr)
    print(f"[E2E]   {preset_cfg}", file=sys.stderr)
    t1 = time.time()
    turns = run_diarization(
        args.audio,
        cluster_threshold=args.cluster_threshold,
        **preset_cfg,
    )
    t_dia = time.time() - t1
    speakers_raw = sorted({t["speaker"] for t in turns})
    print(
        f"[E2E] Diarize done: turns={len(turns)} speakers_raw={speakers_raw} "
        f"elapsed={t_dia:.1f}s rtf={t_dia/asr.duration:.3f}",
        file=sys.stderr,
    )

    # === Stage 2.5: Filter spurious speakers ===
    if args.filter_spurious:
        # 用 max(2s, 1% of audio_duration) 作为阈值,自适应音频时长
        turns_filtered = filter_spurious_speakers(
            turns,
            min_speaker_total=2.0,
            min_speaker_share=0.01,
            audio_duration=asr.duration,
        )
        spurious_dropped = len(set(t["speaker"] for t in turns)) - len(
            set(t["speaker"] for t in turns_filtered)
        )
        if spurious_dropped > 0:
            print(f"[E2E] Filtered {spurious_dropped} spurious speakers (<2s total)", file=sys.stderr)
        turns = turns_filtered

    # === Stage 3: Merge ===
    speakers = sorted({t["speaker"] for t in turns})
    print(f"[E2E] === Stage 3: Merge (final speakers={speakers}) ===", file=sys.stderr)
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
