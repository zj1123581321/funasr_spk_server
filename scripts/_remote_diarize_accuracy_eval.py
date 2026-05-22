"""CUDA 环境 Qwen3 + ort_cuda diarize 准确度评测 (5 段 audio × 5 ablation 配置).

设计要点:
  - 每段 audio 跑 1 次 ASR + raw diarize (耗时大头), 缓存中间结果
  - 5 种后处理 ablation 复用同一份 raw 结果, 只重跑后处理层 (秒级)
  - filter_spurious 总开 (跟 production 一致), 剩 3 层做 single-knob ablation
  - sherpa@cuda 单跑 default-only 做 cross-backend parity check

跑法 (远端 cuda dev box):
  export FUNASR_PROFILE=cuda_dev
  export LD_LIBRARY_PATH=...  # 见 scripts/_remote_*.sh
  cd ~/Dev/projects/funasr_spk_server
  venv/bin/python scripts/_remote_diarize_accuracy_eval.py

输出:
  - stdout: 进度日志 + per-config 结构化数据 (人类可读)
  - tmp_long_audio/cuda_diarize_accuracy_eval.json: 全量结果 JSON, 给报告引用
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 评测集: (label, 相对 path, true_spk_count, 备注)
# 真实人数来源 tmp_long_audio/eval_set/README.md
AUDIOS = [
    ("1spk_real", "tmp_long_audio/eval_set/audio_1spk_real.m4a", 1,
        "杨涛个人频道独白 16min"),
    ("panel_actual_3plus", "tmp_long_audio/eval_set/audio_panel_marked_1spk.m4a", 3,
        "汽车访谈, 标'1人'实际 3+ 人 34min"),
    ("2spk_60min", "tmp_long_audio/eval_set/audio_2spk_60min.mp3", 2,
        "吴明辉 × 程曼祺 双人访谈 60min"),
    # 注: eval_set/audio_4spk.m4a 是 symlink 指向 macOS 本机路径, 在远端 cuda 上失效
    # 这里直接指真实文件 tmp_long_audio/multi_speaker_test/podcast_4spk.m4a
    ("4spk", "tmp_long_audio/multi_speaker_test/podcast_4spk.m4a", 4,
        "女性话题 4 人圆桌 44min (含英文歌)"),
    ("6spk_60min", "tmp_long_audio/eval_set/audio_6spk_60min.m4a", 6,
        "小宇宙 6 人对话 60min"),
]

# 5 种后处理 ablation (filter_spurious 总开)
CONFIGS = [
    ("default",         {"cluster_merge": True,  "short_guard": True,  "silence_align": True}),
    ("no_cluster_merge",{"cluster_merge": False, "short_guard": True,  "silence_align": True}),
    ("no_short_guard",  {"cluster_merge": True,  "short_guard": False, "silence_align": True}),
    ("no_silence_align",{"cluster_merge": True,  "short_guard": True,  "silence_align": False}),
    ("all_off",         {"cluster_merge": False, "short_guard": False, "silence_align": False}),
]


def _build_transcriber():
    """复刻 src.core.qwen3_inproc_pool._default_transcriber_factory."""
    from src.core.config import config
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber

    q = config.qwen3
    tx = Qwen3DiarizeTranscriber(
        asr_model_dir=q.asr_model_dir,
        segmentation_model=q.segmentation_model,
        embedding_model=q.embedding_model,
        num_speakers=q.num_speakers,
        cluster_threshold=q.cluster_threshold,
        num_threads=q.num_threads,
        provider=q.provider,
        language=q.language,
        temperature=q.temperature,
        short_segment_guard_enabled=q.short_segment_guard_enabled,
        short_segment_drop_sec=q.short_segment_drop_sec,
        short_segment_aba_max_mid_sec=q.short_segment_aba_max_mid_sec,
        short_segment_merge_same=q.short_segment_merge_same,
        cluster_merge_enabled=q.cluster_merge_enabled,
        cluster_merge_min_main_share=q.cluster_merge_min_main_share,
        cluster_merge_relabel_threshold=q.cluster_merge_relabel_threshold,
        cluster_merge_main_threshold=q.cluster_merge_main_threshold,
        cluster_merge_dominant_share=q.cluster_merge_dominant_share,
        cluster_merge_dominant_threshold=q.cluster_merge_dominant_threshold,
        cluster_merge_dominant_minor_threshold=q.cluster_merge_dominant_minor_threshold,
        silence_align_enabled=q.silence_align_enabled,
        silence_align_tolerance_sec=q.silence_align_tolerance_sec,
        silence_align_min_segment_dur_sec=q.silence_align_min_segment_dur_sec,
        silence_vad_noise_db=q.silence_vad_noise_db,
        silence_vad_min_silence_sec=q.silence_vad_min_silence_sec,
    )
    return tx


def _summarize_segments(segments) -> dict:
    """从 List[Segment] 算 spk count / durations / shares / segments_count.

    Segment 是 dataclass (start: float, end: float, speaker: int, text: str)
    """
    spk_durations: dict[int, float] = {}
    for seg in segments:
        spk = seg.speaker
        dur = seg.end - seg.start
        spk_durations[spk] = spk_durations.get(spk, 0.0) + dur

    total = sum(spk_durations.values())
    sorted_spk = sorted(spk_durations.items(), key=lambda x: -x[1])
    return {
        "spk_count": len(spk_durations),
        "speakers": [
            {
                "id": f"Speaker{k}",
                "duration": round(v, 2),
                "share": round(v / total, 4) if total > 0 else 0.0,
            }
            for k, v in sorted_spk
        ],
        "segments": len(segments),
        "total_speech": round(total, 2),
    }


def _summarize_raw_turns(raw_turns) -> dict:
    """raw diarize 输出统计 (在任何后处理之前)."""
    cluster_durations: dict[int, float] = {}
    for t in raw_turns:
        spk = t["speaker"]
        dur = t["end"] - t["start"]
        cluster_durations[spk] = cluster_durations.get(spk, 0.0) + dur

    sorted_clusters = sorted(cluster_durations.items(), key=lambda x: -x[1])
    total = sum(cluster_durations.values())
    return {
        "raw_clusters": len(cluster_durations),
        "raw_turns": len(raw_turns),
        "cluster_size_sec": [round(v, 2) for _, v in sorted_clusters],
        "cluster_size_share": [
            round(v / total, 4) if total > 0 else 0.0 for _, v in sorted_clusters
        ],
    }


def _apply_post_pipeline(
    transcriber,
    raw_turns,
    asr_result,
    audio_path: str,
    ablation: dict,
    extractor_fn,
):
    """对 cached (raw_turns, asr_result) 跑指定 ablation 的后处理流水.

    复刻 Qwen3DiarizeTranscriber.transcribe 的 post-processing 段, 但用 ablation dict
    动态控制 cluster_merge / short_guard / silence_align 三个开关.
    """
    from src.core.qwen3.merge import (
        filter_spurious_speakers,
        merge_asr_chunks_and_diarize,
        merge_asr_and_diarize,
    )
    from src.core.qwen3_transcriber import (
        apply_cluster_centroid_merge_to_turns,
        apply_short_segment_guard_to_segments,
        apply_silence_align_to_segments,
    )

    # filter_spurious 总开 (跟 production transcribe 一致)
    turns = filter_spurious_speakers(
        raw_turns,
        min_speaker_total=transcriber.spurious_min_total,
        min_speaker_share=transcriber.spurious_min_share,
        audio_duration=asr_result.duration,
    )

    # cluster_merge (turn-level)
    if ablation["cluster_merge"]:
        transcriber.cluster_merge_enabled = True
        try:
            turns, _ = apply_cluster_centroid_merge_to_turns(
                turns, audio_path, transcriber, extractor_fn=extractor_fn
            )
        except Exception as exc:
            print(f"    [warn] cluster_merge 失败, 跳过: {exc}", flush=True)

    # ASR + diarize 切文本
    if asr_result.chunks:
        segments = merge_asr_chunks_and_diarize(asr_result.chunks, turns)
    else:
        segments = merge_asr_and_diarize(asr_result.text, turns)

    # short_guard (segment-level)
    transcriber.short_segment_guard_enabled = ablation["short_guard"]
    segments, _ = apply_short_segment_guard_to_segments(segments, transcriber)

    # silence_align (segment-level, ffmpeg)
    transcriber.silence_align_enabled = ablation["silence_align"]
    segments, _ = apply_silence_align_to_segments(
        segments, audio_path, asr_result.duration, transcriber
    )

    return segments


def eval_audio(
    transcriber,
    label: str,
    audio_path: str,
    true_spk: int,
    note: str,
    backend: str = "ort_cuda",
    run_ablations: bool = True,
) -> dict:
    """对单段 audio: 跑 1 次 ASR + raw diarize, 再跑 5 配置后处理 ablation.

    backend="sherpa" 时只跑 default 配置 (cross-backend parity check).
    """
    from src.core.config import config
    from src.core.qwen3.asr import run_asr
    from src.core.qwen3.diarize import run_diarization_dispatched
    from src.core.qwen3.diarize_ort import reset_session_cache

    print(f"\n{'='*70}", flush=True)
    print(f"== {label} | true_spk={true_spk} | backend={backend}", flush=True)
    print(f"   {note}", flush=True)
    print(f"   {audio_path}", flush=True)
    print(f"{'='*70}", flush=True)

    if not Path(audio_path).exists():
        print(f"   缺音频, 跳过", flush=True)
        return {"label": label, "backend": backend, "skipped": True}

    engine = transcriber._ensure_engine()
    extractor_fn = transcriber._ensure_embedding_extractor_fn()

    # 1) ASR
    asr_t0 = time.time()
    try:
        asr_result = run_asr(
            audio_path,
            engine,
            language=config.qwen3.language,
            temperature=config.qwen3.temperature,
        )
    except Exception as exc:
        print(f"   ASR 失败: {exc}", flush=True)
        traceback.print_exc()
        return {"label": label, "backend": backend, "error": f"asr: {exc}"}
    asr_wall = time.time() - asr_t0
    print(f"   ASR: wall={asr_wall:.1f}s duration={asr_result.duration:.1f}s "
          f"text={len(asr_result.text)} chars chunks={len(asr_result.chunks) if asr_result.chunks else 0}",
          flush=True)

    # 2) Raw diarize (显式 backend)
    reset_session_cache()  # ort_cuda session cache, 跨 backend 清干净
    diar_t0 = time.time()
    try:
        raw_turns = run_diarization_dispatched(
            audio_path,
            segmentation_model=config.qwen3.segmentation_model,
            embedding_model=config.qwen3.embedding_model,
            num_speakers=config.qwen3.num_speakers,
            cluster_threshold=config.qwen3.cluster_threshold,
            num_threads=config.qwen3.num_threads,
            provider=config.qwen3.provider,
            backend=backend,
        )
    except Exception as exc:
        print(f"   Diarize 失败: {exc}", flush=True)
        traceback.print_exc()
        return {"label": label, "backend": backend, "error": f"diarize: {exc}"}
    diar_wall = time.time() - diar_t0
    raw_stats = _summarize_raw_turns(raw_turns)
    print(f"   Diarize raw: wall={diar_wall:.1f}s clusters={raw_stats['raw_clusters']} "
          f"turns={raw_stats['raw_turns']} sizes={raw_stats['cluster_size_sec'][:8]}",
          flush=True)

    # 3) 5 配置 ablation
    if run_ablations:
        configs_to_run = CONFIGS
    else:
        configs_to_run = [CONFIGS[0]]  # default only

    ablation_results = []
    for cfg_name, cfg in configs_to_run:
        post_t0 = time.time()
        try:
            segments = _apply_post_pipeline(
                transcriber, raw_turns, asr_result, audio_path, cfg, extractor_fn
            )
        except Exception as exc:
            print(f"   [{cfg_name}] 失败: {exc}", flush=True)
            traceback.print_exc()
            ablation_results.append({"config": cfg_name, "error": str(exc)})
            continue
        post_wall = time.time() - post_t0
        summary = _summarize_segments(segments)
        delta = summary["spk_count"] - true_spk
        verdict = "OK" if delta == 0 else ("over" if delta > 0 else "under")
        print(
            f"   [{cfg_name:18s}] spk={summary['spk_count']} ({verdict:>4s} {delta:+d}) "
            f"segs={summary['segments']:4d} post_wall={post_wall:.2f}s "
            f"top3={[(s['id'], s['duration'], s['share']) for s in summary['speakers'][:3]]}",
            flush=True,
        )
        ablation_results.append({
            "config": cfg_name,
            "ablation": cfg,
            "post_wall": round(post_wall, 2),
            **summary,
            "delta_vs_true": delta,
            "verdict": verdict,
        })

    return {
        "label": label,
        "audio": audio_path,
        "note": note,
        "backend": backend,
        "true_spk": true_spk,
        "asr_wall": round(asr_wall, 2),
        "diar_wall": round(diar_wall, 2),
        "asr_duration": round(asr_result.duration, 2),
        **raw_stats,
        "configs": ablation_results,
    }


def main() -> None:
    """跑评测.

    Usage:
      python scripts/_remote_diarize_accuracy_eval.py           # 跑全部 5 段 audio
      python scripts/_remote_diarize_accuracy_eval.py 1         # 只跑前 1 段 (smoke test)
      python scripts/_remote_diarize_accuracy_eval.py 3 ort_cuda # 只跑前 3 段, 跳过 sherpa parity
    """
    from src.core.config import config

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(AUDIOS)
    only_backend = sys.argv[2] if len(sys.argv) > 2 else None  # "ort_cuda" / "sherpa" / None=both
    audios_to_run = AUDIOS[:limit]

    # 启动 banner
    print("=" * 70, flush=True)
    print("CUDA Qwen3 + ort_cuda Diarize 准确度评测", flush=True)
    print(f"  FUNASR_PROFILE = {os.environ.get('FUNASR_PROFILE', '(not set)')}", flush=True)
    print(f"  default_engine = {config.transcription.default_engine}", flush=True)
    print(f"  qwen3.num_threads = {config.qwen3.num_threads}", flush=True)
    print(f"  qwen3.provider = {config.qwen3.provider}", flush=True)
    print(f"  qwen3.cluster_threshold = {config.qwen3.cluster_threshold}", flush=True)
    try:
        from src.core.runtime import describe_runtime, detect_runtime
        runtime = detect_runtime()
        print(f"  runtime = {describe_runtime(runtime)}", flush=True)
    except Exception as exc:
        print(f"  runtime detect failed: {exc}", flush=True)
    print("=" * 70, flush=True)

    transcriber = _build_transcriber()

    all_results = {"audios": []}

    # Phase 1: ort_cuda (主评测, 5 ablation 配置全跑)
    if only_backend in (None, "ort_cuda"):
        print("\n" + "#" * 70, flush=True)
        print(f"# Phase 1: ort_cuda backend ({len(CONFIGS)} ablation 配置, {len(audios_to_run)} 段 audio)",
              flush=True)
        print("#" * 70, flush=True)
        for label, rel_path, true_spk, note in audios_to_run:
            result = eval_audio(
                transcriber,
                label=label,
                audio_path=str(ROOT / rel_path),
                true_spk=true_spk,
                note=note,
                backend="ort_cuda",
                run_ablations=True,
            )
            all_results["audios"].append(result)

    # Phase 2: sherpa@cuda (parity check, 仅 default 配置)
    if only_backend in (None, "sherpa"):
        print("\n" + "#" * 70, flush=True)
        print(f"# Phase 2: sherpa backend on cuda (cross-backend parity, default only, {len(audios_to_run)} 段 audio)",
              flush=True)
        print("#" * 70, flush=True)
        for label, rel_path, true_spk, note in audios_to_run:
            result = eval_audio(
                transcriber,
                label=label,
                audio_path=str(ROOT / rel_path),
                true_spk=true_spk,
                note=note,
                backend="sherpa",
                run_ablations=False,
            )
            all_results["audios"].append(result)

    # 写完整 JSON
    out_json = ROOT / "tmp_long_audio" / "cuda_diarize_accuracy_eval.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}", flush=True)
    print(f"完整结果已保存: {out_json}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
