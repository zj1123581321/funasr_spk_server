"""单进程内 2 个 Qwen3DiarizeTranscriber 实例 + asyncio.gather 并发 probe.

跟 _remote_concurrent_worker.py 的差异:
- 那个: 2 个独立进程, 每进程独立 cuda context, 跨进程 race (MPS 不解)
- 这个: 1 个进程, 2 个 transcriber 实例, 同一 cuda context, race-free 假设

假设: 同进程内多 ORT InferenceSession + 多 llama_context 在 CUDA EP 上 race-free
(llama.cpp 文档明确各 llama_context 独立). 如果 GPU buffer 真隔离, 单进程 gather
能跑完两个 1800s task. LLM 阶段会 serialize on GPU kernel queue (单 CUDA stream),
预期总 wall ≈ 单 task wall * 1.7-2.0 (ASR/diarize 并行省一些).

用法:
  python scripts/_remote_single_proc_concurrent_probe.py <audio> [count=2]
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("FUNASR_QWEN3_ASR_ENCODER_PROVIDER", "cuda")
os.environ.setdefault("FUNASR_QWEN3_DIARIZE_BACKEND", "ort_cuda")


async def transcribe_one(tx, audio_path: str, task_id: str) -> dict:
    t0 = time.time()
    result, _ = await tx.transcribe(
        audio_path=audio_path,
        task_id=task_id,
        progress_callback=None,
        output_format="json",
    )
    wall = time.time() - t0
    rtf = wall / result.duration if result.duration else 0
    return {
        "task_id": task_id,
        "wall": wall,
        "rtf": rtf,
        "speakers": list(result.speakers),
        "segs": len(result.segments),
        "duration": result.duration,
    }


async def main() -> None:
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/audio/podcast_2speakers_1800s.wav"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    from src.core.config import config
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber
    from src.core.runtime import describe_runtime, detect_runtime

    runtime = detect_runtime()
    runtime.validate()
    print(f"[probe] runtime = {describe_runtime(runtime)}", flush=True)

    q = config.qwen3

    def make_tx() -> Qwen3DiarizeTranscriber:
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
        tx.embedding_model = q.embedding_model
        return tx

    print(f"[probe] building {count} transcriber instances...", flush=True)
    txs = [make_tx() for _ in range(count)]

    # serial init (LLM model load 互不干扰, 但避免同时 mmap 大文件)
    t_init_start = time.time()
    for i, tx in enumerate(txs):
        print(f"[probe] init tx[{i}] ...", flush=True)
        await tx.initialize()
    print(f"[probe] all initialized in {time.time() - t_init_start:.2f}s", flush=True)

    # barrier: 同时启动 transcribe
    print(f"[probe] kickoff {count} concurrent transcribe(s) on {audio_path}", flush=True)
    t_kick = time.time()
    tasks = [
        transcribe_one(tx, audio_path, f"single-proc-{i+1}")
        for i, tx in enumerate(txs)
    ]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        print(f"[probe] gather raised: {e}", flush=True)
        raise
    total_wall = time.time() - t_kick
    print(f"[probe] TOTAL_WALL={total_wall:.2f}s", flush=True)
    for r in results:
        if isinstance(r, Exception):
            print(f"[probe] TASK_FAILED: {type(r).__name__}: {r}", flush=True)
        else:
            print(
                f"[probe] TASK_OK {r['task_id']} wall={r['wall']:.2f}s "
                f"RTF={r['rtf']:.3f} speakers={r['speakers']} segs={r['segs']}",
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(main())
