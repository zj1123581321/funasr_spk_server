"""Ad-hoc: Qwen3 N=2 真并发 — 复测 N=3 spk 异常的两个 audio (44min-4spk + 60min-2spk).

实验设计:
  N=3 时 44min-4spk 测出 5 spk, 60min-2spk 测出 4 spk (over-detect).
  N=2 跑同样 2 个 audio:
    - 若 spk=4 和 2 (正确): 证明 N=3 并发压力造成 spk over-detect
    - 若 spk=5 和 4 (跟 N=3 一致): audio 本身的 sherpa diarize 问题, 跟并发无关

监控指标同 N=3 (bench_n3_concurrency.py).

运行:
  venv/bin/python spikes/qwen3_silence_align/scripts/bench_n2_concurrency.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# 复测 N=3 异常的两个 audio
AUDIOS = [
    ("44min-4spk", PROJECT_ROOT / "tmp_long_audio/eval_set/audio_4spk.m4a"),
    ("60min-2spk", PROJECT_ROOT / "tmp_long_audio/eval_set/audio_2spk_60min.mp3"),
]


def all_python_procs(parent_pid: int) -> list[psutil.Process]:
    out = []
    try:
        parent = psutil.Process(parent_pid)
        out.append(parent)
        for child in parent.children(recursive=True):
            try:
                if "python" in (child.name() or "").lower():
                    out.append(child)
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    return out


async def sampler_loop(stop_evt: asyncio.Event, samples: list, interval: float = 5.0):
    parent_pid = os.getpid()
    for p in all_python_procs(parent_pid):
        try:
            p.cpu_percent(None)
        except Exception:
            pass
    await asyncio.sleep(0.5)
    t_start = time.time()
    while not stop_evt.is_set():
        procs = all_python_procs(parent_pid)
        snap = {"t": time.time() - t_start, "n_procs": len(procs), "procs": []}
        for p in procs:
            try:
                cpu = p.cpu_percent(None)
                rss_mb = p.memory_info().rss / (1024 * 1024)
                snap["procs"].append({"pid": p.pid, "cpu": cpu, "rss_mb": rss_mb})
            except psutil.NoSuchProcess:
                continue
        snap["total_cpu"] = sum(x["cpu"] for x in snap["procs"])
        snap["total_rss_mb"] = sum(x["rss_mb"] for x in snap["procs"])
        samples.append(snap)
        print(
            f"  [sample t={snap['t']:5.0f}s] procs={snap['n_procs']} "
            f"total_rss={snap['total_rss_mb']:7.0f}MB",
            flush=True,
        )
        try:
            await asyncio.wait_for(stop_evt.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def one_task(transcriber, name: str, audio: Path):
    t0 = time.time()
    result = await transcriber.transcribe(
        audio_path=str(audio),
        task_id=f"n2-{name}",
        progress_callback=None,
        output_format="json",
    )
    wall = time.time() - t0
    tres, raw = result
    rtf = tres.processing_time / tres.duration if tres.duration else 0
    return {
        "name": name,
        "wall_s": wall,
        "duration_s": tres.duration,
        "processing_time_s": tres.processing_time,
        "rtf_server": rtf,
        "rtf_wall": wall / tres.duration if tres.duration else 0,
        "speakers": tres.speakers,           # 保留原始 list (不止数量)
        "speakers_count": len(tres.speakers),
        "segments": len(tres.segments),
    }


async def main():
    for name, p in AUDIOS:
        assert p.exists(), f"{name} 缺: {p}"

    from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber
    pool = Qwen3PoolTranscriber(pool_size=2)
    print(f"[main] 启动 N=2 worker pool, 加载模型 (~10-20s)...", flush=True)
    await pool.initialize()
    print(f"[main] pool ready, 开始 2 个并发任务", flush=True)

    samples: list = []
    stop_evt = asyncio.Event()
    sampler_task = asyncio.create_task(sampler_loop(stop_evt, samples))

    t0 = time.time()
    results = await asyncio.gather(*[one_task(pool, n, p) for n, p in AUDIOS])
    total_wall = time.time() - t0

    stop_evt.set()
    await sampler_task

    print("\n" + "=" * 70)
    print("N=2 真并发 spk 复测 (M1 Max, qwen3_pool_size=2, num_threads=4)")
    print("=" * 70)
    print(f"总 wall (gather): {total_wall:.1f}s ({total_wall/60:.2f} min)")
    print()
    print(f"{'name':<12} {'audio_s':>8} {'wall_s':>8} {'proc_s':>8} {'RTF_srv':>8} {'spk':>4} {'spk_list':>16} {'segs':>5}")
    expected = {"16min-1spk": 1, "44min-4spk": 4, "60min-2spk": 2}
    for r in results:
        exp = expected.get(r["name"], "?")
        flag = "✓" if r["speakers_count"] == exp else f"⚠️ (exp {exp})"
        print(
            f"{r['name']:<12} {r['duration_s']:8.1f} {r['wall_s']:8.1f} "
            f"{r['processing_time_s']:8.1f} {r['rtf_server']:8.3f} "
            f"{r['speakers_count']:4d} {str(r['speakers']):>16} {r['segments']:5d}  {flag}"
        )

    sum_proc = sum(r["processing_time_s"] for r in results)
    print(f"\n串行 baseline (sum processing_time): {sum_proc:.1f}s")
    speedup = sum_proc / total_wall
    print(f"并发 speedup: {speedup:.2f}x (理论 N=2 = 2.0x, 效率 {speedup/2*100:.0f}%)")

    if samples:
        max_rss = max(s["total_rss_mb"] for s in samples)
        avg_rss = sum(s["total_rss_mb"] for s in samples) / len(samples)
        max_procs = max(s["n_procs"] for s in samples)
        print(f"\n资源占用:")
        print(f"  python procs 峰值: {max_procs}")
        print(f"  RSS 总和:  max={max_rss:.0f} MB ({max_rss/1024:.1f} GB) avg={avg_rss:.0f} MB")

    # 跟 N=3 对比
    print(f"\n=== 跟 N=3 对比 (spk 数) ===")
    n3_spk = {"44min-4spk": 5, "60min-2spk": 4}
    for r in results:
        n3 = n3_spk.get(r["name"], "?")
        exp = expected.get(r["name"], "?")
        verdict = ""
        if r["speakers_count"] == exp:
            verdict = "✅ N=2 恢复正确, N=3 的 over-detect 是并发压力导致"
        elif r["speakers_count"] == n3:
            verdict = "❌ N=2 同样 over-detect, audio 本身 sherpa diarize 问题, 跟并发无关"
        else:
            verdict = "⚠️ 既非 expected 也非 N=3, 三态都不一致"
        print(f"  {r['name']}: N=1 expected={exp} | N=2 实测={r['speakers_count']} | N=3={n3}  → {verdict}")

    import json
    out_path = PROJECT_ROOT / "spikes/qwen3_silence_align/scripts/_bench_n2_result.json"
    with open(out_path, "w") as f:
        json.dump({
            "total_wall_s": total_wall,
            "sum_processing_time_s": sum_proc,
            "speedup": speedup,
            "tasks": results,
            "samples": samples,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nJSON: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
