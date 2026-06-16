"""Ad-hoc: Qwen3 N=3 真并发实测 (16min/44min/60min 差异化 audio).

不入 git, 一次性 perf 测试. 直接调 Qwen3PoolTranscriber, 绕过 server.

监控:
  - 总 wall (gather 整体)
  - 各 task wall / processing_time / RTF / speakers / segments
  - 每 5s 采样: 主进程 + 所有 python 子进程 CPU% + RSS
  - 输出 timeseries 总结 (峰值 / 均值 / wall 内累计)

运行:
  venv/bin/python spikes/qwen3_silence_align/scripts/bench_n3_concurrency.py
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


AUDIOS = [
    ("16min-1spk", PROJECT_ROOT / "tmp_long_audio/eval_set/audio_1spk_real.m4a"),
    ("44min-4spk", PROJECT_ROOT / "tmp_long_audio/eval_set/audio_4spk.m4a"),
    ("60min-2spk", PROJECT_ROOT / "tmp_long_audio/eval_set/audio_2spk_60min.mp3"),
]


def all_python_procs(parent_pid: int) -> list[psutil.Process]:
    """主进程 + 全部 python 子进程 (含 worker subprocess)."""
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
    """每 interval 采样 cpu% + rss, 写入 samples."""
    parent_pid = os.getpid()
    # 先 cpu_percent 预热 (第一次 0)
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
            f"total_cpu={snap['total_cpu']:6.1f}% total_rss={snap['total_rss_mb']:7.0f}MB",
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
        task_id=f"n3-{name}",
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
        "speakers": len(tres.speakers),
        "segments": len(tres.segments),
    }


async def main():
    for name, p in AUDIOS:
        assert p.exists(), f"{name} 缺: {p}"

    from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber
    pool = Qwen3PoolTranscriber(pool_size=3)
    print(f"[main] 启动 N=3 worker pool, 加载模型 (~10-20s)...", flush=True)
    await pool.initialize()
    print(f"[main] pool ready, 开始 3 个并发任务", flush=True)

    samples: list = []
    stop_evt = asyncio.Event()
    sampler_task = asyncio.create_task(sampler_loop(stop_evt, samples))

    t0 = time.time()
    results = await asyncio.gather(*[one_task(pool, n, p) for n, p in AUDIOS])
    total_wall = time.time() - t0

    stop_evt.set()
    await sampler_task

    print("\n" + "=" * 70)
    print(f"N=3 真并发结果 (M1 Max, qwen3_pool_size=3, num_threads=4)")
    print("=" * 70)
    print(f"总 wall (gather): {total_wall:.1f}s ({total_wall/60:.2f} min)")
    print()
    print(f"{'name':<12} {'audio_s':>8} {'wall_s':>8} {'proc_s':>8} {'RTF_srv':>8} {'RTF_wall':>9} {'spk':>4} {'segs':>5}")
    for r in results:
        print(
            f"{r['name']:<12} {r['duration_s']:8.1f} {r['wall_s']:8.1f} "
            f"{r['processing_time_s']:8.1f} {r['rtf_server']:8.3f} "
            f"{r['rtf_wall']:9.3f} {r['speakers']:4d} {r['segments']:5d}"
        )

    # 串行 baseline 推算
    sum_proc = sum(r["processing_time_s"] for r in results)
    print(f"\n串行 baseline (sum processing_time): {sum_proc:.1f}s ({sum_proc/60:.2f} min)")
    speedup = sum_proc / total_wall
    print(f"并发 speedup: {speedup:.2f}x (理论上限 N=3 = 3.0x)")
    print(f"  节省: {sum_proc - total_wall:.1f}s ({(sum_proc - total_wall)/sum_proc*100:.0f}%)")

    # 资源时序汇总
    if samples:
        max_cpu = max(s["total_cpu"] for s in samples)
        avg_cpu = sum(s["total_cpu"] for s in samples) / len(samples)
        max_rss = max(s["total_rss_mb"] for s in samples)
        avg_rss = sum(s["total_rss_mb"] for s in samples) / len(samples)
        max_procs = max(s["n_procs"] for s in samples)
        print(f"\n资源占用 (主+全部 python 子进程, 采样间隔 5s, {len(samples)} 个采样点):")
        print(f"  python procs 峰值: {max_procs} (主 1 + 最多 {max_procs - 1} worker)")
        print(f"  CPU 累计%: max={max_cpu:.1f}% avg={avg_cpu:.1f}%  "
              f"(M1 Max 10 core 上限 1000%)")
        print(f"  RSS 总和:  max={max_rss:.0f} MB avg={avg_rss:.0f} MB  "
              f"({max_rss/1024:.1f} GB)")

    # dump JSON
    import json
    out_path = PROJECT_ROOT / "spikes/qwen3_silence_align/scripts/_bench_n3_result.json"
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
