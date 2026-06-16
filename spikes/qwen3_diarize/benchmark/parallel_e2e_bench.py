"""
真并行端到端 benchmark + 资源监控.

vs e2e_bench.py: 那个版本是同进程串行 ASR -> Diarize.
本脚本 subprocess fork 两个子进程同时跑 ASR 和 Diarize,
然后采集每个子进程 + 系统级的 CPU%/RSS, 输出资源占用报告.

无 sudo 拿不到 GPU/ANE 直接数据 — 用 ps/psutil 间接证据:
  - ASR 子进程 CPU% 低 + 完成快 = 算力在 Metal GPU 上 (ONNX encoder 是 CPU 上的)
  - Diarize 子进程 CPU% 高 + 多核 = CPU EP 在干活

用法:
    venv/bin/python benchmark/parallel_e2e_bench.py <audio> [--preset auto|D]
"""
import argparse
import json
import sys
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.merge import (
    filter_spurious_speakers,
    merge_asr_and_diarize,
    segments_to_rttm,
    segments_to_srt,
)


# 跟 e2e_bench.py 保持一致
PRESETS = {
    "auto": dict(
        segmentation_model="models/sherpa/pyannote-segmentation-3.0/model.onnx",
        embedding_model="models/sherpa/nemo-titanet-small/embedding.onnx",
        provider="cpu",
        num_speakers=-1,
    ),
    "D": dict(
        segmentation_model="models/sherpa/pyannote-segmentation-3.0/model.int8.onnx",
        embedding_model="models/sherpa/nemo-titanet-small/embedding.onnx",
        provider="cpu",
        num_speakers=2,
    ),
}


def proc_monitor(pid: int, samples: list, stop: threading.Event, interval: float = 0.5):
    """采样某个进程的 cpu%/rss, 每 interval 秒一次, 直到 stop 或进程消失."""
    try:
        p = psutil.Process(pid)
        p.cpu_percent(interval=None)  # warmup
    except psutil.NoSuchProcess:
        return
    while not stop.is_set():
        try:
            cpu = p.cpu_percent(interval=None)
            rss = p.memory_info().rss / 1024 / 1024
            samples.append((time.time(), cpu, rss))
        except psutil.NoSuchProcess:
            break
        time.sleep(interval)


def sys_monitor(samples: list, stop: threading.Event, interval: float = 0.5):
    """采样系统级 CPU% (每核 + 总和) 和已用内存."""
    psutil.cpu_percent(percpu=True)  # warmup
    while not stop.is_set():
        per_core = psutil.cpu_percent(percpu=True)
        mem_used = psutil.virtual_memory().used / 1024 / 1024
        samples.append((time.time(), per_core, mem_used))
        time.sleep(interval)


def summarize(label: str, samples: list):
    if not samples:
        print(f"  {label}: no samples", file=sys.stderr)
        return
    cpus = [s[1] for s in samples]
    rsss = [s[2] for s in samples]
    print(
        f"  {label:18s}: cpu% avg={sum(cpus)/len(cpus):5.0f} max={max(cpus):4.0f} "
        f"| rss avg={sum(rsss)/len(rsss):5.0f}MB peak={max(rsss):5.0f}MB "
        f"({len(samples)} samples)",
        file=sys.stderr,
    )


def summarize_sys(samples: list):
    if not samples:
        print(f"  system: no samples", file=sys.stderr)
        return
    totals = [sum(s[1]) for s in samples]   # 每个时间点 = 所有核 % 之和
    rsss = [s[2] for s in samples]
    print(
        f"  {'system total':18s}: cpu% avg={sum(totals)/len(totals):5.0f} max={max(totals):4.0f} "
        f"| mem avg={sum(rsss)/len(rsss):5.0f}MB peak={max(rsss):5.0f}MB "
        f"({len(samples)} samples)",
        file=sys.stderr,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument(
        "--preset",
        default="auto",
        choices=list(PRESETS.keys()),
        help="diarize 预设",
    )
    ap.add_argument(
        "--diarize-threads",
        type=int,
        default=4,
        help="并行时 diarize 用的 CPU 线程数 (留点给 ASR encoder 也跑 CPU)",
    )
    ap.add_argument("--cluster-threshold", type=float, default=0.9)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_name = Path(args.audio).stem

    asr_json = out_dir / f"{audio_name}.par_asr.json"
    dia_json_path = out_dir / f"{audio_name}.par_diarize.json"

    cfg = PRESETS[args.preset]
    # 注意:venv/bin/python 是 symlink 指向系统 python,通过 pyvenv.cfg 标 venv 上下文
    # resolve() 会 dereference symlink → 跑成系统 python 拿不到 venv 的包.
    # 必须用 absolute() 保留 symlink.
    py = str((Path("venv") / "bin" / "python").absolute())

    asr_cmd = [
        py,
        "benchmark/asr_bench.py",
        args.audio,
        "--language",
        args.language,
        "--out-json",
        str(asr_json),
    ]
    dia_cmd = [
        py,
        "benchmark/diarize_bench.py",
        args.audio,
        "--segmentation-model",
        cfg["segmentation_model"],
        "--embedding-model",
        cfg["embedding_model"],
        "--num-speakers",
        str(cfg["num_speakers"]),
        "--cluster-threshold",
        str(args.cluster_threshold),
        "--provider",
        cfg["provider"],
        "--num-threads",
        str(args.diarize_threads),
    ]

    print(f"[PAR] audio={args.audio}", file=sys.stderr)
    print(f"[PAR] preset={args.preset}, diarize_threads={args.diarize_threads}", file=sys.stderr)
    print(f"[PAR] Spawning ASR + Diarize subprocesses...", file=sys.stderr)

    # === Spawn 两个 subprocess 同时启动 ===
    t0 = time.time()
    asr_proc = subprocess.Popen(
        asr_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    dia_proc = subprocess.Popen(
        dia_cmd,
        stdout=open(dia_json_path, "wb"),
        stderr=subprocess.PIPE,
    )

    # === 启动监控线程 ===
    asr_samples: list = []
    dia_samples: list = []
    sys_samples: list = []
    stop_event = threading.Event()

    threads = [
        threading.Thread(target=proc_monitor, args=(asr_proc.pid, asr_samples, stop_event), daemon=True),
        threading.Thread(target=proc_monitor, args=(dia_proc.pid, dia_samples, stop_event), daemon=True),
        threading.Thread(target=sys_monitor, args=(sys_samples, stop_event), daemon=True),
    ]
    for t in threads:
        t.start()

    # === 等两个 subprocess 完成 ===
    asr_rc = asr_proc.wait()
    asr_done = time.time()
    dia_rc = dia_proc.wait()
    dia_done = time.time()

    stop_event.set()
    for t in threads:
        t.join(timeout=2)

    wall_total = max(asr_done, dia_done) - t0
    asr_elapsed = asr_done - t0
    dia_elapsed = dia_done - t0

    asr_stderr = asr_proc.stderr.read().decode(errors="ignore") if asr_proc.stderr else ""
    dia_stderr = dia_proc.stderr.read().decode(errors="ignore") if dia_proc.stderr else ""

    if asr_rc != 0:
        print(f"[PAR] ASR FAILED rc={asr_rc}\n{asr_stderr}", file=sys.stderr)
        sys.exit(1)
    if dia_rc != 0:
        print(f"[PAR] DIARIZE FAILED rc={dia_rc}\n{dia_stderr}", file=sys.stderr)
        sys.exit(1)

    # === 收 ASR / Diarize 结果 ===
    asr_data = json.loads(asr_json.read_text())
    dia_data = json.loads(dia_json_path.read_text())
    audio_dur = asr_data["duration"]
    asr_inner_rtf = asr_data["rtf"]
    dia_inner_rtf = dia_data["rtf"]

    # === Filter + Merge ===
    turns_raw = dia_data["turns"]
    turns_filtered = filter_spurious_speakers(
        turns_raw, min_speaker_total=2.0, min_speaker_share=0.01, audio_duration=audio_dur
    )
    spurious_dropped = len({t["speaker"] for t in turns_raw}) - len({t["speaker"] for t in turns_filtered})

    t_merge_start = time.time()
    segments = merge_asr_and_diarize(asr_data["text"], turns_filtered)
    t_merge = time.time() - t_merge_start

    # === Outputs ===
    json_out = out_dir / f"{audio_name}.par.e2e.json"
    json_out.write_text(json.dumps(
        {
            "audio": args.audio,
            "duration": audio_dur,
            "preset": args.preset,
            "diarize_threads": args.diarize_threads,
            "asr_text": asr_data["text"],
            "asr": {
                "subproc_elapsed": asr_elapsed,
                "subproc_rtf": asr_elapsed / audio_dur,
                "inner_rtf": asr_inner_rtf,
            },
            "diarize": {
                "subproc_elapsed": dia_elapsed,
                "subproc_rtf": dia_elapsed / audio_dur,
                "inner_rtf": dia_inner_rtf,
                "turns_raw_count": len(turns_raw),
                "turns_filtered_count": len(turns_filtered),
                "spurious_dropped": spurious_dropped,
                "speakers": sorted({t["speaker"] for t in turns_filtered}),
                "turns": turns_filtered,
            },
            "e2e": {
                "wall_clock": wall_total,
                "rtf_wall": wall_total / audio_dur,
                "merge_ms": t_merge * 1000,
            },
            "resource": {
                "asr_subproc": {
                    "samples": len(asr_samples),
                    "cpu_pct_avg": sum(s[1] for s in asr_samples) / max(len(asr_samples), 1),
                    "cpu_pct_max": max((s[1] for s in asr_samples), default=0),
                    "rss_mb_peak": max((s[2] for s in asr_samples), default=0),
                },
                "diarize_subproc": {
                    "samples": len(dia_samples),
                    "cpu_pct_avg": sum(s[1] for s in dia_samples) / max(len(dia_samples), 1),
                    "cpu_pct_max": max((s[1] for s in dia_samples), default=0),
                    "rss_mb_peak": max((s[2] for s in dia_samples), default=0),
                },
                "system": {
                    "samples": len(sys_samples),
                    "cpu_pct_avg_total": (
                        sum(sum(s[1]) for s in sys_samples) / max(len(sys_samples), 1)
                    ),
                    "cpu_pct_max_total": max((sum(s[1]) for s in sys_samples), default=0),
                    "mem_used_mb_peak": max((s[2] for s in sys_samples), default=0),
                },
            },
            "segments": [asdict(s) for s in segments],
        },
        ensure_ascii=False,
        indent=2,
    ))

    (out_dir / f"{audio_name}.par.srt").write_text(segments_to_srt(segments))
    (out_dir / f"{audio_name}.par.rttm").write_text(segments_to_rttm(segments, file_id=audio_name))

    # === 报告 ===
    print(file=sys.stderr)
    print(f"=== Parallel E2E ===", file=sys.stderr)
    print(f"  audio_dur          : {audio_dur:.2f}s", file=sys.stderr)
    print(f"  ASR subproc        : {asr_elapsed:.2f}s  (subproc rtf={asr_elapsed/audio_dur:.3f}, inner rtf={asr_inner_rtf:.3f})", file=sys.stderr)
    print(f"  Diarize subproc    : {dia_elapsed:.2f}s  (subproc rtf={dia_elapsed/audio_dur:.3f}, inner rtf={dia_inner_rtf:.3f})", file=sys.stderr)
    print(f"  Merge              : {t_merge*1000:.1f}ms", file=sys.stderr)
    print(f"  WALL CLOCK         : {wall_total:.2f}s  rtf={wall_total/audio_dur:.3f}", file=sys.stderr)
    print(f"  Speakers           : {sorted({s.speaker for s in segments})}  segments={len(segments)}", file=sys.stderr)
    if spurious_dropped > 0:
        print(f"  filter_spurious    : dropped {spurious_dropped} spurious cluster", file=sys.stderr)

    print(file=sys.stderr)
    print(f"=== Resource ===", file=sys.stderr)
    summarize("ASR subproc", asr_samples)
    summarize("Diarize subproc", dia_samples)
    summarize_sys(sys_samples)

    print(file=sys.stderr)
    print(f"  -> {json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
