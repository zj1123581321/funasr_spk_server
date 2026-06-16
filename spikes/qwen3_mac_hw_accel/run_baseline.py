"""Spike: profiling baseline runner.

按 N=1 / N=2 模式 spawn profile_worker 子进程, 同时启 powermetrics 抓硬件 utilization,
跑完合并 timing report.

用法:
    venv/bin/python spikes/qwen3_mac_hw_accel/run_baseline.py \
        --mode n1|n2 \
        --tag baseline_cpu_t8 \
        --num-threads 8 \
        --provider cpu \
        --onnx-provider CPU \
        [--enable-coreml-asr-patch]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPIKE_ROOT = Path(__file__).resolve().parent
EVAL = PROJECT_ROOT / "tmp_long_audio" / "eval_set"
AUDIO_1SPK = EVAL / "audio_1spk_real.m4a"          # 16min
AUDIO_4SPK = EVAL / "audio_4spk.m4a"               # 44min


def start_powermetrics(out_path: Path) -> subprocess.Popen:
    """Spawn powermetrics 后台采样 GPU/ANE/CPU active residency, 1Hz, 输出到文件."""
    cmd = [
        "sudo", "-n", "powermetrics",
        "--samplers", "gpu_power,ane_power,cpu_power",
        "--hide-cpu-duty-cycle",
        "-i", "1000",
        "-f", "text",
    ]
    out_fh = open(out_path, "w")
    proc = subprocess.Popen(cmd, stdout=out_fh, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    print(f"[main] powermetrics PID {proc.pid} -> {out_path}", flush=True)
    return proc


def stop_powermetrics(proc: subprocess.Popen):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception as e:
        print(f"[main] powermetrics stop failed: {e}", flush=True)
        try:
            proc.kill()
        except Exception:
            pass


def spawn_worker(audio: Path, label: str, out_json: Path, num_threads: int,
                 provider: str, onnx_provider: str, coreml_asr_patch: bool,
                 log_path: Path, coreml_units: str = "ALL",
                 coreml_format: str = "MLProgram",
                 coreml_static_shapes: bool = False,
                 coreml_only_frontend: bool = False) -> subprocess.Popen:
    cmd = [
        str(PROJECT_ROOT / "venv" / "bin" / "python"),
        str(SPIKE_ROOT / "profile_worker.py"),
        str(audio), label,
        "--out-json", str(out_json),
        "--num-threads", str(num_threads),
        "--provider", provider,
        "--onnx-provider", onnx_provider,
    ]
    if coreml_asr_patch:
        cmd.append("--enable-coreml-asr-patch")
        cmd += ["--coreml-units", coreml_units, "--coreml-format", coreml_format]
        if coreml_static_shapes:
            cmd.append("--coreml-static-shapes")
        if coreml_only_frontend:
            cmd.append("--coreml-only-frontend")
    env = os.environ.copy()
    env.update({
        "TMPDIR": "/tmp",
        "DYLD_LIBRARY_PATH": str(PROJECT_ROOT / "src/core/vendor/qwen_asr_gguf/inference/bin"),
        "PYTHONUNBUFFERED": "1",
    })
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env,
                            stdout=log_fh, stderr=subprocess.STDOUT)
    print(f"[main] worker[{label}] PID {proc.pid} -> log {log_path}", flush=True)
    return proc, log_fh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["n1", "n2"], required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--provider", type=str, default="cpu")
    p.add_argument("--onnx-provider", type=str, default="CPU")
    p.add_argument("--enable-coreml-asr-patch", action="store_true")
    p.add_argument("--coreml-units", default="ALL",
                   choices=["ALL", "CPUAndGPU", "CPUOnly", "CPUAndNeuralEngine"])
    p.add_argument("--coreml-format", default="MLProgram",
                   choices=["MLProgram", "NeuralNetwork"])
    p.add_argument("--coreml-static-shapes", action="store_true")
    p.add_argument("--coreml-only-frontend", action="store_true")
    p.add_argument("--audio-pick", choices=["1spk", "4spk", "both"], default="both",
                   help="n1 模式选哪个音频 (both 等同 4spk)")
    p.add_argument("--no-powermetrics", action="store_true")
    args = p.parse_args()

    runs_root = SPIKE_ROOT / "runs" / args.tag
    runs_root.mkdir(parents=True, exist_ok=True)

    pm_path = runs_root / "powermetrics.log"
    pm_proc = None
    if not args.no_powermetrics:
        pm_proc = start_powermetrics(pm_path)
        time.sleep(2)

    started_at = time.time()
    workers = []  # list of (proc, log_fh, label, json_path)
    if args.mode == "n1":
        audio = AUDIO_4SPK if args.audio_pick != "1spk" else AUDIO_1SPK
        label = "1spk-16min" if audio == AUDIO_1SPK else "4spk-44min"
        json_path = runs_root / f"{label}.json"
        log_path = runs_root / f"{label}.log"
        proc, fh = spawn_worker(audio, label, json_path, args.num_threads,
                                args.provider, args.onnx_provider,
                                args.enable_coreml_asr_patch, log_path,
                                coreml_units=args.coreml_units,
                                coreml_format=args.coreml_format,
                                coreml_static_shapes=args.coreml_static_shapes,
                                coreml_only_frontend=args.coreml_only_frontend)
        workers.append((proc, fh, label, json_path))
    else:
        for audio, label in [(AUDIO_1SPK, "1spk-16min"), (AUDIO_4SPK, "4spk-44min")]:
            json_path = runs_root / f"{label}.json"
            log_path = runs_root / f"{label}.log"
            proc, fh = spawn_worker(audio, label, json_path, args.num_threads,
                                    args.provider, args.onnx_provider,
                                    args.enable_coreml_asr_patch, log_path,
                                    coreml_units=args.coreml_units,
                                    coreml_format=args.coreml_format,
                                    coreml_static_shapes=args.coreml_static_shapes,
                                    coreml_only_frontend=args.coreml_only_frontend)
            workers.append((proc, fh, label, json_path))

    # wait all
    rc_map = {}
    for proc, fh, label, _ in workers:
        rc = proc.wait()
        fh.close()
        rc_map[label] = rc
        print(f"[main] worker[{label}] exited rc={rc}", flush=True)

    wall = time.time() - started_at

    if pm_proc:
        stop_powermetrics(pm_proc)

    # 合并报告
    summary = {
        "tag": args.tag,
        "mode": args.mode,
        "num_threads": args.num_threads,
        "provider": args.provider,
        "onnx_provider": args.onnx_provider,
        "coreml_asr_patch": args.enable_coreml_asr_patch,
        "wall_total": wall,
        "tasks": {},
    }
    for _, _, label, json_path in workers:
        if json_path.exists():
            with open(json_path) as f:
                d = json.load(f)
            summary["tasks"][label] = {
                "meta": d.get("meta", {}),
                "totals": d.get("totals", {}),
                "wall": d.get("wall", 0),
            }
        else:
            summary["tasks"][label] = {"error": "no json", "rc": rc_map.get(label)}
    summary_path = runs_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印简表
    print("\n" + "=" * 80, flush=True)
    print(f"TAG={args.tag} MODE={args.mode} NUM_THREADS={args.num_threads} "
          f"PROV={args.provider} ASR_ONNX={args.onnx_provider} "
          f"COREML_ASR_PATCH={args.enable_coreml_asr_patch}", flush=True)
    print(f"TOTAL_WALL={wall:.1f}s ({wall/60:.1f}min)", flush=True)
    for label, info in summary["tasks"].items():
        meta = info.get("meta", {})
        print(f"  [{label}] wall={meta.get('wall', 0):.1f}s "
              f"duration={meta.get('duration', 0):.1f}s "
              f"rtf={meta.get('rtf', 0):.3f} "
              f"segs={meta.get('n_segments', 0)} "
              f"asr_text_len={meta.get('asr_text_len', 0)}", flush=True)
        totals = info.get("totals", {})
        for stage, t in sorted(totals.items(), key=lambda kv: -kv[1])[:8]:
            print(f"     {stage:<38} {t:>8.2f}s", flush=True)
    print(f"\nsummary -> {summary_path}", flush=True)
    print(f"powermetrics -> {pm_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
