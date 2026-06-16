"""分析单次 run 的 summary.json + powermetrics.log, 输出对比友好的 Markdown 行.

用法:
    venv/bin/python spikes/qwen3_mac_hw_accel/analyze_run.py <runs/tag dir> [更多 dir...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict


def parse_powermetrics(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    gpu = [float(m.group(1)) for m in re.finditer(r"GPU HW active residency:\s+([\d.]+)%", text)]
    ane = [float(m.group(1)) for m in re.finditer(r"ANE Power:\s+([\d.]+)\s*mW", text)]
    cpu = [float(m.group(1)) for m in re.finditer(r"CPU Power:\s+([\d.]+)\s*mW", text)]
    gpu_p = [float(m.group(1)) for m in re.finditer(r"GPU Power:\s+([\d.]+)\s*mW", text)]
    return {
        "gpu_residency_mean": sum(gpu) / max(len(gpu), 1),
        "gpu_residency_max": max(gpu) if gpu else 0,
        "ane_power_mean_mw": sum(ane) / max(len(ane), 1),
        "ane_power_max_mw": max(ane) if ane else 0,
        "cpu_power_mean_w": sum(cpu) / max(len(cpu), 1) / 1000,
        "cpu_power_max_w": (max(cpu) if cpu else 0) / 1000,
        "gpu_power_mean_w": sum(gpu_p) / max(len(gpu_p), 1) / 1000,
        "n_samples": len(gpu),
    }


KEY_STAGES = [
    ("asr.encoder.backend_onnx", "ASR enc.be"),
    ("asr.encoder.frontend_onnx", "ASR enc.fe"),
    ("asr.encoder.total", "ASR enc"),
    ("asr.llm_decode", "ASR llm.dec"),
    ("asr.run_total", "ASR total"),
    ("sherpa.pipeline.process", "sherpa.pipe"),
    ("sherpa.diarize.total", "sherpa.dia"),
    ("sherpa.embedding.compute", "sherpa.emb"),
    ("cluster_merge.apply", "cluster_m"),
    ("audio.load", "audio.load"),
]


def analyze(run_dir: Path) -> dict:
    sumf = run_dir / "summary.json"
    if not sumf.exists():
        return {"error": f"no summary.json in {run_dir}"}
    s = json.loads(sumf.read_text())
    pm = parse_powermetrics(run_dir / "powermetrics.log")
    return {"summary": s, "powermetrics": pm, "tag": s.get("tag", run_dir.name)}


def fmt_row(label: str, info: dict) -> str:
    s = info["summary"]
    pm = info["powermetrics"]
    tasks = s.get("tasks", {})
    parts = [f"## {label}  (mode={s['mode']}, t={s['num_threads']}, "
             f"sherpa={s['provider']}, asr_onnx={s['onnx_provider']}, "
             f"coreml_asr={s['coreml_asr_patch']})\n"]
    parts.append(f"- **TOTAL_WALL**: {s['wall_total']:.1f}s ({s['wall_total']/60:.2f}min)\n")
    parts.append(f"- **HW**: GPU residency mean={pm.get('gpu_residency_mean', 0):.1f}% "
                 f"max={pm.get('gpu_residency_max', 0):.1f}%, "
                 f"ANE mean={pm.get('ane_power_mean_mw', 0):.0f}mW max={pm.get('ane_power_max_mw', 0):.0f}mW, "
                 f"CPU mean={pm.get('cpu_power_mean_w', 0):.1f}W max={pm.get('cpu_power_max_w', 0):.1f}W\n")
    parts.append("\n| task | wall | dur | RTF | n_seg | " + " | ".join(s for _, s in KEY_STAGES) + " |\n")
    parts.append("|" + "---|" * (5 + len(KEY_STAGES)) + "\n")
    for label_t, info_t in tasks.items():
        meta = info_t.get("meta", {})
        totals = info_t.get("totals", {})
        row = (f"| {label_t} | {meta.get('wall', 0):.1f} | "
               f"{meta.get('duration', 0):.0f} | "
               f"{meta.get('rtf', 0):.3f} | "
               f"{meta.get('n_segments', 0)} | "
               + " | ".join(f"{totals.get(k, 0):.1f}" for k, _ in KEY_STAGES)
               + " |\n")
        parts.append(row)
    return "".join(parts)


def main():
    if len(sys.argv) < 2:
        print("usage: analyze_run.py <runs/tag> [more...]", file=sys.stderr)
        sys.exit(1)
    out = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        info = analyze(p)
        if "error" in info:
            print(info["error"], file=sys.stderr)
            continue
        out.append(fmt_row(info["tag"], info))
    print("\n".join(out))


if __name__ == "__main__":
    main()
