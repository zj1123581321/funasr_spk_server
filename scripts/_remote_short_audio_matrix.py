"""短音频 × 多人数 diarize parity 矩阵 — sherpa vs ort_cuda spk_count 对比.

背景: 2026-06-10 ort_cuda 短音频 under-detect 修复 (sherpa pipeline 忠实移植
+ TitaNet embs 输出修复) 的验收项之一. 从标定评测集长音频切 1/5/10min 片段,
双 backend 只跑 diarize (不跑 ASR, 快). 切片的真实人数没有逐段人工标定,
以 sherpa (Mac 生产路径同款) 为参照基准.

parity 判定用 **后处理之后** 的人数 (filter_spurious + cluster_merge, 即生产
serve 的语义): raw 簇数在嘈杂多人内容上本来就靠后处理收敛, 两套数值实现的
raw 簇数逐一相等不是现实的 bar (raw 数值仅打印参考).

用法 (远端 cuda dev box, LD_LIBRARY_PATH 见 scripts/_remote_run_provider.sh):
  venv/bin/python scripts/_remote_short_audio_matrix.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.config import config
from src.core.qwen3.diarize import run_diarization
from src.core.qwen3.diarize_ort import reset_session_cache, run_diarization_ort_cuda

# (label, 源音频, 全片真实人数 — 仅参考, 切片可能少于全片)
SOURCES = [
    ("1spk", "tmp_long_audio/eval_set/audio_1spk_real.m4a", 1),
    ("2spk", "tmp_long_audio/eval_set/audio_2spk_60min.mp3", 2),
    ("4spk", "tmp_long_audio/multi_speaker_test/podcast_4spk.m4a", 4),
    ("6spk", "tmp_long_audio/eval_set/audio_6spk_60min.m4a", 6),
]
CUT_SECONDS = [60, 300, 600]
CUT_OFFSET = 120  # 跳过片头音乐/介绍


def _cut(src: str, offset: int, duration: int, out: Path) -> None:
    """ffmpeg 切片转 16k mono wav. 两 backend 吃同一份 wav, parity 对比公平."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(offset), "-t", str(duration), "-i", src,
            "-ac", "1", "-ar", "16000", str(out),
        ],
        check=True,
    )


def _spk_count(turns: list[dict]) -> int:
    return len({t["speaker"] for t in turns})


def _post_process(turns: list[dict], wav: str, duration: float, extractor_fn) -> list[dict]:
    """生产 serve 语义的人数收敛: filter_spurious + cluster_merge (turn 级)."""
    from src.core.qwen3.merge import filter_spurious_speakers
    from src.core.qwen3_transcriber import apply_cluster_centroid_merge_to_turns

    out = filter_spurious_speakers(turns, audio_duration=duration)
    out, _ = apply_cluster_centroid_merge_to_turns(
        out, wav, config.qwen3, extractor_fn=extractor_fn
    )
    return out


def main() -> None:
    from src.core.qwen3_transcriber import build_embedding_extractor_fn

    extractor_fn = build_embedding_extractor_fn(config.qwen3)
    kw = dict(
        segmentation_model=config.qwen3.segmentation_model,
        embedding_model=config.qwen3.embedding_model,
        num_speakers=None,
        cluster_threshold=config.qwen3.cluster_threshold,
    )
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for label, src, full_n in SOURCES:
            if not Path(src).exists():
                print(f"[skip] {label}: {src} 不存在", flush=True)
                continue
            for dur in CUT_SECONDS:
                wav = Path(td) / f"{label}_{dur}s.wav"
                _cut(src, CUT_OFFSET, dur, wav)

                t0 = time.time()
                sherpa = run_diarization(
                    str(wav), num_threads=4, provider="cpu", **kw
                )
                t1 = time.time()
                reset_session_cache()
                ort = run_diarization_ort_cuda(str(wav), **kw)
                t2 = time.time()

                raw_s, raw_o = _spk_count(sherpa), _spk_count(ort)
                ns = _spk_count(_post_process(sherpa, str(wav), dur, extractor_fn))
                no = _spk_count(_post_process(ort, str(wav), dur, extractor_fn))
                ok = "OK " if ns == no else "MISMATCH"
                rows.append((label, dur, ns, no, ok))
                print(
                    f"[{ok}] {label} {dur:>4}s (全片 {full_n} 人): "
                    f"post sherpa={ns} ort_cuda={no} "
                    f"(raw {raw_s}/{raw_o}, sherpa {t1-t0:.0f}s / ort {t2-t1:.0f}s)",
                    flush=True,
                )

    mismatches = [r for r in rows if r[4] != "OK "]
    print(f"\n===== {len(rows)} 组, post-process mismatch {len(mismatches)} =====")
    for r in mismatches:
        print(f"  {r[0]} {r[1]}s: sherpa={r[2]} ort={r[3]}")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
