#!/usr/bin/env python3
# coding=utf-8
"""3060 远端 probe — 词级时间戳 (word_align MMS CTC-FA) 的 CUDA RTF + 共存验证.

量两件事:
1. ON vs OFF 端到端 RTF, 增量 = ON-OFF ≈ MMS 对齐部分 (词级时间戳的代价).
2. MMS ONNX CUDAExecutionProvider + llama.cpp CUDA (Qwen3 ASR) + ORT CUDA diarize
   三者进程内共存不 segfault (类比 sherpa CUDA build 撞 llama.cpp 的前车之鉴).

跑法 (远端项目根, 强制 cuda runtime + cuda encoder):
    FUNASR_RUNTIME=cuda FUNASR_QWEN3_ASR_ENCODER_PROVIDER=cuda \
        venv/bin/python scripts/_remote_word_align_rtf_probe.py
"""
import asyncio
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 强制 cuda runtime + cuda encoder (probe 专用, env 在 import config 前设)
os.environ.setdefault("FUNASR_RUNTIME", "cuda")
os.environ.setdefault("FUNASR_QWEN3_ASR_ENCODER_PROVIDER", "cuda")

AUDIO = "tests/fixtures/audio/podcast_2speakers_60s.wav"
MMS_MODEL = "models/qwen3_diarize/ctc_forced_aligner/model.onnx"
N_REPEAT = 2


async def main():
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
        word_align_enabled=True,
        word_align_provider="cuda",  # CUDAExecutionProvider
        word_align_model_path=MMS_MODEL,
        word_align_language="chi",
        word_align_batch_size=q.word_align_batch_size,
    )
    tx.embedding_model = q.embedding_model

    print(f"[probe] runtime=cuda asr_encoder=cuda word_align_provider=cuda model={MMS_MODEL}")
    await tx.initialize()

    async def run(label, task_id):
        t0 = time.time()
        res, raw = await tx.transcribe(AUDIO, task_id, output_format="json", language="chi")
        wall = time.time() - t0
        return wall, res, raw

    # warm-up OFF: 建 ASR engine + diarize backend (排除一次性加载)
    tx.word_align_enabled = False
    w0, res, raw = await run("warm-off", "warm-off")
    dur = res.duration
    print(f"[probe] audio dur={dur:.2f}s  warm-off wall={w0:.2f}s (含模型加载)")

    # OFF baseline
    off_walls = []
    for i in range(N_REPEAT):
        w, res, raw = await run("off", f"off{i}")
        off_walls.append(w)
    print(f"[probe] OFF  walls={[round(x,2) for x in off_walls]}  RTF={[round(x/dur,4) for x in off_walls]}")

    # ON warm: 第一次 ON 含 MMS session build
    tx.word_align_enabled = True
    w_on_warm, res, raw = await run("on-warm", "on-warm")
    print(f"[probe] ON_warm wall={w_on_warm:.2f}s (含 MMS ONNX session build)")
    print(f"[probe] ON_warm word_align stats={raw['word_align']}")

    # ON steady
    on_walls = []
    for i in range(N_REPEAT):
        w, res, raw = await run("on", f"on{i}")
        on_walls.append(w)
    print(f"[probe] ON   walls={[round(x,2) for x in on_walls]}  RTF={[round(x/dur,4) for x in on_walls]}")

    # 结果汇总
    off_rtf = sum(off_walls) / len(off_walls) / dur
    on_rtf = sum(on_walls) / len(on_walls) / dur
    n_words = sum(len(s.words) for s in res.segments if s.words)
    print("=" * 60)
    print(f"[RESULT] audio={dur:.1f}s segments={len(res.segments)} words={n_words}")
    print(f"[RESULT] OFF RTF={off_rtf:.4f}  ON RTF={on_rtf:.4f}  增量Δ={on_rtf-off_rtf:+.4f}")
    print(f"[RESULT] word_align stats={raw['word_align']}")
    print(f"[RESULT] ✅ MMS CUDA + llama.cpp CUDA + ORT diarize 共存未 segfault (跑完未崩)")


if __name__ == "__main__":
    asyncio.run(main())
