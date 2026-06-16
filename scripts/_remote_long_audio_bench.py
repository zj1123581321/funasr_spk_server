"""3060 dev 上长音频 RTF benchmark.

用法:
  python scripts/_remote_long_audio_bench.py <duration_label> <num_threads>

会读取 tests/fixtures/audio/podcast_2speakers_<duration_label>.wav,
调 Qwen3DiarizeTranscriber.transcribe(audio_path),记录 wall RTF / encoder RTF / diarize 耗时。
"""
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


async def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "300s"
    if len(sys.argv) > 2:
        os.environ["FUNASR_QWEN3_NUM_THREADS"] = sys.argv[2]

    os.environ.setdefault("FUNASR_DEFAULT_ENGINE", "qwen3")
    os.environ.setdefault("FUNASR_QWEN3_PROVIDER", "cuda")
    os.environ.setdefault("FUNASR_QWEN3_ASR_ENCODER_PROVIDER", "cuda")

    audio_path = ROOT / f"tests/fixtures/audio/podcast_2speakers_{label}.wav"
    if not audio_path.exists():
        sys.exit(f"audio not found: {audio_path}")

    import soundfile as sf
    info = sf.info(str(audio_path))
    duration = info.frames / info.samplerate
    print(f"[bench] audio={audio_path.name} duration={duration:.2f}s "
          f"num_threads={os.environ.get('FUNASR_QWEN3_NUM_THREADS','?')}")

    from src.core.qwen3_transcriber import get_qwen3_transcriber

    t = get_qwen3_transcriber()

    async def noop(*a, **kw): return None

    t0 = time.time()
    result, raw = await t.transcribe(
        audio_path=str(audio_path),
        task_id="bench",
        progress_callback=noop,
        output_format="json",
    )
    wall = time.time() - t0
    rtf = wall / duration

    n_segs = len(result.segments) if hasattr(result, "segments") else "?"
    n_spk = len(result.speakers) if hasattr(result, "speakers") else "?"
    print(f"[bench] wall={wall:.2f}s rtf={rtf:.3f} segments={n_segs} speakers={n_spk}")


if __name__ == "__main__":
    asyncio.run(main())
