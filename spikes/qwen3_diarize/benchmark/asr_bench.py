"""
ASR benchmark — 跑一遍 ASR,记 RTF/RSS,可选写 JSON.

用法:
    venv/bin/python benchmark/asr_bench.py tests/fixtures/audio/晚点聊-sample-2person-5min.mp3
    venv/bin/python benchmark/asr_bench.py <audio> --out-json output/asr.json
"""
import sys
import json
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.asr import run_asr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    print(f"[ASR] input audio: {args.audio}", file=sys.stderr)
    r = run_asr(args.audio, language=args.language)

    print(
        f"[ASR] duration={r.duration:.2f}s elapsed={r.elapsed:.2f}s "
        f"RTF={r.rtf:.3f} peak_rss={r.peak_rss_mb:.0f}MB rss_delta={r.rss_delta_mb:.0f}MB",
        file=sys.stderr,
    )
    print(f"[ASR] text_len={len(r.text)} word_items={len(r.items)}", file=sys.stderr)
    head = r.text[:160].replace("\n", " ")
    print(f"[ASR] text preview: {head}", file=sys.stderr)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "audio": args.audio,
            "duration": r.duration,
            "elapsed": r.elapsed,
            "rtf": r.rtf,
            "peak_rss_mb": r.peak_rss_mb,
            "rss_delta_mb": r.rss_delta_mb,
            "text": r.text,
            "items": [{"text": it.text, "start": it.start, "end": it.end} for it in r.items],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"[ASR] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
