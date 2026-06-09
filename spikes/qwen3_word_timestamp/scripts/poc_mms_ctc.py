#!/usr/bin/env python3
# coding=utf-8
"""PoC-MMS: MMS CTC Forced Aligner (轻量多语种) 验证 — deskpai ONNX fork

ctc-forced-aligner 1.0.2 (deskpai fork, ONNX 后端, MMS-based, 走 onnxruntime CPU).
多语种 + uroman 罗马化, 内置 30s+2s overlap 切块 (generate_emissions). 中文 language="chi".

目标: 中英混排精度 + RTF, 跟 Qwen aligner (PoC-A) / fa-zh (PoC-B) 头对头.

用法:
    venv/bin/python spikes/qwen3_word_timestamp/scripts/poc_mms_ctc.py \
        --audio tests/fixtures/audio/podcast_2speakers_60s.wav \
        --golden tests/fixtures/golden/podcast_2speakers_60s.golden.json \
        --language chi \
        --out spikes/qwen3_word_timestamp/outputs/poc_mms_60s.json
"""
import argparse
import json
import time
from pathlib import Path


def load_reference_text(golden_path: Path) -> str:
    data = json.loads(golden_path.read_text(encoding="utf-8"))
    return "".join(seg["text"] for seg in data["segments"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--golden", default=None)
    ap.add_argument("--text", default=None)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--language", default="chi", help="中文 chi / 英文 eng / 日 jpn ...")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.text_file:
        ref_text = Path(args.text_file).read_text(encoding="utf-8").replace("\n", " ").strip()
    elif args.text:
        ref_text = args.text
    else:
        ref_text = load_reference_text(Path(args.golden))

    from ctc_forced_aligner import (
        AlignmentSingleton,
        load_audio,
        generate_emissions,
        preprocess_text,
        get_alignments,
        get_spans,
        postprocess_results,
    )

    t_load0 = time.time()
    aln = AlignmentSingleton()  # 首次下 ONNX 到 ~/ctc_forced_aligner/model.onnx
    model = aln.alignment_model
    tokenizer = aln.alignment_tokenizer
    t_load = time.time() - t_load0
    print(f"[PoC-MMS] 模型加载 {t_load:.2f}s (ONNX CPU)")

    waveform = load_audio(args.audio)
    dur = len(waveform) / 16000.0
    print(f"[PoC-MMS] audio dur={dur:.2f}s text_len={len(ref_text)}字 lang={args.language}")

    t0 = time.time()
    emissions, stride = generate_emissions(model, waveform, batch_size=args.batch_size)
    tokens_starred, text_starred = preprocess_text(ref_text, romanize=True, language=args.language)
    segments, scores, blank_token = get_alignments(emissions, tokens_starred, tokenizer)
    spans = get_spans(tokens_starred, segments, blank_token)
    word_ts = postprocess_results(text_starred, spans, stride, scores)
    elapsed = time.time() - t0
    rtf = elapsed / dur if dur > 0 else None

    items = [{"text": w["text"], "start": w["start"], "end": w["end"]} for w in word_ts]
    n = len(items)
    print(f"[PoC-MMS] 对齐耗时 {elapsed:.2f}s  RTF={rtf}  units={n}")
    print("[PoC-MMS] 前 15 单元 (text: start-end s):")
    for it in items[:15]:
        print(f"    {it['text']!r}: {it['start']:.3f}-{it['end']:.3f}")

    Path(args.out).write_text(
        json.dumps(
            {
                "audio": args.audio, "duration_sec": dur, "text_len": len(ref_text),
                "language": args.language, "load_sec": t_load, "align_sec": elapsed,
                "rtf": rtf, "n_units": n, "items": items, "reference_text": ref_text,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[PoC-MMS] 结果落 {args.out}")


if __name__ == "__main__":
    main()
