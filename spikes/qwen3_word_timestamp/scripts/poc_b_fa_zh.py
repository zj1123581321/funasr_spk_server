#!/usr/bin/env python3
# coding=utf-8
"""PoC-B: FunASR fa-zh 轻量 forced aligner 验证

目标:验证"ASR 已有全文 → fa-zh 吃 (audio, 全文) → 字符级时间戳"这条轻量路线的
feasibility,量 RTF + 目测中文准度。fa-zh = iic/speech_timestamp_prediction,
38MB,CPU 友好,首次跑自动从 modelscope 下。

用法:
    venv/bin/python spikes/qwen3_word_timestamp/scripts/poc_b_fa_zh.py \
        --audio tests/fixtures/audio/podcast_2speakers_60s.wav \
        --golden tests/fixtures/golden/podcast_2speakers_60s.golden.json \
        --out spikes/qwen3_word_timestamp/outputs/poc_b_60s.json

参考文本来源:golden json 的 segment text 拼接(模拟 ASR 全文)。
输出:每字 [start_ms, end_ms] + RTF + 耗时,落 JSON 便于后续 PoC-C 当 gold。
"""
import argparse
import json
import time
import wave
from pathlib import Path


def load_reference_text(golden_path: Path) -> str:
    """从 golden json 拼接全文(模拟 ASR 输出的整段文本)"""
    data = json.loads(golden_path.read_text(encoding="utf-8"))
    return "".join(seg["text"] for seg in data["segments"])


def audio_duration_sec(audio_path: Path) -> float:
    """读时长(秒):先试 wav header,失败(mp3/m4a)退 ffprobe"""
    try:
        with wave.open(str(audio_path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(audio_path)],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--golden", default=None, help="golden json,拼接 segment text 当参考全文")
    ap.add_argument("--text", default=None, help="直接给参考文本字符串")
    ap.add_argument("--text-file", default=None, help="参考文本文件(纯文本,优先级最高)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    audio_path = Path(args.audio)
    out_path = Path(args.out)

    if args.text_file:
        ref_text = Path(args.text_file).read_text(encoding="utf-8")
    elif args.text:
        ref_text = args.text
    elif args.golden:
        ref_text = load_reference_text(Path(args.golden))
    else:
        raise SystemExit("需提供 --text-file / --text / --golden 之一")
    dur = audio_duration_sec(audio_path)
    print(f"[PoC-B] audio={audio_path.name} dur={dur:.2f}s text_len={len(ref_text)}字")

    # fa-zh 加载(首次自动下 ~38MB)
    t_load0 = time.time()
    from funasr import AutoModel

    model = AutoModel(model="fa-zh", disable_update=True)
    t_load = time.time() - t_load0
    print(f"[PoC-B] fa-zh 加载耗时 {t_load:.2f}s")

    # 对齐:fa-zh 吃 (sound, text)
    t0 = time.time()
    res = model.generate(
        input=(str(audio_path), ref_text),
        data_type=("sound", "text"),
    )
    elapsed = time.time() - t0
    rtf = (elapsed / dur) if dur > 0 else None

    item = res[0] if isinstance(res, list) and res else res
    timestamp = item.get("timestamp") if isinstance(item, dict) else None
    n_tokens = len(timestamp) if timestamp else 0

    print(f"[PoC-B] 对齐耗时 {elapsed:.2f}s  RTF={rtf}  tokens={n_tokens}")
    if timestamp:
        # 抽样打印前 12 个字的时间戳目测准度
        chars = [c for c in ref_text if not c.isspace()]
        print("[PoC-B] 前 12 字时间戳样本 (字: start_ms-end_ms):")
        for i in range(min(12, n_tokens)):
            ch = chars[i] if i < len(chars) else "?"
            s, e = timestamp[i]
            print(f"    {ch}: {s}-{e}")

    out_path.write_text(
        json.dumps(
            {
                "audio": str(audio_path),
                "duration_sec": dur,
                "text_len": len(ref_text),
                "load_sec": t_load,
                "align_sec": elapsed,
                "rtf": rtf,
                "n_tokens": n_tokens,
                "timestamp": timestamp,
                "reference_text": ref_text,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[PoC-B] 结果落 {out_path}")


if __name__ == "__main__":
    main()
