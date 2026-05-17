"""
PoC v3: 完整复现生产 stack (frontend ONNX CoreML EP + backend mlpackage + sherpa-onnx)
找出 SIGSEGV 的真正触发条件。

Stack 加载顺序:
  1. import + 加载 llama.cpp GGUF + Metal init     [新, 之前 PoC 没有]
  2. import onnxruntime + 加载 frontend ONNX (CoreML EP)  [新]
  3. 加载 backend mlpackage (coremltools)
  4. 跑 warmup encode (frontend ANE + backend mlpackage)
  5. import sherpa_onnx + 加载 embedding extractor  ⚠️ SIGSEGV 触发点
  6. 跑 N 次 encode + embedding

跑法:
    venv/bin/python spikes/qwen3_mac_hw_accel/phase3_backend/poc_full_stack_repro.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models/qwen3_diarize/Qwen3-ASR-1.7B"))
    ap.add_argument("--embedding", default=str(ROOT / "models/qwen3_diarize/sherpa/nemo-titanet-small/embedding.onnx"))
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--load-llm", action="store_true", help="额外加载 GGUF llm (生产真实路径)")
    ap.add_argument("--with-autoreleasepool", action="store_true",
                    help="包 autoreleasepool 在 backend predict 周围")
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"[config] load_llm={args.load_llm}, autoreleasepool={args.with_autoreleasepool}", flush=True)

    model_dir = Path(args.model_dir)
    frontend = str(model_dir / "qwen3_asr_encoder_frontend.onnx")
    backend_mlp = str(model_dir / "qwen3_asr_encoder_backend.mlpackage")

    # ----- Step 1 (optional): llama.cpp GGUF + Metal init -----
    if args.load_llm:
        print("[step1] load llama.cpp GGUF (Metal)", flush=True)
        # build_engine 会加载 ASR 引擎 + LLM, 这里只 import 一下 llm 让 ggml/metal init
        from src.core.vendor.qwen_asr_gguf.inference import llama as _llama_mod
        # 实际加载交给 QwenASREngine, 这里仅 import 让 dylib 装载
        print("[step1] llama module imported", flush=True)

    # ----- Step 2-4: QwenAudioEncoder 完整加载 (frontend CoreML EP + backend mlpackage + warmup) -----
    print("[step2-4] QwenAudioEncoder load + warmup", flush=True)
    t0 = time.time()
    from src.core.vendor.qwen_asr_gguf.inference.encoder import QwenAudioEncoder
    encoder = QwenAudioEncoder(
        frontend_path=frontend,
        backend_path=backend_mlp,
        onnx_provider="COREML_ANE_FULL",
        dml_pad_to=40,
        verbose=True,
    )
    print(f"[step2-4] encoder loaded + warmup in {time.time()-t0:.1f}s", flush=True)

    # ----- Step 5: 加载 sherpa-onnx (⚠️ SIGSEGV 触发点) -----
    print("[step5] import + load sherpa-onnx (⚠️ SIGSEGV 触发点)", flush=True)
    t0 = time.time()
    import sherpa_onnx
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=args.embedding, num_threads=4, provider="cpu", debug=False,
    )
    assert cfg.validate(), "sherpa cfg invalid"
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    print(f"[step5] sherpa loaded in {time.time()-t0:.1f}s ✅ 通过 SIGSEGV 点", flush=True)

    # ----- Step 6: 循环 encode + embedding -----
    dummy_audio = np.zeros(16000 * 30, dtype=np.float32)  # 30s
    print(f"[step6] {args.runs} encode + embedding iters", flush=True)
    t0 = time.time()
    for i in range(args.runs):
        # encoder full encode (mel + frontend + backend)
        if args.with_autoreleasepool:
            import objc
            with objc.autorelease_pool():
                emb, _ = encoder.encode(dummy_audio)
        else:
            emb, _ = encoder.encode(dummy_audio)

        # sherpa embedding
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=dummy_audio[:16000*3])
        stream.input_finished()
        _ = extractor.compute(stream)

        if (i + 1) % 5 == 0:
            print(f"  iter {i+1}/{args.runs} OK", flush=True)

    elapsed = time.time() - t0
    print(f"\n✅ {args.runs} encode+embedding iters in {elapsed:.1f}s, avg {elapsed/args.runs*1000:.1f} ms/iter")
    print(f"完整 stack repro PASS (load_llm={args.load_llm}, autoreleasepool={args.with_autoreleasepool})")


if __name__ == "__main__":
    main()
