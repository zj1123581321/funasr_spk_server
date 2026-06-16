"""
Qwen3-Diarize 引擎模块

三个内部模块, 都是纯函数式 wrapper:
- asr: 调 src.core.vendor.qwen_asr_gguf 跑 GGUF/ONNX ASR
- diarize: 调 sherpa-onnx 跑 speaker diarization
- merge: 把 ASR 全文按 diarize turns 时长比例线性切到各 turn

对外暴露的入口是 Qwen3DiarizeTranscriber (src/core/qwen3_transcriber.py),
本包内的三模块只接受显式参数, 不依赖 config singleton, 方便单测。
"""
