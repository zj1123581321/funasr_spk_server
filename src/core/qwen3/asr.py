"""ASR wrapper — 复用 vendor qwen_asr_gguf 引擎做离线文件转录

数据流:
  audio_file (mp3/wav/...) -> qwen_asr_gguf.QwenASREngine.asr(audio)
    -> TranscribeResult(text, alignment.items=[ForcedAlignItem(text, start, end)])

本文件移植自 spikes/qwen3_diarize/src/asr.py, 改造点:
- 删除 sys.path hack, 直接 import src.core.vendor.qwen_asr_gguf 子包
- 模型路径必须显式传入, 不再硬编码 spike 目录
- 加 loguru 日志, 用同一引擎实例单例化(后续 Qwen3DiarizeTranscriber 维护)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import psutil
from loguru import logger

# 延迟到函数内 import 引擎类, 避免 module import 时就触发 dylib 加载 +
# llama.cpp Metal device 初始化(这些日志很吵, 实际只在第一次构建 engine 时需要)


@dataclass
class WordItem:
    """词级时间戳(production 不带 aligner, 实际仅 segment 级)"""
    text: str
    start: float
    end: float


@dataclass
class ASRResult:
    """ASR 输出"""
    text: str
    items: List[WordItem]   # 词级时间戳
    duration: float          # 音频时长 (sec)
    elapsed: float           # ASR 总耗时 (sec)
    rtf: float               # elapsed / duration
    peak_rss_mb: float       # 进程峰值 RSS (MB)
    rss_delta_mb: float      # ASR 调用引入的 RSS 增量 (MB)


def build_engine_config(
    model_dir: str,
    encoder_frontend_fn: str = "qwen3_asr_encoder_frontend.onnx",
    encoder_backend_fn: str = "qwen3_asr_encoder_backend.onnx",
    llm_fn: str = "qwen3_asr_llm.gguf",
    onnx_provider: str = "CPU",
    llm_use_gpu: bool = True,
    enable_aligner: bool = False,
):
    """构造 ASREngineConfig

    Args:
        model_dir: Qwen3-ASR 模型目录(含 encoder/decoder 文件).
        encoder_frontend_fn/backend_fn: 实际文件名(production 转换版去掉了默认的 .int4 后缀).
        llm_fn: GGUF 文件名.
        onnx_provider: ONNX EP, 默认 CPU(Mac 上 ANE/CoreML 实测无加速).
        llm_use_gpu: llama.cpp 是否走 Metal(Apple Silicon 上必开).
        enable_aligner: production 不带 aligner, 默认 False.

    Returns:
        ASREngineConfig 实例.
    """
    from src.core.vendor.qwen_asr_gguf.inference.schema import ASREngineConfig

    return ASREngineConfig(
        model_dir=model_dir,
        encoder_frontend_fn=encoder_frontend_fn,
        encoder_backend_fn=encoder_backend_fn,
        llm_fn=llm_fn,
        onnx_provider=onnx_provider,
        llm_use_gpu=llm_use_gpu,
        enable_aligner=enable_aligner,
    )


def build_engine(model_dir: str):
    """构造 QwenASREngine 实例(惰性 import, 首次实例化会加载 GGUF + Metal context)"""
    from src.core.vendor.qwen_asr_gguf.inference.asr import QwenASREngine

    cfg = build_engine_config(model_dir=model_dir)
    logger.info(f"加载 Qwen3-ASR 引擎: model_dir={model_dir}")
    engine = QwenASREngine(cfg)
    logger.success("Qwen3-ASR 引擎加载完成")
    return engine


def run_asr(
    audio_file: str,
    engine,
    language: str = "Chinese",
    temperature: float = 0.4,
) -> ASRResult:
    """端到端跑一次 ASR, 返回带 RTF/内存数据的结果.

    Args:
        audio_file: 输入音频文件(任何 soundfile/ffmpeg 支持的格式).
        engine: 已构造的 QwenASREngine 实例(单例由调用方维护).
        language: 识别语言, 默认 "Chinese".
        temperature: decoder 采样温度.

    Returns:
        ASRResult 含 text / segment-level items / RTF / 内存数据.

    实现选择:
      - 不调 engine.transcribe(audio_file, ...) — 它的 duration=0 默认值会让 load_audio 加载 0 秒
      - 自己 load_audio(audio_file) 拿 numpy → engine.asr(audio, ...)
      - 不启 aligner — production 没 aligner 模型, 只拿全文 + segment 级时间(40s chunk)
    """
    from src.core.vendor.qwen_asr_gguf.inference.audio import load_audio

    proc = psutil.Process()
    rss_before = proc.memory_info().rss

    audio = load_audio(audio_file)
    duration = len(audio) / 16000.0

    t0 = time.time()
    result = engine.asr(
        audio=audio,
        context=None,
        language=language,
        temperature=temperature,
    )
    elapsed = time.time() - t0

    rss_after = proc.memory_info().rss
    peak_rss = rss_after  # psutil 在 macOS 不支持 peak_rss, 用 after 作为代理

    items: List[WordItem] = []
    if result.alignment and result.alignment.items:
        for it in result.alignment.items:
            items.append(WordItem(text=it.text, start=it.start_time, end=it.end_time))

    rtf = elapsed / duration if duration > 0 else 0.0
    logger.info(
        f"ASR 完成: duration={duration:.2f}s elapsed={elapsed:.2f}s "
        f"RTF={rtf:.3f} chars={len(result.text)}"
    )
    return ASRResult(
        text=result.text,
        items=items,
        duration=duration,
        elapsed=elapsed,
        rtf=rtf,
        peak_rss_mb=peak_rss / (1024 * 1024),
        rss_delta_mb=(rss_after - rss_before) / (1024 * 1024),
    )
