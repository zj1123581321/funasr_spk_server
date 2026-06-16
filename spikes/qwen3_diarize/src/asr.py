"""
ASR wrapper — 复用 production qwen_asr_gguf 引擎做离线 5min 文件转录

数据流:
  audio_file (mp3/wav/任意 ffmpeg 支持) ─→ qwen_asr_gguf.QwenASREngine.transcribe()
    ─→ TranscribeResult(text, alignment.items=[ForcedAlignItem(text, start_time, end_time)])

引擎位置: src/vendor/qwen_asr_gguf (symlink 自 ~/Production/qwen_asr_server/...)
模型位置: models/production_models/Qwen3-ASR/Qwen3-ASR-1.7B/ (symlink 自 production)
"""
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import psutil

HERE = Path(__file__).resolve().parent
# 让 src/vendor 进 sys.path → 能 import qwen_asr_gguf
sys.path.insert(0, str(HERE / "vendor"))

DEFAULT_MODEL_DIR = HERE.parent / "models" / "production_models" / "Qwen3-ASR" / "Qwen3-ASR-1.7B"


@dataclass
class WordItem:
    text: str
    start: float
    end: float


@dataclass
class ASRResult:
    text: str
    items: List[WordItem]   # 词级时间戳
    duration: float          # 音频时长 (sec)
    elapsed: float           # ASR 总耗时 (sec)
    rtf: float               # elapsed / duration
    peak_rss_mb: float       # 进程峰值 RSS (MB)
    rss_delta_mb: float      # ASR 调用引入的 RSS 增量 (MB)


def run_asr(
    audio_file: str,
    model_dir: Optional[Path] = None,
    language: str = "Chinese",
    temperature: float = 0.4,
) -> ASRResult:
    """端到端跑一次 ASR，返回带 RTF/内存数据的结果.

    实现选择:
      - 不调 engine.transcribe(audio_file, ...) — 它的 duration=0 默认值会让 load_audio 加载 0 秒
      - 自己 load_audio(audio_file) 拿 numpy → engine.asr(audio, ...)
      - 不启 aligner — production 没 aligner 模型,enable_aligner=False;只拿全文 + segment 级时间(40s chunk)
    """
    # 延迟 import，避免 sys.path 还没装好就触发 vendor 包加载
    from qwen_asr_gguf.inference.asr import QwenASREngine
    from qwen_asr_gguf.inference.schema import ASREngineConfig
    from qwen_asr_gguf.inference.audio import load_audio

    mdir = str(model_dir or DEFAULT_MODEL_DIR)
    cfg = ASREngineConfig(
        model_dir=mdir,
        # production 实际部署的文件名:去掉了 schema 默认的 .int4 后缀
        encoder_frontend_fn="qwen3_asr_encoder_frontend.onnx",
        encoder_backend_fn="qwen3_asr_encoder_backend.onnx",
        llm_fn="qwen3_asr_llm.gguf",
        onnx_provider="CPU",           # mac 上 ONNX EP 默认 CPU（encoder 不大，影响小）
        llm_use_gpu=True,              # llama.cpp Metal — GGUF decoder 走 GPU
        enable_aligner=False,          # production 没 aligner 模型;先跑通 ASR 主路
    )

    proc = psutil.Process()
    rss_before = proc.memory_info().rss

    engine = QwenASREngine(cfg)

    # 自己 load_audio,拿到 16kHz mono float32 numpy
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
    peak_rss = rss_after  # psutil 在 macOS 不支持 peak_rss，用 after 作为代理

    items: List[WordItem] = []
    if result.alignment and result.alignment.items:
        for it in result.alignment.items:
            items.append(WordItem(text=it.text, start=it.start_time, end=it.end_time))

    return ASRResult(
        text=result.text,
        items=items,
        duration=duration,
        elapsed=elapsed,
        rtf=elapsed / duration if duration > 0 else 0.0,
        peak_rss_mb=peak_rss / (1024 * 1024),
        rss_delta_mb=(rss_after - rss_before) / (1024 * 1024),
    )
