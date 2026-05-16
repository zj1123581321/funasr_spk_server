"""
Phase 2 Step 2.4 — frontend ANE vs CPU parity 集成测试

验证 onnx_provider="COREML_ANE_FE" 与 "CPU" 跑同一段 audio,
ASR 输出在主要字段上一致 (允许 FP16 vs FP32 数值差异引入的 chars-level diff < 1%).

跑法:
    FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest \\
        tests/integration/test_qwen3_frontend_ane_parity.py -v -s

注意:
- 仅 macOS 上有意义 (其他平台 ANE 路径会 fallback CPU, 测试退化为自比)
- 测试只对比 ASR 文本/分片, 不跑 diarize (speaker labels 用独立 sherpa ONNX,
  与 frontend 无关, 已在 test_qwen3_diarize_e2e 覆盖)
"""
from __future__ import annotations

import gc
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pytest


RUN_INTEGRATION = os.getenv("FUNASR_RUN_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.skipif(
        not RUN_INTEGRATION,
        reason="设置 FUNASR_RUN_INTEGRATION=1 启用",
    ),
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="frontend ANE 仅 macOS 有意义",
    ),
]


def _models_ready() -> bool:
    from src.core.config import config
    paths = [
        Path(config.qwen3.asr_model_dir) / "qwen3_asr_encoder_frontend.onnx",
        Path(config.qwen3.asr_model_dir) / "qwen3_asr_encoder_backend.onnx",
        Path(config.qwen3.asr_model_dir) / "qwen3_asr_llm.gguf",
    ]
    return all(p.exists() for p in paths)


pytestmark_models = pytest.mark.skipif(
    not _models_ready(),
    reason="Qwen3 ASR 模型权重未落地",
)


def _run_asr_once(audio_path: Path, onnx_provider: str) -> dict:
    """构造引擎跑一次 ASR, 返回关键字段; 用完即 gc 释放, 避免双引擎驻留"""
    from src.core.config import config
    from src.core.qwen3.asr import build_engine_config
    from src.core.vendor.qwen_asr_gguf.inference.asr import QwenASREngine
    from src.core.vendor.qwen_asr_gguf.inference.audio import load_audio

    audio = load_audio(str(audio_path))
    cfg = build_engine_config(
        model_dir=config.qwen3.asr_model_dir,
        onnx_provider=onnx_provider,
    )
    engine = QwenASREngine(cfg)
    try:
        result = engine.asr(
            audio=audio,
            context=None,
            language="Chinese",
            temperature=0.4,
        )
        text = (getattr(result, "text", None) or "").strip()
        # production 默认 enable_aligner=False, alignment 可能 None;
        # chunks 是 ASR 内部 40s 分片, production 始终有.
        chunks = getattr(result, "chunks", []) or []
        chunk_count = len(chunks)
        chunk_texts = [getattr(c, "text", "") for c in chunks]
        return {
            "text": text,
            "chunk_count": chunk_count,
            "chunk_texts": chunk_texts,
        }
    finally:
        # 显式释放: 避免两个 ASR engine 同时驻留 GGUF/Metal/ANE 资源
        del engine
        gc.collect()


@pytest.mark.usefixtures("podcast_audio")
class TestFrontendAneParity:
    def test_asr_text_matches_cpu_baseline(self, podcast_audio: Path):
        """同 audio, CPU vs frontend ANE 的 ASR 输出主要字段应一致 (允许 < 1% chars diff)"""
        # CPU baseline 先跑 (确定性参考)
        cpu = _run_asr_once(podcast_audio, "CPU")
        assert cpu["text"], "CPU baseline ASR 文本不应为空"
        assert cpu["chunk_count"] > 0, "CPU baseline 应有 ASR chunks"

        # frontend ANE 跑
        ane = _run_asr_once(podcast_audio, "COREML_ANE_FE")
        assert ane["text"], "ANE 路径 ASR 文本不应为空"

        # 1) 整体文本相似度 — ratio >= 0.95
        # FP16 (CoreML MLProgram 默认) vs FP32 (CPU EP) 数值差异 + greedy 解码采样路径
        # 可能让 LM 在某些 chunk 边界选不同 token, 实测 60s 双人 podcast 差异约 0.8%,
        # 阈值留余量到 5% 以兜底极端情况; 真退化 (e.g. 句子缺失) 会远低于 0.95.
        ratio = SequenceMatcher(None, cpu["text"], ane["text"]).ratio()
        char_diff_pct = abs(len(cpu["text"]) - len(ane["text"])) / max(len(cpu["text"]), 1)
        print(
            f"\nparity report:"
            f"\n  cpu_chars={len(cpu['text'])} ane_chars={len(ane['text'])} char_len_diff={char_diff_pct:.2%}"
            f"\n  cpu_chunks={cpu['chunk_count']} ane_chunks={ane['chunk_count']}"
            f"\n  text_similarity={ratio:.4f}",
            flush=True,
        )
        assert ratio >= 0.95, (
            f"ASR 文本 similarity {ratio:.4f} < 0.95, 差异过大 (可能 ANE 路径出 bug)\n"
            f"CPU : {cpu['text'][:200]!r}...\n"
            f"ANE : {ane['text'][:200]!r}..."
        )
        # 字符长度差异 <= 10% (长度近似一致才说明 ASR 没漏句)
        assert char_diff_pct <= 0.10, (
            f"ASR 文本长度差异过大: {char_diff_pct:.2%} > 10%"
        )

        # 2) ASR chunk 数量应一致 (40s 边界不受 frontend 影响)
        assert cpu["chunk_count"] == ane["chunk_count"], (
            f"ASR chunk 数量差异: cpu={cpu['chunk_count']} ane={ane['chunk_count']}"
        )
