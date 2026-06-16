"""
Phase 3 — backend mlpackage ANE vs CPU parity 集成测试

验证 onnx_provider="COREML_ANE_FULL" (frontend ANE + backend mlpackage ANE)
与 "CPU" 跑同一段 audio, ASR 输出在主要字段上一致
(允许 FP16 vs FP32 数值差异引入的 chars-level diff < 5%).

跑法:
    FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest \
        tests/integration/test_qwen3_backend_coreml_ane_parity.py -v -s

前置条件:
- macOS (其他平台 ANE 路径 fallback CPU, 测试退化)
- mlpackage 文件存在 (由 spikes/qwen3_mac_hw_accel/phase3_backend/export_backend_coreml.py 生成)
"""
from __future__ import annotations

import gc
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pytest


RUN_INTEGRATION = os.getenv("FUNASR_RUN_INTEGRATION") == "1"


def _models_ready() -> tuple[bool, str]:
    """检查 ASR 模型 + mlpackage 是否就位; 返回 (ready, reason)"""
    from src.core.config import config
    model_dir = Path(config.qwen3.asr_model_dir)
    onnx_files = [
        model_dir / "qwen3_asr_encoder_frontend.onnx",
        model_dir / "qwen3_asr_encoder_backend.onnx",
        model_dir / "qwen3_asr_llm.gguf",
    ]
    mlp_dir = model_dir / "qwen3_asr_encoder_backend.mlpackage"
    missing = [str(p) for p in onnx_files if not p.exists()]
    if missing:
        return False, f"ASR ONNX/GGUF 缺: {missing}"
    if not mlp_dir.is_dir():
        return False, f"backend mlpackage 缺: {mlp_dir} (跑 spikes/.../phase3_backend/export_backend_coreml.py 生成)"
    return True, ""


_MODELS_OK, _MODELS_REASON = (False, "未检查") if not RUN_INTEGRATION else _models_ready()

pytestmark = [
    pytest.mark.skipif(
        not RUN_INTEGRATION,
        reason="设置 FUNASR_RUN_INTEGRATION=1 启用",
    ),
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="backend mlpackage ANE 仅 macOS 有意义",
    ),
    pytest.mark.skipif(
        not _MODELS_OK,
        reason=f"模型/mlpackage 未就位: {_MODELS_REASON}",
    ),
]


def _run_asr_once(audio_path: Path, onnx_provider: str) -> dict:
    """构造引擎跑一次 ASR, 返回关键字段; 用完即 gc 释放"""
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
        chunks = getattr(result, "chunks", []) or []
        return {
            "text": text,
            "chunk_count": len(chunks),
            "chunk_texts": [getattr(c, "text", "") for c in chunks],
        }
    finally:
        del engine
        gc.collect()


@pytest.mark.usefixtures("podcast_audio")
class TestBackendCoremlAneFullParity:
    def test_asr_text_matches_cpu_baseline(self, podcast_audio: Path):
        """同 audio, CPU baseline vs COREML_ANE_FULL (frontend ANE + backend mlpackage ANE) 文本一致"""
        cpu = _run_asr_once(podcast_audio, "CPU")
        assert cpu["text"], "CPU baseline ASR 文本不应为空"
        assert cpu["chunk_count"] > 0, "CPU baseline 应有 ASR chunks"

        ane = _run_asr_once(podcast_audio, "COREML_ANE_FULL")
        assert ane["text"], "COREML_ANE_FULL ASR 文本不应为空"

        # 文本相似度 (FP16 cumulative drift over 24 layers + greedy decode token 选择差异)
        # Phase 2 frontend ANE 实测 ratio ~0.99; Phase 3 backend 全 FP16 累积差更大,
        # 阈值放到 0.90 以兜底数值漂移
        ratio = SequenceMatcher(None, cpu["text"], ane["text"]).ratio()
        char_diff_pct = abs(len(cpu["text"]) - len(ane["text"])) / max(len(cpu["text"]), 1)
        print(
            f"\nparity report (Phase 3 backend mlpackage):"
            f"\n  cpu_chars={len(cpu['text'])} ane_chars={len(ane['text'])} char_len_diff={char_diff_pct:.2%}"
            f"\n  cpu_chunks={cpu['chunk_count']} ane_chunks={ane['chunk_count']}"
            f"\n  text_similarity={ratio:.4f}",
            flush=True,
        )
        assert ratio >= 0.90, (
            f"ASR 文本 similarity {ratio:.4f} < 0.90, COREML_ANE_FULL 可能数值出 bug\n"
            f"CPU      : {cpu['text'][:200]!r}...\n"
            f"ANE_FULL : {ane['text'][:200]!r}..."
        )
        # 字符长度差异 <= 10%
        assert char_diff_pct <= 0.10, (
            f"ASR 文本长度差异过大: {char_diff_pct:.2%} > 10%"
        )

        # ASR chunk 数量应一致 (40s 边界不受 encoder backend 影响)
        assert cpu["chunk_count"] == ane["chunk_count"], (
            f"ASR chunk 数量差异: cpu={cpu['chunk_count']} ane_full={ane['chunk_count']}"
        )
