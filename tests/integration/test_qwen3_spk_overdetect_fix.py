"""Qwen3 speaker over-detect 修复 integration test.

调研结论 (docs/开发/archive/spk-over-detect-归因调研结果.md):
- 60min-2spk mp3 audio 在 cd578a8 (worker 加 ffmpeg convert_to_wav) 之后被识别为 4 个 speaker
- 根因: ffmpeg→wav 转码改变了 audio 字节, 通过 pyannote+TitaNet embedding 放大,
  产生 2 个 43.2s / 61.6s 的中长噪声 cluster, 突破 filter_spurious 36s 阈值漏网

本 test 走完整 worker pipeline (mp3 输入, 真 Qwen3 模型, 真 sherpa diarize),
断言 60min 双人访谈最终 speakers 数 == 2.

修复前: speakers_count == 4 (fail)
修复后: speakers_count == 2 (pass)

默认 skip, 需 FUNASR_RUN_INTEGRATION=1 + 60min-2spk audio 落地才跑.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest


RUN_INTEGRATION = os.getenv("FUNASR_RUN_INTEGRATION") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="设置 FUNASR_RUN_INTEGRATION=1 启用 (默认 skip, 真 Qwen3 模型 + 60min audio, ~10min)",
)


AUDIO_2SPK_60MIN = Path("tmp_long_audio/eval_set/audio_2spk_60min.mp3")


def _qwen3_models_ready() -> bool:
    from src.core.config import config
    paths = [
        Path(config.qwen3.asr_model_dir) / "qwen3_asr_encoder_frontend.onnx",
        Path(config.qwen3.asr_model_dir) / "qwen3_asr_encoder_backend.onnx",
        Path(config.qwen3.asr_model_dir) / "qwen3_asr_llm.gguf",
        Path(config.qwen3.segmentation_model),
        Path(config.qwen3.embedding_model),
    ]
    return all(p.exists() for p in paths)


@pytest.fixture
async def qwen3_pool_n1():
    """启 N=1 Qwen3 池 (单跑 60min audio, 避免并发引入额外变量)"""
    from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

    pool_transcriber = Qwen3PoolTranscriber(pool_size=1)
    await pool_transcriber.initialize()
    await asyncio.sleep(1.0)
    yield pool_transcriber
    await pool_transcriber._pool.cleanup()


class TestSpkOverDetectFix:
    """60min-2spk 不应 over-detect"""

    @pytest.mark.asyncio
    async def test_2spk_60min_mp3_no_over_detect(self, qwen3_pool_n1):
        """60min mp3 (2 真实 speaker) 走 worker pipeline, 期望最终 speakers 数 == 2.

        repro audio: 吴明辉 × 程曼祺 双人访谈, 截自 audio_149min.mp3 前 60min.
        修复前 (commit cd578a8 之后): speakers == 4 (over-detect, 2 中长噪声 cluster 漏网)
        修复后 (本 PR): speakers == 2
        """
        if not AUDIO_2SPK_60MIN.exists():
            pytest.skip(f"audio not found: {AUDIO_2SPK_60MIN}")
        if not _qwen3_models_ready():
            pytest.skip("Qwen3 模型权重未落地")

        result = await qwen3_pool_n1.transcribe(
            audio_path=str(AUDIO_2SPK_60MIN),
            task_id="overdetect-fix-60min-2spk",
            output_format="json",
        )

        assert isinstance(result, tuple) and len(result) == 2, (
            f"expected (TranscriptionResult, raw) tuple, got {type(result)}"
        )
        tres, raw = result

        assert len(tres.speakers) == 2, (
            f"60min-2spk audio 应识别为 2 个 speaker, 实测 {len(tres.speakers)}: "
            f"{tres.speakers}. "
            f"如果是 4+ 可能是 over-detect 回归 (调研报告 spk-over-detect-归因调研结果.md)"
        )
