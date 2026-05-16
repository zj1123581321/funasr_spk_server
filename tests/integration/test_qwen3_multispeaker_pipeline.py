"""
PR4 — 多人 pipeline 端到端集成测试.

验证 short-segment guard + cluster centroid merge 集成后, detected speakers
与预期一致 (±1 tolerance).

默认 skip (需 ~2GB Qwen3 模型 + 真转录). 启用:
    FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest \\
        tests/integration/test_qwen3_multispeaker_pipeline.py -v

Cases:
- 1 spk: tts_1speaker_5s.wav (检查 1 ± 1)
- 2 spk: podcast_2speakers_60s.wav (检查 2 ± 1)
- 4 spk: tmp_long_audio/eval_set/audio_4spk.m4a (skipif 文件不存在, 不入 git)
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


RUN_INTEGRATION = os.getenv("FUNASR_RUN_INTEGRATION") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="设置 FUNASR_RUN_INTEGRATION=1 启用 (默认 skip, 加载 ~2GB Qwen3 模型)",
)


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


pytestmark2 = pytest.mark.skipif(
    not _qwen3_models_ready(),
    reason="Qwen3 模型权重未落地, 跑 scripts/download_qwen3_models.sh 后再试",
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "audio"
EVAL_SET = Path(__file__).resolve().parents[2] / "tmp_long_audio" / "eval_set"


@pytest.fixture(scope="module")
def fresh_qwen3_transcriber():
    from src.core.qwen3_transcriber import (
        get_qwen3_transcriber,
        reset_qwen3_transcriber_singleton,
    )
    reset_qwen3_transcriber_singleton()
    yield get_qwen3_transcriber()
    reset_qwen3_transcriber_singleton()


class TestMultispeakerPipeline:
    """跑 1/2/4 speaker 短样本, 验证 detected speakers ± 1."""

    @pytest.mark.asyncio
    async def test_single_speaker_detected(self, fresh_qwen3_transcriber) -> None:
        """tts_1speaker_5s.wav, 1 speaker."""
        audio = FIXTURES / "tts_1speaker_5s.wav"
        if not audio.exists():
            pytest.skip(f"audio fixture 缺失: {audio}")
        result, _raw = await fresh_qwen3_transcriber.transcribe(
            audio_path=str(audio),
            task_id="pr4-1spk",
            progress_callback=None,
            output_format="json",
        )
        n = len(result.speakers)
        assert 1 <= n <= 2, f"1 spk 期望 1±1, 实际 {n}: {result.speakers}"

    @pytest.mark.asyncio
    async def test_two_speakers_detected(self, fresh_qwen3_transcriber) -> None:
        """podcast_2speakers_60s.wav, 2 speakers."""
        audio = FIXTURES / "podcast_2speakers_60s.wav"
        if not audio.exists():
            pytest.skip(f"audio fixture 缺失: {audio}")
        result, _raw = await fresh_qwen3_transcriber.transcribe(
            audio_path=str(audio),
            task_id="pr4-2spk",
            progress_callback=None,
            output_format="json",
        )
        n = len(result.speakers)
        assert 1 <= n <= 3, f"2 spk 期望 2±1, 实际 {n}: {result.speakers}"

    @pytest.mark.asyncio
    async def test_four_speakers_detected_from_eval_set(
        self, fresh_qwen3_transcriber
    ) -> None:
        """eval_set/audio_4spk.m4a, 4 speakers (skipif 文件不存在, 不入 git)."""
        audio = EVAL_SET / "audio_4spk.m4a"
        if not audio.exists():
            pytest.skip(f"eval_set 文件不在本地: {audio}")
        result, _raw = await fresh_qwen3_transcriber.transcribe(
            audio_path=str(audio),
            task_id="pr4-4spk",
            progress_callback=None,
            output_format="json",
        )
        n = len(result.speakers)
        # PoC v12 final: 4 真人 + 1 音乐 cluster, 期望 detected 5 ± 1 (4-6)
        assert 3 <= n <= 6, f"4 spk 期望 4-5±1, 实际 {n}: {result.speakers}"
