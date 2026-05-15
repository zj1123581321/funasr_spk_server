"""sherpa-onnx 离线说话人分离 wrapper。

数据流向:
  audio_path (wav/mp3...) --soundfile--> mono 16kHz float32 ndarray
    --sherpa_onnx.OfflineSpeakerDiarization.process()--> result
    --sort_by_start_time()--> [{start, end, speaker}, ...]

依赖: sherpa-onnx, soundfile, numpy, librosa (仅在采样率 != 16kHz 时触发重采样)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import sherpa_onnx
import soundfile as sf

# 模型相对位置 (相对 spikes/qwen3_diarize 根目录)
_SPIKE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEGMENTATION_MODEL = str(
    _SPIKE_ROOT / "models" / "sherpa" / "pyannote-segmentation-3.0" / "model.onnx"
)
DEFAULT_EMBEDDING_MODEL = str(
    _SPIKE_ROOT / "models" / "sherpa" / "3dspeaker-eres2net" / "embedding.onnx"
)

TARGET_SAMPLE_RATE = 16000


def _load_audio_mono_16k(audio_path: str) -> tuple[np.ndarray, int]:
    """读音频并归一为 16kHz 单声道 float32。"""
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # 仅取第一通道
    if sample_rate != TARGET_SAMPLE_RATE:
        # 延迟 import,只在需要重采样时加载 librosa
        import librosa

        audio = librosa.resample(
            audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
        )
        sample_rate = TARGET_SAMPLE_RATE
    return audio, sample_rate


def _build_pipeline(
    segmentation_model: str,
    embedding_model: str,
    num_speakers: Optional[int],
    cluster_threshold: float,
    num_threads: int,
    provider: str,
    min_duration_on: float,
    min_duration_off: float,
) -> sherpa_onnx.OfflineSpeakerDiarization:
    """构造 OfflineSpeakerDiarization 实例。

    当 num_speakers 为 None,用 cluster_threshold 自适应聚类;
    否则锁定簇数 (num_clusters=num_speakers),threshold 仍传但被忽略。
    """
    # num_speakers=None 时传 -1,这是 sherpa-onnx 的"不指定"约定
    num_clusters = -1 if num_speakers is None else int(num_speakers)

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation_model
            ),
            num_threads=num_threads,
            provider=provider,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=embedding_model,
            num_threads=num_threads,
            provider=provider,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_clusters,
            threshold=cluster_threshold,
        ),
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
    )
    if not config.validate():
        raise RuntimeError(
            "OfflineSpeakerDiarizationConfig 校验失败,检查模型路径是否存在: "
            f"segmentation={segmentation_model} embedding={embedding_model}"
        )
    return sherpa_onnx.OfflineSpeakerDiarization(config)


def run_diarization(
    audio_path: str,
    num_speakers: Optional[int] = None,    # PoC 实测 num_clusters 被吞,改成 None 让 threshold 主导
    cluster_threshold: float = 0.9,         # 0.9 对中文双人 podcast 实测能聚出 2 个 speaker
    segmentation_model: str = DEFAULT_SEGMENTATION_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    num_threads: int = 1,
    provider: str = "cpu",
    min_duration_on: float = 0.3,
    min_duration_off: float = 0.5,
) -> list[dict]:
    """对音频跑 sherpa-onnx-offline-speaker-diarization。

    Args:
        audio_path: 输入音频 (任意 soundfile 支持的格式),非 16kHz 单声道会自动转换。
        num_speakers: 已知说话人数则传 int 锁定簇数;未知则传 None,用阈值聚类。
        cluster_threshold: num_speakers=None 时启用,值越大聚出来的说话人越少。
        segmentation_model: pyannote 分段模型 onnx 路径。
        embedding_model: 3D-Speaker / NeMo embedding 模型 onnx 路径。
        num_threads: ONNX runtime 推理线程数。
        provider: "cpu" / "coreml" / "cuda" 等 sherpa-onnx 支持的 provider。
        min_duration_on / min_duration_off: 分段最小启停时长 (秒)。

    Returns:
        按起始时间排序的 turns list,每项 {"start": float, "end": float, "speaker": int}。
    """
    pipeline = _build_pipeline(
        segmentation_model=segmentation_model,
        embedding_model=embedding_model,
        num_speakers=num_speakers,
        cluster_threshold=cluster_threshold,
        num_threads=num_threads,
        provider=provider,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
    )

    audio, _sr = _load_audio_mono_16k(audio_path)
    result = pipeline.process(audio).sort_by_start_time()

    return [
        {"start": float(r.start), "end": float(r.end), "speaker": int(r.speaker)}
        for r in result
    ]
