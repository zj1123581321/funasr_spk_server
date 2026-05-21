"""sherpa-onnx 离线说话人分离 wrapper

数据流向:
  audio_path (wav/mp3...) --soundfile--> mono 16kHz float32 ndarray
    --sherpa_onnx.OfflineSpeakerDiarization.process()--> result
    --sort_by_start_time()--> [{start, end, speaker}, ...]

依赖: sherpa-onnx, soundfile, numpy, librosa (仅在采样率 != 16kHz 时触发重采样)

本文件移植自 spikes/qwen3_diarize/src/diarize.py, 改造点:
- 模型路径不再硬编码 spike 目录, 必须由调用方传入(默认值给 None, 触发明确报错)
- 加 loguru 日志
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import sherpa_onnx
import soundfile as sf
from loguru import logger

TARGET_SAMPLE_RATE = 16000


def _load_audio_mono_16k(audio_path: str) -> tuple[np.ndarray, int]:
    """读音频并归一为 16kHz 单声道 float32.

    PR3: 优先 soundfile (wav/flac/ogg); soundfile 不支持的格式 (m4a/aac 等)
    fallback 到 librosa.load (会自动调 ffmpeg/audioread 解码).
    """
    try:
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        audio = audio[:, 0]  # 仅取第一通道
    except Exception:
        # m4a/aac/mp3 等 soundfile 不支持的格式
        import librosa
        audio, _sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
        return audio.astype(np.float32), TARGET_SAMPLE_RATE

    if sample_rate != TARGET_SAMPLE_RATE:
        # 延迟 import, 只在需要重采样时加载 librosa
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
    """构造 OfflineSpeakerDiarization 实例

    当 num_speakers 为 None,用 cluster_threshold 自适应聚类;
    否则锁定簇数 (num_clusters=num_speakers),threshold 仍传但被忽略.
    """
    # num_speakers=None 时传 -1, 这是 sherpa-onnx 的"不指定"约定
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
    segmentation_model: str,
    embedding_model: str,
    num_speakers: Optional[int] = None,
    cluster_threshold: float = 0.9,
    num_threads: int = 8,
    provider: str = "cpu",
    min_duration_on: float = 0.3,
    min_duration_off: float = 0.5,
) -> list[dict]:
    """对音频跑 sherpa-onnx-offline-speaker-diarization.

    Args:
        audio_path: 输入音频 (任意 soundfile 支持的格式), 非 16kHz 单声道会自动转换.
        segmentation_model: pyannote 分段模型 onnx 路径(必填, 不再有默认).
        embedding_model: 3D-Speaker / NeMo embedding 模型 onnx 路径(必填).
        num_speakers: 已知说话人数则传 int 锁定簇数; 未知则传 None,用阈值聚类.
        cluster_threshold: num_speakers=None 时启用,值越大聚出来的说话人越少.
        num_threads: ONNX runtime 推理线程数.
        provider: "cpu" / "coreml" / "cuda" 等 sherpa-onnx 支持的 provider.
        min_duration_on / min_duration_off: 分段最小启停时长 (秒).

    Returns:
        按起始时间排序的 turns list, 每项 {"start": float, "end": float, "speaker": int}.
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
    logger.debug(
        f"sherpa diarize start: audio_len={len(audio)/TARGET_SAMPLE_RATE:.2f}s "
        f"num_speakers={num_speakers} threshold={cluster_threshold} "
        f"provider={provider} threads={num_threads}"
    )
    result = pipeline.process(audio).sort_by_start_time()

    return [
        {"start": float(r.start), "end": float(r.end), "speaker": int(r.speaker)}
        for r in result
    ]


def run_diarization_dispatched(
    audio_path: str,
    segmentation_model: str,
    embedding_model: str,
    num_speakers: Optional[int] = None,
    cluster_threshold: float = 0.9,
    num_threads: int = 8,
    provider: str = "cpu",
    min_duration_on: float = 0.3,
    min_duration_off: float = 0.5,
    backend: Optional[str] = None,
) -> list[dict]:
    """根据 runtime / env / 显式 backend 参数选 sherpa 或 ort_cuda 跑 diarize.

    优先级: backend 参数 > FUNASR_QWEN3_DIARIZE_BACKEND env > runtime 推荐.

    跟 run_diarization (sherpa) 行为 parity, 输出 schema 一致:
    [{"start": float, "end": float, "speaker": int}, ...]
    """
    if backend is None:
        from src.core.runtime import detect_runtime

        backend = detect_runtime().recommend_diarize_backend()

    if backend == "ort_cuda":
        from src.core.qwen3.diarize_ort import run_diarization_ort_cuda

        return run_diarization_ort_cuda(
            audio_path,
            segmentation_model=segmentation_model,
            embedding_model=embedding_model,
            num_speakers=num_speakers,
            cluster_threshold=cluster_threshold,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )

    # default: sherpa-onnx
    return run_diarization(
        audio_path,
        segmentation_model=segmentation_model,
        embedding_model=embedding_model,
        num_speakers=num_speakers,
        cluster_threshold=cluster_threshold,
        num_threads=num_threads,
        provider=provider,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
    )
