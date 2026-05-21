"""ORT 直 wrap diarize backend — pyannote-seg + TitaNet + Python clustering.

替代 sherpa-onnx 的 OfflineSpeakerDiarization, 拆 4 个组件用 Python onnxruntime 直跑:

1. pyannote-segmentation-3.0 sliding window (本文件: pyannote_segmentation_pipeline)
2. TitaNet 80-band log-mel preprocessing (TODO commit 6)
3. TitaNet ORT CUDA inference + Python FastClustering (TODO commit 7)

为什么不走 sherpa-onnx GPU build:
sherpa-onnx CUDA build 的 C++ wrapper 跟 llama.cpp CUDA 撞 segfault (见 docs/开发/gpu加速/
2026-05-21-3060-CUDA移植与优化.md). ORT Python API 跟 llama.cpp CUDA 共存稳定 (PoC 通过).
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

# pyannote-segmentation-3.0 模型常量 (固定)
PYANNOTE_CHUNK_SAMPLES = 160000  # 10s @ 16k
PYANNOTE_CHUNK_FRAMES = 589  # 每 chunk 输出 frame 数
PYANNOTE_NUM_SPEAKERS = 3  # max speakers per chunk
PYANNOTE_NUM_POWERSET_CLASSES = 7  # C(3,0)+C(3,1)+C(3,2) = 1+3+3
PYANNOTE_FRAME_RATE_HZ = PYANNOTE_CHUNK_FRAMES / 10.0  # 58.9 Hz
PYANNOTE_STEP_SAMPLES = 16000  # 1s sliding window step (pyannote-audio default)

# powerset class id → multi-label binary (3-speaker)
# class 0 = silence; 1/2/3 = 单 speaker; 4/5/6 = 两 speaker 同时活动
_POWERSET_MAP = np.array(
    [
        [0, 0, 0],  # silence
        [1, 0, 0],  # spk0
        [0, 1, 0],  # spk1
        [0, 0, 1],  # spk2
        [1, 1, 0],  # spk0+1
        [1, 0, 1],  # spk0+2
        [0, 1, 1],  # spk1+2
    ],
    dtype=np.int8,
)


def _iter_audio_chunks(
    audio: np.ndarray,
    chunk_samples: int = PYANNOTE_CHUNK_SAMPLES,
    step_samples: int = PYANNOTE_STEP_SAMPLES,
) -> Iterator[Tuple[int, np.ndarray]]:
    """切 audio 为重叠 chunks, yield (start_sample, chunk).

    保证 chunk shape == (chunk_samples,) (末尾不足时 zero pad). audio 为空时
    不 yield. audio 短于 chunk_samples 时 yield 一次 padded chunk.
    """
    n = int(audio.shape[0])
    if n == 0:
        return
    start = 0
    while True:
        chunk = audio[start : start + chunk_samples]
        if chunk.shape[0] < chunk_samples:
            pad = np.zeros(chunk_samples - chunk.shape[0], dtype=audio.dtype)
            chunk = np.concatenate([chunk, pad])
        yield start, chunk
        if start + chunk_samples >= n:
            break
        start += step_samples


def _powerset_to_multilabel(
    powerset_logits: np.ndarray, num_speakers: int = PYANNOTE_NUM_SPEAKERS
) -> np.ndarray:
    """(..., K) powerset logits / probs → (..., S) multi-label binary.

    每 frame 取 argmax class id, 查表 _POWERSET_MAP 得到 multi-label.
    K=7, S=3 for pyannote-segmentation-3.0.
    """
    cls = np.argmax(powerset_logits, axis=-1)  # (...,)
    return _POWERSET_MAP[cls]  # (..., S)


def _aggregate_chunk_outputs(
    chunk_starts: list[int],
    chunk_outputs: list[np.ndarray],
    audio_samples: int,
    sample_rate: int = 16000,
    frame_rate: float = PYANNOTE_FRAME_RATE_HZ,
) -> np.ndarray:
    """Whisper-style 加权融合 chunk outputs → (T_total_frames, K) 平均.

    每个 chunk 在它的 start_sample 处贡献 (chunk_frames, K), 多 chunk 重叠帧除 count.
    没有任何 chunk 覆盖的帧 (理论不会出现) 用 0 (silence) 兜底.
    """
    if not chunk_outputs:
        raise ValueError("chunk_outputs is empty")
    # audio 真实 frame 数 (floor) — pyannote 输出每 chunk 589 frame 对齐到 10s, audio 末
    # 半 frame 算 pad. 用 floor 保证 audio_frames <= chunks 累计覆盖的 frame.
    audio_frames = int(audio_samples / sample_rate * frame_rate)
    # 最末 chunk 在 frame index 上的 end; 短 audio (< chunk_samples) 时 chunk 输出 frames
    # 也是 589, 可能超过 audio_frames — 此时 total_frames 取 chunk 维度防截断.
    last_start_frame = int(round(chunk_starts[-1] / sample_rate * frame_rate))
    last_end_frame = last_start_frame + chunk_outputs[-1].shape[0]
    total_frames = max(audio_frames, last_end_frame)

    K = chunk_outputs[0].shape[-1]
    accum = np.zeros((total_frames, K), dtype=np.float32)
    count = np.zeros(total_frames, dtype=np.float32)

    for start_sample, out in zip(chunk_starts, chunk_outputs):
        start_frame = int(round(start_sample / sample_rate * frame_rate))
        end_frame = min(start_frame + out.shape[0], total_frames)
        valid = end_frame - start_frame
        if valid <= 0:
            continue
        accum[start_frame:end_frame] += out[:valid]
        count[start_frame:end_frame] += 1.0

    count_safe = np.maximum(count, 1.0)
    out_frames = accum / count_safe[:, None]
    # 短 audio (< 1 frame, 极端 case) 返回 chunk 维度避免空数组; 正常 audio 截到 floor frame 数.
    slice_to = audio_frames if audio_frames > 0 else total_frames
    return out_frames[:slice_to]


def pyannote_segmentation_pipeline(
    audio: np.ndarray,
    ort_session,
    input_name: Optional[str] = None,
    sample_rate: int = 16000,
    chunk_samples: int = PYANNOTE_CHUNK_SAMPLES,
    step_samples: int = PYANNOTE_STEP_SAMPLES,
) -> np.ndarray:
    """完整 pyannote-segmentation-3.0 sliding window pipeline.

    Args:
        audio: 1D float32 16kHz mono
        ort_session: 已加载的 onnxruntime.InferenceSession 或 mock (鸭子类型, 需 run() 和 get_inputs())
        input_name: 若 None 则用 session.get_inputs()[0].name
        sample_rate / chunk_samples / step_samples: 通常用默认

    Returns:
        (T_total_frames, 3) int8 multi-label speaker activity. T_total_frames ≈
        audio_seconds * 58.9.
    """
    if input_name is None:
        input_name = ort_session.get_inputs()[0].name

    starts: list[int] = []
    outputs: list[np.ndarray] = []
    for start, chunk in _iter_audio_chunks(audio, chunk_samples, step_samples):
        x = chunk.reshape(1, 1, -1).astype(np.float32)
        raw = ort_session.run(None, {input_name: x})[0]
        # raw shape (1, frames, K); 去 batch dim
        outputs.append(raw[0])
        starts.append(start)

    agg = _aggregate_chunk_outputs(
        starts, outputs, audio_samples=int(audio.shape[0]), sample_rate=sample_rate
    )
    return _powerset_to_multilabel(agg)
