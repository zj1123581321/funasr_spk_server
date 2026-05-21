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

# 延迟 import: librosa/scipy 只在 mel 计算时用, 不污染 module import 路径
def _scipy_hann_window(size: int) -> np.ndarray:
    from scipy.signal.windows import hann as _hann

    return _hann(size, sym=False).astype(np.float32)


def _librosa_mel_filterbank(
    sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    import librosa

    return librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
    ).astype(np.float32)

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


def compute_titanet_log_mel(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_window_size: int = 400,
    n_window_stride: int = 160,
    n_fft: int = 512,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    preemph: float = 0.97,
    log_zero_guard: float = 2**-24,
    norm_eps: float = 1e-5,
) -> np.ndarray:
    """NeMo TitaNet 风格 80-band log-mel preprocessing (numpy + librosa/scipy).

    步骤跟 NeMo AudioToMelSpectrogramPreprocessor 对齐:
    1. preemphasis: y[t] = x[t] - preemph * x[t-1]  (preemph=0 跳过)
    2. STFT: n_fft=512, hann window n_window_size=400 zero-pad 到 n_fft, hop=160,
       center=True (reflect pad)
    3. power spectrum (|spec|^2)
    4. mel filterbank: librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax=sr/2)
    5. log(mel + log_zero_guard)
    6. per_feature normalize: 沿 time axis 做 z-score, 每个 mel band 独立

    输入: audio (T,) float32 @ sample_rate
    输出: (n_mels, T_mel) float32. ONNX 期望 (B, n_mels, T_mel), 调用方加 batch dim.
    """
    if fmax is None:
        fmax = sample_rate / 2

    audio = np.asarray(audio, dtype=np.float32)

    # 1. preemphasis
    if preemph and preemph != 0.0:
        audio = np.concatenate(
            [[audio[0]], audio[1:] - preemph * audio[:-1]]
        ).astype(np.float32)

    # 2. center=True reflect pad
    pad = n_fft // 2
    if audio.shape[0] < n_fft:
        audio = np.concatenate(
            [audio, np.zeros(n_fft - audio.shape[0], dtype=np.float32)]
        )
    audio_padded = np.pad(audio, pad, mode="reflect")

    # 切 STFT frames
    n_frames = 1 + (audio_padded.shape[0] - n_fft) // n_window_stride
    if n_frames < 1:
        n_frames = 1
    frames = np.lib.stride_tricks.as_strided(
        audio_padded,
        shape=(n_frames, n_fft),
        strides=(audio_padded.strides[0] * n_window_stride, audio_padded.strides[0]),
    ).copy()

    # hann window n_window_size, zero-pad 到 n_fft
    win = _scipy_hann_window(n_window_size)
    if n_window_size < n_fft:
        win = np.pad(win, (0, n_fft - n_window_size)).astype(np.float32)
    windowed = frames * win  # (n_frames, n_fft)

    # 3. STFT power
    spec = np.fft.rfft(windowed, n=n_fft, axis=-1)
    mag_power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)

    # 4. mel filterbank
    mel_basis = _librosa_mel_filterbank(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
    )  # (n_mels, n_fft//2 + 1)
    mel_spec = mag_power @ mel_basis.T  # (n_frames, n_mels)

    # 5. log
    log_mel = np.log(mel_spec + log_zero_guard).astype(np.float32)

    # 6. per_feature normalize (沿 time axis z-score, 每个 band 独立)
    mean = log_mel.mean(axis=0, keepdims=True)
    std = log_mel.std(axis=0, keepdims=True)
    log_mel = (log_mel - mean) / (std + norm_eps)

    # 转 (n_mels, n_frames) 跟 NeMo / sherpa 期望对齐
    return log_mel.T.astype(np.float32)


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
