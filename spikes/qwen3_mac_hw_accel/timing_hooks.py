"""Monkey-patch 各阶段加 timing collector. 不动 src/, 由 profile_worker import 后激活."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Dict, List


class StageTimer:
    """Per-process per-stage 时间累计 + N=2 时配合 worker_label 区分."""

    def __init__(self):
        self._lock = threading.Lock()
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._events: List[dict] = []  # (stage, t_start, t_end) 全量, 后续可画时间线
        self._t0 = time.monotonic()

    def reset(self):
        with self._lock:
            self._totals.clear()
            self._counts.clear()
            self._events.clear()
            self._t0 = time.monotonic()

    def record(self, stage: str, elapsed: float, t_start: float, t_end: float):
        with self._lock:
            self._totals[stage] = self._totals.get(stage, 0.0) + elapsed
            self._counts[stage] = self._counts.get(stage, 0) + 1
            self._events.append({
                "stage": stage,
                "t_start_rel": t_start - self._t0,
                "t_end_rel": t_end - self._t0,
                "elapsed": elapsed,
            })

    @contextmanager
    def span(self, stage: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            t1 = time.monotonic()
            self.record(stage, t1 - t0, t0, t1)

    def report(self) -> dict:
        with self._lock:
            return {
                "totals": dict(self._totals),
                "counts": dict(self._counts),
                "events": list(self._events),
                "wall": time.monotonic() - self._t0,
            }

    def dump_json(self, path: str):
        rep = self.report()
        with open(path, "w") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)


_T = StageTimer()


def get_timer() -> StageTimer:
    return _T


def install_patches():
    """Monkey-patch 各阶段, 重复调用幂等."""
    if getattr(install_patches, "_installed", False):
        return
    install_patches._installed = True

    # 1. vendor encoder: encode / _run_frontend / _run_backend / mel_extractor
    from src.core.vendor.qwen_asr_gguf.inference import encoder as _enc_mod

    QwenAudioEncoder = _enc_mod.QwenAudioEncoder

    _orig_encode = QwenAudioEncoder.encode
    _orig_fe = QwenAudioEncoder._run_frontend
    _orig_be = QwenAudioEncoder._run_backend

    def _patched_encode(self, audio):
        t0 = time.monotonic()
        try:
            return _orig_encode(self, audio)
        finally:
            t1 = time.monotonic()
            _T.record("asr.encoder.total", t1 - t0, t0, t1)

    def _patched_fe(self, mel):
        t0 = time.monotonic()
        try:
            return _orig_fe(self, mel)
        finally:
            t1 = time.monotonic()
            _T.record("asr.encoder.frontend_onnx", t1 - t0, t0, t1)

    def _patched_be(self, hidden):
        t0 = time.monotonic()
        try:
            return _orig_be(self, hidden)
        finally:
            t1 = time.monotonic()
            _T.record("asr.encoder.backend_onnx", t1 - t0, t0, t1)

    QwenAudioEncoder.encode = _patched_encode
    QwenAudioEncoder._run_frontend = _patched_fe
    QwenAudioEncoder._run_backend = _patched_be

    # 2. vendor asr: QwenASREngine._decode (LLM decode 走 Metal)
    from src.core.vendor.qwen_asr_gguf.inference import asr as _asr_mod

    QwenASREngine = _asr_mod.QwenASREngine
    _orig_decode = QwenASREngine._decode
    _orig_safe_decode = QwenASREngine._safe_decode

    def _patched_decode(self, *a, **kw):
        t0 = time.monotonic()
        try:
            return _orig_decode(self, *a, **kw)
        finally:
            t1 = time.monotonic()
            _T.record("asr.llm_decode", t1 - t0, t0, t1)

    def _patched_safe_decode(self, *a, **kw):
        t0 = time.monotonic()
        try:
            return _orig_safe_decode(self, *a, **kw)
        finally:
            t1 = time.monotonic()
            _T.record("asr.safe_decode_total", t1 - t0, t0, t1)

    QwenASREngine._decode = _patched_decode
    QwenASREngine._safe_decode = _patched_safe_decode

    # 3. vendor audio load
    from src.core.vendor.qwen_asr_gguf.inference import audio as _audio_mod
    _orig_load_audio = _audio_mod.load_audio

    def _patched_load_audio(*a, **kw):
        t0 = time.monotonic()
        try:
            return _orig_load_audio(*a, **kw)
        finally:
            t1 = time.monotonic()
            _T.record("audio.load", t1 - t0, t0, t1)

    _audio_mod.load_audio = _patched_load_audio
    # 也要 patch asr.py wrapper 里的 _load_audio_file 间接调用
    from src.core.qwen3 import asr as _qwen3_asr
    _orig_load_audio_file = _qwen3_asr._load_audio_file

    def _patched_load_audio_file(*a, **kw):
        t0 = time.monotonic()
        try:
            return _orig_load_audio_file(*a, **kw)
        finally:
            t1 = time.monotonic()
            _T.record("audio.load_file_wrap", t1 - t0, t0, t1)

    _qwen3_asr._load_audio_file = _patched_load_audio_file

    # 4. sherpa diarize 整体
    from src.core.qwen3 import diarize as _diarize_mod
    _orig_run_diarize = _diarize_mod.run_diarization

    def _patched_run_diarize(*a, **kw):
        t0 = time.monotonic()
        try:
            return _orig_run_diarize(*a, **kw)
        finally:
            t1 = time.monotonic()
            _T.record("sherpa.diarize.total", t1 - t0, t0, t1)

    _diarize_mod.run_diarization = _patched_run_diarize

    # sherpa pipeline.process (segmentation + embedding 都在内部)
    # 通过 _build_pipeline 拿到 pipeline 实例, 但 pipeline 是临时构造的 — 转向 patch sherpa_onnx.OfflineSpeakerDiarization
    try:
        import sherpa_onnx
        _orig_pipe_process = sherpa_onnx.OfflineSpeakerDiarization.process

        def _patched_pipe_process(self, audio):
            t0 = time.monotonic()
            try:
                return _orig_pipe_process(self, audio)
            finally:
                t1 = time.monotonic()
                _T.record("sherpa.pipeline.process", t1 - t0, t0, t1)

        sherpa_onnx.OfflineSpeakerDiarization.process = _patched_pipe_process

        # sherpa Embedding compute (cluster_merge per-segment 用)
        _orig_emb_compute = sherpa_onnx.SpeakerEmbeddingExtractor.compute

        def _patched_emb_compute(self, stream):
            t0 = time.monotonic()
            try:
                return _orig_emb_compute(self, stream)
            finally:
                t1 = time.monotonic()
                _T.record("sherpa.embedding.compute", t1 - t0, t0, t1)

        sherpa_onnx.SpeakerEmbeddingExtractor.compute = _patched_emb_compute
    except Exception as e:
        print(f"[timing_hooks] patch sherpa pipeline failed: {e}", file=sys.stderr)

    # 5. cluster_merge 入口
    from src.core.qwen3 import cluster_merge as _cm_mod
    _orig_apply_cm = _cm_mod.apply_cluster_centroid_merge

    def _patched_apply_cm(*a, **kw):
        t0 = time.monotonic()
        try:
            return _orig_apply_cm(*a, **kw)
        finally:
            t1 = time.monotonic()
            _T.record("cluster_merge.apply", t1 - t0, t0, t1)

    _cm_mod.apply_cluster_centroid_merge = _patched_apply_cm

    # 6. ASR wrapper run_asr (audio_load + encode + decode 总和)
    _orig_run_asr_loaded = _qwen3_asr._run_asr_loaded_audio

    def _patched_run_asr_loaded(*a, **kw):
        t0 = time.monotonic()
        try:
            return _orig_run_asr_loaded(*a, **kw)
        finally:
            t1 = time.monotonic()
            _T.record("asr.run_total", t1 - t0, t0, t1)

    _qwen3_asr._run_asr_loaded_audio = _patched_run_asr_loaded


def fmt_report() -> str:
    rep = get_timer().report()
    lines = []
    lines.append("=" * 70)
    lines.append(f"WALL: {rep['wall']:.2f}s")
    lines.append(f"{'stage':<40} {'total(s)':>10} {'count':>8} {'avg(s)':>10}")
    lines.append("-" * 70)
    for stage, total in sorted(rep["totals"].items(), key=lambda kv: -kv[1]):
        cnt = rep["counts"][stage]
        avg = total / cnt if cnt else 0
        lines.append(f"{stage:<40} {total:>10.2f} {cnt:>8d} {avg:>10.3f}")
    return "\n".join(lines)
