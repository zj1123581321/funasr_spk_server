"""
cluster_merge embedding extractor 122.88s 段长崩溃修复 (2026-06-10).

TitaNet ONNX 导出图的 Where mask 维硬编码 12288 帧 (x10ms hop = 122.88s),
>=123s 的单段输入直接 RuntimeError (sherpa wrapper 与 ORT 直跑同边界).
两层修复:
- A (治根): build_embedding_extractor_fn 段长上限 — 超限段切等宽窗逐窗
  embedding 后平均再 L2 归一化, 底层 extractor 永远收到 <= 上限的段
- B (兜底): build_centroids per-段容错 — 单段 embedding 失败跳过该段,
  不再让整层 cluster_merge 因一段阵亡

交接文档: docs/开发/2026-06-10-cluster_merge-extractor-122s崩溃-新session-prompt.md
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src.core.qwen3_transcriber import (
    MAX_EXTRACTOR_SEGMENT_SEC,
    _cap_extractor_fn,
)

SR = 16000


def _make_raw_recorder(emb=None):
    """构造记录每次调用 (start, end) 的 raw extractor stub.

    emb: 返回的固定 embedding; None 用默认单位向量.
    """
    calls: list[tuple[float, float]] = []
    if emb is None:
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def raw_fn(audio_16k, start, end):
        calls.append((float(start), float(end)))
        return emb

    return raw_fn, calls


class TestCapExtractorFn:
    """方案 A 核心: _cap_extractor_fn 包装后的段长不变量."""

    def test_cap_constant_below_model_limit(self) -> None:
        """上限必须 < 122.88s (TitaNet 导出图硬上限), 留安全余量."""
        assert MAX_EXTRACTOR_SEGMENT_SEC < 122.88

    def test_short_segment_passthrough(self) -> None:
        """<= 上限的段原样透传 raw, 只调一次, span 不变."""
        raw_fn, calls = _make_raw_recorder()
        audio = np.zeros(SR * 130, dtype=np.float32)
        fn = _cap_extractor_fn(raw_fn)
        out = fn(audio, 3.0, 100.0)
        assert calls == [(3.0, 100.0)]
        assert out is not None

    def test_long_segment_every_raw_call_below_cap(self) -> None:
        """>上限的段切窗后, 底层每次调用跨度 <= 上限 (钉死崩溃不变量)."""
        raw_fn, calls = _make_raw_recorder()
        audio = np.zeros(SR * 400, dtype=np.float32)
        fn = _cap_extractor_fn(raw_fn)
        fn(audio, 10.0, 370.0)  # 360s 段
        assert len(calls) >= 3
        for start, end in calls:
            assert end - start <= MAX_EXTRACTOR_SEGMENT_SEC + 1e-6, (
                f"底层调用 [{start:.2f},{end:.2f}] 超过上限 {MAX_EXTRACTOR_SEGMENT_SEC}s"
            )

    def test_long_segment_windows_cover_full_span(self) -> None:
        """切出的窗连续覆盖整段 (无缝隙无重叠), 不丢音频信息."""
        raw_fn, calls = _make_raw_recorder()
        audio = np.zeros(SR * 300, dtype=np.float32)
        fn = _cap_extractor_fn(raw_fn)
        fn(audio, 0.0, 250.0)
        calls_sorted = sorted(calls)
        assert calls_sorted[0][0] == pytest.approx(0.0)
        assert calls_sorted[-1][1] == pytest.approx(250.0)
        for (_, prev_end), (next_start, _) in zip(calls_sorted, calls_sorted[1:]):
            assert next_start == pytest.approx(prev_end)

    def test_long_segment_result_is_l2_normalized_mean(self) -> None:
        """多窗 embedding 平均后 L2 归一化."""
        embs = iter([
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ])

        def raw_fn(audio_16k, start, end):
            return next(embs)

        audio = np.zeros(SR * 250, dtype=np.float32)
        fn = _cap_extractor_fn(raw_fn, max_segment_sec=120.0)
        out = fn(audio, 0.0, 240.0)  # 240s → 正好 2 窗
        expected = np.array([0.5, 0.5]) / np.linalg.norm([0.5, 0.5])
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_window_returning_none_skipped(self) -> None:
        """某窗 raw 返回 None (太短/越界) → 跳过, 用其余窗平均."""
        results = iter([np.array([1.0, 0.0], dtype=np.float32), None])

        def raw_fn(audio_16k, start, end):
            return next(results)

        audio = np.zeros(SR * 250, dtype=np.float32)
        fn = _cap_extractor_fn(raw_fn, max_segment_sec=120.0)
        out = fn(audio, 0.0, 240.0)
        np.testing.assert_allclose(out, np.array([1.0, 0.0]), rtol=1e-6)

    def test_all_windows_none_returns_none(self) -> None:
        """所有窗都 None → 整段返回 None (与 raw 语义一致)."""
        def raw_fn(audio_16k, start, end):
            return None

        audio = np.zeros(SR * 250, dtype=np.float32)
        fn = _cap_extractor_fn(raw_fn, max_segment_sec=120.0)
        assert fn(audio, 0.0, 240.0) is None

    def test_end_clamped_to_audio_length_before_windowing(self) -> None:
        """end 超过音频实际长度先 clamp 再判窗 — 100s 音频请求 [0,300) 实际只有
        100s, 不应切窗 (避免无意义的越界窗)."""
        raw_fn, calls = _make_raw_recorder()
        audio = np.zeros(SR * 100, dtype=np.float32)
        fn = _cap_extractor_fn(raw_fn, max_segment_sec=120.0)
        fn(audio, 0.0, 300.0)
        assert calls == [(0.0, 100.0)]


class TestBuildEmbeddingExtractorFnWiring:
    """方案 A 接线: build_embedding_extractor_fn 返回的 callable 已带段长上限."""

    def test_long_request_never_reaches_sherpa_above_cap(self, monkeypatch) -> None:
        """fake sherpa_onnx 记录每次 accept_waveform 的样本数, 360s 请求下
        每次 <= 上限 (即真实 TitaNet 不会再见到 >122.88s 输入)."""
        waveform_lens: list[int] = []

        class FakeStream:
            pass

        class FakeExtractor:
            def __init__(self, cfg):
                pass

            def create_stream(self):
                return FakeStream()

            def is_ready(self, stream):
                return True

            def compute(self, stream):
                return [1.0, 0.0, 0.0]

        class FakeConfig:
            def __init__(self, **kwargs):
                pass

            def validate(self):
                return True

        fake_sherpa = types.SimpleNamespace(
            SpeakerEmbeddingExtractorConfig=FakeConfig,
            SpeakerEmbeddingExtractor=FakeExtractor,
        )

        def fake_accept(self, sample_rate, waveform):
            waveform_lens.append(len(waveform))

        FakeStream.accept_waveform = fake_accept
        FakeStream.input_finished = lambda self: None

        monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_sherpa)

        from src.core.qwen3_transcriber import build_embedding_extractor_fn

        cfg_like = types.SimpleNamespace(
            embedding_model="/fake/titanet.onnx", num_threads=2, provider="cpu"
        )
        fn = build_embedding_extractor_fn(cfg_like)
        audio = np.zeros(SR * 400, dtype=np.float32)
        out = fn(audio, 0.0, 360.0)
        assert out is not None
        assert len(waveform_lens) >= 3
        max_samples = int(MAX_EXTRACTOR_SEGMENT_SEC * SR)
        for n in waveform_lens:
            assert n <= max_samples, f"sherpa 收到 {n} 样本 (> 上限 {max_samples})"


class TestBuildCentroidsFaultTolerance:
    """方案 B: build_centroids 单段 embedding 失败跳过该段, 不炸整层."""

    def test_single_segment_failure_skipped(self) -> None:
        """3 段中 1 段抛 RuntimeError → centroid 用剩余 2 段算, 不抛."""
        from src.core.qwen3.cluster_merge import build_centroids

        def raw_fn(audio_16k, start, end):
            if start == 10.0:
                raise RuntimeError("BroadcastIterator::Init 12288 by 23313")
            return np.array([1.0, 0.0], dtype=np.float32)

        segs = {
            "0": [
                {"start": 0.0, "end": 5.0},
                {"start": 10.0, "end": 200.0},  # 这段崩
                {"start": 300.0, "end": 305.0},
            ],
        }
        audio = np.zeros(SR * 400, dtype=np.float32)
        centroids = build_centroids(raw_fn, audio, segs)
        assert "0" in centroids
        np.testing.assert_allclose(centroids["0"], np.array([1.0, 0.0]), rtol=1e-6)

    def test_all_segments_fail_speaker_skipped_without_raise(self) -> None:
        """某 speaker 所有段都失败 → 该 speaker 无 centroid, 其他 speaker 正常."""
        from src.core.qwen3.cluster_merge import build_centroids

        def raw_fn(audio_16k, start, end):
            if end - start > 100.0:
                raise RuntimeError("boom")
            return np.array([0.0, 1.0], dtype=np.float32)

        segs = {
            "0": [{"start": 0.0, "end": 150.0}],  # 全崩
            "1": [{"start": 200.0, "end": 210.0}],
        }
        audio = np.zeros(SR * 300, dtype=np.float32)
        centroids = build_centroids(raw_fn, audio, segs)
        assert "0" not in centroids
        assert "1" in centroids
