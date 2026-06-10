"""ort_cuda diarize backend (sherpa pipeline 忠实移植) 单测.

覆盖 (对照 sherpa-onnx offline-speaker-diarization-pyannote-impl.h):
1. run_segmentation_chunks: per-chunk 独立 argmax multilabel (不跨 chunk 平均 logits)
2. compute_speakers_per_frame: 全局 frame 网格 per-frame 说话人数 (count/weight 四舍五入)
3. exclude_overlap: 同帧 ≥2 speaker 活跃整帧清零
4. get_chunk_speaker_sample_indexes: per-(chunk,speaker) 采样区间, <10 活跃帧跳过
5. relabel_chunks / finalize_labels: cluster 重标 + per-frame top-k
6. labels_to_turns: 帧→秒映射 (receptive field scale/offset), gap 合并, min_duration_on 过滤
7. run_diarization_ort_cuda 端到端: 静音 / 单说话人(单 chunk 特例) / slot 置换根因回归
8. run_diarization_dispatched: backend 路由

ORT session 全 mock, 不真实加载. Integration parity 测试在 tests/integration/.

根因背景 (2026-06-10): 旧实现跨 chunk 平均 powerset logits — pyannote 的 speaker
slot 是 chunk 局部的, 跨 chunk 平均会把不同说话人混进同一 slot, 提出来的 turn 含
混合语音, per-turn embedding 全是"混合脸"→ 短音频聚成 1 簇 (under-detect).
sherpa 的正确做法: 每 chunk 独立 argmax, embedding 按 (chunk, local_speaker) 提取.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.qwen3.diarize_ort import (
    PYANNOTE_CHUNK_SAMPLES,
    PYANNOTE_RECEPTIVE_FIELD_SHIFT,
    PYANNOTE_RECEPTIVE_FIELD_SIZE,
    PYANNOTE_STEP_SAMPLES,
)

# 帧→秒映射常量 (sherpa ComputeResult)
_SCALE = PYANNOTE_RECEPTIVE_FIELD_SHIFT / 16000.0  # 0.016875 s/frame
_OFFSET = 0.5 * PYANNOTE_RECEPTIVE_FIELD_SIZE / 16000.0  # ~0.031 s


# ==================== compute_speakers_per_frame ====================


def test_speakers_per_frame_two_chunks_single_speaker_each():
    """chunk0 slot0 全活跃 + chunk1 slot1 全活跃 → 重叠区平均后仍是 1 人/帧."""
    from src.core.qwen3.diarize_ort import compute_speakers_per_frame

    c0 = np.zeros((589, 3), dtype=np.int8)
    c0[:, 0] = 1
    c1 = np.zeros((589, 3), dtype=np.int8)
    c1[:, 1] = 1
    spf = compute_speakers_per_frame([c0, c1])
    # num_frames = (160000 + 16000) // 270 + 1 = 652
    assert spf.shape[0] == (PYANNOTE_CHUNK_SAMPLES + PYANNOTE_STEP_SAMPLES) // PYANNOTE_RECEPTIVE_FIELD_SHIFT + 1
    assert spf[0] == 1
    assert spf[300] == 1  # 两 chunk 重叠区: count=2/weight=2 → 1
    assert spf[-1] == 0  # 尾部无 chunk 覆盖


def test_speakers_per_frame_all_silence_returns_zeros():
    from src.core.qwen3.diarize_ort import compute_speakers_per_frame

    chunks = [np.zeros((589, 3), dtype=np.int8) for _ in range(3)]
    spf = compute_speakers_per_frame(chunks)
    assert int(spf.max()) == 0


# ==================== exclude_overlap ====================


def test_exclude_overlap_zeroes_multi_speaker_frames():
    from src.core.qwen3.diarize_ort import exclude_overlap

    label = np.zeros((10, 3), dtype=np.int8)
    label[2, 1] = 1  # 单 speaker 帧 — 保留
    label[5, 0] = 1
    label[5, 1] = 1  # 双 speaker 帧 — 清零
    out = exclude_overlap(label)
    assert out[2, 1] == 1
    assert out[5].sum() == 0
    # 原 label 不被原地修改
    assert label[5].sum() == 2


# ==================== get_chunk_speaker_sample_indexes ====================


def test_sample_indexes_maps_frames_to_samples_with_chunk_offset():
    """chunk1 的 slot0 全程活跃 → 采样区间带 chunk offset (sherpa 帧→样点映射)."""
    from src.core.qwen3.diarize_ort import get_chunk_speaker_sample_indexes

    c0 = np.zeros((589, 3), dtype=np.int8)
    c0[0:100, 1] = 1  # chunk0 slot1: 100 帧 ≥ 10 → 保留
    c0[200:205, 2] = 1  # chunk0 slot2: 5 帧 < 10 → 跳过
    c1 = np.zeros((589, 3), dtype=np.int8)
    c1[:, 0] = 1  # chunk1 slot0: 全程活跃 (末尾 is_active 收尾分支)
    pairs, ranges = get_chunk_speaker_sample_indexes([c0, c1])

    assert pairs == [(0, 1), (1, 0)]
    # chunk0 slot1: [0, 100) 帧 → [0, int(100/589*160000)) 样点
    assert ranges[0] == [(0, int(100 / 589 * PYANNOTE_CHUNK_SAMPLES))]
    # chunk1 slot0: 活跃到末尾 → end 用 (num_frames-1)/num_frames 映射 + offset 16000
    expect_end = int((589 - 1) / 589 * PYANNOTE_CHUNK_SAMPLES) + PYANNOTE_STEP_SAMPLES
    assert ranges[1] == [(PYANNOTE_STEP_SAMPLES, expect_end)]


def test_sample_indexes_excludes_overlap_frames():
    """重叠帧先剔除再提取: slot0 与 slot1 同帧活跃的区间不进任何 speaker 的采样段."""
    from src.core.qwen3.diarize_ort import get_chunk_speaker_sample_indexes

    c0 = np.zeros((589, 3), dtype=np.int8)
    c0[0:100, 0] = 1
    c0[50:100, 1] = 1  # [50,100) 双人重叠 → 剔除后 slot0 只剩 [0,50), slot1 全没
    pairs, ranges = get_chunk_speaker_sample_indexes([c0])
    assert pairs == [(0, 0)]
    assert ranges[0] == [(0, int(50 / 589 * PYANNOTE_CHUNK_SAMPLES))]


# ==================== relabel_chunks / finalize_labels ====================


def test_relabel_chunks_maps_local_slots_to_cluster_columns():
    from src.core.qwen3.diarize_ort import relabel_chunks

    label = np.zeros((10, 3), dtype=np.int8)
    label[0:4, 0] = 1  # slot0 → cluster 1
    label[6:9, 2] = 1  # slot2 → cluster 0
    label[5, 1] = 1  # slot1 不在映射里 (embedding 被跳过) → 丢弃
    out = relabel_chunks([label], {(0, 0): 1, (0, 2): 0}, num_clusters=2)
    assert len(out) == 1
    assert out[0].shape == (10, 2)
    assert out[0][2, 1] == 1 and out[0][2, 0] == 0
    assert out[0][7, 0] == 1 and out[0][7, 1] == 0
    assert out[0][5].sum() == 0


def test_finalize_labels_picks_topk_clusters_per_frame():
    from src.core.qwen3.diarize_ort import finalize_labels

    count = np.array([[5, 3], [0, 0], [2, 4]], dtype=np.int32)
    spf = np.array([1, 0, 2], dtype=np.int32)
    out = finalize_labels(count, spf)
    assert out[0].tolist() == [1, 0]  # top-1 → cluster0
    assert out[1].tolist() == [0, 0]  # 0 人帧
    assert out[2].tolist() == [1, 1]  # top-2 → 全选


# ==================== labels_to_turns ====================


def test_labels_to_turns_frame_to_seconds_mapping():
    """活跃帧 [10, 100) → [10*scale+offset, 100*scale+offset] 秒."""
    from src.core.qwen3.diarize_ort import labels_to_turns

    final = np.zeros((600, 1), dtype=np.int8)
    final[10:100, 0] = 1
    turns = labels_to_turns(final, min_duration_on=0.3, min_duration_off=0.5)
    assert len(turns) == 1
    np.testing.assert_allclose(turns[0]["start"], 10 * _SCALE + _OFFSET, atol=1e-6)
    np.testing.assert_allclose(turns[0]["end"], 100 * _SCALE + _OFFSET, atol=1e-6)
    assert turns[0]["speaker"] == 0


def test_labels_to_turns_merges_gap_below_min_duration_off():
    """同 speaker 两段 gap 0.169s < min_duration_off=0.5 → 合并为一段."""
    from src.core.qwen3.diarize_ort import labels_to_turns

    final = np.zeros((600, 1), dtype=np.int8)
    final[10:100, 0] = 1
    final[110:200, 0] = 1  # gap = 10 帧 ≈ 0.169s
    turns = labels_to_turns(final, min_duration_on=0.3, min_duration_off=0.5)
    assert len(turns) == 1
    np.testing.assert_allclose(turns[0]["end"], 200 * _SCALE + _OFFSET, atol=1e-6)


def test_labels_to_turns_keeps_gap_above_min_duration_off():
    from src.core.qwen3.diarize_ort import labels_to_turns

    final = np.zeros((600, 1), dtype=np.int8)
    final[10:100, 0] = 1
    final[110:200, 0] = 1
    turns = labels_to_turns(final, min_duration_on=0.3, min_duration_off=0.1)
    assert len(turns) == 2


def test_labels_to_turns_drops_segment_below_min_duration_on():
    """0.17s 段 ≤ min_duration_on=0.3 → 丢弃."""
    from src.core.qwen3.diarize_ort import labels_to_turns

    final = np.zeros((600, 2), dtype=np.int8)
    final[10:20, 0] = 1  # 10 帧 ≈ 0.169s — 丢
    final[100:300, 1] = 1  # 200 帧 ≈ 3.4s — 留
    turns = labels_to_turns(final, min_duration_on=0.3, min_duration_off=0.5)
    assert len(turns) == 1
    assert turns[0]["speaker"] == 1


# ==================== mock sessions ====================


def _make_fake_pyannote_session(class_idx: int):
    sess = MagicMock()
    fake_in = MagicMock()
    fake_in.name = "x"
    sess.get_inputs.return_value = [fake_in]

    def fake_run(_, feed):
        out = np.zeros((1, 589, 7), dtype=np.float32)
        out[..., class_idx] = 1.0
        return [out]

    sess.run.side_effect = fake_run
    return sess


def _make_fake_titanet_session(embedding_vec: np.ndarray):
    sess = MagicMock()
    in_audio = MagicMock()
    in_audio.name = "audio_signal"
    in_len = MagicMock()
    in_len.name = "length"
    sess.get_inputs.return_value = [in_audio, in_len]

    def fake_run(_, feed):
        return [embedding_vec.reshape(1, -1).astype(np.float32)]

    sess.run.side_effect = fake_run
    return sess


# ==================== run_diarization_ort_cuda 端到端 ====================


def test_run_diarization_ort_cuda_returns_empty_for_silent_audio(tmp_path):
    """全 silence 音频 → pyannote 全 class 0 → 空 list."""
    import soundfile as sf

    from src.core.qwen3.diarize_ort import run_diarization_ort_cuda

    audio_path = tmp_path / "silence.wav"
    sf.write(str(audio_path), np.zeros(16000 * 5, dtype=np.float32), 16000)

    with patch(
        "src.core.qwen3.diarize_ort._get_pyannote_session",
        return_value=_make_fake_pyannote_session(class_idx=0),
    ), patch(
        "src.core.qwen3.diarize_ort._get_titanet_session",
        return_value=_make_fake_titanet_session(np.ones(192, dtype=np.float32)),
    ):
        turns = run_diarization_ort_cuda(
            str(audio_path),
            segmentation_model="/fake/seg.onnx",
            embedding_model="/fake/emb.onnx",
        )
    assert turns == []


def test_run_diarization_ort_cuda_single_chunk_special_case(tmp_path):
    """5s 音频 (< 10s 窗) = 单 chunk: 不聚类, 直接用 chunk 局部 slot 输出 (sherpa 特例)."""
    import soundfile as sf

    from src.core.qwen3.diarize_ort import run_diarization_ort_cuda

    audio_path = tmp_path / "voice.wav"
    rng = np.random.RandomState(0)
    sf.write(str(audio_path), (rng.randn(16000 * 5) * 0.1).astype(np.float32), 16000)

    titanet = _make_fake_titanet_session(np.ones(192, dtype=np.float32))
    with patch(
        "src.core.qwen3.diarize_ort._get_pyannote_session",
        return_value=_make_fake_pyannote_session(class_idx=1),
    ), patch(
        "src.core.qwen3.diarize_ort._get_titanet_session",
        return_value=titanet,
    ):
        turns = run_diarization_ort_cuda(
            str(audio_path),
            segmentation_model="/fake/seg.onnx",
            embedding_model="/fake/emb.onnx",
        )
    assert len(turns) >= 1
    for t in turns:
        assert t["speaker"] == 0
        assert t["end"] > t["start"]
    # 单 chunk 特例不应跑 embedding
    titanet.run.assert_not_called()


def test_run_diarization_ort_cuda_survives_slot_permutation_across_chunks(tmp_path):
    """根因回归: pyannote 的 speaker slot 跨 chunk 置换时, 两个说话人不得被合并.

    场景: 20s 音频, 前 10s 说话人 A (DC +0.5), 后 10s 说话人 B (DC -0.5).
    fake pyannote 每个 chunk 用不同的 slot 编号标注 A/B (slot = chunk_idx 轮转),
    模拟真实 pyannote 'slot 是 chunk 局部' 的行为. 旧实现跨 chunk 平均 logits 会
    把 A/B 混进同一 activity → 1 簇; sherpa 式 per-(chunk,slot) embedding 必须出 2 簇.
    """
    import soundfile as sf

    from src.core.qwen3.diarize_ort import run_diarization_ort_cuda

    n = 16000 * 20
    boundary = 16000 * 10
    audio = np.concatenate(
        [np.full(boundary, 0.5, dtype=np.float32), np.full(n - boundary, -0.5, dtype=np.float32)]
    )
    audio_path = tmp_path / "two_spk.wav"
    sf.write(str(audio_path), audio, 16000)

    # fake pyannote: 第 i 个 chunk 把 A 标到 slot i%3, B 标到 slot (i+1)%3
    call_idx = {"i": 0}
    sess = MagicMock()
    fake_in = MagicMock()
    fake_in.name = "x"
    sess.get_inputs.return_value = [fake_in]

    def fake_seg_run(_, feed):
        i = call_idx["i"]
        call_idx["i"] += 1
        start_sample = i * PYANNOTE_STEP_SAMPLES
        f_b = int(np.clip(round((boundary - start_sample) / PYANNOTE_CHUNK_SAMPLES * 589), 0, 589))
        out = np.zeros((1, 589, 7), dtype=np.float32)
        out[0, :f_b, 1 + i % 3] = 1.0  # A → 单人 class (slot i%3)
        out[0, f_b:, 1 + (i + 1) % 3] = 1.0  # B → 单人 class (slot (i+1)%3)
        return [out]

    sess.run.side_effect = fake_seg_run

    # fake embedding: 按音频 DC 符号区分说话人 (A→e1, B→e2, 正交)
    e1 = np.zeros(192, dtype=np.float32)
    e1[0] = 1.0
    e2 = np.zeros(192, dtype=np.float32)
    e2[1] = 1.0

    def fake_embedding(seg_audio, _sess, **kwargs):
        return e1 if float(np.mean(seg_audio)) > 0 else e2

    with patch(
        "src.core.qwen3.diarize_ort._get_pyannote_session", return_value=sess
    ), patch(
        "src.core.qwen3.diarize_ort._get_titanet_session", return_value=MagicMock()
    ), patch(
        "src.core.qwen3.diarize_ort.compute_titanet_embedding", side_effect=fake_embedding
    ):
        turns = run_diarization_ort_cuda(
            str(audio_path),
            segmentation_model="/fake/seg.onnx",
            embedding_model="/fake/emb.onnx",
            cluster_threshold=0.9,
        )

    speakers = {t["speaker"] for t in turns}
    assert len(speakers) == 2, f"slot 置换下应保住 2 speakers, got {turns}"
    # 时间结构: 5s 处与 15s 处属于不同说话人
    def speaker_at(ts: float) -> int:
        for t in turns:
            if t["start"] <= ts <= t["end"]:
                return t["speaker"]
        raise AssertionError(f"{ts}s 没有 turn 覆盖: {turns}")

    assert speaker_at(5.0) != speaker_at(15.0)


# ==================== run_diarization_dispatched ====================


def test_dispatch_default_routes_to_sherpa_on_mac():
    """无 env override 时 Mac 上应该路由到 sherpa."""
    import sys

    from src.core.qwen3 import diarize as diarize_mod

    with patch.object(sys, "platform", "darwin"), patch.object(
        diarize_mod, "run_diarization", return_value=[{"start": 0.0, "end": 1.0, "speaker": 0}]
    ) as sherpa_mock, patch(
        "src.core.qwen3.diarize_ort.run_diarization_ort_cuda",
        return_value=[],
    ) as ort_mock:
        result = diarize_mod.run_diarization_dispatched(
            "/fake.wav",
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
        )
    sherpa_mock.assert_called_once()
    ort_mock.assert_not_called()
    assert result[0]["speaker"] == 0


def test_dispatch_env_override_forces_ort_cuda(monkeypatch):
    """FUNASR_QWEN3_DIARIZE_BACKEND=ort_cuda 强制走 ORT backend."""
    monkeypatch.setenv("FUNASR_QWEN3_DIARIZE_BACKEND", "ort_cuda")
    from src.core.qwen3 import diarize as diarize_mod

    with patch.object(diarize_mod, "run_diarization") as sherpa_mock, patch(
        "src.core.qwen3.diarize_ort.run_diarization_ort_cuda",
        return_value=[{"start": 0.0, "end": 1.0, "speaker": 0}],
    ) as ort_mock:
        result = diarize_mod.run_diarization_dispatched(
            "/fake.wav",
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
        )
    sherpa_mock.assert_not_called()
    ort_mock.assert_called_once()
    assert result[0]["speaker"] == 0


def test_dispatch_explicit_backend_param_overrides_runtime():
    """显式 backend='sherpa' 参数 > runtime 推荐."""
    from src.core.qwen3 import diarize as diarize_mod

    with patch.object(diarize_mod, "run_diarization", return_value=[]) as sherpa_mock, patch(
        "src.core.qwen3.diarize_ort.run_diarization_ort_cuda"
    ) as ort_mock:
        diarize_mod.run_diarization_dispatched(
            "/fake.wav",
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
            backend="sherpa",
        )
    sherpa_mock.assert_called_once()
    ort_mock.assert_not_called()
