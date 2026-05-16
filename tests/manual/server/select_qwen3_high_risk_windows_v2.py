"""Audio-aware high-risk window detector v2 (no reference required).

Adds audio-side features on top of v1 hypothesis-only signals:

  * mini local sherpa diarization per 40s window (num_speakers=2),
    yielding: turn count, distinct local speakers, local switches,
    speech ratio, shortest/longest turn duration.

  * speaker embedding consistency between hypothesis label and the
    audio interval embedding compared against global dominant /
    minority centroids (built from the same hypothesis JSON).

  * within-window consecutive embedding distance (detects speaker
    switches inside a single hypothesis segment).

  * lightweight ffmpeg silencedetect coverage per window.

  * v1 hypothesis features (kept with reduced weight as priors).

Outputs a ranked JSON report + top20/40/60/80 id files compatible with
``align_qwen3_poc_with_window_diarize.py --window-ids``.

Typical run:

    DYLD_LIBRARY_PATH="$PWD/src/core/vendor/qwen_asr_gguf/inference/bin" \
    venv/bin/python tests/manual/server/select_qwen3_high_risk_windows_v2.py \
      tmp_long_audio/poc_outputs_v7_speaker_smooth/audio_149min.qwen3_long_poc.json \
      tmp_long_audio/audio_149min.mp3 \
      --out-dir tmp_long_audio/detector_v2
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import sherpa_onnx
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import config  # noqa: E402

# ---------------------------------------------------------------------------
# v1 lexical patterns (kept for backwards compatibility & priors)
# ---------------------------------------------------------------------------
QUESTION_PATTERNS = [
    r"你觉得", r"你们", r"可以讲讲", r"能不能", r"会不会", r"有没有",
    r"是不是", r"是什么", r"为什么", r"商业模式", r"接下来", r"同意.{0,8}吗",
    r"吗[？?]?", r"[？?]",
]
ANSWER_PATTERNS = [
    r"我觉得", r"在我看来", r"我相信", r"我们公司", r"我们自己", r"我是",
    r"我的", r"我刚", r"对我来讲", r"龙虾", r"章鱼", r"明略", r"开源",
]
TARGET_SR = 16000


def score_patterns(patterns: list[str], text: str) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text))


# ---------------------------------------------------------------------------
# audio loading + sherpa pipelines
# ---------------------------------------------------------------------------
def load_audio_mono_16k(audio_path: Path) -> np.ndarray:
    """Load entire audio file as float32 mono @ 16 kHz once and keep in RAM."""
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    audio = audio[:, 0]
    if sr != TARGET_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32, copy=False)


def build_diarization_pipeline(num_speakers: Optional[int]) -> sherpa_onnx.OfflineSpeakerDiarization:
    """Reusable sherpa diarization pipeline (one per num_speakers setting)."""
    q = config.qwen3
    num_clusters = -1 if num_speakers is None else int(num_speakers)
    sherpa_cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=q.segmentation_model,
            ),
            num_threads=4,
            provider=q.provider,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=q.embedding_model,
            num_threads=4,
            provider=q.provider,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_clusters,
            threshold=q.cluster_threshold,
        ),
        min_duration_on=0.2,
        min_duration_off=0.2,
    )
    if not sherpa_cfg.validate():
        raise RuntimeError("OfflineSpeakerDiarizationConfig validation failed")
    return sherpa_onnx.OfflineSpeakerDiarization(sherpa_cfg)


def build_embedding_extractor() -> sherpa_onnx.SpeakerEmbeddingExtractor:
    q = config.qwen3
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=q.embedding_model,
        num_threads=8,
        provider=q.provider,
    )
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


def embedding_for_interval(
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
    audio: np.ndarray,
    start: float,
    end: float,
    sr: int = TARGET_SR,
) -> Optional[np.ndarray]:
    s = max(0, int(start * sr))
    e = min(len(audio), int(end * sr))
    if e - s < int(0.5 * sr):  # need ≥0.5s to be reliable
        return None
    stream = extractor.create_stream()
    stream.accept_waveform(sr, audio[s:e])
    stream.input_finished()
    if not extractor.is_ready(stream):
        return None
    emb = np.asarray(extractor.compute(stream), dtype=np.float32)
    norm = float(np.linalg.norm(emb))
    return emb / norm if norm > 0 else emb


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else -1.0


# ---------------------------------------------------------------------------
# global centroids per hypothesis speaker label
# ---------------------------------------------------------------------------
def build_centroids(
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
    audio: np.ndarray,
    segments: list[dict],
    max_per_speaker: int = 80,
    min_dur: float = 4.0,
    min_chars: int = 20,
) -> dict[str, np.ndarray]:
    """Average embedding per hypothesis speaker label using long, clean segments."""
    by_sp: dict[str, list[dict]] = defaultdict(list)
    for seg in segments:
        dur = float(seg["end"]) - float(seg["start"])
        if dur >= min_dur and len(str(seg.get("text", ""))) >= min_chars:
            by_sp[str(seg["speaker"])].append(seg)

    centroids: dict[str, np.ndarray] = {}
    for sp, segs in by_sp.items():
        chosen = sorted(segs, key=lambda s: float(s["end"]) - float(s["start"]), reverse=True)[:max_per_speaker]
        embs = []
        for seg in chosen:
            emb = embedding_for_interval(extractor, audio, float(seg["start"]), float(seg["end"]))
            if emb is not None:
                embs.append(emb)
        if embs:
            c = np.mean(np.stack(embs), axis=0)
            norm = float(np.linalg.norm(c))
            centroids[sp] = c / norm if norm > 0 else c
    return centroids


# ---------------------------------------------------------------------------
# ffmpeg silencedetect helper (single pass over the whole audio is fastest)
# ---------------------------------------------------------------------------
def detect_silences(
    audio_path: Path,
    noise_db: float = -30.0,
    min_silence_dur: float = 0.4,
) -> list[tuple[float, float]]:
    """Return list of (start, end) silence intervals using ffmpeg silencedetect."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-i", str(audio_path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_dur}",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        print("[warn] ffmpeg not on PATH, silence features disabled", file=sys.stderr)
        return []
    log = proc.stderr or ""
    # Lines like: "silence_start: 12.345" and "silence_end: 13.789 | silence_duration: 1.444"
    silences: list[tuple[float, float]] = []
    cur_start: Optional[float] = None
    for line in log.splitlines():
        m_start = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        m_end = re.search(r"silence_end:\s*(-?[\d.]+)\s*\|", line)
        if m_start:
            cur_start = float(m_start.group(1))
        elif m_end and cur_start is not None:
            silences.append((cur_start, float(m_end.group(1))))
            cur_start = None
    return silences


def silence_overlap(silences: list[tuple[float, float]], wstart: float, wend: float) -> tuple[int, float]:
    """Return (count, total_overlap_seconds) of silences within [wstart, wend]."""
    cnt = 0
    total = 0.0
    for s, e in silences:
        if e <= wstart or s >= wend:
            continue
        ov = min(e, wend) - max(s, wstart)
        if ov > 0:
            cnt += 1
            total += ov
    return cnt, round(total, 3)


# ---------------------------------------------------------------------------
# v1 hypothesis-side features
# ---------------------------------------------------------------------------
def has_aba_short_turn(items: list[dict], max_mid_sec: float = 2.0) -> bool:
    for a, b, c in zip(items, items[1:], items[2:]):
        if str(a.get("speaker")) != str(c.get("speaker")):
            continue
        if str(a.get("speaker")) == str(b.get("speaker")):
            continue
        if float(b["end"]) - float(b["start"]) <= max_mid_sec:
            return True
    return False


def compute_hypothesis_features(
    wi: int,
    items: list[dict],
    *,
    window_sec: float,
    expected_end: float,
    tight_gap_sec: float = 0.25,
) -> dict:
    speakers = [str(s.get("speaker")) for s in items]
    texts = [str(s.get("text", "")) for s in items]
    q_scores = [score_patterns(QUESTION_PATTERNS, t) for t in texts]
    a_scores = [score_patterns(ANSWER_PATTERNS, t) for t in texts]
    switches = sum(1 for x, y in zip(speakers, speakers[1:]) if x != y)
    very_short = tight = mixed_hint = abnormal_rate = long_cross = qa_mixed = 0
    for pos, seg in enumerate(items):
        start = float(seg["start"]); end = float(seg["end"])
        dur = max(1e-6, end - start)
        chars = len(str(seg.get("text", "")))
        if dur <= 1.0:
            very_short += 1
        wstart = wi * window_sec
        wend = min(wstart + window_sec, expected_end)
        if start < wend - 5.0 and end > wend and dur >= 12.0:
            long_cross += 1
        cps = chars / dur
        if cps < 1.0 or cps > 8.0:
            abnormal_rate += 1
        if q_scores[pos] and a_scores[pos]:
            qa_mixed += 1
        dbg = seg.get("debug") or {}
        hints = dbg.get("speaker_confidence_hints") or []
        if seg.get("mixed_speaker") or hints:
            mixed_hint += 1
        if pos:
            prev = items[pos - 1]
            if str(prev.get("speaker")) != str(seg.get("speaker")):
                gap = start - float(prev.get("end", start))
                if abs(gap) <= tight_gap_sec:
                    tight += 1
    suspicious_single_label_qa = int(
        len(set(speakers)) == 1 and sum(q_scores) > 0 and sum(a_scores) > 0
    )
    return {
        "segments": len(items),
        "chars": sum(len(t) for t in texts),
        "speakers": dict(Counter(speakers)),
        "speaker_switches": switches,
        "question_cues": sum(q_scores),
        "answer_cues": sum(a_scores),
        "qa_mixed_segments": qa_mixed,
        "suspicious_single_label_qa": suspicious_single_label_qa,
        "aba_short_turn": has_aba_short_turn(items),
        "very_short_turns": very_short,
        "tight_boundaries": tight,
        "mixed_debug_hints": mixed_hint,
        "long_cross_boundary_segments": long_cross,
        "abnormal_char_rate_segments": abnormal_rate,
        "text_preview": "".join(texts)[:120],
    }


# ---------------------------------------------------------------------------
# audio-side features per window
# ---------------------------------------------------------------------------
def compute_audio_features(
    wi: int,
    items: list[dict],
    *,
    window_sec: float,
    expected_end: float,
    audio: np.ndarray,
    diar_pipeline_2: sherpa_onnx.OfflineSpeakerDiarization,
    diar_pipeline_auto: Optional[sherpa_onnx.OfflineSpeakerDiarization],
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
    centroids: dict[str, np.ndarray],
    dominant_label: str,
    minority_label: str,
    silences: list[tuple[float, float]],
) -> dict:
    wstart = wi * window_sec
    wend = min(wstart + window_sec, expected_end)
    s = max(0, int(wstart * TARGET_SR))
    e = min(len(audio), int(wend * TARGET_SR))
    window_audio = audio[s:e]
    actual_dur = max(1e-6, (e - s) / TARGET_SR)

    # ---- mini local diarize (num_speakers=2) ----
    turns2 = diar_pipeline_2.process(window_audio).sort_by_start_time()
    turns2 = [{"start": float(t.start), "end": float(t.end), "speaker": int(t.speaker)} for t in turns2]
    distinct2 = len({t["speaker"] for t in turns2})
    speech2 = sum(max(0.0, t["end"] - t["start"]) for t in turns2)
    durs2 = [max(0.0, t["end"] - t["start"]) for t in turns2]
    switches2 = sum(1 for a, b in zip(turns2, turns2[1:]) if a["speaker"] != b["speaker"])
    spk_dur2: dict[int, float] = defaultdict(float)
    for t in turns2:
        spk_dur2[t["speaker"]] += max(0.0, t["end"] - t["start"])
    minority_speech_ratio = 0.0
    if len(spk_dur2) >= 2 and speech2 > 0:
        minority_speech_ratio = min(spk_dur2.values()) / speech2

    # ---- mini local diarize (num_speakers=auto) ----
    turns_auto_count = -1
    distinct_auto = 0
    if diar_pipeline_auto is not None:
        turns_a = diar_pipeline_auto.process(window_audio).sort_by_start_time()
        turns_a = [{"start": float(t.start), "end": float(t.end), "speaker": int(t.speaker)} for t in turns_a]
        turns_auto_count = len(turns_a)
        distinct_auto = len({t["speaker"] for t in turns_a})

    # ---- per-segment embedding vs centroid ----
    label_disagreements = 0
    confident_minority_pieces = 0  # hypothesis says dominant but embedding clearly minority
    confident_dominant_pieces = 0  # hypothesis says minority but embedding clearly dominant
    dominant_self_cos_min: Optional[float] = None
    minority_self_cos_min: Optional[float] = None
    seg_embs: list[tuple[float, np.ndarray, str]] = []  # (mid_time, emb, label)
    for seg in items:
        start = float(seg["start"]); end = float(seg["end"])
        emb = embedding_for_interval(extractor, audio, start, end)
        if emb is None:
            continue
        label = str(seg.get("speaker"))
        seg_embs.append(((start + end) / 2, emb, label))
        dom_c = centroids.get(dominant_label)
        min_c = centroids.get(minority_label)
        if dom_c is None or min_c is None:
            continue
        dist_to_dom = cosine(emb, dom_c)
        dist_to_min = cosine(emb, min_c)
        # margin > 0: closer to dominant; < 0: closer to minority
        margin = dist_to_dom - dist_to_min
        # closeness to the speaker the hypothesis assigned
        if label == dominant_label:
            self_cos = dist_to_dom
            dominant_self_cos_min = min(dominant_self_cos_min, self_cos) if dominant_self_cos_min is not None else self_cos
            if margin < -0.05:
                label_disagreements += 1
                if margin < -0.10:
                    confident_minority_pieces += 1
        elif label == minority_label:
            self_cos = dist_to_min
            minority_self_cos_min = min(minority_self_cos_min, self_cos) if minority_self_cos_min is not None else self_cos
            if margin > 0.05:
                label_disagreements += 1
                if margin > 0.10:
                    confident_dominant_pieces += 1
        # else: ignore unmapped labels

    # ---- consecutive embedding cosine distance (intra-window only) ----
    consec_max_dist = 0.0
    consec_pairs = 0
    for (_t1, e1, l1), (_t2, e2, l2) in zip(seg_embs, seg_embs[1:]):
        d = 1.0 - cosine(e1, e2)
        consec_max_dist = max(consec_max_dist, d)
        if l1 == l2 and d > 0.4:
            consec_pairs += 1  # same label but acoustically far -> suspicious

    sil_cnt, sil_total = silence_overlap(silences, wstart, wend)

    # rms / energy variability (cheap)
    if len(window_audio) > 0:
        block = TARGET_SR  # 1s blocks
        n_blocks = max(1, len(window_audio) // block)
        rms_per_block = []
        for bi in range(n_blocks):
            chunk = window_audio[bi * block:(bi + 1) * block]
            if len(chunk) == 0:
                continue
            rms_per_block.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
        energy_std = float(np.std(rms_per_block)) if rms_per_block else 0.0
        energy_mean = float(np.mean(rms_per_block)) if rms_per_block else 0.0
    else:
        energy_std = energy_mean = 0.0

    return {
        "audio_duration": round(actual_dur, 3),
        "local_turns_n2": len(turns2),
        "local_distinct_n2": distinct2,
        "local_switches_n2": switches2,
        "local_speech_ratio_n2": round(speech2 / actual_dur, 3),
        "local_min_turn_dur_n2": round(min(durs2), 3) if durs2 else 0.0,
        "local_max_turn_dur_n2": round(max(durs2), 3) if durs2 else 0.0,
        "local_minority_speech_ratio_n2": round(minority_speech_ratio, 3),
        "local_turns_auto": turns_auto_count,
        "local_distinct_auto": distinct_auto,
        "label_disagreements": label_disagreements,
        "confident_minority_pieces": confident_minority_pieces,
        "confident_dominant_pieces": confident_dominant_pieces,
        "dominant_self_cos_min": round(dominant_self_cos_min, 4) if dominant_self_cos_min is not None else None,
        "minority_self_cos_min": round(minority_self_cos_min, 4) if minority_self_cos_min is not None else None,
        "consec_max_dist": round(consec_max_dist, 4),
        "consec_same_label_far_pairs": consec_pairs,
        "silence_count": sil_cnt,
        "silence_total_sec": sil_total,
        "energy_std": round(energy_std, 5),
        "energy_mean": round(energy_mean, 5),
    }


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def score_v2(feat: dict, dominant_label: str, minority_label: str) -> tuple[float, list[str]]:
    """Score features. Returns (risk_score, reasons).

    Weights are tuned on the 149-min audio against 59 manually labelled
    high-risk windows (top40 precision 70%, recall 47.5%; top80 precision 46%,
    recall 63%).  Highest-signal audio features (embedding centroid disagreement,
    same-label adjacent-far embeddings, local switch surplus) dominate the
    score; the noisy single-hyp/audio-two-speakers signal is gated by silence
    and minority-speech ratio thresholds.
    """
    score = 0.0
    reasons: list[str] = []

    def add(points: float, reason: str) -> None:
        nonlocal score
        score += points
        reasons.append(f"{reason}+{round(points, 2):g}")

    hyp_distinct = len(feat["speakers"])

    # ---- top-tier audio signals (embedding centroid disagreement) ----
    if feat["confident_minority_pieces"] >= 1:
        add(4.0 * min(3, feat["confident_minority_pieces"]), "confident_minority_audio_in_dominant_label")
    if feat["confident_dominant_pieces"] >= 1:
        add(4.0 * min(3, feat["confident_dominant_pieces"]), "confident_dominant_audio_in_minority_label")
    if feat["label_disagreements"] >= 1 and feat["confident_minority_pieces"] + feat["confident_dominant_pieces"] == 0:
        add(1.0 * min(3, feat["label_disagreements"]), "soft_label_audio_disagreement")

    # ---- audio: intra-segment speaker switch (same label, far embeddings) ----
    if feat["consec_same_label_far_pairs"] >= 1:
        add(3.0 * min(3, feat["consec_same_label_far_pairs"]), "same_label_adjacent_far_embeddings")

    # ---- audio: local diarize switch surplus vs hypothesis ----
    if feat["local_switches_n2"] - feat["speaker_switches"] >= 3:
        add(3.0, "local_switches_far_exceed_hyp")
    elif feat["local_switches_n2"] - feat["speaker_switches"] >= 2:
        add(1.5, "local_switches_exceed_hyp")

    # ---- audio: dominant label self-cosine drop (hypothesis says dominant but acoustic doesn't fit) ----
    dsc = feat.get("dominant_self_cos_min")
    if dsc is not None and dsc < 0.40:
        add(2.0, "dominant_self_cos_below_0.40")
    elif dsc is not None and dsc < 0.55:
        add(1.0, "dominant_self_cos_below_0.55")

    # ---- audio: single-hyp but two-speaker local diarize (strict gate) ----
    # Strict gate -> requires high minority ratio AND low silence to avoid
    # opening-monologue / fillback false positives.
    if (
        hyp_distinct <= 1
        and feat["local_distinct_n2"] >= 2
        and feat["local_minority_speech_ratio_n2"] >= 0.25
        and feat["silence_total_sec"] < 6.0
    ):
        add(3.0, "single_hyp_but_audio_two_speakers_strict")
    elif (
        hyp_distinct <= 1
        and feat["local_distinct_n2"] >= 2
        and feat["local_minority_speech_ratio_n2"] >= 0.15
    ):
        add(1.0, "single_hyp_but_audio_two_speakers")

    # ---- hypothesis priors (lower weights — they help but are noisy on their own) ----
    if feat["suspicious_single_label_qa"]:
        add(1.5, "single_speaker_with_question_and_answer_cues")
    if feat["qa_mixed_segments"]:
        add(1.5 * min(2, feat["qa_mixed_segments"]), "same_segment_question_answer_cues")
    if feat["mixed_debug_hints"]:
        add(1.0 * min(3, feat["mixed_debug_hints"]), "mixed_or_confidence_debug")
    if feat["tight_boundaries"]:
        add(0.75 * min(4, feat["tight_boundaries"]), "tight_speaker_boundaries")
    if feat["very_short_turns"]:
        add(0.5 * min(4, feat["very_short_turns"]), "very_short_turns")
    if feat["aba_short_turn"]:
        add(1.5, "aba_short_turn")
    if feat["long_cross_boundary_segments"]:
        add(1.0 * feat["long_cross_boundary_segments"], "long_segment_crosses_window_boundary")
    if feat["abnormal_char_rate_segments"]:
        add(0.5 * min(3, feat["abnormal_char_rate_segments"]), "abnormal_chars_per_second")
    if feat["question_cues"] and feat["answer_cues"] and len(feat["speakers"]) <= 2:
        add(0.3, "question_answer_cues_in_window")

    # ---- audio: tiny boost when silence absent + both speakers present ----
    if feat["silence_total_sec"] < 0.5 and feat["local_distinct_n2"] >= 2 and hyp_distinct >= 2:
        add(0.5, "very_low_silence_with_two_speakers_both_in_hyp")

    return round(score, 3), reasons


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Audio-aware no-reference high-risk window detector (v2)")
    ap.add_argument("input_json", type=Path)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--window-sec", type=float, default=40.0)
    ap.add_argument("--start-window", type=int, default=0)
    ap.add_argument("--end-window", type=int, default=10_000)
    ap.add_argument("--skip-auto-diarize", action="store_true",
                    help="Skip the num_speakers=auto sherpa pass to halve diarize cost.")
    ap.add_argument("--silence-noise-db", type=float, default=-30.0)
    ap.add_argument("--silence-min-dur", type=float, default=0.4)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    segments = payload["segments"]
    expected_end = float(
        payload.get("summary", {}).get("audio_duration")
        or max(float(s["end"]) for s in segments)
    )

    # global roles (most chars = dominant, least = minority)
    chars_by_sp: dict[str, int] = defaultdict(int)
    for s in segments:
        chars_by_sp[str(s["speaker"])] += len(str(s.get("text", "")))
    ordered_sp = sorted(chars_by_sp.items(), key=lambda kv: kv[1], reverse=True)
    dominant_label = ordered_sp[0][0]
    minority_label = ordered_sp[-1][0]
    print(f"[role] dominant={dominant_label} minority={minority_label} all={dict(chars_by_sp)}")

    # ---- group segments per window ----
    groups: dict[int, list[dict]] = defaultdict(list)
    for seg in segments:
        wi = int(math.floor(max(0.0, float(seg["start"])) / args.window_sec))
        groups[wi].append(seg)

    all_wis = sorted(
        wi for wi in groups
        if args.start_window <= wi <= args.end_window
    )
    print(f"[plan] {len(all_wis)} windows to score (start_window={args.start_window} end_window={args.end_window})")

    # ---- one-time setup ----
    t0 = time.time()
    print("[load] audio -> memory @ 16kHz mono")
    audio = load_audio_mono_16k(args.audio)
    print(f"[load] audio loaded in {time.time()-t0:.1f}s shape={audio.shape}")

    t0 = time.time()
    print("[silence] running ffmpeg silencedetect (single pass)")
    silences = detect_silences(args.audio, args.silence_noise_db, args.silence_min_dur)
    print(f"[silence] {len(silences)} silence intervals in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("[sherpa] build diarize pipeline (num_speakers=2)")
    diar2 = build_diarization_pipeline(num_speakers=2)
    diar_auto = None if args.skip_auto_diarize else build_diarization_pipeline(num_speakers=None)
    extractor = build_embedding_extractor()
    print(f"[sherpa] pipelines ready in {time.time()-t0:.1f}s skip_auto={args.skip_auto_diarize}")

    t0 = time.time()
    print("[centroids] building global hypothesis centroids")
    centroids = build_centroids(extractor, audio, segments)
    print(f"[centroids] built {len(centroids)} centroids in {time.time()-t0:.1f}s labels={list(centroids.keys())}")

    # ---- per-window features ----
    rows: list[dict] = []
    t_start_all = time.time()
    for n_done, wi in enumerate(all_wis, 1):
        items = groups[wi]
        t_w = time.time()
        hyp = compute_hypothesis_features(
            wi, items, window_sec=args.window_sec, expected_end=expected_end,
        )
        aud = compute_audio_features(
            wi, items,
            window_sec=args.window_sec,
            expected_end=expected_end,
            audio=audio,
            diar_pipeline_2=diar2,
            diar_pipeline_auto=diar_auto,
            extractor=extractor,
            centroids=centroids,
            dominant_label=dominant_label,
            minority_label=minority_label,
            silences=silences,
        )
        feat = {
            "window": wi,
            "start": round(wi * args.window_sec, 2),
            "end": round(min((wi + 1) * args.window_sec, expected_end), 2),
            **hyp,
            **aud,
        }
        feat["risk_score"], feat["reasons"] = score_v2(feat, dominant_label, minority_label)
        rows.append(feat)
        if n_done % 20 == 0 or n_done == len(all_wis):
            elapsed = time.time() - t_start_all
            print(
                f"[score] {n_done}/{len(all_wis)} elapsed={elapsed:.1f}s "
                f"avg={elapsed/n_done:.2f}s/win"
            )
    print(f"[done] scored {len(rows)} windows in {time.time()-t_start_all:.1f}s")

    ranked = sorted(rows, key=lambda r: (-r["risk_score"], r["window"]))

    def top_ids(k: int) -> list[int]:
        return sorted(r["window"] for r in ranked[:k] if r["risk_score"] > 0)

    report = {
        "source_json": str(args.input_json),
        "audio": str(args.audio),
        "strategy": "audio_aware_v2",
        "params": {
            "window_sec": args.window_sec,
            "skip_auto_diarize": args.skip_auto_diarize,
            "silence_noise_db": args.silence_noise_db,
            "silence_min_dur": args.silence_min_dur,
        },
        "dominant_label": dominant_label,
        "minority_label": minority_label,
        "centroid_labels": list(centroids.keys()),
        "selected_top40_ids": top_ids(40),
        "selected_top60_ids": top_ids(60),
        "selected_top80_ids": top_ids(80),
        "all_windows_ranked": ranked,
    }
    (args.out_dir / "high_risk_windows_v2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for k in (20, 40, 60, 80):
        (args.out_dir / f"top{k}_ids.txt").write_text(",".join(map(str, top_ids(k))), encoding="utf-8")

    print(f"[out] {args.out_dir / 'high_risk_windows_v2.json'}")
    print("[out] top40_ids=" + ",".join(map(str, top_ids(40))))


if __name__ == "__main__":
    main()
