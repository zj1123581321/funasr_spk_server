"""
Evaluate Qwen3 PoC JSON against a calibrated speaker transcript.

The reference format is the LexGoGo calibrated text export used in PR4:
paragraphs prefixed with speaker labels like ``吴明辉：`` and ``程曼祺：``.

Example:
    python tests/manual/server/evaluate_qwen3_poc_against_reference.py \
      tmp_long_audio/reference_149min_calibrated.txt \
      tmp_long_audio/poc_outputs_v2/audio_149min.qwen3_long_poc.json
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import permutations, product
from pathlib import Path


SPEAKER_MAP = {
    "吴明辉": "W",
    "程曼祺": "C",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    out = []
    for ch in text:
        category = unicodedata.category(ch)
        if category.startswith(("P", "Z", "S")):
            continue
        if ch in "\ufeff\n\r\t":
            continue
        out.append(ch)
    return "".join(out)


def parse_reference(path: Path) -> tuple[str, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_marks = 0
    paragraphs: list[tuple[str, str]] = []
    for line in lines:
        if line.strip() == "---":
            header_marks += 1
            continue
        if header_marks < 2 or not line.strip():
            continue
        m = re.match(r"^(吴明辉|程曼祺)：(.*)$", line.strip())
        if m:
            paragraphs.append((m.group(1), m.group(2)))
        elif paragraphs:
            speaker, text = paragraphs[-1]
            paragraphs[-1] = (speaker, text + line.strip())

    text_parts = []
    speaker_chars: list[str] = []
    for speaker, text in paragraphs:
        norm = normalize_text(text)
        text_parts.append(norm)
        speaker_chars.extend([SPEAKER_MAP[speaker]] * len(norm))
    return "".join(text_parts), speaker_chars


def parse_hypothesis(path: Path) -> tuple[str, list[str], list[int], list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    text_parts = []
    speaker_chars: list[str] = []
    segment_chars: list[int] = []
    segments = payload["segments"]
    for idx, seg in enumerate(segments):
        norm = normalize_text(seg["text"])
        text_parts.append(norm)
        speaker_chars.extend([seg["speaker"]] * len(norm))
        segment_chars.extend([idx] * len(norm))
    return "".join(text_parts), speaker_chars, segment_chars, segments


def edit_stats(ref_text: str, hyp_text: str, opcodes: list[tuple]) -> dict:
    equal = insert = delete = replace_ref = replace_hyp = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            equal += i2 - i1
        elif tag == "insert":
            insert += j2 - j1
        elif tag == "delete":
            delete += i2 - i1
        elif tag == "replace":
            replace_ref += i2 - i1
            replace_hyp += j2 - j1
    edits = insert + delete + max(replace_ref, replace_hyp)
    return {
        "ref_chars": len(ref_text),
        "hyp_chars": len(hyp_text),
        "equal_chars": equal,
        "insert_hyp_chars": insert,
        "delete_ref_chars": delete,
        "replace_ref_chars": replace_ref,
        "replace_hyp_chars": replace_hyp,
        "approx_edits": edits,
        "approx_cer": edits / len(ref_text) if ref_text else 0.0,
        "approx_accuracy": 1.0 - edits / len(ref_text) if ref_text else 0.0,
        "hyp_ref_len_ratio": len(hyp_text) / len(ref_text) if ref_text else 0.0,
    }


def best_speaker_mapping(confusion: Counter[tuple[str, str]]) -> dict[str, str]:
    hyp_labels = sorted({hyp for hyp, _ref in confusion})
    ref_labels = sorted({ref for _hyp, ref in confusion})
    if not hyp_labels or not ref_labels:
        return {}

    if len(hyp_labels) <= len(ref_labels):
        candidates = permutations(ref_labels, len(hyp_labels))
    else:
        # More predicted speakers than reference speakers.  This should not be
        # the normal two-speaker case, but keeps the evaluator usable for 83min.
        candidates = product(ref_labels, repeat=len(hyp_labels))

    best_map: dict[str, str] = {}
    best_score = -1
    for refs in candidates:
        mapping = dict(zip(hyp_labels, refs))
        score = sum(count for (hyp, ref), count in confusion.items() if mapping.get(hyp) == ref)
        if score > best_score:
            best_score = score
            best_map = mapping
    return best_map


def speaker_stats(
    opcodes: list[tuple],
    ref_speakers: list[str],
    hyp_speakers: list[str],
    hyp_segment_chars: list[int],
    segments: list[dict],
) -> dict:
    confusion: Counter[tuple[str, str]] = Counter()
    segment_ref_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "equal":
            continue
        for off in range(i2 - i1):
            hyp_speaker = hyp_speakers[j1 + off]
            ref_speaker = ref_speakers[i1 + off]
            confusion[(hyp_speaker, ref_speaker)] += 1
            segment_ref_counts[hyp_segment_chars[j1 + off]][ref_speaker] += 1

    total = sum(confusion.values())
    mapping = best_speaker_mapping(confusion)
    correct = sum(count for (hyp, ref), count in confusion.items() if mapping.get(hyp) == ref)

    majority_ok = 0
    considered = 0
    wrong_segments = []
    mixed_segments = []
    for idx, counts in segment_ref_counts.items():
        if not counts:
            continue
        considered += 1
        expected = mapping.get(segments[idx]["speaker"])
        segment_total = sum(counts.values())
        majority_label, majority_count = counts.most_common(1)[0]
        if expected == majority_label:
            majority_ok += 1
        else:
            wrong_segments.append((idx, segment_total, dict(counts), segments[idx]))
        if majority_count / segment_total < 0.9:
            mixed_segments.append((idx, segment_total, dict(counts), segments[idx]))

    return {
        "speaker_mapping": mapping,
        "speaker_confusion": {f"{hyp}->{ref}": count for (hyp, ref), count in confusion.items()},
        "speaker_accuracy_equal_chars": correct / total if total else 0.0,
        "segment_majority_accuracy": majority_ok / considered if considered else 0.0,
        "segment_majority_ok": majority_ok,
        "segment_majority_considered": considered,
        "wrong_segment_count": len(wrong_segments),
        "mixed_segment_count": len(mixed_segments),
        "top_wrong_segments": [
            {
                "idx": idx,
                "matched_chars": matched,
                "ref_counts": counts,
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"][:100],
            }
            for idx, matched, counts, seg in sorted(
                wrong_segments,
                key=lambda item: (-item[1], item[0]),
            )[:12]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 PoC JSON against calibrated transcript")
    parser.add_argument("reference", type=Path)
    parser.add_argument("hypothesis_json", type=Path)
    args = parser.parse_args()

    ref_text, ref_speakers = parse_reference(args.reference)
    hyp_text, hyp_speakers, hyp_segment_chars, segments = parse_hypothesis(args.hypothesis_json)
    matcher = SequenceMatcher(None, ref_text, hyp_text, autojunk=False)
    opcodes = matcher.get_opcodes()

    report = {
        "reference": str(args.reference),
        "hypothesis": str(args.hypothesis_json),
        "sequence_matcher_ratio": matcher.ratio(),
        "text": edit_stats(ref_text, hyp_text, opcodes),
        "speaker": speaker_stats(opcodes, ref_speakers, hyp_speakers, hyp_segment_chars, segments),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
