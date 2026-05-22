"""单测: relabel_segments_by_duration_desc — 按 speaker 总时长降序稳定 Speaker ID.

业务问题: cuda 路径 raw cluster id 是 Speaker55, Mac 路径同 audio 是 Speaker41,
客户端看到的 Speaker ID 跨 backend / 跨平台不稳定. 这个 helper 在最末层把内部
int label 按 "总时长降序" 重新映射, 让主说话人始终是 0 (输出 "Speaker1"),
次主是 1 ("Speaker2"), 以此类推.
"""
from __future__ import annotations

from src.core.qwen3.merge import Segment, relabel_segments_by_duration_desc


def test_relabel_multi_speaker_by_duration_desc():
    """3 个 speaker, 时长 [int=5: 100s, int=2: 50s, int=8: 10s].

    期望重映射: int 5 (主) → 0, int 2 → 1, int 8 → 2.
    输出层 +1 后看到 Speaker1 / Speaker2 / Speaker3.
    """
    segs = [
        Segment(start=0.0, end=60.0, speaker=5, text="主说话人 part 1"),
        Segment(start=60.0, end=110.0, speaker=2, text="次主 part 1"),
        Segment(start=110.0, end=120.0, speaker=8, text="第三人 part 1"),
        Segment(start=120.0, end=160.0, speaker=5, text="主说话人 part 2"),
    ]
    out = relabel_segments_by_duration_desc(segs)

    # int 5 总 100s (60+40), int 2 总 50s, int 8 总 10s
    # 重映射: 5→0, 2→1, 8→2
    assert [s.speaker for s in out] == [0, 1, 2, 0]
    # 内容不变 (start/end/text)
    assert [s.text for s in out] == [s.text for s in segs]
    assert [(s.start, s.end) for s in out] == [(s.start, s.end) for s in segs]


def test_relabel_single_speaker_to_zero():
    """单 speaker 场景: 不管原 int 是多少, 都映射到 0.

    这保证 cuda raw cluster id=42 跟 Mac raw cluster id=0 在客户端
    都输出成 "Speaker1" (0+1).
    """
    segs = [
        Segment(start=0.0, end=30.0, speaker=42, text="独白 part 1"),
        Segment(start=30.0, end=60.0, speaker=42, text="独白 part 2"),
    ]
    out = relabel_segments_by_duration_desc(segs)
    assert [s.speaker for s in out] == [0, 0]


def test_relabel_empty_segments_returns_empty():
    """空 segments 直接返回空 list, 不挂."""
    assert relabel_segments_by_duration_desc([]) == []


def test_relabel_tie_breaking_deterministic_by_original_int():
    """两个 speaker 总时长相同时, 按原 int 升序作为 tie breaker.

    避免 dict 遍历顺序导致跨运行不一致.
    """
    # int=9: 50s, int=3: 50s, int=7: 50s  (三人都 50s 平局)
    segs = [
        Segment(start=0.0, end=50.0, speaker=9, text="A"),
        Segment(start=50.0, end=100.0, speaker=3, text="B"),
        Segment(start=100.0, end=150.0, speaker=7, text="C"),
    ]
    out = relabel_segments_by_duration_desc(segs)
    # 平局按原 int 升序: 3 → 0, 7 → 1, 9 → 2
    # 原 [9, 3, 7] → 新 [2, 0, 1]
    assert [s.speaker for s in out] == [2, 0, 1]

    # 再跑一遍应得相同结果 (idempotent given fresh input)
    out2 = relabel_segments_by_duration_desc(segs)
    assert [s.speaker for s in out2] == [2, 0, 1]
