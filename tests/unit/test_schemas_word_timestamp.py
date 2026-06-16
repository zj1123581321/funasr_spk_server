"""
词级时间戳 schema — WordTimestamp model + TranscriptionSegment.words 字段

设计:
- WordTimestamp(text:str, start:float, end:float, confidence:Optional[float]=None)
  绝对秒, confidence 可空.
- TranscriptionSegment 新增 words: Optional[List[WordTimestamp]] = None
  默认 None, 向后兼容 (老 segment 不带 words 仍能构造 + 序列化).
"""
from __future__ import annotations

from src.models.schemas import TranscriptionSegment, WordTimestamp


class TestWordTimestamp:
    def test_construct_with_confidence(self) -> None:
        w = WordTimestamp(text="你好", start=1.0, end=1.5, confidence=0.92)
        assert w.text == "你好"
        assert w.start == 1.0
        assert w.end == 1.5
        assert w.confidence == 0.92

    def test_confidence_defaults_none(self) -> None:
        w = WordTimestamp(text="hello", start=0.0, end=0.3)
        assert w.confidence is None

    def test_serialize_roundtrip(self) -> None:
        w = WordTimestamp(text="word", start=2.0, end=2.4, confidence=0.8)
        data = w.model_dump()
        assert data == {"text": "word", "start": 2.0, "end": 2.4, "confidence": 0.8}
        assert WordTimestamp(**data) == w


class TestTranscriptionSegmentWords:
    def test_words_defaults_none_backward_compat(self) -> None:
        # 老调用方不传 words 仍能构造
        seg = TranscriptionSegment(
            start_time=0.0, end_time=5.0, text="一段话", speaker="Speaker1"
        )
        assert seg.words is None

    def test_words_omitted_in_dump_when_none(self) -> None:
        seg = TranscriptionSegment(
            start_time=0.0, end_time=5.0, text="一段话", speaker="Speaker1"
        )
        # 序列化含 words=None, 不破坏老结构其它字段
        data = seg.model_dump()
        assert data["words"] is None
        assert data["text"] == "一段话"

    def test_words_attached(self) -> None:
        words = [
            WordTimestamp(text="你", start=0.0, end=0.3),
            WordTimestamp(text="好", start=0.3, end=0.6, confidence=0.9),
        ]
        seg = TranscriptionSegment(
            start_time=0.0, end_time=0.6, text="你好", speaker="Speaker1", words=words
        )
        assert len(seg.words) == 2
        assert seg.words[0].text == "你"
        # 嵌套序列化
        dumped = seg.model_dump()
        assert dumped["words"][1]["text"] == "好"
        assert dumped["words"][1]["confidence"] == 0.9
