"""#3 异常分类纪律(只改上层接收端): ErrorKind 枚举 + classify_error 单一分类器

替代散落的 _should_retry_error(中文子串匹配)+ _is_model_error(正则)。isinstance 优先,
字符串兜底(底层引擎仍抛裸 Exception+中文 message, 范围一不改底层 — 发送端类型化异常 = TODOS #3 完整版)。

codex #11: retryable 每 kind 显式(集合判定, 非 "not NON_RETRYABLE_INPUT" 取反 —
否则 QUEUE_FULL / TIMEOUT 会误判可重试)。
"""
from __future__ import annotations

import pytest

from src.core.task_manager import ErrorKind, QueueFullError, classify_error


class TestErrorKindValues:
    def test_values_align_with_record_error_kinds(self):
        assert ErrorKind.QUEUE_FULL.value == "queue_full"
        assert ErrorKind.TIMEOUT.value == "timeout"
        assert ErrorKind.NON_RETRYABLE_INPUT.value == "non_retryable_input"
        assert ErrorKind.MODEL_ERROR.value == "model_error"
        assert ErrorKind.ENGINE_ERROR.value == "engine_error"


class TestRetryable:
    def test_engine_and_model_are_retryable(self):
        assert ErrorKind.ENGINE_ERROR.retryable is True
        assert ErrorKind.MODEL_ERROR.retryable is True

    def test_non_retryable_input_not_retryable(self):
        assert ErrorKind.NON_RETRYABLE_INPUT.retryable is False

    def test_queue_full_and_timeout_not_retryable(self):
        # codex #11: 不能用取反写法把这俩判成可重试
        assert ErrorKind.QUEUE_FULL.retryable is False
        assert ErrorKind.TIMEOUT.retryable is False

    def test_is_model_only_for_model_error(self):
        assert ErrorKind.MODEL_ERROR.is_model is True
        assert ErrorKind.ENGINE_ERROR.is_model is False
        assert ErrorKind.NON_RETRYABLE_INPUT.is_model is False


class TestClassifyError:
    def test_queue_full_by_isinstance(self):
        exc = QueueFullError(retry_after=1, queue_size=20, max_queue_size=20)
        assert classify_error(exc) is ErrorKind.QUEUE_FULL

    @pytest.mark.parametrize("msg", [
        "音频时长过短", "文件不存在", "不支持的文件格式", "文件太大", "认证失败",
    ])
    def test_non_retryable_input_by_marker(self, msg):
        assert classify_error(Exception(f"转录失败: {msg}")) is ErrorKind.NON_RETRYABLE_INPUT

    @pytest.mark.parametrize("msg", [
        "VAD algorithm error", "list index out of range", "index 5 out of bounds",
        "window size mismatch", "dimension error",
    ])
    def test_model_error_by_pattern(self, msg):
        assert classify_error(RuntimeError(msg)) is ErrorKind.MODEL_ERROR

    def test_unknown_falls_back_to_engine_error(self):
        assert classify_error(Exception("something totally unexpected")) is ErrorKind.ENGINE_ERROR
