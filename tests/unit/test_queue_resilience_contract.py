"""
高负载队列机制止血 — 共享契约测试

覆盖：
1. TranscriptionConfig 新增的队列健壮性字段默认值
2. 这些字段的 env override
3. QueueFullError 异常类型（携带 retry_after / queue_size / position）
4. TaskStatus.TIMED_OUT 枚举（看门狗终态化用）

这些是 Lane A / Lane B 共用的契约，先于实现落地（红→绿）。
"""
import os
from unittest.mock import patch

import pytest


class TestTranscriptionConfigQueueFields:
    """新增字段的默认值（单一 source of truth = config.py）"""

    def test_task_retention_ttl_default(self):
        from src.core.config import TranscriptionConfig
        c = TranscriptionConfig()
        # 终态任务保留 TTL，须 ≥ 客户端轮询窗口（默认 1h）
        assert c.task_retention_ttl_seconds == 3600

    def test_task_max_retained_default(self):
        from src.core.config import TranscriptionConfig
        c = TranscriptionConfig()
        # self.tasks 硬数量上限（size-cap 兜底）
        assert c.task_max_retained == 500

    def test_task_cleanup_interval_default(self):
        from src.core.config import TranscriptionConfig
        c = TranscriptionConfig()
        assert c.task_cleanup_interval_seconds == 60

    def test_task_max_processing_default(self):
        from src.core.config import TranscriptionConfig
        c = TranscriptionConfig()
        # processing 看门狗：超此时长仍 PROCESSING → 强制 TIMED_OUT（默认 1h，明显卡死阈值）
        assert c.task_max_processing_seconds == 3600

    def test_upload_session_ttl_default(self):
        from src.core.config import TranscriptionConfig
        c = TranscriptionConfig()
        assert c.upload_session_ttl_seconds == 1800

    def test_upload_session_max_count_default(self):
        from src.core.config import TranscriptionConfig
        c = TranscriptionConfig()
        assert c.upload_session_max_count == 200


class TestQueueFieldsEnvOverride:
    """env 覆盖（走标准 _override_if_set 路径，print_config 可见）"""

    def _reload_config_with_env(self, env: dict):
        """在给定 env 下重新构造 Config，返回 transcription 段"""
        from src.core.config import Config
        with patch.dict(os.environ, env, clear=False):
            cfg = Config.load_from_file()
        return cfg.transcription

    def test_max_queue_size_env_override(self):
        t = self._reload_config_with_env({"FUNASR_MAX_QUEUE_SIZE": "7"})
        assert t.max_queue_size == 7

    def test_task_retention_ttl_env_override(self):
        t = self._reload_config_with_env({"FUNASR_TASK_RETENTION_TTL_SECONDS": "120"})
        assert t.task_retention_ttl_seconds == 120

    def test_task_max_retained_env_override(self):
        t = self._reload_config_with_env({"FUNASR_TASK_MAX_RETAINED": "42"})
        assert t.task_max_retained == 42

    def test_task_max_processing_env_override(self):
        t = self._reload_config_with_env({"FUNASR_TASK_MAX_PROCESSING_SECONDS": "300"})
        assert t.task_max_processing_seconds == 300


class TestQueueFullError:
    """队列满的结构化异常（替代泛化 Exception）"""

    def test_is_exception_subclass(self):
        from src.core.task_manager import QueueFullError
        assert issubclass(QueueFullError, Exception)

    def test_carries_retry_after_and_capacity(self):
        from src.core.task_manager import QueueFullError
        err = QueueFullError(retry_after=12, queue_size=20, max_queue_size=20)
        assert err.retry_after == 12
        assert err.queue_size == 20
        assert err.max_queue_size == 20

    def test_message_human_readable(self):
        from src.core.task_manager import QueueFullError
        err = QueueFullError(retry_after=5, queue_size=20, max_queue_size=20)
        assert "20" in str(err)


class TestTimedOutStatus:
    """看门狗终态化需要的状态"""

    def test_timed_out_enum_exists(self):
        from src.models.schemas import TaskStatus
        assert TaskStatus.TIMED_OUT.value == "timed_out"
