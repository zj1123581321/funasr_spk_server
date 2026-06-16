"""可观测性 (P1) — task_manager 单调计数器 + 指标快照.

核心修正 A1 (codex #14): tasks_terminal_total / errors_total 必须是终态化/catch
点累加的**单调 Counter**, 不能扫 self.tasks 现算 —— self.tasks 被 TTL 淘汰,
扫描值是"当前驻留 gauge", Prometheus rate() 会在任务被清后静默回退.

回归测试钉死: 终态计数在任务被 _evict_terminal_tasks 清掉后**不回退**.
"""
from datetime import datetime, timedelta

import pytest

from src.core.task_manager import TaskManager
from src.models.schemas import TranscriptionTask, TaskStatus


def make_task(task_id, status=TaskStatus.PENDING, completed_ago=None, started_ago=None):
    t = TranscriptionTask(
        task_id=task_id, file_name="x.wav", file_path="/tmp/x.wav",
        file_size=100, file_hash="h-" + task_id, engine="qwen3",
    )
    t.status = status
    if completed_ago is not None:
        t.completed_at = datetime.now() - timedelta(seconds=completed_ago)
    if started_ago is not None:
        t.started_at = datetime.now() - timedelta(seconds=started_ago)
    return t


@pytest.fixture
def cfg_guard():
    from src.core.config import config
    t = config.transcription
    saved = (t.task_retention_ttl_seconds, t.task_max_retained, t.task_max_processing_seconds)
    yield t
    (t.task_retention_ttl_seconds, t.task_max_retained, t.task_max_processing_seconds) = saved


class TestTerminalCounter:
    def test_record_terminal_increments(self):
        mgr = TaskManager()
        mgr._record_terminal(TaskStatus.COMPLETED)
        mgr._record_terminal(TaskStatus.COMPLETED)
        mgr._record_terminal(TaskStatus.FAILED)
        snap = mgr.get_metrics_snapshot()
        assert snap["terminal_total"]["completed"] == 2
        assert snap["terminal_total"]["failed"] == 1

    def test_terminal_counter_survives_eviction(self, cfg_guard):
        """A1 回归: 任务被 TTL 清掉后, terminal_total 不回退（单调）。"""
        cfg_guard.task_retention_ttl_seconds = 100
        cfg_guard.task_max_retained = 1000
        mgr = TaskManager()
        # 模拟一个完成任务: 入 self.tasks + 累加计数器（真实路径同时做这两件事）
        task = make_task("done", TaskStatus.COMPLETED, completed_ago=500)
        mgr.tasks["done"] = task
        mgr._record_terminal(TaskStatus.COMPLETED)
        assert mgr.get_metrics_snapshot()["terminal_total"]["completed"] == 1

        # TTL 清理把任务挤出 self.tasks
        evicted = mgr._evict_terminal_tasks()
        assert evicted == 1
        assert "done" not in mgr.tasks
        # 关键: 计数器是单调的, 清理后仍为 1（不是扫 self.tasks 得 0）
        assert mgr.get_metrics_snapshot()["terminal_total"]["completed"] == 1

    def test_watchdog_increments_timed_out(self, cfg_guard):
        cfg_guard.task_max_processing_seconds = 100
        mgr = TaskManager()
        mgr.tasks["stuck"] = make_task("stuck", TaskStatus.PROCESSING, started_ago=500)
        mgr._terminalize_stale_processing()
        assert mgr.get_metrics_snapshot()["terminal_total"]["timed_out"] == 1

    @pytest.mark.asyncio
    async def test_cancel_increments_cancelled(self):
        mgr = TaskManager()
        mgr.tasks["c"] = make_task("c", TaskStatus.PENDING)
        await mgr.cancel_task("c")
        assert mgr.get_metrics_snapshot()["terminal_total"]["cancelled"] == 1


class TestErrorCounter:
    def test_record_error_increments(self):
        mgr = TaskManager()
        mgr.record_error("engine_error")
        mgr.record_error("engine_error")
        mgr.record_error("queue_full")
        snap = mgr.get_metrics_snapshot()
        assert snap["errors_total"]["engine_error"] == 2
        assert snap["errors_total"]["queue_full"] == 1


class TestMetricsSnapshot:
    def test_snapshot_has_gauges_and_counters(self):
        mgr = TaskManager()
        mgr.tasks["p"] = make_task("p", TaskStatus.PENDING)
        snap = mgr.get_metrics_snapshot()
        # gauges (瞬时)
        for k in ("queue_size", "max_queue_size", "pending", "processing", "tasks_in_memory"):
            assert k in snap, f"缺 gauge {k}"
        # counters (单调) + EMA + engine
        for k in ("terminal_total", "errors_total", "task_seconds_ema", "engine", "pool_size"):
            assert k in snap, f"缺 {k}"
        assert snap["pending"] == 1
        assert snap["tasks_in_memory"] == 1
        assert snap["task_seconds_ema"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
