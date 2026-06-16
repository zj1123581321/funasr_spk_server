"""
PR4 follow-up — FileBasedProcessPool.__del__ 解释器关闭路径的 cleanup safety.

背景: 长 audio + N=2 真并发 e2e 跑完 server 关闭时, __del__ 被解释器在
关停路径触发, 此时 main thread event loop 已不存在, asyncio.get_event_loop()
抛 RuntimeError 污染 stderr (无害, 但 noisy).

修复目标: __del__ 整体 try/except 兜底, 解释器关闭路径静默, 不影响其他
__del__ 跑, 不污染 stderr.
"""
from __future__ import annotations

import asyncio


class TestPoolDelDoesNotRaiseOnClosedEventLoop:
    def test_del_swallows_runtime_error_when_event_loop_unavailable(
        self, monkeypatch, tmp_path
    ) -> None:
        """模拟解释器关闭路径: get_event_loop 抛 RuntimeError → __del__ 不应传播."""
        from src.core.file_based_process_pool import FileBasedProcessPool

        # 构造一个 minimal pool (不真启 worker, 仅设 is_initialized=True 触发 cleanup 路径)
        pool = FileBasedProcessPool(
            pool_size=1,
            worker_entry_script="src/core/qwen3_worker_process.py",
            task_dir=str(tmp_path / "tasks"),
        )
        pool.is_initialized = True

        def _raise_no_loop(*a, **kw):
            raise RuntimeError("There is no current event loop in thread 'MainThread'.")

        monkeypatch.setattr(asyncio, "get_event_loop", _raise_no_loop)

        # __del__ 不应抛 (即使 event loop 已不存在)
        try:
            pool.__del__()
        except Exception as exc:
            raise AssertionError(
                f"__del__ 在解释器关闭路径不应抛任何异常, 实际: {type(exc).__name__}: {exc}"
            )
        finally:
            # 防止 pytest fixture cleanup 时 GC 又触发 __del__
            pool.is_initialized = False
