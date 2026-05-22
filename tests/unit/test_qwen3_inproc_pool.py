"""Qwen3InProcPool — 单进程内 N 个 transcriber 实例 + asyncio.gather 并发池.

设计动机 (docs/开发/gpu加速/2026-05-23-CUDA并发突破.md):
- multi-process pool 在 CUDA + ort_cuda backend 上撞 CUDNN cross-process race
- 单进程内多 llama_context + 多 ORT InferenceSession 共享同一 cuda context, race-free
- 接口与 Qwen3PoolTranscriber 鸭子兼容 (async transcribe, initialize, dispatch)

测试覆盖:
1. 构造: pool_size + transcriber_factory 注入 (DI 便于测试)
2. initialize: 构造 N 个 tx, 调每个 tx.initialize()
3. transcribe: 单调用走第一个可用的 tx
4. 并发 = pool_size: N 个 task 分发到 N 个 tx, 各被调一次
5. 并发 > pool_size: 起 3 个 task in pool_size=2, 第 3 个等到前一个完成才 acquire
6. tx 用完回 pool 可复用
7. tx.transcribe 抛错时 tx 仍回 pool (不漏 slot)
8. 未 initialize 时 transcribe 自动 lazy init
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.schemas import TranscriptionResult, TranscriptionSegment


def _fake_json_result(task_id: str = "t"):
    tres = TranscriptionResult(
        task_id=task_id,
        file_name="a.wav",
        file_hash="h",
        duration=1.0,
        segments=[
            TranscriptionSegment(start_time=0.0, end_time=1.0, text="hi", speaker="Speaker1")
        ],
        speakers=["Speaker1"],
        processing_time=0.1,
    )
    return (tres, {"asr_text": "hi", "engine": "qwen3"})


def _make_tx_mock(tx_id: int = 0, sleep_sec: float = 0.0) -> MagicMock:
    """Mock Qwen3DiarizeTranscriber: initialize 是 noop, transcribe 返回 fake json.

    若 sleep_sec > 0, transcribe 内 await asyncio.sleep 让并发可观察.
    在 mock 上挂 tx_id 字段便于 assert 分发到了哪个 tx.
    """
    tx = MagicMock()
    tx.tx_id = tx_id
    tx.initialize = AsyncMock(return_value=None)

    async def _transcribe(audio_path, task_id, progress_callback=None, output_format="json"):
        if sleep_sec > 0:
            await asyncio.sleep(sleep_sec)
        return _fake_json_result(task_id)

    tx.transcribe = AsyncMock(side_effect=_transcribe)
    return tx


# ==================== 构造 ====================


class TestConstructor:
    def test_pool_size_stored(self):
        from src.core.qwen3_inproc_pool import Qwen3InProcPool

        pool = Qwen3InProcPool(pool_size=3, transcriber_factory=lambda: _make_tx_mock())
        assert pool.pool_size == 3

    def test_transcriber_factory_injected(self):
        """DI: 测试用 mock factory, 不实际加载模型"""
        from src.core.qwen3_inproc_pool import Qwen3InProcPool

        factory_calls = []

        def factory():
            factory_calls.append(1)
            return _make_tx_mock()

        pool = Qwen3InProcPool(pool_size=2, transcriber_factory=factory)
        # 构造时 factory 还不调用 — lazy 在 initialize 内
        assert len(factory_calls) == 0


# ==================== Initialize ====================


class TestInitialize:
    async def test_initialize_builds_pool_size_transcribers(self):
        from src.core.qwen3_inproc_pool import Qwen3InProcPool

        built = []

        def factory():
            tx = _make_tx_mock(tx_id=len(built))
            built.append(tx)
            return tx

        pool = Qwen3InProcPool(pool_size=3, transcriber_factory=factory)
        await pool.initialize()

        assert len(built) == 3
        # 每个 tx 都被 initialize 一次
        for tx in built:
            tx.initialize.assert_awaited_once()

    async def test_initialize_is_idempotent(self):
        """重复 initialize 不应重复构造 tx (省去重复 model load)"""
        from src.core.qwen3_inproc_pool import Qwen3InProcPool

        built = []

        def factory():
            tx = _make_tx_mock(tx_id=len(built))
            built.append(tx)
            return tx

        pool = Qwen3InProcPool(pool_size=2, transcriber_factory=factory)
        await pool.initialize()
        await pool.initialize()  # 第二次调用
        assert len(built) == 2  # 仍然只 2 个, 不是 4


# ==================== Transcribe 基础 ====================


class TestTranscribeBasics:
    async def test_single_transcribe_routes_to_first_tx(self):
        from src.core.qwen3_inproc_pool import Qwen3InProcPool

        tx = _make_tx_mock(tx_id=0)
        pool = Qwen3InProcPool(pool_size=1, transcriber_factory=lambda: tx)
        await pool.initialize()

        result, raw = await pool.transcribe("a.wav", "t1")
        assert result.task_id == "t1"
        tx.transcribe.assert_awaited_once()

    async def test_lazy_initialize_on_first_transcribe(self):
        """未先 initialize 直接 transcribe, pool 自动 init"""
        from src.core.qwen3_inproc_pool import Qwen3InProcPool

        tx = _make_tx_mock(tx_id=0)
        pool = Qwen3InProcPool(pool_size=1, transcriber_factory=lambda: tx)
        # 没调 initialize, 直接 transcribe
        result, _ = await pool.transcribe("a.wav", "t1")
        assert result.task_id == "t1"
        tx.initialize.assert_awaited_once()

    async def test_tx_returned_to_pool_after_use(self):
        """单 tx pool 跑 2 个串行 task, 同一 tx 被调 2 次"""
        from src.core.qwen3_inproc_pool import Qwen3InProcPool

        tx = _make_tx_mock(tx_id=0)
        pool = Qwen3InProcPool(pool_size=1, transcriber_factory=lambda: tx)
        await pool.initialize()

        await pool.transcribe("a.wav", "t1")
        await pool.transcribe("b.wav", "t2")
        assert tx.transcribe.await_count == 2


# ==================== 并发分发 ====================


class TestConcurrentDispatch:
    async def test_two_concurrent_tasks_dispatched_to_separate_tx(self):
        """pool_size=2, 同时起 2 个 task, 各分发到不同 tx, 真并发"""
        from src.core.qwen3_inproc_pool import Qwen3InProcPool

        built = []

        def factory():
            tx = _make_tx_mock(tx_id=len(built), sleep_sec=0.05)
            built.append(tx)
            return tx

        pool = Qwen3InProcPool(pool_size=2, transcriber_factory=factory)
        await pool.initialize()

        results = await asyncio.gather(
            pool.transcribe("a.wav", "t1"),
            pool.transcribe("b.wav", "t2"),
        )
        assert len(results) == 2
        # 两个 tx 各被调一次 (任一 task 不会 acquire 同一个 tx)
        for tx in built:
            tx.transcribe.assert_awaited_once()

    async def test_overflow_task_queues_until_slot_available(self):
        """pool_size=2 + 3 并发 task: 第 3 个等到 1 个完成才 acquire.

        通过测时序 (sleep_sec 控制) 验证: 头 2 个 task ~ T+sleep 完成,
        第 3 个 task ~ T+2*sleep 完成. 用 await 顺序差异验证排队行为.
        """
        from src.core.qwen3_inproc_pool import Qwen3InProcPool

        # 每个 tx 跑一个 task 用 100ms
        built = []

        def factory():
            tx = _make_tx_mock(tx_id=len(built), sleep_sec=0.1)
            built.append(tx)
            return tx

        pool = Qwen3InProcPool(pool_size=2, transcriber_factory=factory)
        await pool.initialize()

        loop_start = asyncio.get_event_loop().time()
        complete_times: dict[str, float] = {}

        async def run_one(task_id: str):
            await pool.transcribe(f"{task_id}.wav", task_id)
            complete_times[task_id] = asyncio.get_event_loop().time() - loop_start

        await asyncio.gather(run_one("t1"), run_one("t2"), run_one("t3"))

        # 头 2 个 (t1+t2) 并发, ~100ms 完成 (允许 0.18 上限有点宽容值)
        early_finishers = sorted(complete_times.values())[:2]
        late_finisher = sorted(complete_times.values())[2]
        assert all(t < 0.18 for t in early_finishers), f"早完成的应 <180ms, got {early_finishers}"
        # 第 3 个等到前一个完成才开始, 应该 >= 0.18s (100ms + 100ms - 一点 overlap)
        assert late_finisher >= 0.18, f"第 3 个应排队 ~200ms, got {late_finisher}"


# ==================== 异常 / 边界 ====================


class TestErrorHandling:
    async def test_tx_returned_to_pool_when_transcribe_raises(self):
        """transcribe 抛错时 tx 必须回 pool, 否则后续 task 无法 acquire (slot 漏掉)"""
        from src.core.qwen3_inproc_pool import Qwen3InProcPool

        # tx[0] 抛错, tx[1] 正常
        tx_failing = _make_tx_mock(tx_id=0)
        tx_failing.transcribe = AsyncMock(side_effect=RuntimeError("boom"))
        tx_ok = _make_tx_mock(tx_id=1)

        txs_iter = iter([tx_failing, tx_ok])
        pool = Qwen3InProcPool(pool_size=2, transcriber_factory=lambda: next(txs_iter))
        await pool.initialize()

        # 第一个 task 抛错
        with pytest.raises(RuntimeError, match="boom"):
            await pool.transcribe("a.wav", "t1")

        # 第二个 task 仍能跑 (pool 没漏 slot)
        result, _ = await pool.transcribe("b.wav", "t2")
        assert result.task_id == "t2"
