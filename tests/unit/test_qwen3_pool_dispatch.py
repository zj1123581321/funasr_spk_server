"""get_qwen3_pool_transcriber() 的 runtime-aware dispatch 测试.

设计:
- CudaRuntime → Qwen3InProcPool (单进程 N 实例, 避开 CUDNN cross-process race)
- MacRuntime / CpuRuntime → Qwen3PoolTranscriber (file-based multi-process pool)
- pool_size 仍从 config.transcription.qwen3_pool_size 读, 两个路径共用

测试用 monkeypatch 替换 detect_runtime, 不依赖真实 platform 探测.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_pool_singleton_before_each_test():
    """每个 test 重置单例, 避免相互污染"""
    from src.core.qwen3_pool_transcriber import reset_qwen3_pool_singleton

    reset_qwen3_pool_singleton()
    yield
    reset_qwen3_pool_singleton()


class TestRuntimeAwareDispatch:
    def test_cuda_runtime_returns_inproc_pool(self, monkeypatch):
        from src.core import qwen3_pool_transcriber as mod
        from src.core.qwen3_inproc_pool import Qwen3InProcPool
        from src.core.runtime import CudaRuntime

        monkeypatch.setattr(mod, "detect_runtime", lambda: CudaRuntime())

        result = mod.get_qwen3_pool_transcriber()
        assert isinstance(result, Qwen3InProcPool)

    def test_mac_runtime_returns_multiprocess_pool(self, monkeypatch):
        from src.core import qwen3_pool_transcriber as mod
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber
        from src.core.runtime import MacRuntime

        monkeypatch.setattr(mod, "detect_runtime", lambda: MacRuntime())

        result = mod.get_qwen3_pool_transcriber()
        assert isinstance(result, Qwen3PoolTranscriber)

    def test_cpu_runtime_returns_multiprocess_pool(self, monkeypatch):
        """Linux 无 GPU (CpuRuntime) 仍走原 multi-process pool"""
        from src.core import qwen3_pool_transcriber as mod
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber
        from src.core.runtime import CpuRuntime

        monkeypatch.setattr(mod, "detect_runtime", lambda: CpuRuntime())

        result = mod.get_qwen3_pool_transcriber()
        assert isinstance(result, Qwen3PoolTranscriber)


class TestPoolSizeRespected:
    def test_inproc_pool_uses_config_pool_size(self, monkeypatch):
        """CUDA 路径 pool_size 从 config 读"""
        from src.core import qwen3_pool_transcriber as mod
        from src.core.runtime import CudaRuntime

        monkeypatch.setattr(mod, "detect_runtime", lambda: CudaRuntime())
        # 改 config 暴露的 pool_size
        from src.core.config import config

        original = config.transcription.qwen3_pool_size
        try:
            config.transcription.qwen3_pool_size = 2
            result = mod.get_qwen3_pool_transcriber()
            assert result.pool_size == 2
        finally:
            config.transcription.qwen3_pool_size = original

    def test_multiprocess_pool_uses_config_pool_size(self, monkeypatch):
        """Mac/CPU 路径 pool_size 透传到 FileBasedProcessPool"""
        from src.core import qwen3_pool_transcriber as mod
        from src.core.runtime import MacRuntime

        monkeypatch.setattr(mod, "detect_runtime", lambda: MacRuntime())
        from src.core.config import config

        original = config.transcription.qwen3_pool_size
        try:
            config.transcription.qwen3_pool_size = 3
            result = mod.get_qwen3_pool_transcriber()
            assert result._pool.pool_size == 3
        finally:
            config.transcription.qwen3_pool_size = original


class TestSingletonCache:
    def test_singleton_returns_same_instance(self, monkeypatch):
        """同一进程内 get_qwen3_pool_transcriber 多次调返回同一 instance"""
        from src.core import qwen3_pool_transcriber as mod
        from src.core.runtime import CudaRuntime

        monkeypatch.setattr(mod, "detect_runtime", lambda: CudaRuntime())

        a = mod.get_qwen3_pool_transcriber()
        b = mod.get_qwen3_pool_transcriber()
        assert a is b
