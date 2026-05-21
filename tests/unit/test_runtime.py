"""runtime platform detection 单元测试.

覆盖 detect_runtime() 三个分支:
- darwin → name="mac_ane"
- linux + CUDA libs 在 → name="cuda"
- 其他 → name="cpu"

不真实加载 onnxruntime / sherpa lib, 全部 mock 平台 + 探测函数.
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_runtime_cache():
    """每个 test 跑前后清掉 module-level cache, 防止 cross-test 污染."""
    import importlib

    if "src.core.runtime" in sys.modules:
        importlib.reload(sys.modules["src.core.runtime"])
    yield
    if "src.core.runtime" in sys.modules:
        importlib.reload(sys.modules["src.core.runtime"])


def test_detect_runtime_returns_mac_on_darwin():
    with patch.object(sys, "platform", "darwin"):
        from src.core.runtime import detect_runtime

        runtime = detect_runtime()
        assert runtime.name == "mac_ane"


def test_detect_runtime_returns_cuda_when_libs_present_on_linux():
    with patch.object(sys, "platform", "linux"):
        from src.core import runtime as runtime_mod

        with patch.object(runtime_mod, "_has_cuda_runtime_available", return_value=True):
            r = runtime_mod.detect_runtime()
            assert r.name == "cuda"


def test_detect_runtime_returns_cpu_on_linux_without_cuda_libs():
    with patch.object(sys, "platform", "linux"):
        from src.core import runtime as runtime_mod

        with patch.object(runtime_mod, "_has_cuda_runtime_available", return_value=False):
            r = runtime_mod.detect_runtime()
            assert r.name == "cpu"


def test_detect_runtime_returns_cpu_on_windows():
    """非 darwin/linux 默认走 cpu, 保守兜底."""
    with patch.object(sys, "platform", "win32"):
        from src.core.runtime import detect_runtime

        r = detect_runtime()
        assert r.name == "cpu"


def test_detect_runtime_force_override_via_env(monkeypatch):
    """FUNASR_RUNTIME=cpu 强制覆盖, 给 dev/CI 留 escape hatch."""
    monkeypatch.setenv("FUNASR_RUNTIME", "cpu")
    with patch.object(sys, "platform", "darwin"):
        from src.core.runtime import detect_runtime

        r = detect_runtime()
        assert r.name == "cpu"


# ==================== validate() fail-fast ====================


def test_mac_validate_is_no_op():
    """MacRuntime.validate() 不做任何检查, 保持当前 production 行为 — 任何状态都不 raise."""
    from src.core.runtime import MacRuntime

    MacRuntime().validate()  # 不抛即通过


def test_cpu_validate_is_no_op():
    from src.core.runtime import CpuRuntime

    CpuRuntime().validate()


def test_cuda_validate_raises_when_cuda_ep_missing():
    """CUDA runtime 强制要求 CUDAExecutionProvider 可用, 缺则 fail-fast (替代 ORT silent fallback)."""
    from src.core import runtime as runtime_mod
    from src.core.runtime import CudaRuntime

    with patch.object(
        runtime_mod, "_available_ort_providers", return_value=["CPUExecutionProvider"]
    ):
        with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
            CudaRuntime().validate()


def test_cuda_validate_ok_when_cuda_ep_present():
    from src.core import runtime as runtime_mod
    from src.core.runtime import CudaRuntime

    with patch.object(
        runtime_mod,
        "_available_ort_providers",
        return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
    ):
        CudaRuntime().validate()  # 不抛即通过
