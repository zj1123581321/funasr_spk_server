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


# ==================== recommend_num_threads() — cpu_count 适配 ====================


def test_mac_recommend_num_threads_returns_4():
    """MacRuntime 固定 4 — Mac PoC 验证最优, 不依赖 cpu_count, 防 production 回归."""
    from src.core.runtime import MacRuntime

    assert MacRuntime().recommend_num_threads() == 4


def test_cuda_recommend_num_threads_small_vcpu_returns_2():
    """4 vCPU 上 sherpa num_threads=2 是 short audio sweet spot (60s wall 7.18s)."""
    from src.core import runtime as runtime_mod
    from src.core.runtime import CudaRuntime

    with patch.object(runtime_mod, "_cpu_count", return_value=4):
        assert CudaRuntime().recommend_num_threads() == 2


def test_cuda_recommend_num_threads_big_vcpu_returns_4():
    """≥8 vCPU 上推荐 4 — 长音频 sweet spot, 跟实测 5min wall 28.09s 一致."""
    from src.core import runtime as runtime_mod
    from src.core.runtime import CudaRuntime

    with patch.object(runtime_mod, "_cpu_count", return_value=8):
        assert CudaRuntime().recommend_num_threads() == 4


def test_cpu_recommend_num_threads_small_vcpu_returns_2():
    from src.core import runtime as runtime_mod
    from src.core.runtime import CpuRuntime

    with patch.object(runtime_mod, "_cpu_count", return_value=4):
        assert CpuRuntime().recommend_num_threads() == 2


def test_cpu_recommend_num_threads_big_vcpu_returns_4():
    from src.core import runtime as runtime_mod
    from src.core.runtime import CpuRuntime

    with patch.object(runtime_mod, "_cpu_count", return_value=16):
        assert CpuRuntime().recommend_num_threads() == 4


# ==================== recommend_diarize_backend() — env override ====================


def test_mac_recommend_diarize_backend_returns_sherpa():
    from src.core.runtime import MacRuntime

    assert MacRuntime().recommend_diarize_backend() == "sherpa"


def test_cuda_recommend_diarize_backend_default_ort_cuda():
    from src.core.runtime import CudaRuntime

    assert CudaRuntime().recommend_diarize_backend() == "ort_cuda"


def test_cpu_recommend_diarize_backend_returns_sherpa():
    from src.core.runtime import CpuRuntime

    assert CpuRuntime().recommend_diarize_backend() == "sherpa"


def test_recommend_diarize_backend_env_override_forces_sherpa_on_cuda(monkeypatch):
    """FUNASR_QWEN3_DIARIZE_BACKEND=sherpa 防回归 escape hatch, CUDA runtime 也能切回 sherpa."""
    monkeypatch.setenv("FUNASR_QWEN3_DIARIZE_BACKEND", "sherpa")
    from src.core.runtime import CudaRuntime

    assert CudaRuntime().recommend_diarize_backend() == "sherpa"


def test_recommend_diarize_backend_env_override_forces_ort_cuda_on_mac(monkeypatch):
    """env 强制 ort_cuda — 给 CUDA dev 在 Mac 上做 schema 验证留口子."""
    monkeypatch.setenv("FUNASR_QWEN3_DIARIZE_BACKEND", "ort_cuda")
    from src.core.runtime import MacRuntime

    assert MacRuntime().recommend_diarize_backend() == "ort_cuda"


# ==================== describe_runtime() — 启动可观测性 ====================


def test_describe_runtime_includes_name_backend_threads_for_mac():
    """启动日志一行讲清楚 runtime, 给运维快速判断."""
    from src.core.runtime import MacRuntime, describe_runtime

    line = describe_runtime(MacRuntime())
    assert "runtime=mac_ane" in line
    assert "diarize_backend=sherpa" in line
    assert "num_threads=4" in line


def test_describe_runtime_for_cuda_runtime():
    from src.core import runtime as runtime_mod
    from src.core.runtime import CudaRuntime, describe_runtime

    with patch.object(runtime_mod, "_cpu_count", return_value=8):
        line = describe_runtime(CudaRuntime())
    assert "runtime=cuda" in line
    assert "diarize_backend=ort_cuda" in line
    assert "num_threads=4" in line
