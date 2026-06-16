"""Lane 1 (#17) VRAM preflight 探针单测 — src/core/gpu_mem.py

覆盖 free_vram_mib() 三路兜底 (正常解析 / 不可用→None / 超时→None /
非数字→None) + CUDA_VISIBLE_DEVICES 设备选择 (codex #12) + has_headroom
纯函数门控 (None→不误杀, 低于阈值→挡).
"""
from __future__ import annotations

import subprocess

import pytest

from src.core import gpu_mem


# ──────────────────────────────────────────────────────────────────────────
# free_vram_mib(): nvidia-smi 解析 + 兜底
# ──────────────────────────────────────────────────────────────────────────
def _fake_run(stdout: str):
    """造一个 subprocess.run 替身, 返回给定 stdout (returncode 0)."""

    def _run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")

    return _run


def test_free_vram_mib_parses_nvidia_smi(monkeypatch):
    """nvidia-smi 正常 → 解析出 free MiB int."""
    monkeypatch.setattr(subprocess, "run", _fake_run("8192\n"))
    assert gpu_mem.free_vram_mib() == 8192


def test_free_vram_mib_strips_whitespace_and_units(monkeypatch):
    """输出带空白/多行 (nounits 后仍可能多卡多行) → 取第一行 int."""
    monkeypatch.setattr(subprocess, "run", _fake_run("  3090 \n"))
    assert gpu_mem.free_vram_mib() == 3090


def test_free_vram_mib_returns_none_when_nvidia_smi_absent(monkeypatch):
    """非 CUDA 机 / nvidia-smi 不在 PATH (FileNotFoundError) → None (不抛)."""

    def _raise(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert gpu_mem.free_vram_mib() is None


def test_free_vram_mib_returns_none_on_timeout(monkeypatch):
    """nvidia-smi 卡死超时 → None."""

    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

    monkeypatch.setattr(subprocess, "run", _raise)
    assert gpu_mem.free_vram_mib() is None


def test_free_vram_mib_returns_none_on_nonzero_exit(monkeypatch):
    """nvidia-smi 返回非零 (check=True 抛 CalledProcessError) → None."""

    def _raise(*a, **k):
        raise subprocess.CalledProcessError(returncode=2, cmd="nvidia-smi")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert gpu_mem.free_vram_mib() is None


def test_free_vram_mib_returns_none_on_garbage(monkeypatch):
    """输出不是数字 (驱动异常) → None, 不崩."""
    monkeypatch.setattr(subprocess, "run", _fake_run("N/A\n"))
    assert gpu_mem.free_vram_mib() is None


# ──────────────────────────────────────────────────────────────────────────
# CUDA_VISIBLE_DEVICES 设备选择 (codex #12: 不读死第一行)
# ──────────────────────────────────────────────────────────────────────────
def test_free_vram_mib_respects_cuda_visible_devices(monkeypatch):
    """CUDA_VISIBLE_DEVICES=1 → nvidia-smi -i 1 查指定卡, 不读死 GPU 0."""
    captured = {}

    def _run(cmd, *a, **k):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="4096\n", stderr="")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(subprocess, "run", _run)
    assert gpu_mem.free_vram_mib() == 4096
    assert "-i" in captured["cmd"]
    assert "1" in captured["cmd"]


def test_free_vram_mib_defaults_to_index_0(monkeypatch):
    """CUDA_VISIBLE_DEVICES 未设 → 默认查 index 0."""
    captured = {}

    def _run(cmd, *a, **k):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="100\n", stderr="")

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(subprocess, "run", _run)
    assert gpu_mem.free_vram_mib() == 100
    assert "-i" in captured["cmd"]
    assert "0" in captured["cmd"]


def test_free_vram_mib_takes_first_of_csv_cuda_visible_devices(monkeypatch):
    """CUDA_VISIBLE_DEVICES='2,3' → 进程的 device 0 = 物理卡 2, 查 -i 2."""
    captured = {}

    def _run(cmd, *a, **k):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="555\n", stderr="")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr(subprocess, "run", _run)
    assert gpu_mem.free_vram_mib() == 555
    assert "2" in captured["cmd"]


# ──────────────────────────────────────────────────────────────────────────
# used_vram_mib(): 同源, 给 poison dispose delta 观测
# ──────────────────────────────────────────────────────────────────────────
def test_used_vram_mib_parses(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run("2864\n"))
    assert gpu_mem.used_vram_mib() == 2864


def test_used_vram_mib_none_on_failure(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert gpu_mem.used_vram_mib() is None


# ──────────────────────────────────────────────────────────────────────────
# has_headroom(): 纯函数门控 (None→不误杀)
# ──────────────────────────────────────────────────────────────────────────
def test_has_headroom_none_probe_does_not_block():
    """探不到显存 (None) → True, 不误杀, 交给 OOM fallback (codex #11)."""
    assert gpu_mem.has_headroom(None, 4608) is True


def test_has_headroom_enough():
    """free ≥ 阈值 → True."""
    assert gpu_mem.has_headroom(5000, 4608) is True
    assert gpu_mem.has_headroom(4608, 4608) is True


def test_has_headroom_insufficient():
    """free < 阈值 → False (走 CPU)."""
    assert gpu_mem.has_headroom(1200, 4608) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
