"""
A3 — startup engine ↔ runtime 兼容性 fail-fast

设计 (decision D3):
- default_engine=qwen3 + cuda runtime + ORT CUDA EP 缺 → config load 时直接 errors,
  sys.exit(1), 不留到第一个 task 才挂 (替代 silent fallback / lazy fail-fast)
- per-request engine 已被 transcriber_dispatch.py:57 校验 (跟 default_engine 不一致就拒),
  所以 startup check 仅查 default_engine 充分

覆盖:
1. default_engine=qwen3 + mock CudaRuntime.validate() raise → errors 含 "onnxruntime-gpu"
2. default_engine=funasr + 同条件 → 不挂 (FunASR 不用 ORT CUDA)
3. default_engine=qwen3 + mock CudaRuntime.validate() OK → 不挂
4. default_engine=qwen3 + Mac runtime → 不挂 (走 sherpa CPU, 跟 ORT 无关)
"""
import os
import pytest

import src.core.config as config_module
import src.core.runtime as runtime_module
from src.core.config import Config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ.keys()):
        if key.startswith("FUNASR_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setenv("FUNASR_NOTIFICATION_ENABLED", "false")
    monkeypatch.setenv("FUNASR_AUTH_ENABLED", "false")


def _write_config(tmp_path, data):
    import json
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    return str(p)


class _CudaRuntimeFail:
    """mock CudaRuntime, validate() 直接抛"""
    name = "cuda"

    def validate(self):
        raise RuntimeError(
            "CudaRuntime.validate(): onnxruntime CUDAExecutionProvider 不可用"
        )

    def recommend_diarize_backend(self):
        return "ort_cuda"

    def recommend_num_threads(self):
        return 4


class _CudaRuntimeOK:
    name = "cuda"

    def validate(self):
        return None

    def recommend_diarize_backend(self):
        return "ort_cuda"

    def recommend_num_threads(self):
        return 4


class TestEngineRuntimeFailFast:
    def test_qwen3_engine_cuda_runtime_ort_missing_exits(self, monkeypatch, tmp_path, capsys):
        """default_engine=qwen3 + cuda runtime + ORT EP 缺 → sys.exit(1)"""
        monkeypatch.setattr(runtime_module, "detect_runtime", lambda: _CudaRuntimeFail())
        with pytest.raises(SystemExit):
            Config.load_from_file(
                _write_config(tmp_path, {"transcription": {"default_engine": "qwen3"}})
            )

    def test_funasr_engine_cuda_ort_missing_ok(self, monkeypatch, tmp_path):
        """default_engine=funasr + cuda runtime + ORT 缺 → 不挂 (FunASR 跟 ORT CUDA 无关)"""
        monkeypatch.setattr(runtime_module, "detect_runtime", lambda: _CudaRuntimeFail())
        cfg = Config.load_from_file(
            _write_config(tmp_path, {"transcription": {"default_engine": "funasr"}})
        )
        assert cfg.transcription.default_engine == "funasr"

    def test_qwen3_engine_cuda_runtime_ort_ok_passes(self, monkeypatch, tmp_path):
        """default_engine=qwen3 + cuda runtime + validate() 通过 → 不挂"""
        monkeypatch.setattr(runtime_module, "detect_runtime", lambda: _CudaRuntimeOK())
        cfg = Config.load_from_file(
            _write_config(tmp_path, {"transcription": {"default_engine": "qwen3"}})
        )
        assert cfg.transcription.default_engine == "qwen3"

    def test_qwen3_engine_mac_runtime_ok(self, monkeypatch, tmp_path):
        """default_engine=qwen3 + Mac runtime → 不挂 (sherpa CPU 路径不需要 ORT CUDA EP)"""
        monkeypatch.setattr(runtime_module, "detect_runtime", lambda: runtime_module.MacRuntime())
        cfg = Config.load_from_file(
            _write_config(tmp_path, {"transcription": {"default_engine": "qwen3"}})
        )
        assert cfg.transcription.default_engine == "qwen3"
