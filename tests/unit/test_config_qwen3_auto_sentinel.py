"""
A2 上 — Qwen3Config num_threads/provider "auto" sentinel via Pydantic model_validator

设计 (decision D1):
- num_threads="auto" → 在 Pydantic model_validator 时一次性解析 = detect_runtime().recommend_num_threads()
- provider="auto" → 解析成 "cpu" (sherpa embedding extractor 在所有 runtime 都用 cpu;
  cuda runtime 上 sherpa diarize 已被 ort_cuda 替代, embedding 只在 cluster_merge 用一次)
- 字段类型仍是 int (Pydantic Union 解析时 "8" 字符串数字也会转 int, 只有 "auto" 走 fallback str)
- 下游 5 个消费点 (transcriber:58/237/399/575, inproc_pool:117) 零改动, 永远拿到 int

覆盖:
1. num_threads="auto" + mock Mac runtime → 4
2. num_threads="auto" + mock CudaRuntime(4vCPU) → 2
3. num_threads="auto" + mock CudaRuntime(8vCPU) → 4
4. num_threads=2 (显式 int) → 2 (不被覆盖)
5. provider="auto" + 任意 runtime → "cpu"
6. provider="cuda" (显式) → "cuda" (不被覆盖)
7. [回归] build_embedding_extractor_fn 用 cfg.num_threads, 不硬编码 4
8. [回归] Qwen3DiarizeTranscriber.__init__ 默认 num_threads=4 (跟 Qwen3Config 一致)
"""
import os
import pytest

from src.core.config import Qwen3Config
import src.core.runtime as runtime_module


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """避免 FUNASR_RUNTIME / dev .env 污染 runtime 探测"""
    for key in list(os.environ.keys()):
        if key.startswith("FUNASR_"):
            monkeypatch.delenv(key, raising=False)


def _mock_runtime(monkeypatch, runtime_obj):
    """patch detect_runtime() 返回指定 runtime 实例"""
    monkeypatch.setattr(runtime_module, "detect_runtime", lambda: runtime_obj)


class TestNumThreadsAuto:
    def test_auto_on_mac_runtime_resolves_to_4(self, monkeypatch):
        _mock_runtime(monkeypatch, runtime_module.MacRuntime())
        cfg = Qwen3Config(num_threads="auto")
        assert cfg.num_threads == 4
        assert isinstance(cfg.num_threads, int)

    def test_auto_on_cuda_4vcpu_resolves_to_2(self, monkeypatch):
        _mock_runtime(monkeypatch, runtime_module.CudaRuntime())
        monkeypatch.setattr(runtime_module, "_cpu_count", lambda: 4)
        cfg = Qwen3Config(num_threads="auto")
        assert cfg.num_threads == 2

    def test_auto_on_cuda_8vcpu_resolves_to_4(self, monkeypatch):
        _mock_runtime(monkeypatch, runtime_module.CudaRuntime())
        monkeypatch.setattr(runtime_module, "_cpu_count", lambda: 8)
        cfg = Qwen3Config(num_threads="auto")
        assert cfg.num_threads == 4

    def test_explicit_int_not_overridden(self, monkeypatch):
        """num_threads=2 (显式 int) 不被 "auto" 解析覆盖"""
        _mock_runtime(monkeypatch, runtime_module.CudaRuntime())
        monkeypatch.setattr(runtime_module, "_cpu_count", lambda: 32)
        cfg = Qwen3Config(num_threads=2)
        assert cfg.num_threads == 2

    def test_default_is_auto_resolving_to_runtime(self, monkeypatch):
        """Qwen3Config() 不传 num_threads → 默认 "auto" → 按当前 runtime 解析"""
        _mock_runtime(monkeypatch, runtime_module.MacRuntime())
        cfg = Qwen3Config()
        assert cfg.num_threads == 4  # MacRuntime 固定 4


class TestProviderAuto:
    def test_provider_auto_resolves_to_cpu_on_mac(self, monkeypatch):
        _mock_runtime(monkeypatch, runtime_module.MacRuntime())
        cfg = Qwen3Config(provider="auto")
        assert cfg.provider == "cpu"

    def test_provider_auto_resolves_to_cpu_on_cuda(self, monkeypatch):
        """cuda runtime 上 sherpa embedding 仍用 cpu (sherpa diarize 已被 ort_cuda 替代)"""
        _mock_runtime(monkeypatch, runtime_module.CudaRuntime())
        cfg = Qwen3Config(provider="auto")
        assert cfg.provider == "cpu"

    def test_explicit_provider_not_overridden(self, monkeypatch):
        _mock_runtime(monkeypatch, runtime_module.CudaRuntime())
        cfg = Qwen3Config(provider="cuda")
        assert cfg.provider == "cuda"


class TestDownstreamRegression:
    """D5: 顺手修 hardcode num_threads=4 + Qwen3DiarizeTranscriber 默认 8"""

    def test_build_embedding_extractor_reads_cfg_num_threads(self, monkeypatch):
        """qwen3_transcriber.py:58 不再硬编码 num_threads=4, 应从 cfg_like 读"""
        import sherpa_onnx
        from src.core.qwen3_transcriber import build_embedding_extractor_fn

        captured = {}

        class _FakeCfg:
            embedding_model = "/tmp/fake_emb.onnx"
            provider = "cpu"
            num_threads = 7  # 任意非 4 值, 验证 cfg 被读到

        # mock sherpa_onnx 避免真实加载
        class _FakeExtractorConfig:
            def __init__(self, model, num_threads, provider, debug):
                captured["num_threads"] = num_threads

            def validate(self):
                return True

        class _FakeExtractor:
            def __init__(self, _cfg):
                pass

        monkeypatch.setattr(sherpa_onnx, "SpeakerEmbeddingExtractorConfig", _FakeExtractorConfig)
        monkeypatch.setattr(sherpa_onnx, "SpeakerEmbeddingExtractor", _FakeExtractor)

        build_embedding_extractor_fn(_FakeCfg())
        assert captured["num_threads"] == 7

    def test_qwen3_diarize_transcriber_default_num_threads_matches_config(self):
        """Qwen3DiarizeTranscriber.__init__ 默认 num_threads 跟 Qwen3Config 默认值一致 (4)

        (Qwen3Config 默认是 "auto" sentinel, 解析后是 4 on Mac;
         transcriber __init__ 默认作为非 factory 路径的兜底, 应跟 mac default 4 一致, 不是历史的 8.)
        """
        import inspect
        from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber

        sig = inspect.signature(Qwen3DiarizeTranscriber.__init__)
        assert sig.parameters["num_threads"].default == 4
