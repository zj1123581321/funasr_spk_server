# qwen_asr_gguf (vendored)

本目录是从 **[HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)**
集成进来的 GGUF/ONNX 混合 ASR 引擎源码(ONNX Runtime encoder/CTC + llama.cpp GGUF decoder,
API 兼容 sherpa-onnx)。

- **上游许可证**:MIT(见本目录 `LICENSE`,版权归 Haujet Zhao)
- **集成方式**:保留 upstream 原始结构,尽量不改源码,便于后续 sync upstream
- 主项目对 vendor 的适配(如复用 logger)见 `src/core/vendor/__init__.py`

依据 MIT 许可证,本目录保留了上游的版权与许可声明。主项目整体许可证见仓库根
`LICENSE`,第三方依赖与模型权重许可清单见根 `THIRD_PARTY_NOTICES.md`。
