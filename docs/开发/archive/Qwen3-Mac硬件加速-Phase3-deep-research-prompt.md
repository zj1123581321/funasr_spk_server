# Deep Research Prompt — Qwen3-ASR backend ONNX 在 CoreML EP 上跑通的可行性

> 给 Gemini / ChatGPT (with deep research / browsing) 复制用
> 目标: 确认 Qwen3-ASR encoder backend ONNX 在 macOS CoreML Execution Provider 上跑通的最快路径

---

## 复制以下内容到 deep research

我有一个 Qwen3-ASR (1.7B) 模型的 encoder backend ONNX 文件 (583 MB), 在 macOS Apple Silicon 上想用 ONNX Runtime + CoreML Execution Provider (CoreML EP) 让它跑在 ANE 上, 但启动时报错 `axis 4 not in [-4, 3]`。

### 已知信息

- **模型来源**: Qwen3-ASR-1.7B, 由 CapsWriter-Offline (GitHub `HaujetZhao/CapsWriter-Offline`) release 提供的 GGUF + ONNX 包里的 backend ONNX
- **ONNX 文件特征**:
  - producer: `onnxruntime.transformers 1.23.2` (被 ORT transformers 优化器跑过, 不是原始 torch.onnx.export 输出)
  - opset: 19 (含 `com.microsoft`, `com.microsoft.nchwc`, `org.pytorch.aten` 等非标准域)
  - 901 个 node, 含 **25 个 `com.microsoft:BiasGelu` 融合 op**, 49 个 LayerNormalization, 96 个 Reshape, 96 个 Transpose (perm 全 4D), 194 个 MatMul, 196 个 Cast, 24 个 Softmax
  - **没有任何 op 的 `axis` / `axes` 属性 = 4**, 也没有 Squeeze/Unsqueeze 通过 input 传 axes=4, 也没有 Reshape 到 5D
  - inputs: `hidden_states [batch, time, 1024]` + `attention_mask [batch, 1, time, time]` (4D mask)
  - outputs: `last_hidden_state [batch, time, 2048]`
- **环境**: macOS 26.2 (Tahoe), M1 Max 64GB, onnxruntime 1.26.0, CoreML provider 编入
- **CoreML EP 配置**: `ModelFormat=MLProgram` + `MLComputeUnits=CPUAndNeuralEngine`
- **frontend ONNX (24MB)** 在同样 CoreML EP 配置下**跑通**, 仅 backend 失败
- **失败的不只 ANE units**: `ALL` units 和 `CPUAndNeuralEngine` units 都触发 axis 4 错误, 说明跟 GPU/ANE 路由无关, 是 op-level partition 问题

### 我需要 deep research 回答的问题

请逐条调研, 给出来源链接 (GitHub issue / discussion / 官方文档 / 论坛):

1. **CoreML EP "axis 4 not in [-4, 3]" 错误的已知报告**:
   - 在 microsoft/onnxruntime issues / discussions 里有没有人报过类似错误?
   - 触发条件通常是哪些 op? (Slice / Gather / Concat / Reshape / Squeeze ?)
   - 最近 6 个月有没有相关 fix / PR?
   - CoreML EP 当前对 tensor rank 的硬性限制是什么 (4D? 5D?)

2. **`com.microsoft` 域融合 op (尤其 BiasGelu) 在 CoreML EP 上的支持现状**:
   - CoreML EP 是否能处理 `com.microsoft:BiasGelu`? 还是 fallback 到 CPU?
   - 当 fallback 发生时, partition 边界是否会引入额外维度导致 axis 错误?
   - 有没有推荐做法让 ORT-optimized ONNX (含 com.microsoft 域) 在 CoreML EP 上跑?

3. **`onnxruntime.transformers` optimizer 是否有"关掉 BiasGelu / 关掉 attention fusion"的选项**:
   - 用 `--use_external_data_format` 之外, 还能怎么控制只做 layer-norm 之类的"安全"优化?
   - 反向工程: 拿到 ORT optimized ONNX 后能不能 "un-fuse" 回 standard ONNX?

4. **Qwen3-ASR 上游 PyTorch checkpoint 可获得性**:
   - HuggingFace `Qwen/Qwen3-ASR-1.7B` 是否公开?
   - ModelScope / 阿里魔搭社区是否有原始权重?
   - HaujetZhao/CapsWriter-Offline 是否在 release 里提供 PyTorch checkpoint 还是只有 ONNX/GGUF?

5. **attention_mask 4D shape `[batch, 1, time, time]` 在 CoreML EP 上的最佳形态**:
   - CoreML 是否支持 4D additive mask 直接 broadcast 加到 attention scores 上?
   - 是否应该改成 3D `[batch, time, time]` 或者拆成 bias 项?
   - HuggingFace transformers 的 ONNX export 配置(`OnnxConfig`) 是否有相关讨论?

6. **coremltools 新版本对高维 op 的支持**:
   - coremltools 8.x 的 model spec (mlprogram 7.x?) 是否扩展了 tensor rank 限制?
   - ONNX Runtime CoreML EP 跟 coremltools 的关系 — CoreML EP 是否会调用 coremltools 转换, 还是自带 converter?

7. **替代路径**:
   - macOS 上跑 ONNX 模型还有哪些后端能用 ANE? (e.g., 自己用 coremltools 把 ONNX → CoreML mlpackage)
   - `executorch` 是否成熟到可以替代 ONNX Runtime + CoreML EP?
   - MLX (Apple 自己的) 跑 transformer encoder 的状态?

8. **类似项目的经验**:
   - 别人在 Apple Silicon 上跑大尺寸 (>500MB) audio encoder ONNX 的故事 (Whisper / SenseVoice / 等等)
   - 他们怎么解决 "ONNX 含 com.microsoft 融合 op + 想用 ANE" 的冲突?

### 回答格式期望

每个问题给:
- **结论** (一句话)
- **关键证据** (1-3 个链接 + 简短引用)
- **建议行动** (如果适用)

整体可信度评估: 哪些问题的答案是社区共识 / 哪些是猜测。

---

## 用完 deep research 之后

把 Gemini/ChatGPT 的回答存到 `docs/开发/Qwen3-Mac硬件加速-Phase3-deep-research-output.md` (或类似命名), Claude 新 session 启动时会读。
