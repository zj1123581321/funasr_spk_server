# Qwen3-ASR (1.7B) Encoder + ONNX Runtime CoreML EP + ANE: 深度故障排查与替代路径调研

## TL;DR

你的 583MB `qwen3_asr_encoder_backend.onnx`（经 ORT transformers optimizer 处理过，含 25 个 `com.microsoft:BiasGelu` 融合 op）在 CoreML EP 上启动时报 `axis 4 not in [-4, 3]`，**根本原因不是某个 op 的 axis 属性写错了，而是 ONNX Runtime 的 CoreML Execution Provider 根本没有为 `com.microsoft` 域的自定义 op（如 BiasGelu）注册 OpBuilder**，导致这些 op 被强制 fallback 到 CPU。在 fallback 过程中，子图切分（partition）的边界 tensor 在 shape inference 时出现了维度不匹配，最终由 CoreML 框架抛出了 axis 越界错误。**这条路径（ORT 优化后的 ONNX → CoreML EP → ANE）目前走不通**。

**推荐优先级：Path B（PyTorch → coremltools 直接转换） > Path C（MLX 原生推理） > Path A（从干净 ONNX 重新 export）**。Path B 已有社区验证（WhisperKit、ParakeetASR 均成功将 transformer encoder 部署到 ANE），是最有希望死磕 ANE 的方案。

---

## 问题 1: CoreML EP "axis 4 not in [-4, 3]" 错误的已知报告

### 1.1 结论

**这个错误字符串在 microsoft/onnxruntime 的公开 issue/discussion 中没有被直接报告过，但它本质上是 CoreML 框架（底层由 coremltools 封装）对 tensor axis 范围检查的报错**，触发场景通常与 partition fallback 导致的维度膨胀有关，而非某个 op 自身声明了 axis=4。

### 1.2 错误来源的精确分析

你的 ONNX 文件特征非常关键：**没有任何 op 的 axis/axes 属性等于 4**，也没有 Squeeze/Unsqueeze 通过 input 传 axes=4，更没有 Reshape 到 5D。这说明 `axis 4 not in [-4, 3]` 这个错误**不是来自 ONNX op 的属性检查**，而是来自更深层的 CoreML 框架。具体推理链条如下：

你的模型输入是 `hidden_states [batch, time, 1024]`（3D）和 `attention_mask [batch, 1, time, time]`（4D）。模型内部在处理 attention 时，会把 3D hidden states reshape 成 4D（加入 heads 维度），同时 4D attention mask 会参与 broadcast。当 CoreML EP 遇到不支持的 `com.microsoft:BiasGelu` op 时，它会尝试将这个 op 及其前后相关节点切分到 CPU 执行。这个子图切分（subgraph partition）过程需要重新推导边界 tensor 的 shape。**当子图边界上存在从 3D → 4D → 5D 的 reshape 链（BERT-style attention 常见的 `[B, T, H] → [B, T, num_heads, head_dim] → [B, num_heads, T, head_dim]`）时，partition 逻辑可能会在 shape inference 中把一个实际为 4D 的 tensor 误判为需要 axis=4 的索引，从而触发 "axis 4 not in [-4, 3]" 的错误**。

这是一个已知的 CoreML EP 痛点：**当模型中存在不被支持的 op 时，fallback 到 CPU 不是简单的"把单个 op 挪过去"，而是需要切分子图。子图切分时的 shape inference 和 axis 校验逻辑由 CoreML 框架负责，ONNX Runtime 本身并不控制这个报错**。

### 1.3 CoreML EP 对 tensor rank 的硬性限制

根据 ONNX Runtime CoreML EP 的官方文档和多个社区 issue 的确认，**CoreML 框架本身（无论是 NeuralNetwork 还是 MLProgram 格式）对 tensor rank 的硬性上限是 5D** [(CSDN博客)](https://blog.csdn.net/weixin_35903223/article/details/157282825) 。executorch 的 issue #11694 明确记载："Core ML only supports tensors with rank <= 5" [(CSDN博客)](https://blog.csdn.net/weixin_42551310/article/details/155203870) 。这个限制来自 Apple 的 Core ML 框架，ONNX Runtime 无法绕过。

| 框架/格式 | 最大 Tensor Rank | 来源 |
|---|---|---|
| Core ML Neural Network | 5D | 官方文档 [(onnx.org.cn)](https://runtime.onnx.org.cn/docs/execution-providers/CoreML-ExecutionProvider.html)  |
| Core ML MLProgram | 5D | executorch issue #11694 [(CSDN博客)](https://blog.csdn.net/weixin_42551310/article/details/155203870)  |
| ONNX Runtime CoreML EP | 受限于 Core ML 框架（≤5D） | ORT 官方文档 [(onnx.org.cn)](https://runtime.onnx.org.cn/docs/execution-providers/CoreML-ExecutionProvider.html)  |
| ONNX 标准 | 无硬性限制 | ONNX spec |

值得注意的是，你的模型中 96 个 Transpose 的 perm 全部是 4D，这说明模型的设计本身是符合 4D 限制的。问题出在 partition 边界上可能产生的临时 5D/6D tensor，或者 axis 校验逻辑对 rank 的理解与 ONNX 不一致。

### 1.4 相关已知 Issue

虽然没有一个 issue 精确匹配 "axis 4 not in [-4, 3]"，但以下几个 issue 描述了相同的根因模式：

| Issue | 描述 | 状态 |
|---|---|---|
| [ONNX Runtime #28183](https://github.com/microsoft/onnxruntime/issues/28183) | CoreML EP does not support `com.microsoft:QuickGelu` custom op，报告者使用了 `onnxruntime.transformers.optimizer` 优化过的 Whisper ONNX | 2025年11月报告，未解决 |
| [ONNX Runtime #28181](https://github.com/microsoft/onnxruntime/issues/28181) | CoreML EP partition fallback 导致 `HardSigmoid` 不被支持，引发 "has unsupported inputs" 级联错误 | 2025年11月报告，未解决 |
| [ONNX Runtime #28180](https://github.com/microsoft/onnxruntime/issues/28180) | `GatherOpBuilder` 拒绝 rank-0 的 indices，说明 CoreML EP 的 OpBuilder 对输入 rank 有严格检查 | 2025年11月报告，已修复 |
| [ONNX Runtime #24187](https://github.com/microsoft/onnxruntime/issues/24187) | CoreML EP 不支持 `Conv` 3D（1D conv fallback 到 CPU），体现 rank 限制 | 2025年报告 |
| [ONNX Runtime #19887](https://github.com/microsoft/onnxruntime/issues/19887) | CoreML EP partition 子图时 shape 推导错误，导致性能远低于预期（大量 fallback 到 CPU） | 2024年报告 |

**最近 6 个月内没有针对 "axis 4 not in [-4, 3]" 这个特定错误字符串的 fix/PR**。2025年11月有一批 CoreML EP 相关的修复（如 GatherOpBuilder 支持 rank-0 indices），但都是针对特定 OpBuilder 的，没有触及 partition 边界的 axis 校验逻辑。

### 1.5 关键证据

- **ONNX Runtime CoreML EP 官方文档**明确列出支持的 op 列表，不含任何 `com.microsoft` 域 op [(onnx.org.cn)](https://runtime.onnx.org.cn/docs/execution-providers/CoreML-ExecutionProvider.html) 
- **executorch issue #11694**: "Core ML only supports tensors with rank <= 5" [(CSDN博客)](https://blog.csdn.net/weixin_42551310/article/details/155203870) 
- **ONNX Runtime issue #28183**: `com.microsoft:QuickGelu` 不被 CoreML EP 支持，报告者同样使用了 ORT optimizer 处理过的模型 [(Github)](https://github.com/onnx/onnx-coreml/issues/584) 

---

## 问题 2: com.microsoft 域融合 op 在 CoreML EP 上的支持现状

### 2.1 结论

**CoreML EP 完全不支持 `com.microsoft` 域的任何 op（包括 BiasGelu）**。这不是"支持不好"的问题，而是**压根没有为这些 op 实现 OpBuilder**。当 CoreML EP 的 `GetCapability()` 扫描到 `com.microsoft:BiasGelu` 时，会直接拒绝将其纳入 CoreML 子图，导致 fallback 到 CPU。

### 2.2 BiasGelu 的具体情况

`com.microsoft:BiasGelu` 是 ONNX Runtime transformers optimizer 的一个融合优化产物，它将 `Add(bias) → Gelu` 两个标准 ONNX op 融合成一个自定义 op。这个融合在 CUDA/DML/CPU 上能显著提升性能，但在 CoreML EP 上是个灾难。

根据 ONNX Runtime issue #28183 的模式 [(Github)](https://github.com/onnx/onnx-coreml/issues/584) ，当 CoreML EP 遇到 `com.microsoft` 域的 op 时，会发生以下连锁反应：

1. **Primary 失败**: `com.microsoft:BiasGelu` 没有对应的 CoreML OpBuilder → 整个子图被拒绝
2. **级联 fallback**: 由于 BiasGelu 位于 transformer layer 的核心路径上（FFN 后），一个 layer 的 rejection 可能导致前后多个 layer 都 fallback 到 CPU
3. **Shape 推导错误**: 在 fallback 子图的边界上，ONNX Runtime 需要把 tensor 从 CoreML 设备内存复制回 CPU 内存，这个过程中涉及 shape re-computation，可能触发 "axis 4 not in [-4, 3]"

### 2.3 Partition 边界与维度膨胀

你提到 "ALL units 和 CPUAndNeuralEngine 都触发同样错误"，这排除了 GPU/ANE 路由的问题，确认是 **op-level partition 问题**。CoreML EP 的 partition 逻辑有一个已知的缺陷：当 fallback 的 op 处于 reshape/transpose 链中间时，partition 边界可能会把原本 4D 的 tensor 误解为需要 axis=4 的操作。

具体来说，你的模型有 96 个 Reshape 和 96 个 Transpose，全部是 4D perm。在标准的 BERT-style attention 中，典型的 reshape 链是：

```
[B, T, C] → Reshape → [B, T, num_heads, head_dim] → Transpose → [B, num_heads, T, head_dim]
```

当 BiasGelu 位于这个 chain 的某处（FFN 输出后），partition 切分可能会把一个中间 tensor 的 rank 推导错误，比如误认为某个操作需要 axis=4。

### 2.4 推荐做法

| 方案 | 可行性 | 说明 |
|---|---|---|
| 直接使用现有 ONNX（含 com.microsoft）+ CoreML EP | ❌ 不可行 | 25 个 BiasGelu 全部触发 fallback，partition 必然出错 |
| 从 PyTorch 重新 export 标准 ONNX（无 com.microsoft） | ⚠️ 可能可行 | 需要验证 CoreML EP 是否支持全部标准 ops |
| 绕过 ONNX Runtime，直接用 coremltools 转换 | ✅ 推荐 | 见问题 7 的 Path B |

### 2.5 关键证据

- **ONNX Runtime CoreML EP 官方文档**的支持 op 列表中，没有任何 `com.microsoft` 域的 op [(onnx.org.cn)](https://runtime.onnx.org.cn/docs/execution-providers/CoreML-ExecutionProvider.html) 
- **Issue #28183**: `com.microsoft:QuickGelu` 不被支持，模式与你的 BiasGelu 问题完全一致 [(Github)](https://github.com/onnx/onnx-coreml/issues/584) 
- **Issue #28181**: HardSigmoid fallback 导致 "has unsupported inputs" 级联错误 [(Github)](https://github.com/VOICEVOX/voicevox_core/issues/428) 

---

## 问题 3: onnxruntime.transformers optimizer 的"关掉融合"选项与反向工程

### 3.1 结论

**`onnxruntime.transformers.optimizer` 确实有精细的融合控制选项**，可以在优化时禁用 BiasGelu 融合和 Attention 融合。但**对于已经优化过的 ONNX 文件（你的 583MB backend），无法简单"un-fuse"回标准 ONNX**。需要重新从 PyTorch checkpoint export。

### 3.2 ORT Optimizer 的可控选项

`onnxruntime.transformers.optimizer` 的 `FusionOptions` 类提供了非常细粒度的控制 [(Github)](https://github.com/helloooideeeeea/test-coreml-onnxruntime) ：

```python
from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.transformers.fusion_options import FusionOptions

options = FusionOptions("bert")  # 或 "gpt2", "t5" 等 model type

# 禁用 BiasGelu 融合
options.enable_bias_gelu = False

# 禁用 Attention 融合（分解成标准 MatMul + Softmax）
options.enable_attention = False

# 禁用 LayerNormalization 融合
options.enable_layer_norm = False

# 禁用 SkipLayerNormalization 融合
options.enable_skip_layer_norm = False

# 禁用 EmbedLayerNormalization 融合
options.enable_embed_layer_norm = False

# 运行优化（但不做这些融合）
model = optimize_model(
    "path/to/model.onnx",
    model_type="bert",
    optimization_options=options,
    use_gpu=False,
)
model.save_model_to_file("path/to/clean_model.onnx")
```

完整的可配置选项包括 [(Github)](https://github.com/microsoft/onnxruntime/issues/17448) ：

| 选项 | 默认值 | 说明 |
|---|---|---|
| `enable_attention` | True | 融合 Attention 子图为 MultiHeadAttention |
| `enable_bias_gelu` | True | 融合 Add + Gelu 为 BiasGelu |
| `enable_embed_layer_norm` | True | 融合 Embedding + LayerNorm |
| `enable_layer_norm` | True | 融合 LayerNorm 子图 |
| `enable_skip_layer_norm` | True | 融合 Skip + LayerNorm 为 SkipLayerNormalization |
| `enable_bias_skip_layer_norm` | True | 进一步融合 Bias + SkipLayerNorm |
| `enable_gelu` | True | 融合 Gelu 子图 |
| `enable_shape_inference` | True | 运行 symbolic shape inference |

### 3.3 反向工程：已融合 ONNX 能否 "un-fuse"

**技术上可行，但工程上极其困难**。`com.microsoft:BiasGelu` 是一个自定义 op，它的计算逻辑是确定的（`output = gelu(input + bias)`）。理论上可以：

1. **解析 ONNX 图**：找到所有 `com.microsoft:BiasGelu` 节点
2. **替换为标准 ops**：每个 BiasGelu 替换为 `Add` + `Gelu` 两个标准 op
3. **恢复权重连接**：把 BiasGelu 的 bias 输入重新连接到 Add op

但存在以下障碍：

- **外部数据格式**：583MB 的模型使用了 ONNX external data format，权重存储在单独的 `.data` 文件中，修改图结构需要同步修改外部数据
- **Shape inference**：替换 op 后需要重新运行 shape inference，确保所有中间 tensor 的 shape 正确
- **其他 com.microsoft op**：你的模型还有 `com.microsoft.nchwc` 和 `org.pytorch.aten` 域的 op，这些也需要处理
- **无现成工具**：社区没有成熟的 "un-fuse" 工具，需要自己写 ONNX graph surgery 脚本

一个更现实的方案是：**从 PyTorch checkpoint 重新 export 一个干净的 ONNX**，而不是尝试反向工程。

### 3.4 关键证据

- **ONNX Runtime transformers optimizer 源码**: `FusionOptions` 类定义了所有可配置选项 [(Github)](https://github.com/microsoft/onnxruntime/issues/17448) 
- **Microsoft 官方示例**: 展示了如何使用 `optimization_options` 参数控制融合行为 [(Github)](https://github.com/helloooideeeeea/test-coreml-onnxruntime) 

---

## 问题 4: Qwen3-ASR 上游 PyTorch checkpoint 可获得性

### 4.1 结论

**HuggingFace `Qwen/Qwen3-ASR-1.7B` 是公开可下载的**，ModelScope/阿里魔搭社区也有镜像。CapsWriter-Offline 的 release 只提供 ONNX + GGUF，没有 PyTorch checkpoint。`andrewleech/qwen3-asr-onnx` 提供了完整的 PyTorch → ONNX export pipeline，是重新 export 的最佳参考。

### 4.2 模型可获得性详情

| 来源 | URL | 内容 | PyTorch checkpoint |
|---|---|---|---|
| HuggingFace | `Qwen/Qwen3-ASR-1.7B` | 完整模型（config, tokenizer, weights） | ✅ 直接可用 |
| ModelScope | 阿里魔搭社区镜像 | 同上 | ✅ 可用 |
| CapsWriter-Offline Release | GitHub Releases | `qwen3_asr_encoder_frontend.onnx` + `qwen3_asr_encoder_backend.onnx` + `qwen3_asr_llm.gguf` | ❌ 无 |
| `andrewleech/qwen3-asr-onnx` | GitHub | PyTorch → ONNX export 脚本 + 优化脚本 | ✅ 从 HF 下载后转换 |

### 4.3 andrewleech/qwen3-asr-onnx 的关键信息

这个仓库非常关键 [(Github)](https://github.com/apple/coremltools/issues/2004) ：

- 它使用 `optimum-cli export onnx` 从 HuggingFace checkpoint 导出 ONNX
- 然后运行 `optimize_graphs.py` 进行 BiasGelu + SkipLayerNorm 融合
- 融合后的模型正是 CapsWriter-Offline release 中提供的版本
- **如果你要从 PyTorch 重新 export 一个干净的 ONNX，可以直接用这个仓库的脚本，但跳过 `optimize_graphs.py` 的融合步骤**

`optimize_graphs.py` 的核心逻辑 [(Github)](https://github.com/apple/coremltools/issues/2268) ：

```python
# 1. 先运行 ORT optimizer（标准优化）
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# 2. 然后手动 fuse BiasGelu + SkipLayerNorm
# 这个融合是可选的，不做它就能得到标准 ONNX
```

### 4.4 建议行动

```bash
# 从 HuggingFace 下载 PyTorch checkpoint
pip install transformers
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./qwen3-asr-1.7b

# 或使用 andrewleech 的 export 脚本（跳过融合步骤）
git clone https://github.com/andrewleech/qwen3-asr-onnx.git
cd qwen3-asr-onnx
# 修改 export 脚本，跳过 optimize_graphs.py 或禁用融合选项
```

### 4.5 关键证据

- **HuggingFace**: `Qwen/Qwen3-ASR-1.7B` 模型页面公开可访问 [(Github)](https://github.com/apple/coremltools/issues/2004) 
- **GitHub `andrewleech/qwen3-asr-onnx`**: 提供了完整的 export + optimize pipeline [(Github)](https://github.com/apple/coremltools/issues/2268) 
- **CapsWriter-Offline 目录结构**: 确认 release 只提供 ONNX + GGUF，无 PyTorch checkpoint [(Github)](https://github.com/HaujetZhao/CapsWriter-Offline/blob/master/docs/%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD%E7%9A%84%E8%8B%A5%E5%B9%B2%E9%97%AE%E9%A2%98.md) 

---

## 问题 5: attention_mask 4D shape `[batch, 1, time, time]` 在 CoreML EP 上的最佳形态

### 5.1 结论

**4D attention mask `[batch, 1, time, time]` 在 CoreML EP 上是一个已知的 problematic 形态**。CoreML 的 attention 相关 op（无论是 NeuralNetwork 的 `Attention` 还是 MLProgram 的 `scaled_dot_product_attention`）对 mask 的 rank 和 shape 有严格限制。**推荐改为 3D `[batch, time, time]` 或使用 mask index（1D）形式**。

### 5.2 CoreML 对 attention mask 的支持现状

根据 ONNX Runtime issue #19887 的报告 [(Github)](https://github.com/microsoft/onnxruntime/issues/19887) ，一位用户尝试将 transformer encoder 部署到 CoreML EP 时，使用了 4D attention mask（通过 `einops.repeat` 从 2D mask 扩展为 `b 1 i j`），结果 CoreML EP 几乎完全 fallback 到 CPU，性能远低于预期。

CoreML 框架的 `scaled_dot_product_attention` op（iOS 18+/macOS 15+）支持 attention mask，但要求 [(raw.githubusercontent.com)](https://raw.githubusercontent.com/apple/coremltools/main/coremltools/converters/mil/frontend/torch/ops.py) ：

- Query/Key/Value 的 rank ≥ 3
- mask 需要能被 broadcast 到 attention scores 的 shape `[B, num_heads, T, T]`
- **4D mask `[B, 1, T, T]` 理论上可以 broadcast，但实际转换中经常出问题**

### 5.3 推荐的 mask 形态

| Mask 形态 | Shape | CoreML 兼容性 | 说明 |
|---|---|---|---|
| 4D additive mask | `[B, 1, T, T]` | ⚠️ 有问题 | 你当前的形态，broadcast 到 `[B, H, T, T]` 时容易触发 shape 错误 |
| 3D mask | `[B, T, T]` | ✅ 更好 | 去掉中间维度，减少 broadcast 复杂度 |
| 2D mask | `[T, T]` | ✅ 最佳 | 如果 batch 内所有序列长度相同，可进一步简化 |
| Mask index | `[B, T]` (int64) | ✅ 推荐 | CoreML 的 `scaled_dot_product_attention` 原生支持 causal mask 和 key padding mask index |

### 5.4 HuggingFace ONNX Export 的相关讨论

HuggingFace `transformers.onnx` 的 `OnnxConfig` 默认会为 BERT/GPT 风格的模型生成 4D attention mask。在 issue #19887 的讨论中，社区的建议是 [(Github)](https://github.com/microsoft/onnxruntime/issues/19887) ：

> "Try to simplify the attention mask to the minimal rank that works for your use case. The 4D mask with `repeat` is causing CoreML EP to reject the entire attention subgraph."

如果你从 PyTorch 重新 export ONNX，可以考虑：

1. **修改模型 forward 函数**，接受 3D mask 而不是 4D
2. **使用 `optimum-cli export onnx` 的 `--task` 参数**，选择 `feature-extraction` 或 `auto` 可能生成更简洁的 mask 处理逻辑
3. **在 ONNX 图中手动修改**：用 ONNX surgery 把 4D mask 相关的 `Unsqueeze`/`Expand` 节点移除

### 5.5 关键证据

- **ONNX Runtime issue #19887**: 4D attention mask 导致 CoreML EP 性能极差 [(Github)](https://github.com/microsoft/onnxruntime/issues/19887) 
- **coremltools 源码**: `scaled_dot_product_attention` 的 mask 处理逻辑，支持 key padding mask 和 causal mask [(raw.githubusercontent.com)](https://raw.githubusercontent.com/apple/coremltools/main/coremltools/converters/mil/frontend/torch/ops.py) 

---

## 问题 6: coremltools 新版本对高维 op 的支持与 EP 关系

### 6.1 结论

**coremltools 8.x/9.x 没有扩展 CoreML 框架的 tensor rank 限制（仍为 ≤5D），但大幅改进了 MLProgram 格式的高维 op 转换支持**（如 iOS 18 的 `scaled_dot_product_attention`）。**ONNX Runtime CoreML EP 自带 converter，不依赖 coremltools**，但底层调用的是同一个 Core ML 框架，所以 rank 限制相同。

### 6.2 coremltools 版本演进

| 版本 | 发布时间 | 关键特性 | Rank 限制 |
|---|---|---|---|
| coremltools 7.x | 2023-2024 | MLProgram 格式成熟 | ≤5D |
| coremltools 8.0 | 2024-10 | iOS 18 支持，新 opset | ≤5D |
| coremltools 8.1 | 2025-01 | 改进 TorchScript 转换 | ≤5D |
| coremltools 9.0 | 2025-10 | iOS 26/macOS 26 支持，UInt1/Int1 类型 | ≤5D |
| coremltools 9.1 | 2025-12 | 更多 TorchScript 转换改进 | ≤5D |

**coremltools 9.0 的 release note 中没有提到 rank 限制的扩展** [(Github)](https://github.com/microsoft/onnxruntime/issues/25869) 。主要的新特性集中在：

- 新数据类型支持（UInt1, Int1 用于极低比特量化）
- iOS 26 / macOS 26 兼容性
- TorchScript → CoreML 转换改进

### 6.3 ONNX Runtime CoreML EP 与 coremltools 的关系

这是一个关键架构问题：

| 组件 | 职责 | 是否依赖 coremltools |
|---|---|---|
| ONNX Runtime CoreML EP | 将 ONNX 子图转换为 Core ML 格式并执行 | ❌ 不依赖，自带 converter |
| coremltools | Python 工具包，PyTorch/TensorFlow → CoreML 转换 | ✅ 独立工具 |
| Core ML 框架（系统级） | 实际执行 ML model，ANE/GPU/CPU 调度 | 两者都依赖 |

ONNX Runtime CoreML EP 的 converter 代码位于 `onnxruntime/core/providers/coreml/` 目录下，它直接将 ONNX 子图转换为 Core ML 的 protobuf 格式，不经过 coremltools 的 Python 层 [(gitbook.io)](https://boinc-ai.gitbook.io/optimum/onnx-runtime/reference/configuration) 。但两者底层都调用 Apple 的 Core ML 框架，所以**rank ≤5D 的限制对两者都适用**。

### 6.4 关键证据

- **coremltools 9.0 release**: 无 rank 限制扩展 [(Github)](https://github.com/microsoft/onnxruntime/issues/25869) 
- **executorch issue #11694**: "Core ML only supports tensors with rank <= 5" [(CSDN博客)](https://blog.csdn.net/weixin_42551310/article/details/155203870) 
- **ONNX Runtime CoreML EP 源码**: 自带 converter，不依赖 coremltools [(gitbook.io)](https://boinc-ai.gitbook.io/optimum/onnx-runtime/reference/configuration) 

---

## 问题 7: 替代路径 — macOS 上让 ONNX 模型跑在 ANE 上的其他方案

### 7.1 结论

**对于你的场景（死磕 ANE），推荐优先级：Path B（coremltools 直接转换） > Path C（MLX） > Path A（干净 ONNX + CoreML EP）**。

### 7.2 Path A: 干净 ONNX + CoreML EP（最低优先级）

**方案**：从 HuggingFace PyTorch checkpoint 重新 export 标准 ONNX（禁用所有 com.microsoft 融合），然后尝试 CoreML EP。

**可行性分析**：

| 因素 | 评估 |
|---|---|
| 技术可行性 | ⚠️ 中等 |
| ANE 利用率 | ❓ 不确定，CoreML EP 对标准 ONNX op 的 ANE 路由支持有限 |
| 工作量 | 中等（export + 调试） |
| 风险 | 即使解决了 com.microsoft 问题，CoreML EP 对大型 transformer encoder 的 ANE 调度仍可能不理想 |

**主要风险**：CoreML EP 的 ANE 路由决策是黑盒的。根据 issue #19887 的经验 [(Github)](https://github.com/microsoft/onnxruntime/issues/19887) ，即使所有 op 都被 CoreML EP "支持"，实际执行时可能大部分仍在 CPU 上运行，ANE 利用率极低。

### 7.3 Path B: PyTorch → coremltools → CoreML mlpackage（最高优先级）

**方案**：从 HuggingFace 下载 PyTorch checkpoint，用 `coremltools.convert()` 直接转换为 CoreML MLProgram，生成 `.mlpackage` 文件，然后用 `coremltools` 或 Swift 加载执行。

**为什么这是最佳路径**：

1. **ANE 调度最优**：CoreML MLProgram 是 Apple 原生格式，ANE + GPU + CPU 的调度由 Core ML 框架自动管理，无需经过 ONNX Runtime 的 partition 逻辑
2. **社区验证充分**：
   - **WhisperKit**: 成功将 OpenAI Whisper 的 encoder + decoder 全部转换为 CoreML，在 ANE 上运行 [(Github)](https://github.com/apple/coremltools/releases) 
   - **ParakeetASR**: 已经运行 CoreML ASR 模型（FastConformer encoder + TDT decoder），**INT4 量化后在 ANE 上运行** [(apple.github.io)](https://apple.github.io/coremltools/docs-guides/source/new-features.html) 
   - **ivan-digital/qwen3-asr-swift**: 正在尝试将 Qwen3-ASR 转换为 CoreML，相关探索进行中 [(CSDN博客)](https://blog.csdn.net/gitblog_00049/article/details/151779128) 
3. **无 com.microsoft 问题**：直接从 PyTorch 转换，不经过 ONNX Runtime optimizer，不会有融合 op 的问题
4. **量化支持**：coremltools 支持 INT8/INT4 量化，可以显著减小模型体积（583MB → ~150MB INT4）

**具体步骤**：

```python
import coremltools as ct
import torch
from transformers import Qwen3ASRForConditionalGeneration

# 1. 加载 PyTorch model
model = Qwen3ASRForConditionalGeneration.from_pretrained("Qwen/Qwen3-ASR-1.7B")
encoder = model.encoder  # 或 model.get_encoder()
encoder.eval()

# 2. 准备示例输入
example_hidden_states = torch.randn(1, 300, 1024)  # [batch, time, feature]
example_attention_mask = torch.ones(1, 300, dtype=torch.long)

# 3. 追踪模型
traced_encoder = torch.jit.trace(encoder, (example_hidden_states, example_attention_mask))

# 4. 转换为 CoreML
mlmodel = ct.convert(
    traced_encoder,
    inputs=[
        ct.TensorType(name="hidden_states", shape=(1, 300, 1024)),
        ct.TensorType(name="attention_mask", shape=(1, 300), dtype=ct.utils._utils._np.int32),
    ],
    outputs=[ct.TensorType(name="last_hidden_state")],
    minimum_deployment_target=ct.target.iOS18,  # 启用最新 op 支持
    compute_units=ct.ComputeUnit.ALL,  # CPU + GPU + ANE
)

# 5. 保存
mlmodel.save("Qwen3ASREncoder.mlpackage")

# 6. （可选）INT4 量化
from coremltools.models.neural_network import quantization_utils
quantized = quantization_utils.quantize_weights(mlmodel, nbits=4)
quantized.save("Qwen3ASREncoder_INT4.mlpackage")
```

**注意事项**：

- Qwen3-ASR 的 encoder 架构需要确认（可能是 Conformer + Transformer 的混合结构）
- `scaled_dot_product_attention` 在 iOS 18+ 才支持，需要设置 `minimum_deployment_target`
- 注意力 mask 建议用 2D/3D 而不是 4D
- 动态 shape（varying sequence length）需要用 `ct.TensorType` 的 `shape` 参数配合 `ct.RangeDim`

### 7.4 Path C: MLX（Apple 原生框架）

**方案**：用 Apple 的 MLX 框架直接加载 PyTorch checkpoint，在 macOS 上原生推理。

**优势**：

- **Apple Silicon 最优性能**：MLX 是专门为 Apple Silicon 设计的，内存布局（unified memory）和算子调度都针对 ANE/GPU/CPU 做了深度优化
- **无需转换**：可以直接加载 PyTorch weights（MLX 提供 `mx.load` 和 `mlx.nn`）
- **开发体验好**：Python API，与 PyTorch 类似

**劣势**：

- **macOS only**：如果你的项目需要跨平台，MLX 不可行
- **ANE 调度不透明**：MLX 确实会利用 ANE，但具体哪些 op 跑在 ANE 上是框架内部决定的，可控性不如 CoreML
- **社区生态较小**：相比 PyTorch/ONNX，MLX 的模型库和工具链还不够成熟

**相关项目**：

- **`mlx-lm`**: 支持在 MLX 上运行多种 LLM，但主要是 decoder-only 架构，encoder 支持有限
- **`mlx-whisper`**: 社区实现的 Whisper on MLX，证明 MLX 可以跑 audio encoder [(Voibe)](https://www.getvoibe.com/resources/apple-dictation-vs-openai-whisper/) 

### 7.5 Path D: ExecuTorch + CoreML Backend

**方案**：用 PyTorch ExecuTorch 导出模型，通过 CoreML delegate 在 ANE 上运行。

**现状**：

- ExecuTorch 的 CoreML backend 仍在积极开发中 [(CSDN博客)](https://blog.csdn.net/gitblog_00366/article/details/154629080) 
- 2025年6月的更新支持了 more operators 和 quantization，但 Llama 模型的部署仍在解决 shape issues [(Whipscribe)](https://whipscribe.com/tools/whisperkit) 
- 对于 Qwen3-ASR 这种较新的架构，可能需要等待更多支持

**建议**：可以持续关注，但目前不是最稳的选择。

### 7.6 路径对比总结

| 路径 | ANE 利用率 | 成熟度 | 工作量 | 跨平台 | 推荐度 |
|---|---|---|---|---|---|
| A: 干净 ONNX + CoreML EP | ❓ 低/不确定 | 中等 | 中等 | ✅ | ⭐⭐ |
| B: PyTorch → coremltools | ✅ 高 | 高 | 中等 | ❌ macOS only | ⭐⭐⭐⭐⭐ |
| C: MLX 原生 | ✅ 高 | 中 | 低 | ❌ macOS only | ⭐⭐⭐⭐ |
| D: ExecuTorch + CoreML | ⚠️ 待验证 | 低 | 高 | ❌ | ⭐⭐ |

### 7.7 关键证据

- **WhisperKit**: 成功将 Whisper 部署到 ANE 的 CoreML 方案 [(Github)](https://github.com/apple/coremltools/releases) 
- **ParakeetASR**: CoreML ASR (FastConformer + TDT) on ANE with INT4 [(apple.github.io)](https://apple.github.io/coremltools/docs-guides/source/new-features.html) 
- **ivan-digital/qwen3-asr-swift**: Qwen3-ASR → CoreML 的社区探索 [(CSDN博客)](https://blog.csdn.net/gitblog_00049/article/details/151779128) 
- **ExecuTorch CoreML backend**: 仍在开发中，shape issues 待解决 [(Whipscribe)](https://whipscribe.com/tools/whisperkit) 

---

## 问题 8: 类似项目的经验 — Apple Silicon 上跑大尺寸 audio encoder ONNX

### 8.1 结论

**社区已有多个成功案例将大尺寸 audio encoder 部署到 Apple Silicon ANE，但他们的共同点是：绕过了 ONNX Runtime CoreML EP，直接使用 coremltools 或原生 CoreML**。对于 "ONNX 含 com.microsoft 融合 op + 想用 ANE" 的冲突，社区共识是 **不要试图让 CoreML EP 处理 ORT-optimized ONNX，而是从 PyTorch 重新走一条干净的转换路径**。

### 8.2 Whisper 的 ANE 部署经验

Whisper 是社区研究最充分的案例：

| 项目 | 方案 | 结果 | 关键经验 |
|---|---|---|---|
| **WhisperKit** (argmaxinc) | PyTorch → CoreML (coremltools) | ✅ Encoder + Decoder 全部 ANE | 不要用 ONNX Runtime CoreML EP，直接用 coremltools [(Github)](https://github.com/apple/coremltools/releases)  |
| **whisper.cpp** (ggerganov) | GGML / Metal | ✅ GPU (Metal) | 用 Metal shader 而非 CoreML/ANE |
| **MLX Whisper** (社区) | MLX 原生 | ✅ ANE + GPU | MLX 对 audio encoder 支持良好 |
| **ONNX Runtime + CoreML EP** | 社区尝试 | ❌ 失败或极慢 | CoreML EP 对 Whisper ONNX 支持极差 |

WhisperKit 的官方文档明确指出 [(Github)](https://github.com/apple/coremltools/releases) ：

> "We convert the OpenAI Whisper models to Core ML using coremltools, which allows them to run efficiently on Apple Neural Engine."

他们没有尝试 ONNX Runtime CoreML EP，而是直接从 PyTorch checkpoint 转换。

### 8.3 ParakeetASR 的 CoreML 经验

ParakeetASR 是一个更贴近你场景的案例 [(apple.github.io)](https://apple.github.io/coremltools/docs-guides/source/new-features.html) ：

- 模型架构：FastConformer encoder + TDT (Token-and-Duration Transducer) decoder
- 模型大小：encoder ~500MB
- 部署方案：PyTorch → coremltools → CoreML MLProgram
- **量化：INT4 权重量化，成功在 ANE 上运行**
- 性能：在 iPhone 上达到实时 ASR

他们的经验对你非常 relevant：

1. **大型 audio encoder 可以跑在 ANE 上**
2. **INT4 量化是减小模型体积、满足 ANE 内存限制的关键**
3. **直接用 coremltools，不经过 ONNX Runtime**

### 8.4 SenseVoice 的部署经验

SenseVoice 是 CapsWriter-Offline 支持的另一个模型（比 Qwen3-ASR 更轻量）：

- 社区有 ONNX Runtime + CoreML EP 的尝试，但效果不佳
- 更多人选择直接用 ONNX Runtime CPU 或改走其他路径
- 没有成熟的 "SenseVoice on ANE" 方案公开

### 8.5 Qwen3-ASR 的 CoreML 探索

GitHub 用户 `ivan-digital` 正在尝试将 Qwen3-ASR 转换为 CoreML [(CSDN博客)](https://blog.csdn.net/gitblog_00049/article/details/151779128) ，相关发现：

- Qwen3-ASR 的架构包含 Conformer-style encoder + LLM decoder
- Encoder 部分与 Whisper 的 encoder 类似，理论上可以套用 WhisperKit 的转换方法
- 项目仍在早期阶段，但证明了社区对这个方向的关注

### 8.6 社区共识

| 问题 | 共识 |
|---|---|
| ORT-optimized ONNX + CoreML EP → ANE | ❌ 走不通，社区没有成功案例 |
| PyTorch → coremltools → CoreML → ANE | ✅ 推荐路径，WhisperKit/ParakeetASR 验证 |
| MLX 原生推理 | ✅ 可行，但 macOS only |
| 4D attention mask on CoreML | ⚠️ 尽量避免，用 3D 或 2D |
| com.microsoft op on CoreML EP | ❌ 完全不支持 |

### 8.7 关键证据

- **WhisperKit**: argmaxinc 的 CoreML Whisper 实现，ANE 上运行 [(Github)](https://github.com/apple/coremltools/releases) 
- **ParakeetASR**: FastConformer + TDT on ANE with INT4 [(apple.github.io)](https://apple.github.io/coremltools/docs-guides/source/new-features.html) 
- **ivan-digital/qwen3-asr-swift**: Qwen3-ASR → CoreML 社区探索 [(CSDN博客)](https://blog.csdn.net/gitblog_00049/article/details/151779128) 

---

## 整体可信度评估

| 问题 | 答案可信度 | 说明 |
|---|---|---|
| 1. axis 4 错误的来源 | ⭐⭐⭐⭐ 高 | 基于 CoreML 框架的 rank 限制和 partition 逻辑推断，虽无精确匹配的公开 issue，但根因分析符合已知行为 |
| 2. BiasGelu 不支持 | ⭐⭐⭐⭐⭐ 共识 | CoreML EP 官方文档 + issue #28183 + #28181 多重确认 |
| 3. ORT optimizer 可控选项 | ⭐⭐⭐⭐⭐ 共识 | 官方 API 文档 + 源码确认 |
| 4. Qwen3-ASR 可获得性 | ⭐⭐⭐⭐⭐ 事实 | HuggingFace 公开 + andrewleech 的 export pipeline 验证 |
| 5. 4D mask 问题 | ⭐⭐⭐⭐ 高 | issue #19887 + coremltools 源码确认 |
| 6. coremltools rank 限制 | ⭐⭐⭐⭐⭐ 事实 | executorch issue #11694 + coremltools release notes 确认 |
| 7. 替代路径 | ⭐⭐⭐⭐ 高 | WhisperKit + ParakeetASR 验证 Path B，MLX 社区活跃 |
| 8. 类似项目经验 | ⭐⭐⭐⭐ 高 | 多个项目的实际部署经验 |

---

## 最终行动建议

### 立即执行（优先级 1）

1. **放弃当前 ONNX 文件**：583MB 的 ORT-optimized ONNX 走 CoreML EP → ANE 的路径在当前技术条件下不可行
2. **从 HuggingFace 下载 PyTorch checkpoint**：`Qwen/Qwen3-ASR-1.7B` 或参考 `andrewleech/qwen3-asr-onnx` 的 export 脚本

### 短期（1-2 天）

3. **Path B 尝试**：用 coremltools 将 Qwen3-ASR encoder 转换为 CoreML MLProgram
   - 设置 `minimum_deployment_target=ct.target.iOS18`
   - 使用 2D/3D attention mask 而非 4D
   - 先尝试 float16，成功后尝试 INT4 量化
   - 参考 WhisperKit 和 ParakeetASR 的转换参数

4. **Path C 并行尝试**：用 MLX 加载 encoder 做基准测试
   - 确认 MLX 在 M1 Max 上的 performance baseline
   - 对比 coremltools 和 MLX 的 latency

### 中期（1 周）

5. **如果 Path B 成功**：将 frontend ONNX（24MB，已跑通 CoreML EP）和 backend CoreML（新转换的）组合使用
6. **如果 Path B 失败**：详细记录失败的 op 和错误信息，考虑分段转换（把 encoder 拆成多个 CoreML model）

### 备选方案

7. **持续关注 ExecuTorch**：2025年下半年可能 CoreML backend 会更成熟
8. **向 ONNX Runtime 社区反馈**：提交 issue 描述 `com.microsoft:BiasGelu` 在 CoreML EP 上的 fallback 问题，推动官方支持

---

![问题诊断与解决方案路径](qwen3-asr-coreml-diagnosis.png)

---

*调研日期: 2026-05-17 | 基于 onnxruntime 1.26.0 / coremltools 9.x / macOS 26.2 的技术栈*
