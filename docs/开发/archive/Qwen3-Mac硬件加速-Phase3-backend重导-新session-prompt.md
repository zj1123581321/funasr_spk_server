# 新 session 启动 prompt — Qwen3-ASR Mac 硬件加速 Phase 3 (backend → ANE)

> 日期: 2026-05-17
> 上一阶段: Phase 1+2 工程化完成 (commit 33c4cd3 之前) + Phase 3 准备调研 (commit 608a078 + 1c7c17c)
> 本阶段目标: 让 backend ONNX (583MB) 也走 ANE, 进一步压 N=2 wall (估算 -10~15%)
> **关键决策**: 主路径 = **Path B (PyTorch → coremltools → .mlpackage, 绕开 ONNX Runtime)**, Path A (重 export unfused ONNX → CoreML EP) 作 fallback。依据见 §1.4。
> 工作方式: PoC 性质, 先验证后端可加载 + parity, 再决定工程化

## 0. 一句话背景

Phase 1+2 工程化后, ASR encoder **backend ONNX 仍占 N=1 1spk wall 的 60% (94.8s)**, 是头号 CPU 瓶颈, N=2 时放大 1.75x (165.9s) 卡死 CPU。

PoC 阶段尝试让 backend 走 CoreML 失败, 报错 `axis 4 not in [-4,3]` (`spikes/qwen3_mac_hw_accel/coreml_asr_encoder.md:37`)。

上 session 末做了 (a) 内部调研, (b) Gemini + Kimi 两份 deep research, 综合结论:
- **业界没有 ORT-optimized ONNX → CoreML EP → ANE 跑通大型 transformer encoder 的成功案例**
- **WhisperKit / ParakeetASR / ivan-digital/qwen3-asr-swift 都走 PyTorch → coremltools → .mlpackage** (Path B)
- 当前 583MB ONNX 含 25 个 `com.microsoft:BiasGelu` 融合 op, CoreML EP 无 OpBuilder → fallback CPU → partition 边界 shape inference 出错 (axis 4 错误根因)

**本 session 把 Path B 跑通**。

---

## 1. 调研发现汇总 (上 session 已完成, 直接读, 不要重做)

### 1.1 backend ONNX 当前状态
- 文件: `models/qwen3_diarize/Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.onnx` (583 MB)
- **producer**: `onnxruntime.transformers 1.23.2` — 被 ORT optimizer 跑过, 不是原始 torch.onnx.export 输出
- opset: 19, 含 `com.microsoft` (BiasGelu × 25) / `com.microsoft.nchwc` / `org.pytorch.aten` 域
- inputs: `hidden_states [batch, time, 1024]` + `attention_mask [batch, 1, time, time]` (4D mask, dynamic time)
- outputs: `last_hidden_state [batch, time, 2048]`
- 全部 `axis` / `axes` 属性扫一遍, 没有任何 = 4 — 错误来自 partition 边界 shape inference, **不是模型 op 自身**

### 1.2 axis 4 错误根因 (deep research 确认)
- **Kimi 引 issue [#28183](https://github.com/microsoft/onnxruntime/issues/28183)** (2025-11): `com.microsoft:QuickGelu` 在 CoreML EP 不支持, 模式跟我们 BiasGelu 一致, 未解决
- **Kimi 引 issue [#28181](https://github.com/microsoft/onnxruntime/issues/28181)**: fallback 级联导致 "has unsupported inputs" 错误
- **Kimi 引 issue [#19887](https://github.com/microsoft/onnxruntime/issues/19887)** (2024): 4D attention mask 让 CoreML EP 几乎完全 fallback CPU
- **executorch issue #11694**: "Core ML only supports tensors with rank <= 5"
- 根因综合: com.microsoft op 无 OpBuilder → fallback → 子图边界 layout transformation 把 4D 升维 5D+ → CoreML 5D 硬上限校验失败

### 1.3 vendor 含完整 PyTorch source!
`src/core/vendor/qwen_asr_gguf/export/qwen3_asr_custom/`:
- `modeling_qwen3_asr_onnx.py` (126 行):
  - `Qwen3ASRFrontendAtomicOnnx` (line 7) — atomic chunk frontend, 已 export 为 frontend ONNX
  - `Qwen3ASRAudioAttentionOnnx` (line 50) — **DML 友好版 attention** (用 `unflatten`/`transpose`, **additive masking 而不是 masked_fill**)
  - **`Qwen3ASRBackendOnnx` (line 87, 全部 30 行)** — backend module, 复用 `audio_tower.layers/ln_post/proj1/proj2`
  - `Qwen3ASREncoderFullOnnx` (line 117) — frontend + backend 串
- `modeling_qwen3_asr.py` (line 603 `Qwen3ASRAudioEncoder`) — 加载 HF weights 的源
- `configuration_qwen3_asr.py` — model_type 定义
- `processing_qwen3_asr.py` — 预处理

**意味着不用从零写模型**, 直接用现成 class + HF weights 跑 `torch.onnx.export` 或 `coremltools.convert`。

### 1.4 路径优先级 (deep research 综合后修订)

| 优先级 | 路径 | 工作量 | 风险 | 备注 |
|---|---|---|---|---|
| **🥇 主路径** | **B. PyTorch → coremltools.convert → .mlpackage** | **2-3 天** | **中** | **绕开 ONNX Runtime**, WhisperKit/ParakeetASR/ivan-digital 验证成功; 引入 coremltools runtime, transcriber 加 mlpackage 调用路径 |
| 🥈 Fallback | A. 重 export unfused ONNX → CoreML EP | 1-3 天 | 中-高 | 保持现有 ORT 架构 (frontend ANE 路径无改动), 但 Kimi 警告 "即使解决 com.microsoft, ANE 调度仍可能黑盒" (issue #19887) |
| ❌ 不做 | C. 找上游 unfused ONNX | — | — | `andrewleech/qwen3-asr-onnx` 是 CapsWriter ONNX 的实际来源, 默认就跑 `optimize_graphs.py` 引入融合; 跳过这个 step 等于走 Path A |
| ❌ 不做 | D. 反向 un-fuse 现有 ONNX | 1-2 周 | 极高 | 两份 research 都强烈反对 (Gemini: "不可逆", Kimi: "无现成工具") |
| ❌ 不做 | E. MLX 重写 | — | — | 跟 llama.cpp Metal 抢同一 GPU dispatch queue (Phase 2 反例); 且 Ivan 文章自己也确认 "MLX single dispatch queue per process" |

**为什么 Path B 比 Path A 强**:
1. **真实成功案例**: WhisperKit (Whisper encoder+decoder), ParakeetASR (FastConformer + TDT, INT4 ANE), Ivan 的 qwen3-asr-swift (同模型, 早期阶段) — 都是 Path B
2. **绕开 ORT partition 黑盒**: Path A 即使解决 com.microsoft, ANE 调度由 CoreML EP 内部决定, 实际 ANE 利用率不可控 (issue #19887 印证)
3. **量化更好**: coremltools 原生支持 INT8/INT4 量化 + palettization, 工程链成熟
4. **配套工程改动可控**: 加 `coremltools` runtime ~50-100 行 Python, 比 ONNX Runtime CoreML EP 调试更确定

**Path B 的成本**:
- 引入 coremltools 依赖 (Python 包 ~200MB, 仅 dev 时需要)
- transcriber 加 mlpackage 加载分支 (跟现有 ONNX 路径并存, frontend 仍走 ONNX)
- backend inference 接口从 `ort.InferenceSession.run()` 改成 CoreML `MLModel.prediction()` 调用 (Python 通过 `coremltools.models.MLModel` 或 `ct.MLModel`)

### 1.5 后端选型 — ANE, 不是 GPU (已锁定)

**结论**: backend 走 **CoreML ANE** (units = `CPUAndNeuralEngine`), 跟 Phase 2 frontend 同样配置。**不要走 GPU/Metal**, 也不要做 MLX 重写。

#### 决策依据
1. **Ivan 文章**: encoder 类(non-autoregressive)→ ANE; Qwen3 用 GPU 是 0.6B autoregressive LLM 翻译, 跟我们 encoder 不同; ANE 跟 GPU 物理隔离, 共享 400GB/s 内存, 同时跑无 contention
2. **Phase 2 PoC**: ALL units(含 GPU)让 llm_decode +200%(37→110s), ANE units 完美解耦
3. **Path B 主路径同步配置**: `coremltools.convert(..., compute_units=ct.ComputeUnit.CPU_AND_NE)`, 不要用 `ALL`

#### Path A/B 共有的 ANE 调度风险 (Kimi 警告)
- issue #19887: 即使所有 op 都"被支持", 实际 ANE 利用率可能极低, 大部分在 CPU 上
- **Step 3 必须用 Instruments / powermetrics 测真实 ANE 占用率**, 不能只看"加载成功"

### 1.6 硬警告清单 (Phase 3 必须验证)

#### 警告 1: 内存放大 (最高优先级)
- Ivan 文章: Parakeet INT4 332MB → CoreML 加载 1,677 MB (5x); INT8 → 2.2x; **MLX 1:1**
- 我们 backend 583MB FP16, 推算 CoreML 加载 1-2 GB (放大 1.7-3.4x); N=2 worker = 2-4 GB ANE 占用, M1 Max 64GB OK 但需测
- Kimi: 用 INT8 量化可压到 ~150MB on-disk, 加载 ~300-400MB, **强烈推荐**
- **Step 3 必加 RSS profiling**: 单 worker baseline vs Path B mlpackage 加载后 ΔRSS

#### 警告 2: dynamic time 维度 (Phase 2 dynamic 跑通不代表 backend 也行)
- Phase 2 frontend 是 atomic chunk (固定 13 帧输出), backend 接受变长 `time` 是真 dynamic
- Ivan 文章 + Gemini 都强烈推荐 **静态 chunk + padding** (FluidAudio 14.96s 固定切片 1.6x 提速)
- Kimi: issue #19887 报告 dynamic mask 让 CoreML 几乎全 fallback CPU
- **Path B export 时**: 用 fixed `time` 或 `ct.RangeDim(lower_bound=N, upper_bound=M)` + `EnumeratedShapes`
- 推荐做法: backend 输入按 chunk 拼接, 例如固定 `time=130` (10s × 13 frames/s) 或 `time=390` (30s), 不够 padding
- 这意味着 inference 时 `_run_backend()` 要改 — 把变长 hidden_states padding 到 fixed time

#### 警告 3: attention_mask 4D shape 是 CoreML 痛点
- Kimi 引 issue #19887: 4D mask `[B, 1, T, T]` 让 CoreML EP 几乎全 CPU fallback
- coremltools `scaled_dot_product_attention` (iOS 18+) 推荐 2D mask 或 key padding index `[B, T]` (int64)
- **export backend 时改 forward 函数**: 内部用 `additive bias` 直接合并到 attention scores, 接口暴露 2D mask
- 现有 `Qwen3ASRBackendOnnx.forward` 接 4D mask, 可能要写 wrapper 转换

#### 警告 4: ANE 64-byte alignment
- hidden=1024/2048 ✅ (1024/32=32, FP16 64-byte 对齐)
- attention head dim 看 modeling 源码 — Phase 2 frontend export 工作意味着 head_dim 也对齐
- 若不对齐: ANE padding 32-64x, RAM 爆炸
- **Step 2 export 前**: 看 `modeling_qwen3_asr.py:603` Qwen3ASRAudioEncoder 的 `head_dim` 配置

#### 警告 5: 不同 M 代 ANE 行为差异
- Ivan 文章: "Neural Engine architecture changed between M2, M3, M4 — numbers may vary"
- Parakeet 在 iPhone 17 Pro 报 "Unknown aneSubType", fallback CPU+GPU 内存 3x
- 我们生产 + dev 都是 M1 Max, **部署文档必须标注"仅 M1 Max 验证, M2/M3/M4 未测"**

---

## 2. 必读 (按顺序)

1. `docs/ref/onnx_ane/gemini.md` — Gemini deep research (8 问题答案, 含 FluidAudio 经验 + 工作量评估)
2. `docs/ref/onnx_ane/kimi.md` — Kimi deep research (8 问题, 含 issue 链 + WhisperKit/ParakeetASR/Ivan 案例 + Path B 完整代码模板)
3. `spikes/qwen3_mac_hw_accel/SUMMARY.md` — Phase 1+2 数据 + Phase 3 路线
4. `spikes/qwen3_mac_hw_accel/coreml_asr_encoder.md` — Phase 2 详细 + axis 4 错误原始报告
5. **`src/core/vendor/qwen_asr_gguf/export/qwen3_asr_custom/modeling_qwen3_asr_onnx.py`** ← 全文 126 行, 核心 export 源码
6. `src/core/vendor/qwen_asr_gguf/export/qwen3_asr_custom/modeling_qwen3_asr.py` 第 603-779 行 `Qwen3ASRAudioEncoder` (weights 来源)
7. `src/core/vendor/qwen_asr_gguf/inference/encoder.py` 第 122-285 行 — frontend/backend 推理逻辑
8. `CLAUDE.md` — 测试约定 + ASR 引擎章节
9. (Path B 实施时) [`andrewleech/qwen3-asr-onnx`](https://github.com/andrewleech/qwen3-asr-onnx) — Kimi 找到的关键 repo, CapsWriter ONNX 实际来源, 含 PyTorch export + optimize pipeline; 我们参考它的**前半段** (export), 跳过 `optimize_graphs.py` 融合
10. (Path B 实施时) [WhisperKit (argmaxinc)](https://github.com/argmaxinc/WhisperKit) + [ParakeetASR](https://github.com/argmaxinc/ParakeetASR) — coremltools 转换参考

---

## 3. 任务流程 (Path B 主路径)

### Step 0 — 启动验证 (15 分钟)
1. 确认起点: `git log --oneline -10` 应看到最近是 `1c7c17c docs(prompt): Phase 3 prompt 锁定后端选型 ANE`
2. 检查 venv:
   ```bash
   venv/bin/python -c 'import torch, onnx, onnxruntime; print(torch.__version__, onnx.__version__, onnxruntime.__version__)'
   ```
   (上 session 已装 onnx 1.21.0)
3. **装 Path B 依赖** (新):
   ```bash
   venv/bin/python -m pip install transformers coremltools
   ```
   注意 coremltools ~200MB, 仅 dev 时需要; production 不需要(直接加载 .mlpackage)
4. 检查 HF / ModelScope 是否能拉:
   ```bash
   venv/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download.__module__)"
   ```
5. 决定 PyTorch checkpoint 来源:
   - 选项 A: HuggingFace `Qwen/Qwen3-ASR-1.7B` (若可访问)
   - 选项 B: ModelScope 阿里魔搭社区镜像 (国内推荐)
   - 选项 C: 看 vendor 是否在 prod 已有 PyTorch checkpoint (`ls ~/Production/qwen_asr_server/models/` 找 .safetensors / .bin)

### Step 1 — 复现 axis 4 错误 (可选, 30 分钟, 验证根因诊断)
1. 写最小复现脚本 `spikes/qwen3_mac_hw_accel/repro_backend_coreml.py`:
   ```python
   import onnxruntime as ort
   sess_opts = ort.SessionOptions()
   providers = [
       ('CoreMLExecutionProvider', {'ModelFormat': 'MLProgram', 'MLComputeUnits': 'CPUAndNeuralEngine'}),
       'CPUExecutionProvider',
   ]
   sess = ort.InferenceSession("models/qwen3_diarize/Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.onnx",
                                sess_options=sess_opts, providers=providers)
   ```
2. 跑, 把 stderr 存到 `spikes/qwen3_mac_hw_accel/coreml_backend_repro.log`, 看错误是不是 `axis 4`
3. 如果是 → 根因诊断确认, 继续 Step 2
4. **不在这里 deep-debug**, deep research 已经定锤, 这一步只是 sanity check
5. **若想跳过**: 直接进 Step 2

### Step 2 — Path B 核心: PyTorch → coremltools 转换 (1 天)

#### Step 2.1 — 加载 PyTorch checkpoint
1. 写脚本 `spikes/qwen3_mac_hw_accel/export_backend_coreml.py`:
   ```python
   from transformers import AutoModel
   model = AutoModel.from_pretrained(
       "Qwen/Qwen3-ASR-1.7B",   # 或 ModelScope 路径
       trust_remote_code=True,
   )
   # 拿到 audio_tower
   audio_tower = model.audio_tower  # 或 model.encoder.audio_tower, 看 modeling 源码
   ```
2. 如果 HF 拉不到, 看 `src/core/vendor/qwen_asr_gguf/export/qwen3_asr_custom/configuration_qwen3_asr.py` 的 `model_type='qwen3_asr_audio_encoder'`, 手动注册

#### Step 2.2 — 构造 backend module + 改 forward 签名 (针对 4D mask 警告)
1. 用现成 `Qwen3ASRBackendOnnx(audio_tower)` 或写 wrapper 接受 2D mask:
   ```python
   from src.core.vendor.qwen_asr_gguf.export.qwen3_asr_custom.modeling_qwen3_asr_onnx import Qwen3ASRBackendOnnx
   import torch.nn as nn

   class BackendCoreMLWrapper(nn.Module):
       def __init__(self, audio_tower):
           super().__init__()
           self.backend = Qwen3ASRBackendOnnx(audio_tower)

       def forward(self, hidden_states, key_padding_mask):
           # key_padding_mask: [B, T] int32 (1 = real, 0 = pad)
           # 内部转 4D additive mask
           b, t = key_padding_mask.shape
           additive = (1.0 - key_padding_mask.float()).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
           additive = additive * -1e4  # 加性 mask
           # broadcast 到 [B, 1, T, T]
           additive = additive.expand(b, 1, t, t)
           return self.backend(hidden_states, attention_mask=additive)
   ```
2. `eval()` + `torch.jit.trace`:
   ```python
   wrapper = BackendCoreMLWrapper(audio_tower).eval()
   example_h = torch.randn(1, 390, 1024)   # batch=1, time=390 (~30s × 13fps), feature=1024
   example_mask = torch.ones(1, 390, dtype=torch.int32)
   traced = torch.jit.trace(wrapper, (example_h, example_mask))
   ```

#### Step 2.3 — coremltools 转换 (注意 static shape!)
1. 转 mlpackage:
   ```python
   import coremltools as ct

   mlmodel = ct.convert(
       traced,
       inputs=[
           ct.TensorType(name="hidden_states", shape=(1, 390, 1024), dtype=ct.utils._utils._np.float16),
           ct.TensorType(name="key_padding_mask", shape=(1, 390), dtype=ct.utils._utils._np.int32),
       ],
       outputs=[ct.TensorType(name="last_hidden_state", dtype=ct.utils._utils._np.float16)],
       minimum_deployment_target=ct.target.macOS15,    # iOS 18+ / macOS 15+
       compute_units=ct.ComputeUnit.CPU_AND_NE,         # ANE only, 不要 ALL
       compute_precision=ct.precision.FLOAT16,
       convert_to="mlprogram",                          # 不要 NeuralNetwork
   )
   mlmodel.save("models/qwen3_diarize/Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.mlpackage")
   ```
2. 关键参数:
   - **static shape (1, 390, 1024)** — 警告 2, 不要 RangeDim 除非确认 work
   - **CPU_AND_NE** — 不要 ALL, 警告 1.5 反例
   - **FLOAT16** — 警告 1, 比 INT4 palettization 更稳
3. 若 fail (op 不支持 / shape inference 错误): 看 stderr, **iterative** 修 wrapper.forward, 不要一次改太多

#### Step 2.4 — (可选) INT8 量化压内存
1. 警告 1 内存放大风险, INT8 是最优解 (Ivan 文章 INT8 比 INT4 在 ANE 上快 3.3x)
2. 用 coremltools `compress_weights`:
   ```python
   from coremltools.optimize.coreml import linear_quantize_weights, OpLinearQuantizerConfig, OptimizationConfig
   config = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
   mlmodel_int8 = linear_quantize_weights(mlmodel, config=config)
   mlmodel_int8.save("models/qwen3_diarize/Qwen3-ASR-1.7B/qwen3_asr_encoder_backend_int8.mlpackage")
   ```
3. 若 INT8 引入 WER 退化 > 1%, 回 FP16

### Step 3 — 加载验证 + ANE 占用率测量 (半天)
1. Python 加载:
   ```python
   import coremltools as ct
   mlmodel = ct.models.MLModel("models/qwen3_diarize/Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.mlpackage",
                                compute_units=ct.ComputeUnit.CPU_AND_NE)
   # 跑一次
   import numpy as np
   h = np.random.randn(1, 390, 1024).astype(np.float16)
   m = np.ones((1, 390), dtype=np.int32)
   out = mlmodel.predict({"hidden_states": h, "key_padding_mask": m})
   ```
2. **RSS profiling** (警告 1):
   ```bash
   # Phase 2 baseline: 单 worker 加载 frontend ANE + backend CPU 的 RSS
   # Phase 3: 单 worker 加载 frontend ANE + backend mlpackage 的 RSS
   # Δ < 2GB OK, > 4GB 警惕
   ```
3. **ANE 占用率测量** (Kimi 警告 #19887):
   ```bash
   sudo powermetrics --samplers ane_power -i 1000 -n 30 &
   venv/bin/python -c "..."  # 跑 100 次 backend.predict
   ```
   - ANE peak 应 > 100mW 持续, mean > 50mW (Phase 2 frontend ANE mean 50mW peak 336mW)
   - 若 ANE 全程 0mW: 说明被静默 fallback CPU, **Path B 失败**, 跳 Step 5 Path A fallback
4. **CPU 残留** (用 `top` 看进程 CPU%): 加载 backend 后跑 inference, CPU% 应该比 Phase 2 backend CPU 低显著

### Step 4 — Parity 测试 (半天)
1. 写脚本 `spikes/qwen3_mac_hw_accel/parity_backend_onnx_vs_coreml.py`:
   - 加载现有 ONNX backend (CPU) + 新 mlpackage backend (ANE)
   - 用同一个真音频跑 frontend (frontend ONNX 已有), 拿到 hidden_states
   - 两个 backend 各跑一次, 比较 `last_hidden_state` 数值
   - 阈值: 余弦相似度 > 0.999, max abs diff < 1e-2 (FP16 数值噪声)
   - 用 `tests/fixtures/audio/podcast_2speakers_60s.wav`
2. 若 parity fail: 检查 wrapper.forward 是否引入数值偏差(mask padding 值, broadcast 顺序)

### Step 5 — 工程化集成 (1 天)

#### Step 5.1 — encoder.py 加 mlpackage 分支
1. 改 `src/core/vendor/qwen_asr_gguf/inference/encoder.py:122` `QwenAudioEncoder.__init__`:
   - 加新 provider 名: `COREML_ANE_FULL` (frontend ONNX ANE + backend mlpackage ANE)
   - 加载 backend 走 `ct.models.MLModel(backend_mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_NE)`
   - frontend 仍走 ONNX (Phase 2 已工作)
2. 改 `_run_backend()` (line 243):
   - 把 `self.sess_be.run(...)` 改成 `self.sess_be_mlmodel.predict({...})`
   - 注意输出格式不同 (dict vs list)
   - 注意 padding hidden_states 到 fixed time = 390

#### Step 5.2 — config knob + escape hatch
1. 改 `src/core/config.py` `qwen3.asr_encoder_provider` 字段说明:
   - `auto`: 平台感知 (macOS → COREML_ANE_FULL)
   - `cpu`: 强制 ONNX CPU (Phase 0 退路)
   - `coreml_ane_fe`: Phase 2 frontend only ANE (现状)
   - `coreml_ane_full`: Phase 3 frontend + backend ANE (新)
2. env: `FUNASR_QWEN3_ASR_ENCODER_PROVIDER=coreml_ane_full`

#### Step 5.3 — build_engine_config 默认值
- 改 `src/core/qwen3/asr.py:60` `build_engine_config`:
  - macOS auto → `COREML_ANE_FULL` (Path B 跑通后)
  - 若 Path B 部分跑通 (例如 INT8 fail FP16 OK), 调成 FP16 默认

#### Step 5.4 — download script
- 改 `scripts/download_qwen3_models.sh` 加 backend.mlpackage 下载源 / prod 镜像源
- 注意 .mlpackage 是目录不是文件, 用 `cp -R`

### Step 6 — TDD 测试 (1 天)

#### unit test
1. `tests/unit/test_qwen3_encoder_coreml_ane_full.py`:
   - mock CoreML loaded model
   - mock `ct.models.MLModel.predict()`
   - 测试 provider=COREML_ANE_FULL 时 sess_fe 是 ONNX CoreML, sess_be 是 mlpackage
   - 测试 platform != darwin 时 fallback ONNX CPU
   - 测试 mlpackage 文件不存在时 fallback frontend ANE + backend CPU

2. `tests/unit/test_qwen3_asr_default_provider.py` 加 COREML_ANE_FULL 平台默认 case

#### integration parity test
- `tests/integration/test_qwen3_backend_coreml_ane_parity.py`:
  - 跟 Phase 2 frontend parity 测试同 pattern
  - `onnx_provider="CPU"` vs `onnx_provider="COREML_ANE_FULL"` 比 ASR text + segments + speaker labels
  - 差异 < 1% chars-level diff

#### 长音频实测
- `FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/ -q` 全过
- 跑 `spikes/qwen3_mac_hw_accel/run_baseline.py --mode n2 --tag verify_phase3 --num-threads 4 --onnx-provider COREML_ANE_FULL`, 期望 wall < 380s (相比 Phase 2 的 415s 再 -8%+)

### Step 7 — 收尾 (半天)
1. 更新 `docs/部署.md`: 加 "Phase 3 backend ANE" 段, 标注 "仅 M1 Max 验证"
2. 写 spike 报告 `spikes/qwen3_mac_hw_accel/phase3_backend_mlpackage.md`: 记录路径 B / 警告处理 / 实测数据
3. 把本 prompt 移到 `docs/开发/archive/`
4. 提 PR (跑全部 unit + integration)

---

## Path A fallback (若 Path B Step 2-3 fail)

**触发条件**:
- Step 2.3 coremltools.convert 反复失败 (op 不支持 / shape inference 错误) 且 wrapper 改 3 次未通过
- 或 Step 3 ANE 占用率全程 0mW (静默 fallback CPU, 无收益)

**Path A 步骤** (跟原计划相同, 但跳过"调研" 直接执行):
1. clone `andrewleech/qwen3-asr-onnx`
2. 跑它的 export 脚本但**跳过 `optimize_graphs.py`**, 拿干净 ONNX
3. 替换 `qwen3_asr_encoder_backend.onnx`, 用 Phase 2 同样的 `COREML_ANE_FE` provider 名扩展(改成 `COREML_ANE_FE_BE`)
4. 若仍 axis 4 错误: 改 backend forward 用 2D mask + RangeDim (跟 Path B 的 wrapper 同样思路)
5. 工程化收尾同 Step 5-7

**Path A 风险确认点**:
- ANE 真实占用率 (跟 Step 3 同样测)
- 若 ANE 仍 0mW → Path A 也失败, 整个 Phase 3 失败 → 接受 Phase 2 现状 -16.1%, 不强行 ship

---

## 4. TDD 流程铁律 (跟 Phase 1+2 一致)
- 红 → 绿 → commit 最小单位
- 每个 commit 测试通过, parity 通过才算 Phase 完成
- commit message 引用具体测试名

## 5. 工作方式约束 (跟 Phase 1+2 一致)
- 中文回应
- venv: `venv/bin/python`
- 环境: `unset TMPDIR; export TMPDIR=/tmp; export DYLD_LIBRARY_PATH="$PWD/src/core/vendor/qwen_asr_gguf/inference/bin"`
- 长任务 (>1min): `Bash run_in_background: true`, 等通知
- 用 ctx_batch_execute / ctx_execute_file 处理大输出
- 不提交 audio/模型/tmp_long_audio/*.log/powermetrics.log

## 6. 反模式 (不做)

- ❌ **跳过 Step 3 ANE 占用率测量** — 这是 Path B 真假的唯一证据, 加载成功 ≠ ANE 跑通 (Kimi 警告 issue #19887)
- ❌ **跳过 RSS profiling** — 1.7B 模型 ANE 加载 OOM 概率不低, N=2 必须先单 worker measure
- ❌ **reverse-engineer 现有 583MB ONNX un-fuse** — 两份 deep research 都反对, 工程量 1-2 周不值
- ❌ **重新讨论后端选型 (ANE vs GPU)** — §1.5 已锁定, 证据完备, 改方向需要新证据反驳
- ❌ **重新讨论 Path B vs A 主路径** — §1.4 已锁定, deep research 数据完备; 只在 Path B 实际 fail 时切 fallback
- ❌ **试 MLX backend 重写** — 跟 llama.cpp 抢 GPU dispatch queue, 复杂度爆炸, Ivan 自己也说 single dispatch queue per process
- ❌ **修改 frontend ANE 路径** — Phase 2 已工作, 不要碰; frontend ONNX + backend mlpackage 共存, frontend 不动
- ❌ **CoreML units = ALL** — Phase 2 反例 + Ivan 文章警告, 抢 llama.cpp Metal
- ❌ **INT4 palettization 量化** — Ivan 文章 ANE INT4 比 INT8 慢 3.3x, 用 INT8 linear_symmetric
- ❌ **Phase 3 失败强行 ship** — escape hatch 已有 (`FUNASR_QWEN3_ASR_ENCODER_PROVIDER=coreml_ane_fe`), 失败回 Phase 2

## 7. 完成标准

### 必达 (Path B 跑通的最低标志)
- [ ] Step 2: `.mlpackage` 生成成功 (FP16 或 INT8)
- [ ] Step 3 警告 1: backend mlpackage 加载 ΔRSS profiling 完成, N=2 内存预算 < 8 GB
- [ ] Step 3 警告 (ANE 占用率): powermetrics 实测 ANE peak > 100mW 持续, mean > 50mW
- [ ] Step 4: ASR text parity, frontend ONNX + backend mlpackage 跟 全 CPU baseline diff < 1%

### 工程化达标
- [ ] Step 5: `COREML_ANE_FULL` provider 名扩展 + escape hatch 字段
- [ ] Step 6 unit test: 至少 4 个新测试 (provider 路由 + platform fallback + missing file fallback + mlpackage shape mismatch)
- [ ] Step 6 integration parity: 全过
- [ ] Step 6 长音频 N=2 wall < 380s (比 Phase 2 的 415s 再 -8%+)
- [ ] Step 7 部署文档加 "Phase 3, 仅 M1 Max 验证"

### Path A fallback 达标 (若主路径 fail)
- [ ] 重 export unfused ONNX → CoreML EP 加载成功 + ANE 实测 > 50mW mean
- [ ] 否则**报告失败原因 + 数据**, 不强行 ship, Phase 2 现状不动

### 全局
- [ ] 所有 unit test 通过 (238 + 新加)
- [ ] 所有 integration test 通过 (FUNASR_RUN_INTEGRATION=1)
- [ ] 跑长音频 verify 实测数据存 `spikes/qwen3_mac_hw_accel/runs/verify_phase3/`

## 8. 开始

1. `git log --oneline -10` 确认起点 (`1c7c17c docs(prompt): Phase 3 prompt 锁定后端选型 ANE`)
2. 读必读 1-9 (尤其 deep research gemini.md + kimi.md)
3. Step 0 venv 检查 + 装 transformers / coremltools
4. **可选**: Step 1 复现 axis 4 错误 (sanity check, 或跳过)
5. **直接进 Step 2.1**: 拉 HF/ModelScope Qwen3-ASR-1.7B checkpoint
6. 不停下来问用户问题, 跑通就 commit, 失败就分析根因
7. **若 Step 2-3 反复 fail (3 次以上)**: 切 Path A fallback, 不要钻牛角尖
8. 完成时输出: 完成的 Step + 关键数据 (ANE mW / RSS / wall) + 下一步建议
