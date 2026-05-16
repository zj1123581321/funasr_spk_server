# 新 session 启动 prompt — Qwen3-ASR Mac 硬件加速 Phase 3 (backend ONNX 重导出)

> 日期: 2026-05-16
> 上一阶段: Phase 1+2 工程化完成 (commit 33c4cd3 之前)
> 本阶段目标: 让 backend ONNX 也能走 CoreML ANE, 进一步压缩 N=2 wall (估算 -10~15%)
> 工作方式: PoC 性质, 先调研 + 复现错误, 再决定是否值得工程化

## 0. 一句话背景

Phase 1+2 工程化后, ASR encoder **backend ONNX 仍占 N=1 1spk wall 的 60% (94.8s)**, 是头号 CPU 瓶颈, 但 N=2 时放大 1.75x (165.9s) 卡死 CPU。

PoC 阶段尝试让 backend 走 CoreML 失败, 报错 `axis 4 not in [-4,3]` (`spikes/qwen3_mac_hw_accel/coreml_asr_encoder.md:37`)。当时判定要"重新 export ONNX 模型, 1-2 周, 数值精度风险高", 留作未来 PoC。

**本 session 是那个未来 PoC。** 上一 session 末做了完整内部调研, 把工作量大幅下修(见下方"内部调研发现")。

---

## 1. 内部调研发现 (上一 session 已完成, 直接读, 不要再做)

### 1.1 backend ONNX 当前状态
- 文件: `models/qwen3_diarize/Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.onnx` (583 MB)
- **producer**: `onnxruntime.transformers 1.23.2` — **关键!** 这是被 ORT transformers 优化过的版本, **不是原始 torch.onnx.export 输出**
- opset: 19 (含 domains: `com.microsoft`, `com.microsoft.nchwc`, `org.pytorch.aten` 等非标准域)
- inputs: `hidden_states ['batch', 'time', 1024]` + `attention_mask ['batch', 1, 'time', 'time']` (4D mask)
- outputs: `last_hidden_state ['batch', 'time', 2048]`
- nodes: 901, 其中 **25 个 `com.microsoft:BiasGelu` 融合 op** (CoreML EP 大概率不支持)

### 1.2 axis 4 错误根因假说 (待验证)
扫了所有 op 的 `axis` / `axes` 属性 + Squeeze/Unsqueeze 的 input axes initializer + Reshape/Transpose 维度, **都没有显式 axis=4**。

所以错误**不是模型自身有 5D op**, 假说:
- (a) CoreML EP partition `com.microsoft:BiasGelu` 等融合 op 时, 在子图边界插入 reshape, 内部 4D 输入被升维到 5D, 引发 `axis 4 of [-4, 3]` 校验失败
- (b) 或 attention_mask 4D `(batch, 1, time, time)` 在 CoreML 某 op 上需要 unsqueeze 到 5D, 同样卡限制

**复现错误是 Step 1 最关键的事**, 拿到完整 stack trace 才能定锤。

### 1.3 vendor 里有完整 PyTorch source! 🎉
`src/core/vendor/qwen_asr_gguf/export/qwen3_asr_custom/modeling_qwen3_asr_onnx.py` 含 4 个 class:
- `Qwen3ASRFrontendAtomicOnnx` (line 7) — chunk frontend, 已 export 为 frontend ONNX
- `Qwen3ASRAudioAttentionOnnx` (line 50) — DML 友好版 attention (用 unflatten/transpose, additive masking)
- **`Qwen3ASRBackendOnnx` (line 87, 全部 30 行)** — backend module, 复用 `audio_tower.layers` + `ln_post` + `proj1/proj2`
- `Qwen3ASREncoderFullOnnx` (line 117) — 串 frontend + backend

`modeling_qwen3_asr.py` 含原 HF Qwen3ASRAudioEncoder (line 603) — 加载 weights 的源 (传给 `Qwen3ASRBackendOnnx(audio_tower)`)。

**意味着: 不需要从零写模型, 直接用现成的 PyTorch class 跑 `torch.onnx.export(Qwen3ASRBackendOnnx, ...)` 就能拿到 unfused ONNX。**

### 1.4 工作量重新评估

| 路径 | 工作量 | 风险 | 备注 |
|---|---|---|---|
| **A. vendor PyTorch source 重 export unfused backend ONNX** | **1-3 天** | 中 | 推荐路径; 写 export 脚本 + 跑 CoreML EP 复现 + parity 验证 |
| B. 找上游 CapsWriter 是否提供 unfused 版 | 0.5-1 天调研 | 低 | 如果有省事; HaujetZhao/CapsWriter-Offline GitHub releases |
| C. 关掉 onnxruntime.transformers BiasGelu 融合 | 1 天试 | 中 | 不确定能否完全避免融合 |
| D. 直接 patch 融合后 ONNX (上一 session 旧估算的"重导 1-2 周") | 1-2 周 | 高 | 不推荐 |

**结论**: 用户原本以为是 D 路径 (1-2 周, 高风险), 实际上 A 路径只要 1-3 天。**这是 Phase 3 值得做的核心理由**。

---

## 2. 必读 (按顺序)

1. `spikes/qwen3_mac_hw_accel/SUMMARY.md` — Phase 1-3 路线 + 实测数据
2. `spikes/qwen3_mac_hw_accel/coreml_asr_encoder.md` — Phase 2 详细 + axis 4 错误描述 + 5 个 variants 取舍
3. `spikes/qwen3_mac_hw_accel/profiling_baseline.md` — N=1/N=2 stage timing baseline
4. **`src/core/vendor/qwen_asr_gguf/export/qwen3_asr_custom/modeling_qwen3_asr_onnx.py`** ← 全文 126 行, **核心**
5. `src/core/vendor/qwen_asr_gguf/export/qwen3_asr_custom/modeling_qwen3_asr.py` 第 603-779 行 `Qwen3ASRAudioEncoder` (weights 来源)
6. `src/core/vendor/qwen_asr_gguf/inference/encoder.py` 第 122-285 行 — frontend/backend 推理逻辑 + COREML_ANE_FE 分支
7. `CLAUDE.md` — 测试约定 + ASR 引擎章节
8. `docs/开发/archive/重构计划-ASR引擎抽象.md` — 不必读, 但若改 transcriber 接口要看

deep research 资料(用户提供):
- `docs/开发/Qwen3-Mac硬件加速-Phase3-deep-research-prompt.md` ← 用户先跑 Gemini/ChatGPT, 把回答 copy 进来

---

## 3. 任务流程

### Step 0 — 启动验证 (15 分钟)
1. 看上一 session 末跑过的 verify 数据: `spikes/qwen3_mac_hw_accel/runs/verify_engineering_phase1_phase2/summary.json` → 4spk wall 415s
2. 起点 commit: `git log --oneline -10` 应看到最近是 `33c4cd3 chore(qwen3): verify Phase 1+2 工程化实测数据`
3. 检查 venv: `venv/bin/python -c 'import torch, onnx, onnxruntime; print(torch.__version__, onnx.__version__, onnxruntime.__version__)'` (上一 session 已装 onnx 1.21.0)
4. 装 transformers (export 时需要): `venv/bin/python -m pip install transformers`

### Step 1 — 复现 axis 4 错误, 拿完整 stack trace (核心!)
1. 写一个最小复现脚本 `spikes/qwen3_mac_hw_accel/repro_backend_coreml.py`:
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
2. **跑这个脚本, 把完整 stderr + stack trace 存到 `spikes/qwen3_mac_hw_accel/coreml_backend_repro.log`**
3. **找具体是哪个 op 触发的错误** — 错误信息里通常有 "in op '...' / node '...'"
4. 看 op 类型: 是 `com.microsoft:BiasGelu` ? 还是 attention_mask 上的 Add ? 还是别的?
5. 这一步定锤"假说 a 还是 b", 决定 Step 2 路径

### Step 2 — 写 backend export 脚本
1. 文件: `spikes/qwen3_mac_hw_accel/export_backend_unfused.py`
2. 加载 weights:
   ```python
   from src.core.vendor.qwen_asr_gguf.export.qwen3_asr_custom.modeling_qwen3_asr import Qwen3ASRAudioEncoder
   from src.core.vendor.qwen_asr_gguf.export.qwen3_asr_custom.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
   # 从 HF 拉原权重 / 或 从现有 ONNX 反解 (有难度) / 或 从 prod 找 PyTorch checkpoint
   ```
3. **关键挑战**: 上游 PyTorch checkpoint 在哪? 选项:
   - HuggingFace `Qwen/Qwen3-ASR-1.7B` (如果公开)
   - ModelScope (国内镜像)
   - CapsWriter-Offline 给的源 → 找 release notes
4. 构造 `Qwen3ASRBackendOnnx(audio_tower)`, 跑 `torch.onnx.export`:
   ```python
   torch.onnx.export(
       backend_module,
       (hidden_states, attention_mask),
       "qwen3_asr_encoder_backend_unfused.onnx",
       input_names=["hidden_states", "attention_mask"],
       output_names=["last_hidden_state"],
       dynamic_axes={
           "hidden_states": {0: "batch", 1: "time"},
           "attention_mask": {0: "batch", 2: "time", 3: "time"},
           "last_hidden_state": {0: "batch", 1: "time"},
       },
       opset_version=17,  # 不要用 19 + ORT 优化
       do_constant_folding=True,
   )
   ```
5. **明确不要**: 不跑 `onnxruntime.transformers.optimizer` 优化, 不引入 com.microsoft 域 op

### Step 3 — 用新 ONNX 跑 CoreML EP 验证
1. 重复 Step 1 脚本, 加载 `qwen3_asr_encoder_backend_unfused.onnx`
2. 期待结果:
   - (a) 加载成功 → 跑 inference → 跟 CPU 结果数值对比, 差异 < 1e-3 算 parity
   - (b) 仍报 axis 4 错误 → 说明问题在模型结构, 走 Step 4
3. 如果是 (a), Phase 3 主体工作完成。Step 5 直接做工程化。

### Step 4 — 如果新 ONNX 仍 fail (假说 b: attention_mask 4D 卡)
1. 检查具体哪个 op 失败 (用 Step 1 同样手法)
2. 可能改动:
   - `Qwen3ASRBackendOnnx.forward` 把 attention_mask 从 4D 改成 3D (broadcast 让 attention 自动扩展)
   - 或者用 additive bias 直接合并到 `attention_weights = q@k.T * scale + mask` (PyTorch source 已经是这种, 但 export 后可能仍是 4D)
3. 写新一版 `Qwen3ASRBackendOnnxV2`, 重 export, 重 CoreML EP 验证
4. 如果改 4 次还不行 → 退回 Step 0 重新评估, 可能放弃

### Step 5 — 工程化 (前提: Step 3-4 成功)
1. 把 unfused backend ONNX 放生产路径
2. 改 `download_qwen3_models.sh` 加 unfused backend 下载源
3. 改 `src/core/vendor/qwen_asr_gguf/inference/encoder.py`:
   - 加 new provider 名: `COREML_ANE_FE_BE` 或叫 `COREML_ANE_FULL`
   - 同时让 frontend + backend 都走 ANE
   - units 仍是 `CPUAndNeuralEngine` (不能 ALL, 会抢 llama.cpp Metal)
4. 改 `src/core/qwen3/asr.py` `build_engine_config` 默认值 + config knob
5. 加 unit test + parity 集成测试 (跟 Phase 2 类似 pattern)
6. 跑长音频 verify, 期待 N=2 4spk wall 从 415s → ~350s (估算)

### Step 6 — 收尾
1. 更新 `docs/部署.md` 加 Phase 3 段落
2. 写新的 spike 报告 `spikes/qwen3_mac_hw_accel/phase3_backend_unfused.md`
3. 把本 prompt 移到 `docs/开发/archive/`
4. 跑全部 unit + integration test, 提 PR

---

## 4. TDD 流程铁律 (跟 Phase 1+2 一致)

- 红 → 绿 → commit 最小单位, 不积累改动
- 每个 commit 测试通过
- Phase 阶段结束跑一次 parity: `FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/`
- commit message 引用具体测试名

## 5. 工作方式约束 (跟 Phase 1+2 一致)

- 中文回应
- venv: `venv/bin/python`
- 环境: `unset TMPDIR; export TMPDIR=/tmp; export DYLD_LIBRARY_PATH="$PWD/src/core/vendor/qwen_asr_gguf/inference/bin"`
- 长任务 (>1min): `Bash run_in_background: true`, 等通知
- 用 ctx_batch_execute / ctx_execute_file 处理大输出
- 不提交 audio/模型/tmp_long_audio/*.log/powermetrics.log (已在 .gitignore)

## 6. 反模式 (不做)

- ❌ 不复现错误就直接重 export — 万一根因不在融合 op, 工作全白做
- ❌ 在没拿到上游 PyTorch checkpoint 之前就承诺 Step 2-5 — checkpoint 找不到是最大风险点
- ❌ 用 `onnxruntime.transformers.optimizer` 跑新 ONNX (会重新引入 com.microsoft 融合, 跟 Phase 3 目的相反)
- ❌ 修改 frontend 路径 — Phase 2 已经验证有效, 不要碰
- ❌ 让 CoreML units = ALL — 会抢 llama.cpp Metal (Phase 2 反例)
- ❌ Phase 3 失败就强行 ship — escape hatch 已经有 (`FUNASR_QWEN3_ASR_ENCODER_PROVIDER=cpu`), 失败就保留 Phase 2 现状

## 7. 完成标准

- [ ] Step 1: axis 4 错误根因定锤 (具体 op 名 + 类型)
- [ ] Step 3 或 Step 4: unfused backend ONNX 在 CoreML EP 上加载成功 + inference 跟 CPU parity
- [ ] (可选, 推荐) Step 5: 工程化, env knob + 默认值 + 文档
- [ ] (可选, 推荐) 长音频 N=2 实测 wall < 380s (相比 Phase 2 的 415s 再 -10%)
- [ ] 所有 unit test 通过
- [ ] 所有 integration test (含 parity) 通过

如果只完成 Step 1-3 不能工程化(数值精度问题, 性能没收益, etc), **报告失败原因 + 数据**, 不强行 ship。Phase 2 现状 -16.1% 已经是稳的, 不要 regress。

## 8. 开始

1. `git log --oneline -10` 确认起点 (33c4cd3)
2. 读必读清单 1-6
3. 看用户提供的 deep research 输出 (如果有)
4. 跑 Step 0 venv 检查 + 装 transformers
5. 直接进 Step 1 — 写复现脚本, 拿完整 stack trace
6. 不停下来问用户问题, 跑通就 commit, 失败就分析根因, 自主前进
7. 完成或卡死时, 输出: 完成的 Step + 关键数据 + 下一步建议
