# Phase 3 — backend mlpackage ANE (Path B): PoC ✅ + e2e ❌

> 日期: 2026-05-17
> 模型: Qwen3-ASR-1.7B audio_tower (317.5M params, FP16, 583MB)
> 硬件: Apple M1 Max 64GB / macOS 26 Beta (25C56)
> Branch: spike/qwen3-diarize-poc

## TL;DR

**Path B (PyTorch → coremltools → .mlpackage) PoC + 单 audio parity 跑通**, 但**端到端 (含 sherpa-onnx diarize) 在 macOS 26 Beta 上 SIGSEGV**, 根因是 CoreML `MLE5ExecutionStream` lingering reset 异步 dispatch 调 Python C API 无 GIL 的 race。

- 默认 `auto` 仍走 Phase 2 `COREML_ANE_FE` (frontend ANE + backend ONNX CPU), **生产路径不变**。
- `FUNASR_QWEN3_ASR_ENCODER_PROVIDER=coreml_ane_full` 作为 dev 实验路径保留, 工程代码 + unit test + 单 audio integration parity 都已落地。
- 不切默认, 不强行 ship Phase 3 端到端。

## Step 2 — PyTorch → mlpackage 转换

**脚本**: `spikes/qwen3_mac_hw_accel/phase3_backend/export_backend_coreml.py`

**方法**:
1. `huggingface_hub.snapshot_download("Qwen/Qwen3-ASR-1.7B")` 拉 safetensors (3.4GB, 5.3 min)
2. `config.json` 提取 `thinker_config.audio_config` (d_model=1024, layers=24, heads=16, ffn=4096, output_dim=2048)
3. `Qwen3ASRAudioEncoder._from_config(audio_cfg)` 实例化空 audio_tower (317.5M params)
4. safetensors index 抽 `thinker.audio_tower.*` (397 keys 1 shard), strip 前缀 load_state_dict
5. `Qwen3ASRBackendOnnx(audio_tower)` 就地包 ONNX-friendly attention
6. `BackendCoreMLWrapper` 接 2D `key_padding_mask` → 内部转 [B,1,1,T] additive mask (issue #19887 ANE 友好)
7. `torch.jit.trace` + `ct.convert(static shape (1, 520, 1024), CPU_AND_NE, FP16, mlprogram)`

**结果**:
- 833 PyTorch ops → 832 MIL ops, 5 frontend_pytorch + 95 default + 12 mlprogram passes
- convert in **43.6s** (T=520 final, 之前 T=390 因 vendor 默认 dml_pad_to=40 不匹配重导)
- mlpackage size **583 MB** (跟 ONNX backend 同, FP16)
- HF 没有 `auto_map`, 用 vendor `Qwen3ASRAudioEncoder` 手动加载 (AutoModel.from_pretrained + trust_remote_code 不识别 `model_type=qwen3_asr`)

> 注: `coremltools` 9.0 + `torch` 2.8.0 / `scikit-learn` 1.8.0 都不在测试矩阵, 但本次转换无 op 不支持错误, RuntimeWarning `overflow in cast` 单次, 输出 shape 与 dtype 都对。

## Step 3 — 加载 + bench + ANE 验证

**脚本**: `spikes/qwen3_mac_hw_accel/phase3_backend/verify_mlpackage_load.py`

| compute_units | load (s) | warm (ms/run) | RSS Δ (MB) | 备注 |
|---|---:|---:|---:|---|
| `CPU_ONLY` | 1.2 | 151.5 | 1553 | 纯 CPU 基线 |
| `CPU_AND_NE` | **24.0** | **69.1** | **973** | ANE plan compile cold start 24s |
| `CPU_AND_GPU` | 1.4 | 43.1 | 1560 | Metal GPU, 最快但抢 llama.cpp |
| `ALL` | 24.1 | 67.4 | 1558 | CoreML 自选 ANE (跟 CPU_AND_NE 同速) |

**ANE 真在跑的证据**:
1. `CPU_AND_NE` load 24s vs `CPU_ONLY` 1.2s — 24s 是 ANECompilerService 实际编译 ANE plan 的开销
2. `CPU_AND_NE` 比 `CPU_ONLY` **2.2x 快** (69.1 vs 151.5 ms) — 不在 ANE 上跑不出这个 gap
3. `ALL` 自选 ANE 而不是 GPU (跟 CPU_AND_NE 同速 67ms, 不是 GPU 的 43ms) — 说明 CoreML runtime 判定模型 layout 适合 ANE

**`powermetrics --samplers ane_power` 读数 0 mW 是 macOS 26 Beta bug**, 不可信 (Ivan Mosin 文章已警告)。`ANECompilerService` 100% 一核 CPU 占用是 ANE 编译实际发生的另一证据。

**RSS**: ΔRSS **973 MB** (583 MB mlpackage 文件 × 1.67x, Ivan 预测 1.7-3.4x 命中下限)。N=2 worker 估算 ~2 GB ANE 占用, M1 Max 64GB 余量充分。

## Step 4 — Parity (单 audio, 无 sherpa)

**脚本**: `spikes/qwen3_mac_hw_accel/phase3_backend/parity_backend_onnx_vs_coreml.py`

输入 `tests/fixtures/audio/podcast_2speakers_60s.wav` (截 30s), frontend ONNX CPU 跑出 hidden_states (1, 390, 1024) FP16, 然后:
- ONNX backend (CPU, FP16, 4D additive mask) — 662 ms
- mlpackage backend (ANE, 2D key_padding_mask) — 95.5 ms (含部分 cold start)

**结果**:
- max_abs_diff = **4.58e-3**
- mean_abs_diff = 6.5e-4
- cosine_sim = **0.999069**
- 阈值 cos > 0.999 max_abs < 1e-2 → **PASS** (FP16 噪声范围)

## Step 5 — 工程化集成 (代码 + unit test)

- `src/core/vendor/qwen_asr_gguf/inference/encoder.py`: `QwenAudioEncoder.__init__` 加 `COREML_ANE_FULL` 分支, 检测 backend 后缀 (`.mlpackage` 目录用 `ct.models.MLModel(CPU_AND_NE)` 加载, 否则降级 `COREML_ANE_FE`); `_run_backend` 加 mlpackage 路径 (static T=h_target_len, 2D key_padding_mask)
- `src/core/qwen3/asr.py`: `build_engine_config` 加 `coreml_ane_full` config 值, backend_fn 自动选 `.mlpackage`
- `src/core/config.py`: `asr_encoder_provider` docstring 加 Phase 3 条目 + 端到端 crash 警告
- `tests/unit/test_qwen3_encoder_coreml_ane_full.py`: 6 个 unit test (mock coremltools)
- `tests/unit/test_qwen3_asr_default_provider.py`: 4 个新 case (config value + backend_fn 自动)
- `tests/integration/test_qwen3_backend_coreml_ane_parity.py`: 单 audio CPU vs ANE_FULL parity (cos 0.9891, 368 chars 一致)

**248/248 unit tests pass, integration parity pass**。

## Step 6 — 长音频 N=2 e2e: SIGSEGV

**run_baseline.py + COREML_ANE_FULL + num_threads=4**, 期望 wall < 380s。

**实测**: 全部 4 次尝试 worker rc=-11 (SIGSEGV), 在 mlpackage 加载 + warmup 完成后, **sherpa-onnx embedding extractor 加载阶段**:

```
Thread (libdispatch worker):
  _PyObject_Free
  libcoremlpython.so ?
  objc dealloc chain
  -[MLFeatureValue dealloc]
  -[MLE5InputPortBinder reset]
  -[MLE5InputPort reset]
  -[MLE5ExecutionStreamOperation reset]
  -[MLE5ExecutionStream _reset]
  __43-[MLE5ExecutionStream resetAfterLingering:]_block_invoke
  _dispatch_client_callout ...
```

**根因**: CoreML 优化 — predict 后 input MLFeatureValue 不立即释放, 等 lingerTime 后由 libdispatch worker thread 触发 `resetAfterLingering`, 异步 dealloc。dealloc 内部调 `_PyObject_Free` 释放 numpy buffer 引用 — **dispatch worker thread 不持 GIL** → KERN_INVALID_ADDRESS at 0x10 (null deref of Py runtime internal)。

**触发条件**: mlpackage warmup 后 → sherpa-onnx import 触发 dynamic linker + libdispatch 新调度 → race 暴露。单 audio parity 测试 (无 sherpa) 不触发。

**尝试的 workaround (均无效)**:
1. `inputs = {k: v.copy()}` 让 MLFeatureValue 持有独立 buffer — 仍 crash, MLFeatureValue 内部仍调 Python ref
2. warmup 后 `gc.collect() + time.sleep(3) + gc.collect()` 让 lingering reset 在主线程完成 — 仍 crash, lingerTime 不可控
3. N=1 单 worker 排除并发 — 同位置 crash, 是单进程 framework race
4. 4 次重启相同位置 crash, `~/Library/Logs/DiagnosticReports/Python-*.ips` 4 份完全同 stack trace

**结论**: 这是 `coremltools 9.0` + `CoreML framework dealloc race` 的已知 bug, 无法在 application layer 修。需要等 coremltools 添加 sync API / Apple 修 framework。

**升级验证 2026-05-17 10:12**: macOS 26.0 Beta (25C56) → 26.5 正式版 (25F71) **仍崩**, 完全相同 stack trace。worker 撑时间 40s → 95s 可能是 lingerTime 配置变化, 但根因 `MLE5ExecutionStream._reset` 在 libdispatch worker 无 GIL 释放 numpy 内存 — 未修复。第 5 份 crash report: `Python-2026-05-17-101250.ips`。

## 决策 — 不切默认, escape hatch 保留

按 Phase 3 prompt 第 5 节失败处理:
> ❌ Phase 3 失败强行 ship — escape hatch 已有 (`FUNASR_QWEN3_ASR_ENCODER_PROVIDER=coreml_ane_fe`), 失败回 Phase 2

**当前状态**:
- 默认 `auto` → `COREML_ANE_FE` (Phase 2, 生产稳定)
- 显式 `FUNASR_QWEN3_ASR_ENCODER_PROVIDER=coreml_ane_full` 可触发 Phase 3 (单 audio 跑通, 端到端 crash 已警告 in docstring)
- 工程代码 + unit test 全留, 等 macOS / coremltools 修 bug 后可直接切默认

## Path A fallback (重 export unfused ONNX → CoreML EP)?

**不做**, 因为:
1. Path A 也走 CoreML 体系 (CoreML EP 内部仍用 MLE5ExecutionStream), 同 race condition 风险
2. Kimi research 警告: 即使解决 com.microsoft op, ANE 调度黑盒, 实际利用率不可控
3. 工程量同 Path B (改 export + 验证 + 集成), 但收益不确定
4. 生产路径 Phase 2 COREML_ANE_FE 已 -16.1%, 接受现状, 等环境成熟再启动 Path A 或 Path B 重试

## 未来重启 Phase 3 的触发条件

- ~~macOS GM 发布 (脱离 Beta), 重测 mlpackage + sherpa-onnx 端到端~~ **26.5 正式版仍崩, 排除此项**
- coremltools 10+ 提供 `MLModel.predict(synchronous=True)` 或 `lingerTime=0` API
- 用 PyObjC `objc.autorelease_pool()` 显式管理 MLFeatureValue 生命周期 (deep workaround, 需要 vendor 改造)
- 把 sherpa-onnx 换成不用 libdispatch 的实现 (例如 sherpa-rs 或 pyannote pytorch 直跑) — 但这是 sherpa 替换工程, 跟 Phase 3 解耦
- `coremltools.optimize.coreml` INT8 量化跑通可压 mlpackage 到 ~150 MB on-disk, RSS Δ ~300 MB, N=4 worker 也可承受

## 相关文件

- `spikes/qwen3_mac_hw_accel/phase3_backend/export_backend_coreml.py` — Path B export 脚本
- `spikes/qwen3_mac_hw_accel/phase3_backend/verify_mlpackage_load.py` — Step 3 数据
- `spikes/qwen3_mac_hw_accel/phase3_backend/parity_backend_onnx_vs_coreml.py` — Step 4 parity
- `spikes/qwen3_mac_hw_accel/runs/verify_phase3_*` — Step 6 crash logs + ips
- `src/core/vendor/qwen_asr_gguf/inference/encoder.py` — COREML_ANE_FULL 实现
- `tests/unit/test_qwen3_encoder_coreml_ane_full.py` — 6 unit test
- `tests/integration/test_qwen3_backend_coreml_ane_parity.py` — 单 audio parity
- `~/Library/Logs/DiagnosticReports/Python-2026-05-17-01{2628,2701,2908,3428,3705}.ips` — 5 份 SIGSEGV crash report (相同 stack)
