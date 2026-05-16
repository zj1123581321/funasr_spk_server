# 新 session 启动 prompt — Qwen3-ASR Mac 硬件加速工程化 (Phase 1 + Phase 2)

> 日期: 2026-05-16
> 上一阶段: PoC 完成 (commit 4f29612 `spike(qwen3): Qwen3-ASR Mac 硬件加速 PoC 完整记录`)
> PoC 结论: N=2 wall 495.6→415.8s (-16.1%), 改两个 knob 不动 vendor 算法
> 本阶段目标: 把 PoC 验证有效的两个 knob (num_threads=4 默认 + ASR encoder frontend ANE 路径) 工程化落地到 src/
> 工作方式: **严格 TDD**, 每个红→绿→commit 是最小单位, 不停下来向用户提问, 测试通过即时 commit, 直到整个任务完成

## Prompt 正文 (复制以下内容到新 session)

你接手 funasr_spk_server 项目的 Qwen3-ASR Mac 硬件加速工程化阶段。上一 session 跑完了完整 PoC, 验证了两个工程化 knob 有效:

1. **Phase 1: `FUNASR_QWEN3_NUM_THREADS` 默认 8 → 4** (零代码逻辑改动, N=2 wall -11.5%)
2. **Phase 2: ASR encoder frontend 走 CoreML ANE** (vendor encoder.py 加 CoreML 分支, 再 -5%)

本 session 把这两个 Phase 工程化落地。**严格 TDD, 自主执行到完成, 非必要不要停下来提问**。

### 项目路径

`/Users/zhanglixing/Dev/projects/250729_funasr_spk_server/funasr_spk_server`

### 必读 (按顺序, 建立 context)

1. `spikes/qwen3_mac_hw_accel/SUMMARY.md` — PoC 总结 + 3 Phase 工程化路线 + 实测数据
2. `spikes/qwen3_mac_hw_accel/num_threads_tuning.md` — Phase 1 数据依据
3. `spikes/qwen3_mac_hw_accel/coreml_asr_encoder.md` — Phase 2 数据依据 + 5 个 variants 取舍
4. `spikes/qwen3_mac_hw_accel/sherpa_coreml.md` — 方向 2 (负面案例), **明确说明不要碰 sherpa CoreML**
5. `CLAUDE.md` — 项目根 (测试约定 / 部署约定)
6. `src/core/config.py` 第 110-130 行 + 第 275-280 行 — Qwen3 pool / num_threads 相关 (Phase 1 改这里)
7. `src/core/vendor/qwen_asr_gguf/inference/encoder.py` 第 122-170 行 — QwenAudioEncoder.__init__ 的 provider 选择逻辑 (Phase 2 加分支这里)
8. `src/core/qwen3/asr.py` 第 60-110 行 — `build_engine_config` 默认 `onnx_provider="CPU"` (Phase 2 改默认 + 改注释)
9. `src/core/qwen3_transcriber.py` 第 200-260 行 — Qwen3DiarizeTranscriber 构造 + 字段 (Phase 2 可能加新字段)
10. `tests/unit/test_qwen3_*.py` — 现有 Qwen3 单元测试, 看测试风格
11. `tests/integration/test_qwen3_server_websocket_e2e.py` — Qwen3 真 e2e 测试 (parity 测试参考)

### TDD 流程铁律 (整个 session 必须遵守)

1. **红 → 绿 → commit 是最小单位**, 不要积累多个改动
2. 写测试 → 跑测试看到失败 (RED) → 改代码 → 跑测试看到通过 (GREEN) → commit
3. **每个 commit 都要测试通过**, 失败的测试不许 commit
4. **每个 PR 阶段结束跑一次 parity**: `FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/`
5. parity 通过才算 Phase 完成
6. commit message 引用具体测试名 + 改动什么 + 解决什么 (参考 `git log --oneline | head -20` 已有的风格)

### 工作方式约束 (与 PoC 同)

- 用中文回应
- venv: `venv/bin/python`
- 环境: `unset TMPDIR; export TMPDIR=/tmp; export DYLD_LIBRARY_PATH="$PWD/src/core/vendor/qwen_asr_gguf/inference/bin"`
- 单元测试: `venv/bin/python -m pytest tests/unit/ -q` (毫秒级, 频繁跑)
- 集成测试: `FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/ -q` (含真模型, 跑一次需要 ~3-5min, Phase 结束跑)
- 用 ctx_batch_execute / ctx_execute_file 处理大输出, Read 用于 Edit 前
- 长任务 (>1min): `Bash run_in_background: true`, 等通知
- 不提交 audio/模型/tmp_long_audio 到 git
- **每个 Phase 开始前先看下 `git log --oneline -5` 确认起点**

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### Phase 1 — `num_threads` 默认改 8→4
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**PoC 数据依据**:
- N=2 t=8: wall 495.6s
- N=2 t=4: wall 438.8s (-11.5%), 单 task RTF 也更好
- t=8 时观察到的 audio.load 268s 异常 (CPU 严重过订阅) 在 t=4 时正常 (4.2s)
- 计算: pool_size=2 × num_threads=4 = 8 ≤ 10 cores (M1 Max), 留 2 core 给系统

**改动范围**:
- `src/core/config.py`: Qwen3 config 中 `num_threads` 字段默认值 8 → 4
  - 查找当前位置: `grep -n 'num_threads' src/core/config.py`
- `.env.example`: 把 `# FUNASR_QWEN3_NUM_THREADS=8` 改成 `FUNASR_QWEN3_NUM_THREADS=4` (取消注释 + 改值), 在上面加一行注释解释为什么是 4
- `docs/部署.md`: 找到 "长音频并发 RTF 翻倍预警" 那段 (大概 160-170 行), 补一段说明 num_threads=4 是 PoC 验证后的最优值, 引用 `spikes/qwen3_mac_hw_accel/SUMMARY.md`

**TDD 步骤**:

**Step 1.1**: 加 unit test 验证默认值改了
- 文件: `tests/unit/test_config_qwen3_num_threads.py` (新建)
- 测试: `test_qwen3_num_threads_default_is_4`
  - import config loader (按现有测试风格, 大概是 `from src.core.config import config`)
  - 不设 env, 加载 config, 断言 `config.transcription.qwen3_num_threads == 4` (或对应字段名)
  - 同时验证字段位置 / 类型
- 测试: `test_qwen3_num_threads_env_override_still_works`
  - 设 `FUNASR_QWEN3_NUM_THREADS=8`, 加载, 断言 == 8 (env 仍能 override)
- 跑测试: 应该红 (默认还是 8)
- 改 `src/core/config.py` 默认 8→4
- 跑测试: 绿
- **commit**: `chore(qwen3): num_threads 默认 8→4 (PoC 验证 N=2 wall -11.5%)`

**Step 1.2**: 更新 .env.example
- 改 `.env.example` 中 `FUNASR_QWEN3_NUM_THREADS` 那行 (取消注释 + 改 4 + 加说明)
- 没专门的测试, 但跑 `venv/bin/python -m pytest tests/unit/ -q` 确认现有 215 个测试 + 你新加的两个都过
- **commit**: `docs(qwen3): .env.example 同步 num_threads=4 默认`

**Step 1.3**: 更新部署文档
- 改 `docs/部署.md` 加 num_threads 段
- 跑现有所有测试确认无破坏
- **commit**: `docs: 部署文档同步 num_threads=4 + PoC 数据链接`

**Step 1.4**: 跑 parity 测试 (集成测试) 关 Phase 1
- `FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/ -q`
- 全过 ✅ 才算 Phase 1 完成

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### Phase 2 — ASR encoder frontend 走 CoreML ANE
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**PoC 数据依据**:
- 单变量 N=2 wall: 495.6 → 458.6s (-7.5%), frontend 24.5→10.9s (1spk) / 38.8→15.8s (4spk)
- 跟 Phase 1 组合 N=2 wall: 495.6 → 415.8s (-16.1%)
- ANE 启用 (mean 50mW max 336mW)
- llm_decode 保持 37s 不退化 (前提: 排除 GPU)
- **唯一 work 的组合**: `MLProgram` format + `CPUAndNeuralEngine` units + only frontend (backend 保 CPU)
- **反例**: `ALL` units 抢 llama.cpp Metal 让 llm_decode +73s; `NeuralNetwork` format silent fallback CPU 无收益; backend 走 CoreML 卡 axis 4 op 兼容报错

**改动范围**:

#### 改动 1: `src/core/vendor/qwen_asr_gguf/inference/encoder.py`
- `QwenAudioEncoder.__init__` 第 122-170 行加新分支
- 新分支识别 `onnx_provider="COREML_ANE_FE"` (字符串值, 不破坏现有 CPU/TRT/DML/CUDA 分支)
- 走 CoreML EP + MLProgram + CPUAndNeuralEngine + only frontend
- backend 强制 CPU
- 如果当前 platform 不是 macOS (`sys.platform != 'darwin'`), 自动 fallback 全 CPU + 打 warning log

参考实现 (从 `spikes/qwen3_mac_hw_accel/profile_worker.py` 的 `_install_coreml_asr_patch` 拷贝核心逻辑):
```python
elif self.onnx_provider == 'COREML_ANE_FE':
    import sys
    if sys.platform != 'darwin':
        if self.verbose:
            print(f"--- [Encoder] COREML_ANE_FE only on macOS, fallback CPU ---")
        # providers 保持 ['CPUExecutionProvider']
    else:
        available = ort.get_available_providers()
        if 'CoreMLExecutionProvider' in available:
            providers_fe = [
                ('CoreMLExecutionProvider', {
                    'ModelFormat': 'MLProgram',
                    'MLComputeUnits': 'CPUAndNeuralEngine',
                    'RequireStaticInputShapes': '0',
                    'EnableOnSubgraphs': '0',
                }),
                'CPUExecutionProvider',
            ]
            providers_be = ['CPUExecutionProvider']  # backend 卡 axis 4, 必须 CPU
            self.sess_fe = ort.InferenceSession(frontend_path, sess_options=sess_opts, providers=providers_fe)
            self.sess_be = ort.InferenceSession(backend_path, sess_options=sess_opts, providers=providers_be)
        else:
            if self.verbose: print("--- [Encoder] CoreML EP unavailable, fallback CPU ---")
            # providers 保持 CPU
```

注意:
- 现有代码用一个 `providers` 变量, 你的新分支需要分开 `providers_fe` / `providers_be` 来实现 frontend-only
- 改 sess_fe / sess_be 的 InferenceSession 构造逻辑, 保留对原 onnx_provider 路径的兼容
- 不要破坏原有 TRT/DML/CUDA/CPU 分支
- 改完保留原 mel_extractor + dtype 检测 + 预热 那段

#### 改动 2: `src/core/qwen3/asr.py`
- `build_engine_config` 默认 `onnx_provider="CPU"` → `onnx_provider="COREML_ANE_FE"` (macOS) / `"CPU"` (其他平台)
- 用 `sys.platform == 'darwin'` 判断
- 改 line 73 那条注释 "Mac 上 ANE/CoreML 实测无加速" → "Mac 上 frontend ANE 验证有效 (-7.5% N=2 wall), backend 卡 axis 4 op 兼容用 CPU. 见 spikes/qwen3_mac_hw_accel/SUMMARY.md"

#### 改动 3: `src/core/config.py` (可选)
- 加 `qwen3_asr_encoder_provider: str = "auto"` 字段 (auto 时按平台决定, "cpu" 强制 CPU, "coreml_ane_fe" 强制 ANE)
- env override: `FUNASR_QWEN3_ASR_ENCODER_PROVIDER`
- 这是 escape hatch, 让生产环境出问题时能一键关掉

**TDD 步骤**:

**Step 2.1**: 加 unit test 验证 COREML_ANE_FE 分支被识别
- 文件: `tests/unit/test_qwen3_encoder_provider.py` (新建)
- 测试: `test_encoder_coreml_ane_fe_on_macos_uses_coreml_ep` (mock `sys.platform = 'darwin'`, mock `ort.get_available_providers` 返回 含 CoreMLExecutionProvider, mock `ort.InferenceSession`, 断言 sess_fe providers 含 CoreML, sess_be providers = [CPU])
- 测试: `test_encoder_coreml_ane_fe_on_linux_fallback_cpu` (mock `sys.platform = 'linux'`, 断言 sess_fe / sess_be 都是 CPU only)
- 测试: `test_encoder_coreml_ane_fe_no_ep_fallback_cpu` (macOS 但 ort 没 CoreMLExecutionProvider, fallback CPU)
- 测试: `test_encoder_default_cpu_unchanged` (onnx_provider='CPU' 时行为跟现在完全一样, 别破坏)
- 跑测试: 红
- 改 `src/core/vendor/qwen_asr_gguf/inference/encoder.py` 加 COREML_ANE_FE 分支
- 跑测试: 绿
- **commit**: `feat(qwen3-vendor): encoder 加 COREML_ANE_FE provider (frontend ANE + backend CPU)`

**Step 2.2**: 加 unit test 验证 build_engine_config 默认值
- 文件: `tests/unit/test_qwen3_asr_default_provider.py` (新建)
- 测试: `test_build_engine_config_default_macos_is_coreml_ane_fe` (mock `sys.platform = 'darwin'`, 调 `build_engine_config(model_dir='/tmp/fake')`, 断言 `cfg.onnx_provider == 'COREML_ANE_FE'`)
- 测试: `test_build_engine_config_default_linux_is_cpu`
- 测试: `test_build_engine_config_explicit_cpu_still_works` (用户传 `onnx_provider='CPU'` 时不被覆盖)
- 跑测试: 红
- 改 `src/core/qwen3/asr.py` 的 `build_engine_config`
- 跑测试: 绿
- **commit**: `feat(qwen3): build_engine_config macOS 默认走 frontend ANE`

**Step 2.3** (可选, 推荐): config knob
- 测试: `test_qwen3_asr_encoder_provider_env_override` 
- 改 `src/core/config.py` 加字段 + env override
- 改 `src/core/qwen3/asr.py` 让 `build_engine_config` 读 config 而非硬编码 (或者通过参数注入)
- 改 transcriber 把 config 字段传给 build_engine
- 测试通过
- **commit**: `feat(qwen3): 加 FUNASR_QWEN3_ASR_ENCODER_PROVIDER env knob (escape hatch)`

**Step 2.4**: parity test (关键!)
- 文件: `tests/integration/test_qwen3_frontend_ane_parity.py` (新建)
- 需要 `FUNASR_RUN_INTEGRATION=1` 守护
- 测试: 跑同一段 audio (`tests/fixtures/audio/podcast_2speakers_60s.wav`) 两次:
  - 一次 `onnx_provider="CPU"`
  - 一次 `onnx_provider="COREML_ANE_FE"`
  - 比较 ASR text + segments + speaker labels — 应当**完全一致** (frontend ONNX 是确定计算, ANE 和 CPU 数值应 bit-exact 或差异在 chars-level diff ratio < 1%)
- 如果有差异: 记录差异类型 + 决定是否能接受 (frontend FP16/FP32 差异可能造成 < 1% diff, 但 segment 数量应一致, speaker labels 应一致, 转录主要内容应一致)
- 跑 parity: 绿
- **commit**: `test(qwen3): frontend ANE vs CPU parity 集成测试`

**Step 2.5**: 跑全部 parity 关 Phase 2
- `FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/ -q`
- 全过 ✅

**Step 2.6**: 更新文档
- 更新 `docs/部署.md` 说明 macOS 平台自动启用 frontend ANE, 引用 `spikes/qwen3_mac_hw_accel/SUMMARY.md`
- 更新 `src/core/qwen3/asr.py` 那条历史注释 ("Mac 上 ANE/CoreML 实测无加速" 已经过时)
- **commit**: `docs(qwen3): 部署文档 + asr.py 注释同步 frontend ANE 路径`

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### Phase 3 (不做, 仅记录)
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**不要做** 修 backend ONNX axis 4 op。原因:
- 要重新 export Qwen3-ASR encoder backend ONNX (1-2 周, 数值精度风险)
- PoC 数据估算再省 -10~15%, 但风险/收益比不值在本 PR 做
- 留给未来 PoC 独立验证

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 完成标准
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- [ ] Phase 1: num_threads=4 默认 + .env.example + 部署文档 (3 个 commits)
- [ ] Phase 1 parity: `FUNASR_RUN_INTEGRATION=1 pytest tests/integration/` 全过
- [ ] Phase 2: encoder.py COREML_ANE_FE 分支 + build_engine_config 默认 + parity test (3-4 个 commits)
- [ ] Phase 2 parity: 集成测试全过, frontend ANE 转录文本与 CPU baseline 一致 (或差异 < 1%)
- [ ] 全部 unit test 通过: `venv/bin/python -m pytest tests/unit/` (215 + 你新加的)
- [ ] 文档完整: 部署文档 / asr.py 注释 / .env.example 都同步
- [ ] 最后一次跑长音频实测 (可选, 但推荐): `venv/bin/python spikes/qwen3_mac_hw_accel/run_baseline.py --mode n2 --tag verify_engineering_phase1+2 --num-threads 4`, 看 wall 是否复现 ~416s

### 反模式 (不做)

- ❌ 改 vendor encoder.py 后不加测试就 commit
- ❌ 一个大 commit 把 Phase 1 + 2 一起塞进去
- ❌ 跳过 parity 测试直接 push
- ❌ 改 src/core/qwen3/diarize.py 让 sherpa 走 coreml (PoC 验证负面 +41.7%, 永久不要做)
- ❌ 把 CoreML units 写成 'ALL' (会抢 llama.cpp Metal, 实测 llm_decode +73s)
- ❌ 用 NeuralNetwork format (silent fallback CPU, 无收益)
- ❌ 试图让 backend 也走 CoreML (axis 4 op 卡, 必须重导 ONNX)

### 开始

1. `git status && git log --oneline -5` 确认起点 (应该在 commit 4f29612 之后)
2. 读 SUMMARY.md / num_threads_tuning.md / coreml_asr_encoder.md 建立 context
3. 直接进 Step 1.1 — 写 test_qwen3_num_threads_default_is_4
4. **不停下来问用户问题**, 跑通就 commit, 失败就 fix, 自主前进直到 Phase 1 + 2 完成
5. 完成后输出最终 commit 列表 + parity 结果 + (可选) verify 长音频实测数据
