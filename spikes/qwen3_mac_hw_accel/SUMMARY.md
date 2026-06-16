# Qwen3-ASR Mac 硬件加速 PoC — 总结报告

> 日期: 2026-05-16
> 分支: spike/qwen3-diarize-poc
> 机器: Apple M1 Max (10 CPU + 32 GPU + 16 ANE), macOS 26.2, 64GB
> 目的: 解决 N=2 worker 并发时 CPU 满 / GPU 空等 / ANE 0 用的硬件资源失配问题

## 一句话结论

**生产改两个 knob 即可获得 -16.1% wall 收益, 无需重写 vendor 引擎**:
1. **`FUNASR_QWEN3_NUM_THREADS=4`** (零代码改动, 节省 11.5%)
2. **ASR encoder frontend 走 CoreML ANE** (`only_frontend + CPUAndNeuralEngine`, 小工程化, 再省 5%)

**这两个 knob 组合 = `d13_combo_t4_feane_n2`, N=2 wall: 495.6s → 415.8s (-16.1%)**

## 实验矩阵 (N=2, 1spk-16min + 4spk-44min 真并发, 各跑一遍)

| Tag | num_threads | sherpa | ASR encoder | Wall (s) | Δ baseline | ANE mean | GPU mean | 备注 |
|---|---:|---|---|---:|---:|---:|---:|---|
| **baseline** | 8 | cpu | CPU | 495.6 | — | 0 mW | 32.4% | 现状, 用户 prompt 中 808s 来自 server, 此处绕开 server 直跑 |
| **d3 t=4** | **4** | cpu | CPU | **438.8** | **-11.5%** ✅ | 6 | 35.1% | 单变量, 零代码 |
| d1d fe-ANE | 8 | cpu | **fe ANE** + be CPU | 458.6 | -7.5% ✅ | **44** | 33.3% | 单变量, ANE 启用 |
| **d13 combo** | **4** | cpu | **fe ANE** + be CPU | **415.8** | **-16.1%** ⭐ | **50** | 35.7% | 组合最优 |
| d2 sherpa CoreML | 8 | **coreml** | CPU | 702.1 | **+41.7%** ❌ | 37 | 24.9% | 反向, 拒收 |

> baseline wall 495s 与 prompt 808s 的差异: prompt 经 server (websocket + 文件 IO + protocol), 本 spike 绕过 server 直接 spawn worker, 路径更纯净。

## 各方向结论

### ✅ 方向 0: Profiling baseline (`profiling_baseline.md`)

- N=1 1spk-16min: wall=157s, RTF=0.162
- N=1 4spk-44min: wall=409s, RTF=0.156
- N=2 (混合): wall=495.6s
- **stage breakdown (N=1 1spk)**: ASR encoder backend 94.8s (**60% wall, 头号 CPU 瓶颈**), sherpa pipeline 72s (并行, 不在串行路径), LLM decode 37.7s (Metal GPU, 唯一 GPU 阶段)
- **N=2 vs N=1 (1spk task)**: 所有 CPU 阶段一致放大 1.74-1.76x, GPU 阶段几乎不放大 (1.12x), 这是用户观察的 "CPU 满, GPU 空等" 量化证据
- **ANE 全程 0 mW** — 证实老结论 "Mac 上 ANE/CoreML 实测无加速" 在 baseline 是事实, 但只因没人切到 CoreML EP

### ✅ 方向 3: num_threads=4 (`num_threads_tuning.md`)

- t=8 → t=4 N=2: wall -11.5% (495.6→438.8s), 1spk RTF 0.252→0.180, 4spk 0.188→0.166 — **单 task 反而更快**
- 副作用: t=8 时观察到 4spk audio.load 异常 268s, t=4 时正常 4s — 原因是 t=8 (2×8=16 thread on 10 core) 严重过订阅, ffmpeg pipe stdout 被 starve
- 这条 knob **零代码、零风险**, 应立刻推到生产
- 计算逻辑: pool_size=2 × num_threads=4 = 8 ≤ 10 cores, 留 2 core 给系统

### ⚠️ 方向 1: ASR encoder CoreML (`coreml_asr_encoder.md`)

- 跑了 5 个 variants (ALL/NeuralNetwork/CPUOnly/CPUAndNeuralEngine × only_frontend)
- **核心发现**:
  - backend ONNX 在 MLProgram format 下 build 时报 `axis 4 not in [-4,3]` (CoreML EP op 兼容 bug), NeuralNetwork format 下 silent fallback CPU
  - CoreML ALL units (含 GPU) **会跟 llama.cpp Metal 抢 GPU**, 让 LLM decode 慢 3x
  - **only_frontend + CPUAndNeuralEngine units** 是唯一 work 的组合:
    - frontend 14.8s → 5.9s (-60%, ANE 启动 mean 48mW max 334)
    - llm_decode 保持 37s 不退化 (排除 GPU 后 Metal 不抢)
    - N=1 wall -6.1%, N=2 wall -7.5%

### ❌ 方向 2: sherpa CoreML (`sherpa_coreml.md`)

- `provider=coreml` 单变量 N=2 wall: 495.6 → 702.1s (**+41.7%**, slowdown)
- sherpa pipeline 慢 56-73% (CPU → CoreML 后单 inference 慢)
- 连带让同进程 ASR encoder CPU EP 也慢
- 原因: pyannote-segmentation 和 nemo-titanet-small 都是小模型多次调用, CoreML EP dispatch overhead > ANE 加速
- **拒收**, 永久禁用

## 推荐工程化路线 (3 Phase)

### Phase 1 — 改 num_threads default (next PR, 半小时工作量)

**改动范围**:
- `src/core/config.py`: `default num_threads` field 默认 8 → 4
- `.env.example`: `FUNASR_QWEN3_NUM_THREADS=4` 取消注释
- `docs/部署.md`: 更新 "长音频并发 RTF 翻倍预警" 段, 加 num_threads 段
- `tests/integration/`: 跑 parity 确认 N=2 不破坏

**收益**: N=2 wall -11.5%, 单 task RTF 更好  
**风险**: 0 (核心算法无任何改变)

### Phase 2 — 加 ASR encoder frontend ANE 选项 (next-next PR, 2-3 天工作量)

**改动范围**:
- `src/core/vendor/qwen_asr_gguf/inference/encoder.py`: 加 CoreML 分支
  ```python
  elif self.onnx_provider == 'COREML_ANE_FE':
      providers_fe = [('CoreMLExecutionProvider', {
          'ModelFormat': 'MLProgram',
          'MLComputeUnits': 'CPUAndNeuralEngine',
          'RequireStaticInputShapes': '0',
      }), 'CPUExecutionProvider']
      providers_be = ['CPUExecutionProvider']
  ```
- `src/core/qwen3/asr.py`: `build_engine_config` 默认 `onnx_provider="COREML_ANE_FE"` (macOS only), 把那条 `Mac 上 ANE/CoreML 实测无加速` 注释改成 "frontend ANE 验证有效, backend 卡 axis 4 op"
- `Qwen3Config`: 加 `asr_encoder_provider: str` 字段, 默认 "coreml_ane_fe"
- 加 unit test: macOS 平台跑 frontend ANE, 非 macOS fallback CPU
- 加 parity test: 转录文本完全一致 (frontend ONNX 是确定计算, ANE 和 CPU 结果应该 bit-exact)

**收益**: 再省 -5%, 总 N=2 wall ~ -16%  
**风险**: 中
  - ANE 在不同 macOS 版本 (M2/M3) 行为可能不同, 需 fallback 测试
  - CoreML 第一次 build 模型有 cache 时间 (5-15s 启动), worker 启动慢
  - MUST 用 CPUAndNeuralEngine, ALL units 会拖 llama.cpp

### Phase 3 — 修 backend ONNX axis 4 (低优先级, 不建议立刻)

要重新 export Qwen3-ASR encoder backend ONNX 模型, 改某个 Slice/Gather op 让 axis ∈ [-4,3]。  
**估算节省**: 若 backend 也能走 ANE, encoder 总耗时可能再省 50-80s, N=2 wall 进一步 -10~15%。  
**估算工作量**: 1-2 周 (要研究 axis 4 是哪个 op, 改 export 脚本, 重新转 ONNX, 验证数值精度, 跑 parity)。  
**风险**: 高 (数值精度风险, 模型版本管理风险)。

## 不上路线

- ❌ **sherpa CoreML** 永久禁用 (方向 2 实测 +41% slowdown)
- ❌ **CoreML ALL units** (含 GPU) 永久禁用 — 跟 llama.cpp Metal 抢
- ❌ **多 ANE 实例并发 (方向 5)** — PoC 没测, 但 d1d 数据显示 ANE 单 worker 用量已经够, 无需再分。多个 worker 各自跑 CoreML session, OS 自己分配 ANE。
- ❌ **ASR encoder/decoder 流水线 (方向 4)** — 改 vendor code 大动作, 不必。Phase 1+2 已经把 wall 拉到 416s, 距离理论下限 (4spk 单 task 408s) 仅 8s gap。

## 关键数据收藏

- baseline (CPU only) GPU residency: **26-32%** → 仍有 ~70% GPU headroom 但被 ASR encoder 阻塞, llama.cpp 单独跑只占 GPU 短时间
- d13 combo GPU residency: **35.7%** → 仅微升, 因为 GPU 真正的工作是 LLM decode (37-107s), 不是 encoder
- ANE 真正能用上, 但只用 ASR encoder frontend 一个模型, **ANE peak 334mW 时 ASR frontend 跑得飞快, 但占总 wall 比例小**
- **CoreML/llama.cpp Metal 资源争抢是个真问题**: ALL units 会让 LLM decode +200% 慢, 必须排除 GPU

## 反思 / 经验

1. **`onnx_provider="CPU"` 的老注释 ("Mac 上 ANE/CoreML 实测无加速") 部分过时**: 当年实测可能用 ALL units, 才有抢 GPU 的反向损失。今天 PoC 证明 `CPUAndNeuralEngine` units + only_frontend 是 ANE 唯一可用姿势。
2. **过订阅 (over-subscription) 是隐性大坑**: 16 thread on 10 core 不只是慢, 还会让 ffmpeg pipe stdout 几十倍阻塞 (audio.load 3s→268s)。这种数据通过 wall 看不出来, 只在 stage timing 暴露。
3. **CoreML EP 在 ORT 1.26 仍有 op 兼容 bug** (axis 4), 想完整用 ANE 必须啃 ONNX 模型层。短期 only_frontend 路线最稳。
4. **方向 2 sherpa CoreML 的负面数据是有价值的**: 不是 ANE 不行, 是这种"小模型多次调用"模式不适合 ANE dispatch overhead — 跟 gemini.md 推荐的 WeSpeaker 单次长 inference 是不同问题。

## 文件清单

```
spikes/qwen3_mac_hw_accel/
├── SUMMARY.md                      # 本文件
├── MATRIX.md                       # 实验矩阵规划
├── profiling_baseline.md           # 方向 0
├── num_threads_tuning.md           # 方向 3
├── coreml_asr_encoder.md           # 方向 1
├── sherpa_coreml.md                # 方向 2 (负面)
├── timing_hooks.py                 # monkey-patch 探针
├── profile_worker.py               # 单 worker entry
├── run_baseline.py                 # N=1/N=2 runner + powermetrics
├── analyze_run.py                  # 分析 summary + powermetrics
├── run_long_concurrent.py          # /tmp 拷过来的原 baseline 脚本
└── runs/                           # 各 tag 的 summary.json + powermetrics.log
    ├── b0_cpu_t8_n1_1spk/
    ├── b0_cpu_t8_n1_4spk/
    ├── b0_cpu_t8_n2/               # ⭐ baseline
    ├── d3_cpu_t4_n2/               # ⭐ winning t=4
    ├── d1a_onlyfe_t8_n1_1spk/      # CoreML ALL fe-only 实验
    ├── d1b_nn_t8_n1_1spk/          # NeuralNetwork format 实验
    ├── d1c_cpuane_t8_n1_1spk/      # CPUAndNeuralEngine fe+be (axis 4)
    ├── d1d_fe_cpuane_t8_n1_1spk/   # ⭐ winning fe-only CPUAndNeuralEngine N=1
    ├── d1d_fe_cpuane_t8_n2/        # winning fe-only N=2
    ├── d2_sherpa_coreml_t8_n2/     # ❌ 方向 2 负面
    └── d13_combo_t4_feane_n2/      # ⭐⭐ 组合最佳
```

## 后续追测建议 (非紧急)

- d3 N=2 t=2 — 看更激进的 thread 收缩 (4 worker thread on 10 core) 是否更好
- d3 N=2 t=12 — 拍 over-subscribe 上界
- d13 + sherpa preset 调整 (e.g. embedding 减 thread) — 看是否进一步省
- pool_size=3 t=3 的极限并发 — 用户已知 pool 上限 3, 看 3×3=9 是否极限
