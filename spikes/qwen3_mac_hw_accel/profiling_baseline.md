# 方向 0: Profiling Baseline (per-stage 时间分布 + 硬件 utilization)

> 实验日期: 2026-05-16
> 机器: Apple M1 Max, 10 CPU cores, 32-core GPU, 16-core ANE, macOS 26.2
> 测量方法: spikes/qwen3_mac_hw_accel/profile_worker.py + timing_hooks (monkey-patch, 不动 src/)
> 硬件采样: powermetrics --samplers gpu_power,ane_power,cpu_power -i 1000 (1Hz)

## 实验设计

| 维度 | 选择 |
|---|---|
| 模型 | Qwen3-ASR-1.7B (encoder.onnx 583MB + frontend 23MB + llm.gguf 1.4G q5_k) |
| Sherpa | pyannote-segmentation-3.0 + nemo-titanet-small (CPU) |
| 音频 | audio_1spk_real.m4a (16min) + audio_4spk.m4a (44min) |
| ASR ONNX EP | CPU (默认, asr.py:73 注释 "ANE/CoreML 实测无加速") |
| Sherpa provider | cpu (默认) |
| num_threads | 8 (FUNASR_QWEN3_NUM_THREADS 默认) |
| 并发模式 | N=1 (单进程) / N=2 (两进程并发, 模拟 pool_size=2) |

## 结果

### Baseline N=1 1spk-16min (b0_cpu_t8_n1_1spk)

> 970s 音频, wall=156.95s, RTF=**0.162**, 43 segments, 1 speaker

#### Per-stage 时间分布

| Stage | 耗时 (s) | 占 wall % | 硬件 | 备注 |
|---|---:|---:|---|---|
| `asr.run_total` | 147.51 | 94% | CPU+GPU | ASR 主路径总和 (内部含 audio_load + encoder + llm_decode) |
| └── `asr.encoder.total` | 109.86 | 70% | **CPU** | mel + frontend + backend ONNX |
| │     └── `asr.encoder.backend_onnx` | **94.83** | **60%** | **CPU** | ⭐ 头号瓶颈 |
| │     └── `asr.encoder.frontend_onnx` | 14.79 | 9% | CPU | |
| └── `asr.llm_decode` | 37.68 | 24% | **GPU (Metal)** | llama.cpp q5_K, 走 Metal pipeline |
| `sherpa.diarize.total` | 75.15 | 48% | **CPU** | 与 ASR **并行**, 不串行加 |
| └── `sherpa.pipeline.process` | 72.02 | 46% | CPU | segmentation + embedding 都在内部 |
| `cluster_merge.apply` | 5.84 | 4% | CPU | 在 ASR+Diarize 之后串行 |
| └── `sherpa.embedding.compute` | 4.97 | 3% | CPU | 43 个 segment per-segment 一次 |
| `audio.load` | 1.85 | 1% | CPU | ffmpeg 解码 m4a + 重采样 16k |

注: `asr.run_total` 与 `sherpa.diarize.total` 通过 `asyncio.run_in_executor` 并行执行, 总 wall ≈ max(asr, diarize) + cluster_merge + 后处理。

#### 硬件 utilization (powermetrics 1Hz, 158 samples)

| 指标 | mean | max | 备注 |
|---|---:|---:|---|
| **GPU HW active residency** | **26.4%** | **100.0%** | LLM decode 时短暂 100%, 其余时间在等 CPU |
| **ANE Power** | **0.0 mW** | **0.0 mW** | ⚠️ 全程 0, ANE **完全未启用** |
| CPU Power | 18.9 W | 29.3 W | 持续高位 |

## 核心发现

1. **ASR encoder backend ONNX (94.83s, 60% wall) 是 N=1 头号瓶颈**, 全部在 CPU 上跑。
2. **GPU 平均 residency 仅 26.4%** — 即便 N=1 单跑, GPU 也大部分时间在等 ASR encoder CPU 跑完才能跑 LLM decode (37s 高占用)。
3. **ANE 全程 0 mW** — `asr.py:73` 注释 "Mac 上 ANE/CoreML 实测无加速" 的结论在 baseline 这里仍然成立 (但只测了 onnx_provider="CPU" 的现状; 方向 1 会测真切到 CoreML EP 是否改变这个事实)。
4. **sherpa diarize 75s 与 ASR 并行**, 在 N=1 不是瓶颈 (被 ASR 147s 完全 cover), 但 N=2 时 2 个 worker 抢 CPU, sherpa 和 ASR 会互相拖慢, 这是用户观察到的"前期 CPU 满, GPU 空等"的核心。
5. **cluster_merge per-segment embedding 5.84s** (43 个 segment, 平均 135ms/seg) — 当前 N=1 不大, 但段数越多越长, 长音频 (4spk-44min 估计 ~150 个 segment) 时可能放大到 20-30s。

## 下一步 (待 N=2 baseline 数据)

- N=2 baseline 跑完后, 计算 stage **放大比例** = N=2 单 task elapsed / N=1 单 task elapsed
- 放大最严重的 stage 是首要优化目标
- 根据放大比预测:
  - ASR encoder backend (CPU) — 若 N=2 时放大 1.5-2x, 方向 1 (CoreML) 价值高
  - sherpa pipeline (CPU) — 若 N=2 时放大 1.5-2x, 方向 2 (coreml) 价值高

### Baseline N=2 (b0_cpu_t8_n2)

> 两个 worker 并发: 1spk-16min + 4spk-44min, wall=**495.6s (8.3min)**

> 注: 本 spike 路径绕过 server, 直接 spawn 2 个 worker process. prompt 中 808s 的 baseline 是经 server 路径, 那里多 ~5-15s/任务的启动 + protocol 开销, 此处更清洁。

| 指标 | 1spk-16min (N=2) | 4spk-44min (N=2) |
|---|---:|---:|
| wall | 244.1s | 493.4s |
| duration | 970s | 2626s |
| RTF | 0.252 | 0.188 |
| n_seg | 41 | 297 |
| **ASR encoder backend (CPU)** | **165.9s** | 60.6s |
| ASR encoder frontend (CPU) | 24.5s | 38.8s |
| ASR encoder total | 190.8s | 100.1s |
| ASR llm_decode (Metal GPU) | 42.2s | 107.6s |
| **sherpa pipeline (CPU)** | **126.6s** | 262.8s |
| sherpa diarize total | 129.8s | 268.5s |
| sherpa embedding compute (cluster_merge) | 6.2s | 11.4s |
| cluster_merge total | 7.2s | 13.4s |
| **audio.load (ffmpeg+resample)** | 1.9s | **268.6s** ⚠️ |

#### N=2 vs N=1 放大比 (1spk task, 同一音频对照)

| Stage | N=1 (s) | N=2 (s) | 放大 |
|---|---:|---:|---:|
| ASR encoder backend (CPU) | 94.8 | 165.9 | **1.75x** |
| ASR encoder frontend (CPU) | 14.8 | 24.5 | 1.66x |
| ASR encoder total (CPU) | 109.9 | 190.8 | 1.74x |
| sherpa pipeline (CPU) | 72.0 | 126.6 | **1.76x** |
| sherpa diarize total (CPU) | 75.2 | 129.8 | 1.73x |
| ASR llm_decode (**Metal GPU**) | 37.7 | 42.2 | **1.12x** |
| cluster_merge (CPU) | 5.8 | 7.2 | 1.24x |
| audio.load (CPU) | 1.9 | 1.9 | 1.00x |

#### 硬件 utilization (N=2)

| 指标 | N=1 mean | N=2 mean | N=1 max | N=2 max |
|---|---:|---:|---:|---:|
| GPU HW active residency | 26.4% | **32.4%** | 100% | 100% |
| ANE Power | 0 mW | **0 mW** | 0 mW | 0 mW |
| CPU Power | 18.9 W | **21.8 W** | 29.3 W | 32.4 W |

## 核心结论 (放在所有方向之前)

1. **CPU stage (ASR encoder + sherpa pipeline) 在 N=2 时一致放大 ~1.75x**, 这正是用户观察到的 "CPU 满, GPU 空等" 的量化证据。
2. **GPU 阶段 (LLM decode) 在 N=2 时几乎不放大 (1.12x)** — Metal pipeline 不被多 worker 抢, **GPU 仍有 ~67% 闲置**。
3. **ANE 完全没用**, baseline 在统计意义上等价于一个 "0 GPU, 0 ANE" 的纯 CPU + 30% GPU 系统。
4. **4spk task 的 audio.load 异常 (268s)** — 待 N=1 4spk 数据对照, 决定是 m4a 本身的 ffmpeg 解码慢, 还是 N=2 抢 CPU 拖出来。
5. **优化的 N=2 wall 上限**: 如果 CPU 放大降到 1.0x (理想异构调度), 1spk wall 应从 244s 降到 140s, 4spk wall 应从 493s 降到 ~350s, **总 wall 从 495s 降到 ~360s**, 节省 ~27%。
6. **ANE 起来的边际收益**: 如果 ANE 能跑 sherpa segmentation/embedding (~5-10W), CPU 释放, **可能进一步降 ~15%**。

## 方向价值预估 (基于 baseline)

| 方向 | 直觉收益 | 风险 |
|---|---|---|
| **方向 3 (num_threads=4)** | 减少 CPU 过订阅, 估算 wall ↓10-15% | 单 task 可能略慢 |
| **方向 1 (ASR encoder CoreML)** | 165s → ?s, 若 ANE 能跑 → wall ↓20-25% | ANE op fallback 风险, CoreML 第一次 compile 慢 |
| **方向 2 (sherpa CoreML)** | 126s → ?s, 若 ANE 能跑 → wall ↓15-20% | sherpa CoreML 真实加速比未知 |
| **方向 1+2 组合** | 理论叠加, 可能逼近 GPU/ANE 全开 | CoreML 两个模型挤 ANE 是否冲突未知 |

> N=1 4spk 跑完后会补 audio.load 的解谜数据。
