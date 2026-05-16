# 方向 3: num_threads 调优 (N=2)

## 实验

固定 N=2 (2 worker process 并发), 跑 1spk-16min + 4spk-44min, 对比不同 num_threads:

| Tag | num_threads | total worker threads | over-subscribe? |
|---|---:|---:|---|
| **b0_cpu_t8_n2** (baseline) | 8 | 16 | yes (16 > 10 cores) |
| **d3_cpu_t4_n2** | 4 | 8 | no (8 ≤ 10 cores) |
| d3_cpu_t12_n2 (TODO) | 12 | 24 | severely yes |

> num_threads 同时作用于 ONNX (ASR encoder) 和 sherpa-onnx (segmentation + embedding) — 通过 `FUNASR_QWEN3_NUM_THREADS` 一个 env 控两边

## 结果

| 指标 | t=8 baseline | t=4 | Δ |
|---|---:|---:|---|
| **TOTAL_WALL** | **495.6s** | **438.8s** | **-56.8s, -11.5%** ✅ |
| 1spk task wall | 244.1s | 174.6s | -69.5s, -28% |
| 4spk task wall | 493.4s | 436.8s | -56.6s, -11% |
| 1spk RTF | 0.252 | 0.180 | -0.072 (更快) |
| 4spk RTF | 0.188 | 0.166 | -0.022 (更快) |
| GPU residency mean | 32.4% | 35.1% | +2.7pp |
| ANE Power mean | 0 mW | **6 mW** (max 142 mW) | 首次出现非零 |
| CPU Power mean | 21.8 W | 19.9 W | -1.9 W |

### Stage breakdown 对比

#### 1spk-16min task (同一音频, N=2 状态)

| Stage | t=8 N=2 | t=4 N=2 | Δ |
|---|---:|---:|---|
| ASR encoder.backend | 165.9 | **110.6** | **-55s, -33%** |
| ASR encoder.frontend | 24.5 | 16.5 | -8s, -33% |
| ASR encoder.total | 190.8 | 127.3 | **-63s, -33%** |
| ASR llm_decode | 42.2 | 37.6 | -5s |
| ASR run_total | 233.0 | 164.9 | -68s |
| sherpa pipeline | 126.6 | **83.7** | **-43s, -34%** |
| sherpa diarize total | 129.8 | 86.8 | -43s |

#### 4spk-44min task

| Stage | t=8 N=2 | t=4 N=2 | 备注 |
|---|---:|---:|---|
| ASR encoder.backend | 60.6 | 269.8 | **t=8 时 audio.load 卡 268s 导致 4spk task 几乎串行后跑 → encoder 没抢资源; t=4 audio.load 正常 (4.2s), 4spk 整段跟 1spk 抢 → encoder 慢** |
| ASR encoder.total | 100.1 | 309.4 | 同上 |
| audio.load | **268.6** ⚠️ | **4.2** ✅ | t=4 修复 audio.load 异常 |
| sherpa pipeline | 262.8 | 209.1 | -54s, -21% |
| sherpa diarize total | 268.5 | 215.0 | -53s |

## 核心发现

1. **t=4 是 easy win**: 11.5% wall 节省, 完全免费 (改一个 env), 风险 0。
2. **t=8 baseline 的 495s 是"被 audio.load 异常人为错峰"造成的伪并发数据**。t=4 才是"真两 task 并发" baseline。
3. **audio.load 异常的根因**: t=8 时 2×8=16 thread 抢 10 core, ffmpeg subprocess 解码 m4a 时 Python read pipe 被严重 CPU starve, 解码 44min 音频从 3-5s 拖到 268s。**这是个真正的 CPU 过订阅 hard evidence**。
4. **ANE 首次出现非零信号 (max 142mW)** — 但 mean 仅 6mW, 大概率是 sherpa segmentation onnxruntime 在 t=4 时换了 thread plan 触发的极小 CoreML 路径, 不是工程级使用。
5. **GPU residency 微升 2.7pp** — t=4 减少 CPU 抢, GPU 等 CPU 的时间略减。

## 建议

- **生产 default 改 `FUNASR_QWEN3_NUM_THREADS=4`** ✅ 必做
- 同时把 `pool_size` 保持 2 (跟核数对齐: 2×4=8 ≤ 10)
- 若未来支持 pool_size=3, 用 `t=3` (3×3=9 ≤ 10)

## 待跑 (低优先级)

- d3 t=12 N=2 — 验证过订阅极端会多慢, 拍 over-subscribe 上界
- d3 t=2 N=2 — 验证 t=2 是不是更好 (2×2=4 ≤ 10, 剩 6 core 给系统)
