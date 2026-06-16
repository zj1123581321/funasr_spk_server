# PoC 实验矩阵

每个 cell 跑一次 `run_baseline.py`, 产物在 `runs/<tag>/summary.json`。

| Tag | Mode | N | num_threads | sherpa provider | ASR onnx | CoreML asr patch | 目的 |
|---|---|---|---|---|---|---|---|
| **b0_cpu_t8_n1_1spk** | n1 | 1 | 8 | cpu | CPU | no | N=1 单跑 1spk 16min stage breakdown |
| **b0_cpu_t8_n1_4spk** | n1 | 1 | 8 | cpu | CPU | no | N=1 单跑 4spk 44min stage breakdown |
| **b0_cpu_t8_n2** | n2 | 2 | 8 | cpu | CPU | no | ⭐ N=2 baseline (生产现状) |
| **d3_cpu_t4_n2** | n2 | 2 | 4 | cpu | CPU | no | 方向 3: num_threads=4 N=2 |
| **d3_cpu_t12_n2** | n2 | 2 | 12 | cpu | CPU | no | 方向 3: num_threads=12 (过订阅) |
| **d1_cpu_t8_n2** | n2 | 2 | 8 | cpu | COREML | yes | 方向 1: ASR encoder CoreML |
| **d2_cpu_t8_n2** | n2 | 2 | 8 | coreml | CPU | no | 方向 2: sherpa coreml |
| **d12_cpu_t8_n2** | n2 | 2 | 8 | coreml | COREML | yes | 方向 1+2 组合 |
| **d123_cpu_t4_n2** | n2 | 2 | 4 | coreml | COREML | yes | 全栈最优候选 (1+2+3 t=4) |

## 决策点
- 跑完 b0 系列, 看 N=1 vs N=2 stage 放大比例, 决定 d1/d2/d12 是否值得 (放大最大的方向 ROI 最高)
- d1 跑完看 asitop/powermetrics GPU/ANE 是否真活动 (CoreML 默默 fallback CPU 是常见坑)
- 若 d1/d2 任何方向无效, 砍 d12/d123
- 若 d3 t=4 显著优于 t=8, 把 t=4 作为组合默认

## 评估指标
- **wall (N=2 总耗时)** — 用户感知
- **单 task RTF** — task 内部
- **per-stage elapsed** — 哪段被并发拖慢
- **GPU/ANE active residency** (powermetrics) — 硬件路由是否真生效
