# 方向 1: ASR encoder CoreML/ANE 路由 — ⚠️ 部分成功 (仅 frontend, backend 卡 op 兼容)

## 实验全部 variants

| Tag | Variant | sess_fe | sess_be | 跑通? | N=1 wall | Δ vs baseline |
|---|---|---|---|---|---|---|
| baseline | CPU-only | CPU | CPU | ✅ | 158.9 | — |
| d1 | ALL + MLProgram (fe+be) | CoreML+CPU | CoreML+CPU | ❌ | — | axis 4 报错 |
| d1a | ALL + MLProgram + only_fe | **CoreML+CPU** | CPU | ✅ | 152.7 | -3.9% (frontend 快, 但 llm_decode +73s 抵消) |
| d1b | ALL + NeuralNetwork (fe+be) | CoreML(silent fb) | CoreML(silent fb) | ✅ | 155.3 | -2.3% (CoreML EP 没真生效, 全 CPU) |
| d1c | CPUAndNeuralEngine + MLProgram (fe+be) | — | — | ❌ | — | axis 4 报错 (跟 GPU 无关) |
| **d1d** | **CPUAndNeuralEngine + MLProgram + only_fe** | **CoreML+CPU** | CPU | ✅ | **149.2** | **-6.1%** ⭐ |

## d1d (winning N=1 variant)

```bash
venv/bin/python spikes/qwen3_mac_hw_accel/run_baseline.py \
    --mode n1 --tag d1d_fe_cpuane_t8_n1_1spk \
    --num-threads 8 --provider cpu --onnx-provider COREML \
    --enable-coreml-asr-patch \
    --coreml-units CPUAndNeuralEngine --coreml-format MLProgram \
    --coreml-only-frontend --audio-pick 1spk
```

| 指标 | baseline | d1a (ALL+fe) | **d1d (cpuane+fe)** | 解读 |
|---|---:|---:|---:|---|
| wall | 158.9 | 152.7 | **149.2** | d1d 最优 |
| frontend_onnx | 14.8 | 6.2 | **5.9** | ANE 加速 ~60% |
| backend_onnx (CPU) | 94.8 | 23.1⚠️ | 93.3 | d1a 异常, d1d 正常 |
| **llm_decode** | 37.7 | **110.8** ❌ | **37.3** ✅ | d1a 抢 Metal 慢 3x, d1d 排 GPU 后正常 |
| GPU residency mean | 26.4% | 35.9% | 26.2% | d1d 跟 baseline 一致 |
| **ANE Power mean** | 0 mW | 43 mW | **48 mW** (max 334) | ANE 真启用 |
| RTF | 0.162 | 0.154 | **0.150** | d1d 最优 |

### 核心 mechanism 发现

1. **ASR encoder backend ONNX 在 CoreML MLProgram 下报 `axis 4 not in [-4,3]`** — 是某个 op (Slice/Gather 类) 的轴属性在 CoreML EP 转换时校验失败。在 ALL 和 CPUAndNeuralEngine 两种 units 下都触发, 说明跟 GPU/ANE 无关, 是 op-level partition 问题。NeuralNetwork format 不报错但 silent fallback 到 CPU (无收益)。

2. **CoreML ALL units (含 GPU) 会跟 llama.cpp Metal 抢 GPU**: d1a 时 frontend CoreML 把 llm_decode 从 37s 拖到 110s, 净收益被吃掉。

3. **CPUAndNeuralEngine units 完美解耦 llama.cpp**: frontend 走 ANE, GPU 留给 LLM decode, 两者独立加速。

## N=2 数据

| Tag | Wall | 1spk RTF | 4spk RTF | ANE mean | GPU mean |
|---|---:|---:|---:|---:|---:|
| baseline (t=8, CPU) | 495.6 | 0.252 | 0.188 | 0 mW | 32.4% |
| d3 (t=4, CPU) | **438.8** | 0.180 | 0.166 | 6 mW | 35.1% |
| **d1d (t=8, fe ANE)** | 458.6 | 0.233 | 0.173 | **44 mW** | 33.3% |

**N=2 d1d 单变量 wall 458.6s (-7.5%)**, frontend 24.5→10.9s (1spk) / 38.8→15.8s (4spk)。
但 d3 t=4 单变量更好 (-11.5%)。

### 副产物 (N=2 d1d)

- audio.load 异常消失 (268.6→3.8s) — frontend ANE 释放了 CPU, ffmpeg pipe 不再阻塞 (跟 d3 t=4 同效应)

## 待跑组合 d13 (t=4 + frontend ANE)

预期: t=4 让 CPU 不过订阅 + frontend ANE 进一步省 CPU + GPU 不抢, 理论收益可能逼近 -15%~ -18%。

## 结论 (方向 1 待 d13 落地后定论)

- **可行性**: 仅 frontend 可走 ANE, backend 卡 axis 4 (需要 ONNX 模型层修改, PoC 不动)
- **单 frontend ANE 节省 8-30s (依音频长度), N=2 wall 收益 -7.5%**, 比 d3 (-11.5%) 弱
- **必须用 CPUAndNeuralEngine units**, ALL units (含 GPU) 会抢 llama.cpp Metal 资源, 反而拖慢
- **MLProgram format 必选**, NeuralNetwork 时 CoreML EP silent fallback
- 工程化代价: vendor encoder.py 加 CoreML 分支 + Qwen3Config 加 ane_frontend 开关
- 风险: ANE 在 macOS 升级或不同设备 (M2/M3) 上行为可能不同, 需要 fallback CPU 测试

## 后续

- ✅ 跑 d13 组合 (t=4 + frontend ANE), 看叠加收益
- ❌ 不修 backend axis 4 (要重新 export ONNX 模型, 不在 PoC 范围)
- ❌ 不尝试 RequireStaticInputShapes — 跟 axis 4 op-level 问题无关
