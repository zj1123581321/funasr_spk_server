# 方向 2: sherpa-onnx 切 CoreML provider — **❌ 负面结果**

## 实验

N=2 t=8 跑 1spk-16min + 4spk-44min, 唯一变量: `provider=coreml` (sherpa-onnx 的 segmentation + embedding 都切 coreml)。

```bash
venv/bin/python spikes/qwen3_mac_hw_accel/run_baseline.py \
    --mode n2 --tag d2_sherpa_coreml_t8_n2 \
    --num-threads 8 --provider coreml --onnx-provider CPU
```

## 结果

| 指标 | baseline (sherpa=cpu) | d2 (sherpa=coreml) | Δ |
|---|---:|---:|---|
| **TOTAL_WALL** | **495.6s** | **702.1s** | **+206.5s, +41.7% (慢!)** ❌ |
| 1spk task wall | 244.1s | 295.4s | +51.3s (+21%) |
| 4spk task wall | 493.4s | 699.2s | +205.8s (+42%) |
| 1spk RTF | 0.252 | 0.304 | +0.052 (慢) |
| 4spk RTF | 0.188 | 0.266 | +0.078 (慢) |
| **sherpa pipeline (1spk)** | 126.6s | **197.9s** | **+71s (+56%)** |
| **sherpa pipeline (4spk)** | 262.8s | **456.3s** | **+193s (+73%)** |
| sherpa embedding (4spk cluster_merge) | 11.4s | 25.6s | +14s (+126%) |
| ASR encoder.backend (4spk) | 60.6 | **516.1s** | **+455s** ⚠️ (连带受害) |
| GPU residency mean | 32.4% | **24.9%** | -7.5pp (反降) |
| **ANE Power mean** | 0 mW | **37 mW** (max 108) | 首次明显 |
| CPU Power mean | 21.8 W | 17.2 W | -4.6W (CPU 没那么累) |

## 解读

1. **ANE 确实启动了** (mean 37mW, max 108mW) — sherpa-onnx 的 CoreML provider 真的把模型送到了 ANE/CoreML 路径。
2. **但加速效果是负的**: sherpa pipeline 慢 56-73%, embedding compute 慢 126%。说明:
   - 这两个模型 (pyannote-segmentation-3.0 + nemo-titanet-small) 在 CoreML EP 上**单个 inference 时间 > CPU**, 而不是更短
   - CoreML EP 的 overhead (model compile, IO copy, ANE handoff) 大于在 ANE 上跑的计算节省
3. **连带 ASR encoder 也变慢**: 4spk task encoder.backend 从 60.6s 跳到 516.1s。同 worker process 中 sherpa CoreML 和 ASR ONNX CPU EP 都跑在 onnxruntime, **CoreML EP 占用了系统级 CoreML/Metal 资源, 让同一个 worker process 的 CPU EP 也慢**。
4. **GPU residency 反而下降** (32% → 24%) — 这非常反直觉, 但解释是: sherpa CoreML 路径走 CoreML stack (CPU+GPU+ANE 混合), 跟 llama.cpp Metal LLM decode 抢 GPU, 但 LLM decode 阶段反而被压抑。

## 假说: 为什么 sherpa CoreML 比 CPU 慢?

- pyannote segmentation 模型是个小的 transformer (10MB 量级), 在 CPU 上 8 thread 跑很快 (~1ms/segment)
- 在 ANE/CoreML 上, 每次 inference 有 ~1-2ms 的 dispatch overhead, **小模型 + 多次调用** 时 overhead 大于计算
- nemo-titanet-small 同理: ResNet 类小模型, CPU 友好
- 这也是为什么 prompt 里 gemini.md 把 WeSpeaker 推荐到 ANE — 它指的是**单个长 inference** 的场景, 不是 sherpa 这种**多次小 inference** 的 pipeline

## 结论

- **方向 2 拒收**, 在生产 absolutely 不能开 `FUNASR_QWEN3_PROVIDER=coreml`
- 这个负面数据有价值: 印证了"小模型 + 多次调用"不适合 CoreML/ANE 路由
- 如果未来要让 sherpa 走 ANE, 必须换更"重"的模型 (e.g. 替换 nemo-titanet-small 为更大的 embedding, 让 ANE 摊薄 dispatch overhead) — 这是工程化级改造, 不在 PoC 范围

## 后续

- ❌ 不跑 d12 (sherpa+ASR 组合) — 已知 d2 负面, d12 不会变好
- ❌ 不跑 sherpa CoreML 的 t=4 variant — 即使省 CPU 抢, 单 inference 慢这件事不会变
