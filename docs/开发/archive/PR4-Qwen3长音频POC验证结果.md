# PR4 Qwen3 长音频 PoC 验证结果

> 日期: 2026-05-16
> 目标: 验证 Qwen3-ASR-1.7B q5/Metal 路线在 Mac 上处理长音频时, 通过宏分段重置 ASR session, 同时保留全局 diarization speaker id, 是否能兼顾质量与 RTF.

## 结论

这次 PoC 验证通过。

- 长音频不再让 Qwen3 ASR 单 session 吃完整 83/149 分钟, 而是使用 12 分钟目标、15 分钟软上限、20 分钟硬上限的宏分段。
- speaker diarization 仍然全局跑一次, 所以 speaker id 跨 ASR 窗口保持一致; ASR 只负责按窗口重置上下文。
- 149min 历史问题中的 `我这个AI` 死循环和 `decode: failed to find a memory slot` 输出异常, 在本次结果扫描中没有出现。
- q5/Metal ASR 单阶段 RTF 稳定在 0.084-0.089; 端到端 RTF 主要由 diarization 决定。

## 输入音频

| 名称 | 本地路径 | 时长 | 大小 |
|---|---|---:|---:|
| 83min m4a | `tmp_long_audio/audio_83min.m4a` | 4954.2s / 82.6min | 76M |
| 149min mp3 | `tmp_long_audio/audio_149min.mp3` | 8937.9s / 149.0min | 103M |

## 分段策略

PoC 默认参数:

| 参数 | 值 |
|---|---:|
| target | 12min |
| soft max | 15min |
| hard max | 20min |
| min segment | 3min |
| boundary source | `ffmpeg silencedetect` |
| silence threshold | `-35dB`, 0.8s |
| overlap | 0s |

说明:

- `ffmpeg silencedetect` 只作为轻量 VAD 边界代理, 用于把切点移动到自然静音附近。
- 如果目标附近没有可用静音, 回退到 12min 固定切点。
- 83min 音频检测到多个长静音, 生成 8 个宏分段; 149min 音频基本没有可用长静音, 生成 12 个宏分段。

## 实测结果

| 音频 | 宏分段 | speaker | 输出段数 | ASR RTF | diarize RTF | 总 RTF | 总耗时 | 峰值 RSS | 质量告警 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 83min m4a | 8 | 8 | 552 | 0.084 | 0.111 | 0.214 | 1058.3s / 17.6min | 4707MB | 0 |
| 149min mp3 | 12 | 2 | 421 | 0.089 | 0.082 | 0.172 | 1540.8s / 25.7min | 6285MB | 0 |

149min 每个 ASR 窗口的 RTF:

| window | 时长 | ASR RTF | 字符数 | warning |
|---:|---:|---:|---:|---:|
| 0 | 720.0s | 0.086 | 5035 | 0 |
| 1 | 720.0s | 0.086 | 5469 | 0 |
| 2 | 720.0s | 0.089 | 5524 | 0 |
| 3 | 720.0s | 0.097 | 5410 | 0 |
| 4 | 720.0s | 0.089 | 5365 | 0 |
| 5 | 720.0s | 0.089 | 5612 | 0 |
| 6 | 720.0s | 0.089 | 5474 | 0 |
| 7 | 720.0s | 0.087 | 5347 | 0 |
| 8 | 720.0s | 0.087 | 5217 | 0 |
| 9 | 720.0s | 0.090 | 5114 | 0 |
| 10 | 720.0s | 0.088 | 5323 | 0 |
| 11 | 1017.9s | 0.088 | 7352 | 0 |

## 对比 PR3 问题

| 问题 | PR3 表现 | PoC 结果 |
|---|---|---|
| 149min 中后段重复 token | 80min 后间歇退化, 146min 后 `我这个AI` 死循环 | 未复现; quality warning 为 0 |
| KV cache decode warning | worker log 大量 `failed to find a memory slot` | 输出文件扫描未命中 |
| 149min 总 RTF | 0.220, 且质量崩坏 | 0.172, 质量扫描通过 |
| 83min 总 RTF | 0.282 | 0.214 |
| 83min speaker 过分散 | 8 speakers | 仍为 8 speakers, 需要后续调 diarization 聚类/过滤 |

## 工程判断

1. 长音频分段必须保留, 且 12min 是当前机器和模型质量的较好默认值。
2. 不建议把 30-50min 作为默认切片长度; 官方/生态报告里反复提到 20min 上限, 当前实测也证明 12-17min 窗口更稳定。
3. speaker 不应按片段独立 diarize 后再做跨段聚类, 至少当前 PoC 阶段应保留全局 diarization, 这样 speaker id 稳定性更高, 工程复杂度也更低。
4. Mac 资源调度上, ASR 与 diarization 是两类资源负载: 当前 sherpa diarization 主要吃 CPU, Qwen3 decoder 主要吃 Metal/GPU。并发 worker 时应避免多个 ASR decoder 同时争抢 Metal, 更适合做 GPU token 桶/信号量排队, CPU diarization 则可以有限并行。
5. 83min speaker 过分散不是 ASR 分段导致, 因为 diarization 是全局跑的; 后续应聚焦 cluster threshold、短 speaker 过滤、已知说话人数提示或二次 speaker embedding 合并。

## 后续准确率迭代

149min 音频对校对稿评估后, 已把初版的线性切字融合升级为 Qwen3-ASR 40s internal chunk 级融合:

| 版本 | 文本准确率 | speaker 字符准确率 | segment 多数派准确率 | 说明 |
|---|---:|---:|---:|---|
| baseline | 90.89% | 90.54% | 74.35% | 12min 窗口内按 turn 时长线性切字 |
| v4 full | 91.04% | 96.89% | 92.67% | 全流程 diarization + ASR + chunk merge |
| v5 fused | 91.12% | 96.80% | 92.17% | 当前最均衡可查看结果 |
| v6 guard | 91.01% | 96.88% | 92.81% | 增加短 turn 标点搜索保护 |

输出路径:

- `tmp_long_audio/poc_outputs_v5_fused/audio_149min.qwen3_long_poc.json`
- `tmp_long_audio/poc_outputs_v4_full/audio_149min.qwen3_long_poc.json`
- `tmp_long_audio/poc_outputs_v6_short_turn_guard/audio_149min.qwen3_long_poc.json`

## 本次产物

- `src/core/qwen3_segment.py`: 长音频宏分段、diarization turn 裁剪、质量告警检测。
- `src/core/qwen3/asr.py`: 新增 `run_asr_window`, 并修正 wrapper 显式使用 engine config 的 `chunk_size` / `memory_num=1`。
- `src/core/qwen3/merge.py`: 新增 Qwen3-ASR internal chunk 与 diarization turn 的 interval join 融合。
- `src/core/qwen3_postprocess.py`: 技术播客术语后处理。
- `tests/manual/server/qwen3_long_audio_poc.py`: 可复现实测的手工 PoC runner。
- `tests/manual/server/fuse_qwen3_poc_speakers.py`: 复现 v5 文本/speaker 融合结果。
- `tests/manual/server/evaluate_qwen3_poc_against_reference.py`: 与 149min 校对稿对比评估。
- `tests/unit/test_qwen3_segment.py`: 分段与质量告警单测。
- `tests/unit/test_qwen3_asr_window.py`: ASR window 加载与 memory config 单测。

## 复现命令

```bash
venv/bin/python tests/manual/server/qwen3_long_audio_poc.py tmp_long_audio/audio_83min.m4a --diarize-threads 4
venv/bin/python tests/manual/server/qwen3_long_audio_poc.py tmp_long_audio/audio_149min.mp3 --diarize-threads 4
venv/bin/python -m pytest tests/unit/test_qwen3_segment.py tests/unit/test_qwen3_asr_window.py
```

输出文件:

- `tmp_long_audio/poc_outputs/audio_83min.qwen3_long_poc.json`
- `tmp_long_audio/poc_outputs/audio_83min.qwen3_long_poc.srt`
- `tmp_long_audio/poc_outputs/audio_149min.qwen3_long_poc.json`
- `tmp_long_audio/poc_outputs/audio_149min.qwen3_long_poc.srt`
