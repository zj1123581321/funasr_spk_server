# Spike: Qwen3 silence-aware 时间戳对齐 — 结论

> **状态 (2026-05-17)**: ✅ 已携进 `src/`（默认开启）。
> - 算法位等价复现 spike 数据: 60s podcast +19.14pp / 60min long +33.32pp
> - 60s podcast RTF 实测 +1% (+95ms on 9.16s baseline)
> - 接入点: `src/core/qwen3_transcriber.py::apply_silence_align_to_segments` (helper) + `src/core/qwen3/merge.py::snap_segments_to_silence` (算法)
> - VAD 工具独立: `src/utils/silence_detect.py::ffmpeg_speech_regions`
> - Env 关闭: `FUNASR_QWEN3_SILENCE_ALIGN_ENABLED=false`
> - 完整 env 列表见 `src/core/config.py::Qwen3Config` 的 `silence_*` 字段

**结论**: PoC 验证有效, **推荐携进 src/**, 改造范围小, 风险可控。

## 一句话总结

在 `src/core/qwen3/merge.py::merge_asr_chunks_and_diarize` 输出后追加一个 **snap-to-silence** 后处理: 复用现成 `ffmpeg silencedetect` 拿静音边界, 把段切点吸附到最近 silence 中点(容忍 2s 内)。**不引入任何模型**, 60s 短样本 align_ratio 从 **54.55% → 73.68%** (+19pp), 60min 长音频从 **27.41% → 60.73%** (+33pp)。

## 三大目标对照

| 目标 | 影响 |
|---|---|
| **转录准确率** | 0 影响 — 不动 ASR 文本, 仅调时间戳 |
| **说话人区分准确度** | 0 影响 — 不动 diarize, 仅吸附 segment.start/end 时间 |
| **资源占用 / 并发** | 几乎 0 — ffmpeg silencedetect 在 60min 上 < 2s, 整体 RTF 影响 <1% |

## 关键数据

### baseline vs v2 align_ratio

| 样本 | baseline | v2 (tol=2.0) | delta |
|---|---|---|---|
| podcast_2speakers_60s.wav | 54.55% | 73.68% | **+19.14pp** |
| audio_2spk_60min.mp3 (60 分钟) | 27.41% | 60.73% | **+33.32pp** |

**长音频提升更大** — 因为长音频 chunk 数多(91 vs 2), chunk 硬边界引入的"切在话中间"问题被放大, silence-aware 改造收益叠加。

### tolerance sweep (60s 短样本)

| tol | align_ratio | snap_starts | snap_ends | skip_zero | min_dur |
|---|---|---|---|---|---|
| 0.3 (baseline) | 54.55% | 0 | 0 | 0 | 0.52s |
| 1.0 | 66.67% | 1 | 4 | 1 | 0.82s |
| **2.0 (推荐)** | **73.68%** | **3** | **6** | **2** | **0.82s** |
| 3.0 | 83.33% | 5 | 7 | 2 | 0.82s |
| 5.0 | 88.89% | 5 | 8 | 2 | 0.82s |

边际递减出现在 tol=3.0 之后, 2.0 是性价比拐点。

### tolerance sweep (60min 长音频)

| tol | align_ratio | snap_s | snap_e | skip_zero |
|---|---|---|---|---|
| 0.3 (baseline) | 27.41% | 11 | 9 | 0 |
| **2.0 (推荐)** | **60.73%** | **99** | **94** | **11** |
| 3.0 | 69.58% | 118 | 111 | 12 |
| 5.0 | 80.69% | 146 | 134 | 24 |

## 实现要点

### 1. VAD 参数必须改

**生产默认 `noise=-35dB / min_silence_sec=0.8s` 在 podcast 场景检测不到任何 silence**(0 个 region)。
spike sweep 结果: **`-25dB / 0.2s` 是 sweet spot** —
- podcast 60s: 32 个 silence intervals, 12 个 golden 切点覆盖 9 个 (75%)
- 60min: 1216 个 speech regions, silence 占总时长 10.4%

未覆盖的 25% 是"0 停顿硬切"(同一时间戳切换说话人), 是 VAD 物理上限。

### 2. 算法: 独立 snap start/end 而非边界吸附

第一版我用"相邻段 boundary 统一吸附"(end[i] = start[i+1] = target), 实测**触发 0 时长段**。

原因: **baseline 输出本身存在 diarize overlap**(例如 segment_9: spk0 [46.98~48.09], segment_10: spk1 [47.79~55.52], 时间重叠 0.30s, 典型"插话"场景), 强行同步 end/start 等于强行解决 overlap → 段被压缩到 0 长度。

**修正算法**: 独立 snap 每个 segment 的 start 和 end, 不要求段连续。配合 `min_segment_dur=0.1s` 预检, snap 后段时长会 < 阈值则回退该 snap。

### 3. baseline 已有 0 时长段问题(spike 范围外)

60min 数据上, baseline 输出本身 min_dur=0.000s — 不是 spike 引入, 是 production `merge_asr_chunks_and_diarize` 自己产出的退化段。

**建议**: 推荐在搬入 src/ 时一并修, 但单独提 PR, 不要混在 silence-aware 改造里。

## 搬进 src/ 的最小改动

1. **`src/core/qwen3/merge.py`**: 加 `snap_segments_to_silence()` + `silence_intervals_from_speech()` 函数(从 `spikes/qwen3_silence_align/align_lib/merge_v2.py` 抄)
2. **`src/core/qwen3_pool_transcriber.py`** 或 **`src/core/qwen3_transcriber.py`**:
   - ASR 调用前一次性算出整段 `speech_regions`(用现成 `ffmpeg_speech_regions`)
   - 在 `merge_asr_chunks_and_diarize` 输出后调 `snap_segments_to_silence`
3. **`src/core/config.py`**: 新增 Qwen3Config 字段
   - `silence_align_enabled: bool = True` (默认开)
   - `silence_align_tolerance_sec: float = 2.0`
   - `silence_align_min_segment_dur_sec: float = 0.1`
   - `silence_vad_noise_db: str = "-25dB"`
   - `silence_vad_min_silence_sec: float = 0.20`
4. **env override**: 对应 `FUNASR_QWEN3_SILENCE_*`
5. **测试**: 加 unit test 覆盖
   - `silence_intervals_from_speech` 边界 case
   - `snap_segments_to_silence` 0 时长保护
   - `merge_v2` 端到端 (用 fixture intermediate JSON)
6. **Parity**: 必跑 `FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/` 确认 FunASR 路径无回归 + 改 Qwen3 golden 或加 `FUNASR_QWEN3_SILENCE_ALIGN_ENABLED=false` 跑老 golden

## 风险点

| 风险 | 缓解 |
|---|---|
| ffmpeg silencedetect 在某些音质差的音频上漏检 | 默认参数已是 sweep 最敏感稳定组合 (-25dB/0.2s); 可 env override |
| snap 把段时长吸偏导致 SRT 字幕看着"延迟"几秒 | tol=2.0 是用户决定的平衡值; min_segment_dur=0.1 保护; 可 env 关闭 |
| 60min 上 align_ratio 只到 60% 不够好 | 已知物理上限: 91 个 chunk 边界中相当一部分落在 ≥5s speech 区间; 要进一步提升需更激进策略(改 chunk_size 或上 forced aligner), 不是本 spike 范围 |

## 不做的事

- ❌ **不上 forced aligner 0.6B**: 资源代价大, 当前 +33pp 提升已经够用
- ❌ **不替换 sherpa-onnx diarize**: 这是 Spike A 的范围, 跟时间对齐解耦
- ❌ **不重写 chunk 内文字切分逻辑**: 复杂度高、收益有限; snap-to-silence 已是最小改造的 80% 价值方案
- ❌ **不集成 LattifAI**: 见 `docs/ref/` 调研, 引入云依赖且核心价值(Lattice-1)我们用不上

## 文件清单

```
spikes/qwen3_silence_align/
├── README.md
├── SUMMARY.md                                 (本文件)
├── align_lib/
│   ├── __init__.py│   ├── metric.py                              核心 metric 算法
│   └── merge_v2.py                            核心 snap-to-silence 算法
├── scripts/
│   ├── dump_intermediate.py                   一次性 dump 中间数据
│   ├── baseline_eval.py                       跑现网 merge 算 baseline
│   └── v2_eval.py                             跑 v2 + 对比 baseline
├── data/
│   ├── podcast_60s.intermediate.json
│   └── audio_2spk_60min.intermediate.json
└── output/
    ├── baseline.json / baseline.srt           60s baseline 段 + SRT
    ├── v2.json / v2.srt                       60s v2 段 + SRT
    ├── baseline_60min.json / baseline_60min.srt
    └── v2_60min.json / v2_60min.srt
```

## 下一步建议

按优先级:

1. **(高)** 携进 src/ — 改动小, 收益清晰, env flag 可关
2. **(中)** 修 baseline 自身的 0 时长段问题(独立 PR)
3. **(低)** Spike A: 说话人区分调参 — 跟时间对齐解耦, 可以并行推进
