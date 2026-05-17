# Qwen3 silence-aware 时间戳对齐 spike

**目标**: 把 Qwen3 引擎 segment 时间戳的精度上限从 40s chunk 边界(粗) 提升到 silence 切点(秒级),不引入任何额外模型。

**背景**: 当前 `src/core/qwen3/merge.py::merge_asr_chunks_and_diarize` 在每个 40s ASR chunk 内,按 diarization turn 时长**比例线性**切分文字 —— 假设语速恒定,实际语速不均时段切点会偏。

**思路**:
1. 在 ASR 调用之前,用 `ffmpeg silencedetect` 一次性算出整段音频的静音边界(成本可忽略)
2. 在每个 40s chunk 内,用静音切点把 chunk 文字切成 micro segments
3. 每个 micro segment 的时间 = 相邻两个 silence 切点之间;归属的 speaker = 时间窗口内 overlap 最大的 diarization turn
4. 段的 start/end **吸附到最近的 silence 切点**

**为什么不上 forced aligner 0.6B**:
- 当前已有 40s chunk 时间戳(免费,ASR 自带)
- VAD silence 边界(免费,ffmpeg 几秒搞定)
- 两个组合够把分辨率从 40s 降到秒级,不增加任何模型负担

## 目录结构

```
spikes/qwen3_silence_align/
├── README.md           本文件
├── .gitignore
├── align_lib/          (避免与项目 src/ 命名冲突, 故改名)
│   ├── __init__.py│   ├── metric.py       自动 metric: 段切点落在 silence 区域的比例
│   └── merge_v2.py     silence-aware merge 函数(PoC)
├── scripts/
│   ├── dump_intermediate.py  跑现网 engine 拿到 chunks + turns,保存 JSON
│   ├── baseline_eval.py      现有 merge → metric (baseline)
│   └── v2_eval.py            silence-aware merge → metric,对比 baseline
├── data/                JSON 中间数据(chunks/turns/speech_regions)
└── output/              段对比结果 + SRT
```

## 复用关系

- 音频样本: `tests/fixtures/audio/podcast_2speakers_60s.wav`(短样本快速迭代)+ `tmp_long_audio/eval_set/audio_149min.*`(长音频最终验证)
- ffmpeg_speech_regions: 从 `tests/manual/server/qwen3_long_audio_poc.py:70` 抄过来(spike 内自包含)
- ASR engine: 复用 `src.core.qwen3.asr.transcribe_offline`(只跑一次拿中间数据)
- diarize: 复用 `src.core.qwen3.diarize`(同上)
- baseline merge: 直接 import `src.core.qwen3.merge.merge_asr_chunks_and_diarize`

## 跑

```bash
# 在仓库根目录, 用 venv python(spike 共用项目 venv, 不另起)
venv/bin/python spikes/qwen3_silence_align/scripts/dump_intermediate.py   # 跑一次, 后续迭代不用再跑
venv/bin/python spikes/qwen3_silence_align/scripts/baseline_eval.py
venv/bin/python spikes/qwen3_silence_align/scripts/v2_eval.py
```

## 验收线

- 自动 metric `align_ratio` (段切点落在 silence 区域的比例):
  - baseline 预期 <50%(假设)
  - v2 目标 **≥95%**
- RTF 影响可忽略(VAD < 1% 总耗时)
- 短样本通过后,在 149min 长音频上验证同样有效

## 状态

spike 阶段, **代码不进 `src/` 主干**。验收后再决定是否搬。
