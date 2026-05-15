# Qwen3-ASR + 说话人区分 PoC

**目标**: 在 M1 Max 64GB 上把 "Qwen3-ASR 1.7B 离线高精度文件转录 + 说话人区分" 跑起来,RTF < 0.15,ANE/GPU 分工不抢资源。

**当前状态**: spike 阶段,代码不进 `src/` 主干。

## 架构

```
        [音频文件]
              │
   ┌──────────┴──────────┐
   ▼                     ▼
[ASR 主路]            [Diarize 副路]
qwen_asr_gguf         sherpa-onnx
- int4 ONNX encoder   - pyannote-seg-3.0 ONNX
- q4_k/q5_k GGUF      - 3D-Speaker eres2net ONNX
- Forced Aligner      - AHC 聚类
(Metal GPU)           (CoreML/CPU EP)
   │                     │
   ▼                     ▼
[ForcedAlignItem 词级时间戳]  [{start, end, speaker_id}]
   └──────────┬──────────┘
              ▼
         [Merger]
   把 word 时间戳贴到 speaker turn
              ▼
       JSON / SRT / RTTM
```

## 复用关系

- **引擎代码**: `src/vendor/qwen_asr_gguf` → symlink 自 `~/Production/qwen_asr_server/core/server/engines/qwen_asr_gguf/`
  - upstream: HaujetZhao/CapsWriter-Offline
  - 这是引擎库层面的复用,不是 service 层面 — production 把它配成流式听写,PoC 把它配成文件转录
- **模型权重**: `models/production_models` → symlink 自 `~/Production/qwen_asr_server/models/`
  - `qwen3_asr_llm.gguf` 1.47GB (q5_k_m 或 q6_k,待 gguf-dump 确认)
  - `qwen3_asr_encoder_*.onnx` 共 ~606MB (int4 量化)
- **sherpa diarization 模型**: `models/sherpa/` (PoC 独立下载,~30MB)

## 跑

```bash
source venv/bin/activate

# ASR
python benchmark/asr_bench.py tests/fixtures/audio/晚点聊-sample-2person-5min.mp3 \
    --out-json output/asr.json

# Diarization
python benchmark/diarize_bench.py tests/fixtures/audio/晚点聊-sample-2person-5min.mp3 \
    --num-speakers 2 --out-json output/diarize.json

# 端到端融合
python benchmark/e2e_bench.py tests/fixtures/audio/晚点聊-sample-2person-5min.mp3
```

## 验收线

- RTF < 0.15 端到端(5min 样本)
- ANE / GPU 分工 — diarize 不抢 ASR 的 Metal GPU
- 输出带词级 timestamp 的 JSON,2 个说话人不串
