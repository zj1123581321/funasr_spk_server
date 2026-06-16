# 多人 speaker 评测集 (eval_set)

PR4 长音频 speaker 识别评测用。统一时长 ≤ 60min，覆盖 1/2/4/6 speaker 场景。
所有原始音频保留在本目录，方便以后用新算法跑同一份对照。

## 文件清单

| 文件 | 来源 | 原始时长 | 此处时长 | 真实人数 | 备注 |
|---|---|---|---|---|---|
| `audio_1spk_real.m4a` | 小宇宙 `media.xyzcdn.net/...loaJUIrdyFbLPtNimBAhn31CMimn.m4a` | 16.2min | 16.2min (不截) | **1** | 杨涛个人频道独白 (微习惯主题) — 真正的单人样本 |
| `audio_panel_marked_1spk.m4a` | 喜马拉雅 `aod.cos.tx.xmcdn.com/...GKwRIMAN3CtaAPyZswSZWFfG.m4a` | 34.1min | 34.1min (不截) | 3+ (标"1人"但实际多人) | 汽车类访谈，含"四零""六三""老编辑""小刚"等多人；保留作为"标错的单人样本"对比 |
| `audio_2spk_60min.mp3` | 项目内 `tmp_long_audio/audio_149min.mp3` 前 60min | 149.0min | 60.0min (ffmpeg `-t 3600 -c copy`) | 2 | 吴明辉 × 程曼祺 双人访谈 |
| `audio_4spk.m4a` (symlink) | `tmp_long_audio/multi_speaker_test/podcast_4spk.m4a` | 43.8min | 43.8min (不截) | 4 | 女性话题 4 人圆桌 (田雨、希拉里 等)，结尾含英文歌 |
| `audio_6spk_60min.m4a` | `media.xyzcdn.net/...ltsmlLxssBYRflRWppKucHUVk6I8.m4a` 前 60min | 113.8min | 60.0min | 6 | 小宇宙播客 6 人对话 |

## 评测 pipeline 阶段（每个样本三阶段输出）

```
<audio>
  ├─ {label}_pipeline/baseline/<name>.qwen3_long_poc.json    # Qwen3 ASR + sherpa global diarize 原始输出
  ├─ {label}_pipeline/merged/<name>.qwen3_long_poc.json      # + cluster centroid merge (新)
  └─ {label}_pipeline/final_v12_d1.5/<name>.qwen3_long_poc.json  # + v12 short-segment guard (d=1.5)
```

每阶段还配 `run.log` 记录命令、参数、耗时。

## 当前评测结果 (2026-05-16)

> **Note**: 本表数据来源是 `tests/manual/server/qwen3_long_audio_poc.py` 三阶段手工脚本(直接 mp3/m4a 读, 不走 production worker)。Production pipeline (`Qwen3PoolTranscriber`) 曾在 `cd578a8` (2026-05-15 worker 加 ffmpeg 转换) 引入 2spk_60min over-detect 回归 (final 4 spk), `fix/qwen3-spk-overdetect` PR (commit `bbdb173` + `a843b27`, 2026-05-17) 修复后跟本 baseline 完全对齐。详见调研报告 `docs/开发/archive/spk-over-detect-归因调研结果.md` §9, production 路径回归测试见 `tests/integration/test_qwen3_spk_overdetect_fix.py`。

| 样本 | 真实人数 | baseline 检测 | merged | final | 段数 | <1s% | 评估 |
|---|---:|---:|---:|---:|---:|---:|---|
| 1spk_real | **1** | 1 | 1 | **1** | 10 | 0.0% | ✓ **完美单人** |
| panel (标"1人"实多人) | 3+ | 5 | 3 | 3 | 249 | 1.2% | ✓ 与实际人数一致 |
| 2spk | 2 | 2 | 2 | **2** | 94 | 0.0% | ✓ |
| 4spk | 4 | 9 | 5 (4 真人 + 1 音乐) | **5** | 257 | 0.0% | ✓ |
| 6spk | 6 | 12 | 6 | **6** | 469 | 0.2% | ✓ |

**当前算法对 1/2/3+/4/6 speaker 全场景准确识别 speaker 数。**

### Final speaker 分布

**1spk_real** (16.2min, 真实 1)：
- Speaker1 15.92 min 100.0% (10 段，median 89.79s，0 切换)

**panel** (34.1min, 标"1人"但实际 3+人)：
- Speaker1 (主持人 / 占主导的男声) 23.05 min 74.9%
- Speaker26 (嘉宾 A / 女声) 4.40 min 14.3%
- Speaker9 (嘉宾 B / 另一男声) 3.32 min 10.8%

**2spk** (60min, 真实 2)：
- Speaker1 (吴明辉) 52.17 min 87.5%
- Speaker3 (程曼祺) 7.43 min 12.5%

**4spk** (43.8min, 真实 4)：
- Speaker2 15.15min 36.3% / Speaker15 14.17min 34.0% / Speaker3 7.17min 17.2% / Speaker7 4.28min 10.2%
- Speaker66 (英文歌) 0.96min 2.3% ← 算法识别非语音保留独立 cluster

**6spk** (60min, 真实 6)：
- Speaker41 18.77min 32.9% / Speaker20 9.65min 16.9% / Speaker74 8.86min 15.5%
- Speaker23 7.68min 13.5% / Speaker13 6.35min 11.1% / Speaker46 5.74min 10.1%

### 性能 (RTF)

| 样本 | 总耗时 | RTF | ASR RTF | Diarize RTF |
|---|---:|---:|---:|---:|
| 1spk_real 16.2min | 157.8s | 0.163 | 0.078 | 0.080 |
| 4spk 43.8min | 417.9s | 0.159 | 0.084 | 0.072 |
| 2spk 60min | 613.8s | 0.170 | 0.088 | 0.080 |
| 1spk_panel 34.1min | 348.0s | 0.170 | 0.087 | 0.078 |

merge + v12 后处理共耗时 < 5 秒 (RTF < 0.001)，可忽略。

## 复现命令

```bash
unset TMPDIR; export TMPDIR=/tmp
export DYLD_LIBRARY_PATH="$PWD/src/core/vendor/qwen_asr_gguf/inference/bin"

# 串行跑 4 个样本完整 pipeline (见 commit 历史里的 shell 脚本)
# 单个样本三阶段:
#   1. baseline (Qwen3 ASR + sherpa diarize)
venv/bin/python tests/manual/server/qwen3_long_audio_poc.py \
  tmp_long_audio/eval_set/audio_4spk.m4a \
  --out-dir tmp_long_audio/eval_set/4spk_pipeline/baseline \
  --target-min 12 --soft-max-min 15 --hard-max-min 20 \
  --boundary-source ffmpeg-silence --min-silence-sec 0.8

#   2. cluster centroid merge (修复 sherpa 多人过度聚类)
venv/bin/python tests/manual/server/merge_qwen3_minor_clusters.py \
  tmp_long_audio/eval_set/4spk_pipeline/baseline/podcast_4spk.qwen3_long_poc.json \
  tmp_long_audio/eval_set/audio_4spk.m4a \
  --out-json tmp_long_audio/eval_set/4spk_pipeline/merged/podcast_4spk.qwen3_long_poc.json \
  --min-main-share 0.03 --relabel-threshold 0.55 --merge-threshold 0.78 \
  --dominant-share 0.6 --dominant-merge-threshold 0.6

#   3. v12 short-segment guard (段级清理)
venv/bin/python tests/manual/server/postprocess_qwen3_short_segment_guard.py \
  tmp_long_audio/eval_set/4spk_pipeline/merged/podcast_4spk.qwen3_long_poc.json \
  --out-json tmp_long_audio/eval_set/4spk_pipeline/final_v12_d1.5/podcast_4spk.qwen3_long_poc.json \
  --short-drop-sec 1.5 --aba-max-mid-sec 1.5

# 分析 speaker 分布
venv/bin/python tests/manual/server/analyze_multispeaker_output.py \
  tmp_long_audio/eval_set/4spk_pipeline/final_v12_d1.5/podcast_4spk.qwen3_long_poc.json
```

## 注意

- 这些音频文件**不入 git**（音频和模型在 `.gitignore` 里）。
- 后续跑新算法时，**不要删除已有 baseline JSON 输出**，可以直接复用 baseline 跑不同的 merge/post-process 变体对照。
- 2spk 是 149min 截取版（前 60min），完整版仍在 `tmp_long_audio/audio_149min.mp3`。
- 评测没有 reference 校对稿（除 2spk 有 149min 完整版的 `reference_149min_calibrated.txt`），多人场景只能用"检测 speaker 数 vs 真实"+ 段长分布 + 人工抽查文本一致性来评。
