# ort_cuda 短音频 under-detect 修复结案

> 课题来源：`2026-06-10-ort_cuda短音频under-detect-新session-prompt.md`（已结案）。
> 现象：60s 双人播客在 3060 ort_cuda 上只出 1 speaker，Mac sherpa 正常出 2；
> parity IoU 测试在 Mac CPU 上也挂（mean 0.22）→ 算法实现问题，非 CUDA 数值问题。

## 根因（两层叠加，都已修复）

### 第一层（结构）：跨 chunk 平均 powerset logits

pyannote-segmentation-3.0 的 3 个 speaker slot 是 **chunk 局部** 的（跨 chunk
不保证同一人占同一 slot）。旧 ort 复刻用 Whisper-style 跨 chunk 加权平均
logits 再 argmax —— 不同说话人的活动被混进同一 slot，聚合 activity 提出的
turn 含混合语音，per-turn TitaNet embedding 全是"混合脸"，短音频快速对话下
complete linkage 聚成 1 簇。长音频 turn 长、纯度高，所以 5/22 评测侥幸全对。

诊断实证（60s podcast，Mac CPU，同一 `fast_clustering` threshold=0.9）：

| 路径 | 簇数 | share |
|---|---|---|
| sherpa C++ 参照 | 2 | 61% / 39%（turn 时长口径） |
| 旧 ort（per-turn embedding） | 2（退化） | 97% / 3% → filter_spurious 清掉 1.8s 残留 → 1 spk |
| sherpa 式 per-(chunk,speaker) embedding | 2 + 0.3s 微簇 | 69% / 31% |

**FastClustering 本身（cosine distance + complete linkage + cutree 距离阈值）
与 scipy 复刻完全等价，不是根因**——对照过 sherpa C++ `fast-clustering.cc`。

### 第二层（调查中挖出的隐藏 bug）：TitaNet 拿错 ONNX 输出

NeMo TitaNet ONNX 有两个输出 `[logits (B,16681), embs (B,192)]`。
`compute_titanet_embedding` 自实现以来一直 `run(None)[0]` 拿 **训练集分类
logits** 当 speaker embedding 聚类。logits 也带说话人信息所以"能用"，但离散
度大：第一层修复后 2spk_60min 长音频上主说话人仍被劈出 142s（4% share）
卫星簇 → 3 spk。取对 `embs` 后卫星簇消失，raw 结构与 sherpa 几乎一致
（87.3%/12.2% vs 87.3%/12.6%）。

## 修复内容

- `src/core/qwen3/diarize_ort.py` 重写为 sherpa C++ pipeline 1:1 移植
  （commit 97a0aa2）：per-chunk 独立 argmax → ExcludeOverlap →
  per-(chunk,speaker) embedding（<10 活跃帧跳过）→ FastClustering →
  ReLabel → 全局 frame 网格 per-frame top-k → 段重建（receptive field
  scale/offset 映射、gap ≤ min_duration_off 合并、≤ min_duration_on 丢弃）。
  单 chunk（≤10s）走 sherpa 同款特例：不聚类，chunk 局部 slot 直接当输出。
- TitaNet embedding 按输出名显式取 `embs`（commit 8761327）。
- 单测含根因回归（slot 跨 chunk 置换不得合并说话人 / 必须取 192 维 embs）。
- parity 验收 bar 收紧（commit c244a00）：spk_count 严格相等 + IoU ≥ 0.95。

## 验收结果（2026-06-10，全部达标）

1. **60s podcast**：Mac CPU 与 3060 ort_cuda 均 = 2 speaker；
   `tests/integration/test_diarize_ort_parity.py` 6/6 绿，mean IoU **0.985**
   （修复前 0.22）。
2. **长音频评测集 5 段回归**（3060，ablation 框架 default 配置）：

   | 段 | true | ort_cuda | sherpa | 判定 |
   |---|---|---|---|---|
   | 1spk_real 16min | 1 | 1 | 1 | ✓ |
   | panel 3+ 34min | 3 | 3 | 3 | ✓ |
   | 2spk_60min | 2 | 2 | 2 | ✓ |
   | 4spk 44min | 4 | 5 | 5 | expected（4 真人 + 英文歌 noise，5/22 基线同款） |
   | 6spk_60min | 6 | 6 | 6 | ✓ |

3. **短音频矩阵**（`scripts/_remote_short_audio_matrix.py`，1/5/10min 切片 ×
   1/2/4/6 人源，后处理人数 parity）：1spk/2spk **全 OK**（含 2spk 全时长
   2=2，under-detect 偏置消除）；4/6spk 嘈杂 panel 切片 ±1~3 双向差异
   （ort 有时更接近真值，如 4spk 600s ort=4 vs sherpa=3），属于
   "raw cluster 行为跨 backend 不逐位兼容"的既有属性，非回归。

## 遗留事项（另案，非本课题引入）

- **cluster_merge 的 embedding extractor 对 >122.88s 的单人长 turn 必崩**
  （TitaNet ONNX 导出产物里 12288 帧 mask 硬编码；2spk_60min 上该层从来
  没真正生效过）。已开新课题交接：
  `2026-06-10-cluster_merge-extractor-122s崩溃-新session-prompt.md`。
- 我们的 numpy mel（复刻 NeMo 原版 hann/reflect-pad）与 sherpa kaldi-fbank
  近似实现的 embedding cos 相似度 0.88~0.96 —— 残差属实现差异，自一致性
  不受影响，无需追平。

## 复跑入口

- Mac parity：`FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/test_diarize_ort_parity.py`
- 3060 长音频回归：`scripts/_remote_diarize_accuracy_eval.py`
- 3060 短音频矩阵：`scripts/_remote_short_audio_matrix.py`
