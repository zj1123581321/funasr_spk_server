# FunASR 转录器期望输出格式

## 概述

`src/core/funasr_transcriber.py` 转录器的输出格式应该与测试脚本产生的结果完全一致，包括：

1. **相同说话人连续句子的自动合并**
2. **精确的时间信息**
3. **标准的 JSON 格式**

## 输入数据格式

FunASR 模型返回的原始数据格式：

```json
[{
  "key": "audio_filename",
  "text": "完整转录文本...",
  "timestamp": [[50, 290], [310, 410], ...],
  "sentence_info": [
    {
      "text": "句子文本",
      "start": 50,      // 毫秒
      "end": 290,       // 毫秒
      "timestamp": [[50, 290]],
      "spk": 0          // 说话人ID (0, 1, 2...)
    },
    ...
  ]
}]
```

## 输出格式

### 1. 未合并的原始输出（15个片段）

```json
{
  "task_id": "test-1",
  "file_name": "spk_extract.mp3",
  "file_hash": "5e06014621b265556822403cba11e9d0",
  "duration": 40.05,
  "processing_time": 5.43,
  "speakers": ["Speaker1", "Speaker2"],
  "segments": [
    {
      "start_time": 0.05,
      "end_time": 0.29,
      "text": "好，",
      "speaker": "Speaker1",
      "duration": 0.24
    },
    {
      "start_time": 0.31,
      "end_time": 1.71,
      "text": "欢迎收听本期的创业内幕。",
      "speaker": "Speaker1",
      "duration": 1.4
    },
    // ... 更多片段
  ]
}
```

### 2. 合并后的期望输出（4个片段）

```json
{
  "task_id": "test-1",
  "file_name": "spk_extract.mp3",
  "file_hash": "5e06014621b265556822403cba11e9d0",
  "duration": 40.05,
  "processing_time": 5.43,
  "speakers": ["Speaker1", "Speaker2"],
  "segments": [
    {
      "start_time": 0.05,
      "end_time": 12.49,
      "text": "好欢迎收听本期的创业内幕我是主持人 lily本期我们请到的嘉宾来自于国内首家自主研发云 CAD 公司卡伦特的技术合伙人李荣璐 lif 李总跟大家打个招呼吧。",
      "speaker": "Speaker1",
      "duration": 12.44
    },
    {
      "start_time": 12.77,
      "end_time": 31.67,
      "text": "大家好我叫李荣录复旦计算机系极其专业的博士曾经在这个 auto desk 担任首位的华人首席工程师也在这个 SAP blackboard 等这些公司呢担任首席架程师然后呢也有将近二十年的软件开发和管理经验嗯，",
      "speaker": "Speaker2",
      "duration": 18.9
    },
    {
      "start_time": 32.11,
      "end_time": 34.01,
      "text": "您当年为什么选择加入卡伦特，",
      "speaker": "Speaker1",
      "duration": 1.9
    },
    {
      "start_time": 34.01,
      "end_time": 39.96,
      "text": "这是一个很有意思的问题啊当年我在复旦读博的时候确实赶上了人工智能的一个，",
      "speaker": "Speaker2",
      "duration": 5.95
    }
  ],
  "transcription_summary": {
    "total_speakers": 2,
    "total_segments": 4,
    "full_text": "好欢迎收听本期的创业内幕我是主持人 lily本期我们请到的嘉宾来自于国内首家自主研发云 CAD 公司卡伦特的技术合伙人李荣璐 lif 李总跟大家打个招呼吧。 大家好我叫李荣录复旦计算机系极其专业的博士曾经在这个 auto desk 担任首位的华人首席工程师也在这个 SAP blackboard 等这些公司呢担任首席架程师然后呢也有将近二十年的软件开发和管理经验嗯， 您当年为什么选择加入卡伦特， 这是一个很有意思的问题啊当年我在复旦读博的时候确实赶上了人工智能的一个，"
  }
}
```

## 合并规则

### 合并条件
- **相同说话人**: `speaker` 字段必须相同
- **时间间隔**: 两个片段之间的时间间隔小于 3 秒
- **连续性**: 片段在时间上是连续的

### 合并逻辑
1. 保留第一个片段的开始时间
2. 使用最后一个片段的结束时间
3. 合并所有片段的文本内容
4. 移除文本末尾的标点符号以避免重复

### 合并效果
- **Speaker1**: 5 个原始片段 → 2 个合并片段 (40% 压缩率)
- **Speaker2**: 10 个原始片段 → 2 个合并片段 (20% 压缩率)
- **总体**: 15 个原始片段 → 4 个合并片段 (73% 压缩率)

## 关键特性

1. **保持时间精度**: 时间戳精确到 0.01 秒
2. **说话人标识**: 使用 "Speaker1", "Speaker2" 格式
3. **自动合并**: 智能合并相同说话人的连续发言
4. **完整信息**: 包含文件哈希、处理时间等元数据
5. **缓存支持**: 模型缓存到本地 `./models` 目录

## 测试验证

运行以下命令验证合并功能：

```bash
python tests/test_merge_function.py
```

期望结果：
- 原始片段: 15 个
- 合并后片段: 4 个
- 合并率: 73%
- 生成文件: `tests/output/merge_test_result.json`