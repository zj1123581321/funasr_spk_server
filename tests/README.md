# 测试文件说明

这个文件夹包含了所有的测试脚本和测试结果。

## 测试脚本

### 1. `simple_transcribe_test.py`
**用途**: 最简单的音频转录测试，不依赖服务器
**功能**: 
- 直接使用 FunASR 转录 samples 文件夹中的音频文件
- 输出转录文本结果

**使用方法**:
```bash
cd tests
python simple_transcribe_test.py
```

### 2. `final_test.py`
**用途**: 全面的 FunASR 功能测试
**功能**:
- 测试基础转录功能
- 测试 VAD（语音活动检测）和标点功能
- 测试说话人识别功能

**使用方法**:
```bash
cd tests
python final_test.py
```

### 3. `test_direct_transcription.py`
**用途**: 详细的转录测试，包含多种测试场景
**功能**:
- 基础转录测试
- 说话人分离测试
- 批量处理测试
- 生成详细的 JSON 结果文件

**使用方法**:
```bash
cd tests
python test_direct_transcription.py
```

## 其他测试文件

### 旧版测试文件
- `test_basic_server.py` - 服务器基础功能测试
- `test_transcriber.py` - transcriber 模块测试
- `test_funasr_simple.py` - 简单的 FunASR 测试
- `test_funasr_utf8.py` - UTF-8 编码版本测试
- `test_direct_funasr.py` - 直接 FunASR 测试
- `test_simple_funasr.py` - 基本功能测试

## 推荐使用

对于快速测试，建议使用：
1. `simple_transcribe_test.py` - 如果只想验证基本转录功能
2. `final_test.py` - 如果想测试完整功能包括说话人识别

## 注意事项

1. 所有测试都需要网络连接，因为可能需要下载模型
2. 首次运行会下载模型文件，需要较长时间
3. 测试音频文件位于 `../samples/` 目录
4. 如果转录结果异常（如重复字符），可能是音频文件质量问题

## 测试结果

测试运行后会在 tests 文件夹中生成以下结果文件：
- `basic_transcription_results.json` - 基础转录结果
- `speaker_diarization_results.json` - 说话人识别结果  
- `batch_processing_results.json` - 批量处理结果