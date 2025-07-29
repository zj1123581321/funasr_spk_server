#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试FunASR功能 - UTF-8编码版本
"""

import os
import sys
import json
import time
from pathlib import Path

# 设置UTF-8编码
if sys.platform == 'win32':
    # Windows系统设置控制台编码为UTF-8
    os.system('chcp 65001 > nul')

# 设置Python输出编码
sys.stdout.reconfigure(encoding='utf-8')

print("开始FunASR功能测试...")

# 检查samples目录
samples_dir = Path("samples")
if not samples_dir.exists():
    print(f"错误: 未找到 {samples_dir} 目录")
    sys.exit(1)

# 列出音频文件
audio_extensions = ['.wav', '.mp3', '.mp4', '.m4a', '.flac']
audio_files = []
for ext in audio_extensions:
    audio_files.extend(samples_dir.glob(f"*{ext}"))

print(f"\n找到 {len(audio_files)} 个音频文件:")
for f in audio_files:
    print(f"  - {f}")

if not audio_files:
    print("错误: 没有找到音频文件")
    sys.exit(1)

# 导入FunASR
try:
    from funasr import AutoModel
    print("\nFunASR导入成功")
except ImportError as e:
    print(f"错误: 无法导入FunASR - {e}")
    sys.exit(1)

# 测试文件
test_file = str(audio_files[0])
print(f"\n使用测试文件: {test_file}")

# 1. 测试基础模型（无说话人识别）
print("\n=== 测试1: 基础语音识别 ===")
try:
    model_basic = AutoModel(
        model="paraformer-zh",
        disable_update=True  # 禁用版本检查
    )
    print("模型加载成功")
    
    result = model_basic.generate(input=test_file)
    print("\n转录结果:")
    
    # 处理结果
    if isinstance(result, list) and len(result) > 0:
        for item in result:
            if 'text' in item:
                text = item['text']
                print(f"文本: {text}")
            if 'timestamp' in item:
                print(f"时间戳数量: {len(item['timestamp'])}")
    
    # 保存结果
    with open("test_result_basic.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("结果已保存到: test_result_basic.json")
    
except Exception as e:
    print(f"基础模型测试失败: {e}")
    import traceback
    traceback.print_exc()

# 2. 测试带说话人识别的模型
print("\n\n=== 测试2: 带说话人识别的语音识别 ===")
try:
    model_spk = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        spk_model="cam++",
        disable_update=True
    )
    print("说话人识别模型加载成功")
    
    print(f"\n处理文件: {test_file}")
    start_time = time.time()
    
    result = model_spk.generate(
        input=test_file,
        batch_size_s=300,
        batch_size_threshold_s=60
    )
    
    processing_time = time.time() - start_time
    print(f"处理时间: {processing_time:.2f}秒")
    
    # 处理结果
    if isinstance(result, list) and len(result) > 0:
        for item in result:
            if 'text' in item:
                print(f"\n完整文本: {item['text']}")
            
            if 'sentence_info' in item:
                sentences = item['sentence_info']
                print(f"\n句子数量: {len(sentences)}")
                
                # 统计说话人
                speakers = {}
                for sent in sentences:
                    spk = sent.get('spk', 'Unknown')
                    if spk not in speakers:
                        speakers[spk] = 0
                    speakers[spk] += 1
                
                print(f"说话人统计: {speakers}")
                
                # 显示前几句
                print("\n前5句内容:")
                for i, sent in enumerate(sentences[:5]):
                    spk = sent.get('spk', 'Unknown')
                    text = sent.get('text', '')
                    start = sent.get('start', 0) / 1000
                    end = sent.get('end', 0) / 1000
                    print(f"{i+1}. [{spk}] {start:.2f}s-{end:.2f}s: {text}")
    
    # 保存结果
    with open("test_result_spk.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n结果已保存到: test_result_spk.json")
    
except Exception as e:
    print(f"说话人识别模型测试失败: {e}")
    import traceback
    traceback.print_exc()

# 3. 测试所有音频文件
print("\n\n=== 测试3: 批量处理所有音频文件 ===")
if len(audio_files) > 1:
    try:
        # 使用已加载的模型
        all_results = []
        
        for audio_file in audio_files[:3]:  # 只处理前3个文件
            print(f"\n处理: {audio_file.name}")
            try:
                start_time = time.time()
                result = model_spk.generate(input=str(audio_file))
                processing_time = time.time() - start_time
                
                # 简单统计
                total_text = ""
                speakers = set()
                
                if isinstance(result, list) and len(result) > 0:
                    for item in result:
                        if 'text' in item:
                            total_text = item['text']
                        if 'sentence_info' in item:
                            for sent in item['sentence_info']:
                                speakers.add(sent.get('spk', 'Unknown'))
                
                summary = {
                    "file": str(audio_file),
                    "processing_time": processing_time,
                    "text_length": len(total_text),
                    "speakers": list(speakers)
                }
                
                all_results.append(summary)
                print(f"  - 处理时间: {processing_time:.2f}秒")
                print(f"  - 文本长度: {len(total_text)}字符")
                print(f"  - 说话人: {', '.join(speakers)}")
                
            except Exception as e:
                print(f"  - 处理失败: {e}")
        
        # 保存汇总结果
        with open("test_results_summary.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print("\n汇总结果已保存到: test_results_summary.json")
        
    except Exception as e:
        print(f"批量处理失败: {e}")

print("\n\n测试完成！")
print("生成的文件:")
print("  - test_result_basic.json: 基础转录结果")
print("  - test_result_spk.json: 说话人识别结果")
print("  - test_results_summary.json: 批量处理汇总")