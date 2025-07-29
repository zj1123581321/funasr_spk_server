"""
测试FunASR基本功能，包括说话人分离
"""

import os
import sys
from pathlib import Path
from funasr import AutoModel
import time
import json

def test_basic_transcription():
    """测试基本转录功能"""
    print("=== 测试基本转录功能 ===")
    
    # 创建模型
    model = AutoModel(
        model="paraformer-zh", 
        model_revision="v2.0.4",
        vad_model="fsmn-vad", 
        vad_model_revision="v2.0.4",
        punc_model="ct-punc-c", 
        punc_model_revision="v2.0.4",
    )
    
    # 测试单个音频文件
    test_file = "samples/asr_example_zh.wav"
    if os.path.exists(test_file):
        print(f"正在转录文件: {test_file}")
        result = model.generate(input=test_file, batch_size_s=300)
        print(f"转录结果: {result}")
        print("-" * 50)
    else:
        print(f"警告: 测试文件 {test_file} 不存在")
    
    return model

def test_speaker_diarization():
    """测试说话人分离功能"""
    print("\n=== 测试说话人分离功能 ===")
    
    # 创建带说话人识别的模型
    model_with_spk = AutoModel(
        model="paraformer-zh", 
        model_revision="v2.0.4",
        vad_model="fsmn-vad", 
        vad_model_revision="v2.0.4",
        punc_model="ct-punc-c", 
        punc_model_revision="v2.0.4",
        spk_model="cam++", 
        spk_model_revision="v2.0.2",
    )
    
    # 测试带有多个说话人的音频
    test_files = [
        "samples/asr_example_zh.wav",
        "samples/multi_speaker_example.wav",
        "samples/vad_example.wav"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n正在处理文件: {test_file}")
            start_time = time.time()
            
            result = model_with_spk.generate(
                input=test_file, 
                batch_size_s=300,
                batch_size_threshold_s=60
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"处理时间: {processing_time:.2f}秒")
            print("完整结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            # 解析结果
            if isinstance(result, list) and len(result) > 0:
                for item in result:
                    if 'sentence_info' in item:
                        print("\n句子级别信息:")
                        for sentence in item['sentence_info']:
                            speaker = sentence.get('spk', 'Unknown')
                            text = sentence.get('text', '')
                            start = sentence.get('start', 0)
                            end = sentence.get('end', 0)
                            print(f"[{speaker}] {start/1000:.2f}s - {end/1000:.2f}s: {text}")
            
            print("-" * 50)
        else:
            print(f"跳过不存在的文件: {test_file}")

def test_all_audio_files():
    """测试samples目录下的所有音频文件"""
    print("\n=== 测试所有音频文件 ===")
    
    # 创建带说话人识别的模型
    model = AutoModel(
        model="paraformer-zh", 
        model_revision="v2.0.4",
        vad_model="fsmn-vad", 
        vad_model_revision="v2.0.4",
        punc_model="ct-punc-c", 
        punc_model_revision="v2.0.4",
        spk_model="cam++", 
        spk_model_revision="v2.0.2",
    )
    
    samples_dir = Path("samples")
    if samples_dir.exists():
        audio_extensions = ['.wav', '.mp3', '.mp4', '.m4a', '.flac']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(samples_dir.glob(f"*{ext}"))
        
        print(f"找到 {len(audio_files)} 个音频文件")
        
        results = []
        for audio_file in audio_files:
            print(f"\n处理: {audio_file}")
            try:
                start_time = time.time()
                
                result = model.generate(
                    input=str(audio_file), 
                    batch_size_s=300,
                    batch_size_threshold_s=60
                )
                
                processing_time = time.time() - start_time
                
                # 保存结果
                result_data = {
                    "file_name": str(audio_file),
                    "processing_time": processing_time,
                    "result": result
                }
                results.append(result_data)
                
                print(f"处理完成，用时: {processing_time:.2f}秒")
                
                # 显示简化结果
                if isinstance(result, list) and len(result) > 0:
                    for item in result:
                        if 'text' in item:
                            print(f"全文: {item['text']}")
                        if 'sentence_info' in item:
                            speakers = set()
                            for sentence in item['sentence_info']:
                                speaker = sentence.get('spk', 'Unknown')
                                speakers.add(speaker)
                            print(f"检测到说话人: {', '.join(speakers)}")
                
            except Exception as e:
                print(f"处理失败: {e}")
        
        # 保存所有结果
        output_file = "test_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n所有结果已保存到: {output_file}")
    else:
        print("samples 目录不存在")

if __name__ == "__main__":
    # 确保在正确的目录运行
    if not os.path.exists("samples"):
        print("错误: 请在项目根目录运行此脚本")
        sys.exit(1)
    
    # 运行测试
    print("开始测试 FunASR 功能...")
    
    # 1. 测试基本转录
    # test_basic_transcription()
    
    # 2. 测试说话人分离
    # test_speaker_diarization()
    
    # 3. 测试所有音频文件
    test_all_audio_files()
    
    print("\n测试完成!")