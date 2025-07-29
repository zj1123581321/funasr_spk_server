"""
简单的FunASR测试 - 测试基本功能
"""

import os
from funasr import AutoModel
import json

print("开始简单的FunASR测试...")

# 检查samples目录
samples_dir = "samples"
if os.path.exists(samples_dir):
    print(f"[OK] 找到 {samples_dir} 目录")
    
    # 列出所有音频文件
    audio_files = []
    for file in os.listdir(samples_dir):
        if file.endswith(('.wav', '.mp3', '.mp4', '.m4a')):
            audio_files.append(os.path.join(samples_dir, file))
    
    print(f"找到 {len(audio_files)} 个音频文件:")
    for f in audio_files:
        print(f"  - {f}")
else:
    print(f"[ERROR] 未找到 {samples_dir} 目录")
    exit(1)

print("\n1. 创建基础模型（不含说话人识别）...")
try:
    model_basic = AutoModel(model="paraformer-zh")
    print("[OK] 基础模型创建成功")
except Exception as e:
    print(f"[ERROR] 基础模型创建失败: {e}")
    exit(1)

# 测试第一个音频文件
if audio_files:
    test_file = audio_files[0]
    print(f"\n2. 测试基础转录: {test_file}")
    try:
        result = model_basic.generate(input=test_file)
        print("[OK] 转录成功!")
        print("结果:", json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[ERROR] 转录失败: {e}")

print("\n3. 创建带说话人识别的模型...")
try:
    model_spk = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc-c", 
        spk_model="cam++",
    )
    print("[OK] 说话人识别模型创建成功")
    
    if audio_files:
        print(f"\n4. 测试说话人识别: {test_file}")
        try:
            result = model_spk.generate(input=test_file, batch_size_s=300)
            print("[OK] 说话人识别成功!")
            print("结果:", json.dumps(result, ensure_ascii=False, indent=2))
            
            # 解析说话人信息
            if isinstance(result, list) and len(result) > 0:
                for item in result:
                    if 'sentence_info' in item:
                        speakers = set()
                        for sentence in item['sentence_info']:
                            speaker = sentence.get('spk', 'Unknown')
                            speakers.add(speaker)
                        print(f"\n检测到的说话人: {', '.join(speakers)}")
                        
                        print("\n详细句子信息:")
                        for sentence in item['sentence_info'][:5]:  # 只显示前5句
                            speaker = sentence.get('spk', 'Unknown')
                            text = sentence.get('text', '')
                            start = sentence.get('start', 0) / 1000
                            end = sentence.get('end', 0) / 1000
                            print(f"[{speaker}] {start:.2f}s - {end:.2f}s: {text}")
                        
                        if len(item['sentence_info']) > 5:
                            print(f"... 还有 {len(item['sentence_info']) - 5} 句")
                            
        except Exception as e:
            print(f"[ERROR] 说话人识别失败: {e}")
            import traceback
            traceback.print_exc()
            
except Exception as e:
    print(f"[ERROR] 说话人识别模型创建失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成!")