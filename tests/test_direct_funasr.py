"""
直接测试FunASR功能，使用最简单的方式
"""

import os
from funasr import AutoModel

print("直接测试FunASR...")

# 查找音频文件
audio_files = []
for file in os.listdir("samples"):
    if file.endswith(('.wav', '.mp3', '.mp4')):
        audio_files.append(os.path.join("samples", file))

if not audio_files:
    print("没有找到音频文件")
    exit(1)

print(f"找到 {len(audio_files)} 个音频文件")

# 1. 测试最基础的模型
print("\n1. 加载最基础的ASR模型...")
try:
    model = AutoModel(model="paraformer-zh", disable_update=True, disable_pbar=True)
    print("✓ 模型加载成功")
    
    # 测试第一个音频
    test_file = audio_files[0]
    print(f"\n2. 测试文件: {test_file}")
    
    result = model.generate(input=test_file, disable_pbar=True)
    
    print("\n3. 结果:")
    if isinstance(result, list) and len(result) > 0:
        data = result[0]
        if 'text' in data:
            text = data['text']
            # 只显示前100个字符
            if len(text) > 100:
                print(f"文本: {text[:100]}...")
            else:
                print(f"文本: {text}")
        else:
            print("结果中没有文本")
            print(f"结果键: {list(data.keys())}")
    else:
        print(f"未知结果格式: {type(result)}")
        
except Exception as e:
    print(f"✗ 基础模型测试失败: {e}")
    import traceback
    traceback.print_exc()

# 2. 测试带VAD的模型
print("\n\n4. 测试带VAD和标点的模型...")
try:
    model_vad = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad", 
        punc_model="ct-punc",
        disable_update=True,
        disable_pbar=True
    )
    print("✓ VAD模型加载成功")
    
    # 测试
    result = model_vad.generate(input=test_file, batch_size_s=100, disable_pbar=True)
    
    print("\n5. VAD模型结果:")
    if isinstance(result, list) and len(result) > 0:
        data = result[0]
        if 'text' in data:
            text = data['text']
            if len(text) > 100:
                print(f"文本: {text[:100]}...")
            else:
                print(f"文本: {text}")
        
        if 'sentence_info' in data:
            print(f"句子数: {len(data['sentence_info'])}")
            if len(data['sentence_info']) > 0:
                first_sent = data['sentence_info'][0]
                print(f"第一句: {first_sent}")
                
except Exception as e:
    print(f"✗ VAD模型测试失败: {e}")

print("\n测试完成!")