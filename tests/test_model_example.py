#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用模型自带的示例音频测试说话人识别
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from funasr import AutoModel


def test_with_model_example():
    """使用模型自带的示例音频"""
    print("=" * 80)
    print("使用模型自带示例音频测试说话人识别")
    print("=" * 80)
    
    # 创建模型
    print("加载FunASR模型（官方说话人识别配置）...")
    try:
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
        print("模型加载成功")
        
        # 使用模型自带的示例音频
        example_audio = f"{model.model_path}/example/asr_example.wav"
        print(f"\n使用示例音频: {example_audio}")
        
        if os.path.exists(example_audio):
            print("示例音频文件存在")
        else:
            print("示例音频文件不存在，查找其他示例...")
            # 查找模型目录下的示例文件
            model_path = Path(model.model_path)
            example_files = list(model_path.rglob("*.wav"))
            if example_files:
                example_audio = str(example_files[0])
                print(f"找到示例音频: {example_audio}")
            else:
                print("没有找到示例音频文件")
                return None
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None
    
    # 执行转录
    print(f"\n开始转录...")
    try:
        result = model.generate(
            input=example_audio, 
            batch_size_s=300, 
            hotword='魔搭'  # 使用官方示例的热词
        )
        print("转录完成")
        
    except Exception as e:
        print(f"转录失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 显示结果
    print("\n" + "="*80)
    print("转录结果:")
    print("="*80)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 分析结果
    print("\n" + "="*80)
    print("结果分析:")
    print("="*80)
    
    if isinstance(result, list) and len(result) > 0:
        data = result[0]
        print(f"可用字段: {list(data.keys())}")
        
        # 检查文本
        if 'text' in data:
            text = data['text']
            print(f"转录文本: {text}")
        
        # 检查说话人信息
        if 'sentence_info' in data:
            sentences = data['sentence_info']
            print(f"句子数量: {len(sentences)}")
            
            if sentences:
                # 检查是否有说话人信息
                first_sentence = sentences[0]
                print(f"第一句结构: {list(first_sentence.keys())}")
                
                if 'spk' in first_sentence:
                    print("✓ 包含说话人信息!")
                    
                    # 统计说话人
                    speakers = {}
                    for sent in sentences:
                        spk = sent.get('spk', 'Unknown')
                        speakers[spk] = speakers.get(spk, 0) + 1
                    
                    print(f"说话人统计: {speakers}")
                    
                    # 显示详细信息
                    print("\n句子详情:")
                    for i, sent in enumerate(sentences[:5]):
                        spk = sent.get('spk', 'Unknown')
                        text = sent.get('text', '')
                        start = sent.get('start', 0) / 1000
                        end = sent.get('end', 0) / 1000  
                        print(f"  {i+1}. [{spk}] {start:.2f}s-{end:.2f}s: {text}")
                
                else:
                    print("✗ 没有找到说话人信息 (spk字段)")
        else:
            print("✗ 没有句子信息 (sentence_info字段)")
    
    return result


if __name__ == "__main__":
    print("FunASR 说话人识别测试 - 使用模型示例音频")
    
    # 检查funasr版本
    try:
        import funasr
        print(f"FunASR 版本: {funasr.__version__}\n")
    except:
        pass
    
    # 运行测试
    result = test_with_model_example()
    
    if result:
        print("\n" + "="*80)
        print("测试完成！请查看上方结果分析")
        print("="*80)
    else:
        print("\n测试失败")