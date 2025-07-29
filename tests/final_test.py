#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终测试脚本 - 验证FunASR功能
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from funasr import AutoModel


def test_with_vad_and_punc():
    """使用VAD和标点的模型测试"""
    print("\n测试1: 使用VAD和标点模型")
    print("-" * 50)
    
    # 查找音频文件
    audio_file = project_root / "samples" / "asr_example.wav"
    if not audio_file.exists():
        # 尝试其他文件
        audio_files = list((project_root / "samples").glob("*.mp3"))
        if audio_files:
            audio_file = audio_files[0]
        else:
            print("错误: 没有找到音频文件")
            return
    
    print(f"音频文件: {audio_file.name}")
    
    # 创建模型（参考示例代码的方式）
    print("加载模型...")
    model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc")
    
    # 转录
    print("开始转录...")
    result = model.generate(
        input=str(audio_file),
        batch_size_s=300
    )
    
    # 显示结果
    if isinstance(result, list) and len(result) > 0:
        data = result[0]
        text = data.get('text', '')
        print(f"\n转录文本: {text}")
        
        # 如果有句子信息
        if 'sentence_info' in data:
            sentences = data['sentence_info']
            print(f"\n句子数: {len(sentences)}")
            print("\n前3句:")
            for i, sent in enumerate(sentences[:3]):
                text = sent.get('text', '')
                start = sent.get('start', 0) / 1000
                end = sent.get('end', 0) / 1000
                print(f"  {i+1}. [{start:.2f}s - {end:.2f}s] {text}")


def test_with_speaker():
    """测试说话人识别功能"""
    print("\n\n测试2: 带说话人识别")
    print("-" * 50)
    
    audio_file = project_root / "samples" / "test_extract.mp3"
    if not audio_file.exists():
        print("跳过: 文件不存在")
        return
    
    print(f"音频文件: {audio_file.name}")
    
    try:
        # 创建带说话人识别的模型
        print("加载说话人识别模型...")
        model = AutoModel(
            model="paraformer-zh", 
            vad_model="fsmn-vad", 
            punc_model="ct-punc",
            spk_model="cam++"
        )
        
        # 转录
        print("开始转录...")
        result = model.generate(
            input=str(audio_file),
            batch_size_s=300
        )
        
        # 显示结果
        if isinstance(result, list) and len(result) > 0:
            data = result[0]
            
            # 统计说话人
            if 'sentence_info' in data:
                sentences = data['sentence_info']
                speakers = {}
                for sent in sentences:
                    spk = sent.get('spk', 'Unknown')
                    speakers[spk] = speakers.get(spk, 0) + 1
                
                print(f"\n说话人统计: {speakers}")
                print(f"总句子数: {len(sentences)}")
                
                # 显示不同说话人的示例
                print("\n不同说话人的句子示例:")
                shown_speakers = set()
                for sent in sentences:
                    spk = sent.get('spk', 'Unknown')
                    if spk not in shown_speakers and len(shown_speakers) < 3:
                        text = sent.get('text', '')
                        print(f"  [{spk}] {text}")
                        shown_speakers.add(spk)
                        
    except Exception as e:
        print(f"说话人识别失败: {e}")
        print("可能是模型尚未下载或配置问题")


def test_simple():
    """最简单的测试"""
    print("\n\n测试3: 最简单的模型")
    print("-" * 50)
    
    # 使用test_extract.mp3
    audio_file = project_root / "samples" / "test_extract.mp3"
    if not audio_file.exists():
        print("文件不存在")
        return
    
    print(f"音频文件: {audio_file.name}")
    
    # 只使用基础模型
    print("加载基础模型...")
    model = AutoModel(model="paraformer-zh")
    
    # 转录
    print("开始转录...")
    result = model.generate(input=str(audio_file))
    
    # 显示结果
    if isinstance(result, list) and len(result) > 0:
        text = result[0].get('text', '')
        # 只显示前200个字符
        if len(text) > 200:
            print(f"\n转录文本（前200字）: {text[:200]}...")
        else:
            print(f"\n转录文本: {text}")


if __name__ == "__main__":
    print("=" * 60)
    print("FunASR 功能测试")
    print("=" * 60)
    
    # 检查funasr版本
    try:
        import funasr
        print(f"FunASR 版本: {funasr.__version__}")
    except:
        pass
    
    # 运行测试
    test_with_vad_and_punc()
    test_with_speaker()
    test_simple()
    
    print("\n\n测试完成！")