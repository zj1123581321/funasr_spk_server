#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专门测试FunASR说话人识别功能
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from funasr import AutoModel


def test_basic_model():
    """测试基础模型（不带说话人识别）"""
    print("=" * 80)
    print("测试1: 基础模型（paraformer-zh）")
    print("=" * 80)
    
    # 查找音频文件
    samples_dir = project_root / "samples"
    audio_file = samples_dir / "test_extract.mp3"
    
    if not audio_file.exists():
        audio_files = list(samples_dir.glob("*.wav")) + list(samples_dir.glob("*.mp3"))
        if audio_files:
            audio_file = audio_files[0]
        else:
            print("错误: 没有找到音频文件")
            return
    
    print(f"音频文件: {audio_file.name}")
    
    # 创建基础模型
    print("加载基础模型...")
    model = AutoModel(model="paraformer-zh")
    
    # 转录
    print("开始转录...")
    result = model.generate(input=str(audio_file))
    
    # 显示完整结果
    print("\n完整结果结构:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 分析结果
    if isinstance(result, list) and len(result) > 0:
        data = result[0]
        print(f"\n结果分析:")
        print(f"- 数据类型: {type(data)}")
        print(f"- 可用键: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        
        if 'text' in data:
            print(f"- 文本: {data['text'][:100]}...")
        
        if 'sentence_info' in data:
            print(f"- 是否有sentence_info: 是 ({len(data['sentence_info'])} 句)")
        else:
            print(f"- 是否有sentence_info: 否")
    
    return result


def test_with_vad_punc():
    """测试带VAD和标点的模型"""
    print("\n" + "=" * 80)
    print("测试2: VAD + 标点模型")
    print("=" * 80)
    
    # 查找音频文件
    samples_dir = project_root / "samples"
    audio_file = samples_dir / "test_extract.mp3"
    
    if not audio_file.exists():
        audio_files = list(samples_dir.glob("*.wav")) + list(samples_dir.glob("*.mp3"))
        if audio_files:
            audio_file = audio_files[0]
        else:
            print("错误: 没有找到音频文件")
            return
    
    print(f"音频文件: {audio_file.name}")
    
    # 创建带VAD和标点的模型
    print("加载VAD+标点模型...")
    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc"
    )
    
    # 转录
    print("开始转录...")
    result = model.generate(input=str(audio_file), batch_size_s=300)
    
    # 显示完整结果
    print("\n完整结果结构:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 分析结果
    if isinstance(result, list) and len(result) > 0:
        data = result[0]
        print(f"\n结果分析:")
        print(f"- 数据类型: {type(data)}")
        print(f"- 可用键: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        
        if 'sentence_info' in data:
            sentences = data['sentence_info']
            print(f"- 句子数量: {len(sentences)}")
            
            # 检查说话人信息
            has_speaker = False
            speakers = set()
            for sent in sentences:
                if 'spk' in sent:
                    has_speaker = True
                    speakers.add(sent['spk'])
            
            print(f"- 是否有说话人信息: {'是' if has_speaker else '否'}")
            if has_speaker:
                print(f"- 说话人列表: {list(speakers)}")
                
                print("\n前5句详细信息:")
                for i, sent in enumerate(sentences[:5]):
                    spk = sent.get('spk', 'Unknown')
                    text = sent.get('text', '')
                    start = sent.get('start', 0)
                    end = sent.get('end', 0)
                    print(f"  {i+1}. [{spk}] {start/1000:.2f}s-{end/1000:.2f}s: {text}")
    
    return result


def test_with_speaker_model():
    """测试带说话人识别的完整模型"""
    print("\n" + "=" * 80)
    print("测试3: 完整模型（包含说话人识别）")
    print("=" * 80)
    
    # 查找音频文件
    samples_dir = project_root / "samples"
    audio_file = samples_dir / "test_extract.mp3"
    
    if not audio_file.exists():
        audio_files = list(samples_dir.glob("*.wav")) + list(samples_dir.glob("*.mp3"))
        if audio_files:
            audio_file = audio_files[0]
        else:
            print("错误: 没有找到音频文件")
            return
    
    print(f"音频文件: {audio_file.name}")
    
    try:
        # 创建完整模型
        print("加载完整模型（包含说话人识别）...")
        model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++"
        )
        
        # 转录
        print("开始转录...")
        result = model.generate(input=str(audio_file), batch_size_s=300)
        
        # 显示完整结果
        print("\n完整结果结构:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 分析结果
        if isinstance(result, list) and len(result) > 0:
            data = result[0]
            print(f"\n结果分析:")
            print(f"- 数据类型: {type(data)}")
            print(f"- 可用键: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            
            if 'sentence_info' in data:
                sentences = data['sentence_info']
                print(f"- 句子数量: {len(sentences)}")
                
                # 统计说话人
                speakers = {}
                for sent in sentences:
                    spk = sent.get('spk', 'Unknown')
                    speakers[spk] = speakers.get(spk, 0) + 1
                
                print(f"- 说话人统计: {speakers}")
                
                print("\n所有句子的说话人信息:")
                for i, sent in enumerate(sentences):
                    spk = sent.get('spk', 'Unknown')
                    text = sent.get('text', '')
                    start = sent.get('start', 0)
                    end = sent.get('end', 0)
                    print(f"  {i+1:2d}. [{spk}] {start/1000:6.2f}s-{end/1000:6.2f}s: {text}")
        
        return result
        
    except Exception as e:
        print(f"说话人识别模型加载失败: {e}")
        print("可能需要下载相关模型或存在兼容性问题")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("FunASR 说话人识别功能测试")
    print("本测试将比较不同模型配置的输出差异")
    print()
    
    # 检查funasr版本
    try:
        import funasr
        print(f"FunASR 版本: {funasr.__version__}")
    except:
        pass
    
    print()
    
    # 运行测试
    result1 = test_basic_model()
    result2 = test_with_vad_punc() 
    result3 = test_with_speaker_model()
    
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print("1. 基础模型: ", "成功" if result1 else "失败")
    print("2. VAD+标点模型: ", "成功" if result2 else "失败")
    print("3. 完整模型（说话人）: ", "成功" if result3 else "失败")
    print()
    print("如果所有模型都只输出文本而没有说话人信息，")
    print("可能是音频文件只有单个说话人，或者模型配置需要调整。")