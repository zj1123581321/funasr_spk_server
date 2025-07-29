#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用官方推荐配置的说话人识别测试
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from funasr import AutoModel


def test_official_speaker_config():
    """使用官方推荐的说话人识别配置"""
    print("=" * 80)
    print("FunASR 官方说话人识别配置测试")
    print("=" * 80)
    
    # 查找音频文件
    samples_dir = project_root / "samples"
    audio_files = list(samples_dir.glob("*.mp3")) + list(samples_dir.glob("*.wav"))
    
    if not audio_files:
        print("错误: 没有找到音频文件")
        return
    
    # 使用第一个音频文件
    audio_file = audio_files[0]
    print(f"测试音频文件: {audio_file.name}")
    print(f"文件大小: {audio_file.stat().st_size / 1024:.1f} KB")
    
    # 创建模型 - 使用官方推荐配置
    print("\n加载FunASR模型（官方说话人识别配置）...")
    print("配置:")
    print("- model: paraformer-zh, v2.0.4")
    print("- vad_model: fsmn-vad, v2.0.4")
    print("- punc_model: ct-punc-c, v2.0.4") 
    print("- spk_model: cam++, v2.0.2")
    
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
        print("✓ 模型加载成功")
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return None
    
    # 执行转录
    print(f"\n开始转录...")
    try:
        result = model.generate(
            input=str(audio_file), 
            batch_size_s=300, 
            hotword=''  # 热词可以为空
        )
        print("✓ 转录完成")
        
    except Exception as e:
        print(f"✗ 转录失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 显示完整结果
    print("\n" + "="*60)
    print("完整转录结果:")
    print("="*60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 分析结果结构
    print("\n" + "="*60)
    print("结果分析:")
    print("="*60)
    
    if isinstance(result, list) and len(result) > 0:
        data = result[0]
        print(f"结果类型: {type(result)}")
        print(f"数据结构: {type(data)}")
        print(f"可用字段: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        
        # 检查文本
        if 'text' in data:
            text = data['text']
            print(f"\n全文长度: {len(text)} 字符")
            print(f"全文预览: {text[:200]}..." if len(text) > 200 else f"全文: {text}")
        
        # 检查句子信息
        if 'sentence_info' in data:
            sentences = data['sentence_info']
            print(f"\n句子信息:")
            print(f"- 句子数量: {len(sentences)}")
            
            if len(sentences) > 0:
                # 检查说话人信息
                speakers = {}
                has_speaker_info = False
                
                for sent in sentences:
                    if 'spk' in sent:
                        has_speaker_info = True
                        spk = sent['spk']
                        speakers[spk] = speakers.get(spk, 0) + 1
                
                print(f"- 是否包含说话人信息: {'是' if has_speaker_info else '否'}")
                
                if has_speaker_info:
                    print(f"- 说话人统计: {speakers}")
                    
                    print(f"\n前10句详细信息:")
                    for i, sent in enumerate(sentences[:10]):
                        spk = sent.get('spk', 'Unknown')
                        text = sent.get('text', '')
                        start = sent.get('start', 0) / 1000  # 转换为秒
                        end = sent.get('end', 0) / 1000
                        print(f"  {i+1:2d}. [{spk}] {start:6.2f}s-{end:6.2f}s: {text}")
                    
                    if len(sentences) > 10:
                        print(f"     ... 还有 {len(sentences) - 10} 句")
                
                else:
                    print("- 句子中没有找到说话人标识 (spk字段)")
                    print("- 显示前3句结构:")
                    for i, sent in enumerate(sentences[:3]):
                        print(f"  {i+1}. 字段: {list(sent.keys())}")
                        print(f"     内容: {sent}")
        else:
            print("✗ 结果中没有 sentence_info 字段")
        
        # 检查其他字段
        other_fields = [k for k in data.keys() if k not in ['text', 'sentence_info', 'key']]
        if other_fields:
            print(f"\n其他字段: {other_fields}")
    
    else:
        print("✗ 结果格式异常")
    
    # 保存结果
    output_file = project_root / "tests" / "speaker_recognition_official_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存到: {output_file}")
    
    return result


if __name__ == "__main__":
    print("使用官方推荐配置测试FunASR说话人识别功能")
    
    # 检查funasr版本
    try:
        import funasr
        print(f"FunASR 版本: {funasr.__version__}")
    except:
        pass
    
    print()
    
    # 运行测试
    result = test_official_speaker_config()
    
    if result:
        print("\n" + "="*80)
        print("测试完成！")
        print("如果看到说话人信息（spk字段），说明功能正常")
        print("如果没有看到，可能是:")
        print("1. 音频文件只有单个说话人")
        print("2. 音频质量问题")
        print("3. 模型版本或配置问题")
        print("="*80)
    else:
        print("\n测试失败，请检查模型配置和网络连接")