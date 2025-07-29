#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的音频转录测试 - 不依赖服务器
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from funasr import AutoModel


def transcribe_audio(audio_path):
    """
    直接转录音频文件 - 使用带说话人识别的完整模型
    
    Args:
        audio_path: 音频文件路径
    
    Returns:
        完整的转录结果
    """
    print(f"正在处理音频文件: {audio_path}")
    
    # 创建带说话人识别的完整模型（参考官方示例）
    print("加载FunASR完整模型（包含说话人识别）...")
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
    
    # 执行转录
    print("开始转录...")
    result = model.generate(
        input=audio_path, 
        batch_size_s=300, 
        hotword=''  # 可以添加热词
    )
    
    return result


def main():
    """主函数"""
    # 查找音频文件
    samples_dir = project_root / "samples"
    
    # 支持的音频格式
    audio_formats = ['.wav', '.mp3', '.mp4', '.m4a', '.flac']
    
    # 查找音频文件
    audio_files = []
    for fmt in audio_formats:
        audio_files.extend(samples_dir.glob(f"*{fmt}"))
    
    if not audio_files:
        print("错误: 在 samples 文件夹中没有找到音频文件")
        return
    
    print(f"找到 {len(audio_files)} 个音频文件\n")
    
    # 转录每个文件
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n=== 文件 {i}/{len(audio_files)} ===")
        
        try:
            # 转录音频
            result = transcribe_audio(str(audio_file))
            
            # 显示结果
            print(f"\n文件名: {audio_file.name}")
            print("=" * 60)
            
            # 显示完整的结果结构
            import json
            print("完整转录结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            # 解析并显示关键信息
            if isinstance(result, list) and len(result) > 0:
                data = result[0]
                
                print("\n关键信息提取:")
                print(f"- 结果类型: {type(result)}")
                print(f"- 数据键: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                
                if 'text' in data:
                    text = data['text']
                    print(f"- 转录文本: {text}")
                    print(f"- 文本长度: {len(text)} 字符")
                
                if 'sentence_info' in data:
                    sentences = data['sentence_info']
                    print(f"- 句子信息: {len(sentences)} 句")
                    
                    # 检查是否有说话人信息
                    speakers = set()
                    for sent in sentences:
                        if 'spk' in sent:
                            speakers.add(sent['spk'])
                    
                    if speakers:
                        print(f"- 检测到说话人: {list(speakers)}")
                        print("\n前3句详细信息:")
                        for i, sent in enumerate(sentences[:3]):
                            spk = sent.get('spk', 'Unknown')
                            text = sent.get('text', '')
                            start = sent.get('start', 0)
                            end = sent.get('end', 0)
                            print(f"  {i+1}. [{spk}] {start}ms-{end}ms: {text}")
                    else:
                        print("- 未检测到说话人信息")
                
                if 'timestamp' in data:
                    timestamps = data['timestamp']
                    print(f"- 时间戳信息: {len(timestamps)} 个")
                    if len(timestamps) > 0:
                        print(f"- 时间戳示例: {timestamps[:3]}...")
            
            print("-" * 60)
            
        except Exception as e:
            print(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n转录测试完成！")


if __name__ == "__main__":
    print("=" * 60)
    print("FunASR 音频转录测试 (不依赖服务器)")
    print("=" * 60)
    print()
    
    # 检查是否安装了 funasr
    try:
        import funasr
        print(f"FunASR 版本: {funasr.__version__}")
    except ImportError:
        print("错误: 未安装 funasr")
        print("请运行: pip install funasr")
        sys.exit(1)
    
    # 运行测试
    main()