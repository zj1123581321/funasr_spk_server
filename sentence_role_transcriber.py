#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
句子级别分角色转录输出生成器

基于FunASR转录结果，生成去除时间戳信息并合并相同说话人相邻句子的分角色转录输出
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def merge_speaker_sentences(sentence_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    合并相邻的相同说话人句子，保留时间信息
    
    Args:
        sentence_info: 原始的句子信息列表，每个元素包含 text, spk, start, end 等字段
    
    Returns:
        合并后的句子列表，相邻相同说话人的句子被合并为一个段落，包含时间信息
    """
    if not sentence_info:
        return []
    
    merged_sentences = []
    current_speaker = None
    current_text = ""
    current_start_time = None
    current_end_time = None
    
    for sentence in sentence_info:
        speaker = sentence.get('spk', 'Unknown')
        text = sentence.get('text', '').strip()
        start_ms = sentence.get('start', 0)
        end_ms = sentence.get('end', 0)
        
        if text:  # 忽略空文本
            if speaker == current_speaker:
                # 相同说话人，合并文本并更新结束时间
                current_text += text
                current_end_time = end_ms  # 更新为最新的结束时间
            else:
                # 不同说话人，保存前一个说话人的合并文本
                if current_text:
                    merged_sentences.append({
                        "speaker": current_speaker,
                        "text": current_text,
                        "start_time": round(current_start_time / 1000, 2),  # 转换为秒
                        "end_time": round(current_end_time / 1000, 2),     # 转换为秒
                        "duration": round((current_end_time - current_start_time) / 1000, 2)
                    })
                
                # 开始新的说话人段落
                current_speaker = speaker
                current_text = text
                current_start_time = start_ms
                current_end_time = end_ms
    
    # 添加最后一个说话人的文本
    if current_text:
        merged_sentences.append({
            "speaker": current_speaker,
            "text": current_text,
            "start_time": round(current_start_time / 1000, 2),  # 转换为秒
            "end_time": round(current_end_time / 1000, 2),     # 转换为秒
            "duration": round((current_end_time - current_start_time) / 1000, 2)
        })
    
    return merged_sentences


def generate_sentence_level_transcription(transcription_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成句子级别的分角色转录输出
    
    Args:
        transcription_result: 包含raw_result的完整转录结果
        
    Returns:
        句子级别的分角色转录结果，去除时间戳信息并合并相同说话人的相邻句子
    """
    # 提取原始结果
    raw_result = transcription_result.get('raw_result', [])
    if not raw_result or len(raw_result) == 0:
        return {"error": "没有找到转录结果"}
    
    data = raw_result[0]
    sentence_info = data.get('sentence_info', [])
    full_text = data.get('text', '')
    
    # 合并相同说话人的相邻句子
    merged_sentences = merge_speaker_sentences(sentence_info)
    
    # 统计说话人信息
    speakers = {}
    for sentence in merged_sentences:
        speaker = str(sentence['speaker'])
        speakers[speaker] = speakers.get(speaker, 0) + 1
    
    # 生成最终输出结构
    result = {
        "file_info": transcription_result.get('file_info', {}),
        "model_config": transcription_result.get('model_config', {}),
        "transcription_summary": {
            "total_speakers": len(speakers),
            "speaker_statistics": speakers,
            "total_segments": len(merged_sentences),
            "full_text_without_timestamps": full_text
        },
        "role_based_transcription": merged_sentences
    }
    
    return result


def process_transcription_file(input_file_path: str, output_file_path: str = None) -> None:
    """
    处理转录结果文件，生成句子级别的分角色转录输出
    
    Args:
        input_file_path: 输入的转录结果JSON文件路径
        output_file_path: 输出文件路径，如果为None则自动生成
    """
    try:
        # 读取输入文件
        with open(input_file_path, 'r', encoding='utf-8') as f:
            transcription_data = json.load(f)
        
        print(f"正在处理转录文件: {input_file_path}")
        
        # 生成句子级别的分角色转录
        result = generate_sentence_level_transcription(transcription_data)
        
        # 确定输出文件路径
        if output_file_path is None:
            input_path = Path(input_file_path)
            output_file_path = input_path.parent / f"sentence_role_{input_path.name}"
        
        # 保存结果
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"句子级别分角色转录结果已保存到: {output_file_path}")
        
        # 显示处理结果摘要
        role_transcription = result.get('role_based_transcription', [])
        summary = result.get('transcription_summary', {})
        
        print(f"\n处理结果摘要:")
        print(f"- 检测到说话人数量: {summary.get('total_speakers', 0)}")
        print(f"- 合并后的句子段落数: {summary.get('total_segments', 0)}")
        print(f"- 说话人统计: {summary.get('speaker_statistics', {})}")
        
        print(f"\n前3个对话段落预览:")
        for i, segment in enumerate(role_transcription[:3]):
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '')[:50] + ('...' if len(segment.get('text', '')) > 50 else '')
            print(f"  [{speaker}]: {text}")
    
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: JSON文件格式错误 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        sys.exit(1)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python sentence_role_transcriber.py <转录结果JSON文件路径> [输出文件路径]")
        print("示例: python sentence_role_transcriber.py tests/output/transcription_result_spk_extract.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("=" * 60)
    print("句子级别分角色转录输出生成器")
    print("=" * 60)
    print()
    
    process_transcription_file(input_file, output_file)


if __name__ == "__main__":
    main()