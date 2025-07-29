#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试合并功能 - 使用模拟的 FunASR 成功数据
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.funasr_transcriber import FunASRTranscriber
from src.models.schemas import TranscriptionSegment


def create_mock_funasr_result():
    """创建模拟的 FunASR 结果 - 基于真实日志数据"""
    return [{
        'key': 'spk_extract',
        'text': '好，欢迎收听本期的创业内幕。我是主持人lily.本期我们请到的嘉宾来自于国内首家自主研发云CAD公司卡伦特的技术合伙人李荣璐lif李总跟大家打个招呼吧。大家好，我叫李荣录。复旦计算机系极其专业的博士，曾经在这个auto desk担任首位的华人首席工程师，也在这个SAP blackboard等这些公司呢担任首席架程师，然后呢也有将近二十年的软件开发和管理经验。嗯，您当年为什么选择加入卡伦特，这是一个很有意思的问题啊，当年我在复旦读博的时候，确实赶上了人工智能的一个。',
        'timestamp': [[50, 290], [310, 410], [410, 590]],  # 简化版
        'sentence_info': [
            # Speaker1 的连续句子
            {'text': '好，', 'start': 50, 'end': 290, 'timestamp': [[50, 290]], 'spk': 0},
            {'text': '欢迎收听本期的创业内幕。', 'start': 310, 'end': 1710, 'timestamp': [[310, 410]], 'spk': 0},
            {'text': '我是主持人 lily。', 'start': 1710, 'end': 2670, 'timestamp': [[1710, 1830]], 'spk': 0},
            {'text': '本期我们请到的嘉宾来自于国内首家自主研发云 CAD 公司卡伦特的技术合伙人李荣璐 lif 李总跟大家打个招呼吧。', 'start': 2950, 'end': 12490, 'timestamp': [[2950, 3090]], 'spk': 0},
            
            # Speaker2 的连续句子
            {'text': '大家好，', 'start': 12770, 'end': 13330, 'timestamp': [[12770, 12910]], 'spk': 1},
            {'text': '我叫李荣录。', 'start': 13550, 'end': 14410, 'timestamp': [[13550, 13690]], 'spk': 1},
            {'text': '复旦计算机系极其专业的博士，', 'start': 14550, 'end': 16970, 'timestamp': [[14550, 14710]], 'spk': 1},
            {'text': '曾经在这个 auto desk 担任首位的华人首席工程师，', 'start': 17330, 'end': 21710, 'timestamp': [[17330, 17450]], 'spk': 1},
            {'text': '也在这个 SAP blackboard 等这些公司呢担任首席架程师，', 'start': 21950, 'end': 26790, 'timestamp': [[21950, 22130]], 'spk': 1},
            {'text': '然后呢也有将近二十年的软件开发和管理经验。', 'start': 27270, 'end': 31050, 'timestamp': [[27270, 27390]], 'spk': 1},
            {'text': '嗯，', 'start': 31430, 'end': 31670, 'timestamp': [[31430, 31670]], 'spk': 1},
            
            # Speaker1 再次发言
            {'text': '您当年为什么选择加入卡伦特，', 'start': 32110, 'end': 34010, 'timestamp': [[32110, 32330]], 'spk': 0},
            
            # Speaker2 继续回答
            {'text': '这是一个很有意思的问题啊，', 'start': 34010, 'end': 36150, 'timestamp': [[34010, 34250]], 'spk': 1},
            {'text': '当年我在复旦读博的时候，', 'start': 36150, 'end': 38070, 'timestamp': [[36150, 36390]], 'spk': 1},
            {'text': '确实赶上了人工智能的一个，', 'start': 38070, 'end': 39960, 'timestamp': [[38070, 38230]], 'spk': 1}
        ]
    }]


def test_merge_function():
    """测试合并功能"""
    print("=== 测试转录片段合并功能 ===\n")
    
    # 创建转录器实例
    transcriber = FunASRTranscriber()
    
    # 创建模拟数据
    mock_result = create_mock_funasr_result()
    
    print("1. 测试原始解析（不合并）:")
    transcriber._should_merge_segments = lambda: False
    segments_no_merge = transcriber._parse_and_merge_segments(mock_result)
    
    print(f"   原始片段数: {len(segments_no_merge)}")
    for i, seg in enumerate(segments_no_merge[:5]):
        print(f"   {i+1}. [{seg.speaker}] {seg.start_time}s-{seg.end_time}s: {seg.text[:30]}...")
    if len(segments_no_merge) > 5:
        print(f"   ... 还有 {len(segments_no_merge) - 5} 个片段")
    
    print("\n2. 测试合并功能:")
    transcriber._should_merge_segments = lambda: True
    segments_merged = transcriber._parse_and_merge_segments(mock_result)
    
    print(f"   合并后片段数: {len(segments_merged)}")
    for i, seg in enumerate(segments_merged):
        print(f"   {i+1}. [{seg.speaker}] {seg.start_time}s-{seg.end_time}s ({seg.end_time-seg.start_time:.1f}s): {seg.text[:50]}...")
    
    # 保存合并后的结果
    output_data = {
        "merge_test_result": {
            "original_segments": len(segments_no_merge),
            "merged_segments": len(segments_merged),
            "segments": [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "text": seg.text,
                    "speaker": seg.speaker,
                    "duration": round(seg.end_time - seg.start_time, 2)
                }
                for seg in segments_merged
            ]
        }
    }
    
    output_path = project_root / "tests" / "output" / "merge_test_result.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] 合并测试结果已保存到: merge_test_result.json")
    
    # 验证合并效果
    print("\n3. 合并效果验证:")
    speaker1_segments = [seg for seg in segments_merged if seg.speaker == "Speaker1"]
    speaker2_segments = [seg for seg in segments_merged if seg.speaker == "Speaker2"]
    
    print(f"   Speaker1 片段数: {len(speaker1_segments)}")
    print(f"   Speaker2 片段数: {len(speaker2_segments)}")
    
    # 检查是否成功合并了连续的同一说话人片段
    total_original_speaker1 = len([seg for seg in segments_no_merge if seg.speaker == "Speaker1"])
    total_original_speaker2 = len([seg for seg in segments_no_merge if seg.speaker == "Speaker2"])
    
    print(f"   Speaker1 合并率: {total_original_speaker1} -> {len(speaker1_segments)} ({len(speaker1_segments)/total_original_speaker1*100:.1f}%)")
    print(f"   Speaker2 合并率: {total_original_speaker2} -> {len(speaker2_segments)} ({len(speaker2_segments)/total_original_speaker2*100:.1f}%)")
    
    return segments_merged


def main():
    """主函数"""
    print("=" * 60)
    print("FunASR 转录片段合并功能测试")
    print("=" * 60)
    
    try:
        result = test_merge_function()
        print(f"\n测试完成！成功生成 {len(result)} 个合并后的片段。")
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()