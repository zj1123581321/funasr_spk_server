#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 src/core/funasr_transcriber.py 的转录功能
"""

import os
import sys
import asyncio
import json
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入要测试的转录器
from src.core.funasr_transcriber import FunASRTranscriber


async def test_transcriber():
    """测试转录器核心功能"""
    print("=== 测试 FunASR 转录器核心功能 ===\n")
    
    # 查找音频文件
    samples_dir = project_root / "samples"
    audio_files = []
    for ext in ['.wav', '.mp3', '.mp4', '.m4a', '.flac']:
        audio_files.extend(samples_dir.glob(f"*{ext}"))
    
    if not audio_files:
        print("错误: 在 samples 文件夹中没有找到音频文件")
        return
    
    # 创建转录器实例
    transcriber = FunASRTranscriber()
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n=== 测试文件 {i}/{len(audio_files)}: {audio_file.name} ===")
        
        try:
            start_time = time.time()
            
            # 进度回调函数
            def progress_callback(progress):
                print(f"进度: {progress}%")
            
            # 调用转录
            result = await transcriber.transcribe(
                audio_path=str(audio_file),
                task_id=f"test-{i}",
                progress_callback=progress_callback,
                enable_speaker=True
            )
            
            processing_time = time.time() - start_time
            
            # 显示结果
            print(f"\n转录结果:")
            print(f"- 任务ID: {result.task_id}")
            print(f"- 文件名: {result.file_name}")
            print(f"- 文件哈希: {result.file_hash}")
            print(f"- 音频时长: {result.duration:.2f}秒")
            print(f"- 处理时间: {result.processing_time:.2f}秒")
            print(f"- 检测到说话人: {result.speakers}")
            print(f"- 转录片段数: {len(result.segments)}")
            
            # 显示转录片段
            print(f"\n转录片段详情:")
            for j, segment in enumerate(result.segments[:5]):  # 只显示前5个片段
                print(f"  {j+1}. [{segment.speaker}] {segment.start_time}s-{segment.end_time}s: {segment.text}")
            
            if len(result.segments) > 5:
                print(f"  ... 还有 {len(result.segments) - 5} 个片段")
            
            # 保存结果到JSON文件
            output_data = {
                "task_id": result.task_id,
                "file_name": result.file_name,
                "file_hash": result.file_hash,
                "duration": result.duration,
                "processing_time": result.processing_time,
                "speakers": result.speakers,
                "segments": [
                    {
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "text": seg.text,
                        "speaker": seg.speaker,
                        "duration": round(seg.end_time - seg.start_time, 2)
                    }
                    for seg in result.segments
                ],
                "transcription_summary": {
                    "total_speakers": len(result.speakers),
                    "total_segments": len(result.segments),
                    "full_text": " ".join([seg.text for seg in result.segments])
                }
            }
            
            # 保存到文件
            output_filename = f"core_test_result_{audio_file.stem}.json"
            output_path = project_root / "tests" / "output" / output_filename
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n[OK] 测试结果已保存到: {output_filename}")
            print(f"  完整路径: {output_path}")
            
            # 验证结果质量
            print(f"\n结果验证:")
            if len(result.segments) > 0:
                print(f"[OK] 成功生成 {len(result.segments)} 个转录片段")
            else:
                print("[WARN] 警告: 没有生成任何转录片段")
            
            if len(result.speakers) > 1:
                print(f"[OK] 成功识别 {len(result.speakers)} 个说话人")
            else:
                print(f"[INFO] 只识别到 {len(result.speakers)} 个说话人")
            
            total_text = " ".join([seg.text for seg in result.segments])
            if len(total_text) > 10:
                print(f"[OK] 转录文本长度: {len(total_text)} 字符")
            else:
                print("[WARN] 警告: 转录文本过短")
            
        except Exception as e:
            print(f"[ERROR] 测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 60)
    
    print("\n=== 测试完成 ===")


async def main():
    """主函数"""
    print("=" * 60)
    print("FunASR 转录器核心功能测试")
    print("=" * 60)
    
    # 设置调试级别日志
    from loguru import logger
    import sys
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    
    try:
        await test_transcriber()
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行异步测试
    asyncio.run(main())