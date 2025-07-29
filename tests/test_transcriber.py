"""
测试transcriber模块的功能
"""
import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from src.core.transcriber import transcriber
from src.utils.file_utils import get_audio_duration

async def test_transcriber():
    """测试转录器功能"""
    print("开始测试Transcriber模块...")
    
    # 查找音频文件
    samples_dir = Path("samples")
    audio_files = list(samples_dir.glob("*.wav")) + list(samples_dir.glob("*.mp3")) + list(samples_dir.glob("*.mp4"))
    
    if not audio_files:
        print("错误: 没有找到音频文件")
        return
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 初始化转录器
    try:
        print("\n初始化转录器...")
        await transcriber.initialize()
        print("转录器初始化成功")
    except Exception as e:
        print(f"转录器初始化失败: {e}")
        return
    
    # 测试每个音频文件
    for audio_file in audio_files[:2]:  # 只测试前2个文件
        print(f"\n{'='*50}")
        print(f"测试文件: {audio_file.name}")
        
        try:
            # 获取音频时长
            duration = get_audio_duration(str(audio_file))
            print(f"音频时长: {duration:.2f}秒")
        except:
            duration = 0
        
        # 定义进度回调
        async def progress_callback(progress: int):
            print(f"进度: {progress}%")
        
        try:
            # 执行转录
            result = await transcriber.transcribe(
                audio_path=str(audio_file),
                task_id=f"test_{audio_file.stem}",
                progress_callback=progress_callback
            )
            
            # 打印结果
            print(f"\n转录结果:")
            print(f"- 任务ID: {result.task_id}")
            print(f"- 文件名: {result.file_name}")
            print(f"- 时长: {result.duration:.2f}秒")
            print(f"- 处理时间: {result.processing_time:.2f}秒")
            print(f"- 说话人: {', '.join(result.speakers)}")
            print(f"- 片段数: {len(result.segments)}")
            
            # 显示前5个片段
            print("\n前5个片段:")
            for i, segment in enumerate(result.segments[:5]):
                print(f"{i+1}. [{segment.speaker}] {segment.start_time:.2f}s-{segment.end_time:.2f}s: {segment.text}")
            
            if len(result.segments) > 5:
                print(f"... 还有 {len(result.segments) - 5} 个片段")
            
            # 保存完整结果
            import json
            output_file = f"transcription_{audio_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result.dict(), f, ensure_ascii=False, indent=2)
            print(f"\n完整结果已保存到: {output_file}")
            
        except Exception as e:
            print(f"\n转录失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # 运行异步测试
    asyncio.run(test_transcriber())