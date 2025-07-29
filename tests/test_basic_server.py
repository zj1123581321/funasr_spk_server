"""
测试服务器基本功能
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from src.core.local_transcriber import transcriber

async def test_server():
    """测试服务器基本功能"""
    print("测试服务器基本功能...")
    
    # 初始化转录器
    print("\n1. 初始化转录器...")
    try:
        await transcriber.initialize()
        print("   转录器初始化成功")
    except Exception as e:
        print(f"   转录器初始化失败: {e}")
        return
    
    # 查找测试文件
    test_files = list(Path("samples").glob("*.wav"))
    if not test_files:
        test_files = list(Path("samples").glob("*.mp3"))
    
    if not test_files:
        print("   没有找到测试音频文件")
        return
    
    test_file = test_files[0]
    print(f"\n2. 测试文件: {test_file}")
    
    # 执行转录
    print("\n3. 执行转录...")
    try:
        result = await transcriber.transcribe(
            audio_path=str(test_file),
            task_id="test_001"
        )
        
        print(f"   转录成功!")
        print(f"   - 文件: {result.file_name}")
        print(f"   - 时长: {result.duration:.2f}秒")
        print(f"   - 处理时间: {result.processing_time:.2f}秒")
        print(f"   - 片段数: {len(result.segments)}")
        
        if result.segments:
            print(f"\n4. 转录内容:")
            for i, seg in enumerate(result.segments[:3]):
                print(f"   {i+1}. {seg.text}")
            if len(result.segments) > 3:
                print(f"   ... (还有 {len(result.segments) - 3} 个片段)")
        
    except Exception as e:
        print(f"   转录失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_server())