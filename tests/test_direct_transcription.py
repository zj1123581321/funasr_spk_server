"""
直接测试音频转录功能，不依赖服务器
"""
import os
import sys
import json
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from funasr import AutoModel

def test_basic_transcription():
    """测试基础音频转录"""
    print("=" * 60)
    print("测试1: 基础音频转录（不带说话人识别）")
    print("=" * 60)
    
    # 查找测试音频文件
    samples_dir = project_root / "samples"
    audio_files = list(samples_dir.glob("*.wav")) + list(samples_dir.glob("*.mp3")) + list(samples_dir.glob("*.mp4"))
    
    if not audio_files:
        print("错误: 没有找到音频文件")
        return
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 创建基础模型
    print("\n加载基础ASR模型...")
    try:
        model = AutoModel(
            model="paraformer-zh",
            device="cpu",
            disable_update=True,
            disable_pbar=True
        )
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 测试每个音频文件
    results = []
    for audio_file in audio_files[:2]:  # 只测试前2个文件
        print(f"\n处理文件: {audio_file.name}")
        start_time = time.time()
        
        try:
            result = model.generate(input=str(audio_file))
            processing_time = time.time() - start_time
            
            # 解析结果
            if isinstance(result, list) and len(result) > 0:
                data = result[0]
                text = data.get('text', '')
                
                # 保存结果
                result_data = {
                    "file": audio_file.name,
                    "text": text,
                    "processing_time": processing_time,
                    "timestamp_count": len(data.get('timestamp', []))
                }
                results.append(result_data)
                
                print(f"  处理时间: {processing_time:.2f}秒")
                print(f"  文本长度: {len(text)}字符")
                print(f"  文本预览: {text[:100]}..." if len(text) > 100 else f"  文本: {text}")
            
        except Exception as e:
            print(f"  处理失败: {e}")
    
    # 保存结果
    output_file = project_root / "tests" / "basic_transcription_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")
    
    return results


def test_speaker_diarization():
    """测试带说话人识别的转录"""
    print("\n" + "=" * 60)
    print("测试2: 带说话人识别的音频转录")
    print("=" * 60)
    
    # 查找测试音频文件
    samples_dir = project_root / "samples"
    audio_files = list(samples_dir.glob("*.wav")) + list(samples_dir.glob("*.mp3"))
    
    if not audio_files:
        print("错误: 没有找到音频文件")
        return
    
    # 创建带说话人识别的模型
    print("\n加载说话人识别模型...")
    try:
        model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++",
            device="cpu",
            disable_update=True,
            disable_pbar=True
        )
        print("说话人识别模型加载成功")
    except Exception as e:
        print(f"说话人识别模型加载失败: {e}")
        print("尝试不带说话人模型...")
        try:
            model = AutoModel(
                model="paraformer-zh",
                vad_model="fsmn-vad",
                punc_model="ct-punc",
                device="cpu",
                disable_update=True,
                disable_pbar=True
            )
            print("VAD+标点模型加载成功")
        except Exception as e2:
            print(f"模型加载失败: {e2}")
            return
    
    # 测试音频文件
    results = []
    test_file = audio_files[0]
    
    print(f"\n处理文件: {test_file.name}")
    start_time = time.time()
    
    try:
        result = model.generate(
            input=str(test_file),
            batch_size_s=300,
            batch_size_threshold_s=60
        )
        processing_time = time.time() - start_time
        
        # 解析结果
        if isinstance(result, list) and len(result) > 0:
            data = result[0]
            
            # 提取信息
            text = data.get('text', '')
            sentence_info = data.get('sentence_info', [])
            
            # 统计说话人
            speakers = {}
            for sent in sentence_info:
                spk = sent.get('spk', 'Unknown')
                if spk not in speakers:
                    speakers[spk] = 0
                speakers[spk] += 1
            
            # 保存结果
            result_data = {
                "file": test_file.name,
                "text": text,
                "processing_time": processing_time,
                "sentence_count": len(sentence_info),
                "speakers": speakers,
                "sentences": sentence_info[:5]  # 只保存前5句
            }
            results.append(result_data)
            
            print(f"  处理时间: {processing_time:.2f}秒")
            print(f"  句子数: {len(sentence_info)}")
            print(f"  说话人: {list(speakers.keys())}")
            
            # 显示前3句
            print("\n  前3句内容:")
            for i, sent in enumerate(sentence_info[:3]):
                spk = sent.get('spk', 'Unknown')
                text = sent.get('text', '')
                start = sent.get('start', 0) / 1000
                end = sent.get('end', 0) / 1000
                print(f"    {i+1}. [{spk}] {start:.2f}s-{end:.2f}s: {text}")
    
    except Exception as e:
        print(f"  处理失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 保存结果
    output_file = project_root / "tests" / "speaker_diarization_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")
    
    return results


def test_batch_processing():
    """测试批量处理多个音频文件"""
    print("\n" + "=" * 60)
    print("测试3: 批量音频处理")
    print("=" * 60)
    
    # 查找所有音频文件
    samples_dir = project_root / "samples"
    audio_extensions = ['.wav', '.mp3', '.mp4', '.m4a', '.flac']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(samples_dir.glob(f"*{ext}"))
    
    if not audio_files:
        print("错误: 没有找到音频文件")
        return
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 创建模型
    print("\n加载模型...")
    try:
        model = AutoModel(
            model="paraformer-zh",
            device="cpu",
            disable_update=True,
            disable_pbar=True
        )
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 批量处理
    results = []
    total_time = 0
    
    for i, audio_file in enumerate(audio_files):
        print(f"\n[{i+1}/{len(audio_files)}] 处理: {audio_file.name}")
        start_time = time.time()
        
        try:
            result = model.generate(input=str(audio_file))
            processing_time = time.time() - start_time
            total_time += processing_time
            
            if isinstance(result, list) and len(result) > 0:
                data = result[0]
                text = data.get('text', '')
                
                results.append({
                    "file": audio_file.name,
                    "text_length": len(text),
                    "processing_time": processing_time,
                    "success": True
                })
                
                print(f"  成功 - 耗时: {processing_time:.2f}秒")
            else:
                results.append({
                    "file": audio_file.name,
                    "success": False,
                    "error": "无效结果格式"
                })
                print("  失败 - 无效结果格式")
                
        except Exception as e:
            results.append({
                "file": audio_file.name,
                "success": False,
                "error": str(e)
            })
            print(f"  失败 - {e}")
    
    # 统计结果
    success_count = sum(1 for r in results if r.get('success', False))
    print(f"\n批量处理完成:")
    print(f"  总文件数: {len(audio_files)}")
    print(f"  成功: {success_count}")
    print(f"  失败: {len(audio_files) - success_count}")
    print(f"  总耗时: {total_time:.2f}秒")
    print(f"  平均耗时: {total_time/len(audio_files):.2f}秒/文件")
    
    # 保存结果
    output_file = project_root / "tests" / "batch_processing_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_files": len(audio_files),
                "success_count": success_count,
                "fail_count": len(audio_files) - success_count,
                "total_time": total_time,
                "average_time": total_time/len(audio_files) if audio_files else 0
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")
    
    return results


if __name__ == "__main__":
    print("FunASR 直接转录测试")
    print("=" * 60)
    print("此测试不依赖服务器，直接使用 FunASR 进行音频转录")
    print()
    
    # 运行测试
    print("开始测试...\n")
    
    # 测试1: 基础转录
    test_basic_transcription()
    
    # 测试2: 说话人识别
    test_speaker_diarization()
    
    # 测试3: 批量处理
    test_batch_processing()
    
    print("\n所有测试完成！")
    print("结果文件保存在 tests 文件夹中")