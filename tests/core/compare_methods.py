#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比测试脚本方法和转录器方法的差异
"""

import os
import sys
import time
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from funasr import AutoModel


def test_direct_method():
    """直接调用FunASR - 与测试脚本完全相同"""
    print("=== 测试直接调用方法 ===")
    
    audio_path = project_root / "samples" / "spk_extract.mp3"
    
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
    
    print("开始转录...")
    result = model.generate(
        input=str(audio_path), 
        batch_size_s=300, 
        hotword=''
    )
    
    print(f"结果类型: {type(result)}")
    print(f"结果内容: {json.dumps(result, ensure_ascii=False, indent=2)}")
    
    return result


def test_cached_method():
    """使用缓存的方法"""
    print("\n=== 测试缓存调用方法 ===")
    
    audio_path = project_root / "samples" / "spk_extract.mp3"
    
    print("加载FunASR完整模型（使用缓存）...")
    model = AutoModel(
        model="paraformer-zh", 
        model_revision="v2.0.4",
        vad_model="fsmn-vad", 
        vad_model_revision="v2.0.4",
        punc_model="ct-punc-c", 
        punc_model_revision="v2.0.4",
        spk_model="cam++", 
        spk_model_revision="v2.0.2",
        # 添加缓存配置
        cache_dir="./models",
        device="cpu",
        disable_update=True,
        disable_pbar=True
    )
    
    print("开始转录...")
    result = model.generate(
        input=str(audio_path), 
        batch_size_s=300, 
        hotword=''
    )
    
    print(f"结果类型: {type(result)}")
    print(f"结果内容: {json.dumps(result, ensure_ascii=False, indent=2)}")
    
    return result


def main():
    """主函数"""
    print("=" * 60)
    print("FunASR 方法对比测试")
    print("=" * 60)
    
    try:
        # 测试直接方法
        direct_result = test_direct_method()
        
        # 测试缓存方法
        cached_result = test_cached_method()
        
        # 对比结果
        print("\n=== 结果对比 ===")
        
        if direct_result and len(direct_result) > 0:
            direct_data = direct_result[0]
            print(f"直接方法结果键: {list(direct_data.keys())}")
            print(f"直接方法有sentence_info: {'sentence_info' in direct_data}")
            if 'sentence_info' in direct_data:
                print(f"直接方法句子数: {len(direct_data['sentence_info'])}")
        
        if cached_result and len(cached_result) > 0:
            cached_data = cached_result[0]
            print(f"缓存方法结果键: {list(cached_data.keys())}")
            print(f"缓存方法有sentence_info: {'sentence_info' in cached_data}")
            if 'sentence_info' in cached_data:
                print(f"缓存方法句子数: {len(cached_data['sentence_info'])}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()