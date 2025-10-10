"""
DirectML FunASR 最小化测试

逐步测试 FunASR 各个组件的 DirectML 兼容性
"""
import sys
import io
import os
from pathlib import Path

# 设置输出编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("DirectML FunASR 最小化测试")
print("=" * 70)

# 测试1: DirectML 基础功能
print("\n【测试1】DirectML 基础功能")
print("-" * 70)

import torch
import torch_directml

print(f"PyTorch: {torch.__version__}")
print(f"DirectML 可用: {torch_directml.is_available()}")
print(f"DirectML 设备: {torch_directml.device_name(0) if torch_directml.is_available() else 'N/A'}")

# 测试2: 应用 DirectML 补丁
print("\n【测试2】应用 DirectML 补丁")
print("-" * 70)

from src.core.patches.directml_patch import apply_directml_patch
apply_directml_patch()
print("✅ DirectML 补丁已应用")

# 测试3: 导入 FunASR (不加载模型)
print("\n【测试3】导入 FunASR")
print("-" * 70)

try:
    from funasr import AutoModel
    print("✅ FunASR 导入成功")
except Exception as e:
    print(f"❌ FunASR 导入失败: {e}")
    sys.exit(1)

# 测试4: 加载单个模型 (ASR 模型)
print("\n【测试4】加载 ASR 模型 (仅主模型)")
print("-" * 70)

try:
    # 只加载 ASR 主模型,不加载 VAD/PUNC/SPK
    model = AutoModel(
        model="paraformer-zh",
        model_revision="v2.0.4",
        device="dml",
        disable_update=True,
        disable_pbar=True
    )

    # 检查设备
    actual_device = str(next(model.model.parameters()).device)
    print(f"✅ ASR 模型加载成功")
    print(f"   模型设备: {actual_device}")

    # 清理
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

except Exception as e:
    print(f"❌ ASR 模型加载失败")
    print(f"   错误类型: {type(e).__name__}")
    print(f"   错误信息: {e}")
    import traceback
    traceback.print_exc()

# 测试5: 加载完整模型 (ASR + VAD)
print("\n【测试5】加载完整模型 (ASR + VAD)")
print("-" * 70)

try:
    model_vad = AutoModel(
        model="paraformer-zh",
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4",
        device="dml",
        disable_update=True,
        disable_pbar=True
    )

    actual_device = str(next(model_vad.model.parameters()).device)
    print(f"✅ ASR+VAD 模型加载成功")
    print(f"   模型设备: {actual_device}")

    del model_vad

except Exception as e:
    print(f"❌ ASR+VAD 模型加载失败")
    print(f"   错误类型: {type(e).__name__}")
    print(f"   错误信息: {e}")
    import traceback
    traceback.print_exc()

# 测试6: 加载完整模型 (ASR + VAD + PUNC)
print("\n【测试6】加载完整模型 (ASR + VAD + PUNC)")
print("-" * 70)

try:
    model_full = AutoModel(
        model="paraformer-zh",
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4",
        punc_model="ct-punc-c",
        punc_model_revision="v2.0.4",
        device="dml",
        disable_update=True,
        disable_pbar=True
    )

    actual_device = str(next(model_full.model.parameters()).device)
    print(f"✅ ASR+VAD+PUNC 模型加载成功")
    print(f"   模型设备: {actual_device}")

    del model_full

except Exception as e:
    print(f"❌ ASR+VAD+PUNC 模型加载失败")
    print(f"   错误类型: {type(e).__name__}")
    print(f"   错误信息: {e}")
    import traceback
    traceback.print_exc()

# 测试7: 加载所有模型 (包括说话人识别)
print("\n【测试7】加载所有模型 (ASR + VAD + PUNC + SPK)")
print("-" * 70)

try:
    model_all = AutoModel(
        model="paraformer-zh",
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4",
        punc_model="ct-punc-c",
        punc_model_revision="v2.0.4",
        spk_model="cam++",
        spk_model_revision="v2.0.2",
        device="dml",
        disable_update=True,
        disable_pbar=True
    )

    actual_device = str(next(model_all.model.parameters()).device)
    print(f"✅ 所有模型加载成功")
    print(f"   主模型设备: {actual_device}")

    # 尝试简单推理
    print("\n【测试8】简单推理测试")
    print("-" * 70)

    audio_file = str(project_root / "samples" / "test.m4a")
    if os.path.exists(audio_file):
        print(f"音频文件: {audio_file}")
        print("开始推理...")

        result = model_all.generate(
            input=audio_file,
            batch_size_s=300,
            hotword=''
        )

        print(f"✅ 推理成功")
        if result and len(result) > 0:
            text = result[0].get('text', '')
            print(f"   转录文本长度: {len(text)} 字符")
            print(f"   文本预览: {text[:100]}...")
    else:
        print(f"⚠️  测试文件不存在: {audio_file}")

except Exception as e:
    print(f"❌ 失败")
    print(f"   错误类型: {type(e).__name__}")
    print(f"   错误信息: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
