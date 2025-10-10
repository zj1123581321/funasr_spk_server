"""
DirectML 错误追踪脚本

在 FunASR 推理过程中插入调试代码,精确定位错误位置
"""
import sys
import io
import os
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch_directml

# ===== 猴子补丁:拦截 Conv1D 调用 =====
original_conv1d = torch.nn.functional.conv1d

def debug_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """调试版本的 conv1d,记录所有调用"""
    try:
        print(f"\n[DEBUG Conv1D]")
        print(f"  input: shape={input.shape}, dtype={input.dtype}, device={input.device}")
        print(f"  weight: shape={weight.shape}, dtype={weight.dtype}, device={weight.device}")
        print(f"  bias: {bias.shape if bias is not None else None}")
        print(f"  stride={stride}, padding={padding}, dilation={dilation}, groups={groups}")

        result = original_conv1d(input, weight, bias, stride, padding, dilation, groups)
        print(f"  output: shape={result.shape}, dtype={result.dtype}")
        print(f"  [成功]")
        return result
    except Exception as e:
        print(f"  [失败] {type(e).__name__}: {e}")
        raise

# 应用猴子补丁
torch.nn.functional.conv1d = debug_conv1d
print("✅ Conv1D 调试补丁已应用\n")

# ===== 加载模型并测试 =====
from src.core.patches.directml_patch import apply_directml_patch
apply_directml_patch()

from funasr import AutoModel

print("=" * 70)
print("DirectML 错误追踪")
print("=" * 70)

# 加载模型
print("\n加载模型...")
model = AutoModel(
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

print(f"\n模型设备: {next(model.model.parameters()).device}")

# 推理
audio_file = str(project_root / "samples" / "test.m4a")
print(f"\n音频文件: {audio_file}")
print("\n开始推理...(将显示所有 Conv1D 调用)\n")
print("=" * 70)

try:
    result = model.generate(input=audio_file, batch_size_s=300, hotword='')
    print("\n" + "=" * 70)
    print("✅ 推理成功!")
except Exception as e:
    print("\n" + "=" * 70)
    print(f"❌ 推理失败")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")
    import traceback
    traceback.print_exc()
