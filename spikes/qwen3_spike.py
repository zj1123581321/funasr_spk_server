"""
Qwen3-ASR 1.7B Spike — 可行性验证脚本

目标（来自 codex review T7）：
- 在写正式集成代码前，先回答几个关键问题：
  1. Qwen3-ASR 1.7B 能加载吗？（model id / 依赖 / 下载耗时）
  2. 单条短音频能跑通推理吗？
  3. 输出结构是什么？（带 timestamps 吗？带 speaker 吗？格式？）
  4. 单进程并发安全吗？（asyncio 同进程多调用）
  5. 资源占用：内存峰值、是否能用 MPS、CPU 占用
  6. 集成复杂度：从 spike 到 production-ready 还要做什么？

使用方式：
    venv/bin/python spikes/qwen3_spike.py \\
        --model-id <modelscope-或-hf-上的-Qwen3-ASR-1.7B-id> \\
        --audio tests/fixtures/audio/tts_1speaker_5s.wav

输出：
    spikes/qwen3_spike_report.md（每跑一次覆盖，可手动归档）

注：
- 本脚本仅用于探索，不入生产代码路径。
- 如果模型 ID 未知，先用 --probe 模式打印一些常见候选地址供查阅。
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# 项目根路径注入（spike 在 src/ 之外）
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Qwen3-ASR 候选模型 ID（需根据实际查阅 modelscope / huggingface 确认）
CANDIDATE_MODEL_IDS = [
    # ModelScope 命名约定
    "Qwen/Qwen3-ASR-1.7B",
    "qwen/Qwen3-ASR-1.7B",
    "iic/Qwen3-ASR-1.7B",
    # HuggingFace 命名约定
    "Qwen/Qwen3-ASR-1.7B-Instruct",
]


def get_memory_mb() -> float:
    """当前进程 RSS 内存 (MB)"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return -1.0


def detect_device() -> str:
    """选择 MPS / CUDA / CPU"""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def probe_candidates():
    """打印候选模型 ID，便于用户挑一个 --model-id 传入"""
    print("=== Qwen3-ASR-1.7B 候选模型 ID（手动验证）===")
    for cid in CANDIDATE_MODEL_IDS:
        print(f"  - {cid}")
    print()
    print("请查阅以下入口确认实际模型 ID 和加载方式：")
    print("  ModelScope: https://modelscope.cn/search?search=Qwen3+ASR")
    print("  HuggingFace: https://huggingface.co/Qwen?search=ASR")
    print()
    print("确认后传入 --model-id 并重跑：")
    print(f"  {sys.argv[0]} --model-id <id> --audio <wav>")


def try_load_via_modelscope(model_id: str, device: str):
    """尝试通过 modelscope.snapshot_download + transformers / funasr 加载"""
    print(f"\n[1/5] 尝试用 modelscope 下载/定位模型: {model_id}")
    t0 = time.time()
    try:
        from modelscope import snapshot_download
        model_dir = snapshot_download(model_id)
        print(f"  ✓ 模型目录: {model_dir}")
        print(f"  耗时: {time.time() - t0:.1f}s")
        return model_dir
    except Exception as e:
        print(f"  ✗ modelscope 加载失败: {type(e).__name__}: {e}")
        return None


def try_load_via_transformers(model_id_or_path: str, device: str):
    """尝试用 transformers AutoModel 加载（多数 Qwen 模型走这条路）"""
    print(f"\n[2/5] 尝试用 transformers 加载: {model_id_or_path}")
    t0 = time.time()
    mem_before = get_memory_mb()
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel
        # 先 processor
        try:
            processor = AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True)
            print(f"  ✓ AutoProcessor 加载成功 ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"  ✗ AutoProcessor 失败: {type(e).__name__}: {e}")
            processor = None
        # 再 model
        t1 = time.time()
        try:
            model = AutoModel.from_pretrained(
                model_id_or_path,
                trust_remote_code=True,
                torch_dtype="auto",
            ).to(device)
            print(f"  ✓ AutoModel 加载成功 ({time.time() - t1:.1f}s)")
        except Exception as e:
            print(f"  ✗ AutoModel 失败: {type(e).__name__}: {e}")
            # 尝试 CausalLM
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id_or_path,
                    trust_remote_code=True,
                    torch_dtype="auto",
                ).to(device)
                print(f"  ✓ AutoModelForCausalLM 加载成功 ({time.time() - t1:.1f}s)")
            except Exception as e2:
                print(f"  ✗ AutoModelForCausalLM 也失败: {type(e2).__name__}: {e2}")
                return None, None

        mem_after = get_memory_mb()
        print(f"  内存增量: {mem_after - mem_before:.0f} MB")
        return processor, model
    except ImportError as e:
        print(f"  ✗ 缺少依赖: {e}")
        return None, None


def try_inference(processor, model, audio_path: Path, device: str):
    """尝试推理"""
    print(f"\n[3/5] 推理: {audio_path}")
    t0 = time.time()
    mem_before = get_memory_mb()
    try:
        # 大多数 ASR 接口是 processor 读音频，model 出文本
        # 但 Qwen3-ASR 的具体调用约定要查它的 model card
        import soundfile as sf
        audio, sr = sf.read(str(audio_path))
        print(f"  音频: {len(audio)/sr:.2f}s, sr={sr}")

        # 通用尝试1：processor(audio, ...) → model.generate(...)
        try:
            inputs = processor(audios=audio, sampling_rate=sr, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=200)
            text = processor.batch_decode(outputs, skip_special_tokens=True)
            print(f"  ✓ 推理成功 ({time.time() - t0:.1f}s)")
            print(f"  输出文本: {text}")
            return {"text": text, "raw": str(outputs)}
        except Exception as e:
            print(f"  ✗ 通用 generate 路径失败: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"  ✗ 推理整体失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None
    finally:
        mem_after = get_memory_mb()
        print(f"  推理内存增量: {mem_after - mem_before:.0f} MB")


def write_report(report_path: Path, info: dict):
    """生成 spike report"""
    lines = [
        "# Qwen3-ASR 1.7B Spike Report",
        "",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"设备: {info.get('device')}",
        f"Python: {sys.version.split()[0]}",
        "",
        "## 关键问答（codex review T7）",
        "",
        f"### Q1. Model ID 与下载",
        f"- 尝试的 model_id: `{info.get('model_id')}`",
        f"- 是否成功定位模型目录: **{info.get('loaded_model_dir') and '✓' or '✗'}**",
        f"- 实际目录: `{info.get('loaded_model_dir', '—')}`",
        "",
        f"### Q2. 模型加载",
        f"- transformers 加载: **{info.get('model_loaded') and '✓' or '✗'}**",
        f"- processor 加载: **{info.get('processor_loaded') and '✓' or '✗'}**",
        f"- 加载内存增量: {info.get('model_memory_mb', '—')} MB",
        "",
        f"### Q3. 推理输出",
        f"- 推理成功: **{info.get('inference_ok') and '✓' or '✗'}**",
        f"- 是否带 timestamps: **{info.get('has_timestamps', '未知')}**",
        f"- 是否带 speaker: **{info.get('has_speaker', '未知（用户表示外部打包）')}**",
        f"- 输出示例: {info.get('output_sample', '—')}",
        "",
        f"### Q4. 并发安全",
        f"- 单进程多调用测试: **未测**（spike v1 仅验单调用）",
        "",
        f"### Q5. 资源占用",
        f"- 加载后内存: {info.get('mem_after_load', '—')} MB",
        f"- 推理时峰值内存增量: {info.get('mem_inference_delta', '—')} MB",
        f"- 设备利用: {info.get('device')}（MPS/CUDA/CPU）",
        "",
        f"### Q6. 集成复杂度估算",
        f"- 当 spike 成功后，从这里到 production-ready Qwen3Transcriber 还要：",
        f"  - [ ] 把 inference 包装到符合 PR1 transcriber 接口的类",
        f"  - [ ] 输出格式适配为 (TranscriptionResult, raw_result)",
        f"  - [ ] 说话人方案集成（用户外部打包）",
        f"  - [ ] 错误处理 + 异常分类（推荐留到 PR2 配合 codex review T5）",
        f"  - [ ] 资源隔离设计（codex T10）",
        f"  - [ ] contract test + parity 验证（条件触发 PR2）",
        "",
        "## 原始日志摘要",
        f"```",
        info.get("log_summary", "（见 stdout）"),
        f"```",
        "",
        "## 结论",
        info.get("conclusion", "（手动填写）"),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[5/5] Report 已生成: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR 1.7B 集成可行性 spike")
    parser.add_argument("--model-id", type=str, help="modelscope/hf model id")
    parser.add_argument("--audio", type=str, help="音频文件路径")
    parser.add_argument("--probe", action="store_true", help="只列出候选 model id 并退出")
    parser.add_argument("--device", type=str, default=None, help="覆盖设备：cpu / mps / cuda")
    parser.add_argument(
        "--report", type=str, default="spikes/qwen3_spike_report.md", help="report 输出路径"
    )
    args = parser.parse_args()

    if args.probe or not args.model_id or not args.audio:
        probe_candidates()
        if not args.probe:
            print("\n⚠ 缺少 --model-id 或 --audio，已进入 probe 模式。")
        return 0

    device = args.device or detect_device()
    print(f"=== Qwen3-ASR Spike ===")
    print(f"device: {device}")
    print(f"model_id: {args.model_id}")
    print(f"audio: {args.audio}")

    info = {"model_id": args.model_id, "device": device}

    # Step 1: 下载/定位
    model_dir = try_load_via_modelscope(args.model_id, device)
    info["loaded_model_dir"] = model_dir

    # Step 2: 加载模型
    load_target = model_dir or args.model_id
    processor, model = try_load_via_transformers(load_target, device)
    info["processor_loaded"] = processor is not None
    info["model_loaded"] = model is not None
    info["mem_after_load"] = f"{get_memory_mb():.0f}"

    # Step 3: 推理
    if processor and model:
        result = try_inference(processor, model, Path(args.audio), device)
        info["inference_ok"] = result is not None
        if result:
            info["output_sample"] = str(result.get("text", ""))[:200]
    else:
        print("\n模型未加载成功，跳过推理。")
        info["inference_ok"] = False

    # Step 4: 写 report
    print(f"\n[4/5] 汇总信息: {json.dumps({k: str(v)[:80] for k, v in info.items()}, ensure_ascii=False)}")
    write_report(Path(args.report), info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
