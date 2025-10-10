"""
Windows GPU (DirectML) 加速快速测试脚本

完整流程测试：VAD + ASR + Speaker Diarization + Punctuation
测试 DirectML 对完整转录流程的加速效果
"""
import os
import sys
import time
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

# 配置日志
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


def patch_funasr_directml_support():
    """临时修复 FunASR 的 DirectML 支持问题"""
    from funasr.auto import auto_model
    import torch_directml

    original_build_model = auto_model.AutoModel.build_model

    @staticmethod
    def patched_build_model(**kwargs):
        """修复后的 build_model 方法，支持 DirectML"""
        assert "model" in kwargs

        # 下载模型配置（如果需要）
        # 注意：在下载阶段必须使用字符串设备名，因为 OmegaConf 不支持设备对象
        if "model_conf" not in kwargs:
            from funasr.download.download_model_from_hub import download_model
            import logging
            logging.info("download models from model hub: {}".format(kwargs.get("hub", "ms")))
            # 暂时将设备对象转换为字符串 "cpu"，避免 OmegaConf 错误
            original_device = kwargs.get("device", "cuda")
            if hasattr(original_device, 'type'):  # 是设备对象
                temp_device = "cpu"  # 临时使用 CPU 字符串
            else:
                temp_device = original_device
            kwargs["device"] = temp_device
            kwargs = download_model(**kwargs)
            # 恢复原始设备
            kwargs["device"] = original_device

        from funasr.train_utils.set_all_random_seed import set_all_random_seed
        set_all_random_seed(kwargs.get("seed", 0))

        # 修复：支持 DirectML 设备
        device = kwargs.get("device", "cuda")

        if device == "cuda" and not torch.cuda.is_available():
            if torch_directml.is_available():
                # 设置为字符串 "dml"，补丁会在模型加载时转换为实际设备
                device = "dml"
                logger.info("CUDA 不可用，切换到 DirectML 设备")
            else:
                device = "cpu"
                logger.info("GPU 不可用，使用 CPU")
        elif device in ["dml", "directml", "privateuseone"]:
            if not torch_directml.is_available():
                logger.warning("DirectML 不可用，回退到 CPU")
                device = "cpu"
            else:
                # 统一使用字符串 "dml"
                device = "dml"
                logger.info("使用 DirectML 设备进行加速")

        # 只有在 CPU 模式下才强制 batch_size=1
        if device == "cpu" and kwargs.get("ngpu", 1) == 0:
            kwargs["batch_size"] = 1

        # 将 "dml" 转换为实际的 DirectML 设备对象，用于后续的模型加载
        actual_device_for_kwargs = device
        if device == "dml":
            actual_device_for_kwargs = torch_directml.device()

        kwargs["device"] = actual_device_for_kwargs
        torch.set_num_threads(kwargs.get("ncpu", 4))

        # 继续原始逻辑（构建 tokenizer, frontend, model）
        from funasr.register import tables
        from funasr.utils.misc import deep_update
        from funasr.train_utils.load_pretrained_model import load_pretrained_model
        from omegaconf import ListConfig

        # build tokenizer
        tokenizer = kwargs.get("tokenizer", None)
        kwargs["tokenizer"] = tokenizer
        kwargs["vocab_size"] = -1

        if tokenizer is not None:
            tokenizers = tokenizer.split(",") if isinstance(tokenizer, str) else tokenizer
            tokenizers_conf = kwargs.get("tokenizer_conf", {})
            tokenizers_build = []
            vocab_sizes = []
            token_lists = []

            token_list_files = kwargs.get("token_lists", [])
            seg_dicts = kwargs.get("seg_dicts", [])

            if not isinstance(tokenizers_conf, (list, tuple, ListConfig)):
                tokenizers_conf = [tokenizers_conf] * len(tokenizers)

            for i, tokenizer in enumerate(tokenizers):
                tokenizer_class = tables.tokenizer_classes.get(tokenizer)
                tokenizer_conf = tokenizers_conf[i]

                if len(token_list_files) > 1:
                    tokenizer_conf["token_list"] = token_list_files[i]
                if len(seg_dicts) > 1:
                    tokenizer_conf["seg_dict"] = seg_dicts[i]

                tokenizer = tokenizer_class(**tokenizer_conf)
                tokenizers_build.append(tokenizer)
                token_list = tokenizer.token_list if hasattr(tokenizer, "token_list") else None
                token_list = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else token_list
                vocab_size = -1
                if token_list is not None:
                    vocab_size = len(token_list)

                if vocab_size == -1 and hasattr(tokenizer, "get_vocab_size"):
                    vocab_size = tokenizer.get_vocab_size()
                token_lists.append(token_list)
                vocab_sizes.append(vocab_size)

            if len(tokenizers_build) <= 1:
                tokenizers_build = tokenizers_build[0]
                token_lists = token_lists[0]
                vocab_sizes = vocab_sizes[0]

            kwargs["tokenizer"] = tokenizers_build
            kwargs["vocab_size"] = vocab_sizes
            kwargs["token_list"] = token_lists

        # build frontend
        frontend = kwargs.get("frontend", None)
        kwargs["input_size"] = None
        if frontend is not None:
            frontend_class = tables.frontend_classes.get(frontend)
            frontend = frontend_class(**kwargs.get("frontend_conf", {}))
            kwargs["input_size"] = frontend.output_size() if hasattr(frontend, "output_size") else None
        kwargs["frontend"] = frontend

        # build model
        model_class = tables.model_classes.get(kwargs["model"])
        assert model_class is not None, f'{kwargs["model"]} is not registered'
        model_conf = {}
        deep_update(model_conf, kwargs.get("model_conf", {}))
        deep_update(model_conf, kwargs)
        model = model_class(**model_conf)

        # init_param
        init_param = kwargs.get("init_param", None)
        if init_param is not None:
            if os.path.exists(init_param):
                import logging
                logging.info(f"Loading pretrained params from {init_param}")
                load_pretrained_model(
                    model=model,
                    path=init_param,
                    ignore_init_mismatch=kwargs.get("ignore_init_mismatch", True),
                    oss_bucket=kwargs.get("oss_bucket", None),
                    scope_map=kwargs.get("scope_map", []),
                    excludes=kwargs.get("excludes", None),
                )

        # fp16/bf16
        if kwargs.get("fp16", False):
            model.to(torch.float16)
        elif kwargs.get("bf16", False):
            model.to(torch.bfloat16)

        # 使用 kwargs 中已经转换好的设备对象
        model.to(kwargs["device"])

        if not kwargs.get("disable_log", True):
            tables.print()

        return model, kwargs

    # 应用补丁
    auto_model.AutoModel.build_model = patched_build_model
    logger.success("✅ FunASR DirectML 支持补丁已应用")


def get_audio_duration(audio_path: str) -> float:
    """获取音频时长"""
    import subprocess
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
            capture_output=True,
            text=True
        )
        return float(result.stdout.strip())
    except:
        return 0


def quick_test(audio_path: str, device: str):
    """快速测试指定设备的性能（单次推理）"""
    from funasr import AutoModel
    import json
    import torch_directml

    logger.info(f"\n{'='*60}")
    logger.info(f"测试设备: {device.upper()}")
    logger.info(f"{'='*60}\n")

    try:
        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return None

        audio_duration = get_audio_duration(audio_path)
        logger.info(f"音频文件: {os.path.basename(audio_path)}")
        logger.info(f"音频时长: {audio_duration:.2f} 秒")

        # 初始化完整模型（VAD + ASR + Speaker Diarization + Punctuation）
        logger.info(f"正在初始化完整模型（包含说话人识别）...")
        start_init = time.time()

        # DirectML 需要特殊处理：使用字符串 "dml" 而不是设备对象
        model_device = device
        if device == "dml":
            logger.info(f"使用 DirectML 设备")

        model = AutoModel(
            model="paraformer-zh",
            model_revision="v2.0.4",
            vad_model="fsmn-vad",
            vad_model_revision="v2.0.4",
            punc_model="ct-punc-c",
            punc_model_revision="v2.0.4",
            spk_model="cam++",  # 说话人识别模型
            spk_model_revision="v2.0.2",
            device=model_device,
            disable_update=True,
            disable_pbar=False  # 显示进度条
        )

        init_time = time.time() - start_init
        logger.info(f"模型初始化耗时: {init_time:.2f} 秒")

        # 检查实际使用的设备
        actual_device_used = str(next(model.model.parameters()).device)
        logger.success(f"✅ 实际使用的设备: {actual_device_used}")

        # 单次推理测试
        logger.info("开始推理测试...")
        start_time = time.time()
        result = model.generate(input=audio_path, batch_size_s=300, hotword='')
        inference_time = time.time() - start_time

        rtf = inference_time / audio_duration if audio_duration > 0 else 0

        logger.success(f"\n{'='*60}")
        logger.success(f"测试结果 ({device.upper()})")
        logger.success(f"{'='*60}")
        logger.success(f"推理时间: {inference_time:.2f} 秒")
        logger.success(f"RTF: {rtf:.4f}")
        logger.success(f"速度倍率: {1/rtf:.2f}x")

        # 保存转录结果到文件
        if result and len(result) > 0:
            result_data = result[0]
            text = result_data.get('text', '')

            # 创建输出目录
            output_dir = project_root / "tests" / "performance" / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 1. 保存完整 JSON 结果
            json_file = output_dir / f"transcription_{device}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 完整结果已保存: {json_file}")

            # 2. 保存纯文本
            txt_file = output_dir / f"transcription_{device}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"✅ 纯文本已保存: {txt_file}")

            # 3. 保存带时间戳和说话人信息的句子
            if 'sentence_info' in result_data:
                sentences_file = output_dir / f"transcription_{device}_sentences.txt"
                with open(sentences_file, 'w', encoding='utf-8') as f:
                    for i, sent in enumerate(result_data['sentence_info'], 1):
                        start_ms = sent.get('start', 0)
                        end_ms = sent.get('end', 0)
                        text = sent.get('text', '')

                        # 提取说话人 - 转换为 Speaker1, Speaker2 格式
                        speaker_id = sent.get('spk', 0)
                        if isinstance(speaker_id, int):
                            speaker = f"Speaker{speaker_id + 1}"
                        else:
                            speaker = "Speaker1"

                        # 转换时间格式
                        start_time = f"{start_ms//60000:02d}:{(start_ms%60000)//1000:02d}.{start_ms%1000:03d}"
                        end_time = f"{end_ms//60000:02d}:{(end_ms%60000)//1000:02d}.{end_ms%1000:03d}"

                        f.write(f"[{i}] {start_time} -> {end_time} | {speaker}\n")
                        f.write(f"{text}\n\n")
                logger.info(f"✅ 带时间戳和说话人信息已保存: {sentences_file}")

            logger.info(f"\n转录结果预览:\n{text[:200]}...\n")

        return {
            'device': device,
            'audio_duration': audio_duration,
            'init_time': init_time,
            'inference_time': inference_time,
            'rtf': rtf,
            'speedup': 1/rtf if rtf > 0 else 0,
            'actual_device': actual_device_used,
            'result': result[0] if result else None
        }

    except Exception as e:
        logger.error(f"测试失败 ({device}): {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主测试函数"""
    # 应用 DirectML 支持补丁
    patch_funasr_directml_support()

    # 测试文件路径
    audio_file = os.path.join(project_root, "samples", "test.m4a")

    if not os.path.exists(audio_file):
        logger.error(f"测试文件不存在: {audio_file}")
        logger.info("请将测试音频文件放置在 samples/test.m4a")
        return

    # 检查设备支持
    logger.info("检查设备支持情况...")
    logger.info(f"  CUDA 可用: {torch.cuda.is_available()}")

    try:
        import torch_directml
        logger.info(f"  DirectML 可用: {torch_directml.is_available()}")
        if torch_directml.is_available():
            logger.info(f"  DirectML 设备数量: {torch_directml.device_count()}")
            if torch_directml.device_count() > 0:
                logger.info(f"  DirectML 设备名称: {torch_directml.device_name(0)}")
    except ImportError:
        logger.warning("  torch_directml 未安装")

    results = {}

    # 测试 CPU
    logger.info("\n" + "🔵 测试 1/2: CPU 模式")
    results['cpu'] = quick_test(audio_file, "cpu")

    # 测试 DirectML
    try:
        import torch_directml
        if torch_directml.is_available():
            logger.info("\n" + "🟢 测试 2/2: DirectML 模式")
            results['dml'] = quick_test(audio_file, "dml")
        else:
            logger.warning("⚠️  DirectML 不可用，跳过 DirectML 测试")
    except ImportError:
        logger.warning("⚠️  torch_directml 未安装，跳过 DirectML 测试")

    # 性能对比
    logger.info(f"\n{'='*80}")
    logger.info("性能对比分析")
    logger.info(f"{'='*80}\n")

    if results['cpu'] and results.get('dml'):
        cpu_time = results['cpu']['inference_time']
        dml_time = results['dml']['inference_time']
        speedup = cpu_time / dml_time

        logger.info(f"{'设备':<15} {'推理时间':<15} {'RTF':<15} {'相对加速':<15}")
        logger.info("-" * 80)
        logger.info(f"{'CPU':<15} {cpu_time:>10.2f}s    {results['cpu']['rtf']:>10.4f}    {'1.00x (baseline)':<15}")

        if speedup > 1.0:
            logger.success(f"{'DirectML':<15} {dml_time:>10.2f}s    {results['dml']['rtf']:>10.4f}    {speedup:>10.2f}x")
        else:
            logger.warning(f"{'DirectML':<15} {dml_time:>10.2f}s    {results['dml']['rtf']:>10.4f}    {speedup:>10.2f}x")

        logger.info("\n" + "="*80)
        logger.info("💡 结论与建议")
        logger.info("="*80)

        if speedup > 1.5:
            logger.success(f"✅ DirectML 加速显著！相比 CPU 提升 {speedup:.2f}x")
            logger.info("\n建议操作：")
            logger.info("1. 修改 .env 文件，设置 FUNASR_DEVICE=dml")
            logger.info("2. 或保持 FUNASR_DEVICE=auto，确保 FUNASR_DEVICE_PRIORITY 包含 dml")
        elif speedup > 1.1:
            logger.info(f"⚠️  DirectML 有小幅提升 ({speedup:.2f}x)")
            logger.info("建议：根据实际需求决定是否启用 DirectML")
        else:
            logger.warning(f"❌ DirectML 性能不佳（仅 {speedup:.2f}x）")
            logger.info("建议：继续使用 CPU 模式")

    elif results['cpu']:
        logger.info("仅完成 CPU 测试")

    # 对比转录结果
    if results.get('cpu') and results.get('dml'):
        logger.info("\n" + "="*80)
        logger.info("📄 转录结果对比")
        logger.info("="*80)

        output_dir = project_root / "tests" / "performance" / "output"

        # 对比文本差异
        cpu_txt = output_dir / "transcription_cpu.txt"
        dml_txt = output_dir / "transcription_dml.txt"

        if cpu_txt.exists() and dml_txt.exists():
            with open(cpu_txt, 'r', encoding='utf-8') as f:
                cpu_text = f.read()
            with open(dml_txt, 'r', encoding='utf-8') as f:
                dml_text = f.read()

            if cpu_text == dml_text:
                logger.success("✅ CPU 和 DirectML 转录结果完全一致")
            else:
                logger.warning("⚠️  CPU 和 DirectML 转录结果存在差异")
                logger.info(f"CPU 文本长度: {len(cpu_text)} 字符")
                logger.info(f"DirectML 文本长度: {len(dml_text)} 字符")

                # 简单的差异统计
                import difflib
                diff_ratio = difflib.SequenceMatcher(None, cpu_text, dml_text).ratio()
                logger.info(f"相似度: {diff_ratio*100:.2f}%")

        logger.info(f"\n输出文件位置: {output_dir}")
        logger.info("生成的文件:")
        logger.info("  - transcription_cpu.json (CPU 完整结果)")
        logger.info("  - transcription_cpu.txt (CPU 纯文本)")
        logger.info("  - transcription_cpu_sentences.txt (CPU 带时间戳)")
        logger.info("  - transcription_dml.json (DirectML 完整结果)")
        logger.info("  - transcription_dml.txt (DirectML 纯文本)")
        logger.info("  - transcription_dml_sentences.txt (DirectML 带时间戳)")


if __name__ == "__main__":
    main()
