"""
DirectML (Windows GPU 加速) 设备补丁

修复 FunASR 对 Windows GPU (DirectML) 的支持问题
原始 FunASR 会在没有 CUDA 时强制回退到 CPU，此补丁添加了 DirectML 设备支持
"""
import os
import torch
from loguru import logger


# 全局标志，避免重复应用补丁
_directml_patch_applied = False


def apply_directml_patch():
    """
    应用 DirectML 设备支持补丁到 FunASR

    此补丁修改 FunASR 的 AutoModel.build_model 方法，使其支持 DirectML 设备
    同时应用 SANM 模型修复补丁,解决数据类型兼容性问题
    """
    global _directml_patch_applied

    # 避免重复应用
    if _directml_patch_applied:
        logger.debug("DirectML 补丁已应用，跳过重复应用")
        return

    # 首先应用 SANM 修复补丁(解决 bool 类型 Conv1D 问题)
    from .funasr_sanm_fix import apply_sanm_fix
    apply_sanm_fix()

    # 应用 LSTM CPU 回退补丁(DirectML 不支持 LSTM)
    from .directml_lstm_fallback import apply_lstm_cpu_fallback
    apply_lstm_cpu_fallback()

    try:
        from funasr.auto import auto_model
    except ImportError:
        logger.error("无法导入 FunASR，请确保已安装 funasr 包")
        raise

    try:
        import torch_directml
    except ImportError:
        logger.error("无法导入 torch_directml，请确保已安装 torch-directml 包")
        raise

    # 保存原始方法
    original_build_model = auto_model.AutoModel.build_model

    @staticmethod
    def patched_build_model(**kwargs):
        """
        修复后的 build_model 方法，支持 DirectML 设备

        主要修改：
        1. 在检测到 CUDA 不可用时，尝试使用 DirectML
        2. 移除了强制 batch_size=1 的限制（仅在 CPU 模式下保留）
        3. 支持显式指定 device="dml" 或 device="privateuseone"

        注意：全程使用字符串设备名，避免设备对象在 OmegaConf 中序列化出错
        """
        assert "model" in kwargs

        # 下载模型配置（如果需要）
        if "model_conf" not in kwargs:
            from funasr.download.download_model_from_hub import download_model
            import logging
            logging.info("download models from model hub: {}".format(kwargs.get("hub", "ms")))
            kwargs = download_model(**kwargs)

        # 设置随机种子
        from funasr.train_utils.set_all_random_seed import set_all_random_seed
        set_all_random_seed(kwargs.get("seed", 0))

        # ========== 关键修改：支持 DirectML 设备 ==========
        device = kwargs.get("device", "cuda")

        # 自动设备选择逻辑
        if device == "cuda" and not torch.cuda.is_available():
            # CUDA 不可用，尝试 DirectML
            if torch_directml.is_available():
                # 使用字符串 "privateuseone" (DirectML 在 PyTorch 中注册的后端名称)
                device = "privateuseone"
                logger.debug("CUDA 不可用，自动切换到 DirectML 设备")
            else:
                device = "cpu"
                logger.debug("GPU 不可用，使用 CPU")

        elif device in ["dml", "directml"]:
            # 显式指定 DirectML，检查可用性并统一转换为 "privateuseone"
            if not torch_directml.is_available():
                logger.warning("DirectML 不可用，回退到 CPU")
                device = "cpu"
            else:
                device = "privateuseone"
                logger.debug("使用 DirectML 设备进行加速")

        elif device == "privateuseone":
            # 已经是正确的设备字符串，检查可用性
            if not torch_directml.is_available():
                logger.warning("DirectML 不可用，回退到 CPU")
                device = "cpu"
            else:
                logger.debug("使用 DirectML 设备进行加速")

        # 只有在 CPU 模式下才强制 batch_size=1
        # DirectML 和 CUDA 可以使用更大的 batch_size
        if device == "cpu" and kwargs.get("ngpu", 1) == 0:
            kwargs["batch_size"] = 1

        # 关键：直接使用字符串设备名，不转换为设备对象
        # 这样可以避免 OmegaConf 序列化问题
        kwargs["device"] = device
        torch.set_num_threads(kwargs.get("ncpu", 4))
        # ========== 修改结束 ==========

        # 以下是原始的 FunASR 逻辑，不做修改
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

        # 使用字符串设备名加载模型，PyTorch 会自动识别 "privateuseone" 后端
        model.to(device)

        if not kwargs.get("disable_log", True):
            tables.print()

        return model, kwargs

    # 应用补丁
    auto_model.AutoModel.build_model = patched_build_model
    _directml_patch_applied = True

    logger.debug("DirectML 补丁已成功应用到 FunASR")


def configure_directml_for_multiprocessing(num_workers: int = 1):
    """
    为多进程环境配置 DirectML

    Args:
        num_workers: worker 进程数量

    注意：
    - DirectML 在多进程环境下，每个进程有独立的 GPU 上下文
    - 需要合理控制每个进程的内存使用，避免 OOM
    """
    try:
        import torch_directml

        if not torch_directml.is_available():
            return

        # DirectML 相关环境变量配置（如果需要）
        # 目前 DirectML 不需要特殊的环境变量设置

        logger.info(f"DirectML 多进程配置完成，worker 数量: {num_workers}")

    except ImportError:
        logger.warning("torch_directml 未安装，跳过 DirectML 多进程配置")
