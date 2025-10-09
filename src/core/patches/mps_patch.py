"""
MPS (Metal Performance Shaders) 设备补丁

修复 FunASR 对 Apple Silicon GPU 的支持问题
原始 FunASR 会在没有 CUDA 时强制回退到 CPU，此补丁添加了 MPS 设备支持
"""
import os
import torch
from loguru import logger


# 全局标志，避免重复应用补丁
_mps_patch_applied = False


def apply_mps_patch():
    """
    应用 MPS 设备支持补丁到 FunASR

    此补丁修改 FunASR 的 AutoModel.build_model 方法，使其支持 MPS 设备
    """
    global _mps_patch_applied

    # 避免重复应用
    if _mps_patch_applied:
        logger.debug("MPS 补丁已应用，跳过重复应用")
        return

    try:
        from funasr.auto import auto_model
    except ImportError:
        logger.error("无法导入 FunASR，请确保已安装 funasr 包")
        raise

    # 保存原始方法
    original_build_model = auto_model.AutoModel.build_model

    @staticmethod
    def patched_build_model(**kwargs):
        """
        修复后的 build_model 方法，支持 MPS 设备

        主要修改：
        1. 在检测到 CUDA 不可用时，尝试使用 MPS
        2. 移除了强制 batch_size=1 的限制（仅在 CPU 模式下保留）
        3. 支持显式指定 device="mps"
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

        # ========== 关键修改：支持 MPS 设备 ==========
        device = kwargs.get("device", "cuda")

        # 自动设备选择逻辑
        if device == "cuda" and not torch.cuda.is_available():
            # CUDA 不可用，尝试 MPS
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps"
                logger.debug("CUDA 不可用，自动切换到 MPS 设备")
            else:
                device = "cpu"
                logger.debug("GPU 不可用，使用 CPU")

        elif device == "mps":
            # 显式指定 MPS，检查可用性
            if not torch.backends.mps.is_available() or not torch.backends.mps.is_built():
                logger.warning("MPS 不可用，回退到 CPU")
                device = "cpu"
            else:
                logger.debug("使用 MPS 设备进行加速")

        # 只有在 CPU 模式下才强制 batch_size=1
        # MPS 和 CUDA 可以使用更大的 batch_size
        if device == "cpu" and kwargs.get("ngpu", 1) == 0:
            kwargs["batch_size"] = 1

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
        model.to(device)

        if not kwargs.get("disable_log", True):
            tables.print()

        return model, kwargs

    # 应用补丁
    auto_model.AutoModel.build_model = patched_build_model
    _mps_patch_applied = True

    logger.debug("MPS 补丁已成功应用到 FunASR")


def configure_mps_for_multiprocessing(num_workers: int = 1):
    """
    为多进程环境配置 MPS

    Args:
        num_workers: worker 进程数量

    注意：
    - MPS 在多进程环境下，每个进程有独立的 GPU 上下文
    - 需要合理控制每个进程的内存使用，避免 OOM
    """
    if not torch.backends.mps.is_available():
        return

    # 设置 MPS 相关环境变量（如果需要）
    # 注意：这些环境变量需要在进程启动前设置
    if "PYTORCH_MPS_PREFER_METAL" not in os.environ:
        os.environ["PYTORCH_MPS_PREFER_METAL"] = "1"
        logger.debug("设置 PYTORCH_MPS_PREFER_METAL=1")

    logger.info(f"MPS 多进程配置完成，worker 数量: {num_workers}")
