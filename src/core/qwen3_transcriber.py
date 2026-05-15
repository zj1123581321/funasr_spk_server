"""
Qwen3-ASR 1.7B 转录器（PR1 占位实现）

PR1 阶段：本模块仅作 dispatch 入口存在，使 transcriber_dispatch 能解析到 qwen3 名。
真正的模型加载和推理逻辑由 spike 脚本（spikes/qwen3_spike.py）验证后，
在 PR2（或 PR1 后续 commit）中落地。

调用 transcribe 会抛 NotImplementedError，明确告知 PR1 阶段未启用。
"""
from typing import Optional


class Qwen3Transcriber:
    """Qwen3-ASR 转录器占位

    接口形状暂时保持和 FunASRTranscriber.transcribe 一致（待 spike 后确认）：
        async def transcribe(audio_path, task_id, progress_callback, output_format)
            -> (TranscriptionResult, raw_result) 元组 | srt_dict
    """

    def __init__(self):
        # 不做任何重量级加载 —— 真正落地时这里才会加载模型
        self._initialized = False

    async def initialize(self):
        raise NotImplementedError(
            "Qwen3Transcriber 在 PR1 阶段仅为占位。"
            "等待 spike (spikes/qwen3_spike.py) 验证后落地实现。"
        )

    async def transcribe(
        self,
        audio_path: str,
        task_id: str,
        progress_callback=None,
        output_format: str = "json",
    ):
        raise NotImplementedError(
            "Qwen3Transcriber.transcribe 在 PR1 阶段未实现。"
            "等 spike (spikes/qwen3_spike.py) 跑通后再落地。"
        )

    # 同步薄断言入口，仅给测试用 —— 避免测试代码强制 async
    def transcribe_sync_stub(self):
        raise NotImplementedError(
            "Qwen3Transcriber 在 PR1 阶段是占位实现，"
            "需要先跑 spike 验证可行性。"
        )


_qwen3_singleton: Optional[Qwen3Transcriber] = None


def get_qwen3_transcriber() -> Qwen3Transcriber:
    """获取 Qwen3 转录器单例（PR1 阶段返回占位实例）"""
    global _qwen3_singleton
    if _qwen3_singleton is None:
        _qwen3_singleton = Qwen3Transcriber()
    return _qwen3_singleton
