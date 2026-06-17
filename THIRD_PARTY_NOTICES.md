# 第三方声明 (Third-Party Notices)

本项目(MIT,见根 `LICENSE`)集成/依赖以下第三方代码、库与模型权重。各项保留其自身
许可证。本清单尽力而为,**下游使用者(尤其商业用途)请以各上游仓库 / 模型卡的最新条款为准**。

---

## 一、随仓库分发的 vendored 源码

| 路径 | 来源 | 许可证 |
|---|---|---|
| `src/core/vendor/qwen_asr_gguf/` | [HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline) | MIT (Copyright © 2026 Haujet Zhao,见该目录 `LICENSE`) |

---

## 二、Python 依赖库(pip 安装,不随仓库分发)

| 库 | 许可证 |
|---|---|
| funasr | MIT |
| sherpa-onnx | Apache-2.0 |
| modelscope | Apache-2.0 |
| torch / torchaudio | BSD-3-Clause |
| onnxruntime / onnxruntime-gpu | MIT |
| pydantic | MIT |
| httpx | BSD-3-Clause |
| websockets | BSD-3-Clause |
| loguru | MIT |
| ffmpeg-python | Apache-2.0 |
| ctc-forced-aligner (deskpai ONNX fork) | 见上游仓库条款 |

> **ffmpeg 二进制**:本项目通过子进程调用系统 `ffmpeg`(非链接),其本体为 LGPL/GPL
> (取决于构建),由使用者自行安装,不随本仓库分发。

完整依赖见 `requirements.txt`。

---

## 三、模型权重(运行时下载,**不在本仓库内**)

模型权重不随仓库分发,由 `scripts/download_qwen3_models.sh` 等在运行时拉取。其许可证
与本仓库代码许可证**相互独立**,使用者需各自遵守对应模型卡条款:

| 模型 | 用途 | 许可证(以模型卡为准) |
|---|---|---|
| **MMS-300M (CTC Forced Aligner)** | word_align 词级时间戳 | ⚠️ **CC-BY-NC-4.0(仅限非商业)** |
| pyannote segmentation-3.0 | diarize 分段 | 代码 MIT;模型需 HuggingFace 接受条款 |
| NeMo TitaNet (speaker embedding) | diarize 嵌入 | 见 NVIDIA NGC 模型卡(通常 CC-BY-4.0) |
| FunASR Paraformer / CAM++ | FunASR 引擎 ASR / 说话人 | 见 ModelScope 模型卡 |
| Qwen3 ASR (GGUF/ONNX) | Qwen3 引擎 ASR | 见 Qwen 模型卡(Apache-2.0 / Tongyi Qianwen 视版本) |

> ⚠️ **商用提示**:其中 **MMS-300M 为 CC-BY-NC-4.0,禁止商业使用**。本项目当前定位为
> 个人/非商业自用,故可用;若将本服务用于商业场景,需将 word_align 替换为可商用的对齐
> 方案,并逐一复核上述各模型卡的商用条款。
