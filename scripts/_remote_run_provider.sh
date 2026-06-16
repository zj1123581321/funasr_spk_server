#!/usr/bin/env bash
# 远端调试用: 指定 provider 跑一遍 e2e podcast json 测试, 输出精简 RTF + token-rate
# 用法: bash scripts/_remote_run_provider.sh <provider> [<label>]
set -e

PROVIDER="${1:-cuda}"
LABEL="${2:-${PROVIDER}}"

cd "$(dirname "$0")/.."
source venv/bin/activate

NV=venv/lib/python3.12/site-packages/nvidia
TRT=venv/lib/python3.12/site-packages/tensorrt_libs
export LD_LIBRARY_PATH="${TRT}:${NV}/cudnn/lib:${NV}/cublas/lib:${NV}/cufft/lib:${NV}/cuda_runtime/lib:${NV}/cuda_nvrtc/lib:${NV}/nvjitlink/lib:${NV}/cusparse/lib:${NV}/curand/lib:${NV}/cusolver/lib:${NV}/cuda_cupti/lib:${LD_LIBRARY_PATH:-}"

echo "=== $LABEL run ==="
FUNASR_RUN_INTEGRATION=1 \
FUNASR_DEFAULT_ENGINE=qwen3 \
FUNASR_QWEN3_PROVIDER=cuda \
FUNASR_QWEN3_ASR_ENCODER_PROVIDER="$PROVIDER" \
PYTHONUNBUFFERED=1 \
python -m pytest tests/integration/test_qwen3_diarize_e2e.py::TestQwen3DiarizeEndToEnd::test_podcast_2speakers_json_mode \
    -s --tb=line 2>&1 \
  | grep -E "Encoder\\]|encoder-timing|RTF|prefill|generate|passed|failed|FAILED|Error|tensorrt|TensorRT|fallback|sherpa|encode_time|elapsed=|耗时|总耗时|Qwen3 转录完成|qwen3-e2e" \
  | tail -50
