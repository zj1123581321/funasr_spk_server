#!/usr/bin/env bash
# 远端跑 1800s ort_cuda 转录, 同时 1Hz 抓 nvidia-smi (GPU mem/util) + ps (CPU%/RSS).
# 输出:
#   /tmp/res_e2e.log     -- transcribe stdout
#   /tmp/res_gpu.tsv     -- nvidia-smi 采样 (timestamp, mem_used_MiB, util_pct)
#   /tmp/res_proc.tsv    -- ps 采样 (timestamp, cpu_pct, rss_MB)
set -e

cd "$(dirname "$0")/.."
source venv/bin/activate

NV=venv/lib/python3.12/site-packages/nvidia
TRT=venv/lib/python3.12/site-packages/tensorrt_libs
export LD_LIBRARY_PATH="${TRT}:${NV}/cudnn/lib:${NV}/cublas/lib:${NV}/cufft/lib:${NV}/cuda_runtime/lib:${NV}/cuda_nvrtc/lib:${NV}/nvjitlink/lib:${NV}/cusparse/lib:${NV}/curand/lib:${NV}/cusolver/lib:${NV}/cuda_cupti/lib:${LD_LIBRARY_PATH:-}"

# 启动 transcribe 后台
FUNASR_QWEN3_DIARIZE_BACKEND=ort_cuda \
FUNASR_QWEN3_ASR_ENCODER_PROVIDER=cuda \
python -c "
import asyncio, sys, time
sys.path.insert(0, '.')
from src.core.qwen3_transcriber import get_qwen3_transcriber, reset_qwen3_transcriber_singleton
from src.core.qwen3.diarize_ort import reset_session_cache

async def main():
    reset_qwen3_transcriber_singleton(); reset_session_cache()
    tx = get_qwen3_transcriber()
    await tx.initialize()
    t0 = time.time()
    result, _ = await tx.transcribe(
        audio_path='tests/fixtures/audio/podcast_2speakers_1800s.wav',
        task_id='res-probe-1800s',
        progress_callback=None, output_format='json')
    wall = time.time() - t0
    print(f'wall={wall:.2f}s RTF={wall/result.duration:.3f} speakers={result.speakers} segs={len(result.segments)}')

asyncio.run(main())
" > /tmp/res_e2e.log 2>&1 &
TRANSCRIBE_PID=$!
echo "transcribe PID: $TRANSCRIBE_PID" >&2

# 1Hz 采样 GPU + 进程
: > /tmp/res_gpu.tsv
: > /tmp/res_proc.tsv
echo -e "ts\tmem_used_MiB\tutil_pct" >> /tmp/res_gpu.tsv
echo -e "ts\tcpu_pct\trss_MB" >> /tmp/res_proc.tsv

while kill -0 "$TRANSCRIBE_PID" 2>/dev/null; do
  TS=$(date +%s)
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk -v ts="$TS" -F', ' '{ print ts"\t"$1"\t"$2 }' >> /tmp/res_gpu.tsv
  ps -p "$TRANSCRIBE_PID" -o pcpu=,rss= 2>/dev/null \
    | awk -v ts="$TS" '{ print ts"\t"$1"\t"($2/1024) }' >> /tmp/res_proc.tsv
  sleep 1
done

wait "$TRANSCRIBE_PID" || true
echo "done" >&2
