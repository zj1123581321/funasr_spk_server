#!/usr/bin/env bash
# 单进程 2 实例 + asyncio.gather 并发跑 1800s, 1Hz 抓 GPU/CPU/RSS.
#
# 跟 _remote_concurrent_probe.sh 的差异: 这个只起 1 个 python 进程,
# 内部 new 2 个 Qwen3DiarizeTranscriber 实例 + asyncio.gather.
#
# 输出:
#   /tmp/sp.log              -- python stdout (含 TOTAL_WALL + TASK_OK/FAILED)
#   /tmp/sp_gpu.tsv          -- GPU 整体采样 (ts, mem_used_MiB, util_pct)
#   /tmp/sp_proc.tsv         -- 该 PID 1Hz 采样 (ts, cpu_pct, rss_MB)

set -e

cd "$(dirname "$0")/.."
source venv/bin/activate

NV=venv/lib/python3.12/site-packages/nvidia
TRT=venv/lib/python3.12/site-packages/tensorrt_libs
export LD_LIBRARY_PATH="${TRT}:${NV}/cudnn/lib:${NV}/cublas/lib:${NV}/cufft/lib:${NV}/cuda_runtime/lib:${NV}/cuda_nvrtc/lib:${NV}/nvjitlink/lib:${NV}/cusparse/lib:${NV}/curand/lib:${NV}/cusolver/lib:${NV}/cuda_cupti/lib:${LD_LIBRARY_PATH:-}"

AUDIO="${1:-tests/fixtures/audio/podcast_2speakers_1800s.wav}"
COUNT="${2:-2}"

rm -f /tmp/sp.log /tmp/sp_gpu.tsv /tmp/sp_proc.tsv

python scripts/_remote_single_proc_concurrent_probe.py "$AUDIO" "$COUNT" > /tmp/sp.log 2>&1 &
PID=$!
echo "launched PID=$PID audio=$AUDIO count=$COUNT" >&2

echo -e "ts\tmem_used_MiB\tutil_pct" > /tmp/sp_gpu.tsv
echo -e "ts\tcpu_pct\trss_MB" > /tmp/sp_proc.tsv

T0=$(date +%s)
while kill -0 $PID 2>/dev/null; do
  TS=$(date +%s)
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk -v ts="$TS" -F', ' '{ print ts"\t"$1"\t"$2 }' >> /tmp/sp_gpu.tsv
  if kill -0 $PID 2>/dev/null; then
    ps -p $PID -o pcpu=,rss= 2>/dev/null \
      | awk -v ts="$TS" '{ print ts"\t"$1"\t"($2/1024) }' >> /tmp/sp_proc.tsv
  fi
  sleep 1
done
T1=$(date +%s)
echo "wall=$((T1 - T0))s" >&2
wait $PID || true
echo "=== sp.log tail ===" >&2
tail -25 /tmp/sp.log
