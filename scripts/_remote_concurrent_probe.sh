#!/usr/bin/env bash
# 2 个独立 transcribe 进程并发 1800s ort_cuda, 同步启动, 1Hz 抓 GPU/CPU/RSS.
# 用 file barrier 让两个 worker model load 完成后再同时开始 transcribe.
#
# 输出:
#   /tmp/c1.log /tmp/c2.log  -- 每个 worker stdout (含 WORKER= 行)
#   /tmp/c_gpu.tsv           -- GPU 整体采样 (ts, mem_used_MiB, util_pct)
#   /tmp/c_proc.tsv          -- 每个 PID 1Hz 采样 (ts, pid, cpu_pct, rss_MB)
#   /tmp/c_total_wall        -- 同步开始后到两个 worker 都退出的 wall

set -e

cd "$(dirname "$0")/.."
source venv/bin/activate

NV=venv/lib/python3.12/site-packages/nvidia
TRT=venv/lib/python3.12/site-packages/tensorrt_libs
export LD_LIBRARY_PATH="${TRT}:${NV}/cudnn/lib:${NV}/cublas/lib:${NV}/cufft/lib:${NV}/cuda_runtime/lib:${NV}/cuda_nvrtc/lib:${NV}/nvjitlink/lib:${NV}/cusparse/lib:${NV}/curand/lib:${NV}/cusolver/lib:${NV}/cuda_cupti/lib:${LD_LIBRARY_PATH:-}"

BARRIER=/tmp/concurrent_barrier
AUDIO=tests/fixtures/audio/podcast_2speakers_1800s.wav

# 清旧文件
rm -f "${BARRIER}".* /tmp/c1.log /tmp/c2.log /tmp/c_gpu.tsv /tmp/c_proc.tsv /tmp/c_total_wall

# 启动 2 个 worker (背靠背启动)
python scripts/_remote_concurrent_worker.py 1 "$AUDIO" "$BARRIER" > /tmp/c1.log 2>&1 &
PID1=$!
python scripts/_remote_concurrent_worker.py 2 "$AUDIO" "$BARRIER" > /tmp/c2.log 2>&1 &
PID2=$!
echo "launched PID1=$PID1 PID2=$PID2" >&2

# 等两个 worker 都 ready (model load 完)
while [ ! -f "${BARRIER}.ready.1" ] || [ ! -f "${BARRIER}.ready.2" ]; do
  # 死亡检查 - 任一 worker 提前死要立即退出
  if ! kill -0 $PID1 2>/dev/null && [ ! -f "${BARRIER}.ready.1" ]; then
    echo "PID1 died before ready" >&2; exit 1
  fi
  if ! kill -0 $PID2 2>/dev/null && [ ! -f "${BARRIER}.ready.2" ]; then
    echo "PID2 died before ready" >&2; exit 1
  fi
  sleep 0.5
done
echo "both ready, releasing barrier" >&2

# 同步开始
T0=$(date +%s.%N)
touch "${BARRIER}.go"

# 1Hz 采样
echo -e "ts\tmem_used_MiB\tutil_pct" > /tmp/c_gpu.tsv
echo -e "ts\tpid\tcpu_pct\trss_MB" > /tmp/c_proc.tsv

while kill -0 $PID1 2>/dev/null || kill -0 $PID2 2>/dev/null; do
  TS=$(date +%s)
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk -v ts="$TS" -F', ' '{ print ts"\t"$1"\t"$2 }' >> /tmp/c_gpu.tsv
  for PID in $PID1 $PID2; do
    if kill -0 $PID 2>/dev/null; then
      ps -p $PID -o pcpu=,rss= 2>/dev/null \
        | awk -v ts="$TS" -v pid="$PID" '{ print ts"\t"pid"\t"$1"\t"($2/1024) }' >> /tmp/c_proc.tsv
    fi
  done
  sleep 1
done

T1=$(date +%s.%N)
echo "scale=2; $T1 - $T0" | bc -l > /tmp/c_total_wall
echo "total_wall=$(cat /tmp/c_total_wall)s" >&2

wait $PID1 || true
wait $PID2 || true

echo "=== worker logs ===" >&2
grep "^WORKER=" /tmp/c1.log /tmp/c2.log || echo "no WORKER= line found (workers may have crashed)" >&2
