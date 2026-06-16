#!/bin/bash
# 并发 N 任务测试 + powermetrics 监控 GPU/ANE/CPU power
# 用法: 假设 sudo 凭证已 cache (sudo -v),否则会停在密码提示
set -e
cd "$(dirname "$0")/.."

AUDIO=tests/fixtures/audio/晚点聊-sample-2person-5min.mp3
PRESET=auto
DIARIZE_THREADS=4

test_n() {
    local N=$1
    local OUTDIR=output/concurrent_n${N}
    mkdir -p $OUTDIR
    rm -f $OUTDIR/run_*.log $OUTDIR/powermetrics.log

    # 启 powermetrics 后台 (sudo 通过 SUDO_PASS 环境变量, 不落盘)
    # 500ms 间隔, 最多 400 samples = 200s 覆盖
    echo "$SUDO_PASS" | sudo -S powermetrics -s gpu_power,ane_power,cpu_power \
        -i 500 -n 400 > $OUTDIR/powermetrics.log 2>&1 &
    PM_PID=$!
    sleep 1.5  # 给 powermetrics 起飞时间

    echo "[N=$N] launching $N parallel_e2e_bench instances..."
    T_START=$(python3 -c 'import time;print(time.time())')

    PIDS=()
    for i in $(seq 1 $N); do
        venv/bin/python benchmark/parallel_e2e_bench.py \
            $AUDIO \
            --preset $PRESET --diarize-threads $DIARIZE_THREADS \
            --out-dir $OUTDIR/run_${i} \
            > $OUTDIR/run_${i}.log 2>&1 &
        PIDS+=($!)
    done

    # 等所有完成
    for p in "${PIDS[@]}"; do
        wait $p || echo "[N=$N] task pid=$p exited non-zero"
    done
    T_END=$(python3 -c 'import time;print(time.time())')

    # 停 powermetrics — 给它点时间 flush
    sleep 1
    echo "$SUDO_PASS" | sudo -S kill -INT $PM_PID 2>/dev/null || true
    wait $PM_PID 2>/dev/null || true

    WALL=$(python3 -c "print(${T_END} - ${T_START})")
    echo "[N=$N] all tasks done, total wall=${WALL}s"

    # 各任务自己的 wall clock
    for i in $(seq 1 $N); do
        grep "WALL CLOCK" $OUTDIR/run_${i}.log | head -1 | sed "s/^/  run${i}: /"
    done

    # parse powermetrics
    python3 <<PY
import re, statistics
samples = {'CPU':[], 'GPU':[], 'ANE':[]}
total_pow = []
with open("$OUTDIR/powermetrics.log") as f:
    for line in f:
        m = re.match(r'(CPU|GPU|ANE) Power:\s*(\d+)\s*mW', line)
        if m:
            samples[m.group(1)].append(int(m.group(2)))
        m2 = re.match(r'Combined Power.*:\s*(\d+)\s*mW', line)
        if m2:
            total_pow.append(int(m2.group(1)))
print(f"  power samples (CPU={len(samples['CPU'])}, GPU={len(samples['GPU'])}, ANE={len(samples['ANE'])}):")
for k, vs in samples.items():
    if vs:
        avg = sum(vs)/len(vs)
        mx = max(vs)
        # 中位数 + p95 给个稳定的"运行时" reading
        med = statistics.median(vs)
        p95 = sorted(vs)[int(len(vs)*0.95)] if len(vs) >= 20 else mx
        print(f"    {k}: avg={avg/1000:5.2f}W  med={med/1000:5.2f}W  p95={p95/1000:5.2f}W  max={mx/1000:5.2f}W")
if total_pow:
    print(f"    Total combined: avg={sum(total_pow)/len(total_pow)/1000:.2f}W max={max(total_pow)/1000:.2f}W")
PY
    echo ""
}

# warmup (跑一次 N=1 不计数,让 macOS page cache GGUF + ONNX)
echo "=== WARMUP (N=1, 数据忽略) ==="
test_n 1 > /dev/null 2>&1 || true
echo ""

for N in 1 2 3 4; do
    echo "===== N=$N ====="
    test_n $N
done
