#!/bin/bash
# 间接 ANE 验证: 跑 diarize 同时采样 CPU%
#   - CPU% > 200% 持续: 负载在 CPU 上, ANE 几乎没用
#   - CPU% < 100% 但 RTF 仍高: 可能瓶颈在 IO / GPU,但显然不是 ANE 加速带来的
# powermetrics 才是直接证据,但需 sudo;此脚本是无 sudo 兜底
set -e
cd "$(dirname "$0")/.."

AUDIO="tests/fixtures/audio/晚点聊-sample-2person-5min.mp3"
LOG=output/ane_proxy_cpu.log
JSON=output/ane_proxy_diarize.json
STDERR=output/ane_proxy_diarize.stderr

mkdir -p output
: > "$LOG"

# 启 diarize 进程后台
venv/bin/python benchmark/diarize_bench.py "$AUDIO" \
    --num-speakers -1 --cluster-threshold 0.9 \
    --provider coreml --num-threads 4 \
    > "$JSON" 2> "$STDERR" &
PID=$!
echo "diarize PID=$PID start_ts=$(date +%s)"
echo "ts,cpu_pct,rss_mb" >> "$LOG"

# 每 5 秒采样一次, 直到进程结束
while kill -0 "$PID" 2>/dev/null; do
    # ps -o %cpu, %mem — macOS 上 %cpu 可超过 100% (多核归一化)
    STAT=$(ps -p "$PID" -o %cpu=,rss= 2>/dev/null | awk '{cpu=$1; rss=$2/1024; printf "%.1f,%.0f", cpu, rss}')
    if [ -n "$STAT" ]; then
        echo "$(date +%s),$STAT" >> "$LOG"
    fi
    sleep 5
done

# diarize 结束后, 给出 CPU% 分布
echo ""
echo "=== ps 采样汇总 ==="
python3 -c "
import csv
rows = list(csv.DictReader(open('$LOG')))
if not rows:
    print('NO SAMPLES')
else:
    cpus = [float(r['cpu_pct']) for r in rows]
    rsss = [float(r['rss_mb']) for r in rows]
    print(f'samples={len(rows)}')
    print(f'cpu%: min={min(cpus):.0f} max={max(cpus):.0f} avg={sum(cpus)/len(cpus):.0f} (>100% 表示多核, >300% 表示 4+ 核并行)')
    print(f'rss_mb: min={min(rsss):.0f} max={max(rsss):.0f}')
"
echo ""
echo "=== diarize 最终结果 ==="
cat "$STDERR"
