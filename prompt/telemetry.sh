#!/bin/bash

# Configuration
if [ -n "$1" ]; then
    LOG_DIR="$1"
    subset="$2"
else
    LOG_DIR="./gpu_logs"
    subset="default_"
fi
LOG_FILE="${LOG_DIR}/${subset}telemetry.csv"
PID_FILE="${LOG_DIR}/nvidia-smi.pid"
GPU_INDEX=0       


mkdir -p "$LOG_DIR"

METRICS="timestamp,name,index,temperature.gpu,power.draw,power.limit,utilization.gpu,utilization.memory,fan.speed,memory.total,memory.used,memory.free,pstate,clocks.gr,clocks.sm,clocks.mem,clocks.video,pcie.link.gen.current,pcie.link.width.current"

# Add a header row to the CSV file
nvidia-smi --query-gpu="$METRICS" --format=csv,nounits -i $GPU_INDEX | head -n 1 > "$LOG_FILE"
echo "Logging GPU telemetry to: $LOG_FILE"

nvidia-smi --query-gpu="$METRICS" \
  --format=csv,noheader,nounits --loop=1 -i $GPU_INDEX >> "$LOG_FILE" &

echo $! > "$PID_FILE"
echo "Started nvidia-smi with PID $(cat $PID_FILE)"

wait