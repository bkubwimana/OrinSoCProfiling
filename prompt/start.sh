#!/usr/bin/env bash
sudo nsys profile --gpu-metrics-devices=1 --force-overwrite=true --capture-range=nvtx --nvtx-capture=TimeCapture --cuda-event-trace=false --trace=cuda,nvtx,cudnn -o deeepseek_prof ./run.sh

if [ $? -eq 0 ]; then
    echo "Profiling completed successfully. Processing tegrastats output..."
else
    echo "Profiling failed with exit code $?"
    exit 1
fi

echo "All tasks completed."