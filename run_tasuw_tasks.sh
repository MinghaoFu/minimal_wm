#!/bin/bash

set -euo pipefail

PID_FILE="logs/train_tasuw.pids"
JOB_FILE="logs/train_tasuw.jobs"
GPU_IDS="${GPU_IDS:-auto}"
TASKS=(
    cartpole-balance 
    # acrobot-swingup
    # hopper-stand
    # hopper-hop
)
# walker-stand,walker-walk,walker-run,cheetah-run,reacher-easy,reacher-hard,cartpole-swingup,hopper-stand,hopper-hop,quadruped-walk,quadruped-run

if [ "${1:-}" = "-f" ]; then
    echo "Task file mode removed; edit TASKS in this script instead."
    exit 1
fi

select_gpus() {
    if [ "$GPU_IDS" = "auto" ]; then
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
                sort -k2 -n | awk -F',' '{print $1}' | tr -d ' '
        fi
    else
        echo "$GPU_IDS" | tr ',' '\n'
    fi
}

mapfile -t AVAILABLE_GPUS < <(select_gpus)
if [ "${#AVAILABLE_GPUS[@]}" -eq 0 ]; then
    echo "No GPUs detected; tasks will run without CUDA_VISIBLE_DEVICES."
else
    echo "Available GPUs: ${AVAILABLE_GPUS[*]}"
fi

mkdir -p logs
echo "Launched at $(date)" > "$JOB_FILE"
> "$PID_FILE"

gpu_idx=0
for task in "${TASKS[@]}"; do
    log_file="logs/${task}.log"
    echo "Starting task: $task"
    gpu=""
    if [ "${#AVAILABLE_GPUS[@]}" -gt 0 ]; then
        gpu="${AVAILABLE_GPUS[$gpu_idx]}"
        gpu_idx=$(( (gpu_idx + 1) % ${#AVAILABLE_GPUS[@]} ))
        echo "  GPU: $gpu"
    fi
    if [ -n "$gpu" ]; then
        nohup env CUDA_VISIBLE_DEVICES="$gpu" DATASET_NAME="$task" ./train_tasuw.sh > "$log_file" 2>&1 &
    else
        nohup env DATASET_NAME="$task" ./train_tasuw.sh > "$log_file" 2>&1 &
    fi
    pid=$!
    echo "$pid" >> "$PID_FILE"
    printf "%s\t%s\t%s\n" "$pid" "$task" "$log_file" >> "$JOB_FILE"
    echo "  PID: $pid, log: $log_file"
done

echo "PIDs saved to: $PID_FILE"
echo "Jobs saved to: $JOB_FILE"
