#!/bin/bash

# Unified runner for UWM variants (tcwm / tcwm_wan / dinowm)
# - Single task: default foreground (debug) run
# - Multiple tasks: default background (nohup) run
# - DEBUG=true forces foreground; DEBUG=false forces background
#
# Usage:
#   FRAMEWORK=tcwm TASKS=lift ./run_uwm.sh
#   FRAMEWORK=dinowm TASKS=lift,pusht ./run_uwm.sh
#   CONFIG_NAME=train_tcwm_wan TASKS=lift ./run_uwm.sh
#   DEBUG=false FRAMEWORK=tcwm TASKS=lift ./run_uwm.sh

set -euo pipefail

PID_FILE="logs/run_uwm.pids"
JOB_FILE="logs/run_uwm.jobs"
GPU_IDS="${GPU_IDS:-auto}"
TASKS_RAW="${TASKS:-}"
FRAMEWORK="${FRAMEWORK:-tcwm}" # tcwm, tcwm_wan, dinowm
CONFIG_NAME="${CONFIG_NAME:-}"

DEFAULT_TASKS=(
  lift
  # can
  # square
  pusht
)

if [ -z "$CONFIG_NAME" ]; then
  case "$FRAMEWORK" in
    tcwm)
      CONFIG_NAME="train_tcwm"
      ;;
    tcwm_wan)
      CONFIG_NAME="train_tcwm_wan"
      ;;
    dinowm)
      CONFIG_NAME="train_dinowm"
      ;;
    *)
      echo "❌ Unknown FRAMEWORK: $FRAMEWORK (expected tcwm | tcwm_wan | dinowm)"
      exit 1
      ;;
  esac
fi

if [ -n "$TASKS_RAW" ]; then
  IFS=',' read -r -a TASKS <<< "$TASKS_RAW"
else
  TASKS=("${DEFAULT_TASKS[@]}")
fi

if [ "${#TASKS[@]}" -eq 0 ]; then
  echo "❌ No tasks specified. Set TASKS=task1,task2 or edit DEFAULT_TASKS in run_uwm.sh."
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

num_tasks="${#TASKS[@]}"
debug_flag="${DEBUG:-}"

if [ -z "$debug_flag" ]; then
  if [ "$num_tasks" -eq 1 ]; then
    debug_flag="true"
  else
    debug_flag="false"
  fi
fi

run_foreground() {
  local task="$1"
  local gpu="$2"
  echo "Starting task (foreground): $task"
  echo "  CONFIG_NAME: $CONFIG_NAME"
  if [ -n "$gpu" ]; then
    echo "  GPU: $gpu"
    env CUDA_VISIBLE_DEVICES="$gpu" DATASET_NAME="$task" DEBUG=true CONFIG_NAME="$CONFIG_NAME" ./train_uwm.sh
  else
    env DATASET_NAME="$task" DEBUG=true CONFIG_NAME="$CONFIG_NAME" ./train_uwm.sh
  fi
}

run_background() {
  local task="$1"
  local gpu="$2"
  local log_file="logs/${task}.log"
  echo "Starting task (background): $task"
  echo "  CONFIG_NAME: $CONFIG_NAME"
  if [ -n "$gpu" ]; then
    echo "  GPU: $gpu"
    nohup env CUDA_VISIBLE_DEVICES="$gpu" DATASET_NAME="$task" DEBUG=false CONFIG_NAME="$CONFIG_NAME" ./train_uwm.sh > "$log_file" 2>&1 &
  else
    nohup env DATASET_NAME="$task" DEBUG=false CONFIG_NAME="$CONFIG_NAME" ./train_uwm.sh > "$log_file" 2>&1 &
  fi
  local pid=$!
  echo "$pid" >> "$PID_FILE"
  printf "%s\t%s\t%s\n" "$pid" "$task" "$log_file" >> "$JOB_FILE"
  echo "  PID: $pid, log: $log_file"
}

gpu_idx=0
if [ "$debug_flag" = "true" ]; then
  # Foreground (serial)
  for task in "${TASKS[@]}"; do
    gpu=""
    if [ "${#AVAILABLE_GPUS[@]}" -gt 0 ]; then
      gpu="${AVAILABLE_GPUS[$gpu_idx]}"
      gpu_idx=$(( (gpu_idx + 1) % ${#AVAILABLE_GPUS[@]} ))
    fi
    run_foreground "$task" "$gpu"
  done
else
  # Background (parallel)
  for task in "${TASKS[@]}"; do
    gpu=""
    if [ "${#AVAILABLE_GPUS[@]}" -gt 0 ]; then
      gpu="${AVAILABLE_GPUS[$gpu_idx]}"
      gpu_idx=$(( (gpu_idx + 1) % ${#AVAILABLE_GPUS[@]} ))
    fi
    run_background "$task" "$gpu"
  done
  echo "PIDs saved to: $PID_FILE"
  echo "Jobs saved to: $JOB_FILE"
fi
