#!/bin/bash
set -euo pipefail

EPOCHS=${EPOCHS:-50}
CONFIG_NAME=${CONFIG_NAME:-train_tcwm}
NUM_GPUS=${NUM_GPUS:-4}
GPU_IDS=${GPU_IDS:-"auto"}
ROLLOUT_NUM=${ROLLOUT_NUM:-4}
ROLLOUT_PARALLEL=${ROLLOUT_PARALLEL:-1}
DEBUG_ROLLOUT=${DEBUG_ROLLOUT:-true}

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate wm310
else
    echo "❌ Conda not found; cannot activate env: wm310"
    exit 1
fi

if [ "${USE_HF_MIRROR:-}" = "1" ]; then
    export HF_ENDPOINT=https://hf-mirror.com
else
    unset HF_ENDPOINT
fi
export HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1

# Pick GPUs with lowest memory usage if auto
if [ "$GPU_IDS" = "auto" ]; then
    if [ "$NUM_GPUS" -eq 1 ]; then
        BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -1 | cut -d',' -f1)
        GPU_IDS=$BEST_GPU
    else
        GPU_IDS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -$NUM_GPUS | cut -d',' -f1 | tr '\n' ',' | sed 's/,$//')
    fi
fi

IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
if [ ${#GPU_LIST[@]} -lt 4 ]; then
  echo "Need 4 GPUs for parallel ablation; got: $GPU_IDS"
  exit 1
fi

run_one () {
  local env_name=$1
  local align=$2
  local gpu=$3
  local desc="align_${align}_${env_name}"
  local log="/home/minghao.fu/workspace/minimal_wm/experiments/ablation_alignment/logs/ablation_alignment_${desc}.out"
  echo "Launching ${desc} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES=${gpu} \
  python /home/minghao.fu/workspace/minimal_wm/train_tcwm.py \
    --config-name=${CONFIG_NAME} \
    training.epochs=${EPOCHS} \
    env=${env_name} \
    alignment.open_alignment=${align} \
    description=${desc} \
    rollout_num=${ROLLOUT_NUM} \
    rollout_parallel=${ROLLOUT_PARALLEL} \
    +rollout_debug=${DEBUG_ROLLOUT} \
    > ${log} 2>&1 &
}

run_one lift true  ${GPU_LIST[0]}
run_one lift false ${GPU_LIST[1]}
run_one pusht true ${GPU_LIST[2]}
run_one pusht false ${GPU_LIST[3]}
