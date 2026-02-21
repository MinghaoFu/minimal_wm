#!/bin/bash

# DINO World Model Training Script
# Supports both single and multi-GPU training with auto GPU selection

# Environment setup
echo "🔧 Setting up environment..."
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate wm310
else
    echo "❌ Conda not found; cannot activate env: wm310"
    exit 1
fi

# Verify environment
echo "🔍 Verifying environment..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Set environment variables
# Optional HF mirror: set USE_HF_MIRROR=1 to enable.
if [ "${USE_HF_MIRROR:-}" = "1" ]; then
    export HF_ENDPOINT=https://hf-mirror.com
else
    unset HF_ENDPOINT
fi
export HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1

# Training configuration
CONFIG_NAME="${CONFIG_NAME:-train_tcwm}" # train_tcwm_wan, train_tcwm, train_dinowm
DEBUG_MODE=${DEBUG:-false}
EPOCHS=${EPOCHS:-100}  
NUM_GPUS=${NUM_GPUS:-1}  
GPU_IDS=${GPU_IDS:-"auto"}  
RESUME=${RESUME:-""}  
PROJECTED_DIM=${PROJECTED_DIM:-}
ENV=${ENV:-""}
DATASET_NAME=${DATASET_NAME:-}
ENV_OVERRIDE=""
ENV_DATASET_NAME=""

echo "🚀 Starting DINO World Model training..."
echo "Config: $CONFIG_NAME"
echo "Debug mode: $DEBUG_MODE"
echo "Epochs: $EPOCHS"
echo "Number of GPUs: $NUM_GPUS"
if [ -n "$DATASET_NAME" ]; then
    echo "Dataset override: $DATASET_NAME"
    if [ -z "$ENV" ]; then
        if [ -f "./conf/env/${DATASET_NAME}.yaml" ]; then
            ENV="$DATASET_NAME"
        else
            # Check if this is an MMBench task name.
            if python - <<'PY'
import json
from pathlib import Path
tasks_fp = Path("/home/minghao.fu/workspace/newt/tasks.json")
if tasks_fp.exists():
    data = json.load(open(tasks_fp))
    import os
    name = os.environ.get("DATASET_NAME", "")
    raise SystemExit(0 if name in data else 1)
raise SystemExit(1)
PY
            then
                ENV="mmbench"
                ENV_DATASET_NAME="$DATASET_NAME"
            else
                ENV_OVERRIDE="robomimic_${DATASET_NAME}_full"
            fi
        fi
    fi
else
    echo "Dataset override: (using config default)"
fi
if [ -n "$RESUME" ]; then
    echo "Resume from: $RESUME"
fi

# Respect pre-set CUDA_VISIBLE_DEVICES (e.g. from run_tcwm_tasks.sh)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_IDS="$CUDA_VISIBLE_DEVICES"
fi

# GPU selection - use simple approach that works with PyTorch
if [ "$GPU_IDS" = "auto" ]; then
    echo "🎯 Auto-selecting best GPUs..."
    if [ "$NUM_GPUS" -eq 1 ]; then
        # Simple selection: find GPU with lowest memory usage  
        BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -1 | cut -d',' -f1)
        export CUDA_VISIBLE_DEVICES=$BEST_GPU
        echo "Selected GPU: $BEST_GPU"
    else
        # For multi-GPU, use first N GPUs with lowest memory
        BEST_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -$NUM_GPUS | cut -d',' -f1 | tr '\n' ',' | sed 's/,$//')
        export CUDA_VISIBLE_DEVICES=$BEST_GPUS
        echo "Selected GPUs: $BEST_GPUS"
    fi
else
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    echo "Using specified GPUs: $GPU_IDS"
fi

# Debug run only (optional)
if [ "$DEBUG_MODE" = "true" ]; then
    echo "🔍 Running debug training (1 epoch)..."
    DEBUG_ARGS="--config-name=$CONFIG_NAME training.epochs=1 debug=true"
    if [ -n "$ENV_OVERRIDE" ]; then
        DEBUG_ARGS="$DEBUG_ARGS env=$ENV_OVERRIDE"
    elif [ -n "$ENV" ]; then
        DEBUG_ARGS="$DEBUG_ARGS env=$ENV"
        if [ -n "$ENV_DATASET_NAME" ]; then
            DEBUG_ARGS="$DEBUG_ARGS env.dataset_name=$ENV_DATASET_NAME"
        fi
    fi
    if [ "$NUM_GPUS" -eq 1 ]; then
        python train_uwm.py $DEBUG_ARGS
    else
        accelerate launch --num_processes=$NUM_GPUS \
            train_uwm.py \
            $DEBUG_ARGS
    fi

    if [ $? -eq 0 ]; then
        echo "✅ Debug run successful."
        exit 0
    else
        echo "❌ Debug run failed, exiting..."
        exit 1
    fi
fi

# Full training
echo "🎯 Starting full training..."

# Set up training command with optional resume path
TRAIN_ARGS="--config-name=$CONFIG_NAME training.epochs=$EPOCHS"
if [ -n "$PROJECTED_DIM" ]; then
    TRAIN_ARGS="$TRAIN_ARGS projected_dim=$PROJECTED_DIM"
fi
if [ -n "$ENV_OVERRIDE" ]; then
    TRAIN_ARGS="$TRAIN_ARGS env=$ENV_OVERRIDE"
elif [ -n "$ENV" ]; then
    TRAIN_ARGS="$TRAIN_ARGS env=$ENV"
    if [ -n "$ENV_DATASET_NAME" ]; then
        TRAIN_ARGS="$TRAIN_ARGS env.dataset_name=$ENV_DATASET_NAME"
    fi
    fi
if [ -n "$RESUME" ]; then
    echo "📂 Will resume from checkpoint: $RESUME"
    TRAIN_ARGS="$TRAIN_ARGS +saved_folder=$RESUME"
    if [ -n "$PROJECTED_DIM" ]; then
        TRAIN_ARGS="$TRAIN_ARGS +projected_dim=$PROJECTED_DIM"
    fi
fi

# Function to copy models and specific config file after training starts (only for full training)
copy_models_and_config() {
    local output_dir=$1
    local config_name=$2
    if [ -n "$output_dir" ] && [ -d "$output_dir" ]; then
        echo "📁 Copying models directory and config file to: $output_dir"
        
        # Copy models directory
        if [ -d "./models" ]; then
            cp -r ./models "$output_dir/"
            echo "✅ Copied ./models to $output_dir/models"
        fi
        
        # Copy the specific config file being used
        if [ -f "./conf/${config_name}.yaml" ]; then
            cp "./conf/${config_name}.yaml" "$output_dir/"
            echo "✅ Copied ./conf/${config_name}.yaml to $output_dir/"
        fi
        
        echo "🎯 Training run is now self-contained with models and config"
    fi
}

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running single GPU training..."
    python train_uwm.py $TRAIN_ARGS 
    TRAIN_PID=$!
else
    echo "Running multi-GPU training with $NUM_GPUS GPUs..."
    # Multi-GPU with accelerate
    accelerate launch \
        --num_processes=$NUM_GPUS \
        --mixed_precision=no \
        train_uwm.py \
        $TRAIN_ARGS \
    TRAIN_PID=$!
fi

# Wait a moment for output directory to be created, then copy models and config
echo "⏳ Waiting for training to initialize and create output directory..."
sleep 3

# Find the most recent output directory created today
OUTPUT_DIR=$(find ./outputs/$(date +%Y-%m-%d) -maxdepth 1 -type d -name "$(date +%H)*" 2>/dev/null | sort | tail -1)
if [ -n "$OUTPUT_DIR" ]; then
    copy_models_and_config "$OUTPUT_DIR" "$CONFIG_NAME"
else
    echo "⚠️ Could not find output directory to copy models and config"
fi

# Wait for training to complete
wait $TRAIN_PID
TRAIN_EXIT_CODE=$?

# Check training status
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Output directory: outputs/$(date +%Y-%m-%d)"
    if [ -n "$OUTPUT_DIR" ]; then
        echo "📂 Models and config saved to: $OUTPUT_DIR"
    fi
else
    echo "❌ Training failed!"
    exit $TRAIN_EXIT_CODE
fi

echo "🎉 Training script completed!"
