#!/bin/bash

# DINO World Model Training Script for Tool_Hang Task
# Tool_hang: 7D actions, 43D proprio, 44D object = 87D total states

set -e

# Environment setup
echo "🔧 Setting up environment..."
cd /data2/minghao/wm
export PATH="/usr/local/anaconda3/bin:$PATH"
eval "$(/usr/local/anaconda3/etc/profile.d/conda.sh)"
conda activate wm310

# Verify environment
echo "🔍 Verifying environment..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Set environment variables
export WANDB_BASE_URL=https://api.bandw.top
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1

# Training configuration
CONFIG_NAME="train_robomimic_compress"
DEBUG_MODE=${DEBUG:-false}
EPOCHS=${EPOCHS:-100}
NUM_GPUS=${NUM_GPUS:-1}
GPU_IDS=${GPU_IDS:-"auto"}
RESUME=${RESUME:-""}
PROJECTED_DIM=${PROJECTED_DIM:-64}
ENV="robomimic_tool_hang_full"

echo "🚀 Starting DINO World Model training for Tool_Hang..."
echo "Task: Tool_Hang (7D actions, 43D proprio, 44D object = 87D states)"
echo "Config: $CONFIG_NAME"
echo "Environment: $ENV"
echo "Debug mode: $DEBUG_MODE"
echo "Epochs: $EPOCHS"
echo "Number of GPUs: $NUM_GPUS"
echo "Projected dim: $PROJECTED_DIM"
if [ -n "$RESUME" ]; then
    echo "Resume from: $RESUME"
fi

# GPU selection
if [ "$GPU_IDS" = "auto" ]; then
    echo "🎯 Auto-selecting best GPUs..."
    if [ "$NUM_GPUS" -eq 1 ]; then
        BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -1 | cut -d',' -f1)
        export CUDA_VISIBLE_DEVICES=$BEST_GPU
        echo "Selected GPU: $BEST_GPU"
    else
        BEST_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -$NUM_GPUS | cut -d',' -f1 | tr '\n' ',' | sed 's/,$//')
        export CUDA_VISIBLE_DEVICES=$BEST_GPUS
        echo "Selected GPUs: $BEST_GPUS"
    fi
else
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    echo "Using specified GPUs: $GPU_IDS"
fi

# Debug run first (optional)
if [ "$DEBUG_MODE" = "true" ]; then
    echo "🔍 Running debug training (1 epoch)..."
    if [ "$NUM_GPUS" -eq 1 ]; then
        python train_robomimic_compress.py \
            --config-name=$CONFIG_NAME \
            env=$ENV \
            training.epochs=1 \
            debug=true
    else
        accelerate launch --num_processes=$NUM_GPUS \
            train_robomimic_compress.py \
            --config-name=$CONFIG_NAME \
            env=$ENV \
            training.epochs=1 \
            debug=true
    fi

    if [ $? -eq 0 ]; then
        echo "✅ Debug run successful, proceeding with full training..."
    else
        echo "❌ Debug run failed, exiting..."
        exit 1
    fi
fi

# Full training
echo "🎯 Starting full training..."

# Set up training command
TRAIN_ARGS="--config-name=$CONFIG_NAME env=$ENV training.epochs=$EPOCHS projected_dim=$PROJECTED_DIM"
if [ -n "$RESUME" ]; then
    echo "📂 Will resume from checkpoint: $RESUME"
    TRAIN_ARGS="$TRAIN_ARGS +saved_folder=$RESUME"
fi

# Function to copy models and config
copy_models_and_config() {
    local output_dir=$1
    local config_name=$2
    if [ -n "$output_dir" ] && [ -d "$output_dir" ]; then
        echo "📁 Copying models directory and config file to: $output_dir"

        if [ -d "./models" ]; then
            cp -r ./models "$output_dir/"
            echo "✅ Copied ./models to $output_dir/models"
        fi

        if [ -f "./conf/${config_name}.yaml" ]; then
            cp "./conf/${config_name}.yaml" "$output_dir/"
            echo "✅ Copied ./conf/${config_name}.yaml to $output_dir/"
        fi

        if [ -f "./conf/env/${ENV}.yaml" ]; then
            cp "./conf/env/${ENV}.yaml" "$output_dir/"
            echo "✅ Copied ./conf/env/${ENV}.yaml to $output_dir/"
        fi

        echo "🎯 Training run is now self-contained"
    fi
}

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running single GPU training..."
    python train_robomimic_compress.py $TRAIN_ARGS &
    TRAIN_PID=$!
else
    echo "Running multi-GPU training with $NUM_GPUS GPUs..."
    accelerate launch \
        --num_processes=$NUM_GPUS \
        --mixed_precision=no \
        train_robomimic_compress.py \
        $TRAIN_ARGS \
        training.batch_size=64 &
    TRAIN_PID=$!
fi

# Wait for output directory creation
echo "⏳ Waiting for training to initialize..."
sleep 30

# Find most recent output directory
OUTPUT_DIR=$(find ./outputs_robomimic_tool_hang_full/$(date +%Y-%m-%d) -maxdepth 1 -type d -name "$(date +%H)*" 2>/dev/null | sort | tail -1)
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR=$(find ./outputs/$(date +%Y-%m-%d) -maxdepth 1 -type d -name "$(date +%H)*" 2>/dev/null | sort | tail -1)
fi

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
    echo "📊 Check WandB for results"
    echo "📁 Output directory: $OUTPUT_DIR"
else
    echo "❌ Training failed!"
    exit $TRAIN_EXIT_CODE
fi

echo "🎉 Tool_hang training script completed!"
