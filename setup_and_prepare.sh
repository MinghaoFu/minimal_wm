#!/bin/bash

# DINO World Model Complete Environment Setup Script
# Updated for complete server setup (Sep 10, 2025)
# Includes: Python 3.10, robomimic/robosuite from source, video conversion fixes

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 DINO World Model Complete Server Setup"
echo "=========================================="
echo "This script will:"
echo "✅ Install miniconda if needed"
echo "✅ Create Python 3.10 environment (wm310)"
echo "✅ Install all packages including robosuite/robomimic from source"
echo "✅ Download and convert robomimic datasets"
echo "✅ Fix video conversion issues"
echo "✅ Setup training scripts"
echo ""

# Install miniconda if not present
if ! command -v conda &> /dev/null; then
    echo "📦 Installing miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
fi

# Ensure conda is available
export PATH="$HOME/miniconda/bin:$PATH"
if [ -x "$HOME/miniconda/bin/conda" ]; then
    eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"
fi

# Add conda to .bashrc permanently if not already there
if ! grep -q "export PATH.*miniconda" ~/.bashrc; then
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"' >> ~/.bashrc
fi

# Set the name of the conda environment
env_name="wm310"

# Define the dataset directory and save directory (update paths as needed)
dataset_dir="/workspace/minghao/data/robomimic"
robosuite_dir="/workspace/minghao/robosuite"
robomimic_dir="/workspace/minghao/robomimic"

# Define the dataset types to download (default to PH)
dataset_types=("ph")

# Check for user-specified dataset types
if [ "${1:-}" ]; then
    IFS=',' read -r -a dataset_types <<< "$1"
fi

# Define the tasks to download (default to 'can')
tasks=("can")

# Check for user-specified tasks
if [ "${2:-}" ]; then
    IFS=',' read -r -a tasks <<< "$2"
fi

# Configure conda and accept Terms of Service
echo "🔧 Configuring conda..."

conda config --set always_yes true
conda config --set changeps1 false

# Remove existing environment if it exists
# if conda info --envs | grep -q "^$env_name "; then
#     echo "🗑️  Removing existing environment: $env_name"
#     conda remove -n $env_name --all -y
# fi

env_exists="$(conda env list | awk '{print $1}' | grep -Fx "$env_name" || true)"
if [ -z "$env_exists" ]; then
    echo "📦 Creating conda environment: $env_name (Python 3.10)"
    conda create -n "$env_name" python=3.10 -y
else
    echo "ℹ️  Conda environment $env_name already exists, skipping creation"
fi

# Activate the conda environment
echo "🔧 Activating conda environment..."
conda init
conda activate "$env_name"

echo "📚 Installing core packages..."

# Upgrade pip first
pip install --upgrade pip
pip install wandb
pip install hydra

# Install Rust for tokenizers compilation
echo "🦀 Installing Rust for tokenizers..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
export PATH="$HOME/.cargo/bin:$PATH"
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc

# Install PyTorch with CUDA support - auto-detect CUDA version
echo "🔍 Detecting CUDA version..."
if python - <<'PY' 2>/dev/null; then
import torch
print(f"ℹ️  PyTorch already installed ({torch.__version__})")
PY
then
    echo "ℹ️  PyTorch detected, skipping install"
else
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1-2)
        echo "Detected CUDA Version: $CUDA_VERSION"
        
        # Map CUDA version to PyTorch index
        if [[ "$CUDA_VERSION" == "12.4" ]] || [[ "$CUDA_VERSION" == "12."* ]]; then
            echo "Installing PyTorch for CUDA 12.x..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        elif [[ "$CUDA_VERSION" == "11."* ]]; then
            echo "Installing PyTorch for CUDA 11.x..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        else
            echo "Installing PyTorch with default CUDA support..."
            pip install torch torchvision
        fi
    else
        echo "No CUDA detected, installing CPU-only PyTorch..."
        pip install torch torchvision
    fi
fi

# Install remaining Python requirements only if missing or mismatched
read -r -d '' REQUIRED_PKGS <<'REQS'
transformers==4.28.0
huggingface_hub==0.23.4
scipy
numpy
Pillow
opencv-python
termcolor
tqdm
diffusers==0.11.1
egl_probe>=1.0.1
h5py
imageio
imageio-ffmpeg
matplotlib
psutil
tensorboard
tensorboardX
accelerate
hydra-core
wandb
einops
decord
hf_transfer
nvitop
pynvml
d4rl
mujoco_py==2.1.2.14
Cython
REQS

mapfile -t missing_pkgs < <(python - <<PY
import pkg_resources
reqs = [r.strip() for r in """${REQUIRED_PKGS}""".splitlines() if r.strip()]
missing = []
for req in reqs:
    try:
        pkg_resources.require([req])
    except pkg_resources.DistributionNotFound:
        missing.append(req)
    except pkg_resources.VersionConflict:
        missing.append(req)
print("\n".join(missing))
PY
)

if ((${#missing_pkgs[@]})); then
    echo "📦 Installing missing Python packages: ${missing_pkgs[*]}"
    pip install "${missing_pkgs[@]}" || {
        echo "⚠️  Failed to install some packages, continuing"
    }
else
    echo "ℹ️  Required Python packages already installed"
fi

# Install robosuite from source (CRITICAL: -e for editable install)
echo "🤖 Installing robosuite from source..."
if [ -d "$robosuite_dir" ]; then
    cd $robosuite_dir
    pip install -e .
    cd -
    echo "✅ Robosuite installed from source"
else
    echo "⚠️  Robosuite directory not found: $robosuite_dir"
    echo "Please clone robosuite repository or update the path"
fi

# Install robomimic from source (CRITICAL: -e for editable install)
echo "🤖 Installing robomimic from source..."
if [ -d "$robomimic_dir" ]; then
    cd $robomimic_dir
    pip install -e .
    cd -
    echo "✅ Robomimic installed from source"
else
    echo "⚠️  Robomimic directory not found: $robomimic_dir"
    echo "Please clone robomimic repository or update the path"
fi

# Setup environment variables
echo "🔧 Setting up environment variables..."
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/minghao/.mujoco/mujoco210/bin:/usr/lib/nvidia' >> ~/.bashrc
echo 'export WANDB_BASE_URL=https://api.bandw.top' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface' >> ~/.bashrc
echo 'export HF_HUB_ENABLE_HF_TRANSFER=1' >> ~/.bashrc

# Load environment variables for current session
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/minghao/.mujoco/mujoco210/bin:/usr/lib/nvidia
export WANDB_BASE_URL=https://api.bandw.top
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1

# Verify PyTorch CUDA installation
echo "🔍 Verifying PyTorch CUDA installation..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Verify enhanced GPU selection
echo "🎯 Testing enhanced GPU selection..."
if [ -f "$script_dir/select_best_gpus.py" ]; then
    echo "Enhanced GPU selection script found ✅"
    python "$script_dir/select_best_gpus.py" single --quiet > /dev/null && echo "GPU selection working ✅"
else
    echo "⚠️  Enhanced GPU selection script not found - please ensure select_best_gpus.py is executable"
fi

# Dataset download and conversion section
echo ""
echo "📥 Dataset Download & Conversion"
echo "================================="

# Create dataset directory if it doesn't exist
mkdir -p $dataset_dir

# Download specified tasks and dataset types
task_arg=$(IFS=','; echo "${tasks[*]}")
type_arg=$(IFS=','; echo "${dataset_types[*]}")
bash "$script_dir/download_datasets/download_datasets.sh" "$task_arg" "$type_arg"

# Update configuration files (FIXED: correct dataset paths)
echo ""
echo "🔧 Updating configuration files..."

# Update dataset path in configuration
config_file="conf/env/robomimic_can.yaml"
if [ -f "$config_file" ]; then
    # Update the data path to point to our converted dataset
    updated_path="$dataset_dir/can/ph_convert_full"
    sed -i "s|data_path:.*|data_path: $updated_path|g" "$config_file"
    echo "✅ Updated dataset path in $config_file to: $updated_path"
else
    echo "⚠️  Configuration file not found: $config_file"
fi

# Make training script executable
if [ -f "train.sh" ]; then
    chmod +x train.sh
    echo "✅ Made train.sh executable"
fi

# Final verification and setup summary
echo ""
echo "🎉 DINO World Model Setup Complete!"
echo "===================================="
echo "🎯 Environment: $env_name (Python 3.10)"
echo "📊 GPU Support: $(python -c "import torch; print('✅ CUDA' if torch.cuda.is_available() else '❌ CPU Only')" 2>/dev/null || echo '❓ Please activate conda environment first')"
echo "🗂️  Dataset Directory: $dataset_dir"
echo "🤖 Robosuite: $([ -d "$robosuite_dir" ] && echo '✅ Installed from source' || echo '⚠️ Not found')"
echo "🤖 Robomimic: $([ -d "$robomimic_dir" ] && echo '✅ Installed from source' || echo '⚠️ Not found')"
echo ""
echo "🚀 Quick Start Commands:"
echo "# Activate environment:"
echo "conda activate $env_name"
echo ""
echo "# Debug training (1 epoch):"
echo "DEBUG=true ./train.sh"
echo ""
echo "# Full training (single GPU):"
echo "./train.sh"
echo ""
echo "# Multi-GPU training (2 GPUs, 50 epochs):"
echo "NUM_GPUS=2 EPOCHS=50 ./train.sh"
echo ""
echo "# Planning evaluation:"
echo "./plan.sh"
echo ""
echo "📖 See CLAUDE.md for detailed instructions and troubleshooting"
echo "🐛 If you encounter issues, check that robosuite/robomimic directories exist"
