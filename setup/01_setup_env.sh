#!/bin/bash

# Step 1: Install/activate conda env + core deps
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

echo "🚀 DINO World Model Environment Setup"
echo "====================================="

# Install miniconda if not present
if ! command -v conda &> /dev/null; then
    echo "📦 Installing miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda"
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

env_name="wm310"

echo "🔧 Configuring conda..."
conda config --set always_yes true
conda config --set changeps1 false

env_exists="$(conda env list | awk '{print $1}' | grep -Fx "$env_name" || true)"
if [ -z "$env_exists" ]; then
    echo "📦 Creating conda environment: $env_name (Python 3.10)"
    conda create -n "$env_name" python=3.10 -y
else
    echo "ℹ️  Conda environment $env_name already exists, skipping creation"
fi

echo "🔧 Activating conda environment..."
conda init
conda activate "$env_name"

echo "📚 Installing core packages..."
pip install --upgrade pip
pip install wandb hydra

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

# Install robosuite / robomimic from source (paths may need update)
robosuite_dir="/workspace/minghao/robosuite"
robomimic_dir="/workspace/minghao/robomimic"

echo "🤖 Installing robosuite from source..."
if [ -d "$robosuite_dir" ]; then
    (cd "$robosuite_dir" && pip install -e .)
    echo "✅ Robosuite installed from source"
else
    echo "⚠️  Robosuite directory not found: $robosuite_dir"
fi

echo "🤖 Installing robomimic from source..."
if [ -d "$robomimic_dir" ]; then
    (cd "$robomimic_dir" && pip install -e .)
    echo "✅ Robomimic installed from source"
else
    echo "⚠️  Robomimic directory not found: $robomimic_dir"
fi

# Env vars
echo "🔧 Setting up environment variables..."
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/minghao/.mujoco/mujoco210/bin:/usr/lib/nvidia' >> ~/.bashrc
echo 'export WANDB_BASE_URL=https://api.bandw.top' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface' >> ~/.bashrc
echo 'export HF_HUB_ENABLE_HF_TRANSFER=1' >> ~/.bashrc

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/minghao/.mujoco/mujoco210/bin:/usr/lib/nvidia
export WANDB_BASE_URL=https://api.bandw.top
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "🔍 Verifying PyTorch CUDA installation..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

echo "✅ Environment setup done."
