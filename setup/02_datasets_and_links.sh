#!/bin/bash

# Step 2: Setup symlinks + download datasets
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

paths_file="$repo_root/conf/paths/paths.yaml"
if [ -f "$paths_file" ]; then
    data_root="$(awk -F':' '/^data_root:/{gsub(/^[[:space:]]+|[[:space:]]+$/,"",$2); print $2}' "$paths_file")"
fi
if [ -z "${data_root:-}" ]; then
    data_root="/mnt/data_nvme1/minghao.fu"
fi

echo "🔗 Setting up experiment output symlinks..."
mkdir -p "$data_root/logs/experiments/tcwm" \
         "$data_root/logs/experiments/tcwm_wan" \
         "$data_root/logs/experiments/dinowm"
mkdir -p "$repo_root/experiments"
ln -sfn "$data_root/logs/experiments/tcwm" "$repo_root/experiments/tcwm"
ln -sfn "$data_root/logs/experiments/tcwm_wan" "$repo_root/experiments/tcwm_wan"
ln -sfn "$data_root/logs/experiments/dinowm" "$repo_root/experiments/dinowm"

echo "🔗 Setting up dataset symlinks..."
mkdir -p "$data_root/datasets"/{robomimic,pusht_dataset,wall_single,point_maze,mmbench,ogbench,deformable}
mkdir -p "$repo_root/data"
ln -sfn "$data_root/datasets/robomimic" "$repo_root/data/robomimic"
ln -sfn "$data_root/datasets/pusht_dataset" "$repo_root/data/pusht_dataset"
ln -sfn "$data_root/datasets/wall_single" "$repo_root/data/wall_single"
ln -sfn "$data_root/datasets/point_maze" "$repo_root/data/point_maze"
ln -sfn "$data_root/datasets/mmbench" "$repo_root/data/mmbench"
ln -sfn "$data_root/datasets/ogbench" "$repo_root/data/ogbench"
ln -sfn "$data_root/datasets/deformable" "$repo_root/data/deformable"

echo "📥 Downloading robomimic datasets..."
task_arg="${1:-}"
type_arg="${2:-}"
if [ -z "$task_arg" ]; then
    task_arg="lift,square,transport,tool_hang,can"
fi
if [ -z "$type_arg" ]; then
    type_arg="ph"
fi

bash "$script_dir/download_datasets/download_datasets.sh" "$task_arg" "$type_arg"

echo "🔧 Updating robomimic config path..."
config_file="$repo_root/conf/env/robomimic_can.yaml"
if [ -f "$config_file" ]; then
    updated_path="$repo_root/data/robomimic/can/ph_convert_full"
    sed -i "s|data_path:.*|data_path: $updated_path|g" "$config_file"
    echo "✅ Updated dataset path in $config_file to: $updated_path"
else
    echo "⚠️  Configuration file not found: $config_file"
fi

echo "✅ Dataset + symlink setup done."
