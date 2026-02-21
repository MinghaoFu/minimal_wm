#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
paths_file="$repo_root/conf/paths/paths.yaml"
if [ -f "$paths_file" ]; then
    data_root="$(awk -F':' '/^data_root:/{gsub(/^[[:space:]]+|[[:space:]]+$/,"",$2); print $2}' "$paths_file")"
fi
if [ -z "${data_root:-}" ]; then
    data_root="/mnt/data_nvme1/minghao.fu"
fi
dataset_root="$data_root/datasets"
robomimic_dir="/workspace/minghao/robomimic"
robosuite_dir="/workspace/minghao/robosuite"
robomimic_dataset_dir="${dataset_root}/robomimic"
export PYTHONPATH="$robomimic_dir:$robosuite_dir:${PYTHONPATH:-}"
# Allow overriding extra library search paths for mujoco / GL (set EXTRA_LD_PATHS if needed)
default_extra_ld="/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/mesa"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${EXTRA_LD_PATHS:-$default_extra_ld}"
# Avoid h5py locking errors when files are on shared FS
export HDF5_USE_FILE_LOCKING=FALSE

# Ensure osfclient is available for pulling auxiliary assets (checkpoints, etc.)
python -m pip install --quiet osfclient

mkdir -p "$dataset_root" "$robomimic_dataset_dir"

# Pull OSF storage (idempotent)
if [ -z "${OSF_VIEWONLY:-}" ]; then
    export OSF_VIEWONLY="a56a296ce3b24cceaf408383a175ce28"
fi

# download dinowm datasets (wall, point_maze, pusht, deformable)
if [ ! -d "${dataset_root}/osfstorage" ]; then
    echo "📥 Cloning OSF storage (project bmw48) into ${dataset_root} ..."
    (cd "$dataset_root" && osf -p bmw48 clone .)
else
    echo "ℹ️ OSF storage already present at ${dataset_root}/osfstorage, skipping clone"
fi

# Check if robomimic is installed
if [ ! -d "$robomimic_dir" ]; then
    echo "❌ Robomimic directory not found: $robomimic_dir"
    echo "Please ensure robomimic is installed from source"
    exit 1
fi

# Define tasks to download (default set)
tasks=("lift" "square" "transport" "tool_hang" "can")

# Define dataset types (default to PH - Paired Human demos)
dataset_types=("ph")

# Check for user-specified tasks
if [ -n "${1:-}" ]; then
    IFS=',' read -r -a tasks <<< "$1"
fi

# Check for user-specified dataset types
if [ -n "${2:-}" ]; then
    IFS=',' read -r -a dataset_types <<< "$2"
fi

echo "📥 Tasks to download: ${tasks[*]}"
echo "📊 Dataset types: ${dataset_types[*]}"
echo ""

# Download and convert each task
for task in "${tasks[@]}"; do
    for dataset_type in "${dataset_types[@]}"; do
        echo "🔄 Processing task: $task, dataset type: $dataset_type"
        echo "=================================================="

        # Step 1: Download raw dataset (skip if already present)
        task_dir="$robomimic_dataset_dir/$task/$dataset_type"
        mkdir -p "$task_dir"

        if [ ! -f "$task_dir/demo_v15.hdf5" ] || [ ! -f "$task_dir/low_dim_v15.hdf5" ]; then
            echo "📥 Downloading raw dataset..."
            python "$robomimic_dir/robomimic/scripts/download_datasets.py" \
                --tasks "$task" \
                --dataset_types "$dataset_type" \
                --hdf5_types all \
                --download_dir "$robomimic_dataset_dir"
        else
            echo "ℹ️ Found existing dataset for $task ($dataset_type), skipping download"
        fi

        # Check if download was successful
        if [ ! -f "$task_dir/demo_v15.hdf5" ] || [ ! -f "$task_dir/low_dim_v15.hdf5" ]; then
            echo "❌ Download failed for $task ($dataset_type)"
            continue
        fi

        echo "✅ Dataset present: $task_dir/demo_v15.hdf5"

        # Step 2: Convert states to images (idempotent)
        image_file="$task_dir/image_384_v15.hdf5"
        if [ ! -f "$image_file" ]; then
            echo "🖼️  Converting states to images..."
            python "$robomimic_dir/robomimic/scripts/dataset_states_to_obs.py" \
                --dataset "$task_dir/demo_v15.hdf5" \
                --output_name "$image_file" \
                --done_mode 2 \
                --camera_names agentview robot0_eye_in_hand \
                --camera_height 384 \
                --camera_width 384
        else
            echo "ℹ️ Image file already exists, skipping conversion"
        fi

        if [ ! -f "$image_file" ]; then
            echo "❌ Image conversion failed for $task ($dataset_type)"
            continue
        fi

        echo "✅ Image conversion completed: $image_file"

        # Step 3: Convert to DINO WM format (use task-specific converter if available)
        conversion_script="${script_dir}/convert_full_robomimic_${task}.py"
        output_path="$robomimic_dataset_dir/$task/${dataset_type}_convert_full"

        if [ -f "$conversion_script" ]; then
            if [ -d "$output_path" ] && [ -f "$output_path/states.pth" ]; then
                echo "ℹ️ DINO WM output already exists at $output_path, skipping conversion"
            else
                echo "🔄 Converting to DINO WM format with $(basename "$conversion_script")..."
                python "$conversion_script" \
                    --source_dir "$task_dir" \
                    --save_data_dir "$output_path"
            fi

            if [ -f "$output_path/states.pth" ] && [ -d "$output_path/obses" ]; then
                video_count=$(find "$output_path/obses" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l)
                echo "✅ DINO WM conversion completed: $video_count video files"
            else
                echo "⚠️  DINO WM conversion may have failed for $task ($dataset_type)"
            fi
        else
            echo "⚠️  No conversion script found for $task at $conversion_script, skipping..."
        fi

        echo "✅ Task $task ($dataset_type) processing completed!"
        echo ""
    done
done

echo "🎉 All tasks downloaded and processed!"
echo ""
echo "📊 Available datasets:"
echo "======================"
for task in "${tasks[@]}"; do
    for dataset_type in "${dataset_types[@]}"; do
        task_dir="$robomimic_dataset_dir/$task/$dataset_type"
        if [ -d "$task_dir" ]; then
            echo "✅ $task ($dataset_type): $task_dir"
            if [ -d "$robomimic_dataset_dir/$task/${dataset_type}_convert_full" ]; then
                echo "   └── DINO WM format: $robomimic_dataset_dir/$task/${dataset_type}_convert_full"
            fi
        fi
    done
done

echo ""
echo "🔧 To use a different task, update your environment config:"
echo "   Edit conf/env/robomimic_can.yaml"
echo "   Change data_path to point to your desired task directory"
echo ""
echo "📋 Example paths:"
for task in "${tasks[@]}"; do
    echo "   - $task: $robomimic_dataset_dir/$task/ph_convert_full"
done
