#!/bin/bash

dataset_dir="/mnt/data_nvme1/minghao.fu/"
robomimic_dir="/home/minghao.fu/workspace/robomimic"
# Set paths
robomimic_dataset_dir="${dataset_dir}/robomimic"

set -e  # Exit on any error
# download dino wm datasets
pip install osfclient

mkdir -p $dataset_dir
mkdir -p $robomimic_dataset_dir

cd $dataset_dir
export OSF_VIEWONLY=a56a296ce3b24cceaf408383a175ce28s
osf -p bmw48 clone .

# Check if robomimic is installed
if [ ! -d "$robomimic_dir" ]; then
    echo "❌ Robomimic directory not found: $robomimic_dir"
    echo "Please ensure robomimic is installed from source"
    exit 1
fi

# Define tasks to download (you can modify this list)
tasks=("lift" "square" "transport" "tool_hang" "can")

# Define dataset types (default to PH - Paired Human demos)
dataset_types=("ph")

# Check for user-specified tasks
if [ "$1" ]; then
    IFS=',' read -r -a tasks <<< "$1"
fi

# Check for user-specified dataset types
if [ "$2" ]; then
    IFS=',' read -r -a dataset_types <<< "$2"
fi

echo "📥 Tasks to download: ${tasks[*]}"
echo "📊 Dataset types: ${dataset_types[*]}"
echo ""

# Create dataset directory if it doesn't exist
mkdir -p $robomimic_dataset_dir

# Download and convert each task
for task in "${tasks[@]}"; do
    for dataset_type in "${dataset_types[@]}"; do
        echo "🔄 Processing task: $task, dataset type: $dataset_type"
        echo "=================================================="

        # Step 1: Download raw dataset
        echo "📥 Downloading raw dataset..."
        python $robomimic_dir/robomimic/scripts/download_datasets.py \
            --tasks $task \
            --dataset_types $dataset_type \
            --hdf5_types all \
            --download_dir $robomimic_dataset_dir

        # Check if download was successful
        task_dir="$robomimic_dataset_dir/$task/$dataset_type"
        if [ ! -f "$task_dir/demo_v15.hdf5" ]; then
            echo "❌ Download failed for $task ($dataset_type)"
            continue
        fi

        echo "✅ Downloaded: $task_dir/demo_v15.hdf5"

        # Step 2: Convert states to images
        echo "🖼️  Converting states to images..."
        python $robomimic_dir/robomimic/scripts/dataset_states_to_obs.py \
            --dataset $task_dir/demo_v15.hdf5 \
            --output_name $task_dir/image_384_v15.hdf5 \
            --done_mode 2 \
            --camera_names agentview robot0_eye_in_hand \
            --camera_height 384 \
            --camera_width 384

        if [ ! -f "$task_dir/image_384_v15.hdf5" ]; then
            echo "❌ Image conversion failed for $task ($dataset_type)"
            continue
        fi

        echo "✅ Image conversion completed: $task_dir/image_384_v15.hdf5"

        # Step 3: Convert to DINO WM format (if conversion script exists)
        output_path="$robomimic_dataset_dir/$task/${dataset_type}_convert"

        if [ -f "./convert_to_wm_input.py" ]; then
            echo "🔄 Converting to DINO WM format..."

            python convert_to_wm_input.py \
                --source_dir "$task_dir" \
                --save_data_dir "$output_path"

            if [ -f "$output_path/states.pth" ] && [ -d "$output_path/obses" ]; then
                video_count=$(ls "$output_path/obses/"*.mp4 2>/dev/null | wc -l)
                echo "✅ DINO WM conversion completed: $video_count video files"
            else
                echo "⚠️  DINO WM conversion may have failed"
            fi
        else
            echo "⚠️  DINO WM conversion script not found, skipping..."
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
            if [ -d "$robomimic_dataset_dir/$task/${dataset_type}_convert" ]; then
                echo "   └── DINO WM format: $robomimic_dataset_dir/$task/${dataset_type}_convert"
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
    echo "   - $task: $robomimic_dataset_dir/$task/ph_convert"
done
