#!/bin/bash
# Convert robomimic datasets to image observations - adapted for /data2/minghao paths
# Usage: ./convert_to_images.sh [task] [dataset_type] [resolution]
# Example: ./convert_to_images.sh square ph 384

set -e

# Default values
TASK=${1:-can}
DATASET_TYPE=${2:-ph}
RESOLUTION=${3:-64}
ROBOMIMIC_DIR="/data2/minghao/robomimic"
DATA_DIR="/data2/minghao/data/robomimic"

# Activate conda environment
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate wm310

echo "=== Converting robomimic dataset to images ==="
echo "Task: $TASK"
echo "Dataset type: $DATASET_TYPE"
echo "Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "Data directory: $DATA_DIR/$TASK/$DATASET_TYPE"
echo ""

# Convert states to observations
python $ROBOMIMIC_DIR/robomimic/scripts/dataset_states_to_obs.py \
    --dataset $DATA_DIR/$TASK/$DATASET_TYPE/demo_v15.hdf5 \
    --output_name $DATA_DIR/$TASK/$DATASET_TYPE/image_${RESOLUTION}_v15.hdf5 \
    --done_mode 2 \
    --camera_names agentview robot0_eye_in_hand \
    --camera_height $RESOLUTION \
    --camera_width $RESOLUTION

echo "✅ Image conversion complete!"
echo "Output: $DATA_DIR/$TASK/$DATASET_TYPE/image_${RESOLUTION}_v15.hdf5"
