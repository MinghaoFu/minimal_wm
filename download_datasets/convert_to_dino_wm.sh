#!/bin/bash
# Convert robomimic datasets to DINO WM format - adapted for /data2/minghao paths
# Usage: ./convert_to_dino_wm.sh [task]
# Example: ./convert_to_dino_wm.sh square

set -e

# Default values
TASK=${1:-can}
DATASET_TYPE="ph"
DATA_DIR="/data2/minghao/data/robomimic"
SOURCE_DIR="$DATA_DIR/$TASK/$DATASET_TYPE"
OUTPUT_DIR="$DATA_DIR/$TASK/${DATASET_TYPE}_convert_full"

# Activate conda environment
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate wm310

echo "=== Converting to DINO WM format ==="
echo "Task: $TASK"
echo "Source: $SOURCE_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if conversion script exists
SCRIPT_NAME="convert_full_robomimic_${TASK}.py"
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "❌ Error: Conversion script not found: $SCRIPT_NAME"
    echo "Available scripts:"
    ls -1 convert_full_robomimic_*.py 2>/dev/null || echo "  None found"
    exit 1
fi

# Run conversion
python $SCRIPT_NAME \
    --source_dir $SOURCE_DIR \
    --save_data_dir $OUTPUT_DIR

echo "✅ DINO WM conversion complete!"
echo "Output directory: $OUTPUT_DIR"
