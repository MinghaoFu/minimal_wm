#!/bin/bash
# Convert robomimic datasets to DINO WM format using task-specific converters.
# Usage: ./convert_to_wm_input.sh [task ...]
# Example: ./convert_to_wm_input.sh square

set -euo pipefail

if [ "$#" -gt 0 ]; then
    TASKS=("$@")
else
    TASKS=(can)
fi

DATASET_TYPE="ph"
DATA_DIR="/workspace/minghao/data/robomimic"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for TASK in "${TASKS[@]}"; do
    SOURCE_DIR="$DATA_DIR/$TASK/$DATASET_TYPE"
    OUTPUT_DIR="$DATA_DIR/$TASK/${DATASET_TYPE}_convert_full"
    SCRIPT_NAME="$SCRIPT_DIR/convert_full_robomimic_${TASK}.py"

    echo "=== Converting to WM format ==="
    echo "Task: $TASK"
    echo "Source: $SOURCE_DIR"
    echo "Output: $OUTPUT_DIR"
    echo ""

    # Check if conversion script exists
    if [ ! -f "$SCRIPT_NAME" ]; then
        echo "❌ Error: Conversion script not found: $SCRIPT_NAME"
        echo "Available scripts:"
        ls -1 "$SCRIPT_DIR"/convert_full_robomimic_*.py 2>/dev/null || echo "  None found"
        exit 1
    fi

    # Run conversion
    python "$SCRIPT_NAME" \
        --source_dir "$SOURCE_DIR" \
        --save_data_dir "$OUTPUT_DIR"

    echo "✅ DINO WM conversion complete!"
    echo "Output directory: $OUTPUT_DIR"
done
