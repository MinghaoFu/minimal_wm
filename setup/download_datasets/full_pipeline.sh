#!/bin/bash
# Complete dataset preparation pipeline for WM datasets.
# Usage: ./full_pipeline.sh [task]
# Example: ./full_pipeline.sh square

set -euo pipefail

TASK=${1:-} # tasks=("lift" "square" "transport" "tool_hang" "can")
# go through all tasks
TASKS = ("lift" "square" "transport" "tool_hang" "can")
DATASET_TYPE="ph"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PATHS_FILE="$REPO_ROOT/conf/paths/paths.yaml"
if [ -f "$PATHS_FILE" ]; then
  DATA_ROOT="$(awk -F':' '/^data_root:/{gsub(/^[[:space:]]+|[[:space:]]+$/,"",$2); print $2}' "$PATHS_FILE")"
fi
if [ -z "${DATA_ROOT:-}" ]; then
  DATA_ROOT="/mnt/data_nvme1/minghao.fu"
fi
DATA_DIR="$DATA_ROOT/datasets/robomimic"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        Download and Convert Datasets Pipeline for WM           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Task: $TASK"
echo "Dataset type: $DATASET_TYPE"
echo ""

# Step 1: Download, image conversion, and DINO WM conversion (all-in-one)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Downloading + converting (images & DINO WM)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash "$SCRIPT_DIR/download_datasets.sh" "$TASK" "$DATASET_TYPE"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ Pipeline Complete!                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Dataset ready at: $DATA_DIR/$TASK/${DATASET_TYPE}_convert_full"
echo ""
echo "Next steps:"
echo "  1. Update your config file to point to the new dataset path"
echo "  2. Run training with: ./train_mimic.sh"
