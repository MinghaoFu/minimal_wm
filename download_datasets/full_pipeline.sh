#!/bin/bash
# Complete dataset preparation pipeline - adapted for /data2/minghao paths
# Usage: ./full_pipeline.sh [task]
# Example: ./full_pipeline.sh square

set -e

TASK=${1:-can}
DATASET_TYPE="ph"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        Robomimic Dataset Full Pipeline for DINO WM            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Task: $TASK"
echo "Dataset type: $DATASET_TYPE"
echo ""

# Step 1: Download dataset
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1/3: Downloading dataset..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./download_robomimic_datasets.sh $TASK $DATASET_TYPE

# Step 2: Convert to images
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2/3: Converting to image observations (384x384)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./convert_to_images.sh $TASK $DATASET_TYPE 384

# Step 3: Convert to DINO WM format
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3/3: Converting to DINO WM format..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./convert_to_dino_wm.sh $TASK

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ Pipeline Complete!                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Dataset ready at: /data2/minghao/data/robomimic/$TASK/${DATASET_TYPE}_convert_full"
echo ""
echo "Next steps:"
echo "  1. Update your config file to point to the new dataset path"
echo "  2. Run training with: ./train_mimic.sh"
