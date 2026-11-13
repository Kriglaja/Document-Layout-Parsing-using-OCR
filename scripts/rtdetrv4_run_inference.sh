#!/bin/bash
set -e

# Find the repository root directory (one level up from 'scripts/')
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Move into the rtdetrv4 workspace where configs and tools live
cd "$REPO_ROOT/rtdetrv4"

# Ensure input path argument is provided
if [ -z "$1" ]; then
    echo "Error: No input file provided!"
    echo "Usage: ./scripts/rtdetrv4_run_inference.sh <path_to_image_or_video> [device]"
    exit 1
fi

# Activate the virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
else
    echo "Error: Virtual environment not found!"
    exit 1
fi

# Run inference
INPUT_PATH="$1"
DEVICE="${2:-cuda:0}"

python tools/inference/torch_inf.py \
  -c configs/rtv4/rtv4_hgnetv2_s_coco.yml \
  -r checkpoints/rtv4_hgnetv2_s_coco.pth \
  --input "$INPUT_PATH" \
  --device "$DEVICE"
