#!/usr/bin/env bash
set -Eeuo pipefail

# Resolve repository paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RTDETR_DIR="$REPO_ROOT/rtdetrv4"

# Configuration
INPUT_PATH="${1:?Usage: rtdetrv4_run_inference.sh <input> [device] [checkpoint]}"
DEVICE="${2:-cuda:0}"
CHECKPOINT="${3:-$RTDETR_DIR/checkpoints/rtv4_hgnetv2_s_coco.pth}"
CONFIG="${RTDETR_CONFIG:-configs/rtv4/rtv4_hgnetv2_s_coco.yml}"

# Environment
if [[ -d "$REPO_ROOT/venv" ]]; then
    source "$REPO_ROOT/venv/bin/activate"
elif [[ -d "$RTDETR_DIR/venv" ]]; then
    source "$RTDETR_DIR/venv/bin/activate"
else
    echo "Error: Virtual environment not found."
    exit 1
fi

# Validate inputs
if [[ ! -f "$INPUT_PATH" ]]; then
    echo "Error: Input file does not exist: $INPUT_PATH"
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Execute inference
cd "$RTDETR_DIR"

python tools/inference/torch_inf.py \
    -c "$CONFIG" \
    -r "$CHECKPOINT" \
    --input "$INPUT_PATH" \
    --device "$DEVICE"
