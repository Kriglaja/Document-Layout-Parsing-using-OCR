#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/work/document-layout-parsing}"
REPO_URL="${REPO_URL:-https://github.com/Kriglaja/Document-Layout-Parsing-using-OCR.git}"

TRAIN_RUN="$WORKSPACE_ROOT/runs/rtdetrv4-s-doclaynet-seed42"
EXPORT_RUN="$WORKSPACE_ROOT/runs/rtdetrv4-s-doclaynet-seed42-export"

REPO_DIR="$EXPORT_RUN/repository"
VENV_DIR="$TRAIN_RUN/venv"
RESULT_DIR="$EXPORT_RUN/results/val-full"
LOG_FILE="$RESULT_DIR/run.log"

CHECKPOINT="$TRAIN_RUN/output/best_stg1.pth"
SAVED_CONFIG="$TRAIN_RUN/results/runtime-config.yml"

DATA_ROOT="$WORKSPACE_ROOT/data/doclaynet"
ASSET_ROOT="$WORKSPACE_ROOT/assets"

DINOV3_SOURCE="$ASSET_ROOT/dinov3"
DINOV3_WEIGHTS="$ASSET_ROOT/dinov3_vitb16_pretrain_lvd1689m.pth"
STUDENT_WEIGHTS="$ASSET_ROOT/rtv4_hgnetv2_s_coco.pth"

mkdir -p "$RESULT_DIR"

# Start this attempt with an independent log.
if [[ -f "$LOG_FILE" ]]; then
    mv \
        "$LOG_FILE" \
        "$RESULT_DIR/run-previous-$(date +%Y%m%d%H%M%S).log"
fi

exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
    local code=$?

    printf \
        'FAILED\nExit code: %s\nCompleted: %s\n' \
        "$code" \
        "$(date -Is)" \
        > "$RESULT_DIR/FAILED.txt"

    exit "$code"
}

trap on_error ERR

rm -f \
    "$RESULT_DIR/FAILED.txt" \
    "$RESULT_DIR/SUCCESS.txt"

echo "=== RT-DETRv4-S full validation prediction export ==="
date -Is

required_paths=(
    "$CHECKPOINT"
    "$SAVED_CONFIG"
    "$DATA_ROOT/PNG"
    "$DATA_ROOT/COCO/val.json"
    "$DINOV3_SOURCE"
    "$DINOV3_WEIGHTS"
    "$STUDENT_WEIGHTS"
)

for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing required path: $path"
        exit 3
    fi
done

echo "=== Persistent storage ==="
df -h "$WORKSPACE_ROOT"

echo "=== GPU information ==="
nvidia-smi | tee "$RESULT_DIR/nvidia-smi.txt"

# During infrastructure development, always use a fresh clone of current main.
# This avoids stale persistent branches and old pinned commits.
rm -rf "$REPO_DIR"

git clone \
    --branch main \
    --single-branch \
    "$REPO_URL" \
    "$REPO_DIR"

CURRENT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"

printf '%s\n' "$CURRENT_COMMIT" \
    | tee "$RESULT_DIR/repository-commit.txt"

echo "Repository commit: $CURRENT_COMMIT"

# Ensure this job cannot accidentally run the pre-fix exporter.
if ! grep -q 'solver_output_dir' \
    "$REPO_DIR/rtdetrv4/export_predictions.py"; then

    echo "ERROR: export_predictions.py does not contain solver_output_dir fix"
    exit 4
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Missing seed-42 training environment: $VENV_DIR"
    exit 5
fi

source "$VENV_DIR/bin/activate"

echo "=== Checking required Python packages ==="

python - <<'PYDEPS'
import faster_coco_eval
import torch
import torchvision

print("Required Python dependencies are available")
PYDEPS

python - <<'PYENV' \
    | tee "$RESULT_DIR/environment.txt"

import sys

import faster_coco_eval
import torch
import torchvision

print("python:", sys.version.replace("\n", " "))
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print(
    "faster_coco_eval:",
    getattr(faster_coco_eval, "__version__", "unknown"),
)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")

print("gpu_count:", torch.cuda.device_count())
print("gpu:", torch.cuda.get_device_name(0))
print(
    "gpu_memory_gib:",
    torch.cuda.get_device_properties(0).total_memory / 1024**3,
)
PYENV

RTDETR_DIR="$REPO_DIR/rtdetrv4"

cd "$RTDETR_DIR"

ln -sfn "$DINOV3_SOURCE" dinov3

mkdir -p pretrain

ln -sfn \
    "$DINOV3_WEIGHTS" \
    pretrain/dinov3_vitb16_pretrain_lvd1689m.pth

ln -sfn \
    "$STUDENT_WEIGHTS" \
    pretrain/rtv4_hgnetv2_s_coco.pth

# The saved runtime configuration has a relative __include__. Copying the
# configuration beside the included YAML preserves the original resolution,
# validation loader, categories, and model definition.
RUNTIME_CONFIG="$RTDETR_DIR/configs/rtv4/rtv4_hgnetv2_s_doclaynet_1024_seed42_export_runtime.yml"

cp \
    "$SAVED_CONFIG" \
    "$RUNTIME_CONFIG"

cp \
    "$RUNTIME_CONFIG" \
    "$RESULT_DIR/runtime-config.yml"

RAW="$RESULT_DIR/val-raw.json"
META="$RESULT_DIR/val-export-metadata.json"
SANITIZED="$RESULT_DIR/val-sanitized.json"

echo "=== Exporting full validation predictions ==="

python export_predictions.py \
    --config "$RUNTIME_CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output "$RAW" \
    --metadata-out "$META" \
    --confidence 0.001 \
    --max-detections 300 \
    --device cuda

cd "$REPO_DIR"

echo "=== Sanitizing predictions ==="

python eval/sanitize_predictions.py \
    --gt "$DATA_ROOT/COCO/val.json" \
    --pred "$RAW" \
    --out "$SANITIZED" \
    --report-out "$RESULT_DIR/val-sanitation.json"

echo "=== Validating sanitized predictions ==="

python eval/validate_predictions.py \
    --gt "$DATA_ROOT/COCO/val.json" \
    --pred "$SANITIZED" \
    --summary-out "$RESULT_DIR/val-sanitized-validation.json"

echo "=== Evaluating sanitized predictions ==="

python eval/coco_eval.py \
    --gt "$DATA_ROOT/COCO/val.json" \
    --pred "$SANITIZED" \
    --max-dets 1 10 100 \
    --out "$RESULT_DIR/val-sanitized-metrics.json"

echo "=== Recording hashes and file sizes ==="

sha256sum \
    "$RESULT_DIR"/*.json \
    > "$RESULT_DIR/json-sha256.txt"

find "$RESULT_DIR" \
    -maxdepth 1 \
    -type f \
    -printf '%f %s bytes\n' \
    | sort \
    > "$RESULT_DIR/output-files.txt"

printf \
    'SUCCESS\nCompleted: %s\nRepository commit: %s\nResults: %s\n' \
    "$(date -Is)" \
    "$CURRENT_COMMIT" \
    "$RESULT_DIR" \
    > "$RESULT_DIR/SUCCESS.txt"

echo
echo "=== Full validation export completed successfully ==="
echo "Results: $RESULT_DIR"
echo
echo "Sanitation report:"
cat "$RESULT_DIR/val-sanitation.json"
echo
echo "Sanitized metrics:"
cat "$RESULT_DIR/val-sanitized-metrics.json"
