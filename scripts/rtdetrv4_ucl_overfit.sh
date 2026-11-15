#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/work/document-layout-parsing}"
JOB_NAME="${JOB_NAME:-rtdetrv4-s-overfit}"
REPO_URL="${REPO_URL:-https://github.com/Kriglaja/Document-Layout-Parsing-using-OCR.git}"
REPO_REF="${REPO_REF:-main}"

DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/data/doclaynet}"
ASSET_ROOT="${ASSET_ROOT:-$WORKSPACE_ROOT/assets}"
DINOV3_SOURCE="${DINOV3_SOURCE:-$ASSET_ROOT/dinov3}"
DINOV3_WEIGHTS="${DINOV3_WEIGHTS:-$ASSET_ROOT/dinov3_vitb16_pretrain_lvd1689m.pth}"
STUDENT_WEIGHTS="${STUDENT_WEIGHTS:-$ASSET_ROOT/rtv4_hgnetv2_s_coco.pth}"

RUN_ROOT="$WORKSPACE_ROOT/runs/$JOB_NAME"
REPO_DIR="$RUN_ROOT/repository"
VENV_DIR="$RUN_ROOT/venv"
RESULT_DIR="$RUN_ROOT/results"
TRAIN_OUTPUT="$RESULT_DIR/training-output"
LOG_FILE="$RESULT_DIR/run.log"

mkdir -p "$RUN_ROOT" "$RESULT_DIR" "$TRAIN_OUTPUT"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
  local code=$?
  printf 'FAILED\nExit code: %s\nCompleted: %s\n' \
    "$code" "$(date -Is)" > "$RESULT_DIR/FAILED.txt"
  exit "$code"
}
trap on_error ERR
rm -f "$RESULT_DIR/FAILED.txt" "$RESULT_DIR/SUCCESS.txt"

echo "=== RT-DETRv4-S UCloud overfit reproduction ==="
date -Is

command -v nvidia-smi >/dev/null
command -v python >/dev/null
command -v git >/dev/null

nvidia-smi | tee "$RESULT_DIR/nvidia-smi.txt"
python --version | tee "$RESULT_DIR/host-python-version.txt"

required_paths=(
  "$DATA_ROOT/PNG"
  "$DATA_ROOT/subsets/train_32_all_classes.json"
  "$DINOV3_SOURCE"
  "$DINOV3_WEIGHTS"
  "$STUDENT_WEIGHTS"
)

for path in "${required_paths[@]}"; do
  if [ ! -e "$path" ]; then
    echo "Missing required path: $path"
    exit 3
  fi
done

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
git fetch origin
git checkout "$REPO_REF"
git rev-parse HEAD | tee "$RESULT_DIR/repository-commit.txt"

if [ ! -d "$VENV_DIR" ]; then
  python -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
if [ -f "rtdetrv4/requirements.txt" ]; then
  grep -Ev '^(torch|torchvision)([<=> ].*)?$' \
    rtdetrv4/requirements.txt > "$RUN_ROOT/requirements-without-torch.txt"
  python -m pip install -r "$RUN_ROOT/requirements-without-torch.txt"
fi

python --version | tee "$RESULT_DIR/venv-python-version.txt"
python - <<'PY' | tee "$RESULT_DIR/torch-version.txt"
import torch
import torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print("cuda runtime:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
PY

RTDETR_DIR="$REPO_DIR/rtdetrv4"
cd "$RTDETR_DIR"

ln -sfn "$DINOV3_SOURCE" dinov3
mkdir -p pretrain
ln -sfn "$DINOV3_WEIGHTS" pretrain/dinov3_vitb16_pretrain_lvd1689m.pth
ln -sfn "$STUDENT_WEIGHTS" pretrain/rtv4_hgnetv2_s_coco.pth

python - <<PY
import torch
for path in (
    "pretrain/dinov3_vitb16_pretrain_lvd1689m.pth",
    "pretrain/rtv4_hgnetv2_s_coco.pth",
):
    print("Verifying:", path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    print("Loaded:", type(state).__name__)
PY

SOURCE_CONFIG="configs/rtv4/rtv4_hgnetv2_s_doclaynet_overfit.yml"
RUNTIME_CONFIG="$RUN_ROOT/rtv4_hgnetv2_s_doclaynet_overfit_ucloud.yml"

if [ ! -f "$SOURCE_CONFIG" ]; then
  echo "Missing configuration in repository: $SOURCE_CONFIG"
  exit 4
fi

python - <<PY
from pathlib import Path
import yaml

source = Path("$SOURCE_CONFIG")
destination = Path("$RUNTIME_CONFIG")
data_root = Path("$DATA_ROOT")
output_dir = Path("$TRAIN_OUTPUT")

with source.open(encoding="utf-8") as file:
    config = yaml.safe_load(file)

subset = str(data_root / "subsets" / "train_32_all_classes.json")
image_root = str(data_root / "PNG")

for loader_name in ("train_dataloader", "val_dataloader"):
    dataset = config[loader_name]["dataset"]
    dataset["img_folder"] = image_root
    dataset["ann_file"] = subset

config["output_dir"] = str(output_dir)
config["epoches"] = 100
config["checkpoint_freq"] = 10
config["print_freq"] = 8
config["warmup_iter"] = 32
config["lr_warmup_scheduler"] = {
    "type": "LinearWarmup",
    "warmup_duration": 32,
}

train_transforms = config["train_dataloader"]["dataset"]["transforms"]
train_transforms["policy"] = None
train_transforms["mosaic_prob"] = 0.0
config["train_dataloader"]["collate_fn"]["mixup_prob"] = 0.0
config["train_dataloader"]["collate_fn"]["base_size_repeat"] = None

with destination.open("w", encoding="utf-8") as file:
    yaml.safe_dump(config, file, sort_keys=False)

print("Runtime config:", destination)
print("Dataset:", subset)
print("Images:", image_root)
print("Output:", output_dir)
PY

cp "$RUNTIME_CONFIG" "$RESULT_DIR/runtime-config.yml"

echo "Starting training..."
python train.py \
  -c "$RUNTIME_CONFIG" \
  -t pretrain/rtv4_hgnetv2_s_coco.pth \
  --use-amp \
  --seed=0

BEST_CHECKPOINT="$TRAIN_OUTPUT/best_stg1.pth"
if [ ! -s "$BEST_CHECKPOINT" ]; then
  echo "Training completed but best checkpoint is missing: $BEST_CHECKPOINT"
  exit 5
fi

echo "Reloading best checkpoint for evaluation..."
python train.py \
  -c "$RUNTIME_CONFIG" \
  -r "$BEST_CHECKPOINT" \
  --test-only

find "$TRAIN_OUTPUT" -maxdepth 1 -type f \
  -printf '%f %s bytes\n' | sort | tee "$RESULT_DIR/output-files.txt"

printf 'SUCCESS\nCompleted: %s\nBest checkpoint: %s\n' \
  "$(date -Is)" "$BEST_CHECKPOINT" > "$RESULT_DIR/SUCCESS.txt"

echo "Completed successfully. Results: $RESULT_DIR"
