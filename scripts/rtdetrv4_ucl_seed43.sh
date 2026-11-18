#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/work/document-layout-parsing}"
RUN_NAME="rtdetrv4-s-doclaynet-seed43"
SEED=43
REPO_URL="${REPO_URL:-https://github.com/Kriglaja/Document-Layout-Parsing-using-OCR.git}"
REPO_REF="${REPO_REF:-main}"

DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/data/doclaynet}"
ASSET_ROOT="${ASSET_ROOT:-$WORKSPACE_ROOT/assets}"
DINOV3_SOURCE="${DINOV3_SOURCE:-$ASSET_ROOT/dinov3}"
DINOV3_WEIGHTS="${DINOV3_WEIGHTS:-$ASSET_ROOT/dinov3_vitb16_pretrain_lvd1689m.pth}"
STUDENT_WEIGHTS="${STUDENT_WEIGHTS:-$ASSET_ROOT/rtv4_hgnetv2_s_coco.pth}"

RUN_ROOT="$WORKSPACE_ROOT/runs/$RUN_NAME"
REPO_DIR="$RUN_ROOT/repository"
VENV_DIR="$RUN_ROOT/venv"
RESULT_DIR="$RUN_ROOT/results"
TRAIN_OUTPUT="$RUN_ROOT/output"
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

echo "=== RT-DETRv4-S DocLayNet production fine-tuning: seed 43 ==="
date -Is

command -v nvidia-smi >/dev/null
command -v python >/dev/null
command -v git >/dev/null

nvidia-smi | tee "$RESULT_DIR/nvidia-smi.txt"
python --version | tee "$RESULT_DIR/host-python-version.txt"

required_paths=(
  "$DATA_ROOT/PNG"
  "$DATA_ROOT/COCO/train.json"
  "$DATA_ROOT/COCO/val.json"
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
python - <<'PYVERS' | tee "$RESULT_DIR/torch-version.txt"
import torch
import torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print("cuda runtime:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
PYVERS

RTDETR_DIR="$REPO_DIR/rtdetrv4"
cd "$RTDETR_DIR"

ln -sfn "$DINOV3_SOURCE" dinov3
mkdir -p pretrain
ln -sfn "$DINOV3_WEIGHTS" pretrain/dinov3_vitb16_pretrain_lvd1689m.pth
ln -sfn "$STUDENT_WEIGHTS" pretrain/rtv4_hgnetv2_s_coco.pth

python - <<'PYCHECKPOINTS'
import torch
for path in (
    "pretrain/dinov3_vitb16_pretrain_lvd1689m.pth",
    "pretrain/rtv4_hgnetv2_s_coco.pth",
):
    print("Verifying:", path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    print("Loaded:", type(state).__name__)
PYCHECKPOINTS

SOURCE_CONFIG="configs/rtv4/rtv4_hgnetv2_s_doclaynet_1024.yml"
RUNTIME_CONFIG="$RTDETR_DIR/configs/rtv4/rtv4_hgnetv2_s_doclaynet_1024_seed43_runtime.yml"

if [ ! -f "$SOURCE_CONFIG" ]; then
  echo "Missing configuration in repository: $SOURCE_CONFIG"
  exit 4
fi

python - <<PYCONFIG
from pathlib import Path
import yaml

source = Path("$SOURCE_CONFIG")
destination = Path("$RUNTIME_CONFIG")
data_root = Path("$DATA_ROOT")
output_dir = Path("$TRAIN_OUTPUT")

with source.open(encoding="utf-8") as file:
    config = yaml.safe_load(file)

image_root = str(data_root / "PNG")
train_json = str(data_root / "COCO" / "train.json")
val_json = str(data_root / "COCO" / "val.json")

train_loader = config["train_dataloader"]
val_loader = config["val_dataloader"]

train_loader["dataset"]["img_folder"] = image_root
train_loader["dataset"]["ann_file"] = train_json
train_loader["num_workers"] = 16
train_loader["total_batch_size"] = 16
train_loader["drop_last"] = True

val_loader["dataset"]["img_folder"] = image_root
val_loader["dataset"]["ann_file"] = val_json
val_loader["num_workers"] = 16
val_loader["total_batch_size"] = 16
val_loader["drop_last"] = False

config["output_dir"] = str(output_dir)
config["epoches"] = 100
config["checkpoint_freq"] = 10
config["print_freq"] = 100
config["use_amp"] = False
config["eval_spatial_size"] = [1024, 1024]

with destination.open("w", encoding="utf-8") as file:
    yaml.safe_dump(config, file, sort_keys=False)

print("Runtime config:", destination)
print("Training JSON:", train_json)
print("Validation JSON:", val_json)
print("Images:", image_root)
print("Output:", output_dir)
print("Training batch:", train_loader["total_batch_size"])
print("Validation batch:", val_loader["total_batch_size"])
PYCONFIG

cp "$RUNTIME_CONFIG" "$RESULT_DIR/runtime-config.yml"

python - <<'PYGPU'
import torch
count = torch.cuda.device_count()
if count != 1:
    raise SystemExit(f"Expected exactly one CUDA device, found {count}")
name = torch.cuda.get_device_name(0)
memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
print("Selected GPU:", name)
print(f"GPU memory: {memory_gib:.1f} GiB")
if "MIG" in name:
    raise SystemExit("Expected a complete B200, not a MIG slice")
if "B200" not in name:
    raise SystemExit(f"Expected NVIDIA B200, found {name}")
PYGPU

python - <<PYDATA
import json
from pathlib import Path

root = Path("$DATA_ROOT")
image_root = root / "PNG"
expected = {
    "train": (69375, 941123),
    "val": (6489, 99816),
}

for split, expected_counts in expected.items():
    annotation_path = root / "COCO" / f"{split}.json"
    with annotation_path.open(encoding="utf-8") as file:
        coco = json.load(file)
    actual_counts = (len(coco["images"]), len(coco["annotations"]))
    print(f"{split}: {actual_counts}")
    if actual_counts != expected_counts:
        raise SystemExit(
            f"{split} count mismatch: expected {expected_counts}, found {actual_counts}"
        )
    missing = []
    for image in coco["images"]:
        image_path = image_root / image["file_name"]
        if not image_path.is_file():
            missing.append(str(image_path))
            if len(missing) == 10:
                break
    if missing:
        print("Missing image files:")
        print("\n".join(missing))
        raise SystemExit(f"{split} contains missing image files")

print("Dataset preflight passed")
PYDATA

echo "Starting training..."
python train.py \
  -c "$RUNTIME_CONFIG" \
  -t pretrain/rtv4_hgnetv2_s_coco.pth \
  --seed="$SEED"

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
