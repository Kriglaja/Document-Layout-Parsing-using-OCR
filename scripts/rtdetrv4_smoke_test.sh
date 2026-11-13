#!/usr/bin/env bash
set -Eeuo pipefail

# The selected UCloud folder is mounted as /work/document-layout-parsing.
PERSISTENT_ROOT="/work/document-layout-parsing"
RUN_DIR="$PERSISTENT_ROOT/rtdetrv4-smoke-test"
REPO_DIR="$RUN_DIR/RT-DETRv4"
VENV_DIR="$RUN_DIR/venv"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
RESULT_DIR="$RUN_DIR/results"
LOG_FILE="$RESULT_DIR/run.log"

mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$RESULT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
  code=$?
  printf 'FAILED\nExit code: %s\nSee run.log for details.\n' "$code" > "$RESULT_DIR/FAILED.txt"
  exit "$code"
}
trap on_error ERR
rm -f "$RESULT_DIR/FAILED.txt" "$RESULT_DIR/SUCCESS.txt"

echo "=== RT-DETRv4-S UCloud smoke test ==="
date -Is

INPUT_IMAGE="$PERSISTENT_ROOT/smoke-test/test.jpeg"
if [ ! -f "$INPUT_IMAGE" ]; then
  INPUT_IMAGE="$(find "$PERSISTENT_ROOT" -type f \( -iname 'test.jpeg' -o -iname 'test.jpg' -o -iname 'test.png' \) -print -quit)"
fi
if [ -z "${INPUT_IMAGE:-}" ] || [ ! -f "$INPUT_IMAGE" ]; then
  echo "Could not find test.jpeg, test.jpg, or test.png under $PERSISTENT_ROOT"
  exit 2
fi
echo "Input image: $INPUT_IMAGE"

nvidia-smi | tee "$RESULT_DIR/nvidia-smi.txt"
python --version | tee "$RESULT_DIR/python-version.txt"

if [ ! -d "$VENV_DIR" ]; then
  python -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone https://github.com/AIMIA-EAI/RT-DETRv4.git "$REPO_DIR"
fi
cd "$REPO_DIR"
git rev-parse HEAD | tee "$RESULT_DIR/upstream-commit.txt"

grep -Ev '^(torch|torchvision)([<=> ].*)?$' requirements.txt > "$RUN_DIR/requirements-without-torch.txt"
python -m pip install -r "$RUN_DIR/requirements-without-torch.txt"
python -m pip install -r tools/inference/requirements.txt
python -m pip install gdown opencv-python-headless

python - <<'PY' | tee "$RESULT_DIR/environment-check.txt"
import cv2
import torch
print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA:", torch.version.cuda)
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access CUDA")
print("GPU:", torch.cuda.get_device_name(0))
PY

CHECKPOINT="$CHECKPOINT_DIR/rtv4_hgnetv2_s_coco.pth"
if [ ! -s "$CHECKPOINT" ]; then
  gdown 'https://drive.google.com/uc?id=1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z' -O "$CHECKPOINT"
fi

# Verify checkpoint integrity (ensuring gdown fetched the actual .pth file and not an HTML warning page)
python - <<EOF
import torch
checkpoint_path = "$CHECKPOINT"
try:
    print(f"Verifying checkpoint integrity: {checkpoint_path}")
    # Load safely onto CPU to test parsing
    torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print("Checkpoint verification passed successfully.")
except Exception as e:
    print(f"ERROR: Checkpoint file is corrupted or not a valid PyTorch model file: {e}")
    import sys
    sys.exit(1)
EOF

ls -lh "$CHECKPOINT" | tee "$RESULT_DIR/checkpoint.txt"

python tools/inference/torch_inf.py \
  -c configs/rtv4/rtv4_hgnetv2_s_coco.yml \
  -r "$CHECKPOINT" \
  --input "$INPUT_IMAGE" \
  --device cuda:0

find "$REPO_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
  -mmin -15 -print | tee "$RESULT_DIR/generated-images.txt"

printf 'SUCCESS\nCompleted: %s\nInput: %s\n' "$(date -Is)" "$INPUT_IMAGE" > "$RESULT_DIR/SUCCESS.txt"
echo "Smoke test completed. Review: $RESULT_DIR"
