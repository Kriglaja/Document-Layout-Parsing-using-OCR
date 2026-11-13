 #!/usr/bin/env bash
set -Eeuo pipefail

# ==========================================
# Configurable Parameters (Override via CLI or Env)
# ==========================================
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/work/document-layout-parsing}"
JOB_NAME="${JOB_NAME:-rtdetrv4-run}"
REPO_URL="${REPO_URL:-https://github.com/Kriglaja/Document-Layout-Parsing-using-OCR.git}"
DEVICE="${DEVICE:-cuda:0}"

# Derived Paths
RUN_DIR="$WORKSPACE_ROOT/$JOB_NAME"
REPO_DIR="$RUN_DIR/Document-Layout-Parsing-using-OCR"
VENV_DIR="$RUN_DIR/venv"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
RESULT_DIR="$RUN_DIR/results"
LOG_FILE="$RESULT_DIR/run.log"

mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$RESULT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
  local code=$?
  printf 'FAILED\nExit code: %s\nSee run.log for details.\n' "$code" > "$RESULT_DIR/FAILED.txt"
  exit "$code"
}
trap on_error ERR
rm -f "$RESULT_DIR/FAILED.txt" "$RESULT_DIR/SUCCESS.txt"

echo "=== RT-DETRv4 Generic Pipeline Runner ==="
date -Is

# ==========================================
# Input Resolution (Accepts CLI argument or scans workspace)
# ==========================================
INPUT_PATH="${1:-}"
if [ -z "$INPUT_PATH" ]; then
  # Default fallback: look for test images or scan workspace
  INPUT_PATH="$(find "$WORKSPACE_ROOT" -type f \( -iname 'test.jpeg' -o -iname 'test.jpg' -o -iname 'test.png' \) -print -quit)"
fi

if [ -z "${INPUT_PATH:-}" ] || [ ! -f "$INPUT_PATH" ]; then
  echo "Error: No valid input file provided or found under $WORKSPACE_ROOT"
  echo "Usage: ./run_pipeline.sh [path_to_image_or_video]"
  exit 2
fi
echo "Input target: $INPUT_PATH"

# ==========================================
# Environment Provisioning
# ==========================================
nvidia-smi | tee "$RESULT_DIR/nvidia-smi.txt"
python --version | tee "$RESULT_DIR/python-version.txt"

if [ ! -d "$VENV_DIR" ]; then
  python -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
else
  cd "$REPO_DIR" && git pull origin main || true
fi
cd "$REPO_DIR"
git rev-parse HEAD | tee "$RESULT_DIR/upstream-commit.txt"

if [ -f "rtdetrv4/requirements.txt" ]; then
  grep -Ev '^(torch|torchvision)([<=> ].*)?$' rtdetrv4/requirements.txt > "$RUN_DIR/requirements-without-torch.txt"
  python -m pip install -r "$RUN_DIR/requirements-without-torch.txt"
fi
python -m pip install gdown opencv-python-headless

# ==========================================
# Checkpoint Acquisition & Verification
# ==========================================
CHECKPOINT="$CHECKPOINT_DIR/rtv4_hgnetv2_s_coco.pth"
if [ ! -s "$CHECKPOINT" ]; then
  echo "Downloading model weights..."
  gdown 'https://drive.google.com/uc?id=1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z' -O "$CHECKPOINT"
fi

python - <<EOF
import torch
checkpoint_path = "$CHECKPOINT"
try:
    print(f"Verifying checkpoint integrity: {checkpoint_path}")
    torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print("Checkpoint verification passed successfully.")
except Exception as e:
    print(f"ERROR: Checkpoint file is corrupted or not a valid PyTorch model file: {e}")
    import sys
    sys.exit(1)
EOF

ls -lh "$CHECKPOINT" | tee "$RESULT_DIR/checkpoint.txt"

# ==========================================
# Execution (Inference / Evaluation)
# ==========================================
cd rtdetrv4
python tools/inference/torch_inf.py \
  -c configs/rtv4/rtv4_hgnetv2_s_coco.yml \
  -r "$CHECKPOINT" \
  --input "$INPUT_PATH" \
  --device "$DEVICE"

# Collect outputs
find . -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.mp4' \) \
  -mmin -15 -print | tee "$RESULT_DIR/generated-outputs.txt"

printf 'SUCCESS\nCompleted: %s\nInput: %s\n' "$(date -Is)" "$INPUT_PATH" > "$RESULT_DIR/SUCCESS.txt"
echo "Pipeline execution completed successfully. Review results in: $RESULT_DIR"
