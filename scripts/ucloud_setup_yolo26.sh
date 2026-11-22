#!/usr/bin/env bash
set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_URL="https://github.com/Kriglaja/Document-Layout-Parsing-using-OCR.git"
REPO_BRANCH="main"

MODEL_REVISION="6728bcbd5cba8be72810d366cb3a1c9e4f893824"
MODEL_NAME="yolo26s-doclaynet.pt"
MODEL_URL="https://huggingface.co/hantian/yolo-doclaynet/resolve/${MODEL_REVISION}/${MODEL_NAME}?download=true"

# By default, this creates the project inside the UCloud job staging directory.
# See the persistent-storage note below.
WORK_ROOT="${UCLOUD_WORK_ROOT:-${PWD}/yolo26-ucloud-work}"
REPO_DIR="${WORK_ROOT}/Document-Layout-Parsing-using-OCR"

echo "============================================================"
echo "YOLO26-S DocLayNet UCloud setup"
echo "============================================================"
echo "Initial job directory: ${PWD}"
echo "Work directory: ${WORK_ROOT}"
echo "Repository directory: ${REPO_DIR}"
echo "Repository branch: ${REPO_BRANCH}"
echo

# ---------------------------------------------------------------------------
# GPU information
# ---------------------------------------------------------------------------

echo "=== GPU information ==="
nvidia-smi

# ---------------------------------------------------------------------------
# Clone the repository
# ---------------------------------------------------------------------------

mkdir -p "${WORK_ROOT}"

echo "=== Cloning repository ==="

if [[ -d "${REPO_DIR}/.git" ]]; then
    echo "Existing repository found. Updating it."

    git -C "${REPO_DIR}" fetch origin "${REPO_BRANCH}"
    git -C "${REPO_DIR}" checkout "${REPO_BRANCH}"
    git -C "${REPO_DIR}" pull --ff-only origin "${REPO_BRANCH}"
else
    git clone \
        --branch "${REPO_BRANCH}" \
        --single-branch \
        "${REPO_URL}" \
        "${REPO_DIR}"
fi

cd "${REPO_DIR}"

echo "Repository commit:"
git rev-parse HEAD

# ---------------------------------------------------------------------------
# Verify that required repository files exist
# ---------------------------------------------------------------------------

echo "=== Checking required files ==="

if [[ ! -f "yolo26/smoke_test.py" ]]; then
    echo "ERROR: yolo26/smoke_test.py was not found."
    echo "Verify that smoke_test.py is committed to the main branch."
    exit 1
fi

mkdir -p yolo26/checkpoints
mkdir -p yolo26/artifacts

# ---------------------------------------------------------------------------
# Create the Python environment
# ---------------------------------------------------------------------------

echo "=== Creating Python virtual environment ==="

python3 -m venv .venv-yolo26
source .venv-yolo26/bin/activate

python -m pip install --upgrade pip setuptools wheel

# Preserve the PyTorch installation supplied by the UCloud environment where
# possible, because that installation should match the available CUDA driver.
if python -c "import torch" >/dev/null 2>&1; then
    echo "Existing PyTorch installation detected."
else
    echo "PyTorch was not found inside the virtual environment."
    echo "Installing PyTorch and torchvision from PyPI."
    python -m pip install torch torchvision
fi

echo "=== Installing YOLO dependencies ==="

if [[ -f "yolo26/requirements.txt" ]]; then
    python -m pip install -r yolo26/requirements.txt
else
    python -m pip install ultralytics huggingface_hub pillow pyyaml
fi

# ---------------------------------------------------------------------------
# Check CUDA before downloading and loading the model
# ---------------------------------------------------------------------------

echo "=== Checking PyTorch and CUDA ==="

python - <<'PY'
import sys
import torch

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: PyTorch cannot access CUDA. "
        "Check the UCloud image and GPU allocation."
    )

print(f"GPU count: {torch.cuda.device_count()}")

for index in range(torch.cuda.device_count()):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
PY

# ---------------------------------------------------------------------------
# Download the immutable YOLO26-S checkpoint
# ---------------------------------------------------------------------------

MODEL_PATH="yolo26/checkpoints/${MODEL_NAME}"

echo "=== Downloading YOLO26-S checkpoint ==="

if [[ -s "${MODEL_PATH}" ]]; then
    echo "Checkpoint already exists:"
    echo "${MODEL_PATH}"
else
    curl \
        --fail \
        --location \
        --retry 5 \
        --retry-delay 5 \
        --output "${MODEL_PATH}.part" \
        "${MODEL_URL}"

    mv "${MODEL_PATH}.part" "${MODEL_PATH}"
fi

echo "Checkpoint size:"
du -h "${MODEL_PATH}"

# ---------------------------------------------------------------------------
# Record checkpoint provenance
# ---------------------------------------------------------------------------

echo "=== Recording checkpoint provenance ==="

sha256sum "${MODEL_PATH}" \
    | tee yolo26/artifacts/checkpoint-sha256.txt

cat > yolo26/artifacts/checkpoint-source.txt <<EOF
repository=https://huggingface.co/hantian/yolo-doclaynet
revision=${MODEL_REVISION}
filename=${MODEL_NAME}
source_url=${MODEL_URL}
downloaded_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

git rev-parse HEAD \
    > yolo26/artifacts/repository-commit-before-setup.txt

# ---------------------------------------------------------------------------
# Run the smoke test
# ---------------------------------------------------------------------------

echo "=== Running YOLO26-S smoke test ==="

python yolo26/smoke_test.py \
    --model "${MODEL_PATH}" \
    --output-dir yolo26/artifacts/smoke-test

# ---------------------------------------------------------------------------
# Record the software environment
# ---------------------------------------------------------------------------

echo "=== Recording runtime environment ==="

python - <<'PY' | tee yolo26/artifacts/runtime-environment.txt
import platform
import sys

import torch
import ultralytics

print(f"python={sys.version.replace(chr(10), ' ')}")
print(f"platform={platform.platform()}")
print(f"torch={torch.__version__}")
print(f"ultralytics={ultralytics.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"gpu_count={torch.cuda.device_count()}")

for index in range(torch.cuda.device_count()):
    print(f"gpu_{index}={torch.cuda.get_device_name(index)}")
PY

python -m pip freeze > yolo26/artifacts/pip-freeze.txt

# ---------------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------------

echo
echo "============================================================"
echo "YOLO26-S setup and smoke test completed successfully"
echo "============================================================"
echo "Repository: ${REPO_DIR}"
echo "Checkpoint: ${REPO_DIR}/${MODEL_PATH}"
echo "Artifacts: ${REPO_DIR}/yolo26/artifacts"
echo
echo "Smoke-test summary:"
cat yolo26/artifacts/smoke-test/smoke-test-summary.json
echo