#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

REVISION="6728bcbd5cba8be72810d366cb3a1c9e4f893824"
FILENAME="yolo26s-doclaynet.pt"
OUTPUT_DIR="${SCRIPT_DIR}/checkpoints"
OUTPUT_PATH="${OUTPUT_DIR}/${FILENAME}"
URL="https://huggingface.co/hantian/yolo-doclaynet/resolve/${REVISION}/${FILENAME}?download=true"

mkdir -p "${OUTPUT_DIR}"

if [[ -s "${OUTPUT_PATH}" ]]; then
    echo "Checkpoint already exists: ${OUTPUT_PATH}"
else
    curl \
        --fail \
        --location \
        --retry 5 \
        --retry-delay 5 \
        --output "${OUTPUT_PATH}.part" \
        "${URL}"

    mv "${OUTPUT_PATH}.part" "${OUTPUT_PATH}"
fi

sha256sum "${OUTPUT_PATH}"