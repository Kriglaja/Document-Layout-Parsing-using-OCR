#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/rtdetrv4"

echo "==> Creating fresh RT-DETRv4 virtual environment in rtdetrv4/venv..."
python3 -m venv venv

echo "==> Activating virtual environment..."
source venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing RT-DETRv4 requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found in rtdetrv4/"
fi

echo "==> RT-DETRv4 environment setup complete!"
