#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"

ensure_system_deps() {
    if command -v curl >/dev/null 2>&1; then
        return
    fi

    if command -v apt-get >/dev/null 2>&1; then
        apt-get update
        apt-get install -y curl ca-certificates libgl1 libglib2.0-0
        return
    fi

    echo "curl is required to install uv, and apt-get was not found." >&2
    exit 1
}

ensure_uv() {
    export PATH="${HOME}/.local/bin:${PATH}"
    if command -v uv >/dev/null 2>&1; then
        return
    fi

    ensure_system_deps
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"

    if ! command -v uv >/dev/null 2>&1; then
        echo "uv installation finished, but uv is still not on PATH." >&2
        echo "Try: export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
        exit 1
    fi
}

ensure_uv

echo "[ACT] Using uv: $(command -v uv)"
uv python install "${PYTHON_VERSION}"
uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

uv pip install --upgrade pip setuptools wheel

uv pip install \
    torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 \
    --index-url "${PYTORCH_INDEX_URL}"

uv pip install \
    numpy==1.26.4 \
    pyquaternion==0.9.9 \
    PyYAML==6.0.2 \
    rospkg==1.5.0 \
    pexpect==4.8.0 \
    mujoco==2.3.3 \
    dm-control==1.0.9 \
    matplotlib==3.7.1 \
    einops==0.6.0 \
    packaging==23.0 \
    h5py==3.8.0 \
    ipython==8.12.0 \
    opencv-python==4.7.0.72 \
    tqdm==4.67.1 \
    tensorboard

python - <<'PY'
import torch
import torchvision
import cv2
import h5py
import numpy

print("python ok")
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
print("cv2:", cv2.__version__)
print("h5py:", h5py.__version__)
print("numpy:", numpy.__version__)
PY

echo
echo "[ACT] Environment is ready."
echo "Activate it with: source ${SCRIPT_DIR}/${VENV_DIR}/bin/activate"
