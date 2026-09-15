#!/usr/bin/env bash
set -euo pipefail

# One-command environment setup for dex2bench's vendored Isaac-GR00T N1.5 policy.
#
# Usage:
#   bash policy/GR00T_n15/setup_env.sh
#   bash policy/GR00T_n15/setup_env.sh --skip-model-download
#   bash policy/GR00T_n15/setup_env.sh --model-id nv-community/GR00T-N1.5-3B-WaveHand
#
# Defaults:
#   - conda env: GR00T_n15
#   - Python: 3.10
#   - pip index: Tsinghua mirror
#   - model source: ModelScope
#   - model id: nv-community/GR00T-N1.5-3B-WaveHand
#
# Note: as of this integration, the public ModelScope result available for GR00T
# N1.5 is a N1.5-derived WaveHand checkpoint. Override --model-id if you have a
# ModelScope mirror of the base nvidia/GR00T-N1.5-3B checkpoint.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_NAME="${GR00T_CONDA_ENV:-GR00T_n15}"
PYTHON_VERSION="${GR00T_PYTHON_VERSION:-3.10}"
PIP_INDEX_URL_VALUE="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST_VALUE="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
USE_PIP_MIRROR=1
INSTALL_DEPS=1
DOWNLOAD_MODEL=1
MODELSCOPE_MODEL_ID="${GR00T_MODELSCOPE_MODEL_ID:-nv-community/GR00T-N1.5-3B-WaveHand}"
MODELSCOPE_CACHE_DIR="${MODELSCOPE_CACHE:-${REPO_ROOT}/.cache/modelscope}"
HF_HOME_VALUE="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
HF_HUB_CACHE_VALUE="${HF_HUB_CACHE:-${HF_HOME_VALUE}/hub}"
HF_MODULES_CACHE_VALUE="${HF_MODULES_CACHE:-${HF_HOME_VALUE}/modules}"
TORCH_HOME_VALUE="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"
WANDB_DIR_VALUE="${WANDB_DIR:-${REPO_ROOT}/.cache/wandb}"
LOCAL_MODEL_PATH=""

usage() {
    cat <<EOF
One-command environment setup for dex2bench's vendored Isaac-GR00T N1.5 policy.

Usage:
  bash policy/GR00T_n15/setup_env.sh
  bash policy/GR00T_n15/setup_env.sh --skip-model-download
  bash policy/GR00T_n15/setup_env.sh --model-id nv-community/GR00T-N1.5-3B-WaveHand

Defaults:
  - conda env: GR00T_n15
  - Python: 3.10
  - pip index: Tsinghua mirror
  - model source: ModelScope
  - model id: nv-community/GR00T-N1.5-3B-WaveHand

Note: the default ModelScope id is a N1.5-derived WaveHand checkpoint. Override
--model-id if you have a ModelScope mirror of the base nvidia/GR00T-N1.5-3B checkpoint.

Options:
  --env-name NAME             Conda environment name. Default: ${ENV_NAME}
  --python VERSION            Python version for a new env. Default: ${PYTHON_VERSION}
  --model-id ID               ModelScope model id. Default: ${MODELSCOPE_MODEL_ID}
  --model-cache DIR           ModelScope cache dir. Default: ${MODELSCOPE_CACHE_DIR}
  --skip-model-download       Install environment only; do not download weights.
  --skip-install              Do not install Python dependencies.
  --pip-index-url URL         Pip index URL. Default: ${PIP_INDEX_URL_VALUE}
  --no-pip-mirror             Use pip's default index.
  -h, --help                  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            ENV_NAME="${2:?--env-name requires a value}"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="${2:?--python requires a value}"
            shift 2
            ;;
        --model-id)
            MODELSCOPE_MODEL_ID="${2:?--model-id requires a value}"
            shift 2
            ;;
        --model-cache)
            MODELSCOPE_CACHE_DIR="${2:?--model-cache requires a value}"
            shift 2
            ;;
        --skip-model-download)
            DOWNLOAD_MODEL=0
            shift
            ;;
        --skip-install)
            INSTALL_DEPS=0
            shift
            ;;
        --pip-index-url)
            PIP_INDEX_URL_VALUE="${2:?--pip-index-url requires a value}"
            USE_PIP_MIRROR=1
            shift 2
            ;;
        --no-pip-mirror)
            USE_PIP_MIRROR=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[gr00t-setup] ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BIN="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
elif [[ -x "${REPO_ROOT}/../miniconda3/bin/conda" ]]; then
    CONDA_BIN="${REPO_ROOT}/../miniconda3/bin/conda"
else
    echo "[gr00t-setup] ERROR: conda not found. Install Miniconda/Anaconda first." >&2
    exit 1
fi

CONDA_BASE="$("${CONDA_BIN}" info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

mkdir -p \
    "${HF_HUB_CACHE_VALUE}" \
    "${HF_MODULES_CACHE_VALUE}" \
    "${TORCH_HOME_VALUE}" \
    "${WANDB_DIR_VALUE}" \
    "${MODELSCOPE_CACHE_DIR}"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "[gr00t-setup] Reusing conda env: ${ENV_NAME}"
else
    echo "[gr00t-setup] Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda activate "${ENV_NAME}"

ENV_PREFIX="${CONDA_PREFIX}"
ACTIVATE_DIR="${ENV_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${ENV_PREFIX}/etc/conda/deactivate.d"
mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

cat > "${ACTIVATE_DIR}/gr00t_cache.sh" <<EOF
export _GR00T_OLD_HF_HOME="\${HF_HOME:-}"
export _GR00T_OLD_HF_HUB_CACHE="\${HF_HUB_CACHE:-}"
export _GR00T_OLD_HF_MODULES_CACHE="\${HF_MODULES_CACHE:-}"
export _GR00T_OLD_TORCH_HOME="\${TORCH_HOME:-}"
export _GR00T_OLD_WANDB_DIR="\${WANDB_DIR:-}"
export _GR00T_OLD_MODELSCOPE_CACHE="\${MODELSCOPE_CACHE:-}"
export HF_HOME="${HF_HOME_VALUE}"
export HF_HUB_CACHE="${HF_HUB_CACHE_VALUE}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE_VALUE}"
export TORCH_HOME="${TORCH_HOME_VALUE}"
export WANDB_DIR="${WANDB_DIR_VALUE}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE_DIR}"
EOF

cat > "${DEACTIVATE_DIR}/gr00t_cache.sh" <<'EOF'
if [[ -n "${_GR00T_OLD_HF_HOME:-}" ]]; then export HF_HOME="${_GR00T_OLD_HF_HOME}"; else unset HF_HOME; fi
if [[ -n "${_GR00T_OLD_HF_HUB_CACHE:-}" ]]; then export HF_HUB_CACHE="${_GR00T_OLD_HF_HUB_CACHE}"; else unset HF_HUB_CACHE; fi
if [[ -n "${_GR00T_OLD_HF_MODULES_CACHE:-}" ]]; then export HF_MODULES_CACHE="${_GR00T_OLD_HF_MODULES_CACHE}"; else unset HF_MODULES_CACHE; fi
if [[ -n "${_GR00T_OLD_TORCH_HOME:-}" ]]; then export TORCH_HOME="${_GR00T_OLD_TORCH_HOME}"; else unset TORCH_HOME; fi
if [[ -n "${_GR00T_OLD_WANDB_DIR:-}" ]]; then export WANDB_DIR="${_GR00T_OLD_WANDB_DIR}"; else unset WANDB_DIR; fi
if [[ -n "${_GR00T_OLD_MODELSCOPE_CACHE:-}" ]]; then export MODELSCOPE_CACHE="${_GR00T_OLD_MODELSCOPE_CACHE}"; else unset MODELSCOPE_CACHE; fi
unset _GR00T_OLD_HF_HOME _GR00T_OLD_HF_HUB_CACHE _GR00T_OLD_HF_MODULES_CACHE
unset _GR00T_OLD_TORCH_HOME _GR00T_OLD_WANDB_DIR _GR00T_OLD_MODELSCOPE_CACHE
EOF

export HF_HOME="${HF_HOME_VALUE}"
export HF_HUB_CACHE="${HF_HUB_CACHE_VALUE}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE_VALUE}"
export TORCH_HOME="${TORCH_HOME_VALUE}"
export WANDB_DIR="${WANDB_DIR_VALUE}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE_DIR}"

PIP_INDEX_ARGS=()
if [[ "${USE_PIP_MIRROR}" == "1" ]]; then
    PIP_INDEX_ARGS=(-i "${PIP_INDEX_URL_VALUE}" --trusted-host "${PIP_TRUSTED_HOST_VALUE}" --timeout 120 --retries 10)
fi

if [[ "${INSTALL_DEPS}" == "1" ]]; then
    echo "[gr00t-setup] Installing GR00T dependencies into ${ENV_NAME}"
    PYTHONNOUSERSITE=1 python -m pip install "${PIP_INDEX_ARGS[@]}" --upgrade pip
    PYTHONNOUSERSITE=1 python -m pip install "${PIP_INDEX_ARGS[@]}" -e "${SCRIPT_DIR}[base]" modelscope
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${SCRIPT_DIR}/src:${PYTHONPATH:-}"

if [[ "${DOWNLOAD_MODEL}" == "1" ]]; then
    MODEL_PATH_FILE="$(mktemp)"
    echo "[gr00t-setup] Downloading ModelScope model: ${MODELSCOPE_MODEL_ID}"
    PYTHONNOUSERSITE=1 python - "${MODELSCOPE_MODEL_ID}" "${MODELSCOPE_CACHE_DIR}" "${MODEL_PATH_FILE}" <<'PY'
import sys
from pathlib import Path
from modelscope import snapshot_download

model_id, cache_dir, output_file = sys.argv[1:4]
local_path = snapshot_download(model_id, cache_dir=cache_dir)
Path(output_file).write_text(str(local_path), encoding="utf-8")
print(f"[gr00t-setup] ModelScope local path: {local_path}")
PY
    LOCAL_MODEL_PATH="$(cat "${MODEL_PATH_FILE}")"
    rm -f "${MODEL_PATH_FILE}"
    export GR00T_BASE_MODEL_PATH="${LOCAL_MODEL_PATH}"
fi

ENV_FILE="${SCRIPT_DIR}/.env"
cat > "${ENV_FILE}" <<EOF
# Generated by policy/GR00T_n15/setup_env.sh
export GR00T_CONDA_ENV="${ENV_NAME}"
export GR00T_MODELSCOPE_MODEL_ID="${MODELSCOPE_MODEL_ID}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE_DIR}"
export HF_HOME="${HF_HOME_VALUE}"
export HF_HUB_CACHE="${HF_HUB_CACHE_VALUE}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE_VALUE}"
export TORCH_HOME="${TORCH_HOME_VALUE}"
export WANDB_DIR="${WANDB_DIR_VALUE}"
EOF

if [[ -n "${LOCAL_MODEL_PATH}" ]]; then
    cat >> "${ENV_FILE}" <<EOF
export GR00T_BASE_MODEL_PATH="${LOCAL_MODEL_PATH}"
EOF
fi

echo "[gr00t-setup] Running import checks"
PYTHONNOUSERSITE=1 python - <<'PY'
import os
import torch
import transformers
import modelscope
from policy.GR00T_n15.gr00t_dex2bench_config import Dex2BenchGR00TDataConfig

cfg = Dex2BenchGR00TDataConfig()
print(f"[gr00t-setup] python ok")
print(f"[gr00t-setup] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, cuda_count={torch.cuda.device_count()}")
print(f"[gr00t-setup] transformers={transformers.__version__}")
print(f"[gr00t-setup] modelscope={getattr(modelscope, '__version__', 'unknown')}")
print(f"[gr00t-setup] gr00t max dims={cfg.max_state_dim}/{cfg.max_action_dim}")
if os.environ.get("GR00T_BASE_MODEL_PATH"):
    print(f"[gr00t-setup] base model={os.environ['GR00T_BASE_MODEL_PATH']}")
PY

echo "[gr00t-setup] Done."
echo "[gr00t-setup] Activate with: conda activate ${ENV_NAME}"
echo "[gr00t-setup] Local config written to: ${ENV_FILE}"
