#!/bin/bash
# Train ACT on a dex2bench task.
#
# Usage:
#   bash train.sh 06_fruit_bowl_loading
#   bash train.sh 06_fruit_bowl_loading 0  # GPU 0
#   bash train.sh 06_fruit_bowl_loading 0 1  # GPU 0, seed 1
#   bash train.sh 06_fruit_bowl_loading 0 1 /path/to/replay /path/to/output
#   bash train.sh 06_fruit_bowl_loading 0 1 cov_only
#   bash train.sh 06_fruit_bowl_loading 0 1 inv_cov
#
# Inputs:
#   Default: ../../outputs/ur5_rh56dfx/scenes/{task}/replay
#
# Outputs:
#   Default: ../../outputs/logs/act/{task}/exp_{datetime}/
#   Override with the optional 5th argument, or CKPT_DIR=/path/to/output.
#
# View training curves:
#   tensorboard --logdir ../../outputs/logs/act/{task}/exp_.../tb_logs

TASK=${1:?"Usage: bash train.sh TASK [gpu_id] [seed] [dataset_dir] [ckpt_dir]"}
GPU_ID=${2:-0}
SEED=${3:-0}
DATASET_DIR=${4:-${DATASET_DIR:-}}
CKPT_DIR=${5:-${CKPT_DIR:-}}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
cd "$(dirname "$0")"
DEX2BENCH_ROOT="$(cd ../.. && pwd)"

PYTHON_BIN=${PYTHON_BIN:-}
if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -x ".venv/bin/python" ]]; then
        PYTHON_BIN=".venv/bin/python"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        echo "No Python interpreter found. Run 'bash env.sh' in policy/ACT, then retry." >&2
        exit 1
    fi
fi

if ! "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1; then
    echo "Selected Python cannot import torch: ${PYTHON_BIN}" >&2
    echo "Run 'bash env.sh' in policy/ACT to rebuild the ACT environment, then retry." >&2
    echo "You can also set PYTHON_BIN=/path/to/python when calling this script." >&2
    exit 1
fi

_configure_torch_cuda_libs() {
    local torch_lib_dir
    torch_lib_dir=$("${PYTHON_BIN}" -c "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / 'lib')" 2>/dev/null || true)
    if [[ -z "${torch_lib_dir}" || ! -d "${torch_lib_dir}" ]]; then
        return
    fi

    if [[ ! -e "${torch_lib_dir}/libnvrtc.so" ]]; then
        local nvrtc_lib
        nvrtc_lib=$(find "${torch_lib_dir}" -maxdepth 1 -name "libnvrtc-*.so.*" -print -quit)
        if [[ -n "${nvrtc_lib}" ]]; then
            ln -s "$(basename "${nvrtc_lib}")" "${torch_lib_dir}/libnvrtc.so"
        fi
    fi

    export LD_LIBRARY_PATH="${torch_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}
_configure_torch_cuda_libs

# Auto-determine state_dim from robot_key in deploy_policy.yml
_yaml_scalar() {
    local key="$1"
    grep -m1 -E "^[[:space:]]*${key}:" deploy_policy.yml 2>/dev/null | sed -E "s/^[[:space:]]*${key}:[[:space:]]*//; s/[[:space:]]+#.*$//; s/^[[:space:]]+//; s/[[:space:]]+$//"
}
ROBOT_KEY="${ROBOT_KEY:-$(_yaml_scalar robot_key)}"
USE_ACTIVE_DOF="${USE_ACTIVE_DOF:-$(_yaml_scalar use_active_dof)}"
USE_ACTIVE_DOF="${USE_ACTIVE_DOF:-false}"
STATE_DIM=36
ACTIVE_DOF_ARGS=()
if [[ "${USE_ACTIVE_DOF}" == "true" ]]; then
    ACTIVE_DOF_ARGS+=(--use_active_dof)
else
    ACTIVE_DOF_ARGS+=(--no-use_active_dof)
fi
if [[ -n "${ROBOT_KEY}" ]]; then
    ACTIVE_DOF_ARGS+=(--robot_key "${ROBOT_KEY}")
fi
if [[ -n "${ROBOT_KEY}" ]]; then
    _AUTO_DIM=$("${PYTHON_BIN}" -c "import sys; sys.path.insert(0, '${DEX2BENCH_ROOT}'); from robots.active_dof_utils import get_active_dof_info; info = get_active_dof_info('${ROBOT_KEY}'); use_active = '${USE_ACTIVE_DOF}'.lower() == 'true'; print(info.active_dof if use_active else info.full_dof)" 2>/dev/null)
    if [[ -n "${_AUTO_DIM}" ]]; then
        STATE_DIM="${_AUTO_DIM}"
    fi
fi

case "${DATASET_DIR}" in
    cov|cov_only|cov-only)
        DATASET_DIR="../../outputs/ur5_rh56dfx/scenes/${TASK}/replay-cov"
        CKPT_DIR=${CKPT_DIR:-"../../outputs/logs/act/${TASK}_cov_only"}
        ;;
    inv_cov|inv-cov|inv+cov|full)
        DATASET_DIR="../../outputs/ur5_rh56dfx/scenes/${TASK}/replay-inv-cov"
        CKPT_DIR=${CKPT_DIR:-"../../outputs/logs/act/${TASK}_inv_cov"}
        ;;
esac

EXTRA_ARGS=()
if [ -n "${DATASET_DIR}" ]; then
    EXTRA_ARGS+=(--dataset_dir "${DATASET_DIR}")
fi
if [ -n "${CKPT_DIR}" ]; then
    EXTRA_ARGS+=(--ckpt_dir "${CKPT_DIR}")
fi

"${PYTHON_BIN}" imitate_episodes.py \
    --task "${TASK}" \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 30 \
    --hidden_dim 512 \
    --batch_size 32 \
    --dim_feedforward 3200 \
    --num_epochs 6000 \
    --lr 1e-5 \
    --state_dim ${STATE_DIM} \
    "${ACTIVE_DOF_ARGS[@]}" \
    --step_delay 0.12 \
    --seed "${SEED}" \
    "${EXTRA_ARGS[@]}"
