#!/bin/bash
# Finetune Pi0.5 on a dex2bench task (LoRA, loads pretrained pi05_base from S3).
#
# Usage:
#   bash finetune.sh --task TASK [--gpu-use GPU_IDS] [--exp-name EXP]
#   bash finetune.sh --task TASK --gpu-use GPU_IDS --exp-name EXP --dataset-dir /path/to/replay
#   bash finetune.sh --task TASK --exp-name EXP --output-dir /path/to/logs/pi05
#   bash finetune.sh --task TASK --batch-size 32 --num-workers 4
#   bash finetune.sh TASK [gpu_use] [exp_name]   # legacy positional form
#
# Optional flags:
#   --task NAME                 Dex2bench task name, e.g. 21_condiment_box_loading.
#   --gpu-use IDS               CUDA_VISIBLE_DEVICES value, e.g. 0 or 0,1,2,3.
#   --exp-name NAME             Experiment/checkpoint directory name.
#   --dataset-dir PATH          Override replay HDF5 directory.
#   --data-dir PATH             Alias for --dataset-dir.
#   --output-dir PATH           Override checkpoint base dir.
#   --checkpoint-base-dir PATH  Alias for --output-dir.
#   --assets-base-dir PATH      Override norm-stats/assets dir. Missing norm stats are computed automatically before training.
#   --prompt TEXT               Override language prompt instead of reading scenes/<TASK>.yaml.
#   --train-config NAME         Override train_config_name from deploy_policy.yml.
#   --sii, -sii                 Use hard-coded conda env pi05 instead of uv-managed .venv.
#   --recompute-norm-stats      Recompute norm_stats.json even if it already exists.
#   --batch-size N              Forwarded to scripts/train.py.
#   --num-workers N             Forwarded to scripts/train.py.
#   -- ARGS...                  Everything after `--` is passed through to scripts/train.py.

set -e

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DISABLED="${WANDB_DISABLED:-false}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

usage() {
    sed -n '1,30p' "$0" >&2
}

CALLER_CWD="$(pwd)"
POSITIONAL=()
EXTRA_TRAIN_ARGS=()
TASK_OVERRIDE=""
GPU_USE_OVERRIDE=""
EXP_NAME_OVERRIDE=""
DATA_DIR_OVERRIDE=""
CHECKPOINT_BASE_DIR=""
ASSETS_BASE_DIR=""
ASSETS_BASE_DIR_EXPLICIT=0
PROMPT_OVERRIDE=""
TRAIN_CONFIG_OVERRIDE=""
ROBOT_KEY_OVERRIDE=""
TRAIN_CONFIG=pi05_base_dex2bench_lora
_USE_SII=false
RECOMPUTE_NORM_STATS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            TASK_OVERRIDE=${2:?"$1 requires a task name"}
            shift 2
            ;;
        --gpu-use|--gpus)
            GPU_USE_OVERRIDE=${2:?"$1 requires a CUDA_VISIBLE_DEVICES value"}
            shift 2
            ;;
        --exp-name)
            EXP_NAME_OVERRIDE=${2:?"$1 requires an experiment name"}
            shift 2
            ;;
        --dataset-dir|--data-dir)
            DATA_DIR_OVERRIDE=${2:?"$1 requires a path"}
            shift 2
            ;;
        --output-dir|--checkpoint-base-dir)
            CHECKPOINT_BASE_DIR=${2:?"$1 requires a path"}
            shift 2
            ;;
        --assets-base-dir)
            ASSETS_BASE_DIR=${2:?"$1 requires a path"}
            ASSETS_BASE_DIR_EXPLICIT=1
            shift 2
            ;;
        --prompt)
            PROMPT_OVERRIDE=${2:?"$1 requires text"}
            shift 2
            ;;
        --train-config)
            TRAIN_CONFIG_OVERRIDE=${2:?"$1 requires a config name"}
            shift 2
            ;;
        --robot-key)
            ROBOT_KEY_OVERRIDE=${2:?"$1 requires a robot key"}
            shift 2
            ;;
        --robot-key=*)
            ROBOT_KEY_OVERRIDE="${1#*=}"
            shift
            ;;
        --sii|-sii)
            _USE_SII=true
            shift
            ;;
        --recompute-norm-stats|--recompute_norm_stats)
            RECOMPUTE_NORM_STATS=true
            shift
            ;;
        --batch-size|--num-workers|--num-train-steps|--log-interval|--save-interval|--keep-period|--seed|--fsdp-devices|--lr-schedule.peak-lr|--lr-schedule.warmup-steps|--lr-schedule.decay-steps|--lr-schedule.decay-lr)
            EXTRA_TRAIN_ARGS+=("$1" "${2:?"$1 requires a value"}")
            shift 2
            ;;
        --no-save-train-state|--save-train-state)
            EXTRA_TRAIN_ARGS+=("$1")
            shift
            ;;
        --task=*)
            TASK_OVERRIDE="${1#*=}"
            shift
            ;;
        --gpu-use=*|--gpus=*)
            GPU_USE_OVERRIDE="${1#*=}"
            shift
            ;;
        --exp-name=*)
            EXP_NAME_OVERRIDE="${1#*=}"
            shift
            ;;
        --batch-size=*|--num-workers=*|--num-train-steps=*|--log-interval=*|--save-interval=*|--keep-period=*|--seed=*|--fsdp-devices=*|--lr-schedule.peak-lr=*|--lr-schedule.warmup-steps=*|--lr-schedule.decay-steps=*|--lr-schedule.decay-lr=*|--save-train-state=*|--no-save-train-state=*)
            EXTRA_TRAIN_ARGS+=("$1")
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                EXTRA_TRAIN_ARGS+=("$1")
                shift
            done
            ;;
        --*)
            echo "[finetune] ERROR: unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

TASK="${TASK_OVERRIDE}"
GPU_USE="${GPU_USE_OVERRIDE}"
EXP_NAME="${EXP_NAME_OVERRIDE}"
pos_i=0

if [[ -z "${TASK}" && ${pos_i} -lt ${#POSITIONAL[@]} ]]; then
    TASK=${POSITIONAL[$pos_i]}
    ((pos_i+=1))
fi
if [[ -z "${GPU_USE}" && ${pos_i} -lt ${#POSITIONAL[@]} ]]; then
    GPU_USE=${POSITIONAL[$pos_i]}
    ((pos_i+=1))
fi
if [[ -z "${EXP_NAME}" && ${pos_i} -lt ${#POSITIONAL[@]} ]]; then
    EXP_NAME=${POSITIONAL[$pos_i]}
    ((pos_i+=1))
fi

if [[ -z "${TASK}" ]]; then
    echo "[finetune] ERROR: TASK is required" >&2
    usage
    exit 1
fi
if [[ ${pos_i} -lt ${#POSITIONAL[@]} ]]; then
    echo "[finetune] ERROR: too many positional arguments: ${POSITIONAL[*]:$pos_i}" >&2
    usage
    exit 1
fi

GPU_USE=${GPU_USE:-0}
EXP_NAME=${EXP_NAME:-exp_$(date +%Y%m%d_%H%M%S)}

cd "$(dirname "$0")"
DEX2BENCH_ROOT="$(cd ../.. && pwd)"
DEPLOY_CONFIG="${PWD}/deploy_policy.yml"

if [[ "${_USE_SII}" == "true" ]]; then
    source "${DEX2BENCH_ROOT}/../miniconda3/bin/activate" pi05
fi

_abs_path() {
    local path="$1"
    python3 -c 'import os, sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${path}"
}

_abs_from_caller() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        _abs_path "${path}"
    else
        (cd "${CALLER_CWD}" && _abs_path "${path}")
    fi
}

_yaml_scalar() {
    local key="$1"
    local value=""
    if [[ -f "${DEPLOY_CONFIG}" ]]; then
        value=$(grep -m1 -E "^[[:space:]]*${key}:" "${DEPLOY_CONFIG}" | sed -E "s/^[[:space:]]*${key}:[[:space:]]*//; s/[[:space:]]+#.*$//; s/^[[:space:]]+//; s/[[:space:]]+$//")
    fi
    if [[ "${value}" == "null" || "${value}" == '""' || "${value}" == "''" ]]; then
        value=""
    fi
    echo "${value}"
}

TRAIN_CONFIG_FROM_YML=$(_yaml_scalar train_config_name)
if [[ -n "${TRAIN_CONFIG_FROM_YML}" ]]; then
    TRAIN_CONFIG="${TRAIN_CONFIG_FROM_YML}"
fi
if [[ -n "${TRAIN_CONFIG_OVERRIDE}" ]]; then
    TRAIN_CONFIG="${TRAIN_CONFIG_OVERRIDE}"
fi

ACTION_DIM=$(_yaml_scalar action_dim)
STATE_DIM=$(_yaml_scalar state_dim)

USE_ACTIVE_DOF=$(_yaml_scalar use_active_dof)
ROBOT_KEY=$(_yaml_scalar robot_key)
USE_ACTIVE_DOF=${USE_ACTIVE_DOF:-true}

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${DEX2BENCH_ROOT}/../OpenpiCache}"
if [[ -n "${DATA_DIR_OVERRIDE}" ]]; then
    DATA_DIR=$(_abs_from_caller "${DATA_DIR_OVERRIDE}")
else
    DATA_DIR="${DEX2BENCH_ROOT}/outputs/ur5_rh56dfx/scenes/${TASK}/replay"
fi
SCENE_YAML="${DEX2BENCH_ROOT}/scenes/${TASK}.yaml"

if [[ -n "${CHECKPOINT_BASE_DIR}" ]]; then
    CHECKPOINT_BASE_DIR=$(_abs_from_caller "${CHECKPOINT_BASE_DIR}")
fi
if [[ -n "${ASSETS_BASE_DIR}" ]]; then
    ASSETS_BASE_DIR=$(_abs_from_caller "${ASSETS_BASE_DIR}")
elif [[ -n "${CHECKPOINT_BASE_DIR}" ]]; then
    ASSETS_BASE_DIR="${CHECKPOINT_BASE_DIR}/_assets"
fi

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "[finetune] ERROR: data dir not found: ${DATA_DIR}" >&2
    exit 1
fi

if [[ "${USE_ACTIVE_DOF}" == "true" ]]; then
    _PYTHON_FOR_META=python3
    if [[ "${_USE_SII}" == "true" ]]; then
        _PYTHON_FOR_META=python
    elif [[ -x ".venv/bin/python" ]]; then
        _PYTHON_FOR_META=".venv/bin/python"
    fi
    _AUTO_INFO=$("${_PYTHON_FOR_META}" -c '
import pathlib
import sys
sys.path.insert(0, sys.argv[1])
from robots.active_dof_utils import get_active_dof_info, get_active_dof_info_for_hdf5

dataset_dir = pathlib.Path(sys.argv[2])
robot_key = sys.argv[3] or None
info = None
if robot_key:
    info = get_active_dof_info(robot_key)
if info is None:
    for path in sorted(dataset_dir.glob("*.hdf5")):
        try:
            info = get_active_dof_info_for_hdf5(str(path))
        except Exception:
            info = None
        if info is not None:
            break
if info is not None:
    print(f"{info.robot_key} {info.active_dof}")
' "${DEX2BENCH_ROOT}" "${DATA_DIR}" "${ROBOT_KEY}")
    if [[ -n "${_AUTO_INFO}" ]]; then
        ROBOT_KEY="${_AUTO_INFO% *}"
        _AUTO_DIM="${_AUTO_INFO##* }"
        ACTION_DIM="${_AUTO_DIM}"
        STATE_DIM="${_AUTO_DIM}"
    fi
fi

# --robot-key 命令行参数优先级最高，覆盖 yml 和自动检测，同时重新计算 dims
if [[ -n "${ROBOT_KEY_OVERRIDE}" ]]; then
    ROBOT_KEY="${ROBOT_KEY_OVERRIDE}"
    echo "[finetune] robot_key overridden by --robot-key: ${ROBOT_KEY}"
    if [[ "${USE_ACTIVE_DOF}" == "true" ]]; then
        _OVERRIDE_DIMS=$("${_PYTHON_FOR_META}" -c "
import sys
sys.path.insert(0, '${DEX2BENCH_ROOT}')
from robots.active_dof_utils import get_active_dof_info
info = get_active_dof_info('${ROBOT_KEY_OVERRIDE}')
if info is not None:
    print(info.active_dof)
")
        if [[ -n "${_OVERRIDE_DIMS}" ]]; then
            ACTION_DIM="${_OVERRIDE_DIMS}"
            STATE_DIM="${_OVERRIDE_DIMS}"
            echo "[finetune] robot_key override: dims=${_OVERRIDE_DIMS}"
        fi
    fi
fi

ACTION_DIM=${ACTION_DIM:-36}
STATE_DIM=${STATE_DIM:-${ACTION_DIM}}

_extra_args_str="${EXTRA_TRAIN_ARGS[*]:-}"
_yml_default() {
    local flag="$1" key="$2"
    if [[ "${_extra_args_str}" != *"${flag}"* ]]; then
        local val=$(_yaml_scalar "${key}")
        if [[ -n "${val}" ]]; then
            EXTRA_TRAIN_ARGS+=("${flag}" "${val}")
        fi
    fi
}
_yml_default "--batch-size"                batch_size
_yml_default "--fsdp-devices"              fsdp_devices
_yml_default "--num-workers"               num_workers
_yml_default "--num-train-steps"           num_train_steps
_yml_default "--log-interval"              log_interval
_yml_default "--save-interval"             save_interval
_yml_default "--lr-schedule.peak-lr"       lr_schedule_peak_lr
_yml_default "--lr-schedule.warmup-steps"  lr_schedule_warmup_steps
_yml_default "--lr-schedule.decay-steps"   lr_schedule_decay_steps
_yml_default "--lr-schedule.decay-lr"     lr_schedule_decay_lr
_yml_default "--model.action_horizon"   train_action_horizon

if (( STATE_DIM > ACTION_DIM )); then
    echo "[finetune] ERROR: state_dim (${STATE_DIM}) cannot exceed action_dim (${ACTION_DIM})" >&2
    exit 1
fi

export PI05_DEX2BENCH_ACTION_DIM="${ACTION_DIM}"
export PI05_DEX2BENCH_STATE_DIM="${STATE_DIM}"
export PI05_DEX2BENCH_USE_ACTIVE_DOF="${USE_ACTIVE_DOF}"
export PI05_DEX2BENCH_ROBOT_KEY="${ROBOT_KEY}"

if [[ -n "${PROMPT_OVERRIDE}" ]]; then
    PROMPT="${PROMPT_OVERRIDE}"
else
    PROMPT=""
    if [[ -f "${SCENE_YAML}" ]]; then
        PROMPT=$(grep -m1 -E '^description:' "${SCENE_YAML}" | sed -E 's/^description:[[:space:]]*"?([^"]*)"?[[:space:]]*$/\1/')
    fi
    if [[ -z "${PROMPT}" ]]; then
        PROMPT="perform task: ${TASK//_/ }"
    fi
fi

REPO_ID="dex2bench:${DATA_DIR}|${PROMPT}"

TRAIN_ARGS=(
    scripts/train.py "${TRAIN_CONFIG}"
    --exp-name="${EXP_NAME}"
    --data.repo_id="${REPO_ID}"
    --overwrite
)
if [[ -n "${CHECKPOINT_BASE_DIR}" ]]; then
    TRAIN_ARGS+=(--checkpoint-base-dir="${CHECKPOINT_BASE_DIR}")
fi
if [[ -n "${ASSETS_BASE_DIR}" ]]; then
    TRAIN_ARGS+=(--assets-base-dir="${ASSETS_BASE_DIR}")
    if (( ASSETS_BASE_DIR_EXPLICIT )); then
        TRAIN_ARGS+=(--data.assets.assets_dir="${ASSETS_BASE_DIR}")
    fi
fi
if [[ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]]; then
    TRAIN_ARGS+=("${EXTRA_TRAIN_ARGS[@]}")
fi

if [[ -n "${CHECKPOINT_BASE_DIR}" ]]; then
    CHECKPOINT_OUTPUT="${CHECKPOINT_BASE_DIR}/${TRAIN_CONFIG}/${EXP_NAME}/<step>/"
else
    CHECKPOINT_OUTPUT="../../outputs/logs/pi05/${TRAIN_CONFIG}/${EXP_NAME}/<step>/"
fi
if [[ -n "${ASSETS_BASE_DIR}" ]]; then
    if (( ASSETS_BASE_DIR_EXPLICIT )); then
        NORM_STATS_PATH="${ASSETS_BASE_DIR}/norm_stats.json"
    else
        NORM_STATS_PATH="${ASSETS_BASE_DIR}/${TRAIN_CONFIG}/dex2bench/norm_stats.json"
    fi
else
    NORM_STATS_PATH="../../outputs/logs/pi05/_assets/${TRAIN_CONFIG}/dex2bench/norm_stats.json"
fi

RUN_ENV_LABEL="uv"
if [[ "${_USE_SII}" == "true" ]]; then
    RUN_ENV_LABEL="conda:pi05"
fi

cat <<EOF
[finetune] task        = ${TASK}
[finetune] gpu_use     = ${GPU_USE}
[finetune] exp_name    = ${EXP_NAME}
[finetune] train_config= ${TRAIN_CONFIG}
[finetune] action_dim  = ${ACTION_DIM}
[finetune] state_dim   = ${STATE_DIM}
[finetune] data_dir    = ${DATA_DIR}
[finetune] prompt      = ${PROMPT}
[finetune] ckpt_out    = ${CHECKPOINT_OUTPUT}
[finetune] norm_stats  = ${NORM_STATS_PATH}
[finetune] recompute_stats = ${RECOMPUTE_NORM_STATS}
[finetune] run_env     = ${RUN_ENV_LABEL}
EOF

if [[ "${_USE_SII}" != "true" ]]; then
    if command -v uv >/dev/null 2>&1; then
        UV_BIN=uv
    elif [[ -x "$HOME/.local/bin/uv" ]]; then
        UV_BIN="$HOME/.local/bin/uv"
    else
        echo "[finetune] ERROR: 'uv' not found. Pi05 needs an isolated uv venv." >&2
        echo "[finetune]   Install:  pip install uv   (or:  curl -LsSf https://astral.sh/uv/install.sh | sh)" >&2
        echo "[finetune]   Then:     cd policy/pi05 && GIT_LFS_SKIP_SMUDGE=1 uv sync" >&2
        exit 1
    fi

    if [[ ! -x ".venv/bin/python" ]]; then
        echo "[finetune] ERROR: policy/pi05/.venv not found. Run first:" >&2
        echo "[finetune]   cd policy/pi05 && GIT_LFS_SKIP_SMUDGE=1 uv sync" >&2
        exit 1
    fi
fi

_HOMING_PYTHON=python3
if [[ "${_USE_SII}" == "true" ]]; then
    _HOMING_PYTHON=python
elif [[ -x ".venv/bin/python" ]]; then
    _HOMING_PYTHON=".venv/bin/python"
fi

echo -n "[finetune] Checking meta/homing_start_sim_step for all episodes... "
_HOMING_CHECK=$("${_HOMING_PYTHON}" -c '
import h5py
import pathlib
import sys

paths = sorted(pathlib.Path(sys.argv[1]).glob("episode_*.hdf5"))
if not paths:
    print("EMPTY 0")
    sys.exit(0)
failed = []
for path in paths:
    with h5py.File(path, "r") as f:
        homing = f.get("meta/homing_start_sim_step")
        if homing is None or int(homing[()]) < 0:
            failed.append(path.name)
if failed:
    print(f"FAIL {len(failed)} {len(paths)} " + " ".join(failed[:5]))
else:
    print(f"OK {len(paths)}")
' "${DATA_DIR}" 2>/dev/null)
if [[ -z "${_HOMING_CHECK}" ]]; then
    echo ""
    echo "[finetune] ERROR: failed to check homing_start_sim_step with ${_HOMING_PYTHON}" >&2
    exit 1
elif [[ "${_HOMING_CHECK}" == EMPTY* ]]; then
    echo ""
    echo "[finetune] ERROR: no episode_*.hdf5 files found in ${DATA_DIR}" >&2
    exit 1
elif [[ "${_HOMING_CHECK}" == FAIL* ]]; then
    echo ""
    echo "[finetune] ERROR: some episodes are missing meta/homing_start_sim_step or have value < 0." >&2
    echo "[finetune]        ${_HOMING_CHECK}" >&2
    echo "[finetune]        GR00T-aligned truncation requires valid homing markers for all episodes." >&2
    exit 1
fi
echo "${_HOMING_CHECK}"

if [[ "${RECOMPUTE_NORM_STATS}" == "true" || ! -f "${NORM_STATS_PATH}" ]]; then
    NORM_STATS_DIR="$(dirname "${NORM_STATS_PATH}")"
    if [[ "${RECOMPUTE_NORM_STATS}" == "true" && -f "${NORM_STATS_PATH}" ]]; then
        echo "[finetune] recomputing existing norm stats before training:"
    else
        echo "[finetune] norm stats not found; computing before training:"
    fi
    echo "[finetune]   output_dir=${NORM_STATS_DIR}"
    if [[ "${_USE_SII}" == "true" ]]; then
        python scripts/compute_norm_stats.py \
            --config-name "${TRAIN_CONFIG}" \
            --dataset-dir "${DATA_DIR}" \
            --output-dir "${NORM_STATS_DIR}" \
            --prompt "${PROMPT}"
    else
        "${UV_BIN}" run scripts/compute_norm_stats.py \
            --config-name "${TRAIN_CONFIG}" \
            --dataset-dir "${DATA_DIR}" \
            --output-dir "${NORM_STATS_DIR}" \
            --prompt "${PROMPT}"
    fi
    if [[ ! -f "${NORM_STATS_PATH}" ]]; then
        echo "[finetune] ERROR: norm stats were not created: ${NORM_STATS_PATH}" >&2
        exit 1
    fi
else
    echo "[finetune] using existing norm stats: ${NORM_STATS_PATH}"
fi

export CUDA_VISIBLE_DEVICES=${GPU_USE}
if [[ "${_USE_SII}" == "true" ]]; then
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python "${TRAIN_ARGS[@]}"
else
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 "${UV_BIN}" run "${TRAIN_ARGS[@]}"
fi
