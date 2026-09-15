#!/bin/bash
# Train Diffusion Policy on a dex2bench task.
#
# Usage:
#   bash policy/DP/train.sh TASK_NUM [FLAGS]
#
#   TASK_NUM    Task number, e.g. 44 for microwave_bowl_loading.
#               Auto-searches dataset under:
#                 ../teleopdata/dataset/${TASK_NUM}_*
#
# Flags (all optional):
#   --gpu IDS             CUDA_VISIBLE_DEVICES.            Default: 0,1,2,3,4,5,6,7
#   --lr LR               Peak learning rate.              Default: 1e-4
#   --epochs N            Number of training epochs.       Default: 300
#   --warmup N            LR warmup steps.                 Default: 500
#   --batch N             Global batch size.               Default: 512
#   --workers N           DataLoader workers per GPU.      Default: 32
#   --seed N              Random seed.                     Default: 42
#   --resume              Resume from latest checkpoint.  Default: on (use --no-resume to start fresh)
#   --no-tmux             Run in foreground (tmux is on by default).
#   --tag SUFFIX           Append suffix to output dir (e.g. dp_SUFFIX).  Default: home
#   --dataset PATH         Use this exact HDF5 episodes dir (skip auto-search).
#   --dry-run             Print the command without running.
#
# Examples:
#   bash policy/DP/train.sh 44                              # all defaults
#   bash policy/DP/train.sh 44 --gpu 0,1,2,3 --lr 1.5e-4   # 4-GPU
#   bash policy/DP/train.sh 44 --epochs 600                   # long training (tmux by default)
#   bash policy/DP/train.sh 44 --no-tmux                      # run in foreground
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEX2BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# =========================================================================
# Defaults
# =========================================================================
GPU="0,1,2,3,4,5,6,7"
LR="1e-4"
EPOCHS=300
WARMUP=500
BATCH=512
WORKERS=32
SEED=42
RESUME=true
USE_TMUX=true
DRY_RUN=false
WANDB_MODE="offline"
TAG="home"
CONFIG_NAME="robot_dp_36_dex2scene_real.yaml"

# =========================================================================
# Parse arguments
# =========================================================================
TASK_NUM=""
DATASET_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)       GPU="$2";         shift 2 ;;
        --gpu=*)     GPU="${1#*=}";    shift ;;
        --lr)        LR="$2";          shift 2 ;;
        --lr=*)      LR="${1#*=}";     shift ;;
        --epochs)    EPOCHS="$2";      shift 2 ;;
        --epochs=*)  EPOCHS="${1#*=}"; shift ;;
        --warmup)    WARMUP="$2";      shift 2 ;;
        --warmup=*)  WARMUP="${1#*=}"; shift ;;
        --batch)     BATCH="$2";       shift 2 ;;
        --batch=*)   BATCH="${1#*=}";  shift ;;
        --workers)   WORKERS="$2";     shift 2 ;;
        --workers=*) WORKERS="${1#*=}"; shift ;;
        --seed)      SEED="$2";        shift 2 ;;
        --seed=*)    SEED="${1#*=}";   shift ;;
        --resume)    RESUME=true;      shift ;;
        --no-resume)  RESUME=false;     shift ;;
        --no-tmux)   USE_TMUX=false;   shift ;;
        --dry-run)   DRY_RUN=true;     shift ;;
        --tag)       TAG="$2";         shift 2 ;;
        --tag=*)     TAG="${1#*=}";    shift ;;
        --dataset)   DATASET_OVERRIDE="$2"; shift 2 ;;
        --dataset=*) DATASET_OVERRIDE="${1#*=}"; shift ;;
        --wandb)     WANDB_MODE="$2";  shift 2 ;;
        --wandb=*)   WANDB_MODE="${1#*=}"; shift ;;
        --config)    CONFIG_NAME="$2";  shift 2 ;;
        --config=*)  CONFIG_NAME="${1#*=}"; shift ;;
        -h|--help)   sed -n '2,27p' "$0" >&2; exit 0 ;;
        --*)         echo "ERROR: unknown flag $1" >&2; exit 1 ;;
        *)
            if [[ -z "$TASK_NUM" ]]; then TASK_NUM="$1"
            else echo "ERROR: too many positional args" >&2; exit 1; fi
            shift ;;
    esac
done

if [[ -z "$TASK_NUM" ]]; then
    echo "ERROR: TASK_NUM required.  Usage: bash train.sh TASK_NUM [FLAGS]" >&2
    exit 1
fi

# =========================================================================
# Early checkpoint check — runs BEFORE conda/model loading.
# =========================================================================
CKPT_BASE="${DEX2BENCH_ROOT}/../policy_ckpt"
if [[ "$RESUME" == "true" && "$DRY_RUN" != "true" ]]; then
    # Search only DP-specific output dirs for DP-format checkpoints
    if [[ -n "$TAG" ]]; then
        _any_ckpt=$(find "$CKPT_BASE/${TASK_NUM}" -path "*dp*${TAG}*/checkpoints*" -name "bs*_ep*.ckpt" ! -name "*_optimizer*" 2>/dev/null | head -1)
        if [[ -z "$_any_ckpt" ]]; then
            echo "[DP] Tag '${TAG}' — no existing checkpoints found, starting fresh."
            RESUME=false
        fi
    else
        _any_ckpt=$(find "$CKPT_BASE/${TASK_NUM}" -path "*dp*/checkpoints*" -name "bs*_ep*.ckpt" ! -name "*_optimizer*" 2>/dev/null | head -1)
    fi
    if [[ "$RESUME" == "true" && -n "${_any_ckpt:-}" ]]; then
        echo ""
        echo -e "\033[33m[DP] Found existing checkpoints for task ${TASK_NUM}:\033[0m"
        if [[ -n "$TAG" ]]; then
            find "$CKPT_BASE/${TASK_NUM}" -path "*dp*${TAG}*/checkpoints*" -name "bs*_ep*.ckpt" ! -name "*_optimizer*" 2>/dev/null | sort | while read -r _f; do
                echo "       $(basename "$(dirname "$(dirname "$_f")")")/$(basename "$_f")"
            done
        else
            find "$CKPT_BASE/${TASK_NUM}" -path "*dp*/checkpoints*" -name "bs*_ep*.ckpt" ! -name "*_optimizer*" 2>/dev/null | sort | while read -r _f; do
                echo "       $(basename "$(dirname "$(dirname "$_f")")")/$(basename "$_f")"
            done
        fi
        echo -n "Resume training? [y/N] "
        read -r _confirm
        if [[ "$_confirm" != "y" && "$_confirm" != "Y" ]]; then
            echo "Aborted. Use --no-resume to start fresh, or --tag NEW_TAG for a new run."
            exit 0
        fi
    fi
fi

# =========================================================================
# Activate conda env
# =========================================================================
source "${DEX2BENCH_ROOT}/../miniconda3/bin/activate" dp

# Use local SSD cache for torch hub models (avoids per-process download)
export TORCH_HOME="${DEX2BENCH_ROOT}/../.cache/torch"

# =========================================================================
# Auto-detect dataset directory (or use --dataset override)
# =========================================================================
if [[ -n "$DATASET_OVERRIDE" ]]; then
    DATASET_DIR="$DATASET_OVERRIDE"
    if [[ ! -d "$DATASET_DIR" ]]; then
        echo "ERROR: --dataset path does not exist: ${DATASET_DIR}" >&2
        exit 1
    fi
else
    DATASET_SEARCH=(
        "${DEX2BENCH_ROOT}/../teleopdata/dataset"
    )

    DATASET_DIR=""
    for _base in "${DATASET_SEARCH[@]}"; do
        _match=$(find "$_base" -maxdepth 1 -type d -name "${TASK_NUM}_*" 2>/dev/null | head -1)
        if [[ -n "$_match" && -d "${_match}/replay-generalization" ]]; then
            DATASET_DIR="${_match}/replay-generalization"
            break
        fi
    done

    if [[ -z "$DATASET_DIR" ]]; then
        echo "ERROR: no dataset for task ${TASK_NUM}" >&2
        for _base in "${DATASET_SEARCH[@]}"; do
            echo "       checked ${_base}/${TASK_NUM}_*/replay-generalization" >&2
        done
        exit 1
    fi
fi

TASK_NAME="$(basename "$(dirname "$DATASET_DIR")")"

# =========================================================================
# Zarr path (truncated at first success frame)
# =========================================================================
ZARR_PATH="$(echo "$DATASET_DIR" | sed 's|/teleopdata/dataset/|/teleopdata_zarr_truncate/dataset/|').zarr"

if [[ ! -d "$ZARR_PATH" && "$DRY_RUN" != "true" ]]; then
    # ---------------------------------------------------------------------
    # Pre-check: ALL episodes must have homing_start_sim_step > 0
    # ---------------------------------------------------------------------
    echo -n "[DP] Checking meta/homing_start_sim_step for all episodes... "
    _chk_result=$(
        python -c "
import sys, glob, h5py
paths = sorted(glob.glob('${DATASET_DIR}/episode_*.hdf5'))
if not paths:
    print('EMPTY 0')
    sys.exit(0)
failed = []
for p in paths:
    with h5py.File(p, 'r') as f:
        h = f.get('meta/homing_start_sim_step')
        if h is None or int(h[()]) < 0:
            failed.append(p.split('/')[-1])
if failed:
    print(len(failed), len(paths))
    for name in failed[:5]:
        print(name)
    sys.exit(0)
print('OK', len(paths))
" 2>/dev/null
    )
    if [[ "$_chk_result" =~ ^([0-9]+)\ ([0-9]+)$ ]]; then
        _n_fail="${BASH_REMATCH[1]}"
        _n_total="${BASH_REMATCH[2]}"
        echo ""
        echo -e "\033[31mERROR: ${_n_fail}/${_n_total} episodes missing homing_start_sim_step or value < 0.\033[0m" >&2
        echo "       Truncation requires ALL episodes to have homing_start_sim_step > 0. Aborting." >&2
        exit 1
    fi
    echo "$_chk_result"

    # ---------------------------------------------------------------------
    # Convert HDF5 → Zarr with success-frame truncation
    # ---------------------------------------------------------------------
    echo -e "\033[33m[DP] converting HDF5 → Zarr (truncated at homing): ${DATASET_DIR}\033[0m"
    python "$SCRIPT_DIR/process_hdf5_to_zarr.py" \
        --dataset_dir "$DATASET_DIR" --output "$ZARR_PATH" \
        --truncate_at_homing
    echo -e "\033[33m[DP] done.\033[0m"
fi

# =========================================================================
# Auto-detect robot_key / action_dim
# =========================================================================
_yaml_scalar() {
    grep -m1 "^[[:space:]]*${1}:" "$SCRIPT_DIR/deploy_policy.yml" 2>/dev/null \
        | sed -E "s/^[^:]*:[[:space:]]*//; s/[[:space:]]+#.*//; s/^[[:space:]]+//; s/[[:space:]]+$//" || true
}

_detect_from_hdf5() {
    python -c "
import sys, glob, os
sys.path.insert(0, '${DEX2BENCH_ROOT}')
from robots.active_dof_utils import get_active_dof_info_for_hdf5, robot_key_from_hdf5, get_active_dof_info
ds = '${DATASET_DIR}'
paths = sorted(glob.glob(os.path.join(ds, 'episode_*.hdf5')))
if not paths: sys.exit(1)
adi = get_active_dof_info_for_hdf5(paths[0])
if adi is not None: print(adi.robot_key, adi.active_dof); sys.exit(0)
rk = robot_key_from_hdf5(paths[0])
if rk is not None:
    adi = get_active_dof_info(rk); print(adi.robot_key, adi.active_dof); sys.exit(0)
sys.exit(1)
" 2>/dev/null
}

ROBOT_KEY=""; ACTION_DIM=""
USE_ACTIVE_DOF=$(_yaml_scalar use_active_dof); USE_ACTIVE_DOF=${USE_ACTIVE_DOF:-true}
if [[ "$USE_ACTIVE_DOF" == "true" ]]; then
    _h5=$(_detect_from_hdf5) || true
    if [[ -n "$_h5" ]]; then
        ROBOT_KEY=$(echo "$_h5" | awk '{print $1}')
        ACTION_DIM=$(echo "$_h5" | awk '{print $2}')
    fi
    if [[ -z "$ROBOT_KEY" ]]; then
        ROBOT_KEY=$(_yaml_scalar robot_key)
        if [[ -n "$ROBOT_KEY" ]]; then
            ACTION_DIM=$(python -c "
import sys; sys.path.insert(0,'${DEX2BENCH_ROOT}')
from robots.active_dof_utils import get_active_dof_info
print(get_active_dof_info('${ROBOT_KEY}').active_dof)
" 2>/dev/null | tail -1)
        fi
    fi
    if [[ -z "$ROBOT_KEY" || -z "$ACTION_DIM" || "$ACTION_DIM" -le 0 ]] 2>/dev/null; then
        echo "ERROR: failed to detect robot_key / action_dim from dataset ${DATASET_DIR}" >&2
        echo "       robot_key=${ROBOT_KEY:-<empty>}  action_dim=${ACTION_DIM:-<empty>}" >&2
        exit 1
    fi
fi

# =========================================================================
# Checkpoint output dir
# =========================================================================
CKPT_BASE="${DEX2BENCH_ROOT}/../policy_ckpt"
OUTPUT_DIR="${CKPT_BASE}/${TASK_NUM}/${ROBOT_KEY:-default}/dp${TAG:+_${TAG}}"

# =========================================================================
# Resume: pass override if checkpoints exist (user already confirmed).
# =========================================================================

# =========================================================================
# GPU setup
# =========================================================================
IFS=',' read -ra _gpu_arr <<< "$GPU"
NUM_GPUS=${#_gpu_arr[@]}
if [[ $((BATCH % NUM_GPUS)) -ne 0 ]]; then
    echo "ERROR: global_batch (${BATCH}) must be divisible by GPU count (${NUM_GPUS})" >&2
    exit 1
fi
PER_GPU=$((BATCH / NUM_GPUS))
# Auto-pick master_port from first GPU to avoid conflicts when running
# multiple DDP jobs on the same node (e.g. two 4-GPU splits).
MASTER_PORT=$((29500 + ${_gpu_arr[0]}))

export CUDA_VISIBLE_DEVICES="$GPU"
if [[ "$NUM_GPUS" -gt 1 ]]; then
    TORCHRUN="$(dirname "$(which python)")/torchrun"
    [[ -x "$TORCHRUN" ]] || TORCHRUN="torchrun"
    LAUNCHER=("$TORCHRUN" --nproc_per_node="$NUM_GPUS" --master_port "$MASTER_PORT")
    DEVICE="cuda"; MULTI_GPU="true"
else
    LAUNCHER=(python)
    DEVICE="cuda:0"; MULTI_GPU="false"
fi

# =========================================================================
# NCCL
# =========================================================================
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
export NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export HYDRA_FULL_ERROR=1

# =========================================================================
# Build hydra overrides
# =========================================================================
HYDRA_OVERRIDES=(
    "+task.dataset.dataset_dir=${DATASET_DIR}"
    "task.dataset.zarr_path=${ZARR_PATH}"
    "hydra.run.dir=${OUTPUT_DIR}"
    "training.seed=${SEED}"
    "training.device=${DEVICE}"
    "training.multi_gpu=${MULTI_GPU}"
    "training.num_epochs=${EPOCHS}"
    "training.lr_warmup_steps=${WARMUP}"
    "training.checkpoint_every=5"
    "+training.max_keep_checkpoints=2"
    "+training.milestone_epochs=100"
    "training.global_batch_size=${BATCH}"
    "dataloader.batch_size=${PER_GPU}"
    "val_dataloader.batch_size=${PER_GPU}"
    "dataloader.num_workers=${WORKERS}"
    "dataloader.persistent_workers=true"
    "optimizer.lr=${LR}"
    "task.use_active_dof=${USE_ACTIVE_DOF}"
    "task.action_dim=${ACTION_DIM}"
    "task.name=${TASK_NAME}"
    "logging.mode=${WANDB_MODE}"
)
[[ -n "$ROBOT_KEY" ]] && HYDRA_OVERRIDES+=("task.robot_key=${ROBOT_KEY}")
[[ "$RESUME" == "true" ]] && HYDRA_OVERRIDES+=("training.resume=true")

# =========================================================================
# Lifecycle management (kill residual GPU processes on exit)
# =========================================================================
_GPU_IDS="${GPU//,/ }"
_dp_kill_procs() {
    echo -e "\033[31m[DP] cleaning up...\033[0m" >&2
    if [[ -n "${_TRAIN_PID:-}" ]] && kill -0 "${_TRAIN_PID}" 2>/dev/null; then
        _pgid=$(ps -o pgid= -p "${_TRAIN_PID}" 2>/dev/null | tr -d ' ')
        [[ -n "$_pgid" ]] && kill -TERM -- -"$_pgid" 2>/dev/null || true
        sleep 2
        kill -KILL "${_TRAIN_PID}" 2>/dev/null || true
    fi
    for _gid in $_GPU_IDS; do
        _pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$_gid" 2>/dev/null | tr '\n' ' ' || true)
        [[ -n "${_pids// /}" ]] && kill -KILL $_pids 2>/dev/null || true
    done
    echo -e "\033[31m[DP] cleanup done\033[0m" >&2
}
trap '_dp_kill_procs; exit 1' SIGINT SIGTERM

# =========================================================================
# Summary
# =========================================================================
cat <<EOF

  ╔══════════════════════════════════════════════════════════╗
  ║  DP Training — ${TASK_NUM}  (${TASK_NAME})
  ╠══════════════════════════════════════════════════════════╣
  ║  dataset    ${DATASET_DIR}
  ║  config     ${CONFIG_NAME}
  ║  zarr       ${ZARR_PATH}
  ║  output     ${OUTPUT_DIR}
  ║  robot_key  ${ROBOT_KEY:-auto}   action_dim=${ACTION_DIM}
  ║  GPU(s)     ${GPU} (${NUM_GPUS}×)   seed=${SEED}
  ║  batch      ${BATCH} (per-gpu=${PER_GPU})   workers=${WORKERS}
  ║  lr         ${LR}   warmup=${WARMUP}   epochs=${EPOCHS}
  ║  ckpt       every 5ep   keep=2   milestone=100
  ║  resume     ${RESUME}   wandb=${WANDB_MODE}
  ╚══════════════════════════════════════════════════════════╝

EOF

CMD=("${LAUNCHER[@]}" train.py --config-name="${CONFIG_NAME}" "${HYDRA_OVERRIDES[@]}")

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] $(printf '%q ' "${CMD[@]}")"
    exit 0
fi

# =========================================================================
# Save human-readable training record
# =========================================================================
mkdir -p "$OUTPUT_DIR"
cat > "${OUTPUT_DIR}/train_params.txt" <<EOF
DP Training — ${TASK_NUM}  (${TASK_NAME})
==========================================
started:      $(date '+%Y-%m-%d %H:%M:%S')
task_num:     ${TASK_NUM}
task_name:    ${TASK_NAME}
dataset:      ${DATASET_DIR}
config:       ${CONFIG_NAME}
output:       ${OUTPUT_DIR}
robot_key:    ${ROBOT_KEY:-auto}
action_dim:   ${ACTION_DIM}

GPU:          ${GPU} (${NUM_GPUS}x)
batch:        ${BATCH} (per-gpu=${PER_GPU})
epochs:       ${EPOCHS}
lr:           ${LR}
warmup:       ${WARMUP}
workers:      ${WORKERS}
seed:         ${SEED}
resume:       ${RESUME}
wandb:        ${WANDB_MODE}
tag:          ${TAG:-<none>}

command:
  ${CMD[*]}
EOF
echo "[DP] Training record: ${OUTPUT_DIR}/train_params.txt"

LOG_FILE="${OUTPUT_DIR}/train_$(date +%m%d_%H%M).log"
_tee_cmd="2>&1 | tee -a '${LOG_FILE}'"
echo "[DP] Log: ${LOG_FILE}"

# =========================================================================
# Launch
# =========================================================================
if [[ "$USE_TMUX" == "true" ]]; then
    SESSION="dp-${TASK_NUM}"
    tmux set -g mouse on 2>/dev/null || true
    tmux new-session -d -s "$SESSION" 2>/dev/null || {
        SESSION="dp-${TASK_NUM}-$$"
        tmux new-session -d -s "$SESSION"
    }
    _cmd_str=$(printf '%q ' "${CMD[@]}")
    tmux send-keys -t "$SESSION" "cd ${SCRIPT_DIR} && export CUDA_VISIBLE_DEVICES=${GPU} && ${_cmd_str} ${_tee_cmd}" C-m
    echo "[DP] tmux session: $SESSION  (attach: tmux attach -t $SESSION)"
else
    "${CMD[@]}" &
    _TRAIN_PID=$!
    wait "${_TRAIN_PID}" || true
    _dp_kill_procs
fi
