#!/bin/bash
# Train GR00T N1.5 on a dex2bench task (HDF5-native mode).
#
# Usage:
#   bash policy/GR00T_n15_Tactile/train.sh TASK_NUM [FLAGS]
#
#   TASK_NUM    Task number, e.g. 60 for breadbasket_fast_food_loading.
#               Auto-searches dataset under:
#                 ../teleopdata/dataset/${TASK_NUM}_*
#
# Flags (all optional):
#   --gpu IDS         CUDA_VISIBLE_DEVICES.                Default: 0
#   --lr LR            Peak learning rate.                  Default: 1e-4
#   --steps N          Max training steps.                  Default: 20000
#   --batch N          Global batch size.                   Default: 64
#   --save-steps N     Checkpoint save interval (steps).    Default: 1000
#   --milestone-steps N  Milestone ckpt interval (steps).   Default: 0 (off)
#   --workers N        DataLoader worker count.             Default: 12
#   --seed N           Random seed.                         Default: 42
#   --tag SUFFIX       Append suffix to output dir.         Default: bs64_step20000_gpu1_lr1
#   --resume           Resume from latest checkpoint (disabled by default; DP defaults on, GR00T does not).
#   --dataset PATH     Use this exact HDF5 episodes dir (skip auto-search).
#   --no-tmux          Run in foreground (tmux on by default).
#   --dry-run          Print the command without running.
#
# Examples:
#   bash policy/GR00T_n15_Tactile/train.sh 60                                      # all defaults
#   bash policy/GR00T_n15_Tactile/train.sh 60 --gpu 0,1,2,3,4,5,6,7 --batch 64    # 8-GPU
#   bash policy/GR00T_n15_Tactile/train.sh 60 --lr 2e-4 --steps 10000 --tag v1     # with tag
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# =========================================================================
# Defaults (matching the user's preferred config)
# =========================================================================
GPU="0"
LR="1e-4"
MAX_STEPS=20000
BATCH=64
SAVE_STEPS=1000
MILESTONE_STEPS=0
WORKERS=12
SEED=42
TAG="bs64_step20000_gpu1_lr1"
RESUME=false
USE_TMUX=true
DRY_RUN=false
HDF5_NATIVE=1
CAMERA_MODE="4cam"
VIDEO_BACKEND="torchvision_av"

# =========================================================================
# Parse arguments
# =========================================================================
TASK_NUM=""
DATASET_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)          GPU="$2";           shift 2 ;;
        --gpu=*)        GPU="${1#*=}";      shift ;;
        --lr)           LR="$2";            shift 2 ;;
        --lr=*)         LR="${1#*=}";       shift ;;
        --steps)        MAX_STEPS="$2";     shift 2 ;;
        --steps=*)      MAX_STEPS="${1#*=}"; shift ;;
        --batch)        BATCH="$2";         shift 2 ;;
        --batch=*)      BATCH="${1#*=}";    shift ;;
        --save-steps)   SAVE_STEPS="$2";    shift 2 ;;
        --save-steps=*) SAVE_STEPS="${1#*=}"; shift ;;
        --milestone-steps)   MILESTONE_STEPS="$2";    shift 2 ;;
        --milestone-steps=*) MILESTONE_STEPS="${1#*=}"; shift ;;
        --workers)      WORKERS="$2";       shift 2 ;;
        --workers=*)    WORKERS="${1#*=}";  shift ;;
        --seed)         SEED="$2";          shift 2 ;;
        --seed=*)       SEED="${1#*=}";     shift ;;
        --tag)          TAG="$2";           shift 2 ;;
        --tag=*)        TAG="${1#*=}";      shift ;;
        --dataset)      DATASET_OVERRIDE="$2"; shift 2 ;;
        --dataset=*)    DATASET_OVERRIDE="${1#*=}"; shift ;;
        --resume)       RESUME=true;        shift ;;
        --no-resume)    RESUME=false;       shift ;;
        --no-tmux)      USE_TMUX=false;     shift ;;
        --dry-run)      DRY_RUN=true;       shift ;;
        -h|--help)      sed -n '2,27p' "$0" >&2; exit 0 ;;
        --*)            echo "ERROR: unknown flag $1" >&2; exit 1 ;;
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
CKPT_BASE="../policy_ckpt"
if [[ "$RESUME" == "true" && "$DRY_RUN" != "true" ]]; then
    # Search only GR00T-specific output dirs for GR00T-format checkpoints
    if [[ -n "$TAG" ]]; then
        _search_dir="$CKPT_BASE/${TASK_NUM}"
        _any_ckpt=$(find "$_search_dir" -path "*gr00t*${TAG}*" -name "checkpoint-*" -type d 2>/dev/null | head -1)
        if [[ -z "$_any_ckpt" ]]; then
            echo "[GR00T] Tag '${TAG}' — no existing checkpoints found, starting fresh."
            RESUME=false
        fi
    else
        _any_ckpt=$(find "$CKPT_BASE/${TASK_NUM}" -path "*gr00t*" -name "checkpoint-*" -type d 2>/dev/null | head -1)
    fi
    if [[ "$RESUME" == "true" && -n "${_any_ckpt:-}" ]]; then
        echo ""
        echo -e "\033[33m[GR00T] Found existing checkpoints for task ${TASK_NUM}:\033[0m"
        find "$CKPT_BASE/${TASK_NUM}" -path "*gr00t*" -name "checkpoint-*" -type d 2>/dev/null | sort | while read -r _f; do
            echo "       $(dirname "$_f" | xargs basename)/$(basename "$_f")"
        done
        echo -n "Resume training? [y/N] "
        read -r _confirm
        if [[ "$_confirm" != "y" && "$_confirm" != "Y" ]]; then
            echo "Aborted. Use --no-resume to start fresh, or --tag NEW_TAG for a new run."
            exit 0
        fi
    fi
fi

# =========================================================================
# Conda env
# =========================================================================
source ../miniconda3/bin/activate groot

# =========================================================================
# Auto-detect dataset (or use --dataset override)
# =========================================================================
if [[ -n "$DATASET_OVERRIDE" ]]; then
    DATASET_DIR="$DATASET_OVERRIDE"
    if [[ ! -d "$DATASET_DIR" ]]; then
        echo "ERROR: --dataset path does not exist: ${DATASET_DIR}" >&2
        exit 1
    fi
else
    DATASET_SEARCH=(
        "../teleopdata/dataset"
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
TASK="${TASK_NAME}"  # GR00T uses full task name for scene lookup

# =========================================================================
# Prompt from scene YAML
# =========================================================================
_scene_prompt() {
    local yaml="${REPO_ROOT}/scenes/${TASK}.yaml"
    local p=""
    if [[ -f "$yaml" ]]; then
        p=$(grep -m1 '^description:' "$yaml" | sed -E 's/^description:[[:space:]]*"?([^"]*)"?[[:space:]]*$/\1/' || true)
    fi
    [[ -z "$p" ]] && p="perform task: ${TASK//_/ }"
    echo "$p"
}
PROMPT=$(_scene_prompt)

# =========================================================================
# Pre-check: ALL episodes must have homing_start_sim_step > 0
# (Consistent with DP's train.sh which does the same check before
#  process_hdf5_to_zarr.py --truncate_at_homing.)
# =========================================================================
if [[ "$DRY_RUN" != "true" ]]; then
    echo -n "[GR00T] Checking meta/homing_start_sim_step for all episodes... "
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
fi

# =========================================================================
# YAML helper
# =========================================================================
_yaml_scalar() {
    grep -m1 "^[[:space:]]*${1}:" "$SCRIPT_DIR/deploy_policy.yml" 2>/dev/null \
        | sed -E "s/^[^:]*:[[:space:]]*//; s/[[:space:]]+#.*//; s/^[[:space:]]+//; s/[[:space:]]+$//" || true
}

# =========================================================================
# Auto-detect robot_key and active dimensions from HDF5
# =========================================================================
ROBOT_KEY=""; STATE_DIM=""; ACTION_DIM=""
_FIRST_HDF5=$(find "$DATASET_DIR" -maxdepth 1 -name "episode_*.hdf5" -print -quit 2>/dev/null || true)
if [[ -n "$_FIRST_HDF5" ]]; then
    ROBOT_KEY=$(python -c "
import h5py
with h5py.File('${_FIRST_HDF5}', 'r') as f:
    rk = f['meta']['robot_key'][()]
    rk = rk.decode() if isinstance(rk, bytes) else str(rk)
    print(rk)
" 2>/dev/null || true)
    if [[ -n "$ROBOT_KEY" ]]; then
        _adim=$(python -c "
import sys; sys.path.insert(0,'${REPO_ROOT}')
from robots.active_dof_utils import get_active_dof_info
print(get_active_dof_info('${ROBOT_KEY}').active_dof)
" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
        STATE_DIM="${_adim:-}"
        ACTION_DIM="${_adim:-}"
    fi
fi

# Fallsbacks from deploy_policy.yml
ROBOT_KEY="${ROBOT_KEY:-$(_yaml_scalar robot_key)}"
STATE_DIM="${STATE_DIM:-$(_yaml_scalar state_dim)}"
ACTION_DIM="${ACTION_DIM:-$(_yaml_scalar action_dim)}"
MAX_STATE_DIM=64
MAX_ACTION_DIM=64

# =========================================================================
# GPU
# =========================================================================
IFS=',' read -ra _ga <<< "$GPU"
NUM_GPUS=${#_ga[@]}
export CUDA_VISIBLE_DEVICES="$GPU"

# =========================================================================
# NCCL + env
# =========================================================================
export NCCL_P2P_DISABLE=1
export HF_HUB_CACHE="../.cache/huggingface/hub"
export HF_HOME="../.cache/huggingface"
export PYTHONPATH="${REPO_ROOT}/policy/GR00T_n15_Tactile/src:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=3
export NO_ALBUMENTATIONS_UPDATE=1

# =========================================================================
# Output dir
# =========================================================================
CKPT_BASE="../policy_ckpt"
OUTPUT_DIR="${CKPT_BASE}/${TASK_NUM}/${ROBOT_KEY:-default}/gr00t_n15_trunc${TAG:+_${TAG}}"

# =========================================================================
# Camera map (4cam default)
# =========================================================================
HDF5_CAMERA_MAP=(
    --hdf5-camera-map "stereo_left=cam_stereo_left"
    --hdf5-camera-map "stereo_right=cam_stereo_right"
    --hdf5-camera-map "right_wrist=cam_wrist_right"
    --hdf5-camera-map "left_wrist=cam_wrist_left"
)

# =========================================================================
# Resume: pass --resume if checkpoints exist and user confirmed.
# (Prompt already happened before model loading.)
# =========================================================================
RESUME_ARGS=()
if [[ "$RESUME" == "true" && "$DRY_RUN" != "true" ]]; then
    _ckpt_dirs=$(find "$OUTPUT_DIR" -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | head -1)
    [[ -n "$_ckpt_dirs" ]] && RESUME_ARGS=(--resume)
fi

# =========================================================================
# Base model
# =========================================================================
BASE_MODEL_PATH=$(_yaml_scalar base_model_path)
[[ "$BASE_MODEL_PATH" == "null" ]] && BASE_MODEL_PATH=""
BASE_MODEL_ARGS=()
[[ -n "$BASE_MODEL_PATH" ]] && BASE_MODEL_ARGS=(--base-model-path "$BASE_MODEL_PATH")
TACTILE_HEIGHT=$(_yaml_scalar tactile_height)
TACTILE_WIDTH=$(_yaml_scalar tactile_width)
TACTILE_HEIGHT=${TACTILE_HEIGHT:-120}
TACTILE_WIDTH=${TACTILE_WIDTH:-120}

# =========================================================================
# Build command
# =========================================================================
ACTIVE_DOF_ARGS=(--hdf5-use-active-dof)
[[ -n "$ROBOT_KEY" ]]  && ACTIVE_DOF_ARGS+=(--hdf5-robot-key "$ROBOT_KEY")
[[ -n "$STATE_DIM" ]]  && ACTIVE_DOF_ARGS+=(--hdf5-state-dim "$STATE_DIM")
[[ -n "$ACTION_DIM" ]] && ACTIVE_DOF_ARGS+=(--hdf5-action-dim "$ACTION_DIM")

CMD=(
    python policy/GR00T_n15_Tactile/scripts/gr00t_finetune.py
    --hdf5-native
    --dataset-path "$DATASET_DIR"
    --output-dir "$OUTPUT_DIR"
    --data-config "policy.GR00T_n15_Tactile.gr00t_dex2bench_config:Dex2BenchGR00TDataConfig"
    --embodiment-tag new_embodiment
    --num-gpus "$NUM_GPUS"
    --batch-size "$BATCH"
    --max-steps "$MAX_STEPS"
    --save-steps "$SAVE_STEPS"
    --milestone-steps "$MILESTONE_STEPS"
    --dataloader-num-workers "$WORKERS"
    --seed "$SEED"
    --max-state-dim "$MAX_STATE_DIM"
    --max-action-dim "$MAX_ACTION_DIM"
    --video-backend "$VIDEO_BACKEND"
    --learning-rate "$LR"
    --report-to tensorboard
    --hdf5-prompt "$PROMPT"
    --hdf5-truncate-at-homing
    --hdf5-tactile-height "$TACTILE_HEIGHT"
    --hdf5-tactile-width "$TACTILE_WIDTH"
    "${BASE_MODEL_ARGS[@]}"
    "${HDF5_CAMERA_MAP[@]}"
    "${ACTIVE_DOF_ARGS[@]}"
    "${RESUME_ARGS[@]}"
)

# =========================================================================
# Summary
# =========================================================================
cat <<EOF

  ╔══════════════════════════════════════════════════════════╗
  ║  GR00T N1.5 Training — ${TASK_NAME}
  ╠══════════════════════════════════════════════════════════╣
  ║  dataset     ${DATASET_DIR}
  ║  output      ${OUTPUT_DIR}
  ║  prompt      ${PROMPT}
  ║  robot_key   ${ROBOT_KEY:-auto}   dim=${STATE_DIM:-?}
  ║  GPU(s)      ${GPU} (${NUM_GPUS}×)   seed=${SEED}
  ║  batch       ${BATCH}   steps=${MAX_STEPS}   save_every=${SAVE_STEPS}   milestone=${MILESTONE_STEPS}
  ║  lr          ${LR}   workers=${WORKERS}
  ║  truncate    homing-start (always on)
  ║  resume      ${RESUME}
  ╚══════════════════════════════════════════════════════════╝

EOF

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] $(printf '%q ' "${CMD[@]}")"
    exit 0
fi

# =========================================================================
# Save human-readable training record to output dir
# =========================================================================
mkdir -p "$OUTPUT_DIR"
cat > "${OUTPUT_DIR}/train_params.txt" <<EOF
GR00T N1.5 Training — ${TASK_NAME}
==================================
started:      $(date '+%Y-%m-%d %H:%M:%S')
task_num:     ${TASK_NUM}
task_name:    ${TASK_NAME}
dataset:      ${DATASET_DIR}
output:       ${OUTPUT_DIR}
prompt:       ${PROMPT}
robot_key:    ${ROBOT_KEY:-auto}
state_dim:    ${STATE_DIM:-?}
action_dim:   ${ACTION_DIM:-?}

GPU:          ${GPU} (${NUM_GPUS}x)
batch_size:   ${BATCH}
max_steps:    ${MAX_STEPS}
save_steps:   ${SAVE_STEPS}
milestone_steps: ${MILESTONE_STEPS}
learning_rate:${LR}
workers:      ${WORKERS}
seed:         ${SEED}
video_backend:${VIDEO_BACKEND}
camera_mode:  ${CAMERA_MODE}
resume:       ${RESUME}
tag:          ${TAG:-<none>}

command:
  ${CMD[*]}
EOF
echo "[GR00T] Training record saved: ${OUTPUT_DIR}/train_params.txt"

# Also tee stdout to a log file inside output_dir (via a wrapper script).
# The actual python command is wrapped to capture all output.
LOG_FILE="${OUTPUT_DIR}/train_$(date +%m%d_%H%M).log"
_tee_cmd="2>&1 | tee -a '${LOG_FILE}'"
echo "[GR00T] Log: ${LOG_FILE}"

# =========================================================================
# Launch
# =========================================================================
if [[ "$USE_TMUX" == "true" ]]; then
    SESSION="gr00t-${TASK_NUM}"
    tmux set -g mouse on 2>/dev/null || true
    tmux new-session -d -s "$SESSION" 2>/dev/null || {
        SESSION="gr00t-${TASK_NUM}-$$"
        tmux new-session -d -s "$SESSION"
    }
    _cmd_str=$(printf '%q ' "${CMD[@]}")
    tmux send-keys -t "$SESSION" "cd ${REPO_ROOT} && ${_cmd_str} ${_tee_cmd}" C-m
    echo "[GR00T] tmux session: $SESSION  (attach: tmux attach -t $SESSION)"
else
    exec "${CMD[@]}"
fi
