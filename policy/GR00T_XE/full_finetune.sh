#!/bin/bash
# GR00T XE Full (Single-Stage) Finetune Launcher
# ==============================================
#
# Trains ONE task end-to-end from the base GR00T-N1.5-3B -- no cross-embodiment
# pretrain, no checkpoint handover.  Same recipe GR00T n15 was trained with
# (learning rate 1e-4, 20000 steps, batch 64, one GPU), so a run that works here
# proves the XE data pipeline (64-dim unified space, EE arm dims + IK decode) and
# the inference pipeline are sound, and leaves the two-stage pretrain/finetune
# as the only explanation for the eval failures.
#
# This is NOT finetune.sh with other hyperparameters: finetune.sh starts from
# policy_ckpt/gr00t_xe_pretrain, this one starts from the base model.  The
# difference matters to finetune.py, which has to load the base with its own
# config (action_dim=32) and let the action head be rebuilt at 64.
#
# Usage:
#   bash policy/GR00T_XE/full_finetune.sh [TASK_NUM] [FLAGS]
#
#   TASK_NUM                 Task number.                       Default: 08
#
# Flags:
#   --base-model-path PATH   Base GR00T-N1.5-3B dir.   Default: $BASE_MODEL_DEFAULT
#   --gpu IDS                CUDA_VISIBLE_DEVICES.              Default: 0
#   --lr LR                  Peak learning rate.                Default: 1e-4
#   --steps N                Max training steps.                Default: 20000
#   --batch N                Global batch size.                 Default: 64
#   --save-steps N           Checkpoint save interval.          Default: 1000
#   --workers N              DataLoader worker count.           Default: 12
#   --seed N                 Random seed.                       Default: 42
#   --tag SUFFIX             Append suffix to output dir.
#   --dataset PATH           Override replay HDF5 dir.
#   --prompt TEXT            Language prompt.   Default: scenes/<task>.yaml description
#   --resume                 Resume from latest checkpoint.
#   --no-tmux                Run in foreground.
#   --dry-run                Print the command without running.
#
# Examples:
#   bash policy/GR00T_XE/full_finetune.sh                     # task 08, all defaults
#   bash policy/GR00T_XE/full_finetune.sh 34 --tag v1
#   bash policy/GR00T_XE/full_finetune.sh 08 --batch 32 --steps 10000 --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CKPT_BASE="/inspire/hdd2/project/roboticsystem2/public/DEX2BENCH/zdj/policy_ckpt"
BASE_MODEL_DEFAULT="/inspire/hdd2/project/roboticsystem2/ky26063/bench2dex_stuff/zyd/GR00T-N1.5-3B"

# =========================================================================
# Defaults (n15's recipe: single-stage, 1e-4, 20000 steps, batch 64, 1 GPU)
# =========================================================================
TASK_NUM="08"
GPU="0"
LR="1e-4"
MAX_STEPS=20000
BATCH=64
SAVE_STEPS=1000
WORKERS=12
SEED=42
TAG=""
RESUME=false
USE_TMUX=true
DRY_RUN=false
BASE_MODEL_PATH=""
DATASET_OVERRIDE=""
PROMPT_OVERRIDE=""

# =========================================================================
# Parse arguments
# =========================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-model-path)   BASE_MODEL_PATH="$2"; shift 2 ;;
        --base-model-path=*) BASE_MODEL_PATH="${1#*=}"; shift ;;
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
        --workers)      WORKERS="$2";       shift 2 ;;
        --workers=*)    WORKERS="${1#*=}";  shift ;;
        --seed)         SEED="$2";          shift 2 ;;
        --seed=*)       SEED="${1#*=}";     shift ;;
        --tag)          TAG="$2";           shift 2 ;;
        --tag=*)        TAG="${1#*=}";      shift ;;
        --dataset)      DATASET_OVERRIDE="$2"; shift 2 ;;
        --dataset=*)    DATASET_OVERRIDE="${1#*=}"; shift ;;
        --prompt)       PROMPT_OVERRIDE="$2"; shift 2 ;;
        --prompt=*)     PROMPT_OVERRIDE="${1#*=}"; shift ;;
        --resume)       RESUME=true;        shift ;;
        --no-resume)    RESUME=false;       shift ;;
        --no-tmux)      USE_TMUX=false;     shift ;;
        --dry-run)      DRY_RUN=true;       shift ;;
        -h|--help)
            sed -n '2,36p' "$0" >&2
            exit 0
            ;;
        --*)            echo "ERROR: unknown flag $1" >&2; exit 1 ;;
        *)
            if [[ "$TASK_NUM" == "08" && -z "${_TASK_SET:-}" ]]; then
                TASK_NUM="$1"; _TASK_SET=1
            else
                echo "ERROR: too many positional args" >&2; exit 1
            fi
            shift ;;
    esac
done

if [[ -z "$BASE_MODEL_PATH" ]]; then
    BASE_MODEL_PATH="$BASE_MODEL_DEFAULT"
fi

# =========================================================================
# Conda env
# =========================================================================
source /inspire/hdd/project/roboticsystem2/ky26063/bench2dex_stuff/miniconda3/bin/activate groot

# =========================================================================
# NCCL + env
# =========================================================================
export NCCL_P2P_DISABLE=1
export HF_HUB_CACHE="/inspire/ssd/project/roboticsystem2/ky26063/bench2dex_stuff/.cache/huggingface/hub"
export HF_HOME="/inspire/ssd/project/roboticsystem2/ky26063/bench2dex_stuff/.cache/huggingface"
export PYTHONPATH="${REPO_ROOT}/policy/GR00T_XE/src:${REPO_ROOT}/policy/GR00T_XE:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=3
export NO_ALBUMENTATIONS_UPDATE=1

# =========================================================================
# Base model -- REQUIRED.  finetune.sh treats a missing --base-model-path as
# "skip the flag" because it only uses it as a fallback; here it IS the starting
# point, so a missing or non-directory path must abort rather than quietly train
# something else.
# =========================================================================
if [[ ! -d "$BASE_MODEL_PATH" ]]; then
    echo "ERROR: base model dir not found: ${BASE_MODEL_PATH}" >&2
    echo "       This script trains FROM the base model; without it there is nothing" >&2
    echo "       to start from.  Pass --base-model-path PATH." >&2
    exit 1
fi
if [[ ! -f "$BASE_MODEL_PATH/config.json" ]] \
   || ! compgen -G "$BASE_MODEL_PATH/model*.safetensors*" > /dev/null; then
    echo "ERROR: ${BASE_MODEL_PATH} has no config.json + model*.safetensors*" >&2
    echo "       (looks like a docs-only snapshot, not a loadable checkpoint)" >&2
    exit 1
fi
# The whole point of starting from base is a 32-dim action head that finetune.py
# rebuilds at 64.  If it is already 64 this is not the base model and the run
# would not be the single-stage control it claims to be.
_base_adim=$(python -c "
import json, sys
c = json.load(open('${BASE_MODEL_PATH}/config.json'))
ah = c.get('action_head_cfg') or {}
print((ah.get('action_dim') if isinstance(ah, dict) else getattr(ah, 'action_dim', None)) or c.get('action_dim'))
" 2>/dev/null || true)
if [[ "$_base_adim" != "32" ]]; then
    echo "ERROR: ${BASE_MODEL_PATH} reports action_dim=${_base_adim:-<unreadable>}, expected 32." >&2
    echo "       That is the stock GR00T-N1.5-3B value.  Point --base-model-path at the" >&2
    echo "       base model, not an XE checkpoint (use finetune.sh for pretrain-based runs)." >&2
    exit 1
fi

# =========================================================================
# Auto-detect dataset
# =========================================================================
if [[ -n "$DATASET_OVERRIDE" ]]; then
    DATASET_DIR="$DATASET_OVERRIDE"
else
    DATASET_SEARCH=(
        "/inspire/hdd2/project/roboticsystem2/public/DEX2BENCH/zdj/teleopdata/dataset"
        "/inspire/ssd/project/roboticsystem2/ky26063/bench2dex_stuff/teleopdata/dataset"
    )
    DATASET_DIR=""
    for _base in "${DATASET_SEARCH[@]}"; do
        _match=$(find "$_base" -maxdepth 1 -type d -name "${TASK_NUM}_*" 2>/dev/null | head -1)
        if [[ -n "$_match" && -d "${_match}/replay-generalization" ]]; then
            DATASET_DIR="${_match}/replay-generalization"
            break
        fi
    done
fi

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "ERROR: no dataset for task ${TASK_NUM}" >&2
    exit 1
fi

TASK_NAME="$(basename "$(dirname "$DATASET_DIR")")"

# =========================================================================
# Prompt from scene YAML
# =========================================================================
# The prompt MUST equal what the eval feeds the model: run_policy.py passes
# task["description"] (= yaml.safe_load(scenes/<task>.yaml)["description"]) down
# to policy_sessions, which hands it to the policy as `instruction`.  Parse the
# YAML rather than grepping it -- `grep -m1 '^description:'` returns only the
# FIRST PHYSICAL LINE of a wrapped scalar.
_scene_prompt() {
    if [[ -n "$PROMPT_OVERRIDE" ]]; then
        printf '%s' "$PROMPT_OVERRIDE"
        return 0
    fi
    local yaml="${REPO_ROOT}/scenes/${TASK_NAME}.yaml"
    if [[ ! -f "$yaml" ]]; then
        echo "ERROR: scene YAML not found: $yaml" >&2
        echo "       The training prompt must match the eval's task description." >&2
        echo "       Pass --prompt '...' to set it explicitly." >&2
        return 1
    fi
    python - "$yaml" <<'PY'
import sys

import yaml

with open(sys.argv[1]) as fh:
    desc = (yaml.safe_load(fh) or {}).get("description")
if not desc or not str(desc).strip():
    sys.stderr.write(f"ERROR: no 'description:' key in {sys.argv[1]}\n")
    sys.exit(1)
# Raw (no whitespace munging): byte-identical to what run_policy.py sends.
sys.stdout.write(str(desc))
PY
}
if ! PROMPT=$(_scene_prompt); then
    echo "ERROR: could not determine the language prompt for ${TASK_NAME}." >&2
    exit 1
fi

# =========================================================================
# Auto-detect robot_key and active dimensions
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

# =========================================================================
# GPU + env
# =========================================================================
IFS=',' read -ra _ga <<< "$GPU"
NUM_GPUS=${#_ga[@]}
export CUDA_VISIBLE_DEVICES="$GPU"
export NCCL_P2P_DISABLE=1
export HF_HUB_CACHE="/inspire/ssd/project/roboticsystem2/ky26063/bench2dex_stuff/.cache/huggingface/hub"
export HF_HOME="/inspire/ssd/project/roboticsystem2/ky26063/bench2dex_stuff/.cache/huggingface"
export PYTHONPATH="${REPO_ROOT}/policy/GR00T_XE/src:${REPO_ROOT}/policy/GR00T_XE:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=3
export NO_ALBUMENTATIONS_UPDATE=1

# =========================================================================
# Output dir -- "gr00t_xe_full", distinct from finetune.sh's "gr00t_xe_ft" so a
# single-stage run can never overwrite (or be mistaken for) a pretrain-based one.
# The eval script names its run dir after `basename "$M"`, so this tag is what
# shows up under output/end_eval/.
# =========================================================================
TAG_SUFFIX="${TAG:+_${TAG}}"
OUTPUT_DIR="${CKPT_BASE}/${TASK_NUM}/${ROBOT_KEY:-default}/gr00t_xe_full${TAG_SUFFIX}"

# =========================================================================
# Build command
# =========================================================================
CMD=(
    python policy/GR00T_XE/finetune.py
    --dataset-path "$DATASET_DIR"
    --init-from-base
    --base-model-path "$BASE_MODEL_PATH"
    --output-dir "$OUTPUT_DIR"
    --max-steps "$MAX_STEPS"
    --batch-size "$BATCH"
    --num-gpus "$NUM_GPUS"
    --save-steps "$SAVE_STEPS"
    --dataloader-num-workers "$WORKERS"
    --seed "$SEED"
    --learning-rate "$LR"
    --hdf5-native
    --hdf5-truncate-at-homing
    --hdf5-use-active-dof
    --hdf5-prompt "$PROMPT"
    --max-state-dim 64
    --max-action-dim 64
    --report-to tensorboard
    --video-backend torchvision_av
)
if [[ -n "$ROBOT_KEY" ]]; then
    CMD+=(--hdf5-robot-key "$ROBOT_KEY")
fi
if [[ -n "$STATE_DIM" ]]; then
    CMD+=(--hdf5-state-dim "$STATE_DIM" --hdf5-action-dim "$ACTION_DIM")
fi
if [[ "$RESUME" == "true" ]]; then
    CMD+=(--resume)
fi

# =========================================================================
# Summary
# =========================================================================
cat <<EOF

  ╔══════════════════════════════════════════════════════════╗
  ║  GR00T XE FULL (single-stage) Finetune — ${TASK_NAME}
  ╠══════════════════════════════════════════════════════════╣
  ║  dataset     ${DATASET_DIR}
  ║  base model  ${BASE_MODEL_PATH}   (action_dim=32 → rebuilt at 64)
  ║  output      ${OUTPUT_DIR}
  ║  robot_key   ${ROBOT_KEY:-auto}   dim=${STATE_DIM:-?}
  ║  GPU(s)      ${GPU} (${NUM_GPUS}x)   seed=${SEED}
  ║  batch       ${BATCH}   steps=${MAX_STEPS}   save_every=${SAVE_STEPS}
  ║  lr          ${LR}   workers=${WORKERS}
  ║  pretrain    NONE (this is the point of the run)
  ║  resume      ${RESUME}
  ╚══════════════════════════════════════════════════════════╝

EOF

echo "[FullFinetune] Language prompt: ${PROMPT}"
echo "[FullFinetune]   (must match the eval's task['description'])"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] $(printf '%q ' "${CMD[@]}")"
    exit 0
fi

# =========================================================================
# Save invocation record
# =========================================================================
mkdir -p "$OUTPUT_DIR"
cat > "${OUTPUT_DIR}/full_finetune_params.txt" <<EOF
GR00T XE FULL Finetune (single-stage, from base) — ${TASK_NAME}
=============================================================
started:      $(date '+%Y-%m-%d %H:%M:%S')
task_num:     ${TASK_NUM}
task_name:    ${TASK_NAME}
dataset:      ${DATASET_DIR}
init_from:    BASE MODEL ${BASE_MODEL_PATH}  (no XE pretrain)
output:       ${OUTPUT_DIR}
robot_key:    ${ROBOT_KEY:-auto}
state_dim:    ${STATE_DIM:-?}
action_dim:   ${ACTION_DIM:-?}
GPU:          ${GPU} (${NUM_GPUS}x)
batch_size:   ${BATCH}
max_steps:    ${MAX_STEPS}
save_steps:   ${SAVE_STEPS}
learning_rate:${LR}
workers:      ${WORKERS}
seed:         ${SEED}
resume:       ${RESUME}
tag:          ${TAG:-<none>}
prompt:       ${PROMPT}

command:
  ${CMD[*]}
EOF
echo "[FullFinetune] Record saved: ${OUTPUT_DIR}/full_finetune_params.txt"

# =========================================================================
# Launch — write to script file to avoid tmux send-keys truncation
# =========================================================================
LOG_FILE="${OUTPUT_DIR}/full_finetune_$(date +%m%d_%H%M).log"
_CMD_FILE="${OUTPUT_DIR}/_run_cmd.sh"
# Use printf %q to properly escape each argument (handles spaces in prompt)
_cmd_str=$(printf '%q ' "${CMD[@]}")
cat > "$_CMD_FILE" << CMDPART
#!/bin/bash
cd '${REPO_ROOT}'
export CUDA_VISIBLE_DEVICES='${GPU}'
export HF_HUB_CACHE='${HF_HUB_CACHE}'
export HF_HOME='${HF_HOME}'
export PYTHONPATH='${PYTHONPATH}'
export TF_CPP_MIN_LOG_LEVEL=3
export NO_ALBUMENTATIONS_UPDATE=1
export NCCL_P2P_DISABLE=1
${_cmd_str} 2>&1 | tee -a '${LOG_FILE}'
CMDPART
chmod +x "$_CMD_FILE"
echo "[FullFinetune] Log: ${LOG_FILE}"
echo "[FullFinetune] Command script: ${_CMD_FILE}"

if [[ "$USE_TMUX" == "true" ]]; then
    SESSION="gr00t-xe-full-${TASK_NUM}"
    tmux set -g mouse on 2>/dev/null || true
    tmux new-session -d -s "$SESSION" 2>/dev/null || {
        SESSION="gr00t-xe-full-${TASK_NUM}-$$"
        tmux new-session -d -s "$SESSION"
    }
    tmux send-keys -t "$SESSION" "bash '${_CMD_FILE}'" C-m
    echo "[FullFinetune] tmux session: $SESSION  (attach: tmux attach -t $SESSION)"
    echo "[FullFinetune] Log file: ${LOG_FILE}"
else
    exec bash "${_CMD_FILE}"
fi
