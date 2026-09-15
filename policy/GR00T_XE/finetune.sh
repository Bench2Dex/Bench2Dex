#!/bin/bash
# GR00T XE Single-Task Finetune Launcher
# ======================================
#
# Loads a pretrained GR00T XE checkpoint and finetunes on one task.
#
# Usage:
#   bash policy/GR00T_XE/finetune.sh TASK_NUM [FLAGS]
#
# Flags:
#   --pretrained-ckpt PATH   Pretrained checkpoint dir.    Default: auto-search
#   --gpu IDS                CUDA_VISIBLE_DEVICES.         Default: 0
#   --lr LR                  Peak learning rate.           Default: 1e-4
#   --steps N                Max training steps.           Default: 5000
#   --batch N                Per-GPU batch size.           Default: 64
#   --save-steps N           Checkpoint save interval.     Default: 500
#   --workers N              DataLoader worker count.      Default: 12
#   --seed N                 Random seed.                  Default: 42
#   --tag SUFFIX             Append suffix to output dir.
#   --dataset PATH           Override replay HDF5 dir.
#   --prompt TEXT            Language prompt.   Default: scenes/<task>.yaml description
#   --resume                 Resume from latest checkpoint.
#   --no-tmux                Run in foreground.
#   --dry-run                Print the command without running.
#
# Examples:
#   bash policy/GR00T_XE/finetune.sh 73
#   bash policy/GR00T_XE/finetune.sh 73 --pretrained-ckpt /ckpt/gr00t_xe_pretrain/checkpoint-520000
#   bash policy/GR00T_XE/finetune.sh 73 --gpu 0,1 --batch 32 --lr 5e-5 --tag v2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# =========================================================================
# Defaults (finetune: low LR, few steps — model already pretrained)
# =========================================================================
GPU="0"
LR="5e-5"
MAX_STEPS=3000
BATCH=64
SAVE_STEPS=1000
WORKERS=12
SEED=42
TAG=""
RESUME=false
USE_TMUX=true
DRY_RUN=false
PRETRAINED_CKPT=""
DATASET_OVERRIDE=""
PROMPT_OVERRIDE=""

# =========================================================================
# Parse arguments
# =========================================================================
TASK_NUM=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrained-ckpt)   PRETRAINED_CKPT="$2"; shift 2 ;;
        --pretrained-ckpt=*) PRETRAINED_CKPT="${1#*=}"; shift ;;
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
            sed -n '2,30p' "$0" >&2
            exit 0
            ;;
        --*)            echo "ERROR: unknown flag $1" >&2; exit 1 ;;
        *)
            if [[ -z "$TASK_NUM" ]]; then TASK_NUM="$1"
            else echo "ERROR: too many positional args" >&2; exit 1; fi
            shift ;;
    esac
done

if [[ -z "$TASK_NUM" ]]; then
    echo "ERROR: TASK_NUM required.  Usage: bash finetune.sh TASK_NUM [FLAGS]" >&2
    exit 1
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
# Auto-detect pretrained checkpoint
# =========================================================================
CKPT_BASE="/inspire/hdd2/project/roboticsystem2/public/DEX2BENCH/zdj/policy_ckpt"

# A directory is usable as ``--pretrained-ckpt`` when it has a config plus at
# least one weight shard (``save_pretrained`` writes the run root only when
# training ends, so an in-flight run has checkpoints but no root weights).
_ckpt_has_weights() {
    [[ -f "$1/config.json" ]] && compgen -G "$1/model*.safetensors*" > /dev/null
}

if [[ -z "$PRETRAINED_CKPT" ]]; then
    # Auto-detect.  The canonical name ``gr00t_xe_pretrain`` always wins — if it
    # exists at all we never silently fall through to a *differently named*
    # checkpoint: after a re-pretrain the old model sits next to the new one as
    # ``gr00t_xe_pretrain_no_lang`` / ``*_old``, and picking that one would train
    # the whole finetune on the wrong (language-dead) base without saying so.
    CANONICAL="${CKPT_BASE}/gr00t_xe_pretrain"
    if [[ -d "$CANONICAL" ]]; then
        if _ckpt_has_weights "$CANONICAL"; then
            PRETRAINED_CKPT="$CANONICAL"
        else
            _last=$(find "$CANONICAL" -maxdepth 1 -type d -name 'checkpoint-[0-9]*' \
                        -printf '%f\n' 2>/dev/null | sed 's/checkpoint-//' | sort -n | tail -1)
            if [[ -n "$_last" ]] && _ckpt_has_weights "${CANONICAL}/checkpoint-${_last}"; then
                PRETRAINED_CKPT="${CANONICAL}/checkpoint-${_last}"
                echo "[Finetune] WARNING: ${CANONICAL} has no root weights yet (pretrain still running?)" >&2
                echo "[Finetune]          using the latest checkpoint instead: ${PRETRAINED_CKPT}" >&2
            else
                echo "ERROR: ${CANONICAL} exists but holds no usable weights" >&2
                echo "       (no root model, no checkpoint-<step> with shards)." >&2
                echo "       If the pretrain just started, wait for it to finish;" >&2
                echo "       otherwise pass --pretrained-ckpt PATH explicitly." >&2
                exit 1
            fi
        fi
    else
        # Canonical dir gone (e.g. renamed by hand): fall back to the newest
        # same-family run, but say which one, because it may be an archived
        # model that is no longer what the eval/finetune corpus expects.
        while IFS= read -r _d; do
            if [[ -n "$_d" ]] && _ckpt_has_weights "$_d"; then
                PRETRAINED_CKPT="$_d"
                break
            fi
        done < <(find "$CKPT_BASE" -maxdepth 1 -type d -name 'gr00t_xe_pretrain*' ! -name '*_old' \
                     -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2-)
        if [[ -z "$PRETRAINED_CKPT" ]]; then
            echo "ERROR: no 'gr00t_xe_pretrain*' checkpoint with weights under ${CKPT_BASE}." >&2
            echo "       Use --pretrained-ckpt PATH" >&2
            exit 1
        fi
        echo "[Finetune] WARNING: ${CANONICAL} does not exist; using ${PRETRAINED_CKPT}" >&2
    fi
    echo "[Finetune] Auto-detected pretrained ckpt: ${PRETRAINED_CKPT} (override with --pretrained-ckpt)"
fi
echo "[Finetune] Pretrained ckpt: ${PRETRAINED_CKPT}"

# =========================================================================
# Prompt from scene YAML
# =========================================================================
# The prompt MUST equal what the eval feeds the model: run_policy.py:2184 passes
# task["description"] (= yaml.safe_load(scenes/<task>.yaml)["description"]) down
# to policy_sessions, which hands it to the policy as `instruction`
# (script/policy_sessions.py:_prepare_observation).  The old implementation used
# `grep -m1 '^description:'`, which returns only the FIRST PHYSICAL LINE of the
# YAML scalar — for a description that wraps (34_fridge_wine_interhand_pour,
# 61_medicine_shoebox_pack) the model was finetuned on a half-sentence while the
# eval asked for the whole sentence.  Parse the YAML instead of grepping it.
_scene_prompt() {
    if [[ -n "$PROMPT_OVERRIDE" ]]; then
        printf '%s' "$PROMPT_OVERRIDE"
        return 0
    fi
    local yaml="${REPO_ROOT}/scenes/${TASK_NAME}.yaml"
    if [[ ! -f "$yaml" ]]; then
        echo "ERROR: scene YAML not found: $yaml" >&2
        echo "       The finetune prompt must match the eval's task description." >&2
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
# Output dir
# =========================================================================
TAG_SUFFIX="${TAG:+_${TAG}}"
OUTPUT_DIR="${CKPT_BASE}/${TASK_NUM}/${ROBOT_KEY:-default}/gr00t_xe_ft${TAG_SUFFIX}"

# =========================================================================
# Build command
# =========================================================================
CMD=(
    python policy/GR00T_XE/finetune.py
    --dataset-path "$DATASET_DIR"
    --pretrained-ckpt "$PRETRAINED_CKPT"
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

# Use local base model cache
BASE_MODEL_PATH="/inspire/hdd2/project/roboticsystem2/ky26063/bench2dex_stuff/zyd/GR00T-N1.5-3B"
if [[ -d "$BASE_MODEL_PATH" ]]; then
    CMD+=(--base-model-path "$BASE_MODEL_PATH")
fi

# =========================================================================
# Summary
# =========================================================================
cat <<EOF

  ╔══════════════════════════════════════════════════════════╗
  ║  GR00T XE Single-Task Finetune — ${TASK_NAME}
  ╠══════════════════════════════════════════════════════════╣
  ║  dataset     ${DATASET_DIR}
  ║  pretrained  ${PRETRAINED_CKPT}
  ║  output      ${OUTPUT_DIR}
  ║  robot_key   ${ROBOT_KEY:-auto}   dim=${STATE_DIM:-?}
  ║  GPU(s)      ${GPU} (${NUM_GPUS}x)   seed=${SEED}
  ║  batch       ${BATCH}   steps=${MAX_STEPS}   save_every=${SAVE_STEPS}
  ║  lr          ${LR}   workers=${WORKERS}
  ║  resume      ${RESUME}
  ╚══════════════════════════════════════════════════════════╝

EOF

echo "[Finetune] Language prompt: ${PROMPT}"
echo "[Finetune]   (must match the eval's task['description'] — see run_policy.py:2184)"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] $(printf '%q ' "${CMD[@]}")"
    exit 0
fi

# =========================================================================
# Save invocation record
# =========================================================================
mkdir -p "$OUTPUT_DIR"
cat > "${OUTPUT_DIR}/finetune_params.txt" <<EOF
GR00T XE Finetune — ${TASK_NAME}
===============================
started:      $(date '+%Y-%m-%d %H:%M:%S')
task_num:     ${TASK_NUM}
task_name:    ${TASK_NAME}
dataset:      ${DATASET_DIR}
pretrained:   ${PRETRAINED_CKPT}
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
echo "[Finetune] Record saved: ${OUTPUT_DIR}/finetune_params.txt"

# =========================================================================
# Launch — write to script file to avoid tmux send-keys truncation
# =========================================================================
LOG_FILE="${OUTPUT_DIR}/finetune_$(date +%m%d_%H%M).log"
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
echo "[Finetune] Log: ${LOG_FILE}"
echo "[Finetune] Command script: ${_CMD_FILE}"

if [[ "$USE_TMUX" == "true" ]]; then
    SESSION="gr00t-xe-ft-${TASK_NUM}"
    tmux set -g mouse on 2>/dev/null || true
    tmux new-session -d -s "$SESSION" 2>/dev/null || {
        SESSION="gr00t-xe-ft-${TASK_NUM}-$$"
        tmux new-session -d -s "$SESSION"
    }
    tmux send-keys -t "$SESSION" "bash '${_CMD_FILE}'" C-m
    echo "[Finetune] tmux session: $SESSION  (attach: tmux attach -t $SESSION)"
    echo "[Finetune] Log file: ${LOG_FILE}"
else
    exec bash "${_CMD_FILE}"
fi