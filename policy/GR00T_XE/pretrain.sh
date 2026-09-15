#!/bin/bash
# Cross-Embodiment GR00T XE Pretrain Launcher
# ===========================================
#
# Loads ALL available task datasets and trains a unified policy head.
# VLM stays frozen; only DiT + projector are trained.
#
# Usage:
#   bash policy/GR00T_XE/pretrain.sh [FLAGS]
#
# Flags:
#   --gpu IDS         CUDA_VISIBLE_DEVICES.              Default: 0
#   --lr LR            Peak learning rate.                Default: 1e-4
#   --steps N          Max training steps.                Default: 520000
#   --batch N          Per-GPU batch size.                Default: 64
#   --save-steps N     Checkpoint save interval.          Default: 2000
#   --workers N        DataLoader worker count.           Default: 12
#   --seed N           Random seed.                       Default: 42
#   --tag SUFFIX       Append suffix to output dir.
#   --prompt TEXT      Global language prompt. Default: per-task scene YAML
#                      `description` (match the eval — do NOT leave it constant).
#   --no-scene-prompt  Disable per-task prompts (legacy constant "perform task").
#   --resume           Resume from latest checkpoint.
#   --no-tmux          Run in foreground.
#   --dry-run          Print the command without running.
#
# Examples:
#   bash policy/GR00T_XE/pretrain.sh                                    # all defaults
#   bash policy/GR00T_XE/pretrain.sh --gpu 0,1,2,3,4,5,6,7 --batch 64 # 8-GPU
#   bash policy/GR00T_XE/pretrain.sh --lr 2e-4 --steps 100000 --tag v1  # with tag
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# =========================================================================
# Defaults
# =========================================================================
GPU="0"
LR="1e-4"
MAX_STEPS=20000
BATCH=64
SAVE_STEPS=10000
WORKERS=12
SEED=42
TAG=""
RESUME=false
MILESTONE_STEPS=15000
USE_TMUX=true
DRY_RUN=false
PROMPT_OVERRIDE=""
SCENE_PROMPT=true

# =========================================================================
# Parse arguments
# =========================================================================
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
        --workers)      WORKERS="$2";       shift 2 ;;
        --workers=*)    WORKERS="${1#*=}";  shift ;;
        --seed)         SEED="$2";          shift 2 ;;
        --seed=*)       SEED="${1#*=}";     shift ;;
        --tag)          TAG="$2";           shift 2 ;;
        --tag=*)        TAG="${1#*=}";      shift ;;
        --prompt)       PROMPT_OVERRIDE="$2"; shift 2 ;;
        --prompt=*)     PROMPT_OVERRIDE="${1#*=}"; shift ;;
        --no-scene-prompt) SCENE_PROMPT=false; shift ;;
        --resume)       RESUME=true;        shift ;;
        --no-resume)    RESUME=false;       shift ;;
        --milestone-steps)   MILESTONE_STEPS="$2"; shift 2 ;;
        --milestone-steps=*) MILESTONE_STEPS="${1#*=}"; shift ;;
        --no-tmux)      USE_TMUX=false;     shift ;;
        --dry-run)      DRY_RUN=true;       shift ;;
        -h|--help)
            sed -n '2,27p' "$0" >&2
            exit 0
            ;;
        --*)            echo "ERROR: unknown flag $1" >&2; exit 1 ;;
        *)              echo "ERROR: unexpected positional arg: $1" >&2; exit 1 ;;
    esac
done

# =========================================================================
# Conda env
# =========================================================================
source /inspire/hdd/project/roboticsystem2/ky26063/bench2dex_stuff/miniconda3/bin/activate groot

# =========================================================================
# Auto-detect ALL task datasets
# =========================================================================
DATASET_BASE="/inspire/hdd2/project/roboticsystem2/public/DEX2BENCH/zdj/teleopdata/dataset"
# maxdepth=2 only — skip robot-test subdirs
TASK_DIRS=$(find "$DATASET_BASE" -maxdepth 2 -type d -name "replay-generalization" 2>/dev/null | sort -u || true)
# Exclude 86 (deprecated short version of 73)
TASK_DIRS=$(echo "$TASK_DIRS" | grep -v "86_short_jigsaw_puzzle" || true)

if [[ -z "$TASK_DIRS" ]]; then
    echo "ERROR: no replay-generalization dirs found under $DATASET_BASE" >&2
    exit 1
fi

TASK_COUNT=$(echo "$TASK_DIRS" | wc -l)
echo "[Pretrain] Found $TASK_COUNT task datasets"

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
export HF_HUB_CACHE="/inspire/ssd/project/roboticsystem2/ky26063/bench2dex_stuff/.cache/huggingface/hub"
export HF_HOME="/inspire/ssd/project/roboticsystem2/ky26063/bench2dex_stuff/.cache/huggingface"
export PYTHONPATH="${REPO_ROOT}/policy/GR00T_XE/src:${REPO_ROOT}/policy/GR00T_XE:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=3
export NO_ALBUMENTATIONS_UPDATE=1

# =========================================================================
# Output dir
# =========================================================================
CKPT_BASE="/inspire/hdd2/project/roboticsystem2/public/DEX2BENCH/zdj/policy_ckpt"
TAG_SUFFIX="${TAG:+_${TAG}}"
OUTPUT_DIR="${CKPT_BASE}/gr00t_xe_pretrain${TAG_SUFFIX}"

# =========================================================================
# Build command
# =========================================================================
# ``--dataset-path`` is a ``List[str]`` field, which tyro exposes with
# nargs="+" — repeating the flag makes argparse keep only the LAST occurrence.
# That silently collapsed the 26-task pretrain to 80_gaming_desk_setup alone
# (see the 2026-09-06 run: every rank printed ``dataset_path: ['.../80_...']``).
# All paths must therefore follow ONE flag.
DATASET_ARGS=(--dataset-path)
CMD_STR="python policy/GR00T_XE/pretrain.py --dataset-path"
while IFS= read -r d; do
    [[ -z "$d" ]] && continue
    DATASET_ARGS+=("$d")
    CMD_STR+=" '${d}'"
done <<< "$TASK_DIRS"

# Hard guard: pretrain.py aborts if its parsed dataset list does not match this.
export XE_EXPECTED_DATASET_COUNT="${TASK_COUNT}"
CMD_STR+=" --output-dir '${OUTPUT_DIR}'"
CMD_STR+=" --max-steps ${MAX_STEPS}"
CMD_STR+=" --batch-size ${BATCH}"
CMD_STR+=" --num-gpus ${NUM_GPUS}"
CMD_STR+=" --save-steps ${SAVE_STEPS}"
CMD_STR+=" --dataloader-num-workers ${WORKERS}"
CMD_STR+=" --seed ${SEED}"
CMD_STR+=" --learning-rate ${LR}"
CMD_STR+=" --hdf5-native"
# Language conditioning: pretrain used to run with NO prompt at all, so all
# 1.69M samples carried the constant "perform task" (gr00t_hdf5_dataset.py:612)
# while finetune/eval use scenes/<task>.yaml's `description` — the model's
# language channel therefore learned nothing.  Default is now per-task scene
# descriptions; --prompt forces one string, --no-scene-prompt restores the old
# (useless) constant.
if [[ -n "$PROMPT_OVERRIDE" ]]; then
    CMD_STR+=" --hdf5-prompt $(printf '%q' "$PROMPT_OVERRIDE")"
    PROMPT_DESC="global: ${PROMPT_OVERRIDE}"
elif [[ "$SCENE_PROMPT" == "true" ]]; then
    CMD_STR+=" --hdf5-prompt-from-scene"
    PROMPT_DESC="per-task scene YAML description (${REPO_ROOT}/scenes)"
else
    CMD_STR+=" --no-hdf5-prompt-from-scene"
    PROMPT_DESC="DISABLED — constant 'perform task' (language channel is dead)"
fi
CMD_STR+=" --hdf5-truncate-at-homing"
CMD_STR+=" --hdf5-use-active-dof"
CMD_STR+=" --max-state-dim 64"
CMD_STR+=" --max-action-dim 64"
CMD_STR+=" --report-to tensorboard"
CMD_STR+=" --video-backend torchvision_av"
CMD_STR+=" --milestone-steps ${MILESTONE_STEPS}"

if [[ "$RESUME" == "true" ]]; then
    CMD_STR+=" --resume"
fi

# Use local base model cache
BASE_MODEL_PATH="/inspire/hdd2/project/roboticsystem2/ky26063/bench2dex_stuff/zyd/GR00T-N1.5-3B"
if [[ -d "$BASE_MODEL_PATH" ]]; then
    CMD_STR+=" --base-model-path '${BASE_MODEL_PATH}'"
fi

# =========================================================================
# Summary
# =========================================================================
cat <<EOF

  ╔══════════════════════════════════════════════════════════╗
  ║  GR00T XE Cross-Embodiment Pretrain
  ╠══════════════════════════════════════════════════════════╣
  ║  tasks       ${TASK_COUNT}
  ║  output      ${OUTPUT_DIR}
  ║  GPU(s)      ${GPU} (${NUM_GPUS}x)   seed=${SEED}
  ║  batch       ${BATCH}   steps=${MAX_STEPS}   save_every=${SAVE_STEPS}
  ║  lr          ${LR}   workers=${WORKERS}
  ║  resume      ${RESUME}
  ║  prompt      ${PROMPT_DESC}
  ╚══════════════════════════════════════════════════════════╝

EOF

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] ${CMD_STR}"
    exit 0
fi

# =========================================================================
# Save invocation record
# =========================================================================
mkdir -p "$OUTPUT_DIR"
cat > "${OUTPUT_DIR}/pretrain_params.txt" <<EOF
GR00T XE Pretrain — Cross-Embodiment
====================================
started:      $(date '+%Y-%m-%d %H:%M:%S')
tasks:        ${TASK_COUNT}
output:       ${OUTPUT_DIR}
GPU:          ${GPU} (${NUM_GPUS}x)
batch_size:   ${BATCH}
max_steps:    ${MAX_STEPS}
save_steps:   ${SAVE_STEPS}
learning_rate:${LR}
workers:      ${WORKERS}
seed:         ${SEED}
resume:       ${RESUME}
tag:          ${TAG:-<none>}
prompt:       ${PROMPT_DESC}

command:
  ${CMD_STR}
EOF
echo "[Pretrain] Record saved: ${OUTPUT_DIR}/pretrain_params.txt"

# =========================================================================
# Launch
# =========================================================================
LOG_FILE="${OUTPUT_DIR}/pretrain_$(date +%m%d_%H%M).log"

# Write command to a temp script file so tmux send-keys doesn't truncate
_CMD_FILE="${OUTPUT_DIR}/_run_cmd.sh"
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
export XE_EXPECTED_DATASET_COUNT='${TASK_COUNT}'
${CMD_STR} 2>&1 | tee -a '${LOG_FILE}'
CMDPART
chmod +x "$_CMD_FILE"
echo "[Pretrain] Log: ${LOG_FILE}"
echo "[Pretrain] Command script: ${_CMD_FILE}"

if [[ "$USE_TMUX" == "true" ]]; then
    SESSION="gr00t-xe-pretrain"
    tmux set -g mouse on 2>/dev/null || true
    tmux new-session -d -s "$SESSION" 2>/dev/null || {
        SESSION="gr00t-xe-pretrain-$$"
        tmux new-session -d -s "$SESSION"
    }
    tmux send-keys -t "$SESSION" "bash '${_CMD_FILE}'" C-m
    echo "[Pretrain] tmux session: $SESSION  (attach: tmux attach -t $SESSION)"
    echo "[Pretrain] Log file: ${LOG_FILE}"
else
    exec bash "${_CMD_FILE}"
fi