#!/bin/bash
# auto_eval.sh — Run generalization profiles sequentially.
# Each profile: 50 episodes split into two 25-ep batches.
#
# Usage:
#   bash script/auto_eval.sh --task 51_toilet_lid_cleaner_pour
#   bash script/auto_eval.sh --task 09_cleaner_moisturizer_box_loading --profiles none
#   bash script/auto_eval.sh --task 06_fruit_bowl_loading --robot multi_ur5_rh56dfx_with_flange --gui

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────
TASK=""
ROBOT=""
CKPT_DIR=""
PROFILES="none,cov_only,inv_only,inv_cov"
POLICY_TYPE="ACT"
HEADLESS=true
BATCH=25
TOTAL_EP=50
BASE_SEED=100000000
EPISODE_STEPS="${EPISODE_STEPS:-}"
MAX_STEPS="${MAX_STEPS:-}"
RECORD_SUBDIR=""
ANCHOR_DIR=""
USE_ACTIVE_DOF=""
ACTIVE_DOF_ARG=""

# ── Parse CLI ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)       TASK="$2"; shift 2 ;;
        --task=*)     TASK="${1#*=}"; shift ;;
        --robot)      ROBOT="$2"; shift 2 ;;
        --robot=*)    ROBOT="${1#*=}"; shift ;;
        --profiles)   PROFILES="$2"; shift 2 ;;
        --profiles=*) PROFILES="${1#*=}"; shift ;;
        --ckpt-dir)   CKPT_DIR="$2"; shift 2 ;;
        --ckpt-dir=*) CKPT_DIR="${1#*=}"; shift ;;
        --policy-type)   POLICY_TYPE="$2"; shift 2 ;;
        --policy-type=*) POLICY_TYPE="${1#*=}"; shift ;;
        --gui)        HEADLESS=false; shift ;;
        --record-subdir)   RECORD_SUBDIR="$2"; shift 2 ;;
        --record-subdir=*) RECORD_SUBDIR="${1#*=}"; shift ;;
        --anchor-dir)   ANCHOR_DIR="$2"; shift 2 ;;
        --anchor-dir=*) ANCHOR_DIR="${1#*=}"; shift ;;
        --use-active-dof)   USE_ACTIVE_DOF="$2"; ACTIVE_DOF_ARG="--use-active-dof $2"; shift 2 ;;
        --use-active-dof=*) USE_ACTIVE_DOF="${1#*=}"; ACTIVE_DOF_ARG="--use-active-dof ${1#*=}"; shift ;;
        -h|--help)
            echo "Usage: bash script/auto_eval.sh --task <TASK> [--profiles p1,p2,...] [--robot <R>] [--ckpt-dir <D>] [--policy-type ACT|DP] [--gui]"
            exit 0 ;;
        *)
            echo "[ERROR] Unknown option: $1"
            exit 1 ;;
    esac
done

if [[ -z "$TASK" ]]; then
    echo "[ERROR] --task is required."
    echo "Usage: bash script/auto_eval.sh --task <TASK> [--profiles p1,p2,...] [--robot <R>] [--ckpt-dir <D>] [--policy-type ACT|DP] [--gui]"
    exit 1
fi

TASK_NUM="${TASK:0:2}"

# ── Auto-detect robot ─────────────────────────────────────────────────
if [[ -z "$ROBOT" ]]; then
    CKPT_BASE="../policy_ckpt/${TASK_NUM}"
    if [[ -d "$CKPT_BASE" ]]; then
        ROBOT=$(ls -1 "$CKPT_BASE" 2>/dev/null | head -1)
        [[ -n "$ROBOT" ]] && echo "[auto_eval] robot: $ROBOT (auto-detected)"
    fi
    if [[ -z "$ROBOT" ]]; then
        echo "[ERROR] Could not auto-detect robot. Specify --robot."
        exit 1
    fi
fi

# ── Paths ─────────────────────────────────────────────────────────────
TASK_YAML="scenes/${TASK}.yaml"
if [[ -z "$CKPT_DIR" ]]; then
    CKPT_DIR="../policy_ckpt/${TASK_NUM}/${ROBOT}/act_active"
fi
if [[ -z "$ANCHOR_DIR" ]]; then
    ANCHOR_DIR="../teleopdata/dataset/${TASK}/replay-generalization"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
export DISPLAY="${DISPLAY:-:0}"

IFS=',' read -ra PROFILE_ARRAY <<< "$PROFILES"

# ── Episode budget ────────────────────────────────────────────────────
source script/eval_budget.sh
eval_budget_compute "$TASK" "$EPISODE_STEPS" "$MAX_STEPS" || exit 1
EPISODE_STEPS="$EVAL_EPISODE_STEPS"
eval_budget_log

# ── Headless flag ─────────────────────────────────────────────────────
HEADLESS_FLAG=""
$HEADLESS && HEADLESS_FLAG="--headless"

# ── Helpers ───────────────────────────────────────────────────────────
kill_gpu_python() {
    local gpu_pids
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    if [ -n "$gpu_pids" ]; then
        echo "  [gpu] killing stuck processes: $gpu_pids"
        for pid in $gpu_pids; do
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 5
    fi
}

merge_batch_recordings() {
    local src="$1" dst="$2"
    local count=0
    for sub in success failure; do
        local src_sub="$src/$sub"
        local dst_sub="$dst/$sub"
        [ -d "$src_sub" ] || continue
        mkdir -p "$dst_sub"
        shopt -s nullglob
        for f in "$src_sub"/*.hdf5; do
            mv "$f" "$dst_sub/"
            ((count++)) || true
        done
        shopt -u nullglob
    done
    [ "$count" -gt 0 ] && echo "    merged $count recordings (success+failure)"
}

run_batch() {
    local profile="$1" start_ep="$2" num_ep="$3" tmp_dir="$4" label="$5"
    echo "  $label (start=$start_ep n=$num_ep)"
    rm -rf "$tmp_dir"
    mkdir -p "$tmp_dir"
    bash policy/ACT/eval_double_env.sh "$TASK" "$CKPT_DIR" \
        --generalization-profile "$profile" \
        --anchor-dir "$ANCHOR_DIR" \
        --record-dir "$tmp_dir" \
        --start-episode "$start_ep" \
        --num-episodes "$num_ep" \
        --seed "$BASE_SEED" \
        $HEADLESS_FLAG \
        $ACTIVE_DOF_ARG \
        || echo "  [WARN] eval_double_env.sh exited with code $? (profile=$profile $label)"
}

# ── Validate ──────────────────────────────────────────────────────────
echo "============================================================"
echo "auto_eval  task=$TASK(num=$TASK_NUM)  robot=$ROBOT  policy=$POLICY_TYPE"
echo "  profiles=${PROFILE_ARRAY[*]}  total_ep=$TOTAL_EP  batch=$BATCH  seed=$BASE_SEED"
echo "  ckpt=$CKPT_DIR  headless=$HEADLESS  record_all=true  anchor_dir=$ANCHOR_DIR"
echo "============================================================"

if [[ ! -f "$TASK_YAML" ]]; then
    echo "[ERROR] Task YAML not found: $TASK_YAML"
    exit 1
fi
if [[ ! -d "$CKPT_DIR" ]]; then
    echo "[ERROR] Checkpoint dir not found: $CKPT_DIR"
    exit 1
fi
if [[ ! -d "$ANCHOR_DIR" ]]; then
    echo "[WARN] Anchor dir not found: $ANCHOR_DIR"
fi

# ── Main ──────────────────────────────────────────────────────────────
START_TIME=$(date +%s)

for PROFILE in "${PROFILE_ARRAY[@]}"; do
    PROFILE=$(echo "$PROFILE" | xargs)
    PROFILE_START=$(date +%s)
    echo ""
    echo "=== $PROFILE ==="

    FINAL_DIR="$PROJECT_DIR/outputs/inference_recordings/${POLICY_TYPE}/${TASK_NUM}${RECORD_SUBDIR}/${PROFILE}"
    rm -rf "$FINAL_DIR"
    mkdir -p "$FINAL_DIR"

    # Batch 1: 0 .. BATCH-1
    TMP1="$PROJECT_DIR/outputs/inference_recordings/${POLICY_TYPE}/${TASK_NUM}${RECORD_SUBDIR}/tmp_${PROFILE}_1"
    run_batch "$PROFILE" 0 "$BATCH" "$TMP1" "batch1 (0-$((BATCH-1)))"
    merge_batch_recordings "$TMP1" "$FINAL_DIR"
    rm -rf "$TMP1"
    echo "  batch1 done: $(find "$FINAL_DIR" -name "*.hdf5" 2>/dev/null | wc -l) episodes"

    kill_gpu_python
    sleep 10

    # Batch 2: BATCH .. TOTAL_EP-1
    TMP2="$PROJECT_DIR/outputs/inference_recordings/${POLICY_TYPE}/${TASK_NUM}${RECORD_SUBDIR}/tmp_${PROFILE}_2"
    run_batch "$PROFILE" $((BATCH + 1)) "$BATCH" "$TMP2" "batch2 ($BATCH-$((TOTAL_EP-1)))"
    merge_batch_recordings "$TMP2" "$FINAL_DIR"
    rm -rf "$TMP2"

    N_TOTAL=$(find "$FINAL_DIR" -name "*.hdf5" 2>/dev/null | wc -l)
    echo "  $PROFILE done: $N_TOTAL episodes ($(( $(date +%s) - PROFILE_START ))s)"

    kill_gpu_python
    sleep 5
done

echo ""
echo "=== all done ($(( $(date +%s) - START_TIME ))s) ==="
for PROFILE in "${PROFILE_ARRAY[@]}"; do
    PROFILE=$(echo "$PROFILE" | xargs)
    N=$(find "$PROJECT_DIR/outputs/inference_recordings/${POLICY_TYPE}/${TASK_NUM}/${PROFILE}" -name "*.hdf5" 2>/dev/null | wc -l)
    echo "  $PROFILE: $N"
done
