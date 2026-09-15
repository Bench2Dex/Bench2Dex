#!/bin/bash
# Batch-add TacMap tactile data to replay-generalization HDF5 episodes.
#
# For each task number provided, finds the matching replay-generalization
# directory under the dataset root and runs tactile-only replay in-place.
#
# Usage:
#   bash tools/replay/add_tactile_to_replay.sh 06
#   bash tools/replay/add_tactile_to_replay.sh 06 44 60
#   bash tools/replay/add_tactile_to_replay.sh --dry-run 06 44
#   bash tools/replay/add_tactile_to_replay.sh --dataset-root /custom/path 06
#
# Environment:
#   Must be run from a Python environment with Isaac Sim + h5py available.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_DATASET_ROOT="${SCRIPT_DIR}/../teleopdata/dataset"

# ── helpers ────────────────────────────────────────────────────────────────

require_value() {
    if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
        echo "[ERROR] missing value for $1" >&2
        exit 2
    fi
}

usage() {
    cat >&2 <<'EOF'
Usage: bash tools/add_tactile_to_replay.sh [OPTIONS] TASK_NUM [TASK_NUM ...]

  Add TacMap tactile data to every HDF5 episode in the replay-generalization
  directory for each specified task.  Existing data (qpos, objects, cameras,
  meta, etc.) is preserved — only robot/tactile is added.

Options:
  --dataset-root PATH   Base dataset directory.
                        Default: .../teleopdata/dataset
  --dry-run             Print what would be processed without running.
  -h, --help            Show this help.

Examples:
  bash tools/add_tactile_to_replay.sh 06
  bash tools/add_tactile_to_replay.sh 06 44 60
  bash tools/add_tactile_to_replay.sh --dry-run 06 44 60
EOF
}

# ── parse args ─────────────────────────────────────────────────────────────

DATASET_ROOT="$DEFAULT_DATASET_ROOT"
DRY_RUN=false
TASKS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-root)
            require_value "$@"
            DATASET_ROOT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "[ERROR] Unknown option: $1" >&2
            usage
            exit 2
            ;;
        -*)
            echo "[ERROR] Unknown option: $1" >&2
            usage
            exit 2
            ;;
        *)
            TASKS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "[ERROR] At least one task number is required." >&2
    usage
    exit 2
fi

# ── resolve task directories ───────────────────────────────────────────────

TASK_DIRS=()       # indexed array: task_num
TASK_PATHS=()      # indexed array: resolved path

for num in "${TASKS[@]}"; do
    # Glob for <num>_*/replay-generalization
    matches=()
    for d in "$DATASET_ROOT"/${num}_*/replay-generalization; do
        if [[ -d "$d" ]]; then
            matches+=("$d")
        fi
    done

    if [[ ${#matches[@]} -eq 0 ]]; then
        echo "[SKIP] Task ${num}: no replay-generalization/ under $DATASET_ROOT/${num}_*" >&2
        continue
    fi

    if [[ ${#matches[@]} -gt 1 ]]; then
        echo "[WARN] Task ${num}: multiple matches, using first: ${matches[0]}" >&2
    fi

    # Deduplicate: skip if num already in TASK_DIRS
    already=false
    for existing in "${TASK_DIRS[@]}"; do
        if [[ "$existing" == "$num" ]]; then
            already=true
            break
        fi
    done
    if $already; then
        continue
    fi

    TASK_DIRS+=("$num")
    TASK_PATHS+=("${matches[0]}")
done

if [[ ${#TASK_DIRS[@]} -eq 0 ]]; then
    echo "[ERROR] No valid task directories found." >&2
    exit 1
fi

# ── summary ────────────────────────────────────────────────────────────────

echo "============================================================"
echo "Add Tactile to Replay-Generalization"
echo "Dataset root: $DATASET_ROOT"
echo "Tasks (${#TASK_DIRS[@]}): ${TASK_DIRS[*]}"
echo "Dry run: $DRY_RUN"
echo "============================================================"

if $DRY_RUN; then
    echo ""
    total_eps=0
    for i in "${!TASK_DIRS[@]}"; do
        num="${TASK_DIRS[$i]}"
        dir="${TASK_PATHS[$i]}"
        count=$(ls "$dir"/episode_*.hdf5 2>/dev/null | wc -l)
        total_eps=$((total_eps + count))
        printf "  Task %-4s  %4d episodes  %s\n" "$num" "$count" "$dir"
    done
    echo ""
    echo "[dry-run] Would process $total_eps episodes across ${#TASK_DIRS[@]} tasks."
    echo "[dry-run] Remove --dry-run to execute."
    exit 0
fi

# ── process each task ──────────────────────────────────────────────────────

TOTAL=${#TASK_DIRS[@]}
DONE=0
FAILED_TASKS=()
LOG_FILE="${SCRIPT_DIR}/../tactile.txt"

# ── initialize log ─────────────────────────────────────────────────────────

_now() { date '+%Y-%m-%d %H:%M:%S'; }

cat >> "$LOG_FILE" <<EOF

============================================================
Tactile replay run
Started: $(_now)
Tasks (${TOTAL}): ${TASK_DIRS[*]}
Dataset root: $DATASET_ROOT
============================================================

EOF

cd "$SCRIPT_DIR"

for i in "${!TASK_DIRS[@]}"; do
    DONE=$((DONE + 1))
    num="${TASK_DIRS[$i]}"
    dir="${TASK_PATHS[$i]}"
    count=$(ls "$dir"/episode_*.hdf5 2>/dev/null | wc -l)

    echo ""
    echo "============================================================"
    echo "[$DONE/$TOTAL] Task $num  ($count episodes)"
    echo "  dir: $dir"
    echo "============================================================"

    # Mark current task as running in log (visible while task is in progress)
    OK_COUNT=$((DONE - 1 - ${#FAILED_TASKS[@]}))
    echo "$(_now) | task=$num | RUNNING | progress: $DONE/$TOTAL tasks, $OK_COUNT ok, ${#FAILED_TASKS[@]} failed" >> "$LOG_FILE"

    # Capture batch_replay output for parsing while still printing to console
    BATCH_LOG=$(mktemp)
    TASK_RESULT="OK"
    if python tools/replay/batch_replay.py \
        --origin-dir "$dir" \
        --replay-dir "$dir" \
        --tactile-only \
        --headless 2>&1 | tee "$BATCH_LOG"; then
        echo "[$DONE/$TOTAL] Task $num: OK"
    else
        TASK_RESULT="FAILED"
        FAILED_TASKS+=("$num")
        echo "[$DONE/$TOTAL] Task $num: FAILED" >&2
    fi

    # Parse per-episode counts from batch_replay output
    EP_OK=$(grep '^  OK:' "$BATCH_LOG" | awk '{print $2}' || echo "?")
    EP_FAIL=$(grep '^  Failed:' "$BATCH_LOG" | awk '{print $2}' || echo "?")
    EP_SKIP=$(grep '^  Skipped:' "$BATCH_LOG" | awk '{print $2}' || echo "?")
    # Collect failed episode names if any
    EP_FAILED_NAMES=""
    if grep -q 'Failed episodes:' "$BATCH_LOG" 2>/dev/null; then
        EP_FAILED_NAMES=$(sed -n '/Failed episodes:/,$ p' "$BATCH_LOG" | tail -n +2 | sed 's/^  //' | tr '\n' ' ')
        EP_FAILED_NAMES=" | failed_eps: ${EP_FAILED_NAMES% }"
    fi
    rm -f "$BATCH_LOG"

    OK_COUNT=$((DONE - ${#FAILED_TASKS[@]}))
    cat >> "$LOG_FILE" <<EOF
$(_now) | task=$num | $TASK_RESULT | ep_ok=$EP_OK ep_fail=$EP_FAIL ep_skip=$EP_SKIP$EP_FAILED_NAMES | progress: $DONE/$TOTAL tasks, $OK_COUNT ok, ${#FAILED_TASKS[@]} failed
EOF
done

# ── final report ───────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "All done."
echo "  Tasks total:  $TOTAL"
echo "  Tasks ok:     $((TOTAL - ${#FAILED_TASKS[@]}))"
echo "  Tasks failed: ${#FAILED_TASKS[@]}"
if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
    echo "  Failed tasks: ${FAILED_TASKS[*]}"
fi
echo "============================================================"

cat >> "$LOG_FILE" <<EOF

============================================================
Finished: $(_now)
Tasks total:  $TOTAL
Tasks ok:     $((TOTAL - ${#FAILED_TASKS[@]}))
Tasks failed: ${#FAILED_TASKS[@]}
EOF
if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
    echo "Failed tasks: ${FAILED_TASKS[*]}" >> "$LOG_FILE"
    echo "Log: $LOG_FILE" >&2
    exit 1
fi
echo "============================================================" >> "$LOG_FILE"
echo "Log: $LOG_FILE"
