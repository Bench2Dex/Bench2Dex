#!/usr/bin/env bash
# Batch screenshot HDF5 episodes from a custom 6D camera pose.
# Captures one PNG per episode at frame 0, then moves to the next episode.
#
# Usage:
#   bash tools/replay/batch_screenshot_6d.sh <task_number>
#
# Examples:
#   bash tools/replay/batch_screenshot_6d.sh 44
#   bash tools/replay/batch_screenshot_6d.sh 44 --camera-pos 0.0,-1.36,2.48,50.0,0.0,0.0
#   bash tools/replay/batch_screenshot_6d.sh 43 --width 1920 --height 1080
#
# The script auto-resolves the task directory under:
#   ../teleopdata/dataset/
#
# Screenshots are saved to:
#   ../jcy/pic/<task_name>/
#
# Episodes captured (standard_50 selection):
#   - episode_000000 ~ episode_000024  (no _1 suffix, 25 episodes)
#   - episode_000025_1 ~ episode_000049_1  (with _1 suffix, 25 episodes)
#   Total: 50 episodes
#
# Generalization is RESTORED from each HDF5 (no resampling).
# Original HDF5 files are NEVER modified.
#
# Set OVERWRITE=0 to skip episodes that already have a PNG:
#   OVERWRITE=0 bash tools/replay/batch_screenshot_6d.sh 44

set -e

# ── Directories ──────────────────────────────────────────────────────────────
DATASET_DIR="../teleopdata/dataset"
PIC_DIR="../jcy/pic"

# ── Parse task number ────────────────────────────────────────────────────────
if [ -z "$1" ]; then
    echo "Usage: bash tools/replay/batch_screenshot_6d.sh <task_number> [additional args]"
    echo "Example: bash tools/replay/batch_screenshot_6d.sh 44"
    exit 1
fi

TASK_NUM="$1"
shift  # remove task number from args, remaining "$@" forwarded to Python

# Find task directory by prefix match
TASK_DIR=$(ls -d "${DATASET_DIR}/${TASK_NUM}_"*/ 2>/dev/null | head -1)
if [ -z "${TASK_DIR}" ]; then
    echo "[ERROR] No task directory found matching: ${DATASET_DIR}/${TASK_NUM}_*"
    echo "Available task directories:"
    ls -d "${DATASET_DIR}/"*/ 2>/dev/null | xargs -I{} basename {} | head -20
    exit 1
fi

TASK_NAME=$(basename "${TASK_DIR}")
ORIGIN_DIR="${TASK_DIR}replay-generalization"
OUTPUT_DIR="${PIC_DIR}/${TASK_NAME}"

if [ ! -d "${ORIGIN_DIR}" ]; then
    echo "[ERROR] replay-generalization directory not found: ${ORIGIN_DIR}"
    exit 1
fi

echo "============================================"
echo "[batch_screenshot_6d]"
echo "  Task number: ${TASK_NUM}"
echo "  Task name:   ${TASK_NAME}"
echo "  Origin dir:  ${ORIGIN_DIR}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "============================================"

# Run from dex2bench repo root
cd "$(dirname "$0")/../.."

python tools/replay/batch_screenshot_6d.py \
    --origin-dir "${ORIGIN_DIR}" \
    --output-dir  "${OUTPUT_DIR}" \
    --episode-select standard_50 \
    "$@"
