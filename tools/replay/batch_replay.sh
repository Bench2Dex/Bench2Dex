#!/usr/bin/env bash
# Batch replay HDF5 episodes under SCENE_DIR/origin into SCENE_DIR/replay.
# Defaults to RGB capture unless capture flags are passed.
#
# Edit SCENE_DIR below to replay a different task, then run:
#   bash tools/replay/batch_replay.sh
#
# Set OVERWRITE=0 to skip already-replayed episodes:
#   OVERWRITE=0 bash tools/replay/batch_replay.sh
#
# Capture flags are forwarded to batch_replay.py and can be combined:
#   bash tools/replay/batch_replay.sh --enable-depth
#   bash tools/replay/batch_replay.sh --enable-rgb --enable-depth
#   bash tools/replay/batch_replay.sh --enable-tactile
#   bash tools/replay/batch_replay.sh --enable-rgb --enable-depth --enable-tactile
#   bash tools/replay/batch_replay.sh --tactile-only

# python tools/replay/batch_replay.py \
#     --origin-dir ../teleopdata/dataset/06_fruit_bowl_loading/origin-generalization-double \
#     --replay-dir ../teleopdata/dataset/06_fruit_bowl_loading/replay-generalization-double \
#     --enable-rgb \
#     --resample-groups background,table_surface,light,camera \
#     --generalization-split seen \
#     --headless
set -e

# ── Configure here ────────────────────────────────────────────────────────────
SCENE_DIR="outputs/ur5_rh56dfx/scenes/06_fruit_bowl_loading"
# ─────────────────────────────────────────────────────────────────────────────

ORIGIN_DIR="${SCENE_DIR}/origin"
REPLAY_DIR="${SCENE_DIR}/replay"

# Run from dex2bench repo root
cd "$(dirname "$0")/../.."

python tools/replay/batch_replay.py \
    --origin-dir "${ORIGIN_DIR}" \
    --replay-dir  "${REPLAY_DIR}" \
    "$@"
