#!/bin/bash
# Batch replay HDF5 episodes 55–65 with wrist cameras only, RGB only,
# camera view generalization OFF.
#
# Usage:
#   bash tools/batch_replay_wrist.sh
#
# Camera generalization:
#   --enable-generalization is NOT passed, so camera perturbation
#   (position / rotation / distance offsets) is DISABLED.
#   CameraRig receives camera_generalization_sample=None -> all offsets are zero.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="../output/hdf5"
OUTPUT_DIR="../output/hdf5_new_w"

mkdir -p "$OUTPUT_DIR"

# Episode → robot mapping (for logging only; robot_key is read from HDF5 metadata)
declare -A ROBOT_MAP=(
  [55]="multi_iiwa7_with_sharpa"
  [56]="multi_jaka_zu7_dexhand021_with_flange"
  [57]="multi_panda_with_orca"
  [58]="multi_panda_with_allegro"
  [59]="multi_xarm7_with_ability"
  [60]="multi_xarm7_with_leap"
  [61]="multi_ur5_rh5dg2_with_flange"
  [62]="multi_ur5_rh56dfx_with_flange"
  [63]="multi_ur5_shadow_hand_with_flange"
  [64]="multi_ur5_schunk_hand_with_flange"
  [65]="multi_ur5_wuji_with_flange"
)

TOTAL=11
COUNT=0

for ep in $(seq 55 65); do
  COUNT=$((COUNT + 1))
  HDF5_FILE="${INPUT_DIR}/episode_$(printf '%06d' $ep).hdf5"
  if [[ ! -f "$HDF5_FILE" ]]; then
    echo "[SKIP] File not found: $HDF5_FILE"
    continue
  fi

  echo ""
  echo "============================================================"
  echo "[$COUNT/$TOTAL] Replaying episode $ep (${ROBOT_MAP[$ep]})"
  echo "  Input:  $HDF5_FILE"
  echo "  Output: $OUTPUT_DIR"
  echo "  Cameras: cam_wrist_right, cam_wrist_left  (generalization OFF)"
  echo "============================================================"

  cd "$SCRIPT_DIR"
  python replay.py \
    --hdf5 "$HDF5_FILE" \
    --enable-rgb \
    --cameras cam_wrist_right cam_wrist_left \
    --output "$OUTPUT_DIR" \
    --headless
  # NOTE: --enable-generalization is intentionally OMITTED → camera
  # view perturbation (position / rotation / distance offsets) is OFF.

  echo "[$COUNT/$TOTAL] Done: episode $ep"
done

echo ""
echo "============================================================"
echo "All $TOTAL episodes replayed."
echo "Output: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/
echo "============================================================"
