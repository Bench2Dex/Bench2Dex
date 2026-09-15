#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEX2SCENE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATASET_DIR="${1:-${DEX2SCENE_ROOT}/../tactile_leverage}"
CKPT_DIR="${2:-${DEX2SCENE_ROOT}/../policy_ckpt/tactile_act}"

if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi

PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${DEX2SCENE_ROOT}"

mkdir -p "${CKPT_DIR}"

# Manage the log file here so callers do not need the directory to pre-exist
# for shell redirection (e.g. `> ckpt_dir/train.log 2>&1` fails without it).
if [[ -n "${TACTILE_TRAIN_LOG:-1}" ]]; then
    exec > >(tee "${CKPT_DIR}/train.log") 2>&1
fi

"${PYTHON_BIN}" -m policy.tactile_policy.train \
  --dataset-dir "${DATASET_DIR}" \
  --ckpt-dir "${CKPT_DIR}" \
  --chunk-size 30 \
  --backbone resnet18 \
  "$@"
