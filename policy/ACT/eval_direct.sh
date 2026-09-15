#!/bin/bash
# 直接推理评测（不经过 server/client）。
# 修改下面 "默认配置" 区域的变量即可切换评测对象，无需每次粘贴长命令。
#
# Usage:
#   bash policy/ACT/eval_direct.sh                # 用脚本里的默认值跑
#   bash policy/ACT/eval_direct.sh TASK CKPT_DIR  # 覆盖 task 和 ckpt 路径
#   bash policy/ACT/eval_direct.sh TASK CKPT_DIR policy_last.ckpt 0 50 100000000 --max-steps 2400
#
# 命令行参数（可选，会覆盖脚本里的默认值）：
#   $1  TASK
#   $2  CKPT_DIR
#   $3  CKPT_NAME
#   $4  GPU_ID
#   $5  NUM_EPISODES
#   $6  SEED
#   之后可追加 --episode-steps N 或 --max-steps N

set -euo pipefail

# ---- 任务 ----
TASK="${TASK:-}"

# ---- checkpoint ----
CKPT_DIR="${CKPT_DIR:-}"
CKPT_NAME="policy_last.ckpt"

# ---- 机器人 ----
ROBOT_KEY="${ROBOT_KEY:-multi_ur5_rh56dfx_with_flange}"

# ---- 推理参数 ----
GPU_ID=0
NUM_EPISODES=50
SEED=100000000
EPISODE_STEPS="${EPISODE_STEPS:-}"
MAX_STEPS="${MAX_STEPS:-}"
WARMUP_STEPS=60

# ---- ACT 模型参数 ----
STATE_DIM=36
USE_ACTIVE_DOF="${USE_ACTIVE_DOF:-false}"
TEMPORAL_AGG_K=0.1

# ---- generalization ----
DATASET_ROOT="${DATASET_ROOT:-}"
ANCHOR_EPISODE="${ANCHOR_EPISODE:-episode_000001.hdf5}"
GEN_PROFILE="${GEN_PROFILE:-none}"   # none | cov_only | inv_only | inv_cov

# ---- 其他 ----
HEADLESS="--headless"          # 无头模式；留空去掉 "--headless" 则开窗口

# 位置参数覆盖（可选）
if [[ $# -gt 0 && "$1" != --* ]]; then TASK=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then CKPT_DIR=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then CKPT_NAME=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then GPU_ID=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then NUM_EPISODES=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then SEED=$1; shift; fi

if [[ -z "${TASK}" ]]; then
    echo "Usage: bash eval_direct.sh TASK [ckpt_dir] [ckpt_name] [gpu] [num_episodes] [seed]" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "$(dirname "$0")/../.."

source script/eval_budget.sh
eval_budget_parse_options "$@"
EPISODE_STEPS_OVERRIDE="${EVAL_EPISODE_STEPS_OVERRIDE:-${EPISODE_STEPS:-}}"
MAX_STEPS_OVERRIDE="${EVAL_MAX_STEPS_OVERRIDE:-${MAX_STEPS:-}}"
if [[ -n "${EVAL_EPISODE_STEPS_OVERRIDE}" && -z "${EVAL_MAX_STEPS_OVERRIDE}" ]]; then
    MAX_STEPS_OVERRIDE=""
fi
eval_budget_compute "${TASK}" "${EPISODE_STEPS_OVERRIDE}" "${MAX_STEPS_OVERRIDE}"
EPISODE_STEPS="${EVAL_EPISODE_STEPS}"

# ---------- anchor hdf5 ----------
ANCHOR_HDF5="${ANCHOR_HDF5:-${DATASET_ROOT}/${TASK}/replay-generalization/${ANCHOR_EPISODE}}"
if [[ ! -f "${ANCHOR_HDF5}" ]]; then
    echo "ERROR: anchor HDF5 not found: ${ANCHOR_HDF5}" >&2
    exit 1
fi

_AUTO_STATE_DIM=$(python3 - <<PY
from robots.active_dof_utils import get_active_dof_info
info = get_active_dof_info("${ROBOT_KEY}")
use_active = "${USE_ACTIVE_DOF}".lower() == "true"
print(info.active_dof if use_active else info.full_dof)
PY
)
if [[ -n "${_AUTO_STATE_DIM}" ]]; then
    STATE_DIM="${_AUTO_STATE_DIM}"
fi

# ---------- output dir ----------
CKPT_SHORT=$(echo "${CKPT_NAME}" | sed 's/^policy_//' | sed 's/\.ckpt$//')
TIMESTAMP=$(date +%y%m%d_%H%M)
OUTPUT_DIR="${OUTPUT_DIR:-./output/metric_${TASK}_${CKPT_SHORT}_${TIMESTAMP}_${GEN_PROFILE}_double}"

# ---------- active dof ----------
if [[ "${USE_ACTIVE_DOF}" == "true" ]]; then
    ACTIVE_DOF_ARG=""
else
    ACTIVE_DOF_ARG="--no-active-dof"
fi

echo "============================================"
echo "  ACT direct eval"
echo "  task          : ${TASK}"
echo "  ckpt          : ${CKPT_DIR}/${CKPT_NAME}"
echo "  robot_key     : ${ROBOT_KEY}"
echo "  active_dof    : ${USE_ACTIVE_DOF}"
echo "  state_dim     : ${STATE_DIM}"
echo "  temporal_agg_k: ${TEMPORAL_AGG_K}"
echo "  episode_steps : ${EPISODE_STEPS}"
echo "  max_steps     : ${EVAL_MAX_STEPS}"
echo "  episodes      : ${NUM_EPISODES}"
echo "  seed          : ${SEED}"
echo "  gpu           : ${GPU_ID}"
echo "  anchor_hdf5   : ${ANCHOR_HDF5}"
echo "  gen_profile   : ${GEN_PROFILE}"
echo "  output        : ${OUTPUT_DIR}"
echo "============================================"
eval_budget_log

python run_policy.py \
    --policy-type ACT \
    --task "scenes/${TASK}.yaml" \
    --ckpt-dir "${CKPT_DIR}" \
    --ckpt-name "${CKPT_NAME}" \
    --robot-key "${ROBOT_KEY}" \
    ${ACTIVE_DOF_ARG} \
    --state-dim "${STATE_DIM}" \
    --enable-rgb \
    --temporal-agg \
    --temporal-agg-k "${TEMPORAL_AGG_K}" \
    --episode-steps "${EPISODE_STEPS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --num-episodes "${NUM_EPISODES}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}" \
    --enable-generalization \
    --generalization-profile "${GEN_PROFILE}" \
    --anchor-hdf5 "${ANCHOR_HDF5}" \
    ${HEADLESS} \
    "${EVAL_REMAINING_ARGS[@]}"
