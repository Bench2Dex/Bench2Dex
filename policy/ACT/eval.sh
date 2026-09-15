#!/bin/bash
set -euo pipefail

# Evaluate a trained ACT policy through the server/client entrypoint.
#
# Usage:
#   bash ./policy/ACT/eval.sh TASK CKPT_DIR [ckpt_name] [gpu_id] [num_episodes] [seed] [extra overrides...]
#
# Args:
#   TASK       task folder name, e.g. 06_fruit_bowl_loading
#   CKPT_DIR   kept for compatibility with eval_double_env.sh; ignored here if
#              you connect to an already running server.
#   CKPT_NAME  (optional) checkpoint file name, default: policy_last.ckpt
#   GPU_ID     (optional) CUDA device id, default: 0
#   NUM_EPISODES (optional) max episode count, default: 0 (infinite)
#   SEED       (optional) eval seed namespace base, default: 100000000
#   --episode-steps N  Override policy-step budget.
#   --max-steps N      Override physics-step budget.

TASK=${1:?"Usage: bash eval.sh TASK CKPT_DIR [ckpt_name] [gpu_id]"}
CKPT_DIR=${2:?"Usage: bash eval.sh TASK CKPT_DIR [ckpt_name] [gpu_id]"}
shift 2

CKPT_NAME=policy_last.ckpt
GPU_ID=0
NUM_EPISODES=0
SEED=100000000
if [[ $# -gt 0 && "$1" != --* ]]; then CKPT_NAME=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then GPU_ID=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then NUM_EPISODES=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then SEED=$1; shift; fi

export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Change to dex2bench root (two levels up from policy/ACT/)
cd "$(dirname "$0")/../.."

source script/eval_budget.sh
eval_budget_parse_options "$@"
eval_budget_compute "${TASK}" "${EVAL_EPISODE_STEPS_OVERRIDE}" "${EVAL_MAX_STEPS_OVERRIDE}"
eval_budget_log

python script/eval_policy_client.py \
    --config policy/ACT/deploy_policy.yml \
    --overrides \
    --policy_name ACT \
    --task_name "${TASK}" \
    --ckpt_dir "${CKPT_DIR}" \
    --ckpt_name "${CKPT_NAME}" \
    --num_episodes "${NUM_EPISODES}" \
    --seed "${SEED}" \
    --episode_steps "${EVAL_EPISODE_STEPS}" \
    --warmup_steps 60 \
    --temporal_agg true \
    --temporal_agg_k 0.2 \
    "${EVAL_REMAINING_ARGS[@]}"
