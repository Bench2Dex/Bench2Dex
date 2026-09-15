#!/bin/bash
set -euo pipefail

# Evaluate a trained DP policy through the server/client entrypoint.
#
# Usage:
#   bash ./policy/DP/eval.sh TASK CKPT_PATH_OR_DIR [ckpt_name] [gpu_id] [num_episodes] [seed] [extra run_policy args...]
#   Extra budget overrides: --episode-steps N or --max-steps N.
#
# Examples:
#   bash ./policy/DP/eval.sh 42_trash_disposal data/outputs/run/checkpoints/replay-generalization-42 last.ckpt 0 10 42
#   bash ./policy/DP/eval.sh 42_trash_disposal data/outputs/run/checkpoints/replay-generalization-42/last.ckpt

TASK=${1:?"Usage: bash eval.sh TASK CKPT_PATH_OR_DIR [ckpt_name] [gpu_id] [num_episodes] [seed]"}
CKPT_INPUT=${2:?"Usage: bash eval.sh TASK CKPT_PATH_OR_DIR [ckpt_name] [gpu_id] [num_episodes] [seed]"}
shift 2

CKPT_NAME=last.ckpt
GPU_ID=0
NUM_EPISODES=0
SEED=100000000
policy_name=DP
if [[ $# -gt 0 && "$1" != --* ]]; then CKPT_NAME=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then GPU_ID=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then NUM_EPISODES=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then SEED=$1; shift; fi

if [ -d "${CKPT_INPUT}" ]; then
    CKPT_DIR="${CKPT_INPUT}"
else
    CKPT_DIR="$(dirname "${CKPT_INPUT}")"
    CKPT_NAME="$(basename "${CKPT_INPUT}")"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../.."

source script/eval_budget.sh
eval_budget_parse_options "$@"
eval_budget_compute "${TASK}" "${EVAL_EPISODE_STEPS_OVERRIDE}" "${EVAL_MAX_STEPS_OVERRIDE}"
eval_budget_log

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo -e "\033[33mDP dex2scene eval\033[0m"
echo -e "\033[33mtask: ${TASK}\033[0m"
echo -e "\033[33mcheckpoint: ${CKPT_DIR}/${CKPT_NAME}\033[0m"
echo -e "\033[33mgpu: ${GPU_ID}, seed: ${SEED}, num_episodes: ${NUM_EPISODES}\033[0m"

python script/eval_policy_client.py \
    --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --policy_name ${policy_name} \
    --task_name "${TASK}" \
    --checkpoint_path "${CKPT_DIR}/${CKPT_NAME}" \
    --training_config_path "policy/${policy_name}/diffusion_policy/config/robot_dp_36_dex2scene_pretrained.yaml" \
    --num_episodes "${NUM_EPISODES}" \
    --seed "${SEED}" \
    --episode_steps "${EVAL_EPISODE_STEPS}" \
    --warmup_steps 60 \
    "${EVAL_REMAINING_ARGS[@]}"
