#!/bin/bash
set -euo pipefail

# Evaluate ACT through server/client entrypoint.
#
# Usage:
#   bash eval_double_env.sh TASK CKPT_DIR [positionals] [--key value ...]
#
# Positional (in order):
#   CKPT_NAME      checkpoint file name, default: policy_best.ckpt
#   GPU_ID         CUDA device id, default: 0
#   NUM_EPISODES   episodes to run, default: 50
#   SEED           eval seed, default: 100000000
#
# Named flags (all optional):
#   --generalization-profile PROFILE   none|cov_only|inv_only|inv_cov (default: none)
#   --anchor-dir DIR                   anchor HDF5 directory
#   --record-dir DIR                   output directory
#   --start-episode N                  starting episode index (default: 0)
#   --episode-steps N                  policy-step budget
#   --max-steps N                      physics-step budget
#   --ckpt-name NAME                   checkpoint file name (overrides positional)
#   --num-episodes N                   episodes to run (overrides positional)
#   --seed N                           eval seed (overrides positional)
#   --headless                         run without GUI

TASK=${1:?"Usage: bash eval_double_env.sh TASK CKPT_DIR [args...]"}
CKPT_DIR=${2:?"Usage: bash eval_double_env.sh TASK CKPT_DIR [args...]"}
shift 2

# ── Defaults ──────────────────────────────────────────────────────────
CKPT_NAME=policy_best.ckpt
GPU_ID=0
NUM_EPISODES=50
SEED=100000000
GEN_PROFILE=none
ANCHOR_DIR=""
RECORD_DIR=""
START_EPISODE=0
EPISODE_STEPS_OVERRIDE=""
MAX_STEPS_OVERRIDE=""
HEADLESS=false
HEADLESS_ARG=""
USE_ACTIVE_DOF=true
ACTIVE_DOF_ARG="--use_active_dof true"

# ── Positional args ───────────────────────────────────────────────────
if [[ $# -gt 0 && "$1" != --* ]]; then CKPT_NAME=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then GPU_ID=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then NUM_EPISODES=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then SEED=$1; shift; fi

# ── Named args ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --generalization-profile)   GEN_PROFILE="$2"; shift 2 ;;
        --generalization-profile=*) GEN_PROFILE="${1#*=}"; shift ;;
        --anchor-dir)   ANCHOR_DIR="$2"; shift 2 ;;
        --anchor-dir=*) ANCHOR_DIR="${1#*=}"; shift ;;
        --record-dir)   RECORD_DIR="$2"; shift 2 ;;
        --record-dir=*) RECORD_DIR="${1#*=}"; shift ;;
        --start-episode)   START_EPISODE="$2"; shift 2 ;;
        --start-episode=*) START_EPISODE="${1#*=}"; shift ;;
        --episode-steps)   EPISODE_STEPS_OVERRIDE="$2"; shift 2 ;;
        --episode-steps=*) EPISODE_STEPS_OVERRIDE="${1#*=}"; shift ;;
        --max-steps)   MAX_STEPS_OVERRIDE="$2"; shift 2 ;;
        --max-steps=*) MAX_STEPS_OVERRIDE="${1#*=}"; shift ;;
        --ckpt-name)   CKPT_NAME="$2"; shift 2 ;;
        --ckpt-name=*) CKPT_NAME="${1#*=}"; shift ;;
        --num-episodes)   NUM_EPISODES="$2"; shift 2 ;;
        --num-episodes=*) NUM_EPISODES="${1#*=}"; shift ;;
        --seed)   SEED="$2"; shift 2 ;;
        --seed=*) SEED="${1#*=}"; shift ;;
        --headless) HEADLESS=true; HEADLESS_ARG="--headless true"; shift ;;
        --use-active-dof)   USE_ACTIVE_DOF="$2"; ACTIVE_DOF_ARG="--use_active_dof $2"; shift 2 ;;
        --use-active-dof=*) USE_ACTIVE_DOF="${1#*=}"; ACTIVE_DOF_ARG="--use_active_dof ${1#*=}"; shift ;;
        *) shift ;;  # ignore unknown, passed via EVAL_REMAINING_ARGS
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../.."

source script/eval_budget.sh
eval_budget_parse_options "$@"
eval_budget_compute "${TASK}" "${EPISODE_STEPS_OVERRIDE}" "${MAX_STEPS_OVERRIDE}"
eval_budget_log

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"

FREE_PORT=$(python3 - << 'PYEOF'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PYEOF
)

python script/policy_model_server.py \
    --host 127.0.0.1 \
    --port "${FREE_PORT}" \
    --config policy/ACT/deploy_policy.yml \
    --overrides \
    --policy_name ACT \
    --task_name "${TASK}" \
    --robot_key "" \
    --ckpt_dir "${CKPT_DIR}" \
    --ckpt_name "${CKPT_NAME}" \
    --seed "${SEED}" \
    --temporal_agg true \
    --temporal_agg_k 0.2 \
    ${ACTIVE_DOF_ARG} &
SERVER_PID=$!

trap "kill ${SERVER_PID} 2>/dev/null || true" EXIT
sleep 3

python script/eval_policy_client.py \
    --host 127.0.0.1 \
    --port "${FREE_PORT}" \
    --config policy/ACT/deploy_policy.yml \
    --overrides \
    --policy_name ACT \
    --task_name "${TASK}" \
    --ckpt_dir "${CKPT_DIR}" \
    --ckpt_name "${CKPT_NAME}" \
    --num_episodes "${NUM_EPISODES}" \
    --seed "${SEED}" \
    --episode_steps "${EVAL_EPISODE_STEPS}" \
    --start_episode "${START_EPISODE}" \
    --warmup_steps 60 \
    --temporal_agg true \
    --temporal_agg_k 0.2 \
    ${ACTIVE_DOF_ARG} \
    --generalization_profile "${GEN_PROFILE}" \
    ${RECORD_DIR:+--record_dir "${RECORD_DIR}"} \
    ${RECORD_DIR:+--record_all true} \
    ${ANCHOR_DIR:+--anchor_dir "${ANCHOR_DIR}"} \
    ${HEADLESS_ARG} \
    "${EVAL_REMAINING_ARGS[@]}"
