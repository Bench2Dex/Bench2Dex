#!/usr/bin/env bash
# ============================================================================
# GR00T XE (Cross-Embodiment) 评测脚本 — server/client 模式
#
# 用法:
#   bash policy/GR00T_XE/eval_double_env.sh <TASK_ID> [CHANNEL] [--key value ...]
#
# 与 GR00T_n15 的 eval 一致，但使用 GR00T_XE 的 deploy_policy 和模型。
# 模型输出 64D 统一 action，在 deploy_policy 内部完成 IK/手部映射转换。
# ============================================================================
set -euo pipefail

TASK_ID="${1:?Usage: eval_double_env.sh <TASK_ID> [--channel CHANNEL] [--key value ...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../.."
REPO_ROOT="$(pwd)"
POLICY_CKPT_ROOT="${REPO_ROOT}/../policy_ckpt"
TELEOPDATA_ROOT="${REPO_ROOT}/../teleopdata/dataset"

die() { echo -e "\033[31m[FATAL] $*\033[0m" >&2; exit 1; }
warn() { echo -e "\033[33m[WARN] $*\033[0m" >&2; }
info() { echo -e "\033[34m[info] $*\033[0m"; }

# resolve task name
TASK_GLOB=(scenes/${TASK_ID}_*.yaml)
if [[ ! -f "${TASK_GLOB[0]}" ]]; then
    die "No scene file found matching: scenes/${TASK_ID}_*.yaml"
fi
SCENE_FILE="${TASK_GLOB[0]}"
TASK_NAME="$(basename "${SCENE_FILE}" .yaml)"
info "Task: ${TASK_NAME}"

# defaults
CHANNEL="none"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
NUM_EPISODES=50
START_EPISODE=1
EPISODES_PER_PROCESS=25
SEED=100000000
MODEL_PATH="${GR00T_XE_MODEL_PATH:-}"
ROBOT_KEY="${GR00T_XE_ROBOT_KEY:-}"
RECORD_ALL=true

# parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --channel) CHANNEL="$2"; shift 2 ;;
        --model_path|--model-path) MODEL_PATH="$2"; shift 2 ;;
        --robot_key|--robot-key) ROBOT_KEY="$2"; shift 2 ;;
        --cuda|--gpu) GPU_ID="$2"; shift 2 ;;
        --num_episodes|--num-episodes) NUM_EPISODES="$2"; shift 2 ;;
        --start_episode|--start-episode) START_EPISODE="$2"; shift 2 ;;
        --episodes_per_process|--episodes-per-process) EPISODES_PER_PROCESS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --record_all) RECORD_ALL=true; shift ;;
        --no_record) RECORD_ALL=false; shift ;;
        *) shift ;;
    esac
done

# normalize channel
CHANNEL=$(echo "$CHANNEL" | tr '[:upper:]' '[:lower:]')
case "$CHANNEL" in
    ""|none) CHANNEL="none" ;;
    cov|cov_only) CHANNEL="cov_only" ;;
    inv|inv_only) CHANNEL="inv_only" ;;
    inv_cov|inv+cov|full) CHANNEL="inv_cov" ;;
    *) die "Unknown channel: $CHANNEL" ;;
esac

# auto-discover model_path
if [[ -z "${MODEL_PATH}" ]]; then
    shopt -s nullglob
    _ckpt_candidates=("${POLICY_CKPT_ROOT}/${TASK_ID}"/*/gr00t_xe*)
    shopt -u nullglob
    for _c in "${_ckpt_candidates[@]}"; do
        if [[ -f "${_c}/config.json" ]]; then
            MODEL_PATH="$_c"
            break
        fi
    done
fi
if [[ -z "${MODEL_PATH}" ]]; then
    die "No GR00T XE checkpoint found. Set GR00T_XE_MODEL_PATH or place under ${POLICY_CKPT_ROOT}/${TASK_ID}/*/gr00t_xe*"
fi
info "Model: ${MODEL_PATH}"

# auto-discover robot_key
if [[ -z "${ROBOT_KEY}" ]]; then
    if [[ -f "${MODEL_PATH}/robot_key.txt" ]]; then
        ROBOT_KEY="$(tr -d '[:space:]' < "${MODEL_PATH}/robot_key.txt")"
    else
        ROBOT_KEY="$(basename "$(dirname "${MODEL_PATH}")")"
    fi
fi
info "Robot: ${ROBOT_KEY}"

# anchor dir
ANCHOR_DIR="${TELEOPDATA_ROOT}/${TASK_NAME}/replay-generalization"
[[ -d "${ANCHOR_DIR}" ]] || ANCHOR_DIR=""

# output dir
TIMESTAMP="$(date +%m%d_%H%M)"
SHORT_ROBOT="${ROBOT_KEY#multi_}"
SHORT_ROBOT="${SHORT_ROBOT//_with_flange/}"
SHORT_ROBOT="${SHORT_ROBOT//_with_leap/}"
SHORT_ROBOT="${SHORT_ROBOT//ur5_/}"
EVAL_ROOT="${REPO_ROOT}/../output/end_eval/${TASK_ID}/gr00t_xe_${CHANNEL}_${TIMESTAMP}_${SHORT_ROBOT}"
CHANNEL_DIR="${EVAL_ROOT}/${CHANNEL}"
mkdir -p "${CHANNEL_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/policy/GR00T_n15/src:${REPO_ROOT}/policy/GR00T_XE/src:${PYTHONPATH:-}"
export GR00T_MAX_STATE_DIM=64
export GR00T_MAX_ACTION_DIM=64

# find a free port
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# start server
info "Starting server on port ${PORT}..."
(
    # Use groot env (has pytorch3d + pinocchio 2.7.0 + numpy 1.26.4)
    source /inspire/hdd/project/roboticsystem2/ky26063/bench2dex_stuff/miniconda3/bin/activate groot
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/policy/GR00T_n15/src:${REPO_ROOT}/policy/GR00T_XE/src:${PYTHONPATH:-}"
    exec python script/policy_model_server.py \
        --host 127.0.0.1 --port "${PORT}" \
        --config policy/GR00T_XE/deploy_policy.yml \
        --overrides \
        --policy_name GR00T_XE \
        --task_name "${TASK_NAME}" \
        --model_path "${MODEL_PATH}" \
        --ckpt_dir "${MODEL_PATH}" \
        --robot_key "${ROBOT_KEY}" \
        --seed "${SEED}" \
        --state_dim 64 --action_dim 64 \
        --use_active_dof true
) &
SERVER_PID=$!
trap "kill ${SERVER_PID} 2>/dev/null || true" EXIT

# wait for server
info "Waiting for server..."
for _i in {1..120}; do
    if python3 -c "from script.policy_rpc import RemotePolicyClient; RemotePolicyClient('127.0.0.1', ${PORT}, timeout_s=1).close()" 2>/dev/null; then
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        die "Server exited before opening port ${PORT}"
    fi
    sleep 1
done

# ── budget compute (per-task max steps) ────────────────────────────────────

source script/eval_budget.sh
eval_budget_compute "${TASK_NAME}" "" ""
eval_budget_log
EPISODE_STEPS="${EVAL_EPISODE_STEPS}"

# launch client
info "Client: channel=${CHANNEL} episodes=${START_EPISODE}-$((START_EPISODE + NUM_EPISODES - 1))"

source /inspire/hdd/project/roboticsystem2/ky26063/bench2dex_stuff/isaaclab_setup/env_new.sh
export DISPLAY=:99
setup_nvidia_libs
activate_isaaclab

RECORD_ARGS=()
if [[ "${RECORD_ALL}" == "true" ]]; then
    RECORD_ARGS=(--record_dir "${CHANNEL_DIR}/episodes" --record_all true)
fi

ANCHOR_ARGS=()
if [[ -n "${ANCHOR_DIR}" && "${CHANNEL}" != "inv_cov" ]]; then
    ANCHOR_ARGS=(--anchor-dir "${ANCHOR_DIR}")
fi

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy_client.py \
    --host 127.0.0.1 --port "${PORT}" \
    --config policy/GR00T_XE/deploy_policy.yml \
    --overrides \
    --policy_name GR00T_XE \
    --task_name "${TASK_NAME}" \
    --model_path "${MODEL_PATH}" \
    --ckpt_dir "${MODEL_PATH}" \
    --robot_key "${ROBOT_KEY}" \
    --seed "${SEED}" \
    --state_dim 64 --action_dim 64 \
    --use_active_dof true \
    --episode_steps "${EPISODE_STEPS}" \
    --warmup_steps 60 \
    --generalization_profile "${CHANNEL}" \
    --output_dir "${CHANNEL_DIR}" \
    --num_episodes "${NUM_EPISODES}" \
    --start_episode "${START_EPISODE}" \
    "${ANCHOR_ARGS[@]}" \
    "${RECORD_ARGS[@]}"

# cleanup
kill "${SERVER_PID}" 2>/dev/null || true
trap - EXIT

# summary
if [[ -f "${CHANNEL_DIR}/per_episode.jsonl" ]]; then
    python3 -c "
import json
rows = [json.loads(l) for l in open('${CHANNEL_DIR}/per_episode.jsonl') if l.strip()]
ok = sum(1 for r in rows if r.get('success'))
print(f'GR00T XE ${TASK_NAME} ${CHANNEL}: {ok}/{len(rows)} ({ok/len(rows)*100:.1f}%)' if rows else 'No results')
"
fi

info "Done. Output: ${CHANNEL_DIR}"