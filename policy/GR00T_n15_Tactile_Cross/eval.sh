#!/usr/bin/env bash
set -euo pipefail

task_name=${1:?Usage: eval.sh TASK_NAME MODEL_PATH [gpu_id] [num_episodes] [seed] [extra overrides...]}
model_path=${2:?Usage: eval.sh TASK_NAME MODEL_PATH [gpu_id] [num_episodes] [seed] [extra overrides...]}
shift 2

gpu_id=0
num_episodes=20
seed=100000000
ckpt_name=${GR00T_CKPT_NAME:-gr00t_n15}
if [[ $# -gt 0 && "$1" != --* ]]; then gpu_id=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then num_episodes=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then seed=$1; shift; fi

export CUDA_VISIBLE_DEVICES=${gpu_id}

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/policy/GR00T_n15_Tactile_Cross/src:${PYTHONPATH:-}"

source script/eval_budget.sh
eval_budget_parse_options "$@"
eval_budget_compute "${task_name}" "${EVAL_EPISODE_STEPS_OVERRIDE}" "${EVAL_MAX_STEPS_OVERRIDE}"
eval_budget_log

_yaml_scalar() {
    local key="$1"
    local value=""
    value=$(grep -m1 -E "^[[:space:]]*${key}:" policy/GR00T_n15_Tactile_Cross/deploy_policy.yml | sed -E "s/^[[:space:]]*${key}:[[:space:]]*//; s/[[:space:]]+#.*$//; s/^[[:space:]]+//; s/[[:space:]]+$//" || true)
    if [[ "${value}" == "null" ]]; then
        value=""
    fi
    printf '%s\n' "${value}"
}

state_dim=${GR00T_STATE_DIM:-$(_yaml_scalar state_dim)}
action_dim=${GR00T_ACTION_DIM:-$(_yaml_scalar action_dim)}
max_state_dim=${GR00T_MAX_STATE_DIM:-$(_yaml_scalar max_state_dim)}
max_action_dim=${GR00T_MAX_ACTION_DIM:-$(_yaml_scalar max_action_dim)}
max_state_dim=${max_state_dim:-64}
max_action_dim=${max_action_dim:-64}
if [[ -n "${state_dim}" && "${state_dim}" -gt "${max_state_dim}" ]]; then
    max_state_dim="${state_dim}"
fi
if [[ -n "${action_dim}" && "${action_dim}" -gt "${max_action_dim}" ]]; then
    max_action_dim="${action_dim}"
fi
export GR00T_MAX_STATE_DIM="${max_state_dim}"
export GR00T_MAX_ACTION_DIM="${max_action_dim}"

python script/eval_policy_client.py \
    --config policy/GR00T_n15_Tactile_Cross/deploy_policy.yml \
    --overrides \
    --policy_name GR00T_n15 \
    --task_name "${task_name}" \
    --model_path "${model_path}" \
    --ckpt_dir "${model_path}" \
    --ckpt_name "${ckpt_name}" \
    --seed "${seed}" \
    --num_episodes "${num_episodes}" \
    --episode_steps "${EVAL_EPISODE_STEPS}" \
    --warmup_steps 60 \
    "${EVAL_REMAINING_ARGS[@]}"
