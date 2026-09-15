#!/usr/bin/env bash

# Shared rollout budget helper for policy eval scripts.
# run_policy.py interprets --episode-steps as policy steps, then multiplies by
# POLICY_STRIDE to get the physics-step budget.

eval_budget_python() {
    if [[ -n "${EVAL_BUDGET_PYTHON:-}" ]]; then
        printf '%s\n' "${EVAL_BUDGET_PYTHON}"
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    elif command -v python >/dev/null 2>&1; then
        command -v python
    else
        echo "[eval] ERROR: python3 or python is required to compute eval budget." >&2
        return 1
    fi
}

eval_budget_parse_options() {
    EVAL_EPISODE_STEPS_OVERRIDE=""
    EVAL_MAX_STEPS_OVERRIDE=""
    EVAL_REMAINING_ARGS=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --episode-steps|--episode_steps)
                if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
                    echo "[eval] ERROR: missing value for $1" >&2
                    return 2
                fi
                EVAL_EPISODE_STEPS_OVERRIDE=$2
                shift 2
                ;;
            --episode-steps=*|--episode_steps=*)
                EVAL_EPISODE_STEPS_OVERRIDE="${1#*=}"
                if [[ -z "${EVAL_EPISODE_STEPS_OVERRIDE}" ]]; then
                    echo "[eval] ERROR: missing value for ${1%%=*}" >&2
                    return 2
                fi
                shift
                ;;
            --max-steps|--max_steps)
                if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
                    echo "[eval] ERROR: missing value for $1" >&2
                    return 2
                fi
                EVAL_MAX_STEPS_OVERRIDE=$2
                shift 2
                ;;
            --max-steps=*|--max_steps=*)
                EVAL_MAX_STEPS_OVERRIDE="${1#*=}"
                if [[ -z "${EVAL_MAX_STEPS_OVERRIDE}" ]]; then
                    echo "[eval] ERROR: missing value for ${1%%=*}" >&2
                    return 2
                fi
                shift
                ;;
            *)
                EVAL_REMAINING_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

eval_budget_compute() {
    local task="${1:?eval_budget_compute requires TASK}"
    local episode_steps_override="${2:-}"
    local max_steps_override="${3:-}"
    local default_episode_steps="${4:-${EVAL_DEFAULT_EPISODE_STEPS:-800}}"
    local policy_stride="${EVAL_POLICY_STRIDE:-3}"
    local scale="${EXPERT_MAX_STEPS_SCALE:-1.5}"
    local physics_hz="${EVAL_PHYSICS_HZ:-60}"
    local py
    py=$(eval_budget_python) || return

    local output
    output=$("${py}" - \
        "${task}" \
        "${episode_steps_override}" \
        "${max_steps_override}" \
        "${policy_stride}" \
        "${scale}" \
        "${default_episode_steps}" \
        "${physics_hz}" <<'PY'
import math
import pathlib
import re
import sys

task, episode_override, max_override, stride_s, scale_s, default_s, physics_hz_s = sys.argv[1:]
stride = int(stride_s)
scale = float(scale_s)
default_episode_steps = int(default_s)
physics_hz = float(physics_hz_s)

if stride <= 0:
    raise SystemExit("policy stride must be positive")
if default_episode_steps <= 0:
    raise SystemExit("default episode steps must be positive")
if physics_hz <= 0:
    raise SystemExit("physics Hz must be positive")

def positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise SystemExit(f"{name} must be positive, got {parsed}")
    return parsed

if max_override:
    max_steps_requested = positive_int(max_override, "max_steps")
    episode_steps = int(math.ceil(max_steps_requested / stride))
    max_steps = episode_steps * stride
    source = f"--max-steps {max_steps_requested}"
elif episode_override:
    episode_steps = positive_int(episode_override, "episode_steps")
    max_steps = episode_steps * stride
    source = f"--episode-steps {episode_steps}"
else:
    scene_path = pathlib.Path("scenes") / f"{task}.yaml"
    expert_time_step = None
    expert_time_s = None
    if scene_path.is_file():
        step_pattern = re.compile(r"^\s*expert_time_step\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:#.*)?$")
        seconds_pattern = re.compile(r"^\s*expert_time_s\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:#.*)?$")
        with scene_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if expert_time_step is None:
                    match = step_pattern.match(line)
                    if match:
                        expert_time_step = float(match.group(1))
                if expert_time_s is None:
                    match = seconds_pattern.match(line)
                    if match:
                        expert_time_s = float(match.group(1))
                if expert_time_step is not None and expert_time_s is not None:
                    break

    if expert_time_step is not None:
        target_max_steps = expert_time_step * scale
        episode_steps = int(math.ceil(target_max_steps / stride))
        max_steps = episode_steps * stride
        source = f"expert_time_step={expert_time_step:g} * {scale:g}"
    elif expert_time_s is not None:
        target_max_steps = expert_time_s * physics_hz * scale
        episode_steps = int(math.ceil(target_max_steps / stride))
        max_steps = episode_steps * stride
        source = f"expert_time_s={expert_time_s:g}s * {physics_hz:g}Hz * {scale:g}"
    else:
        episode_steps = default_episode_steps
        max_steps = episode_steps * stride
        source = "fallback default (expert_time_step/expert_time_s missing)"

print(f"{episode_steps}\t{max_steps}\t{source}")
PY
    ) || return

    IFS=$'\t' read -r EVAL_EPISODE_STEPS EVAL_MAX_STEPS EVAL_BUDGET_SOURCE <<< "${output}"
}

eval_budget_log() {
    echo "[eval] episode_steps=${EVAL_EPISODE_STEPS} policy steps, max_steps=${EVAL_MAX_STEPS} physics steps (${EVAL_BUDGET_SOURCE})"
}
