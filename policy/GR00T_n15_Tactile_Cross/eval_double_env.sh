#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# GR00T N1.5 split server/client evaluation — one-line launcher
#
# Usage:
#   bash eval_double_env.sh <TASK_ID> [CHANNEL] [--key value ...]
#
# Channels (positional or --channel):
#   none      No generalization (default)
#   cov       Covariate generalization only
#   inv       Invariant generalization only
#   inv_cov   Both covariate and invariant (no anchor)
#   all       Run all 4 channels sequentially and aggregate results
#
# Isaac restart (--episodes-per-process, default 25):
#   Each process = start Isaac -> run N episodes -> kill Isaac.
#   Restarting releases GPU memory and resets scene state.
#
# Examples:
#   bash eval_double_env.sh 44                          # channel=none, restart every 25 ep
#   bash eval_double_env.sh 44 cov                      # positional channel
#   bash eval_double_env.sh 44 all                      # positional all: 4 channels
#   bash eval_double_env.sh 44 --channel inv_cov
#   bash eval_double_env.sh 44 all --episodes-per-process 30
#   bash eval_double_env.sh 44 --channel all --num-episodes 100 --episodes-per-process 50
#   bash eval_double_env.sh 44 --channel_num 4 --resume       # resume latest 4-channel run
#   bash eval_double_env.sh 44 --channel_num 2                # inv_only -> inv_cov
#   bash eval_double_env.sh 44 --cuda 1                                 # use GPU 1
#   bash eval_double_env.sh 44 all --cuda 2                             # use GPU 2
# ============================================================================

TASK_ID="${1:?Usage: eval_double_env.sh <TASK_ID> [--channel CHANNEL] [--key value ...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/tactile_checkpoint.sh"
cd "${SCRIPT_DIR}/../.."
REPO_ROOT="$(pwd)"
POLICY_CKPT_ROOT="${REPO_ROOT}/../policy_ckpt"
TELEOPDATA_ROOT="${REPO_ROOT}/../teleopdata/dataset"

# ── helpers ──────────────────────────────────────────────────────────────────

die() { echo -e "\033[31m[FATAL] $*\033[0m" >&2; exit 1; }
warn() { echo -e "\033[33m[WARN] $*\033[0m" >&2; }
info() { echo -e "\033[34m[info] $*\033[0m"; }

# ── resolve a working Python with numpy (robust across tmux/conda/venv) ─────

_PYTHON=""

_resolve_python() {
    if [[ -n "${_PYTHON}" ]]; then
        printf '%s\n' "${_PYTHON}"
        return 0
    fi

    local _candidates=()
    local _venv

    # 1. GR00T venv (most likely to have numpy + project deps)
    _venv="${GR00T_VENV:-${REPO_ROOT}/policy/GR00T_n15_Tactile_Cross/.venv}"
    [[ -f "${_venv}/bin/python" ]] && _candidates+=("${_venv}/bin/python")

    # 2. dex2bench venv (has numpy)
    _venv="${DEX2BENCH_VENV:-./.venv}"
    [[ -f "${_venv}/bin/python" ]] && _candidates+=("${_venv}/bin/python")

    # 3. System python3 / python (may be conda or system)
    command -v python3 >/dev/null 2>&1 && _candidates+=(python3)
    command -v python  >/dev/null 2>&1 && _candidates+=(python)

    local _py
    for _py in "${_candidates[@]}"; do
        if "${_py}" -c "import numpy" 2>/dev/null; then
            _PYTHON="${_py}"
            printf '%s\n' "${_PYTHON}"
            return 0
        fi
    done

    die "No working Python with numpy found. Tried: ${_candidates[*]}. Set GR00T_VENV or DEX2BENCH_VENV, or activate a conda env with numpy."
}

# ── global cleanup: ensure no zombie server is left behind ──────────────────

_ACTIVE_SERVER_PID=""

_global_cleanup() {
    if [[ -n "${_ACTIVE_SERVER_PID}" ]] && kill -0 "${_ACTIVE_SERVER_PID}" 2>/dev/null; then
        echo -e "\033[31m[cleanup] Killing server PID=${_ACTIVE_SERVER_PID} (signal/exit)\033[0m"
        kill "${_ACTIVE_SERVER_PID}" 2>/dev/null || true
        wait "${_ACTIVE_SERVER_PID}" 2>/dev/null || true
        _ACTIVE_SERVER_PID=""
    fi
}

trap _global_cleanup EXIT INT TERM

# ── early --tmux detection: re-exec inside tmux if requested and not already there

_USE_TMUX=false
_USE_SII=false
_TMUX_SESSION_NAME=""
_tmux_scan_args=("$@")
while [[ ${#_tmux_scan_args[@]} -gt 0 ]]; do
    case "${_tmux_scan_args[0]}" in
        --tmux) _USE_TMUX=true ;;
        --tmux-session-name)
            if [[ ${#_tmux_scan_args[@]} -ge 2 && "${_tmux_scan_args[1]}" != --* ]]; then
                _TMUX_SESSION_NAME="${_tmux_scan_args[1]}"
                _tmux_scan_args=("${_tmux_scan_args[@]:1}")
            fi ;;
        --tmux-session-name=*)
            _TMUX_SESSION_NAME="${_tmux_scan_args[0]#*=}" ;;
    esac
    _tmux_scan_args=("${_tmux_scan_args[@]:1}")
done

if [[ "${_USE_TMUX}" == "true" ]] && [[ -z "${TMUX:-}" ]]; then
    _self="$(readlink -f "$0")"
    _filtered_args=()
    for _arg in "${TASK_ID}" "$@"; do
        case "$_arg" in
            --tmux) continue ;;
            --tmux-session-name) continue ;;
            --tmux-session-name=*) continue ;;
            *) _filtered_args+=("$_arg") ;;
        esac
    done

    _sess_name="${_TMUX_SESSION_NAME:-gr00t_e${TASK_ID}_$(date +%m%d_%H%M)}"
    info "Re-launching inside tmux session: ${_sess_name}"

    _inner_cmd="source ~/.bashrc 2>/dev/null || true; cd ${REPO_ROOT} && exec bash ${_self}"
    for _a in "${_filtered_args[@]}"; do
        _inner_cmd="${_inner_cmd} ${_a}"
    done
    exec tmux new-session -s "${_sess_name}" "${_inner_cmd}"
    exit 1
fi

_normalize_channel() {
    local raw="${1:-none}"
    raw="$(echo "$raw" | tr '[:upper:]' '[:lower:]')"
    case "$raw" in
        ""|none)                     echo "none" ;;
        cov|cov_only|cov-only)       echo "cov_only" ;;
        inv|inv_only|inv-only)       echo "inv_only" ;;
        inv_cov|inv+cov|full)        echo "inv_cov" ;;
        all)                         echo "all" ;;
        *) die "Unknown channel: $raw (expected: none/cov/inv/inv_cov/all)" ;;
    esac
}

_apply_channel_selection() {
    if [[ -n "${CHANNEL_NUM:-}" ]]; then
        case "${CHANNEL_NUM}" in
            1)
                CHANNEL="channel_num_1"
                CHANNEL_LIST=(inv_cov)
                CHANNEL_MODE_TOKEN="inv_cov"
                MULTI_CHANNEL=false
                ;;
            2)
                CHANNEL="channel_num_2"
                CHANNEL_LIST=(inv_only inv_cov)
                CHANNEL_MODE_TOKEN="ch2"
                MULTI_CHANNEL=true
                ;;
            3)
                CHANNEL="channel_num_3"
                CHANNEL_LIST=(cov_only inv_only inv_cov)
                CHANNEL_MODE_TOKEN="ch3"
                MULTI_CHANNEL=true
                ;;
            4)
                CHANNEL="channel_num_4"
                CHANNEL_LIST=(none cov_only inv_only inv_cov)
                CHANNEL_MODE_TOKEN="all"
                MULTI_CHANNEL=true
                ;;
            *) die "--channel_num must be one of 1/2/3/4, got: ${CHANNEL_NUM}" ;;
        esac
        return
    fi

    if [[ "${CHANNEL}" == "all" ]]; then
        CHANNEL_LIST=(none cov_only inv_only inv_cov)
        CHANNEL_MODE_TOKEN="all"
        MULTI_CHANNEL=true
    else
        CHANNEL_LIST=("${CHANNEL}")
        CHANNEL_MODE_TOKEN="${CHANNEL}"
        MULTI_CHANNEL=false
    fi
}

_short_robot() {
    # "multi_ur5_rh5dg2_with_flange" -> "rh5dg2"
    local key="$1"
    local short="${key#multi_}"
    short="${short//_with_flange/}"
    short="${short//_with_leap/}"
    short="${short//ur5_/}"
    short="${short//ur5e_/}"
    echo "${short}"
}

_shopt_state() {
    if shopt -q "$1"; then echo "on"; else echo "off"; fi
}

_restore_shopt() {
    local _name="$1" _state="$2"
    if [[ "${_state}" == "on" ]]; then shopt -s "${_name}"; else shopt -u "${_name}"; fi
}

_require_value() {
    if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
        die "Missing value for $1"
    fi
}

_parse_option_value() {
    local arg="$1"
    if [[ "${arg}" == *=* ]]; then
        local value="${arg#*=}"
        [[ -n "${value}" ]] || die "Missing value for ${arg%%=*}"
        printf '%s\n' "${value}"
    else
        _require_value "${arg}" "${2:-}"
        printf '%s\n' "$2"
    fi
}

# ── phase 1: resolve task name from scenes/ ──────────────────────────────────

TASK_GLOB=(scenes/${TASK_ID}_*.yaml)
if [[ ! -f "${TASK_GLOB[0]}" ]]; then
    die "No scene file found matching: scenes/${TASK_ID}_*.yaml"
fi
SCENE_FILE="${TASK_GLOB[0]}"
TASK_NAME="$(basename "${SCENE_FILE}" .yaml)"
info "Task: ${TASK_NAME} (${SCENE_FILE})"

# ── phase 2: resolve channel ─────────────────────────────────────────────────

CHANNEL="none"
CHANNEL_NUM=""
CHANNEL_MODE_TOKEN="none"
MULTI_CHANNEL=false
if [[ $# -gt 0 && "$1" != --* ]]; then
    CHANNEL="$(_normalize_channel "$1")"
    shift
fi
_apply_channel_selection
info "Channel: ${CHANNEL_MODE_TOKEN} (${CHANNEL_LIST[*]})"

# ── phase 3: parse --model_path / --robot_key early (before auto-discover) ───

MODEL_PATH="${GR00T_MODEL_PATH:-}"
ROBOT_KEY="${GR00T_ROBOT_KEY:-}"

# Scan remaining args for --model_path and --robot_key before auto-discovery,
# so the eval dir reflects the user's explicit choice.
_remaining_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path|--model-path)
            MODEL_PATH="$2"; shift 2 ;;
        --robot_key|--robot-key)
            ROBOT_KEY="$2"; shift 2 ;;
        *)
            _remaining_args+=("$1")
            if [[ $# -gt 1 && "$2" != --* ]]; then
                _remaining_args+=("$2"); shift
            fi
            shift ;;
    esac
done
set -- "${_remaining_args[@]}"

# ── phase 4: auto-discover checkpoint ────────────────────────────────────────

# Look only for gr00t_n15_*tactile_cross* under policy_ckpt/<TASK_ID>/*/. The helper
# prefers the exact basename, then variants and their checkpoint-* children.
if [[ -z "${MODEL_PATH}" ]]; then
    MODEL_PATH="$(tactile_checkpoint_discover "${POLICY_CKPT_ROOT}" "${TASK_ID}")"
fi
if [[ -z "${MODEL_PATH}" ]]; then
    die "No valid Cross tactile GR00T checkpoint found under ${POLICY_CKPT_ROOT}/${TASK_ID}/*/gr00t_n15_*tactile_cross*"
fi
info "Model: ${MODEL_PATH}"

# ── phase 5: resolve robot key and use_active_dof ────────────────────────────

# Priority: explicit --robot_key override > robot_key.txt in ckpt > dir name
if [[ -z "${ROBOT_KEY}" ]]; then
    if [[ -f "${MODEL_PATH}/robot_key.txt" ]]; then
        ROBOT_KEY="$(tr -d '[:space:]' < "${MODEL_PATH}/robot_key.txt")"
    else
        ROBOT_KEY="$(tactile_checkpoint_robot_key "${MODEL_PATH}")"
    fi
fi
info "Robot: ${ROBOT_KEY}"

# Auto-detect use_active_dof: GR00T always requires active DOF.
USE_ACTIVE_DOF="${GR00T_USE_ACTIVE_DOF:-true}"
info "use_active_dof=${USE_ACTIVE_DOF}"

# ── phase 6: resolve anchor HDF5 (per-channel override still possible) ──────

# anchor is resolved per-channel later; set a default path for channels that need it
DEFAULT_ANCHOR_DIR="${TELEOPDATA_ROOT}/${TASK_NAME}/replay-generalization"
ANCHOR_DIR_USER="${GR00T_ANCHOR_DIR:-}"  # user may override via env or --anchor-dir flag

# ── phase 7: output naming ───────────────────────────────────────────────────

SHORT_ROBOT="$(_short_robot "${ROBOT_KEY}")"
CKPT_BASENAME="$(basename "${MODEL_PATH}")"
TIMESTAMP="$(date +%m%d_%H%M)"
RECORD_ALL="${RECORD_ALL:-true}"
# EVAL_ROOT is resolved after option parsing because --channel may appear below.

# ── phase 8: parse remaining args ────────────────────────────────────────────

EXTRA_OVERRIDES=()
RUN_POLICY_EXTRA=()
EVAL_REMAINING_ARGS=()
START_EPISODE="${START_EPISODE:-}"
EPISODES_PER_PROCESS="${EPISODES_PER_PROCESS:-25}"
CUDA_DEVICE=""
RESUME_ENABLED=false
RESUME_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --anchor-dir|--anchor-dir)
            _require_value "$1" "${2:-}"
            ANCHOR_DIR_USER="$2"; shift 2 ;;
        --model_path|--model-path)
            warn "--model_path should be passed before other flags; MODEL_PATH already set, ignoring"
            shift 2 ;;
        --robot_key|--robot-key)
            warn "--robot_key should be passed before other flags; ROBOT_KEY already set, ignoring"
            shift 2 ;;
        --use_active_dof|--use-active-dof)
            _require_value "$1" "${2:-}"
            USE_ACTIVE_DOF="$2"; shift 2 ;;
        --channel)
            _require_value "$1" "${2:-}"
            CHANNEL="$(_normalize_channel "$2")"
            _apply_channel_selection
            shift 2 ;;
        --channel_num|--channel-num)
            CHANNEL_NUM="$(_parse_option_value "$1" "${2:-}")"
            _apply_channel_selection
            shift 2 ;;
        --channel_num=*|--channel-num=*)
            CHANNEL_NUM="$(_parse_option_value "$1")"
            _apply_channel_selection
            shift ;;
        --resume)
            RESUME_ENABLED=true
            shift ;;
        --resume_dir|--resume-dir)
            RESUME_ENABLED=true
            RESUME_DIR="$(_parse_option_value "$1" "${2:-}")"
            shift 2 ;;
        --resume_dir=*|--resume-dir=*)
            RESUME_ENABLED=true
            RESUME_DIR="$(_parse_option_value "$1")"
            shift ;;
        --cuda|--cuda-devices|--gpu)
            CUDA_DEVICE="$(_parse_option_value "$1" "${2:-}")"
            shift 2 ;;
        --cuda=*|--cuda-devices=*|--gpu=*)
            CUDA_DEVICE="$(_parse_option_value "$1")"
            shift ;;
        --record_all|--record-all)
            RECORD_ALL=true
            RUN_POLICY_EXTRA+=("--record-all")
            shift ;;
        --episodes_per_process|--episodes-per-process|\
        --isaac_restart_every|--isaac-restart-every|\
        --restart_every|--restart-every|\
        --chunk_episodes|--chunk-episodes)
            EPISODES_PER_PROCESS="$(_parse_option_value "$1" "${2:-}")"
            shift 2 ;;
        --episodes_per_process=*|--episodes-per-process=*|\
        --isaac_restart_every=*|--isaac-restart-every=*|\
        --restart_every=*|--restart-every=*|\
        --chunk_episodes=*|--chunk-episodes=*)
            EPISODES_PER_PROCESS="$(_parse_option_value "$1")"
            shift ;;
        --num_episodes|--num-episodes)
            NUM_EPISODES="$(_parse_option_value "$1" "${2:-}")"
            shift 2 ;;
        --num_episodes=*|--num-episodes=*)
            NUM_EPISODES="$(_parse_option_value "$1")"
            shift ;;
        --start_episode|--start-episode)
            START_EPISODE="$(_parse_option_value "$1" "${2:-}")"
            shift 2 ;;
        --start_episode=*|--start-episode=*)
            START_EPISODE="$(_parse_option_value "$1")"
            shift ;;
        --episode_steps|--episode-steps|\
        --warmup_steps|--warmup-steps|\
        --seed|\
        --record_dir|--record-dir|\
        --output_dir|--output-dir|\
        --generalization_profile|--generalization-profile|\
        --generalization_split|--generalization-split|\
        --generalization_config|--generalization-config)
            _require_value "$1" "${2:-}"
            EXTRA_OVERRIDES+=("$1" "$2"); shift 2 ;;
        --sii) _USE_SII=true; shift ;;
        --fd)
            USE_UV=true; shift ;;
        --tmux) shift ;;  # handled by early detection; no-op here
        --tmux-session-name)
            _require_value "$1" "${2:-}"
            shift 2 ;;
        --tmux-session-name=*)
            shift ;;
        --*)
            RUN_POLICY_EXTRA+=("$1")
            if [[ $# -gt 1 && "$2" != --* ]]; then
                RUN_POLICY_EXTRA+=("$2"); shift
            fi
            shift ;;
        *)
            if [[ $# -gt 1 && "$2" != --* ]]; then
                EXTRA_OVERRIDES+=("$1" "$2"); shift 2
            else
                RUN_POLICY_EXTRA+=("$1"); shift
            fi ;;
    esac
done

_apply_channel_selection
# Tactile GR00T eval outputs go under zdj/output_zdj/TactileGr00t_Cross/<TASK_ID>/
# (previously ${REPO_ROOT}/../output/end_eval/${TASK_ID}). Derived from REPO_ROOT
# so the repo stays relocatable; resolves to an absolute path at runtime.
EVAL_ROOT_PARENT="${REPO_ROOT}/../zdj/output_zdj/TactileGr00t_Cross/${TASK_ID}"
EVAL_NAME_PREFIX="gr00t_${CHANNEL_MODE_TOKEN}_"
EVAL_ROOT="${EVAL_ROOT_PARENT}/${EVAL_NAME_PREFIX}${TIMESTAMP}_${SHORT_ROBOT}"
info "Channel: ${CHANNEL_MODE_TOKEN} (${CHANNEL_LIST[*]})"
info "record_all=${RECORD_ALL}"
if [[ "${USE_ACTIVE_DOF}" != "true" ]]; then
    die "GR00T evaluation requires use_active_dof=true, got '${USE_ACTIVE_DOF}'. Aborting."
fi

# ── budget integration ───────────────────────────────────────────────────────

GPU_ID="${CUDA_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
SEED="${SEED:-100000000}"
NUM_EPISODES="${NUM_EPISODES:-50}"
START_EPISODE="${START_EPISODE:-1}"
if ! [[ "${NUM_EPISODES}" =~ ^[0-9]+$ ]] || [[ "${NUM_EPISODES}" -le 0 ]]; then
    die "--num-episodes must be a positive integer, got: ${NUM_EPISODES}"
fi
if ! [[ "${START_EPISODE}" =~ ^[0-9]+$ ]] || [[ "${START_EPISODE}" -le 0 ]]; then
    die "--start-episode must be a positive integer, got: ${START_EPISODE}"
fi
if ! [[ "${EPISODES_PER_PROCESS}" =~ ^[0-9]+$ ]]; then
    die "--episodes-per-process must be a non-negative integer, got: ${EPISODES_PER_PROCESS}"
fi

source script/eval_budget.sh
eval_budget_parse_options "${EXTRA_OVERRIDES[@]}"
eval_budget_compute "${TASK_NAME}" "${EVAL_EPISODE_STEPS_OVERRIDE}" "${EVAL_MAX_STEPS_OVERRIDE}"
eval_budget_log

_do_isaac_restart=false
if [[ "${EPISODES_PER_PROCESS}" -gt 0 && "${EPISODES_PER_PROCESS}" -lt "${NUM_EPISODES}" ]]; then
    _do_isaac_restart=true
fi
info "Episodes: start=${START_EPISODE} num=${NUM_EPISODES} episodes-per-process=${EPISODES_PER_PROCESS} isaac-restart=${_do_isaac_restart}"

# ── GR00T-specific env setup (computed once, used inside _run_one_process) ───

export PYTHONPATH="${REPO_ROOT}/policy/GR00T_n15_Tactile_Cross/src:${PYTHONPATH:-}"
CKPT_NAME="${GR00T_CKPT_NAME:-gr00t_n15_tactile_cross}"
# Display policy name: <task>/<robot>/<ckpt_dir>/<ckpt_name> relative to policy_ckpt
GR00T_DISPLAY_POLICY_NAME="${TASK_ID}/${ROBOT_KEY}/${CKPT_BASENAME}/${CKPT_NAME}"

_yaml_scalar() {
    local key="$1"
    local value=""
    local config_path="${REPO_ROOT}/policy/GR00T_n15_Tactile_Cross/deploy_policy.yml"
    value=$(grep -m1 -E "^[[:space:]]*${key}:" "${config_path}" | sed -E "s/^[[:space:]]*${key}:[[:space:]]*//; s/[[:space:]]+#.*$//; s/^[[:space:]]+//; s/[[:space:]]+$//" || true)
    [[ "${value}" == "null" ]] && value=""
    printf '%s\n' "${value}"
}

STATE_DIM=${GR00T_STATE_DIM:-$(_yaml_scalar state_dim)}
ACTION_DIM=${GR00T_ACTION_DIM:-$(_yaml_scalar action_dim)}
MAX_STATE_DIM=${GR00T_MAX_STATE_DIM:-$(_yaml_scalar max_state_dim)}
MAX_ACTION_DIM=${GR00T_MAX_ACTION_DIM:-$(_yaml_scalar max_action_dim)}
MAX_STATE_DIM=${MAX_STATE_DIM:-64}
MAX_ACTION_DIM=${MAX_ACTION_DIM:-64}
[[ -n "${STATE_DIM}" && "${STATE_DIM}" -gt "${MAX_STATE_DIM}" ]] && MAX_STATE_DIM="${STATE_DIM}"
[[ -n "${ACTION_DIM}" && "${ACTION_DIM}" -gt "${MAX_ACTION_DIM}" ]] && MAX_ACTION_DIM="${ACTION_DIM}"

# ── server overrides template (profile-independent) ──────────────────────────

SERVER_OVERRIDES_TMPL=(
    --policy_name GR00T_n15_Tactile_Cross
    --task_name "${TASK_NAME}"
    --model_path "${MODEL_PATH}"
    --ckpt_dir "${MODEL_PATH}"
    --ckpt_name "${CKPT_NAME}"
    --seed "${SEED}"
    --use_active_dof "${USE_ACTIVE_DOF}"
    --robot_key "${ROBOT_KEY}"
)
for ((i=0; i<${#EXTRA_OVERRIDES[@]}; i+=2)); do
    SERVER_OVERRIDES_TMPL+=("${EXTRA_OVERRIDES[$i]}" "${EXTRA_OVERRIDES[$((i+1))]}")
done

# ── helper: resolve anchor HDF5 for a given channel ─────────────────────────

_resolve_anchor_for_channel() {
    local _ch="$1"
    # inv_cov does not need anchor
    if [[ "${_ch}" == "inv_cov" ]]; then
        echo ""
        return
    fi
    # Use user-specified anchor if set, otherwise default
    local _a="${ANCHOR_DIR_USER:-${DEFAULT_ANCHOR_DIR}}"
    if [[ -z "${_a}" || ! -d "${_a}" ]]; then
        die "Anchor dir not found for channel ${_ch}: ${_a}. Set --anchor_dir (replay-generalization dir)."
    fi
    echo "${_a}"
}

# ── record setup ─────────────────────────────────────────────────────────────

# Enabled whenever RECORD_ALL is true (the default).
if [[ "${RECORD_ALL}" == "true" ]]; then
    RECORD_DIR_TMPL="${RECORD_DIR_TMPL:-enabled}"
else
    RECORD_DIR_TMPL=""
fi

# Add --record-all to RUN_POLICY_EXTRA by default when RECORD_ALL is true
if [[ "${RECORD_ALL}" == "true" ]]; then
    _has_record_all=false
    for _arg in "${RUN_POLICY_EXTRA[@]}"; do
        if [[ "${_arg}" == "--record-all" ]]; then
            _has_record_all=true
            break
        fi
    done
    if [[ "${_has_record_all}" != "true" ]]; then
        RUN_POLICY_EXTRA+=("--record-all")
    fi
fi

# ── uv / conda mode resolution ────────────────────────────────────────────────

USE_UV="${USE_UV:-false}"

if [[ "${USE_UV}" == "true" ]]; then
    GR00T_VENV="${GR00T_VENV:-${REPO_ROOT}/policy/GR00T_n15_Tactile_Cross/.venv}"
    DEX2BENCH_VENV="${DEX2BENCH_VENV:-./.venv}"

    if [[ ! -f "${GR00T_VENV}/bin/activate" ]]; then
        die "GR00T venv not found: ${GR00T_VENV}. Set GR00T_VENV=/path/to/.venv"
    fi
    if [[ ! -f "${DEX2BENCH_VENV}/bin/activate" ]]; then
        die "dex2bench venv not found: ${DEX2BENCH_VENV}. Set DEX2BENCH_VENV=/path/to/.venv"
    fi
    info "uv mode: GR00T_VENV=${GR00T_VENV}"
    info "uv mode: DEX2BENCH_VENV=${DEX2BENCH_VENV}"
fi

# Resolve a working Python (with numpy) for server readiness check, budget, etc.
_resolve_python >/dev/null
info "Resolved Python: ${_PYTHON}"

declare -A RESUME_START_BY_CHANNEL=()
declare -A RESUME_COUNT_BY_CHANNEL=()
declare -A RESUME_APPEND_BY_CHANNEL=()
declare -A RESUME_STATUS_BY_CHANNEL=()

_init_resume_defaults() {
    local _ch
    for _ch in "${CHANNEL_LIST[@]}"; do
        RESUME_START_BY_CHANNEL["${_ch}"]="${START_EPISODE}"
        RESUME_COUNT_BY_CHANNEL["${_ch}"]="${NUM_EPISODES}"
        RESUME_APPEND_BY_CHANNEL["${_ch}"]="false"
        RESUME_STATUS_BY_CHANNEL["${_ch}"]="fresh"
    done
}

_apply_resume_plan() {
    _init_resume_defaults
    if [[ "${RESUME_ENABLED}" != "true" ]]; then
        info "Eval root: ${EVAL_ROOT}"
        return
    fi

    local _cmd=(
        script/eval_resume.py plan
        --eval-root-parent "${EVAL_ROOT_PARENT}"
        --name-prefix "${EVAL_NAME_PREFIX}"
        --robot-suffix "${SHORT_ROBOT}"
        --target-start "${START_EPISODE}"
        --target-count "${NUM_EPISODES}"
        --channels "${CHANNEL_LIST[@]}"
    )
    if [[ "${MULTI_CHANNEL}" == "true" ]]; then
        _cmd+=(--multi-channel)
    fi
    if [[ -n "${RESUME_DIR}" ]]; then
        _cmd+=(--resume-dir "${RESUME_DIR}")
    fi

    local _plan
    if ! _plan="$("${_PYTHON}" "${_cmd[@]}" 2>&1)"; then
        die "Resume planning failed: ${_plan}"
    fi

    local _kind _field1 _field2 _field3 _field4 _field5 _field6 _field7
    while IFS=$'\t' read -r _kind _field1 _field2 _field3 _field4 _field5 _field6 _field7; do
        [[ -n "${_kind}" ]] || continue
        case "${_kind}" in
            ROOT)
                EVAL_ROOT="${_field1}"
                ;;
            CHANNEL)
                RESUME_STATUS_BY_CHANNEL["${_field1}"]="${_field2}"
                RESUME_START_BY_CHANNEL["${_field1}"]="${_field3}"
                RESUME_COUNT_BY_CHANNEL["${_field1}"]="${_field4}"
                RESUME_APPEND_BY_CHANNEL["${_field1}"]="${_field5}"
                info "Resume ${_field1}: status=${_field2} completed=${_field6}/${NUM_EPISODES} next=${_field3} remaining=${_field4}"
                ;;
            *)
                die "Unexpected resume plan line: ${_kind}"
                ;;
        esac
    done <<< "${_plan}"

    info "Resume root: ${EVAL_ROOT}"
}

_apply_resume_plan

# ══════════════════════════════════════════════════════════════════════════════
# Core: run one Isaac process — start server, run N episodes, kill server
# ══════════════════════════════════════════════════════════════════════════════

_run_one_process() {
    local _channel="$1"
    local _start_ep="$2"
    local _num_ep="$3"
    local _output_dir="$4"
    local _record_dir="$5"
    local _append_mode="$6"

    local _anchor
    _anchor="$(_resolve_anchor_for_channel "${_channel}")"

    # ── build overrides for this process ────────────────────────────────────
    local _server_overrides=("${SERVER_OVERRIDES_TMPL[@]}")
    local _client_overrides=("${SERVER_OVERRIDES_TMPL[@]}"
        --episode_steps "${EVAL_EPISODE_STEPS}"
        --warmup_steps 60
        --generalization_profile "${_channel}"
        --output_dir "${_output_dir}"
        --num_episodes "${_num_ep}"
        --start_episode "${_start_ep}"
        --policy_display_name "${GR00T_DISPLAY_POLICY_NAME}"
    )
    if [[ -n "${_anchor}" ]]; then
        _client_overrides+=(--anchor-dir "${_anchor}")
    fi
    if [[ "${_append_mode}" == "true" ]]; then
        _client_overrides+=(
            --append_output true
            --append_record_dir true
        )
    fi

    # Build run_policy_args with record-dir if set
    local _run_policy_args="["
    local __sep=""
    if [[ -n "${_record_dir}" ]]; then
        _run_policy_args+="'--record-dir', '${_record_dir}'"
        __sep=", "
    fi
    for _arg in "${RUN_POLICY_EXTRA[@]}"; do
        _run_policy_args+="${__sep}'${_arg}'"
        __sep=", "
    done
    _run_policy_args+="]"

    # ── get a free port ─────────────────────────────────────────────────────
    local _port
    _port=$("${_PYTHON}" -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

    # ── pre-start cleanup: kill any zombie servers on this GPU ────────────
    # Belt-and-suspenders: catches leaks from previous abnormal exits.
    local _zombies
    _zombies=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
    if [[ -n "${_zombies}" ]]; then
        for _zp in ${_zombies}; do
            if [[ "${_zp}" =~ ^[0-9]+$ ]] && ps -p "${_zp}" -o cmd --no-headers 2>/dev/null | grep -q "policy_model_server"; then
                warn "Pre-start cleanup: killing zombie policy server PID=${_zp}"
                kill -9 "${_zp}" 2>/dev/null || true
            fi
        done
        sleep 1  # let GPU memory settle
    fi

    # ── start server ────────────────────────────────────────────────────────
    info "Starting Isaac server on port ${_port} (channel=${_channel}, ep ${_start_ep}-$((_start_ep + _num_ep - 1)))"
    (
        set -euo pipefail
        export PS1='$ '
        if [[ "${USE_UV}" == "true" ]]; then
            source "${GR00T_VENV}/bin/activate"
        elif [[ "${_USE_SII}" == "true" ]]; then
            source ../miniconda3/bin/activate groot
        else
            eval "$(conda shell.bash hook)"
            conda activate GR00T_n15
        fi
        export CUDA_VISIBLE_DEVICES="${GPU_ID}"
        export PYTHONPATH="${REPO_ROOT}/policy/GR00T_n15_Tactile_Cross/src:${PYTHONPATH:-}"
        export GR00T_MAX_STATE_DIM="${MAX_STATE_DIM}"
        export GR00T_MAX_ACTION_DIM="${MAX_ACTION_DIM}"
        PYTHONWARNINGS=ignore::UserWarning \
        exec python script/policy_model_server.py \
            --host 127.0.0.1 \
            --port "${_port}" \
            --config policy/GR00T_n15_Tactile_Cross/deploy_policy.yml \
            --overrides \
            "${_server_overrides[@]}"
    ) &
    local _server_pid=$!
    _ACTIVE_SERVER_PID="${_server_pid}"

    _cleanup_server() {
        if [[ -z "${_server_pid}" ]] || ! kill -0 "${_server_pid}" 2>/dev/null; then
            if [[ "${_ACTIVE_SERVER_PID}" == "${_server_pid}" ]]; then
                _ACTIVE_SERVER_PID=""
            fi
            return 0
        fi
        echo -e "\033[31m[cleanup] Stopping server PID=${_server_pid}\033[0m"
        # 1. SIGTERM — give the process a chance to shut down gracefully
        kill "${_server_pid}" 2>/dev/null || true
        # 2. Wait up to 5s for graceful exit
        local _waited=0
        while kill -0 "${_server_pid}" 2>/dev/null && [[ "${_waited}" -lt 5 ]]; do
            sleep 1
            _waited=$((_waited + 1))
        done
        # 3. SIGKILL — force-kill the process group if still alive
        if kill -0 "${_server_pid}" 2>/dev/null; then
            echo -e "\033[31m[cleanup] Force-killing server PID=${_server_pid} (process group)\033[0m"
            kill -9 -"${_server_pid}" 2>/dev/null || true   # negative PID = process group
            kill -9 "${_server_pid}" 2>/dev/null || true
            sleep 1
        fi
        # 4. Wait for the zombie to be reaped
        wait "${_server_pid}" 2>/dev/null || true
        if [[ "${_ACTIVE_SERVER_PID}" == "${_server_pid}" ]]; then
            _ACTIVE_SERVER_PID=""
        fi
    }

    # ── wait for server ─────────────────────────────────────────────────────
    local _ready=0
    for _i in {1..600}; do
        if "${_PYTHON}" -c "from script.policy_rpc import RemotePolicyClient; c=RemotePolicyClient('127.0.0.1', ${_port}, timeout_s=1); c.close()" >/dev/null 2>&1; then
            _ready=1
            break
        fi
        if ! kill -0 "${_server_pid}" 2>/dev/null; then
            echo -e "\033[31m[error] policy server exited before opening port ${_port}\033[0m" >&2
            _ACTIVE_SERVER_PID=""
            return 1
        fi
        sleep 1
    done
    if [[ "${_ready}" != "1" ]]; then
        echo -e "\033[31m[error] Timed out waiting for policy server on port ${_port}\033[0m" >&2
        _cleanup_server
        return 1
    fi

    # ── launch client ───────────────────────────────────────────────────────
    export PS1='$ '
    if [[ "${USE_UV}" == "true" ]]; then
        source "${DEX2BENCH_VENV}/bin/activate"
    elif [[ "${_USE_SII}" == "true" ]]; then
        source ../isaaclab_setup/env_new.sh
        start_xvfb
        activate_isaaclab
        setup_nvidia_libs
    else
        eval "$(conda shell.bash hook)"
        conda activate dex2bench
    fi

    local _end_ep=$((_start_ep + _num_ep - 1))
    info "Client: channel=${_channel} episodes ${_start_ep}-${_end_ep} count=${_num_ep} append=${_append_mode}"

    local _client_rc=0
    PYTHONWARNINGS=ignore::UserWarning \
    python policy/GR00T_n15_Tactile_Cross/tactile_eval_client.py \
        --host 127.0.0.1 \
        --port "${_port}" \
        --config policy/GR00T_n15_Tactile_Cross/deploy_policy.yml \
        --overrides \
        "${_client_overrides[@]}" \
        --run_policy_args "${_run_policy_args}" \
        "${EVAL_REMAINING_ARGS[@]}" || _client_rc=$?

    # ── kill server ─────────────────────────────────────────────────────────
    _cleanup_server

    return "${_client_rc}"
}

# ══════════════════════════════════════════════════════════════════════════════
# Run one channel with optional per-process restart
# ══════════════════════════════════════════════════════════════════════════════

_run_channel() {
    local _channel="$1"
    local _channel_dir="$2"
    local _start_episode="$3"
    local _num_episodes="$4"
    local _initial_append="$5"

    mkdir -p "${_channel_dir}"

    info "===== Channel: ${_channel} ====="

    if [[ "${_num_episodes}" -le 0 ]]; then
        info "Channel ${_channel}: already complete; skipping"
        return 0
    fi

    if [[ "${_do_isaac_restart}" == "true" ]]; then
        # Chunked mode: restart Isaac between chunks, but keep one canonical
        # channel output directory. run_policy.py owns append/truncate semantics.
        local _chunk_start="${_start_episode}"
        local _remaining="${_num_episodes}"
        local _chunk_idx=0
        local _record_dir=""
        if [[ -n "${RECORD_DIR_TMPL:-}" ]]; then
            _record_dir="${_channel_dir}/episodes"
        fi

        while [[ "${_remaining}" -gt 0 ]]; do
            local _chunk_count="${EPISODES_PER_PROCESS}"
            if [[ "${_chunk_count}" -gt "${_remaining}" ]]; then
                _chunk_count="${_remaining}"
            fi
            local _chunk_end=$((_chunk_start + _chunk_count - 1))
            local _append_mode="false"
            if [[ "${_initial_append}" == "true" || "${_chunk_idx}" -gt 0 ]]; then
                _append_mode="true"
            fi

            info "Process $((_chunk_idx + 1)): channel=${_channel} episodes ${_chunk_start}-${_chunk_end} output=${_channel_dir}"
            if ! _run_one_process "${_channel}" "${_chunk_start}" "${_chunk_count}" \
                "${_channel_dir}" "${_record_dir}" "${_append_mode}"; then
                warn "Channel ${_channel}: process $((_chunk_idx + 1)) failed (episodes ${_chunk_start}-${_chunk_end})"
                return 1
            fi

            _chunk_start=$((_chunk_end + 1))
            _remaining=$((_remaining - _chunk_count))
            _chunk_idx=$((_chunk_idx + 1))
        done
    else
        # ── single-process mode: all episodes in one Isaac process ──────────
        local _record_dir=""
        if [[ -n "${RECORD_DIR_TMPL:-}" ]]; then
            _record_dir="${_channel_dir}/episodes"
        fi
        if ! _run_one_process "${_channel}" "${_start_episode}" "${_num_episodes}" \
            "${_channel_dir}" "${_record_dir}" "${_initial_append}"; then
            warn "Channel ${_channel} failed"
            return 1
        fi
    fi

    info "Channel ${_channel}: done"
    return 0
}

# ══════════════════════════════════════════════════════════════════════════════
# Aggregate: compute per-channel success rates and write TSV + CSV
# ══════════════════════════════════════════════════════════════════════════════

_compute_success_rate() {
    local _per_episode="$1"
    if [[ ! -f "${_per_episode}" ]]; then
        echo "-1\tn/a"
        return
    fi
    "${_PYTHON}" -c "
import json, sys
rows = []
with open('${_per_episode}') as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
total = len(rows)
successes = sum(1 for r in rows if r.get('success'))
rate = successes / total if total > 0 else 0.0
print(f'{total}\t{successes}\t{rate:.6f}\t{rate*100:.2f}%')
"
}

_aggregate_results() {
    local _root="$1"

    # ── Simple success-rate TSV (quick overview) ────────────────────────────
    local _tsv="${_root}/success_rates.tsv"
    printf 'channel\tepisodes\tsuccesses\tsuccess_rate\tpct\toutput_dir\n' > "${_tsv}"
    local _overall_total=0 _overall_success=0
    for _ch in "${CHANNEL_LIST[@]}"; do
        local _ch_dir="${_root}/${_ch}"
        local _stats
        _stats="$(_compute_success_rate "${_ch_dir}/per_episode.jsonl")"
        printf '%s\t%s\t%s\n' "${_ch}" "${_stats}" "${_ch_dir}" >> "${_tsv}"
        local _t _s
        _t=$(echo "${_stats}" | cut -f1)
        _s=$(echo "${_stats}" | cut -f2)
        if [[ "${_t}" =~ ^[0-9]+$ ]]; then _overall_total=$((_overall_total + _t)); fi
        if [[ "${_s}" =~ ^[0-9]+$ ]]; then _overall_success=$((_overall_success + _s)); fi
    done
    local _overall_rate="n/a"
    if [[ "${_overall_total}" -gt 0 ]]; then
        _overall_rate="$("${_PYTHON}" -c "import sys; s=int(sys.argv[1]); t=int(sys.argv[2]); print(f'{s}/{t} = {s/t*100:.2f}%')" "${_overall_success}" "${_overall_total}")"
    fi
    # This mixes baseline and three perturbation distributions.  Keep it only
    # as an explicit diagnostic count; it is neither overall task performance
    # nor a robustness metric.
    printf 'MIXED_DIAGNOSTIC\t%d\t%d\t%s\t%s\n' "${_overall_total}" "${_overall_success}" "${_overall_rate}" "" >> "${_tsv}"

    # ── Full metric comparison CSV ──────────────────────────────────────────
    local _csv="${_root}/metrics_comparison.csv"
    "${_PYTHON}" - "${_root}" "${_tsv}" "${_csv}" "${CHANNEL_LIST[@]}" <<'PY'
import csv, json, os, sys

root = sys.argv[1]
tsv_path = sys.argv[2]
csv_path = sys.argv[3]
channels = sys.argv[4:]

# Metric priority: lower number = higher priority (appears first)
# Format: "metric_path" -> priority
# Metric paths are flattened: "group.key" from core_metrics / diagnostic_metrics
METRIC_PRIORITY = {
    # ── Priority 1: Completion ──
    "completion.success_rate":                     1,
    # ── Priority 2: Efficiency ──
    "efficiency.task_efficiency":                  2,
    "efficiency.avg_time_to_success_s":            2,
    "efficiency.avg_steps_to_success":             2,
    "efficiency.avg_policy_steps_to_success":      2,
    "efficiency.avg_policy_queries_to_success":    2,
    "efficiency.expert_time_s":                    3,
    # ── Priority 3: Progress ──
    "progress.latched_stage_completion_rate":      3,
    "progress.current_stage_completion_rate":      4,
    "progress.latched_chain_depth_progress_score": 4,
    "progress.current_chain_depth_progress_score": 4,
    "progress.avg_latched_chain_depth":            4,
    "progress.avg_current_chain_depth":            4,
    # ── Priority 4: Safety ──
    "safety.safe_success_rate":                    5,
    "safety.hard_violation_rate":                  5,
    "safety.drop_rate":                            6,
    "safety.high_speed_violation_rate":            6,
    "safety.robot_constraint_violation_rate":      6,
    "safety.safety_violation_step_rate":           7,
    "safety.safety_violation_events_per_step":     7,
    # ── Priority 5: Robustness ──
    "robustness.baseline_success_rate":            8,
    "robustness.overall_perturbed_success_rate":   8,
    "robustness.robust_ratio":                     8,
    "robustness.baseline_n":                       9,
    "robustness.perturbed_n":                      9,
    # ── Priority 6: Robot motion ──
    "robot_motion.joint_acceleration_rms":         10,
    "robot_motion.joint_effort_rms":               10,
    "robot_motion.joint_jerk_rms":                 10,
    "robot_motion.joint_velocity_rms":             10,
    # ── Priority 7: Grasp ──
    "grasp.kinematic_grasp_stability_index":       11,
    "grasp.mean_kinematic_grasp_stability":        11,
    # ── Priority 8: Tool / Runtime / Diagnostics ──
    "tool.tool_selection_accuracy":                12,
    "tool.tool_switch_success_rate":               12,
    "runtime.avg_steps":                           13,
    "runtime.error_count":                         13,
    "completion.ever_instant_success_rate":        14,
    "completion.at_end_success_rate":              14,
}


def flatten_metrics(metrics_dict, prefix=""):
    """Recursively flatten a nested dict of metrics into dot-separated keys."""
    result = {}
    for key, val in metrics_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict) and not any(isinstance(v, (list, dict)) for v in val.values()):
            # Shallow dict with scalar values -> flatten
            for sub_k, sub_v in val.items():
                result[f"{full_key}.{sub_k}"] = sub_v
        elif isinstance(val, dict):
            result.update(flatten_metrics(val, full_key))
        elif not isinstance(val, (list, dict)):
            result[full_key] = val
    return result


def metric_priority(key):
    """Return priority for a metric key."""
    # Direct match
    if key in METRIC_PRIORITY:
        return METRIC_PRIORITY[key]
    # Try stripping the source prefix (core_metrics. / diagnostic_metrics.)
    for prefix in ["core_metrics.", "diagnostic_metrics."]:
        if key.startswith(prefix):
            short = key[len(prefix):]
            if short in METRIC_PRIORITY:
                return METRIC_PRIORITY[short]
    # Default low priority
    return 99


# Collect all metrics from all channels
channel_data = {}
all_metric_keys = set()

for ch in channels:
    ch_dir = os.path.join(root, ch)
    summary_path = os.path.join(ch_dir, "summary.json")
    if not os.path.isfile(summary_path):
        channel_data[ch] = {}
        continue

    with open(summary_path) as f:
        summary = json.load(f)

    metrics = {}
    for section in ["core_metrics", "diagnostic_metrics"]:
        if section in summary:
            flat = flatten_metrics(summary[section], section)
            metrics.update(flat)

    # Also include metadata fields (n_episodes, successes)
    meta = summary.get("metadata", {})
    for k in ["n_episodes", "successes", "error_count"]:
        if k in meta:
            metrics[f"metadata.{k}"] = meta[k]

    channel_data[ch] = metrics
    all_metric_keys.update(metrics.keys())


# Sort keys by priority, then alphabetically within same priority
def sort_key(k):
    return (metric_priority(k), k)

sorted_keys = sorted(all_metric_keys, key=sort_key)

# Write CSV
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["metric", "priority"] + channels
    writer.writerow(header)

    for key in sorted_keys:
        prio = metric_priority(key)
        row = [key, prio]
        for ch in channels:
            val = channel_data.get(ch, {}).get(key, "")
            row.append(val)
        writer.writerow(row)

print(f"[aggregate] Full metrics CSV: {csv_path}")
PY

    # Robustness is a cross-channel quantity and must not be fabricated inside
    # any one channel summary.  When all four channels are available, compute
    # the authoritative root-level result and audit that cov_only/inv_only
    # actually preserve the factors their protocol names promise.
    local _full_channel_set=true
    local _required_channel
    for _required_channel in none cov_only inv_only inv_cov; do
        if [[ ! -f "${_root}/${_required_channel}/per_episode.jsonl" ]]; then
            _full_channel_set=false
            break
        fi
    done
    if [[ "${_full_channel_set}" == "true" ]]; then
        if ! "${_PYTHON}" tools/compute_robustness.py "${_root}" --expected-n "${NUM_EPISODES}"; then
            warn "Robustness protocol audit failed; see ${_root}/robustness_summary.json. Ratios are withheld when invalid."
        fi
    fi

    echo
    info "Success rates:"
    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "${_tsv}"
    else
        cat "${_tsv}"
    fi
    info "Summary TSV: ${_tsv}"
    info "Metrics CSV:  ${_csv}"
}

# ══════════════════════════════════════════════════════════════════════════════
# Main: iterate over channels and run
# ══════════════════════════════════════════════════════════════════════════════

mkdir -p "${EVAL_ROOT}"

# Set up record-dir template: each channel gets its own episodes/ subdirectory
# Enabled whenever RECORD_ALL is true (the default).
if [[ "${RECORD_ALL}" == "true" ]]; then
    RECORD_DIR_TMPL="${RECORD_DIR_TMPL:-enabled}"
else
    RECORD_DIR_TMPL=""
fi

# Add --record-all to RUN_POLICY_EXTRA by default when RECORD_ALL is true
if [[ "${RECORD_ALL}" == "true" ]]; then
    # only add if not already present (e.g. from CLI --record-all)
    _has_record_all=false
    for _arg in "${RUN_POLICY_EXTRA[@]}"; do
        if [[ "${_arg}" == "--record-all" ]]; then
            _has_record_all=true
            break
        fi
    done
    if [[ "${_has_record_all}" != "true" ]]; then
        RUN_POLICY_EXTRA+=("--record-all")
    fi
fi

_any_failed=false

for _current_channel in "${CHANNEL_LIST[@]}"; do
    if [[ "${MULTI_CHANNEL}" == "true" ]]; then
        _channel_dir="${EVAL_ROOT}/${_current_channel}"
    else
        _channel_dir="${EVAL_ROOT}"
    fi

    _channel_start="${RESUME_START_BY_CHANNEL[${_current_channel}]:-${START_EPISODE}}"
    _channel_count="${RESUME_COUNT_BY_CHANNEL[${_current_channel}]:-${NUM_EPISODES}}"
    _channel_append="${RESUME_APPEND_BY_CHANNEL[${_current_channel}]:-false}"
    _channel_status="${RESUME_STATUS_BY_CHANNEL[${_current_channel}]:-fresh}"

    info "Channel plan: ${_current_channel} status=${_channel_status} start=${_channel_start} count=${_channel_count} append=${_channel_append}"
    if ! _run_channel "${_current_channel}" "${_channel_dir}" "${_channel_start}" "${_channel_count}" "${_channel_append}"; then
        warn "Channel ${_current_channel} FAILED"
        _any_failed=true
        if [[ "${MULTI_CHANNEL}" != "true" ]]; then
            exit 1
        fi
        # For multi-channel runs, continue to next channel.
        continue
    fi
done

# ── aggregate if running multiple channels ───────────────────────────────────
if [[ "${MULTI_CHANNEL}" == "true" ]]; then
    _aggregate_results "${EVAL_ROOT}"

    # Append 4-channel results to every_eval_tactile.csv
    EVERY_EVAL_RESULT_FILE="/inspire/hdd/project/roboticsystem2/ky26063/bench2dex_stuff/jcy/result/every_eval_tactile.csv" \
    bash "${REPO_ROOT}/script/_append_eval_result.sh" \
        "${TASK_ID}" "${TASK_NAME}" "${EVAL_ROOT}" || true
fi

if [[ "${_any_failed}" == "true" ]]; then
    echo -e "\033[31m[FATAL] One or more channels failed\033[0m" >&2
    exit 1
fi

info "Done. Output: ${EVAL_ROOT}"
