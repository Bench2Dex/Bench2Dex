#!/bin/bash
set -euo pipefail

# Run Pi0.5 in split server/client mode for environment isolation:
#   - server: lives in policy/pi05/.venv (uv-managed jax env)
#   - client: lives in the dex2bench Isaac env
#
# Usage:
#   bash ./policy/pi05/eval_double_env.sh TASK ROBOT_KEY CKPT_DIR [train_config_name] [gpu_id] [num_episodes] [seed] [options]
#   Defaults: train_config_name=pi05_base_dex2bench_lora, gpu_id=0, num_episodes=50, seed=100000000
# bash policy/pi05/eval_double_env.sh     06_fruit_bowl_loading    multi_ur5_rh56dfx_with_flange    ../teleopdata_ckpt/ckpt/06/rh56dfx/pi05/pi05_base_dex2bench_lora/06_pi05_256/16000     pi05_base_dex2bench_lora     0     20     100000000
#
# Options:
#   --channel PROFILE                 Alias for --generalization-profile.
#                                     Accepts none/baseline, cov/cov_only, inv/inv_only, inv_cov/full.
#   --generalization-profile PROFILE  none | cov_only | inv_only | inv_cov.
#   --anchor-dir PATH                Required by none/cov_only/inv_only profiles.
#                                    Cycles replay HDF5 episodes as anchors.
#   --anchor-hdf5 PATH               Legacy alias: uses dirname(PATH) as --anchor-dir.
#   --generalization-split SPLIT      seen | unseen | all; default is unseen.
#   --generalization-config PATH      Override configs/scene/generalization.yaml.
#   --enable-generalization [BOOL]    Enable scene generalization; default is true.
#   --disable-generalization          Disable scene generalization.
#   --robot-key ROBOT_KEY             Override positional robot key.
#   --use-active-dof [BOOL]           Override use_active_dof; default is true.
#   --active-dof / --no-active-dof    Shorthand for --use-active-dof true/false.
#   --episode-steps N                 Override policy-step budget passed to run_policy.py.
#   --max-steps N                     Override physics-step budget; converted to policy steps.
#   --start-episode N                 Start episode index for deterministic batched eval.
#   --output-dir PATH                 Directory for per_episode.jsonl, per_task.json, summary.json.
#   --record-dir PATH                 Directory for successful-episode HDF5 recordings.
#   --record-all                      With --record-dir, also save failed episodes.
#   --record-success                  Auto-create a successful-episode recording directory.
#   --episodes-per-process N          Restart server/client every N episodes.
#   Dash and underscore option names are both accepted.

usage() {
    cat >&2 <<'EOF'
Usage:
  bash ./policy/pi05/eval_double_env.sh TASK ROBOT_KEY CKPT_DIR [train_config_name] [gpu_id] [num_episodes] [seed] [options]
  Defaults: train_config_name=pi05_base_dex2bench_lora, gpu_id=0, num_episodes=50, seed=100000000

Options:
  --channel PROFILE                 Alias for --generalization-profile.
                                    Accepts none/baseline, cov/cov_only, inv/inv_only, inv_cov/full.
  --generalization-profile PROFILE  none | cov_only | inv_only | inv_cov.
  --anchor-dir PATH                Required by none/cov_only/inv_only profiles.
                                    Cycles replay HDF5 episodes as anchors.
  --anchor-hdf5 PATH               Legacy alias: uses dirname(PATH) as --anchor-dir.
  --generalization-split SPLIT      seen | unseen | all; default is unseen.
  --generalization-config PATH      Override configs/scene/generalization.yaml.
  --enable-generalization [BOOL]    Enable scene generalization; default is true.
  --disable-generalization          Disable scene generalization.
  --robot-key ROBOT_KEY             Override positional robot key.
  --use-active-dof [BOOL]           Override use_active_dof; default is true.
  --active-dof / --no-active-dof    Shorthand for --use-active-dof true/false.
  --episode-steps N                 Override policy-step budget passed to run_policy.py.
  --max-steps N                     Override physics-step budget; converted to policy steps.
  --start-episode N                 Start episode index for deterministic batched eval.
  --output-dir PATH                 Directory for per_episode.jsonl, per_task.json, summary.json.
  --record-dir PATH                 Directory for successful-episode HDF5 recordings.
  --record-all                      With --record-dir, also save failed episodes.
  --record-success                  Auto-create a successful-episode recording directory.
  --episodes-per-process N          Restart server/client every N episodes.
  --sii                             Use conda envs (pi05 server, isaaclab client)
                                    instead of uv-managed venv.
  Dash and underscore option names are both accepted.
EOF
}

normalize_bool() {
    case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|y|on) echo "true" ;;
        0|false|no|n|off) echo "false" ;;
        *)
            echo "[eval] ERROR: expected boolean, got '$1'" >&2
            exit 2
            ;;
    esac
}

normalize_profile() {
    case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
        none|baseline) echo "none" ;;
        cov|cov_only|cov-only) echo "cov_only" ;;
        inv|inv_only|inv-only) echo "inv_only" ;;
        inv_cov|inv-cov|inv+cov|full) echo "inv_cov" ;;
        *)
            echo "[eval] ERROR: unsupported generalization profile/channel '$1'" >&2
            exit 2
            ;;
    esac
}

require_value() {
    if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
        echo "[eval] ERROR: missing value for $1" >&2
        usage
        exit 2
    fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -lt 2 ]]; then
    usage
    exit 2
fi

TASK=$1
ROBOT_KEY=$2
CKPT_DIR=$3
shift 3

TRAIN_CONFIG=pi05_base_dex2bench_full
GPU_ID=0
NUM_EPISODES=50
SEED=100000000

if [[ $# -gt 0 && "$1" != --* ]]; then TRAIN_CONFIG=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then GPU_ID=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then NUM_EPISODES=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then SEED=$1; shift; fi

policy_name=pi05
USE_ACTIVE_DOF=true
ENABLE_GENERALIZATION=true
GENERALIZATION_PROFILE=""
INV_COV_SEED=""
GENERALIZATION_SPLIT=unseen
GENERALIZATION_CONFIG=""
ANCHOR_DIR=""
ANCHOR_HDF5=""
EPISODE_STEPS_OVERRIDE=""
MAX_STEPS_OVERRIDE=""
START_EPISODE=""
OUTPUT_DIR=""
RECORD_DIR=""
RECORD_ALL=false
RECORD_SUCCESS=false
EPISODES_PER_PROCESS=""
_USE_SII=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --channel|--eval-channel|--eval_channel|--generalization-profile|--generalization_profile)
            require_value "$1" "${2:-}"
            GENERALIZATION_PROFILE=$(normalize_profile "$2")
            shift 2
            ;;
        --anchor-dir|--anchor_dir)
            require_value "$1" "${2:-}"
            ANCHOR_DIR=$2
            shift 2
            ;;
        --anchor-hdf5|--anchor_hdf5)
            require_value "$1" "${2:-}"
            ANCHOR_HDF5=$2
            shift 2
            ;;
        --generalization-split|--generalization_split)
            require_value "$1" "${2:-}"
            GENERALIZATION_SPLIT=$2
            shift 2
            ;;
        --inv-cov-seed|--inv_cov_seed)
            require_value "$1" "${2:-}"
            INV_COV_SEED=$2
            shift 2
            ;;
        --generalization-config|--generalization_config)
            require_value "$1" "${2:-}"
            GENERALIZATION_CONFIG=$2
            shift 2
            ;;
        --enable-generalization|--enable_generalization)
            if [[ $# -gt 1 && "$2" != --* ]]; then
                ENABLE_GENERALIZATION=$(normalize_bool "$2")
                shift 2
            else
                ENABLE_GENERALIZATION=true
                shift
            fi
            ;;
        --disable-generalization|--disable_generalization|--no-generalization|--no_generalization|--no-enable-generalization|--no_enable_generalization)
            ENABLE_GENERALIZATION=false
            shift
            ;;
        --robot-key|--robot_key)
            require_value "$1" "${2:-}"
            ROBOT_KEY=$2
            echo "[eval] robot_key overridden by --robot-key flag: ${ROBOT_KEY}" >&2
            shift 2
            ;;
        --use-active-dof|--use_active_dof)
            if [[ $# -gt 1 && "$2" != --* ]]; then
                USE_ACTIVE_DOF=$(normalize_bool "$2")
                shift 2
            else
                USE_ACTIVE_DOF=true
                shift
            fi
            ;;
        --active-dof|--active_dof)
            USE_ACTIVE_DOF=true
            shift
            ;;
        --no-active-dof|--no_active_dof)
            USE_ACTIVE_DOF=false
            shift
            ;;
        --episode-steps|--episode_steps)
            require_value "$1" "${2:-}"
            EPISODE_STEPS_OVERRIDE=$2
            shift 2
            ;;
        --episode-steps=*|--episode_steps=*)
            EPISODE_STEPS_OVERRIDE="${1#*=}"
            if [[ -z "${EPISODE_STEPS_OVERRIDE}" ]]; then
                echo "[eval] ERROR: missing value for ${1%%=*}" >&2
                usage
                exit 2
            fi
            shift
            ;;
        --max-steps|--max_steps)
            require_value "$1" "${2:-}"
            MAX_STEPS_OVERRIDE=$2
            shift 2
            ;;
        --max-steps=*|--max_steps=*)
            MAX_STEPS_OVERRIDE="${1#*=}"
            if [[ -z "${MAX_STEPS_OVERRIDE}" ]]; then
                echo "[eval] ERROR: missing value for ${1%%=*}" >&2
                usage
                exit 2
            fi
            shift
            ;;
        --start-episode|--start_episode)
            require_value "$1" "${2:-}"
            START_EPISODE=$2
            shift 2
            ;;
        --start-episode=*|--start_episode=*)
            START_EPISODE="${1#*=}"
            if [[ -z "${START_EPISODE}" ]]; then
                echo "[eval] ERROR: missing value for ${1%%=*}" >&2
                usage
                exit 2
            fi
            shift
            ;;
        --output-dir|--output_dir)
            require_value "$1" "${2:-}"
            OUTPUT_DIR=$2
            shift 2
            ;;
        --output-dir=*|--output_dir=*)
            OUTPUT_DIR="${1#*=}"
            if [[ -z "${OUTPUT_DIR}" ]]; then
                echo "[eval] ERROR: missing value for ${1%%=*}" >&2
                usage
                exit 2
            fi
            shift
            ;;
        --record-dir|--record_dir)
            require_value "$1" "${2:-}"
            RECORD_DIR=$2
            shift 2
            ;;
        --record-dir=*|--record_dir=*)
            RECORD_DIR="${1#*=}"
            if [[ -z "${RECORD_DIR}" ]]; then
                echo "[eval] ERROR: missing value for ${1%%=*}" >&2
                usage
                exit 2
            fi
            shift
            ;;
        --record-all|--record_all)
            RECORD_ALL=true
            shift
            ;;
        --no-record-all|--no_record_all)
            RECORD_ALL=false
            shift
            ;;
        --record-success|--record_success|--save-success-videos|--save_success_videos)
            RECORD_SUCCESS=true
            shift
            ;;
        --no-record-success|--no_record_success)
            RECORD_SUCCESS=false
            shift
            ;;
        --episodes-per-process|--episodes_per_process)
            require_value "$1" "${2:-}"
            EPISODES_PER_PROCESS=$2
            shift 2
            ;;
        --episodes-per-process=*|--episodes_per_process=*)
            EPISODES_PER_PROCESS="${1#*=}"
            if [[ -z "${EPISODES_PER_PROCESS}" ]]; then
                echo "[eval] ERROR: missing value for ${1%%=*}" >&2
                usage
                exit 2
            fi
            shift
            ;;
        --sii) _USE_SII=true; shift ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[eval] ERROR: unknown option '$1'" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "${ENABLE_GENERALIZATION}" ]] && {
    [[ -n "${GENERALIZATION_PROFILE}" ]] || [[ -n "${GENERALIZATION_SPLIT}" ]] ||
    [[ -n "${GENERALIZATION_CONFIG}" ]] || [[ -n "${ANCHOR_DIR}" ]] || [[ -n "${ANCHOR_HDF5}" ]]
}; then
    ENABLE_GENERALIZATION=true
fi

if [[ -z "${ANCHOR_DIR}" && -n "${ANCHOR_HDF5}" ]]; then
    if [[ ! -f "${ANCHOR_HDF5}" ]]; then
        echo "[eval] ERROR: anchor HDF5 not found: ${ANCHOR_HDF5}" >&2
        exit 2
    fi
    ANCHOR_DIR="$(dirname "${ANCHOR_HDF5}")"
fi
if [[ -n "${ANCHOR_DIR}" && ! -d "${ANCHOR_DIR}" ]]; then
    echo "[eval] ERROR: anchor directory not found: ${ANCHOR_DIR}" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../.."

source script/eval_budget.sh
eval_budget_compute "${TASK}" "${EPISODE_STEPS_OVERRIDE}" "${MAX_STEPS_OVERRIDE}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ -n "${EPISODES_PER_PROCESS}" ]]; then
    if ! [[ "${EPISODES_PER_PROCESS}" =~ ^[0-9]+$ ]] || (( EPISODES_PER_PROCESS <= 0 )); then
        echo "[eval] ERROR: --episodes-per-process must be a positive integer, got '${EPISODES_PER_PROCESS}'" >&2
        exit 2
    fi
    if ! [[ "${NUM_EPISODES}" =~ ^[0-9]+$ ]] || (( NUM_EPISODES <= 0 )); then
        echo "[eval] ERROR: --episodes-per-process requires a positive num_episodes." >&2
        exit 2
    fi
fi

if [[ -z "${START_EPISODE}" ]]; then
    START_EPISODE=1
fi

RUN_TIMESTAMP="$(date +%m%d_%H%M%S)"
TASK_ID="${TASK%%_*}"
CKPT_LABEL="$(basename "${CKPT_DIR}")"
PROFILE_LABEL="${GENERALIZATION_PROFILE:-none}"
if [[ -n "${EPISODES_PER_PROCESS}" && -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="$(cd .. && pwd)/output/metric/${TASK_ID}_${policy_name}_${CKPT_LABEL}_${PROFILE_LABEL}_${RUN_TIMESTAMP}"
fi
if [[ "${RECORD_SUCCESS}" == "true" && -z "${RECORD_DIR}" ]]; then
    if [[ -n "${OUTPUT_DIR}" ]]; then
        RECORD_DIR="${OUTPUT_DIR}/success_records"
    else
        RECORD_DIR="$(cd .. && pwd)/output/inference_recordings/${TASK}/${policy_name}_${CKPT_LABEL}_${PROFILE_LABEL}_${RUN_TIMESTAMP}"
    fi
fi

get_free_port() {
    python3 - << 'EOF'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
EOF
}

# Resolve uv for the policy server (Pi0.5 needs the isolated jax venv).
if [[ "${_USE_SII}" != "true" ]]; then
    if command -v uv >/dev/null 2>&1; then
        UV_BIN=uv
    elif [[ -x "$HOME/.local/bin/uv" ]]; then
        UV_BIN="$HOME/.local/bin/uv"
    else
        echo "[eval] ERROR: 'uv' not found. Pi0.5 needs the isolated venv built via 'uv sync'." >&2
        exit 1
    fi

    if [[ ! -x "policy/pi05/.venv/bin/python" ]]; then
        echo "[eval] ERROR: policy/pi05/.venv not found. Run first:" >&2
        echo "[eval]   cd policy/pi05 && GIT_LFS_SKIP_SMUDGE=1 uv sync" >&2
        exit 1
    fi
fi

SERVER_OVERRIDES=(
    --policy_name "${policy_name}"
    --task_name "${TASK}"
    --train_config_name "${TRAIN_CONFIG}"
    --checkpoint_path "${CKPT_DIR}"
    --seed "${SEED}"
)
CLIENT_OVERRIDES=(
    --policy_name "${policy_name}"
    --task_name "${TASK}"
    --ckpt_dir "${CKPT_DIR}"
    --num_episodes "${NUM_EPISODES}"
    --seed "${SEED}"
    --episode_steps "${EVAL_EPISODE_STEPS}"
    --warmup_steps 60
)

append_both_override() {
    local key="$1"
    local value="$2"
    if [[ -n "${value}" ]]; then
        SERVER_OVERRIDES+=("--${key}" "${value}")
        CLIENT_OVERRIDES+=("--${key}" "${value}")
    fi
}

append_both_override robot_key "${ROBOT_KEY}"
append_both_override use_active_dof "${USE_ACTIVE_DOF}"
append_both_override enable_generalization "${ENABLE_GENERALIZATION}"
append_both_override generalization_profile "${GENERALIZATION_PROFILE}"
append_both_override generalization_split "${GENERALIZATION_SPLIT}"
append_both_override generalization_config "${GENERALIZATION_CONFIG}"
append_both_override inv_cov_seed "${INV_COV_SEED}"
append_both_override anchor_dir "${ANCHOR_DIR}"
if [[ -n "${OUTPUT_DIR}" ]]; then
    CLIENT_OVERRIDES+=(--output_dir "${OUTPUT_DIR}")
fi
if [[ -n "${RECORD_DIR}" ]]; then
    CLIENT_OVERRIDES+=(--record_dir "${RECORD_DIR}")
fi
if [[ "${RECORD_ALL}" == "true" ]]; then
    CLIENT_OVERRIDES+=(--record_all true)
fi
if [[ -n "${START_EPISODE}" ]]; then
    CLIENT_OVERRIDES+=(--start_episode "${START_EPISODE}")
fi

echo "[eval] policy=${policy_name} task=${TASK} train_config=${TRAIN_CONFIG} gpu=${GPU_ID} episodes=${NUM_EPISODES} seed=${SEED}"
eval_budget_log
[[ -n "${ROBOT_KEY}" ]] && echo "[eval] robot_key=${ROBOT_KEY}"
[[ -n "${USE_ACTIVE_DOF}" ]] && echo "[eval] use_active_dof=${USE_ACTIVE_DOF}"
[[ -n "${ENABLE_GENERALIZATION}" ]] && echo "[eval] enable_generalization=${ENABLE_GENERALIZATION}"
[[ -n "${GENERALIZATION_PROFILE}" ]] && echo "[eval] generalization_profile=${GENERALIZATION_PROFILE}"
[[ -n "${GENERALIZATION_SPLIT}" ]] && echo "[eval] generalization_split=${GENERALIZATION_SPLIT}"
[[ -n "${ANCHOR_DIR}" ]] && echo "[eval] anchor_dir=${ANCHOR_DIR}"
[[ -n "${OUTPUT_DIR}" ]] && echo "[eval] output_dir=${OUTPUT_DIR}"
[[ -n "${RECORD_DIR}" ]] && echo "[eval] record_dir=${RECORD_DIR}"
[[ "${RECORD_ALL}" == "true" ]] && echo "[eval] record_all=true"
[[ "${RECORD_SUCCESS}" == "true" ]] && echo "[eval] record_success=true"
[[ -n "${EPISODES_PER_PROCESS}" ]] && echo "[eval] episodes_per_process=${EPISODES_PER_PROCESS}"
[[ -n "${START_EPISODE}" ]] && echo "[eval] start_episode=${START_EPISODE}"

SERVER_PID=""
cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap cleanup_server EXIT INT TERM

run_eval_chunk() {
    local chunk_count="$1"
    local chunk_start="$2"
    local append_mode="$3"
    local free_port
    free_port="$(get_free_port)"

    echo "[eval] chunk start_episode=${chunk_start} episodes=${chunk_count}"
    if [[ "${_USE_SII}" == "true" ]]; then
        source ../miniconda3/bin/activate pi05
        XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5} \
        python script/policy_model_server.py \
            --host 127.0.0.1 \
            --port "${free_port}" \
            --config "${SCRIPT_DIR}/deploy_policy.yml" \
            --overrides \
            "${SERVER_OVERRIDES[@]}" &
    else
        XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5} \
        "${UV_BIN}" --directory policy/pi05 run python ../../script/policy_model_server.py \
            --host 127.0.0.1 \
            --port "${free_port}" \
            --config "${SCRIPT_DIR}/deploy_policy.yml" \
            --overrides \
            "${SERVER_OVERRIDES[@]}" &
    fi
    SERVER_PID=$!

    SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-300}
    echo "[eval] Waiting for Pi0.5 server on 127.0.0.1:${free_port} ..."
    local server_ready=0
    local i
    for ((i=0; i<SERVER_READY_TIMEOUT; i++)); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[eval] ERROR: Pi0.5 server exited before becoming ready." >&2
            wait "${SERVER_PID}" || true
            SERVER_PID=""
            return 1
        fi
        if command -v ss >/dev/null 2>&1; then
            if ss -H -ltn "sport = :${free_port}" | grep -q .; then
                server_ready=1
                echo "[eval] Pi0.5 server is ready."
                break
            fi
        elif timeout 1 bash -c ":</dev/tcp/127.0.0.1/${free_port}" 2>/dev/null; then
            server_ready=1
            echo "[eval] Pi0.5 server is ready."
            break
        fi
        sleep 1
    done
    if [[ "${server_ready}" != "1" ]]; then
        echo "[eval] ERROR: Timed out waiting for Pi0.5 server after ${SERVER_READY_TIMEOUT}s." >&2
        cleanup_server
        return 1
    fi

    local chunk_overrides=(
        --num_episodes "${chunk_count}"
        --start_episode "${chunk_start}"
    )
    if [[ "${append_mode}" == "true" ]]; then
        chunk_overrides+=(--append_output true)
        if [[ -n "${RECORD_DIR}" ]]; then
            chunk_overrides+=(--append_record_dir true)
        fi
    fi

    local status=0
    if [[ "${_USE_SII}" == "true" ]]; then
        source ../isaaclab_setup/env_new.sh
        # Safe Xvfb start: reuse if already running, clean up stale lock otherwise
        if ! pgrep -f "Xvfb.*:99" >/dev/null 2>&1; then
            rm -f /tmp/.X99-lock 2>/dev/null || true
            start_xvfb
        else
            export DISPLAY=:99
            echo "[isaaclab] Xvfb already running on :99, reusing"
        fi
        activate_isaaclab
        setup_nvidia_libs
    fi
    python script/eval_policy_client.py \
        --host 127.0.0.1 \
        --port "${free_port}" \
        --config "${SCRIPT_DIR}/deploy_policy.yml" \
        --overrides \
        "${CLIENT_OVERRIDES[@]}" \
        "${chunk_overrides[@]}" || status=$?
    cleanup_server
    return "${status}"
}

if [[ -n "${EPISODES_PER_PROCESS}" ]]; then
    remaining="${NUM_EPISODES}"
    chunk_start="${START_EPISODE}"
    append_mode=false
    while (( remaining > 0 )); do
        chunk_count="${EPISODES_PER_PROCESS}"
        if (( chunk_count > remaining )); then
            chunk_count="${remaining}"
        fi
        run_eval_chunk "${chunk_count}" "${chunk_start}" "${append_mode}"
        append_mode=true
        chunk_start=$((chunk_start + chunk_count))
        remaining=$((remaining - chunk_count))
    done
else
    run_eval_chunk "${NUM_EPISODES}" "${START_EPISODE}" false
fi
