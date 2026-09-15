#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Multi-task sequential eval driver.
#
# Wraps policy/<POLICY>/eval_double_env.sh: runs each task's full channel set,
# then waits (poll) for GPU / policy processes to go idle before starting the
# next task. --tmux wraps the WHOLE sequence in one tmux session; child
# invocations never get --tmux (avoids nested sessions).
#
# Normally invoked via a per-policy wrapper:
#   bash policy/DP/eval_tasks_seq.sh        07 08 09 all --sii --headless --tmux
#   bash policy/GR00T_n15/eval_tasks_seq.sh 07 08 09 all --sii --headless --tmux
#
# Direct use (set POLICY):
#   POLICY=DP bash script/eval_tasks_seq.sh 07 08 09 all --sii --headless --tmux
#
# Args: <TASK_ID>... [CHANNEL] [--key value ...] [--tmux]
#   - Leading positionals are task IDs (must match scenes/<id>_*.yaml).
#   - The first positional that is a channel keyword (none/cov/inv/inv_cov/cov_then_inv/all
#     or aliases) becomes the channel; the rest are forwarded flags.
#   - --channel / --channel_num flag forms are forwarded to each child verbatim.
#
# Env:
#   POLICY            DP | GR00T_n15  (set by per-policy wrapper; default DP)
#   SEQ_MIN_WAIT_S    min grace after a task before polling      (default 30)
#   SEQ_MAX_WAIT_S    max wait between tasks, then force-continue (default 600)
#   SEQ_POLL_S        poll interval                               (default 10)
#   CUDA_VISIBLE_DEVICES  target GPU (forwarded + used for idle check)
#
# Idle check (between tasks): no policy_model_server / eval_policy_client /
# run_policy / isaac process, and no such process on the target GPU per
# nvidia-smi. A min grace covers GPU-memory release lag; a max cap prevents
# hanging forever on a shared GPU where other users' jobs keep it busy.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

POLICY="${POLICY:-DP}"
EVAL_SCRIPT="${REPO_ROOT}/policy/${POLICY}/eval_double_env.sh"
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo -e "\033[31m[seq][FATAL] eval script not found: ${EVAL_SCRIPT} (POLICY=${POLICY})\033[0m" >&2
    exit 1
fi

info() { echo -e "\033[34m[seq][info] $*\033[0m"; }
warn() { echo -e "\033[33m[seq][warn] $*\033[0m" >&2; }
die()  { echo -e "\033[31m[seq][FATAL] $*\033[0m" >&2; exit 1; }

SEQ_MIN_WAIT_S="${SEQ_MIN_WAIT_S:-30}"
SEQ_MAX_WAIT_S="${SEQ_MAX_WAIT_S:-600}"
SEQ_POLL_S="${SEQ_POLL_S:-10}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
# Per-task timing log (one row per task, appended). Override with SEQ_TIME_LOG.
TIME_LOG="${SEQ_TIME_LOG:-../jcy/task_time.txt}"

# ── tmux: wrap the whole sequence in one session; strip --tmux from children ─

_USE_TMUX=false
_TMUX_SESSION_NAME=""
_scan=("$@")
while [[ ${#_scan[@]} -gt 0 ]]; do
    case "${_scan[0]}" in
        --tmux) _USE_TMUX=true ;;
        --tmux-session-name)
            if [[ ${#_scan[@]} -ge 2 && "${_scan[1]}" != --* ]]; then
                _TMUX_SESSION_NAME="${_scan[1]}"
                _scan=("${_scan[@]:1}")
            fi ;;
        --tmux-session-name=*) _TMUX_SESSION_NAME="${_scan[0]#*=}" ;;
    esac
    _scan=("${_scan[@]:1}")
done

if [[ "${_USE_TMUX}" == "true" && -z "${TMUX:-}" ]]; then
    _self="$(readlink -f "$0")"
    _filtered=()
    _rest=("$@")
    while [[ ${#_rest[@]} -gt 0 ]]; do
        case "${_rest[0]}" in
            --tmux) _rest=("${_rest[@]:1}") ;;
            --tmux-session-name) [[ ${#_rest[@]} -ge 2 ]] && _rest=("${_rest[@]:2}") || _rest=("${_rest[@]:1}") ;;
            --tmux-session-name=*) _rest=("${_rest[@]:1}") ;;
            *) _filtered+=("${_rest[0]}"); _rest=("${_rest[@]:1}") ;;
        esac
    done
    _first="${_filtered[0]:-task}"
    _sess="${_TMUX_SESSION_NAME:-${POLICY}_seq_${_first}_$(date +%m%d_%H%M)}"
    info "Re-launching inside tmux session: ${_sess}"
    _cmd="source ~/.bashrc 2>/dev/null || true; cd ${REPO_ROOT} && exec env POLICY=${POLICY} CUDA_VISIBLE_DEVICES=${GPU_ID} bash ${_self}"
    for _a in "${_filtered[@]}"; do _cmd="${_cmd} $(printf %q "${_a}")"; done
    exec tmux new-session -s "${_sess}" "${_cmd}"
fi

# ── parse: <TASK_ID>... [CHANNEL] [flags] ────────────────────────────────────

TASK_IDS=()
CHANNEL_POSITIONAL=""
while [[ $# -gt 0 && "$1" != --* ]]; do
    _tok="$1"
    case "$(echo "${_tok}" | tr '[:upper:]' '[:lower:]')" in
        none|cov|cov_only|cov-only|inv|inv_only|inv-only|inv_cov|inv+cov|full|all|cov_then_inv)
            CHANNEL_POSITIONAL="${_tok}"; shift; break ;;
    esac
    # validate as task id (scene exists)
    shopt -s nullglob
    _scenes=(scenes/${_tok}_*.yaml)
    shopt -u nullglob
    [[ ${#_scenes[@]} -gt 0 ]] || die "Unknown positional arg '${_tok}' (not a channel keyword and no scenes/${_tok}_*.yaml). Usage: <TASK_ID>... [CHANNEL] [--key value ...]"
    TASK_IDS+=("${_tok}"); shift
done
[[ ${#TASK_IDS[@]} -gt 0 ]] || die "No task IDs given. Usage: <TASK_ID>... [CHANNEL] [--key value ...]"

# remaining flags, stripped of --tmux (children never get it)
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tmux) shift ;;
        --tmux-session-name) shift 2 ;;
        --tmux-session-name=*) shift ;;
        *) FORWARD_ARGS+=("$1"); shift ;;
    esac
done

info "Policy: ${POLICY}"
info "Eval script: ${EVAL_SCRIPT}"
info "Tasks (${#TASK_IDS[@]}): ${TASK_IDS[*]}"
info "Channel: ${CHANNEL_POSITIONAL:-<none, child default or --channel*>}"
info "Forward flags: ${FORWARD_ARGS[*]:-<none>}"
info "Inter-task wait: min=${SEQ_MIN_WAIT_S}s max=${SEQ_MAX_WAIT_S}s poll=${SEQ_POLL_S}s gpu=${GPU_ID}"

# ── idle check: no policy/isaac process, and none on the target GPU ──────────
# Reuses the same signals as eval_double_env.sh's zombie-cleanup
# (nvidia-smi --query-compute-apps + policy_model_server match).

_busy_pids() {
    local pids=""
    pids="$(pgrep -f 'policy_model_server\.py|eval_policy_client\.py|run_policy\.py' 2>/dev/null | tr '\n' ' ' || true)"
    local gpu_pids
    gpu_pids="$(nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]' | tr '\n' ' ' || true)"
    local p
    for p in ${gpu_pids}; do
        if ps -p "${p}" -o cmd= 2>/dev/null | grep -qE 'policy_model_server|eval_policy_client|run_policy|isaac'; then
            pids="${pids} ${p}"
        fi
    done
    echo "${pids}"
}

_wait_idle() {
    info "Waiting for GPU/policy processes to go idle (min ${SEQ_MIN_WAIT_S}s, max ${SEQ_MAX_WAIT_S}s)..."
    local waited=0
    while [[ ${waited} -lt ${SEQ_MIN_WAIT_S} ]]; do
        sleep "${SEQ_POLL_S}"; waited=$((waited + SEQ_POLL_S))
    done
    while [[ ${waited} -lt ${SEQ_MAX_WAIT_S} ]]; do
        local busy; busy="$(_busy_pids)"
        if [[ -z "${busy//[[:space:]]/}" ]]; then
            info "Idle after ${waited}s."
            return 0
        fi
        warn "Still busy (pids:${busy}); waiting ${SEQ_POLL_S}s ... (${waited}/${SEQ_MAX_WAIT_S}s)"
        sleep "${SEQ_POLL_S}"; waited=$((waited + SEQ_POLL_S))
    done
    warn "Reached max wait ${SEQ_MAX_WAIT_S}s — proceeding to next task anyway."
}

# ── per-task timing: append one row to TIME_LOG ──────────────────────────────

_log_time() {
    # args: policy task channel rc elapsed_s start_iso end_iso
    local _policy="$1" _task="$2" _channel="$3" _rc="$4" _elapsed="$5" _s_iso="$6" _e_iso="$7"
    mkdir -p "$(dirname "${TIME_LOG}")"
    if [[ ! -f "${TIME_LOG}" ]]; then
        printf 'start_iso\tend_iso\tpolicy\ttask\tchannel\trc\telapsed_s\n' > "${TIME_LOG}"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${_s_iso}" "${_e_iso}" "${_policy}" "${_task}" "${_channel}" "${_rc}" "${_elapsed}" >> "${TIME_LOG}"
}

# ── main loop ───────────────────────────────────────────────────────────────

_total_rc=0
for ((i = 0; i < ${#TASK_IDS[@]}; i++)); do
    _t="${TASK_IDS[$i]}"
    info "========== Task $((i + 1))/${#TASK_IDS[@]}: ${_t} (channel=${CHANNEL_POSITIONAL:-<fwd>}) =========="

    if [[ -n "${CHANNEL_POSITIONAL}" ]]; then
        _child_args=("${_t}" "${CHANNEL_POSITIONAL}" "${FORWARD_ARGS[@]}")
    else
        _child_args=("${_t}" "${FORWARD_ARGS[@]}")
    fi

    _t_start_epoch=$(date +%s)
    _t_start_iso=$(date '+%Y-%m-%d %H:%M:%S')

    set +e
    bash "${EVAL_SCRIPT}" "${_child_args[@]}"
    _rc=$?
    set -e

    _t_end_epoch=$(date +%s)
    _t_end_iso=$(date '+%Y-%m-%d %H:%M:%S')
    _elapsed=$((_t_end_epoch - _t_start_epoch))
    info "Task ${_t} elapsed ${_elapsed}s (rc=${_rc})."
    _log_time "${POLICY}" "${_t}" "${CHANNEL_POSITIONAL:-<flag>}" "${_rc}" "${_elapsed}" "${_t_start_iso}" "${_t_end_iso}"

    if [[ ${_rc} -ne 0 ]]; then
        warn "Task ${_t} exited with code ${_rc}; continuing to next task."
        _total_rc=${_rc}
    fi

    if [[ $((i + 1)) -lt ${#TASK_IDS[@]} ]]; then
        _wait_idle
    else
        info "Last task done; skipping inter-task wait."
    fi
done

info "All tasks finished. last_nonzero_rc=${_total_rc}"
exit "${_total_rc}"
