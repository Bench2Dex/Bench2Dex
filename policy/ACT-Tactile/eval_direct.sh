#!/usr/bin/env bash
# Direct Isaac Sim evaluation for tactile ACT.
#
# Supports --episodes-per-process to restart the sim every N episodes and
# release CUDA memory, matching pi05 eval_all_channels.sh behaviour.
#
# TACTILE_GEN_PROFILE defaults to ALL four channels (none cov inv inv_cov);
# set it (comma-separated) to run only specific ones, e.g. TACTILE_GEN_PROFILE=none.
#
# Usage (env-var style, same as before):
#   TACTILE_TASK=26_canned_food_tray_line_arrangement \
#   TACTILE_CKPT_DIR=/path/to/ckpt \
#   bash policy/ACT-Tactile/eval_direct.sh [options]

set -euo pipefail

if (( BASH_VERSINFO[0] < 4 )); then
    echo "[eval_direct] ERROR: Bash 4+ is required; found ${BASH_VERSION}. " \
         "Run this wrapper in the Isaac/Linux environment." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TACTILE_RUNNER="${SCRIPT_DIR}/tactile_run_policy.py"

[[ -f "${TACTILE_RUNNER}" ]] || {
    echo "[eval_direct] ERROR: tactile runner not found: ${TACTILE_RUNNER}" >&2
    exit 2
}

cd "${REPO_ROOT}"

TACTILE_TASK="${TACTILE_TASK:-}"
TACTILE_CKPT_DIR="${TACTILE_CKPT_DIR:-}"
TACTILE_CKPT_NAME="${TACTILE_CKPT_NAME:-policy_best.ckpt}"
TACTILE_TEMPORAL_AGG_K="${TACTILE_TEMPORAL_AGG_K:-0.1}"
TACTILE_ROBOT_KEY="${TACTILE_ROBOT_KEY:-}"
TACTILE_GPU_ID="${TACTILE_GPU_ID:-0}"
TACTILE_NUM_EPISODES="${TACTILE_NUM_EPISODES:-10}"
TACTILE_EPISODE_STEPS="${TACTILE_EPISODE_STEPS:-}"
TACTILE_SEED="${TACTILE_SEED:-100000000}"
TACTILE_GEN_PROFILE="${TACTILE_GEN_PROFILE:-}"
TACTILE_ANCHOR_HDF5="${TACTILE_ANCHOR_HDF5:-}"
TACTILE_MAX_STEPS="${TACTILE_MAX_STEPS:-}"
TACTILE_OUTPUT_DIR="${TACTILE_OUTPUT_DIR:-}"
TACTILE_EPISODES_PER_PROCESS="${TACTILE_EPISODES_PER_PROCESS:-0}"
ISAAC_PYTHON="${ISAAC_PYTHON:-python}"

# ── parsed options (initialised empty) ───────────────────────────────────────
OUTPUT_ROOT=""
RECORD_DIR=""
RECORD_ALL=false
APPEND_RECORD_DIR=false
APPEND_OUTPUT=false
DUMP_TACTILE_ATTENTION=false
ANCHOR_DIR=""
ANCHOR_HDF5=""
EXTRA_ARGS=()

# ── helpers ──────────────────────────────────────────────────────────────────

usage() {
    cat >&2 <<'EOF'
Usage:
  TACTILE_TASK=<scene> TACTILE_CKPT_DIR=<dir> bash policy/ACT-Tactile/eval_direct.sh [options]

Environment variables:
  TACTILE_TASK                   Scene name, e.g. 73_jigsaw_puzzle_assembly
  TACTILE_CKPT_DIR               Checkpoint directory
  TACTILE_CKPT_NAME              Checkpoint filename (default: policy_best.ckpt)
  TACTILE_TEMPORAL_AGG_K         Temporal aggregation decay (default: 0.1)
  TACTILE_ROBOT_KEY              Optional robot key for validation
  TACTILE_GPU_ID                 CUDA device (default: 0)
  TACTILE_NUM_EPISODES           Total episodes per channel (default: 10)
  TACTILE_EPISODES_PER_PROCESS   Restart sim every N episodes (0 = single run, no restart)
  TACTILE_EPISODE_STEPS          Steps per episode (default: 400)
  TACTILE_SEED                   Base seed (default: 100000000)
  TACTILE_GEN_PROFILE            Channels to run, comma-separated. Default: ALL
                                 (none,cov,inv,inv_cov). e.g. TACTILE_GEN_PROFILE=none
  TACTILE_ANCHOR_HDF5            Anchor HDF5 for none/cov/inv channels
  TACTILE_OUTPUT_DIR             Output directory for results
  ISAAC_PYTHON                   Python binary (default: python)

Options (overrides / extras):
  --output-root PATH             Parent directory; auto-creates <profile>/<timestamp>/ under it.
  --record-dir PATH              Directory to save HDF5 snapshots
  --record-all                   Save ALL episodes (not just successes)
  --append-record-dir            Do not clear existing --record-dir at startup
  --append-output                Append to existing per_episode.jsonl
  --dump-tactile-attention       Log attention-pooling weights to tactile_attention.jsonl
  --anchor-dir PATH              Directory of replay HDF5 episodes (overrides TACTILE_ANCHOR_HDF5)
  --anchor-hdf5 PATH             Single anchor HDF5 file
  --gpu-id N                     CUDA device (overrides TACTILE_GPU_ID env var)
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
        echo "[eval_direct] ERROR: missing value for $1" >&2
        usage
        exit 2
    fi
}

# ── parse options ────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --record-dir|--record_dir)
            require_value "$1" "${2:-}"
            RECORD_DIR="$2"
            shift 2
            ;;
        --record-all|--record_all)
            RECORD_ALL=true
            shift
            ;;
        --append-record-dir|--append_record_dir)
            APPEND_RECORD_DIR=true
            shift
            ;;
        --append-output|--append_output)
            APPEND_OUTPUT=true
            shift
            ;;
        --dump-tactile-attention|--dump_tactile_attention)
            DUMP_TACTILE_ATTENTION=true
            shift
            ;;
        --anchor-dir|--anchor_dir)
            require_value "$1" "${2:-}"
            ANCHOR_DIR="$2"
            shift 2
            ;;
        --anchor-hdf5|--anchor_hdf5)
            require_value "$1" "${2:-}"
            ANCHOR_HDF5="$2"
            shift 2
            ;;
        --episode-steps|--episode_steps)
            require_value "$1" "${2:-}"
            TACTILE_EPISODE_STEPS="$2"
            shift 2
            ;;
        --max-steps|--max_steps)
            require_value "$1" "${2:-}"
            TACTILE_MAX_STEPS="$2"
            shift 2
            ;;
        --output-root|--output_root)
            require_value "$1" "${2:-}"
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --gpu-id|--gpu_id)
            require_value "$1" "${2:-}"
            TACTILE_GPU_ID="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            # Pass through any unknown args to tactile_run_policy.py
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── resolve anchor ───────────────────────────────────────────────────────────

# --anchor-dir takes priority over --anchor-hdf5 over TACTILE_ANCHOR_HDF5
if [[ -n "${ANCHOR_DIR}" ]]; then
    ANCHOR_HDF5=""
elif [[ -n "${ANCHOR_HDF5}" ]]; then
    :  # keep --anchor-hdf5 value
elif [[ -n "${TACTILE_ANCHOR_HDF5}" ]]; then
    ANCHOR_HDF5="${TACTILE_ANCHOR_HDF5}"
fi

finalize_output() {
    local output_dir="$1"
    local policy_name="$2"
    local dex2bench_root
    dex2bench_root="${REPO_ROOT}"
    "${ISAAC_PYTHON}" -c "
import json, pathlib, sys
from dataclasses import fields

root_dir = pathlib.Path('${dex2bench_root}')
output_dir = pathlib.Path('${output_dir}')
policy_name = '${policy_name}'
sys.path.insert(0, str(root_dir))
from benchmark.metrics import EpisodeResult
from benchmark.results import write_results

episode_path = output_dir / 'per_episode.jsonl'
field_names = {field.name for field in fields(EpisodeResult)}
results = []
with episode_path.open('r') as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        payload = {key: value for key, value in row.items() if key in field_names}
        results.append(EpisodeResult(**payload))
write_results(output_dir, policy_name, results)
print(f'[finalize] summary.json written to {output_dir}')
"
}

# ── validate required args ───────────────────────────────────────────────────

if [[ -z "${TACTILE_TASK}" || -z "${TACTILE_CKPT_DIR}" ]]; then
    usage
    exit 2
fi

if [[ "${TACTILE_TASK}" != *.yaml ]]; then
    TACTILE_TASK="scenes/${TACTILE_TASK}.yaml"
fi

# ── resolve channel list (default: all four) ─────────────────────────────────

PROFILE_LIST=()
if [[ -z "${TACTILE_GEN_PROFILE}" ]]; then
    PROFILE_LIST=(none cov inv inv_cov)
else
    IFS=',' read -ra PROFILE_LIST <<< "${TACTILE_GEN_PROFILE}"
fi
echo "[tactile_eval] channels: ${PROFILE_LIST[*]}"

ROBOT_KEY_ARGS=()
if [[ -n "${TACTILE_ROBOT_KEY}" ]]; then
    ROBOT_KEY_ARGS+=(--robot-key "${TACTILE_ROBOT_KEY}")
fi

OUTPUT_APPEND_ARGS=()
if [[ "${APPEND_OUTPUT}" == "true" ]]; then
    OUTPUT_APPEND_ARGS+=(--append-output)
fi

export CUDA_VISIBLE_DEVICES="${TACTILE_GPU_ID}"

# ── shared timestamp for this invocation ─────────────────────────────────────

timestamp=$(date +%Y%m%d_%H%M%S)
_ORIG_OUTPUT_DIR="${TACTILE_OUTPUT_DIR}"

# ── per-profile helpers ──────────────────────────────────────────────────────

resolve_profile_output_dir() {
    local profile="$1"
    if [[ -n "${OUTPUT_ROOT}" ]]; then
        echo "${OUTPUT_ROOT}/${timestamp}/${profile}"
    elif [[ -n "${TACTILE_OUTPUT_DIR}" ]]; then
        # TACTILE_OUTPUT_DIR given: insert timestamp before profile
        # e.g. .../26/none → .../26/<timestamp>/none
        local parent profile_dir
        parent="$(dirname "${TACTILE_OUTPUT_DIR}")"
        profile_dir="${TACTILE_OUTPUT_DIR##*/}"
        echo "${parent}/${timestamp}/${profile}"
    else
        # fully auto-generate
        local task_base task_id robot_short
        task_base="${TACTILE_TASK##*/}"
        task_base="${task_base%.yaml}"
        task_id="${task_base%%_*}"
        robot_short="${TACTILE_ROBOT_KEY:-unknown}"
        robot_short="${robot_short#multi_}"
        robot_short="${robot_short//_with_/_}"
        robot_short="${robot_short//_flange/}"
        echo "$(pwd)/output_zdj/tactileACT/${task_id}/${timestamp}/${profile}"
    fi
}

write_invocation() {
    local output_dir="$1"
    local profile="$2"
    mkdir -p "${output_dir}"
    {
        echo "# Invocation saved by policy/ACT-Tactile/eval_direct.sh"
        echo "# Date: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "# Host: $(hostname)"
        echo "# PWD: $(pwd)"
        echo
        echo "TACTILE_TASK=${TACTILE_TASK} \\"
        echo "TACTILE_CKPT_DIR=${TACTILE_CKPT_DIR} \\"
        echo "TACTILE_CKPT_NAME=${TACTILE_CKPT_NAME} \\"
        echo "TACTILE_GEN_PROFILE=${profile} \\"
        echo "TACTILE_ROBOT_KEY=${TACTILE_ROBOT_KEY} \\"
        echo "TACTILE_NUM_EPISODES=${TACTILE_NUM_EPISODES} \\"
        echo "TACTILE_EPISODES_PER_PROCESS=${TACTILE_EPISODES_PER_PROCESS} \\"
        echo "TACTILE_GPU_ID=${TACTILE_GPU_ID} \\"
        [[ -n "${TACTILE_EPISODE_STEPS}" ]] && echo "TACTILE_EPISODE_STEPS=${TACTILE_EPISODE_STEPS} \\"
        [[ -n "${TACTILE_MAX_STEPS}" ]] && echo "TACTILE_MAX_STEPS=${TACTILE_MAX_STEPS} \\"
        [[ -n "${TACTILE_ANCHOR_HDF5}" ]] && echo "TACTILE_ANCHOR_HDF5=${TACTILE_ANCHOR_HDF5} \\"
        [[ -n "${_ORIG_OUTPUT_DIR}" ]] && echo "TACTILE_OUTPUT_DIR=${_ORIG_OUTPUT_DIR} \\"
        [[ -n "${OUTPUT_ROOT}" ]] && echo "  --output-root ${OUTPUT_ROOT} \\"
        echo "ISAAC_PYTHON=${ISAAC_PYTHON} \\"
        echo "bash policy/ACT-Tactile/eval_direct.sh \\"
        [[ -n "${ANCHOR_DIR}" ]] && echo "  --anchor-dir ${ANCHOR_DIR} \\"
        [[ -n "${ANCHOR_HDF5}" ]] && echo "  --anchor-hdf5 ${ANCHOR_HDF5} \\"
        [[ -n "${RECORD_DIR}" ]] && echo "  --record-dir ${RECORD_DIR} \\"
        [[ "${RECORD_ALL}" == "true" ]] && echo "  --record-all \\"
        [[ "${APPEND_RECORD_DIR}" == "true" ]] && echo "  --append-record-dir \\"
        [[ "${APPEND_OUTPUT}" == "true" ]] && echo "  --append-output \\"
        [[ "${DUMP_TACTILE_ATTENTION}" == "true" ]] && echo "  --dump-tactile-attention \\"
        for _extra in "${EXTRA_ARGS[@]}"; do
            echo "  ${_extra} \\"
        done
    } > "${output_dir}/invocation.txt"
}

build_policy_args() {
    local output_dir="$1"
    local num_episodes="$2"
    local start_episode="$3"
    local profile="$4"

    # Resolve record-dir: explicit --record-dir wins; auto-derive from output_dir
    # when --record-all is given, so recordings follow the timestamped output.
    local _record_dir="${RECORD_DIR}"
    if [[ "${RECORD_ALL}" == "true" && -z "${_record_dir}" && -n "${output_dir}" ]]; then
        _record_dir="${output_dir}/records"
    fi

    local gen_args=(--enable-generalization --generalization-profile "${profile}")
    if [[ "${profile}" != "inv_cov" && "${profile}" != "inv+cov" && "${profile}" != "full" ]]; then
        if [[ -n "${ANCHOR_DIR}" ]]; then
            gen_args+=(--anchor-dir "${ANCHOR_DIR}")
        elif [[ -n "${ANCHOR_HDF5}" ]]; then
            gen_args+=(--anchor-hdf5 "${ANCHOR_HDF5}")
        else
            echo "ERROR: anchor-dir or anchor-hdf5 is required for channel ${profile}" >&2
            exit 2
        fi
    fi

    local args=(
        --policy-type TACTILE
        --task "${TACTILE_TASK}"
        --ckpt-dir "${TACTILE_CKPT_DIR}"
        --ckpt-name "${TACTILE_CKPT_NAME}"
        --enable-rgb
        --active-dof
        --temporal-agg
        --temporal-agg-k "${TACTILE_TEMPORAL_AGG_K}"
        --device "cuda:0"
        --num-episodes "${num_episodes}"
        --start-episode "${start_episode}"
        --seed "${TACTILE_SEED}"
        --headless
    )
    if [[ -n "${TACTILE_EPISODE_STEPS}" ]]; then
        args+=(--episode-steps "${TACTILE_EPISODE_STEPS}")
    fi
    if [[ -n "${TACTILE_MAX_STEPS}" ]]; then
        args+=(--max-steps "${TACTILE_MAX_STEPS}")
    fi
    if [[ -n "${output_dir}" ]]; then
        args+=(--output-dir "${output_dir}")
    fi
    args+=(
        "${gen_args[@]}"
        "${ROBOT_KEY_ARGS[@]}"
    )
    if [[ -n "${_record_dir}" ]]; then
        args+=(--record-dir "${_record_dir}")
        if [[ "${RECORD_ALL}" == "true" ]]; then
            args+=(--record-all)
        fi
        if [[ "${APPEND_RECORD_DIR}" == "true" ]]; then
            args+=(--append-record-dir)
        fi
    fi
    if [[ "${DUMP_TACTILE_ATTENTION}" == "true" ]]; then
        args+=(--dump-tactile-attention)
    fi
    args+=(
        "${OUTPUT_APPEND_ARGS[@]}"
        "${EXTRA_ARGS[@]}"
    )
    printf '%s\n' "${args[@]}"
}

plot_attention() {
    local output_dir="$1"
    local jsonl="${output_dir}/tactile_attention.jsonl"
    if [[ "${DUMP_TACTILE_ATTENTION}" != "true" || ! -s "${jsonl}" ]]; then
        return 0
    fi
    # Pull checkpoint site names for axis labels
    local site_names
    site_names="$("${ISAAC_PYTHON}" -c "
import torch, sys
ckpt = torch.load('${TACTILE_CKPT_DIR}/${TACTILE_CKPT_NAME}', map_location='cpu', weights_only=False)
sites = ckpt['site_names']
print(' '.join(str(s) for s in sites))
" 2>/dev/null || true)"
    local plot_args=(tools/vis/plot_tactile_attention.py "${output_dir}")
    if [[ -n "${site_names}" ]]; then
        plot_args+=(--site-names ${site_names})
    fi
    "${ISAAC_PYTHON}" "${plot_args[@]}" \
        || echo "[tactile_eval] WARN: attention plot failed (JSONL kept at ${jsonl})" >&2
}

# ── run one channel (single or multi process) ────────────────────────────────

run_channel() {
    local profile="$1"
    local output_dir
    output_dir="$(resolve_profile_output_dir "${profile}")"

    echo
    echo "[tactile_eval] ===== channel: ${profile} ====="
    echo "[tactile_eval] output dir: ${output_dir}"
    write_invocation "${output_dir}" "${profile}"

    if (( TACTILE_EPISODES_PER_PROCESS <= 0 || TACTILE_EPISODES_PER_PROCESS >= TACTILE_NUM_EPISODES )); then
        # ── single-process path ──────────────────────────────────────────────
        mapfile -t _ARGS < <(build_policy_args "${output_dir}" "${TACTILE_NUM_EPISODES}" 1 "${profile}")
        "${ISAAC_PYTHON}" "${TACTILE_RUNNER}" "${_ARGS[@]}"
        local rc=$?
        if (( rc == 0 )) && [[ "${DUMP_TACTILE_ATTENTION}" == "true" ]]; then
            plot_attention "${output_dir}"
        fi
        return $rc
    fi

    # ── multi-process path ───────────────────────────────────────────────────
    echo "[tactile_eval] episodes_per_process=${TACTILE_EPISODES_PER_PROCESS} total=${TACTILE_NUM_EPISODES}"

    mkdir -p "${output_dir}"
    : > "${output_dir}/per_episode.jsonl"
    if [[ "${DUMP_TACTILE_ATTENTION}" == "true" ]]; then
        : > "${output_dir}/tactile_attention.jsonl"
        : > "${output_dir}/modality_attention.jsonl"
    fi

    CHUNKS_DIR="${output_dir}/chunks"
    mkdir -p "${CHUNKS_DIR}"

    local start_episode=1
    local remaining="${TACTILE_NUM_EPISODES}"
    local chunk_status=0

    while (( remaining > 0 )); do
        local count="${TACTILE_EPISODES_PER_PROCESS}"
        if (( count > remaining )); then
            count="${remaining}"
        fi
        local end_episode=$((start_episode + count - 1))
        local chunk_label chunk_dir
        chunk_label="$(printf '%04d_%04d' "${start_episode}" "${end_episode}")"
        chunk_dir="${CHUNKS_DIR}/${chunk_label}"
        mkdir -p "${chunk_dir}"

        echo "[tactile_eval] ${profile}: episodes ${start_episode}-${end_episode} (${count})"

        # Build args with chunk_dir as output-dir, NOT the channel output_dir
        mapfile -t _ARGS < <(build_policy_args "${chunk_dir}" "${count}" "${start_episode}" "${profile}")

        if "${ISAAC_PYTHON}" "${TACTILE_RUNNER}" "${_ARGS[@]}"; then
            if [[ -s "${chunk_dir}/per_episode.jsonl" ]]; then
                cat "${chunk_dir}/per_episode.jsonl" >> "${output_dir}/per_episode.jsonl"
            else
                echo "[tactile_eval] ERROR: missing chunk result: ${chunk_dir}/per_episode.jsonl" >&2
                chunk_status=1
                break
            fi
            # Merge attention dumps when enabled
            if [[ -s "${chunk_dir}/tactile_attention.jsonl" ]]; then
                cat "${chunk_dir}/tactile_attention.jsonl" >> "${output_dir}/tactile_attention.jsonl"
            fi
            if [[ -s "${chunk_dir}/modality_attention.jsonl" ]]; then
                cat "${chunk_dir}/modality_attention.jsonl" >> "${output_dir}/modality_attention.jsonl"
            fi
            if [[ -s "${chunk_dir}/tactile_attention.jsonl" || -s "${chunk_dir}/modality_attention.jsonl" ]]; then
                plot_attention "${output_dir}"
            fi
        else
            chunk_status=$?
            break
        fi

        start_episode=$((end_episode + 1))
        remaining=$((remaining - count))
    done

    if (( chunk_status == 0 )); then
        POLICY_NAME="TACTILE:${TACTILE_CKPT_NAME%.ckpt}"
        finalize_output "${output_dir}" "${POLICY_NAME}"
        plot_attention "${output_dir}"
        echo "[tactile_eval] Done: ${output_dir}"
        return 0
    else
        echo "[tactile_eval] FAILED (exit=${chunk_status})" >&2
        return "${chunk_status}"
    fi
}

# ── main: iterate channels ───────────────────────────────────────────────────

_global_status=0
_RESULTS_TABLE=()
for _profile in "${PROFILE_LIST[@]}"; do
    if run_channel "${_profile}"; then
        _RESULTS_TABLE+=("${_profile}|OK|$(resolve_profile_output_dir "${_profile}")")
    else
        _rc=$?
        _RESULTS_TABLE+=("${_profile}|FAILED(${_rc})|$(resolve_profile_output_dir "${_profile}")")
        _global_status=$((_global_status == 0 ? _rc : _global_status))
    fi
done

echo
echo "[tactile_eval] ==== channel summary ===="
for _row in "${_RESULTS_TABLE[@]}"; do
    echo "[tactile_eval] ${_row}"
done

# ── write aggregated success_rates.tsv into the shared timestamp dir ─────────

_summary_dir="$(dirname "$(resolve_profile_output_dir "${PROFILE_LIST[0]}")")"
_RESULT_TSV="${_summary_dir}/success_rates.tsv"
{
    printf 'channel\tstatus\tepisodes\tsuccesses\tsuccess_rate\tsuccess_percent\toutput_dir\n'
    for _profile in "${PROFILE_LIST[@]}"; do
        _channel_dir="$(resolve_profile_output_dir "${_profile}")"
        _status="OK"
        _stats="$("${ISAAC_PYTHON}" -c "
import json, pathlib, sys
output_dir = pathlib.Path('${_channel_dir}')
summary_path = output_dir / 'summary.json'
episode_path = output_dir / 'per_episode.jsonl'
episodes, successes, rate = None, None, None
if summary_path.is_file():
    with summary_path.open() as f:
        summary = json.load(f)
    metadata = summary.get('metadata') or {}
    completion = (summary.get('core_metrics') or {}).get('completion') or {}
    episodes = metadata.get('n_episodes')
    successes = metadata.get('successes')
    rate = completion.get('success_rate')
if rate is None and episode_path.is_file():
    rows = [json.loads(l) for l in episode_path.read_text().splitlines() if l.strip()]
    episodes = len(rows)
    successes = sum(1 for r in rows if bool(r.get('success')))
    rate = successes / episodes if episodes else None
fmt = lambda v: '-' if v is None else str(v)
if isinstance(rate, (int, float)):
    print(f'{fmt(episodes)}\t{fmt(successes)}\t{rate:.6f}\t{rate*100:.2f}%')
else:
    print(f'{fmt(episodes)}\t{fmt(successes)}\t-\t-')
" 2>/dev/null || echo "-\t-\t-\t-")"
        printf '%s\t%s\t%s\t%s\n' "${_profile}" "${_status}" "${_stats}" "${_channel_dir}"
    done
} > "${_RESULT_TSV}"
echo "[tactile_eval] summary saved: ${_RESULT_TSV}"
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "${_RESULT_TSV}"
else
    cat "${_RESULT_TSV}"
fi

if (( _global_status == 0 )); then
    echo "[tactile_eval] All channels done."
else
    echo "[tactile_eval] Finished with failures (exit=${_global_status})" >&2
fi
exit "${_global_status}"
