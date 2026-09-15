#!/bin/bash
set -euo pipefail

# Run pi05 evaluation for all four generalization channels and summarize success rates.
#
# Usage:
#   bash policy/pi05/eval_all_channels.sh TASK ROBOT_KEY CKPT_DIR ANCHOR_DIR [train_config] [gpu_id] [num_episodes] [seed] [options]
#
# Example:
#   bash policy/pi05/eval_all_channels.sh \
#     44_microwave_bowl_loading multi_ur5_schunk_hand_with_flange \
#     /path/to/pi05_ckpt /path/to/replay-generalization \
#     --episodes-per-process 10

usage() {
    cat >&2 <<'EOF'
Usage:
  bash policy/pi05/eval_all_channels.sh TASK ROBOT_KEY CKPT_DIR ANCHOR_DIR [train_config] [gpu_id] [num_episodes] [seed] [options]

Required:
  TASK          Scene name, e.g. 44_microwave_bowl_loading.
  ROBOT_KEY     Robot registry key used by the checkpoint.
  CKPT_DIR      pi05 checkpoint step directory.
  ANCHOR_DIR    Replay HDF5 directory to cycle as anchors for none/cov/inv channels.

Optional positionals:
  train_config  Default: pi05_base_dex2bench_lora
  gpu_id        Default: 0
  num_episodes  Default: 50
  seed          Default: 100000000

Options:
  --output-root PATH            Parent output directory for all four channels.
  --anchor-dir PATH             Override the positional anchor directory.
  --generalization-split SPLIT  seen | unseen | all; default: unseen.
  --generalization-config PATH  Override configs/scene/generalization.yaml.
  --episode-steps N             Override policy-step budget.
  --max-steps N                 Override physics-step budget; converted to policy steps.
  --episodes-per-process N      Restart Isaac/server every N episodes to release CUDA memory.
  --record-success              Save successful-episode HDF5 recordings and MP4 videos.
  --record-all                  With recording enabled, also save failed-episode HDF5 recordings and MP4 videos.
  --record-res N                Max recorded image dimension; default: DEX2BENCH_RECORD_RES or 240.
  --record-jpeg N               JPEG quality for HDF5 frames; default: DEX2BENCH_RECORD_JPEG or 25.
  --record-stride N             Record every N policy steps; default: DEX2BENCH_RECORD_STRIDE or 3.
  --record-cams LIST            Comma-separated cameras; default: cam_overhead. Use "all" for all cameras.
  --video-fps N                 MP4 export FPS; default: 20.
  --video-cell-width N          Per-camera tile width for MP4 export; default: 480.
  --video-cell-height N         Per-camera tile height for MP4 export; default: 360.
  --use-active-dof [BOOL]       Default: true.
  --active-dof                  Use active DOF.
  --no-active-dof               Disable active DOF.
  --sii                         Use conda envs (pi05 server, isaaclab client)
                                instead of uv-managed venv.
  --keep-going                  Continue if one channel fails.
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
        echo "[eval-all] ERROR: missing value for $1" >&2
        usage
        exit 2
    fi
}

normalize_bool() {
    case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|y|on) echo "true" ;;
        0|false|no|n|off) echo "false" ;;
        *)
            echo "[eval-all] ERROR: expected boolean, got '$1'" >&2
            exit 2
            ;;
    esac
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -lt 4 ]]; then
    usage
    exit 2
fi

TASK=$1
ROBOT_KEY=$2
CKPT_DIR=$3
ANCHOR_ARG=$4
shift 4

POLICY_NAME=pi05
TRAIN_CONFIG=pi05_base_dex2bench_full
GPU_ID=0
NUM_EPISODES=50
SEED=100000000

if [[ $# -gt 0 && "$1" != --* ]]; then TRAIN_CONFIG=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then GPU_ID=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then NUM_EPISODES=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then SEED=$1; shift; fi

GENERALIZATION_SPLIT=unseen
GENERALIZATION_CONFIG=""
ANCHOR_DIR="${ANCHOR_ARG}"
EPISODE_STEPS_OVERRIDE=""
MAX_STEPS_OVERRIDE=""
EPISODES_PER_PROCESS=0
USE_ACTIVE_DOF=true
OUTPUT_ROOT=""
INV_COV_SEED=""
CHANNELS_SELECTION=""
KEEP_GOING=false
_USE_SII=false
RECORD_SUCCESS=false
RECORD_ALL=false
RECORD_RES="${DEX2BENCH_RECORD_RES:-240}"
RECORD_JPEG="${DEX2BENCH_RECORD_JPEG:-25}"
RECORD_STRIDE="${DEX2BENCH_RECORD_STRIDE:-3}"
RECORD_CAMS="cam_overhead"
VIDEO_FPS=20
VIDEO_CELL_WIDTH=480
VIDEO_CELL_HEIGHT=360

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-root|--output_root)
            require_value "$1" "${2:-}"
            OUTPUT_ROOT=$2
            shift 2
            ;;
        --anchor-dir|--anchor_dir)
            require_value "$1" "${2:-}"
            ANCHOR_DIR=$2
            shift 2
            ;;
        --generalization-split|--generalization_split)
            require_value "$1" "${2:-}"
            GENERALIZATION_SPLIT=$2
            shift 2
            ;;
        --generalization-config|--generalization_config)
            require_value "$1" "${2:-}"
            GENERALIZATION_CONFIG=$2
            shift 2
            ;;
        --episode-steps|--episode_steps)
            require_value "$1" "${2:-}"
            EPISODE_STEPS_OVERRIDE=$2
            shift 2
            ;;
        --max-steps|--max_steps)
            require_value "$1" "${2:-}"
            MAX_STEPS_OVERRIDE=$2
            shift 2
            ;;
        --inv-cov-seed|--inv_cov_seed)
            require_value "$1" "${2:-}"
            INV_COV_SEED=$2
            shift 2
            ;;
        --episodes-per-process|--episodes_per_process)
            require_value "$1" "${2:-}"
            EPISODES_PER_PROCESS=$2
            shift 2
            ;;
        --record-success|--record_success|--save-success-videos|--save_success_videos)
            RECORD_SUCCESS=true
            shift
            ;;
        --no-record-success|--no_record_success)
            RECORD_SUCCESS=false
            RECORD_ALL=false
            shift
            ;;
        --record-all|--record_all)
            RECORD_SUCCESS=true
            RECORD_ALL=true
            shift
            ;;
        --no-record-all|--no_record_all)
            RECORD_ALL=false
            shift
            ;;
        --record-res|--record_res)
            require_value "$1" "${2:-}"
            RECORD_RES=$2
            shift 2
            ;;
        --record-jpeg|--record_jpeg)
            require_value "$1" "${2:-}"
            RECORD_JPEG=$2
            shift 2
            ;;
        --record-stride|--record_stride)
            require_value "$1" "${2:-}"
            RECORD_STRIDE=$2
            shift 2
            ;;
        --record-cams|--record_cams)
            require_value "$1" "${2:-}"
            RECORD_CAMS=$2
            case "$(printf '%s' "${RECORD_CAMS}" | tr '[:upper:]' '[:lower:]')" in
                all) RECORD_CAMS="" ;;
            esac
            shift 2
            ;;
        --video-fps|--video_fps)
            require_value "$1" "${2:-}"
            VIDEO_FPS=$2
            shift 2
            ;;
        --video-cell-width|--video_cell_width)
            require_value "$1" "${2:-}"
            VIDEO_CELL_WIDTH=$2
            shift 2
            ;;
        --video-cell-height|--video_cell_height)
            require_value "$1" "${2:-}"
            VIDEO_CELL_HEIGHT=$2
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
        --sii)
            _USE_SII=true
            shift
            ;;
        --keep-going|--keep_going)
            KEEP_GOING=true
            shift
            ;;
        --channels|--channel)
            require_value "$1" "${2:-}"
            CHANNELS_SELECTION=$2
            shift 2
            ;;
        --channels=*)
            CHANNELS_SELECTION="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[eval-all] ERROR: unknown option '$1'" >&2
            usage
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 自备环境:载入 isaaclab_setup 工具函数并激活 dex2bench(Isaac)环境,保证本脚本
# 汇总步骤(写 summary.json / success_rates.tsv)的裸 `python` 可用(4090 系统默认
# 只有 python3)。内层 eval_double_env.sh 的 server/client 环境由 --sii 自行管理;
# dexbench 无 activate.d 钩子,不会污染 LD_LIBRARY_PATH,切换互不影响。
source "${ROOT_DIR}/../isaaclab_setup/env_new.sh"
activate_isaaclab

if [[ ! -f "${ROOT_DIR}/scenes/${TASK}.yaml" ]]; then
    echo "[eval-all] ERROR: task YAML not found: ${ROOT_DIR}/scenes/${TASK}.yaml" >&2
    exit 2
fi
if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "[eval-all] ERROR: checkpoint directory not found: ${CKPT_DIR}" >&2
    exit 2
fi
if [[ -f "${ANCHOR_DIR}" ]]; then
    echo "[eval-all] WARN: third positional anchor is a file; using its directory: ${ANCHOR_DIR}" >&2
    ANCHOR_DIR="$(dirname "${ANCHOR_DIR}")"
fi
if [[ ! -d "${ANCHOR_DIR}" ]]; then
    echo "[eval-all] ERROR: anchor directory not found: ${ANCHOR_DIR}" >&2
    exit 2
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    ckpt_run=$(basename "$(dirname "${CKPT_DIR}")")
    ckpt_step=$(basename "${CKPT_DIR}")
    OUTPUT_ROOT="${ROOT_DIR}/../dex2bench/output_zdj/${POLICY_NAME}_four_channels/${TASK}/${ROBOT_KEY}/${ckpt_run}_${ckpt_step}_${timestamp}"
fi
mkdir -p "${OUTPUT_ROOT}"

RESULT_TSV="${OUTPUT_ROOT}/success_rates.tsv"
printf 'channel\tstatus\tepisodes\tsuccesses\tsuccess_rate\tsuccess_percent\toutput_dir\n' > "${RESULT_TSV}"

if [[ "${RECORD_SUCCESS}" == "true" ]]; then
    export DEX2BENCH_RECORD_RES="${RECORD_RES}"
    export DEX2BENCH_RECORD_JPEG="${RECORD_JPEG}"
    export DEX2BENCH_RECORD_STRIDE="${RECORD_STRIDE}"
    export DEX2BENCH_RECORD_CAMS="${RECORD_CAMS}"
fi

append_result_row() {
    local channel="$1"
    local status="$2"
    local output_dir="$3"
    local stats
    stats=$(python - "${output_dir}/summary.json" "${output_dir}/per_episode.jsonl" <<'PY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
episode_path = pathlib.Path(sys.argv[2])
episodes = None
successes = None
rate = None

if summary_path.is_file():
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    metadata = summary.get("metadata") or {}
    completion = (summary.get("core_metrics") or {}).get("completion") or {}
    episodes = metadata.get("n_episodes")
    successes = metadata.get("successes")
    rate = completion.get("success_rate")

if rate is None and episode_path.is_file():
    rows = []
    with episode_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    episodes = len(rows)
    successes = sum(1 for row in rows if bool(row.get("success")))
    rate = successes / episodes if episodes else None

def fmt(value):
    return "-" if value is None else str(value)

if isinstance(rate, (int, float)):
    print(f"{fmt(episodes)}\t{fmt(successes)}\t{rate:.6f}\t{rate * 100:.2f}%")
else:
    print(f"{fmt(episodes)}\t{fmt(successes)}\t-\t-")
PY
)
    printf '%s\t%s\t%s\t%s\n' "${channel}" "${status}" "${stats}" "${output_dir}" >> "${RESULT_TSV}"
}

finalize_channel_output() {
    local channel="$1"
    local output_dir="$2"
    python - "${ROOT_DIR}" "${output_dir}" "${POLICY_NAME}:${channel}" <<'PY'
import json
import pathlib
import sys
from dataclasses import fields

root_dir = pathlib.Path(sys.argv[1])
output_dir = pathlib.Path(sys.argv[2])
policy_name = sys.argv[3]
sys.path.insert(0, str(root_dir))

from benchmark.metrics import EpisodeResult
from benchmark.results import write_results

episode_path = output_dir / "per_episode.jsonl"
if not episode_path.is_file():
    raise SystemExit(f"missing per_episode.jsonl: {episode_path}")

field_names = {field.name for field in fields(EpisodeResult)}
results = []
with episode_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        payload = {key: value for key, value in row.items() if key in field_names}
        results.append(EpisodeResult(**payload))

write_results(output_dir, policy_name, results)
PY
}

print_table() {
    echo
    echo "[eval-all] success rates:"
    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "${RESULT_TSV}"
    else
        cat "${RESULT_TSV}"
    fi
    echo "[eval-all] summary saved to: ${RESULT_TSV}"
}

export_hdf5_videos() {
    local video_dir="$1"
    local prefix="$2"
    local label="$3"
    shift 3
    local hdf5_files=("$@")

    if (( ${#hdf5_files[@]} == 0 )); then
        return 0
    fi

    mkdir -p "${video_dir}"
    local exported=0
    local hdf5
    for hdf5 in "${hdf5_files[@]}"; do
        local base
        base=$(basename "${hdf5%.hdf5}")
        local out="${video_dir}/${prefix}_${base}_grid.mp4"
        python "${ROOT_DIR}/tools/export/export_video.py" "${hdf5}" \
            --output "${out}" \
            --fps "${VIDEO_FPS}" \
            --cell-width "${VIDEO_CELL_WIDTH}" \
            --cell-height "${VIDEO_CELL_HEIGHT}"
        exported=$((exported + 1))
    done
    echo "[eval-all] ${prefix}: exported ${exported} ${label} video(s) to ${video_dir}"
}

export_success_videos() {
    local record_dir="$1"
    local video_dir="$2"
    local prefix="$3"

    if [[ "${RECORD_SUCCESS}" != "true" ]]; then
        return 0
    fi
    if [[ "${RECORD_ALL}" == "true" ]]; then
        echo "[eval-all] ${prefix}: record-all mode, skipping MP4 export (HDF5 recordings saved)"
        return 0
    fi
    if [[ ! -d "${record_dir}" ]]; then
        echo "[eval-all] ${prefix}: no recordings directory: ${record_dir}"
        return 0
    fi

    shopt -s nullglob
    local success_files=("${record_dir}/success"/*.hdf5)
    local failure_files=("${record_dir}/failure"/*.hdf5)
    local direct_files=("${record_dir}"/*.hdf5)
    shopt -u nullglob

    local exported=0
    if (( ${#success_files[@]} > 0 )); then
        if [[ "${RECORD_ALL}" == "true" ]]; then
            export_hdf5_videos "${video_dir}/success" "${prefix}" "successful" "${success_files[@]}"
        else
            export_hdf5_videos "${video_dir}" "${prefix}" "successful" "${success_files[@]}"
        fi
        exported=$((exported + ${#success_files[@]}))
    fi
    if [[ "${RECORD_ALL}" == "true" && ${#failure_files[@]} -gt 0 ]]; then
        export_hdf5_videos "${video_dir}/failure" "${prefix}" "failed" "${failure_files[@]}"
        exported=$((exported + ${#failure_files[@]}))
    fi
    if (( exported == 0 && ${#direct_files[@]} > 0 )); then
        export_hdf5_videos "${video_dir}" "${prefix}" "recorded" "${direct_files[@]}"
        exported=$((exported + ${#direct_files[@]}))
    fi
    if (( exported == 0 )); then
        echo "[eval-all] ${prefix}: no recordings to export"
    fi
}

echo "[eval-all] policy=${POLICY_NAME} task=${TASK} robot=${ROBOT_KEY} ckpt=${CKPT_DIR}"
echo "[eval-all] anchor_dir=${ANCHOR_DIR}"
echo "[eval-all] output_root=${OUTPUT_ROOT}"
if (( EPISODES_PER_PROCESS > 0 && EPISODES_PER_PROCESS < NUM_EPISODES )); then
    echo "[eval-all] episodes_per_process=${EPISODES_PER_PROCESS} (process restart enabled)"
fi
if [[ "${RECORD_SUCCESS}" == "true" ]]; then
    echo "[eval-all] record_success=true res=${RECORD_RES} jpeg=${RECORD_JPEG} stride=${RECORD_STRIDE} cams=${RECORD_CAMS:-all}"
fi
if [[ "${RECORD_ALL}" == "true" ]]; then
    echo "[eval-all] record_all=true"
fi

run_channel_once() {
    local channel="$1"
    local count="$2"
    local start_episode="$3"
    local output_dir="$4"
    local record_dir="${5:-}"
    local args=(
        "${TASK}"
        "${ROBOT_KEY}"
        "${CKPT_DIR}"
        "${TRAIN_CONFIG}"
        "${GPU_ID}"
        "${count}"
        "${SEED}"
        --channel "${channel}"
        --use-active-dof "${USE_ACTIVE_DOF}"
        --generalization-split "${GENERALIZATION_SPLIT}"
        --output-dir "${output_dir}"
        --start-episode "${start_episode}"
    )
    if [[ -n "${GENERALIZATION_CONFIG}" ]]; then
        args+=(--generalization-config "${GENERALIZATION_CONFIG}")
    fi
    if [[ -n "${EPISODE_STEPS_OVERRIDE}" ]]; then
        args+=(--episode-steps "${EPISODE_STEPS_OVERRIDE}")
    fi
    if [[ -n "${MAX_STEPS_OVERRIDE}" ]]; then
        args+=(--max-steps "${MAX_STEPS_OVERRIDE}")
    fi
    if [[ "${channel}" != "inv_cov" ]]; then
        args+=(--anchor-dir "${ANCHOR_DIR}")
    fi
    if [[ -n "${record_dir}" ]]; then
        args+=(--record-dir "${record_dir}")
    fi
    if [[ "${RECORD_ALL}" == "true" ]]; then
        args+=(--record-all)
    fi
    if [[ -n "${INV_COV_SEED}" ]]; then
        args+=(--inv-cov-seed "${INV_COV_SEED}")
    fi
    if [[ "${_USE_SII}" == "true" ]]; then
        args+=(--sii)
    fi
    bash "${SCRIPT_DIR}/eval_double_env.sh" "${args[@]}"
}

CHANNELS=(none cov inv inv_cov)
if [[ -n "${CHANNELS_SELECTION}" ]]; then
    CHANNELS=(${CHANNELS_SELECTION//,/ })
fi
for channel in "${CHANNELS[@]}"; do
    channel_dir="${OUTPUT_ROOT}/${channel}"
    mkdir -p "${channel_dir}"

    echo
    echo "[eval-all] ===== ${channel} ====="

    if (( EPISODES_PER_PROCESS > 0 && EPISODES_PER_PROCESS < NUM_EPISODES )); then
        : > "${channel_dir}/per_episode.jsonl"
        chunks_dir="${channel_dir}/chunks"
        mkdir -p "${chunks_dir}"
        start_episode=1
        remaining="${NUM_EPISODES}"
        channel_status=0
        while (( remaining > 0 )); do
            count="${EPISODES_PER_PROCESS}"
            if (( count > remaining )); then
                count="${remaining}"
            fi
            end_episode=$((start_episode + count - 1))
            chunk_label="$(printf '%04d_%04d' "${start_episode}" "${end_episode}")"
            chunk_dir="${chunks_dir}/${chunk_label}"
            record_dir=""
            if [[ "${RECORD_SUCCESS}" == "true" ]]; then
                record_dir="${chunk_dir}/success_records"
            fi
            mkdir -p "${chunk_dir}"
            echo "[eval-all] ${channel}: episodes ${start_episode}-${end_episode} (${count})"
            if run_channel_once "${channel}" "${count}" "${start_episode}" "${chunk_dir}" "${record_dir}"; then
                video_dir="${channel_dir}/success_videos"
                if [[ "${RECORD_ALL}" == "true" ]]; then
                    video_dir="${channel_dir}/record_videos"
                fi
                if ! export_success_videos "${record_dir}" "${video_dir}" "${chunk_label}"; then
                    echo "[eval-all] WARN: failed to export recorded videos for ${channel} ${chunk_label}" >&2
                fi
                if [[ -s "${chunk_dir}/per_episode.jsonl" ]]; then
                    cat "${chunk_dir}/per_episode.jsonl" >> "${channel_dir}/per_episode.jsonl"
                else
                    echo "[eval-all] ERROR: missing chunk result: ${chunk_dir}/per_episode.jsonl" >&2
                    channel_status=1
                    break
                fi
            else
                channel_status=$?
                break
            fi
            start_episode=$((end_episode + 1))
            remaining=$((remaining - count))
        done
        if (( channel_status == 0 )); then
            finalize_channel_output "${channel}" "${channel_dir}"
            append_result_row "${channel}" "OK" "${channel_dir}"
        else
            append_result_row "${channel}" "FAILED(${channel_status})" "${channel_dir}"
            print_table
            if [[ "${KEEP_GOING}" != "true" ]]; then
                exit "${channel_status}"
            fi
        fi
    else
        record_dir=""
        if [[ "${RECORD_SUCCESS}" == "true" ]]; then
            record_dir="${channel_dir}/success_records"
        fi
        if run_channel_once "${channel}" "${NUM_EPISODES}" 1 "${channel_dir}" "${record_dir}"; then
            video_dir="${channel_dir}/success_videos"
            if [[ "${RECORD_ALL}" == "true" ]]; then
                video_dir="${channel_dir}/record_videos"
            fi
            if ! export_success_videos "${record_dir}" "${video_dir}" "$(printf '%04d_%04d' 1 "${NUM_EPISODES}")"; then
                echo "[eval-all] WARN: failed to export recorded videos for ${channel}" >&2
            fi
            finalize_channel_output "${channel}" "${channel_dir}"
            append_result_row "${channel}" "OK" "${channel_dir}"
        else
            rc=$?
            append_result_row "${channel}" "FAILED(${rc})" "${channel_dir}"
            print_table
            if [[ "${KEEP_GOING}" != "true" ]]; then
                exit "${rc}"
            fi
        fi
    fi
done

print_table
