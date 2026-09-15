#!/usr/bin/env bash

tactile_checkpoint_is_valid() (
    local path="$1"
    local model

    [[ -f "${path}/config.json" ]] || return 1
    shopt -s nullglob
    for model in "${path}"/model-*.safetensors; do
        [[ -f "${model}" ]] && return 0
    done
    return 1
)

tactile_checkpoint_discover() (
    local checkpoint_root="$1"
    local task_id="$2"
    local candidate nested nested_step best_nested best_step
    local LC_ALL=C
    local -a candidates exact variants ordered nested_candidates

    shopt -s nullglob
    candidates=("${checkpoint_root}/${task_id}"/*/gr00t_n15_tactile*)
    exact=()
    variants=()
    for candidate in "${candidates[@]}"; do
        if [[ "$(basename "${candidate}")" == gr00t_n15_tactile ]]; then
            exact+=("${candidate}")
        else
            variants+=("${candidate}")
        fi
    done
    ordered=("${exact[@]}" "${variants[@]}")

    for candidate in "${ordered[@]}"; do
        if tactile_checkpoint_is_valid "${candidate}"; then
            printf '%s\n' "${candidate}"
            return 0
        fi

        best_nested=""
        best_step=""
        nested_candidates=("${candidate}"/checkpoint-*)
        for nested in "${nested_candidates[@]}"; do
            nested_step="${nested##*/}"
            nested_step="${nested_step#checkpoint-}"
            if [[ "${nested_step}" =~ ^[0-9]+$ ]]; then
                while [[ "${nested_step}" == 0* ]]; do
                    nested_step="${nested_step#0}"
                done
                [[ -n "${nested_step}" ]] || nested_step=0
                if tactile_checkpoint_is_valid "${nested}" &&
                    { [[ -z "${best_nested}" ]] ||
                        ((${#nested_step} > ${#best_step})) ||
                        { ((${#nested_step} == ${#best_step})) &&
                            [[ "${nested_step}" > "${best_step}" ]]; }; }; then
                    best_nested="${nested}"
                    best_step="${nested_step}"
                fi
            fi
        done
        if [[ -n "${best_nested}" ]]; then
            printf '%s\n' "${best_nested}"
            return 0
        fi
    done
)

tactile_checkpoint_robot_key() {
    local model_path="${1%/}"
    local parent

    parent="$(dirname "${model_path}")"
    if [[ "$(basename "${model_path}")" == checkpoint-* ]]; then
        parent="$(dirname "${parent}")"
    fi
    basename "${parent}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    case "${1:-}" in
        discover)
            [[ $# -eq 3 ]] || exit 2
            tactile_checkpoint_discover "$2" "$3"
            ;;
        robot-key)
            [[ $# -eq 2 ]] || exit 2
            tactile_checkpoint_robot_key "$2"
            ;;
        *)
            echo "Usage: $0 {discover CHECKPOINT_ROOT TASK_ID|robot-key MODEL_PATH}" >&2
            exit 2
            ;;
    esac
fi
