#!/usr/bin/env bash
set -euo pipefail

# Multi-task sequential wrapper for GR00T_n15_Tactile_Cross eval. Forwards to script/eval_tasks_seq.sh.
# Usage: bash policy/GR00T_n15_Tactile_Cross/eval_tasks_seq.sh <TASK_ID>... [CHANNEL] [--key value ...] [--tmux]
# Example: bash policy/GR00T_n15_Tactile_Cross/eval_tasks_seq.sh 07 08 09 all --sii --headless --tmux

_d="$(cd "$(dirname "$0")" && pwd)"
_r="$(cd "${_d}/../.." && pwd)"
exec env POLICY=GR00T_n15_Tactile_Cross bash "${_r}/script/eval_tasks_seq.sh" "$@"
