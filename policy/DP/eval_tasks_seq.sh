#!/usr/bin/env bash
set -euo pipefail

# Multi-task sequential wrapper for DP eval. Forwards to script/eval_tasks_seq.sh.
# Usage: bash policy/DP/eval_tasks_seq.sh <TASK_ID>... [CHANNEL] [--key value ...] [--tmux]
# Example: bash policy/DP/eval_tasks_seq.sh 07 08 09 all --sii --headless --tmux

_d="$(cd "$(dirname "$0")" && pwd)"
_r="$(cd "${_d}/../.." && pwd)"
exec env POLICY=DP bash "${_r}/script/eval_tasks_seq.sh" "$@"
