#!/bin/bash
# Append 4-channel eval results to every_eval.csv after evaluation completes.
#
# Usage:
#   bash script/_append_eval_result.sh <task_num> <task_name> <search_dir>
#
#   search_dir  The eval root directory containing success_rates.tsv, or
#               the end_eval/<TASK>/ parent dir containing channel subdirs.
#
# Output (appended to RESULT_FILE):
#   ==========================================
#   Task: 03 (some_task_name)  Policy: DP
#   Start time: 2026-09-01 16:16
#   End time:   2026-09-01 22:39:08
#   Elapsed:    6h23m
#   (blank line)
#   <contents of latest success_rates.tsv>
#   (blank line)
set -euo pipefail

TASK_NUM="${1:?Usage: _append_eval_result.sh <task_num> <task_name> <search_dir>}"
TASK_NAME="${2:?}"
SEARCH_DIR="${3:?}"

RESULT_FILE="${EVERY_EVAL_RESULT_FILE:-../jcy/result/every_eval.csv}"

# Find the most recent success_rates.tsv under the search dir
# Look both directly in the dir and in subdirectories
TSV=""
if [[ -f "${SEARCH_DIR}/success_rates.tsv" ]]; then
    TSV="${SEARCH_DIR}/success_rates.tsv"
else
    TSV=$(ls -t "${SEARCH_DIR}"/*/success_rates.tsv 2>/dev/null | head -1 || true)
fi

if [[ -z "${TSV}" ]]; then
    echo "[Eval] WARNING: no success_rates.tsv found in ${SEARCH_DIR}" >&2
    exit 0
fi

# Auto-detect policy from the eval directory name
# dp_all_...  → DP
# gr00t_all_... → GR00T_n15
_eval_dir="$(dirname "${TSV}")"
_eval_name="$(basename "${_eval_dir}")"
POLICY=""
case "${_eval_name}" in
    dp_*)         POLICY="DP" ;;
    gr00t_*)      POLICY="GR00T_n15" ;;
    *)            POLICY="unknown" ;;
esac

mkdir -p "$(dirname "${RESULT_FILE}")"

# Format TSV with aligned columns
_aligned_tsv=$(python3 -c "
import sys
lines = open('${TSV}').read().strip().split('\n')
rows = [l.split('\t') for l in lines]
rows = [r for r in rows if any(c for c in r)]
if not rows: sys.exit(0)
ncols = max(len(r) for r in rows)
widths = [0] * ncols
for r in rows:
    for i in range(min(len(r), ncols)):
        widths[i] = max(widths[i], len(r[i]))
for r in rows:
    padded = [r[i].ljust(widths[i]) if i < len(r) else ''.ljust(widths[i]) for i in range(ncols)]
    print('  ' + '  '.join(padded))
")

{
    printf '==========================================\n'
    printf 'Task: %s (%s)  Policy: %s\n' "${TASK_NUM}" "${TASK_NAME}" "${POLICY}"
    printf 'Eval time: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf '\n'
    printf '%s\n' "${_aligned_tsv}"
    printf '\n'
} >> "${RESULT_FILE}"

echo "[Eval] Results appended to ${RESULT_FILE} (source: ${TSV}, policy: ${POLICY})"
