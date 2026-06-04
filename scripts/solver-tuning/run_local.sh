#!/usr/bin/env bash
#
# SLURM-free local runner: run every cell of a manifest sequentially on this
# machine. For testing the harness end-to-end on a small case before submitting
# the real sweep to SLURM (use sweep.sbatch there).
#
# Usage:
#   run_local.sh <manifest.jsonl> <cases_dir> <runs_dir> <cobre_binary> [threads]
#
# Example (small case, 4 threads):
#   ./prep_cases.sh examples/4ree cases/local
#   python3 grid.py --backend highs --stage 1 > m.s1.jsonl
#   ./run_local.sh m.s1.jsonl cases/local runs $PWD/target/release/cobre 4
#   python3 aggregate.py --runs runs --backend highs --stage 1
#
# Cells are resumable (a cell whose result.json exists is skipped) and failures
# are non-fatal: a cell that errors is logged and the run continues to the next
# cell; the script exits non-zero at the end if any cell failed.

set -euo pipefail

readonly HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MANIFEST="${1:?usage: run_local.sh <manifest> <cases_dir> <runs_dir> <cobre_binary> [threads]}"
readonly CASES="${2:?cases dir}"
readonly RUNS="${3:?runs dir}"
readonly BIN="${4:?cobre binary}"
readonly THREADS="${5:-4}"

n="$(grep -c . "$MANIFEST")"
echo "Running $n cell(s) locally (threads=$THREADS) ..."

# Continue-on-failure: a single cell that errors (e.g. an infeasible subproblem
# under an aggressive solver profile) must not abort the remaining cells. Capture
# each cell's exit code without tripping `set -e` (the `|| rc=$?` idiom), log it,
# and keep going; exit non-zero at the end if any cell failed.
declare -a failed=()
for ((i = 0; i < n; i++)); do
    rc=0
    python3 "$HERE/run_cell.py" \
        --manifest "$MANIFEST" \
        --index "$i" \
        --cases "$CASES" \
        --runs "$RUNS" \
        --binary "$BIN" \
        --threads "$THREADS" || rc=$?
    if ((rc != 0)); then
        echo "[warn] cell index $i exited $rc (continuing)"
        failed+=("$i")
    fi
done

readonly AGG="python3 $HERE/aggregate.py --runs $RUNS --backend <b> --stage <n>"
if ((${#failed[@]} > 0)); then
    echo "Done with ${#failed[@]}/$n cell(s) failed at index(es): ${failed[*]} — re-run to retry (completed cells skip)."
    echo "Aggregate with: $AGG"
    exit 1
fi
echo "Done. Aggregate with: $AGG"
