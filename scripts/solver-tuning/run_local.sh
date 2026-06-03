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
# Cells are resumable: a cell whose result.json exists is skipped.

set -euo pipefail

readonly HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MANIFEST="${1:?usage: run_local.sh <manifest> <cases_dir> <runs_dir> <cobre_binary> [threads]}"
readonly CASES="${2:?cases dir}"
readonly RUNS="${3:?runs dir}"
readonly BIN="${4:?cobre binary}"
readonly THREADS="${5:-4}"

n="$(grep -c . "$MANIFEST")"
echo "Running $n cell(s) locally (threads=$THREADS) ..."
for ((i = 0; i < n; i++)); do
    python3 "$HERE/run_cell.py" \
        --manifest "$MANIFEST" \
        --index "$i" \
        --cases "$CASES" \
        --runs "$RUNS" \
        --binary "$BIN" \
        --threads "$THREADS"
done
echo "Done. Aggregate with: python3 $HERE/aggregate.py --runs $RUNS --backend <b> --stage <n>"
