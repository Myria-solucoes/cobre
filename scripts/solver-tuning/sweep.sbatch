#!/usr/bin/env bash
#
# SLURM array job: run one tuning cell per array task, one full node each,
# 96 worker threads, no MPI.
#
# Submit FROM the harness dir (the one holding run_cell.py) so SLURM_SUBMIT_DIR
# can locate it (see HERE below). After prep_cases.sh + grid.py:
#   cd /path/to/solver-tuning
#   N=$(wc -l < manifests/highs.s1.jsonl)
#   TUNE_MANIFEST=manifests/highs.s1.jsonl \
#   TUNE_CASES=cases/highs TUNE_RUNS=runs TUNE_BIN=/path/to/cobre \
#   sbatch --array=0-$((N-1)) sweep.sbatch
#
# Optional env: TUNE_THREADS (default 96), TUNE_REPO (cobre repo for git commit),
# TUNE_HARNESS_DIR (dir with run_cell.py; defaults to the sbatch submit dir).
#
#SBATCH --job-name=cobre-tune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --output=cobre-tune-%A_%a.out
#SBATCH --error=cobre-tune-%A_%a.err
# --- Cluster-specific: edit per site (a submit-time --partition/--time wins) ---
# Partition + walltime match the c7a cluster (decomp-c7a-48x, 8h cap). A single
# 10-iter tuning cell finishes well under 8h; raise --time only if a deeper case
# needs it, and stay within the partition's MaxTime.
#SBATCH --partition=decomp-c7a-48x
#SBATCH --time=08:00:00

set -euo pipefail

# SLURM copies the batch script to a spool dir and runs it from there, so
# ${BASH_SOURCE[0]} / $0 point at the spooled copy, NOT this file's real
# location. Resolve the harness dir from TUNE_HARNESS_DIR, else the submit dir
# (correct when you sbatch from the harness dir), else this script's dir (local
# / non-SLURM execution).
readonly HERE="${TUNE_HARNESS_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
if [[ ! -f "$HERE/run_cell.py" ]]; then
    echo "sweep.sbatch: run_cell.py not found in '$HERE'." >&2
    echo "  Submit from the harness dir, or set TUNE_HARNESS_DIR=/path/to/solver-tuning." >&2
    exit 1
fi
: "${TUNE_MANIFEST:?set TUNE_MANIFEST}"
: "${TUNE_CASES:?set TUNE_CASES}"
: "${TUNE_RUNS:?set TUNE_RUNS}"
: "${TUNE_BIN:?set TUNE_BIN}"
: "${SLURM_ARRAY_TASK_ID:?run via sbatch --array}"

readonly THREADS="${TUNE_THREADS:-${SLURM_CPUS_PER_TASK:-96}}"
# Pin any OpenMP-style libraries to the same width as cobre's worker pool.
export OMP_NUM_THREADS="$THREADS"

exec python3 "$HERE/run_cell.py" \
    --manifest "$TUNE_MANIFEST" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --cases "$TUNE_CASES" \
    --runs "$TUNE_RUNS" \
    --binary "$TUNE_BIN" \
    --threads "$THREADS" \
    --repo "${TUNE_REPO:-$HERE/../..}"
