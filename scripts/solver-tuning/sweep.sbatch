#!/usr/bin/env bash
#
# SLURM array job: run one tuning cell per array task, one full node each,
# 96 worker threads, no MPI.
#
# Submit (after prep_cases.sh + grid.py):
#   N=$(wc -l < manifests/highs.s1.jsonl)
#   TUNE_MANIFEST=manifests/highs.s1.jsonl \
#   TUNE_CASES=cases/highs TUNE_RUNS=runs TUNE_BIN=/path/to/cobre \
#   sbatch --array=0-$((N-1)) scripts/solver-tuning/sweep.sbatch
#
# Optional env: TUNE_THREADS (default 96), TUNE_REPO (cobre repo for git commit).
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

readonly HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
