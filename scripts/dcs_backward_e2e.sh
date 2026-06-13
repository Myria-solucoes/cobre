#!/usr/bin/env bash
# dcs_backward_e2e.sh
#
# OPERATOR-RUN end-to-end validation gate for Dynamic Cut Selection (DCS) in the
# backward pass. This script is run manually by the operator on a machine with
# MPI installed and an operator-supplied case directory (e.g. cobre_set_24_sc2).
# It is NEVER dispatched to an automated agent and is NOT part of `cargo test`
# or any CI job.
#
# It performs two checks:
#
#   1. RANK-COUNT INVARIANCE (bit-identical, exact `cmp`):
#      Run `method = "dynamic"` at 1 MPI rank and at 2 MPI ranks on the SAME
#      case directory and assert the convergence output is byte-for-byte
#      identical. DCS results must be invariant to the MPI rank count — this is
#      cobre's hard determinism rule, applied WITHIN the dynamic mode.
#
#   2. CROSS-MODE CONVERGED-BOUND AGREEMENT (relative tolerance):
#      Compare the CONVERGED lower/upper bound of a `method = "dynamic"` run
#      against an all-cuts run within a relative tolerance (default 1e-3).
#      DCS is exact at the optimum but takes a different lazy path, so
#      per-iteration bounds drift across modes (expected). Only the CONVERGED
#      bound is compared, never per-iteration values, and never bit-for-bit
#      across modes.
#
# Because the CLI selects the cut-selection method from the case's
# `config.json` (`training.cut_selection.method`), the operator supplies one
# case directory configured for `dynamic` and one configured for all-cuts
# (cut selection disabled / a non-dynamic method). The rank-invariance check
# uses the dynamic case directory.
#
# Usage:
#   cargo build --release --features mpi
#   bash scripts/dcs_backward_e2e.sh \
#       --dynamic-case  /path/to/cobre_set_24_sc2_dynamic \
#       --allcuts-case  /path/to/cobre_set_24_sc2_allcuts \
#       [--tol 1e-3] [--threads 4]
#
# Exit codes:
#   0  -- both checks passed
#   1  -- a check failed (rank divergence, or bound disagreement beyond --tol)
#   2  -- a prerequisite is missing (mpirun / cobre binary / case dir / python3)
#
# This is a manual verification script for machines with MPI installed. It is
# NOT intended for CI execution and adds no CI job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

COBRE_BIN="${REPO_ROOT}/target/release/cobre"
OUT_DIR="${REPO_ROOT}/target/dcs_backward_e2e"

# Defaults (overridable via flags).
TOL="1e-3"
THREADS="4"
DYNAMIC_CASE=""
ALLCUTS_CASE=""

timestamp() { date '+%Y-%m-%dT%H:%M:%S'; }
log() { echo "[$(timestamp)] $*"; }
error() { echo "[$(timestamp)] ERROR: $*" >&2; }

usage() {
    cat >&2 <<'USAGE'
Usage: dcs_backward_e2e.sh --dynamic-case DIR --allcuts-case DIR [--tol REL] [--threads N]

  --dynamic-case DIR   Case directory with training.cut_selection.method = "dynamic".
  --allcuts-case DIR   Case directory with the all-cuts configuration.
  --tol REL            Relative tolerance for the cross-mode converged-bound
                       comparison (default 1e-3 = 0.1%). Compares CONVERGED
                       bound only, never per-iteration.
  --threads N          Intra-node threads per run (default 4).
USAGE
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dynamic-case) DYNAMIC_CASE="$2"; shift 2 ;;
            --allcuts-case) ALLCUTS_CASE="$2"; shift 2 ;;
            --tol) TOL="$2"; shift 2 ;;
            --threads) THREADS="$2"; shift 2 ;;
            -h | --help) usage; exit 0 ;;
            *) error "unknown argument: $1"; usage; exit 2 ;;
        esac
    done
    if [[ -z "${DYNAMIC_CASE}" || -z "${ALLCUTS_CASE}" ]]; then
        error "--dynamic-case and --allcuts-case are required."
        usage
        exit 2
    fi
}

check_prerequisites() {
    if ! command -v mpirun >/dev/null 2>&1; then
        error "mpirun not found. Install OpenMPI or MPICH and ensure mpirun is on PATH."
        exit 2
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        error "python3 not found (needed to read converged bounds from metadata.json)."
        exit 2
    fi
    if [[ ! -x "${COBRE_BIN}" ]]; then
        error "target/release/cobre not found. Build with: cargo build --release --features mpi"
        exit 2
    fi
    if [[ ! -d "${DYNAMIC_CASE}" ]]; then
        error "dynamic case directory not found: ${DYNAMIC_CASE}"
        exit 2
    fi
    if [[ ! -d "${ALLCUTS_CASE}" ]]; then
        error "all-cuts case directory not found: ${ALLCUTS_CASE}"
        exit 2
    fi
}

# Run `cobre run` with the given rank count, case dir, and output dir.
run_cobre() {
    local ranks="$1" case_dir="$2" out="$3"
    mkdir -p "${out}"
    local rc=0
    if [[ "${ranks}" -eq 1 ]]; then
        "${COBRE_BIN}" run "${case_dir}" --output "${out}" --threads "${THREADS}" || rc=$?
    else
        mpirun -np "${ranks}" "${COBRE_BIN}" run "${case_dir}" --output "${out}" \
            --threads "${THREADS}" || rc=$?
    fi
    if [[ "${rc}" -ne 0 ]]; then
        error "cobre run failed (ranks=${ranks}, case=${case_dir}, exit=${rc})."
        exit "${rc}"
    fi
}

# Read bounds.final_lower_bound and bounds.final_upper_bound from a run's
# training/metadata.json. Echoes "LB UB" (UB is "nan" when absent).
read_bounds() {
    local out="$1"
    local meta="${out}/training/metadata.json"
    if [[ ! -f "${meta}" ]]; then
        error "expected metadata file not found: ${meta}"
        exit 1
    fi
    python3 - "${meta}" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    m = json.load(f)
b = m["bounds"]
lb = b["final_lower_bound"]
ub = b.get("final_upper_bound")
print(lb, ub if ub is not None else "nan")
PY
}

# Relative comparison: |a-b| / max(1, |b|) <= tol. Echoes "PASS"/"FAIL rel".
rel_compare() {
    local a="$1" b="$2" tol="$3"
    python3 - "$a" "$b" "$tol" <<'PY'
import math, sys
a, b, tol = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
if math.isnan(a) and math.isnan(b):
    print("PASS 0.0"); sys.exit(0)
if math.isnan(a) or math.isnan(b):
    print("FAIL nan-mismatch"); sys.exit(0)
rel = abs(a - b) / max(1.0, abs(b))
print(("PASS" if rel <= tol else "FAIL"), rel)
PY
}

main() {
    parse_args "$@"
    check_prerequisites

    rm -rf "${OUT_DIR}"
    log "============================================================"
    log "DCS backward end-to-end gate (OPERATOR-RUN)"
    log "Dynamic case : ${DYNAMIC_CASE}"
    log "All-cuts case: ${ALLCUTS_CASE}"
    log "Rel tolerance: ${TOL}   Threads: ${THREADS}"
    log "============================================================"

    # --- Check 1: rank-count invariance (dynamic, 1 vs 2 ranks) ---
    log "--- Check 1: rank-count invariance (method=dynamic, 1 vs 2 ranks) ---"
    run_cobre 1 "${DYNAMIC_CASE}" "${OUT_DIR}/dyn_rank1"
    run_cobre 2 "${DYNAMIC_CASE}" "${OUT_DIR}/dyn_rank2"

    local conv1="${OUT_DIR}/dyn_rank1/training/convergence.parquet"
    local conv2="${OUT_DIR}/dyn_rank2/training/convergence.parquet"
    if [[ ! -f "${conv1}" || ! -f "${conv2}" ]]; then
        error "convergence.parquet missing for one of the rank runs."
        exit 1
    fi
    local rank_ok=0
    if cmp --silent "${conv1}" "${conv2}"; then
        log "PASS: dynamic convergence.parquet byte-identical across 1 and 2 ranks."
    else
        error "FAIL: dynamic convergence.parquet differs between 1 and 2 ranks (rank-invariance violated)."
        rank_ok=1
    fi

    # --- Check 2: cross-mode converged-bound agreement (dynamic vs all-cuts) ---
    log "--- Check 2: cross-mode converged-bound agreement (dynamic vs all-cuts) ---"
    # Reuse the dynamic 1-rank run; produce a single-rank all-cuts run.
    run_cobre 1 "${ALLCUTS_CASE}" "${OUT_DIR}/allcuts_rank1"

    read -r dyn_lb dyn_ub < <(read_bounds "${OUT_DIR}/dyn_rank1")
    read -r all_lb all_ub < <(read_bounds "${OUT_DIR}/allcuts_rank1")
    log "Converged LB: dynamic=${dyn_lb}  all-cuts=${all_lb}"
    log "Converged UB: dynamic=${dyn_ub}  all-cuts=${all_ub}"

    local lb_res ub_res
    lb_res="$(rel_compare "${dyn_lb}" "${all_lb}" "${TOL}")"
    ub_res="$(rel_compare "${dyn_ub}" "${all_ub}" "${TOL}")"
    log "Converged LB rel-diff: ${lb_res} (tol ${TOL})"
    log "Converged UB rel-diff: ${ub_res} (tol ${TOL})"

    local bound_ok=0
    case "${lb_res} ${ub_res}" in
        "PASS"*" PASS"*) log "PASS: converged LB and UB agree within ${TOL}." ;;
        *) error "FAIL: converged bound disagreement beyond ${TOL}."; bound_ok=1 ;;
    esac

    log "============================================================"
    if [[ "${rank_ok}" -eq 0 && "${bound_ok}" -eq 0 ]]; then
        log "OVERALL: PASS"
        exit 0
    fi
    error "OVERALL: FAIL"
    exit 1
}

main "$@"
