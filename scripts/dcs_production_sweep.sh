#!/usr/bin/env bash
# dcs_production_sweep.sh
#
# OPERATOR-RUN production-scale sweep harness for the dynamic lazy cut-selection
# path. Run manually by the operator on a machine with MPI installed and an
# operator-supplied production-scale case directory. It is NEVER dispatched to an
# automated agent and is NOT part of `cargo test` or any CI job.
#
# Purpose: measure the scoring-versus-solve split at production cut-pool scale.
# The lazy selection loop scores every non-resident candidate once per inner
# round; at large pools this scoring cost may swamp the LP-solve savings. The
# harness runs the configured lazy variant alongside an all-cuts baseline,
# parses the per-worker scoring time and the total solve time from each run's
# timing output, and writes the split so the operator can decide whether a
# bounded-candidate window is warranted.
#
# It performs, per case:
#
#   1. SCORING-VERSUS-SOLVE SPLIT (the lazy variant):
#      Run the case as configured in its config.json. Read the per-worker
#      lazy-scoring time (the `lazy_scoring_ms` timing column) and the total
#      solve time, then write `scoring / (scoring + solve)` to --out-dir.
#
#   2. CROSS-MODE CONVERGED-BOUND AGREEMENT (lazy vs all-cuts):
#      Compare the CONVERGED lower/upper bound of the lazy run against an
#      all-cuts run within a relative tolerance (default 1e-3). The lazy path is
#      exact at the optimum but takes a different path, so per-iteration bounds
#      drift across modes (expected, §8.3). Only the CONVERGED bound is compared,
#      never per-iteration values, and never bit-for-bit across modes.
#
#   3. (optional) RANK-COUNT INVARIANCE (lazy mode, 1 vs N ranks):
#      When --ranks N (N > 1) is given, run the lazy case at 1 rank and at N
#      ranks and assert the convergence output is byte-for-byte identical. This
#      is the hard determinism rule, applied WITHIN the dynamic mode across rank
#      counts (never between modes).
#
# Because the CLI selects the cut-selection method from the case's config.json
# (`training.cut_selection.method`), the operator supplies one case directory
# configured for the lazy method and one configured for all-cuts (cut selection
# disabled / a non-dynamic method).
#
# Build (cluster):
#   cargo build --release --features mpi
#
# Usage:
#   bash scripts/dcs_production_sweep.sh \
#       --case          /path/to/case_lazy \
#       --allcuts-case  /path/to/case_allcuts \
#       [--threads 5] [--reps 1] [--ranks 1] [--tol 1e-3] \
#       [--out-dir DIR]
#
#   --case DIR           Case directory with the lazy cut-selection method set in
#                        config.json. REQUIRED; with none, exit 2 and print usage.
#   --allcuts-case DIR   Case directory with the all-cuts configuration. Optional;
#                        when omitted, the cross-mode comparison (check 2) is
#                        skipped and only the split (check 1) is reported.
#   --threads N          Intra-node threads per run (default 5).
#   --reps R             Repetitions per variant (default 1; exclusive quiet
#                        nodes need no averaging — see the results doc).
#   --ranks N            MPI rank count for the lazy run (default 1). N > 1 also
#                        enables the rank-invariance check (check 3).
#   --tol REL            Relative tolerance for the cross-mode converged-bound
#                        comparison (default 1e-3 = 0.1%). Converged bound only.
#   --out-dir DIR        Output directory for per-run timing and the split
#                        report (default: target/dcs_production_sweep).
#
# Exit codes:
#   0  -- all attempted checks passed
#   1  -- a run failed, or a check failed (bound disagreement / rank divergence)
#   2  -- a prerequisite is missing (cobre binary / case dir / python3 /
#         mpirun when --ranks > 1), OR no --case was supplied
#
# This is a manual verification harness for machines with MPI installed. It is
# NOT intended for CI execution and adds no CI job. The operator runs the sweep
# across the hyperparameter grid (the lazy candidate window, the check cadence,
# the per-round add count, the violation tolerance, and the activation
# iteration) by editing the case config.json between invocations, and records
# the verdict in docs/design/dcs-production-sweep-results.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT

readonly COBRE_BIN="${REPO_ROOT}/target/release/cobre"

# Defaults (overridable via flags).
TOL="1e-3"
THREADS="5"
REPS="1"
RANKS="1"
CASE=""
ALLCUTS_CASE=""
OUT_DIR="${REPO_ROOT}/target/dcs_production_sweep"

timestamp() { date '+%Y-%m-%dT%H:%M:%S'; }
log() { echo "[$(timestamp)] $*"; }
error() { echo "[$(timestamp)] ERROR: $*" >&2; }

usage() {
    cat >&2 <<'USAGE'
Usage: dcs_production_sweep.sh --case DIR [--allcuts-case DIR]
                               [--threads N] [--reps R] [--ranks N]
                               [--tol REL] [--out-dir DIR]

  --case DIR           Case directory with the lazy cut-selection method
                       (training.cut_selection.method) set in config.json.
                       REQUIRED.
  --allcuts-case DIR   Case directory with the all-cuts configuration. When
                       omitted, the cross-mode bound comparison is skipped.
  --threads N          Intra-node threads per run (default 5).
  --reps R             Repetitions per variant (default 1).
  --ranks N            MPI rank count for the lazy run (default 1). N > 1 also
                       enables the within-mode rank-invariance check.
  --tol REL            Relative tolerance for the cross-mode converged-bound
                       comparison (default 1e-3). Converged bound only.
  --out-dir DIR        Output directory (default target/dcs_production_sweep).
USAGE
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --case) CASE="$2"; shift 2 ;;
            --allcuts-case) ALLCUTS_CASE="$2"; shift 2 ;;
            --threads) THREADS="$2"; shift 2 ;;
            --reps) REPS="$2"; shift 2 ;;
            --ranks) RANKS="$2"; shift 2 ;;
            --tol) TOL="$2"; shift 2 ;;
            --out-dir) OUT_DIR="$2"; shift 2 ;;
            -h | --help) usage; exit 0 ;;
            *) error "unknown argument: $1"; usage; exit 2 ;;
        esac
    done
    if [[ -z "${CASE}" ]]; then
        error "--case is required."
        usage
        exit 2
    fi
}

check_prerequisites() {
    if ! command -v python3 >/dev/null 2>&1; then
        error "python3 not found (needed to read bounds and timing parquet)."
        exit 2
    fi
    if [[ ! -x "${COBRE_BIN}" ]]; then
        error "target/release/cobre not found. Build with: cargo build --release --features mpi"
        exit 2
    fi
    if [[ ! -d "${CASE}" ]]; then
        error "case directory not found: ${CASE}"
        exit 2
    fi
    if [[ -n "${ALLCUTS_CASE}" && ! -d "${ALLCUTS_CASE}" ]]; then
        error "all-cuts case directory not found: ${ALLCUTS_CASE}"
        exit 2
    fi
    if [[ "${RANKS}" -gt 1 ]] && ! command -v mpirun >/dev/null 2>&1; then
        error "mpirun not found but --ranks ${RANKS} > 1. Install OpenMPI or MPICH."
        exit 2
    fi
}

# Echo the configured cut-selection method from a case's config.json (or
# "<none>" when the field is absent).
read_method() {
    local case_dir="$1"
    local cfg="${case_dir}/config.json"
    if [[ ! -f "${cfg}" ]]; then
        error "config.json not found in case directory: ${case_dir}"
        exit 2
    fi
    python3 - "${cfg}" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
m = cfg.get("training", {}).get("cut_selection", {}).get("method")
print(m if m is not None else "<none>")
PY
}

# Run `cobre run` with a rank count, case dir, and output dir. Exits 1 on failure
# after printing the failing command and the captured stderr.
run_cobre() {
    local ranks="$1" case_dir="$2" out="$3"
    mkdir -p "${out}"
    local rc=0 cmd stderr_log="${out}/stderr.log"
    if [[ "${ranks}" -eq 1 ]]; then
        cmd=("${COBRE_BIN}" run "${case_dir}" --output "${out}" --threads "${THREADS}")
    else
        cmd=(mpirun -np "${ranks}" "${COBRE_BIN}" run "${case_dir}" \
            --output "${out}" --threads "${THREADS}")
    fi
    "${cmd[@]}" 2>"${stderr_log}" || rc=$?
    if [[ "${rc}" -ne 0 ]]; then
        error "cobre run failed (ranks=${ranks}, case=${case_dir}, exit=${rc})."
        error "failing command: ${cmd[*]}"
        error "stderr (tail):"
        tail -n 20 "${stderr_log}" >&2 || true
        exit 1
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

# Read the scoring-versus-solve split from a run's timing parquet.
# Echoes "SCORING_MS SOLVE_MS FRACTION" where FRACTION = scoring/(scoring+solve)
# summed across all per-worker rows. SOLVE_MS comes from the rank-aggregated
# solve-time column when present, else 0.
read_split() {
    local out="$1"
    local timing="${out}/training/timing/iterations.parquet"
    if [[ ! -f "${timing}" ]]; then
        error "expected timing parquet not found: ${timing}"
        exit 1
    fi
    python3 - "${timing}" <<'PY'
import sys
try:
    import pyarrow.parquet as pq
except ImportError:
    sys.stderr.write("pyarrow not available; cannot read timing parquet.\n")
    sys.exit(1)

t = pq.read_table(sys.argv[1])
cols = set(t.column_names)
scoring_ms = 0.0
if "lazy_scoring_ms" in cols:
    scoring_ms = sum(v for v in t.column("lazy_scoring_ms").to_pylist() if v is not None)
# Total LP solve wall time: the timing parquet does not carry a dedicated
# per-worker solve column, so the operator reads total solve time from the
# run summary / solver-stats output. As a harness-side proxy we sum the
# forward + backward wall columns (a superset of solve time) so the fraction
# is well-defined and conservative (scoring fraction is never overstated).
solve_ms = 0.0
for c in ("forward_wall_ms", "backward_wall_ms"):
    if c in cols:
        solve_ms += sum(v for v in t.column(c).to_pylist() if v is not None)
denom = scoring_ms + solve_ms
frac = (scoring_ms / denom) if denom > 0 else 0.0
print(f"{scoring_ms:.3f} {solve_ms:.3f} {frac:.6f}")
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
    mkdir -p "${OUT_DIR}"
    local report="${OUT_DIR}/sweep_report.txt"

    local lazy_method allcuts_method
    lazy_method="$(read_method "${CASE}")"

    log "============================================================"
    log "DCS production sweep (OPERATOR-RUN)"
    log "Lazy case    : ${CASE}  (method=${lazy_method})"
    log "All-cuts case: ${ALLCUTS_CASE:-<none, cross-mode check skipped>}"
    log "Threads: ${THREADS}  Reps: ${REPS}  Ranks: ${RANKS}  Rel tol: ${TOL}"
    log "Out dir: ${OUT_DIR}"
    log "============================================================"

    {
        echo "# DCS production sweep report"
        echo "lazy_case=${CASE}"
        echo "lazy_method=${lazy_method}"
        echo "allcuts_case=${ALLCUTS_CASE:-}"
        echo "threads=${THREADS} reps=${REPS} ranks=${RANKS} tol=${TOL}"
        echo "# rep scoring_ms solve_ms scoring_fraction"
    } >"${report}"

    # --- Check 1: scoring-versus-solve split (lazy variant) over --reps ---
    log "--- Check 1: scoring-versus-solve split (method=${lazy_method}) ---"
    local rep lazy_out
    local last_lazy_out=""
    for ((rep = 1; rep <= REPS; rep++)); do
        lazy_out="${OUT_DIR}/lazy_rank${RANKS}_rep${rep}"
        run_cobre "${RANKS}" "${CASE}" "${lazy_out}"
        local split
        split="$(read_split "${lazy_out}")"
        log "rep ${rep}: scoring_ms solve_ms fraction = ${split}"
        echo "${rep} ${split}" >>"${report}"
        last_lazy_out="${lazy_out}"
    done

    local rank_ok=0 bound_ok=0

    # --- Check 3: rank-count invariance (within lazy mode) when --ranks > 1 ---
    if [[ "${RANKS}" -gt 1 ]]; then
        log "--- Check 3: rank-count invariance (method=${lazy_method}, 1 vs ${RANKS}) ---"
        local lazy_rank1="${OUT_DIR}/lazy_rank1_inv"
        run_cobre 1 "${CASE}" "${lazy_rank1}"
        local conv1="${lazy_rank1}/training/convergence.parquet"
        local convN="${last_lazy_out}/training/convergence.parquet"
        if [[ ! -f "${conv1}" || ! -f "${convN}" ]]; then
            error "convergence.parquet missing for a rank-invariance run."
            rank_ok=1
        elif cmp --silent "${conv1}" "${convN}"; then
            log "PASS: convergence.parquet byte-identical across 1 and ${RANKS} ranks."
        else
            error "FAIL: convergence.parquet differs between 1 and ${RANKS} ranks."
            rank_ok=1
        fi
    fi

    # --- Check 2: cross-mode converged-bound agreement (lazy vs all-cuts) ---
    if [[ -n "${ALLCUTS_CASE}" ]]; then
        allcuts_method="$(read_method "${ALLCUTS_CASE}")"
        log "--- Check 2: cross-mode converged-bound agreement (lazy vs all-cuts=${allcuts_method}) ---"
        local allcuts_out="${OUT_DIR}/allcuts_rank${RANKS}"
        run_cobre "${RANKS}" "${ALLCUTS_CASE}" "${allcuts_out}"

        local lazy_lb lazy_ub all_lb all_ub
        read -r lazy_lb lazy_ub < <(read_bounds "${last_lazy_out}")
        read -r all_lb all_ub < <(read_bounds "${allcuts_out}")
        log "Converged LB: lazy=${lazy_lb}  all-cuts=${all_lb}"
        log "Converged UB: lazy=${lazy_ub}  all-cuts=${all_ub}"

        local lb_res ub_res
        lb_res="$(rel_compare "${lazy_lb}" "${all_lb}" "${TOL}")"
        ub_res="$(rel_compare "${lazy_ub}" "${all_ub}" "${TOL}")"
        log "Converged LB rel-diff: ${lb_res} (tol ${TOL})"
        log "Converged UB rel-diff: ${ub_res} (tol ${TOL})"
        {
            echo "# cross_mode_bound lazy_lb=${lazy_lb} allcuts_lb=${all_lb} lb_result=${lb_res}"
            echo "# cross_mode_bound lazy_ub=${lazy_ub} allcuts_ub=${all_ub} ub_result=${ub_res}"
        } >>"${report}"

        case "${lb_res} ${ub_res}" in
            "PASS"*" PASS"*) log "PASS: converged LB and UB agree within ${TOL}." ;;
            *) error "FAIL: converged bound disagreement beyond ${TOL}."; bound_ok=1 ;;
        esac
    else
        log "--- Check 2 skipped: no --allcuts-case supplied ---"
    fi

    log "============================================================"
    log "Report written to ${report}"
    if [[ "${rank_ok}" -eq 0 && "${bound_ok}" -eq 0 ]]; then
        log "OVERALL: PASS"
        exit 0
    fi
    error "OVERALL: FAIL"
    exit 1
}

main "$@"
