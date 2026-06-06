# DCS backward end-to-end validation gate (operator-run)

`dcs_backward_e2e.sh` is the manual end-to-end gate that signs off Dynamic Cut
Selection (DCS) in the backward pass. It is **operator-run only** — run by hand
on a machine with MPI installed against an operator-supplied case directory
(e.g. `cobre_set_24_sc2`). It is **never dispatched to an automated agent**, is
**not** part of `cargo test`, and adds **no CI job** (long-running/production
validation stays manual).

The fast, automated correctness gates (DCS-vs-all-cuts cut exactness within
`1e-9`, finite-`k1` window, run-to-run determinism) live in the in-crate test
suite (`cargo test -p cobre-sddp`, both `highs` and `clp`, plus a
`--features slow-tests` sweep). This script covers the two checks that need a
full multi-stage run and MPI:

1. **Rank-count invariance (bit-identical).** `method = "dynamic"` at 1 MPI rank
   vs 2 MPI ranks on the same case must produce **byte-for-byte identical**
   convergence output. This is cobre's hard determinism rule, applied WITHIN the
   dynamic mode. The check is an exact `cmp` of `training/convergence.parquet`.

2. **Cross-mode converged-bound agreement (relative tolerance).** A
   `method = "dynamic"` run vs an all-cuts run must agree on the **converged**
   lower/upper bound within a relative tolerance (default `1e-3` = 0.1%). DCS is
   exact at the optimum but takes a different lazy solve path, so per-iteration
   bounds drift across modes — this is expected. The gate therefore compares the
   **converged** bound only (read from `training/metadata.json` →
   `bounds.final_lower_bound` / `bounds.final_upper_bound`), **never**
   per-iteration values and **never** bit-for-bit across modes.

## What is and is not compared

| Comparison            | Mode pair                        | Criterion                           |
| --------------------- | -------------------------------- | ----------------------------------- |
| `convergence.parquet` | dynamic 1-rank vs dynamic 2-rank | exact `cmp` (bit-identical)         |
| converged LB / UB     | dynamic vs all-cuts              | relative ≤ `--tol` (default `1e-3`) |

Do **not** expect bit-for-bit equality between dynamic and all-cuts modes — only
within the dynamic mode across rank counts.

## Why two case directories

The cut-selection method is read from the case's
`config.json` (`training.cut_selection.method`); there is no CLI override. The
operator therefore supplies:

- `--dynamic-case` — a case directory configured with
  `training.cut_selection.method = "dynamic"`.
- `--allcuts-case` — the same case configured for all-cuts (cut selection
  disabled, or a non-dynamic method).

Both are operator-supplied and are **not committed to the repository**. The
rank-invariance check uses the dynamic case directory.

## Running

```bash
cargo build --release --features mpi

bash scripts/dcs_backward_e2e.sh \
    --dynamic-case  /path/to/cobre_set_24_sc2_dynamic \
    --allcuts-case  /path/to/cobre_set_24_sc2_allcuts \
    --tol 1e-3 \
    --threads 4
```

- `--tol` (default `1e-3`): relative tolerance for the converged-bound
  comparison. Tighten or loosen based on the case's observed cross-mode drift.
- `--threads` (default `4`): intra-node threads per run.

Exit codes: `0` both checks passed; `1` a check failed; `2` a prerequisite is
missing (`mpirun`, the `cobre` binary, a case directory, or `python3`).

## Definition of Done

This gate passes when, on the operator's `cobre_set_24_sc2` (or equivalent):

- [ ] `method = "dynamic"` at 1 rank and 2 ranks produce byte-identical
      `training/convergence.parquet` (rank-count invariance).
- [ ] The converged LB and UB of the dynamic run agree with the all-cuts run
      within the relative tolerance (default `1e-3`).

This gate is **operator-run and is never agent-dispatched**. The operator
supplies and runs the case directories; they are not committed to the
repository, and no CI job runs this script or flips `--features slow-tests`.
