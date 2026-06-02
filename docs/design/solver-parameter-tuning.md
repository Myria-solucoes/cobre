---
title: "Solver parameter tuning — procedure, benchmark case spec, and harness"
date: 2026-06-02
status: design
tags: [cobre, solver, highs, clp, tuning, benchmark, wall-clock]
companion:
  ["ideas/highs-solver-options-sddp.md", "ideas/clp-solver-options-sddp.md"]
---

# Solver parameter tuning

Goal: find the LP-solver parameter set that minimizes SDDP wall clock for each
backend (HiGHS, CLP) **independently**, with local threading and no MPI. Cobre's
runtime is largely solver-bound, and the dominant variable cost is the backward
pass solving many LPs sequentially per worker against a deep cut pool. The
production target is an HPC run (192 cores, 192 forward passes, 2 nodes); the
benchmark reproduces the _per-solve_ regime of that run at local scale.

## 1. Current configuration vs. the cost-model docs

The per-phase profiles live in `crates/cobre-sddp/src/solver_phase.rs`; the
HiGHS hardcoded options live in `crates/cobre-solver/src/highs.rs`
(`default_options`). Current values and where they diverge from the companion
cost-model docs (the tuning candidates):

### HiGHS

| Knob                                        | Current              | `highs-solver-options-sddp.md` rec  | Tune          |
| ------------------------------------------- | -------------------- | ----------------------------------- | ------------- |
| `presolve`                                  | `"on"` (all phases)  | `"off"` — "critical for warm-start" | ⭐ headline   |
| `simplex_dual_edge_weight_strategy`         | `1` Devex            | `2` SteepestEdge, pinned            | yes           |
| `simplex_scale_strategy`                    | `0` Off              | `2` Equilibration                   | yes           |
| `simplex_price_strategy`                    | fwd/sim `1`, bwd `2` | `3` RowSwitchColSwitch              | yes           |
| `dual_simplex_cost_perturbation_multiplier` | `0.0` off            | off                                 | aligned       |
| `primal/dual_feasibility_tolerance`         | `1e-9`               | `1e-7` default                      | optional axis |

> Enum-label note: the price-strategy integer→name mapping must be confirmed
> against the linked HiGHS version before interpreting results.

### CLP

| Knob                                | Current                | `clp-solver-options-sddp.md` rec | Tune                         |
| ----------------------------------- | ---------------------- | -------------------------------- | ---------------------------- |
| `perturbation`                      | `102` off              | `102`                            | aligned (tuning risks duals) |
| `dual_pricing_mode`                 | fwd/sim `3`, bwd `1`   | `1` pinned in backward           | yes (fwd/sim)                |
| `factorization_frequency`           | bwd `200`, fwd/sim `0` | `200` / tunable                  | yes (cadence)                |
| `scaling`                           | `0` off                | equilibrium `1` / auto `3` ok    | yes                          |
| `primal/dual_feasibility_tolerance` | `1e-9`                 | `1e-7` default                   | optional axis                |

## 2. The tuning seam (implemented)

Solver profiles are compile-time `const`s with no runtime override. A
**benchmark-only** override is wired so a single build per backend can sweep
many parameter sets. It is **entirely inert unless a `COBRE_TUNE_*` variable is
set** — the production path returns the compile-time const profile unchanged and
stays deterministic. Overrides are _global_ (applied to every phase).

- Profile-field overrides: `crates/cobre-sddp/src/solver_phase.rs` →
  `Phase::profile()` (module `tuning`).
- HiGHS `presolve` (not a profile field): `crates/cobre-solver/src/highs.rs` →
  `apply_profile` (`presolve_tuning_override`).

When any override is active, the effective settings are logged once to stderr
(`[cobre tune] …`) so each run records exactly what it executed.

### Environment-variable schema

HiGHS:

| Variable                       | Type | Values                                                |
| ------------------------------ | ---- | ----------------------------------------------------- |
| `COBRE_TUNE_HIGHS_PRESOLVE`    | str  | `on` \| `off` \| `choose`                             |
| `COBRE_TUNE_HIGHS_EDGE_WEIGHT` | i32  | `-1` Choose, `0` Dantzig, `1` Devex, `2` SteepestEdge |
| `COBRE_TUNE_HIGHS_SCALE`       | i32  | `0` Off … `4` (see HiGHS)                             |
| `COBRE_TUNE_HIGHS_PRICE`       | i32  | `0`–`3`                                               |
| `COBRE_TUNE_HIGHS_PRIMAL_TOL`  | f64  | e.g. `1e-7`                                           |
| `COBRE_TUNE_HIGHS_DUAL_TOL`    | f64  | e.g. `1e-7`                                           |
| `COBRE_TUNE_HIGHS_ITER_LIMIT`  | u32  | per-attempt cap (`0` = heuristic)                     |

CLP:

| Variable                      | Type | Values                             |
| ----------------------------- | ---- | ---------------------------------- |
| `COBRE_TUNE_CLP_PERTURBATION` | i32  | `50` \| `100` \| `102`             |
| `COBRE_TUNE_CLP_SCALING`      | i32  | `0`–`4`                            |
| `COBRE_TUNE_CLP_PRICING_MODE` | i32  | `0`–`3` (`1` full DSE)             |
| `COBRE_TUNE_CLP_FACTOR_FREQ`  | i32  | `0` default, else refactor cadence |
| `COBRE_TUNE_CLP_PRIMAL_TOL`   | f64  |                                    |
| `COBRE_TUNE_CLP_DUAL_TOL`     | f64  |                                    |
| `COBRE_TUNE_CLP_ITER_LIMIT`   | u32  |                                    |
| `COBRE_TUNE_CLP_ALGORITHM`    | str  | `dual` \| `primal`                 |

A present-but-malformed value logs a warning and is ignored (treated as unset).

## 3. Benchmark case spec (to build)

The full case specification — system sizes, acceptance criteria, and the three
run-mode `config.json` templates (A correctness / B small-fast / C scale probe)
— lives in its own document: **`docs/design/solver-tuning-benchmark-case.md`**.

In brief: one synthetic-but-realistic hydrothermal study (~15–25 reservoir
hydros, ~15–30 thermals, 60 monthly stages, PAR(1–2), 20–40 backward openings),
sized so LP solve time dominates wall time and the cut pool can grow deep. The
same system drives all three modes; Mode C pre-builds a deep cut pool once
(`store_basis`) and then warm-starts a 1–2-iteration probe — the production
per-solve regime.

## 4. Full-factorial grid (per backend)

HiGHS axes (confirm price-enum labels first):

- `presolve` ∈ {on, off}
- `edge_weight` ∈ {1 Devex, 2 SteepestEdge} (+ optionally 0 Dantzig)
- `scale` ∈ {0 off, 2 equilibration}
- `price` ∈ {1, 2, 3}
- (optional) tolerance ∈ {1e-9, 1e-7}

→ 2×2×2×3 = **24** combos (×2 tolerance = 48; ×3 edge = 36/72).

CLP axes:

- `scaling` ∈ {0, 1, 3}
- `pricing_mode` ∈ {1, 2, 3}
- `factor_freq` ∈ {0, 100, 200, 400}
- (optional, correctness-sensitive) `perturbation` ∈ {102, 100}

→ 3×3×4 = **36** combos (×2 perturbation = 72).

Each combo is run on Mode C (primary), with the survivors re-checked on Mode A
(correctness gate) and Mode B (small-problem regression). `perturbation=100` and
`primal=true` are correctness-sensitive — flag, don't silently accept a speedup
that perturbs duals or changes the policy.

## 5. Metrics, correctness guard, noise control

**Metrics** (`<out>/training/metadata.json`):

- `solve_stats.backward_solve_seconds` — primary (the dominant, variance-heavy phase)
- `solve_stats.forward_solve_seconds`, `duration_seconds` — secondary
- `solve_stats.{retried, failed}` — a config that increases retries is suspect
- Per-iteration `<out>/training/timing/iterations.parquet`:
  `backward_wall_ms`, `bwd_load_imbalance_ms` — backward variance/tail

**Correctness guard** (every variant): final lower bound and Mode-A first-stage
cost within relative tolerance of the frozen reference. Set the tolerance from
the baseline's run-to-run drift; bit-identity is _not_ expected across configs
(different pivots → same optimum within solver tolerance). Any variant failing
the guard is rejected regardless of speed.

**Noise control**: pin `--threads`, `--quiet`, otherwise-idle machine; discard a
warm-up run; run N=5 per variant; report **median and min** (min ≈ least-
contended truth) and Δ% vs. baseline. Same system, same seed, same frozen policy
for all Mode-C variants.

**Significance**: <±2% noise; ±2–5% borderline (re-run); >±5% real.

## 6. Harness flow (per backend)

```
build once:  cargo build --release [--no-default-features --features clp]
prebuild:    run Mode-C phase 1 (baseline) → MASTER/policy/  (store_basis)
for combo in grid:
  for rep in 1..=N:
    out=$(fresh dir); cp -r MASTER/policy out/<policy.path>
    COBRE_TUNE_<BACKEND>_* … cobre run <case> --output out --threads <cores> --quiet
    parse out/training/metadata.json → append (combo, rep, metrics) to results.csv
aggregate:   median/min per combo, Δ% vs baseline, correctness pass/fail
confirm:     re-run survivors on Mode A (gate) and Mode B (regression)
```

Results are appended incrementally (resumable). The `[cobre tune]` stderr line
from each run is captured alongside its metrics to verify the active config.

## 7. Open items before execution

- Confirm the HiGHS `simplex_price_strategy` integer→name mapping for the linked version.
- Build the benchmark study to §3 and pick `K` (deep-pool target) via `probe_k_disaggregated`.
- Decide the correctness tolerance from baseline run-to-run drift.
