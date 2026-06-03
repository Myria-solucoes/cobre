---
title: "Solver parameter tuning — procedure, benchmark case spec, and harness"
date: 2026-06-02
status: design
tags: [cobre, solver, highs, clp, tuning, benchmark, wall-clock]
companion:
  ["ideas/highs-solver-options-sddp.md", "ideas/clp-solver-options-sddp.md"]
references: ["ideas/research-report.md"]
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

### CLP

| Knob                                | Current                | `clp-solver-options-sddp.md` rec | Tune                         |
| ----------------------------------- | ---------------------- | -------------------------------- | ---------------------------- |
| `perturbation`                      | `102` off              | `102`                            | aligned (tuning risks duals) |
| `dual_pricing_mode`                 | fwd/sim `3`, bwd `1`   | `1` pinned in backward           | yes (fwd/sim)                |
| `factorization_frequency`           | bwd `200`, fwd/sim `0` | `200` / tunable                  | yes (cadence)                |
| `scaling`                           | `0` off                | equilibrium `1` / auto `3` ok    | yes                          |
| `primal/dual_feasibility_tolerance` | `1e-9`                 | `1e-7` default                   | optional axis                |

## 1a. Correctness floor vs. pure speed levers (source-grounded)

`ideas/research-report.md` (source-grounded against the vendored HiGHS / CLP)
sharpens the framing: the knobs split into a **correctness floor that must be
SET, not tuned**, and **pure speed levers that are swept**.

**Correctness floor** — protects both cut validity (the cut gradient _is_ the
dual vector) and the warm start:

| Floor knob           | HiGHS                                           | CLP                                | cobre status                    |
| -------------------- | ----------------------------------------------- | ---------------------------------- | ------------------------------- |
| algorithm            | `simplex_strategy=1` (dual serial)              | `model.dual()`                     | ✅ default                      |
| perturbation         | `dual_simplex_cost_perturbation_multiplier=0.0` | `perturbation=102`                 | ✅ default                      |
| dual feasibility tol | `1e-9`                                          | `1e-9`                             | ✅ default                      |
| presolve             | **`off`**                                       | n/a (dual() path doesn't presolve) | ⚠️ **cobre ships `on` (SS4.1)** |

cobre's compiled defaults already meet the floor **except HiGHS `presolve`**
(shipped `on` per spec SS4.1; the report argues `off` is required so the saved
basis stays index-for-index valid). So the floor is one env var
(`COBRE_TUNE_HIGHS_PRESOLVE=off`) and the experiment confirms it empirically with
a single `presolve=on` cell.

**Pure speed levers** — swept on top of the floor (no dual-accuracy risk):

| Lever                 | HiGHS                                          | CLP                           | Report's hypothesis                                                                                       |
| --------------------- | ---------------------------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------- |
| edge weight / pricing | `simplex_dual_edge_weight_strategy` ∈ {-1,1,2} | `dual_pricing_mode` ∈ {0,1,3} | few-pivot resolves under-amortize SteepestEdge init → Devex (HiGHS) / uninitialized weights (CLP) may win |
| price strategy        | `simplex_price_strategy=3`                     | implicit                      | RowSwitchColSwitch suits sparse LPs                                                                       |
| scaling               | `simplex_scale_strategy` 0 vs 2                | `scaling` 0 vs 1              | hydro LPs moderately scaled; cobre prescaler already conditions                                           |
| factorization freq    | n/a                                            | `factorization_frequency`     | **near-irrelevant** for few-pivot resolves                                                                |

**Enum values — CONFIRMED from the vendored HiGHS source**
(`crates/cobre-solver/vendor/HiGHS/highs/simplex/SimplexConst.h`):

- `SimplexPriceStrategy` = **0 Col · 1 Row · 2 RowSwitch · 3 RowSwitchColSwitch**
  (so cobre's backward `price=2` is _RowSwitch_; the recommended default is `3`).
- `SimplexEdgeWeightStrategy` = **-1 Choose · 0 Dantzig · 1 Devex · 2 SteepestEdge**
  (range `{-1,2}`; the OPTANO `{-1,3}` was wrong).
- `SimplexStrategy` 1 = Dual serial.

(Still confirm the HiGHS `simplex_scale_strategy` equilibration index.)

**Correctness gate — cut validity.** Per the report, the decisive correctness
check is that a config does not produce an _invalid_ cut: at convergence the
lower bound must not exceed the upper bound (`LB ≤ UB`). `LB > UB` means a cut
sliced off the true optimum (perturbation / loose dual tolerance). The harness
flags this as `INVALID` (distinct from a run `FAIL`). This is stronger than
comparing LB across configs (which legitimately drifts via alternate optima).

**Key diagnostic — pivots-per-resolve.** The report's interpretive lever: a
warm-started resolve should take tens of pivots; consistently _hundreds_ means
the warm start is failing (scaling drift, tolerance-induced cycling, or an
**alien** basis load forcing a fresh INVERT). cobre tracks `simplex_iterations`
and a non-alien basis-rejection counter internally but **does not yet surface
them in `training/metadata.json`** — see §7.

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

## 4. Experiment plan (staged OFAT — implemented in `scripts/solver-tuning/`)

**Full run per cell** (each cell = one complete `cobre run`), **staged per
backend** with a manual gate, sized for a tight budget (~a few dozen full-node
runs). The grid is `scripts/solver-tuning/grid.py`.

- **Stage 1 — set the floor, OFAT the speed levers.** Reference cell = the
  correctness floor (HiGHS `presolve=off`; CLP defaults). One confirmatory cell
  measures `presolve=on`. Each remaining cell changes one pure-speed lever:
  - HiGHS: `edge_weight` ∈ {-1, 2} (baseline 1=Devex), `scale=2`, `price=3`.
  - CLP: `scaling=1`, `pricing_mode` ∈ {0 uninitialized, 1 full DSE}, `factor_freq=100`.
    → ~6 cells (HiGHS) / ~5 (CLP).
- **Manual gate.** `aggregate.py` ranks cells, applies the cut-validity gate, and
  suggests the fastest valid env → you choose `winner.json`.
- **Stage 2 — accelerator matrix** with the winner solver env fixed: warm-start
  {full, core, off} × cut-selection {none, level1, dominated} = 9 cells.
  `ws-full.sel-none` is the reference; `ws-off.sel-{level1,dominated}` is the
  SPTcpp-style cell (see `accelerator-effectiveness-research.md`).

Repeats: 1 for screening, 3 for the 2–3 finalists (full runs are deterministic in
_result_; reps only quantify 96-thread timing noise — take the **min**).

## 5. Metrics, correctness gate, noise control

**Metrics** (`<out>/training/metadata.json`, parsed by `run_cell.py`):

- `solve_stats.backward_solve_seconds` — primary (dominant, variance-heavy phase)
- `solve_stats.forward_solve_seconds`, `duration_seconds` — secondary
- `solve_stats.{retried, failed}` — a config that raises retries is suspect
- `bounds.{final_lower_bound, final_upper_bound}` — for the cut-validity gate

**Correctness gate** (per cell, in `aggregate.py`):

- `INVALID` if `LB > UB` — a cut sliced off the optimum (perturbation / loose
  dual tolerance). The primary, config-independent gate (§1a).
- `FAIL` if a run errored or had failed solves.
- `PASS` otherwise. Cross-config LB _drift_ is expected (alternate optima) and is
  reported as gap%, not failed.

**Noise control**: `--threads 96` pinned, `--exclusive` node, `--quiet`; reps and
take the **min**; same case copy + seed per cut-selection method.

**Significance**: <±2% noise; ±2–5% borderline (re-run); >±5% real.

## 6. Harness

Implemented in `scripts/solver-tuning/` (runbook in its `README.md`); no prebuild
stage — each cell is a full run:

```
prep_cases.sh <case> cases/<backend>           # one full case copy per cut-selection method
grid.py --backend <b> --stage 1 > m.s1.jsonl   # Stage-1 manifest
sbatch --array=0-N sweep.sbatch                # 1 node/cell, 96 threads, no MPI
aggregate.py --runs runs --backend <b> --stage 1   # ranked table + cut-validity gate + suggested winner
grid.py --backend <b> --stage 2 --winner winner.json > m.s2.jsonl
sbatch --array=0-N sweep.sbatch ; aggregate.py … --stage 2
```

Cells are resumable (skip if `result.json` exists); each writes
`tune_params.json` (full env echo, git commit, host, timestamps) + `result.json`
(parsed metrics). The `[cobre tune]` stderr line records the active config.

## 7. Open items before execution

- ✅ HiGHS `simplex_price_strategy` enum **confirmed** from vendored source
  (0 Col / 1 Row / 2 RowSwitch / 3 RowSwitchColSwitch).
- **Surface the two key diagnostics in `training/metadata.json`** — the
  highest-value instrumentation: `simplex_iterations` (→ pivots-per-resolve) and
  the non-alien basis-rejection count (→ forced INVERTs). Both are tracked
  internally (`SolverStatistics`) but not emitted; surfacing them needs a field
  on `MetadataTrainingSolveStats` + the training summary + Python parity. Without
  them the sweep measures wall-clock but cannot _diagnose_ a failing warm start.
- Confirm the HiGHS `simplex_scale_strategy` equilibration index; and CLP
  `domination_epsilon` (placeholder in `patch_config.py`).
- Build the benchmark case to `solver-tuning-benchmark-case.md`.
- Set the cut-validity / drift tolerance from the baseline's observed gap.
