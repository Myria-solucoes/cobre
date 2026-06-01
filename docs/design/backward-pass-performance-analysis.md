# Backward-pass performance analysis — production case investigation

**Status**: Investigation report. No code changes recommended in this
document; the hypotheses below are candidates for follow-up experiments.
Source line references in this report are a snapshot from the original
investigation and may have drifted; treat them as approximate anchors, not
exact locations.

## 1. Context

In a 192-worker production run (2 MPI ranks × 96 rayon threads, NEWAVE-class
case with hundreds of hydros, K_noise = 20 backward openings, 64 stages, cuts
with ~2000 coefficients), the forward-pass / backward-pass wall-clock ratio
reaches **~47× at iteration 1** and grows toward the algorithmic floor as
iterations progress. We expected the ratio to be roughly the K_noise factor
(~20×). The ~2.4× gap above the algorithmic floor is the subject of this
investigation, alongside the absolute growth of backward wall time across
iterations.

The investigation has three goals:

1. Map every timing sample in the codebase so we know what each Parquet
   field actually measures.
2. Attribute backward-pass wall clock to its components (LP solve,
   synchronization, load imbalance, FFI setup, etc.).
3. Compare cobre's approach to SPTcpp (a C++ SDDP solver using CLP) to
   identify architectural differences that could explain the gap.

## 2. Timing-sample taxonomy

The `training/timing/iterations.parquet` file mixes rank-level and per-worker
rows. The fields are sampled at the following sites.

### Per-worker rows (rank=N, worker_id=K)

| Field              | Sampled at                             | Measures                                                                                                                                                                                                                                   |
| ------------------ | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `forward_wall_ms`  | `forward_pass_state.rs:664` → `:795`   | Total wall time of `run_forward_worker` for the worker's full scenario partition, across all stages.                                                                                                                                       |
| `backward_wall_ms` | `backward_pass_state.rs:974` → `:1001` | Sum over backward stages of the per-stage parallel-region wall time for this worker.                                                                                                                                                       |
| `bwd_setup_ms`     | `backward_pass_state.rs:561-564`       | Sum over stages of `SolverStatsDelta(load_model + set_bounds + basis_set)`. Dominated by `load_model` FFI because `load_backward_lp` reloads the model **twice per stage per worker** (once at `:926` and once per trial point at `:979`). |
| `fwd_setup_ms`     | `forward_pass_state.rs:551`            | Same FFI sum but for the forward pass. Small because forward doesn't reload the LP per scenario.                                                                                                                                           |

### Rank-level rows (rank=N, worker_id=NULL)

| Field                        | Sampled at                                                  | Measures                                                                                                                                                           |
| ---------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `cut_sync_ms`                | `backward_pass_state.rs:850-858`                            | Sum over stages of `cut_sync_bufs.pack_local_cuts + sync_packed_cuts` — MPI cut delivery between ranks.                                                            |
| `state_exchange_ms`          | `backward_pass_state.rs:742-748`                            | Sum over stages of `comm.allgatherv` for trial-point state vectors. Fires only when `local_work > 0`.                                                              |
| `cut_batch_build_ms`         | `backward_pass_state.rs:774-788`                            | Sum over stages of `build_delta_cut_row_batch_into` — sparse cut-row delta batch construction.                                                                     |
| `mpi_allreduce_ms`           | `training_output.rs:249`                                    | **Mislabelled** — actually `partial.forward_sync_ms`, the forward-pass `allgatherv` for scenario costs (not a backward MPI cost).                                  |
| `bwd_load_imbalance_ms`      | `backward_pass_state.rs:576` (`collect_stage_timing_stats`) | Per-stage `(max_worker_solve_time − mean_worker_solve_time)`, summed over stages. "Slack time" workers spend waiting for the slowest peer at each stage's barrier. |
| `bwd_scheduling_overhead_ms` | `backward_pass_state.rs:582`                                | Per-stage `(parallel_wall_ms − max_worker_solve_time)`, summed. Rayon dispatch + sequential pre/post per stage.                                                    |
| `lower_bound_ms`             | `training_session/mod.rs` LB phase                          | Wall time of LB evaluation.                                                                                                                                        |
| `overhead_ms`                | `training_output.rs:230`                                    | Residual: `iteration_time − (forward + backward + cut_selection + cut_selection_allgather + forward_sync + lb_eval)`.                                              |

### Forward `lp_solves` writer caveat (resolved)

Until commit `92946003` on this branch (`fix(stats): aggregate forward stage
stats across MPI ranks before logging`), the forward-pass solver-stats writer
emitted rank-0's local stage-stats only — `lp_solves` was rank-0-only (96
LPs/stage in the user's 192-worker setup), while the time fields happened to
look right relative to per-worker wall clock because they're rank-local sums
matching the rank-local worker count. This produced per-LP ratios biased by
the `num_ranks` factor for any analysis assuming `lp_solves` was the global
count.

The fix uses the existing `pack_delta_scalars` infrastructure to allreduce
forward `SolverStatsDelta` across ranks before logging. Backward stats use a
separate per-`(rank, worker_id)` allgatherv path (`backward_pass_state.rs`)
and were never affected. `retry_level_histogram` and `basis_reconstructions`
remain rank-local on forward rows because `pack_delta_scalars` excludes them
(see `crates/cobre-cli/src/commands/run.rs:1225`).

## 3. Wall-clock attribution from production data

Production run, iteration 1 backward (user wall ~89 s, mean per-worker wall =
88,992 ms):

| Component                                                                      |   Time (ms) |                             Share of wall |
| ------------------------------------------------------------------------------ | ----------: | ----------------------------------------: |
| Pure LP solving (cumulative across 192 workers)                                |  16,672,702 |                          ~95% of cum wall |
| `bwd_setup_ms` (load_model FFI dominated, cumulative per rank)                 |      49,123 | ~256 ms/worker (~0.3% of per-worker wall) |
| Other per-worker overhead (basis save, cut row construction, Rust bookkeeping) | ~7 s/worker |                    ~7% of max-worker wall |
| `cut_sync_ms` (rank-serial MPI cut delivery)                                   |       1,499 |                          **1.7%** of wall |
| `state_exchange_ms` (rank-serial MPI allgatherv)                               |         144 |                          **0.2%** of wall |
| `cut_batch_build_ms` (rank-serial CPU)                                         |         134 |                              0.2% of wall |
| `bwd_load_imbalance_ms` (workers waiting for stragglers)                       |      18,389 |                         **20.7%** of wall |
| `bwd_scheduling_overhead_ms` (rayon dispatch)                                  |       2,560 |                              2.9% of wall |

**Conclusion 1**: MPI synchronization is **not** the bottleneck. The combined
MPI cost (`cut_sync_ms + state_exchange_ms`) is 1.7-2.1% of backward wall
across iterations 1-2.

**Conclusion 2**: LP solving genuinely dominates backward wall — ~95% of
cumulative-across-workers wall is real simplex pivoting. The fix space is
mostly inside the LP solver and the model the solver receives.

**Conclusion 3**: Load imbalance is the only sync-style cost worth attacking
in the short term — 20% of wall at iter 1, 26% at iter 2, growing with cut
accumulation as some scenarios become disproportionately harder.

### Iter 1 → iter 2 growth

| Component                              |    iter 1 |     iter 2 |         Δ |
| -------------------------------------- | --------: | ---------: | --------: |
| Backward wall (max worker)             | 93,905 ms | 152,684 ms |      +63% |
| Cumulative LP solve                    | 16.67M ms |  25.97M ms |      +56% |
| Cumulative `bwd_setup_ms` (load_model) |      49 k |      117 k | **+138%** |
| Load imbalance                         | 18,389 ms |  39,770 ms | **+116%** |
| Cut sync                               |  1,499 ms |   3,189 ms |     +113% |

Three components are growing ~2× per iteration: `load_model` FFI, load
imbalance, and cut sync. The first two will dominate by iter 5-10 unless
addressed.

## 4. SPTcpp comparison

SPTcpp (`~/git/SPTcpp`) is a C++ SDDP implementation using COIN-OR/CLP via
`OsiClpSolverInterface`. We inspected the PDDE method (`C_MetodoSolucao_PDDE.cpp`),
the solver wrapper (`C_Solver.h` `SolverCLP` class at line 1404-3586), and the
LP-model layer (`C_ModeloOtimizacao.cpp`).

### 4.1 Parallelism model

| Aspect                            | cobre                                                                         | SPTcpp                                                                          |
| --------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Threading                         | MPI ranks + rayon threads                                                     | **MPI processes only** (no `omp parallel`, no `std::thread`)                    |
| User's 192-worker equivalent      | 2 ranks × 96 threads                                                          | Would be 192 MPI processes                                                      |
| Per-thread state                  | Each thread owns a `SolverWorkspace`, shared via `BasisStoreSliceMut`         | One solver per process, fully isolated memory                                   |
| Scenario distribution within rank | Static partition (`forward_pass_state.rs:367` / `backward_pass_state.rs:969`) | Static partition across MPI processes                                           |
| MPI cut/state communication       | `allreduce`, `allgatherv` (collective)                                        | **Point-to-point** `MPI_Send`/`MPI_Recv` (`C_MetodoSolucao_PDDE.cpp:1914,1934`) |

`grep -rn "omp parallel\|pragma omp\|std::thread\|std::async" /SPTcpp/src/`
returns empty.

### 4.2 CLP solver invocation — minimalist

The entire CLP setup per solve, from `C_Solver.h:1732-1748`:

```cpp
clp->setLogLevel(verbose);
clp->scaling(escalonamento);              // = 1 (equilibrium) — default
clp->setPerturbation(50);                  // moderate anti-cycling
clp->setMaximumSeconds(tempoLimite);
clp->setMaximumWallSeconds(tempoLimite);

if (tipoMetodoSolver == TipoMetodoSolver_dual_simplex) {
    clp->initialDualSolve();               // fresh solve (presolve included)
}
```

Configuration comparison:

| Knob                         | cobre (current branch tip)                               | SPTcpp                                |
| ---------------------------- | -------------------------------------------------------- | ------------------------------------- |
| Primal feasibility tolerance | `1e-4`                                                   | CLP default (typically `1e-7`)        |
| Dual feasibility tolerance   | `1e-6`                                                   | CLP default (typically `1e-7`)        |
| Scaling                      | `simplex_scale_strategy = 4` (max-value)                 | `escalonamento = 1` (equilibrium)     |
| Presolve                     | `on`                                                     | `on` (default in `initialDualSolve`)  |
| Cost perturbation            | HiGHS default (currently 0 in cobre's `default_options`) | **`setPerturbation(50)`** every solve |
| Iteration limit              | profile-controlled cap                                   | CLP default (effectively unlimited)   |
| Method                       | dual simplex                                             | dual simplex (`initialDualSolve`)     |

### 4.3 No warm-start, no basis reuse

```sh
grep -n "setBasis\|getBasis\|warmStart\|crash" C_Solver.h
```

returns **empty**. SPTcpp does not save or restore a basis between LP solves.
Each `initialDualSolve()` is a fresh solve including presolve, crash basis,
factorize, and dual simplex.

Cobre invests heavily in warm-start infrastructure:

- `BasisStore` keyed by `(scenario, stage)`
- `reconstruct_basis` handling cut-row mapping via slot identity
- Per-`(scenario, stage)` basis broadcast across MPI ranks
  (`broadcast_basis_cache`)
- HiGHS internal basis retention for the implicit ω>0 chain
- 5% explicit + 95% implicit warm-start in backward
  (cf. `backward.rs:589-593`)

SPTcpp has none of these — relying entirely on CLP's `initialDualSolve()`
plus equilibrium scaling plus moderate cost perturbation to keep each fresh
solve fast.

### 4.4 State-fixing strategy

In `C_ModeloOtimizacao.cpp:622-623`, SPTcpp fixes the backward state by
tightening column bounds:

```cpp
solver->setLimInferior(idVariavelDecisao, valor);
solver->setLimSuperior(idVariavelDecisao, valor);
```

Cobre fixes state via **`n_state` extra equality rows** added to every
backward LP, populated via the `PatchBuffer` Category 1 (storage-fixing) and
Category 2 (lag-fixing) and Category 6 (anticipated-state-fixing) patches.
The cut subgradient is extracted from the duals on these state-fixing rows.

For the user's case `n_state` is in the 200-800 range. Every backward LP
carries 200-800 additional rows that SPTcpp's equivalent LP does not.

The math is equivalent: a column with `lb == ub == v` produces the same
shadow price (reduced cost) at the optimum as a row equality `x == v` does
(dual price). Either is recoverable as a cut subgradient.

### 4.5 Retry ladder — 5 levels vs cobre's 12

`C_Solver.h:3271-3372` (`otimizarComTratamentoInviabilidade`):

| Level        | Strategy                                                                                  |
| ------------ | ----------------------------------------------------------------------------------------- |
| L5 (initial) | cold restart with `resetar()`                                                             |
| L4           | swap simplex method (dual ↔ primal)                                                       |
| L3           | flip scaling (on ↔ off)                                                                   |
| L2           | change tolerance — **bidirectional**: `1e-9` if currently in `[1e-10, 1e-8]`, else `1e-6` |
| L1           | switch to barrier (IPM)                                                                   |

Note the bidirectional tolerance toggle at L2 — SPTcpp acknowledges that
sometimes _tightening_ tolerance can break a degenerate-near-tolerance
cycle, where cobre's ladder is monotonically looser-on-retry.

### 4.6 Cut activation/deactivation via RHS toggling

`C_ModeloOtimizacao.cpp:115-117` (`anularCortesExternos`):

```cpp
solver->setRHSRestricao(ineCB, -infinito);
```

Cuts stay in the row matrix once added. To "deactivate" a cut (e.g. external
cuts for a stage that's not currently being solved), SPTcpp sets the RHS to
`-INF`, making the constraint trivially satisfied. Cheaper than row removal
because it preserves the LU factorization.

Cobre's cut activity tracking is more sophisticated (`metadata_sync_window`,
`active_window`, `active_count`) but has more bookkeeping per cut.

## 5. Hypotheses for cobre's residual backward wall-clock cost

Ranked by expected impact.

### Hypothesis A — State-fixing via column bounds, not row equalities

**Architectural divergence**: cobre adds `n_state` equality rows to every
backward LP for state fixing (`PatchBuffer` Categories 1+2+6). SPTcpp uses
column bounds.

**Expected impact**: 15-40% reduction in backward per-LP solve time. The
ratio scales as `1 − num_template_rows / (num_template_rows + n_state)`.
For typical cobre LPs with 1000-2000 base rows + 200-800 state-fix rows,
this is the largest single architectural lever.

**Mechanism**:

- Fewer rows → smaller LU factor → fewer FLOPs per pivot
- Fewer dual values to compute and read at solve end
- Smaller basis to capture/restore for warm-start
- Smaller `PatchBuffer` (Categories 1, 2, 6 collapse to column-bound updates)

**Implementation surface**: medium-large

- `indexer.rs` — drop the `state_fixing` row block from the layout
- `lp_builder/patch.rs` — replace row-bound patches with column-bound patches
  for Categories 1, 2, 6
- `backward.rs` (`extract_duals_from_view` at line 399) — switch
  state-coefficient extraction from `view.dual[..n_state]` to
  `view.reduced_costs[state_column_indices]`
- Sign convention audit: row dual vs reduced cost have the same magnitude for
  fixed variables but the sign convention in cut intercept computation needs
  verifying
- Documentation in `lp_builder/mod.rs` LP layout description

**Reversibility**: medium. The state-fix row machinery can be kept behind a
feature flag during transition.

### Hypothesis B — Warm-start machinery may be net-negative under loose-tolerance config

SPTcpp's success with no warm-start (only `initialDualSolve` from scratch)
suggests the warm-start benefit measured at tight tolerance (1e-9) may be
much smaller at the current loose tolerance (1e-4 primal, 1e-6 dual).

**Mechanism**: at 1e-4 primal, the simplex converges in fewer "polishing"
pivots near optimum, so the marginal benefit of starting near optimum is
small. Meanwhile cobre still pays:

- `get_basis` FFI per ω=0 save (`backward.rs:516, 532`)
- `set_basis_non_alien` FFI per ω=0 warm-start (`highs.rs:1402`)
- `reconstruct_basis` Rust work per ω=0 (`stage_solve.rs:197-205`)
- `BasisStore` MPI broadcast per iteration (`broadcast_basis_cache`)
- Per-worker `BasisStoreSliceMut` borrow tracking

**Expected impact**: cleanup of ~500 ms/worker `bwd_setup_ms` plus reduced
memory traffic. Less per-LP work, but each LP solve might be marginally
slower without warm-start — net depends on the magnitude.

**Test**: temporarily pass `solve(None)` everywhere (force cold-start path).
Measure iter 1 backward wall clock and compare to current warm-start path.

**Implementation surface**: small (single-flag flip in `stage_solve.rs` to
force cold path). Reversibility: trivial.

### Hypothesis C — 192 MPI processes might outperform 2 ranks × 96 threads

SPTcpp's MPI-only model avoids:

- Rayon scheduling overhead
- Thread-pool synchronization per parallel region
- Shared cache pressure across threads on the same NUMA node
- Borrow-checker overhead on `BasisStoreSliceMut`, `WorkspacePool`, etc.

**Trade-off**: 192 MPI processes increase MPI communication volume (more
ranks → more peers in collectives). With our current ~2% MPI cost share,
even a 5× increase in MPI cost is only ~10% of wall clock — likely smaller
than the threading overhead savings.

**Test**: rerun production with `--ranks 192 --threads 1` (assuming the CLI
supports this) and compare backward wall clock to `--ranks 2 --threads 96`.

**Implementation surface**: zero code change for the test itself.
Reversibility: trivial. If wins materialize, longer-term we may want to
reduce or eliminate the threading layer.

### Hypothesis D — Reconsider `load_backward_lp` reload per trial point

`backward_pass_state.rs:976-979`:

```rust
for m in start_m..end_m {
    // Reload LP per trial point to reset HiGHS's internal simplex
    // basis, factorization, and RNG position.
    load_backward_lp(ws, succ);
    ...
}
```

This call is responsible for the ~500 ms/worker `bwd_setup_ms`. With one
trial point per worker per stage in the user's case, this fires
`63 stages × 2 calls/stage = 126` `load_model` FFI invocations per worker
per iteration.

The justification ("reset HiGHS's internal simplex basis, factorization, and
RNG position") is historical. In the current configuration:

- The explicit `set_basis_non_alien` call in `solve(Some(&basis))` overwrites
  the basis at ω=0 anyway
- The factorization gets rebuilt on the first pivot when the basis is offered
- RNG matters only if `random_seed` is being used

A "soft reset" — `clear_solver()` instead of `load_backward_lp()` — may
preserve the loaded model and option settings (including equilibration
scaling) while resetting simplex internal state.

**Expected impact**: cumulative `bwd_setup_ms` drops from ~50-120 k ms/iter
to negligible. **Per-iteration saving: 5-10% of backward wall**, growing with
iteration count.

**Risk**: HiGHS's `clear_solver` behavior with custom scaling needs
verification — the equilibration cache might not survive. Worth A/B testing
on the deterministic case.

**Implementation surface**: small — single function swap. Reversibility:
trivial.

### Hypothesis E — Borrow `setPerturbation(50)` (anti-cycling) from SPTcpp

SPTcpp sets cost perturbation to 50 on every solve. HiGHS has
`dual_simplex_cost_perturbation_multiplier`, currently 0 (off) in cobre's
`default_options()`. The retry-level L0 (`apply_retry_level_options:836`)
sets it to 1.0 — but only as a retry escalation, not as a default.

If degenerate cycling is a contributor to the per-LP cost variance that
drives load imbalance, enabling moderate perturbation by default could
break cycles before they become expensive.

**Expected impact**: modest reduction in iteration count variance across
workers → reduced load imbalance share. Probably 1-3% wall clock.

**Implementation surface**: trivial (one-line change in `default_options`).
Reversibility: trivial.

### Hypothesis F — Longest-processing-time worker scheduling

Backward currently uses `partition(local_work, n_workers, w)` —
static partition by scenario index (`backward_pass_state.rs:969`). Some
scenarios are intrinsically harder; with random assignment, the
fastest worker finishes 12 seconds before the slowest at iter 1.

**Mechanism**: track per-scenario backward solve time on iter 1, sort
scenarios by descending time, then on iter 2+ assign in LPT
(longest-processing-time-first) order. The bookkeeping is one
`Vec<(scenario_idx, last_solve_time_ms)>` per rank, sorted before each
iteration.

**Expected impact**: load imbalance drops from 20% to ~5-8% of wall =
**12-15% backward wall reduction** on later iterations.

**Implementation surface**: small. Reversibility: trivial.

### Hypothesis G — Cut aging / activity-based pruning

Cut bundle grows ~K_noise cuts per backward iteration. By iter 50 the master
LP has ~1,000 cuts × ~2,000 coefficients = 2M nonzeros — every solve scans
all of them.

The infrastructure is partly in place — `metadata_sync_window_contribution`
already tracks per-cut activity bits, and `active_window` accumulates them.
Need a "prune below threshold" sweep after each backward iteration.

**Expected impact**: from iter 10 onward, master LP nonzeros stay bounded
→ per-LP solve time stabilizes instead of growing. Wall-clock impact scales
with iteration count.

**Risk**: aggressive aging can hurt cut quality. Conservative threshold
("inactive for ≥ 10 iterations") + safety net ("never drop the most-recent
K cuts") mitigate this.

**Implementation surface**: medium.

## 6. Recommended sequence

| Priority | Hypothesis                                         | Effort       | Expected wall reduction       |
| -------- | -------------------------------------------------- | ------------ | ----------------------------- |
| 1        | F — LPT scheduling                                 | Small        | 12-15% backward (later iters) |
| 2        | D — Drop per-trial-point `load_backward_lp` reload | Small        | 5-10% backward, growing       |
| 3        | E — Default cost perturbation = 1.0-5.0            | Trivial      | 1-3% backward                 |
| 4        | B — Disable warm-start (test)                      | Trivial      | TBD                           |
| 5        | C — 192-rank-no-threads (test)                     | Trivial      | TBD                           |
| 6        | A — Column-bound state-fixing                      | Medium-large | 15-40% backward per-LP        |
| 7        | G — Cut aging                                      | Medium       | Scales with iter count        |

The top-3 are quick wins that compound. Hypothesis A is the largest single
lever but the largest implementation surface and warrants its own focused
investigation cycle.

## 7. References

### Cobre files cited in this report

- `crates/cobre-sddp/src/backward_pass_state.rs` — backward orchestration
- `crates/cobre-sddp/src/forward_pass_state.rs` — forward orchestration
- `crates/cobre-sddp/src/backward.rs` — per-trial-point backward solve
- `crates/cobre-sddp/src/lp_builder/patch.rs` — state-fixing row patches
- `crates/cobre-sddp/src/indexer.rs` — LP row/column layout including state-fix rows
- `crates/cobre-sddp/src/solver_stats.rs` — `SolverStatsDelta`, `pack_delta_scalars`
- `crates/cobre-sddp/src/training_session/mod.rs` — iteration loop, MPI sync
- `crates/cobre-sddp/src/training_output.rs` — parquet field mapping (note `mpi_allreduce_ms` mislabelling)
- `crates/cobre-sddp/src/stage_solve.rs` — warm-start vs cold-start path
- `crates/cobre-solver/src/highs.rs` — HiGHS FFI wrapper, default options, retry ladder
- `crates/cobre-cli/src/commands/run.rs` — `aggregate_solver_stats`, `delta_to_stats_row`

### SPTcpp files cited

- `~/git/SPTcpp/src/C_MetodoSolucao_PDDE.cpp` — PDDE main loop
- `~/git/SPTcpp/src/C_Solver.h` — `SolverCLP` class (line 1404-3586)
- `~/git/SPTcpp/src/C_ModeloOtimizacao.cpp` — LP model layer, state fixing
