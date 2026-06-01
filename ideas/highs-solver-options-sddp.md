---
title: "HiGHS solver options for SDDP — cost model and configuration"
date: 2026-05-28
status: assessment
tags: [cobre, highs, simplex, dse, devex, pami, sddp, cost-model]
source: "ERGO-Code/HiGHS @ 1.14.0-dev (HEAD 2026-04-08)"
---

# HiGHS solver options — what each one actually controls, and what it costs

## 0. Bottom line on the cost question

A simplex iteration in HiGHS does **not** scale with the LP dimensions. It scales with
the **density of intermediate vectors** (`row_ep`, `row_ap`, `col_aq`, `DSE_Vector`) and
with the **nonzero count of the columns of L and U** that those vectors touch. HiGHS
chooses a "hyper-sparse" code path that traverses only the pivots actually reachable from
the RHS support — for sparse LPs the cost is closer to `O(nnz(B^{-1}·v))` than to
`O(num_row)`. The thresholds that switch between hyper-sparse and dense-style sweeps are
`kHyperCancel = 0.05` and `kHyperResult = 0.10` for the L/U solves, and `0.10`/`0.75` for
the PRICE step (`util/HFactorConst.h`, `util/HighsSparseMatrix.h:25`,
`simplex/HEkk.cpp:2833`). For SDDP stage LPs — network-structured, ~few hundred rows,
mostly sparse cuts — the iteration cost is dominated by the **pricing strategy** and the
**number of iterations**, not by the LP size. That is the lever to optimize.

If you want one default that is almost certainly the right shape for the SDDP inner loop:

```
solver                              = "simplex"
simplex_strategy                    = 1     # Dual serial (already default)
simplex_dual_edge_weight_strategy   = 2     # Steepest Edge, no Devex switch
                                            # (default = Choose → DSE+switch; pin it)
simplex_price_strategy              = 3     # RowSwitchColSwitch (default)
simplex_scale_strategy              = 2     # Equilibration (default)
simplex_crash_strategy              = 0     # Off (default — we have a warm basis)
presolve                            = "off"  # Critical for warm-start preservation
parallel                            = "off"  # Don't compete with Rayon
run_crossover                       = "on"  # irrelevant if you don't use IPM
simplex_update_limit                = 5000  # default; consider raising if refactor too often
```

The deviations from default I'd suggest auditing:
- **`presolve = "off"`**: presolve changes the LP, so the stored basis from the previous
  solve no longer matches the actual matrix HiGHS solves. With `choose`/`on`, you can lose
  warm-start every solve. Set to `off` once you have a warm basis pipeline.
- **`simplex_dual_edge_weight_strategy = 2`** (Steepest Edge, pinned): the default
  `Choose` allows HiGHS to switch from DSE to Devex if DSE weights drift; for warm-start
  SDDP loops with few iterations per solve, the switch heuristic adds overhead without
  payoff. Pin it.
- **`parallel = "off"`**: the default `choose` may activate PAMI on multi-core hosts. PAMI
  is *intra-solve* parallelism that fights your *inter-solve* Rayon parallelism for cores.

Everything below explains why.

---

## 1. The cost of one simplex iteration, decomposed

A dual simplex iteration in HiGHS does, in order
(`simplex/HEkkDual.cpp::iterate`, lines ~1200–1300):

| Step | What it computes | Where it scales | File / function |
|------|------------------|------------------|-----------------|
| **CHUZR** | pick row to leave | scan over infeasibilities, weighted by `dual_edge_weight_[i]` | `HEkkDualRHS::chooseNormal` |
| **BTRAN** | `row_ep = B^{-T} e_p` | hyper-sparse triangular solve | `HSimplexNla::btran` → `HFactor::btranU/L` |
| **PRICE** | `row_ap = row_ep^T A` (pivotal row) | depends on densities and strategy | `HEkk::tableauRowPrice` |
| **CHUZC + BFRT** | pick column to enter, bound-flipping ratio test | scan over candidate nonbasic columns in support of `row_ap` | `HEkkDualRow::chooseFinal` |
| **FTRAN** | `col_aq = B^{-1} a_q` | hyper-sparse triangular solve | `HFactor::ftranL/U` |
| **FTRAN-DSE** | extra solve to update DSE weights | only if `edge_weight_mode == DSE` | `HEkkDual::updateFtranDSE` (`:2030`) |
| **Update factor** | rank-1 update of B and L/U | small, ≪ rebuild | `HFactor::updateFT/PF/...` |
| **Update weights** | update DSE or Devex weights | `O(nnz(col_aq))` | `HEkk::updateDualSteepestEdgeWeights` (`:2089`) |

Two things determine wall time per iteration: the **density of the intermediate vectors**
and the **edge-weight scheme** (extra FTRAN for DSE).

### 1.1 Hyper-sparse vs sparse-style solves — the dominant cost driver

The BTRAN/FTRAN code chooses between two code paths at every solve
(`util/HFactor.cpp:1545`, `:1593`, `:1668`, `:1775`):

```cpp
const bool sparse_solve = rhs.count < 0
                       || current_density > kHyperCancel       // 0.05
                       || expected_density > kHyperFtranL;     // 0.15 etc.
```

- **Hyper-sparse path** (`solveHyper`, `util/HFactor.cpp:61`): depth-first traversal of L
  (or U) starting from the support of the RHS. Cost is proportional to the *pivots
  actually reached* and the nonzero entries in those pivot columns. Internal accounting
  records `synthetic_tick += count_pivot * 20 + count_entry * 10`. For SDDP stage LPs,
  where the RHS of BTRAN is a unit vector `e_p`, `row_ep` is initially extremely sparse
  and stays sparse through `B^{-T}` for typical network structures — this is the path you
  almost always want.
- **Sparse-storage but dense-traversal path**: a single sweep `for i = 0 .. num_row` over
  all pivots, multiplying by the RHS entry. Cost is `O(num_row + nnz(L))`. Used when the
  RHS or expected result is dense enough that graph traversal overhead is not worth it.

**Thresholds** (`util/HFactorConst.h`):

```
kHyperCancel  = 0.05   // current RHS density above this → dense path
kHyperResult  = 0.10   // expected result density below this → hyper-sparse pays off
kHyperFtranL  = 0.15
kHyperFtranU  = 0.10
kHyperBtranL  = 0.10
kHyperBtranU  = 0.15
```

Each operation has a separate threshold because the typical density distribution of
ftran-L vs btran-U etc. differs. The decision uses a **running average** of past
densities (`info_.row_ep_density`, `info_.row_ap_density`, etc., updated after each
operation in `updateOperationResultDensity`).

**What this means for SDDP**: your stage LPs are sparse, dense paths essentially never
fire, and per-iteration cost scales with whatever the hyper-sparse path costs on
*your* basis structure — which depends on fill-in of L/U. Forrest–Tomlin updates
gradually grow the L/U fill until the synthetic-clock or update-limit refactor restores
it. The relevant lever is refactor cadence (§4), not LP dimension.

### 1.2 PRICE — the matrix-vector multiply against A

PRICE computes `row_ap = row_ep^T · A` (the pivotal row of the simplex tableau). This is
the only step where the *constraint matrix A* enters per-iteration. Three modes
(`HEkk::tableauRowPrice`, `simplex/HEkk.cpp:2840`):

| Mode | What it does | Cost |
|------|--------------|------|
| **Column-wise PRICE** (`priceByColumn`) | for each nonbasic column j: `row_ap[j] = sum_i row_ep[i] * A[i,j]` | `O(nnz(A))` per call |
| **Row-wise PRICE** (`priceByRow`) | uses row-wise copy `ar_matrix_`; for each i in support(row_ep): scatter `A_row[i]` into row_ap | `O(sum_{i ∈ supp(row_ep)} nnz(A_row_i))` — hyper-sparse in row_ep |
| **Row-wise with switch** (`priceByRowWithSwitch`) | hyper-sparse, but if `row_ap` density exceeds `kHyperPriceDensity = 0.1` mid-computation, switch to dense | same as row-wise when sparse, falls back when not |

Decision logic in `HEkk::choosePriceTechnique` (`:2825`):

```cpp
const double density_for_column_price_switch = 0.75;
use_col_price = (price_strategy == kSimplexPriceStrategyCol)
             || (price_strategy == kSimplexPriceStrategyRowSwitchColSwitch
                 && row_ep_density > 0.75);   // averaged density
use_row_price_w_switch = price_strategy ∈ {RowSwitch, RowSwitchColSwitch};
```

The default `simplex_price_strategy = 3` (`RowSwitchColSwitch`) means: hyper-sparse row
PRICE almost always, switch to column PRICE if `row_ep_density > 0.75` (i.e. `row_ep`
has become so dense that row-wise scattering is no longer worth it). For SDDP, this is
the right default; you don't need to touch it.

PRICE is also where the **row-wise copy of A** (`ar_matrix_`) matters. HiGHS maintains
both `lp_.a_matrix_` (column-wise) and `ar_matrix_` (row-wise) for the simplex. Adding a
row triggers reconstruction of both (you'd already pay this for cut injection).

### 1.3 CHUZR and the edge-weight scheme — the iteration-count driver

`HEkkDualRHS::chooseNormal` picks the row `p` that maximizes
`(infeasibility_i)^2 / dual_edge_weight_[i]` (the steepest-edge ratio). The choice of how
`dual_edge_weight_[i]` is maintained is the single biggest knob:

| Strategy | `dual_edge_weight_[i]` is... | Update cost | Iteration count |
|----------|------------------------------|-------------|------------------|
| **Dantzig** (0) | all `1.0` | none | highest (degenerate cycling common) |
| **Devex** (1) | a *cheap approximation* of the true norm of row `i` of `B^{-1}` | `O(nnz(col_aq))` per iteration | medium |
| **Steepest Edge** = DSE (2) | the *exact* `‖B^{-T} e_i‖^2` | one extra FTRAN per iteration | lowest |

The default `simplex_dual_edge_weight_strategy = -1` (Choose) maps to **DSE with
allowed switch to Devex** (`HEkkDual::interpretDualEdgeWeightStrategy`,
`:2300`):

```cpp
if (strategy == kSimplexEdgeWeightStrategyChoose) {
  edge_weight_mode = EdgeWeightMode::kSteepestEdge;
  allow_dual_steepest_edge_to_devex_switch = true;
}
```

The switch fires (`HEkk::switchToDevex`) when the DSE weight drift exceeds
`dual_steepest_edge_weight_log_error_threshold`, indicating that the cheap updates have
diverged from the true norms and the extra FTRAN_DSE is no longer paying off.

### 1.4 The DSE extra cost concretely

`HEkkDual::updateFtranDSE` (`:2030`) performs an **additional FTRAN** per iteration to
update the dual edge weights:

```cpp
analysis->simplexTimerStart(FtranDseClock);
simplex_nla->ftranInScaledSpace(*DSE_Vector,
                                ekk_instance_.info_.row_DSE_density,
                                ...);
analysis->simplexTimerStop(FtranDseClock);
```

So per iteration: BTRAN + PRICE + FTRAN + **FTRAN_DSE** (DSE only) + weight updates. The
verified weight check in CHUZR (`HEkkDual::chooseRow`, lines 1407–1480) can also force
**a second BTRAN** if the updated weight diverges too far from the recomputed norm — the
"acceptDualSteepestEdgeWeight" gate at `kAcceptDseWeightThreshold = 0.25`.

So one DSE iteration ≈ 2.0–2.5× the linear-algebra cost of one Dantzig iteration. The
trade is: DSE typically uses **3–5× fewer iterations** on hard LPs. For warm-started LPs
where you converge in 10–50 iterations, the iteration-count win is tighter and pricing
overhead can flip the balance.

---

## 2. Top-level options

### 2.1 `solver` — pick the algorithm family

`solver`: `"choose" | "simplex" | "ipm" | "ipx" | "hipo" | "pdlp" | "qpasm" | "hipdlp"`
(default `"choose"`).

- **`simplex`** is the dual/primal simplex described above. It's the only path that supports
  warm-starting from a basis. For SDDP this is the only sensible choice.
- **`ipm`** (interior-point, the HiPO solver) and **`ipx`** (the IPX interior-point)
  cannot warm-start from a basis. Each solve starts from scratch. Per-solve they can be
  faster than cold simplex on large dense LPs, but for SDDP's warm-restart pattern they
  throw away the very thing that makes SDDP cheap. `run_crossover = "on"` (default) means
  IPM finishes with a basis from crossover, but that only helps the *next* solve if you
  switch back to simplex.
- **`pdlp`** is the matrix-free primal-dual hybrid gradient method. First-order, no
  factorization, scales to very large LPs but produces only an approximate solution and
  no basis. Wrong tool for SDDP.

Verdict for SDDP: pin `solver = "simplex"`. Don't let `choose` ever route you to IPM/PDLP
on a stage LP.

### 2.2 `presolve` — danger zone for warm-start

`presolve`: `"off" | "choose" | "on"` (default `"choose"`).

Presolve eliminates redundant rows, fixes variables, substitutes free singletons, etc.
The resulting "presolved LP" is what the simplex actually solves, then postsolve maps the
solution back. The problem for SDDP: **the basis you stored from solve k may not be a
valid basis for the presolved LP of solve k+1**, because presolve has changed which
variables exist and which rows remain.

In `Highs::run`, if `kkt_tolerance` was not changed from default, presolve will run on
`choose`. When you've added cut rows since the last solve, presolve may eliminate some of
those rows as redundant or simplify others — and now your warm basis is mismatched.

For the SDDP inner loop, where you want every solve to use the previous basis (or one
held in `BackwardBasisStore`), **set `presolve = "off"`**. The one-time presolve at the
*first* stage solve can still be useful if you do it explicitly with a fresh solve and
then use the postsolve-mapped basis as the warm basis going forward.

### 2.3 `parallel` — `"choose"` will activate PAMI, which you don't want

`parallel`: `"off" | "choose" | "on"` (default `"choose"`).

If `parallel == "on"` and `simplex_strategy == kSimplexStrategyDual` (the default) and
the number of threads available is ≥ 2, HiGHS will silently switch to PAMI
(`kSimplexStrategyDualMulti`) — `simplex/HEkk.cpp:1726`:

```cpp
if (options.parallel == kHighsOnString &&
    simplex_strategy == kSimplexStrategyDual) {
  if (max_threads >= kDualMultiMinConcurrency)
    simplex_strategy = kSimplexStrategyDualMulti;
}
```

PAMI is *intra-solve* parallelism (Huangfu–Hall): it overlaps the major iterations of the
dual simplex, allowing multiple chooseRow/BTRAN/PRICE/CHUZC sequences to run concurrently
on different cores. It's a research-grade speedup for *one large LP*. In Cobre's backward
pass the parallelism is *inter-solve* (Rayon over openings); PAMI inside each opening
would compete with Rayon's worker threads for cores. **Pin `parallel = "off"`**.

### 2.4 `run_crossover` — only matters if you use IPM

`run_crossover`: `"off" | "choose" | "on"` (default `"on"`). When `solver = "ipm"`,
controls whether to do crossover to produce a basis at the end. Irrelevant for `simplex`.

---

## 3. Simplex options

### 3.1 `simplex_strategy` — pick the simplex variant

| Value | Constant | Meaning |
|-------|----------|---------|
| 0 | `Choose` | If LP is primal infeasible (the usual case from a fresh basis), use dual serial. If primal feasible, use primal. See `HEkk.cpp:1707`. |
| 1 | `Dual` (Plain) | Serial dual simplex. **Default**, and the right choice for SDDP. |
| 2 | `DualTasks` (SIP) | Dual simplex with task-parallel BTRAN and PRICE (Single Iteration Parallel). Requires `simplex_max_concurrency ≥ kDualTasksMinConcurrency = 3`. |
| 3 | `DualMulti` (PAMI) | Parallel dual simplex Across Multiple Iterations. Concurrency ≥ 2. |
| 4 | `Primal` | Primal simplex. Use only if you start primal feasible. |

For SDDP:
- **Dual serial (1)** is the right default. After a bound change, the previous primal
  solution is typically primal-infeasible (the new bound cuts the old optimum out), so
  dual simplex is the natural warm-start.
- **Primal (4)** is wrong here — your warm state is rarely primal-feasible mid-loop.
- **SIP (2)** and **PAMI (3)** are intra-solve parallelism, conflicting with Rayon. Skip.

### 3.2 `simplex_dual_edge_weight_strategy` — the iteration-count knob

Already explained in §1.3. Concretely:

| Value | Mode | Per-iter cost | Notes |
|-------|------|---------------|-------|
| -1 | Choose | Variable | Maps to **DSE with switch to Devex**. Default. |
| 0 | Dantzig | Cheapest | Rarely competitive; weights = 1.0. |
| 1 | Devex | Cheap | `O(nnz(col_aq))` weight update per iteration. |
| 2 | Steepest Edge | Most expensive | Adds one full FTRAN_DSE per iteration. Lowest iteration count. |

**For warm-started SDDP loops, three regimes:**

- **Small bound changes, few iterations expected (10–30)**: Dantzig or Devex may actually
  win because DSE's per-iteration overhead doesn't amortize over enough iterations.
  Empirically testable.
- **Larger changes or first-iteration solves (50–500 iter)**: DSE wins.
- **First sweep of a study (cold or coarse warm start, 500+ iter)**: DSE wins clearly.

Worth a benchmark. The HiGHS default (Choose → DSE+switch) is the safe choice; if you
profile and find DSE pricing dominates the inner-loop budget, swap to Devex (1) and
measure. Pin whichever you choose — don't leave the auto-switch on for a tight loop.

### 3.3 `simplex_primal_edge_weight_strategy`

Same value range, same mechanism but for primal simplex pricing (CHUZC). Default Choose
typically picks Devex for primal (primal steepest edge is more expensive to maintain than
dual). Irrelevant for SDDP since you use dual.

### 3.4 `simplex_price_strategy`

| Value | Constant | Behaviour |
|-------|----------|-----------|
| 0 | Col | Always column-wise PRICE (`O(nnz(A))`). |
| 1 | Row | Always row-wise PRICE, no switch. |
| 2 | RowSwitch | Row-wise PRICE, mid-computation switch to dense if `row_ap` density > 0.1. |
| 3 | RowSwitchColSwitch | Like RowSwitch, *plus* switch to col-PRICE if `row_ep_density > 0.75`. **Default.** |

Default is right for SDDP. The row-wise PRICE with switch is the hyper-sparse default; the
col-switch is a safety net for the rare iterations when `row_ep` becomes very dense.

### 3.5 `simplex_scale_strategy`

| Value | Strategy |
|-------|----------|
| 0 | Off |
| 1 | Choose |
| 2 | Equilibration (default) |
| 3 | Forced equilibration |
| 4 | Max value |

Scaling rebalances the row/column norms of A before solving to improve conditioning.
Default equilibration is fine for SDDP. Off only if you can prove your LP is already
well-scaled (you can't, in general; reservoir energy units, MW, cost in $/MWh differ by
orders of magnitude).

### 3.6 `simplex_crash_strategy`

| Value | Strategy |
|-------|----------|
| 0 | Off (default) |
| 1 | LTSSF (Bixby-style crash) |
| 2 | Bixby |

Crash constructs a starting basis from scratch when none is provided. **Off is correct
for SDDP** because you always have a warm basis (either from the previous opening, the
stored `BackwardBasisStore`, or the forward pass). Crash would discard your warm
information.

### 3.7 `simplex_iteration_limit` and `simplex_update_limit`

- `simplex_iteration_limit` (default `kHighsIInf`): per-solve iteration cap. You probably
  want a finite cap (say 50_000) as a defensive measure in case a numerical issue causes a
  solve to stall — never silently spin forever inside the backward pass.
- `simplex_update_limit` (default 5000): max number of `UPDATE` operations to L/U before a
  forced refactor (`kRebuildReasonUpdateLimitReached`). For SDDP, with the synthetic-clock
  refactor (§1.1) usually firing well before 5000, this rarely matters. Raise it only if
  you see a lot of forced refactors in `solver/iterations.parquet`.

### 3.8 `simplex_dualize_strategy` and `simplex_permute_strategy`

Both default off. Dualization swaps min/max and primal/dual to potentially shrink the LP
when the dual is smaller; useful only for specific shape patterns (many fewer rows than
columns). Permutation reorders rows/cols pre-solve. Neither buys you anything for SDDP
stage LPs; ignore.

### 3.9 `simplex_unscaled_solution_strategy`

| Value | Strategy |
|-------|----------|
| 0 | None |
| 1 | Refine (default) — solve scaled, refine on unscaled |
| 2 | Direct — solve unscaled |

Default refine is correct. Direct can hurt conditioning on poorly-scaled LPs.

---

## 4. Numerical tolerances

These set the gates that determine when the simplex considers itself converged and
what counts as a valid pivot.

| Option | Default | Meaning |
|--------|---------|---------|
| `primal_feasibility_tolerance` | `1e-7` | Max allowed primal infeasibility |
| `dual_feasibility_tolerance` | `1e-7` | Max allowed dual infeasibility |
| `primal_residual_tolerance` | `1e-7` | Max residual on `Ax − b` for converged solution |
| `dual_residual_tolerance` | `1e-7` | Max residual on `A^T y + s − c` |
| `kkt_tolerance` | `1e-7` | Master tolerance: if changed, overrides all four above |
| `optimality_tolerance` | `1e-7` | Duality gap target |
| `dual_simplex_pivot_growth_tolerance` | small | Reject pivots whose absolute value scaled by row norm is too small (numerically dangerous) |
| `dual_steepest_edge_weight_error_tolerance` | `kHighsInf` (no limit) | Reject DSE weight if recomputed differs from updated by more than this |
| `dual_steepest_edge_weight_log_error_threshold` | (set in code) | Trigger DSE → Devex switch when log error exceeds this |
| `infinite_bound` | `1e20` | Treat |bound| ≥ this as infinite |
| `infinite_cost` | `1e20` | Treat |cost coef| ≥ this as infinite |
| `small_matrix_value` | `1e-9` | Treat |A entries| ≤ this as zero |
| `large_matrix_value` | `1e15` | Reject A entries ≥ this |
| `simplex_initial_condition_tolerance` | `1e14` | Reject ill-conditioned initial basis |

For SDDP, the defaults are sane. The two worth watching:

- **`primal_feasibility_tolerance` / `dual_feasibility_tolerance`**: if you have very
  small reservoir capacities (Hm³ scale) or very small cost coefficients in the cuts,
  a `1e-7` absolute tolerance may be too loose. The user-facing `kkt_tolerance` knob
  unifies these. Keep an eye on whether your stage values vs cuts are within ~7 decades.
- **`simplex_initial_condition_tolerance`**: if the basis you supply has a condition
  number above this, HiGHS will reject it and crash to a fresh basis. The default `1e14`
  is permissive; if you see condition-rejection messages, increase it (this is your
  `simplex_initial_condition_check = true` default kicking in).

---

## 5. Refactorization and rebuild reasons

Within a single solve, HiGHS refactorizes the basis whenever any of these fire
(`SimplexConst.h:107`):

```
kRebuildReasonUpdateLimitReached            // update_count ≥ simplex_update_limit
kRebuildReasonSyntheticClockSaysInvert      // synthetic_tick ≥ build_tick (heuristic)
kRebuildReasonPossiblyOptimal               // CHUZR found nothing
kRebuildReasonPossiblyPhase1Feasible        // phase 1 likely done
kRebuildReasonPossiblyPrimalUnbounded
kRebuildReasonPossiblyDualUnbounded
kRebuildReasonPossiblySingularBasis         // numerical trouble
kRebuildReasonPrimalInfeasibleInPrimalSimplex
kRebuildReasonChooseColumnFail              // numerical trouble in BFRT
kRebuildReasonForceRefactor
kRebuildReasonExcessivePrimalValue
```

The synthetic-clock rule (`HEkk::updateFactor`, `:3079–3085`) is the dominant cause of
refactor mid-solve in healthy LPs:

```cpp
bool reinvert_syntheticClock = total_synthetic_tick_ >= build_synthetic_tick_;
bool performed_min_updates   = update_count >= kSyntheticTickReinversionMinUpdateCount; // 50
if (reinvert_syntheticClock && performed_min_updates)
  *hint = kRebuildReasonSyntheticClockSaysInvert;
```

— refactor when the accumulated cost of UPDATE operations equals the cost of a fresh
INVERT, but never before 50 updates. For SDDP, this is the rate-limiter on L/U fill
growth, and it's adaptive — you don't usually need to touch it.

The setting `no_unnecessary_rebuild_refactor = true` (default) avoids refactoring on the
rebuild that happens after CHUZR signals optimality if the existing factorization is
fresh enough.

---

## 6. The IPM and PDLP options (briefly, for completeness)

You won't use these for SDDP, but for ecosystem completeness:

### 6.1 IPM (`solver = "ipm"`)
- `ipm_optimality_tolerance` (default `1e-8`): the duality gap target.
- `ipm_iteration_limit`: cap on barrier iterations.
- `run_crossover = "on"`: produce a basis via crossover after convergence.

IPM solves the LP by Newton steps on a log-barrier reformulation; cost per iteration is
~O(nnz(A) + nnz(L_AAᵀ)) for forming and factorizing the normal equations. No warm start.

### 6.2 PDLP (`solver = "pdlp"`)
- `pdlp_iteration_limit`: cap on PDHG iterations.
- `pdlp_scaling_mode`: pre-scaling.
- `pdlp_ruiz_iterations`: Ruiz scaling iterations.
- `pdlp_restart_strategy`: when to restart the gradient.
- `pdlp_step_size_strategy`: adaptive step size scheme.
- `pdlp_optimality_tolerance`: relative tolerance (looser than simplex/IPM).

PDLP is matrix-free; cost per iteration is O(nnz(A)) two matrix-vector products. Scales
to very large LPs but won't help warm-start workflows.

### 6.3 iCrash
A set of options prefixed `icrash_*` controls a heuristic crash that runs an
approximate-minimization scheme before simplex starts (default off). Forget about it for
SDDP — you have a warm basis.

---

## 7. Putting it together for the SDDP backward pass

For the inner opening loop (P1 in the previous assessment), the configuration that
matches your structure is:

```rust
// HiGHS options for Cobre's per-(worker, stage) persistent solver:
opts.solver = "simplex";
opts.simplex_strategy = 1;                          // Dual serial
opts.simplex_dual_edge_weight_strategy = 2;         // Steepest Edge, pinned
opts.simplex_price_strategy = 3;                    // default RowSwitchColSwitch
opts.simplex_scale_strategy = 2;                    // equilibration
opts.simplex_crash_strategy = 0;                    // off, we have a warm basis
opts.simplex_initial_condition_check = true;        // default, but watch the log
opts.presolve = "off";                              // CRITICAL for warm-start
opts.parallel = "off";                              // don't fight Rayon
opts.simplex_iteration_limit = 50_000;              // defensive cap
opts.output_flag = false;                           // silence per-solve logs in inner loop
opts.primal_feasibility_tolerance = 1e-7;           // default
opts.dual_feasibility_tolerance = 1e-7;             // default
```

Two empirical tests worth running once you're on persistent instances:

1. **Pin DSE vs Devex** (`simplex_dual_edge_weight_strategy = 2` vs `1`). For warm-start
   loops with few iterations per opening, Devex's lower per-iteration cost may flip the
   balance. Measure on a representative stage, ~K_noise openings.
2. **`simplex_update_limit`** tuning. If `per_opening_stats` shows many
   `kRebuildReasonUpdateLimitReached` events, raise the limit; if you see
   `kRebuildReasonSyntheticClockSaysInvert` dominating, the LU fill is the cost driver,
   not the update count.

What is *not* a useful knob to chase:

- **Scaling strategy** other than equilibration. Diminishing returns and risk of
  destabilizing the dual feasibility check.
- **`simplex_dualize_strategy`**. SDDP stage LPs have ~equal numbers of constraints and
  variables; dualization won't help.
- **PAMI / SIP**. Categorically wrong for inter-solve Rayon parallelism.
- **IPM / PDLP**. Wrong tool for warm-start.

---

## 8. Source references

All file paths under `highs/` in the HiGHS repository (1.14.0-dev, HEAD 2026-04-08):

- **Option definitions**: `lp_data/HighsOptions.h:883–950, 1489–1546`
- **Strategy enums and constants**: `simplex/SimplexConst.h:18–148`
- **Hyper-sparse thresholds**: `util/HFactorConst.h:36–62`, `util/HighsSparseMatrix.h:25`
- **Solve dispatch**: `simplex/HEkk.cpp:1050–1090, 1700–1760`
- **Dual iterate**: `simplex/HEkkDual.cpp:1200–1295` (single), `:iterateMulti` (PAMI)
- **CHUZR + DSE verify**: `simplex/HEkkDual.cpp:1407–1480`
- **CHUZC + PRICE**: `simplex/HEkkDual.cpp:1546–1660`
- **PRICE technique choice**: `simplex/HEkk.cpp:2825–2895`
- **Hyper-sparse triangular solve**: `util/HFactor.cpp:61–135` (`solveHyper`), `:1529–1648`
- **DSE FTRAN and weight update**: `simplex/HEkkDual.cpp:2020–2070`, `simplex/HEkk.cpp:2089–2230`
- **Edge-weight strategy interpretation**: `simplex/HEkkDual.cpp:2300–2325`
- **Refactor triggers**: `simplex/HEkk.cpp:3069–3100`
