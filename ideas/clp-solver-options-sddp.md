---
title: "CLP solver options for SDDP — cost model and configuration"
date: 2026-05-28
status: assessment
tags: [cobre, clp, simplex, dse, devex, sddp, cost-model]
source: "coin-or/Clp (HEAD 2026-05-16)"
companion: "highs-solver-options-sddp.md"
---

# CLP solver options — what each one actually controls, and what it costs

## 0. Bottom line on the cost question

CLP's per-iteration cost has the **same fundamental scaling as HiGHS**: the simplex
iteration is dominated by BTRAN + PRICE + FTRAN + (DSE only) FTRAN-DSE, and the cost of
each scales with the **density of intermediate vectors** and the **nonzero count of the
columns of L/U touched**, not with the LP dimensions. CLP's PRICE step also has a
density-adaptive row-vs-column switch (`ClpPackedMatrix::transposeTimes`,
`src/ClpPackedMatrix.cpp:706`), and its triangular solves go through `CoinFactorization`
which has its own sparse RHS handling. The differences from HiGHS are in *threshold
heuristics* and *plumbing*, not in the asymptotic cost model.

Where CLP differs meaningfully from HiGHS:

- **The DSE class has 4 explicit modes** (`ClpDualRowSteepest`, `src/ClpDualRowSteepest.hpp:88`),
  not just on/off. Mode 3 is "adaptive partial scan" and is the **default**. This is
  *similar to but distinct from* HiGHS's DSE-with-switch-to-Devex.
- **Perturbation is on by default** (`perturbation_ = 100`, auto-perturb when needed),
  whereas HiGHS does not perturb by default. This matters for tight tolerances and warm-
  start determinism.
- **Pricing class is a runtime-pluggable object** (`setDualRowPivotAlgorithm`,
  `setPrimalColumnPivotAlgorithm`), not a `simplex_xxx_edge_weight_strategy` integer.
  Pricing is implemented as a polymorphic class, so you swap it by instantiating
  `ClpDualRowDantzig` / `ClpDualRowSteepest` and passing it in.
- **Two factorization engines side-by-side** (`coinFactorizationA_` = classic
  Forrest–Tomlin sparse; `coinFactorizationB_` = dense/small/OSL alternatives) plus a
  separate **Abc (vectorized) factorization** if compiled with `ABC_INHERIT`.
- **`startFinishOptions`** (already covered in the comparison doc) gives caller-driven
  control over factorization persistence across solves — a runtime knob HiGHS does not
  expose.

If you want the SDDP-tuned CLP configuration:

```cpp
ClpSimplex model;
// Pricing: keep Steepest Edge (the default), pin mode to "full" (1) for tight
// warm-start loops — partial scan can miss the best candidate after small
// bound perturbations
ClpDualRowSteepest steepest(1);          // mode 1 = full DSE; default ctor uses mode 3
model.setDualRowPivotAlgorithm(steepest);

// Factorization
model.setFactorizationFrequency(200);    // refactor every 200 updates — default is fine
                                          // but tunable

// Tolerances (defaults are the right scale)
model.setPrimalTolerance(1.0e-7);
model.setDualTolerance(1.0e-7);

// Perturbation: DISABLE for SDDP warm-start loops
model.setPerturbation(102);              // 102 = "don't try perturbing again"
                                          // Default is 100 = auto-perturb; that
                                          // can silently change your LP between solves

// Scaling: equilibrium (default is auto, which is also reasonable)
model.scaling(1);

// Persist factorization and skip init across solves
const int sfOpts = 1 + 2;                // bit 1: keep factorization at end
                                          // bit 2: reuse old factorization if same nrows
                                          // bit 4 omitted (it's "work in progress" in the header)

// In the per-opening loop:
//   change bounds (state-variable RHS for opening ω)
//   model.dual(0, sfOpts);              // or use markHotStart/solveFromHotStart
//   extract row duals
```

For the parallel backward pass (multi-instance, one per Rayon worker), the high-leverage
path is **`markHotStart` + bound changes per opening + `solveFromHotStart` + `fastDual`**.
That snapshots the factorization once per stage and reuses it across all openings without
running CHUZR's full setup loop.

Everything below explains why those are the right choices and what every other knob does.

---

## 1. The cost of one CLP dual simplex iteration

The dual iteration loop is `ClpSimplexDual::whileIterating` (`src/ClpSimplexDual.cpp:973`).
Each iteration does, in order:

| Step | What it computes | Where it's done | Cost scaling |
|------|------------------|------------------|--------------|
| **CHUZR** | pick row to leave | `dualRowPivot_->pivotRow()` | scan over support of `infeasible_` (an indexed sparse vector), weighted by edge weights |
| **BTRAN** | `row_ep = B^{-T} e_p` | `factorization_->updateColumnTranspose` | sparse triangular solve in `CoinFactorization` |
| **PRICE** | `row_ap = -row_ep^T A` | `matrix_->transposeTimes(...)` | density-adaptive: row-wise hyper-sparse or column-wise dense |
| **CHUZC + BFRT** | pick entering column, bound-flipping ratio test | `dualColumn(...)` | scan over support of `row_ap` |
| **FTRAN** | `col_aq = B^{-1} a_q` | `factorization_->updateColumn` | sparse triangular solve |
| **FTRAN-DSE** | extra solve for DSE weights | `dualRowPivot_->updateWeights(...)` | `factorization_->updateTwoColumnsFT` — fused FT update of `col_aq` and `row_ep`-derived vector |
| **Update factor** | rank-1 update of L/U | `factorization_->replaceColumn` | small η-vector |
| **Update weights** | DSE or Devex weight update | inside `updateWeights` after FT | `O(nnz(col_aq))` |

So the structure is the same as HiGHS. The cost story is the same too:

### 1.1 Sparse-vs-dense PRICE — the CLP analog of HiGHS's hyper-sparse PRICE

`ClpPackedMatrix::transposeTimes(model, scalar, rowArray, y, columnArray)`
(`src/ClpPackedMatrix.cpp:706`) is the simplex PRICE. It chooses between row-wise and
column-wise dispatch:

```cpp
double factor = (numberRows < 100) ? 0.25 : 0.35;
factor = 0.5;  // override
if (numberActiveColumns_ * sizeof(double) > 1000000) {
  if (numberRows * 10 < numberActiveColumns_)      factor *= 0.333333;
  else if (numberRows * 4 < numberActiveColumns_)  factor *= 0.5;
  else if (numberRows * 2 < numberActiveColumns_)  factor *= 0.666667;
}
if (!packed) factor *= 0.9;
if (columnCopy_) factor *= 0.7;

if (numberInRowArray > factor * numberRows || !rowCopy) {
  transposeTimesByColumn(...);  // O(nnz(A)) — iterate all nonbasic columns
} else {
  // row-wise PRICE via transposeTimesByRow
}
```

The threshold `factor` lands somewhere in `[0.10, 0.50]` depending on aspect ratio and
cache-size guesses. **This is much looser than HiGHS's 0.10** (`kHyperPriceDensity`).
CLP biases more toward column PRICE for moderately-dense `row_ep`. For SDDP stage LPs
where `row_ep` is typically very sparse (a few dozen entries out of hundreds of rows), the
row-wise path will fire in both solvers, but CLP's wider tolerance means it stays on the
row path even as densification creeps in.

The row-wise PRICE itself (`transposeTimesByRow`, `src/ClpPackedMatrix.cpp:1307`) has its
own sparse-vs-dense decision **inside the row path**:

```cpp
CoinBigIndex numberCovered = 0;
int numberColumns = matrix_->getNumCols();
bool sparse = true;
for (int i = 0; i < numberInRowArray; i++) {
  int iRow = whichRow[i];
  numberCovered += rowStart[iRow + 1] - rowStart[iRow];
  if (numberCovered > numberColumns) {
    sparse = false;
    break;
  }
}
if (sparse) {
  // gutsOfTransposeTimesByRowGE3: hyper-sparse — iterate only rows in support
  //   of rowArray, scatter into a workspace, deduplicate
} else {
  // gutsOfTransposeTimesByRowGEK: walk every column, accumulate
}
```

So **CLP does hyper-sparse PRICE** when `sum_{i ∈ supp(rowArray)} nnz(A_row[i]) ≤
numColumns`. Below that cap, cost is exactly the sum of nonzeros in the touched rows of
A — identical in spirit to HiGHS's `priceByRow`.

### 1.2 The triangular solves — what CLP does in the BTRAN/FTRAN

CLP's `factorization_->updateColumnTranspose(...)` and `updateColumn(...)` are
implemented in `CoinFactorization` (CoinUtils, not shipped with the CLP source tree, but
called as a library). The factorization is **Forrest–Tomlin** with classic product-form
η-vectors for updates. The sparse RHS handling is older and less aggressive than HiGHS's
graph-traversal `solveHyper`: `CoinFactorization` uses sparseThreshold-based decisions
(`factorization_->sparseThreshold(1)` is set in the `ClpSimplex` constructor — sparse mode
on) rather than a density threshold per solve.

The practical consequence: CLP's BTRAN/FTRAN are still sparse for sparse RHS, but for
moderate-density RHS, HiGHS will typically have a slightly faster solve path because it
maintains finer density statistics per operation (`info_.row_ep_density`,
`info_.row_ap_density`, `info_.row_DSE_density`) and uses them to pick between hyper-
sparse and dense-traversal code paths individually. CLP picks once at startup
(`sparseThreshold`) and largely sticks with it.

For SDDP — sparse stage LPs, sparse RHS — both end up on the sparse code paths; the gap
is small but measurable on per-iteration timing.

### 1.3 CHUZR — the iteration-count driver, again the edge-weight scheme

`ClpDualRowSteepest::pivotRow()` (`src/ClpDualRowSteepest.cpp:179`) scans the support of
`infeasible_` (a `CoinIndexedVector` of currently-infeasible rows). For each candidate
row `i`, it considers `infeas[i]^2 / weights_[i]` — the steepest-edge ratio. Selection
depends on `mode_`:

| Mode | Behaviour | When set |
|------|-----------|----------|
| 0 | Uninitialized weights (all 1.0) — effectively Dantzig | Set by user; rarely useful |
| 1 | **Full DSE** — weights tracked exactly, all infeasible rows scanned | Most accurate but most work |
| 2 | Partial — weights tracked, but only `max(2000, number/8)` candidates scanned per CHUZR | Faster on easy LPs |
| 3 | Adaptive — starts as 2, may switch to 1 | **Default** |

The relevant code:

```cpp
int numberWanted;
if (mode_ < 2)        numberWanted = number + 1;       // full scan
else if (mode_ == 2)  numberWanted = std::max(2000, number / 8);
else {                                                  // mode 3
  int numberElements = model_->factorization()->numberElements();
  double ratio = numberElements / numberRows;
  numberWanted = std::max(2000, number / 8);
  // ratio-dependent further adjustment
}
```

**Mode 3 (default) trades pricing quality for scan time on easy LPs.** For SDDP warm-
start loops, the LP is "easy" in the sense that the optimal is close to the warm basis,
but you still want the best candidate (otherwise you take more iterations). Pin mode 1
in the inner loop for predictable iteration counts.

### 1.4 DSE weight maintenance — the extra FTRAN per iteration

`ClpDualRowSteepest::updateWeights` (`src/ClpDualRowSteepest.cpp:375`) does:

```cpp
// permute and pack row_ep into spare
...
model_->factorization()->updateTwoColumnsFT(spare2, updatedColumn, spare, ...);
// — this is the fused FT update of BOTH col_aq and the DSE auxiliary vector
// computes the FTRAN-DSE solve and the standard FTRAN simultaneously
```

The `updateTwoColumnsFT` call is CLP's optimization: it fuses the DSE FTRAN and the
standard col_aq FTRAN into one factorization traversal, sharing the L solve. HiGHS
does these as two separate operations (`HEkk::iterate` does FTRAN, then `updateFtranDSE`
does FTRAN_DSE). On well-structured factorizations the cost is similar; the difference
is wall-time-constant.

After the fused FT, CLP applies the standard DSE update formula:
`weights_new[i] = max(weights_old[i] - 2·(col_aq[i] / alpha)·row_ep_in_basis[i],
new_pivotal_weight · col_aq[i]^2)`.

So per DSE iteration: BTRAN + PRICE + FTRAN-fused-with-FTRAN-DSE + weight update. The
incremental cost over Dantzig is one full triangular solve (the fused part).

### 1.5 Net cost-model comparison with HiGHS

| Aspect | HiGHS | CLP |
|--------|-------|-----|
| Hyper-sparse triangular solves | `solveHyper`: graph-traversal, per-call density decision (`kHyperCancel`, `kHyperFtranL/U`, etc.) | `CoinFactorization` sparse path: startup-set sparseThreshold; less per-call adaptation |
| Hyper-sparse PRICE | Per-call density decision: row-wise if `row_ep_density < 0.75`, switch to dense within if `row_ap_density > 0.10` | Per-call density decision via `factor * numberRows` heuristic; row vs col cutoff ≈ 0.10–0.50; internal sparse vs dense via `numberCovered` cap |
| DSE FTRAN | Separate `FtranDseClock` solve | Fused `updateTwoColumnsFT` |
| Per-iter density tracking | Running averages of every operation's density used for next decision | Coarser; sparseThreshold mostly static |
| Bottom line | Slightly finer per-call adaptation; more bookkeeping | Slightly fewer code paths, equivalent asymptotic cost |

For sparse SDDP LPs the two are within constant factors of each other per iteration. The
**iteration count** is set by the pricing strategy and the perturbation regime; that's
where the bigger lever is.

---

## 2. Top-level options

CLP's API is C++ objects rather than a string-keyed options table. There is no `solver
= "simplex"` because `ClpSimplex` *is* the simplex solver; for IPM you'd use
`ClpInterior`, and for crash heuristics you'd call `crash()` explicitly.

### 2.1 Algorithm selection — call the method you want

| Method | Algorithm |
|--------|-----------|
| `dual(ifValuesPass, startFinishOptions)` | Dual simplex |
| `primal(ifValuesPass, startFinishOptions)` | Primal simplex |
| `barrier(crossover, startFinishOptions)` | Interior-point (Cholesky-based) with optional crossover |
| `initialSolve()` / `initialDualSolve()` / `initialPrimalSolve()` / `initialBarrierSolve()` | One-shot solve from cold; presolve + crash + simplex |
| `markHotStart()` / `solveFromHotStart()` / `unmarkHotStart()` | Snapshot-restore for repeated bound-modified re-solves |
| `strongBranching(...)` | Internal use; thousands of bound-modified dual re-solves |
| `fastDual2(...)` | Stripped-down dual for B&B nodes |

For SDDP: **always `dual(...)` with `startFinishOptions = 1 | 2`**, or the `markHotStart`
/`solveFromHotStart` trio for the opening loop.

### 2.2 `setAlgorithm(int)` — just a field, not a strategy

`algorithm_` records which algorithm last ran (`0` initially; positive after primal,
negative after dual). It's read by some heuristics inside CLP but is not the user-facing
switch. Ignore in favour of just calling `dual()` or `primal()`.

### 2.3 `ClpSolve` — the convenience wrapper

`ClpSolve` (separate header) is a presolve-and-solve wrapper used by `initialSolve()`. It
chooses presolve, scaling, crash, and algorithm based on a `SolveType` enum
(`useDual`, `usePrimal`, `useBarrier`, `automatic`, etc.) and `PresolveType`
(`presolveOn`, `presolveOff`, etc.). For SDDP you generally bypass `ClpSolve` and call
`dual()` directly on a configured model.

---

## 3. Pricing — pluggable classes, not a strategy integer

CLP installs pricing as a polymorphic object. Default at construction
(`src/ClpSimplex.cpp:156`):

```cpp
dualRowPivot_   = new ClpDualRowSteepest();          // mode 3 (adaptive partial)
primalColumnPivot_ = new ClpPrimalColumnSteepest();
```

You swap pricing by passing a different object:

```cpp
ClpDualRowDantzig dantzig;
model.setDualRowPivotAlgorithm(dantzig);

ClpDualRowSteepest steepest(1);   // mode 1 = full DSE
model.setDualRowPivotAlgorithm(steepest);
```

### 3.1 `ClpDualRowDantzig` — pure infeasibility-only pricing

Picks the row with maximum primal infeasibility, no edge weights. **Cheapest per
iteration, highest iteration count.** Counterpart to HiGHS's `Dantzig` strategy.

### 3.2 `ClpDualRowSteepest` — DSE with four modes

Mode is set at construction (`ClpDualRowSteepest(int mode = 3)`). See §1.3 above. Notes:

- The class also has a **`saveWeights`/`passInSavedWeights`** API
  (`src/ClpDualRowSteepest.hpp:43, 53`). This lets you persist the DSE weight vector
  across solves (separately from the basis). For SDDP, after the first solve at a stage,
  you can capture `savedWeights()` and pass it back at the start of the next sweep — the
  DSE weights from the previous sweep are usable warm-start data because the basis is
  similar.
- There is no separate "Devex" class in CLP analogous to HiGHS's Devex. The closest is
  using `ClpDualRowSteepest(mode = 0)` (uninitialized weights) which behaves like Dantzig
  initially. CLP relies on DSE's partial-scan modes (2 and 3) to give the cost reduction
  that Devex provides in HiGHS, rather than a separate cheaper-approximation scheme.
- The `Persistence` enum (`keep` vs `normal`) controls whether DSE arrays are torn down
  at end of solve. For warm-start loops, set persistence to `keep` so the weights are
  available for the next `dual()` call.

### 3.3 `ClpPrimalColumnSteepest` — primal pricing

Same idea for the primal simplex (pick column with steepest-edge reduced cost). Has
its own mode set. Used only if you run primal simplex, which for SDDP you don't.

### 3.4 `ClpPrimalColumnDantzig` — primal Dantzig

Reduced-cost-only pricing for primal. Cheap per iteration, lots of iterations.

### 3.5 Pricing-class summary (with HiGHS equivalents)

| CLP class | HiGHS analog | Per-iter cost | Iteration count |
|-----------|--------------|---------------|------------------|
| `ClpDualRowDantzig` | `simplex_dual_edge_weight_strategy = 0` (Dantzig) | cheapest | highest |
| `ClpDualRowSteepest(1)` (full) | `simplex_dual_edge_weight_strategy = 2` (Steepest Edge, pinned) | full DSE | lowest |
| `ClpDualRowSteepest(2)` / `(3)` | (closest: HiGHS DSE→Devex auto switch) | partial scan | middle |
| (no separate class) | `simplex_dual_edge_weight_strategy = 1` (Devex) | (CLP's mode 2/3 plays this role differently) | — |

---

## 4. PRICE strategy — implicit, density-driven

CLP does not expose a `simplex_price_strategy` enum like HiGHS. Its PRICE adaptation is
built into `ClpPackedMatrix::transposeTimes` (§1.1): row-wise hyper-sparse by default,
switching to column-wise when `row_ep` density exceeds the heuristic threshold. This is
roughly equivalent to HiGHS's `RowSwitchColSwitch` default.

If you want to force a specific path, the `vectorMode_` field
(`setVectorMode(int)`, `src/ClpSimplex.hpp:1489`) and the `specialOptions_` bits expose
some control, but no clean enum. The default behaviour is correct for SDDP.

### 4.1 Row copy of A — `rowCopy_`

CLP maintains a row-wise copy `rowCopy_` of `matrix_` whenever the simplex is active
(constructed in `createRim` / `gutsOfDelete`). This is what enables row-wise PRICE.
You can disable it with the `moreSpecialOptions_` bit 1024 ("don't do row copy of
factorization") to save memory at the cost of slower PRICE. For SDDP, keep the row copy.

### 4.2 `scaledMatrix_` and `columnCopy_`

`scaledMatrix_` is a separately-scaled copy used when scaling is active. `columnCopy_`
is an optional aligned column copy for SIMD column PRICE in the Abc path. The defaults
construct both on demand; you don't normally touch them.

---

## 5. Factorization — engine, frequency, sparse/dense

### 5.1 Two factorization engines side-by-side

CLP's `ClpFactorization` wraps **two** distinct factorization classes
(`src/ClpFactorization.hpp:520`):

```cpp
CoinFactorization      *coinFactorizationA_;   // classic FT sparse, lives in CoinUtils
CoinOtherFactorization *coinFactorizationB_;   // dense / small / OSL variants
```

Plus, if compiled with `ABC_INHERIT`, a separate `AbcSimplexFactorization` for the
vectorized "new CLP" engine (`src/AbcSimplexFactorization.hpp`). The class files
`CoinAbcFactorization{1..5}.cpp` are the vectorized SIMD kernels.

`forceB_` controls forced selection of engine B:
- `1` = dense
- `2` = small
- `3` = OSL

Thresholds for *automatic* switching to engine B:
- `goDenseThreshold_` — switch to dense if `numberRows ≤ this`
- `goSmallThreshold_` — switch to small if `numberRows ≤ this`
- `goOslThreshold_` — switch to OSL if `numberRows ≤ this`

For SDDP, stage LPs have a few hundred rows — well above the dense threshold but small
enough that the FT sparse engine is fine. Default A engine is correct.

You can force the choice with `forceOtherFactorization(int which)`
(`src/ClpFactorization.hpp:419`) for benchmarking.

### 5.2 `setFactorizationFrequency(int)` — refactor cadence

Equivalent to HiGHS's `simplex_update_limit`. Refactor every N updates. Default is set
adaptively at solve start based on problem size; usually lands around 200–400 for
moderately-sized LPs. Configure via `setFactorizationFrequency(200)` or similar.

`maximumPivots()` on the factorization object
(`src/ClpFactorization.hpp:175, 181`) is the same number from a different vantage point:

```cpp
factorization_->maximumPivots(200);   // matches setFactorizationFrequency(200)
```

For SDDP per-opening loops (few updates per re-solve from a warm basis), the
factorization rarely hits this limit anyway — the limit matters most for the first cold
solve of a sweep.

### 5.3 `setSparseFactorization(bool)`

`setSparseFactorization(true)` (the default — set in constructor via
`factorization_->sparseThreshold(1)`) tells `CoinFactorization` to use sparse code paths
for triangular solves. **Always true for SDDP.** Off only if you're solving a single
small dense LP.

### 5.4 `setInitialDenseFactorization(bool)`

Forces the very first factorization to use a dense LU regardless of sparsity. Used in
edge cases where the initial basis is fully dense. Default off.

### 5.5 `setNumberRefinements(int)`

Iterative refinement after factorization to improve numerical accuracy. Default 0 (no
refinement). For ill-conditioned LPs raise to 1–2; per-solve cost is moderate. If you see
`largestPrimalError_` or `largestDualError_` growing across openings in your stats, this
is a relevant knob.

---

## 6. Scaling — `scaling(int)`

`scalingFlag_` modes (`src/ClpModel.hpp:724`, default `3`):

| Value | Meaning |
|-------|---------|
| 0 | Off |
| 1 | Equilibrium scaling |
| 2 | Geometric scaling |
| 3 | Auto (chooses 1 or 2 based on heuristics) — **default** |
| 4 | Auto-but-as-initial-solve-in-B&B |

`setAutomaticScaling(bool)` toggles whether to rescale dynamically when the basis grows
ill-conditioned during a solve. Default off.

For SDDP, equilibrium (1) or auto (3) is fine. Direct comparison to HiGHS:
`scaling(1)` ≈ `simplex_scale_strategy = 2`.

---

## 7. Perturbation — important, **different default** from HiGHS

`setPerturbation(int)` (`src/ClpSimplex.hpp:714`). Modes:

| Value | Meaning |
|-------|---------|
| 50 | Switch on perturbation immediately |
| **100** | **Auto-perturb if dual takes too long (1.0e-6 of largest nonzero) — default** |
| 101 | "We are perturbed" (status) |
| **102** | **Don't try perturbing again — disables perturbation entirely** |

CLP's dual simplex perturbs the cost vector by small amounts to avoid degenerate cycling
when the algorithm appears to be stalling. **By default (`perturbation_ = 100`), this can
fire silently mid-solve.**

Why this matters for SDDP:

- A perturbed solve returns the optimum of a *slightly modified LP*, not your original
  one. The duals you extract for cut construction will be off by `O(perturbation)`.
- The cost vector mutation can affect determinism across runs (perturbations are seeded
  but the trigger condition depends on iteration progress).
- For warm-start loops where you expect 10–100 iterations per opening, perturbation
  rarely triggers, but if it does the effect compounds across sweeps.

**For SDDP, set `setPerturbation(102)` once configuration is done.** This disables
perturbation entirely. If you see degenerate cycling (`numberTimesOptimal_` growing,
many "going round again" log messages), revisit — but for clean SDDP stage LPs with a
good warm basis, perturbation is a defensive feature you don't need and a determinism
risk you do want to avoid.

HiGHS does not have an equivalent automatic perturbation; this is a meaningful
behavioural difference between the two solvers.

---

## 8. Crash — `crash(double gap, int pivot)`

Constructs a starting basis from scratch. Three strategies (the second argument):

| `pivot` | Meaning |
|---------|---------|
| 0 | Off |
| 1 | Bixby-style crash |
| 2 | Other crash heuristic |

Default: not called. Like HiGHS, **off is correct for SDDP** because you always have a
warm basis (forward pass, previous sweep, or `BackwardBasisStore`).

---

## 9. Tolerances and numerics

| Method | Default | Meaning |
|--------|---------|---------|
| `setPrimalTolerance(double)` | `1.0e-7` | Max allowed primal infeasibility per variable |
| `setDualTolerance(double)` | `1.0e-7` | Max allowed dual infeasibility per reduced cost |
| `setZeroTolerance(double)` | `1.0e-13` | Treat values below this as zero in PRICE/CHUZC scans |
| `setLargeValue(double)` | `1.0e15` | Treat |bounds/costs| ≥ this as effectively infinite |
| `setDualBound(double)` | `1.0e10` | Phase 1 dual-feasibility working bound; smaller = tighter perturbation, looser = larger problem |
| `setInfeasibilityCost(double)` | `1.0e10` | Coefficient on artificial variables in phase 1 |
| `setAlphaAccuracy(double)` | `-1` (off) | If positive, reject pivots whose row vs column alpha differ by more than this |
| `acceptablePivot_` | `1.0e-8` | Reject pivots with absolute value below this — numerical safety |

For SDDP defaults are sane. **`dualBound`** is the only one worth watching — it bounds
how far the simplex can let dual variables drift from their natural range before
clamping. If your cuts produce dual values on the order of `1e8` and `dualBound = 1e10`
default is fine; if cut duals reach `1e9+`, raise it.

`setLargestPrimalError(double)` / `setLargestDualError(double)` set the *post-solve*
sanity-check thresholds. Defaults match `1.0e-7`.

---

## 10. `startFinishOptions` and `whatsChanged_` — warm-start machinery

Covered in the HiGHS-vs-CLP comparison doc. The decisive runtime knob:

```cpp
model.dual(0, /* startFinishOptions = */ 1 + 2);
//                                     ^   ^
//                                     |   bit 2: reuse old factorization if same nrows
//                                     bit 1: keep work areas + factorization at end
```

Bit 4 ("skip init based on whatsChanged_") is marked "work in progress" in the header
comment (`src/ClpSimplex.hpp:328`). Use it with caution and re-test if you enable it.

For pure bound changes across openings, bits 1+2 give you the full warm-start path —
no factorization rebuild, no work-area reallocation. This is the supported equivalent of
HiGHS's "persistent instance + bound change" pattern, but more explicit.

The `whatsChanged_` field (`src/ClpModel.hpp:935`) is a bitmask the caller sets to tell
CLP "only these things changed since last solve":

```cpp
model.setWhatsChanged(511);   // first 9 bits: claim full warm state valid
                                // — this is what markHotStart sets internally
```

For the inner loop, the `markHotStart` API handles all of this for you and is the
cleaner path.

---

## 11. Refactor and rebuild reasons

CLP refactorizes when any of these hit (analog of HiGHS's `RebuildReason` enum, though
CLP's are scattered through `ClpSimplexDual::statusOfProblemInDual` rather than enumerated
in one place):

- Update count ≥ `maximumPivots()` (the explicit cap)
- Numerical trouble detected (pivot growth, large residuals, accuracy check fails in
  `replaceColumn`)
- Going to phase 2 from phase 1
- `forceFactorization_` flag set externally
- Approaching optimal — refactorize for clean accuracy check
- Iterative refinement requested

`dontFactorizePivots_` (`src/ClpSimplex.cpp:127`) is a counter of pivots done without
factorizing — capped by the frequency setting.

There is no synthetic-clock heuristic in CLP. The refactor decision is **purely
count-based plus numerical-trouble-driven**. This is simpler than HiGHS's
synthetic-clock and gives more predictable behaviour, but can miss the optimal refactor
point on hard LPs where L/U fill grows quickly.

---

## 12. Special options bits (the long list)

`setSpecialOptions(int)` and `setMoreSpecialOptions(int)` are bitmasks for advanced
behaviours. A few that are SDDP-relevant:

- `moreSpecialOptions_ & 8`: no free or superBasic variables. Set if your LP has no
  free variables; small speedup.
- `moreSpecialOptions_ & 16`: check `replaceColumn` accuracy before updating — useful
  defensive flag if you see numerical issues.
- `moreSpecialOptions_ & 32`: say optimal if primal feasible (relaxed convergence; do
  not use for SDDP, you need true optimality).
- `moreSpecialOptions_ & 4194304`: tolerances have been changed by code (informational).
- `specialOptions_ & 1048576`: stop when primal feasible after N-1000000 iterations
  — phase-1-only mode, not relevant.

The bulk of these flags are CBC integration plumbing. The defaults are correct for
standalone LP solving. Don't twiddle unless profiling tells you to.

---

## 13. The Abc / "new CLP" path

If CLP is compiled with `ABC_INHERIT`, the `AbcSimplex` family becomes available as a
parallel hierarchy with vectorized factorization and matrix operations:

- `AbcSimplex`, `AbcSimplexDual`, `AbcSimplexPrimal`
- `AbcSimplexFactorization` wrapping `CoinAbcFactorization{1..5}`
- `AbcMatrix` with SIMD-friendly column layout

You switch on the Abc path with `setAbcState(int)` after compile-time enablement. The
per-iteration kernels are faster on modern x86 due to AVX2 vectorization, but the
algorithmic structure is unchanged. The Abc factorization is documented as fragile in
practice and recent commits in the CLP repo have been hardening thread-safety and ASAN
issues around it (`ClpRacingSolver`, May 2026 commits).

For SDDP, the conservative choice is the classic engine. If you build CLP and want to
benchmark Abc, enable `ABC_INHERIT` and measure on representative stage LPs — but
treat this as a research spike, not a default.

---

## 14. Putting it together for the SDDP backward pass

```cpp
// Per-worker persistent ClpSimplex for stage t:
ClpSimplex model = /* load LP, factorize once */;

// Pricing: full DSE, pinned (no adaptive mode 3)
ClpDualRowSteepest pricing(1);
pricing.setPersistence(ClpDualRowSteepest::keep);  // keep weights across solves
model.setDualRowPivotAlgorithm(pricing);

// Factorization: classic FT sparse (default), refactor every 200 updates
model.setFactorizationFrequency(200);
model.setSparseFactorization(true);   // already default

// Scaling: equilibrium
model.scaling(1);

// Perturbation: DISABLE — critical for determinism and clean duals
model.setPerturbation(102);

// Tolerances: defaults
// (primalTolerance = dualTolerance = 1e-7, zeroTolerance = 1e-13)

// First (cold-ish) solve at stage start:
model.dual(0, /* startFinishOptions = */ 1);   // keep factorization for next solve

// Snapshot for the opening loop:
void *hotStart = nullptr;
model.markHotStart(hotStart);                  // captures factorization + state

// For each opening ω:
for (omega : openings) {
    // Change state-variable bounds (the RHS of the stage LP)
    for (int j : state_vars) {
        model.setColumnBounds(j, lower_ω[j], upper_ω[j]);
    }
    model.solveFromHotStart(hotStart);         // reuses factorization, dual hot start
    // Extract row duals for cut construction:
    const double *duals = model.dualRowSolution();
    // ...
}

model.unmarkHotStart(hotStart);
```

What to measure once this is running:

1. **Iterations per opening.** Should be 10–50 for small bound perturbations. If you see
   hundreds, the warm basis is too far from optimal — either the bounds change is large
   or the basis is stale.
2. **`largestPrimalError_` and `largestDualError_` across openings.** Should stay below
   `1e-7`. Growth suggests numerical drift; raise `numberRefinements_` to 1.
3. **Refactorizations per sweep.** If many, raise `setFactorizationFrequency`. If none
   ever, the cap is irrelevant.
4. **Wall time per opening relative to HiGHS on the same stage.** This is the actual
   solver comparison.

---

## 15. Side-by-side option map: HiGHS ↔ CLP

| HiGHS option (value/meaning) | CLP equivalent |
|------------------------------|----------------|
| `solver = "simplex"` | (use `ClpSimplex`) |
| `simplex_strategy = 1` (Dual serial) | `model.dual(0, ...)` |
| `simplex_strategy = 4` (Primal) | `model.primal(0, ...)` |
| `simplex_strategy = 3` (PAMI) | (no direct equivalent; `ClpRacingSolver` is different — opportunistic parallel solves) |
| `simplex_dual_edge_weight_strategy = 2` (SE pinned) | `ClpDualRowSteepest(1)` |
| `simplex_dual_edge_weight_strategy = 1` (Devex) | (no direct class; closest is `ClpDualRowSteepest(0)` for uninitialized weights) |
| `simplex_dual_edge_weight_strategy = 0` (Dantzig) | `ClpDualRowDantzig` |
| `simplex_price_strategy = 3` (RowSwitchColSwitch) | (always-on density-adaptive in `ClpPackedMatrix::transposeTimes`) |
| `simplex_scale_strategy = 2` (Equilibration) | `model.scaling(1)` |
| `simplex_crash_strategy = 0` | (don't call `crash()`) |
| `simplex_update_limit = 5000` | `setFactorizationFrequency(N)` / `factorization_->maximumPivots(N)` |
| `simplex_iteration_limit = N` | `setIntParam(ClpMaxNumIteration, N)` |
| `presolve = "off"` | (don't call `ClpPresolve` / `initialSolve`; call `dual()` directly) |
| `parallel = "off"` | (default; CLP has no PAMI) |
| `primal_feasibility_tolerance` | `setPrimalTolerance` |
| `dual_feasibility_tolerance` | `setDualTolerance` |
| (no equivalent) | `setPerturbation(102)` — disable CLP's auto-perturbation |
| (no equivalent — `kExtendInvertWhenAddingRows` disabled) | (no equivalent either; row addition rebuilds factor) |
| `freezeBasis` (deprecated) | `markHotStart` / `solveFromHotStart` (the supported equivalent — but within-instance only) |

---

## 16. Source references

All file paths under `src/` in the CLP repository (HEAD 2026-05-16):

- **Field defaults**: `ClpSimplex.cpp:60–155` (constructor initializer list)
- **DSE pricing class**: `ClpDualRowSteepest.hpp:88` (modes), `ClpDualRowSteepest.cpp:179`
  (pivotRow), `:375` (updateWeights → `updateTwoColumnsFT`)
- **Dantzig pricing**: `ClpDualRowDantzig.cpp`
- **Iteration loop**: `ClpSimplexDual.cpp:973` (`whileIterating`), `:1285` (BTRAN +
  PRICE + CHUZC sequence)
- **PRICE**: `ClpPackedMatrix.cpp:706` (`transposeTimes`), `:1307` (`transposeTimesByRow`,
  hyper-sparse branch)
- **Factorization wrapper**: `ClpFactorization.hpp:520` (A/B engines + Abc),
  `:175–186` (`maximumPivots`)
- **Hot-start trio**: `ClpSimplex.cpp:6852` (`markHotStart`), `:6881`
  (`solveFromHotStart` → `setFactorization` + `fastDual`), `:7010` (`unmarkHotStart`)
- **`startFinishOptions` documentation**: `ClpSimplex.hpp:328`
- **`whatsChanged_`**: `ClpModel.hpp:935`
- **Scaling**: `ClpModel.cpp:4718`, modes documented at `ClpModel.hpp:724`
- **Perturbation**: `ClpSimplex.hpp:702` (modes), implementation in
  `ClpSimplexDual.cpp` around the auto-perturb trigger
- **Special options bit documentation**: `ClpSimplex.hpp:1390–1450`
- **Abc / new-CLP entry point**: `AbcSimplexFactorization.hpp`,
  `CoinAbcFactorization{1..5}.cpp`
