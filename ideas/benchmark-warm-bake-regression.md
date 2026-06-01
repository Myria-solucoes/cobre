# Benchmark: Warm-Start Bake Performance Regression

## Background

### The regression

Commit `3286c1b` switched template baking from `build_cut_row_batch_into`
(active cuts only) to `build_warm_start_cut_batch_into` (all populated cuts,
with inactive ones encoded at sentinel `[-INF, +INF]` bounds). This caused a
3-4x slowdown by iteration 10 on the 192-forward-pass production case.

Timing comparison (iteration 10):

| Metric        | Before regression    | After regression             |
| ------------- | -------------------- | ---------------------------- |
| Forward pass  | 9.4s (plateau)       | 37.3s (growing)              |
| Backward pass | 179s (plateau)       | 609s (growing)               |
| Lower bound   | 4.760e12 (improving) | 4.715e12 (stalled at iter 4) |

Before the regression, forward/backward times plateau because cut selection
deactivates stale cuts, and the active-only bake excludes them from the LP.
After the regression, deactivated cuts remain as sentinel-bounded rows in the
LP, so LP size grows ~192 rows per iteration without bound.

### Root cause: HiGHS skips presolve on warm-start

We traced through the vendored HiGHS source and found the full causal chain:

1. `load_model` (via `Highs_passLp`) invalidates the basis (`basis_.useful = false`).
2. `solve(Some(basis))` calls our `cobre_highs_set_basis_non_alien` FFI, which
   calls `Highs::setBasis(basis)` with `alien = false`. This sets
   `basis_.valid = true` and `basis_.useful = true`.
3. In `Highs::optimizeModel` (Highs.cpp:1370), the decision:
   ```cpp
   if ((unconstrained_lp || has_basis || without_presolve) && solver_will_use_basis)
   ```
   When `has_basis = true`, HiGHS **skips presolve entirely** and solves the LP
   directly with the full constraint matrix.
4. HiGHS presolve (`HPresolve.cpp:246`, `isRedundant()`) **would** eliminate
   `[-INF, +INF]` rows perfectly — free rows are always identified as redundant.
   But it never gets the chance to run.
5. `changeRowBounds` and `addRows` both preserve `basis_.useful = true`, so
   subsequent solves also skip presolve.
6. Result: every LP solve carries ALL populated cuts as rows. By iteration 10
   with 192 forward passes, that's ~1920 rows vs ~600 active.

Even though sentinel-bounded rows' slacks are BASIC (contributing unit-vector
columns to the basis — minimal LU fill), the overhead is real:

- `load_model` copies a 3x larger CSC matrix
- Basis setup iterates all m rows
- LU factorization dimension is m×m
- FTRAN/BTRAN, pricing, and solution extraction all scale with m
- Memory working set is larger → worse cache behavior

### The design question for iterative cut selection

The warm-start bake was infrastructure for the iterative cut selection plan
(Estratégia SC from Diniz et al. 2020). The iterative selection loop needs to
toggle cuts between active/inactive cheaply:

- **Sentinel rows + `set_row_bounds`**: All cuts are LP rows. Toggle via bound
  changes, which preserves the LU factorization. Fast re-solves (~k pivots).
  But the initial factorization per trial point is on the FULL matrix.

- **Active-only LP + `add_rows`**: Only active cuts are LP rows. Re-include
  violated cuts via `add_rows`. But `add_rows` clears the simplex state
  (`kExtendInvertWhenAddingRows = false` in HiGHS), so each re-solve is a cold
  start with a fresh factorization.

We don't know which is faster without empirical data. The sentinel approach
has a large fixed cost (factorizing the full matrix) offset by cheap re-solves.
The add_rows approach has small factorizations but loses warm-start between
re-solves. The trade-off depends on HiGHS-specific costs that we can't predict
from theory alone.

## Benchmark plan

Run the production case (192 forward passes, 20 iterations, MPI) under four
configurations. Compare the parquet outputs.

### Run 1 — Pre-regression baseline (active-only bake, warm-start everywhere)

Revert the bake function to `build_cut_row_batch_into` (active cuts only).
This is the known-good configuration where LP size plateaus.

**Changes:**

File `crates/cobre-sddp/src/training_session/mod.rs`:

- Line 31: change import

  ```rust
  // FROM:
  forward::{build_warm_start_cut_batch_into, sync_forward},
  // TO:
  forward::{build_cut_row_batch_into, sync_forward},
  ```

- Line 965: change function call
  ```rust
  // FROM:
  build_warm_start_cut_batch_into(
  // TO:
  build_cut_row_batch_into(
  ```

No other changes. Forward and backward both use warm-start basis as before.

### Run 2 — Current code (regressed, all-populated bake)

No code changes. This is the current HEAD state after commit `3286c1b` with
`build_warm_start_cut_batch_into`. Confirms the regression quantitatively.

### Run 3 — All-populated bake, no basis in forward pass

Keep `build_warm_start_cut_batch_into` but disable basis warm-start in the
forward pass only. This tests whether HiGHS presolve fires on cold forward
solves and eliminates the sentinel rows.

**Changes:**

File `crates/cobre-sddp/src/forward.rs`:

- Line 1093: force stored_basis to None
  ```rust
  // FROM:
  stored_basis: basis_slice.get_mut(m, t).as_ref(),
  // TO:
  stored_basis: None,
  ```

The backward pass retains warm-start basis reconstruction as-is.

### Run 4 — All-populated bake, no basis anywhere

Same as Run 3 plus disable basis in the backward pass. This fully isolates the
effect of presolve on the all-populated LP, with no warm-start in either phase.

**Changes (in addition to Run 3's forward.rs change):**

File `crates/cobre-sddp/src/backward.rs`:

- Line 589: force stored_basis to None unconditionally
  ```rust
  // FROM:
  let stored_basis = if omega == 0 {
      resolve_backward_basis(basis_slice, m, s)
  } else {
      None
  };
  // TO:
  let stored_basis: Option<&_> = None;
  ```

## What to measure

All data is already in the existing parquet outputs. No new instrumentation
needed.

### Primary metrics (from `training/solver/iterations.parquet`)

For each run, aggregate per-iteration per-phase (forward vs backward):

| Column                  | What it tells us                                    |
| ----------------------- | --------------------------------------------------- |
| `simplex_iterations`    | Total pivot count — warm-start benefit vs LP size   |
| `solve_time_ms`         | Wall-clock LP solve time — the bottom-line metric   |
| `load_model_time_ms`    | Template loading cost — scales with baked row count |
| `basis_set_time_ms`     | Basis reconstruction overhead                       |
| `set_bounds_time_ms`    | Bound-patching time (state pinning + noise patches) |
| `basis_offered`         | Count of warm-start calls — confirms on/off per run |
| `basis_reconstructions` | Count of basis reconstructions applied              |

### Secondary metrics (from `training/convergence.parquet`)

| Column             | What it tells us                                     |
| ------------------ | ---------------------------------------------------- |
| `lower_bound`      | LB trajectory — must match across runs (same policy) |
| `cuts_active`      | Active cut count — should be identical across runs   |
| `time_forward_ms`  | Forward wall time — cross-check with solver parquet  |
| `time_backward_ms` | Backward wall time                                   |

### Per-stage cut metrics (from cut selection parquet, if available)

| Column           | What it tells us                                     |
| ---------------- | ---------------------------------------------------- |
| `cuts_populated` | Total ever-added cuts — grows monotonically          |
| `cuts_in_lp`     | Rows baked into template — Run1: active, Run2-4: all |
| `cuts_active`    | After selection — should be identical across runs    |

## Expected outcomes and interpretation

**Run 1 vs Run 2**: Quantifies the regression. We expect Run 2 to show:

- `solve_time_ms` growing linearly per iteration (vs plateau in Run 1)
- `load_model_time_ms` growing (larger templates)
- `simplex_iterations` similar or slightly higher (same active constraints, more overhead)
- `lower_bound` may stall earlier in Run 2 (numerical noise from larger basis)

**Run 3 (no forward basis)**: The key experiment. Two possible outcomes:

- **If forward `solve_time_ms` matches Run 1**: Presolve successfully eliminates
  sentinel rows on cold solves. The sentinel-row approach is viable for
  iterative selection (where we control when to use warm-start vs cold-start).
  The regression fix is: don't use basis warm-start in forward when sentinel
  rows are present.
- **If forward `solve_time_ms` is between Run 1 and Run 2**: Presolve helps but
  cold-start pivots add overhead. Partial win — may need to weigh
  presolve benefit vs warm-start loss.
- **If forward `solve_time_ms` matches Run 2**: Presolve isn't running or isn't
  helping. The sentinel-row approach is fundamentally incompatible with HiGHS
  for performance. Must use the add_rows approach for iterative selection.

**Run 3 backward** (unchanged from Run 2): Backward should still show the
regression since it uses warm-start basis. This confirms the cause is
warm-start → presolve bypass, not something else.

**Run 4 (no basis anywhere)**: Shows the pure presolve-on-all-populated
performance for both phases. If significantly faster than Run 2, it confirms
presolve eliminates free rows. Comparing Run 4 vs Run 1 shows the net cost of
presolve + cold-start pivots vs warm-start on a smaller LP.

## How to run

Each configuration is a 1-3 line code change. Build and run the production
case identically for each. Save the output directories separately for
comparison.

```bash
# For each run:
# 1. Apply the code changes described above
# 2. Build: cargo build --release --workspace
# 3. Run the production case (same config, same MPI layout)
# 4. Copy the output directory: cp -r output/ benchmark/run{N}/
```

Compare the parquet files using any tool (Python/pandas, R/arrow, DuckDB, etc).
The key comparison is `solve_time_ms` and `simplex_iterations` grouped by
`(iteration, phase)` across the four runs.
