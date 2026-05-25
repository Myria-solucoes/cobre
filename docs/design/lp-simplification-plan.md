# LP simplification — implementation plan

**Status**: Scoped. Ready to ticketing.

This document is the implementation plan for three interrelated LP-shape
simplifications motivated by the SPTcpp comparison in
[`backward-pass-performance-analysis.md`](backward-pass-performance-analysis.md).
The goal is to reduce per-LP solve cost on the backward pass by shrinking the
LP, eliminating maintenance overhead on cut rows, and removing complex basis
machinery whose effectiveness is uncertain.

The three changes are nominally independent but interact at the basis layer.
This plan groups them into three phases with a decision point between each.

---

## 1. Objective and constraints

### Goals

1. Reduce per-LP simplex cost (fewer rows in backward LPs).
2. Simplify the cut-row machinery (stop adding/removing rows; rely on RHS
   toggling).
3. Eliminate basis tracking for cut rows; let simplex re-discover which cuts
   are binding from scratch every solve.

### Hard constraints

- **`load_backward_lp` per trial point must stay.** A past experiment
  removed it and cobre lost bit-for-bit reproducibility. Non-negotiable.
- **MPI determinism** must hold: same input → same output regardless of
  rank count or input ordering (declaration-order invariance, per
  `CLAUDE.md`).
- **Python parity**: every output file written by the CLI must also be
  written by `cobre-python`. New columns / dropped columns require touching
  both write paths.

### Acknowledged trade-offs (user-confirmed)

- **Outputs will differ from current cobre.** This is acceptable because
  the LP changes are correctness-preserving but mechanically different
  (HiGHS picks different optimal vertices on degenerate LPs).
- **Self-consistency is required**: the new version, given the same
  `(seed, config, input)`, must reproduce its own results across runs.
  This matches cobre's current reproducibility guarantee at the new
  baseline.
- Parity hashes (`d01-d15`, `d17`, etc.) will be re-pinned post-change;
  bit-identical match with the current branch is not a release gate.

---

## 2. Decisions (locked)

| Question               | Decision                                                                                                             |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Q2 — Cut prune variant | **2a (full)**: every cut ever generated stays in LP forever. Cut selection only toggles RHS. Memory growth deferred. |
| Q3 — Feature flag      | **Hard switch.** Single code path; rollback via `git revert`.                                                        |
| Q5 — Cut aging         | Out of scope for this plan. Separate future work.                                                                    |
| Trade-off              | New version differs bit-for-bit from old; must self-reproduce given same inputs.                                     |

### Q1 microbenchmark result — locked

A unit test (`crates/cobre-solver/tests/_q1_sign_convention_probe.rs`)
verified the HiGHS sign convention for fixed-at-bound columns by solving
two equivalent LPs:

- **LP-R**: `x1 ∈ [0, 10]` with equality row `x1 == 7`
- **LP-C**: `x1` fixed via column bounds `lb = ub = 7`

Both produce identical primal solutions. Crucially:

```
LP-R: dual[fixing_row] (row 1)   = +5
LP-C: reduced_cost[x1] (col 0)   = +5
```

**Same value, same sign.** Phase 1's dual-extraction switch is a 1-for-1
swap:

- Current: `pi_j = -view.dual[state_fixing_row_j]` (with row_scale unscaling)
- Phase 1: `pi_j = -view.reduced_costs[state_column_j]` (with col_scale unscaling)

No sign flip needed.

### Q6 — anticipated-thermal Category 6 review

Analyzed `lp_builder/patch.rs:fill_anticipated_state_patches` (lines
407-434). The current Category 6 code iterates `(slot, plant)` in
slot-major plant-minor order and writes A\*K row patches. Conversion to
column bounds is mechanically clean:

- Target: `col_indices[buf_slot] = ant_state_col_start + slot * n_ant + plant`
- Value: `state[ant_state_col_start + off]` (same expression)
- Scaling: `col_scale[col]` instead of `row_scale[row]` (both currently
  empty Vec under the active solver config — the offline prescaler was
  disabled)

Padding-zero invariant is preserved by construction: padding slots have
`state_value = 0`, so `lb = ub = 0` enforces the same constraint as the
equality row `x = 0` did before. Reduced cost for padding columns is 0 by
the same algebra that made the row dual 0 (no objective coefficient, no
participation in non-zero rows of the cut subgradient path).

**No anticipated-thermal-specific blocker.**

---

## 3. The three phases

### Phase 1 — State-fixing via column bounds

**Change**: replace the `n_state` state-fixing equality rows
(`storage_fixing`, `lag_fixing`, `anticipated_state_fixing`) with tight
column bounds (`lb == ub == state_value`) on the existing state variables.

**Expected gain**: 15-40% reduction in backward per-LP solve time.

#### Implementation surface

**`crates/cobre-sddp/src/indexer.rs`**

- Remove the `storage_fixing`, `lag_fixing`, and `anticipated_state_fixing`
  row ranges from `StageIndexer` (these become empty `0..0` ranges).
- All downstream row blocks (`z_inflow_rows`, structural constraint rows,
  cut rows) shift up by `n_state` row indices.
- The base row count returned in `base_rows[stage]` drops by `n_state`.
- The cut sparse mask machinery (`set_nonzero_mask`) is unaffected — it
  operates on state-column indices not on row indices.

**`crates/cobre-sddp/src/lp_builder/layout.rs`**

- `StageLayout` row-count computation drops the state-fixing block (N + N\*L
  - A\*K rows removed).
- Existing tests that assert layout dimensions need updating.

**`crates/cobre-sddp/src/lp_builder/matrix.rs`**

- Remove emit steps for state-fixing rows:
  - Storage-fixing emit loop (Cat 1 rows) → delete.
  - Lag-fixing emit loop (Cat 2 rows) → delete.
  - Anticipated-state-fixing emit loop (Cat 6 rows) → delete.
- Z_inflow definition rows (the AR dynamics) still emit — they reference
  state variables as columns, not rows.

**`crates/cobre-sddp/src/lp_builder/patch.rs`** — the biggest change

- `PatchBuffer` grows a parallel column-bound buffer block:
  ```rust
  pub col_indices: Vec<usize>,
  pub col_lower:   Vec<f64>,
  pub col_upper:   Vec<f64>,
  ```
- `PatchBuffer::new(...)` capacity: pre-allocates `N + N*L + A*K` slots in
  the col buffers (Categories 1, 2, 6).
- Replace the three `fill_*_patches` methods that emit rows for Categories
  1, 2, 6 with new `fill_*_col_patches` methods that emit column bound
  updates.
- `forward_patch_count` shrinks to Categories 3, 4, 5 only (row patches).
- New `forward_col_patch_count` returns the count of column-bound patches.

**`crates/cobre-sddp/src/forward.rs` and
`crates/cobre-sddp/src/backward.rs`**

- After the patch-buffer fill, two FFI calls instead of one:
  ```rust
  ws.solver.set_col_bounds(
      &ws.patch_buf.col_indices[..ws.patch_buf.forward_col_patch_count()],
      &ws.patch_buf.col_lower[..ws.patch_buf.forward_col_patch_count()],
      &ws.patch_buf.col_upper[..ws.patch_buf.forward_col_patch_count()],
  );
  ws.solver.set_row_bounds(
      &ws.patch_buf.indices[..ws.patch_buf.forward_patch_count()],
      &ws.patch_buf.lower[..ws.patch_buf.forward_patch_count()],
      &ws.patch_buf.upper[..ws.patch_buf.forward_patch_count()],
  );
  ```

**`crates/cobre-sddp/src/backward.rs:extract_duals_from_view`** (line 399)

- Swap from `view.dual[..n_state]` to a gather over
  `view.reduced_costs[state_column_indices]`.
- The state-column index list is contiguous in three blocks (storage,
  lags, anticipated_state) and can be precomputed once at indexer setup
  as `state_to_lp_column_block: Vec<Range<usize>>`.
- Unscaling: `col_scale[col]` if present, else identity. Current solver
  config has empty `col_scale` so this is a no-op for now.

**`crates/cobre-sddp/src/lower_bound.rs`**

- Same dual → reduced-cost swap as backward.

**`crates/cobre-sddp/src/lp_builder/mod.rs`**

- Update the module-level LP layout documentation.
- The row layout no longer mentions state-fixing rows.

#### Tests

- New unit test in `lp_builder/patch.rs` for `fill_*_col_patches`.
- Regression: closed-form LB tests on small deterministic cases — verify
  the LP solves to the same objective.
- Cut subgradient parity: take a stage LP, solve both row-equality and
  column-bound variants, verify `-view.dual[state_row]` ≡
  `-view.reduced_costs[state_col]` within tolerance.
- Parity hash regression suite — expected to shift; new hashes pinned in
  the same PR.
- Declaration-order invariance probe — must remain bit-identical for the
  same input ordering.

#### Risks

- The `state_to_lp_column` function (used in cut row construction) maps
  state vector indices to LP column indices. It already does this
  correctly for forward LPs (where state has no fixing rows). Phase 1
  simply removes the now-redundant row indexing. No behavioural change to
  state_to_lp_column needed.
- Anticipated-thermal padding-zero invariant: verified preserved (Q6
  analysis above).
- The cut-coefficient unscaling step in `build_cut_row_batch_into` uses
  `col_scale` already. No new scaling path.

---

### Phase 2 — Cut deactivation via RHS, not row removal

**Change**: cuts are added once via `add_rows` and stay in the LP for the
entire run. To "deactivate" a cut (cut-selection drops it), set the RHS to
`-INF`. The LP template is updated in the backward pass, inside each
stage, after cut synchronization (per user direction).

**Expected gain**: smaller per-iteration delta in LP shape — `add_rows`
fires only for genuinely new cuts. Cut row positions become stable across
iterations, unlocking Phase 3.

#### Implementation surface

**`crates/cobre-sddp/src/cut.rs` (`FutureCostFunction`)**

- `CutPool` becomes append-only: every `add_cut` appends to a persistent
  list, returns a stable slot id. No "drop slot" path.
- New `set_active(slot, bool)` method writes:
  - `false`: sentinel RHS (`f64::NEG_INFINITY` or HiGHS's `kHighsInf`
    negated)
  - `true`: original intercept
- `metadata.active` becomes purely a bookkeeping flag for cut-selection
  scoring; the LP carries the cut regardless.

**`crates/cobre-sddp/src/cut_selection.rs`**

- `DeactivationSet` no longer drops cuts from the pool. It produces a list
  of `(slot, new_rhs)` updates to apply.
- The selection algorithm continues to read activity bits / window state
  as before.

**`crates/cobre-sddp/src/cut_sync.rs`**

- MPI cut sync changes:
  - New cuts: gathered + added to local pool + appended to LP via
    `add_rows`.
  - Activations / deactivations: gathered as (rank, slot, new_rhs)
    triples + applied via `set_row_bounds` on the appropriate cut row.
- Wire format change requires a wire-version bump.

**`crates/cobre-sddp/src/backward_pass_state.rs` (per user direction)**

- Inside each backward stage, after cut synchronization (`cut_sync_ms`
  block), the LP stage template is updated:
  - New cuts: appended via `add_rows` (same as current delta-cut path).
  - Deactivations / reactivations: `set_row_bounds` with the new RHS
    values.
- The pre-baked template no longer needs the "active" filter for cuts
  produced in prior iterations — they're all in the LP, with the right
  RHS.

**`crates/cobre-sddp/src/lp_builder/template.rs` and
`crates/cobre-solver/src/baking.rs`**

- Template baking includes every cut ever generated. The bake step adds
  all stored cuts at their current RHS; deactivated cuts have RHS at the
  sentinel.

**`crates/cobre-cli/src/commands/run.rs` and
`crates/cobre-python/src/run.rs`**

- Cut count metrics: distinguish "cuts in LP" (always grows) from "cuts
  active" (varies). Add an `cuts_in_lp_count` column to relevant outputs.
- Python parity: identical schema must be emitted from `cobre-python`.

#### Tests

- LB/UB convergence on bundled deterministic cases — same gap-closure
  trajectory.
- Cut-selection unit tests need updating (no more "drop" path).
- MPI cut-sync wire format test — verify new payload encodes new cuts +
  RHS toggles.
- Verify that `set_row_bounds(cut_row_idx, -INF, +INF)` makes the
  constraint trivially satisfied (no effect on LP solution).

#### Risks

- **Unbounded memory growth** (per user decision, accepted for now). At
  192 cuts/iter × 50 iter = ~10k cuts × 2000 nonzero coefficients = 20M
  nonzeros in the master LP. Should fit in memory but pricing scans this
  every solve.
- Numerical issues with `f64::NEG_INFINITY` as RHS: HiGHS expects a
  representable bound. Use `-1e30` or HiGHS's documented infinity
  sentinel.
- Inactive cuts still affect pricing (their rows get scanned). For very
  large bundles this can become measurable.

#### Mitigation

- Cut aging (Phase 5 future work) becomes more important. If we
  permanently retain every cut, we'll eventually need a prune strategy.
  Tracked separately.

---

### Phase 3 — Drop basis tracking for cut rows; initialise as non-basic

**Change**: stop storing per-cut-row `BasisStatus` in `CapturedBasis`. On
basis restore, initialise every cut row as `NONBASIC_AT_LOWER`. Let dual
simplex re-discover binding cuts on every solve.

**Expected gain**: simpler `reconstruct_basis`, smaller broadcast payload,
elimination of an entire class of cut-slot-consistency bugs.

#### Implementation surface

**`crates/cobre-sddp/src/workspace.rs`**

- `CapturedBasis.row_status`: shrinks to `template_num_rows` entries (no
  cut row statuses).
- `BASIS_BROADCAST_WIRE_VERSION` bumps.
- The serialization functions (`to_broadcast_payload` /
  `try_from_broadcast_payload`) drop the cut-row-status bytes.

**`crates/cobre-sddp/src/basis_reconstruct.rs`**

- The cut-slot-identity mapping is dropped. `reconstruct_basis` now:
  1. Restores template rows verbatim from stored `row_status`.
  2. Pads `[template_num_rows, num_rows)` with `NONBASIC_AT_LOWER` for
     every cut row in the new LP.
- The `metadata_sync_window` reads from this path can be simplified or
  removed if no longer needed elsewhere.

**`crates/cobre-sddp/src/cut.rs`**

- `metadata_sync_window` / `active_window` machinery: keep only if used
  by cut-selection scoring. Otherwise drop.

**Cut-row slot mapping in `reconstruct_basis`** — no longer needed:

- `ReconstructionSource.cut_metadata`: drops (or simplifies to bookkeeping).
- `basis_activity_window` parameter: drops.

#### Tests

- Per-LP iteration count regression — measure how many extra pivots are
  required without warm cut basis. Expected: ~5-20 extra pivots per solve.
- Convergence behaviour: LB/UB unchanged, iteration count to reach a
  given gap may shift slightly.
- `BASIS_BROADCAST_WIRE_VERSION` bump test.

#### Risks

- Extra simplex pivots per solve. Likely small but needs measurement.
  Could partially offset Phase 1's gains.
- Cut-selection algorithms that depend on `active_window` for scoring may
  need re-tuning if window machinery is dropped.
- Loss of the cut-activity diagnostic for debugging cut quality (can be
  preserved as bookkeeping if needed).

---

## 4. Suggested execution order

| Phase | Effort       | Risk                           | Expected gain                            | Depends on |
| ----- | ------------ | ------------------------------ | ---------------------------------------- | ---------- |
| 1     | Medium       | Low (Q1 verified, Q6 verified) | 15-40% per-LP                            | —          |
| 2     | Medium-large | Medium (memory growth)         | Smaller per-iter delta + unlocks Phase 3 | —          |
| 3     | Medium       | Low (mostly deletions)         | Simplification + small wall-clock        | 2          |

**Recommended order**:

1. Phase 1 first. Largest expected per-LP win. Lands as a single PR.
2. **Decision point A** (after Phase 1 production test): if Phase 1
   delivers expected gain, proceed. If less than expected, investigate
   before committing to Phase 2.
3. Phase 2 second. Unlocks Phase 3 by making cut row positions stable.
4. **Decision point B** (after Phase 2 production test): verify cut
   bundle memory growth is acceptable on the production case before
   continuing.
5. Phase 3 last. Pure simplification with small additional perf upside.

---

## 5. Next step

Phase 1 is ready to be broken into atomic implementation tickets via
`/plan`. Recommended ticket structure:

1. Add column-bound buffer block to `PatchBuffer`.
2. Implement `fill_storage_col_patches`, `fill_lag_col_patches`,
   `fill_anticipated_state_col_patches` (mirrors current row-patch
   helpers).
3. Remove `storage_fixing`, `lag_fixing`, `anticipated_state_fixing` row
   blocks from `StageIndexer`.
4. Remove state-fixing row emit from `lp_builder/matrix.rs`.
5. Update `forward.rs` and `backward.rs` to use `set_col_bounds` for state
   patches.
6. Swap `extract_duals_from_view` to use `reduced_costs` instead of
   `dual`.
7. Update `lower_bound.rs` similarly.
8. Update LP-layout documentation in `lp_builder/mod.rs`.
9. Re-pin parity hashes.

Each ticket should land with regression tests passing on the bundled
deterministic case before moving on.

---

## 6. References

- `docs/design/backward-pass-performance-analysis.md` — production data
  motivating these changes.
- `crates/cobre-solver/tests/_q1_sign_convention_probe.rs` — Q1 sign
  convention verification test.
- `~/git/SPTcpp/src/C_ModeloOtimizacao.cpp:622-623` — column-bound state
  fixing reference.
- `~/git/SPTcpp/src/C_ModeloOtimizacao.cpp:115-117` — RHS-toggle cut
  deactivation reference.
- Cobre files that will be touched (per phase): listed in §3.
