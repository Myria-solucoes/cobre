# LP Layout Architecture Spike — `StageIndexer` vs `StageLayout` (§5 follow-up)

**Question:** Is the "two representations of one layout" duplication (`lp-construction-bloat-analysis.md` §5) best addressed by the mechanical C1 cleanup, or does it warrant a principled `StateLayout` / `StageStructure` redesign?
**Method:** read-only investigation of the consumer split, block-count reality, and the sacred wire/contract surface; produced a design + recommendation.
**Date:** 2026-06-18.
**Status:** analysis only — no code changed. **Decisions made** (see §6) and recorded for the deferred plan epics.

---

## 0. Verdict up front

The hypothesis was **directionally correct on the math but refuted on the payoff.** The block-invariant vs block-dependent split is real and clean, but the principled `StateLayout`/`StageStructure` redesign does **not** fix a latent bug and does **not** shrink the duplication meaningfully — because the state↔LP-column resolvers straddle both worlds, and because the real fragility (a uniform-block-count assumption) lives in `simulation/extraction.rs`, which the redesign leaves untouched.

> **Recommendation: mechanical C1 (accessor delegation) + fix the latent extraction bug — NOT the big redesign.**

---

## 1. Consumer split (Q1)

The split falls almost perfectly along the block-count axis.

**Block-INVARIANT _state_ fields** (`n_state`, `theta`, `storage`, `inflow_lags`, `anticipated_state`, `storage_in`, `state_to_lp_*`, `k_max`, `max_par_order`, `n_anticipated`, `hydro_count`, `nonzero_state_indices`, `z_inflow`) — consumed by the SDDP-contract hot paths, **zero equipment-range reads**:

- `cut/row.rs` (forward cut-row build): `n_state`, `theta`, `state_to_lp_column` + the column map, `nonzero_state_indices`, `anticipated_state`, `n_anticipated`, `k_max`, `lp_column_for_state`.
- `cut/dcs.rs`: `n_state`, `theta`, `state_to_lp_column`.
- `training/backward/duals_extraction.rs`: `n_state`, `state_to_lp_incoming_column`.
- `training/backward_pass_state.rs`, `training/forward_pass_state.rs`, `training/lower_bound.rs`: `n_state`, `hydro_count`, `max_par_order`, `inflow_lags`.
- `stochastic/noise.rs`: `z_inflow`, `inflow_lags`, `n_state`, `hydro_count`, `anticipated_state`, `anticipated_decision` (the decision cols are stage-level / A-wide, not per-block — also effectively invariant).

**Block-DEPENDENT _equipment/row_ fields** (`turbine`, `spillage`, `diversion`, `thermal`, `line_*`, `deficit`, `excess`, `generation`, `*_slack`, `*_rows`, `ncs_generation`, `n_blks`) — consumed almost exclusively by **one file**:

- `simulation/extraction.rs` is essentially the entire block-dependent consumer set (it strides equipment columns by `n_blks`).
- `training/forward/stage_solve.rs` and `training/backward/lp_setup.rs` read only `ncs_generation.start` (a cursor), and pair it with a **per-stage** stride (see §2).

**Conclusion:** ~95% of the hot SDDP-contract reads touch block-invariant state fields. The SDDP contract surface is block-invariant; the block-dependent equipment geometry is consumed almost entirely by `simulation/extraction.rs`.

---

## 2. Block-count reality + the latent bug (Q2 — the decisive finding)

**Global indexer construction:** `setup/mod.rs::build_wired_indexer` builds the single global `StageIndexer` from **stage 0's** block count. Its own contract comment states the layout presumes every stage shares that count, and that `StageLayout` (which reads each stage's own `stage.blocks.len()`) is the per-stage authority — so the two structs _can_ disagree.

**Does block count vary per stage today?** **No.** Every shipped example study uses a uniform block count across stages. The multi-resolution / decomposition cases vary stage **duration/season**, with block count uniform. **Block count is an axis orthogonal to temporal resolution.** However, `cobre-core::Stage` owns its own `Vec<Block>` and `cobre-io` does **not** validate uniform block count — so divergence is structurally reachable via hand-authored input, just not exercised.

**Is there a latent bug? Yes — masked by the uniform-count invariant, and it lives in `simulation/extraction.rs`, not in the global indexer's existence:**

- `simulation/extraction.rs` reads the **global** `indexer.n_blks` (= stage 0's) and strides **every** equipment column with it: `turbine.start + h*n_blks + b`, `thermal.start + t*n_blks + b`, `line_fwd.start + l*n_blks + b`, deficit, excess, generation-below-slack, etc.
- The **rest** of the pipeline correctly uses the **per-stage** `ctx.block_counts_per_stage[t]`: `simulation/pipeline.rs` (`extract_sim_stage_result`), `training/forward/stage_solve.rs`, `training/backward/lp_setup.rs` (for NCS/load).
- With uniform counts the two coincide; with **varying** counts, simulation primal extraction would read the **wrong columns** — silently wrong outputs that still compile and run.

The training cut/bound contracts are **safe** — they touch only invariant state fields + per-stage strides.

---

## 3. `n_state` stability + sacred contract surface (Q3)

`n_state` is genuinely block-count-invariant: `n_state = hydro_count*(1 + max_par_order) + n_anticipated*k_max` (no `n_blks` term). The entire state+aux column prefix (`storage` → `inflow_lags` → `anticipated_state` → `z_inflow` (one per hydro, not per-block) → `storage_in` → `theta`) is block-invariant; equipment begins at `theta + 1`.

Contracts the redesign must NOT touch (all confirmed clean w.r.t. `n_state`):

- **`cut/wire.rs`** — wire size and the coefficient array depend only on `n_state`, no equipment ranges. SACRED (own version byte + reject test).
- **`workspace/workspace.rs` `CapturedBasis::{to,try_from}_broadcast_payload`** — per-stage LP dims (from templates) + `n_state` for state capacity; no global-equipment dependency. SACRED.
- **`cut/fcf.rs`, `cut_sync` (delegates to `cut::wire`), `resolved_parameters`** — all keyed on `n_state`, untouched by any split.
- **State-pinning resolver** `state_to_lp_incoming_column` — `storage_in.start + j` / `inflow_lags.start + (j−N)` / `anticipated_state.start + (j−lag_end)`, all in the invariant prefix. SAFE.

**The one straddle (critical design constraint):** the _outgoing_-state resolver `state_to_lp_column` returns `anticipated_state_out.start + plant` for the `slot == K_p−1` anticipated case. `anticipated_state_out` sits **after** the block-dependent `thermal` block, so its start **is** block-dependent. Therefore the outgoing-state resolver cannot live in a purely block-invariant `StateLayout` — it needs one block-dependent cursor (the storage and lag branches of the same resolver are invariant).

---

## 4. Design + why the redesign isn't worth it (Q4)

The clean `StateLayout` (block-invariant) + `StageStructure` (per-stage, all-real-ranges, dissolving the `Sentinels` vestiges) factoring is **sound conceptually**, but three facts collapse its value:

1. **It doesn't fix the latent bug.** The bug is `simulation/extraction.rs` striding with the global `indexer.n_blks` instead of `ctx.block_counts_per_stage[t]`. That is fixed by a per-stage stride swap (+ optionally a uniform-count guard), **independent** of any struct refactor.
2. **The straddle** (`anticipated_state_out` is block-dependent) means the resolvers can't cleanly move to a block-invariant `StateLayout` without splitting/duplicating a cursor — re-introducing the coupling the split claims to remove.
3. **Severe blast radius, zero payoff on the sacred surface.** The global `StageIndexer` is referenced ~hundreds of times across ~30 files; `setup/accessors.rs::stage_indexer()` returns `&StageIndexer` and is the public seam consumed by simulation, CLI, Python, integration tests, and benches (`pub use lp::indexer::StageIndexer` at the crate root). Splitting the public type is a breaking change to that seam — while `n_state`, the wire formats, and cut/dual extraction are _already_ clean and gain nothing.

### Three independently-shippable steps (risk-ranked)

- **(A) LOWEST RISK, HIGHEST VALUE — fix/guard the real bug.** Either **(i)** make `simulation/extraction.rs` stride with the per-stage block count (thread `ctx.block_counts_per_stage[t]` into the extraction helpers, matching NCS/load) — making varying block counts actually correct; **OR (ii)** if varying block counts are out of scope, add a validation/`debug_assert!` (in `cobre-io` or `build_wired_indexer`) that every stage's `blocks.len()` equals stage 0's — turning silent-wrong-output into a loud failure. ~10–30 LOC, zero contract risk.
- **(B) Mechanical C1** — replace `StageLayout`'s ~38 read-through `col_*_start`/`row_*_start` scalars with `#[inline]` accessors delegating to `self.indexer.<range>.start`. ~70–80 LOC, ~120 call sites in matrix/template. Verify hash-neutrality (actual-hash neutrality on the same machine, NOT baseline match). Pure tidiness, modest payoff.
- **(C) Sentinels removal** (bloat-doc B1/A2) — delete the dead generic/pumping twins now (zero-risk; A2 done); full `Sentinels` removal with the one-line `sddp.md` reword later. The only place the "dissolve Sentinels into live ranges" idea pays off — achievable **without** the `StateLayout` split.

### When would the redesign be worth it?

Only if Cobre commits to **first-class support for per-stage-varying block counts** (true multi-resolution where a quarterly stage genuinely has a different block count than a weekly one). At that point per-stage structural layouts must be threaded everywhere (killing the global equipment indexer), the bug fix (A-i) and the `StateLayout`/`StageStructure` split converge, and the redesign becomes the natural shape. Absent that product decision, the redesign is effort without payoff.

### Migration outline (only if the redesign is later taken — incremental, compile-checked)

1. Introduce `StateLayout { n_state, hydro_count, max_par_order, k_max, n_anticipated, storage, inflow_lags, anticipated_state, storage_in, z_inflow, theta, nonzero_state_indices, state_to_lp_column_map }` + the two resolvers (passing the one block-dependent `anticipated_state_out_start` as an arg, not a field). Make `StageIndexer` embed it; delegating accessors keep external callers unchanged. Compiles green.
2. Migrate `cut/*`, `duals_extraction`, `lower_bound`, `*_pass_state` to take `&StateLayout` — the ~95%-invariant consumers; mechanical, each file independently. `n_state`/wire untouched.
3. Build a per-stage `StageStructure` (real ranges for equipment + generic/pumping/NCS, dissolving `Sentinels`) inside `StageLayout::new`; thread it through `simulation/extraction.rs`, **replacing the global-`n_blks` strides with per-stage ones** — this is where bug (A) is actually fixed. Largest blast radius; the only step that justifies the exercise.
4. Retire the global equipment indexer; `stage_indexer()` returns `&StateLayout` for state consumers, per-stage `StageStructure` for extraction.

**Determinism / wire risk:** LOW for steps 1–2 (no `n_state`/iteration-order change). MEDIUM for step 3 (extraction column indices change behavior under varying counts — hash-verify on uniform-count cases for neutrality; most test coverage needed). No wire-format byte-layout change at any step (`n_state` formula preserved).

---

## 5. Key files

- `crates/cobre-sddp/src/lp/indexer/layout.rs` — `StageIndexer`, `Sentinels`
- `crates/cobre-sddp/src/lp/indexer/constructors.rs` — `n_state` formula + column order
- `crates/cobre-sddp/src/lp/indexer/state_mapping.rs` — the straddling resolvers (`state_to_lp_column`, `state_to_lp_incoming_column`)
- `crates/cobre-sddp/src/lp/builder/layout.rs` — `StageLayout` (the duplication; `StageLayout::new`)
- `crates/cobre-sddp/src/setup/mod.rs` — `build_wired_indexer` (global construction from stage 0)
- `crates/cobre-sddp/src/setup/accessors.rs` — `stage_indexer()` (the public seam)
- `crates/cobre-sddp/src/simulation/extraction.rs` — **the latent bug** (global-`n_blks` equipment strides)
- `crates/cobre-sddp/src/cut/wire.rs`, `crates/cobre-sddp/src/workspace/workspace.rs` — sacred wire formats keyed on `n_state` (NOT touched by the split)

---

## 6. Decisions made (recorded for the deferred plan epics)

1. **§5 layout duplication → mechanical C1 only.** No `StateLayout`/`StageStructure` redesign. Epic-04 C1 stays the accessor-delegation refactor; B1 `Sentinels` removal remains valid without any struct split.
2. **The latent extraction block-count bug → fix per-stage (step A-i).** Thread `ctx.block_counts_per_stage[t]` into `simulation/extraction.rs` so per-stage-varying block counts produce correct outputs. This is a **new correctness ticket** to add to the deferred epics (it was not in the original plan; it is arguably the highest-value outcome of the spike). Verify hash-neutrality on uniform-count cases.

### Implication for the plan

When Epic-03+ is resumed, refine the deferred epics around these decisions:

- Epic-04 C1 = mechanical accessor delegation (as planned).
- Add a new ticket (Epic-04 or a dedicated correctness epic): "fix `simulation/extraction.rs` to stride equipment columns by per-stage block count," with hash-neutrality verification and ideally a new deterministic case exercising per-stage-varying block counts.
