# `crates/cobre-sddp/src/lp/` — Bloat & Necessity Analysis

**Scope:** the LP-construction/access submodule of `cobre-sddp` (`crates/cobre-sddp/src/lp/`).
**Question:** do we actually need all this complexity, or is there removable bloat and over-abstraction?
**Method:** six parallel specialist analyses + first-hand verification of every actionable claim.
**Date:** 2026-06-18.
**Status:** analysis only — no code was changed.

---

## 1. Headline verdict

The submodule is **20,393 LOC = ~8,200 production + ~12,200 test (60% tests)**. The perception that it is "very large and complex" is correct in _magnitude_, but the cause is **mostly irreducible domain complexity, not pervasive bloat**.

> **~85–90% of the production code is essential.** The genuinely removable surface — with **zero risk to any correctness, performance, or determinism contract** — is roughly **250–400 production/doc LOC + 500–650 test LOC**, i.e. **~4–6% of the submodule**.

There is exactly **one genuinely vestigial abstraction** (`Sentinels`), a handful of **dead / over-exposed symbols**, **one mechanical structural duplication** (a layout cache), and a modest layer of **redundant tests and duplicated documentation**. Everything else is warranted by the shape of the problem: a stage LP with ~20 contiguous column regions and ~15 row regions, a CSC assembly path, a hot per-solve patch path, FPHA/evaporation/anticipated-thermal/pumping physics, and bit-for-bit determinism.

---

## 2. Methodology

Six independent analyses were run in parallel, each read-only, each scoped to one concern, then cross-checked against each other and against first-hand `grep`/read verification:

| Lens                               | Focus                                                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| External-usage cartography         | What the rest of the codebase actually consumes from `lp/` — the minimal necessary public surface.                       |
| Indexer-cluster review             | `indexer/{mod,layout,constructors,state_mapping,anticipated,sparse_state}.rs` — vestiges, over-abstraction, duplication. |
| Builder-cluster review             | `builder/{mod,matrix,layout,template,patch,scaling}.rs` — CSC assembly, hot path, layout duplication, docs.              |
| Generic-constraint lowering review | `generic_constraints.rs` — over-generality vs. validated input, test ratio.                                              |
| Test-suite review                  | All `#[cfg(test)]` modules — duplication, over-specification, load-bearing vs. removable.                                |
| Branch-vs-`develop` comparison     | What the recent refactor changed; whether new abstractions pull their weight.                                            |

Where the analyses disagreed (see §9), the disagreement is reported rather than resolved silently.

---

## 3. Quantitative profile

Approximate production/test split per file (test counted from the first `#[cfg(test)]`; `matrix.rs` has several inline test modules so its split is the loosest):

| File                       |      Total |       Prod |        Test |
| -------------------------- | ---------: | ---------: | ----------: |
| `builder/matrix.rs`        |      4,958 |     ~2,094 |      ~2,864 |
| `generic_constraints.rs`   |      2,807 |       ~822 |      ~1,985 |
| `builder/layout.rs`        |      2,405 |     ~1,079 |      ~1,326 |
| `builder/template.rs`      |      2,297 |       ~835 |      ~1,462 |
| `indexer/constructors.rs`  |      2,270 |       ~706 |      ~1,564 |
| `builder/patch.rs`         |      1,292 |       ~617 |        ~675 |
| `indexer/anticipated.rs`   |      1,269 |       ~133 |      ~1,136 |
| `indexer/layout.rs`        |        866 |       ~799 |         ~67 |
| `indexer/state_mapping.rs` |        759 |       ~181 |        ~578 |
| `indexer/sparse_state.rs`  |        453 |       ~127 |        ~326 |
| `builder/scaling.rs`       |        415 |       ~215 |        ~200 |
| `builder/mod.rs`           |        302 |        302 |           0 |
| `indexer/mod.rs`           |        149 |        149 |           0 |
| `indexer/test_fixtures.rs` |        124 |        ~97 |         ~27 |
| `lp/mod.rs`                |         27 |         27 |           0 |
| **Total**                  | **20,393** | **~8,183** | **~12,210** |

Notable: `indexer/anticipated.rs` carries an **8.5:1 test:production ratio** (133 prod / 1,136 test) — the single largest imbalance, and the largest test-cleanup target.

---

## 4. What we ACTUALLY need (the essential core — do not touch)

These are the load-bearing pieces. Each was verified to encode a correctness, performance, or determinism contract that a "simplification" would silently break.

| Area                                                                   | Why it is irreducible                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`builder/patch.rs` — `PatchBuffer` + all `fill_*` methods**          | The hot path: `fill_forward_patches` / `fill_load_patches` / `fill_z_inflow_patches` / `fill_col_state_patches` run millions of times per iteration. Pre-allocated buffers are _intentional_ (no hot-path allocation); every field is live.                                                                                                                                               |
| **`builder/matrix.rs` — CSC fill logic**                               | FPHA generation constraint carries `−γᵥ/2` on **both** storage columns (average-storage contract); water-balance sign conventions (turbine/spillage/diversion `+τ`, cascade upstream `−τ`, withdrawal `∓ζ`); derated bound sources. Each `fill_*` stanza name _is_ its documentation.                                                                                                     |
| **`builder/scaling.rs`**                                               | Geometric-mean LP conditioning (`D_r·A·D_c`). Compute/apply × col/row split is the natural shape; col and row paths legitimately differ.                                                                                                                                                                                                                                                  |
| **`indexer/state_mapping.rs` — resolvers**                             | `state_to_lp_incoming_column` is the sole sanctioned state-pinning / dual-extraction entry (the column-bound pinning contract). `finalize_state_column_map` precomputes by _calling_ the resolver (never re-implements arithmetic) with a fallback — correctly layered, not redundant.                                                                                                    |
| **`indexer/layout.rs` — `StageIndexer`**                               | 60-field canonical layout map, **381 external references**. The backbone of forward/backward/simulation column and row resolution.                                                                                                                                                                                                                                                        |
| **`builder/template.rs` — `StageTemplates` / `build_stage_templates`** | The per-stage structural LP + the metadata conduit (`base_rows`, NCS/pumping offsets, discount factors) that feeds simulation and setup. Discount-factor fields are private behind a getter/setter to avoid leaking placeholder `1.0` values.                                                                                                                                             |
| **The good consolidations**                                            | `for_each_fpha_plane` (single owner of the FPHA row-cursor prefix sum), `BlockSlackFamily` + `fill_block_family` (the correct table-driven operational-slack loop), the column accessors (`turbine_col`, `spillage_col`, …), `block_col_range` (exhaustive `ElementKind` routing), `AnticipatedLayout` (cohesive grouping), and the `is_stage_level` collapse (a real LP-size reduction). |
| **Satellite types**                                                    | `EvaporationIndices` / `FphaRowRange` are small named-field structs that _prevent_ a stride-aliasing bug — keep them even though they have no external references. `FphaColumnLayout` / `EquipmentCounts` are used externally.                                                                                                                                                            |
| **Invariant test sweeps**                                              | The `anticipated_invariants` sweep (non-overlap, contiguity, state-formula, theta placement across ~1,900 configurations) and the declaration-order-invariance template probe catch what end-to-end deterministic cases cannot isolate.                                                                                                                                                   |

**Public API is minimally over-exported.** Every heavily-used export has real callers: `StageIndexer` (381), `build_stage_templates` (163), `FphaColumnLayout` (121), `PatchBuffer` (96), `EquipmentCounts` (52), `StageTemplates` (17), `EvapConfig` (12), `GenericConstraintRowEntry` (3). Only five public symbols are exported but used solely inside `lp/` (see §6).

---

## 5. The cross-cutting structural finding: three representations of one layout

The LP column/row geometry is expressed in **three parallel places**, which is the root of the "feels over-abstracted" impression:

1. **`StageIndexer`** (`indexer/layout.rs`) — geometry as `Range<usize>` fields (`turbine`, `spillage`, `thermal`, …). The canonical, globally-wired map.
2. **`StageLayout`** (`builder/layout.rs`) — **embeds a whole `StageIndexer`** (its `indexer` field) and then **re-flattens ~38 of its ranges** into scalar `col_*_start` / `row_*_start` fields (e.g. `col_turbine_start = idx.turbine.start`). The matrix-fill helpers read the flat scalars; they reach through `layout.indexer.*` essentially never.
3. **Module-doc ASCII tables** — the same magic-number geometry (`N*(2+L)+A*K_max`, …) is drawn in **both** `builder/mod.rs` and `indexer/mod.rs` headers.

Key clarifications established during analysis:

- This is **representation duplication, not independent computation.** `StageLayout::new` builds the indexer once, then copies cursors out. The scalars cannot _drift_ from the indexer (they are read-through assignments), but they double the field surface and the maintenance reading-burden.
- A **full merge of the two structs is blocked** and should not be attempted: they have different lifetimes (one `StageIndexer` is wired globally; a `StageLayout` is per-stage and reads each stage's own block count, NCS active set, generic constraints, and pumping) and a deliberate ownership split (the per-stage live offsets for generic/pumping/NCS belong to `StageLayout`, not the indexer).
- The doc duplication is **two-way, not three-way** — `indexer/layout.rs` holds only the pinning contract, not a third copy.

The actionable cleanups that follow from this are C1 (replace the read-through scalars with inline accessors) and D1 (de-duplicate the ASCII tables) below.

---

## 6. Dead / internal-only public symbols

Exported `pub` but used only inside `lp/` (or not at all) — candidates to demote to `pub(crate)` or delete:

| Symbol                               | Reality                                                                                                                                               |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ar_dynamics_row_offset` (`builder`) | Zero external references; one internal caller; body is `base_row + h`. Over-exposed and over-documented.                                              |
| `EvaporationIndices` (type)          | No external code references it (only rustdoc mentions). **Keep the type** — it guards stride aliasing — but it need not be `pub`.                     |
| `FphaRowRange` (type)                | Same as above.                                                                                                                                        |
| `Sentinels` (type)                   | Never read externally. Vestigial (see §7).                                                                                                            |
| `generic_constraints` (module)       | `pub(crate)`; confirmed zero raw-path consumers outside `lp/`. Its only external product is `GenericConstraintRowEntry`, defined in `builder/mod.rs`. |

---

## 7. Deep-dive: `Sentinels` — the one genuine vestige (highest-value cleanup)

`indexer/layout.rs::Sentinels` is a sub-struct of **seven permanently-zero fields**:

- `storage_fixing`, `lag_fixing`, `anticipated_state_fixing` — all permanently `0..0`. Relics of the **old row-based state-fixing design**, which was replaced by column-bound pinning (`state_to_lp_incoming_column`). State is no longer pinned with equality rows, so these row ranges carry nothing.
- `generic_constraint_rows`, `generic_constraint_slack`, `n_generic_constraints_active`, `pumping_flow` — permanently `0..0`/`0`. Their **own doc comments name `StageLayout` as the live owner**; they are dead twins of fields that exist and work elsewhere.

Evidence that it is pure ceremony:

- **Zero production reads** of `sentinels.*` outside `lp/indexer/` (the one external hit is a doc comment).
- The only in-cluster non-test uses feed the three `*_fixing` fields through `push_nonempty`, which skips empty ranges — provable no-ops.
- It is kept alive by **~10 tests that assert the fields equal `0..0`** (a test that a constant is constant) and by a `FORBIDDEN`-tagged doc contract whose entire content is "this struct does nothing; do not wire it to anything."
- A parallel vestige exists on the builder side: `StageLayout::row_anticipated_state_fixing_start`, marked `#[allow(dead_code)]`, assigned `0`, read only by its own self-test.

**Recommendation:** delete the struct (or, as a zero-risk first step, just the four generic/pumping dead twins). Full removal requires **one reword in `.claude/rules/sddp.md`**, which currently cites the `0..0` sentinel field docs — the _contract_ (pin via column bounds) is preserved; only the dead physical fields disappear.

---

## 8. The bloat surface (ranked, with confidence and risk)

### Tier A — Quick wins (high confidence, ~zero risk, no contract impact)

| #   | Finding                                                 | Evidence                                                                                                                    | Action                                                           | ~LOC                |
| --- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------- |
| A1  | **`from_stage_template` is dead**                       | `pub fn`, but every reference repo-wide is in-cluster (definition + doctest + 4 self-tests + 2 doc mentions). Zero callers. | Delete method + 4 tests                                          | ~10 prod + ~75 test |
| A2  | **`Sentinels` 4 generic/pumping fields are dead twins** | Permanently `0..0`/`0`; live versions owned by `StageLayout`.                                                               | Delete the 4 fields                                              | ~30                 |
| A3  | **`ar_dynamics_row_offset` over-exposed**               | `pub` + re-exported for `base_row + h`; 0 external refs, 1 internal caller.                                                 | Inline (keep the pitfall comment); drop the pub export + 2 tests | ~35                 |
| A4  | **`row_anticipated_state_fixing_start` (builder twin)** | `#[allow(dead_code)]`; read only by its own test.                                                                           | Delete field + test                                              | ~40                 |
| A5  | **Tautological assertions**                             | `assert_eq!(idx.turbine.start, idx.turbine.start)` in `with_equipment_column_index_formulas` — cannot fail.                 | Fix or delete                                                    | ~5                  |

### Tier B — Needs an SDDP-owner decision

| #   | Finding                                                                                                                                                                                                             | Trade-off                                                                                                                                                                                                                                                                                                                                                                    | ~LOC                        |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| B1  | **`Sentinels` struct, full removal** (§7)                                                                                                                                                                           | Highest-value cleanup. Requires one reword in `.claude/rules/sddp.md`. Contract preserved; only dead fields removed.                                                                                                                                                                                                                                                         | ~70 prod + ~40 test         |
| B2  | **`is_anticipated_fishing_active` / `anticipated_fishing_active_at_stage`** — the predicate **always returns `true`**; its three guard call sites (`matrix.rs` ×2, `simulation/extraction.rs`) are provable no-ops. | **Option A:** inline `true`, delete both fns + vacuous tests. **Option B (recommended):** keep the predicate as a documented seam mirroring the _real_ gate `is_anticipated_decision_active`, delete only the vacuous "returns-true-everywhere"/"lockstep" tests. Removing the seam is a modelling-intent decision. Also clean up stale `:NNN` line-refs in nearby comments. | ~30 prod + large test chunk |

### Tier C — Structural (mechanical, modest payoff, higher blast radius)

| #   | Finding                                                                                                                 | Reality                                                                                                                                            | ~LOC / sites                                                                                                                                                                                                                            |
| --- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C1  | **`StageLayout` read-through scalar fields** (§5)                                                                       | Embeds a `StageIndexer`, re-flattens ~38 ranges into `col_*_start`/`row_*_start`. Not independent computation; full merge blocked.                 | Replace the ~38 read-through scalars with `#[inline]` accessors delegating to `self.indexer.<range>.start`. **~70–80 LOC, ~120 call sites** in matrix/template. Verify hash-neutrality. _Tidy, low-risk-but-mechanical, modest payoff._ |
| C2  | **`fill_operational_violation_rows`** — 4 near-identical `for blk` loops differing only in `(row_start, lower, upper)`. | Startup path, no contract touched.                                                                                                                 | Fold into one `[(start,lo,hi);4]` descriptor loop. ~25                                                                                                                                                                                  |
| C3  | **`<S: BuildHasher>` generic** threaded through 14 signatures in `generic_constraints.rs`.                              | Verified: the real `ctx.*_pos` are plain `HashMap<EntityId, usize>`. Generic = pure signature noise; determinism is safe (lookups, not iteration). | Pin to the concrete type; drop `<S>`. ~15 + signature noise                                                                                                                                                                             |

### Tier D — Documentation

| #   | Finding                                                                                                                                                                                                                       | Action                                                                                                                                                | ~LOC     |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| D1  | **Two-way layout ASCII duplication** (`builder/mod.rs` ↔ `indexer/mod.rs`) — same magic-number geometry in both headers.                                                                                                      | `builder/mod.rs` keeps only its unique patch-sequence docs; link to `crate::indexer` for geometry. (The feared three-way duplication does not exist.) | ~110 doc |
| D2  | **Doc-drift in `cobre-core::constraints::generic_constraint::VariableRef`** — per-variant docs say `block_id: None = sum over all blocks`, but the resolver does `unwrap_or(block_idx)` (single collapsed row, no summation). | Fix the doc text to match behavior. Doc-only.                                                                                                         | small    |

### Tier E — Tests (~500–650 LOC, high confidence)

| Location                   | What is redundant                                                                                                       | Replaced by                                                                                                   | ~LOC |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---- |
| `indexer/anticipated.rs`   | ~10 fishing-gate tests asserting an always-`true` predicate; two are exact duplicates.                                  | The `anticipated_invariants` sweep already covers placement/always-active parametrically.                     | ~200 |
| `indexer/constructors.rs`  | Six single-field `N=3/L=2` tests; sentinel `0..0` asserted in 4 places; `anticipated_state_fixing_mirrors_state` sweep. | `row_column_symmetry_3_2` + `state_fixing_rows_collapsed_to_empty_in_all_constructors` cover the same ground. | ~210 |
| `indexer/state_mapping.rs` | Three standalone branch tests duplicating the `R4.a/R4.b/R4.c` dispatch coverage.                                       | The `R4.*` tests.                                                                                             | ~100 |
| `builder/patch.rs`         | Capacity tests rehearsing `N*(1+L)+A*K`; one exact duplicate.                                                           | One parameterized capacity test (zero / unit / production scale).                                             | ~60  |
| `indexer/test_fixtures.rs` | `shared_eq_with_anticipated_matches_legacy_fixture_shape` — tests the fixture builder itself.                           | Downstream consumers fail immediately if the fixture shape changes.                                           | ~27  |

---

## 9. Cross-analysis disagreement (surfaced, not resolved silently)

The generic-constraint review and the test-suite review **disagree** on `generic_constraints.rs` tests:

- The generic-constraint review flags **~400–700 additional removable test LOC** — the bulk are single-arm column-arithmetic checks differing only by entity position / block index, all exercising the same shared resolver helper; collapse them into one table-driven test (one row per match arm preserves coverage).
- The test-suite review says **keep all 44** — each pins per-arm column arithmetic below the level any end-to-end case isolates; a wrong arm produces silently wrong LP coefficients.

**Reconciliation:** the _coverage_ is load-bearing; the _form_ is reducible. A table-driven rewrite that keeps one row per match arm satisfies both. Treat this as **optional / lower priority**: consolidate only the most blatant near-duplicates, keep one representative `block_id = Some` and one position-≠-0 case, and leave every multi-column / dispatch / classifier test untouched. The conservative path saves less but risks nothing.

---

## 10. Branch-vs-`develop` comparison — did the refactor meet its goal?

The recent work changed `lp/` by **+5,947 / −1,200**. Bucketed:

- **~18–20% new features** — pumping stations, realized-inflow exposure, and reversible-plant support. Legitimate growth.
- **~30–40% supporting abstractions** — column accessors, `block_col_range`, `AnticipatedLayout`, the named build output, the anticipated-decision gate predicate, the evaporation stride constants. **Every one was judged to pull its weight** (genuine dedup or aliasing-hazard guards). The FPHA plane-walker unification actively _removed_ ~300 LOC of duplicated loop logic.
- **~40–50% documentation, refactor churn, and test expansion.**

**Net assessment:** the refactor was a **net-positive** effort — it consolidated real duplication (the FPHA walker is a clear win) and added clean feature support, with the new abstractions well-grounded rather than speculative. But it **fell short of fully "removing technical debt"**: it left vestiges untouched (`from_stage_template`, `ar_dynamics_row_offset`, the always-true predicate, the two-way doc duplication, the `StageLayout` scalar cache) and _added_ ceremony by formalizing dead fields into `Sentinels` (including new dead pumping/generic twins). The net debt needle barely moved; the module grew mostly for legitimate reasons.

The single premature abstraction introduced on the branch is the `Sentinels` grouping. No other speculative generality was found.

---

## 11. Recommended sequence

1. **Tier A + D** — quick wins + doc de-duplication. ~290 LOC, near-zero risk, no decision required. Do first.
2. **Tier B1 (`Sentinels`)** — the highest-value cleanup; needs sign-off on the one-line `sddp.md` reword.
3. **Tier E** — high-confidence test de-duplication, ~500–650 LOC.
4. **Tier B2 / C** — optional; B2 and C1 need a modelling/scope decision.
5. **§9 (generic_constraints test collapse)** — optional, lowest priority, form-only.

**Aggregate realistic reduction with no contract risk:** ~250–400 production/doc LOC + ~500–650 test LOC in the high-confidence tiers, scaling toward ~400 prod/doc + ~1,200 test if the optional tiers and the disputed test collapse are taken — roughly **4–8% of the submodule**.

---

## 12. Guardrails (must not be violated by any cleanup)

- **Bit-for-bit determinism** regardless of entity declaration order — canonical iteration order must be preserved.
- **No allocation on the hot path** — `PatchBuffer` pre-allocation is intentional.
- **No `Box<dyn Trait>`** — enum dispatch only.
- **Numerical contracts are sacred** — Benders subgradient sign (`rc_scaled / col_scale`, divided not multiplied), FPHA `−γᵥ/2` on both storage columns, column-bound state pinning via `state_to_lp_incoming_column`, append-only slot-identity cut pool.
- **Verify by hash-neutrality**, not baseline match — local parity baselines do not reproduce; confirm a change is neutral by comparing the actual hash before/after the change on the same machine.
- **Contract / Rationale / Intent comments are load-bearing** — only narrator/redundant comments are removable.
