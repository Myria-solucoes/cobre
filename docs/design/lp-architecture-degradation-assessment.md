# `crates/cobre-sddp/src/lp/` — Architecture Degradation & Ideal-State Assessment

**Scope:** the LP-construction/access submodule of `cobre-sddp` (`crates/cobre-sddp/src/lp/`),
plus the directly-coupled `setup/`, `simulation/extraction.rs`, and the hot-path patch callers.
**Question:** after the bloat cleanup, what _architectural and code-quality_ degradations remain,
and what is missing to reach the ideal architecture for the problem this submodule solves?
**Method:** four parallel read-only specialist deep-dives over the current (post-cleanup) code,
cross-checked against first-hand reads and the two prior design docs.
**Date:** 2026-06-19.
**Status:** analysis only — no code changed. This is a point-in-time snapshot; line anchors will drift,
symbol anchors will not.

This is the **third pass** on this submodule:

1. `lp-construction-bloat-analysis.md` — "is there removable bloat?" → ~85–90% essential; ~4–8% removed.
2. `lp-layout-architecture-spike.md` — "redesign the `StageIndexer`/`StageLayout` split?" → no; **but flagged a
   latent correctness bug** (Decision #2) as the highest-value follow-up.
3. **This doc** — "what degradations remain, and what is the gap to the ideal?"

---

## 0. Verdict up front

The big-picture decomposition is **sound and must not be restructured** — both prior passes hold. The submodule
correctly models its problem (deterministically map domain entities to a per-stage LP template + cheap
per-iteration patches), and the genuine domain complexity is irreducible.

The real degradations are **not over-abstraction**. They are three forms of _"correctness enforced by
convention, comment, or copy instead of by construction,"_ plus **one unfixed latent correctness bug**. The gap
to the ideal is therefore **evolutionary — convert convention → structure — not a redesign.**

> The single most important finding: the prior spike's highest-value item — a latent **silent-wrong-output**
> bug — was never fixed (the executed cleanup was scoped to bloat). This pass re-confirms it and shows it is
> **worse and more reachable** than the spike framed it. It is the one genuinely urgent item, and its proper
> resolution is to **complete the per-stage support, not to block the inputs that trigger it.**

---

## 1. The correctness gap (highest priority)

This is not architecture — it is a **silent wrong-output bug** that compiles, runs, and passes the entire test
suite, reachable from **unvalidated user input**. Two coupled instances, both rooted in the same structural
flaw: the per-stage-vs-global geometry split is **applied inconsistently**.

### 1.1 Simulation extraction strides equipment columns by the global stage-0 block count

`simulation/extraction.rs` decodes **every** equipment-column family — `extract_hydro_per_block`,
`extract_thermals`, `extract_exchanges`, `extract_buses`, `extract_non_controllables`,
`extract_pumping_stations`, `extract_hydro_no_turbine` — with the **global** `indexer.n_blks` (wired once from
stage 0 in `setup/mod.rs::build_wired_indexer`). The rest of the pipeline correctly strides with the **per-stage**
`ctx.block_counts_per_stage[t]` (`training/forward/stage_solve.rs`, `training/backward/lp_setup.rs`,
`simulation/pipeline.rs`). The LP _template_ itself is per-stage — `StageLayout::new` reads
`let n_blks = stage.blocks.len()` (`builder/layout.rs`). So each stage's columns are striped by _that_ stage's
block count, but simulation extraction reads them back with stage 0's. Under per-stage-varying block counts,
extraction reads the **wrong columns** → silently wrong reported outputs. _(This is the spike's Decision #2,
still open.)_

### 1.2 The NCS bound patch corrupts the LP _solve_, and the trigger is ordinary input

Worse than a reporting error: the global `StageIndexer.ncs_generation` range is wired once from stage 0
(`setup/mod.rs`, guarded only by a release-stripped `debug_assert_eq!` that checks NCS _start_ alignment, not
block counts), but the hot-path NCS column-bound patch strides it **per-stage**
(`training/forward/stage_solve.rs`, `training/backward/lp_setup.rs`, `simulation/pipeline.rs`). When the realized
NCS geometry differs from stage 0's, the patch writes bounds onto the **wrong columns during optimization** —
corrupting the solution, not just the report. There is a live `TODO(per-stage-block-count)` at
`indexer/constructors.rs` acknowledging this.

The realized NCS geometry differs from stage 0's whenever an NCS source **commissions or decommissions
mid-horizon**: `identify_active_ncs` (`builder/layout.rs`) filters the active set per stage on
`entry_stage_id`/`exit_stage_id`, both parsed from user input, with **no semantic validator** rejecting or even
warning on commissioning windows. So the trigger is not the exotic "varying block counts" — it is an ordinary,
documented modeling feature. No shipped example exercises it, which is why it is invisible.

### 1.3 The tell: NCS is the lone outlier

**Pumping stations** are a structurally identical block-major family with the _same_ entry/exit-window
mechanism, and they are handled **correctly per-stage everywhere** (`StageTemplates::pumping_col_starts: Vec`,
`n_pumping_per_stage: Vec`). NCS alone uses the global copy-back. The two families use **opposite strategies for
the same problem** — which both names the bug and points at the fix.

### 1.4 The fix is to complete the support, not to block the input

> **Design stance (project owner): solve the real problem; do not block user options.** A release-level guard
> that rejects non-uniform block counts and NCS commissioning windows would convert silent-wrong-output into a
> loud failure for ~10 LOC — but it does so by _forbidding_ legitimate modeling inputs. That is rejected here.
> The correct fix is to make NCS follow the **pumping per-stage pattern** (per-stage `ncs_col_starts[t]` on the
> patch path) and to thread the per-stage block count into `simulation/extraction.rs`. This **makes per-stage-
> varying geometry actually correct**, discharges both §1.1 and §1.2, and removes one of the SSOT mirrors in §2
> (D1). It is the one architectural fix that is also a correctness fix.

Verification must be by hash-neutrality on uniform-count cases (no behavior change there), plus a **new
deterministic case** exercising per-stage-varying block counts / an NCS commissioning window — the coverage
that is missing today.

---

## 2. Architectural degradations (ranked)

### D1 — The layout single-source-of-truth is not type-enforced; geometry lives in 4+ hand-synced owners

`StageIndexer` is the _nominal_ source of truth, but the column/row order is independently recomputed or
re-asserted in:

1. `StageIndexer` (`indexer/layout.rs` ranges + `constructors.rs` formulas) — the nominal SSOT.
2. `StageLayout::new` (`builder/layout.rs`) — recomputes the _live per-stage_ offsets (`col_ncs_start`,
   `col_pumping_start`, `row_generic_start`, the anticipated decision/state-out/fishing starts, `num_cols`,
   `num_rows`).
3. `setup::build_wired_indexer` (`setup/mod.rs`) — re-derives `ncs_generation` from stage-0 starts and **copies
   it back into the indexer** (the §1.2 mirror — the one that is _actively inconsistent_, not just redundant).
4. `generic_constraints::block_col_range` + the resolvers — re-apply `range.start + pos*n_blks + eff_blk`
   against the indexer directly.
5. `PatchBuffer` capacity formulas (`builder/patch.rs`) — re-encode the column/row counts.
6. The `#[cfg(test)]` layout assertions (`builder/layout.rs`) — re-assert the geometry as test invariants.

The indexer's own docs admit the split ownership (`ncs_generation` "populated after construction by the NCS
wiring in setup … NOT by `StageLayout::new`"; `EquipmentCounts::n_pumping` "accepted for structural symmetry …
but not read"). Every new column/row family must be threaded through ≥4 sites in agreement; drift is caught only
by debug-asserts and tests, never the compiler. **This is the structural root of §1.**

### D2 — The block-stride is a contract-by-comment, not a type

The core indexing operation `start + entity*n_blks + blk` (block-major) is hand-rolled at **~63 sites**
across `matrix.rs`, `generic_constraints.rs`, `builder/layout.rs`, `builder/patch.rs`, and
`simulation/extraction.rs`. Only ~12 route through the one centralizing accessor, `StageLayout::block_col`,
whose Contract comment names the wrong-but-compiling transposed form `blk*n_entities + entity` as the trap.

The centralization is **incomplete and mis-located**:

- `block_col` lives on `StageLayout`, but the two largest consumers — the `generic_constraints` resolvers and
  `simulation/extraction.rs` — hold `&StageIndexer` and **cannot call it**, so they re-open-code the stride
  (this is the _one_ genuine duplication the constraint-path analysis isolated: the stride **expression** is
  written twice, in `resolve_block_variable` and `StageLayout::block_col`; the base _offsets_ are single-owned).
- There is **no row analogue** — every block-major row (`row_*_start() + entity*n_blks + blk`) is hand-rolled.

A **typed multi-shape block-index primitive defined on `StageIndexer`** (the SSOT) would let the matrix-fill,
resolver, and extraction paths share one owner and convert the transposed-stride trap from prose into a compile
guarantee.

> **Caveat (load-bearing):** the primitive must model **≥2 geometries**, not a single flat `entity*n_blks+blk`.
> FPHA planes stride `blk*n_planes + p_idx` with a per-hydro base advancing by `n_blks*n_planes`, and the
> deficit column is a 3-term stride `bus*max_segments*n_blks + seg*n_blks + blk`. A naive one-shape primitive
> would force exactly the most error-prone sites to keep bypassing it, leaving the trap open where it matters
> most.

### D3 — Determinism is enforced by convention, not structure

Bit-for-bit declaration-order invariance (a hard project rule) rests on two **unenforced conventions**:

- **(A)** every entity `Vec` is ID-sorted upstream by `cobre-core::SystemBuilder::build`; `lp/` never sees raw
  declaration order.
- **(B)** the position maps (`hydro_pos`/`thermal_pos`/`line_pos`/`bus_pos`/`pumping_pos`) are `HashMap` used
  **only for `.get()`** — never iterated.

Neither is enforced by the type system or by lint. The workspace `[lints]` and `clippy.toml` carry **no
`disallowed-types`/`disallowed-methods`** for `HashMap` iteration. A maintainer can add a `HashMap`-iterating
fill, or push CSC entries after the per-column row-sort, with **zero compile-time or lint feedback** — only a
release-stripped `debug_assert`. The fast safety net is thin: only **two** structural CSC-permutation tests
exist (`matrix.rs::csc_byte_identical_under_permuted_declaration_order`, which uses a single bus so bus order is
unscrambleable; and `template.rs::lp_template_invariant_under_anticipated_index_permutation`). **No fast test
permutes bus/line/thermal/generic-constraint declaration order** — a `HashMap`-iteration regression in the
bus-balance fill would surface only in a slow multi-bus D-case parity run.

A latent sharp edge: the per-column CSC row-sort (`sort_unstable_by_key(|(row,_)| row)`) does **not** canonicalize
duplicate `(col,row)` keys, so a user constraint with two terms on the same variable has input-dependent
summation order (deterministic w.r.t. fixed input, but not canonicalized).

**Structural fix:** `BTreeMap` for the position maps (or a lint-guarded newtype) + a fast permutation test over
_all_ entity families.

### D4 — `matrix.rs` is a 4,946-LOC god-module that regenerates debt

Its production half is three **disjoint representation concerns** that share only the read-only build context and
write **disjoint** buffers:

| Concern                                          | ~LOC | Output buffer                       |
| ------------------------------------------------ | ---- | ----------------------------------- |
| Column-bound + objective fill (`fill_*_columns`) | ~754 | `col_lower`/`col_upper`/`objective` |
| `for_each_fpha_plane` shared row-cursor iterator | ~67  | (shared cursor)                     |
| Row-bound fill (`fill_*_rows`)                   | ~399 | `row_lower`/`row_upper`             |
| CSC-entry fill (`fill_*_entries`)                | ~718 | `col_entries`                       |
| `build_stage_matrix_entries` + `assemble_csc`    | ~70  | orchestration                       |

The clean decomposition is **by representation** (`builder/columns.rs` / `builder/rows.rs` / `builder/entries.rs`,
with `for_each_fpha_plane` shared by rows+entries) — **not by entity family**, which would split the FPHA
row-cursor invariant (`for_each_fpha_plane` drives both `fill_fpha_rows` and `fill_fpha_entries` off one cursor).
Blast radius is one caller (`build_single_stage_template` in `template.rs`); **determinism/hash risk is zero** —
the orchestrators preserve the exact fill-call sequence, so the CSC byte layout is invariant.

The size **causes a recurring debt class** — fresh instances found this pass:

- **Two more dead `_stage: &Stage` params** (`fill_operational_violation_rows`, `fill_operational_violation_entries`),
  identical to the ones the cleanup removed from the fishing fills — i.e. the earlier dead-param fix was
  **incomplete**.
- The **misattached doc block**: the 26-line 3-step doc for `fill_generic_constraint_entries` is glued to
  `struct LpMatrixBuffers`, leaving the function it describes undocumented.
- **Duplicated test fixtures** across the 6 in-file `#[cfg(test)]` modules (`make_ctx` ×4, `two_block_stage` ×4,
  `zero_hydro_penalties` ×3) — forced by the 6-concern split (`use super::*` is house-banned).

Splitting by representation localizes each fill family with its own test module + one fixture set, removing the
regeneration mechanism.

---

## 3. What is _not_ a degradation (do not over-correct)

- **The "dual constraint-construction path" is correctly layered, not redundant.** There is **one** LP subsystem:
  one column SSOT (`StageIndexer`), one CSC assembler (`assemble_csc`), one shared `col_entries` buffer
  (allocated once in `template.rs::build_stage_output`, written by both the built-in fills and
  `fill_generic_constraint_entries`, assembled once). The built-ins are the **closed, contract-bearing** core;
  the generic resolver is the **open, user-authored extension** — "two doors into the same room." Unification is
  **blocked by expressiveness, not performance** (both are startup-only, not hot path): the generic vocabulary
  (`VariableRef`/`ConstraintExpression` in infra-generic `cobre-core`) structurally cannot express
  incoming-state coefficients (FPHA's `−γᵥ/2` on `v_in`), theta/epigraph, per-plane FPHA row families, or
  derived `τ`/`ζ·ψ` couplings — and adding solver-internal `VariableRef` variants would breach the `cobre-core`
  genericity hard rule. **Keep two.**
- **The `StageIndexer`/`StageLayout` split is justified — and this pass _corrects_ the spike.** The spike said the
  split's redesign isn't worth it _"absent per-stage-varying block counts."_ But per-stage-varying _geometry_
  already exists today via NCS commissioning windows, anticipated-decision-active filtering, per-stage active
  generic constraints, and FPHA hydro selection. `StageLayout` is a legitimately **thick per-stage object** that
  correctly owns the variable geometry; it cannot collapse into the global indexer. The split is right — it is
  just **applied inconsistently** (NCS is the outlier, §1.3). The `StateLayout`/`StageStructure` redesign the
  spike sketched remains unjustified; the only piece now _forced_ is the localized per-stage NCS fix.
- **~2/3 of high-value contracts are already type/test-backed.** FPHA `−γᵥ/2` is pinned by named regression tests
  (`tests/template_integration.rs::fpha_ac3_v_in_column_entries` / `fpha_ac4_v_out_column_entries`, asserting the
  literal `−0.25`); the no-state-fixing-rows contract is pinned structurally by
  `matrix.rs::state_fixing_diagonals_absent_from_csc`; `block_col_range`'s family↔range pairing by its
  exhaustive-match test; the PatchBuffer category layout by named per-category tests. The contract-by-comment
  surface is real but smaller than the comment density suggests.

---

## 4. Test & code-quality gaps

- **Highest-value test gap — the storage-balance cascade routing.** `fill_state_and_water_entries` writes
  turbine/spillage/diversion `+τ`, cascade-upstream `−τ`, and AR-lag `−ζ·ψ` into the water-balance row, with
  **no focused CSC assertion**. A sign flip is invisible to the fast suite and caught only by the slow
  `d03-two-hydro-cascade` parity D-case. This is the most load-bearing prose contract with the weakest backstop
  — it should be pinned like FPHA, with a focused multi-reservoir CSC test (two upstream hydros, assert `−τ` on
  the upstream turbine/spillage in the downstream water row).
- **Three resolver arms the cleanup _found_ but never closed:** `HydroDiversion` (column math covered indirectly
  via the family test, but no arm test), `HydroWithdrawal` and `NonControllableCurtailment` (untested
  `=> vec![]` stubs). Plus the `resolve_block_variable` miss-path is tested via one arm only, and the
  block-independence predicate `true`-arms are tested only transitively.
- **The determinism fast net** is missing a permutation test over bus/line/thermal/generic-constraint order
  (D3).

---

## 5. Residual / carried-forward cleanup findings

Not architecture, but the open debt ledger this pass should record:

- **Incomplete dead-param removal:** `fill_operational_violation_rows` / `fill_operational_violation_entries`
  still carry dead `_stage` params (D4).
- **Misattached doc block** in `matrix.rs` (D4).
- **A second live `.claude/` doc reference:** `indexer/state_mapping.rs` module header still cites
  `.claude/rules/sddp.md` (the same N3-banned form fixed in `indexer/layout.rs`). Note the tension: `CLAUDE.md`
  itself points rustdoc at that file, so this is a known carve-out vs. N3 — reconcile the rule once.
- **A stale path** in a test doc header (`tests/declaration_order_invariance_anticipated.rs` cites
  `src/lp_builder/template.rs`; the real path is `src/lp/builder/template.rs`).
- **Pre-existing (predating the comment rules):** a `"Per Decision 3"` plan-token (N4) and a dead
  `indexer-layout-impact.md` doc-path (N3) in `indexer/constructors.rs`.

---

## 6. The ideal architecture (target shape)

The problem is modeled correctly; the ideal is the same model with four **"convention → construction"**
upgrades. This is evolutionary, not a rewrite.

1. **One typed layout owner with a multi-shape block-index primitive.** Every column/row derived through a single
   `BlockGrid`-style type on `StageIndexer` (modeling the flat, FPHA-plane, and deficit-segment shapes); the
   transposed-stride trap becomes a compile error; the stride expression has one home (D1 + D2). Per-stage
   _variable_ geometry stays owned by `StageLayout` — **consistently**, with NCS conformed to the pumping
   per-stage pattern, which also discharges §1 and removes the actively-inconsistent SSOT mirror.
2. **Determinism by type:** `BTreeMap` (or a lint-guarded newtype) for the position maps + a fast permutation
   test over all entity families (D3).
3. **`matrix.rs` split by representation** (columns/rows/entries) — localizing each fill family with its own test
   module and one fixture set, removing the debt-regeneration mechanism (D4).
4. **Mechanical contracts pushed to tests** — the cascade-`τ`/`ζ·ψ` family pinned like FPHA; every resolver arm
   and built-in family focus-tested — so the dense Contract-comment layer guards only the genuinely irreducible
   invariants (§4).

What the ideal explicitly does **not** include: a unified descriptor engine (the dual path is correctly two,
§3); the spike's `StateLayout`/`StageStructure` redesign (unjustified, §3); and a user-blocking validation guard
(the per-stage fix supports the input, it does not forbid it, §1.4).

---

## 7. Recommended sequence (real-fix-first)

| #   | Action                                                                                                                                                                                                        | Addresses                  | Effort | Risk                                                              |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | ------ | ----------------------------------------------------------------- |
| 1   | **Make NCS per-stage like pumping** (per-stage `ncs_col_starts[t]` on the patch path) **+ per-stage stride in `simulation/extraction.rs`**; add a varying-block-count / NCS-commissioning deterministic case. | §1.1, §1.2, one D1 mirror  | M      | med (hash-verify on uniform cases; new case for the varying path) |
| 2   | **Typed multi-shape block-index primitive on `StageIndexer`**; route matrix-fill, resolvers, and extraction through it.                                                                                       | D2, the stride duplication | M      | low (hash-neutral)                                                |
| 3   | **Split `matrix.rs` by representation** + finish the dead-param / misattached-doc / duplicated-fixture cleanup.                                                                                               | D4, §5                     | M      | ~none                                                             |
| 4   | **Determinism structural guard** (`BTreeMap` position maps or a `disallowed-methods` lint) + a multi-entity CSC permutation test.                                                                             | D3                         | S      | low                                                               |
| 5   | **Cascade-`τ` focus test** + the three missing resolver-arm tests + reconcile the residual `.claude/`/stale-path drift.                                                                                       | §4, §5                     | S      | none                                                              |

**Net:** the architecture is largely _good_ — there is little to remove and nothing big to redesign. What is
missing is the conversion of its **conventions, comments, and copies into compiler-and-test-enforced
structure** — and, urgently, the one latent correctness bug that three documents have now flagged and zero
passes have fixed. The fix for that bug is to **finish the per-stage support**, not to forbid the inputs that
expose it.

---

## 8. Key files

- `crates/cobre-sddp/src/simulation/extraction.rs` — §1.1 (global-`n_blks` equipment strides).
- `crates/cobre-sddp/src/setup/mod.rs` — `build_wired_indexer` NCS copy-back + debug-only guard (§1.2, D1).
- `crates/cobre-sddp/src/lp/indexer/constructors.rs` — `n_state` formula; `TODO(per-stage-block-count)` (§1.2).
- `crates/cobre-sddp/src/lp/builder/layout.rs` — `StageLayout`, `StageLayout::new`, `block_col`,
  `identify_active_ncs`, per-stage geometry (§1, D1, D2).
- `crates/cobre-sddp/src/lp/indexer/layout.rs` — `StageIndexer`, the `ncs_generation` / `n_pumping` ownership
  notes (D1).
- `crates/cobre-sddp/src/lp/builder/matrix.rs` — the god-module; `fill_state_and_water_entries` (cascade `τ`),
  `for_each_fpha_plane`, `fill_*_columns`/`fill_*_rows`/`fill_*_entries` (D4, §4).
- `crates/cobre-sddp/src/lp/builder/template.rs` — `build_stage_output` (single buffer + assembler);
  `ncs_col_starts` / `pumping_col_starts` per-stage Vecs (§1.3, §3).
- `crates/cobre-sddp/src/lp/generic_constraints.rs` — `resolve_variable_ref`, `block_col_range`,
  `resolve_block_variable` (the open extension path; D2, §3, §4).
- `crates/cobre-sddp/src/lp/builder/patch.rs` — `PatchBuffer` capacity formulas (D1).
- `crates/cobre-core/src/system/builder.rs` — `SystemBuilder::build` ID-sort (the determinism linchpin, D3).
- `crates/cobre-sddp/src/training/forward/stage_solve.rs`, `training/backward/lp_setup.rs`,
  `simulation/pipeline.rs` — the correct per-stage callers that NCS/extraction diverge from (§1).
- `docs/design/lp-construction-bloat-analysis.md`, `docs/design/lp-layout-architecture-spike.md` — passes 1 & 2.
