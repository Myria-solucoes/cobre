# `crates/cobre-sddp/src/lp/builder/` — Design Analysis (Simplification Lens)

**Scope:** the LP template-construction submodule (`crates/cobre-sddp/src/lp/builder/`):
`StageLayout` + the per-stage geometry, the template orchestrator, the column/row/entry fills, the
hot-path `PatchBuffer`, and scaling. This is the **role-(b)** half of the LP layout (per-stage
geometry), companion to the **role-(a)** indexer analysis.
**Question (the lens):** the same parsimony / untangling lens as the indexer pass — is the
machinery proportionate, and what is the cleanest decomposition?
**Companion:** `lp-indexer-simplification-assessment.md` (role (a)). Its §8 `StateLayout` /
Geometry split spans both docs; `builder/` owns the **Geometry** side, and this doc grounds what
that side actually needs.
**Starting condition (differs from the indexer):** `builder/` has just been through the
`matrix.rs`-split refactor (`columns.rs` / `rows.rs` / `entries.rs`) and the `BlockGrid` migration;
read as-is, not from any earlier snapshot.
**Method:** first-hand reads + targeted greps.
**Status:** analysis only — no code changed. Point-in-time; symbol anchors stable, line anchors
will drift.
**Date:** 2026-06-20.

---

## 1. The `builder/` terrain

| File             | prod LOC | Concern                                                            | Role                                    |
| ---------------- | -------- | ------------------------------------------------------------------ | --------------------------------------- |
| `layout.rs`      | ~1204    | `StageLayout` + satellites + `identify_*` / generic-row helpers    | **role (b) core** — per-stage geometry  |
| `template.rs`    | ~835     | `build_stage_templates`, ctx build, `assemble_*`, `StageTemplates` | orchestrator                            |
| `columns.rs`     | ~841     | ~25 `fill_*_columns` (bounds + objective)                          | fills — column representation           |
| `rows.rs`        | ~318     | 6 `fill_*_rows` (row bounds)                                       | fills — row representation              |
| `entries.rs`     | ~897     | CSC matrix-entry fills (73% tests)                                 | fills — matrix representation           |
| `fpha_cursor.rs` | 69       | `for_each_fpha_plane` (shared row cursor)                          | fills — shared FPHA seam                |
| `patch.rs`       | ~603     | `PatchBuffer`                                                      | **hot path** — per-solve mutation       |
| `scaling.rs`     | ~215     | col/row scaling                                                    | numeric conditioning                    |
| `mod.rs`         | ~128     | constants + `GenericConstraintRowEntry`                            | shared types (read in the indexer pass) |

Scan flags to resolve as we go: `entries.rs` is 73% tests (calibrate — likely earned for CSC
coefficients); `rows.rs` has **zero in-file tests** while its sibling fills carry large test
modules (a post-split placement question).

---

## 2. Per-file analysis (the living part)

### 2.1 `layout.rs` — `StageLayout` (the per-stage geometry owner)

**What it is.** ~1204 prod LOC (≈53% tests). A ~30-field struct plus four satellite structs
(`ResolvedTables`, `TemplateBuildCtx`, `AnticipatedLayout`, `GenericConstraintLayout`), the
`identify_{fpha,evap,active_ncs}_hydros` helpers, `enumerate_generic_constraint_rows`, and a
~45-method accessor surface. Structurally it is **three things stacked**: (i) an **embedded full
`StageIndexer`** (`indexer: StageIndexer`), rebuilt every stage; (ii) the per-stage-only families
the indexer cannot represent — NCS active set, pumping, generic constraints, anticipated
active-row counts, FPHA/evap local maps, `zeta`, `num_cols`/`num_rows`; (iii) a unifying accessor
layer over (i)⊕(ii).

**SL1 — it embeds and rebuilds the _whole_ indexer per stage (the redundant recompute, geometry
side).** `StageLayout::new` calls `StageIndexer::with_equipment_and_evaporation` every stage,
producing a fresh indexer whose **role-(a) half is stage-invariant** (storage / inflow_lags /
anticipated_state / n_state ranges are identical across stages) and whose two caches
(`nonzero_state_indices`, `state_to_lp_column_map`) are left **empty** (only `build_wired_indexer`
finalizes those, on the global instance). So the per-stage indexer is used only for its geometry
ranges + `resolve_variable_ref` in generic-constraint lowering. This is the geometry-side
confirmation of indexer §8: `StageLayout` should hold a **handle to the shared `StateLayout`** +
own role-(b) geometry directly, not embed a full duplicate indexer.

**SL2 — the ~30 read-through accessors are a symptom of the embed, not an inherent need.** Roughly
thirty one-line accessors (`col_theta` → `indexer.theta`, `col_thermal_start` →
`indexer.thermal.start`, …) exist purely to delegate to the embedded indexer "so the offset lives
in one place and cannot drift from a flattened copy" (the stated rationale — good single-owner
intent). But under the §8 split, role-(b) geometry becomes `StageLayout`'s **own fields** (so
`col_turbine_start()` etc. collapse to direct reads) and the role-(a) ones (`col_theta`,
`col_storage_in_start`, `col_inflow_lags_start`, `col_anticipated_state_start`, `n_state`,
`col_z_inflow_start`) become reads on the `StateLayout` handle. The accessor layer largely
dissolves; its size today is a cost of storing geometry inside an embedded object.

**SL3 — `StageLayout::new` is the principal consumer of the indexer's F4 recovery accessors.**
`col_generation_start` → `indexer.generation_col_start()`, `col_evap_start` →
`indexer.evap_col_start()`, and the empty-hydro fallbacks for `col_ncs_start` and
`row_min_generation_start` route through `post_equipment_col_start()` / `post_equipment_row_start()`.
So the F4 normalization-discards-the-cursor defect (indexer §7.3) has its blast radius **here**:
fixing the representation (carry real cursors, don't collapse to `0..0`) removes these consumers
too.

**SL4 — the genuine, irreducible per-stage core (do not over-correct).** The NCS active set
(`identify_active_ncs` + `col_ncs_start`/`n_ncs`/`active_ncs_indices`), pumping
(`col_pumping_start`/`n_pumping`), generic-constraint rows + slack columns
(`enumerate_generic_constraint_rows`), and the anticipated active-row counts
(`AnticipatedLayout`) are exactly what _must_ be per-stage and cannot live on a global indexer.
This is the legitimate reason a per-stage layout object exists. It stays.

**SL5 — two `#[allow(dead_code)]` fields.** `n_anticipated_state_out_def_rows` (read by `matrix`
debug-asserts; lint false-positive from cross-module field access) and `n_ant_state` (asserted in
tests; production recomputes `n_anticipated * k_max` inline). The latter is a stored-but-unread
field — mild redundancy.

**Calibration — not smells.** `StageLayout::new`'s `#[allow(too_many_lines)]` is earned (a
sequential offset chain whose value is the auditable read-vs-recompute ordering); the satellite
struct groupings are cohesion-only (documented as "struct shape, not value"); the block-major
accessors correctly delegate to `BlockGrid` (epic-02); and `TemplateBuildCtx`'s position maps are
`BTreeMap` — the **determinism-by-construction** change already landed (a genuine structural
improvement, the kind the indexer doc §6 advocates).

**§8 tie-in (what this file becomes).** `StageLayout` = `&StateLayout` (shared, role a) + its own
role-(b) geometry fields + the per-stage extras (SL4). Deletions on this side: the embedded
role-(a) recompute (SL1), ~half the read-through accessors (SL2), and the F4-recovery consumption
(SL3). The per-stage extras and the generic-constraint machinery are untouched. Net: `StageLayout`
gets materially smaller and its dependency on `StateLayout` becomes an explicit handle instead of
a per-stage clone.

### 2.2 `template.rs` — the per-stage template orchestrator

**What it is.** ~835 prod LOC (≈53% tests). The driver: `StageTemplates` (the wide output bundle),
`build_stage_templates` (public entry), `build_template_build_ctx` (assembles the shared
`TemplateBuildCtx`), `build_single_stage_template` (one stage), and
`assemble_stage_templates_output` (transpose).

**The cleanest file in the cluster — and that is a calibration point.** Unlike the indexer, this
file is largely _well-designed_, and saying so matters: it shows the cluster's problems are
localized debt, not house style.

- `build_single_stage_template` is a clean linear composition: `StageLayout::new` →
  `columns::fill_stage_columns` → `rows::fill_stage_rows` → `entries::build_stage_matrix_entries` →
  `fill_generic_constraint_entries` → objective scale → CSC row-sort → `assemble_csc`. The epic-03
  representation split pays off here: each fill is a separate, sequenced module call.
- `StageBuildOutput` → `assemble_stage_templates_output` is a deliberate **per-stage-struct → SoA
  transpose**: one transpose point, named fields (not threaded tuples), "add a datum = one field +
  one transpose line." A good, cache-friendly output shape.
- The discount-factor **two-phase init is done right** — the F2 pattern, _encapsulated_:
  `discount_factors`/`cumulative_discount_factors` are **private** with a getter and a single
  `set_discount_factors` that recomputes the cumulative from the per-stage atomically (cannot
  drift), explicitly to stop a caller reading the `1.0` placeholder as the real value. This is the
  safe version of the indexer's F2 (public empty cache + dense fallback): the codebase knows the
  right pattern; the indexer simply does not use it.
- COST_SCALE_FACTOR objective scaling is a **single-point** transform (all coeffs except `theta`,
  with a load-bearing contract comment on why `theta` is exempt) rather than scattered per-fill —
  safer by construction.
- Position maps are `BTreeMap` (epic-04 determinism, again).

**§8 tie-in — this is where the per-stage recompute is _driven_, and where `StateLayout` is
built.** The per-stage loop calls `build_single_stage_template` → `StageLayout::new` → a fresh full
`StageIndexer` every stage (the SL1 recompute). So `template.rs` is the concrete home of the
indexer §8.5 wiring: `build_template_build_ctx` would construct the shared `StateLayout` **once**
(next to the ctx), and `build_single_stage_template` would build only per-stage geometry + hand it
the `&StateLayout`. No new abstraction — one construction simply moves up out of the loop.

**Minor redundancies (both §8-adjacent).** (1) The cumulative discount factors are computed
**twice** — once in `build_template_build_ctx` (real, for the anticipated objective at build time)
and again in postprocess via `set_discount_factors` (the output's theta coefficient); the
placeholder-then-postprocess split forces the recompute. (2) The anticipated metadata vecs are
computed once in the ctx, then **cloned into each per-stage indexer** in `StageLayout::new` — a
per-stage clone the §8 shared-`StateLayout` ownership removes.

**Calibration — not smells.** The sequential per-stage build loop is startup-only (parallelizing
buys nothing on the cold path and complicates determinism); `assemble_*`'s
`#[allow(too_many_arguments)]` is earned (distinct lifetimes/ownership, one cold call);
`StageTemplates`'s width (~16 parallel per-stage Vecs) is the inherent output contract of an LP
build that emits a lot of per-stage metadata, not bloat.

### 2.3 `columns.rs` — column bounds + objective fills

**What it is.** ~841 prod LOC (≈48% tests). One orchestrator `fill_stage_columns` + ~22
`fill_*_columns` helpers (each writing one disjoint column family's bounds/objective into a shared
`ColumnBufs`), plus the `BlockSlackFamily` enum + `fill_block_family`.

**Mostly clean — a second positive data point.** Like `template.rs`: a flat per-family helper
list, each small and focused; the epic-03 representation split visibly working. Two things are
notably _good_:

- **`BlockSlackFamily` + `fill_block_family` is DRY done right.** The four operational-violation
  slack families differ on exactly three axes (activation predicate, column accessor, cost field);
  one enum-dispatched function selects all three via `match` instead of four near-identical fills.
  Closed-set enum dispatch (the house rule), not `Box<dyn>`.
- The dense contract comments are **earned** — the `withdrawal_slack` sign-flip cap
  (`R ∈ [0,T]` / `[T,0]`), the `ncs` stochastic-overwrite note, the discount-NPV
  anticipated-objective formula are all genuinely subtle, silent-wrong-if-broken invariants.

**The anticipated footprint surfaces again (the through-line).** Five of the ~22 helpers touch
anticipated columns/objective (`fill_anticipated_state_columns`,
`fill_anticipated_state_out_columns`, `fill_anticipated_decision_columns`,
`fill_anticipated_decision_objective`, `zero_anticipated_delivery_thermal_cost`). Within them:

- **`is_anticipated_decision_active` is recomputed ~3× per plant per stage** (state-out,
  decision-bounds, decision-objective each re-derive the active set), and the decision **bounds and
  objective are two separate passes** that each re-look-up `delivery_stage` + `thermal_idx` +
  `thermal_bounds`. Merging them (one loop per plant) removes the duplicate lookups. Minor
  (startup), but it is the same anticipated-machinery density found in the indexer (§7.2 bug,
  redundant `state_out`, dead iterator).
- **`zero_anticipated_delivery_thermal_cost` is a write-then-zero with an implicit ordering
  contract.** `fill_thermal_columns` writes the standard per-block cost for _all_ thermals; this
  helper then overwrites the anticipated plants' cost back to `0.0` (the cost is charged at the
  decision column instead, to avoid double-counting). It carries a "must run AFTER
  `fill_thermal_columns`" comment — so this one fill is _order-dependent_ in an otherwise
  commutative (disjoint-range) list. Documented, but fragile: reorder the list and the double-count
  returns silently. A cleaner shape has `fill_thermal_columns` skip anticipated plants' cost
  outright.

**Cross-cutting contract worth flagging.** The "read the **resolved per-stage** bound, never the
entity declaration" trap recurs by comment at ≥3 sites (`fill_diversion_columns`,
`fill_thermal_columns`, `fill_block_family`): reading `hydro.diversion.max_flow_m3s` instead of
`resolved.bounds.hydro_bounds(...).max_diversion_m3s` silently drops per-stage overrides while
compiling. A real systemic silent-wrong trap guarded only by prose — a candidate for a type-level
guard (deny fill-time access to declaration-time bounds), though that is an infra-crate change.

**Net.** `columns.rs` is well-factored; no architectural defect of its own. Its findings are (i)
more evidence the cluster's debt is localized (this file and `template.rs` are clean), and (ii) the
**anticipated machinery is the densest, most-redundant feature across both indexer and builder** —
the highest-leverage redesign target, and the one the §7.2 bug + §8 relocation already touch.

### 2.4 `rows.rs` + `fpha_cursor.rs` — row-bound fills and the shared FPHA cursor

**`fpha_cursor.rs` — the exemplar of good factoring in the cluster (69 LOC, no in-file tests
needed).** `for_each_fpha_plane` is the single owner of the FPHA row-cursor arithmetic; both
`fill_fpha_rows` (bounds) and `fill_fpha_entries` (coefficients) drive off it, so a one-sided edit
cannot land the row bounds and the matrix coefficients on different rows. It handles the
variable-plane-count trap correctly (per-hydro base advances by the cumulative `n_blks*n_planes`
prefix sum, not a uniform stride), routes through `BlockGrid::fpha_plane` (the block-OUTER /
plane-INNER shape), and the closure is a monomorphised `FnMut` borrowing its buffer — no
`Box<dyn>`, no intermediate `Vec`, zero alloc, byte-identical push order. This is precisely the
"shared cursor as single owner" discipline the indexer's hand-rolled strides lack — the positive
model the redesign should generalize.

**`rows.rs` — clean, mirrors `columns.rs`.** ~318 prod LOC. One orchestrator + 8 per-family fills
(water-balance, load-balance, FPHA-via-cursor, evaporation, operational-violation,
anticipated-fishing, anticipated-state-out-def, z-inflow), each writing a disjoint row range with
the constraint math documented. No architectural defect.

**The "zero in-file tests" question — resolved: placement, not a gap.** The row-fill functions have
**no direct test callers**; their `row_lower`/`row_upper` contracts are asserted **transitively**
through full-template builds — in the integration suite (`tests/template_integration.rs`,
`tests/integration.rs`, `tests/determinism.rs`, …) and in `entries.rs`'s own test module (which
builds the whole template, so it asserts row bounds alongside the CSC entries). Since rows and
entries are built together and share the FPHA cursor, co-testing them in `entries.rs` is
defensible. The mild residual: the **simple row-bound RHS formulas** — `zeta·(base − withdrawal)`
(water-balance), `base` (z-inflow), `intercept` (FPHA/evaporation), the operational senses — have
**no fast focused unit test**, so a sign/scale regression in them surfaces only in an integration
or slow D-case run. This is the row-side analogue of the assessment's cascade-`τ` test-gap note
(milder, since the RHS is simpler than the matrix coefficients): a candidate for a small
co-located row-bounds module if fast coverage is wanted.

**Two minor notes.** (1) Anticipated footprint again: 2 of the 8 fills are anticipated
(`fill_anticipated_fishing_rows` dense/always-active, `fill_anticipated_state_out_def_rows`
sparse/active-only). (2) A tiny idiom inconsistency: the four operational-violation families are
dispatched by an **enum** (`BlockSlackFamily`) in `columns.rs` but by an inline **tuple-array
descriptor** in `rows.rs::fill_operational_violation_rows` — same four families, two shapes across
the two files. Harmless, but a shared descriptor would unify them.

### 2.5 `entries.rs` — CSC matrix-entry fills (the correctness heart)

**What it is.** ~897 prod LOC (≈73% tests — the largest builder file). ~11 per-concern fills
writing `(row, coeff)` into `col_entries`, the `build_stage_matrix_entries` orchestrator (10
fills), `assemble_csc`, and `LpMatrixBuffers`. Well-organized despite its size — the same
per-concern structure as `columns.rs`/`rows.rs`, plus a clean CSC assembler.

**The 73% test ratio is EARNED — the calibration answer.** This file is where every silent-wrong
sign/coefficient error lives, so heavy testing is the correct response (the `state_mapping.rs`
pattern, not the `BlockGrid` one): the FPHA `−γᵥ/2`-on-BOTH-storage-columns average-storage
contract (D06-pinned; the doc names the wrong-but-compiling `−γᵥ/2`-on-`v`-alone alternative); the
water-balance cascade routing (turbine/spillage/diversion `+τ` own row, cascade-upstream `−τ`,
diversion-into `−τ`, AR-lag `−ζ·ψ`, evap `+ζ`, withdrawal `∓ζ`); the load-balance signs (FPHA `+1`
vs `ρ·q`, line fwd/rev, pumping power `−consumption` as a negative injection, deficit `+1` / excess
`−1`). These are genuinely treacherous; the density guards them.

**The one real coverage note — the cascade-`τ` focus test is still open.** Despite the 73% ratio,
the assessment's #1 test-gap stands: `fill_state_and_water_entries`'s multi-reservoir cascade
`±τ` / `−ζ·ψ` routing has **no focused CSC assertion** — caught only by the slow `d03` cascade
D-case. The fix is queued but not landed (the pending cascade-`τ` water-row CSC test in the
remaining hardening work). So "high test %" ≠ "cascade routing covered": the most load-bearing
prose contract in the builder still has the weakest fast backstop.

**Anticipated footprint, fifth appearance.** Two fills here are anticipated
(`fill_anticipated_fishing_entries`; `fill_anticipated_state_out_def_entries` — the latter writes
the `+1/−1` identity of the redundant `state_out` def row, §7.2 F-redundant). Tally across both
clusters: anticipated now spans the indexer (state-mapping + layout + constructors), `columns.rs`
(5 fns), `rows.rs` (2), and `entries.rs` (2) — ~4 files, ~15+ functions. It is unambiguously the
most cross-cutting feature in the LP build, and the redesign's highest-leverage consolidation
target.

**Minor.** The AR-lag `psi` iteration (`col_inflow_lags_start + lag*n_h + h`, nonzero-`psi` guard)
is written twice — `−ζ·ψ` in the water row (`fill_state_and_water_entries`) and `−ψ` in the
z-inflow row (`fill_z_inflow_entries`). Same stride + guard, different coefficient/row; a shared
lag-walker (à la `for_each_fpha_plane`) would unify them. It is also another hand-rolled
state-vector (lag-major) stride — the D2 family `BlockGrid` does not cover — now at its 3rd+ site.

**Positives.** `assemble_csc` carries the load-bearing sorted-entries debug-assert (the caller owns
the sort; the assembler refuses to mask a missing sort rather than silently re-sorting — exactly
right for a contract whose violation HiGHS/CLP would misfactorize). `fpha_local_index` is a
single-owner cached reverse map shared by the load-balance and op-violation fills (no per-call
rebuild). The i32-cast sites carry bounded-size rationale.

**Net.** No architectural defect; the correctness heart is sound and appropriately tested. The
actionable items are (i) land the cascade-`τ` focus test (closes the assessment's #1 gap), (ii) the
anticipated consolidation (cross-cluster), (iii) optionally a shared lag-walker.

### 2.6 `patch.rs` (hot path) + `scaling.rs` (conditioning)

**`patch.rs` — the cluster's only hot-path file, and exemplary HPC code.** `PatchBuffer`
pre-allocates both regions once in `new` (row-bound `N+M·B+N`; column-bound `N·(1+L)+A·K`) and the
`fill_*` methods overwrite in place — **zero hot-path allocation**, the workspace-reuse hard rule
done right. Judged against the HPC bar (not clean-code aesthetics), it passes:

- The patch categories map exactly to the state-pinning contract: Categories 1/2/6
  (storage/lag/anticipated-state) → **column bounds** (`fill_col_state_patches`), Categories 3/4/5
  (noise/load/z-inflow) → **row bounds**. Scaling directions honor `sddp.md`: state-fixing divides
  by `col_scale`, row patches multiply by `row_scale`, and Category-3 noise is deliberately **not**
  prescaled (the `_row_scale` param is accepted-but-unused with a documented double-scaling-trap
  rationale). Earned.
- `active_load_patches`/`active_z_inflow_patches` track the per-stage slice length so
  `forward_patch_count()` returns the exact prefix — correct per-stage-varying-block handling on the
  hot path. `fill_load_patches` threads `BlockGrid` (epic-02 ticket-024).

Two threads recur even here: **anticipated footprint, 6th appearance** (Category 6 =
`fill_anticipated_state_col_patches`, an A·K hot-path region); and the **hand-rolled state-vector
strides** (`n + lag*n + h`, `slot*n_ant + plant`) — the lag-major / slot-major family `BlockGrid`
does not cover, now at its 4th site. §8 tie-in: `fill_col_state_patches` reads
`indexer.{storage_in,inflow_lags,anticipated_state}.start` — pure role-(a); under §8 it takes a
`StateLayout` handle, not the full `StageIndexer`.

**`scaling.rs` — clean, standard, no findings.** Geometric-mean `1/√(max·min)` col/row prescaling
(the `D_r·A·D_c` conditioning) + `compute_noise_scale`, applied **offline** at setup (not hot
path). Scaling directions correct (col: values·d, objective·d, bounds/d; row: values·d, bounds·d),
clippy allows earned (CSC non-negativity, `j+1` index). A small, well-contained numeric module.

### 2.7 Builder synthesis (all files analyzed)

The builder cluster is fully mapped. The verdict is **calibrated, not uniformly critical**:

- **Most of the builder is clean and well-engineered.** `template.rs` (orchestrator),
  `columns.rs`/`rows.rs`/`entries.rs`/`fpha_cursor.rs` (the post-split fills), `patch.rs` (hot
  path), and `scaling.rs` are all sound — epic-02/03 (the `matrix.rs` split + `BlockGrid`) and the
  HPC discipline genuinely worked. `fpha_cursor.rs` is the positive exemplar; `entries.rs`'s heavy
  testing is earned; `patch.rs` is textbook no-alloc reuse.
- **Debt is concentrated in exactly two places.** (1) **`layout.rs`'s `StageLayout`** — the
  embedded-and-rebuilt-per-stage `StageIndexer` (redundant role-(a) recompute), the ~30
  read-through accessors, and the F4 recovery-accessor consumption; the **Geometry side of the §8
  split**, which shrinks materially when `StateLayout` lifts out. (2) **The anticipated machinery's
  cross-file spread** — confirmed across ~5 files and ~16+ functions (indexer
  state-mapping/layout/constructors, `columns` ×5, `rows` ×2, `entries` ×2, `patch` ×1), carrying
  the §7.2 latent bug, the redundant `state_out` alias, the dead iterator, the write-then-zero
  ordering trap, and the 3×-recomputed active set.
- **Two cross-cutting traps guarded only by prose** (the D2/D3 family): the hand-rolled
  **state-vector strides** (lag-major / anticipated slot-major) at ~4 sites that `BlockGrid` does
  not cover; and the **"read resolved per-stage, never entity-declaration" bound trap** at ≥3 sites.
- **One open coverage gap:** the cascade-`τ` water-row focus test (pending hardening work).

**The two highest-leverage redesign levers, now seen from both clusters:**

1. **The `StateLayout` / Geometry split (§8).** Slims `StageIndexer` _and_ `StageLayout`, fixes the
   §7.2 bug by construction (option B), and removes the F2 (dual-mode + dense fallback) and F4
   (range-collapse recovery API) families. It is _subtractive_ — separation, not new abstraction.
2. **Anticipated machinery — mostly irreducible (corrected; see §2.8).** The cross-file spread is
   the honest cost of a genuinely multi-faceted, _correct_ feature, not bloat: the fishing row, the
   ring-buffer state, and the decision column are all required by the math (sddp-specialist verdict,
   §2.8). The removable surface is **small**: relocate `anticipated_state_out` via §7.2 option B
   (which _is_ the `StateLayout` work in lever 1, NOT a separate cut — and a _relocation_, never a
   _deletion_), merge the 3× active-set recompute, delete the dead iterator, de-dup the inlined
   activation predicate, reconsider the write-then-zero. Modest, not the headline.

**Secondary (weigh against the over-engineering caution):** a typed **state-vector stride
primitive** (the lag-major/slot-major analogue of `BlockGrid`) would close the ~4 hand-rolled
state-stride sites — but it is _additive_ machinery, and `BlockGrid` itself is the one piece
already judged heavy (§6 cross-ref). Treat as optional; a `for_each_fpha_plane`-style shared
_walker_ (a closure, not a type) is the lighter answer where a stride recurs (e.g., the AR-lag
`psi` loop).

### 2.8 Formulation note — why anticipated dispatch needs equality rows (sddp-specialist verdict)

"Why does anticipated dispatch keep equality rows when inflow state migrated to pure column-bound
pinning?" has a precise, math-level answer. There are **two** equality-row families, with opposite
necessity.

**The principle.** Column-bound pinning suffices for any state value that re-enters its stage only
as a fixed **coefficient** — the storage head term, the inflow lag's `−ζ·ψ` term in the
water/z-inflow rows. Such a value is _consumed_: once the column is pinned, the term is a constant
the LP cannot alter and the dispatcher must satisfy nothing. It **cannot** express a state value
that imposes a **delivery obligation** across several free decision variables — that is a
Σ-equality (a hyperplane, not a box), so no column bound reproduces it. The anticipated commitment
is **the only state in the model carrying such an obligation** (a must-deliver _output_ of a prior
recourse, vs. the inflow lag's consumed _input_) — which is why it alone needs an equality row.

**Irreducible (keep):**

- **`anticipated_fishing`** (`Σ_blk hours·gen[blk] = hours_total · commitment`) — the delivery
  obligation; structurally impossible as a column bound (it couples _n_blk_ free generation
  decisions to one pinned state scalar; the dispatcher keeps the block-profile degree of freedom).
- **`anticipated_state` ring buffer** — a genuine `K_i`-deep Markov state (the in-flight delivery
  pipeline), the exact analogue of the PAR(p) lag vector; pinned via column bounds exactly like the
  inflow lag (no equality row used for the pinning). Padding to `k_max` is a deliberate dense-layout
  choice, not a defect.
- **`anticipated_decision`** — the actual priced decision variable.

**`state_out` is a cut _target_, not a cut _source_ — and is _relocatable_, not deletable.** The
Benders subgradient is harvested in the backward pass from the **incoming ring-buffer column's**
reduced cost (`state_to_lp_incoming_column`), never from `state_out`; `state_out` appears only in
the forward pass as the column a future cut row writes onto. So removing it threatens not
subgradient _validity_ — only _basis stability_. Collapsing it onto `decision` is deterministically
identical but **(a)** does not fix the §7.2 bug (`decision.start` is also `n_blks`-dependent) and
**(b)** drops the future-cost gradient onto a bound-active, NPV-costed column → dual-vertex drift +
warm-start **basis rejections**. Verdict: **option B (relocate `state_out` into the stage-invariant
state region) strictly dominates collapse** — it keeps the dedicated free carrier (basis
uniformity) _and_ fixes the bug by construction. The §7.2 "delete A cols + A rows" framing is
superseded: the column **moves**, it does not disappear.

**Net for the plan.** The anticipated feature is far more irreducible than its footprint suggests.
The only mechanical wins are the dead iterator and the predicate de-dup; the structural win is the
`state_out` relocation, which _is_ the §8/`StateLayout` + option-B work — not a separate
"anticipated consolidation."

| Builder file     | Status               | Note                                                               |
| ---------------- | -------------------- | ------------------------------------------------------------------ |
| `layout.rs`      | **analyzed**         | §2.1 — role-(b) core; redundant indexer embed + read-through layer |
| `template.rs`    | **analyzed**         | §2.2 — cleanest file; loop driver of the §8 recompute              |
| `columns.rs`     | **analyzed**         | §2.3 — clean fills; anticipated footprint + write-then-zero        |
| `rows.rs`        | **analyzed**         | §2.4 — clean; bounds tested via full-template builds               |
| `entries.rs`     | **analyzed**         | §2.5 — correctness heart; earned tests; cascade-τ gap open         |
| `fpha_cursor.rs` | **analyzed**         | §2.4 — exemplar single-cursor owner                                |
| `patch.rs`       | **analyzed**         | §2.6 — exemplary no-alloc HPC; anticipated 6th appearance          |
| `scaling.rs`     | **analyzed**         | §2.6 — clean standard conditioning; no findings                    |
| `mod.rs`         | analyzed (cross-ref) | constants + `GenericConstraintRowEntry` (indexer pass)             |

---

## 3. Key symbols

- `crates/cobre-sddp/src/lp/builder/layout.rs` — `StageLayout` (the `indexer` field; `new`; the
  `col_*`/`row_*` read-through accessors; the `block_col`/`*_col` BlockGrid accessors),
  `TemplateBuildCtx`, `ResolvedTables`, `AnticipatedLayout`, `identify_active_ncs`,
  `enumerate_generic_constraint_rows`.
- `crates/cobre-sddp/src/lp/indexer/...` — the role-(a) counterpart; see
  `lp-indexer-simplification-assessment.md` (esp. §8, the `StateLayout` extraction this file's
  Geometry side pairs with).
