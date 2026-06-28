# StateLayout / CutStateLayout relationship redesign

Design proposal — owner decision pending. This document evaluates how to make the
relationship between the full LP state-vector descriptor and its per-stage cut
projection legible **without changing behavior**. The chosen direction is a pure,
hash-neutral refactor implemented in a separate phase after the `## Decision`
section below is filled in.

Scope is the `StateLayout` / `CutStateLayout` pair only. This is adjacent to —
but deliberately does **not** absorb — the broader `StageIndexer` / `StageLayout`
/ `StageEquipmentGeometry` geometry-triplication tension (no committed design doc
for that exists in this tree at the time of writing). Nothing proposed here may
conflict with a future collapse of that triplication, but that work stays out of
scope.

## Problem

The LP state vector is described by two types whose names read as siblings of one
concept but whose roles diverge:

- **`StateLayout`** (`crates/cobre-sddp/src/lp/indexer/state_layout.rs`) is the
  global, full LP state-vector descriptor. It owns the column arithmetic — the
  outgoing resolver `state_to_lp_column` (cached by `state_to_lp_column_map` /
  read through `lp_column_for_state`) and the incoming resolver
  `state_to_lp_incoming_column` — plus `nonzero_state_indices`, `theta`, the
  state-region ranges, and `is_anticipated_decision_active`. One instance per
  study; it drives LP construction across roughly two dozen files.
- **`CutStateLayout`** (`crates/cobre-sddp/src/lp/indexer/cut_state_layout.rs`) is
  a per-stage projection exposing only the cut-state dimensions a stage's
  `StageStateConfig` enables (storage and/or inflow lags; anticipated state
  always included). It holds four precomputed vectors — `incoming_columns`,
  `outgoing_columns`, and the parallel `render_coeff_indices` / `render_columns`
  render pair — each built by delegating to `StateLayout`. One instance per cut
  pool; used only for cut storage sizing, dual extraction, DCS scoring, and
  cut-row rendering.

Three things make the pairing hard to read:

1. **Naming.** `StateLayout` vs `CutStateLayout` look like two flavours of one
   abstraction. They are not: one is the LP-driving descriptor, the other is a
   narrow masked projection of it.
2. **Containment / derivation is invisible.** `CutStateLayout::new(global:
&StateLayout, …)` derives every column from the global layout and keeps no
   back-reference. After construction the projection is a standalone struct of
   four `Vec<usize>`; nothing in its type signals "this was projected from a
   `StateLayout` and is meaningless without one". The structural tie exists only
   at the constructor boundary.
3. **Role overlap in the method surface.** Both types expose
   `state_to_lp_incoming_column(j)` and an outgoing resolver
   (`StateLayout::lp_column_for_state` / `CutStateLayout::state_to_lp_outgoing_column`)
   with the same names and shapes, so they read as interchangeable when they are
   not — `StateLayout`'s `j` indexes the full `[0, n_state)`, `CutStateLayout`'s
   `j` indexes the enabled subset.

### Load-bearing contracts every direction must preserve

These hold today and are the proof obligations for any refactor (each is covered
by a named test in `cut_state_layout.rs` or the D-case parity suite):

- **Default-identity.** An all-enabled projection (`storage: true, inflow_lags:
true`) is index-for-index equal to the global layout:
  `n_state() == global.n_state`, `state_to_lp_incoming_column(j)` agrees for every
  `j`, and `render_pairs()` reproduces `global.nonzero_state_indices` exactly.
  Tests `default_projection_is_identity`, `default_render_matches_global_nonzero_mask`.
- **Delegation (no drift-copy).** The projection never reimplements the column
  arithmetic; it delegates every column to
  `StateLayout::state_to_lp_incoming_column` (incoming) and
  `StateLayout::lp_column_for_state` (outgoing). The resolver arithmetic is
  defined **only** in `state_layout.rs`.
- **Incoming vs outgoing resolver semantics.** Dual extraction reads the
  **incoming** column (`state_to_lp_incoming_column`); cut-row rendering and DCS
  scoring read the **outgoing** column. The two differ for storage and lags
  (sddp.md "Benders cut sign & subgradient extraction" / "State pinning uses
  column bounds"). The incoming/outgoing distinction is itself a correctness
  contract, not a naming nicety.
- **Hot-path precompute.** Extraction, rendering, and DCS scoring read
  precomputed `Vec<usize>` columns with a single indexed access (or a zipped
  iterator for render pairs). No direction may regress these reads to per-call
  resolver arithmetic or hot-path allocation. The render pair must stay a
  parallel-vector zip, not a per-call filter over `nonzero_state_indices`.
- **Hash-neutrality.** Pure structural/naming change: every D-case parity hash and
  pinned lower bound is unchanged, including the reduced-config case.

### Hot-path inventory (what the precompute feeds)

| Consumer (file)                       | Reads                            | Frequency                             |
| ------------------------------------- | -------------------------------- | ------------------------------------- |
| `backward/duals_extraction.rs`        | `incoming_columns[j]` per `j`    | every backward LP solve               |
| `cut/row.rs` (`push_cut_row` etc.)    | `render_pairs()`, `render_len()` | every cut-row rebake (per active cut) |
| `training/forward/delta_cut_batch.rs` | `render_pairs()`, `render_len()` | every forward delta-cut batch         |
| `cut/dcs.rs` (`score`)                | `outgoing_columns[j]` per `j`    | every DCS candidate-scoring pass      |
| `training/lower_bound.rs`             | `n_state()`, resolvers           | every lower-bound evaluation          |

The precompute is the reason `CutStateLayout` exists as a materialized struct
rather than an on-the-fly computation. Any redesign that borrows or unifies must
keep these as flat indexed reads.

## Candidate directions

Each is evaluated against the actual post-019 code, with concrete trade-offs.
Blast radius is split into **mechanical** (compiler-guided rename/signature
churn the compiler finds for you) and **judgement** (sites needing human review).

### (A) Role-clarifying rename

Keep both types and both precomputed-vector layouts exactly as they are; rename so
the names signal full-LP-descriptor vs per-stage-projection. Concretely: rename
`CutStateLayout` → `CutStateProjection` (a `…Projection` suffix), optionally rename
the file to `cut_state_projection.rs`, and update the re-export and module docs.
`StateLayout` keeps its name (it is the base descriptor the projection is "of").
The method surface is unchanged.

- **Naming clarity.** Good. `…Projection` directly names the role and reads as
  "a projection of the state layout", killing the sibling-variants misread (#1).
  The two `state_to_lp_incoming_column` methods still share a name, so role
  overlap (#3) is only partially addressed — but the type name now disambiguates
  which one a call site holds.
- **Containment legibility.** Unchanged. The derivation is still only visible at
  the `::new(global, …)` boundary; the projection still keeps no back-reference
  (#2 not addressed). The name hints at the relationship without enforcing it.
- **Hot-path cost.** Zero. Byte-identical struct, methods, and precompute.
- **Blast radius.** Lowest. Mechanical: ~15 production files + ~9 test/bench sites
  rename the type token; the re-export in `indexer/mod.rs`; the field names
  `cut_state_layouts` (optional). Judgement: near-zero — a rename the compiler
  fully verifies. The doc-vocabulary update to `sddp.md` + indexer module docs is
  the only non-mechanical edit.

### (B) Explicit lightweight view

Replace the owned-vectors struct with a borrowing view `CutStateView<'a>` that
holds `&'a StateLayout` + the enabled-dimension mask, making the containment
explicit in the type (#2 addressed directly). The resolver methods compose the
borrow with the mask.

The critical constraint: the current type precomputes four `Vec<usize>` read on
the hot path. A naive view that resolves `state_to_lp_incoming_column(j)` per call
(walk the mask to find the `j`-th enabled global index, then call the global
resolver) **regresses the hot path** from one indexed read to a mask walk plus an
arithmetic resolve — unacceptable per the precompute contract. To keep the
precompute, the view must still **own** the four vectors, so it becomes
`StateLayout` borrow + mask + the same four `Vec<usize>` — i.e. the current struct
plus a lifetime and a redundant back-reference.

- **Naming clarity.** Good — `…View` signals a borrowed projection.
- **Containment legibility.** Best of the four for the borrow itself: the `&'a
StateLayout` field makes "this is a view of that layout" explicit and
  compiler-enforced. But see the cost below — to satisfy the hot path the view
  ends up carrying the precomputed vectors anyway, so the borrow is mostly
  documentation rather than the thing doing the work.
- **Hot-path cost.** Zero **only if** the precomputed vectors are retained
  (mandatory). If retained, identical reads to today. The trap the ticket calls
  out is real: a borrow-and-resolve view without the vectors regresses every
  hot-path consumer.
- **Blast radius.** Highest. A lifetime parameter on the projection type infects
  every holder: `StageData.cut_state_layouts: Vec<CutStateLayout>` →
  `Vec<CutStateView<'a>>` forces a lifetime onto `StageData`, `TrainingContext`,
  the simulation context, the session slice, and the workspace context — a
  lifetime that currently does not exist on those owners. The 5 external test
  crates and 2 benches that build the vector independently each grow a borrow they
  must keep alive against the `StateLayout`. Mechanical churn is large; judgement
  churn is real (every storage site that currently owns the projection now borrows,
  changing drop order and ownership). This is the only direction that changes
  ownership rather than just names.

### (C) Unify into one type + mask

Remove `CutStateLayout` entirely. Add an enabled-dimension mask to `StateLayout`
(all-enabled by default) and a `project(mask)` method that returns the per-pool
projected resolvers / render pairs. The cut projection becomes
`layout.project(config)`.

- **Naming clarity.** Removes the confusing pair by deleting one name. But it
  **merges the two roles into one type** — `StateLayout` would then carry both the
  LP-driving descriptor concern and the cut-projection concern, the opposite of
  the single-responsibility split the current two types encode. The infra-rule
  spirit (one type, one role) argues against this.
- **Containment legibility.** N/A — there is no containment once unified. But the
  per-pool projected data still has to live somewhere: `project()` must return a
  materialized object holding the four precomputed vectors (to honour the hot
  path), so in practice this re-introduces a projection struct as the return type
  — you have not actually removed the second type, only renamed it to
  `StateLayout::Projection` and moved its constructor onto `StateLayout`.
- **Hot-path cost.** Zero if `project()` returns a precomputed-vector struct;
  regresses if it returns a lazy view (same trap as B).
- **Blast radius.** Medium-high and **conflict-prone**. Every `StateLayout` call
  site (~25 files, far more than touch `CutStateLayout`) now faces a type that
  grew a mask field and a projection role. More importantly, this is the direction
  most likely to **collide with the StageIndexer/StageLayout unification** the
  ticket says to stay clear of: loading more responsibility onto `StateLayout`
  works against any future move to collapse the geometry carriers. Higher risk for
  the same behavioural outcome.

### (D) Factor a column-arithmetic owner from the dimension-set

Split the column arithmetic (the incoming/outgoing resolvers + the
`state_to_lp_column_map` cache) into a `StateColumns` owner, and make both the full
layout and the cut projection _dimension-set descriptors_ that reference the same
owner with different enabled-index sets. "One contains the other" becomes "both
reference one arithmetic owner".

- **Naming clarity.** Good in principle — names the real shared concept
  (column arithmetic) and the real difference (which dimensions are enabled). But
  it introduces a **third** type into a pair the ticket wants to make _simpler_,
  and the new `StateColumns` boundary is itself a new thing readers must learn.
- **Containment legibility.** Principled: the duplication-free relationship is
  explicit and symmetric. Strongest conceptual model of the four.
- **Hot-path cost.** Zero if the cut projection still materializes its own
  precomputed vectors (it must). The shared owner serves construction; the hot
  path still reads the projection's own flat vectors.
- **Blast radius.** High and **squarely in the deferred zone.** Factoring a
  column-arithmetic owner out of `StateLayout` restructures `StateLayout` itself —
  exactly the kind of geometry-carrier surgery the StageIndexer/StageLayout
  unification will want to own. Doing it here pre-empts that effort and risks a
  merge conflict / rework. The ticket explicitly defers this class of change.
  Best model, wrong time.

## Comparison

| Direction            | Naming               | Containment      | Hot path                   | Blast radius (mech / judgement)       | Conflict w/ deferred geometry work  |
| -------------------- | -------------------- | ---------------- | -------------------------- | ------------------------------------- | ----------------------------------- |
| **(A) Rename**       | Good                 | Hinted           | Zero                       | Low / ~none                           | None                                |
| **(B) View**         | Good                 | Best (borrow)    | Zero _iff_ vectors kept    | High / real (lifetime infects owners) | Low                                 |
| **(C) Unify+mask**   | Mixed (merges roles) | N/A              | Zero _iff_ struct returned | Med-high / real                       | **High** (loads StateLayout)        |
| **(D) Factor owner** | Good                 | Best (symmetric) | Zero _iff_ vectors kept    | High / real                           | **High** (restructures StateLayout) |

Two findings dominate:

1. **The precompute is non-negotiable and it pins the shape.** Every direction
   that claims to remove the projection struct (B's lazy form, C, D) discovers it
   must re-materialize the same four `Vec<usize>` to honour the hot path. So none
   of B/C/D actually eliminates the projection object — they relocate or rename it
   while paying extra structural cost (a lifetime, a merged role, or a third type).
2. **B, C, and D each restructure ownership or `StateLayout` itself**, and C and D
   reach directly into the geometry-carrier surgery the ticket defers to the
   StageIndexer/StageLayout effort.

## Recommendation

**(A) Role-clarifying rename, with a documentation tightening of the
containment.** Rename `CutStateLayout` → `CutStateProjection` (file →
`cut_state_projection.rs`), update the re-export and the indexer/`sddp.md`
vocabulary, and — the one addition beyond a bare rename — state the derivation
contract explicitly in the projection's type doc and its `::new` doc: "a
projection **of** a `StateLayout`; meaningless without the global layout it was
built from; delegates all column arithmetic to it." This pays down naming (#1)
and the role-overlap misread (#3 via the disambiguating type name), and makes the
containment (#2) legible in prose where it cannot yet be legible in the type.

Key reason: **the precompute contract means the projection object cannot actually
be eliminated**, so B/C/D buy a structurally heavier design (a viral lifetime, a
merged-role `StateLayout`, or a third type) for the _same_ runtime behaviour and
the _same_ materialized vectors — while C and D additionally collide with the
deferred geometry-carrier unification. (A) achieves the ticket's stated goal
(make the relationship legible without changing behavior) at the lowest blast
radius, with the strongest hash-neutrality guarantee (the compiler verifies a
rename), and with zero entanglement with the deferred work.

If the owner wants the containment **enforced** by the type rather than only
documented, the recommended synthesis is **A + the back-reference half of D's
idea, without the third type**: keep one projection type, rename it as in (A),
and add a non-owning `&'a StateLayout` (or an `Arc<StateLayout>` if the lifetime
proves viral) field used _only at construction and for debug assertions_, leaving
the precomputed vectors as the hot-path reads. This makes "contains the other"
explicit and compiler-checked while preserving the precompute and avoiding the
ownership upheaval of full (B). Note the lifetime/`Arc` cost is the same viral-
ownership risk that makes full (B) expensive, so adopt this only if enforced
containment is judged worth that cost; otherwise plain (A) is sufficient.

Full (B), (C), and (D) are not recommended now: (C) and (D) restructure
`StateLayout` and pre-empt the StageIndexer/StageLayout unification the ticket
defers; (B) infects every projection-holder (`StageData`, `TrainingContext`, the
simulation/session/workspace contexts, 5 external test crates, 2 benches) with a
lifetime for a containment win that documentation in (A) captures at a fraction of
the cost.

## Decision

Approved direction: **(A) role-clarifying rename + prose containment contract.**
`CutStateLayout` is renamed to `CutStateProjection` (file
`cut_state_layout.rs` → `cut_state_projection.rs`, module and re-export renamed to
match), and the containment is stated explicitly in the renamed type's doc and its
`new` constructor: the projection is _of_ a `StateLayout`, delegates all column
arithmetic to the global layout, and carries only an enabled-dimension mask plus
precomputed column vectors. The existing default-identity contract is retained.

(B), (C), and (D) are rejected. (B)'s borrowing view is viral for no runtime gain:
the hot-path precompute makes a materialized projection struct unavoidable, so the
`&StateLayout` borrow would carry the same four precomputed vectors while infecting
every projection-holder with a lifetime. (C) and (D) restructure `StateLayout`
itself and collide with the deferred unification of the per-stage geometry carriers
(`StageIndexer` / `StageLayout` / `StageEquipmentGeometry`), which owns that
surgery.

The refactor is a pure rename plus doc edits: no column mapping, cut value, or
Benders sign changes, and every parity hash and pinned lower bound is unchanged.
