# State-Space Views — one owner, typed total views

**Status**: design proposal, awaiting owner review. No implementation has
started; every claim about current code was verified against the live tree on
the date of writing.

**Discipline**: hash-neutral. The proposal changes no LP byte, no output byte,
and no hot-path allocation profile. Every migration step must hold the golden
parity suite byte-identical on both solver backends.

---

## 1. Problem

The stage-invariant state vector has one geometry but **four index spaces**:

1. the state-vector dimension `j ∈ [0, n_state)` (`StateDim`),
2. its **incoming** LP column (the pinned blocks; `InCol`),
3. its **outgoing** LP column (identity for storage/buckets/ring, lag remap
   through `z_inflow`; `OutCol`),
4. the **cut-slot** index — the enabled-subset reindexing that
   `CutStateProjection` exposes when a stage's `StageStateConfig` disables
   storage or lag dimensions. Stored cut `coefficients` slices, dual-extraction
   buffers, and the policy-export manifest are all indexed in this space.

Spaces 1–3 are typed and their resolvers live on `StateLayout` — the walks are
compile-safe and the region boundaries have a single owner
(`REGION_ORDER` + `StateLayout::state_dim_range`). But the residue below still
holds only by convention:

- **The cut-slot space is a type pun on `StateDim`.**
  `CutStateProjection::state_to_lp_incoming_column(j: StateDim)` and
  `state_to_lp_outgoing_column(j: StateDim)` take a _reduced_ slot index, not a
  global dimension. Call sites (`extract_duals_from_view`,
  `build_stage_entity_manifest`, DCS scoring) wrap reduced loop counters in
  `StateDim::new(j)`. Under a non-identity projection the two spaces diverge,
  and nothing at the type level stops a global dimension from reaching a
  slot-indexed accessor or vice versa — exactly the confusion class the typed
  vocabulary exists to make uncompilable, still open on this one axis.
- **Region classification is duplicated.** The `REGION_ORDER` scan with its
  `unwrap_or(StateRegion::Anticipated)` fallback appears verbatim in both
  `state_to_lp_column` and `state_to_lp_incoming_column`.
- **Per-region facts have no single queryable owner.** What a region _does_ is
  spread as match arms across four functions: its outgoing mapping (identity
  vs lag remap) in `state_to_lp_column`, its incoming pinned-block base in
  `state_to_lp_incoming_column`, its padding rule in `set_nonzero_mask`, and
  its cut gating (config-gated vs always-included) in
  `CutStateProjection::new`. Each site is individually single-sourced, but
  answering "what is the buckets region's contract?" requires reading four
  functions — the role-clarity gap that motivated this redesign.
- **Raw-`usize` collection tail.** `state_to_lp_column_map: Vec<usize>`
  (element type is semantically `OutCol`), `nonzero_state_indices: Vec<usize>`
  (semantically `StateDim`), and the projection's `render_coeff_indices:
Vec<usize>` (semantically cut slots) carry untyped elements.
- **Agreement is asserted, not derived.** The default-identity contract (an
  all-enabled projection reproduces the global resolvers index-for-index) and
  the projection-length/`n_state` relationship are `debug_assert`s plus
  fixture tests, not structural properties or property tests.
- **Role mixing.** `StateLayout` also hosts anticipated-decision _temporal
  gating_ (`is_anticipated_decision_active`,
  `is_anticipated_decision_active_for_delivery`,
  `anticipated_resolution_for`) — commissioning/horizon logic, not column
  geometry. The type reads as "geometry plus whatever needed its fields."

`CutStateProjection` itself is healthy: its constructor already delegates all
column arithmetic to the global resolvers and its render subset to the global
nonzero mask. The problem is not duplication of arithmetic — it is the untyped
fourth space, the duplicated classification, the scattered region facts, and
the by-assertion agreement.

## 2. Constraints (inherited, binding)

1. **Bit-determinism** — declaration-order invariance and run-to-run
   reproducibility, byte-for-byte, on both backends.
2. **Hot-path precompute** — per-solve paths read precomputed maps; no
   per-call resolution, no allocation. The materialized projection vectors and
   the flat column map stay.
3. **Named-site auditability** — correctness contracts are pinned to named
   code sites (`.claude/rules/sddp.md` names `state_to_lp_incoming_column`'s
   bucket arm, the column-bound pinning discipline, the ring's identity
   resolution). A design that makes coefficient/column resolution generic
   destroys the audit trail; this constraint has previously disqualified
   descriptor-table architectures for the equipment region and applies with
   equal force here.
4. **Zero-cost types** — `#[repr(transparent)]`, `Copy`, arithmetic-free
   newtypes; formulas stay with their owning value types.

## 3. Proposed design — three moves

### Move 1 — `CutSlot`: type the fourth space

A `CutSlot` newtype (same shape as the existing vocabulary:
`#[repr(transparent)]`, `Copy`, `new`/`get`, no arithmetic) becomes the index
type of the enabled-subset space:

```rust
/// Index into a stage's enabled cut-state subset (the space stored cut
/// `coefficients` are indexed in). Distinct from `StateDim`, the global
/// state-vector dimension: under a non-identity projection the two diverge.
#[repr(transparent)]
pub struct CutSlot(usize);

impl CutStateProjection {
    pub fn n_slots(&self) -> usize;
    pub fn incoming_column(&self, s: CutSlot) -> InCol;
    pub fn outgoing_column(&self, s: CutSlot) -> OutCol;
    /// Render pairs become (CutSlot, OutCol).
    pub fn render_pairs(&self) -> impl ExactSizeIterator<Item = (CutSlot, OutCol)> + '_;
}
```

Consequences:

- `extract_duals_from_view`, cut-row rendering, DCS scoring, and
  `build_stage_entity_manifest` iterate `CutSlot`s; handing a global
  `StateDim` to a projection accessor (or a `CutSlot` to a `StateLayout`
  resolver) fails to compile. Two `compile_fail` doctest pins seal the axis.
- The cut pool's `state_dimension` is documented as the **cut-slot-space**
  dimension — which is also the space `validate_policy_load`'s
  `state_dimension` equality check operates in. The pun's resolution makes
  that contract statement precise for free.
- The projection's internal vectors become `Vec<InCol>` / `Vec<OutCol>` /
  `Vec<CutSlot>` (they are already typed on two of four).

The accessor renames (`state_to_lp_incoming_column` →
`incoming_column`) are deliberate: after the flip, the old names would claim a
`StateDim` domain they no longer have. Renaming at the same time as the type
flip keeps the compiler enumerating every call site exactly once.

### Move 2 — one classifier, one owner per region fact

**Classification.** A single `classify` owns the region scan both resolvers
currently duplicate:

```rust
impl StateLayout {
    /// Classify a global dimension into (region, offset-within-region).
    /// Total on [0, n_state): REGION_ORDER's ranges partition it contiguously.
    pub(crate) fn classify(&self, j: StateDim) -> (StateRegion, usize);
}
```

`state_to_lp_column` and `state_to_lp_incoming_column` both consume it; the
`unwrap_or(Anticipated)` fallback exists once, in the one function whose doc
states the partition-coverage argument.

**Per-region facts.** Extend the `state_dim_range` precedent — an exhaustive
match per fact, each fact owned once — to the remaining three conventions:

```rust
impl StateLayout {
    /// The incoming pinned block's start column for a region — the single
    /// owner of the "which block does pinning write" fact.
    /// Storage → storage_in, Lag → inflow_lags, Buckets → transit_buckets_in,
    /// Anticipated → anticipated_state.
    pub(crate) fn incoming_block_start(&self, region: StateRegion) -> usize;
}

impl StateRegion {
    /// Whether a stage's config enables this region's cut-state dimensions.
    /// Storage/Lag are config-gated; Buckets/Anticipated are always included.
    pub(crate) fn cut_enabled(self, config: StageStateConfig) -> bool;
}
```

The outgoing mapping (identity vs lag remap) and the padding rule (none vs
per-entity active counts) remain where they are — each already lives in
exactly one match (`state_to_lp_column`, `set_nonzero_mask`) — but after this
move every region fact is answerable from a named single-owner site, and a new
`StateRegion` variant fails to compile at each fact it has not declared.
Resolvers stay hand-written exhaustive matches; there is no descriptor table
(see §5).

The two `.claude/rules/sddp.md` contracts that today live as resolver arms
become structural corollaries: "buckets resolve through an explicit arm, never
the anticipated catch-all" is now "`incoming_block_start(Buckets)` is a match
arm with no `_`", and the catch-all itself survives only inside `classify`'s
single documented totality argument.

**Derived agreement.** The default-identity contract and the partition
property move from fixture assertions to property tests (small-parameter
`proptest` over hydro count, lag order, bucket count, anticipated
plants/leads, and both gate flags), asserting:

- `REGION_ORDER` ranges partition `[0, n_state)` contiguously;
- an all-enabled projection reproduces both global resolvers index-for-index
  and the global render exactly;
- every gated projection's slots map to the same columns the global resolvers
  produce for the corresponding surviving dimensions.

These run in the default test pass (pure arithmetic, no solver).

### Move 3 — role separation

- The anticipated-decision gating helpers
  (`is_anticipated_decision_active`,
  `is_anticipated_decision_active_for_delivery`,
  `anticipated_resolution_for`) move out of the geometry impl into a sibling
  module (`lp/indexer/anticipated_gate.rs` or equivalent), keeping their
  current signatures and data access. They are temporal gating over
  commissioning windows and horizon bounds; co-locating them with column
  arithmetic is what makes `StateLayout` read as a god-type.
- The typed collection tail flips element types where the pub surface allows:
  `state_to_lp_column_map: Vec<OutCol>` (readers via `lp_column_for_state`
  unchanged), `nonzero_state_indices: Vec<StateDim>` — compiler-enumerated
  consumer flips, no formula changes.
- Optionally (owner decision, §6): rename `StateLayout` → `StateSpace` as a
  terminal cosmetic step, once it is the single owner of classification,
  region facts, and views — the name then describes the role.

## 4. What deliberately does not change

- **The LP column layout** — every range, every offset, every byte.
- **The materialized hot-path caches** — the projection's per-stage vectors
  and the flat `state_to_lp_column_map` stay precomputed; views cache, they do
  not re-resolve.
- **`CutStateProjection` remains a per-stage-config materialized projection.**
  A lazy "identity view" variant that branches at read time was considered and
  rejected: it trades a structural guarantee for a hot-path branch and a
  second code shape, and the property tests pin identity more cheaply.
- **The `pub` range fields** (`storage`, `inflow_lags`, …) — consumers
  (patching, extraction, LP fill) read them as today.
- **The equipment region** — `StageLayout`/`StageGeometry` and the value-type
  formula owners are out of scope; this proposal is confined to the state
  region and its cut projection.

## 5. Rejected shapes

- **A region descriptor table** (per-region struct of closures/offsets driving
  generic folds in construction, resolution, and rendering). Rejected on
  named-site auditability: generic iteration is what makes a pinned resolver
  arm unauditable, the same ground on which the equipment-region descriptor
  schema was measured and declined. The design keeps hand-written exhaustive
  matches and single-sources the _facts_ they consume, not the control flow.
- **Unifying `CutStateProjection` into `StateLayout`** (one type, mask
  argument on every resolver). Rejected: the projection is per-stage state
  while the layout is study-global; merging couples their lifetimes and puts
  a config parameter on every hot-path call — a worse API than two types with
  one typed seam.
- **A lazy identity-view enum** (`Projection::Identity | Subset`). Rejected —
  see §4.

## 6. Open decisions for the owner

1. **Terminal rename** `StateLayout` → `StateSpace`: adopt (mechanical,
   compiler-enumerated, done last) or keep the current name. Recommendation:
   adopt — after the moves the type genuinely is the state space's owner, and
   the rename separates it from the equipment-region `StageLayout` family,
   with which it shares an easily-confused name today.
2. **Gating-helper destination**: sibling module in `lp/indexer` (minimal,
   recommended) vs relocation toward `lead_time` (heavier: the helpers read
   layout fields; would need data threading for no behavioral gain).
3. **Reverse view**: whether the projection exposes
   `global_dim(s: CutSlot) -> StateDim`. Decide at implementation from real
   consumer need; do not add speculatively.
4. **Accessor rename scope**: flip names together with types (recommended;
   one compiler-enumerated pass) vs keep old names on new types.

## 7. Migration outline

Additive-then-seal, every step gated on the full golden parity suite (both
backends), the crate test suite, and the doctest pins; a one-time bench spot
check where the projection sits on the extraction path:

1. `CutSlot` vocabulary + projection accessor flip (the one breaking-internal
   step; compiler enumerates all consumers) + two `compile_fail` pins.
2. `classify` extraction; both resolvers consume it (byte-identical by
   construction — same scan, one home).
3. Region fact owners (`incoming_block_start`, `cut_enabled`); the incoming
   resolver and the projection constructor consume them.
4. Property tests (partition, default-identity, gated-projection agreement).
5. Typed collection tail + gating-helper relocation.
6. Optional terminal rename (owner decision 1) + rustdoc/`sddp.md` pointer
   refresh.

Steps are small enough to be individual tickets; none depends on a solver
change; each leaves the tree releasable.

## 8. Success criteria

- Wrapping a reduced slot counter in `StateDim` at a projection accessor —
  today's live pattern at the dual-extraction, render, and manifest sites —
  **fails to compile**, and the two new pins prove it stays that way.
- The `REGION_ORDER` scan exists in exactly one function (`classify`);
  grep-verifiable.
- Each per-region fact (dimension range, incoming block, outgoing mapping,
  padding rule, cut gating) is answerable from one named site; a new
  `StateRegion` variant fails to compile until every fact declares it.
- Property tests pin partition coverage and projection agreement across the
  parameter space, in the default test pass.
- Every golden parity hash byte-identical on both backends; benches flat.
- `StateLayout` (or `StateSpace`) contains no temporal gating logic.
