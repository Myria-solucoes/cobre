# Reserved Seams and Deferred Debt

> **Status:** Living register — tracks shipped state (what is reserved vs deferred today). Every entry is self-guarding; re-derive against the live tree before acting.

This document tracks two related but distinct things about the workspace's
non-obvious inert surfaces:

- The **reserved-seam register**: config fields, struct fields, and functions
  that are loaded, validated, and/or compiled, but have no production consumer
  today — each entered here with an **owner** (the subsystem responsible for
  wiring or retiring it) and a **consuming milestone** (the concrete condition
  that activates it). A candidate that cannot be given both is not registered;
  it is dead code to remove, not a seam to reserve. This operationalizes the
  project's "unwired config is reserved, not dead" rule — a claim is only
  trustworthy here if it is checkable, not merely asserted.
- The **deferred-debt register**: a separate, broader class of architectural
  debt (not limited to unwired seams) tracked alongside this one.

For known **structural/algorithmic** limitations of the node-graph engine
(as opposed to unwired config or code) — single-initial-node,
single-boundary-policy terminal cost (a per-leaf terminal future-cost function
needs multi-policy input), enumerated-selection restrictions — see
[`policy-graph-limitations.md`](policy-graph-limitations.md); this document
does not restate those.

A seam leaves this register the moment it is wired (delete the entry, wire the
reader) or the moment its milestone is ruled out (delete the seam itself, since
an unreachable milestone converts a reserved seam into dead code).

## Reserved-seam register

### Water travel-time in-transit bucket topology (`StudySetup.transit_bucket_topology`)

**What it is.** `crates/cobre-sddp/src/setup/mod.rs` derives one
`TransitBucketTopology` per study (`bucket_topology::build_transit_bucket_topology`)
and threads it into state-layout sizing, the LP builder's arc-table wiring, and
the bucket initial-condition seed — all as a **local** variable consumed during
`StudySetup` construction. The same value is then also stored on
`StudySetup.transit_bucket_topology` (`#[allow(dead_code)]`), but no code reads
the field back off `StudySetup` after construction; every real consumer already
received the value via the local, one derivation, no second call.

**Owner.** The water travel-time feature (`crates/cobre-sddp/src/setup/bucket_topology.rs`,
`crates/cobre-sddp/src/lp/builder/{template,entries}.rs`; see the "Water travel
time" contracts in `.claude/rules/sddp.md`).

**Consuming milestone.** The first caller that needs the topology after
`StudySetup` has already been built — for example a resume/warm-start path that
re-validates topology consistency against a loaded checkpoint, or an
output/diagnostic surface reporting the resolved bucket topology without
re-deriving it. The `#[allow(dead_code)]` on the field re-fires the moment such
a reader lands, which is the field's own activation signal.

### `LipschitzConfig.mode` and its enclosing `UpperBoundEvaluationConfig`

**What it is.** `crates/cobre-io/src/config/training.rs` declares
`UpperBoundEvaluationConfig` (`enabled`, `initial_iteration`,
`interval_iterations`, and a nested `LipschitzConfig` with `mode`,
`fallback_value`, `scale_factor`) for a vertex-based inner-approximation upper
bound. The whole struct is loaded, schema-exported, and round-trips through
`config.json → upper_bound_evaluation`, but no field of it — not `mode`, not
`enabled` — is read anywhere in `crates/cobre-sddp`. This is a config-only stub
for a feature that has not been implemented, not merely one inert field inside
an otherwise-wired struct.

**Owner.** The vertex-based upper-bound-evaluation feature — currently a
config-surface stub with no solver-side implementation.

**Consuming milestone.** The architecture-and-debt-register audit that decides
whether to implement vertex-based inner approximation (wiring `enabled` /
`initial_iteration` / `interval_iterations` / `lipschitz.*` into the upper-bound
estimator) or retire the config surface as never-shipped. Until that
disposition is made, the fields stay reserved rather than removed, per the
"unwired config is reserved, not dead" rule — but this is the one entry in this
register whose milestone is a decision, not an implementation trigger, so it
is a stronger candidate for removal than the others here if that decision goes
against wiring it.

## Verified NOT reserved

`historical_years` (`cobre_core::scenario::ScenarioSource`,
`crates/cobre-core/src/model/scenario.rs`) is consumed:
`crates/cobre-sddp/src/setup/{mod.rs,stochastic_pipeline.rs}` thread
`training_source.historical_years.as_ref()` / `simulation_source.historical_years.as_ref()`
into `discover_historical_windows`
(`crates/cobre-stochastic/src/sampling/window.rs`), which filters candidate
history years by the user pool, and the same `Option<&HistoricalYears>` is read
directly by `crates/cobre-stochastic/src/sampling/historical.rs`'s window
validation. It is deliberately absent from the register above; a prior
interface-review note calling it inert predates this wiring.

The hydro `storage_violation_below_cost` and `filling_target_violation_cost`
penalties (`HydroPenalties`, `crates/cobre-core/src/model/resolved/penalties.rs`)
are consumed for **filling-phase** hydros: `fill_filling_target_columns`
(`crates/cobre-sddp/src/lp/builder/columns.rs`) writes
`filling_target_violation_cost` as the objective coefficient on the `σ_fill`
target-shortfall slack at every Filling stage, and
`fill_filled_min_storage_floor_columns` writes `storage_violation_below_cost` on
the `σ^{v-}` operating-floor slack at every Operating stage of a filling hydro;
the paired soft `≥` rows live in `lp/builder/rows.rs`, and the penalty ordering
is validated (a filling-target penalty must exceed the max deficit cost). They
are unconsumed only for ordinary non-filling hydros, whose storage bounds are
HARD so no slack column exists — the case a prior note over-generalized to
"always 0". Consumed, not reserved: deliberately absent from the register above.

## Post-migration architecture audit

A judgment reading of the node-native engine after the graph, node axis, and
traversal machinery landed. It answers three questions: where each new
structure lives, whether each new seam has exactly one owner, and whether any
`stage`-keyed structure is residue where a node/pool key belongs. It is a
snapshot of the reading, not a mechanical gate — the mechanical checks live in
the "Audit-evidence" section below; re-derive both against the live tree rather
than trusting the prose as a frozen state.

### Module map

- **Graph type.** `NodeGraph` (one `NodeRuntime` per node) lives in
  `crates/cobre-sddp/src/setup/node_graph.rs`, alongside the typed node axis
  (`NodePos`, the dense position; `NodeId`, the declared id) and the enumerated
  walk plan (`EnumeratedPlan`).
- **Single construction dispatcher.** `build_node_graph` is the one entry point;
  it chooses `build_chain_node_graph` for an empty declared `nodes[]` and
  `build_declared_node_graph` otherwise, both producing the same representation
  consumed uniformly downstream.
- **Frontier walk / reverse-topological sweep.** `NodeGraph::stage_frontier`,
  `frontier_node`, and `any_stage_node` resolve a stage's nodes; the backward
  reverse-topological level partition is `NodeGraph::backward_cut_levels`. The
  forward root-to-leaf walk is `EnumeratedPlan::walk_path`.
- **`node → pool` map.** `NodeRuntime.pool_id`, assigned once during graph
  construction and read back via `NodeGraph::node_pool_ids`.
- **Visit records.** `VisitedStatesArchive` (one `NodeStates` per node) in
  `crates/cobre-sddp/src/training/visited_states.rs`.
- **Module homes still fit.** `training/` holds the passes (session, forward,
  backward, lower bound); `cut/` holds the cut substrate (pool, future-cost
  function, wire, sync, selection, dynamic selection, basis reconstruction);
  `workspace/` holds per-worker arenas and the captured basis. No module has
  accreted a node-work concern under a stale name: the graph machinery is homed
  in `setup/`, the traversal drivers in `training/forward` and
  `training/backward`, and each enumerated fork is a named sibling of its
  sampled counterpart, not an inline special case.

### One-owner checks over the new seams

Each seam below has exactly one construction site and one interpretation site;
no second site was found.

- **Node-graph home.** Constructed only through `build_node_graph`; interpreted
  uniformly — no chain-vs-tree shape predicate reaches training dispatch (the
  `is_chain` grep in the Audit-evidence section returns zero).
- **`node → pool` map.** `pool_id` is written once during graph build; read via
  `node_pool_ids` and consumed by `FutureCostFunction::new_per_pool` to size one
  pool per node.
- **Capacity function.** `pool_capacity` (`cut/fcf.rs`) has a single definition
  and a single caller (`FutureCostFunction::new_per_pool`).
- **Visit-record layout.** `VisitedStatesArchive` / `NodeStates` are owned
  solely by `training/visited_states.rs`.
- **Basis node tag.** `CapturedBasis.node_id` (a `NodeId`) is written once at
  capture and read once at apply (`reconstruct_basis`); a mismatch cold-starts
  rather than warm-starting, never a wrong answer.
- **Cut-wire header.** The cut record's version and per-record `node_id` are
  encoded and decoded only in `cut/wire.rs` (`CUT_WIRE_VERSION`); the
  captured-basis broadcast header (`BASIS_BROADCAST_WIRE_VERSION`) is encoded
  and decoded only by `CapturedBasis::to_broadcast_payload` /
  `try_from_broadcast_payload`. Each header has one encode and one decode owner.

### Stage-residue classification

- **Genuine per-stage (kept).** Stage LP templates (the LP structure is
  stage-shared because a node's state affects only the inflow realization, not
  the constraint structure), per-stage equipment geometry, per-stage cumulative
  discount factors, and the stage calendar / season cast. Each is correctly
  keyed by stage.
- **Residue (a stage index where a node/pool key belongs): none.** Cut pools are
  keyed by `pool_id`, not stage — `pool_stage[pool_id] == StageIdx(t)` holds only
  on a chain and diverges under branching. The chain reaches its values as the
  degenerate node graph (the empty-`nodes[]` path through `build_node_graph`),
  never via a shape fork; the architecture reads symmetric.

### Whole-lifecycle scope (beyond the node axis)

A full `cobre run` read (entrypoint → setup → policy → training → simulation →
outputs) found the largest structural debt outside the node axis, in the setup
config-projection layer and in the CLI/Python output orchestration. Both are
register items below, not node-axis findings, and neither is executed here:

- The `Config → StudyParams / ConstructionConfig / BroadcastConfig` projection
  and the CLI non-root stochastic-context reconstruction
  (`reconstruct_stochastic_context_non_root` /
  `rebuild_historical_library_non_root`), a rank-0 mirror that must stay
  bit-identical across a crate boundary.
- The output orchestration mirrored by hand across the CLI and Python crates.

## Deferred-debt register

Everything the plan deliberately did not finish. Each entry names an **owner**
(the role responsible for closing it) and a **trigger** (a durable condition
that makes the work worth doing — never a date or a release number). An item
that cannot be given both is not register-ready. Every symbol and path cited
resolves against the live tree.

### Enumerated interior-node outgoing-state exchange for multi-rank branching graphs

**What it is.** The enumerated forward populates a node's persisted outgoing
state only on the ranks whose assigned paths visit it, zero-filling every other
rank's slot; the replicated backward partitions a cut-generating node's
successor openings across all ranks regardless. This is sound only when every
cut-generating node lies on every root→leaf path (a deterministic trunk with a
terminal fan). A multi-rank enumerated run over a graph with interior branching
is therefore **hard-rejected before any solve**: `enumerated_requires_state_exchange`
(`setup/node_graph.rs`) returns the first offending interior node, and the
training-session constructor turns that into a validation error rather than
letting a rank cut against a zeroed incoming state. The deferred fix adds a
cross-rank exchange (or broadcast) of interior-node outgoing state so a rank
that never visits a node can still cut against its true incoming state.

**Owner.** The SDDP training-engine owner. A design decision is required first:
which rank's solve is canonical when a node is solved redundantly across ranks.

**Trigger.** Multi-rank enumerated training over interior-branching graphs is
required. Closing it also needs a real multi-rank MPI multi-level-branching
fixture that the in-process single-rank test harness cannot express.

### Enumerated backward observability gap

**What it is.** The enumerated backward driver (`run_enumerated_backward`)
returns an empty per-stage worker-statistics vector, so the solver-statistics
log carries no backward-phase rows under enumerated traversal. Cuts and bounds
are unaffected (correctness- and determinism-neutral).

**Owner.** The training owner.

**Trigger.** Enumerated-run backward telemetry / solver-profile output is
required.

### Predecessor-distinctness debug assertion

**What it is.** The parent lookup and parent-map construction carry a
debug-build assertion that a node has at most one predecessor; it is relaxed to
exclude sibling pairs, so it is vacuous at in-degree 1 (every node in a tree)
and bites only a future node reached by two or more edges.

**Owner.** The setup / SDDP owner.

**Trigger.** Recombination / dense-transition (DAG) graph work.

### Enumerated recombination seam

**What it is.** A node with in-degree ≥ 2 is hard-rejected under enumerated
selection at study setup (`reject_recombining_node_enumeration`), because the
engine reconstructs a node's incoming continuous state from a single-predecessor
map and would otherwise solve the join once under one arbitrarily chosen
parent's state. Full support needs per-prefix state reconstruction and a
prefix-based rewrite of the exact-bound path, with SDDP-specialist sign-off.
This seam and the single-predecessor restriction are documented in
[`policy-graph-limitations.md`](policy-graph-limitations.md).

**Owner.** The setup / SDDP owner.

**Trigger.** Recombining-graph support under exact enumeration.

### Oracle test-harness duplication

**What it is.** `crates/cobre-sddp/tests/extensive_form_oracle.rs` carries a
verbatim copy of the comparison harness in
`crates/cobre-sddp/tests/branching_value_oracle.rs` (the `close` tolerance
helper and its scaffolding) and pays a second static link of the solver,
against the test cost-discipline. Two resolution options for an owner pick:
promote the shared `close` / tolerance helpers into `cobre_sddp::test_support`,
or fold both fixtures into one test binary.

**Owner.** The test-infrastructure owner.

**Trigger.** The next consolidation of the branching oracle test suite.

### Cut-pool slot-index newtype

**What it is.** Cut-pool slot indices are a raw integer across many call sites
adjacent to the slot-identity basis-reconstruction hot path; a dedicated newtype
would make slot-vs-row confusion a compile error. Deferred because the change
touches many sites next to a protected hot path.

**Owner.** The cut-pool / training owner.

**Trigger.** The planned traversal-stride index rename lands in the same
neighborhood (a deliberate typed-index sweep of the cut-pool indices).

### Superseded cut-sync public methods

**What it is.** Three methods on `CutSyncBuffers` — `sync_cuts`,
`pack_local_records`, `sync_packed_records` — have no production call site (they
are the legacy single-pool exchange, superseded by the per-level batched
`sync_level_records`), yet remain re-exported public API on a published crate,
so removal is a breaking change. Their test suite gives false "still used"
confidence.

**Owner.** The comm / training owner.

**Trigger.** The next licensed public-API break.

### Python-binding Rust tests invisible to CI

**What it is.** The Python-binding crate is excluded from the workspace, and its
CI job runs only the Python build plus pytest, so the crate's Rust `#[cfg(test)]`
modules are never compiled or run in CI. Fix: either wire a `cargo check --tests`
for the crate, or hoist a shared node-graph test-fixture builder into
`cobre_sddp::test_support` so those Rust tests live in a CI-visible crate.

**Owner.** The build / CI owner.

**Trigger.** Systemic — the next CI-configuration pass (a Rust test regression in
that crate would otherwise ship unseen).

### Backend-scoped parity-roster caveat

**What it is.** The opening-order determinism check and three of the
wire-exchange parity gates are exact-comparison only under the bit-exact
backend and are skipped or relaxed on the other LP backend, but the parity
roster does not state this backend scoping. A one-line clarification belongs in
the roster document. It is registered rather than applied here: applying it
would touch a second file beyond this register.

**Owner.** The parity / test owner.

**Trigger.** The next parity-roster revision.

### Horizon-type config field long-term fate

**What it is.** The policy-graph horizon-type field (`graph_type` in
`crates/cobre-io/src/stages.rs`) accepts only the finite-horizon value; the
cyclic value parses but is rejected as reserved during conversion. Under the
node-native engine the graph declaration itself is the structure, so the field
is a deletion candidate.

**Owner.** The setup / config owner.

**Trigger.** The next licensed input-format break.

### Reserved and consumed knobs (snapshot correction)

**What it is.** `LipschitzConfig.mode` (`crates/cobre-io/src/config/training.rs`)
is a single-valued, unconsumed string inside the vertex-based
upper-bound-evaluation config; it is already carried in the reserved-seam
register above, never swept, and that register owns its disposition. In
contrast, `historical_years` is **not** inert — it is consumed by the window
and historical samplers (see "Verified NOT reserved" above), correcting the
stale interface-review note that called it inert.

**Owner.** The upper-bound-evaluation feature owner (for the reserved knob).

**Trigger.** The decision to implement or retire vertex-based inner
approximation.

### Boundary-policy source-node

**What it is.** The boundary-policy config (`BoundaryPolicy`,
`crates/cobre-io/src/config/policy.rs`) addresses its source by stage
(`source_stage`), has no source-node selector, shares one leaf pool
unconditionally, and rejects a multi-node source. It relocates once, under a
study-level boundary configuration.

**Owner.** The setup / config owner.

**Trigger.** That relocation (a study-level boundary redesign).

### Stage-configuration engine-scoping

**What it is.** The stage-configuration file carries several roles at once
(calendar, blocks, openings, seasons, risk, state-variable toggles,
discounting) and forces every future study kind to declare this algorithm's
uncertainty and risk axes. A first step is taken: the per-stage opening count
(`num_openings`) is conditionally required — required only for generated stages.
The full split — scoping the algorithm-specific axes out of the shared stage
file — is larger.

**Owner.** The setup / config owner.

**Trigger.** The full engine-scoping split of the stage file.

### Within-node weighted opening draw (chain-parity break)

**What it is.** Today a visited node draws one within-node opening uniformly over
its own opening sub-range. A future change replacing that uniform draw with an
inverse-CDF (probability-weighted) draw will select a different opening for the
same seed, breaking chain parity at that time. This numeric break is **separate**
from the input-format break already absorbed by the node-per-stage migration:
that migration's "one break" accounting does not cover it — the weighted-opening
draw introduces its own numeric break, stated here explicitly.

**Owner.** The SDDP / sampling owner.

**Trigger.** The weighted within-node opening feature landing.

### Anticipated-commitment per-block tension

**What it is.** The generic-constraint and bound validators forbid a block
argument on an anticipated thermal's commitment (an anticipated decision is a
stage-level scalar), which is in tension with per-block delivery. The tension is
parked.

**Owner.** The I/O constraints owner.

**Trigger.** Per-block delivery work on anticipated commitments.

### Stage-calendar crate home

**What it is.** The stage-calendar / season-cast machinery (`StageCalendar`,
`season_cast`) currently lives in the stochastic crate, an in-plan extension of
the existing season-cast module. A relocation to a temporal module in the core
crate is registered as a structural follow-up.

**Owner.** The stochastic / temporal owner.

**Trigger.** When the season-machinery relocation is worth its scope.

### External-noise take/fill glue duplication

**What it is.** The mem-take/fill buffer glue around `fill_external_opening_noise`
is duplicated across the frozen-path backward worker
(`training/backward/by_node.rs`) and the DCS-path backward worker
(`training/backward/by_scenario.rs`). Extracting a shared helper crosses the
frozen-hot-path / DCS boundary that the basis-reconstruction module protects, so
it is an architectural call, not a drive-by. The adjacent backward-buffer
pre-allocation duplication across the two schedulers folds with the same
refactor.

**Owner.** The training owner.

**Trigger.** A deliberate frozen-hot-path / DCS-boundary refactor.

### Branching-engine housekeeping follow-ups

**What it is.** Three small deferred items from the branching work, each with its
own owner and trigger:

- A checkpoint-write regression observed on a fan-structured branching study.
  **Owner.** The training owner. **Trigger.** Reproduced when checkpointing a
  fan-structured branching study.
- The two-rank captured-basis-broadcast round-trip test
  (`broadcast_basis_cache`) belongs in a dedicated wire-format test home rather
  than its current location. **Owner.** The training / test owner. **Trigger.**
  The next MPI wire-format test reorganization.
- A design note describing an order/count-based cut-exchange header no longer
  matches the shipped self-describing, per-record node-tagged cut format;
  reconcile the note to the shipped format. **Owner.** The doc / comment owner.
  **Trigger.** The next cut-exchange wire-format touch.

### SDDP.jl-compatibility limit: multi-state first stage

**What it is.** A structural limit is deferred here: a multi-state / reserved-root
first stage — an initial probability distribution over several first-stage nodes. It
is documented in [`policy-graph-limitations.md`](policy-graph-limitations.md) and
cross-referenced here rather than restated.

The companion terminal-side item once bundled here — a per-terminal-state
continuation value — is **not** a deferred limit. The shared terminal pool holds one
boundary future-cost function evaluated at each leaf's own ending state, which is the
correct representation for the supported single-boundary-policy input (see
[`policy-graph-limitations.md`](policy-graph-limitations.md)). A distinct future-cost
function per leaf would need per-leaf boundary input — more than one boundary
policy — and that multi-policy regime is tracked by the **Boundary-policy
source-node** reserved-seam entry above, not restated here.

**Owner.** The SDDP engine owner.

**Trigger.** A model that begins from an uncertain initial regime (the multi-state
first stage).

### Retired input spellings (recorded as retired, not deferred)

**What it is.** Four legacy spellings were removed outright with no compatibility
alias: the two backward-scheduler method spellings (`trial_point`,
`opening_block`); the root-level forward-pass / scenario-count selection
aliases; the legacy scenario-count spelling of the per-stage opening count; and
the reclaimed cut-record wire slot (an id-4 field once named for a domination
count, now reused for the intercept). These are recorded as retired, not awaiting
a trigger.

**Owner.** The setup / config owner.

**Guard (not a trigger).** Any config using a removed spelling fails with an
unknown-field / unknown-variant deserialize error, pinned by the reject tests in
`crates/cobre-io/src/config/{training,simulation}.rs` and the FlatBuffers schema
conformance test — the invariant that stands in for a version snapshot.

## Deferred-debt register — whole-lifecycle audit findings

Findings of the full `cobre run` lifecycle read, described by behavior. Each
names an owner and a trigger. The structural items are escalated (too large for
a behaviour-neutral consolidation); the byte-neutral consolidations were
executed in the consolidation work and are recorded here so a future audit does
not re-raise them.

### Structural — escalated (a dedicated follow-up redesign)

#### Setup config-projection sprawl + CLI non-root reconstruction

**What it is.** The run configuration is re-projected through three
near-isomorphic in-memory structs — an in-process params projection
(`StudyParams`), a construction-config projection (`ConstructionConfig`), and a
broadcast projection (`BroadcastConfig`) — kept in lockstep by hand, so one
config knob touches several sites; and the CLI crate hand-rolls a mirror of the
rank-0 stochastic pipeline for non-root ranks
(`reconstruct_stochastic_context_non_root` /
`rebuild_historical_library_non_root`) that must stay bit-identical across a
crate boundary. The setup god-struct (`StudySetup`) additionally fuses immutable
inputs, mutable run-config, and produced output. A silent MPI-vs-local
divergence is the expensive failure mode. Target: one postcard-safe resolved-run
config projection broadcast on the wire and consumed by both the local and
non-root paths; a domain-crate-owned stochastic-context builder parameterized by
reuse-vs-reread; and a lifecycle split of the god-struct into immutable inputs /
mutable run-params / produced output. This is the audit's only structural item
of the highest severity and the anchor of its lifecycle-modeling north star.

**Owner.** The architecture owner.

**Trigger.** A dedicated setup-layer redesign (the postcard tagged-enum
serialization boundary is solved once, there).

#### CLI/Python output orchestration hand-mirror

**What it is.** Every output file the CLI writes must also be written by the
Python bindings; the two write paths are hand-mirrored (which writers, in what
order, under what guards), held together only by the Python-parity rule plus
mirror comments, with no shared owner — confirmed across both training and
simulation outputs, including the census scenario-summary tuple-reshape copied
on both sides and pinned by a test that itself exists in both crates. Target:
hoist the shared "output set + guards" (the pattern the Python `*_if_any`
helpers already prove) into a crate both the CLI and Python depend on.

**Owner.** The architecture owner.

**Trigger.** The setup-layer redesign.

#### Untyped state-family primitive + colliding entity dictionaries

**What it is.** The policy state slot's `entity_type` is an untyped small integer
whose state-family dictionary lives downstream in the policy writer, while a
second, overlapping physical-output-entity dictionary
(`crates/cobre-io/src/output/dictionary.rs`) shares the same integer prefix with
divergent meanings — a grep hazard and a type-safety gap (no live bug; the two
dictionaries are used disjointly). Target: a typed state-family enum owned next
to the slot type, and disambiguated names for the two dictionaries.

**Owner.** The I/O owner.

**Trigger.** The next policy / output schema touch.

#### Cut-selection paradigm conflation

**What it is.** One cut-selection type (`CutSelectionStrategy`) fuses two
paradigms — periodic value-based selection and lazy per-solve dynamic selection —
but the dynamic variant does not honor the shared interface: it early-returns
empty from the value sweep, its real logic lives elsewhere, and the driver
carries an unreachable guard for it. Target: model the two paradigms as distinct
internal types unified only at the config boundary.

**Owner.** The training owner.

**Trigger.** A third selection paradigm, or the next cut-selection change.

#### MPI-ephemeral wire-version bytes on same-binary broadcast formats

**What it is.** Two broadcast formats — the cut wire (`CUT_WIRE_VERSION`) and the
captured-basis payload (`BASIS_BROADCAST_WIRE_VERSION`) — carry a disk-style
version byte on an all-ranks broadcast where every rank runs the same binary, so
a cross-version mismatch cannot occur in one run; the byte's real value is a
corruption tripwire, not a version negotiator. The simulation exchange
(version-free, length-prefixed all-gather) is the honest counter-example. A
third such format — a resolved-parameters broadcast envelope — previously fit
this pattern but was **removed outright as dead code** (it had no production
caller), so the pattern now spans two live formats, not three. Target: reframe
the two as fixed magic / format tags, or document a real persisted-format intent.

**Owner.** The comm / I/O owner.

**Trigger.** The next wire-format change.

### Byte-neutral consolidations — already executed

Recorded so a future audit does not re-raise them. Each was byte-neutral with
the sacred-parity baselines green.

- **Sampled arms promoted to co-equal engines.** The forward and simulation
  passes now dispatch their sampled arm through a named engine (`run_sampled` in
  `training/forward_pass_state.rs`; `run_sampled_simulation` in
  `simulation/state.rs`), matching the backward pass's clean two-arm dispatcher —
  the original sampled path is no longer left inline.
- **Node-graph query surface as methods.** The queries that took a `&NodeGraph`
  first and answered questions about it are now methods on `impl NodeGraph`
  (`setup/node_graph.rs`), not free functions.
- **Write-partition helper.** The simulation writer's repeated
  create-dir/write/push skeleton is now one `write_partition` helper
  (`crates/cobre-io/src/output/simulation_writer.rs`).

**Owner.** The training and I/O owners (as executed). **Trigger.** None —
already done.

### Removed dead code — recorded as removed

**What it is.** The resolved-parameters MPI broadcast pair (a postcard
serialize/deserialize envelope with its own version byte) had zero production
callers — the resolved-parameters table is built per-rank and never broadcast —
and was **deleted outright** under the no-dead-code directive, together with its
public re-export, its wire-format checklist entry, and its reserved-seam
register entry. It is recorded here as removed, **not** as a reserved seam, so a
future audit does not resurrect it as either live or reserved.

**Owner.** The training owner. **Trigger.** None — removed.

**Guard.** The serialize / deserialize symbols no longer resolve anywhere in the
workspace.

### Organizational / quality follow-ups (low priority)

- **God-functions with weak extraction rationale.** The cut-management driver and
  the per-node backward compute function each inline separable sub-phases that
  extract cleanly as `&mut self` methods; the per-node successor reification is
  duplicated between the sampled and enumerated backward paths, and one shared
  reification extraction resolves both the god-function length and the copy.
  **Owner.** The training owner. **Trigger.** The next substantive backward-pass
  change.
- **Backward-scheduler buffer-prealloc duplication.** The two backward schedulers
  duplicate their per-worker buffer pre-allocation; a shared prepare-buffers
  helper folds it, adjacent to the external-noise glue dedup registered above.
  **Owner.** The training owner. **Trigger.** The same frozen-hot-path / DCS-boundary
  refactor.
- **Mega-file / inline-test-giant asymmetry.** The workspace module is a flat
  mega-file holding many per-worker arena structs, and the LP-builder
  entries/columns modules carry giant inline test modules while their sibling
  builder submodules use extracted test files. Split each into a directory module
  / sibling test file matching the crate's prevailing convention. **Owner.** The
  training owner. **Trigger.** Navigability-driven, low priority.
- **Policy-dir resolve triplication + naming split.** The policy-directory
  resolve-and-guard skeleton is repeated across the warm-start, resume, and
  simulation load sites; extract a shared resolver. Two naming conventions for the
  same borrowed/owned duality coexist; unify them. One manifest name is stale (it
  is a whole-comparison bundle carrying node/pool counts, not a per-stage record).
  **Owner.** The policy owner. **Trigger.** The next policy-load change.
- **Basis-reconstruct hardening.** Add the truncation debug-assertion at the
  reconstruction entry and drop the always-zero telemetry field; the positional
  demotion of excess basic cuts is a count-balance necessity, not a quality bug —
  measure the basis-consistency-failure ratio before investing in a
  metadata-driven demotion. **Owner.** The training owner. **Trigger.** A
  warm-start-quality investigation.
- **Shared solve-prep re-home.** The shared solve-prep single owner lives under
  the training module, yet the simulation pass depends on it — a
  simulation→training dependency the shared-primitive hoist's own principle names
  an inversion. A byte-neutral move to a neutral home plus import retarget
  resolves it; land it with the simulation-dispatch symmetry work so the
  simulation call site is retargeted once. **Owner.** The training owner.
  **Trigger.** The simulation-dispatch symmetry refactor.

### Cleared (recorded so a future audit does not re-raise)

The computed-parameter resolver is not a god-function (cohesive); the water
block-mode fill duplication is deliberate-by-design (two distinct LP
formulations); and the cut-pool substrate, the shared solve-prep, the training
session, the captured-basis codec, and the entity-family writers/extractors are
positive references, not debt.

## Audit-evidence

The following mechanical checks were run against the tree at the time this
document was authored. Each command is regenerable — re-run it to get the
current state rather than trusting the prose below as a frozen count.

### Clippy, full feature set

```
cargo clippy --workspace --all-targets \
  --features "mpi numa shared-memory serde schema slow-tests flatc-conformance test-support" \
  -- -D warnings
```

Zero warnings, zero errors, across the full declared feature set (including
`mpi`, built against the local MPICH toolchain).

### `#[allow(...)]` census

Regenerate with:

```
grep -rn '#!\?\[allow(' crates/*/src --include='*.rs'
```

Every hit falls into one of three classes, and none is plan-dead-unconsumed:

- **Load-bearing.** Numeric-cast lints (`cast_possible_truncation`,
  `cast_precision_loss`, `cast_sign_loss`, `cast_possible_wrap`) and
  refactor-decision lints (`too_many_arguments`, `too_many_lines`,
  `type_complexity`, `struct_field_names`, `implicit_hasher`,
  `needless_pass_by_value`, and similar) on production code, each carrying a
  `// Rationale:` comment naming the non-obvious choice the lint would
  otherwise flag — the majority of the census.
- **Reserved-seam (Voice 4).** `dead_code` attributes each paired with a
  comment naming what will consume the item once a specific reader lands (the
  water travel-time topology and Lipschitz entries above are examples; several
  more of the same shape exist in `lp/builder/{layout,template}.rs`,
  `production/fpha_fitting/`, and `cobre-solver`'s FFI binding modules, the
  last being the standard `#![allow(dead_code)]` convention for a raw
  1:1-mapped C binding surface).
- **Symmetry-or-test-retention.** `unwrap_used` / `expect_used` / `panic` /
  `float_cmp` clusters on `#[cfg(test)]` modules (test code is exempt from the
  library's `unwrap_used = "deny"`), plus a handful of fields/functions kept
  for API symmetry and exercised only by unit tests (each with a `// Rationale:`
  naming the symmetric counterpart or the asserting test).

`#[allow(deprecated)]` sites (`crates/cobre-stochastic/src/{lib.rs,par/mod.rs,par/evaluate.rs}`)
are a fourth, narrower class tied to the deprecated-with-fallback surface;
verifying that surface is out of scope for this sweep (a dedicated ticket owns
the clean-break gate).

### `cargo machete`

```
cargo machete
cargo machete crates/cobre-python
```

Both report no unused dependencies.

### Graph-shape-predicate grep

```
grep -rn '\bis_chain\b' crates/cobre-sddp/src --include='*.rs'
grep -rn 'nodes\.is_empty()' crates/cobre-sddp/src --include='*.rs'
grep -rn 'graph\.is_none()\|graph\.is_some()' crates/cobre-sddp/src --include='*.rs'
```

`is_chain` returns zero hits — no such predicate exists; the engine is
node-native and does not special-case chains by name. The remaining hits:

- `policy/policy_load.rs` (`compare_graph_manifest_identity`,
  `resolve_warm_start_counts`) — policy-load validation and boundary-injection
  pool resolution, both one-time at load time, not per-iteration.
- `setup/node_graph.rs`'s `graph.nodes.is_empty()` — a one-time,
  setup-construction-time choice between `build_chain_node_graph` and the
  declared-graph builder, both producing the same `NodeGraph` representation
  the rest of the engine consumes uniformly afterward; not re-checked inside
  the forward/backward training loop.
- `setup/node_graph.rs`'s `frontier.next().is_none()` /
  `candidates.next().is_none()` / `parent[succ.child].is_none()` —
  `debug_assert!` invariant checks inside diagnostic helpers
  (`frontier_node`, `node_parent`, `build_parent_map`), not `if`-forks.
- `training/forward/enumerated.rs`'s `parent[node].is_none()` — root-vs-interior
  node detection inside the forward worker loop. This is universal graph-walking
  logic present in any DAG (every node either has a parent or is a root); it is
  not a chain-vs-tree shape fork and does not distinguish a chain study from a
  branching one.

None of these is an `is_chain`-style special case reaching training dispatch.
