# Terminal-leaf solve fusion, cross-iteration basis reuse, and static terminal pool

> **Status:** Proposal — analysis + target design, not yet implemented. Code is cited by symbol; verify against the tree before implementing. Snapshot figures are calibration-time context, not standing claims.

## Summary

In enumerated (exhaustive) traversal, the terminal-stage leaf LPs are solved
**twice per iteration** — once on the forward pass and once on the backward pass
— and the backward solve is **cold**, discarding an optimal basis the forward
already captured. Because the terminal stage is the one stage whose LP is
invariant between the two passes (a leaf has no successor, so its future-cost
term never changes within an iteration), that second solve is redundant for the
subset of leaves whose opening is a single declared scenario. Separately, every
leaf carries its future-cost cuts in **two** places — a dynamic `CutPool` and a
per-iteration re-baked LP template — even though a leaf never adds or removes a
cut for the life of the run.

This document proposes three related, independently-shippable changes, all scoped
strictly to the terminal stage:

1. **Fusion** — extract the leaf's state duals during the forward solve and feed
   them to the backward cut, eliminating the redundant backward leaf solve.
2. **Cross-iteration basis reuse on leaves** — already present on the forward;
   the point here is to preserve it under fusion and (optionally) extend it to the
   backward for the leaves fusion does not cover.
3. **Static terminal pool** — bake the terminal LP once and drop the dynamic
   `CutPool` + per-iteration refreeze for the terminal stage, roughly halving the
   terminal-pool memory that dominates a wide terminal fan.

The workload these target is the enumerated **terminal fan** — a chain (or narrow
graph) opening into many terminal leaves, which is the DECOMP-like configuration
where a **boundary policy** injects a terminal future cost function. There, leaves
dominate the node count and the boundary cut pool is large and held per worker, so
the redundant solve and the duplicated pool are where both compute and the
observed per-worker memory concentrate.

It also answers a design question that arises alongside: **no, a per-leaf boundary
pool is not needed** for a single-boundary-policy input (see
[Per-leaf boundary pool](#per-leaf-boundary-pool-not-needed)).

## Current state (with code anchors)

### Enumerated traversal

Dispatch is `Traversal::Enumerated`: `ForwardPassState::run` →
`run_enumerated_forward`, `BackwardPassState::run` → `run_enumerated_backward`,
and simulation via `run_enumerated_simulation`. All three run a stage-synchronous
outer loop over `0..num_stages`, parallelizing across the distinct nodes within a
stage; each distinct node is solved **once** per pass.

- **Forward** (`crates/cobre-sddp/src/training/forward/enumerated.rs`) —
  `solve_forward_node` solves each node (leaves included) at its parent's
  captured outgoing state, extracts the raw stage cost and the outgoing LP state,
  and on the frozen (non-DCS) path captures the node's optimal basis into the
  session `BasisStore`, keyed by `(m_rep[node], node)`. It **discards duals**: the
  scatter into the dense backward records does `rec.dual.clear()`.
- **Backward** (`crates/cobre-sddp/src/training/backward/replicated.rs`, driver
  `training/backward_pass_state.rs`) — iterates `NodeGraph::backward_cut_levels()`,
  which enumerates **cut-generating nodes only** (a node with ≥1 successor; a leaf
  never generates a cut). For each such node it solves **its children's** LPs
  (every child × every opening) at the node's forward trial state, extracts the
  state subgradient (`extract_state_duals_only`, `rc / col_scale`), risk-aggregates
  once over the flattened successor×opening vector, and appends one cut. This path
  passes `stored_basis: None` and does `reset_solver_state()` + a fresh
  `load_backward_lp` per child — i.e. **cold**.
- **Simulation** (`crates/cobre-sddp/src/simulation/enumerated.rs`) — a wholly
  separate pass; solves each own-path node once and `re_expand`s the result into
  every owning leaf's output row. No reuse of training solves.

### The terminal leaf, specifically

The terminal-stage leaves are solved on **both** passes: once in the forward, and
once in the backward as the successors of the penultimate (last cut-generating)
stage. Whether the two solves are the _same LP_ depends on the leaf's opening
source (`NodeGraph`, `node_opening_range` / `node_pinned_scenario`):

- **External leaf** (`scenario_id` present ⇒ `NodeOpenings(External, k, len = 1)`):
  the forward solves the single pinned column `k`; the backward's
  `successor_outcome_count` sees `len == 1` and solves the **same** single column
  `k` with byte-identical noise, at the **same** incoming state (the forward
  parent's captured outgoing state), against the **same** terminal template. This
  is the **same LP** — the fusion-eligible case.
- **Generated leaf** (`scenario_id` absent ⇒ `NodeOpenings(Generated, 0, n)`): the
  forward samples **one** hash-selected opening while the backward integrates
  **all** `n` openings. Not the same LP, and the backward's exhaustive integration
  is a hard correctness contract (see [Guardrails](#correctness-invariants--guardrails)).

The forward **already reuses the leaf basis across iterations**: the worker reads
`store.get(local_m, node)` and passes it as `stored_basis`; `run_stage_solve`
applies it under the node-tag filter (only when the stored `node_id` matches),
then the fresh basis is swapped back into the store. So a leaf's forward solve at
iteration _i_ warm-starts from iteration _i−1_'s basis; only the incoming-state
column bounds changed, which the dual simplex absorbs cheaply. The backward does
**not** consult the store — it re-solves the leaf cold every iteration even though
the forward captured that exact leaf's optimal basis moments earlier.

### Terminal FCF is stored twice

The forward's non-DCS solve loads the baked template `frozen[pool_id]` into the
solver, and separately passes the dynamic `fcf.pools[pool_id]` (`CutPool`) to
`run_stage_solve` for slot-identity basis reconstruction. `frozen[pool_id]` is the
per-iteration refreeze of the pool's `active_cuts()`. For the **terminal** pool
these two structures hold the same fixed boundary cuts, and the refreeze re-bakes
identical bytes every iteration — because a leaf never adds or removes a cut. The
`inject_boundary_cuts` path even builds the terminal pool with
`CutPool::new_with_warm_start(…, boundary_cuts)` "keeping capacity for new training
cuts" — capacity that is never used, since the terminal pool receives no cuts.

## Findings

**F-A — Redundant terminal backward solve (the fusion opportunity).** Every
terminal leaf is re-solved on the backward, unconditionally and cold. For External
(single-opening) leaves that solve is bit-for-bit the same LP the forward already
solved. The forward has the duals available in its `SolutionView.reduced_costs`
and simply clears them; the backward needs exactly those duals for the
penultimate-stage cut. So the second solve is eliminable for External terminal
leaves.

**F-B — Cross-iteration leaf basis reuse exists on the forward, not the
backward.** The forward warm-starts each leaf from the prior iteration's basis via
`BasisStore`. Under fusion the backward leaf solve disappears, so the leaf's only
solve per iteration is the already-warm forward one — the requirement "reuse the
leaf basis across iterations" is then satisfied by construction. For the leaves
fusion does **not** cover (Generated), the backward remains cold; wiring
`BasisStore` into the replicated backward is an optional, separately-shippable
speed-up (no compute is removed, only pivots).

**F-C — Leaves double-store their FCF.** The dynamic `CutPool` and the per-iteration
baked template are redundant for the terminal stage, whose cut set is fixed. The
pool's growth machinery, slot-identity reconciliation, and per-iteration refreeze
are all dead weight there. This is the memory the user identified as doubled, and
it is largest exactly in the boundary-fan case.

## Proposed design

All three changes are gated on the same predicate — **terminal stage** — and
F-A/F-B additionally on **External (single) opening**. None of them touches
interior nodes or Generated leaves.

### 1. Terminal-leaf forward/backward fusion

During the forward solve of an External terminal leaf, extract the state
subgradient the same way the backward does (`extract_state_duals_only`, the
`rc / col_scale` contract) and persist `(objective, duals)` alongside the leaf's
`out_state` in the enumerated forward scratch (the record whose `dual` field is
today cleared). In `run_enumerated_backward`, when a cut-generating node's
successor is an External terminal leaf, consume the stored `(objective, duals)`
instead of loading and solving that child's LP.

- The forward already knows a node is terminal (`horizon.is_terminal`) and
  External (`node_pinned_scenario(node).is_some()` with `node_opening_range` len 1),
  so the predicate is local.
- The cut assembly (`assemble_outcome_weights`, `RiskMeasure::aggregate_cut_into`)
  is unchanged — only the _source_ of a terminal successor's outcome slice changes
  from "solve now" to "read the forward-captured slice."
- Non-terminal successors, and Generated terminal successors, keep solving exactly
  as today.

### 2. Cross-iteration basis reuse

No new mechanism is required for the fused (External) leaves — the forward's
existing `BasisStore` warm-start carries them. Preserve it: the fusion must keep
capturing the leaf basis on the forward so iteration _i+1_ still warm-starts.

Optionally, for Generated terminal leaves (and, more generally, for the replicated
backward), pass the node-tagged `BasisStore` entry as `stored_basis` in
`solve_replicated_outcome_slice` instead of `None`, letting `reconstruct_basis`
(the frozen hot-path entry point) warm-start the child. This is a pure pivot-count
optimization; it changes no output at a unique optimum and is bounded by the same
determinism caveat below at a degenerate one.

### 3. Static terminal pool

Bake the terminal LP template once — with the boundary cuts as fixed rows (or, with
no boundary policy, the plain terminal template with θ pinned) — and stop
materializing the terminal `CutPool` as a growable, per-iteration-refrozen
structure:

- Skip the per-iteration refreeze for the terminal pool (its `active_cuts()` never
  change).
- Drop the dynamic `CutPool` copy of the boundary cuts once the static template is
  built, so the coefficients live in one place, not two.
- Terminal basis handling simplifies: the template is structurally invariant across
  iterations, so only column bounds change and the stored basis maps 1:1 — no
  slot-identity reconciliation is needed at the terminal stage. This must be a
  _terminal-only_ short-circuit; the interior slot-identity reconstruction contract
  is unchanged.

The `terminal_has_boundary_cuts` flag (today read from the terminal pool's
`warm_start_count`) becomes a property of the static terminal template.

### Per-leaf boundary pool: not needed

A per-leaf boundary pool is **not** required for a single-boundary-policy input,
and would not improve modeling fidelity for it.

A boundary policy is a future **cost function** of the ending state,
`F(x) = maxₖ (αₖ + βₖ·x)`, not a scalar. Every terminal leaf evaluates the **same**
`F` at its **own** ending state `x_leaf` (storage _and_ inflow-lag state are both in
the state vector), so a wetter ending (higher storage / higher recent-inflow lag
state) and a drier ending already receive **different** terminal values from one
shared pool. The shared terminal pool is therefore semantically correct here, and
the static single template in change 3 is the right representation: one read-only
`F` shared by all leaves, each pinned to its own incoming state via column bounds.

A genuinely per-leaf boundary FCF (distinct `α/β` per leaf) is only meaningful when
terminal nodes carry a discrete continuation regime **not** encoded in the
continuous state — the general Markovian per-terminal-node value. That requires
per-leaf boundary **input** (more than one boundary policy), which is explicitly
out of scope: the user supplies a single boundary policy and expects no per-terminal
scenario differentiation. So this reframes the previously-registered
"per-terminal-state continuation value" limitation
(`reserved-seams-and-deferred-debt.md`, `policy-graph-limitations.md`) as **not a
limitation for the supported single-input use case** — it is a feature that would
require new input, not a fix to the current one.

## Correctness invariants & guardrails

These are the lines the implementation must not cross; each maps to an existing
`.claude/rules/sddp.md` contract.

- **Terminal stage only.** Interior nodes' backward LPs gain cuts from deeper
  levels processed earlier in the same backward pass, so a non-leaf node's forward
  and backward LPs genuinely differ. Fusion and the static pool must be gated on
  `is_terminal`; applying either to an interior node is wrong-but-compiling.
- **External openings only for fusion.** Generated terminal leaves must keep the
  exhaustive backward integration ("The branching backward integrates every
  successor exhaustively"). Reusing a single forward opening there understates the
  cut and drives `final_lb` above `final_ub` — an invalid bound that still
  converges.
- **Duals are a valid subgradient.** Any optimal dual of the leaf LP is a valid
  Benders subgradient, so the fused cut is a valid supporting hyperplane; lower and
  upper bounds stay valid regardless of which optimal vertex the dual comes from.
- **Slot-identity basis reconstruction is untouched off the terminal stage.** The
  terminal static-template short-circuit replaces slot-identity matching _only_
  where the cut set is provably invariant; every non-terminal pool keeps
  `reconstruct_basis` and the append-only slot-identity contract.
- **Node-tag warm-start filter is preserved.** The optional backward warm-start
  (change 2) must pass the basis through the same `node_id` match that
  `run_stage_solve` enforces — a cross-node warm-start is a silent wrong-vertex on
  the CLP backend.

## Determinism & re-baselining

Today the backward solves the terminal leaf **cold**; fusion instead seeds the
penultimate cut from the **forward** leaf vertex. At a unique optimum these are the
same vertex → byte-identical output. At a degenerate optimum they can differ — a
valid, permitted `hot ≠ cold` divergence (the determinism contract explicitly
allows a warm/cold solve to report a different-but-equally-valid vertex), but a
behavior change that shifts cuts and `final_lb`.

Plan:

1. Implement behind the terminal/External predicate; run the golden parity suite
   (`parity_hash_*`) on both backends without regen.
2. If every golden case reproduces its committed hash, the change is byte-neutral —
   ship without a re-baseline.
3. If a golden case is degenerate at a terminal leaf and its hash moves, confirm
   result-neutrality by capturing `(final_lb, final_ub, cut pool)` on the original
   and changed code and checking the _bounds and cost_ are equal (a moved _dual
   vertex_ is expected), then re-baseline **both** `parity_baselines/` (HiGHS) and
   `parity_baselines_clp/` (CLP) in the same change, per the release checklist.

Change 2's optional backward warm-start carries the same caveat; change 3 (static
template) is result-neutral by construction (it re-bakes the identical rows), so it
should be byte-identical — verify with the same suite.

## Memory analysis

Change 3 removes one full copy of the terminal FCF coefficients per worker. In a
terminal fan the terminal pool holds the boundary cut set (up to the boundary
policy's cut count × terminal state dimension), and it is held per worker. Storing
it once (static template) instead of twice (pool + refrozen template) is the direct
lever on the per-worker footprint that made high thread counts run out of memory in
the DECOMP-like configuration. It also removes the per-iteration refreeze work for
the terminal stage. Quantify the actual saving on the benchmark case during
implementation rather than pinning a figure here.

## Interaction with the boundary policy

The boundary FCF _is_ the terminal stage's fixed future-cost term, so all three
changes compound with a boundary policy loaded:

- Fusion halves the number of times the (large) boundary FCF is evaluated at the
  boundary, because the leaf is priced once instead of twice.
- The static terminal template (change 3) is the natural home for the injected
  boundary cuts — `inject_boundary_cuts` bakes them into the one static template
  instead of a growable pool + refreeze.
- The single shared terminal template is the correct representation for a single
  boundary policy (see [Per-leaf boundary pool](#per-leaf-boundary-pool-not-needed)).

No change to the boundary load/validate/reconcile path
(`policy/policy_load.rs`, `policy/reconcile.rs`) is required; only where the
reconciled terminal cuts land (static template vs dynamic pool).

## Phasing

1. **Change 3 (static terminal pool)** first — result-neutral, removes the memory
   doubling and the refreeze, and simplifies terminal basis handling, making the
   other two easier to reason about.
2. **Change 1 (fusion)** — the compute win; gated on terminal + External; carries
   the re-baseline check.
3. **Change 2 (backward warm-start for Generated leaves)** — optional pivot
   reduction for the leaves fusion does not cover.

## Test plan

- **Analytical:** a small terminal-fan fixture with a known optimum where the
  fused cut is derived by hand from the forward leaf duals; assert the produced cut
  matches (extends the `branching_value_oracle` style).
- **Behavioral:** `final_lb == final_ub` to tolerance on the fan, unchanged by
  fusion; a boundary-policy fan asserting the terminal value equals the boundary
  `F(x_leaf)` per leaf from the single shared template.
- **Determinism:** the existing `opening_order_determinism` and enumerated
  reproducibility gates must stay green (thread/rank-shape invariance).
- **Parity:** run `parity_hash_*` on both backends; re-baseline only under the
  degenerate-terminal-leaf condition above.
- **Memory:** measure per-worker RSS on the DECOMP-like benchmark before/after
  change 3 (not a committed number; a regression check during implementation).

## Code anchors

- Enumerated forward + basis capture + dual-clear:
  `crates/cobre-sddp/src/training/forward/enumerated.rs`
  (`solve_forward_node`, `enumerated_stage_worker`, the `store.get`/scatter,
  `rec.dual.clear()`).
- Enumerated backward cold re-solve:
  `crates/cobre-sddp/src/training/backward/replicated.rs`
  (`run_backward_node_replicated`, `solve_replicated_outcome_slice`),
  driver `crates/cobre-sddp/src/training/backward_pass_state.rs`.
- Opening-source asymmetry + backward levels:
  `crates/cobre-sddp/src/setup/node_graph.rs`
  (`node_opening_range`, `node_pinned_scenario`, `successor_outcome_count`,
  `backward_cut_levels`, `assemble_outcome_weights`).
- Dual extraction / stage solve:
  `crates/cobre-sddp/src/training/backward/duals_extraction.rs`
  (`extract_state_duals_only`), `crates/cobre-sddp/src/solve/stage_solve.rs`
  (`run_stage_solve`, the node-tag filter), `crates/cobre-sddp/src/cut/basis_reconstruct.rs`.
- Terminal FCF storage / boundary injection:
  `crates/cobre-sddp/src/cut/pool.rs` (`CutPool`),
  `crates/cobre-sddp/src/cut/fcf.rs`,
  `crates/cobre-sddp/src/policy/policy_load.rs` (`inject_boundary_cuts`),
  terminal θ pin in `training/forward/enumerated.rs` and
  `lp/builder/columns.rs` (`fill_theta_column`).
- Contracts: `.claude/rules/sddp.md` ("The branching backward integrates every
  successor exhaustively", "Cut pool is append-only; basis matches by slot
  identity", "A stored basis warm-starts only at its own node (node-tag)", "State
  pinning uses column bounds, not equality rows", "No EWMA upper bound").
- Related: `docs/design/enumerated-traversal-distribution.md`,
  `docs/design/policy-graph-limitations.md`,
  `docs/design/reserved-seams-and-deferred-debt.md`.
