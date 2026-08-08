# Reserved Seams and Deferred Debt

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
(as opposed to unwired config or code) — single-initial-node, shared terminal
value, enumerated-selection restrictions — see
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

## Deferred-debt register

_Populated separately; not part of this sweep._

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
