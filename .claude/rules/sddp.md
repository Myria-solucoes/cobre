---
paths:
  - "crates/cobre-sddp/**/*.rs"
---

# SDDP Numerical & Algorithm Conventions

Hard-won correctness contracts of the SDDP solver. Each one is a _contract_, not
a style preference: a plausible-looking deviation produces wrong bounds, rejected
warm-starts, or silently understated cuts that still compile and pass most tests.
Verify against the cited code before changing any of them.

## Benders cut sign & subgradient extraction

The FCF stores the **raw subgradient** `∂Q/∂x` as a cut's `coefficients` (it is
_not_ negated at storage). That subgradient is the incoming-state column's
reduced cost **divided** by `col_scale`:
`∂Q/∂x_orig = rc_scaled / col_scale[col]` — divided, not multiplied, because the
pin sets `v_scaled = v_orig / col_scale`. Cut-row construction then negates the
gradient so the LP row reads `−∇·x + θ ≥ intercept`, yielding the Benders cut
`θ ≥ Q(x̂) + π'(x − x̂)`.
Read: `training/backward/duals_extraction.rs` (`extract_duals_from_view`), `cut/fcf.rs`, and
`cut::row::push_scaled_coefficient`, where `batch.values.push(-coeff * d)`
applies the negation.

## State pinning uses column bounds, not equality rows

Incoming state is pinned with `set_col_bounds` on the incoming-state LP column.
There is no state-fixing row range in the LP; incoming state is pinned entirely
via column bounds. Always resolve the LP
column — for both pinning and dual extraction — via
`StateLayout::state_to_lp_incoming_column`; never assume a fixing-row index.
Read: `lp/indexer/state_layout.rs`.

## FPHA uses average storage

The FPHA generation constraint is
`g ≤ γ₀ + (γᵥ/2)·(V_in + V_out) + γ_q·q (+ γ_s·s)`. The `−γᵥ/2` coefficient
appears on **both** the incoming and outgoing storage columns — not on `V_out`
alone. (Discovered during deterministic case D06.)
Read: `lp/builder/entries.rs` (`fill_fpha_entries` — pushes `−γᵥ/2` onto both the
incoming- and outgoing-storage columns), `lp/builder/rows.rs` (`fill_fpha_rows`),
and `lp/builder/template.rs`.

## Cut pool is append-only; basis matches by slot identity

Cuts are never removed from the LP. Deactivation toggles a cut row's RHS bounds
to the `±f64::INFINITY` sentinel (trivially satisfied); every cut keeps a stable
slot index for the lifetime of the run. The per-iteration template refreeze
encodes **only active cuts** (one row per `active_cuts()` entry), not inactive
cuts at sentinel bounds. Warm-start basis reconstruction therefore matches stored
cut rows to current LP rows by **`CutPool` slot identity**, never by row count.
`reconstruct_basis` is the single hot-path entry point — never bypass it.
Read: `cut/pool.rs`, `cut/basis_reconstruct.rs`.

## NCS stochastic availability is a dimensionless factor

Non-controllable-source availability `α_r(ω) ∈ [0, 1]` is dimensionless. The
realized cap is `A_r = max_gen · clamp(mean + std·η, 0, 1)`. The
`non_controllable_models.parquet` stores `(mean, std)` **as factors**, not as MW.
Read: `stochastic/noise.rs` (`transform_ncs_noise`, `compute_effective_eta`).

## Lower-bound evaluation must patch NCS

`evaluate_lower_bound` patches NCS column bounds per opening via
`transform_ncs_noise`, exactly as the forward and backward passes do. Skipping
the patch understates the bound (a real bug caught during D15). The patch inputs
ride on `LbEvalSpec` (`ncs_max_gen`, `ncs_allow_curtailment`).
Read: `training/lower_bound.rs`.

## Per-stage exchange in the backward pass

`exchange()` is called inside the backward loop, once per stage, not in a
separate pre-pass before the loop.
Read: `training/backward_pass_state.rs`.

## No EWMA upper bound

`ConvergenceMonitor::upper_bound()` returns the raw per-iteration upper bound —
there is no exponentially-weighted smoothing. Gap closure is immediate for
deterministic cases.
Read: `convergence/convergence.rs`.

## Spillage is frozen `[0, 0]` during PreFilling

A `PreFilling` hydro's spillage column is pinned `[0, 0]` — no dam exists yet to
spill from, and its incremental inflow has already left via the short-circuit, so a
free spillage column injects phantom water onto the first active downstream hydro's
water-balance row (a conservation violation). The freeze is gated on
`Phase::PreFilling` ALONE. Two wrong-but-compiling alternatives: extending the
freeze to `Filling` removes the legitimate over-dam relief valve an impounding
reservoir needs (D40); gating on `filling.is_none()` leaves the phantom-spill hole
open for a filling hydro in its own `PreFilling` sub-phase (D38, D39). Turbine and
diversion differ — they are frozen in BOTH `PreFilling` and `Filling` (no installed
machinery), whereas spillage is legitimately free in `Filling`.
Read: `lp/builder/columns.rs` (`fill_spillage_columns`). Cases: D38, D39, D42
(phantom PreFilling spill removed); D40 (legitimate Filling-phase spill retained).

## Water travel time

A declared upstream→downstream arc introduces in-transit "bucket" state: one
Markov-1 volume slot per `(downstream plant, lag)` absorbs water in flight. With
the feature compiled in but no arc declared (`n_buckets == 0`), every path below
collapses to the pre-bucket layout byte-for-byte; the moment any arc is
declared, each of the following is a contract.

### In-transit bucket dynamics & sign

The bucket-definition row is a ring shift, `b_d^out = b_{d+1}^in + k_d·D_i`:
`fill_transit_bucket_definition_entries` emits the structural `+1`/`−1` terms and
`fill_arc_release_block_entries` deposits the arc's `k_d`-weighted release from
the SAME release column that also carries `k_0` onto the balance row — never a
separate once-per-stage family. Incoming buckets are pinned via column bounds,
resolved through `StateLayout::state_to_lp_incoming_column`'s explicit bucket
arm, never the `anticipated_state` catch-all. Subgradient extraction divides the
incoming bucket column's reduced cost by `col_scale` (`extract_duals_from_view`,
the same rc/col_scale contract as storage); the cut row renders the **outgoing**
bucket column through `StateLayout::lp_column_for_state`'s identity arm and
multiplies `col_scale` back on via `push_scaled_coefficient` — divided on
extract, multiplied on render, identical to storage. Swapping which column is
pinned/read, or dividing on render instead of extract, prices the in-transit
water in the wrong direction — a wrong bound that still compiles. A fold
implementation (crossing mass absorbed same-stage, no bucket at all) can reach
the same total cost as the correct one, so total cost alone cannot discriminate
— only the dual's sign/magnitude and the per-stage delivery split do.
Read: `lp/builder/entries.rs` (`fill_transit_bucket_definition_entries`,
`fill_arc_release_block_entries`), `lp/indexer/state_layout.rs`
(`StateLayout::state_to_lp_incoming_column`, `StateLayout::lp_column_for_state`),
`training/backward/duals_extraction.rs` (`extract_duals_from_view`), `cut/row.rs`
(`push_scaled_coefficient`, `push_cut_row`). Pinned by the bucket-arm
column-resolution tests (outgoing resolves by identity, incoming resolves to the
pinned column via an explicit arm, never the anticipated catch-all) and the
per-stage-visit bucket-pinning regressions in the backward pass and lower-bound
evaluation; a sub-stage-delay bucket-dual regression is the fold-discriminating
pin for the sign/magnitude itself.

### k-factor conservation

`resolve_spread` sums the stage-clock weights to `Σ_d k_d = 1` per arc per
anchor stage (`debug_assert`-enforced), and `fill_arc_release_block_entries`
asserts the same sum immediately before it deposits. A closed-form ceiling
depth (e.g. `⌈t_v/h_t⌉`) is a plausible-looking replacement for the resolver's
overlap-based depth and silently drops trailing mass on a non-uniform calendar
— conservation violated, not a compile error.
Read: `lead_time/mod.rs` (`resolve_spread`), `lp/builder/entries.rs`
(`fill_arc_release_block_entries`). Pinned by the resolver's monthly-then-weekly
counterexample regression (asserting the correct, deeper depth against the
closed-form ceiling's shallower, wrong one) and the stage-level conservation
regression exercising the `Σ_d k_d = 1` debug_assert directly across
non-uniform calendars; a mixed-calendar end-to-end regression extends the pin
to delivered-plus-horizon-drop equalling released, per arc, to floating-point
tolerance.

### Canonical bucket ordering

Bucket columns sort by the downstream plant's canonical
`(operational_start_date, id)` index — the same order `System::hydros` already
carries — then by lag; never by raw declared id, never by cascade-traversal
order. `build_transit_bucket_topology` derives `column_order` from that canonical
iteration alone. Emitting buckets in traversal order instead makes the state
layout input-declaration-order-dependent, breaking the
declaration-order-invariance hard rule.
Read: `setup/bucket_topology.rs` (`build_transit_bucket_topology`,
`TransitBucketTopology::column_order`). Pinned by the bucket column-order
declaration-invariance regression: two systems differing only in the
declaration order of their hydros produce identical `column_order`,
`per_plant_depth`, and `n_buckets`.

### Stage-0 seed: windowed IC anchor

`build_initial_transit_bucket_state` seeds every declared arc's stage-0
incoming buckets directly from its `past_defluences` windows — never a
positional walk over a fixed pre-study calendar. For upstream hydro `i`'s
window `[start_date, end_date)`, `e_off = start_0 − end_date` and
`width = end_date − start_date` feed `ic_anchor_k` exactly as it already
takes `(cumulative_before, period_duration)`: the windowed derivation lives
entirely in how the caller computes those two offsets from calendar dates,
never inside `ic_anchor_k` itself. A hydro may carry multiple, non-contiguous
windows; the seed must `filter` over every window with a matching `hydro_id`
and deposit each one independently
(`volume = width · M3S_TO_HM3 · value_m3s`, `seed[start+d] += k[d] · volume`)
— a `.find()` would silently keep only the first window and drop the rest,
understating the seed with no error. There is no `past_inflows` fallback:
`cobre-io`'s `validate_travel_time` row-5 gate guarantees every declared
arc's windows cover `[start_0 − t_v, start_0)` before setup ever runs this
seed.
Read: `setup/bucket_seed.rs` (`build_initial_transit_bucket_state`),
`setup/bucket_topology.rs` (`ic_anchor_k`). Pinned by the single-window
unroll regression (the `k`-weighted deposit matches the closed-form
half-share), the gapped-two-window additive regression (two non-contiguous
windows for one arc contribute independently), and the seed's own
declaration-order-invariance regression (distinct from, and in addition to,
the topology-level ordering pin above).

### Terminal credit deferred

`horizon_cap_active` caps each stage's active lag at `n_stages − 1 − t`, the
deepest lag whose target stage still lands inside the horizon;
`build_transit_bucket_row_pos` gates the per-stage LP fill on that cap, so a lag beyond
it gets no bucket-definition row at that stage — dropped by construction, not
retained and silently zeroed elsewhere. `fill_arc_release_block_entries` /
`fill_arc_release_chrono_block_entries` drop the matching deposit share rather
than write it to a stale row index, and `fill_transit_bucket_columns` freezes the
masked slot's outgoing column `[0, 0]` (the commissioning-dormant-column
convention) so no row is needed to define it. The complementary guarantee is
why dropping the row is safe: the finite horizon's zero terminal value
(`HorizonMode::Finite`, the only implemented mode) makes a masked slot's cut
coefficient structurally zero, so no solution loses value by never routing
water into it — the residual mass has no receiving stage either way. This
under-values end-of-horizon upstream release; it is a documented target-stage
imprecision, not a bug to patch by capping
`TransitBucketTopology::per_plant_depth`/`column_order` too — those size from the
global max over every anchor and must retain what the earliest stages need.
Read: `setup/bucket_topology.rs` (`horizon_cap_active`), `lp/builder/layout.rs`
(`build_transit_bucket_row_pos`), `lp/builder/columns.rs` (`fill_transit_bucket_columns`).
Pinned by the horizon-depth-cap regression (the last stage's active-lag cap
reaches zero, so no slot targets past the horizon), `build_transit_bucket_row_pos`'s
own consumption regression (that same cap sequence emitting correspondingly
fewer rows), and a sub-stage-delay case's last-stage release, whose dropped
share surfaces as an uneven per-stage delivery split rather than a credited
one.

### Sub-contracts: mode-independent sizing, aggregation consistency, fixed delivery density

The bucket state stays a pure function of stage lengths, never of
`n_blks`/`block_mode`, only because each of the following holds:

- **Depth from stage lengths alone.** Bucket depth and `n_buckets` derive from
  the per-stage calendar and the pre-study anchor alone
  (`study_stage_durations`, `build_transit_bucket_topology`) — never from `n_blks` or
  `block_mode`. Deriving any part of the depth inside a block-aware code path
  re-couples the state dimension to how a stage happens to be resolved.
- **Shared arrival density.** A chronological stage's per-block deposit shares
  `block_deposits`/`within_stage_routing` and the stage-level `stage_weights`
  come from the same shared arrival density (`resolve_spread`'s
  `stage_weights`/`block_deposits`/`within_stage_routing`,
  `resolve_block_factors`'s `BlockFactors`), so `Σ_b w_b·χ_{b,d} = k_d` holds
  by construction. Building `block_deposits`/`within_stage_routing` from one
  density and `stage_weights` from another lets the chronological and
  parallel cuts diverge and silently breaks conservation.
- **Fixed delivery density.** A maturing bucket delivers into its arrival
  stage's blocks through a fixed, `block_mode`-independent template density
  (`resolve_chrono_arrival_density`), never by tracking which origin block a
  unit came from. Tracking origin-to-arrival-block correlation would grow the
  bucket into a per-block vector whose length scales with the receiving
  stage's `n_blks` — re-violating the depth-from-stage-lengths property above.

Read: `lead_time/mod.rs` (`resolve_spread`'s
`block_deposits`/`within_stage_routing`/`arrival_density` fields,
`resolve_block_factors`'s `BlockFactors`), `lp/builder/entries.rs`
(`fill_chronological_water_entries`, `resolve_chrono_arrival_density`). Pinned
by the shared-density-consistency regression exercising the aggregation
debug_assert directly, the chronological block-table regression matching the
worked kappa/chi numbers, and the `K = 1` chronological-vs-parallel
byte-identity regression; a state-dimension-equality regression across
parallel and chronological builds is the direct pin for mode-independent
sizing.
