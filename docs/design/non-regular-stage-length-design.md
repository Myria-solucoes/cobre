# Non-regular stage-length support — unified design

Companion to `non-regular-stage-length-support.md`, which _catalogs_ what is
deferred. This document is the _design_: how each of the three periodic/lag
features — **inflow PAR**, **water travel time**, **anticipated dispatch** —
bridges its own periodic or lag structure onto arbitrary stage calendars
(non-uniform, mixed-resolution, month/period-boundary-straddling), so cobre
supports these features "in any kind of study."

**Status:** §1 Inflow PAR — designed (Monthly implemented; Weekly/Custom designed
and guarded). §2 Water travel time and §3 Anticipated dispatch — to be designed.

---

## 0. Shared foundation

The three features are three instances of one problem — _map a periodic/lag
structure onto stages whose durations and boundaries the structure does not
control_ — so they should share one calendar core and two conventions.

**Shared calendar core (already present):**

- `cobre_core::window_period_overlaps` (`temporal/overlap.rs`) — hour-exact
  overlap of a stage window against period windows.
- `cobre-sddp lead_time::{resolve_spread, resolve_point, AnticipatedResolution}`
  — spread (k-weighted) and single-point (end-anchored decider) lag resolution
  over an arbitrary `stage_lengths_hours`.
- `SeasonMap::season_for_date` — date → season for Monthly / Weekly (ISO week,
  53→52 fold) / Custom (date ranges).

**Convergence opportunity.** The inflow-PAR bridge currently uses its own
`days_in_period` day-overlap in `lag_transition.rs` rather than the hour-exact
`window_period_overlaps` the water/anticipated resolvers use. The unified target
is one overlap primitive behind all three.

**Two conventions, applied uniformly:**

1. **Anchor = the earliest STUDY period a stage overlaps** (pre-study overlap
   excluded). This reduces to _end-anchor at the study-start edge_ (the first
   stage joins the first study period even when calendar-dominant days fall in a
   prior pre-study period) while keeping a trailing tiling-stage in the period it
   tiles. Used wherever a stage must be assigned a _single_ period.
2. **A boundary-straddling stage has two separate roles** — never conflate them:
   (a) a **single-period assignment** (which season's statistics standardize an
   inflow / which stage a commitment is decided in), and (b) a **per-day/hour
   overlap accounting** (PAR lag weights, water bucket k-weights, anticipated
   deposit shares). Role (a) picks one; role (b) splits by overlap.

---

## 1. Inflow PAR

### 1.1 How the bridge works today

The PAR(p) model is **periodic**; the study runs on **stages** whose durations it
does not control. Today's bridge is **monthly PAR + duration-weighted lag
accumulation**:

- `stochastic/lag_transition.rs::compute_monthly_transition` emits a per-stage
  `StageLagTransition {accumulate_weight, spillover_weight, finalize_period}`.
  Weekly stages of a month share ONE monthly draw
  (`precompute_noise_groups` groups by `(season_id, year)`); each stage's inflow
  accumulates into the month's average weighted by its day-share; the average
  finalizes at the last stage of the month and shifts into lag-1; a stage crossing
  the forward month boundary spills into the next month via `spillover_weight`.
- `sampling/external.rs::standardize_external_inflow` inverts the PAR to recover
  noise η, advancing the lag chain by the same accumulate/finalize/spillover
  pattern.
- `build_initial_state` places `HydroPastInflows.values_m3s` as ordinal monthly
  lags (index 0 = most recent month).

**Already cycle-general (no monthly assumption):** `SeasonMap::season_for_date`
(all three cycles); the PAR fit derives `n_seasons = season_map.seasons.len()`
(`par/fitting/ar_coefficients.rs`) and buckets history via `season_for_date`, so a
52-season weekly `SeasonMap` fits a weekly PAR with no fitting-code change;
cobre-io parses `monthly|weekly|custom`; setup honors the user's `season_map`.

**Monthly-only (the gap):** `lag_transition.rs` returns a zeroed no-op transition
for `Weekly|Custom` and a zero seed for non-Monthly
(`compute_recent_observation_seed`); the annual PAR(p)-A extension
(`par/fitting/annual.rs`) hard-codes a 12-month rolling average;
`HydroPastInflows.season_ids: Option<Vec<u32>>` is parsed + length-validated but
read by no solver code.

`RecentObservation {hydro_id, start_date, end_date, value_m3s}` is observed inflow
for the elapsed part of the study's first PAR period; `compute_recent_observation_seed`
→ `RecentObservationSeed {accum_seed, weight_seed}` primes the first lag
(blend of observed + simulated), applied at every trajectory start
(`training/forward_pass_state.rs`). Monthly-only today.

### 1.2 Two regimes

- **Regime A — monthly hydrology, sub-monthly operation** (DECOMP). Monthly PAR;
  weekly stages share the monthly draw and disaggregate it. No weekly history
  needed. The common case.
- **Regime B — weekly hydrology.** Weekly-period PAR (S=52); needs weekly history;
  weekly draws, no disaggregation.

### 1.3 Design decisions

- **Scope.** Implement **Monthly** now; **Weekly** and **Custom** are designed and
  **hard-rejected at setup** until shipped (replacing today's silent no-op, which
  produces frozen/wrong lags).
- **One generic bridge.** Generalize the monthly-hardcoded transition into a
  **period-provider-agnostic overlap bridge**: given a stage window and the period
  windows (calendar months / ISO weeks / Custom date ranges, all from the
  `SeasonMap`), compute day-weighted accumulate + forward spillover + finalize
  identically. The three cycles differ ONLY in how period windows are enumerated.
- **Anchor.** A stage's standardization season = the _earliest study period it
  overlaps_ (§0 convention 1). cobre **derives** `Stage.season_id` from dates +
  `cycle_type` via `season_for_date`; the input field is kept as an **optional
  override** (derive if absent, validate if present).
- **Two roles separated** (§0 convention 2): the single-season assignment drives
  the noise group + per-season statistics; the lag accounting attributes each day
  to its real calendar period. Pre-study days are excluded (their lags come from
  history / `RecentObservation`).
- **Disaggregation — input-free, day-weighted.** For monthly-sourced values feeding
  weekly stages (Regime A), a weekly stage's inflow = the day-weighted average of
  the monthly rate(s) it overlaps: an interior week gets its month's rate; a
  boundary week spanning months m,m+1 with (a,b) days gets
  `(a·rate_m + b·rate_{m+1}) / (a+b)`. This is the exact inverse of the day-weighted
  lag re-aggregation. **Consequence:** no intra-month weekly variability — genuine
  weekly variability requires Regime B (weekly history). A weekly-average (MLT)
  profile input and stochastic disaggregation are deferred.
- **PAR(p)-A** is **monthly-exclusive by design.** A non-monthly cycle has no
  annual/long-memory term; a weekly study runs a pure weekly PAR(p).
- **`season_ids`** on `HydroPastInflows` is wired so past lags carry their period
  resolution (needed once non-monthly lags exist).
- **`RecentObservation`** seeding is generalized to the non-monthly cycles as part
  of the generic bridge.
- **Custom tiling validation.** Monthly/Weekly tile the cycle by construction;
  Custom seasons are user-defined and may leave gaps or overlap
  (`season_for_date` → `None` for an uncovered date, silently dropping
  observations/stages). Custom therefore requires a validation that its ranges form
  a **complete, non-overlapping partition of the repeating cycle**.

### 1.4 Implementation notes

- **Boundary blend needs both months' values.** A week straddling two _drawn_
  months blends `rate_m` and `rate_{m+1}`, so the forward pass must have both
  monthly draws before disaggregating the boundary week (materialize the
  trajectory's monthly values, then split to weeks). An ordering constraint.
- **Leading-boundary sourcing.** A first week whose early days are pre-study blends
  the pre-study month's rate with the first study period's draw; source the
  pre-study portion from `RecentObservation` when supplied, else the historical
  monthly average.

### 1.5 Implementation surface

Localized (fitting, `season_for_date`, and parsing are already cycle-general): the
generic overlap bridge in `lag_transition.rs`; the day-weighted disaggregation in
scenario build; the derived `Stage.season_id`; the `season_ids` wiring; and the
setup rejections (non-monthly bridge until shipped, non-monthly PAR(p)-A, Custom
tiling).

---

## 2. Water travel time

Water travel time is modeled as in-LP k-weighted volume buckets per downstream
plant. The multi-resolution problem is the analogue of §1's boundary problem —
distribute an arriving quantity against the _arrival_ stage's real calendar, not
a template — and it resolves with the same shared convention (§0 role 2).

### 2.1 How arrivals work today

Two spreads, built at setup:

- **Stage-level `k_d`** (`setup/bucket_topology.rs::build_arc_spread_k`): the
  fraction of a release at a stage arriving at `stage+d`, overlapping the arrival
  window against the full non-uniform calendar. **Block-mode-independent and
  calendar-exact for any stage lengths**; the transit-bucket state is per-depth
  (block-count-independent). This decides _which stage_ water arrives in.
- **Block-resolved spread** (`build_arc_spread_chrono`): `SpreadResolution`
  (`block_deposits`, `within_stage_routing`, `arrival_density`), built only for
  chronological _sender_ stages, resolved against the _sender's_ blocks.

The maturing bucket `b_1^in` arrives differently by the _arrival_ stage's mode:

- **Parallel arrival** (`fill_parallel_water_entries`): one water-balance row;
  `b_1^in` enters with a single `-1.0`. No density — parallel blocks are
  load-duration slices sharing one storage state, so there is no within-stage
  arrival time.
- **Chronological arrival** (`fill_chronological_water_entries`): `n_blks`
  sequential storage rows; `b_1^in` is spread by `-ρ_b`
  (`resolve_chrono_arrival_density`). Density exists only here.

### 2.2 The gap (WTT-1) and where it bites

`resolve_chrono_arrival_density` resolves `ρ` against the _sender's_ blocks (not
the arrival stage's), reads only lag `d=1` from the immediately-previous stage,
and falls back to a uniform density when the sender is parallel or the block
counts mismatch. So on a chronological arrival stage whose partition differs from
the sender's — the multi-resolution case — the travel-time-resolved arrival
timing is lost. Parallel arrival is unaffected (single `-1.0`), so the whole gap
is confined to chronological arrival stages.

### 2.3 Design decisions

- **Support both block modes.** Parallel and chronological arrival both fully
  supported under multi-resolution.
- **Parallel arrival unchanged** — single `-1.0`, calendar-exact `k_d`. Already
  correct for any sender and any stage lengths.
- **Chronological arrival = arrival-frame `ρ` recomputation.** Compute `ρ` in the
  _arrival stage's own frame_ — against its own blocks, from its own calendar
  position relative to each contributing source, independent of the source's block
  mode. One reframing closing three holes together: (a) the arrival-stage
  partition, (b) the parallel-sender → chronological-arrival cell (today uniform),
  (c) all source lags maturing at the stage, not just `d=1`.
- **Multi-lag blend** (the constraint-respecting representative) for arrival
  stage `A`:
  `ρ_b = (Σ_d κ_d · φ_{d,b}) / (Σ_d κ_d)`, with `κ_d = k_d^(A−d)` the arrival
  fraction of source `A−d` (from `arc_spread_k`) and `φ_{d,b}` the lag-`d` density
  resolved against `A`'s own blocks. A fixed, release-independent coefficient;
  reduces to `φ_1` when `τ` < one stage length.
- **Single-`ρ` residual is an inherent bound.** Block-agnostic state ⟹ one
  aggregated maturing bucket ⟹ one fixed `ρ`. The exact per-block split is a convex
  combination of the `φ_d` with _release-dependent_ weights, which no fixed
  coefficient can match; block-resolving the buckets is forbidden. A documented
  contract, same category as the WTT-3 terminal drop.
- **Heterogeneous-travel-time confluence into a chronological plant** stays a
  guarded reject (the `non-regular-stage-length-support.md` WTT-2 item), unchanged.
  Parallel confluence is fine.

### 2.4 Implementation surface

Generalize `resolve_delivery` to split against each arrival stage's own blocks;
add a setup precompute of the per-`(arc, chronological arrival stage)`
arrival-frame `ρ` (blend over source lags, block-mode-independent); reduce
`resolve_chrono_arrival_density` to a lookup. No state-layout change (buckets stay
per-depth) — a pure coefficient refinement on the single `b_1^in` column, so
chronological water parity digests move (re-baseline) while parallel cases stay
byte-identical.

## 3. Anticipated dispatch

Anticipated dispatch models GNL-style (LNG) advance commitment: a thermal
generation level chosen `K` stages before delivery. It rides the same in-LP ring
as water travel time (§0), but it carries a **point commitment**, not a spreadable
volume — so its non-regular-stage surface is different in kind. Water's is a
within-stage density (§2); anticipated's is the multiplicity of deliveries
(fan-out) and, above all, the horizon boundary (the GNL cost-to-go coupling).

### 3.1 The ring, and how it differs from water

**State.** `anticipated_slots_out` (outgoing / identity) and `anticipated_state`
(incoming / pinned), both `A · k_max`, plant-minor slot-major, inside
`[N(1+L)+B, n_state)` (`lp/indexer/state_layout.rs`). Block-count-independent, and
the **cut projection covers it** — the FCF carries coefficients on the commitment
state, which §3.3 depends on.

**Transition (definition) rows.** A shift + a deposit, exactly like water's bucket
ring: `slot_d^out − slot_{d+1}^in = 0` (interior shift) and
`slot_{K−1}^out = decision_col` (deposit at the decision stage). Propagated by
copy-outgoing.

**Delivery.** At ring slot 0 the committed level binds the plant's generation via
the **fishing constraint** — `Σ_blk gen_blk · h_blk == committed · H` — a single
stage-summed equality, not a per-block additive term.

**Resolver.** `resolve_point` / `AnticipatedResolution` (`lead_time/mod.rs`)
produce the delivery-anchored decider `c(m)`, the decision sets `C(t)`, the ring
depth `k_max`, and the fan-out width `max_fanout`.

The family resemblance to water is real (one ring, block-agnostic state,
shift+deposit, delivery at the shallow slot), but the roles invert:

| Dimension              | Water travel time                                                          | Anticipated dispatch                                                                                          |
| ---------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Quantity on the ring   | continuous volume (hm³)                                                    | commitment level (MW)                                                                                         |
| Source / deposit       | a _fraction_ of a release, `k_d`-spread + `χ/κ` block-resolved             | a fresh **decision column** the LP chooses, one per delivery                                                  |
| Spread across stages   | yes — `k_d`, mass-conserving                                               | no — one delivery stage (fan-out is the analogue, guarded)                                                    |
| Delivery role          | **additive source** on the water balance (`−ρ_b` chrono / `−1.0` parallel) | **equality constraint** pinning generation to slot 0                                                          |
| Block-mode sensitivity | `ρ` is chronological-only (WTT-1 lives there)                              | fishing sums over blocks — block-mode-agnostic, no density                                                    |
| Horizon past the edge  | **lossy** drop (WTT-3, protected)                                          | **conditionally** loss-free (§3.3): lossless under a zero terminal, but priced under an injected boundary FCF |
| Many-into-one          | confluence (arcs aggregate in state); heterogeneous `t_v` rejected (WTT-2) | fan-out (one decision → many deliveries); rejected (AD-1)                                                     |

The block-mode asymmetry is the same lens as §2: water _delivery is additive on the
per-block balance_, so where-in-the-stage matters; anticipated _delivery is one
stage-summed equality_, so block mode is irrelevant to it — anticipated has no
density to resolve, only _which stage_ (`c(m)`), already calendar-exact.

### 3.2 The gaps

- **AD-1 — fan-out.** A coarse decision stage anchoring several finer deliveries
  (`|C(t)| > 1`). The training LP is **built** — the column geometry reserves
  `col_anticipated_decision_start + j·n_anticipated + local_idx` for
  `j in 0..max_fanout`, and the deposit/fishing construction loops
  `for (j, delivery_stage) in genuine_decisions_at(t)` — but it is **never
  exercised** (setup rejects `max_fanout > 1`), so those `j > 0` paths are
  unverified. The one hard blocker is the **output extractor**
  (`compute_anticipated_decision_mw` asserts a single decision per stage). Fan-out
  requires a **coarse-decision → fine-delivery** calendar (the _reverse_ of DECOMP,
  fine-early → coarse-late), so `|C(t)| ≤ 1` and fan-out never occurs in a DECOMP
  study — a genuine edge case.
- **AD-2 — `LeadTime` physical-horizon reject.** `check_anticipated_thermals`
  rejects `LeadStages K > n_stages` but has no `LeadTime` analogue
  (`TODO(anticipated-physical-horizon-gate)`); a `LeadTime` Δ past the whole
  horizon is unvalidated (benign: resolves to an all-past decider).
- **AD-3 — single-slot decision resolver.** `resolve_anticipated_decision`
  (`generic_constraints.rs`) resolves a generic-constraint `AnticipatedDecision`
  reference to `j = 0`; fine while fan-out is rejected. Sub-note of AD-1.
- **AD-4 — GNL post-horizon boundary signaling.** The horizon-boundary item — §3.3.

### 3.3 The GNL post-horizon boundary condition (AD-4)

**What DECOMP does.** GNL thermals decide dispatch `K` intervals ahead (`dadgnl`
`NL` lag, 1–2). The `GS` register declares the week count of the month(s)
**after** the study horizon precisely so DECOMP can **signal** GNL dispatch for
post-horizon delivery. The commitment is a **state variable**, and the terminal
cost-to-go (from NEWAVE) prices it: the discounted future generation cost
`c_i/(1+β)^K · GT_i^{T+K}` appears **in the final subproblem's objective**, "not
constrained by demand equations within the horizon — only by capacity and
anticipation rules." So a post-horizon commitment is pinned by the **boundary
condition**, not by an in-horizon demand row.

**Why this is XS-1, not a new feature.** Three facts line up: (1) DECOMP prices the
commitment _state_ via the terminal FCF; (2) in cobre the anticipated commitment is
_state_, and cuts project onto the ring — so a cut pool already carries a
coefficient on an in-flight commitment; (3) the XS-1 seam injects a coarse study's
cost-to-go as the current study's boundary condition. Put together: a boundary FCF
injected from a coarse (monthly) source study **carries anticipated-ring
coefficients**, which price — and so let the fine (weekly) stages decide — a GNL
dispatch delivered past the current horizon. The `EntitySlot.delivery_anchor` we
already emit aligns the ring slots across the two resolutions by absolute
year-month.

**The recipe (the operator's mental model).** Run the monthly study; extract its
month-2 cost-to-go (the cut pool at the month-2/3 boundary); build a
weekly-first + monthly study and inject that FCF as the boundary condition. With a
2-interval GNL lead, a commitment decided in a month-1 weekly stage delivers past
the horizon; the injected FCF's coefficient on that ring slot values it, so the LP
decides it.

**Why the loss-free gate is conditional.** `is_anticipated_decision_active_for_delivery`'s
strict `stage_idx + K_i < n_stages` drop is exactly correct under a **finite,
zero-terminal** horizon — a post-horizon commitment is worth zero, so dropping it
and creating-then-zeroing it coincide. Under an **injected boundary FCF**, the drop
silently zeroes the very slot the FCF would price. So the gate must be **relaxed at
the boundary in boundary-FCF mode** — the ending ring must hold the in-flight
post-horizon commitments.

**Built vs needed.**

- _Built:_ commitment-as-state + cut projection onto the ring; the
  `EntitySlot.delivery_anchor` emission.
- _Needed (the XS-1 consumer + one new piece):_ (1) the boundary-cut load-path
  re-index keyed on `(thermal_id, delivery_anchor)` (still positional /
  `state_dimension`-gated today); (2) the **conditional loss-free-gate relaxation**
  under boundary-FCF mode; (3) two correctness checks — the discount `1/(1+β)^K`
  must survive the resolution crossing, and the coarse source study must itself
  model the GNL anticipation so its FCF actually prices the commitment state.

### 3.4 Design decisions

- **AD-1 (fan-out): not supported now — harden the guard + TODO.** Keep the
  `max_fanout > 1` reject; add a durable `TODO(anticipated-fanout-output)` at the
  guard and at `compute_anticipated_decision_mw` (the coupled sites). Do not claim
  the training LP fan-out path works until a hand-derived fan-out case verifies it.
  Revisit only for a real coarse→fine anticipated study.
- **AD-2: add the `LeadTime` horizon-exceed validation** mirroring `LeadStages`;
  drop the TODO. Independent of everything else.
- **AD-3:** sub-note of AD-1; leave `j = 0` while fan-out is rejected. If fan-out
  is ever built, decide the reference policy (reject a fanning plant, or name a
  delivery).
- **AD-4: via XS-1 + a conditional gate relaxation.** Not a standalone build. The
  loss-free-horizon contract becomes **conditional on the terminal mode** (finite ⇒
  drop; injected boundary FCF ⇒ relax and let the FCF price the ending ring). Land
  it with XS-1, not before.

### 3.5 Implementation surface

AD-2 is a self-contained `cobre-io` validation. AD-1 is a guard-hardening + a
durable TODO (no LP change). AD-4 is the anticipated-ring slice of the XS-1
consumer work (`policy_load.rs` anchor-keyed re-index) plus the conditional gate
relaxation in `state_layout.rs`; it must land with XS-1 and re-baselines nothing
until a boundary-FCF study is actually run.

---

## 4. Implementation roadmap

The design decomposes into three groups, ordered by dependency and blast radius.
Each is a coherent unit for an implementation plan; the catalog's
severity-ordered priority list (`non-regular-stage-length-support.md`) is the
complementary "which matters most" view.

### Group 1 — Validation & guard hardening (small, independent, parity-neutral)

Close the latent/unvalidated gaps loudly. All are `cobre-io` / setup validation or
guard changes with **no LP coefficient change**, so no parity movement; ship as one
batch, in any order.

- **AD-2 — `LeadTime` horizon-exceed validation.** Add it in
  `check_anticipated_thermals` mirroring the `LeadStages K > n_stages` reject; drop
  `TODO(anticipated-physical-horizon-gate)`.
- **AD-1 — fan-out guard hardening.** Make the `max_fanout > 1` reject robust and
  clearly worded; add durable `TODO(anticipated-fanout-output)` at the guard AND at
  `compute_anticipated_decision_mw` (the coupled sites). No LP change; do not
  exercise the unverified fan-out path.
- **Inflow-PAR non-monthly guard.** Convert the silent `Weekly | Custom` no-op in
  `lag_transition.rs` (and `compute_recent_observation_seed`'s zero-seed) into a hard
  setup reject, plus the non-monthly `PAR(p)-A` reject and the Custom
  complete-tiling check (§1.3). A non-monthly study then fails loudly instead of
  producing frozen/zero lags. This **subsumes PAR-1** — the zero mid-period seed is
  unreachable behind the reject — until the generic bridge ships.

### Group 2 — Overlap / arrival generalization (structural, parity-moving)

Coefficient refinements that reuse the shared overlap primitive (§0) and move
chronological parity digests (re-baseline on the canonical machine; parallel-only
cases stay byte-identical, as a guard).

- **WTT-1 — arrival-frame `ρ`** (§2.3–2.4): `resolve_delivery` against each arrival
  stage's own blocks + a setup precompute of the k-weighted blend + the consumer
  lookup. Covers all four sender×arrival cells including parallel-sender →
  chronological-arrival.
- **Inflow-PAR generic bridge** (§1.3), when a real weekly study is needed: the
  period-provider-agnostic overlap bridge + day-weighted disaggregation + derived
  `season_id` + `season_ids` wiring + the non-monthly `RecentObservation` seed. This
  lifts the Group-1 non-monthly reject. Larger; gated on demand.

### Group 3 — Cross-study boundary coupling (the convergence epic)

The largest, cross-crate work; delivers coarse→mixed FCF injection AND GNL boundary
signaling. Prerequisite (`EntitySlot.delivery_anchor`) already built.

- **XS-1 — boundary-cut re-index** keyed on `(downstream_id, delivery_anchor)` /
  `(thermal_id, delivery_anchor)` + the boundary-coupling policy (`policy_load.rs`).
- **AD-4 — conditional loss-free-gate relaxation** under boundary-FCF mode (§3.3),
  landing on top of XS-1, plus the discount-across-resolution and
  source-study-models-GNL checks.

### Sequencing

**Group 1 first** — independent, parity-neutral, and it turns every silent
latent-gap into a loud reject (the safest immediate win; the two anticipated items
plus the inflow-PAR guard go together here). **Group 2's WTT-1 next** — self-contained,
closes the one latent-risk-with-no-coverage item, and establishes the arrival-frame
pattern. **Group 3 last** — biggest, cross-crate, and the design already leans on the
delivery-anchor being emitted; it delivers XS-1 and AD-4 together. The inflow-PAR
generic bridge slots into Group 2 whenever a weekly-study demand is real.

**Not scheduled (protect / leave):** WTT-3 (terminal credit — coupled to
non-finite-horizon work, not to XS-1) and WTT-4 (advisories, correct as-is).
