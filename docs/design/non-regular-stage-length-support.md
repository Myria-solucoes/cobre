# Non-regular stage-length support — deferred work

## Purpose and scope

Three modeling features consult the stage calendar to place events in time:
the **inflow PAR model**, **water travel time**, and **anticipated thermal
dispatch**. Each shipped with simplifying assumptions about the calendar being
_regular_ — uniform stage length, or a clean weekly-then-monthly nesting. This
document catalogs every case involving **non-regular / heterogeneous /
mixed-resolution / irregular stage lengths** that is currently deferred,
rejected, or only partially handled, so the residual work is visible and
actionable rather than rediscovered. It is organized by feature — inflow PAR
(§A), water travel time (§B), anticipated dispatch (§C) — plus one cross-cutting
seam that couples water and anticipated state across studies (§D), so the answer
to "what is missing for full support of feature X in any kind of study" is one
section.

"Non-regular stage length" here covers: stages of differing duration
(multi-resolution / decomposition — e.g. weekly stages followed by monthly
stages), a lag Δ that is not a clean multiple of stage length, "fine-first" mixed
calendars, month-length irregularity (28/30/31 days), and any code path that
assumes a single uniform stage length.

### The resolver core is already calendar-general

The two resolver entry points — `resolve_spread` and `resolve_point` in
`crates/cobre-sddp/src/lead_time/mod.rs` — and the hour-exact overlap primitive
`window_period_overlaps` in `crates/cobre-core/src/model/temporal/overlap.rs`
consume an arbitrary `stage_lengths_hours: &[f64]` and handle non-uniform and
gapped calendars correctly (covered by `test_non_uniform_stages_depth_three` and
`test_skipped_intermediate_stage_stays_contiguous`). Both the water buckets and
the anticipated ring now consume this shared resolver. **The deferred work below
is almost entirely in the _consumers_** — LP delivery density, fan-out, the
inflow PAR clock, and cross-study boundary coupling — not in the resolver
primitives.

### How to read the severity column

- **Guarded gap** — the unsupported case is caught by a hard config-time reject or
  a load-time warning. Safe: no wrong output, only a missing capability.
- **Latent risk** — a valid input can silently produce an approximate or wrong
  result with no loud signal. These are the dangerous ones.
- **Protected contract** — current behavior is a _documented, bounded imprecision_
  pinned by an invariant (and a named regression). The imprecision is accepted;
  the invariant that keeps it safe must never regress.

## Status at a glance

| ID    | Deferred item                                                        | Feature          | Severity                    |
| ----- | -------------------------------------------------------------------- | ---------------- | --------------------------- |
| PAR-1 | Mid-period PAR seed computed only under a Monthly season cycle       | Inflow PAR       | Latent risk (warned)        |
| PAR-2 | Inflow PAR fixed to a monthly clock; IC `season_ids` inert           | Inflow PAR       | Latent risk (modeling)      |
| WTT-1 | Multi-resolution arrival-stage delivery density (fixed template)     | Water travel     | **Latent risk**             |
| WTT-2 | Chronological confluence with heterogeneous travel times             | Water travel     | Guarded gap                 |
| WTT-3 | Terminal credit for residual in-transit water past the horizon       | Water travel     | Protected contract          |
| WTT-4 | Advisory-only resolution-mismatch signals                            | Water travel     | Guarded (informational)     |
| AD-1  | Anticipated fan-out (one decision → several delivery stages)         | Anticipated      | Guarded gap                 |
| AD-2  | No physical-horizon rejection for a `LeadTime` lead past the horizon | Anticipated      | Latent risk (unvalidated)   |
| AD-3  | Single-slot anticipated decision resolver (fan-out prerequisite)     | Anticipated      | Guarded gap (sub-note AD-1) |
| XS-1  | Cross-study boundary FCF coupling across resolutions                 | Water + Anticip. | Guarded gap                 |

The anticipated hours-mode migration (calendar-anchored `LeadTime`, the in-LP
delivery-anchored ring, deletion of the out-of-LP shift) is **done** and is
recorded under "Already handled" (§E), not as a gap — its loss-free horizon
behavior is the one place anticipated is _stronger_ than water (contrast AD vs.
WTT-3 below).

---

## §A — Inflow PAR models

The inflow PAR model is the feature least advanced toward calendar-generality.
Both items below are owned by the stochastic / PAR layer, not the travel-time or
anticipated work, and both are silent-ish (one warned, one purely modeling).

**Design:** PAR-1 and PAR-2 now have a worked design — see
`non-regular-stage-length-design.md` §1. The agreed shape: one
period-provider-agnostic overlap bridge (Monthly / Weekly / Custom), an
input-free day-weighted disaggregation, `season_id` derived under the
earliest-study-period anchor, and `RecentObservation`/`season_ids` generalized.
Monthly stays the only implemented path; Weekly/Custom are hard-rejected at setup
until shipped (replacing today's silent no-op).

### PAR-1 — Mid-period PAR seed only under a Monthly season cycle

**Limitation.** The pre-study mid-period lag seed for the PAR accumulator is
computed only for a Monthly season cycle; Weekly / Custom cycles (or a missing
first-stage season id / season map) receive a zero mid-period seed.

**Where.** `crates/cobre-sddp/src/stochastic/lag_transition.rs` —
`compute_recent_observation_seed` returns a zero seed for any non-Monthly cycle
(tagged with a `historical-replay-non-monthly` TODO). Warned, not silent, at load:
`crates/cobre-io/src/validation/semantic/travel_time.rs` —
`check_recent_observations_non_monthly_seed_gap` emits a model-quality warning
naming the cycle.

**Current behavior.** Warned, then zero-seed: setup proceeds, the mid-period lag
seed is zero, and a load-time advisory names the cycle. Diagnosed, but the produced
seed is a wrong (zeroed) value for a legitimately non-Monthly study.

**What full support requires.** Generalize the season-anchored month-hours math
(`month_total_hours`, `find_season_year_monthly`) to arbitrary cycle lengths keyed
off the cycle type. Small, self-contained blast radius in `lag_transition.rs`
(plus tests, and dropping the warning).

**Severity: Latent risk (warned).** Produces a wrong (zero) seed on non-Monthly
input, but a hard-wired load warning removes the silence. The fix is cheap.

### PAR-2 — Inflow PAR fixed to a monthly clock; IC `season_ids` inert

**Limitation.** `past_inflows` is an ordinal monthly PAR lag, and the PAR(p) model
is always fitted on the monthly clock regardless of study stage length. A weekly /
sub-monthly study still carries a monthly PAR. The `season_ids` resolution tag on
the IC history fields is parsed and length-validated but read by no solver code — a
half-built resolution tag.

**Where.** `crates/cobre-stochastic/src/sampling/external.rs` —
`standardize_external_inflow` advances the past-lag buffer on the monthly PAR clock
and never reads pre-study durations. `crates/cobre-sddp/src/setup/mod.rs` —
`build_initial_state` places `past_inflows` positionally as monthly lags.

**Current behavior.** Silently monthly. A weekly study whose inflows are genuinely
sub-monthly is modeled on a monthly PAR — a modeling limitation, not a crash.
Decomposition-style studies escape it because their weekly stages still carry
monthly season ids.

**What full support requires.** A resolution-aware PAR clock (sub-monthly fitting
and transitions) plus wiring the currently-inert `season_ids` tag. Large, and owned
outside the travel-time and anticipated features.

**Severity: Latent risk (modeling).** Silently monthly, but a known scope boundary
rather than a defect introduced by the calendar work. Flag it so a weekly-inflow
study is not silently trusted.

---

## §B — Water travel time

The water bucket ring is calendar-general within the horizon; the gaps are in
multi-resolution _delivery density_, heterogeneous confluence, and the accepted
terminal-drop contract.

### WTT-1 — Multi-resolution arrival-stage delivery density

**Limitation.** When an in-transit water bucket matures into an arrival stage whose
own block partition differs from the sending stage's, the delivery across that
arrival stage's blocks is not resolved against the _arrival_ stage's real calendar;
it reuses a fixed template density derived from the _sending_ stage's single-lag
delivery row (or a duration-weighted uniform fallback). A genuine multi-resolution
arrival target (weekly sender → monthly receiver, or the reverse) gets an
approximate within-arrival-stage block split.

**Where.** `crates/cobre-sddp/src/lead_time/mod.rs` — `resolve_spread` /
`resolve_delivery` split every reached arrival stage against the _anchor's_ block
partition, not the target stage's. `crates/cobre-sddp/src/lp/builder/entries.rs` —
`resolve_chrono_arrival_density` re-reads the sending stage's own delivery row
fresh each stage (no per-origin/per-lag memory), with a duration-weighted uniform
fallback for the study's first stage or a parallel→chronological transition.

**Current behavior.** Wrong-but-bounded and compiles: total mass is conserved
(`Σ ρ = 1`, guarded by a `debug_assert` in `fill_chronological_water_entries`), so
the LP stays feasible; only the _intra-arrival-stage block distribution_ of the
delivered slug is approximate. No crash, no stage-level bound error.

**What full support requires.** Extend `resolve_spread` to accept a
per-arrival-stage block partition; thread each arrival stage's own `blocks` into
`resolve_delivery`; replace the fixed-template read in
`resolve_chrono_arrival_density` with a per-target-stage density. Blast radius:
`lead_time/mod.rs` (`resolve_spread`/`resolve_delivery` signatures and the spread
resolution's arrival-density shape), `setup/bucket_topology.rs`
(`build_arc_spread_chrono`), `lp/builder/entries.rs`
(`resolve_chrono_arrival_density`, `fill_chronological_water_entries`), plus a new
decomposition-arrival regression case.

**Severity: Latent risk.** The only item that is silently approximate on a real
multi-resolution input _and_ has no regression covering it. Bounded (mass-
conserving), but the coverage gap is the real hazard. It is specific to water: the
anticipated ring delivers a scalar committed power level per delivery stage, not a
block-distributed slug, so it has no within-arrival-stage density analogue (the
anticipated multi-resolution concern is fan-out, AD-1, not density).

### WTT-2 — Chronological confluence with heterogeneous travel times

**Limitation.** Two or more declared arcs feeding one downstream plant with
_differing_ `travel_time_hours`, while any study stage is chronological, is
unsupported.

**Where.** `crates/cobre-io/src/validation/semantic/travel_time.rs` —
`check_chronological_confluence_heterogeneous_travel_time` rejects with a
"not implemented" error. A defensive `debug_assert` in
`resolve_chrono_arrival_density` (`entries.rs`) mirrors it downstream.

**Current behavior.** Rejected loudly at config time. The check deliberately
over-rejects a _superset_: it rejects every chronological study stage when arcs
disagree, not only the stages whose per-arc spread actually diverges, because the
infrastructure (I/O) crate has no access to the downstream per-arc spread
computation needed to check precisely.

**What full support requires.** Per-arc, per-stage-pair spread resolution to detect
_actual_ disagreement, plus a merged multi-arc arrival density in
`resolve_chrono_arrival_density` (which today asserts all contributing arcs agree).
Medium blast radius: a validation-precision refactor plus the arrival-density
merge.

**Severity: Guarded gap.** Hard reject; pure feature gap, no wrong output.

### WTT-3 — Terminal credit for residual in-transit water past the horizon

**Limitation.** In-transit water whose target stage falls past the finite horizon
is dropped, not credited a terminal value, so end-of-horizon upstream release is
under-valued. A sub-assumption in depth sizing pads the calendar with copies of the
_trailing_ stage's duration for the dropped tail (i.e. assumes the horizon would
continue at the last stage's cadence).

**Where.** `crates/cobre-sddp/src/setup/bucket_topology.rs` — `horizon_cap_active`
(caps active lag at `n_stages − 1 − stage`) and `extend_for_resolution` (sizes
depth for the beyond-horizon tail but does not credit it).
`crates/cobre-sddp/src/lp/builder/layout.rs` — `build_transit_bucket_row_pos` gates
per-stage row emission on the cap. `crates/cobre-sddp/src/lp/builder/columns.rs` —
`fill_transit_bucket_columns` freezes a masked slot's outgoing column to `[0, 0]`.
The safety hinge is `crates/cobre-sddp/src/horizon_mode.rs`: only
`HorizonMode::Finite` is implemented, and the drop is provably harmless **only**
because a finite horizon's zero terminal value makes a masked slot's cut
coefficient structurally zero.

**Current behavior.** Wrong-but-documented and bounded: under-values end-of-horizon
release, but is correct _by construction_ under a finite horizon. The drop is a row
omission, not a silent zeroing elsewhere. This is water's **lossy** terminal
handling — a real k-weighted release _is_ discarded. Anticipated dispatch does the
opposite (§C, loss-free): its out-of-horizon decision is never created, so there is
nothing to drop. Do not conflate the two.

**What full support requires.** A terminal value function on residual buckets,
coupled to non-finite-horizon work. If ever built, the two-part safety argument
(row omission **plus** zero-terminal-value inertness) is the contract to _replace_,
not patch. Large blast radius (couples to horizon-mode work).

**Severity: Protected contract.** Pinned by an invariant and named regressions; the
row-omission + zero-terminal-value inertness pair must never regress into a silent
stale-row write. The under-valuation itself is accepted and documented.

### WTT-4 — Advisory-only resolution-mismatch signals

**Limitation.** A travel time that is tiny relative to a coarse stage, or that
exceeds the remaining horizon from some stage onward, is advised but not modeled
specially — both are resolution-mismatch symptoms (a fine lag on a coarse
calendar).

**Where.** `crates/cobre-io/src/validation/semantic/travel_time.rs` —
`check_negligible_ratio` and `check_horizon_inertness`, both model-quality
warnings.

**Current behavior.** Warned; setup proceeds with the exact (possibly economically
inert) modeling. No wrong output — depth sizing stays capped by the remaining
horizon.

**Severity: Guarded (informational).** Intentional advisories, listed for
completeness because they are where a coarse/fine mismatch surfaces.

---

## §C — Anticipated dispatch

The anticipated hours-mode migration is complete (§E). What remains is fan-out
(the anticipated analogue of confluence), one missing `LeadTime` horizon
validation, the single-slot decision resolver that fan-out will need, and — at the
horizon boundary — GNL post-horizon commitment signaling (AD-4), which turns out to
be a consumer of XS-1 (§D) rather than a standalone item.

### The loss-free horizon property is CONDITIONAL on the terminal mode

`is_anticipated_decision_active_for_delivery`
(`crates/cobre-sddp/src/lp/indexer/state_layout.rs`) gates a decision on the strict
clause `stage_idx + K_i < n_stages`. A commitment whose delivery stage would fall
at or past `n_stages` is **never created** — the decision column, its deposit row,
and its fishing row are all absent, not zeroed-after-the-fact. Under a **finite,
zero-terminal horizon** this is provably correct and lossless: with nothing to value
the ending anticipated ring, a boundary commitment would optimize to zero, so
dropping it and creating-then-zeroing it are the same answer (strictly stronger than
water's WTT-3, which discards a real weighted release under a bounded contract).

But the property is **conditional on that zero terminal value.** Under an injected
boundary future-cost function (XS-1) that prices the ending anticipated ring — the
DECOMP GNL boundary condition — the drop is no longer correct: the FCF's coefficient
on the in-flight commitment is exactly what pins a post-horizon decision, and the
strict gate would silently zero it. So the anticipated ring DOES inherit an
end-of-horizon item after all: AD-4, the conditional relaxation of this gate under a
boundary FCF (see §D XS-1 and `non-regular-stage-length-design.md` §3).

### AD-1 — Anticipated fan-out (one decision feeding several delivery stages)

**Limitation.** In `LeadTime` mode on a coarsening calendar, one coarse decision
stage can anchor several finer delivery stages (`|C(t)| > 1`) — the anticipated
mirror of water confluence. This "fan-out" is not built: a single plant depositing
into multiple future delivery stages has no per-delivery deposit column, no
per-delivery fishing coupling, and no per-delivery-stage output.

**Where.** `crates/cobre-sddp/src/setup/mod.rs` — `build_wired_indexer` rejects
`AnticipatedResolution::max_fanout > 1` with `SddpError::Validation` at setup, so
the case never reaches the LP builder. `AnticipatedResolution::max_fanout` in
`crates/cobre-sddp/src/lead_time/mod.rs` computes the maximum decision-set size.

**Current behavior.** Rejected loudly at setup. Single-decider `LeadTime`
(`|C(t)| = 1`, the common case, and every `LeadStages` config) is fully supported
end-to-end; only the multi-delivery fan-out is gated off.

**What full support requires.** Per-delivery deposit rows/columns and per-delivery
fishing coupling in the ring; a fan-out summation discipline (never a single-term
regression) and cross-delivery conservation assertions (sum across delivery stages,
never total-cost-only); and per-delivery-stage anticipated output. The single-slot
decision resolver (AD-3) is a prerequisite. If fan-out introduces a cross-consumer
heterogeneity case, budget for the same crate-boundary-forces-a-superset-reject
trade-off as WTT-2 rather than threading per-arc computation into the I/O crate.

**Severity: Guarded gap.** Hard reject at setup; pure capability gap, no wrong
output. Arises only in `LeadTime` mode on a coarsening calendar.

### AD-2 — No physical-horizon rejection for a `LeadTime` lead past the horizon

**Limitation.** A `LeadStages` lead is rejected when `K > n_stages` (the plant can
never deliver within the horizon). `LeadTime` has no analogue: a physical lead Δ
that exceeds the whole study horizon is not rejected at validation.

**Where.** `crates/cobre-io/src/validation/semantic/thermal.rs` —
`check_anticipated_thermals`, at the `cfg.lead_stages()` fall-through carrying the
`TODO(anticipated-physical-horizon-gate)` tag: the `K > n_stages` rejection runs
for `LeadStages` only, and `LeadTime` falls through it.

**Current behavior.** Unvalidated, but not obviously wrong: a Δ past the horizon
resolves (via `resolve_point`) to an all-past decider, so no decision is genuine and
the plant dispatches as an ordinary thermal — no anticipation, no crash. The gap is
the missing loud signal, not a known wrong output.

**What full support requires.** A calendar-derived horizon-exceed check for
`LeadTime` mirroring the `LeadStages` `K > n_stages` reject (the resolver already
exposes the depth needed), then drop the TODO.

**Severity: Latent risk (unvalidated).** No loud signal on a misconfigured
`LeadTime` lead; behavior is currently benign but unpinned. Cheap, self-contained.

### AD-3 — Single-slot anticipated decision resolver

**Limitation.** The decision-column resolver returns exactly one column per
anticipated plant, so it cannot express a plant that deposits into more than one
delivery stage (fan-out, AD-1).

**Where.** `crates/cobre-sddp/src/lp/generic_constraints.rs` —
`resolve_anticipated_decision` returns `vec![(anticipated_decision_start +
local_idx, 1.0)]` (a single slot-0 entry) or an empty vec.

**Current behavior.** Correct and complete for the single-decider case that ships;
structurally unable to represent fan-out. It is the LP-side prerequisite AD-1 must
generalize.

**Severity: Guarded gap (sub-note of AD-1).** Not independently reachable — the
fan-out setup reject (AD-1) keeps every shipped study single-slot.

### AD-4 — GNL post-horizon commitment signaling at the boundary

**Limitation.** DECOMP lets GNL (LNG) thermals decide anticipated dispatch for
delivery weeks BEYOND the horizon end (the `dadgnl` `GS` register declares the
post-horizon month's week count for exactly this), valued by the NEWAVE boundary
cost-to-go. cobre's loss-free gate drops any commitment whose delivery is past
`n_stages`, so it cannot signal a post-horizon GNL commitment.

**Where.** The gate is `is_anticipated_decision_active_for_delivery`
(`state_layout.rs`, strict `stage_idx + K_i < n_stages`). The valuation path is the
boundary-FCF injection, XS-1 (§D): the anticipated commitment is already state
(`anticipated_slots_out` / `anticipated_state`, covered by the cut projection), so an
injected FCF carrying anticipated-ring coefficients would price it.

**Current behavior.** Correct under a finite / zero-terminal horizon (a post-horizon
commitment is worth zero, so the drop is exact). Missing only once a boundary FCF
values the ending ring.

**What full support requires.** NOT a standalone feature — it is the anticipated-ring
consumer of XS-1 plus one new piece: a **conditional relaxation of the loss-free
gate** under boundary-FCF mode, so the ending ring holds the in-flight post-horizon
commitments for the injected FCF to price; the XS-1 anchor-keyed re-index aligns the
source study's ring to the current one; and the discount `1/(1+β)^K` must survive the
resolution crossing. Fully worked in `non-regular-stage-length-design.md` §3.

**Severity: Guarded gap (coupled to XS-1 + terminal-mode).** No wrong output today
(finite horizon → the dropped commitment is genuinely zero-valued); becomes live only
with an injected boundary FCF.

---

## §D — Cross-cutting: cross-study boundary FCF coupling (XS-1)

This seam couples **both** water bucket state and anticipated ring state across
studies, so it belongs to neither feature alone.

**Limitation.** Injecting a coarse-resolution terminal future-cost function (e.g. a
monthly upstream study) into a mixed-resolution current study (e.g. weekly+monthly)
has no working path when the state dimensions differ (the bucket block size, or a
differently-resolved anticipated ring). Boundary cuts cannot yet be re-indexed
across resolutions by the consumer.

**Prerequisite — DONE.** `EntitySlot` now carries a canonical absolute
`delivery_anchor` (year-month), emitted by the manifest builder and stable across
resolutions: `crates/cobre-sddp/src/policy/policy_export.rs` sets `delivery_anchor`
per bucket lag (`anchor_at`) and per anticipated slot (the delivery stage's
year-month), and `crates/cobre-io/src/output/policy/records.rs` owns the wire field
plus `ENTITY_SLOT_DELIVERY_ANCHOR_SENTINEL` (forward-compatible: a reader of a
buffer predating the field yields the sentinel). Round-trip and delivery-anchor
manifest tests pin it.

**Consumer — STILL DEFERRED.** `crates/cobre-sddp/src/policy/policy_load.rs` —
`validate_policy_load` / the boundary-cut load path still key on `state_dimension`
equality and positional slot identity; the module docs name the anchor-keyed
"rules that replace today's positional cut copy" as the future work, not present
behavior. So the anchor is _emitted_ but not yet _consumed_ for re-indexing.

**Current behavior.** The coarse-to-mixed workflow is rejected loudly (dimension /
positional mismatch). Safe, but the real workflow cannot run.

**What full support requires.**

1. Rewrite the boundary-cut load path to length-tolerant, family-aware re-indexing
   keyed on `(downstream_id, delivery_anchor)` and `(thermal_id, delivery_anchor)`
   — the anchor it now has.
2. Add a boundary-coupling policy (buckets: reject / zero-fill / redistribute;
   anticipated: align-by-delivery-anchor, drop-out-of-window-with-warning) carried
   on the boundary-injection load kind.

Large, cross-crate consumer-side blast radius; the dual-owned wire change is
already paid.

**Named consumer — GNL post-horizon signaling (AD-4).** The DECOMP GNL boundary
condition is a direct consumer of this seam. Because the anticipated commitment is
state and the cut projection covers the ring, a boundary FCF injected from a coarse
(monthly) source study carries a coefficient on the in-flight commitment, which
prices — and so lets the fine (weekly) stages decide — a GNL dispatch delivered past
the current horizon. It needs the anchor-keyed anticipated re-index above PLUS a
conditional relaxation of the loss-free gate (AD-4, §C); the
`(thermal_id, delivery_anchor)` key aligns a post-horizon delivery by its absolute
year-month across the two resolutions.

**Severity: Guarded gap.** Hard-rejected today; the highest-value deferred
capability now that its wire prerequisite has landed.

---

## Priority ordering

**Fix first — latent risk, thin or absent coverage on real inputs:**

1. **WTT-1 — multi-resolution arrival-stage delivery density.** The only item
   silently approximate on a real input with no regression. Add a
   decomposition-arrival regression and decide upgrade-vs-accept.
2. **PAR-1 — non-Monthly mid-period PAR seed.** Produces a wrong (zero) seed;
   cheap, self-contained; currently masked only by a warning.
3. **AD-2 — `LeadTime` physical-horizon reject.** Cheap validation gap; removes the
   one unvalidated anticipated `LeadTime` case (drop the TODO).

**Fix as designed — capability, guarded today:**

4. **AD-1 (+ AD-3) — anticipated fan-out.** Setup-rejected; build per-delivery
   deposits/fishing/output with fan-out summation and cross-delivery conservation,
   generalizing the single-slot resolver. Hand-derive every regression's expected
   value (the confluence-summation discipline transfers directly).
5. **XS-1 — coarse-to-mixed boundary coupling.** Highest capability value; the
   `EntitySlot.delivery_anchor` prerequisite has landed, so this is now a
   consumer-side re-index + coupling policy.
6. **WTT-2 — precise heterogeneous-confluence support.** Guarded; only worth it if a
   real study needs mixed-travel-time confluence under chronological blocks.
7. **PAR-2 — sub-monthly PAR clock.** Owned by the PAR boundary; largest blast
   radius, lowest travel-time relevance.

**Never a "fix" — protect / leave:**

8. **WTT-3 — terminal credit.** A protected contract coupled to non-existent
   cyclic-horizon work. Protect the row-omission + zero-terminal-value invariant;
   do not patch it.
9. **WTT-4 — advisories.** Correct as-is.

---

## §E — Already handled — not gaps

To avoid re-filing resolved work as deferred:

- **Anticipated hours mode + in-LP delivery-anchored ring.** Done. `AnticipatedConfig`
  is now a two-mode enum — `LeadStages(u32)` (stage count, calendar-blind) and
  `LeadTime(f64)` (physical hours, delivery-anchored, same clock as a water arc's
  `travel_time_hours`) — in `crates/cobre-core/src/entities/thermal.rs`. Setup
  consumes `resolve_point`; the out-of-LP `shift_anticipated_state` is **deleted**
  and replaced by the in-LP ring (per-slot outgoing columns + definition rows,
  generalizing the water bucket ring). A single-decider `LeadTime` study solves
  end-to-end through both the in-code path and the on-disk `run_pipeline` path.
  The residual anticipated gaps are only fan-out (AD-1/AD-3), the `LeadTime`
  horizon reject (AD-2), and the boundary-coupling consumer (XS-1) — not the mode
  itself.
- **Anticipated end-of-horizon handling is loss-free.** The strict
  `stage_idx + K_i < n_stages` gate means an out-of-horizon decision is never
  created — there is no anticipated analogue of water's WTT-3 terminal drop (see
  §C). This is a resolved property, not a deferred imprecision.
- **`EntitySlot` delivery-calendar anchor.** The wire prerequisite for XS-1 is
  emitted and round-tripped (see §D); only the load-side re-index consumes it, and
  that is the remaining XS-1 work.
- **Pre-study defluence weekly/monthly truncation.** Resolved. The initial-
  condition defluence history was migrated to calendar-windowed records with a hard
  coverage gate (`check_defluence_coverage` in
  `crates/cobre-io/src/validation/semantic/travel_time.rs`), and the bucket seed now
  overlaps each window directly (`build_initial_transit_bucket_state` in
  `crates/cobre-sddp/src/setup/bucket_seed.rs`). The old pre-study-stage positional
  walk and the silent truncation are gone; **there is no `past_inflows` fallback**.
  A residual (inherent, not a bug): supplying only a single coarse defluence window
  yields a uniform-rate smear over that window's width — the remedy is finer input
  windows, not code.
- **Pre-study anticipated commitments.** The anticipated commitment history
  (`past_anticipated_commitments`) is validated against a calendar-derived count of
  pre-study-committed delivery stages (`required_anticipated_commitment_count`) for
  both lead modes — a hard coverage gate, no fallback — mirroring the defluence
  coverage rule above.
- **The resolver primitives being non-uniform-unsafe.** `resolve_point`,
  `resolve_spread`, and `window_period_overlaps` are calendar-general and
  unit-tested on non-uniform and gapped calendars. The deferred work is in
  consumers, never the primitive.
- **Depth-padding cadence.** `extend_for_resolution` only sizes depth and k-weights
  for the beyond-horizon tail that the horizon cap drops (WTT-3); within-horizon
  k-weights use the real calendar. It is a sub-note of the terminal-drop contract,
  not an independent bug.
