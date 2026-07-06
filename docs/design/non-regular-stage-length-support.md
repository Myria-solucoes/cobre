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

| ID    | Deferred item                                                           | Feature          | Severity                    |
| ----- | ----------------------------------------------------------------------- | ---------------- | --------------------------- |
| PAR-1 | Mid-period PAR seed only under a Monthly season cycle                  | Inflow PAR       | Latent risk (warned)        |
| PAR-2 | Inflow PAR fixed to a monthly clock; IC `season_ids` inert              | Inflow PAR       | Latent risk (modeling)      |
| WTT-1 | Multi-resolution arrival-stage delivery density (resolved: arrival-frame blend) | Water travel     | Protected contract |
| WTT-2 | Chronological confluence with heterogeneous travel times                | Water travel     | Guarded gap                 |
| WTT-3 | Terminal credit for residual in-transit water past the horizon          | Water travel     | Protected contract          |
| WTT-4 | Advisory-only resolution-mismatch signals                               | Water travel     | Guarded (informational)     |
| AD-1  | Anticipated fan-out (one decision → several delivery stages)            | Anticipated      | Guarded gap                 |
| AD-2  | `LeadTime` lead past the study horizon — rejected, mirrors `LeadStages` | Anticipated      | Guarded gap                 |
| AD-3  | Single-slot anticipated decision resolver (fan-out prerequisite)        | Anticipated      | Guarded gap (sub-note AD-1) |
| XS-1  | Cross-study boundary FCF coupling across resolutions                    | Water + Anticip. | Guarded gap                 |

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

**Design:** PAR-1 and PAR-2 share the same period-provider-agnostic overlap
bridge design — see `non-regular-stage-length-design.md` §1. The agreed shape:
one Monthly / Weekly / Custom overlap bridge, an input-free day-weighted
disaggregation, `season_id` derived under the earliest-study-period anchor,
and `RecentObservation`/`season_ids` generalized. Monthly stays the only
implemented path today; a non-Monthly (Weekly/Custom) season cycle combined
with inflow PAR modeling is **not** rejected at setup (see the design finding
below for why a blanket reject was tried and reverted). One related check is
permanent by design, not scoped to the bridge: a non-Monthly cycle supplying
an inflow annual component (PAR(p)-A) is always rejected, because the
annual/long-memory extension is monthly-exclusive by design.

**Design finding — a blanket non-Monthly reject false-positives on
multi-resolution studies.** A setup-time reject for "non-Monthly season cycle
+ live inflow PAR" and a companion "`Custom` season ranges must tile the
repeating annual calendar without gap or overlap" check were both
implemented and then reverted. A `Custom` season cycle is also cobre's
multi-resolution encoding: a study can declare monthly and quarterly season
definitions that intentionally coexist and overlap (a fine-grained monthly
PAR/lag clock layered under a quarterly decomposition envelope), still
combined with inflow PAR. Both reverted checks assumed a `Custom` cycle
always denotes a single, non-overlapping partition of the year, and rejected
the legitimate layered case as if it were the historical mid-period-seed
gap. The fix is not a blanket cycle-type reject or a no-overlap tiling rule;
it is the multi-resolution-aware overlap bridge itself
(`non-regular-stage-length-design.md` §1.3), generalized to recognize a
resolution-layered `Custom` cycle as distinct from an under-specified one
before any reject or seed computation runs against it. That bridge remains
the closing work for PAR-1; until it ships, a non-Monthly cycle combined
with inflow PAR is accepted with only the load-time mid-period-seed-gap
advisory (below).

### PAR-1 — Mid-period PAR seed only under a Monthly season cycle

**Limitation.** The pre-study mid-period lag seed for the PAR accumulator is
computed only for a Monthly season cycle; Weekly / Custom cycles (or a missing
first-stage season id / season map) receive a zero mid-period seed.

**Where.** `crates/cobre-sddp/src/stochastic/lag_transition.rs` —
`compute_recent_observation_seed` returns a zero seed for any non-Monthly cycle
(tagged with a `historical-replay-non-monthly` TODO). Warned, not silent, at
load: `crates/cobre-io/src/validation/semantic/travel_time.rs` —
`check_recent_observations_non_monthly_seed_gap` emits a model-quality warning
naming the cycle.

**Current behavior.** Warned, then zero-seed: setup proceeds, the mid-period
lag seed is zero, and a load-time advisory names the cycle. Diagnosed, but the
produced seed is a wrong (zeroed) value for a legitimately non-Monthly study.

**Reverted attempt.** A setup-time hard reject for this gap — "non-Monthly
cycle + live inflow PAR order" — was implemented and then reverted, along with
a companion `Custom`-cycle tiling check; see the design finding in the §A
introduction above for why both false-positive on a legitimate
multi-resolution study.

**What full support requires.** The multi-resolution-aware, period-provider-
agnostic overlap bridge (`non-regular-stage-length-design.md` §1.3) — one that
distinguishes a resolution-layered `Custom` cycle from an under-specified one
— before either a setup reject or a corrected non-Monthly seed computation can
be built on top of it. Not a fix scoped to the seed alone.

**Severity: Latent risk (warned).** Produces a wrong (zero) seed on
non-Monthly input, but a load-time warning removes the silence. The
generic-bridge work above is the only path to closing it correctly.

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

The water bucket ring is calendar-general within the horizon. The
multi-resolution arrival-stage delivery split is resolved (WTT-1, below); the
remaining gaps are heterogeneous confluence and the accepted terminal-drop
contract.

### WTT-1 — Multi-resolution arrival-stage delivery density (resolved: arrival-frame blend)

**Limitation (historical).** When an in-transit water bucket matured into an
arrival stage whose own block partition differed from the sending stage's, the
delivery across that arrival stage's blocks was not resolved against the
_arrival_ stage's real calendar; it reused a fixed template density derived
from the _sending_ stage's single-lag delivery row (or a duration-weighted
uniform fallback). A genuine multi-resolution arrival target (weekly sender →
monthly receiver, or the reverse) got an approximate within-arrival-stage
block split, and a parallel sender maturing into a chronological arrival stage
always collapsed to the duration-weighted uniform fallback regardless of the
travel time.

**Where.** `crates/cobre-sddp/src/setup/bucket_topology.rs` —
`build_arc_arrival_density` now precomputes, per declared arc and per
chronological arrival stage, the blend of every contributing source stage's
own delivery density resolved directly against the _arrival_ stage's own
blocks (`resolve_arrival_density_at` in `lead_time/mod.rs`), weighted by each
source's stage-clock weight (`build_arc_stage_weights`). This covers a
parallel source reaching a chronological arrival stage exactly like a
chronological one — both contribute a weight/density pair to the same blend.
`crates/cobre-sddp/src/lp/builder/entries.rs` — `resolve_chrono_arrival_density`
now looks up this precomputed table entry verbatim instead of re-deriving a
density from the sending stage's own row; the duration-weighted uniform
density remains only as the fallback for the genuine no-source case (the
study's first stage).

**Current behavior.** Resolved: the delivered split at a multi-resolution
arrival stage is resolved in the arrival stage's own frame, blended over every
source lag that reaches it — including the parallel-sender case, which no
longer collapses to the duration-weighted uniform density. Mass conservation
is unchanged (`Σ arrival_density = 1`, guarded by a `debug_assert` in
`fill_chronological_water_entries`), and a hand-derived regression — two
source stages maturing into one coarser chronological arrival stage at
different lags — cross-checks the closed-form arrival-frame blend against the
delivered LP split, backend-agnostically (HiGHS and CLP).

**The residual: one fixed, block-agnostic density per maturing bucket.** The
arrival-frame blend is still a single density vector per arc per arrival
stage: it carries no per-origin/per-lag memory beyond the weighted blend
itself, and does not vary with which unit of water (by origin block) is
asked. Tracking finer origin-to-arrival-block correlation would grow the
bucket into a per-block state vector whose size scales with the receiving
stage's block count, re-violating the "bucket depth is a pure function of
stage lengths" property the ring depends on. This is an accepted, documented
imprecision — the same category as the WTT-3 terminal-credit bound below —
not scheduled for further work; only the mass-conservation invariant that
keeps it safe must never regress.

**Severity: Protected contract.** Moved out of Latent risk: the
multi-resolution arrival split is now resolved and pinned by a regression.
The remaining single-density residual is a bounded, documented imprecision,
not a silent approximation. It is specific to water: the anticipated ring
delivers a scalar committed power level per delivery stage, not a
block-distributed slug, so it has no within-arrival-stage density analogue
(the anticipated multi-resolution concern is fan-out, AD-1, not density).

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

The anticipated hours-mode migration is complete (§E), and so is the `LeadTime`
horizon-exceed validation (AD-2, below) — it now mirrors the `LeadStages`
`K > n_stages` reject. What remains is fan-out (the anticipated analogue of
confluence), the single-slot decision resolver that fan-out will need, and — at
the horizon boundary — GNL post-horizon commitment signaling (AD-4), which turns
out to be a consumer of XS-1 (§D) rather than a standalone item.

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
`AnticipatedResolution::max_fanout > 1` with `SddpError::Validation` at setup,
naming the fanning plant, so the case never reaches the LP builder.
`AnticipatedResolution::max_fanout` in `crates/cobre-sddp/src/lead_time/mod.rs`
computes the maximum decision-set size.

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

### AD-2 — `LeadTime` lead past the horizon (resolved: validated)

**Limitation (historical).** A `LeadStages` lead is rejected when `K > n_stages`
(the plant can never deliver within the horizon). `LeadTime` had no analogue: a
physical lead Δ that exceeds the whole study horizon was not rejected at
validation.

**Where.** `crates/cobre-io/src/validation/semantic/thermal.rs` —
`check_anticipated_thermals` now sums the study's stage durations into a
`total_horizon_hours` and rejects any `AnticipatedConfig::LeadTime(delta_hours)`
whose `delta_hours` exceeds it, mirroring the `LeadStages` `K > n_stages` reject.
The `TODO(anticipated-physical-horizon-gate)` tag is dropped.

**Current behavior.** Rejected loudly at setup, naming the thermal, the
configured lead, and the study's total horizon in hours. A `LeadTime` lead past
the horizon can no longer resolve silently to an all-past decider and dispatch as
an ordinary thermal with no anticipation.

**Severity: Guarded gap.** Moved out of Latent risk: the case is now caught by a
hard config-time reject, the same category as the `LeadStages` reject it
mirrors. The reject is permanent — a physical lead longer than the study horizon
is never a valid configuration — not a placeholder for a future capability.

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

1. **PAR-1 — non-Monthly mid-period PAR seed.** Produces a wrong (zero) seed;
   masked only by a warning. The fix is the multi-resolution-aware overlap
   bridge (see the design finding in §A), not a narrower reject — a blanket
   reject was tried and reverted because it false-positives on
   multi-resolution `Custom` cycles.

WTT-1 — the multi-resolution arrival-stage delivery density — is done (§B)
and no longer appears in this list. AD-2 — the `LeadTime` horizon-exceed
reject — is done (§C) and no longer appears in this list either.

**Fix as designed — capability, guarded today:**

2. **AD-1 (+ AD-3) — anticipated fan-out.** Setup-rejected; build per-delivery
   deposits/fishing/output with fan-out summation and cross-delivery conservation,
   generalizing the single-slot resolver. Hand-derive every regression's expected
   value (the confluence-summation discipline transfers directly).
3. **XS-1 — coarse-to-mixed boundary coupling.** Highest capability value; the
   `EntitySlot.delivery_anchor` prerequisite has landed, so this is now a
   consumer-side re-index + coupling policy.
4. **WTT-2 — precise heterogeneous-confluence support.** Guarded; only worth it if a
   real study needs mixed-travel-time confluence under chronological blocks.
5. **PAR-2 — sub-monthly PAR clock.** Owned by the PAR boundary; largest blast
   radius, lowest travel-time relevance.

**Never a "fix" — protect / leave:**

6. **WTT-1 — single-density residual.** A protected contract, the same
   category as WTT-3: protect the arrival-frame blend's mass-conservation
   invariant; the residual itself (one fixed, block-agnostic density per
   maturing bucket) is accepted, not scheduled for further work.
7. **WTT-3 — terminal credit.** A protected contract coupled to non-existent
   cyclic-horizon work. Protect the row-omission + zero-terminal-value invariant;
   do not patch it.
8. **WTT-4 — advisories.** Correct as-is.

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
  The residual anticipated gaps are only fan-out (AD-1/AD-3) and the
  boundary-coupling consumer (XS-1) — not the mode itself; the `LeadTime`
  horizon-exceed reject (AD-2) is also done.
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
