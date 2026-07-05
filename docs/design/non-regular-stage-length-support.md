# Non-regular stage-length support — deferred work

## Purpose and scope

The water-travel-time feature and the shared calendar-anchored lead-time resolver
made several simplifying assumptions about the stage calendar being _regular_ —
either uniform stage length, or a clean weekly-then-monthly nesting. This document
catalogs every case involving **non-regular / heterogeneous / mixed-resolution /
irregular stage lengths** that is currently deferred, rejected, or only partially
handled, so the residual work is visible and actionable rather than rediscovered.

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
`test_skipped_intermediate_stage_stays_contiguous`). **The deferred work below is
almost entirely in the _consumers_** — LP delivery-row density, the anticipated
lead-time refactor, cross-study boundary coupling, and the inflow PAR clock — not
in the resolver primitives.

### How to read the severity column

- **Guarded gap** — the unsupported case is caught by a hard config-time reject or
  a load-time warning. Safe: no wrong output, only a missing capability.
- **Latent risk** — a valid input can silently produce an approximate or wrong
  result with no loud signal. These are the dangerous ones.
- **Protected contract** — current behavior is a _documented, bounded imprecision_
  pinned by an invariant (and a named regression). The imprecision is accepted;
  the invariant that keeps it safe must never regress.

## Summary

| #   | Deferred item                                                                | Severity                            |
| --- | ---------------------------------------------------------------------------- | ----------------------------------- |
| 1   | Multi-resolution arrival-stage delivery density (fixed template)             | **Latent risk**                     |
| 2   | Chronological confluence with heterogeneous travel times                     | Guarded gap                         |
| 3   | Terminal credit for residual in-transit water past the horizon               | Protected contract                  |
| 4   | Mid-period PAR seed computed only under a Monthly season cycle               | Latent risk (warned)                |
| 5   | Inflow PAR model fixed to a monthly clock; IC `season_ids` inert             | Latent risk (modeling)              |
| 6   | Anticipated pre-commitment history is stage-count-indexed; no hours mode yet | Guarded gap (activates on refactor) |
| 7   | Cross-study boundary FCF coupling across resolutions (coarse → mixed)        | Guarded gap                         |
| 8   | Advisory-only resolution-mismatch signals (negligible ratio / inert horizon) | Guarded (informational)             |

---

## Item 1 — Multi-resolution arrival-stage delivery density

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

**Why deferred.** Accepted as a v1 simplification with the conservation guard in
place. No current regression exercises a multi-resolution/decomposition arrival
target explicitly (the shipped water cases use uniform or simple mixed calendars).

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
conserving), but the coverage gap is the real hazard — and it re-applies verbatim
to the anticipated lead-time delivery (see "What the refactor inherits").

## Item 2 — Chronological confluence with heterogeneous travel times

**Limitation.** Two or more declared arcs feeding one downstream plant with
_differing_ `travel_time_hours`, while any study stage is chronological, is
unsupported.

**Where.** `crates/cobre-io/src/validation/semantic/travel_time.rs` —
`check_chronological_confluence_heterogeneous_travel_time` rejects with a
"not implemented" error ("chronological confluence with heterogeneous travel times
is unsupported in v1"). A defensive `debug_assert` in
`resolve_chrono_arrival_density` (`entries.rs`) mirrors it downstream.

**Current behavior.** Rejected loudly at config time. The check deliberately
over-rejects a _superset_: it rejects every chronological study stage when arcs
disagree, not only the stages whose per-arc spread actually diverges, because the
infrastructure (I/O) crate has no access to the downstream per-arc spread
computation needed to check precisely.

**Why deferred.** A documented, accepted v1 imprecision driven by the
infrastructure-crate genericity boundary — the precise check would require the
downstream per-arc computation inside (or exposed to) the I/O crate.

**What full support requires.** Per-arc, per-stage-pair spread resolution to detect
_actual_ disagreement, plus a merged multi-arc arrival density in
`resolve_chrono_arrival_density` (which today asserts all contributing arcs agree).
Medium blast radius: a validation-precision refactor plus the arrival-density
merge.

**Severity: Guarded gap.** Hard reject; pure feature gap, no wrong output.

## Item 3 — Terminal credit for residual in-transit water past the horizon

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
omission, not a silent zeroing elsewhere.

**Why deferred.** An explicit non-goal — no terminal value function on residual
buckets — because a meaningful terminal credit only exists once a non-finite
(cyclic / terminal-FCF) horizon mode exists, which is itself out of scope.

**What full support requires.** A terminal value function on residual buckets,
coupled to non-finite-horizon work. If ever built, the two-part safety argument
(row omission **plus** zero-terminal-value inertness) is the contract to _replace_,
not patch. Large blast radius (couples to horizon-mode work).

**Severity: Protected contract.** Pinned by an invariant and named regressions; the
row-omission + zero-terminal-value inertness pair must never regress into a silent
stale-row write. The under-valuation itself is accepted and documented.

## Item 4 — Mid-period PAR seed only under a Monthly season cycle

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

**Why deferred.** Recorded as the `historical-replay-non-monthly` follow-up (the
TODO plus the load warning).

**What full support requires.** Generalize the season-anchored month-hours math
(`month_total_hours`, `find_season_year_monthly`) to arbitrary cycle lengths keyed
off the cycle type. Small, self-contained blast radius in `lag_transition.rs`
(plus tests, and dropping the warning).

**Severity: Latent risk (warned).** Produces a wrong (zero) seed on non-Monthly
input, but a hard-wired load warning removes the silence. The fix is cheap.

## Item 5 — Inflow PAR model fixed to a monthly clock; IC `season_ids` inert

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

**Why deferred.** This is the PAR-ownership boundary: the inflow model stays
monthly by design in v1, and the resolution-aware PAR clock is owned by the
stochastic / PAR layer, not the travel-time feature.

**What full support requires.** A resolution-aware PAR clock (sub-monthly fitting
and transitions) plus wiring the currently-inert `season_ids` tag. Large, and owned
outside the travel-time feature.

**Severity: Latent risk (modeling).** Silently monthly, but a known scope boundary
rather than a defect introduced by the travel-time work. Flag it so a weekly-inflow
study is not silently trusted.

## Item 6 — Anticipated pre-commitment history is stage-count-indexed

**Limitation.** Anticipated-thermal pre-commitment history is stored as one MW
value per early study stage (`values_mw` with length equal to the lead stage
count) — a stage-count ordinal. There is no calendar-anchored (hours-based)
delivery model, so on a non-uniform calendar the stage-count index is
calendar-broken.

**Where.** `crates/cobre-core/src/entities/thermal.rs` — `AnticipatedConfig`
carries only a `lead_stages` stage count. The ring is shifted out-of-LP by
`crates/cobre-sddp/src/stochastic/noise.rs` — `shift_anticipated_state`.

**Current behavior.** The shipped anticipated feature offers **only** the
stage-count mode, which is calendar-blind by construction (its decider takes no
calendar). It is therefore **safe today** — no wrong output — precisely because the
calendar-sensitive hours mode is not yet wired.

**Why deferred.** Sequenced as the anticipated lead-time refactor (after water).
The redesign carries a hard constraint: the pre-commitment history must become a
study-stage-to-delivery-stage ordinal reindex, **not** a date-windowed record like
`past_defluences`, and **not** a silent fallback.

**What full support requires.** Add an hours-based lead time alongside the stage
count, consume `resolve_point` at setup, build the in-LP anticipated ring (deleting
`shift_anticipated_state`), add fan-out for multi-delivery deciders and a
zero-lead diagnostic, and the ordinal-reindex history redesign. This is the whole
of the anticipated lead-time refactor now in progress.

**Severity: Guarded gap today** — only the calendar-blind mode ships. Becomes
correctness-critical the moment the hours mode lands (see below).

## Item 7 — Cross-study boundary FCF coupling across resolutions

**Limitation.** Injecting a coarse-resolution terminal future-cost function (e.g. a
monthly upstream study) into a mixed-resolution current study (e.g. weekly+monthly)
has no working path when the state dimensions differ (the bucket block size, or a
differently-resolved anticipated ring). Boundary cuts cannot be re-indexed across
resolutions because manifest slots carry no delivery-calendar anchor.

**Where.** `crates/cobre-sddp/src/policy/policy_load.rs` — `load_boundary_cuts`
hard-rejects on the state-dimension guard and matches slots positionally.
`crates/cobre-io/src/output/policy/records.rs` — `EntitySlot` is
`(entity_type, entity_id, subindex, was_active)` with **no delivery-calendar
anchor**; a bucket's maturity-lag `subindex` and the anticipated ring slot are
stage-clock-relative and misalign across resolutions. The mandatory policy-load
validation deliberately skips stage-count equality for the boundary-injection kind
— the one sanctioned mixed-resolution seam (a monthly source study may legitimately
feed a weekly+monthly current study).

**Current behavior.** Rejected loudly (dimension / positional mismatch). Safe, but
the real coarse-to-mixed workflow cannot run at all.

**Why deferred.** The delivery anchor depends on `resolve_point` (landing in the
anticipated refactor), and the family-fill coupling is a separate follow-up. Only
the requirement is recorded so far — no coupling code yet.

**What full support requires.**

1. Extend `EntitySlot` with a canonical absolute delivery anchor (year-month /
   date), emitted by `resolve_point` and stable across resolutions — a dual-owned
   wire-format change (`schemas/policy.fbs`, the hand-rolled writer/reader slot
   constants in `records.rs`, and `build_stage_entity_manifest` in
   `policy_export.rs`), needing a round-trip test **and** a reject-old-version test.
2. Rewrite `load_boundary_cuts` to length-tolerant, family-aware re-indexing keyed
   on `(downstream_id, delivery_anchor)` and `(thermal_id, delivery_anchor)`.
3. Add a boundary-coupling policy (buckets: reject / zero-fill / redistribute;
   anticipated: align-by-delivery-anchor, drop-out-of-window-with-warning) carried
   on the boundary-injection load kind.
   Large, cross-crate blast radius.

**Severity: Guarded gap** today, but the highest-value deferred capability. Its
prerequisite — the `EntitySlot` delivery anchor — should land **before** the in-LP
anticipated ring, or it becomes a retrofit of the manifest and every boundary path.

## Item 8 — Advisory-only resolution-mismatch signals

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

**Why deferred / required.** Nothing required — these are intentional advisories,
listed for completeness because they are where a coarse/fine mismatch surfaces.

**Severity: Guarded (informational).**

---

## What the anticipated lead-time refactor inherits or activates

The anticipated refactor reuses `resolve_point`'s end-anchored decider and its
mixed-calendar depth logic, both already calendar-general. It adds no _new_
non-regular assumption to the resolver core — but the hours-based lead-time mode
**activates** the calendar sensitivity that the stage-count mode currently
suppresses, and the ring **inherits** several deferred items rather than resolving
them:

- **Calendar-dependent decision sets (Item 6).** The stage-count mode is
  calendar-blind and safe. The hours mode makes the per-stage decision set
  `K_i(t)` stage-length-dependent: a fixed Δ resolves to a _different_ number of
  lags at a weekly stage than at a monthly stage. Two concrete inheritances:
  1. The pre-commitment IC reindex must be an ordinal study-stage-to-delivery-stage
     mapping (not a date-windowing). The exact numeric trap to re-verify: the
     initial-condition anchor depth uses **no `−1`** (the full overlap count),
     whereas the in-study arrival depth _does_ subtract one for the same-stage
     share — the precise spot where a transcription error recurs. Hand-derive both.
  2. Fan-out for multi-delivery deciders is the mirror of confluence: on a fine
     decision stage feeding several coarse-lagged deliveries (or the reverse), one
     plant deposits into multiple future delivery stages. The summation discipline
     (never a single-term regression) and cross-source conservation assertions
     (sum across delivery stages, never total-cost-only) transfer directly.
- **Horizon truncation (Item 3).** A ring reaching its own horizon inherits the
  same two-part safety argument (row omission **plus** zero-terminal-value
  inertness under a finite horizon). It must apply two-sided masking — row layer
  _and_ column layer — from the start, with a dense-row-count regression at a
  horizon-truncated stage, not merely a state-dimension cap.
- **Multi-resolution delivery density (Item 1) re-applies verbatim.** If the
  anticipated delivery ever lands on a decomposition-mode stage, the same
  fixed-template approximation applies — flag it again rather than assuming it was
  closed on the water side.
- **Possible new superset reject.** If fan-out introduces an analogous
  cross-consumer heterogeneity case, budget for the same
  crate-boundary-forces-a-superset-reject trade-off as heterogeneous confluence,
  rather than threading per-arc computation into the I/O crate.

## Priority ordering

**Fix first — latent risk, thin or absent coverage on real inputs:**

1. **Item 1 — multi-resolution arrival-stage delivery density.** The only item
   silently approximate on a real input with no regression, and it re-applies to
   the anticipated delivery. Add a decomposition-arrival regression and decide
   upgrade-vs-accept _before_ the anticipated ring lands on a decomposition stage.
2. **Item 7's prerequisite — the `EntitySlot` delivery-calendar anchor.** Not
   silently wrong (it hard-rejects), but a hard blocker that should land before the
   in-LP anticipated ring; it is a dual-owned wire change and slipping it forces a
   retrofit.
3. **Item 4 — non-Monthly mid-period PAR seed.** Produces a wrong (zero) seed;
   cheap, self-contained; currently masked only by a warning.

**Fix as designed — correctness-critical once the mode activates:**

4. **Item 6 → the whole anticipated hours mode.** Safe only because absent; the
   moment it ships, the ordinal-reindex constraint, the no-`−1` IC-anchor depth,
   the fan-out summation, and the two-sided horizon masking are all
   correctness-critical. Hand-derive every regression's expected value.

**Feature gaps — guarded, schedule by demand:**

5. **Item 7 (full) — coarse-to-mixed boundary coupling.** High capability value,
   hard-rejected today; a follow-up after the anchor exists.
6. **Item 2 — precise heterogeneous-confluence support.** Guarded; only worth it if
   a real study needs mixed-travel-time confluence under chronological blocks.
7. **Item 5 — sub-monthly PAR clock.** Owned by the PAR boundary; largest blast
   radius, lowest travel-time relevance.

**Never a "fix" — protect / leave:**

8. **Item 3 — terminal credit.** A protected contract coupled to non-existent
   cyclic-horizon work. Protect the row-omission + zero-terminal-value invariant;
   do not patch it.
9. **Item 8 — advisories.** Correct as-is.

## Already handled — not gaps

To avoid re-filing resolved work as deferred:

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
- **The resolver primitives being non-uniform-unsafe.** `resolve_point`,
  `resolve_spread`, and `window_period_overlaps` are calendar-general and
  unit-tested on non-uniform and gapped calendars. The deferred work is in
  consumers, never the primitive.
- **Depth-padding cadence.** `extend_for_resolution` only sizes depth and k-weights
  for the beyond-horizon tail that the horizon cap drops (Item 3); within-horizon
  k-weights use the real calendar. It is a sub-note of the terminal-drop contract,
  not an independent bug.
