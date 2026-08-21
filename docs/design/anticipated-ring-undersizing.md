# Anticipated ring under-sizing when the pre-study run exceeds the ring depth

**Status: Open bug — decision record.** Confirmed by an executed solve on v0.15.0
(HiGHS). It is a **silent wrong committed value**, not a crash or a reject. It is
**not** reachable on the DECOMP bridge mirror-shift path, and no shipped
deterministic deck exercises it; it triggers when a `LeadTime` anticipated thermal
has more leading pre-study-decided deliveries than the resolved ring depth
(`n_none > k_max`), which arises when the post-study calendar has fewer stages than
the study (`n_post < n_stages`).

This is distinct from the retired `k_max == 0` panic (that was the all-`K=0`
sub-stage regime, guarded and fixed). Here `k_max >= 1`; the ring is merely one (or
more) slots too shallow.

---

## Symptom

The last pre-study-seeded study stage delivers an **earlier** study stage's
committed MW instead of its own. No error, no `Infeasible` — the study solves and
reports the wrong anticipated generation.

## Reproduction (executed, v0.15.0 HiGHS)

A study of four stages — three operative weeks (168 h) then one month (648 h),
horizon 1152 h — on an anticipated `LeadTime` thermal with
`anticipated_config.lead_time_hours = 1160` (>= horizon, so **all four** study
deliveries resolve pre-study), and **distinct** non-zero
`past_anticipated_commitments` tiling all four study stages (`100, 200, 300, 400`
MW). The terminal boundary is omitted so the seeds alone pin the fished generation.

- With **one** post-study monthly stage (`n_post = 1 < n_stages = 4`): the observed
  `anticipated_committed_mw` per study stage is `100, 200, 300, `**`100`**` ` — the
  last stage delivers stage 0's value. Cross-checked on energy:
  `generation_mwh` at stage 3 = `64800 = 100 × 648`, not `259200 = 400 × 648`.
- With **four** post-study stages (`n_post = n_stages`, the mirror-shift): the same
  seeds deliver correctly (`100, 200, 300, 400`).

So the defect is specific to `n_post < n_stages`.

## Root cause

- The ring depth is sized from occupancy in `AnticipatedResolution::resolve`
  (`crates/cobre-sddp/src/lead_time/mod.rs`): `k_max = max_t occupancy[t]`.
  `occupancy` counts strictly-future deliveries (`m > t`), which is correct only
  when an in-study **deposit** recycles each just-fished slot. When the deciders
  cluster late (few post-study deliveries), the early study stages get no deposit,
  the leading pre-study seeds are never recycled, and `k_max = n_none − 1 < n_none`.
- The stage-0 seed writes `.take(k_i)` with `k_i = anticipated_lead_stages`
  (= `occupancy_max`) in `crates/cobre-sddp/src/setup/mod.rs`, **silently dropping**
  the overflow window that cobre-io's `lead_delivery_stage_count`
  (`crates/cobre-io/src/validation/semantic/thermal.rs`) just **required** the deck
  to tile at coverage 1.0. Nothing cross-checks the two counts.
- Modular aliasing: `slot(m) = m mod k_max`
  (`StateSpace::commitment_hold_in_study_offset`) maps `m` and `m + k_max` to the
  same slot; with `k_max = n_none − 1`, the dropped last seed's fishing reads the
  aliased earlier slot (`m = k_max ≡ 0`), delivering that earlier stage's value.

## Reachability

- **Safe** on the DECOMP bridge mirror-shift — it always emits `n_post = n_stages`,
  giving one in-study decision per study stage, `k_max = n_none`, no aliasing.
- **Safe** on every shipped deterministic deck: `d55` is `k_max = 1` with a single
  zero seed; `d34`/`d37` are `LeadStages` with `n_none = k_max`. No deck combines a
  `LeadTime` lead-at-or-past-horizon, `n_post < n_stages`, and distinct non-zero
  seeds.
- **Triggers** for a hand-authored case (or any future emitter) with
  `n_none > k_max`.

## Fix candidates

1. Size `k_max = max(occupancy_max, n_none)` so the ring holds every
   simultaneously-in-flight pre-study seed — the minimal fix that makes the case
   solve correctly.
2. In cobre-io, reject when `lead_delivery_stage_count > resolved k_max` — turn the
   silent seed drop into a loud validation error rather than a wrong answer.

Either removes the silent wrong answer; (1) additionally makes the configuration
usable rather than rejected.

## Invariant to pin with a regression test

For any anticipated `LeadTime` plant, every delivery window cobre-io requires
(`lead_delivery_stage_count`) must map to a **distinct** ring slot, and the last
pre-study-seeded delivery must fish its **own** committed value — never an aliased
earlier stage's. A test that seeds distinct leading values under `n_post < n_stages`
and asserts the last stage's `anticipated_committed_mw` equals its own seed
reproduces the defect today.
