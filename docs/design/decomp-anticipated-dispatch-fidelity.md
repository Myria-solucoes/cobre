# Representing DECOMP anticipated dispatch with fidelity — problem statement

**Status: Design brief (problem statement).** This document describes the situation
and the goal only. It states what must be represented, how cobre models it today,
the constraints confirmed by executed verification on v0.15.0, and the fidelity
gaps that remain. It deliberately proposes **no** solution — the design is
owner-led and will be written separately.

Two options are **out of scope** by owner decision and are not to be revisited
here: extending the study horizon (adding study stages), and lifting the E2 scope
boundary by carrying a pre-study decision through the ring into the post-study.

---

## 1. What the DECOMP case is (what we must represent)

Reference deck: `~/git/cobre-bridge/example/decomp-mar-26-rv2-reduced`
(`dadgnl.rv2` = GNL data, `relgnl.rv2` = the operation report).

- **Two GNL (fuel-constrained) thermals:** SANTA CRUZ (code 86, SE) and PSERGIPE I
  (code 224, NE). Each declares an **anticipation lag of 2 whole months**
  (`dadgnl` `NL` register).
- **Study horizon:** 2026-03-14 → 2026-05-01 — three operative weeks of March
  (Sat–Fri, 168 h each) then the April month (648 h). **Four stages, 1152 h.** This
  horizon is fixed; the number of study stages must not change.
- **The anticipation rule:** a dispatch decided in a study period commits generation
  **2 months later**. So the four study stages decide four future deliveries:

  | study stage  | decides delivery for | date                     |
  | ------------ | -------------------- | ------------------------ |
  | 0 (Mar wk 1) | Sem 10 (a week)      | 2026-05-16 → 05-23       |
  | 1 (Mar wk 2) | Sem 11 (a week)      | 2026-05-23 → 05-30       |
  | 2 (Mar wk 3) | Sem 12 (a week)      | 2026-05-30 → 06-06       |
  | 3 (April)    | the June remainder   | 2026-06-06 → end of June |

- **Já-comandadas (already dispatched):** the two weeks immediately after the
  horizon — Sem 8 (2026-05-02) and Sem 9 (2026-05-09) — carry generation **already
  committed in prior weekly revisions** (`dadgnl` `GL` "GERACOES DE TERMICAS GNL JA
  COMANDADAS"; flagged `*` "Geracao definida em revisoes anteriores" in `relgnl`).
  They are fixed inputs, decided before this study, delivered after it. This study
  therefore begins deciding new generation at **2026-05-16**, not 05-01.
- **In-study generation:** whatever the GNL plants generate inside Mar–Apr is itself
  fixed by commitments decided 2 months earlier (pre-study). In this deck those are
  all `0`; in general they are the já-comandadas for the study months.
- **Cost & discount:** an anticipated commitment's fuel cost is charged at the
  **decision** stage but **discounted to the delivery date**. The discount factor is
  the one that belongs to the real delivery moment (e.g. 05-16), compounded from the
  study start at the configured annual rate.

## 2. How cobre models anticipated dispatch

- **Extended delivery axis.** Deliveries are indexed on `[0, n_delivery)` where
  `n_delivery = n_stages + n_post`; the axis chains the study-stage **durations**
  with the `post_study_stages` **durations** (`delivery_stage_durations`).
- **Decider (end-anchored).** For a `LeadTime` plant, delivery `m` is decided at the
  stage containing `stage_end(m) − lead_time_hours` (`resolve_decider_physical`).
  `None` ⟹ the decision is pre-study.
- **Carriers.** An in-study decision deposits into an anticipated-ring slot, is
  carried to the terminal, and is priced against the boundary FCF through the
  generic `β·state` cut projection. A pre-study commitment that delivers **into the
  study** is a `past_anticipated_commitment` seed. A pre-study commitment that
  delivers **after the study** has no carrier (see §3, E2).
- **Discount of the anticipated cost.** The commitment column's cost and discount
  are read **at the delivery stage** (`cost and discount at delivery_stage`), i.e.
  the delivery stage's cumulative discount factor on the extended axis. The delivery
  stage's position/date on that axis is therefore what fixes the discount applied.

## 3. What cobre does with this deck today (the mirror-shift) and its constraints

The bridge (`cobre-bridge/src/cobre_bridge/decomp/anticipated.py`) maps the deck
with a **mirror-shift**: it sets a single global lead `H` ≈ the study span in
operative weeks and builds the post-study calendar as the **study stages shifted
forward by `H`**, so every post-study delivery is decided by exactly one study stage
(stage `m` → post-study stage `m`), and the leading study stages are tiled with
`past_anticipated_commitments`.

Executed verification on the v0.15.0 binary established the following hard facts:

- **Post-study contiguity is mandatory.** The post-study calendar must begin exactly
  at the study horizon end (05-01):

  > `first post-study stage starts 2026-05-16 but the study horizon ends
2026-05-01; the post-study calendar must begin exactly at the study horizon
end.`
  > So 05-01→05-16 cannot be skipped or left as a gap; it must be covered by
  > post-study stages.

- **E2 rejects the faithful resolution.** With the real 2-month lead and the
  post-study calendar covering 05-01→05-16, the já-comandada weeks (Sem 8, Sem 9)
  are pre-study-decided **and** post-study-delivered → no carrier:

  > `Thermal 94: a commitment decided at a pre-study stage delivers into post-study
stage index(es) [0, 1, 2]; a pre-study-decided, post-study-delivered commitment
has no carrier on the delivery axis. Shorten the lead so the decision falls
within the study horizon.`
  > E2 fires for **any** post-study stage whose end lands in `(horizon_end, lead]`.
  > It is a ratified scope boundary, not a bug.

- **The mapping is monotone and the earliest delivery is pinned to 05-01.** Earlier
  study stages map to earlier post-study deliveries, and the earliest post-study
  delivery sits at the horizon end. So study stage 0's delivery cannot be moved to
  05-16 by any choice of intervals or lead: a long enough lead to reach 05-16 makes
  05-01→05-16 E2; a short enough lead to avoid E2 puts study stage 0's delivery at
  ~05-01.

- **Ring under-sizing bug** (`anticipated-ring-undersizing.md`). Independently, when
  the leading pre-study run exceeds the ring depth (`n_none > k_max`, i.e.
  `n_post < n_stages`), the last seeded study stage silently delivers an earlier
  stage's committed MW. Off the mirror-shift path, but any redesign that changes the
  post-study/lead structure must stay clear of it.

## 4. The fidelity gaps that remain

1. **Wrong delivery dates → wrong discount.** The mirror-shift dates its deliveries
   05-01 → 05-16 instead of the real 05-16 → June. Study stage 0's commitment is
   discounted as if delivered ~2 weeks early; the last commitment as if delivered
   ~1 month early. Because the discount is read at the delivery stage (§2), the
   present value of the anticipated fuel cost is off — on the order of a fraction of
   a percent per commitment at a 12 %/yr rate, growing with the mis-dating, and
   varying by deck. This is a real cost error, not cosmetic.

2. **Já-comandada fixed generation is dropped.** The mirror-shift re-uses the
   05-02 / 05-09 date-slots as in-study **free** decisions and does not honor their
   fixed já-comandada MW (the bridge records this as "an accepted modelling loss").

## 5. The goal

A solution must, on this fixed 4-stage study:

1. **Fidelity of the anticipation resolution.** The four study stages decide the four
   deliveries at their **real dates** — Sem 10 (05-16), Sem 11 (05-23), Sem 12
   (05-30), and the June remainder — not the mirror-shifted 05-01→05-16 dates.
2. **No new study stages.** The study stays exactly the four Mar–Apr stages
   (1152 h). Any representation of the future May/June deliveries must live outside
   the study-stage set.
3. **Correct discounting.** The anticipated thermal generation is delivered — and
   its committed cost discounted — at the **right moment** (its real delivery date),
   so the present value the study charges is exact.
4. **Correct treatment of the já-comandadas.** The fixed, prior-revision generation
   in the near-post-study weeks (Sem 8 / Sem 9) is either honored or explicitly and
   correctly accounted, not silently re-interpreted as a free decision.

## 6. The core tension a solution must resolve

Today three things are welded together and cannot all hold at once:

- the **delivery date** (which sets the discount, and must be the real 05-16+),
- the **axis position / decider** (which must keep the decision in-study to have a
  carrier, i.e. avoid E2), and
- the **contiguity rule** (post-study starts at the horizon end, so 05-01→05-16 is
  on the axis).

The mirror-shift resolves the tension by collapsing the delivery date onto the axis
position (start at 05-01), which is what breaks the discount. A faithful solution
has to **decouple the delivery date used for discounting from the delivery's axis
position used for the decider/carrier** — or otherwise price an anticipated delivery
at its true future date without placing a carrier-bearing post-study stage in the
já-comandada window. Achieving that on a fixed study-stage set is the design
problem.

## 7. Reference material

- Deck: `~/git/cobre-bridge/example/decomp-mar-26-rv2-reduced/{dadgnl.rv2,relgnl.rv2}`;
  converted cobre case: `.../cobre-mar-26-rv2-reduced`.
- Bridge mapping: `cobre-bridge/src/cobre_bridge/decomp/anticipated.py`
  (`convert_gnl`, `_study_lead_hours`, `_build_post_study_calendar`,
  `_cobre_safe_lead_hours`).
- cobre model & contracts: `anticipated-thermals-and-water-travel-time.md` (this
  directory); the decider/axis/seed code in `crates/cobre-sddp/src/lead_time/`,
  `.../setup/mod.rs`, `.../lp/indexer/`, and the validation in
  `crates/cobre-io/src/validation/semantic/thermal.rs`.
- Filed defect: `anticipated-ring-undersizing.md`.
