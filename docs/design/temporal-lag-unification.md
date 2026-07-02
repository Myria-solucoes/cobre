# Temporal Lag Unification — Water Travel Time & Anticipated Dispatch

Design memo (formulation link; companion to
`water-travel-time-sddp-analysis.md`). Two features carry a **fixed physical
time offset between a decision at one stage and its effect at a later stage**,
discretized against a stage calendar that is in general **non-uniform**
(weekly stages followed by monthly stages, the PMO structure):

1. **Water travel time** — an upstream release at stage $t$ arrives downstream
   $t_v$ hours later (greenfield; formulation in the companion memo).
2. **Anticipated thermal dispatch** — a commitment decided at stage $t$ fixes
   the plant's generation at a later stage (shipped; `AnticipatedConfig`).

Today the second takes its lag **directly as a stage count**
(`AnticipatedConfig::lead_stages: u32`, `crates/cobre-core/src/entities/thermal.rs`)
with no time semantics — §4 shows this is not merely imprecise on a non-uniform
calendar but **semantically broken** (many-to-one maturation collisions,
delivery gaps). This memo (i) grounds the water-travel-time design against the
DECOMP reference model, (ii) works the key stage-length × block-structure ×
mode scenarios, (iii) derives the **one preprocessing engine** both features
need, and (iv) proposes the anticipated-dispatch refactoring.

---

## 1. DECOMP consistency (the external anchor)

Source: _Modelo DECOMP — Manual de Referência_ (CEPEL, Outubro 2021),
§4.5.6, §4.5.14.2, §4.5.15, §5.3. (A local copy is kept outside the repo
tree.) Four findings, each mapped to the cobre design:

| #   | DECOMP (manual)                                                                                                                                                                                                                                                                                                                                                                  | cobre design element it validates                                                                                                                                                                                                                                                                                                                      |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | §4.5.6 (the umbrella travel-time section: considered "nas restrições de balanço hídrico … e nos cortes de Benders") + §5.3: propagation via **fixed proportionality factors** $k_0^t, k_1^{t-1}, k_2^{t-2}, k_3^{t-3}$ on upstream defluence — arrivals at stage $t$ come from releases at $t, t{-}1, \dots$ with fractional factors, **including a same-stage share $k_0 > 0$** | The uniform-density overlap $k_d$ curve **and the exact-overlap convention** (companion memo §2.5.5): DECOMP keeps a same-stage fraction and carries fractional cross-stage mass — it does **not** fold                                                                                                                                                |
| 2   | §5.3.1–§5.3.2: the Benders cut is the **textbook multi-lag $E^k$ cut**; the state vector is $x_t = (v_1, v_2, d_1, d_2)$ — storage **plus raw defluence volumes** $d_i^t = q_i^t + s_i^t$, cut terms on $x_t, x_{t-1}, \dots, x_{t-L}$                                                                                                                                           | Companion memo §2.2 proves this multi-lag form and the bucket lifting are the **same object in two coordinate systems** ($b = M\mathcal{D}$); cobre implements the lifted (Markov-1) coordinates                                                                                                                                                       |
| 3   | §4.5.15 (and the balance-by-patamar forms): the lagged arrival is distributed across the arrival stage's patamares as $(Q^{t-tv} + S^{t-tv})_k = \frac{d_k}{D}(Q^{t-tv} + S^{t-tv})$ — **duration-proportional**                                                                                                                                                                 | Sub-contract 3's fixed template delivery density $\rho$ (companion memo §2.5.3): DECOMP's $d_k/D$ **is** the uniform $\rho$ over a parallel stage's blocks                                                                                                                                                                                             |
| 4   | §4.5.14.2, Figs. 5.5b/5.5c (**an ENA/inflow-propagation example**, not a defluence balance): $T_v = 15$ d against **weekly then monthly stages** — weekly stage arrivals composed as $\tfrac67$ of lag-2 plus $\tfrac17$ of lag-3 **inflows**; the monthly stage keeps $\tfrac{\Delta t - 15}{\Delta t}$ same-stage                                                              | The stage-clock overlap arithmetic on a **non-uniform calendar** with stage-varying depth — cited as **arithmetic cross-validation** (DECOMP uses one calendar engine for inflow and defluence propagation); the defluence-side anchors are rows 1–3. §3 scenario S1a reproduces the $\tfrac67 / \tfrac17$ **exactly** from the shared uniform density |

Two deliberate divergences, both grounded:

- **Coordinates.** DECOMP carries raw lagged defluences in the state and pays
  the multi-lag cut bookkeeping; cobre lifts to k-weighted volume buckets
  (Markov-1) because its cut pool, broadcast payload, and `StateLayout` are
  built on a one-step state vector of fixed per-stage meaning. The companion
  memo §2.2 is the equivalence proof; any future DECOMP-parity comparison of
  _states_ (not policies) must translate through the Hankel map $M$ (stage-dependent
  $M_t$ on a non-uniform calendar — companion §2.2's calendar-generalization note).
- **Chronological blocks.** DECOMP has no chronological mode (patamares are
  simultaneous load slices), so it never attributes the crossing mass by
  block. cobre's block-resolved deposit $\chi_{b,d}$ (companion memo §2.5.2)
  is a strict refinement that degenerates to DECOMP's stage-uniform treatment
  in parallel mode and at $K = 1$.

---

## 2. The unifying abstraction

Both features are instances of one object:

> A **calendar-anchored lag**: a physical offset $\Delta$ (hours) between an
> action anchored at stage $t$ and its effect, resolved against the stage
> calendar $h_1, \dots, h_T$ into per-stage integer **depths** and (where the
> effect is a spreadable quantity) fractional **weights**.

The two features differ only in the **discretization semantics** of the
effect:

| Aspect            | Water travel time                                                                                                                                                                                                                                                                                                                                                                                                                      | Anticipated dispatch                                                                      |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Effect quantity   | a **volume** (spreadable, additive)                                                                                                                                                                                                                                                                                                                                                                                                    | a **generation level** (a point commitment, not additive)                                 |
| Discretization    | **spread**: uniform-density window overlap → fractional $k_d(t)$, $\sum_d k_d = 1$, same-stage $k_0$                                                                                                                                                                                                                                                                                                                                   | **point**: one decision stage per delivery stage (see §4.3 — delivery-anchored inversion) |
| Per-stage depth   | $L_{\text{arc}}(t)$ = **deepest** future stage the shifted window reaches (max $d$, against the real calendar — NOT the overlap count: skipped intermediate lags still need contiguous transit slots; $\lceil t_v/h_t\rceil$ only on a uniform calendar). Ring sizing takes the max over **all anchors including the pre-study (IC) periods**, whose coarser windows can reach deeper than any in-study anchor — companion memo §2.5.1 | $K_i(t)$ = count of committed-but-undelivered future stages (derived, §4.3)               |
| State object      | in-transit volume buckets, one scalar per (arc, maturity)                                                                                                                                                                                                                                                                                                                                                                              | committed-generation slots, one per (plant, undelivered stage)                            |
| Block interaction | chronological: block-resolved deposit $\chi_{b,d}$ / routing $\kappa$ / delivery $\rho$                                                                                                                                                                                                                                                                                                                                                | none — commitments are stage-level decisions (already `n_blks`-independent)               |
| Many-to-one       | volumes **add** — the bucket transition sums naturally                                                                                                                                                                                                                                                                                                                                                                                 | levels **collide** — must be excluded by construction (§4.2–§4.3)                         |

Both consume the **same calendar-overlap pattern** (precedent: the
`days_in_period` / `month_total_hours` / `find_season_year_monthly` family in
`crates/cobre-sddp/src/stochastic/lag_transition.rs`) and both emit
stage-varying depths consumed by the **same state pattern** — global-max
dimension with a per-stage active mask (`anticipated_state`'s
$k_{max}$-global / $K_i$-active / padding-masked discipline,
`state_layout.rs::set_nonzero_mask`). And both must obey the same hard
contract: **depths are a pure function of ($\Delta$, stage calendar) on the
stage clock — never of `n_blks` or `block_mode`** (companion memo §2.5,
sub-contract 1).

### 2.1 The temporal lag resolver (proposed shared component)

A setup-phase precompute — deterministic, single-threaded, canonical entity
order — with two entry points over one overlap engine:

```
resolve_spread(Δ, calendar, anchor_stage t)
  → { depth L(t), weights k_d(t) (stage clock),
      per-block χ_{b,d}(t), κ_{b→b'}(t) for chronological anchor stages,
      delivery densities ρ(t+d) per arrival stage }

resolve_point(Δ, calendar)
  → { for each delivery stage m: decision stage c(m) (or PRE-STUDY),
      per-stage outgoing-commitment sets C(t) = { m : c(m) = t },
      depth K(t) = |{ m > t : c(m) ≤ t }| }
```

Placement (genericity-checked): split in two. The **interval-overlap primitive**
(intersect a shifted window with calendar periods) must be a **new, hour-resolution,
multi-slot** implementation — the existing `days_in_period` / `month_total_hours`
(`cobre-sddp`'s `stochastic/lag_transition.rs`, `pub(crate)`) are **day-granular**,
`compute_monthly_transition` is two-slot, the transition machinery is
**monthly-cycle-gated** (`Weekly`/`Custom` season cycles take a no-op path — the very
calendars the DECOMP validation targets), and `Stage::start_date`/`end_date` are
day-resolution `NaiveDate` (a sub-day $t_v$ cannot anchor on dates at all); they are
the **pattern** precedent, not reusable code. The new primitive builds its stage
clock from **cumulative `duration_hours`** — hour-exact, cycle-agnostic. The new primitive is genuinely generic and
could live in `cobre-core`'s temporal module — but only stated in calendar
vocabulary. The two
**resolvers** are not: their outputs (in-transit volume buckets, committed-generation
slots, per-stage state depths) are solver-state concepts, and the
infrastructure-genericity hard rule (zero algorithm-specific references in
`cobre-core`) plus the auto-loaded comment rules would be violated by documenting them
there. The resolvers live in `cobre-sddp` (a sibling of the lag-transition precompute),
consumed by setup (`StudySetup` construction), not by any hot path.

### 2.2 Transition mechanics — in-LP vs out-of-LP (an asymmetry the unification should resolve)

The two features move state between stages by **different mechanisms today**, and
the difference is load-bearing:

- **Water buckets transition INSIDE the LP.** The deposit is a linear function of
  the **current stage's decisions** (release columns), so the bucket-definition
  rows ($b_d^{\text{out}} = b_{d+1}^{\text{in}} + \text{deposit}_d$) must be solved
  simultaneously with the dispatch — the "ring shift" is **row indexing in the
  per-stage template**. State propagation between stages is _copy the outgoing
  solution values and pin them as the next stage's incoming bounds_ — byte-for-byte
  the storage mechanism. **No new shift code in `stochastic/noise.rs` exists on the
  water side**; the shared "anticipated pattern" claim covers the
  global-max/per-stage-mask **sizing discipline only**, not the shift mechanics.
- **Anticipated state transitions OUTSIDE the LP.** `noise.rs::shift_anticipated_state`
  moves the ring between solves, reading the decision from the primal solution and
  re-indexing slots in Rust — with the constant-per-plant slot arithmetic that §4.3's
  scope list shows breaking under per-stage $K_i(t)$.

This asymmetry suggests the stronger unification: **move the anticipated ring
in-LP** (the recommended mechanism in §4.3). Give every anticipated slot an
outgoing column and a definition row —
$\text{slot}_d^{\text{out}} = \text{slot}_{d+1}^{\text{in}} + [\text{decision for
delivery } t{+}d,\ \text{if } c(t{+}d) = t]$ — exactly the bucket structure. Then:
per-stage $K_i(t)$, fan-out deposits ($|C(t)| > 1$), and the $K = 0$ degeneracy are
all **template-construction concerns** (the template is already built per stage),
state propagation becomes uniformly "copy outgoing values" for storage, buckets, and
anticipated alike, and the constant-keyed `shift_anticipated_state` is deleted rather
than generalized. A further structural benefit: in-LP **restores the stage-invariant
cut-column map for the anticipated block** — outgoing slot columns sit at fixed
offsets and per-stage depth is handled by masking (like storage and buckets), so
`state_to_lp_column`'s constant-lead shift-map branch disappears instead of becoming
per-stage. The in-LP path is not without precedent: the code already carries two
in-LP anticipated constructs (the `anticipated_state_out` definition row and the
matured-slot generation coupling), of which the per-slot ring is the natural
generalization. Costs, stated honestly: ~$A \cdot k_{max}$ additional columns and
definition rows per stage LP; the anticipated _mechanics_ change even for uniform
calendars (same optimal values and cuts — pinned by a uniform-calendar
**value/cut-identity anchor**, §4.3; note the anchor's dual-degeneracy caveat — a
degenerate optimum can admit multiple valid subgradients across formulations, which
is precisely why the anchor must be _pinned by a test case_, not assumed — and not
byte-identical LPs, since the LP gains rows/columns); and the warm-start basis
footprint of the anticipated block changes shape. The conservative alternative — keep the out-of-LP shift and teach it
per-stage arithmetic — preserves LP bytes on uniform calendars but carries all three
§4.3 scope items as bespoke shift/resolver code. Surfaced as a mechanism fork in
§4.3; in-LP is recommended because it converts three special cases into the
already-required template machinery.

### 2.3 Scalar delay vs propagation curve (config future-proofing)

DECOMP's §5.3 formulation is a set of **fixed proportionality factors**
$k_0, k_1, \dots$ — of which "pure translation by $t_v$ under a uniform release
density" (this design's v1) is the special case obtained by overlapping the shifted
window with the calendar. Real channel routing disperses as well as delays (a
Muskingum-style arrival **curve**, not a shifted rectangle), and DESSEM-class
short-term models head that way. The resolver should therefore be **architected
density-first**: `resolve_spread` takes an **arrival density** per arc and computes
every output ($k_d$, $\chi_{b,d}$, $\kappa_{b \to b'}$, $\rho$, and the depth = max
reach of the density's support) by overlap against nested calendar partitions.
The v1 config is a scalar `travel_time_hours` (density = uniform-translation), and a
future per-arc curve (piecewise weights over hours) becomes a **pure config
extension** — zero change to the state machinery, the depth rule, or the contracts.
Stated precisely: the **state machinery, depth, and contracts depend only on the
density's support** (a curve with the same support is a drop-in; a longer tail
extends the depth), while the **deposit weights** $k_d, \chi_{b,d}, \kappa, \rho$
**do depend on the shape** — that is the point of a routing curve — and are
recomputed from the density by the resolver, which is exactly why `resolve_spread`
takes the density as input: the shape change touches resolver output values, never
state/contract code. Surfaced as an architecture requirement now precisely so the
curve does not force a redesign later.

---

## 3. Scenario matrix — water travel time

Scenarios chosen to span the regimes that exist in production calendars.
$L$ is the per-arc bucket depth at that stage; all values follow from the
uniform-density overlap (companion memo §2.5.1) and are **identical in
parallel and chronological mode** (sub-contract 1). "Chronological adds"
describes the stateless refinement only.

| #   | Calendar / stage                                                                                          | $t_v$         | Stage-clock result                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Chronological adds                                                                                                                                                  |
| --- | --------------------------------------------------------------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S1a | weekly (168 h), PMO zone                                                                                  | 15 d (360 h)  | $L = 3$: $k_2 = 6/7$, $k_3 = 1/7$ ($k_0 = k_1 = 0$) — **matches DECOMP Fig. 5.5b exactly**                                                                                                                                                                                                                                                                                                                                                                                 | with daily blocks: per-block deposit splits between lag-2/lag-3 buckets by block position                                                                           |
| S1b | monthly (720 h), same arc                                                                                 | 15 d (360 h)  | $L = 1$: $k_0 = 1/2$, $k_1 = 1/2$ — DECOMP Fig. 5.5c's $\frac{\Delta t - 15}{\Delta t}$ shape                                                                                                                                                                                                                                                                                                                                                                              | in-stage half routed $b \to b{+}15$ (daily blocks); second half deposited block-resolvedly                                                                          |
| S1c | weekly→monthly transition                                                                                 | 15 d          | depths $(3, 3, 2, 1)$ across weeks 1–4 then $1$ (monthly): e.g. week 3's window $[29, 36)$ d skips week 4 entirely and lands in the month ($k_2 = 1$, slot 1 transit-only); global max 3, per-stage **reachability** mask — the variable-calendar case DECOMP works in §4.5.14.2                                                                                                                                                                                           | same masking; block factors recomputed per stage                                                                                                                    |
| S1d | **monthly→weekly transition**                                                                             | 15 d          | **the closed-form counterexample**: monthly anchor, window $[15, 45)$ d → $k_0 = 1/2$, then $k_1 = 7/30,\ k_2 = 7/30,\ k_3 = 1/30$ over three weekly stages — true depth **3**, while $\lceil t_v/h_t\rceil = \lceil 360/720\rceil = 1$ would drop $8/30 \approx 27\%$ of the total release ($8/15 \approx 53\%$ of the crossing mass); only the general max-reach depth (companion §2.5.1) is correct                                                                     | same                                                                                                                                                                |
| S2  | monthly parallel, planning                                                                                | 6 h           | $L = 1$, $k_1 = 6/720 \approx 0.8\%$ — bucket carries negligible mass; **setup advisory** ("consider not declaring") rather than silent fold                                                                                                                                                                                                                                                                                                                               | n/a (parallel)                                                                                                                                                      |
| S3  | daily chronological, 24 blk                                                                               | 6 h           | $L = 1$, $k_1 = 25\%$                                                                                                                                                                                                                                                                                                                                                                                                                                                      | blocks 0–17 route $b \to b{+}6$ in-stage; blocks 18–23 deposit fully ($\chi = 1$) — the DESSEM-like payoff                                                          |
| S4  | monthly, exact multiple                                                                                   | 720 h         | $L = 1$, $k_0 = 0$, $k_1 = 1$ — whole release crosses exactly one boundary                                                                                                                                                                                                                                                                                                                                                                                                 | all blocks deposit fully                                                                                                                                            |
| S5  | monthly, inter-basin                                                                                      | 75 d (1800 h) | $L = 3$: $k_2 = k_3 = 1/2$, $k_1 = 0$ — slot 1 carries only transit mass (deposits zero, ring shift passes through)                                                                                                                                                                                                                                                                                                                                                        | deposits split between the lag-2/lag-3 buckets by block position                                                                                                    |
| S6  | mixed modes across stages                                                                                 | any           | identical $B$ at every stage (sub-contract 1); cuts structurally portable                                                                                                                                                                                                                                                                                                                                                                                                  | deposits $\chi_{b,d}$ vs $k_d$ and delivery $\rho$ vs single-row differ per stage's own mode                                                                        |
| S7  | train parallel, simulate chronological, travel time ON                                                    | any           | same state dimension in both runs (sub-contract 1) → the trained policy **loads**; the cut is a valid lower approximation of the cost-to-go **under the parallel training dynamics** (stage-uniform deposits), evaluated against chronological trajectories whose bucket states follow the block-resolved deposits — the exact `coarse-train, fine-simulate` framing `book/src/guide/block-modes.md` already establishes for $K \ge 2$, extended to the bucket coordinates | simulation-side deposits/delivery are block-resolved; the policy's quality under the finer dynamics is an empirical question, not one the loading mechanism answers |
| S8  | one arc spanning mixed-mode stages (parallel release stage → chronological arrival stage, or the reverse) | any           | bucket count and meaning unchanged (stage-clock); **each stage's own `block_mode` governs its own side**: the release stage's mode picks the deposit form ($k_d$ vs $\chi_{b,d}$), the arrival stage's mode picks the delivery form (single row vs $\rho$-spread) — the four combinations compose freely because the bucket is the mode-independent interface between them (sub-contract 3)                                                                                | e.g. parallel release deposits stage-uniformly; the chronological arrival stage still spreads the scalar by its own $\rho$                                          |

Observations the matrix forces:

- **S1a is the DECOMP arithmetic cross-validation**: the same numbers fall out of the
  shared density with no tuning. Cited precisely: DECOMP §4.5.14.2 is an
  **ENA/inflow-propagation** example, not a defluence balance — it validates the
  overlap **arithmetic** (one calendar engine for both quantities in DECOMP); the
  defluence-side anchors are §5.3's $k$-factor propagation and the per-patamar
  $d_k/D$ split of the lagged defluence (companion memo §2.5.4).
- **S1d is why the depth is the max-reach against the real calendar, not a closed
  form**: the ceiling formula silently loses ~27 % of the total release (~53 % of
  the crossing mass) at a coarse→fine calendar transition (water non-conservation),
  and over-allocates phantom dims at the fine→coarse transition. The resolver must
  compute depths against the real calendar.
- **S7/S8 are the mode-mixing guarantees**: the bucket is the mode-independent
  interface — each stage's own `block_mode` governs only its own deposit/delivery
  form, and cross-mode policy loading inherits exactly the coarse-train/
  fine-simulate semantics the chronological feature already documents.
- **S5 shows depth ≠ nonzero-factor count**: intermediate buckets with zero
  deposit still carry transit mass through the ring shift. The per-stage mask
  excludes only slots that are structurally absent at that stage, not slots
  with zero deposit — the distinction matters for `nonzero_state_indices`
  (mask by _reachability_, not by $k_d \ne 0$; a reachable slot's cut
  coefficient is legitimately nonzero).
- **S2 is the honest cost of exact overlap** and the reason the setup
  advisory exists; it is a documentation/validation concern, not a model
  change.

---

## 4. Anticipated dispatch on a non-uniform calendar

### 4.1 The current model breaks — concretely

`AnticipatedConfig { lead_stages: u32 }` deposits the commitment decided at
stage $t$ at ring slot $K_i - 1$; it matures (fixes the plant's generation)
exactly $K_i$ stage-shifts later. On a uniform calendar, stage-shifts and
physical time agree. On the PMO calendar (4 weekly stages, days 0–28, then
monthly), a plant with a physical 30-day lead:

| decision stage $t$ | starts day | + 30 d = matures day | maturation stage  | required $K(t)$ |
| ------------------ | ---------- | -------------------- | ----------------- | --------------- |
| 1 (week)           | 0          | 30                   | 5 (month, d28–58) | 4               |
| 2 (week)           | 7          | 37                   | 5                 | 3               |
| 3 (week)           | 14         | 44                   | 5                 | 2               |
| 4 (week)           | 21         | 51                   | 5                 | 1               |
| 5 (month)          | 28         | 58                   | 6                 | 1               |

No constant `lead_stages` reproduces this column. Worse, the correct
per-stage $K(t)$ exposes two structural defects of decision-anchored
maturation on non-uniform calendars:

- **Many-to-one collision**: stages 1–4 all mature at stage 5 — four
  commitments claim to fix one stage's generation.
- **Gaps** (the mirror case, monthly→weekly transition): a fine-grained zone
  after a coarse zone leaves delivery stages no commitment matures at.
- **Degeneracy**: a lead shorter than the stage ($K(t) = 0$, e.g. a 30-day
  lead inside a 31-day month) means decision and delivery share a stage — the
  anticipation constraint vanishes there, the exact analogue of water's
  $k_0$ same-stage share. Note this is **structurally unrepresentable** in the
  shipped machinery (`shift_anticipated_state`'s `k_i − 1` slot arithmetic
  underflows; `lead_stages ≥ 1` is validated and asserted) — handled by the
  §4.3 sub-stage-lead fork, never by underflow.

### 4.2 Why the water semantics does not transfer directly

Water volumes **add**: two releases maturing at the same stage sum into one
bucket — many-to-one is free. A generation **level** does not add: two
commitments for the same delivery stage contradict each other. So the spread
discretization (fractional $k_d$) is wrong for point commitments unless the
commitment is reinterpreted as deliverable **energy** (then the water math
applies verbatim — surfaced as the alternative in §4.4, not recommended,
because the shipped semantics pins a generation level).

### 4.3 The fix — anchor at the DELIVERY stage (recommended)

Invert the mapping. The physical statement of anticipation is "generation at
stage $m$ must be decided $\Delta$ ahead," so the decider is derived from the
**delivery** stage:

$$
c(m) \;=\; \text{the stage containing } (\text{start}_m - \Delta),
$$

with $c(m)$ before the horizon start meaning the commitment comes from the
**initial conditions** (`past_anticipated_commitments` — the input cobre
already has). Properties, by construction:

- **No collisions**: each delivery stage has exactly one decider.
- **No gaps**: every delivery stage (past the IC boundary) is committed.
- **One decision stage may commit several delivery stages** (a monthly stage
  before a weekly zone decides all the weekly stages that fall $\Delta$ after
  it): the decision column set at stage $t$ is $C(t) = \{ m : c(m) = t \}$ —
  multiple anticipated-decision columns per plant per stage where the
  calendar demands it.
- **Per-stage state depth** $K_i(t) = |\{ m > t : c(m) \le t \}|$ — the count
  of committed-but-undelivered stages — replaces the constant `lead_stages`;
  $k_{max} = \max_t K_i(t)$, with the global-max + per-stage-mask **pattern**
  carried over from `anticipated_state`.

**Honest implementation scope — this is NOT a no-op on the shipped machinery**
(verified against the code; three concrete breaks):

1. **The ring deposit and the cut-column resolver are keyed on a per-plant
   CONSTANT.** `noise.rs::shift_anticipated_state` deposits one value at slot
   `k_i − 1` from `layout.anticipated_lead_stages[plant]`, and
   `state_layout.rs::state_to_lp_column` resolves the matured slot by comparing
   against the same constant through a **single stage-0 map** applied at every
   stage. Per-stage $K_i(t)$ moves both the deposit slot and the matured-slot
   target stage-by-stage — the anticipated block needs a **per-stage** column
   resolution (or a redesigned slot indexing), abandoning the
   single-stage-0-map property **for that block only** (storage/lag/bucket
   blocks keep it).
2. **Fan-out deposits.** Where $|C(t)| > 1$, the plant deposits $|C(t)|$ values
   from $|C(t)|$ distinct decision columns into distinct slots in one stage;
   the shipped shift writes exactly one `newest` slot per plant. The deposit
   loop becomes $O(|C(t)|)$ per plant per stage.
3. **$K_i(t) = 0$ is structurally unrepresentable today** — the shift's
   `k_i − 1` slot arithmetic underflows and both the config validation and a
   debug assertion pin `lead_stages ≥ 1`. See the degeneracy fork below.

**Sub-stage lead ($\Delta < h_m$, the water-$k_0$ analogue) — surfaced fork:**
when $c(m) = m$ the commitment would be decided inside its own delivery stage,
i.e. no anticipation binds there. Options: (i) **exclude-from-anticipation
(recommended)** — delivery stages with $c(m) = m$ carry no commitment and the
stage LP dispatches the plant freely, logged as a per-stage diagnostic (the
constraint genuinely does not exist at that stage); (ii) hard validation error
(forbid leads shorter than any stage in the plant's window) — safer but
rejects legitimate mixed calendars where only the coarse stages degenerate.
Either way the shipped $K \ge 1$ contract is revisited explicitly, never
underflowed.

On a uniform calendar with $\Delta$ an exact stage multiple this degenerates
**exactly** to today's model ($|C(t)| = 1$, $K_i(t) = $ `lead_stages`, no
degeneracy), so the refactor is behavior-preserving where the current model is
well-defined.

**Recourse-feasibility contract (independent-review finding; pre-existing,
widened by the refactor).** The delivered commitment is a **hard equality with no
slack** (the matured slot pins the plant's stage generation via an equality row),
while the commitment's bound comes from the **decision** stage's capacity and the
delivery stage enforces **its own** per-stage `[min, max]_gen`. If capacity drops
across the lead window (derating, maintenance schedules) — or a must-run floor
rises — the delivery subproblem is **infeasible**, and cobre generates no
feasibility cuts: that is a hard run failure, not a graceful degradation. (The
same mechanism is why the §4.1 collision case is fatal: two deciders impose two
contradictory equalities.) **Adopting one arm is a correctness PRECONDITION, not a
UX nicety (FCF review): an infeasible delivery subproblem violates relatively
complete recourse — an SDDP convergence hypothesis — so the feature is not sound
without it.** The arms: (i) **validation (recommended)** — reject a config where
any anticipated plant's generation bounds change within any of its active lead
windows, the loud-diagnostic house style; (ii) relax the delivery row to a
penalized bracket (changes shipped semantics); (iii) clamp the commitment bound to
the minimum capacity over its delivery window (silently conservative). This
applies to today's `lead_stages` model too — the refactor merely widens the
exposed window set on non-uniform calendars.

**End-of-horizon disclosure (symmetric to water's, opposite sign).** The horizon
gate drops the **decision** in the last $K_i$ stages — commitments that would
mature past $T$ are simply not made, so the plant dispatches **freely** there: an
_optimistic_ end-effect (under-costs the tail). Water's terminal drop is the
mirror image — the _value_ of a made release is lost, a _pessimistic_ end-effect
(documented in the companion memo §5). Both are accepted, documented
imprecisions; this one was previously undocumented.

**Commissioning-window corner (state-transfer review).** Two boundary rules the
plan must state alongside the horizon gate: (i) the per-stage bucket depth is
structurally capped, $L_{\text{arc}}(t) \le n_{\text{stages}} - t$ — no bucket slot
may target a stage past $T$ (validation row 3b's advisory covers the economics; the
cap is the sizing rule); (ii) an arc whose **downstream hydro exits**
(`exit_stage_id`) inside an arrival window is the commissioning analogue of the
horizon gate — in-transit water destined for an exited plant has no receiving
balance. Options mirror the horizon treatment: drop-with-advisory (consistent with
the terminal-drop convention) or reject at validation; the plan adopts one
explicitly, never silently.

**Mechanism fork (how the ring is realized) — surfaced, in-LP recommended.** The
scope items above assume the shipped **out-of-LP** shift is generalized in place.
§2.2 identifies the alternative that removes them at the root: realize the
anticipated ring **in-LP** with per-slot outgoing columns and definition rows, the
exact structure the water buckets use. Then per-stage $K_i(t)$, fan-out deposits,
and $K = 0$ are template-construction concerns (the template is already per-stage),
state propagation is uniformly "copy outgoing values," and
`shift_anticipated_state`'s constant-keyed arithmetic is deleted, not taught new
tricks. Costs: ~$A \cdot k_{max}$ extra columns/rows per stage LP and a changed
anticipated basis footprint; plus two lockstep consumers the state-transfer review
identified — the simulation output extraction reads the committed MW from the
**primal** at the matured slot's column (`compute_anticipated_decision_mw`), whose
role the in-LP ring changes, and **both** production shift call sites (forward
stage-solve and the simulation pipeline) switch from shift to plain copy in the
same increment. **Regression anchor either way:** on a uniform
calendar with $\Delta$ an exact stage multiple, the refactored model must produce
**identical optimal values, states, and cuts** to today's (a value/cut-identity
anchor — under the in-LP mechanism the LP itself is structurally different, so the
anchor is on solutions, not LP bytes; under the conservative out-of-LP mechanism
byte-identity of the LP is additionally achievable and should be pinned).

### 4.4 Config migration (surfaced fork)

| Option                                                    | Behavior                                                                                                                                                                                                          | Assessment                                                                                                                    |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **A. Time-based field, stage-count legacy (recommended)** | Add `lead_time` (physical duration) as the primary spec; keep `lead_stages` accepted, valid **only** when every stage in the plant's active window has equal length — else a hard validation error naming the fix | Non-uniform breakage becomes a loud diagnostic; uniform studies unchanged; one deprecation path                               |
| B. Time-based only (breaking)                             | Replace `lead_stages` outright                                                                                                                                                                                    | Cleanest end state; breaks every existing config now                                                                          |
| C. Energy-commitment semantics                            | Reinterpret the commitment as energy spread by the water $k_d$ math                                                                                                                                               | Unifies discretizations fully but silently changes shipped dispatch semantics — rejected unless the owner wants new semantics |

Units for `lead_time`: hours (matching $t_v$) with serde-side acceptance of
`{days: N}` sugar — decided at implementation planning, not here.

---

## 5. Strong / weak points (honest assessment)

**Strong:**

1. **DECOMP-consistent where DECOMP has an answer** (§1, four-point mapping;
   S1a reproduces the manual's worked factors exactly) and strictly more
   expressive where it does not (chronological block attribution).
2. **Exactness**: the bucket state is the textbook multi-lag cut in lifted
   coordinates (companion memo §2.2) — no approximation is introduced by the
   lifting itself; the only approximations are named and bounded (cross-stage
   origin-block aggregation, sub-contract 3; uniform-release density, shared
   with DECOMP).
3. **One invariant, two features**: the `block_mode`/`n_blks`-independent
   stage-clock state discipline now governs water buckets AND the repaired
   anticipated state; both reuse the shipped masking pattern and one overlap
   engine — less new machinery than either feature built alone.
4. **The anticipated fix is behavior-preserving on uniform calendars** and
   turns a silent wrong answer on non-uniform calendars into either a correct
   answer (delivery-anchored) or a hard diagnostic (legacy path).

**Weak / risks:**

1. **Exact overlap costs one bucket per declared arc even when negligible**
   (S2). Mitigation: setup advisory; arcs are opt-in. Accepted by owner.
2. **Cross-stage delivery drops origin-block↔arrival-block correlation**
   (sub-contract 3) — bounded, documented, and exactly what DECOMP's
   $d_k/D$ rule also drops.
3. **Uniform-release density is an assumption** for parallel-mode sending
   (blocks are load slices with no time identity). Shared with DECOMP;
   chronological mode removes it on the sending side.
4. **State-comparison with DECOMP requires the $M$-map translation**
   (different coordinates for the same model); policy-value comparisons are
   unaffected.
5. **Delivery-anchored anticipation is a real rework of the anticipated
   machinery**, not a parameter swap: multiple decision columns per stage
   ($|C(t)| > 1$) in the indexer/builder, fan-out deposits, and an explicit
   sub-stage-lead ($K = 0$) semantics hold under **either** §4.3 mechanism. The
   rework's _depth_ depends on that fork: the recommended **in-LP** ring
   **deletes** the shift code and **keeps** the single stage-0 cut-column map
   for the anticipated block (per-stage variation handled by masking, like
   storage/buckets); the conservative **out-of-LP** path instead reworks
   `shift_anticipated_state` and the column resolver in place, abandoning the
   stage-0 map for that block (§4.3's implementation-scope list). The masking
   _pattern_ carries over in both.
6. **The bucket-block insertion is likewise not a code no-op** on the water
   side: `state_to_lp_incoming_column`'s catch-all anticipated branch,
   `state_to_lp_column`'s guards, `set_nonzero_mask`'s loop order, and
   `n_state` must be rewritten together, pinned by a
   bucket-index-resolves-to-bucket-column regression (companion memo §3.1).
7. **CLP basis status-code latent bug** is amplified by any added cut
   dimensions (pre-existing; tracked in the companion memo §8.3).
8. **Two features, one resolver** couples their correctness: an overlap-engine
   bug hits both. Counterweight: one well-tested engine beats two divergent
   copies of the same arithmetic (the consistency contract, sub-contract 2,
   is testable in isolation).

---

## 6. What this changes in the companion memo and the plan

- The water-travel-time formulation is **unchanged** by this memo — §1 adds
  external validation, §3 adds the scenario evidence. Its §2.5.4 shared
  density, §2.5.5 exact-overlap resolution, and sub-contracts 1–3 all stand.
- The **implementation plan gains a shared first epic**: the temporal lag
  resolver (overlap engine + spread/point entry points + tests pinning S1a's
  DECOMP factors, **S1d's non-uniform depth counterexample**, and the §4.1
  per-stage $K(t)$ table), consumed by the water feature now and the
  anticipated refactor when scheduled.
- The **anticipated refactor is a separate feature** (own plan, own
  validation) that can land before, with, or after water travel time — the
  resolver is the only shared artifact. Recommended order: resolver + water
  first (greenfield, no behavior change elsewhere), anticipated refactor
  second (touches shipped behavior, needs its own regression net: the §4.3
  uniform-calendar value/cut-identity anchor — plus LP byte-identity if the
  conservative out-of-LP mechanism is chosen — and non-uniform
  previously-broken configs now diagnosed or fixed).
- **Outputs and Python parity (hard rule, previously unstated).** Travel time
  creates user-visible quantities: per-arc in-transit volumes (the bucket
  states — the `Afl_Tviagem` analogue DECOMP reports in its own diagnostics)
  and the delayed-arrival contribution to each downstream water balance.
  Whether these ship as new simulation output columns, a new per-arc output
  file, or stay internal is an **output-design decision for the plan** — but
  whatever is written by the CLI (`write_training_outputs` /
  `write_simulation_outputs`) must also be written by the Python bindings
  (`cobre-python`'s run paths), per the workspace parity rule. The IC input
  (`past_defluences`) rides the shared `cobre_io::Config`/schema path, so
  config-side parity is automatic; output-side parity is not and must be a
  ticket, not an afterthought.

Open decisions for the owner: §4.4 config-migration option (A recommended);
the §4.3 mechanism fork (in-LP ring recommended); whether the anticipated
refactor enters the same implementation plan as water travel time or a
follow-up plan (recommended: follow-up, shared resolver epic first); and the
output-design item above (what travel-time data ships in outputs).

---

## 7. Plan-enabling artifacts

The formulation above is adversarially settled; what converts it into a plan is
the verification net, the validation surface, and the config surface. All three
are specified here so the implementation tickets pin them, not invent them.

### 7.1 Analytic regression-case designs (the feature's deterministic cases)

Following the workspace convention (one analytic case per modeled feature, in the
deterministic-case family), four cases with hand-computable expectations. Each is
specified to be small enough to verify by hand and sharp enough that a plausible
implementation bug flips its expected value.

**W-1 — travel-time-off byte identity (anchor, not analytic).** Any existing
deterministic case, built twice: with no travel time declared, and with the
feature compiled in but no arc declared. Expected: byte-identical LP, cuts, and
outputs (`B = 0`; companion memo §2.5.6 anchor 1).

**W-2 — sub-stage delay, parallel, 2 stages (the bucket-value case).**
Two stages, uniform 720 h, one block, parallel. Cascade `u → j`: `u` with
100 hm³ initial storage, zero inflow; `j` run-of-river (zero storage), zero
inflow; both with toy productivity 1 MWh/hm³. One thermal at 10 $/MWh (ample
capacity), demand 200 MWh per stage, deficit never binding.
`travel_time_hours = 360` on the arc → $k_0 = k_1 = 1/2$, $L = 1$.
Hand-derivable optimum (verified by enumeration): release **everything at
stage 1** ($x_1 = 100$) — stage-1 release delivers both its halves inside the
horizon, stage-2 release loses its $k_1$ half past the horizon (the deferred
terminal drop, §0.1 decision 3 of the companion memo). Expected values to pin:
total cost **2000 $** (thermal 50 MWh at stage 1 + 150 MWh at stage 2); `j`
receives 50 hm³ at stage 1 (same-stage share) and 50 hm³ at stage 2 (bucket
delivery); water value of `u` storage **−20 $/hm³** (one marginal hm³ → 1 MWh
at `u` + ½ at `j` stage 1 + ½ at `j` stage 2); **bucket subgradient −10 $/hm³**
(one in-transit hm³ → 1 MWh at `j` stage 2, displacing thermal). The bucket
dual is the case's teeth: `rc/col_scale` extraction on the incoming bucket
column must produce exactly −10. **The test must assert the −10 dual and the
50/150 per-stage thermal split, never total cost alone** — a fold
implementation (no bucket, crossing half absorbed same-stage) also reaches
total cost 2000, so total cost is fold-blind; the dual (fold has no bucket
column) and the per-stage split (fold's stage-2 thermal differs) are the
discriminators. Missing delivery is caught by cost (2500 ≠ 2000); a
wrong-sign coefficient by the dual (+10 ≠ −10).

**W-3 — mixed calendar depth (the S1d resolver case).** One 720 h monthly
stage then three 168 h weekly stages (plus a tail stage as needed);
`travel_time_hours = 360`. Pins the resolver table directly: monthly anchor →
$k_0 = 1/2$, $k_1 = 7/30$, $k_2 = 7/30$, $k_3 = 1/30$, depth **3** (the
closed-form $\lceil t_v/h_t \rceil = 1$ would drop 8/30 of the release);
per-stage depths vary and the global max sizes the block with masked slots.
End-to-end assertion: conservation — total delivered downstream + horizon drop
= total released, per arc, to fp tolerance.

**W-4 — chronological attribution + K=1 parity (the χ case).** The owner's
touchstone: one 720 h stage with 3 × 240 h chronological blocks,
`travel_time_hours = 250`, followed by one more stage. Pins the block tables of
the companion memo's example (iii): $\kappa_{B0 \to B1} = 230/240$,
$\kappa_{B0 \to B2} = 10/240$, $\kappa_{B1 \to B2} = 230/240$,
$\chi = (0,\ 10/240,\ 1)$, delivery $\rho = (96\%,\ 4\%)$, duration-weighted
deposit $= 250/720$. Companion anchor: the same system with $K = 1$ (single
720 h block) must be **byte-identical to parallel mode with travel time ON**
($\chi_{0,d} = k_d$; companion memo §2.5.2 fact 3).

**A-1 — anticipated resolver + refactor anchor.** (a) Resolver unit tests, two
**separate** fixtures: (i) the PMO collision/gap calendar (4 weekly + ~30-day
monthly, $\Delta = 30$ d) — expected delivery-anchored $c(m)$ / $K(t)$ / $C(t)$
values **computed from the calendar** (note: §4.1's table is the
decision-anchored breakage illustration, not these outputs — the resolver's
delivery-anchored values are derived, not read off it), including the
IC-boundary deliveries; (ii) a 31-day-month fixture with a 30-day lead pinning
the $K = 0$ degeneracy and its §4.3 exclude-with-diagnostic handling. (b)
Refactor anchor: any existing anticipated deterministic case on a uniform
calendar — identical optimal values, states, and cuts before/after the
refactor (plus LP-byte identity if the conservative out-of-LP mechanism is
chosen; §4.3).

### 7.2 Validation-rule inventory (consolidated)

| #   | Condition                                                                                                                                                            | Response                                                                                                                                                                                                                                                                                                                |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `travel_time_hours` negative or non-finite                                                                                                                           | hard error at config validation                                                                                                                                                                                                                                                                                         |
| 2   | `travel_time_hours = 0` declared                                                                                                                                     | treat as undeclared (today's instantaneous model) + advisory log — never a bucket                                                                                                                                                                                                                                       |
| 3   | $\max_t t_v/h_t$ below a smallness threshold                                                                                                                         | advisory log ("consider not declaring"); never a silent fold (§2.5.5 of the companion memo)                                                                                                                                                                                                                             |
| 3b  | $t_v$ exceeding the remaining horizon at some stage (all cross-stage water drops past $T$)                                                                           | advisory log ("travel time exceeds the horizon from stage $t$; the arc is economically inert there") — sizing is safe ($L_{\text{arc}}(t)$ capped by $n_{\text{stages}} - t$), so advisory, not error                                                                                                                   |
| 4   | `past_defluences` history shorter than $t_v^{\max}$ (REQUIRE option)                                                                                                 | hard error naming the missing periods; under the derive-fallback, logged caveat instead                                                                                                                                                                                                                                 |
| 5   | legacy `lead_stages` with unequal stage lengths in the plant's active window                                                                                         | hard error naming `lead_time` as the fix (§4.4 option A)                                                                                                                                                                                                                                                                |
| 6   | `lead_time` yielding $K(t) = 0$ at some stages                                                                                                                       | per-stage diagnostic; plant unconstrained at those delivery stages (§4.3 fork, recommended arm)                                                                                                                                                                                                                         |
| 7   | policy load: recorded `state_dimension` ≠ current `n_state`                                                                                                          | hard error (the NEW load-time check — companion memo §6 item 5)                                                                                                                                                                                                                                                         |
| 8   | $\sum_d k_d \ne 1$ per (arc, stage) after resolution                                                                                                                 | `debug_assert` (conservation contract, companion memo §6.1)                                                                                                                                                                                                                                                             |
| 9   | $\sum_b w_b \chi_{b,d} \ne k_d$ per (arc, stage, lag)                                                                                                                | `debug_assert` (shared-density consistency, sub-contract 2)                                                                                                                                                                                                                                                             |
| 10  | anticipated plant's `[min, max]_gen` changes within any of its active lead windows                                                                                   | hard error (the §4.3 recourse-feasibility contract, recommended arm — a delivered hard-equality commitment can otherwise be infeasible at the delivery stage; no feasibility cuts exist)                                                                                                                                |
| 11  | downstream hydro of a declared arc exits (`exit_stage_id`) inside an arrival window                                                                                  | surfaced fork (§4.3 commissioning corner): drop-with-advisory (mirrors the terminal-drop convention) or hard error — the plan adopts one explicitly, never silently                                                                                                                                                     |
| 12  | declared arc's downstream plant in `PreFilling`/`Filling` — or **before its entry window** (absent from the LP entirely, no balance row) — during any arrival window | correctness precondition (both FCF reviews — frozen or absent relief valves make a pinned arrival infeasible or silently non-conserving; no feasibility cuts): **validation** (reject) or **routing** (deliver via the incremental-inflow short-circuit) — one must land; water memo §3.3 recourse is conditional on it |

### 7.3 Config surface sketch (decided at planning, constrained here)

- **Water:** `travel_time_hours: f64` on the **upstream hydro's cascade arc**
  (its outflow edge to `downstream_id`) — one scalar per arc, matching the
  DECOMP TVIAG pairing. **v1 scope: the main cascade arc only; diversion arcs
  excluded** (surfaced — a diversion travel time is the same machinery on
  another arc set, deferred until a case needs it). Density-first resolver per
  §2.3 makes a future per-arc curve a config-only extension.
- **Anticipated:** `anticipated_config.lead_time` (physical duration, hours;
  calendar-sugar forms decided at planning) added alongside the legacy
  `lead_stages` per §4.4 option A.
- **IC:** `past_defluences` in the initial-conditions input, per arc, per
  pre-study calendar period following the `past_inflows` period convention
  (companion memo §4.2). Two wire constraints (state-transfer review): the field is
  **appended at the end** of the `InitialConditions` struct (postcard is positional —
  mid-struct insertion breaks the MPI broadcast wire), and its seed precompute is a
  **fresh calendar-agnostic sibling** of `compute_recent_observation_seed`, not a
  call into it (that path is monthly-cycle-gated and day-granular, §2.1).
- All three ride the shared `cobre_io::Config`/schemars path, so config-side
  Python parity and schema regeneration are automatic; only **outputs** need
  the explicit parity ticket (§6).

---

## 8. Owner decision sheet (the complete gate before planning)

Every open decision, consolidated. Each is surfaced in detail at the cited
section; this sheet is the action interface. D1–D8 are open; D0 is a standing
prior decision listed for completeness.

| #   | Decision                              | Options                                                                                                                                | Recommendation                                                                                                                                                                                                                                                                                                                                                                                      | What it changes downstream                                                                                                                                                                             |
| --- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| D1  | Shipping order (water §8.5)           | both-together (skeleton + block refinement, one landing) vs core-first (stage-uniform chronological attribution as documented interim) | **both-together**                                                                                                                                                                                                                                                                                                                                                                                   | plan epic structure; whether an interim distortion ships. De-risked: the state-transfer machinery is identical under both arms (χ/κ/ρ are within-stage coefficients that never touch the state vector) |
| D2  | Anticipated config migration (§4.4)   | A: add time-based `lead_time`, legacy `lead_stages` valid only on uniform windows · B: breaking replace · C: energy semantics          | **A**                                                                                                                                                                                                                                                                                                                                                                                               | thermals.json schema; validation row 5; deprecation path                                                                                                                                               |
| D3  | Ring mechanism (§4.3, §2.2)           | in-LP ring (per-slot definition rows; deletes the shift, keeps the stage-0 map) vs conservative out-of-LP rework                       | **split expert opinion** — state-transfer review: in-LP (unifies with storage, less code, code-validated against two precedents); FCF review: out-of-LP (in-LP changes the anticipated LP shape even on uniform calendars — larger blast radius for a behavior-touching follow-up whose unification payoff is then speculative). Both are correct; the choice is risk appetite for the D6 follow-up | whether `shift_anticipated_state` is deleted or generalized; anchor type (value/cut vs + LP-byte); blast radius of the anticipated follow-up plan                                                      |
| D4  | Sub-stage lead $K=0$ semantics (§4.3) | exclude-from-anticipation + per-stage diagnostic vs hard validation error                                                              | **exclude + diagnostic**                                                                                                                                                                                                                                                                                                                                                                            | validation row 6; behavior on mixed calendars with coarse stages                                                                                                                                       |
| D5  | Water IC input (water §4.3)           | REQUIRE `past_defluences` with derived-from-`past_inflows` fallback vs derive-only vs zero-seed                                        | **REQUIRE + derived fallback**                                                                                                                                                                                                                                                                                                                                                                      | initial-conditions schema; validation row 4; first-$L$-stages accuracy                                                                                                                                 |
| D6  | Refactor sequencing (§6)              | anticipated refactor as follow-up plan sharing the resolver epic vs same plan as water                                                 | **follow-up plan**                                                                                                                                                                                                                                                                                                                                                                                  | plan boundaries; regression-net timing                                                                                                                                                                 |
| D7  | Output design (§6)                    | what travel-time data ships: bucket states and/or delayed-arrival contributions; new columns vs per-arc file vs internal-only          | **open — owner picks visibility**; parity ticket either way                                                                                                                                                                                                                                                                                                                                         | simulation output schema; the Python-parity ticket                                                                                                                                                     |
| D8  | Recourse-feasibility arm (§4.3)       | validation (reject bounds-change within active lead windows) vs penalized bracket vs clamp-to-window-min                               | **validation** — and adopting SOME arm is a correctness precondition (relatively complete recourse), not optional                                                                                                                                                                                                                                                                                   | validation row 10; applies to the CURRENT `lead_stages` model too                                                                                                                                      |
| D0  | Terminal credit (standing)            | water §5 / §0.1 decision 3: defer `V_eff`; residual horizon buckets dropped, documented (broadened to every arc, §8.4 lock #3)         | previously locked — no action unless owner reopens                                                                                                                                                                                                                                                                                                                                                  | —                                                                                                                                                                                                      |

Accepting all recommendations resolves D1–D6 and D8; D7 (output visibility)
has no default and needs an explicit choice. With the sheet answered, both
memos feed the implementation plan directly.
