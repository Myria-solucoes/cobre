# Hydro filling: per-stage volume-target model (DECOMP-aligned)

> **Status**: Shipped — this reformulation is the live system. This document **replaces the earlier
> reset-based draft of the same file** (the entry-stage state-reset reformulation),
> which is abandoned in favor of the DECOMP-aligned minimum-rate / continuous-handoff
> model below. Companion to `hydro-dead-volume-filling.md` (the shipped v1).
>
> **Relative to v1** it supersedes §3.1 (impound cap → minimum-rate floors), §3.3
> (terminal `σ_fill` → per-stage targets), §5.3 (`filling_inflow_m3s` semantics),
> §5.4 (validation), and the §7/§8 entries for filling-inflow semantics and `σ_fill`
> placement. It **keeps** v1's §3.2 (PreFilling short-circuit, minus the dropped
> cap-row port), §3.4 (the soft operating floor — kept, family renamed, penalty
> ordering flipped), §3.5 (FPHA exclusion), §3.6 (gating), §3.7 (continuous handoff),
> §5.1 (no new field), and §5.2.
>
> **Reference**: DECOMP's _enchimento de volume morto_ — Manual de Referência §4.5.9
> and Manual do Usuário §3.4.6.10 (registers VM/DF). This model adopts DECOMP's
> minimum-target-volume mechanism and continuous handoff. It deliberately does **not**
> mirror DECOMP's separate DF (filling min-outflow) register — cobre's per-stage
> `min_outflow_m3s` already covers it. CEPEL's Portuguese register/field names are
> **not** inherited; cobre uses English identifiers throughout (§7).
>
> **Contracts it must not disturb**: `.claude/rules/sddp.md` (Benders cut sign/scale,
> column-bound state pinning, append-only cut pool with slot-identity warm start) and
> the `filling_phase` predicate in `lp/builder/mod.rs`.

## TL;DR

A commissioned hydro fills toward its dead volume (`min_storage_hm3`) at a per-stage
**minimum accumulation rate** (`filling_min_rate_m3s`). From that rate the model
derives a per-stage **minimum target-storage trajectory** computed **backward** from
`min_storage`; storage falling below the trajectory is penalized by a soft floor.
Filling is **uncapped** — over-filling and catch-up are allowed (the reservoir may
fill faster than the minimum schedule if inflow and future value warrant it).

At `entry_stage_id` the end-of-filling storage flows into operation **continuously**
(no reset, no pin). In operation the hydro uses the **same `min_storage` threshold as
every other hydro**, but realized as a **soft floor + high penalty** rather than a
hard column bound — because, having just filled, it may legitimately enter operation
near or (under genuine starvation) below the dead volume, and a hard bound would be
infeasible there. The high penalty makes the soft floor behave like the hard floor in
every feasible scenario, so the filled hydro is indistinguishable from a never-filled
hydro in operation. This is the **only** asymmetry, and it is scoped to filled hydros
and self-limiting (§3.3, §9).

Two penalties, ordered so the operating floor dominates the fill schedule (§5):
`storage_violation_below_cost ≥ filling_target_violation_cost`, with the operating
floor ideally `≥ deficit_cost` (it mimics a hard floor); the fill schedule is a
softer, lower-priority obligation.

The model reuses cobre's existing inputs (no new field; `min_outflow_m3s` covers the
filling period) and existing LP machinery (the `filling_target` row/column family,
widened from terminal-only to per-stage; the renamed `filled_min_storage_floor`
family). Cut validity is **v1's** — there is no reset, so no new dual-extraction trap.

---

## 1. Relationship to DECOMP

DECOMP represents dead-volume filling through a per-stage **minimum storage flow rate**
(register **VM**), from which it computes **minimum target volumes** such that the
stored volume at each stage is _at least_ those values (Ref. §4.5.9: "calcula volumes
meta mínimos de modo que o volume armazenado … seja, no mínimo, igual a estes
valores"). The filling deadline is an input; a shortfall is reported, not erased.

| DECOMP                                                    | This model                                         | Decision                                       |
| --------------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------- |
| Per-stage **minimum** fill rate (VM)                      | `filling_min_rate_m3s` (minimum accumulation rate) | **adopt** (flips v1's _cap_ semantics)         |
| Cumulative **minimum target volumes** `≥` floor           | per-stage `V_target` soft floors, **uncapped**     | **adopt** (catch-up/over-fill allowed)         |
| **Continuous** handoff into operation; shortfall reported | continuous handoff; soft operating floor           | **adopt** (no reset)                           |
| Separate **filling min-outflow** (register DF)            | reuse the existing per-stage `min_outflow_m3s`     | **decline the duplicate** — one flexible field |
| Filling deadline (prazo)                                  | `entry_stage_id`                                   | already present                                |

The one place this model is _stricter_ than a naive DECOMP reading: the operating
`min_storage` floor is driven by a high penalty so it behaves like the hard floor of
a never-filled plant (DECOMP reports the violation; cobre additionally makes it
expensive enough to avoid in practice — §3.3, §5).

## 2. What changes

| Concern                 | Shipped v1                                                      | This model                                                                                |
| ----------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `filling_inflow_m3s`    | per-stage retention **cap** (max impound)                       | `filling_min_rate_m3s` — per-stage **minimum** accumulation rate (§4)                     |
| Filling schedule        | single terminal `σ_fill` at `entry−1`                           | per-stage **minimum target floors** `V_target[t]`, backward-anchored (§3.1)               |
| Impound cap row         | retention cap (equality) + PreFilling-upstream port             | **removed** — uncapped; PreFilling routes onto the balance row only (§3.4)                |
| Handoff                 | continuous (pin chain)                                          | **continuous (no reset)** — kept (§3.2)                                                   |
| Operating floor         | soft `σ^{v-}`, `filling_floor` family                           | soft floor at `min_storage`, **`filled_min_storage_floor`** family — kept, renamed (§3.3) |
| Filling min-outflow     | existing `min_outflow_m3s`                                      | **existing `min_outflow_m3s`** — kept (§4)                                                |
| Penalties               | `filling_target_violation_cost`, `storage_violation_below_cost` | **both kept**, ordering **flipped** (§5)                                                  |
| Fill anchor / new field | (implicitly `min_storage`)                                      | `min_storage_hm3` — **no new field** (§4)                                                 |

Relative to the **abandoned reset-based draft**: this model **drops** the §3.2
entry-stage reset / RHS-fold, **drops** the proposed `initial_operating_volume_hm3`
field, **keeps** `storage_violation_below_cost` (the reset draft removed it), and uses
a **minimum-floor `≥`** rather than an increment **equality**.

## 3. Formulation

The three-phase lifecycle (PreFilling → Filling → Operating), keyed on `stage.id`, is
unchanged from v1 §1. The dense storage column, declaration-order invariance, and the
subgradient contract (`extract_duals_from_view`: cut coefficient = incoming-storage
reduced cost ÷ `col_scale`, never synthesized) all hold unchanged.

### 3.1 Filling: per-stage minimum target floors (backward-anchored)

`filling_min_rate_m3s` (`= rate[t]`, m³/s) is the **minimum rate at which the
reservoir must accumulate storage** at filling stage `t`. With `ζ_t = layout.zeta` the
per-stage m³/s→hm³ factor, the per-stage minimum target storage is computed **backward**
from the dead volume. Let `L = entry_stage_id − 1` be the last filling stage:

```
V_target[L] = min_storage_hm3
V_target[t] = V_target[t+1] − ζ_{t+1} · rate[t+1]      (start_stage_id ≤ t < L)
            = min_storage_hm3 − Σ_{s=t+1}^{L} ζ_s · rate_s
```

Each filling stage carries one soft floor on the **outgoing storage**:

```
v_out[t] + σ_fill[t] ≥ V_target[t],   σ_fill[t] ≥ 0,
cost(σ_fill) = filling_target_violation_cost
```

- **Uncapped.** There is no upper bound on the rise. The reservoir may exceed
  `V_target[t]` (fill ahead of schedule) and may catch up after a lagging stage; only
  _falling below_ the trajectory is penalized. This mirrors DECOMP's `≥` minimum
  targets and matches the physics (no reason to forbid filling faster when inflow and
  the future water value warrant it; the floor guarantees minimum progress).
- **Backward anchoring** makes the last floor exactly `min_storage` (the reservoir is
  softly required to reach the dead volume by entry) without a separate sum-to-fill
  check, and clips the trajectory at `min_storage` so over-provisioned rates relax the
  _earliest_ floors to slack rather than demanding over-fill.
- **Generation, turbine, FPHA, diversion** are off during Filling (v1 §3.6, unchanged).
- **Min-outflow** during Filling is served via the existing `min_outflow_m3s` and its
  `outflow_violation_below` slack (v1 §3.1, unchanged).
- **Realization**: reuse v1's `filling_target` row/column family, widened from
  terminal-only to **every filling stage** (`identify_filling_target_hydros` fires for
  `start_stage_id ≤ id < entry_stage_id`), with per-stage RHS `V_target[t]`. v1's
  impound-cap retention row (`fill_filling_retention_rows`) is **removed**.

### 3.2 Continuous handoff (no reset)

The end-of-filling storage flows into the first operating stage through the existing
incoming-state pin chain (v1 §3.7) — **no reset, no pin to a fixed value**. The last
filling floor (`V_target[L] = min_storage`) softly targets the reservoir to reach the
dead volume by `entry_stage_id`; whatever it actually reaches is what operation starts
from. The Benders storage coordinate is therefore **never decoupled** at the boundary,
so the cut machinery is identical to any normal stage-to-stage transition (§6).

### 3.3 Operating: soft `min_storage` floor for filled hydros (`filled_min_storage_floor`)

A never-filled hydro pins `min_storage_hm3` as a **hard** column lower bound. A hydro
that went through filling uses the **same `min_storage_hm3` threshold**, but realized
as a **soft floor + slack**:

```
v_out + σ^{v-} ≥ min_storage_hm3,   σ^{v-} ≥ 0,
cost(σ^{v-}) = storage_violation_below_cost
```

at **every** operating stage of the filled hydro. The threshold is not a new "recovery"
level — it is the plant's actual `min_storage`. The floor is soft (not a hard bound)
because, with no reset, the hydro may enter operation at or below `min_storage` under
genuine starvation, where a hard bound would be infeasible at the first operating stage.

> **Contract — `min_storage` is hard for every hydro EXCEPT one filled during the
> study.** The discriminator is `hydro.filling.is_some()`, evaluated per-hydro; carry
> it as a comment at the construction site. The high `storage_violation_below_cost`
> (§5) drives `σ^{v-} → 0` in every feasible scenario, so the soft floor behaves like
> the hard floor and the filled hydro is indistinguishable from a never-filled hydro
> in operation. Do **not** globalize this soft floor to all hydros (it would let the
> optimizer cheaply violate dead volume system-wide), and do **not** make it hard for
> filled hydros (it would re-introduce first-operating-stage infeasibility).

Family: **`filled_min_storage_floor`** (was v1's `filling_floor`) — see §7.

### 3.4 PreFilling cascade short-circuit (cap-port dropped)

The PreFilling short-circuit (`fill_prefilling_shortcircuit`) is unchanged from v1
§3.2: the absent dam's incremental inflow, transitive upstream releases, and
withdrawal route onto its first non-PreFilling downstream's **water-balance row**.
Because the impound-cap row is removed (§3.1, §2), the v1 cap-row port (the commit that
"counts PreFilling-upstream inflow in the impound cap") is **dropped** — routed inflow
lands on the balance row only, and there is no cap row to receive it.

### 3.5 FPHA exclusion and generation/turbine/diversion/evaporation gating

Unchanged from v1 §3.5–§3.6: FPHA rows excluded at PreFilling/Filling; generation and
turbine pinned `[0,0]` at PreFilling/Filling; diversion `[0,0]` for filling hydros in
all phases; evaporation zero in PreFilling, normal in Filling and Operating.

## 4. Input model

- **`filling_min_rate_m3s`** (renamed from `filling_inflow_m3s`, on `FillingConfig`):
  per-stage minimum accumulation rate (m³/s). Semantics: the minimum rate at which the
  reservoir must fill; it generates the per-stage minimum target trajectory (§3.1).
  Validated `≥ 0`. The resolved per-stage hydro bounds already carry this value
  per stage, so a per-stage schedule needs no new wiring.
- **No new field.** The fill anchor is `min_storage_hm3` (existing). The reset draft's
  `initial_operating_volume_hm3` is **not** added; `min_storage` serves double duty as
  both the backward-trajectory anchor (filling) and the floor threshold (operating).
- **Seed**: `InitialConditions.filling_storage` (existing) — `0` (empty pit) when
  `start_stage_id > 0`, or a partial level in `[0, min_storage)` when `start_stage_id
== 0` (study starts mid-filling).
- **Min-outflow during filling**: the existing per-stage `min_outflow_m3s`. A user who
  wants a different minimum flow during filling than during operation sets different
  per-stage values in that one field — no DF-style duplicate input.
- **Validation — `≥` sufficiency check.** The minimum-rate schedule must be able to
  reach the dead volume from the seed:

  ```
  Σ_{s=start_stage_id}^{entry_stage_id−1} ζ_s · rate_s  ≥  (min_storage_hm3 − seed)
  ```

  One-sided (over-provisioning is allowed — it relaxes the earliest floors to slack;
  only under-provisioning is rejected, since backward folding would then demand more
  gain than one minimum-rate stage provides). This replaces the reset draft's
  two-sided exact-equality (no float `==`). It needs resolved per-stage `ζ_s`, so it
  lives in the **ζ-aware** validation home — the cobre-io semantic validator or a
  cobre-sddp setup check, **not** cobre-core (which is layout-agnostic by the
  infra-genericity rule). The ζ-free guards (`start_stage_id < entry_stage_id`, seed
  in `[0, min_storage)`, entry-only / no `exit_stage_id` for filling hydros) stay in
  `validate_filling_configs`.

## 5. Penalties: two costs, operating floor dominant

Two distinct events, two distinct penalties — kept separate, ordered so the operating
floor is the most sacred:

```
filling_target_violation_cost   <   deficit_cost   ≤   storage_violation_below_cost
       (best-effort fill schedule)                        (mimics the hard floor)
```

- **`storage_violation_below_cost`** (operating `σ^{v-}`): should be the highest
  relevant penalty — ideally `≥ deficit_cost` — so under scarcity the optimizer sheds
  load before drawing a filled hydro below its dead volume, exactly as a never-filled
  hydro's hard floor would force. This is what makes the filled hydro behave like the
  others.
- **`filling_target_violation_cost`** (filling `σ_fill`): a softer, lower-priority
  schedule penalty; it can sit _below_ `deficit_cost` (do not shed load merely to keep
  a new reservoir on its fill schedule).
- **Ordering check.** v1's `check_penalty_ordering` is a strict tier chain
  (`filling_target > storage_below > deficit > constraint > resource > 0`). **Reorder
  it** to `storage_violation_below_cost > deficit_cost > filling_target_violation_cost`:
  keep the existing `storage_below > deficit` and `deficit > constraint > resource > 0`
  checks, and **replace** v1's `filling_target > storage_below` check with
  `deficit_cost > filling_target_violation_cost` — the fill schedule is **not as hard as
  load shedding**. `storage_below > filling_target` then holds transitively, and the
  existing `storage_below > deficit` check already supplies the "mimics the hard floor"
  comparison, so **no separate/duplicate deficit advisory is added**.
  `filling_target`'s position relative to the operational-constraint tier is **left to
  the study's calibration** (whether filling outranks routine operational slacks, or a
  mandatory min-outflow-during-filling outranks filling, is regime-dependent — the
  validator does not pin it). All checks stay `ModelQuality` (non-blocking); both
  penalty fields are **kept**.
- **Why not a single penalty.** Merging the two would force the fill schedule to be as
  sacred as the dead-volume floor: a single penalty high enough to mimic the hard
  operating floor would also make the optimizer shed load to stay on the fill
  schedule — over-prioritizing a best-effort ramp against real load. Keeping them
  separate lets the operating floor be hard-like while the schedule stays soft.
- **Calibration.** A "very high" `storage_violation_below_cost` interacts with
  `COST_SCALE_FACTOR`; choose it large enough to dominate, not so large it degrades LP
  conditioning.

## 6. Cut validity

Identical to v1 — there is no reset, so the riskiest mechanism of the abandoned draft
(the entry-stage RHS-fold and its stale-`β` trap) **does not exist here**.

- The storage coordinate keeps its dense index and is never decoupled or relocated at
  any phase boundary; the incoming-state column resolves through the same
  `StateLayout::state_to_lp_incoming_column` map, so cut coefficients are
  phase-invariant.
- The per-stage filling floors (`σ_fill`) and the operating floor (`σ^{v-}`) both put
  `+1` on `v_out` only (the same shape as v1's `σ_fill`/`σ^{v-}` rows). Their duals
  fold into the **single** incoming-storage reduced cost that `extract_duals_from_view`
  already harvests — never hand-extract a soft-row dual.
- Continuous handoff means `β` crosses the entry boundary normally (the filling
  predecessor's storage _does_ influence the future cost — through `σ_fill` within
  filling and, on shortfall, through `σ^{v-}` after entry — which is physically
  correct, unlike the reset's flat cut).
- The append-only cut pool keeps every cut at a stable slot; no cut row is relocated by
  a phase change, so slot-identity warm-start reconstruction is unaffected.
- **Lower-bound parity**: `evaluate_lower_bound` must apply the same per-stage template
  gating as forward/backward (the NCS-class precedent). All gating is template-driven
  and stage-deterministic, so forward/backward/lower-bound/simulation inherit it from
  the shared per-stage template — no solve-time patch.

## 7. Naming

CEPEL's Portuguese register names (VM, DF) and the term _meta_ (target/goal) are not
inherited. Math symbols may stay terse/Greek; code identifiers must be unambiguous.

| Concept                                                      | Identifier                                                             |
| ------------------------------------------------------------ | ---------------------------------------------------------------------- |
| Min accumulation-rate input                                  | `filling_min_rate_m3s` _(renamed from `filling_inflow_m3s`)_           |
| Rising fill schedule (filling phase)                         | `filling_target` family _(kept; widened to per-stage)_                 |
| Soft `min_storage` floor for filled hydros (operating phase) | **`filled_min_storage_floor`** family _(renamed from `filling_floor`)_ |
| Per-stage target storage (math)                              | `V_target[t]`                                                          |
| Filling shortfall slack (math)                               | `σ_fill[t]`                                                            |
| Operating-floor slack (math)                                 | `σ^{v-}`                                                               |
| Filling penalty (softer)                                     | `filling_target_violation_cost` _(kept)_                               |
| Operating-floor penalty (dominant, `≥` deficit)              | `storage_violation_below_cost` _(kept)_                                |

Unchanged, already-correct names: `filling_phase`, `fill_prefilling_shortcircuit`,
`FillingConfig`, `InitialConditions.filling_storage`. Removed entirely (name moot):
`fill_filling_retention_rows` (the impound-cap row).

## 8. Migration

- **From v1**: rename `filling_inflow_m3s → filling_min_rate_m3s` and flip its
  semantics (cap → minimum rate); widen `identify_filling_target_hydros` from terminal
  to every filling stage and set the per-stage RHS to `V_target[t]`; remove
  `fill_filling_retention_rows` (and its PreFilling-upstream port); rename the
  `filling_floor` family → `filled_min_storage_floor`; flip `check_penalty_ordering`.
  Keep continuous handoff, the soft operating floor, FPHA exclusion, gating, and the
  PreFilling short-circuit.
- **From the abandoned reset draft**: drop the entry-stage reset / RHS-fold; drop the
  `initial_operating_volume_hm3` field; keep `storage_violation_below_cost` (do **not**
  remove it); use the minimum-floor `≥` instead of the increment equality.
- **Doc-comments**: rewrite `FillingConfig.filling_min_rate_m3s` (minimum accumulation
  rate, not an inflow and not a cap); add the §3.3 `min_storage`-hard-except-filled
  Contract at the `filled_min_storage_floor` construction site.
- **Cases / baselines**: re-derive `d38` (per-stage `σ_fill` floors + continuous start,
  no reset) and `d39` (PreFilling-upstream regression — routed inflow on the balance
  row only); add a `d40` filling-cascade case (a Filling reservoir downstream of
  another Filling reservoir — confirm each carries its own per-stage floor and they
  couple only through normal cascade releases). Re-confirm parity-neutrality for the
  non-filling world. Re-baseline both backends across the three pinned baseline
  locations.

## 9. Accepted trade-offs

- **One asymmetry, scoped and self-limiting.** A filled hydro carries a soft
  `min_storage` floor in operation while every other hydro has a hard one (§3.3). This
  is the price of the continuous handoff (no reset). It is scoped to filled hydros (the
  only ones that can legitimately enter operation near the dead volume) and neutralized
  in practice by the high `storage_violation_below_cost`, which keeps `σ^{v-} → 0`
  except where a hard floor would itself be infeasible (genuine starvation). The
  semantic validator can warn (`ModelQuality`) when a filled hydro's
  `storage_violation_below_cost` is not large relative to `deficit_cost`.
- **More faithful than the reset.** Unlike the abandoned reset draft, this model never
  invents boundary water: a filling shortfall carries forward continuously and is paid
  as a real recourse cost, matching DECOMP's continuous handoff and reported violation.
- **Filling cascades** (a Filling reservoir downstream of another Filling reservoir):
  no special handling — each carries its own per-stage floor; they couple only through
  normal cascade releases (the upstream's release is the downstream's balance-row
  inflow; the floor rows carry no inflow term). Confirm with the `d40` case (§8).
