# Hydro dead-volume filling and commissioning

> **Status**: Shipped (v1). Authoritative specification for commissioning windows
> on hydro reservoirs and the dead-volume filling lifecycle. The shipped successor,
> `hydro-filling-volume-target-reformulation.md`, supersedes the formulation
> sections §3.1 (impound cap → minimum-rate floors) and §3.3 (terminal `σ_fill` →
> per-stage targets); the input-model sections §5.3 (`filling_inflow_m3s` →
> `filling_min_rate_m3s`) and §5.4 (validation); the §7 case assertions; and the two
> §8 rows flagged **[Superseded]** below (filling-inflow semantics, `σ_fill`
> placement). It **keeps** §3.2 (PreFilling short-circuit, minus the cap-row port),
> §3.4 (the soft operating floor — kept, family renamed `filled_min_storage_floor`,
> penalty ordering flipped), §3.5, §3.6, §3.7 (continuous handoff), §5.1 (no new
> field), §5.2, and the remaining §8 decisions. The successor has shipped;
> those superseded sections are now governed by
> `hydro-filling-volume-target-reformulation.md`.
> Hydro is the last and hardest of the six commissionable entity
> types: its storage is a Benders **state coordinate**, so it cannot use the
> column-omission/zero-influence mechanism the other five use — it must keep its
> column and suspend only its _operation_, and it carries a real filling feature
> on top.
>
> **Authoritative external spec** (`~/git/cobre-docs`):
> `src/specs/math/system-elements.md` §"Dead-Volume Filling" and
> `src/specs/math/penalty-system.md` §"Dead-Volume Filling Specifics" — CEPEL's
> _enchimento de volume morto_. This document grounds and refines that spec
> against the live LP builder, and records two corrections to a naive reading
> (§3.1, §3.2).
>
> **Contracts it must not disturb**: `.claude/rules/sddp.md` (Benders cut
> sign/scale, column-bound state pinning, FPHA average storage, append-only cut
> pool with slot-identity warm start) and the shared `commissioning_active`
> predicate in `lp/builder/mod.rs` that the other five entity types use.

## TL;DR

A commissioned hydro passes through **PreFilling → Filling → Operating**. The
storage state column exists in every phase (the state dimension is fixed); what
changes per phase is the _physics attached to it_, derived inline from
`FillingConfig.start_stage_id` and `Hydro.entry_stage_id` — no cached per-stage
masks. Two facts make hydro special:

1. **Filling inflow is a retained _portion_ of the natural inflow** (impounded,
   removed from the cascade), not external or replacement water. PAR and AR-lag
   coupling run normally during filling.
2. **PreFilling is a cascade _short-circuit_** — before the dam exists, the river
   flows past its site, so the absent reservoir's inflow, upstream releases, and
   withdrawal all transfer to its downstream neighbor.

Cut validity holds across all phase boundaries because the storage column is
never omitted or relocated, and every cut coefficient is the storage column's
reduced cost divided by `col_scale` (never synthesized).

---

## 1. The lifecycle

A hydro with no `FillingConfig` is **Operating** at every stage — bit-identical
to today (the parity-neutrality contract). A commissioned hydro carries a
`FillingConfig { start_stage_id, filling_inflow_m3s }` and an `entry_stage_id`,
and passes through three phases keyed on the stage **id** (`stage.id`, not the
stage index):

| Phase          | Stage ids                                      | Meaning                                                                                                                      |
| -------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **PreFilling** | `start_stage_id > 0` and `id < start_stage_id` | The dam does not exist yet; the river flows past its site.                                                                   |
| **Filling**    | `start_stage_id ≤ id < entry_stage_id`         | The reservoir exists and impounds water toward the dead volume, but is not yet a generating plant.                           |
| **Operating**  | `id ≥ entry_stage_id`                          | A normal plant — identical to a non-commissioned hydro, plus a soft floor while it recovers from a possibly-deficient start. |

`start_stage_id == 0` ⇒ no PreFilling (Filling begins at stage 0; this is how a
study that _starts mid-filling_ is expressed). `start_stage_id > 0` ⇒ PreFilling
exists and the seed in `InitialConditions::filling_storage` is `0` (empty pit).

The filling **target** is the dead volume `min_storage_hm3` (no separate target
field): the reservoir fills from the seed up to `min_storage_hm3`.

## 2. Per-phase behavior

| Aspect                   | PreFilling                                                 | Filling                                                                                          | Operating                                                                                     |
| ------------------------ | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| Storage state column     | present, **frozen** `[seed, seed]`, **off the water path** | present, accumulating                                                                            | present                                                                                       |
| Storage upper bound      | `max_storage_hm3`                                          | `max_storage_hm3`                                                                                | `max_storage_hm3`                                                                             |
| Storage lower floor      | n/a (frozen)                                               | **OFF** — `[0, max_storage_hm3]`                                                                 | hard `min_storage_hm3` (normal hydro) / **soft `σ^{v-}`** (a hydro that went through filling) |
| Generation / turbine     | `[0,0]`                                                    | **`[0,0]`** (turbines not installed)                                                             | normal                                                                                        |
| FPHA production row      | excluded                                                   | **excluded** (zero productivity; operating-range fit invalid below `min_storage`)                | normal (if FPHA model)                                                                        |
| Diversion                | `0`                                                        | **`0`** (always, for filling hydros)                                                             | normal                                                                                        |
| Inflow                   | routed to downstream (§3.2)                                | full natural inflow arrives; **`filling_inflow_m3s` portion impounded**, remainder spills (§3.1) | PAR dispatch                                                                                  |
| AR-lag coupling          | (routed to downstream)                                     | **normal**                                                                                       | normal                                                                                        |
| Evaporation              | `0` (no surface; not transferred)                          | **normal**                                                                                       | normal                                                                                        |
| Withdrawal               | **transferred to downstream** (§3.2)                       | served from own reservoir                                                                        | served from own reservoir                                                                     |
| Cascade coupling         | **short-circuit** (§3.2)                                   | normal (impounds)                                                                                | normal                                                                                        |
| `σ_fill` terminal target | —                                                          | at id `entry_stage_id − 1` only (§3.3)                                                           | —                                                                                             |
| `σ^{v-}` recovery floor  | —                                                          | —                                                                                                | filling hydros only (§3.4)                                                                    |

## 3. LP formulation

The live water-balance row for hydro `h` (built in `fill_state_and_water_entries`
in `lp/builder/entries.rs` and `fill_water_balance_rows` in `lp/builder/rows.rs`)
is the storage-accumulation identity `v_h − v_h_in + ζ·(outflows − inflows + losses) = ζ·(base − withdrawal)`,
with `ζ = layout.zeta` the per-stage m³/s→hm³ factor and `τ_k` the per-block
equivalent. The subgradient contract (`.claude/rules/sddp.md`,
`extract_duals_from_view`): the cut coefficient on the storage state is the
incoming-storage column's reduced cost **divided** by `col_scale`, never
synthesized.

The phase predicate is a new `filling_phase(start, entry, stage_id)` beside
`commissioning_active` in `lp/builder/mod.rs`.

### 3.1 Filling water balance — retained-portion inflow (Correction 1)

During Filling the reservoir sits in the cascade and receives its **full natural
inflow** (incremental via PAR + upstream releases). PAR and the AR-lag coupling
are **unchanged from Operating** — the row keeps its base RHS, its noise patch,
and its AR-lag terms. The only structural changes vs Operating are: generation
and turbine pinned `[0,0]`, the FPHA row excluded, the storage floor off, and an
**impound retention** mechanism.

`filling_inflow_m3s` (`= F_h`) is the portion of the natural inflow **impounded**
to raise storage; the remainder passes downstream as spillage. Impound is a
**cap, not an equality**: the reservoir impounds _at most_ `F_h` per stage and
spills the excess; when natural inflow is short it impounds what it can and the
terminal `σ_fill` (§3.3) catches the cumulative shortfall. This is realized as
one new **retention row** per filling hydro per stage:

```
total release (spillage + diversion≡0) ≥ natural inflow into the reservoir − F_h
```

(coefficients `ζ`/`τ_k` as in the standard balance). It forces the reservoir to
pass everything beyond the impound cap downstream, so storage can rise by at most
`ζ·F_h` per stage. Rising _less_ (insufficient inflow) is permitted and is
exactly what `σ_fill` penalizes terminally. The standard balance equality row is
kept unchanged; the retention inequality is _additional_. Withdrawal during
Filling is served from the reservoir as usual; min-outflow rides the existing
`outflow_violation_below` operational slack when the reservoir is too low.

> **Correction to a naive reading**: the filling inflow is **not** an external or
> replacement inflow (`RHS = ζ·filling_inflow`); it is a retained portion of the
> natural inflow with PAR untouched. The `FillingConfig.filling_inflow_m3s`
> doc-comment, which currently describes "constant inflow applied … from a fixed
> inflow source", is wrong and must be rewritten (§5.3).

### 3.2 PreFilling cascade short-circuit (Correction 2)

Before `start_stage_id` the dam does not exist; the river flows past its site to
`d = downstream_id`. The absent hydro `h` is fully bypassed and **all** its water
interactions transfer to `d`:

- `h`'s own water-balance row collapses to the frozen-storage identity
  `v_h − v_h_in = 0`, `v_h ∈ [seed, seed]`, `v_h_in` pinned to the seed — no
  inflow/outflow/AR/evaporation/withdrawal terms. **The storage column keeps its
  dense index** (no omission, no relocation — the coordinate-drift trap, §4).
- `d`'s water-balance row gains: `h`'s incremental inflow (the cleanest
  realization relocates the `z_h`→water coupling from `h`'s row to `d`'s row,
  leaving the PAR machinery untouched), `h`'s upstream releases (re-route the
  cascade edge `upstream(h)→h→d` to `upstream(h)→d`), and `h`'s withdrawal
  (served by `d`, subject to `d`'s availability via the existing withdrawal
  slack).
- `h`'s **evaporation is zero** (no reservoir surface) and is **not** transferred.
- **No-downstream / sink case** (`downstream_id == None`): `h`'s water simply
  exits the system, exactly as a terminal hydro's outflow does today.

At `start_stage_id` the balance falls back to Filling (§3.1): the reservoir enters
the cascade and impounds, with generation/turbine still pinned `[0,0]`.

### 3.3 `σ_fill` terminal target (filling-hydro-scoped)

A new stage-level slack column + `≥` row, emitted **only** at the stage with id
`entry_stage_id − 1`, only for filling hydros (sparse, but dense at the terminal
stage via `geometry_per_stage`):

```
v_h + σ_fill ≥ min_storage_hm3,   σ_fill ≥ 0,   cost = filling_target_violation_cost
```

`filling_target_violation_cost` is the highest penalty in the system (a
semantic-validator check warns — non-blocking — when it does not exceed
`storage_violation_below_cost`; the ordering is advised, not hard-enforced). The
per-stage impound cap (§3.1) rate-limits the fill, so a single terminal target is
sufficient — no per-stage trajectory target is needed.

### 3.4 `σ^{v-}` soft operating floor (filling hydros only)

For a hydro that **went through filling**, the Operating-phase lower bound is
soft so it can recover from a deficient start without infeasibility:

```
v_h + σ^{v-} ≥ min_storage_hm3,   σ^{v-} ≥ 0,   cost = storage_violation_below_cost
```

realized by relaxing the storage column's hard lower bound (currently
`col_lower = min_storage_hm3` in `fill_storage_columns`) to `0` and adding the
soft-floor row + slack. **Normal hydros keep the hard floor** — this is a
deliberate scoping decision (the spec makes min-storage soft for _all_ hydros;
we scope it to filling hydros because they are rare and are the only ones that
can legitimately start below the dead volume). The high penalty drives `σ^{v-} →
0` the moment storage recovers above `min_storage`, so "soft for the whole
operating horizon" only ever bites during a genuine deficit. The discriminator is
`hydro.filling.is_some()`, evaluated per-hydro — carry it as a Contract comment
at the construction site so a future reader does not "simplify" it into a global
soft floor (which would let the optimizer cheaply violate dead volume
system-wide).

### 3.5 FPHA per-stage exclusion

In `identify_fpha_hydros` (`lp/builder/layout.rs`), exclude a hydro from
`fpha_hydro_indices` at stages where it is PreFilling or Filling. The per-hydro
hyperplane fit is unchanged (still fit once over the operating range); only the
row emission is gated. This is the natural site — it already takes the stage
index and filters per stage — so FPHA exclusion needs no new geometry. Rationale:
a non-operating hydro has zero productivity, and the operating-range fit is
invalid below `min_storage` where a filling reservoir sits.

### 3.6 Generation / turbine / diversion / evaporation gating

Mirror the thermal commissioning pattern (`fill_thermal_columns`: dormant ⇒
**both** bounds `[0,0]`):

- Generation and turbine: PreFilling and Filling ⇒ `[0,0]`.
- Diversion: `[0,0]` for filling hydros in **all** phases.
- Evaporation: excluded in PreFilling (no surface); normal in Filling and
  Operating.

### 3.7 Seed wiring and the filling→operating handoff

`build_initial_state` (`setup/mod.rs`) currently reads only `ic.storage`; add a
loop over `ic.filling_storage` writing the seed for filling hydros (a filling
hydro appears in `filling_storage`, never `storage` — an invariant already
documented and validated). The transition at `entry_stage_id` needs **no special
code**: the end-of-filling storage becomes the operating initial state through
the existing incoming-state pin chain (the forward pass writes the previous
stage's outgoing storage into the next stage's pinned incoming column). Verify by
case assertion (§7).

## 4. Cut validity across the phase boundaries

The Benders state is the storage vector (plus AR-lag state). For every hydro and
every phase the storage column keeps its dense index and the incoming-state
column resolves via the same `StateLayout::state_to_lp_incoming_column` map — the
load-bearing invariant: the cut coefficient vector is indexed by state
coordinate, and that indexing is phase-invariant.

- **PreFilling, downstream `d`.** `h`'s water is routed onto `d`'s row in real
  matrix coefficients (not an approximation), so by LP duality the reduced cost
  of `d`'s pinned incoming-storage column is the _true_ sensitivity of the routed
  problem. The cut coefficient `rc / col_scale` is a valid subgradient.
- **PreFilling, absent `h`.** With `v_h_in` relocated out of `h`'s row and `h`'s
  row a frozen identity, perturbing `v̂_h` changes nothing, so `∂Q/∂v̂_h = 0` — a
  valid _flat_ cut (a non-existent reservoir's storage has zero marginal value).
  Trap: if the relocation is botched and `h`'s row stays coupled to upstream,
  `β_h` goes stale-nonzero.
- **Filling.** The balance is the standard equality plus the retention
  inequality, all real coefficients with PAR/AR-lag present — a standard Benders
  minorant, same machinery as any Operating stage.
- **Soft duals.** `σ_fill`, `σ^{v-}`, and the retention-row duals couple to the
  storage column through the constraint matrix, so LP duality delivers their
  combined effect as the **single** reduced cost of the incoming-storage column.
  Read that one reduced cost (and divide by `col_scale`); never separately extract
  the soft-row duals and add them by hand. The existing `extract_duals_from_view`
  already does the right thing.
- **Cross-boundary.** The storage coordinate's meaning is continuous (reservoir
  volume of `h`; zero/seed when absent, rising during filling, operating
  thereafter) and its index is constant, so a cut harvested at any stage bounds
  that stage's value function regardless of the neighbor's phase. The append-only
  cut pool keeps every cut at a stable slot; slot-identity warm-start
  reconstruction is unaffected by phase changes (no cut row is relocated).

**Implementer trap checklist:** (1) never synthesize a coefficient — always
`rc / col_scale`; (2) never omit/relocate the storage column at any phase;
(3) never hand-combine soft-row duals; (4) in PreFilling, relocate `v_h_in` _out
of_ `h`'s row, else `β_h` goes stale-nonzero; (5) `evaluate_lower_bound` must
apply the _same_ phase gating as forward/backward (the NCS-class precedent).

## 5. Input model

### 5.1 Fields — no new field is needed

> **[Still current under the successor.]** The successor adds **no** new field — it
> renames `filling_inflow_m3s` → `filling_min_rate_m3s` and flips its semantics (§5.3),
> and anchors the fill target on the existing `min_storage_hm3`. So "no new field is
> needed" holds for both v1 and the successor.

The model is fully specified by existing fields:

- **`FillingConfig`** (on `Hydro.filling`): `start_stage_id`, `filling_inflow_m3s`.
- **`Hydro`**: `entry_stage_id`, `exit_stage_id` (rejected for filling hydros),
  `min_storage_hm3` (the filling target), `max_storage_hm3`, `min_outflow_m3s`,
  `downstream_id` (the short-circuit target).
- **`InitialConditions.filling_storage`**: the stage-0 seed, validated to
  `[0, min_storage_hm3]` and mutually exclusive with `.storage`.
- **`HydroPenalties`**: `filling_target_violation_cost` (`σ_fill`),
  `storage_violation_below_cost` (`σ^{v-}`), `water_withdrawal_violation_*`.
- **Resolved per-stage hydro bounds** already carry `filling_inflow_m3s` and
  `water_withdrawal_m3s` per stage, so both can vary per stage without a new
  field.

A "filling initial storage" field on `FillingConfig` was considered and
**rejected**: the stage-0 storage is a per-run _initial condition_ (it varies per
run of the same system), so it belongs in `InitialConditions.filling_storage`
alongside `.storage` for operating hydros — putting it on the per-entity
`FillingConfig` would mix a per-run state into a per-system config. It is also
redundant: pre-filling freezes storage, so the impounding-start level _is_ the
stage-0 seed carried forward.

### 5.2 `filling_storage` ↔ `start_stage_id`

- **Study starts mid-filling**: `start_stage_id = 0`, `filling_storage` = the
  partially-filled stage-0 level (in `[0, min_storage_hm3]`). The reservoir is in
  the Filling phase from stage 0. (Note: the operating-hydro `.storage` map,
  validated `[min_storage, max_storage]`, _cannot_ hold a below-dead-volume
  level — `.filling_storage` is the correct home.)
- **Filling begins within the study** (`start_stage_id > 0`): `filling_storage =
0` (empty pit), held frozen through PreFilling and released at `start_stage_id`.

### 5.3 Semantic correction: `filling_inflow_m3s`

The current `FillingConfig.filling_inflow_m3s` doc describes an external/applied
inflow. Rewrite it to: _the retained portion of the reservoir's natural inflow
per stage, removed (impounded) from the cascade to raise storage; the remainder
passes downstream; PAR and AR-lag coupling remain normal during filling — not an
external or replacement inflow._

### 5.4 Validation rules (no new fields, just guards)

> **[Superseded in part]** The successor reinterprets `filling_min_rate_m3s` as a
> per-stage minimum accumulation rate and adds a one-sided **`≥` sufficiency check**
> (the minimum-rate trajectory must reach `min_storage`); it **keeps**
> `storage_violation_below_cost` (only the penalty ordering flips). The guards below
> otherwise describe the **shipped v1** model.

- `entry_stage_id.is_some()` ⟺ `filling.is_some()` (entry requires filling and
  vice-versa).
- `start_stage_id < entry_stage_id`.
- `entry_stage_id < horizon` (the reservoir must operate at least one stage).
- `filling_storage ∈ [0, min_storage_hm3)`.
- reject `exit_stage_id` on a filling hydro (hydro is entry-only — exit is
  physically ill-posed for a state-carrying reservoir).
- `filling_inflow_m3s ≥ 0` (a zero cap means "pass everything, rely on natural
  accumulation"). The original `> 0` check was relaxed to `≥ 0` during v1 and ships
  that way — `validate_filling_configs` rejects only `< 0` (and NaN).

## 6. Implementation touch-points

All gating is template-driven (the phase is deterministic, stage-only), so
forward/backward/lower-bound/simulation inherit it from the shared per-stage
template — but the lower-bound build must be verified to use the same template
(the NCS-class precedent).

| Concern                                       | Site                                                                                                      |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Phase predicate                               | new `filling_phase` in `lp/builder/mod.rs`                                                                |
| Cascade short-circuit entries                 | `lp/builder/entries.rs` `fill_state_and_water_entries`                                                    |
| Filling retention row                         | new fills in `lp/builder/entries.rs` + `rows.rs`                                                          |
| `σ_fill` / `σ^{v-}` columns + rows + geometry | `columns.rs`, `rows.rs`, `entries.rs`, `layout.rs`, `template.rs` (`StageGeometry`, `geometry_per_stage`) |
| FPHA exclusion                                | `layout.rs` `identify_fpha_hydros`                                                                        |
| Evaporation exclusion (PreFilling)            | `layout.rs` `identify_evap_hydros` (must take the stage index)                                            |
| Generation/turbine/diversion gating           | `columns.rs` (`fill_turbine_columns`, FPHA-generation, `fill_diversion_columns`)                          |
| Storage floor relax + seed                    | `columns.rs` `fill_storage_columns`; `setup/mod.rs` `build_initial_state`                                 |
| Lower-bound parity                            | `training/lower_bound.rs` `evaluate_lower_bound` (must apply the same gating)                             |
| Validation                                    | `cobre-core` `validate_filling_configs`, `cobre-io` semantic validator                                    |
| Doc fixes                                     | `cobre-core/src/entities/hydro.rs` (§5.3, §5.2)                                                           |
| Output parity                                 | if filling adds output columns, wire both the CLI and Python paths (Python-parity hard rule)              |

## 7. Deterministic cases: `d38-dead-volume-filling`, `d39-prefilling-upstream-of-filling`

> **[Assertions superseded]** Under the successor, the terminal `σ_fill` assertion
> below becomes **per-stage minimum target floors** (`σ_fill[t]`); the operating start
> is **continuous** (no reset/pin); and the `σ^{v-}` operating floor is **kept** (soft
> `min_storage` floor, family renamed `filled_min_storage_floor`). The topology is
> unchanged, and a d40 filling-cascade case is added. The companion d39 case keeps its
> topology but its routed-inflow assertion moves from the impound-cap row to the
> balance row (the cap row is removed — §3.4).

A cascade where a **mid-cascade** hydro fills, so the short-circuit routes to a
_real_ downstream (not a sink), with **varying block counts across the phase
boundaries** (exercises the per-stage geometry and the per-stage `τ`). Suggested
topology `H1 → H2 → H3` with `H2` the filling hydro (`start_stage_id = 2`,
`entry_stage_id = 4`, horizon ≥ 6), plus a non-filling control hydro to confirm
bit-identical-to-today behavior.

Assertions:

1. **Short-circuit routing**: during PreFilling, `H3`'s water-balance row carries
   `H2`'s incremental inflow + `H1`'s releases + `H2`'s withdrawal; `H2`'s row is
   the frozen identity.
2. **Zero generation/turbine** for all stages before `entry`.
3. **Accumulation**: `H2` storage frozen at the seed in PreFilling; rises (capped
   at `ζ·F_h` per stage) during Filling.
4. **`σ_fill` / `σ^{v-}` under short inflow**: a low-inflow scenario leaves
   `σ_fill > 0` at id `entry−1`; a recovery scenario drives `σ^{v-} → 0` in
   Operating.
5. **Entry handoff**: end-of-filling storage at id `entry−1` equals the incoming
   storage at id `entry` (the pin chain).
6. **Cut / warm-start validity across both boundaries**: monotone lower bound;
   no basis-rejection spike at the boundary stages.
7. **Both-backend baselines** (HiGHS + CLP), with the no-filling control hydro
   confirming the formulation is parity-neutral for the normal hydro world.

**Companion case — `d39-prefilling-upstream-of-filling`.** A regression where a
PreFilling hydro sits **upstream** of a Filling hydro at the same stage, exercising
the v1 fix that counts the absent upstream dam's routed inflow in the Filling
downstream's impound-cap row (the retention-row port). The reformulation drops that
port — its per-stage target row carries no inflow term (reformulation §3.4) — so
d39's assertion changes to "routed inflow lands on the **balance** row only".

## 8. Confirmed design decisions

The genuine forks raised during design, and how they were resolved:

| Fork                                       | Decision                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Filling fidelity                           | **Full spec gradual filling** (not an instantaneous-fill MVP).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| FPHA neutralization                        | **Per-stage exclusion** of a non-operating hydro (zero-productivity FPH; fit unchanged per-hydro).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Evaporation during filling                 | **Modeled normally** (the reservoir has a surface and loses water); zero only in PreFilling (no surface).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Hydro exit                                 | **Entry-only** — `exit_stage_id` rejected; exit is ill-posed for a state-carrying reservoir.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Min-storage softening scope                | **Soft floor (`σ^{v-}`) only for hydros that went through filling**, across their whole operating horizon (self-limiting via the high penalty). Normal hydros keep the hard floor — a deliberate deviation from the spec's all-hydros-soft, scoped to the rare filling hydro to avoid a system-wide formulation change. **[Retained by the successor (§3.3) — the soft floor stays; family renamed `filling_floor` → `filled_min_storage_floor`, and the penalty ordering flips so the operating floor dominates the fill schedule and should be `≥ deficit_cost` (mimicking the hard floor). The reset-based draft that proposed removing it is abandoned.]** |
| Filling inflow semantics                   | **Retained portion** of natural inflow (impounded, removed from the cascade), cap not equality; PAR runs normally. **[Superseded — successor §3.1: the per-stage _cap_ becomes a per-stage _minimum accumulation rate_ (uncapped, `≥` floors); `filling_inflow_m3s` → `filling_min_rate_m3s`. The retained-portion / PAR-normal semantics persist.]**                                                                                                                                                                                                                                                                                                          |
| PreFilling cascade                         | **Short-circuit** — the absent dam's incremental inflow, upstream releases, and withdrawal transfer to its downstream; evaporation/diversion nil.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Entry without filling                      | **Rejected** — `entry_stage_id` requires a `FillingConfig`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `σ_fill` placement                         | **Terminal-only** at id `entry−1` (the per-stage impound cap rate-limits the fill, so no intermediate targets are needed). **[Superseded — successor §3.1: the cap is removed, so a per-stage schedule is needed; the terminal `σ_fill` becomes **per-stage minimum target floors** `V_target[t]`, backward-anchored to `min_storage`.]**                                                                                                                                                                                                                                                                                                                      |
| `filling_storage` for `start_stage_id > 0` | **Always `0`** (empty pit).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
