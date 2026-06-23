# Hydro filling: volume-target reformulation

> **Status**: Proposed reformulation — not yet implemented. Companion to
> `hydro-dead-volume-filling.md` (the shipped v1 model). It **supersedes** that
> document's §3.1 (impound cap), §3.3 (terminal `σ_fill`), §3.4 (`σ^{v-}` soft
> operating floor), §3.7 (filling→operating handoff), and §5.3 (filling-inflow
> semantics). It **keeps unchanged**: §1 lifecycle, §3.2 PreFilling cascade
> short-circuit, §3.5 FPHA exclusion, §3.6 generation/turbine/diversion/evaporation
> gating, and the §4 cut-validity invariants — except where §3.2 below adds a new
> requirement (the state reset).
>
> **Validation**: a formulation review against the live LP builder returned
> SOUND-with-caveats. The caveats are not optional and are folded in below as hard
> requirements — chiefly the state-reset realization (§3.2, including the
> storage-only / AR-lag-continuous boundary) and the cut-validity trap (§5).
>
> **Contracts it must not disturb**: `.claude/rules/sddp.md` (Benders cut
> sign/scale, column-bound state pinning, FPHA average storage, append-only cut
> pool with slot-identity warm start) and the shared `commissioning_active` /
> `filling_phase` predicates in `lp/builder/mod.rs`.

## TL;DR

The shipped model makes a plant that **went through filling** carry a soft
storage floor (`σ^{v-}`) at **every** operating stage, across its whole operating
horizon — so it can dip below its dead volume by paying
`storage_violation_below_cost` _at any point in operation_, years after filling
and unrelated to it. That soft floor is more permissive than intended (v1's own
§3.4 warns a future reader not to globalize it), and it is the only soft-floor
cushion in the system.

This reformulation **confines** the legitimate "below dead volume" state to the
filling phase, where it belongs. The enabling idea: **guarantee that operation
starts at a fixed fill target `v_fill ≥ min_storage`** (pin the operating-start
storage to it), so the under-filled-start case that `σ^{v-}` existed to absorb
cannot reach the operating phase. With that guarantee the operating phase uses a
**hard** floor at `min_storage_hm3` — structurally identical to a never-filled
plant. The filling phase is reframed as a **per-stage retention target with a
below-slack** (modeled like water withdrawal), and the terminal `σ_fill`
collapses into those per-stage slacks.

This is a **modeling trade-off, not a pure bug-fix.** v1 carries a genuine
filling shortfall forward into operation and lets the plant recover via the soft
floor — arguably the more physically faithful choice. This reformulation instead
asserts operation begins at `v_fill` regardless and pays any shortfall as a
filling-phase penalty; in a dry stochastic scenario it thereby _invents_ a small
amount of boundary water (§6). The win is structural simplicity (the operating
phase is now the normal-plant regime) and a tightly-scoped below-dead-volume
window, not a strictly more accurate model.

Net structural change: **retire two row/column families** (`σ_fill` terminal
target and `σ^{v-}` operating floor) and the impound-cap row; **add one** per-stage
target family and a boundary state-reset; **add one** input field
(`initial_operating_volume_hm3`, the fill target `v_fill`, defaulting to the dead
volume); **remove** `storage_violation_below_cost` (now unused).

---

## 1. What changes, per mechanism

| Mechanism                      | Shipped v1                                                                                  | Reformulation                                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Filling impound                | per-stage **cap** `rise ≤ ζ·F_h` (inequality, no slack)                                     | per-stage **target** `(v_out − v_in) + σ_below = ζ·target` (equality with below-slack) (§3.1)        |
| Filling shortfall penalty      | a single **terminal** `σ_fill` at id `entry−1`                                              | the **per-stage** `σ_below` slacks (their sum is the terminal deficit) (§3.1)                        |
| Filling→operating handoff      | end-of-filling storage flows in via the pin chain (continuous)                              | operating-start storage **pinned to the fill target `v_fill`** — a storage-only reset (§3.2)         |
| Fill target / start level      | implicitly the dead volume (v1 fills toward `min_storage`)                                  | a **configurable** `v_fill = initial_operating_volume_hm3 ≥ min_storage`, default = dead volume (§4) |
| Operating storage floor        | **soft** `σ^{v-}` at every stage (filling hydros only)                                      | **hard** `min_storage_hm3`, identical to a normal plant — `σ^{v-}` family removed (§3.3)             |
| PreFilling short-circuit       | routes the absent dam's water onto the downstream's balance + the impound-cap retention row | **unchanged**: only the balance-row routing applies; the v1 retention-row port is dropped (§3.4)     |
| `filling_inflow_m3s`           | a single per-stage cap value                                                                | a **per-filling-stage schedule** (vector), uniform-flow default, strongly validated (§4)             |
| `storage_violation_below_cost` | the `σ^{v-}` cost                                                                           | **removed** (the field and its ordering validation) (§7)                                             |

Unchanged: the three-phase lifecycle keyed on `stage.id`, the PreFilling
frozen-identity row + cascade short-circuit, FPHA exclusion, the `[0,0]`
generation/turbine/diversion gating, and the dense-storage-column / declaration-
order invariants.

## 2. Locked decisions

1. **Pin to the fill target.** Operation always begins at the fill target
   `v_fill`; a filling shortfall is paid as a per-stage penalty and **not** carried
   forward as a reduced operating-start storage. This is what licenses the hard
   operating floor at `min_storage`.
2. **The fill target is configurable, `≥ min_storage`.** `v_fill =
initial_operating_volume_hm3` defaults to `min_storage_hm3` (fill exactly to the
   dead volume, the v1 convention) but may be set higher to commission with a
   buffer above the dead volume. The buffer `v_fill − min_storage` is also the
   robustness mitigation for first-operating-stage feasibility (§3.3, the
   resolution of v1 §7-Q1).
3. **Target replaces the cap.** Each filling stage carries a retention target with
   a below-slack; over-impounding is bounded by the target itself.
4. **Per-stage user-supplied schedule, uniform default, strong validation.**
   `filling_inflow_m3s` becomes a per-filling-stage schedule (a different value per
   filling stage is allowed), defaulting to the internal uniform resolution, with
   validation that the schedule is consistent with reaching `v_fill` (§4).
5. **`storage_violation_below_cost` is removed.** With `σ^{v-}` gone the field and
   its `< filling_target_violation_cost` ordering validation are deleted (§7 — a
   breaking config change).

## 3. Formulation

### 3.1 Filling: per-stage retention target with a below-slack

Each filling stage `t` carries one equality, on the storage **rise**:

```
(v_out − v_in) + σ_below[t] = ζ_t · target[t],   σ_below[t] ≥ 0,
cost(σ_below) = filling_target_violation_cost
```

with `ζ_t = layout.zeta` the per-stage m³/s→hm³ factor. Because `σ_below ≥ 0`,
this both **caps** the rise at `ζ_t·target[t]` (no over-impound) and **penalizes**
any shortfall — it unifies the v1 impound cap and the terminal `σ_fill` into one
per-stage mechanism. Behavior:

- **Wet stage** (impoundable inflow ≥ `ζ_t·target[t]`): rise = `ζ_t·target[t]`
  exactly, `σ_below = 0`, the excess is released (spillage) — the cap behavior.
- **Dry stage** (impoundable inflow < `ζ_t·target[t]`): rise = available inflow,
  `σ_below = ζ_t·target[t] − rise > 0` — the shortfall is penalized.

> **Contract — filling-hydro spillage must stay unbounded above.** The wet-stage
> cap behavior relies on the reservoir being able to spill the excess so the rise
> can always be held down to `ζ_t·target[t]`. The equality `(v_out − v_in) +
σ_below = ζ_t·target` with `σ_below ≥ 0` is well-posed (σ_below never forced
> negative) **only because** the spillage column is unbounded above (`col_upper =
+∞` in `fill_storage_columns`). If a future change bounds filling-hydro spillage,
> an extremely wet stage could force the rise above the target with no σ_below to
> absorb it ⇒ infeasible. Preserve the unbounded-spillage invariant for filling
> hydros.

The per-stage slacks **replace** the v1 terminal `σ_fill`. **When the schedule is
validated to sum to the fill (§4)** — `Σ_t ζ_t·target[t] = v_fill − seed` — the
per-stage slacks telescope to the terminal deficit: `Σ_t σ_below[t] = v_fill −
v_(entry−1)`, exactly what `σ_fill` measured. So penalizing the schedule loses
nothing versus penalizing the terminal volume, at a smoother per-stage signal,
**and this conservation identity depends on that §4 sum check** — a schedule that
does not sum to `v_fill − seed` breaks it (§4 rejects such schedules). Keeping
both the per-stage and a terminal penalty would **double-penalize** the same
shortfall; use the **same cost field** (`filling_target_violation_cost`) so the
aggregate penalty is conserved.

#### The withdrawal analogy — and the one thing it pins down

The mechanism is the structural mirror of water withdrawal, and the analogy is
load-bearing because it fixes **what the slack is anchored to**:

| Water withdrawal                     | Filling retention                                          |
| ------------------------------------ | ---------------------------------------------------------- |
| demand `W` (schedule)                | target retention `target[t]` (schedule)                    |
| served `w` (decision, moves balance) | **achieved retention = `v_out − v_in`** (moves balance)    |
| slack = `W − w` (unmet demand)       | slack = `ζ_t·target[t] − (v_out − v_in)` (unmet retention) |

The slack must sit on the **achieved retention (the storage rise)**, never on
"did the scheduled natural inflow physically arrive." This is decisive **because
we pin the operating-start volume regardless** (§3.2): if the slack were tied to
raw inflow availability — a fixed number per scenario, independent of release —
the optimizer could dump the inflow downstream for energy value, pay the same
penalty as if it had filled, **and still start operation at `v_fill`**, banking
the water's value twice. Anchoring the slack to the rise closes that arbitrage:
releasing instead of retaining shrinks `v_out − v_in`, which grows `σ_below`,
which costs. Done the withdrawal way (slack = schedule − actually-achieved), the
penalty genuinely incentivizes filling.

`target[t]` is a **retention target on the natural inflow already in the
balance** — not an extra inflow source added to it, and **not a term in the
retention row** (the row touches only `v_out`, `v_in`, and `σ_below`; the inflow
enters solely through the water-balance row). PAR and AR-lag coupling run normally
during filling (carried over from v1 §3.1). That the retention row carries no
inflow term is what makes §3.4's short-circuit routing a no-op.

### 3.2 The filling→operating boundary: pin to the fill target (state reset)

At the first operating stage (`id == entry_stage_id`), the incoming-storage state
is set to the fill target `v_fill`, **decoupled** from the last filling stage's
outgoing storage. So operation always starts at `v_fill` regardless of how filling
actually went. No Benders cut crosses this boundary **in the storage coordinate** —
the filling sub-tree and the operating sub-tree are independent in that dimension,
exactly as the stage-0 seed roots the storage coordinate.

**The reset is storage-coordinate-only.** The Benders state is storage **plus
AR-lag inflow memory**. The reset touches only the storage coordinate; the AR-lag
state crosses the entry boundary **continuously** (the hydrology does not reset
when the reservoir is declared full), and its subgradient `β` is harvested
normally. The RHS-fold below touches only the storage column, so AR-lag coupling
is preserved by construction — but do not "extend" the reset to zero the AR-lag
coupling, which would discard the operating phase's dependence on filling-era
inflows.

**Hard requirement — realize the reset by RHS-fold, NOT as a column-bound
pin.** `fill_col_state_patches` unconditionally pins every hydro's incoming-storage
column to the forward-propagated trial point, and `extract_duals_from_view`
unconditionally harvests that same column's reduced cost as the cut subgradient
`β_h`, for **every** state coordinate. If the reset is realized by pinning the
entry-stage incoming column to the constant `v_fill` via bounds, that column's
reduced cost (the dual of `lb==ub`) is generally **nonzero**, and the cut machinery
attaches it to the **last filling stage's** value function — a stale-nonzero slope
on a coordinate whose true marginal value is zero. That is a wrong cut that
compiles (the §5 trap-1 failure mode).

The correct realization: **at the entry stage only, drop the `−1` coefficient on
the incoming-storage column from the reservoir-balance row and fold the constant
`v_fill` into that row's RHS**, so the incoming-storage column appears in no row
and carries no objective cost ⇒ its reduced cost is structurally `0` ⇒ `β_h = 0`
(a correct flat cut on the filling predecessor). The forward-propagated
`current_state[h]` still flows in via the (now-harmless) pin but is ignored by the
entry-stage LP.

> **Not the same construction as PreFilling.** PreFilling makes the **whole**
> balance row a frozen identity (no inflow/release/FPHA/evaporation terms at all),
> which is what makes its storage column dead. The entry stage is a **full
> operating stage**: keep its entire operating balance (releases, FPHA, evaporation)
> and drop **only** the incoming-storage coefficient. Do not collapse the operating
> balance to a frozen identity — that would zero out real operating physics. Same
> `β_h = 0` outcome, different construction.

The storage column keeps its dense index at every stage (no omission, no relocation
of the _column_ — only its row coupling changes at this one stage), so the
dense-storage and declaration-order invariants hold.

### 3.3 Operating: hard floor, no `σ^{v-}`

With operation guaranteed to start at `v_fill ≥ min_storage` (§3.2), the
operating-phase storage floor is **hard** at `min_storage_hm3` — bit-identical to a
never-filled plant. The entire `σ^{v-}` operating-floor row/column family is
**removed**, and `storage_violation_below_cost` is removed with it (§7).

Feasibility: at the first operating stage `v_in = v_fill ≥ min_storage`, so the
hard floor `v_out ≥ min_storage` allows a net draw of up to `v_fill − min_storage`
before it binds. The recourse set is **exactly the one a normal plant starting at
`min_storage` has** — this is not a regression, it is the inherited normal-plant
regime:

- **evaporation** rides its own soft slack (the evaporation column is symmetric
  `[−q_max, +q_max]` and the linearized loss carries `f_evap±` violation slacks in
  `fill_evaporation_columns`), so the LP can under-realize evaporation at
  `evaporation_violation_cost` rather than be forced below the floor — evaporation
  is **not** an unconditional hard loss;
- **min-outflow** is a soft `≥` riding the existing `outflow_violation_below` slack;
- the **inflow slack** `σ_inf` can inject water into the balance — **but only when
  the inflow method provides one** (`has_penalty` ⇒ `Penalty` /
  `TruncationWithPenalty`; `None` / `Truncation` carry no inflow slack).

So genuine first-operating-stage infeasibility is a **narrow corner**: `v_fill =
min_storage` (no buffer) **and** a slack-free inflow method **and** the realized net
of (deterministic balance + evaporation-at-slack + min-outflow-at-slack) still
drives `v_out < min_storage`. The buffer `v_fill − min_storage` is the mitigation
(§2 decision 2): choose `v_fill > min_storage` when the first operating stages may
be dry, especially under a slack-free inflow method. A mid-operation drought beyond
the buffer is handled exactly as for any normal plant — reduce outflow toward
`min_outflow` and, ultimately, load deficit. The filled plant is genuinely in the
normal-plant regime.

> **Contract**: dropping `σ^{v-}` is sound **only** because the reset (§3.2)
> guarantees operation starts at `v_fill ≥ min_storage`. Tie the two together with a
> comment at the entry-stage reset site — weakening the reset back to inheriting the
> (possibly short) filling storage would re-introduce first-operating-stage
> infeasibility with no slack to absorb it.

### 3.4 PreFilling short-circuit: balance-row routing only (no target-row port)

The PreFilling cascade short-circuit (`fill_prefilling_shortcircuit`) routes an
absent upstream dam's realized inflow (`z`), transitive upstream releases, and
diversion onto its first non-PreFilling downstream's **water-balance row**. That
routing is **unchanged** and remains phase-agnostic in the downstream `d`.

v1 _additionally_ pushed the routed terms onto a Filling downstream's
impound-retention row, because that row carried explicit inflow terms (`−ζ·z_h`
and the upstream-release terms — see the shipped `fill_filling_retention_rows`,
whose RHS is `−ζ·F_h` with the inflow on the LHS). The fix that wired this is the
v1 commit that "counts PreFilling-upstream inflow in the impound cap."

The reformulation's per-stage **target row** (§3.1) carries **no inflow or release
term** — it touches only `v_out`, `v_in`, and `σ_below`. Routed inflow enters the
water-balance row, the balance determines `v_out`, and the target row caps the
resulting rise via `σ_below`. The routed inflow's entire effect on the Filling
downstream is therefore already mediated through the balance, so **the v1
retention-row port is dropped — there is no target-row term to update.** Re-adding
it would be meaningless (the row has no inflow coefficient to receive the terms)
and is explicitly _not_ required. This removes a v1 "hard requirement" and a
migration step rather than adding one.

## 4. Input model: fill target, per-stage schedule, validation

### 4.1 The fill target `v_fill`

A new field `initial_operating_volume_hm3` (`= v_fill`) on `FillingConfig` —
**per-entity** commissioning config (per-system, not per-run), so it lives beside
`start_stage_id` / `filling_inflow_m3s`, **not** in `InitialConditions` (which
holds the per-run seed). Validation: `min_storage_hm3 ≤ initial_operating_volume_hm3
≤ max_storage_hm3`. **Default when absent: `min_storage_hm3`** (the v1 convention —
fill exactly to the dead volume), so omitting the field reproduces the dead-volume
fill target. `v_fill` is used by the §3.2 pin and the §4.2 schedule validation.

### 4.2 Per-stage filling schedule + validation

`filling_inflow_m3s` becomes a **per-filling-stage schedule**. The resolved
per-stage hydro bounds already carry `filling_inflow_m3s` per stage, so a per-stage
vector needs no fundamentally new wiring — only the schema/parse surface and the
validation change.

- **Default — uniform flow.** When the user supplies no per-stage override, derive
  a constant impound rate `target[t] = F` such that `F · Σ_t ζ_t = v_fill − seed`.
  Uniform _flow_ (constant m³/s) is the physically natural default: the per-stage
  volume `ζ_t·F` auto-scales with stage duration and block count, and it degrades
  gracefully across phase boundaries where `ζ_t` changes. (Uniform _volume_ per
  stage would impose a time-varying flow that spikes in short stages.)
- **Per-stage overrides** are allowed (a different value per filling stage) and
  must satisfy the sum below.
- **Strong validation (hard error).** The schedule must land the trajectory exactly
  at the fill target:

  ```
  | Σ_t ζ_t · target[t] − (v_fill − seed) | ≤ ε_rel · |v_fill − seed|
  ```

  using the **resolved per-stage** `ζ_t` (not a single global `ζ`) and a **relative
  tolerance** (never a float `==`). Reject either side:

  - A sum **below** the required fill leaves the reservoir structurally short of
    `v_fill` with no slack able to recover it (the equality `rise + σ_below = ζ·target`
    forbids exceeding the per-stage target, so a later wet stage cannot make up an
    earlier shortfall).
  - A sum **above** the required fill lets the reservoir **over-impound past
    `v_fill` with `σ_below = 0` and no penalty signal** (when inflow permits), up to
    `max_storage` — and that excess is then _discarded_ by the §3.2 reset (water
    that could have served downstream energy, silently lost). It also breaks the
    §3.1 conservation identity. (Note: a too-large sum does **not** force `σ_below >
0`; the optimizer drives `σ_below → 0` whenever inflow allows, which is exactly
    the over-impound case — so "reject sum-above" is about waste and well-posedness,
    not about a forced penalty.)

- `seed` is `InitialConditions.filling_storage` for the hydro: `0` (empty pit) when
  `start_stage_id > 0`, or a partial level in `[0, min_storage)` when
  `start_stage_id == 0` (study starts mid-filling).
- **The denominator `v_fill − seed` is always strictly positive**, so the relative
  tolerance is well-posed and the uniform-flow default `F = (v_fill − seed) / Σ_t
ζ_t` is finite and nonzero: the seed is validated to `[0, min_storage)` and
  `v_fill ≥ min_storage`, so `seed < min_storage ≤ v_fill`. No zero-fill /
  divide-by-zero degenerate case is reachable, and the validator may rely on this
  invariant. `validate_filling_configs` already guarantees `start_stage_id <
entry_stage_id`, so at least one filling stage always exists.

## 5. Cut validity: the new trap + parity obligations

Extends the §4 trap checklist of the v1 document.

1. **Stale-`β` at the entry boundary.** At `id == entry`, the state reset must zero
   the harvested `β_h` by relocating `v_in` out of the entry-stage balance
   (RHS-fold to `v_fill`), **not** by pinning the incoming column to a constant via
   bounds — a bound-pin leaves a nonzero reduced cost that becomes a stale-nonzero
   slope on the filling predecessor's value function (a wrong cut that compiles).
   Mirrors the PreFilling absent-dam flat-cut construction. **The reset is
   storage-coordinate-only: the AR-lag state and its `β` cross the boundary
   unchanged** (§3.2) — do not zero them.

2. **Bake the reset into the per-stage template, not a solve-time patch.** Realize
   the RHS-fold where PreFilling lives — in the per-stage `StageTemplate`
   (matrix/RHS build), **not** as a forward/backward-only solve-time patch — so
   every consumer (forward, backward, lower-bound) inherits it from the same baked
   template. The lower bound is not at risk today (`evaluate_lower_bound` solves
   **stage 0 only**, and the entry stage can never be stage 0 since `start_stage_id
< entry_stage_id` forces `entry_stage_id ≥ 1`), so the v1-draft "verify the LB
   applies the reset" framing was moot. The durable contract is the template
   baking: a reset applied only on the forward/backward solve path would diverge
   from any consumer that rebuilds from the template — the same template-vs-patch
   discipline the cut pool and PreFilling already follow.

No new dual-extraction trap is introduced: the target-row dual and the removed
operating floor both stay within the "soft-row dual folds into the single
incoming-storage reduced cost" contract that `extract_duals_from_view` already
honors. The target row puts `+1` on `v_out` and `−1` on `v_in` (a difference row,
unlike the v1 `σ_fill`/`σ^{v-}` rows that touch only `v_out`); the `−1` feeds `β_h`
naturally through LP duality and needs no special handling — do **not** hand-extract
the target-row dual.

(The v1 reformulation draft listed a second trap — "target-row routing parity" —
requiring the short-circuit to push routed inflow onto the target row. That trap is
**retired**: the target row carries no inflow term, so there is nothing to route
onto it; see §3.4.)

## 6. Accepted trade-offs

- **Decoupling / no over-fill incentive.** The reset severs the storage-dimension
  future-cost signal from operation into filling, so the optimizer never
  voluntarily over-fills above `v_fill` (there is no reward), and the filling
  sub-tree's cuts bound only its own within-filling costs (`σ_below`, withdrawal,
  spillage). This matches the intent "operation always starts at `v_fill`
  regardless of how filling went," and loses only the (second-order) marginal value
  of ending filling above `v_fill`.
- **The boundary is an asserted state, not an equivalence.** In a dry scenario
  where filling falls short, the pin asserts a starting volume the reservoir did not
  physically accumulate. v1 instead carries the real deficit into operation and
  recovers via `σ^{v-}`; this reformulation pays the deficit as a filling penalty
  (`Σ_t ζ_t·σ_below[t]`, at `filling_target_violation_cost` — expected to dominate
  the other recourse penalties, anchored by §7's re-targeted ordering check) and
  resets to `v_fill`.
  These are **not equivalent**: a fixed filling penalty for the deficit volume is
  not the same as operation actually starting low and incurring scenario-dependent
  downstream deficit/recovery costs over the operating horizon. The reformulation is
  a deliberate modeling choice (structural symmetry + a confined below-dead-volume
  window), not a strictly more accurate model.
- **When the reset is exact.** The pin is **bit-exact** in any scenario —
  deterministic or stochastic — where the schedule is _achievable_, i.e. inflow
  meets `ζ_t·target[t]` at every filling stage ⇒ `σ_below = 0` ⇒ `v_(entry−1) =
v_fill` ⇒ the pin reasserts a value the physics already produced. Note that the §4
  sum check guarantees the schedule _sums_ correctly but **not** that inflow meets
  it stage-by-stage; a deterministic study with front-loaded targets and
  back-loaded inflow can still leave `σ_below > 0` and invent a little boundary
  water.
- **Localized non-conservation.** The invented volume appears solely as the
  entry-stage incoming-storage constant and is **never routed as release onto any
  downstream row** (the entry-stage releases derive from `v_fill` like any operating
  release), so cascade conservation downstream is intact — the discontinuity is
  local to the reset boundary's own mass balance.

## 7. Resolved questions

- **Fill-target level — RESOLVED (configurable).** v1 §7-Q1 is resolved in favor of
  a separately-configurable fill target `v_fill = initial_operating_volume_hm3 ≥
min_storage` (§4.1), defaulting to the dead volume. §3.2's pin and §4.2's
  validation use `v_fill`; the operating **floor** stays hard at `min_storage`. The
  buffer `v_fill − min_storage` is the robustness mitigation for §3.3
  first-operating-stage feasibility.
- **`storage_violation_below_cost` — RESOLVED (remove).** With `σ^{v-}` removed the
  field is unused. **Remove it.** This is a **breaking config change** — bump
  accordingly and call it out in the migration. (A future per-hydro soft-floor
  option, if ever wanted, reintroduces its own field rather than relying on this
  one.)
- **Re-anchor the penalty ordering — do not just delete it.**
  `storage_violation_below_cost` is the **only** comparand in the live
  penalty-ordering check (`check_penalty_ordering`, a non-blocking `ModelQuality`
  warning that asserts `filling_target_violation_cost > storage_violation_below_cost`).
  Removing the field orphans that check, leaving `filling_target_violation_cost` —
  the penalty the **entire filling mechanism depends on dominating** all other
  recourse, so the optimizer fills rather than deficits or spills — with **zero**
  ordering validation. "Highest penalty in the system" is asserted in prose
  (`resolved::penalties`, `rows.rs`) but never hard-enforced. **Re-target the
  check** to a direct `filling_target_violation_cost > deficit_cost` warning
  (keeping the existing `ModelQuality`/warning severity convention); the owner may
  promote it to a hard error if the dominance must be guaranteed rather than
  advised. Shipping the dominant filling penalty ordering-unanchored is the one real
  spec-completeness gap this removal would otherwise introduce.
- **Filling cascades** (a Filling reservoir downstream of another Filling
  reservoir, both in Filling at the same stage) — believed to need **no** special
  handling. Each reservoir carries its own per-stage target + slack; they couple
  only through the normal cascade releases (the upstream retains its target and
  passes the rest, which the downstream sees as inflow on its **balance** row — and
  per §3.4 the target row needs no inflow term, so no special routing is required
  even when both are filling). A downstream that consequently falls short absorbs it
  in its own `σ_below` — physically correct. Confirm with a cascade case (the d40
  suggestion below).

## 8. Migration from the shipped model

- **Retire**: the impound-cap retention row (becomes the §3.1 target equality), the
  terminal `σ_fill` column/row/geometry family (folds into per-stage `σ_below`), the
  `σ^{v-}` operating-floor column/row/geometry family (removed entirely), and the
  `storage_violation_below_cost` field (its ordering validation is **re-anchored**
  onto the deficit cost, not merely deleted — §7).
- **Add**: the per-stage target column/row/geometry family, the §3.2 entry-stage
  RHS-fold reset, and the `initial_operating_volume_hm3` field (§4.1) on
  `FillingConfig` with its `[min_storage, max_storage]` validation and dead-volume
  default.
- **Rework**: `fill_storage_columns` (operating floor goes back to hard for filling
  hydros), the input schema/parse for the per-stage schedule and the new fill-target
  field, and `validate_filling_configs` / the `cobre-io` semantic validator for the
  §4.2 sum check (against `v_fill`) and the fill-target bounds, and **re-anchor**
  `check_penalty_ordering` onto the deficit cost (§7). **No rework of
  `fill_prefilling_shortcircuit`** beyond deleting its retention-row push (§3.4) —
  the balance-row routing is unchanged.
- **Re-derive** the deterministic cases and both-backend parity baselines: `d38`
  (its `σ_fill`/`σ^{v-}` assertions change to per-stage `σ_below` + the pinned
  operating start at `v_fill`) and `d39` (the PreFilling-upstream regression, now
  asserting the routed inflow lands on the **balance** row only and the target row
  is inflow-free). Add a `d40` filling-cascade case to discharge §7's cascade
  question. Re-confirm parity-neutrality for the non-filling world.
- **Doc-comment fixes**: `FillingConfig.filling_inflow_m3s` (now a per-stage
  retention _target_, not a cap), the new `initial_operating_volume_hm3` doc, and
  the entry-stage reset Contract (§3.2/§3.3).
- **Preserve (do not regress)**: filling-hydro spillage stays unbounded above (§3.1
  contract), and the reset is baked into the per-stage template rather than applied
  as a solve-time-only patch (§5).

The new model touches the same sites the v1 implementation does
(`lp/builder/{mod,entries,rows,columns,layout,template}.rs`, `setup/mod.rs`,
`training/lower_bound.rs`, the validators, and the per-stage geometry), so it is a
focused re-formulation rather than new architecture — but it is a real one, and the
state-reset (§3.2) is the part to implement and review with the most care.
