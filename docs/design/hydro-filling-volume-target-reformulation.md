# Hydro filling: volume-target reformulation

> **Status**: Proposed reformulation — not yet implemented. Companion to
> `hydro-dead-volume-filling.md` (the shipped v1 model). It **supersedes** that
> document's §3.1 (impound cap), §3.3 (terminal `σ_fill`), §3.4 (`σ^{v-}` soft
> operating floor), §3.7 (filling→operating handoff), and §5.3 (filling-inflow
> semantics). It **keeps unchanged**: §1 lifecycle, §3.2 PreFilling cascade
> short-circuit, §3.5 FPHA exclusion, §3.6 generation/turbine/diversion/evaporation
> gating, and the §4 cut-validity invariants — except where §3.2/§3.4 below add new
> requirements.
>
> **Validation**: a formulation review against the live LP builder returned
> SOUND-with-caveats. The caveats are not optional and are folded in below as hard
> requirements — chiefly the state-reset realization (§3.2), the routing port
> (§3.4), and the two new cut-validity traps (§5).
>
> **Contracts it must not disturb**: `.claude/rules/sddp.md` (Benders cut
> sign/scale, column-bound state pinning, FPHA average storage, append-only cut
> pool with slot-identity warm start) and the shared `commissioning_active` /
> `filling_phase` predicates in `lp/builder/mod.rs`.

## TL;DR

The shipped model makes a plant that **went through filling** carry a soft
storage floor (`σ^{v-}`) at **every** operating stage: it can dip below its dead
volume by paying `storage_violation_below_cost`, while an otherwise-identical
plant that was never filled has a **hard** floor and physically cannot. Two
equivalent reservoirs obey different operating constraints purely because of
commissioning history — a modeling inconsistency, not just an aesthetic one.

This reformulation removes that asymmetry. The enabling idea: **guarantee that
operation starts at the dead volume** (pin the operating-start storage to it), so
the under-filled-start case that `σ^{v-}` existed to absorb cannot arise. With
that guarantee the operating phase uses a **hard** floor — structurally identical
to a normal plant. The filling phase is reframed as a **per-stage retention
target with a below-slack** (modeled like water withdrawal), and the terminal
`σ_fill` collapses into those per-stage slacks.

Net structural change: **retire two row/column families** (`σ_fill` terminal
target and `σ^{v-}` operating floor) and the impound-cap row; **add one** per-stage
target family and a boundary state-reset.

---

## 1. What changes, per mechanism

| Mechanism                 | Shipped v1                                                                                  | Reformulation                                                                                 |
| ------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Filling impound           | per-stage **cap** `rise ≤ ζ·F_h` (inequality, no slack)                                     | per-stage **target** `(v_out − v_in) + σ_below = ζ·target` (equality with below-slack) (§3.1) |
| Filling shortfall penalty | a single **terminal** `σ_fill` at id `entry−1`                                              | the **per-stage** `σ_below` slacks (their sum is the terminal deficit) (§3.1)                 |
| Filling→operating handoff | end-of-filling storage flows in via the pin chain (continuous)                              | operating-start storage **pinned to the dead volume** — a state reset (§3.2)                  |
| Operating storage floor   | **soft** `σ^{v-}` at every stage (filling hydros only)                                      | **hard** `min_storage_hm3`, identical to a normal plant — `σ^{v-}` family removed (§3.3)      |
| PreFilling short-circuit  | routes the absent dam's water onto the downstream's balance + the impound-cap retention row | unchanged, but the retention-row routing **ports to the new target row** (§3.4)               |
| `filling_inflow_m3s`      | a single per-stage cap value                                                                | a **per-filling-stage schedule** (vector), uniform-flow default, strongly validated (§4)      |

Unchanged: the three-phase lifecycle keyed on `stage.id`, the PreFilling
frozen-identity row + cascade short-circuit, FPHA exclusion, the `[0,0]`
generation/turbine/diversion gating, and the dense-storage-column / declaration-
order invariants.

## 2. Locked decisions

1. **Pin to the dead volume.** Operation always begins at the dead volume; a
   filling shortfall is paid as a per-stage penalty and **not** carried forward as
   a reduced operating-start storage. This is what licenses the hard operating
   floor.
2. **Target replaces the cap.** Each filling stage carries a retention target with
   a below-slack; over-impounding is bounded by the target itself.
3. **Per-stage user-supplied schedule, uniform default, strong validation.**
   `filling_inflow_m3s` becomes a per-filling-stage schedule (a different value per
   filling stage is allowed), defaulting to the internal uniform resolution, with
   validation that the schedule is consistent with reaching the dead volume (§4).

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

The per-stage slacks **replace** the v1 terminal `σ_fill`: their sum
`Σ_t σ_below[t]` equals the end-of-filling deficit `dead_volume − v_(entry−1)` —
exactly what `σ_fill` measured — so penalizing the schedule loses nothing versus
penalizing the terminal volume, at a smoother per-stage signal. Keep both would
**double-penalize** the same shortfall; use the **same cost field**
(`filling_target_violation_cost`) so the aggregate penalty is conserved.

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
penalty as if it had filled, **and still start operation full**, banking the
water's value twice. Anchoring the slack to the rise closes that arbitrage:
releasing instead of retaining shrinks `v_out − v_in`, which grows `σ_below`,
which costs. Done the withdrawal way (slack = schedule − actually-achieved), the
penalty genuinely incentivizes filling.

`target[t]` is a **retention target on the natural inflow already in the
balance** — not an extra inflow source added to it. PAR and AR-lag coupling run
normally during filling (carried over from v1 §3.1).

### 3.2 The filling→operating boundary: pin to the dead volume (state reset)

At the first operating stage (`id == entry_stage_id`), the incoming-storage state
is set to the dead volume (`min_storage_hm3`), **decoupled** from the last filling
stage's outgoing storage. So operation always starts at the dead volume regardless
of how filling actually went. No Benders cut crosses this boundary in the storage
coordinate — the filling sub-tree and the operating sub-tree are independent in
that dimension, exactly as the stage-0 seed roots the storage coordinate.

**Hard requirement — realize the reset the PreFilling way, NOT as a column-bound
pin.** `fill_col_state_patches` unconditionally pins every hydro's incoming-storage
column to the forward-propagated trial point, and `extract_duals_from_view`
unconditionally harvests that same column's reduced cost as the cut subgradient
`β_h`, for **every** state coordinate. If the reset is realized by pinning the
entry-stage incoming column to the constant `dead_volume` via bounds, that column's
reduced cost (the dual of `lb==ub`) is generally **nonzero**, and the cut machinery
attaches it to the **last filling stage's** value function — a stale-nonzero slope
on a coordinate whose true marginal value is zero. That is a wrong cut that
compiles (the §4 trap-4 failure mode).

The correct realization mirrors the PreFilling frozen-identity construction:
**relocate `v_in` out of the entry-stage reservoir-balance row and fold the
constant `dead_volume` into that row's RHS**, so the incoming-storage column is a
dead variable at the entry stage ⇒ its reduced cost is `0` ⇒ `β_h = 0` (a correct
flat cut on the filling predecessor). The forward-propagated `current_state[h]`
still flows in but is ignored by the entry-stage LP. The storage column keeps its
dense index at every stage (no omission, no relocation of the _column_ — only its
row coupling changes at this one stage), so the dense-storage and declaration-order
invariants hold.

### 3.3 Operating: hard floor, no `σ^{v-}`

With operation guaranteed to start at the dead volume (§3.2), the operating-phase
storage floor is **hard** at `min_storage_hm3` — bit-identical to a never-filled
plant. The entire `σ^{v-}` operating-floor row/column family is **removed**.
`storage_violation_below_cost` becomes unused (reserved; see §7).

Feasibility: at the first operating stage `v_in = dead_volume = min_storage`, so
`v_out ≥ min_storage` requires only `release ≤ inflow` — always achievable. A
mid-operation drought is handled exactly as it is for any normal plant: reduce
outflow toward `min_outflow` (which itself rides the existing
`outflow_violation_below` slack) and, ultimately, load deficit. The filled plant
is now genuinely in the normal-plant regime.

> **Contract**: dropping `σ^{v-}` is sound **only** because the reset (§3.2)
> guarantees operation starts at the dead volume. Tie the two together with a
> comment at the entry-stage reset site — weakening the reset back to inheriting
> the (possibly short) filling storage would re-introduce first-operating-stage
> infeasibility.

### 3.4 PreFilling-upstream routing ports to the target row

The PreFilling cascade short-circuit (`fill_prefilling_shortcircuit`) routes an
absent upstream dam's realized inflow (`z`), transitive upstream releases, and
diversion onto its first non-PreFilling downstream's water-balance row, and — for a
**Filling** downstream — onto that downstream's retention/cap row, so the cap sees
the routed natural inflow. The new per-stage **target** row (§3.1) **replaces** the
retention row, so the short-circuit's routing must push the same terms onto the
**target** row. This is a direct port, not new logic, and it is **not optional**:
omitting it makes the target's effective inflow undercount the routed water and
mis-constrains the rise (a too-weak constraint — a silent correctness gap, not a
wrong cut).

## 4. Input model: per-stage filling schedule + validation

`filling_inflow_m3s` becomes a **per-filling-stage schedule**. The resolved
per-stage hydro bounds already carry `filling_inflow_m3s` per stage, so a per-stage
vector needs no fundamentally new wiring — only the schema/parse surface and the
validation change.

- **Default — uniform flow.** When the user supplies no per-stage override, derive
  a constant impound rate `target[t] = F` such that `F · Σ_t ζ_t = dead_volume −
seed`. Uniform _flow_ (constant m³/s) is the physically natural default: the
  per-stage volume `ζ_t·F` auto-scales with stage duration and block count, and it
  degrades gracefully across phase boundaries where `ζ_t` changes. (Uniform
  _volume_ per stage would impose a time-varying flow that spikes in short stages.)
- **Per-stage overrides** are allowed (a different value per filling stage) and
  must satisfy the sum below.
- **Strong validation (hard error).** The schedule must land the trajectory exactly
  at the dead volume:

  ```
  | Σ_t ζ_t · target[t] − (dead_volume − seed) | ≤ ε_rel · |dead_volume − seed|
  ```

  using the **resolved per-stage** `ζ_t` (not a single global `ζ`) and a **relative
  tolerance** (never a float `==`). A sum _below_ the required fill leaves the
  reservoir structurally short of the dead volume with no slack able to recover it
  (the equality forbids exceeding the target); a sum _above_ makes `σ_below`
  structurally non-zero. Reject either.

- `seed` is `InitialConditions.filling_storage` for the hydro: `0` (empty pit) when
  `start_stage_id > 0`, or a partial level in `[0, min_storage)` when
  `start_stage_id == 0` (study starts mid-filling).

## 5. Cut validity: two new traps

Both extend the §4 trap checklist of the v1 document.

1. **Stale-`β` at the entry boundary.** At `id == entry`, the state reset must zero
   the harvested `β_h` by relocating `v_in` out of the entry-stage balance
   (RHS-fold to the dead volume), **not** by pinning the incoming column to a
   constant via bounds — a bound-pin leaves a nonzero reduced cost that becomes a
   stale-nonzero slope on the filling predecessor's value function (a wrong cut that
   compiles). Mirrors the PreFilling absent-dam flat-cut construction.
2. **Target-row routing parity.** The PreFilling short-circuit must push the routed
   inflow onto the per-stage **target** row (the retention row's successor), or the
   target's effective inflow undercounts (§3.4).

No new dual-extraction trap is introduced: the target-row dual and the removed
operating floor both stay within the "soft-row dual folds into the single
incoming-storage reduced cost" contract that `extract_duals_from_view` already
honors. The target row puts `+1` on `v_out` and `−1` on `v_in` (a difference row,
unlike the v1 `σ_fill`/`σ^{v-}` rows that touch only `v_out`); the `−1` feeds `β_h`
naturally through LP duality and needs no special handling — do **not** hand-extract
the target-row dual.

## 6. Accepted trade-offs

- **Decoupling / no over-fill incentive.** The reset severs the storage-dimension
  future-cost signal from operation into filling, so the optimizer never
  voluntarily over-fills above the dead volume (there is no reward), and the filling
  sub-tree's cuts bound only its own within-filling costs (`σ_below`, withdrawal,
  spillage). This matches the intent "operation always starts at the dead volume
  regardless of how filling went," and loses only the (second-order) marginal value
  of ending filling above the dead volume.
- **Localized non-conservation at the boundary.** In a dry **stochastic** scenario
  where filling falls short, the pin asserts a starting volume the reservoir did not
  physically accumulate — bounded by `Σ_t ζ_t·σ_below[t]` and paid for, terminally
  equivalently, at the highest system penalty. This is **exact for deterministic
  studies** (achievable targets ⇒ `σ_below = 0` ⇒ the pin reasserts a value the
  physics already produced, bit-exact), and in stochastic studies it is a **local
  storage-state discontinuity at the reset boundary only** — the invented volume
  appears solely as the entry-stage incoming-storage constant and is **never routed
  as release onto any downstream row**, so cascade conservation downstream is intact.

## 7. Open questions

- **Fill-target level.** This reformulation keeps the v1 convention that the fill
  target _is_ the dead volume (`min_storage_hm3`). A real commissioning may instead
  fill to an _operational_ level **above** the dead volume (start with a buffer). If
  that is wanted, the target becomes a separately-configurable "initial operating
  volume" `≥ min_storage`, and §3.2's pin / §4's validation use that level instead
  of `min_storage`. Decide before locking.
- **Filling cascades** (a Filling reservoir downstream of another Filling
  reservoir, both in Filling at the same stage). Each reservoir carries its own
  per-stage target + slack; they couple only through the cascade releases (the
  upstream retains its target and passes the rest, which the downstream sees as
  inflow). A downstream that consequently falls short absorbs it in its own
  `σ_below` — physically correct. Believed to need no special handling beyond the
  §3.4 routing port; confirm with a cascade case.
- **`storage_violation_below_cost`.** With `σ^{v-}` removed it is unused. Keep it as
  a reserved penalty field (forward-compatible with a future per-hydro soft-floor
  option) or remove it — a cleanup decision, not a formulation one.

## 8. Migration from the shipped model

- **Retire**: the impound-cap retention row (becomes the §3.1 target equality), the
  terminal `σ_fill` column/row/geometry family (folds into per-stage `σ_below`), and
  the `σ^{v-}` operating-floor column/row/geometry family (removed entirely).
- **Add**: the per-stage target column/row/geometry family, and the §3.2 entry-stage
  RHS-fold reset.
- **Rework**: `fill_storage_columns` (operating floor goes back to hard for filling
  hydros), `fill_prefilling_shortcircuit` (route onto the target row), the input
  schema/parse for the per-stage schedule, and `validate_filling_configs` /
  the `cobre-io` semantic validator for the §4 sum check.
- **Re-derive** the deterministic cases and both-backend parity baselines: `d38`
  (its `σ_fill`/`σ^{v-}` assertions change to per-stage `σ_below` + the pinned
  operating start) and `d39` (the PreFilling-upstream regression, now asserting the
  routed inflow lands on the **target** row). Re-confirm parity-neutrality for the
  non-filling world.
- **Doc-comment fixes**: `FillingConfig.filling_inflow_m3s` (now a per-stage
  retention _target_, not a cap), and the entry-stage reset Contract (§3.3).

The new model touches the same sites the v1 implementation does
(`lp/builder/{mod,entries,rows,columns,layout,template}.rs`, `setup/mod.rs`,
`training/lower_bound.rs`, the validators, and the per-stage geometry), so it is a
focused re-formulation rather than new architecture — but it is a real one, and the
state-reset (§3.2) is the part to implement and review with the most care.
