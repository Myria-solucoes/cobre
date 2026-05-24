# Inflow vs Anticipated State Layout — Critical Comparison

Code under review: branch `feat/anticipated-thermals-pre-horizon-seeding`, commit `241c057`.

## SUPERSEDED — verdict reversed 2026-05-23

> **WARNING**: the original "symmetric" verdict in this document is WRONG.
> An empirical follow-up showed that for K≥2 anticipated plants under the
> legacy fishing predicate, decisions `d_t` for `t ≥ 1` are silently driven
> to zero by a spurious coefficient cancellation in the cut row. See the
> end of this document for the empirical evidence and a follow-up plan.
>
> Root cause: `state_to_lp_column`'s Less branch maps slot K-2 to
> `state_col[K-1]`, whose LP-solution value is `incoming - decision` (Cat 6
> with decision-write coefficient), NOT `incoming`. The cut row therefore
> develops a spurious term proportional to the current-stage decision.
>
> The inflow path does NOT have this defect because its lag-fixing Cat 6
> rows have no decision-write coefficient — `state_col[lag_l]` equals
> `incoming_lag_l` exactly. The post-shift `z_inflow` value lives in a
> SEPARATE LP column with its own definition row. The anticipated path is
> missing that separate column.
>
> The original body of this document (below) correctly identifies the
> layout mappings but incorrectly concludes that the two are structurally
> equivalent. Read the empirical addendum at the end before the body.

---

This note answers a single question raised about `state_to_lp_column`'s
three-branch remap for anticipated-state slots: is the anticipated layout
structurally symmetric to the inflow-lag layout, or is the remap a symptom
of an asymmetry that we should refactor away?

The short answer (verdict at the end): ~~**the two layouts ARE structurally
symmetric**~~ **They are NOT symmetric — see the addendum at the end.**
The "three branches" of `state_to_lp_column`'s anticipated
arm correspond one-to-one to the "two branches" of the inflow-lag arm
plus the padding case. ~~Once the lag and anticipated arms are written in
the same form, the remap is the _same_ pattern (shift then innovation),
not an asymmetry.~~ The mapping has the same SHAPE but lands on a
DIFFERENT KIND of LP column for the anticipated case: a column whose value
is corrupted by the decision-write coefficient.

## Layout summary

Notation: $N$ = `hydro_count`, $L$ = `max_par_order`, $A$ = `n_anticipated`,
$K$ = `k_max`. Indices follow SDDP.jl-style state-then-control LP layout.

| Aspect                                     | Inflow-lag block                                                                                                                                             | Anticipated-state block                                                                                                                                                                   |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| State range                                | `inflow_lags = [N, N(1+L))` (`indexer.rs:830`)                                                                                                               | `anticipated_state = [N(1+L), N(1+L)+AK)` (`indexer.rs:1090-1102`)                                                                                                                        |
| Layout order                               | Lag-major: `start + l*N + h` (`indexer.rs:251-254`)                                                                                                          | Slot-major: `start + slot*A + plant` (`indexer.rs:289-297`)                                                                                                                               |
| State-fixing rows                          | `lag_fixing = [N, N(1+L))`, diagonal `+1` (`matrix.rs:1094-1101`)                                                                                            | `anticipated_state_fixing = [N, N(1+L)+AK)`, diagonal `+1` (`matrix.rs:1014-1026`)                                                                                                        |
| Innovation column at stage t               | `z_inflow = [N(1+L)+AK, …+N)` (`indexer.rs:1104-1105`)                                                                                                       | `anticipated_decision` columns (`indexer.rs:1124-1130`)                                                                                                                                   |
| Innovation injection into state-fixing row | None at stage t (the z-inflow row defines `z_h - sum(psi_l * lag_in[h,l]) = base+sigma*eta`, `matrix.rs:1629-1659`)                                          | At slot `K_i-1` for plant `i`: column `anticipated_decision.start+i` adds `+1` to row `state_fixing.start + (K_i-1)*A + i` (`matrix.rs:1040-1062`)                                        |
| Shift function                             | `shift_lag_state` (`noise.rs:161-183`): newest lag (slot 0) ← `z_inflow` primal; older lags shift up by one                                                  | `shift_anticipated_state` (`noise.rs:253-297`): slot `K_i-1` ← decision primal; older slots `0..K_i-2` shift down by one; slots `K_i..K` zero-padded                                      |
| Direction in ring                          | "Innovation at slot 0, older walks up to slot L-1, oldest falls out"                                                                                         | "Innovation at slot K_i-1, older walks down to slot 0, oldest (slot 0) falls out into fishing row at successor"                                                                           |
| `state_to_lp_column` remap                 | `lag == 0` → `z_inflow.start + h`; `lag >= 1` → `inflow_lags.start + (lag-1)*N + h` (`indexer.rs:1434-1442`)                                                 | `slot+1 == K_i` → `anticipated_decision.start + plant`; `slot+1 < K_i` → `anticipated_state.start + (slot+1)*A + plant`; `slot+1 > K_i` → `j` identity (padding) (`indexer.rs:1394-1430`) |
| Dual extraction                            | `view.dual[lag_fixing]` → cut coefficient at state index `j` (`backward.rs:407-416`)                                                                         | `view.dual[anticipated_state_fixing]` → cut coefficient at state index `j` (same slice, same loop)                                                                                        |
| Intercept formula                          | $\alpha = Q - \pi^\top \hat x$, with $\hat x$ = post-shift state (`backward.rs:466-474`)                                                                     | Same formula, same $\hat x$ (uniform over all `n_state` indices)                                                                                                                          |
| Trial-point capture                        | `current_state` = primal[0..n_state] then overwritten by `shift_lag_state` (`forward.rs:1047-1085`); captured into `state_at_capture` (`forward.rs:799-800`) | Same `current_state` buffer, overwritten by `shift_anticipated_state` (`forward.rs:1086-1091`); same capture                                                                              |

## The "shift then innovation" pattern, written symmetrically

The pattern is identical when we drop the "lag goes up, slot goes down"
surface difference. In both blocks, the **outgoing state vector at the
end of stage $t$** is

$$
x_t^{\text{slot}=k} = \begin{cases}
\text{innovation at stage } t & \text{if } k \text{ is the newest slot} \\
x_{t-1}^{\text{slot}=k \pm 1} & \text{if } k \text{ is an interior slot (shift)} \\
0 & \text{padding for plants with } K_i < K_{\max} \text{ (anticipated only)}
\end{cases}
$$

For inflows, "newest slot" is `lag=0`, "shift" is `lag ← lag-1` from
incoming, no padding because all hydros share the same `max_par_order`.
For anticipated, "newest slot" is `slot=K_i-1`, "shift" is `slot ← slot+1`
from incoming (older commitments drift toward slot 0 to meet their
delivery date), padding exists because `K_i <= K_{max}`.

The `state_to_lp_column` remap is what every LP-level Benders construction
must do whenever the post-shift state lives in _different LP columns_ than
the pre-shift state. This is the case for both blocks:

- **Inflow lag 0**: the outgoing-state slot holds the realized inflow
  $z_{t,h}$, which lives in LP column `z_inflow.start + h`, not in the
  lag column. (`indexer.rs:1438-1439`)
- **Inflow lag $l \ge 1$**: the outgoing slot holds the value that was
  pinned in `lag_fixing` row `l-1` of stage $t$, whose column is
  `inflow_lags.start + (l-1)*N + h`. (`indexer.rs:1440-1442`)
- **Anticipated slot $K_i-1$**: the outgoing slot holds the new decision,
  which lives in `anticipated_decision.start + plant`.
  (`indexer.rs:1401-1402`)
- **Anticipated slot $s < K_i-1$**: the outgoing slot holds the value
  that was pinned in `anticipated_state_fixing` row `slot+1` of stage $t$.
  (`indexer.rs:1403-1405`)
- **Anticipated slot $s > K_i-1$**: padding, identity remap is safe
  because the slot is zero-valued and its cut coefficient is zero
  (see the five-step invariant chain at `indexer.rs:1406-1426`).

If you ignore padding (which exists in anticipated only because of
heterogeneous $K_i$), the two cases are literally the same code with
indices renamed.

## Where the apparent asymmetry comes from

The user's suspicion was: _inflow does not need a "decision-write
coefficient on the state-fixing row", but anticipated does (`+1` on
`anticipated_decision[plant]` into `state_fixing[K_i-1, plant]`,
`matrix.rs:1057-1060`). Doesn't that break symmetry?_

It does **not** break symmetry — it is the symmetric counterpart of the
z-inflow definition row in the inflow block. Compare:

- **Inflow**: the realized inflow $z_{t,h}$ at stage $t$ is _not_ in the
  state-fixing row; it has its own equation row,
  `z_inflow_def: z_h - sum_l psi_l * lag_in[h,l] = base + sigma * eta`
  (`matrix.rs:1629-1659`). The z-inflow row gives a free column $z_h$ its
  value, but the state-fixing rows pin only the _incoming_ lags (the
  $L$ values transferred from stage $t-1$, not the $L+1$ values needed
  to describe the outgoing state). The "+1 innovation injection" happens
  implicitly because `state_to_lp_column` simply points lag-0 cut
  coefficients at the `z_inflow` column, which carries the dual that
  _via the z-inflow row_ propagates to the noise RHS and the older lags.
- **Anticipated**: the new commitment at stage $t$ is _not_ a separate
  noise-driven definition (it is a deterministic decision), so the LP has
  no analogue of `z_inflow_def`. Instead the constraint that "the value
  in slot $K_i-1$ equals the new decision" is folded directly into the
  state-fixing row at row `state_fixing.start + (K_i-1)*A + plant`,
  written as `state_col − decision_col = 0`. The `+1` on the decision
  column (`matrix.rs:1060`) and the `+1` on the state column
  (`matrix.rs:1023`) together encode that single equality.

Both blocks pay for the "innovation" in _some_ row of the LP. Inflow
splits it across two equations (`lag_fixing` plus `z_inflow_def`);
anticipated folds both into the `anticipated_state_fixing` row. This
choice changes _which_ LP column carries the dual of the innovation
slot's outgoing-state value, and `state_to_lp_column` knows to look up
the correct column in each case. Calling that asymmetry would be like
calling a derivative chain rule asymmetric.

## Verdict: SYMMETRIC

The two layouts are structurally symmetric in every observable property
of the cut machinery:

1. State-fixing-row block is diagonal `+1` on the incoming state
   columns (`matrix.rs:1094-1101` for inflow,
   `matrix.rs:1014-1026` for anticipated).
2. Innovation enters via a separate column (`z_inflow` /
   `anticipated_decision`), wired into exactly one equation per slot
   (`z_inflow_def` for inflow; `state_fixing[K_i-1]` directly for
   anticipated). Both wirings are local to the slot that receives the
   innovation.
3. Shift function writes innovation at one end of the ring and copies
   older values one position toward the "fall-out" end (`shift_lag_state`
   newest=lag 0 vs. `shift_anticipated_state` newest=slot $K_i-1$).
4. Trial-point capture is the post-shift state for both blocks
   (`forward.rs:1047-1093`).
5. Dual extraction is uniform: `view.dual[0..n_state]` is copied verbatim
   into the cut coefficients (`backward.rs:407-416`); the intercept
   formula $\alpha = Q - \pi^\top \hat x$ is applied with no per-block
   special case (`backward.rs:466-474`).
6. `state_to_lp_column` performs a slot-aware remap whose branches are
   _forced_ by the LP geometry: each block has $L+1$ (or $K_i+1$)
   conceptual outgoing slots but only $L$ (or $K_i$) state-fixing-row
   columns, so the newest outgoing slot necessarily references the
   innovation column. The remap is the _only_ place where the geometry
   surfaces, and it is local, pure, branch-by-shape.

The three-branch form on the anticipated arm
(`Equal`/`Less`/`Greater` at `indexer.rs:1401-1426`) is _required_ by
the heterogeneous $K_i$: padding slots only exist because plants share
a common ring width $K_{\max}$ but consume different prefixes
$[0, K_i)$. Inflow has the same dichotomy implicitly (one "newest"
branch, one "older" branch) and would have a third branch as well if
inflow lags were padded to a common length larger than each hydro's
actual AR order. Indeed, that is _exactly_ what the lag-padding code in
`set_nonzero_mask` deals with (`indexer.rs:1583-1596`).

## Bug-risk analysis

Symmetry is preserved, but the user is right that this is a load-bearing
invariant we should be vigilant about. Concretely:

1. **Padding-slot invariant**: the `Ordering::Greater` arm of
   `state_to_lp_column` returns `j` (identity) on the assumption that
   `shift_anticipated_state` writes `0.0` to padding slots (`noise.rs:292-295`).
   The `debug_assert!` at `forward.rs:353-362` catches a non-zero padding
   coefficient. **Risk**: if pre-horizon seeding ever injects a non-zero
   into a padding slot (the `AnticipatedCommitmentHistory` path mentioned
   at `indexer.rs:1420-1425`), the identity remap silently aliases a
   padding-cut-coefficient onto its own LP column — which is _not_ a
   real LP column for that plant. Mitigation: the debug_assert fires
   on first occurrence. **Action item**: when implementing pre-horizon
   seeding, replace the `Greater` arm with a proper column remap and
   re-prove the invariant.

2. **Order of LP-build helpers**: `fill_anticipated_state_fixing_entries`
   (`matrix.rs:1014`) and `fill_anticipated_decision_state_write_entries`
   (`matrix.rs:1040`) must both write into the same row for slot $K_i-1$,
   and the CSC build (`matrix.rs:1770-1771`) calls them in that order.
   If a future refactor inverts the order or sorts entries differently
   inside HiGHS, the duality semantics are unchanged (`+1` on state col,
   `+1` on decision col, RHS=0), but a `debug_assert` on column-entry
   ordering would catch a subtle re-sort bug.

3. **`shift_anticipated_state` is not called in the backward pass**
   (`backward.rs:56-62`). This is correct (the trial point is already the
   post-shift state captured in the forward pass), but it is fragile:
   if a future refactor mistakenly re-shifts in the backward pass, the
   intercept $\alpha = Q - \pi^\top \hat x$ would use a doubly-shifted
   $\hat x$ and silently corrupt cuts. The two integration tests cited
   at `backward.rs:60-62` (`anticipated_backward_cut_k1.rs`,
   `anticipated_backward_cut_k2.rs`) are the regression guard.

4. **Inflow lag 0 is the only "innovation slot" for that block**; the
   z-inflow row is purely deterministic given the lag values and the
   noise realization. Anticipated has $K_i-1$ as innovation slot per
   plant, and each plant has its own innovation row. Symmetry is
   pointwise but the count differs (one for inflow, $A$ for anticipated).
   This is a counting difference, not a structural asymmetry.

The bug classes that the inflow block does _not_ expose because it never
has padded slots — namely, "non-zero coefficient on an identity-remapped
slot" — are guarded for the anticipated block by the `debug_assert!` at
`forward.rs:353-362` and `forward.rs:506-516`. That defence-in-depth
should stay.

## Recommendation

~~**Keep current design.**~~ — **REVERSED**. See the empirical addendum
below for the corrected verdict.

The two layouts are symmetric in surface mapping (the state_to_lp_column
arms have the same shape), but the LP column they map to is structurally
different:

- For inflows, the Less branch maps to `state_col[lag_l-1]` whose
  LP-solution value equals `incoming_lag_l-1` exactly (pure Cat 6
  identity).
- For anticipated, the Less branch maps to `state_col[slot s+1]` —
  which when `s+1 == K_p-1` equals `incoming_slot_{K-1} - decision_col`
  (Cat 6 plus decision-write). This corrupts the cut coefficient.

The "decision write folded into the state-fixing row" choice is the
specific structural defect. Inflows separate the post-shift value into
its own column (`z_inflow`) with its own definition row, leaving the
state-fixing row pure. Anticipated does not. This was the asymmetry the
original verdict missed.

---

## Empirical addendum — verdict reversed

**Test**: `simulation_ring_buffer_shifts_anticipated_state_k2` in
`crates/cobre-sddp/tests/anticipated_simulation_ring_buffer.rs` with
K=2, n_stages=6, anticipated cost 10 $/MWh, backup 5000 $/MWh, load 150 MW,
seed values_mw = [100.0, 50.0].

**Analytical optimum**: every in-horizon `d_t` (for t = 0, 1, 2, 3 — the
stages where the decision column is active per the strict boundary
predicate `t + K_i < n_stages`) should saturate at `max_gen = 200 MW`
because the future backup-savings benefit per unit `d_t` (5000·H·D*{t+K}
per MWh) far exceeds the commit cost (10·H·D*{t+K} per MWh).

**Observed under legacy code (post-Epic-03 revert)**:

```
[K=2 diag] stage=0 d_t=Some(100.0)  c_t=None
[K=2 diag] stage=1 d_t=Some(0.0)    c_t=None
[K=2 diag] stage=2 d_t=Some(0.0)    c_t=Some(100.0)
[K=2 diag] stage=3 d_t=Some(0.0)    c_t=Some(-0.0)
[K=2 diag] stage=4 d_t=None         c_t=Some(-0.0)
[K=2 diag] stage=5 d_t=None         c_t=Some(-0.0)
```

(The `d_0 = 100` saturation is at the anticipated thermal's max_gen.)

Only `d_0` saturates. `d_1`, `d_2`, `d_3` stay at zero. `committed_at(2)`
correctly delivers `d_0 = 100` via the ring-buffer shift to slot 0 at
stage 2's fishing. But stages 3+ deliver zero MW from the anticipated
thermal because the LP refused to commit at stages 1+.

The user-visible consequence: cobre's anticipated thermals dispatch
correctly at stage 0 only; for all subsequent in-horizon decision stages,
the LP fails to exploit the cost asymmetry. For a real study with
multi-year horizons, this means anticipated thermals deliver during
one stage and then stay silent, wasting cheap capacity.

**Existing tests don't catch this** because they assert:

- `d_0 != 0` (passes — `d_0 = 100`)
- `committed_at(t=K) == decision_at(t=0)` (passes — shift propagation works mechanically)

Neither assertion exercises `d_t` for `t ≥ 1`.

## Root cause (algebraic trace)

The cut at stage 1's FCF from stage 2's backward pass has:

- `π_slot_0 = -5000·H·D_2` (stage 2 fishing pins g_a_2 = incoming_slot_0_at_stage_2)
- `π_slot_1 = -5000·H·D_3` (slot 1 at stage 2 propagates via shift to slot 0 at stage 3's fishing)

`state_to_lp_column` at stage 1 for K=2 routes:

- slot 0 → `state_col[slot 1]` (Less branch)
- slot 1 → `anticipated_decision_col` (Equal branch)

`state_col[slot 1]` at stage 1 is constrained by Cat 6 row
`state_col[slot 1] + d_1 = incoming_slot_1` → its LP-solution value is
`incoming_slot_1 - d_1 = d_0 - d_1` (where `d_0` is the previous-stage
constant patched into the RHS).

The cut row in stage 1's LP becomes (LHS ≥ alpha form):

```
θ_1 + 5000·H·D_2 · (d_0 - d_1) + 5000·H·D_3 · d_1 ≥ alpha_intercept
```

Coefficient on `d_1`: `-5000·H·D_2 + 5000·H·D_3 = -5000·H·(D_2 - D_3)`.

With NPV discount `D_2 > D_3` (earlier stages discounted less), this is
a small negative number ≈ `-5000·H·β·D_3` for small `β`.

Total LP cost-vs-d_1 coefficient = commit cost + cut tightening =
`+10·H·D_3 + 5000·H·(D_2 - D_3)` ≈ `H·D_3·(10 + 5000·β)`.

For typical β=5%: `D_3·H·260 > 0` → LP picks `d_1 = 0`.

The bug: slot 0's coefficient `-5000·H·D_2` lands on a column
(`state_col[slot 1]`) whose VALUE includes `-d_1`, producing a
`+5000·H·D_2 · d_1` term in the cut row. This **cancels** the
beneficial `-5000·H·D_3 · d_1` term that comes from slot 1's coefficient
landing on the decision column. The cancellation grows tighter as
β → 0, and the residual is always positive (since `D_2 > D_3`), so
the LP picks `d_1 = 0` regardless of how favorable the cost asymmetry is.

## Why the original verdict was wrong

The original analysis (above this addendum) correctly identified that
the anticipated layout has "decision write folded into the state-fixing
row" while inflows have "a separate z_inflow column". It declared this
"equivalent" without checking whether the LP-solution value of the
column targeted by `state_to_lp_column` equals the value the cut wants
to reference.

Inflow path: `state_col[lag_l]` = `incoming_lag_l` (pure identity). ✓
Anticipated path: `state_col[slot K-1]` = `incoming - decision` (Cat 6
plus decision write). The cut targets this column expecting `incoming`,
but the LP sees `incoming - decision`, producing the spurious
`-decision` term in the cut row.

The original verdict treated `state_col[slot K-1]` as if it were the
post-shift outgoing state column (analogous to inflow's lag column). It
is not. The post-shift value is computed by `shift_anticipated_state`
into the WORKSPACE state vector, not into any LP column. The
`state_col[slot K-1]` column is described in
`anticipated_simulation_ring_buffer.rs:9-14` (module doc) as
"a residual that has no physical meaning" — which is exactly the
problem when the cut machinery treats it as if it carried the post-shift
value.

## Recommended fix

The architectural change is the same as the originally-deferred "Option
C" from the pre-horizon-seeding plan:

1. Drop the decision-write coefficient from the Cat 6 row at slot K-1.
   Now `state_col[slot K-1]` equals `incoming_slot_K-1` (pure identity,
   matching inflow's pattern).
2. Add a separate LP column `state_out_slot_K-1[plant]` per stage with
   its own definition row: `state_out_slot_K-1 = decision_col`. This
   captures the post-shift value, mirroring `z_inflow_def`.
3. Update `state_to_lp_column`:
   - slot K-1 (Equal branch) → `state_out_slot_K-1[plant]` column
     (equivalent to `decision_col` since they're pinned equal).
   - slot s < K-1 (Less branch) → `state_col[slot s+1]` (now pure
     identity to `incoming_slot_{s+1}`, no decision-write corruption).
   - Padding (Greater branch) → unchanged.

After this fix:

- The K=1 same-stage collision (Epic 03 always-active flip) disappears
  because slot 0 of fishing-read is `state_col[slot 0]` (pure identity)
  while the decision write lives on `state_out_slot_K-1` (a different
  column).
- The K≥2 d_t-stuck-at-zero bug disappears because the cut targets a
  column whose value equals `incoming_slot_K-1`, not `incoming - decision`.
- The pre-horizon seeded values_mw[k] become deliverable because the
  fishing constraint at stages [0, K-1) reads `state_col[slot 0]` which
  equals the seed value (pure identity), not a corrupted residual.

Estimated effort: 5-8 tickets in a new plan. The fix touches
`indexer.rs` (column layout), `lp_builder/layout.rs` (column count),
`lp_builder/matrix.rs` (the new definition row + Cat 6 cleanup),
`lp_builder/patch.rs` (RHS patching for the new row), and `workspace.rs`
(wire format adjustment for the new column).
