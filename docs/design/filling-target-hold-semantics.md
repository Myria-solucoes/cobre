# Filling-target hold semantics: the soft penalty admits an LP-optimal dead-volume release

## Status

Observation / open spec question. No code defect — documents an intentional
modeling property and the open question of whether any use case requires a
guarantee the current model deliberately does not provide.

## Summary

A hydro in its `Filling` phase is steered toward its dead volume by a **soft**
filling-target constraint, not a hard hold. When a downstream active reservoir is
water-starved and can monetize water (an uncapped turbine), the LP can find it
optimal to **release a filling reservoir's impounded water — up to its entire dead
volume — at the last Filling stage** to avert downstream deficit. This is the
correct optimum for the model as specified; it is not a solver error and not a
penalty mis-calibration.

## Mechanism

The filling-target row is soft by construction (`fill_filling_target_rows`): it
emits `v_h + σ_fill ≥ V_target[t]` with `row_upper = +∞`, so the `σ_fill` slack
relaxes it and a hydro that fills short keeps a feasible LP. The slack is priced at
`filling_target_violation_cost` ($/hm³) in the objective
(`fill_filling_target_columns`); the target trajectory `V_target[t]` is the
backward fold from the dead volume (`build_filling_v_target`).

Because the constraint is soft, the LP compares the **cost of holding** (paying the
`σ_fill` penalty only if it under-fills) against the **cost of dumping** (paying the
`σ_fill` penalty to release water now, minus the downstream thermal/deficit it
displaces). When the displaced downstream cost per unit (deficit at
`deficit_cost`, typically far above `filling_target_violation_cost`) exceeds the
filling-target penalty, releasing the impounded water is the genuine optimum. The
`σ_fill` slack driven to the full dead volume is the constraint faithfully
reporting the shortfall, not a failure.

A finite horizon with zero discount amplifies the effect: there is no future stage
in which the held water earns its keep, so the end-of-horizon release incentive is
strongest at the last Filling stage.

## Evidence in the deterministic suite

Two cases share byte-identical penalty calibration (`filling_target_violation_cost`
500 $/hm³, `deficit_cost` 1000 $/MWh, `spillage_cost` 0.01), both finite-horizon
and zero-discount:

- The filling-cascade case documents this dump as the expected LP optimum in its
  fixture generator and **prevents** it by capping the downstream turbine small
  enough that the system cannot monetize a last-stage release — the reservoirs then
  impound as intended. The cap is a deliberate topology choice, not a penalty
  change.
- The dead-volume filling case has an uncapped downstream turbine, so once an
  upstream phantom-water artifact was removed (see the PreFilling spillage contract
  in `.claude/rules/sddp.md`) and the downstream reservoir became correctly
  water-starved, the filling reservoir releases its dead-volume water at the last
  Filling stage to feed it. Its simulation contract asserts the per-stage water
  balance, that the release reaches the downstream reservoir, and that `σ_fill`
  books the shortfall — encoding this as correct, conservation-respecting behavior.

## Open question

The model provides **no hard "hold the dam during Filling" guarantee** — by design.
If a future requirement needs one (e.g., a regulatory or operational rule that a
filling reservoir must reach its dead volume before any release to a downstream
plant), it is a modeling-feature gap, not a bug fix. Candidate mechanisms, in
increasing strength:

1. A `filling_target_violation_cost` provably exceeding the maximal downstream
   monetization (fragile — depends on downstream topology and prices).
2. A release/turbine cap on the filling hydro itself, or downstream (the existing
   topology-shaping approach).
3. A hard-floor variant of the filling-target row (a true `v_h ≥ V_target[t]`
   column-bound or hard row) gated behind an opt-in flag — this changes the
   feasibility surface and must preserve a feasible LP for genuinely
   under-filling inflows, so it needs its own design.

Until a use case requires it, the soft penalty is the correct default: it steers
toward the dead volume while keeping the LP feasible and letting a dominating
deficit cost override the target when that is genuinely cheaper for the system.
