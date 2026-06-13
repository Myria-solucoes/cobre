---
paths:
  - "crates/cobre-sddp/**/*.rs"
---

# SDDP Numerical & Algorithm Conventions

Hard-won correctness contracts of the SDDP solver. Each one is a _contract_, not
a style preference: a plausible-looking deviation produces wrong bounds, rejected
warm-starts, or silently understated cuts that still compile and pass most tests.
Verify against the cited code before changing any of them.

## Benders cut sign & subgradient extraction

The FCF stores the **raw subgradient** `∂Q/∂x` as a cut's `coefficients` (it is
_not_ negated at storage). That subgradient is the incoming-state column's
reduced cost **divided** by `col_scale`:
`∂Q/∂x_orig = rc_scaled / col_scale[col]` — divided, not multiplied, because the
pin sets `v_scaled = v_orig / col_scale`. Cut-row construction then negates the
gradient so the LP row reads `−∇·x + θ ≥ intercept`, yielding the Benders cut
`θ ≥ Q(x̂) + π'(x − x̂)`.
Read: `backward.rs` (`extract_duals_from_view`), `cut/fcf.rs`, and
`cut::row::push_scaled_coefficient`, where `batch.values.push(-coeff * d)`
applies the negation.

## State pinning uses column bounds, not equality rows

Incoming state is pinned with `set_col_bounds` on the incoming-state LP column.
The `storage_fixing`, `lag_fixing`, and `anticipated_state_fixing` ranges in
`StageIndexer` are permanent empty sentinels (`0..0`). Always resolve the LP
column — for both pinning and dual extraction — via
`StageIndexer::state_to_lp_incoming_column`; never assume a fixing-row index.
Read: `lp/indexer.rs`.

## FPHA uses average storage

The FPHA generation constraint is
`g ≤ γ₀ + (γᵥ/2)·(V_in + V_out) + γ_q·q (+ γ_s·s)`. The `−γᵥ/2` coefficient
appears on **both** the incoming and outgoing storage columns — not on `V_out`
alone. (Discovered during deterministic case D06.)
Read: `lp/builder/matrix.rs`, `lp/builder/template.rs`.

## Cut pool is append-only; basis matches by slot identity

Cuts are never removed from the LP. Deactivation toggles a cut row's RHS bounds
to the `±f64::INFINITY` sentinel (trivially satisfied); every cut keeps a stable
slot index for the lifetime of the run. The per-iteration template rebake
encodes **only active cuts** (one row per `active_cuts()` entry), not inactive
cuts at sentinel bounds. Warm-start basis reconstruction therefore matches stored
cut rows to current LP rows by **`CutPool` slot identity**, never by row count.
`reconstruct_basis` is the single hot-path entry point — never bypass it.
Read: `cut/pool.rs`, `cut/basis_reconstruct.rs`.

## NCS stochastic availability is a dimensionless factor

Non-controllable-source availability `α_r(ω) ∈ [0, 1]` is dimensionless. The
realized cap is `A_r = max_gen · clamp(mean + std·η, 0, 1)`. The
`non_controllable_models.parquet` stores `(mean, std)` **as factors**, not as MW.
Read: `stochastic/noise.rs` (`transform_ncs_noise`, `compute_effective_eta`).

## Lower-bound evaluation must patch NCS

`evaluate_lower_bound` patches NCS column bounds per opening via
`transform_ncs_noise`, exactly as the forward and backward passes do. Skipping
the patch understates the bound (a real bug caught during D15). The patch inputs
ride on `LbEvalSpec` (`ncs_max_gen`, `ncs_allow_curtailment`).
Read: `lower_bound.rs`.

## Per-stage exchange in the backward pass

`exchange()` is called inside the backward loop, once per stage, not in a
separate pre-pass before the loop.
Read: `backward_pass_state.rs`.

## No EWMA upper bound

`ConvergenceMonitor::upper_bound()` returns the raw per-iteration upper bound —
there is no exponentially-weighted smoothing. Gap closure is immediate for
deterministic cases.
Read: `convergence/convergence.rs`.
