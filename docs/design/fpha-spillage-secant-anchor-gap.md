# FPHA spillage secant (`γ_S`) is anchored at the wrong representative flow

**Status:** Gap / bug report (not yet fixed) — parity-affecting, needs a dedicated
re-blessed change
**Scope:** `crates/cobre-sddp/src/production/fpha_fitting/secant.rs`
**Author:** investigation 2026-06-15, validated against real NEWAVE output
(`rodada_2000_sem_pos_fpha`)

## Summary

cobre's computed-FPHA fit charges the **spillage secant `γ_S` (`Coef. Qver`) to
the wrong planes**. Each plane's `γ_S` is anchored at the grid point where the
plane is the **loosest** (maximum) upper bound — which, for a steep high-head
plane, is the maximum-turbined-flow corner — instead of the plane's own
**binding (tangency)** region. Because `|γ_S| ≈ ρ·q_rep·(d h_tail / d q_out)`
scales with the anchor's turbined flow `q_rep`, this:

1. **over-steepens** `γ_S` on the high-head plane (TUCURUI: −0.0326 vs the NEWAVE
   reference −0.0024, **≈13×** too steep), and
2. **concentrates** `γ_S` on ~2 planes and emits **0** on the rest, where NEWAVE
   spreads a graduated `γ_S` across (nearly) all planes.

Both symptoms share one root cause (below). The inflated `γ_S` forces the LP to
over-turbine for the same MW precisely in wet / full-reservoir stages — the
"realized productivity drops more steeply in some stages" behaviour.

## The defect

`representative_operating_point` (in `secant.rs`) selects each plane's anchor by
scanning the spill = 0 fit grid and keeping the point where the plane is the
**maximum** over all planes, then the point of largest generation among those:

```rust
let this_val = plane.evaluate(v, q, 0.0);
let max_val  = planes.iter().map(|p| p.evaluate(v, q, 0.0)).fold(f64::NEG_INFINITY, f64::max);
if max_val - this_val <= 1e-8 {            // plane is the MAX (loosest) bound here
    let generation = pf.evaluate(v, q, 0.0);
    if best.is_none_or(|(_, _, best_gh)| generation > best_gh) { best = Some((v, q, generation)); }
}
```

This is the wrong envelope. FPHA is a **concave over-approximation** applied by the
LP as `g ≤ plane_k` for every plane, so the **binding / active** plane at any
`(v, q)` is the **minimum** over planes — the tangency facet. This is a stated
contract in `compute_alpha_fpha` (`alpha.rs`), which regresses the **min**
envelope and explicitly names regressing the **max** envelope "the
wrong-but-compiling alternative." `representative_operating_point` uses exactly
that max envelope, and even contradicts its own doc comment ("where this plane is
the _active_ (tightest) upper bound … the operating region the plane actually
governs" — that region is the `argmin`, not the `argmax`).

### Why both symptoms fall out of the one inversion

- **Concentration / zeros.** A plane gets a representative point only where it is
  the `argmax`; only the ~2 extreme planes are ever the max on the grid, so every
  other plane returns `None` and is emitted with `γ_S = 0`.
- **Over-steepening.** The steep high-`γ_q` plane is the `argmax` at the high-Q
  corner, so its secant is anchored at `q_rep ≈ q_max` and `|γ_S|` is inflated.

The correct `argmin` (active / tangency) selection gives every plane its own
governing region → a representative point each → a graduated `γ_S` across all
planes at the right magnitudes, matching NEWAVE. **A single fix addresses both.**

## Empirical evidence

Compared the FPHA cuts of two matched decks:

- NEWAVE reference: `~/git/cobre-bridge/example/newave_rodada_2000_sem_pos_fpha/fpha/fpha_cortes.csv`
  (`Coef. Qver` = `γ_S`; `Coef. Qlat` = 0 throughout — the no-lateral case).
- cobre output: `~/git/cobre-bridge/example/cobre_rodada_2000_sem_pos_fpha/output/hydro_models/fpha_hyperplanes.parquet`.

**TUCURUI** (NEWAVE `USIH=275`, cobre `hydro_id=143`), planes sorted by `γ_q`
(high `γ_q` ⇒ high head ⇒ low flow):

|                                        | low-head plane            | …                  | high-head plane                       |
| -------------------------------------- | ------------------------- | ------------------ | ------------------------------------- |
| NEWAVE — 11 planes, **all 11 nonzero** | `γ_q=0.350 → γ_S=−0.0265` | …                  | `γ_q=0.619 → γ_S=−0.00243` (gentlest) |
| cobre — 7 planes, **2 nonzero**        | `γ_q=0.361 → γ_S=−0.0163` | 5 planes → `γ_S=0` | `γ_q=0.612 → γ_S=−0.0326` (steepest)  |

NEWAVE gives its high-head plane the **gentlest** penalty; cobre gives the same
plane the **steepest** — the inverted anchor, reproduced to the digit.

**Systematic across plants** (NEWAVE planes/nonzero, steepest `γ_S` vs cobre):

| plant        | NEWAVE         | cobre               |
| ------------ | -------------- | ------------------- |
| A. VERMELHA  | 11/11, −0.0097 | 3/2, −0.0110        |
| A.S. LIMA    | 4/4, −0.0047   | 2/2, −0.0099 (2.1×) |
| B. COQUEIROS | 4/4, −0.0040   | 5/2, −0.0098 (2.4×) |
| CACH.DOURADA | 4/4, −0.0172   | 3/2, −0.0269 (1.6×) |
| BELO MONTE   | 4/4, −0.0197   | 5/2, −0.0242        |

Two invariants hold everywhere: NEWAVE spreads `γ_S` across (nearly) all planes;
cobre concentrates it on ~2 and over-steepens the survivor (1.2–2.4× typical,
≈13× for high-head TUCURUI).

## What is already correct (not the problem)

- **`S_max = 2·MLT`** (`resolve_s_max`), with the `2·max_turbined` fallback when
  there is no inflow history.
- **`α` handling.** `γ_S` is fit from `pf.evaluate(...)` _after_ `α`-scaling, so
  `α` corrects only `γ_0 + γ_V·V + γ_Q·Q`, never `γ_S·S` (and uniform `α`-scaling
  does not move the anchor point).
- The secant **slope estimator** itself (free-intercept OLS over `[0, S_max]`) vs
  DECOMP's through-origin secant to a per-point `S_ref` is a second-order
  difference — not the cause. Leave it for a separate increment.

## Proposed fix

In `representative_operating_point`, select the anchor where the plane is the
**active (minimum / binding)** bound — its tangency region — not the maximum.
Concretely, find the grid point(s) where `plane.evaluate(v,q,0)` is within
tolerance of the `min` over planes (and ideally where it touches the exact
production surface `pf.evaluate(v,q,0)`), and pick the representative `(v, q)`
there. Keep the determinism discipline already present (closed-form
`build_grid`, fixed scan order, strict-`>` tie-break) and the `γ_S ≤ 0` sign
clamp in `fit_gamma_s`.

## Constraints for the fix

- **Parity-affecting.** This changes every computed-FPHA `γ_S` → the FPHA LP row
  → duals → the D07 computed-FPHA parity baseline (HiGHS **and** CLP). It must be
  a **dedicated, re-blessed change**, not folded into an additive feature; pair
  the γ-change with a convergence + outer-approximation sanity check and re-bless
  both baselines.
- **Regression test by plane.** Add a multi-plane concave fixture asserting the
  high-`γ_q` plane receives the **gentle** `γ_S` (and a TUCURUI-style numeric
  check), so the inversion cannot silently return — the current single-plane
  secant tests do not exercise the anchor.
- **Both backends green**; declaration-order bit-determinism preserved.

## Sequencing

This belongs with the FPHA fitting work (`plans/fpha-tailrace-modeling/`) or as a
standalone fix — **not** in `setup-phase-observability` (which is additive and
re-blesses nothing). Given the ≈13× error on high-head plants in wet stages, it
is higher priority than the remaining observability polish.

## References

- DECOMP user manual §5.4 "Modelagem do Vertimento"; CEPEL FPHA page
  (`geracao_energia/funcao_producao_hidreletrica.html`).
- `secant.rs::representative_operating_point` / `fit_gamma_s`;
  `alpha.rs::compute_alpha_fpha` (the min-envelope contract this violates).
- Matched decks: `~/git/cobre-bridge/example/{newave,cobre}_rodada_2000_sem_pos_fpha`.
