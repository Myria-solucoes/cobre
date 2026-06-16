# FPHA fitting — CEPEL reference audit plan

**Status:** Audit plan (not yet executed). Reference-first validation of the
computed-FPHA fitting pipeline before any code change.
**Scope:** `crates/cobre-sddp/src/production/fpha_fitting/` — grid → convex hull →
`α_FPHA` → lateral/spillage secant (`γ_S`), under the **`Qlat = 0` (spillage-only)**
case that the production decks use.
**Companion:** `docs/design/fpha-spillage-secant-anchor-gap.md` documents the
suspected `γ_S` representative-point inversion (argmax vs argmin). This plan
**audits that diagnosis against the authoritative reference rather than adopting
it on faith.**

## Why this audit exists

A cobre-bridge investigation reported that cobre over-steepens the spillage
secant on high-head planes (≈13× vs NEWAVE on TUCURUI) and concentrates it on ~2
planes, and proposed that `representative_operating_point` (`secant.rs`) anchors
each plane at the **argmax** (loosest) envelope point instead of the **argmin**
(active/binding) tangency. That argmin reasoning is _internally_ consistent with
cobre's own min-envelope contract in `compute_alpha_fpha` (`alpha.rs`) — but
internal consistency is not proof of conformance to the CEPEL method. **The CEPEL
reference is the arbiter.** This plan reads the reference section by section and
checks each fitting stage against it, so the eventual fix is grounded in the
documented algorithm, not in one agent's interpretation.

A second, possibly deeper question surfaced while mapping the code: cobre's
`γ_S` axis is described in `secant.rs` as the **lateral-flow** (`Qlat`) axis, yet
the NEWAVE coefficient the comparison used (`Coef. Qver`) is the **spillage**
(`Qver`) coefficient. The gap doc treated `γ_S` and spillage as the same thing.
Whether cobre's `γ_S` models lateral flow, spillage, or both — and whether that
matches the reference under `Qlat = 0` — must be resolved **first**, because it
can change the entire diagnosis (wrong anchor vs wrong axis vs wrong estimator).

## Authoritative sources (read in this order)

1. **CEPEL FPHA reference page** (primary):
   `https://see.cepel.br/manual/libs/latest/usinas_hidreletricas/geracao_energia/funcao_producao_hidreletrica.html`
   Read carefully, section by section. Fetch with WebFetch.
2. **Methodology repo** `~/git/cobre-docs` — cross-check the FPHA fitting + exact
   tailrace & lateral-flow design pages (the "definitive FPHA fitting … CEPEL
   alignment" material).
3. **Matched decks** for empirical comparison:
   `~/git/cobre-bridge/example/{newave,cobre}_rodada_2000_sem_pos_fpha` — NEWAVE
   `fpha/fpha_cortes.csv` (`Coef. Qver`, `Coef. Qlat = 0`) vs cobre
   `output/hydro_models/fpha_hyperplanes.parquet`.

## Code under audit (verified symbols)

| Stage           | File                                            | Key symbols                                                                                |
| --------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Grid            | `fpha_fitting/grid.rs`                          | `build_grid` (single source for the V×Q grid, iterated by hull/alpha/secant)               |
| Convex hull     | `fpha_fitting/hull_fit.rs`                      | `build_cloud`, `facet_to_plane` (sets `gamma_s = 0.0`), `RawPlane`                         |
| Hull FFI        | `fpha_fitting/{geometry,hull}/…`, `hull/ffi.rs` | qhull 3-D FFI + `Hyperplane3d`                                                             |
| α factor        | `fpha_fitting/alpha.rs`                         | `compute_alpha_fpha`, `scale_plane_affine`                                                 |
| Secant `γ_S`    | `fpha_fitting/secant.rs`                        | `resolve_s_max`, `representative_operating_point`, `fit_gamma_s`, `fit_gamma_s_for_planes` |
| Plane reduction | `fpha_fitting/reduction.rs`                     | `reduce_planes_angle`, `reduce_planes_distance`                                            |
| Validation      | `fpha_fitting/selection.rs`                     | `validate_fitted_planes` (`gamma_s ≤ 1e-10`, `alpha > 0`)                                  |
| Orchestration   | `fpha_fitting/mod.rs`                           | `fit_planes` flow: hull → α → scale → secant → validate → reduce                           |

## Stage-by-stage audit

### Stage 0 — Terminology & axis semantics (resolve FIRST)

- [ ] From the reference, pin the exact definition and units of each FPHA
      argument: stored volume `V`, turbined flow `Q`, **spillage `Qver`**, and
      **lateral flow `Qlat`**. Note which the production function `phi` depends on
      and the sign each contributes.
- [ ] Determine what cobre's `γ_S` / "s" axis actually represents
      (`secant.rs` calls it lateral flow; `RawPlane::evaluate` uses
      `gamma_s * s`). Decide whether, under `Qlat = 0`, cobre's `γ_S` is intended
      to carry the **spillage** effect (matching NEWAVE `Coef. Qver`) or is a
      distinct lateral-flow term that should be **absent** in the `Qver`-only
      comparison.
- [ ] **Decision gate:** if `γ_S` is lateral-flow and `Qver` is a separate axis
      cobre does not fit, the entire "γ_S too steep vs Coef. Qver" comparison is
      an apples-to-oranges error and the gap doc's conclusion must be revised.
      Record the finding before proceeding.

### Stage 1 — Grid construction (`grid.rs::build_grid`)

- [ ] Reference: how is the (volume, turbined-flow) sampling grid defined —
      endpoints, count, spacing (linear?), and the zero-flow / installed-capacity
      bounds?
- [ ] Compare `build_grid`: are V and Q sampled over the correct ranges
      (`FittingBounds`), with the zero-flow floor and capacity ceiling the
      reference prescribes? Confirm the grid is the single source iterated by the
      hull cloud, the α regression, and the secant scan (no divergent grids).
- [ ] Confirm declaration-order/bit-determinism of the closed-form grid.

### Stage 2 — Convex hull / qhull (`hull_fit.rs`, `hull/ffi.rs`)

- [ ] Reference: is the production cloud over-approximated by an **upper** convex
      hull, and is the LP-consumed surface the **min** (lower envelope of the
      upper facets) — i.e. a concave over-approximation `g ≤ plane_k`?
- [ ] Compare `build_cloud` (cloud points from `phi` over the grid at `s = 0`
      with the zero-flow floor / capacity clip) and `facet_to_plane` (upper-facet
      → `RawPlane`, `gamma_s = 0` at this stage). Confirm we pass the right point
      set to qhull and keep the correct (upper) facets.
- [ ] Confirm signs: `upper_envelope_planes_have_valid_coefficient_signs` and the
      outer-approximation property hold per the reference.

### Stage 3 — `α_FPHA` least-squares (`alpha.rs::compute_alpha_fpha`)

- [ ] Reference: confirm `α` is a least-squares correction minimizing the gap
      between the hull surface and the sampled production over the grid, and over
      which axis/points (the reference's α definition).
- [ ] Compare `compute_alpha_fpha`: verify it regresses against the **min
      envelope** the LP consumes (the stated contract), evaluated at `s = 0`, and
      that `scale_plane_affine` scales `gamma_0/v/q/s` uniformly (so `α` never
      moves the secant anchor). Confirm the closed-form matches the reference's
      LS normal equations (cf. `closed_form_alpha_matches_hand_computed_ratio`).
- [ ] Confirm `α` uses the same grid as Stage 1 and is spillage-zero-only
      (`regression_is_spillage_zero_only`).

### Stage 4 — Secant `γ_S` (`secant.rs`) — the crux

- [ ] Reference: how is the spillage (and/or lateral-flow) marginal `γ_S`
      defined? Is it (a) a least-squares secant of `phi` vs `Qver` over a flow
      range, (b) anchored at a specific representative `(V, Q)`, and (c) the same
      slope for all planes or per-plane? Capture the reference's **representative
      flow** definition exactly.
- [ ] Compare `fit_gamma_s`: confirm it already performs a **least-squares**
      secant over an evenly-spaced sample `∈ [0, S_max]` (it does) — so the
      user's "secant should be least-squares" is satisfied; the open questions are
      the **sample axis** (lateral flow vs spillage, Stage 0) and the **anchor**.
- [ ] Compare `representative_operating_point`: it currently keeps the grid point
      where the plane is the **max** over planes (loosest). Check against the
      reference's representative point. If the reference's tangency/active region
      is the **argmin** (as `alpha.rs`'s min-envelope contract implies), this is
      the inversion the gap doc flags — confirm or refute from the reference, not
      from internal consistency alone.
- [ ] Compare `resolve_s_max`: reference basis for the flow upper bound. cobre
      uses `S_max = 2 × long-term mean inflow` with a `2 × max_turbined` fallback.
      Confirm the reference's `S_max`/range and the history-less fallback.
- [ ] Confirm the sign clamp (`γ_S ≤ 0`, enforced in `validate_fitted_planes`)
      matches the reference (spillage must not increase production).

### Stage 5 — Plane reduction & validation (context, not suspected)

- [ ] Confirm `reduce_planes_angle` / `reduce_planes_distance` (similar-hyperplane
      simplification) and `validate_fitted_planes` do not themselves distort
      `γ_S` distribution across planes (the gap doc's "concentrated on ~2 planes"
      symptom could be amplified by reduction). Verify reduction runs _after_ the
      secant fit and preserves the per-plane `γ_S`.

## Empirical cross-check (after the reference read)

- [ ] Re-derive, by hand from the reference, the expected `γ_S` for TUCURUI's
      high-head plane and compare to both NEWAVE (`Coef. Qver = -0.00243`,
      gentlest) and cobre (`-0.0326`, steepest). Confirm whether the inversion
      reproduces and whether argmin selection would match NEWAVE's graduated
      spread across (nearly) all planes.
- [ ] Build a small multi-plane concave fixture and compute `γ_S` under both
      argmax (current) and argmin (proposed) anchors; check which matches the
      reference-derived expectation.

## Decision points to record (do not pre-commit)

1. Is `γ_S` lateral flow or spillage, and does the NEWAVE `Coef. Qver`
   comparison even apply under `Qlat = 0`? (Stage 0 — gates everything.)
2. Is the defect a **wrong anchor** (argmax→argmin), a **wrong axis**
   (lateral vs spillage), a **wrong estimator shape**, or a comparison artifact?
3. Is any change parity-affecting? (Yes if it touches `γ_S`/`α`/hull → D07
   computed-FPHA baseline, HiGHS **and** CLP — see the gap doc's constraints.)

## Execution constraints (when the fix eventually lands)

- **Parity-affecting** changes to `γ_S`/`α`/hull require a **dedicated,
  re-blessed** change (re-bless D07 on both backends), a **by-plane** regression
  test (assert the high-`γ_q` plane gets the gentle `γ_S`), and a convergence +
  outer-approximation sanity check. **Not** folded into an additive feature.
- Both LP backends green; declaration-order bit-determinism preserved.
- Dispatch: `sddp-specialist` (formulation/conformance) → `hpc-rust-developer`
  (implementation) per the multi-domain chain.

## References

- CEPEL FPHA page (URL above); `~/git/cobre-docs` FPHA methodology pages.
- `docs/design/fpha-spillage-secant-anchor-gap.md` (the diagnosis under audit).
- `fpha_fitting/{grid,hull_fit,alpha,secant,reduction,selection,mod}.rs`.
- Matched decks under `~/git/cobre-bridge/example/`.
