# FPHA fitting & tailrace modeling — CEPEL alignment

**Status:** Draft / design loop (no code changes yet)
**Scope:** `cobre-core`, `cobre-io`, `cobre-sddp`, `book/`, schemas, `CHANGELOG.md`
**Depends on:** `docs/design/hydro-production-model-cleanup.md` (esp. Change 5 —
the convex-hull sampling evaluates `P = ρ_esp · q · h_net`, so `EfficiencyModel`
must already be folded into `ρ_esp`).
**Author:** design loop, 2026-06-04

## Summary

Bring cobre's computed-FPHA pipeline in line with CEPEL/NEWAVE on three fronts:

1. **Per-stage fits** — honor the per-stage-range `fpha_config` the input model
   already accepts, so different stages get different hyperplanes (today the fit
   is stage-independent).
2. **CEPEL convex-hull fitting** — **fully replace** the current tangent-plane
   sampling + greedy plane selection + `kappa` shrink (deleted, not kept as a
   fallback) with CEPEL's procedure: sample the exact production function on a
   grid, build the **3-D convex hull** of the `(V,Q,GH)` cloud via the **`qhull`**
   library (concave envelope), apply the **least-squares `α_FPHA`** correction,
   and add the **spillage/lateral secant** per plane. This becomes the **sole**
   FPHA fitting path; all code supporting the old fitter is removed.
3. **Tailrace families as a table** — a new optional parquet (analogous to
   `hydro_geometry.parquet`) giving tailrace height as a function of outflow and
   **downstream-reservoir level**, interpolated by the downstream plant's stage
   reference volume — i.e., the remanso (backwater) treatment.

The `Q_jus` composition (lateral inflows from posts, upstream defluences,
participation factors `k_jus^{qa}`/`k_jus^{qd}`) is **deferred**.

## Background

### CEPEL/NEWAVE (source: see.cepel.br LIBS manual)

**FPHA construction — 5 steps** (`funcao_producao_hidreletrica.html`):

1. **Window + grid** in the `V×Q` plane: width `Δ_V` around a reference volume
   `V0` (clamped to `[Vmin,Vmax]`), `Q` over its full domain, `NPTV×NPTQ` points.
2. **Exact generation** `GH(V,Q)` at each grid point, with **spillage = 0 and
   lateral flows = 0** → a 3-D cloud `(V,Q,GH)`.
3. **Convex hull** of the cloud (+ a closing point `(V̄,Q̄,0)`) — "a 'menor'
   função côncava cujo gráfico está acima da função original não-côncava" → `M`
   planes. Implemented with the **C++ `qhull`** library; non-concave points fall
   inside the hull and drop out.
4. **`α_FPHA` correction** — `FPHA = α_FPHA · FPHA₀`, where
   `α_FPHA = Σ FPHA₀·FPH / Σ FPHA₀²` minimizes MSE on a `V×Q` grid (spillage = 0).
   Balances the optimistic-in-non-concave / pessimistic-in-concave bias.
5. **Spillage/lateral secant** — per plane, a secant slope `γ_S` in the
   "vazão lateral" axis (aggregates spillage + lateral terms), over `[0,S_max]`
   with `S_max = 2·MLT` (or `2·turbine capacity`), fit by MSE. Only when spillage
   flags the tailrace.

Final row: `GH ≤ α_FPHA·[γ₀^k + γ_V^k·V + γ_Q^k·Q + γ_S^k·(Q_lat − Q̂_lat)]`.
NEWAVE/DECOMP build a **separate FPHA per period** (γ differ by month).

**Tangent-in-volume alternative** (used in DESSEM for weekly/monthly reservoirs):
hull "apenas em relação ao eixo do turbinamento", with the volume effect from an
**analytic derivative**; the reference-volume point is split into two points 1%
of useful volume apart. Motivation stated verbatim: fewer cuts (non-concave 3-D
behavior inflates cut count) and cheaper hull. **This is cobre's current
philosophy.**

**Tailrace families** (`canal_fuga.html`): the piecewise-polynomial tailrace is
provided as **families indexed by `HrefJus`** = "Altura de montante do
reservatório de jusante de referência para a curva de jusante" (the downstream
reservoir's reference forebay level). Each family is piecewise quartics in
`Q_jus` with validity bounds. **Remanso**: plant _i_'s tailrace depends on the
downstream plant `J_i`'s forebay `h_mon(V_j)`; since `V_j` is a decision
variable, the **initial/forecasted (reference) reservoir status** selects/
interpolates the family. Remanso is "desconsiderada, por padrão, quando … é
fornecida através dos Polinômios por partes pois … já levam em consideração esse
efeito … na sua calibração" — i.e., the families _are_ the remanso mechanism.

### cobre today

- **Fitting** (`fpha_fitting.rs`): 3-D tangent-plane sampling over `(v,q,s)` →
  `eliminate_redundant` (keep planes active at some grid point) → `select_planes`
  (greedy, envelope-preserving) → `kappa = min(φ/maxplane) ≤ 1` shrink. A
  _heuristic_ envelope, not a hull; **no `α`-style balancing**; spillage is
  **grid-sampled** (cap `0.5·q_max`), not a secant.
- **Stage-independence** (`hydro_models.rs`): `resolve_production_models` fits
  **once per hydro** at the first study stage and reuses it for all stages, even
  though `find_fpha_config_for_stage` can return different `fpha_config` per
  stage range. Precomputed FPHA already supports per-stage rows
  (`(hydro_id, Some(stage_id))`); computed FPHA does not.
- **Tailrace** (`TailraceModel` on the entity): a single `Polynomial` or
  piecewise-**linear** `Piecewise` in total outflow `q+s`. No families, no
  downstream coupling, no remanso.
- **Reference volume per stage** already exists: `HydroReferenceVolumeFractions`
  - the `reference_volume_hm3` override feed `ρ_eq = ρ_esp·h_eq`. The hook to get
    a per-stage reference volume is therefore already present.

## Goals

- Make computed-FPHA fitting **stage-aware**, honoring per-stage-range
  `fpha_config` (volume window etc.) and producing per-stage hyperplane rows.
- Replace the heuristic envelope with CEPEL's **convex-hull + `α_FPHA` + spillage
  secant** procedure.
- Add an optional **tailrace-families table** keyed by downstream-reservoir
  level, interpolated by the downstream plant's stage reference volume; feed it
  into the production-function sampling so the hull reflects remanso.

## Non-goals (deferred)

- The `Q_jus` composition: lateral inflows from posts, upstream-plant
  defluences, and the participation factors `k_jus^{Q}/k_jus^{S}/k_jus^{qa}/
k_jus^{qd}`. cobre keeps `Q_jus = q + s` (weights ≡ 1). Documented as the next
  increment.
- Travel-time (water-delay) effects on `Q_jus`.
- Precomputed-FPHA changes (this spec only touches the _computed_ path).

---

## Workstream 1 — Stage-dependent FPHA fitting

**Problem.** The fit is hardcoded stage-independent; per-stage-range `fpha_config`
is parsed but ignored by the compute path.

**Change.** In `resolve_production_models_from_artifacts`, fit **once per
production-model `SelectionMode` entry** instead of once per hydro — the same
granularity already used to resolve `equivalent_productivity` (Decision A):

- **`seasonal` model** → one fit **per season**; every stage in that season maps
  to its season's plane set.
- **`stage_ranges` model** → one fit **per range**; every stage in the range maps
  to that range's plane set.

Each entry's fit uses that entry's `fpha_config` (window/discretization) and the
downstream reference level resolved at the same granularity (WS3). Emit per-stage
export rows (`FphaHyperplaneRow.stage_id = Some(stage.id)`) by expanding each
entry's plane set across its stages — the precomputed reader already supports
per-stage rows. (Internally, dedup identical entries so a season/range with
unchanged inputs is fitted once.)

## Workstream 2 — CEPEL convex-hull fitting

**This is the _only_ FPHA fitting path.** The qhull pipeline fully **replaces**
the current tangent-plane fitter — there is no dual path, fallback, or feature
flag. The previous implementation and everything that exists solely to support it
is **deleted** (see "Dead code to remove" below); leaving it in violates the
zero-dead-code rule.

**Change.** Replace `sample_tangent_planes` + `eliminate_redundant` +
`select_planes` + `compute_kappa` with the CEPEL pipeline:

1. Grid over the fit window (`V×Q`); evaluate **exact** `P = ρ_esp·q·h_net` with
   spillage = 0 → the `(V,Q,GH)` point cloud (+ the closing point `(V̄,Q̄,0)`).
2. **3-D convex hull** of the cloud → upper-envelope facets = the concave-envelope
   planes `γ₀ + γ_V·V + γ_Q·Q`.
3. **`α_FPHA`** least-squares correction (replaces `kappa`).
4. **Spillage secant** per plane over `[0, S_max]`, `S_max = 2·MLT` (fallback
   `2·max_turbined`), giving `γ_s` (replaces the grid-sampled spillage axis).

Spillage is **not** a hull dimension (matching CEPEL) — the hull is over
`(V,Q,GH)` and spillage enters only through the step-4 secant.

**Hull implementation (Decision B — resolved: full 3-D via `qhull`).** Use the
C++ **`qhull`** library for the 3-D hull, for exact NEWAVE parity. Integration:

- Prefer an existing Rust binding (`qhull`/`qhull-sys` crate) if its build and
  licensing fit; otherwise a thin vendored FFI. Isolate **all `unsafe`** in one
  module (the `cobre-sddp` `gemm.rs` precedent) and **extend `cobre-sddp`'s
  existing `unsafe_code` override** to cover it (it is already overridden for
  `matrixmultiply`). New build dependency on the `qhull` C library.
- **Determinism is the central risk and a hard requirement.** Mitigations to
  specify and test: canonically **sort the input cloud** before the call;
  canonically **sort the output facets/planes** after; **disable joggle
  (`QJ`)** (randomized perturbation) and instead handle degeneracies via qhull's
  deterministic merged-facet path with fixed options; assert **bit-identical**
  hyperplanes across input orderings and across MPI rank counts in tests. If a
  platform/version of qhull cannot be pinned to deterministic behavior, that is a
  blocking finding to escalate before merge.

**Correctness note.** `kappa ≤ 1` is a worst-case _shrink_ (conservative
overestimate control); `α_FPHA` is a least-squares _balance_ (can be ≷ 1). The
existing validations (`gamma_v ≥ 0`, `gamma_s ≤ 0`, `kappa ∈ (0,1]`) must be
revised — `kappa` is retired; keep the `γ` sign checks, add an `α_FPHA > 0` check.

**Dead code to remove (no longer reachable once qhull is the sole fitter).** The
new pipeline samples exact production **values** and hulls them — it needs no
analytic gradient, tangent plane, redundancy pass, greedy selection, or `kappa`.
Delete (with their tests and any now-orphaned imports):

- `sample_tangent_planes`, `compute_tangent_plane`, `RawHyperplane`'s
  tangent-construction role, and `ProductionFunction::partial_derivatives` (the
  analytic gradient).
- `eliminate_redundant`, `select_planes`, `compute_grid_errors` (the hull yields
  the plane set directly; `max_planes_per_hydro` capping is dropped — if plane
  count later needs trimming, adopt CEPEL's similar-hyperplane merge, tracked
  separately, **not** the old greedy selector).
- `compute_kappa`, the `kappa`/`low_kappa_warning` fields on `FphaFitResult`, the
  `intercept = gamma_0 * kappa` scaling, the `InvalidKappa` error, and the
  kappa-branch of `validate_fitted_planes`.
- Gradient-only helpers with no remaining caller: `ForebayTable::height_derivative`,
  `evaluate_tailrace_derivative`, `evaluate_losses_factor`, and `locate_tailrace`
  usage that only served the derivative.
- The kappa-warning plumbing downstream: `kappa_warnings` on
  `PrepareHydroModelsResult` / `resolve_production_models*` return tuples,
  `HydroModelSummary.kappa_warnings`, and its CLI display — replaced by the
  `α_FPHA` diagnostic (or removed).

Keep what the value path still needs: `ProductionFunction::{new, evaluate}`,
`ForebayTable::{new, height, v_min, v_max}`, `evaluate_tailrace`,
`evaluate_losses`, and the (adapted) grid builder.

## Workstream 3 — Tailrace families as a table

**Data model.** A new optional parquet, e.g. **`system/tailrace_curves.parquet`**,
mirroring the `hydro_geometry.parquet` pattern. Proposed columns:

| column               | meaning                                                                                                              |
| -------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `hydro_id`           | plant whose tailrace this describes                                                                                  |
| `downstream_level_m` | the downstream-reservoir reference level keying the family (`HrefJus`); `NULL`/single value ⇒ no remanso (one curve) |
| `outflow_m3s`        | total outflow sample point (`q+s`)                                                                                   |
| `tailrace_height_m`  | tailrace elevation at this `(level, outflow)`                                                                        |

Within a family, interpolate `tailrace_height` linearly in `outflow`; **between
families, interpolate by the downstream plant's stage reference level** (a
bilinear lookup, exactly how `ForebayTable` interpolates `hydro_geometry`).

**Downstream level resolution.** For plant _i_ with `downstream_id = J_i`: take
`J_i`'s **stage reference volume** (`reference_volume_fractions` / override, the
same value used for `ρ_eq`) → `J_i`'s forebay level via _its_ `hydro_geometry`
table → select/interpolate plant _i_'s tailrace family at that level. No
`downstream_id` (run-of-river / last plant) ⇒ single curve, no remanso.

**Family key (Decision C — resolved: level).** The table is keyed by downstream
**level (m)** (matches NEWAVE `HrefJus`, physical), not downstream volume.

**Representation (Decision C — resolved: standalone parquet).** This is a
**standalone optional parquet**, consistent with `hydro_geometry` /
`fpha_hyperplanes` — _not_ a new `TailraceModel` enum variant. The entity keeps
`Polynomial`/`Piecewise` for the simple, no-remanso case; the table is opted into
via the production-model config and gates the remanso path.

**Feeding the fit.** The table changes only **how `h_tail(q+s)` is evaluated**
inside the production function during sampling (WS2 step 1). The hull, `α_FPHA`,
and secant are unchanged — exactly your point that this "does not interfere with
the fitting approach; it only changes the sampled point cloud."

---

## How the pieces compose (computed-FPHA, revised)

```
for each hydro h with a computed-FPHA config:
  for each SelectionMode entry e of h (one per season, or one per range) (WS1):
      window     ← e.fpha_config.fitting_window
      h_tail(·)  ← tailrace table family @ downstream(h) ref level for e   (WS3)
                   (or the entity TailraceModel if no table)
      cloud      ← grid over window; P = ρ_esp·q·h_net(v,q,0)              (WS2.1)
      planes     ← qhull 3-D convex hull(cloud + closing point)           (WS2.2)
      α          ← least-squares FPHA₀ vs FPH on V×Q grid                  (WS2.3)
      γ_s        ← spillage secant per plane over [0, S_max]               (WS2.4)
      emit planes (scaled by α) across e's stages as FphaHyperplaneRow
```

## Resolved decisions

| #     | Decision                      | Resolution                                                                                                                         |
| ----- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **A** | Stage-fit granularity         | **Per `SelectionMode` entry** — one fit per season (`seasonal`) or per range (`stage_ranges`), mirroring `equivalent_productivity` |
| **B** | Hull implementation           | **Full 3-D hull via `qhull` FFI** (NEWAVE parity); isolate `unsafe`, extend `cobre-sddp` override, canonicalize for determinism    |
| **C** | Tailrace-table family key     | Downstream **level (m)** (matches `HrefJus`)                                                                                       |
| **D** | Tailrace-table representation | **Standalone optional parquet** (consistent with other extension tables; not a `TailraceModel` variant)                            |

## Dependencies & sequencing

1. **Cleanup spec Change 5 first** (`ρ_esp` unification) — WS2 samples
   `P = ρ_esp·q·h_net`. WS2 also further rewrites the same `fpha_fitting.rs`
   touched by Change 5, so land Change 5, then this.
2. **WS2 (hull fitting)** and **WS1 (stage-awareness)** can land together — both
   restructure `resolve_production_models` + `fpha_fitting.rs`.
3. **WS3 (tailrace table)** builds on WS1 (needs the per-stage downstream
   reference level) and WS2 (feeds the sampler).

## Validation & testing

- **Determinism (gating):** bit-identical hyperplanes regardless of hydro/stage
  input ordering and across MPI rank counts. Includes a dedicated test that
  shuffles the input cloud and asserts identical `qhull` output — the
  canonical-sort-in / canonical-sort-out contract. Pin the qhull version; if it
  cannot be made deterministic, escalate (blocking).
- **Re-bless:** every computed-FPHA deterministic case and the slow-tests
  plane-selection suite — γ values change (qhull vs. heuristic, `α` vs. `kappa`,
  secant vs. grid spillage).
- **Unit:** qhull hull on known clouds (incl. degenerate/coplanar inputs);
  `α_FPHA` closed form; secant slope on a synthetic tailrace; tailrace-table
  bilinear interpolation (incl. between-family interpolation by downstream level)
  and the no-downstream single-curve path; per-season vs. per-range fits differ
  when their windows / downstream levels differ.
- **Cross-check (optional):** compare cobre hyperplanes vs. NEWAVE for a shared
  case to bound the approximation gap.

## Out of scope / future

- `Q_jus` composition (posts, upstream defluences, participation factors) and
  water travel-time — the next increment after this lands.
- A genuine head-dependent (hill-chart) `ρ(Q,h)` instead of constant `ρ_esp`.

## Next step

Decisions A–D are settled (see table). Open the **qhull determinism + build
integration** as the first investigation spike when planning, since it gates WS2
and adds a C-library build dependency. Then generate the implementation plan into
`plans/` via `/plan`. Keep this spec independent of the hydro-production cleanup
spec; they share `fpha_fitting.rs` but are sequenced (cleanup → this).
