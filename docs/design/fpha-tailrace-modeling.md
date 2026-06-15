# FPHA fitting, exact tailrace & lateral-flow modeling — CEPEL alignment

**Status:** Implemented — WS1–WS3 + WS5 shipped (per-stage convex-hull fit with
`α_FPHA` + lateral secant, exact piecewise-quartic tailrace with `HrefJus`
families, mandatory similar-hyperplane reduction). WS4b/c (the explicit
external-lateral `Qlat` coefficient) deferred per Decision J.
**Scope:** `cobre-core`, `cobre-io`, `cobre-sddp`, `cobre-cli`, `cobre-python`,
`book/`, schemas, `CHANGELOG.md`
**Author:** design loop, 2026-06-04; **revised 2026-06-13** — verified line-by-line
against the CEPEL LIBS manual (all equations transcribed from the source pages,
not paraphrased); scope decisions E–H resolved. **Revised 2026-06-14** — validated
against real NEWAVE FPHA output (a reference case with 151 plants over 64 periods)
and added decisions J (defer the explicit external-lateral `Qlat` coefficient to a
future version) and K (run-of-river / single-volume `NPTV = 1` hull handling). See
"Validation against reference FPHA output".

**Source of truth (CEPEL LIBS manual, read in full):**

- FPHA construction & simplification — `geracao_energia/funcao_producao_hidreletrica.html`
- Tailrace (canal de fuga) & lateral flow — `componentes_usinas/canal_fuga.html`
- Hydric balance — `restricoes_operativas/balanco_hidrico_usinas_hidreletricas.html`

> These are external CEPEL pages, not repo-relative paths. The equations below
> are the authoritative reference for this work; the methodology repo
> (`~/git/cobre-docs/`) is the place to mirror them into committed theory docs.

## Summary

Bring cobre's computed-FPHA pipeline into line with CEPEL/NEWAVE/DECOMP on five
fronts:

1. **Per-stage fits** (WS1) — honor the per-stage-range `fpha_config` the input
   already accepts; today the fit is stage-independent.
2. **qhull hull + `α_FPHA` + proper lateral secant** (WS2) — fully replace the
   tangent-plane sampling + greedy selection + `kappa` shrink with CEPEL's exact
   procedure (3-D convex hull, least-squares `α_FPHA`, per-plane MSE secant on the
   **lateral-flow** axis).
3. **Exact tailrace** (WS3) — piecewise **quartics** in the downstream flow
   `Q_jus` (not piecewise-linear), with **`HrefJus` families** that implement the
   **remanso** (backwater) coupling to the downstream reservoir's level.
4. **General `Q_jus` lateral-flow composition** (WS4) — own turbine/spill,
   upstream defluences, and post incremental inflows with participation factors,
   **designed so smart defaults collapse to `Q_jus = Q + S`** — simple plants need
   zero new input and behave exactly as today. **v1 models only the own-spill
   secant** (`γ_S` = own spillage = the reference output's `Qver`); the explicit
   **external-lateral `Qlat`** coefficient (upstream defluences + post inflows) is
   **deferred** (Decision J) — in the reference case `Qlat` is zero for every plant,
   so v1 emits no `Qlat` term.
5. **Mandatory similar-hyperplane simplification** (WS5) — a post-fit pass that
   merges near-parallel / near-coincident planes, implementing **both** CEPEL
   methods (normal-vector **angle** and MSE **distance**). This is a hard
   performance requirement, not an optional extra.

**Constant productivity for v1.** The hull is sampled with a **constant**
`ρ_esp` (`P = ρ_esp · q · h_net`). Variable productivity `ρ(Q, h_liq)` and
flow-dependent losses `h_PerdH(Q)` (CEPEL's GAM-grid path) are a clearly-scoped
**future increment** (see Out of scope). This spec carries **no dependency** on
any separate ρ-unification design.

## Non-goals (deferred, separate tracks)

- **Hydric-balance completion** — travel time `τ_{i,j}` (`M_tv`) and
  reversible/pumped plants (`M_eb`, elevatórias) are **not** in this effort. This
  work consumes cobre's existing **same-stage** upstream turbine + spill columns
  as the defluence `Q_def = Q + S` that feeds `Q_jus`. The full balance alignment
  is its own design (see "Relationship to the hydric balance").
- **Variable productivity / variable losses** — `ρ(Q, h_liq)` and `h_PerdH(Q)`
  GAM grids (CEPEL cards `HIDRELETRICA-PRODUTIBILIDADE-ESPECIFICA-GRADE` /
  `HIDRELETRICA-PERDA-HIDRAULICA-GRADE`). v1 keeps constant `ρ_esp` and the
  existing constant/factor losses.
- **Explicit external-lateral `Qlat` coefficient** (Decision J) — the reference
  FPHA cut carries two distinct lateral coefficients: `Qver` (the plant's OWN
  spillage) and `Qlat` (EXTERNAL lateral = upstream defluences + post incremental
  inflows). v1 fits and wires **only the own-spill secant** (`γ_S` = `Qver`); the
  external `Qlat` term (the WS4b/c work) is **deferred to a future version**, gated
  on detecting that a plant actually has lateral-inflow contributions. In the
  reference case `Qlat` is zero for all 151 plants (no lateral-flow plants), so the
  common case emits no `Qlat` term — and v1 needs no new LP term for it.
- **Tangent-in-volume FPHA variant** (DESSEM weekly/monthly reservoirs) — cobre
  uses the full 3-D hull for genuinely volume-varying plants; the
  tangent-in-volume construction is **promoted from out-of-scope to the candidate
  mechanism for the run-of-river / single-volume case** (Decision K, see WS2): two
  volume points 1% of useful volume apart keep the 3-D hull non-degenerate while
  yielding `γ_V ≈ 0`. It is otherwise not used for multi-volume reservoirs.
- **FPHAD (dynamic/iterative FPHA)** — CEPEL's iterative cut introduction is
  conceptually cobre's dynamic cut selection (DCS); no new work here.

---

## Part A — The CEPEL reference (authoritative, transcribed)

### A1. Exact production function (FPH)

For plant `i` with total turbine flow `Q_i = Σ_j q_j`:

```
GH_i = ρ_i(Q_i, h_liq) · Q_i · [ h_mon(V) − h_jus(Q_jus) − h_PerdH(Q) ]
```

- `h_mon(V)` — forebay (cota de montante), nonlinear in storage (cota-volume
  polynomial). **In the LP, `V` is the _average_ storage** `(V_in + V_out)/2`;
  for PDDE/SDDP the dual × the **initial-volume** coefficient enters the water
  value (Benders cut). _cobre already does both — keep._
- `h_jus(Q_jus)` — tailrace (cota de jusante), function of the **downstream flow**
  `Q_jus` (WS4), not just `q + s`.
- `h_PerdH` — head losses: factor `k_PerdH` (p.u. of gross head), constant metres,
  **or** `f_PerdH(Q)`. v1 keeps factor/constant; `f_PerdH(Q)` is future.
- `ρ` — specific productivity `[MW / ((m³/s)·m)]`. v1 constant; `f_ρ(Q, h_liq)`
  is future.
- `h_liq = h_bruta − h_PerdH`, `h_bruta = h_mon(V) − h_jus(Q_jus)`.

### A2. FPHA construction — five steps (exact)

1. **Window + grid.** Volume window of width `Δ_V` about reference `V_0`,
   **clamped to `[V_min, V_max]`**; `Q` over its full domain; `NPTV × NPTQ` grid
   in the `V × Q` plane. `Δ_V`, `NPTV`, `NPTQ` are model/period-specific and may
   be overridden per plant/period.
2. **Exact generation** `G̃H(Ṽ_i, Q̃_j) = FPH(Ṽ_i, Q̃_j)` at each grid point,
   **with spillage = 0 and lateral flows = 0** → 3-D cloud `(Ṽ, Q̃, G̃H)`.
3. **Convex hull** of the cloud **plus the closing point `(V̄, Q̄, 0)`** (to close
   the region under the curve) → the **smallest concave function whose graph lies
   above** the original non-concave FPH → `M` planes. Implemented with the C++
   **qhull** library. Points in non-concave regions fall inside the hull and drop
   out; raising `NPTV/NPTQ` there does not help.
4. **`α_FPHA` correction.** The raw hull gives `FPHA_0` (optimistic where FPH is
   non-concave, pessimistic where concave). Multiply by a scalar `α_FPHA` to
   balance the bias, minimizing MSE on an `m × n` `V × Q` grid **at spillage = 0**:

   ```
   α_FPHA = ( Σ_i Σ_j  FPHA_0(V_i,Q_j) · FPH(V_i,Q_j) )
            / ( Σ_i Σ_j  FPHA_0(V_i,Q_j)² )

   FPHA(V,Q) = α_FPHA · FPHA_0(V,Q)
   ```

   **The regression must use spillage = 0 only** (an explicit CEPEL note): adding
   the spillage axis would pull `α` toward the larger-deviation spillage region
   and degrade the no-spill region that dominates operation. Using the fit
   breakpoints as the regression grid is acceptable (they are natural optima).

5. **Lateral-flow secant.** Per plane `k`, a secant slope `γ_S^k` on an extra FPHA
   axis — the **lateral flow `Q_lat`** (WS4), aggregating everything that raises
   the tailrace but does **not** drive generation. Fit by MSE over
   `Q_lat ∈ [0, S_max]`, with **`S_max = 2·MLT`** (long-term mean flow) **or
   `2 × turbine capacity`** if MLT = 0. Applies **only** to plants where spillage
   influences the tailrace (a registry flag). The reference offset `Q̂_lat` is
   **0** for the spill-only case, or the **mean of post incremental inflows** when
   posts contribute (WS4).

**Final FPHA row** (`NFP_i` planes per plant `i`):

```
GH_i ≤ α_FPHA · [ γ_0^k + γ_V^k · V_i + γ_Q^k · Q_i + γ_S^k · (Q_lat,i − Q̂_lat,i) ]
```

Spillage is **not** a hull dimension (the hull is over `(V, Q, GH)` with
spillage = 0); it enters only through the step-5 secant. The LP includes a
heavily-penalized **violation slack** `f_FPHA` so subproblems stay feasible.

### A3. Tailrace (canal de fuga) — exact

```
h_jus(Q_jus) = a_cf0 + a_cf1·Q_jus + a_cf2·Q_jus² + a_cf3·Q_jus³ + a_cf4·Q_jus⁴   (single quartic)
```

Or **piecewise quartics** — the domain of `Q_jus` is split into segments `k`,
each a degree-4 polynomial valid on `[Q_jus_inf^k, Q_jus_sup^k]`, with C⁰
(preferably C¹) continuity:

```
h_jus(Q_jus) = h_jus^{ijus(Q_jus)}(Q_jus),
h_jus^k = a_cf0^k + a_cf1^k·Q_jus + … + a_cf4^k·Q_jus⁴   on  [Q_jus_inf^k, Q_jus_sup^k]
```

**Remanso (backwater).** When plants are close, plant `i`'s tailrace depends on
the **downstream plant `J_i`'s forebay** `h_mon(V_j)`. Since `V_j` is a decision
variable, the **initial/forecasted reservoir status** selects/interpolates the
tailrace curve **family**, each family keyed by **`HrefJus`** = the downstream
reservoir's reference forebay level (m). Remanso is **off by default when
piecewise polynomials are supplied** — per ONS, those polynomials already embed
the backwater effect in their calibration. So: **families are the remanso
mechanism.**

### A4. Lateral flow / downstream flow `Q_jus` — exact

```
Q_jus,i = k_jus^Q · Q_i
        + k_jus^S · S_i
        + Σ_{j∈Ω^qa_i}  k_jus^{qa}_{i,j} · Qa_j          (post/gauge incremental inflows)
        + Σ_{j∈Ω^qd_i}  k_jus^{qd}_{i,j} · Q_def,j        (upstream defluences, Q_def,j = Q_j + S_j)
```

- `Qa_j` are **data** (incremental inflow at a gauge, per stage/scenario) — they
  shift the constraint **RHS**.
- `Q_def,j = Q_j + S_j` are **decision variables** — they enter the constraint
  **matrix**, coupling the FPHA row to upstream plants' columns.
- Travel time on the `Q_def,j` terms is **ignored** in this effort (Non-goal).
- "For the vast majority of plants, `Q_jus = Q` or `Q + S`."

**The FPHA secant axis is `Q_lat`, _not_ `Q_jus`.** The own-turbine term is folded
into `γ_Q` (and any `k_jus^Q ≠ 1` is baked into `γ_Q`), so the secant axis is:

```
Q_lat,i = k_jus^S · S_i + Σ k_jus^{qa}·Qa_j + Σ k_jus^{qd}·Q_def,j   ( = Q_jus,i − k_jus^Q·Q_i )
```

### A5. Similar-hyperplane simplification — the mandatory performance pass (exact)

A **post-processing** pass over the fitted planes of each plant (does **not**
alter construction; CEPEL reports negligible overhead). Goal: remove redundant
near-parallel/near-coincident cuts to shrink the LP. Two **mutually exclusive**
methods, one tolerance, applied uniformly to all plants; consecutive pairs are
compared and, if "similar," **replaced by their mean hyperplane**:

- **Angle method** — for normals `n₁, n₂`:

  ```
  θ = arccos( (n₁ · n₂) / (‖n₁‖ · ‖n₂‖) )
  ```

  Merge if `θ < ε` (ε in **degrees**, ∈ [0, 90]). **Fully deterministic** from
  coefficients.

- **Distance method** — draw `N` random points in the two planes' active region:

  ```
  EQM = (1/N) Σ_{i=1..N} (gh_{1,i} − gh_{2,i})²,    δ = EQM / gh²_max
  ```

  Merge if `δ < ε` (ε in **percent**). `gh_max` = upper bound of the plant's
  generation window. **Uses an RNG → determinism hazard (see Determinism).**

- **Origin-plane invariant (both methods):** the plane through the origin —
  RHS = 0 **and** useful-volume coefficient = 0 — is **never** merged. This
  guarantees **zero generation at zero turbining**.

CEPEL surfaces this via two cards — `…REDUCAO-CORTES-ANGULO-PADRAO` (angle, deg)
and `…REDUCAO-CORTES-DISTANCIA-PADRAO` (distance, %), mutually exclusive.

### A6. Other confirmed CEPEL facts (context)

- **Per-period FPHA** — DECOMP/NEWAVE build a separate FPHA per period; γ differ
  by month. (WS1.)
- **Average volume in the LP** — `V` → `(V_in + V_out)/2`. _cobre already does._
- **Deviation diagnostics** — `oper_desvio_fpha` / `oper_desvio_medio_fpha` report
  per-point and aggregate FPHA-vs-exact deviations (signed up/down, by
  reservoir/run-of-river). These are the proper replacement for cobre's
  `kappa_warnings`.
- **"Fuga"/violation** — tiny spill penalty + slightly larger turbine penalty to
  avoid turbine-spillage; `f_FPHA` slack for feasibility.

---

## Part B — cobre today (verified against HEAD)

- **Fitting** (`crates/cobre-sddp/src/production/fpha_fitting/`): 3-D
  tangent-plane sampling over `(v, q, s)` (`tangent.rs`) →
  `eliminate_redundant` → greedy `select_planes` → `compute_kappa`
  (`selection.rs`). A heuristic envelope, **not a hull**; **no `α`-style balance**;
  spillage is **grid-sampled** (cap `0.5·q_max`), **not a secant**. (Note: cobre's
  fitter is full 3-D tangent sampling — neither CEPEL's hull nor the
  tangent-in-volume variant.)
- **Validation** (`selection.rs`): `γ_v ≥ 0`, `γ_s ≤ 0` (`gamma_s ≤ 1e-10`),
  `kappa ∈ (0, 1]` (`InvalidKappa`); `intercept = γ_0 · kappa`;
  `low_kappa_warning` when `kappa < 0.95`.
- **Stage-independence** (`production/hydro_models/production.rs`):
  `resolve_production_models_from_artifacts` fits **once per hydro** at the first
  study stage and reuses for all stages, even though `find_fpha_config_for_stage`
  can return different `fpha_config` per stage range. Precomputed FPHA supports
  per-stage rows (`FphaHyperplaneRow.stage_id = Some(...)`); computed does not
  (always `None`).
- **Tailrace** (`TailraceModel` on `crates/cobre-core/src/entities/hydro.rs`):
  single `Polynomial` (Horner) **or** piecewise-**linear** `Piecewise`, in total
  outflow `q + s`. Evaluated in `fpha_fitting/geometry.rs`
  (`evaluate_tailrace`). **No families, no downstream coupling, no remanso, no
  quartic pieces.**
- **Losses** (`geometry.rs`): `Factor{k}` (× gross head) or `Constant{m}`; the
  `turbined_m3s` argument is accepted but **ignored** (so `f_PerdH(Q)` is
  unimplemented).
- **Water balance** (`crates/cobre-sddp/src/lp/builder/matrix.rs`,
  `fill_state_and_water_entries`): storage continuity with turbine, spill,
  **same-stage** cascade upstream (`cascade.upstream`), diversion
  (`fill_diversion_columns`), evaporation, withdrawal slacks, inflow-non-negativity
  slack. **No travel time, no reversible plants, no post incremental laterals.**
  `Q_jus` is implicit (`q + s`); there is no lateral-post stream and no
  participation factors.
- **Reference volume per stage** — `HydroReferenceVolumeFractions` + override
  already resolves a per-stage reference volume (used for `ρ_eq = ρ_esp · h_eq`).
  The hook WS3 needs for the downstream reference level already exists.

---

## Part C — Workstreams

### WS1 — Stage-dependent FPHA fitting

**Problem.** Fit is hardcoded stage-independent; per-stage-range `fpha_config` is
parsed but ignored on the compute path.

**Change.** In `resolve_production_models_from_artifacts`, fit **once per
production-model `SelectionMode` entry** (the granularity already used for
`equivalent_productivity`, Decision A):

- `seasonal` → one fit per season; every stage in a season maps to that season's
  plane set.
- `stage_ranges` → one fit per range.

Each entry uses its own `fpha_config` (window/discretization) and the downstream
reference level resolved at the same granularity (WS3). Emit per-stage export rows
(`FphaHyperplaneRow.stage_id = Some(stage.id)`) by expanding each entry's plane
set across its stages; dedup identical entries so an unchanged season/range is
fitted once.

### WS2 — qhull hull + `α_FPHA` + lateral secant (the sole fitting path)

**This fully replaces** `sample_tangent_planes` + `eliminate_redundant` +
`select_planes` + `compute_kappa`. No dual path, fallback, or feature flag; the
old fitter and everything existing solely to support it is deleted (see "Dead
code").

1. Grid over the fit window (`V × Q`); evaluate **exact** `P = ρ_esp · q · h_net`
   with **spillage = 0, lateral = 0** → the `(V, Q, GH)` cloud + closing point
   `(V̄, Q̄, 0)`.
2. **3-D convex hull** via **qhull** → upper-envelope facets =
   `γ_0 + γ_V·V + γ_Q·Q`.
3. **`α_FPHA`** least-squares correction (A2.4), spillage = 0 grid.
4. **Lateral secant** `γ_S` per plane over `[0, S_max]` (A2.5), axis = `Q_lat`
   (WS4), offset `Q̂_lat`.

**Hull implementation (Decisions B + I).** C++ **qhull** for the 3-D hull (NEWAVE
parity), integrated as a **hand-rolled FFI at `gemm.rs` scale** (Decision I) — not
a third-party wrapper crate:

- **Reentrant `libqhull_r` only**, compiled directly with the `cc` crate (no
  CMake, no `bindgen`/libclang — its ~17 `.c` files _are_ the whole build). One C
  shim (`qhull_wrapper.c`) takes the cloud and returns facet hyperplanes (normal +
  offset); one `unsafe extern "C"` binds it; a thin safe wrapper owns RAII +
  canonical sort. Mirrors the solver FFI layering and the single-`unsafe`-fn
  discipline of `gemm.rs`, and keeps the workspace clear of `bindgen`/libclang
  (the hand-written-`extern` house style). Vendoring & placement: see "qhull
  integration" below.
- **Determinism is a hard requirement.** Canonically **sort the input cloud**
  before the call and the **output facets** after; **disable joggle (`QJ`)** and
  handle degeneracies via qhull's deterministic merged-facet path with fixed
  options; assert **bit-identical** hyperplanes across input orderings and MPI
  rank counts. If qhull cannot be pinned to deterministic behavior on a supported
  platform, that is a **blocking** escalation.

**qhull integration (Decision I) — vendoring, placement, build.**

- **Vendoring (resolved): a git submodule** at `crates/cobre-sddp/vendor/qhull`
  pinned to tag `2020.2`, mirroring how `cobre-solver` vendors HiGHS/Clp/CoinUtils;
  `build.rs` `cc`-compiles only `src/libqhull_r/*.c`. Rationale: this matches the
  repo's single established pattern for vendored C and adds the **fewest new
  artifact types** (a `.gitmodules` line + a gitlink — no committed third-party
  `.c/.h` in our tree, no bespoke refresh script, no `.gitignore` negation block).
  An earlier draft chose a trimmed vendored copy to avoid CMake and
  `submodules: recursive` friction, but both are **already required by the solver
  submodules** (cmake is invoked by `cobre-solver/build.rs`; every CI workflow
  already runs `submodules: recursive`), so those arguments do not differentiate
  here. The submodule carries qhull's full repo (unused C++ libs/CLI/tests) — the
  same trade the solvers already accept; `cargo package` ships submodule files, so
  the published crate stays self-contained. Version bumps are a `git submodule
update` (no maintained vendor script).
- **Placement (resolved): inline in `cobre-sddp`.** The vendored source,
  `build.rs` (`cc` build of `libqhull_r`), the `qhull_wrapper.c` shim, the one
  `unsafe extern "C"`, and the safe wrapper live in `cobre-sddp`, reusing its
  **existing `unsafe_code = "allow"` override** (already in place for `gemm.rs`).
  This adds cobre-sddp's **first `build.rs`/C build** and widens its `unsafe`
  surface from `gemm` to `{gemm, qhull}` — both isolated behind single safe
  functions with `// SAFETY:` blocks. (A dedicated generic leaf crate was
  considered and set aside in favor of fewer moving parts and no new published
  member.)
- **License:** `cargo-deny` gains no new entry (the vendored C is invisible to it,
  like HiGHS/Clp today); add the **Qhull** license to `THIRD_PARTY_NOTICES.md`.

**Validation revisions.** `kappa` is retired; keep `γ_v ≥ 0` and `γ_s ≤ 0`; add
`α_FPHA > 0`. (`α` is a least-squares balance that can be ≷ 1, unlike `kappa ≤ 1`.)

**Run-of-river / single-volume (`NPTV = 1`) handling (Decision K).** Validation
against the reference output showed run-of-river plants are fit with `Npt_V = 1`
(`Vmin = Vmax` → a single volume point) and `Coef.Vutil = 0`; in the reference
case 89 of 151 plants are run-of-river. A single-volume cloud is **coplanar in
`V`**, which the 3-D hull cannot fit as posed: cobre's current
`resolve_fitting_bounds` rejects it at the `v_min >= v_max` guard
(`EmptyFittingWindow`) and the `n_volume_points >= 2` guard
(`InsufficientDiscretization`), and even past those guards a degenerate (zero-`V`-
range) cloud returns `HullError::Degenerate`. So a run-of-river plant cannot be fit
on the computed hull path as-is — it errors. v1 adds an explicit single-volume
path that yields clean `γ_V = 0` planes. The mechanism is the **tangent-in-volume
trick** — split the single volume into two points 1% of useful volume apart so the
3-D hull stays non-degenerate while the `V` slope collapses to ≈ 0 — promoted here
from the out-of-scope list (a 2-D `(Q, GH)` hull yielding explicit `γ_V = 0` planes
is the alternative). The run-of-river fitter must still produce a valid concave
over-approximation in `Q` and pass the determinism gate. This must land before the
computed-FPHA re-bless, since re-bless cannot pass while run-of-river plants fail
to fit.

**Dead code to remove** (with tests/orphaned imports): `sample_tangent_planes`,
`compute_tangent_plane`, `RawHyperplane`'s tangent role,
`ProductionFunction::partial_derivatives`; `eliminate_redundant`, `select_planes`,
`compute_grid_errors`, `max_planes_per_hydro` capping (plane trimming is now WS5,
**not** the greedy selector); `compute_kappa`, the `kappa`/`low_kappa_warning`
fields, `intercept = γ_0·kappa` scaling, `InvalidKappa`, the kappa branch of
`validate_fitted_planes`; gradient-only helpers with no caller
(`ForebayTable::height_derivative`, `evaluate_tailrace_derivative`,
`evaluate_losses_factor`, derivative-only `locate_tailrace` use); the kappa-warning
plumbing (`kappa_warnings` tuples, `HydroModelSummary.kappa_warnings`, CLI
display) — replaced by the `α_FPHA` / deviation diagnostics.

Keep: `ProductionFunction::{new, evaluate}`,
`ForebayTable::{new, height, v_min, v_max}`, `evaluate_tailrace`,
`evaluate_losses`, the (adapted) grid builder.

### WS3 — Exact tailrace: piecewise quartics + `HrefJus` families (remanso)

**Data model.** A new optional parquet, e.g. **`system/tailrace_curves.parquet`**,
mirroring the `hydro_geometry.parquet` pattern. **Carry the quartic coefficients
and validity bounds** (not sampled `(outflow, height)` points — those cannot
reproduce CEPEL's quartic shape):

| column          | meaning                                                                                                           |
| --------------- | ----------------------------------------------------------------------------------------------------------------- |
| `hydro_id`      | plant whose tailrace this describes                                                                               |
| `href_jus_m`    | downstream-reservoir reference level keying the family (`HrefJus`); `NULL`/single value ⇒ no remanso (one family) |
| `segment_id`    | piece index within the family                                                                                     |
| `q_jus_inf_m3s` | segment lower validity bound                                                                                      |
| `q_jus_sup_m3s` | segment upper validity bound                                                                                      |
| `a_cf0 … a_cf4` | degree-4 polynomial coefficients for the segment                                                                  |

- **Within a family:** select the segment by `Q_jus`, evaluate the quartic.
- **Between families:** interpolate by the **downstream plant's stage reference
  level** (bilinear, as `ForebayTable` interpolates `hydro_geometry`).

**Downstream level resolution.** For plant `i` with `downstream_id = J_i`: take
`J_i`'s **stage reference volume** (`reference_volume_fractions`/override) → its
forebay via _its_ `hydro_geometry` → select/interpolate plant `i`'s tailrace
family at that level. No `downstream_id` (run-of-river / last plant) ⇒ single
family, no remanso.

**Feeding the fit.** The table changes only **how `h_jus(Q_jus)` is evaluated**
inside the production function during sampling (WS2.1) and the secant (WS2.4). The
hull, `α_FPHA`, and secant procedure are unchanged.

**Decisions (resolved):** family key = downstream **level (m)** (`HrefJus`),
not volume (Decision C); representation = **standalone optional parquet**, not a
`TailraceModel` enum variant (Decision D). The entity keeps `Polynomial`/
`Piecewise` for the simple no-remanso case.

### WS4 — General `Q_jus` lateral-flow composition (smart defaults)

**Principle (Decision E, revised 2026-06-14): v1 models only the own-spill secant
(`γ_S` = own spillage = the reference output's `Qver`); the external-lateral `Qlat`
coefficient (upstream defluences + post incremental inflows, the WS4b/c work) is
deferred to a future version (Decision J), gated on detecting that a plant actually
has lateral-inflow contributions. When absent — the common case — no `Qlat` term is
emitted.** The original Decision E ("full CEPEL `Q_jus` composition with smart
defaults collapsing to `Q_jus = Q + S`") remains the eventual target shape; the
revision narrows v1 to the own-spill axis because validation against the reference
case found `Coef.Qlat = 0` for all 151 plants. The LP already carries a `γ_s·s`
spill term in the FPHA average-storage form (`g ≤ γ₀ + (γᵥ/2)·(V_in+V_out) + γ_q·q +
γ_s·s`); the own-spill secant feeds that existing coefficient, so **v1 needs no new
LP term** — the own-spill secant (`γ_S`) and its existing spill term are already in
place. The richer composition below documents the deferred future shape.

**Data model.** An optional config (parquet or production-model-config section),
per hydro:

| field                           | default                                      | effect                                                     |
| ------------------------------- | -------------------------------------------- | ---------------------------------------------------------- |
| `spill_affects_tailrace`        | `true`                                       | gates whether the `γ_S` lateral secant exists              |
| `k_jus_q`                       | `1.0`                                        | own-turbine factor (folded into `γ_Q`)                     |
| `k_jus_s`                       | `1.0` if `spill_affects_tailrace` else `0.0` | own-spill factor in `Q_lat`                                |
| upstream defluence contributors | empty list of `(source_hydro_id, k_jus_qd)`  | adds `Σ k_jus_qd·(Q_j + S_j)` to `Q_lat`                   |
| post inflow contributors        | empty list of `(gauge_id, k_jus_qa)`         | adds `Σ k_jus_qa·Qa_j` to `Q_lat`; sets `Q̂_lat` = mean(Qa) |

**Default collapse:** with all defaults, `Q_jus = 1·Q + 1·S = Q + S`,
`Q_lat = S`, `Q̂_lat = 0` — **bit-identical in intent to cobre today**. No new
files required for simple plants.

**LP wiring** (`lp/builder/`):

- **Own spill `k_jus_s·S_i`** — already a column; scales the existing spill term
  in the FPHA row's lateral axis.
- **Upstream defluence `Σ k_jus_qd·(Q_j + S_j)`** — couples the FPHA row to
  upstream plants' turbine + spill columns (those columns already exist for the
  water balance). **Same-stage only** (travel time deferred).
- **Post incremental `Σ k_jus_qa·Qa_j`** — a **data** stream (per stage/scenario);
  enters the **RHS**, with `Q̂_lat` (mean over the period/scenarios) as the secant
  reference. Requires a new optional per-gauge incremental-inflow input.

**Sequencing within WS4 (smart-default staging).** Because the model is general
but defaults are inert, the LP/data work can land incrementally without changing
results for plants that don't opt in: (1) own spill/turbine factors + the
proper secant axis; (2) upstream-defluence coupling; (3) post incremental
inflows + `Q̂_lat`. Each later stage is a no-op for plants with empty
contributor lists.

### WS5 — Mandatory similar-hyperplane simplification (both methods)

**Decision F: implement both CEPEL methods.** A post-fit pass over each plant's
plane set (per stage), config-selected, mutually exclusive, applied uniformly:

- **Angle method** — merge consecutive pairs with `θ < ε_angle` (degrees) into
  the mean hyperplane. Deterministic.
- **Distance method** — merge consecutive pairs with `δ = EQM/gh²_max < ε_dist`
  (percent), `EQM` over `N` sampled points in the active region.
- **Origin-plane invariant** — never merge the RHS = 0 ∧ `γ_V` = 0 plane.

**Config.** Two mutually-exclusive options (mirroring the CEPEL cards), e.g. an
`fpha_plane_reduction` block: `{ method: angle, tolerance_deg }` **xor**
`{ method: distance, tolerance_pct, n_samples }`. Off by default (no reduction);
when set, applied to all plants. `compile_error!`-style mutual exclusivity is at
the config layer, not the build.

**Determinism (gating).** The **distance** method's `N` random points must use a
**canonically-seeded deterministic RNG** (seed derived from stable plant/stage/
plane-pair identity, never wall-clock/`rand::thread_rng`), and the merge order
must be canonical, so results are **bit-identical across input ordering and MPI
rank counts** — same as everywhere else in cobre. The **angle** method is
inherently deterministic. Both must pass the shuffle-invariance test.

---

## How the pieces compose (computed-FPHA, revised)

```
for each hydro h with a computed-FPHA config:
  for each SelectionMode entry e of h (one per season, or one per range)  (WS1):
      window     ← e.fpha_config.fitting_window
      h_jus(·)   ← tailrace family @ downstream(h) ref level for e        (WS3)
                   (or entity TailraceModel if no table)
      latcfg     ← Q_jus/Q_lat composition for h (smart defaults)         (WS4)
      cloud      ← grid over window; P = ρ_esp·q·h_net(v,q, spill=0,lat=0) (WS2.1)
      planes     ← qhull 3-D convex hull(cloud + closing point)           (WS2.2)
      α          ← least-squares FPHA₀ vs FPH on V×Q grid (spill=0)        (WS2.3)
      γ_S        ← per-plane secant over Q_lat ∈ [0, S_max]               (WS2.4)
      planes     ← merge similar hyperplanes (angle xor distance)         (WS5)
      emit planes (scaled by α) across e's stages as FphaHyperplaneRow
LP build:  FPHA row gains the Q_lat term (own spill, upstream defluences,  (WS4)
           post-inflow RHS offset); violation slack f_FPHA as today.
```

---

## Resolved decisions

| #     | Decision                       | Resolution                                                                                                                                                                                                                                                                                                                                                                                     |
| ----- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A** | Stage-fit granularity          | Per `SelectionMode` entry (one fit per season / range), mirroring `equivalent_productivity`                                                                                                                                                                                                                                                                                                    |
| **B** | Hull implementation            | Full 3-D hull via `qhull` (NEWAVE parity); deterministic via canonical sort-in/out + no joggle (`QJ`)                                                                                                                                                                                                                                                                                          |
| **C** | Tailrace-family key            | Downstream **level (m)** (`HrefJus`)                                                                                                                                                                                                                                                                                                                                                           |
| **D** | Tailrace-table representation  | Standalone optional parquet, **carrying piecewise-quartic coefficients + validity bounds** (not sampled points)                                                                                                                                                                                                                                                                                |
| **E** | Lateral-flow scope             | **Full CEPEL `Q_jus` composition**, with **smart defaults collapsing to `Q_jus = Q + S`** (zero new input for simple plants); staged LP wiring                                                                                                                                                                                                                                                 |
| **F** | Similar-hyperplane reduction   | **Both** methods (angle + distance), CEPEL-parity, mutually exclusive, off by default; deterministic RNG for distance                                                                                                                                                                                                                                                                          |
| **G** | Hydric-balance completion      | **Out of scope / separate track**; consume existing same-stage upstream `Q + S` as `Q_def`                                                                                                                                                                                                                                                                                                     |
| **H** | Variable ρ / ρ-unification dep | **Dropped**; constant `ρ_esp` for v1; variable `ρ(Q,h)` / `f_PerdH(Q)` is a future increment                                                                                                                                                                                                                                                                                                   |
| **I** | qhull integration              | **Hand-rolled FFI at `gemm.rs` scale** (no wrapper crate, no `bindgen`): `cc`-built `libqhull_r` from a **git submodule** (`crates/cobre-sddp/vendor/qhull` @ `2020.2`, mirroring the solver submodules), **inline in `cobre-sddp`** under its existing `unsafe` override; Qhull license → `THIRD_PARTY_NOTICES.md`                                                                            |
| **J** | External-lateral `Qlat` scope  | **Deferred to a future version.** v1 fits/wires only the own-spill secant (`γ_S` = own spillage = reference output's `Qver`); the external `Qlat` coefficient (upstream defluences + post inflows, WS4b/c) is gated on detecting a plant has lateral-inflow contributions. Reference case: `Qlat = 0` for all 151 plants, so the common case emits no `Qlat` term and v1 needs no new LP term. |
| **K** | Run-of-river / single-volume   | **Add an explicit single-volume (`NPTV = 1`, `Vmin = Vmax`) fitting path** yielding clean `γ_V = 0` planes; current code errors (`EmptyFittingWindow` / `InsufficientDiscretization` guards, then `HullError::Degenerate`). Mechanism = the tangent-in-volume trick (two volume points 1% of useful volume apart), promoted from out-of-scope; must precede the computed-FPHA re-bless.        |

## Dependencies & sequencing

1. **WS2 (hull) + WS1 (stage-awareness)** land together — both restructure
   `resolve_production_models` + `fpha_fitting/`. (No external ρ-unification
   dependency — Decision H.)
2. **WS3 (tailrace families)** builds on WS1 (per-stage downstream reference
   level) and feeds WS2's sampler.
3. **WS4 (lateral composition)** — its **secant axis** is needed by WS2.4, so the
   `Q_lat` definition (even if just `Q_lat = S` by default) lands with WS2; the
   upstream-defluence and post-inflow LP wiring stage in afterward (smart-default
   staging) without changing simple-plant results.
4. **WS5 (simplification)** is a post-fit pass — lands after WS2 produces planes;
   independent of WS3/WS4.

## Relationship to the hydric balance (out of scope, for reference)

CEPEL's full balance (transcribed):

```
V_i^t = V_i^{t-1} + ς[ Q_inc − Q_ev,i − Q_out,i
       + Σ_p ( Σ_{j∈M_i}(Q_j+S_j)
             + Σ_{j∈M_tv,i}(Q_j+S_j)^{t−τ_ij}
             − (Q_i+S_i)
             + Σ_{j∈M_dv,i} Q_dv,j
             + Σ_{j∈M_eb,i} Q_b,j ) ]
```

cobre covers `Q_inc, Q_ev, Q_out (withdrawal), M_i (same-stage), M_dv (diversion)`.
**Missing (separate track):** `M_tv` (travel time `τ`) and `M_eb`
(reversible/pumped). This effort uses the existing same-stage `Q_j + S_j` as the
`Q_def` feeding `Q_jus` (WS4). There is also an optional **per-block balance** for
run-of-river plants (CEPEL `HIDRELETRICA-BALANCO-HIDRICO-PATAMAR`) — noted, not in
scope.

## Validation against reference FPHA output

The design was checked against real NEWAVE FPHA output for a reference case
(151 plants, 64 periods): a fitted-cuts report and a config-echo report. The check
confirmed the approach is largely on-track and pinned two adjustments (Decisions J
and K above).

**Confirmed on-track:**

- **Per-period cuts** — the reference fits a separate FPHA per period, matching
  cobre's per-stage fitting (WS1).
- **Coefficient mapping** — the reference columns map to cobre's coefficients:
  `FCorrec` = `α` (the least-squares correction), `RHS` = `γ₀`, `Coef.Vutil` =
  `γ_V`, `Coef.Qtur` = `γ_Q`, `Coef.Qver` = the own-spill secant (cobre's `γ_S`).
- **Signs** — `γ_q ≥ 0`, the own-spill/lateral coefficient `≤ 0`, `γ_v ≥ 0`,
  consistent with cobre's validation rules.
- **`α` (FCorrec) range** — observed in `[0.971, 1.0]`, i.e. mostly `≤ 1` (FPH
  mostly concave), but cobre's `α` may be `≷ 1` and that is fine (the few cases
  where FPH is locally non-concave pull `α` above 1).

**Adjustments pinned:**

- **Decision J** — the reference cut carries two lateral coefficients, `Coef.Qver`
  (own spill) and `Coef.Qlat` (external lateral). `Coef.Qlat` is **zero for all 151
  plants** (no lateral-flow plants in a typical case), so v1 models only the
  own-spill secant and defers the explicit `Qlat` coefficient.
- **Decision K** — run-of-river plants are fit with `Npt_V = 1` (single volume
  point, `Vmin = Vmax`) and `Coef.Vutil = 0`; 89 of 151 plants are run-of-river.
  v1 adds an explicit single-volume hull path (see WS2).

**Discretization sizing.** The reference grid is small — `NPTQ = 5`, `NPTV = 5` for
reservoirs and `NPTV = 1` for run-of-river. cobre's **default discretization should
be ~5** to keep cut counts compact: in the reference the per-plant·period cut count
has a median of 4 and a maximum of 22. (cobre's current default is `5` per axis,
which matches.)

**Distance-method `gh_max`.** For the WS5 distance-method merge, `gh_max` is the
per-plant **generation-window upper bound** (the reference's "Janela GH"), not a
global constant.

**Output-schema forward consideration (kappa-retirement / export-row work).** To
enable a direct diff against the reference cuts, the FPHA export should carry
`alpha` (= `FCorrec`), `gamma_v`, `gamma_q`, and `gamma_qver` (own spill, = today's
`gamma_s`), and **drop `kappa`** (retired in the kappa-retirement work). The
external `gamma_qlat` column is deferred with Decision J. This is a forward
consideration for the kappa-retirement step and any export-row schema work — no
code change is made here.

## Validation & testing

- **Determinism (gating):** bit-identical hyperplanes regardless of hydro/stage
  input ordering and MPI rank count, for (a) the qhull hull (shuffle-the-cloud
  test, canonical-sort-in/out), and (b) WS5's distance method (seeded-RNG +
  canonical merge-order test). Pin the qhull version; non-deterministic qhull is a
  blocking escalation.
- **Re-bless:** every computed-FPHA deterministic case and the slow-tests
  plane-selection suite — γ values change (qhull vs heuristic, `α` vs `kappa`,
  secant vs grid spillage, plus WS5 merges).
- **Unit:** qhull on known/degenerate clouds; `α_FPHA` closed form; secant slope
  on a synthetic tailrace; tailrace **piecewise-quartic** evaluation + segment
  selection + between-family bilinear interpolation + no-downstream single-family
  path; **`Q_jus`/`Q_lat` smart-default collapse** (empty contributors ⇒
  `Q_jus = Q + S`); upstream-defluence coupling; post-inflow RHS offset and
  `Q̂_lat`; WS5 angle + distance merges, **origin-plane never merged**,
  shuffle-invariance; per-season vs per-range fits differ when windows / downstream
  levels differ.
- **Cross-check (optional):** cobre hyperplanes vs NEWAVE for a shared case to
  bound the approximation gap; FPHA deviation diagnostics analogous to
  `oper_desvio_fpha`.

## Python parity

Every output the CLI writes for the above (FPHA hyperplane export, any tailrace /
lateral / deviation outputs) must be mirrored in `cobre-python` (`run.rs`), per
the project Python-parity rule.

## Out of scope / future

- **Hydric-balance completion** — travel time `τ` (`M_tv`) and reversible/pumped
  plants (`M_eb`); per-block run-of-river balance. (Separate design.)
- **Variable productivity / losses** — `ρ(Q, h_liq)` and `h_PerdH(Q)` GAM grids
  (CEPEL cards `HIDRELETRICA-PRODUTIBILIDADE-ESPECIFICA-GRADE` /
  `HIDRELETRICA-PERDA-HIDRAULICA-GRADE`); bilinear/linear grid interpolation.
- **Explicit external-lateral `Qlat` coefficient** (Decision J) — upstream
  defluences + post incremental inflows; gated on detecting a plant has
  lateral-inflow contributions. Reference case has `Qlat = 0` for all plants.
- **Tangent-in-volume FPHA for multi-volume reservoirs** (DESSEM weekly/monthly
  reservoirs). Note: the tangent-in-volume construction is **used** in v1 for the
  run-of-river / single-volume case (Decision K, WS2) — only its use for genuinely
  volume-varying reservoirs is out of scope.
- **FPHAD** (dynamic/iterative cut introduction) — conceptually cobre's DCS.

## Next step

Decisions A–K are settled. The first planning spike is **qhull determinism +
build integration** (gates WS2, adds a C-library dependency). Then generate the
implementation plan into `plans/` via `/plan`. The `Q_lat` secant-axis definition
(WS4 default `Q_lat = S`) must be fixed before WS2's secant is implemented, even
though the richer lateral wiring (upstream defluences, posts) stages in later.
