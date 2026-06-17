//! FPHA hyperplane fitting from reservoir geometry.
//!
//! Turns a hydro plant's Volume-Height-Area (VHA) curve into a set of
//! hyperplanes approximating the production function `phi(v, q, s)`. The pipeline
//! evaluates the production function on a uniform `(V, Q)` grid (at spillage = 0),
//! takes the 3-D convex hull of the resulting `(V, Q, generation)` cloud, and reads its
//! upper-envelope facets as a piecewise-linear concave outer approximation.
//!
//! # Submodule layout
//!
//! - `error` — `FphaFittingError`, the validation-error enum every fallible step returns.
//! - `geometry` — `FittingBounds` + `resolve_fitting_bounds`, the `ForebayTable`
//!   VHA interpolation, and the tailrace / hydraulic-loss evaluators
//!   (`evaluate_tailrace`, `evaluate_losses`, …).
//! - `production` — the head-conversion constant and `ProductionFunction`, the
//!   evaluable `phi(v, q, s)` with analytical derivatives.
//! - `hull_fit` — `fit_hull_planes`, the convex-hull fitter that builds the
//!   `(V, Q, generation)` cloud and extracts upper-envelope planes (the production path).
//!   Also owns `RawPlane`, the four-coefficient carrier the whole pipeline threads.
//! - `grid` — the single authoritative owner of the uniform grid formula
//!   (`GridParams` + `build_grid`), shared by the hull cloud, the α regression,
//!   the deviation diagnostic, and the secant's representative-point scan.
//! - `alpha` — `compute_alpha_fpha` + `scale_plane_affine`, the least-squares
//!   `α_FPHA` correction and the whole-affine scaling it drives.
//! - `deviation` — `compute_fit_deviation` + `FphaFitDeviation`, the fit-quality
//!   measure of the final plane set against the exact production function (the
//!   replacement for the retired `kappa` shrink).
//! - `secant` — `fit_gamma_s_for_planes`, the per-plane lateral-flow `γ_S` secant.
//! - `tailrace` — `TailraceFamilies` + `build_tailrace_families_map`, the exact
//!   piecewise-quartic backwater-coupled tailrace evaluation.
//! - `reduction` — `reduce_planes`, the post-fit similar-hyperplane merge.
//! - `rng` — the deterministic seeded PRNG and identity hash the `Distance`
//!   reduction arm samples with.
//! - `selection` — the coefficient-sign and `α_FPHA > 0` validation
//!   (`validate_fitted_planes`).
//!
//! The orchestration entry point `fit_fpha_planes` and its result `FphaFitResult`
//! live here in `mod`, co-located with the re-export surface.
//!
//! The `pub(crate)` symbols this module re-exports from its submodules
//! (`FphaFittingError`, `ForebayTable`, `evaluate_losses`, `evaluate_tailrace`,
//! `TailraceSource`, `TailraceFamilies`, `build_tailrace_families_map`,
//! `FphaFitDeviation`) plus the two defined here in `mod` (`FphaFitResult`,
//! `fit_fpha_planes`) form the `crate::fpha_fitting::Symbol` surface that resolves
//! verbatim for every cross-cluster consumer. This doc names them with backtick
//! spans rather than intra-doc links: a `pub(crate)` module linking to
//! `pub(crate)` items would otherwise risk `rustdoc::private_intra_doc_links`.

use cobre_core::Hydro;
use cobre_io::extensions::{FphaColumnLayout, HydroGeometryRow};

use crate::hydro_models::FphaPlane;

mod alpha;
mod deviation;
mod error;
mod geometry;
mod grid;
mod hull_fit;
mod production;
mod reduction;
mod rng;
mod secant;
mod selection;
mod tailrace;

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::similar_names
)]
mod tests;

use cobre_io::extensions::PlaneReductionConfig;

use alpha::{compute_alpha_fpha, scale_plane_affine};
use deviation::{collect_fit_deviation_points, compute_fit_deviation};
use geometry::resolve_fitting_bounds;
use hull_fit::{RawPlane, fit_hull_planes};
use production::ProductionFunction;
use reduction::reduce_planes;
use secant::fit_gamma_s_for_planes;
use selection::validate_fitted_planes;

pub(crate) use deviation::{FphaDeviationPoint, FphaFitDeviation};
pub(crate) use error::FphaFittingError;
pub(crate) use geometry::{ForebayTable, evaluate_losses, evaluate_tailrace};
pub(crate) use production::TailraceSource;
pub(crate) use tailrace::{TailraceFamilies, build_tailrace_families_map};

// ── Top-level fitting pipeline ────────────────────────────────────────────────

/// Combined result of the FPHA fitting pipeline.
///
/// Returned by [`fit_fpha_planes`] to expose the fitted hyperplanes alongside the
/// scalar `alpha` correction the whole affine function was scaled by, so the
/// export-row builder can recover the unscaled raw coefficients.
///
/// # Contract — `planes` carry the α-scaled whole affine function (Voice 1)
///
/// Each plane is `α·FPHA_0`: `intercept`, `gamma_v`, `gamma_q`, and `gamma_s` are
/// all `α · (raw hull coefficient)`. To recover a raw coefficient divide by
/// `alpha` (`plane.intercept / alpha`, etc.). `alpha` is the single factor every
/// coefficient was scaled by — the export row builder writes the α-scaled
/// coefficients verbatim with the IO `kappa` column pinned to `1.0`, so the
/// precomputed read-back (`intercept = gamma_0 · kappa`) round-trips `α·FPHA_0`
/// exactly. Treating the gradients as still-raw and re-applying an intercept-only
/// shrink would double-correct the export.
#[derive(Debug)]
pub(crate) struct FphaFitResult {
    /// Fitted hyperplanes, each the α-scaled whole affine function `α·FPHA_0`.
    pub planes: Vec<FphaPlane>,
    /// Least-squares `α_FPHA` correction every plane coefficient was scaled by.
    ///
    /// May be ≷ 1: `α` is an MSE balance between the raw hull envelope and the
    /// exact production function, not a bounded-below-1 factor. Divide a plane
    /// coefficient by `alpha` to recover the raw hull coefficient.
    // Intent/Seam: the export-row builder writes the α-scaled coefficients
    // verbatim (with the IO `kappa` column pinned to 1.0), so it needs no
    // un-scaling and does not read this field today. It is surfaced for a future
    // consumer that recovers the raw hull coefficients and is read by the fitter
    // unit tests; the allow re-fires when a non-test reader is removed.
    #[allow(dead_code)]
    pub alpha: f64,
    /// Fit-quality deviation of the emitted plane set from the exact production
    /// function on the spill = 0 grid.
    ///
    /// Measured on the FINAL (post-reduction) planes — the set the LP applies.
    /// The resolver warns when [`FphaFitDeviation::exceeds_warn_threshold`]; this
    /// is the operator-facing fit-quality signal that replaces the retired
    /// low-`kappa` warning.
    pub deviation: FphaFitDeviation,
    /// Per-sampled-point residuals the [`Self::deviation`] aggregate was reduced
    /// from, on the same spill = 0 grid — empty unless the caller requested them.
    ///
    /// Populated only when `collect_deviation_points` is passed to
    /// [`fit_fpha_planes`]; otherwise empty with zero collection overhead, leaving
    /// the fit bit-identical. Carried up so the opt-in
    /// `fpha_deviation_points.parquet` export can plot exactly where on the grid a
    /// fit diverges.
    pub deviation_points: Vec<FphaDeviationPoint>,
}

/// Fit FPHA hyperplanes for a single hydro plant from its VHA curve geometry.
///
/// This is the top-level entry point for the computed FPHA path. It orchestrates
/// the full pipeline:
///
/// 1. **Forebay table** — build `ForebayTable` from the VHA curve rows.
/// 2. **Production function** — build `ProductionFunction` from the forebay table
///    and the hydro plant's tailrace, hydraulic loss, and efficiency models.
/// 3. **Fitting bounds** — resolve volume range and grid counts from the config.
/// 4. **Hull fit** — build the `(V, Q, generation)` production cloud (spillage = 0,
///    lateral = 0), take the 3-D convex hull, and read its upper-envelope facets
///    as raw planes. The zero-flow (`q = 0`) column anchors the lower closure, so
///    no synthetic closing point is added.
/// 5. **Alpha correction** — compute the least-squares `α_FPHA` on the spill = 0
///    grid and scale every plane's whole affine function by `α`.
/// 6. **Lateral secant** — fit the lateral-flow secant `γ_S` over
///    `lateral_flow ∈ [0, S_max]` on the α-scaled planes (smart default `lateral_flow = S`,
///    `reference lateral flow = 0`; `S_max = 2·long-term mean inflow` or the `2 × max_turbined` fallback).
/// 7. **Validation** — require `α_FPHA > 0` and the coefficient signs (including
///    the fitted `γ_S ≤ 0`).
/// 8. **Conversion** — convert each α-scaled, `γ_S`-fitted `RawPlane` to
///    `FphaPlane` (`intercept = α·gamma_0`, `γ_v` / `γ_q` = `α·gamma_*`, `γ_S`
///    the fitted secant).
///
/// The returned `Vec<FphaPlane>` is structurally identical to what the precomputed
/// path produces from `fpha_hyperplanes.parquet`: the LP builder treats both paths
/// identically.
///
/// # Errors
///
/// Any step in the pipeline can fail. All errors propagate via `?` and are
/// variants of [`FphaFittingError`]. The caller receives a descriptive error
/// that includes the hydro plant name.
///
/// # Parameters
///
/// - `forebay_rows` — VHA curve rows for the hydro plant, sorted ascending by
///   `volume_hm3` (as returned by `cobre_io::extensions::parse_hydro_geometry`).
/// - `hydro` — resolved hydro plant entity supplying physical bounds and models.
/// - `config` — FPHA fitting configuration (grid sizes, optional fitting window).
/// - `long_term_mean_inflow_m3s` — long-term mean natural inflow \[m³/s\] for the lateral-secant
///   `S_max = 2·long-term mean inflow`; `0.0` (no inflow history) selects the `2 × max_turbined`
///   fallback (see `secant::resolve_s_max`).
/// - `tailrace_source` — the resolved [`TailraceSource`]. The caller resolves it
///   per (hydro, entry): [`TailraceSource::Families`] when the plant has a
///   `tailrace_curves` table, else [`TailraceSource::Entity`] mirroring the
///   entity-level model (the inert fallback).
/// - `reduction` — the optional file-level similar-hyperplane reduction config.
///   `Some(cfg)` runs the post-fit merge of near-parallel planes after the first
///   `validate_fitted_planes` and before the `FphaPlane` conversion; `None` is a
///   literal skip — the validated `scaled` set is converted unchanged, so the
///   fit is bit-identical to a run without any reduction config.
/// - `hydro_id` — the plant's `EntityId` `i32` (stable identity). Used only by
///   the `Distance` reduction arm to seed its sampling PRNG; the `Angle` arm and
///   the `None` skip ignore it.
/// - `entry_level_bits` — the per-`SelectionMode`-entry downstream-level bits
///   (`downstream_level_m.map(f64::to_bits).unwrap_or(0)`), the same dedup-key
///   component that identifies the stage/entry. Part of the `Distance` arm's
///   stable seed identity; ignored by the `Angle` arm and the `None` skip.
/// - `collect_deviation_points` — when `true`, after the aggregate is computed
///   the per-sampled-point residuals are collected onto
///   [`FphaFitResult::deviation_points`] for the opt-in deviation-points export.
///   `false` collects nothing (the field stays empty) with zero overhead, so the
///   fit is bit-identical regardless of the flag — the collection is read-only
///   over the already-emitted planes and never feeds back into the fit.
pub(crate) fn fit_fpha_planes(
    forebay_rows: &[HydroGeometryRow],
    hydro: &Hydro,
    config: &FphaColumnLayout,
    long_term_mean_inflow_m3s: f64,
    tailrace_source: TailraceSource,
    reduction: Option<&PlaneReductionConfig>,
    hydro_id: i32,
    entry_level_bits: u64,
    collect_deviation_points: bool,
) -> Result<FphaFitResult, FphaFittingError> {
    let forebay = ForebayTable::new(forebay_rows, &hydro.name)?;

    let pf = ProductionFunction::new(
        forebay.clone(),
        tailrace_source,
        hydro.hydraulic_losses.as_ref(),
        hydro.efficiency.as_ref(),
        hydro.max_turbined_m3s,
        hydro.name.clone(),
    )
    .with_max_generation_mw(hydro.max_generation_mw);

    let bounds = resolve_fitting_bounds(config, hydro, &forebay)?;

    // Hull-based fitter: the upper-envelope facets of the (V, Q, generation) production
    // cloud. Any hull failure maps to a typed FPHA error instead of panicking, so
    // one pathological hydro does not abort the whole fitting loop. The dominant
    // case is a degenerate (collinear/coplanar) cloud — e.g. a constant-head
    // linear production function; the non-finite/compute/alloc cases cannot arise
    // from a finite, in-range cloud and fold into the same typed error.
    let hull_planes = fit_hull_planes(&pf, &bounds).map_err(|_err| {
        FphaFittingError::DegenerateProductionCloud {
            hydro_name: hydro.name.clone(),
        }
    })?;

    // α_FPHA is the least-squares balance between the raw hull envelope (FPHA_0)
    // and the exact FPH on the spill = 0 grid. It is computed from the RAW hull
    // planes (FPHA_0), then scales the whole affine function of each.
    let alpha = compute_alpha_fpha(&hull_planes, &pf, &bounds);

    // Scale the whole affine function by α first, then fit the lateral-flow secant
    // γ_S on the α-scaled planes. γ_S replaces the provisional 0.0 each plane
    // carries out of the hull/α steps. The secant order matters: the
    // representative operating point per plane is read from the α-scaled envelope,
    // and γ_S is the only coefficient the secant sets — α scaling does not touch
    // the (already-0) γ_S, so scaling-then-fitting is the correct composition.
    let mut scaled: Vec<RawPlane> = hull_planes
        .iter()
        .map(|raw| scale_plane_affine(raw, alpha))
        .collect();

    // Smart default: lateral_flow = S (own spill), reference lateral flow = 0. `long_term_mean_inflow_m3s` is the per-hydro
    // long-term mean inflow, giving `S_max = 2·long-term mean inflow`; a hydro with no inflow
    // history passes `long_term_mean_inflow_m3s = 0.0` and falls back to `2 × max_turbined` (see
    // `secant::resolve_s_max`). The secant is fit for every plant.
    fit_gamma_s_for_planes(&mut scaled, &pf, &bounds, long_term_mean_inflow_m3s);

    // Validation runs on the α-scaled, γ_S-fitted planes so the `γ_s ≤ 1e-10`
    // sign rule sees the actual emitted lateral coefficient (not the provisional
    // 0). A positive α preserves the γ_v / γ_q signs, and the secant clamps γ_S
    // to ≤ 0, so the scaled planes satisfy the same sign contract as the raw set.
    validate_fitted_planes(&scaled, alpha, &hydro.name)?;

    // Similar-hyperplane reduction. `None` is a LITERAL SKIP — the validated
    // `scaled` set is converted unchanged, so a run without a reduction config is
    // bit-identical to the pre-reduction pipeline (no call into `reduction`). With
    // `Some(cfg)` the merge runs on the α-scaled, γ_S-fitted, sign-validated set,
    // and the reduced set is re-validated: the component-wise mean of two
    // sign-valid planes is itself sign-valid, so this guard cannot fail for
    // well-formed input, but it is cheap and makes the contract explicit.
    let reduced = match reduction {
        Some(cfg) => {
            let merged = reduce_planes(&scaled, cfg, &pf, &bounds, hydro_id, entry_level_bits);
            validate_fitted_planes(&merged, alpha, &hydro.name)?;
            merged
        }
        None => scaled,
    };

    // Fit-quality diagnostic on the FINAL emitted set. Measured on `reduced`, not
    // `scaled`, so it reflects the planes the LP actually applies after any
    // similar-hyperplane merge — measuring the pre-reduction set would understate
    // the gap a merge introduces.
    let deviation = compute_fit_deviation(&reduced, &pf, &bounds);

    // Per-sampled-point residuals, gated by `collect_deviation_points`. When
    // `false` this is an empty Vec with NO walk over the grid — zero overhead, so
    // the fit (and every value it returns) is bit-identical to a run with the flag
    // off. The collection is read-only over the FINAL `reduced` set on the SAME
    // spill = 0 grid the aggregate used, so it cannot perturb the fit.
    let deviation_points = if collect_deviation_points {
        collect_fit_deviation_points(&reduced, &pf, &bounds)
    } else {
        Vec::new()
    };

    // Conversion: each plane is the α-scaled whole affine function α·FPHA_0 with
    // the fitted γ_S. The intercept AND the γ_v / γ_q gradients carry the α factor.
    let planes = reduced
        .iter()
        .map(|scaled_plane| FphaPlane {
            intercept: scaled_plane.gamma_0,
            gamma_v: scaled_plane.gamma_v,
            gamma_q: scaled_plane.gamma_q,
            gamma_s: scaled_plane.gamma_s,
        })
        .collect();

    Ok(FphaFitResult {
        planes,
        alpha,
        deviation,
        deviation_points,
    })
}
