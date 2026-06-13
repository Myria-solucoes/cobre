//! FPHA hyperplane fitting from reservoir geometry.
//!
//! Turns a hydro plant's Volume-Height-Area (VHA) curve into a set of tangent
//! hyperplanes approximating the production function `phi(v, q, s)`. The pipeline
//! evaluates the production function at the points of a uniform 3D grid and fits a
//! piecewise-linear outer approximation.
//!
//! # Submodule layout
//!
//! - `error` — `FphaFittingError`, the validation-error enum every fallible step returns.
//! - `geometry` — `FittingBounds` + `resolve_fitting_bounds`, the `ForebayTable`
//!   VHA interpolation, and the tailrace / hydraulic-loss evaluators
//!   (`evaluate_tailrace`, `evaluate_losses`, …).
//! - `production` — the head-conversion constant and `ProductionFunction`, the
//!   evaluable `phi(v, q, s)` with analytical derivatives.
//! - `tangent` — `RawHyperplane`, `compute_tangent_plane`, and the grid sweep
//!   `sample_tangent_planes`.
//! - `grid` — the single authoritative owner of the uniform 3D grid formula
//!   (`GridParams` + `build_grid`), shared by the sampling and selection steps.
//! - `selection` — redundancy elimination, greedy plane selection, kappa
//!   computation, and the coefficient-sign validation.
//!
//! The orchestration entry point `fit_fpha_planes` and its result `FphaFitResult`
//! live here in `mod`, co-located with the re-export surface.
//!
//! Six `pub(crate)` symbols form the `crate::fpha_fitting::Symbol` surface that
//! resolves verbatim for every cross-cluster consumer: four are re-exported from
//! submodules (`FphaFittingError`, `ForebayTable`, `evaluate_losses`,
//! `evaluate_tailrace`) and two are defined here in `mod` (`FphaFitResult`,
//! `fit_fpha_planes`). All six are `pub(crate)`, so this doc names them with
//! backtick spans rather than intra-doc links (a `pub(crate)` module linking to
//! `pub(crate)` items would otherwise risk `rustdoc::private_intra_doc_links`).

use cobre_core::Hydro;
use cobre_io::extensions::{FphaColumnLayout, HydroGeometryRow};

use crate::hydro_models::FphaPlane;

mod error;
mod geometry;
mod grid;
mod production;
mod selection;
mod tangent;

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

use geometry::resolve_fitting_bounds;
use production::ProductionFunction;
use selection::{compute_kappa, eliminate_redundant, select_planes, validate_fitted_planes};
use tangent::sample_tangent_planes;

pub(crate) use error::FphaFittingError;
pub(crate) use geometry::{ForebayTable, evaluate_losses, evaluate_tailrace};

// ── Top-level fitting pipeline ────────────────────────────────────────────────

/// Combined result of the FPHA fitting pipeline.
///
/// Returned by [`fit_fpha_planes`] to expose both the fitted hyperplanes
/// and the unscaled `kappa` correction factor so that callers can reconstruct
/// the original `gamma_0` values for export.
///
/// The relationship between fields is:
///
/// ```text
/// plane.intercept = raw_gamma_0 * kappa
/// ```
///
/// To recover the unscaled `gamma_0` from a plane: `plane.intercept / kappa`.
#[derive(Debug)]
pub(crate) struct FphaFitResult {
    /// Fitted hyperplanes with pre-scaled intercepts (`gamma_0 * kappa`).
    pub planes: Vec<FphaPlane>,
    /// Nominal head correction factor κ ∈ (0, 1] applied during fitting.
    pub kappa: f64,
    /// Non-`None` when kappa < 0.95, carrying the kappa value for structured
    /// warning display by the caller.  The fitting result is still valid in
    /// this case; the warning is informational.
    pub low_kappa_warning: Option<f64>,
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
/// 4. **Sampling** — sample tangent hyperplanes on the 3D grid.
/// 5. **Redundancy elimination** — discard planes that are never the tightest bound.
/// 6. **Heuristic selection** — reduce to at most `max_planes_per_hydro` planes.
/// 7. **Kappa computation** — compute the correction factor on the selected planes.
/// 8. **Validation** — verify kappa and coefficient signs.
/// 9. **Conversion** — convert each `RawHyperplane` to `FphaPlane` with
///    `intercept = gamma_0 * kappa`.
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
pub(crate) fn fit_fpha_planes(
    forebay_rows: &[HydroGeometryRow],
    hydro: &Hydro,
    config: &FphaColumnLayout,
) -> Result<FphaFitResult, FphaFittingError> {
    let forebay = ForebayTable::new(forebay_rows, &hydro.name)?;

    let pf = ProductionFunction::new(
        forebay.clone(),
        hydro.tailrace.as_ref(),
        hydro.hydraulic_losses.as_ref(),
        hydro.efficiency.as_ref(),
        hydro.max_turbined_m3s,
        hydro.name.clone(),
    );

    let bounds = resolve_fitting_bounds(config, hydro, &forebay)?;

    let sampled = sample_tangent_planes(&pf, &bounds);
    let non_redundant = eliminate_redundant(&sampled, &pf, &bounds);
    let selected = select_planes(&non_redundant, &pf, &bounds);
    let kappa = compute_kappa(&selected, &pf, &bounds);

    let low_kappa_warning = validate_fitted_planes(&selected, kappa, &hydro.name)?;

    let planes = selected
        .iter()
        .map(|raw| FphaPlane {
            intercept: raw.gamma_0 * kappa,
            gamma_v: raw.gamma_v,
            gamma_q: raw.gamma_q,
            gamma_s: raw.gamma_s,
        })
        .collect();

    Ok(FphaFitResult {
        planes,
        kappa,
        low_kappa_warning,
    })
}
