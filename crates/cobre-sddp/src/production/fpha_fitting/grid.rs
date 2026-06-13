//! Single authoritative owner of the uniform 3D FPHA fitting grid.
//!
//! Owns [`GridParams`] and [`build_grid`]. The grid formula is shared by the
//! `tangent` sampling step and the `selection` elimination / error / kappa steps;
//! both reach it via `super::grid::build_grid`. Duplicating the formula in either
//! caller would let the sampling grid and the error grid drift apart, silently
//! invalidating the outer-approximation guarantee.

use super::geometry::FittingBounds;
use super::production::ProductionFunction;

// ── Grid construction ─────────────────────────────────────────────────────────

/// Precomputed grid axis values for the fitting 3D grid over `(v, q, s)`.
///
/// Computed once by [`build_grid`] and reused across all pipeline steps that
/// iterate the same grid (`sample_tangent_planes`, `eliminate_redundant`,
/// `compute_grid_errors`, and `compute_kappa`). Centralising the formula
/// here eliminates the risk of the four call-sites diverging.
//
// Rationale: promoted to `pub(crate)` (with its axis fields) so it remains the
// single grid-formula owner shared by the `tangent` sampling step and the
// `selection` elimination/error steps. Inlining a second copy of the axis
// formula into either caller to avoid the cross-submodule call would let the
// sampling grid and the error grid drift apart — the lint that would force this
// back to private is the price of keeping exactly one owner.
#[allow(clippy::struct_field_names)]
pub(crate) struct GridParams {
    /// Volume axis: `n_volume_points` values from `v_min` to `v_max` (inclusive).
    pub(crate) v_points: Vec<f64>,
    /// Flow axis: `n_flow_points` values from `q_min` to `q_max` (inclusive).
    pub(crate) q_points: Vec<f64>,
    /// Spillage axis: `n_spillage_points` values from `0` to `s_max` (inclusive).
    pub(crate) s_points: Vec<f64>,
}

/// Build the uniform 3D grid for FPHA fitting.
///
/// Constructs three uniform axis vectors that define the grid used consistently
/// across `sample_tangent_planes`, `eliminate_redundant`,
/// `compute_grid_errors`, and `compute_kappa`.
///
/// ## Axis formulas
///
/// - **Volume**: `n_volume_points` values from `bounds.v_min` to `bounds.v_max`.
/// - **Flow**: `n_flow_points` values from `q_min` to `pf.max_turbined_m3s`,
///   where `q_min = max(1.0, pf.max_turbined_m3s * 0.01)`.  The lower bound
///   avoids `q = 0` where the tangent plane is degenerate.
/// - **Spillage**: `n_spillage_points` values from `0.0` to
///   `pf.max_turbined_m3s * 0.5`.  Spillage `s = 0` is always the first point.
///
/// All axes are inclusive at both endpoints.
//
// Rationale: promoted to `pub(crate)` so `super::tangent` and `super::selection`
// call this single definition instead of each holding a private copy of the
// axis formula — see [`GridParams`] for the drift hazard a second copy creates.
pub(crate) fn build_grid(pf: &ProductionFunction, bounds: &FittingBounds) -> GridParams {
    let n_v = bounds.n_volume_points;
    let n_q = bounds.n_flow_points;
    let n_s = bounds.n_spillage_points;

    let v_range = bounds.v_max - bounds.v_min;
    #[allow(clippy::cast_possible_truncation)]
    let v_denom = f64::from((n_v - 1) as u32);

    let q_min = (pf.max_turbined_m3s * 0.01_f64).max(1.0_f64);
    let q_range = pf.max_turbined_m3s - q_min;
    #[allow(clippy::cast_possible_truncation)]
    let q_denom = f64::from((n_q - 1) as u32);

    let s_max = pf.max_turbined_m3s * 0.5_f64;
    #[allow(clippy::cast_possible_truncation)]
    let s_denom = f64::from((n_s - 1) as u32);

    #[allow(clippy::cast_possible_truncation)]
    let v_points: Vec<f64> = (0..n_v)
        .map(|i| bounds.v_min + f64::from(i as u32) * v_range / v_denom)
        .collect();
    #[allow(clippy::cast_possible_truncation)]
    let q_points: Vec<f64> = (0..n_q)
        .map(|j| q_min + f64::from(j as u32) * q_range / q_denom)
        .collect();
    #[allow(clippy::cast_possible_truncation)]
    let s_points: Vec<f64> = (0..n_s)
        .map(|k| f64::from(k as u32) * s_max / s_denom)
        .collect();

    GridParams {
        v_points,
        q_points,
        s_points,
    }
}
