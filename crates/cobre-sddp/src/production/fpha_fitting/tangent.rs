//! Tangent-plane primitive and grid sampling for the FPHA fitting pipeline.
//!
//! Owns [`RawHyperplane`] (an unscaled tangent plane), [`compute_tangent_plane`]
//! (the per-point tangent), and [`sample_tangent_planes`] (the grid sweep that
//! produces the raw candidate set). Sampling reaches the single grid-formula
//! owner via `super::grid::build_grid`.

use super::geometry::FittingBounds;
use super::grid::build_grid;
use super::production::ProductionFunction;

/// An unscaled tangent hyperplane to the production function `phi(v, q, s)`.
///
/// Represents the tangent plane at a specific operating point `(v0, q0, s0)`:
/// `g(v, q, s) = gamma_0 + gamma_v * v + gamma_q * q + gamma_s * s`
///
/// The intercept is NOT scaled by kappa (contrast with `cobre_core::FphaPlane`,
/// where `intercept = gamma_0 * kappa`). By construction, the tangent-point
/// identity holds: `evaluate(v0, q0, s0) == phi(v0, q0, s0)`.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_field_names)]
pub(crate) struct RawHyperplane {
    /// Intercept (NOT scaled by kappa).
    pub gamma_0: f64,
    /// Volume gradient [MW / (hm³)].
    pub gamma_v: f64,
    /// Turbined flow gradient [MW / (m³/s)].
    pub gamma_q: f64,
    /// Spillage flow gradient [MW / (m³/s)].
    pub gamma_s: f64,
}

impl RawHyperplane {
    /// Evaluates the hyperplane at `(v, q, s)`: `gamma_0 + gamma_v*v + gamma_q*q + gamma_s*s`.
    pub(crate) fn evaluate(&self, v: f64, q: f64, s: f64) -> f64 {
        self.gamma_0 + self.gamma_v * v + self.gamma_q * q + self.gamma_s * s
    }
}

/// Computes the tangent hyperplane to `pf` at operating point `(v, q, s)`.
///
/// Returns `None` for degenerate operating points where the tangent plane is
/// not meaningful for the concave envelope:
/// - `q <= 0.0`: zero turbined flow yields zero production.
/// - `phi(v, q, s) <= 0.0`: non-positive production (e.g., net head ≤ 0).
///
/// The returned [`RawHyperplane`] satisfies the tangent-point identity:
/// `plane.evaluate(v, q, s) == pf.evaluate(v, q, s)` exactly (by construction).
///
/// # Parameters
///
/// - `pf` — production function to differentiate.
/// - `v` — reservoir volume \[hm³\].
/// - `q` — turbined flow \[m³/s\].
/// - `s` — spillage flow \[m³/s\].
pub(crate) fn compute_tangent_plane(
    pf: &ProductionFunction,
    v: f64,
    q: f64,
    s: f64,
) -> Option<RawHyperplane> {
    if q <= 0.0 {
        return None;
    }
    let phi_val = pf.evaluate(v, q, s);
    if phi_val <= 0.0 {
        return None;
    }
    let (dv, dq, ds) = pf.partial_derivatives(v, q, s);
    let gamma_0 = phi_val - dv * v - dq * q - ds * s;
    Some(RawHyperplane {
        gamma_0,
        gamma_v: dv,
        gamma_q: dq,
        gamma_s: ds,
    })
}

// ── Grid sampling ─────────────────────────────────────────────────────────────

/// Sample tangent hyperplanes at all points of a uniform 3D grid over `(v, q, s)`.
///
/// Constructs three uniform grids from the bounds provided in `bounds`:
///
/// - **Volume** grid: `n_volume_points` values from `bounds.v_min` to `bounds.v_max`
///   (inclusive endpoints).
/// - **Flow** grid: `n_flow_points` values from `q_min` to `pf.max_turbined_m3s`
///   (inclusive endpoints), where `q_min = max(1.0, pf.max_turbined_m3s * 0.01)`.
///   The lower bound avoids `q = 0` where the tangent plane is degenerate.
/// - **Spillage** grid: `n_spillage_points` values from `0.0` to
///   `pf.max_turbined_m3s * 0.5` (inclusive endpoints). Spillage `s = 0` is
///   always the first grid point.
///
/// For each `(v_i, q_j, s_k)` triple on the grid, calls [`compute_tangent_plane`]
/// and collects all `Some` results. Degenerate operating points (zero flow or
/// non-positive production) are silently dropped.
///
/// # Returns
///
/// A `Vec<RawHyperplane>` of length up to
/// `n_volume_points * n_flow_points * n_spillage_points`.
/// Returns an empty vector if every grid point is degenerate.
///
/// # Parameters
///
/// - `pf` — production function to differentiate.
/// - `bounds` — resolved fitting bounds supplying the volume range and grid counts.
pub(crate) fn sample_tangent_planes(
    pf: &ProductionFunction,
    bounds: &FittingBounds,
) -> Vec<RawHyperplane> {
    let grid = build_grid(pf, bounds);
    let n_v = grid.v_points.len();
    let n_q = grid.q_points.len();
    let n_s = grid.s_points.len();

    let mut planes = Vec::with_capacity(n_v * n_q * n_s);

    for &v in &grid.v_points {
        for &q in &grid.q_points {
            for &s in &grid.s_points {
                if let Some(plane) = compute_tangent_plane(pf, v, q, s) {
                    planes.push(plane);
                }
            }
        }
    }

    planes
}
