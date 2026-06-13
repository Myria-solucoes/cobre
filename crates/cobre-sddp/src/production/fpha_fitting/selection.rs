//! Redundancy elimination, greedy selection, kappa, and validation steps.
//!
//! Owns the back half of the fitting pipeline: [`eliminate_redundant`],
//! [`compute_max_approximation_error`], `compute_grid_errors`, [`select_planes`],
//! [`compute_kappa`], and [`validate_fitted_planes`] (the `γᵥ ≥ 0` / `γ_q ≥ 0` /
//! `γ_s ≤ 0` sign-convention checks). All grid-iterating steps reach the single
//! grid-formula owner via `super::grid::build_grid`.

use super::error::FphaFittingError;
use super::geometry::FittingBounds;
use super::grid::build_grid;
use super::production::ProductionFunction;
use super::tangent::RawHyperplane;

// ── Redundancy elimination ────────────────────────────────────────────────────

/// Remove hyperplanes that are never the tightest bound at any grid point.
///
/// A plane is **active** if there exists at least one point `(v_i, q_j, s_k)` on
/// the same 3D grid used by `sample_tangent_planes` where its value is within
/// `1e-8` of the maximum over all planes at that point.  Planes that are never
/// active are redundant — they are always dominated by some other plane — and are
/// discarded.
///
/// After dominance filtering, near-identical planes (all four coefficients
/// differing by less than `1e-8`) are further deduplicated: only the first
/// occurrence of each unique plane is retained.  This ensures that a linear
/// production function (e.g., constant head with no tailrace) produces exactly
/// one surviving plane rather than many identical copies.
///
/// The grid is reconstructed from `pf` and `bounds` using the identical formula
/// as `sample_tangent_planes`, so the set of test points is consistent with the
/// sampling step.
///
/// # Guarantee
///
/// If `planes` is non-empty, at least one plane always achieves the maximum at
/// some grid point and therefore survives.
///
/// # Returns
///
/// The deduplicated subset of `planes` that are active at least once.  Returns an
/// empty vector if and only if `planes` is empty.
///
/// # Parameters
///
/// - `planes` — candidate hyperplanes (typically produced by `sample_tangent_planes`).
/// - `pf` — production function supplying grid parameters.
/// - `bounds` — resolved fitting bounds supplying the volume range and grid counts.
pub(crate) fn eliminate_redundant(
    planes: &[RawHyperplane],
    pf: &ProductionFunction,
    bounds: &FittingBounds,
) -> Vec<RawHyperplane> {
    if planes.is_empty() {
        return Vec::new();
    }

    let grid = build_grid(pf, bounds);
    let mut active = vec![false; planes.len()];

    for &v in &grid.v_points {
        for &q in &grid.q_points {
            for &s in &grid.s_points {
                // Find the maximum plane value at this grid point.
                let max_val = planes
                    .iter()
                    .map(|p| p.evaluate(v, q, s))
                    .fold(f64::NEG_INFINITY, f64::max);

                // Mark all planes within 1e-8 of the maximum as active.
                for (idx, plane) in planes.iter().enumerate() {
                    if max_val - plane.evaluate(v, q, s) <= 1e-8 {
                        active[idx] = true;
                    }
                }
            }
        }
    }

    let active_planes: Vec<RawHyperplane> = planes
        .iter()
        .zip(active.iter())
        .filter_map(|(p, &is_active)| if is_active { Some(*p) } else { None })
        .collect();

    // Deduplicate near-identical planes (< 1e-8 on all coefficients).
    let mut unique: Vec<RawHyperplane> = Vec::with_capacity(active_planes.len());
    'outer: for candidate in &active_planes {
        for existing in &unique {
            if (candidate.gamma_0 - existing.gamma_0).abs() < 1e-8
                && (candidate.gamma_v - existing.gamma_v).abs() < 1e-8
                && (candidate.gamma_q - existing.gamma_q).abs() < 1e-8
                && (candidate.gamma_s - existing.gamma_s).abs() < 1e-8
            {
                continue 'outer;
            }
        }
        unique.push(*candidate);
    }
    unique
}

// ── Heuristic plane selection ─────────────────────────────────────────────────

/// Compute the maximum approximation error of a hyperplane envelope over the fitting grid.
///
/// For every grid point `(v_i, q_j, s_k)` reconstructed with the same formula as
/// `sample_tangent_planes`, the error at that point is:
///
/// ```text
/// error(v, q, s) = max_m(plane_m(v, q, s)) - phi(v, q, s)
/// ```
///
/// Because the envelope is a concave outer approximation, `error >= 0` everywhere.
/// The returned value is the maximum error over all grid points, i.e., how "loose"
/// the approximation is — lower is better.
///
/// Returns `0.0` when `planes` is empty (no envelope, no error defined).
///
/// # Parameters
///
/// - `planes` — hyperplanes forming the concave envelope.
/// - `pf` — production function used for ground-truth evaluation.
/// - `bounds` — resolved fitting bounds supplying the volume range and grid counts.
///
/// Retained for integration tests that verify approximation quality.
// Rationale: exercised only by integration tests that verify FPHA plane-approximation
// quality against the ground-truth production function; the production fitting path
// selects planes without re-measuring max error, so the dead_code lint fires here.
#[allow(dead_code)]
pub(crate) fn compute_max_approximation_error(
    planes: &[RawHyperplane],
    pf: &ProductionFunction,
    bounds: &FittingBounds,
) -> f64 {
    compute_grid_errors(planes, pf, bounds)
        .into_iter()
        .fold(0.0_f64, f64::max)
}

/// Compute signed per-grid-point approximation errors `envelope(v,q,s) - phi(v,q,s)`.
///
/// Positive values indicate the envelope is loose at that point (correct for an
/// outer approximation).  Negative values indicate a violation (envelope < phi),
/// which can occur when planes were sampled at different operating points and the
/// production function is not globally concave.
///
/// The grid is the same 3D uniform grid used by `sample_tangent_planes` and
/// [`eliminate_redundant`].  Returns a `Vec` of length `n_vol * n_flow * n_spill`.
/// When `planes` is empty, every entry is `f64::NEG_INFINITY`.
fn compute_grid_errors(
    planes: &[RawHyperplane],
    pf: &ProductionFunction,
    bounds: &FittingBounds,
) -> Vec<f64> {
    let grid = build_grid(pf, bounds);
    let n = grid.v_points.len() * grid.q_points.len() * grid.s_points.len();
    let mut errors = Vec::with_capacity(n);

    for &v in &grid.v_points {
        for &q in &grid.q_points {
            for &s in &grid.s_points {
                let phi_val = pf.evaluate(v, q, s);
                let envelope_val = if planes.is_empty() {
                    f64::NEG_INFINITY
                } else {
                    planes
                        .iter()
                        .map(|p| p.evaluate(v, q, s))
                        .fold(f64::NEG_INFINITY, f64::max)
                };
                errors.push(envelope_val - phi_val);
            }
        }
    }

    errors
}

/// Select at most `bounds.max_planes_per_hydro` hyperplanes using a greedy removal heuristic.
///
/// The selection minimises the maximum approximation error (as measured by
/// [`compute_max_approximation_error`]) subject to the cardinality constraint
/// `|result| <= max_planes_per_hydro`.
///
/// ## Algorithm
///
/// 1. **Passthrough**: if `planes.len() <= max_planes_per_hydro`, all planes are
///    returned unchanged.
/// 2. **Greedy removal**: while the current count exceeds the target, evaluate the
///    increase in maximum approximation error that would result from removing each
///    remaining plane, then permanently remove the plane whose removal causes the
///    smallest increase.
///
/// ## Properties
///
/// - The returned planes are a subset of the input.
/// - Returns at most `max_planes_per_hydro` planes.  If removing any further plane
///   would violate the outer-approximation property (minimum grid error would drop
///   below `-1e-8`), the function stops early and may return more planes than the
///   target.
/// - The envelope property is preserved whenever early-stop is not triggered:
///   after selection, `max_m(plane_m(v,q,s)) >= phi(v,q,s)` still holds at every
///   grid point.
/// - Returns an empty `Vec` when `planes` is empty.
///
/// ## Complexity
///
/// The greedy step is O(n² × `grid_size`) where n = `planes.len()` and
/// `grid_size` = `n_vol × n_flow × n_spill`. For n ≤ 40 and `grid_size` = 125 this
/// is ≈ 200 000 evaluations per removal step — negligible for preprocessing.
///
/// # Parameters
///
/// - `planes` — non-redundant candidate hyperplanes (output of [`eliminate_redundant`]).
/// - `pf` — production function used for error evaluation.
/// - `bounds` — resolved fitting bounds; `bounds.max_planes_per_hydro` is the target.
pub(crate) fn select_planes(
    planes: &[RawHyperplane],
    pf: &ProductionFunction,
    bounds: &FittingBounds,
) -> Vec<RawHyperplane> {
    if planes.len() <= bounds.max_planes_per_hydro {
        return planes.to_vec();
    }

    let target = bounds.max_planes_per_hydro;
    let mut current: Vec<RawHyperplane> = planes.to_vec();
    let mut scratch: Vec<RawHyperplane> = Vec::with_capacity(current.len());
    let envelope_tol = -1e-8_f64;

    while current.len() > target {
        let n = current.len();
        let mut best_idx = 0_usize;
        let mut best_is_valid = false;
        let mut best_max_error = f64::INFINITY;

        for remove_idx in 0..n {
            scratch.clear();
            scratch.extend(
                current.iter().enumerate().filter_map(
                    |(i, &p)| {
                        if i == remove_idx { None } else { Some(p) }
                    },
                ),
            );

            let errors = compute_grid_errors(&scratch, pf, bounds);
            let min_err = errors.iter().copied().fold(f64::INFINITY, f64::min);
            let max_err = errors.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let is_valid = min_err >= envelope_tol;
            let max_err_nonneg = max_err.max(0.0);

            let prefer = if is_valid && !best_is_valid {
                true
            } else if is_valid == best_is_valid {
                max_err_nonneg < best_max_error
            } else {
                false
            };

            if prefer {
                best_is_valid = is_valid;
                best_max_error = max_err_nonneg;
                best_idx = remove_idx;
            }
        }

        if !best_is_valid {
            break;
        }

        current.swap_remove(best_idx);
    }

    current
}

// ── Kappa computation ─────────────────────────────────────────────────────────

/// Compute the kappa correction factor for a set of selected hyperplanes.
///
/// Kappa is defined as the minimum over all grid points of the ratio between the
/// exact production value and the maximum hyperplane value at that point:
///
/// ```text
/// kappa = min_{(v_i, q_j, s_k)} { phi(v_i, q_j, s_k) / max_m(plane_m(v_i, q_j, s_k)) }
/// ```
///
/// A kappa of 1.0 means the concave envelope is tight at every grid point.
/// Values less than 1.0 indicate the envelope overestimates the true production
/// at some points; multiplying each intercept by kappa pulls the envelope down
/// to eliminate the overestimation.
///
/// # Grid
///
/// The same 3D grid formula used by `sample_tangent_planes` and
/// [`eliminate_redundant`] is applied here, ensuring consistent coverage.
///
/// # Returns
///
/// The minimum `phi / max_plane` ratio over all grid points where both `phi > 0`
/// and `max_plane > 0`. Returns `1.0` if no such grid point exists (degenerate
/// case where all points have zero production).
///
/// # Parameters
///
/// - `planes` — selected hyperplanes (output of [`select_planes`]).
/// - `pf` — production function used for ground-truth evaluation.
/// - `bounds` — resolved fitting bounds supplying the volume range and grid counts.
pub(crate) fn compute_kappa(
    planes: &[RawHyperplane],
    pf: &ProductionFunction,
    bounds: &FittingBounds,
) -> f64 {
    if planes.is_empty() {
        return 1.0;
    }

    let grid = build_grid(pf, bounds);
    let mut min_ratio = f64::MAX;
    let mut found_valid = false;

    for &v in &grid.v_points {
        for &q in &grid.q_points {
            for &s in &grid.s_points {
                let phi_val = pf.evaluate(v, q, s);
                let max_plane = planes
                    .iter()
                    .map(|p| p.evaluate(v, q, s))
                    .fold(f64::NEG_INFINITY, f64::max);

                if phi_val > 0.0 && max_plane > 0.0 {
                    min_ratio = min_ratio.min(phi_val / max_plane);
                    found_valid = true;
                }
            }
        }
    }

    if found_valid { min_ratio } else { 1.0 }
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Validate a set of selected hyperplanes and their kappa correction factor.
///
/// Checks that:
/// 1. At least one plane exists — zero planes cannot form an LP constraint set.
/// 2. Kappa is in `(0, 1]` — values outside this range indicate a degenerate
///    or overestimating fitting result.
/// 3. Each plane's `gamma_v > -1e-10` (effectively > 0, allowing for rounding).
/// 4. Each plane's `gamma_q > -1e-10` (turbining must have non-negative marginal value).
/// 5. Each plane's `gamma_s <= 1e-10` (spillage must have non-positive marginal value).
///
/// A kappa below 0.95 indicates the envelope overestimates the production function
/// significantly.  The function still returns `Ok(Some(kappa))` in this case —
/// the caller is responsible for surfacing the warning through structured diagnostics.
///
/// # Returns
///
/// - `Ok(None)` — validation passed and kappa >= 0.95 (no warning).
/// - `Ok(Some(kappa))` — validation passed but kappa < 0.95 (low-kappa warning).
/// - `Err(...)` — a hard validation failure.
///
/// # Errors
///
/// | Condition | Error variant |
/// |-----------|---------------|
/// | `planes` is empty | `FphaFittingError::NoHyperplanesProduced` |
/// | `kappa <= 0` or `kappa > 1` | `FphaFittingError::InvalidKappa` |
/// | `gamma_v < -1e-10` for any plane | `FphaFittingError::InvalidCoefficient` |
/// | `gamma_q < -1e-10` for any plane | `FphaFittingError::InvalidCoefficient` |
/// | `gamma_s > 1e-10` for any plane | `FphaFittingError::InvalidCoefficient` |
///
/// # Parameters
///
/// - `planes` — selected hyperplanes after heuristic reduction.
/// - `kappa` — the correction factor computed by [`compute_kappa`].
/// - `hydro_name` — plant name used in error messages.
pub(crate) fn validate_fitted_planes(
    planes: &[RawHyperplane],
    kappa: f64,
    hydro_name: &str,
) -> Result<Option<f64>, FphaFittingError> {
    if planes.is_empty() {
        return Err(FphaFittingError::NoHyperplanesProduced {
            hydro_name: hydro_name.to_owned(),
        });
    }

    if kappa <= 0.0 || kappa > 1.0 {
        return Err(FphaFittingError::InvalidKappa {
            hydro_name: hydro_name.to_owned(),
            kappa,
        });
    }

    let low_kappa = if kappa < 0.95 { Some(kappa) } else { None };

    for (idx, plane) in planes.iter().enumerate() {
        if plane.gamma_v < -1e-10 {
            return Err(FphaFittingError::InvalidCoefficient {
                hydro_name: hydro_name.to_owned(),
                plane_index: idx,
                detail: format!(
                    "gamma_v={:.6e} must be >= 0 (more storage should increase production)",
                    plane.gamma_v
                ),
            });
        }
        if plane.gamma_q < -1e-10 {
            return Err(FphaFittingError::InvalidCoefficient {
                hydro_name: hydro_name.to_owned(),
                plane_index: idx,
                detail: format!(
                    "gamma_q={:.6e} must be >= 0 (turbined flow should produce power)",
                    plane.gamma_q
                ),
            });
        }
        if plane.gamma_s > 1e-10 {
            return Err(FphaFittingError::InvalidCoefficient {
                hydro_name: hydro_name.to_owned(),
                plane_index: idx,
                detail: format!(
                    "gamma_s={:.6e} must be <= 0 (spillage should not increase production)",
                    plane.gamma_s
                ),
            });
        }
    }

    Ok(low_kappa)
}
