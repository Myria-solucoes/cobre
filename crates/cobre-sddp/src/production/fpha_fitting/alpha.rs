//! Least-squares `α_FPHA` correction on the spill = 0 grid.
//!
//! Owns [`compute_alpha_fpha`] and [`scale_plane_affine`]. The raw upper-envelope
//! hull (`FPHA_0`) is optimistic where the true production function is non-concave
//! and pessimistic where it is concave; this bias is corrected with a single
//! scalar `α_FPHA` chosen to minimise the mean-squared error between `α·FPHA_0` and
//! the exact `FPH` over the `(V, Q)` fit grid. The closed form is
//!
//! ```text
//! α_FPHA = ( Σ_i Σ_j FPHA_0(V_i,Q_j)·FPH(V_i,Q_j) ) / ( Σ_i Σ_j FPHA_0(V_i,Q_j)² )
//! FPHA(V,Q) = α_FPHA · FPHA_0(V,Q)
//! ```
//!
//! All grid iteration reaches the single grid-formula owner via
//! `super::grid::build_grid`, so the regression grid never drifts from the cloud
//! and selection grids.

use super::geometry::FittingBounds;
use super::grid::build_grid;
use super::hull_fit::RawPlane;
use super::production::ProductionFunction;

// ── α_FPHA least-squares correction ───────────────────────────────────────────

/// Compute the least-squares `α_FPHA` correction factor for a raw hull envelope.
///
/// Iterates the `(V, Q)` fit grid once, accumulating `Σ FPHA_0·FPH` and
/// `Σ FPHA_0²`, then returns their ratio.
///
/// # Contract — the regression is spillage = 0, lateral = 0 only (Voice 1)
///
/// Both `FPHA_0(V_i,Q_j)` (the pointwise max over the raw hull planes) and
/// `FPH(V_i,Q_j)` (`pf.evaluate(v, q, 0.0)`) are evaluated at **spillage = 0** and
/// lateral = 0 — there is no spill axis in the regression. The wrong-but-compiling
/// alternative is to fold the `build_grid` spillage axis into the sum: that pulls
/// `α` toward the larger-deviation spill region and degrades the dominant no-spill
/// region that governs operation. The `s_points` axis is
/// therefore deliberately ignored here, exactly as `super::hull_fit::build_cloud`
/// ignores it.
///
/// # Contract — degenerate denominator returns the neutral `α = 1.0` (Voice 1)
///
/// When `Σ FPHA_0² == 0` (an all-zero-production hydro, e.g. net head ≤ 0 over the
/// whole grid) the ratio is undefined. This returns the neutral `α = 1.0` rather
/// than dividing by zero or returning a typed error: `α = 1.0` leaves the (already
/// zero) planes unchanged, so a degenerate hydro does not hard-error the fit; the
/// downstream `NoHyperplanesProduced` / sign validation still rejects a genuinely
/// unusable envelope. Returning an error here would abort the whole fitting loop
/// for a hydro whose zero planes are harmless to scale.
///
/// # Parameters
///
/// - `planes` — raw hull planes (`FPHA_0`), pre-scaling.
/// - `pf` — production function supplying `FPH` and the grid axes.
/// - `bounds` — resolved fitting bounds supplying the volume range and grid counts.
pub(crate) fn compute_alpha_fpha(
    planes: &[RawPlane],
    pf: &ProductionFunction,
    bounds: &FittingBounds,
) -> f64 {
    if planes.is_empty() {
        return 1.0;
    }

    let grid = build_grid(pf, bounds);

    let mut sum_fpha0_fph = 0.0_f64;
    let mut sum_fpha0_sq = 0.0_f64;

    // Spillage = 0, lateral = 0: the s axis is intentionally not iterated (see the
    // spillage-only contract above). `RawPlane::evaluate(v, q, 0.0)` and
    // `pf.evaluate_capped(v, q, 0.0)` both pin spillage to 0. The regression uses
    // the capped output — the same the cloud hull was built from — so the α scale
    // balances the envelope against the model it actually approximates.
    for &v in &grid.v_points {
        for &q in &grid.q_points {
            let fpha0 = planes
                .iter()
                .map(|p| p.evaluate(v, q, 0.0))
                .fold(f64::NEG_INFINITY, f64::max);
            let fph = pf.evaluate_capped(v, q, 0.0);
            sum_fpha0_fph += fpha0 * fph;
            sum_fpha0_sq += fpha0 * fpha0;
        }
    }

    if sum_fpha0_sq == 0.0 {
        1.0
    } else {
        sum_fpha0_fph / sum_fpha0_sq
    }
}

/// Scale the whole affine function of a raw hull plane by `α`.
///
/// # Contract — `α` scales every coefficient, not just the intercept (Voice 1)
///
/// `FPHA(V,Q) = α·FPHA_0(V,Q) = α·γ₀ + α·γ_V·V + α·γ_Q·Q (+ α·γ_S)`, so `α`
/// multiplies `γ₀`, `γ_V`, `γ_Q`, and the provisional `γ_S = 0` alike. The
/// wrong-but-compiling alternative is to scale only the intercept (`α·γ₀` with raw
/// gradients) — that was the role of the old `kappa` shrink and is NOT equivalent
/// to scaling the affine function. This whole-affine scaling is exactly why
/// `α ≠ kappa`.
pub(crate) fn scale_plane_affine(plane: &RawPlane, alpha: f64) -> RawPlane {
    RawPlane {
        gamma_0: plane.gamma_0 * alpha,
        gamma_v: plane.gamma_v * alpha,
        gamma_q: plane.gamma_q * alpha,
        gamma_s: plane.gamma_s * alpha,
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::similar_names
)]
mod tests {
    use cobre_core::{EfficiencyModel, EntityId, HydraulicLossesModel, TailraceModel};
    use cobre_io::extensions::HydroGeometryRow;

    use super::super::geometry::{FittingBounds, ForebayTable};
    use super::super::hull_fit::RawPlane;
    use super::super::production::{ProductionFunction, TailraceSource};
    use super::{compute_alpha_fpha, scale_plane_affine};

    fn row(volume_hm3: f64, height_m: f64) -> HydroGeometryRow {
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3,
            height_m,
            area_km2: 0.0,
        }
    }

    /// A 2×2 `(V, Q)` grid over `[0, 30_000]` hm³. Spillage axis count is 2 but is
    /// never iterated by `compute_alpha_fpha`.
    fn small_bounds() -> FittingBounds {
        FittingBounds {
            v_min: 0.0,
            v_max: 30_000.0,
            n_volume_points: 2,
            n_flow_points: 2,
            n_spillage_points: 2,
            max_planes_per_hydro: 10,
            single_volume: false,
        }
    }

    /// A strictly concave production function (rising-then-flattening forebay plus
    /// a head-dropping tailrace), so the hull yields several distinct facets.
    fn concave_production_function() -> ProductionFunction {
        let rows = vec![
            row(0.0, 380.0),
            row(10_000.0, 396.0),
            row(20_000.0, 404.0),
            row(30_000.0, 408.0),
        ];
        let forebay = ForebayTable::new(&rows, "Concave").expect("valid VHA curve");
        let tailrace = TailraceModel::Polynomial {
            coefficients: vec![0.0, 0.0008, -2e-8],
        };
        ProductionFunction::new(
            forebay,
            TailraceSource::Entity(Some(tailrace)),
            Some(&HydraulicLossesModel::Constant { value_m: 2.0 }),
            Some(&EfficiencyModel::Constant { value: 0.92 }),
            3_000.0,
            "Concave".to_owned(),
        )
    }

    /// A flat-head production function whose net head is forced to 0 everywhere:
    /// the forebay sits below the constant tailrace, so `pf.evaluate` clamps to 0
    /// at every grid point. `FPHA_0` over an all-zero cloud is 0, exercising the
    /// degenerate `Σ FPHA_0² == 0` denominator.
    fn zero_production_function() -> ProductionFunction {
        let rows = vec![row(0.0, 100.0), row(30_000.0, 100.0)];
        let forebay = ForebayTable::new(&rows, "Zero").expect("valid VHA curve");
        // Constant tailrace at 200 m > forebay 100 m → gross head negative → net
        // head clamped to 0 → production 0 everywhere.
        let tailrace = TailraceModel::Polynomial {
            coefficients: vec![200.0],
        };
        ProductionFunction::new(
            forebay,
            TailraceSource::Entity(Some(tailrace)),
            None,
            None,
            3_000.0,
            "Zero".to_owned(),
        )
    }

    /// Closed-form α on a hand-computable grid matches `Σ FPHA_0·FPH / Σ FPHA_0²`.
    ///
    /// Uses a single raw plane so `FPHA_0(V,Q)` is exactly that plane, making the
    /// reference sum trivial to recompute independently of `compute_alpha_fpha`.
    #[test]
    fn closed_form_alpha_matches_hand_computed_ratio() {
        let pf = concave_production_function();
        let bounds = small_bounds();

        // One raw plane: FPHA_0 = gamma_0 + gamma_v*V + gamma_q*Q.
        let planes = vec![RawPlane {
            gamma_0: 50.0,
            gamma_v: 0.01,
            gamma_q: 0.3,
            gamma_s: 0.0,
        }];

        // Independent reference accumulation over the same (V, Q) grid at s = 0.
        // The flow axis runs from 0 to max_turbined; the regression uses the capped
        // output, mirroring `compute_alpha_fpha`.
        let v_pts = [0.0_f64, 30_000.0];
        let q_pts = [0.0_f64, pf.max_turbined_m3s];
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for &v in &v_pts {
            for &q in &q_pts {
                let fpha0 = planes[0].evaluate(v, q, 0.0);
                let fph = pf.evaluate_capped(v, q, 0.0);
                num += fpha0 * fph;
                den += fpha0 * fpha0;
            }
        }
        let expected = num / den;

        let alpha = compute_alpha_fpha(&planes, &pf, &bounds);
        assert!(
            (alpha - expected).abs() < 1e-12,
            "alpha={alpha} must match hand-computed ratio {expected}"
        );
    }

    /// The regression evaluates both `FPHA_0` and `FPH` at spillage = 0: feeding a
    /// plane with a non-zero `gamma_s` must not change α (the s axis is not summed).
    #[test]
    fn regression_is_spillage_zero_only() {
        let pf = concave_production_function();
        let bounds = small_bounds();

        let base = RawPlane {
            gamma_0: 50.0,
            gamma_v: 0.01,
            gamma_q: 0.3,
            gamma_s: 0.0,
        };
        // Same plane but with a spurious spill gradient. Because the regression
        // pins s = 0, evaluate(v, q, 0.0) ignores gamma_s, so α is unchanged.
        let with_spill = RawPlane {
            gamma_s: -0.5,
            ..base
        };

        let alpha_base = compute_alpha_fpha(&[base], &pf, &bounds);
        let alpha_spill = compute_alpha_fpha(&[with_spill], &pf, &bounds);
        assert_eq!(
            alpha_base, alpha_spill,
            "alpha must be independent of gamma_s (regression is spillage = 0)"
        );

        // And FPH is the s = 0 term: this production function's tailrace rises
        // with total outflow, so evaluating at s > 0 lowers net head and yields a
        // strictly smaller FPH than at s = 0. The regression uses pf.evaluate(.,.,0),
        // never the s > 0 value — confirming spillage is pinned to 0 in FPH too.
        let v = 30_000.0_f64;
        let q = pf.max_turbined_m3s;
        let fph_s0 = pf.evaluate(v, q, 0.0);
        let fph_s_pos = pf.evaluate(v, q, 500.0);
        assert!(
            fph_s_pos < fph_s0,
            "spillage must lower FPH (s>0 head drop); regression must use the s=0 value"
        );
    }

    /// `scale_plane_affine` scales the whole affine function: the scaled plane
    /// equals `α·raw_plane(V,Q)` at every probe point, intercept and gradients alike.
    #[test]
    fn scaling_is_full_affine() {
        let raw = RawPlane {
            gamma_0: 123.0,
            gamma_v: 0.04,
            gamma_q: 0.7,
            gamma_s: -0.2,
        };
        let alpha = 0.83_f64;
        let scaled = scale_plane_affine(&raw, alpha);

        // Every coefficient scaled.
        assert!((scaled.gamma_0 - alpha * raw.gamma_0).abs() < 1e-12);
        assert!((scaled.gamma_v - alpha * raw.gamma_v).abs() < 1e-12);
        assert!((scaled.gamma_q - alpha * raw.gamma_q).abs() < 1e-12);
        assert!((scaled.gamma_s - alpha * raw.gamma_s).abs() < 1e-12);

        // The whole affine function scales, not just the intercept.
        for &(v, q, s) in &[
            (0.0, 1.0, 0.0),
            (10_000.0, 500.0, 30.0),
            (30_000.0, 0.0, 0.0),
        ] {
            let scaled_val = scaled.evaluate(v, q, s);
            let alpha_raw_val = alpha * raw.evaluate(v, q, s);
            assert!(
                (scaled_val - alpha_raw_val).abs() < 1e-12,
                "scaled_plane({v},{q},{s})={scaled_val} must equal alpha*raw={alpha_raw_val}"
            );
        }
    }

    /// All-zero production → `Σ FPHA_0² == 0` → neutral `α = 1.0`, no panic.
    #[test]
    fn degenerate_denominator_returns_neutral_alpha() {
        let pf = zero_production_function();
        let bounds = small_bounds();

        // All-zero raw planes (an all-zero-production hull). FPHA_0 = 0 everywhere,
        // so Σ FPHA_0² = 0 — the degenerate denominator.
        let planes = vec![RawPlane {
            gamma_0: 0.0,
            gamma_v: 0.0,
            gamma_q: 0.0,
            gamma_s: 0.0,
        }];

        let alpha = compute_alpha_fpha(&planes, &pf, &bounds);
        assert_eq!(
            alpha, 1.0,
            "all-zero production must yield the neutral alpha = 1.0"
        );
    }

    /// Empty plane set short-circuits to the neutral α = 1.0.
    #[test]
    fn empty_planes_returns_neutral_alpha() {
        let pf = concave_production_function();
        let bounds = small_bounds();
        let alpha = compute_alpha_fpha(&[], &pf, &bounds);
        assert_eq!(alpha, 1.0, "empty plane set must yield neutral alpha = 1.0");
    }
}
