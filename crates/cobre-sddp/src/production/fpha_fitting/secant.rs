//! Per-plane lateral-flow secant `γ_S` on the `Q_lat` axis.
//!
//! Owns [`resolve_s_max`], [`representative_operating_point`], and
//! [`fit_gamma_s`]. The fitter adds one extra FPHA axis — the **lateral
//! flow `Q_lat`**, aggregating everything that raises the tailrace without
//! driving generation — and fits a single secant `γ_S` per plane by
//! least-squares over `Q_lat ∈ [0, S_max]`.
//!
//! # The smart default (`Q_lat = S`, `Q̂_lat = 0`)
//!
//! With smart defaults the lateral axis collapses to own spill: `Q_lat = S` and
//! the reference offset `Q̂_lat = 0`. The richer `Q_lat` composition (upstream
//! defluences, post inflows) and its LP wiring are a later refinement; here only
//! the secant axis and its default land. The tailrace already sees total outflow
//! `q + s` (`super::geometry::evaluate_tailrace` is called at `q_out = q + s` by
//! `ProductionFunction::net_head`), so evaluating `pf.evaluate(v, q, s = Q_lat)`
//! makes the tailrace see `Q_jus = q + Q_lat` — exactly the spill-only default.
//!
//! # Secant slope (1-D least-squares, not a hull axis)
//!
//! The slope is a per-plane 1-D ordinary-least-squares regression of `GH` against
//! `Q_lat` over a FIXED evenly-spaced sample of `Q_lat ∈ [0, S_max]`:
//!
//! ```text
//! γ_S = Σ_i (x_i − x̄)(y_i − ȳ) / Σ_i (x_i − x̄)²
//! ```
//!
//! where `x_i` are the `Q_lat` samples and `y_i = GH(V_k, q_k, Q_lat_i)` at the
//! plane's representative operating point. This is NOT a spillage hull/grid axis —
//! it is a 1-D fit per plane, deterministic and order-independent (the sample is a
//! closed-form evenly-spaced grid, so the result does not depend on entity or
//! iteration order).

use super::geometry::FittingBounds;
use super::hull_fit::RawPlane;
use super::production::ProductionFunction;

/// Number of evenly-spaced `Q_lat` samples used for the per-plane secant fit.
///
/// Fixed (not config-driven) so the secant is deterministic and order-independent:
/// the sample is a closed-form grid over `[0, S_max]`, identical for every plane
/// and every run regardless of input ordering.
const N_QLAT_SAMPLES: usize = 9;

/// Resolve the lateral-flow sample upper bound `S_max` for the secant fit.
///
/// `S_max = 2·MLT` when the long-term mean inflow `MLT > 0`, else
/// `2 × max_turbined_m3s`.
///
/// # Contract — `MLT > 0` strictly, else fall back (Voice 1)
///
/// The `MLT > 0` test is strict: a hydro with no inflow history yields `mlt = 0.0`
/// (the canonical-order mean of an empty series) and MUST take the
/// `2 × max_turbined_m3s` fallback. Using `mlt >= 0.0` instead would collapse the
/// fallback to `S_max = 0` for a history-less hydro and zero out every `γ_S`
/// (the degenerate guard in [`fit_gamma_s`]), silently dropping the lateral
/// correction. The caller (`super::fit_fpha_planes`) supplies the per-hydro
/// long-term mean natural inflow as `mlt`.
pub(crate) fn resolve_s_max(mlt: f64, max_turbined_m3s: f64) -> f64 {
    if mlt > 0.0 {
        2.0 * mlt
    } else {
        2.0 * max_turbined_m3s
    }
}

/// Representative operating point `(V, q)` at which a plane's `γ_S` secant is fit.
///
/// # Rationale — the envelope-max grid point is the representative point (Voice 2)
///
/// The secant slope `dGH/dQ_lat = ρ·q·d(h_net)/dQ_lat` depends on `(V, q)`, so each
/// plane needs one representative operating point. The chosen point is the
/// spill = 0 fit-grid point where this plane is the *active* (tightest) upper bound
/// and attains the largest `GH` among the points it dominates — i.e. the operating
/// region the plane actually governs. Fitting `γ_S` there makes the lateral
/// correction match where the plane is used. A fixed `(V_ref, q_max)` corner would
/// be defensible but mismatched for planes that govern a low-flow or low-volume
/// region; the active-point choice tracks each plane's own region.
///
/// Determinism: the grid is the closed-form `super::grid::build_grid` grid and the
/// scan is in fixed `(v, q)` order with a strict `>` improvement test, so ties
/// resolve to the first point in iteration order — order-independent across runs.
///
/// Returns `None` when the plane is never the active bound on the grid (e.g. a
/// fully dominated plane); the caller then leaves `γ_S = 0` for that plane.
fn representative_operating_point(
    plane: &RawPlane,
    planes: &[RawPlane],
    pf: &ProductionFunction,
    bounds: &FittingBounds,
) -> Option<(f64, f64)> {
    let grid = super::grid::build_grid(pf, bounds);

    let mut best: Option<(f64, f64, f64)> = None; // (v, q, gh) at the active max
    for &v in &grid.v_points {
        for &q in &grid.q_points {
            // Spill = 0: the representative point is on the no-lateral surface; the
            // secant then sweeps Q_lat away from this anchor.
            let this_val = plane.evaluate(v, q, 0.0);
            let max_val = planes
                .iter()
                .map(|p| p.evaluate(v, q, 0.0))
                .fold(f64::NEG_INFINITY, f64::max);

            // Active iff within the same tolerance redundancy elimination uses.
            if max_val - this_val <= 1e-8 {
                let gh = pf.evaluate(v, q, 0.0);
                if best.is_none_or(|(_, _, best_gh)| gh > best_gh) {
                    best = Some((v, q, gh));
                }
            }
        }
    }

    best.map(|(v, q, _)| (v, q))
}

/// Fit the lateral-flow secant `γ_S` for a single plane at `(v_rep, q_rep)`.
///
/// Sweeps a FIXED evenly-spaced sample of `Q_lat ∈ [0, S_max]` (`N_QLAT_SAMPLES`
/// points, inclusive endpoints), evaluates `GH = pf.evaluate(v_rep, q_rep, Q_lat)`
/// at each, and returns the ordinary-least-squares slope of `GH` vs `Q_lat`.
///
/// # Contract — `γ_S ≤ 0` (Voice 1)
///
/// More lateral flow raises the tailrace, lowers net head, and lowers `GH`, so the
/// fitted slope is non-positive. A tiny positive slope from floating-point noise is
/// clamped to `0.0` so every emitted plane satisfies the
/// `validate_fitted_planes` rule `gamma_s ≤ 1e-10`. The wrong-but-compiling
/// alternative — returning the raw slope — would let a `+1e-15` round-off bypass
/// the sign contract and surface as an invalid coefficient downstream.
///
/// # Contract — degenerate `S_max ≤ 0` returns the neutral `γ_S = 0` (Voice 1)
///
/// When `S_max ≤ 0` (MLT = 0 AND `max_turbined` = 0) the sample collapses to a single
/// point and `Σ(x_i − x̄)² = 0` — the regression slope is undefined. This returns
/// `0.0` (no lateral sensitivity) rather than dividing by zero. The same neutral
/// path covers a flat `GH` (`Σ(x_i − x̄)(y_i − ȳ) = 0`).
fn fit_gamma_s(pf: &ProductionFunction, v_rep: f64, q_rep: f64, s_max: f64) -> f64 {
    if s_max <= 0.0 {
        return 0.0;
    }

    // Evenly-spaced Q_lat sample over [0, S_max], inclusive endpoints. The first
    // sample (Q_lat = 0) is the no-lateral anchor; Q̂_lat = 0 is the reference.
    #[allow(clippy::cast_possible_truncation)]
    let denom = f64::from((N_QLAT_SAMPLES - 1) as u32);

    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    for i in 0..N_QLAT_SAMPLES {
        #[allow(clippy::cast_possible_truncation)]
        let q_lat = f64::from(i as u32) * s_max / denom;
        let gh = pf.evaluate(v_rep, q_rep, q_lat);
        sum_x += q_lat;
        sum_y += gh;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = N_QLAT_SAMPLES as f64;
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    let mut cov = 0.0_f64;
    let mut var_x = 0.0_f64;
    for i in 0..N_QLAT_SAMPLES {
        #[allow(clippy::cast_possible_truncation)]
        let q_lat = f64::from(i as u32) * s_max / denom;
        let gh = pf.evaluate(v_rep, q_rep, q_lat);
        let dx = q_lat - mean_x;
        cov += dx * (gh - mean_y);
        var_x += dx * dx;
    }

    if var_x == 0.0 {
        return 0.0;
    }

    let slope = cov / var_x;
    // Clamp tiny positive round-off to 0 to uphold the γ_S ≤ 0 sign contract.
    slope.min(0.0)
}

/// Fit `γ_S` for every plane in place, using the smart default `Q_lat = S`.
///
/// For each α-scaled plane, resolves a representative operating point and fits the
/// lateral secant over `Q_lat ∈ [0, S_max]`, writing the resulting non-positive
/// slope onto `plane.gamma_s`. The α-scaled gradients (`gamma_v`, `gamma_q`) and
/// intercept are untouched — `γ_S` is the only coefficient this step sets.
///
/// The secant is fit for every plant: spillage raises the tailrace and lowers net
/// head universally, so there is no per-plant opt-out. The degenerate `S_max ≤ 0`
/// guard in [`fit_gamma_s`] is the only path that yields `γ_S = 0`.
///
/// # Parameters
///
/// - `planes` — the α-scaled selected planes; mutated in place to carry `γ_S`.
/// - `pf` — production function supplying `GH(V, q, Q_lat)` via the tailrace.
/// - `bounds` — resolved fitting bounds supplying the representative-point grid.
/// - `mlt` — long-term mean inflow \[m³/s\] for `S_max`; `0.0` (no inflow history)
///   selects the `2 × max_turbined` fallback (see [`resolve_s_max`]).
pub(crate) fn fit_gamma_s_for_planes(
    planes: &mut [RawPlane],
    pf: &ProductionFunction,
    bounds: &FittingBounds,
    mlt: f64,
) {
    let s_max = resolve_s_max(mlt, pf.max_turbined_m3s);

    // Snapshot the envelope before mutation: the representative-point scan reads
    // the spill = 0 plane values, which γ_S does not affect, so reading the live
    // slice would be equivalent — the snapshot makes the read-vs-write separation
    // explicit and keeps the active-point selection independent of fit order.
    let snapshot: Vec<RawPlane> = planes.to_vec();
    for plane in planes.iter_mut() {
        plane.gamma_s = match representative_operating_point(plane, &snapshot, pf, bounds) {
            Some((v_rep, q_rep)) => fit_gamma_s(pf, v_rep, q_rep, s_max),
            None => 0.0,
        };
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::doc_markdown
)]
mod tests {
    use cobre_core::{EfficiencyModel, EntityId, HydraulicLossesModel, TailraceModel};
    use cobre_io::extensions::HydroGeometryRow;

    use super::super::geometry::{FittingBounds, ForebayTable};
    use super::super::hull_fit::RawPlane;
    use super::super::production::ProductionFunction;
    use super::{
        N_QLAT_SAMPLES, fit_gamma_s_for_planes, representative_operating_point, resolve_s_max,
    };

    fn row(volume_hm3: f64, height_m: f64) -> HydroGeometryRow {
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3,
            height_m,
            area_km2: 0.0,
        }
    }

    fn small_bounds() -> FittingBounds {
        FittingBounds {
            v_min: 0.0,
            v_max: 30_000.0,
            n_volume_points: 3,
            n_flow_points: 3,
            n_spillage_points: 2,
            max_planes_per_hydro: 10,
            single_volume: false,
        }
    }

    /// A production function whose tailrace rises with total outflow, so spillage /
    /// lateral flow lowers net head and `GH`. Constant losses keep `d(h_net)/dQ_lat`
    /// equal to `−dh_tail/dQ_out`.
    fn spill_sensitive_pf() -> ProductionFunction {
        let rows = vec![
            row(0.0, 380.0),
            row(10_000.0, 396.0),
            row(20_000.0, 404.0),
            row(30_000.0, 408.0),
        ];
        let forebay = ForebayTable::new(&rows, "SpillSensitive").expect("valid VHA curve");
        // Linear tailrace: h_tail(q_out) = 5.0 + 0.001 * q_out, so dh_tail/dq_out is
        // the constant 0.001 — a hand-computable secant slope.
        let tailrace = TailraceModel::Polynomial {
            coefficients: vec![5.0, 0.001],
        };
        ProductionFunction::new(
            forebay,
            Some(&tailrace),
            Some(&HydraulicLossesModel::Constant { value_m: 0.0 }),
            Some(&EfficiencyModel::Constant { value: 0.9 }),
            3_000.0,
            "SpillSensitive".to_owned(),
        )
    }

    /// A flat-tailrace production function: net head is independent of outflow, so
    /// `GH` is flat in `Q_lat` and the secant slope is exactly 0.
    fn flat_tailrace_pf() -> ProductionFunction {
        let rows = vec![row(0.0, 400.0), row(30_000.0, 420.0)];
        let forebay = ForebayTable::new(&rows, "Flat").expect("valid VHA curve");
        let tailrace = TailraceModel::Polynomial {
            coefficients: vec![10.0],
        };
        ProductionFunction::new(
            forebay,
            Some(&tailrace),
            None,
            Some(&EfficiencyModel::Constant { value: 1.0 }),
            3_000.0,
            "Flat".to_owned(),
        )
    }

    /// `resolve_s_max`: MLT > 0 ⇒ 2·MLT; MLT = 0 ⇒ 2·max_turbined (both branches).
    #[test]
    fn s_max_resolution_picks_mlt_then_falls_back() {
        // MLT > 0 branch.
        assert_eq!(resolve_s_max(1_500.0, 3_000.0), 3_000.0);
        // MLT = 0 fallback branch.
        assert_eq!(resolve_s_max(0.0, 3_000.0), 6_000.0);
        // Negative/garbage MLT also falls back (treated as not-supplied).
        assert_eq!(resolve_s_max(-1.0, 2_000.0), 4_000.0);
    }

    /// The fitted `γ_S` equals the hand-computed OLS secant slope of `GH` vs
    /// `Q_lat` over `[0, S_max]` to within 1e-10, and is strictly negative.
    #[test]
    fn gamma_s_matches_hand_computed_ols_slope() {
        let pf = spill_sensitive_pf();
        let bounds = small_bounds();

        // Single plane → it is the active bound everywhere, so the representative
        // point is the grid point with the largest GH at s = 0.
        let mut planes = vec![RawPlane {
            gamma_0: 0.0,
            gamma_v: 0.001,
            gamma_q: 0.1,
            gamma_s: 0.0,
        }];

        let (v_rep, q_rep) = representative_operating_point(&planes[0], &planes, &pf, &bounds)
            .expect("active plane");

        // MLT = 0 → S_max = 2 * 3000 = 6000.
        let s_max = resolve_s_max(0.0, pf.max_turbined_m3s);
        assert_eq!(s_max, 6_000.0);

        // Independent OLS slope over the same fixed Q_lat sample.
        let denom = (N_QLAT_SAMPLES - 1) as f64;
        let xs: Vec<f64> = (0..N_QLAT_SAMPLES)
            .map(|i| i as f64 * s_max / denom)
            .collect();
        let ys: Vec<f64> = xs.iter().map(|&x| pf.evaluate(v_rep, q_rep, x)).collect();
        let n = N_QLAT_SAMPLES as f64;
        let mean_x = xs.iter().sum::<f64>() / n;
        let mean_y = ys.iter().sum::<f64>() / n;
        let cov: f64 = xs
            .iter()
            .zip(&ys)
            .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
            .sum();
        let var_x: f64 = xs.iter().map(|&x| (x - mean_x).powi(2)).sum();
        let expected = cov / var_x;

        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, 0.0);

        assert!(
            (planes[0].gamma_s - expected).abs() < 1e-10,
            "gamma_s={} must equal hand-computed OLS slope {expected}",
            planes[0].gamma_s
        );
        assert!(
            planes[0].gamma_s < 0.0,
            "rising tailrace must yield a strictly negative gamma_s, got {}",
            planes[0].gamma_s
        );
    }

    /// Default axis is own-spill only: the secant uses `Q_lat = S` (the spillage
    /// argument of `evaluate`) with `Q̂_lat = 0` (the Q_lat = 0 anchor). Feeding the
    /// lateral flow as the spillage argument is exactly what the tailrace sees as
    /// `q + Q_lat`, so a rising tailrace yields a negative slope.
    #[test]
    fn default_axis_is_own_spill_qhat_zero() {
        let pf = spill_sensitive_pf();
        let bounds = small_bounds();
        let mut planes = vec![RawPlane {
            gamma_0: 0.0,
            gamma_v: 0.001,
            gamma_q: 0.1,
            gamma_s: 0.0,
        }];

        // Pick the representative point and confirm GH at Q_lat = 0 (the anchor,
        // Q̂_lat = 0) exceeds GH at Q_lat = S_max: the axis is own-spill on the
        // tailrace (q + Q_lat), so more lateral lowers GH.
        let (v_rep, q_rep) = representative_operating_point(&planes[0], &planes, &pf, &bounds)
            .expect("active plane");
        let s_max = resolve_s_max(0.0, pf.max_turbined_m3s);
        let gh_anchor = pf.evaluate(v_rep, q_rep, 0.0);
        let gh_smax = pf.evaluate(v_rep, q_rep, s_max);
        assert!(
            gh_smax < gh_anchor,
            "own-spill axis: GH at Q_lat=S_max must be below the Q̂_lat=0 anchor"
        );

        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, 0.0);
        assert!(planes[0].gamma_s < 0.0);
    }

    /// Flat tailrace → flat `GH` in `Q_lat` → secant slope exactly 0 (the neutral
    /// path for `Σ(x−x̄)(y−ȳ) = 0`).
    #[test]
    fn flat_tailrace_yields_zero_slope() {
        let pf = flat_tailrace_pf();
        let bounds = small_bounds();
        let mut planes = vec![RawPlane {
            gamma_0: 0.0,
            gamma_v: 0.0005,
            gamma_q: 0.2,
            gamma_s: 0.0,
        }];
        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, 0.0);
        assert_eq!(
            planes[0].gamma_s, 0.0,
            "flat tailrace must yield gamma_s = 0"
        );
    }

    /// Degenerate `S_max == 0` (MLT = 0 AND max_turbined = 0) → neutral `γ_S = 0`,
    /// no divide-by-zero.
    #[test]
    fn degenerate_s_max_zero_is_neutral() {
        let rows = vec![row(0.0, 400.0), row(30_000.0, 420.0)];
        let forebay = ForebayTable::new(&rows, "ZeroTurb").expect("valid VHA curve");
        let tailrace = TailraceModel::Polynomial {
            coefficients: vec![5.0, 0.001],
        };
        // max_turbined = 0 → with MLT = 0, S_max = 0.
        let pf = ProductionFunction::new(
            forebay,
            Some(&tailrace),
            None,
            Some(&EfficiencyModel::Constant { value: 0.9 }),
            0.0,
            "ZeroTurb".to_owned(),
        );
        assert_eq!(resolve_s_max(0.0, pf.max_turbined_m3s), 0.0);

        let bounds = small_bounds();
        let mut planes = vec![RawPlane {
            gamma_0: 0.0,
            gamma_v: 0.001,
            gamma_q: 0.1,
            gamma_s: 0.0,
        }];
        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, 0.0);
        assert_eq!(
            planes[0].gamma_s, 0.0,
            "S_max = 0 must yield neutral gamma_s"
        );
    }

    /// Every fitted plane satisfies the `γ_s ≤ 1e-10` sign rule on a multi-plane,
    /// spill-sensitive fixture (the validation contract `validate_fitted_planes`
    /// enforces downstream).
    #[test]
    fn all_planes_satisfy_sign_rule() {
        let pf = spill_sensitive_pf();
        let bounds = small_bounds();
        let mut planes = vec![
            RawPlane {
                gamma_0: 0.0,
                gamma_v: 0.001,
                gamma_q: 0.12,
                gamma_s: 0.0,
            },
            RawPlane {
                gamma_0: 50.0,
                gamma_v: 0.0008,
                gamma_q: 0.08,
                gamma_s: 0.0,
            },
        ];
        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, 0.0);
        for plane in &planes {
            assert!(
                plane.gamma_s <= 1e-10,
                "every plane must satisfy gamma_s <= 1e-10, got {}",
                plane.gamma_s
            );
        }
    }
}
