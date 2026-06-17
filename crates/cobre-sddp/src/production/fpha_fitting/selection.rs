//! Coefficient-sign and `α_FPHA > 0` validation of the fitted plane set.
//!
//! Owns [`validate_fitted_planes`], the final gate of the fitting pipeline: it
//! rejects an empty plane set, a non-positive `α_FPHA`, and any plane whose
//! coefficient signs violate the physical contract (`γ_v ≥ 0`, `γ_q ≥ 0`,
//! `γ_s ≤ 0`).

use super::error::FphaFittingError;
use super::hull_fit::RawPlane;

/// Validate a fitted plane set and its `α_FPHA` correction factor.
///
/// Checks that:
/// 1. At least one plane exists — zero planes cannot form an LP constraint set.
/// 2. `α_FPHA > 0` — the least-squares correction must be strictly positive; a
///    non-positive `α` would flip every coefficient sign or zero the envelope.
/// 3. Each plane's `gamma_v ≥ -1e-10` (effectively ≥ 0, allowing for rounding).
/// 4. Each plane's `gamma_q ≥ -1e-10` (turbining must have non-negative marginal value).
/// 5. Each plane's `gamma_s ≤ 1e-10` (spillage must have non-positive marginal value).
///
/// # Errors
///
/// | Condition | Error variant |
/// |-----------|---------------|
/// | `planes` is empty | [`FphaFittingError::NoHyperplanesProduced`] |
/// | `alpha <= 0` | [`FphaFittingError::NonPositiveAlpha`] |
/// | `gamma_v < -1e-10` for any plane | [`FphaFittingError::InvalidCoefficient`] |
/// | `gamma_q < -1e-10` for any plane | [`FphaFittingError::InvalidCoefficient`] |
/// | `gamma_s > 1e-10` for any plane | [`FphaFittingError::InvalidCoefficient`] |
///
/// # Parameters
///
/// - `planes` — the α-scaled, `γ_S`-fitted planes.
/// - `alpha` — the least-squares `α_FPHA` correction factor applied to the planes.
/// - `hydro_name` — plant name used in error messages.
pub(crate) fn validate_fitted_planes(
    planes: &[RawPlane],
    alpha: f64,
    hydro_name: &str,
) -> Result<(), FphaFittingError> {
    if planes.is_empty() {
        return Err(FphaFittingError::NoHyperplanesProduced {
            hydro_name: hydro_name.to_owned(),
        });
    }

    if alpha <= 0.0 {
        return Err(FphaFittingError::NonPositiveAlpha {
            hydro_name: hydro_name.to_owned(),
            alpha,
        });
    }

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

    Ok(())
}
