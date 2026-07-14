//! Validation of PAR model parameters.
//!
//! Fatal failures return [`StochasticError::InvalidParParameters`]; non-fatal
//! issues accumulate as [`ParWarning`]s in a [`ParValidationReport`]. See
//! [`validate_par_parameters`] for the checks performed.
//!
//! [`StochasticError::InvalidParParameters`]: crate::StochasticError::InvalidParParameters

use cobre_core::InflowModel;

use crate::StochasticError;

// ---------------------------------------------------------------------------
// ParValidationReport
// ---------------------------------------------------------------------------

/// Result of PAR parameter validation: the accumulated non-fatal warnings.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::InflowModel};
/// use cobre_stochastic::par::validation::validate_par_parameters;
///
/// let model = InflowModel {
///     hydro_id: EntityId(1),
///     stage_id: 3,
///     mean_m3s: 100.0,
///     std_m3s: 30.0,
///     ar_coefficients: vec![0.3],
///     residual_std_ratio: 0.954,
///     annual: None,
/// };
///
/// let report = validate_par_parameters(&[model]).unwrap();
/// assert!(report.warnings.is_empty());
/// ```
#[derive(Debug)]
pub struct ParValidationReport {
    /// Non-fatal warnings (e.g., near-unit-circle roots, low residual variance).
    pub warnings: Vec<ParWarning>,
}

// ---------------------------------------------------------------------------
// ParWarning
// ---------------------------------------------------------------------------

/// A non-fatal PAR validation warning.
///
/// Warnings are accumulated in [`ParValidationReport`] and do not abort
/// validation. The calling algorithm may inspect them to log diagnostics or
/// apply additional checks.
#[derive(Debug, Clone)]
pub enum ParWarning {
    /// AR polynomial has roots near the unit circle (potential instability).
    ///
    /// Reserved for a future stationarity check; not emitted yet.
    NearUnitCircleRoot {
        /// Identifier of the hydro plant whose AR polynomial has near-unit roots.
        hydro_id: i32,
        /// Stage index at which the near-unit root was detected.
        stage_id: i32,
        /// Magnitude of the smallest root of the AR characteristic polynomial.
        min_root_magnitude: f64,
    },

    /// Residual variance very small relative to sample variance, suggesting the
    /// AR fit may be overfitted (see [`validate_par_parameters`] for the threshold).
    LowResidualVariance {
        /// Identifier of the hydro plant with low residual variance.
        hydro_id: i32,
        /// Stage index at which the low residual variance was detected.
        stage_id: i32,
        /// Explained variance `R² = 1 − residual_std_ratio²`: the AR fit's
        /// coefficient of determination for this `(hydro, season)`.
        explained_variance: f64,
    },
}

// ---------------------------------------------------------------------------
// validate_par_parameters
// ---------------------------------------------------------------------------

/// Validate PAR parameters for consistency and model quality:
///
/// 1. **Positive sample std** (fatal): a model with `ar_order() > 0` must have
///    `std_m3s > 0` — zero std cannot normalize the AR coefficients.
/// 2. **Low residual variance** (warning): explained variance `R² = 1 −
///    residual_std_ratio² > 0.99` (the AR fit explains more than 99% of the
///    seasonal variance) appends a [`ParWarning::LowResidualVariance`].
///
/// # Errors
///
/// Returns [`StochasticError::InvalidParParameters`] when an [`InflowModel`]
/// has `ar_order() > 0` but `std_m3s == 0.0`.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::InflowModel};
/// use cobre_stochastic::par::validation::validate_par_parameters;
///
/// // Valid AR(1) model: no warnings expected.
/// let valid = InflowModel {
///     hydro_id: EntityId(1),
///     stage_id: 0,
///     mean_m3s: 150.0,
///     std_m3s: 30.0,
///     ar_coefficients: vec![0.3],
///     residual_std_ratio: 0.954,
///     annual: None,
/// };
/// let report = validate_par_parameters(&[valid]).unwrap();
/// assert!(report.warnings.is_empty());
///
/// // Invalid: zero std with nonzero AR order.
/// let bad = InflowModel {
///     hydro_id: EntityId(2),
///     stage_id: 1,
///     mean_m3s: 100.0,
///     std_m3s: 0.0,
///     ar_coefficients: vec![0.3],
///     residual_std_ratio: 0.954,
///     annual: None,
/// };
/// let result = validate_par_parameters(&[bad]);
/// assert!(result.is_err());
/// ```
pub fn validate_par_parameters(
    inflow_models: &[InflowModel],
) -> Result<ParValidationReport, StochasticError> {
    let mut warnings = Vec::new();

    for model in inflow_models {
        if model.ar_order() > 0 && model.std_m3s == 0.0 {
            return Err(StochasticError::InvalidParParameters {
                hydro_id: model.hydro_id.0,
                stage_id: model.stage_id,
                reason: format!(
                    "zero standard deviation with ar_order={}: \
                     AR model requires nonzero variance to normalize coefficients",
                    model.ar_order()
                ),
            });
        }

        // explained_variance > 0.99 <=> residual_std_ratio^2 < 0.01: same trigger, inverted scale.
        let explained_variance = 1.0 - model.residual_std_ratio * model.residual_std_ratio;
        if explained_variance > 0.99 {
            warnings.push(ParWarning::LowResidualVariance {
                hydro_id: model.hydro_id.0,
                stage_id: model.stage_id,
                explained_variance,
            });
        }
    }

    Ok(ParValidationReport { warnings })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use cobre_core::{EntityId, scenario::InflowModel};

    use super::{ParWarning, validate_par_parameters};
    use crate::StochasticError;

    fn make_model(
        hydro_id: i32,
        stage_id: i32,
        std_m3s: f64,
        ar_coefficients: Vec<f64>,
        residual_std_ratio: f64,
    ) -> InflowModel {
        InflowModel {
            hydro_id: EntityId(hydro_id),
            stage_id,
            mean_m3s: 100.0,
            std_m3s,
            ar_coefficients,
            residual_std_ratio,
            annual: None,
        }
    }

    fn make_model_with_annual(
        hydro_id: i32,
        stage_id: i32,
        std_m3s: f64,
        ar_coefficients: Vec<f64>,
        residual_std_ratio: f64,
    ) -> InflowModel {
        use cobre_core::scenario::AnnualComponent;
        InflowModel {
            hydro_id: EntityId(hydro_id),
            stage_id,
            mean_m3s: 100.0,
            std_m3s,
            ar_coefficients,
            residual_std_ratio,
            annual: Some(AnnualComponent {
                coefficient: 0.15,
                mean_m3s: 90.0,
                std_m3s: 12.0,
            }),
        }
    }

    #[test]
    fn empty_input_returns_empty_report() {
        let report = validate_par_parameters(&[]).unwrap();
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn ar_order_zero_produces_no_warnings() {
        let model = make_model(1, 5, 30.0, vec![], 1.0);
        let report = validate_par_parameters(&[model]).unwrap();
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn ar_order_zero_with_zero_std_is_valid() {
        let model = make_model(3, 0, 0.0, vec![], 1.0);
        let result = validate_par_parameters(&[model]);
        assert!(result.is_ok());
    }

    #[test]
    fn zero_std_with_nonzero_ar_order_returns_error() {
        let model = make_model(1, 1, 0.0, vec![0.3], 0.954);
        let result = validate_par_parameters(&[model]);

        assert!(result.is_err());
        match result.unwrap_err() {
            StochasticError::InvalidParParameters {
                hydro_id,
                stage_id,
                reason,
            } => {
                assert_eq!(hydro_id, 1);
                assert_eq!(stage_id, 1);
                assert!(reason.contains("zero standard deviation"));
            }
            other => panic!("expected InvalidParParameters, got {other:?}"),
        }
    }

    #[test]
    fn valid_ar1_model_produces_no_warnings() {
        let model = make_model(1, 0, 30.0, vec![0.3], 0.954);
        let report = validate_par_parameters(&[model]).unwrap();
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn low_residual_variance_reports_explained_variance() {
        let model = make_model(7, 12, 30.0, vec![0.4], 0.05);
        let report = validate_par_parameters(&[model]).unwrap();

        assert_eq!(report.warnings.len(), 1);

        match &report.warnings[0] {
            ParWarning::LowResidualVariance {
                hydro_id,
                stage_id,
                explained_variance,
            } => {
                assert_eq!(*hydro_id, 7);
                assert_eq!(*stage_id, 12);
                assert!(
                    (explained_variance - (1.0 - 0.05_f64 * 0.05_f64)).abs() < f64::EPSILON,
                    "explained_variance must be 1 - residual_std_ratio^2"
                );
            }
            other @ ParWarning::NearUnitCircleRoot { .. } => {
                panic!("expected LowResidualVariance, got {other:?}")
            }
        }
    }

    #[test]
    fn explained_variance_at_boundary_no_warning() {
        let model = make_model(2, 3, 30.0, vec![0.3], 0.1);
        let report = validate_par_parameters(&[model]).unwrap();
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn first_fatal_error_stops_iteration() {
        let bad = make_model(1, 0, 0.0, vec![0.3], 0.954);
        let warn_model = make_model(2, 1, 30.0, vec![0.3], 0.05);
        let result = validate_par_parameters(&[bad, warn_model]);
        assert!(result.is_err());
    }

    #[test]
    fn multiple_warnings_accumulated() {
        let m1 = make_model(1, 0, 30.0, vec![0.3], 0.05);
        let m2 = make_model(2, 1, 25.0, vec![0.4], 0.09);
        let report = validate_par_parameters(&[m1, m2]).unwrap();
        assert_eq!(report.warnings.len(), 2);
    }

    #[test]
    fn mixed_models_accumulate_only_applicable_warnings() {
        let clean = make_model(1, 0, 30.0, vec![0.3], 0.954);
        let warn_model = make_model(2, 1, 30.0, vec![0.4], 0.05);
        let ar0 = make_model(3, 2, 20.0, vec![], 1.0);

        let report = validate_par_parameters(&[clean, warn_model, ar0]).unwrap();

        assert_eq!(
            report.warnings.len(),
            1,
            "only the low-variance model should warn"
        );
        match &report.warnings[0] {
            ParWarning::LowResidualVariance { hydro_id, .. } => {
                assert_eq!(*hydro_id, 2);
            }
            other @ ParWarning::NearUnitCircleRoot { .. } => {
                panic!("expected LowResidualVariance, got {other:?}")
            }
        }
    }

    // -----------------------------------------------------------------------
    // Tests for models with annual: Some(_) (PAR(p)-A extension)
    // -----------------------------------------------------------------------

    #[test]
    fn validate_with_annual_some_no_warnings() {
        let model = make_model_with_annual(1, 0, 30.0, vec![0.3], 0.954);
        let report = validate_par_parameters(&[model]).unwrap();
        assert!(
            report.warnings.is_empty(),
            "annual: Some(_) must not trigger spurious warnings"
        );
    }

    #[test]
    fn validate_with_annual_some_low_residual_warns() {
        let model = make_model_with_annual(2, 3, 30.0, vec![0.4], 0.05);
        let report = validate_par_parameters(&[model]).unwrap();
        assert_eq!(
            report.warnings.len(),
            1,
            "exactly one LowResidualVariance warning expected"
        );
        match &report.warnings[0] {
            ParWarning::LowResidualVariance {
                hydro_id,
                stage_id,
                explained_variance,
            } => {
                assert_eq!(*hydro_id, 2);
                assert_eq!(*stage_id, 3);
                assert!(
                    (explained_variance - (1.0 - 0.05_f64 * 0.05_f64)).abs() < f64::EPSILON,
                    "explained_variance must be 1 - residual_std_ratio^2"
                );
            }
            other @ ParWarning::NearUnitCircleRoot { .. } => {
                panic!("expected LowResidualVariance, got {other:?}")
            }
        }
    }

    #[test]
    fn validate_with_annual_some_zero_std_errors() {
        let model = make_model_with_annual(5, 7, 0.0, vec![0.3], 0.954);
        let result = validate_par_parameters(&[model]);
        assert!(result.is_err());
        match result.unwrap_err() {
            StochasticError::InvalidParParameters {
                hydro_id,
                stage_id,
                reason,
            } => {
                assert_eq!(hydro_id, 5);
                assert_eq!(stage_id, 7);
                assert!(
                    reason.contains("zero standard deviation"),
                    "reason must mention zero standard deviation, got: {reason}"
                );
            }
            other => panic!("expected InvalidParParameters, got {other:?}"),
        }
    }
}
