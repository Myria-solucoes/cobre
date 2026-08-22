//! Metadata for the pre-solver preparation phases exercised by `cobre validate`.
//!
//! This module provides the shared [`PrepPhase`] enum and the pure mapping function
//! [`prep_phase_metadata`] that both the CLI (`cobre validate`) and the Python binding
//! (`cobre.io.validate`) use to derive a human-readable file label and a structured
//! error-kind string from a [`SddpError`].
//!
//! # Why this lives in `cobre-sddp`
//!
//! The mapping touches [`SddpError`] variants, which are defined here.
//! `cobre-io` cannot import from `cobre-sddp` (that would create a cycle), so the
//! shared logic must live in `cobre-sddp` or above it in the dependency graph.

use crate::SddpError;

// ── PrepPhase ─────────────────────────────────────────────────────────────────

/// Which pre-solver preparation phase produced an error.
///
/// Each variant corresponds to one of the four SDDP preparation steps that
/// follow the six-layer cobre-io loading pipeline:
///
/// | Phase           | Function called                        | Typical trigger file           |
/// |-----------------|----------------------------------------|-------------------------------|
/// | [`Config`]      | `StudyParams::from_config`             | `config.json`                 |
/// | [`Stochastic`]  | `prepare_stochastic`                   | `scenarios/inflow_history.parquet` |
/// | [`HydroModels`] | `prepare_hydro_models_from_artifacts`  | `system/hydro_production_models.json` |
///
/// [`Config`]: PrepPhase::Config
/// [`Stochastic`]: PrepPhase::Stochastic
/// [`HydroModels`]: PrepPhase::HydroModels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrepPhase {
    /// `StudyParams::from_config` (config.json parsing and semantic validation).
    Config,
    /// `prepare_stochastic` (PAR estimation, opening trees, stochastic context).
    Stochastic,
    /// `prepare_hydro_models_from_artifacts` (production/evaporation models).
    HydroModels,
}

// ── prep_phase_metadata ───────────────────────────────────────────────────────

/// Return the structured error kind and best-effort file label for a
/// preparation-phase error.
///
/// Both the CLI and the Python binding call this to derive:
///
/// * `kind` — a stable, camel-cased string suitable for programmatic filtering
///   (e.g. `"ConfigValidationError"`, `"StochasticPreparationError"`).
/// * `file_label` — a short relative path pointing to the file most likely
///   responsible for the error, suitable for user-facing diagnostics.
///
/// # Returns
///
/// `(kind, file_label)` where both are `'static str` references.
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::validate_phases::{PrepPhase, prep_phase_metadata};
/// use cobre_sddp::SddpError;
///
/// let err = SddpError::Validation("unsupported stopping rule".to_string());
/// let (kind, file) = prep_phase_metadata(PrepPhase::Config, &err);
/// assert_eq!(kind, "ConfigValidationError");
/// assert_eq!(file, "config.json");
/// ```
#[must_use]
pub fn prep_phase_metadata(phase: PrepPhase, err: &SddpError) -> (&'static str, &'static str) {
    let kind = match phase {
        PrepPhase::Config => "ConfigValidationError",
        PrepPhase::Stochastic => "StochasticPreparationError",
        PrepPhase::HydroModels => "HydroModelsPreparationError",
    };

    let file_label = match (phase, err) {
        (PrepPhase::Config, _) => "config.json",
        (PrepPhase::Stochastic, SddpError::Stochastic(_)) => "scenarios/inflow_history.parquet",
        (PrepPhase::Stochastic, _) => "scenarios/",
        (PrepPhase::HydroModels, _) => "system/hydro_production_models.json",
    };

    (kind, file_label)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use cobre_stochastic::StochasticError;

    use super::*;

    #[test]
    fn config_phase_always_returns_config_json() {
        let err = SddpError::Validation("bad window".to_string());
        let (kind, file) = prep_phase_metadata(PrepPhase::Config, &err);
        assert_eq!(kind, "ConfigValidationError");
        assert_eq!(file, "config.json");
    }

    #[test]
    fn stochastic_phase_with_stochastic_error_returns_inflow_history() {
        let err = SddpError::Stochastic(StochasticError::InsufficientData {
            context: "only 2 years".to_string(),
        });
        let (kind, file) = prep_phase_metadata(PrepPhase::Stochastic, &err);
        assert_eq!(kind, "StochasticPreparationError");
        assert_eq!(file, "scenarios/inflow_history.parquet");
    }

    #[test]
    fn stochastic_phase_with_non_stochastic_error_returns_scenarios_dir() {
        let err = SddpError::Validation("something else".to_string());
        let (kind, file) = prep_phase_metadata(PrepPhase::Stochastic, &err);
        assert_eq!(kind, "StochasticPreparationError");
        assert_eq!(file, "scenarios/");
    }

    #[test]
    fn hydro_models_phase_returns_production_models_file() {
        let err = SddpError::Validation("model error".to_string());
        let (kind, file) = prep_phase_metadata(PrepPhase::HydroModels, &err);
        assert_eq!(kind, "HydroModelsPreparationError");
        assert_eq!(file, "system/hydro_production_models.json");
    }

    #[test]
    fn all_phases_produce_non_empty_kind_and_file() {
        let err = SddpError::Validation("test".to_string());
        for phase in [
            PrepPhase::Config,
            PrepPhase::Stochastic,
            PrepPhase::HydroModels,
        ] {
            let (kind, file) = prep_phase_metadata(phase, &err);
            assert!(!kind.is_empty(), "kind must not be empty for {phase:?}");
            assert!(
                !file.is_empty(),
                "file_label must not be empty for {phase:?}"
            );
        }
    }
}
