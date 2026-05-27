//! Configuration types for `config.json`.
//!
//! [`Config`] is the top-level deserialized representation of `config.json`.
//! Use [`parse_config`] to load and validate the file.
//!
//! All optional sections use `#[serde(default)]` so that a minimal `config.json`
//! containing only the mandatory `training` fields deserializes cleanly.
//!
//! # Mandatory fields
//!
//! The following fields have no defaults and must be present in `config.json`:
//!
//! - `training.forward_passes` — number of scenario trajectories per iteration
//! - `training.stopping_rules` — at least one rule entry (must include `iteration_limit`)
//!
//! # Examples
//!
//! ```no_run
//! use cobre_io::config::parse_config;
//! use std::path::Path;
//!
//! let cfg = parse_config(Path::new("case/config.json")).unwrap();
//! println!("forward_passes = {:?}", cfg.training.forward_passes);
//! ```

pub mod energy;
pub mod estimation;
pub mod exports;
pub mod modeling;
pub mod policy;
pub mod scenario_source;
pub mod simulation;
pub mod training;

// Re-export all public types so downstream callers continue to use
// `cobre_io::config::Foo` without knowing which submodule owns `Foo`.
pub use energy::EnergyConfig;
pub use estimation::{EstimationConfig, OrderSelectionMethod};
pub use exports::ExportsConfig;
pub use modeling::{InflowNonNegativityConfig, InflowNonNegativityMethod, ModelingConfig};
pub use policy::{BoundaryPolicy, CheckpointingConfig, PolicyConfig, PolicyMode};
pub use scenario_source::{RawClassConfigEntry, RawHistoricalYearsConfig, RawScenarioSourceConfig};
pub use simulation::SimulationConfig;
pub use training::{
    LipschitzConfig, RowSelectionConfig, StoppingRuleConfig, TrainingConfig, TrainingSolverConfig,
    UpperBoundEvaluationConfig,
};

use cobre_core::scenario::{HistoricalYears, SamplingScheme, ScenarioSource};

use crate::LoadError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level deserialized representation of `config.json`.
///
/// All sections except `training` are optional; their defaults are applied by
/// serde when the section is absent from the JSON.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Config {
    /// JSON schema URI — informational, not validated.
    #[serde(rename = "$schema")]
    pub schema: Option<String>,

    /// Modeling options (inflow non-negativity treatment).
    #[serde(default)]
    pub modeling: ModelingConfig,

    /// Training parameters — contains mandatory fields.
    pub training: TrainingConfig,

    /// Upper-bound evaluation via inner approximation.
    #[serde(default)]
    pub upper_bound_evaluation: UpperBoundEvaluationConfig,

    /// Policy directory settings (warm-start / resume).
    #[serde(default)]
    pub policy: PolicyConfig,

    /// Post-training simulation settings.
    #[serde(default)]
    pub simulation: SimulationConfig,

    /// Export flags controlling which outputs are written to disk.
    #[serde(default)]
    pub exports: ExportsConfig,

    /// Time series estimation settings for automatic model parameter fitting.
    #[serde(default)]
    pub estimation: EstimationConfig,

    /// Energy conversion settings (reference volume fraction for FPHA hydros).
    #[serde(default)]
    pub energy: EnergyConfig,
}

/// Load and validate `config.json` from `path`.
///
/// Reads the JSON file, deserializes it into a [`Config`] struct (applying
/// `#[serde(default)]` for optional sections), then performs post-deserialization
/// validation of mandatory fields.
///
/// # Errors
///
/// | Condition                         | Error variant                 |
/// | --------------------------------- | ----------------------------- |
/// | File not found / read failure     | [`LoadError::IoError`]        |
/// | Invalid JSON syntax               | [`LoadError::ParseError`]     |
/// | `training.forward_passes` missing | [`LoadError::SchemaError`]    |
/// | `training.stopping_rules` missing | [`LoadError::SchemaError`]    |
/// | Unknown stopping rule `"type"`    | [`LoadError::SchemaError`]    |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::config::parse_config;
/// use std::path::Path;
///
/// let cfg = parse_config(Path::new("case/config.json")).unwrap();
/// assert!(cfg.training.forward_passes.unwrap_or(0) > 0);
/// ```
pub fn parse_config(path: &Path) -> Result<Config, LoadError> {
    let raw = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let config: Config = serde_json::from_str(&raw).map_err(|e| {
        // serde_json errors carry a message that describes the field or syntax problem.
        // Unknown enum variants in a tagged enum produce a deserialization error whose
        // message contains the unknown variant name — surfaced to the caller as
        // SchemaError when the field is identifiable, otherwise as ParseError.
        let msg = e.to_string();
        if msg.contains("unknown variant") || msg.contains("missing field") {
            LoadError::SchemaError {
                path: path.to_path_buf(),
                field: extract_field_from_serde_msg(&msg),
                message: msg,
            }
        } else {
            LoadError::parse(path, msg)
        }
    })?;

    validate_config(&config, path)?;

    Ok(config)
}

/// Extract a field name hint from a `serde_json` error message.
///
/// Extracts the identifier between backticks, returning a best-effort field name
/// or `"<unknown>"` when no match is found.
fn extract_field_from_serde_msg(msg: &str) -> String {
    if let Some(start) = msg.find('`')
        && let Some(end) = msg[start + 1..].find('`')
    {
        return msg[start + 1..start + 1 + end].to_string();
    }
    "<unknown>".to_string()
}

/// Post-deserialization validation for mandatory fields.
///
/// Checks that `forward_passes` and `stopping_rules` are present in the config.
pub(crate) fn validate_config(config: &Config, path: &Path) -> Result<(), LoadError> {
    if config.training.forward_passes.is_none() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "training.forward_passes".to_string(),
            message: "required field is missing".to_string(),
        });
    }

    if config.training.stopping_rules.is_none() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "training.stopping_rules".to_string(),
            message: "required field is missing".to_string(),
        });
    }

    let frac = config.energy.reference_volume_fraction;
    if frac.is_nan() || frac <= 0.0 || frac > 1.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "energy.reference_volume_fraction".to_string(),
            message: format!("must be in (0.0, 1.0] (exclusive zero, inclusive one), got {frac}"),
        });
    }

    Ok(())
}

// ── ScenarioSource helpers ───────────────────────────────────────────────────

/// Convert a `scheme` string from `config.json` to [`SamplingScheme`].
///
/// `field` is the dot-separated JSON path to the scheme key (e.g.
/// `"training.scenario_source.inflow.scheme"`), used verbatim in the error
/// message so the caller can identify which field has the invalid value.
fn convert_sampling_scheme_cfg(
    s: &str,
    field: &str,
    path: &Path,
) -> Result<SamplingScheme, LoadError> {
    match s {
        "in_sample" => Ok(SamplingScheme::InSample),
        "out_of_sample" => Ok(SamplingScheme::OutOfSample),
        "external" => Ok(SamplingScheme::External),
        "historical" => Ok(SamplingScheme::Historical),
        other => Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: field.to_string(),
            message: format!(
                "unknown scheme '{other}', expected one of: in_sample, out_of_sample, external, historical"
            ),
        }),
    }
}

/// Convert a per-class config entry to its [`SamplingScheme`], defaulting to
/// `in_sample` when the entry is absent.
fn convert_class_scheme_cfg(
    class: Option<&RawClassConfigEntry>,
    section: &str,
    class_name: &str,
    path: &Path,
) -> Result<SamplingScheme, LoadError> {
    convert_sampling_scheme_cfg(
        class.map_or("in_sample", |c| c.scheme.as_str()),
        &format!("{section}.scenario_source.{class_name}.scheme"),
        path,
    )
}

/// Convert `Option<RawScenarioSourceConfig>` into a [`ScenarioSource`].
///
/// `section` is either `"training"` or `"simulation"`, used to build field
/// paths in error messages that reference `config.json`.
///
/// Returns `ScenarioSource::default()` (all `InSample`, no seed, no years)
/// when `raw` is `None`.
fn convert_scenario_source_config(
    raw: Option<&RawScenarioSourceConfig>,
    section: &str,
    path: &Path,
) -> Result<ScenarioSource, LoadError> {
    let Some(r) = raw else {
        return Ok(ScenarioSource::default());
    };

    let inflow_scheme = convert_class_scheme_cfg(r.inflow.as_ref(), section, "inflow", path)?;
    let load_scheme = convert_class_scheme_cfg(r.load.as_ref(), section, "load", path)?;
    let ncs_scheme = convert_class_scheme_cfg(r.ncs.as_ref(), section, "ncs", path)?;

    let source = ScenarioSource {
        inflow_scheme,
        load_scheme,
        ncs_scheme,
        seed: r.seed,
        historical_years: r.historical_years.as_ref().map(|hy| match hy {
            RawHistoricalYearsConfig::List(years) => HistoricalYears::List(years.clone()),
            RawHistoricalYearsConfig::Range { from, to } => HistoricalYears::Range {
                from: *from,
                to: *to,
            },
        }),
    };

    validate_scenario_source_cfg(&source, section, path)?;

    Ok(source)
}

/// Tier-1 structural validation of a parsed [`ScenarioSource`] from `config.json`.
///
/// ## Checks performed
///
/// - `historical_years` must not be specified if no class uses `Historical`.
/// - `seed` is required when any class uses `OutOfSample`, `Historical`, or `External`.
/// - `Historical` scheme is only valid for the `inflow` class.
/// - If `historical_years` is a `Range`, `from` must be `<= to`.
fn validate_scenario_source_cfg(
    source: &ScenarioSource,
    section: &str,
    path: &Path,
) -> Result<(), LoadError> {
    let uses_historical = source.inflow_scheme == SamplingScheme::Historical
        || source.load_scheme == SamplingScheme::Historical
        || source.ncs_scheme == SamplingScheme::Historical;

    if source.historical_years.is_some() && !uses_historical {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("{section}.scenario_source.historical_years"),
            message: "historical_years is specified but no class uses the 'historical' scheme"
                .to_string(),
        });
    }

    // Historical scheme is only valid for inflow class
    if source.load_scheme == SamplingScheme::Historical {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("{section}.scenario_source.load.scheme"),
            message: "historical scheme is only valid for the inflow class".to_string(),
        });
    }

    if source.ncs_scheme == SamplingScheme::Historical {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("{section}.scenario_source.ncs.scheme"),
            message: "historical scheme is only valid for the inflow class".to_string(),
        });
    }

    // Seed is required unless all classes are InSample
    let all_in_sample = source.inflow_scheme == SamplingScheme::InSample
        && source.load_scheme == SamplingScheme::InSample
        && source.ncs_scheme == SamplingScheme::InSample;
    if !all_in_sample && source.seed.is_none() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("{section}.scenario_source.seed"),
            message:
                "seed is required when any class uses out_of_sample, historical, or external scheme"
                    .to_string(),
        });
    }

    if let Some(HistoricalYears::Range { from, to }) = source.historical_years
        && from > to
    {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("{section}.scenario_source.historical_years"),
            message: format!("range 'from' ({from}) must be <= 'to' ({to})"),
        });
    }

    Ok(())
}

impl Config {
    /// Resolve the training-phase [`ScenarioSource`].
    ///
    /// When `training.scenario_source` is absent, returns `ScenarioSource::default()`
    /// (all classes `InSample`, no seed, no historical years).
    ///
    /// # Errors
    ///
    /// Returns `LoadError::SchemaError` if the raw config contains an invalid
    /// scheme string, Historical on a non-inflow class, or seed/year validation
    /// failures.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cobre_io::config::parse_config;
    /// use std::path::Path;
    ///
    /// let cfg = parse_config(Path::new("case/config.json")).unwrap();
    /// let source = cfg.training_scenario_source(Path::new("case/config.json")).unwrap();
    /// ```
    pub fn training_scenario_source(&self, path: &Path) -> Result<ScenarioSource, LoadError> {
        convert_scenario_source_config(self.training.scenario_source.as_ref(), "training", path)
    }

    /// Resolve the simulation-phase [`ScenarioSource`].
    ///
    /// Falls back to `training_scenario_source()` when
    /// `simulation.scenario_source` is absent.
    ///
    /// # Errors
    ///
    /// Returns `LoadError::SchemaError` on validation failures in either the
    /// simulation or training scenario source.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cobre_io::config::parse_config;
    /// use std::path::Path;
    ///
    /// let cfg = parse_config(Path::new("case/config.json")).unwrap();
    /// let source = cfg.simulation_scenario_source(Path::new("case/config.json")).unwrap();
    /// ```
    pub fn simulation_scenario_source(&self, path: &Path) -> Result<ScenarioSource, LoadError> {
        if self.simulation.scenario_source.is_some() {
            convert_scenario_source_config(
                self.simulation.scenario_source.as_ref(),
                "simulation",
                path,
            )
        } else {
            self.training_scenario_source(path)
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown
)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_config(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    /// AC-1: minimal config returns Ok with correct forward_passes and all
    /// optional sections at their default values.
    #[test]
    fn test_parse_minimal_config() {
        let f = write_config(
            r#"{"training": {"tree_seed": 42, "forward_passes": 192, "stopping_rules": [{"type": "iteration_limit", "limit": 50}]}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();

        // Mandatory field present and correct
        assert_eq!(cfg.training.forward_passes, Some(192));

        // tree_seed is optional
        assert_eq!(cfg.training.tree_seed, Some(42));

        // Defaults applied to optional sections
        assert_eq!(cfg.training.stopping_mode, "any");
        assert!(cfg.training.enabled);
        assert_eq!(
            cfg.modeling.inflow_non_negativity.method,
            InflowNonNegativityMethod::Penalty
        );
        assert!(!cfg.simulation.enabled);
        assert_eq!(cfg.simulation.num_scenarios, 2000);
        assert_eq!(cfg.policy.mode, PolicyMode::Fresh);
        assert_eq!(cfg.policy.path, "./policy");
        assert!(cfg.policy.validate_compatibility);
    }

    /// AC-2: missing `training.forward_passes` → SchemaError with field name.
    #[test]
    fn test_missing_forward_passes() {
        let f = write_config(
            r#"{"training": {"tree_seed": 1, "stopping_rules": [{"type": "iteration_limit", "limit": 10}]}}"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("forward_passes"),
                    "field should contain 'forward_passes', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC-2 variant: missing `training.stopping_rules` → SchemaError.
    #[test]
    fn test_missing_stopping_rules() {
        let f = write_config(r#"{"training": {"tree_seed": 1, "forward_passes": 100}}"#);
        let err = parse_config(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("stopping_rules"),
                    "field should contain 'stopping_rules', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC-3: nonexistent file → IoError with matching path.
    #[test]
    fn test_nonexistent_file() {
        let path = std::path::Path::new("/nonexistent/path/config.json");
        let err = parse_config(path).unwrap_err();
        match &err {
            LoadError::IoError { path: p, .. } => {
                assert_eq!(p, path);
            }
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    /// AC-4: full config with all sections → Ok with non-default values.
    #[test]
    fn test_parse_full_config() {
        let json = r#"{
          "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/config.schema.json",
          "modeling": {
            "inflow_non_negativity": {
              "method": "penalty"
            }
          },
          "training": {
            "tree_seed": 42,
            "forward_passes": 192,
            "stopping_rules": [
              {"type": "iteration_limit", "limit": 50},
              {"type": "bound_stalling", "iterations": 10, "tolerance": 0.0001}
            ],
            "stopping_mode": "any",
            "cut_selection": {
              "enabled": true,
              "method": "domination"
            }
          },
          "upper_bound_evaluation": {
            "enabled": true,
            "initial_iteration": 10,
            "interval_iterations": 5
          },
          "policy": {
            "path": "./policy",
            "mode": "fresh",
            "checkpointing": {
              "enabled": true,
              "initial_iteration": 10,
              "interval_iterations": 10,
              "store_basis": true,
              "compress": true
            },
            "validate_compatibility": true
          },
          "simulation": {
            "enabled": true,
            "num_scenarios": 2000
          },
          "exports": {
            "states": true,
            "stochastic": true
          }
        }"#;

        let f = write_config(json);
        let cfg = parse_config(f.path()).unwrap();

        // Modeling
        assert_eq!(
            cfg.modeling.inflow_non_negativity.method,
            InflowNonNegativityMethod::Penalty
        );

        // Training
        assert_eq!(cfg.training.forward_passes, Some(192));
        assert_eq!(cfg.training.stopping_mode, "any");
        let rules = cfg.training.stopping_rules.as_ref().unwrap();
        assert_eq!(rules.len(), 2);
        let cut_sel = &cfg.training.cut_selection;
        assert_eq!(cut_sel.enabled, Some(true));
        assert_eq!(cut_sel.method.as_deref(), Some("domination"));

        // Upper bound
        assert_eq!(cfg.upper_bound_evaluation.enabled, Some(true));
        assert_eq!(cfg.upper_bound_evaluation.initial_iteration, Some(10));

        // Policy
        assert_eq!(cfg.policy.mode, PolicyMode::Fresh);
        assert!(cfg.policy.validate_compatibility);
        assert_eq!(cfg.policy.checkpointing.enabled, Some(true));

        // Simulation
        assert!(cfg.simulation.enabled);
        assert_eq!(cfg.simulation.num_scenarios, 2000);

        // Exports
        assert!(cfg.exports.states);
        assert!(cfg.exports.stochastic);
    }

    /// AC-5: invalid JSON syntax → ParseError.
    #[test]
    fn test_invalid_json_syntax() {
        let f = write_config(r#"{"training": {not valid json}}"#);
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError, got: {err:?}"
        );
    }

    /// All 4 JSON-configurable stopping rule variants deserialize correctly.
    ///
    /// The `GracefulShutdown` variant is runtime-only and has no JSON representation
    /// per the stopping-rule-trait spec (SS4.1).
    #[test]
    fn test_stopping_rule_variants() {
        let json = r#"{
          "training": {
            "forward_passes": 10,
            "stopping_rules": [
              {"type": "iteration_limit", "limit": 100},
              {"type": "time_limit", "seconds": 3600.0},
              {"type": "bound_stalling", "iterations": 10, "tolerance": 0.0001},
              {
                "type": "simulation",
                "replications": 100,
                "period": 20,
                "bound_window": 5,
                "distance_tol": 0.01,
                "bound_tol": 0.0001
              }
            ]
          }
        }"#;

        let f = write_config(json);
        let cfg = parse_config(f.path()).unwrap();
        let rules = cfg.training.stopping_rules.unwrap();
        assert_eq!(rules.len(), 4);

        assert!(matches!(
            rules[0],
            StoppingRuleConfig::IterationLimit { limit: 100 }
        ));
        assert!(
            matches!(rules[1], StoppingRuleConfig::TimeLimit { seconds } if (seconds - 3600.0).abs() < f64::EPSILON)
        );
        assert!(matches!(
            rules[2],
            StoppingRuleConfig::BoundStalling { iterations: 10, .. }
        ));
        assert!(matches!(
            rules[3],
            StoppingRuleConfig::Simulation {
                replications: 100,
                period: 20,
                ..
            }
        ));
    }

    /// Unknown stopping rule type → SchemaError (not a panic or ParseError).
    #[test]
    fn test_unknown_stopping_rule_type() {
        let f = write_config(
            r#"{"training": {"forward_passes": 10, "stopping_rules": [{"type": "nonexistent_rule"}]}}"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError for unknown rule type, got: {err:?}"
        );
    }

    /// `Config` has no `version` field — the struct does not
    /// expose `.version` and the field is not present after deserialization.
    #[test]
    fn test_config_has_no_version_field() {
        let f = write_config(
            r#"{"training": {"forward_passes": 1, "stopping_rules": [{"type": "iteration_limit", "limit": 10}]}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        // The struct must not have a `version` field — verified by compilation.
        // We also check that the $schema field is None when absent from JSON.
        assert!(cfg.schema.is_none(), "schema should be None when absent");
    }

    /// JSON with `"$schema"` property is accepted and the field
    /// value is stored correctly.
    #[test]
    fn test_schema_field_accepted() {
        let f = write_config(
            r#"{
            "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/config.schema.json",
            "training": {
                "forward_passes": 1,
                "stopping_rules": [{"type": "iteration_limit", "limit": 10}]
            }
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert_eq!(
            cfg.schema.as_deref(),
            Some(
                "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/config.schema.json"
            ),
            "schema field should be stored when present in JSON"
        );
    }

    /// Invalid `policy.mode` values are rejected at parse time.
    #[test]
    fn test_invalid_policy_mode_rejected() {
        let f = write_config(
            r#"{"training": {"forward_passes": 1, "stopping_rules": [{"type": "iteration_limit", "limit": 10}]}, "policy": {"mode": "warmstart"}}"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError for invalid policy.mode, got: {err:?}"
        );
    }

    /// JSON that contains the dead `"version"` property must now be rejected
    /// because `Config` uses `deny_unknown_fields`. Old case dirs that still
    /// contain this key will fail to parse — which is the desired behaviour.
    #[test]
    fn test_legacy_version_field_rejected() {
        let f = write_config(
            r#"{
            "version": "1.0.0",
            "training": {
                "forward_passes": 1,
                "stopping_rules": [{"type": "iteration_limit", "limit": 10}]
            }
        }"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            matches!(
                err,
                LoadError::ParseError { .. } | LoadError::SchemaError { .. }
            ),
            "expected parse/schema error for unknown 'version' field, got: {err:?}"
        );
    }

    /// `"truncation"` is accepted as a method value and round-trips correctly
    /// through `parse_config`.
    #[test]
    fn test_truncation_method_accepted() {
        let f = write_config(
            r#"{
            "modeling": {
                "inflow_non_negativity": {
                    "method": "truncation"
                }
            },
            "training": {
                "forward_passes": 10,
                "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
            }
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert_eq!(
            cfg.modeling.inflow_non_negativity.method,
            InflowNonNegativityMethod::Truncation,
            "method field should round-trip as Truncation"
        );
    }

    /// An unknown inflow non-negativity method string is rejected at parse time.
    #[test]
    fn test_unknown_inflow_method_rejected() {
        let f = write_config(
            r#"{
            "modeling": {
                "inflow_non_negativity": {
                    "method": "bogus_method"
                }
            },
            "training": {
                "forward_passes": 10,
                "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
            }
        }"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            matches!(
                err,
                LoadError::SchemaError { .. } | LoadError::ParseError { .. }
            ),
            "expected parse/schema error for unknown method, got: {err:?}"
        );
    }

    /// AC-035-1: `config.json` without `"estimation"` section → all three defaults applied.
    #[test]
    fn test_estimation_config_defaults() {
        let f = write_config(
            r#"{"training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert_eq!(cfg.estimation.max_order, 6);
        assert!(
            matches!(cfg.estimation.order_selection, OrderSelectionMethod::Pacf),
            "default order_selection should be Pacf"
        );
        assert_eq!(cfg.estimation.min_observations_per_season, 30);
    }

    /// AC-035-2: `"order_selection": "fixed"` is now a hard parse error.
    #[test]
    fn test_estimation_config_order_selection_fixed_rejected() {
        let f = write_config(
            r#"{
            "training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
            "estimation": {"max_order": 3, "order_selection": "fixed", "min_observations_per_season": 20}
        }"#,
        );
        let result = parse_config(f.path());
        assert!(
            result.is_err(),
            "\"fixed\" order_selection must now be a parse error"
        );
    }

    /// AC-035-2b: `"order_selection": "pacf"` deserializes to `Pacf` with no warning.
    #[test]
    fn test_estimation_config_order_selection_pacf() {
        let f = write_config(
            r#"{
            "training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
            "estimation": {"max_order": 4, "order_selection": "pacf", "min_observations_per_season": 15}
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert_eq!(cfg.estimation.max_order, 4);
        assert!(
            matches!(cfg.estimation.order_selection, OrderSelectionMethod::Pacf),
            "explicit 'pacf' must deserialize to Pacf"
        );
        assert_eq!(cfg.estimation.min_observations_per_season, 15);
    }

    /// AC-035-3: unknown `order_selection` value → `LoadError::SchemaError` with
    /// message containing `"unknown variant"`.
    #[test]
    fn test_estimation_config_unknown_order_selection() {
        let f = write_config(
            r#"{
            "training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
            "estimation": {"order_selection": "bogus"}
        }"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("unknown variant"),
                    "message should contain 'unknown variant', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `exports.stochastic: true` deserializes correctly.
    ///
    /// Verifies that a `config.json` with `"exports": {"stochastic": true}` round-trips
    /// the field as `true` in `ExportsConfig`.
    #[test]
    fn test_exports_stochastic_explicit_true() {
        let f = write_config(
            r#"{
            "training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
            "exports": {"stochastic": true}
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert!(
            cfg.exports.stochastic,
            "exports.stochastic should be true when set in config"
        );
    }

    /// `exports.stochastic` defaults to `false` when the field is absent.
    ///
    /// Verifies that a `config.json` without the `stochastic` field in the
    /// `exports` section resolves to `false` via `#[serde(default)]`.
    #[test]
    fn test_exports_stochastic_defaults_to_false() {
        let f = write_config(
            r#"{
            "training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]}
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert!(
            !cfg.exports.stochastic,
            "exports.stochastic should default to false when absent"
        );
    }

    // ── ScenarioSource parsing tests ──────────────────────────────────────────

    const MINIMAL_TRAINING: &str =
        r#"{"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]}"#;

    fn write_with_training_scenario_source(scenario_source_json: &str) -> NamedTempFile {
        write_config(&format!(
            r#"{{"training": {{"forward_passes": 10, "stopping_rules": [{{"type": "iteration_limit", "limit": 5}}], "scenario_source": {scenario_source_json}}}}}"#
        ))
    }

    fn write_with_both_scenario_sources(
        training_json: &str,
        simulation_json: &str,
    ) -> NamedTempFile {
        write_config(&format!(
            r#"{{"training": {{"forward_passes": 10, "stopping_rules": [{{"type": "iteration_limit", "limit": 5}}], "scenario_source": {training_json}}}, "simulation": {{"scenario_source": {simulation_json}}}}}"#
        ))
    }

    /// Absent `training.scenario_source` → all InSample, no seed, no historical_years.
    #[test]
    fn test_training_scenario_source_default() {
        let f = write_config(&format!(r#"{{"training": {MINIMAL_TRAINING}}}"#));
        let cfg = parse_config(f.path()).unwrap();
        let source = cfg.training_scenario_source(f.path()).unwrap();
        assert_eq!(source, ScenarioSource::default());
        assert_eq!(source.inflow_scheme, SamplingScheme::InSample);
        assert_eq!(source.load_scheme, SamplingScheme::InSample);
        assert_eq!(source.ncs_scheme, SamplingScheme::InSample);
        assert_eq!(source.seed, None);
        assert_eq!(source.historical_years, None);
    }

    /// Explicit per-class schemes are parsed correctly.
    #[test]
    fn test_training_scenario_source_explicit() {
        let f = write_with_training_scenario_source(
            r#"{"seed": 42, "inflow": {"scheme": "historical"}, "historical_years": [1940, 1953]}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        let source = cfg.training_scenario_source(f.path()).unwrap();
        assert_eq!(source.inflow_scheme, SamplingScheme::Historical);
        assert_eq!(source.load_scheme, SamplingScheme::InSample);
        assert_eq!(source.ncs_scheme, SamplingScheme::InSample);
        assert_eq!(source.seed, Some(42));
        assert_eq!(
            source.historical_years,
            Some(HistoricalYears::List(vec![1940, 1953]))
        );
    }

    /// Absent `simulation.scenario_source` falls back to `training_scenario_source()`.
    #[test]
    fn test_simulation_scenario_source_fallback() {
        let f = write_with_training_scenario_source(
            r#"{"seed": 7, "inflow": {"scheme": "out_of_sample"}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        let training = cfg.training_scenario_source(f.path()).unwrap();
        let simulation = cfg.simulation_scenario_source(f.path()).unwrap();
        assert_eq!(training, simulation);
        assert_eq!(simulation.inflow_scheme, SamplingScheme::OutOfSample);
        assert_eq!(simulation.seed, Some(7));
    }

    /// Both sections present with different schemes → different `ScenarioSource` values returned.
    #[test]
    fn test_simulation_scenario_source_independent() {
        let f = write_with_both_scenario_sources(
            r#"{"seed": 1, "inflow": {"scheme": "out_of_sample"}}"#,
            r#"{"seed": 2, "load": {"scheme": "out_of_sample"}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        let training = cfg.training_scenario_source(f.path()).unwrap();
        let simulation = cfg.simulation_scenario_source(f.path()).unwrap();
        assert_ne!(training, simulation);
        assert_eq!(training.inflow_scheme, SamplingScheme::OutOfSample);
        assert_eq!(training.load_scheme, SamplingScheme::InSample);
        assert_eq!(simulation.inflow_scheme, SamplingScheme::InSample);
        assert_eq!(simulation.load_scheme, SamplingScheme::OutOfSample);
    }

    /// Historical scheme on inflow class is accepted.
    #[test]
    fn test_scenario_source_historical_inflow_valid() {
        let f = write_with_training_scenario_source(
            r#"{"seed": 99, "inflow": {"scheme": "historical"}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        let source = cfg.training_scenario_source(f.path()).unwrap();
        assert_eq!(source.inflow_scheme, SamplingScheme::Historical);
    }

    /// Historical on load class → SchemaError.
    #[test]
    fn test_scenario_source_historical_load_rejected() {
        let f = write_config(&format!(
            r#"{{"training": {MINIMAL_TRAINING}, "simulation": {{"scenario_source": {{"seed": 1, "load": {{"scheme": "historical"}}}}}}}}"#
        ));
        let cfg = parse_config(f.path()).unwrap();
        let err = cfg.simulation_scenario_source(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, field, .. } => {
                assert!(
                    message.contains("historical scheme is only valid for the inflow class"),
                    "unexpected message: {message}"
                );
                assert!(field.contains("load.scheme"), "unexpected field: {field}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Historical on ncs class → SchemaError.
    #[test]
    fn test_scenario_source_historical_ncs_rejected() {
        let f =
            write_with_training_scenario_source(r#"{"seed": 1, "ncs": {"scheme": "historical"}}"#);
        let cfg = parse_config(f.path()).unwrap();
        let err = cfg.training_scenario_source(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, field, .. } => {
                assert!(
                    message.contains("historical scheme is only valid for the inflow class"),
                    "unexpected message: {message}"
                );
                assert!(field.contains("ncs.scheme"), "unexpected field: {field}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// OutOfSample without seed → SchemaError.
    #[test]
    fn test_scenario_source_seed_required_for_oos() {
        let f = write_with_training_scenario_source(r#"{"inflow": {"scheme": "out_of_sample"}}"#);
        let cfg = parse_config(f.path()).unwrap();
        let err = cfg.training_scenario_source(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, field, .. } => {
                assert!(
                    message.contains("seed is required"),
                    "unexpected message: {message}"
                );
                assert!(field.contains("seed"), "unexpected field: {field}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Range form of `historical_years` parses correctly.
    #[test]
    fn test_scenario_source_historical_years_range() {
        let f = write_with_training_scenario_source(
            r#"{"seed": 5, "inflow": {"scheme": "historical"}, "historical_years": {"from": 1940, "to": 2010}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        let source = cfg.training_scenario_source(f.path()).unwrap();
        assert_eq!(
            source.historical_years,
            Some(HistoricalYears::Range {
                from: 1940,
                to: 2010
            })
        );
    }

    /// `historical_years` specified without any Historical scheme → SchemaError.
    #[test]
    fn test_scenario_source_historical_years_without_historical_scheme() {
        let f = write_with_training_scenario_source(
            r#"{"seed": 1, "inflow": {"scheme": "out_of_sample"}, "historical_years": [1990, 2000]}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        let err = cfg.training_scenario_source(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains(
                        "historical_years is specified but no class uses the 'historical' scheme"
                    ),
                    "unexpected message: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `simulation.sampling_scheme` (dead field) is now rejected because
    /// `SimulationConfig` uses `deny_unknown_fields`. Old case dirs must remove
    /// this key before loading.
    #[test]
    fn test_dead_sampling_scheme_field_rejected() {
        let f = write_config(
            r#"{
            "training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
            "simulation": {"enabled": true, "sampling_scheme": {"type": "in_sample"}}
        }"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            matches!(
                err,
                LoadError::ParseError { .. } | LoadError::SchemaError { .. }
            ),
            "expected parse/schema error for unknown 'sampling_scheme' field, got: {err:?}"
        );
    }

    /// max_active_per_stage serde roundtrip: Some(100) serializes and deserializes correctly.
    #[test]
    fn max_active_per_stage_serde_roundtrip() {
        let original = RowSelectionConfig {
            enabled: Some(true),
            method: Some("level1".to_string()),
            threshold: None,
            memory_window: None,
            domination_epsilon: None,
            check_frequency: None,
            cut_activity_tolerance: None,
            max_active_per_stage: Some(100),
            basis_activity_window: Some(7),
            tie_tolerance: None,
        };
        let json = serde_json::to_string(&original).unwrap();
        let roundtripped: RowSelectionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtripped.max_active_per_stage, Some(100));
        assert_eq!(roundtripped.enabled, Some(true));
        assert_eq!(roundtripped.method.as_deref(), Some("level1"));
        assert_eq!(roundtripped.basis_activity_window, Some(7));
    }

    /// max_active_per_stage absent from JSON deserializes to None.
    #[test]
    fn max_active_per_stage_absent_defaults_none() {
        let f = write_config(
            r#"{
            "training": {
                "forward_passes": 10,
                "stopping_rules": [{"type": "iteration_limit", "limit": 5}],
                "cut_selection": {"enabled": true, "method": "level1"}
            }
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert!(
            cfg.training.cut_selection.max_active_per_stage.is_none(),
            "max_active_per_stage must be None when absent from config.json"
        );
    }

    /// `policy.boundary` with `path` and `source_stage` deserializes
    /// to `Some(BoundaryPolicy { .. })` with the correct field values.
    #[test]
    fn test_boundary_policy_present() {
        let f = write_config(
            r#"{
            "training": {
                "forward_passes": 10,
                "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
            },
            "policy": {
                "mode": "fresh",
                "boundary": {
                    "path": "../monthly/policy",
                    "source_stage": 2
                }
            }
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        let boundary = cfg.policy.boundary.unwrap();
        assert_eq!(boundary.path, "../monthly/policy");
        assert_eq!(boundary.source_stage, 2);
    }

    /// `policy` without a `boundary` key deserializes to `None`.
    #[test]
    fn test_boundary_policy_absent() {
        let f = write_config(
            r#"{
            "training": {
                "forward_passes": 10,
                "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
            },
            "policy": {}
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert!(
            cfg.policy.boundary.is_none(),
            "boundary must be None when the key is absent"
        );
    }

    /// `"boundary": null` deserializes to `None`.
    #[test]
    fn test_boundary_policy_explicit_null() {
        let f = write_config(
            r#"{
            "training": {
                "forward_passes": 10,
                "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
            },
            "policy": { "boundary": null }
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert!(
            cfg.policy.boundary.is_none(),
            "boundary must be None when explicitly null"
        );
    }

    /// `PolicyConfig::default()` has `boundary` set to `None`.
    #[test]
    fn test_policy_config_default_boundary_is_none() {
        assert!(
            PolicyConfig::default().boundary.is_none(),
            "default PolicyConfig must have boundary = None"
        );
    }

    /// Round-trip: serialize `PolicyConfig` with `Some(BoundaryPolicy)`
    /// to JSON and deserialize back; values are preserved.
    #[test]
    fn test_boundary_policy_round_trip() {
        let original = PolicyConfig {
            path: "./policy".to_string(),
            mode: PolicyMode::Fresh,
            validate_compatibility: true,
            checkpointing: CheckpointingConfig::default(),
            boundary: Some(BoundaryPolicy {
                path: "../monthly/policy".to_string(),
                source_stage: 5,
            }),
        };
        let json = serde_json::to_string(&original).unwrap();
        let restored: PolicyConfig = serde_json::from_str(&json).unwrap();
        let boundary = restored.boundary.unwrap();
        assert_eq!(boundary.path, "../monthly/policy");
        assert_eq!(boundary.source_stage, 5);
    }

    // ── RowSelectionConfig::threshold tests ──────────────────────────────────

    /// AC: `threshold` is accepted and round-trips for `level1`.
    #[test]
    fn test_row_selection_threshold_accepted() {
        let json = r#"{"threshold": 5}"#;
        let cfg: RowSelectionConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.threshold, Some(5), "threshold must be stored");
    }

    /// Stale `exports` keys (`training`, `cuts`, `vertices`, `simulation`,
    /// `forward_detail`, `backward_detail`, `compression`) are now rejected
    /// because `ExportsConfig` uses `deny_unknown_fields`. Old case dirs that
    /// still contain these keys must remove them before loading.
    #[test]
    fn parse_config_rejects_removed_exports_fields() {
        let json = r#"{
            "training": { "forward_passes": 4, "stopping_rules": [] },
            "exports": {
                "training": true,
                "cuts": false,
                "vertices": true,
                "simulation": true,
                "forward_detail": true,
                "backward_detail": true,
                "compression": "zstd"
            }
        }"#;
        let result = serde_json::from_str::<Config>(json);
        assert!(
            result.is_err(),
            "expected parse error for stale exports fields, got Ok"
        );
    }

    // ── OrderSelectionMethod::PacfAnnual tests ────────────────────────────────

    /// `"pacf_annual"` round-trips through serde_json.
    ///
    /// Deserialization must produce `PacfAnnual`; serialization must produce
    /// the `"pacf_annual"` string.
    #[test]
    fn order_selection_pacf_annual_round_trip() {
        let parsed: OrderSelectionMethod = serde_json::from_str("\"pacf_annual\"").unwrap();
        assert!(
            matches!(parsed, OrderSelectionMethod::PacfAnnual),
            "\"pacf_annual\" must deserialize to PacfAnnual, got: {parsed:?}"
        );
        let serialized = serde_json::to_string(&OrderSelectionMethod::PacfAnnual).unwrap();
        assert_eq!(
            serialized, "\"pacf_annual\"",
            "PacfAnnual must serialize to \"pacf_annual\", got: {serialized}"
        );
    }

    /// An unknown variant error must mention `"pacf_annual"` as an expected
    /// variant so users know the option exists.
    #[test]
    fn order_selection_unknown_variant_lists_pacf_annual() {
        let err = serde_json::from_str::<OrderSelectionMethod>("\"pacf_seasonal\"").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("pacf_annual"),
            "error message must contain \"pacf_annual\", got: {msg}"
        );
    }

    /// The default variant must remain `Pacf`; `PacfAnnual` is opt-in.
    #[test]
    fn order_selection_default_is_pacf() {
        assert!(
            matches!(OrderSelectionMethod::default(), OrderSelectionMethod::Pacf),
            "default must be Pacf, not PacfAnnual"
        );
    }

    /// `"fixed"` is no longer a valid value and must hard-error on parse.
    #[test]
    fn order_selection_fixed_rejected() {
        let result: Result<OrderSelectionMethod, _> = serde_json::from_str("\"fixed\"");
        assert!(
            result.is_err(),
            "\"fixed\" must be rejected; expected an error"
        );
    }

    // ── EnergyConfig tests ────────────────────────────────────────────────────

    /// AC: absent `energy` section → `reference_volume_fraction` defaults to 0.65.
    #[test]
    fn energy_config_defaults_to_065_when_absent() {
        let f = write_config(
            r#"{"training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert!(
            (cfg.energy.reference_volume_fraction - 0.65).abs() < f64::EPSILON,
            "default reference_volume_fraction should be 0.65, got: {}",
            cfg.energy.reference_volume_fraction
        );
    }

    /// AC: explicit `reference_volume_fraction` round-trips correctly.
    #[test]
    fn energy_config_round_trips_explicit_value() {
        let f = write_config(
            r#"{
            "training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
            "energy": {"reference_volume_fraction": 0.7}
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert!(
            (cfg.energy.reference_volume_fraction - 0.7).abs() < f64::EPSILON,
            "reference_volume_fraction should be 0.7, got: {}",
            cfg.energy.reference_volume_fraction
        );
    }

    /// AC: `reference_volume_fraction: 0.0` → SchemaError naming the field.
    #[test]
    fn energy_config_rejects_zero_fraction() {
        let f = write_config(
            r#"{
            "training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
            "energy": {"reference_volume_fraction": 0.0}
        }"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("energy.reference_volume_fraction"),
                    "field should name energy.reference_volume_fraction, got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: `reference_volume_fraction: 1.5` → SchemaError (above 1.0).
    #[test]
    fn energy_config_rejects_value_above_one() {
        let f = write_config(
            r#"{
            "training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
            "energy": {"reference_volume_fraction": 1.5}
        }"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError for fraction > 1.0, got: {err:?}"
        );
    }

    /// AC: negative `reference_volume_fraction` → SchemaError.
    #[test]
    fn energy_config_rejects_negative_value() {
        let f = write_config(
            r#"{
            "training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
            "energy": {"reference_volume_fraction": -0.1}
        }"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError for negative fraction, got: {err:?}"
        );
    }

    /// AC: NaN `reference_volume_fraction` → SchemaError.
    #[test]
    fn energy_config_rejects_nan() {
        // JSON does not support NaN literals; we test by direct struct validation.
        // Build an EnergyConfig with NaN and confirm validate_config catches it.
        let cfg = Config {
            schema: None,
            modeling: ModelingConfig::default(),
            training: TrainingConfig {
                enabled: true,
                tree_seed: None,
                forward_passes: Some(10),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 5 }]),
                stopping_mode: "any".to_string(),
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                scenario_source: None,
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: SimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
            energy: EnergyConfig {
                reference_volume_fraction: f64::NAN,
            },
        };
        let path = std::path::Path::new("config.json");
        let err = validate_config(&cfg, path).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("energy.reference_volume_fraction"),
                    "field should name energy.reference_volume_fraction, got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }
}
