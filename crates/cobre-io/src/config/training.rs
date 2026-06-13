//! Training-phase configuration types for `config.json → training`.

use serde::{Deserialize, Serialize};

use super::scenario_source::RawScenarioSourceConfig;

/// Training parameters (`config.json → training`).
///
/// `forward_passes` and `stopping_rules` are mandatory — the loader returns
/// [`crate::LoadError::SchemaError`] if either is absent.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TrainingConfig {
    /// Enable the training phase. When `false`, skip directly to simulation.
    #[serde(default = "TrainingConfig::default_enabled")]
    pub enabled: bool,

    /// Random seed for the opening scenario tree (reproducible training).
    #[serde(default)]
    pub tree_seed: Option<i64>,

    /// Number of forward-pass scenario trajectories $M$ per iteration.
    ///
    /// **Mandatory** — no default. The loader rejects any config that omits this field.
    pub forward_passes: Option<u32>,

    /// List of stopping rule configurations.
    ///
    /// **Mandatory** — no default. Must contain at least one `iteration_limit` rule.
    pub stopping_rules: Option<Vec<StoppingRuleConfig>>,

    /// How multiple stopping rules combine: `"any"` (OR) or `"all"` (AND).
    #[serde(default = "TrainingConfig::default_stopping_mode")]
    pub stopping_mode: String,

    /// Row-selection settings.
    // Rationale: the type stays algorithm-neutral (`RowSelectionConfig`) per the
    // infrastructure genericity rule, while the serialized key uses the
    // domain-standard term every practitioner types. The key/type divergence is
    // deliberate, not an unfinished rename.
    #[serde(default)]
    pub cut_selection: RowSelectionConfig,

    /// LP solver retry settings.
    #[serde(default)]
    pub solver: TrainingSolverConfig,

    /// Scenario source configuration for the training forward pass.
    /// When absent, all classes default to `in_sample`.
    #[serde(default)]
    pub scenario_source: Option<RawScenarioSourceConfig>,
}

impl TrainingConfig {
    pub(super) fn default_enabled() -> bool {
        true
    }

    pub(super) fn default_stopping_mode() -> String {
        "any".to_string()
    }

    // Note: Default impl is not provided for TrainingConfig because forward_passes
    // and stopping_rules are mandatory and have no sensible defaults.
}

/// Row-selection settings (`config.json → training.cut_selection`).
///
/// Row selection bounds the per-solve LP size by limiting how many constraint
/// rows from the row pool are carried into each solve. `selection` chooses the
/// method and carries only that method's parameters; omitting it (the default)
/// disables row selection.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RowSelectionConfig {
    /// Minimum dual-multiplier magnitude for a constraint row to count as
    /// binding at a solution point. Rows whose dual value falls below this are
    /// treated as inactive in activity tracking. Default `0.0` when absent.
    #[serde(default)]
    pub row_activity_tolerance: Option<f64>,

    /// Hard cap on active rows per stage LP, enforced after the selection
    /// method runs. Rows are evicted least-recently-active first, tie-broken by
    /// least-frequently-active; rows added in the current iteration are never
    /// evicted. `None` (default) = no cap.
    #[serde(default)]
    pub max_active_per_stage: Option<u32>,

    /// Active selection method and its parameters. Absent/`null` (default)
    /// disables row selection.
    #[serde(default)]
    pub selection: Option<SelectionMethod>,
}

/// Row-selection method and its method-specific parameters.
///
/// Internally tagged on `method`; each variant carries only the fields it uses,
/// so supplying a parameter that does not belong to the chosen method is a
/// load-time error under `deny_unknown_fields`, and a misspelled `method` is an
/// `unknown variant` error at parse time.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum SelectionMethod {
    /// Level-1: retain any row near-optimal at some visited state.
    Level1 {
        /// Tie tolerance: a row is active at a state when within this of the
        /// best row value there. Default `1e-10`.
        #[serde(default = "default_tie_tolerance")]
        tie_tolerance: f64,
        /// Iterations between periodic pruning checks. Must be `> 0`. Default `5`.
        #[serde(default = "default_check_frequency")]
        check_frequency: u32,
    },
    /// Limited-memory Level-1: retain only the oldest eligible near-optimal row
    /// per visited state.
    Lml1 {
        /// Tie tolerance: a row is active at a state when within this of the
        /// best row value there. Default `1e-10`.
        #[serde(default = "default_tie_tolerance")]
        tie_tolerance: f64,
        /// Iterations between periodic pruning checks. Must be `> 0`. Default `5`.
        #[serde(default = "default_check_frequency")]
        check_frequency: u32,
    },
    /// Domination: remove rows dominated at all visited states.
    Domination {
        /// Activity tolerance: a row survives if within this of the maximum at
        /// any visited state. Required (no default).
        domination_tolerance: f64,
        /// Iterations between periodic pruning checks. Must be `> 0`. Default `5`.
        #[serde(default = "default_check_frequency")]
        check_frequency: u32,
    },
    /// Dynamic: a per-solve lazy loop that loads only a small resident subset of
    /// rows per solve while retaining the full pool.
    Dynamic {
        /// First 1-based iteration at which the lazy loop becomes active.
        /// Must be `>= 1`. Default `2`.
        #[serde(default = "default_start_iteration")]
        start_iteration: u32,
        /// Number of most-recent iterations whose rows seed the initial resident
        /// set. `0` is valid (seeds only the current iteration). Default `5`.
        #[serde(default = "default_seed_window")]
        seed_window: u32,
        /// Only rows generated within the last `candidate_recency` iterations are
        /// scored. `None` (default) = unbounded: every pool row is a candidate,
        /// which preserves exactness. `Some(n)` (must be `>= 1`) makes the loop
        /// deliberately inexact — rows older than the window are never added.
        #[serde(default)]
        candidate_recency: Option<u32>,
        /// Maximum rows added per lazy-solve round. Must be `>= 1`. Default `10`.
        #[serde(default = "default_max_added_per_round")]
        max_added_per_round: u32,
        /// Violation tolerance for accepting a candidate row. Must be `> 0`.
        /// Default `1e-10`.
        #[serde(default = "default_violation_tolerance")]
        violation_tolerance: f64,
    },
}

fn default_tie_tolerance() -> f64 {
    1e-10
}

fn default_check_frequency() -> u32 {
    5
}

fn default_start_iteration() -> u32 {
    2
}

fn default_seed_window() -> u32 {
    5
}

fn default_max_added_per_round() -> u32 {
    10
}

fn default_violation_tolerance() -> f64 {
    1e-10
}

/// LP solver retry settings (`config.json → training.solver`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TrainingSolverConfig {
    /// Maximum solver retry attempts before propagating a hard error.
    pub retry_max_attempts: u32,

    /// Total time budget in seconds across all retry attempts for one solve.
    pub retry_time_budget_seconds: f64,
}

impl Default for TrainingSolverConfig {
    fn default() -> Self {
        Self {
            retry_max_attempts: 5,
            retry_time_budget_seconds: 30.0,
        }
    }
}

/// Deserialized configuration for one entry in `training.stopping_rules[]`.
///
/// Uses a `"type"` discriminator field (internally tagged) with `snake_case`
/// variant names matching the JSON schema.
///
/// The `GracefulShutdown` rule has no JSON representation — it is injected at
/// runtime by `StoppingRuleSet` construction and is never deserialized.
///
/// # Examples
///
/// ```
/// use cobre_io::config::StoppingRuleConfig;
///
/// let json = r#"{"type": "iteration_limit", "limit": 100}"#;
/// let rule: StoppingRuleConfig = serde_json::from_str(json).unwrap();
/// assert!(matches!(rule, StoppingRuleConfig::IterationLimit { limit: 100 }));
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum StoppingRuleConfig {
    /// Stop after a fixed number of iterations. **Mandatory** — every rule set must
    /// contain at least one `iteration_limit` rule.
    IterationLimit {
        /// Maximum iteration count $k_{max}$.
        limit: u32,
    },
    /// Stop after a wall-clock time limit.
    TimeLimit {
        /// Time limit in seconds.
        seconds: f64,
    },
    /// Stop when the lower bound stalls (relative improvement falls below tolerance).
    BoundStalling {
        /// Window size $\tau$ (number of past iterations to compare).
        iterations: u32,
        /// Relative improvement threshold.
        tolerance: f64,
    },
    /// Stop when both the bound and simulated policy costs have stabilized.
    Simulation {
        /// Number of Monte Carlo forward simulations per check.
        replications: u32,
        /// Iterations between checks.
        period: u32,
        /// Number of past iterations for bound stability check.
        bound_window: u32,
        /// Normalized distance threshold between consecutive simulation results.
        distance_tol: f64,
        /// Relative tolerance for bound stability.
        bound_tol: f64,
    },
}

/// Upper-bound evaluation settings (`config.json → upper_bound_evaluation`).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct UpperBoundEvaluationConfig {
    /// Enable vertex-based inner approximation for upper bound computation.
    #[serde(default)]
    pub enabled: Option<bool>,

    /// First iteration to compute the upper bound.
    #[serde(default)]
    pub initial_iteration: Option<u32>,

    /// Iterations between upper-bound evaluations.
    #[serde(default)]
    pub interval_iterations: Option<u32>,

    /// Lipschitz constant settings.
    #[serde(default)]
    pub lipschitz: LipschitzConfig,
}

/// Lipschitz constant settings for inner approximation.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LipschitzConfig {
    /// Computation mode: `"auto"`.
    #[serde(default)]
    pub mode: Option<String>,

    /// Fallback value when automatic computation fails.
    #[serde(default)]
    pub fallback_value: Option<f64>,

    /// Multiplicative safety margin applied to computed Lipschitz constants.
    #[serde(default)]
    pub scale_factor: Option<f64>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::{SelectionMethod, TrainingConfig};

    /// A `dynamic` selection block round-trips through the tagged enum, with
    /// every method-specific field landing in the `Dynamic` variant.
    #[test]
    fn dynamic_selection_block_round_trips() {
        let json = r#"{
            "forward_passes": 4,
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "cut_selection": {
                "row_activity_tolerance": 1e-6,
                "max_active_per_stage": 4000,
                "selection": {
                    "method": "dynamic",
                    "start_iteration": 5,
                    "seed_window": 0,
                    "candidate_recency": 20,
                    "max_added_per_round": 3,
                    "violation_tolerance": 1e-9
                }
            }
        }"#;
        let cfg: TrainingConfig = serde_json::from_str(json).unwrap();
        let cs = &cfg.cut_selection;
        assert_eq!(cs.row_activity_tolerance, Some(1e-6));
        assert_eq!(cs.max_active_per_stage, Some(4000));
        match cs.selection.as_ref().expect("selection present") {
            SelectionMethod::Dynamic {
                start_iteration,
                seed_window,
                candidate_recency,
                max_added_per_round,
                violation_tolerance,
            } => {
                assert_eq!(*start_iteration, 5);
                assert_eq!(*seed_window, 0);
                assert_eq!(*candidate_recency, Some(20));
                assert_eq!(*max_added_per_round, 3);
                assert!((*violation_tolerance - 1e-9).abs() < f64::EPSILON);
            }
            other => panic!("expected Dynamic, got {other:?}"),
        }
    }

    /// A `level1` selection block round-trips and fills the variant defaults
    /// when its fields are omitted.
    #[test]
    fn level1_selection_block_round_trips_with_defaults() {
        let json = r#"{
            "forward_passes": 4,
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "cut_selection": { "selection": { "method": "level1" } }
        }"#;
        let cfg: TrainingConfig = serde_json::from_str(json).unwrap();
        match cfg
            .cut_selection
            .selection
            .as_ref()
            .expect("selection present")
        {
            SelectionMethod::Level1 {
                tie_tolerance,
                check_frequency,
            } => {
                assert!((*tie_tolerance - 1e-10).abs() < 1e-20);
                assert_eq!(*check_frequency, 5);
            }
            other => panic!("expected Level1, got {other:?}"),
        }
    }

    /// Omitting `selection` disables row selection (the default).
    #[test]
    fn omitting_selection_disables_row_selection() {
        let json = r#"{
            "forward_passes": 4,
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "cut_selection": {}
        }"#;
        let cfg: TrainingConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.cut_selection.selection.is_none());
    }

    /// A parameter that belongs to a different method is a deserialize error
    /// under `deny_unknown_fields` (here `max_added_per_round` under `level1`).
    #[test]
    fn wrong_method_field_is_deserialize_error() {
        let json = r#"{
            "forward_passes": 4,
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "cut_selection": {
                "selection": { "method": "level1", "max_added_per_round": 3 }
            }
        }"#;
        let result = serde_json::from_str::<TrainingConfig>(json);
        assert!(
            result.is_err(),
            "a Dynamic-only field under level1 must be rejected"
        );
    }

    /// A misspelled `method` is an unknown-variant deserialize error.
    #[test]
    fn bad_method_string_is_deserialize_error() {
        let json = r#"{
            "forward_passes": 4,
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "cut_selection": { "selection": { "method": "dynmic" } }
        }"#;
        let result = serde_json::from_str::<TrainingConfig>(json);
        assert!(result.is_err(), "an unknown method tag must be rejected");
    }

    /// `domination` without its required `domination_tolerance` is a
    /// missing-field deserialize error.
    #[test]
    fn domination_without_tolerance_is_missing_field_error() {
        let json = r#"{
            "forward_passes": 4,
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "cut_selection": { "selection": { "method": "domination" } }
        }"#;
        let result = serde_json::from_str::<TrainingConfig>(json);
        assert!(
            result.is_err(),
            "domination requires domination_tolerance; absence must be rejected"
        );
    }
}
