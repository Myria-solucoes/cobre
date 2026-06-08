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
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RowSelectionConfig {
    /// Enable row pruning.
    #[serde(default)]
    pub enabled: Option<bool>,

    /// Method: `"level1"`, `"lml1"`, `"domination"`, or `"dynamic"`.
    #[serde(default)]
    pub method: Option<String>,

    /// Deprecated. Silently ignored. Retained for backward compatibility with
    /// existing config files.
    ///
    /// Previously used as an activity-count threshold for the `"level1"`
    /// row-selection method. The correct algorithm (de Matos 2015) is
    /// value-based and uses `tie_tolerance` instead.
    #[serde(default)]
    pub threshold: Option<u32>,

    /// Deprecated. Silently ignored. Retained for backward compatibility with
    /// existing config files.
    ///
    /// Previously used as the memory window size for the `"lml1"` method.
    /// The correct algorithm (Guigues 2017/2019) is value-based and uses
    /// `tie_tolerance` instead.
    #[serde(default)]
    pub memory_window: Option<u32>,

    /// Tie-tolerance for the `"level1"` and `"lml1"` value-based row-selection
    /// methods.
    ///
    /// A row is considered tied at an evaluation point when its value is
    /// within `tie_tolerance` of the best active row value. Default: `1e-10`.
    /// Ignored by the `"domination"` method.
    #[serde(default)]
    pub tie_tolerance: Option<f64>,

    /// Epsilon for the `"domination"` method.
    ///
    /// Required when `method = "domination"`. Accepts fractional values
    /// (e.g., `1e-6`).
    #[serde(default)]
    pub domination_epsilon: Option<f64>,

    /// Iterations between periodic pruning checks for the `"level1"`, `"lml1"`,
    /// and `"domination"` methods. Must be `> 0`. Not used by `method =
    /// "dynamic"`, which uses `active_window` for its `k2` active-set seed
    /// window.
    #[serde(default)]
    pub check_frequency: Option<u32>,

    /// First 1-based training iteration at which the dynamic lazy loop becomes
    /// active. Ignored unless `method = "dynamic"`. Default: `2`.
    #[serde(default)]
    pub start_iteration: Option<u32>,

    /// Active-set seed window (`k2`) for `method = "dynamic"`: the number of
    /// most recent iterations whose cuts seed the initial active set. `0` is
    /// valid and meaningful — it seeds only the current iteration's cuts,
    /// matching NEWAVE's `selcor.dat`
    /// `TAMANHO DA JANELA DE CORTES ATIVOS (k2) = 0`. Default: `5`. Ignored
    /// unless `method = "dynamic"`.
    #[serde(default)]
    pub active_window: Option<u32>,

    /// Candidate-recency window for `method = "dynamic"`: only candidates
    /// generated within the last `candidate_window` iterations are scored.
    /// `None` (the default) means an unbounded window (every pool entry is a
    /// candidate). Ignored unless `method = "dynamic"`.
    #[serde(default)]
    pub candidate_window: Option<u32>,

    /// Maximum number of cuts added per lazy-solve round for `method =
    /// "dynamic"` (`>= 1`). Default: `10`. Ignored unless `method = "dynamic"`.
    #[serde(default)]
    pub nadic: Option<u32>,

    /// Violation tolerance for accepting a candidate row under `method =
    /// "dynamic"` (`> 0`). When absent, falls back to `tie_tolerance`. Default:
    /// `1e-10`. Ignored unless `method = "dynamic"`.
    #[serde(default)]
    pub violation_tolerance: Option<f64>,

    /// Minimum dual multiplier magnitude for a constraint row to be counted as
    /// binding at a given solution point.
    ///
    /// Rows whose dual value falls below this threshold are treated as inactive
    /// in activity-tracking computations. Increase to reduce noise from
    /// near-zero duals; decrease to be more inclusive.
    #[serde(default)]
    pub cut_activity_tolerance: Option<f64>,

    /// DEPRECATED. Silently ignored as of this release; will be removed in
    /// the next. Retained for one release so existing config files continue
    /// to deserialise.
    ///
    /// Previously controlled a sliding-window mask used during basis
    /// reconstruction. Reconstruction now matches stored constraint rows by
    /// slot identity alone, which makes the window value unobservable. See
    /// `CHANGELOG.md` for the deprecation announcement.
    #[serde(default)]
    #[deprecated(
        since = "0.1.0",
        note = "field is ignored; basis reconstruction now uses slot-identity matching. Will be removed next release."
    )]
    pub basis_activity_window: Option<u32>,

    /// Row budget per stage: maximum number of constraint rows allowed to be
    /// active in a single stage's LP.
    ///
    /// When `Some(n)`, the training loop enforces a hard cap of `n` active rows
    /// per stage after the pruning strategy has been applied. Rows are evicted
    /// in order of staleness (least recently active first), tie-broken by
    /// overall usage frequency (least frequently active first). Rows added in
    /// the current iteration are never evicted.
    ///
    /// When `None` (the default), no hard cap is enforced.
    #[serde(default)]
    pub max_active_per_stage: Option<u32>,
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
#[allow(clippy::unwrap_used)]
mod tests {
    use super::TrainingConfig;

    /// The four first-class dynamic cut-selection fields deserialize under
    /// `deny_unknown_fields`.
    #[test]
    fn dynamic_cut_selection_fields_deserialize() {
        let json = r#"{
            "forward_passes": 4,
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "cut_selection": {
                "enabled": true,
                "method": "dynamic",
                "start_iteration": 5,
                "candidate_window": 20,
                "nadic": 3,
                "violation_tolerance": 1e-9,
                "check_frequency": 7
            }
        }"#;
        let cfg: TrainingConfig = serde_json::from_str(json).unwrap();
        let cs = &cfg.cut_selection;
        assert_eq!(cs.method.as_deref(), Some("dynamic"));
        assert_eq!(cs.start_iteration, Some(5));
        assert_eq!(cs.candidate_window, Some(20));
        assert_eq!(cs.nadic, Some(3));
        assert_eq!(cs.violation_tolerance, Some(1e-9));
        assert_eq!(cs.check_frequency, Some(7));
    }

    /// Omitting the new fields leaves them `None` (backward compatible).
    #[test]
    fn dynamic_cut_selection_fields_default_to_none() {
        let json = r#"{
            "forward_passes": 4,
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "cut_selection": { "enabled": true, "method": "dynamic" }
        }"#;
        let cfg: TrainingConfig = serde_json::from_str(json).unwrap();
        let cs = &cfg.cut_selection;
        assert_eq!(cs.start_iteration, None);
        assert_eq!(cs.candidate_window, None);
        assert_eq!(cs.nadic, None);
        assert_eq!(cs.violation_tolerance, None);
    }
}
