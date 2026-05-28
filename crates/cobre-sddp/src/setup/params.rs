//! `StudyParams`, `ConstructionConfig`, and associated constants extracted from `setup/mod.rs`.

use cobre_core::ScalarParameter;
use cobre_io::config::StoppingRuleConfig;

use crate::{
    InflowNonNegativityMethod, SddpError,
    cut_selection::{CutSelectionStrategy, parse_cut_selection_config},
    stopping_rule::{StoppingMode, StoppingRule, StoppingRuleSet},
};

/// Default number of forward-pass trajectories when not specified in config.
pub const DEFAULT_FORWARD_PASSES: u32 = 1;

/// Default maximum iterations when no stopping rule specifies an iteration limit.
pub const DEFAULT_MAX_ITERATIONS: u64 = 100;

/// Default random seed for stochastic scenario generation.
pub const DEFAULT_SEED: u64 = 42;

// ---------------------------------------------------------------------------
// StudyParams
// ---------------------------------------------------------------------------

/// Scalar parameters extracted from a [`cobre_io::Config`].
///
/// Centralises config-to-domain conversion for both [`StudySetup::new`](super::StudySetup::new)
/// and `BroadcastConfig::from_config`. The struct owns all
/// values so it can be passed by value without lifetime dependencies.
#[derive(Debug, Clone)]
pub struct StudyParams {
    /// Random seed for noise generation.
    pub seed: u64,
    /// Number of forward-pass trajectories per training iteration.
    pub forward_passes: u32,
    /// Stopping rule set (rules + mode) governing when training halts.
    pub stopping_rule_set: StoppingRuleSet,
    /// Number of simulation scenarios (0 if simulation is disabled).
    pub n_scenarios: u32,
    /// Buffer capacity for the simulation output channel.
    pub io_channel_capacity: usize,
    /// Policy directory path string.
    pub policy_path: String,
    /// Inflow non-negativity enforcement method.
    pub inflow_method: InflowNonNegativityMethod,
    /// Optional cut selection strategy (None means cut selection is disabled).
    pub cut_selection: Option<CutSelectionStrategy>,
    /// Minimum dual multiplier for a cut to count as binding (`0.0` if unset).
    pub cut_activity_tolerance: f64,
    /// Maximum number of active cuts per stage (hard cap on LP size).
    ///
    /// `None` means no cap is enforced. Derived from
    /// `config.training.cut_selection.max_active_per_stage`.
    pub budget: Option<u32>,
}

impl StudyParams {
    /// Extract study parameters from a validated [`cobre_io::Config`].
    ///
    /// # Errors
    ///
    /// - [`SddpError::Validation`] if cut selection config is invalid.
    pub fn from_config(config: &cobre_io::Config) -> Result<Self, SddpError> {
        let seed = config
            .training
            .tree_seed
            .map_or(DEFAULT_SEED, i64::unsigned_abs);

        let forward_passes = config
            .training
            .forward_passes
            .unwrap_or(DEFAULT_FORWARD_PASSES);

        let rule_configs = match &config.training.stopping_rules {
            Some(rules) if !rules.is_empty() => rules.clone(),
            _ => vec![StoppingRuleConfig::IterationLimit {
                limit: u32::try_from(DEFAULT_MAX_ITERATIONS).unwrap_or(u32::MAX),
            }],
        };

        let stopping_rules: Vec<StoppingRule> = rule_configs
            .into_iter()
            .map(|c| match c {
                StoppingRuleConfig::IterationLimit { limit } => Ok(StoppingRule::IterationLimit {
                    limit: u64::from(limit),
                }),
                StoppingRuleConfig::TimeLimit { seconds } => {
                    Ok(StoppingRule::TimeLimit { seconds })
                }
                StoppingRuleConfig::BoundStalling {
                    iterations,
                    tolerance,
                } => Ok(StoppingRule::BoundStalling {
                    iterations: u64::from(iterations),
                    tolerance,
                }),
                StoppingRuleConfig::Simulation { .. } => Err(SddpError::Validation(
                    "simulation-based stopping rule is not yet implemented; \
                     use iteration_limit, time_limit, or bound_stalling"
                        .to_string(),
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let stopping_mode = if config.training.stopping_mode.eq_ignore_ascii_case("all") {
            StoppingMode::All
        } else {
            StoppingMode::Any
        };

        let stopping_rule_set = StoppingRuleSet {
            rules: stopping_rules,
            mode: stopping_mode,
        };

        let n_scenarios = if config.simulation.enabled {
            config.simulation.num_scenarios
        } else {
            0
        };

        let io_channel_capacity =
            usize::try_from(config.simulation.io_channel_capacity).unwrap_or(64);

        let policy_path = config.policy.path.clone();

        let inflow_method = InflowNonNegativityMethod::from(&config.modeling.inflow_non_negativity);

        let cut_selection = parse_cut_selection_config(&config.training.cut_selection)
            .map_err(|msg| SddpError::Validation(format!("cut_selection config error: {msg}")))?;

        let cut_activity_tolerance = config
            .training
            .cut_selection
            .cut_activity_tolerance
            .unwrap_or(0.0);

        // Emit a one-shot deprecation warning when the user-supplied TOML carries
        // `training.cut_selection.basis_activity_window`. The field has no
        // internal consumer after the basis-reconstruction classifier was
        // removed; reading it here, ignoring the value, and warning preserves
        // backward compatibility for one release. The field itself stays on
        // `RowSelectionConfig` (with `#[deprecated]`) and is dropped from the
        // schema in the next release.
        #[allow(deprecated)]
        if config
            .training
            .cut_selection
            .basis_activity_window
            .is_some()
        {
            tracing::warn!(
                "training.cut_selection.basis_activity_window is deprecated and \
                 will be removed in the next release; the value is ignored \
                 because basis reconstruction now matches stored cut rows by \
                 slot identity alone. Please remove the field from config.json."
            );
        }

        let budget = config.training.cut_selection.max_active_per_stage;

        // Warn when the budget is so tight that every iteration will immediately
        // evict all cuts older than the current one.  This is not an error —
        // the solver remains correct — but it usually indicates a misconfiguration.
        if let Some(b) = budget {
            // world_size is not available here; use 1 as a conservative estimate.
            // The CLI/Python layer may emit a more precise warning with the real
            // world_size after broadcast.
            if u64::from(b) < u64::from(forward_passes) {
                tracing::warn!(
                    "max_active_per_stage ({b}) is less than forward_passes \
                     ({forward_passes}); budget enforcement will evict all \
                     non-current-iteration cuts every iteration"
                );
            }
        }

        Ok(Self {
            seed,
            forward_passes,
            stopping_rule_set,
            n_scenarios,
            io_channel_capacity,
            policy_path,
            inflow_method,
            cut_selection,
            cut_activity_tolerance,
            budget,
        })
    }

    /// Convert into a [`ConstructionConfig`] for [`StudySetup::from_broadcast_params`](super::StudySetup::from_broadcast_params).
    ///
    /// Sets `export_states = false`; callers should use
    /// [`StudySetup::set_export_states`](super::StudySetup::set_export_states) to enable state export after construction.
    #[must_use]
    pub fn into_construction_config(self) -> ConstructionConfig {
        ConstructionConfig {
            seed: self.seed,
            forward_passes: self.forward_passes,
            stopping_rule_set: self.stopping_rule_set,
            n_scenarios: self.n_scenarios,
            io_channel_capacity: self.io_channel_capacity,
            policy_path: self.policy_path,
            inflow_method: self.inflow_method,
            cut_selection: self.cut_selection,
            cut_activity_tolerance: self.cut_activity_tolerance,
            budget: self.budget,
            export_states: false,
            scalar_parameters: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ConstructionConfig
// ---------------------------------------------------------------------------

/// Scalar and config parameters bundled for [`StudySetup::from_broadcast_params`](super::StudySetup::from_broadcast_params).
///
/// Groups parameters to reduce argument count. Construct via
/// [`StudyParams::into_construction_config`] from a [`cobre_io::Config`],
/// or populate fields directly from a broadcast config.
#[derive(Debug, Clone)]
pub struct ConstructionConfig {
    /// Random seed for noise generation.
    pub seed: u64,
    /// Number of forward-pass trajectories per training iteration.
    pub forward_passes: u32,
    /// Stopping rule set (rules + mode) governing when training halts.
    pub stopping_rule_set: StoppingRuleSet,
    /// Number of simulation scenarios (0 if simulation is disabled).
    pub n_scenarios: u32,
    /// Buffer capacity for the simulation output channel.
    pub io_channel_capacity: usize,
    /// Policy directory path string.
    pub policy_path: String,
    /// Inflow non-negativity enforcement method.
    pub inflow_method: InflowNonNegativityMethod,
    /// Optional cut selection strategy (`None` means cut selection is disabled).
    pub cut_selection: Option<CutSelectionStrategy>,
    /// Minimum dual multiplier for a cut to count as binding (`0.0` if unset).
    pub cut_activity_tolerance: f64,
    /// Maximum number of active cuts per stage (hard cap on LP size).
    ///
    /// `None` means no cap is enforced. Derived from
    /// `config.training.cut_selection.max_active_per_stage`.
    pub budget: Option<u32>,
    /// Whether the caller wants the visited-states archive for export.
    ///
    /// When `true`, the archive is allocated during training regardless of the
    /// cut selection strategy. Defaults to `false`; set based on
    /// `exports.states`.
    pub export_states: bool,
    /// Loaded `system/scalar_parameters.json` entries, or empty when the file is
    /// absent or the manifest flag `system_scalar_parameters_json` is `false`.
    /// Consumed by `build_resolved_parameters` to populate the per-`(parameter_id,
    /// stage_idx)` lookup table used by the LP builder.
    pub scalar_parameters: Vec<ScalarParameter>,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::{Arc, Mutex};

    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use tracing::{Event, Level, Metadata, Subscriber, span};

    use super::StudyParams;

    // ---------------------------------------------------------------------------
    // Minimal WARN-capturing subscriber for use in tests.
    // ---------------------------------------------------------------------------

    /// Records all WARN-level event messages into a shared `Vec<String>`.
    struct WarnRecorder {
        messages: Arc<Mutex<Vec<String>>>,
    }

    impl WarnRecorder {
        fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
            let messages = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    messages: Arc::clone(&messages),
                },
                messages,
            )
        }
    }

    impl Subscriber for WarnRecorder {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            *metadata.level() <= Level::WARN
        }

        fn new_span(&self, _attrs: &span::Attributes<'_>) -> span::Id {
            span::Id::from_u64(1)
        }

        fn record(&self, _span: &span::Id, _values: &span::Record<'_>) {}

        fn record_follows_from(&self, _span: &span::Id, _follows: &span::Id) {}

        fn event(&self, event: &Event<'_>) {
            if *event.metadata().level() == Level::WARN {
                struct MessageVisitor(String);
                impl tracing::field::Visit for MessageVisitor {
                    fn record_debug(
                        &mut self,
                        field: &tracing::field::Field,
                        value: &dyn std::fmt::Debug,
                    ) {
                        if field.name() == "message" {
                            self.0 = format!("{value:?}");
                        }
                    }

                    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                        if field.name() == "message" {
                            self.0 = value.to_string();
                        }
                    }
                }
                let mut visitor = MessageVisitor(String::new());
                event.record(&mut visitor);
                self.messages.lock().unwrap().push(visitor.0);
            }
        }

        fn enter(&self, _span: &span::Id) {}

        fn exit(&self, _span: &span::Id) {}
    }

    /// Build a minimal `cobre_io::Config` with the given
    /// `basis_activity_window` value in `training.cut_selection`. The field
    /// is deprecated and its value is ignored by `StudyParams::from_config`;
    /// the helper drives the deprecation-warning tests below.
    #[allow(deprecated)]
    fn config_with_window(window: Option<u32>) -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                forward_passes: Some(1),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: "any".to_string(),
                cut_selection: RowSelectionConfig {
                    basis_activity_window: window,
                    ..RowSelectionConfig::default()
                },
                solver: TrainingSolverConfig::default(),
                scenario_source: None,
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
            energy: cobre_io::EnergyConfig::default(),
        }
    }

    /// `StudyParams::from_config` must emit a WARN-level tracing event
    /// naming `basis_activity_window` whenever the user supplies the field,
    /// regardless of the value (including formerly out-of-range values).
    /// Construction must succeed because the value is ignored.
    #[test]
    fn study_params_warns_on_deprecated_basis_activity_window() {
        for window in [Some(0), Some(5), Some(31), Some(100)] {
            let (subscriber, messages) = WarnRecorder::new();
            tracing::subscriber::with_default(subscriber, || {
                StudyParams::from_config(&config_with_window(window))
                    .expect("any basis_activity_window value must succeed (field is ignored)");
            });
            let recorded = messages.lock().unwrap();
            let relevant: Vec<&str> = recorded
                .iter()
                .map(std::string::String::as_str)
                .filter(|msg| msg.contains("basis_activity_window"))
                .collect();
            assert!(
                !relevant.is_empty(),
                "expected a deprecation WARN mentioning 'basis_activity_window' for value {window:?}, got: {recorded:?}"
            );
            assert!(
                relevant.iter().any(|msg| msg.contains("deprecated")),
                "deprecation WARN must contain the word 'deprecated' for value {window:?}, got: {relevant:?}"
            );
            assert!(
                relevant.iter().any(|msg| msg.contains("ignored")),
                "deprecation WARN must say the value is 'ignored' for value {window:?}, got: {relevant:?}"
            );
        }
    }

    /// `StudyParams::from_config` must NOT emit a `basis_activity_window`
    /// WARN when the field is absent from the user config.
    #[test]
    fn study_params_silent_when_basis_activity_window_absent() {
        let (subscriber, messages) = WarnRecorder::new();
        tracing::subscriber::with_default(subscriber, || {
            StudyParams::from_config(&config_with_window(None))
                .expect("absent basis_activity_window must succeed");
        });
        let recorded = messages.lock().unwrap();
        let relevant: Vec<&str> = recorded
            .iter()
            .map(std::string::String::as_str)
            .filter(|msg| msg.contains("basis_activity_window"))
            .collect();
        assert!(
            relevant.is_empty(),
            "no deprecation WARN must fire when basis_activity_window is absent, got: {recorded:?}"
        );
    }

    /// Build a minimal `cobre_io::Config` with `max_active_per_stage` and
    /// `forward_passes` set so that the budget-below-forward-passes warning fires.
    fn config_with_budget_below_forward_passes() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                forward_passes: Some(2),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: "any".to_string(),
                cut_selection: RowSelectionConfig {
                    max_active_per_stage: Some(1),
                    ..RowSelectionConfig::default()
                },
                solver: TrainingSolverConfig::default(),
                scenario_source: None,
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
            energy: cobre_io::EnergyConfig::default(),
        }
    }

    /// Build a minimal `cobre_io::Config` whose stopping rules contain a
    /// `Simulation` entry.
    fn config_with_simulation_stopping_rule() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                forward_passes: Some(1),
                stopping_rules: Some(vec![StoppingRuleConfig::Simulation {
                    replications: 100,
                    period: 12,
                    bound_window: 10,
                    distance_tol: 0.05,
                    bound_tol: 0.01,
                }]),
                stopping_mode: "any".to_string(),
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                scenario_source: None,
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
            energy: cobre_io::EnergyConfig::default(),
        }
    }

    /// AC: `from_config` must return `SddpError::Validation` when the stopping
    /// rules list contains a `simulation_based` entry, because the feature is
    /// not yet implemented. Silent no-op (fold into iteration limit) is
    /// forbidden.
    #[test]
    fn from_config_rejects_simulation_stopping_rule() {
        use crate::SddpError;

        let err = StudyParams::from_config(&config_with_simulation_stopping_rule())
            .expect_err("Simulation stopping rule must be rejected");
        assert!(
            matches!(err, SddpError::Validation(_)),
            "expected SddpError::Validation, got: {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("simulation-based stopping rule"),
            "error message must mention 'simulation-based stopping rule'; got: {msg}"
        );
        assert!(
            msg.contains("not yet implemented"),
            "error message must say 'not yet implemented'; got: {msg}"
        );
    }

    /// AC: when `max_active_per_stage` is less than `forward_passes`, `StudyParams::from_config`
    /// emits a WARN-level tracing event whose message contains `max_active_per_stage`.
    #[test]
    fn study_params_warns_when_budget_below_forward_passes() {
        let (subscriber, messages) = WarnRecorder::new();
        tracing::subscriber::with_default(subscriber, || {
            let _params = StudyParams::from_config(&config_with_budget_below_forward_passes())
                .expect("config is valid; warning must not prevent construction");
        });
        let recorded = messages.lock().unwrap();
        let relevant: Vec<&str> = recorded
            .iter()
            .map(std::string::String::as_str)
            .filter(|msg| msg.contains("max_active_per_stage"))
            .collect();
        assert!(
            !relevant.is_empty(),
            "expected at least one WARN event containing 'max_active_per_stage', got: {recorded:?}"
        );
    }
}
