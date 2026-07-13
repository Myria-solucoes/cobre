//! Postcard-serializable types for MPI broadcast from rank 0 to all ranks.

use cobre_comm::Communicator;
use cobre_core::scenario::ScenarioSource;
use cobre_io::Config;
use cobre_io::PolicyMode;
use cobre_sddp::{
    CutSelectionStrategy, DEFAULT_MAX_ITERATIONS, InflowNonNegativityMethod, StoppingMode,
    StoppingRule, StoppingRuleSet, StudyParams,
};

use crate::error::CliError;

use serde::Serialize;
use serde::de::DeserializeOwned;

/// Postcard-serializable stopping rule.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) enum BroadcastStoppingRule {
    IterationLimit { limit: u64 },
    TimeLimit { seconds: f64 },
    BoundStalling { iterations: u64, tolerance: f64 },
}

/// Postcard-serializable stopping mode.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(crate) enum BroadcastStoppingMode {
    Any,
    All,
}

/// Configuration snapshot broadcast from rank 0 to all ranks.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct BroadcastConfig {
    pub(crate) seed: u64,
    pub(crate) forward_passes: u32,
    pub(crate) stopping_rules: Vec<BroadcastStoppingRule>,
    pub(crate) stopping_mode: BroadcastStoppingMode,
    pub(crate) n_scenarios: u32,
    pub(crate) io_channel_capacity: u32,
    pub(crate) policy_path: String,
    pub(crate) inflow_method: InflowNonNegativityMethod,
    pub(crate) cut_selection: Option<CutSelectionStrategy>,
    pub(crate) cut_activity_tolerance: f64,
    /// When `false`, all ranks skip training and proceed to simulation (or exit).
    pub(crate) training_enabled: bool,
    /// Policy initialization mode.
    pub(crate) policy_mode: PolicyMode,
    /// Whether the visited-states archive is allocated for export.
    pub(crate) export_states: bool,
    /// Hard cap on active rows per stage; `None` means no cap. Sourced from
    /// `config.training.cut_selection.max_active_per_stage`.
    pub(crate) budget: Option<u32>,
    /// Scenario source for the training forward pass, broadcast so non-root
    /// ranks build the stochastic context with matching sampling schemes.
    pub(crate) training_source: ScenarioSource,
    /// Scenario source for the post-training simulation forward pass.
    pub(crate) simulation_source: ScenarioSource,
}

impl BroadcastConfig {
    pub(crate) fn from_config(config: &Config) -> Result<Self, CliError> {
        let params = StudyParams::from_config(config).map_err(CliError::from)?;
        // Sentinel path: the scenario-source helpers use it only for historical-years
        // look-up and error messages, neither exercised here.
        let sentinel_path = std::path::Path::new("config.json");
        let training_source = config
            .training_scenario_source(sentinel_path)
            .map_err(CliError::from)?;
        let simulation_source = config
            .simulation_scenario_source(sentinel_path)
            .map_err(CliError::from)?;

        let stopping_rules = params
            .stopping_rule_set
            .rules
            .iter()
            .map(|r| match r {
                StoppingRule::IterationLimit { limit } => {
                    BroadcastStoppingRule::IterationLimit { limit: *limit }
                }
                StoppingRule::TimeLimit { seconds } => {
                    BroadcastStoppingRule::TimeLimit { seconds: *seconds }
                }
                StoppingRule::BoundStalling {
                    iterations,
                    tolerance,
                } => BroadcastStoppingRule::BoundStalling {
                    iterations: *iterations,
                    tolerance: *tolerance,
                },
                // SimulationBased and GracefulShutdown evaluate on rank 0 only and are
                // not broadcastable; non-root ranks fall back to an iteration limit.
                StoppingRule::SimulationBased { .. } | StoppingRule::GracefulShutdown => {
                    tracing::warn!(
                        "stopping rule not broadcastable, \
                         substituting IterationLimit({DEFAULT_MAX_ITERATIONS})"
                    );
                    BroadcastStoppingRule::IterationLimit {
                        limit: DEFAULT_MAX_ITERATIONS,
                    }
                }
            })
            .collect();

        let stopping_mode = match params.stopping_rule_set.mode {
            StoppingMode::All => BroadcastStoppingMode::All,
            StoppingMode::Any => BroadcastStoppingMode::Any,
        };

        let cut_selection = params.cut_selection.clone();

        Ok(Self {
            seed: params.seed,
            forward_passes: params.forward_passes,
            stopping_rules,
            stopping_mode,
            n_scenarios: params.n_scenarios,
            io_channel_capacity: u32::try_from(params.io_channel_capacity).unwrap_or(64),
            policy_path: params.policy_path,
            inflow_method: params.inflow_method,
            cut_selection,
            cut_activity_tolerance: params.cut_activity_tolerance,
            training_enabled: config.training.enabled,
            policy_mode: config.policy.mode,
            export_states: config.exports.states,
            budget: params.budget,
            training_source,
            simulation_source,
        })
    }
}

/// Postcard-serializable wrapper for [`OpeningTree`] broadcast.
///
/// Reconstructs the tree via [`OpeningTree::from_parts`] on all ranks.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct BroadcastOpeningTree {
    pub(crate) data: Vec<f64>,
    pub(crate) openings_per_stage: Vec<usize>,
    pub(crate) dim: usize,
}

pub(crate) fn stopping_rules_from_broadcast(cfg: &BroadcastConfig) -> StoppingRuleSet {
    let rules = cfg
        .stopping_rules
        .iter()
        .map(|r| match r {
            BroadcastStoppingRule::IterationLimit { limit } => {
                StoppingRule::IterationLimit { limit: *limit }
            }
            BroadcastStoppingRule::TimeLimit { seconds } => {
                StoppingRule::TimeLimit { seconds: *seconds }
            }
            BroadcastStoppingRule::BoundStalling {
                iterations,
                tolerance,
            } => StoppingRule::BoundStalling {
                iterations: *iterations,
                tolerance: *tolerance,
            },
        })
        .collect();

    let mode = match cfg.stopping_mode {
        BroadcastStoppingMode::All => StoppingMode::All,
        BroadcastStoppingMode::Any => StoppingMode::Any,
    };

    StoppingRuleSet { rules, mode }
}

/// Broadcast a serializable value from rank 0 to all ranks.
///
/// A broadcast length of 0 signals rank 0 failure, letting all ranks return an error
/// in lockstep rather than deadlocking.
///
/// # Errors
///
/// Returns [`CliError::Internal`] on serialization, broadcast, or deserialization failure.
pub(crate) fn broadcast_value<T, C>(value: Option<T>, comm: &C) -> Result<T, CliError>
where
    T: Serialize + DeserializeOwned,
    C: Communicator,
{
    let is_root = comm.rank() == 0;

    let serialized: Vec<u8> = if is_root {
        match value {
            Some(ref v) => postcard::to_allocvec(v).map_err(|e| CliError::Internal {
                message: format!("serialization error: {e}"),
            })?,
            None => Vec::new(),
        }
    } else {
        Vec::new()
    };

    let raw_len = serialized.len();
    #[allow(clippy::cast_possible_truncation)]
    let mut len_buf = [raw_len as u64];
    comm.broadcast(&mut len_buf, 0)
        .map_err(|e| CliError::Internal {
            message: format!("broadcast error (length): {e}"),
        })?;

    let len = usize::try_from(len_buf[0]).map_err(|e| CliError::Internal {
        message: format!("broadcast error (length conversion): {e}"),
    })?;
    if len == 0 {
        return Err(CliError::Internal {
            message: "rank 0 signaled broadcast failure (length 0)".to_string(),
        });
    }

    let mut bytes = if is_root { serialized } else { vec![0u8; len] };
    comm.broadcast(&mut bytes, 0)
        .map_err(|e| CliError::Internal {
            message: format!("broadcast error (data): {e}"),
        })?;

    if is_root {
        value.ok_or_else(|| CliError::Internal {
            message: "broadcast_value: root value disappeared after serialization".to_string(),
        })
    } else {
        postcard::from_bytes(&bytes).map_err(|e| CliError::Internal {
            message: format!("deserialization error: {e}"),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::{BroadcastOpeningTree, broadcast_value};

    #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Simple {
        x: f64,
        label: String,
    }

    #[test]
    fn broadcast_value_local_round_trips_simple() {
        let comm = cobre_comm::LocalBackend;
        let original = Simple {
            x: std::f64::consts::PI,
            label: "test".to_string(),
        };
        let result = broadcast_value(Some(original.clone()), &comm).unwrap();
        assert_eq!(result, original);
    }

    #[test]
    fn broadcast_value_local_round_trips_vec() {
        let comm = cobre_comm::LocalBackend;
        let original: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let result = broadcast_value(Some(original.clone()), &comm).unwrap();
        assert_eq!(result, original);
    }

    #[test]
    fn broadcast_value_local_round_trips_config_like() {
        #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
        struct ConfigLike {
            forward_passes: u32,
            seed: Option<i64>,
        }

        let comm = cobre_comm::LocalBackend;
        let original = ConfigLike {
            forward_passes: 4,
            seed: Some(42),
        };
        let result = broadcast_value(Some(original.clone()), &comm).unwrap();
        assert_eq!(result, original);
    }

    #[test]
    fn broadcast_value_returns_err_when_root_passes_none() {
        let comm = cobre_comm::LocalBackend;
        let result: Result<Simple, _> = broadcast_value(None, &comm);
        assert!(result.is_err(), "expected Err when root passes None");
        let err = result.unwrap_err();
        assert!(
            matches!(err, crate::error::CliError::Internal { .. }),
            "expected CliError::Internal, got: {err:?}"
        );
    }

    /// Gated on `mpi`: exercises via `LocalBackend` the same path MPI runs invoke.
    #[cfg(feature = "mpi")]
    #[test]
    fn broadcast_value_round_trips_u64() {
        let comm = cobre_comm::LocalBackend;
        let value: u64 = 42;
        let result = broadcast_value(Some(value), &comm).unwrap();
        assert_eq!(result, 42u64);
    }

    // ------------------------------------------------------------------
    // BroadcastOpeningTree tests
    // ------------------------------------------------------------------

    #[test]
    fn broadcast_opening_tree_round_trips_via_postcard() {
        let original = BroadcastOpeningTree {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            openings_per_stage: vec![2, 1],
            dim: 3,
        };
        let bytes = postcard::to_allocvec(&original).unwrap();
        let decoded: BroadcastOpeningTree = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.data, original.data, "data must survive round-trip");
        assert_eq!(
            decoded.openings_per_stage, original.openings_per_stage,
            "openings_per_stage must survive round-trip"
        );
        assert_eq!(decoded.dim, original.dim, "dim must survive round-trip");
    }

    // ------------------------------------------------------------------
    // BroadcastConfig tests
    // ------------------------------------------------------------------

    #[test]
    fn broadcast_config_propagates_training_enabled() {
        use super::BroadcastConfig;

        // training.enabled omitted from JSON → defaults to true.
        let enabled_json = r#"{ "training": {} }"#;
        let enabled_config: cobre_io::Config = serde_json::from_str(enabled_json).unwrap();
        let bcast = BroadcastConfig::from_config(&enabled_config).unwrap();
        assert!(
            bcast.training_enabled,
            "training_enabled should default to true"
        );

        let disabled_json = r#"{ "training": { "enabled": false } }"#;
        let disabled_config: cobre_io::Config = serde_json::from_str(disabled_json).unwrap();
        let bcast = BroadcastConfig::from_config(&disabled_config).unwrap();
        assert!(
            !bcast.training_enabled,
            "training_enabled should be false when config.training.enabled is false"
        );
    }

    /// Postcard serialization round-trip for `BroadcastConfig`.
    #[test]
    fn broadcast_config_roundtrips_via_postcard() {
        use cobre_core::scenario::{SamplingScheme, ScenarioSource};

        use super::BroadcastConfig;

        let json = r#"{
            "training": {
                "forward_passes": 4,
                "stopping_rules": [
                    { "type": "iteration_limit", "limit": 10 }
                ]
            }
        }"#;
        let config: cobre_io::Config = serde_json::from_str(json).unwrap();
        let original = BroadcastConfig::from_config(&config).unwrap();

        let bytes = postcard::to_allocvec(&original)
            .expect("postcard serialization of BroadcastConfig must succeed");
        let decoded: BroadcastConfig = postcard::from_bytes(&bytes)
            .expect("postcard deserialization of BroadcastConfig must succeed");

        assert_eq!(decoded.seed, original.seed);
        assert_eq!(decoded.seed, 42u64);
        assert_eq!(decoded.forward_passes, original.forward_passes);
        assert_eq!(decoded.forward_passes, 4u32);
        assert_eq!(decoded.n_scenarios, original.n_scenarios);
        assert_eq!(decoded.n_scenarios, 0u32);
        assert_eq!(decoded.training_source, original.training_source);
        let default_source = ScenarioSource::default();
        assert_eq!(
            decoded.training_source.inflow_scheme,
            default_source.inflow_scheme
        );
        assert_eq!(
            decoded.training_source.inflow_scheme,
            SamplingScheme::InSample
        );
        assert_eq!(decoded.simulation_source, original.simulation_source);
        assert_eq!(
            decoded.simulation_source.inflow_scheme,
            SamplingScheme::InSample
        );
    }

    /// Guardrail: catches a future serializer switching to named fields and
    /// re-emitting stale field names into the postcard wire bytes.
    #[test]
    fn broadcast_config_wire_excludes_deleted_fields() {
        use super::BroadcastConfig;

        let json = r#"{
            "training": {
                "forward_passes": 4,
                "stopping_rules": [
                    { "type": "iteration_limit", "limit": 10 }
                ]
            }
        }"#;
        let config: cobre_io::Config = serde_json::from_str(json).unwrap();
        let bcast = BroadcastConfig::from_config(&config).unwrap();
        let bytes = postcard::to_allocvec(&bcast).expect("postcard serialization must succeed");

        let as_string = String::from_utf8_lossy(&bytes);
        for stale in [
            "warm_start_basis_mode",
            "canonical_state_strategy",
            "basis_padding",
        ] {
            assert!(
                !as_string.contains(stale),
                "BroadcastConfig postcard bytes must not contain '{stale}'"
            );
        }
    }

    #[test]
    fn broadcast_optional_opening_tree_local_round_trips() {
        use cobre_stochastic::context::OpeningTree;

        let comm = cobre_comm::LocalBackend;

        // Some(None) = no user-supplied tree.
        let no_tree: Option<BroadcastOpeningTree> = None;
        let result = broadcast_value(Some(no_tree), &comm).unwrap();
        assert!(result.is_none(), "Some(None) must round-trip to None");

        // Some(Some(..)) = a user-supplied tree.
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let ops = vec![2];
        let dim = 2usize;
        let source_tree = OpeningTree::from_parts(data.clone(), ops.clone(), dim);
        let bcast = Some(BroadcastOpeningTree {
            data: source_tree.data().to_vec(),
            openings_per_stage: source_tree.openings_per_stage_slice().to_vec(),
            dim: source_tree.dim(),
        });
        let result = broadcast_value(Some(bcast), &comm).unwrap();
        let bt = result.unwrap();
        let reconstructed = OpeningTree::from_parts(bt.data, bt.openings_per_stage, bt.dim);
        assert_eq!(
            reconstructed.data(),
            source_tree.data(),
            "reconstructed tree data must match source"
        );
        assert_eq!(
            reconstructed.dim(),
            source_tree.dim(),
            "reconstructed tree dim must match source"
        );
        assert_eq!(
            reconstructed.openings_per_stage_slice(),
            source_tree.openings_per_stage_slice(),
            "reconstructed tree openings_per_stage must match source"
        );
    }
}
