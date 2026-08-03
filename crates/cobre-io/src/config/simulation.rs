//! Post-training simulation configuration types for `config.json → simulation`.

use serde::{Deserialize, Serialize};

use super::scenario_source::RawScenarioSourceConfig;
use super::training::PhaseSolverProfileConfig;

/// Default scenario count when neither `simulation.selection` nor the
/// `simulation.num_scenarios` alias is present. Sole owner of the value; the
/// `Default` impl and the count resolver both read it.
pub(crate) const DEFAULT_NUM_SCENARIOS: u32 = 2000;

/// Post-training simulation settings (`config.json → simulation`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SimulationConfig {
    /// Enable post-training simulation.
    pub enabled: bool,

    /// Number of simulation scenarios. Deprecated alias for the `sampled`
    /// selection count; declaring it alongside `selection` is rejected at load.
    /// Absent resolves to the default sampled count.
    pub num_scenarios: Option<u32>,

    /// Bounded channel capacity between simulation threads and the I/O writer thread.
    pub io_channel_capacity: u32,

    /// Scenario source configuration for the post-training simulation forward pass.
    /// When absent, falls back to the training scenario source.
    #[serde(default)]
    pub scenario_source: Option<RawScenarioSourceConfig>,

    /// Simulation solver profile. Absent leaves the phase's built-in
    /// tuned profile.
    #[serde(default)]
    pub solver: Option<PhaseSolverProfileConfig>,

    /// Phase-level scenario selection. Absent resolves the count from the
    /// `num_scenarios` alias (the default sampled behaviour).
    #[serde(default)]
    pub selection: Option<SimulationSelection>,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            // None, not Some(DEFAULT_NUM_SCENARIOS): container-level serde default
            // fills an omitted `num_scenarios` from here, and a Some default would
            // make every present `selection` collide with the defaulted alias.
            // The default count is applied by the resolver, not this field.
            num_scenarios: None,
            io_channel_capacity: 64,
            scenario_source: None,
            solver: None,
            selection: None,
        }
    }
}

/// Post-training scenario selection and its method-specific parameters
/// (`config.json → simulation.selection`).
///
/// Internally tagged on `method`; the tag is the semantic selection word, never
/// a mechanism name. `sampled` draws `num_scenarios` trajectories; `enumerated`
/// walks the scenario set exhaustively. Each variant carries only its own
/// parameters, so pairing a count with `enumerated` is a parse error under
/// `deny_unknown_fields` rather than a runtime-gated combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum SimulationSelection {
    /// Sampled scenarios: draw `num_scenarios` trajectories.
    Sampled {
        /// Number of simulation trajectories to draw.
        num_scenarios: u32,
    },
    /// Exhaustive enumeration of the scenario set.
    // A braced variant, not a unit one: serde enforces `deny_unknown_fields`
    // only for braced variants of an internally tagged enum, and this variant
    // must reject a stray `num_scenarios`.
    Enumerated {},
}

/// Effective simulation scenario-count resolution
/// ([`Config::resolve_num_scenarios`](super::Config::resolve_num_scenarios)):
/// either a concrete sampled count or a signal that the count is derived from
/// the policy graph downstream, since config load holds no graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumScenariosResolution {
    /// `num_scenarios` sampled simulation trajectories.
    Sampled(u32),
    /// Exhaustive enumeration. `declared` is the optional `num_scenarios`
    /// alias, when declared alongside `enumerated` as a census cross-check
    /// value (never a conflict, unlike the `sampled` alias-vs-selection
    /// pairing) — resolved against the graph-derived count downstream.
    Enumerated {
        /// The declared census cross-check count, if any.
        declared: Option<u32>,
    },
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::{SimulationConfig, SimulationSelection};
    use crate::config::training::{PriceStrategy, ScaleStrategy};

    /// A `sampled` phase selection round-trips into the `Sampled` variant and
    /// leaves the deprecated `num_scenarios` alias absent.
    #[test]
    fn sampled_selection_round_trips() {
        let json = r#"{"enabled": true, "selection": {"method": "sampled", "num_scenarios": 500}}"#;
        let cfg: SimulationConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            cfg.selection,
            Some(SimulationSelection::Sampled { num_scenarios: 500 })
        );
        assert_eq!(cfg.num_scenarios, None);
    }

    /// A count under `enumerated` is unrepresentable — `deny_unknown_fields` on
    /// the braced variant rejects it at parse time.
    #[test]
    fn enumerated_selection_with_count_is_deserialize_error() {
        let json =
            r#"{"enabled": true, "selection": {"method": "enumerated", "num_scenarios": 500}}"#;
        let result = serde_json::from_str::<SimulationConfig>(json);
        assert!(
            result.is_err(),
            "a count under enumerated must be rejected as unrepresentable"
        );
    }

    /// The `num_scenarios` alias defaults to absent; the resolver, not this
    /// field, supplies the default count.
    #[test]
    fn num_scenarios_alias_defaults_to_none() {
        assert_eq!(SimulationConfig::default().num_scenarios, None);
    }

    #[test]
    fn simulation_solver_profile_block_round_trips() {
        let json = r#"{
            "enabled": true,
            "num_scenarios": 500,
            "solver": {
                "scale": "off",
                "price": "row"
            }
        }"#;
        let cfg: SimulationConfig = serde_json::from_str(json).unwrap();
        let solver = cfg.solver.as_ref().expect("solver present");
        assert_eq!(solver.scale, Some(ScaleStrategy::Off));
        assert_eq!(solver.price, Some(PriceStrategy::Row));
    }

    #[test]
    fn simulation_solver_profile_absent_is_none() {
        assert!(SimulationConfig::default().solver.is_none());
    }

    #[test]
    fn simulation_solver_profile_steepest_edge_fallback_threshold_round_trips() {
        let json = r#"{
            "enabled": true,
            "num_scenarios": 500,
            "solver": {
                "steepest_edge_devex_fallback_threshold": 12.5
            }
        }"#;
        let cfg: SimulationConfig = serde_json::from_str(json).unwrap();
        let solver = cfg.solver.as_ref().expect("solver present");
        assert_eq!(solver.steepest_edge_devex_fallback_threshold, Some(12.5));
    }
}
