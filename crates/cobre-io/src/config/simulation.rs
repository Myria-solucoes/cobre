//! Post-training simulation configuration types for `config.json → simulation`.

use serde::{Deserialize, Serialize};

use super::scenario_source::RawScenarioSourceConfig;
use super::training::PhaseSolverProfileConfig;

/// Post-training simulation settings (`config.json → simulation`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SimulationConfig {
    /// Enable post-training simulation.
    pub enabled: bool,

    /// Number of simulation scenarios.
    pub num_scenarios: u32,

    /// Bounded channel capacity between simulation threads and the I/O writer thread.
    pub io_channel_capacity: u32,

    /// Scenario source configuration for the post-training simulation forward pass.
    /// When absent, falls back to the training scenario source.
    #[serde(default)]
    pub scenario_source: Option<RawScenarioSourceConfig>,

    /// Simulation solver profile. Absent leaves the backend defaults.
    #[serde(default)]
    pub solver: Option<PhaseSolverProfileConfig>,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_scenarios: 2000,
            io_channel_capacity: 64,
            scenario_source: None,
            solver: None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::SimulationConfig;
    use crate::config::training::{PriceStrategy, ScaleStrategy};

    #[test]
    fn simulation_solver_profile_block_round_trips() {
        let json = r#"{
            "enabled": true,
            "num_scenarios": 500,
            "solver": {
                "preset": "sim_v1",
                "scale": "off",
                "price": "row"
            }
        }"#;
        let cfg: SimulationConfig = serde_json::from_str(json).unwrap();
        let solver = cfg.solver.as_ref().expect("solver present");
        assert_eq!(solver.preset.as_deref(), Some("sim_v1"));
        assert_eq!(solver.scale, Some(ScaleStrategy::Off));
        assert_eq!(solver.price, Some(PriceStrategy::Row));
    }

    #[test]
    fn simulation_solver_profile_absent_is_none() {
        assert!(SimulationConfig::default().solver.is_none());
    }

    #[test]
    fn simulation_solver_profile_dse_devex_fallback_threshold_round_trips() {
        let json = r#"{
            "enabled": true,
            "num_scenarios": 500,
            "solver": {
                "dse_devex_fallback_threshold": 12.5
            }
        }"#;
        let cfg: SimulationConfig = serde_json::from_str(json).unwrap();
        let solver = cfg.solver.as_ref().expect("solver present");
        assert_eq!(solver.dse_devex_fallback_threshold, Some(12.5));
    }
}
