//! Post-training simulation configuration types for `config.json → simulation`.

use serde::{Deserialize, Serialize};

use super::scenario_source::RawScenarioSourceConfig;

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
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_scenarios: 2000,
            io_channel_capacity: 64,
            scenario_source: None,
        }
    }
}
