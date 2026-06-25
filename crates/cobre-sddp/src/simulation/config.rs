//! Simulation configuration type for the SDDP policy evaluation phase.
//!
//! [`SimulationConfig`] bundles all parameters that control the simulation
//! pipeline: number of scenarios to evaluate and the bounded channel capacity
//! that throttles the background I/O thread.

/// Parameters controlling the SDDP simulation pipeline.
///
/// No `Default`: every field must be set explicitly to prevent silent
/// misconfiguration.
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::simulation::SimulationConfig;
///
/// let config = SimulationConfig {
///     n_scenarios: 500,
///     io_channel_capacity: 32,
/// };
/// assert_eq!(config.n_scenarios, 500);
/// assert_eq!(config.io_channel_capacity, 32);
/// ```
#[derive(Debug)]
pub struct SimulationConfig {
    /// Total number of scenarios to simulate across all MPI ranks. Must be at
    /// least 1. (Distribution strategy: simulation-architecture.md SS3.1.)
    pub n_scenarios: u32,

    /// Bounded capacity of the
    /// [`SimulationScenarioResult`](crate::simulation::SimulationScenarioResult)
    /// channel to the background I/O thread; a full channel blocks simulation
    /// threads, providing backpressure.
    pub io_channel_capacity: usize,
}

#[cfg(test)]
mod tests {
    use super::SimulationConfig;

    #[test]
    fn simulation_config_construction() {
        let config = SimulationConfig {
            n_scenarios: 2000,
            io_channel_capacity: 64,
        };
        assert_eq!(config.n_scenarios, 2000);
        assert_eq!(config.io_channel_capacity, 64);
    }

    #[test]
    fn simulation_config_arbitrary_values() {
        let config = SimulationConfig {
            n_scenarios: 1,
            io_channel_capacity: 1,
        };
        assert_eq!(config.n_scenarios, 1);
        assert_eq!(config.io_channel_capacity, 1);
    }

    #[test]
    fn simulation_config_debug_non_empty() {
        let config = SimulationConfig {
            n_scenarios: 100,
            io_channel_capacity: 16,
        };
        let debug = format!("{config:?}");
        assert!(!debug.is_empty());
        assert!(
            debug.contains("n_scenarios"),
            "debug must contain field name: {debug}"
        );
        assert!(
            debug.contains("io_channel_capacity"),
            "debug must contain field name: {debug}"
        );
    }
}
