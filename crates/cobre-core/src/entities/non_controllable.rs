//! Non-controllable generation source entity — intermittent wind/solar.
//!
//! A [`NonControllableSource`] represents intermittent generation (wind, solar,
//! run-of-river) that cannot be dispatched. Generation is injected into the
//! network at stochastic availability levels, with optional curtailment.
//! LP variables, forward/backward patching, and simulation extraction are
//! fully integrated.

use crate::EntityId;

/// Intermittent generation source that cannot be dispatched.
///
/// A `NonControllableSource` injects generation into the network up to
/// `max_generation_mw × α × factor`, where `α` is the per-(stage, scenario)
/// availability ratio (drawn from the stochastic model) and `factor` is the
/// per-(stage, block) shape factor.
///
/// When `allow_curtailment == true` (the default) the LP may dispatch
/// anywhere in `[0, max × α × factor]` at the cost of `curtailment_cost`
/// per curtailed `MWh`. When `allow_curtailment == false` the LP pins
/// dispatch to the realized availability (`col_lower = col_upper = max × α
/// × factor`) — the **must-run** regime for non-simulated aggregate
/// generation pre-netted from load before the dispatch LP runs.
///
/// Source: `system/non_controllable.json`. See Input System Entities SS1.9.8.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NonControllableSource {
    /// Unique source identifier.
    pub id: EntityId,
    /// Human-readable source name.
    pub name: String,
    /// Bus to which this source's generation is injected.
    pub bus_id: EntityId,
    /// Stage index when the source enters service. None = always exists.
    pub entry_stage_id: Option<i32>,
    /// Stage index when the source is decommissioned. None = never decommissioned.
    pub exit_stage_id: Option<i32>,
    /// Maximum generation (installed capacity) \[MW\].
    pub max_generation_mw: f64,
    /// Whether the LP is allowed to curtail this source.
    ///
    /// `true` (default) — fully curtailable: `col_lower = 0`,
    /// `col_upper = max × α × factor`; the LP pays
    /// `curtailment_cost × (col_upper − dispatch) × hours` per block.
    ///
    /// `false` — must-run: `col_lower = col_upper = max × α × factor`,
    /// dispatch is pinned to the realized availability for every scenario.
    /// Models non-simulated aggregate generation (small hydro, biomass,
    /// wind, solar, distributed generation) pre-netted from load before the
    /// dispatch LP runs.
    pub allow_curtailment: bool,
    /// Resolved cost per `MWh` of curtailed generation \[$/`MWh`\].
    ///
    /// This is a resolved field — defaults are applied during loading so this
    /// value is always ready for LP construction without further lookup.
    /// Unused when `allow_curtailment == false` (no curtailment slack).
    pub curtailment_cost: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_controllable_construction() {
        let source = NonControllableSource {
            id: EntityId::from(1),
            name: "Eólica Caetité".to_string(),
            bus_id: EntityId::from(7),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 300.0,
            allow_curtailment: true,
            curtailment_cost: 0.01,
        };

        assert_eq!(source.id, EntityId::from(1));
        assert_eq!(source.name, "Eólica Caetité");
        assert_eq!(source.bus_id, EntityId::from(7));
        assert_eq!(source.entry_stage_id, None);
        assert_eq!(source.exit_stage_id, None);
        assert!(source.allow_curtailment);
        assert!((source.max_generation_mw - 300.0).abs() < f64::EPSILON);
        assert!((source.curtailment_cost - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_non_controllable_curtailment_cost() {
        let source = NonControllableSource {
            id: EntityId::from(2),
            name: "Solar Pirapora".to_string(),
            bus_id: EntityId::from(3),
            entry_stage_id: Some(12),
            exit_stage_id: None,
            max_generation_mw: 400.0,
            allow_curtailment: true,
            curtailment_cost: 5.0,
        };

        assert!((source.curtailment_cost - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_non_controllable_must_run() {
        // allow_curtailment == false models must-run (non-curtailable) aggregates.
        let source = NonControllableSource {
            id: EntityId::from(3),
            name: "PCH Aggregate NE".to_string(),
            bus_id: EntityId::from(0),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 120.0,
            allow_curtailment: false,
            curtailment_cost: 0.0344,
        };

        assert!(!source.allow_curtailment);
    }
}
