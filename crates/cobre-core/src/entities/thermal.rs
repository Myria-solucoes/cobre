//! Thermal plant entity — generation with MW bounds and cost.

use crate::EntityId;
use chrono::NaiveDate;

/// Anticipated dispatch configuration for thermal plants requiring advance commitment.
///
/// `lead_stages` is `u32` so negative JSON literals are rejected at deserialise
/// time (zero is rejected by the semantic validator). `deny_unknown_fields`
/// mirrors the IO-layer `RawAnticipatedConfig` so deserialisation paths that
/// bypass the IO raw parser still reject unknown keys; postcard is positional and
/// ignores the attribute on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct AnticipatedConfig {
    /// Number of stages of dispatch anticipation. Must be ≥ 1.
    pub lead_stages: u32,
}

/// Thermal power plant with a scalar marginal cost.
///
/// See Input System Entities SS1.9.5.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Thermal {
    /// Unique thermal plant identifier.
    pub id: EntityId,
    /// Human-readable plant name.
    pub name: String,
    /// Date the entity enters service (ISO 8601).
    pub operational_start_date: NaiveDate,
    /// Bus to which this plant's generation is injected.
    pub bus_id: EntityId,
    /// Stage index when the plant enters service. None = always exists.
    pub entry_stage_id: Option<i32>,
    /// Stage index when the plant is decommissioned. None = never decommissioned.
    pub exit_stage_id: Option<i32>,
    /// Marginal cost of generation \[$/`MWh`\].
    pub cost_per_mwh: f64,
    /// Minimum electrical generation (minimum stable load) \[MW\].
    pub min_generation_mw: f64,
    /// Maximum electrical generation (installed capacity) \[MW\].
    pub max_generation_mw: f64,
    /// Anticipated dispatch configuration. None = no anticipation lag.
    pub anticipated_config: Option<AnticipatedConfig>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_construction() {
        let thermal = Thermal {
            id: EntityId::from(1),
            name: "Angra 1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(10),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 657.0,
            anticipated_config: None,
        };

        assert_eq!(thermal.id, EntityId::from(1));
        assert_eq!(thermal.name, "Angra 1");
        assert_eq!(thermal.bus_id, EntityId::from(10));
        assert_eq!(thermal.entry_stage_id, None);
        assert_eq!(thermal.exit_stage_id, None);
        assert_eq!(thermal.cost_per_mwh, 50.0);
        assert_eq!(thermal.min_generation_mw, 0.0);
        assert_eq!(thermal.max_generation_mw, 657.0);
        assert_eq!(thermal.anticipated_config, None);
    }

    #[test]
    fn test_thermal_with_anticipated() {
        let thermal = Thermal {
            id: EntityId::from(2),
            name: "Pecém I".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(20),
            entry_stage_id: Some(1),
            exit_stage_id: Some(120),
            cost_per_mwh: 120.0,
            min_generation_mw: 100.0,
            max_generation_mw: 360.0,
            anticipated_config: Some(AnticipatedConfig { lead_stages: 2 }),
        };

        assert_eq!(
            thermal.anticipated_config,
            Some(AnticipatedConfig { lead_stages: 2 })
        );
        assert_eq!(thermal.entry_stage_id, Some(1));
        assert_eq!(thermal.exit_stage_id, Some(120));
    }

    #[test]
    fn test_anticipated_config_lead_stages_as_usize_5() {
        let config = AnticipatedConfig { lead_stages: 5 };
        assert_eq!(config.lead_stages as usize, 5_usize);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_thermal_serde_roundtrip() {
        let thermal = Thermal {
            id: EntityId::from(2),
            name: "Pecém I".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(20),
            entry_stage_id: Some(1),
            exit_stage_id: Some(120),
            cost_per_mwh: 80.0,
            min_generation_mw: 100.0,
            max_generation_mw: 360.0,
            anticipated_config: Some(AnticipatedConfig { lead_stages: 2 }),
        };
        let json = serde_json::to_string(&thermal).unwrap();
        let deserialized: Thermal = serde_json::from_str(&json).unwrap();
        assert_eq!(thermal, deserialized);
        assert!(json.contains("\"anticipated_config\":{\"lead_stages\":2}"));
    }
}
