//! Postcard serialization helpers for MPI broadcast of [`System`] and
//! [`ScalarParameter`] collections.
//!
//! Cobre uses `postcard` (not `bincode`) for MPI serialization (see CLAUDE.md hard rules).
//! These helpers serialize payloads to compact byte buffers for broadcast
//! and deserialize them on worker ranks.
//!
//! ## Why broadcast mirror types
//!
//! [`ParameterKind`] and [`ComputedParameter`] use serde internally-tagged
//! enums (`#[serde(tag = "...")]`) to drive the user-facing JSON schema. Postcard
//! does not support that representation. To keep MPI broadcast working without
//! per-rank disk reads, the public `serialize_parameters` /
//! `deserialize_parameters` helpers convert through tag-free mirror types
//! ([`BroadcastScalarParameter`], [`BroadcastParameterKind`],
//! [`BroadcastComputedParameter`]) on the wire and reconstruct the in-memory
//! shape on the receiving end. This mirrors the `BroadcastConfig` pattern in
//! `cobre-cli` used for [`crate::Config`].
//!
//! # Usage
//!
//! On rank 0, load the case and serialize:
//!
//! ```rust,ignore
//! let system = cobre_io::load_case(&path)?;
//! let bytes = cobre_io::serialize_system(&system)?;
//! // broadcast bytes via MPI ...
//! ```
//!
//! On worker ranks, deserialize after receiving:
//!
//! ```rust,ignore
//! // ... receive bytes via MPI
//! let system = cobre_io::deserialize_system(&bytes)?;
//! // system.bus(id) works immediately — indices are rebuilt
//! ```

use cobre_core::{ComputedParameter, EntityId, ParameterKind, ScalarParameter, System};
use serde::{Deserialize, Serialize};

use crate::LoadError;

// ── Broadcast mirror types (tag-free, postcard-compatible) ──────────────────

/// Postcard-safe mirror of [`ScalarParameter`], holding a [`BroadcastParameterKind`]
/// in place of [`ParameterKind`]. Convert with `From` in both directions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BroadcastScalarParameter {
    /// Unique parameter identifier.
    pub id: EntityId,
    /// Short name used in reports and log output.
    pub name: String,
    /// Kind in the broadcast-safe representation.
    pub kind: BroadcastParameterKind,
}

/// Postcard-safe mirror of [`ParameterKind`]. Uses externally-tagged enum
/// encoding (the serde default), which postcard supports natively.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BroadcastParameterKind {
    /// Single scalar value applied to every stage.
    Constant(f64),
    /// Dense `Vec<f64>` indexed by stage (0-based).
    PerStage(Vec<f64>),
    /// Sorted, deduplicated `(season_id, value)` pairs.
    Seasonal(Vec<(i32, f64)>),
    /// Computed-parameter specification.
    Computed(BroadcastComputedParameter),
}

/// Postcard-safe mirror of [`ComputedParameter`]. Externally-tagged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BroadcastComputedParameter {
    /// Equivalent productivity coefficient (`ρ_eq`).
    EquivalentProductivity(EntityId),
    /// Accumulated productivity coefficient (`ρ_acum`).
    AccumulatedProductivity(EntityId),
    /// Reference reservoir volume (`V_ref`).
    ReferenceVolume(EntityId),
    /// Reference turbine flow (`Q_ref`).
    ReferenceTurbine(EntityId),
    /// Minimum operational storage (`V_min`).
    MinStorage(EntityId),
    /// Maximum operational storage (`V_max`).
    MaxStorage(EntityId),
    /// Specific productivity (`ρ_esp`).
    SpecificProductivity(EntityId),
}

impl From<&ScalarParameter> for BroadcastScalarParameter {
    fn from(p: &ScalarParameter) -> Self {
        Self {
            id: p.id,
            name: p.name.clone(),
            kind: BroadcastParameterKind::from(&p.kind),
        }
    }
}

impl From<BroadcastScalarParameter> for ScalarParameter {
    fn from(b: BroadcastScalarParameter) -> Self {
        Self {
            id: b.id,
            name: b.name,
            kind: ParameterKind::from(b.kind),
        }
    }
}

impl From<&ParameterKind> for BroadcastParameterKind {
    fn from(k: &ParameterKind) -> Self {
        match k {
            ParameterKind::Constant { value } => Self::Constant(*value),
            ParameterKind::PerStage { values } => Self::PerStage(values.clone()),
            ParameterKind::Seasonal { values } => Self::Seasonal(values.clone()),
            ParameterKind::Computed { computed_spec } => {
                Self::Computed(BroadcastComputedParameter::from(*computed_spec))
            }
        }
    }
}

impl From<BroadcastParameterKind> for ParameterKind {
    fn from(b: BroadcastParameterKind) -> Self {
        match b {
            BroadcastParameterKind::Constant(value) => Self::Constant { value },
            BroadcastParameterKind::PerStage(values) => Self::PerStage { values },
            BroadcastParameterKind::Seasonal(values) => Self::Seasonal { values },
            BroadcastParameterKind::Computed(c) => Self::Computed {
                computed_spec: ComputedParameter::from(c),
            },
        }
    }
}

impl From<ComputedParameter> for BroadcastComputedParameter {
    fn from(c: ComputedParameter) -> Self {
        match c {
            ComputedParameter::EquivalentProductivity { hydro_id } => {
                Self::EquivalentProductivity(hydro_id)
            }
            ComputedParameter::AccumulatedProductivity { hydro_id } => {
                Self::AccumulatedProductivity(hydro_id)
            }
            ComputedParameter::ReferenceVolume { hydro_id } => Self::ReferenceVolume(hydro_id),
            ComputedParameter::ReferenceTurbine { hydro_id } => Self::ReferenceTurbine(hydro_id),
            ComputedParameter::MinStorage { hydro_id } => Self::MinStorage(hydro_id),
            ComputedParameter::MaxStorage { hydro_id } => Self::MaxStorage(hydro_id),
            ComputedParameter::SpecificProductivity { hydro_id } => {
                Self::SpecificProductivity(hydro_id)
            }
        }
    }
}

impl From<BroadcastComputedParameter> for ComputedParameter {
    fn from(b: BroadcastComputedParameter) -> Self {
        match b {
            BroadcastComputedParameter::EquivalentProductivity(hydro_id) => {
                Self::EquivalentProductivity { hydro_id }
            }
            BroadcastComputedParameter::AccumulatedProductivity(hydro_id) => {
                Self::AccumulatedProductivity { hydro_id }
            }
            BroadcastComputedParameter::ReferenceVolume(hydro_id) => {
                Self::ReferenceVolume { hydro_id }
            }
            BroadcastComputedParameter::ReferenceTurbine(hydro_id) => {
                Self::ReferenceTurbine { hydro_id }
            }
            BroadcastComputedParameter::MinStorage(hydro_id) => Self::MinStorage { hydro_id },
            BroadcastComputedParameter::MaxStorage(hydro_id) => Self::MaxStorage { hydro_id },
            BroadcastComputedParameter::SpecificProductivity(hydro_id) => {
                Self::SpecificProductivity { hydro_id }
            }
        }
    }
}

/// Serialize a [`System`] to a postcard byte buffer for MPI broadcast.
///
/// The returned `Vec<u8>` is suitable for broadcasting over MPI. The recipient
/// must call [`deserialize_system`] to reconstruct the [`System`] with working
/// O(1) lookup indices.
///
/// # Errors
///
/// Returns [`LoadError::ParseError`] with path `"<broadcast>"` if postcard
/// encounters an unsupported type during serialization. This should not occur
/// in practice given the types used in [`System`].
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
/// use cobre_io::serialize_system;
///
/// let bus = Bus {
///     id: EntityId(1),
///     name: "Main Bus".to_string(),
///     operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
///     deficit_segments: vec![DeficitSegment { depth_mw: None, cost_per_mwh: 500.0 }],
///     excess_cost: 0.0,
/// };
/// let system = SystemBuilder::new().buses(vec![bus]).build().unwrap();
/// let bytes = serialize_system(&system).unwrap();
/// assert!(!bytes.is_empty());
/// ```
pub fn serialize_system(system: &System) -> Result<Vec<u8>, LoadError> {
    postcard::to_allocvec(system)
        .map_err(|e| LoadError::parse("<broadcast>", format!("postcard serialization: {e}")))
}

/// Deserialize a [`System`] from a postcard byte buffer received via MPI broadcast.
///
/// `System`'s `Deserialize` impl rebuilds lookup indices unconditionally, so
/// O(1) entity lookups (e.g., `system.bus(id)`) work immediately on the
/// returned value.
///
/// # Errors
///
/// Returns [`LoadError::ParseError`] with path `"<broadcast>"` if the byte slice
/// is corrupted, truncated, or not a valid postcard encoding of [`System`].
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
/// use cobre_io::{deserialize_system, serialize_system};
///
/// let bus = Bus {
///     id: EntityId(1),
///     name: "Main Bus".to_string(),
///     operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
///     deficit_segments: vec![DeficitSegment { depth_mw: None, cost_per_mwh: 500.0 }],
///     excess_cost: 0.0,
/// };
/// let system = SystemBuilder::new().buses(vec![bus]).build().unwrap();
/// let bytes = serialize_system(&system).unwrap();
/// let restored = deserialize_system(&bytes).unwrap();
/// assert_eq!(restored.n_buses(), 1);
/// assert!(restored.bus(EntityId(1)).is_some());
/// ```
pub fn deserialize_system(bytes: &[u8]) -> Result<System, LoadError> {
    postcard::from_bytes(bytes)
        .map_err(|e| LoadError::parse("<broadcast>", format!("postcard deserialization: {e}")))
}

/// Serialize a list of [`ScalarParameter`] to a postcard byte buffer for MPI
/// broadcast.
///
/// The returned `Vec<u8>` encodes the entire slice as a single postcard payload
/// including a varint length prefix, so the recipient can deserialize without a
/// separate length-broadcast step. The caller is responsible for the MPI
/// broadcast call itself.
///
/// # Errors
///
/// Returns [`LoadError::ParseError`] with path `"<broadcast>"` if postcard
/// serialization fails.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, ParameterKind, ScalarParameter};
/// use cobre_io::serialize_parameters;
///
/// let param = ScalarParameter {
///     id: EntityId(1),
///     name: "rho_eq_h1".to_string(),
///     kind: ParameterKind::Constant { value: 3.6 },
/// };
/// let bytes = serialize_parameters(&[param.clone()]).unwrap();
/// assert!(!bytes.is_empty());
///
/// let restored = cobre_io::deserialize_parameters(&bytes).unwrap();
/// assert_eq!(restored, vec![param]);
/// ```
pub fn serialize_parameters(parameters: &[ScalarParameter]) -> Result<Vec<u8>, LoadError> {
    let mirror: Vec<BroadcastScalarParameter> = parameters
        .iter()
        .map(BroadcastScalarParameter::from)
        .collect();
    postcard::to_allocvec(&mirror)
        .map_err(|e| LoadError::parse("<broadcast>", format!("postcard serialization: {e}")))
}

/// Deserialize a `Vec<ScalarParameter>` from a postcard byte buffer received
/// via MPI broadcast.
///
/// The byte buffer must have been produced by [`serialize_parameters`]. An empty
/// slice or a corrupted buffer returns an error; this function never silently
/// discards data.
///
/// # Errors
///
/// Returns [`LoadError::ParseError`] with path `"<broadcast>"` if the byte
/// slice is corrupted, truncated, or not a valid postcard encoding of
/// `Vec<ScalarParameter>`.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, ParameterKind, ScalarParameter};
/// use cobre_io::{deserialize_parameters, serialize_parameters};
///
/// let param = ScalarParameter {
///     id: EntityId(1),
///     name: "rho_eq_h1".to_string(),
///     kind: ParameterKind::Constant { value: 3.6 },
/// };
/// let bytes = serialize_parameters(&[param.clone()]).unwrap();
/// let restored = deserialize_parameters(&bytes).unwrap();
/// assert_eq!(restored, vec![param]);
/// ```
pub fn deserialize_parameters(bytes: &[u8]) -> Result<Vec<ScalarParameter>, LoadError> {
    let mirror: Vec<BroadcastScalarParameter> = postcard::from_bytes(bytes)
        .map_err(|e| LoadError::parse("<broadcast>", format!("postcard deserialization: {e}")))?;
    Ok(mirror.into_iter().map(ScalarParameter::from).collect())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use cobre_core::{
        AnticipatedCommitmentHistory, Bus, ComputedParameter, DeficitSegment, EntityId, Hydro,
        HydroGenerationModel, HydroPenalties, InitialConditions, ParameterKind, ScalarParameter,
        SystemBuilder, Thermal, entities::AnticipatedConfig,
    };

    fn minimal_bus(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("Bus {id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        }
    }

    fn minimal_thermal(id: i32, bus_id: i32) -> Thermal {
        Thermal {
            id: EntityId(id),
            name: format!("Thermal {id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: None,
        }
    }

    fn zero_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    fn minimal_hydro(id: i32, bus_id: i32) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("Hydro {id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(bus_id),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 1000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 200.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 200.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_hydro_penalties(),
        };
        hydro.declare_mirror_unit_group();
        hydro
    }

    #[test]
    fn test_round_trip_minimal_system() {
        let bus = minimal_bus(1);
        let system = SystemBuilder::new().buses(vec![bus]).build().unwrap();

        let bytes = serialize_system(&system).unwrap();
        assert!(!bytes.is_empty());

        let restored = deserialize_system(&bytes).unwrap();

        assert_eq!(restored.n_buses(), system.n_buses());
        assert!(restored.bus(EntityId(1)).is_some());
    }

    #[test]
    fn test_round_trip_populated_system() {
        let buses = vec![minimal_bus(1), minimal_bus(2)];
        let thermals = vec![minimal_thermal(1, 1), minimal_thermal(2, 2)];
        let hydros = vec![minimal_hydro(1, 1)];

        let system = SystemBuilder::new()
            .buses(buses)
            .thermals(thermals)
            .hydros(hydros)
            .build()
            .unwrap();

        let bytes = serialize_system(&system).unwrap();
        let restored = deserialize_system(&bytes).unwrap();

        assert_eq!(restored.n_buses(), system.n_buses());
        assert_eq!(restored.n_thermals(), system.n_thermals());
        assert_eq!(restored.n_hydros(), system.n_hydros());

        assert!(restored.bus(EntityId(1)).is_some());
        assert!(restored.bus(EntityId(2)).is_some());
        assert!(restored.thermal(EntityId(1)).is_some());
        assert!(restored.thermal(EntityId(2)).is_some());
        assert!(restored.hydro(EntityId(1)).is_some());

        assert_eq!(restored, system);
    }

    #[test]
    fn test_round_trip_anticipated_thermal_system() {
        // Guards against a future Thermal field reorder or AnticipatedConfig schema
        // change silently breaking broadcast deserialization of `anticipated_config`.
        let buses = vec![minimal_bus(1), minimal_bus(2)];
        let mut anticipated = minimal_thermal(10, 1);
        anticipated.anticipated_config = Some(AnticipatedConfig::LeadStages(2));
        let regular = minimal_thermal(20, 2);
        let thermals = vec![anticipated, regular];

        let system = SystemBuilder::new()
            .buses(buses)
            .thermals(thermals)
            .build()
            .unwrap();

        let bytes = serialize_system(&system).unwrap();
        let restored = deserialize_system(&bytes).unwrap();

        assert_eq!(restored.n_thermals(), system.n_thermals());
        let Some(restored_anticipated) = restored.thermal(EntityId(10)) else {
            panic!("thermal 10 must round-trip");
        };
        assert_eq!(
            restored_anticipated.anticipated_config,
            Some(AnticipatedConfig::LeadStages(2)),
            "anticipated_config must survive broadcast round-trip"
        );
        let Some(restored_regular) = restored.thermal(EntityId(20)) else {
            panic!("thermal 20 must round-trip");
        };
        assert_eq!(
            restored_regular.anticipated_config, None,
            "non-anticipated thermal must remain None after round-trip"
        );

        assert_eq!(restored, system);
    }

    #[test]
    fn test_deserialize_corrupted_bytes() {
        let result = deserialize_system(&[0u8; 4]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("<broadcast>"));
        assert!(matches!(err, LoadError::ParseError { .. }));
    }

    #[test]
    fn test_deserialize_empty_bytes() {
        let result = deserialize_system(&[]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, LoadError::ParseError { .. }));
        assert!(err.to_string().contains("<broadcast>"));
    }

    #[test]
    fn test_serialized_size_reasonable() {
        let bus = minimal_bus(1);
        let system = SystemBuilder::new().buses(vec![bus]).build().unwrap();
        let bytes = serialize_system(&system).unwrap();
        assert!(bytes.len() < 1024);
    }

    /// Build a `Vec<ScalarParameter>` with one instance of each of the four
    /// `ParameterKind` variants, covering all code paths through the
    /// postcard serialization layer.
    fn four_kinds_fixture() -> Vec<ScalarParameter> {
        vec![
            ScalarParameter {
                id: EntityId(1),
                name: "constant_param".to_string(),
                kind: ParameterKind::Constant { value: 1.5 },
            },
            ScalarParameter {
                id: EntityId(2),
                name: "per_stage_param".to_string(),
                kind: ParameterKind::PerStage {
                    values: vec![1.0, 2.0, 3.0],
                },
            },
            ScalarParameter {
                id: EntityId(3),
                name: "seasonal_param".to_string(),
                kind: ParameterKind::new_seasonal(vec![(2, 1.0), (1, 0.5)]),
            },
            ScalarParameter {
                id: EntityId(4),
                name: "computed_param".to_string(),
                kind: ParameterKind::Computed {
                    computed_spec: ComputedParameter::EquivalentProductivity {
                        hydro_id: EntityId(7),
                    },
                },
            },
        ]
    }

    #[test]
    fn round_trip_all_four_parameter_kinds() {
        let original = four_kinds_fixture();
        let bytes = serialize_parameters(&original).unwrap();
        assert!(!bytes.is_empty());
        let restored = deserialize_parameters(&bytes).unwrap();
        assert_eq!(restored, original);
    }

    #[test]
    fn serialize_parameters_is_deterministic() {
        let params = four_kinds_fixture();
        let bytes_a = serialize_parameters(&params).unwrap();
        let bytes_b = serialize_parameters(&params).unwrap();
        assert_eq!(bytes_a, bytes_b);
    }

    #[test]
    fn deserialize_parameters_rejects_corrupted_bytes() {
        let result = deserialize_parameters(&[0xFF, 0xFE, 0xFD, 0xFC]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, LoadError::ParseError { .. }));
        assert!(err.to_string().contains("<broadcast>"));
    }

    #[test]
    fn deserialize_parameters_rejects_empty_buffer() {
        let result = deserialize_parameters(&[]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, LoadError::ParseError { .. }));
        assert!(err.to_string().contains("<broadcast>"));
    }

    /// `InitialConditions` with `past_anticipated_commitments` survives postcard
    /// round-trip end-to-end, covering the MPI broadcast path.
    #[test]
    fn test_broadcast_initial_conditions_round_trips_past_anticipated_commitments() {
        let original = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![
                AnticipatedCommitmentHistory {
                    thermal_id: EntityId(1),
                    values_mw: vec![120.0, 180.0],
                },
                AnticipatedCommitmentHistory {
                    thermal_id: EntityId(7),
                    values_mw: vec![50.0, 75.0, 100.0, 200.0],
                },
            ],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        let bytes = postcard::to_allocvec(&original).unwrap();
        assert!(!bytes.is_empty());

        let restored: InitialConditions = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(restored, original);
        assert_eq!(restored.past_anticipated_commitments.len(), 2);
        assert_eq!(
            restored.past_anticipated_commitments[0].thermal_id,
            EntityId(1)
        );
        assert_eq!(
            restored.past_anticipated_commitments[0].values_mw,
            vec![120.0, 180.0]
        );
        assert_eq!(
            restored.past_anticipated_commitments[1].thermal_id,
            EntityId(7)
        );
        assert_eq!(
            restored.past_anticipated_commitments[1].values_mw,
            vec![50.0, 75.0, 100.0, 200.0]
        );
    }

    /// Given an `InitialConditions` with an empty `past_anticipated_commitments`
    /// vec, when serialized via postcard and deserialized back, the result has
    /// an empty vec (no panics, no wire-format corruption).
    #[test]
    fn test_broadcast_initial_conditions_empty_past_anticipated_commitments() {
        let original = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        let bytes = postcard::to_allocvec(&original).unwrap();
        assert!(!bytes.is_empty());

        let restored: InitialConditions = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(restored, original);
        assert!(restored.past_anticipated_commitments.is_empty());
    }
}
