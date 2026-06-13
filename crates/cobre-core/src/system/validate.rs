//! Construction-time validation helpers for [`SystemBuilder::build()`].
//!
//! Provides the duplicate-id check, the cross-reference validation pass (every
//! entity field that names another entity by [`EntityId`] is checked against the
//! appropriate lookup index, accumulating all violations), and the hydro
//! filling-config check — plus the `HasId` keying trait and the `build_index` /
//! `build_stage_index` lookup-index builders shared with [`System::rebuild_indices`].
//!
//! The items the sibling `builder` module drives are `pub(crate)`; the per-entity
//! `validate_*_refs` helpers are reached only from `validate_cross_references` and
//! stay private.

use std::collections::HashMap;

use crate::{
    Bus, EnergyContract, EntityId, Hydro, Line, NonControllableSource, PumpingStation, Stage,
    Thermal, ValidationError,
};

pub(crate) trait HasId {
    fn entity_id(&self) -> EntityId;
}

impl HasId for Bus {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}
impl HasId for Line {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}
impl HasId for Hydro {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}
impl HasId for Thermal {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}
impl HasId for PumpingStation {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}
impl HasId for EnergyContract {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}
impl HasId for NonControllableSource {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}

pub(crate) fn build_index<T: HasId>(entities: &[T]) -> HashMap<EntityId, usize> {
    let mut index = HashMap::with_capacity(entities.len());
    for (i, entity) in entities.iter().enumerate() {
        index.insert(entity.entity_id(), i);
    }
    index
}

/// Build a stage lookup index from the canonical-ordered stages vec.
///
/// Keys are `i32` stage IDs (which can be negative for pre-study stages).
pub(crate) fn build_stage_index(stages: &[Stage]) -> HashMap<i32, usize> {
    let mut index = HashMap::with_capacity(stages.len());
    for (i, stage) in stages.iter().enumerate() {
        index.insert(stage.id, i);
    }
    index
}

/// Detect duplicate entity IDs in a **sorted** slice by scanning adjacent pairs.
///
/// Contract: the caller must sort `entities` by `entity_id()` ascending before
/// calling (the builder sorts every collection in `build`). On unsorted input
/// the adjacent-pair scan silently misses non-adjacent duplicates, letting a
/// colliding id slip past validation.
pub(crate) fn check_duplicates<T: HasId>(
    entities: &[T],
    entity_type: &'static str,
    errors: &mut Vec<ValidationError>,
) {
    for window in entities.windows(2) {
        if window[0].entity_id() == window[1].entity_id() {
            errors.push(ValidationError::DuplicateId {
                entity_type,
                id: window[0].entity_id(),
            });
        }
    }
}

/// Detect duplicate stage IDs in a slice already sorted ascending by `Stage::id`.
///
/// Stages key on a domain-level `i32` id (negative for pre-study stages) rather
/// than [`EntityId`], so they cannot reuse the [`HasId`]-generic
/// [`check_duplicates`]; this is the parallel i32-keyed scan. The duplicate id
/// is reported via [`ValidationError::DuplicateId`] with `entity_type: "Stage"`,
/// wrapping the stage id in [`EntityId`] purely as the error's id carrier.
///
/// Contract: `stages` must be sorted ascending by `id` (the caller sorts in
/// [`SystemBuilder::build`](super::SystemBuilder::build) before calling). The
/// adjacent-pair scan misses non-adjacent duplicates on unsorted input, which
/// would let `build_stage_index` silently overwrite a colliding stage and
/// produce a broken index with no validation error.
pub(crate) fn check_duplicate_stages(stages: &[Stage], errors: &mut Vec<ValidationError>) {
    for window in stages.windows(2) {
        if window[0].id == window[1].id {
            errors.push(ValidationError::DuplicateId {
                entity_type: "Stage",
                id: EntityId(window[0].id),
            });
        }
    }
}

/// Bundles entity slices needed for cross-reference validation.
pub(crate) struct CrossRefEntities<'a> {
    pub(crate) lines: &'a [Line],
    pub(crate) hydros: &'a [Hydro],
    pub(crate) thermals: &'a [Thermal],
    pub(crate) pumping_stations: &'a [PumpingStation],
    pub(crate) contracts: &'a [EnergyContract],
    pub(crate) non_controllable_sources: &'a [NonControllableSource],
}

/// Validate all cross-reference fields across entity collections.
///
/// Checks every entity field that references another entity by [`EntityId`]
/// against the appropriate index. All invalid references are appended to
/// `errors` — no short-circuiting on first error.
///
/// This function runs after duplicate checking passes and after indices are
/// built, but before topology construction.
pub(crate) fn validate_cross_references(
    entities: &CrossRefEntities<'_>,
    bus_index: &HashMap<EntityId, usize>,
    hydro_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    validate_line_refs(entities.lines, bus_index, errors);
    validate_hydro_refs(entities.hydros, bus_index, hydro_index, errors);
    validate_thermal_refs(entities.thermals, bus_index, errors);
    validate_pumping_station_refs(entities.pumping_stations, bus_index, hydro_index, errors);
    validate_contract_refs(entities.contracts, bus_index, errors);
    validate_ncs_refs(entities.non_controllable_sources, bus_index, errors);
}

fn validate_line_refs(
    lines: &[Line],
    bus_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for line in lines {
        if !bus_index.contains_key(&line.source_bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "Line",
                source_id: line.id,
                field_name: "source_bus_id",
                referenced_id: line.source_bus_id,
                expected_type: "Bus",
            });
        }
        if !bus_index.contains_key(&line.target_bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "Line",
                source_id: line.id,
                field_name: "target_bus_id",
                referenced_id: line.target_bus_id,
                expected_type: "Bus",
            });
        }
    }
}

fn validate_hydro_refs(
    hydros: &[Hydro],
    bus_index: &HashMap<EntityId, usize>,
    hydro_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for hydro in hydros {
        if !bus_index.contains_key(&hydro.bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "Hydro",
                source_id: hydro.id,
                field_name: "bus_id",
                referenced_id: hydro.bus_id,
                expected_type: "Bus",
            });
        }
        if let Some(downstream_id) = hydro.downstream_id
            && !hydro_index.contains_key(&downstream_id)
        {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "Hydro",
                source_id: hydro.id,
                field_name: "downstream_id",
                referenced_id: downstream_id,
                expected_type: "Hydro",
            });
        }
        if let Some(ref diversion) = hydro.diversion
            && !hydro_index.contains_key(&diversion.downstream_id)
        {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "Hydro",
                source_id: hydro.id,
                field_name: "diversion.downstream_id",
                referenced_id: diversion.downstream_id,
                expected_type: "Hydro",
            });
        }
    }
}

fn validate_thermal_refs(
    thermals: &[Thermal],
    bus_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for thermal in thermals {
        if !bus_index.contains_key(&thermal.bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "Thermal",
                source_id: thermal.id,
                field_name: "bus_id",
                referenced_id: thermal.bus_id,
                expected_type: "Bus",
            });
        }
    }
}

fn validate_pumping_station_refs(
    pumping_stations: &[PumpingStation],
    bus_index: &HashMap<EntityId, usize>,
    hydro_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for ps in pumping_stations {
        if !bus_index.contains_key(&ps.bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "PumpingStation",
                source_id: ps.id,
                field_name: "bus_id",
                referenced_id: ps.bus_id,
                expected_type: "Bus",
            });
        }
        if !hydro_index.contains_key(&ps.source_hydro_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "PumpingStation",
                source_id: ps.id,
                field_name: "source_hydro_id",
                referenced_id: ps.source_hydro_id,
                expected_type: "Hydro",
            });
        }
        if !hydro_index.contains_key(&ps.destination_hydro_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "PumpingStation",
                source_id: ps.id,
                field_name: "destination_hydro_id",
                referenced_id: ps.destination_hydro_id,
                expected_type: "Hydro",
            });
        }
    }
}

fn validate_contract_refs(
    contracts: &[EnergyContract],
    bus_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for contract in contracts {
        if !bus_index.contains_key(&contract.bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "EnergyContract",
                source_id: contract.id,
                field_name: "bus_id",
                referenced_id: contract.bus_id,
                expected_type: "Bus",
            });
        }
    }
}

fn validate_ncs_refs(
    non_controllable_sources: &[NonControllableSource],
    bus_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for ncs in non_controllable_sources {
        if !bus_index.contains_key(&ncs.bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "NonControllableSource",
                source_id: ncs.id,
                field_name: "bus_id",
                referenced_id: ncs.bus_id,
                expected_type: "Bus",
            });
        }
    }
}

/// Validate filling configurations for all hydros that have one.
///
/// For each hydro with `filling: Some(config)`:
/// - `filling_inflow_m3s` must be positive (> 0.0).
/// - `entry_stage_id` must be set (`Some`), since filling requires a known start stage.
///
/// All violations are appended to `errors` — no short-circuiting on first error.
pub(crate) fn validate_filling_configs(hydros: &[Hydro], errors: &mut Vec<ValidationError>) {
    for hydro in hydros {
        if let Some(filling) = &hydro.filling {
            if filling.filling_inflow_m3s.is_nan() || filling.filling_inflow_m3s <= 0.0 {
                errors.push(ValidationError::InvalidFillingConfig {
                    hydro_id: hydro.id,
                    reason: "filling_inflow_m3s must be positive".to_string(),
                });
            }
            if hydro.entry_stage_id.is_none() {
                errors.push(ValidationError::InvalidFillingConfig {
                    hydro_id: hydro.id,
                    reason: "filling requires entry_stage_id to be set".to_string(),
                });
            }
        }
    }
}
