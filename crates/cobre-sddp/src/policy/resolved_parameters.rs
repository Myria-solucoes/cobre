//! Pre-materialised lookup table: `(parameter_id, stage_idx)` → `f64`.
//!
//! ## Basis-cache invariance
//!
//! Resolved values freeze into [`StageTemplate`](cobre_solver::StageTemplate)
//! entries at construction and are never rebuilt; the hot-path solver patches
//! only row bounds, so LP matrix coefficients stay identical across iterations
//! and the warm-start basis cache needs no invalidation. A future change that
//! mutates parameter values across iterations (adaptive updates) breaks this and
//! must rebuild or invalidate the affected `StageTemplate` entries.

use std::collections::HashMap;

use cobre_core::{ComputedParameter, EntityId, Hydro, ParameterKind, ScalarParameter};
use thiserror::Error;

use crate::energy_conversion::{EnergyConversionSet, HydroEnergyProductivityOverride};
use crate::stage_key::StageId;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by [`build_resolved_parameters`].
#[derive(Debug, Error, PartialEq)]
pub enum ResolvedParametersError {
    /// A `PerStage` parameter vector has the wrong length.
    #[error("parameter '{name}': PerStage vector has {got} values but n_stages = {expected}")]
    PerStageLengthMismatch {
        /// Parameter name from the source record.
        name: String,
        /// Required length (`n_stages`).
        expected: usize,
        /// Actual vector length.
        got: usize,
    },

    /// A `Seasonal` parameter has no entry for the season required by a stage.
    #[error(
        "parameter '{name}': no seasonal value for season_id={season_id} (needed by stage {stage_idx})"
    )]
    MissingSeason {
        /// Parameter name from the source record.
        name: String,
        /// The season ID that had no matching entry.
        season_id: i32,
        /// The stage index that needed the season.
        stage_idx: usize,
    },

    /// A `Computed` parameter references a hydro plant that is not in `hydros`.
    #[error("parameter '{name}': computed variant references unknown hydro_id={hydro_id:?}")]
    UnknownHydro {
        /// Parameter name from the source record.
        name: String,
        /// The hydro entity ID that was not found.
        hydro_id: EntityId,
    },

    /// A `SpecificProductivity` variant could not be resolved because neither
    /// the override table nor the hydro entity record has a value.
    #[error(
        "parameter '{name}': no specific productivity (`ρ_esp`) available for hydro_id={hydro_id:?}"
    )]
    MissingSpecificProductivity {
        /// Parameter name from the source record.
        name: String,
        /// The hydro entity ID whose `ρ_esp` could not be found.
        hydro_id: EntityId,
    },
}

// ---------------------------------------------------------------------------
// ResolvedParameters
// ---------------------------------------------------------------------------

/// Dense lookup table mapping `(parameter_id, stage_idx)` → `f64`.
///
/// `id_to_slot` is sorted ascending by key — for declaration-order invariance
/// (a `HashMap` would not be deterministic) and `O(log n)` binary-search lookup.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResolvedParameters {
    /// Outer index: parameter slot (matches `Vec<ScalarParameter>` order).
    /// Inner index: `stage_idx` in `0..n_stages`.
    pub per_param: Vec<Vec<f64>>,
    /// Maps `EntityId.0` of the parameter to its slot in `per_param`.
    pub id_to_slot: Vec<(i32, usize)>,
}

impl ResolvedParameters {
    /// Return the resolved `f64` value for `(id, stage_idx)`.
    ///
    /// On a miss (unknown `id` or out-of-range `stage_idx`) returns `0.0` —
    /// mirroring the LP-build site sentinel — and `debug_assert`s in debug builds.
    #[must_use]
    pub fn get(&self, id: EntityId, stage_idx: usize) -> f64 {
        if let Ok(pos) = self.id_to_slot.binary_search_by_key(&id.0, |(k, _)| *k) {
            let slot = self.id_to_slot[pos].1;
            let row = &self.per_param[slot];
            if stage_idx < row.len() {
                row[stage_idx]
            } else {
                debug_assert!(
                    false,
                    "ResolvedParameters miss: id={id:?}, stage={stage_idx} (row len={})",
                    row.len()
                );
                0.0
            }
        } else {
            debug_assert!(
                false,
                "ResolvedParameters miss: id={id:?}, stage={stage_idx}"
            );
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Build a [`ResolvedParameters`] table from assembled scalar parameters and
/// energy-conversion data.
///
/// # Errors
///
/// Returns a [`ResolvedParametersError`] on the first failure encountered:
/// - [`PerStageLengthMismatch`](ResolvedParametersError::PerStageLengthMismatch)
///   when a `PerStage` vector length ≠ `n_stages`.
/// - [`MissingSeason`](ResolvedParametersError::MissingSeason) when a
///   `Seasonal` entry is absent for a required season.
/// - [`UnknownHydro`](ResolvedParametersError::UnknownHydro) when a `Computed`
///   variant references a `hydro_id` not present in `hydros`.
/// - [`MissingSpecificProductivity`](ResolvedParametersError::MissingSpecificProductivity)
///   when neither the override table nor the hydro entity record has a `ρ_esp`
///   value.
///
/// The constructor never panics; `n_stages == 0` produces an empty table.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, ParameterKind, ScalarParameter};
/// use cobre_sddp::energy_conversion::{
///     EnergyConversionSet, HydroEnergyProductivityOverride,
/// };
/// use cobre_sddp::stage_key::StageId;
/// use cobre_sddp::resolved_parameters::build_resolved_parameters;
///
/// let params = vec![ScalarParameter {
///     id: EntityId(1),
///     name: "constant_coeff".to_string(),
///     kind: ParameterKind::Constant { value: 3.6 },
/// }];
/// let ec = EnergyConversionSet::new(vec![], vec![], 0, 4);
/// let overrides = HydroEnergyProductivityOverride::default();
/// let stage_ids = [StageId(0), StageId(1), StageId(2), StageId(3)];
///
/// let table = build_resolved_parameters(&params, &ec, &overrides, &[], &[0, 0, 1, 1], &stage_ids, 4)
///     .unwrap();
///
/// assert!((table.get(EntityId(1), 0) - 3.6).abs() < 1e-12);
/// assert!((table.get(EntityId(1), 3) - 3.6).abs() < 1e-12);
/// ```
pub fn build_resolved_parameters(
    parameters: &[ScalarParameter],
    energy_conversion: &EnergyConversionSet,
    override_table: &HydroEnergyProductivityOverride,
    hydros: &[Hydro],
    stage_to_season: &[i32],
    stage_ids: &[StageId],
    n_stages: usize,
) -> Result<ResolvedParameters, ResolvedParametersError> {
    debug_assert_eq!(
        stage_ids.len(),
        n_stages,
        "stage_ids must carry one domain StageId per study stage"
    );
    let hydro_index: HashMap<EntityId, usize> = hydros
        .iter()
        .enumerate()
        .map(|(idx, h)| (h.id, idx))
        .collect();

    let mut per_param: Vec<Vec<f64>> = Vec::with_capacity(parameters.len());
    let mut id_to_slot: Vec<(i32, usize)> = Vec::with_capacity(parameters.len());

    for (slot, param) in parameters.iter().enumerate() {
        let values = resolve_kind(
            &param.kind,
            &param.name,
            n_stages,
            stage_to_season,
            stage_ids,
            energy_conversion,
            override_table,
            hydros,
            &hydro_index,
        )?;
        per_param.push(values);
        id_to_slot.push((param.id.0, slot));
    }

    // Uniqueness is enforced upstream (the JSON reader); the debug_assert makes
    // a duplicate observable at the resolver boundary.
    id_to_slot.sort_by_key(|(k, _)| *k);
    debug_assert!(
        id_to_slot.windows(2).all(|w| w[0].0 != w[1].0),
        "duplicate EntityId.0 values in parameters: {:?}",
        id_to_slot
            .windows(2)
            .filter(|w| w[0].0 == w[1].0)
            .map(|w| w[0].0)
            .collect::<Vec<_>>()
    );

    Ok(ResolvedParameters {
        per_param,
        id_to_slot,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Resolve a single [`ParameterKind`] into a `Vec<f64>` of length `n_stages`.
// Rationale: `resolve_computed` mirrors this arity so the two are interchangeable
// at the dispatch site; a context struct would just move the arity to the literal.
#[allow(clippy::too_many_arguments)]
fn resolve_kind(
    kind: &ParameterKind,
    name: &str,
    n_stages: usize,
    stage_to_season: &[i32],
    stage_ids: &[StageId],
    energy_conversion: &EnergyConversionSet,
    override_table: &HydroEnergyProductivityOverride,
    hydros: &[Hydro],
    hydro_index: &HashMap<EntityId, usize>,
) -> Result<Vec<f64>, ResolvedParametersError> {
    match kind {
        ParameterKind::Constant { value: c } => Ok(vec![*c; n_stages]),

        ParameterKind::PerStage { values: v } => {
            if v.len() != n_stages {
                return Err(ResolvedParametersError::PerStageLengthMismatch {
                    name: name.to_string(),
                    expected: n_stages,
                    got: v.len(),
                });
            }
            Ok(v.clone())
        }

        ParameterKind::Seasonal { values: pairs } => {
            let mut values = Vec::with_capacity(n_stages);
            for (t, &season_id) in stage_to_season.iter().enumerate() {
                // `pairs` is sorted ascending by season_id (invariant from
                // `ParameterKind::new_seasonal`).
                match pairs.binary_search_by_key(&season_id, |(k, _)| *k) {
                    Ok(pos) => values.push(pairs[pos].1),
                    Err(_) => {
                        return Err(ResolvedParametersError::MissingSeason {
                            name: name.to_string(),
                            season_id,
                            stage_idx: t,
                        });
                    }
                }
            }
            Ok(values)
        }

        ParameterKind::Computed { computed_spec: cp } => resolve_computed(
            *cp,
            name,
            n_stages,
            stage_to_season,
            stage_ids,
            energy_conversion,
            override_table,
            hydros,
            hydro_index,
        ),
    }
}

/// Resolve a [`ComputedParameter`] into a `Vec<f64>` of length `n_stages`.
// Rationale: the signature mirrors `resolve_kind` so the two are interchangeable
// at the `ParameterKind::Computed` dispatch site; a context struct would just move
// the arity to the literal.
#[allow(clippy::too_many_arguments)]
fn resolve_computed(
    cp: ComputedParameter,
    name: &str,
    n_stages: usize,
    _stage_to_season: &[i32],
    stage_ids: &[StageId],
    energy_conversion: &EnergyConversionSet,
    override_table: &HydroEnergyProductivityOverride,
    hydros: &[Hydro],
    hydro_index: &HashMap<EntityId, usize>,
) -> Result<Vec<f64>, ResolvedParametersError> {
    let hydro_id = match cp {
        ComputedParameter::EquivalentProductivity { hydro_id }
        | ComputedParameter::AccumulatedProductivity { hydro_id }
        | ComputedParameter::ReferenceVolume { hydro_id }
        | ComputedParameter::ReferenceTurbine { hydro_id }
        | ComputedParameter::MinStorage { hydro_id }
        | ComputedParameter::MaxStorage { hydro_id }
        | ComputedParameter::SpecificProductivity { hydro_id } => hydro_id,
    };

    let hydro_idx = hydro_index.get(&hydro_id).copied().ok_or_else(|| {
        ResolvedParametersError::UnknownHydro {
            name: name.to_string(),
            hydro_id,
        }
    })?;

    let hydro = &hydros[hydro_idx];

    // SpecificProductivity is resolved per-stage below, not replicated once: it may
    // carry stage-specific overrides.
    match cp {
        ComputedParameter::MinStorage { .. } => {
            return Ok(vec![hydro.min_storage_hm3; n_stages]);
        }
        ComputedParameter::MaxStorage { .. } => {
            return Ok(vec![hydro.max_storage_hm3; n_stages]);
        }
        _ => {}
    }

    // Stage-varying resolution for the remaining five variants.
    let mut values = Vec::with_capacity(n_stages);
    for (t, &stage_id) in stage_ids.iter().enumerate().take(n_stages) {
        let value = match cp {
            ComputedParameter::EquivalentProductivity { .. } => {
                energy_conversion
                    .conversion(hydro_idx, t)
                    .equivalent_productivity_mw_per_m3s
            }
            ComputedParameter::AccumulatedProductivity { .. } => {
                energy_conversion.accumulated_productivity(hydro_idx, t)
            }
            ComputedParameter::ReferenceVolume { .. } => {
                // Intentionally NO parquet `v_ref` override tier — unlike the
                // sibling `ReferenceTurbine` below, which consults the `q_ref`
                // override. Re-adding one "for symmetry" resurrects a retired
                // second source of truth that silently shadows the JSON value.
                energy_conversion
                    .conversion(hydro_idx, t)
                    .reference_volume_hm3
            }
            ComputedParameter::ReferenceTurbine { .. } => {
                // Keyed by the domain StageId (matches how the override table is
                // built and validated), never by the study position `t` used for
                // the position-indexed `energy_conversion` fallback below.
                override_table
                    .reference_outflow(hydro_id, stage_id)
                    .unwrap_or(
                        energy_conversion
                            .conversion(hydro_idx, t)
                            .reference_outflow_m3s,
                    )
            }
            ComputedParameter::MinStorage { .. } => {
                // Handled above via early return; unreachable here.
                hydro.min_storage_hm3
            }
            ComputedParameter::MaxStorage { .. } => {
                // Handled above via early return; unreachable here.
                hydro.max_storage_hm3
            }
            ComputedParameter::SpecificProductivity { .. } => override_table
                .specific_productivity(hydro_id, stage_id)
                .or(hydro.specific_productivity_mw_per_m3s_per_m)
                .ok_or_else(|| ResolvedParametersError::MissingSpecificProductivity {
                    name: name.to_string(),
                    hydro_id,
                })?,
        };
        values.push(value);
    }
    Ok(values)
}

// ---------------------------------------------------------------------------
// Postcard broadcast helpers
// ---------------------------------------------------------------------------

/// Wire format version for the [`ResolvedParameters`] postcard envelope.
///
/// Decoding rejects any mismatch so a heterogeneous MPI cluster (new binary on
/// one rank, old on another) fails fast rather than misinterpreting bytes. Bump
/// in lockstep with any envelope-layout change.
pub const RESOLVED_PARAMETERS_WIRE_VERSION: u32 = 1;

/// Version-tagged envelope wrapping a [`ResolvedParameters`] payload for MPI
/// broadcast; decode rejects a non-matching `version`.
#[derive(serde::Serialize, serde::Deserialize)]
struct ResolvedParametersWireEnvelope {
    version: u32,
    payload: ResolvedParameters,
}

/// Serialize a [`ResolvedParameters`] table to a versioned postcard envelope for
/// MPI broadcast. Pure ser/de; the caller owns the broadcast call.
///
/// # Errors
///
/// Returns [`crate::SddpError::Validation`] if postcard serialization fails (not
/// reachable with the current struct layout; the path satisfies the
/// `Result<_, SddpError>` contract shared by other broadcast helpers).
///
/// # Examples
///
/// ```
/// use cobre_sddp::resolved_parameters::{
///     build_resolved_parameters, deserialize_resolved_parameters,
///     serialize_resolved_parameters,
/// };
/// use cobre_core::{EntityId, ParameterKind, ScalarParameter};
/// use cobre_sddp::energy_conversion::{EnergyConversionSet, HydroEnergyProductivityOverride};
/// use cobre_sddp::stage_key::StageId;
///
/// let params = vec![ScalarParameter {
///     id: EntityId(1),
///     name: "rho_eq".to_string(),
///     kind: ParameterKind::Constant { value: 3.6 },
/// }];
/// let ec = EnergyConversionSet::new(vec![], vec![], 0, 4);
/// let overrides = HydroEnergyProductivityOverride::default();
/// let stage_ids = [StageId(0), StageId(1), StageId(2), StageId(3)];
/// let table = build_resolved_parameters(&params, &ec, &overrides, &[], &[0, 0, 1, 1], &stage_ids, 4)
///     .unwrap();
///
/// let bytes = serialize_resolved_parameters(&table).unwrap();
/// let restored = deserialize_resolved_parameters(&bytes).unwrap();
/// assert_eq!(
///     table.get(EntityId(1), 0).to_bits(),
///     restored.get(EntityId(1), 0).to_bits()
/// );
/// ```
pub fn serialize_resolved_parameters(
    table: &ResolvedParameters,
) -> Result<Vec<u8>, crate::error::SddpError> {
    let envelope = ResolvedParametersWireEnvelope {
        version: RESOLVED_PARAMETERS_WIRE_VERSION,
        payload: table.clone(),
    };
    postcard::to_allocvec(&envelope).map_err(|e| {
        crate::error::SddpError::Validation(format!("postcard resolved_parameters: {e}"))
    })
}

/// Deserialize a [`ResolvedParameters`] table from a postcard byte buffer
/// produced by [`serialize_resolved_parameters`].
///
/// A version mismatch returns a distinct error from corruption, never silently
/// accepting stale bytes.
///
/// # Errors
///
/// - [`crate::SddpError::Validation`] — byte slice is corrupted, truncated, or not a
///   valid postcard encoding of the versioned envelope.
/// - [`crate::SddpError::WireVersionMismatch`] — the encoded version does not match
///   [`RESOLVED_PARAMETERS_WIRE_VERSION`]; a binary mismatch across MPI ranks.
///
/// # Examples
///
/// ```
/// use cobre_sddp::resolved_parameters::{
///     deserialize_resolved_parameters, serialize_resolved_parameters,
/// };
/// use cobre_core::{EntityId, ParameterKind, ScalarParameter};
/// use cobre_sddp::energy_conversion::{EnergyConversionSet, HydroEnergyProductivityOverride};
/// use cobre_sddp::stage_key::StageId;
/// use cobre_sddp::resolved_parameters::build_resolved_parameters;
///
/// let params = vec![ScalarParameter {
///     id: EntityId(1),
///     name: "rho_eq".to_string(),
///     kind: ParameterKind::Constant { value: 3.6 },
/// }];
/// let ec = EnergyConversionSet::new(vec![], vec![], 0, 4);
/// let overrides = HydroEnergyProductivityOverride::default();
/// let stage_ids = [StageId(0), StageId(1), StageId(2), StageId(3)];
/// let table = build_resolved_parameters(&params, &ec, &overrides, &[], &[0, 0, 1, 1], &stage_ids, 4)
///     .unwrap();
///
/// let bytes = serialize_resolved_parameters(&table).unwrap();
/// let restored = deserialize_resolved_parameters(&bytes).unwrap();
/// assert_eq!(
///     table.get(EntityId(1), 2).to_bits(),
///     restored.get(EntityId(1), 2).to_bits()
/// );
/// ```
pub fn deserialize_resolved_parameters(
    bytes: &[u8],
) -> Result<ResolvedParameters, crate::error::SddpError> {
    let envelope: ResolvedParametersWireEnvelope = postcard::from_bytes(bytes).map_err(|e| {
        crate::error::SddpError::Validation(format!("postcard resolved_parameters: {e}"))
    })?;
    if envelope.version != RESOLVED_PARAMETERS_WIRE_VERSION {
        return Err(crate::error::SddpError::WireVersionMismatch {
            encoded: envelope.version,
            expected: RESOLVED_PARAMETERS_WIRE_VERSION,
        });
    }
    Ok(envelope.payload)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use cobre_core::{
        ComputedParameter, EntityId, ParameterKind, ScalarParameter,
        entities::hydro::{HydroGenerationModel, HydroPenalties},
    };

    use cobre_io::HydroEnergyProductivityRow;

    use super::*;
    use crate::energy_conversion::{
        EnergyConversion, EnergyConversionSet, build_hydro_energy_productivity_override,
    };

    // -------------------------------------------------------------------------
    // Shared test helpers
    // -------------------------------------------------------------------------

    /// Build a minimal [`Hydro`] with the given storage limits and optional
    /// `specific_productivity_mw_per_m3s_per_m`.
    fn make_hydro(
        id: i32,
        min_storage: f64,
        max_storage: f64,
        specific_productivity: Option<f64>,
    ) -> Hydro {
        Hydro {
            id: EntityId(id),
            name: format!("h{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_storage_hm3: min_storage,
            max_storage_hm3: max_storage,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 500.0,
            specific_productivity_mw_per_m3s_per_m: specific_productivity,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
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
            },
        }
    }

    /// Build an [`EnergyConversionSet`] with `n_hydros` hydros, each with
    /// `n_stages` stages. Every `(hydro, stage)` cell is populated with
    /// predictable values derived from the hydro and stage indices.
    fn make_energy_conversion(n_hydros: usize, n_stages: usize) -> EnergyConversionSet {
        let per_hydro_stage: Vec<Vec<EnergyConversion>> = (0..n_hydros)
            .map(|h| {
                (0..n_stages)
                    .map(|t| EnergyConversion {
                        equivalent_productivity_mw_per_m3s: 0.42 + h as f64 + t as f64 * 0.01,
                        reference_volume_hm3: 1000.0 + h as f64 * 100.0 + t as f64,
                        reference_outflow_m3s: 500.0 + h as f64 * 50.0 + t as f64 * 0.1,
                    })
                    .collect()
            })
            .collect();
        let accumulated: Vec<Vec<f64>> = (0..n_hydros)
            .map(|h| {
                (0..n_stages)
                    .map(|t| 0.90 + h as f64 * 0.05 + t as f64 * 0.001)
                    .collect()
            })
            .collect();
        EnergyConversionSet::new(per_hydro_stage, accumulated, n_hydros, n_stages)
    }

    /// `StageId(0)..StageId(n_stages - 1)`: the 0-based domain ids every test
    /// fixture in this module uses (no pre-study-stage offset), so study
    /// position and domain id coincide — matching production behavior on the
    /// existing 0-based studies this ticket must leave unchanged.
    fn stage_ids_0_based(n_stages: usize) -> Vec<StageId> {
        (0..n_stages)
            .map(|s| StageId(i32::try_from(s).expect("test stage count fits in i32")))
            .collect()
    }

    /// Return `(hydros, energy_conversion, override_table, stage_to_season, stage_ids)`
    /// for tests that need a consistent set of inputs.
    fn make_setup_inputs(
        n_stages: usize,
    ) -> (
        Vec<Hydro>,
        EnergyConversionSet,
        HydroEnergyProductivityOverride,
        Vec<i32>,
        Vec<StageId>,
    ) {
        let hydros = vec![
            make_hydro(0, 50.0, 2000.0, Some(0.0085)),
            make_hydro(1, 100.0, 5000.0, Some(0.0090)),
        ];
        let energy_conversion = make_energy_conversion(2, n_stages);
        let override_table = HydroEnergyProductivityOverride::default();
        let stage_to_season: Vec<i32> = (0..n_stages).map(|t| (t % 4) as i32).collect();
        let stage_ids = stage_ids_0_based(n_stages);
        (
            hydros,
            energy_conversion,
            override_table,
            stage_to_season,
            stage_ids,
        )
    }

    fn make_param(id: i32, kind: ParameterKind) -> ScalarParameter {
        ScalarParameter {
            id: EntityId(id),
            name: format!("param_{id}"),
            kind,
        }
    }

    // -------------------------------------------------------------------------
    // AC-1: Constant kind fills all stages
    // -------------------------------------------------------------------------

    #[test]
    fn constant_kind_fills_all_stages() {
        let params = vec![make_param(0, ParameterKind::Constant { value: 3.6 })];
        let ec = EnergyConversionSet::new(vec![], vec![], 0, 4);
        let overrides = HydroEnergyProductivityOverride::default();
        let stage_to_season = vec![0i32; 4];
        let stage_ids = stage_ids_0_based(4);

        let table = build_resolved_parameters(
            &params,
            &ec,
            &overrides,
            &[],
            &stage_to_season,
            &stage_ids,
            4,
        )
        .unwrap();

        assert!((table.get(EntityId(0), 0) - 3.6).abs() < 1e-12);
        assert!((table.get(EntityId(0), 1) - 3.6).abs() < 1e-12);
        assert!((table.get(EntityId(0), 2) - 3.6).abs() < 1e-12);
        assert!((table.get(EntityId(0), 3) - 3.6).abs() < 1e-12);
    }

    // -------------------------------------------------------------------------
    // AC-2: PerStage length mismatch errors
    // -------------------------------------------------------------------------

    #[test]
    fn per_stage_kind_length_mismatch_errors() {
        let params = vec![make_param(
            0,
            ParameterKind::PerStage {
                values: vec![1.0, 2.0],
            },
        )];
        let ec = EnergyConversionSet::new(vec![], vec![], 0, 3);
        let overrides = HydroEnergyProductivityOverride::default();
        let stage_to_season = vec![0i32; 3];
        let stage_ids = stage_ids_0_based(3);

        let result = build_resolved_parameters(
            &params,
            &ec,
            &overrides,
            &[],
            &stage_to_season,
            &stage_ids,
            3,
        );

        assert!(matches!(
            result,
            Err(ResolvedParametersError::PerStageLengthMismatch {
                expected: 3,
                got: 2,
                ..
            })
        ));
    }

    // -------------------------------------------------------------------------
    // AC-3: Seasonal kind maps stage to value
    // -------------------------------------------------------------------------

    #[test]
    fn seasonal_kind_maps_stage_to_value() {
        let params = vec![make_param(
            0,
            ParameterKind::Seasonal {
                values: vec![(0, 0.5), (1, 1.5)],
            },
        )];
        let ec = EnergyConversionSet::new(vec![], vec![], 0, 3);
        let overrides = HydroEnergyProductivityOverride::default();
        let stage_to_season = vec![0i32, 1, 0];
        let stage_ids = stage_ids_0_based(3);

        let table = build_resolved_parameters(
            &params,
            &ec,
            &overrides,
            &[],
            &stage_to_season,
            &stage_ids,
            3,
        )
        .unwrap();

        assert!((table.get(EntityId(0), 0) - 0.5).abs() < 1e-12);
        assert!((table.get(EntityId(0), 1) - 1.5).abs() < 1e-12);
        assert!((table.get(EntityId(0), 2) - 0.5).abs() < 1e-12);
    }

    // -------------------------------------------------------------------------
    // AC-4: Computed EquivalentProductivity reads EnergyConversionSet
    // -------------------------------------------------------------------------

    #[test]
    fn computed_equivalent_productivity_reads_energy_conversion() {
        let n_stages = 4;
        let (hydros, energy_conversion, override_table, stage_to_season, stage_ids) =
            make_setup_inputs(n_stages);

        let params = vec![make_param(
            0,
            ParameterKind::Computed {
                computed_spec: ComputedParameter::EquivalentProductivity {
                    hydro_id: EntityId(0),
                },
            },
        )];

        let table = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            n_stages,
        )
        .unwrap();

        let expected = energy_conversion
            .conversion(0, 0)
            .equivalent_productivity_mw_per_m3s;
        assert!((table.get(EntityId(0), 0) - expected).abs() < 1e-12);
        // Validate all stages
        for t in 0..n_stages {
            let exp_t = energy_conversion
                .conversion(0, t)
                .equivalent_productivity_mw_per_m3s;
            assert!(
                (table.get(EntityId(0), t) - exp_t).abs() < 1e-12,
                "stage {t}: expected {exp_t}, got {}",
                table.get(EntityId(0), t)
            );
        }
    }

    // -------------------------------------------------------------------------
    // ReferenceVolume reads the JSON-sourced energy-conversion cell directly
    // (no parquet `v_ref` override tier) — the surviving half of the asymmetry.
    // -------------------------------------------------------------------------

    #[test]
    fn reference_volume_reads_energy_conversion_cell() {
        let n_stages = 4;
        let (hydros, energy_conversion, override_table, stage_to_season, stage_ids) =
            make_setup_inputs(n_stages);

        let params = vec![make_param(
            0,
            ParameterKind::Computed {
                computed_spec: ComputedParameter::ReferenceVolume {
                    hydro_id: EntityId(1),
                },
            },
        )];

        let table = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            n_stages,
        )
        .unwrap();

        // For every stage the resolved value is bit-for-bit the energy-conversion
        // cell — there is no override branch that could shadow it.
        for t in 0..n_stages {
            let expected = energy_conversion.conversion(1, t).reference_volume_hm3;
            assert_eq!(
                table.get(EntityId(0), t).to_bits(),
                expected.to_bits(),
                "stage {t}: ReferenceVolume must equal the energy-conversion cell"
            );
        }
    }

    // -------------------------------------------------------------------------
    // ReferenceTurbine STILL consults the parquet `q_ref` override (the other
    // half of the intentional asymmetry documented at the resolution site).
    // -------------------------------------------------------------------------

    #[test]
    fn reference_turbine_consults_parquet_override() {
        let n_stages = 3;
        let (hydros, energy_conversion, _default_override, stage_to_season, stage_ids) =
            make_setup_inputs(n_stages);

        // A per-hydro-default q_ref override that differs from every
        // energy-conversion `reference_outflow_m3s` cell.
        let override_q_ref = 12_345.0;
        let override_table =
            build_hydro_energy_productivity_override(&[HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: None,
                equivalent_productivity_mw_per_m3s: None,
                reference_outflow_m3s: Some(override_q_ref),
                specific_productivity_mw_per_m3s_per_m: None,
            }])
            .unwrap();

        let params = vec![make_param(
            0,
            ParameterKind::Computed {
                computed_spec: ComputedParameter::ReferenceTurbine {
                    hydro_id: EntityId(1),
                },
            },
        )];

        let table = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            n_stages,
        )
        .unwrap();

        for t in 0..n_stages {
            assert_eq!(
                table.get(EntityId(0), t).to_bits(),
                override_q_ref.to_bits(),
                "stage {t}: ReferenceTurbine must honor the parquet q_ref override"
            );
        }
    }

    /// A stage-specific override (not the per-hydro default) resolves by
    /// domain `StageId`, not by the study position `t`. The study has a
    /// single stage whose position is 0 but whose domain id is 60; keying
    /// the lookup at position 0 would miss the `(hydro, 60)` entry and fall
    /// through to the `energy_conversion` / hydro-entity fallback instead.
    #[test]
    fn reference_turbine_and_specific_productivity_key_by_domain_stage_id_not_position() {
        let hydros = vec![make_hydro(1, 100.0, 5000.0, Some(0.0090))];
        let energy_conversion = make_energy_conversion(1, 1);
        let stage_to_season = vec![0i32];
        let stage_ids = vec![StageId(60)];

        let override_q_ref = 777.0;
        let override_esp = 0.0123;
        let override_table =
            build_hydro_energy_productivity_override(&[HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: Some(60),
                equivalent_productivity_mw_per_m3s: None,
                reference_outflow_m3s: Some(override_q_ref),
                specific_productivity_mw_per_m3s_per_m: Some(override_esp),
            }])
            .unwrap();

        let params = vec![
            make_param(
                0,
                ParameterKind::Computed {
                    computed_spec: ComputedParameter::ReferenceTurbine {
                        hydro_id: EntityId(1),
                    },
                },
            ),
            make_param(
                1,
                ParameterKind::Computed {
                    computed_spec: ComputedParameter::SpecificProductivity {
                        hydro_id: EntityId(1),
                    },
                },
            ),
        ];

        let table = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            1,
        )
        .unwrap();

        assert_eq!(
            table.get(EntityId(0), 0).to_bits(),
            override_q_ref.to_bits(),
            "ReferenceTurbine must resolve the StageId(60) override at position 0"
        );
        assert_eq!(
            table.get(EntityId(1), 0).to_bits(),
            override_esp.to_bits(),
            "SpecificProductivity must resolve the StageId(60) override at position 0"
        );
    }

    // -------------------------------------------------------------------------
    // AC-5: SpecificProductivity missing returns error
    // -------------------------------------------------------------------------

    #[test]
    fn computed_specific_productivity_missing_returns_error() {
        let n_stages = 2;
        // Hydro with no specific_productivity
        let hydros = vec![make_hydro(0, 50.0, 2000.0, None)];
        let energy_conversion = make_energy_conversion(1, n_stages);
        let override_table = HydroEnergyProductivityOverride::default();
        let stage_to_season = vec![0i32; n_stages];
        let stage_ids = stage_ids_0_based(n_stages);

        let params = vec![make_param(
            0,
            ParameterKind::Computed {
                computed_spec: ComputedParameter::SpecificProductivity {
                    hydro_id: EntityId(0),
                },
            },
        )];

        let result = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            n_stages,
        );

        assert!(matches!(
            result,
            Err(ResolvedParametersError::MissingSpecificProductivity {
                hydro_id: EntityId(0),
                ..
            })
        ));
    }

    // -------------------------------------------------------------------------
    // AC-6: Declaration-order invariance
    // -------------------------------------------------------------------------

    #[test]
    fn declaration_order_invariance() {
        let n_stages = 3;
        let (hydros, energy_conversion, override_table, stage_to_season, stage_ids) =
            make_setup_inputs(n_stages);

        let param_a = make_param(10, ParameterKind::Constant { value: 1.0 });
        let param_b = make_param(
            20,
            ParameterKind::PerStage {
                values: vec![2.0, 3.0, 4.0],
            },
        );
        let param_c = make_param(
            30,
            ParameterKind::Computed {
                computed_spec: ComputedParameter::AccumulatedProductivity {
                    hydro_id: EntityId(0),
                },
            },
        );

        let params_abc = vec![param_a.clone(), param_b.clone(), param_c.clone()];
        let params_cab = vec![param_c.clone(), param_a.clone(), param_b.clone()];

        let table_abc = build_resolved_parameters(
            &params_abc,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            n_stages,
        )
        .unwrap();
        let table_cab = build_resolved_parameters(
            &params_cab,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            n_stages,
        )
        .unwrap();

        for id in [EntityId(10), EntityId(20), EntityId(30)] {
            for t in 0..n_stages {
                assert_eq!(
                    table_abc.get(id, t).to_bits(),
                    table_cab.get(id, t).to_bits(),
                    "bit mismatch for id={id:?}, stage={t}"
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // UnknownHydro variant
    // -------------------------------------------------------------------------

    #[test]
    fn unknown_hydro_in_computed_returns_error() {
        let n_stages = 2;
        let (hydros, energy_conversion, override_table, stage_to_season, stage_ids) =
            make_setup_inputs(n_stages);

        // Use a hydro_id that does not exist in the hydros slice (ids 0 and 1 exist).
        let params = vec![make_param(
            0,
            ParameterKind::Computed {
                computed_spec: ComputedParameter::MinStorage {
                    hydro_id: EntityId(99),
                },
            },
        )];

        let result = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            n_stages,
        );

        assert!(matches!(
            result,
            Err(ResolvedParametersError::UnknownHydro {
                hydro_id: EntityId(99),
                ..
            })
        ));
    }

    // -------------------------------------------------------------------------
    // MinStorage / MaxStorage: stage-invariant hydro field
    // -------------------------------------------------------------------------

    #[test]
    fn min_max_storage_use_stage_invariant_hydro_field() {
        let n_stages = 5;
        let (hydros, energy_conversion, override_table, stage_to_season, stage_ids) =
            make_setup_inputs(n_stages);

        let params = vec![
            make_param(
                0,
                ParameterKind::Computed {
                    computed_spec: ComputedParameter::MinStorage {
                        hydro_id: EntityId(0),
                    },
                },
            ),
            make_param(
                1,
                ParameterKind::Computed {
                    computed_spec: ComputedParameter::MaxStorage {
                        hydro_id: EntityId(1),
                    },
                },
            ),
        ];

        let table = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            n_stages,
        )
        .unwrap();

        // hydro 0: min_storage = 50.0
        for t in 0..n_stages {
            assert!(
                (table.get(EntityId(0), t) - 50.0).abs() < 1e-12,
                "min_storage mismatch at stage {t}"
            );
        }
        // hydro 1: max_storage = 5000.0
        for t in 0..n_stages {
            assert!(
                (table.get(EntityId(1), t) - 5000.0).abs() < 1e-12,
                "max_storage mismatch at stage {t}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Empty parameter slice: n_stages = 0 does not panic
    // -------------------------------------------------------------------------

    #[test]
    fn empty_parameters_n_stages_zero_is_ok() {
        let ec = EnergyConversionSet::new(vec![], vec![], 0, 0);
        let overrides = HydroEnergyProductivityOverride::default();

        let table = build_resolved_parameters(&[], &ec, &overrides, &[], &[], &[], 0).unwrap();
        // Nothing to query — just verify it doesn't panic.
        let _ = table;
    }

    // -------------------------------------------------------------------------
    // Broadcast helper tests
    // -------------------------------------------------------------------------

    /// Build a populated [`ResolvedParameters`] exercising `Constant`,
    /// `PerStage`, and `Computed(EquivalentProductivity)` resolution paths.
    fn broadcast_fixture() -> ResolvedParameters {
        let n_stages = 4;
        let (hydros, energy_conversion, override_table, stage_to_season, stage_ids) =
            make_setup_inputs(n_stages);

        let params = vec![
            make_param(10, ParameterKind::Constant { value: 3.6 }),
            make_param(
                20,
                ParameterKind::PerStage {
                    values: vec![1.0, 2.0, 3.0, 4.0],
                },
            ),
            make_param(
                30,
                ParameterKind::Computed {
                    computed_spec: ComputedParameter::EquivalentProductivity {
                        hydro_id: EntityId(0),
                    },
                },
            ),
        ];

        build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            n_stages,
        )
        .unwrap()
    }

    #[test]
    fn round_trip_populated_resolved_parameters() {
        let original = broadcast_fixture();
        let bytes = super::serialize_resolved_parameters(&original).unwrap();
        assert!(!bytes.is_empty());
        let restored = super::deserialize_resolved_parameters(&bytes).unwrap();

        for id in [EntityId(10), EntityId(20), EntityId(30)] {
            for t in 0..4_usize {
                assert_eq!(
                    original.get(id, t).to_bits(),
                    restored.get(id, t).to_bits(),
                    "bit mismatch for id={id:?}, stage={t}"
                );
            }
        }
    }

    #[test]
    fn serialize_resolved_parameters_is_deterministic() {
        let table = broadcast_fixture();
        let bytes_a = super::serialize_resolved_parameters(&table).unwrap();
        let bytes_b = super::serialize_resolved_parameters(&table).unwrap();
        assert_eq!(bytes_a, bytes_b, "postcard encoding must be deterministic");
    }

    #[test]
    fn deserialize_resolved_parameters_rejects_corrupted_bytes() {
        let result = super::deserialize_resolved_parameters(&[0xFF, 0xFE, 0xFD, 0xFC]);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("postcard resolved_parameters"),
            "error message must contain 'postcard resolved_parameters', got: {msg}"
        );
    }

    #[test]
    fn deserialize_resolved_parameters_rejects_empty_buffer() {
        let result = super::deserialize_resolved_parameters(&[]);
        assert!(result.is_err(), "empty buffer must return Err");
    }

    #[test]
    fn round_trip_empty_resolved_parameters() {
        let table = ResolvedParameters::default();
        let bytes = super::serialize_resolved_parameters(&table).unwrap();
        // Postcard encodes empty Vecs as a short varint-prefixed payload.
        assert!(
            !bytes.is_empty(),
            "even an empty table must produce non-empty bytes"
        );
        let restored = super::deserialize_resolved_parameters(&bytes).unwrap();
        // Both tables are empty; any get() on an unknown id returns 0.0 via debug_assert path.
        let _ = restored;
    }

    // -------------------------------------------------------------------------
    // Wire envelope round-trip tests
    // -------------------------------------------------------------------------

    #[test]
    fn resolved_parameters_wire_envelope_round_trip() {
        let original = broadcast_fixture();
        let bytes = super::serialize_resolved_parameters(&original).unwrap();
        assert!(!bytes.is_empty(), "encoded envelope must be non-empty");
        let restored = super::deserialize_resolved_parameters(&bytes).unwrap();

        for id in [EntityId(10), EntityId(20), EntityId(30)] {
            for t in 0..4_usize {
                assert_eq!(
                    original.get(id, t).to_bits(),
                    restored.get(id, t).to_bits(),
                    "bit mismatch for id={id:?}, stage={t}"
                );
            }
        }
    }

    #[test]
    fn resolved_parameters_wire_envelope_rejects_old_version() {
        // Encode an envelope with a version older than RESOLVED_PARAMETERS_WIRE_VERSION.
        let old_version: u32 = super::RESOLVED_PARAMETERS_WIRE_VERSION.saturating_sub(1);
        let envelope = super::ResolvedParametersWireEnvelope {
            version: old_version,
            payload: ResolvedParameters::default(),
        };
        let bytes = postcard::to_allocvec(&envelope).expect("postcard encoding must not fail");

        let result = super::deserialize_resolved_parameters(&bytes);
        assert!(
            result.is_err(),
            "decoder must reject an old-version envelope"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("version"),
            "error message must contain 'version', got: {msg}"
        );
    }
}
