//! Pre-materialised lookup table: `(parameter_id, stage_idx)` → `f64`.
//!
//! [`ResolvedParameters`] is built once before LP construction by
//! [`build_resolved_parameters`]. It resolves all four [`ParameterKind`]
//! variants — `Constant`, `PerStage`, `Seasonal`, and `Computed` — into a
//! dense two-dimensional array indexed by parameter slot and stage index.
//!
//! The seven [`ComputedParameter`] variants are resolved against
//! [`EnergyConversionSet`] and [`HydroEnergyProductivityOverride`] for
//! hydro-indexed quantities, or directly from the [`Hydro`] entity record for
//! stage-invariant storage limits.
//!
//! ## Index layout
//!
//! The outer dimension of `per_param` preserves input declaration order so
//! that the table is deterministic regardless of parameter ordering in the
//! source file. [`id_to_slot`](ResolvedParameters::id_to_slot) is sorted
//! ascending by `EntityId.0` and searched with `binary_search` at query time,
//! giving `O(log n)` lookup with no hashing.

use std::collections::HashMap;

use cobre_core::{ComputedParameter, EntityId, Hydro, ParameterKind, ScalarParameter};
use thiserror::Error;

use crate::energy_conversion::{EnergyConversionSet, HydroEnergyProductivityOverride};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by [`build_resolved_parameters`].
///
/// Each variant carries the parameter `name` so callers can produce
/// user-friendly diagnostics without re-looking it up.
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
/// Built once before LP construction by [`build_resolved_parameters`] and
/// queried by the LP builder via [`get`](ResolvedParameters::get).
///
/// ## Memory layout
///
/// The outer dimension (`per_param`) is indexed by parameter slot in
/// declaration order. Each inner `Vec<f64>` has length `n_stages` and is
/// populated contiguously in stage order for cache-friendly sequential access.
///
/// ## Lookup
///
/// [`id_to_slot`](ResolvedParameters) is a sorted `Vec<(i32, usize)>` that
/// maps `EntityId.0` to a slot index via `binary_search`. This preserves
/// declaration-order invariance under postcard serialisation and avoids the
/// non-determinism of `HashMap`.
#[derive(Debug, Default, Clone)]
pub struct ResolvedParameters {
    /// Outer index: parameter slot (dense, matches `Vec<ScalarParameter>` order).
    /// Inner index: `stage_idx` in `0..n_stages`.
    per_param: Vec<Vec<f64>>,
    /// Maps `EntityId.0` of the parameter to its slot in `per_param`.
    /// Sorted ascending by key for declaration-order invariance and `O(log n)`
    /// binary-search lookup.
    id_to_slot: Vec<(i32, usize)>,
}

impl ResolvedParameters {
    /// Return the resolved `f64` value for `(id, stage_idx)`.
    ///
    /// Performs a binary search over `id_to_slot` and returns
    /// `per_param[slot][stage_idx]`.
    ///
    /// In debug builds, asserts on a miss (unknown `id` or out-of-range
    /// `stage_idx`) and returns `0.0` — mirroring the LP-build site sentinel.
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
/// Iterates `parameters` in declaration order, resolves each [`ParameterKind`]
/// variant, and stores the resulting `Vec<f64>` (length `n_stages`) at the
/// corresponding slot.
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
/// use cobre_sddp::resolved_parameters::build_resolved_parameters;
///
/// let params = vec![ScalarParameter {
///     id: EntityId(1),
///     name: "constant_coeff".to_string(),
///     kind: ParameterKind::Constant(3.6),
/// }];
/// let ec = EnergyConversionSet::new(vec![], vec![], 0, 4);
/// let overrides = HydroEnergyProductivityOverride::default();
///
/// let table = build_resolved_parameters(&params, &ec, &overrides, &[], &[0, 0, 1, 1], 4)
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
    n_stages: usize,
) -> Result<ResolvedParameters, ResolvedParametersError> {
    // Build a hydro_id → positional index map for O(1) lookup during Computed
    // variant resolution. The map is built once and dropped before returning.
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
            energy_conversion,
            override_table,
            hydros,
            &hydro_index,
        )?;
        per_param.push(values);
        id_to_slot.push((param.id.0, slot));
    }

    // Sort by EntityId.0 for O(log n) binary-search lookup. Adjacent-equality
    // check documents the uniqueness invariant (Epic 04 already enforced it
    // on the input but a debug_assert here makes it observable at the resolver
    // boundary).
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
#[allow(clippy::too_many_arguments)]
fn resolve_kind(
    kind: &ParameterKind,
    name: &str,
    n_stages: usize,
    stage_to_season: &[i32],
    energy_conversion: &EnergyConversionSet,
    override_table: &HydroEnergyProductivityOverride,
    hydros: &[Hydro],
    hydro_index: &HashMap<EntityId, usize>,
) -> Result<Vec<f64>, ResolvedParametersError> {
    match kind {
        ParameterKind::Constant(c) => Ok(vec![*c; n_stages]),

        ParameterKind::PerStage(v) => {
            if v.len() != n_stages {
                return Err(ResolvedParametersError::PerStageLengthMismatch {
                    name: name.to_string(),
                    expected: n_stages,
                    got: v.len(),
                });
            }
            Ok(v.clone())
        }

        ParameterKind::Seasonal(pairs) => {
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

        ParameterKind::Computed(cp) => resolve_computed(
            *cp,
            name,
            n_stages,
            stage_to_season,
            energy_conversion,
            override_table,
            hydros,
            hydro_index,
        ),
    }
}

/// Resolve a [`ComputedParameter`] into a `Vec<f64>` of length `n_stages`.
#[allow(clippy::too_many_arguments)]
fn resolve_computed(
    cp: ComputedParameter,
    name: &str,
    n_stages: usize,
    stage_to_season: &[i32],
    energy_conversion: &EnergyConversionSet,
    override_table: &HydroEnergyProductivityOverride,
    hydros: &[Hydro],
    hydro_index: &HashMap<EntityId, usize>,
) -> Result<Vec<f64>, ResolvedParametersError> {
    // Extract hydro_id from whichever variant is active (all seven carry it).
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

    // Stage-invariant variants can be resolved once and replicated.
    // SpecificProductivity and all remaining variants are handled per-stage below
    // (SpecificProductivity could be replicated but may have stage-specific overrides,
    // so correctness requires per-stage resolution).
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
    for t in 0..n_stages {
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
                override_table.reference_volume(hydro_id, t).unwrap_or(
                    energy_conversion
                        .conversion(hydro_idx, t)
                        .reference_volume_hm3,
                )
            }
            ComputedParameter::ReferenceTurbine { .. } => {
                override_table.reference_outflow(hydro_id, t).unwrap_or(
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
                .specific_productivity(hydro_id, t)
                .or(hydro.specific_productivity_mw_per_m3s_per_m)
                .ok_or_else(|| ResolvedParametersError::MissingSpecificProductivity {
                    name: name.to_string(),
                    hydro_id,
                })?,
        };
        // `stage_to_season` is borrowed but unused for stage-varying resolution
        // (it's only needed for Seasonal lookup). Suppress the lint.
        let _ = stage_to_season;
        values.push(value);
    }
    Ok(values)
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

    use super::*;
    use crate::energy_conversion::{EnergyConversion, EnergyConversionSet};

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
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.85,
            },
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
                fpha_turbined_cost: 0.0,
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

    /// Return `(hydros, energy_conversion, override_table, stage_to_season)`
    /// for tests that need a consistent set of inputs.
    fn make_setup_inputs(
        n_stages: usize,
    ) -> (
        Vec<Hydro>,
        EnergyConversionSet,
        HydroEnergyProductivityOverride,
        Vec<i32>,
    ) {
        let hydros = vec![
            make_hydro(0, 50.0, 2000.0, Some(0.0085)),
            make_hydro(1, 100.0, 5000.0, Some(0.0090)),
        ];
        let energy_conversion = make_energy_conversion(2, n_stages);
        let override_table = HydroEnergyProductivityOverride::default();
        let stage_to_season: Vec<i32> = (0..n_stages).map(|t| (t % 4) as i32).collect();
        (hydros, energy_conversion, override_table, stage_to_season)
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
        let params = vec![make_param(0, ParameterKind::Constant(3.6))];
        let ec = EnergyConversionSet::new(vec![], vec![], 0, 4);
        let overrides = HydroEnergyProductivityOverride::default();
        let stage_to_season = vec![0i32; 4];

        let table =
            build_resolved_parameters(&params, &ec, &overrides, &[], &stage_to_season, 4).unwrap();

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
        let params = vec![make_param(0, ParameterKind::PerStage(vec![1.0, 2.0]))];
        let ec = EnergyConversionSet::new(vec![], vec![], 0, 3);
        let overrides = HydroEnergyProductivityOverride::default();
        let stage_to_season = vec![0i32; 3];

        let result = build_resolved_parameters(&params, &ec, &overrides, &[], &stage_to_season, 3);

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
            ParameterKind::Seasonal(vec![(0, 0.5), (1, 1.5)]),
        )];
        let ec = EnergyConversionSet::new(vec![], vec![], 0, 3);
        let overrides = HydroEnergyProductivityOverride::default();
        let stage_to_season = vec![0i32, 1, 0];

        let table =
            build_resolved_parameters(&params, &ec, &overrides, &[], &stage_to_season, 3).unwrap();

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
        let (hydros, energy_conversion, override_table, stage_to_season) =
            make_setup_inputs(n_stages);

        let params = vec![make_param(
            0,
            ParameterKind::Computed(ComputedParameter::EquivalentProductivity {
                hydro_id: EntityId(0),
            }),
        )];

        let table = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
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

        let params = vec![make_param(
            0,
            ParameterKind::Computed(ComputedParameter::SpecificProductivity {
                hydro_id: EntityId(0),
            }),
        )];

        let result = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
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
        let (hydros, energy_conversion, override_table, stage_to_season) =
            make_setup_inputs(n_stages);

        let param_a = make_param(10, ParameterKind::Constant(1.0));
        let param_b = make_param(20, ParameterKind::PerStage(vec![2.0, 3.0, 4.0]));
        let param_c = make_param(
            30,
            ParameterKind::Computed(ComputedParameter::AccumulatedProductivity {
                hydro_id: EntityId(0),
            }),
        );

        let params_abc = vec![param_a.clone(), param_b.clone(), param_c.clone()];
        let params_cab = vec![param_c.clone(), param_a.clone(), param_b.clone()];

        let table_abc = build_resolved_parameters(
            &params_abc,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            n_stages,
        )
        .unwrap();
        let table_cab = build_resolved_parameters(
            &params_cab,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
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
        let (hydros, energy_conversion, override_table, stage_to_season) =
            make_setup_inputs(n_stages);

        // Use a hydro_id that does not exist in the hydros slice (ids 0 and 1 exist).
        let params = vec![make_param(
            0,
            ParameterKind::Computed(ComputedParameter::MinStorage {
                hydro_id: EntityId(99),
            }),
        )];

        let result = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
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
        let (hydros, energy_conversion, override_table, stage_to_season) =
            make_setup_inputs(n_stages);

        let params = vec![
            make_param(
                0,
                ParameterKind::Computed(ComputedParameter::MinStorage {
                    hydro_id: EntityId(0),
                }),
            ),
            make_param(
                1,
                ParameterKind::Computed(ComputedParameter::MaxStorage {
                    hydro_id: EntityId(1),
                }),
            ),
        ];

        let table = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
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

        let table = build_resolved_parameters(&[], &ec, &overrides, &[], &[], 0).unwrap();
        // Nothing to query — just verify it doesn't panic.
        let _ = table;
    }
}
