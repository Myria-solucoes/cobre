//! Energy-conversion builder: derives the [`EnergyConversionSet`] for the case.

use std::collections::HashMap;

use cobre_core::{CascadeTopology, EntityId, Hydro, HydroGenerationModel};
use cobre_io::{HydroGeometryRow, HydroReferenceVolumeFractions};

use super::productivity_override::HydroEnergyProductivityOverride;
use super::types::{EnergyConversion, EnergyConversionError, EnergyConversionSet};
use crate::fpha_fitting::{ForebayTable, evaluate_losses, evaluate_tailrace};

/// Build the [`EnergyConversionSet`] for the case.
///
/// For FPHA hydros, `ρ_eq` is resolved from three sources in priority order:
///
/// 1. `override_table` — a per-`(hydro, stage)` user-supplied value from the
///    optional `system/hydro_energy_productivity.parquet` rows.
/// 2. VHA geometry + `ρ_esp` — derived via `ρ_esp · h_eq(V_ref, Q_ref)`.
/// 3. Neither — returns [`EnergyConversionError::FphaMissingEquivalentProductivity`].
///
/// For non-FPHA hydros, `ρ_eq` is read from `production_models`; passing `None`
/// yields `0.0` (only appropriate in tests that do not require accurate `ρ_eq`).
///
/// # Errors
///
/// - [`EnergyConversionError::InvalidStorageRange`] — `max_storage_hm3 < min_storage_hm3`.
/// - [`EnergyConversionError::NegativeMaxTurbined`] — `max_turbined_m3s < 0`.
/// - [`EnergyConversionError::ForebayTableInvalid`] — VHA rows fail forebay-table validation.
/// - [`EnergyConversionError::NonPositiveEquivalentHead`] — derived `h_eq ≤ 0`.
/// - [`EnergyConversionError::FphaMissingEquivalentProductivity`] — FPHA hydro has no
///   usable `ρ_eq` source for a given stage and no override entry.
/// - [`EnergyConversionError::CascadeIndexMismatch`] — cascade built from different hydro set.
/// - [`EnergyConversionError::DanglingDownstream`] — dangling downstream reference.
#[allow(clippy::missing_errors_doc)]
pub fn build_energy_conversion_set<S: std::hash::BuildHasher>(
    hydros: &[Hydro],
    n_stages: usize,
    cascade: &CascadeTopology,
    reference_volume_fractions: &HydroReferenceVolumeFractions,
    vha_rows_by_hydro: &HashMap<EntityId, Vec<HydroGeometryRow>, S>,
    override_table: Option<&HydroEnergyProductivityOverride>,
    production_models: Option<&crate::hydro_models::ProductionModelSet>,
) -> Result<EnergyConversionSet, EnergyConversionError> {
    let n_hydros = hydros.len();

    let mut per_hydro_stage: Vec<Vec<EnergyConversion>> = Vec::with_capacity(n_hydros);

    for (h_idx, hydro) in hydros.iter().enumerate() {
        let v_min = hydro.min_storage_hm3;
        let v_max = hydro.max_storage_hm3;
        let q_max = hydro.max_turbined_m3s;

        if v_max < v_min {
            return Err(EnergyConversionError::InvalidStorageRange {
                hydro_id: hydro.id,
                v_min,
                v_max,
            });
        }
        if q_max < 0.0 {
            return Err(EnergyConversionError::NegativeMaxTurbined {
                hydro_id: hydro.id,
                q_max,
            });
        }

        let fpha_derivation = if matches!(hydro.generation_model, HydroGenerationModel::Fpha) {
            match (
                vha_rows_by_hydro.get(&hydro.id),
                hydro.specific_productivity_mw_per_m3s_per_m,
            ) {
                (Some(rows), Some(rho_esp)) => {
                    let table = ForebayTable::new(rows, &hydro.name).map_err(|e| {
                        EnergyConversionError::ForebayTableInvalid {
                            hydro_id: hydro.id,
                            message: e.to_string(),
                        }
                    })?;
                    Some((table, rho_esp))
                }
                _ => None,
            }
        } else {
            None
        };

        let mut row: Vec<EnergyConversion> = Vec::with_capacity(n_stages);
        for stage in 0..n_stages {
            let reference_volume_hm3 = reference_volume_fractions.get(hydro.id, stage);

            let productivity = if matches!(hydro.generation_model, HydroGenerationModel::Fpha) {
                0.0
            } else {
                production_models.map_or(0.0, |pm| match pm.model(h_idx, stage) {
                    crate::hydro_models::ResolvedProductionModel::ConstantProductivity {
                        productivity,
                    } => *productivity,
                    crate::hydro_models::ResolvedProductionModel::Fpha { .. } => 0.0,
                })
            };
            let mut conversion =
                derive_conversion_for_hydro(hydro, reference_volume_hm3, productivity);

            if matches!(hydro.generation_model, HydroGenerationModel::Fpha) {
                // FPHA ρ_eq: parquet override wins over the VHA + ρ_esp derivation.
                let parquet_rho_eq =
                    override_table.and_then(|o| o.equivalent_productivity(hydro.id, stage));
                let rho_eq = if let Some(value) = parquet_rho_eq {
                    value
                } else if let Some((ref table, rho_esp)) = fpha_derivation {
                    let h_eq = fpha_equivalent_head(
                        hydro,
                        conversion.reference_volume_hm3,
                        conversion.reference_outflow_m3s,
                        table,
                    )?;
                    rho_esp * h_eq
                } else {
                    return Err(EnergyConversionError::FphaMissingEquivalentProductivity {
                        hydro_id: hydro.id,
                        hydro_name: hydro.name.clone(),
                        stage,
                    });
                };
                conversion.equivalent_productivity_mw_per_m3s = rho_eq;
            }

            row.push(conversion);
        }
        per_hydro_stage.push(row);
    }

    let topo_len = cascade.topological_order().len();
    if topo_len != n_hydros {
        return Err(EnergyConversionError::CascadeIndexMismatch {
            expected: n_hydros,
            got: topo_len,
        });
    }

    let mut id_to_index: HashMap<EntityId, usize> = HashMap::with_capacity(n_hydros);
    for (idx, h) in hydros.iter().enumerate() {
        id_to_index.insert(h.id, idx);
    }

    // Reverse topological order (downstream before upstream): each plant's
    // downstream ρ_acum is fully computed before it is summed in.
    let mut accumulated = vec![vec![0.0_f64; n_stages]; n_hydros];
    for t in 0..n_stages {
        for id in cascade.topological_order().iter().rev() {
            let h_idx =
                *id_to_index
                    .get(id)
                    .ok_or(EnergyConversionError::CascadeIndexMismatch {
                        expected: n_hydros,
                        got: topo_len,
                    })?;
            let rho_eq = per_hydro_stage[h_idx][t].equivalent_productivity_mw_per_m3s;
            let downstream_contrib = if let Some(ds_id) = cascade.downstream(*id) {
                let ds_idx =
                    *id_to_index
                        .get(&ds_id)
                        .ok_or(EnergyConversionError::DanglingDownstream {
                            hydro_id: *id,
                            downstream_id: ds_id,
                        })?;
                accumulated[ds_idx][t]
            } else {
                0.0
            };
            accumulated[h_idx][t] = rho_eq + downstream_contrib;
        }
    }

    Ok(EnergyConversionSet::new(
        per_hydro_stage,
        accumulated,
        n_hydros,
        n_stages,
    ))
}

/// Derive the per-`(hydro, stage)` [`EnergyConversion`] cell.
///
/// `reference_volume_hm3` (absolute hm³) is stored verbatim — the
/// `v_min + fraction·(..)` span is applied once at resolver construction, never
/// re-applied here. For FPHA hydros pass `productivity = 0.0`;
/// `build_energy_conversion_set` overwrites it via the FPHA derivation.
fn derive_conversion_for_hydro(
    hydro: &Hydro,
    reference_volume_hm3: f64,
    productivity: f64,
) -> EnergyConversion {
    let reference_outflow_m3s = hydro.max_turbined_m3s;
    EnergyConversion {
        equivalent_productivity_mw_per_m3s: productivity,
        reference_volume_hm3,
        reference_outflow_m3s,
    }
}

/// Equivalent head `h_eq = h_fore(V_ref) − h_tail(Q_ref) − h_loss`.
///
/// Returns `None` when `h_eq <= 0.0`; [`fpha_equivalent_head`] surfaces that case
/// as an error instead.
fn equivalent_head(hydro: &Hydro, table: &ForebayTable, v_ref: f64, q_ref: f64) -> Option<f64> {
    let h_fore = table.height(v_ref);
    let h_tail = hydro
        .tailrace
        .as_ref()
        .map_or(0.0, |t| evaluate_tailrace(t, q_ref));
    let h_loss = hydro
        .hydraulic_losses
        .as_ref()
        .map_or(0.0, |m| evaluate_losses(m, h_fore - h_tail, q_ref));
    let h_eq = h_fore - h_tail - h_loss;
    (h_eq > 0.0).then_some(h_eq)
}

/// FPHA equivalent head ([`equivalent_head`]), erroring on non-positive results.
///
/// Returns [`EnergyConversionError::NonPositiveEquivalentHead`] when `h_eq <= 0.0`
/// (which would yield a non-physical `ρ_eq`).
fn fpha_equivalent_head(
    hydro: &Hydro,
    v_ref: f64,
    q_ref: f64,
    table: &ForebayTable,
) -> Result<f64, EnergyConversionError> {
    if let Some(h_eq) = equivalent_head(hydro, table, v_ref, q_ref) {
        Ok(h_eq)
    } else {
        let h_fore = table.height(v_ref);
        let h_tail = hydro
            .tailrace
            .as_ref()
            .map_or(0.0, |t| evaluate_tailrace(t, q_ref));
        let h_loss = hydro
            .hydraulic_losses
            .as_ref()
            .map_or(0.0, |m| evaluate_losses(m, h_fore - h_tail, q_ref));
        Err(EnergyConversionError::NonPositiveEquivalentHead {
            hydro_id: hydro.id,
            h_eq: h_fore - h_tail - h_loss,
        })
    }
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::unwrap_used
)]
mod tests {
    use cobre_core::{
        CascadeTopology, EntityId, HydraulicLossesModel, Hydro, HydroGenerationModel,
        HydroPenalties,
    };
    use cobre_io::{
        HydroEnergyProductivityRow, HydroGeometryRow, HydroReferenceVolumeFractions,
        build_hydro_reference_volumes_resolved,
    };

    use super::super::productivity_override::build_hydro_energy_productivity_override;
    use super::super::types::EnergyConversionError;
    use super::*;
    use crate::hydro_models::{ProductionModelSet, ResolvedProductionModel};

    fn penalties_zero() -> HydroPenalties {
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

    fn make_hydro(id: i32, downstream: Option<i32>) -> Hydro {
        Hydro {
            id: EntityId::from(id),
            name: format!("Hydro {id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(1),
            downstream_id: downstream.map(EntityId::from),
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 45.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: penalties_zero(),
        }
    }

    /// Resolver returning, for every `(hydro, stage)`, the absolute hm³ for the
    /// 0.65 default fraction resolved against each plant's band — the value the
    /// production pipeline feeds after wiring the JSON-sourced reference volume.
    fn make_resolver(hydros: &[Hydro]) -> HydroReferenceVolumeFractions {
        resolved_resolver(hydros, 0.65, 2)
    }

    /// Build a resolver whose `get(hydro, stage)` returns the absolute hm³ for
    /// `fraction` resolved against each plant's `[v_min, v_max]` band — the
    /// resolved-hm³ semantics the consumers read after this wiring.
    fn resolved_resolver(
        hydros: &[Hydro],
        fraction: f64,
        n_stages: usize,
    ) -> HydroReferenceVolumeFractions {
        let resolved: Vec<(EntityId, usize, f64)> = hydros
            .iter()
            .flat_map(|h| {
                let v = h.min_storage_hm3 + fraction * (h.max_storage_hm3 - h.min_storage_hm3);
                (0..n_stages).map(move |s| (h.id, s, v))
            })
            .collect();
        build_hydro_reference_volumes_resolved(&resolved, 0.0)
    }

    #[test]
    fn builder_returns_grid_with_expected_dimensions() {
        // hydro id=1 (downstream=2) and hydro id=2 (terminal), both ρ_eq=1.0.
        // After the cascade walk:
        //   ρ_acum(id=2) = 1.0            (terminal, no downstream contrib)
        //   ρ_acum(id=1) = 1.0 + 1.0 = 2.0
        let n_stages = 2;
        let hydros = vec![make_hydro(1, Some(2)), make_hydro(2, None)];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = make_resolver(&hydros);
        let pm = production_set(&[1.0, 1.0], n_stages);

        let set = build_energy_conversion_set(
            &hydros,
            n_stages,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        assert_eq!(set.n_hydros(), 2);
        assert_eq!(set.n_stages(), n_stages);
        // hydro index 0 = id=1 (upstream), index 1 = id=2 (terminal).
        for s in 0..n_stages {
            assert_eq!(set.accumulated_productivity(1, s), 1.0); // id=2, terminal
            assert_eq!(set.accumulated_productivity(0, s), 2.0); // id=1, upstream of id=2
        }
    }

    fn make_hydro_with(
        id: i32,
        model: HydroGenerationModel,
        v_min: f64,
        v_max: f64,
        q_max: f64,
        specific: Option<f64>,
    ) -> Hydro {
        let mut h = make_hydro(id, None);
        h.generation_model = model;
        h.min_storage_hm3 = v_min;
        h.max_storage_hm3 = v_max;
        h.max_turbined_m3s = q_max;
        h.specific_productivity_mw_per_m3s_per_m = specific;
        h
    }

    fn constant_resolver(
        hydros: &[Hydro],
        fraction: f64,
        n_stages: usize,
    ) -> HydroReferenceVolumeFractions {
        resolved_resolver(hydros, fraction, n_stages)
    }

    /// Build a `ProductionModelSet` where every (hydro, stage) cell uses
    /// `ConstantProductivity` with the given per-hydro productivity values.
    ///
    /// `productivities[h]` is the productivity for hydro at declaration index `h`.
    fn production_set(productivities: &[f64], n_stages: usize) -> ProductionModelSet {
        let n_hydros = productivities.len();
        let models = productivities
            .iter()
            .map(|&p| {
                vec![ResolvedProductionModel::ConstantProductivity { productivity: p }; n_stages]
            })
            .collect();
        ProductionModelSet::new(models, n_hydros, n_stages)
    }

    #[test]
    fn constant_productivity_yields_input_scalar() {
        let n_stages = 2;
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity,
            100.0,
            200.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, n_stages);
        let pm = production_set(&[0.9], n_stages);

        let set = build_energy_conversion_set(
            &hydros,
            n_stages,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        let c = set.conversion(0, 0);
        assert_eq!(c.equivalent_productivity_mw_per_m3s, 0.9);
        assert_eq!(c.reference_volume_hm3, 100.0 + 0.65 * (200.0 - 100.0));
        assert_eq!(c.reference_outflow_m3s, 50.0);
    }

    #[test]
    fn linearized_head_yields_input_scalar() {
        let n_stages = 1;
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::LinearizedHead,
            100.0,
            200.0,
            40.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.5, n_stages);
        let pm = production_set(&[1.2], n_stages);

        let set = build_energy_conversion_set(
            &hydros,
            n_stages,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        assert_eq!(set.conversion(0, 0).equivalent_productivity_mw_per_m3s, 1.2);
    }

    #[test]
    fn reference_volume_uses_fraction() {
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity,
            100.0,
            200.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);

        for f in [0.1_f64, 0.5, 1.0] {
            let resolver = constant_resolver(&hydros, f, 1);
            let set = build_energy_conversion_set(
                &hydros,
                1,
                &cascade,
                &resolver,
                &HashMap::new(),
                None,
                None,
            )
            .expect("builder succeeds");
            let expected = 100.0 + f * (200.0 - 100.0);
            assert!(
                (set.conversion(0, 0).reference_volume_hm3 - expected).abs() < f64::EPSILON,
                "fraction {f}: V_ref expected {expected}, got {}",
                set.conversion(0, 0).reference_volume_hm3
            );
        }
    }

    #[test]
    fn per_season_override_produces_different_v_ref_per_stage() {
        let n_stages = 4;
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity,
            100.0,
            200.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        // 4 stages alternating season 0 (fraction 0.50 → 150 hm³) and season 1
        // (fraction 0.70 → 170 hm³) on the [100, 200] band, resolved per stage.
        let id = hydros[0].id;
        let per_stage_hm3 = vec![
            (id, 0, 150.0),
            (id, 1, 170.0),
            (id, 2, 150.0),
            (id, 3, 170.0),
        ];
        let resolver = build_hydro_reference_volumes_resolved(&per_stage_hm3, 0.0);
        let pm = production_set(&[0.9], n_stages);
        let set = build_energy_conversion_set(
            &hydros,
            n_stages,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        let expected = [150.0_f64, 170.0, 150.0, 170.0];
        for (s, want) in expected.iter().enumerate() {
            assert!(
                (set.conversion(0, s).reference_volume_hm3 - want).abs() < f64::EPSILON,
                "stage {s}: expected V_ref {want}, got {}",
                set.conversion(0, s).reference_volume_hm3
            );
            assert_eq!(set.conversion(0, s).equivalent_productivity_mw_per_m3s, 0.9);
        }
    }

    #[test]
    fn invalid_storage_range_is_rejected() {
        let hydros = vec![make_hydro_with(
            42,
            HydroGenerationModel::ConstantProductivity,
            200.0,
            100.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);

        let err = build_energy_conversion_set(
            &hydros,
            1,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::InvalidStorageRange { hydro_id, .. } => {
                assert_eq!(hydro_id, hydros[0].id);
            }
            other => panic!("expected InvalidStorageRange, got: {other:?}"),
        }
    }

    #[test]
    fn negative_max_turbined_is_rejected() {
        let hydros = vec![make_hydro_with(
            42,
            HydroGenerationModel::ConstantProductivity,
            100.0,
            200.0,
            -1.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);

        let err = build_energy_conversion_set(
            &hydros,
            1,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::NegativeMaxTurbined { hydro_id, q_max } => {
                assert_eq!(hydro_id, hydros[0].id);
                assert_eq!(q_max, -1.0);
            }
            other => panic!("expected NegativeMaxTurbined, got: {other:?}"),
        }
    }

    /// An FPHA hydro that supplies ρ_esp but no VHA geometry is rejected with
    /// `FphaMissingEquivalentProductivity` (a permissive fallback to ρ_eq=0.0
    /// would silently mis-model the plant).
    #[test]
    fn fpha_hydro_missing_vha_is_rejected_with_actionable_error() {
        let hydros = vec![make_hydro_with(
            5,
            HydroGenerationModel::Fpha,
            100.0,
            200.0,
            50.0,
            Some(0.01),
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.5, 1);
        let err = build_energy_conversion_set(
            &hydros,
            1,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::FphaMissingEquivalentProductivity {
                hydro_id,
                ref hydro_name,
                stage,
            } => {
                assert_eq!(hydro_id, hydros[0].id);
                assert!(
                    hydro_name.contains("Hydro 5"),
                    "error should mention the hydro name, got: {hydro_name}"
                );
                assert_eq!(stage, 0);
            }
            other => panic!("expected FphaMissingEquivalentProductivity, got: {other:?}"),
        }
    }

    // ── FPHA derivation tests ──────────────────────────────────────────────

    /// Create a flat VHA table (constant height) for testing.
    fn vha_constant_height(hydro_id: EntityId, height: f64) -> (EntityId, Vec<HydroGeometryRow>) {
        (
            hydro_id,
            vec![
                HydroGeometryRow {
                    hydro_id,
                    volume_hm3: 0.0,
                    height_m: height,
                    area_km2: 1.0,
                },
                HydroGeometryRow {
                    hydro_id,
                    volume_hm3: 1000.0,
                    height_m: height,
                    area_km2: 1.0,
                },
            ],
        )
    }

    fn fpha_hydro_for_tests(id: i32) -> Hydro {
        make_hydro_with(
            id,
            HydroGenerationModel::Fpha,
            100.0,
            200.0,
            50.0,
            Some(0.0090),
        )
    }

    #[test]
    fn fpha_rho_eq_from_vha_no_tailrace_no_losses() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let (id, rows) = vha_constant_height(hydros[0].id, 400.0);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let set = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &map, None, None)
            .expect("builder succeeds");

        let got = set.conversion(0, 0).equivalent_productivity_mw_per_m3s;
        let expected = 0.0090 * 400.0;
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn fpha_rho_eq_with_factor_losses() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.tailrace = None;
        hydro.hydraulic_losses = Some(HydraulicLossesModel::Factor { value: 0.05 });
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let (id, rows) = vha_constant_height(hydros[0].id, 400.0);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let set = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &map, None, None)
            .expect("builder succeeds");

        // h_eq = 400 - 0 - 0.05 * 400 = 380; rho_eq = 0.009 * 380 = 3.42
        let got = set.conversion(0, 0).equivalent_productivity_mw_per_m3s;
        let expected = 0.0090 * 380.0;
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn fpha_rho_eq_with_constant_losses() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.tailrace = None;
        hydro.hydraulic_losses = Some(HydraulicLossesModel::Constant { value_m: 5.0 });
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let (id, rows) = vha_constant_height(hydros[0].id, 400.0);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let set = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &map, None, None)
            .expect("builder succeeds");

        // h_eq = 400 - 0 - 5 = 395; rho_eq = 0.009 * 395 = 3.555
        let got = set.conversion(0, 0).equivalent_productivity_mw_per_m3s;
        let expected = 0.0090 * 395.0;
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }

    /// An FPHA hydro that supplies VHA geometry but no ρ_esp is rejected (a
    /// permissive fallback to ρ_eq=0.0 would silently mis-model the plant).
    #[test]
    fn fpha_missing_rho_esp_is_rejected() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.specific_productivity_mw_per_m3s_per_m = None;
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let (id, rows) = vha_constant_height(hydros[0].id, 400.0);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let err = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &map, None, None)
            .unwrap_err();
        match err {
            EnergyConversionError::FphaMissingEquivalentProductivity {
                hydro_id,
                ref hydro_name,
                stage: _,
            } => {
                assert_eq!(hydro_id, hydros[0].id);
                assert!(
                    hydro_name.contains("Hydro 7"),
                    "error should mention hydro name, got: {hydro_name}"
                );
            }
            other => panic!("expected FphaMissingEquivalentProductivity, got: {other:?}"),
        }
    }

    /// An FPHA hydro that supplies ρ_esp but no VHA geometry is rejected (a
    /// permissive fallback to ρ_eq=0.0 would silently mis-model the plant).
    #[test]
    fn fpha_missing_vha_is_rejected() {
        let hydro = fpha_hydro_for_tests(7); // has rho_esp = Some(0.009)
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);

        let err = build_energy_conversion_set(
            &hydros,
            1,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::FphaMissingEquivalentProductivity {
                hydro_id,
                ref hydro_name,
                stage: _,
            } => {
                assert_eq!(hydro_id, hydros[0].id);
                assert!(
                    hydro_name.contains("Hydro 7"),
                    "error should mention hydro name, got: {hydro_name}"
                );
            }
            other => panic!("expected FphaMissingEquivalentProductivity, got: {other:?}"),
        }
    }

    #[test]
    fn fpha_rejects_non_positive_h_eq() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.tailrace = None;
        // Constant loss equal to h_fore -> h_eq = 0 -> non-positive -> Err.
        hydro.hydraulic_losses = Some(HydraulicLossesModel::Constant { value_m: 400.0 });
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let (id, rows) = vha_constant_height(hydros[0].id, 400.0);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let err = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &map, None, None)
            .unwrap_err();
        match err {
            EnergyConversionError::NonPositiveEquivalentHead { hydro_id, h_eq } => {
                assert_eq!(hydro_id, hydros[0].id);
                assert!(h_eq <= 0.0);
            }
            other => panic!("expected NonPositiveEquivalentHead, got: {other:?}"),
        }
    }

    #[test]
    fn fpha_propagates_forebay_table_error() {
        // An EMPTY VHA fails ForebayTable::new (InsufficientPoints). A single row
        // is now valid (constant run-of-river forebay), so only the zero-row case
        // still propagates as ForebayTableInvalid.
        let hydro = fpha_hydro_for_tests(7);
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let mut map = HashMap::new();
        map.insert(hydros[0].id, Vec::<HydroGeometryRow>::new());

        let err = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &map, None, None)
            .unwrap_err();
        match err {
            EnergyConversionError::ForebayTableInvalid { hydro_id, .. } => {
                assert_eq!(hydro_id, hydros[0].id);
            }
            other => panic!("expected ForebayTableInvalid, got: {other:?}"),
        }
    }

    // ── cascade accumulator tests ─────────────────────────────────────────────

    /// A->B->C linear cascade. ρ_eq values: A=2.0, B=3.0, C=5.0 at stage 0.
    /// Expected ρ_acum: C=5.0, B=3+5=8.0, A=2+8=10.0.
    #[test]
    fn linear_cascade_accumulates_three_levels() {
        let mut a = make_hydro(0, Some(1));
        a.generation_model = HydroGenerationModel::ConstantProductivity;
        let mut b = make_hydro(1, Some(2));
        b.generation_model = HydroGenerationModel::ConstantProductivity;
        let mut c = make_hydro(2, None);
        c.generation_model = HydroGenerationModel::ConstantProductivity;
        let hydros = vec![a, b, c];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        // Declaration order: A(idx=0)=2.0, B(idx=1)=3.0, C(idx=2)=5.0.
        let pm = production_set(&[2.0, 3.0, 5.0], 1);

        let set = build_energy_conversion_set(
            &hydros,
            1,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        // Index by declaration order: A=0, B=1, C=2.
        assert_eq!(set.accumulated_productivity(0, 0), 10.0); // A
        assert_eq!(set.accumulated_productivity(1, 0), 8.0); // B
        assert_eq!(set.accumulated_productivity(2, 0), 5.0); // C
    }

    /// A->C and B->C branching cascade. ρ_eq: A=1.0, B=2.0, C=4.0.
    /// ρ_acum: C=4.0, A=1+4=5.0, B=2+4=6.0.
    #[test]
    fn branching_cascade_accumulates_correctly() {
        let mut a = make_hydro(0, Some(2));
        a.generation_model = HydroGenerationModel::ConstantProductivity;
        let mut b = make_hydro(1, Some(2));
        b.generation_model = HydroGenerationModel::ConstantProductivity;
        let mut c = make_hydro(2, None);
        c.generation_model = HydroGenerationModel::ConstantProductivity;
        let hydros = vec![a, b, c];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        // Declaration order: A(idx=0)=1.0, B(idx=1)=2.0, C(idx=2)=4.0.
        let pm = production_set(&[1.0, 2.0, 4.0], 1);

        let set = build_energy_conversion_set(
            &hydros,
            1,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        // A(idx=0), B(idx=1), C(idx=2).
        assert_eq!(set.accumulated_productivity(2, 0), 4.0); // C terminal
        assert_eq!(set.accumulated_productivity(0, 0), 5.0); // A = 1 + 4
        assert_eq!(set.accumulated_productivity(1, 0), 6.0); // B = 2 + 4
    }

    /// Build the same A->B->C cascade with two different declaration orders and
    /// confirm that ρ_acum indexed by EntityId is bit-for-bit identical.
    #[test]
    fn declaration_order_invariance() {
        let make_linear = |order: &[i32]| {
            let downstream = |id: i32| match id {
                0 => Some(1),
                1 => Some(2),
                _ => None,
            };
            let hydros: Vec<Hydro> = order
                .iter()
                .map(|&id| {
                    let mut h = make_hydro(id, downstream(id));
                    h.generation_model = HydroGenerationModel::ConstantProductivity;
                    h
                })
                .collect();
            hydros
        };

        let hydros_abc = make_linear(&[0, 1, 2]);
        let hydros_cab = make_linear(&[2, 0, 1]);

        let cascade_abc = CascadeTopology::build(&hydros_abc);
        let cascade_cab = CascadeTopology::build(&hydros_cab);

        let resolver_abc = constant_resolver(&hydros_abc, 0.65, 1);
        let resolver_cab = constant_resolver(&hydros_cab, 0.65, 1);

        let set_abc = build_energy_conversion_set(
            &hydros_abc,
            1,
            &cascade_abc,
            &resolver_abc,
            &HashMap::new(),
            None,
            None,
        )
        .expect("abc order");
        let set_cab = build_energy_conversion_set(
            &hydros_cab,
            1,
            &cascade_cab,
            &resolver_cab,
            &HashMap::new(),
            None,
            None,
        )
        .expect("cab order");

        let idx_abc: HashMap<i32, usize> = hydros_abc
            .iter()
            .enumerate()
            .map(|(i, h)| (h.id.0, i))
            .collect();
        let idx_cab: HashMap<i32, usize> = hydros_cab
            .iter()
            .enumerate()
            .map(|(i, h)| (h.id.0, i))
            .collect();

        for entity_id in [0_i32, 1, 2] {
            let val_abc = set_abc.accumulated_productivity(idx_abc[&entity_id], 0);
            let val_cab = set_cab.accumulated_productivity(idx_cab[&entity_id], 0);
            assert_eq!(
                val_abc.to_bits(),
                val_cab.to_bits(),
                "entity {entity_id}: abc={val_abc}, cab={val_cab}"
            );
        }
    }

    /// A hydro whose downstream_id points to an EntityId not in the hydros slice
    /// must return DanglingDownstream.
    #[test]
    fn dangling_downstream_is_rejected() {
        // Two hydros (id=0 downstream=99, id=1 terminal) so topo length stays 2
        // == hydros.len(); otherwise the shorter topo would fire
        // CascadeIndexMismatch before DanglingDownstream can.
        let mut h0 = make_hydro(0, Some(99));
        h0.generation_model = HydroGenerationModel::ConstantProductivity;
        let mut h1 = make_hydro(1, None);
        h1.generation_model = HydroGenerationModel::ConstantProductivity;
        let hydros = vec![h0, h1];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);

        let err = build_energy_conversion_set(
            &hydros,
            1,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::DanglingDownstream {
                hydro_id,
                downstream_id,
            } => {
                assert_eq!(hydro_id, EntityId::from(0));
                assert_eq!(downstream_id, EntityId::from(99));
            }
            other => panic!("expected DanglingDownstream, got: {other:?}"),
        }
    }

    /// When cascade.topological_order().len() != hydros.len(), return
    /// CascadeIndexMismatch.
    #[test]
    fn cascade_index_mismatch_is_rejected() {
        let h0 = make_hydro(0, Some(1));
        let h1 = make_hydro(1, None);
        let h2 = make_hydro(2, None);

        let short_cascade = CascadeTopology::build(&[h0.clone(), h1.clone()]);

        let hydros_three = vec![h0, h1, h2];
        let resolver = constant_resolver(&hydros_three, 0.65, 1);

        let err = build_energy_conversion_set(
            &hydros_three,
            1,
            &short_cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::CascadeIndexMismatch { expected, got } => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            other => panic!("expected CascadeIndexMismatch, got: {other:?}"),
        }
    }

    /// FPHA hydro with no VHA and no rho_esp but an override that returns a
    /// constant value for every stage succeeds, and the resolved
    /// equivalent_productivity equals the override value.
    #[test]
    fn fpha_with_override_only_succeeds() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.specific_productivity_mw_per_m3s_per_m = None;
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 3);
        let override_table =
            build_hydro_energy_productivity_override(&[HydroEnergyProductivityRow {
                hydro_id: hydros[0].id,
                stage_id: None,
                equivalent_productivity_mw_per_m3s: Some(2.5),
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            }])
            .expect("override builds");

        let set = build_energy_conversion_set(
            &hydros,
            3,
            &cascade,
            &resolver,
            &HashMap::new(),
            Some(&override_table),
            None,
        )
        .expect("builder succeeds with override only");

        for s in 0..3 {
            let cell = set.conversion(0, s);
            assert!(
                (cell.equivalent_productivity_mw_per_m3s - 2.5).abs() < 1e-12,
                "stage {s}: expected 2.5, got {}",
                cell.equivalent_productivity_mw_per_m3s
            );
        }
    }

    /// A ConstantProductivity hydro with no VHA, no rho_esp, and no override
    /// must succeed — the FPHA correctness gate must not apply to non-FPHA
    /// generation models. The productivity is supplied via `ProductionModelSet`;
    /// the test verifies the value is passed through to the conversion cell.
    #[test]
    fn constant_productivity_bypasses_gate() {
        let mut hydro = make_hydro(0, None);
        hydro.generation_model = HydroGenerationModel::ConstantProductivity;
        hydro.specific_productivity_mw_per_m3s_per_m = None;
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let pm = production_set(&[1.5], 1);

        let set = build_energy_conversion_set(
            &hydros,
            1,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("non-FPHA hydro succeeds despite missing FPHA inputs");

        let cell = set.conversion(0, 0);
        assert!((cell.equivalent_productivity_mw_per_m3s - 1.5).abs() < 1e-12);
    }

    /// For a non-FPHA hydro at stage 0 with both a JSON-resolved productivity
    /// (0.9) and a parquet override row supplying 0.85, the override wins.
    ///
    /// For a non-FPHA hydro, `build_energy_conversion_set` reads `ρ_eq` from
    /// `ProductionModelSet` as the single source of truth. Whether the value
    /// originated from JSON or from the parquet override is resolved upstream
    /// in `prepare_hydro_models`. This test confirms the value-flow contract:
    /// what `pm` says is what the conversion stores.
    #[test]
    fn test_non_fpha_reads_productivity_from_production_model_set() {
        let n_stages = 3;
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity,
            100.0,
            200.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, n_stages);
        let pm = production_set(&[0.85], n_stages);

        let set = build_energy_conversion_set(
            &hydros,
            n_stages,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        for s in 0..n_stages {
            let cell = set.conversion(0, s);
            assert!(
                (cell.equivalent_productivity_mw_per_m3s - 0.85).abs() < 1e-12,
                "stage {s}: expected 0.85 from pm, got {}",
                cell.equivalent_productivity_mw_per_m3s
            );
        }
    }

    /// The override table is consulted ONLY for FPHA hydros (where it
    /// replaces the VHA + `ρ_esp` derivation). For non-FPHA hydros the
    /// override has no effect at this layer because `ProductionModelSet`
    /// is already authoritative; the `prepare_hydro_models` pipeline is
    /// responsible for folding the override into pm.
    #[test]
    fn test_non_fpha_override_table_not_consulted_at_build_site() {
        let n_stages = 1;
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity,
            100.0,
            200.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, n_stages);
        // pm supplies 0.9 — that's the resolved value the LP and conversion
        // both see.
        let pm = production_set(&[0.9], n_stages);
        // Override table contains an inconsistent value that must NOT win at
        // this layer (such an inconsistency would be caught upstream by
        // `cobre_io::validation::productivity_resolution`).
        let override_table =
            build_hydro_energy_productivity_override(&[HydroEnergyProductivityRow {
                hydro_id: hydros[0].id,
                stage_id: Some(0),
                equivalent_productivity_mw_per_m3s: Some(0.42),
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            }])
            .expect("override builds");

        let set = build_energy_conversion_set(
            &hydros,
            n_stages,
            &cascade,
            &resolver,
            &HashMap::new(),
            Some(&override_table),
            Some(&pm),
        )
        .expect("builder succeeds");

        let cell = set.conversion(0, 0);
        assert!(
            (cell.equivalent_productivity_mw_per_m3s - 0.9).abs() < 1e-12,
            "non-FPHA path must read from pm, not from override; got {}",
            cell.equivalent_productivity_mw_per_m3s
        );
    }

    /// For a non-FPHA hydro with no parquet override and a JSON-resolved
    /// productivity of 0.9, every stage receives 0.9. Regression for the
    /// JSON-only path.
    #[test]
    fn test_non_fpha_json_only_path_unchanged() {
        let n_stages = 2;
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity,
            100.0,
            200.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, n_stages);
        let pm = production_set(&[0.9], n_stages);

        let set = build_energy_conversion_set(
            &hydros,
            n_stages,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        for s in 0..n_stages {
            let cell = set.conversion(0, s);
            assert!(
                (cell.equivalent_productivity_mw_per_m3s - 0.9).abs() < 1e-12,
                "stage {s}: expected 0.9 from JSON-only path, got {}",
                cell.equivalent_productivity_mw_per_m3s
            );
        }
    }
}
