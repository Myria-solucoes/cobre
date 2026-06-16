//! Export-row builder for the resolved evaporation models.
//!
//! Converts the resolved [`EvaporationModelSet`](super::types::EvaporationModelSet)
//! plus its reference-volume provenance into a flat `Vec<cobre_io::EvaporationModelRow>`
//! that the CLI and Python write sites persist to `evaporation_models.parquet`.
//! The summary builder counts the same models for display; this builder emits the
//! per-`(hydro, stage)` coefficient rows that capture the actual numbers.

use cobre_core::System;
use cobre_io::EvaporationModelRow;

use super::types::{EvaporationModel, EvaporationReferenceSource, PrepareHydroModelsResult};

/// Provenance tag string for a [`EvaporationReferenceSource`].
///
/// The two literals match the `source` column values the
/// [`cobre_io::extensions::parse_evaporation_models`] reader round-trips; the
/// writer carries the string verbatim. Kept as a private mapping so the two
/// values are defined once and stay in lockstep with the enum variants.
fn reference_source_tag(source: EvaporationReferenceSource) -> &'static str {
    match source {
        EvaporationReferenceSource::UserSupplied => "user_supplied",
        EvaporationReferenceSource::DefaultMidpoint => "default_midpoint",
    }
}

/// Build the flat evaporation-model export rows from the pipeline result.
///
/// Iterates `system.hydros()` in canonical (declaration) order so the emitted
/// stream is declaration-order bit-deterministic: hydro position `h` indexes
/// both `result.evaporation.model(h)` and
/// `result.provenance.evaporation_reference_sources[h]`, which the resolver
/// builds in lockstep with `system.hydros()`.
///
/// Row shape per hydro:
///
/// - [`EvaporationModel::None`] hydros contribute no row.
/// - A [`EvaporationModel::Linearized`] with a single stage entry emits one row
///   with `stage_id: None` — the coefficient applies to every stage.
/// - A [`EvaporationModel::Linearized`] with multiple stage entries emits one
///   row per stage, `stage_id: Some(stage.id)`, each carrying that stage's
///   `intercept_m3s`, `volume_slope_m3s_per_hm3`, and `reference_volume_hm3`
///   from `reference_volumes_hm3[stage]`.
///
/// The per-stage `coefficients`/`reference_volumes_hm3` vectors are indexed by
/// study-stage position; `study_stages[i].id` supplies the domain-level
/// `stage_id` for index `i`. Study stages are iterated in canonical order
/// (the `id >= 0` filter on `system.stages()`), so within each hydro the rows
/// ascend by `stage_id` and the full stream is in `(hydro_id, stage_id)` order
/// by construction.
#[must_use]
pub fn build_evaporation_model_rows(
    result: &PrepareHydroModelsResult,
    system: &System,
) -> Vec<EvaporationModelRow> {
    // Study stages in canonical order; index `i` maps a coefficient slot to its
    // domain-level `stage.id`. Pre-study stages (`id < 0`) carry no evaporation
    // coefficients, so they are excluded to keep `study_stages[i]` aligned with
    // the per-stage `coefficients`/`reference_volumes_hm3` vectors.
    let study_stage_ids: Vec<i32> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect();

    let mut rows: Vec<EvaporationModelRow> = Vec::new();

    for (hydro_pos, hydro) in system.hydros().iter().enumerate() {
        let EvaporationModel::Linearized {
            coefficients,
            reference_volumes_hm3,
        } = result.evaporation.model(hydro_pos)
        else {
            continue;
        };

        let source =
            reference_source_tag(result.provenance.evaporation_reference_sources[hydro_pos].1)
                .to_string();

        if coefficients.len() == 1 {
            // Single coefficient: one row covering all stages (`stage_id: None`).
            rows.push(EvaporationModelRow {
                hydro_id: hydro.id,
                stage_id: None,
                intercept_m3s: coefficients[0].intercept_m3s,
                volume_slope_m3s_per_hm3: coefficients[0].volume_slope_m3s_per_hm3,
                reference_volume_hm3: reference_volumes_hm3[0],
                source: source.clone(),
            });
        } else {
            // Per-stage coefficients: one row per study stage, tagged with the
            // domain-level `stage.id` at the matching coefficient position.
            for (i, coeff) in coefficients.iter().enumerate() {
                rows.push(EvaporationModelRow {
                    hydro_id: hydro.id,
                    stage_id: Some(study_stage_ids[i]),
                    intercept_m3s: coeff.intercept_m3s,
                    volume_slope_m3s_per_hm3: coeff.volume_slope_m3s_per_hm3,
                    reference_volume_hm3: reference_volumes_hm3[i],
                    source: source.clone(),
                });
            }
        }
    }

    rows
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::{
        Bus, DeficitSegment, EntityId, SystemBuilder,
        entities::hydro::{HydroGenerationModel, HydroPenalties},
        scenario::CorrelationModel,
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };

    use crate::production::hydro_models::*;

    fn zero_penalties() -> HydroPenalties {
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

    fn make_hydro(id: i32) -> cobre_core::entities::hydro::Hydro {
        cobre_core::entities::hydro::Hydro {
            id: EntityId::from(id),
            name: format!("Hydro{id}"),
            bus_id: EntityId::from(10),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 100.0,
            max_storage_hm3: 2000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 500.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 1000.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_penalties(),
        }
    }

    fn make_stage(id: i32, season_id: usize) -> Stage {
        Stage {
            index: id.max(0) as usize,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap_or_default(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap_or_default(),
            season_id: Some(season_id),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 3,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn make_system(
        hydros: Vec<cobre_core::entities::hydro::Hydro>,
        stages: Vec<Stage>,
    ) -> cobre_core::System {
        let bus = Bus {
            id: EntityId(10),
            name: "B10".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(hydros)
            .stages(stages)
            .correlation(CorrelationModel::default())
            .build()
            .unwrap()
    }

    /// Build a `PrepareHydroModelsResult` carrying the supplied per-hydro
    /// evaporation models, with constant production and matching provenance.
    fn make_result(
        hydro_ids: &[i32],
        evap_models: Vec<EvaporationModel>,
        ref_sources: Vec<EvaporationReferenceSource>,
        n_stages: usize,
    ) -> PrepareHydroModelsResult {
        let n_hydros = hydro_ids.len();
        let models: Vec<Vec<ResolvedProductionModel>> = (0..n_hydros)
            .map(|_| {
                (0..n_stages)
                    .map(|_| ResolvedProductionModel::ConstantProductivity { productivity: 0.95 })
                    .collect()
            })
            .collect();
        let production = ProductionModelSet::new(models, n_hydros, n_stages);
        let production_sources = hydro_ids
            .iter()
            .map(|&id| (EntityId(id), ProductionModelSource::DefaultConstant))
            .collect();
        let evaporation_sources = hydro_ids
            .iter()
            .zip(&evap_models)
            .map(|(&id, m)| {
                let src = match m {
                    EvaporationModel::None => EvaporationSource::NotModeled,
                    EvaporationModel::Linearized { .. } => {
                        EvaporationSource::LinearizedFromGeometry
                    }
                };
                (EntityId(id), src)
            })
            .collect();
        let evaporation_reference_sources = hydro_ids
            .iter()
            .zip(ref_sources)
            .map(|(&id, src)| (EntityId(id), src))
            .collect();
        PrepareHydroModelsResult {
            production,
            productivity_override:
                crate::energy_conversion::HydroEnergyProductivityOverride::default(),
            evaporation: EvaporationModelSet::new(evap_models),
            provenance: HydroModelProvenance {
                production_sources,
                evaporation_sources,
                evaporation_reference_sources,
            },
            fpha_export_rows: Vec::new(),
            reference_volumes_hm3: Vec::new(),
            vha_geometry_by_hydro: std::collections::HashMap::new(),
        }
    }

    /// A mixed set produces rows only for `Linearized` hydros, in canonical
    /// `(hydro_id, stage_id)` order, with the matching source tag.
    #[test]
    fn build_evaporation_model_rows_skips_none() {
        // Hydros 1, 2, 3: only 2 (UserSupplied) and 3 (DefaultMidpoint) have
        // single-stage Linearized models; hydro 1 has None.
        let hydro_ids = [1i32, 2, 3];
        let hydros = hydro_ids.iter().map(|&id| make_hydro(id)).collect();
        let system = make_system(hydros, vec![make_stage(0, 0)]);

        let evap_models = vec![
            EvaporationModel::None,
            EvaporationModel::Linearized {
                coefficients: vec![LinearizedEvaporation {
                    intercept_m3s: 2.0,
                    volume_slope_m3s_per_hm3: 0.02,
                }],
                reference_volumes_hm3: vec![250.0],
            },
            EvaporationModel::Linearized {
                coefficients: vec![LinearizedEvaporation {
                    intercept_m3s: 3.0,
                    volume_slope_m3s_per_hm3: 0.03,
                }],
                reference_volumes_hm3: vec![350.0],
            },
        ];
        let ref_sources = vec![
            EvaporationReferenceSource::DefaultMidpoint,
            EvaporationReferenceSource::UserSupplied,
            EvaporationReferenceSource::DefaultMidpoint,
        ];
        let result = make_result(&hydro_ids, evap_models, ref_sources, 1);

        let rows = build_evaporation_model_rows(&result, &system);

        assert_eq!(rows.len(), 2, "only the two Linearized hydros produce rows");

        // Row 0: hydro 2, single-stage → stage_id None, user_supplied.
        assert_eq!(rows[0].hydro_id, EntityId(2));
        assert_eq!(rows[0].stage_id, None);
        assert_eq!(rows[0].intercept_m3s, 2.0);
        assert_eq!(rows[0].volume_slope_m3s_per_hm3, 0.02);
        assert_eq!(rows[0].reference_volume_hm3, 250.0);
        assert_eq!(rows[0].source, "user_supplied");

        // Row 1: hydro 3, single-stage → stage_id None, default_midpoint.
        assert_eq!(rows[1].hydro_id, EntityId(3));
        assert_eq!(rows[1].stage_id, None);
        assert_eq!(rows[1].intercept_m3s, 3.0);
        assert_eq!(rows[1].source, "default_midpoint");

        // Canonical order: hydro_id ascends.
        assert!(rows[0].hydro_id.0 < rows[1].hydro_id.0);
    }

    /// A multi-stage `Linearized` model yields one row per stage with the right
    /// `stage_id` and per-stage coefficients.
    #[test]
    fn build_evaporation_model_rows_per_stage() {
        // Single hydro id=7 with a 3-stage Linearized model; stages 0, 1, 2.
        let hydro_ids = [7i32];
        let hydros = hydro_ids.iter().map(|&id| make_hydro(id)).collect();
        let stages = vec![make_stage(0, 0), make_stage(1, 1), make_stage(2, 2)];
        let system = make_system(hydros, stages);

        let evap_models = vec![EvaporationModel::Linearized {
            coefficients: vec![
                LinearizedEvaporation {
                    intercept_m3s: 1.0,
                    volume_slope_m3s_per_hm3: 0.01,
                },
                LinearizedEvaporation {
                    intercept_m3s: 2.0,
                    volume_slope_m3s_per_hm3: 0.02,
                },
                LinearizedEvaporation {
                    intercept_m3s: 3.0,
                    volume_slope_m3s_per_hm3: 0.03,
                },
            ],
            reference_volumes_hm3: vec![100.0, 200.0, 300.0],
        }];
        let ref_sources = vec![EvaporationReferenceSource::UserSupplied];
        let result = make_result(&hydro_ids, evap_models, ref_sources, 3);

        let rows = build_evaporation_model_rows(&result, &system);

        assert_eq!(rows.len(), 3, "one row per study stage");
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row.hydro_id, EntityId(7));
            assert_eq!(
                row.stage_id,
                Some(i as i32),
                "stage_id must be the domain-level stage id at position {i}"
            );
            assert_eq!(row.intercept_m3s, (i + 1) as f64);
            assert_eq!(row.volume_slope_m3s_per_hm3, 0.01 * (i + 1) as f64);
            assert_eq!(row.reference_volume_hm3, 100.0 * (i + 1) as f64);
            assert_eq!(row.source, "user_supplied");
        }

        // Rows ascend by stage_id (canonical within the hydro).
        assert!(
            rows.windows(2).all(|w| w[0].stage_id < w[1].stage_id),
            "rows must ascend by stage_id"
        );
    }
}
