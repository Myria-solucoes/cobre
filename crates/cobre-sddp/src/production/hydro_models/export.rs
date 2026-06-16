//! Export-row builder for the resolved evaporation models.
//!
//! Converts the resolved [`EvaporationModelSet`](super::types::EvaporationModelSet)
//! plus its reference-volume provenance into a flat `Vec<cobre_io::EvaporationModelRow>`
//! that the CLI and Python write sites persist to `evaporation_models.parquet`.
//! The summary builder counts the same models for display; this builder emits the
//! per-`(hydro, stage)` coefficient rows that capture the actual numbers.

use cobre_core::System;
use cobre_io::{DeviationSummary, DeviationWorstEntry, EvaporationModelRow};

use super::types::{
    EvaporationModel, EvaporationReferenceSource, FphaFitDeviationEntry, PrepareHydroModelsResult,
};

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

/// Borrow the canonical-order per-sampled-point FPHA deviation rows from the
/// pipeline result.
///
/// A thin pass-through: the resolver already built these rows in canonical
/// `(hydro_id, stage_id, grid)` order during its sequential flatten, so there is
/// nothing to recompute or re-sort here. It exists beside
/// [`build_evaporation_model_rows`] so the CLI and Python write sites reach the
/// rows through one accessor mirroring the evaporation recipe. Empty unless the
/// run opted in via `config.exports.fpha_deviation_points`.
#[must_use]
pub fn build_fpha_deviation_point_rows(
    result: &PrepareHydroModelsResult,
) -> &[cobre_io::FphaDeviationPointRow] {
    &result.fpha_deviation_point_rows
}

/// Roll up the per-`(hydro, stage)` computed-FPHA fit deviations into the generic
/// run-level [`cobre_io::DeviationSummary`] persisted in `training/metadata.json`.
///
/// Returns `None` on an empty slice — a non-computed-FPHA run records no
/// deviations, and the metadata section is then omitted. Otherwise returns the
/// rollup:
///
/// - `n_entries` — the slice length,
/// - `mean_abs` — the arithmetic mean of every entry's `mean_abs_mw`,
/// - `max_abs` — the maximum of every entry's `max_abs_mw`,
/// - `worst_relative` / `worst_entry` — the entry with the largest `relative`.
///
/// # Determinism — first-seen wins on a relative tie (Voice 1 / D5)
///
/// The worst-entry scan keeps the first entry reaching the running maximum
/// (strict `>`), so a tie resolves to the earliest entry in the slice. `entries`
/// arrives in canonical `(hydro, stage)` order from the resolver, so the chosen
/// winner is declaration-order invariant. Using `>=` (last-seen wins) would still
/// be deterministic but would silently flip which plant is reported on a tie;
/// the strict `>` pins it to the canonical-first entry. The builder never
/// re-sorts the slice — that would break the carried canonical order.
#[must_use]
pub fn build_deviation_summary(entries: &[FphaFitDeviationEntry]) -> Option<DeviationSummary> {
    if entries.is_empty() {
        return None;
    }

    // `entries` is non-empty here, so `n_entries >= 1` and the mean divisor is
    // never zero. The count fits a `u32` — the per-distinct-fit entry count is
    // bounded by hydros × stages, far below `u32::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    let n_entries = entries.len() as u32;

    let sum_mean_abs: f64 = entries.iter().map(|e| e.mean_abs_mw).sum();
    #[allow(clippy::cast_precision_loss)]
    let mean_abs = sum_mean_abs / entries.len() as f64;

    let max_abs = entries
        .iter()
        .map(|e| e.max_abs_mw)
        .fold(f64::NEG_INFINITY, f64::max);

    // First-seen wins on a `relative` tie (strict `>`); see the contract above.
    // Seed the fold with the first entry and fold the rest, so a one-element slice
    // yields it directly and there is no fallible `reduce`/`unwrap` path: the
    // `is_empty()` guard above already makes `entries[0]` safe. Using `reduce(..)
    // .unwrap_or(entries[0])` would carry an unreachable fallback that misreads as
    // "the scan can fail".
    let worst = entries[1..].iter().copied().fold(entries[0], |acc, e| {
        if e.relative > acc.relative { e } else { acc }
    });

    Some(DeviationSummary {
        n_entries,
        mean_abs,
        max_abs,
        worst_relative: worst.relative,
        worst_entry: Some(DeviationWorstEntry {
            entity_id: worst.hydro_id.0,
            stage_id: worst.stage_id,
            relative: worst.relative,
            mean_abs: worst.mean_abs_mw,
            max_abs: worst.max_abs_mw,
        }),
    })
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
            fpha_fit_deviations: Vec::new(),
            fpha_deviation_point_rows: Vec::new(),
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

    // ── build_deviation_summary ────────────────────────────────────────────────

    fn dev_entry(
        hydro_id: i32,
        stage_id: i32,
        mean_abs_mw: f64,
        max_abs_mw: f64,
        mean_signed_mw: f64,
        relative: f64,
    ) -> FphaFitDeviationEntry {
        FphaFitDeviationEntry {
            hydro_id: EntityId(hydro_id),
            stage_id,
            mean_abs_mw,
            max_abs_mw,
            mean_signed_mw,
            relative,
        }
    }

    /// An empty slice records no deviation, so the section is omitted.
    #[test]
    fn build_deviation_summary_empty_is_none() {
        assert!(build_deviation_summary(&[]).is_none());
    }

    /// A non-empty slice rolls up the count, the mean of `mean_abs_mw`, the max of
    /// `max_abs_mw`, and the entry with the largest `relative`.
    #[test]
    fn build_deviation_summary_rolls_up_entries() {
        let entries = vec![
            dev_entry(11, 1, 2.0, 10.0, 1.0, 0.01),
            dev_entry(12, 4, 8.0, 31.7, -2.0, 0.062),
            dev_entry(13, 2, 4.1, 20.0, 0.5, 0.03),
        ];

        let summary =
            build_deviation_summary(&entries).expect("non-empty slice must yield a summary");

        assert_eq!(summary.n_entries, 3);
        // mean of mean_abs_mw = (2.0 + 8.0 + 4.1) / 3.
        assert!((summary.mean_abs - (2.0 + 8.0 + 4.1) / 3.0).abs() < 1e-12);
        // max of max_abs_mw.
        assert_eq!(summary.max_abs, 31.7);
        // worst by relative is the second entry (hydro 12, stage 4).
        assert_eq!(summary.worst_relative, 0.062);
        let worst = summary.worst_entry.expect("worst entry must be present");
        assert_eq!(worst.entity_id, 12);
        assert_eq!(worst.stage_id, 4);
        assert_eq!(worst.relative, 0.062);
        assert_eq!(worst.mean_abs, 8.0);
        assert_eq!(worst.max_abs, 31.7);
    }

    /// A single entry rolls up to itself: the mean equals its `mean_abs_mw`, the
    /// max equals its `max_abs_mw`, and it is its own worst entry.
    #[test]
    fn build_deviation_summary_single_entry() {
        let entries = vec![dev_entry(7, 3, 5.5, 12.5, 1.0, 0.04)];
        let summary = build_deviation_summary(&entries).expect("single entry must yield a summary");
        assert_eq!(summary.n_entries, 1);
        assert_eq!(summary.mean_abs, 5.5);
        assert_eq!(summary.max_abs, 12.5);
        assert_eq!(summary.worst_relative, 0.04);
        let worst = summary.worst_entry.expect("worst entry must be present");
        assert_eq!(worst.entity_id, 7);
        assert_eq!(worst.stage_id, 3);
    }

    /// On a `relative` tie the canonical-first entry wins (strict `>`), so the
    /// reported worst entry is declaration-order invariant.
    #[test]
    fn build_deviation_summary_tie_keeps_first() {
        let entries = vec![
            dev_entry(20, 1, 3.0, 9.0, 0.0, 0.05),
            dev_entry(21, 2, 7.0, 15.0, 0.0, 0.05),
        ];
        let summary = build_deviation_summary(&entries).expect("must yield a summary");
        let worst = summary.worst_entry.expect("worst entry must be present");
        assert_eq!(
            worst.entity_id, 20,
            "the canonical-first entry must win a relative tie"
        );
    }
}
