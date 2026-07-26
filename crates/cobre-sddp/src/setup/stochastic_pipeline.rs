//! Stochastic preprocessing pipeline: PAR estimation, opening tree loading, and stochastic context construction.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use cobre_core::temporal::SeasonCycleType::Monthly;
use cobre_core::temporal::SeasonMap;
use cobre_core::{
    EntityId, System,
    scenario::{SamplingScheme, ScenarioSource},
};
use cobre_io::Config;
use cobre_io::LoadFactorEntry;
use cobre_io::ValidationContext;
use cobre_io::scenarios::assemble_opening_tree;
use cobre_io::scenarios::estimation::estimate_from_history;
use cobre_io::scenarios::load_noise_openings;
use cobre_io::scenarios::parse_load_factors;
use cobre_io::scenarios::validate_noise_openings;
use cobre_io::validate_structure;
use cobre_stochastic::BlockFactorPair;
use cobre_stochastic::ClassSchemes;
use cobre_stochastic::DerivedInflowSeeds;
use cobre_stochastic::HistoricalScenarioLibrary;
use cobre_stochastic::PrecomputedPar;
use cobre_stochastic::build_stochastic_context;
use cobre_stochastic::derive_inflow_seeds;
use cobre_stochastic::discover_historical_windows;
use cobre_stochastic::noise_entity_order;
use cobre_stochastic::normal::precompute::EntityFactorEntry;
use cobre_stochastic::par::lag_transition::derive_downstream_par_order;
use cobre_stochastic::par::lag_transition::precompute_noise_groups;
use cobre_stochastic::par::lag_transition::precompute_stage_lag_transitions;
use cobre_stochastic::standardize_historical_windows;
use cobre_stochastic::{OpeningTreeInputs, StochasticContext, context::OpeningTree};

use crate::{EstimationPath, EstimationReport, SddpError};

/// Result of the stochastic preprocessing pipeline.
#[derive(Debug)]
pub struct PrepareStochasticResult {
    /// Updated system with estimated PAR models (if estimation ran).
    pub system: System,
    /// Built stochastic context, ready to pass to [`crate::setup::StudySetup::new`].
    pub stochastic: StochasticContext,
    /// Estimation report (`Some` if `inflow_history.parquet` was present and
    /// `inflow_seasonal_stats.parquet` was absent, triggering auto-estimation).
    pub estimation_report: Option<EstimationReport>,
    /// Which estimation path row was taken during preprocessing.
    pub estimation_path: EstimationPath,
}

/// Load and validate a user-supplied opening tree from `scenarios/noise_openings.parquet`.
///
/// Returns `Ok(None)` if the file is absent.
///
/// # Errors
///
/// Returns [`SddpError::Io`] if the file exists but cannot be read or fails validation.
fn load_user_opening_tree_inner(
    case_dir: &Path,
    system: &System,
) -> Result<Option<OpeningTree>, SddpError> {
    let mut ctx = ValidationContext::new();
    let manifest = validate_structure(case_dir, &mut ctx);

    if !manifest.scenarios_noise_openings_parquet {
        return Ok(None);
    }

    let path = case_dir.join("scenarios").join("noise_openings.parquet");

    let rows = load_noise_openings(Some(&path))?;

    let expected_dim = noise_entity_order(system).dim();

    let expected_stages = system.stages().iter().filter(|s| s.id >= 0).count();
    let mut openings_by_stage: BTreeMap<i32, BTreeSet<u32>> = BTreeMap::new();
    for row in &rows {
        openings_by_stage
            .entry(row.stage_id)
            .or_default()
            .insert(row.opening_index);
    }
    let openings_per_stage: Vec<usize> = openings_by_stage.values().map(BTreeSet::len).collect();

    validate_noise_openings(&rows, expected_dim, expected_stages, &openings_per_stage)?;

    let tree = assemble_opening_tree(rows, expected_dim);
    Ok(Some(tree))
}

/// Noise-group ids for the study stages, delegating to [`precompute_noise_groups`].
///
/// Single owner: every site needing per-stage noise-group ids calls this rather
/// than inlining the filter — an inlined copy diverges whenever two consecutive
/// stages share a `(season_id, year)` group, changing which openings a rank solves
/// against.
#[must_use]
pub fn study_stage_noise_group_ids(system: &System) -> Vec<u32> {
    let study_stages: Vec<_> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .cloned()
        .collect();
    precompute_noise_groups(&study_stages)
}

/// Build NCS entity factor entries from `System::resolved_ncs_factors()`.
///
/// Converts the dense 3D table into `(entity_id, stage_id, block_pairs)` tuples
/// for `PrecomputedNormal::build`.
#[must_use]
pub fn build_ncs_factor_entries(system: &System) -> Vec<(EntityId, i32, Vec<BlockFactorPair>)> {
    let stochastic_ncs: BTreeSet<EntityId> = system.ncs_models().iter().map(|m| m.ncs_id).collect();

    if stochastic_ncs.is_empty() {
        return Vec::new();
    }

    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let ncs_ids: Vec<EntityId> = system
        .non_controllable_sources()
        .iter()
        .map(|n| n.id)
        .collect();

    let mut entries = Vec::new();
    for (ncs_idx, ncs_id) in ncs_ids.iter().enumerate() {
        if !stochastic_ncs.contains(ncs_id) {
            continue;
        }
        for (stage_idx, stage) in study_stages.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let block_pairs: Vec<BlockFactorPair> = stage
                .blocks
                .iter()
                .enumerate()
                .map(|(block_idx, _)| {
                    let factor = system
                        .resolved_ncs_factors()
                        .factor(ncs_idx, stage_idx, block_idx);
                    // block_idx is a small count (< 1000 in practice); fits in i32.
                    (block_idx as i32, factor)
                })
                .collect();
            entries.push((*ncs_id, stage.id, block_pairs));
        }
    }
    entries
}

/// Load `scenarios/load_factors.json` from the case directory, returning an
/// empty vec when the file is absent.
///
/// # Errors
///
/// Returns [`SddpError`] if the file exists but cannot be read or parsed.
pub fn load_load_factors_for_stochastic(
    case_dir: &Path,
) -> Result<Vec<LoadFactorEntry>, SddpError> {
    let path = case_dir.join("scenarios").join("load_factors.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    parse_load_factors(&path).map_err(SddpError::from)
}

/// Build the `HistoricalScenarioLibrary` for the opening tree when any stage
/// uses `NoiseMethod::HistoricalResiduals`. Returns `None` when no stage
/// needs historical draws.
fn build_opening_tree_library(
    system: &System,
    training_source: &ScenarioSource,
) -> Result<Option<HistoricalScenarioLibrary>, SddpError> {
    use cobre_core::temporal::NoiseMethod;
    let needs_historical_tree = system
        .stages()
        .iter()
        .any(|s| s.id >= 0 && s.scenario_config.noise_method == NoiseMethod::HistoricalResiduals);
    if !needs_historical_tree {
        return Ok(None);
    }
    let study_stages: Vec<_> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .cloned()
        .collect();
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
    let cycle_len = system
        .policy_graph()
        .season_map
        .as_ref()
        .map(|sm| sm.seasons.len());
    let par = PrecomputedPar::build(system.inflow_models(), &study_stages, &hydro_ids, cycle_len)?;
    let max_order = par.max_order();
    let user_pool = training_source.historical_years.as_ref();
    let window_years = discover_historical_windows(
        system.inflow_history(),
        &hydro_ids,
        &study_stages,
        max_order,
        user_pool,
        system.policy_graph().season_map.as_ref(),
        1,
    )?;
    let mut lib = HistoricalScenarioLibrary::new(
        window_years.len(),
        study_stages.len(),
        hydro_ids.len(),
        max_order,
        window_years.clone(),
    );
    // η-inversion rolling chain must match the forward-pass lag accumulator;
    // `max_order` width covers all AR lags.
    let season_map_ref = system.policy_graph().season_map.as_ref();
    // `precompute_stage_lag_transitions` requires a non-optional &SeasonMap.
    let noop_season_map = SeasonMap {
        cycle_type: Monthly,
        seasons: Vec::new(),
    };
    let effective_season_map: &SeasonMap = season_map_ref.unwrap_or(&noop_season_map);
    let downstream_par_order =
        derive_downstream_par_order(&study_stages, max_order, season_map_ref);
    let stage_lag_transitions =
        precompute_stage_lag_transitions(&study_stages, effective_season_map, downstream_par_order);
    let derived_inflow_seeds = match study_stages.first() {
        None => DerivedInflowSeeds::zero(hydro_ids.len(), max_order),
        Some(first_stage) => derive_inflow_seeds(
            system.inflow_history(),
            &system.initial_conditions().recent_observations,
            system.hydros(),
            first_stage,
            effective_season_map,
            max_order,
        ),
    };
    standardize_historical_windows(
        &mut lib,
        system.inflow_history(),
        &hydro_ids,
        &study_stages,
        &par,
        &window_years,
        season_map_ref,
        &derived_inflow_seeds.lag_values,
        max_order,
        &derived_inflow_seeds.accum,
        &derived_inflow_seeds.weight,
        &stage_lag_transitions,
        downstream_par_order,
    );
    Ok(Some(lib))
}

/// Compute per-stage external scenario counts for opening tree clamping.
///
/// When any entity class uses External sampling, the external library is padded
/// to a uniform scenario count after loading. The opening tree generator must
/// clamp per-stage openings to the pre-padding raw count. The element-wise
/// minimum across entity classes is returned.
fn compute_external_scenario_counts(
    system: &System,
    training_source: &ScenarioSource,
) -> Option<Vec<usize>> {
    let n_stages = system.stages().iter().filter(|s| s.id >= 0).count();
    let noise_order = noise_entity_order(system);

    let inflow_counts: Option<Vec<usize>> =
        if training_source.inflow_scheme == SamplingScheme::External && n_stages > 0 {
            let external_rows = system.external_scenarios();
            let n_hydros = noise_order.hydro_ids.len();
            let mut rows_per_stage = vec![0usize; n_stages];
            #[allow(clippy::cast_sign_loss)]
            for row in external_rows {
                let s = row.stage_id as usize;
                if s < n_stages {
                    rows_per_stage[s] += 1;
                }
            }
            Some(if n_hydros > 0 {
                rows_per_stage.iter().map(|&r| r / n_hydros).collect()
            } else {
                vec![0usize; n_stages]
            })
        } else {
            None
        };

    let load_counts: Option<Vec<usize>> =
        if training_source.load_scheme == SamplingScheme::External && n_stages > 0 {
            let external_rows = system.external_load_scenarios();
            let n_buses = noise_order.load_bus_ids.len();
            let mut rows_per_stage = vec![0usize; n_stages];
            #[allow(clippy::cast_sign_loss)]
            for row in external_rows {
                let s = row.stage_id as usize;
                if s < n_stages {
                    rows_per_stage[s] += 1;
                }
            }
            Some(if n_buses > 0 {
                rows_per_stage.iter().map(|&r| r / n_buses).collect()
            } else {
                vec![0usize; n_stages]
            })
        } else {
            None
        };

    let ncs_counts: Option<Vec<usize>> =
        if training_source.ncs_scheme == SamplingScheme::External && n_stages > 0 {
            let external_rows = system.external_ncs_scenarios();
            let n_ncs = noise_order.ncs_entity_ids.len();
            let mut rows_per_stage = vec![0usize; n_stages];
            #[allow(clippy::cast_sign_loss)]
            for row in external_rows {
                let s = row.stage_id as usize;
                if s < n_stages {
                    rows_per_stage[s] += 1;
                }
            }
            Some(if n_ncs > 0 {
                rows_per_stage.iter().map(|&r| r / n_ncs).collect()
            } else {
                vec![0usize; n_stages]
            })
        } else {
            None
        };

    match (inflow_counts, load_counts, ncs_counts) {
        (None, None, None) => None,
        (Some(a), None, None) => Some(a),
        (None, Some(b), None) => Some(b),
        (None, None, Some(c)) => Some(c),
        (Some(a), Some(b), None) => Some(a.iter().zip(b.iter()).map(|(&x, &y)| x.min(y)).collect()),
        (Some(a), None, Some(c)) => Some(a.iter().zip(c.iter()).map(|(&x, &y)| x.min(y)).collect()),
        (None, Some(b), Some(c)) => Some(b.iter().zip(c.iter()).map(|(&x, &y)| x.min(y)).collect()),
        (Some(a), Some(b), Some(c)) => Some(
            a.iter()
                .zip(b.iter())
                .zip(c.iter())
                .map(|((&x, &y), &z)| x.min(y).min(z))
                .collect(),
        ),
    }
}

/// Run the stochastic preprocessing pipeline: PAR estimation, block factor
/// loading, opening-tree library construction, and stochastic context build.
///
/// # Errors
///
/// Returns [`SddpError::Io`] on file read/parse/validation failure,
/// or [`SddpError::Stochastic`] on PAR/decomposition failure.
pub fn prepare_stochastic(
    system: System,
    case_dir: &Path,
    config: &Config,
    seed: u64,
    training_source: &ScenarioSource,
) -> Result<PrepareStochasticResult, SddpError> {
    let (system, estimation_report, estimation_path) =
        estimate_from_history(system, case_dir, config)?;

    let user_opening_tree = load_user_opening_tree_inner(case_dir, &system)?;

    let load_factor_entries = load_load_factors_for_stochastic(case_dir)?;
    let block_pairs: Vec<Vec<BlockFactorPair>> = load_factor_entries
        .iter()
        .map(|e| {
            e.block_factors
                .iter()
                .map(|bf| (bf.block_id, bf.factor))
                .collect()
        })
        .collect();
    let entity_factor_entries: Vec<EntityFactorEntry<'_>> = load_factor_entries
        .iter()
        .zip(block_pairs.iter())
        .map(|(e, pairs)| (e.bus_id, e.stage_id, pairs.as_slice()))
        .collect();

    let ncs_factor_entries = build_ncs_factor_entries(&system);
    let ncs_entity_factor_entries: Vec<EntityFactorEntry<'_>> = ncs_factor_entries
        .iter()
        .map(|(ncs_id, stage_id, pairs)| (*ncs_id, *stage_id, pairs.as_slice()))
        .collect();

    let opening_tree_library = build_opening_tree_library(&system, training_source)?;
    let external_scenario_counts = compute_external_scenario_counts(&system, training_source);

    let opening_tree_noise_group_ids = study_stage_noise_group_ids(&system);

    let forward_seed = training_source.seed.map(i64::unsigned_abs);
    let stochastic = build_stochastic_context(
        &system,
        seed,
        forward_seed,
        &entity_factor_entries,
        &ncs_entity_factor_entries,
        OpeningTreeInputs {
            user_tree: user_opening_tree,
            historical_library: opening_tree_library.as_ref(),
            external_scenario_counts,
            noise_group_ids: Some(opening_tree_noise_group_ids),
        },
        ClassSchemes {
            inflow: Some(training_source.inflow_scheme),
            load: Some(training_source.load_scheme),
            ncs: Some(training_source.ncs_scheme),
        },
    )?;

    Ok(PrepareStochasticResult {
        system,
        stochastic,
        estimation_report,
        estimation_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractStageBounds, HydroStageBounds,
        HydroStagePenalties, InitialConditions, LineStageBounds, LineStagePenalties,
        NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults, PumpingStageBounds,
        ResolvedBounds, ResolvedPenalties, SystemBuilder, ThermalStageBounds,
        entities::{
            bus::{Bus, DeficitSegment},
            hydro::{Hydro, HydroGenerationModel, HydroPenalties},
        },
        scenario::{InflowHistoryRow, InflowModel, LoadModel},
        temporal::{
            Block, BlockMode, NoiseMethod, PolicyGraph, PolicyGraphType, ScenarioSourceConfig,
            SeasonCycleType, SeasonDefinition, SeasonMap, Stage, StageLagTransition,
            StageRiskConfig, StageStateConfig,
        },
    };
    use cobre_stochastic::par::lag_transition::{
        derive_downstream_par_order, precompute_stage_lag_transitions,
    };
    use cobre_stochastic::{
        PrecomputedPar,
        par::lag_kernel::{DownstreamLagAccum, LagMajor, PrimaryLagAccum, advance_lag_chain},
        solve_par_noise,
    };

    /// Season definitions for the monthly->quarterly fixture below: three
    /// monthly seasons, then two quarterly seasons (`id >= 12`) — the same
    /// shape as the DLC fixture in `forward_sampler_integration.rs`.
    fn ring_season_map() -> SeasonMap {
        let def = |id: usize, month_start: u32, month_end: Option<u32>| SeasonDefinition {
            id,
            label: format!("S{id}"),
            month_start,
            day_start: None,
            month_end,
            day_end: None,
        };
        SeasonMap {
            cycle_type: SeasonCycleType::Custom,
            seasons: vec![
                def(0, 1, None),
                def(1, 2, None),
                def(2, 3, None),
                def(12, 4, Some(6)),
                def(13, 7, Some(9)),
            ],
        }
    }

    /// A `System` with 5 study stages (three monthly, Jan-Mar 2026, then two
    /// quarterly, Q2-Q3 2026) and one hydro on `HistoricalResiduals`, wired so
    /// `build_opening_tree_library` runs its production derivation end to end
    /// rather than a hand-supplied `downstream_par_order`.
    struct RingFixture {
        system: System,
        stages: Vec<Stage>,
        hydro_id: EntityId,
        raw: [f64; 5],
        past_inflow_seed: f64,
    }

    // Rationale: one flat literal `System` fixture (bus, hydro, 5 stages,
    // inflow models, bounds, penalties) mirroring `setup/tests.rs`'s
    // `minimal_system_2_hydros_with_history`; splitting it fragments a
    // single-purpose, single-call fixture across artificial sub-functions.
    #[allow(clippy::too_many_lines)]
    fn build_ring_fixture() -> RingFixture {
        let hydro_id = EntityId(2);
        let bus_id = EntityId(1);

        let stage = |index: usize,
                     id: i32,
                     start: NaiveDate,
                     end: NaiveDate,
                     season_id: usize,
                     duration_hours: f64| Stage {
            index,
            id,
            start_date: start,
            end_date: end,
            season_id: Some(season_id),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: true,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::HistoricalResiduals,
            },
        };

        let stages = vec![
            stage(
                0,
                0,
                NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(),
                0,
                31.0 * 24.0,
            ),
            stage(
                1,
                1,
                NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 3, 1).unwrap(),
                1,
                28.0 * 24.0,
            ),
            stage(
                2,
                2,
                NaiveDate::from_ymd_opt(2026, 3, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
                2,
                31.0 * 24.0,
            ),
            stage(
                3,
                3,
                NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 7, 1).unwrap(),
                12,
                91.0 * 24.0,
            ),
            stage(
                4,
                4,
                NaiveDate::from_ymd_opt(2026, 7, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 10, 1).unwrap(),
                13,
                92.0 * 24.0,
            ),
        ];

        let raw = [80.0, 150.0, 40.0, 190.0, 60.0];
        let past_inflow_seed = 70.0;

        let mut inflow_models = Vec::with_capacity(6);
        for stage_id in -1..5_i32 {
            inflow_models.push(InflowModel {
                hydro_id,
                stage_id,
                mean_m3s: 100.0,
                std_m3s: 20.0,
                ar_coefficients: if stage_id >= 0 { vec![0.6] } else { vec![] },
                residual_std_ratio: 1.0,
                annual: None,
            });
        }

        let mut inflow_history: Vec<InflowHistoryRow> = stages
            .iter()
            .enumerate()
            .map(|(t, stage)| InflowHistoryRow {
                hydro_id,
                start_date: stage.start_date,
                end_date: stage.end_date,
                value_m3s: raw[t],
            })
            .collect();
        // `discover_historical_windows` also requires one lag observation
        // (season 13, the wraparound lag season for PAR order 1 immediately
        // before the first study season). Its value is never read by
        // `standardize_historical_windows`, which only consumes the 5
        // study-stage entries above.
        let lag_start = NaiveDate::from_ymd_opt(2025, 7, 15).unwrap();
        inflow_history.push(InflowHistoryRow {
            hydro_id,
            start_date: lag_start,
            end_date: lag_start.succ_opt().unwrap(),
            value_m3s: 999.0,
        });

        let load_models: Vec<LoadModel> = (0..5_i32)
            .map(|stage_id| LoadModel {
                bus_id,
                stage_id,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        let bus = Bus {
            id: bus_id,
            name: "B1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        let hydro = Hydro {
            id: hydro_id,
            name: "H1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id,
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
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
        };

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 5,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 100.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 250.0,
                    max_diversion_m3s: None,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                    cost_per_mwh: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 5,
            },
            &PenaltiesDefaults {
                hydro: HydroStagePenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 500.0,
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
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let policy_graph = PolicyGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: vec![],
            season_map: Some(ring_season_map()),
        };

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages.clone())
            .inflow_models(inflow_models)
            .load_models(load_models)
            .inflow_history(inflow_history)
            .bounds(bounds)
            .penalties(penalties)
            .policy_graph(policy_graph)
            .initial_conditions(InitialConditions {
                storage: vec![],
                filling_storage: vec![],
                past_anticipated_commitments: vec![],
                recent_observations: vec![],
                past_defluences: vec![],
            })
            .build()
            .expect("ring fixture: valid system");

        RingFixture {
            system,
            stages,
            hydro_id,
            raw,
            past_inflow_seed,
        }
    }

    /// Drive `advance_lag_chain::<LagMajor>` across `fx`'s 5 stages, returning
    /// the incoming (pre-advance) lag at stage 4 — the value the AR(1) model
    /// there reads. `downstream_par_order = 0` (`accumulator`/`completed_lags`
    /// empty) reproduces the literal-`0` regression `build_opening_tree_library`
    /// used to pass to `standardize_historical_windows`; a positive value
    /// reproduces the ring-aware, fixed behavior.
    fn ring_fixture_incoming_lag_at_stage4(
        fx: &RingFixture,
        transitions: &[StageLagTransition],
        downstream_par_order: usize,
    ) -> f64 {
        let layout = LagMajor {
            entity_count: 1,
            max_order: 1,
        };
        let mut lag_state = vec![fx.past_inflow_seed];
        let mut incoming = vec![0.0];
        let mut primary_acc = vec![0.0];
        let mut primary_w = vec![0.0];
        let mut ds_acc = if downstream_par_order > 0 {
            vec![0.0]
        } else {
            Vec::new()
        };
        let mut ds_completed = vec![0.0; downstream_par_order];
        let mut ds_w = 0.0_f64;
        let mut ds_n = 0usize;
        let mut incoming_stage4 = 0.0;

        for (t, (stage_lag, &raw)) in transitions.iter().zip(fx.raw.iter()).enumerate() {
            incoming.copy_from_slice(&lag_state);
            if t == 4 {
                incoming_stage4 = incoming[0];
            }
            let mut primary = PrimaryLagAccum {
                accumulator: &mut primary_acc,
                weight_accum: &mut primary_w,
            };
            let mut downstream = DownstreamLagAccum {
                accumulator: &mut ds_acc,
                weight_accum: &mut ds_w,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n,
                par_order: downstream_par_order,
            };
            advance_lag_chain(
                layout,
                &mut lag_state,
                &incoming,
                &[raw],
                stage_lag,
                &mut primary,
                &mut downstream,
            );
        }
        incoming_stage4
    }

    /// `build_opening_tree_library` (rank-0 opening-tree build) must thread
    /// `derive_downstream_par_order`'s result into `standardize_historical_windows`,
    /// not a literal `0`: the resulting eta at the quarterly transition must
    /// match the independent ring-aware forward-chain oracle and differ from
    /// the primary-only advance the literal-`0` regression produces.
    #[test]
    fn build_opening_tree_library_ring_aware_eta_at_quarterly_transition() {
        let fx = build_ring_fixture();
        let training_source = ScenarioSource {
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: None,
            historical_years: None,
        };

        let lib = build_opening_tree_library(&fx.system, &training_source)
            .expect("build_opening_tree_library must succeed")
            .expect("HistoricalResiduals noise method must build a library");
        assert_eq!(
            lib.n_windows(),
            1,
            "exactly one historical window (2026) is discoverable"
        );

        let par =
            PrecomputedPar::build(fx.system.inflow_models(), &fx.stages, &[fx.hydro_id], None)
                .expect("oracle PrecomputedPar must build");
        assert_eq!(par.max_order(), 1);

        let derived = derive_downstream_par_order(
            &fx.stages,
            par.max_order(),
            fx.system.policy_graph().season_map.as_ref(),
        );
        assert_eq!(
            derived,
            par.max_order(),
            "the fixture crosses season_id >= 12 at stage 3; derive_downstream_par_order \
             must gate to par.max_order(), not 0"
        );
        let season_map = ring_season_map();
        let transitions = precompute_stage_lag_transitions(&fx.stages, &season_map, derived);

        let ring_aware_incoming = ring_fixture_incoming_lag_at_stage4(&fx, &transitions, derived);
        let naive_incoming = ring_fixture_incoming_lag_at_stage4(&fx, &transitions, 0);
        assert!(
            (ring_aware_incoming - naive_incoming).abs() > 1.0,
            "ring-rebuilt lag feeding stage 4 must differ from the primary-only advance, \
             got ring_aware={ring_aware_incoming} vs naive={naive_incoming}"
        );

        let det_base = par.deterministic_base(4, 0);
        let psi = par.psi_slice(4, 0);
        let sigma = par.sigma(4, 0);
        let expected_ring_aware_eta =
            solve_par_noise(det_base, psi, &[ring_aware_incoming], sigma, fx.raw[4]);
        let expected_naive_eta =
            solve_par_noise(det_base, psi, &[naive_incoming], sigma, fx.raw[4]);

        let eta_stage4 = lib.eta_slice(0, 4)[0];
        assert_eq!(
            eta_stage4, expected_ring_aware_eta,
            "build_opening_tree_library's eta at the quarterly transition must match the \
             independent ring-aware forward-chain oracle"
        );
        assert!(
            (eta_stage4 - expected_naive_eta).abs() > 1e-6,
            "build_opening_tree_library's eta must differ from the primary-only advance \
             (the literal downstream_par_order=0 regression value), got eta={eta_stage4} \
             vs naive={expected_naive_eta}"
        );
    }
}
