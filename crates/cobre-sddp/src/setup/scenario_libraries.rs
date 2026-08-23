//! Scenario library builders for historical and external sampling schemes.
//!
//! Builders are not factored generically because external types have different
//! standardization semantics.

use cobre_core::{
    EntityId, InflowHistoryRow, Stage,
    scenario::{
        ExternalLoadRow, ExternalNcsRow, ExternalScenarioRow, HistoricalYears, LoadModel, NcsModel,
        SamplingScheme,
    },
    temporal::{SeasonMap, StageLagTransition},
};
use cobre_io::StageIdResolver;
use cobre_stochastic::{
    ExternalScenarioLibrary, HistoricalScenarioLibrary, PrecomputedPar,
    derive_external_sample_moments, discover_historical_windows, pad_library_to_uniform,
    standardize_external_inflow, standardize_external_load, standardize_external_ncs,
    standardize_historical_windows, validate_external_library, validate_historical_library,
};

use crate::SddpError;

/// Build and validate a [`HistoricalScenarioLibrary`] for inflow.
///
/// `derived_lag_values`/`derived_accum`/`derived_weight` and
/// `stage_lag_transitions` seed the rolling η-inversion chain (mirroring
/// `build_external_inflow_library`). Pass the shared
/// `StudySetup::derived_inflow_seeds` fields and the pre-computed transitions
/// so that every forward pass starting from the same derived seed exactly
/// reconstructs the raw historical observations.
///
/// # Errors
///
/// Returns `SddpError::Stochastic` on window discovery or validation failure.
// Rationale: mirrors standardize_historical_windows's own arity; a context
// struct would just relocate the arity, not reduce it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_historical_inflow_library(
    inflow_history: &[InflowHistoryRow],
    hydro_ids: &[EntityId],
    stages: &[Stage],
    par: &PrecomputedPar,
    season_map: Option<&SeasonMap>,
    derived_lag_values: &[f64],
    l_state: usize,
    derived_accum: &[f64],
    derived_weight: &[f64],
    stage_lag_transitions: &[StageLagTransition],
    user_pool: Option<&HistoricalYears>,
    forward_passes: u32,
    downstream_par_order: usize,
) -> Result<HistoricalScenarioLibrary, SddpError> {
    let max_order = par.max_order();
    let window_years = discover_historical_windows(
        inflow_history,
        hydro_ids,
        stages,
        max_order,
        user_pool,
        season_map,
        forward_passes,
    )
    .map_err(SddpError::Stochastic)?;
    let mut library = HistoricalScenarioLibrary::new(
        window_years.len(),
        stages.len(),
        hydro_ids.len(),
        max_order,
        window_years.clone(),
    );
    standardize_historical_windows(
        &mut library,
        inflow_history,
        hydro_ids,
        stages,
        par,
        &window_years,
        season_map,
        derived_lag_values,
        l_state,
        derived_accum,
        derived_weight,
        stage_lag_transitions,
        downstream_par_order,
    );
    validate_historical_library(
        &library,
        inflow_history,
        hydro_ids,
        stages,
        max_order,
        user_pool,
        forward_passes,
    )
    .map_err(SddpError::Stochastic)?;
    Ok(library)
}

/// Build and validate an [`ExternalScenarioLibrary`] for inflow.
///
/// # Errors
///
/// Returns `SddpError::Stochastic` on validation failure.
// Rationale: mirrors standardize_external_inflow's own arity; a context
// struct would just relocate the arity, not reduce it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_external_inflow_library(
    external_rows: &[ExternalScenarioRow],
    hydro_ids: &[EntityId],
    stages: &[Stage],
    par: &PrecomputedPar,
    derived_lag_values: &[f64],
    l_state: usize,
    derived_accum: &[f64],
    derived_weight: &[f64],
    stage_lag_transitions: &[StageLagTransition],
    forward_passes: u32,
    downstream_par_order: usize,
) -> Result<ExternalScenarioLibrary, SddpError> {
    let n_stages = stages.len();
    let n_hydros = hydro_ids.len();
    let row_entity_ids: std::collections::HashSet<EntityId> =
        external_rows.iter().map(|r| r.hydro_id).collect();
    let resolver =
        StageIdResolver::from_study_stage_ids(&stages.iter().map(|s| s.id).collect::<Vec<_>>());
    let mut rows_per_stage = vec![0usize; n_stages];
    for row in external_rows {
        if let Some(idx) = resolver.resolve(row.stage_id) {
            rows_per_stage[idx] += 1;
        }
    }
    let per_stage_scenarios: Vec<usize> = if n_hydros > 0 {
        rows_per_stage.iter().map(|&r| r / n_hydros).collect()
    } else {
        vec![0usize; n_stages]
    };
    let n_scenarios_ext = per_stage_scenarios.iter().copied().max().unwrap_or(0);
    let mut library = ExternalScenarioLibrary::new(
        n_stages,
        n_scenarios_ext,
        n_hydros,
        "inflow",
        per_stage_scenarios,
    );
    standardize_external_inflow(
        &mut library,
        external_rows,
        hydro_ids,
        stages,
        par,
        derived_lag_values,
        l_state,
        derived_accum,
        derived_weight,
        stage_lag_transitions,
        downstream_par_order,
    );
    validate_external_library(
        &library,
        hydro_ids,
        &row_entity_ids,
        &rows_per_stage,
        n_stages,
        forward_passes,
    )
    .map_err(SddpError::Stochastic)?;
    pad_library_to_uniform(&mut library);
    Ok(library)
}

/// Build a derived-moment model slice under `External`: one entry per
/// `(entity, stage)` with the sample moments [`derive_external_sample_moments`]
/// computes over `external_rows`, mapped into the class's model type via
/// `constructor`. Feeds the SAME `(μ, σ)` pair into `standardize_external_load`
/// / `standardize_external_ncs` that `cobre_stochastic::context` derives from
/// the same rows for reconstruction, so the standardize/reconstruct round trip
/// holds (design doc §3–§4.1) — the two call sites must keep deriving from the
/// same rows.
///
/// `row_fields`' `stage_id` is the row's raw declared domain id; `resolver`
/// maps it to the canonical 0-based index `derive_external_sample_moments`
/// requires, the same [`StageIdResolver`] the rule-47 validator resolves
/// against, so a gapped or non-0-based deck's rows land on the same stage the
/// validator's own σ=0 decision reads. An unresolvable `stage_id` is dropped,
/// never fed to the reduction.
fn external_derived_models<R, FR, M, FM>(
    external_rows: &[R],
    entity_ids: &[EntityId],
    stages: &[Stage],
    resolver: &StageIdResolver,
    row_fields: FR,
    constructor: FM,
) -> Vec<M>
where
    FR: Fn(&R) -> (EntityId, i32, i32, f64),
    FM: Fn(EntityId, i32, f64, f64) -> M,
{
    let n_entities = entity_ids.len();
    let resolved_rows: Vec<(EntityId, i32, i32, f64)> = external_rows
        .iter()
        .filter_map(|row| {
            let (entity_id, stage_id, scenario_id, value) = row_fields(row);
            let stage_idx = resolver.resolve(stage_id)?;
            let stage_idx_i32 = i32::try_from(stage_idx).ok()?;
            Some((entity_id, stage_idx_i32, scenario_id, value))
        })
        .collect();
    let moments = derive_external_sample_moments(
        &resolved_rows,
        entity_ids,
        stages.len(),
        |&(entity_id, stage_idx, scenario_id, value)| (entity_id, stage_idx, scenario_id, value),
    );
    let mut models = Vec::with_capacity(stages.len() * n_entities);
    for (stage_idx, stage) in stages.iter().enumerate() {
        for (entity_idx, &entity_id) in entity_ids.iter().enumerate() {
            let (mean, std) = moments[stage_idx * n_entities + entity_idx];
            models.push(constructor(entity_id, stage.id, mean, std));
        }
    }
    models
}

/// Build and validate an [`ExternalScenarioLibrary`] for load.
///
/// Canonical bus ID list from `load_models`, filtered by
/// [`LoadModel::is_noise_member`] — the same authority `noise_entity_order`
/// consumes, so a σ=0 bus keeps its noise-vector slot under the external
/// scheme. `load_scheme` is the CALLING phase's own resolved scheme; a phase
/// whose scheme diverges from the training-derived noise-vector width is
/// caught by `assert_external_library_widths`, not here.
///
/// # Errors
///
/// Returns `SddpError::Stochastic` on validation failure.
pub(crate) fn build_external_load_library(
    external_rows: &[ExternalLoadRow],
    load_models: &[LoadModel],
    load_scheme: SamplingScheme,
    stages: &[Stage],
    forward_passes: u32,
) -> Result<ExternalScenarioLibrary, SddpError> {
    let n_stages = stages.len();
    let mut bus_ids: Vec<EntityId> = load_models
        .iter()
        .filter(|m| m.is_noise_member(load_scheme))
        .map(|m| m.bus_id)
        .collect();
    bus_ids.sort_unstable_by_key(|id| id.0);
    bus_ids.dedup();
    let n_buses = bus_ids.len();
    let row_entity_ids: std::collections::HashSet<EntityId> =
        external_rows.iter().map(|r| r.bus_id).collect();
    let resolver =
        StageIdResolver::from_study_stage_ids(&stages.iter().map(|s| s.id).collect::<Vec<_>>());
    let mut rows_per_stage = vec![0usize; n_stages];
    for row in external_rows {
        if let Some(idx) = resolver.resolve(row.stage_id) {
            rows_per_stage[idx] += 1;
        }
    }
    let per_stage_scenarios: Vec<usize> = if n_buses > 0 {
        rows_per_stage.iter().map(|&r| r / n_buses).collect()
    } else {
        vec![0usize; n_stages]
    };
    let n_scenarios_ext = per_stage_scenarios.iter().copied().max().unwrap_or(0);
    let mut library = ExternalScenarioLibrary::new(
        n_stages,
        n_scenarios_ext,
        n_buses,
        "load",
        per_stage_scenarios,
    );
    let derived_load_models = external_derived_models(
        external_rows,
        &bus_ids,
        stages,
        &resolver,
        |row| (row.bus_id, row.stage_id, row.scenario_id, row.value_mw),
        |bus_id, stage_id, mean_mw, std_mw| LoadModel {
            bus_id,
            stage_id,
            mean_mw,
            std_mw,
        },
    );
    standardize_external_load(
        &mut library,
        external_rows,
        &bus_ids,
        &derived_load_models,
        stages,
    );
    validate_external_library(
        &library,
        &bus_ids,
        &row_entity_ids,
        &rows_per_stage,
        n_stages,
        forward_passes,
    )
    .map_err(SddpError::Stochastic)?;
    pad_library_to_uniform(&mut library);
    Ok(library)
}

/// Build and validate an [`ExternalScenarioLibrary`] for NCS.
///
/// Uses canonical NCS ID list from `ncs_models` (all NCS entities, sorted and deduped).
///
/// # Errors
///
/// Returns `SddpError::Stochastic` on validation failure.
pub(crate) fn build_external_ncs_library(
    external_rows: &[ExternalNcsRow],
    ncs_models: &[NcsModel],
    stages: &[Stage],
    forward_passes: u32,
) -> Result<ExternalScenarioLibrary, SddpError> {
    let n_stages = stages.len();
    let mut ncs_ids: Vec<EntityId> = ncs_models.iter().map(|m| m.ncs_id).collect();
    ncs_ids.sort_unstable_by_key(|id| id.0);
    ncs_ids.dedup();
    let n_ncs = ncs_ids.len();
    let row_entity_ids: std::collections::HashSet<EntityId> =
        external_rows.iter().map(|r| r.ncs_id).collect();
    let resolver =
        StageIdResolver::from_study_stage_ids(&stages.iter().map(|s| s.id).collect::<Vec<_>>());
    let mut rows_per_stage = vec![0usize; n_stages];
    for row in external_rows {
        if let Some(idx) = resolver.resolve(row.stage_id) {
            rows_per_stage[idx] += 1;
        }
    }
    let per_stage_scenarios: Vec<usize> = if n_ncs > 0 {
        rows_per_stage.iter().map(|&r| r / n_ncs).collect()
    } else {
        vec![0usize; n_stages]
    };
    let n_scenarios_ext = per_stage_scenarios.iter().copied().max().unwrap_or(0);
    let mut library =
        ExternalScenarioLibrary::new(n_stages, n_scenarios_ext, n_ncs, "ncs", per_stage_scenarios);
    let derived_ncs_models = external_derived_models(
        external_rows,
        &ncs_ids,
        stages,
        &resolver,
        |row| (row.ncs_id, row.stage_id, row.scenario_id, row.value),
        |ncs_id, stage_id, mean, std| NcsModel {
            ncs_id,
            stage_id,
            mean,
            std,
        },
    );
    standardize_external_ncs(
        &mut library,
        external_rows,
        &ncs_ids,
        &derived_ncs_models,
        stages,
    );
    validate_external_library(
        &library,
        &ncs_ids,
        &row_entity_ids,
        &rows_per_stage,
        n_stages,
        forward_passes,
    )
    .map_err(SddpError::Stochastic)?;
    pad_library_to_uniform(&mut library);
    Ok(library)
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::{
        Block, BlockMode, InflowModel, NoiseMethod, ScenarioSourceConfig, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_stochastic::StochasticError;

    use super::{
        EntityId, ExternalLoadRow, ExternalScenarioRow, LoadModel, PrecomputedPar, SamplingScheme,
        SddpError, Stage, StageLagTransition, build_external_inflow_library,
        build_external_load_library, derive_external_sample_moments,
    };

    fn single_stage(id: i32) -> Stage {
        Stage {
            index: 0,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn finalizing_transition() -> StageLagTransition {
        StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: true,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: false,
        }
    }

    /// A `sigma=0` hydro (deterministic PAR) whose external row does not match
    /// the deterministic value trips `solve_par_noise`'s `NEG_INFINITY`
    /// sentinel. `build_external_inflow_library` must surface it as a V3.7
    /// rejection through the real `standardize`-then-`validate` wiring, not
    /// silently accept it (the wiring previously ran `validate` against the
    /// still-zero-filled buffer, before `standardize_external_inflow` ever
    /// wrote eta, so V3.7 could never fire).
    #[test]
    fn external_inflow_sigma_zero_mismatch_rejected_by_v3_7() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];
        let stages = vec![single_stage(0)];

        let models = vec![InflowModel {
            hydro_id,
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        }];
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids, None).unwrap();

        let rows = vec![ExternalScenarioRow {
            stage_id: 0,
            scenario_id: 0,
            hydro_id,
            value_m3s: 999.0,
        }];
        let transitions = vec![finalizing_transition()];

        let result = build_external_inflow_library(
            &rows,
            &hydro_ids,
            &stages,
            &par,
            &[],
            0,
            &[],
            &[],
            &transitions,
            1,
            0,
        );

        match result {
            Err(SddpError::Stochastic(StochasticError::InsufficientData { context })) => {
                assert!(
                    context.contains("V3.7"),
                    "expected a V3.7 rejection, got: {context}"
                );
            }
            other => panic!("expected a V3.7 rejection, got: {other:?}"),
        }
    }

    /// Negative control for the test above: when the external row matches the
    /// deterministic value exactly, `solve_par_noise` returns `0.0` (not the
    /// `NEG_INFINITY` sentinel) and the library builds successfully.
    #[test]
    fn external_inflow_sigma_zero_match_accepted() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];
        let stages = vec![single_stage(0)];

        let models = vec![InflowModel {
            hydro_id,
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        }];
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids, None).unwrap();

        let rows = vec![ExternalScenarioRow {
            stage_id: 0,
            scenario_id: 0,
            hydro_id,
            value_m3s: 100.0,
        }];
        let transitions = vec![finalizing_transition()];

        let result = build_external_inflow_library(
            &rows,
            &hydro_ids,
            &stages,
            &par,
            &[],
            0,
            &[],
            &[],
            &transitions,
            1,
            0,
        );

        assert!(result.is_ok(), "expected Ok(()), got: {result:?}");
    }

    /// AC1: a σ=0 External LOAD bus reconstructs the external value, not a
    /// deliberately-disagreeing seasonal `mean_mw` (design doc §3/§4.1).
    #[test]
    fn external_load_sigma_zero_reconstructs_external_value_not_seasonal_mean() {
        let bus_id = EntityId(1);
        let stages = vec![single_stage(0)];
        let seasonal_load_models = vec![LoadModel {
            bus_id,
            stage_id: 0,
            mean_mw: 999.0,
            std_mw: 0.0,
        }];
        let external_rows = vec![ExternalLoadRow {
            bus_id,
            stage_id: 0,
            scenario_id: 0,
            value_mw: 123.0,
        }];

        let library = build_external_load_library(
            &external_rows,
            &seasonal_load_models,
            SamplingScheme::External,
            &stages,
            1,
        )
        .expect("a single-column external load library must build");

        let moments = derive_external_sample_moments(
            &external_rows,
            &[bus_id],
            1,
            |row: &ExternalLoadRow| (row.bus_id, row.stage_id, row.scenario_id, row.value_mw),
        );
        let (mean, std) = moments[0];
        assert!(
            (mean - 123.0).abs() < 1e-10,
            "the standardization moment source must be the external sample, not the \
             seasonal mean_mw (999.0)"
        );
        assert!(
            std.abs() < 1e-10,
            "a single external sample -> sigma=0 exactly"
        );

        let eta = library.eta_slice(0, 0)[0];
        let realized = (mean + std * eta).max(0.0);
        assert!(
            (realized - 123.0).abs() < 1e-10,
            "reconstruction must equal the external value (123.0), not the seasonal \
             mean_mw (999.0); got {realized}"
        );
    }

    /// AC2 + regression proof: a σ=0 External AR(0) inflow reconstructs the
    /// external value, including at a stage whose real scenario count is
    /// SMALLER than another stage's — a branching root's sole observation,
    /// the non-uniform shape the V3.7-vs-padding-order fix (design doc
    /// §3/§4.2) must not reject — and the padded phantom slot replicates the
    /// correct value, not a rejected/garbage sentinel.
    #[test]
    fn external_inflow_ar0_nonuniform_scenario_counts_reconstructs_external_values() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];
        let stages = vec![single_stage(0), single_stage(1)];

        // Stage 0: a single real external scenario (sigma derives to 0
        // exactly). Stage 1: two distinct real scenarios (sigma > 0).
        let external_rows = vec![
            ExternalScenarioRow {
                stage_id: 0,
                scenario_id: 0,
                hydro_id,
                value_m3s: 60.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 0,
                hydro_id,
                value_m3s: 10.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 1,
                hydro_id,
                value_m3s: 30.0,
            },
        ];

        // Mirror context.rs's AR(0) override: derive moments over the
        // external samples and rebuild PrecomputedPar from them.
        let moments = derive_external_sample_moments(
            &external_rows,
            &hydro_ids,
            2,
            |row: &ExternalScenarioRow| {
                (row.hydro_id, row.stage_id, row.scenario_id, row.value_m3s)
            },
        );
        let overridden_models: Vec<InflowModel> = (0..2_usize)
            .map(|s| {
                let (mean_m3s, std_m3s) = moments[s];
                InflowModel {
                    hydro_id,
                    stage_id: i32::try_from(s).unwrap(),
                    mean_m3s,
                    std_m3s,
                    ar_coefficients: vec![],
                    residual_std_ratio: 1.0,
                    annual: None,
                }
            })
            .collect();
        let par = PrecomputedPar::build(&overridden_models, &stages, &hydro_ids, None).unwrap();
        assert!(
            par.sigma(0, 0).abs() < 1e-10,
            "stage 0's single real scenario must derive sigma=0 exactly"
        );
        assert!(
            par.sigma(1, 0) > 0.0,
            "stage 1's two distinct real scenarios must derive sigma>0"
        );

        let transitions = vec![finalizing_transition(), finalizing_transition()];
        let library = build_external_inflow_library(
            &external_rows,
            &hydro_ids,
            &stages,
            &par,
            &[],
            0,
            &[],
            &[],
            &transitions,
            2,
            0,
        )
        .expect("V3.7 must not reject stage 0 for having fewer real scenarios than stage 1");

        let reconstruct = |stage: usize, scenario: usize| {
            let eta = library.eta_slice(stage, scenario)[0];
            par.deterministic_base(stage, 0) + par.sigma(stage, 0) * eta
        };

        assert!(
            (reconstruct(0, 0) - 60.0).abs() < 1e-10,
            "stage 0's real scenario must reconstruct the external value"
        );
        assert!(
            (reconstruct(1, 0) - 10.0).abs() < 1e-10,
            "stage 1's first real scenario must reconstruct the external value"
        );
        assert!(
            (reconstruct(1, 1) - 30.0).abs() < 1e-10,
            "stage 1's second real scenario must reconstruct the external value"
        );
        assert!(
            (reconstruct(0, 1) - 60.0).abs() < 1e-10,
            "the padded phantom slot at stage 0 must replicate the real root value \
             (60.0), not a rejected/garbage sentinel"
        );
    }

    /// Regression pin: an AR(p > 0) hydro in a non-uniform-per-stage External
    /// deck — stage 0 has one real scenario, a later stage has k — must NOT
    /// have a real later-stage slot's stored eta inverted against a
    /// fabricated phantom lag (`0.0`) instead of the lag the runtime forward
    /// pass actually feeds: every stage-1 branch descends from the SAME
    /// stage-0 root, so `accumulate_and_shift_lag_state`
    /// (`stochastic/noise.rs`) carries that root's own realized inflow to
    /// every child regardless of `scenario_id`. `standardize_external_inflow`
    /// (`sampling/external.rs`) replicates a stage's real raw values to
    /// uniform width BEFORE `run_eta_inversion`'s lag-chain advance for
    /// exactly this reason. `lag_realized` below is sourced independently of
    /// that fix: the branching root's own external value, recovered via this
    /// AR(1) hydro's own stage-0 stored eta (verified equal to the root's raw
    /// value) — never the lag the inversion used internally, which would mask
    /// a real bug were the fix ever weakened.
    #[test]
    fn external_inflow_ar1_nonuniform_scenario_counts_round_trip() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];
        let stages = vec![single_stage(0), single_stage(1)];

        // AR(1), untouched seasonal model: this ticket never overrides an
        // AR(p > 0) hydro's moments with derived external samples.
        let seasonal_model = |stage_id| InflowModel {
            hydro_id,
            stage_id,
            mean_m3s: 100.0,
            std_m3s: 30.0,
            ar_coefficients: vec![0.5],
            residual_std_ratio: 1.0,
            annual: None,
        };
        let seasonal_models = vec![seasonal_model(0), seasonal_model(1)];
        let par = PrecomputedPar::build(&seasonal_models, &stages, &hydro_ids, None).unwrap();
        let det_base = par.deterministic_base(0, 0);
        let psi = par.psi_slice(0, 0)[0];
        let sigma = par.sigma(0, 0);

        // Stage 0: a single real external scenario (the branching root).
        // Stage 1: two distinct real branches, both descending from that SAME
        // root.
        let external_rows = vec![
            ExternalScenarioRow {
                stage_id: 0,
                scenario_id: 0,
                hydro_id,
                value_m3s: 200.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 0,
                hydro_id,
                value_m3s: 150.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 1,
                hydro_id,
                value_m3s: 250.0,
            },
        ];
        let derived_lag_values = [0.0_f64];
        let transitions = vec![finalizing_transition(), finalizing_transition()];

        let library = build_external_inflow_library(
            &external_rows,
            &hydro_ids,
            &stages,
            &par,
            &derived_lag_values,
            1,
            &[],
            &[],
            &transitions,
            2,
            0,
        )
        .expect("V3.7 must not reject stage 0 for having fewer real scenarios than stage 1");

        let stage0_eta_real = library.eta_slice(0, 0)[0];
        let lag_realized = det_base + psi * derived_lag_values[0] + sigma * stage0_eta_real;
        assert!(
            (lag_realized - 200.0).abs() < 1e-10,
            "the branching root's own realized inflow must equal its external value; \
             got {lag_realized}"
        );

        let reconstruct_stage1 = |scenario: usize| {
            let eta = library.eta_slice(1, scenario)[0];
            det_base + psi * lag_realized + sigma * eta
        };

        let external_stage1 = [150.0_f64, 250.0_f64];
        for (scenario, &expected) in external_stage1.iter().enumerate() {
            let realized = reconstruct_stage1(scenario);
            assert!(
                (realized - expected).abs() < 1e-6,
                "stage 1 scenario {scenario}: reconstructed {realized}, expected {expected} \
                 (external value) using the REAL parent lag {lag_realized}"
            );
        }
    }

    /// Epic-02 review regression: a gapped/non-0-based External LOAD deck
    /// (declared stage ids `2`/`5`, never `0`/`1`) must resolve through the
    /// same canonical `stage_id -> index` mapping cobre-io's rule-47
    /// validator uses. Pre-fix, `rows_per_stage`'s `row.stage_id as usize`
    /// bound-check (`< n_stages`) silently dropped every row of a gapped
    /// deck outright, since the raw ids (2, 5) both exceed `n_stages` (2).
    #[test]
    fn external_load_library_resolves_gapped_stage_ids() {
        let bus_id = EntityId(1);
        let stages = vec![single_stage(2), single_stage(5)];
        let seasonal_load_models = vec![
            LoadModel {
                bus_id,
                stage_id: 2,
                mean_mw: 999.0,
                std_mw: 0.0,
            },
            LoadModel {
                bus_id,
                stage_id: 5,
                mean_mw: 999.0,
                std_mw: 0.0,
            },
        ];
        let external_rows = vec![
            ExternalLoadRow {
                bus_id,
                stage_id: 2,
                scenario_id: 0,
                value_mw: 123.0,
            },
            ExternalLoadRow {
                bus_id,
                stage_id: 5,
                scenario_id: 0,
                value_mw: 456.0,
            },
        ];

        let library = build_external_load_library(
            &external_rows,
            &seasonal_load_models,
            SamplingScheme::External,
            &stages,
            1,
        )
        .expect("a gapped-stage-id external load deck must build, not drop every row");

        // Resolve the declared ids the same way the fixed engine now must:
        // gapped id 2 -> canonical position 0, gapped id 5 -> position 1.
        let resolved_rows: Vec<(EntityId, i32, i32, f64)> = external_rows
            .iter()
            .map(|row| {
                let resolved = i32::from(row.stage_id == 5);
                (row.bus_id, resolved, row.scenario_id, row.value_mw)
            })
            .collect();
        let moments = derive_external_sample_moments(
            &resolved_rows,
            &[bus_id],
            2,
            |&(bus, stage_idx, scenario_id, value)| (bus, stage_idx, scenario_id, value),
        );

        for (resolved_idx, expected) in [(0usize, 123.0_f64), (1usize, 456.0_f64)] {
            let (mean, std) = moments[resolved_idx];
            let eta = library.eta_slice(resolved_idx, 0)[0];
            let realized = (mean + std * eta).max(0.0);
            assert!(
                (realized - expected).abs() < 1e-10,
                "gapped declared stage id must resolve to canonical position {resolved_idx}; \
                 got {realized}, expected {expected}"
            );
        }
    }

    /// Epic-02 review regression, inflow counterpart: a gapped/non-0-based
    /// External INFLOW deck (declared stage ids `2`/`5`) must resolve
    /// through the same canonical mapping — both in the `rows_per_stage`
    /// count feeding V3.7 and in `standardize_external_inflow`'s own
    /// per-stage fill. Pre-fix, both silently dropped every row of this
    /// deck, since the raw ids exceed `n_stages` (2).
    #[test]
    fn external_inflow_library_resolves_gapped_stage_ids() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];
        let stages = vec![single_stage(2), single_stage(5)];

        let external_rows = vec![
            ExternalScenarioRow {
                stage_id: 2,
                scenario_id: 0,
                hydro_id,
                value_m3s: 200.0,
            },
            ExternalScenarioRow {
                stage_id: 5,
                scenario_id: 0,
                hydro_id,
                value_m3s: 400.0,
            },
        ];

        // Mirror context.rs's AR(0) override: derive moments over the
        // RESOLVED (canonical) stage positions, the same mapping the engine
        // and the validator must both apply to the raw declared ids (2, 5).
        let resolved_rows: Vec<(EntityId, i32, i32, f64)> = external_rows
            .iter()
            .map(|row| {
                let resolved = i32::from(row.stage_id == 5);
                (row.hydro_id, resolved, row.scenario_id, row.value_m3s)
            })
            .collect();
        let moments = derive_external_sample_moments(
            &resolved_rows,
            &hydro_ids,
            2,
            |&(hydro, stage_idx, scenario_id, value)| (hydro, stage_idx, scenario_id, value),
        );
        let overridden_models: Vec<InflowModel> = stages
            .iter()
            .enumerate()
            .map(|(idx, stage)| {
                let (mean_m3s, std_m3s) = moments[idx];
                InflowModel {
                    hydro_id,
                    stage_id: stage.id,
                    mean_m3s,
                    std_m3s,
                    ar_coefficients: vec![],
                    residual_std_ratio: 1.0,
                    annual: None,
                }
            })
            .collect();
        let par = PrecomputedPar::build(&overridden_models, &stages, &hydro_ids, None).unwrap();
        assert!(par.sigma(0, 0).abs() < 1e-10);
        assert!(par.sigma(1, 0).abs() < 1e-10);

        let transitions = vec![finalizing_transition(), finalizing_transition()];
        let library = build_external_inflow_library(
            &external_rows,
            &hydro_ids,
            &stages,
            &par,
            &[],
            0,
            &[],
            &[],
            &transitions,
            1,
            0,
        )
        .expect("a gapped-stage-id external inflow deck must build, not drop every row");

        let reconstruct = |stage: usize| {
            let eta = library.eta_slice(stage, 0)[0];
            par.deterministic_base(stage, 0) + par.sigma(stage, 0) * eta
        };
        assert!(
            (reconstruct(0) - 200.0).abs() < 1e-10,
            "gapped declared id 2 must resolve to canonical position 0"
        );
        assert!(
            (reconstruct(1) - 400.0).abs() < 1e-10,
            "gapped declared id 5 must resolve to canonical position 1"
        );
    }
}
