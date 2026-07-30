//! Scenario library builders for historical and external sampling schemes.
//!
//! Builders are not factored generically because external types have different
//! standardization semantics.

use cobre_core::{
    EntityId, InflowHistoryRow, Stage,
    scenario::{
        ExternalLoadRow, ExternalNcsRow, ExternalScenarioRow, HistoricalYears, LoadModel, NcsModel,
    },
    temporal::{SeasonMap, StageLagTransition},
};
use cobre_stochastic::{
    ExternalScenarioLibrary, HistoricalScenarioLibrary, PrecomputedPar,
    discover_historical_windows, pad_library_to_uniform, standardize_external_inflow,
    standardize_external_load, standardize_external_ncs, standardize_historical_windows,
    validate_external_library, validate_historical_library,
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
    let mut rows_per_stage = vec![0usize; n_stages];
    #[allow(clippy::cast_sign_loss)]
    for row in external_rows {
        let s = row.stage_id as usize;
        if s < n_stages {
            rows_per_stage[s] += 1;
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

/// Build and validate an [`ExternalScenarioLibrary`] for load.
///
/// Uses canonical bus ID list from `load_models` (buses with `std_mw > 0.0`).
///
/// # Errors
///
/// Returns `SddpError::Stochastic` on validation failure.
pub(crate) fn build_external_load_library(
    external_rows: &[ExternalLoadRow],
    load_models: &[LoadModel],
    stages: &[Stage],
    forward_passes: u32,
) -> Result<ExternalScenarioLibrary, SddpError> {
    let n_stages = stages.len();
    let mut bus_ids: Vec<EntityId> = load_models
        .iter()
        .filter(|m| m.std_mw > 0.0)
        .map(|m| m.bus_id)
        .collect();
    bus_ids.sort_unstable_by_key(|id| id.0);
    bus_ids.dedup();
    let n_buses = bus_ids.len();
    let row_entity_ids: std::collections::HashSet<EntityId> =
        external_rows.iter().map(|r| r.bus_id).collect();
    let mut rows_per_stage = vec![0usize; n_stages];
    #[allow(clippy::cast_sign_loss)]
    for row in external_rows {
        let s = row.stage_id as usize;
        if s < n_stages {
            rows_per_stage[s] += 1;
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
    standardize_external_load(&mut library, external_rows, &bus_ids, load_models, n_stages);
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
    let mut rows_per_stage = vec![0usize; n_stages];
    #[allow(clippy::cast_sign_loss)]
    for row in external_rows {
        let s = row.stage_id as usize;
        if s < n_stages {
            rows_per_stage[s] += 1;
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
    standardize_external_ncs(&mut library, external_rows, &ncs_ids, ncs_models, n_stages);
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
        EntityId, ExternalScenarioRow, PrecomputedPar, SddpError, Stage, StageLagTransition,
        build_external_inflow_library,
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
}
