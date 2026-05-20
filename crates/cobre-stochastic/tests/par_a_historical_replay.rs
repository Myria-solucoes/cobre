#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

//! Regression test: PAR(p)-A annual + historical replay must reproduce the
//! historical observation exactly when the stage-0 lag state equals the
//! window's pre-study lags.
//!
//! ## Bug context
//!
//! When `PrecomputedPar` materialises a model with `annual: Some(_)`, the
//! `psi` slice is widened to 12 and the standardised annual coefficient ψ̂ is
//! spread across the extra positions as `psi_hat / 12`. The LP layout iterates
//! the full slice, but a prior version of the PAR primitives accepted an
//! explicit `order` parameter and iterated only `psi[0..order]`. That made
//! `solve_par_noise` silently miss the annual contribution at standardisation
//! time, and forward replays with PAR(p)-A active diverged from the raw
//! historical observation even when the initial lag state matched the window's
//! pre-study lags exactly (about 11% on the convertido / NEWAVE 1983-anchored
//! comparison case — see `historical.rs` and `evaluate.rs` for details).
//!
//! This test pins the invariant directly: standardise a historical library
//! with PAR-A on, then reconstruct each study stage with the matched lag
//! state, and require bit-level agreement with the original observation.

use chrono::NaiveDate;
use cobre_core::{
    EntityId,
    scenario::{AnnualComponent, InflowHistoryRow, InflowModel},
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_stochastic::{
    HistoricalScenarioLibrary, evaluate_par_batch, par::precompute::PrecomputedPar,
    standardize_historical_windows,
};

fn monthly_stage(index: usize, season_id: usize) -> Stage {
    let month = u32::try_from(season_id).expect("season_id fits in u32") + 1;
    Stage {
        index,
        id: i32::try_from(index).expect("index fits in i32"),
        start_date: NaiveDate::from_ymd_opt(2024, month, 1).expect("valid date"),
        end_date: NaiveDate::from_ymd_opt(2024, month, 28).expect("valid date"),
        season_id: Some(season_id),
        blocks: vec![Block {
            index: 0,
            name: "SINGLE".to_string(),
            duration_hours: 720.0,
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

fn row(hydro_id: EntityId, year: i32, month0: u32, value: f64) -> InflowHistoryRow {
    InflowHistoryRow {
        hydro_id,
        date: NaiveDate::from_ymd_opt(year, month0 + 1, 1).expect("valid date"),
        value_m3s: value,
    }
}

/// AR(1) + PAR(p)-A annual on a single hydro across 12 monthly stages with
/// one historical window must reproduce each stage's observation exactly when
/// the LP lag state at stage 0 equals the window's pre-study lags.
#[test]
fn par_a_historical_replay_roundtrip() {
    let hydro = EntityId(1);
    let stages: Vec<Stage> = (0..12).map(|i| monthly_stage(i, i)).collect();

    // Build models: AR(1) with annual component on every study stage, plus the
    // 12 pre-study stages (-12..0) so the lag-stage stats lookup resolves
    // without falling through to season-only stats.
    let annual = AnnualComponent {
        coefficient: -0.18,
        mean_m3s: 95.0,
        std_m3s: 22.0,
    };
    let mut all_models: Vec<InflowModel> = (-12_i32..0)
        .map(|sid| InflowModel {
            hydro_id: hydro,
            stage_id: sid,
            mean_m3s: 90.0 + f64::from(sid).abs() * 0.5,
            std_m3s: 20.0,
            ar_coefficients: vec![0.4],
            residual_std_ratio: 0.85,
            annual: Some(annual.clone()),
        })
        .collect();
    all_models.extend((0_i32..12).map(|sid| InflowModel {
        hydro_id: hydro,
        stage_id: sid,
        mean_m3s: 100.0 + f64::from(sid) * 5.0,
        std_m3s: 25.0 + f64::from(sid) * 1.0,
        ar_coefficients: vec![0.4],
        residual_std_ratio: 0.85,
        annual: Some(annual.clone()),
    }));
    let par = PrecomputedPar::build(&all_models, &stages, &[hydro]).expect("PAR build");

    // PAR-A widens psi stride to 12. Confirm we are exercising the buggy code path.
    assert_eq!(
        par.max_order(),
        12,
        "PAR-A must widen max_order to 12; test does not exercise the fix otherwise"
    );

    // Build the inflow history: 24 months covering the pre-window lags
    // (1989, all 12 months) plus the study window (1990, all 12 months).
    // The observations are arbitrary but distinct so a roundtrip miss is loud.
    let window_year: i32 = 1990;
    let pre_window_year: i32 = window_year - 1;
    let history_values: Vec<f64> = (0..24).map(|i| 80.0 + 7.5 * f64::from(i)).collect();
    let mut history: Vec<InflowHistoryRow> = Vec::with_capacity(24);
    for m in 0..12_u32 {
        history.push(row(hydro, pre_window_year, m, history_values[m as usize]));
    }
    for m in 0..12_u32 {
        history.push(row(hydro, window_year, m, history_values[12 + m as usize]));
    }

    // Standardise the historical library for this single window.
    let mut library = HistoricalScenarioLibrary::new(1, 12, 1, par.max_order(), vec![window_year]);
    standardize_historical_windows(
        &mut library,
        &history,
        &[hydro],
        &stages,
        &par,
        &[window_year],
        None,
    );

    // Initial lag state at stage 0 == window's pre-study lags. With matched
    // lags, evaluate_par_batch(η, lags) must reproduce the original target.
    //
    // Lag layout in lag_matrix: [lag * n_hydros + hydro]; lag 0 is most recent.
    let max_order = par.max_order();
    let n_hydros = 1;
    let mut lag_matrix: Vec<f64> = library.lag_slice(0).to_vec();
    assert_eq!(lag_matrix.len(), max_order * n_hydros);

    let mut reconstructed = vec![0.0_f64];
    for t in 0..12 {
        let eta = library.eta_slice(0, t);
        evaluate_par_batch(&par, t, &lag_matrix, eta, &mut reconstructed);

        let target = history_values[12 + t];
        let got = reconstructed[0];
        assert!(
            (got - target).abs() < 1e-9,
            "stage {t}: reconstructed {got:.12} != target {target:.12} \
             (diff {:.3e}) — PAR-A annual contribution likely dropped \
             from η at standardisation time",
            (got - target).abs(),
        );

        // Advance the lag state: shift older lags back one slot, insert this
        // stage's realised inflow as the new lag-1.
        for lag in (1..max_order).rev() {
            lag_matrix[lag * n_hydros] = lag_matrix[(lag - 1) * n_hydros];
        }
        lag_matrix[0] = got;
    }
}
