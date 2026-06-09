#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

//! Regression tests: PAR(p)-A annual + historical replay must reproduce the
//! historical observation exactly when the stage-0 lag state equals
//! `past_inflows` (the rolling-chain seed).
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
//! A second bug: `standardize_historical_windows` was inverting η against the
//! window's own pre-study lags rather than `past_inflows`. This caused an
//! algebraic offset whenever the two differed (see `historical.rs` module doc
//! and dc96030). The fix re-roots the chain at `past_inflows`.
//!
//! ## Tests
//!
//! | ID | Description |
//! |----|-------------|
//! | T1 | AR(1)+PAR-A, `past_inflows` == window pre-study lags (original regression) |
//! | T2 | AR(1)+PAR-A, `past_inflows` deliberately differ from window pre-study lags |
//! | T3 | AR(0), `past_inflows` == window lags (no-AR regression baseline) |
//! | T4 | AR(1), `past_inflows` has fewer entries than `max_order` (absent hydros = 0) |
//! | T5 | Two windows, `past_inflows` shared; both windows replay exactly |
//! | T6 | Non-trivial `stage_lag_transitions` guard fires in debug builds |

use chrono::NaiveDate;
use cobre_core::{
    EntityId, HydroPastInflows,
    scenario::{AnnualComponent, InflowHistoryRow, InflowModel},
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageLagTransition,
        StageRiskConfig, StageStateConfig,
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

/// Build a uniform-monthly `StageLagTransition` (finalise every stage).
fn uniform_monthly_transition() -> StageLagTransition {
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

/// Advance `lag_state` (h-major: `lag_state[h * safe_max_order + l]`) by one
/// uniform-monthly step using the realised value for `n_hydros == 1`.
fn advance_lag_state_uniform(lag_state: &mut [f64], realised: f64, max_order: usize) {
    let safe = max_order.max(1);
    for l in (1..safe).rev() {
        lag_state[l] = lag_state[l - 1];
    }
    lag_state[0] = realised;
}

// ---------------------------------------------------------------------------
// T1: AR(1)+PAR-A, past_inflows == window pre-study lags (original regression)
// ---------------------------------------------------------------------------

/// AR(1) + PAR(p)-A annual on a single hydro across 12 monthly stages with
/// one historical window must reproduce each stage's observation exactly when
/// `past_inflows` equals the window's own pre-study lags.
///
/// This is the original regression test, now adapted to pass `past_inflows`
/// matching the window antecedent lags so that η is inverted against the same
/// x₀ that the forward pass starts from.
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
    let par = PrecomputedPar::build(&all_models, &stages, &[hydro], None).expect("PAR build");

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

    // past_inflows: the window's own pre-study observations in lag order
    // (lag-1 first = Dec 1989 = history_values[11], ..., lag-12 = Jan 1989 = history_values[0]).
    let max_order = par.max_order(); // == 12
    let past_values_m3s: Vec<f64> = (0..max_order)
        .map(|lag| history_values[11 - lag]) // lag-1=Dec, lag-2=Nov, ...
        .collect();
    let past_inflows = vec![HydroPastInflows {
        hydro_id: hydro,
        values_m3s: past_values_m3s.clone(),
        season_ids: None,
    }];
    let transitions: Vec<StageLagTransition> =
        (0..12).map(|_| uniform_monthly_transition()).collect();

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
        &past_inflows,
        &transitions,
    );

    // Seed lag_state from past_inflows (h-major layout).
    let n_hydros = 1;
    let safe_max_order = max_order.max(1);
    let mut lag_state = vec![0.0_f64; n_hydros * safe_max_order];
    for (lag, &v) in past_values_m3s.iter().enumerate().take(max_order) {
        lag_state[lag] = v;
    }

    // For evaluate_par_batch the lag_matrix layout is [lag * n_hydros + hydro].
    // With n_hydros=1, lag_state and lag_matrix are identical layouts.
    let mut reconstructed = vec![0.0_f64];
    for t in 0..12 {
        let eta = library.eta_slice(0, t);
        evaluate_par_batch(&par, t, &lag_state, eta, &mut reconstructed);

        let target = history_values[12 + t];
        let got = reconstructed[0];
        assert!(
            (got - target).abs() < 1e-9,
            "T1 stage {t}: reconstructed {got:.12} != target {target:.12} \
             (diff {:.3e}) — PAR-A annual contribution likely dropped \
             from η at standardisation time",
            (got - target).abs(),
        );

        advance_lag_state_uniform(&mut lag_state, got, max_order);
    }
}

// ---------------------------------------------------------------------------
// T2: AR(1)+PAR-A, past_inflows deliberately differ from window pre-study lags
// ---------------------------------------------------------------------------

/// When `past_inflows` differ from the window's own historical antecedent lags,
/// the η values are inverted against `past_inflows` (not the window lags). A
/// forward pass starting from `past_inflows` must still exactly reproduce the
/// original target at every stage, while a forward pass starting from the window
/// lags would diverge.
///
/// This directly tests the algebraic correctness of the re-rooting fix.
#[test]
fn t2_past_inflows_differ_from_window_lags_roundtrip() {
    let hydro = EntityId(1);
    let stages: Vec<Stage> = (0..12).map(|i| monthly_stage(i, i)).collect();

    let annual = AnnualComponent {
        coefficient: -0.18,
        mean_m3s: 95.0,
        std_m3s: 22.0,
    };
    let mut all_models: Vec<InflowModel> = (-12_i32..0)
        .map(|sid| InflowModel {
            hydro_id: hydro,
            stage_id: sid,
            mean_m3s: 90.0,
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
    let par = PrecomputedPar::build(&all_models, &stages, &[hydro], None).expect("PAR build");
    let max_order = par.max_order();
    assert_eq!(max_order, 12);

    let window_year: i32 = 1990;
    let pre_window_year: i32 = window_year - 1;
    // Window lags: 1989 observations (arbitrary values).
    let window_lag_values: Vec<f64> = (0..12).map(|i| 80.0 + 3.0 * f64::from(i)).collect();
    // Study values: 1990 observations.
    let study_values: Vec<f64> = (0..12).map(|i| 150.0 + 5.0 * f64::from(i)).collect();

    let mut history: Vec<InflowHistoryRow> = Vec::with_capacity(24);
    for m in 0..12_u32 {
        history.push(row(
            hydro,
            pre_window_year,
            m,
            window_lag_values[m as usize],
        ));
    }
    for m in 0..12_u32 {
        history.push(row(hydro, window_year, m, study_values[m as usize]));
    }

    // past_inflows deliberately different from window lags: use a flat 200.0.
    let past_values_m3s: Vec<f64> = vec![200.0; max_order];
    let past_inflows = vec![HydroPastInflows {
        hydro_id: hydro,
        values_m3s: past_values_m3s.clone(),
        season_ids: None,
    }];
    let transitions: Vec<StageLagTransition> =
        (0..12).map(|_| uniform_monthly_transition()).collect();

    let mut library = HistoricalScenarioLibrary::new(1, 12, 1, max_order, vec![window_year]);
    standardize_historical_windows(
        &mut library,
        &history,
        &[hydro],
        &stages,
        &par,
        &[window_year],
        None,
        &past_inflows,
        &transitions,
    );

    // Replay from past_inflows — must reproduce study_values exactly.
    let safe_max_order = max_order.max(1);
    let mut lag_state = vec![0.0_f64; safe_max_order];
    for (lag, &v) in past_values_m3s.iter().enumerate().take(max_order) {
        lag_state[lag] = v;
    }

    let mut reconstructed = vec![0.0_f64];
    for (t, &target) in study_values.iter().enumerate() {
        let eta = library.eta_slice(0, t);
        evaluate_par_batch(&par, t, &lag_state, eta, &mut reconstructed);

        let got = reconstructed[0];
        assert!(
            (got - target).abs() < 1e-9,
            "T2 stage {t}: replay from past_inflows: reconstructed {got:.12} != \
             target {target:.12} (diff {:.3e})",
            (got - target).abs(),
        );

        advance_lag_state_uniform(&mut lag_state, got, max_order);
    }
}

// ---------------------------------------------------------------------------
// T3: AR(0), past_inflows == window lags (no-AR regression baseline)
// ---------------------------------------------------------------------------

/// AR(0) with PAR-A annual, `past_inflows` matching window lags. Max order is
/// 12 due to PAR-A. With zero AR coefficient, `η = (obs - det_base) / sigma`;
/// replay from `past_inflows` must reproduce each observation exactly.
#[test]
fn t3_ar0_par_a_roundtrip() {
    let hydro = EntityId(1);
    let stages: Vec<Stage> = (0..12).map(|i| monthly_stage(i, i)).collect();

    let annual = AnnualComponent {
        coefficient: -0.18,
        mean_m3s: 95.0,
        std_m3s: 22.0,
    };
    // AR(0): ar_coefficients empty; annual widens max_order to 12.
    let mut all_models: Vec<InflowModel> = (-12_i32..0)
        .map(|sid| InflowModel {
            hydro_id: hydro,
            stage_id: sid,
            mean_m3s: 90.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: Some(annual.clone()),
        })
        .collect();
    all_models.extend((0_i32..12).map(|sid| InflowModel {
        hydro_id: hydro,
        stage_id: sid,
        mean_m3s: 100.0 + f64::from(sid) * 5.0,
        std_m3s: 25.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: Some(annual.clone()),
    }));
    let par = PrecomputedPar::build(&all_models, &stages, &[hydro], None).expect("PAR build");
    let max_order = par.max_order();
    assert_eq!(max_order, 12, "PAR-A must widen max_order to 12");

    let window_year: i32 = 1990;
    let pre_window_year: i32 = window_year - 1;
    let pre_values: Vec<f64> = (0..12).map(|i| 70.0 + 4.0 * f64::from(i)).collect();
    let study_values: Vec<f64> = (0..12).map(|i| 130.0 + 6.0 * f64::from(i)).collect();

    let mut history: Vec<InflowHistoryRow> = Vec::with_capacity(24);
    for m in 0..12_u32 {
        history.push(row(hydro, pre_window_year, m, pre_values[m as usize]));
    }
    for m in 0..12_u32 {
        history.push(row(hydro, window_year, m, study_values[m as usize]));
    }

    // past_inflows == window's own pre-study lags (lag-1=Dec 1989 first).
    let past_values_m3s: Vec<f64> = (0..max_order).map(|lag| pre_values[11 - lag]).collect();
    let past_inflows = vec![HydroPastInflows {
        hydro_id: hydro,
        values_m3s: past_values_m3s.clone(),
        season_ids: None,
    }];
    let transitions: Vec<StageLagTransition> =
        (0..12).map(|_| uniform_monthly_transition()).collect();

    let mut library = HistoricalScenarioLibrary::new(1, 12, 1, max_order, vec![window_year]);
    standardize_historical_windows(
        &mut library,
        &history,
        &[hydro],
        &stages,
        &par,
        &[window_year],
        None,
        &past_inflows,
        &transitions,
    );

    let safe_max_order = max_order.max(1);
    let mut lag_state = vec![0.0_f64; safe_max_order];
    for (lag, &v) in past_values_m3s.iter().enumerate().take(max_order) {
        lag_state[lag] = v;
    }

    let mut reconstructed = vec![0.0_f64];
    for (t, &target) in study_values.iter().enumerate() {
        let eta = library.eta_slice(0, t);
        evaluate_par_batch(&par, t, &lag_state, eta, &mut reconstructed);

        let got = reconstructed[0];
        assert!(
            (got - target).abs() < 1e-9,
            "T3 stage {t}: AR(0)+PAR-A reconstructed {got:.12} != target {target:.12} \
             (diff {:.3e})",
            (got - target).abs(),
        );

        advance_lag_state_uniform(&mut lag_state, got, max_order);
    }
}

// ---------------------------------------------------------------------------
// T4: AR(1), past_inflows has fewer entries than max_order (absent = 0)
// ---------------------------------------------------------------------------

/// When `past_inflows.values_m3s` is shorter than `max_order`, missing lag
/// slots default to `0.0`. Replay seeded from the same truncated `past_inflows`
/// must still exactly reproduce all study observations.
#[test]
fn t4_past_inflows_shorter_than_max_order_roundtrip() {
    let hydro = EntityId(1);
    // Use 4 stages, AR(2) (max_order == 2).
    let stages: Vec<Stage> = (0..4).map(|i| monthly_stage(i, i)).collect();

    // AR(2), no annual (max_order == 2).
    let mut all_models: Vec<InflowModel> = (-2_i32..0)
        .map(|sid| InflowModel {
            hydro_id: hydro,
            stage_id: sid,
            mean_m3s: 100.0,
            std_m3s: 20.0,
            ar_coefficients: vec![0.3, 0.15],
            residual_std_ratio: 0.9,
            annual: None,
        })
        .collect();
    all_models.extend((0_i32..4).map(|sid| InflowModel {
        hydro_id: hydro,
        stage_id: sid,
        mean_m3s: 110.0 + f64::from(sid) * 3.0,
        std_m3s: 20.0,
        ar_coefficients: vec![0.3, 0.15],
        residual_std_ratio: 0.9,
        annual: None,
    }));
    let par = PrecomputedPar::build(&all_models, &stages, &[hydro], None).expect("PAR build");
    assert_eq!(par.max_order(), 2);

    let window_year: i32 = 2000;
    // We need lags for the 2 months before the window.
    // Season ordering for 4-stage study starting at season 0: pre-study
    // seasons are 2 and 3 (Nov, Dec) of 1999.
    let pre_2 = 55.0_f64; // lag-2: Nov 1999
    let pre_1 = 66.0_f64; // lag-1: Dec 1999
    let study_values = [130.0_f64, 145.0, 120.0, 110.0];

    let mut history: Vec<InflowHistoryRow> = Vec::new();
    // Add the full year 1999 so the lags are available.
    for m in 0..12_u32 {
        let v = if m == 10 {
            pre_2
        } else if m == 11 {
            pre_1
        } else {
            100.0
        };
        history.push(row(hydro, 1999, m, v));
    }
    for (t, &v) in study_values.iter().enumerate() {
        history.push(row(
            hydro,
            window_year,
            u32::try_from(t).expect("t fits u32"),
            v,
        ));
    }

    // past_inflows: only 1 entry (lag-1 = 66.0); lag-2 defaults to 0.0.
    let past_inflows = vec![HydroPastInflows {
        hydro_id: hydro,
        values_m3s: vec![66.0], // shorter than max_order=2
        season_ids: None,
    }];
    let transitions: Vec<StageLagTransition> =
        (0..4).map(|_| uniform_monthly_transition()).collect();

    let mut library = HistoricalScenarioLibrary::new(1, 4, 1, par.max_order(), vec![window_year]);
    standardize_historical_windows(
        &mut library,
        &history,
        &[hydro],
        &stages,
        &par,
        &[window_year],
        None,
        &past_inflows,
        &transitions,
    );

    // Seed lag_state from the same truncated past_inflows (lag-2 = 0.0).
    let max_order = par.max_order();
    let safe = max_order.max(1);
    let mut lag_state = vec![0.0_f64; safe];
    lag_state[0] = 66.0; // lag-1
    // lag_state[1] == 0.0 (absent)

    let mut reconstructed = vec![0.0_f64];
    for (t, &target) in study_values.iter().enumerate() {
        let eta = library.eta_slice(0, t);
        evaluate_par_batch(&par, t, &lag_state, eta, &mut reconstructed);

        let got = reconstructed[0];
        assert!(
            (got - target).abs() < 1e-9,
            "T4 stage {t}: truncated past_inflows: reconstructed {got:.12} != \
             target {target:.12} (diff {:.3e})",
            (got - target).abs(),
        );

        advance_lag_state_uniform(&mut lag_state, got, max_order);
    }
}

// ---------------------------------------------------------------------------
// T5: Two windows, past_inflows shared; both windows replay exactly
// ---------------------------------------------------------------------------

/// With two historical windows and a shared `past_inflows`, both windows must
/// replay their respective targets exactly when the forward pass starts from
/// `past_inflows`.
#[test]
fn t5_two_windows_shared_past_inflows_roundtrip() {
    let hydro = EntityId(1);
    let stages: Vec<Stage> = (0..12).map(|i| monthly_stage(i, i)).collect();

    let annual = AnnualComponent {
        coefficient: -0.18,
        mean_m3s: 95.0,
        std_m3s: 22.0,
    };
    let mut all_models: Vec<InflowModel> = (-12_i32..0)
        .map(|sid| InflowModel {
            hydro_id: hydro,
            stage_id: sid,
            mean_m3s: 90.0,
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
    let par = PrecomputedPar::build(&all_models, &stages, &[hydro], None).expect("PAR build");
    let max_order = par.max_order();
    assert_eq!(max_order, 12);

    // Two window years with non-overlapping year ranges so the obs_table has
    // no collisions: 1990 (pre-window 1989) and 1992 (pre-window 1991).
    let window_years = [1990_i32, 1992];
    // Unique observations per window: window w uses values 80+7.5*i + w*50.
    // pre_year = w_year - 1; both use distinct calendar years.
    let history_for = |w_year: i32, offset: f64| {
        let pre_year = w_year - 1;
        let mut h: Vec<InflowHistoryRow> = Vec::with_capacity(24);
        for m in 0..12_u32 {
            h.push(row(hydro, pre_year, m, 80.0 + 7.5 * f64::from(m) + offset));
        }
        for m in 0..12_u32 {
            h.push(row(hydro, w_year, m, 160.0 + 3.0 * f64::from(m) + offset));
        }
        h
    };
    let mut full_history = history_for(1990, 0.0); // 1989 + 1990
    full_history.extend(history_for(1992, 50.0)); // 1991 + 1992 — no overlap with 1989/1990

    // Shared past_inflows: a constant 150.0 for all lags — different from both
    // window antecedent lags.
    let past_values_m3s = vec![150.0_f64; max_order];
    let past_inflows = vec![HydroPastInflows {
        hydro_id: hydro,
        values_m3s: past_values_m3s.clone(),
        season_ids: None,
    }];
    let transitions: Vec<StageLagTransition> =
        (0..12).map(|_| uniform_monthly_transition()).collect();

    let mut library = HistoricalScenarioLibrary::new(2, 12, 1, max_order, window_years.to_vec());
    standardize_historical_windows(
        &mut library,
        &full_history,
        &[hydro],
        &stages,
        &par,
        &window_years,
        None,
        &past_inflows,
        &transitions,
    );

    // Both windows must replay exactly when starting from past_inflows.
    let safe = max_order.max(1);
    let mut reconstructed = vec![0.0_f64];
    for (w, &w_year) in window_years.iter().enumerate() {
        let offset = if w == 0 { 0.0 } else { 50.0 };
        let mut lag_state = vec![0.0_f64; safe];
        for (lag, &v) in past_values_m3s.iter().enumerate().take(max_order) {
            lag_state[lag] = v;
        }

        for t in 0..12_usize {
            let eta = library.eta_slice(w, t);
            evaluate_par_batch(&par, t, &lag_state, eta, &mut reconstructed);

            #[allow(clippy::cast_precision_loss)] // t in 0..12, exact in f64
            let target = 160.0 + 3.0 * (t as f64) + offset;
            let got = reconstructed[0];
            assert!(
                (got - target).abs() < 1e-9,
                "T5 window={w} (year={w_year}) stage {t}: reconstructed {got:.12} != \
                 target {target:.12} (diff {:.3e})",
                (got - target).abs(),
            );

            advance_lag_state_uniform(&mut lag_state, got, max_order);
        }
    }
}

// ---------------------------------------------------------------------------
// T6: Non-trivial stage_lag_transitions guard fires in debug builds
// ---------------------------------------------------------------------------

/// Passing a non-trivial `StageLagTransition` (e.g., `finalize_period = false`)
/// must trigger the `debug_assert!` guard in
/// `standardize_historical_windows` in debug builds.
///
/// In release builds the guard is elided; this test is `#[cfg(debug_assertions)]`.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "historical-replay-non-monthly")]
fn t6_non_trivial_transitions_guard_fires() {
    let hydro = EntityId(1);
    let stages: Vec<Stage> = vec![monthly_stage(0, 0), monthly_stage(1, 1)];
    let models: Vec<InflowModel> = (0_i32..2)
        .map(|sid| InflowModel {
            hydro_id: hydro,
            stage_id: sid,
            mean_m3s: 100.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();
    let par = PrecomputedPar::build(&models, &stages, &[hydro], None).expect("PAR build");

    let history = vec![row(hydro, 2000, 0, 110.0), row(hydro, 2000, 1, 90.0)];

    // A non-trivial transition: finalize_period = false (accumulating, not finalising).
    let non_trivial = StageLagTransition {
        accumulate_weight: 0.5,
        spillover_weight: 0.5,
        finalize_period: false, // <-- non-uniform: triggers guard
        accumulate_downstream: false,
        downstream_accumulate_weight: 0.0,
        downstream_spillover_weight: 0.0,
        downstream_finalize: false,
        rebuild_from_downstream: false,
    };
    let transitions = vec![non_trivial, non_trivial];

    let mut lib = HistoricalScenarioLibrary::new(1, 2, 1, 0, vec![2000]);
    // Should panic with "historical-replay-non-monthly" in the debug_assert message.
    standardize_historical_windows(
        &mut lib,
        &history,
        &[hydro],
        &stages,
        &par,
        &[2000],
        None,
        &[],
        &transitions,
    );
}
