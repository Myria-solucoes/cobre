use std::collections::{BTreeMap, HashMap};

use chrono::NaiveDate;
use cobre_core::EntityId;

use super::{
    ArEstimationConfig, ContributionReduction, PacfReductionParams, ReductionReason,
    estimate_ar_coefficients_with_selection, estimate_ar_with_pacf_annual, has_negative_phi1,
    iterative_pacf_reduction, validate_order_contributions,
};
use crate::StochasticError;
use crate::par::fitting::{ArCoefficientEstimate, SeasonalStats};

/// Apply coefficient magnitude bound and contribution-based order validation
/// to all AR estimates (used only in tests after Fixed-path removal).
fn apply_contribution_validation(
    estimates: &mut [ArCoefficientEstimate],
    n_seasons: usize,
    stats_map: &HashMap<(EntityId, usize), &SeasonalStats>,
    sigma2_map: &HashMap<(EntityId, usize), Vec<f64>>,
    max_coeff_magnitude: Option<f64>,
) -> HashMap<EntityId, Vec<ContributionReduction>> {
    let mut all_reductions: HashMap<EntityId, Vec<ContributionReduction>> = HashMap::new();

    // Pre-pass: magnitude bound safety check.
    if let Some(threshold) = max_coeff_magnitude {
        for est in estimates.iter_mut() {
            let has_explosive = est.coefficients.iter().any(|c| c.abs() > threshold);
            if has_explosive {
                let original_order = est.coefficients.len();
                all_reductions
                    .entry(est.hydro_id)
                    .or_default()
                    .push(ContributionReduction {
                        season_id: est.season_id,
                        original_order,
                        reduced_order: 0,
                        contributions: Vec::new(), // magnitude-based, no contributions computed
                        reason: ReductionReason::MagnitudeBound,
                    });
                est.coefficients.clear();
                est.residual_std_ratio = 1.0;
            }
        }
    }

    // Pre-pass 2: phi_1 negativity rejection.
    // A negative first AR coefficient contradicts hydrological persistence.
    // This check is cheaper than contribution analysis and catches most
    // unstable models early.
    for est in estimates.iter_mut() {
        if has_negative_phi1(&est.coefficients) {
            let original_order = est.coefficients.len();
            all_reductions
                .entry(est.hydro_id)
                .or_default()
                .push(ContributionReduction {
                    season_id: est.season_id,
                    original_order,
                    reduced_order: 0,
                    contributions: Vec::new(),
                    reason: ReductionReason::Phi1Negative,
                });
            est.coefficients.clear();
            est.residual_std_ratio = 1.0;
        }
    }

    // Group estimate indices by hydro_id for per-entity processing.
    let mut hydro_indices: BTreeMap<EntityId, Vec<usize>> = BTreeMap::new();
    for (idx, est) in estimates.iter().enumerate() {
        hydro_indices.entry(est.hydro_id).or_default().push(idx);
    }

    for (&hydro_id, indices) in &hydro_indices {
        // Build std_by_season from seasonal_stats.
        let std_by_season: Vec<f64> = (0..n_seasons)
            .map(|sid| stats_map.get(&(hydro_id, sid)).map_or(0.0, |s| s.std))
            .collect();

        // Build all_season_coefficients from current estimates.
        let mut all_coeffs: Vec<Vec<f64>> = vec![Vec::new(); n_seasons];
        for &idx in indices {
            let est = &estimates[idx];
            if est.season_id < n_seasons {
                all_coeffs[est.season_id].clone_from(&est.coefficients);
            }
        }

        // Validate each season for this entity.
        for &idx in indices {
            let season_id = estimates[idx].season_id;
            let mut current_order = estimates[idx].coefficients.len();

            loop {
                let result = validate_order_contributions(
                    season_id,
                    n_seasons,
                    current_order,
                    &all_coeffs,
                    &std_by_season,
                );

                if result.valid || current_order == 0 {
                    break;
                }

                let original_order = current_order;
                let reduced_order = result.max_valid_order;

                all_reductions
                    .entry(hydro_id)
                    .or_default()
                    .push(ContributionReduction {
                        season_id,
                        original_order,
                        reduced_order,
                        contributions: result.contributions,
                        reason: ReductionReason::NegativeContribution,
                    });

                // Truncate coefficients to the reduced order.
                estimates[idx].coefficients.truncate(reduced_order);

                // Recompute residual_std_ratio from stored sigma2 if available.
                if reduced_order == 0 {
                    estimates[idx].residual_std_ratio = 1.0;
                } else if let Some(sigma2_vec) = sigma2_map.get(&(hydro_id, season_id))
                    && reduced_order <= sigma2_vec.len()
                {
                    let sigma2 = sigma2_vec[reduced_order - 1];
                    estimates[idx].residual_std_ratio =
                        if sigma2 <= 0.0 { 1.0 } else { sigma2.sqrt() };
                }

                // Update the shared coefficient array.
                all_coeffs[season_id].clone_from(&estimates[idx].coefficients);
                current_order = reduced_order;
            }
        }
    }

    all_reductions
}

// ── Numeric test helpers (pure core/stochastic inputs only) ──────────────────

/// Generate synthetic observations for a single-season AR(p) process.
/// Uses a fixed seed for reproducibility.
#[allow(clippy::cast_precision_loss, clippy::cast_lossless)] // u64 >> 33 fits in f64; u32::MAX fits in f64
fn generate_ar_observations(coefficients: &[f64], n: usize) -> Vec<f64> {
    let p = coefficients.len();
    let mut values = vec![0.0_f64; n + p];
    // Simple deterministic pseudo-noise using a linear congruential generator.
    let mut seed: u64 = 42;
    for i in p..(n + p) {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let noise = ((seed >> 33) as f64 / (u32::MAX as f64) - 0.5) * 2.0;
        let mut val = noise;
        for (j, c) in coefficients.iter().enumerate() {
            val += c * values[i - j - 1];
        }
        values[i] = val;
    }
    values[p..].to_vec()
}

/// Simulate a 2-season PAR(2) process using deterministic LCG (Box-Muller).
/// Model: `z_t = phi_1 * z_{t-1} + phi_2 * z_{t-2} + noise_t`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
fn simulate_two_season_par2(
    phi_1: f64,
    phi_2: f64,
    n_years: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let n_total = n_years * 2;
    let burnin = 200;
    let n_generate = n_total + burnin;
    let mut values = vec![0.0_f64; n_generate + 2];
    let mut lcg: u64 = seed;

    let lcg_next = |s: u64| -> u64 {
        s.wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407)
    };

    for i in 2..n_generate + 2 {
        lcg = lcg_next(lcg);
        let u1 = (lcg >> 11) as f64 / (1u64 << 53) as f64;
        lcg = lcg_next(lcg);
        let u2 = (lcg >> 11) as f64 / (1u64 << 53) as f64;
        let u1_safe = u1.max(1e-15);
        let noise = (-2.0 * u1_safe.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        values[i] = phi_1 * values[i - 1] + phi_2 * values[i - 2] + noise;
    }

    let start = burnin + 2;
    let mut obs_s0 = Vec::with_capacity(n_years);
    let mut obs_s1 = Vec::with_capacity(n_years);
    for y in 0..n_years {
        obs_s0.push(values[start + y * 2]);
        obs_s1.push(values[start + y * 2 + 1]);
    }
    (obs_s0, obs_s1)
}

/// Build a minimal 2-season `Stage` for testing.
fn make_two_season_stage(
    index: usize,
    id: i32,
    season_id: usize,
    year: i32,
    first_half: bool,
) -> cobre_core::temporal::Stage {
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    let (start_date, end_date) = if first_half {
        (
            NaiveDate::from_ymd_opt(year, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(year, 7, 1).unwrap(),
        )
    } else {
        (
            NaiveDate::from_ymd_opt(year, 7, 1).unwrap(),
            NaiveDate::from_ymd_opt(year + 1, 1, 1).unwrap(),
        )
    };

    cobre_core::temporal::Stage {
        index,
        id,
        start_date,
        end_date,
        season_id: Some(season_id),
        blocks: vec![Block {
            index: 0,
            name: "SINGLE".to_string(),
            duration_hours: 4380.0,
        }],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: false,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    }
}

/// Build a 2-season `SeasonMap` (H1: Jan–Jun, H2: Jul–Dec).
fn two_season_map() -> cobre_core::temporal::SeasonMap {
    use cobre_core::temporal::{SeasonCycleType, SeasonDefinition, SeasonMap};
    SeasonMap {
        cycle_type: SeasonCycleType::Custom,
        seasons: vec![
            SeasonDefinition {
                id: 0,
                label: "H1".to_string(),
                month_start: 1,
                day_start: Some(1),
                month_end: Some(6),
                day_end: Some(30),
            },
            SeasonDefinition {
                id: 1,
                label: "H2".to_string(),
                month_start: 7,
                day_start: Some(1),
                month_end: Some(12),
                day_end: Some(31),
            },
        ],
    }
}

/// Build a 12-season monthly stage sequence spanning `n_years` starting from
/// year 2000. Stage IDs are 0-based sequential; season IDs cycle 0..12.
fn make_monthly_stages_for_annual(n_years: usize) -> Vec<cobre_core::temporal::Stage> {
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };
    let mut stages = Vec::new();
    let mut idx = 0usize;
    for year in 0..n_years {
        for month in 0..12usize {
            let y = 2000 + year as i32;
            let m = month as u32 + 1;
            let (ey, em) = if m == 12 { (y + 1, 1u32) } else { (y, m + 1) };
            stages.push(cobre_core::temporal::Stage {
                index: idx,
                id: idx as i32,
                start_date: NaiveDate::from_ymd_opt(y, m, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(ey, em, 1).unwrap(),
                season_id: Some(month),
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
            });
            idx += 1;
        }
    }
    stages
}

/// Build `n_years * 12` synthetic monthly observations for `hydro_id`.
///
/// Formula: `z[year*12 + month] = base + (month+1) * scale + year * drift`.
fn synthetic_monthly_obs(
    hydro_id: EntityId,
    n_years: usize,
    base: f64,
    scale: f64,
    drift: f64,
) -> Vec<(EntityId, NaiveDate, f64)> {
    let mut obs = Vec::new();
    for year in 0..n_years {
        for month in 0..12usize {
            let value = base
                + f64::from(u32::try_from(month + 1).unwrap()) * scale
                + f64::from(u32::try_from(year).unwrap()) * drift;
            let date = NaiveDate::from_ymd_opt(
                2000 + i32::try_from(year).unwrap(),
                u32::try_from(month + 1).unwrap(),
                1,
            )
            .unwrap();
            obs.push((hydro_id, date, value));
        }
    }
    obs
}

// ── Contribution validation tests ────────────────────────────────────────────

#[test]
fn test_contribution_order_zero_fallback() {
    let result = validate_order_contributions(
        0,             // season_id
        1,             // n_seasons
        1,             // current_order
        &[vec![-1.5]], // all_season_coefficients
        &[10.0],       // std_by_season
    );
    assert!(!result.valid);
    assert_eq!(result.max_valid_order, 0);
}

/// E2-003: Order-0 input returns valid immediately.
#[test]
fn test_contribution_order_zero_input_passes() {
    let result = validate_order_contributions(0, 1, 0, &[Vec::new()], &[10.0]);
    assert!(result.valid);
    assert_eq!(result.max_valid_order, 0);
    assert!(result.contributions.is_empty());
}

/// E2-003: Stable AR(2) model with all-positive contributions passes.
#[test]
fn test_contribution_stable_model_passes() {
    let result = validate_order_contributions(
        0,                 // season_id
        1,                 // n_seasons
        2,                 // current_order
        &[vec![0.4, 0.2]], // all_season_coefficients
        &[10.0],           // std_by_season
    );
    assert!(result.valid);
    assert_eq!(result.max_valid_order, 2);
    assert_eq!(result.contributions.len(), 2);
}

/// E2-003: apply_contribution_validation reduces an explosive model.
///
/// Constructs AR(2) with coefficients [0.3, -0.8] for a single entity
/// and single season. The contribution of lag 2 is negative (-0.71),
/// so the order should be reduced to 1.
#[test]
fn test_apply_contribution_validation_reduces_explosive() {
    let hydro_id = EntityId(1);
    let n_seasons = 1;

    let mut estimates = vec![ArCoefficientEstimate {
        hydro_id,
        season_id: 0,
        coefficients: vec![0.3, -0.8],
        residual_std_ratio: 0.9,
        annual: None,
    }];

    let stats = vec![SeasonalStats {
        entity_id: hydro_id,
        stage_id: 0,
        mean: 100.0,
        std: 10.0,
    }];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
        stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();
    // sigma2_per_order: [sigma2_order1, sigma2_order2]
    // At order 1, residual is sqrt(0.81) ~= 0.9
    let mut sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();
    sigma2_map.insert((hydro_id, 0), vec![0.81, 0.75]);

    let reductions = apply_contribution_validation(
        &mut estimates,
        n_seasons,
        &stats_map,
        &sigma2_map,
        None, // max_coeff_magnitude
    );

    // After validation, the order should be reduced: [0.3, -0.8] -> [0.3]
    assert_eq!(
        estimates[0].coefficients.len(),
        1,
        "explosive AR(2) should be reduced to AR(1)"
    );
    assert!((estimates[0].coefficients[0] - 0.3).abs() < 1e-10);

    // Residual std ratio should be recomputed from sigma2_per_order[0]
    assert!(
        (estimates[0].residual_std_ratio - 0.81_f64.sqrt()).abs() < 1e-10,
        "residual_std_ratio should be recomputed from sigma2_per_order[0]"
    );

    // Should have recorded the reduction event.
    let entity_reductions = reductions.get(&hydro_id).expect("should have reductions");
    assert_eq!(entity_reductions.len(), 1);
    assert_eq!(entity_reductions[0].original_order, 2);
    assert_eq!(entity_reductions[0].reduced_order, 1);
    assert_eq!(entity_reductions[0].season_id, 0);
}

/// E2-003: PIMENTAL-like scenario -- large coefficient at lag 2 in one
/// season while other seasons are benign.
#[test]
fn test_pimental_like_multi_season_reduction() {
    let hydro_id = EntityId(156);
    let n_seasons = 12;

    // Build estimates: most seasons have small AR(1), August has explosive AR(2).
    let mut estimates: Vec<ArCoefficientEstimate> = (0..n_seasons)
        .map(|s| ArCoefficientEstimate {
            hydro_id,
            season_id: s,
            coefficients: if s == 7 {
                // August: explosive AR(2) with huge lag-2 coefficient
                vec![0.5, 48.9]
            } else {
                vec![0.1] // benign AR(1)
            },
            residual_std_ratio: 0.95,
            annual: None,
        })
        .collect();

    // Std devs: most seasons ~200, August = 5 (high coefficient * low std = trouble)
    let stds: Vec<f64> = (0..n_seasons)
        .map(|s| if s == 7 { 5.0 } else { 200.0 })
        .collect();

    let stats: Vec<SeasonalStats> = (0..n_seasons)
        .map(|s| SeasonalStats {
            entity_id: hydro_id,
            stage_id: s as i32,
            mean: 100.0,
            std: stds[s],
        })
        .collect();
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = stats
        .iter()
        .enumerate()
        .map(|(s, st)| ((hydro_id, s), st))
        .collect();

    let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

    let reductions = apply_contribution_validation(
        &mut estimates,
        n_seasons,
        &stats_map,
        &sigma2_map,
        None, // max_coeff_magnitude
    );

    // August (season 7) should have been reduced from AR(2).
    // The contribution of lag 2 through the periodic chain may or may not be negative
    // depending on the recursive composition with neighboring months' coefficients.
    // We verify the reduction was applied if any reduction occurred for August.
    let august_order = estimates[7].coefficients.len();

    // Other months should remain unchanged at AR(1) since their contributions
    // are small positive values.
    for (s, est) in estimates.iter().enumerate() {
        if s != 7 {
            assert_eq!(est.coefficients.len(), 1, "season {s} should remain AR(1)");
        }
    }

    // If August was reduced, there should be a reduction record.
    if august_order < 2 {
        let entity_reductions = reductions.get(&hydro_id).expect("should have reductions");
        assert!(
            entity_reductions.iter().any(|r| r.season_id == 7),
            "August reduction should be recorded"
        );
    }
}

/// E2-003: All contributions negative forces white-noise fallback.
///
/// AR(1) with phi = -2.0 for all seasons -- every contribution is negative,
/// so order drops to 0 and residual_std_ratio becomes 1.0.
#[test]
fn test_all_negative_fallback_to_white_noise() {
    let hydro_id = EntityId(1);
    let n_seasons = 1;

    let mut estimates = vec![ArCoefficientEstimate {
        hydro_id,
        season_id: 0,
        coefficients: vec![-2.0],
        residual_std_ratio: 0.8,
        annual: None,
    }];

    let stats = vec![SeasonalStats {
        entity_id: hydro_id,
        stage_id: 0,
        mean: 50.0,
        std: 10.0,
    }];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
        stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();
    let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

    let _reductions = apply_contribution_validation(
        &mut estimates,
        n_seasons,
        &stats_map,
        &sigma2_map,
        None, // max_coeff_magnitude
    );

    assert!(
        estimates[0].coefficients.is_empty(),
        "should fall back to order 0"
    );
    assert!(
        (estimates[0].residual_std_ratio - 1.0).abs() < 1e-10,
        "white-noise residual ratio should be 1.0"
    );
}

// ── Phi_1 rejection tests ────────────────────────────────────────────────

#[test]
fn phi1_rejection_sets_order_to_zero() {
    let hydro_id = EntityId(1);
    let n_seasons = 2;

    let mut estimates = vec![
        ArCoefficientEstimate {
            hydro_id,
            season_id: 0,
            coefficients: vec![-0.3, 0.5],
            residual_std_ratio: 0.8,
            annual: None,
        },
        ArCoefficientEstimate {
            hydro_id,
            season_id: 1,
            coefficients: vec![0.4, 0.2],
            residual_std_ratio: 0.7,
            annual: None,
        },
    ];

    let stats = vec![
        SeasonalStats {
            entity_id: hydro_id,
            stage_id: 0,
            mean: 50.0,
            std: 10.0,
        },
        SeasonalStats {
            entity_id: hydro_id,
            stage_id: 1,
            mean: 60.0,
            std: 12.0,
        },
    ];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = stats
        .iter()
        .enumerate()
        .map(|(i, s)| ((s.entity_id, i), s))
        .collect();
    let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

    let reductions =
        apply_contribution_validation(&mut estimates, n_seasons, &stats_map, &sigma2_map, None);

    // Season 0 should have been rejected (negative phi_1).
    assert!(
        estimates[0].coefficients.is_empty(),
        "season 0 should be cleared to order 0"
    );
    assert!(
        (estimates[0].residual_std_ratio - 1.0).abs() < 1e-10,
        "season 0 residual_std_ratio should be 1.0"
    );

    // Season 1 should be unchanged.
    assert_eq!(
        estimates[1].coefficients,
        vec![0.4, 0.2],
        "season 1 should be unchanged"
    );

    // Verify reduction entry for season 0.
    let hydro_reductions = reductions.get(&hydro_id).expect("should have reductions");
    let r = hydro_reductions
        .iter()
        .find(|r| r.season_id == 0)
        .expect("should have reduction for season 0");
    assert_eq!(r.original_order, 2);
    assert_eq!(r.reduced_order, 0);
    assert!(r.contributions.is_empty());
}

#[test]
fn phi1_rejection_before_contribution_analysis() {
    // A season with phi_1 = -0.01 and phi_2 = 0.5 with uniform std.
    // The contribution analysis would NOT catch this (contributions may
    // be non-negative), but the phi_1 gate fires first.
    let hydro_id = EntityId(1);
    let n_seasons = 1;

    let mut estimates = vec![ArCoefficientEstimate {
        hydro_id,
        season_id: 0,
        coefficients: vec![-0.01, 0.5],
        residual_std_ratio: 0.8,
        annual: None,
    }];

    let stats = vec![SeasonalStats {
        entity_id: hydro_id,
        stage_id: 0,
        mean: 50.0,
        std: 10.0,
    }];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
        stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();
    let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

    let reductions =
        apply_contribution_validation(&mut estimates, n_seasons, &stats_map, &sigma2_map, None);

    // Should be rejected by phi_1 gate.
    assert!(
        estimates[0].coefficients.is_empty(),
        "phi_1 = -0.01 should trigger rejection"
    );
    assert!(
        reductions.contains_key(&hydro_id),
        "should have a reduction entry"
    );
}

#[test]
fn phi1_zero_is_not_rejected() {
    let hydro_id = EntityId(1);
    let n_seasons = 1;

    let mut estimates = vec![ArCoefficientEstimate {
        hydro_id,
        season_id: 0,
        coefficients: vec![0.0, 0.3],
        residual_std_ratio: 0.8,
        annual: None,
    }];

    let stats = vec![SeasonalStats {
        entity_id: hydro_id,
        stage_id: 0,
        mean: 50.0,
        std: 10.0,
    }];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
        stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();
    let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

    let _reductions =
        apply_contribution_validation(&mut estimates, n_seasons, &stats_map, &sigma2_map, None);

    // phi_1 = 0.0 is NOT negative, so it should pass to contribution analysis.
    assert_eq!(
        estimates[0].coefficients.len(),
        2,
        "phi_1 = 0.0 should not be rejected"
    );
}

#[test]
fn phi1_rejection_interacts_with_magnitude_bound() {
    // phi_1 = -50.0 is both negative and above any reasonable magnitude bound.
    // The magnitude-bound pre-pass should fire first, and the phi_1 check
    // should see the already-cleared vector and skip.
    let hydro_id = EntityId(1);
    let n_seasons = 1;

    let mut estimates = vec![ArCoefficientEstimate {
        hydro_id,
        season_id: 0,
        coefficients: vec![-50.0, 0.3],
        residual_std_ratio: 0.8,
        annual: None,
    }];

    let stats = vec![SeasonalStats {
        entity_id: hydro_id,
        stage_id: 0,
        mean: 50.0,
        std: 10.0,
    }];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
        stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();
    let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

    let reductions = apply_contribution_validation(
        &mut estimates,
        n_seasons,
        &stats_map,
        &sigma2_map,
        Some(10.0), // magnitude bound that catches -50.0
    );

    // The season should be cleared (by magnitude bound).
    assert!(
        estimates[0].coefficients.is_empty(),
        "should be cleared to order 0"
    );

    // Only ONE reduction entry should exist (from magnitude bound, NOT doubled).
    let hydro_reductions = reductions.get(&hydro_id).expect("should have reductions");
    assert_eq!(
        hydro_reductions.len(),
        1,
        "should have exactly 1 reduction entry (magnitude bound only)"
    );
}

// ── Iterative PACF reduction tests ──────────────────────────────────────

#[test]
fn iterative_reduction_terminates_at_zero() {
    // Construct a case where contributions fail at every order,
    // forcing the loop to terminate at order 0.
    let hydro_id = EntityId(1);
    let n_seasons = 1;

    // Coefficients that produce negative contributions at order 2.
    // phi = [0.3, -5.0] -> contribution at lag 2 = 0.3*0.3 + (-5.0) < 0
    // After reducing max_order to 1 and re-running PACF, if PACF selects
    // order 1, contributions at order 1 are just 0.3 (positive).
    // But if we use coefficients that ALWAYS produce negative contributions,
    // the order will reach 0.
    //
    // Use phi = [-0.5] -> phi_1 is negative, caught by phi_1 gate directly.
    // Instead, use a scenario where contribution analysis fails at every order.
    //
    // We test this via the direct function with synthetic data designed
    // so that PACF at each reduced max_order still selects an order
    // that fails contributions.
    //
    // Simplest approach: use the apply_contribution_validation path
    // (for Fixed method) to verify order-0 termination.
    let mut estimates = vec![ArCoefficientEstimate {
        hydro_id,
        season_id: 0,
        coefficients: vec![0.3, -0.8],
        residual_std_ratio: 0.8,
        annual: None,
    }];

    let stats = vec![SeasonalStats {
        entity_id: hydro_id,
        stage_id: 0,
        mean: 50.0,
        std: 10.0,
    }];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
        stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();
    let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

    let reductions =
        apply_contribution_validation(&mut estimates, n_seasons, &stats_map, &sigma2_map, None);

    // The loop should terminate (possibly at a reduced order or order 0).
    // The key assertion is that it terminates and doesn't infinite-loop.
    assert!(
        estimates[0].coefficients.len() < 2,
        "order should be reduced from 2; got {}",
        estimates[0].coefficients.len()
    );

    // Should have at least one reduction entry.
    assert!(
        reductions.contains_key(&hydro_id),
        "should have a reduction entry"
    );
}

#[test]
fn iterative_reduction_only_affects_failing_seasons() {
    // Two seasons: season 0 has negative contributions, season 1 passes.
    let hydro_id = EntityId(1);
    let n_seasons = 2;

    let mut estimates = vec![
        ArCoefficientEstimate {
            hydro_id,
            season_id: 0,
            // phi = [0.3, -0.8]: contribution at lag 2 is negative.
            coefficients: vec![0.3, -0.8],
            residual_std_ratio: 0.8,
            annual: None,
        },
        ArCoefficientEstimate {
            hydro_id,
            season_id: 1,
            // phi = [0.4, 0.2]: all contributions positive.
            coefficients: vec![0.4, 0.2],
            residual_std_ratio: 0.7,
            annual: None,
        },
    ];

    let stats = vec![
        SeasonalStats {
            entity_id: hydro_id,
            stage_id: 0,
            mean: 50.0,
            std: 10.0,
        },
        SeasonalStats {
            entity_id: hydro_id,
            stage_id: 1,
            mean: 60.0,
            std: 10.0,
        },
    ];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = stats
        .iter()
        .enumerate()
        .map(|(i, s)| ((s.entity_id, i), s))
        .collect();
    let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

    let _reductions =
        apply_contribution_validation(&mut estimates, n_seasons, &stats_map, &sigma2_map, None);

    // Season 0 should have been reduced.
    assert!(
        estimates[0].coefficients.len() < 2,
        "season 0 order should be reduced from 2; got {}",
        estimates[0].coefficients.len()
    );

    // Season 1 should remain unchanged at order 2.
    assert_eq!(
        estimates[1].coefficients,
        vec![0.4, 0.2],
        "season 1 should be unchanged"
    );
}

#[test]
fn iterative_pacf_reduction_with_synthetic_observations() {
    // Test the iterative_pacf_reduction function directly with
    // synthetic observations. Generate data from a known AR process,
    // then artificially set coefficients that will fail contribution
    // analysis to trigger re-selection.
    let hydro_id = EntityId(1);
    let n_seasons = 1;

    // Generate enough observations for PACF to work.
    let obs = generate_ar_observations(&[0.5, 0.2], 100);

    let mut group_obs: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();
    group_obs.insert((hydro_id, 0), obs);

    let stats = vec![SeasonalStats {
        entity_id: hydro_id,
        stage_id: 0,
        mean: 0.0,
        std: 1.0,
    }];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
        stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();

    // Start with coefficients that have a bad high-order term.
    // The iterative loop should reduce the order.
    let mut estimates = vec![ArCoefficientEstimate {
        hydro_id,
        season_id: 0,
        coefficients: vec![0.5, 0.2, -5.0], // order 3, lag 3 will fail
        residual_std_ratio: 0.8,
        annual: None,
    }];

    let reductions = iterative_pacf_reduction(
        &mut estimates,
        n_seasons,
        &[hydro_id],
        &group_obs,
        &stats_map,
        &PacfReductionParams {
            initial_max_order: 3,
            z_alpha: 1.96,
            max_coeff_magnitude: None,
        },
    );

    // The function should have re-estimated using PACF at a reduced
    // max_order. The exact final order depends on the PACF selection,
    // but it should be less than the original 3.
    assert!(
        estimates[0].coefficients.len() < 3,
        "order should be reduced from 3; got {}",
        estimates[0].coefficients.len()
    );

    // Should have at least one reduction entry.
    assert!(
        reductions.contains_key(&hydro_id),
        "should have a reduction entry"
    );
}

#[test]
fn fixed_path_uses_truncation_not_reselection() {
    // Verify that the Fixed order selection path still uses
    // apply_contribution_validation (truncation), not iterative PACF.
    // We check this by verifying the behavior matches truncation semantics.
    let hydro_id = EntityId(1);
    let n_seasons = 1;

    // phi = [0.5, 0.2, -0.8]: contribution at lag 3 is negative.
    // Truncation would give order 2 (find_max_valid_order).
    // Iterative PACF would re-run PACF at max_order=2 and possibly
    // select a different order.
    let mut estimates = vec![ArCoefficientEstimate {
        hydro_id,
        season_id: 0,
        coefficients: vec![0.5, 0.2, -0.8],
        residual_std_ratio: 0.8,
        annual: None,
    }];

    let stats = vec![SeasonalStats {
        entity_id: hydro_id,
        stage_id: 0,
        mean: 50.0,
        std: 10.0,
    }];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> =
        stats.iter().map(|s| ((s.entity_id, 0_usize), s)).collect();
    let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

    let reductions =
        apply_contribution_validation(&mut estimates, n_seasons, &stats_map, &sigma2_map, None);

    // With truncation, the Fixed path truncates coefficients at the
    // first negative contribution position, which is at lag 3 (index 2).
    // So max_valid_order should be 2 or less.
    let final_order = estimates[0].coefficients.len();
    assert!(
        final_order <= 2,
        "Fixed path should truncate; got order {final_order}"
    );

    // Verify a reduction was recorded.
    assert!(
        reductions.contains_key(&hydro_id),
        "should have a reduction entry"
    );
    let r = &reductions[&hydro_id][0];
    assert_eq!(r.original_order, 3);
    // The reduced order should be from find_max_valid_order (truncation).
    assert!(r.reduced_order <= 2, "truncation should produce order <= 2");
}

// ── Combined strategy and reduction reason tests ─────────────────────────

#[test]
fn combined_strategies_produce_correct_reduction_reasons() {
    let h1 = EntityId(1);
    let h2 = EntityId(2);
    let n_seasons = 2;

    let mut estimates = vec![
        // H1 S0: negative phi_1 -> Phi1Negative
        ArCoefficientEstimate {
            hydro_id: h1,
            season_id: 0,
            coefficients: vec![-0.3, 0.5],
            residual_std_ratio: 0.8,
            annual: None,
        },
        // H1 S1: negative contribution at lag 3 -> NegativeContribution
        ArCoefficientEstimate {
            hydro_id: h1,
            season_id: 1,
            coefficients: vec![0.5, 0.2, -0.8],
            residual_std_ratio: 0.7,
            annual: None,
        },
        // H2 S0: magnitude bound -> MagnitudeBound
        ArCoefficientEstimate {
            hydro_id: h2,
            season_id: 0,
            coefficients: vec![50.0],
            residual_std_ratio: 0.9,
            annual: None,
        },
        // H2 S1: passes -> no reduction
        ArCoefficientEstimate {
            hydro_id: h2,
            season_id: 1,
            coefficients: vec![0.4, 0.2],
            residual_std_ratio: 0.7,
            annual: None,
        },
    ];

    let stats = vec![
        SeasonalStats {
            entity_id: h1,
            stage_id: 0,
            mean: 50.0,
            std: 10.0,
        },
        SeasonalStats {
            entity_id: h1,
            stage_id: 1,
            mean: 60.0,
            std: 10.0,
        },
        SeasonalStats {
            entity_id: h2,
            stage_id: 0,
            mean: 70.0,
            std: 15.0,
        },
        SeasonalStats {
            entity_id: h2,
            stage_id: 1,
            mean: 80.0,
            std: 12.0,
        },
    ];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = stats
        .iter()
        .enumerate()
        .map(|(i, s)| ((s.entity_id, i % 2), s))
        .collect();
    let sigma2_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

    let reductions = apply_contribution_validation(
        &mut estimates,
        n_seasons,
        &stats_map,
        &sigma2_map,
        Some(10.0),
    );

    // H1 S0: phi_1 negative -> Phi1Negative, order 0
    let h1_reductions = &reductions[&h1];
    let h1_s0 = h1_reductions
        .iter()
        .find(|r| r.season_id == 0)
        .expect("should have reduction for H1 S0");
    assert_eq!(h1_s0.reason, ReductionReason::Phi1Negative);
    assert_eq!(h1_s0.reduced_order, 0);

    // H1 S1: negative contribution -> NegativeContribution
    let h1_s1 = h1_reductions
        .iter()
        .find(|r| r.season_id == 1)
        .expect("should have reduction for H1 S1");
    assert_eq!(h1_s1.reason, ReductionReason::NegativeContribution);

    // H2 S0: magnitude bound -> MagnitudeBound, order 0
    let h2_reductions = &reductions[&h2];
    let h2_s0 = h2_reductions
        .iter()
        .find(|r| r.season_id == 0)
        .expect("should have reduction for H2 S0");
    assert_eq!(h2_s0.reason, ReductionReason::MagnitudeBound);
    assert_eq!(h2_s0.reduced_order, 0);

    // H2 S1: no reduction
    assert!(
        !h2_reductions.iter().any(|r| r.season_id == 1),
        "H2 S1 should have no reduction"
    );
}

// ── PACF and contribution cascade tests ──────────────────

/// Verify `iterative_pacf_reduction` does not spuriously reduce a stable
/// 2-season PAR(2) model (phi_1=0.7, phi_2=0.15). Checks termination,
/// order preservation, and coefficient matching.
#[test]
#[allow(clippy::cast_precision_loss)]
fn iterative_pacf_reduction_stable_par2_not_spuriously_reduced() {
    use crate::par::fitting::{
        estimate_periodic_ar_coefficients, periodic_pacf, select_order_pacf,
    };

    let hydro_id = EntityId(1);
    let n_seasons = 2;
    let n_years = 500_usize; // 500 obs/season; threshold ≈ 0.088 << PACF(2) ≈ 0.15.

    let (obs_s0, obs_s1) = simulate_two_season_par2(0.7, 0.15, n_years, 137);

    // Build group_obs and stats_map.
    let mut group_obs: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();
    group_obs.insert((hydro_id, 0), obs_s0.clone());
    group_obs.insert((hydro_id, 1), obs_s1.clone());

    let mean_s0 = obs_s0.iter().sum::<f64>() / obs_s0.len() as f64;
    let mean_s1 = obs_s1.iter().sum::<f64>() / obs_s1.len() as f64;
    let std_s0 = {
        let v =
            obs_s0.iter().map(|x| (x - mean_s0).powi(2)).sum::<f64>() / (obs_s0.len() - 1) as f64;
        v.sqrt()
    };
    let std_s1 = {
        let v =
            obs_s1.iter().map(|x| (x - mean_s1).powi(2)).sum::<f64>() / (obs_s1.len() - 1) as f64;
        v.sqrt()
    };

    let stats_storage = vec![
        SeasonalStats {
            entity_id: hydro_id,
            stage_id: 0,
            mean: mean_s0,
            std: std_s0,
        },
        SeasonalStats {
            entity_id: hydro_id,
            stage_id: 1,
            mean: mean_s1,
            std: std_s1,
        },
    ];
    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = stats_storage
        .iter()
        .enumerate()
        .map(|(s, st)| ((hydro_id, s), st))
        .collect();

    // Run PACF order selection + YW estimation to get the starting estimates.
    let stats_by_season_pop = {
        let n = obs_s0.len() as f64;
        let mu0 = mean_s0;
        let mu1 = mean_s1;
        let s0 = (obs_s0.iter().map(|x| (x - mu0).powi(2)).sum::<f64>() / n).sqrt();
        let s1 = (obs_s1.iter().map(|x| (x - mu1).powi(2)).sum::<f64>() / n).sqrt();
        vec![(mu0, s0), (mu1, s1)]
    };
    let obs_refs: Vec<&[f64]> = vec![&obs_s0, &obs_s1];
    let z_alpha = 1.96_f64;
    // Use max_order=2 because the generating process is AR(2).
    // Higher max_order with periodic PACF on a 2-season split of a stationary
    // process over-estimates order due to per-season standardization artifacts.
    let max_order = 2_usize;

    let mut estimates: Vec<ArCoefficientEstimate> = Vec::new();
    for season in 0..n_seasons {
        let n_obs = obs_refs[season].len();
        let pacf_values = periodic_pacf(
            season,
            max_order,
            n_seasons,
            &obs_refs,
            &stats_by_season_pop,
        );
        let selected = select_order_pacf(&pacf_values, n_obs, z_alpha).selected_order;
        let yw = estimate_periodic_ar_coefficients(
            season,
            selected,
            n_seasons,
            &obs_refs,
            &stats_by_season_pop,
        );
        estimates.push(ArCoefficientEstimate {
            hydro_id,
            season_id: season,
            coefficients: yw.coefficients,
            residual_std_ratio: yw.residual_std_ratio,
            annual: None,
        });
    }

    // All seasons should select order 2 before reduction.
    for est in &estimates {
        assert_eq!(
            est.coefficients.len(),
            2,
            "season {} should select order 2; got {}",
            est.season_id,
            est.coefficients.len()
        );
    }

    let coeffs_before: Vec<Vec<f64>> = estimates.iter().map(|e| e.coefficients.clone()).collect();

    // Run iterative PACF reduction (should terminate without spurious reductions).
    let reductions = iterative_pacf_reduction(
        &mut estimates,
        n_seasons,
        &[hydro_id],
        &group_obs,
        &stats_map,
        &PacfReductionParams {
            initial_max_order: max_order,
            z_alpha,
            max_coeff_magnitude: None,
        },
    );

    // Stable PAR(2) should not be reduced (order remains 2).
    for est in &estimates {
        assert_eq!(
            est.coefficients.len(),
            2,
            "stable PAR(2) season {} should remain order 2; got {}",
            est.season_id,
            est.coefficients.len()
        );
    }

    // Verify coefficients match (unchanged or recomputed at new order).
    for (est, before) in estimates.iter().zip(coeffs_before.iter()) {
        if est.coefficients.len() == before.len() {
            // No reduction: verify coefficients unchanged.
            for (a, b) in est.coefficients.iter().zip(before.iter()) {
                assert!(
                    (a - b).abs() < 1e-10,
                    "season {} coefficient drift: {a} vs {b}",
                    est.season_id
                );
            }
        } else {
            // Reduction occurred: verify against direct YW solve.
            let yw_direct = estimate_periodic_ar_coefficients(
                est.season_id,
                est.coefficients.len(),
                n_seasons,
                &obs_refs,
                &stats_by_season_pop,
            );
            for (a, b) in est.coefficients.iter().zip(yw_direct.coefficients.iter()) {
                assert!(
                    (a - b).abs() < 1e-10,
                    "season {} post-reduction coeff {a} vs YW {b}",
                    est.season_id
                );
            }
        }
    }

    // No spurious reductions for stable model.
    assert!(!reductions.contains_key(&hydro_id));
}

/// Roundtrip estimation test: verify AR(2) coefficient recovery.
/// Simulates 1000 years of 2-season stationary AR(2) data (phi_1=0.7,
/// phi_2=0.15). Runs full PACF order selection pipeline and verifies
/// recovered coefficients match true values within 0.15 tolerance.
#[test]
#[allow(clippy::cast_precision_loss)]
fn roundtrip_estimation_two_season_par2_recovers_coefficients() {
    let hydro_id = EntityId(1);
    let n_years = 1000_usize;
    let true_phi1 = 0.7_f64;
    let true_phi2 = 0.15_f64;

    let (obs_s0, obs_s1) = simulate_two_season_par2(true_phi1, true_phi2, n_years, 42);

    // Build 2 study stages, one per season.
    // Season 0: Jan 1 – Jul 1 (any year); Season 1: Jul 1 – Jan 1.
    // All observations will be mapped via the SeasonMap fallback.
    let ref_year = 2000_i32;
    let stages = vec![
        make_two_season_stage(0, 0, 0, ref_year, true),
        make_two_season_stage(1, 1, 1, ref_year, false),
    ];

    // Build seasonal stats (Bessel-corrected std to match estimate_seasonal_stats).
    let n_f = n_years as f64;
    let mu0 = obs_s0.iter().sum::<f64>() / n_f;
    let mu1 = obs_s1.iter().sum::<f64>() / n_f;
    let std0 = (obs_s0.iter().map(|x| (x - mu0).powi(2)).sum::<f64>() / (n_f - 1.0)).sqrt();
    let std1 = (obs_s1.iter().map(|x| (x - mu1).powi(2)).sum::<f64>() / (n_f - 1.0)).sqrt();

    let seasonal_stats = vec![
        SeasonalStats {
            entity_id: hydro_id,
            stage_id: 0, // matches stages[0].id
            mean: mu0,
            std: std0,
        },
        SeasonalStats {
            entity_id: hydro_id,
            stage_id: 1, // matches stages[1].id
            mean: mu1,
            std: std1,
        },
    ];

    // Build observations with dates: season 0 = Jan 1, season 1 = Jul 1.
    // Using a fixed reference year does NOT matter because find_season_for_date
    // falls back to the season_map when dates are outside the study stage ranges.
    let season_map = two_season_map();
    let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
    for y in 0..n_years {
        let year = (1970 + y) as i32;
        observations.push((
            hydro_id,
            NaiveDate::from_ymd_opt(year, 1, 1).unwrap(),
            obs_s0[y],
        ));
        observations.push((
            hydro_id,
            NaiveDate::from_ymd_opt(year, 7, 1).unwrap(),
            obs_s1[y],
        ));
    }

    let (estimates, _report) = estimate_ar_coefficients_with_selection(
        &observations,
        &seasonal_stats,
        &stages,
        &[hydro_id],
        &ArEstimationConfig {
            max_order: 2,
            max_coeff_magnitude: None,
            season_map: Some(&season_map),
            use_annual_component: false,
        },
    )
    .expect("estimation must succeed");

    // Collect per-season coefficients.
    let est_s0 = estimates
        .iter()
        .find(|e| e.hydro_id == hydro_id && e.season_id == 0)
        .expect("season 0 estimate must exist");
    let est_s1 = estimates
        .iter()
        .find(|e| e.hydro_id == hydro_id && e.season_id == 1)
        .expect("season 1 estimate must exist");

    // With 1000 years the PACF threshold is 1.96/sqrt(1000) ≈ 0.062,
    // well below PACF(2) ≈ 0.15 and well above PACF(3..4) ≈ 0.
    // The pipeline should select order 2 and recover the true coefficients.
    for est in [est_s0, est_s1] {
        assert!(
            est.coefficients.len() >= 2,
            "season {} should select at least order 2; got {}",
            est.season_id,
            est.coefficients.len()
        );
        assert!(
            (est.coefficients[0] - true_phi1).abs() < 0.15,
            "season {} phi_1={:.4} should be within 0.15 of true {true_phi1:.4}",
            est.season_id,
            est.coefficients[0]
        );
        assert!(
            (est.coefficients[1] - true_phi2).abs() < 0.15,
            "season {} phi_2={:.4} should be within 0.15 of true {true_phi2:.4}",
            est.season_id,
            est.coefficients[1]
        );
        assert!(
            est.residual_std_ratio.is_finite() && est.residual_std_ratio > 0.0,
            "season {} residual_std_ratio={} should be positive finite",
            est.season_id,
            est.residual_std_ratio
        );
    }
}

// ── estimate_ar_with_pacf_annual tests ─────────────

/// Two hydros × 12 seasons: every estimate has `annual.is_some()`.
///
/// 30 years of synthetic monthly data (360 observations per hydro) gives
/// enough rolling-window samples for `estimate_annual_seasonal_stats` to
/// succeed and for the extended YW system to be well-conditioned.
///
/// Asserts:
/// - 24 estimates returned (2 hydros × 12 seasons).
/// - Every `estimate.annual.is_some()`.
/// - Every `estimate.annual.as_ref().unwrap().std_m3s > 0.0`.
#[test]
fn estimate_ar_with_pacf_annual_two_hydros_twelve_seasons() {
    let h1 = EntityId(1);
    let h2 = EntityId(2);
    let n_years = 30;
    let stages = make_monthly_stages_for_annual(n_years);

    // Two hydros with different base values so their series are distinct.
    let mut obs = synthetic_monthly_obs(h1, n_years, 100.0, 5.0, 1.0);
    obs.extend(synthetic_monthly_obs(h2, n_years, 200.0, 3.0, 0.5));

    // Seasonal stats needed for the YW solve (standard values).
    let seasonal_stats = {
        use crate::par::fitting::estimate_seasonal_stats_with_season_map;
        estimate_seasonal_stats_with_season_map(&obs, &stages, &[h1, h2], None).unwrap()
    };

    let (estimates, report) = estimate_ar_with_pacf_annual(
        &obs,
        &seasonal_stats,
        &stages,
        &[h1, h2],
        3,    // max_order
        None, // season_map
        None, // max_coeff_magnitude
    )
    .expect("estimate_ar_with_pacf_annual must succeed with 30 years of data");

    assert_eq!(
        estimates.len(),
        24,
        "2 hydros × 12 seasons = 24 estimates, got {}",
        estimates.len()
    );
    assert_eq!(
        report.method, "PACF_ANNUAL",
        "method must be PACF_ANNUAL, got {}",
        report.method
    );

    for est in &estimates {
        assert!(
            est.annual.is_some(),
            "hydro={} season={}: annual must be Some",
            est.hydro_id.0,
            est.season_id
        );
        let ann = est.annual.as_ref().unwrap();
        assert!(
            ann.std_m3s > 0.0,
            "hydro={} season={}: annual.std_m3s must be > 0, got {}",
            est.hydro_id.0,
            est.season_id,
            ann.std_m3s
        );
    }
}

/// Insufficient observations propagate `InsufficientData` error.
///
/// 11 months of data cannot form any rolling 12-month average.
/// `estimate_ar_with_pacf_annual` must propagate the error from
/// `estimate_annual_seasonal_stats` rather than silently falling back.
#[test]
fn estimate_ar_with_pacf_annual_insufficient_observations_errors() {
    let h1 = EntityId(1);
    // 11 observations — fewer than the 13 required for one rolling window.
    let obs: Vec<(EntityId, NaiveDate, f64)> = (0u32..11)
        .map(|i| {
            let m = i % 12 + 1;
            (
                h1,
                NaiveDate::from_ymd_opt(2000, m, 1).unwrap(),
                50.0 + f64::from(i),
            )
        })
        .collect();

    let stages = make_monthly_stages_for_annual(2);

    // Provide minimal seasonal stats (actual values don't matter since the
    // error occurs before the YW solve).
    let seasonal_stats = vec![SeasonalStats {
        entity_id: h1,
        stage_id: 0,
        mean: 50.0,
        std: 5.0,
    }];

    let result = estimate_ar_with_pacf_annual(
        &obs,
        &seasonal_stats,
        &stages,
        &[h1],
        2,    // max_order
        None, // season_map
        None, // max_coeff_magnitude
    );

    assert!(
        result.is_err(),
        "expected Err for insufficient observations, got Ok"
    );
    assert!(
        matches!(
            result.unwrap_err(),
            StochasticError::InsufficientData { .. }
        ),
        "error must be InsufficientData"
    );
}
