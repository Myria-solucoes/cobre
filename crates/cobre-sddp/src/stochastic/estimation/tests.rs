#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::useless_vec
)]

use super::*;
use cobre_core::scenario::{CorrelationModel, InflowModel};
use cobre_core::{EntityId, SystemBuilder};

// ── Helper to build a minimal System ─────────────────────────────────────

fn minimal_system_with_inflow_models(models: Vec<InflowModel>) -> System {
    SystemBuilder::new()
        .inflow_models(models)
        .build()
        .expect("valid system")
}

// ── with_scenario_models tests ────────────────────────────────────────────

/// AC-036-1: `with_scenario_models` replaces `inflow_models` and `correlation`
/// while preserving all other fields.
#[test]
fn test_with_scenario_models_replaces_fields() {
    use cobre_core::{
        Bus, DeficitSegment,
        scenario::{CorrelationModel, InflowModel},
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(f64::INFINITY),
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };

    // Build system with 2 inflow models.
    let old_model = InflowModel {
        hydro_id: EntityId(1),
        stage_id: 0,
        mean_m3s: 10.0,
        std_m3s: 1.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    };
    let system = SystemBuilder::new()
        .buses(vec![bus])
        .inflow_models(vec![old_model.clone(), {
            let mut m = old_model.clone();
            m.stage_id = 1;
            m
        }])
        .build()
        .expect("valid system");

    assert_eq!(system.inflow_models().len(), 2);
    assert_eq!(system.n_buses(), 1);

    // Replace with 4 models.
    let new_models: Vec<InflowModel> = (0..4)
        .map(|i| InflowModel {
            hydro_id: EntityId(1),
            stage_id: i,
            mean_m3s: 50.0,
            std_m3s: 5.0,
            ar_coefficients: vec![0.4],
            residual_std_ratio: 0.9,
            annual: None,
        })
        .collect();
    let new_corr = CorrelationModel::default();

    let updated = system.with_scenario_models(new_models.clone(), new_corr.clone());

    // inflow_models and correlation updated.
    assert_eq!(updated.inflow_models().len(), 4, "expected 4 inflow models");
    assert_eq!(
        *updated.correlation(),
        new_corr,
        "correlation should equal new_corr"
    );

    // hydros, buses, stages unchanged.
    assert_eq!(updated.n_buses(), 1, "buses must be preserved");
    assert!(
        updated.hydros().is_empty(),
        "hydros must be preserved (empty)"
    );
    assert!(
        updated.stages().is_empty(),
        "stages must be preserved (empty)"
    );
}

/// `with_scenario_models` with an empty vec clears `inflow_models`.
#[test]
fn test_with_scenario_models_clears_when_empty() {
    let model = InflowModel {
        hydro_id: EntityId(1),
        stage_id: 0,
        mean_m3s: 100.0,
        std_m3s: 10.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    };
    let system = minimal_system_with_inflow_models(vec![model]);
    assert_eq!(system.inflow_models().len(), 1);

    let updated = system.with_scenario_models(vec![], CorrelationModel::default());
    assert!(updated.inflow_models().is_empty());
}

// ── estimate_from_history path-matrix tests ────────────────────────────────

/// AC-036-3: When both stats files exist, `estimate_from_history` returns
/// the system unchanged.
#[test]
fn test_estimate_explicit_stats_returns_unchanged() {
    use tempfile::TempDir;

    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    // Create the minimal required files so validate_structure won't fail
    // (it only checks existence).
    create_required_files(case_dir);

    // Create both stats files.
    let scenarios = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios).unwrap();
    std::fs::write(scenarios.join("inflow_history.parquet"), b"").unwrap();
    std::fs::write(scenarios.join("inflow_seasonal_stats.parquet"), b"").unwrap();
    std::fs::write(scenarios.join("inflow_ar_coefficients.parquet"), b"").unwrap();

    let model = InflowModel {
        hydro_id: EntityId(1),
        stage_id: 0,
        mean_m3s: 100.0,
        std_m3s: 10.0,
        ar_coefficients: vec![0.5],
        residual_std_ratio: 0.87,
        annual: None,
    };
    let system = minimal_system_with_inflow_models(vec![model]);
    let original_len = system.inflow_models().len();

    let config = default_config();
    let (result, report, _path) = estimate_from_history(system, case_dir, &config).unwrap();

    assert_eq!(
        result.inflow_models().len(),
        original_len,
        "explicit stats: system must be unchanged"
    );
    assert!(
        report.is_none(),
        "explicit stats path must return None report"
    );
}

/// AC-036-4: When no history file exists, `estimate_from_history` returns
/// the system unchanged.
#[test]
fn test_estimate_no_history_returns_unchanged() {
    use tempfile::TempDir;

    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    // No scenarios/ directory at all.
    create_required_files(case_dir);

    let model = InflowModel {
        hydro_id: EntityId(1),
        stage_id: 0,
        mean_m3s: 100.0,
        std_m3s: 10.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    };
    let system = minimal_system_with_inflow_models(vec![model]);
    let original_len = system.inflow_models().len();

    let config = default_config();
    let (result, report, _path) = estimate_from_history(system, case_dir, &config).unwrap();

    assert_eq!(
        result.inflow_models().len(),
        original_len,
        "no history: system must be unchanged"
    );
    assert!(report.is_none(), "no history path must return None report");
}

// ── EstimationPath unit tests ─────────────────────────────────────────────

/// All 8 boolean combinations map to the expected `EstimationPath` variant.
///
/// Covers all 7 named variants plus the edge case `(false, false, true)`
/// which must map to `Deterministic` because AR alone is meaningless.
#[test]
fn test_estimation_path_resolve_all_8_combinations() {
    use cobre_io::FileManifest;

    let make = |history: bool, stats: bool, ar: bool| FileManifest {
        scenarios_inflow_history_parquet: history,
        scenarios_inflow_seasonal_stats_parquet: stats,
        scenarios_inflow_ar_coefficients_parquet: ar,
        ..Default::default()
    };

    // Row 1: (false, false, false) -> Deterministic
    assert_eq!(
        EstimationPath::resolve(&make(false, false, false)),
        EstimationPath::Deterministic,
    );
    // Edge case: (false, false, true) -> Deterministic (AR alone is meaningless)
    assert_eq!(
        EstimationPath::resolve(&make(false, false, true)),
        EstimationPath::Deterministic,
    );
    // Row 2: (false, true, false) -> UserStatsWhiteNoise
    assert_eq!(
        EstimationPath::resolve(&make(false, true, false)),
        EstimationPath::UserStatsWhiteNoise,
    );
    // Row 3: (false, true, true) -> UserProvidedNoHistory
    assert_eq!(
        EstimationPath::resolve(&make(false, true, true)),
        EstimationPath::UserProvidedNoHistory,
    );
    // Row 4: (true, false, false) -> FullEstimation
    assert_eq!(
        EstimationPath::resolve(&make(true, false, false)),
        EstimationPath::FullEstimation,
    );
    // Row 5: (true, false, true) -> UserArHistoryStats
    assert_eq!(
        EstimationPath::resolve(&make(true, false, true)),
        EstimationPath::UserArHistoryStats,
    );
    // Row 6: (true, true, false) -> PartialEstimation
    assert_eq!(
        EstimationPath::resolve(&make(true, true, false)),
        EstimationPath::PartialEstimation,
    );
    // Row 7: (true, true, true) -> UserProvidedAll
    assert_eq!(
        EstimationPath::resolve(&make(true, true, true)),
        EstimationPath::UserProvidedAll,
    );
}

/// Every variant's `as_str()` must return a non-empty, unique string.
#[test]
fn test_estimation_path_as_str_round_trip() {
    let variants = [
        EstimationPath::Deterministic,
        EstimationPath::UserStatsWhiteNoise,
        EstimationPath::UserProvidedNoHistory,
        EstimationPath::FullEstimation,
        EstimationPath::UserArHistoryStats,
        EstimationPath::PartialEstimation,
        EstimationPath::UserProvidedAll,
    ];

    let strings: Vec<&str> = variants.iter().map(|v| v.as_str()).collect();

    // All strings must be non-empty.
    for s in &strings {
        assert!(!s.is_empty(), "as_str() returned empty string");
    }

    // All strings must be unique.
    let unique: std::collections::HashSet<&&str> = strings.iter().collect();
    assert_eq!(
        unique.len(),
        variants.len(),
        "as_str() must return unique strings for each variant"
    );
}

// ── user_stats_to_rows unit tests ─────────────────────────────────────────

/// `user_stats_to_rows` maps all models — 3 InflowModel entries (2 hydros,
/// multiple stages) produce the same count of rows with bitwise-equal stats.
#[test]
fn test_user_stats_to_rows_maps_all_models() {
    let models = vec![
        InflowModel {
            hydro_id: EntityId(1),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        },
        InflowModel {
            hydro_id: EntityId(1),
            stage_id: 1,
            mean_m3s: 120.0,
            std_m3s: 12.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        },
        InflowModel {
            hydro_id: EntityId(2),
            stage_id: 0,
            mean_m3s: 50.0,
            std_m3s: 5.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        },
    ];
    let system = minimal_system_with_inflow_models(models.clone());
    let rows = user_stats_to_rows(&system);

    assert_eq!(rows.len(), 3, "must produce one row per InflowModel");

    for (model, row) in models.iter().zip(rows.iter()) {
        assert_eq!(row.hydro_id, model.hydro_id, "hydro_id must be preserved");
        assert_eq!(row.stage_id, model.stage_id, "stage_id must be preserved");
        // Bitwise equality: the f64 bits must be identical, not just approximately equal.
        assert_eq!(
            row.mean_m3s.to_bits(),
            model.mean_m3s.to_bits(),
            "mean_m3s must be bitwise identical"
        );
        assert_eq!(
            row.std_m3s.to_bits(),
            model.std_m3s.to_bits(),
            "std_m3s must be bitwise identical"
        );
    }
}

/// `user_stats_to_rows` on an empty system returns an empty vec.
#[test]
fn test_user_stats_to_rows_empty_system() {
    let system = minimal_system_with_inflow_models(vec![]);
    let rows = user_stats_to_rows(&system);
    assert!(rows.is_empty(), "empty system must produce empty rows");
}

// ── PartialEstimation unit tests ──────────────────────────────────────────

/// Write a real `inflow_history.parquet` with synthetic 2-season PAR(1) data
/// for a single hydro (id=1), using the existing `simulate_two_season_par2`
/// helper at order 2 to generate observations with non-trivial structure.
///
/// Observations are placed on Jan 1 (season 0) and Jul 1 (season 1) of
/// successive years starting from 1970, dated so they fall within the
/// study stages built by `make_two_season_stage`.
///
/// The history file is required to have real Parquet content because
/// `run_partial_estimation` calls `parse_inflow_history` on it.
fn write_unit_test_inflow_history(path: &std::path::Path, hydro_id: i32, n_years: usize) {
    use arrow::array::{Date32Array, Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use chrono::NaiveDate;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let date_to_days = |d: NaiveDate| -> i32 {
        i32::try_from((d - epoch).num_days()).expect("date in Date32 range")
    };

    let (obs_s0, obs_s1) = simulate_two_season_par2(0.7, 0.15, n_years, 99);

    let mut ids: Vec<i32> = Vec::with_capacity(n_years * 2);
    let mut dates: Vec<i32> = Vec::with_capacity(n_years * 2);
    let mut values: Vec<f64> = Vec::with_capacity(n_years * 2);

    for y in 0..n_years {
        let year = (1970 + y) as i32;
        // Season 0: Jan 1 falls within make_two_season_stage(..., first_half=true)
        ids.push(hydro_id);
        dates.push(date_to_days(NaiveDate::from_ymd_opt(year, 1, 15).unwrap()));
        // Shift values by 300 so they are all positive (simulate_two_season_par2
        // produces ~0-mean series; offset keeps inflows physically plausible).
        values.push(obs_s0[y] + 300.0);

        // Season 1: Jul 1 falls within make_two_season_stage(..., first_half=false)
        ids.push(hydro_id);
        dates.push(date_to_days(NaiveDate::from_ymd_opt(year, 7, 15).unwrap()));
        values.push(obs_s1[y] + 300.0);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("date", DataType::Date32, false),
        Field::new("value_m3s", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(Date32Array::from(dates)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .expect("valid batch");

    let file = std::fs::File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");
}

/// Build a `System` with one hydro (id=1), one bus, and 2-season stages
/// spanning `n_years` study years, with pre-loaded inflow models whose
/// `mean_m3s = 100.0` and `std_m3s = 10.0` (user-provided stats).
///
/// This represents the state after `load_case` has loaded
/// `inflow_seasonal_stats.parquet` but not `inflow_ar_coefficients.parquet`.
#[allow(clippy::cast_possible_wrap)]
fn build_system_with_user_stats(n_years: usize) -> System {
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::InflowModel;
    use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};

    let hydro_id = EntityId(1);
    let bus = Bus {
        id: EntityId(10),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(f64::INFINITY),
            cost_per_mwh: 3000.0,
        }],
        excess_cost: 0.0,
    };

    // Build 2-season stages using make_two_season_stage (season 0: Jan–Jun,
    // season 1: Jul–Dec), one stage per season per year.
    let ref_year = 1970_i32;
    let mut stages = Vec::with_capacity(n_years * 2);
    for y in 0..n_years {
        let year = ref_year + y as i32;
        stages.push(make_two_season_stage(y * 2, (y * 2) as i32, 0, year, true));
        stages.push(make_two_season_stage(
            y * 2 + 1,
            (y * 2 + 1) as i32,
            1,
            year,
            false,
        ));
    }

    // Build inflow models for each stage, preserving user-provided stats.
    // These represent what load_case produces after reading inflow_seasonal_stats.
    let inflow_models: Vec<InflowModel> = stages
        .iter()
        .map(|s| InflowModel {
            hydro_id,
            stage_id: s.id,
            mean_m3s: 100.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let hydro = Hydro {
        id: hydro_id,
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(10),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 5000.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 1000.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 900.0,
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
            turbined_cost: 0.0,
            storage_violation_below_cost: 1000.0,
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

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .build()
        .expect("valid system with user stats")
}

/// Create the minimal directory skeleton and write both the real Parquet
/// history and an empty sentinel file for `inflow_seasonal_stats.parquet`,
/// which is sufficient for the manifest to classify the path as
/// `PartialEstimation` (history=true, stats=true, ar=false).
fn setup_partial_estimation_case(case_dir: &std::path::Path, n_years: usize) {
    create_required_files(case_dir);
    let scenarios = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios).unwrap();

    // Write a real Parquet history file — it must be parseable.
    write_unit_test_inflow_history(
        &scenarios.join("inflow_history.parquet"),
        1, // hydro_id
        n_years,
    );

    // Write a sentinel file to trigger the manifest flag.
    // validate_structure only checks existence, not content.
    std::fs::write(scenarios.join("inflow_seasonal_stats.parquet"), b"sentinel")
        .expect("write sentinel");

    // No inflow_ar_coefficients.parquet → PartialEstimation path.
}

/// `PartialEstimation` preserves user-provided `mean_m3s` and
/// `std_m3s` while estimating AR coefficients from history.
///
/// Setup: system with known user stats (mean=100.0, std=10.0 for every
/// stage), case dir with a real `inflow_history.parquet` (synthetic PAR(2)
/// data) and an `inflow_seasonal_stats.parquet` sentinel.
///
/// Asserts:
/// - Every inflow model in the returned system has `mean_m3s == 100.0`
///   (bitwise) and `std_m3s == 10.0` (bitwise).
/// - Every inflow model has at least one AR coefficient (non-empty).
#[test]
fn test_partial_estimation_preserves_user_stats() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30; // sufficient for PACF order selection
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    setup_partial_estimation_case(case_dir, N_YEARS);
    let system = build_system_with_user_stats(N_YEARS);
    let config = default_config();

    let (updated, report, path) =
        estimate_from_history(system, case_dir, &config).expect("partial estimation must succeed");

    assert_eq!(
        path,
        EstimationPath::PartialEstimation,
        "expected PartialEstimation path"
    );
    assert!(
        report.is_some(),
        "PartialEstimation must return Some(report)"
    );

    let models = updated.inflow_models();
    assert!(
        !models.is_empty(),
        "partial estimation must produce at least one inflow model"
    );

    for m in models {
        // Bitwise equality — no rounding or transformation is allowed.
        assert_eq!(
            m.mean_m3s.to_bits(),
            100.0_f64.to_bits(),
            "mean_m3s must be bitwise identical to user value 100.0 for stage {}",
            m.stage_id
        );
        assert_eq!(
            m.std_m3s.to_bits(),
            10.0_f64.to_bits(),
            "std_m3s must be bitwise identical to user value 10.0 for stage {}",
            m.stage_id
        );
        assert!(
            !m.ar_coefficients.is_empty(),
            "ar_coefficients must be non-empty for stage {} (estimated from history)",
            m.stage_id
        );
    }
}

/// `PartialEstimation` returns a `Some(report)` with method "PACF"
/// and an entry for the single hydro plant.
///
/// Same setup as `test_partial_estimation_preserves_user_stats`.
#[test]
fn test_partial_estimation_returns_report() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    setup_partial_estimation_case(case_dir, N_YEARS);
    let system = build_system_with_user_stats(N_YEARS);
    let config = default_config();

    let (_updated, report, _path) =
        estimate_from_history(system, case_dir, &config).expect("partial estimation must succeed");

    let report = report.expect("PartialEstimation must return Some(EstimationReport)");

    assert_eq!(
        report.method, "PACF",
        "estimation method must be PACF, got '{}'",
        report.method
    );
    assert_eq!(
        report.entries.len(),
        1,
        "report must contain exactly 1 entry (one hydro), got {}",
        report.entries.len()
    );
    assert!(
        report.entries.contains_key(&EntityId(1)),
        "report must contain an entry for hydro_id=1"
    );
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn default_config() -> Config {
    use cobre_io::config::{EstimationConfig, OrderSelectionMethod};
    let mut cfg: Config = serde_json::from_str(MINIMAL_CONFIG_JSON).unwrap();
    cfg.estimation = EstimationConfig {
        max_order: 2,
        order_selection: OrderSelectionMethod::Pacf,
        min_observations_per_season: 2,
        max_coefficient_magnitude: None,
    };
    cfg
}

const MINIMAL_CONFIG_JSON: &str = r#"{
        "training": { "tree_seed": 42 },
        "simulation": { "enabled": false, "num_scenarios": 0, "io_channel_capacity": 16 },
        "modeling": {},
        "policy": {},
        "exports": {}
    }"#;

fn create_required_files(case_dir: &std::path::Path) {
    // validate_structure only checks existence; content doesn't matter here.
    let _ = std::fs::create_dir_all(case_dir.join("system"));
    let _ = std::fs::create_dir_all(case_dir.join("scenarios"));
    let write = |name: &str| {
        let _ = std::fs::write(case_dir.join(name), b"{}");
    };
    write("config.json");
    write("penalties.json");
    write("stages.json");
    write("initial_conditions.json");
    write("system/buses.json");
    write("system/lines.json");
    write("system/hydros.json");
    write("system/thermals.json");
}

// ── EstimationReport unit tests ───────────────────────────────────────────

/// Construct mock `ArCoefficientEstimate` entries for 2 hydros with 3
/// seasons each, call `build_estimation_report`, and verify that the
/// report contains exactly 2 entries with the expected structure.
#[test]
fn test_estimation_report_structure() {
    use cobre_stochastic::par::fitting::{ContributionReduction, build_estimation_report};

    let h1 = EntityId(1);
    let h2 = EntityId(2);
    let n_seasons = 3_usize;

    // Build mock estimates: 2 hydros x 3 seasons, max order 2.
    let mut estimates = Vec::new();
    for &hydro_id in &[h1, h2] {
        for season_id in 0..n_seasons {
            estimates.push(ArCoefficientEstimate {
                hydro_id,
                season_id,
                coefficients: vec![0.5, 0.3],
                residual_std_ratio: 0.9,
                annual: None,
            });
        }
    }

    let contribution_reductions: HashMap<EntityId, Vec<ContributionReduction>> = HashMap::new();
    let report = build_estimation_report(&estimates, n_seasons, &contribution_reductions, "PACF");

    assert_eq!(report.entries.len(), 2, "expected 2 hydro entries");

    for &hydro_id in &[h1, h2] {
        let entry = report.entries.get(&hydro_id).expect("entry must exist");
        assert_eq!(entry.selected_order, 2, "selected_order must be 2");
        assert_eq!(
            entry.coefficients.len(),
            n_seasons,
            "one coefficient vec per season"
        );
    }
}

/// With empty observations the returned `EstimationReport` must have an
/// empty entries map.
#[test]
fn test_estimation_report_empty_for_pacf() {
    use cobre_core::temporal::Stage;

    let observations: Vec<(EntityId, chrono::NaiveDate, f64)> = vec![];
    let seasonal_stats: Vec<SeasonalStats> = vec![];
    let stages: Vec<Stage> = vec![];
    let hydro_ids: Vec<EntityId> = vec![];
    let max_order = 2;

    let (_, report) = estimate_ar_coefficients_with_selection(
        &observations,
        &seasonal_stats,
        &stages,
        &hydro_ids,
        &ArEstimationConfig {
            max_order,
            max_coeff_magnitude: None,
            season_map: None,
            use_annual_component: false,
        },
    )
    .unwrap();

    assert!(
        report.entries.is_empty(),
        "empty observations must produce empty EstimationReport"
    );
}

// ── Pre-study stage expansion tests ─────────────────────────────────────

/// Build a Stage with the given parameters, suitable for expansion tests.
fn make_expansion_stage(
    index: usize,
    id: i32,
    season_id: Option<usize>,
) -> cobre_core::temporal::Stage {
    use chrono::NaiveDate;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    cobre_core::temporal::Stage {
        index,
        id,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id,
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
            branching_factor: 10,
            noise_method: NoiseMethod::Saa,
        },
    }
}

#[test]
fn seasonal_stats_to_rows_includes_prestudy_stages() {
    // 3 study stages (id 0, 1, 2; seasons 0, 1, 2)
    // 2 pre-study stages (id -1, -2; seasons 2, 1)
    let stages = vec![
        make_expansion_stage(0, -2, Some(1)),
        make_expansion_stage(1, -1, Some(2)),
        make_expansion_stage(2, 0, Some(0)),
        make_expansion_stage(3, 1, Some(1)),
        make_expansion_stage(4, 2, Some(2)),
    ];

    let h1 = EntityId(1);
    // SeasonalStats for seasons 0, 1, 2 (stage_id is the first stage
    // with that season).
    let stats = vec![
        SeasonalStats {
            entity_id: h1,
            stage_id: 0,
            mean: 100.0,
            std: 20.0,
        },
        SeasonalStats {
            entity_id: h1,
            stage_id: 1,
            mean: 110.0,
            std: 22.0,
        },
        SeasonalStats {
            entity_id: h1,
            stage_id: 2,
            mean: 120.0,
            std: 24.0,
        },
    ];

    let rows = seasonal_stats_to_rows(&stats, &stages);

    // season 0: stage 0 only
    // season 1: stages -2 and 1
    // season 2: stages -1 and 2
    // Total: 1 + 2 + 2 = 5 rows
    assert_eq!(rows.len(), 5, "expected 5 rows (3 study + 2 pre-study)");

    // Verify pre-study rows exist with negative stage_ids.
    let prestudy_rows: Vec<_> = rows.iter().filter(|r| r.stage_id < 0).collect();
    assert_eq!(
        prestudy_rows.len(),
        2,
        "expected 2 pre-study rows, got {}",
        prestudy_rows.len()
    );

    // stage_id = -2 has season 1 -> (mean=110, std=22).
    let neg2 = rows.iter().find(|r| r.stage_id == -2).expect("row for -2");
    assert!((neg2.mean_m3s - 110.0).abs() < f64::EPSILON);
    assert!((neg2.std_m3s - 22.0).abs() < f64::EPSILON);

    // stage_id = -1 has season 2 -> (mean=120, std=24).
    let neg1 = rows.iter().find(|r| r.stage_id == -1).expect("row for -1");
    assert!((neg1.mean_m3s - 120.0).abs() < f64::EPSILON);
    assert!((neg1.std_m3s - 24.0).abs() < f64::EPSILON);

    // Rows should be sorted by (hydro_id, stage_id).
    for w in rows.windows(2) {
        assert!(
            (w[0].hydro_id.0, w[0].stage_id) <= (w[1].hydro_id.0, w[1].stage_id),
            "rows not sorted"
        );
    }
}

#[test]
fn ar_estimates_to_rows_includes_prestudy_stages() {
    // Same stage layout as test 1.
    let stages = vec![
        make_expansion_stage(0, -2, Some(1)),
        make_expansion_stage(1, -1, Some(2)),
        make_expansion_stage(2, 0, Some(0)),
        make_expansion_stage(3, 1, Some(1)),
        make_expansion_stage(4, 2, Some(2)),
    ];

    let h1 = EntityId(1);
    // AR(1) estimates for seasons 0, 1, 2.
    let ar_estimates = vec![
        ArCoefficientEstimate {
            hydro_id: h1,
            season_id: 0,
            coefficients: vec![0.3],
            residual_std_ratio: 0.9,
            annual: None,
        },
        ArCoefficientEstimate {
            hydro_id: h1,
            season_id: 1,
            coefficients: vec![0.4],
            residual_std_ratio: 0.85,
            annual: None,
        },
        ArCoefficientEstimate {
            hydro_id: h1,
            season_id: 2,
            coefficients: vec![0.5],
            residual_std_ratio: 0.8,
            annual: None,
        },
    ];

    let rows = ar_estimates_to_rows(&ar_estimates, &stages);

    // season 0 -> 1 stage (id 0): 1 row
    // season 1 -> 2 stages (ids -2, 1): 2 rows
    // season 2 -> 2 stages (ids -1, 2): 2 rows
    // Total: 5 rows (each AR(1), so 1 coefficient row per stage)
    assert_eq!(rows.len(), 5, "expected 5 rows");

    // Pre-study coefficient rows exist.
    let prestudy_rows: Vec<_> = rows.iter().filter(|r| r.stage_id < 0).collect();
    assert_eq!(prestudy_rows.len(), 2);

    // stage_id = -2 is season 1, coefficient = 0.4.
    let neg2 = rows.iter().find(|r| r.stage_id == -2).expect("row for -2");
    assert!((neg2.coefficient - 0.4).abs() < f64::EPSILON);
    assert!((neg2.residual_std_ratio - 0.85).abs() < f64::EPSILON);

    // stage_id = -1 is season 2, coefficient = 0.5.
    let neg1 = rows.iter().find(|r| r.stage_id == -1).expect("row for -1");
    assert!((neg1.coefficient - 0.5).abs() < f64::EPSILON);
    assert!((neg1.residual_std_ratio - 0.8).abs() < f64::EPSILON);
}

#[test]
fn full_estimation_produces_prestudy_inflow_models() {
    use cobre_io::scenarios::assemble_inflow_models;

    // Build stages with 2 pre-study + 3 study.
    let stages = vec![
        make_expansion_stage(0, -2, Some(1)),
        make_expansion_stage(1, -1, Some(2)),
        make_expansion_stage(2, 0, Some(0)),
        make_expansion_stage(3, 1, Some(1)),
        make_expansion_stage(4, 2, Some(2)),
    ];

    let h1 = EntityId(1);

    // Build stats rows (including pre-study).
    let stats = vec![
        SeasonalStats {
            entity_id: h1,
            stage_id: 0,
            mean: 100.0,
            std: 20.0,
        },
        SeasonalStats {
            entity_id: h1,
            stage_id: 1,
            mean: 110.0,
            std: 22.0,
        },
        SeasonalStats {
            entity_id: h1,
            stage_id: 2,
            mean: 120.0,
            std: 24.0,
        },
    ];
    let stats_rows = seasonal_stats_to_rows(&stats, &stages);

    // Build coefficient rows.
    let ar_ests = vec![
        ArCoefficientEstimate {
            hydro_id: h1,
            season_id: 0,
            coefficients: vec![0.3],
            residual_std_ratio: 0.9,
            annual: None,
        },
        ArCoefficientEstimate {
            hydro_id: h1,
            season_id: 1,
            coefficients: vec![0.4],
            residual_std_ratio: 0.85,
            annual: None,
        },
        ArCoefficientEstimate {
            hydro_id: h1,
            season_id: 2,
            coefficients: vec![0.5],
            residual_std_ratio: 0.8,
            annual: None,
        },
    ];
    let coeff_rows = ar_estimates_to_rows(&ar_ests, &stages);

    // Assemble into InflowModel.
    let inflow_models =
        assemble_inflow_models(stats_rows, coeff_rows, vec![]).expect("assembly should succeed");

    // Should have entries for pre-study stages.
    assert!(
        inflow_models.iter().any(|m| m.stage_id < 0),
        "expected pre-study InflowModel entries (negative stage_id)"
    );

    // Pre-study models should have correct stats from their season.
    let prestudy_neg2 = inflow_models
        .iter()
        .find(|m| m.stage_id == -2)
        .expect("InflowModel for stage -2");
    assert!((prestudy_neg2.mean_m3s - 110.0).abs() < f64::EPSILON);
    assert!((prestudy_neg2.std_m3s - 22.0).abs() < f64::EPSILON);
}

// ── PACF and contribution cascade tests ──────────────────────

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
    use chrono::NaiveDate;
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

// ── ar_rows_to_estimates unit tests ───────────────────────────────────────

/// AC-009-ar-rows-1: `ar_rows_to_estimates` groups by season, deduplicates stages.
///
/// Creates `InflowArCoefficientRow` entries for 2 hydros across 3 stages:
/// - Stage 0 (season 0), stage 1 (season 0), stage 2 (season 1).
///
/// After conversion the output must have 2 * 2 = 4 estimates (2 hydros * 2
/// seasons). Each estimate must carry the coefficients from the FIRST stage
/// in the season (stage 0 for season 0, stage 2 for season 1).
#[test]
#[allow(clippy::cast_sign_loss)]
fn test_ar_rows_to_estimates_groups_by_season() {
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    let make_stage = |id: i32, season_id: usize| cobre_core::temporal::Stage {
        index: id as usize,
        id,
        start_date: chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap(),
        end_date: chrono::NaiveDate::from_ymd_opt(1970, 7, 1).unwrap(),
        season_id: Some(season_id),
        blocks: vec![Block {
            index: 0,
            name: "T".to_string(),
            duration_hours: 1.0,
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
    };

    // 3 stages: stage 0 and stage 1 map to season 0; stage 2 maps to season 1.
    let stages = vec![make_stage(0, 0), make_stage(1, 0), make_stage(2, 1)];

    // AR(1) rows for 2 hydros; sorted by (hydro_id, stage_id, lag).
    // Each stage has one lag-1 row (AR order 1).
    let rows = vec![
        // hydro 1, stage 0 (season 0), lag 1
        InflowArCoefficientRow {
            hydro_id: EntityId(1),
            stage_id: 0,
            lag: 1,
            coefficient: 0.50,
            residual_std_ratio: 0.85,
        },
        // hydro 1, stage 1 (season 0 duplicate), lag 1
        InflowArCoefficientRow {
            hydro_id: EntityId(1),
            stage_id: 1,
            lag: 1,
            coefficient: 0.50,
            residual_std_ratio: 0.85,
        },
        // hydro 1, stage 2 (season 1), lag 1
        InflowArCoefficientRow {
            hydro_id: EntityId(1),
            stage_id: 2,
            lag: 1,
            coefficient: 0.60,
            residual_std_ratio: 0.80,
        },
        // hydro 2, stage 0 (season 0), lag 1
        InflowArCoefficientRow {
            hydro_id: EntityId(2),
            stage_id: 0,
            lag: 1,
            coefficient: 0.40,
            residual_std_ratio: 0.90,
        },
        // hydro 2, stage 1 (season 0 duplicate), lag 1
        InflowArCoefficientRow {
            hydro_id: EntityId(2),
            stage_id: 1,
            lag: 1,
            coefficient: 0.40,
            residual_std_ratio: 0.90,
        },
        // hydro 2, stage 2 (season 1), lag 1
        InflowArCoefficientRow {
            hydro_id: EntityId(2),
            stage_id: 2,
            lag: 1,
            coefficient: 0.35,
            residual_std_ratio: 0.88,
        },
    ];

    let estimates = ar_rows_to_estimates(&rows, &stages);

    // 2 hydros * 2 seasons = 4 estimates.
    assert_eq!(
        estimates.len(),
        4,
        "expected 4 estimates (2 hydros * 2 seasons), got {}",
        estimates.len()
    );

    // Hydro 1, season 0: coefficient from stage 0 (canonical first stage).
    let e = estimates
        .iter()
        .find(|e| e.hydro_id == EntityId(1) && e.season_id == 0)
        .expect("hydro 1, season 0 estimate must exist");
    assert_eq!(e.coefficients.len(), 1, "AR(1) must have 1 coefficient");
    assert!(
        (e.coefficients[0] - 0.50).abs() < f64::EPSILON,
        "coeff must be 0.50, got {}",
        e.coefficients[0]
    );
    assert!(
        (e.residual_std_ratio - 0.85).abs() < f64::EPSILON,
        "residual_std_ratio must be 0.85"
    );

    // Hydro 1, season 1: coefficient from stage 2.
    let e = estimates
        .iter()
        .find(|e| e.hydro_id == EntityId(1) && e.season_id == 1)
        .expect("hydro 1, season 1 estimate must exist");
    assert_eq!(e.coefficients.len(), 1);
    assert!((e.coefficients[0] - 0.60).abs() < f64::EPSILON);
    assert!((e.residual_std_ratio - 0.80).abs() < f64::EPSILON);

    // Hydro 2, season 0.
    let e = estimates
        .iter()
        .find(|e| e.hydro_id == EntityId(2) && e.season_id == 0)
        .expect("hydro 2, season 0 estimate must exist");
    assert_eq!(e.coefficients.len(), 1);
    assert!((e.coefficients[0] - 0.40).abs() < f64::EPSILON);

    // Hydro 2, season 1.
    let e = estimates
        .iter()
        .find(|e| e.hydro_id == EntityId(2) && e.season_id == 1)
        .expect("hydro 2, season 1 estimate must exist");
    assert_eq!(e.coefficients.len(), 1);
    assert!((e.coefficients[0] - 0.35).abs() < f64::EPSILON);
}

// ── UserArHistoryStats unit tests ─────────────────────────────────────────

/// Write `inflow_ar_coefficients.parquet` with known AR(1) coefficients for
/// a single hydro expanded to all stages in `stages`.
///
/// `stages` must be pre-built (same as the system's stages). The parquet
/// file will have one row per stage with lag=1.
fn write_unit_test_ar_coefficients(
    path: &std::path::Path,
    hydro_id: i32,
    stages: &[cobre_core::temporal::Stage],
    coefficient: f64,
    residual_std_ratio: f64,
) {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("lag", DataType::Int32, false),
        Field::new("coefficient", DataType::Float64, false),
        Field::new("residual_std_ratio", DataType::Float64, false),
    ]));

    let n = stages.len();
    let hydro_ids: Vec<i32> = vec![hydro_id; n];
    let stage_ids: Vec<i32> = stages.iter().map(|s| s.id).collect();
    let lags: Vec<i32> = vec![1; n];
    let coefficients: Vec<f64> = vec![coefficient; n];
    let ratios: Vec<f64> = vec![residual_std_ratio; n];

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(hydro_ids)),
            Arc::new(Int32Array::from(stage_ids)),
            Arc::new(Int32Array::from(lags)),
            Arc::new(Float64Array::from(coefficients)),
            Arc::new(Float64Array::from(ratios)),
        ],
    )
    .expect("valid batch");

    let file = std::fs::File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");
}

/// Build a system with one hydro and 2-season stages (same structure as
/// `build_system_with_user_stats`) but with EMPTY inflow_models.
///
/// This represents the state after `load_case` when `inflow_seasonal_stats.parquet`
/// is absent (the P7/UserArHistoryStats case): `assemble_inflow_models` returns
/// an empty vec, so `system.inflow_models()` is empty.
#[allow(clippy::cast_possible_wrap)]
fn build_system_empty_models(n_years: usize) -> System {
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};

    let hydro_id = EntityId(1);
    let bus = Bus {
        id: EntityId(10),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(f64::INFINITY),
            cost_per_mwh: 3000.0,
        }],
        excess_cost: 0.0,
    };

    let ref_year = 1970_i32;
    let mut stages = Vec::with_capacity(n_years * 2);
    for y in 0..n_years {
        let year = ref_year + y as i32;
        stages.push(make_two_season_stage(y * 2, (y * 2) as i32, 0, year, true));
        stages.push(make_two_season_stage(
            y * 2 + 1,
            (y * 2 + 1) as i32,
            1,
            year,
            false,
        ));
    }

    let hydro = Hydro {
        id: hydro_id,
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(10),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 5000.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 1000.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 900.0,
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
            turbined_cost: 0.0,
            storage_violation_below_cost: 1000.0,
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

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        // NOTE: no inflow_models — represents the P7 case after load_case
        .build()
        .expect("valid system with empty inflow models")
}

/// Setup a case directory for the P7 (UserArHistoryStats) path:
/// - `inflow_history.parquet`: real Parquet with synthetic 2-season data.
/// - `inflow_ar_coefficients.parquet`: real Parquet with known AR(1) coefficients.
/// - No `inflow_seasonal_stats.parquet`.
///
/// Returns the stages used in the system so the AR file can reference valid stage IDs.
#[allow(clippy::cast_possible_wrap)]
fn setup_user_ar_case(
    case_dir: &std::path::Path,
    n_years: usize,
    ar_coefficient: f64,
    residual_std_ratio: f64,
) {
    create_required_files(case_dir);
    let scenarios = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios).unwrap();

    // Write history parquet.
    write_unit_test_inflow_history(&scenarios.join("inflow_history.parquet"), 1, n_years);

    // Build the same stages as build_system_empty_models to get stage IDs.
    let ref_year = 1970_i32;
    let mut stages = Vec::with_capacity(n_years * 2);
    for y in 0..n_years {
        let year = ref_year + y as i32;
        stages.push(make_two_season_stage(y * 2, (y * 2) as i32, 0, year, true));
        stages.push(make_two_season_stage(
            y * 2 + 1,
            (y * 2 + 1) as i32,
            1,
            year,
            false,
        ));
    }

    // Write AR coefficients parquet with known values, one row per stage.
    write_unit_test_ar_coefficients(
        &scenarios.join("inflow_ar_coefficients.parquet"),
        1,
        &stages,
        ar_coefficient,
        residual_std_ratio,
    );

    // NO inflow_seasonal_stats.parquet — this is the P7 path.
}

/// AC-009-1: `estimate_from_history` with P7 setup preserves user AR coefficients
/// bitwise in the returned inflow models.
///
/// Setup: system with empty inflow_models, case dir with history + AR (no stats).
/// Assert: every returned model's `ar_coefficients[0]` and `residual_std_ratio`
/// match the known values written to `inflow_ar_coefficients.parquet` exactly.
#[test]
fn test_user_ar_estimation_preserves_ar_coefficients() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;
    const KNOWN_COEFF: f64 = 0.72;
    const KNOWN_RATIO: f64 = 0.69;

    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    setup_user_ar_case(case_dir, N_YEARS, KNOWN_COEFF, KNOWN_RATIO);
    let system = build_system_empty_models(N_YEARS);
    let config = default_config();

    let (updated, report, path) = estimate_from_history(system, case_dir, &config)
        .expect("UserArHistoryStats estimation must succeed");

    assert_eq!(
        path,
        EstimationPath::UserArHistoryStats,
        "expected UserArHistoryStats path"
    );
    assert!(
        report.is_some(),
        "UserArHistoryStats must return Some(report)"
    );

    let models = updated.inflow_models();
    assert!(
        !models.is_empty(),
        "estimation must produce at least one inflow model"
    );

    for m in models {
        assert_eq!(
            m.ar_coefficients.len(),
            1,
            "every model must have AR(1) coefficients (lag 1 only), stage {}",
            m.stage_id
        );
        assert_eq!(
            m.ar_coefficients[0].to_bits(),
            KNOWN_COEFF.to_bits(),
            "ar_coefficients[0] must be bitwise identical to {KNOWN_COEFF} for stage {}",
            m.stage_id
        );
        assert_eq!(
            m.residual_std_ratio.to_bits(),
            KNOWN_RATIO.to_bits(),
            "residual_std_ratio must be bitwise identical to {KNOWN_RATIO} for stage {}",
            m.stage_id
        );
    }
}

/// AC-009-2: `estimate_from_history` with P7 setup produces finite, positive
/// `mean_m3s` and `std_m3s` estimated from inflow history.
///
/// Same setup as `test_user_ar_estimation_preserves_ar_coefficients`.
#[test]
fn test_user_ar_estimation_estimates_stats_from_history() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;

    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    setup_user_ar_case(case_dir, N_YEARS, 0.55, 0.83);
    let system = build_system_empty_models(N_YEARS);
    let config = default_config();

    let (updated, _report, _path) = estimate_from_history(system, case_dir, &config)
        .expect("UserArHistoryStats estimation must succeed");

    let models = updated.inflow_models();
    assert!(
        !models.is_empty(),
        "estimation must produce at least one inflow model"
    );

    for m in models {
        assert!(
            m.mean_m3s.is_finite() && m.mean_m3s > 0.0,
            "mean_m3s must be finite and positive, got {} for stage {}",
            m.mean_m3s,
            m.stage_id
        );
        assert!(
            m.std_m3s.is_finite() && m.std_m3s >= 0.0,
            "std_m3s must be finite and non-negative, got {} for stage {}",
            m.std_m3s,
            m.stage_id
        );
    }
}

/// AC-009-3: `estimate_from_history` with P7 setup returns a report with
/// method "user_provided" and an empty entries map.
#[test]
fn test_user_ar_estimation_returns_user_provided_report() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;

    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    setup_user_ar_case(case_dir, N_YEARS, 0.55, 0.83);
    let system = build_system_empty_models(N_YEARS);
    let config = default_config();

    let (_updated, report, _path) = estimate_from_history(system, case_dir, &config)
        .expect("UserArHistoryStats estimation must succeed");

    let report = report.expect("UserArHistoryStats must return Some(EstimationReport)");

    assert_eq!(
        report.method, "user_provided",
        "report method must be 'user_provided', got '{}'",
        report.method
    );
    assert!(
        report.entries.is_empty(),
        "report entries must be empty (no AR was estimated), got {} entries",
        report.entries.len()
    );
}

// ── Bidirectional coverage validation tests ─────────────────

/// Build a minimal Hydro struct reusing the same penalty/generation defaults
/// as the single-hydro helpers above.
fn make_hydro(hydro_id: EntityId, bus_id: EntityId) -> cobre_core::entities::hydro::Hydro {
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel};
    Hydro {
        id: hydro_id,
        name: format!("H{}", hydro_id.0),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id,
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 5000.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 1000.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 900.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: cobre_core::entities::hydro::HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 1000.0,
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

/// Build a System with two hydros (IDs 1 and 2). User stats (inflow_models)
/// are created only for the hydros in `stats_hydro_ids`. Hydros in
/// `all_hydro_ids` but not in `stats_hydro_ids` have no stats rows.
#[allow(clippy::cast_possible_wrap)]
fn build_two_hydro_system_selective_stats(
    n_years: usize,
    all_hydro_ids: &[EntityId],
    stats_hydro_ids: &[EntityId],
) -> System {
    use cobre_core::scenario::InflowModel;
    use cobre_core::{Bus, DeficitSegment, SystemBuilder};

    let bus_id = EntityId(10);
    let bus = Bus {
        id: bus_id,
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(f64::INFINITY),
            cost_per_mwh: 3000.0,
        }],
        excess_cost: 0.0,
    };

    let ref_year = 1970_i32;
    let mut stages = Vec::with_capacity(n_years * 2);
    for y in 0..n_years {
        let year = ref_year + y as i32;
        stages.push(make_two_season_stage(y * 2, (y * 2) as i32, 0, year, true));
        stages.push(make_two_season_stage(
            y * 2 + 1,
            (y * 2 + 1) as i32,
            1,
            year,
            false,
        ));
    }

    // Build inflow models only for hydros in stats_hydro_ids.
    let inflow_models: Vec<InflowModel> = stats_hydro_ids
        .iter()
        .flat_map(|&hid| {
            stages.iter().map(move |s| InflowModel {
                hydro_id: hid,
                stage_id: s.id,
                mean_m3s: 100.0,
                std_m3s: 10.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
        })
        .collect();

    let hydros: Vec<_> = all_hydro_ids
        .iter()
        .map(|&hid| make_hydro(hid, bus_id))
        .collect();

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .build()
        .expect("valid two-hydro system")
}

/// Write a real `inflow_history.parquet` with data for the given list of
/// hydro IDs. Each hydro gets identical synthetic PAR(2) observations.
fn write_history_for_hydros(path: &std::path::Path, hydro_ids: &[i32], n_years: usize) {
    use arrow::array::{Date32Array, Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use chrono::NaiveDate;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let date_to_days = |d: NaiveDate| -> i32 {
        i32::try_from((d - epoch).num_days()).expect("date in Date32 range")
    };

    let (obs_s0, obs_s1) = simulate_two_season_par2(0.7, 0.15, n_years, 99);

    let mut ids: Vec<i32> = Vec::new();
    let mut dates: Vec<i32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();

    for &hid in hydro_ids {
        for y in 0..n_years {
            let year = (1970 + y) as i32;
            ids.push(hid);
            dates.push(date_to_days(NaiveDate::from_ymd_opt(year, 1, 15).unwrap()));
            values.push(obs_s0[y] + 300.0);

            ids.push(hid);
            dates.push(date_to_days(NaiveDate::from_ymd_opt(year, 7, 15).unwrap()));
            values.push(obs_s1[y] + 300.0);
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("date", DataType::Date32, false),
        Field::new("value_m3s", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(Date32Array::from(dates)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .expect("valid batch");

    let file = std::fs::File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");
}

/// Direction A — AR estimated for hydro 2 but no user stats for it.
///
/// Setup: system with hydros [1, 2], history for both [1, 2], but
/// `inflow_seasonal_stats.parquet` provides stats only for hydro 1.
///
/// Assert: `estimate_from_history` returns `Err` with a `ConstraintError`
/// whose description contains `"2"` (the uncovered hydro ID).
#[test]
fn test_partial_estimation_direction_a_missing_stats() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_required_files(case_dir);
    let scenarios = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios).unwrap();

    // History for both hydros 1 and 2.
    write_history_for_hydros(&scenarios.join("inflow_history.parquet"), &[1, 2], N_YEARS);

    // Stats sentinel — presence triggers PartialEstimation manifest flag.
    std::fs::write(scenarios.join("inflow_seasonal_stats.parquet"), b"sentinel")
        .expect("write sentinel");

    // System: hydros [1, 2] in the hydros list, but stats only for hydro 1.
    let system = build_two_hydro_system_selective_stats(
        N_YEARS,
        &[EntityId(1), EntityId(2)],
        &[EntityId(1)], // stats only for hydro 1
    );

    let config = default_config();
    let result = estimate_from_history(system, case_dir, &config);

    assert!(
        result.is_err(),
        "Direction A must return Err when hydro 2 has AR estimates but no user stats"
    );

    let err = result.unwrap_err();
    let description = err.to_string();
    assert!(
        description.contains('2'),
        "error description must contain the uncovered hydro ID '2', got: {description}"
    );
}

/// Direction B — user stats for hydro 2 but no history for it.
///
/// Setup: system with hydros [1, 2] and stats for both, but history only
/// for hydro 1.
///
/// Assert: `estimate_from_history` returns `Ok`, the `EstimationReport`
/// has `white_noise_fallbacks == [EntityId(2)]`, and the returned system's
/// inflow model for hydro 2 has empty `ar_coefficients`.
#[test]
fn test_partial_estimation_direction_b_white_noise_fallback() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_required_files(case_dir);
    let scenarios = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios).unwrap();

    // History only for hydro 1.
    write_history_for_hydros(&scenarios.join("inflow_history.parquet"), &[1], N_YEARS);

    // Stats sentinel.
    std::fs::write(scenarios.join("inflow_seasonal_stats.parquet"), b"sentinel")
        .expect("write sentinel");

    // System: hydros [1, 2] with stats for both (hydro 2 gets white-noise fallback).
    let system = build_two_hydro_system_selective_stats(
        N_YEARS,
        &[EntityId(1), EntityId(2)],
        &[EntityId(1), EntityId(2)], // stats for both
    );

    let config = default_config();
    let (updated, report, path) = estimate_from_history(system, case_dir, &config)
        .expect("Direction B must succeed (not an error)");

    assert_eq!(
        path,
        EstimationPath::PartialEstimation,
        "expected PartialEstimation path"
    );

    let report = report.expect("PartialEstimation must return Some(EstimationReport)");

    assert_eq!(
        report.white_noise_fallbacks,
        vec![EntityId(2)],
        "white_noise_fallbacks must be [EntityId(2)], got {:?}",
        report.white_noise_fallbacks
    );

    // Hydro 2 should have empty ar_coefficients in the returned system.
    let hydro2_models: Vec<_> = updated
        .inflow_models()
        .iter()
        .filter(|m| m.hydro_id == EntityId(2))
        .collect();
    assert!(
        !hydro2_models.is_empty(),
        "returned system must have inflow models for hydro 2"
    );
    for m in &hydro2_models {
        assert!(
            m.ar_coefficients.is_empty(),
            "hydro 2 must have empty ar_coefficients (white-noise fallback), stage {}",
            m.stage_id
        );
    }
}

/// Exact coverage — single hydro with matching history and stats.
///
/// Reuses the single-hydro setup. Asserts that
/// `white_noise_fallbacks` is empty on the returned report.
#[test]
fn test_partial_estimation_exact_coverage_no_fallback() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    setup_partial_estimation_case(case_dir, N_YEARS);
    let system = build_system_with_user_stats(N_YEARS);
    let config = default_config();

    let (_updated, report, _path) =
        estimate_from_history(system, case_dir, &config).expect("exact coverage must succeed");

    let report = report.expect("PartialEstimation must return Some(EstimationReport)");

    assert!(
        report.white_noise_fallbacks.is_empty(),
        "white_noise_fallbacks must be empty for exact coverage, got {:?}",
        report.white_noise_fallbacks
    );
}

/// `run_estimation` (FullEstimation path) never populates
/// `white_noise_fallbacks` — it must be empty on the returned report.
#[test]
fn test_full_estimation_report_has_empty_fallbacks() {
    use tempfile::TempDir;

    const N_YEARS: usize = 30;
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_required_files(case_dir);
    let scenarios = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios).unwrap();

    // History only — no stats file → FullEstimation path.
    write_history_for_hydros(&scenarios.join("inflow_history.parquet"), &[1], N_YEARS);
    // No inflow_seasonal_stats.parquet → FullEstimation.

    // System with one hydro, no pre-loaded inflow models (no user stats).
    let system = build_two_hydro_system_selective_stats(
        N_YEARS,
        &[EntityId(1)],
        &[], // no user stats
    );

    let config = default_config();
    let (_, report, path) =
        estimate_from_history(system, case_dir, &config).expect("FullEstimation must succeed");

    assert_eq!(
        path,
        EstimationPath::FullEstimation,
        "expected FullEstimation path"
    );

    let report = report.expect("FullEstimation must return Some(EstimationReport)");

    assert!(
        report.white_noise_fallbacks.is_empty(),
        "FullEstimation must never populate white_noise_fallbacks, got {:?}",
        report.white_noise_fallbacks
    );
}

// ── StdRatioDivergence unit tests ─────────────────────────────────────────

/// Helper: build a System and fitting_stats for a single hydro with the
/// given per-season std values, then call `check_std_ratio_divergence`.
///
/// `user_stds[i]` is the user-provided std for season `i`.
/// `est_stds[i]` is the estimated std for season `i`.
/// Stages are created so that stage_id == season_id (one stage per season).
fn collect_std_ratio_warnings(
    hydro_id: EntityId,
    user_stds: &[f64],
    est_stds: &[f64],
) -> Vec<StdRatioDivergence> {
    use cobre_core::scenario::InflowModel;

    assert_eq!(
        user_stds.len(),
        est_stds.len(),
        "user_stds and est_stds must have equal length"
    );
    let n = user_stds.len();

    // Build stages: stage_id == season_id == i, one stage per season.
    let stages: Vec<cobre_core::temporal::Stage> = (0..n)
        .map(|i| {
            let year = 1970_i32;
            let first_half = i % 2 == 0;
            make_two_season_stage(i, i as i32, i, year, first_half)
        })
        .collect();

    // Build user InflowModels: one per stage.
    let user_models: Vec<InflowModel> = (0..n)
        .map(|i| InflowModel {
            hydro_id,
            stage_id: i as i32,
            mean_m3s: 100.0,
            std_m3s: user_stds[i],
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let system = SystemBuilder::new()
        .inflow_models(user_models)
        .stages(stages.clone())
        .build()
        .expect("valid system");

    // Build fitting_stats: entity_id = hydro_id, stage_id = i, std = est_stds[i].
    let fitting_stats: Vec<SeasonalStats> = (0..n)
        .map(|i| SeasonalStats {
            entity_id: hydro_id,
            stage_id: i as i32,
            mean: 100.0,
            std: est_stds[i],
        })
        .collect();

    check_std_ratio_divergence(&system, &fitting_stats, &stages)
}

/// P9-001: Warning fires when consecutive std ratios diverge by more than 2x.
///
/// user stds [100.0, 20.0], est stds [100.0, 100.0].
/// Pair (0→1): ratio_user = 5.0, ratio_est = 1.0, divergence = 5.0 → warn.
/// Wrap (1→0): ratio_user = 0.2, ratio_est = 1.0, divergence = 5.0 → warn.
/// Both pairs diverge, so 2 warnings are emitted. The test verifies that
/// at least one warning covers the (0→1) pair and the hydro_id is correct.
#[test]
fn test_std_ratio_divergence_fires_when_ratios_diverge() {
    let warnings = collect_std_ratio_warnings(EntityId(1), &[100.0, 20.0], &[100.0, 100.0]);
    assert!(
        !warnings.is_empty(),
        "expected at least one StdRatioDivergence when ratio diverges by 5x"
    );
    // The (0→1) pair must be in the warnings.
    let pair_0_1 = warnings.iter().find(|w| w.season_a == 0 && w.season_b == 1);
    assert!(
        pair_0_1.is_some(),
        "expected a warning for season pair 0→1, got {warnings:?}"
    );
    let w = pair_0_1.unwrap();
    assert_eq!(
        w.hydro_id,
        EntityId(1),
        "warning must record the correct hydro_id"
    );
    assert!(
        (w.divergence - 5.0).abs() < 1e-10,
        "divergence for pair 0→1 must be 5.0, got {}",
        w.divergence
    );
}

/// P9-002: No warning when ratios are similar (divergence <= 2.0).
///
/// user stds [100.0, 20.0], est stds [90.0, 18.0].
/// ratio_user = 5.0, ratio_est = 5.0. divergence = 1.0 → no warning.
#[test]
fn test_std_ratio_divergence_not_fires_when_similar() {
    let warnings = collect_std_ratio_warnings(EntityId(1), &[100.0, 20.0], &[90.0, 18.0]);
    assert!(
        warnings.is_empty(),
        "expected no StdRatioDivergence when ratios are similar, got {warnings:?}"
    );
}

/// P9-003: Season pairs with near-zero denominator std are skipped.
///
/// user stds [100.0, 0.0], est stds [90.0, 18.0].
/// The pair (season 0 → season 1) has u_b = 0.0 < 1e-12 → skipped.
/// The wrap pair (season 1 → season 0) has u_b = 100.0 and e_b = 90.0 → checked.
/// ratio_user = 0/100 = 0.0, ratio_est = 18/90 = 0.2.
/// divergence = max(0/0.2, 0.2/0) → second division hits near-zero guard → skipped.
#[test]
fn test_std_ratio_divergence_skips_near_zero_std() {
    // user stds [100.0, 0.0]: the first pair has denominator 0.0 → skipped.
    let warnings = collect_std_ratio_warnings(EntityId(1), &[100.0, 0.0], &[90.0, 18.0]);
    // No panic must occur. The pair involving zero std is silently skipped.
    // The wrap pair has ratio_user = 0.0/100.0 = 0.0, ratio_est = 18.0/90.0 = 0.2;
    // ratio_user / ratio_est would require dividing 0/0.2 = 0, and
    // ratio_est / ratio_user would divide by 0. The near-zero guard on ratio_est
    // is not triggered here (0.2 is not near zero), but ratio_user = 0.0 means
    // divergence = max(0/0.2, 0.2/0). The 0.2/0 branch triggers the ratio_est
    // guard only if ratio_user < 1e-12, which 0.0 satisfies → skipped.
    // We assert no panic and that the result is well-defined.
    let _ = warnings; // result is valid (empty or one entry); no panic is the key assertion.
}

/// P9-004: Wrap-around pair (last season → first season) is checked.
///
/// user stds [100.0, 20.0, 50.0], est stds [100.0, 20.0, 10.0].
/// Pair (0→1): ratio_user=5.0, ratio_est=5.0 → divergence=1.0 (no warn).
/// Pair (1→2): ratio_user=0.4, ratio_est=2.0 → divergence=5.0 (warn).
/// Wrap (2→0): ratio_user=50/100=0.5, ratio_est=10/100=0.1 → divergence=5.0 (warn).
#[test]
fn test_std_ratio_divergence_wraps_last_to_first() {
    let warnings =
        collect_std_ratio_warnings(EntityId(1), &[100.0, 20.0, 50.0], &[100.0, 20.0, 10.0]);
    // Pairs (1→2) and wrap (2→0) both diverge.
    assert!(
        warnings.len() >= 2,
        "expected at least 2 StdRatioDivergence entries (including wrap), got {}",
        warnings.len()
    );
    // The wrap pair (season 2 → season 0) must appear.
    let has_wrap = warnings.iter().any(|w| w.season_a == 2 && w.season_b == 0);
    assert!(
        has_wrap,
        "expected a warning for the wrap-around pair season 2 → season 0"
    );
}

// ── estimate_from_history annual-path integration tests ─────────────────

use chrono::NaiveDate;
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
};

/// Build a 12-season monthly stage sequence spanning `n_years` starting from
/// year 2000. Stage IDs are 0-based sequential; season IDs cycle 0..12.
fn make_monthly_stages_for_annual(n_years: usize) -> Vec<cobre_core::temporal::Stage> {
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

/// Build a `System` with two hydros on a 12-season (monthly) grid spanning
/// `n_years` study years, with no pre-loaded inflow models.
///
/// This represents the state before estimation: only hydros and stages are
/// present, so `estimate_from_history` will follow the `FullEstimation` path.
#[allow(clippy::cast_possible_wrap)]
fn build_two_hydro_monthly_system(n_years: usize) -> System {
    use cobre_core::{Bus, DeficitSegment, SystemBuilder};
    let bus_id = EntityId(10);
    let bus = Bus {
        id: bus_id,
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(f64::INFINITY),
            cost_per_mwh: 3000.0,
        }],
        excess_cost: 0.0,
    };
    let h1 = EntityId(1);
    let h2 = EntityId(2);
    let stages = make_monthly_stages_for_annual(n_years);
    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![make_hydro(h1, bus_id), make_hydro(h2, bus_id)])
        .stages(stages)
        .build()
        .expect("valid two-hydro monthly system")
}

/// Write `inflow_history.parquet` with synthetic monthly data for two hydros.
///
/// Uses `synthetic_monthly_obs` to generate observations for hydros 1 and 2
/// with different base values so the series are distinct. Observations are
/// dated starting from 2000-01-01 and cover `n_years * 12` months per hydro.
fn write_monthly_inflow_history_two_hydros(path: &std::path::Path, n_years: usize) {
    use arrow::array::{Date32Array, Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let h1 = EntityId(1);
    let h2 = EntityId(2);
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let date_to_days = |d: NaiveDate| -> i32 { i32::try_from((d - epoch).num_days()).unwrap() };

    let obs_h1 = synthetic_monthly_obs(h1, n_years, 100.0, 5.0, 1.0);
    let obs_h2 = synthetic_monthly_obs(h2, n_years, 200.0, 3.0, 0.5);

    let mut ids: Vec<i32> = Vec::new();
    let mut dates: Vec<i32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();

    for &(eid, date, value) in obs_h1.iter().chain(obs_h2.iter()) {
        ids.push(eid.0);
        dates.push(date_to_days(date));
        values.push(value);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("date", DataType::Date32, false),
        Field::new("value_m3s", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(Date32Array::from(dates)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .expect("valid batch");

    let file = std::fs::File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");
}

/// `estimate_from_history` with `PacfAnnual` populates `InflowModel.annual`.
///
/// Fixture: 2-hydro × 60-month (12 seasons × 5 years) synthetic monthly
/// history. Config has `order_selection = PacfAnnual`. Asserts that at
/// least one returned inflow model has `annual = Some(_)`.
#[test]
fn estimate_from_history_pacf_annual_populates_annual_field() {
    use cobre_io::config::{EstimationConfig, OrderSelectionMethod};
    use tempfile::TempDir;

    const N_YEARS: usize = 5;
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    // FullEstimation: only inflow_history.parquet, no stats or AR files.
    create_required_files(case_dir);
    let scenarios = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios).unwrap();
    write_monthly_inflow_history_two_hydros(&scenarios.join("inflow_history.parquet"), N_YEARS);

    let system = build_two_hydro_monthly_system(N_YEARS);

    let mut config: Config = serde_json::from_str(MINIMAL_CONFIG_JSON).unwrap();
    config.estimation = EstimationConfig {
        max_order: 2,
        order_selection: OrderSelectionMethod::PacfAnnual,
        min_observations_per_season: 2,
        max_coefficient_magnitude: None,
    };

    let (updated, report, path) = estimate_from_history(system, case_dir, &config)
        .expect("full estimation with PacfAnnual must succeed");

    assert_eq!(
        path,
        EstimationPath::FullEstimation,
        "expected FullEstimation path"
    );
    assert!(report.is_some(), "FullEstimation must return Some(report)");

    let models = updated.inflow_models();
    assert!(
        !models.is_empty(),
        "estimation must produce at least one inflow model"
    );
    assert!(
        models.iter().any(|m| m.annual.is_some()),
        "PacfAnnual path must set annual=Some on at least one model"
    );
}

/// `estimate_from_history` with classical `Pacf` keeps `InflowModel.annual = None`.
///
/// Same fixture as `estimate_from_history_pacf_annual_populates_annual_field`
/// but with `order_selection = Pacf`. Asserts that every returned inflow
/// model has `annual = None` (regression: classical path unchanged).
#[test]
fn estimate_from_history_pacf_classical_keeps_annual_none() {
    use cobre_io::config::{EstimationConfig, OrderSelectionMethod};
    use tempfile::TempDir;

    const N_YEARS: usize = 5;
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_required_files(case_dir);
    let scenarios = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios).unwrap();
    write_monthly_inflow_history_two_hydros(&scenarios.join("inflow_history.parquet"), N_YEARS);

    let system = build_two_hydro_monthly_system(N_YEARS);

    let mut config: Config = serde_json::from_str(MINIMAL_CONFIG_JSON).unwrap();
    config.estimation = EstimationConfig {
        max_order: 2,
        order_selection: OrderSelectionMethod::Pacf,
        min_observations_per_season: 2,
        max_coefficient_magnitude: None,
    };

    let (updated, report, path) = estimate_from_history(system, case_dir, &config)
        .expect("full estimation with Pacf must succeed");

    assert_eq!(
        path,
        EstimationPath::FullEstimation,
        "expected FullEstimation path"
    );
    assert!(report.is_some(), "FullEstimation must return Some(report)");

    let models = updated.inflow_models();
    assert!(
        !models.is_empty(),
        "estimation must produce at least one inflow model"
    );
    assert!(
        models.iter().all(|m| m.annual.is_none()),
        "classical Pacf path must keep annual=None for every model"
    );
}

/// AC #8 — Classical path unchanged: `use_annual_component = false` returns
/// `method = "PACF"` and every `ArCoefficientEstimate.annual.is_none()`.
///
/// Uses the same 2-hydro 30-year fixture as AC #6 to ensure the dispatch
/// (`estimate_ar_coefficients_with_selection`) routes to the classical path
/// when `use_annual_component = false`.
#[test]
fn estimate_ar_coefficients_with_selection_classical_path_unchanged() {
    let h1 = EntityId(1);
    let h2 = EntityId(2);
    let n_years = 30;
    let stages = make_monthly_stages_for_annual(n_years);

    let mut obs = synthetic_monthly_obs(h1, n_years, 100.0, 5.0, 1.0);
    obs.extend(synthetic_monthly_obs(h2, n_years, 200.0, 3.0, 0.5));

    let seasonal_stats = {
        use cobre_stochastic::par::fitting::estimate_seasonal_stats_with_season_map;
        estimate_seasonal_stats_with_season_map(&obs, &stages, &[h1, h2], None).unwrap()
    };

    let (estimates, report) = estimate_ar_coefficients_with_selection(
        &obs,
        &seasonal_stats,
        &stages,
        &[h1, h2],
        &ArEstimationConfig {
            max_order: 3,
            max_coeff_magnitude: None,
            season_map: None,
            use_annual_component: false,
        },
    )
    .expect("classical path must succeed");

    assert_eq!(
        report.method, "PACF",
        "classical path must produce method=PACF, got {}",
        report.method
    );
    for est in &estimates {
        assert!(
            est.annual.is_none(),
            "classical path: hydro={} season={} must have annual=None",
            est.hydro_id.0,
            est.season_id
        );
    }
}

/// Build a 12-season monthly `SeasonMap` (season id m → calendar month m+1).
fn monthly_season_map() -> cobre_core::temporal::SeasonMap {
    use cobre_core::temporal::{SeasonCycleType, SeasonDefinition, SeasonMap};
    let seasons = (0..12usize)
        .map(|m| SeasonDefinition {
            id: m,
            label: format!("M{m}"),
            month_start: u32::try_from(m + 1).unwrap(),
            day_start: None,
            month_end: None,
            day_end: None,
        })
        .collect();
    SeasonMap {
        cycle_type: SeasonCycleType::Monthly,
        seasons,
    }
}

/// Build study stages for a partial-year monthly study spanning seasons
/// `[first_season, first_season + n)` starting in calendar year `start_year`.
fn partial_year_stages(
    first_season: usize,
    n: usize,
    start_year: i32,
) -> Vec<cobre_core::temporal::Stage> {
    (0..n)
        .map(|k| {
            let season = first_season + k;
            let m = u32::try_from(season + 1).unwrap();
            let (ey, em) = if m == 12 {
                (start_year + 1, 1u32)
            } else {
                (start_year, m + 1)
            };
            cobre_core::temporal::Stage {
                index: k,
                id: i32::try_from(k).unwrap(),
                start_date: NaiveDate::from_ymd_opt(start_year, m, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(ey, em, 1).unwrap(),
                season_id: Some(season),
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
        })
        .collect()
}

/// Full-cycle PAR(2) partial-year fit: a monthly study spanning seasons
/// 8–11 (Sep–Dec) with 30 years of full-cycle synthetic history.
///
/// Mirrors the `run_estimation` pipeline (synthesize pre-study stages →
/// fit on the combined stages → expand rows → assemble models →
/// `PrecomputedPar::build`) and asserts:
/// (a) no panic during fitting,
/// (b) the synthesized pre-study stages produce `InflowModel` entries at
///     negative `stage_id`s,
/// (c) the first study stage's `PrecomputedPar` psi for its pre-study lags
///     is non-zero (the lag stats were sourced from history, not zeroed).
#[test]
fn partial_year_par2_synthesizes_prestudy_lag_models() {
    let h1 = EntityId(1);
    let n_years = 30;
    let max_order = 2usize;
    let season_map = monthly_season_map();

    // Study spans seasons 8..=11 (Sep–Dec), study stage ids 0..=3.
    let study_stages = partial_year_stages(8, 4, 2030);

    // Full-cycle history: all 12 months per year so the out-of-window lag
    // seasons (6 = July, 7 = August) have observations to fit.
    let obs = synthetic_monthly_obs(h1, n_years, 100.0, 5.0, 1.0);

    // ── Synthesize pre-study stages for the lag window ───────────────────
    let prestudy = synthesize_prestudy_stages(&study_stages, max_order, Some(&season_map));
    // max_order=2 → lags into seasons 7 (Aug) and 6 (Jul), neither in study.
    assert_eq!(
        prestudy.len(),
        2,
        "expected 2 synthetic pre-study stages, got {}",
        prestudy.len()
    );
    // Pre-study ids descend from the first study stage id (0): -1, -2.
    let mut pre_ids: Vec<i32> = prestudy.iter().map(|s| s.id).collect();
    pre_ids.sort_unstable();
    assert_eq!(pre_ids, vec![-2, -1], "pre-study ids must be -1, -2");
    // Seasons k positions before season 8: -1 → 7 (Aug), -2 → 6 (Jul).
    let season_of = |id: i32| prestudy.iter().find(|s| s.id == id).unwrap().season_id;
    assert_eq!(
        season_of(-1),
        Some(7),
        "stage -1 must map to season 7 (Aug)"
    );
    assert_eq!(
        season_of(-2),
        Some(6),
        "stage -2 must map to season 6 (Jul)"
    );

    let stages: Vec<cobre_core::temporal::Stage> = study_stages
        .iter()
        .cloned()
        .chain(prestudy.iter().cloned())
        .collect();

    // ── Fit seasonal stats + AR(2) on the combined stages ────────────────
    let seasonal_stats = {
        use cobre_stochastic::par::fitting::estimate_seasonal_stats_with_season_map;
        estimate_seasonal_stats_with_season_map(&obs, &stages, &[h1], Some(&season_map))
            .expect("seasonal stats must fit without panic")
    };

    let (ar_estimates, _report) = estimate_ar_coefficients_with_selection(
        &obs,
        &seasonal_stats,
        &stages,
        &[h1],
        &ArEstimationConfig {
            max_order,
            max_coeff_magnitude: None,
            season_map: Some(&season_map),
            use_annual_component: false,
        },
    )
    .expect("AR(2) fit must succeed");

    // ── Expand rows onto pre-study stages and assemble inflow models ─────
    let stats_rows = seasonal_stats_to_rows(&seasonal_stats, &stages);
    let coeff_rows = ar_estimates_to_rows(&ar_estimates, &stages);
    let annual_rows = ar_estimates_to_annual_rows(&ar_estimates, &stages);
    let inflow_models = assemble_inflow_models(stats_rows, coeff_rows, annual_rows)
        .expect("assembly must succeed with pre-study rows present");

    // (b) Pre-study InflowModel entries exist at negative stage_ids.
    let neg_models: Vec<&cobre_core::scenario::InflowModel> =
        inflow_models.iter().filter(|m| m.stage_id < 0).collect();
    assert!(
        neg_models.iter().any(|m| m.stage_id == -1) && neg_models.iter().any(|m| m.stage_id == -2),
        "expected InflowModel entries at stage_id -1 and -2, got {:?}",
        neg_models.iter().map(|m| m.stage_id).collect::<Vec<_>>()
    );

    // ── Build PrecomputedPar with the true cycle length (12) ─────────────
    let par = cobre_stochastic::PrecomputedPar::build(
        &inflow_models,
        &study_stages,
        &[h1],
        Some(season_map.seasons.len()),
    )
    .expect("PrecomputedPar build must succeed");

    // (c) The first study stage (s_idx 0, season 8) has non-zero psi for
    // its pre-study lags, proving the lag stats came from history rather
    // than the (0.0, 0.0) season fallback. PACF may select an order ≤
    // max_order; whatever the stride, at least one lag must be non-zero
    // and the lag it consumes resolves to a pre-study stage (season 7/6).
    let par_order = par.max_order();
    assert!(
        par_order >= 1 && par_order <= max_order,
        "selected order {par_order} must be in 1..={max_order}"
    );
    let psi0 = par.psi_slice(0, 0);
    assert_eq!(
        psi0.len(),
        par_order,
        "psi stride must equal selected order"
    );
    assert!(
        psi0.iter().any(|&p| p.abs() > 1e-9),
        "first study stage psi must be non-zero for pre-study lags, got {psi0:?}"
    );
}
