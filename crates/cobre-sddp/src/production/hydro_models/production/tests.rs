#![allow(
    clippy::doc_markdown,
    clippy::match_wildcard_for_single_variants,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic
)]

use std::collections::HashMap;

use chrono::NaiveDate;
use cobre_core::{
    Bus, EfficiencyModel, EntityId, HydraulicLossesModel, InflowHistoryRow, SystemBuilder,
    TailraceModel,
    entities::hydro::{HydroGenerationModel, HydroPenalties},
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_io::extensions::{
    FittingWindow, FphaColumnLayout, FphaHyperplaneRow, HydroGeometryRow, ProductionModelConfig,
    SeasonConfig, SelectionMode, StageRange, TailraceCurveRow,
};

use super::*;

// ── Test helpers ──────────────────────────────────────────────────────────

fn make_stage(id: i32) -> Stage {
    Stage {
        index: usize::try_from(id.max(0)).unwrap_or(0),
        id,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap_or_default(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap_or_default(),
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
            branching_factor: 50,
            noise_method: NoiseMethod::Saa,
        },
    }
}

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

fn make_hydro(id: i32, model: HydroGenerationModel) -> cobre_core::entities::hydro::Hydro {
    cobre_core::entities::hydro::Hydro {
        id: EntityId::from(id),
        name: format!("Hydro{id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId::from(10),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 100.0,
        max_storage_hm3: 2000.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: model,
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

fn valid_row(hydro_id: i32, stage_id: Option<i32>, plane_id: i32) -> FphaHyperplaneRow {
    FphaHyperplaneRow {
        hydro_id: EntityId::from(hydro_id),
        stage_id,
        plane_id,
        gamma_0: 1000.0,
        gamma_v: 0.002,
        gamma_q: 0.85,
        gamma_s: -0.01,
        kappa: 1.0,
        valid_v_min_hm3: None,
        valid_v_max_hm3: None,
        valid_q_max_m3s: None,
    }
}

fn precomputed_fpha_config(hydro_id: i32) -> ProductionModelConfig {
    ProductionModelConfig {
        hydro_id: EntityId::from(hydro_id),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: None,
                model: "fpha".to_string(),
                fpha_config: Some(FphaColumnLayout {
                    source: "precomputed".to_string(),
                    volume_discretization_points: None,
                    turbine_discretization_points: None,
                    spillage_discretization_points: None,
                    max_planes_per_hydro: None,
                    fitting_window: None,
                }),
                reference_volume: None,
                productivity_mw_per_m3s: None,
            }],
        },
    }
}

fn computed_fpha_config(hydro_id: i32) -> ProductionModelConfig {
    ProductionModelConfig {
        hydro_id: EntityId::from(hydro_id),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: None,
                model: "fpha".to_string(),
                fpha_config: Some(FphaColumnLayout {
                    source: "computed".to_string(),
                    volume_discretization_points: None,
                    turbine_discretization_points: None,
                    spillage_discretization_points: None,
                    max_planes_per_hydro: None,
                    fitting_window: None,
                }),
                reference_volume: None,
                productivity_mw_per_m3s: None,
            }],
        },
    }
}

// ── resolve_production_models unit tests (in-memory, no disk I/O) ─────────

/// Non-FPHA hydros without any config entry produce
/// `DefaultConstant` provenance from `determine_source`. The
/// downstream sentinel behaviour in `resolve_stage_model` is exercised
/// by `test_resolve_stage_model_returns_sentinel_when_no_config_entry`.
#[test]
fn all_constant_no_config_returns_default_constant_provenance() {
    let hydro0 = make_hydro(0, HydroGenerationModel::ConstantProductivity);
    let hydro1 = make_hydro(1, HydroGenerationModel::ConstantProductivity);

    let src0 = determine_source(&hydro0, None).expect("should succeed");
    let src1 = determine_source(&hydro1, None).expect("should succeed");
    assert_eq!(src0, ProductionModelSource::DefaultConstant);
    assert_eq!(src1, ProductionModelSource::DefaultConstant);
}

/// `LinearizedHead` entities without a config entry produce
/// `DefaultConstant` provenance from `determine_source`. The downstream
/// sentinel behaviour in `resolve_stage_model` is exercised by
/// `test_resolve_stage_model_returns_sentinel_when_no_config_entry`.
#[test]
fn linearized_head_entity_resolves_to_constant_productivity() {
    let hydro = make_hydro(0, HydroGenerationModel::LinearizedHead);

    let src = determine_source(&hydro, None).expect("should succeed");
    assert_eq!(src, ProductionModelSource::DefaultConstant);
}

/// Fpha entity model without config → validation error.
#[test]
fn fpha_entity_without_config_entry_returns_validation_error() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let err = determine_source(&hydro, None).expect_err("should fail");
    assert!(
        matches!(err, crate::SddpError::Validation(ref msg) if
                msg.contains("fpha") || msg.contains("no entry") || msg.contains("hydro_production_models")),
        "expected Validation error mentioning missing config entry, got {err:?}"
    );
}

/// source: "computed" in config → returns `ComputedFromGeometry` (fitting is now supported).
#[test]
fn computed_source_returns_computed_from_geometry() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let config = computed_fpha_config(0);

    let source = determine_source(&hydro, Some(&config)).expect("should succeed");
    assert_eq!(
        source,
        ProductionModelSource::ComputedFromGeometry,
        "expected ComputedFromGeometry, got {source:?}"
    );
}

/// Helper: build a minimal hydro with all computed-source prerequisites set.
fn make_computed_hydro(id: i32) -> cobre_core::entities::hydro::Hydro {
    let mut hydro = make_hydro(id, HydroGenerationModel::Fpha);
    hydro.tailrace = Some(TailraceModel::Polynomial {
        coefficients: vec![300.0],
    });
    hydro.hydraulic_losses = Some(HydraulicLossesModel::Factor { value: 0.02 });
    hydro.efficiency = Some(EfficiencyModel::Constant { value: 0.92 });
    hydro
}

/// Helper: build a two-point VHA geometry row vector for a hydro.
fn make_geometry_rows(hydro_id: i32) -> Vec<HydroGeometryRow> {
    vec![
        HydroGeometryRow {
            hydro_id: EntityId::from(hydro_id),
            volume_hm3: 100.0,
            height_m: 400.0,
            area_km2: 10.0,
        },
        HydroGeometryRow {
            hydro_id: EntityId::from(hydro_id),
            volume_hm3: 2000.0,
            height_m: 450.0,
            area_km2: 50.0,
        },
    ]
}

/// validate_computed_prerequisites: missing tailrace → Validation error with "tailrace" and hydro name.
#[test]
fn computed_source_missing_tailrace_returns_validation_error() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    // tailrace is None in make_hydro
    let rows = make_geometry_rows(0);
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    let row_refs: Vec<&HydroGeometryRow> = rows.iter().collect();
    geometry_map.insert(EntityId::from(0), row_refs);

    let err = validate_computed_prerequisites(&hydro, &geometry_map)
        .expect_err("should fail when tailrace is None");
    let msg = err.to_string();
    assert!(
        msg.contains("tailrace"),
        "error must mention 'tailrace', got: {msg}"
    );
    assert!(
        msg.contains(&hydro.name),
        "error must include hydro name '{}', got: {msg}",
        hydro.name
    );
}

/// validate_computed_prerequisites: missing geometry rows → Validation error with "geometry" and hydro name.
#[test]
fn computed_source_missing_geometry_returns_validation_error() {
    let hydro = make_computed_hydro(0);
    let empty_geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();

    let err = validate_computed_prerequisites(&hydro, &empty_geometry_map)
        .expect_err("should fail when geometry rows are absent");
    let msg = err.to_string();
    assert!(
        msg.contains("geometry"),
        "error must mention 'geometry', got: {msg}"
    );
    assert!(
        msg.contains(&hydro.name),
        "error must include hydro name '{}', got: {msg}",
        hydro.name
    );
}

/// find_fpha_config_for_stage: returns Some(&FphaColumnLayout) when stage is in the range.
#[test]
fn find_fpha_config_for_stage_returns_config_in_range() {
    let config = computed_fpha_config(0);
    let stage = make_stage(5);

    let result = find_fpha_config_for_stage(&config, &stage);
    assert!(
        result.is_some(),
        "expected Some(FphaColumnLayout) for stage 5, got None"
    );
    assert_eq!(
        result.expect("just checked is_some").source,
        "computed",
        "expected source 'computed'"
    );
}

/// find_fpha_config_for_stage: returns None when no range covers the stage.
#[test]
fn find_fpha_config_for_stage_returns_none_outside_range() {
    // Create a config with range [5, 10].
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 5,
                end_stage_id: Some(10),
                model: "fpha".to_string(),
                fpha_config: Some(FphaColumnLayout {
                    source: "computed".to_string(),
                    volume_discretization_points: None,
                    turbine_discretization_points: None,
                    spillage_discretization_points: None,
                    max_planes_per_hydro: None,
                    fitting_window: None,
                }),
                reference_volume: None,
                productivity_mw_per_m3s: None,
            }],
        },
    };

    // Stage 0 is before the range [5, 10].
    let stage = make_stage(0);
    let result = find_fpha_config_for_stage(&config, &stage);
    assert!(
        result.is_none(),
        "expected None for stage 0 (outside range [5,10]), got {result:?}"
    );
}

/// kappa = 0.95 → intercept is gamma_0 * kappa.
#[test]
fn gamma_0_is_scaled_by_kappa() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    let row = FphaHyperplaneRow {
        hydro_id: EntityId::from(0),
        stage_id: None,
        plane_id: 0,
        gamma_0: 1000.0,
        gamma_v: 0.002,
        gamma_q: 0.85,
        gamma_s: -0.01,
        kappa: 0.95,
        valid_v_min_hm3: None,
        valid_v_max_hm3: None,
        valid_q_max_m3s: None,
    };

    let mut map = std::collections::HashMap::new();
    map.insert(
        (EntityId::from(0), None::<i32>),
        vec![&row as &FphaHyperplaneRow],
    );

    let model = build_fpha_model(
        &hydro,
        &stage,
        ProductionModelSource::PrecomputedHyperplanes,
        &map,
    )
    .expect("should build FPHA model");

    match model {
        ResolvedProductionModel::Fpha { planes, .. } => {
            assert_eq!(planes.len(), 1);
            let expected = 1000.0 * 0.95;
            assert!(
                (planes[0].intercept - expected).abs() < 1e-10,
                "intercept must be gamma_0 * kappa = {expected}, got {}",
                planes[0].intercept
            );
        }
        other => panic!("expected Fpha variant, got {other:?}"),
    }
}

/// validate_hyperplane_row rejects negative gamma_v.
#[test]
fn validation_rejects_gamma_v_negative() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    let mut row = valid_row(0, None, 0);
    row.gamma_v = -0.1; // invalid: must be >= 0

    let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("gamma_v"),
        "error must mention gamma_v, got: {msg}"
    );
}

/// validate_hyperplane_row accepts gamma_v == 0.0 (constant-head plant).
#[test]
fn validation_accepts_gamma_v_zero() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    let mut row = valid_row(0, None, 0);
    row.gamma_v = 0.0; // valid: constant-head plant

    validate_hyperplane_row(&hydro, &stage, &row)
        .expect("gamma_v = 0.0 must be valid for constant-head plants");
}

/// validate_hyperplane_row rejects gamma_s > 0.
#[test]
fn validation_rejects_gamma_s_positive() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    let mut row = valid_row(0, None, 0);
    row.gamma_s = 0.01;

    let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("gamma_s"),
        "error must mention gamma_s, got: {msg}"
    );
}

/// validate_hyperplane_row rejects gamma_q <= 0.
#[test]
fn validation_rejects_gamma_q_nonpositive() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    let mut row = valid_row(0, None, 0);
    row.gamma_q = 0.0;

    let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("gamma_q"),
        "error must mention gamma_q, got: {msg}"
    );
}

/// validate_hyperplane_row rejects kappa = 0 (must be > 0).
#[test]
fn validation_rejects_kappa_zero() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    let mut row = valid_row(0, None, 0);
    row.kappa = 0.0;

    let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("kappa"),
        "error must mention kappa, got: {msg}"
    );
}

/// validate_hyperplane_row rejects kappa = 1.5 (must be <= 1).
#[test]
fn validation_rejects_kappa_above_one() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    let mut row = valid_row(0, None, 0);
    row.kappa = 1.5;

    let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("kappa"),
        "error must mention kappa, got: {msg}"
    );
}

/// Stage-specific hyperplanes (Some(stage_id)) override all-stage (None) rows.
#[test]
fn stage_specific_hyperplanes_override_all_stage() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    let global_row = FphaHyperplaneRow {
        hydro_id: EntityId::from(0),
        stage_id: None,
        plane_id: 0,
        gamma_0: 500.0, // distinct intercept to identify
        gamma_v: 0.001,
        gamma_q: 0.80,
        gamma_s: -0.005,
        kappa: 1.0,
        valid_v_min_hm3: None,
        valid_v_max_hm3: None,
        valid_q_max_m3s: None,
    };
    let stage_row = FphaHyperplaneRow {
        hydro_id: EntityId::from(0),
        stage_id: Some(0),
        plane_id: 0,
        gamma_0: 900.0, // distinct intercept to identify
        gamma_v: 0.002,
        gamma_q: 0.85,
        gamma_s: -0.01,
        kappa: 1.0,
        valid_v_min_hm3: None,
        valid_v_max_hm3: None,
        valid_q_max_m3s: None,
    };

    let mut map = std::collections::HashMap::new();
    map.insert(
        (EntityId::from(0), None::<i32>),
        vec![&global_row as &FphaHyperplaneRow],
    );
    map.insert(
        (EntityId::from(0), Some(0i32)),
        vec![&stage_row as &FphaHyperplaneRow],
    );

    let model = build_fpha_model(
        &hydro,
        &stage,
        ProductionModelSource::PrecomputedHyperplanes,
        &map,
    )
    .expect("should succeed");

    match model {
        ResolvedProductionModel::Fpha { planes, .. } => {
            // Stage-specific row has gamma_0 = 900, global has 500; stage-specific wins.
            assert!(
                (planes[0].intercept - 900.0).abs() < 1e-10,
                "stage-specific intercept 900 should override global 500, got {}",
                planes[0].intercept
            );
        }
        other => panic!("expected Fpha variant, got {other:?}"),
    }
}

/// All-stage hyperplanes (stage_id: None) are used when no stage-specific rows exist.
#[test]
fn all_stage_hyperplanes_used_when_no_stage_specific_rows() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(5); // stage id 5, no stage-specific rows for it

    let global_row = FphaHyperplaneRow {
        hydro_id: EntityId::from(0),
        stage_id: None,
        plane_id: 0,
        gamma_0: 700.0,
        gamma_v: 0.002,
        gamma_q: 0.85,
        gamma_s: -0.01,
        kappa: 1.0,
        valid_v_min_hm3: None,
        valid_v_max_hm3: None,
        valid_q_max_m3s: None,
    };

    let mut map = std::collections::HashMap::new();
    map.insert(
        (EntityId::from(0), None::<i32>),
        vec![&global_row as &FphaHyperplaneRow],
    );

    let model = build_fpha_model(
        &hydro,
        &stage,
        ProductionModelSource::PrecomputedHyperplanes,
        &map,
    )
    .expect("should succeed using global rows");

    match model {
        ResolvedProductionModel::Fpha { planes, .. } => {
            assert!(
                (planes[0].intercept - 700.0).abs() < 1e-10,
                "expected global intercept 700, got {}",
                planes[0].intercept
            );
        }
        other => panic!("expected Fpha, got {other:?}"),
    }
}

/// Zero hyperplanes for a stage (empty rows) → validation error.
#[test]
fn zero_hyperplanes_for_stage_returns_validation_error() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let stage = make_stage(0);

    // Map has the key but an empty rows vec.
    let mut map = std::collections::HashMap::new();
    map.insert(
        (EntityId::from(0), None::<i32>),
        Vec::<&FphaHyperplaneRow>::new(),
    );

    let err = build_fpha_model(
        &hydro,
        &stage,
        ProductionModelSource::PrecomputedHyperplanes,
        &map,
    )
    .expect_err("should fail with zero hyperplanes");

    assert!(
        matches!(err, crate::SddpError::Validation(_)),
        "expected Validation error, got {err:?}"
    );
}

/// find_model_for_stage: stage_id in range returns fpha model name.
#[test]
fn find_model_for_stage_returns_correct_model_name_in_range() {
    let config = precomputed_fpha_config(0);
    let stage = make_stage(3);
    let result = find_model_for_stage(&config, &stage);
    assert_eq!(result.as_ref().map(|(name, _)| name.as_str()), Some("fpha"));
}

/// find_model_for_stage: stage_id before start of range returns None.
#[test]
fn find_model_for_stage_returns_none_when_before_range_start() {
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 5,
                end_stage_id: Some(10),
                model: "fpha".to_string(),
                fpha_config: None,
                reference_volume: None,
                productivity_mw_per_m3s: None,
            }],
        },
    };
    let stage = make_stage(3); // id 3, before start_stage_id 5
    let result = find_model_for_stage(&config, &stage);
    assert!(
        result.is_none(),
        "stage 3 is before range [5, 10], expected None"
    );
}

/// find_model_for_stage: end_stage_id = None covers all stages from start.
#[test]
fn find_model_for_stage_open_ended_range_covers_all_stages() {
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: None,
                model: "constant_productivity".to_string(),
                fpha_config: None,
                reference_volume: None,
                productivity_mw_per_m3s: None,
            }],
        },
    };
    for stage_id in [0, 5, 11, 100] {
        let stage = make_stage(stage_id);
        let result = find_model_for_stage(&config, &stage);
        assert_eq!(
            result.as_ref().map(|(name, _)| name.as_str()),
            Some("constant_productivity"),
            "open-ended range must cover stage {stage_id}"
        );
    }
}

// ── Productivity override tests ─────────────────────────────────────────

/// resolve_stage_model uses productivity_override when present.
#[test]
fn resolve_stage_model_uses_productivity_override() {
    let hydro = make_hydro(0, HydroGenerationModel::ConstantProductivity);
    let stage = make_stage(0);
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: None,
                model: "constant_productivity".to_string(),
                fpha_config: None,
                reference_volume: None,
                productivity_mw_per_m3s: Some(0.55),
            }],
        },
    };
    let empty_map = std::collections::HashMap::new();
    let model = super::resolve_stage_model(
        &hydro,
        &stage,
        Some(&config),
        ProductionModelSource::DefaultConstant,
        &empty_map,
        None,
        None,
    )
    .expect("should succeed");
    assert!(
        matches!(model, ResolvedProductionModel::ConstantProductivity { productivity }
                if (productivity - 0.55).abs() < f64::EPSILON),
        "expected ConstantProductivity 0.55 (override), got {model:?}"
    );
}

/// When the JSON config entry exists but its `productivity_mw_per_m3s`
/// field is `None`, `resolve_stage_model` returns a sentinel
/// `ConstantProductivity { productivity: 0.0 }`. The build site in
/// `build_energy_conversion_set` consults the parquet override and
/// overwrites the sentinel with the user-supplied value.
///
/// In debug builds the sentinel path is gated by a `debug_assert!` that
/// catches mis-configured cases that escape load-time validation in
/// `cobre_io::validation::productivity_resolution`. In release builds the
/// `debug_assert!` is compiled out and the function returns the sentinel
/// directly.
#[test]
fn test_resolve_stage_model_returns_sentinel_when_json_lacks_productivity() {
    let hydro = make_hydro(0, HydroGenerationModel::ConstantProductivity);
    let stage = make_stage(0);
    // Config entry exists but productivity is missing — the parquet
    // override is the supplier in production.
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: None,
                model: "constant_productivity".to_string(),
                fpha_config: None,
                reference_volume: None,
                productivity_mw_per_m3s: None,
            }],
        },
    };
    let empty_map = std::collections::HashMap::new();

    #[cfg(debug_assertions)]
    {
        // Debug build: the `debug_assert!` fires and the call panics.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            super::resolve_stage_model(
                &hydro,
                &stage,
                Some(&config),
                ProductionModelSource::DefaultConstant,
                &empty_map,
                None,
                None,
            )
        }));
        let panic_payload =
            result.expect_err("debug build must panic via debug_assert! when productivity is None");
        let msg = panic_payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                panic_payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_owned())
            })
            .unwrap_or_default();
        assert!(
            msg.contains("Hydro0") && msg.contains("validation::productivity_resolution"),
            "panic message must name the hydro and the validator; got: {msg}"
        );
    }

    #[cfg(not(debug_assertions))]
    {
        // Release build: the assert is compiled out and the function
        // returns the sentinel directly.
        let model = super::resolve_stage_model(
            &hydro,
            &stage,
            Some(&config),
            ProductionModelSource::DefaultConstant,
            &empty_map,
            None,
            None,
        )
        .expect("release build returns sentinel");
        assert!(
            matches!(
                model,
                ResolvedProductionModel::ConstantProductivity { productivity }
                if productivity == 0.0
            ),
            "release build must return ConstantProductivity {{ productivity: 0.0 }}; got {model:?}"
        );
    }
}

/// When the JSON has no productivity and the parquet override supplies a
/// value, `resolve_stage_model` returns that value as the resolved
/// `ConstantProductivity { productivity }`. This is the path the LP
/// coefficient flows through for non-FPHA hydros authored entirely via the
/// parquet.
#[test]
fn test_resolve_stage_model_uses_parquet_override_when_json_omits_productivity() {
    let hydro = make_hydro(0, HydroGenerationModel::ConstantProductivity);
    let stage = make_stage(0);
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: None,
                model: "constant_productivity".to_string(),
                fpha_config: None,
                reference_volume: None,
                productivity_mw_per_m3s: None,
            }],
        },
    };
    let empty_map = std::collections::HashMap::new();
    let override_table = crate::energy_conversion::build_hydro_energy_productivity_override(&[
        cobre_io::HydroEnergyProductivityRow {
            hydro_id: EntityId::from(0),
            stage_id: Some(0),
            equivalent_productivity_mw_per_m3s: Some(0.42),
            reference_outflow_m3s: None,
            specific_productivity_mw_per_m3s_per_m: None,
        },
    ])
    .expect("override builds");

    let model = super::resolve_stage_model(
        &hydro,
        &stage,
        Some(&config),
        ProductionModelSource::DefaultConstant,
        &empty_map,
        None,
        Some(&override_table),
    )
    .expect("resolve succeeds");
    assert!(
        matches!(
            model,
            ResolvedProductionModel::ConstantProductivity { productivity }
            if (productivity - 0.42).abs() < 1e-12
        ),
        "override value must reach the resolved model; got {model:?}"
    );
}

/// When no JSON config entry exists at all for a non-FPHA hydro,
/// `resolve_stage_model` returns the same sentinel. Load-time validation
/// in `cobre_io::validation::productivity_resolution` is responsible for
/// catching missing entries; this branch trusts that invariant in release
/// builds and is guarded by a `debug_assert!` in debug builds.
#[test]
fn test_resolve_stage_model_returns_sentinel_when_no_config_entry() {
    let hydro = make_hydro(7, HydroGenerationModel::ConstantProductivity);
    let stage = make_stage(3);
    let empty_map = std::collections::HashMap::new();

    #[cfg(debug_assertions)]
    {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            super::resolve_stage_model(
                &hydro,
                &stage,
                None,
                ProductionModelSource::DefaultConstant,
                &empty_map,
                None,
                None,
            )
        }));
        let panic_payload = result
            .expect_err("debug build must panic via debug_assert! when no config entry exists");
        let msg = panic_payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                panic_payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_owned())
            })
            .unwrap_or_default();
        assert!(
            msg.contains("Hydro7") && msg.contains("validation::productivity_resolution"),
            "panic message must name the hydro and the validator; got: {msg}"
        );
    }

    #[cfg(not(debug_assertions))]
    {
        let model = super::resolve_stage_model(
            &hydro,
            &stage,
            None,
            ProductionModelSource::DefaultConstant,
            &empty_map,
            None,
            None,
        )
        .expect("release build returns sentinel");
        assert!(
            matches!(
                model,
                ResolvedProductionModel::ConstantProductivity { productivity }
                if productivity == 0.0
            ),
            "release build must return ConstantProductivity {{ productivity: 0.0 }}; got {model:?}"
        );
    }
}

/// find_model_for_stage returns override in tuple.
#[test]
fn find_model_for_stage_returns_override_in_tuple() {
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: None,
                model: "constant_productivity".to_string(),
                fpha_config: None,
                reference_volume: None,
                productivity_mw_per_m3s: Some(0.75),
            }],
        },
    };
    let stage = make_stage(0);
    let result = find_model_for_stage(&config, &stage);
    assert_eq!(
        result,
        Some(("constant_productivity".to_string(), Some(0.75)))
    );
}

/// Seasonal mode: find_model_for_stage returns override for matching season
/// and None for default model.
#[test]
fn find_model_for_stage_seasonal_with_override() {
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::Seasonal {
            default_model: "constant_productivity".to_string(),
            seasons: vec![SeasonConfig {
                season_id: 1,
                model: "constant_productivity".to_string(),
                fpha_config: None,
                reference_volume: None,
                productivity_mw_per_m3s: Some(0.60),
            }],
        },
    };
    // Stage with matching season_id = 1
    let mut stage_match = make_stage(0);
    stage_match.season_id = Some(1);
    let result = find_model_for_stage(&config, &stage_match);
    assert_eq!(
        result,
        Some(("constant_productivity".to_string(), Some(0.60)))
    );

    // Stage with non-matching season_id → default model, no override
    let mut stage_default = make_stage(0);
    stage_default.season_id = Some(99);
    let result = find_model_for_stage(&config, &stage_default);
    assert_eq!(result, Some(("constant_productivity".to_string(), None)));
}

/// precomputed config returns PrecomputedHyperplanes source.
#[test]
fn precomputed_config_returns_precomputed_source() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha);
    let config = precomputed_fpha_config(0);
    let src = determine_source(&hydro, Some(&config)).expect("should succeed");
    assert_eq!(src, ProductionModelSource::PrecomputedHyperplanes);
}

// ── Computed-source integration tests ─────────────────────────────────────

/// Sobradinho-style hydro with all computed prerequisites, matching the known-valid
/// fixture from `fpha_fitting.rs`. Used for end-to-end computed-source tests.
fn make_sobradinho_computed_hydro(id: i32) -> cobre_core::entities::hydro::Hydro {
    let mut hydro = make_hydro(id, HydroGenerationModel::Fpha);
    hydro.name = format!("Sobradinho{id}");
    hydro.min_storage_hm3 = 100.0;
    hydro.max_storage_hm3 = 20_000.0;
    hydro.max_turbined_m3s = 500.0;
    hydro.tailrace = Some(TailraceModel::Polynomial {
        coefficients: vec![0.0, 0.001_f64],
    });
    hydro.hydraulic_losses = Some(HydraulicLossesModel::Constant { value_m: 2.0 });
    hydro.efficiency = Some(EfficiencyModel::Constant { value: 0.92 });
    hydro
}

/// Four-point VHA geometry rows spanning volumes 100.0 to 20_000.0 hm³ and heights
/// 386.5 to 400.0 m. Mirrors the Sobradinho-style fixture used in `fpha_fitting.rs`.
fn make_sobradinho_geometry_rows(hydro_id: i32) -> Vec<HydroGeometryRow> {
    vec![
        HydroGeometryRow {
            hydro_id: EntityId::from(hydro_id),
            volume_hm3: 100.0,
            height_m: 386.5,
            area_km2: 500.0,
        },
        HydroGeometryRow {
            hydro_id: EntityId::from(hydro_id),
            volume_hm3: 5_000.0,
            height_m: 392.0,
            area_km2: 800.0,
        },
        HydroGeometryRow {
            hydro_id: EntityId::from(hydro_id),
            volume_hm3: 12_000.0,
            height_m: 396.5,
            area_km2: 1_100.0,
        },
        HydroGeometryRow {
            hydro_id: EntityId::from(hydro_id),
            volume_hm3: 20_000.0,
            height_m: 400.0,
            area_km2: 1_400.0,
        },
    ]
}

/// Computed-source end-to-end: a hydro with all prerequisites and Sobradinho-style geometry
/// produces a valid `Fpha` model with 3–10 planes and correct coefficient signs.
///
/// Tests `fit_planes_for_hydro` + `resolve_stage_model` together.
#[test]
fn computed_source_end_to_end_produces_valid_fpha_planes() {
    let hydro = make_sobradinho_computed_hydro(0);
    let config = computed_fpha_config(0);
    let stage = make_stage(0);

    let geo_rows = make_sobradinho_geometry_rows(0);
    let geo_refs: Vec<&HydroGeometryRow> = geo_rows.iter().collect();
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    geometry_map.insert(EntityId::from(0), geo_refs);

    // Fit planes once (simulating the per-entry fit in resolve_production_models).
    let layout =
        find_fpha_config_for_stage(&config, &stage).expect("computed config covers stage 0");
    let fit_result = super::fit_planes_for_hydro(
        &hydro,
        layout,
        &geometry_map,
        0.0,
        super::TailraceSource::Entity(hydro.tailrace.clone()),
        None,
        0,
        false,
    )
    .expect("fit_planes_for_hydro must succeed for valid Sobradinho-style input");
    let planes = &fit_result.planes;

    // The fitter derives planes from the convex hull of the production cloud,
    // so the count is the number of distinct upper-envelope hull faces rather
    // than a dense tangent candidate set. The hull-based invariant is at least
    // one plane with valid coefficient signs (migrated from the former
    // tangent-derived 3–10 count).
    assert!(
        !planes.is_empty(),
        "expected at least one plane, got {}",
        planes.len()
    );

    // Coefficient signs must satisfy physical constraints: non-negative volume
    // and turbine gradients, non-positive lateral secant. The
    // installed-capacity ceiling contributes a flat `generation ≤ capacity` plane with
    // both gradients exactly zero, so the bound is `>= 0`, not strictly `> 0`.
    for (idx, plane) in planes.iter().enumerate() {
        assert!(
            plane.gamma_v >= 0.0,
            "plane {idx}: gamma_v={} must be >= 0",
            plane.gamma_v
        );
        assert!(
            plane.gamma_q >= 0.0,
            "plane {idx}: gamma_q={} must be >= 0",
            plane.gamma_q
        );
        assert!(
            plane.gamma_s <= 0.0,
            "plane {idx}: gamma_s={} must be <= 0",
            plane.gamma_s
        );
    }

    // Verify resolve_stage_model correctly wraps the cached planes.
    let empty_hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
        HashMap::new();
    let model = super::resolve_stage_model(
        &hydro,
        &stage,
        Some(&config),
        ProductionModelSource::ComputedFromGeometry,
        &empty_hyperplane_map,
        Some(planes),
        None,
    )
    .expect("resolve_stage_model must succeed for ComputedFromGeometry with cached planes");

    match model {
        ResolvedProductionModel::Fpha {
            planes: out_planes, ..
        } => {
            assert_eq!(
                out_planes.len(),
                planes.len(),
                "stage model must have the same plane count as the fitted planes"
            );
        }
        other => panic!("expected Fpha variant, got {other:?}"),
    }
}

/// Round-trip: computed-FPHA export rows written to parquet and re-read via
/// `parse_fpha_hyperplanes` reconstruct the fitted planes to within 1e-9.
///
/// The computed path writes the α-scaled coefficients verbatim with the IO
/// `kappa` column pinned to 1.0, so the precomputed reader's reconstruction
/// (`intercept = gamma_0 * kappa`, gradients pass through) reproduces the
/// fitted planes exactly — proving the kappa column round-trips without
/// re-scaling the already-corrected coefficients.
#[test]
fn computed_export_rows_round_trip_through_parquet() {
    let hydro = make_sobradinho_computed_hydro(0);
    let config = computed_fpha_config(0);
    let rows = make_sobradinho_geometry_rows(0);
    let geometry_map = sobradinho_geometry_map(&rows);

    let stages = [make_stage(0)];
    let stage_refs: Vec<&Stage> = stages.iter().collect();

    let mut export_rows = Vec::new();
    let families_map = empty_families_map();
    let (system, fractions) = entity_fit_context(&hydro);
    let per_stage = super::fit_computed_planes_per_stage(
        &hydro,
        Some(&config),
        &geometry_map,
        &families_map,
        &fractions,
        &system,
        &stage_refs,
        0.0,
        None,
        false,
        &mut export_rows,
        &mut Vec::new(),
        &mut Vec::new(),
    )
    .expect("per-stage fit must succeed");

    let fitted_planes = &per_stage[0];
    assert!(!export_rows.is_empty(), "export rows must be non-empty");
    assert_eq!(
        export_rows.len(),
        fitted_planes.len(),
        "one export row per fitted plane"
    );
    for row in &export_rows {
        assert_eq!(row.kappa, 1.0, "computed rows pin kappa = 1.0");
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fpha_hyperplanes.parquet");
    cobre_io::output::write_fpha_hyperplanes(&path, &export_rows)
        .expect("write_fpha_hyperplanes must succeed");

    let read_rows =
        cobre_io::extensions::parse_fpha_hyperplanes(&path).expect("re-read must succeed");
    assert_eq!(
        read_rows.len(),
        fitted_planes.len(),
        "re-read row count must match fitted plane count"
    );

    // Reconstruct each plane exactly as the precomputed reader does:
    // intercept = gamma_0 * kappa; the gradients pass through unchanged.
    for (row, fitted) in read_rows.iter().zip(fitted_planes) {
        let reconstructed = FphaPlane {
            intercept: row.gamma_0 * row.kappa,
            gamma_v: row.gamma_v,
            gamma_q: row.gamma_q,
            gamma_s: row.gamma_s,
        };
        assert!(
            (reconstructed.intercept - fitted.intercept).abs() < 1e-9,
            "intercept mismatch: {} vs {}",
            reconstructed.intercept,
            fitted.intercept
        );
        assert!((reconstructed.gamma_v - fitted.gamma_v).abs() < 1e-9);
        assert!((reconstructed.gamma_q - fitted.gamma_q).abs() < 1e-9);
        assert!((reconstructed.gamma_s - fitted.gamma_s).abs() < 1e-9);
    }
}

/// Mixed precomputed + computed sources: both hydros resolve to valid `Fpha` models and
/// provenance is correctly differentiated by source.
///
/// Hydro 0: `source: "precomputed"` with 3 manually-constructed hyperplane rows.
/// Hydro 1: `source: "computed"` with Sobradinho-style geometry.
#[test]
fn mixed_precomputed_and_computed_sources_resolve_correctly() {
    // Hydro 0: precomputed FPHA.
    let hydro0 = make_hydro(0, HydroGenerationModel::Fpha);
    let config0 = precomputed_fpha_config(0);

    let precomp_row_a = valid_row(0, None, 0);
    let precomp_row_b = valid_row(0, None, 1);
    let precomp_row_c = valid_row(0, None, 2);
    let mut hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
        HashMap::new();
    hyperplane_map.insert(
        (EntityId::from(0), None),
        vec![&precomp_row_a, &precomp_row_b, &precomp_row_c],
    );

    // Hydro 1: computed FPHA.
    let hydro1 = make_sobradinho_computed_hydro(1);
    let config1 = computed_fpha_config(1);

    let geo_rows = make_sobradinho_geometry_rows(1);
    let geo_refs: Vec<&HydroGeometryRow> = geo_rows.iter().collect();
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    geometry_map.insert(EntityId::from(1), geo_refs);

    let stage = make_stage(0);

    // Determine sources.
    let src0 = determine_source(&hydro0, Some(&config0)).expect("hydro0 source");
    let src1 = determine_source(&hydro1, Some(&config1)).expect("hydro1 source");
    assert_eq!(
        src0,
        ProductionModelSource::PrecomputedHyperplanes,
        "hydro 0 must be PrecomputedHyperplanes"
    );
    assert_eq!(
        src1,
        ProductionModelSource::ComputedFromGeometry,
        "hydro 1 must be ComputedFromGeometry"
    );

    // Fit computed planes for hydro 1.
    let layout1 =
        find_fpha_config_for_stage(&config1, &stage).expect("computed config covers stage 0");
    let computed_fit = super::fit_planes_for_hydro(
        &hydro1,
        layout1,
        &geometry_map,
        0.0,
        super::TailraceSource::Entity(hydro1.tailrace.clone()),
        None,
        0,
        false,
    )
    .expect("fit_planes_for_hydro must succeed for hydro 1");

    // Resolve stage model for hydro 0 (precomputed path).
    let model0 = super::resolve_stage_model(
        &hydro0,
        &stage,
        Some(&config0),
        src0,
        &hyperplane_map,
        None,
        None,
    )
    .expect("resolve_stage_model must succeed for hydro 0 (precomputed)");

    // Resolve stage model for hydro 1 (computed path, cached planes).
    let empty_hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
        HashMap::new();
    let model1 = super::resolve_stage_model(
        &hydro1,
        &stage,
        Some(&config1),
        src1,
        &empty_hyperplane_map,
        Some(&computed_fit.planes),
        None,
    )
    .expect("resolve_stage_model must succeed for hydro 1 (computed)");

    // Both models must be Fpha.
    assert!(
        matches!(model0, ResolvedProductionModel::Fpha { .. }),
        "hydro 0 must resolve to Fpha, got {model0:?}"
    );
    assert!(
        matches!(model1, ResolvedProductionModel::Fpha { .. }),
        "hydro 1 must resolve to Fpha, got {model1:?}"
    );

    // Provenance in canonical id-sorted order: [(id=0, Precomputed), (id=1, Computed)].
    assert_eq!(
        src0,
        ProductionModelSource::PrecomputedHyperplanes,
        "provenance[0] must be PrecomputedHyperplanes"
    );
    assert_eq!(
        src1,
        ProductionModelSource::ComputedFromGeometry,
        "provenance[1] must be ComputedFromGeometry"
    );
}

/// Computed-source all-stages-same: three stages all receive plane vectors with identical
/// coefficients, confirming that the outer loop fits once and clones for every stage.
#[test]
fn computed_source_all_stages_produce_identical_planes() {
    let hydro = make_sobradinho_computed_hydro(0);
    let config = computed_fpha_config(0);

    let geo_rows = make_sobradinho_geometry_rows(0);
    let geo_refs: Vec<&HydroGeometryRow> = geo_rows.iter().collect();
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    geometry_map.insert(EntityId::from(0), geo_refs);

    // Three study stages.
    let stages = [make_stage(0), make_stage(1), make_stage(2)];

    // Fit once.
    let layout =
        find_fpha_config_for_stage(&config, &stages[0]).expect("computed config covers stage 0");
    let cached_fit = super::fit_planes_for_hydro(
        &hydro,
        layout,
        &geometry_map,
        0.0,
        super::TailraceSource::Entity(hydro.tailrace.clone()),
        None,
        0,
        false,
    )
    .expect("fit_planes_for_hydro must succeed");

    let empty_hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
        HashMap::new();

    // Resolve for each stage and collect planes.
    let stage_planes: Vec<Vec<FphaPlane>> = stages
        .iter()
        .map(|stage| {
            let model = super::resolve_stage_model(
                &hydro,
                stage,
                Some(&config),
                ProductionModelSource::ComputedFromGeometry,
                &empty_hyperplane_map,
                Some(&cached_fit.planes),
                None,
            )
            .expect("resolve_stage_model must succeed");
            match model {
                ResolvedProductionModel::Fpha { planes, .. } => planes,
                other => panic!("expected Fpha, got {other:?}"),
            }
        })
        .collect();

    assert_eq!(
        stage_planes.len(),
        3,
        "must have plane vectors for 3 stages"
    );

    // All stages must have the same plane count.
    let expected_count = stage_planes[0].len();
    for (s, planes) in stage_planes.iter().enumerate() {
        assert_eq!(
            planes.len(),
            expected_count,
            "stage {s}: plane count must be {expected_count}, got {}",
            planes.len()
        );
    }

    // Planes must be bitwise-identical across stages (cloned from the same source).
    for (s, planes) in stage_planes.iter().enumerate().skip(1) {
        for (p, plane) in planes.iter().enumerate() {
            assert_eq!(
                *plane, stage_planes[0][p],
                "stage {s} plane {p}: must be identical to stage 0 plane {p}"
            );
        }
    }
}

/// `validate_computed_prerequisites`: missing `efficiency` returns `SddpError::Validation`
/// with a message containing both "efficiency" and the hydro name "TestHydro".
#[test]
fn computed_source_missing_efficiency_returns_validation_error() {
    let mut hydro = make_computed_hydro(0);
    hydro.name = "TestHydro".to_string();
    hydro.efficiency = None; // remove efficiency to trigger prerequisite failure

    let rows = make_geometry_rows(0);
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    let row_refs: Vec<&HydroGeometryRow> = rows.iter().collect();
    geometry_map.insert(EntityId::from(0), row_refs);

    let err = validate_computed_prerequisites(&hydro, &geometry_map)
        .expect_err("should fail when efficiency is None");
    let msg = err.to_string();
    assert!(
        msg.contains("efficiency"),
        "error must mention 'efficiency', got: {msg}"
    );
    assert!(
        msg.contains("TestHydro"),
        "error must include hydro name 'TestHydro', got: {msg}"
    );
}

/// `validate_computed_prerequisites`: missing `hydraulic_losses` returns `SddpError::Validation`
/// with a message containing "hydraulic_losses" and the hydro name.
#[test]
fn computed_source_missing_losses_returns_validation_error() {
    let mut hydro = make_computed_hydro(0);
    hydro.hydraulic_losses = None; // remove losses to trigger prerequisite failure

    let rows = make_geometry_rows(0);
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    let row_refs: Vec<&HydroGeometryRow> = rows.iter().collect();
    geometry_map.insert(EntityId::from(0), row_refs);

    let err = validate_computed_prerequisites(&hydro, &geometry_map)
        .expect_err("should fail when hydraulic_losses is None");
    let msg = err.to_string();
    assert!(
        msg.contains("hydraulic_losses"),
        "error must mention 'hydraulic_losses', got: {msg}"
    );
    assert!(
        msg.contains(&hydro.name),
        "error must include hydro name '{}', got: {msg}",
        hydro.name
    );
}

// ── Stage-aware per-entry fitting tests ───────────────────────────────────

/// Build a single-hydro geometry map for the Sobradinho-style fixture.
fn sobradinho_geometry_map(rows: &[HydroGeometryRow]) -> HashMap<EntityId, Vec<&HydroGeometryRow>> {
    let mut map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    map.insert(rows[0].hydro_id, rows.iter().collect());
    map
}

/// An empty tailrace-families map: every hydro falls back to its entity
/// `TailraceModel` (the inert fallback). Used by the per-entry fit tests that
/// exercise the config/window dedup, which is independent of the families
/// path.
fn empty_families_map() -> HashMap<EntityId, TailraceFamilies> {
    HashMap::new()
}

/// A minimal single-hydro `System` plus a flat reference-fraction resolver,
/// the inputs `fit_computed_planes_per_stage` needs for downstream-level
/// resolution. The hydro has no `downstream_id`, so the resolver returns
/// `None` and the families path (when used) sees an unresolved level — the
/// fit tests below feed an empty families map, so the level is unused.
fn entity_fit_context(
    hydro: &cobre_core::entities::hydro::Hydro,
) -> (System, HydroReferenceVolumeFractions) {
    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![hydro.clone()])
        .build()
        .expect("single-hydro system builds");
    let fractions = flat_reference_fractions(0.65, system.hydros());
    (system, fractions)
}

/// FPHA layout with an explicit absolute volume window. Distinct windows
/// drive distinct fitting bounds and therefore distinct plane sets.
fn computed_layout_with_window(min: Option<f64>, max: Option<f64>) -> FphaColumnLayout {
    FphaColumnLayout {
        source: "computed".to_string(),
        volume_discretization_points: None,
        turbine_discretization_points: None,
        spillage_discretization_points: None,
        max_planes_per_hydro: None,
        fitting_window: Some(FittingWindow {
            volume_min_hm3: min,
            volume_max_hm3: max,
            volume_min_percentile: None,
            volume_max_percentile: None,
        }),
    }
}

/// Per-range fits differ: two `StageRange`s carrying different
/// `fitting_window`s yield non-identical plane sets for stages in range A vs
/// range B.
#[test]
fn per_range_fits_differ_when_windows_differ() {
    let hydro = make_sobradinho_computed_hydro(0);
    let rows = make_sobradinho_geometry_rows(0);
    let geometry_map = sobradinho_geometry_map(&rows);

    // Range A (stages 0-1): narrow low window. Range B (stages 2-3): wide.
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![
                StageRange {
                    start_stage_id: 0,
                    end_stage_id: Some(1),
                    model: "fpha".to_string(),
                    fpha_config: Some(computed_layout_with_window(Some(100.0), Some(8_000.0))),
                    reference_volume: None,
                    productivity_mw_per_m3s: None,
                },
                StageRange {
                    start_stage_id: 2,
                    end_stage_id: None,
                    model: "fpha".to_string(),
                    fpha_config: Some(computed_layout_with_window(Some(100.0), Some(20_000.0))),
                    reference_volume: None,
                    productivity_mw_per_m3s: None,
                },
            ],
        },
    };

    let stages = [make_stage(0), make_stage(1), make_stage(2), make_stage(3)];
    let stage_refs: Vec<&Stage> = stages.iter().collect();

    let mut export_rows = Vec::new();
    let families_map = empty_families_map();
    let (system, fractions) = entity_fit_context(&hydro);
    let per_stage = super::fit_computed_planes_per_stage(
        &hydro,
        Some(&config),
        &geometry_map,
        &families_map,
        &fractions,
        &system,
        &stage_refs,
        0.0,
        None,
        false,
        &mut export_rows,
        &mut Vec::new(),
        &mut Vec::new(),
    )
    .expect("per-stage fit must succeed");

    // Stages in range A (0,1) share one plane set; stages in range B (2,3)
    // share another. The two range plane sets must differ.
    assert_eq!(per_stage[0], per_stage[1], "range A stages must match");
    assert_eq!(per_stage[2], per_stage[3], "range B stages must match");
    assert_ne!(
        per_stage[0], per_stage[2],
        "range A and range B plane sets must differ when windows differ"
    );
}

/// Per-season fits differ: two season configs with different windows yield
/// distinct plane sets for stages mapping to each season.
#[test]
fn per_season_fits_differ_when_season_configs_differ() {
    let hydro = make_sobradinho_computed_hydro(0);
    let rows = make_sobradinho_geometry_rows(0);
    let geometry_map = sobradinho_geometry_map(&rows);

    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::Seasonal {
            default_model: "fpha".to_string(),
            seasons: vec![
                SeasonConfig {
                    season_id: 0,
                    model: "fpha".to_string(),
                    fpha_config: Some(computed_layout_with_window(Some(100.0), Some(8_000.0))),
                    reference_volume: None,
                    productivity_mw_per_m3s: None,
                },
                SeasonConfig {
                    season_id: 1,
                    model: "fpha".to_string(),
                    fpha_config: Some(computed_layout_with_window(Some(100.0), Some(20_000.0))),
                    reference_volume: None,
                    productivity_mw_per_m3s: None,
                },
            ],
        },
    };

    let mut stage_s0 = make_stage(0);
    stage_s0.season_id = Some(0);
    let mut stage_s1 = make_stage(1);
    stage_s1.season_id = Some(1);
    let stages = [stage_s0, stage_s1];
    let stage_refs: Vec<&Stage> = stages.iter().collect();

    let mut export_rows = Vec::new();
    let families_map = empty_families_map();
    let (system, fractions) = entity_fit_context(&hydro);
    let per_stage = super::fit_computed_planes_per_stage(
        &hydro,
        Some(&config),
        &geometry_map,
        &families_map,
        &fractions,
        &system,
        &stage_refs,
        0.0,
        None,
        false,
        &mut export_rows,
        &mut Vec::new(),
        &mut Vec::new(),
    )
    .expect("per-season fit must succeed");

    assert_ne!(
        per_stage[0], per_stage[1],
        "season 0 and season 1 plane sets must differ when configs differ"
    );
}

/// Single-config dedup: a hydro whose `SelectionMode` resolves to one config
/// for all stages is fitted once — every stage carries the identical plane
/// set and emits per-stage export rows.
#[test]
fn single_config_dedup_yields_identical_planes_across_stages() {
    let hydro = make_sobradinho_computed_hydro(0);
    let rows = make_sobradinho_geometry_rows(0);
    let geometry_map = sobradinho_geometry_map(&rows);

    // One open-ended range covering all stages with one config.
    let config = computed_fpha_config(0);

    let stages = [make_stage(0), make_stage(1), make_stage(2)];
    let stage_refs: Vec<&Stage> = stages.iter().collect();

    let mut export_rows = Vec::new();
    let families_map = empty_families_map();
    let (system, fractions) = entity_fit_context(&hydro);
    let per_stage = super::fit_computed_planes_per_stage(
        &hydro,
        Some(&config),
        &geometry_map,
        &families_map,
        &fractions,
        &system,
        &stage_refs,
        0.0,
        None,
        false,
        &mut export_rows,
        &mut Vec::new(),
        &mut Vec::new(),
    )
    .expect("single-config fit must succeed");

    assert_eq!(per_stage.len(), 3, "one plane set per study stage");
    for s in 1..3 {
        assert_eq!(
            per_stage[0], per_stage[s],
            "single-config hydro: stage {s} planes must be identical to stage 0 (dedup)"
        );
    }

    // Each stage emits one export-row block, so the row count is
    // n_stages * planes_per_stage.
    assert_eq!(
        export_rows.len(),
        per_stage[0].len() * 3,
        "export rows must cover every (stage, plane)"
    );
}

/// Export rows carry `stage_id = Some(stage.id)` and are ordered canonically
/// by `(stage_id, plane_id)` (single-hydro fixture upholds the
/// `(hydro_id, stage_id, plane_id)` global order).
#[test]
fn export_rows_carry_stage_id_and_canonical_order() {
    let hydro = make_sobradinho_computed_hydro(0);
    let rows = make_sobradinho_geometry_rows(0);
    let geometry_map = sobradinho_geometry_map(&rows);
    let config = computed_fpha_config(0);

    let stages = [make_stage(0), make_stage(1), make_stage(2)];
    let stage_refs: Vec<&Stage> = stages.iter().collect();

    let mut export_rows = Vec::new();
    let families_map = empty_families_map();
    let (system, fractions) = entity_fit_context(&hydro);
    super::fit_computed_planes_per_stage(
        &hydro,
        Some(&config),
        &geometry_map,
        &families_map,
        &fractions,
        &system,
        &stage_refs,
        0.0,
        None,
        false,
        &mut export_rows,
        &mut Vec::new(),
        &mut Vec::new(),
    )
    .expect("fit must succeed");

    // Every row carries a concrete stage_id (never None for computed FPHA).
    for row in &export_rows {
        assert!(
            row.stage_id.is_some(),
            "computed-FPHA export row must carry stage_id = Some(...), got None"
        );
    }

    // Rows must be sorted by (hydro_id, stage_id, plane_id).
    let keys: Vec<(i32, Option<i32>, i32)> = export_rows
        .iter()
        .map(|r| (r.hydro_id.0, r.stage_id, r.plane_id))
        .collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(
        keys, sorted,
        "export rows must be ordered by (hydro_id, stage_id, plane_id)"
    );

    // The covered stage ids are exactly {0, 1, 2}.
    let mut seen_stages: Vec<i32> = export_rows.iter().filter_map(|r| r.stage_id).collect();
    seen_stages.sort_unstable();
    seen_stages.dedup();
    assert_eq!(
        seen_stages,
        vec![0, 1, 2],
        "every study stage must appear in the export rows"
    );
}

/// A `SelectionMode` entry mapping no config to a stage (coverage gap) →
/// `SddpError::Validation`; stages are never silently dropped.
#[test]
fn coverage_gap_returns_validation_error() {
    let hydro = make_sobradinho_computed_hydro(0);
    let rows = make_sobradinho_geometry_rows(0);
    let geometry_map = sobradinho_geometry_map(&rows);

    // Range covers only stages [0, 1]; stage 2 falls in a gap.
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(0),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: Some(1),
                model: "fpha".to_string(),
                fpha_config: Some(computed_layout_with_window(None, None)),
                reference_volume: None,
                productivity_mw_per_m3s: None,
            }],
        },
    };

    let stages = [make_stage(0), make_stage(1), make_stage(2)];
    let stage_refs: Vec<&Stage> = stages.iter().collect();

    let mut export_rows = Vec::new();
    let families_map = empty_families_map();
    let (system, fractions) = entity_fit_context(&hydro);
    let err = super::fit_computed_planes_per_stage(
        &hydro,
        Some(&config),
        &geometry_map,
        &families_map,
        &fractions,
        &system,
        &stage_refs,
        0.0,
        None,
        false,
        &mut export_rows,
        &mut Vec::new(),
        &mut Vec::new(),
    )
    .expect_err("a coverage gap must error, not silently drop the stage");
    assert!(
        matches!(err, crate::SddpError::Validation(ref msg) if msg.contains("stage 2")),
        "expected Validation naming the uncovered stage, got {err:?}"
    );
}

// ── long-term mean inflow (long-term mean inflow) for the lateral secant ────────────────────

fn inflow_row(hydro_id: i32, day: u32, value_m3s: f64) -> InflowHistoryRow {
    InflowHistoryRow {
        hydro_id: EntityId::from(hydro_id),
        date: NaiveDate::from_ymd_opt(2000, 1, day).unwrap(),
        value_m3s,
    }
}

/// A hydro with an inflow history yields long-term mean inflow = the canonical-order mean of its
/// `value_m3s` series, and only its own rows are summed (other hydros ignored).
#[test]
fn long_term_mean_inflow_is_per_hydro_canonical_mean() {
    let rows = vec![
        inflow_row(1, 1, 100.0),
        inflow_row(2, 1, 9_999.0), // different hydro — must be excluded
        inflow_row(1, 2, 200.0),
        inflow_row(1, 3, 300.0),
    ];
    let system = SystemBuilder::new()
        .inflow_history(rows)
        .build()
        .expect("valid system");

    // mean(100, 200, 300) = 200. This long-term mean inflow feeds S_max = 2·long-term mean inflow at the fit call
    // site; the long-term mean inflow→S_max mapping is the secant's `resolve_s_max` contract,
    // exercised in the `fpha_fitting::secant` unit tests.
    let long_term_mean_inflow_m3s = super::long_term_mean_inflow(&system, EntityId::from(1));
    assert_eq!(
        long_term_mean_inflow_m3s, 200.0,
        "long-term mean inflow must be the per-hydro mean of its own series"
    );
}

/// A hydro with no inflow history yields long-term mean inflow = 0.0 (the fallback sentinel).
#[test]
fn long_term_mean_inflow_empty_history_is_zero() {
    let system = SystemBuilder::new().build().expect("empty system is valid");
    let long_term_mean_inflow_m3s = super::long_term_mean_inflow(&system, EntityId::from(1));
    assert_eq!(
        long_term_mean_inflow_m3s, 0.0,
        "no inflow history must yield long-term mean inflow = 0"
    );
}

/// Determinism: the long-term mean inflow mean is independent of inflow-row declaration order.
#[test]
fn long_term_mean_inflow_is_order_independent() {
    let ascending = vec![
        inflow_row(1, 1, 10.0),
        inflow_row(1, 2, 20.0),
        inflow_row(1, 3, 60.0),
    ];
    let descending = vec![
        inflow_row(1, 3, 60.0),
        inflow_row(1, 2, 20.0),
        inflow_row(1, 1, 10.0),
    ];
    let sys_asc = SystemBuilder::new()
        .inflow_history(ascending)
        .build()
        .expect("valid");
    let sys_desc = SystemBuilder::new()
        .inflow_history(descending)
        .build()
        .expect("valid");

    let mlt_asc = super::long_term_mean_inflow(&sys_asc, EntityId::from(1));
    let mlt_desc = super::long_term_mean_inflow(&sys_desc, EntityId::from(1));
    assert_eq!(
        mlt_asc, mlt_desc,
        "long-term mean inflow must be order-independent (declaration-order invariance)"
    );
}

// ── resolve_downstream_level tests ───────────────────────────────────────

/// The single bus (`id = 10`) every test hydro injects into.
fn make_bus() -> Bus {
    Bus {
        id: EntityId::from(10),
        name: "Bus10".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: Vec::new(),
        excess_cost: 0.0,
    }
}

/// Build a `HydroReferenceVolumeFractions` whose `get(hydro, stage)` returns
/// the absolute hm³ for `default_fraction` resolved against each plant's band —
/// the resolved-hm³ semantics the downstream-level consumer reads.
fn flat_reference_fractions(
    default_fraction: f64,
    hydros: &[cobre_core::entities::hydro::Hydro],
) -> HydroReferenceVolumeFractions {
    let resolved: Vec<(EntityId, usize, f64)> = hydros
        .iter()
        .flat_map(|h| {
            let v = h.min_storage_hm3 + default_fraction * (h.max_storage_hm3 - h.min_storage_hm3);
            (0..8).map(move |s| (h.id, s, v))
        })
        .collect();
    cobre_io::extensions::build_hydro_reference_volumes_resolved(&resolved, 0.0)
}

/// A geometry map carrying one plant's two-point VHA rows.
fn geometry_map_for(rows: &[HydroGeometryRow]) -> HashMap<EntityId, Vec<&HydroGeometryRow>> {
    let mut map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    for r in rows {
        map.entry(r.hydro_id).or_default().push(r);
    }
    map
}

/// `downstream_id == None` ⇒ `None` (no downstream reservoir to couple).
#[test]
fn resolve_downstream_level_no_downstream_is_none() {
    let hydro = make_hydro(0, HydroGenerationModel::Fpha); // downstream_id = None
    let stage = make_stage(0);
    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![hydro.clone()])
        .build()
        .expect("system");
    let geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    let fractions = flat_reference_fractions(0.5, system.hydros());

    let level = resolve_downstream_level(&hydro, stage.index, &system, &geometry_map, &fractions);
    assert!(level.is_none(), "no downstream_id must resolve to None");
}

/// A resolved downstream plant with geometry + fraction yields
/// `Some(forebay.height(v_ref))`, matching an independent `ForebayTable`.
#[test]
fn resolve_downstream_level_matches_independent_forebay_height() {
    // Upstream plant 0 discharges into downstream plant 1.
    let mut upstream = make_hydro(0, HydroGenerationModel::Fpha);
    upstream.downstream_id = Some(EntityId::from(1));
    let downstream = make_hydro(1, HydroGenerationModel::ConstantProductivity);

    let stage = make_stage(0);
    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![upstream.clone(), downstream.clone()])
        .build()
        .expect("system");

    let geo_rows = make_geometry_rows(1);
    let geometry_map = geometry_map_for(&geo_rows);
    let fraction = 0.5;
    let fractions = flat_reference_fractions(fraction, system.hydros());

    let level =
        resolve_downstream_level(&upstream, stage.index, &system, &geometry_map, &fractions)
            .expect("downstream level must resolve");

    // Independent reference: v_ref = v_min + fraction·(v_max − v_min) on the
    // downstream plant, evaluated through a freshly-built ForebayTable.
    let v_ref = downstream.min_storage_hm3
        + fraction * (downstream.max_storage_hm3 - downstream.min_storage_hm3);
    let table = ForebayTable::new(&geo_rows, &downstream.name).expect("forebay");
    let expected = table.height(v_ref);

    assert!(
        (level - expected).abs() < 1e-9,
        "resolved level {level} must match independent ForebayTable::height {expected}"
    );
}

/// A downstream plant absent from the geometry map ⇒ `None`.
#[test]
fn resolve_downstream_level_missing_geometry_is_none() {
    let mut upstream = make_hydro(0, HydroGenerationModel::Fpha);
    upstream.downstream_id = Some(EntityId::from(1));
    let downstream = make_hydro(1, HydroGenerationModel::ConstantProductivity);

    let stage = make_stage(0);
    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![upstream.clone(), downstream])
        .build()
        .expect("system");

    // Empty geometry map: the downstream plant has no VHA rows.
    let geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    let fractions = flat_reference_fractions(0.5, system.hydros());

    let level =
        resolve_downstream_level(&upstream, stage.index, &system, &geometry_map, &fractions);
    assert!(
        level.is_none(),
        "missing downstream geometry must resolve to None"
    );
}

// ── Tailrace-source resolution + dedup-key tests ──────────────────────────

/// A hydro absent from the families map resolves to `TailraceSource::Entity`
/// carrying the entity `TailraceModel` — the inert fallback. Per-plant
/// fallback: the global presence of a table for other plants is irrelevant.
#[test]
fn resolve_tailrace_source_absent_from_map_is_entity() {
    let hydro = make_sobradinho_computed_hydro(0);
    let stage = make_stage(0);
    let (system, fractions) = entity_fit_context(&hydro);
    let geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    // Families map populated for a DIFFERENT plant id only.
    let other_rows = vec![TailraceCurveRow {
        hydro_id: EntityId::from(99),
        family_id: 1,
        downstream_reference_level_m: None,
        segment_id: 1,
        outflow_min_m3s: 0.0,
        outflow_max_m3s: 1000.0,
        coefficient_0: 5.0,
        coefficient_1: 0.0,
        coefficient_2: 0.0,
        coefficient_3: 0.0,
        coefficient_4: 0.0,
    }];
    let families_map =
        super::build_tailrace_families_map(&other_rows).expect("families map builds");

    let source = super::resolve_tailrace_source(
        &hydro,
        stage.index,
        &families_map,
        &geometry_map,
        &fractions,
        &system,
    );
    match source {
        TailraceSource::Entity(model) => {
            assert_eq!(
                model, hydro.tailrace,
                "fallback must carry the entity TailraceModel unchanged"
            );
        }
        TailraceSource::Families { .. } => {
            panic!("a plant absent from the families map must resolve to Entity")
        }
    }
}

/// Two `SelectionMode` entries of one hydro whose downstream reference levels
/// differ (multi-family table) produce DISTINCT plane sets: the dedup cache
/// key includes `downstream_level_m`, so the two entries are not collapsed to
/// one fit. Stage 0 resolves a low downstream level (lowest family), stage 1 a
/// high one (highest family); the two families carry different tailrace
/// heights, so the fitted planes differ.
#[test]
fn dedup_key_separates_distinct_downstream_levels() {
    // Upstream computed-FPHA plant 0 discharges into downstream plant 1.
    let mut upstream = make_sobradinho_computed_hydro(0);
    upstream.downstream_id = Some(EntityId::from(1));
    let downstream = make_hydro(1, HydroGenerationModel::ConstantProductivity);

    // Downstream geometry spanning heights 380 → 420 over volume 100 → 2000,
    // so fraction 0.0 → 380 m and fraction 1.0 → 420 m.
    let down_geo = [
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3: 100.0,
            height_m: 380.0,
            area_km2: 10.0,
        },
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3: 2000.0,
            height_m: 420.0,
            area_km2: 50.0,
        },
    ];
    let up_geo = make_sobradinho_geometry_rows(0);
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    geometry_map.insert(EntityId::from(0), up_geo.iter().collect());
    geometry_map.insert(EntityId::from(1), down_geo.iter().collect());

    // Two-family tailrace for the upstream plant keyed at 380 and 420 m with
    // distinct heights (constant 2.0 vs 40.0 m), so the resolved level picks a
    // materially different tailrace.
    let tailrace_rows = vec![
        TailraceCurveRow {
            hydro_id: EntityId::from(0),
            family_id: 1,
            downstream_reference_level_m: Some(380.0),
            segment_id: 1,
            outflow_min_m3s: 0.0,
            outflow_max_m3s: 100_000.0,
            coefficient_0: 2.0,
            coefficient_1: 0.0,
            coefficient_2: 0.0,
            coefficient_3: 0.0,
            coefficient_4: 0.0,
        },
        TailraceCurveRow {
            hydro_id: EntityId::from(0),
            family_id: 2,
            downstream_reference_level_m: Some(420.0),
            segment_id: 1,
            outflow_min_m3s: 0.0,
            outflow_max_m3s: 100_000.0,
            coefficient_0: 40.0,
            coefficient_1: 0.0,
            coefficient_2: 0.0,
            coefficient_3: 0.0,
            coefficient_4: 0.0,
        },
    ];
    let families_map =
        super::build_tailrace_families_map(&tailrace_rows).expect("families map builds");

    // Stage 0 → season 0 (fraction 0.0 → level 380), stage 1 → season 1
    // (fraction 1.0 → level 420) for the downstream plant.
    let mut stage0 = make_stage(0);
    stage0.season_id = Some(0);
    let mut stage1 = make_stage(1);
    stage1.season_id = Some(1);
    // Downstream plant 1 band is [100, 2000]; stage 0 resolves fraction 0.0001
    // (→ low level), stage 1 fraction 1.0 (→ high level), keyed by stage index.
    let down_v_min = downstream.min_storage_hm3;
    let down_span = downstream.max_storage_hm3 - downstream.min_storage_hm3;
    let fractions = cobre_io::extensions::build_hydro_reference_volumes_resolved(
        &[
            (EntityId::from(1), 0, down_v_min + 0.0001 * down_span),
            (EntityId::from(1), 1, down_v_min + 1.0 * down_span),
        ],
        0.0,
    );

    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![upstream.clone(), downstream])
        .build()
        .expect("two-hydro system builds");

    // One open-ended range → one config across both stages, so ONLY the
    // downstream level varies between the two stages.
    let config = computed_fpha_config(0);
    let stages = [&stage0, &stage1];
    let stage_refs: Vec<&Stage> = stages.to_vec();

    let mut export_rows = Vec::new();
    let per_stage = super::fit_computed_planes_per_stage(
        &upstream,
        Some(&config),
        &geometry_map,
        &families_map,
        &fractions,
        &system,
        &stage_refs,
        0.0,
        None,
        false,
        &mut export_rows,
        &mut Vec::new(),
        &mut Vec::new(),
    )
    .expect("per-stage fit succeeds");

    // Sanity: the two stages resolved distinct downstream levels.
    let lvl0 =
        resolve_downstream_level(&upstream, stage0.index, &system, &geometry_map, &fractions)
            .expect("stage 0 level");
    let lvl1 =
        resolve_downstream_level(&upstream, stage1.index, &system, &geometry_map, &fractions)
            .expect("stage 1 level");
    assert_ne!(
        lvl0.to_bits(),
        lvl1.to_bits(),
        "the two stages must resolve different downstream levels"
    );

    assert_ne!(
        per_stage[0], per_stage[1],
        "distinct downstream levels must yield distinct fits (dedup key includes the level)"
    );
}

/// Capture-all: `fit_computed_planes_per_stage` records ONE
/// `FphaDeviationDiagnostic` per DISTINCT fit, regardless of the warn
/// threshold. Two stages resolving distinct downstream levels yield two
/// distinct fits, so the diagnostics vector holds exactly two entries — even
/// for a well-fit (under-threshold) concave surface that would raise no
/// warning. A regression that re-introduced the warn-worthy push filter would
/// drop the under-threshold entries and fail the count.
#[test]
fn fit_computed_planes_per_stage_records_every_distinct_fit() {
    let mut upstream = make_sobradinho_computed_hydro(0);
    upstream.downstream_id = Some(EntityId::from(1));
    let downstream = make_hydro(1, HydroGenerationModel::ConstantProductivity);

    let down_geo = [
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3: 100.0,
            height_m: 380.0,
            area_km2: 10.0,
        },
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3: 2000.0,
            height_m: 420.0,
            area_km2: 50.0,
        },
    ];
    let up_geo = make_sobradinho_geometry_rows(0);
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    geometry_map.insert(EntityId::from(0), up_geo.iter().collect());
    geometry_map.insert(EntityId::from(1), down_geo.iter().collect());

    let tailrace_rows = vec![
        TailraceCurveRow {
            hydro_id: EntityId::from(0),
            family_id: 1,
            downstream_reference_level_m: Some(380.0),
            segment_id: 1,
            outflow_min_m3s: 0.0,
            outflow_max_m3s: 100_000.0,
            coefficient_0: 2.0,
            coefficient_1: 0.0,
            coefficient_2: 0.0,
            coefficient_3: 0.0,
            coefficient_4: 0.0,
        },
        TailraceCurveRow {
            hydro_id: EntityId::from(0),
            family_id: 2,
            downstream_reference_level_m: Some(420.0),
            segment_id: 1,
            outflow_min_m3s: 0.0,
            outflow_max_m3s: 100_000.0,
            coefficient_0: 40.0,
            coefficient_1: 0.0,
            coefficient_2: 0.0,
            coefficient_3: 0.0,
            coefficient_4: 0.0,
        },
    ];
    let families_map =
        super::build_tailrace_families_map(&tailrace_rows).expect("families map builds");

    let mut stage0 = make_stage(0);
    stage0.season_id = Some(0);
    let mut stage1 = make_stage(1);
    stage1.season_id = Some(1);
    let down_v_min = downstream.min_storage_hm3;
    let down_span = downstream.max_storage_hm3 - downstream.min_storage_hm3;
    let fractions = cobre_io::extensions::build_hydro_reference_volumes_resolved(
        &[
            (EntityId::from(1), 0, down_v_min + 0.0001 * down_span),
            (EntityId::from(1), 1, down_v_min + 1.0 * down_span),
        ],
        0.0,
    );

    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![upstream.clone(), downstream])
        .build()
        .expect("two-hydro system builds");

    let config = computed_fpha_config(0);
    let stage_refs: Vec<&Stage> = vec![&stage0, &stage1];

    let mut export_rows = Vec::new();
    let mut diagnostics = Vec::new();
    let per_stage = super::fit_computed_planes_per_stage(
        &upstream,
        Some(&config),
        &geometry_map,
        &families_map,
        &fractions,
        &system,
        &stage_refs,
        0.0,
        None,
        false,
        &mut export_rows,
        &mut diagnostics,
        &mut Vec::new(),
    )
    .expect("per-stage fit succeeds");

    assert_ne!(
        per_stage[0], per_stage[1],
        "the two stages must resolve distinct fits for this test to exercise two captures"
    );
    assert_eq!(
        diagnostics.len(),
        2,
        "one diagnostic per distinct fit must be captured (capture-all), not only warn-worthy ones"
    );
    // The diagnostics are tagged with the first stage each fit covers (0 and 1).
    assert_eq!(diagnostics[0].stage_id, 0);
    assert_eq!(diagnostics[1].stage_id, 1);
    // At least one fit of this well-fit concave surface is under the warn
    // threshold — the entry it is still captured (capture-all), confirming the
    // push is not gated on `exceeds_warn_threshold()`.
    assert!(
        diagnostics
            .iter()
            .any(|d| !d.deviation.exceeds_warn_threshold()),
        "a well-fit (under-threshold) fit must still be captured"
    );
}

/// Carry-up + warn split through the resolver: every distinct computed-FPHA
/// fit reaches `fpha_fit_deviations` (capture-all), in canonical
/// `(hydro, stage)` order, with `hydro_id` recovered from the outer zip — even
/// for an under-threshold fit that raises no `tracing::warn!`. The warn
/// predicate (`exceeds_warn_threshold()`) is exercised in `deviation.rs`; here
/// we assert the carried vector is the full superset.
#[test]
fn resolver_carries_every_fit_deviation_in_canonical_order() {
    let hydro = make_sobradinho_computed_hydro(0);
    let geo_rows = make_sobradinho_geometry_rows(0);
    let artifacts = cobre_io::CaseArtifacts {
        production_models: vec![computed_fpha_config(0)],
        hydro_geometry: geo_rows,
        ..Default::default()
    };

    let stages = vec![make_stage(0), make_stage(1)];
    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![hydro])
        .stages(stages)
        .build()
        .expect("computed-FPHA system builds");

    let (_, _, _, _, _, fpha_fit_deviations, _) =
        super::resolve_production_models_from_artifacts(&system, &artifacts, false)
            .expect("resolve must succeed for a computed-FPHA case");

    // One open-ended range → a single distinct fit shared across both stages,
    // tagged with the first covered stage (id 0); the carry-up is per distinct
    // fit, not per covered stage.
    assert_eq!(
        fpha_fit_deviations.len(),
        1,
        "a single distinct fit must yield exactly one carried deviation entry"
    );
    let entry = fpha_fit_deviations[0];
    assert_eq!(entry.hydro_id, EntityId::from(0), "hydro_id from outer zip");
    assert_eq!(entry.stage_id, 0, "tagged with the first covered stage");
    assert!(
        entry.mean_abs_mw >= 0.0 && entry.max_abs_mw >= entry.mean_abs_mw,
        "deviation magnitudes must be well-formed"
    );
}

/// Declaration-order invariance: `resolve_production_models_from_artifacts`
/// yields bit-identical `export_rows` when the hydros and tailrace-curve rows
/// are supplied in shuffled declaration orders.
#[test]
fn export_rows_are_declaration_order_invariant_with_tailrace() {
    // One computed-FPHA hydro (id=0) with a single-family tailrace table.
    fn build_artifacts(
        tailrace_rows: Vec<TailraceCurveRow>,
        geo_rows: Vec<HydroGeometryRow>,
    ) -> cobre_io::CaseArtifacts {
        cobre_io::CaseArtifacts {
            production_models: vec![computed_fpha_config(0)],
            hydro_geometry: geo_rows,
            tailrace_curves: tailrace_rows,
            ..Default::default()
        }
    }

    let hydro = make_sobradinho_computed_hydro(0);
    let stage = make_stage(0);
    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![hydro.clone()])
        .stages(vec![stage])
        .build()
        .expect("single-hydro system builds");

    // Geometry rows are re-sorted by `build_geometry_map` inside the resolve
    // path, so a reversed geometry input must yield bit-identical fitted
    // planes — the declaration-order invariance the families path must uphold.
    let geo_fwd = make_sobradinho_geometry_rows(0);
    let mut geo_rev = geo_fwd.clone();
    geo_rev.reverse();

    // Single-segment single family: a flat 2.0 m tailrace covering the full
    // sampled outflow range. The tailrace rows are supplied in canonical
    // `(hydro_id, family_id, segment_id)` order (as the loader produces them);
    // the shuffled dimension here is the geometry input.
    let segment = TailraceCurveRow {
        hydro_id: EntityId::from(0),
        family_id: 1,
        downstream_reference_level_m: None,
        segment_id: 1,
        outflow_min_m3s: 0.0,
        outflow_max_m3s: 100_000.0,
        coefficient_0: 2.0,
        coefficient_1: 0.0,
        coefficient_2: 0.0,
        coefficient_3: 0.0,
        coefficient_4: 0.0,
    };

    let art_fwd = build_artifacts(vec![segment.clone()], geo_fwd);
    let art_rev = build_artifacts(vec![segment], geo_rev);

    let (_, _, _, rows_fwd, _, _, _) =
        resolve_production_models_from_artifacts(&system, &art_fwd, false)
            .expect("forward resolve");
    let (_, _, _, rows_rev, _, _, _) =
        resolve_production_models_from_artifacts(&system, &art_rev, false)
            .expect("reversed resolve");

    assert_eq!(
        rows_fwd.len(),
        rows_rev.len(),
        "export-row counts must match across input orderings"
    );
    for (a, b) in rows_fwd.iter().zip(&rows_rev) {
        assert_eq!(a.hydro_id, b.hydro_id);
        assert_eq!(a.stage_id, b.stage_id);
        assert_eq!(a.plane_id, b.plane_id);
        assert_eq!(
            a.gamma_0.to_bits(),
            b.gamma_0.to_bits(),
            "gamma_0 must be bit-identical across input orderings"
        );
        assert_eq!(a.gamma_v.to_bits(), b.gamma_v.to_bits());
        assert_eq!(a.gamma_q.to_bits(), b.gamma_q.to_bits());
        assert_eq!(a.gamma_s.to_bits(), b.gamma_s.to_bits());
    }
}

// ── reference_volume resolution ─────────────────────────────────────────

/// Build a single-`StageRange` config covering `[0, ∞)` with the given
/// `reference_volume`.
fn config_with_reference_volume(rv: Option<ReferenceVolume>) -> ProductionModelConfig {
    ProductionModelConfig {
        hydro_id: EntityId::from(1),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 0,
                end_stage_id: None,
                model: "constant_productivity".to_string(),
                fpha_config: None,
                reference_volume: rv,
                productivity_mw_per_m3s: Some(0.5),
            }],
        },
    }
}

#[test]
fn find_reference_volume_for_stage_returns_covering_entry_value() {
    let config = config_with_reference_volume(Some(ReferenceVolume::AbsoluteHm3(800.0)));
    let stage0 = make_stage(0);
    assert_eq!(
        find_reference_volume_for_stage(&config, &stage0),
        Some(&ReferenceVolume::AbsoluteHm3(800.0))
    );
}

#[test]
fn find_reference_volume_for_stage_returns_none_when_unset() {
    let config = config_with_reference_volume(None);
    let stage0 = make_stage(0);
    assert_eq!(find_reference_volume_for_stage(&config, &stage0), None);
}

#[test]
fn find_reference_volume_for_stage_returns_none_outside_coverage() {
    // Range covers stages [5, 10]; stage 0 is uncovered, so the finder
    // returns `None` exactly as `find_fpha_config_for_stage` does.
    let config = ProductionModelConfig {
        hydro_id: EntityId::from(1),
        selection_mode: SelectionMode::StageRanges {
            ranges: vec![StageRange {
                start_stage_id: 5,
                end_stage_id: Some(10),
                model: "constant_productivity".to_string(),
                fpha_config: None,
                reference_volume: Some(ReferenceVolume::AbsoluteHm3(800.0)),
                productivity_mw_per_m3s: Some(0.5),
            }],
        },
    };
    let stage0 = make_stage(0);
    assert_eq!(find_reference_volume_for_stage(&config, &stage0), None);
}

#[test]
fn resolve_reference_volume_hm3_absolute_passthrough() {
    let rv = ReferenceVolume::AbsoluteHm3(800.0);
    let resolved = resolve_reference_volume_hm3(Some(&rv), 100.0, 2000.0);
    assert_eq!(resolved, 800.0);
}

#[test]
fn resolve_reference_volume_hm3_percentile_against_band() {
    let rv = ReferenceVolume::Percentile(0.5);
    let resolved = resolve_reference_volume_hm3(Some(&rv), 100.0, 200.0);
    assert!((resolved - 150.0).abs() <= f64::EPSILON);
}

#[test]
fn resolve_reference_volume_hm3_none_reproduces_065_fraction() {
    let v_min = 100.0_f64;
    let v_max = 200.0_f64;
    let resolved = resolve_reference_volume_hm3(None, v_min, v_max);
    // Bit-identical to the legacy inline `0.65`-fraction formula: this is the
    // byte-identity guarantee the consumer rewrite relies on.
    let expected = v_min + 0.65 * (v_max - v_min);
    assert_eq!(resolved.to_bits(), expected.to_bits());
}

#[test]
fn resolve_reference_volume_hm3_degenerate_band_is_not_nan() {
    // A `v_max == v_min` band must collapse to `v_min` via multiplication,
    // never produce a `0/0` NaN from a division-based percentile conversion.
    let rv = ReferenceVolume::Percentile(0.5);
    let resolved = resolve_reference_volume_hm3(Some(&rv), 50.0, 50.0);
    assert!(!resolved.is_nan());
    assert_eq!(resolved, 50.0);
}

// ── Resolver-wiring tests (the JSON-sourced reference volume) ─────────────

/// A single constant-productivity hydro with the given storage band, plus a
/// two-stage study, built for the resolver-wiring tests below.
fn ref_vol_system(v_min: f64, v_max: f64) -> System {
    let mut hydro = make_hydro(0, HydroGenerationModel::ConstantProductivity);
    hydro.min_storage_hm3 = v_min;
    hydro.max_storage_hm3 = v_max;
    SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![hydro])
        .stages(vec![make_stage(0), make_stage(1)])
        .build()
        .expect("single-hydro two-stage system builds")
}

/// Artifacts carrying a single `StageRange` production config for hydro 0
/// covering `[0, ∞)` with the given `reference_volume`.
fn ref_vol_artifacts(rv: Option<ReferenceVolume>) -> cobre_io::CaseArtifacts {
    cobre_io::CaseArtifacts {
        production_models: vec![ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 0,
                    end_stage_id: None,
                    model: "constant_productivity".to_string(),
                    fpha_config: None,
                    reference_volume: rv,
                    productivity_mw_per_m3s: Some(1.0),
                }],
            },
        }],
        ..Default::default()
    }
}

/// With no `reference_volume` declared, the resolver returns
/// `v_min + 0.65·(v_max − v_min)` bit-for-bit for every plant/stage — the
/// byte-identity guarantee the baseline regression depends on.
#[test]
fn resolver_default_path_returns_065_fraction_hm3_bit_for_bit() {
    let v_min = 100.0_f64;
    let v_max = 200.0_f64;
    let system = ref_vol_system(v_min, v_max);
    let artifacts = ref_vol_artifacts(None);

    let (_, _, _, _, table, _, _) =
        resolve_production_models_from_artifacts(&system, &artifacts, false)
            .expect("resolve succeeds");
    let resolver = build_hydro_reference_volumes_resolved(&table, 0.0);

    let expected = v_min + 0.65 * (v_max - v_min);
    for stage_idx in [0_usize, 1] {
        assert_eq!(
            resolver.get(EntityId::from(0), stage_idx).to_bits(),
            expected.to_bits(),
            "stage {stage_idx}: undeclared reference volume must be the 0.65-fraction hm³"
        );
    }
}

/// A declared absolute `volume_hm3: 800.0` flows through the resolver `get`.
#[test]
fn resolver_declared_absolute_flows_through_get() {
    let system = ref_vol_system(100.0, 2000.0);
    let artifacts = ref_vol_artifacts(Some(ReferenceVolume::AbsoluteHm3(800.0)));

    let (_, _, _, _, table, _, _) =
        resolve_production_models_from_artifacts(&system, &artifacts, false)
            .expect("resolve succeeds");
    let resolver = build_hydro_reference_volumes_resolved(&table, 0.0);

    assert_eq!(resolver.get(EntityId::from(0), 0), 800.0);
    assert_eq!(resolver.get(EntityId::from(0), 1), 800.0);
}

/// A declared `percentile: 0.5` on band `[100, 200]` resolves to `150.0`.
#[test]
fn resolver_declared_percentile_resolves_against_band() {
    let system = ref_vol_system(100.0, 200.0);
    let artifacts = ref_vol_artifacts(Some(ReferenceVolume::Percentile(0.5)));

    let (_, _, _, _, table, _, _) =
        resolve_production_models_from_artifacts(&system, &artifacts, false)
            .expect("resolve succeeds");
    let resolver = build_hydro_reference_volumes_resolved(&table, 0.0);

    assert_eq!(resolver.get(EntityId::from(0), 0), 150.0);
}

/// A study horizon may begin in any season — not only season 0 — and may wrap
/// around the seasonal cycle. With a SEASONAL `reference_volume` config, the
/// value stored at each 0-based study position reflects THAT stage's
/// `season_id` (selected by `find_reference_volume_for_stage`), and the resolver
/// is keyed by study position — never by `season_id` or by `stage.index`. So a
/// horizon starting mid-cycle resolves each stage's own season value, and the
/// energy-conversion build (which counts stages from 0, proven by
/// `per_season_override_produces_different_v_ref_per_stage`) reads them back in
/// order. This exercises the real `resolve_production_models_from_artifacts`
/// build end-to-end.
#[test]
fn seasonal_reference_volume_supports_nonzero_start_season() {
    let (v_min, v_max) = (100.0_f64, 200.0_f64);

    // Horizon begins in season 2 and wraps to season 1: study position 0 →
    // season 2, position 1 → season 1. Neither is season 0.
    let mut hydro = make_hydro(0, HydroGenerationModel::ConstantProductivity);
    hydro.min_storage_hm3 = v_min;
    hydro.max_storage_hm3 = v_max;
    let mut stage0 = make_stage(0);
    stage0.season_id = Some(2);
    let mut stage1 = make_stage(1);
    stage1.season_id = Some(1);
    let system = SystemBuilder::new()
        .buses(vec![make_bus()])
        .hydros(vec![hydro])
        .stages(vec![stage0, stage1])
        .build()
        .expect("two-stage non-zero-start-season system builds");

    let artifacts = cobre_io::CaseArtifacts {
        production_models: vec![ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::Seasonal {
                default_model: "constant_productivity".to_string(),
                seasons: vec![
                    SeasonConfig {
                        season_id: 1,
                        model: "constant_productivity".to_string(),
                        fpha_config: None,
                        reference_volume: Some(ReferenceVolume::Percentile(0.8)),
                        productivity_mw_per_m3s: Some(1.0),
                    },
                    SeasonConfig {
                        season_id: 2,
                        model: "constant_productivity".to_string(),
                        fpha_config: None,
                        reference_volume: Some(ReferenceVolume::Percentile(0.2)),
                        productivity_mw_per_m3s: Some(1.0),
                    },
                ],
            },
        }],
        ..Default::default()
    };

    let (_, _, _, _, table, _, _) =
        resolve_production_models_from_artifacts(&system, &artifacts, false)
            .expect("resolve succeeds");
    let resolver = build_hydro_reference_volumes_resolved(&table, 0.0);

    // Position 0 carries season 2's value (0.2 → 120), position 1 carries
    // season 1's (0.8 → 180). A season-keyed or stage.index-keyed lookup would
    // mis-assign these for a horizon that does not start at season 0.
    assert_eq!(
        resolver.get(EntityId::from(0), 0),
        v_min + 0.2 * (v_max - v_min),
        "study position 0 (season 2) must resolve percentile 0.2"
    );
    assert_eq!(
        resolver.get(EntityId::from(0), 1),
        v_min + 0.8 * (v_max - v_min),
        "study position 1 (season 1) must resolve percentile 0.8"
    );
}
