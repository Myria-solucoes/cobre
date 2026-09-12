//! Constants below were captured on the pre-change code; a diff in them means the
//! forward noise stream moved.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use std::collections::BTreeMap;

use chrono::NaiveDate;
use cobre_core::{
    Bus, DeficitSegment, EntityId, SystemBuilder,
    entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
        SamplingScheme,
    },
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_stochastic::{
    ClassDimensions,
    context::{ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context},
    correlation::resolve::DecomposedCorrelation,
    generate_opening_tree,
    sampling::{
        ForwardNoiseTables, ForwardSampler, ForwardSamplerConfig, SampleRequest,
        build_forward_sampler,
    },
    tree::generate::OpeningTreeGenerationInputs,
};

const GOLDEN_SEQUENCE: [f64; 48] = [
    -0.080_725_442_736_187_18,
    0.342_169_703_875_618_6,
    1.849_818_605_734_19,
    -1.108_196_839_820_201_8,
    -0.779_542_349_566_169_5,
    -0.295_284_160_939_788_5,
    0.576_407_613_977_412_9,
    1.194_651_590_486_514,
    -0.652_782_348_056_772_2,
    0.004_955_670_217_361_959,
    0.696_519_746_226_396_6,
    -2.881_805_549_003_158_4,
    -2.459_925_019_377_221_3,
    -0.668_281_316_820_787_3,
    0.017_417_691_321_618_904,
    0.680_724_290_146_252_7,
    -8.21,
    -8.21,
    -8.21,
    -0.430_727_299_975_868_17,
    0.0,
    0.430_727_299_975_868_06,
    -8.21,
    -0.764_709_674_990_941_9,
    -8.21,
    -8.21,
    0.0,
    -0.430_727_299_975_868_17,
    -8.21,
    0.430_727_299_975_868_06,
    0.0,
    -1.220_640_349_639_383,
    -1.072_385_873_667_138,
    -0.157_124_096_842_055_58,
    0.087_711_240_490_644_9,
    0.770_845_438_329_447_1,
    1.006_628_194_622_461_3,
    -1.303_805_206_386_656_5,
    -0.167_933_641_235_421_45,
    0.542_772_246_963_319_6,
    0.157_086_514_183_187_86,
    -1.949_857_445_865_825_8,
    -0.090_633_019_333_538_7,
    -0.331_743_552_456_976_84,
    -0.879_316_021_632_609_9,
    0.877_941_307_496_606_9,
    0.891_338_480_484_778_2,
    0.337_851_871_717_413_1,
];

const GOLDEN_WIDE_S0_O0: [f64; 70] = [
    0.261_943_978_666_802_36,
    0.127_889_353_586_245_6,
    -0.397_279_971_931_077_9,
    0.260_598_250_741_210_85,
    0.105_779_504_081_037_67,
    0.893_746_933_495_428_1,
    -1.343_253_373_302_420_8,
    0.092_521_056_368_748_44,
    -1.238_486_988_715_376_2,
    0.554_255_718_755_035,
    0.653_610_230_622_547_7,
    -1.041_250_121_454_886_6,
    0.386_666_049_609_057_3,
    1.133_666_403_097_759_2,
    -0.226_238_634_032_527_7,
    0.225_323_484_452_808_36,
    1.261_331_954_543_671_9,
    -0.450_374_834_499_717_1,
    1.230_461_538_878_261_4,
    0.292_739_676_474_220_17,
    -0.420_218_021_805_944_1,
    -1.654_856_993_187_656,
    1.504_837_998_773_783_4,
    -0.002_252_242_451_901_694_7,
    0.229_162_099_307_601_4,
    -0.104_673_117_545_754_37,
    -1.503_430_227_024_682_9,
    -0.005_976_751_182_632_97,
    -0.807_785_330_003_612_1,
    -0.729_196_429_459_687_3,
    -1.478_225_681_275_69,
    -1.385_178_214_711_543,
    -0.019_444_656_016_684_07,
    -0.728_736_399_628_431_7,
    -0.366_729_314_351_177,
    1.055_206_087_526_587_4,
    0.381_122_526_742_886_77,
    0.031_861_444_545_165_92,
    -1.458_715_231_306_354_5,
    0.170_958_400_988_427_68,
    -0.168_848_052_028_692_9,
    0.061_222_412_313_849_78,
    -1.094_028_115_684_667_8,
    -0.164_298_889_510_819_17,
    0.649_241_502_060_317_3,
    0.812_822_186_738_101_7,
    0.190_802_024_947_911_87,
    0.550_820_451_612_900_2,
    -0.276_802_046_353_518_57,
    -1.085_787_901_578_905_1,
    0.479_195_526_656_396_8,
    1.035_696_150_711_604_3,
    0.734_861_037_432_464_4,
    -0.171_897_587_695_379_36,
    -0.768_803_273_593_813_2,
    0.373_641_904_424_883_7,
    -0.050_268_087_882_327_825,
    0.071_530_581_330_186_04,
    -0.754_801_503_344_352_4,
    -0.402_241_416_296_409_8,
    -0.498_269_241_895_616_55,
    1.035_545_800_847_415,
    -0.003_801_806_202_421_025_7,
    -0.242_196_127_856_020_15,
    -0.560_809_407_557_846,
    -0.469_451_756_654_139_5,
    0.169_197_703_157_331_49,
    0.341_386_821_571_796_9,
    1.005_807_869_934_130_6,
    -0.274_053_317_089_533_95,
];
const GOLDEN_WIDE_S0_O2_D0: f64 = 0.441_985_571_180_052_8;
const GOLDEN_WIDE_S0_O2_D69: f64 = -0.110_289_738_555_546_02;
const GOLDEN_WIDE_S1_O1_D0: f64 = 0.311_586_211_850_554_1;
const GOLDEN_WIDE_S1_O1_D69: f64 = 0.653_769_318_292_650_7;

fn make_sampler_config<'a>(
    scheme: SamplingScheme,
    ctx: &'a StochasticContext,
    stages: &'a [Stage],
) -> ForwardSamplerConfig<'a> {
    let dim = ctx.dim();
    ForwardSamplerConfig {
        class_schemes: ClassSchemes {
            inflow: Some(scheme),
            load: Some(scheme),
            ncs: Some(scheme),
        },
        ctx,
        stages,
        dims: ClassDimensions {
            n_hydros: dim,
            n_load_buses: 0,
            n_ncs: 0,
        },
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
    }
}

fn make_bus(id: i32) -> Bus {
    Bus {
        id: EntityId(id),
        name: format!("Bus{id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    }
}

fn make_hydro(id: i32) -> Hydro {
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(id),
        name: format!("H{id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 100.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
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
    hydro.declare_mirror_unit_group(EntityId(0));
    hydro
}

fn make_stage(index: usize, id: i32, bf: usize, method: NoiseMethod) -> Stage {
    Stage {
        index,
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
            branching_factor: bf,
            noise_method: method,
        },
    }
}

fn make_inflow_model(hydro_id: i32, stage_id: i32) -> InflowModel {
    InflowModel {
        hydro_id: EntityId(hydro_id),
        stage_id,
        mean_m3s: 100.0,
        std_m3s: 30.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    }
}

fn identity_correlation(ids: &[i32]) -> CorrelationModel {
    correlated_correlation(ids, 0.0)
}

fn correlated_correlation(ids: &[i32], rho: f64) -> CorrelationModel {
    let n = ids.len();
    let matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
        .collect();
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: ids
                    .iter()
                    .map(|&id| CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(id),
                    })
                    .collect(),
                matrix,
            }],
        },
    );
    CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    }
}

fn build_test_system(methods: &[NoiseMethod], correlation: CorrelationModel) -> cobre_core::System {
    assert_eq!(methods.len(), 3, "must supply exactly 3 per-stage methods");
    let hydros = vec![make_hydro(1), make_hydro(2)];
    let stages = vec![
        make_stage(0, 0, 5, methods[0]),
        make_stage(1, 1, 5, methods[1]),
        make_stage(2, 2, 5, methods[2]),
    ];
    let inflow_models = vec![
        make_inflow_model(1, 0),
        make_inflow_model(1, 1),
        make_inflow_model(1, 2),
        make_inflow_model(2, 0),
        make_inflow_model(2, 1),
        make_inflow_model(2, 2),
    ];
    SystemBuilder::new()
        .buses(vec![make_bus(0)])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .build()
        .unwrap()
}

fn build_test_ctx(system: &cobre_core::System, forward_seed: Option<u64>) -> StochasticContext {
    build_stochastic_context(
        system,
        42,
        forward_seed,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .unwrap()
}

fn stages_from_system(system: &cobre_core::System) -> Vec<Stage> {
    system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .cloned()
        .collect()
}

/// Rebuild `sampler`'s noise tables for one `(iteration, total, groups)`
/// triple — the `SampleRequest.tables` every `sample()` call in this file needs.
fn tables_for(
    sampler: &ForwardSampler<'_>,
    iteration: u32,
    total: u32,
    groups: &[u32],
) -> ForwardNoiseTables {
    let mut tables = ForwardNoiseTables::default();
    sampler
        .rebuild_noise_tables(iteration, total, groups, &mut tables)
        .expect("test fixtures never exceed the Sobol dimension cap");
    tables
}

#[test]
fn out_of_sample_point_method_golden_sequence() {
    let system = build_test_system(
        &[
            NoiseMethod::QmcSobol,
            NoiseMethod::QmcHalton,
            NoiseMethod::Lhs,
        ],
        identity_correlation(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, Some(99));
    let stages = stages_from_system(&system);
    let sampler = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ))
    .unwrap();
    let dim = ctx.dim();
    let total_scenarios: u32 = 4;

    let mut noise_buf = vec![0.0f64; dim];
    let mut perm_scratch = vec![0usize; total_scenarios as usize];

    let mut idx = 0usize;
    for stage_idx in 0..3_usize {
        for noise_group_id in 0..2_u32 {
            // This sweep probes the same stage_idx under two synthetic
            // noise_group_id values — a table is keyed by stage_idx alone, so
            // it must be rebuilt per noise_group_id here to stay bit-identical
            // to the golden's pre-table per-draw derivation.
            let mut groups = vec![0u32; 3];
            groups[stage_idx] = noise_group_id;
            let tables = tables_for(&sampler, 0, total_scenarios, &groups);
            for scenario in 0..total_scenarios {
                let result = sampler
                    .sample(SampleRequest {
                        iteration: 0,
                        scenario,
                        stage: u32::try_from(stage_idx).unwrap(),
                        stage_idx,
                        noise_buf: &mut noise_buf,
                        perm_scratch: &mut perm_scratch,
                        total_scenarios,
                        noise_group_id,
                        node_opening_offset: 0,
                        node_opening_len: 0,
                        pinned_scenario: None,
                        tables: &tables,
                    })
                    .unwrap();

                let slice = result.as_slice();
                for (d, &v) in slice.iter().enumerate() {
                    assert_eq!(
                        v, GOLDEN_SEQUENCE[idx],
                        "stage_idx={stage_idx} noise_group_id={noise_group_id} \
                         scenario={scenario} dim={d}"
                    );
                    idx += 1;
                }
            }
        }
    }
}

#[test]
fn wide_correlation_group_opening_tree_golden() {
    let entity_ids: Vec<i32> = (1..=70).collect();
    let entity_order: Vec<EntityId> = entity_ids.iter().copied().map(EntityId).collect();
    let corr = DecomposedCorrelation::build(&correlated_correlation(&entity_ids, 0.4)).unwrap();

    let stages = vec![
        make_stage(0, 0, 3, NoiseMethod::Saa),
        make_stage(1, 1, 3, NoiseMethod::Saa),
    ];

    let dims = ClassDimensions {
        n_hydros: 70,
        n_load_buses: 0,
        n_ncs: 0,
    };
    let tree = generate_opening_tree(
        42,
        &stages,
        70,
        &corr,
        &entity_order,
        dims,
        &OpeningTreeGenerationInputs::default(),
    )
    .unwrap();

    let s0_o0 = tree.opening(0, 0);
    for (d, &v) in s0_o0.iter().enumerate() {
        assert_eq!(v, GOLDEN_WIDE_S0_O0[d], "stage=0 opening=0 dim={d}");
    }

    assert_eq!(
        tree.opening(0, 2)[0],
        GOLDEN_WIDE_S0_O2_D0,
        "stage=0 opening=2 dim=0"
    );
    assert_eq!(
        tree.opening(0, 2)[69],
        GOLDEN_WIDE_S0_O2_D69,
        "stage=0 opening=2 dim=69"
    );
    assert_eq!(
        tree.opening(1, 1)[0],
        GOLDEN_WIDE_S1_O1_D0,
        "stage=1 opening=1 dim=0"
    );
    assert_eq!(
        tree.opening(1, 1)[69],
        GOLDEN_WIDE_S1_O1_D69,
        "stage=1 opening=1 dim=69"
    );
}
