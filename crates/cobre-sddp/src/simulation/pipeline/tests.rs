#![allow(clippy::unwrap_used, clippy::panic, clippy::too_many_lines)]

use std::collections::HashMap;
use std::sync::mpsc;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::scenario::SamplingScheme;
use cobre_solver::{
    Basis, LpSolution, ProfiledSolver, RowBatch, SolverError, SolverInterface, SolverStatistics,
    StageTemplate,
};
use cobre_stochastic::StochasticContext;

use super::SimulationOutputSpec;
use crate::{
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    horizon_mode::HorizonMode,
    inflow_method::InflowNonNegativityMethod,
    lp_builder::PatchBuffer,
    simulation::{
        config::SimulationConfig,
        error::SimulationError,
        extraction::EntityCounts,
        state::{SimulationInputs, SimulationState},
    },
    workspace::{BackwardAccumulators, CapturedBasis, ScratchBuffers, SolverWorkspace},
};

/// Thin test helper that mirrors the old `simulate` free-function signature
/// and delegates to `SimulationState::run`. Allows all 24 existing test call
/// sites to stay unchanged apart from the function name.
#[allow(clippy::too_many_arguments)]
fn run_simulate<S, C: cobre_comm::Communicator>(
    workspaces: &mut [SolverWorkspace<S>],
    ctx: &StageContext<'_>,
    fcf: &crate::FutureCostFunction,
    training_ctx: &TrainingContext<'_>,
    config: &SimulationConfig,
    output: SimulationOutputSpec<'_>,
    frozen_templates: Option<&[cobre_solver::StageTemplate]>,
    stage_bases: &[Option<CapturedBasis>],
    comm: &C,
) -> Result<super::SimulationRunResult, SimulationError>
where
    S: cobre_solver::SolverInterface<Profile = cobre_solver::ActiveProfile> + Send,
{
    let num_stages = training_ctx.horizon.num_stages();
    let mut state = SimulationState::new(num_stages);
    let mut inputs = SimulationInputs {
        workspaces,
        ctx,
        fcf,
        training_ctx,
        config,
        output,
        frozen_templates,
        stage_bases,
        comm,
    };
    state.run(&mut inputs)
}

// ── Stub communicator ────────────────────────────────────────────────────

/// Single-rank stub communicator for unit tests.
struct StubComm {
    rank: usize,
    size: usize,
}

impl Communicator for StubComm {
    fn allgatherv<T: CommData>(
        &self,
        _send: &[T],
        _recv: &mut [T],
        _counts: &[usize],
        _displs: &[usize],
    ) -> Result<(), CommError> {
        unreachable!("StubComm allgatherv not used in simulate tests")
    }

    fn allreduce<T: CommData>(
        &self,
        _send: &[T],
        _recv: &mut [T],
        _op: ReduceOp,
    ) -> Result<(), CommError> {
        unreachable!("StubComm allreduce not used in simulate tests")
    }

    fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
        unreachable!("StubComm broadcast not used in simulate tests")
    }

    fn barrier(&self) -> Result<(), CommError> {
        Ok(())
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.size
    }

    fn abort(&self, error_code: i32) -> ! {
        std::process::exit(error_code)
    }
}

// ── Mock solver ──────────────────────────────────────────────────────────

/// Mock solver that returns a configurable fixed `LpSolution` on every solve.
///
/// Optionally returns `SolverError::Infeasible` at a specific (0-based) solve
/// call index (counting across both cold-start and warm-start calls).
///
/// `load_count` and `add_rows_count` track how many times `load_model` and
/// `add_rows` were called, used by the frozen-path acceptance tests.
///
/// `solve_count` counts calls to the cold-start `solve(None)` path only.
/// `solve_with_basis_count` counts calls to the warm-start `solve(Some(&basis))`
/// path only.  `call_count` tracks the combined total across both paths and
/// is used by the infeasibility injection logic.
///
/// `recorded_basis` captures the last `Basis` passed to `solve(Some(&basis))`,
/// used by the warm-start reconstruction acceptance tests.
///
/// `reconstruction_counter` counts `record_reconstruction_stats` invocations
/// (one per warm-start solve that applied a stored basis via slot reconciliation).
struct MockSolver {
    solution: LpSolution,
    infeasible_at: Option<usize>,
    call_count: usize,
    buf_primal: Vec<f64>,
    buf_dual: Vec<f64>,
    buf_reduced_costs: Vec<f64>,
    /// Number of `load_model` calls since construction.
    load_count: usize,
    /// Number of `add_rows` calls since construction.
    add_rows_count: usize,
    /// Number of cold-start `solve(None)` calls.
    solve_count: usize,
    /// Number of warm-start `solve(Some(&basis))` calls.
    solve_with_basis_count: usize,
    /// Last `Basis` passed to `solve(Some(&basis))`, if any.
    recorded_basis: Option<Basis>,
    /// Cumulative `preserved` argument from `record_reconstruction_stats`.
    reconstruction_counter: u32,
}

impl MockSolver {
    fn always_ok(solution: LpSolution) -> Self {
        let buf_primal = solution.primal.clone();
        let buf_dual = solution.dual.clone();
        let buf_reduced_costs = solution.reduced_costs.clone();
        Self {
            solution,
            infeasible_at: None,
            call_count: 0,
            buf_primal,
            buf_dual,
            buf_reduced_costs,
            load_count: 0,
            add_rows_count: 0,
            solve_count: 0,
            solve_with_basis_count: 0,
            recorded_basis: None,
            reconstruction_counter: 0,
        }
    }

    fn do_solve(&mut self) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
        let call = self.call_count;
        self.call_count += 1;
        if self.infeasible_at == Some(call) {
            return Err(SolverError::Infeasible);
        }
        self.buf_primal.clone_from(&self.solution.primal);
        self.buf_dual.clone_from(&self.solution.dual);
        self.buf_reduced_costs
            .clone_from(&self.solution.reduced_costs);
        Ok(cobre_solver::SolutionView {
            objective: self.solution.objective,
            primal: &self.buf_primal,
            dual: &self.buf_dual,
            reduced_costs: &self.buf_reduced_costs,
            iterations: self.solution.iterations,
            solve_time_seconds: self.solution.solve_time_seconds,
        })
    }
}

impl SolverInterface for MockSolver {
    type Profile = cobre_solver::ActiveProfile;

    fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}

    fn solver_name_version(&self) -> String {
        "MockSolver 0.0.0".to_string()
    }
    fn load_model(&mut self, _template: &StageTemplate) {
        self.load_count += 1;
    }
    fn add_rows(&mut self, _cuts: &RowBatch) {
        self.add_rows_count += 1;
    }
    fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn solve(
        &mut self,
        basis: Option<&Basis>,
    ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
        if let Some(b) = basis {
            self.solve_with_basis_count += 1;
            self.recorded_basis = Some(b.clone());
        } else {
            self.solve_count += 1;
        }
        self.do_solve()
    }
    fn get_basis(&mut self, _out: &mut Basis) {}
    fn record_reconstruction_stats(&mut self) {
        self.reconstruction_counter += 1;
    }
    fn statistics(&self) -> SolverStatistics {
        SolverStatistics::default()
    }
    fn statistics_into(&self, out: &mut SolverStatistics) {
        out.copy_from(&SolverStatistics::default());
    }
    fn name(&self) -> &'static str {
        "Mock"
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Minimal valid stage template for N=1 hydro, L=0 PAR order.
///
/// Column layout (N=1, L=0):
/// - col 0: `storage_out` (no NZ in structural rows)
/// - col 1: `z_inflow` (no NZ — `z_inflow` row at row 1)
/// - col 2: `storage_in` (1 NZ: row 0, storage-fixing row)
/// - col 3: `theta` (no NZ)
///
/// Row layout:
/// - row 0: storage-fixing (`storage_out` fixed to incoming state)
/// - row 1: `z_inflow` definition row
fn minimal_template_1_0() -> StageTemplate {
    StageTemplate {
        num_cols: 4,
        num_rows: 2,
        num_nz: 1,
        // CSC col_starts: 4 cols + 1 sentinel = 5 entries.
        // col 0 (storage_out): 0 NZ
        // col 1 (z_inflow):    0 NZ
        // col 2 (storage_in):  1 NZ at row 0
        // col 3 (theta):       0 NZ
        col_starts: vec![0_i32, 0, 0, 1, 1],
        row_indices: vec![0_i32],
        values: vec![1.0],
        col_lower: vec![0.0, f64::NEG_INFINITY, 0.0, 0.0],
        col_upper: vec![f64::INFINITY; 4],
        objective: vec![0.0, 0.0, 0.0, 1.0],
        row_lower: vec![0.0, 0.0],
        row_upper: vec![0.0, 0.0],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 1,
        n_hydro: 1,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

/// Build a fixed `LpSolution` for the minimal N=1 L=0 template.
///
/// N=1 L=0 column layout: `storage`(0), `z_inflow`(1), `storage_in`(2), `theta`(3).
/// `primal[3] = theta_val`, `objective = objective`.
fn fixed_solution(objective: f64, theta_val: f64) -> LpSolution {
    let num_cols = 4; // storage(0), z_inflow(1), storage_in(2), theta(3)
    let mut primal = vec![0.0_f64; num_cols];
    primal[3] = theta_val; // theta at col 3 (N=1, L=0 → theta = N*(3+L) = 3)
    LpSolution {
        objective,
        primal,
        dual: vec![0.0_f64; 2],
        reduced_costs: vec![0.0_f64; num_cols],
        iterations: 0,
        solve_time_seconds: 0.0,
    }
}

/// Build a minimal `EntityCounts` for 1 hydro, no other entities.
fn entity_counts_1_hydro() -> EntityCounts {
    EntityCounts {
        hydro_ids: vec![1],
        hydro_productivities: vec![1.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    }
}

/// Build a minimal stochastic context for 1 hydro, `n_stages` stages.
fn make_stochastic_context(n_stages: usize) -> StochasticContext {
    use std::collections::BTreeMap;

    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
    };
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
    use cobre_stochastic::context::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let hydro = Hydro {
        id: EntityId(1),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(0),
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
    let make_stage = |idx: usize, id: i32| Stage {
        index: idx,
        id,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
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
    };
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| make_stage(i, i32::try_from(i).unwrap()))
        .collect();
    let inflow = |stage_id: i32| InflowModel {
        hydro_id: EntityId(1),
        stage_id,
        mean_m3s: 100.0,
        std_m3s: 30.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    };
    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| inflow(i32::try_from(i).unwrap()))
        .collect();
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: vec![CorrelationEntity {
                    entity_type: "inflow".to_string(),
                    id: EntityId(1),
                }],
                matrix: vec![vec![1.0]],
            }],
        },
    );
    let correlation = CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    };
    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .build()
        .unwrap();
    build_stochastic_context(
        &system,
        42,
        None,
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

// ── Helpers ──────────────────────────────────────────────────────────────

/// Build per-stage hydro productivities for 1 hydro with productivity 1.0.
///
/// Returns `n_stages` inner vecs each containing a single `1.0` entry,
/// matching `entity_counts_1_hydro().hydro_productivities`.
fn hydro_productivities_1hydro(n_stages: usize) -> Vec<Vec<f64>> {
    vec![vec![1.0]; n_stages]
}

/// Build a zero-valued [`crate::energy_conversion::EnergyConversionSet`] for tests
/// that do not assert on energy fields.
fn zero_energy_conversion(
    n_hydros: usize,
    n_stages: usize,
) -> crate::energy_conversion::EnergyConversionSet {
    use crate::energy_conversion::{EnergyConversion, EnergyConversionSet};
    let zero_ec = EnergyConversion {
        equivalent_productivity_mw_per_m3s: 0.0,
        reference_volume_hm3: 0.0,
        reference_outflow_m3s: 0.0,
    };
    EnergyConversionSet::new(
        vec![vec![zero_ec; n_stages]; n_hydros],
        vec![vec![0.0_f64; n_stages]; n_hydros],
        n_hydros,
        n_stages,
    )
}

/// Wrap a `MockSolver` in a single-workspace slice for `simulate()` calls.
///
/// All tests use a single workspace (serial execution) so that existing
/// assertions about scenario ordering and call counts remain valid.
fn single_workspace(solver: MockSolver) -> Vec<SolverWorkspace<MockSolver>> {
    vec![SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(solver),
        patch_buf: PatchBuffer::new(1, 0, 0, 0, 0, 0, 0), // N=1, L=0
        current_state: Vec::with_capacity(1),
        scratch: ScratchBuffers {
            noise_buf: Vec::new(),
            inflow_m3s_buf: Vec::new(),
            lag_matrix_buf: Vec::new(),
            par_inflow_buf: Vec::new(),
            eta_floor_buf: Vec::new(),
            zero_targets_buf: Vec::new(),
            ncs_col_upper_buf: Vec::new(),
            ncs_col_lower_buf: Vec::new(),
            ncs_col_indices_buf: Vec::new(),
            ncs_col_lower_active_buf: Vec::new(),
            ncs_col_upper_active_buf: Vec::new(),
            last_ncs_col_start: usize::MAX,
            ncs_col_upper_extract_buf: Vec::new(),
            load_rhs_buf: Vec::new(),
            row_lower_buf: Vec::new(),
            z_inflow_rhs_buf: Vec::new(),
            effective_eta_buf: Vec::new(),
            unscaled_primal: Vec::new(),
            unscaled_dual: Vec::new(),
            lag_accumulator: vec![],
            lag_weight_accum: 0.0,
            downstream_accumulator: Vec::new(),
            downstream_weight_accum: 0.0,
            downstream_completed_lags: Vec::new(),
            downstream_n_completed: 0,
            recon_slot_lookup: Vec::new(),
            trajectory_costs_buf: Vec::new(),
            raw_noise_buf: Vec::new(),
            perm_scratch: Vec::new(),
        },
        scratch_basis: Basis::new(0, 0),
        backward_accum: BackwardAccumulators::default(),
        worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
    }]
}

// ── Load noise simulation tests ───────────────────────────────────────────

/// Build a stochastic context with 1 hydro and 1 stochastic load bus for
/// simulation load noise tests.
///
/// The context has 1 stage with branching factor 3.  The load model uses
/// `bus_id=1` (distinct from the hydro bus at `bus_id=0`), so the noise
/// vector has dimension 2: `[inflow_eta, load_eta]`.
fn make_stochastic_context_1_hydro_1_load_bus_sim(mean_mw: f64, std_mw: f64) -> StochasticContext {
    use std::collections::BTreeMap;

    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{CorrelationModel, InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
    use cobre_stochastic::context::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

    let bus0 = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let bus1 = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let hydro = Hydro {
        id: EntityId(10),
        name: "H10".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(0),
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
    let stage = Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
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
    };
    let inflow_model = InflowModel {
        hydro_id: EntityId(10),
        stage_id: 0,
        mean_m3s: 100.0,
        std_m3s: 20.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    };
    let load_model = LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw,
        std_mw,
    };
    let correlation = CorrelationModel {
        method: "spectral".to_string(),
        profiles: BTreeMap::new(),
        schedule: vec![],
    };
    let system = SystemBuilder::new()
        .buses(vec![bus0, bus1])
        .hydros(vec![hydro])
        .stages(vec![stage])
        .inflow_models(vec![inflow_model])
        .load_models(vec![load_model])
        .correlation(correlation)
        .build()
        .unwrap();
    build_stochastic_context(
        &system,
        42,
        None,
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

/// When a simulation has 1 stochastic load bus (mean=300, std=30),
/// verify that `load_rhs_buf` is populated with a positive value.
#[test]
fn simulation_load_patches_applied() {
    let n_stages = 1;
    let template = StageTemplate {
        num_cols: 3,
        num_rows: 3,
        num_nz: 1,
        col_starts: vec![0_i32, 0, 1, 1],
        row_indices: vec![0_i32],
        values: vec![1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
        objective: vec![0.0, 0.0, 1.0],
        row_lower: vec![0.0, 100.0, 300.0], // row 2 = load balance with mean=300
        row_upper: vec![0.0, 100.0, 300.0],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 3,
        n_hydro: 1,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };
    let templates = vec![template];
    let base_rows = vec![1usize]; // water-balance rows start at row 1

    let n_load_buses = 1usize;
    let stochastic = make_stochastic_context_1_hydro_1_load_bus_sim(300.0, 30.0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let config = SimulationConfig {
        n_scenarios: 1,
        io_channel_capacity: 4,
    };
    let state = crate::test_support::state_layout(1, 0);
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = fixed_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (tx, _rx) = mpsc::sync_channel(4);

    let mut workspaces = vec![SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(solver),
        patch_buf: PatchBuffer::new(1, 0, n_load_buses, 1, 0, 0, 0),
        current_state: Vec::with_capacity(1),
        scratch: ScratchBuffers {
            noise_buf: Vec::new(),
            inflow_m3s_buf: Vec::new(),
            lag_matrix_buf: Vec::new(),
            par_inflow_buf: Vec::new(),
            eta_floor_buf: Vec::new(),
            zero_targets_buf: Vec::new(),
            ncs_col_upper_buf: Vec::new(),
            ncs_col_lower_buf: Vec::new(),
            ncs_col_indices_buf: Vec::new(),
            ncs_col_lower_active_buf: Vec::new(),
            ncs_col_upper_active_buf: Vec::new(),
            last_ncs_col_start: usize::MAX,
            ncs_col_upper_extract_buf: Vec::new(),
            load_rhs_buf: Vec::with_capacity(n_load_buses),
            row_lower_buf: Vec::new(),
            z_inflow_rhs_buf: Vec::new(),
            effective_eta_buf: Vec::new(),
            unscaled_primal: Vec::new(),
            unscaled_dual: Vec::new(),
            lag_accumulator: vec![],
            lag_weight_accum: 0.0,
            downstream_accumulator: Vec::new(),
            downstream_weight_accum: 0.0,
            downstream_completed_lags: Vec::new(),
            downstream_n_completed: 0,
            recon_slot_lookup: Vec::new(),
            trajectory_costs_buf: Vec::new(),
            raw_noise_buf: Vec::new(),
            perm_scratch: Vec::new(),
        },
        scratch_basis: Basis::new(0, 0),
        backward_accum: BackwardAccumulators::default(),
        worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
    }];

    // load_balance_row_starts[0]=2 (load balance row is row 2 in the template).
    // load_bus_indices=[0] (bus position 0 in the block layout).
    let load_balance_row_starts = vec![2usize];
    let load_bus_indices = vec![0usize];
    let block_counts_per_stage = vec![1usize];
    let noise_scale = vec![1.0_f64]; // 1 hydro, 1 stage

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    run_simulate(
        &mut workspaces,
        &StageContext {
            geometry_per_stage: &[],
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &noise_scale,
            n_hydros: 1,
            n_load_buses,
            load_balance_row_starts: &load_balance_row_starts,
            load_bus_indices: &load_bus_indices,
            block_counts_per_stage: &block_counts_per_stage,
            ncs_col_starts: &[],
            n_ncs: 0,
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            anticipated_windows: &[],
            study_stage_ids: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        &fcf,
        &TrainingContext {
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &crate::test_support::all_enabled_cut_state_layouts(
                &state, n_stages,
            ),
            study_dims: &crate::test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx,
            zeta_per_stage: &[],
            block_hours_per_stage: &[],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &[],
            ncs_col_starts: &[],
            n_ncs: 0,
            pumping_col_starts: &[],
            n_pumping: 0,
            geometry_per_stage: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &[],
            contract_is_import: &[],
            ncs_entity_ids_per_stage: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
        },
        None,
        &[],
        &comm,
    )
    .unwrap();

    // The load noise path must have populated load_rhs_buf.
    assert_eq!(
        workspaces[0].scratch.load_rhs_buf.len(),
        n_load_buses,
        "load_rhs_buf must have 1 entry (1 load bus x 1 block)"
    );
    assert!(
        workspaces[0].scratch.load_rhs_buf[0] > 0.0,
        "realization must be positive with mean=300, std=30: got {}",
        workspaces[0].scratch.load_rhs_buf[0]
    );

    // Verify the formula: d = max(0, mean + std * eta) * factor.
    // The exact eta drawn from the opening tree depends on the seed, but
    // we can verify formula consistency by back-computing eta from the
    // observed realization (d > 0 implies eta = (d - mean) / std).
    let d_observed = workspaces[0].scratch.load_rhs_buf[0];
    let mean_mw_val = 300.0_f64;
    let std_mw_val = 30.0_f64;
    assert!(
        d_observed != mean_mw_val,
        "realization must differ from template mean (noise was applied)"
    );
    let eta_back = (d_observed - mean_mw_val) / std_mw_val;
    let recomputed = (mean_mw_val + std_mw_val * eta_back).max(0.0);
    assert!(
        (d_observed - recomputed).abs() < 1e-10,
        "formula consistency: d={d_observed}, eta_back={eta_back}, recomputed={recomputed}"
    );

    // The load patch must also be reflected in the patch buffer.
    let load_start = 1; // n_hydros = 1
    assert_eq!(
        workspaces[0].patch_buf.lower[load_start], workspaces[0].scratch.load_rhs_buf[0],
        "patch_buf lower at load slot must equal load_rhs_buf[0]"
    );
    assert_eq!(
        workspaces[0].patch_buf.upper[load_start], workspaces[0].scratch.load_rhs_buf[0],
        "patch_buf upper at load slot must equal load_rhs_buf[0] (equality constraint)"
    );

    // Verify that the extraction row_lower_buf was patched with the
    // stochastic realization (not the template mean 300.0).
    // row_lower_buf[2] is the load balance row (row index 2).
    assert!(
        !workspaces[0].scratch.row_lower_buf.is_empty(),
        "row_lower_buf must be populated for extraction"
    );
    assert_eq!(
        workspaces[0].scratch.row_lower_buf[2], d_observed,
        "extraction row_lower_buf must contain stochastic load, not template mean"
    );
    assert!(
        (workspaces[0].scratch.row_lower_buf[2] - mean_mw_val).abs() > 1e-6,
        "extracted load_mw must differ from template mean {mean_mw_val}: got {}",
        workspaces[0].scratch.row_lower_buf[2]
    );
}

/// when `n_load_buses == 0`,
/// `load_rhs_buf` remains empty and `forward_patch_count` equals `N`.
///
/// With N=1, L=0: `forward_patch_count = 1`.
#[test]
fn simulation_no_load_buses_unchanged() {
    let n_stages = 1;
    let templates = vec![minimal_template_1_0()];
    let base_rows = vec![0usize];

    let stochastic = make_stochastic_context(n_stages);
    let state = crate::test_support::state_layout(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let config = SimulationConfig {
        n_scenarios: 1,
        io_channel_capacity: 4,
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = fixed_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (tx, _rx) = mpsc::sync_channel(4);

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace(solver);

    run_simulate(
        &mut workspaces,
        &StageContext {
            geometry_per_stage: &[],
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1],
            ncs_col_starts: &[],
            n_ncs: 0,
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            anticipated_windows: &[],
            study_stage_ids: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        &fcf,
        &TrainingContext {
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &crate::test_support::all_enabled_cut_state_layouts(
                &state, n_stages,
            ),
            study_dims: &crate::test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx,
            zeta_per_stage: &[],
            block_hours_per_stage: &[],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &[],
            ncs_col_starts: &[],
            n_ncs: 0,
            pumping_col_starts: &[],
            n_pumping: 0,
            geometry_per_stage: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &[],
            contract_is_import: &[],
            ncs_entity_ids_per_stage: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
        },
        None,
        &[],
        &comm,
    )
    .unwrap();

    // load_rhs_buf must remain empty when n_load_buses=0.
    assert!(
        workspaces[0].scratch.load_rhs_buf.is_empty(),
        "load_rhs_buf must be empty when n_load_buses=0"
    );
    // forward_patch_count = N = 1 (no load patches added).
    assert_eq!(
        workspaces[0].patch_buf.forward_patch_count(),
        1,
        "forward_patch_count must be N=1 when n_load_buses=0, got {}",
        workspaces[0].patch_buf.forward_patch_count()
    );
}

/// When load noise is present,
/// `noise_buf` still contains only inflow values (not contaminated by load noise).
///
/// `noise_buf` contains inflow realizations for the `n_hydros` hydros.
/// After simulate runs with `n_hydros=1` and `n_load_buses=1`, `noise_buf`
/// must have exactly 1 entry (inflow), while `load_rhs_buf` has 1 entry
/// (load).  The two buffers must not overlap.
#[test]
fn simulation_inflow_extraction_unaffected() {
    let n_stages = 1;
    let template = StageTemplate {
        num_cols: 3,
        num_rows: 3,
        num_nz: 1,
        col_starts: vec![0_i32, 0, 1, 1],
        row_indices: vec![0_i32],
        values: vec![1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
        objective: vec![0.0, 0.0, 1.0],
        row_lower: vec![0.0, 100.0, 300.0],
        row_upper: vec![0.0, 100.0, 300.0],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 3,
        n_hydro: 1,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };
    let templates = vec![template];
    let base_rows = vec![1usize];

    let n_load_buses = 1usize;
    let stochastic = make_stochastic_context_1_hydro_1_load_bus_sim(300.0, 30.0);
    let state = crate::test_support::state_layout(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let config = SimulationConfig {
        n_scenarios: 1,
        io_channel_capacity: 4,
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = fixed_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (tx, _rx) = mpsc::sync_channel(4);

    let mut workspaces = vec![SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(solver),
        patch_buf: PatchBuffer::new(1, 0, n_load_buses, 1, 0, 0, 0),
        current_state: Vec::with_capacity(1),
        scratch: ScratchBuffers {
            noise_buf: Vec::new(),
            inflow_m3s_buf: Vec::new(),
            lag_matrix_buf: Vec::new(),
            par_inflow_buf: Vec::new(),
            eta_floor_buf: Vec::new(),
            zero_targets_buf: Vec::new(),
            ncs_col_upper_buf: Vec::new(),
            ncs_col_lower_buf: Vec::new(),
            ncs_col_indices_buf: Vec::new(),
            ncs_col_lower_active_buf: Vec::new(),
            ncs_col_upper_active_buf: Vec::new(),
            last_ncs_col_start: usize::MAX,
            ncs_col_upper_extract_buf: Vec::new(),
            load_rhs_buf: Vec::with_capacity(n_load_buses),
            row_lower_buf: Vec::new(),
            z_inflow_rhs_buf: Vec::new(),
            effective_eta_buf: Vec::new(),
            unscaled_primal: Vec::new(),
            unscaled_dual: Vec::new(),
            lag_accumulator: vec![],
            lag_weight_accum: 0.0,
            downstream_accumulator: Vec::new(),
            downstream_weight_accum: 0.0,
            downstream_completed_lags: Vec::new(),
            downstream_n_completed: 0,
            recon_slot_lookup: Vec::new(),
            trajectory_costs_buf: Vec::new(),
            raw_noise_buf: Vec::new(),
            perm_scratch: Vec::new(),
        },
        scratch_basis: Basis::new(0, 0),
        backward_accum: BackwardAccumulators::default(),
        worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
    }];

    let load_balance_row_starts = vec![2usize];
    let load_bus_indices = vec![0usize];
    let block_counts_per_stage = vec![1usize];
    let noise_scale = vec![1.0_f64];

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    run_simulate(
        &mut workspaces,
        &StageContext {
            geometry_per_stage: &[],
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &noise_scale,
            n_hydros: 1,
            n_load_buses,
            load_balance_row_starts: &load_balance_row_starts,
            load_bus_indices: &load_bus_indices,
            block_counts_per_stage: &block_counts_per_stage,
            ncs_col_starts: &[],
            n_ncs: 0,
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            anticipated_windows: &[],
            study_stage_ids: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        &fcf,
        &TrainingContext {
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &crate::test_support::all_enabled_cut_state_layouts(
                &state, n_stages,
            ),
            study_dims: &crate::test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx,
            zeta_per_stage: &[],
            block_hours_per_stage: &[],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &[],
            ncs_col_starts: &[],
            n_ncs: 0,
            pumping_col_starts: &[],
            n_pumping: 0,
            geometry_per_stage: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &[],
            contract_is_import: &[],
            ncs_entity_ids_per_stage: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
        },
        None,
        &[],
        &comm,
    )
    .unwrap();

    // noise_buf contains only inflow values (n_hydros=1 entries).
    // It must not be contaminated by load noise (which lives in load_rhs_buf).
    assert_eq!(
        workspaces[0].scratch.noise_buf.len(),
        1,
        "noise_buf must have 1 entry (1 hydro), not contaminated by load noise: len={}",
        workspaces[0].scratch.noise_buf.len()
    );
    // The inflow noise must be a reasonable value near mean_rhs=100.
    // With noise_scale=1.0 and mean_rhs=100 (from row_lower[base_rows[0]+0]=100):
    //   noise_buf[0] = 100.0 + 1.0 * eta_inflow
    // For any |eta_inflow| <= 5 this remains in [75, 125] for practical draws.
    assert!(
        workspaces[0].scratch.noise_buf[0] > 50.0 && workspaces[0].scratch.noise_buf[0] < 200.0,
        "noise_buf[0] must be a reasonable inflow value near 100.0, got {}",
        workspaces[0].scratch.noise_buf[0]
    );
    // load_rhs_buf must also be populated (confirms both buffers coexist).
    assert_eq!(
        workspaces[0].scratch.load_rhs_buf.len(),
        n_load_buses,
        "load_rhs_buf must have 1 entry alongside noise_buf"
    );
}

// ── Inflow truncation unit tests ──────────────────────────────────────────

/// Build a `StochasticContext` for 1 hydro, 1 stage with configurable
/// `mean_m3s` and `std_m3s`.  Used by the truncation tests below.
fn make_stochastic_1h_1s(mean_m3s: f64, std_m3s: f64) -> StochasticContext {
    use std::collections::BTreeMap;

    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
    };
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
    use cobre_stochastic::context::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let hydro = Hydro {
        id: EntityId(1),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(0),
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
    let stage = Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
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
    };
    let inflow_model = InflowModel {
        hydro_id: EntityId(1),
        stage_id: 0,
        mean_m3s,
        std_m3s,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    };
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: vec![CorrelationEntity {
                    entity_type: "inflow".to_string(),
                    id: EntityId(1),
                }],
                matrix: vec![vec![1.0]],
            }],
        },
    );
    let correlation = CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    };
    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(vec![stage])
        .inflow_models(vec![inflow_model])
        .correlation(correlation)
        .build()
        .unwrap();
    build_stochastic_context(
        &system,
        42,
        None,
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

/// Build a stage template for N=1 hydro, L=0 PAR, with `row_lower[0] = base_rhs`.
///
/// Used by truncation tests so that the water-balance base RHS is configurable.
fn minimal_template_1_0_with_base(base_rhs: f64) -> StageTemplate {
    StageTemplate {
        num_cols: 3,
        num_rows: 1,
        num_nz: 1,
        col_starts: vec![0_i32, 0, 1, 1],
        row_indices: vec![0_i32],
        values: vec![1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
        objective: vec![0.0, 0.0, 1.0],
        row_lower: vec![base_rhs],
        row_upper: vec![base_rhs],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 1,
        n_hydro: 1,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

/// Build a workspace with `zero_targets_buf` pre-populated to `hydro_count` zeros.
///
/// The standard `single_workspace` helper leaves `zero_targets_buf` empty because
/// existing tests do not reach the truncation branch.  The truncation tests use
/// 1 hydro and must have `zero_targets_buf[..1]` accessible, so they use this
/// helper instead.
fn single_workspace_with_hydros(
    solver: MockSolver,
    hydro_count: usize,
) -> Vec<SolverWorkspace<MockSolver>> {
    vec![SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(solver),
        patch_buf: PatchBuffer::new(hydro_count, 0, 0, 0, 0, 0, 0),
        current_state: Vec::with_capacity(hydro_count),
        scratch: ScratchBuffers {
            noise_buf: Vec::new(),
            inflow_m3s_buf: Vec::new(),
            lag_matrix_buf: Vec::new(),
            par_inflow_buf: Vec::new(),
            eta_floor_buf: Vec::new(),
            zero_targets_buf: vec![0.0_f64; hydro_count],
            ncs_col_upper_buf: Vec::new(),
            ncs_col_lower_buf: Vec::new(),
            ncs_col_indices_buf: Vec::new(),
            ncs_col_lower_active_buf: Vec::new(),
            ncs_col_upper_active_buf: Vec::new(),
            last_ncs_col_start: usize::MAX,
            ncs_col_upper_extract_buf: Vec::new(),
            load_rhs_buf: Vec::new(),
            row_lower_buf: Vec::new(),
            z_inflow_rhs_buf: Vec::new(),
            effective_eta_buf: Vec::new(),
            unscaled_primal: Vec::new(),
            unscaled_dual: Vec::new(),
            lag_accumulator: vec![],
            lag_weight_accum: 0.0,
            downstream_accumulator: Vec::new(),
            downstream_weight_accum: 0.0,
            downstream_completed_lags: Vec::new(),
            downstream_n_completed: 0,
            recon_slot_lookup: Vec::new(),
            trajectory_costs_buf: Vec::new(),
            raw_noise_buf: Vec::new(),
            perm_scratch: Vec::new(),
        },
        scratch_basis: Basis::new(0, 0),
        backward_accum: BackwardAccumulators::default(),
        worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
    }]
}

/// AC: truncation clamps negative inflow noise in the simulation pipeline.
///
/// Set `mean_m3s = -1000.0` and `std_m3s = 1.0` so that the deterministic
/// PAR base alone would produce a hugely negative inflow for any sample.
///
/// With `InflowNonNegativityMethod::Truncation` active, the simulation pipeline
/// must clamp `eta` to the floor that produces zero inflow.  As a result,
/// `noise_buf[0] = base_rhs + noise_scale * eta_clamped >= 0.0` for all
/// scenarios processed.
///
/// Concretely (zeta=1): `base_rhs = -1000`, `noise_scale = 1`.
/// `eta_floor` = (0 - mean) / sigma = 1000. So `noise_buf`\[0\] = -1000 + 1\*1000 = 0.
#[test]
fn simulation_truncation_clamps_negative_inflow_noise() {
    let mean_m3s = -1000.0_f64;
    let sigma = 1.0_f64;
    let zeta = 1.0_f64; // simplified: treat zeta=1
    let base_rhs = zeta * mean_m3s;
    let noise_scale_val = zeta * sigma;

    let n_stages = 1;
    let stochastic = make_stochastic_1h_1s(mean_m3s, sigma);
    let template = minimal_template_1_0_with_base(base_rhs);
    let templates = vec![template];
    let base_rows = vec![0_usize];
    let noise_scale = vec![noise_scale_val];

    let state = crate::test_support::state_layout(1, 0);
    let fcf = FutureCostFunction::new(n_stages, state.n_state, 1, 10, &vec![0; n_stages]);
    let config = SimulationConfig {
        n_scenarios: 4,
        io_channel_capacity: 16,
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![0.0_f64];

    let solution = fixed_solution(0.0, 0.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (tx, _rx) = mpsc::sync_channel(16);

    // Use the hydro-aware workspace builder so zero_targets_buf[..1] is valid.
    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace_with_hydros(solver, 1);
    run_simulate(
        &mut workspaces,
        &StageContext {
            geometry_per_stage: &[],
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &noise_scale,
            n_hydros: 1,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[n_stages],
            ncs_col_starts: &[],
            n_ncs: 0,
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            anticipated_windows: &[],
            study_stage_ids: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        &fcf,
        &TrainingContext {
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &crate::test_support::all_enabled_cut_state_layouts(
                &state, n_stages,
            ),
            study_dims: &crate::test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::Truncation,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx,
            zeta_per_stage: &[],
            block_hours_per_stage: &[],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &[],
            ncs_col_starts: &[],
            n_ncs: 0,
            pumping_col_starts: &[],
            n_pumping: 0,
            geometry_per_stage: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &[],
            contract_is_import: &[],
            ncs_entity_ids_per_stage: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
        },
        None,
        &[],
        &comm,
    )
    .unwrap();

    // After truncation the noise_buf contains the value from the last stage solved
    // for the last scenario.  All scenarios must have produced a clamped value >= 0.
    // We verify the workspace buffer left by the final scenario-stage pair.
    assert_eq!(
        workspaces[0].scratch.noise_buf.len(),
        1,
        "noise_buf must have exactly 1 entry for 1 hydro"
    );
    assert!(
        workspaces[0].scratch.noise_buf[0] >= 0.0,
        "after truncation, noise_buf[0] must be >= 0 (inflow cannot be negative), got {}",
        workspaces[0].scratch.noise_buf[0]
    );
}

/// AC: `InflowNonNegativityMethod::None` in the simulation pipeline produces
/// raw (potentially negative) noise values, unchanged from the pre-fix behavior.
///
/// With `mean_m3s = -1000.0` and `std_m3s = 1.0`, the PAR inflow is always
/// deeply negative.  The `None` path must NOT clamp eta, so the noise buffer
/// value must be negative (`base_rhs` + `noise_scale` \* `raw_eta` << 0).
#[test]
fn simulation_none_method_produces_raw_negative_noise() {
    let mean_m3s = -1000.0_f64;
    let sigma = 1.0_f64;
    let zeta = 1.0_f64;
    let base_rhs = zeta * mean_m3s;
    let noise_scale_val = zeta * sigma;

    let n_stages = 1;
    let stochastic = make_stochastic_1h_1s(mean_m3s, sigma);
    let template = minimal_template_1_0_with_base(base_rhs);
    let templates = vec![template];
    let base_rows = vec![0_usize];
    let noise_scale = vec![noise_scale_val];

    let state = crate::test_support::state_layout(1, 0);
    let fcf = FutureCostFunction::new(n_stages, state.n_state, 1, 10, &vec![0; n_stages]);
    let config = SimulationConfig {
        n_scenarios: 4,
        io_channel_capacity: 16,
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![0.0_f64];

    let solution = fixed_solution(0.0, 0.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (tx, _rx) = mpsc::sync_channel(16);

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace_with_hydros(solver, 1);
    run_simulate(
        &mut workspaces,
        &StageContext {
            geometry_per_stage: &[],
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &noise_scale,
            n_hydros: 1,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[n_stages],
            ncs_col_starts: &[],
            n_ncs: 0,
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            anticipated_windows: &[],
            study_stage_ids: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        &fcf,
        &TrainingContext {
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &crate::test_support::all_enabled_cut_state_layouts(
                &state, n_stages,
            ),
            study_dims: &crate::test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx,
            zeta_per_stage: &[],
            block_hours_per_stage: &[],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &[],
            ncs_col_starts: &[],
            n_ncs: 0,
            pumping_col_starts: &[],
            n_pumping: 0,
            geometry_per_stage: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &[],
            contract_is_import: &[],
            ncs_entity_ids_per_stage: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
        },
        None,
        &[],
        &comm,
    )
    .unwrap();

    assert_eq!(
        workspaces[0].scratch.noise_buf.len(),
        1,
        "noise_buf must have exactly 1 entry for 1 hydro"
    );
    // With None, no clamping occurs.  base_rhs=-1000 and noise_scale=1, so
    // noise_buf[0] = -1000 + 1 * eta.  For |eta| < 5 this remains << 0.
    assert!(
        workspaces[0].scratch.noise_buf[0] < 0.0,
        "with None method, noise_buf[0] must be negative (raw eta applied), got {}",
        workspaces[0].scratch.noise_buf[0]
    );
}

// -----------------------------------------------------------------------
// Simulation DCS integration tests (real ActiveSolver)
// -----------------------------------------------------------------------
//
// Exercise the simulation DCS branch in `solve_simulation_stage`: load the
// cut-free base, patch the pinned incoming state, solve the cut pool lazily,
// and extract the primal. The LP shapes mirror the forward/backward DCS
// fixtures (a coupling row ties storage_out col0 to the pinned storage_in
// col2, and cuts constrain theta against col0). The theta-sensitive
// observable is `SimulationStageResult.future_cost = primal[theta] * SCALE`,
// which distinguishes the converged theta and so catches a wrong (frozen)
// template load.
mod dcs_simulation {
    use std::collections::HashMap;
    use std::sync::mpsc;

    use cobre_solver::{ActiveSolver, StageTemplate};

    use super::super::{
        SimLookups, SimStageIds, SimStageLoadSpec, SimulationOutputSpec, solve_simulation_stage,
    };
    use crate::context::{StageContext, TrainingContext};
    use crate::cut::FutureCostFunction;
    use crate::cut_selection::CutMetadata;
    use crate::dcs::DcsParams;
    use crate::energy_conversion::{EnergyConversion, EnergyConversionSet};
    use crate::horizon_mode::HorizonMode;

    use crate::inflow_method::InflowNonNegativityMethod;
    use crate::lp_builder::PatchBuffer;
    use crate::simulation::extraction::EntityCounts;
    use crate::simulation::types::SimulationStageResult;
    use crate::workspace::{SolverWorkspace, WorkspaceSizing};

    const X_HAT: f64 = 2.0;

    /// Cut-free base: cols `[storage_out=0, z_inflow=1, storage_in=2,
    /// theta=3]`, one coupling row `storage_out - storage_in = 0`, minimise
    /// `theta`. `storage_in` is pinned to `x_hat`; cuts constrain `theta`
    /// against `storage_out` (col 0).
    fn sim_core_template() -> StageTemplate {
        StageTemplate {
            num_cols: 4,
            num_rows: 1,
            num_nz: 2,
            col_starts: vec![0_i32, 1, 1, 2, 2],
            row_indices: vec![0_i32, 0],
            values: vec![1.0, -1.0],
            col_lower: vec![0.0, 0.0, 0.0, -1.0e6],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY, 1.0e6],
            objective: vec![0.0, 0.0, 0.0, 1.0],
            row_lower: vec![0.0],
            row_upper: vec![0.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// All-cuts frozen template: cut-free base + the three pool cuts frozen as
    /// structural rows 1..4 (slot order). `num_rows = 4`.
    fn sim_all_cuts_frozen() -> StageTemplate {
        StageTemplate {
            num_cols: 4,
            num_rows: 4,
            num_nz: 6,
            col_starts: vec![0_i32, 2, 2, 3, 6],
            row_indices: vec![0_i32, 2, 0, 1, 2, 3],
            values: vec![1.0, -2.0, -1.0, 1.0, 1.0, 1.0],
            col_lower: vec![0.0, 0.0, 0.0, -1.0e6],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY, 1.0e6],
            objective: vec![0.0, 0.0, 0.0, 1.0],
            row_lower: vec![0.0, 1.0, 0.0, 3.0],
            row_upper: vec![0.0, f64::INFINITY, f64::INFINITY, f64::INFINITY],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Frozen template carrying a single DOMINATING spurious cut
    /// (`-5*col0 + theta >= 0`, floor 10 at `x_hat = 2`, NOT in the pool).
    fn sim_frozen_dominating_cut() -> StageTemplate {
        StageTemplate {
            num_cols: 4,
            num_rows: 2,
            num_nz: 4,
            col_starts: vec![0_i32, 2, 2, 3, 4],
            row_indices: vec![0_i32, 1, 0, 1],
            values: vec![1.0, -5.0, -1.0, 1.0],
            col_lower: vec![0.0, 0.0, 0.0, -1.0e6],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY, 1.0e6],
            objective: vec![0.0, 0.0, 0.0, 1.0],
            row_lower: vec![0.0, 0.0],
            row_upper: vec![0.0, f64::INFINITY],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Pool of three cuts on incoming storage; seeded so the DCS initial set
    /// omits the binding slot 1 (stale `last_active_iter`).
    fn sim_pool() -> FutureCostFunction {
        let mut fcf = FutureCostFunction::new(1, 1, 8, 10, &[0]);
        fcf.add_cut(0, 0, 0, 1.0, &[0.0]);
        fcf.add_cut(0, 0, 1, 0.0, &[2.0]); // binding: floor 2*x_hat = 4
        fcf.add_cut(0, 0, 2, 3.0, &[0.0]);
        let meta = |generated: u64, last: u64| CutMetadata {
            iteration_generated: generated,
            forward_pass_index: 0,
            active_count: 0,
            last_active_iter: last,
        };
        fcf.pools[0].metadata[0] = meta(1, 5);
        fcf.pools[0].metadata[1] = meta(1, 1);
        fcf.pools[0].metadata[2] = meta(1, 5);
        fcf
    }

    fn sim_active_workspace() -> SolverWorkspace<ActiveSolver> {
        let sizing = WorkspaceSizing {
            hydro_count: 1,
            max_par_order: 0,
            n_load_buses: 0,
            max_blocks: 0,
            n_buckets: 0,
            downstream_par_order: 0,
            max_openings: 1,
            initial_pool_capacity: 16,
            n_state: 1,
            max_local_fwd: 1,
            total_forward_passes: 1,
            noise_dim: 1,
            n_anticipated: 0,
            k_max: 0,
        };
        let solver = ActiveSolver::new().expect("ActiveSolver::new()");
        SolverWorkspace::new(
            0,
            0,
            solver,
            PatchBuffer::new(1, 0, 0, 0, 0, 0, 0),
            1,
            sizing,
        )
    }

    fn dcs_params() -> DcsParams {
        DcsParams {
            k1: None,
            k2: 2,
            nadic: 10,
            epsilon_viol: 1e-10,
            start_iteration: 2,
            max_inner_iterations: 50,
        }
    }

    /// Solve one simulation stage with the given `dcs` option and `frozen`
    /// template, returning `(immediate_cost, SimulationStageResult)`.
    #[allow(clippy::too_many_lines)]
    fn run_one_sim_stage(
        dcs: Option<DcsParams>,
        frozen: &StageTemplate,
    ) -> (f64, SimulationStageResult) {
        let state = crate::test_support::state_layout(1, 0);
        let core = sim_core_template();
        let templates = vec![core.clone()];
        let base_rows = vec![0_usize];
        let stochastic = super::make_stochastic_context(1);
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let fcf = sim_pool();
        let entity_counts = EntityCounts {
            hydro_ids: vec![1],
            hydro_productivities: vec![1.0],
            thermal_ids: vec![],
            line_ids: vec![],
            bus_ids: vec![],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        };
        let hprod = vec![vec![1.0]];
        let zero_ec = EnergyConversion {
            equivalent_productivity_mw_per_m3s: 0.0,
            reference_volume_hm3: 0.0,
            reference_outflow_m3s: 0.0,
        };
        let ec = EnergyConversionSet::new(vec![vec![zero_ec; 1]], vec![vec![0.0_f64; 1]], 1, 1);

        let mut ws = sim_active_workspace();
        ws.current_state.clear();
        ws.current_state.push(X_HAT);
        // Pre-fill the noise/load/z-inflow scratch the patch reads (empty for
        // a zero-noise single-hydro fixture, mirroring the caller's setup).
        ws.scratch.noise_buf.clear();
        ws.scratch.load_rhs_buf.clear();
        ws.scratch.z_inflow_rhs_buf.clear();
        ws.scratch.ncs_col_upper_buf.clear();
        ws.scratch.inflow_m3s_buf.clear();
        ws.scratch.inflow_m3s_buf.push(0.0);

        let ctx = StageContext {
            geometry_per_stage: &[],
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[1.0],
            n_hydros: 1,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize],
            ncs_col_starts: &[],
            n_ncs: 0,
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            anticipated_windows: &[],
            study_stage_ids: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let study_dims = crate::test_support::study_dims();
        let training_ctx = TrainingContext {
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &crate::test_support::all_enabled_cut_state_layouts(&state, 1),
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: cobre_core::scenario::SamplingScheme::InSample,
            load_scheme: cobre_core::scenario::SamplingScheme::InSample,
            ncs_scheme: cobre_core::scenario::SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            dcs,
        };

        let (tx, _rx) = mpsc::sync_channel(4);
        let diversion: HashMap<cobre_core::EntityId, Vec<usize>> = HashMap::new();
        let output = SimulationOutputSpec {
            result_tx: &tx,
            zeta_per_stage: &[1.0],
            block_hours_per_stage: &[vec![1.0]],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &[],
            ncs_col_starts: &[],
            n_ncs: 0,
            pumping_col_starts: &[],
            n_pumping: 0,
            geometry_per_stage: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &[],
            contract_is_import: &[],
            ncs_entity_ids_per_stage: &[],
            diversion_upstream: &diversion,
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
        };
        let ids = SimStageIds {
            t: 0,
            stage_id_u32: 0,
            scenario_id: 0,
        };
        let load_spec = SimStageLoadSpec {
            frozen_template: frozen,
            warm_basis: None,
        };
        let lookups = SimLookups::build(&crate::test_support::study_dims(), &[], 0, 1);

        let (immediate, result) = solve_simulation_stage(
            &mut ws,
            &ctx,
            &fcf,
            &training_ctx,
            &load_spec,
            &output,
            &ids,
            &lookups,
        )
        .expect("simulation stage solve must succeed");
        (immediate, result)
    }

    /// The stage-level cost record (block 0) carrying total/immediate/future.
    fn stage_cost(
        result: &SimulationStageResult,
    ) -> &crate::simulation::types::SimulationCostResult {
        result
            .costs
            .first()
            .expect("simulation stage result must carry a cost record")
    }

    /// AC1: the DCS branch (binding cut omitted from the seed) yields the
    /// same immediate cost and primal-derived fields as the frozen all-cuts
    /// path within 1e-9.
    #[test]
    fn sim_dcs_exact_matches_all_cuts() {
        let all_cuts = sim_all_cuts_frozen();
        let (frozen_imm, frozen) = run_one_sim_stage(None, &all_cuts);
        let (dcs_imm, dcs) = run_one_sim_stage(Some(dcs_params()), &all_cuts);

        // Returned immediate cost (the f64) must match within 1e-9.
        assert!(
            (frozen_imm - dcs_imm).abs() < 1e-9,
            "immediate cost: frozen {frozen_imm} vs DCS {dcs_imm}"
        );
        let bc = stage_cost(&frozen);
        let dc = stage_cost(&dcs);
        assert!(
            (bc.immediate_cost - dc.immediate_cost).abs() < 1e-9,
            "record immediate_cost: frozen {} vs DCS {}",
            bc.immediate_cost,
            dc.immediate_cost
        );
        // future_cost = theta * SCALE; the binding cut floor is 4 at x_hat=2.
        assert!(
            (bc.future_cost - dc.future_cost).abs() < 1e-9,
            "future_cost: frozen {} vs DCS {}",
            bc.future_cost,
            dc.future_cost
        );
        assert!(
            (bc.total_cost - dc.total_cost).abs() < 1e-9,
            "total_cost: frozen {} vs DCS {}",
            bc.total_cost,
            dc.total_cost
        );
        assert!(
            (dc.future_cost - 4.0 * super::super::COST_SCALE_FACTOR).abs() < 1e-3,
            "DCS future_cost must reflect the binding cut theta=4, got {}",
            dc.future_cost
        );
    }

    /// AC2: a frozen template embedding a DOMINATING cut (floor 10, NOT in the
    /// pool) must NOT change the DCS result — proving the cut-free
    /// `ctx.templates[t]` is loaded, not `load_spec.frozen_template`. A wrong
    /// load surfaces as `future_cost` reflecting `theta = 10`.
    #[test]
    fn sim_dcs_frozen_cuts_present_uses_cut_free_core() {
        let all_cuts_template = sim_all_cuts_frozen();
        let dominating = sim_frozen_dominating_cut();
        let (_, frozen) = run_one_sim_stage(None, &all_cuts_template);
        // DCS is handed the dominating frozen template but must ignore it and
        // load the cut-free base, recovering the all-cuts (theta=4) result.
        let (_, dcs) = run_one_sim_stage(Some(dcs_params()), &dominating);

        let ac = stage_cost(&frozen);
        let dc = stage_cost(&dcs);
        assert!(
            (ac.future_cost - dc.future_cost).abs() < 1e-9,
            "future_cost: all-cuts {} vs DCS {} (DCS must ignore the dominating \
                 frozen cut and load the cut-free base)",
            ac.future_cost,
            dc.future_cost
        );
        assert!(
            (ac.immediate_cost - dc.immediate_cost).abs() < 1e-9,
            "immediate_cost: all-cuts {} vs DCS {}",
            ac.immediate_cost,
            dc.immediate_cost
        );
    }

    /// Complementary unit check of the `DcsParams::from_strategy` mapping
    /// that `simulation_ctx()` applies (the end-to-end `simulation_ctx()`
    /// wiring is covered by `simulation_ctx_propagates_dynamic_dcs_from_setup`
    /// in `setup/mod.rs`): a dynamic strategy yields `Some`, any other
    /// variant yields `None`.
    #[test]
    fn from_strategy_gates_dynamic_dcs() {
        use crate::cut_selection::CutSelectionStrategy;
        let dynamic = CutSelectionStrategy::Dynamic {
            k1: None,
            k2: 5,
            nadic: 10,
            epsilon_viol: 1e-10,
            start_iteration: 2,
        };
        let params = DcsParams::from_strategy(&dynamic)
            .expect("dynamic strategy must map to Some(DcsParams)");
        assert_eq!(params.k2, 5);
        let dominated = CutSelectionStrategy::Dominated {
            threshold: 1e-6,
            check_frequency: 10,
        };
        assert!(DcsParams::from_strategy(&dominated).is_none());
    }
}
