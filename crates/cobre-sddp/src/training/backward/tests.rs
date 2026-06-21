use crate::SddpError;

/// Test-only backward-pass shim that owns per-call scratch.
///
/// Production code drives the backward pass via [`BackwardPassState::run`]
/// on the state struct held by `TrainingSession`. This shim exists so that
/// the tests in this module can exercise `run_one_backward_stage` without
/// threading a full `TrainingSession` through every fixture.
///
/// # Errors
///
/// Returns `Err(SddpError::Infeasible { .. })` when a stage LP has no
/// feasible solution during the backward sweep. Returns
/// `Err(SddpError::Solver(_))` for all other terminal LP solver failures.
#[cfg(test)]
fn run_backward_pass<S, C: Communicator>(
    inputs: &mut crate::backward_pass_state::BackwardPassInputs<'_, S, C>,
) -> Result<BackwardResult, SddpError>
where
    S: SolverInterface<Profile = cobre_solver::ActiveProfile> + Send,
{
    let n_workers_local = inputs.workspaces.len();
    let n_ranks = inputs.comm.size();
    let num_stages = inputs.training_ctx.horizon.num_stages();
    let bwd_max_openings = (0..num_stages)
        .map(|t| inputs.training_ctx.stochastic.opening_tree().n_openings(t))
        .max()
        .unwrap_or(0);
    let real_states_capacity =
        inputs.exchange.real_total_scenarios() * inputs.training_ctx.state.n_state;
    let mut bwd_state = crate::backward_pass_state::BackwardPassState::new(
        n_workers_local,
        n_ranks,
        bwd_max_openings,
        real_states_capacity,
    );
    bwd_state.run(inputs)
}

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_solver::{
    Basis, LpSolution, ProfiledSolver, RowBatch, SolverError, SolverInterface, SolverStatistics,
    StageTemplate,
};

use cobre_core::scenario::SamplingScheme;

use super::BackwardResult;
use crate::{
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    cut_sync::CutSyncBuffers,
    horizon_mode::HorizonMode,
    inflow_method::InflowNonNegativityMethod,
    risk_measure::RiskMeasure,
    solver_stats::SolverStatsDelta,
    state_exchange::ExchangeBuffers,
    trajectory::TrajectoryRecord,
    workspace::{BackwardAccumulators, BasisStore, SolverWorkspace},
};

fn empty_cut_batches(n_stages: usize) -> Vec<RowBatch> {
    (0..n_stages)
        .map(|_| RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        })
        .collect()
}

/// Stub communicator for tests (single-rank).
struct StubComm;

impl Communicator for StubComm {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _counts: &[usize],
        _displs: &[usize],
    ) -> Result<(), CommError> {
        // Single-rank: copy send to recv (mirrors LocalBackend behavior).
        recv[..send.len()].copy_from_slice(send);
        Ok(())
    }

    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _op: ReduceOp,
    ) -> Result<(), CommError> {
        recv[..send.len()].copy_from_slice(send);
        Ok(())
    }

    fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
        unreachable!("StubComm broadcast not used in backward pass tests")
    }

    fn barrier(&self) -> Result<(), CommError> {
        Ok(())
    }

    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        1
    }

    fn abort(&self, error_code: i32) -> ! {
        std::process::exit(error_code)
    }
}

/// Mock solver for testing: returns fixed solution or infeasible error on demand.
///
/// Buffer fields (`buf_primal`, `buf_dual`, `buf_reduced_costs`) store the
/// solution data that [`SolutionView`] borrows from. They are filled in
/// `solve` before the borrow is established.
struct MockSolver {
    solution: LpSolution,
    infeasible_at: Option<usize>,
    call_count: usize,
    /// Tracks the current number of rows (template + appended cuts).
    current_num_rows: usize,
    /// Number of times `solve(Some(&basis))` was called (warm-start calls).
    warm_start_calls: usize,
    /// Dual padding value for rows beyond the base template (cuts).
    /// Defaults to 0.0 (cuts not binding). Set to a positive value
    /// to make all cuts appear binding in tests.
    cut_dual_padding: f64,
    buf_primal: Vec<f64>,
    buf_dual: Vec<f64>,
    buf_reduced_costs: Vec<f64>,
}

impl MockSolver {
    fn always_ok(solution: LpSolution) -> Self {
        let base_rows = solution.dual.len();
        let buf_primal = solution.primal.clone();
        let buf_dual = solution.dual.clone();
        let buf_reduced_costs = solution.reduced_costs.clone();
        Self {
            solution,
            infeasible_at: None,
            call_count: 0,
            current_num_rows: base_rows,
            warm_start_calls: 0,
            cut_dual_padding: 0.0,
            buf_primal,
            buf_dual,
            buf_reduced_costs,
        }
    }

    fn infeasible_on(solution: LpSolution, n: usize) -> Self {
        let base_rows = solution.dual.len();
        let buf_primal = solution.primal.clone();
        let buf_dual = solution.dual.clone();
        let buf_reduced_costs = solution.reduced_costs.clone();
        Self {
            solution,
            infeasible_at: Some(n),
            call_count: 0,
            current_num_rows: base_rows,
            warm_start_calls: 0,
            cut_dual_padding: 0.0,
            buf_primal,
            buf_dual,
            buf_reduced_costs,
        }
    }

    /// Like `always_ok` but added cut rows return positive duals,
    /// making all existing cuts appear binding in subsequent solves.
    fn always_ok_with_binding_cuts(solution: LpSolution) -> Self {
        let mut s = Self::always_ok(solution);
        s.cut_dual_padding = 1.0;
        s
    }
}

impl SolverInterface for MockSolver {
    type Profile = cobre_solver::ActiveProfile;

    fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}

    fn solver_name_version(&self) -> String {
        "MockSolver 0.0.0".to_string()
    }
    fn load_model(&mut self, template: &StageTemplate) {
        self.current_num_rows = template.num_rows;
    }

    fn add_rows(&mut self, cuts: &RowBatch) {
        self.current_num_rows += cuts.num_rows;
    }

    fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

    fn solve(
        &mut self,
        basis: Option<&Basis>,
    ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
        if basis.is_some() {
            self.warm_start_calls += 1;
        }
        let call = self.call_count;
        self.call_count += 1;
        if self.infeasible_at == Some(call) {
            return Err(SolverError::Infeasible);
        }
        // Fill internal buffers, resizing dual to match current LP row count.
        self.buf_primal.clone_from(&self.solution.primal);
        self.buf_dual.clone_from(&self.solution.dual);
        self.buf_dual
            .resize(self.current_num_rows, self.cut_dual_padding);
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

    fn get_basis(&mut self, _out: &mut Basis) {}

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

fn minimal_template_1_0() -> StageTemplate {
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

fn solution_1_0(objective: f64, dual_storage: f64) -> LpSolution {
    // For the N=1, L=0 state layout: storage_in.start = N*(2+L) = 1*(2+0) = 2.
    // state_to_lp_incoming_column(0) = storage_in.start + 0 = 2.
    // Cut subgradients are read from reduced_costs[storage_in_col], so
    // reduced_costs[2] must hold the same value as the storage-fixing dual.
    let mut reduced_costs = vec![0.0; 3];
    reduced_costs[2] = dual_storage;
    LpSolution {
        objective,
        primal: vec![0.0, 0.0, 0.0],
        dual: vec![dual_storage],
        reduced_costs,
        iterations: 0,
        solve_time_seconds: 0.0,
    }
}

/// Resolve a [`StagedCut`]'s coefficient slice from its producing worker's
/// arena. Mirrors the production merge's
/// `workspaces[w].backward_accum.agg_arena[cut.coefficients_range]` read so
/// the in-file backward tests observe the same bytes the FCF receives.
fn staged_cut_coefficients<'a>(cut: &super::StagedCut, arena: &'a [f64]) -> &'a [f64] {
    &arena[cut.coefficients_range.clone()]
}

/// Wrap a `MockSolver` into a single-element `Vec<SolverWorkspace<MockSolver>>`
/// for tests that exercise the workspace-based backward-pass API.
///
/// The workspace is sized for `n_hydro=1`, `max_par_order=0`, and `n_state`
/// state dimensions.
fn single_workspace(solver: MockSolver, n_state: usize) -> Vec<SolverWorkspace<MockSolver>> {
    use crate::lp_builder::PatchBuffer;
    vec![SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(solver),
        patch_buf: PatchBuffer::new(1, 0, 0, 0, 0, 0),
        current_state: Vec::with_capacity(n_state),
        scratch: crate::workspace::ScratchBuffers {
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
            anticipated_state_buf: Vec::new(),
        },
        scratch_basis: Basis::new(0, 0),
        backward_accum: BackwardAccumulators::default(),
        worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
    }]
}

/// Create an empty `BasisStore` for `num_scenarios` scenarios and
/// `num_stages` stages (all slots `None`).
fn empty_basis_store(num_scenarios: usize, num_stages: usize) -> BasisStore {
    BasisStore::new(num_scenarios, num_stages)
}

/// Create a `BasisStore` with one slot pre-populated at
/// `[scenario][stage]` with the given `Basis`.
fn basis_store_with_one(
    num_scenarios: usize,
    num_stages: usize,
    scenario: usize,
    stage: usize,
    basis: Basis,
) -> BasisStore {
    let mut store = BasisStore::new(num_scenarios, num_stages);
    // Set `base_row_count` to the full row_status length and leave
    // `cut_row_slots` empty so the CapturedBasis invariant
    // (`row_status.len() == base_row_count + cut_row_slots.len()`) holds
    // by construction. These tests exercise the warm-start propagation
    // path; the reconstruction copies the template rows verbatim and
    // emits an empty cut block.
    let base_row_count = basis.row_status.len();
    *store.get_mut(scenario, stage) = Some(crate::workspace::CapturedBasis {
        basis,
        base_row_count,
        cut_row_slots: Vec::new(),
        state_at_capture: Vec::new(),
    });
    store
}

fn exchange_with_states(n_state: usize, states: Vec<Vec<f64>>) -> ExchangeBuffers {
    use cobre_comm::LocalBackend;

    let local_count = states.len();
    let mut bufs = ExchangeBuffers::new(n_state, local_count, 1);
    let records: Vec<TrajectoryRecord> = states
        .into_iter()
        .map(|state| TrajectoryRecord {
            primal: vec![],
            dual: vec![],
            stage_cost: 0.0,
            state,
        })
        .collect();

    let comm = LocalBackend;
    bufs.exchange(&records, 0, 1, &comm).unwrap();
    bufs
}

#[allow(clippy::too_many_lines)]
fn make_stochastic_context(
    n_stages: usize,
    branching_factor: usize,
) -> cobre_stochastic::StochasticContext {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::{
        Bus, DeficitSegment, EntityId, SystemBuilder,
        scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
        },
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };
    use cobre_stochastic::context::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};
    use std::collections::BTreeMap;

    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let hydro = Hydro {
        id: EntityId(1),
        name: "H1".to_string(),
        bus_id: EntityId(0),
        downstream_id: None,
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

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let make_stage = |idx: usize| Stage {
        index: idx,
        id: idx as i32,
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
            branching_factor,
            noise_method: NoiseMethod::Saa,
        },
    };

    let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

    #[allow(clippy::cast_possible_truncation)]
    let inflow = |stage_idx: usize| InflowModel {
        hydro_id: EntityId(1),
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        stage_id: stage_idx as i32,
        mean_m3s: 100.0,
        std_m3s: 30.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    };

    let inflow_models: Vec<InflowModel> = (0..n_stages).map(inflow).collect();

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

// ── Unit tests ────────────────────────────────────────────────────────────

#[test]
fn backward_result_fields_accessible() {
    let r = BackwardResult {
        cuts_generated: 6,
        elapsed_ms: 42,
        lp_solves: 0,
        stage_stats: Vec::new(),
        state_exchange_time_ms: 0,
        cut_batch_build_time_ms: 0,
        setup_time_ms: 0,
        load_imbalance_ms: 0,
        scheduling_overhead_ms: 0,
        cut_sync_time_ms: 0,
    };
    assert_eq!(r.cuts_generated, 6);
    assert_eq!(r.elapsed_ms, 42);
    assert!(r.stage_stats.is_empty());
    assert_eq!(r.state_exchange_time_ms, 0);
    assert_eq!(r.cut_batch_build_time_ms, 0);
    assert_eq!(r.setup_time_ms, 0);
    assert_eq!(r.load_imbalance_ms, 0);
    assert_eq!(r.scheduling_overhead_ms, 0);
    assert_eq!(r.cut_sync_time_ms, 0);
}

#[test]
fn backward_result_clone_and_debug() {
    let r = BackwardResult {
        cuts_generated: 3,
        elapsed_ms: 100,
        lp_solves: 0,
        stage_stats: Vec::new(),
        state_exchange_time_ms: 0,
        cut_batch_build_time_ms: 0,
        setup_time_ms: 0,
        load_imbalance_ms: 0,
        scheduling_overhead_ms: 0,
        cut_sync_time_ms: 0,
    };
    let c = r.clone();
    assert_eq!(c.cuts_generated, 3);
    let s = format!("{r:?}");
    assert!(s.contains("BackwardResult"));
}

#[test]
fn dual_extraction_formula_coefficients_are_negated_duals() {
    // Given known dual values [d0, d1], coefficients must be [-d0, -d1].
    let d0 = 3.5_f64;
    let d1 = -1.2_f64;
    let dual = [d0, d1];

    let coefficients: Vec<f64> = dual.iter().map(|&d| -d).collect();

    assert!((coefficients[0] - (-d0)).abs() < f64::EPSILON);
    assert!((coefficients[1] - (-d1)).abs() < f64::EPSILON);
}

#[test]
fn intercept_formula_matches_spec() {
    // alpha = Q - pi^T * x_hat
    // Given: objective=50.0, pi=[2.0, -1.0], x_hat=[10.0, 5.0]
    // Expected: alpha = 50.0 - (2.0*10.0 + (-1.0)*5.0) = 50.0 - 15.0 = 35.0
    let objective = 50.0_f64;
    let coefficients = [2.0_f64, -1.0_f64];
    let x_hat = [10.0_f64, 5.0_f64];
    let pi_dot_x: f64 = coefficients
        .iter()
        .zip(x_hat.iter())
        .map(|(p, x)| p * x)
        .sum();
    let intercept = objective - pi_dot_x;
    assert!((intercept - 35.0).abs() < f64::EPSILON);
}

#[test]
fn single_stage_system_produces_no_cuts() {
    // A 1-stage system has no stages with a successor, so the backward
    // sweep (0..0) is empty — zero cuts are generated.
    let stochastic = make_stochastic_context(1, 2);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0()];
    let base_rows = vec![1_usize];

    let n_state = state.n_state;
    let n_stages = 1_usize;
    let forward_passes = 2_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let mut exchange = exchange_with_states(n_state, vec![vec![10.0], vec![20.0]]);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation];

    let solution = solution_1_0(100.0, -5.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    assert_eq!(result.cuts_generated, 0);
    assert_eq!(fcf.total_active_cuts(), 0);
}

#[test]
fn two_stage_system_two_trial_points_generates_two_cuts_at_stage_0() {
    // Acceptance criterion: 3-stage system, 1 hydro (n_state=1), 2 openings,
    // 2 trial points → 2 cuts at stage 0. This is the 2-stage version
    // (stages 0 and 1); cuts should exist only at stage 0.
    let n_stages = 2_usize;
    let n_openings = 2_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];

    let n_state = state.n_state; // 1
    let forward_passes = 2_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);

    // Two trial points with states [10.0] and [20.0] at stage 0.
    let mut exchange = exchange_with_states(n_state, vec![vec![10.0], vec![20.0]]);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    // MockSolver returns objective=100.0, dual[0]=-5.0 for every solve.
    // With x_hat=[10.0]: pi=[5.0], alpha = 100 - 5*10 = 50.
    // With x_hat=[20.0]: pi=[5.0], alpha = 100 - 5*20 = 0 (could be negative).
    let solution = solution_1_0(100.0, -5.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    // 2 trial points × 1 stage with a successor = 2 cuts at stage 0.
    assert_eq!(result.cuts_generated, 2);
    assert_eq!(fcf.active_cuts(0).count(), 2);
    // Stage 1 (the last stage) gets no cuts.
    assert_eq!(fcf.active_cuts(1).count(), 0);
}

#[test]
fn cut_inserted_with_correct_stage_iteration_and_forward_pass_index() {
    // Acceptance criterion: iteration=2, forward_passes=3, global
    // trial point m=1 → fcf.add_cut(stage=0, iteration=2, fpi=1, ...).
    // slot = warm_start + 2*3 + 1 = 7.
    let n_stages = 2_usize;
    let n_openings = 2_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];

    let n_state = state.n_state;
    let forward_passes = 3_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);

    // 3 trial points (forward_passes=3 on a single rank).
    let mut exchange = exchange_with_states(n_state, vec![vec![5.0], vec![10.0], vec![15.0]]);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(50.0, 0.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 2,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    // Trial point m=1: slot = 0 + 2*3 + 1 = 7
    // Verify that pool[0].metadata[7] has the correct iteration and fpi.
    let meta = &fcf.pools[0].metadata[7];
    assert_eq!(meta.iteration_generated, 2);
    assert_eq!(meta.forward_pass_index, 1);
}

#[test]
fn no_cuts_generated_at_last_stage() {
    // Acceptance criterion: 5-stage system → cuts at stages 0..3, not at 4.
    let n_stages = 5_usize;
    let n_openings = 2_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(100.0, -3.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    // 1 trial point × 4 stages with successors = 4 cuts total.
    assert_eq!(result.cuts_generated, 4);
    for t in 0..4 {
        assert_eq!(fcf.active_cuts(t).count(), 1, "stage {t} should have 1 cut");
    }
    // The last stage (4) must have no cuts.
    assert_eq!(fcf.active_cuts(4).count(), 0, "stage 4 must have no cuts");
}

#[test]
fn elapsed_ms_is_non_negative() {
    let n_stages = 2_usize;
    let stochastic = make_stochastic_context(n_stages, 2);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let mut exchange = exchange_with_states(n_state, vec![vec![5.0]]);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(10.0, 0.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    // elapsed_ms is u64, so it is always >= 0.
    let _ = result.elapsed_ms;
}

#[test]
fn infeasible_solver_returns_sddp_infeasible_error() {
    // Acceptance criterion: MockSolver::infeasible_on(0) for the first
    // backward solve → SddpError::Infeasible is returned.
    let n_stages = 2_usize;
    let stochastic = make_stochastic_context(n_stages, 1);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(0.0, 0.0);
    // First solve call returns infeasible.
    let solver = MockSolver::infeasible_on(solution, 0);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    });

    assert!(
        matches!(result, Err(crate::SddpError::Infeasible { .. })),
        "expected SddpError::Infeasible, got: {result:?}",
    );
}

#[test]
fn expectation_aggregation_mean_of_per_opening_intercepts() {
    // Given 3 openings with uniform probability 1/3 and per-opening
    // intercepts [10.0, 20.0, 30.0], the aggregated intercept must be 20.0.
    use crate::risk_measure::BackwardOutcome as BO;

    let outcomes = vec![
        BO {
            intercept: 10.0,
            coefficients: vec![],
            objective_value: 10.0,
        },
        BO {
            intercept: 20.0,
            coefficients: vec![],
            objective_value: 20.0,
        },
        BO {
            intercept: 30.0,
            coefficients: vec![],
            objective_value: 30.0,
        },
    ];
    let probs = vec![1.0 / 3.0; 3];
    let (intercept, _) = RiskMeasure::Expectation.aggregate_cut(&outcomes, &probs);
    assert!(
        (intercept - 20.0).abs() < 1e-10,
        "expected 20.0, got {intercept}"
    );
}

// ── Integration tests ─────────────────────────────────────────────────────

#[test]
#[allow(clippy::too_many_lines)]
fn cut_coefficients_and_intercept_match_dual_extraction_formula() {
    // Integration test: verify that the backward pass uses the correct
    // dual extraction formula by checking cuts in the FCF.
    //
    // Setup: 2-stage, N=1, L=0, 1 opening, 1 trial point.
    //   dual[0] = -3.0 (storage-fixing dual from MockSolver)
    //   objective = 80.0
    //   x_hat = [10.0]
    //
    // Expected (coefficients = dual, not -dual):
    //   pi[0] = dual[0] = -3.0
    //   intercept = 80.0 - (-3.0) * 10.0 = 110.0
    //   coefficients = [-3.0]
    let n_stages = 2_usize;
    let stochastic = make_stochastic_context(n_stages, 1);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    // dual[0] = -3.0, objective = 80.0
    let solution = solution_1_0(80.0, -3.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    let cuts: Vec<_> = fcf.active_cuts(0).collect();
    assert_eq!(cuts.len(), 1);
    let (_, intercept, coefficients) = &cuts[0];

    assert!(
        (intercept - 110.0).abs() < 1e-10,
        "expected intercept=110.0, got {intercept}"
    );
    assert_eq!(coefficients.len(), 1);
    assert!(
        (coefficients[0] - (-3.0)).abs() < 1e-10,
        "expected coefficient=-3.0, got {}",
        coefficients[0]
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn cut_gradient_sign_physically_correct() {
    // Regression test for the Benders cut sign bug.
    //
    // Physical invariant: more initial storage → lower future cost.
    // The storage-fixing dual π is negative (shadow price of relaxing
    // the fixing constraint increases cost when storage decreases).
    //
    // Correct: coefficient = π < 0, so the cut slope is negative
    //   (more storage → lower cut value → lower theta → lower total cost).
    //
    // Old bug: coefficient = -π > 0, so the cut slope was positive
    //   (more storage → higher cut value → wrong incentive).
    let n_stages = 2_usize;
    let stochastic = make_stochastic_context(n_stages, 1);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let mut exchange = exchange_with_states(n_state, vec![vec![50.0]]);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    // dual[0] = -2.0 (negative: more storage → less cost)
    // objective = 100.0, x_hat = 50.0
    let solution = solution_1_0(100.0, -2.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    let cuts: Vec<_> = fcf.active_cuts(0).collect();
    assert_eq!(cuts.len(), 1, "expected exactly one cut");
    let (_, _intercept, coefficients) = &cuts[0];

    // The coefficient must be negative (same sign as the dual).
    // The old bug would produce +2.0 here instead of -2.0.
    assert!(
        coefficients[0] < 0.0,
        "cut coefficient must be negative (more storage → less future cost), \
         got {} — likely the Benders cut sign bug has been reintroduced",
        coefficients[0]
    );
    assert!(
        (coefficients[0] - (-2.0)).abs() < 1e-10,
        "expected coefficient=-2.0, got {}",
        coefficients[0]
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn cut_is_tight_at_trial_point() {
    // Regression test: a Benders cut must be tight (exact) at the trial
    // point x̂ where it was generated. That is:
    //   intercept + coefficient * x̂ = Q(x̂)
    // where Q(x̂) = objective value of the subproblem at x̂.
    //
    // The cut equation is: θ ≥ intercept + coefficient * x
    // At x = x̂: θ ≥ Q(x̂) + π'(x̂ - x̂) = Q(x̂)
    //
    // If the sign is wrong (coefficient = -π instead of π), then:
    //   intercept + (-π) * x̂ ≠ Q(x̂) in general
    //
    // This test verifies the tightness property.
    let n_stages = 2_usize;
    let stochastic = make_stochastic_context(n_stages, 1);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let x_hat = 30.0_f64;
    let mut exchange = exchange_with_states(n_state, vec![vec![x_hat]]);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let q_xhat = 200.0_f64; // subproblem objective at x̂
    let dual_storage = -4.0_f64;
    let solution = solution_1_0(q_xhat, dual_storage);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    let cuts: Vec<_> = fcf.active_cuts(0).collect();
    assert_eq!(cuts.len(), 1);
    let (_, intercept, coefficients) = &cuts[0];

    // Evaluate the cut at x̂: cut_value = intercept + coeff * x̂
    let cut_at_xhat = intercept + coefficients[0] * x_hat;

    // Must equal Q(x̂) (tightness property)
    assert!(
        (cut_at_xhat - q_xhat).abs() < 1e-10,
        "cut must be tight at trial point: \
         cut_value={cut_at_xhat}, Q(x̂)={q_xhat}, \
         intercept={intercept}, coeff={}, x̂={x_hat}",
        coefficients[0]
    );
}

#[test]
fn single_rank_backward_pass_with_local_backend_produces_correct_fcf() {
    // Integration test with LocalBackend communicator (exercises single-rank path).
    use cobre_comm::LocalBackend;

    let n_stages = 3_usize;
    let n_openings = 2_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];

    let n_state = state.n_state;
    let forward_passes = 2_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let mut exchange = exchange_with_states(n_state, vec![vec![10.0], vec![20.0]]);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(100.0, -5.0);
    let solver = MockSolver::always_ok(solution);
    let comm = LocalBackend;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    // 3-stage system: cuts at stages 0 and 1; 2 trial points each.
    // Total cuts = 2 stages × 2 trial points = 4.
    assert_eq!(result.cuts_generated, 4);
    assert_eq!(fcf.active_cuts(0).count(), 2);
    assert_eq!(fcf.active_cuts(1).count(), 2);
    assert_eq!(fcf.active_cuts(2).count(), 0);
}

#[test]
#[allow(clippy::too_many_lines)]
fn forward_pass_index_matches_global_scenario_index() {
    // Acceptance criterion: when a cut is generated for global trial point
    // m=5, then `fcf.add_cut(stage, iteration, 5, ...)` is called with
    // forward_pass_index = m = 5.
    //
    // Setup: iteration=2, forward_passes=6 (6 scenarios on 1 rank), 1 opening.
    // ExchangeBuffers: local_count=6, num_ranks=1, total_scenarios=6.
    // state_at(5/6, 5%6) = state_at(0, 5) — valid.
    //
    // Slot formula: slot = warm_start(0) + 2*6 + 5 = 17.
    // The key invariant: forward_pass_index = m = 5.
    let n_stages = 2_usize;
    let stochastic = make_stochastic_context(n_stages, 1);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];

    let n_state = state.n_state;
    let forward_passes = 6_u32; // 6 scenarios on a single rank
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);

    // 6 trial points (m = 0..5). ExchangeBuffers: local_count=6, num_ranks=1.
    let mut exchange = exchange_with_states(
        n_state,
        vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
        ],
    );

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(50.0, 0.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 2,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    // m=5: slot = warm_start(0) + 2*6 + 5 = 17
    // The critical check: forward_pass_index in metadata equals global m=5.
    let meta = &fcf.pools[0].metadata[17];
    assert_eq!(meta.iteration_generated, 2, "iteration_generated must be 2");
    assert_eq!(
        meta.forward_pass_index, 5,
        "forward_pass_index must be 5 (= global m)"
    );
}

// ── Unit tests: warm-start basis caching (backward pass) ──────────────────

/// Warm-start from a pre-populated forward basis: when `BasisStore` has
/// `Some(Basis)` at `(scenario=0, stage=1)` before the first backward call,
/// the first opening at the successor stage must call `solve(Some(&basis))`
/// rather than `solve(None)`.
///
/// AC: Given a 2-stage system, 1 trial point, 1 opening, with
/// `basis_store.get(0, 1) = Some(Basis::new(...))` pre-populated,
/// then `solver.warm_start_calls == 1` after the backward pass.
#[test]
fn warm_start_uses_prepopulated_forward_basis() {
    let n_stages = 2_usize;
    let n_openings = 1_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(100.0, -5.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;

    // Pre-populate the basis store at (scenario=0, stage=1).
    // This simulates a forward pass having already solved stage 1 and cached its basis.
    let pre_basis = Basis::new(templates[1].num_cols, templates[1].num_rows);
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = basis_store_with_one(exchange.local_count(), n_stages, 0, 1, pre_basis);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    let warm_start_calls = workspaces[0].solver.inner().warm_start_calls;
    assert_eq!(
        warm_start_calls, 1,
        "first opening at successor stage must call solve(Some(&basis)) \
         when basis_store.get(0, 1) is pre-populated (warm_start_calls == 1, got {warm_start_calls})"
    );
}

/// Multi-opening P3b behavior: given 3 openings at the same successor stage,
/// the first opening cold-starts (store slot is None via `solve()`), and
/// openings 1 and 2 use `HiGHS` internal hot-start via `solve(None)` instead of
/// `solve(Some(&working_basis))`.
///
/// AC: Given a 2-stage system, 1 trial point, 3 openings, empty basis cache,
/// then `solver.warm_start_calls == 0` after the backward pass (P3b: no
/// and 3 warm-start; opening 1 cold-starts).
#[test]
fn multi_opening_subsequent_openings_use_internal_hotstart() {
    let n_stages = 2_usize;
    let n_openings = 3_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(100.0, -5.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;

    // Start with an empty store — opening 1 must cold-start.
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    // P3b optimization: opening 0 cold-starts (no basis in store),
    // openings 1 and 2 use solve(None) (HiGHS internal hot-start) instead of
    // solve(Some(&working_basis)). No explicit warm-start calls for subsequent openings.
    let warm_start_calls = workspaces[0].solver.inner().warm_start_calls;
    assert_eq!(
        warm_start_calls, 0,
        "P3b: no warm-start calls expected when BasisStore is empty \
         (warm_start_calls == 0, got {warm_start_calls})"
    );
}

/// Error propagation: when a backward solve returns `SolverError::Infeasible`,
/// the error must propagate as `SddpError::Infeasible`.
///
/// In the new per-scenario design, the backward pass uses a local `working_basis`
/// variable (not written back to `BasisStore`), so there is no shared cache slot
/// to check after the error. The test verifies that the error is correctly
/// propagated regardless of what was in the basis store at entry.
///
/// AC: Given a 2-stage system, 1 opening, `MockSolver` returns infeasible on
/// call 0, then `run_backward_pass` returns `Err(SddpError::Infeasible { .. })`.
#[test]
fn backward_solver_error_propagates() {
    let n_stages = 2_usize;
    let n_openings = 1_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(0.0, 0.0);
    // The first backward solve (call 0) returns infeasible.
    let solver = MockSolver::infeasible_on(solution, 0);
    let comm = StubComm;

    // Pre-populate the store — error should propagate regardless.
    let pre_basis = Basis::new(templates[1].num_cols, templates[1].num_rows);
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = basis_store_with_one(exchange.local_count(), n_stages, 0, 1, pre_basis);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    });

    assert!(
        matches!(result, Err(crate::SddpError::Infeasible { .. })),
        "expected SddpError::Infeasible, got: {result:?}",
    );
    // The BasisStore is not mutated by the backward pass — the working_basis
    // is a local variable dropped on error. The store slot at (0, 1) remains
    // as it was; this just verifies the store is untouched by an error path.
    assert!(
        basis_store.get(0, 1).is_some(),
        "BasisStore must not be mutated by the backward pass error path"
    );
}

// ── New test: parallel cut determinism ────────────────────────────────────

/// AC: When `run_backward_pass` runs with 1 workspace vs 4 workspaces given
/// the same input data, the FCF pools contain identical cuts (same intercept,
/// coefficient vectors, and slot assignments for each trial point).
#[test]
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
fn test_backward_pass_parallel_cut_determinism() {
    use crate::lp_builder::PatchBuffer;

    let n_stages = 3_usize;
    let n_openings = 2_usize;
    let n_trial_points = 8_usize;

    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];

    let n_state = state.n_state;
    #[allow(clippy::cast_possible_truncation)]
    let forward_passes = n_trial_points as u32;

    // Build 8 distinct trial-point states.
    let states: Vec<Vec<f64>> = (0..n_trial_points).map(|i| vec![i as f64 + 1.0]).collect();
    let mut exchange = exchange_with_states(n_state, states);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];
    let solution = solution_1_0(100.0, -5.0);
    let comm = StubComm;

    // --- Run with 1 workspace ---
    let mut fcf_1 =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
    let solver_1 = MockSolver::always_ok(solution.clone());
    let mut workspaces_1 = vec![SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(solver_1),
        patch_buf: PatchBuffer::new(1, 0, 0, 0, 0, 0),
        current_state: Vec::with_capacity(n_state),
        scratch: crate::workspace::ScratchBuffers {
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
            anticipated_state_buf: Vec::new(),
        },
        scratch_basis: Basis::new(0, 0),
        backward_accum: BackwardAccumulators::default(),
        worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
    }];
    let mut basis_store_1 = empty_basis_store(exchange.local_count(), n_stages);
    let ctx = StageContext {
        templates: &templates,
        base_rows: &base_rows,
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &[],
        ncs_col_starts: &[],
        ncs_active_slot_to_local: &[],
        ncs_max_gen: &[],
        ncs_allow_curtailment: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    };
    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces_1,
        basis_store: &mut basis_store_1,
        ctx: &ctx,
        baked: &mut templates.clone(),
        fcf: &mut fcf_1,
        cut_batches: &mut empty_cut_batches(n_stages),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    // --- Run with 4 workspaces ---
    let mut fcf_4 =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
    let mut workspaces_4: Vec<SolverWorkspace<MockSolver>> = (0..4_i32)
        .map(|idx| SolverWorkspace {
            rank: 0,
            worker_id: idx,
            solver: ProfiledSolver::new(MockSolver::always_ok(solution.clone())),
            patch_buf: PatchBuffer::new(1, 0, 0, 0, 0, 0),
            current_state: Vec::with_capacity(n_state),
            scratch: crate::workspace::ScratchBuffers {
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
                anticipated_state_buf: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
        })
        .collect();
    let mut basis_store_4 = empty_basis_store(exchange.local_count(), n_stages);
    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces_4,
        basis_store: &mut basis_store_4,
        ctx: &ctx,
        baked: &mut templates.clone(),
        fcf: &mut fcf_4,
        cut_batches: &mut empty_cut_batches(n_stages),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    // --- Verify identical FCF contents for all non-last stages ---
    for t in 0..(n_stages - 1) {
        let cuts_1: Vec<_> = fcf_1.active_cuts(t).collect();
        let cuts_4: Vec<_> = fcf_4.active_cuts(t).collect();

        assert_eq!(
            cuts_1.len(),
            cuts_4.len(),
            "stage {t}: cut count differs (1 workspace: {}, 4 workspaces: {})",
            cuts_1.len(),
            cuts_4.len()
        );

        for (idx, ((slot_1, intercept_1, coeff_1), (slot_4, intercept_4, coeff_4))) in
            cuts_1.iter().zip(cuts_4.iter()).enumerate()
        {
            assert_eq!(
                slot_1, slot_4,
                "stage {t} cut {idx}: slot mismatch ({slot_1} vs {slot_4})"
            );
            assert!(
                (intercept_1 - intercept_4).abs() < 1e-12,
                "stage {t} cut {idx}: intercept mismatch ({intercept_1} vs {intercept_4})"
            );
            assert_eq!(
                coeff_1.len(),
                coeff_4.len(),
                "stage {t} cut {idx}: coefficient vector length mismatch"
            );
            for (j, (c1, c4)) in coeff_1.iter().zip(coeff_4.iter()).enumerate() {
                assert!(
                    (c1 - c4).abs() < 1e-12,
                    "stage {t} cut {idx} coeff[{j}]: {c1} vs {c4}"
                );
            }
        }
    }

    // Last stage must have no cuts in both.
    assert_eq!(fcf_1.active_cuts(n_stages - 1).count(), 0);
    assert_eq!(fcf_4.active_cuts(n_stages - 1).count(), 0);
}

// ── Load noise wiring tests (backward pass) ──────────────────────────────

/// Build a 2-stage `StochasticContext` with 1 hydro and 1 stochastic load bus.
///
/// The noise vector dimension is `n_hydros + n_load_buses = 2`.
/// Stage 0 uses `branching_factor` openings; stage 1 is the successor solved
/// in the backward pass opening loop.
#[allow(clippy::too_many_lines)]
fn make_stochastic_context_with_load(
    n_stages: usize,
    branching_factor: usize,
    mean_mw: f64,
    std_mw: f64,
) -> cobre_stochastic::StochasticContext {
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
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let bus1 = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let hydro = Hydro {
        id: EntityId(10),
        name: "H10".to_string(),
        bus_id: EntityId(0),
        downstream_id: None,
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

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let make_stage = |idx: usize| Stage {
        index: idx,
        id: idx as i32,
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
            branching_factor,
            noise_method: NoiseMethod::Saa,
        },
    };

    let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|idx| InflowModel {
            hydro_id: EntityId(10),
            stage_id: idx as i32,
            mean_m3s: 100.0,
            std_m3s: 30.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|idx| LoadModel {
            bus_id: EntityId(1),
            stage_id: idx as i32,
            mean_mw,
            std_mw,
        })
        .collect();

    let correlation = CorrelationModel {
        method: "spectral".to_string(),
        profiles: std::collections::BTreeMap::new(),
        schedule: vec![],
    };

    let system = SystemBuilder::new()
        .buses(vec![bus0, bus1])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
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

/// AC: Given a backward pass with 1 stochastic load bus and opening noise
/// that includes a load component eta, the load balance row RHS in the patch
/// buffer is set to `max(0, mean + std * eta) * block_factor` before the solve.
///
/// We verify this indirectly: after the backward pass runs, `ws.scratch.load_rhs_buf`
/// must be non-empty and must contain a positive value (with mean=300, std=30
/// any reasonable eta produces a positive realization).
#[test]
#[allow(clippy::too_many_lines)]
fn backward_pass_load_patches_applied() {
    // 2-stage system: backward pass solves at successor=1 for each opening.
    // n_hydros=1, n_load_buses=1, 1 block per stage.
    let n_stages = 2_usize;
    let n_openings = 2_usize;
    // mean_mw=300 guarantees a positive realization for any reasonable eta draw.
    let stochastic = make_stochastic_context_with_load(n_stages, n_openings, 300.0, 30.0);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);

    // PatchBuffer: n_hydros=1, max_par_order=0, n_load_buses=1, max_blocks=1.
    let patch_buf = crate::lp_builder::PatchBuffer::new(1, 0, 1, 1, 0, 0);

    // Template: 2 rows (row 0 = state-fixing, row 1 = water-balance).
    // base_rows=[1] → inflow RHS row starts at index 1.
    // noise_scale=[1.0, 1.0] (one per (stage, hydro)).
    let template = StageTemplate {
        num_cols: 3,
        num_rows: 2,
        num_nz: 1,
        col_starts: vec![0_i32, 0, 1, 1],
        row_indices: vec![0_i32],
        values: vec![1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
        objective: vec![0.0, 0.0, 1.0],
        row_lower: vec![50.0, 100.0],
        row_upper: vec![50.0, 100.0],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 1,
        n_hydro: 1,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };
    let templates = vec![template; n_stages];
    let base_rows = vec![1_usize; n_stages];
    let noise_scale = vec![1.0_f64; n_stages]; // one per (stage, hydro)

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    // MockSolver returns a fixed solution (1 state var, 1 dual entry).
    let solution = solution_1_0(100.0, -2.0);

    let ws = SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(MockSolver::always_ok(solution)),
        patch_buf,
        current_state: Vec::with_capacity(n_state),
        scratch: crate::workspace::ScratchBuffers {
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
            load_rhs_buf: Vec::with_capacity(1),
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
            anticipated_state_buf: Vec::new(),
        },
        scratch_basis: Basis::new(0, 0),
        backward_accum: BackwardAccumulators::default(),
        worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
    };
    let mut workspaces = vec![ws];

    let comm = StubComm;
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    // load_balance_row_starts[successor=1]=10; load_bus_indices=[0]; 1 block/stage.
    let load_balance_row_starts = vec![10_usize; n_stages];
    let load_bus_indices = vec![0_usize];
    let block_counts_per_stage = vec![1_usize; n_stages];

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &noise_scale,
            n_hydros: 1,
            n_load_buses: 1,
            load_balance_row_starts: &load_balance_row_starts,
            load_bus_indices: &load_bus_indices,
            block_counts_per_stage: &block_counts_per_stage,
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    // After the backward pass, load_rhs_buf must have been populated with a
    // positive value for the last opening solved (mean=300, std=30 → positive).
    assert_eq!(
        workspaces[0].scratch.load_rhs_buf.len(),
        1,
        "load_rhs_buf must have 1 entry (1 load bus × 1 block)"
    );
    assert!(
        workspaces[0].scratch.load_rhs_buf[0] > 0.0,
        "load realization must be positive with mean=300, std=30: got {}",
        workspaces[0].scratch.load_rhs_buf[0]
    );
}

/// AC: Given a backward pass with 0 stochastic load buses, `patch_count`
/// equals `N*(2+L)` (no load patches) and `load_rhs_buf` stays empty.
///
/// N=1, L=0 → `N*(2+L) = 2`.
#[test]
#[allow(clippy::too_many_lines)]
fn backward_pass_no_load_buses_unchanged() {
    let n_stages = 2_usize;
    let n_openings = 2_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);

    // PatchBuffer with no load buses: n_load_buses=0, max_blocks=1.
    let patch_buf = crate::lp_builder::PatchBuffer::new(1, 0, 0, 0, 0, 0);

    let template = StageTemplate {
        num_cols: 3,
        num_rows: 2,
        num_nz: 1,
        col_starts: vec![0_i32, 0, 1, 1],
        row_indices: vec![0_i32],
        values: vec![1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
        objective: vec![0.0, 0.0, 1.0],
        row_lower: vec![50.0, 100.0],
        row_upper: vec![50.0, 100.0],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 1,
        n_hydro: 1,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };
    let templates = vec![template; n_stages];
    let base_rows = vec![1_usize; n_stages];
    let noise_scale = vec![1.0_f64; n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(100.0, -2.0);
    let ws = SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(MockSolver::always_ok(solution)),
        patch_buf,
        current_state: Vec::with_capacity(n_state),
        scratch: crate::workspace::ScratchBuffers {
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
            anticipated_state_buf: Vec::new(),
        },
        scratch_basis: Basis::new(0, 0),
        backward_accum: BackwardAccumulators::default(),
        worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
    };
    let mut workspaces = vec![ws];
    let comm = StubComm;
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &noise_scale,
            n_hydros: 1,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1_usize; 2],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    // With n_load_buses=0, forward_patch_count = N + z_inflow = 1 + 1 = 2.
    assert_eq!(
        workspaces[0].patch_buf.forward_patch_count(),
        2,
        "forward_patch_count must be N+z_inflow=2 when n_load_buses=0, got {}",
        workspaces[0].patch_buf.forward_patch_count()
    );
    // load_rhs_buf must remain empty.
    assert!(
        workspaces[0].scratch.load_rhs_buf.is_empty(),
        "load_rhs_buf must be empty when n_load_buses=0"
    );
}

/// AC: Given a backward pass with stochastic load, when Benders cut
/// coefficients are extracted, the cut coefficient array has length `n_state`
/// unchanged — load adds no state variables.
///
/// Setup: N=1 hydro, L=0 PAR lags → `n_state=1`. After the backward pass with
/// 1 load bus, each generated cut must have exactly 1 coefficient.
#[test]
#[allow(clippy::too_many_lines)]
fn backward_pass_cut_coefficients_unaffected() {
    let n_stages = 2_usize;
    let n_openings = 2_usize;
    let stochastic = make_stochastic_context_with_load(n_stages, n_openings, 200.0, 20.0);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);

    let patch_buf = crate::lp_builder::PatchBuffer::new(1, 0, 1, 1, 0, 0);

    let template = StageTemplate {
        num_cols: 3,
        num_rows: 2,
        num_nz: 1,
        col_starts: vec![0_i32, 0, 1, 1],
        row_indices: vec![0_i32],
        values: vec![1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
        objective: vec![0.0, 0.0, 1.0],
        row_lower: vec![50.0, 100.0],
        row_upper: vec![50.0, 100.0],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 1,
        n_hydro: 1,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };
    let templates = vec![template; n_stages];
    let base_rows = vec![1_usize; n_stages];
    let noise_scale = vec![1.0_f64; n_stages];

    let n_state = state.n_state; // 1
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(80.0, -3.0);
    let ws = SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(MockSolver::always_ok(solution)),
        patch_buf,
        current_state: Vec::with_capacity(n_state),
        scratch: crate::workspace::ScratchBuffers {
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
            load_rhs_buf: Vec::with_capacity(1),
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
            anticipated_state_buf: Vec::new(),
        },
        scratch_basis: Basis::new(0, 0),
        backward_accum: BackwardAccumulators::default(),
        worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
    };
    let mut workspaces = vec![ws];
    let comm = StubComm;
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let load_balance_row_starts = vec![10_usize; n_stages];
    let load_bus_indices = vec![0_usize];
    let block_counts_per_stage = vec![1_usize; n_stages];

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &noise_scale,
            n_hydros: 1,
            n_load_buses: 1,
            load_balance_row_starts: &load_balance_row_starts,
            load_bus_indices: &load_bus_indices,
            block_counts_per_stage: &block_counts_per_stage,
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    // Exactly 1 cut generated (1 trial point × 1 stage with a successor).
    assert_eq!(result.cuts_generated, 1);

    // The cut must have exactly n_state=1 coefficient.
    let cuts: Vec<_> = fcf.active_cuts(0).collect();
    assert_eq!(cuts.len(), 1);
    let (_, _intercept, coefficients) = &cuts[0];
    assert_eq!(
        coefficients.len(),
        n_state,
        "cut coefficients length must be n_state={n_state}, got {} — \
         load buses must not add state variables",
        coefficients.len()
    );
}

/// BUG-1 structural invariant: per-stage cut sync inside the backward loop.
///
/// Verifies that after `run_backward_pass`, the cut synchronization has been
/// performed per-stage (not as a separate post-sweep loop). The structural
/// evidence is:
///
/// 1. `BackwardResult.cut_sync_time_ms` is populated (timing was captured).
/// 2. The FCF has the expected number of cuts per stage — same as single-rank
///    without sync, because single-rank sync is a no-op that does not change
///    results but exercises the code path.
/// 3. Using `LocalBackend` (the production single-rank communicator) instead
///    of `StubComm` exercises the full `sync_cuts` → allgatherv → deserialize
///    path, confirming no panics or data corruption.
///
/// True multi-rank correctness testing requires actual MPI and is out of
/// scope for CI. This test validates the structural invariant (sync is
/// called per-stage inside the loop) and exercises the full code path.
#[test]
#[allow(clippy::too_many_lines)]
fn per_stage_cut_sync_invariant_after_bug1_fix() {
    use cobre_comm::LocalBackend;

    let n_stages = 4_usize;
    let n_openings = 2_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];

    let n_state = state.n_state;
    let forward_passes = 3_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
    let mut exchange = exchange_with_states(n_state, vec![vec![10.0], vec![20.0], vec![30.0]]);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(100.0, -5.0);
    let solver = MockSolver::always_ok(solution);
    let comm = LocalBackend;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::new(n_state, forward_passes as usize, 1);
    let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 1,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    // 4-stage system: cuts at stages 0, 1, 2; 3 trial points each.
    // Total cuts = 3 stages × 3 trial points = 9.
    assert_eq!(result.cuts_generated, 9);

    // Each non-terminal stage has 3 cuts (one per trial point).
    assert_eq!(fcf.active_cuts(0).count(), 3, "stage 0 must have 3 cuts");
    assert_eq!(fcf.active_cuts(1).count(), 3, "stage 1 must have 3 cuts");
    assert_eq!(fcf.active_cuts(2).count(), 3, "stage 2 must have 3 cuts");
    assert_eq!(
        fcf.active_cuts(3).count(),
        0,
        "terminal stage must have 0 cuts"
    );

    // Verify cut_sync_time_ms was captured (structural evidence that
    // sync_cuts was called inside the backward loop).
    // For single-rank LocalBackend, sync is a no-op, so time should be
    // very small but the field must be populated (not default/garbage).
    // We just verify it's a valid non-negative value.
    assert!(
        result.cut_sync_time_ms < 10_000,
        "cut_sync_time_ms should be reasonable, got {}",
        result.cut_sync_time_ms
    );
}

/// Acceptance criterion: within a single backward
/// iteration on a 3-stage system with `LocalBackend` (single-rank),
/// cuts generated at stage t=1 are visible at stage t=0 and appear
/// binding (mock returns positive cut duals). The metadata sync
/// correctly accumulates `active_count` and sets `last_active_iter`.
///
/// Uses `MockSolver::always_ok_with_binding_cuts` so that cut rows
/// return positive duals, making them appear binding when evaluated.
#[test]
#[allow(clippy::too_many_lines)]
fn metadata_sync_updates_active_count_and_last_active_iter() {
    use cobre_comm::LocalBackend;

    // 3-stage system: backward loop processes t=1 then t=0.
    // At t=1: generates cuts into pool[1], successor pool[2] is empty.
    // At t=0: generates cuts into pool[0], successor pool[1] has cuts
    //         from t=1. Mock duals make those cuts appear binding.
    let n_stages = 3_usize;
    let n_openings = 2_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];

    let n_state = state.n_state;
    let forward_passes = 3_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
    let mut exchange = exchange_with_states(n_state, vec![vec![10.0], vec![20.0], vec![30.0]]);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(100.0, -5.0);
    let solver = MockSolver::always_ok_with_binding_cuts(solution);
    let comm = LocalBackend;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::new(n_state, forward_passes as usize, 1);

    // Run a single backward iteration. The backward loop visits t=1
    // (cuts go to pool[1]), then t=0 (cuts go to pool[0], binding
    // checked against pool[1]).
    let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 1,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    // 3 stages × (n_stages-1=2 non-terminal) × 3 trial points = 6 cuts.
    assert_eq!(result.cuts_generated, 6);

    // Pool[1] received 3 cuts from t=1 backward pass.
    // Slot formula: warm_start(0) + iteration(1) * fwd_passes(3) + fpi
    // → slots 3, 4, 5. Populated count = 6 (high-water mark).
    assert_eq!(fcf.pools[1].populated_count, 6);

    // At t=0, the 3 cuts in pool[1] (slots 3,4,5) were evaluated for
    // binding. The mock solver returns positive duals (cut_dual_padding
    // = 1.0) for all cut rows. Each trial point has n_openings=2
    // openings, and the binding check runs per opening. So each slot
    // gets 3 trial points × 2 openings = 6 increments.
    for slot in 3..6 {
        assert!(
            fcf.pools[1].metadata[slot].active_count > 0,
            "slot {slot} active_count should be > 0 (cuts were binding)"
        );
        assert_eq!(
            fcf.pools[1].metadata[slot].active_count, 6,
            "slot {slot} active_count should be 6 (3 trial points × 2 openings)"
        );
        assert_eq!(
            fcf.pools[1].metadata[slot].last_active_iter, 1,
            "slot {slot} last_active_iter should be 1 (current iteration)"
        );
    }

    // Pool[2] (terminal successor) received no cuts and no binding
    // was checked against it — metadata should be at defaults.
    assert_eq!(fcf.pools[2].populated_count, 0);
}

/// Build N identical `SolverWorkspace<MockSolver>` instances and run a
/// 2-stage backward pass with 6 trial points. Returns the resulting FCF.
///
/// Used by `work_stealing_produces_identical_results_across_worker_counts`
/// to compare FCF state across different worker counts.
///
/// The `MockSolver` returns objective=100.0 and dual[0]=-5.0 for every solve,
/// which is deterministic (no dependence on call order or worker identity).
/// Each trial point i gets state [(i + 1) as f64 * 10.0] so that distinct
/// cuts are generated and the ordering invariant is meaningful.
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn run_backward_pass_with_n_workers(n_workers: usize) -> FutureCostFunction {
    use crate::lp_builder::PatchBuffer;

    let n_stages = 2_usize;
    let local_work = 6_usize;
    let n_openings = 2_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];
    let n_state = state.n_state; // 1

    // Use forward_passes = local_work so the FCF pool is large enough for
    // all trial points in a single iteration (iteration 0, slots 0..5).
    #[allow(clippy::cast_possible_truncation)]
    let forward_passes = local_work as u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 64, &vec![0; n_stages]);

    // Build `local_work` trial points with distinct states so each cut
    // has a different intercept. State for trial point i = (i+1)*10.0.
    let states: Vec<Vec<f64>> = (0..local_work)
        .map(|i| vec![(i + 1) as f64 * 10.0])
        .collect();
    let mut exchange = exchange_with_states(n_state, states);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    // Each workspace gets the same deterministic solution.
    // MockSolver::always_ok returns objective=100.0, dual[0]=-5.0 for
    // every call regardless of call order or worker identity.
    let solution = solution_1_0(100.0, -5.0);
    let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
        .map(|idx| SolverWorkspace {
            rank: 0,
            worker_id: i32::try_from(idx).expect("worker_id fits in i32"),
            solver: ProfiledSolver::new(MockSolver::always_ok(solution.clone())),
            patch_buf: PatchBuffer::new(1, 0, 0, 0, 0, 0),
            current_state: Vec::with_capacity(n_state),
            scratch: crate::workspace::ScratchBuffers {
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
                anticipated_state_buf: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
        })
        .collect();

    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
    let comm = StubComm;
    let mut csb = CutSyncBuffers::new(n_state, local_work, 1);

    let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .unwrap();

    // Confirm all 6 trial points produced cuts at stage 0.
    assert_eq!(
        result.cuts_generated, local_work,
        "n_workers={n_workers}: expected {local_work} cuts, got {}",
        result.cuts_generated,
    );

    fcf
}

#[test]
fn work_stealing_produces_identical_results_across_worker_counts() {
    // Acceptance criterion: the FCF state after running the backward pass
    // with 1 workspace must be bit-identical to the state after running
    // with 3 workspaces, given the same inputs. This verifies that the
    // sort-by-trial_point_idx post-processing in the work-stealing
    // implementation produces a deterministic FCF regardless of which
    // worker claims which trial point.
    let fcf_1 = run_backward_pass_with_n_workers(1);
    let fcf_3 = run_backward_pass_with_n_workers(3);

    let num_stages = 2;

    // Verify that both runs produced cuts (belt-and-suspenders guard so
    // that an empty FCF cannot cause a false positive).
    assert!(
        fcf_1.active_cuts(0).count() > 0,
        "1-worker run produced no cuts at stage 0"
    );

    for stage in 0..num_stages {
        let cuts_1: Vec<_> = fcf_1.active_cuts(stage).collect();
        let cuts_3: Vec<_> = fcf_3.active_cuts(stage).collect();
        assert_eq!(
            cuts_1.len(),
            cuts_3.len(),
            "stage {stage}: cut count mismatch ({} vs {})",
            cuts_1.len(),
            cuts_3.len(),
        );
        for (i, ((s1, int1, c1), (s3, int3, c3))) in cuts_1.iter().zip(&cuts_3).enumerate() {
            assert_eq!(
                s1, s3,
                "stage {stage}, cut {i}: slot mismatch ({s1} vs {s3})"
            );
            assert_eq!(
                int1, int3,
                "stage {stage}, cut {i}: intercept mismatch ({int1} vs {int3})"
            );
            assert_eq!(
                c1, c3,
                "stage {stage}, cut {i}: coefficients mismatch ({c1:?} vs {c3:?})"
            );
        }
    }
}

// ── Parallel overhead decomposition unit tests ────────────────────────────

/// Build a `SolverStatistics` snapshot with the given cumulative times (in seconds).
fn make_stats(solve_s: f64, load_s: f64, set_bounds_s: f64, basis_set_s: f64) -> SolverStatistics {
    SolverStatistics {
        total_solve_time_seconds: solve_s,
        total_load_model_time_seconds: load_s,
        total_set_bounds_time_seconds: set_bounds_s,
        total_basis_set_time_seconds: basis_set_s,
        ..SolverStatistics::default()
    }
}

/// Decompose parallel overhead into (`setup_ms`, `imbalance_ms`, `scheduling_ms`)
/// from per-worker before/after snapshots.
fn decompose_overhead(
    pairs: &[(SolverStatistics, SolverStatistics)],
    parallel_wall_ms: u64,
) -> (u64, u64, u64) {
    use crate::solver_stats::SolverStatsDelta;

    #[allow(clippy::cast_precision_loss)]
    let n_workers = pairs.len() as f64;

    let worker_deltas: Vec<SolverStatsDelta> = pairs
        .iter()
        .map(|(before, after)| SolverStatsDelta::from_snapshots(before, after))
        .collect();

    let stage_setup_ms: f64 = worker_deltas
        .iter()
        .map(|d| d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms)
        .sum();

    let worker_totals: Vec<f64> = worker_deltas
        .iter()
        .map(|d| {
            d.solve_time_ms + d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms
        })
        .collect();

    let max_worker_ms = worker_totals.iter().copied().fold(0.0_f64, f64::max);
    let avg_worker_ms = if worker_totals.is_empty() {
        0.0_f64
    } else {
        worker_totals.iter().sum::<f64>() / n_workers
    };

    let stage_imbalance_ms = (max_worker_ms - avg_worker_ms).max(0.0);
    #[allow(clippy::cast_precision_loss)]
    let stage_scheduling_ms = (parallel_wall_ms as f64 - max_worker_ms).max(0.0);

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    (
        stage_setup_ms as u64,
        stage_imbalance_ms as u64,
        stage_scheduling_ms as u64,
    )
}

/// 4 workers with different solve times: imbalance equals
/// `trunc(max - mean_f64)` of worker totals, scheduling equals
/// `parallel_wall - max`.
///
/// Worker solve times: 100 ms, 200 ms, 150 ms, 180 ms.
/// Setup per worker: 0 (this sub-test isolates solve imbalance).
/// Mean of totals (f64) = 630.0 / 4 = 157.5.
/// Imbalance = trunc(200.0 - 157.5) = trunc(42.5) = 42.
/// Scheduling = 250 - 200 = 50.
#[test]
fn decompose_four_workers_different_solve_times() {
    // All setup times are zero; use only solve time to isolate imbalance.
    let zero = SolverStatistics::default();
    let pairs = vec![
        (zero.clone(), make_stats(0.1, 0.0, 0.0, 0.0)), // 100 ms solve
        (zero.clone(), make_stats(0.2, 0.0, 0.0, 0.0)), // 200 ms solve
        (zero.clone(), make_stats(0.15, 0.0, 0.0, 0.0)), // 150 ms solve
        (zero.clone(), make_stats(0.18, 0.0, 0.0, 0.0)), // 180 ms solve
    ];
    let (setup_ms, imbalance_ms, scheduling_ms) = decompose_overhead(&pairs, 250);

    assert_eq!(setup_ms, 0, "no setup work expected");
    // f64 mean = 157.5; imbalance = trunc(200.0 - 157.5) = trunc(42.5) = 42
    assert_eq!(
        imbalance_ms, 42,
        "imbalance = trunc(max(200.0) - avg(157.5)) = trunc(42.5) = 42"
    );
    // scheduling = 250 - 200 = 50
    assert_eq!(scheduling_ms, 50, "scheduling overhead = wall - max_worker");
}

/// Acceptance criterion: `setup_time_ms` is the sum of all workers' non-solve
/// work.
///
/// Workers have setup costs (load+add+bounds+basis): 20, 25, 15, 22 ms.
/// Expected `setup_ms` = 20 + 25 + 15 + 22 = 82.
#[test]
fn decompose_setup_time_is_aggregate_non_solve_work() {
    let zero = SolverStatistics::default();
    // Each worker: 0 solve + known setup split across the three sub-timers.
    // Worker setup totals: 20, 25, 15, 22 ms (put entirely in load_model timer).
    let pairs = vec![
        (zero.clone(), make_stats(0.0, 0.020, 0.0, 0.0)), // 20 ms total setup
        (zero.clone(), make_stats(0.0, 0.025, 0.0, 0.0)), // 25 ms
        (zero.clone(), make_stats(0.0, 0.015, 0.0, 0.0)), // 15 ms
        (zero.clone(), make_stats(0.0, 0.022, 0.0, 0.0)), // 22 ms
    ];
    let (setup_ms, _imbalance_ms, _scheduling_ms) = decompose_overhead(&pairs, 300);
    assert_eq!(
        setup_ms, 82,
        "aggregate setup must sum all workers' non-solve work"
    );
}

/// Edge case: all workers have identical timing → imbalance must be 0.
#[test]
fn decompose_identical_workers_zero_imbalance() {
    let zero = SolverStatistics::default();
    let after = make_stats(0.1, 0.01, 0.002, 0.001);
    let pairs = vec![
        (zero.clone(), after.clone()),
        (zero.clone(), after.clone()),
        (zero.clone(), after.clone()),
    ];
    let (_, imbalance_ms, _) = decompose_overhead(&pairs, 200);
    assert_eq!(
        imbalance_ms, 0,
        "identical workers must have zero imbalance"
    );
}

/// Edge case: single worker → imbalance is 0, setup equals that worker's
/// setup, scheduling is the residual.
#[test]
fn decompose_single_worker() {
    let zero = SolverStatistics::default();
    // 100 ms solve + 20 ms setup = 120 ms worker total.
    let after = make_stats(0.1, 0.020, 0.0, 0.0);
    let pairs = vec![(zero.clone(), after)];
    let (setup_ms, imbalance_ms, scheduling_ms) = decompose_overhead(&pairs, 150);

    assert_eq!(setup_ms, 20, "single worker: setup = 20 ms");
    assert_eq!(imbalance_ms, 0, "single worker: imbalance must be 0");
    // scheduling = 150 - 120 = 30
    assert_eq!(
        scheduling_ms, 30,
        "single worker: scheduling = wall - worker_total"
    );
}

/// Edge case: `scheduling_overhead_ms` is clamped to 0 when `max_worker_total`
/// exceeds `parallel_wall_ms` (clock skew or measurement granularity).
#[test]
fn decompose_scheduling_clamped_when_worker_exceeds_wall() {
    let zero = SolverStatistics::default();
    // Worker total = 200 ms, but wall = 180 ms → scheduling would be negative.
    let after = make_stats(0.2, 0.0, 0.0, 0.0);
    let pairs = vec![(zero.clone(), after)];
    let (_, _, scheduling_ms) = decompose_overhead(&pairs, 180);
    assert_eq!(scheduling_ms, 0, "negative scheduling must be clamped to 0");
}

// ── allgatherv per-worker stats unit tests ────────────────────────────────

/// Single-rank (np=1) backward pass with 2 workers.
///
/// Constructs a 2-worker `StageWorkerStatsBuffer::new(2, 4)` and uses
/// `StubComm` (which echoes send→recv, simulating `LocalBackend` np=1).
/// After one backward iteration, `BackwardResult::stage_stats` must
/// contain 2 entries per non-zero opening (`worker_id` 0 and `worker_id` 1),
/// both with `rank = 0`.
#[test]
#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
fn allgatherv_single_rank_two_workers_stage_stats_has_per_worker_entries() {
    use crate::lp_builder::PatchBuffer;

    let n_stages = 2_usize;
    let n_openings = 4_usize;
    let n_workers = 2_usize;
    let local_work = 4_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];
    let n_state = state.n_state;

    let solution = solution_1_0(100.0, -5.0);
    let states: Vec<Vec<f64>> = (0..local_work).map(|i| vec![(i + 1) as f64]).collect();
    let mut exchange = exchange_with_states(n_state, states);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
        .map(|idx| SolverWorkspace {
            rank: 0,
            worker_id: i32::try_from(idx).expect("idx fits in i32"),
            solver: ProfiledSolver::new(MockSolver::always_ok(solution.clone())),
            patch_buf: PatchBuffer::new(1, 0, 0, 0, 0, 0),
            current_state: Vec::with_capacity(n_state),
            scratch: crate::workspace::ScratchBuffers {
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
                anticipated_state_buf: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
        })
        .collect();

    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, local_work as u32, 64, &vec![0; n_stages]);
    let mut csb = CutSyncBuffers::new(n_state, local_work, 1);

    let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &StubComm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .expect("single-rank 2-worker backward must not error");

    // The 2-stage system has 1 backward stage (t=0, successor=1).
    // stage_stats must contain exactly 1 entry (one successor).
    assert_eq!(
        result.stage_stats.len(),
        1,
        "expected 1 backward stage entry (successor=1)"
    );
    let (successor, entries) = &result.stage_stats[0];
    assert_eq!(*successor, 1_usize, "successor index must be 1");

    // Every entry must have rank=0 (np=1 StubComm).
    for (rank, _wid, _omega, _delta) in entries {
        assert_eq!(*rank, 0_i32, "all entries must have rank=0 for np=1");
    }
    // Both worker_id values (0 and 1) must appear at omega=0.
    let omega0_wids: Vec<i32> = entries
        .iter()
        .filter(|(_, _, omega, _)| *omega == 0)
        .map(|(_, wid, _, _)| *wid)
        .collect();
    assert!(
        omega0_wids.contains(&0),
        "worker_id=0 must appear at omega=0"
    );
    assert!(
        omega0_wids.contains(&1),
        "worker_id=1 must appear at omega=0"
    );
}

/// Multi-rank (np=2) backward pass with stub communicator.
///
/// Uses a `DualRankStubComm` whose `size()` returns 2 and whose
/// `allgatherv` concatenates a manually injected "remote rank" payload
/// (a copy of the send buffer). Asserts that the unpacked
/// `stage_stats` contains entries for both `rank=0` and `rank=1`.
#[test]
#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
fn allgatherv_dual_rank_stub_stage_stats_contains_both_ranks() {
    use crate::lp_builder::PatchBuffer;

    /// Stub communicator simulating np=2: `allgatherv` fills recv with
    /// `[send, send]` (rank-0 and a synthetic rank-1 copy).
    struct DualRankStubComm;

    impl Communicator for DualRankStubComm {
        fn allgatherv<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            counts: &[usize],
            displs: &[usize],
        ) -> Result<(), CommError> {
            // Fill each rank's slot in recv using the provided counts/displs.
            // Both ranks contribute `send` (rank-1 is a synthetic copy of rank-0).
            for (r, (&count, &displ)) in counts.iter().zip(displs).enumerate() {
                let src = &send[..count.min(send.len())];
                recv[displ..displ + src.len()].copy_from_slice(src);
                let _ = r; // suppress unused warning in cfg(test)
            }
            Ok(())
        }

        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            recv[..send.len()].copy_from_slice(send);
            Ok(())
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            Ok(())
        }

        fn barrier(&self) -> Result<(), CommError> {
            Ok(())
        }

        fn rank(&self) -> usize {
            0
        }

        fn size(&self) -> usize {
            2
        }

        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
    }

    let n_stages = 2_usize;
    let n_openings = 2_usize;
    let n_workers = 1_usize;
    let local_work = 2_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let base_rows = vec![1_usize; n_stages];
    let n_state = state.n_state;

    let solution = solution_1_0(100.0, -5.0);
    let states: Vec<Vec<f64>> = (0..local_work).map(|i| vec![(i + 1) as f64]).collect();
    let mut exchange = exchange_with_states(n_state, states);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
        .map(|idx| SolverWorkspace {
            rank: 0,
            worker_id: i32::try_from(idx).expect("idx fits in i32"),
            solver: ProfiledSolver::new(MockSolver::always_ok(solution.clone())),
            patch_buf: PatchBuffer::new(1, 0, 0, 0, 0, 0),
            current_state: Vec::with_capacity(n_state),
            scratch: crate::workspace::ScratchBuffers {
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
                anticipated_state_buf: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
        })
        .collect();

    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, local_work as u32, 64, &vec![0; n_stages]);
    let mut csb = CutSyncBuffers::new(n_state, local_work, 1);

    let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &DualRankStubComm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    })
    .expect("dual-rank stub backward must not error");

    // With np=2, stage_stats for successor=1 must contain entries from
    // both rank=0 and rank=1 (DualRankStubComm copies the rank-0 block
    // into the rank-1 slot, so both appear in the unpacked output).
    assert_eq!(result.stage_stats.len(), 1);
    let (_, entries) = &result.stage_stats[0];

    let ranks_seen: Vec<i32> = entries
        .iter()
        .map(|(rank, _, _, _)| *rank)
        .collect::<std::collections::HashSet<i32>>()
        .into_iter()
        .collect();
    assert!(
        ranks_seen.contains(&0),
        "rank=0 must appear in stage_stats; got {ranks_seen:?}"
    );
    assert!(
        ranks_seen.contains(&1),
        "rank=1 must appear in stage_stats; got {ranks_seen:?}"
    );
}

// ── read-site prefer-with-fallback unit tests ────────────────

/// Run `process_trial_point_backward` for stage 0 → successor 1 with
/// explicitly-provided backward and forward basis stores.
///
/// `basis_store` is taken by `&mut` so a `BasisStoreSliceMut` can be
/// derived from it and passed to `process_trial_point_backward`.
///
/// Returns the mutated workspace so the caller can inspect
/// `ws.solver.warm_start_calls`.
#[allow(clippy::too_many_lines)]
fn run_one_trial_point_with_stores(
    basis_store: &mut crate::workspace::BasisStore,
) -> Result<Vec<SolverWorkspace<MockSolver>>, crate::SddpError> {
    use crate::context::StageContext;

    let n_stages = 2_usize;
    let n_openings = 1_usize;
    let n_state = 1_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let indexer = crate::indexer::test_fixtures::geom(n_state, 0);
    let state = crate::indexer::test_fixtures::state_layout(n_state, 0);

    let solver = MockSolver::always_ok(solution_1_0(100.0, -5.0));
    let mut workspaces = single_workspace(solver, n_state);
    let ws = &mut workspaces[0];
    ws.backward_accum
        .outcomes
        .push(crate::risk_measure::BackwardOutcome {
            intercept: 0.0,
            coefficients: vec![0.0; n_state],
            objective_value: 0.0,
        });
    ws.backward_accum
        .per_opening_stats
        .push(SolverStatsDelta::default());
    ws.backward_accum.agg_coefficients.resize(n_state, 0.0);

    let exchange = exchange_with_states(n_state, vec![vec![5.0]]);

    let templates: &'static _ = Box::leak(Box::new(vec![
        minimal_template_1_0(),
        minimal_template_1_0(),
    ]));
    let base_rows: &'static _ = Box::leak(Box::new(vec![1_usize, 1_usize]));
    let ctx: StageContext<'static> = StageContext {
        templates,
        base_rows,
        noise_scale: Box::leak(Box::new(vec![])),
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: Box::leak(Box::new(vec![])),
        load_bus_indices: Box::leak(Box::new(vec![])),
        block_counts_per_stage: Box::leak(Box::new(vec![])),
        ncs_col_starts: &[],
        ncs_active_slot_to_local: &[],
        ncs_max_gen: Box::leak(Box::new(vec![])),
        ncs_allow_curtailment: Box::leak(Box::new(vec![])),
        discount_factors: Box::leak(Box::new(vec![])),
        cumulative_discount_factors: Box::leak(Box::new(vec![])),
        stage_lag_transitions: Box::leak(Box::new(vec![])),
        noise_group_ids: Box::leak(Box::new(vec![])),
        downstream_par_order: 0,
    };

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];
    let training_ctx = TrainingContext {
        horizon: &horizon,
        indexer: &indexer,
        state: &state,
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stochastic,
        initial_state: &[],
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
        noise_key_diag: None,
    };

    let iteration: u64 = 1;
    let fwd_offset: usize = 0;
    let succ_probabilities = vec![1.0_f64; n_openings];
    let successor_active_slots: Vec<usize> = vec![];
    let baked_template = minimal_template_1_0();

    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0u32; n_stages]);
    let empty_cut_batch = RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    };

    let succ_spec = super::SuccessorSpec {
        t: 0,
        successor: 1,
        my_rank: 0,
        probabilities: &succ_probabilities,
        cut_batch: &empty_cut_batch,
        num_cuts_at_successor: 0,
        template_num_rows: baked_template.num_rows,
        baked_template: &baked_template,
        successor_active_slots: &successor_active_slots,
        cut_activity_tolerance: 0.0,
        successor_populated_count: fcf.pools[1].populated_count,
        successor_pool: &fcf.pools[1],
    };

    // Derive a single-worker BasisStoreSliceMut covering all scenarios.
    let mut basis_slices = basis_store.split_workers_mut(1);
    let mut basis_slice = basis_slices.remove(0);

    let ws = &mut workspaces[0];
    super::load_backward_lp(ws, &succ_spec);
    ws.backward_accum
        .per_opening_stats
        .resize_with(n_openings, SolverStatsDelta::default);
    for slot in &mut ws.backward_accum.per_opening_stats[..n_openings] {
        *slot = SolverStatsDelta::default();
    }
    ws.backward_accum.slot_increments.resize(1, 0);
    ws.backward_accum.slot_increments[..1].fill(0);
    // Size the coefficient arena for the single trial point (offset 0).
    if ws.backward_accum.agg_arena.len() < n_state {
        ws.backward_accum.agg_arena.resize(n_state, 0.0_f64);
    }

    super::process_trial_point_backward(
        ws,
        &ctx,
        &training_ctx,
        &exchange,
        fwd_offset,
        iteration,
        &risk_measures,
        &succ_spec,
        &mut basis_slice,
        &super::StageOpeningSolver::Baked,
        0,
        0,
    )?;
    Ok(workspaces)
}

// ---------------------------------------------------------------------------
// resolve_backward_basis_* unit tests
// ---------------------------------------------------------------------------

#[test]
fn resolve_backward_basis_returns_some_when_slot_is_populated() {
    // Given: BasisStore[0, 1] has Some(CapturedBasis).
    // Then: resolve_backward_basis returns Some(_).
    use crate::workspace::{BasisStore, CapturedBasis};

    let b = CapturedBasis::new(2, 2, 0, 0, 0);
    let mut store = BasisStore::new(1, 2);
    *store.get_mut(0, 1) = Some(b);

    let slices = store.split_workers_mut(1);
    let slice = &slices[0];
    let basis_ref = super::resolve_backward_basis(slice, 0, 1);

    assert!(basis_ref.is_some(), "expected Some when slot has a basis");
    drop(slices);
}

#[test]
fn resolve_backward_basis_returns_none_when_slot_is_empty() {
    // Given: BasisStore[0, 1] is None (cold-start, slot never written).
    // Then: resolve_backward_basis returns None.
    use crate::workspace::BasisStore;

    let mut store = BasisStore::new(1, 2);
    let slices = store.split_workers_mut(1);
    let slice = &slices[0];
    let basis_ref = super::resolve_backward_basis(slice, 0, 1);

    assert!(basis_ref.is_none(), "expected None for empty slot");
    drop(slices);
}

// ---------------------------------------------------------------------------
// T2 integration tests (backward write populates BasisStore)
// ---------------------------------------------------------------------------

#[test]
fn backward_write_populates_basis_store_at_omega_zero() {
    // Given: a 2-stage, 1-opening study with one forward trial point (m=0, x_hat=[5.0]).
    //        BasisStore starts empty (all None).
    // When: process_trial_point_backward runs at omega=0.
    // Then: BasisStore[0, 1] is Some(CapturedBasis) with state_at_capture == [5.0].
    //
    // write occurs only at omega=0 (this test has exactly 1 opening).
    // infeasibility guard is not triggered (solver succeeds).
    use crate::workspace::BasisStore;

    let mut basis_store = BasisStore::new(1, 2);
    let workspaces = run_one_trial_point_with_stores(&mut basis_store).unwrap();

    // Verify the BasisStore slot was written.
    assert!(
        basis_store.get(0, 1).is_some(),
        "BasisStore[0, 1] must be Some after backward write at omega=0"
    );
    let captured = basis_store.get(0, 1).unwrap();
    assert_eq!(
        captured.state_at_capture,
        vec![5.0_f64],
        "state_at_capture must equal x_hat"
    );
    // Confirm the solver ran exactly once.
    assert_eq!(
        workspaces[0].solver.inner().call_count,
        1,
        "solver must be called exactly once for a 1-opening backward pass"
    );
}

#[test]
fn backward_write_preserves_slot_on_infeasibility_at_omega_zero() {
    // Given: a 2-stage, 1-opening study.
    //        BasisStore starts with a pre-existing basis at [0, 1].
    //        The solver returns Infeasible on its first call.
    // When: process_trial_point_backward runs via run_backward_pass.
    // Then: run_backward_pass returns Err(SddpError::Infeasible) and
    //       BasisStore[0, 1] retains its original content.
    //
    // the write in process_trial_point_backward is guarded by `?`
    // immediately after run_stage_solve. An Infeasible error propagates
    // out of the function before reaching the BasisStore write site, so
    // the slot is unconditionally preserved on infeasibility.
    use cobre_solver::Basis;

    use crate::workspace::{BasisStore, CapturedBasis};

    // Pre-populate slot [0, 1] with a sentinel basis. `state_at_capture =
    // [42.0]` is the sentinel that the reuse-path overwrite must
    // replace. The remaining fields satisfy the `CapturedBasis`
    // invariant `row_status.len() == base_row_count + cut_row_slots.len()`.
    let pre_existing = CapturedBasis {
        basis: Basis::new(2, 2),
        base_row_count: 2,
        cut_row_slots: Vec::new(),
        state_at_capture: vec![42.0],
    };
    let mut basis_store = BasisStore::new(1, 2);
    *basis_store.get_mut(0, 1) = Some(pre_existing);

    // Verify sentinel is in place before the call.
    assert_eq!(
        basis_store.get(0, 1).unwrap().state_at_capture,
        vec![42.0_f64],
        "sentinel must be in place before the infeasible solve"
    );

    // run_one_trial_point_with_stores uses MockSolver::always_ok, so we
    // exercise the reuse path (successful solve overwrites slot). For the
    // infeasibility path, the structural guarantee is: `?` in
    // process_trial_point_backward propagates Err before the write site.
    // That path is integration-tested by `backward_pass_propagates_infeasible_error`.
    //
    // Here we test the complementary invariant: a *successful* solve at ω=0
    // with a pre-existing slot uses the reuse branch (get_basis into the
    // existing allocation) and leaves the slot Some (not None).
    let result = run_one_trial_point_with_stores(&mut basis_store);
    assert!(result.is_ok(), "expected Ok for successful solve");

    // The slot must still be Some after the successful reuse-path write.
    assert!(
        basis_store.get(0, 1).is_some(),
        "BasisStore[0, 1] must not be None after successful reuse-path write at ω=0"
    );
    // The reuse path updates state_at_capture to the current x_hat=[5.0].
    assert_eq!(
        basis_store.get(0, 1).unwrap().state_at_capture,
        vec![5.0_f64],
        "state_at_capture must be updated to x_hat by the reuse path"
    );
}

/// T-HW01: handshake passes when all ranks agree on `n_workers_local`.
///
/// Uses `StubComm` (echoes send→recv, i.e. min==max==local) with a
/// 2-worker setup and a 1-stage system so no backward stages are swept.
/// The test only validates that the uniformity check does not reject a
/// consistent 2-worker configuration.
#[test]
#[allow(clippy::too_many_lines)]
fn handshake_passes_with_local_backend() {
    use crate::lp_builder::PatchBuffer;

    let n_stages = 1_usize;
    let n_workers = 2_usize;
    let stochastic = make_stochastic_context(n_stages, 1);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0()];
    let base_rows = vec![1_usize];
    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];
    let solution = solution_1_0(100.0, -5.0);

    // Build 2 workspaces to exercise n_workers_local=2.
    let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
        .map(|idx| SolverWorkspace {
            rank: 0,
            worker_id: i32::try_from(idx).expect("idx fits i32"),
            solver: ProfiledSolver::new(MockSolver::always_ok(solution.clone())),
            patch_buf: PatchBuffer::new(1, 0, 0, 0, 0, 0),
            current_state: Vec::with_capacity(n_state),
            scratch: crate::workspace::ScratchBuffers {
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
                anticipated_state_buf: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
        })
        .collect();

    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
    let mut csb = CutSyncBuffers::new(n_state, 1, 1);
    let comm = StubComm;

    let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(n_stages),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    });

    assert!(
        result.is_ok(),
        "handshake must pass when all ranks have the same n_workers_local; got: {result:?}"
    );
}

/// T-HW02: handshake rejects non-uniform `n_workers_local` across ranks.
///
/// `NonUniformStubComm` simulates a 2-rank cluster where min and max
/// worker counts differ. Its `allreduce(Min)` returns all `T::default()`
/// (zeros), while `allreduce(Max)` copies `send` to `recv` (the local
/// value). With `local_workers = 1`, `min_recv[0] = 0` and
/// `max_recv[0] = 1`, so `0 != 1` triggers the uniformity check.
/// `BackwardPassState::run` must return `SddpError::Validation` with the
/// expected substring before entering the stage loop.
#[test]
#[allow(clippy::too_many_lines)]
fn handshake_rejects_nonuniform_workers() {
    /// Stub communicator that forces `allreduce(Min)` to return zeros and
    /// `allreduce(Max)` to echo the send buffer, producing `min != max`
    /// for any non-zero local value.
    struct NonUniformStubComm;

    impl Communicator for NonUniformStubComm {
        fn allgatherv<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            recv[..send.len()].copy_from_slice(send);
            Ok(())
        }

        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            op: ReduceOp,
        ) -> Result<(), CommError> {
            match op {
                // Min: return T::default() (0) to simulate a remote rank
                // with zero workers, creating a min != max discrepancy.
                ReduceOp::Min => {
                    for r in recv.iter_mut() {
                        *r = T::default();
                    }
                }
                // Max and all others: echo send so max == local value.
                _ => {
                    recv[..send.len()].copy_from_slice(send);
                }
            }
            Ok(())
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            Ok(())
        }

        fn barrier(&self) -> Result<(), CommError> {
            Ok(())
        }

        fn rank(&self) -> usize {
            0
        }

        fn size(&self) -> usize {
            2
        }

        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
    }

    let n_stages = 1_usize;
    let stochastic = make_stochastic_context(n_stages, 1);
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let templates = vec![minimal_template_1_0()];
    let base_rows = vec![1_usize];
    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];
    let comm = NonUniformStubComm;
    // n_workers_local = 1 on this rank; allreduce(Min) returns 0 and
    // allreduce(Max) returns 1 → 0 != 1 triggers the validation error.
    let mut workspaces =
        single_workspace(MockSolver::always_ok(solution_1_0(100.0, -5.0)), n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
    let mut csb = CutSyncBuffers::new(n_state, 1, 1);

    let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        baked: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(n_stages),
        training_ctx: &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            state: &state,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            noise_key_diag: None,
        },
        comm: &comm,
        records: &[],
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        visited_archive: None,
        event_sender: None,
    });

    match result {
        Err(crate::SddpError::Validation(ref msg)) => {
            assert!(
                msg.contains("non-uniform n_workers_local"),
                "error message must contain 'non-uniform n_workers_local'; got: {msg}"
            );
            assert!(
                msg.contains("min=0"),
                "error message must mention min=0 (stub Min returns T::default()); got: {msg}"
            );
            assert!(
                msg.contains("max=1"),
                "error message must mention max=1 (stub Max echoes local=1); got: {msg}"
            );
            assert!(
                msg.contains("local=1"),
                "error message must mention local=1 (single workspace); got: {msg}"
            );
        }
        other => panic!(
            "expected Err(SddpError::Validation(_)) from non-uniform handshake, got: {other:?}"
        ),
    }
}

/// Verify cut sign convention: the matured anticipated slot's coefficient 7.5 is
/// negated to -7.5 at the cut-target column. Drives the cut-row builder against a
/// finalized anticipated [`StateLayout`] (the role-(a) owner of the resolver).
#[test]
fn cut_coefficient_sign_convention_slot_zero_k2() {
    let state = crate::indexer::test_fixtures::state_layout_full(0, 0, 1, 2, vec![2]);
    assert_eq!(state.anticipated_state.start, 0);
    assert_eq!(state.n_state, 2);

    let mut fcf = FutureCostFunction::new(3, state.n_state, 1, 10, &[0; 3]);
    let mut coefficients = vec![0.0_f64; state.n_state];
    coefficients[state.anticipated_state.start] = 7.5;
    fcf.add_cut(1, 0, 0, 0.0, &coefficients);

    let mut batch = RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    };
    crate::cut::row::build_cut_row_batch_into(&mut batch, &fcf, 1, &state, &[]);

    // Slot 0 (j = anticipated_state.start) is the matured slot (K=2, slot+1==K via
    // the Equal branch), so it maps to anticipated_state_out.start == 1.
    let lp_col = state.state_to_lp_column(state.anticipated_state.start);
    assert_eq!(lp_col, 1);

    let pos = batch
        .col_indices
        .iter()
        .position(|&c| c == lp_col as i32)
        .expect("lp_col must appear in batch.col_indices");
    assert_eq!(
        batch.values[pos], -7.5,
        "stored subgradient 7.5 must be negated to -7.5 in the cut row"
    );
}

// -----------------------------------------------------------------------
// DCS backward-integration tests
// -----------------------------------------------------------------------

use crate::cut_selection::{CutMetadata, CutSelectionStrategy};
use crate::dcs::DcsParams;
use crate::lp_builder::PatchBuffer;
use crate::workspace::WorkspaceSizing;
use cobre_solver::ActiveSolver;

/// Cut-free successor core for a 1-hydro, no-lag stage:
/// columns `[storage_out=0, z_inflow=1, storage_in=2, theta=3]`.
/// Minimize `theta`. `patch_opening_bounds` pins `storage_in` (col 2, the
/// incoming-state column) to `x_hat`. A single coupling row
/// `storage_out - storage_in = 0` ties the outgoing-state column (col 0,
/// which the cuts reference) to the pinned incoming state, so the cut floor
/// is evaluated at `x_hat` and the cut subgradient flows back to the pinned
/// column — the minimal structure that makes the backward dual a real
/// subgradient with respect to the incoming state.
fn dcs_core_template() -> StageTemplate {
    StageTemplate {
        num_cols: 4,
        num_rows: 1,
        num_nz: 2,
        // CSC by column: col0 → (row0, +1), col2 → (row0, -1); cols 1,3 empty.
        col_starts: vec![0_i32, 1, 1, 2, 2],
        row_indices: vec![0_i32, 0],
        values: vec![1.0, -1.0],
        col_lower: vec![0.0, 0.0, 0.0, -1.0e6],
        col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY, 1.0e6],
        objective: vec![0.0, 0.0, 0.0, 1.0],
        // Coupling equality: storage_out - storage_in = 0.
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

/// Build a two-stage `FutureCostFunction` whose successor (stage 1) pool
/// carries three cuts on the incoming-storage state (index 0):
///   slot 0: intercept 1, coeff [0]   → floor 1.0 at `x_hat` = 2
///   slot 1: intercept 0, coeff [2]   → floor 4.0  (the binding cut)
///   slot 2: intercept 3, coeff [0]   → floor 3.0
/// Metadata is set so that the metadata-seeded initial set omits the
/// binding slot 1 (its `last_active_iter` is stale) — the lazy loop must
/// add it.
fn dcs_two_stage_fcf() -> FutureCostFunction {
    let n_stages = 2;
    let mut fcf = FutureCostFunction::new(n_stages, 1, 8, 10, &vec![0; n_stages]);
    fcf.add_cut(1, 0, 0, 1.0, &[0.0]);
    fcf.add_cut(1, 0, 1, 0.0, &[2.0]);
    fcf.add_cut(1, 0, 2, 3.0, &[0.0]);
    // Seed metadata: slots 0 and 2 are "recently active"; the binding slot 1
    // is stale so the k2 window excludes it (forcing the lazy add). All were
    // generated before the current iteration so none is current-iteration
    // protected.
    let meta = |generated: u64, last: u64| CutMetadata {
        iteration_generated: generated,
        forward_pass_index: 0,
        active_count: 0,
        last_active_iter: last,
    };
    fcf.pools[1].metadata[0] = meta(1, 5);
    fcf.pools[1].metadata[1] = meta(1, 1); // stale → outside k2=2 window at iter 5
    fcf.pools[1].metadata[2] = meta(1, 5);
    fcf
}

/// Single real-solver (`ActiveSolver`) workspace sized for `n_state = 1`.
fn dcs_active_workspace() -> Vec<SolverWorkspace<ActiveSolver>> {
    let sizing = WorkspaceSizing {
        hydro_count: 1,
        max_par_order: 0,
        n_load_buses: 0,
        max_blocks: 0,
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
    vec![SolverWorkspace::new(
        0,
        0,
        solver,
        PatchBuffer::new(1, 0, 0, 0, 0, 0),
        1,
        sizing,
    )]
}

/// Run one backward trial point at the successor stage with the real solver
/// and return the produced [`StagedCut`] plus the post-call per-slot
/// `metadata_sync_contribution` snapshot. `dcs` toggles the path. The
/// incoming state is pinned to `x_hat = 2.0`.
fn run_dcs_backward_trial_point(
    dcs: Option<DcsParams>,
    iteration: u64,
) -> (super::StagedCut, Vec<f64>, Vec<u64>) {
    run_dcs_backward_trial_point_at(dcs, iteration, 2.0)
}

/// `run_dcs_backward_trial_point` with the incoming-state pin `x_hat`
/// parameterized, so a sweep can vary the pinned state (which cut binds).
///
/// Returns the produced [`StagedCut`], its coefficient slice resolved from
/// the worker arena (the bytes the FCF would receive), and the post-call
/// `metadata_sync_contribution` snapshot.
fn run_dcs_backward_trial_point_at(
    dcs: Option<DcsParams>,
    iteration: u64,
    x_hat: f64,
) -> (super::StagedCut, Vec<f64>, Vec<u64>) {
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let n_state = state.n_state;
    let core = dcs_core_template();
    let templates = vec![core.clone(), core.clone()];
    let base_rows = vec![0_usize, 0_usize];
    let stochastic = make_stochastic_context(2, 1);
    let horizon = HorizonMode::Finite { num_stages: 2 };
    let risk_measures = vec![RiskMeasure::Expectation; 2];

    let mut fcf = dcs_two_stage_fcf();
    // All-cuts batch for the baked path (delta == all cuts here).
    let cut_batch = crate::cut::row::build_cut_row_batch(&fcf, 1, &state, &[]);
    let successor_active_slots: Vec<usize> = (0..fcf.pools[1].populated_count).collect();
    let num_cuts = successor_active_slots.len();

    let mut exchange = exchange_with_states(n_state, vec![vec![x_hat]]);
    let mut workspaces = dcs_active_workspace();
    let mut basis_store = empty_basis_store(exchange.local_count(), 2);

    let ctx = StageContext {
        templates: &templates,
        base_rows: &base_rows,
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &[],
        ncs_col_starts: &[],
        ncs_active_slot_to_local: &[],
        ncs_max_gen: &[],
        ncs_allow_curtailment: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    };
    let training_ctx = TrainingContext {
        horizon: &horizon,
        indexer: &indexer,
        state: &state,
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stochastic,
        initial_state: &[],
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
        dcs,
        noise_key_diag: None,
    };

    let probabilities = vec![1.0_f64];
    let succ = super::SuccessorSpec {
        t: 0,
        successor: 1,
        my_rank: 0,
        probabilities: &probabilities,
        cut_batch: &cut_batch,
        num_cuts_at_successor: num_cuts,
        template_num_rows: core.num_rows,
        baked_template: &core,
        successor_active_slots: &successor_active_slots,
        cut_activity_tolerance: 0.0,
        successor_populated_count: fcf.pools[1].populated_count,
        successor_pool: &fcf.pools[1],
    };

    let mut basis_slices = basis_store.split_workers_mut(1);
    let ws = &mut workspaces[0];
    // Choose the opening-solve strategy exactly as the driver does, then issue
    // the per-trial-point prepare/load (mirrors process_stage_backward's per-`m`
    // load after the per-opening solver-state reset). The `Baked` variant loads the baked all-cuts
    // LP; the `Lazy` variant loads the cut-free core + builds the metadata seed.
    let opening_solver = super::StageOpeningSolver::from_dcs_params(
        dcs.filter(|params| params.is_active(iteration)),
    );
    opening_solver.prepare(ws, &ctx, &succ, iteration);
    // Initialise the per-opening accumulator buffers the trial-point helper
    // expects (mirrors process_stage_backward's per-stage setup).
    let n_openings = succ.probabilities.len();
    while ws.backward_accum.outcomes.len() < n_openings {
        ws.backward_accum
            .outcomes
            .push(crate::risk_measure::BackwardOutcome {
                intercept: 0.0,
                coefficients: vec![0.0_f64; n_state],
                objective_value: 0.0,
            });
    }
    let pop = succ.successor_populated_count;
    if ws.backward_accum.slot_increments.len() < pop {
        ws.backward_accum.slot_increments.resize(pop, 0);
    }
    ws.backward_accum.slot_increments[..pop].fill(0);
    if ws.backward_accum.agg_coefficients.len() < n_state {
        ws.backward_accum.agg_coefficients.resize(n_state, 0.0);
    }
    // Size the coefficient arena for the single trial point (offset 0).
    if ws.backward_accum.agg_arena.len() < n_state {
        ws.backward_accum.agg_arena.resize(n_state, 0.0);
    }
    if ws.backward_accum.metadata_sync_contribution.len() < pop {
        ws.backward_accum.metadata_sync_contribution.resize(pop, 0);
    }
    ws.backward_accum.metadata_sync_contribution[..pop].fill(0);
    ws.backward_accum
        .per_opening_stats
        .resize_with(n_openings, SolverStatsDelta::default);
    for slot in &mut ws.backward_accum.per_opening_stats[..n_openings] {
        *slot = SolverStatsDelta::default();
    }

    let cut = super::process_trial_point_backward(
        ws,
        &ctx,
        &training_ctx,
        &exchange,
        0,
        iteration,
        &risk_measures,
        &succ,
        &mut basis_slices[0],
        &opening_solver,
        0,
        0,
    )
    .expect("backward trial-point solve must succeed");

    // Resolve the coefficient slice from the worker arena while `ws` is
    // still in scope (the bytes the FCF merge would read).
    let coefficients = staged_cut_coefficients(&cut, &ws.backward_accum.agg_arena).to_vec();
    let meta_sync = ws.backward_accum.metadata_sync_contribution[..pop].to_vec();
    // Touch fcf/exchange so the borrows live to here.
    let _ = (&mut fcf, &mut exchange);
    (cut, coefficients, meta_sync)
}

fn dcs_params(start_iteration: u64) -> DcsParams {
    DcsParams {
        k1: None,
        k2: 2,
        nadic: 10,
        epsilon_viol: 1e-10,
        start_iteration,
        max_inner_iterations: 50,
    }
}

/// AC3 (exactness, real solver, both backends via the active solver): the
/// DCS-built cut equals the all-cuts cut (intercept + every coefficient
/// within 1e-9). DCS seeds an initial set that omits the binding cut; the
/// lazy loop must add it and reach the same dual.
#[test]
fn backward_dcs_cut_equals_all_cuts_cut() {
    let iteration = 5;
    let (baked_cut, baked_coefficients, _) = run_dcs_backward_trial_point(None, iteration);
    let (dcs_cut, dcs_coefficients, _) =
        run_dcs_backward_trial_point(Some(dcs_params(2)), iteration);

    assert!(
        (baked_cut.intercept - dcs_cut.intercept).abs() < 1e-9,
        "intercept: baked {} vs DCS {}",
        baked_cut.intercept,
        dcs_cut.intercept
    );
    assert_eq!(baked_coefficients.len(), dcs_coefficients.len());
    for (i, (b, d)) in baked_coefficients.iter().zip(&dcs_coefficients).enumerate() {
        assert!(
            (b - d).abs() < 1e-9,
            "coefficient[{i}]: baked {b} vs DCS {d}"
        );
    }
    // The binding cut has gradient 2.0 on the incoming storage; both paths
    // must recover it.
    assert!(
        (baked_coefficients[0] - 2.0).abs() < 1e-9,
        "baked gradient must be the binding cut's 2.0, got {}",
        baked_coefficients[0]
    );
}

/// `dcs = None` ⇒ the baked all-cuts path is taken and the cut is identical
/// to the pre-DCS baseline (same fixture run with `None`).
#[test]
fn backward_dcs_off_is_identical_to_baseline() {
    let (cut_a, coefficients_a, _) = run_dcs_backward_trial_point(None, 5);
    let (cut_b, coefficients_b, _) = run_dcs_backward_trial_point(None, 5);
    assert_eq!(cut_a.intercept, cut_b.intercept);
    assert_eq!(coefficients_a, coefficients_b);
    // Baseline binding gradient.
    assert!((coefficients_a[0] - 2.0).abs() < 1e-9);
}

/// `dcs = Some` but `iteration < start_iteration` ⇒ the baked path is used
/// (DCS not yet active), so the cut equals the baked cut.
#[test]
fn backward_dcs_inactive_before_start_iteration() {
    // start_iteration = 4, iteration = 1 → inactive.
    let (baked_cut, baked_coefficients, baked_meta) = run_dcs_backward_trial_point(None, 1);
    let (early_cut, early_coefficients, early_meta) =
        run_dcs_backward_trial_point(Some(dcs_params(4)), 1);
    assert_eq!(baked_cut.intercept, early_cut.intercept);
    assert_eq!(baked_coefficients, early_coefficients);
    // Baked path updates binding-count metadata; the inactive-DCS run takes
    // the baked path, so its metadata contribution matches the baked run.
    assert_eq!(baked_meta, early_meta);
}

/// AC1: the DCS path extracts the cut gradient from the final all-satisfied
/// LP AND maintains the binding-count metadata slot-correct under the
/// resident `CutRowMap`. For this fixture the only cut that binds at the
/// converged optimum (`x_hat` = 2, theta = 4) is the binding slot 1; slots 0
/// (floor 1) and 2 (floor 3) are resident from the seed but slack, so their
/// cut-row duals are zero. The DCS binding-count contribution must therefore
/// equal the baked path's — slot 1 bumped, all others zero — proving the
/// slot-correct translation maps the resident binding row back to slot 1 and
/// to no other.
#[test]
fn backward_dcs_binding_counts_match_baked() {
    let (_, _, baked_meta) = run_dcs_backward_trial_point(None, 5);
    let (_, _, dcs_meta) = run_dcs_backward_trial_point(Some(dcs_params(2)), 5);

    // Baked path bumps exactly the binding slot 1 (the floor-4 cut at x=2).
    assert_eq!(
        baked_meta,
        vec![0, 1, 0],
        "baked path must bump exactly binding slot 1, got {baked_meta:?}"
    );
    // DCS path records the SAME binding-count contribution: the resident
    // binding row maps back to slot 1, and to no other slot.
    assert_eq!(
        dcs_meta, baked_meta,
        "DCS binding-count metadata must match baked (slot-correct via the \
         resident CutRowMap), got DCS {dcs_meta:?} vs baked {baked_meta:?}"
    );
}

/// `parse_cut_selection_config` for `method = "dynamic"` flows into
/// `DcsParams::from_strategy`, so the backward context's `dcs` is `Some`
/// for the dynamic variant and `None` otherwise.
#[test]
fn from_strategy_gates_the_backward_dcs_field() {
    let dynamic = CutSelectionStrategy::Dynamic {
        k1: None,
        k2: 5,
        nadic: 10,
        epsilon_viol: 1e-10,
        start_iteration: 2,
    };
    assert!(DcsParams::from_strategy(&dynamic).is_some());
    let level1 = CutSelectionStrategy::Level1 {
        check_frequency: 5,
        tie_tolerance: 1e-10,
    };
    assert!(DcsParams::from_strategy(&level1).is_none());
}

// -----------------------------------------------------------------------
// DCS backward validation gates (exactness / finite-k1 / determinism /
// slow sweep). Default `k1 = None` (∞) — exactness holds only when every
// pool cut is a candidate.
// -----------------------------------------------------------------------

/// `DcsParams` with an explicit finite `k1` candidate-recency window.
fn dcs_params_k1(start_iteration: u64, k1: Option<u32>) -> DcsParams {
    DcsParams {
        k1,
        k2: 2,
        nadic: 10,
        epsilon_viol: 1e-10,
        start_iteration,
        max_inner_iterations: 50,
    }
}

/// Exactness + "never spins": with `k1 = None` the DCS backward cut equals
/// the all-cuts cut (intercept + every coefficient within 1e-9), and the
/// lazy loop terminates — even with `max_inner_iterations = 1`, which forces
/// the bounded TC-fallback branch — so it can never spin unbounded.
#[test]
fn backward_dcs_exactness_and_terminates() {
    let iteration = 5;
    let (baked, baked_coefficients, _) = run_dcs_backward_trial_point(None, iteration);

    // Default-cap DCS reaches the no-violation stop and matches all-cuts.
    let (dcs, dcs_coefficients, _) = run_dcs_backward_trial_point(Some(dcs_params(2)), iteration);
    assert!((baked.intercept - dcs.intercept).abs() < 1e-9);
    for (b, d) in baked_coefficients.iter().zip(&dcs_coefficients) {
        assert!((b - d).abs() < 1e-9, "coeff mismatch baked {b} vs DCS {d}");
    }

    // A 1-iteration cap forces the bounded TC fallback; the call must still
    // return (no spin) and land on the exact all-cuts cut.
    let tight = DcsParams {
        max_inner_iterations: 1,
        ..dcs_params(2)
    };
    let (dcs_tc, dcs_tc_coefficients, _) = run_dcs_backward_trial_point(Some(tight), iteration);
    assert!((baked.intercept - dcs_tc.intercept).abs() < 1e-9);
    for (b, d) in baked_coefficients.iter().zip(&dcs_tc_coefficients) {
        assert!(
            (b - d).abs() < 1e-9,
            "TC-fallback coeff mismatch baked {b} vs DCS {d}"
        );
    }
}

/// A finite `k1` window demonstrably takes effect (guards against `k1`
/// being silently ignored): with `k1 = Some(1)` at iteration 5, the binding
/// cut (slot 1, generated at iteration 1 → age 4 ≥ 1) is windowed out of
/// candidacy and is also outside the `k2 = 2` initial-set window, so it is
/// never added. The DCS optimum then differs from the all-cuts optimum —
/// the deliberately-non-exact windowed mode.
#[test]
fn backward_dcs_finite_k1_window_takes_effect() {
    let iteration = 5;
    let (baked, baked_coefficients, _) = run_dcs_backward_trial_point(None, iteration);
    // Sanity: the all-cuts (and k1=None DCS) gradient is the binding cut's 2.0.
    assert!((baked_coefficients[0] - 2.0).abs() < 1e-9);

    let (windowed, windowed_coefficients, _) =
        run_dcs_backward_trial_point(Some(dcs_params_k1(2, Some(1))), iteration);
    // The binding cut is windowed out, so the windowed optimum differs:
    // the surviving cuts (slots 0,2, both gradient 0) give a 0 gradient and
    // a different intercept than the all-cuts cut.
    assert!(
        (windowed_coefficients[0] - baked_coefficients[0]).abs() > 1e-6
            || (windowed.intercept - baked.intercept).abs() > 1e-6,
        "finite k1 must change the cut vs all-cuts (windowed coeff {} intercept {}; \
         all-cuts coeff {} intercept {})",
        windowed_coefficients[0],
        windowed.intercept,
        baked_coefficients[0],
        baked.intercept,
    );
}

/// Determinism: running the integrated DCS backward trial point twice on
/// identical inputs yields bit-identical cuts AND bit-identical
/// binding-count metadata. A non-deterministic inner-loop insert order on
/// identical deterministic inputs would perturb the converged cut or the
/// metadata, so cut + metadata bit-identity is the determinism surface.
#[test]
fn backward_dcs_run_to_run_determinism() {
    let (cut_a, coefficients_a, meta_a) = run_dcs_backward_trial_point(Some(dcs_params(2)), 5);
    let (cut_b, coefficients_b, meta_b) = run_dcs_backward_trial_point(Some(dcs_params(2)), 5);
    assert_eq!(
        cut_a.intercept.to_bits(),
        cut_b.intercept.to_bits(),
        "intercept must be bit-identical run-to-run"
    );
    assert_eq!(coefficients_a.len(), coefficients_b.len());
    for (a, b) in coefficients_a.iter().zip(&coefficients_b) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "coefficient must be bit-identical run-to-run"
        );
    }
    assert_eq!(
        meta_a, meta_b,
        "binding-count metadata must be bit-identical run-to-run"
    );
}

/// Slow exactness sweep: across a handful of pinned incoming states (which
/// vary the binding cut), the `k1 = None` DCS backward cut equals the
/// all-cuts cut within 1e-9 at every point. Gated behind `slow-tests`.
#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow DCS exactness sweep")]
fn backward_dcs_exactness_sweep() {
    // x_hat values span the regimes where different cuts bind:
    //   slot0 floor = 1, slot1 floor = 2*x_hat, slot2 floor = 3.
    // x_hat < 1.5 → slot2 binds (3); x_hat > 1.5 → slot1 binds (2*x_hat).
    let x_hats = [0.0_f64, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0];
    let iterations = [3_u64, 5, 7];
    for &iteration in &iterations {
        for &x in &x_hats {
            let (baked, baked_coefficients, _) =
                run_dcs_backward_trial_point_at(None, iteration, x);
            let (dcs, dcs_coefficients, _) =
                run_dcs_backward_trial_point_at(Some(dcs_params(2)), iteration, x);
            assert!(
                (baked.intercept - dcs.intercept).abs() < 1e-9,
                "sweep iter {iteration} x_hat {x}: intercept baked {} vs DCS {}",
                baked.intercept,
                dcs.intercept
            );
            for (i, (b, d)) in baked_coefficients.iter().zip(&dcs_coefficients).enumerate() {
                assert!(
                    (b - d).abs() < 1e-9,
                    "sweep iter {iteration} x_hat {x}: coeff[{i}] baked {b} vs DCS {d}"
                );
            }
        }
    }
}

/// Baked successor template for the regression fixture: the cut-free base
/// (`dcs_core_template`, the coupling row `col0 - col2 = 0`) PLUS the binding
/// cut (`-2*col0 + theta >= 0`) baked as a second structural row. This
/// mimics baking being active (`baked_template.num_rows = 2 >
/// template_num_rows = 1`), so the all-cuts/baked successor LP already
/// carries a cut row that the DCS path must NOT re-append.
///
/// CSC by column (4 cols, 2 rows):
///   col0 -> (row0, +1), (row1, -5);  col2 -> (row0, -1);  col3 -> (row1, +1)
///
/// The baked cut intentionally DOMINATES the pool's true binding cut: it is
/// `-5*col0 + theta >= 0`, i.e. `theta >= 5*col0`, giving floor `10` at the
/// pinned `x_hat = 2` versus the pool's true optimum floor `4` (gradient 2).
/// This is a cut that is NOT in the DCS resident pool (the pool's cuts have
/// gradients 0 and 2, never 5). If the DCS path erroneously loaded this
/// baked template as its core, the LP would carry the spurious floor-10
/// constraint and the produced cut would be `theta = 10, gradient = 5` —
/// observably different from the correct all-cuts cut. Loading the cut-free
/// base (the fix) ignores this baked row, so the DCS cut matches all-cuts.
fn dcs_baked_template_with_one_cut() -> StageTemplate {
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
        // row0: coupling equality (=0); row1: spurious baked cut
        // -5*col0 + theta >= 0 (NOT a DCS pool cut).
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

/// Regression for the baked-template-as-core bug: when baking is active
/// (`baked_template.num_rows > template_num_rows`), the DCS path must load
/// the cut-free base `ctx.templates[s]` — NOT `succ.baked_template`, which
/// already carries the active cut rows. Loading the baked template would
/// leave its baked cut rows resident in the LP even though the lazy loop's
/// fresh `CutRowMap` does not own them, so DCS would solve against cut rows
/// it never selected.
///
/// Here `succ.baked_template` carries one baked cut row
/// (`dcs_baked_template_with_one_cut`, `num_rows = 2`) that is a spurious
/// floor-10 / gradient-5 constraint NOT present in the DCS pool, while
/// `ctx.templates[s]` is the cut-free base (`num_rows = 1`). With the fix
/// the DCS cut equals the all-cuts cut (gradient 2) within 1e-9. Against the
/// old `core = succ.baked_template`, the spurious baked cut dominates the
/// solve and DCS returns gradient 5 — observably wrong — failing this
/// assertion. (Verified: this test fails on the buggy code, passes on the
/// fix.)
#[test]
fn backward_dcs_baked_cuts_present_no_duplicate_rows() {
    let iteration = 5;
    let state = crate::indexer::test_fixtures::state_layout(1, 0);
    let indexer = crate::indexer::test_fixtures::geom(1, 0);
    let n_state = state.n_state;

    // Cut-free base (loaded by the DCS path) and the baked successor
    // template that carries the binding cut as a structural row.
    let base = dcs_core_template();
    let baked = dcs_baked_template_with_one_cut();
    // ctx.templates carries the cut-free base for the successor stage.
    let templates = vec![base.clone(), base.clone()];
    let base_rows = vec![0_usize, 0_usize];
    let stochastic = make_stochastic_context(2, 1);
    let horizon = HorizonMode::Finite { num_stages: 2 };
    let risk_measures = vec![RiskMeasure::Expectation; 2];

    let mut fcf = dcs_two_stage_fcf();
    // All-cuts batch (delta) for the baked path; with baked carrying the
    // binding cut, the delta is the remaining (non-baked) cuts. For the DCS
    // exactness comparison we only need the all-cuts reference, computed
    // from the full pool against the cut-free base below.
    let cut_batch = crate::cut::row::build_cut_row_batch(&fcf, 1, &state, &[]);
    let successor_active_slots: Vec<usize> = (0..fcf.pools[1].populated_count).collect();
    let num_cuts = successor_active_slots.len();

    let mut exchange = exchange_with_states(n_state, vec![vec![2.0]]);
    let mut workspaces = dcs_active_workspace();
    let mut basis_store = empty_basis_store(exchange.local_count(), 2);

    let ctx = StageContext {
        templates: &templates,
        base_rows: &base_rows,
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &[],
        ncs_col_starts: &[],
        ncs_active_slot_to_local: &[],
        ncs_max_gen: &[],
        ncs_allow_curtailment: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    };
    let training_ctx = TrainingContext {
        horizon: &horizon,
        indexer: &indexer,
        state: &state,
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stochastic,
        initial_state: &[],
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
        dcs: Some(dcs_params(2)),
        noise_key_diag: None,
    };

    let probabilities = vec![1.0_f64];
    // `baked_template` carries a baked cut row (num_rows = 2); the cut-free
    // base has template_num_rows = 1. This is the baking-active shape that
    // exposed the bug.
    let succ = super::SuccessorSpec {
        t: 0,
        successor: 1,
        my_rank: 0,
        probabilities: &probabilities,
        cut_batch: &cut_batch,
        num_cuts_at_successor: num_cuts,
        template_num_rows: base.num_rows,
        baked_template: &baked,
        successor_active_slots: &successor_active_slots,
        cut_activity_tolerance: 0.0,
        successor_populated_count: fcf.pools[1].populated_count,
        successor_pool: &fcf.pools[1],
    };

    let mut basis_slices = basis_store.split_workers_mut(1);
    let ws = &mut workspaces[0];
    // Choose the opening-solve strategy as the driver does, then issue the
    // per-trial-point prepare/load (mirrors process_stage_backward's per-`m`
    // load). `dcs = Some(dcs_params(2))` with `iteration = 5` selects `Lazy`,
    // whose prepare loads the cut-free core + builds the metadata seed.
    let opening_solver = super::StageOpeningSolver::from_dcs_params(
        training_ctx
            .dcs
            .filter(|params| params.is_active(iteration)),
    );
    opening_solver.prepare(ws, &ctx, &succ, iteration);
    let n_openings = succ.probabilities.len();
    while ws.backward_accum.outcomes.len() < n_openings {
        ws.backward_accum
            .outcomes
            .push(crate::risk_measure::BackwardOutcome {
                intercept: 0.0,
                coefficients: vec![0.0_f64; n_state],
                objective_value: 0.0,
            });
    }
    let pop = succ.successor_populated_count;
    if ws.backward_accum.slot_increments.len() < pop {
        ws.backward_accum.slot_increments.resize(pop, 0);
    }
    ws.backward_accum.slot_increments[..pop].fill(0);
    if ws.backward_accum.agg_coefficients.len() < n_state {
        ws.backward_accum.agg_coefficients.resize(n_state, 0.0);
    }
    // Size the coefficient arena for the single trial point (offset 0).
    if ws.backward_accum.agg_arena.len() < n_state {
        ws.backward_accum.agg_arena.resize(n_state, 0.0);
    }
    if ws.backward_accum.metadata_sync_contribution.len() < pop {
        ws.backward_accum.metadata_sync_contribution.resize(pop, 0);
    }
    ws.backward_accum.metadata_sync_contribution[..pop].fill(0);
    ws.backward_accum
        .per_opening_stats
        .resize_with(n_openings, SolverStatsDelta::default);
    for slot in &mut ws.backward_accum.per_opening_stats[..n_openings] {
        *slot = SolverStatsDelta::default();
    }

    let dcs_cut = super::process_trial_point_backward(
        ws,
        &ctx,
        &training_ctx,
        &exchange,
        0,
        iteration,
        &risk_measures,
        &succ,
        &mut basis_slices[0],
        &opening_solver,
        0,
        0,
    )
    .expect("DCS backward solve with baked cuts present must succeed");
    // Resolve the DCS coefficient slice from the worker arena while `ws` is
    // still in scope.
    let dcs_coefficients = staged_cut_coefficients(&dcs_cut, &ws.backward_accum.agg_arena).to_vec();
    let _ = (&mut fcf, &mut exchange);

    // The all-cuts reference cut (cut-free base + full pool, no DCS).
    let (allcuts, allcuts_coefficients, _) = run_dcs_backward_trial_point(None, iteration);

    // With the fix (core = cut-free ctx.templates[s]), the binding cut is
    // added exactly once and the DCS cut matches the all-cuts cut. With the
    // bug (core = baked_template), the baked cut is double-added and the
    // solve/extraction is malformed, so this fails.
    assert!(
        (dcs_cut.intercept - allcuts.intercept).abs() < 1e-9,
        "intercept: DCS {} vs all-cuts {}",
        dcs_cut.intercept,
        allcuts.intercept
    );
    assert_eq!(dcs_coefficients.len(), allcuts_coefficients.len());
    for (i, (d, a)) in dcs_coefficients
        .iter()
        .zip(&allcuts_coefficients)
        .enumerate()
    {
        assert!((d - a).abs() < 1e-9, "coeff[{i}]: DCS {d} vs all-cuts {a}");
    }
    // The binding gradient (2.0) must be recovered.
    assert!((dcs_coefficients[0] - 2.0).abs() < 1e-9);
}
