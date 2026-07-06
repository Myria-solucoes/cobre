//! Lower bound evaluation for iterative LP-based optimization algorithms.
//!
//! [`evaluate_lower_bound`] computes the risk-adjusted lower bound by solving
//! the stage-0 LP for every opening in the scenario tree and aggregating the
//! per-opening objectives through the stage-0 risk measure, then broadcasts the
//! scalar from rank 0.
//!
//! Must be called **after** the backward pass and cut sync so the FCF holds the
//! latest cuts when the LPs are solved.

use std::ops::Range;

use cobre_comm::Communicator;
use cobre_solver::{RowBatch, SolverError, SolverInterface};
use cobre_stochastic::{OpeningTree, StochasticContext, evaluate_par_batch, solve_par_noise_batch};

use crate::{
    cut::FutureCostFunction,
    cut::row::build_cut_row_batch_into,
    error::SddpError,
    indexer::StateLayout,
    inflow_method::InflowNonNegativityMethod,
    lag_transition::DisaggregationWeight,
    lp_builder::COST_SCALE_FACTOR,
    lp_builder::PatchBuffer,
    noise::{
        NcsNoiseOffsets, compute_disaggregation_next_rate, compute_effective_eta, has_par_model,
        transform_ncs_noise,
    },
    risk_measure::RiskMeasure,
};
use cobre_solver::StageTemplate;

/// Stage-0 inputs for [`evaluate_lower_bound`], bundled to reduce parameter count.
///
/// The lower bound evaluates stage 0 only, so every field is its stage-0 value;
/// the slice/range fields come from the stage-0 `StageContext` (and its
/// `StageGeometry`). NCS-related fields are empty — and NCS patching is skipped —
/// when no stochastic NCS entities exist.
pub struct LbEvalSpec<'a> {
    /// Stage-0 LP template.
    pub template: &'a StageTemplate,
    /// AR-dynamics base row.
    pub base_row: usize,
    /// ζ·σ inflow-noise scale per hydro.
    pub noise_scale: &'a [f64],
    /// Hydros carrying inflow noise.
    pub n_hydros: usize,
    /// Opening tree of noise realizations.
    pub opening_tree: &'a OpeningTree,
    /// Objective risk measure.
    pub risk_measure: &'a RiskMeasure,
    /// `Some` patches stochastic NCS column bounds per opening via `transform_ncs_noise`; `None` skips.
    pub stochastic: Option<&'a StochasticContext>,
    /// Offset to the NCS noise dimensions in the raw noise vector.
    pub n_load_buses: usize,
    /// MW, id-sorted (the order `transform_ncs_noise` emits its bound buffers).
    pub ncs_max_gen: &'a [f64],
    /// Aligned 1:1 with `ncs_max_gen`; `false` pins the column to availability.
    pub ncs_allow_curtailment: &'a [bool],
    /// Dense NCS column index per stochastic slot; see
    /// [`crate::context::StageContext::ncs_stochastic_dense_col`].
    pub ncs_stochastic_dense_col: &'a [usize],
    /// Keep the forward, backward, and lower-bound patch sites identical — the
    /// "patch NCS identically" contract; a divergence understates the bound (D15).
    pub ncs_stochastic_windows: &'a [(Option<i32>, Option<i32>)],
    /// Commissioning key the dormancy predicate compares the windows against.
    pub stage_id: i32,
    /// Blocks at stage 0.
    pub block_count: usize,
    /// LP column range for NCS generation.
    pub ncs_generation: Range<usize>,
    /// Always `0`: column-bound state pinning leaves no rows before the z-inflow block.
    pub z_inflow_row_start: usize,
    /// `Truncation`/`TruncationWithPenalty` clamp negative PAR(p) inflows to zero before patching.
    pub inflow_method: &'a InflowNonNegativityMethod,
    /// Stage-0 day-weighted disaggregation weight; `next_day_weight > 0.0` triggers
    /// the conditional-mean blend mirroring the backward pass's peek (no sampler
    /// at the lower bound, so the source-B rate uses `next_eta = 0`).
    pub disagg_weight: DisaggregationWeight,
    /// ζ at stage 0, scaling the disaggregation blend (`StageContext::zeta_s[0]`).
    pub zeta: f64,
}

/// Per-evaluation scratch buffers for [`evaluate_lower_bound`] on rank 0.
///
/// Allocated once on `IterationScratch` and reused across iterations: the first
/// call grows the `Vec` capacities, later calls refill them in place. Never
/// replace a reused buffer with a fresh `Vec` — that reintroduces the
/// per-iteration allocation this struct exists to avoid.
// `_buf` postfix is shared across fields by design.
#[allow(clippy::struct_field_names)]
pub struct LbEvalScratch {
    /// Per-opening noise realization (one entry per hydro).
    pub noise_buf: Vec<f64>,
    /// Z-inflow RHS per hydro for PAR(p) rows.
    pub z_inflow_rhs_buf: Vec<f64>,
    /// NCS column upper bounds in full stochastic-slot order (`transform_ncs_noise` per opening).
    pub ncs_col_upper_buf: Vec<f64>,
    /// Stage-0 active NCS column indices (built once before the opening loop).
    pub ncs_col_indices_buf: Vec<usize>,
    /// NCS column lower bounds, parallel to `ncs_col_upper_buf`.
    pub ncs_col_lower_buf: Vec<f64>,
    /// Active-subset gather (lower): `ncs_col_{lower,upper}_buf` run in full slot
    /// order, so gathering only the active slots here keeps the set-bounds
    /// index/lower/upper buffers equal-length at a strict-subset stage 0.
    pub ncs_col_lower_active_buf: Vec<f64>,
    /// Active-subset gather (upper), parallel to `ncs_col_lower_active_buf`.
    pub ncs_col_upper_active_buf: Vec<f64>,
    /// PAR lag matrix (constant across openings).
    pub lag_matrix_buf: Vec<f64>,
    /// Per-hydro eta floor from lags (constant across openings).
    pub eta_floor_buf: Vec<f64>,
    /// Per-hydro PAR inflow per opening.
    pub par_inflow_buf: Vec<f64>,
    /// Per-hydro effective eta after clamping (per opening).
    pub effective_eta_buf: Vec<f64>,
    /// Per-hydro zero-target vector for truncation precompute.
    pub zero_targets_buf: Vec<f64>,
    /// Uniform per-opening probabilities for risk-measure aggregation.
    pub uniform_prob_buf: Vec<f64>,
    /// Per-opening objective values from `lb_evaluate_stage_0`.
    pub objectives_buf: Vec<f64>,
    /// Regime A disaggregation source-B rate (next-period conditional mean).
    pub disagg_next_rate_buf: Vec<f64>,
    /// Disaggregation peek's shifted-lag scratch, kept separate from
    /// `lag_matrix_buf` — that buffer is precomputed once and read across every
    /// opening in the loop below, so writing the peek's shift into it would
    /// corrupt the truncation precompute for the next opening.
    pub disagg_shift_buf: Vec<f64>,
}

impl LbEvalScratch {
    /// Empty buffers; no allocation until the first `evaluate_lower_bound` call.
    #[must_use]
    pub fn new() -> Self {
        Self {
            noise_buf: Vec::new(),
            z_inflow_rhs_buf: Vec::new(),
            ncs_col_upper_buf: Vec::new(),
            ncs_col_indices_buf: Vec::new(),
            ncs_col_lower_buf: Vec::new(),
            ncs_col_lower_active_buf: Vec::new(),
            ncs_col_upper_active_buf: Vec::new(),
            lag_matrix_buf: Vec::new(),
            eta_floor_buf: Vec::new(),
            par_inflow_buf: Vec::new(),
            effective_eta_buf: Vec::new(),
            zero_targets_buf: Vec::new(),
            uniform_prob_buf: Vec::new(),
            objectives_buf: Vec::new(),
            disagg_next_rate_buf: Vec::new(),
            disagg_shift_buf: Vec::new(),
        }
    }
}

impl Default for LbEvalScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Groups the mutable scratch refs for [`evaluate_lower_bound`] so its signature
/// stays under clippy's `too-many-arguments-threshold`. Build via
/// [`LbEvalScratchBundle::from_scratch_fields`] (disjoint-borrow factory).
pub struct LbEvalScratchBundle<'a> {
    /// Reusable LP row-bound patch buffer.
    pub patch_buf: &'a mut PatchBuffer,
    /// Stage-0 cut row batch for the lower-bound LP.
    pub lb_cut_batch: &'a mut cobre_solver::RowBatch,
    /// Cut row map for append-only lower-bound LP management.
    pub lb_cut_row_map: Option<&'a mut crate::cut::CutRowMap>,
    /// Reusable per-evaluation scratch buffers.
    pub lb_scratch: &'a mut LbEvalScratch,
}

impl<'a> LbEvalScratchBundle<'a> {
    /// Take the disjoint `IterationScratch` fields separately — so the borrow
    /// checker can verify non-aliasing — then bundle them; the same disjoint-borrow
    /// factory pattern as `BackwardPassInputs::from_session_fields`.
    pub fn from_scratch_fields(
        patch_buf: &'a mut PatchBuffer,
        lb_cut_batch: &'a mut cobre_solver::RowBatch,
        lb_cut_row_map: Option<&'a mut crate::cut::CutRowMap>,
        lb_scratch: &'a mut LbEvalScratch,
    ) -> Self {
        Self {
            patch_buf,
            lb_cut_batch,
            lb_cut_row_map,
            lb_scratch,
        }
    }
}

/// Rank-0 setup: pre-populate the constant NCS column-index buffer and run the
/// append-only LP load. Only called on rank 0.
fn lb_init_rank0<S: SolverInterface>(
    solver: &mut S,
    fcf: &FutureCostFunction,
    spec: &LbEvalSpec<'_>,
    state_layout: &StateLayout,
    cut_state: &crate::indexer::CutStateProjection,
    lb_cut_batch: &mut RowBatch,
    lb_cut_row_map: Option<&mut crate::cut::CutRowMap>,
    scratch: &mut LbEvalScratch,
) {
    scratch.ncs_col_upper_buf.clear();
    scratch.ncs_col_indices_buf.clear();
    scratch.ncs_col_lower_buf.clear();

    // Indices are constant across openings — build once here; the per-opening
    // bound buffers are gathered inside the loop.
    if let Some(stoch) = spec.stochastic {
        let n_stochastic_ncs = stoch.n_stochastic_ncs();
        if n_stochastic_ncs > 0 && !spec.ncs_generation.is_empty() {
            crate::noise::build_dense_ncs_col_indices(
                spec.ncs_stochastic_dense_col,
                spec.ncs_generation.start,
                spec.block_count,
                &mut scratch.ncs_col_indices_buf,
            );
        }
    }

    scratch.par_inflow_buf.resize(spec.n_hydros, 0.0);

    // Append-only: cuts are never removed, keeping the lower bound monotone across
    // iterations. The CutRowMap-less branch (tests) rebuilds the model each call.
    if let Some(row_map) = lb_cut_row_map {
        if row_map.total_cut_rows() == 0 {
            solver.load_model(spec.template);
        }
        crate::cut::row::append_new_cuts_to_lp(
            solver,
            fcf,
            0,
            state_layout,
            cut_state,
            &spec.template.col_scale,
            row_map,
            lb_cut_batch,
        );
    } else {
        build_cut_row_batch_into(
            lb_cut_batch,
            fcf,
            0,
            state_layout,
            cut_state,
            &spec.template.col_scale,
        );
        solver.load_model(spec.template);
        if lb_cut_batch.num_rows > 0 {
            solver.add_rows(lb_cut_batch);
        }
    }
}

/// Truncation precompute (PAR lag matrix + eta floor, constant across openings),
/// then a per-opening LP solve writing each objective into `scratch.objectives_buf`.
///
/// # Errors
///
/// Returns [`SddpError::Infeasible`] if any opening LP is infeasible, or
/// [`SddpError::Solver`] for other LP solve failures.
// Rationale: the per-opening loop interleaves several correctness-critical
// sequential steps (truncation, disaggregation blend, NCS patch) that must not
// be split across functions without threading their shared scratch state.
#[allow(clippy::too_many_lines)]
fn lb_evaluate_stage_0<S: SolverInterface>(
    solver: &mut S,
    spec: &LbEvalSpec<'_>,
    patch_buf: &mut PatchBuffer,
    initial_state: &[f64],
    state_layout: &StateLayout,
    scratch: &mut LbEvalScratch,
) -> Result<(), SddpError> {
    let n_openings = spec.opening_tree.n_openings(0);
    let n_hydros = spec.n_hydros;
    let base_row = spec.base_row;

    let needs_truncation = matches!(
        spec.inflow_method,
        InflowNonNegativityMethod::Truncation | InflowNonNegativityMethod::TruncationWithPenalty
    );

    let par_lp_opt = spec.stochastic.map(StochasticContext::par);
    let truncation_par = if needs_truncation {
        par_lp_opt.filter(|p| p.n_stages() > 0 && p.n_hydros() == n_hydros)
    } else {
        None
    };

    if let Some(par_lp) = truncation_par {
        let max_order = state_layout.max_par_order;
        let lag_len = max_order * n_hydros;
        scratch.lag_matrix_buf.resize(lag_len, 0.0);
        for h in 0..n_hydros {
            for l in 0..max_order {
                scratch.lag_matrix_buf[l * n_hydros + h] =
                    initial_state[state_layout.inflow_lags.start + l * n_hydros + h];
            }
        }

        scratch.eta_floor_buf.resize(n_hydros, f64::NEG_INFINITY);
        scratch.zero_targets_buf.clear();
        scratch.zero_targets_buf.resize(n_hydros, 0.0);
        solve_par_noise_batch(
            par_lp,
            0,
            &scratch.lag_matrix_buf,
            &scratch.zero_targets_buf,
            &mut scratch.eta_floor_buf,
        );
    }

    // Uses disagg_shift_buf, never lag_matrix_buf: the latter is precomputed
    // once above and re-read every opening below, so writing the peek's shift
    // into it would corrupt the next opening's truncation precompute.
    let disagg_par = if spec.disagg_weight.next_day_weight > 0.0 {
        spec.stochastic
            .filter(|stoch| has_par_model(stoch, n_hydros))
    } else {
        None
    };
    if disagg_par.is_some() {
        scratch.zero_targets_buf.clear();
        scratch.zero_targets_buf.resize(n_hydros, 0.0);
    }

    scratch.objectives_buf.clear();

    for opening_idx in 0..n_openings {
        let raw_noise = spec.opening_tree.opening(0, opening_idx);
        scratch.noise_buf.clear();
        scratch.z_inflow_rhs_buf.clear();

        if let Some(par_lp) = truncation_par {
            // Slice raw_noise to its hydro prefix: it spans hydros + load buses + NCS,
            // but evaluate_par_batch wants only the n_hydros PAR series.
            evaluate_par_batch(
                par_lp,
                0,
                &scratch.lag_matrix_buf,
                &raw_noise[..n_hydros],
                &mut scratch.par_inflow_buf,
            );
        }

        compute_effective_eta(
            raw_noise,
            n_hydros,
            *spec.inflow_method,
            &scratch.par_inflow_buf,
            &scratch.eta_floor_buf,
            &mut scratch.effective_eta_buf,
        );

        for (h, &eta_eff) in scratch.effective_eta_buf.iter().enumerate() {
            scratch
                .noise_buf
                .push(spec.template.row_lower[base_row + h] + spec.noise_scale[h] * eta_eff);
            let z_rhs = spec.stochastic.map_or(0.0, |stoch| {
                let par_lp = stoch.par();
                if par_lp.n_stages() > 0 && par_lp.n_hydros() == n_hydros {
                    par_lp.deterministic_base(0, h) + par_lp.sigma(0, h) * eta_eff
                } else {
                    0.0
                }
            });
            scratch.z_inflow_rhs_buf.push(z_rhs);
        }

        if let Some(stoch) = disagg_par {
            // Unreachable given precompute_disaggregation_weights's invariant
            // (next_day_weight > 0.0 implies next_period_stage is Some).
            let next_stage = spec.disagg_weight.next_period_stage.unwrap_or_else(|| {
                debug_assert!(
                    false,
                    "next_day_weight > 0.0 with next_period_stage == None violates \
                     the precompute_disaggregation_weights invariant"
                );
                0
            });
            let eta_zero = &scratch.zero_targets_buf[..n_hydros];
            compute_disaggregation_next_rate(
                state_layout,
                initial_state,
                &scratch.z_inflow_rhs_buf,
                stoch,
                next_stage,
                eta_zero,
                &mut scratch.disagg_shift_buf,
                &mut scratch.disagg_next_rate_buf,
            );
            for h in 0..n_hydros {
                scratch.noise_buf[h] += spec.zeta
                    * spec.disagg_weight.next_day_weight
                    * (scratch.disagg_next_rate_buf[h] - scratch.z_inflow_rhs_buf[h]);
            }
        }

        patch_buf.fill_col_state_patches(state_layout, initial_state, &spec.template.col_scale);
        patch_buf.fill_forward_patches(
            state_layout,
            initial_state,
            &scratch.noise_buf,
            base_row,
            &spec.template.row_scale,
        );
        patch_buf.fill_z_inflow_patches(
            spec.z_inflow_row_start,
            &scratch.z_inflow_rhs_buf,
            &spec.template.row_scale,
        );
        let cp = patch_buf.state_col_patch_count();
        solver.set_col_bounds(
            &patch_buf.col_indices[..cp],
            &patch_buf.col_lower[..cp],
            &patch_buf.col_upper[..cp],
        );
        let n_patches = patch_buf.forward_patch_count();
        solver.set_row_bounds(
            &patch_buf.indices[..n_patches],
            &patch_buf.lower[..n_patches],
            &patch_buf.upper[..n_patches],
        );

        // The NCS bound patch MUST stay inside the per-opening loop — each opening's
        // noise changes the available NCS generation; hoisting it understates the
        // bound (D15, `d15_non_controllable_source`).
        if let Some(stoch) = spec.stochastic {
            let n_stochastic_ncs = stoch.n_stochastic_ncs();
            if n_stochastic_ncs > 0 && !spec.ncs_generation.is_empty() {
                transform_ncs_noise(
                    raw_noise,
                    &NcsNoiseOffsets {
                        n_hydros,
                        n_load_buses: spec.n_load_buses,
                    },
                    stoch,
                    0,
                    spec.block_count,
                    spec.ncs_max_gen,
                    spec.ncs_allow_curtailment,
                    &mut scratch.ncs_col_lower_buf,
                    &mut scratch.ncs_col_upper_buf,
                );
                // Gather the active slots' bounds, forcing `[0, 0]` for slots dormant
                // at the lower-bound stage — the same zeroing the forward/backward
                // patch sites apply.
                crate::noise::gather_dense_ncs_bounds(
                    spec.ncs_stochastic_windows,
                    spec.stage_id,
                    spec.block_count,
                    &scratch.ncs_col_lower_buf,
                    &scratch.ncs_col_upper_buf,
                    &mut scratch.ncs_col_lower_active_buf,
                    &mut scratch.ncs_col_upper_active_buf,
                );
                solver.set_col_bounds(
                    &scratch.ncs_col_indices_buf,
                    &scratch.ncs_col_lower_active_buf,
                    &scratch.ncs_col_upper_active_buf,
                );
            }
        }

        let view = solver.solve(None).map_err(|e| match e {
            SolverError::Infeasible => SddpError::Infeasible {
                stage: 0,
                iteration: 0,
                scenario: opening_idx,
            },
            other => SddpError::Solver(other),
        })?;
        scratch.objectives_buf.push(view.objective);
    }

    Ok(())
}

/// Apply `risk_measure` to the per-opening objectives (uniform probabilities),
/// scale by [`COST_SCALE_FACTOR`], then broadcast the scalar from rank 0.
///
/// # Errors
///
/// Returns [`SddpError::Communication`] if the broadcast fails.
fn lb_aggregate_and_broadcast<C: Communicator>(
    objectives: &[f64],
    risk_measure: &RiskMeasure,
    uniform_prob_buf: &mut Vec<f64>,
    comm: &C,
) -> Result<f64, SddpError> {
    #[allow(clippy::cast_precision_loss)]
    let uniform_prob = 1.0_f64 / objectives.len() as f64;
    uniform_prob_buf.clear();
    uniform_prob_buf.resize(objectives.len(), uniform_prob);
    let mut lb =
        risk_measure.evaluate_risk(objectives, uniform_prob_buf.as_slice()) * COST_SCALE_FACTOR;
    comm.broadcast(std::slice::from_mut(&mut lb), 0)
        .map_err(SddpError::from)?;
    Ok(lb)
}

/// Evaluate the global lower bound for the current FCF approximation.
///
/// Only rank 0 runs the stage-0 opening loop and applies the risk measure; the
/// resulting scalar is broadcast to all ranks. `initial_state` is the known `x_0`
/// (length `state.n_state`). See [`LbEvalSpec`] and [`LbEvalScratchBundle`].
///
/// # Errors
///
/// - [`SddpError::Infeasible`] — a stage-0 opening LP is infeasible (a modelling
///   error; stage 0 should always be feasible via the penalty/recourse structure).
/// - [`SddpError::Solver`] — LP solve failed for another reason.
/// - [`SddpError::Communication`] — the broadcast to non-root ranks failed.
///
/// # Panics
///
/// Panics if `spec.opening_tree.n_openings(0) == 0` on rank 0 — stage 0 must have
/// at least one opening (a caller contract).
pub fn evaluate_lower_bound<S: SolverInterface, C: Communicator>(
    solver: &mut S,
    fcf: &FutureCostFunction,
    initial_state: &[f64],
    state_layout: &StateLayout,
    cut_state: &crate::indexer::CutStateProjection,
    scratch: &mut LbEvalScratchBundle<'_>,
    spec: &LbEvalSpec<'_>,
    comm: &C,
) -> Result<f64, SddpError> {
    let mut lb = 0.0_f64;

    if comm.rank() == 0 {
        assert!(
            spec.opening_tree.n_openings(0) > 0,
            "evaluate_lower_bound: stage 0 must have at least one opening"
        );

        lb_init_rank0(
            solver,
            fcf,
            spec,
            state_layout,
            cut_state,
            scratch.lb_cut_batch,
            scratch.lb_cut_row_map.as_deref_mut(),
            scratch.lb_scratch,
        );

        lb_evaluate_stage_0(
            solver,
            spec,
            scratch.patch_buf,
            initial_state,
            state_layout,
            scratch.lb_scratch,
        )?;

        return lb_aggregate_and_broadcast(
            &scratch.lb_scratch.objectives_buf,
            spec.risk_measure,
            &mut scratch.lb_scratch.uniform_prob_buf,
            comm,
        );
    }

    comm.broadcast(std::slice::from_mut(&mut lb), 0)
        .map_err(SddpError::from)?;
    Ok(lb)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]
mod tests {
    use super::{
        LbEvalScratch, LbEvalScratchBundle, LbEvalSpec, evaluate_lower_bound, lb_evaluate_stage_0,
    };
    use crate::{
        cut::FutureCostFunction, error::SddpError, inflow_method::InflowNonNegativityMethod,
        lag_transition::DisaggregationWeight, lp_builder::PatchBuffer, risk_measure::RiskMeasure,
    };
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
    use cobre_solver::{
        Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };
    use cobre_stochastic::OpeningTree;

    fn empty_row_batch() -> RowBatch {
        RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        }
    }

    /// Return the owned locals needed to build an [`LbEvalScratchBundle`] for tests.
    fn make_lb_locals() -> (RowBatch, LbEvalScratch) {
        (empty_row_batch(), LbEvalScratch::new())
    }

    /// Minimal stage template for N=1 hydro, L=0 PAR order.
    ///
    /// Column layout: [storage (0), `storage_in` (1), theta (2)]
    /// Row layout: [`storage_fixing` (0)]
    fn minimal_template() -> StageTemplate {
        StageTemplate {
            num_cols: 3,
            num_rows: 1,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1], // col 1 (storage_in) has NZ at row 0
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
            objective: vec![0.0, 0.0, 1.0], // minimise theta
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

    /// Build an `OpeningTree` with `n_openings` openings at stage 0.
    ///
    /// Uses `generate_opening_tree` with a single-entity identity-correlation
    /// model. Because `MockSolver` ignores the noise values returned by
    /// `opening_tree.opening(...)`, the tree only needs to have the right shape
    /// (correct stage count and branching factor at stage 0).
    ///
    /// `dim = 1` throughout (single hydro, the N=1, L=0 state layout).
    fn simple_opening_tree(n_openings: usize) -> OpeningTree {
        use chrono::NaiveDate;
        use cobre_core::{
            EntityId,
            scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
            temporal::{
                Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
                StageStateConfig,
            },
        };
        use cobre_stochastic::correlation::resolve::DecomposedCorrelation;
        use std::collections::BTreeMap;

        // Single study stage with the requested branching factor.
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
                branching_factor: n_openings,
                noise_method: NoiseMethod::Saa,
            },
        };

        let entity_id = EntityId(1);
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
                    entities: vec![CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: entity_id,
                    }],
                    matrix: vec![vec![1.0]],
                }],
            },
        );
        let corr_model = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };
        let decomposed = DecomposedCorrelation::build(&corr_model).unwrap();
        let entity_order = vec![entity_id];

        cobre_stochastic::tree::generate::generate_opening_tree(
            42,
            &[stage],
            1, // dim = 1 hydro
            &decomposed,
            &entity_order,
            cobre_stochastic::ClassDimensions {
                n_hydros: 1,
                n_load_buses: 0,
                n_ncs: 0,
            },
            &cobre_stochastic::tree::generate::OpeningTreeGenerationInputs::default(),
        )
        .unwrap()
    }

    // ── Mock communicator ────────────────────────────────────────────────────

    /// Single-rank stub communicator. broadcast is a no-op (identity operation).
    struct LocalComm;

    impl Communicator for LocalComm {
        fn allgatherv<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            unreachable!("LocalComm allgatherv not used in lower_bound tests")
        }

        fn allreduce<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            unreachable!("LocalComm allreduce not used in lower_bound tests")
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            // Single rank: no-op. The value is already in buf from the rank-0 computation.
            Ok(())
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

    /// Communicator that fails on `broadcast` with `CommError::CollectiveFailed`.
    struct FailingBcastComm;

    impl Communicator for FailingBcastComm {
        fn allgatherv<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            unreachable!()
        }

        fn allreduce<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            unreachable!()
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            Err(CommError::CollectiveFailed {
                operation: "broadcast",
                mpi_error_code: -1,
                message: "test-induced broadcast failure".to_string(),
            })
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

    // ── Mock solver ──────────────────────────────────────────────────────────

    /// Mock solver that records `set_col_bounds` calls and returns configurable
    /// objective values in sequence.
    ///
    /// Each call to `solve()` returns the next value from `objectives`. If
    /// `infeasible_on_call` is set and the call index matches, returns
    /// `SolverError::Infeasible` instead.
    struct MockSolver {
        objectives: Vec<f64>,
        call_count: usize,
        infeasible_on_call: Option<usize>,
        /// Number of times `set_col_bounds` was called.
        set_col_bounds_calls: usize,
    }

    impl MockSolver {
        fn with_objectives(objectives: Vec<f64>) -> Self {
            Self {
                objectives,
                call_count: 0,
                infeasible_on_call: None,
                set_col_bounds_calls: 0,
            }
        }

        fn infeasible_on_first() -> Self {
            Self {
                objectives: vec![0.0],
                call_count: 0,
                infeasible_on_call: Some(0),
                set_col_bounds_calls: 0,
            }
        }
    }

    impl SolverInterface for MockSolver {
        type Profile = cobre_solver::ActiveProfile;

        fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}

        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
        fn load_model(&mut self, _template: &StageTemplate) {}
        fn add_rows(&mut self, _cuts: &RowBatch) {}
        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {
            self.set_col_bounds_calls += 1;
        }

        fn solve(
            &mut self,
            _basis: Option<&Basis>,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            let call = self.call_count;
            self.call_count += 1;
            if self.infeasible_on_call == Some(call) {
                return Err(SolverError::Infeasible);
            }
            let obj = self.objectives[call % self.objectives.len()];
            // evaluate_lower_bound reads only `view.objective`, so the slices stay empty.
            Ok(cobre_solver::SolutionView {
                objective: obj,
                primal: &[],
                dual: &[],
                reduced_costs: &[],
                iterations: 0,
                solve_time_seconds: 0.0,
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

    // ── Shared test setup ────────────────────────────────────────────────────

    fn make_fcf(n_stages: usize, n_state: usize) -> FutureCostFunction {
        // max_cuts=100, n_transfer=0
        FutureCostFunction::new(n_stages, n_state, 2, 100, &vec![0; n_stages])
    }

    // ── Unit tests ───────────────────────────────────────────────────────────

    /// AC1: 1 opening, Expectation — LB equals the single LP objective.
    #[test]
    fn one_opening_expectation_lb_equals_single_objective() {
        let state = crate::test_support::state_layout(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, state.n_state);
        let initial_state = vec![0.0_f64; state.n_state];
        let mut patch_buf = PatchBuffer::new(state.hydro_count, state.max_par_order, 0, 0, 0, 0, 0);
        let opening_tree = simple_opening_tree(1);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let mut solver = MockSolver::with_objectives(vec![100.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            stage_id: 0,
            block_count: 1,
            ncs_generation: 0..0,
            z_inflow_row_start: 0,
            inflow_method: &InflowNonNegativityMethod::None,
            disagg_weight: DisaggregationWeight::interior(0),
            zeta: 0.0,
        };
        let (mut row_batch, mut lb_scratch) = make_lb_locals();
        let mut bundle = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch,
            None,
            &mut lb_scratch,
        );
        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &state,
            &crate::test_support::cut_state_projection(&state),
            &mut bundle,
            &spec,
            &comm,
        )
        .unwrap();

        assert!(
            (lb - 100_000_000.0).abs() < 1e-7,
            "single opening expectation LB must equal objective 100.0 * COST_SCALE_FACTOR = 100_000_000.0, got {lb}"
        );
    }

    /// AC2: 3 openings, Expectation — LB equals mean of objectives.
    #[test]
    fn three_openings_expectation_lb_equals_mean() {
        let state = crate::test_support::state_layout(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, state.n_state);
        let initial_state = vec![0.0_f64; state.n_state];
        let mut patch_buf = PatchBuffer::new(state.hydro_count, state.max_par_order, 0, 0, 0, 0, 0);
        let opening_tree = simple_opening_tree(3);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let mut solver = MockSolver::with_objectives(vec![60.0, 80.0, 100.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            stage_id: 0,
            block_count: 1,
            ncs_generation: 0..0,
            z_inflow_row_start: 0,
            inflow_method: &InflowNonNegativityMethod::None,
            disagg_weight: DisaggregationWeight::interior(0),
            zeta: 0.0,
        };
        let (mut row_batch_lb, mut lb_scratch_lb) = make_lb_locals();
        let mut bundle_lb = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb,
            None,
            &mut lb_scratch_lb,
        );
        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &state,
            &crate::test_support::cut_state_projection(&state),
            &mut bundle_lb,
            &spec,
            &comm,
        )
        .unwrap();

        // E[60, 80, 100] with uniform probs = (60+80+100)/3 = 80.0; * COST_SCALE_FACTOR = 80_000_000.0
        assert!(
            (lb - 80_000_000.0).abs() < 1e-7,
            "three openings expectation LB must equal 80_000_000.0, got {lb}"
        );
    }

    /// AC3: 2 openings, CVaR(alpha=0.5, lambda=1.0) — pure `CVaR` selects worst.
    #[test]
    fn two_openings_pure_cvar_alpha_half_lb_equals_worst() {
        let state = crate::test_support::state_layout(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, state.n_state);
        let initial_state = vec![0.0_f64; state.n_state];
        let mut patch_buf = PatchBuffer::new(state.hydro_count, state.max_par_order, 0, 0, 0, 0, 0);
        let opening_tree = simple_opening_tree(2);
        // CVaR(alpha=0.5, lambda=1.0): pure CVaR; upper bound per scenario =
        // p / alpha = 0.5 / 0.5 = 1.0. With 2 equal-probability scenarios the
        // greedy allocation places all mass on the worst scenario.
        let rm = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 1.0,
        };
        let comm = LocalComm;

        let mut solver = MockSolver::with_objectives(vec![50.0, 150.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            stage_id: 0,
            block_count: 1,
            ncs_generation: 0..0,
            z_inflow_row_start: 0,
            inflow_method: &InflowNonNegativityMethod::None,
            disagg_weight: DisaggregationWeight::interior(0),
            zeta: 0.0,
        };
        let (mut row_batch_lb, mut lb_scratch_lb) = make_lb_locals();
        let mut bundle_lb = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb,
            None,
            &mut lb_scratch_lb,
        );
        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &state,
            &crate::test_support::cut_state_projection(&state),
            &mut bundle_lb,
            &spec,
            &comm,
        )
        .unwrap();

        // CVaR(alpha=0.5, lambda=1.0) with 2 uniform-probability openings
        // concentrates all weight on the worst (150.0); * COST_SCALE_FACTOR = 150_000_000.0.
        assert!(
            (lb - 150_000_000.0).abs() < 1e-7,
            "pure CVaR(0.5, 1.0) with 2 openings must equal 150_000_000.0, got {lb}"
        );
    }

    /// AC4 (extra): 2 openings, CVaR(alpha=1.0, lambda=1.0) = Expectation.
    #[test]
    fn two_openings_cvar_alpha_one_equals_expectation() {
        let state = crate::test_support::state_layout(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, state.n_state);
        let initial_state = vec![0.0_f64; state.n_state];
        let mut patch_buf = PatchBuffer::new(state.hydro_count, state.max_par_order, 0, 0, 0, 0, 0);
        let opening_tree = simple_opening_tree(2);
        let rm = RiskMeasure::CVaR {
            alpha: 1.0,
            lambda: 1.0,
        };
        let comm = LocalComm;

        let mut solver = MockSolver::with_objectives(vec![50.0, 150.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            stage_id: 0,
            block_count: 1,
            ncs_generation: 0..0,
            z_inflow_row_start: 0,
            inflow_method: &InflowNonNegativityMethod::None,
            disagg_weight: DisaggregationWeight::interior(0),
            zeta: 0.0,
        };
        let (mut row_batch_lb, mut lb_scratch_lb) = make_lb_locals();
        let mut bundle_lb = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb,
            None,
            &mut lb_scratch_lb,
        );
        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &state,
            &crate::test_support::cut_state_projection(&state),
            &mut bundle_lb,
            &spec,
            &comm,
        )
        .unwrap();

        // CVaR(alpha=1) = Expectation = (50+150)/2 = 100.0; * COST_SCALE_FACTOR = 100_000_000.0
        assert!(
            (lb - 100_000_000.0).abs() < 1e-7,
            "CVaR(alpha=1, lambda=1) must equal expectation 100_000_000.0, got {lb}"
        );
    }

    /// AC5: solver returns Infeasible for the first opening — must propagate as `SddpError::Infeasible`.
    #[test]
    fn infeasible_solve_maps_to_sddp_infeasible() {
        let state = crate::test_support::state_layout(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, state.n_state);
        let initial_state = vec![0.0_f64; state.n_state];
        let mut patch_buf = PatchBuffer::new(state.hydro_count, state.max_par_order, 0, 0, 0, 0, 0);
        let opening_tree = simple_opening_tree(1);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let mut solver = MockSolver::infeasible_on_first();

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            stage_id: 0,
            block_count: 1,
            ncs_generation: 0..0,
            z_inflow_row_start: 0,
            inflow_method: &InflowNonNegativityMethod::None,
            disagg_weight: DisaggregationWeight::interior(0),
            zeta: 0.0,
        };
        let (mut row_batch_result, mut lb_scratch_result) = make_lb_locals();
        let mut bundle_result = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_result,
            None,
            &mut lb_scratch_result,
        );
        let result = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &state,
            &crate::test_support::cut_state_projection(&state),
            &mut bundle_result,
            &spec,
            &comm,
        );

        assert!(
            matches!(result, Err(SddpError::Infeasible { stage: 0, .. })),
            "infeasible solver must produce SddpError::Infeasible at stage 0, got {result:?}"
        );
    }

    /// AC6: broadcast failure maps to `SddpError::Communication`.
    #[test]
    fn broadcast_failure_maps_to_communication_error() {
        let state = crate::test_support::state_layout(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, state.n_state);
        let initial_state = vec![0.0_f64; state.n_state];
        let mut patch_buf = PatchBuffer::new(state.hydro_count, state.max_par_order, 0, 0, 0, 0, 0);
        let opening_tree = simple_opening_tree(1);
        let rm = RiskMeasure::Expectation;
        let comm = FailingBcastComm;

        let mut solver = MockSolver::with_objectives(vec![100.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            stage_id: 0,
            block_count: 1,
            ncs_generation: 0..0,
            z_inflow_row_start: 0,
            inflow_method: &InflowNonNegativityMethod::None,
            disagg_weight: DisaggregationWeight::interior(0),
            zeta: 0.0,
        };
        let (mut row_batch_result, mut lb_scratch_result) = make_lb_locals();
        let mut bundle_result = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_result,
            None,
            &mut lb_scratch_result,
        );
        let result = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &state,
            &crate::test_support::cut_state_projection(&state),
            &mut bundle_result,
            &spec,
            &comm,
        );

        assert!(
            matches!(result, Err(SddpError::Communication(_))),
            "broadcast failure must produce SddpError::Communication, got {result:?}"
        );
    }

    // ── Integration tests ────────────────────────────────────────────────────

    /// Integration: full round-trip with `LocalComm` and 2 openings.
    ///
    /// Verifies that the function correctly integrates with `build_cut_row_batch`
    /// (`cut_batch` with 0 cuts still produces the right result), `fill_forward_patches`,
    /// and `RiskMeasure::Expectation`.
    #[test]
    fn integration_two_openings_local_backend_expectation() {
        let state = crate::test_support::state_layout(1, 0);
        let template = minimal_template();
        // Start with 0 cuts (empty FCF).
        let fcf = make_fcf(2, state.n_state);
        let initial_state = vec![50.0_f64]; // non-zero initial state
        let mut patch_buf = PatchBuffer::new(state.hydro_count, state.max_par_order, 0, 0, 0, 0, 0);
        let opening_tree = simple_opening_tree(2);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let mut solver = MockSolver::with_objectives(vec![200.0, 300.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            stage_id: 0,
            block_count: 1,
            ncs_generation: 0..0,
            z_inflow_row_start: 0,
            inflow_method: &InflowNonNegativityMethod::None,
            disagg_weight: DisaggregationWeight::interior(0),
            zeta: 0.0,
        };
        let (mut row_batch_lb, mut lb_scratch_lb) = make_lb_locals();
        let mut bundle_lb = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb,
            None,
            &mut lb_scratch_lb,
        );
        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &state,
            &crate::test_support::cut_state_projection(&state),
            &mut bundle_lb,
            &spec,
            &comm,
        )
        .unwrap();

        // E[200, 300] = 250.0; * COST_SCALE_FACTOR = 250_000_000.0
        assert!(
            (lb - 250_000_000.0).abs() < 1e-7,
            "integration round-trip must produce 250_000_000.0, got {lb}"
        );
    }

    /// Integration: monotonicity — adding cuts can only increase the LB.
    ///
    /// This test calls `evaluate_lower_bound` twice: first with 0 cuts, then
    /// with objectives set higher (simulating tighter cuts). The second LB
    /// must be >= the first.
    #[test]
    fn integration_monotonicity_more_cuts_yields_higher_or_equal_lb() {
        let state = crate::test_support::state_layout(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, state.n_state);
        let initial_state = vec![0.0_f64];
        let mut patch_buf = PatchBuffer::new(state.hydro_count, state.max_par_order, 0, 0, 0, 0, 0);
        let opening_tree = simple_opening_tree(2);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            stage_id: 0,
            block_count: 1,
            ncs_generation: 0..0,
            z_inflow_row_start: 0,
            inflow_method: &InflowNonNegativityMethod::None,
            disagg_weight: DisaggregationWeight::interior(0),
            zeta: 0.0,
        };

        // First call: solver returns [50, 100] → LB = 75.
        let mut solver1 = MockSolver::with_objectives(vec![50.0, 100.0]);
        let (mut row_batch_lb1, mut lb_scratch_lb1) = make_lb_locals();
        let mut bundle_lb1 = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb1,
            None,
            &mut lb_scratch_lb1,
        );
        let lb1 = evaluate_lower_bound(
            &mut solver1,
            &fcf,
            &initial_state,
            &state,
            &crate::test_support::cut_state_projection(&state),
            &mut bundle_lb1,
            &spec,
            &comm,
        )
        .unwrap();

        // Second call: solver returns [80, 120] → LB = 100 (tighter cuts raise obj).
        let mut solver2 = MockSolver::with_objectives(vec![80.0, 120.0]);
        let (mut row_batch_lb2, mut lb_scratch_lb2) = make_lb_locals();
        let mut bundle_lb2 = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb2,
            None,
            &mut lb_scratch_lb2,
        );
        let lb2 = evaluate_lower_bound(
            &mut solver2,
            &fcf,
            &initial_state,
            &state,
            &crate::test_support::cut_state_projection(&state),
            &mut bundle_lb2,
            &spec,
            &comm,
        )
        .unwrap();

        assert!(
            lb2 >= lb1,
            "second LB ({lb2}) must be >= first LB ({lb1}) when cuts are tighter"
        );
    }

    // ── Inflow truncation tests ─────────────────────────────────────────────

    /// `None` method passes raw noise through unchanged (regression test).
    ///
    /// With `stochastic: None`, the truncation path is a no-op since
    /// `has_par == false`. This validates that the `compute_effective_eta`
    /// control flow works correctly when no PAR model is present.
    #[test]
    fn test_lb_none_method_unchanged() {
        let state = crate::test_support::state_layout(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, state.n_state);
        let initial_state = vec![0.0_f64; state.n_state];
        let mut patch_buf = PatchBuffer::new(state.hydro_count, state.max_par_order, 0, 0, 0, 0, 0);
        let opening_tree = simple_opening_tree(2);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let mut solver = MockSolver::with_objectives(vec![60.0, 80.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            stage_id: 0,
            block_count: 1,
            ncs_generation: 0..0,
            z_inflow_row_start: 0,
            inflow_method: &InflowNonNegativityMethod::None,
            disagg_weight: DisaggregationWeight::interior(0),
            zeta: 0.0,
        };
        let (mut row_batch_lb, mut lb_scratch_lb) = make_lb_locals();
        let mut bundle_lb = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_lb,
            None,
            &mut lb_scratch_lb,
        );
        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &state,
            &crate::test_support::cut_state_projection(&state),
            &mut bundle_lb,
            &spec,
            &comm,
        )
        .unwrap();

        // E[60, 80] = 70.0; * COST_SCALE_FACTOR = 70_000_000.0
        assert!(
            (lb - 70_000_000.0).abs() < 1e-7,
            "None method must produce correct LB, got {lb}"
        );
    }

    /// `Truncation` method does not cause a crash or infeasibility.
    ///
    /// With `stochastic: None`, the truncation path is a no-op since
    /// `has_par == false`, but this validates that the control flow
    /// (`needs_truncation` = true, `truncation_par` = `None`) does not panic.
    #[test]
    fn test_lb_truncation_no_crash() {
        let state = crate::test_support::state_layout(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, state.n_state);
        let initial_state = vec![0.0_f64; state.n_state];
        let mut patch_buf = PatchBuffer::new(state.hydro_count, state.max_par_order, 0, 0, 0, 0, 0);
        let opening_tree = simple_opening_tree(1);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let mut solver = MockSolver::with_objectives(vec![100.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            stage_id: 0,
            block_count: 1,
            ncs_generation: 0..0,
            z_inflow_row_start: 0,
            inflow_method: &InflowNonNegativityMethod::Truncation,
            disagg_weight: DisaggregationWeight::interior(0),
            zeta: 0.0,
        };
        let (mut row_batch_result, mut lb_scratch_result) = make_lb_locals();
        let mut bundle_result = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_result,
            None,
            &mut lb_scratch_result,
        );
        let result = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &state,
            &crate::test_support::cut_state_projection(&state),
            &mut bundle_result,
            &spec,
            &comm,
        );

        assert!(
            result.is_ok(),
            "Truncation method must not panic or fail, got {result:?}"
        );
    }

    /// `TruncationWithPenalty` method does not cause a crash or infeasibility.
    #[test]
    fn test_lb_truncation_with_penalty_no_crash() {
        let state = crate::test_support::state_layout(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, state.n_state);
        let initial_state = vec![0.0_f64; state.n_state];
        let mut patch_buf = PatchBuffer::new(state.hydro_count, state.max_par_order, 0, 0, 0, 0, 0);
        let opening_tree = simple_opening_tree(1);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let mut solver = MockSolver::with_objectives(vec![100.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            stage_id: 0,
            block_count: 1,
            ncs_generation: 0..0,
            z_inflow_row_start: 0,
            inflow_method: &InflowNonNegativityMethod::TruncationWithPenalty,
            disagg_weight: DisaggregationWeight::interior(0),
            zeta: 0.0,
        };
        let (mut row_batch_result, mut lb_scratch_result) = make_lb_locals();
        let mut bundle_result = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch_result,
            None,
            &mut lb_scratch_result,
        );
        let result = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &state,
            &crate::test_support::cut_state_projection(&state),
            &mut bundle_result,
            &spec,
            &comm,
        );

        assert!(
            result.is_ok(),
            "TruncationWithPenalty method must not panic or fail, got {result:?}"
        );
    }

    // ── NCS column-bound patching regression test ────────────────────────────

    /// Correctness guard for the D15 NCS column-bound patch contract: NCS column
    /// bounds must be patched *per opening*, not once before the loop or not at
    /// all, so the solver receives one `set_col_bounds` call per opening.
    // `clippy::too_many_lines`: the inline `System`/`StochasticContext` fixture and
    // its per-opening assertions are one coherent scenario; splitting them into
    // helpers would scatter the setup the assertions depend on and obscure the test.
    // `clippy::similar_names`: the role-(a) `state` handle reads next to `stage`-
    // named locals; both are established names.
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    #[test]
    fn lb_evaluate_stage_0_patches_ncs_bounds_per_opening() {
        use cobre_core::{
            Bus, DeficitSegment, EntityId, SystemBuilder,
            entities::non_controllable::NonControllableSource,
            scenario::{
                CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
                NcsModel, SamplingScheme,
            },
            temporal::{
                Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
                StageStateConfig,
            },
        };
        use cobre_stochastic::context::{
            ClassSchemes, OpeningTreeInputs, build_stochastic_context,
        };
        use std::collections::BTreeMap;

        let n_openings = 3_usize;
        let n_ncs = 1_usize;
        let block_count = 1_usize;
        let ncs_entity_id = EntityId(10);

        // Build a minimal System with one bus and one NCS entity.
        let bus = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };

        let ncs_source = NonControllableSource {
            id: ncs_entity_id,
            name: "W1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(0),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 100.0,
            allow_curtailment: true,
            curtailment_cost: 0.0,
        };

        let stage = Stage {
            index: 0,
            id: 0,
            start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: n_openings,
                noise_method: NoiseMethod::Saa,
            },
        };

        // NCS model: mean=0.5, std=0.1 availability factor.
        let ncs_model = NcsModel {
            ncs_id: ncs_entity_id,
            stage_id: 0,
            mean: 0.5,
            std: 0.1,
        };

        // Correlation: single NCS entity, identity correlation.
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "ncs_group".to_string(),
                    entities: vec![CorrelationEntity {
                        entity_type: "ncs".to_string(),
                        id: ncs_entity_id,
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
            .non_controllable_sources(vec![ncs_source])
            .stages(vec![stage])
            .ncs_models(vec![ncs_model])
            .correlation(correlation)
            .build()
            .unwrap();

        let stoch = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: None,
                load: None,
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        assert_eq!(
            stoch.n_stochastic_ncs(),
            n_ncs,
            "StochasticContext must report {n_ncs} stochastic NCS entity"
        );

        let opening_tree = stoch.opening_tree();

        // Build a template with 1 NCS generation column (col index 0).
        // The NCS generation column range is 0..block_count (= 0..1).
        let template = StageTemplate {
            num_cols: 1,
            num_rows: 0,
            num_nz: 0,
            col_starts: vec![0_i32, 0],
            row_indices: vec![],
            values: vec![],
            col_lower: vec![0.0],
            col_upper: vec![100.0],
            objective: vec![0.0],
            row_lower: vec![],
            row_upper: vec![],
            n_state: 0,
            n_transfer: 0,
            n_dual_relevant: 0,
            n_hydro: 0,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };

        let state = crate::test_support::state_layout(0, 0);
        let ncs_max_gen = vec![100.0_f64; n_ncs];
        let ncs_allow_curtailment = vec![true; n_ncs];
        // Dense: every stochastic slot maps to its own NCS column (slot order ==
        // system order here), and every slot is windowless (active at every stage),
        // so none is commissioning-dormant at stage 0.
        let ncs_stochastic_dense_col: Vec<usize> = (0..n_ncs).collect();
        let ncs_stochastic_windows: Vec<(Option<i32>, Option<i32>)> = vec![(None, None); n_ncs];

        let spec = LbEvalSpec {
            template: &template,
            base_row: 0,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree,
            risk_measure: &RiskMeasure::Expectation,
            stochastic: Some(&stoch),
            n_load_buses: 0,
            ncs_max_gen: &ncs_max_gen,
            ncs_allow_curtailment: &ncs_allow_curtailment,
            ncs_stochastic_dense_col: &ncs_stochastic_dense_col,
            ncs_stochastic_windows: &ncs_stochastic_windows,
            stage_id: 0,
            block_count,
            ncs_generation: 0..block_count,
            z_inflow_row_start: 0,
            inflow_method: &InflowNonNegativityMethod::None,
            disagg_weight: DisaggregationWeight::interior(0),
            zeta: 0.0,
        };

        // This test calls lb_evaluate_stage_0 directly, so it seeds the NCS index
        // buffer that lb_init_rank0 would otherwise build; the lower/upper bound
        // buffers are left empty — the loop refills them per opening.
        let mut lb_scratch = LbEvalScratch::new();
        for ncs_idx in 0..n_ncs {
            for blk in 0..block_count {
                lb_scratch
                    .ncs_col_indices_buf
                    .push(spec.ncs_generation.start + ncs_idx * block_count + blk);
            }
        }

        let mut patch_buf = PatchBuffer::new(0, 0, 0, 0, 0, 0, 0);
        let initial_state: Vec<f64> = Vec::new();
        let actual_n_openings = opening_tree.n_openings(0);
        let mut solver =
            MockSolver::with_objectives(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]);

        lb_evaluate_stage_0(
            &mut solver,
            &spec,
            &mut patch_buf,
            &initial_state,
            &state,
            &mut lb_scratch,
        )
        .unwrap();

        assert_eq!(
            solver.set_col_bounds_calls,
            2 * actual_n_openings,
            "set_col_bounds must be called twice per opening ({actual_n_openings} openings), \
             got {} calls — NCS bounds are not being patched per opening",
            solver.set_col_bounds_calls
        );
        assert!(
            actual_n_openings > 0,
            "opening tree must have at least one opening at stage 0"
        );
    }

    // ── Scratch reuse regression test ────────────────────────────────────────

    /// Verify that `LbEvalScratch` buffers are reused across consecutive calls.
    ///
    /// Calls `evaluate_lower_bound` twice on the same scratch and verifies that
    /// `noise_buf.capacity()` does not decrease on the second call (i.e., no
    /// reallocation occurred). This guards against regressions that would re-
    /// introduce per-iteration heap allocation on the lower-bound hot path.
    #[test]
    fn lb_eval_scratch_reuses_buffers_across_calls() {
        // Use n_hydros = 1 so that noise_buf gets populated (capacity grows to 1
        // after the first call). The template must have at least 1 row to avoid
        // index-out-of-bounds in fill_forward_patches when n_hydros = 1.
        let state = crate::test_support::state_layout(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, state.n_state);
        let initial_state = vec![0.0_f64; state.n_state];
        let mut patch_buf = PatchBuffer::new(state.hydro_count, state.max_par_order, 0, 0, 0, 0, 0);
        let opening_tree = simple_opening_tree(1);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let spec = LbEvalSpec {
            template: &template,
            base_row: 0,
            noise_scale: &[1.0],
            n_hydros: 1,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            stage_id: 0,
            block_count: 1,
            ncs_generation: 0..0,
            z_inflow_row_start: 0,
            inflow_method: &InflowNonNegativityMethod::None,
            disagg_weight: DisaggregationWeight::interior(0),
            zeta: 0.0,
        };

        let mut row_batch = empty_row_batch();
        let mut lb_scratch = LbEvalScratch::new();

        let mut solver1 = MockSolver::with_objectives(vec![10.0]);
        {
            let mut bundle = LbEvalScratchBundle::from_scratch_fields(
                &mut patch_buf,
                &mut row_batch,
                None,
                &mut lb_scratch,
            );
            evaluate_lower_bound(
                &mut solver1,
                &fcf,
                &initial_state,
                &state,
                &crate::test_support::cut_state_projection(&state),
                &mut bundle,
                &spec,
                &comm,
            )
            .unwrap();
        }

        let cap_after_first = lb_scratch.noise_buf.capacity();
        assert!(
            cap_after_first > 0,
            "noise_buf must have nonzero capacity after first call (n_hydros = 1)"
        );

        let mut solver2 = MockSolver::with_objectives(vec![20.0]);
        {
            let mut bundle = LbEvalScratchBundle::from_scratch_fields(
                &mut patch_buf,
                &mut row_batch,
                None,
                &mut lb_scratch,
            );
            evaluate_lower_bound(
                &mut solver2,
                &fcf,
                &initial_state,
                &state,
                &crate::test_support::cut_state_projection(&state),
                &mut bundle,
                &spec,
                &comm,
            )
            .unwrap();
        }

        let cap_after_second = lb_scratch.noise_buf.capacity();
        assert_eq!(
            cap_after_second, cap_after_first,
            "noise_buf capacity must be stable across calls (first={cap_after_first}, second={cap_after_second}); \
             a decrease indicates reallocation on the lower-bound hot path"
        );
    }

    // ── Filling phase-gating inheritance (template-driven, no per-opening patch) ─
    //
    // Filling gating is stage-deterministic (a function of `stage.id` + `FillingConfig`),
    // so it lives entirely in the per-stage `StageTemplate` and `noise_scale` the lower
    // bound already loads — it inherits filling structure by construction, with no
    // per-opening patch (unlike NCS, whose per-opening stochastic draw forces a
    // re-patch). Hand-wiring a filling patch here would duplicate template structure
    // into the hot path; the source-text guard below fails that edit.

    /// Build the per-stage templates for a study whose hydros exercise filling at
    /// stage 0, via the SAME `build_stage_templates` (`geometry_per_stage`) path
    /// the training loop uses.
    ///
    /// Two filling hydros, both white-noise (`max_par_order = 0`), independent (no
    /// cascade), on a single bus:
    /// - `H_A` (id 3): `start_stage_id = 0`, `entry_stage_id = 1`. At stage 0 it is
    ///   in the terminal `Filling` stage (`entry − 1 == 0`), so stage 0 carries the
    ///   `filling_target`/`σ_fill` row+column family. Its noise is NOT zeroed
    ///   (Filling keeps PAR noise).
    /// - `H_B` (id 4): `start_stage_id = 2`, `entry_stage_id = 4`. At stage 0
    ///   (`id 0 < start 2`) it is `PreFilling`, so `compute_noise_scale` zeros its
    ///   stage-0 `noise_scale` entry.
    ///
    /// Returns the built [`StageTemplates`] plus the system indices of the two
    /// hydros (id-sorted, so `H_A`→0, `H_B`→1).
    // Rationale: this is a verbose `System`-builder fixture (bounds, penalties,
    // inflow models, two hydros over five stages); the nested default-bounds /
    // default-penalties `fn` helpers keep the per-field defaults local to the
    // fixture. Splitting it or hoisting the helpers out would scatter the fixture
    // without making it clearer — the same allow-set the sibling `minimal_system`
    // fixtures in `setup` carry.
    #[allow(
        clippy::too_many_lines,
        clippy::items_after_statements,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn filling_study_templates() -> (crate::lp_builder::StageTemplates, usize, usize) {
        use chrono::NaiveDate;
        use cobre_core::scenario::InflowModel;
        use cobre_core::{
            Block, BlockMode, BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties,
            ContractStageBounds, DeficitSegment, EntityId, FillingConfig, Hydro,
            HydroGenerationModel, HydroPenalties, HydroStageBounds, HydroStagePenalties,
            LineStageBounds, LineStagePenalties, NcsStagePenalties, NoiseMethod,
            PenaltiesCountsSpec, PenaltiesDefaults, PumpingStageBounds, ResolvedBounds,
            ResolvedPenalties, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
            SystemBuilder, ThermalStageBounds,
        };
        use cobre_stochastic::par::precompute::PrecomputedPar;

        let n_hydros = 2_usize;
        // Five study stages (ids 0..=4) span every filling phase of both hydros.
        let n_stages = 5_usize;

        fn zero_hydro_penalties() -> HydroPenalties {
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
                inflow_nonnegativity_cost: 0.0,
            }
        }

        // A filling hydro with the given start/entry stage ids. ConstantProductivity
        // keeps the LP simple (no FPHA rows); the soft-floor/target slacks come from
        // the filling family, not the generation model.
        let filling_hydro = |id: i32, start_stage_id: i32, entry: i32| Hydro {
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: Some(entry),
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: Some(FillingConfig {
                start_stage_id,
                filling_min_rate_m3s: 0.0,
            }),
            penalties: zero_hydro_penalties(),
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let hydros = vec![filling_hydro(3, 0, 1), filling_hydro(4, 2, 4)];

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
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
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
            })
            .collect();

        // White-noise inflow models (non-zero std so the Operating/Filling
        // noise_scale is non-zero where the PreFilling zeroing is the contrast).
        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .flat_map(|s| {
                [EntityId(3), EntityId(4)]
                    .into_iter()
                    .map(move |hid| InflowModel {
                        hydro_id: hid,
                        stage_id: s as i32,
                        mean_m3s: 80.0,
                        std_m3s: 20.0,
                        ar_coefficients: vec![],
                        residual_std_ratio: 1.0,
                        annual: None,
                    })
            })
            .collect();

        fn default_hydro_bounds() -> HydroStageBounds {
            HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                max_diversion_m3s: None,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_penalties() -> HydroStagePenalties {
            HydroStagePenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
                filling_target_violation_cost: 100.0,
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
                inflow_nonnegativity_cost: 0.0,
            }
        }

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                    cost_per_mwh: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("filling_study_templates: valid system");

        // Every fixture stage has id >= 0, so the full slices match the study-stage
        // filter `build_stage_templates` applies internally (it filters id >= 0 too).
        let par_lp = PrecomputedPar::build(
            system.inflow_models(),
            system.stages(),
            &[EntityId(3), EntityId(4)],
            None,
        )
        .expect("white-noise PrecomputedPar build");
        let normal_lp = cobre_stochastic::normal::precompute::PrecomputedNormal::default();
        let hydro_models =
            crate::hydro_models::PrepareHydroModelsResult::default_from_system(&system);
        let resolved_params = crate::resolved_parameters::ResolvedParameters {
            per_param: vec![],
            id_to_slot: vec![],
        };

        let templates = crate::lp_builder::build_stage_templates(
            &system,
            InflowNonNegativityMethod::None,
            &par_lp,
            &normal_lp,
            &hydro_models.production,
            &hydro_models.evaporation,
            &resolved_params,
        )
        .expect("build_stage_templates: filling system");

        // Id-sorted: H_A (id 3) → 0, H_B (id 4) → 1.
        (templates, 0, 1)
    }

    /// Build a stage-0 [`OpeningTree`] for the two filling hydros (ids 3, 4) with
    /// `n_openings` openings, matching the inflow noise dimension the
    /// [`filling_study_templates`] system carries (`dim = 2`).
    ///
    /// The per-opening noise vector must have at least `n_hydros` entries because
    /// `lb_evaluate_stage_0` slices `raw_noise[..n_hydros]`; a 1-hydro tree (the
    /// sibling `simple_opening_tree`) would under-size it. Identity correlation
    /// between the two inflow entities keeps the tree shape trivial.
    fn filling_opening_tree(n_openings: usize) -> OpeningTree {
        use chrono::NaiveDate;
        use cobre_core::{
            EntityId,
            scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
            temporal::{
                Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
                StageStateConfig,
            },
        };
        use cobre_stochastic::correlation::resolve::DecomposedCorrelation;
        use std::collections::BTreeMap;

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
                branching_factor: n_openings,
                noise_method: NoiseMethod::Saa,
            },
        };

        let entity_order = vec![EntityId(3), EntityId(4)];
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g_inflow".to_string(),
                    entities: vec![
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId(3),
                        },
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId(4),
                        },
                    ],
                    matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                }],
            },
        );
        let corr_model = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };
        let decomposed = DecomposedCorrelation::build(&corr_model).unwrap();

        cobre_stochastic::tree::generate::generate_opening_tree(
            42,
            &[stage],
            2, // dim = 2 hydros
            &decomposed,
            &entity_order,
            cobre_stochastic::ClassDimensions {
                n_hydros: 2,
                n_load_buses: 0,
                n_ncs: 0,
            },
            &cobre_stochastic::tree::generate::OpeningTreeGenerationInputs::default(),
        )
        .unwrap()
    }

    /// AC1: the lower-bound `StageTemplate` (= forward/backward `templates[0]`)
    /// carries the filling row families present in the production template — the
    /// per-stage `filling_target`/`σ_fill` family at EVERY Filling stage (the
    /// per-stage widening, not only the terminal `entry − 1` stage) and the renamed
    /// `filled_min_storage_floor`/`σ^{v-}` operating-floor family — and the
    /// `PreFilling` hydro's stage-0 `noise_scale` entry is `0.0`.
    ///
    /// The lower bound's `LbEvalSpec.template` is bound to `stage_ctx.templates[0]`
    /// and its `noise_scale` to `stage_ctx.noise_scale` — the SAME objects the
    /// forward/backward passes load. So inspecting the `build_stage_templates`
    /// output IS inspecting what the lower bound consumes: there is no separate
    /// lower-bound template build to diverge.
    #[test]
    fn lower_bound_template_matches_forward_for_filling_stage() {
        let (templates, h_a, h_b) = filling_study_templates();
        let n_hydros = templates.n_hydros;
        assert_eq!(n_hydros, 2, "fixture has two filling hydros");

        // Stage 0 is H_A's terminal Filling stage: the per-stage geometry the
        // template was frozen with must carry the filling_target (σ_fill) row family
        // plus the σ_fill slack column.
        let geom0 = &templates.geometry_per_stage[0];
        assert!(
            !geom0.filling_target.is_empty(),
            "stage 0 (H_A entry − 1) must carry a filling_target/σ_fill row, got {:?}",
            geom0.filling_target
        );
        assert!(
            !geom0.filling_target_col.is_empty(),
            "stage 0 must carry the σ_fill slack column, got {:?}",
            geom0.filling_target_col
        );

        // The filling families must lie INSIDE the structural region of the same
        // `templates[0]` the lower bound loads — i.e. they are real rows/columns of
        // the LP `evaluate_lower_bound` solves, not phantom geometry. (A row at
        // `>= num_rows` would alias a cut row.)
        let tpl0 = &templates.templates[0];
        assert!(
            geom0.filling_target.end <= tpl0.num_rows,
            "filling rows must lie within templates[0].num_rows ({}): target {:?}",
            tpl0.num_rows,
            geom0.filling_target
        );
        assert!(
            geom0.filling_target_col.end <= tpl0.num_cols,
            "σ_fill column must lie within templates[0].num_cols ({}): {:?}",
            tpl0.num_cols,
            geom0.filling_target_col
        );

        // Per-stage widening: the `filling_target`/`σ_fill` family fires at EVERY
        // Filling stage (`start ≤ id < entry`), not only the terminal `entry − 1`
        // one. H_A (start 0, entry 1) is Filling at id 0 only; H_B (start 2, entry 4)
        // is Filling at ids 2 AND 3. A pre-widening build that emitted the family only
        // at the terminal stage would leave stages 2/3 empty — the regression pinned.
        let filling_stage_ids = [0_usize, 2, 3];
        for &fs in &filling_stage_ids {
            let geom = &templates.geometry_per_stage[fs];
            assert!(
                !geom.filling_target.is_empty(),
                "stage {fs} is a Filling stage and must carry a filling_target/σ_fill \
                 row family (per-stage widening), got {:?}",
                geom.filling_target
            );
            assert!(
                !geom.filling_target_col.is_empty(),
                "stage {fs} is a Filling stage and must carry the σ_fill slack column \
                 (per-stage widening), got {:?}",
                geom.filling_target_col
            );
            assert_eq!(
                geom.filling_target_hydro_indices.len(),
                geom.filling_target_col.len(),
                "stage {fs}: the sparse filling_target hydro-index list must be parallel \
                 to the σ_fill column range"
            );
        }
        // Contrast: the non-filling stages carry NO filling_target family, so the
        // assertion above is a real per-stage signal, not a tautology that would also
        // pass for an always-on family. Stage 1: H_A Operating (id 1 ≥ entry 1),
        // H_B PreFilling (id 1 < start 2). Stage 4: H_B Operating (id 4 ≥ entry 4).
        for &ns in &[1_usize, 4] {
            let geom = &templates.geometry_per_stage[ns];
            assert!(
                geom.filling_target.is_empty(),
                "stage {ns} is NOT a Filling stage for either hydro, so it must carry \
                 no filling_target/σ_fill row, got {:?}",
                geom.filling_target
            );
        }

        // A later Operating stage of H_B (id 4 == entry) must carry the
        // filled_min_storage_floor/σ^{v-} family — proving the third filling row family is
        // template-driven too (it just does not fire at stage 0 for these hydros).
        let geom4 = &templates.geometry_per_stage[4];
        assert!(
            !geom4.filled_min_storage_floor.is_empty(),
            "stage 4 (H_B Operating) must carry a filled_min_storage_floor/σ^{{v-}} row, got {:?}",
            geom4.filled_min_storage_floor
        );

        // PreFilling noise-scale zeroing: H_B is PreFilling at stage 0 (id 0 <
        // start 2), so its stage-0 noise_scale entry is exactly 0.0 — the
        // frozen-storage-identity freeze (the PreFilling row-pinning contract,
        // unrelated to the frozen-template LP mode) the lower bound inherits via
        // `spec.noise_scale`. H_A is Filling at stage 0 (not PreFilling), so its
        // entry is NOT zeroed — the contrast that makes the zeroing non-vacuous.
        let stage0_noise_a = templates.noise_scale[h_a];
        let stage0_noise_b = templates.noise_scale[h_b];
        assert_eq!(
            stage0_noise_b, 0.0,
            "PreFilling hydro H_B must have a zeroed stage-0 noise_scale, got {stage0_noise_b}"
        );
        assert!(
            stage0_noise_a > 0.0,
            "control: Filling hydro H_A keeps a non-zero stage-0 noise_scale ({stage0_noise_a}) \
             so the PreFilling zeroing is a real contrast, not vacuous"
        );
    }

    /// AC2: the lower-bound evaluation code references NO filling-specific symbol —
    /// filling structure arrives ONLY via the loaded template and the `noise_scale`
    /// vector, never via a hand-written per-opening patch.
    ///
    /// Modeled on the `lp_builder_never_references_dual_extraction` guard: a
    /// future "simplification" that hand-wires a filling patch into `lb_init_rank0`
    /// / `lb_evaluate_stage_0` (mirroring the NCS per-opening patch, which filling
    /// does NOT need because it is stage-deterministic) would re-introduce the
    /// filling symbols here and fail. The needles are assembled from char fragments
    /// so this guard's own source text does not contain the literals (else the test
    /// would flag itself), exactly as that guard's source-text scan does — the
    /// filling families are template-driven, not gated by hand-written code here.
    #[test]
    fn lower_bound_never_references_filling_gating() {
        // Filling-gating symbols assembled from fragments so they are absent from
        // this file's own bytes (the source text scanned below is this very file).
        // The needle set tracks the live filling row families: the per-stage
        // `σ_fill`/`filling_target` family and the renamed `filled_min_storage_floor`
        // (`σ^{v-}`) operating-floor family. There is no `filling_retention` needle:
        // the retention family was removed (the Filling phase keeps PAR noise via
        // `noise_scale`, not a retention row), so referencing it would itself be a
        // stale symbol — the rename/removal is mirrored here so the guard cannot rot
        // back to the abandoned family name.
        let needles: [String; 4] = [
            ["filling", "_phase"].concat(),
            ["Phase", "::"].concat(),
            ["sigma", "_fill"].concat(),
            ["filled", "_min_storage_floor"].concat(),
        ];

        // The full lower-bound module source. Only the PRODUCTION region (above the
        // `#[cfg(test)] mod tests`) is the code under test; the test module legitimately
        // names filling symbols (this guard, the fixture, the AC1/AC3 assertions), so
        // scanning the whole file would flag the tests themselves. Split on the test
        // module attribute and scan only the production prefix.
        let src = include_str!("lower_bound.rs");
        let prod_src = src
            .split("#[cfg(test)]")
            .next()
            .expect("module has a production region before the test module");

        let mut offenders: Vec<&str> = Vec::new();
        for needle in &needles {
            if prod_src.contains(needle.as_str()) {
                offenders.push(needle.as_str());
            }
        }
        assert!(
            offenders.is_empty(),
            "lower-bound production code must reference NO filling-gating symbol (filling \
             structure arrives only via the loaded template + noise_scale vector, never a \
             hand-written per-opening patch); offending symbols: {offenders:?}"
        );
    }

    /// AC3: `evaluate_lower_bound` returns a finite, valid bound for a filling-hydro
    /// system whose stage-0 LP carries the filling slack columns (`σ_fill`/`σ^{v-}`).
    ///
    /// Runs against the REAL solver (`ActiveSolver`) loading the production
    /// `templates[0]`, so the solve genuinely sees the filling-structured LP — a
    /// `MockSolver` would ignore the template and prove nothing about the LP shape.
    /// The assertion is structural/qualitative (finite, valid minorant; filling
    /// slack columns present); no numeric bound is pinned, because the HiGHS/CLP
    /// floating-point result is not reproducible across backends/hosts.
    #[test]
    fn evaluate_lower_bound_uses_filling_structured_template() {
        let (templates, _h_a, _h_b) = filling_study_templates();

        // Precondition: stage 0 carries the σ_fill slack column block, so the LP the
        // lower bound solves is the filling-structured LP (not an un-gated one).
        let geom0 = &templates.geometry_per_stage[0];
        assert!(
            !geom0.filling_target_col.is_empty(),
            "stage 0 must carry the σ_fill slack column for this test to be meaningful"
        );
        let tpl0 = &templates.templates[0];
        assert!(
            geom0.filling_target_col.end <= tpl0.num_cols,
            "σ_fill slack column must be a real column of templates[0]"
        );

        // The per-opening water-balance noise patch reads `spec.noise_scale` —
        // including H_B's PreFilling-zeroed stage-0 entry — exactly as the
        // forward/backward passes do, so the bound sees the same stage-0 constraints.
        let mut solver = cobre_solver::ActiveSolver::new().expect("ActiveSolver::new");
        let comm = LocalComm;

        // 2 storage states (one per hydro), white noise ⇒ max_par_order = 0.
        let state = crate::test_support::state_layout(2, 0);
        let fcf = make_fcf(templates.templates.len(), state.n_state);
        let initial_state = vec![0.0_f64; state.n_state];
        let mut patch_buf = PatchBuffer::new(state.hydro_count, state.max_par_order, 0, 0, 0, 0, 0);
        let opening_tree = filling_opening_tree(1);
        let rm = RiskMeasure::Expectation;

        let spec = LbEvalSpec {
            template: &templates.templates[0],
            base_row: templates.base_rows[0],
            noise_scale: &templates.noise_scale,
            n_hydros: 2,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            stage_id: 0,
            block_count: 1,
            ncs_generation: 0..0,
            z_inflow_row_start: templates.geometry_per_stage[0].z_inflow_row_start,
            inflow_method: &InflowNonNegativityMethod::None,
            disagg_weight: DisaggregationWeight::interior(0),
            zeta: 0.0,
        };
        let (mut row_batch, mut lb_scratch) = make_lb_locals();
        let mut bundle = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch,
            None,
            &mut lb_scratch,
        );

        let lb = evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &state,
            &crate::test_support::cut_state_projection(&state),
            &mut bundle,
            &spec,
            &comm,
        );

        let lb = lb.expect("filling-structured stage-0 LP must be feasible (penalty recourse)");
        assert!(
            lb.is_finite(),
            "lower bound over the filling-structured LP must be finite, got {lb}"
        );
    }

    /// Regression: the LB-eval consumer path (`evaluate_lower_bound` →
    /// `lb_evaluate_stage_0`) inherits the `PatchBuffer` single-owner fix —
    /// every travel-time bucket incoming column is pinned to `initial_state`, a
    /// value constant across the opening loop (not re-derived per opening), so a
    /// single-opening stage-0 evaluation already exercises the per-stage-visit
    /// pinning contract.
    #[test]
    fn evaluate_lower_bound_pins_transit_bucket_incoming_columns() {
        let state = crate::test_support::state_layout_with_transit_buckets(
            0,
            0,
            2,
            vec![(0, 0), (0, 1)],
            0,
            0,
            vec![],
        );
        assert_eq!(state.n_state, 2);

        let template =
            crate::test_support::transit_bucket_only_template(state.theta + 1, state.n_state);
        let fcf = make_fcf(1, state.n_state);
        let initial_state = vec![7.0_f64, 11.0];
        let mut patch_buf = PatchBuffer::new(0, 0, 0, 0, state.n_buckets, 0, 0);
        let opening_tree = simple_opening_tree(1);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;
        let mut solver = MockSolver::with_objectives(vec![0.0]);

        let spec = LbEvalSpec {
            template: &template,
            base_row: 0,
            noise_scale: &[],
            n_hydros: 0,
            opening_tree: &opening_tree,
            risk_measure: &rm,
            stochastic: None,
            n_load_buses: 0,
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            stage_id: 0,
            block_count: 0,
            ncs_generation: 0..0,
            z_inflow_row_start: 0,
            inflow_method: &InflowNonNegativityMethod::None,
            disagg_weight: DisaggregationWeight::interior(0),
            zeta: 0.0,
        };
        let (mut row_batch, mut lb_scratch) = make_lb_locals();
        let mut bundle = LbEvalScratchBundle::from_scratch_fields(
            &mut patch_buf,
            &mut row_batch,
            None,
            &mut lb_scratch,
        );

        evaluate_lower_bound(
            &mut solver,
            &fcf,
            &initial_state,
            &state,
            &crate::test_support::cut_state_projection(&state),
            &mut bundle,
            &spec,
            &comm,
        )
        .unwrap();

        let cp = bundle.patch_buf.state_col_patch_count();
        assert_eq!(
            cp, 2,
            "state_col_patch_count must equal n_buckets when N=0, A=0"
        );
        for (i, &expected) in initial_state.iter().enumerate() {
            let col = state.transit_buckets_in.start + i;
            let pos = bundle.patch_buf.col_indices[..cp]
                .iter()
                .position(|&c| c == col)
                .unwrap_or_else(|| panic!("bucket incoming column {col} must be pinned"));
            assert_eq!(bundle.patch_buf.col_lower[pos], expected);
            assert_eq!(bundle.patch_buf.col_upper[pos], expected);
        }
    }

    // ── Regime A day-weighted disaggregation ─────────────────────────────────

    /// The lower bound's boundary-stage blend uses the CONDITIONAL MEAN
    /// (`next_eta = 0`) for the source-B rate — the same arm the backward pass
    /// takes, since neither evaluation point holds a sampler. Hand-derives the
    /// expected blended noise RHS independently of `lb_evaluate_stage_0`'s own
    /// arithmetic: `noise_scale == 0.0` zeroes the pre-blend term, isolating
    /// `zeta * next_day_weight * (next_rate - anchor_rate)`; AR(0) with `std_m3s == 0.0` at
    /// both stages makes `anchor_rate`/`next_rate` exactly the two stages' means,
    /// independent of the opening's realized eta.
    // `clippy::too_many_lines`: the inline `System`/`StochasticContext` fixture is
    // one coherent scenario; splitting it into helpers would scatter the setup
    // the hand-derived assertions depend on.
    #[allow(clippy::too_many_lines)]
    #[test]
    fn test_disaggregation_lower_bound_applies_boundary_blend() {
        use cobre_core::{
            Bus, DeficitSegment, EntityId, Hydro, HydroGenerationModel, HydroPenalties,
            InflowModel, SystemBuilder,
            scenario::{
                CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
                SamplingScheme,
            },
            temporal::{
                Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
                StageStateConfig,
            },
        };
        use cobre_stochastic::context::{
            ClassSchemes, OpeningTreeInputs, build_stochastic_context,
        };
        use std::collections::BTreeMap;

        let mean_a = 100.0_f64;
        let mean_b = 150.0_f64;

        let bus = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let hydro = Hydro {
            id: EntityId(1),
            name: "H1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
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

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let make_stage = |idx: usize| Stage {
            index: idx,
            id: idx as i32,
            start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
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
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        };
        let stages: Vec<Stage> = (0..2).map(make_stage).collect();

        let inflow_models = vec![
            InflowModel {
                hydro_id: EntityId(1),
                stage_id: 0,
                mean_m3s: mean_a,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            },
            InflowModel {
                hydro_id: EntityId(1),
                stage_id: 1,
                mean_m3s: mean_b,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            },
        ];

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

        let stoch = build_stochastic_context(
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
        .unwrap();

        let state = crate::test_support::state_layout(1, 0);
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
            row_lower: vec![0.0, 0.0],
            row_upper: vec![0.0, 0.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        let initial_state = vec![0.0_f64; state.n_state];
        let mut patch_buf = PatchBuffer::new(state.hydro_count, state.max_par_order, 0, 0, 0, 0, 0);
        let opening_tree = simple_opening_tree(1);

        let disagg_weight = DisaggregationWeight {
            anchor_period: 0,
            next_period: Some(1),
            next_period_stage: Some(1),
            anchor_day_weight: 6.0 / 7.0,
            next_day_weight: 1.0 / 7.0,
        };
        let zeta = 1.0_f64;

        let spec = LbEvalSpec {
            template: &template,
            base_row: 1,
            noise_scale: &[0.0],
            n_hydros: 1,
            opening_tree: &opening_tree,
            risk_measure: &RiskMeasure::Expectation,
            stochastic: Some(&stoch),
            n_load_buses: 0,
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            stage_id: 0,
            block_count: 1,
            ncs_generation: 0..0,
            z_inflow_row_start: 0,
            inflow_method: &InflowNonNegativityMethod::None,
            disagg_weight,
            zeta,
        };

        let mut scratch = LbEvalScratch::new();
        let mut solver = MockSolver::with_objectives(vec![0.0]);

        lb_evaluate_stage_0(
            &mut solver,
            &spec,
            &mut patch_buf,
            &initial_state,
            &state,
            &mut scratch,
        )
        .unwrap();

        let expected = zeta * disagg_weight.next_day_weight * (mean_b - mean_a);
        assert!(
            (scratch.noise_buf[0] - expected).abs() < 1e-9,
            "boundary-stage LB noise RHS must equal the conditional-mean blend \
             {expected}, got {}",
            scratch.noise_buf[0]
        );
        assert!(
            (scratch.disagg_next_rate_buf[0] - mean_b).abs() < 1e-9,
            "disagg_next_rate_buf must equal the source-B conditional mean {mean_b}, got {}",
            scratch.disagg_next_rate_buf[0]
        );
        assert!(
            (scratch.z_inflow_rhs_buf[0] - mean_a).abs() < 1e-9,
            "z_inflow_rhs_buf must stay the pure anchor rate {mean_a} (never blended), got {}",
            scratch.z_inflow_rhs_buf[0]
        );
    }
}
