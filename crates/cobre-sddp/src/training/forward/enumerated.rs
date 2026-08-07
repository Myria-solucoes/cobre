//! Enumerated all-paths forward execution engine.
//!
//! Selected by `selection = enumerated`; the sampled M-trajectory driver
//! (`forward_pass_state::run_forward_worker`) owns `selection = sampled` and is
//! untouched. The engine visits every distinct node once per iteration
//! (`enumerated_node_visit_counts`), pins each visit at its parent visit's
//! outgoing state, and reconstructs each root→leaf path's cost from its visited
//! nodes' stage costs — the exact bound `Σ_ℓ P(ℓ)·C(ℓ)` the caller reduces with
//! the compensated `Σ w·c` primitive.
//!
//! Determinism mirrors the by-node backward scheduler
//! (`.claude/rules/sddp.md` "By-node scheduler is warm-start-only"): a stage's
//! visits are claimed from a shared atomic counter in any order, written into a
//! per-node arena keyed by canonical node position, and aggregated in canonical
//! order — so the exact bound, the lower bound, and the generated cut set are
//! bit-identical across worker, thread, and rank counts.

use cobre_solver::{SolverInterface, StageTemplate};
use cobre_stochastic::{ClassSampleRequest, ForwardSampler, SampleRequest};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    dcs::{DcsParams, DcsSolveContext, build_initial_resident_set, lazy_solve_preloaded},
    error::SddpError,
    noise::{DownstreamAccumState, LagAccumState, accumulate_and_shift_lag_state},
    setup::node_graph::{
        EnumeratedPlan, NodeGraph, NodeId, NodePos, StageIdx, TypedVec, forward_solve_counts,
        node_opening_range, node_pinned_scenario, stage_frontier,
    },
    solver_stats::SolverStatsDelta,
    stage_solve::{StageInputs, fill_unscaled, run_stage_solve},
    training::claim_scatter::{ClaimCursor, canonical_scatter},
    training::stage_solve_prep::{
        InflowNoise, LoadNoise, StageSolvePrep, StageSolvePrepParams, StateSource,
    },
    trajectory::TrajectoryRecord,
    workspace::{BasisStore, CapturedBasis, ScratchBuffers, SolverWorkspace},
};

use super::write_capture_metadata;

/// Trajectory-carried scratch accumulators captured at a node's outgoing edge
/// and restored as its children's incoming state — the water-travel/derived-lag
/// counterpart of the LP-column state that rides the record's `state` field.
/// Empty/zero on a study with no PAR lags or downstream travel time (the K-fan
/// path), where every field below is length-0 or `0.0`.
#[derive(Clone, Default)]
struct AccumSnapshot {
    lag_accumulator: Vec<f64>,
    lag_weight_accum: Vec<f64>,
    downstream_accumulator: Vec<f64>,
    downstream_weight_accum: f64,
    downstream_completed_lags: Vec<f64>,
    downstream_n_completed: usize,
}

impl AccumSnapshot {
    fn capture_from(&mut self, ws_scratch: &ScratchBuffers) {
        self.lag_accumulator.clear();
        self.lag_accumulator
            .extend_from_slice(&ws_scratch.lag_accumulator);
        self.lag_weight_accum.clear();
        self.lag_weight_accum
            .extend_from_slice(&ws_scratch.lag_weight_accum);
        self.downstream_accumulator.clear();
        self.downstream_accumulator
            .extend_from_slice(&ws_scratch.downstream_accumulator);
        self.downstream_weight_accum = ws_scratch.downstream_weight_accum;
        self.downstream_completed_lags.clear();
        self.downstream_completed_lags
            .extend_from_slice(&ws_scratch.downstream_completed_lags);
        self.downstream_n_completed = ws_scratch.downstream_n_completed;
    }

    fn copy_into(&self, dst: &mut AccumSnapshot) {
        dst.lag_accumulator.clear();
        dst.lag_accumulator.extend_from_slice(&self.lag_accumulator);
        dst.lag_weight_accum.clear();
        dst.lag_weight_accum
            .extend_from_slice(&self.lag_weight_accum);
        dst.downstream_accumulator.clear();
        dst.downstream_accumulator
            .extend_from_slice(&self.downstream_accumulator);
        dst.downstream_weight_accum = self.downstream_weight_accum;
        dst.downstream_completed_lags.clear();
        dst.downstream_completed_lags
            .extend_from_slice(&self.downstream_completed_lags);
        dst.downstream_n_completed = self.downstream_n_completed;
    }

    fn restore_into(&self, ws_scratch: &mut ScratchBuffers) {
        ws_scratch.lag_accumulator.clear();
        ws_scratch
            .lag_accumulator
            .extend_from_slice(&self.lag_accumulator);
        ws_scratch.lag_weight_accum.clear();
        ws_scratch
            .lag_weight_accum
            .extend_from_slice(&self.lag_weight_accum);
        ws_scratch.downstream_accumulator.clear();
        ws_scratch
            .downstream_accumulator
            .extend_from_slice(&self.downstream_accumulator);
        ws_scratch.downstream_weight_accum = self.downstream_weight_accum;
        ws_scratch.downstream_completed_lags.clear();
        ws_scratch
            .downstream_completed_lags
            .extend_from_slice(&self.downstream_completed_lags);
        ws_scratch.downstream_n_completed = self.downstream_n_completed;
    }
}

/// One node's forward visit outcome, keyed by canonical node position in the
/// per-node arena and produced worker-locally in the claim loop.
#[derive(Clone, Default)]
struct NodeVisit {
    /// Canonical node position this visit solved.
    node: NodePos,
    /// Raw (pre-cumulative-discount) stage cost, the same quantity the sampled
    /// driver accumulates via `cum_d * stage_cost`.
    stage_cost: f64,
    node_id: NodeId,
    /// Outgoing LP state (length `n_state`) — a child visit's incoming state and
    /// the value scattered into every visiting path's record `state` field.
    out_state: Vec<f64>,
    accum: AccumSnapshot,
    /// Post-solve captured basis (frozen path only; `None` on the DCS path).
    basis: Option<CapturedBasis>,
}

/// Reused per-iteration mutable scratch for the enumerated forward engine —
/// the per-node visit arena and per-worker capture buffers. Meaningful only
/// while [`crate::setup::node_graph::Traversal`] is `Enumerated`; grows to
/// size against the graph-invariant [`EnumeratedPlan`] on the first call and
/// is reused in place thereafter, so no allocation occurs after warm-up.
/// Always present on `ForwardPassState` (never `Option`): dispatch is driven
/// solely by the resolved `Traversal`, read fresh each call, so there is no
/// separate flag this scratch's population could drift out of sync with.
#[derive(Default)]
pub(crate) struct EnumeratedForwardScratch {
    /// Per-node visit arena, indexed by canonical node position.
    arena: TypedVec<NodePos, NodeVisit>,
    /// Per-node `arena` population flag, reset each run.
    solved: TypedVec<NodePos, bool>,
    /// Nodes this rank must solve this run (on its assigned paths), reset each run.
    on_my_paths: TypedVec<NodePos, bool>,
    /// Representative local path index for each node — the smallest path in
    /// this rank's range visiting it, minus `fwd_offset`. One value, three
    /// consumption roles at its read sites: the noise-sampling seed
    /// (`local_m`/`global_scenario`), the basis-store row
    /// (`BasisStore::get`/`get_mut`), and the solve/stats bookkeeping index
    /// (`solve_forward_node`'s `scenario_index`).
    m_rep: TypedVec<NodePos, usize>,
    /// Reused per-stage claim unit list (canonical node order).
    stage_units: Vec<NodePos>,
    /// Per-worker capture buffers (worker-local; scattered into `arena` after
    /// each stage's parallel region).
    worker_captures: Vec<Vec<NodeVisit>>,
    /// Per-worker, per-stage solver-stats accumulator, reused across iterations.
    worker_stage_stats: Vec<Vec<SolverStatsDelta>>,
}

impl EnumeratedForwardScratch {
    /// Grow `arena`/`solved`/`on_my_paths`/`m_rep`/`stage_units` to `plan`'s
    /// node count (each `arena` slot's `out_state` to `n_state`); a no-op once
    /// sized, since neither dimension changes across a training run's
    /// iterations. `worker_captures`/`worker_stage_stats` grow separately,
    /// against the worker count, inside [`run_enumerated_forward`].
    fn ensure_sized(&mut self, plan: &EnumeratedPlan, n_state: usize) {
        let n_nodes = plan.parent.len();
        if self.arena.len() == n_nodes {
            return;
        }
        self.arena = (0..n_nodes)
            .map(|_| NodeVisit {
                node: NodePos(usize::MAX),
                stage_cost: 0.0,
                node_id: NodeId(0),
                out_state: vec![0.0; n_state],
                accum: AccumSnapshot::default(),
                basis: None,
            })
            .collect();
        self.solved = vec![false; n_nodes].into();
        self.on_my_paths = vec![false; n_nodes].into();
        self.m_rep = vec![0; n_nodes].into();
        self.stage_units = Vec::with_capacity(n_nodes);
    }
}

/// Read-only per-run captures the enumerated claim loop shares across workers.
pub(crate) struct EnumeratedParams<'a> {
    pub num_stages: usize,
    pub iteration: u64,
    pub fwd_offset: usize,
    pub local_forward_passes: usize,
    pub total_forward_passes: usize,
    pub terminal_has_boundary_cuts: bool,
    pub noise_dim: usize,
    pub initial_state: &'a [f64],
    pub lag_accum_seed: &'a [f64],
    pub lag_weight_seed: &'a [f64],
    pub ctx: &'a StageContext<'a>,
    pub frozen: &'a [StageTemplate],
    pub fcf: &'a FutureCostFunction,
    pub training_ctx: &'a TrainingContext<'a>,
    pub sampler: &'a ForwardSampler<'a>,
    pub dcs: Option<DcsParams>,
}

/// Aggregate return of the enumerated forward driver.
pub(crate) struct EnumeratedForwardResult {
    pub scenario_costs: Vec<f64>,
    pub lp_solves: u64,
    pub stage_stats: Vec<SolverStatsDelta>,
}

/// Solve one node's LP at the incoming state already installed on
/// `ws.current_state`, warm-started from `stored_basis` (matched by node id in
/// `run_stage_solve`); leaves `ws.current_state` holding the outgoing state and
/// returns the raw stage cost plus the captured basis (frozen path only).
///
/// Mirrors `forward_pass_state::run_forward_stage`'s solve/record/advance
/// sequence but keys the basis by node — `stored_basis` is read immutably
/// (shared across the claim loop), and the frozen-path capture is written into
/// `basis_out` in place (reusing its buffer via `get_or_insert_with`, so no
/// per-solve allocation), for the caller's sequential post-region scatter. So a
/// dynamically-claimed node never races the session basis store. The DCS path
/// captures no basis (leaving `basis_out` untouched), mirroring
/// `run_forward_stage`.
///
/// `scenario_index` (the caller's `EnumeratedForwardScratch::m_rep[node]`) is
/// this call's solve/stats-bookkeeping role only — `DcsSolveContext`'s and
/// `StageInputs`' own `scenario_index` field; the SAME underlying value's
/// basis-store and noise-sampling roles are the caller's concern
/// (`BasisStore::get`/`get_mut`, `global_scenario`).
///
/// # Errors
///
/// Propagates `SddpError::Infeasible`/`SddpError::Solver` from the LP solve.
// RATIONALE: one (node, prefix) LP solve threading disjoint partial borrows of the
// solver workspace, cut state, and basis store; the pin-noise-solve-capture sequence is
// one correctness-critical unit — extracting it would pass the whole workspace for no gain
// or re-introduce the argument count.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn solve_forward_node<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    params: &EnumeratedParams<'_>,
    node: NodePos,
    t: StageIdx,
    scenario_index: usize,
    raw_noise: &[f64],
    stored_basis: Option<&CapturedBasis>,
    basis_out: &mut Option<CapturedBasis>,
) -> Result<f64, SddpError> {
    let training_ctx = params.training_ctx;
    let ctx = params.ctx;
    let node_graph = training_ctx.node_graph;
    let pool_id = node_graph.nodes[node].pool_id;
    let node_id = node_graph.node_ids[node];
    let state = training_ctx.state;
    let horizon = training_ctx.horizon;
    let pool = &params.fcf.pools[pool_id];

    // Reset + reload per solve so the landed vertex cannot depend on which nodes
    // a worker solved before it (determinism across worker counts); on the DCS
    // path `run_stage_solve`/`lazy_solve_preloaded` loads the cut-free base
    // instead, so the frozen load is skipped — mirrors the sampled driver.
    ws.solver.reset_solver_state();
    if params.dcs.is_none() {
        ws.solver.load_model(&params.frozen[pool_id]);
    } else {
        ws.solver.load_model(ctx.template(t));
    }

    let prep_params = StageSolvePrepParams {
        state_source: StateSource(&ws.current_state),
        load_noise: LoadNoise::Present,
        inflow_noise: InflowNoise::Transform,
        raw_noise,
    };
    StageSolvePrep::run(
        &mut ws.solver,
        &mut ws.patch_buf,
        &mut ws.scratch,
        ctx,
        training_ctx,
        t,
        &prep_params,
    )?;
    if horizon.is_terminal(t.next().0) && !params.terminal_has_boundary_cuts {
        ws.solver.set_col_bounds(&[state.theta], &[0.0], &[0.0]);
    }

    let col_scale = &ctx.template(t).col_scale;
    let mut unscaled_primal: Vec<f64> = std::mem::take(&mut ws.scratch.unscaled_primal);

    let view_objective: f64 = if let Some(dcs_params) = params.dcs {
        build_initial_resident_set(
            pool,
            params.iteration,
            dcs_params.k2,
            &mut ws.backward_accum.dcs_initial_resident,
        );
        let dcs_ctx = DcsSolveContext {
            stage_index: t,
            scenario_index,
            iteration: Some(params.iteration),
            continue_carry: false,
            node_id,
        };
        lazy_solve_preloaded(
            &mut ws.solver,
            ctx.template(t),
            pool,
            state,
            &training_ctx.cut_state_layouts[pool_id],
            col_scale,
            None,
            &ws.backward_accum.dcs_initial_resident,
            &dcs_params,
            &mut ws.backward_accum.dcs_solve,
            dcs_ctx,
        )?;
        let view = ws.backward_accum.dcs_solve.result_view();
        let objective = view.objective;
        fill_unscaled(&mut unscaled_primal, view.primal, col_scale);
        objective
    } else {
        let inputs = StageInputs {
            stage_context: ctx,
            pool,
            stored_basis,
            stage_index: t,
            scenario_index,
            iteration: Some(params.iteration),
            node_id,
        };
        let view = run_stage_solve(ws, &inputs)?;
        let objective = view.objective;
        fill_unscaled(&mut unscaled_primal, view.primal, col_scale);
        objective
    };

    let d_t = ctx.discount_factor(t);
    let stage_cost = (view_objective - d_t * unscaled_primal[state.theta]) * ctx.cost_scale_factor;

    let lag_start = state.inflow_lags.start;
    let lag_len = state.hydro_count * state.max_par_order;
    ws.scratch.lag_matrix_buf.clear();
    ws.scratch
        .lag_matrix_buf
        .extend_from_slice(&ws.current_state[lag_start..lag_start + lag_len]);

    ws.current_state.clear();
    ws.current_state
        .extend_from_slice(&unscaled_primal[..state.n_state]);
    let stage_lag = ctx.stage_lag(t);
    let downstream_par_order = ws
        .scratch
        .downstream_completed_lags
        .len()
        .checked_div(ws.scratch.lag_accumulator.len())
        .unwrap_or(0);
    accumulate_and_shift_lag_state(
        &mut ws.current_state,
        &ws.scratch.lag_matrix_buf,
        &unscaled_primal,
        state,
        &stage_lag,
        &mut LagAccumState {
            accumulator: &mut ws.scratch.lag_accumulator,
            weight_accum: &mut ws.scratch.lag_weight_accum,
        },
        &mut DownstreamAccumState {
            accumulator: &mut ws.scratch.downstream_accumulator,
            weight_accum: &mut ws.scratch.downstream_weight_accum,
            completed_lags: &mut ws.scratch.downstream_completed_lags,
            n_completed: &mut ws.scratch.downstream_n_completed,
            par_order: downstream_par_order,
        },
    );
    ws.scratch.unscaled_primal = unscaled_primal;

    if params.dcs.is_none() {
        let basis_row_capacity = params.frozen[pool_id].num_rows;
        let cut_row_count = basis_row_capacity.saturating_sub(ctx.template(t).num_rows);
        // Reuse `basis_out`'s buffer if present (get_basis + write_capture_metadata
        // fully overwrite every field), so no per-solve allocation.
        let cap = basis_out.get_or_insert_with(|| {
            CapturedBasis::new(
                ctx.template(t).num_cols,
                basis_row_capacity,
                ctx.template(t).num_rows,
                cut_row_count,
                state.n_state,
                node_id,
            )
        });
        ws.solver.get_basis(&mut cap.basis);
        write_capture_metadata(
            cap,
            pool,
            ctx.template(t).num_rows,
            cut_row_count,
            &ws.current_state[..state.n_state],
            node_id,
        );
    }

    Ok(stage_cost)
}

/// Execute the enumerated all-paths forward for one iteration on this rank.
///
/// Solves every node on this rank's assigned root→leaf paths exactly once
/// (trunk nodes shared by all paths are replicated across ranks; a rank owns
/// the complete subtrees under its path range, so no cross-rank forward state
/// exchange is needed), scatters each node's outcome into the dense
/// `records[m * num_stages + t]` layout the backward pass reads, and returns the
/// per-path costs in canonical path order for the exact `Σ P(ℓ)·C(ℓ)` bound.
///
/// # Errors
///
/// Propagates `SddpError::Infeasible`/`SddpError::Solver` from any node solve.
// RATIONALE: the stage-synced outer loop, per-stage (node, prefix) claim distribution, and
// canonical-order arena scatter are one reproducibility-critical sequence; splitting it
// would fragment the claim-order-independence contract for no clarity gain.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub(crate) fn run_enumerated_forward<S>(
    plan: &EnumeratedPlan,
    scratch: &mut EnumeratedForwardScratch,
    workspaces: &mut [SolverWorkspace<S>],
    basis_store: &mut BasisStore,
    records: &mut [TrajectoryRecord],
    params: &EnumeratedParams<'_>,
) -> Result<EnumeratedForwardResult, SddpError>
where
    S: SolverInterface + Send,
{
    let node_graph = params.training_ctx.node_graph;
    let num_stages = params.num_stages;
    let n_state = params.training_ctx.state.n_state;
    let n_workers = workspaces.len().max(1);
    let path_range = params.fwd_offset..params.fwd_offset + params.local_forward_passes;

    scratch.ensure_sized(plan, n_state);

    // Reset the per-run arena membership. `on_my_paths`/`m_rep` derive from this
    // rank's path range; `solved` tracks arena population for the child lookup.
    for flag in &mut scratch.on_my_paths {
        *flag = false;
    }
    for flag in &mut scratch.solved {
        *flag = false;
    }
    // Mark every node on this rank's paths and record each node's representative
    // (smallest) local path index for its warm-start basis slot. Paths are
    // canonical, so the smallest visiting path in range is the first walk to
    // reach the node.
    let mut node_seq: Vec<NodePos> = Vec::with_capacity(num_stages);
    for m in path_range.clone() {
        node_seq.clear();
        walk_root_to_leaf(&plan.parent, plan.paths.leaf[m], num_stages, &mut node_seq);
        let local_m = m - params.fwd_offset;
        for &node in &node_seq {
            if !scratch.on_my_paths[node] {
                scratch.on_my_paths[node] = true;
                scratch.m_rep[node] = local_m;
            }
        }
    }

    // Grow per-worker scratch to the current worker count / stage span.
    if scratch.worker_captures.len() != n_workers {
        scratch.worker_captures = (0..n_workers).map(|_| Vec::new()).collect();
        scratch.worker_stage_stats = (0..n_workers)
            .map(|_| {
                (0..num_stages)
                    .map(|_| SolverStatsDelta::default())
                    .collect()
            })
            .collect();
    }
    for inner in &mut scratch.worker_stage_stats {
        for d in inner.iter_mut() {
            d.reset_in_place();
        }
    }

    let mut lp_solves = 0u64;
    for t in (0..num_stages).map(StageIdx) {
        // Units: the nodes at stage `t` on this rank's paths, canonical order.
        scratch.stage_units.clear();
        for node in stage_frontier(node_graph, t) {
            if scratch.on_my_paths[node] {
                scratch.stage_units.push(node);
            }
        }
        if scratch.stage_units.is_empty() {
            continue;
        }

        let cursor = ClaimCursor::new(scratch.stage_units.len());
        let capture_basis = params.dcs.is_none();
        // Immutable borrows for the parallel region: the previous stage's arena
        // (children read parents), the plan, and the session basis store (warm
        // start read only — captures are written into per-worker slots for a
        // sequential post-region scatter).
        let arena = &scratch.arena;
        let solved = &scratch.solved;
        let m_rep = &scratch.m_rep;
        let stage_units = &scratch.stage_units;
        let parent = &plan.parent;
        let store: &BasisStore = basis_store;

        let worker_counts: Vec<Result<usize, SddpError>> = workspaces
            .par_iter_mut()
            .zip(scratch.worker_captures.par_iter_mut())
            .zip(scratch.worker_stage_stats.par_iter_mut())
            .map(|((ws, captures), stage_stats)| {
                enumerated_stage_worker(
                    ws,
                    captures,
                    stage_stats,
                    &cursor,
                    t,
                    params,
                    stage_units,
                    parent,
                    arena,
                    solved,
                    m_rep,
                    store,
                )
            })
            .collect();

        // Propagate the first worker error by value (`SddpError` is not `Clone`),
        // mirroring `by_node_finish`'s `res?` handling.
        let counts = worker_counts
            .into_iter()
            .collect::<Result<Vec<usize>, SddpError>>()?;
        lp_solves += counts.iter().map(|&c| c as u64).sum::<u64>();

        // Sequential scatter: copy each worker's captures into the per-node arena
        // and swap each captured basis into the session store, keyed by canonical
        // node — claim/worker order cannot reach the arena, so the outcome is
        // worker-count invariant. Copying (not moving) keeps every worker-slot
        // buffer alive for reuse next stage.
        for (w, i) in canonical_scatter(&counts) {
            let node = scratch.worker_captures[w][i].node;
            // The basis-store role of `m_rep`: the row `BasisStore::get_mut`
            // swaps this node's freshly captured basis into.
            let basis_scenario = scratch.m_rep[node];
            {
                let src = &scratch.worker_captures[w][i];
                let dst = &mut scratch.arena[node];
                dst.node = node;
                dst.stage_cost = src.stage_cost;
                dst.node_id = src.node_id;
                dst.out_state.clear();
                dst.out_state.extend_from_slice(&src.out_state);
                src.accum.copy_into(&mut dst.accum);
            }
            scratch.solved[node] = true;
            if capture_basis {
                std::mem::swap(
                    basis_store.get_mut(basis_scenario, node),
                    &mut scratch.worker_captures[w][i].basis,
                );
            }
        }
    }

    // Aggregate per-stage stats across workers.
    let mut stage_stats: Vec<SolverStatsDelta> = (0..num_stages)
        .map(|_| SolverStatsDelta::default())
        .collect();
    for inner in &scratch.worker_stage_stats {
        for (dst, src) in stage_stats.iter_mut().zip(inner.iter()) {
            SolverStatsDelta::accumulate_into(dst, src);
        }
    }

    // Scatter the arena into the dense record layout and reconstruct per-path
    // costs in canonical path order (ascending trajectory id within this rank).
    let mut scenario_costs = Vec::with_capacity(params.local_forward_passes);
    let mut seq: Vec<NodePos> = Vec::with_capacity(num_stages);
    for m in path_range.clone() {
        let local_m = m - params.fwd_offset;
        seq.clear();
        walk_root_to_leaf(&plan.parent, plan.paths.leaf[m], num_stages, &mut seq);
        debug_assert_eq!(seq.len(), num_stages, "path must span every stage");
        let mut cost = 0.0_f64;
        for (t, &node) in seq.iter().enumerate() {
            debug_assert!(scratch.solved[node], "node on an owned path was not solved");
            let visit = &scratch.arena[node];
            let cum_d = params
                .ctx
                .cumulative_discount_factors
                .get(t)
                .copied()
                .unwrap_or(1.0);
            cost += cum_d * visit.stage_cost;
            let rec = &mut records[local_m * num_stages + t];
            rec.primal.clear();
            rec.dual.clear();
            rec.stage_cost = visit.stage_cost;
            rec.node_id = visit.node_id;
            rec.state.clear();
            rec.state.extend_from_slice(&visit.out_state);
            debug_assert_eq!(
                rec.state.len(),
                n_state,
                "record state must be n_state long"
            );
        }
        scenario_costs.push(cost);
    }

    // Single-rank pins the dedup factor: solving every node once totals
    // `Σ forward_solve_counts`. Multi-rank replicates trunk nodes per rank, so
    // the equality holds only when this rank owns every path.
    #[cfg(debug_assertions)]
    if params.local_forward_passes == params.total_forward_passes {
        let expected = expected_single_rank_solves(node_graph)?;
        debug_assert_eq!(
            lp_solves, expected,
            "single-rank enumerated solve total must equal Σ forward_solve_counts"
        );
    }

    Ok(EnumeratedForwardResult {
        scenario_costs,
        lp_solves,
        stage_stats,
    })
}

/// Walk a leaf up to its root via the single-predecessor parent map, writing the
/// root→leaf node sequence (ascending stage) into `out`.
fn walk_root_to_leaf(
    parent: &TypedVec<NodePos, Option<NodePos>>,
    leaf: NodePos,
    num_stages: usize,
    out: &mut Vec<NodePos>,
) {
    out.clear();
    let mut cur = Some(leaf);
    while let Some(node) = cur {
        out.push(node);
        cur = parent[node];
    }
    out.reverse();
    debug_assert_eq!(
        out.len(),
        num_stages,
        "root→leaf path must visit exactly one node per stage"
    );
}

/// One worker's claim loop over a stage's node units. Claims units from the
/// shared atomic counter in any order; each solved node's outcome is written into
/// a reused worker-local capture slot for the caller's sequential, canonical-order
/// scatter. Returns the count of slots this worker filled (`captures[..count]`).
/// The capture slots — and every buffer they hold (`out_state`, the accumulator
/// snapshot, the captured basis) — persist across stages and iterations, so no
/// allocation occurs after warm-up.
// RATIONALE: the per-worker claim loop threads disjoint partial borrows of one
// &mut SolverWorkspace plus the shared claim counter and per-node arena; extracting would
// pass the whole workspace for no gain or re-introduce the argument count.
#[allow(clippy::too_many_arguments)]
fn enumerated_stage_worker<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    captures: &mut Vec<NodeVisit>,
    stage_stats: &mut [SolverStatsDelta],
    cursor: &ClaimCursor,
    t: StageIdx,
    params: &EnumeratedParams<'_>,
    stage_units: &[NodePos],
    parent: &TypedVec<NodePos, Option<NodePos>>,
    arena: &TypedVec<NodePos, NodeVisit>,
    solved: &TypedVec<NodePos, bool>,
    m_rep: &TypedVec<NodePos, usize>,
    store: &BasisStore,
) -> Result<usize, SddpError> {
    let node_graph = params.training_ctx.node_graph;
    let state_space = params.training_ctx.state;
    let n_state = state_space.n_state;

    let mut raw_noise_buf = std::mem::take(&mut ws.scratch.raw_noise_buf);
    raw_noise_buf.resize(params.noise_dim, 0.0_f64);
    let mut perm_scratch = std::mem::take(&mut ws.scratch.perm_scratch);
    perm_scratch.resize(params.total_forward_passes.max(1), 0_usize);

    #[allow(clippy::cast_possible_truncation)]
    let total_scenarios_u32 = params.total_forward_passes as u32;

    let mut count = 0usize;
    while let Some(u) = cursor.claim() {
        let node = stage_units[u];
        let local_m = m_rep[node];
        let global_scenario = params.fwd_offset + local_m;

        // Install the incoming state: the parent visit's outgoing state (already
        // scattered in the previous stage's sequential pass), or the initial
        // state at a root.
        ws.current_state.clear();
        if let Some(p) = parent[node] {
            debug_assert!(solved[p], "parent visit must be solved before its child");
            ws.current_state.extend_from_slice(&arena[p].out_state);
            arena[p].accum.restore_into(&mut ws.scratch);
        } else {
            ws.current_state.extend_from_slice(params.initial_state);
            seed_root_accumulators(ws, params);
        }

        #[allow(clippy::cast_possible_truncation)]
        let (i32_it, s32, t32) = (params.iteration as u32, global_scenario as u32, t.0 as u32);
        let (node_opening_offset, node_opening_len) = node_opening_range(node_graph, node);
        let pinned_scenario = node_pinned_scenario(node_graph, node);

        if parent[node].is_none() {
            let class_req = ClassSampleRequest {
                iteration: i32_it,
                scenario: s32,
                stage: t32,
                stage_idx: t.0,
                total_scenarios: total_scenarios_u32,
                noise_group_id: 0,
                node_opening_offset,
                node_opening_len,
                pinned_scenario,
            };
            params.sampler.apply_initial_state(
                &class_req,
                &mut ws.current_state,
                state_space.inflow_lags.start,
            );
        }

        let noise = params.sampler.sample(SampleRequest {
            iteration: i32_it,
            scenario: s32,
            stage: t32,
            stage_idx: t.0,
            noise_buf: &mut raw_noise_buf,
            perm_scratch: &mut perm_scratch,
            total_scenarios: total_scenarios_u32,
            noise_group_id: params.ctx.noise_group_id_at(t),
            node_opening_offset,
            node_opening_len,
            pinned_scenario,
        })?;

        // Reuse the capture slot at `count`, growing only until the widest stage's
        // frontier is covered; every buffer inside is then reused across stages
        // and iterations.
        if captures.len() <= count {
            captures.push(NodeVisit::default());
        }
        let stored = store.get(local_m, node);
        let stats_before = ws.solver.statistics();
        // `noise`/`stored` borrow the worker-local `raw_noise_buf` and the shared
        // read-only basis store, never `ws`, so both coexist with the `&mut ws`
        // solve — no per-solve copy needed (mirrors the sampled driver).
        let raw_noise = noise.as_slice();
        let stage_cost = solve_forward_node(
            ws,
            params,
            node,
            t,
            local_m,
            raw_noise,
            stored,
            &mut captures[count].basis,
        )?;
        let delta = SolverStatsDelta::from_snapshots(&stats_before, &ws.solver.statistics());
        SolverStatsDelta::accumulate_into(&mut stage_stats[t.0], &delta);

        let visit = &mut captures[count];
        visit.node = node;
        visit.stage_cost = stage_cost;
        visit.node_id = node_graph.node_ids[node];
        visit.out_state.clear();
        visit
            .out_state
            .extend_from_slice(&ws.current_state[..n_state]);
        visit.accum.capture_from(&ws.scratch);
        count += 1;
    }

    ws.scratch.raw_noise_buf = raw_noise_buf;
    ws.scratch.perm_scratch = perm_scratch;
    Ok(count)
}

/// Seed the trajectory-carried scratch accumulators at a root visit, matching
/// the sampled driver's stage-0 seed (`DerivedInflowSeeds` copy or zero-fill).
fn seed_root_accumulators<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    params: &EnumeratedParams<'_>,
) {
    if params.lag_accum_seed.is_empty() {
        ws.scratch.lag_accumulator.fill(0.0);
        ws.scratch.lag_weight_accum.fill(0.0);
    } else {
        ws.scratch.lag_accumulator[..params.lag_accum_seed.len()]
            .copy_from_slice(params.lag_accum_seed);
        ws.scratch.lag_weight_accum[..params.lag_weight_seed.len()]
            .copy_from_slice(params.lag_weight_seed);
    }
    ws.scratch.downstream_accumulator.fill(0.0);
    ws.scratch.downstream_weight_accum = 0.0;
    ws.scratch.downstream_completed_lags.fill(0.0);
    ws.scratch.downstream_n_completed = 0;
}

/// The single-rank per-iteration solve total: `Σ forward_solve_counts`.
/// Exposed so the caller can `debug_assert` the enumerated engine's realized
/// single-rank solve count against it (the dedup scale invariant).
///
/// # Errors
///
/// Propagates the `u64` path-product overflow guard.
pub(crate) fn expected_single_rank_solves(node_graph: &NodeGraph) -> Result<u64, SddpError> {
    Ok(forward_solve_counts(node_graph)?.into_iter().sum())
}
