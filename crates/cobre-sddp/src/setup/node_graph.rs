//! Runtime node graph: node identity, canonical successor order, the
//! `node → pool` map, and per-node opening/weight views — the node-native
//! engine's single traversal structure, recorded as a
//! [`crate::context::TrainingContext`] field.
//!
//! Absent `nodes[]` this degenerates byte-exactly to one node per stage, 1:1
//! pools, and uniform `q`.

use std::collections::HashMap;

use cobre_core::temporal::PolicyGraph;
use cobre_io::StageIdResolver;
use cobre_stochastic::StochasticContext;

use crate::error::SddpError;

/// Which substrate a node's [`NodeOpenings`] view addresses. Both are
/// single-storage, read-only realization stores — `Generated` reads
/// [`StochasticContext::opening_tree`], `External` reads the standardized
/// external library for the node's stage (`cobre_stochastic::ExternalScenarioLibrary`,
/// via `ScenarioLibraries::training`) — never a copy of realization values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpeningSource {
    /// `offset..offset+len` are opening indices into this node's stage's
    /// `OpeningTree` block (`StochasticContext::opening_tree`).
    Generated,
    /// `offset` is a raw scenario column (`Node::realization_id`) in the
    /// standardized external library at this node's stage; `len` is always
    /// `1` — within-node weighted openings are deferred.
    External,
}

/// A node's Ω (opening set) view: `offset`/`len` address `source`'s own
/// per-stage ordinal space, never a copy of the realization values.
///
/// `q` is the uniform per-opening weight, computed as `1.0 / (len as f64)` —
/// the exact bit pattern
/// `chain_degeneracy_one_node_per_stage_1to1_pools_uniform_q_bit_pattern` pins
/// for a generated node's `len` openings, and (by the same formula,
/// `len == 1`) exactly `1.0` for a degenerate `External` view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeOpenings {
    /// Which substrate `offset`/`len` address.
    pub source: OpeningSource,
    /// Starting opening/scenario index within the node's stage, in `source`'s
    /// own per-stage ordinal space.
    pub offset: usize,
    /// Number of consecutive openings/scenarios in this node's Ω, starting at
    /// `offset`.
    pub len: usize,
    /// Uniform per-opening weight.
    pub q: f64,
}

impl NodeOpenings {
    #[allow(clippy::cast_precision_loss)]
    fn new(source: OpeningSource, offset: usize, len: usize) -> Self {
        Self {
            source,
            offset,
            len,
            q: 1.0 / (len as f64),
        }
    }
}

/// One out-edge from a node to a canonical-order successor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeSuccessor {
    /// Dense canonical position of the child in [`NodeGraph::nodes`] — not
    /// its declared JSON node id.
    pub child: usize,
    /// Normalized transition probability `P(n -> m)` — cobre-io's load-time,
    /// once-only Neumaier normalization (`normalize_out_edge_probabilities`);
    /// never re-normalized here.
    pub probability: f64,
}

/// One runtime node: its stage, pool, and Ω view. Discount is deliberately
/// NOT carried here — it stays per-stage, reached through `stage` (`t(n)`) on
/// `StageContext::cumulative_discount_factors`; a per-node array would store
/// `|nodes|` copies of `T` distinct values.
#[derive(Debug, Clone, Copy)]
pub struct NodeRuntime {
    /// Study-stage array index (`stage_data.stages[stage]`), not the JSON
    /// stage id.
    pub stage: usize,
    /// Pool id into `FutureCostFunction::pools` (pool *contents* — capacity,
    /// archives, basis tagging — are out of scope here). Leaves
    /// share one id; a node with successors always owns its own.
    pub pool_id: usize,
    /// This node's Ω view.
    pub openings: NodeOpenings,
}

/// The runtime node graph: every node in canonical (ascending declared
/// id) order, its pool assignment, its Ω view, and its canonically-ordered
/// successor list.
///
/// Absent `nodes[]`, `build_node_graph` synthesizes the byte-exact chain
/// degeneracy: one node per stage, `node_ids[t] = t as i32` (a synthetic
/// id — the source declared no `nodes[]`), pools 1:1.
#[derive(Debug, Clone)]
pub struct NodeGraph {
    /// Declared JSON node id at each canonical position (`node_ids.len() ==
    /// nodes.len()`).
    pub node_ids: Vec<i32>,
    /// Runtime nodes, canonical (ascending `node_ids`) order.
    pub nodes: Vec<NodeRuntime>,
    /// `successors[i]` is node `i`'s out-edge list, ascending child node id —
    /// the load-bearing canonical order (`CVaR`'s tie-break is index-order-sensitive).
    pub successors: Vec<Vec<NodeSuccessor>>,
    /// `max(pool_id) + 1`.
    pub n_pools: usize,
    /// `pool_stage[p]` is the study-stage index of the node(s) owning pool `p`
    /// — the base LP structure a per-pool frozen overlay composes onto
    /// (`freeze(templates[pool_stage[p]], active_cuts(p))`). Well-defined
    /// because every node sharing a pool shares one stage (a non-leaf pool has
    /// exactly one owner; the shared leaf pool's leaves are all terminal).
    /// The frozen overlay MUST index its base by this, never by a pool ordinal
    /// used as a stage: on a branching graph `pool_id != stage`. On a chain
    /// `pool_stage[t] == t`.
    pub pool_stage: Vec<usize>,
}

/// Build the runtime node graph from the validated, normalized `PolicyGraph`.
///
/// MUST run after `build_scenario_libraries`: an `External`-bound node's
/// Ω addresses the standardized library's raw scenario axis, so binding
/// earlier would race the library's own standardization.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if a transition or node names an id that
/// resolves to no declared node/stage — unreachable given upstream
/// structural validation, asserted rather than silently ignored.
pub(crate) fn build_node_graph(
    graph: &PolicyGraph,
    n_stages: usize,
    resolver: &StageIdResolver,
    stochastic: &StochasticContext,
) -> Result<NodeGraph, SddpError> {
    if graph.nodes.is_empty() {
        return Ok(build_chain_node_graph(
            graph, n_stages, resolver, stochastic,
        ));
    }
    build_declared_node_graph(graph, resolver, stochastic)
}

/// Chain degeneracy: one node per stage, pools 1:1, `q` the
/// `1.0 / (n as f64)` bit pattern (never a normalized accumulation), the
/// per-stage cumulative discount arrays untouched (this function never reads
/// or writes them).
fn build_chain_node_graph(
    graph: &PolicyGraph,
    n_stages: usize,
    resolver: &StageIdResolver,
    stochastic: &StochasticContext,
) -> NodeGraph {
    let tree = stochastic.opening_tree();
    let mut nodes = Vec::with_capacity(n_stages);
    let mut successors = Vec::with_capacity(n_stages);
    for t in 0..n_stages {
        let len = tree.n_openings(t);
        nodes.push(NodeRuntime {
            stage: t,
            pool_id: t,
            openings: NodeOpenings::new(OpeningSource::Generated, 0, len),
        });
        if t + 1 < n_stages {
            successors.push(vec![NodeSuccessor {
                child: t + 1,
                probability: chain_transition_probability(graph, resolver, t),
            }]);
        } else {
            successors.push(Vec::new());
        }
    }
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let node_ids: Vec<i32> = (0..n_stages as i32).collect();
    NodeGraph {
        node_ids,
        nodes,
        successors,
        n_pools: n_stages,
        pool_stage: (0..n_stages).collect(),
    }
}

/// The chain dialect's departing-edge probability for stage index `t`, read
/// off `graph.transitions` (already normalized to 1.0 within 1 ULP by
/// `normalize_out_edge_probabilities` at load time, when declared); `1.0`
/// when no transition departs the stage — the fully-implicit chain, where a
/// single deterministic out-edge is structurally required.
fn chain_transition_probability(graph: &PolicyGraph, resolver: &StageIdResolver, t: usize) -> f64 {
    let Some(stage_id) = resolver.id_at(t) else {
        return 1.0;
    };
    graph
        .transitions
        .iter()
        .find(|tr| tr.source_id == stage_id)
        .map_or(1.0, |tr| tr.probability)
}

/// The declared (`nodes[]` non-empty) node graph: canonical id order is
/// derived locally (never trusted from the caller's iteration order), the
/// `node → pool` map applies leaf sharing unconditionally under leaf-ness —
/// no boundary-source discriminator — and each node's Ω binds through
/// its own `realization_id` (a single external-library column, degenerate
/// `|Ω| = 1`) or the stage's generated set (`realization_id: None`).
///
/// # Errors
///
/// Returns [`SddpError::Validation`] on a duplicate node id, or on a transition
/// / node naming an id that resolves to no declared node/stage.
fn build_declared_node_graph(
    graph: &PolicyGraph,
    resolver: &StageIdResolver,
    stochastic: &StochasticContext,
) -> Result<NodeGraph, SddpError> {
    let tree = stochastic.opening_tree();

    let mut order: Vec<usize> = (0..graph.nodes.len()).collect();
    order.sort_by_key(|&i| graph.nodes[i].id);
    let node_ids: Vec<i32> = order.iter().map(|&i| graph.nodes[i].id).collect();
    // Reject duplicate node ids: the forward/backward warm-start apply guard
    // matches a stored basis to the current LP by `CapturedBasis.node_id`, so a
    // duplicate id would false-match one node's basis onto another's LP (the CLP
    // backend accepts a shape-mismatched basis silently). Ascending order makes
    // duplicates adjacent.
    if let Some(dup) = node_ids
        .iter()
        .zip(node_ids.iter().skip(1))
        .find_map(|(a, b)| (a == b).then_some(*a))
    {
        return Err(SddpError::Validation(format!(
            "node graph: duplicate node id {dup} declared more than once"
        )));
    }
    let id_to_position: HashMap<i32, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(pos, &id)| (id, pos))
        .collect();
    let n = order.len();

    // Out-edges per source, canonical (ascending child node id) order — the
    // load-bearing order CVaR's tie-break depends on.
    let mut out_edge_idx: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, tr) in graph.transitions.iter().enumerate() {
        let src_pos = *id_to_position.get(&tr.source_id).ok_or_else(|| {
            SddpError::Validation(format!(
                "node graph: transition source id {} names no declared node",
                tr.source_id
            ))
        })?;
        out_edge_idx[src_pos].push(i);
    }

    let mut successors: Vec<Vec<NodeSuccessor>> = Vec::with_capacity(n);
    let mut is_leaf = vec![true; n];
    for (pos, edges) in out_edge_idx.iter_mut().enumerate() {
        edges.sort_by_key(|&i| graph.transitions[i].target_id);
        let mut list = Vec::with_capacity(edges.len());
        for &i in edges.iter() {
            let tr = &graph.transitions[i];
            let child_pos = *id_to_position.get(&tr.target_id).ok_or_else(|| {
                SddpError::Validation(format!(
                    "node graph: transition target id {} names no declared node",
                    tr.target_id
                ))
            })?;
            list.push(NodeSuccessor {
                child: child_pos,
                probability: tr.probability,
            });
        }
        is_leaf[pos] = list.is_empty();
        successors.push(list);
    }

    // node -> pool: non-leaves own a pool each, in canonical node order; all
    // leaves then share ONE trailing pool id — leaf-ness is the whole
    // condition, unconditional, no boundary-source discriminator.
    let mut pool_id = vec![0usize; n];
    let mut next_pool = 0usize;
    for pos in 0..n {
        if !is_leaf[pos] {
            pool_id[pos] = next_pool;
            next_pool += 1;
        }
    }
    let has_leaf = is_leaf.iter().any(|&leaf| leaf);
    let shared_leaf_pool = next_pool;
    if has_leaf {
        for pos in 0..n {
            if is_leaf[pos] {
                pool_id[pos] = shared_leaf_pool;
            }
        }
    }
    let n_pools = if has_leaf { next_pool + 1 } else { next_pool };

    let mut nodes = Vec::with_capacity(n);
    for &pos in &order {
        let node = &graph.nodes[pos];
        let canonical_pos = id_to_position[&node.id];
        let stage = resolver.resolve(node.stage_id).ok_or_else(|| {
            SddpError::Validation(format!(
                "node graph: node {} stage id {} names no declared study stage",
                node.id, node.stage_id
            ))
        })?;
        let openings = match node.realization_id {
            Some(k) => {
                let offset = usize::try_from(k).map_err(|_| {
                    SddpError::Validation(format!(
                        "node graph: node {} realization_id {k} is negative",
                        node.id
                    ))
                })?;
                NodeOpenings::new(OpeningSource::External, offset, 1)
            }
            None => NodeOpenings::new(OpeningSource::Generated, 0, tree.n_openings(stage)),
        };
        nodes.push(NodeRuntime {
            stage,
            pool_id: pool_id[canonical_pos],
            openings,
        });
    }

    // pool -> stage: every node sharing a pool shares one stage (a non-leaf
    // pool has one owner; leaves sharing the terminal pool are all terminal),
    // so the last writer per pool is also the only stage value written. The
    // debug-only conflict check guards a future leaf-terminality relaxation or a
    // programmatic construction path that bypasses cobre-io validation.
    let mut pool_stage = vec![0usize; n_pools];
    #[cfg(debug_assertions)]
    let mut pool_stage_seen = vec![false; n_pools];
    for node in &nodes {
        #[cfg(debug_assertions)]
        {
            if pool_stage_seen[node.pool_id] {
                debug_assert_eq!(
                    pool_stage[node.pool_id], node.stage,
                    "node graph: pool {} owned by nodes at differing stages; every node \
                     sharing a pool must share one stage",
                    node.pool_id,
                );
            }
            pool_stage_seen[node.pool_id] = true;
        }
        pool_stage[node.pool_id] = node.stage;
    }

    // Every edge goes t -> t+1 (upstream structural validation); every
    // successor of a node therefore sits exactly one stage downstream —
    // asserted, not re-derived, since this is what makes pool dimension
    // well-defined by construction with no heterogeneity rule.
    for (pos, node) in nodes.iter().enumerate() {
        for succ in &successors[pos] {
            debug_assert_eq!(
                nodes[succ.child].stage,
                node.stage + 1,
                "node graph: successor stage must be exactly one downstream of its parent \
                 (t -> t+1) — parent stage {}, child stage {}",
                node.stage,
                nodes[succ.child].stage
            );
        }
    }

    Ok(NodeGraph {
        node_ids,
        nodes,
        successors,
        n_pools,
        pool_stage,
    })
}

/// Flatten `successors` into `out`, canonical (ascending child node id) order,
/// each entry the un-re-normalized product `P(n→child)·q_{child,ω}` — both
/// factors are already load-time-normalized on `node_graph`. `CVaR`'s tail
/// weighting is index-order-sensitive (sddp.md "Backward opening order is
/// warm-start-only"), so the caller-supplied canonical order is load-bearing.
///
/// Single owner for the backward pass
/// (`training::backward_pass_state::assemble_successor_outcome_weights`) and
/// the lower-bound root evaluation
/// (`training::lower_bound::assemble_outcome_weights`) — both delegate here so
/// the length precompute, weight formula, and fill order can never diverge
/// between the two call sites.
pub(crate) fn assemble_outcome_weights(
    node_graph: &NodeGraph,
    successors: &[NodeSuccessor],
    out: &mut Vec<f64>,
) {
    let expected_len = successor_outcome_count(node_graph, successors);
    out.clear();
    out.reserve(expected_len);
    for succ in successors {
        let child = &node_graph.nodes[succ.child];
        let weight = succ.probability * child.openings.q;
        out.extend(std::iter::repeat_n(weight, child.openings.len));
    }
    debug_assert_eq!(
        out.len(),
        expected_len,
        "assemble_outcome_weights: assembled outcome set must have length Σ_(m∈successors)|Ω_m|"
    );
}

/// `Σ_(m∈successors)|Ω_m|` — a node's flattened outcome-set length, the exact
/// quantity [`assemble_outcome_weights`] fills `out` to. Standalone so a
/// workspace-sizing caller (`max_successor_outcome_count`) shares this
/// flattening instead of re-deriving it.
fn successor_outcome_count(node_graph: &NodeGraph, successors: &[NodeSuccessor]) -> usize {
    successors
        .iter()
        .map(|s| node_graph.nodes[s.child].openings.len)
        .sum()
}

/// Maximum, over every node, of its own [`successor_outcome_count`] — the
/// backward pass's true per-node `n_openings` upper bound
/// (`compute_one_backward_node`'s `n_openings` is exactly this quantity for
/// whichever node it processes). A leaf's empty successor list contributes
/// `0`. Absent `nodes[]`, node `t`'s successor is stage `t+1`, so this is
/// `max` over stages `1..num_stages` of `tree.n_openings` — it excludes stage
/// 0's own opening count (no node's successor is ever the first stage), a
/// DIFFERENT quantity the lower-bound evaluation needs separately: a caller
/// sizing a buffer shared with that evaluation takes the max of both, rather
/// than assuming this function alone covers it.
pub(crate) fn max_successor_outcome_count(graph: &NodeGraph) -> usize {
    graph
        .successors
        .iter()
        .map(|succs| successor_outcome_count(graph, succs))
        .max()
        .unwrap_or(0)
}

/// Reverse-topological cut-sharing levels: cut-generating nodes (those with at
/// least one successor — a leaf produces no cut) grouped by stage, the outer
/// list descending by stage and each inner list ascending by canonical node
/// position. Sibling subtrees at one level are independent (nested per-node
/// risk), so a level is one cut-sharing barrier. Absent `nodes[]` every stage
/// carries one such node, so each level holds exactly one node (== that stage)
/// and the sweep reduces to the reversed stage loop.
pub(crate) fn backward_cut_levels(graph: &NodeGraph) -> Vec<Vec<usize>> {
    let cut_generating = |pos: usize| !graph.successors[pos].is_empty();
    let Some(max_stage) = graph
        .nodes
        .iter()
        .enumerate()
        .filter(|&(pos, _)| cut_generating(pos))
        .map(|(_, n)| n.stage)
        .max()
    else {
        return Vec::new();
    };

    let mut levels: Vec<Vec<usize>> = Vec::new();
    for stage in (0..=max_stage).rev() {
        let level: Vec<usize> = graph
            .nodes
            .iter()
            .enumerate()
            .filter(|&(pos, n)| n.stage == stage && cut_generating(pos))
            .map(|(pos, _)| pos)
            .collect();
        if !level.is_empty() {
            levels.push(level);
        }
    }
    levels
}

/// Number of fully-enumerated scenarios the node graph encodes: the sum over
/// every root→leaf path of the product of `openings.len` along that path
/// (`f(n) = |Ω_n| · Σ_child f(child)`, `f(leaf) = |Ω_leaf|`, then summed over
/// the predecessor-free roots). A leaf has no successors; a root has no
/// predecessor.
///
/// Overflow-safe: a `K^T` fan overflows `u64` for even a modest branching
/// factor and horizon, so every product and sum is checked and an overflow is
/// an `Err`, never a wrapped count. This owns the count only — the derived-count
/// reconciliation that rejects an over-large enumeration consumes it elsewhere.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] when the path-product-sum exceeds `u64`.
pub(crate) fn enumerated_scenario_count(graph: &NodeGraph) -> Result<u64, SddpError> {
    fn overflow_err() -> SddpError {
        SddpError::Validation(
            "enumerated scenario count exceeds u64: the policy graph's root→leaf \
             path-product-sum (Π |Ω| per path, summed over paths) overflows"
                .to_string(),
        )
    }

    let n = graph.nodes.len();
    let mut has_predecessor = vec![false; n];
    for succs in &graph.successors {
        for succ in succs {
            has_predecessor[succ.child] = true;
        }
    }

    // Resolve `f` in descending-stage order: every successor sits exactly one
    // stage downstream (t -> t+1), so a node's children are already resolved
    // when it is visited.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| graph.nodes[b].stage.cmp(&graph.nodes[a].stage));

    let mut subtree = vec![0_u64; n];
    for &pos in &order {
        let len = u64::try_from(graph.nodes[pos].openings.len).map_err(|_| overflow_err())?;
        let children = if graph.successors[pos].is_empty() {
            1_u64
        } else {
            let mut acc = 0_u64;
            for succ in &graph.successors[pos] {
                acc = acc
                    .checked_add(subtree[succ.child])
                    .ok_or_else(overflow_err)?;
            }
            acc
        };
        subtree[pos] = len.checked_mul(children).ok_or_else(overflow_err)?;
    }

    let mut total = 0_u64;
    for (pos, &has_pred) in has_predecessor.iter().enumerate() {
        if !has_pred {
            total = total.checked_add(subtree[pos]).ok_or_else(overflow_err)?;
        }
    }
    Ok(total)
}

/// Standard-deviation multiplier for [`sampled_visit_bound`]'s per-node
/// margin (mean + this many σ under the binomial approximation for a node's
/// realized visit count). A chain node's probability is exactly `1.0`, so its
/// variance term is exactly `0.0` regardless of this value — the chain
/// identity does not depend on the multiplier chosen here.
const VISIT_BOUND_SIGMA_MULTIPLIER: f64 = 3.0;

/// Per-pool construction-time visit floor under sampled forward-pass
/// selection: for each node `n`, `⌈F·P(n) + σ·√(F·P(n)·(1−P(n)))⌉` where
/// `P(n)` is the root→`n` product of (already load-time-normalized) edge
/// probabilities (`P(root) = 1`, `P(m) = Σ_{n: m∈n's successors} P(n)·prob(n→m)`),
/// summed over the nodes sharing a pool id and capped at `F` (only `F` passes
/// exist, so no pool can realize more visits than that). This is the
/// top-down dual of [`enumerated_scenario_count`]'s bottom-up opening-count
/// walk, propagated in the same ascending/descending-stage order.
///
/// A chain node's single successor normalizes to exactly `1.0`
/// (`chain_transition_probability`), so `P(n) = 1.0`, the variance term is
/// exactly `0.0`, and the floor is `F` bit-for-bit for every pool — the
/// sacred-parity chain identity falls out of this one formula with no shape
/// branch.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
pub(crate) fn sampled_visit_bound(graph: &NodeGraph, forward_passes: u32) -> Vec<u64> {
    let n = graph.nodes.len();
    let mut has_predecessor = vec![false; n];
    for succs in &graph.successors {
        for succ in succs {
            has_predecessor[succ.child] = true;
        }
    }

    // Ascending-stage order: every successor sits exactly one stage
    // downstream (t -> t+1), so a node's own probability is fully
    // accumulated from its predecessors before it propagates to its children.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| graph.nodes[i].stage);

    let mut prob = vec![0.0f64; n];
    for &i in &order {
        if !has_predecessor[i] {
            prob[i] = 1.0;
        }
        for succ in &graph.successors[i] {
            prob[succ.child] += prob[i] * succ.probability;
        }
    }

    let f = f64::from(forward_passes);
    let mut pool_sum = vec![0u64; graph.n_pools];
    for (i, node) in graph.nodes.iter().enumerate() {
        let p = prob[i];
        let mean = f * p;
        let variance = (f * p * (1.0 - p)).max(0.0);
        let margin = mean + VISIT_BOUND_SIGMA_MULTIPLIER * variance.sqrt();
        pool_sum[node.pool_id] += margin.ceil() as u64;
    }

    let cap = u64::from(forward_passes);
    pool_sum.into_iter().map(|s| s.min(cap)).collect()
}

/// Per-node exact prefix count under exhaustive (`enumerated`) forward-pass
/// selection: `π(n)`, the number of distinct root→node prefixes,
/// `π(n) = Σ_{(parent, n) edges} π(parent)·|Ω_parent|`, `π(root) = 1`, in
/// canonical node-position order. The per-node dual of
/// [`enumerated_scenario_count`]'s bottom-up opening walk;
/// [`enumerated_visit_bound`] sums it per pool. The enumerated forward driver
/// solves node `n` exactly `π(n)` times per iteration. Overflow-safe
/// (`checked_mul`/`checked_add`), mirroring [`enumerated_scenario_count`].
///
/// # Errors
///
/// Returns [`SddpError::Validation`] when a path-product-sum exceeds `u64`.
pub(crate) fn enumerated_node_visit_counts(graph: &NodeGraph) -> Result<Vec<u64>, SddpError> {
    fn overflow_err() -> SddpError {
        SddpError::Validation(
            "enumerated visit bound exceeds u64: the policy graph's root→node \
             path-product-sum overflows"
                .to_string(),
        )
    }

    let n = graph.nodes.len();
    let mut has_predecessor = vec![false; n];
    for succs in &graph.successors {
        for succ in succs {
            has_predecessor[succ.child] = true;
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| graph.nodes[i].stage);

    let mut prefix_count = vec![0_u64; n];
    for &i in &order {
        if !has_predecessor[i] {
            prefix_count[i] = 1;
        }
        let len = u64::try_from(graph.nodes[i].openings.len).map_err(|_| overflow_err())?;
        let contribution = prefix_count[i].checked_mul(len).ok_or_else(overflow_err)?;
        for succ in &graph.successors[i] {
            prefix_count[succ.child] = prefix_count[succ.child]
                .checked_add(contribution)
                .ok_or_else(overflow_err)?;
        }
    }
    Ok(prefix_count)
}

/// Per-pool exact visit count under exhaustive (`enumerated`) forward-pass
/// selection: [`enumerated_node_visit_counts`] summed over the nodes sharing
/// each pool id (leaf-sharing aggregates several nodes into one pool). The
/// top-down dual of [`enumerated_scenario_count`]'s bottom-up walk:
/// `Σ_leaf π(leaf)·|Ω_leaf| == enumerated_scenario_count(graph)`, and the
/// enumerated forward driver's single-rank per-iteration solve total is
/// `Σ` of this vector. Overflow-safe, mirroring that function's guard.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] when a path-product-sum exceeds `u64`.
pub(crate) fn enumerated_visit_bound(graph: &NodeGraph) -> Result<Vec<u64>, SddpError> {
    let prefix_count = enumerated_node_visit_counts(graph)?;
    let mut pool_sum = vec![0_u64; graph.n_pools];
    for (i, node) in graph.nodes.iter().enumerate() {
        pool_sum[node.pool_id] = pool_sum[node.pool_id]
            .checked_add(prefix_count[i])
            .ok_or_else(|| {
                SddpError::Validation(
                    "enumerated visit bound exceeds u64: the policy graph's root→node \
                     path-product-sum overflows"
                        .to_string(),
                )
            })?;
    }
    Ok(pool_sum)
}

// ── Driver-shared traversal helpers ─────────────────────────────────────────
//
// Structural resolution helpers shared by the forward training driver
// (`training::forward_pass_state`) and the simulation driver
// (`simulation::pipeline`/`simulation::state`) — both walk this graph the
// same way, so the resolution logic has one owner.

/// Ascending node-graph positions with `nodes[pos].stage == stage` — the
/// stage's alive frontier. Structural only (reads [`NodeGraph::nodes`]; no
/// sampling or enumeration selection): a synthesized or declared chain's
/// frontier is always a singleton, and below a recombination join a declared
/// graph's frontier carries more than one node.
pub(crate) fn stage_frontier(
    node_graph: &NodeGraph,
    stage: usize,
) -> impl Iterator<Item = usize> + '_ {
    node_graph
        .nodes
        .iter()
        .enumerate()
        .filter(move |(_, n)| n.stage == stage)
        .map(|(pos, _)| pos)
}

/// The study's sole root: stage `stage`'s one alive node. Used only for
/// stage-0 root resolution — every sampled trajectory starts here (a
/// multi-root graph is out of scope for the sampled walk) — so a stage
/// carrying more than one node here is a `debug_assert`-checked construction
/// invariant, not a silent default.
pub(crate) fn frontier_node(node_graph: &NodeGraph, stage: usize) -> usize {
    let mut frontier = stage_frontier(node_graph, stage);
    let node = frontier.next();
    debug_assert!(
        node.is_some(),
        "frontier_node: stage {stage} carries no alive node"
    );
    debug_assert!(
        frontier.next().is_none(),
        "frontier_node: stage {stage} carries more than one alive node"
    );
    // `stage` itself is a safe fallback consistent with the debug-checked invariants above.
    node.unwrap_or(stage)
}

/// Any one node at `stage` — the first in ascending canonical order. Unlike
/// [`frontier_node`], this tolerates a stage carrying several nodes: every
/// node at the terminal stage is a leaf (no edge crosses the horizon), and
/// leaves sharing a stage share ONE pool (this module's leaf-sharing rule),
/// so any representative resolves the same pool.
pub(crate) fn any_stage_node(node_graph: &NodeGraph, stage: usize) -> usize {
    let node = stage_frontier(node_graph, stage).next();
    debug_assert!(
        node.is_some(),
        "any_stage_node: stage {stage} carries no alive node"
    );
    node.unwrap_or(stage)
}

/// `node`'s single canonical predecessor, or `None` at a root. A node reached
/// by more than one edge is a recombination join — `debug_assert`-checked
/// (construction invariant), never silently resolved to an arbitrary
/// predecessor. A structural inspection utility, not a hot-path call — the
/// forward walk needs no parent lookup (every trajectory carries its own
/// previous-stage record directly); its only callers today are tests.
#[cfg(test)]
pub(crate) fn node_parent(node_graph: &NodeGraph, node: usize) -> Option<usize> {
    let mut candidates = node_graph
        .successors
        .iter()
        .enumerate()
        .filter(|(_, succs)| succs.iter().any(|s| s.child == node))
        .map(|(pos, _)| pos);
    let parent = candidates.next();
    debug_assert!(
        candidates.next().is_none(),
        "node_parent: node {node} has more than one predecessor (a recombination join)"
    );
    parent
}

/// A visit's node-Ω sub-range for the within-node opening draw: `node`'s own
/// `(offset, len)` when its view addresses the generated tree, or the full
/// stage range as a safe fallback when it instead binds to an
/// external-library column (a node-level identity orthogonal to the study's
/// class-level sampling scheme — validated upstream, not re-checked here). A
/// chain-degenerate node's Generated view already spans the whole stage, so
/// the chain path reduces to `(0, tree.n_openings(stage_idx))` either way.
pub(crate) fn node_opening_range(
    node_graph: &NodeGraph,
    node: usize,
    stochastic: &StochasticContext,
    stage_idx: usize,
) -> (usize, usize) {
    let openings = node_graph.nodes[node].openings;
    match openings.source {
        OpeningSource::Generated => (openings.offset, openings.len),
        OpeningSource::External => (0, stochastic.opening_tree().n_openings(stage_idx)),
    }
}

/// Single canonical predecessor of every node (`None` at a root), indexed by
/// canonical node position. The enumerated forward driver reads it to pin each
/// visited node's incoming state to its parent visit's outgoing state (every
/// edge is `t -> t+1`, so a node's parent sits exactly one stage upstream). A
/// node reached by more than one edge is a recombination join, which the graph
/// does not yet admit — `debug_assert`-checked, never silently resolved to an
/// arbitrary predecessor.
pub(crate) fn build_parent_map(node_graph: &NodeGraph) -> Vec<Option<usize>> {
    let mut parent = vec![None; node_graph.nodes.len()];
    for (pos, succs) in node_graph.successors.iter().enumerate() {
        for succ in succs {
            debug_assert!(
                parent[succ.child].is_none(),
                "build_parent_map: node {} has more than one predecessor (a recombination join)",
                succ.child
            );
            parent[succ.child] = Some(pos);
        }
    }
    parent
}

/// Canonical root→leaf path enumeration for the exhaustive (`enumerated`)
/// forward: one entry per fully-enumerated scenario path, in depth-first order
/// with roots and successors both taken in canonical (ascending node id)
/// order — the order `enumerated_scenario_count` counts and the exact upper
/// bound reduces over.
///
/// Every admitted enumerated graph has `|Ω_n| == 1` at every node (within-node
/// opening enumeration is deferred and rejected upstream), so a path is a
/// distinct root→leaf node sequence and the path count is
/// [`enumerated_scenario_count`].
pub(crate) struct EnumeratedForwardPaths {
    /// Leaf node position of each path, canonical order. Length = path count.
    pub leaf: Vec<usize>,
    /// `P(ℓ)` of each path, canonical order: the fixed-order root→leaf product
    /// of edge transition probabilities and within-node opening weights `q`
    /// (both already load-time-normalized). Length = path count.
    pub weight: Vec<f64>,
}

/// Enumerate every root→leaf path in canonical order (see
/// [`EnumeratedForwardPaths`]). Roots are the predecessor-free nodes in
/// ascending canonical position; each root's subtree is walked depth-first in
/// canonical successor order.
pub(crate) fn enumerate_forward_paths(node_graph: &NodeGraph) -> EnumeratedForwardPaths {
    fn walk(
        node_graph: &NodeGraph,
        node: usize,
        weight_before_node: f64,
        leaf: &mut Vec<usize>,
        weight: &mut Vec<f64>,
    ) {
        let w = weight_before_node * node_graph.nodes[node].openings.q;
        let succs = &node_graph.successors[node];
        if succs.is_empty() {
            leaf.push(node);
            weight.push(w);
            return;
        }
        for succ in succs {
            walk(node_graph, succ.child, w * succ.probability, leaf, weight);
        }
    }

    let n = node_graph.nodes.len();
    let mut has_predecessor = vec![false; n];
    for succs in &node_graph.successors {
        for succ in succs {
            has_predecessor[succ.child] = true;
        }
    }
    let mut paths = EnumeratedForwardPaths {
        leaf: Vec::new(),
        weight: Vec::new(),
    };
    for (root, &has_pred) in has_predecessor.iter().enumerate() {
        if !has_pred {
            walk(node_graph, root, 1.0, &mut paths.leaf, &mut paths.weight);
        }
    }
    paths
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cut::FutureCostFunction;
    use cobre_core::temporal::{Node as PolicyNode, PolicyGraphType, Transition};
    use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};
    use std::collections::HashMap as StdHashMap;

    fn transition(source_id: i32, target_id: i32, probability: f64) -> Transition {
        Transition {
            source_id,
            target_id,
            probability,
            annual_discount_rate_override: None,
        }
    }

    fn node(id: i32, stage_id: i32, realization_id: Option<i32>) -> PolicyNode {
        PolicyNode {
            id,
            stage_id,
            realization_id,
            label: None,
        }
    }

    fn empty_graph() -> PolicyGraph {
        PolicyGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: Vec::new(),
            nodes: Vec::new(),
            stage_discount_rate_overrides: StdHashMap::new(),
            season_map: None,
        }
    }

    /// A minimal, generated-only `StochasticContext` over `n_stages` stages
    /// with `n_hydros` hydro entities carrying independent noise and
    /// `branching_factor` openings per stage — enough to exercise
    /// `stochastic.opening_tree()` without any external library.
    #[allow(clippy::too_many_lines)]
    fn stochastic_context(
        n_stages: usize,
        n_hydros: usize,
        branching_factor: usize,
    ) -> StochasticContext {
        use chrono::NaiveDate;
        use cobre_core::entities::bus::{Bus, DeficitSegment};
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };
        use cobre_core::{
            BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, EntityId,
            HydroBlockBounds, HydroStageBounds, HydroStagePenalties, InflowModel, LineBlockBounds,
            LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
            PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder,
            ThermalBlockBounds, ThermalStageBounds,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        let hydros: Vec<Hydro> = (0..n_hydros)
            .map(|h| {
                let mut hydro = Hydro {
                    unit_groups: Vec::new(),
                    id: EntityId(10 + h as i32),
                    name: format!("H{h}"),
                    operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    downstream_id: None,
                    travel_time_hours: None,
                    entry_stage_id: None,
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
                    filling: None,
                    penalties: HydroPenalties {
                        spillage_cost: 0.01,
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
                hydro.declare_mirror_unit_group(EntityId(1));
                hydro
            })
            .collect();
        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: None,
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
            })
            .collect();
        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .flat_map(|i| {
                (0..n_hydros).map(move |h| InflowModel {
                    hydro_id: EntityId(10 + h as i32),
                    stage_id: i as i32,
                    mean_m3s: 80.0,
                    std_m3s: 20.0,
                    ar_coefficients: vec![],
                    residual_std_ratio: 1.0,
                    annual: None,
                })
            })
            .collect();
        let n_st = n_stages.max(1);
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: n_st,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 100.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 250.0,
                    max_diversion_m3s: None,
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
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
                n_stages: n_st,
            },
            &PenaltiesDefaults {
                hydro: HydroStagePenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 500.0,
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
            .expect("stochastic_context fixture: valid system");

        build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: None,
                load: None,
                ncs: None,
            },
        )
        .expect("stochastic_context fixture: build_stochastic_context")
    }

    // ── Chain degeneracy ────────────────────────────────────────────────────

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn chain_degeneracy_one_node_per_stage_1to1_pools_uniform_q_bit_pattern() {
        let n_stages = 4;
        let branching = 5;
        let stochastic = stochastic_context(n_stages, 1, branching);
        let graph = empty_graph();
        let study_stage_ids: Vec<i32> = (0..n_stages as i32).collect();
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);

        let ng = build_node_graph(&graph, n_stages, &resolver, &stochastic).unwrap();

        assert_eq!(ng.nodes.len(), n_stages, "one node per stage");
        assert_eq!(ng.n_pools, n_stages, "pools 1:1");
        for (t, n) in ng.nodes.iter().enumerate() {
            assert_eq!(n.stage, t);
            assert_eq!(n.pool_id, t, "chain pools are 1:1 (identity)");
            assert_eq!(n.openings.source, OpeningSource::Generated);
            assert_eq!(n.openings.len, branching);
            let expected_q = 1.0 / (branching as f64);
            assert_eq!(
                n.openings.q.to_bits(),
                expected_q.to_bits(),
                "q must be the exact 1.0/(n as f64) bit pattern, not a normalized accumulation"
            );
        }
        // Last stage is the terminal leaf: no successors.
        assert!(ng.successors[n_stages - 1].is_empty());
        for t in 0..n_stages - 1 {
            assert_eq!(ng.successors[t].len(), 1);
            assert_eq!(ng.successors[t][0].child, t + 1);
            assert_eq!(ng.successors[t][0].probability, 1.0);
        }
    }

    #[test]
    fn chain_degeneracy_does_not_touch_discount_arrays() {
        // NodeGraph carries no discount-related field at all — the type
        // itself is the proof; this test pins that `build_node_graph`'s
        // output is silent on discount by construction (no field to inspect).
        let n_stages = 3;
        let stochastic = stochastic_context(n_stages, 1, 2);
        let graph = empty_graph();
        let study_stage_ids: Vec<i32> = (0..n_stages as i32).collect();
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, n_stages, &resolver, &stochastic).unwrap();
        assert_eq!(ng.nodes.len(), n_stages);
        // Compile-time: NodeRuntime/NodeGraph have no discount field to read.
    }

    // ── Leaf pool sharing ───────────────────────────────────────────────────

    #[test]
    fn k_fan_leaves_share_one_pool_id_non_leaf_owns_its_own() {
        // Root at stage 0 (id 0), K=4 leaves at stage 1 (ids 1..4).
        let stochastic = stochastic_context(2, 1, 4);
        let graph = PolicyGraph {
            nodes: vec![
                node(0, 0, None),
                node(1, 1, None),
                node(2, 1, None),
                node(3, 1, None),
                node(4, 1, None),
            ],
            transitions: vec![
                transition(0, 1, 0.25),
                transition(0, 2, 0.25),
                transition(0, 3, 0.25),
                transition(0, 4, 0.25),
            ],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);

        let ng = build_node_graph(&graph, 2, &resolver, &stochastic).unwrap();

        // Root (position 0) owns its own pool.
        let root_pool = ng.nodes[0].pool_id;
        // The four leaves (positions 1..4) share exactly one pool id, distinct
        // from the root's.
        let leaf_pools: Vec<usize> = ng.nodes[1..=4].iter().map(|n| n.pool_id).collect();
        assert!(
            leaf_pools.iter().all(|&p| p == leaf_pools[0]),
            "all leaves must share one pool id: {leaf_pools:?}"
        );
        assert_ne!(
            root_pool, leaf_pools[0],
            "a node with successors never shares the leaf pool"
        );
        assert_eq!(ng.n_pools, 2, "one pool for the root, one shared leaf pool");
    }

    #[test]
    fn duplicate_node_id_is_rejected_at_build() {
        // Two nodes declared with the same id 1; the basis apply-guard keys on
        // node id, so a duplicate must be rejected before any solve.
        let stochastic = stochastic_context(2, 1, 2);
        let graph = PolicyGraph {
            nodes: vec![node(0, 0, None), node(1, 1, None), node(1, 1, None)],
            transitions: vec![transition(0, 1, 1.0)],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);

        let err = build_node_graph(&graph, 2, &resolver, &stochastic)
            .expect_err("a duplicate node id must be rejected");
        match err {
            SddpError::Validation(msg) => {
                assert!(
                    msg.contains("duplicate node id 1"),
                    "the rejection must name the duplicate id: {msg}"
                );
            }
            other => panic!("expected SddpError::Validation, got {other:?}"),
        }
    }

    #[test]
    fn pool_stage_maps_each_pool_to_its_owning_stage_over_a_k_fan() {
        // Root (stage 0, own pool) fans into two interior nodes at stage 1 (each
        // its own pool) with two leaves each at stage 2 (shared leaf pool).
        let stochastic = stochastic_context(3, 1, 2);
        let graph = PolicyGraph {
            nodes: vec![
                node(0, 0, None),
                node(1, 1, None),
                node(2, 1, None),
                node(3, 2, None),
                node(4, 2, None),
                node(5, 2, None),
                node(6, 2, None),
            ],
            transitions: vec![
                transition(0, 1, 0.5),
                transition(0, 2, 0.5),
                transition(1, 3, 0.5),
                transition(1, 4, 0.5),
                transition(2, 5, 0.5),
                transition(2, 6, 0.5),
            ],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1, 2];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, 3, &resolver, &stochastic).unwrap();

        assert_eq!(ng.pool_stage.len(), ng.n_pools);
        // Every pool's recorded stage must equal the stage of every node owning
        // it (leaf pool: all owners terminal → one shared stage).
        for (pos, n) in ng.nodes.iter().enumerate() {
            assert_eq!(
                ng.pool_stage[n.pool_id], n.stage,
                "pool_stage[pool of node {pos}] must equal that node's stage"
            );
        }
        // The shared leaf pool maps to the terminal stage 2.
        let leaf_pool = ng.nodes[3].pool_id;
        assert_eq!(ng.pool_stage[leaf_pool], 2);
    }

    #[test]
    fn pool_stage_chain_is_identity() {
        let n_stages = 4;
        let stochastic = stochastic_context(n_stages, 1, 3);
        let study_stage_ids: Vec<i32> = (0..n_stages as i32).collect();
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&empty_graph(), n_stages, &resolver, &stochastic).unwrap();
        assert_eq!(ng.pool_stage, vec![0, 1, 2, 3]);
    }

    #[test]
    fn leaf_sharing_guard_is_never_applied_to_a_node_with_successors() {
        // A 2-stage chain-like fan where stage 1 also has ITS OWN successor
        // (stage 2) — stage-1 node must own its own pool, not share.
        let stochastic = stochastic_context(3, 1, 2);
        let graph = PolicyGraph {
            nodes: vec![node(0, 0, None), node(1, 1, None), node(2, 2, None)],
            transitions: vec![transition(0, 1, 1.0), transition(1, 2, 1.0)],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1, 2];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);

        let ng = build_node_graph(&graph, 3, &resolver, &stochastic).unwrap();

        // Only node 2 (position 2) is a leaf; nodes 0 and 1 have successors
        // and must own distinct pools from each other and from the leaf pool.
        let pools: Vec<usize> = ng.nodes.iter().map(|n| n.pool_id).collect();
        assert_ne!(pools[0], pools[1], "non-leaf nodes never share a pool");
        assert_ne!(
            pools[1], pools[2],
            "the leaf pool is distinct from its parent's"
        );
        assert_eq!(ng.n_pools, 3);
    }

    // ── Canonical successor order ───────────────────────────────────────────

    #[test]
    fn successor_order_is_ascending_child_node_id_not_declaration_order() {
        // Declare the root's out-edges and the nodes themselves in reverse /
        // shuffled order; the runtime successor list must still come out
        // ascending by child node id.
        let stochastic = stochastic_context(2, 1, 3);
        let graph = PolicyGraph {
            // Nodes declared out of id order (5, 0, 3, 1).
            nodes: vec![
                node(5, 1, None),
                node(0, 0, None),
                node(3, 1, None),
                node(1, 1, None),
            ],
            // Transitions declared out of target-id order.
            transitions: vec![
                transition(0, 5, 0.2),
                transition(0, 1, 0.3),
                transition(0, 3, 0.5),
            ],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);

        let ng = build_node_graph(&graph, 2, &resolver, &stochastic).unwrap();

        // Canonical node order is ascending declared id: 0, 1, 3, 5.
        assert_eq!(ng.node_ids, vec![0, 1, 3, 5]);
        let root_pos = ng.node_ids.iter().position(|&id| id == 0).unwrap();
        let child_ids: Vec<i32> = ng.successors[root_pos]
            .iter()
            .map(|s| ng.node_ids[s.child])
            .collect();
        assert_eq!(
            child_ids,
            vec![1, 3, 5],
            "successors must be ascending child node id regardless of declaration order"
        );
    }

    // ── Ω views: no realization copy ────────────────────────────────────────

    #[test]
    fn generated_node_omega_is_a_view_offset_len_q_into_the_opening_tree() {
        let n_stages = 2;
        let branching = 6;
        let stochastic = stochastic_context(n_stages, 1, branching);
        let graph = empty_graph();
        let study_stage_ids: Vec<i32> = (0..n_stages as i32).collect();
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, n_stages, &resolver, &stochastic).unwrap();

        for n in &ng.nodes {
            assert_eq!(n.openings.source, OpeningSource::Generated);
            assert_eq!(n.openings.offset, 0);
            assert_eq!(
                n.openings.len,
                stochastic.opening_tree().n_openings(n.stage)
            );
        }
        // NodeOpenings carries no realization buffer field — inspection of
        // the type itself (offset/len/q, no `Vec<f64>`) is the "no second
        // copy" proof; std::mem::size_of pins it stays a small POD view.
        assert!(std::mem::size_of::<NodeOpenings>() <= 32);
    }

    #[test]
    fn external_node_omega_is_a_degenerate_view_at_realization_id() {
        let stochastic = stochastic_context(1, 1, 1);
        let graph = PolicyGraph {
            nodes: vec![node(0, 0, Some(7))],
            transitions: vec![],
            ..empty_graph()
        };
        let study_stage_ids = [0];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);

        let ng = build_node_graph(&graph, 1, &resolver, &stochastic).unwrap();

        assert_eq!(ng.nodes.len(), 1);
        let openings = ng.nodes[0].openings;
        assert_eq!(openings.source, OpeningSource::External);
        assert_eq!(openings.offset, 7);
        assert_eq!(openings.len, 1);
        assert_eq!(openings.q.to_bits(), 1.0_f64.to_bits());
    }

    // ── Discount: per-stage, never per-node ─────────────────────────────────

    #[test]
    fn nodes_at_the_same_stage_resolve_through_the_same_stage_index() {
        // Two sibling nodes at stage 1 (a K=2 fan) both carry `stage == 1`;
        // a discount consumer reading `cumulative_discount_factors[node.stage]`
        // necessarily reads the SAME array slot for both — no per-edge rate
        // is consulted here and no agreement check is performed (there is
        // nothing to agree over: one shared index).
        let stochastic = stochastic_context(2, 1, 2);
        let graph = PolicyGraph {
            nodes: vec![node(0, 0, None), node(1, 1, None), node(2, 1, None)],
            transitions: vec![transition(0, 1, 0.5), transition(0, 2, 0.5)],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, 2, &resolver, &stochastic).unwrap();

        let sibling_stages: Vec<usize> = ng.nodes[1..=2].iter().map(|n| n.stage).collect();
        assert_eq!(sibling_stages, vec![1, 1]);
    }

    // ── Rank-invariance (mpiexec -n1 vs -n2 substitute) ─────────────────────

    #[test]
    fn node_graph_construction_is_deterministic_across_independent_builds() {
        // NodeGraph construction reads nothing rank-count- or thread-count-
        // dependent (no RNG, no partition split); building it twice from
        // identical inputs — the same guarantee a real `mpiexec -n1` vs
        // `-n2` run relies on, since every rank receives the identical
        // already-broadcast `System`/config and independently reconstructs
        // an identical `StochasticContext` — must produce bitwise-identical
        // results.
        let stochastic_a = stochastic_context(3, 2, 3);
        let stochastic_b = stochastic_context(3, 2, 3);
        let graph = PolicyGraph {
            nodes: vec![
                node(0, 0, None),
                node(1, 1, None),
                node(2, 1, None),
                node(3, 2, None),
                node(4, 2, None),
                node(5, 2, None),
                node(6, 2, None),
            ],
            transitions: vec![
                transition(0, 1, 0.5),
                transition(0, 2, 0.5),
                transition(1, 3, 0.5),
                transition(1, 4, 0.5),
                transition(2, 5, 0.5),
                transition(2, 6, 0.5),
            ],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1, 2];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);

        let a = build_node_graph(&graph, 3, &resolver, &stochastic_a).unwrap();
        let b = build_node_graph(&graph, 3, &resolver, &stochastic_b).unwrap();

        assert_eq!(a.node_ids, b.node_ids);
        assert_eq!(a.n_pools, b.n_pools);
        for (na, nb) in a.nodes.iter().zip(b.nodes.iter()) {
            assert_eq!(na.stage, nb.stage);
            assert_eq!(na.pool_id, nb.pool_id);
            assert_eq!(na.openings.source, nb.openings.source);
            assert_eq!(na.openings.offset, nb.openings.offset);
            assert_eq!(na.openings.len, nb.openings.len);
            assert_eq!(na.openings.q.to_bits(), nb.openings.q.to_bits());
        }
        for (sa, sb) in a.successors.iter().zip(b.successors.iter()) {
            assert_eq!(sa.len(), sb.len());
            for (ea, eb) in sa.iter().zip(sb.iter()) {
                assert_eq!(ea.child, eb.child);
                assert_eq!(ea.probability.to_bits(), eb.probability.to_bits());
            }
        }
    }

    // ── sampled_visit_bound: chain identity ─────────────────────────────────

    #[test]
    fn sampled_visit_bound_chain_matches_forward_passes_over_a_sweep() {
        let n_stages = 4;
        let stochastic = stochastic_context(n_stages, 1, 5);
        let graph = empty_graph();
        let study_stage_ids: Vec<i32> = (0..n_stages as i32).collect();
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, n_stages, &resolver, &stochastic).unwrap();

        for &(warm_start, max_iterations, forward_passes) in &[
            (0u32, 1u64, 1u32),
            (5, 10, 100),
            (0, 100, 1),
            (1000, 1, 4096),
            (7, 37, 13),
            (12, 1, 1),
        ] {
            let visit_bound = sampled_visit_bound(&ng, forward_passes);
            assert_eq!(visit_bound.len(), ng.n_pools);
            for &vb in &visit_bound {
                assert_eq!(
                    vb,
                    u64::from(forward_passes),
                    "chain visit_bound must equal forward_passes exactly"
                );
            }

            let fcf = FutureCostFunction::new_per_pool(
                &vec![1; ng.n_pools],
                1,
                forward_passes,
                max_iterations,
                &vec![warm_start; ng.n_pools],
                &visit_bound,
            );
            for pool in &fcf.pools {
                assert_eq!(
                    pool.capacity,
                    warm_start as usize + max_iterations as usize * forward_passes as usize,
                    "chain capacity must equal warm_start + max_iterations * forward_passes exactly"
                );
            }
        }
    }

    #[test]
    fn sampled_visit_bound_declared_chain_matches_synthesized_chain() {
        let n_stages = 3;
        let forward_passes = 17u32;
        let stochastic_a = stochastic_context(n_stages, 1, 5);
        let stochastic_b = stochastic_context(n_stages, 1, 5);
        let study_stage_ids: Vec<i32> = (0..n_stages as i32).collect();
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);

        let ng_synthesized =
            build_node_graph(&empty_graph(), n_stages, &resolver, &stochastic_a).unwrap();

        // The same chain, declared explicitly (one node per stage, a single
        // successor at probability 1.0 each) rather than synthesized from an
        // empty `nodes[]`.
        let declared = PolicyGraph {
            nodes: vec![node(0, 0, None), node(1, 1, None), node(2, 2, None)],
            transitions: vec![transition(0, 1, 1.0), transition(1, 2, 1.0)],
            ..empty_graph()
        };
        let ng_declared = build_node_graph(&declared, n_stages, &resolver, &stochastic_b).unwrap();

        assert_eq!(
            sampled_visit_bound(&ng_synthesized, forward_passes),
            sampled_visit_bound(&ng_declared, forward_passes),
            "a declared chain and the synthesized one must derive identical visit bounds"
        );
    }

    // ── enumerated_visit_bound: exact prefix counts ─────────────────────────

    #[test]
    fn enumerated_visit_bound_k_fan_matches_hand_computed_prefix_counts() {
        // Root (stage 0, own pool) fans into K=4 leaves (stage 1, sharing ONE
        // pool per the leaf-sharing rule). branching_factor=1 gives every
        // node's own opening count exactly 1, so each root->leaf path
        // contributes exactly 1.
        let stochastic = stochastic_context(2, 1, 1);
        let graph = PolicyGraph {
            nodes: vec![
                node(0, 0, None),
                node(1, 1, None),
                node(2, 1, None),
                node(3, 1, None),
                node(4, 1, None),
            ],
            transitions: vec![
                transition(0, 1, 0.25),
                transition(0, 2, 0.25),
                transition(0, 3, 0.25),
                transition(0, 4, 0.25),
            ],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, 2, &resolver, &stochastic).unwrap();
        assert_eq!(ng.n_pools, 2, "one pool for the root, one shared leaf pool");

        let visit_bound = enumerated_visit_bound(&ng).unwrap();
        let root_pool = ng.nodes[0].pool_id;
        let leaf_pool = ng.nodes[1].pool_id;
        assert_eq!(
            visit_bound[root_pool], 1,
            "root: a single root prefix, π(root) = 1"
        );
        assert_eq!(
            visit_bound[leaf_pool], 4,
            "shared leaf pool: 4 leaves each contribute π(leaf_i) = 1, summed"
        );

        let warm_start = 5u32;
        let max_iterations = 10u64;
        let fcf = FutureCostFunction::new_per_pool(
            &vec![1; ng.n_pools],
            1,
            1,
            max_iterations,
            &vec![warm_start; ng.n_pools],
            &visit_bound,
        );
        assert_eq!(
            fcf.pools[root_pool].capacity,
            warm_start as usize + max_iterations as usize
        );
        assert_eq!(
            fcf.pools[leaf_pool].capacity,
            warm_start as usize + max_iterations as usize * 4
        );
    }

    #[test]
    fn enumerated_visit_bound_duality_with_enumerated_scenario_count() {
        // K-fan (branching_factor=1): duality holds with every leaf's own
        // contribution at 1.
        let stochastic_kfan = stochastic_context(2, 1, 1);
        let graph_kfan = PolicyGraph {
            nodes: vec![node(0, 0, None), node(1, 1, None), node(2, 1, None)],
            transitions: vec![transition(0, 1, 0.5), transition(0, 2, 0.5)],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng_kfan = build_node_graph(&graph_kfan, 2, &resolver, &stochastic_kfan).unwrap();
        assert_prefix_duality(&ng_kfan);

        // A non-DECOMP shape: the root carries its OWN within-node branching
        // (|Ω_root| = 3), so a child's prefix count is not simply the node
        // count -- the case that distinguishes the prefix-product formula
        // from a bare per-node count.
        let stochastic_branchy = stochastic_context(2, 1, 3);
        let graph_branchy = PolicyGraph {
            nodes: vec![node(0, 0, None), node(1, 1, None), node(2, 1, None)],
            transitions: vec![transition(0, 1, 0.5), transition(0, 2, 0.5)],
            ..empty_graph()
        };
        let ng_branchy =
            build_node_graph(&graph_branchy, 2, &resolver, &stochastic_branchy).unwrap();
        let visit_bound = enumerated_visit_bound(&ng_branchy).unwrap();
        assert_eq!(
            visit_bound[ng_branchy.nodes[1].pool_id], 6,
            "the shared leaf pool sums both children's π = |Ω_root| each, not divided between them"
        );
        assert_prefix_duality(&ng_branchy);
    }

    // ── backward_cut_levels: reverse-topological level decomposition ────────

    #[test]
    fn backward_cut_levels_chain_is_one_node_per_level_descending_stage() {
        let n_stages = 4;
        let stochastic = stochastic_context(n_stages, 1, 3);
        let study_stage_ids: Vec<i32> = (0..n_stages as i32).collect();
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&empty_graph(), n_stages, &resolver, &stochastic).unwrap();

        let levels = backward_cut_levels(&ng);
        // The terminal leaf (stage 3) generates no cut; stages 2,1,0 each hold
        // exactly one cut-generating node, descending.
        assert_eq!(levels, vec![vec![2], vec![1], vec![0]]);
    }

    #[test]
    fn backward_cut_levels_multi_node_level_is_ascending_node_id_descending_stage() {
        // Root (stage 0) fans into two INTERIOR nodes at stage 1 (each with its
        // own two leaves at stage 2). The stage-1 level therefore carries two
        // cut-generating nodes; the leaves generate none.
        let stochastic = stochastic_context(3, 1, 2);
        let graph = PolicyGraph {
            nodes: vec![
                node(0, 0, None),
                node(1, 1, None),
                node(2, 1, None),
                node(3, 2, None),
                node(4, 2, None),
                node(5, 2, None),
                node(6, 2, None),
            ],
            transitions: vec![
                transition(0, 1, 0.5),
                transition(0, 2, 0.5),
                transition(1, 3, 0.5),
                transition(1, 4, 0.5),
                transition(2, 5, 0.5),
                transition(2, 6, 0.5),
            ],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1, 2];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, 3, &resolver, &stochastic).unwrap();

        let levels = backward_cut_levels(&ng);
        // Stage 1 (both interior nodes, ascending position) before stage 0.
        assert_eq!(levels, vec![vec![1, 2], vec![0]]);
        for level in &levels {
            assert!(
                level.windows(2).all(|w| w[0] < w[1]),
                "each level's node positions must be strictly ascending"
            );
            let stages: Vec<usize> = level.iter().map(|&p| ng.nodes[p].stage).collect();
            assert!(
                stages.windows(2).all(|w| w[0] == w[1]),
                "every node in a level shares one stage"
            );
        }
    }

    /// `Σ_leaf π(leaf)·|Ω_leaf| == enumerated_scenario_count(graph)` — the
    /// prefix-count/scenario-count duality [`enumerated_visit_bound`] cites.
    /// Assumes every leaf sharing a pool carries the same `openings.len`
    /// (true of every fixture this helper is called on).
    fn assert_prefix_duality(graph: &NodeGraph) {
        let visit_bound = enumerated_visit_bound(graph).unwrap();
        let expected = enumerated_scenario_count(graph).unwrap();

        let mut leaf_len_by_pool: HashMap<usize, usize> = HashMap::new();
        for (i, n) in graph.nodes.iter().enumerate() {
            if graph.successors[i].is_empty() {
                leaf_len_by_pool.insert(n.pool_id, n.openings.len);
            }
        }
        let total: u64 = leaf_len_by_pool
            .iter()
            .map(|(&pool, &len)| visit_bound[pool] * len as u64)
            .sum();
        assert_eq!(
            total, expected,
            "Σ_leaf π(leaf)*|Ω_leaf| must equal enumerated_scenario_count"
        );
    }

    // ── enumerated forward traversal: per-node counts, paths, parents ───────

    /// A 3-stage K-fan (root -> fan -> leaf), `branching_factor` 1, non-uniform
    /// root out-edges `i / Σj` — the shape the enumerated forward driver walks.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn enumerated_k_fan(k: usize) -> NodeGraph {
        let stochastic = stochastic_context(3, 1, 1);
        let mut nodes = vec![node(0, 0, None)];
        let mut transitions = Vec::new();
        let total: f64 = (1..=k).map(|i| i as f64).sum();
        for i in 1..=k {
            let fan = i as i32;
            let leaf = fan + k as i32;
            nodes.push(node(fan, 1, None));
            nodes.push(node(leaf, 2, None));
            transitions.push(transition(0, fan, (i as f64) / total));
            transitions.push(transition(fan, leaf, 1.0));
        }
        let graph = PolicyGraph {
            nodes,
            transitions,
            ..empty_graph()
        };
        let study_stage_ids = [0, 1, 2];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        build_node_graph(&graph, 3, &resolver, &stochastic).unwrap()
    }

    #[test]
    fn enumerated_node_visit_counts_are_one_per_node_and_total_below_naive() {
        let k = 5usize;
        let ng = enumerated_k_fan(k);
        let counts = enumerated_node_visit_counts(&ng).unwrap();
        assert_eq!(counts.len(), ng.nodes.len());
        assert!(
            counts.iter().all(|&c| c == 1),
            "every |Ω|=1 tree node has exactly one root->node prefix: {counts:?}"
        );

        let total: u64 = enumerated_visit_bound(&ng).unwrap().into_iter().sum();
        assert_eq!(
            total,
            ng.nodes.len() as u64,
            "Σ enumerated_visit_bound sums π(n)=1 over every node = the node count"
        );
        // The dedup total beats naive per-path re-solving.
        let paths = enumerated_scenario_count(&ng).unwrap();
        let path_length = 3u64;
        assert_eq!(paths, k as u64, "one path per leaf on a |Ω|=1 K-fan");
        assert!(
            total < paths * path_length,
            "Σ enumerated_visit_bound ({total}) must be strictly below \
             enumerated_scenario_count * path_length ({})",
            paths * path_length
        );
    }

    #[test]
    fn enumerate_forward_paths_weights_are_canonical_edge_probabilities() {
        let k = 4usize;
        let ng = enumerated_k_fan(k);
        let paths = enumerate_forward_paths(&ng);
        assert_eq!(paths.leaf.len(), k, "one path per leaf");
        assert_eq!(paths.weight.len(), k);

        // Canonical DFS order visits the root's successors ascending; each
        // successor's edge probability IS that path's P(ℓ) (q = 1, single leaf
        // edge at prob 1).
        let root = (0..ng.nodes.len())
            .find(|&p| ng.nodes[p].stage == 0)
            .unwrap();
        let expected: Vec<f64> = ng.successors[root].iter().map(|s| s.probability).collect();
        for (got, want) in paths.weight.iter().zip(expected.iter()) {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "path weight must be the canonical edge probability, bit-for-bit"
            );
        }
        let sum: f64 = paths.weight.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "enumerated path probabilities must sum to 1 (got {sum})"
        );
    }

    #[test]
    fn build_parent_map_resolves_the_single_predecessor() {
        let ng = enumerated_k_fan(3);
        let parent = build_parent_map(&ng);
        let root = (0..ng.nodes.len())
            .find(|&p| ng.nodes[p].stage == 0)
            .unwrap();
        assert_eq!(parent[root], None, "the root has no predecessor");
        for (pos, node) in ng.nodes.iter().enumerate() {
            if node.stage == 0 {
                continue;
            }
            let p = parent[pos].expect("a non-root node has exactly one parent");
            assert_eq!(
                ng.nodes[p].stage,
                node.stage - 1,
                "a node's parent sits exactly one stage upstream"
            );
        }
    }
}
