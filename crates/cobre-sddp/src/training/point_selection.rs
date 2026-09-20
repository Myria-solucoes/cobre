use crate::{
    cut::pool::CutPool,
    indexer::{CutSlot, CutStateProjection},
    setup::{NodeId, NodePos},
};
use cobre_io::config::training::TrialPointSelection;
use std::num::NonZeroUsize;

use super::state_exchange::ExchangeBuffers;

#[derive(Default)]
pub(super) struct SelectionScratch {
    pub points: Vec<usize>,
    ranges: Vec<f64>,
    distances: Vec<f64>,
    chosen: Vec<bool>,
    exploration: Vec<(u64, usize)>,
    unique: Vec<usize>,
    audit_points: Vec<usize>,
    audit_positions: Vec<usize>,
    budget_floor: Vec<usize>,
    first_iteration: Option<u64>,
    unique_count: usize,
}

#[derive(Clone, Copy)]
pub(super) struct PointAuditContext<'a> {
    pub node: NodePos,
    pub node_id: NodeId,
    pub iteration: u64,
    pub visit_offset: usize,
    pub pool: &'a CutPool,
    pub projection: &'a CutStateProjection,
    pub states: &'a ExchangeBuffers,
}

fn scramble(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

impl SelectionScratch {
    pub fn select(
        &mut self,
        mut config: TrialPointSelection,
        iteration: u64,
        node: NodePos,
        candidates: &[usize],
        states: &ExchangeBuffers,
    ) {
        self.audit_points.clear();
        let first = *self.first_iteration.get_or_insert(iteration);
        if self.budget_floor.len() <= node.0 {
            self.budget_floor.resize(node.0 + 1, 0);
        }
        if config.audit_relative_tolerance.is_some() {
            let floor = if first > 1 {
                candidates.len()
            } else {
                self.budget_floor[node.0]
            };
            config.initial_points = NonZeroUsize::new(config.initial_points.get().max(floor))
                .unwrap_or(config.initial_points);
        }
        let mut unique = std::mem::take(&mut self.unique);
        unique.clear();
        unique.extend_from_slice(candidates);
        if config.deduplicate {
            unique.sort_unstable_by(|&a, &b| {
                states
                    .state_at(0, a)
                    .iter()
                    .map(|x| x.to_bits())
                    .cmp(states.state_at(0, b).iter().map(|x| x.to_bits()))
                    .then(a.cmp(&b))
            });
            unique.dedup_by(|a, b| {
                states
                    .state_at(0, *a)
                    .iter()
                    .map(|x| x.to_bits())
                    .eq(states.state_at(0, *b).iter().map(|x| x.to_bits()))
            });
            unique.sort_unstable();
        }
        self.unique_count = unique.len();
        self.select_distinct(config, iteration, &unique, states);
        self.unique = unique;
    }

    pub fn audit(&mut self, tolerance: Option<f64>, ctx: PointAuditContext<'_>) -> f64 {
        let Some(tolerance) = tolerance else {
            return 0.0;
        };
        self.audit_positions.clear();
        self.audit_positions
            .extend(self.audit_points.iter().filter_map(|m| {
                self.points
                    .binary_search(m)
                    .ok()
                    .map(|pos| pos + ctx.visit_offset)
            }));
        self.audit_positions.sort_unstable();
        let mut maximum = 0.0_f64;
        for &m in &self.audit_points {
            let state = ctx.states.state_at(0, m);
            let mut retained = f64::NEG_INFINITY;
            let mut probed = f64::NEG_INFINITY;
            for (slot, intercept, coefficients) in ctx.pool.active_cuts() {
                let value = intercept
                    + coefficients
                        .iter()
                        .enumerate()
                        .map(|(j, c)| {
                            c * state[ctx.projection.global_state_index(CutSlot::new(j)).get()]
                        })
                        .sum::<f64>();
                let meta = ctx.pool.metadata(slot);
                if meta.iteration_generated == ctx.iteration
                    && meta.node == ctx.node_id
                    && self
                        .audit_positions
                        .binary_search(&(meta.forward_pass_index as usize))
                        .is_ok()
                {
                    probed = probed.max(value);
                } else {
                    retained = retained.max(value);
                }
            }
            if probed.is_finite() {
                maximum = maximum.max(((probed - retained) / probed.abs().max(1.0)).max(0.0));
            }
        }
        if maximum > tolerance {
            self.budget_floor[ctx.node.0] = self.budget_floor[ctx.node.0]
                .max(self.points.len().saturating_mul(2))
                .min(self.unique_count);
        }
        tracing::info!(target: "cobre::point_audit", node = ctx.node.0,
            iteration = ctx.iteration, probes = self.audit_points.len(),
            max_relative_improvement = maximum, next_minimum_points = self.budget_floor[ctx.node.0],
            "backward point audit");
        maximum
    }

    fn select_distinct(
        &mut self,
        config: TrialPointSelection,
        iteration: u64,
        candidates: &[usize],
        states: &ExchangeBuffers,
    ) {
        self.points.clear();
        let n = candidates.len();
        if n == 0 {
            return;
        }
        let full_from = config.full_from_iteration.get() as u64;
        if iteration >= full_from || iteration.is_multiple_of(config.full_every.get() as u64) {
            self.points.extend_from_slice(candidates);
            return;
        }
        let initial = config.initial_points.get().min(n);
        let additional = (n - initial) as u128 * u128::from(iteration.saturating_sub(1))
            / u128::from(full_from.saturating_sub(1).max(1));
        let diverse = initial + usize::try_from(additional).unwrap_or(n - initial);
        let exploration = config.exploration_points.get().min(n - diverse);
        if diverse + exploration >= n {
            self.points.extend_from_slice(candidates);
            return;
        }
        let dim = states.state_at(0, candidates[0]).len();
        self.ranges.resize(dim, 0.0);
        for j in 0..dim {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &m in candidates {
                let value = states.state_at(0, m)[j];
                lo = lo.min(value);
                hi = hi.max(value);
            }
            self.ranges[j] = (hi - lo).max(1e-12);
        }
        self.chosen.clear();
        self.chosen.resize(n, false);
        self.distances.clear();
        self.distances.resize(n, f64::INFINITY);
        let mut next = 0;
        for _ in 0..diverse {
            self.chosen[next] = true;
            self.points.push(candidates[next]);
            let anchor = states.state_at(0, candidates[next]);
            for (i, &m) in candidates.iter().enumerate() {
                let distance = states
                    .state_at(0, m)
                    .iter()
                    .zip(anchor)
                    .zip(&self.ranges)
                    .map(|((&a, &b), &scale)| ((a - b) / scale).powi(2))
                    .sum::<f64>();
                self.distances[i] = self.distances[i].min(distance);
            }
            next = (0..n)
                .filter(|&i| !self.chosen[i])
                .max_by(|&a, &b| {
                    self.distances[a]
                        .total_cmp(&self.distances[b])
                        .then(b.cmp(&a))
                })
                .unwrap_or(0);
        }
        self.exploration.clear();
        self.exploration.extend(
            candidates
                .iter()
                .enumerate()
                .filter(|(i, _)| !self.chosen[*i])
                .map(|(_, &m)| (scramble((m as u64) ^ scramble(iteration)), m)),
        );
        self.exploration.sort_unstable();
        self.audit_points
            .extend(self.exploration.iter().take(exploration).map(|&(_, m)| m));
        self.points.extend_from_slice(&self.audit_points);
        // Cut slots and basis windows require ascending original scenario identity.
        self.points.sort_unstable();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    fn config() -> TrialPointSelection {
        TrialPointSelection {
            deduplicate: false,
            audit_relative_tolerance: None,
            initial_points: NonZeroUsize::new(2).unwrap(),
            exploration_points: NonZeroUsize::new(1).unwrap(),
            full_every: NonZeroUsize::new(5).unwrap(),
            full_from_iteration: NonZeroUsize::new(10).unwrap(),
        }
    }

    #[test]
    fn diversity_retains_distant_states_before_exploration() {
        use super::super::trajectory::TrajectoryRecord;
        use crate::setup::{NodeId, StageIdx};
        let records: Vec<_> = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 100.0]
            .iter()
            .map(|&x| TrajectoryRecord {
                primal: vec![],
                dual: vec![],
                stage_cost: 0.0,
                node_id: NodeId(0),
                state: vec![x],
            })
            .collect();
        let mut states = ExchangeBuffers::new(1, 8, 1);
        states
            .exchange(&records, StageIdx(0), 1, &cobre_comm::LocalBackend)
            .unwrap();
        let mut scratch = SelectionScratch::default();
        scratch.select(
            config(),
            1,
            NodePos(0),
            &(0..8).collect::<Vec<_>>(),
            &states,
        );
        assert!(scratch.points.contains(&0));
        assert!(scratch.points.contains(&7));
        assert_eq!(scratch.points.len(), 3);
    }

    #[test]
    fn audit_and_final_refinement_preserve_every_candidate() {
        let states = ExchangeBuffers::new(1, 8, 1);
        let candidates = vec![0, 2, 4, 7];
        let mut scratch = SelectionScratch::default();
        for iteration in [5, 10, 11] {
            scratch.select(config(), iteration, NodePos(0), &candidates, &states);
            assert_eq!(scratch.points, candidates);
        }
    }

    #[test]
    fn subset_is_reproducible_unique_and_budget_grows() {
        let states = ExchangeBuffers::new(1, 8, 1);
        let candidates: Vec<_> = (0..8).collect();
        let mut scratch = SelectionScratch::default();
        scratch.select(config(), 1, NodePos(0), &candidates, &states);
        let first = scratch.points.clone();
        assert_eq!(first.len(), 3);
        assert!(first.windows(2).all(|w| w[0] < w[1]));
        scratch.select(config(), 1, NodePos(0), &candidates, &states);
        assert_eq!(scratch.points, first);
        scratch.select(config(), 8, NodePos(0), &candidates, &states);
        assert!(scratch.points.len() > first.len());
    }
    #[test]
    fn dedup_uses_complete_state_and_preserves_canonical_identity() {
        use super::super::trajectory::TrajectoryRecord;
        use crate::setup::StageIdx;
        let records: Vec<_> = [
            vec![1.0, 2.0],
            vec![1.0, 3.0],
            vec![1.0, 2.0],
            vec![1.0, -0.0],
            vec![1.0, 0.0],
        ]
        .into_iter()
        .map(|state| TrajectoryRecord {
            primal: vec![],
            dual: vec![],
            stage_cost: 0.0,
            node_id: NodeId(0),
            state,
        })
        .collect();
        let mut states = ExchangeBuffers::new(2, 5, 1);
        states
            .exchange(&records, StageIdx(0), 1, &cobre_comm::LocalBackend)
            .unwrap();
        let mut cfg = config();
        cfg.deduplicate = true;
        let mut scratch = SelectionScratch::default();
        scratch.select(cfg, 10, NodePos(0), &[0, 1, 2, 3, 4], &states);
        assert_eq!(scratch.points, vec![0, 1, 3, 4]);
    }

    #[test]
    fn audit_increases_only_the_affected_node_budget_and_resume_is_conservative() {
        use crate::indexer::StateSpace;
        use cobre_core::temporal::StageStateConfig;
        let states = ExchangeBuffers::new(1, 16, 1);
        let candidates: Vec<_> = (0..16).collect();
        let mut cfg = config();
        cfg.audit_relative_tolerance = Some(0.01);
        let mut scratch = SelectionScratch::default();
        scratch.select(cfg, 1, NodePos(0), &candidates, &states);
        let probe = scratch
            .points
            .binary_search(&scratch.audit_points[0])
            .unwrap();
        let mut pool = CutPool::new(64, 1, 16, 0);
        pool.add_cut(NodeId(0), 0, 0, 10.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, probe as u32, 20.0, &[0.0]);
        let global = StateSpace::new(1, 0, 0, Vec::new(), 0, 0, Vec::new(), &[0]);
        let projection = CutStateProjection::new(
            &global,
            StageStateConfig {
                storage: true,
                inflow_lags: true,
            },
        );
        let improvement = scratch.audit(
            Some(0.01),
            PointAuditContext {
                node: NodePos(0),
                node_id: NodeId(0),
                iteration: 1,
                visit_offset: 0,
                pool: &pool,
                projection: &projection,
                states: &states,
            },
        );
        assert_eq!(improvement, 0.5);
        scratch.select(cfg, 1, NodePos(0), &candidates, &states);
        assert_eq!(scratch.points.len(), 7);
        scratch.select(cfg, 1, NodePos(1), &candidates, &states);
        assert_eq!(scratch.points.len(), 3);
        let mut resumed = SelectionScratch::default();
        resumed.select(cfg, 2, NodePos(0), &candidates, &states);
        assert_eq!(resumed.points, candidates);
    }
}
