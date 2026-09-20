use cobre_io::config::training::TrialPointSelection;

use super::state_exchange::ExchangeBuffers;

#[derive(Default)]
pub(super) struct SelectionScratch {
    pub points: Vec<usize>,
    ranges: Vec<f64>,
    distances: Vec<f64>,
    chosen: Vec<bool>,
    exploration: Vec<(u64, usize)>,
}

fn scramble(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

impl SelectionScratch {
    pub fn select(
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
        self.points
            .extend(self.exploration.iter().take(exploration).map(|&(_, m)| m));
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
        scratch.select(config(), 1, &(0..8).collect::<Vec<_>>(), &states);
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
            scratch.select(config(), iteration, &candidates, &states);
            assert_eq!(scratch.points, candidates);
        }
    }

    #[test]
    fn subset_is_reproducible_unique_and_budget_grows() {
        let states = ExchangeBuffers::new(1, 8, 1);
        let candidates: Vec<_> = (0..8).collect();
        let mut scratch = SelectionScratch::default();
        scratch.select(config(), 1, &candidates, &states);
        let first = scratch.points.clone();
        assert_eq!(first.len(), 3);
        assert!(first.windows(2).all(|w| w[0] < w[1]));
        scratch.select(config(), 1, &candidates, &states);
        assert_eq!(scratch.points, first);
        scratch.select(config(), 8, &candidates, &states);
        assert!(scratch.points.len() > first.len());
    }
}
