use super::IterationRecord;

pub(crate) const NAME: &str = "lb_stability_v1";
pub(crate) const WINDOW_ITERATIONS: u32 = 3;
pub(crate) const CLASSIFICATION: &str = "heuristic";
pub(crate) const STOP_ELIGIBLE: bool = false;

pub(crate) fn values(records: &[IterationRecord]) -> Vec<Option<f64>> {
    let mut best_lower_bounds = Vec::with_capacity(records.len());
    let mut best = f64::NEG_INFINITY;

    for record in records {
        best = best.max(record.lower_bound);
        best_lower_bounds.push(best);
    }

    best_lower_bounds
        .iter()
        .enumerate()
        .map(|(index, current)| {
            let window_start = index.checked_sub(WINDOW_ITERATIONS as usize - 1)?;
            let denominator = current.abs().max(1.0);
            Some((current - best_lower_bounds[window_start]) / denominator * 100.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::values;
    use crate::output::IterationRecord;

    fn record(iteration: u32, lower_bound: f64) -> IterationRecord {
        IterationRecord {
            iteration,
            lower_bound,
            upper_bound: 0.0,
            upper_bound_std: 0.0,
            gap_percent: None,
            cuts_added: 0,
            cuts_removed: 0,
            cuts_active: 0,
            time_forward_ms: 0,
            time_backward_ms: 0,
            time_total_ms: 0,
            time_forward_wall_ms: 0,
            time_backward_wall_ms: 0,
            time_cut_selection_ms: 0,
            time_mpi_allreduce_ms: 0,
            time_cut_sync_ms: 0,
            time_lower_bound_ms: 0,
            time_state_exchange_ms: 0,
            time_cut_batch_build_ms: 0,
            time_bwd_load_imbalance_ms: 0,
            time_bwd_scheduling_overhead_ms: 0,
            time_fwd_load_imbalance_ms: 0,
            time_fwd_scheduling_overhead_ms: 0,
            time_overhead_ms: 0,
            solve_time_ms: 0.0,
            forward_passes: 0,
            lp_solves: 0,
            mean_rows_in_lp: 0.0,
        }
    }

    #[test]
    fn lb_stability_uses_monotone_best_bound_over_three_observations() {
        let records = [
            record(1, 100.0),
            record(2, 110.0),
            record(3, 105.0),
            record(4, 121.0),
        ];

        let result = values(&records);

        assert_eq!(result[0], None);
        assert_eq!(result[1], None);
        assert_eq!(result[2], Some(10.0 / 110.0 * 100.0));
        assert_eq!(result[3], Some(11.0 / 121.0 * 100.0));
    }
}
