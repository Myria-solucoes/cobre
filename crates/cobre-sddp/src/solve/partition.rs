//! Deterministic static work partition shared by the training passes and
//! simulation.
//!
//! Lives in `solve/` (the shared LP-solve seam) rather than in any single pass
//! module because all of forward, backward, and simulation distribute work
//! across workers with the identical range arithmetic — a single owner keeps
//! their splits bit-identical.

/// Compute the scenario range `[start, end)` for worker `worker_id` when
/// distributing `n_scenarios` scenarios across `n_workers` workers.
///
/// Uses ceiling-division for the first `n_scenarios % n_workers` workers so
/// that extra scenarios are assigned to lower-index workers. This is a
/// deterministic, static partition — scenario-to-worker assignment is
/// identical regardless of thread scheduling order.
///
/// Returns `(start, end)` where `start == end` when `worker_id` receives zero
/// scenarios (only occurs when `n_workers > n_scenarios`).
#[inline]
pub(crate) fn partition(n_scenarios: usize, n_workers: usize, worker_id: usize) -> (usize, usize) {
    if n_workers == 0 {
        return (0, 0);
    }
    let base = n_scenarios / n_workers;
    let remainder = n_scenarios % n_workers;
    // First `remainder` workers get `base + 1` scenarios; the rest get `base`.
    let start = base * worker_id + worker_id.min(remainder);
    let end = start + base + usize::from(worker_id < remainder);
    (start, end)
}
