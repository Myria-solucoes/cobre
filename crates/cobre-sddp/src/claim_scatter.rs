//! Shared work-claim and scatter primitives for the crate's two dynamic
//! parallel-region schedulers: the by-node backward pass
//! (`training::backward::by_node`) and the enumerated forward engine
//! (`training::forward::enumerated`). Both claim units from a shared atomic
//! counter in any order, then scatter each worker's captures into a shared
//! arena in ascending `(worker, item)` order so the result is independent of
//! claim order (sddp.md "By-node scheduler is warm-start-only"). Only these
//! two narrow, genuinely-identical primitives live here — the claim loop
//! BODIES and the scatter-item WRITES stay at each caller, since their
//! borrow shapes and per-item work differ.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Shared work-claim counter: workers race `fetch_add`-then-bound-check to
/// pull the next unit index, in any order.
pub(crate) struct ClaimCursor {
    next: AtomicUsize,
    total: usize,
}

impl ClaimCursor {
    pub(crate) fn new(total: usize) -> Self {
        Self {
            next: AtomicUsize::new(0),
            total,
        }
    }

    /// Claim the next unit index, or `None` once every unit in `0..total`
    /// has been claimed.
    ///
    /// `Relaxed` ordering: the counter carries no data of its own, only the
    /// claimed index — each unit's actual memory effects are visible to the
    /// caller's own sequential post-region scatter, not through this counter.
    pub(crate) fn claim(&self) -> Option<usize> {
        let u = self.next.fetch_add(1, Ordering::Relaxed);
        (u < self.total).then_some(u)
    }
}

/// Ascending `(worker, item)` pairs over each worker's own claimed-item
/// count — the canonical scatter order that makes cut/arena aggregation
/// independent of claim order and worker count. `counts[w]` is worker `w`'s
/// own item count (already resolved from its own `Result`, since the two
/// callers resolve the first worker error differently — `by_node_finish`
/// eagerly, the enumerated engine via a `collect::<Result<Vec<_>, _>>()?`);
/// this borrows `counts` and owns nothing, so it allocates no scratch of its
/// own. Each caller supplies its own per-item write (the arena addressing and
/// copied fields differ between `by_node_finish` and the enumerated engine's
/// post-region scatter) — this owns only the shared traversal.
pub(crate) fn canonical_scatter(counts: &[usize]) -> impl Iterator<Item = (usize, usize)> + '_ {
    counts
        .iter()
        .enumerate()
        .flat_map(|(w, &count)| (0..count).map(move |i| (w, i)))
}

#[cfg(test)]
mod tests {
    use super::{ClaimCursor, canonical_scatter};

    #[test]
    fn claim_cursor_yields_each_unit_exactly_once_then_none() {
        let cursor = ClaimCursor::new(5);
        let mut claimed: Vec<usize> = std::iter::from_fn(|| cursor.claim()).collect();
        claimed.sort_unstable();
        assert_eq!(claimed, vec![0, 1, 2, 3, 4]);
        assert_eq!(cursor.claim(), None);
    }

    #[test]
    fn claim_cursor_zero_total_yields_nothing() {
        let cursor = ClaimCursor::new(0);
        assert_eq!(cursor.claim(), None);
    }

    #[test]
    fn canonical_scatter_visits_worker_item_pairs_in_ascending_order() {
        let counts = [2usize, 0, 1];
        let visited: Vec<(usize, usize)> = canonical_scatter(&counts).collect();
        assert_eq!(visited, vec![(0, 0), (0, 1), (2, 0)]);
    }

    #[test]
    fn canonical_scatter_empty_counts_yields_nothing() {
        let counts: [usize; 0] = [];
        assert_eq!(canonical_scatter(&counts).count(), 0);
    }

    #[test]
    fn canonical_scatter_all_zero_counts_yields_nothing() {
        let counts = [0usize, 0, 0];
        assert_eq!(canonical_scatter(&counts).count(), 0);
    }
}
