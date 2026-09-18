//! Per-worker, per-family drift tally: `max_abs`/`max_rel`/`clamped_count` for
//! each bounded state family, recorded inline by the outgoing-state read-back
//! seam and folded across ranks before the collective. Drift is tallied, not
//! judged: the only reducible members are Max-floats and a Sum-integer —
//! no float sum/mean — which is what keeps the cross-rank reduction
//! order-invariant and bit-reproducible with no compensated summation.

// Rationale (dead_code): `record` is wired into the outgoing-state read-back
// seam; `merge`/`reset` and the reducing getters have no production caller
// until the end-of-run cross-rank reduction lands.
#![cfg_attr(not(test), allow(dead_code))]

/// Per-family drift. `max_abs`/`max_rel` reduce by [`f64::max`],
/// `clamped_count` by integer add — both commutative and associative, so
/// [`DriftTally::merge`] is order-invariant. Never add a float
/// sum/mean/total member here: an order-dependent float sum would
/// need compensated summation to stay bit-reproducible across rank/thread
/// counts, defeating the reason this shape exists.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FamilyDrift {
    pub(crate) max_abs: f64,
    pub(crate) max_rel: f64,
    pub(crate) clamped_count: u64,
}

/// Indexes [`DriftTally`]'s three bounded families. Internal only — distinct
/// from the persisted `cobre_io::StateFamily` key the metadata mirror uses.
#[derive(Debug, Clone, Copy)]
pub(crate) enum StateFamilyKind {
    Storage,
    TransitBuckets,
    CommitmentHold,
}

/// Per-worker drift tally over the three bounded state families (storage,
/// transit buckets, commitment hold); inflow lags never clamp and carry no
/// entry.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DriftTally {
    pub(crate) storage: FamilyDrift,
    pub(crate) transit_buckets: FamilyDrift,
    pub(crate) commitment_hold: FamilyDrift,
}

impl DriftTally {
    /// Records one clamped dimension: `drift_abs = |value − clamped|` (≥ 0,
    /// finite — the seam excludes NaN/inverted boxes by construction) against
    /// `clamp_target = clamped`. The relative denominator is
    /// `max(1, |clamp_target|)`, well-defined even when the
    /// family's box width is infinite. Never call with a NaN/±Inf
    /// `drift_abs` — the parallel reduction depends on finite operands for
    /// `MPI_MAX` reproducibility.
    pub(crate) fn record(&mut self, family: StateFamilyKind, drift_abs: f64, clamp_target: f64) {
        debug_assert!(
            drift_abs.is_finite() && drift_abs >= 0.0,
            "record: drift_abs must be finite and non-negative"
        );
        let f = self.family_mut(family);
        f.max_abs = f.max_abs.max(drift_abs);
        f.max_rel = f.max_rel.max(drift_abs / clamp_target.abs().max(1.0));
        f.clamped_count += 1;
    }

    /// Folds `other` into `self` per family: `max_abs`/`max_rel` by
    /// [`f64::max`], `clamped_count` by integer add. Commutative and
    /// associative, so the per-rank fold before the collective is
    /// order-invariant.
    pub(crate) fn merge(&mut self, other: &DriftTally) {
        Self::merge_family(&mut self.storage, &other.storage);
        Self::merge_family(&mut self.transit_buckets, &other.transit_buckets);
        Self::merge_family(&mut self.commitment_hold, &other.commitment_hold);
    }

    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    fn merge_family(dst: &mut FamilyDrift, src: &FamilyDrift) {
        dst.max_abs = dst.max_abs.max(src.max_abs);
        dst.max_rel = dst.max_rel.max(src.max_rel);
        dst.clamped_count += src.clamped_count;
    }

    fn family_mut(&mut self, family: StateFamilyKind) -> &mut FamilyDrift {
        match family {
            StateFamilyKind::Storage => &mut self.storage,
            StateFamilyKind::TransitBuckets => &mut self.transit_buckets,
            StateFamilyKind::CommitmentHold => &mut self.commitment_hold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drift_tally_merge_is_commutative() {
        let mut a = DriftTally::default();
        a.record(StateFamilyKind::Storage, 1.5, 10.0);
        a.record(StateFamilyKind::CommitmentHold, 0.2, 0.1);

        let mut b = DriftTally::default();
        b.record(StateFamilyKind::Storage, 3.0, 2.0);
        b.record(StateFamilyKind::TransitBuckets, 0.7, 5.0);

        let mut a_merge_b = a;
        a_merge_b.merge(&b);
        let mut b_merge_a = b;
        b_merge_a.merge(&a);

        for (x, y) in [
            (a_merge_b.storage, b_merge_a.storage),
            (a_merge_b.transit_buckets, b_merge_a.transit_buckets),
            (a_merge_b.commitment_hold, b_merge_a.commitment_hold),
        ] {
            assert_eq!(x.max_abs, y.max_abs);
            assert_eq!(x.max_rel, y.max_rel);
            assert_eq!(x.clamped_count, y.clamped_count);
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn drift_tally_rel_denominator_is_max_one_abs_target() {
        let mut tally = DriftTally::default();
        tally.record(StateFamilyKind::CommitmentHold, 0.5, 0.25);

        assert_eq!(tally.commitment_hold.max_abs, 0.5);
        assert_eq!(tally.commitment_hold.max_rel, 0.5);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn drift_tally_default_is_all_zero() {
        let tally = DriftTally::default();
        for family in [tally.storage, tally.transit_buckets, tally.commitment_hold] {
            assert_eq!(family.clamped_count, 0);
            assert_eq!(family.max_abs, 0.0);
            assert_eq!(family.max_rel, 0.0);
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn drift_tally_clamped_count_zero_implies_zero_maxes() {
        let mut tally = DriftTally::default();
        tally.record(StateFamilyKind::Storage, 4.0, 1.0);
        tally.reset();

        for family in [tally.storage, tally.transit_buckets, tally.commitment_hold] {
            assert!(family.clamped_count == 0 && family.max_abs == 0.0 && family.max_rel == 0.0);
        }
    }
}
