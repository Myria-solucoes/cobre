//! Per-worker, per-family drift tally: `max_abs`/`max_rel`/`clamped_count` for
//! each bounded state family, recorded inline by the outgoing-state read-back
//! seam and folded across ranks before the collective. Drift is tallied, not
//! judged: the only reducible members are Max-floats and a Sum-integer —
//! no float sum/mean — which is what keeps the cross-rank reduction
//! order-invariant and bit-reproducible with no compensated summation.

/// Coordinator warns when a reduced family `max_rel` exceeds this; set well
/// above the ~1e-9 feasibility tolerance so routine solver drift stays silent.
pub(crate) const MAX_REL_WARN_THRESHOLD: f64 = 1e-6;

/// Per-family drift. `max_abs`/`max_rel` reduce by [`f64::max`],
/// `clamped_count` by integer add — both commutative and associative, so
/// [`DriftTally::merge`] is order-invariant. Never add a float
/// sum/mean/total member here: an order-dependent float sum would
/// need compensated summation to stay bit-reproducible across rank/thread
/// counts, defeating the reason this shape exists.
#[derive(Debug, Clone, Copy, Default)]
pub struct FamilyDrift {
    pub(crate) max_abs: f64,
    pub(crate) max_rel: f64,
    pub(crate) clamped_count: u64,
}

/// Indexes [`DriftTally`]'s three bounded families. Internal only — the
/// persisted [`cobre_io::DriftSummary`] mirror keys the same families by named
/// field, never by this enum.
#[derive(Debug, Clone, Copy)]
pub(crate) enum StateFamilyKind {
    Storage,
    TransitBuckets,
    CommitmentHold,
}

impl StateFamilyKind {
    fn name(self) -> &'static str {
        match self {
            StateFamilyKind::Storage => "storage",
            StateFamilyKind::TransitBuckets => "transit_buckets",
            StateFamilyKind::CommitmentHold => "commitment_hold",
        }
    }
}

/// Per-worker drift tally over the three bounded state families (storage,
/// transit buckets, commitment hold); inflow lags never clamp and carry no
/// entry.
#[derive(Debug, Clone, Copy, Default)]
pub struct DriftTally {
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

    /// Packs the tally into the two fixed-size cross-rank reduction buffers: the
    /// six `max_abs`/`max_rel` magnitudes into a `MAX` buffer and the three
    /// `clamped_count`s (cast to `f64`) into a `SUM` buffer. The split is what
    /// keeps the reduction bit-reproducible: magnitudes reduce with `MPI_MAX`
    /// (exact, order-invariant) and counts with `MPI_SUM` on `f64` values that
    /// are exact and order-invariant below `2^53` — never a float sum of the
    /// magnitudes, which would need compensated summation. The precondition
    /// that only finite, non-negative values enter is upheld by [`Self::record`].
    pub(crate) fn to_reduce_buffers(self) -> ([f64; 6], [f64; 3]) {
        let max_buf = [
            self.storage.max_abs,
            self.storage.max_rel,
            self.transit_buckets.max_abs,
            self.transit_buckets.max_rel,
            self.commitment_hold.max_abs,
            self.commitment_hold.max_rel,
        ];
        #[allow(clippy::cast_precision_loss)]
        let sum_buf = [
            self.storage.clamped_count as f64,
            self.transit_buckets.clamped_count as f64,
            self.commitment_hold.clamped_count as f64,
        ];
        (max_buf, sum_buf)
    }

    /// Reconstructs a tally from the two reduced buffers produced by
    /// [`Self::to_reduce_buffers`]. Counts are recovered by `round`, exact for
    /// the reduced `f64` sums below `2^53`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn from_reduce_buffers(max_buf: &[f64; 6], sum_buf: &[f64; 3]) -> Self {
        Self {
            storage: FamilyDrift {
                max_abs: max_buf[0],
                max_rel: max_buf[1],
                clamped_count: sum_buf[0].round() as u64,
            },
            transit_buckets: FamilyDrift {
                max_abs: max_buf[2],
                max_rel: max_buf[3],
                clamped_count: sum_buf[1].round() as u64,
            },
            commitment_hold: FamilyDrift {
                max_abs: max_buf[4],
                max_rel: max_buf[5],
                clamped_count: sum_buf[2].round() as u64,
            },
        }
    }

    /// The family with the largest `max_rel` and that value; ties resolve to the
    /// earlier family in declaration order, so the pick is deterministic.
    pub(crate) fn worst_max_rel(&self) -> (StateFamilyKind, f64) {
        let mut worst = (StateFamilyKind::Storage, self.storage.max_rel);
        if self.transit_buckets.max_rel > worst.1 {
            worst = (
                StateFamilyKind::TransitBuckets,
                self.transit_buckets.max_rel,
            );
        }
        if self.commitment_hold.max_rel > worst.1 {
            worst = (
                StateFamilyKind::CommitmentHold,
                self.commitment_hold.max_rel,
            );
        }
        worst
    }

    /// Emits one coordinator-only `WARN` naming the worst family and its
    /// `max_rel`/`max_abs` when that `max_rel` exceeds [`MAX_REL_WARN_THRESHOLD`].
    /// Gated on `is_coordinator` so it fires once per phase, not once per rank.
    pub(crate) fn warn_if_exceeds(&self, is_coordinator: bool, phase: &str) {
        if !is_coordinator {
            return;
        }
        let (family, max_rel) = self.worst_max_rel();
        if max_rel <= MAX_REL_WARN_THRESHOLD {
            return;
        }
        let max_abs = match family {
            StateFamilyKind::Storage => self.storage.max_abs,
            StateFamilyKind::TransitBuckets => self.transit_buckets.max_abs,
            StateFamilyKind::CommitmentHold => self.commitment_hold.max_abs,
        };
        tracing::warn!(
            phase,
            family = family.name(),
            max_rel,
            max_abs,
            "state read-back drift exceeded the diagnostic threshold"
        );
    }

    // Test-only: the tally accumulates over a whole phase and each phase runs on
    // freshly built workspaces, so no production site resets it.
    #[cfg(test)]
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

/// Roll up the reduced [`DriftTally`] into the run-level
/// [`cobre_io::DriftSummary`] the training and simulation `metadata.json`
/// carry. Returns `None` when no family clamped (`clamped_count == 0` for all
/// three) so the section is omitted; the single owner both frontends call, so
/// the CLI and Python `drift` sections cannot diverge.
#[must_use]
pub fn build_drift_summary(tally: &DriftTally) -> Option<cobre_io::DriftSummary> {
    if tally.storage.clamped_count == 0
        && tally.transit_buckets.clamped_count == 0
        && tally.commitment_hold.clamped_count == 0
    {
        return None;
    }
    Some(cobre_io::DriftSummary {
        storage: map_family(&tally.storage),
        transit_buckets: map_family(&tally.transit_buckets),
        commitment_hold: map_family(&tally.commitment_hold),
    })
}

fn map_family(family: &FamilyDrift) -> cobre_io::FamilyDrift {
    cobre_io::FamilyDrift {
        max_abs: family.max_abs,
        max_rel: family.max_rel,
        clamped_count: family.clamped_count,
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

    #[test]
    #[allow(clippy::float_cmp)]
    fn drift_tally_reduction_single_rank_round_trips_to_merge() {
        use cobre_comm::{Communicator, LocalBackend, ReduceOp};

        let mut w0 = DriftTally::default();
        w0.record(StateFamilyKind::Storage, 1.5, 10.0);
        w0.record(StateFamilyKind::CommitmentHold, 0.2, 0.1);
        let mut w1 = DriftTally::default();
        w1.record(StateFamilyKind::Storage, 3.0, 2.0);
        w1.record(StateFamilyKind::TransitBuckets, 0.7, 5.0);
        let mut w2 = DriftTally::default();
        w2.record(StateFamilyKind::TransitBuckets, 4.0, 1.0);
        w2.record(StateFamilyKind::CommitmentHold, 9.0, 3.0);

        let mut expected = DriftTally::default();
        for w in [&w0, &w1, &w2] {
            expected.merge(w);
        }

        let (max_buf, sum_buf) = expected.to_reduce_buffers();
        let comm = LocalBackend;
        let mut max_out = [0.0_f64; 6];
        let mut sum_out = [0.0_f64; 3];
        comm.allreduce(&max_buf, &mut max_out, ReduceOp::Max)
            .unwrap();
        comm.allreduce(&sum_buf, &mut sum_out, ReduceOp::Sum)
            .unwrap();
        let reconstructed = DriftTally::from_reduce_buffers(&max_out, &sum_out);

        for (r, e) in [
            (reconstructed.storage, expected.storage),
            (reconstructed.transit_buckets, expected.transit_buckets),
            (reconstructed.commitment_hold, expected.commitment_hold),
        ] {
            assert_eq!(r.max_abs, e.max_abs);
            assert_eq!(r.max_rel, e.max_rel);
            assert_eq!(r.clamped_count, e.clamped_count);
        }
    }

    #[test]
    fn drift_tally_reduction_coordinator_warn_fires_above_and_is_silent_below() {
        let mut above = DriftTally::default();
        above.record(StateFamilyKind::TransitBuckets, 1.0, 1.0);

        let (recorder, messages) = WarnRecorder::new();
        tracing::subscriber::with_default(recorder, || {
            above.warn_if_exceeds(true, "test");
        });
        {
            let msgs = messages.lock().unwrap();
            assert_eq!(
                msgs.len(),
                1,
                "coordinator above threshold warns exactly once"
            );
            assert!(
                msgs[0].contains("transit_buckets"),
                "warn names the worst family: {:?}",
                msgs[0]
            );
        }

        let mut below = DriftTally::default();
        below.record(StateFamilyKind::Storage, 1e-9, 1.0);
        let (recorder, messages) = WarnRecorder::new();
        tracing::subscriber::with_default(recorder, || {
            below.warn_if_exceeds(true, "test");
        });
        assert!(
            messages.lock().unwrap().is_empty(),
            "below the threshold the coordinator is silent"
        );

        let (recorder, messages) = WarnRecorder::new();
        tracing::subscriber::with_default(recorder, || {
            above.warn_if_exceeds(false, "test");
        });
        assert!(
            messages.lock().unwrap().is_empty(),
            "a non-coordinator rank never warns"
        );
    }

    use std::sync::{Arc, Mutex};

    use tracing::{Event, Level, Metadata, Subscriber, span};

    /// Records WARN-level events as `"{family}|{message}"` into a shared buffer.
    struct WarnRecorder {
        messages: Arc<Mutex<Vec<String>>>,
    }

    impl WarnRecorder {
        fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
            let messages = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    messages: Arc::clone(&messages),
                },
                messages,
            )
        }
    }

    impl Subscriber for WarnRecorder {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            *metadata.level() <= Level::WARN
        }

        fn new_span(&self, _attrs: &span::Attributes<'_>) -> span::Id {
            span::Id::from_u64(1)
        }

        fn record(&self, _span: &span::Id, _values: &span::Record<'_>) {}

        fn record_follows_from(&self, _span: &span::Id, _follows: &span::Id) {}

        fn event(&self, event: &Event<'_>) {
            if *event.metadata().level() != Level::WARN {
                return;
            }
            let mut visitor = FieldVisitor {
                message: String::new(),
                family: String::new(),
            };
            event.record(&mut visitor);
            self.messages
                .lock()
                .unwrap()
                .push(format!("{}|{}", visitor.family, visitor.message));
        }

        fn enter(&self, _span: &span::Id) {}

        fn exit(&self, _span: &span::Id) {}
    }

    struct FieldVisitor {
        message: String,
        family: String,
    }

    impl tracing::field::Visit for FieldVisitor {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            match field.name() {
                "message" => self.message = format!("{value:?}"),
                "family" => self.family = format!("{value:?}"),
                _ => {}
            }
        }

        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            match field.name() {
                "message" => self.message = value.to_string(),
                "family" => self.family = value.to_string(),
                _ => {}
            }
        }
    }
}
