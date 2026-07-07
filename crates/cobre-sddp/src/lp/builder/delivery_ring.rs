//! Generic lagged-delivery ring primitive shared by the travel-time
//! water-bucket ring and the anticipated-thermal ring: a borrowed outgoing
//! block (identity-resolved) and a paired incoming block (pinned) advance one
//! Markov-1 slot per stage through the same interior shift-row skeleton and
//! paired row-cap/column-freeze masking. The two rings differ only in how
//! each deposits into its newest slot ([`DeliverySemantics`]) and what its
//! terminal slot means ([`TerminalMode`]); both differences are expressed as
//! enum parameters, never a second skeleton implementation.
//!
//! [`StateLayout`](crate::indexer::StateLayout) remains the sole owner of the
//! out/in state-index ranges: a [`DeliveryRing`] borrows them for one
//! construction and never re-derives or persists an independent copy. The
//! block-mode-coupled per-lag deposit fill stays at each ring's own call
//! site; this module owns only the shared skeleton.

use std::ops::Range;

use super::columns::ColumnBufs;

/// Terminal-slot semantics a ring's masked boundary represents. Both variants
/// drive identical masking output today (frozen `[0, 0]` outgoing column, no
/// definition row); the tag is the seam for a future terminal mode whose
/// masked slot retains a value instead of dropping one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TerminalMode {
    /// A masked slot discards a genuine share the ring would otherwise
    /// deposit (the water ring's horizon boundary).
    ZeroTerminalDrop,
    /// A masked slot never held a value in the first place — no share is
    /// dropped because none was ever computed (the anticipated ring's
    /// horizon boundary).
    BoundaryFcfRetain,
}

/// Delivery semantics a ring deposit follows, matched only at
/// [`DeliveryRing::emit_deposit`] — the shift/mask skeleton is
/// δ-independent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum DeliverySemantics<'a> {
    /// Deposits a lag-weighted share from a shared release column into
    /// several ring slots at once (the water ring); `density` is the
    /// per-lag arrival weights, `Σ density == 1.0`.
    // Voice 4: water's conservation check goes straight through
    // `DeliveryRing::assert_density_sums_to_one`, never through this
    // dispatch — no production call site constructs this variant.
    #[allow(dead_code)]
    AdditiveSource {
        /// Per-lag arrival weight, validated by
        /// [`DeliveryRing::assert_density_sums_to_one`].
        density: &'a [f64],
    },
    /// Pins one ring slot to exactly one decision column (the anticipated
    /// ring); carries no data.
    EqualityPin,
}

/// A lagged-delivery ring over one dense, slot-major/lane-minor state-column
/// grid: `n_lanes` parallel delivery lanes (plants), each `depth` slots deep.
/// Borrows its outgoing/incoming column blocks from
/// [`StateLayout`](crate::indexer::StateLayout) — never a private copy of the
/// ranges.
#[derive(Debug, Clone)]
pub(crate) struct DeliveryRing {
    /// Outgoing (identity-resolved) column block, size `n_lanes * depth`.
    out_block: Range<usize>,
    /// Incoming (pinned) column block, size `n_lanes * depth`.
    in_block: Range<usize>,
    /// Parallel delivery lanes (plants) sharing this ring.
    n_lanes: usize,
    /// Slots per lane.
    depth: usize,
    /// This ring's terminal-slot semantics; see [`TerminalMode`].
    // Voice 4: no method reads `self.terminal` yet — both rings' masking is
    // identical regardless of terminal mode today. Refires once a
    // differentiated terminal mode reads it.
    #[allow(dead_code)]
    terminal: TerminalMode,
}

impl DeliveryRing {
    /// Constructs a ring over the borrowed out/in state blocks.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `out_block.len()` or `in_block.len()` differs from
    /// `n_lanes * depth`.
    #[must_use]
    pub(crate) fn new(
        out_block: Range<usize>,
        in_block: Range<usize>,
        n_lanes: usize,
        depth: usize,
        terminal: TerminalMode,
    ) -> Self {
        let dense_len = n_lanes * depth;
        debug_assert_eq!(
            out_block.len(),
            dense_len,
            "out_block must be sized n_lanes * depth ({dense_len}), got {}",
            out_block.len()
        );
        debug_assert_eq!(
            in_block.len(),
            dense_len,
            "in_block must be sized n_lanes * depth ({dense_len}), got {}",
            in_block.len()
        );
        Self {
            out_block,
            in_block,
            n_lanes,
            depth,
            terminal,
        }
    }

    /// Outgoing-block column for ring position `(slot, lane)` — the single
    /// owner of the ring's addressing arithmetic (slot-major, lane-minor).
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `slot >= depth` or `lane >= n_lanes`.
    #[must_use]
    pub(crate) fn out_col(&self, slot: usize, lane: usize) -> usize {
        debug_assert!(
            slot < self.depth,
            "slot {slot} must be < depth {}",
            self.depth
        );
        debug_assert!(
            lane < self.n_lanes,
            "lane {lane} must be < n_lanes {}",
            self.n_lanes
        );
        self.out_block.start + slot * self.n_lanes + lane
    }

    /// Incoming-block column for ring position `(slot, lane)`; see
    /// [`Self::out_col`].
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `slot >= depth` or `lane >= n_lanes`.
    #[must_use]
    pub(crate) fn in_col(&self, slot: usize, lane: usize) -> usize {
        debug_assert!(
            slot < self.depth,
            "slot {slot} must be < depth {}",
            self.depth
        );
        debug_assert!(
            lane < self.n_lanes,
            "lane {lane} must be < n_lanes {}",
            self.n_lanes
        );
        self.in_block.start + slot * self.n_lanes + lane
    }

    /// Decomposes a flat block-relative offset (`col − out_block.start` or
    /// `col − in_block.start`) into `(slot, lane)` — the exact inverse of
    /// [`Self::out_col`]/[`Self::in_col`]'s slot-major, lane-minor addressing.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `offset >= n_lanes * depth`.
    #[must_use]
    pub(crate) fn slot_lane_at(&self, offset: usize) -> (usize, usize) {
        debug_assert!(
            offset < self.n_lanes * self.depth,
            "offset {offset} must be < n_lanes*depth {}",
            self.n_lanes * self.depth
        );
        (offset / self.n_lanes, offset % self.n_lanes)
    }

    /// Emits the ring's interior shift-row CSC entries: `out[slot] +1`, and
    /// `in[slot+1] −1` when a deeper ring position exists for the same lane
    /// (`slot + 1 < depth`). `row_pos[slot * n_lanes + lane]` gives this
    /// stage's compact row position, `None` when masked — the paired column
    /// freeze for a masked position lives in [`Self::freeze_masked_columns`],
    /// never here. Returns the count of rows emitted.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `row_pos.len() != n_lanes * depth`.
    pub(crate) fn emit_shift_rows(
        &self,
        row_pos: &[Option<usize>],
        row_start: usize,
        col_entries: &mut [Vec<(usize, f64)>],
    ) -> usize {
        if self.n_lanes == 0 || self.depth == 0 {
            debug_assert!(
                row_pos.is_empty(),
                "row_pos must be empty when n_lanes or depth is 0"
            );
            return 0;
        }
        debug_assert_eq!(
            row_pos.len(),
            self.n_lanes * self.depth,
            "row_pos must be sized n_lanes * depth (dense, slot-major, lane-minor)"
        );
        let mut n_reachable = 0_usize;
        for (flat, pos) in row_pos.iter().enumerate() {
            let Some(pos) = *pos else { continue };
            let slot = flat / self.n_lanes;
            let lane = flat % self.n_lanes;
            let row = row_start + pos;
            col_entries[self.out_col(slot, lane)].push((row, 1.0));
            if slot + 1 < self.depth {
                col_entries[self.in_col(slot + 1, lane)].push((row, -1.0));
            }
            n_reachable += 1;
        }
        n_reachable
    }

    /// Freezes every masked ring position (`row_pos[i] == None`) to `[0, 0]`
    /// and leaves every reachable position at `reachable_bound` — the paired
    /// column half of the masking contract [`Self::emit_shift_rows`]
    /// discharges the row half for. `reachable_bound` is the ring's open
    /// default (water's implicit `[0, inf)`, anticipated's signed
    /// `(-inf, inf)`); the masked bound is always `[0, 0]` regardless of
    /// `reachable_bound` or [`TerminalMode`] — scale-independent, so no
    /// column bound is ever rescaled to share the freeze between rings.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `row_pos.len() != n_lanes * depth`.
    pub(super) fn freeze_masked_columns(
        &self,
        row_pos: &[Option<usize>],
        col_base: usize,
        reachable_bound: (f64, f64),
        bufs: &mut ColumnBufs<'_>,
    ) {
        debug_assert_eq!(
            row_pos.len(),
            self.n_lanes * self.depth,
            "row_pos must be sized n_lanes * depth (dense, slot-major, lane-minor)"
        );
        let (reachable_lower, reachable_upper) = reachable_bound;
        for (offset, pos) in row_pos.iter().enumerate() {
            let col = col_base + offset;
            if pos.is_some() {
                bufs.col_lower[col] = reachable_lower;
                bufs.col_upper[col] = reachable_upper;
            } else {
                bufs.col_lower[col] = 0.0;
                bufs.col_upper[col] = 0.0;
            }
        }
    }

    /// Emits the δ-matched ring deposit at `(slot, lane)`'s `row`.
    /// `EqualityPin` pins the ring's outgoing column to `decision_col`
    /// (`+1` on `out_col(slot, lane)`, `−1` on `decision_col`).
    /// `AdditiveSource` emits no entry here — the block-mode-coupled per-lag
    /// fill stays at the call site — and only re-validates its density's
    /// conservation invariant via [`Self::assert_density_sums_to_one`].
    pub(crate) fn emit_deposit(
        &self,
        slot: usize,
        lane: usize,
        row: usize,
        decision_col: usize,
        semantics: DeliverySemantics<'_>,
        col_entries: &mut [Vec<(usize, f64)>],
    ) {
        match semantics {
            DeliverySemantics::EqualityPin => {
                col_entries[self.out_col(slot, lane)].push((row, 1.0));
                col_entries[decision_col].push((row, -1.0));
            }
            DeliverySemantics::AdditiveSource { density } => {
                Self::assert_density_sums_to_one(density);
            }
        }
    }

    /// Ring slot targeted by lane `lane`'s `lag`-th deposit (`lag >= 1`;
    /// `lag == 0` is the same-stage share, never a ring slot), returned as a
    /// flat `row_pos`/column index (`(lag - 1) * n_lanes + lane`) the caller
    /// resolves against its own per-stage reachability table.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `lag == 0`, the resulting slot is `>= depth`, or
    /// `lane >= n_lanes`.
    #[must_use]
    pub(crate) fn slot_target(&self, lane: usize, lag: usize) -> usize {
        debug_assert!(
            lag >= 1,
            "lag must be >= 1; lag 0 is the same-stage share, never a ring slot"
        );
        let slot = lag - 1;
        debug_assert!(
            slot < self.depth,
            "slot {slot} must be < depth {}",
            self.depth
        );
        debug_assert!(
            lane < self.n_lanes,
            "lane {lane} must be < n_lanes {}",
            self.n_lanes
        );
        slot * self.n_lanes + lane
    }

    /// Debug-asserts `density` sums to `1.0` — the k-factor / arrival-density
    /// conservation invariant every [`DeliverySemantics::AdditiveSource`]
    /// deposit must uphold.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `density` does not sum to `1.0` within `1e-9`.
    pub(crate) fn assert_density_sums_to_one(density: &[f64]) {
        debug_assert!(
            (density.iter().sum::<f64>() - 1.0).abs() < 1e-9,
            "delivery density must sum to 1.0, got {density:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{ColumnBufs, DeliveryRing, DeliverySemantics, TerminalMode};

    /// One ring instance per lane (`n_lanes = 1`), mirroring the water ring's
    /// per-plant contiguous addressing: lane A is 3 slots deep (every
    /// interior slot has a deeper neighbor except the last), lane B is 1 slot
    /// deep (its only slot has no deeper neighbor at all). Hand-computed
    /// against the ring's own addressing formula, never against
    /// `fill_transit_bucket_definition_entries`'s output.
    #[test]
    fn emit_shift_rows_drops_the_shift_term_past_a_lanes_own_depth() {
        let ring_a = DeliveryRing::new(100..103, 200..203, 1, 3, TerminalMode::ZeroTerminalDrop);
        let row_pos_a = vec![Some(0), Some(1), Some(2)];
        let mut col_entries_a: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 210];
        let n_a = ring_a.emit_shift_rows(&row_pos_a, 900, &mut col_entries_a);
        assert_eq!(n_a, 3);
        // slot 0: out[100] +1, in[201] -1 (deeper neighbor at slot 1 exists).
        assert_eq!(col_entries_a[100], vec![(900, 1.0)]);
        assert_eq!(col_entries_a[201], vec![(900, -1.0)]);
        // slot 1: out[101] +1, in[202] -1 (deeper neighbor at slot 2 exists).
        assert_eq!(col_entries_a[101], vec![(901, 1.0)]);
        assert_eq!(col_entries_a[202], vec![(901, -1.0)]);
        // slot 2 (the lane's last slot): out[102] +1, NO shift term — slot 3
        // does not exist for this lane's own depth of 3.
        assert_eq!(col_entries_a[102], vec![(902, 1.0)]);
        assert!(col_entries_a[203].is_empty());

        let ring_b = DeliveryRing::new(150..151, 250..251, 1, 1, TerminalMode::ZeroTerminalDrop);
        let row_pos_b = vec![Some(0)];
        let mut col_entries_b: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 260];
        let n_b = ring_b.emit_shift_rows(&row_pos_b, 950, &mut col_entries_b);
        assert_eq!(n_b, 1);
        // A single-slot lane never has a deeper neighbor.
        assert_eq!(col_entries_b[150], vec![(950, 1.0)]);
        assert!(col_entries_b[250].is_empty());
    }

    /// One ring shared across `n_lanes = 2` (dense, slot-major/lane-minor),
    /// mirroring the anticipated ring: lane 0 has two interior rows (slots 0
    /// and 1; slot 2 is its own deposit slot, out of scope here), lane 1 has
    /// none (its own deposit slot is slot 0, so every slot is masked in this
    /// table). Slot 1's shift term still targets slot 2's incoming column
    /// even though slot 2 has no row of its own — the incoming column is a
    /// state variable regardless of whether it gets its own definition row.
    #[test]
    fn emit_shift_rows_dense_grid_shares_one_ring_across_heterogeneous_lanes() {
        let ring = DeliveryRing::new(300..306, 400..406, 2, 3, TerminalMode::BoundaryFcfRetain);
        // flat = slot * n_lanes + lane.
        let row_pos = vec![
            Some(0), // slot 0, lane 0
            None,    // slot 0, lane 1
            Some(1), // slot 1, lane 0
            None,    // slot 1, lane 1
            None,    // slot 2, lane 0 (deposit slot, not an interior row)
            None,    // slot 2, lane 1
        ];
        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 410];
        let n = ring.emit_shift_rows(&row_pos, 500, &mut col_entries);
        assert_eq!(n, 2);
        // slot 0, lane 0: out[300] +1, in[402] -1 (slot 1, lane 0).
        assert_eq!(col_entries[300], vec![(500, 1.0)]);
        assert_eq!(col_entries[402], vec![(500, -1.0)]);
        // slot 1, lane 0: out[302] +1, in[404] -1 (slot 2, lane 0), even
        // though slot 2/lane 0 itself has no row in this table.
        assert_eq!(col_entries[302], vec![(501, 1.0)]);
        assert_eq!(col_entries[404], vec![(501, -1.0)]);
        // Lane 1 contributes no rows at all.
        for &col in &[301, 303, 305] {
            assert!(
                col_entries[col].is_empty(),
                "lane 1 column {col} must stay untouched"
            );
        }
    }

    /// A masked position freezes to `[0, 0]` with no dependence on the
    /// ring's `reachable_bound` or `TerminalMode` — the shared-masking
    /// assertion, checked for both a water-like open reachable bound and an
    /// anticipated-like signed one, and for both terminal modes.
    #[test]
    fn freeze_masked_columns_masks_identically_across_reachable_bound_and_terminal_mode() {
        let row_pos = vec![Some(0), None, Some(1)];
        let bounds: [((f64, f64), &str); 2] = [
            ((0.0, f64::INFINITY), "water-like [0, inf)"),
            (
                (f64::NEG_INFINITY, f64::INFINITY),
                "anticipated-like (-inf, inf)",
            ),
        ];
        let terminals = [
            TerminalMode::ZeroTerminalDrop,
            TerminalMode::BoundaryFcfRetain,
        ];

        for terminal in terminals {
            for (reachable_bound, label) in bounds {
                let ring = DeliveryRing::new(50..53, 150..153, 1, 3, terminal);
                let mut col_lower = vec![-9.0; 60];
                let mut col_upper = vec![9.0; 60];
                let mut objective = vec![0.0; 60];
                let mut bufs = ColumnBufs {
                    col_lower: &mut col_lower,
                    col_upper: &mut col_upper,
                    objective: &mut objective,
                };
                ring.freeze_masked_columns(&row_pos, 50, reachable_bound, &mut bufs);

                assert_eq!(
                    col_lower[50], reachable_bound.0,
                    "{label}, {terminal:?}: reachable col 50 lower"
                );
                assert_eq!(
                    col_upper[50], reachable_bound.1,
                    "{label}, {terminal:?}: reachable col 50 upper"
                );
                assert_eq!(
                    col_lower[51], 0.0,
                    "{label}, {terminal:?}: masked col 51 lower"
                );
                assert_eq!(
                    col_upper[51], 0.0,
                    "{label}, {terminal:?}: masked col 51 upper"
                );
                assert_eq!(
                    col_lower[52], reachable_bound.0,
                    "{label}, {terminal:?}: reachable col 52 lower"
                );
                assert_eq!(
                    col_upper[52], reachable_bound.1,
                    "{label}, {terminal:?}: reachable col 52 upper"
                );
            }
        }
    }

    #[test]
    fn out_col_in_col_addressing_is_slot_major_lane_minor() {
        let ring = DeliveryRing::new(1000..1006, 2000..2006, 3, 2, TerminalMode::ZeroTerminalDrop);
        // slot 0: lanes 0, 1, 2 occupy the first n_lanes columns.
        assert_eq!(ring.out_col(0, 0), 1000);
        assert_eq!(ring.out_col(0, 1), 1001);
        assert_eq!(ring.out_col(0, 2), 1002);
        // slot 1: advances by a full n_lanes stride.
        assert_eq!(ring.out_col(1, 0), 1003);
        assert_eq!(ring.out_col(1, 1), 1004);
        assert_eq!(ring.out_col(1, 2), 1005);
        assert_eq!(ring.in_col(0, 0), 2000);
        assert_eq!(ring.in_col(1, 2), 2005);
    }

    /// `slot_lane_at` is the exact inverse of `out_col`/`in_col`: recovering
    /// `(slot, lane)` from every block-relative offset in a 3-lane, 2-deep
    /// ring round-trips through `out_col` back to the same offset.
    #[test]
    fn slot_lane_at_is_the_inverse_of_out_col_in_col() {
        let ring = DeliveryRing::new(1000..1006, 2000..2006, 3, 2, TerminalMode::ZeroTerminalDrop);
        assert_eq!(ring.slot_lane_at(0), (0, 0));
        assert_eq!(ring.slot_lane_at(1), (0, 1));
        assert_eq!(ring.slot_lane_at(2), (0, 2));
        assert_eq!(ring.slot_lane_at(3), (1, 0));
        assert_eq!(ring.slot_lane_at(4), (1, 1));
        assert_eq!(ring.slot_lane_at(5), (1, 2));

        for offset in 0..6 {
            let (slot, lane) = ring.slot_lane_at(offset);
            assert_eq!(
                ring.out_col(slot, lane) - 1000,
                offset,
                "slot_lane_at({offset}) must round-trip through out_col"
            );
            assert_eq!(
                ring.in_col(slot, lane) - 2000,
                offset,
                "slot_lane_at({offset}) must round-trip through in_col"
            );
        }
    }

    /// `EqualityPin`'s deposit is the ring's only real entry emission: the
    /// `(+1, -1)` slot↔decision pair, under a `col_scale = 1.0` column
    /// context — the freeze pins `[0, 0]`, never a scaled bound, so no bound
    /// needs adjusting for the column's scale.
    #[test]
    fn equality_pin_deposit_emits_the_slot_decision_pair_at_unit_col_scale() {
        let ring = DeliveryRing::new(10..13, 20..23, 1, 3, TerminalMode::BoundaryFcfRetain);
        let decision_col = 99;
        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 100];
        let col_scale = 1.0_f64;

        ring.emit_deposit(
            2,
            0,
            777,
            decision_col,
            DeliverySemantics::EqualityPin,
            &mut col_entries,
        );

        let out_col = ring.out_col(2, 0);
        assert_eq!(col_entries[out_col], vec![(777, col_scale)]);
        assert_eq!(col_entries[decision_col], vec![(777, -col_scale)]);
    }

    /// `AdditiveSource`'s deposit arm emits no entries — it only
    /// re-validates conservation; the per-block fill stays at the call site.
    #[test]
    fn additive_source_deposit_emits_no_entries() {
        let ring = DeliveryRing::new(10..13, 20..23, 1, 3, TerminalMode::ZeroTerminalDrop);
        let density = [0.25, 0.5, 0.25];
        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 100];

        ring.emit_deposit(
            0,
            0,
            1,
            99,
            DeliverySemantics::AdditiveSource { density: &density },
            &mut col_entries,
        );

        assert!(col_entries.iter().all(Vec::is_empty));
    }

    #[test]
    fn slot_target_maps_lag_to_the_flat_row_pos_index() {
        let ring = DeliveryRing::new(0..9, 0..9, 3, 3, TerminalMode::ZeroTerminalDrop);
        // lane 1, lag 1 -> slot 0 -> flat index 0 * 3 + 1.
        assert_eq!(ring.slot_target(1, 1), 1);
        // lane 2, lag 3 -> slot 2 -> flat index 2 * 3 + 2.
        assert_eq!(ring.slot_target(2, 3), 8);
    }

    #[test]
    fn assert_density_sums_to_one_accepts_a_conserved_density() {
        DeliveryRing::assert_density_sums_to_one(&[0.5, 0.3, 0.2]);
        DeliveryRing::assert_density_sums_to_one(&[1.0]);
    }

    #[test]
    #[should_panic(expected = "delivery density must sum to 1.0")]
    fn assert_density_sums_to_one_rejects_a_non_conserved_density() {
        DeliveryRing::assert_density_sums_to_one(&[0.5, 0.2]);
    }
}
