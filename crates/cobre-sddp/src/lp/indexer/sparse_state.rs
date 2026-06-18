//! Nonzero-state mask construction for [`StageIndexer`].
//!
//! Owns [`StageIndexer::set_nonzero_mask`] (which state-vector indices may carry
//! nonzero cut coefficients) and the test-only `StageIndexer::finalize_for_test`
//! helper that mirrors the production `build_wired_indexer` finalization step.

use super::layout::StageIndexer;

impl StageIndexer {
    /// Compute and store the nonzero state index mask from per-hydro
    /// lag-state-slot counts and per-plant anticipated lead-stage counts.
    ///
    /// `lag_counts` must have length `hydro_count`. Each entry is the number of
    /// lag-state slots that may carry non-zero cut coefficients for that hydro
    /// (0 means no AR lags). Indices `[0, N)` (storage) are always included.
    /// For each hydro `h`, lag indices
    /// `inflow_lags.start + l * hydro_count + h` are included for
    /// `l in 0..lag_counts[h]`.
    ///
    /// `anticipated_lead_stages` must have length `n_anticipated`. Each entry
    /// is the per-plant occupied-slot count `K_i` (`0..K_i` of the
    /// anticipated-state ring buffer at plant `i`). The trailing
    /// `k_max - K_i` slots are padding and are excluded from the mask: no
    /// decision variable writes to those columns, so their cut coefficients
    /// are structurally zero. Including padded slots would over-estimate cut
    /// hyperplanes (the direct analogue of the PAR(p)-A bug, where
    /// padded lag slots were included and shifted the cut above the LP value
    /// at the visited state).
    ///
    /// Layout for the anticipated block:
    /// `anticipated_state.start + slot * n_anticipated + plant`. The loop
    /// iterates slot-first, plant-second so the emitted indices stay
    /// monotonically increasing.
    ///
    /// The correct value for `lag_counts[h]` is
    /// `PrecomputedPar::effective_lag_count(h)`, **not** `PrecomputedPar::order(h)`.
    /// For PAR(p)-A hydros, `effective_lag_count` equals `max_order` (= 12) so
    /// that the `ψ̂/12` annual contributions on lag slots `order..max_order` are
    /// included in cut rows. Using `order(h)` would truncate those slots and
    /// produce over-estimating cuts (the same failure mode as anticipated
    /// padding, but on the lag block).
    ///
    /// After calling, `nonzero_state_indices` is sorted in ascending order and
    /// has no duplicates. If `max_par_order == 0` or all hydros use their full
    /// `max_par_order` slots, the mask covers all lag indices; if
    /// `n_anticipated == 0` no anticipated entries are appended.
    ///
    /// # Worked example
    ///
    /// One anticipated plant, `K_0 = 2`, `k_max = 3`, `n_anticipated = 1`,
    /// `hydro_count = 0`, `max_par_order = 0`,
    /// `anticipated_state.start = X` (for the corresponding indexer
    /// configuration). With `lag_counts = &[]` and
    /// `anticipated_lead_stages = &[2]`, the mask is:
    ///
    /// - Storage: empty (no hydros).
    /// - Lag: empty (no AR lags).
    /// - Anticipated: slot 0 emits `X + 0 * 1 + 0 = X`; slot 1 emits
    ///   `X + 1 * 1 + 0 = X + 1`; slot 2 is padding (`slot >= K_0`) and is
    ///   excluded.
    ///
    /// Resulting mask: `[X, X + 1]`.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `lag_counts.len() != hydro_count`,
    /// `anticipated_lead_stages.len() != n_anticipated`, any
    /// `lag_counts[h] > max_par_order`, or any
    /// `anticipated_lead_stages[p] > k_max`.
    pub fn set_nonzero_mask(&mut self, lag_counts: &[usize], anticipated_lead_stages: &[usize]) {
        debug_assert_eq!(lag_counts.len(), self.hydro_count);
        debug_assert_eq!(anticipated_lead_stages.len(), self.n_anticipated);

        let n_lag_active: usize = lag_counts.iter().copied().sum();
        let n_ant_active: usize = anticipated_lead_stages.iter().copied().sum();
        let mut mask = Vec::with_capacity(self.hydro_count + n_lag_active + n_ant_active);

        // Storage indices.
        for h in 0..self.hydro_count {
            mask.push(h);
        }

        // Lag indices: lag-major layout, iterate lag-first to produce sorted indices.
        for lag in 0..self.max_par_order {
            for (h, &lag_count) in lag_counts.iter().enumerate() {
                debug_assert!(lag_count <= self.max_par_order);
                if lag < lag_count {
                    mask.push(self.inflow_lags.start + lag * self.hydro_count + h);
                }
            }
        }

        // Anticipated state: slots `0..K_i` for plant `i`.
        // Layout: anticipated_state.start + slot * n_anticipated + plant.
        for slot in 0..self.k_max {
            for (plant, &k_i) in anticipated_lead_stages.iter().enumerate() {
                debug_assert!(k_i <= self.k_max);
                if slot < k_i {
                    mask.push(self.anticipated_state.start + slot * self.n_anticipated + plant);
                }
            }
        }

        debug_assert!(
            mask.windows(2).all(|w| w[0] < w[1]),
            "nonzero_state_indices must be sorted and unique"
        );

        self.nonzero_state_indices = mask;
    }

    /// Finalize a test-constructed indexer the way production
    /// [`build_wired_indexer`](crate::setup) does: populate the nonzero-state
    /// mask, then the `state_to_lp_column` precompute map.
    ///
    /// Production studies build their mask from per-hydro
    /// `PrecomputedPar::effective_lag_count`, but test indexers built via
    /// [`StageIndexer::new`] / [`StageIndexer::with_equipment_and_evaporation`]
    /// have no PAR model. This helper uses the full `max_par_order` for every
    /// hydro (so the mask covers every lag slot — the same coverage the old
    /// dense path emitted) and the indexer's own `anticipated_lead_stages`. For
    /// a storage-only indexer this yields mask `[0, n_state)` ascending,
    /// reproducing the pre-unification dense output byte-for-byte.
    ///
    /// Use this at any test site that feeds the indexer into the forward-pass
    /// cut-row hot path (`build_cut_row_batch_into`), which is mask-driven and
    /// requires a finalized mask.
    #[cfg(test)]
    pub fn finalize_for_test(&mut self) {
        let lag_counts = vec![self.max_par_order; self.hydro_count];
        let anticipated_k = self.anticipated_lead_stages.clone();
        self.set_nonzero_mask(&lag_counts, &anticipated_k);
        self.finalize_state_column_map();
    }
}

#[cfg(test)]
mod tests {
    use crate::indexer::test_fixtures::{eq_with_anticipated, evap, fpha};
    use crate::indexer::{EquipmentCounts, StageIndexer};

    // ── Nonzero state mask tests ───────────────────────────────────────────

    #[test]
    fn nonzero_mask_default_is_empty() {
        let idx = StageIndexer::new(4, 6);
        assert!(
            idx.nonzero_state_indices.is_empty(),
            "default mask must be empty (dense path)"
        );
    }

    #[test]
    fn nonzero_mask_mixed_ar_orders() {
        // 4 hydros (N=4), max_par_order=6 (L=6), ar_orders=[0, 1, 3, 6]
        // inflow_lags.start = N = 4
        // Lag-major layout: slot = 4 + lag * N + h
        let mut idx = StageIndexer::new(4, 6);
        idx.set_nonzero_mask(&[0, 1, 3, 6], &[]);

        // Storage: [0, 1, 2, 3]
        // lag0: h1→4+0*4+1=5, h2→6, h3→7
        // lag1: h2→4+1*4+2=10, h3→11
        // lag2: h2→4+2*4+2=14, h3→15
        // lag3: h3→4+3*4+3=19
        // lag4: h3→4+4*4+3=23
        // lag5: h3→4+5*4+3=27
        // Total: 4 + 0 + 1 + 3 + 6 = 14
        assert_eq!(
            idx.nonzero_state_indices.len(),
            14,
            "mask length: 4 storage + 0 + 1 + 3 + 6 = 14"
        );

        assert_eq!(&idx.nonzero_state_indices[..4], &[0, 1, 2, 3]);
        assert_eq!(
            &idx.nonzero_state_indices[4..],
            &[5, 6, 7, 10, 11, 14, 15, 19, 23, 27]
        );

        assert!(idx.nonzero_state_indices.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn nonzero_mask_zero_par_order() {
        // max_par_order=0: no lags, mask = storage only
        let mut idx = StageIndexer::new(3, 0);
        idx.set_nonzero_mask(&[0, 0, 0], &[]);
        assert_eq!(idx.nonzero_state_indices.len(), 3);
        assert_eq!(&idx.nonzero_state_indices, &[0, 1, 2]);
    }

    #[test]
    fn nonzero_mask_all_full_order() {
        // All hydros at max AR order: mask covers all n_state indices
        let mut idx = StageIndexer::new(2, 3);
        idx.set_nonzero_mask(&[3, 3], &[]);
        // n_state = 2*(1+3) = 8, mask should have 2 + 2*3 = 8
        assert_eq!(idx.nonzero_state_indices.len(), 8);
        assert_eq!(idx.nonzero_state_indices.len(), idx.n_state);
    }

    /// Regression test for the PAR(p)-A cut sparse-mask bug.
    ///
    /// `lag_counts` is the per-hydro count of lag-state
    /// slots that may carry non-zero cut coefficients — equal to
    /// `PrecomputedPar::effective_lag_count(h)`. When PAR(p)-A annual is active
    /// on a hydro this is `max_par_order` (= 12) even though the classical AR
    /// order is smaller, because `ψ̂/12` fills the trailing lag slots.
    ///
    /// Passing `par.order(h)` here instead of `effective_lag_count(h)` omits
    /// state coefficients on slots `order..max_par_order`, producing
    /// over-estimating cuts (LB > UB at convergence).
    #[test]
    fn nonzero_mask_par_a_includes_full_psi_stride() {
        // Two hydros: hydro 0 has classical AR(4); hydro 1 has PAR(4)-A and
        // therefore uses all 12 lag slots. max_par_order = 12 (widened by
        // PrecomputedPar when any model has an annual component).
        let mut idx = StageIndexer::new(2, 12);
        idx.set_nonzero_mask(&[4, 12], &[]);

        // n_state = 2 * (1 + 12) = 26.
        // Mask = [storage 0..2] + [lag * 2 + h for lag in 0..lag_count[h]]
        //      = [0, 1] + [hydro 0 lags 0..4] + [hydro 1 lags 0..12]
        //      = 2 + 4 + 12 = 18 entries.
        assert_eq!(
            idx.nonzero_state_indices.len(),
            18,
            "PAR-A hydro must contribute all 12 lag slots to the cut mask; \
             omitting slots 4..12 (where ψ̂/12 lives) shifts the cut hyperplane \
             above the LP value at the visited state (over-estimating cuts)."
        );

        // Storage indices.
        assert_eq!(&idx.nonzero_state_indices[..2], &[0, 1]);

        // Hydro 0 (lag_count = 4): expect lag slots at indices
        //   inflow_lags.start + lag * hydro_count + h = 2 + lag*2 + 0 for lag in 0..4
        // → {2, 4, 6, 8}.
        // Hydro 1 (lag_count = 12): expect lag slots at indices
        //   2 + lag*2 + 1 for lag in 0..12 → {3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25}.
        // Mask is sorted globally. Confirm a few discriminating positions.
        assert!(
            idx.nonzero_state_indices.contains(&25),
            "lag-11 slot for hydro 1 (the trailing PAR-A annual slot) must be in the mask"
        );
        assert!(
            !idx.nonzero_state_indices.contains(&10),
            "lag-4 slot for hydro 0 (classical AR(4)) must NOT be in the mask"
        );
        // Sorted.
        assert!(idx.nonzero_state_indices.windows(2).all(|w| w[0] < w[1]));
    }

    // ── Anticipated-state nonzero mask tests ───────────────────────────────

    /// Every anticipated plant uses every slot (`K_i == k_max`): all
    /// `n_anticipated * k_max` anticipated indices are included in the mask.
    #[test]
    fn nonzero_mask_anticipated_state_full_kmax() {
        // 2 anticipated plants, k_max = 3, no hydros, no lags.
        // anticipated_state.start = 0 (no storage, no lag block).
        // Layout: start + slot * n_anticipated + plant.
        let mut idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(0, 0, 0, 0, 1, 1, false, 2, 3),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert_eq!(idx.anticipated_state.start, 0);
        assert_eq!(idx.n_anticipated, 2);
        assert_eq!(idx.k_max, 3);

        idx.set_nonzero_mask(&[], &[3, 3]);

        // Every slot is occupied, so all 6 indices appear.
        // slot=0: 0, 1; slot=1: 2, 3; slot=2: 4, 5.
        assert_eq!(idx.nonzero_state_indices, vec![0, 1, 2, 3, 4, 5]);
    }

    /// `K_i < k_max` for some plants: padded slots are excluded.
    /// Configuration: `n_anticipated = 2`, `k_max = 3`,
    /// `anticipated_lead_stages = [3, 1]`.
    #[test]
    fn nonzero_mask_anticipated_state_partial_padding() {
        // 3 hydros, max_par_order = 2, 2 anticipated plants, k_max = 3.
        // inflow_lags = [3, 9), anticipated_state.start = 9.
        let mut idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, 2, 3),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert_eq!(idx.anticipated_state.start, 9);
        idx.set_nonzero_mask(&[2, 2, 2], &[3, 1]);

        // Storage [0, 1, 2] + lag (h0,h1,h2 full = 6 slots) +
        // anticipated: slot=0 plant=0 → 9, plant=1 → 10 (K_1=1 so slot 0 included);
        //              slot=1 plant=0 → 11 (K_0=3); plant=1 → padded (slot 1 >= 1).
        //              slot=2 plant=0 → 13 (K_0=3); plant=1 → padded.
        // The anticipated portion expected: [9, 10, 11, 13].
        let mask = &idx.nonzero_state_indices;
        let ant_portion: Vec<usize> = mask.iter().copied().filter(|&i| i >= 9).collect();
        assert_eq!(ant_portion, vec![9, 10, 11, 13]);
        // Padded slots NOT present.
        assert!(!mask.contains(&12));
        assert!(!mask.contains(&14));
    }

    /// Anticipated-only: no hydros, no lags, only anticipated state.
    #[test]
    fn nonzero_mask_anticipated_state_only_no_hydros() {
        let mut idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(0, 0, 0, 0, 1, 1, false, 1, 2),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert_eq!(idx.hydro_count, 0);
        assert_eq!(idx.anticipated_state.start, 0);

        idx.set_nonzero_mask(&[], &[2]);

        // Only anticipated indices: slot=0 plant=0 → 0; slot=1 plant=0 → 1.
        assert_eq!(idx.nonzero_state_indices, vec![0, 1]);
    }

    /// Heterogeneous `K_i` across plants, including a plant with `K_i = k_max`
    /// and another with `K_i < k_max`.
    #[test]
    fn nonzero_mask_anticipated_state_mixed_k_values() {
        // n_anticipated = 3, k_max = 4. Lead stages = [4, 2, 1].
        // anticipated_state.start = 0 (no hydros).
        // eq_with_anticipated defaults to K_i = k_max for all plants, but we
        // need [4, 2, 1]. Build `EquipmentCounts` directly to override the
        // anticipated_lead_stages field.
        let counts = EquipmentCounts {
            hydro_count: 0,
            max_par_order: 0,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 1,
            n_blks: 1,
            has_inflow_penalty: false,
            max_deficit_segments: 1,
            n_anticipated: 3,
            k_max: 4,
            anticipated_lead_stages: vec![4, 2, 1],
            anticipated_thermal_indices: vec![0, 1, 2],
            n_pumping: 0,
        };
        let mut idx = StageIndexer::with_equipment_and_evaporation(
            &counts,
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        idx.set_nonzero_mask(&[], &[4, 2, 1]);

        // slot=0 plant=0→0, plant=1→1, plant=2→2 (all K_i > 0).
        // slot=1 plant=0→3, plant=1→4 (K_1=2). plant=2 padded.
        // slot=2 plant=0→6 (K_0=4). plant=1 padded. plant=2 padded.
        // slot=3 plant=0→9 (K_0=4). Others padded.
        // Expected: [0, 1, 2, 3, 4, 6, 9].
        assert_eq!(idx.nonzero_state_indices, vec![0, 1, 2, 3, 4, 6, 9]);
    }

    /// `n_anticipated == 0` reproduces the pre-anticipated behaviour exactly.
    #[test]
    fn nonzero_mask_anticipated_state_zero_anticipated_matches_existing() {
        let mut idx_with = StageIndexer::new(4, 6);
        idx_with.set_nonzero_mask(&[0, 1, 3, 6], &[]);

        // Same expected mask as the existing `nonzero_mask_mixed_ar_orders`
        // test: [0,1,2,3] (storage) + [5,6,7,10,11,14,15,19,23,27] (lags).
        assert_eq!(
            idx_with.nonzero_state_indices,
            vec![0, 1, 2, 3, 5, 6, 7, 10, 11, 14, 15, 19, 23, 27]
        );
    }

    /// The extended mask is sorted ascending with no duplicates.
    #[test]
    fn nonzero_mask_anticipated_state_sorted_ascending() {
        // Mixed configuration: 3 hydros with mixed lag_counts + 2 anticipated
        // plants with mixed K_i. The slot-major iteration over anticipated
        // must keep the global mask sorted.
        let mut idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, 2, 3),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        idx.set_nonzero_mask(&[1, 2, 0], &[2, 3]);

        assert!(
            idx.nonzero_state_indices.windows(2).all(|w| w[0] < w[1]),
            "mask must be strictly ascending with no duplicates: {:?}",
            idx.nonzero_state_indices
        );
    }

    /// Plant with `K_i == k_max` (boundary, no padding): all its slots are
    /// included.
    #[test]
    fn nonzero_mask_anticipated_state_boundary_k_eq_kmax() {
        // 1 anticipated plant, K_0 = k_max = 3, no hydros.
        let mut idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(0, 0, 0, 0, 1, 1, false, 1, 3),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        idx.set_nonzero_mask(&[], &[3]);

        // All k_max slots included: slot 0,1,2 → indices 0,1,2.
        assert_eq!(idx.nonzero_state_indices, vec![0, 1, 2]);
    }

    /// `K_i == 0` excludes all slots for that plant (defensive — the parse
    /// layer rejects `K_i == 0`, but the helper must remain robust if
    /// invoked with zero).
    #[test]
    fn nonzero_mask_anticipated_state_boundary_k_zero_excluded() {
        // 2 anticipated plants, k_max = 2. Lead stages = [2, 0].
        let counts = EquipmentCounts {
            hydro_count: 0,
            max_par_order: 0,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 1,
            n_blks: 1,
            has_inflow_penalty: false,
            max_deficit_segments: 1,
            n_anticipated: 2,
            k_max: 2,
            anticipated_lead_stages: vec![2, 0],
            anticipated_thermal_indices: vec![0, 1],
            n_pumping: 0,
        };
        let mut idx = StageIndexer::with_equipment_and_evaporation(
            &counts,
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        idx.set_nonzero_mask(&[], &[2, 0]);

        // Plant 0 (K_0=2) emits slot=0→0, slot=1→2. Plant 1 (K_1=0) emits
        // nothing. Expected mask: [0, 2].
        assert_eq!(idx.nonzero_state_indices, vec![0, 2]);
    }
}
