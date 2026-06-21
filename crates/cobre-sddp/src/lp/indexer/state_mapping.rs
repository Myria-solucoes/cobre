//! State-vector-to-LP-column resolvers for [`StageIndexer`].
//!
//! Owns the state-pinning contract (state pinning uses column bounds, not
//! equality rows):
//! [`StageIndexer::state_to_lp_incoming_column`] is the single authoritative
//! incoming-state column resolver; the LP column for both pinning and dual
//! extraction is always resolved through it, never by assuming a fixing-row
//! index. The companion [`StageIndexer::state_to_lp_column`] maps the outgoing
//! state vector to the LP columns a forward-pass cut row references.

use super::layout::StageIndexer;

impl StageIndexer {
    /// Map a state-vector index to the LP column it should reference in a cut.
    ///
    /// The outgoing state after `shift_lag_state` stores:
    /// - `[0, N)`: outgoing storage → LP column `j` (identity mapping)
    /// - `[N + 0·N + h]`: outgoing lag 0 for hydro `h` = realised inflow
    ///   → LP column `z_inflow.start + h`
    /// - `[N + l·N + h]` for `l ≥ 1`: outgoing lag `l` = incoming lag `l − 1`
    ///   → LP column `N + (l − 1)·N + h`
    /// - `[N*(1+L), N*(1+L) + n_anticipated*K_max)`: `anticipated_state` slots
    ///   → shift-aware mapping (mirrors the inflow-lag pattern structurally):
    ///   - `slot == K_p − 1` for plant `p`: the post-shift outgoing slot carries
    ///     the decision committed at stage `t`. The Equal branch returns
    ///     `anticipated_state_out.start + p`. That target column is
    ///     stage-invariant: `anticipated_state_out` lives in the state region
    ///     (immediately after the `anticipated_state` ring buffer), so its
    ///     offset is a pure function of `N`, `L`, `A`, `k_max` and the single
    ///     global stage-0 cut map resolves onto the correct column at every
    ///     stage regardless of per-stage block counts. The column is pinned to
    ///     `decision_col[p]` by the `anticipated_state_out_def` equality row
    ///     (`anticipated_state_out[p] − decision_col[p] = 0`). The state-fixing
    ///     row at slot `K_p − 1` is PURE IDENTITY under this layout (no
    ///     decision-write coefficient).
    ///   - `slot < K_p − 1`: the successor's slot `i` = predecessor's incoming
    ///     slot `i + 1` (shift); returns
    ///     `anticipated_state.start + (slot + 1) * n_anticipated + p`.
    ///   - `slot > K_p − 1`: padding (unused for this plant, pinned to 0 by the
    ///     state-fixing row); returns `j` (identity; safe default).
    ///
    /// The `anticipated_state` branch is evaluated regardless of `max_par_order`
    /// because the shift semantics apply even when there are no inflow lags.
    /// The `max_par_order == 0` early-return only guards the lag-remap block
    /// (which has zero length when `max_par_order == 0`).
    #[inline]
    #[must_use]
    pub fn state_to_lp_column(&self, j: usize) -> usize {
        let n = self.hydro_count;
        if j < n {
            return j;
        }
        // Anticipated-state mapping (must check before early return on max_par_order == 0).
        if self.n_anticipated > 0 && j >= self.anticipated_state.start {
            let ant_block_size = self.n_anticipated * self.k_max;
            if j < self.anticipated_state.start + ant_block_size {
                let offset = j - self.anticipated_state.start;
                let slot = offset / self.n_anticipated;
                let plant = offset % self.n_anticipated;
                let k_p = self.anticipated_lead_stages[plant];
                return match (slot + 1).cmp(&k_p) {
                    std::cmp::Ordering::Equal => self.anticipated_state_out.start + plant,
                    std::cmp::Ordering::Less => {
                        self.anticipated_state.start + (slot + 1) * self.n_anticipated + plant
                    }
                    // INVARIANT: Padding slot — `slot >= k_p` means this ring-buffer
                    // entry belongs to plant `plant` but exceeds its lead time `K_i`.
                    // Padding slots exist because the ring buffer is sized to `k_max`
                    // slots (the system-wide maximum), but plant `plant` only uses
                    // `k_p = K_i` of them. These slots are safe to pass through as
                    // their own state-column index (not a decision-variable column)
                    // because the following 5-step chain guarantees their LP dual is 0:
                    //   1. `shift_anticipated_state` initialises padding slots to 0.0.
                    //   2. The corresponding state-fixing row has RHS 0 (from step 1).
                    //   3. The LP solver pins the slot value to 0 via the equality row.
                    //   4. A zero-valued variable at a zero-RHS equality has dual 0.
                    //   5. Zero duals produce zero cut coefficients, which are no-ops
                    //      in the cut row (neither pruned nor corrupted).
                    //
                    // Pre-horizon seeding is implemented (see `setup/mod.rs` —
                    // `setup_anticipated_state`). The padding-zero invariant is
                    // preserved because seeds populate slots `[0, K_i)` only;
                    // slots `[K_i, k_max)` remain zero (debug_assert! guards
                    // this in `setup/mod.rs`). The identity return is therefore
                    // safe for padding slots under always-active fishing.
                    std::cmp::Ordering::Greater => j,
                };
            }
            return j;
        }
        if self.max_par_order == 0 {
            return j;
        }
        // Lag block: slot-to-column mapping.
        let offset = j - n;
        let h = offset % n;
        let lag = offset / n;
        if lag == 0 {
            self.z_inflow.start + h
        } else {
            n + (lag - 1) * n + h
        }
    }

    /// Fill [`state_to_lp_column_map`](Self::state_to_lp_column_map) by calling
    /// [`state_to_lp_column`](Self::state_to_lp_column) for every
    /// `j ∈ [0, n_state)`.
    ///
    /// Call once after the state layout is finalized (e.g. after
    /// [`set_nonzero_mask`](Self::set_nonzero_mask) in study setup). The map is a
    /// pure cache of the resolver — it never reimplements the mapping arithmetic.
    pub fn finalize_state_column_map(&mut self) {
        self.state_to_lp_column_map.clear();
        self.state_to_lp_column_map.reserve(self.n_state);
        for j in 0..self.n_state {
            self.state_to_lp_column_map.push(self.state_to_lp_column(j));
        }
        debug_assert_eq!(self.state_to_lp_column_map.len(), self.n_state);
    }

    /// Read the precomputed `state_to_lp_column(j)`, falling back to the live
    /// resolver when the map has not been finalized.
    ///
    /// The fallback guarantees correctness for any indexer (e.g. test indexers
    /// that skip [`finalize_state_column_map`](Self::finalize_state_column_map))
    /// without editing every construction site.
    #[inline]
    #[must_use]
    pub fn lp_column_for_state(&self, j: usize) -> usize {
        self.state_to_lp_column_map
            .get(j)
            .copied()
            .unwrap_or_else(|| self.state_to_lp_column(j))
    }

    /// Map a state-vector index to the LP column pinned by
    /// [`fill_col_state_patches`](crate::lp_builder::PatchBuffer::fill_col_state_patches).
    ///
    /// This is the **incoming-state column** — the column whose bound is set to
    /// `lb = ub = v` when state-fixing is applied via `set_col_bounds`. The
    /// column indices returned here are exactly those written into
    /// `PatchBuffer::col_indices[..state_col_patch_count()]` by
    /// `fill_col_state_patches`, in state-vector order.
    ///
    /// Use this method in the backward pass to read `view.reduced_costs[col]`
    /// for the cut subgradient, one entry per state-vector component
    /// `j ∈ [0, n_state)`.
    ///
    /// ## Mapping by range
    ///
    /// - `j ∈ [0, N)` (storage): returns `self.storage_in.start + j`.
    /// - `j ∈ [N, N*(1+L))` (AR lags): returns
    ///   `self.inflow_lags.start + (j − N)`.
    /// - `j ∈ [N*(1+L), n_state)` (anticipated state): returns
    ///   `self.anticipated_state.start + (j − N*(1+L))`.
    ///
    /// ## Contrast with [`state_to_lp_column`]
    ///
    /// [`state_to_lp_column`] returns the **outgoing** column used for
    /// cut-row coefficient construction in the forward pass (`forward.rs`).
    /// For the storage range, `state_to_lp_column(j) = j` (the outgoing
    /// storage column), while this method returns `storage_in.start + j`
    /// (the incoming storage column). The two columns are related via the
    /// water-balance equality row, and by KKT duality the reduced cost on
    /// the incoming column equals the dual of the equivalent equality row
    /// that a row-based state-fixing formulation would produce.
    ///
    /// [`state_to_lp_column`]: Self::state_to_lp_column
    #[inline]
    #[must_use]
    pub fn state_to_lp_incoming_column(&self, j: usize) -> usize {
        let n = self.hydro_count;
        let lag_end = n * (1 + self.max_par_order);
        if j < n {
            self.storage_in.start + j
        } else if j < lag_end {
            self.inflow_lags.start + (j - n)
        } else {
            self.anticipated_state.start + (j - lag_end)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::indexer::test_fixtures::{eq_with_anticipated, evap, fpha};
    use crate::indexer::{EquipmentCounts, StageIndexer};

    // ── state_to_lp_column precompute tests ─────────────────────────────────

    /// A finalized indexer carrying storage + AR lags + anticipated thermals
    /// (every `state_to_lp_column` branch) must have
    /// `lp_column_for_state(j) == state_to_lp_column(j)` for every state index.
    #[test]
    fn lp_column_map_matches_resolver_with_lags_and_anticipated() {
        // hydro_count=3, max_par_order=2, n_anticipated=2 (K = [1, 2], k_max=2).
        let mut idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 3,
                max_par_order: 2,
                n_thermals: 2,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 2,
                k_max: 2,
                anticipated_lead_stages: vec![1, 2],
                anticipated_thermal_indices: vec![0, 1],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Finalize both layout-derived caches, as study setup does.
        let lag_counts = vec![2_usize; idx.hydro_count];
        let anticipated_k = idx.anticipated_lead_stages.clone();
        idx.set_nonzero_mask(&lag_counts, &anticipated_k);
        idx.finalize_state_column_map();

        assert_eq!(idx.state_to_lp_column_map.len(), idx.n_state);
        for j in 0..idx.n_state {
            assert_eq!(
                idx.lp_column_for_state(j),
                idx.state_to_lp_column(j),
                "finalized map must match the resolver at j={j}"
            );
        }

        // Un-finalized clone: the fallback must still match the resolver.
        let mut unfinalized = idx.clone();
        unfinalized.state_to_lp_column_map.clear();
        for j in 0..unfinalized.n_state {
            assert_eq!(
                unfinalized.lp_column_for_state(j),
                unfinalized.state_to_lp_column(j),
                "fallback must match the resolver at j={j} when un-finalized"
            );
        }
    }

    /// Storage-only (`max_par_order == 0`, no anticipated): the mask is exactly
    /// `[0, n_state)` ascending and `lp_column_for_state(j) == j` — the
    /// dense→sparse bit-identity premise for the unified cut-row loop.
    #[test]
    fn lp_column_map_storage_only_mask_is_full_range() {
        let mut idx = StageIndexer::new(3, 0);
        idx.set_nonzero_mask(&[0, 0, 0], &[]);
        idx.finalize_state_column_map();

        assert_eq!(idx.nonzero_state_indices, vec![0, 1, 2]);
        assert_eq!(idx.nonzero_state_indices.len(), idx.n_state);
        for j in 0..idx.n_state {
            assert_eq!(idx.lp_column_for_state(j), j);
        }
    }

    // ── state_to_lp_column tests ──────────────────────────────────────────────

    /// R4.a: `anticipated_state` indices use the shift-aware mapping when
    /// `max_par_order == 0 && n_anticipated > 0`. The `anticipated_state` branch
    /// runs before the `max_par_order == 0` lag-block guard; verify the shift
    /// semantics apply even when there are no inflow lags.
    ///
    /// Assertions reflect the shift-aware mapping (the anticipated-state
    /// branch in `state_to_lp_column` runs before the `max_par_order==0`
    /// early-return).
    #[test]
    fn state_to_lp_column_anticipated_identity_no_lag() {
        // N=1, L=0, n_anticipated=1, k_max=2, anticipated_lead_stages=[2].
        // n_state = 1*(1+0) + 1*2 = 3.
        // anticipated_state = [1, 3); slot 0 at j=1, slot 1 at j=2.
        // anticipated_state.start = N*(1+L) = 1.
        // anticipated_decision.start = theta+1 = 6 (no hydro/thermal cols).
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(1, 0, 0, 0, 1, 1, false, 1, 2),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Storage index: identity.
        assert_eq!(idx.state_to_lp_column(0), 0);
        // Anticipated-state slot 0 (j=1): slot+1=1 < k_p=2 → shift to slot 1.
        // Returns anticipated_state.start + 1*n_anticipated + 0 = 1+1 = 2.
        assert_eq!(idx.state_to_lp_column(1), 2);
        // Anticipated-state slot 1 (j=2): slot+1=2 == k_p=2 → state-out channel.
        // Returns anticipated_state_out.start + 0.
        assert_eq!(idx.state_to_lp_column(2), idx.anticipated_state_out.start);
    }

    /// R4.b: `anticipated_state` indices use the shift-aware mapping when
    /// `max_par_order > 0 && n_anticipated > 0`.  The fixture uses N=1, L=1,
    /// `n_anticipated=1`, `K_max=2`, `anticipated_lead_stages=[2]`.
    ///
    /// Assertions reflect the shift-aware mapping (identity is wrong for
    /// `K_max >= 2` because the decision-write slot `K_p - 1` and the shifted
    /// slots map to different LP columns than the incoming state's own).
    #[test]
    fn state_to_lp_column_anticipated_identity_with_lag() {
        // N=1, L=1, n_anticipated=1, k_max=2, anticipated_lead_stages=[2].
        // n_state = 1*(1+1) + 1*2 = 4.
        // Layout: j=0 storage, j=1 lag-0, j=2 ant slot-0, j=3 ant slot-1.
        // anticipated_state.start = N*(1+L) = 2.
        // z_inflow.start = anticipated_state_end = 2 + 2 = 4.
        // anticipated_decision.start = theta + 1 = 7 (no hydro/thermal columns).
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(1, 1, 0, 0, 1, 1, false, 1, 2),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Storage: identity.
        assert_eq!(idx.state_to_lp_column(0), 0);
        // Lag block: remapped.
        // j=1: offset=0, h=0, lag=0 → z_inflow.start + 0 = 4.
        assert_eq!(idx.state_to_lp_column(1), idx.z_inflow.start);
        // Anticipated-state slot 0 (j=2): slot+1=1 < k_p=2 → shift to slot 1.
        // Returns anticipated_state.start + 1*n_anticipated + 0 = 2+1 = 3.
        assert_eq!(idx.state_to_lp_column(2), 3);
        // Anticipated-state slot 1 (j=3): slot+1=2 == k_p=2 → state-out channel.
        // Returns anticipated_state_out.start + 0.
        assert_eq!(idx.state_to_lp_column(3), idx.anticipated_state_out.start);
    }

    /// R4.c: lag-remap branch is preserved when `n_anticipated == 0` and
    /// `max_par_order > 0`.  The anticipated-state guard must not fire
    /// when there are no anticipated thermals.
    #[test]
    fn state_to_lp_column_lag_remap_preserved_no_anticipated() {
        // N=1, L=1, n_anticipated=0 — classic PAR(p) case.
        // n_state = 1*(1+1) = 2. Layout: j=0 storage, j=1 lag-0.
        let idx = StageIndexer::new(1, 1);
        assert_eq!(idx.n_anticipated, 0);
        // Storage: identity.
        assert_eq!(idx.state_to_lp_column(0), 0);
        // Lag block j=1: offset=0, h=0, lag=0 → z_inflow.start + 0.
        // For StageIndexer::new(1,1): z_inflow = N*(1+L)..N*(2+L) = 2..3.
        assert_eq!(idx.z_inflow.start, 2);
        assert_eq!(idx.state_to_lp_column(1), 2);
    }

    /// State-out channel branch: slot `K_p - 1` of an anticipated plant maps to
    /// the `anticipated_state_out` column for that plant (not `anticipated_decision`
    /// directly). The `anticipated_state_out` variable is pinned to the decision
    /// column by the `anticipated_state_out_def` equality row, so cut coefficients
    /// on the state-out column correctly express the Benders subgradient.
    #[test]
    fn state_to_lp_column_anticipated_decision_channel() {
        // N=0, L=0, n_anticipated=1, k_max=2, anticipated_lead_stages=[2].
        // n_state = 0*(1+0) + 1*2 = 2.
        // anticipated_state = [0, 2); slot 0 at j=0, slot 1 at j=1.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(0, 0, 0, 0, 1, 1, false, 1, 2),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Slot K_p - 1 = 1 (the highest slot for plant 0) → anticipated_state_out column.
        let slot_k_minus_1 = idx.anticipated_state.start + (idx.k_max - 1) * idx.n_anticipated;
        assert_eq!(
            idx.state_to_lp_column(slot_k_minus_1),
            idx.anticipated_state_out.start,
        );
        assert_ne!(
            idx.state_to_lp_column(slot_k_minus_1),
            idx.anticipated_decision.start,
            "Equal branch must NOT route to anticipated_decision"
        );
    }

    /// Equal-branch boundary at `k_max == 1`: with a single ring-buffer slot,
    /// slot 0 is the only slot and `slot + 1 == k_p == 1`, so the Equal branch
    /// fires immediately on slot 0 — there is no Less (shift) or Greater
    /// (padding) slot to reach first. The slot must route to the
    /// `anticipated_state_out` column (the Equal-branch target), not back to
    /// `anticipated_decision`. The other anticipated tests use `k_max >= 2`,
    /// where slot 0 takes the Less branch, so this is the only coverage of the
    /// `k_max == 1` Equal path.
    #[test]
    fn state_to_lp_column_equal_branch_k_max_one() {
        // N=0, L=0, n_anticipated=1, k_max=1, anticipated_lead_stages=[1].
        // n_state = 0*(1+0) + 1*1 = 1.
        // anticipated_state = [0, 1); the lone slot 0 is at j=0.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(0, 0, 0, 0, 1, 1, false, 1, 1),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Slot 0 of plant 0: slot+1 = 1 == k_p = 1 → Equal branch → state-out.
        let slot_0 = idx.anticipated_state.start;
        assert_eq!(
            idx.state_to_lp_column(slot_0),
            idx.anticipated_state_out.start,
            "k_max==1 slot 0 must route to anticipated_state_out via the Equal branch",
        );
        assert_ne!(
            idx.state_to_lp_column(slot_0),
            idx.anticipated_decision.start,
            "Equal branch must NOT route to anticipated_decision",
        );
    }

    /// Shift branch: an `anticipated_state` slot `i < K_p - 1` maps to the
    /// predecessor stage's `anticipated_state` column at slot `i + 1` (the
    /// shift). Successor's slot `i` comes from predecessor's incoming slot
    /// `i + 1` after `shift_anticipated_state` runs.
    #[test]
    fn state_to_lp_column_anticipated_shift() {
        // Same fixture as R5.a: single plant, K=2.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(0, 0, 0, 0, 1, 1, false, 1, 2),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Slot 0 → shift to slot 1's column.
        let slot_0 = idx.anticipated_state.start;
        let slot_1 = idx.anticipated_state.start + idx.n_anticipated;
        assert_eq!(idx.state_to_lp_column(slot_0), slot_1);
    }

    /// Padding branch: an `anticipated_state` slot `i > K_p - 1` (padding
    /// for a plant with `K_p < K_max`) maps to identity `j`. Padding slots
    /// are pinned to 0 by the state-fixing row, so the identity mapping is a
    /// safe default that does not introduce wrong cuts.
    #[test]
    fn state_to_lp_column_anticipated_padding_slot_identity() {
        // Two plants: plant 0 has K_p=1 (only slot 0 is in-use), plant 1 has
        // K_p=3 (slots 0, 1, 2 all in-use). k_max=3 so plant 0 has padding
        // at slots 1 and 2.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 0,
                max_par_order: 0,
                n_thermals: 2,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 2,
                k_max: 3,
                anticipated_lead_stages: vec![1, 3],
                anticipated_thermal_indices: vec![0, 1],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Plant 0 padding: slot 1 at j = ant_start + 1*2 + 0, slot 2 at j = ant_start + 2*2 + 0.
        let pad_slot_1_plant_0 = idx.anticipated_state.start + idx.n_anticipated;
        let pad_slot_2_plant_0 = idx.anticipated_state.start + 2 * idx.n_anticipated;
        assert_eq!(
            idx.state_to_lp_column(pad_slot_1_plant_0),
            pad_slot_1_plant_0
        );
        assert_eq!(
            idx.state_to_lp_column(pad_slot_2_plant_0),
            pad_slot_2_plant_0
        );
    }

    /// Multi-plant layout: correct routing for all `(slot, plant)`
    /// combinations in a two-plant K=2 fixture.
    #[test]
    fn state_to_lp_column_anticipated_multi_plant_layout() {
        // Two plants, both with K_p=2; k_max=2.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 0,
                max_par_order: 0,
                n_thermals: 2,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 2,
                k_max: 2,
                anticipated_lead_stages: vec![2, 2],
                anticipated_thermal_indices: vec![0, 1],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Layout: slot * n_anticipated + plant. n_anticipated=2.
        //   j=ant_start+0 → slot 0, plant 0; shift → ant_start + 1*2 + 0 = ant_start + 2
        //   j=ant_start+1 → slot 0, plant 1; shift → ant_start + 1*2 + 1 = ant_start + 3
        //   j=ant_start+2 → slot 1, plant 0; state-out → anticipated_state_out.start + 0
        //   j=ant_start+3 → slot 1, plant 1; state-out → anticipated_state_out.start + 1
        let s = idx.anticipated_state.start;
        let so = idx.anticipated_state_out.start;
        assert_eq!(idx.state_to_lp_column(s), s + 2);
        assert_eq!(idx.state_to_lp_column(s + 1), s + 3);
        assert_eq!(idx.state_to_lp_column(s + 2), so);
        assert_eq!(idx.state_to_lp_column(s + 3), so + 1);
    }

    // ── state_to_lp_incoming_column tests ────────────────────────────────────

    /// Storage range: for a `with_equipment` indexer with `N=3, L=2, A=0`,
    /// `state_to_lp_incoming_column(j)` for `j ∈ [0, N)` returns
    /// `storage_in.start + j`.
    #[test]
    fn state_to_lp_incoming_column_storage_range() {
        // N=3, L=2: storage_in.start = N*(2+L) = 3*4 = 12.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, 0, 0),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // storage_in.start should be N*(2+L) = 12.
        assert_eq!(idx.storage_in.start, 12);
        for j in 0..3_usize {
            assert_eq!(
                idx.state_to_lp_incoming_column(j),
                idx.storage_in.start + j,
                "j={j}: expected storage_in.start + {j}"
            );
        }
    }

    /// AR lag range: for a `with_equipment` indexer with `N=3, L=2, A=0`,
    /// `state_to_lp_incoming_column(j)` for `j ∈ [N, N*(1+L))` returns
    /// `inflow_lags.start + (j − N)`.
    #[test]
    fn state_to_lp_incoming_column_lag_range() {
        // N=3, L=2: inflow_lags = 3..9.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, 0, 0),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert_eq!(idx.inflow_lags.start, 3);
        for j in 3..9_usize {
            assert_eq!(
                idx.state_to_lp_incoming_column(j),
                idx.inflow_lags.start + (j - 3),
                "j={j}: expected inflow_lags.start + {}",
                j - 3
            );
        }
    }

    /// Anticipated-state range: for an indexer with `N=0, L=0, A=1, K=2`,
    /// `state_to_lp_incoming_column(j)` for `j ∈ [0, n_state)` returns
    /// `anticipated_state.start + j` (since `lag_end` = N*(1+L) = 0).
    #[test]
    fn state_to_lp_incoming_column_anticipated_range() {
        // N=0, L=0, A=1, K=2: n_state = 0 + 1*2 = 2.
        // anticipated_state.start = N*(1+L) = 0.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(0, 0, 0, 0, 1, 1, false, 1, 2),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert_eq!(idx.anticipated_state.start, 0);
        assert_eq!(idx.n_state, 2);
        for j in 0..2_usize {
            assert_eq!(
                idx.state_to_lp_incoming_column(j),
                idx.anticipated_state.start + j,
                "j={j}: expected anticipated_state.start + {j}"
            );
        }
    }

    /// Combined boundary-case test: `N=3, L=2, A=1, K=2`.
    /// Checks j = 0, 2, 3, 8, 9, 10 (the boundary points from the spec).
    #[test]
    fn state_to_lp_incoming_column_combined_with_equipment_indexer() {
        // N=3, L=2, A=1, K=2:
        //   n_state = N*(1+L) + A*K = 3*3 + 1*2 = 11.
        //   storage_in.start = N*(2+L) + A*K_max = 3*4 + 1*2 = 14.
        //   inflow_lags.start = N = 3.
        //   anticipated_state.start = N*(1+L) = 9.
        //   lag_end = N*(1+L) = 9.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, 1, 2),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert_eq!(idx.n_state, 11);
        // j=0: storage range → storage_in.start + 0.
        assert_eq!(
            idx.state_to_lp_incoming_column(0),
            idx.storage_in.start,
            "j=0"
        );
        // j=2: storage range → storage_in.start + 2.
        assert_eq!(
            idx.state_to_lp_incoming_column(2),
            idx.storage_in.start + 2,
            "j=2"
        );
        // j=3: first lag → inflow_lags.start + 0.
        assert_eq!(
            idx.state_to_lp_incoming_column(3),
            idx.inflow_lags.start,
            "j=3"
        );
        // j=8: last lag → inflow_lags.start + 5.
        assert_eq!(
            idx.state_to_lp_incoming_column(8),
            idx.inflow_lags.start + 5,
            "j=8"
        );
        // j=9: first anticipated-state → anticipated_state.start + 0.
        assert_eq!(
            idx.state_to_lp_incoming_column(9),
            idx.anticipated_state.start,
            "j=9"
        );
        // j=10: last anticipated-state → anticipated_state.start + 1.
        assert_eq!(
            idx.state_to_lp_incoming_column(10),
            idx.anticipated_state.start + 1,
            "j=10"
        );
        // All returned columns must be within the LP's column range.
        for j in 0..idx.n_state {
            let col = idx.state_to_lp_incoming_column(j);
            assert!(
                col < idx.theta + 1,
                "j={j}: column {col} out of range (theta={})",
                idx.theta
            );
        }
    }

    /// For the lag range, `state_to_lp_incoming_column` and `state_to_lp_column`
    /// return different values under `with_equipment` (the former returns the
    /// incoming-lag column, the latter returns the `z_inflow` or lag-out column).
    /// For the storage range under `with_equipment`, the results differ because
    /// `storage_in.start > 0` while `state_to_lp_column` returns `j` (outgoing,
    /// which starts at column 0).
    #[test]
    fn state_to_lp_incoming_column_differs_from_state_to_lp_column_for_lag() {
        // N=2, L=1: storage_in.start = N*(2+L) = 2*3 = 6.
        // state_to_lp_column(0) = 0 (outgoing storage).
        // state_to_lp_incoming_column(0) = storage_in.start + 0 = 6.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(2, 1, 0, 0, 1, 1, false, 0, 0),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Storage range: incoming ≠ outgoing under with_equipment.
        assert_ne!(
            idx.state_to_lp_incoming_column(0),
            idx.state_to_lp_column(0),
            "storage range should differ: incoming={} outgoing={}",
            idx.state_to_lp_incoming_column(0),
            idx.state_to_lp_column(0)
        );
        // j=0: incoming returns storage_in.start, outgoing returns 0.
        assert_eq!(idx.state_to_lp_incoming_column(0), idx.storage_in.start);
        assert_eq!(idx.state_to_lp_column(0), 0);

        // Lag range (j=2, j=3): incoming returns inflow_lags column;
        // outgoing returns z_inflow (lag=0) or lag-out (lag>=1) column.
        // j=2: lag 0, hydro 0. incoming = inflow_lags.start + 0.
        //                       outgoing = z_inflow.start + 0.
        assert_eq!(
            idx.state_to_lp_incoming_column(2),
            idx.inflow_lags.start,
            "j=2 incoming should be inflow_lags.start"
        );
        assert_eq!(
            idx.state_to_lp_column(2),
            idx.z_inflow.start,
            "j=2 outgoing should be z_inflow.start"
        );
        assert_ne!(
            idx.state_to_lp_incoming_column(2),
            idx.state_to_lp_column(2),
            "lag range should differ for j=2"
        );
        // j=3: lag 0, hydro 1. incoming = inflow_lags.start + 1.
        //                       outgoing = z_inflow.start + 1.
        assert_ne!(
            idx.state_to_lp_incoming_column(3),
            idx.state_to_lp_column(3),
            "lag range should differ for j=3"
        );
    }

    /// The Equal branch resolves the matured anticipated slot into the
    /// **state region**: for `j = anticipated_state.start + (K_p − 1)*A + plant`
    /// it returns `anticipated_state_out.start + plant`, and that column lies
    /// in `[anticipated_state.end, theta)` — i.e. inside the relocated
    /// state-region block, not the control region.
    #[test]
    fn state_to_lp_column_equal_branch_resolves_into_state_region() {
        // N=3, L=2, A=2, k_max=3, uniform K_p = 3.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 4, 0, 1, 1, false, 2, 3),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let a = idx.n_anticipated;
        for plant in 0..a {
            let k_p = idx.anticipated_lead_stages[plant];
            let j = idx.anticipated_state.start + (k_p - 1) * a + plant;
            let col = idx.state_to_lp_column(j);
            assert_eq!(
                col,
                idx.anticipated_state_out.start + plant,
                "Equal branch must return anticipated_state_out.start + plant"
            );
            // Inside the state region: at/after the ring-buffer end, before theta.
            assert!(
                col >= idx.anticipated_state.end,
                "resolved column {col} must be >= anticipated_state.end {}",
                idx.anticipated_state.end
            );
            assert!(
                col < idx.theta,
                "resolved column {col} must be < theta {}",
                idx.theta
            );
        }
    }

    /// Stage-invariance: two indexers with identical `N, L, A, k_max` but
    /// different `n_blks` and `n_thermals` resolve every matured anticipated
    /// slot to the **same** column. This is the property the relocation buys —
    /// the global stage-0 cut map lands on the correct column at every stage
    /// regardless of per-stage block counts. Under the old control-region
    /// placement the Equal-branch target rode the n_blks-dependent `thermal.end`
    /// and would differ between these two indexers.
    #[test]
    fn state_to_lp_column_equal_branch_stage_invariant_across_block_counts() {
        let build = |n_blks: usize, n_thermals: usize| {
            StageIndexer::with_equipment_and_evaporation(
                &EquipmentCounts {
                    hydro_count: 3,
                    max_par_order: 2,
                    n_thermals,
                    n_lines: 1,
                    n_buses: 2,
                    n_blks,
                    has_inflow_penalty: false,
                    max_deficit_segments: 1,
                    n_anticipated: 2,
                    k_max: 3,
                    anticipated_lead_stages: vec![2, 3],
                    anticipated_thermal_indices: vec![0, 1],
                    n_pumping: 0,
                },
                &fpha(vec![], vec![]),
                &evap(vec![]),
            )
        };
        let idx_a = build(1, 2);
        let idx_b = build(4, 5);

        // Same state-region geometry (N, L, A, k_max identical).
        assert_eq!(idx_a.anticipated_state, idx_b.anticipated_state);
        assert_eq!(idx_a.anticipated_state_out, idx_b.anticipated_state_out);
        assert_eq!(idx_a.n_state, idx_b.n_state);

        // But different control-region geometry (proves the test is meaningful).
        assert_ne!(idx_a.thermal.end, idx_b.thermal.end);

        // Every matured anticipated slot resolves to the same column.
        let a = idx_a.n_anticipated;
        for slot in 0..idx_a.k_max {
            for plant in 0..a {
                let j = idx_a.anticipated_state.start + slot * a + plant;
                assert_eq!(
                    idx_a.state_to_lp_column(j),
                    idx_b.state_to_lp_column(j),
                    "slot={slot} plant={plant} must resolve identically across block counts"
                );
            }
        }
    }
}
