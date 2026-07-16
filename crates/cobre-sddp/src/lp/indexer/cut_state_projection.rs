//! Storage-scoped projection of the global [`StateSpace`] onto the cut-state
//! dimensions a stage enables, for cut storage, subgradient extraction (incoming
//! columns), and cut rendering (outgoing columns).

use cobre_core::temporal::StageStateConfig;

use super::{CutSlot, InCol, OutCol, REGION_ORDER, StateDim, StateSpace};

/// A storage-scoped view of the global state vector exposing only the cut-state
/// dimensions a stage enables, plus travel-time buckets and anticipated state
/// (always included, never gated by [`StageStateConfig`]).
///
/// A projection **of** a [`StateSpace`], not a sibling layout: it delegates all
/// column arithmetic to the global layout and never drives the LP.
///
/// ## Default-identity contract
///
/// When `storage: true, inflow_lags: true` the projection is the identity:
/// [`Self::n_slots`] equals [`StateSpace::n_state`],
/// [`Self::incoming_column`] equals the global incoming resolver for
/// every `j`, and [`Self::render_pairs`] reproduces the global
/// `nonzero_state_indices` render exactly. Holds because the construction walks
/// the state index space in `REGION_ORDER`'s storage → lag → buckets →
/// anticipated order — the single owner [`StateSpace::set_nonzero_mask`] also
/// walks. The forbidden alternative is reordering dimensions (e.g. anticipated
/// before buckets), landing a cut coefficient on the wrong LP column and
/// silently corrupting every existing study.
///
/// ## Incoming vs outgoing vs render
///
/// Incoming ([`Self::incoming_column`]) and outgoing
/// ([`Self::outgoing_column`]) both span the full enabled range
/// `[0, n_slots())`, including structurally-zero AR- and anticipated-padding
/// slots, because dual extraction and DCS scoring read every enabled dimension.
/// Render ([`Self::render_pairs`]) is the **nonzero subset** (padding dropped,
/// matching the global `nonzero_state_indices`), because a cut row emits no entry
/// for a structurally-zero coefficient.
#[derive(Debug, Clone)]
pub struct CutStateProjection {
    /// LP incoming column per cut slot (extraction hot path).
    incoming_columns: Vec<InCol>,

    /// LP outgoing column per cut slot (DCS scoring); parallel to
    /// [`Self::incoming_columns`].
    outgoing_columns: Vec<OutCol>,

    /// Cut slot per render pair, nonzero-subset order; parallel to
    /// [`Self::render_columns`].
    render_coeff_indices: Vec<CutSlot>,

    /// LP outgoing column per render pair; parallel to
    /// [`Self::render_coeff_indices`].
    render_columns: Vec<OutCol>,
}

impl CutStateProjection {
    /// Project the global [`StateSpace`] onto the cut-state dimensions
    /// `state_config` enables, with travel-time buckets and anticipated state
    /// always included, walking `REGION_ORDER`'s storage → lag → buckets →
    /// anticipated order (the default-identity contract) via
    /// `StateSpace`'s `state_dim_range`, the sole owner of the boundary
    /// arithmetic.
    ///
    /// The per-slot fan-in cannot swap roles: an outgoing column pushed onto
    /// the incoming-column vector fails to compile.
    ///
    /// ```compile_fail
    /// use cobre_sddp::indexer::{InCol, StateDim, StateSpace};
    ///
    /// let global = StateSpace::new(1, 0, 0, Vec::new(), 0, 0, Vec::new(), &[0]);
    /// let outgoing = global.lp_column_for_state(StateDim::new(0));
    /// let mut incoming_columns: Vec<InCol> = Vec::new();
    /// incoming_columns.push(outgoing); // fan-in swap: outgoing pushed onto the incoming-column vec
    /// ```
    #[must_use]
    pub fn new(global: &StateSpace, state_config: StageStateConfig) -> Self {
        let mut incoming_columns = Vec::new();
        let mut outgoing_columns = Vec::new();
        let mut render_coeff_indices = Vec::new();
        let mut render_columns = Vec::new();

        let mut push_dim = |g: StateDim| {
            let reduced_j = incoming_columns.len();
            let outgoing = global.lp_column_for_state(g);
            incoming_columns.push(global.state_to_lp_incoming_column(g));
            outgoing_columns.push(outgoing);
            // Drop padding slots, never zero-fill: keeps the default render
            // bit-identical to the global nonzero_state_indices render.
            if global.nonzero_state_indices.binary_search(&g).is_ok() {
                render_coeff_indices.push(CutSlot::new(reduced_j));
                render_columns.push(outgoing);
            }
        };

        for region in REGION_ORDER {
            if region.cut_enabled(state_config) {
                for g in global.state_dim_range(region) {
                    push_dim(StateDim::new(g));
                }
            }
        }

        debug_assert!(
            !(state_config.storage && state_config.inflow_lags)
                || incoming_columns.len() == global.n_state,
            "default (all-enabled) projection must reproduce the global n_state \
             ({}); got {}",
            global.n_state,
            incoming_columns.len()
        );

        Self {
            incoming_columns,
            outgoing_columns,
            render_coeff_indices,
            render_columns,
        }
    }

    /// Count of enabled cut-state dimensions:
    /// `(storage ? N : 0) + (inflow_lags ? N*L : 0) + B + A*k_max`.
    #[inline]
    #[must_use]
    pub fn n_slots(&self) -> usize {
        self.incoming_columns.len()
    }

    /// Map a cut slot `s ∈ [0, n_slots())` to its LP incoming-state column,
    /// indexing the enabled subset in storage → lag → buckets → anticipated
    /// order.
    ///
    /// Neither the global [`StateDim`] nor the outgoing-column role [`OutCol`]
    /// can substitute for the [`CutSlot`] `s`: state-dimension and cut-slot-space
    /// diverge under a non-identity projection, and outgoing is this accessor's
    /// return role, not its parameter.
    ///
    /// ```compile_fail
    /// use cobre_core::temporal::StageStateConfig;
    /// use cobre_sddp::indexer::{CutStateProjection, StateDim, StateSpace};
    ///
    /// let global = StateSpace::new(1, 0, 0, Vec::new(), 0, 0, Vec::new(), &[0]);
    /// let cut = CutStateProjection::new(
    ///     &global,
    ///     StageStateConfig { storage: true, inflow_lags: true },
    /// );
    /// let _ = cut.incoming_column(StateDim::new(0)); // StateDim substituted for CutSlot
    /// ```
    ///
    /// ```compile_fail
    /// use cobre_core::temporal::StageStateConfig;
    /// use cobre_sddp::indexer::{CutStateProjection, OutCol, StateSpace};
    ///
    /// let global = StateSpace::new(1, 0, 0, Vec::new(), 0, 0, Vec::new(), &[0]);
    /// let cut = CutStateProjection::new(
    ///     &global,
    ///     StageStateConfig { storage: true, inflow_lags: true },
    /// );
    /// let _ = cut.incoming_column(OutCol::new(0)); // OutCol substituted for CutSlot
    /// ```
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `s >= n_slots()`.
    #[inline]
    #[must_use]
    pub fn incoming_column(&self, s: CutSlot) -> InCol {
        let j = s.get();
        debug_assert!(
            j < self.n_slots(),
            "cut slot {j} out of bounds (n_slots = {})",
            self.n_slots()
        );
        self.incoming_columns[j]
    }

    /// Map a cut slot `s ∈ [0, n_slots())` to its LP **outgoing**-state
    /// column — the column its coefficient is dotted against in DCS scoring,
    /// indexing the enabled subset in storage → lag → buckets → anticipated
    /// order.
    ///
    /// The per-pool analogue of [`StateSpace::lp_column_for_state`], spanning the
    /// full enabled range (padding included); see the struct-level "Incoming vs
    /// outgoing vs render" section for the contrast with [`Self::render_pairs`].
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `s >= n_slots()`.
    #[inline]
    #[must_use]
    pub fn outgoing_column(&self, s: CutSlot) -> OutCol {
        let j = s.get();
        debug_assert!(
            j < self.n_slots(),
            "cut slot {j} out of bounds (n_slots = {})",
            self.n_slots()
        );
        self.outgoing_columns[j]
    }

    /// Iterate the cut-row render pairs `(cut_slot, outgoing_lp_column)` for this
    /// pool, in nonzero-subset order (storage → lag → buckets → anticipated,
    /// padding slots dropped).
    ///
    /// `cut_slot` indexes a stored cut's `coefficients` slice (length
    /// [`Self::n_slots`]); `outgoing_lp_column` is where the cut-row builder places
    /// the negated, scaled coefficient. For an all-enabled study this reproduces
    /// the global `nonzero_state_indices` render — same reduced index, same column.
    #[inline]
    #[must_use]
    pub fn render_pairs(&self) -> impl ExactSizeIterator<Item = (CutSlot, OutCol)> + '_ {
        self.render_coeff_indices
            .iter()
            .copied()
            .zip(self.render_columns.iter().copied())
    }

    /// Count of cut-row render entries (nonzero-subset, padding dropped) — the
    /// per-cut non-zero state-coefficient count a builder emits before the theta
    /// entry.
    #[inline]
    #[must_use]
    pub fn render_len(&self) -> usize {
        self.render_coeff_indices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{CutSlot, CutStateProjection, InCol, OutCol, StageStateConfig, StateDim};
    use crate::indexer::StateSpace;

    fn finalized(
        hydro_count: usize,
        max_par_order: usize,
        n_anticipated: usize,
        k_max: usize,
        anticipated_lead_stages: Vec<usize>,
    ) -> StateSpace {
        let lag_counts = vec![max_par_order; hydro_count];
        StateSpace::new(
            hydro_count,
            max_par_order,
            0,
            Vec::new(),
            n_anticipated,
            k_max,
            anticipated_lead_stages,
            &lag_counts,
        )
    }

    /// Like [`finalized`] but with a declared bucket block.
    fn finalized_with_transit_buckets(
        hydro_count: usize,
        max_par_order: usize,
        n_buckets: usize,
        transit_bucket_column_order: Vec<(usize, usize)>,
        n_anticipated: usize,
        k_max: usize,
        anticipated_lead_stages: Vec<usize>,
    ) -> StateSpace {
        let lag_counts = vec![max_par_order; hydro_count];
        StateSpace::new(
            hydro_count,
            max_par_order,
            n_buckets,
            transit_bucket_column_order,
            n_anticipated,
            k_max,
            anticipated_lead_stages,
            &lag_counts,
        )
    }

    const ALL_ENABLED: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    const STORAGE_ONLY: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: false,
    };

    #[test]
    fn default_projection_is_identity() {
        let global = finalized(3, 2, 0, 0, vec![]);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(global.n_state, 9);
        assert_eq!(cut.n_slots(), global.n_state);
        for j in 0..global.n_state {
            assert_eq!(
                cut.incoming_column(CutSlot::new(j)),
                global.state_to_lp_incoming_column(StateDim::new(j)),
                "default projection must match the global resolver at j={j}"
            );
        }
    }

    #[test]
    fn storage_only_projection() {
        let global = finalized(3, 2, 0, 0, vec![]);
        let cut = CutStateProjection::new(&global, STORAGE_ONLY);

        assert_eq!(cut.n_slots(), 3);
        for j in 0..3 {
            assert_eq!(
                cut.incoming_column(CutSlot::new(j)),
                InCol::new(global.storage_in.start + j),
                "storage-only slot {j} must map to storage_in.start + {j}"
            );
        }
    }

    /// `inflow_lags` disabled drops the lag dims but keeps anticipated:
    /// `n_slots() = N + A*k_max = 2 + 2 = 4`.
    #[test]
    fn storage_only_with_anticipated_includes_anticipated() {
        let global = finalized(2, 1, 1, 2, vec![2]);
        let cut = CutStateProjection::new(&global, STORAGE_ONLY);

        assert_eq!(cut.n_slots(), 4);
        for j in 0..2 {
            assert_eq!(
                cut.incoming_column(CutSlot::new(j)),
                InCol::new(global.storage_in.start + j)
            );
        }
        for i in 0..2 {
            assert_eq!(
                cut.incoming_column(CutSlot::new(2 + i)),
                InCol::new(global.anticipated_state.start + i),
                "anticipated slot {i} must map to anticipated_state.start + {i}"
            );
        }
    }

    /// `storage: false`: cut state begins at the first enabled dimension, the
    /// inflow lags (`N*L = 2` slots, no storage or anticipated).
    #[test]
    fn storage_disabled_begins_at_first_enabled_dimension() {
        let global = finalized(2, 1, 0, 0, vec![]);
        let cut = CutStateProjection::new(
            &global,
            StageStateConfig {
                storage: false,
                inflow_lags: true,
            },
        );

        assert_eq!(cut.n_slots(), global.hydro_count * global.max_par_order);
        for i in 0..cut.n_slots() {
            assert_eq!(
                cut.incoming_column(CutSlot::new(i)),
                InCol::new(global.inflow_lags.start + i),
                "first enabled dimension is the inflow lags: slot {i}"
            );
        }
    }

    // ── Outgoing projection / render-pairs tests ──────────────────────────────

    #[test]
    fn default_render_matches_global_nonzero_mask() {
        let global = finalized(3, 2, 0, 0, vec![]);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(cut.n_slots(), global.n_state);
        for j in 0..global.n_state {
            assert_eq!(
                cut.outgoing_column(CutSlot::new(j)),
                global.lp_column_for_state(StateDim::new(j)),
                "default outgoing projection must match the global resolver at j={j}"
            );
        }

        let rendered: Vec<(CutSlot, OutCol)> = cut.render_pairs().collect();
        let global_render: Vec<(CutSlot, OutCol)> = global
            .nonzero_state_indices
            .iter()
            .map(|&g| (CutSlot::new(g.get()), global.lp_column_for_state(g)))
            .collect();
        assert_eq!(
            rendered, global_render,
            "render_pairs must reproduce the global nonzero_state_indices render"
        );
        assert_eq!(cut.render_len(), global.nonzero_state_indices.len());
    }

    #[test]
    fn storage_only_render_touches_only_storage_columns() {
        let global = finalized(3, 2, 0, 0, vec![]);
        let cut = CutStateProjection::new(&global, STORAGE_ONLY);

        assert_eq!(cut.n_slots(), 3);
        assert_eq!(cut.render_len(), 3);

        let rendered: Vec<(CutSlot, OutCol)> = cut.render_pairs().collect();
        // Storage is identity under the outgoing resolver: state_to_lp_column(j)=j.
        assert_eq!(
            rendered,
            vec![
                (CutSlot::new(0), OutCol::new(0)),
                (CutSlot::new(1), OutCol::new(1)),
                (CutSlot::new(2), OutCol::new(2))
            ]
        );

        // No rendered column lands in the lag block [N, N*(1+L)) = [3, 9).
        for (_, col) in &rendered {
            assert!(
                !global.inflow_lags.contains(&col.get()),
                "storage-only render must touch no lag column (got {})",
                col.get()
            );
        }
    }

    /// AR-padding slots are dropped from the render but kept in the full enabled
    /// range. Per-hydro lag counts `[1, 3]` give hydro 0 padding at lags 1,2, so
    /// the render (nonzero subset) is shorter than `n_slots`.
    #[test]
    fn render_drops_ar_padding_slots() {
        let lag_counts = [1usize, 3];
        let global = StateSpace::new(2, 3, 0, Vec::new(), 0, 0, vec![], &lag_counts);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(cut.n_slots(), global.n_state);
        assert_eq!(cut.render_len(), global.nonzero_state_indices.len());
        assert!(
            cut.render_len() < cut.n_slots(),
            "padding slots must be dropped, so render_len < n_slots"
        );

        let rendered: Vec<(CutSlot, OutCol)> = cut.render_pairs().collect();
        let global_render: Vec<(CutSlot, OutCol)> = global
            .nonzero_state_indices
            .iter()
            .map(|&g| (CutSlot::new(g.get()), global.lp_column_for_state(g)))
            .collect();
        assert_eq!(rendered, global_render);
    }

    // ── Bucket block tests ────────────────────────────────────────────────────

    /// `inflow_lags` disabled, but the bucket block stays included (never gated
    /// by `StageStateConfig`): `n_slots() = N + B + A*k_max = 2 + 2 + 2 = 6`, with
    /// bucket slots between storage and anticipated.
    #[test]
    fn bucket_block_always_included_with_storage_only() {
        let global = finalized_with_transit_buckets(2, 1, 2, vec![(0, 1), (1, 1)], 1, 2, vec![2]);
        let cut = CutStateProjection::new(&global, STORAGE_ONLY);

        assert_eq!(cut.n_slots(), 6);

        for j in 0..2 {
            assert_eq!(
                cut.incoming_column(CutSlot::new(j)),
                InCol::new(global.storage_in.start + j),
                "storage slot {j} must map to storage_in.start + {j}"
            );
        }
        for i in 0..2 {
            assert_eq!(
                cut.incoming_column(CutSlot::new(2 + i)),
                InCol::new(global.transit_buckets_in.start + i),
                "bucket slot {i} must map to transit_buckets_in.start + {i} despite \
                 inflow_lags disabled"
            );
        }
        for i in 0..2 {
            assert_eq!(
                cut.incoming_column(CutSlot::new(4 + i)),
                InCol::new(global.anticipated_state.start + i),
                "anticipated slot {i} must map to anticipated_state.start + {i}"
            );
        }
    }

    /// The ordering regression guard for the storage→lag→buckets→anticipated walk:
    /// swapping buckets and anticipated would misassign reduced coefficient
    /// indices even though every `(index, column)` pair still resolves to a valid
    /// column.
    #[test]
    fn bucket_render_pairs_sit_between_lag_and_anticipated() {
        let global = finalized_with_transit_buckets(2, 1, 2, vec![(0, 1), (1, 1)], 1, 2, vec![2]);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(global.transit_buckets_out, 4..6);
        assert_eq!(cut.n_slots(), global.n_state);

        let rendered: Vec<(CutSlot, OutCol)> = cut.render_pairs().collect();
        let global_render: Vec<(CutSlot, OutCol)> = global
            .nonzero_state_indices
            .iter()
            .map(|&g| (CutSlot::new(g.get()), global.lp_column_for_state(g)))
            .collect();
        assert_eq!(
            rendered, global_render,
            "render_pairs must match the global walk order exactly, \
             storage→lag→buckets→anticipated"
        );

        let transit_bucket_positions: Vec<usize> = rendered
            .iter()
            .enumerate()
            .filter(|&(_, &(_, col))| global.transit_buckets_out.contains(&col.get()))
            .map(|(pos, _)| pos)
            .collect();
        assert_eq!(
            transit_bucket_positions,
            vec![4, 5],
            "bucket render pairs must land after the lag block (positions 0..4) \
             and before the anticipated block (positions 6..8)"
        );
    }

    /// `B == 0` regression guard for the always-included bucket block: the
    /// projection reproduces `[0, StateSpace::n_state)` and its render
    /// byte-identically to the pre-bucket walk.
    #[test]
    fn b_zero_projection_matches_pre_transit_bucket_walk() {
        let global = finalized_with_transit_buckets(3, 2, 0, vec![], 2, 2, vec![1, 2]);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(global.n_buckets, 0);
        assert_eq!(cut.n_slots(), global.n_state);
        for j in 0..global.n_state {
            assert_eq!(
                cut.incoming_column(CutSlot::new(j)),
                global.state_to_lp_incoming_column(StateDim::new(j)),
                "B==0 default projection must match the global resolver at j={j}"
            );
        }

        let rendered: Vec<(CutSlot, OutCol)> = cut.render_pairs().collect();
        let global_render: Vec<(CutSlot, OutCol)> = global
            .nonzero_state_indices
            .iter()
            .map(|&g| (CutSlot::new(g.get()), global.lp_column_for_state(g)))
            .collect();
        assert_eq!(
            rendered, global_render,
            "B==0 render must reproduce the global nonzero_state_indices render"
        );
    }
}

#[cfg(test)]
mod proptests {
    use proptest::prelude::*;
    use proptest::test_runner::RngSeed;

    use super::{CutSlot, CutStateProjection, OutCol, StageStateConfig, StateDim};
    use crate::indexer::StateSpace;

    const ALL_ENABLED: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };

    /// Fixed cases/seed so a failing shrink is reproducible run-to-run (the
    /// determinism discipline applied to test generation itself).
    fn fixed_config() -> ProptestConfig {
        ProptestConfig {
            cases: 256,
            rng_seed: RngSeed::Fixed(42),
            ..ProptestConfig::default()
        }
    }

    /// A valid [`StateSpace`] over the small parameter space (`hydro_count`,
    /// `max_par_order`, `n_buckets`, `n_anticipated`, `k_max` each `0..=4` or
    /// `0..=3`), with every dependent-length vector sized and bounded to satisfy
    /// `StateSpace::new`'s debug-asserts:
    /// `transit_bucket_column_order.len() == n_buckets`,
    /// `anticipated_lead_stages.len() == n_anticipated` (each `<= k_max`), and
    /// `effective_lag_count.len() == hydro_count` (each `<= max_par_order`).
    fn state_layout_strategy() -> impl Strategy<Value = StateSpace> {
        (0..=4usize, 0..=3usize, 0..=3usize, 0..=3usize, 0..=3usize)
            .prop_flat_map(
                |(hydro_count, max_par_order, n_buckets, n_anticipated, k_max)| {
                    (
                        Just(hydro_count),
                        Just(max_par_order),
                        Just(n_buckets),
                        Just(n_anticipated),
                        Just(k_max),
                        prop::collection::vec((0..=4usize, 0..=3usize), n_buckets),
                        prop::collection::vec(0..=k_max, n_anticipated),
                        prop::collection::vec(0..=max_par_order, hydro_count),
                    )
                },
            )
            .prop_map(
                |(
                    hydro_count,
                    max_par_order,
                    n_buckets,
                    n_anticipated,
                    k_max,
                    transit_bucket_column_order,
                    anticipated_lead_stages,
                    effective_lag_count,
                )| {
                    StateSpace::new(
                        hydro_count,
                        max_par_order,
                        n_buckets,
                        transit_bucket_column_order,
                        n_anticipated,
                        k_max,
                        anticipated_lead_stages,
                        &effective_lag_count,
                    )
                },
            )
    }

    /// A [`StateSpace`] paired with an arbitrary [`StageStateConfig`] gate
    /// combination, for the gated-projection agreement property.
    fn state_layout_and_config_strategy() -> impl Strategy<Value = (StateSpace, StageStateConfig)> {
        (state_layout_strategy(), any::<bool>(), any::<bool>()).prop_map(
            |(global, storage, inflow_lags)| {
                (
                    global,
                    StageStateConfig {
                        storage,
                        inflow_lags,
                    },
                )
            },
        )
    }

    proptest! {
        #![proptest_config(fixed_config())]

        /// Generalizes `state_dim_ranges_partition_n_state_contiguously`: the
        /// four `state_dim_*_range` regions partition `[0, n_state)` contiguously
        /// (no gap, no overlap) for every generated shape.
        #[test]
        fn partition(global in state_layout_strategy()) {
            let storage = global.state_dim_storage_range();
            let lag = global.state_dim_lag_range();
            let bucket = global.state_dim_bucket_range();
            let anticipated = global.state_dim_anticipated_range();

            prop_assert_eq!(storage.start, 0);
            prop_assert_eq!(lag.start, storage.end);
            prop_assert_eq!(bucket.start, lag.end);
            prop_assert_eq!(anticipated.start, bucket.end);
            prop_assert_eq!(anticipated.end, global.n_state);
        }

        /// Generalizes `default_projection_is_identity` and
        /// `default_render_matches_global_nonzero_mask`: an all-enabled
        /// projection reproduces the global resolvers index-for-index, and
        /// `render_pairs` reproduces the global `nonzero_state_indices` render
        /// exactly, for every generated shape.
        #[test]
        fn default_identity(global in state_layout_strategy()) {
            let cut = CutStateProjection::new(&global, ALL_ENABLED);

            prop_assert_eq!(cut.n_slots(), global.n_state);
            for j in 0..global.n_state {
                prop_assert_eq!(
                    cut.incoming_column(CutSlot::new(j)),
                    global.state_to_lp_incoming_column(StateDim::new(j))
                );
                prop_assert_eq!(
                    cut.outgoing_column(CutSlot::new(j)),
                    global.lp_column_for_state(StateDim::new(j))
                );
            }

            let rendered: Vec<(CutSlot, OutCol)> = cut.render_pairs().collect();
            let global_render: Vec<(CutSlot, OutCol)> = global
                .nonzero_state_indices
                .iter()
                .map(|&g| (CutSlot::new(g.get()), global.lp_column_for_state(g)))
                .collect();
            prop_assert_eq!(rendered, global_render);
        }

        /// Generalizes `bucket_render_pairs_sit_between_lag_and_anticipated` over
        /// every gate combination. The expected mapping is built from the
        /// concrete per-region block formulas (the raw range fields, with the
        /// storage → lag → buckets → anticipated order written out literally) —
        /// never from `REGION_ORDER` or the resolvers the constructor itself
        /// walks, which would shift both sides of the assertion in lockstep
        /// and turn a walk-order or resolver bug green.
        #[test]
        fn gated_projection_agreement(
            (global, state_config) in state_layout_and_config_strategy()
        ) {
            let cut = CutStateProjection::new(&global, state_config);

            let n = global.hydro_count;
            let mut expected: Vec<(usize, usize)> = Vec::new();
            if state_config.storage {
                for h in 0..n {
                    expected.push((global.storage_in.start + h, global.storage.start + h));
                }
            }
            if state_config.inflow_lags {
                for lag in 0..global.max_par_order {
                    for h in 0..n {
                        let outgoing = if lag == 0 {
                            global.z_inflow.start + h
                        } else {
                            global.inflow_lags.start + (lag - 1) * n + h
                        };
                        expected.push((global.inflow_lags.start + lag * n + h, outgoing));
                    }
                }
            }
            for b in 0..global.n_buckets {
                expected.push((
                    global.transit_buckets_in.start + b,
                    global.transit_buckets_out.start + b,
                ));
            }
            for o in 0..global.n_anticipated * global.k_max {
                expected.push((
                    global.anticipated_state.start + o,
                    global.anticipated_slots_out.start + o,
                ));
            }

            prop_assert_eq!(expected.len(), cut.n_slots());
            for (slot, &(incoming, outgoing)) in expected.iter().enumerate() {
                prop_assert_eq!(cut.incoming_column(CutSlot::new(slot)).get(), incoming);
                prop_assert_eq!(cut.outgoing_column(CutSlot::new(slot)).get(), outgoing);
            }
        }
    }
}
