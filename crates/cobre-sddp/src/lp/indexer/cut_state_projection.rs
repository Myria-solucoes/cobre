//! Storage-scoped projection of the global [`StateLayout`] for cut storage,
//! extraction (incoming), and rendering (outgoing).
//!
//! [`CutStateProjection`] exposes only the cut-state dimensions a stage's
//! [`StageStateConfig`] enables — storage and/or inflow lags — with travel-time
//! buckets and anticipated state always included (neither is governed by
//! [`StageStateConfig`]). It maps a
//! cut slot index `j ∈ [0, n_state())` to the LP incoming-state column its
//! subgradient is read from ([`Self::state_to_lp_incoming_column`]) or the
//! outgoing column its coefficient is dotted against in DCS scoring
//! ([`Self::state_to_lp_outgoing_column`]), and exposes the nonzero-subset render
//! pairs a cut row is built from ([`Self::render_pairs`]) — all in
//! storage → lag → buckets → anticipated order.
//!
//! The global [`StateLayout`] is unchanged: it still drives the LP. This carries
//! no LP-build responsibility and delegates every column to
//! [`StateLayout::state_to_lp_incoming_column`] (incoming) or
//! [`StateLayout::lp_column_for_state`] (outgoing).

use cobre_core::temporal::StageStateConfig;

use super::StateLayout;

/// A storage-scoped view of the global state vector exposing only the cut-state
/// dimensions a stage enables, plus anticipated state (always included).
///
/// This is a projection **of** a [`StateLayout`], not a sibling layout: it
/// delegates all column arithmetic to the global layout and carries only an
/// enabled-dimension mask (the [`StageStateConfig`] applied at construction) plus
/// the precomputed column vectors that mask selects. It is meaningless without the
/// [`StateLayout`] it was built from, and it never drives the LP. The default
/// (all-enabled) projection is index-identical to that global layout (see the
/// default-identity contract below).
///
/// This is the foundational projection that sizes a stage's cut pool to its
/// enabled cut-state dimensions, gates which incoming-state columns dual
/// extraction reads, and gates which outgoing columns a stored cut renders onto —
/// so a stage with `inflow_lags: false` carries no lag coefficients in its cuts
/// and its cut rows touch no lag column.
///
/// ## Default-identity contract
///
/// When `storage: true, inflow_lags: true`, [`Self::n_state`] equals the global
/// [`StateLayout::n_state`], [`Self::state_to_lp_incoming_column`] equals the
/// global incoming resolver for every `j`, and [`Self::render_pairs`] reproduces
/// the global `nonzero_state_indices` render (same reduced index `j`, same
/// outgoing column) exactly. The construction walks the global state index space
/// in storage → lag → buckets → anticipated order, so the default projection
/// reproduces `[0, StateLayout::n_state)` exactly. The forbidden alternative is
/// reordering dimensions (e.g. anticipated before buckets) so a cut coefficient
/// lands on the wrong LP column, silently corrupting every existing study.
///
/// ## Incoming vs outgoing vs render
///
/// The incoming projection ([`Self::state_to_lp_incoming_column`]) and the
/// outgoing projection ([`Self::state_to_lp_outgoing_column`]) both span the full
/// enabled range `[0, n_state())` — including AR-padding and anticipated-padding
/// slots whose coefficients are structurally zero — because dual extraction and
/// DCS dot-product scoring read every enabled dimension. The render projection
/// ([`Self::render_pairs`]) is the **nonzero subset** of that range (padding slots
/// dropped, matching the global `nonzero_state_indices`), because a cut row emits
/// no entry for a structurally-zero padding coefficient.
#[derive(Debug, Clone)]
pub struct CutStateProjection {
    /// Precomputed LP incoming-state column for every cut slot `j`, read on the
    /// extraction hot path.
    incoming_columns: Vec<usize>,

    /// Precomputed LP outgoing-state column for every cut slot `j`, read on the
    /// DCS scoring hot path; parallel to [`Self::incoming_columns`] (same `j`,
    /// same enabled dimension).
    outgoing_columns: Vec<usize>,

    /// Reduced coefficient index `j ∈ [0, n_state())` for each render pair, in
    /// nonzero-subset order. Parallel to [`Self::render_columns`].
    render_coeff_indices: Vec<usize>,

    /// LP outgoing-state column for each render pair, in nonzero-subset order.
    /// Parallel to [`Self::render_coeff_indices`].
    render_columns: Vec<usize>,
}

impl CutStateProjection {
    /// Project the global [`StateLayout`] onto the cut-state dimensions enabled by
    /// `state_config`, with travel-time buckets and anticipated state always
    /// included.
    ///
    /// Dimensions are walked in storage → lag → buckets → anticipated order —
    /// the same order [`StateLayout::set_nonzero_mask`] builds its global mask
    /// in, which is what makes the all-enabled default projection reproduce it
    /// exactly. A storage index `[0, N)` is included iff `state_config.storage`;
    /// an inflow-lag index `[N, N*(1+L))` iff `state_config.inflow_lags`; a
    /// bucket index `[N*(1+L), N*(1+L) + B)` always (mirroring anticipated:
    /// neither is governed by `state_config`); an anticipated-state index
    /// `[N*(1+L) + B, N*(1+L) + B + A*k_max)` always. The render subset keeps
    /// only indices in `global.nonzero_state_indices` (padding slots dropped),
    /// so the default projection reproduces the global `nonzero_state_indices`
    /// render exactly.
    #[must_use]
    pub fn new(global: &StateLayout, state_config: StageStateConfig) -> Self {
        let n = global.hydro_count;
        let lag_end = n * (1 + global.max_par_order);
        let transit_bucket_end = lag_end + global.n_buckets;
        let anticipated_end = transit_bucket_end + global.n_anticipated * global.k_max;

        let mut incoming_columns = Vec::new();
        let mut outgoing_columns = Vec::new();
        let mut render_coeff_indices = Vec::new();
        let mut render_columns = Vec::new();

        let push_dim = |global: &StateLayout,
                        incoming_columns: &mut Vec<usize>,
                        outgoing_columns: &mut Vec<usize>,
                        render_coeff_indices: &mut Vec<usize>,
                        render_columns: &mut Vec<usize>,
                        g: usize| {
            let reduced_j = incoming_columns.len();
            let outgoing = global.lp_column_for_state(g);
            incoming_columns.push(global.state_to_lp_incoming_column(g));
            outgoing_columns.push(outgoing);
            // Drop padding slots (not zero-fill): a dropped structurally-zero
            // coefficient keeps the default render bit-identical to the global
            // nonzero_state_indices render.
            if global.nonzero_state_indices.binary_search(&g).is_ok() {
                render_coeff_indices.push(reduced_j);
                render_columns.push(outgoing);
            }
        };

        if state_config.storage {
            for g in 0..n {
                push_dim(
                    global,
                    &mut incoming_columns,
                    &mut outgoing_columns,
                    &mut render_coeff_indices,
                    &mut render_columns,
                    g,
                );
            }
        }
        if state_config.inflow_lags {
            for g in n..lag_end {
                push_dim(
                    global,
                    &mut incoming_columns,
                    &mut outgoing_columns,
                    &mut render_coeff_indices,
                    &mut render_columns,
                    g,
                );
            }
        }
        // Travel-time buckets: always included, the anticipated pattern — a
        // per-stage `StageStateConfig` reduction here would shrink pool
        // `state_dimension` and misalign the intercept dot against the global
        // trial state.
        for g in lag_end..transit_bucket_end {
            push_dim(
                global,
                &mut incoming_columns,
                &mut outgoing_columns,
                &mut render_coeff_indices,
                &mut render_columns,
                g,
            );
        }
        for g in transit_bucket_end..anticipated_end {
            push_dim(
                global,
                &mut incoming_columns,
                &mut outgoing_columns,
                &mut render_coeff_indices,
                &mut render_columns,
                g,
            );
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
    pub fn n_state(&self) -> usize {
        self.incoming_columns.len()
    }

    /// Map a cut slot index `j ∈ [0, n_state())` to its LP incoming-state column,
    /// indexing the enabled subset in storage → lag → buckets → anticipated
    /// order.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `j >= n_state()`.
    #[inline]
    #[must_use]
    pub fn state_to_lp_incoming_column(&self, j: usize) -> usize {
        debug_assert!(
            j < self.n_state(),
            "cut state index {j} out of bounds (n_state = {})",
            self.n_state()
        );
        self.incoming_columns[j]
    }

    /// Map a cut slot index `j ∈ [0, n_state())` to its LP **outgoing**-state
    /// column — the column its coefficient is dotted against in DCS scoring,
    /// indexing the enabled subset in storage → lag → buckets → anticipated
    /// order.
    ///
    /// The per-pool analogue of [`StateLayout::lp_column_for_state`], spanning the
    /// full enabled range (padding included); see the struct-level "Incoming vs
    /// outgoing vs render" section for the contrast with [`Self::render_pairs`].
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `j >= n_state()`.
    #[inline]
    #[must_use]
    pub fn state_to_lp_outgoing_column(&self, j: usize) -> usize {
        debug_assert!(
            j < self.n_state(),
            "cut state index {j} out of bounds (n_state = {})",
            self.n_state()
        );
        self.outgoing_columns[j]
    }

    /// Iterate the cut-row render pairs `(reduced_coeff_index, outgoing_lp_column)`
    /// for this pool, in nonzero-subset order (storage → lag → buckets →
    /// anticipated, padding slots dropped).
    ///
    /// `reduced_coeff_index` indexes a stored cut's `coefficients` slice (length
    /// [`Self::n_state`]); `outgoing_lp_column` is where the cut-row builder places
    /// the negated, scaled coefficient. For an all-enabled study this reproduces
    /// the global `nonzero_state_indices` render — same reduced index, same column.
    #[inline]
    #[must_use]
    pub fn render_pairs(&self) -> impl ExactSizeIterator<Item = (usize, usize)> + '_ {
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
    use super::{CutStateProjection, StageStateConfig};
    use crate::indexer::StateLayout;

    fn finalized(
        hydro_count: usize,
        max_par_order: usize,
        n_anticipated: usize,
        k_max: usize,
        anticipated_lead_stages: Vec<usize>,
    ) -> StateLayout {
        let lag_counts = vec![max_par_order; hydro_count];
        StateLayout::new(
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

    /// Same as [`finalized`] but with a declared bucket block, mirroring
    /// `state_layout.rs`'s own bucket-aware test helper.
    fn finalized_with_transit_buckets(
        hydro_count: usize,
        max_par_order: usize,
        n_buckets: usize,
        transit_bucket_column_order: Vec<(usize, usize)>,
        n_anticipated: usize,
        k_max: usize,
        anticipated_lead_stages: Vec<usize>,
    ) -> StateLayout {
        let lag_counts = vec![max_par_order; hydro_count];
        StateLayout::new(
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

    /// Default projection (`storage + inflow_lags`) over `N=3, L=2, A=0` is the
    /// identity: `n_state()` equals the global `StateLayout::n_state` (9) and the
    /// resolver agrees with the global resolver for every `j ∈ [0, 9)`.
    #[test]
    fn default_projection_is_identity() {
        let global = finalized(3, 2, 0, 0, vec![]);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(global.n_state, 9);
        assert_eq!(cut.n_state(), global.n_state);
        for j in 0..global.n_state {
            assert_eq!(
                cut.state_to_lp_incoming_column(j),
                global.state_to_lp_incoming_column(j),
                "default projection must match the global resolver at j={j}"
            );
        }
    }

    /// Storage-only projection (`N=3, L=2, A=0`): `n_state()` is 3 and each cut
    /// slot maps to `storage_in.start + j`.
    #[test]
    fn storage_only_projection() {
        let global = finalized(3, 2, 0, 0, vec![]);
        let cut = CutStateProjection::new(&global, STORAGE_ONLY);

        assert_eq!(cut.n_state(), 3);
        for j in 0..3 {
            assert_eq!(
                cut.state_to_lp_incoming_column(j),
                global.storage_in.start + j,
                "storage-only slot {j} must map to storage_in.start + {j}"
            );
        }
    }

    /// Storage-only with anticipated (`N=2, L=1, A=1, k_max=2`): `inflow_lags`
    /// disabled drops the lag dimensions but anticipated state stays included, so
    /// `n_state()` is `2 + A*k_max = 4`. The two storage slots map to
    /// `storage_in.start + j`; the two anticipated slots map to
    /// `anticipated_state.start + i`.
    #[test]
    fn storage_only_with_anticipated_includes_anticipated() {
        let global = finalized(2, 1, 1, 2, vec![2]);
        let cut = CutStateProjection::new(&global, STORAGE_ONLY);

        assert_eq!(cut.n_state(), 4);
        for j in 0..2 {
            assert_eq!(
                cut.state_to_lp_incoming_column(j),
                global.storage_in.start + j
            );
        }
        for i in 0..2 {
            assert_eq!(
                cut.state_to_lp_incoming_column(2 + i),
                global.anticipated_state.start + i,
                "anticipated slot {i} must map to anticipated_state.start + {i}"
            );
        }
    }

    /// `storage: false` edge case (`N=2, L=1, A=0`, `inflow_lags: true`): cut
    /// state begins at the first enabled dimension (the inflow lags). With
    /// `N*L = 2` lag slots and no storage or anticipated, `n_state()` is 2 and
    /// each slot maps to `inflow_lags.start + i`.
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

        assert_eq!(cut.n_state(), global.hydro_count * global.max_par_order);
        for i in 0..cut.n_state() {
            assert_eq!(
                cut.state_to_lp_incoming_column(i),
                global.inflow_lags.start + i,
                "first enabled dimension is the inflow lags: slot {i}"
            );
        }
    }

    // ── Outgoing projection / render-pairs tests ──────────────────────────────

    /// Default projection (`storage + inflow_lags`) over `N=3, L=2, A=0`: the
    /// outgoing resolver agrees with the global `lp_column_for_state` for every
    /// `j`, and `render_pairs` reproduces the global `nonzero_state_indices` render
    /// exactly (reduced index == global index, column == global outgoing column).
    #[test]
    fn default_render_matches_global_nonzero_mask() {
        let global = finalized(3, 2, 0, 0, vec![]);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(cut.n_state(), global.n_state);
        for j in 0..global.n_state {
            assert_eq!(
                cut.state_to_lp_outgoing_column(j),
                global.lp_column_for_state(j),
                "default outgoing projection must match the global resolver at j={j}"
            );
        }

        let rendered: Vec<(usize, usize)> = cut.render_pairs().collect();
        let global_render: Vec<(usize, usize)> = global
            .nonzero_state_indices
            .iter()
            .map(|&g| (g, global.lp_column_for_state(g)))
            .collect();
        assert_eq!(
            rendered, global_render,
            "render_pairs must reproduce the global nonzero_state_indices render"
        );
        assert_eq!(cut.render_len(), global.nonzero_state_indices.len());
    }

    /// Storage-only projection (`N=3, L=2, A=0`): the cut row renders ONLY the
    /// three storage columns. `render_len()` is 3, each render pair maps reduced
    /// index `j` to storage outgoing column `j` (identity for storage), and no lag
    /// column appears.
    #[test]
    fn storage_only_render_touches_only_storage_columns() {
        let global = finalized(3, 2, 0, 0, vec![]);
        let cut = CutStateProjection::new(&global, STORAGE_ONLY);

        assert_eq!(cut.n_state(), 3);
        assert_eq!(cut.render_len(), 3);

        let rendered: Vec<(usize, usize)> = cut.render_pairs().collect();
        // Storage is identity under the outgoing resolver: state_to_lp_column(j)=j.
        assert_eq!(rendered, vec![(0, 0), (1, 1), (2, 2)]);

        // No rendered column lands in the lag block [N, N*(1+L)) = [3, 9).
        for (_, col) in &rendered {
            assert!(
                !global.inflow_lags.contains(col),
                "storage-only render must touch no lag column (got {col})"
            );
        }
    }

    /// AR-padding slots are dropped from the render but kept in the full enabled
    /// range. With `N=2, L=3` and per-hydro effective lag counts `[1, 3]`, hydro 0
    /// has padding at lags 1,2. The full enabled range is `N*(1+L)=8`, but the
    /// render keeps only the global nonzero subset (storage + non-padding lags).
    #[test]
    fn render_drops_ar_padding_slots() {
        let lag_counts = [1usize, 3];
        let global = StateLayout::new(2, 3, 0, Vec::new(), 0, 0, vec![], &lag_counts);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        // Full enabled range spans every dimension (incl. padding).
        assert_eq!(cut.n_state(), global.n_state);
        // Render keeps only the nonzero subset.
        assert_eq!(cut.render_len(), global.nonzero_state_indices.len());
        assert!(
            cut.render_len() < cut.n_state(),
            "padding slots must be dropped, so render_len < n_state"
        );

        let rendered: Vec<(usize, usize)> = cut.render_pairs().collect();
        let global_render: Vec<(usize, usize)> = global
            .nonzero_state_indices
            .iter()
            .map(|&g| (g, global.lp_column_for_state(g)))
            .collect();
        assert_eq!(rendered, global_render);
    }

    // ── Bucket block tests ────────────────────────────────────────────────────

    /// Storage-only projection with buckets present (`N=2, L=1, B=2, A=1,
    /// k_max=2`): `inflow_lags` disabled drops the lag dimensions, but the
    /// bucket block stays included — the anticipated always-included pattern,
    /// never gated by `StageStateConfig`. `n_state()` is `N + B + A*k_max = 6`;
    /// the two bucket slots sit between the storage slots and the anticipated
    /// slots.
    #[test]
    fn bucket_block_always_included_with_storage_only() {
        let global = finalized_with_transit_buckets(2, 1, 2, vec![(0, 1), (1, 1)], 1, 2, vec![2]);
        let cut = CutStateProjection::new(&global, STORAGE_ONLY);

        assert_eq!(cut.n_state(), 6);

        for j in 0..2 {
            assert_eq!(
                cut.state_to_lp_incoming_column(j),
                global.storage_in.start + j,
                "storage slot {j} must map to storage_in.start + {j}"
            );
        }
        for i in 0..2 {
            assert_eq!(
                cut.state_to_lp_incoming_column(2 + i),
                global.transit_buckets_in.start + i,
                "bucket slot {i} must map to transit_buckets_in.start + {i} despite \
                 inflow_lags disabled"
            );
        }
        for i in 0..2 {
            assert_eq!(
                cut.state_to_lp_incoming_column(4 + i),
                global.anticipated_state.start + i,
                "anticipated slot {i} must map to anticipated_state.start + {i}"
            );
        }
    }

    /// Bucket render pairs sit strictly between the lag block and the
    /// anticipated block (`N=2, L=1, B=2, A=1, k_max=2`, all-enabled, no
    /// padding): the render reproduces the global `nonzero_state_indices` walk
    /// exactly — the ordering regression guard for "walk order MUST be
    /// storage→lag→buckets→anticipated" (swapping buckets and anticipated
    /// would misassign reduced coefficient indices even though every
    /// `(index, column)` pair individually still resolves to a valid column).
    #[test]
    fn bucket_render_pairs_sit_between_lag_and_anticipated() {
        let global = finalized_with_transit_buckets(2, 1, 2, vec![(0, 1), (1, 1)], 1, 2, vec![2]);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(global.transit_buckets_out, 4..6);
        assert_eq!(cut.n_state(), global.n_state);

        let rendered: Vec<(usize, usize)> = cut.render_pairs().collect();
        let global_render: Vec<(usize, usize)> = global
            .nonzero_state_indices
            .iter()
            .map(|&g| (g, global.lp_column_for_state(g)))
            .collect();
        assert_eq!(
            rendered, global_render,
            "render_pairs must match the global walk order exactly, \
             storage→lag→buckets→anticipated"
        );

        let transit_bucket_positions: Vec<usize> = rendered
            .iter()
            .enumerate()
            .filter(|&(_, &(_, col))| global.transit_buckets_out.contains(&col))
            .map(|(pos, _)| pos)
            .collect();
        assert_eq!(
            transit_bucket_positions,
            vec![4, 5],
            "bucket render pairs must land after the lag block (positions 0..4) \
             and before the anticipated block (positions 6..8)"
        );
    }

    /// `B == 0`, reached via the bucket-aware constructor: the projection
    /// reproduces `[0, StateLayout::n_state)` and its render exactly,
    /// byte-identical to the pre-bucket walk — the required regression guard
    /// for the always-included bucket block inserted between lag and
    /// anticipated.
    #[test]
    fn b_zero_projection_matches_pre_transit_bucket_walk() {
        let global = finalized_with_transit_buckets(3, 2, 0, vec![], 2, 2, vec![1, 2]);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(global.n_buckets, 0);
        assert_eq!(cut.n_state(), global.n_state);
        for j in 0..global.n_state {
            assert_eq!(
                cut.state_to_lp_incoming_column(j),
                global.state_to_lp_incoming_column(j),
                "B==0 default projection must match the global resolver at j={j}"
            );
        }

        let rendered: Vec<(usize, usize)> = cut.render_pairs().collect();
        let global_render: Vec<(usize, usize)> = global
            .nonzero_state_indices
            .iter()
            .map(|&g| (g, global.lp_column_for_state(g)))
            .collect();
        assert_eq!(
            rendered, global_render,
            "B==0 render must reproduce the global nonzero_state_indices render"
        );
    }
}
