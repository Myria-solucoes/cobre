//! The stage-invariant state-vector layout and its LP-column resolvers.
//!
//! [`StateLayout`] owns the role-(a) concern of the stage LP: the
//! stage-invariant state-vector column ranges, the two layout-derived caches
//! (`nonzero_state_indices`, `state_to_lp_column_map`), and the resolvers that
//! map a state-vector index to the LP column a cut row references
//! ([`StateLayout::state_to_lp_column`]) or the incoming-state column a cut
//! subgradient is read from ([`StateLayout::state_to_lp_incoming_column`]).
//!
//! It carries the state-pinning contract (state pinning uses column bounds, not
//! equality rows): [`StateLayout::state_to_lp_incoming_column`] is the single
//! authoritative incoming-state column resolver; the LP column for both pinning
//! and dual extraction is always resolved through it, never by assuming a
//! fixing-row index. The companion [`StateLayout::state_to_lp_column`] maps the
//! outgoing state vector to the LP columns a forward-pass cut row references.
//!
//! Unlike the satellite control/equipment geometry, every offset here is a pure
//! function of `N` (`hydro_count`), `L` (`max_par_order`), `A`
//! (`n_anticipated`), and `k_max` — independent of `n_blks`/`n_thermals` — so a
//! single global stage-0 layout resolves onto the correct column at every stage
//! regardless of per-stage block counts.

use std::ops::Range;

/// Stage-invariant state-vector layout for one SDDP stage subproblem.
///
/// Computed once from the state dimensions (`N`, `L`, `A`, `k_max`) plus the
/// per-hydro effective lag-slot counts. Both layout-derived caches
/// ([`Self::nonzero_state_indices`] and [`Self::state_to_lp_column_map`]) are
/// finalized at construction by the single [`StateLayout::new`] constructor —
/// there is no two-phase init on this type.
///
/// ## Column layout
///
/// ```text
/// [0, N)                                    storage               — outgoing storage volumes (N = hydro_count)
/// [N, N*(1+L))                              inflow_lags           — AR lag variables (L lags per hydro)
/// [N*(1+L), N*(1+L) + A*k_max)              anticipated_state     — anticipated thermal commitment state slots
/// [N*(1+L) + A*k_max, … + A)                anticipated_state_out — cut-target columns (one per anticipated plant)
/// [… + A, … + A + N)                        z_inflow              — realized inflow (auxiliary, not state)
/// [… + A + N, … + A + 2*N)                  storage_in            — incoming storage volumes
/// N*(3+L) + A*k_max + A                      theta                 — future cost variable (scalar)
/// ```
///
/// `anticipated_state_out` lives in the state region (it is a cut TARGET column,
/// not a state-vector dimension) and does **not** contribute to [`Self::n_state`].
#[derive(Debug, Clone)]
pub struct StateLayout {
    /// Column range `[0, N)` for outgoing storage volumes.
    ///
    /// Each entry `storage[h]` is the column index of hydro plant `h`'s
    /// outgoing storage volume.
    pub storage: Range<usize>,

    /// Column range `[N, N*(1+L))` for AR lag variables.
    ///
    /// Lag variables are stored in lag-major order: all hydros for lag 0,
    /// then all hydros for lag 1, etc. The column index for hydro `h` at
    /// lag `l` (0-indexed, lag 0 = most recent) is:
    /// `inflow_lags.start + l * hydro_count + h`.
    pub inflow_lags: Range<usize>,

    /// Column range `[N*(1+L), N*(1+L) + n_anticipated*K_max)` for
    /// anticipated thermal commitment state slots.
    ///
    /// Ring-buffer block mirroring the inflow-lag layout: slot
    /// `k = 0..K_max` for anticipated plant `i = 0..n_anticipated` lives at
    /// column `anticipated_state.start + k * n_anticipated + i` (slot-major,
    /// plant-minor). Slot 0 holds the commitment maturing at the current
    /// stage; slot `K_max - 1` holds the commitment that matures
    /// `K_max - 1` stages from now.
    ///
    /// Empty (`0..0`) when `n_anticipated == 0`.
    pub anticipated_state: Range<usize>,

    /// Column range `[N*(2+L) + A*K_max, N*(3+L) + A*K_max)` for incoming
    /// storage volumes.
    ///
    /// Pinned to the preceding stage's outgoing `storage` solution values via
    /// `set_col_bounds` on these columns — not via equality rows (the LP has no
    /// state-fixing row range). Resolve the column with
    /// [`StateLayout::state_to_lp_incoming_column`].
    pub storage_in: Range<usize>,

    /// Column range for realized-inflow variables `z_h`, one per hydro.
    ///
    /// These free columns (lower = -inf, upper = +inf, zero cost) represent the
    /// total natural inflow `Z_t_h` at each hydro, defined by the z-inflow
    /// equality constraints. After solving, `primal[z_inflow.start + h]` gives
    /// the realized inflow for hydro h.
    ///
    /// Empty when `hydro_count == 0`.
    pub z_inflow: Range<usize>,

    /// Column range for the anticipated-thermal outgoing-state variables,
    /// one column per anticipated plant (stage-level, NOT per-block).
    ///
    /// Length: `n_anticipated`. Placed in the **stage-invariant state region**,
    /// immediately after the [`Self::anticipated_state`] ring buffer and before
    /// [`Self::z_inflow`], so that
    /// `anticipated_state_out.start == anticipated_state.end` and its offset is
    /// a pure function of `N`, `L`, `A`, `k_max` — independent of `n_blks` and
    /// `n_thermals`. This is what makes it a sound cut TARGET: the global
    /// stage-0 cut map resolves the matured anticipated slot here (via
    /// [`Self::state_to_lp_column`]'s Equal branch) onto the same column at
    /// every stage regardless of per-stage block counts.
    ///
    /// Despite living in the state region, it is NOT a state-vector dimension
    /// and does NOT contribute to [`Self::n_state`]. Together with the
    /// `anticipated_state_out` definition row it is pinned to the corresponding
    /// anticipated-decision column by an equality constraint
    /// (`anticipated_state_out[p] − decision_col[p] = 0`).
    ///
    /// Empty (`0..0`) when `n_anticipated == 0`.
    pub anticipated_state_out: Range<usize>,

    /// Column index `N*(3+L) + A*K_max + A` for the future cost variable (theta).
    ///
    /// Scalar: there is exactly one theta variable per stage LP.
    pub theta: usize,

    /// State-vector dimension.
    ///
    /// Without anticipated thermals: `N*(1+L)`.
    /// With `A` anticipated thermals at `K_max` lead stages each:
    /// `N*(1+L) + A*K_max`.
    ///
    /// The state vector consists of the `N` outgoing storage volumes followed
    /// by the `N*L` lag variables (and `A*K_max` anticipated-state slots when
    /// anticipated thermals are present).
    ///
    /// ## Semantic distinction
    ///
    /// `n_state` is the state-vector **dimension** used by cut storage and
    /// broadcast payloads. It is **not** a valid LP row index. Do not slice
    /// the LP row buffer as `[0, n_state)` — no state-fixing rows exist.
    /// Use [`StateLayout::state_to_lp_incoming_column`] to resolve the
    /// column index for state-pinning and cut-subgradient extraction.
    pub n_state: usize,

    /// Number of operating hydro plants (N).
    pub hydro_count: usize,

    /// Maximum PAR order across all operating hydros (L).
    ///
    /// All hydros use a uniform lag stride of `max_par_order`, enabling
    /// contiguous memory access and SIMD vectorisation over the lag dimension.
    pub max_par_order: usize,

    /// Number of anticipated thermals (plants with
    /// `anticipated_config.is_some()`).
    ///
    /// Zero when no anticipated plants exist.
    pub n_anticipated: usize,

    /// Maximum `lead_stages` across the anticipated thermals (`K_max`).
    ///
    /// Zero when `n_anticipated == 0`.
    pub k_max: usize,

    /// Per-plant `lead_stages` (`K_i`) for the anticipated thermals.
    ///
    /// Length [`Self::n_anticipated`]; indexed by anticipated-local position
    /// (0-indexed within the anticipated subset). Empty when
    /// `n_anticipated == 0`.
    pub anticipated_lead_stages: Vec<usize>,

    /// Indices of state dimensions whose cut coefficients can be nonzero.
    ///
    /// Storage indices `[0, N)` are always included. Lag indices `[N, N*(1+L))`
    /// are included only when `lag < effective_lag_count[hydro]`. Hydros with AR
    /// order < `max_par_order` have padded lag slots whose duals are
    /// structurally zero.
    pub nonzero_state_indices: Vec<usize>,

    /// Precomputed `state_to_lp_column(j)` for every `j ∈ [0, n_state)`.
    ///
    /// Built once at construction; read on the forward-pass cut-row hot path.
    pub state_to_lp_column_map: Vec<usize>,
}

impl StateLayout {
    /// Construct a finalized [`StateLayout`] from the state dimensions and the
    /// per-hydro effective lag-slot counts.
    ///
    /// `effective_lag_count` must have length `hydro_count`; each entry is
    /// `PrecomputedPar::effective_lag_count(h)` — the number of lag-state slots
    /// that may carry non-zero cut coefficients for that hydro. The same input
    /// `build_wired_indexer` uses today.
    ///
    /// Both layout-derived caches are finalized at construction in the
    /// production order: [`Self::set_nonzero_mask`] then
    /// [`Self::finalize_state_column_map`]. There is no two-phase init on this
    /// type — the returned value is ready for the cut-row hot path.
    ///
    /// # Panics (debug builds only)
    ///
    /// Inherits the [`Self::set_nonzero_mask`] and
    /// [`Self::finalize_state_column_map`] debug assertions:
    /// `effective_lag_count.len() == hydro_count`,
    /// `anticipated_lead_stages.len() == n_anticipated`, lag/lead bounds, and
    /// `state_to_lp_column_map.len() == n_state`.
    #[must_use]
    pub fn new(
        hydro_count: usize,
        max_par_order: usize,
        n_anticipated: usize,
        k_max: usize,
        anticipated_lead_stages: Vec<usize>,
        effective_lag_count: &[usize],
    ) -> Self {
        let n = hydro_count;
        let l = max_par_order;
        let n_ant_state = n_anticipated * k_max;

        // Sequential-offset chain: each range starts at the previous range's
        // `.end`, so the whole state region is auditable in one linear read.
        // Optional blocks empty-normalise to `0..0`; use the `*_end` bindings
        // (not `range.end`) for downstream arithmetic so the shift survives the
        // `0..0` collapse.
        let storage = 0..n;
        let inflow_lags = n..n * (1 + l);

        let anticipated_state_start = n * (1 + l);
        let anticipated_state_end = anticipated_state_start + n_ant_state;
        let anticipated_state = if n_ant_state > 0 {
            anticipated_state_start..anticipated_state_end
        } else {
            0..0
        };

        let anticipated_state_out_start = anticipated_state_end;
        let anticipated_state_out_end = anticipated_state_out_start + n_anticipated;
        let anticipated_state_out = if n_anticipated > 0 {
            anticipated_state_out_start..anticipated_state_out_end
        } else {
            0..0
        };

        let z_inflow_start = anticipated_state_out_end;
        let z_inflow = z_inflow_start..z_inflow_start + n;
        let storage_in_start = z_inflow.end;
        let storage_in = storage_in_start..storage_in_start + n;
        let theta = storage_in.end;

        // `anticipated_state_out` is a cut TARGET column, not a state-vector
        // dimension, so it does NOT enter `n_state`. Adding it would corrupt
        // cut-pool storage sizes.
        let n_state = n * (1 + l) + n_ant_state;

        let mut layout = Self {
            storage,
            inflow_lags,
            anticipated_state,
            storage_in,
            z_inflow,
            anticipated_state_out,
            theta,
            n_state,
            hydro_count,
            max_par_order,
            n_anticipated,
            k_max,
            anticipated_lead_stages,
            nonzero_state_indices: Vec::new(),
            state_to_lp_column_map: Vec::new(),
        };

        // Finalize both layout-derived caches at construction, in the order
        // production study setup uses: mask first, then the column-map cache.
        let anticipated_k = layout.anticipated_lead_stages.clone();
        layout.set_nonzero_mask(effective_lag_count, &anticipated_k);
        layout.finalize_state_column_map();
        layout
    }

    /// First column of the control region (`theta + 1`).
    ///
    /// The control/equipment geometry begins here; the state region occupies
    /// `[0, theta]`. Geometry reads this to anchor its equipment column ranges.
    #[inline]
    #[must_use]
    pub fn control_region_start(&self) -> usize {
        self.theta + 1
    }

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

    /// Read the precomputed `state_to_lp_column(j)` from the always-finalized
    /// [`state_to_lp_column_map`](Self::state_to_lp_column_map).
    ///
    /// [`StateLayout::new`] finalizes the map for every state index in its
    /// constructor (`state_to_lp_column_map.len() == n_state`), so there is no
    /// un-finalized layout and no live-resolver fallback: the indexed read is
    /// always in range for `j ∈ [0, n_state)`.
    #[inline]
    #[must_use]
    pub fn lp_column_for_state(&self, j: usize) -> usize {
        debug_assert_eq!(
            self.state_to_lp_column_map.len(),
            self.n_state,
            "state_to_lp_column_map must be finalized to n_state length"
        );
        self.state_to_lp_column_map[j]
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
    /// `anticipated_state.start = X` (for the corresponding layout
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
}

#[cfg(test)]
mod tests {
    use super::StateLayout;

    /// Build a [`StateLayout`] finalized the way production `build_wired_indexer`
    /// does: full `max_par_order` lag stride for every hydro (the coverage the
    /// dense path emits for test layouts without a PAR model) and the layout's
    /// own `anticipated_lead_stages`.
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
            n_anticipated,
            k_max,
            anticipated_lead_stages,
            &lag_counts,
        )
    }

    // ── state_to_lp_column precompute tests ─────────────────────────────────

    /// A finalized layout carrying storage + AR lags + anticipated thermals
    /// (every `state_to_lp_column` branch) must have
    /// `lp_column_for_state(j) == state_to_lp_column(j)` for every state index.
    #[test]
    fn lp_column_map_matches_resolver_with_lags_and_anticipated() {
        // hydro_count=3, max_par_order=2, n_anticipated=2 (K = [1, 2], k_max=2).
        let idx = finalized(3, 2, 2, 2, vec![1, 2]);

        assert_eq!(idx.state_to_lp_column_map.len(), idx.n_state);
        for j in 0..idx.n_state {
            assert_eq!(
                idx.lp_column_for_state(j),
                idx.state_to_lp_column(j),
                "finalized map must match the resolver at j={j}"
            );
        }
    }

    /// `StateLayout::new` always finalizes `state_to_lp_column_map` to `n_state`
    /// length, so `lp_column_for_state` reads the precomputed map directly with
    /// no live-resolver fallback. Cover every distinct layout shape (storage-only,
    /// storage + lags, and storage + lags + anticipated) to pin the
    /// always-finalized invariant the fallback removal relies on.
    #[test]
    fn lp_column_for_state_map_always_finalized() {
        for idx in [
            finalized(0, 0, 0, 0, vec![]),     // pure-thermal: n_state == 0
            finalized(3, 0, 0, 0, vec![]),     // storage-only
            finalized(2, 3, 0, 0, vec![]),     // storage + lags
            finalized(3, 2, 2, 2, vec![1, 2]), // storage + lags + anticipated
        ] {
            assert_eq!(
                idx.state_to_lp_column_map.len(),
                idx.n_state,
                "constructor must finalize the column map to n_state length"
            );
            for j in 0..idx.n_state {
                assert_eq!(idx.lp_column_for_state(j), idx.state_to_lp_column_map[j]);
            }
        }
    }

    /// Storage-only (`max_par_order == 0`, no anticipated): the mask is exactly
    /// `[0, n_state)` ascending and `lp_column_for_state(j) == j` — the
    /// dense→sparse bit-identity premise for the unified cut-row loop.
    #[test]
    fn lp_column_map_storage_only_mask_is_full_range() {
        let idx = finalized(3, 0, 0, 0, vec![]);

        assert_eq!(idx.nonzero_state_indices, vec![0, 1, 2]);
        assert_eq!(idx.nonzero_state_indices.len(), idx.n_state);
        for j in 0..idx.n_state {
            assert_eq!(idx.lp_column_for_state(j), j);
        }
    }

    // ── state_to_lp_column tests ──────────────────────────────────────────────

    /// `anticipated_state` indices use the shift-aware mapping when
    /// `max_par_order == 0 && n_anticipated > 0`. The `anticipated_state` branch
    /// runs before the `max_par_order == 0` lag-block guard; verify the shift
    /// semantics apply even when there are no inflow lags.
    #[test]
    fn state_to_lp_column_anticipated_identity_no_lag() {
        // N=1, L=0, n_anticipated=1, k_max=2, anticipated_lead_stages=[2].
        // n_state = 1*(1+0) + 1*2 = 3.
        // anticipated_state = [1, 3); slot 0 at j=1, slot 1 at j=2.
        // anticipated_state.start = N*(1+L) = 1.
        let idx = finalized(1, 0, 1, 2, vec![2]);
        // Storage index: identity.
        assert_eq!(idx.state_to_lp_column(0), 0);
        // Anticipated-state slot 0 (j=1): slot+1=1 < k_p=2 → shift to slot 1.
        // Returns anticipated_state.start + 1*n_anticipated + 0 = 1+1 = 2.
        assert_eq!(idx.state_to_lp_column(1), 2);
        // Anticipated-state slot 1 (j=2): slot+1=2 == k_p=2 → state-out channel.
        // Returns anticipated_state_out.start + 0.
        assert_eq!(idx.state_to_lp_column(2), idx.anticipated_state_out.start);
    }

    /// `anticipated_state` indices use the shift-aware mapping when
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
        // z_inflow.start = anticipated_state_out_end = 2 + 2 + 1 = 5.
        let idx = finalized(1, 1, 1, 2, vec![2]);
        // Storage: identity.
        assert_eq!(idx.state_to_lp_column(0), 0);
        // Lag block: remapped.
        // j=1: offset=0, h=0, lag=0 → z_inflow.start + 0.
        assert_eq!(idx.state_to_lp_column(1), idx.z_inflow.start);
        // Anticipated-state slot 0 (j=2): slot+1=1 < k_p=2 → shift to slot 1.
        // Returns anticipated_state.start + 1*n_anticipated + 0 = 2+1 = 3.
        assert_eq!(idx.state_to_lp_column(2), 3);
        // Anticipated-state slot 1 (j=3): slot+1=2 == k_p=2 → state-out channel.
        // Returns anticipated_state_out.start + 0.
        assert_eq!(idx.state_to_lp_column(3), idx.anticipated_state_out.start);
    }

    /// Lag-remap branch is preserved when `n_anticipated == 0` and
    /// `max_par_order > 0`.  The anticipated-state guard must not fire
    /// when there are no anticipated thermals.
    #[test]
    fn state_to_lp_column_lag_remap_preserved_no_anticipated() {
        // N=1, L=1, n_anticipated=0 — classic PAR(p) case.
        // n_state = 1*(1+1) = 2. Layout: j=0 storage, j=1 lag-0.
        let idx = finalized(1, 1, 0, 0, vec![]);
        assert_eq!(idx.n_anticipated, 0);
        // Storage: identity.
        assert_eq!(idx.state_to_lp_column(0), 0);
        // Lag block j=1: offset=0, h=0, lag=0 → z_inflow.start + 0.
        // For N=1, L=0 anticipated: z_inflow = N*(1+L)..N*(2+L) = 2..3.
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
        let idx = finalized(0, 0, 1, 2, vec![2]);
        // Slot K_p - 1 = 1 (the highest slot for plant 0) → anticipated_state_out column.
        let slot_k_minus_1 = idx.anticipated_state.start + (idx.k_max - 1) * idx.n_anticipated;
        assert_eq!(
            idx.state_to_lp_column(slot_k_minus_1),
            idx.anticipated_state_out.start,
        );
    }

    /// Equal-branch boundary at `k_max == 1`: with a single ring-buffer slot,
    /// slot 0 is the only slot and `slot + 1 == k_p == 1`, so the Equal branch
    /// fires immediately on slot 0 — there is no Less (shift) or Greater
    /// (padding) slot to reach first. The slot must route to the
    /// `anticipated_state_out` column (the Equal-branch target). The other
    /// anticipated tests use `k_max >= 2`, where slot 0 takes the Less branch,
    /// so this is the only coverage of the `k_max == 1` Equal path.
    #[test]
    fn state_to_lp_column_equal_branch_k_max_one() {
        // N=0, L=0, n_anticipated=1, k_max=1, anticipated_lead_stages=[1].
        // n_state = 0*(1+0) + 1*1 = 1.
        // anticipated_state = [0, 1); the lone slot 0 is at j=0.
        let idx = finalized(0, 0, 1, 1, vec![1]);
        // Slot 0 of plant 0: slot+1 = 1 == k_p = 1 → Equal branch → state-out.
        let slot_0 = idx.anticipated_state.start;
        assert_eq!(
            idx.state_to_lp_column(slot_0),
            idx.anticipated_state_out.start,
            "k_max==1 slot 0 must route to anticipated_state_out via the Equal branch",
        );
    }

    /// Shift branch: an `anticipated_state` slot `i < K_p - 1` maps to the
    /// predecessor stage's `anticipated_state` column at slot `i + 1` (the
    /// shift). Successor's slot `i` comes from predecessor's incoming slot
    /// `i + 1` after `shift_anticipated_state` runs.
    #[test]
    fn state_to_lp_column_anticipated_shift() {
        // Single plant, K=2.
        let idx = finalized(0, 0, 1, 2, vec![2]);
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
        let idx = finalized(0, 0, 2, 3, vec![1, 3]);
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
        let idx = finalized(0, 0, 2, 2, vec![2, 2]);
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

    /// The Equal branch resolves the matured anticipated slot into the
    /// **state region**: for `j = anticipated_state.start + (K_p − 1)*A + plant`
    /// it returns `anticipated_state_out.start + plant`, and that column lies
    /// in `[anticipated_state.end, theta)` — i.e. inside the relocated
    /// state-region block, not the control region.
    #[test]
    fn state_to_lp_column_equal_branch_resolves_into_state_region() {
        // N=3, L=2, A=2, k_max=3, uniform K_p = 3.
        let idx = finalized(3, 2, 2, 3, vec![3, 3]);
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

    // ── state_to_lp_incoming_column tests ────────────────────────────────────

    /// Storage range: for a layout with `N=3, L=2, A=0`,
    /// `state_to_lp_incoming_column(j)` for `j ∈ [0, N)` returns
    /// `storage_in.start + j`.
    #[test]
    fn state_to_lp_incoming_column_storage_range() {
        // N=3, L=2: storage_in.start = N*(2+L) = 3*4 = 12.
        let idx = finalized(3, 2, 0, 0, vec![]);
        assert_eq!(idx.storage_in.start, 12);
        for j in 0..3_usize {
            assert_eq!(
                idx.state_to_lp_incoming_column(j),
                idx.storage_in.start + j,
                "j={j}: expected storage_in.start + {j}"
            );
        }
    }

    /// AR lag range: for a layout with `N=3, L=2, A=0`,
    /// `state_to_lp_incoming_column(j)` for `j ∈ [N, N*(1+L))` returns
    /// `inflow_lags.start + (j − N)`.
    #[test]
    fn state_to_lp_incoming_column_lag_range() {
        // N=3, L=2: inflow_lags = 3..9.
        let idx = finalized(3, 2, 0, 0, vec![]);
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

    /// Anticipated-state range: for a layout with `N=0, L=0, A=1, K=2`,
    /// `state_to_lp_incoming_column(j)` for `j ∈ [0, n_state)` returns
    /// `anticipated_state.start + j` (since `lag_end` = N*(1+L) = 0).
    #[test]
    fn state_to_lp_incoming_column_anticipated_range() {
        // N=0, L=0, A=1, K=2: n_state = 0 + 1*2 = 2.
        // anticipated_state.start = N*(1+L) = 0.
        let idx = finalized(0, 0, 1, 2, vec![2]);
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
    fn state_to_lp_incoming_column_combined_layout() {
        // N=3, L=2, A=1, K=2:
        //   n_state = N*(1+L) + A*K = 3*3 + 1*2 = 11.
        //   storage_in.start = N*(2+L) + A*K_max + A = 3*4 + 1*2 + 1 = 15.
        //   inflow_lags.start = N = 3.
        //   anticipated_state.start = N*(1+L) = 9.
        //   lag_end = N*(1+L) = 9.
        let idx = finalized(3, 2, 1, 2, vec![2]);
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
    /// return different values (the former returns the incoming-lag column, the
    /// latter returns the `z_inflow` or lag-out column). For the storage range
    /// the results differ because `storage_in.start > 0` while
    /// `state_to_lp_column` returns `j` (outgoing, which starts at column 0).
    #[test]
    fn state_to_lp_incoming_column_differs_from_state_to_lp_column_for_lag() {
        // N=2, L=1: storage_in.start = N*(2+L) = 2*3 = 6.
        // state_to_lp_column(0) = 0 (outgoing storage).
        // state_to_lp_incoming_column(0) = storage_in.start + 0 = 6.
        let idx = finalized(2, 1, 0, 0, vec![]);
        // Storage range: incoming ≠ outgoing.
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

    // ── Nonzero state mask tests ───────────────────────────────────────────

    #[test]
    fn nonzero_mask_mixed_ar_orders() {
        // 4 hydros (N=4), max_par_order=6 (L=6), ar_orders=[0, 1, 3, 6]
        // inflow_lags.start = N = 4
        // Lag-major layout: slot = 4 + lag * N + h
        let mut idx = finalized(4, 6, 0, 0, vec![]);
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
        let mut idx = finalized(3, 0, 0, 0, vec![]);
        idx.set_nonzero_mask(&[0, 0, 0], &[]);
        assert_eq!(idx.nonzero_state_indices.len(), 3);
        assert_eq!(&idx.nonzero_state_indices, &[0, 1, 2]);
    }

    #[test]
    fn nonzero_mask_all_full_order() {
        // All hydros at max AR order: mask covers all n_state indices
        let mut idx = finalized(2, 3, 0, 0, vec![]);
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
        let mut idx = finalized(2, 12, 0, 0, vec![]);
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
        let mut idx = finalized(0, 0, 2, 3, vec![3, 3]);

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
        let mut idx = finalized(3, 2, 2, 3, vec![3, 1]);

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
        let mut idx = finalized(0, 0, 1, 2, vec![2]);

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
        let mut idx = finalized(0, 0, 3, 4, vec![4, 2, 1]);

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
        let mut idx_with = finalized(4, 6, 0, 0, vec![]);
        idx_with.set_nonzero_mask(&[0, 1, 3, 6], &[]);

        // Same expected mask as `nonzero_mask_mixed_ar_orders`:
        // [0,1,2,3] (storage) + [5,6,7,10,11,14,15,19,23,27] (lags).
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
        let mut idx = finalized(3, 2, 2, 3, vec![2, 3]);

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
        let mut idx = finalized(0, 0, 1, 3, vec![3]);

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
        let mut idx = finalized(0, 0, 2, 2, vec![2, 0]);

        idx.set_nonzero_mask(&[], &[2, 0]);

        // Plant 0 (K_0=2) emits slot=0→0, slot=1→2. Plant 1 (K_1=0) emits
        // nothing. Expected mask: [0, 2].
        assert_eq!(idx.nonzero_state_indices, vec![0, 2]);
    }
}
