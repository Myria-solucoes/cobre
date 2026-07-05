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
//! function of `N` (`hydro_count`), `L` (`max_par_order`), `B` (`n_buckets`), `A`
//! (`n_anticipated`), and `k_max` — independent of `n_blks`/`n_thermals` — so a
//! single global stage-0 layout resolves onto the correct column at every stage
//! regardless of per-stage block counts.

use std::ops::Range;

use crate::lead_time::AnticipatedResolution;

/// Stage-invariant state-vector layout for one SDDP stage subproblem.
///
/// Computed once from the state dimensions (`N`, `L`, `B`, `A`, `k_max`) plus
/// the per-hydro effective lag-slot counts. Both layout-derived caches
/// ([`Self::nonzero_state_indices`] and [`Self::state_to_lp_column_map`]) are
/// finalized at construction by the single [`StateLayout::new`] constructor —
/// there is no two-phase init on this type.
///
/// ## Column layout
///
/// ```text
/// [0, N)                                     storage               — outgoing storage volumes (N = hydro_count)
/// [N, N*(1+L))                               inflow_lags           — AR lag variables (L lags per hydro)
/// [N*(1+L), N*(1+L) + B)                     transit_buckets_out   — travel-time bucket state (outgoing, identity)
/// [N*(1+L) + B, N*(1+L) + B + A*k_max)       anticipated_slots_out — anticipated-ring outgoing slots (outgoing, identity)
/// [… + A*k_max, … + N)                       z_inflow              — realized inflow (auxiliary, not state)
/// [… + N, … + 2*N)                           storage_in            — incoming storage volumes
/// [… + 2*N, … + 2*N + B)                     transit_buckets_in    — incoming travel-time bucket volumes (pinned)
/// [… + B, … + B + A*k_max)                   anticipated_state     — incoming anticipated-ring slots (pinned)
/// … + A*k_max                                 theta                 — future cost variable (scalar)
/// ```
///
/// The anticipated ring mirrors the travel-time bucket ring construct-for-
/// construct: an outgoing block ([`Self::anticipated_slots_out`], identity-
/// resolved by [`Self::state_to_lp_column`], contributing to [`Self::n_state`])
/// and a separate incoming block ([`Self::anticipated_state`], pinned via
/// [`Self::state_to_lp_incoming_column`]) — never one dual-purpose range shifted
/// out-of-LP. Both DO contribute to [`Self::n_state`] (one state dimension per
/// ring slot, the `storage`/bucket convention) — there is no separate cut-target
/// alias column anymore; the plant's own newest slot (`k_i − 1`) IS the cut
/// target, pinned to the anticipated-decision column by the
/// `anticipated_state_out_def` equality row.
#[derive(Debug, Clone)]
pub struct StateLayout {
    /// Column range `[0, N)` for outgoing storage volumes.
    pub storage: Range<usize>,

    /// Column range `[N, N*(1+L))` for AR lag variables.
    ///
    /// Lag variables are stored in lag-major order: all hydros for lag 0,
    /// then all hydros for lag 1, etc. The column index for hydro `h` at
    /// lag `l` (0-indexed, lag 0 = most recent) is:
    /// `inflow_lags.start + l * hydro_count + h`.
    pub inflow_lags: Range<usize>,

    /// Column range `[N*(1+L), N*(1+L) + B)` for travel-time in-transit bucket
    /// state, immediately after [`Self::inflow_lags`] and before
    /// [`Self::anticipated_slots_out`] so anticipated offsets shift by a
    /// constant `B`.
    ///
    /// Outgoing bucket state maps to this column by identity — the `storage`
    /// convention, not the `z_inflow` lag remap.
    ///
    /// Empty (`0..0`) when [`Self::n_buckets`] `== 0`.
    pub transit_buckets_out: Range<usize>,

    /// Column range `[N*(1+L) + B, N*(1+L) + B + n_anticipated*k_max)` for the
    /// anticipated ring's OUTGOING slots.
    ///
    /// Ring-buffer block mirroring [`Self::transit_buckets_out`]: slot
    /// `k = 0..k_max` for anticipated plant `i = 0..n_anticipated` lives at
    /// column `anticipated_slots_out.start + k * n_anticipated + i`
    /// (slot-major, plant-minor). Each slot is a genuine LP column defined by
    /// an in-LP definition row — a plain ring shift
    /// (`slot_k^out = slot_{k+1}^in`) for `k < k_i − 1`, or the
    /// delivery-decision deposit (`slot_{k_i-1}^out = decision_col`) for
    /// plant `i`'s own newest slot `k_i − 1` — never resolved out-of-LP. Slots
    /// `k >= k_i` are stage-invariant padding, frozen `[0, 0]`.
    ///
    /// Outgoing anticipated state maps to this column by identity — the
    /// `transit_buckets_out` convention.
    ///
    /// Empty (`0..0`) when `n_anticipated * k_max == 0`.
    pub anticipated_slots_out: Range<usize>,

    /// Incoming storage volumes — the column block immediately after `z_inflow`
    /// (see the module-level column-layout diagram for the offset chain).
    ///
    /// Pinned to the preceding stage's outgoing `storage` solution values via
    /// `set_col_bounds` on these columns — not via equality rows (the LP has no
    /// state-fixing row range). Resolve the column with
    /// [`StateLayout::state_to_lp_incoming_column`].
    pub storage_in: Range<usize>,

    /// Column range for incoming travel-time bucket volumes, immediately after
    /// [`Self::storage_in`].
    ///
    /// Pinned like `storage_in` via `set_col_bounds` — not equality rows.
    /// Resolve with [`StateLayout::state_to_lp_incoming_column`].
    ///
    /// Empty (`0..0`) when [`Self::n_buckets`] `== 0`.
    pub transit_buckets_in: Range<usize>,

    /// Column range for realized-inflow variables `z_h`, one per hydro.
    ///
    /// Free columns (lower = -inf, upper = +inf, zero cost) holding total
    /// natural inflow `Z_t_h`, defined by the z-inflow equality constraints.
    ///
    /// Empty when `hydro_count == 0`.
    pub z_inflow: Range<usize>,

    /// Column range for the anticipated ring's INCOMING slots, immediately
    /// after [`Self::transit_buckets_in`] — the `transit_buckets_in`
    /// convention: a dedicated pinned-incoming block after the
    /// outgoing/`z_inflow`/`storage_in` chain, not interleaved with
    /// [`Self::anticipated_slots_out`].
    ///
    /// Same slot-major, plant-minor layout as `anticipated_slots_out`. Pinned
    /// via `set_col_bounds` — not equality rows (the LP has no state-fixing row
    /// range). Resolve with [`StateLayout::state_to_lp_incoming_column`]; slot
    /// 0 (the commitment maturing this stage) is also read directly by
    /// [`crate::lp_builder`]'s anticipated-fishing row fill.
    ///
    /// Empty (`0..0`) when `n_anticipated * k_max == 0`.
    pub anticipated_state: Range<usize>,

    /// The future cost variable (theta) — the final column, immediately after
    /// `anticipated_state` (see the module-level column-layout diagram).
    ///
    /// Scalar: there is exactly one theta variable per stage LP.
    pub theta: usize,

    /// State-vector dimension: `N*(1+L) + B`, or `N*(1+L) + B + A*K_max` with
    /// anticipated thermals.
    ///
    /// The dimension used by cut storage and broadcast payloads — **not** a
    /// valid LP row index. Do not slice the LP row buffer as `[0, n_state)`: no
    /// state-fixing rows exist; resolve the pinning/subgradient column via
    /// [`StateLayout::state_to_lp_incoming_column`].
    pub n_state: usize,

    /// Number of operating hydro plants (N).
    pub hydro_count: usize,

    /// Maximum PAR order across all operating hydros (L).
    ///
    /// Every hydro uses a uniform lag stride of `max_par_order`.
    pub max_par_order: usize,

    /// Global travel-time bucket count `B` (`Σ_j per_plant_depth[j]`), `0` when
    /// no arc is declared.
    pub n_buckets: usize,

    /// Number of anticipated thermals (plants with
    /// `anticipated_config.is_some()`).
    pub n_anticipated: usize,

    /// Maximum `lead_stages` across the anticipated thermals (`K_max`).
    pub k_max: usize,

    /// Per-plant `lead_stages` (`K_i`), indexed by anticipated-local position.
    ///
    /// Length [`Self::n_anticipated`].
    pub anticipated_lead_stages: Vec<usize>,

    /// Delivery-anchored point-commitment resolution per anticipated plant
    /// (anticipated-local order), threaded from setup for the manifest's
    /// delivery-anchor population ([`crate::policy::policy_export`]).
    /// Default-empty until [`Self::set_anticipated_resolution`] attaches it.
    pub(crate) anticipated_resolution: AnticipatedResolution,

    /// Canonical `(plant_canonical_idx, lag)` pair per bucket state-vector
    /// dimension, in [`Self::transit_buckets_out`] order.
    pub transit_bucket_column_order: Vec<(usize, usize)>,

    /// Indices of state dimensions whose cut coefficients can be nonzero.
    ///
    /// Storage indices `[0, N)` are always included. Lag indices `[N, N*(1+L))`
    /// are included only when `lag < effective_lag_count[hydro]`. Hydros with AR
    /// order < `max_par_order` have padded lag slots whose duals are
    /// structurally zero.
    pub nonzero_state_indices: Vec<usize>,

    /// Precomputed `state_to_lp_column(j)` for every `j ∈ [0, n_state)`, read on
    /// the forward-pass cut-row hot path.
    pub state_to_lp_column_map: Vec<usize>,
}

impl StateLayout {
    /// Construct a finalized [`StateLayout`] from the state dimensions and the
    /// per-hydro effective lag-slot counts.
    ///
    /// `effective_lag_count` must have length `hydro_count`; each entry is
    /// `PrecomputedPar::effective_lag_count(h)` — the number of lag-state slots
    /// that may carry non-zero cut coefficients for that hydro. `n_buckets` and
    /// `transit_bucket_column_order` are the travel-time bucket count and its canonical
    /// `(plant, lag)` order (`transit_bucket_column_order.len() == n_buckets`); `n_buckets
    /// == 0` reproduces the pre-bucket layout byte-for-byte.
    ///
    /// Both layout-derived caches are finalized here, so the returned value is
    /// ready for the cut-row hot path with no two-phase init.
    ///
    /// # Panics (debug builds only)
    ///
    /// Inherits the [`Self::set_nonzero_mask`] and
    /// [`Self::finalize_state_column_map`] debug assertions:
    /// `transit_bucket_column_order.len() == n_buckets`,
    /// `effective_lag_count.len() == hydro_count`,
    /// `anticipated_lead_stages.len() == n_anticipated`, lag/lead bounds, and
    /// `state_to_lp_column_map.len() == n_state`.
    #[must_use]
    pub fn new(
        hydro_count: usize,
        max_par_order: usize,
        n_buckets: usize,
        transit_bucket_column_order: Vec<(usize, usize)>,
        n_anticipated: usize,
        k_max: usize,
        anticipated_lead_stages: Vec<usize>,
        effective_lag_count: &[usize],
    ) -> Self {
        debug_assert_eq!(
            transit_bucket_column_order.len(),
            n_buckets,
            "transit_bucket_column_order must have exactly n_buckets entries"
        );

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

        let transit_buckets_out_start = n * (1 + l);
        let transit_buckets_out_end = transit_buckets_out_start + n_buckets;
        let transit_buckets_out = if n_buckets > 0 {
            transit_buckets_out_start..transit_buckets_out_end
        } else {
            0..0
        };

        let anticipated_slots_out_start = transit_buckets_out_end;
        let anticipated_slots_out_end = anticipated_slots_out_start + n_ant_state;
        let anticipated_slots_out = if n_ant_state > 0 {
            anticipated_slots_out_start..anticipated_slots_out_end
        } else {
            0..0
        };

        let z_inflow_start = anticipated_slots_out_end;
        let z_inflow = z_inflow_start..z_inflow_start + n;
        let storage_in_start = z_inflow.end;
        let storage_in = storage_in_start..storage_in_start + n;

        let transit_buckets_in_start = storage_in.end;
        let transit_buckets_in_end = transit_buckets_in_start + n_buckets;
        let transit_buckets_in = if n_buckets > 0 {
            transit_buckets_in_start..transit_buckets_in_end
        } else {
            0..0
        };

        let anticipated_state_start = transit_buckets_in_end;
        let anticipated_state_end = anticipated_state_start + n_ant_state;
        let anticipated_state = if n_ant_state > 0 {
            anticipated_state_start..anticipated_state_end
        } else {
            0..0
        };

        let theta = anticipated_state_end;

        // Both the outgoing (`anticipated_slots_out`) and incoming
        // (`anticipated_state`) ring blocks describe the SAME `A*k_max` state
        // dimensions (the `transit_buckets_out`/`transit_buckets_in` pairing),
        // so `n_ant_state` enters `n_state` once, not twice.
        let n_state = n * (1 + l) + n_buckets + n_ant_state;

        let mut layout = Self {
            storage,
            inflow_lags,
            transit_buckets_out,
            anticipated_slots_out,
            storage_in,
            transit_buckets_in,
            z_inflow,
            anticipated_state,
            theta,
            n_state,
            hydro_count,
            max_par_order,
            n_buckets,
            n_anticipated,
            k_max,
            anticipated_lead_stages,
            anticipated_resolution: AnticipatedResolution::default(),
            transit_bucket_column_order,
            nonzero_state_indices: Vec::new(),
            state_to_lp_column_map: Vec::new(),
        };

        let anticipated_k = layout.anticipated_lead_stages.clone();
        layout.set_nonzero_mask(effective_lag_count, &anticipated_k);
        layout.finalize_state_column_map();
        layout
    }

    /// Attach the setup-computed [`AnticipatedResolution`].
    ///
    /// The single call site is setup's `build_wired_indexer`; every other
    /// `StateLayout` construction leaves the default-empty resolution, since a
    /// second `resolve_point` call site is forbidden — this one threads through.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if the delivery-anchored ring depth exceeds the sized `k_max`, or
    /// the resolution does not carry exactly one `PointResolution` per plant.
    pub(crate) fn set_anticipated_resolution(&mut self, resolution: AnticipatedResolution) {
        debug_assert!(
            resolution.k_max <= self.k_max,
            "delivery-anchored ring depth {} exceeds the sized k_max {}",
            resolution.k_max,
            self.k_max
        );
        debug_assert_eq!(
            resolution.per_plant.len(),
            self.n_anticipated,
            "resolution must carry one PointResolution per anticipated plant"
        );
        self.anticipated_resolution = resolution;
    }

    /// First column of the control region (`theta + 1`): the state region
    /// occupies `[0, theta]`, equipment columns follow from here.
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
    /// - `[N*(1+L), N*(1+L) + B)`: `transit_buckets_out` slots → LP column `j`
    ///   (identity, the `storage` convention — no lag remap)
    /// - `[N*(1+L) + B, N*(1+L) + B + n_anticipated*k_max)`: `anticipated_slots_out`
    ///   slots → LP column `j` (identity, the same convention). Each slot is a
    ///   genuine outgoing LP column defined by an in-LP definition row (plain
    ///   shift or delivery-decision deposit; see
    ///   [`Self::anticipated_slots_out`]), so no shift-aware remap is needed
    ///   here — the ring transition is resolved by the LP, not by this map.
    #[inline]
    #[must_use]
    pub fn state_to_lp_column(&self, j: usize) -> usize {
        let n = self.hydro_count;
        if j < n {
            return j;
        }
        // Must precede the lag arithmetic below: bucket and anticipated-ring
        // indices are not lag indices, and the modular lag decode would
        // silently misresolve them once `max_par_order > 0`.
        if self.transit_buckets_out.contains(&j) || self.anticipated_slots_out.contains(&j) {
            return j;
        }
        if self.max_par_order == 0 {
            return j;
        }
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
    /// The map is a pure cache of the resolver — it never reimplements the
    /// mapping arithmetic.
    pub fn finalize_state_column_map(&mut self) {
        self.state_to_lp_column_map.clear();
        self.state_to_lp_column_map.reserve(self.n_state);
        for j in 0..self.n_state {
            self.state_to_lp_column_map.push(self.state_to_lp_column(j));
        }
        debug_assert_eq!(self.state_to_lp_column_map.len(), self.n_state);
    }

    /// Read the precomputed `state_to_lp_column(j)` from
    /// [`state_to_lp_column_map`](Self::state_to_lp_column_map).
    ///
    /// [`StateLayout::new`] always finalizes the map to `n_state` length, so the
    /// indexed read is in range for `j ∈ [0, n_state)` with no live-resolver
    /// fallback.
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
    /// - `j ∈ [N*(1+L), N*(1+L) + B)` (travel-time buckets): returns
    ///   `self.transit_buckets_in.start + (j − N*(1+L))` — an explicit arm, not the
    ///   anticipated catch-all below.
    /// - `j ∈ [N*(1+L) + B, n_state)` (anticipated state): returns
    ///   `self.anticipated_state.start + (j − N*(1+L) − B)`.
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
        let transit_bucket_end = lag_end + self.n_buckets;
        if j < n {
            self.storage_in.start + j
        } else if j < lag_end {
            self.inflow_lags.start + (j - n)
        } else if j < transit_bucket_end {
            self.transit_buckets_in.start + (j - lag_end)
        } else {
            self.anticipated_state.start + (j - transit_bucket_end)
        }
    }

    /// Whether anticipated plant `local_idx` emits a decision column and an
    /// `anticipated_state_out_def` row at `stage_idx`.
    ///
    /// The single cross-module owner of the anticipated-decision gate, the
    /// conjunction of two clauses on the **delivery** stage `t + K_i`:
    ///
    /// 1. **Strict horizon clause** — the commitment delivered `K_i` stages later
    ///    still falls **inside** the horizon, `stage_idx + K_i < n_stages`. The
    ///    `<` is strict and load-bearing: a `<=` gate would price a commitment
    ///    delivered at `stage_idx + K_i == n_stages`, whose delivery stage falls
    ///    outside `[0, n_stages)` and has no delivery LP.
    /// 2. **Operation-window clause** — the delivery stage is commissioning-active
    ///    for this plant, `commissioning_active(entry_i, exit_i, id(t + K_i))`. A
    ///    decision priced now matures at `t + K_i`; if that stage is outside the
    ///    plant's `[entry, exit)` operation window the matured generation column is
    ///    pinned to `[0, 0]`, so committing to it would buy generation that can
    ///    never be delivered. Keying on the DELIVERY stage (not the decision stage
    ///    `t`) is the shift the commissioning window needs: the decision is placed
    ///    `K_i` stages ahead of the first generating stage, so a decision at
    ///    `entry − K_i` (pre-entry) legitimately delivers at `entry`.
    ///
    /// The horizon clause is evaluated first and short-circuits, so the
    /// delivery-stage lookup `study_stage_ids[stage_idx + K_i]` is only reached
    /// when the index is in range.
    ///
    /// The per-plant windows and stage-id map flow in by reference, keeping the
    /// type a pure layout carrier.
    ///
    /// `anticipated_windows` is indexed by anticipated-local position
    /// (`0..n_anticipated`); `study_stage_ids` by study stage index.
    #[inline]
    #[must_use]
    pub fn is_anticipated_decision_active(
        &self,
        local_idx: usize,
        stage_idx: usize,
        n_stages: usize,
        anticipated_windows: &[(Option<i32>, Option<i32>)],
        study_stage_ids: &[i32],
    ) -> bool {
        debug_assert!(
            local_idx < self.anticipated_lead_stages.len(),
            "local_idx {local_idx} out of bounds (n_anticipated = {})",
            self.anticipated_lead_stages.len(),
        );
        debug_assert_eq!(
            anticipated_windows.len(),
            self.anticipated_lead_stages.len(),
            "anticipated_windows must have one entry per anticipated plant",
        );
        let delivery_stage = stage_idx.saturating_add(self.anticipated_lead_stages[local_idx]);
        self.is_anticipated_decision_active_for_delivery(
            local_idx,
            delivery_stage,
            n_stages,
            anticipated_windows,
            study_stage_ids,
        )
    }

    /// Whether anticipated plant `local_idx`'s commitment maturing at an
    /// EXPLICIT `delivery_stage` is active — the same horizon + commissioning
    /// gate as [`Self::is_anticipated_decision_active`], generalized so a
    /// fanned-out decision column (whose delivery stage is read from
    /// `PointResolution::genuine_decisions_at`, not derived as a constant
    /// offset from the decision stage) can compose the gate on ITS OWN
    /// delivery stage. [`Self::is_anticipated_decision_active`] delegates
    /// here with the constant-lead derived delivery stage — a pure refactor,
    /// not a behavior change.
    #[inline]
    #[must_use]
    pub fn is_anticipated_decision_active_for_delivery(
        &self,
        local_idx: usize,
        delivery_stage: usize,
        n_stages: usize,
        anticipated_windows: &[(Option<i32>, Option<i32>)],
        study_stage_ids: &[i32],
    ) -> bool {
        debug_assert!(
            local_idx < anticipated_windows.len(),
            "local_idx {local_idx} out of bounds (anticipated_windows.len() = {})",
            anticipated_windows.len(),
        );
        if delivery_stage >= n_stages {
            return false;
        }
        debug_assert!(
            delivery_stage < study_stage_ids.len(),
            "delivery_stage {delivery_stage} out of bounds for study_stage_ids \
             (len {})",
            study_stage_ids.len(),
        );
        let (entry, exit) = anticipated_windows[local_idx];
        crate::lp_builder::commissioning_active(entry, exit, study_stage_ids[delivery_stage])
    }

    /// Plant `local_idx`'s effective delivery-anchored resolution: the
    /// setup-threaded [`crate::lead_time::PointResolution`] when
    /// [`Self::anticipated_resolution`] is attached (production always
    /// attaches it via [`Self::set_anticipated_resolution`]), or an on-the-fly
    /// `LeadTime::Stages`-equivalent resolution built from the plant's
    /// constant [`Self::anticipated_lead_stages`] otherwise — the fixture
    /// fallback that keeps every existing constant-lead unit test
    /// byte-identical without threading a resolution through it.
    #[must_use]
    pub(crate) fn anticipated_resolution_for(
        &self,
        local_idx: usize,
        n_stages: usize,
    ) -> std::borrow::Cow<'_, crate::lead_time::PointResolution> {
        if !self.anticipated_resolution.per_plant.is_empty() {
            return std::borrow::Cow::Borrowed(&self.anticipated_resolution.per_plant[local_idx]);
        }
        let lead = u32::try_from(self.anticipated_lead_stages[local_idx]).unwrap_or(u32::MAX);
        std::borrow::Cow::Owned(crate::lead_time::resolve_point(
            crate::lead_time::LeadTime::Stages(lead),
            &[],
            n_stages,
        ))
    }

    /// Effective fan-out width for the decision-column geometry: the
    /// setup-threaded [`crate::lead_time::AnticipatedResolution::max_fanout`]
    /// when a resolution is attached, or `1` (single-decider) / `0` (no
    /// anticipation) derived from the constant [`Self::anticipated_lead_stages`]
    /// otherwise — mirrors [`Self::anticipated_resolution_for`]'s fallback.
    #[must_use]
    pub(crate) fn anticipated_max_fanout(&self) -> usize {
        if !self.anticipated_resolution.per_plant.is_empty() {
            return self.anticipated_resolution.max_fanout;
        }
        usize::from(self.anticipated_lead_stages.iter().any(|&k| k >= 1))
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
    /// Every travel-time bucket slot `transit_buckets_out.start..transit_buckets_out.end` is
    /// always included — bucket depth is already sized as the per-stage
    /// reachability union, so (unlike anticipated) there is no padding to
    /// exclude.
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
    /// `anticipated_slots_out.start + slot * n_anticipated + plant`. The loop
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
    /// After calling, `nonzero_state_indices` is sorted ascending with no
    /// duplicates.
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
        let mut mask =
            Vec::with_capacity(self.hydro_count + n_lag_active + self.n_buckets + n_ant_active);

        for h in 0..self.hydro_count {
            mask.push(h);
        }

        // Iterate lag-first (lag-major layout) to emit sorted indices.
        for lag in 0..self.max_par_order {
            for (h, &lag_count) in lag_counts.iter().enumerate() {
                debug_assert!(lag_count <= self.max_par_order);
                if lag < lag_count {
                    mask.push(self.inflow_lags.start + lag * self.hydro_count + h);
                }
            }
        }

        // Bucket depth is already sized as the per-stage reachability union (no
        // shared-stride padding like anticipated), so every declared slot is
        // reachable and the whole range is included, mirroring `storage` above.
        mask.extend(self.transit_buckets_out.clone());

        // Anticipated state: emit slots `0..K_i` for plant `i`, skipping padding.
        for slot in 0..self.k_max {
            for (plant, &k_i) in anticipated_lead_stages.iter().enumerate() {
                debug_assert!(k_i <= self.k_max);
                if slot < k_i {
                    mask.push(self.anticipated_slots_out.start + slot * self.n_anticipated + plant);
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
            0,
            Vec::new(),
            n_anticipated,
            k_max,
            anticipated_lead_stages,
            &lag_counts,
        )
    }

    /// Same as [`finalized`] but with a declared bucket block
    /// (`n_buckets`/`transit_bucket_column_order`), for the bucket-arm resolver and mask
    /// tests.
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
            finalized_with_transit_buckets(3, 2, 2, vec![(0, 1), (0, 2)], 2, 2, vec![1, 2]), // storage + lags + buckets + anticipated
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

    // ── is_anticipated_decision_active tests ───────────────────────────────────

    /// The strict horizon clause is active iff `stage_idx + K_i < n_stages`:
    /// active strictly inside the horizon, inactive at the `==` boundary and
    /// beyond. The `==` boundary must be inactive (a `<=` gate would price a
    /// commitment whose delivery stage falls outside `[0, n_stages)`). With
    /// windowless plants `(None, None)` the operation-window clause is always
    /// `true`, so this isolates the horizon clause.
    #[test]
    fn is_anticipated_decision_active_strict_horizon_gate() {
        // Two plants: K_0 = 1, K_1 = 2; k_max = 2; n_stages = 5.
        let idx = finalized(0, 0, 2, 2, vec![1, 2]);
        let n_stages = 5;
        // Windowless: the operation-window clause is identically true.
        let windows = [(None, None); 2];
        let stage_ids = [0, 1, 2, 3, 4];

        // Plant 0 (K=1): active while stage_idx + 1 < 5, i.e. stage_idx <= 3.
        assert!(idx.is_anticipated_decision_active(0, 0, n_stages, &windows, &stage_ids));
        assert!(idx.is_anticipated_decision_active(0, 3, n_stages, &windows, &stage_ids));
        // == boundary (stage_idx + K_i == n_stages): inactive.
        assert!(!idx.is_anticipated_decision_active(0, 4, n_stages, &windows, &stage_ids));
        // Beyond the boundary: inactive.
        assert!(!idx.is_anticipated_decision_active(0, 5, n_stages, &windows, &stage_ids));

        // Plant 1 (K=2): active while stage_idx + 2 < 5, i.e. stage_idx <= 2.
        assert!(idx.is_anticipated_decision_active(1, 2, n_stages, &windows, &stage_ids));
        // == boundary: inactive.
        assert!(!idx.is_anticipated_decision_active(1, 3, n_stages, &windows, &stage_ids));
        // Beyond: inactive.
        assert!(!idx.is_anticipated_decision_active(1, 4, n_stages, &windows, &stage_ids));
    }

    /// The operation-window clause gates on the DELIVERY stage's `stage.id`, not
    /// the decision stage. A single plant (K=2) with window `[entry=2, exit=4)`
    /// over a 6-stage horizon (stage ids `[0..6)`):
    ///
    /// - decision at `t=0` delivers at `id(2)=2` ∈ `[2,4)` → ACTIVE (pre-entry
    ///   decision: priced at `entry − K`),
    /// - decision at `t=1` delivers at `id(3)=3` ∈ `[2,4)` → ACTIVE,
    /// - decision at `t=2` delivers at `id(4)=4` ∉ `[2,4)` (half-open exit) →
    ///   INACTIVE (post-exit drain begins),
    /// - decision at `t=3` delivers at `id(5)=5` ∉ `[2,4)` → INACTIVE,
    /// - decision at `t < 0` not applicable; decision at `t=4` (`t+K=6 == n`) is
    ///   horizon-inactive regardless of window.
    ///
    /// The forbidden alternative — keying on the decision stage `t` — would make
    /// `t=2` (inside `[2,4)`) active and `t=0` (outside) inactive, the exact
    /// inversion the shift exists to prevent.
    #[test]
    fn is_anticipated_decision_active_delivery_stage_window_gate() {
        // One plant, K=2, k_max=2; n_stages=6; window [entry=2, exit=4).
        let idx = finalized(0, 0, 1, 2, vec![2]);
        let n_stages = 6;
        let windows = [(Some(2), Some(4))];
        let stage_ids = [0, 1, 2, 3, 4, 5];

        // Pre-entry decision (t=0): delivers at id 2 ∈ [2,4) → active.
        assert!(idx.is_anticipated_decision_active(0, 0, n_stages, &windows, &stage_ids));
        // t=1: delivers at id 3 ∈ [2,4) → active.
        assert!(idx.is_anticipated_decision_active(0, 1, n_stages, &windows, &stage_ids));
        // t=2: delivers at id 4 ∉ [2,4) (half-open exit) → inactive (drain).
        assert!(!idx.is_anticipated_decision_active(0, 2, n_stages, &windows, &stage_ids));
        // t=3: delivers at id 5 ∉ [2,4) → inactive.
        assert!(!idx.is_anticipated_decision_active(0, 3, n_stages, &windows, &stage_ids));
        // t=4: t+K=6 == n_stages → horizon-inactive (short-circuits before window).
        assert!(!idx.is_anticipated_decision_active(0, 4, n_stages, &windows, &stage_ids));
    }

    // ── state_to_lp_column tests ──────────────────────────────────────────────

    /// `anticipated_slots_out` indices resolve by identity when
    /// `max_par_order == 0 && n_anticipated > 0` — the ring transition
    /// (shift/deposit) is now an in-LP definition row, not a resolver-side
    /// remap. The `anticipated_slots_out` branch runs before the
    /// `max_par_order == 0` lag-block guard; verify identity holds even when
    /// there are no inflow lags.
    #[test]
    fn state_to_lp_column_anticipated_slots_out_is_identity_no_lag() {
        // N=1, L=0, n_anticipated=1, k_max=2, anticipated_lead_stages=[2].
        // n_state = 1*(1+0) + 1*2 = 3.
        // anticipated_slots_out = [1, 3); slot 0 at j=1, slot 1 at j=2.
        let idx = finalized(1, 0, 1, 2, vec![2]);
        assert_eq!(idx.anticipated_slots_out, 1..3);
        // Storage index: identity.
        assert_eq!(idx.state_to_lp_column(0), 0);
        // Both ring slots: identity.
        assert_eq!(idx.state_to_lp_column(1), 1);
        assert_eq!(idx.state_to_lp_column(2), 2);
    }

    /// `anticipated_slots_out` indices resolve by identity when
    /// `max_par_order > 0 && n_anticipated > 0`, and the lag-remap branch for
    /// non-anticipated indices is unaffected. Fixture: N=1, L=1,
    /// `n_anticipated=1`, `k_max=2`, `anticipated_lead_stages=[2]`.
    #[test]
    fn state_to_lp_column_anticipated_slots_out_is_identity_with_lag() {
        // N=1, L=1, n_anticipated=1, k_max=2, anticipated_lead_stages=[2].
        // n_state = 1*(1+1) + 1*2 = 4.
        // Layout: j=0 storage, j=1 lag-0, j=2 ant slot-0, j=3 ant slot-1.
        let idx = finalized(1, 1, 1, 2, vec![2]);
        assert_eq!(idx.anticipated_slots_out, 2..4);
        // Storage: identity.
        assert_eq!(idx.state_to_lp_column(0), 0);
        // Lag block: remapped (unaffected by the anticipated-ring change).
        // j=1: offset=0, h=0, lag=0 → z_inflow.start + 0.
        assert_eq!(idx.state_to_lp_column(1), idx.z_inflow.start);
        // Both ring slots: identity.
        assert_eq!(idx.state_to_lp_column(2), 2);
        assert_eq!(idx.state_to_lp_column(3), 3);
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

    /// Every anticipated ring slot resolves by identity, regardless of which
    /// slot is a plant's own newest (delivery-deposit) slot, a plain-shift
    /// interior slot, or a stage-invariant padding slot beyond that plant's own
    /// `K_p`: the in-LP definition rows (built in `lp/builder`) resolve the
    /// ring transition, so `state_to_lp_column` never special-cases a slot
    /// index the way the deleted constant-lead shift-map did.
    #[test]
    fn state_to_lp_column_anticipated_slots_out_identity_multi_plant_heterogeneous_k() {
        // Two plants: plant 0 has K_p=1 (only slot 0 is in-use), plant 1 has
        // K_p=3 (slots 0, 1, 2 all in-use). k_max=3 so plant 0 has padding
        // at slots 1 and 2.
        let idx = finalized(0, 0, 2, 3, vec![1, 3]);
        assert_eq!(idx.anticipated_slots_out, 0..6);
        for j in idx.anticipated_slots_out.clone() {
            assert_eq!(
                idx.state_to_lp_column(j),
                j,
                "anticipated ring slot {j} must resolve by identity"
            );
        }
    }

    /// The anticipated-ring identity resolution lands inside the **state
    /// region**: `anticipated_slots_out` sits strictly between
    /// `transit_buckets_out` and `theta`, never inside the control region.
    #[test]
    fn state_to_lp_column_anticipated_slots_out_resolves_into_state_region() {
        // N=3, L=2, A=2, k_max=3, uniform K_p = 3.
        let idx = finalized(3, 2, 2, 3, vec![3, 3]);
        for j in idx.anticipated_slots_out.clone() {
            let col = idx.state_to_lp_column(j);
            assert_eq!(col, j, "identity resolution");
            assert!(
                col >= idx.transit_buckets_out.end,
                "resolved column {col} must be >= transit_buckets_out.end {}",
                idx.transit_buckets_out.end
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
    /// `anticipated_state.start + j` (since `lag_end` = N*(1+L) = 0). With
    /// `N=0` every non-anticipated block collapses to `0..0`, so
    /// `anticipated_state` (the relocated incoming block, after
    /// `anticipated_slots_out`/`z_inflow`/`storage_in`/`transit_buckets_in`)
    /// starts at `anticipated_slots_out`'s own width (`A*K = 2`), not `0`.
    #[test]
    fn state_to_lp_incoming_column_anticipated_range() {
        // N=0, L=0, A=1, K=2: n_state = 0 + 1*2 = 2.
        let idx = finalized(0, 0, 1, 2, vec![2]);
        assert_eq!(idx.anticipated_state.start, 2);
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
        //   inflow_lags.start = N = 3.
        //   storage_in.start = N*(1+L) + A*K + N = 3*3 + 1*2 + 3 = 14
        //     (storage + inflow_lags + anticipated_slots_out + z_inflow).
        //   anticipated_state.start = storage_in.start + N = 17 (transit_buckets_in
        //     is empty; the relocated incoming block follows storage_in directly).
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
        // anticipated_slots_out.start = 0 (no storage, no lag block).
        // Layout: start + slot * n_anticipated + plant.
        let mut idx = finalized(0, 0, 2, 3, vec![3, 3]);

        assert_eq!(idx.anticipated_slots_out.start, 0);
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
        // inflow_lags = [3, 9), anticipated_slots_out.start = 9.
        let mut idx = finalized(3, 2, 2, 3, vec![3, 1]);

        assert_eq!(idx.anticipated_slots_out.start, 9);
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
        assert_eq!(idx.anticipated_slots_out.start, 0);

        idx.set_nonzero_mask(&[], &[2]);

        // Only anticipated indices: slot=0 plant=0 → 0; slot=1 plant=0 → 1.
        assert_eq!(idx.nonzero_state_indices, vec![0, 1]);
    }

    /// Heterogeneous `K_i` across plants, including a plant with `K_i = k_max`
    /// and another with `K_i < k_max`.
    #[test]
    fn nonzero_mask_anticipated_state_mixed_k_values() {
        // n_anticipated = 3, k_max = 4. Lead stages = [4, 2, 1].
        // anticipated_slots_out.start = 0 (no hydros).
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

    // ── Bucket block tests ─────────────────────────────────────────────────

    /// Bucket-arm resolution for `state_to_lp_column`: outgoing bucket state
    /// maps to its LP column by identity (the `storage` convention), not the
    /// lag remap — verified with lags present so the bucket check must
    /// correctly intercept before the modular lag decode.
    #[test]
    fn state_to_lp_column_transit_bucket_arm_is_identity() {
        // N=2, L=2 (lags present), B=3, no anticipated.
        let idx =
            finalized_with_transit_buckets(2, 2, 3, vec![(0, 1), (0, 2), (1, 1)], 0, 0, vec![]);

        assert_eq!(idx.transit_buckets_out, 6..9);
        for j in idx.transit_buckets_out.clone() {
            assert_eq!(
                idx.state_to_lp_column(j),
                j,
                "bucket state index {j} must map to its LP column by identity"
            );
        }
        // The last lag index (j=5, just below the bucket block) still resolves
        // via the lag remap, proving the bucket check does not swallow lag
        // indices.
        assert_eq!(idx.state_to_lp_column(5), 3);
    }

    /// Bucket-arm resolution for `state_to_lp_incoming_column`: bucket indices
    /// resolve to the pinned `transit_buckets_in` column via an explicit arm, not the
    /// anticipated catch-all — verified with anticipated state present so the
    /// catch-all `else` is live and would otherwise swallow them.
    #[test]
    fn state_to_lp_incoming_column_transit_bucket_arm_is_pinned_not_anticipated() {
        // N=2, L=1, B=2, A=1 (k_max=2, K=[2]).
        let idx = finalized_with_transit_buckets(2, 1, 2, vec![(0, 1), (0, 2)], 1, 2, vec![2]);

        assert_eq!(idx.transit_buckets_in, 12..14);
        assert_eq!(idx.anticipated_state.start, 14);

        // Bucket state indices: j=4, j=5 (lag_end = N*(1+L) = 4).
        assert_eq!(
            idx.state_to_lp_incoming_column(4),
            idx.transit_buckets_in.start
        );
        assert_eq!(
            idx.state_to_lp_incoming_column(5),
            idx.transit_buckets_in.start + 1
        );
        assert_ne!(
            idx.state_to_lp_incoming_column(4),
            idx.anticipated_state.start,
            "bucket index must not resolve to the anticipated catch-all"
        );

        // The first anticipated-state index (j=6) still resolves via the
        // catch-all, immediately after the bucket range.
        assert_eq!(
            idx.state_to_lp_incoming_column(6),
            idx.anticipated_state.start
        );
    }

    /// `state_to_lp_column_map.len() == n_state` for a bucket-only layout (no
    /// lags, no anticipated) — the map-length invariant isolated from the
    /// other blocks.
    #[test]
    fn state_to_lp_column_map_length_matches_n_state_with_transit_buckets_only() {
        let idx =
            finalized_with_transit_buckets(0, 0, 3, vec![(0, 1), (0, 2), (0, 3)], 0, 0, vec![]);

        assert_eq!(idx.n_state, 3);
        assert_eq!(idx.state_to_lp_column_map.len(), idx.n_state);
        for j in 0..idx.n_state {
            assert_eq!(idx.lp_column_for_state(j), j, "bucket-only identity map");
        }
    }

    /// Mask contiguity with a masked bucket slot: a bucket block is always
    /// fully included (mirroring `storage`), coexisting with a genuinely
    /// excluded lag slot from an unrelated block — the overall mask stays
    /// sorted with the excluded lag index as the only gap.
    #[test]
    fn nonzero_mask_transit_bucket_block_full_range_with_masked_lag_slot() {
        // N=2, L=2, B=2 (single plant, depth 2), no anticipated. Hydro 0 has
        // lag_count=1 (lag slot 1 masked out); hydro 1 has lag_count=2 (full).
        let mut idx = finalized_with_transit_buckets(2, 2, 2, vec![(0, 1), (0, 2)], 0, 0, vec![]);

        idx.set_nonzero_mask(&[1, 2], &[]);

        assert_eq!(idx.transit_buckets_out, 6..8);
        assert_eq!(
            idx.nonzero_state_indices,
            vec![0, 1, 2, 3, 5, 6, 7],
            "hydro 0's lag-1 slot (index 4) must be excluded while the full \
             bucket block (6, 7) is included"
        );
        assert!(
            idx.nonzero_state_indices.windows(2).all(|w| w[0] < w[1]),
            "mask must stay sorted and unique with a masked slot present"
        );
    }

    /// `B == 0` (no declared travel-time arc) collapses `transit_buckets_out`/
    /// `transit_buckets_in` to `0..0` and leaves every other offset the literal
    /// formula for `N=3, L=2, A=2, k_max=2`; a stray `+0` that reorders the
    /// sequential-offset chain would move one of these off its hardcoded value.
    #[test]
    fn state_layout_b_zero_is_byte_identical_to_pre_transit_bucket_layout() {
        let idx = finalized(3, 2, 2, 2, vec![1, 2]);

        assert_eq!(idx.n_buckets, 0);
        assert!(idx.transit_bucket_column_order.is_empty());
        assert_eq!(idx.transit_buckets_out, 0..0);
        assert_eq!(idx.transit_buckets_in, 0..0);

        assert_eq!(idx.storage, 0..3);
        assert_eq!(idx.inflow_lags, 3..9);
        assert_eq!(idx.anticipated_slots_out, 9..13);
        assert_eq!(idx.z_inflow, 13..16);
        assert_eq!(idx.storage_in, 16..19);
        assert_eq!(idx.anticipated_state, 19..23);
        assert_eq!(idx.theta, 23);
        assert_eq!(idx.n_state, 13);
    }

    /// `A * k_max == 0` (no anticipated thermals) collapses `anticipated_slots_out`/
    /// `anticipated_state` to `0..0` and reproduces the pre-anticipated-ring
    /// layout byte-for-byte (`N=3, L=2, B=2`) — the AC#1 collapse anchor.
    #[test]
    fn state_layout_a_zero_collapses_to_pre_anticipated_ring_layout() {
        let idx = finalized_with_transit_buckets(3, 2, 2, vec![(0, 1), (0, 2)], 0, 0, vec![]);

        assert_eq!(idx.n_anticipated, 0);
        assert_eq!(idx.k_max, 0);
        assert_eq!(idx.anticipated_slots_out, 0..0);
        assert_eq!(idx.anticipated_state, 0..0);

        assert_eq!(idx.storage, 0..3);
        assert_eq!(idx.inflow_lags, 3..9);
        assert_eq!(idx.transit_buckets_out, 9..11);
        assert_eq!(idx.z_inflow, 11..14);
        assert_eq!(idx.storage_in, 14..17);
        assert_eq!(idx.transit_buckets_in, 17..19);
        assert_eq!(idx.theta, 19);
        assert_eq!(idx.n_state, 11);

        for j in 0..idx.n_state {
            assert_eq!(
                idx.lp_column_for_state(j),
                idx.state_to_lp_column(j),
                "A==0 collapse must not disturb the storage/lag/bucket resolvers"
            );
        }
    }

    // ── In-LP anticipated ring: masking + collapse ─────────────

    /// `k_max == 0` collapses the ring to empty even when `n_anticipated > 0`
    /// (defensive: `n_ant_state = n_anticipated * k_max` is the sole gate, not
    /// `n_anticipated` alone), matching the `A * k_max == 0` layout exactly.
    #[test]
    fn anticipated_ring_k_max_zero_collapses_even_with_plants_declared() {
        let zero_k_max = StateLayout::new(3, 2, 0, Vec::new(), 2, 0, vec![0, 0], &[2, 2, 2]);
        let no_plants = StateLayout::new(3, 2, 0, Vec::new(), 0, 0, vec![], &[2, 2, 2]);

        assert_eq!(zero_k_max.anticipated_slots_out, 0..0);
        assert_eq!(zero_k_max.anticipated_state, 0..0);
        assert_eq!(zero_k_max.theta, no_plants.theta);
        assert_eq!(zero_k_max.n_state, no_plants.n_state);
        assert_eq!(
            zero_k_max.state_to_lp_column_map,
            no_plants.state_to_lp_column_map
        );
    }
}
