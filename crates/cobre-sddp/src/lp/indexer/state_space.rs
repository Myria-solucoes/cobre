//! The stage-invariant state-vector layout and its LP-column resolvers.
//!
//! State pinning uses column bounds, not equality rows:
//! [`StateSpace::state_to_lp_incoming_column`] is the single authoritative
//! incoming-state column resolver — the LP column for both pinning and dual
//! extraction is always resolved through it, never by assuming a fixing-row
//! index. The companion [`StateSpace::state_to_lp_column`] maps the outgoing
//! state vector to the LP columns a forward-pass cut row references.
//!
//! Every offset here is a pure function of `N` (`hydro_count`), `L`
//! (`max_par_order`), `B` (`n_buckets`), `A` (`n_anticipated`), `k_max`, and
//! `W` (`n_commitment`, the terminal commitment-hold post-horizon window
//! count) — independent of `n_blks`/`n_thermals` — so a single global stage-0
//! layout resolves onto the correct column at every stage regardless of
//! per-stage block counts.

use std::ops::Range;

use super::{InCol, OutCol, RangeCursor, StateDim};
use crate::lead_time::AnticipatedResolution;

use cobre_core::EntityId;
use cobre_core::temporal::StageStateConfig;

/// Stage-invariant state-vector layout for one SDDP stage subproblem.
///
/// ## Column layout
///
/// ```text
/// [0, N)                                     storage               — outgoing storage volumes (N = hydro_count)
/// [N, N*(1+L))                               inflow_lags           — AR lag variables (L lags per hydro)
/// [N*(1+L), N*(1+L) + B)                     transit_buckets_out   — travel-time bucket state (outgoing, identity)
/// [N*(1+L) + B, N*(1+L) + B + S)             commit_out            — commitment-hold outgoing slots (outgoing, identity)
/// [… + S, … + N)                             z_inflow              — realized inflow (auxiliary, not state)
/// [… + N, … + 2*N)                           storage_in            — incoming storage volumes
/// [… + 2*N, … + 2*N + B)                     transit_buckets_in    — incoming travel-time bucket volumes (pinned)
/// [… + B, … + B + S)                         commit_in             — incoming commitment-hold slots (pinned)
/// … + S                                       theta                 — future cost variable (scalar)
/// ```
///
/// `S = A*k_max + W`: the merged commitment-hold region packs the in-study
/// anticipated-ring slots and the terminal post-horizon lanes into one
/// contiguous out/in pair each — an outgoing block ([`Self::commit_out`],
/// identity-resolved, contributing to [`Self::n_state`]) and a separate
/// incoming block ([`Self::commit_in`], pinned via
/// [`Self::state_to_lp_incoming_column`]) — never one dual-purpose range
/// shifted out-of-LP. The leading `A*k_max` slots are in-study, slot-major/
/// plant-minor, keyed delivery-target-modular (delivery target `m`'s slot is
/// `m mod k_max`); the trailing `W` (= [`Self::n_commitment`]) slots are the
/// terminal post-horizon lanes, one per declared window, with no depth axis
/// (one slot per window, not per lead-stage). Like the bucket block, the
/// whole region is always cut-enabled, ignoring [`StageStateConfig`] (see
/// [`StateRegion::cut_enabled`]).
#[derive(Debug, Clone)]
pub struct StateSpace {
    /// Outgoing storage volumes.
    pub storage: Range<usize>,

    /// AR lag variables, lag-major: all hydros for lag 0, then lag 1, …. Hydro
    /// `h` at lag `l` is at `inflow_lags.start + l * hydro_count + h`.
    pub inflow_lags: Range<usize>,

    /// Travel-time in-transit bucket state. Outgoing bucket state maps to its
    /// LP column by identity — the `storage` convention, not the `z_inflow`
    /// lag remap.
    pub transit_buckets_out: Range<usize>,

    /// Commitment-hold OUTGOING slots: the leading `n_anticipated * k_max`
    /// in-study anticipated-ring slots, slot-major/plant-minor (slot `k` for
    /// plant `i` at `commit_out.start + k * n_anticipated + i` —
    /// [`Self::commitment_hold_in_study_offset`]), followed by the trailing
    /// `n_commitment` terminal post-horizon lanes, one per declared window
    /// (window `w` at `commit_out.start + n_anticipated * k_max + w` —
    /// [`Self::commitment_hold_post_horizon_offset`]). Every slot is a genuine
    /// LP column resolved by identity (the `transit_buckets_out` convention)
    /// and defined by an in-LP definition row — never resolved out-of-LP. An
    /// in-study slot `k >= k_i` for a plant whose own reachable depth is
    /// smaller is padding, frozen `[0, 0]`; a post-horizon lane is never
    /// padding (a window either resolves to a lane or is dropped before
    /// reaching this layout).
    pub commit_out: Range<usize>,

    /// Incoming storage volumes, pinned via
    /// [`StateSpace::state_to_lp_incoming_column`].
    pub storage_in: Range<usize>,

    /// Incoming travel-time bucket volumes, pinned via
    /// [`StateSpace::state_to_lp_incoming_column`].
    pub transit_buckets_in: Range<usize>,

    /// Realized-inflow variables `z_h`, one per hydro.
    pub z_inflow: Range<usize>,

    /// Commitment-hold INCOMING slots, the same leading in-study /
    /// trailing post-horizon layout as [`Self::commit_out`], pinned via
    /// [`StateSpace::state_to_lp_incoming_column`]. The in-study slot maturing
    /// this stage is also read directly by [`crate::lp_builder`]'s
    /// commitment-fishing row fill.
    pub commit_in: Range<usize>,

    /// Future cost variable (theta) column.
    pub theta: usize,

    /// State-vector dimension used by cut storage and broadcast payloads —
    /// **not** a valid LP row index. No state-fixing rows exist; do not slice
    /// the LP row buffer as `[0, n_state)`. Resolve the pinning/subgradient
    /// column via [`StateSpace::state_to_lp_incoming_column`].
    pub n_state: usize,

    /// Number of operating hydro plants (N).
    pub hydro_count: usize,

    /// Maximum PAR order across all operating hydros (L); every hydro uses this
    /// uniform lag stride.
    pub max_par_order: usize,

    /// Global travel-time bucket count `B` (`Σ_j per_plant_depth[j]`), `0` when
    /// no arc is declared.
    pub n_buckets: usize,

    /// Number of anticipated thermals (plants with
    /// `anticipated_config.is_some()`).
    pub n_anticipated: usize,

    /// Maximum `lead_stages` across the anticipated thermals (`K_max`).
    pub k_max: usize,

    /// Per-plant `lead_stages` (`K_i`), indexed by anticipated-local position;
    /// length [`Self::n_anticipated`].
    pub anticipated_lead_stages: Vec<usize>,

    /// Delivery-anchored point-commitment resolution per anticipated plant
    /// (anticipated-local order); default-empty until
    /// [`Self::set_anticipated_resolution`] attaches it.
    pub(crate) anticipated_resolution: AnticipatedResolution,

    /// Terminal commitment-hold post-horizon window count (`W`); the trailing
    /// width of both [`Self::commit_out`] and [`Self::commit_in`], after their
    /// shared leading `n_anticipated * k_max` in-study width.
    pub n_commitment: usize,

    /// Per-window in-study decider stage index, length [`Self::n_commitment`],
    /// in [`Self::commit_out`]'s post-horizon-lane order: window `w` latches
    /// at stage `commitment_decider_stage[w]` and carries by identity at
    /// every other stage.
    pub(crate) commitment_decider_stage: Vec<usize>,

    /// Per-window owning thermal id, length [`Self::n_commitment`], parallel to
    /// [`Self::commitment_decider_stage`]. [`crate::policy_export::build_stage_entity_manifest`]
    /// derives a window's manifest `subindex` from the count of prior entries
    /// sharing its owner — never a raw window index — so two windows of the
    /// same thermal get distinct subindices.
    pub(crate) commitment_window_thermal_id: Vec<EntityId>,

    /// Canonical `(plant_canonical_idx, lag)` pair per bucket state-vector
    /// dimension, in [`Self::transit_buckets_out`] order.
    pub transit_bucket_column_order: Vec<(usize, usize)>,

    /// State dimensions whose cut coefficients can be nonzero (padded lag/ring
    /// slots excluded); computed by [`Self::set_nonzero_mask`].
    pub nonzero_state_indices: Vec<StateDim>,

    /// Precomputed `state_to_lp_column(j)` for every `j ∈ [0, n_state)` (cut-row
    /// hot path).
    pub state_to_lp_column_map: Vec<OutCol>,
}

/// One of the four stage-invariant state-vector regions, in the canonical walk
/// order [`REGION_ORDER`] declares.
#[derive(Debug, Clone, Copy)]
pub(crate) enum StateRegion {
    /// Outgoing storage volumes — [`StateSpace::state_dim_storage_range`].
    Storage,
    /// AR inflow lags — [`StateSpace::state_dim_lag_range`].
    Lag,
    /// Travel-time in-transit buckets — [`StateSpace::state_dim_bucket_range`].
    Buckets,
    /// Merged in-study anticipated-ring + terminal post-horizon commitment
    /// slots — [`StateSpace::state_dim_commitment_hold_range`].
    CommitmentHold,
}

/// The single owner of the `storage → lag → buckets → commitment hold`
/// state-region walk order. [`StateSpace::state_to_lp_column`],
/// [`StateSpace::state_to_lp_incoming_column`],
/// [`StateSpace::set_nonzero_mask`], and [`super::CutStateProjection::new`]
/// all dispatch over this array via [`StateSpace::state_dim_range`]; a
/// variant added to [`StateRegion`] fails to compile at any of those sites
/// (each matches it exhaustively, no `_` catch-all) until handled.
pub(crate) const REGION_ORDER: [StateRegion; 4] = [
    StateRegion::Storage,
    StateRegion::Lag,
    StateRegion::Buckets,
    StateRegion::CommitmentHold,
];

impl StateRegion {
    /// Whether `region`'s state dimensions are cut-enabled under `config`: storage
    /// and lag are config-gated; buckets and the merged commitment-hold region
    /// are always included, regardless of `config` — every declared in-study or
    /// post-horizon commitment slot is a priced FCF dimension at every pool, the
    /// terminal pool included, so a loaded boundary cut's coefficient lands on
    /// the right column. Gating either on `config` shrinks the cut pool's
    /// state-dimension below the global trial state and misaligns the intercept
    /// dot ([`super::CutStateProjection::new`]).
    #[inline]
    #[must_use]
    pub(crate) fn cut_enabled(self, config: StageStateConfig) -> bool {
        match self {
            StateRegion::Storage => config.storage,
            StateRegion::Lag => config.inflow_lags,
            StateRegion::Buckets | StateRegion::CommitmentHold => true,
        }
    }
}

/// The two addressing forms a [`StateSpace::commit_out`]/
/// [`StateSpace::commit_in`]-relative offset decodes to — the exact inverse
/// of [`StateSpace::commitment_hold_in_study_offset`] and
/// [`StateSpace::commitment_hold_post_horizon_offset`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommitmentHoldAddress {
    /// In-study anticipated-ring position: anticipated-local plant `plant`'s
    /// modular slot `slot = m mod k_max` for some delivery target `m`.
    InStudy { plant: usize, slot: usize },
    /// Terminal post-horizon lane for declared window `window`.
    PostHorizon { window: usize },
}

impl StateSpace {
    /// Construct a finalized [`StateSpace`] from the state dimensions and the
    /// per-hydro effective lag-slot counts.
    ///
    /// `effective_lag_count` must have length `hydro_count`; each entry is
    /// `PrecomputedPar::effective_lag_count(h)` — the count of lag slots that
    /// may carry non-zero cut coefficients (see [`Self::set_nonzero_mask`] for
    /// why `order(h)` is wrong). `transit_bucket_column_order` is the buckets'
    /// canonical `(plant, lag)` order (`len() == n_buckets`); `n_buckets == 0`
    /// reproduces the pre-bucket layout byte-for-byte.
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
        Self::new_with_commitment_hold_windows(
            hydro_count,
            max_par_order,
            n_buckets,
            transit_bucket_column_order,
            n_anticipated,
            k_max,
            anticipated_lead_stages,
            effective_lag_count,
            0,
            Vec::new(),
            Vec::new(),
        )
    }

    /// [`Self::new`] plus the terminal commitment-hold windows: `n_commitment`
    /// declared post-horizon windows, each latching at its own
    /// `commitment_decider_stage[w]` and owned by
    /// `commitment_window_thermal_id[w]` (both length `n_commitment`,
    /// [`Self::commit_out`]'s post-horizon-lane order). `n_commitment == 0`
    /// with `n_anticipated * k_max == 0` collapses
    /// [`Self::commit_out`]/[`Self::commit_in`] to `0..0` and reproduces
    /// [`Self::new`]'s layout byte-for-byte — the single owner every other
    /// constructor delegates to.
    ///
    /// # Panics (debug builds only)
    ///
    /// Inherits [`Self::new`]'s panics, plus panics if
    /// `commitment_decider_stage.len() != n_commitment` or
    /// `commitment_window_thermal_id.len() != n_commitment`.
    // Rationale: each argument sizes an independent state region (storage, lags,
    // buckets, the anticipated ring, the commitment block); `Self::new` forwards
    // eight of these unchanged, so splitting the eleven into a sub-struct would just
    // wrap-then-immediately-destructure at the sole call site (`resolve_state_layout`).
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub(crate) fn new_with_commitment_hold_windows(
        hydro_count: usize,
        max_par_order: usize,
        n_buckets: usize,
        transit_bucket_column_order: Vec<(usize, usize)>,
        n_anticipated: usize,
        k_max: usize,
        anticipated_lead_stages: Vec<usize>,
        effective_lag_count: &[usize],
        n_commitment: usize,
        commitment_decider_stage: Vec<usize>,
        commitment_window_thermal_id: Vec<EntityId>,
    ) -> Self {
        debug_assert_eq!(
            transit_bucket_column_order.len(),
            n_buckets,
            "transit_bucket_column_order must have exactly n_buckets entries"
        );
        debug_assert_eq!(
            commitment_decider_stage.len(),
            n_commitment,
            "commitment_decider_stage must have exactly n_commitment entries"
        );
        debug_assert_eq!(
            commitment_window_thermal_id.len(),
            n_commitment,
            "commitment_window_thermal_id must have exactly n_commitment entries"
        );

        let n = hydro_count;
        let l = max_par_order;
        let n_ant_state = n_anticipated * k_max;
        let n_commit_hold = n_ant_state + n_commitment;

        // Optional blocks collapse to the literal `0..0`, not `RangeCursor::alloc`'s
        // `pos..pos` — skipping `alloc` when a count is `0` is safe because a
        // zero-length allocation leaves `cursor.pos()` unchanged either way, so
        // every downstream offset is identical regardless of which branch runs.
        let mut cursor = RangeCursor::new(0);
        let storage = cursor.alloc(n);
        let inflow_lags = cursor.alloc(n * l);
        let transit_buckets_out = if n_buckets > 0 {
            cursor.alloc(n_buckets)
        } else {
            0..0
        };
        // One alloc call spanning both the leading in-study slots and the
        // trailing post-horizon lanes — `commit_out`/`commit_in` are single
        // contiguous ranges, never a separately-allocated pair stitched
        // together after the fact.
        let commit_out = if n_commit_hold > 0 {
            cursor.alloc(n_commit_hold)
        } else {
            0..0
        };
        let z_inflow = cursor.alloc(n);
        let storage_in = cursor.alloc(n);
        let transit_buckets_in = if n_buckets > 0 {
            cursor.alloc(n_buckets)
        } else {
            0..0
        };
        let commit_in = if n_commit_hold > 0 {
            cursor.alloc(n_commit_hold)
        } else {
            0..0
        };

        let theta = cursor.pos();

        // Outgoing and incoming commitment-hold pairs describe the SAME state
        // dimensions each, so `n_commit_hold` enters `n_state` once, not twice.
        let n_state = n * (1 + l) + n_buckets + n_commit_hold;

        debug_assert_eq!(
            commit_out.len(),
            n_ant_state + n_commitment,
            "commit_out must span exactly the in-study slots plus the post-horizon lanes"
        );
        debug_assert_eq!(
            commit_out.len(),
            commit_in.len(),
            "commit_out and commit_in must have equal width"
        );

        let mut layout = Self {
            storage,
            inflow_lags,
            transit_buckets_out,
            commit_out,
            storage_in,
            transit_buckets_in,
            z_inflow,
            commit_in,
            theta,
            n_state,
            hydro_count,
            max_par_order,
            n_buckets,
            n_anticipated,
            k_max,
            anticipated_lead_stages,
            anticipated_resolution: AnticipatedResolution::default(),
            n_commitment,
            commitment_decider_stage,
            commitment_window_thermal_id,
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

    // ── Canonical state-dimension region boundaries ─────────────────────────
    // GLOBAL STATE INDEX space, not LP columns. `REGION_ORDER` is the single
    // owner of the storage → lag → bucket → anticipated walk order;
    // `state_dim_range` dispatches each `StateRegion` to the accessor below,
    // so every REGION_ORDER-driven call site shares one boundary derivation.

    /// State-dimension region `[0, N)` — storage.
    #[inline]
    #[must_use]
    pub(crate) fn state_dim_storage_range(&self) -> Range<usize> {
        0..self.hydro_count
    }

    /// State-dimension region `[N, N*(1+L))` — AR inflow lags.
    #[inline]
    #[must_use]
    pub(crate) fn state_dim_lag_range(&self) -> Range<usize> {
        let n = self.hydro_count;
        n..n * (1 + self.max_par_order)
    }

    /// State-dimension region `[N*(1+L), N*(1+L) + B)` — travel-time buckets.
    #[inline]
    #[must_use]
    pub(crate) fn state_dim_bucket_range(&self) -> Range<usize> {
        let start = self.state_dim_lag_range().end;
        start..start + self.n_buckets
    }

    /// State-dimension region `[N*(1+L) + B, N*(1+L) + B + S)` — the merged
    /// commitment-hold region (`S = A*k_max + W`): the leading `A*k_max`
    /// in-study anticipated-ring slots followed by the trailing `W` (=
    /// [`Self::n_commitment`]) terminal post-horizon lanes.
    #[inline]
    #[must_use]
    pub(crate) fn state_dim_commitment_hold_range(&self) -> Range<usize> {
        let start = self.state_dim_bucket_range().end;
        start..start + self.n_anticipated * self.k_max + self.n_commitment
    }

    /// Resolve `region`'s state-dimension range through the matching
    /// `state_dim_*_range` accessor above — the exhaustive dispatch every
    /// [`REGION_ORDER`]-driven call site uses instead of re-deriving region
    /// boundaries inline.
    #[inline]
    #[must_use]
    pub(crate) fn state_dim_range(&self, region: StateRegion) -> Range<usize> {
        match region {
            StateRegion::Storage => self.state_dim_storage_range(),
            StateRegion::Lag => self.state_dim_lag_range(),
            StateRegion::Buckets => self.state_dim_bucket_range(),
            StateRegion::CommitmentHold => self.state_dim_commitment_hold_range(),
        }
    }

    /// `region`'s incoming pinned-block start column — the single owner of
    /// which block [`Self::state_to_lp_incoming_column`] pins.
    #[inline]
    #[must_use]
    pub(crate) fn incoming_block_start(&self, region: StateRegion) -> usize {
        match region {
            StateRegion::Storage => self.storage_in.start,
            StateRegion::Lag => self.inflow_lags.start,
            StateRegion::Buckets => self.transit_buckets_in.start,
            StateRegion::CommitmentHold => self.commit_in.start,
        }
    }

    /// Classify `j` into its [`StateRegion`] and its offset within that
    /// region's [`Self::state_dim_range`] — the single scan both
    /// [`Self::state_to_lp_column`] and [`Self::state_to_lp_incoming_column`]
    /// consume.
    ///
    /// [`REGION_ORDER`]'s four ranges partition `[0, n_state)` contiguously
    /// (`state_dim_ranges_partition_n_state_contiguously`), so `find` only
    /// misses for an out-of-range `j`; `unwrap_or` keeps this function total
    /// instead of adding a panic path for that case.
    #[inline]
    #[must_use]
    pub(crate) fn classify(&self, j: StateDim) -> (StateRegion, usize) {
        let j = j.get();
        let region = REGION_ORDER
            .into_iter()
            .find(|&region| self.state_dim_range(region).contains(&j))
            .unwrap_or(StateRegion::CommitmentHold);
        (region, j - self.state_dim_range(region).start)
    }

    /// Map a state-vector index to the LP column it references in a cut.
    ///
    /// Classifies `j` into its `StateRegion` via `REGION_ORDER` before any
    /// lag arithmetic runs, then resolves through an exhaustive match:
    /// storage, `transit_buckets_out`, and `commit_out` map by identity. Lag
    /// indices remap to the outgoing state after `shift_lag_state`: lag 0 is
    /// realised inflow → `z_inflow.start + h`; lag `l ≥ 1` is the previous
    /// stage's lag `l − 1` → `inflow_lags.start + (l − 1)·N + h`. Classifying
    /// first — rather than falling through an `if`/`else` chain — is what
    /// keeps buckets/commitment-hold from ever reaching the lag decode.
    ///
    /// A bare `usize` cannot skip the [`StateDim`] wrap at this resolver
    /// boundary:
    ///
    /// ```compile_fail
    /// use cobre_sddp::indexer::StateSpace;
    ///
    /// let state = StateSpace::new(1, 0, 0, Vec::new(), 0, 0, Vec::new(), &[0]);
    /// let _col = state.state_to_lp_column(0); // bare usize handed where StateDim is required
    /// ```
    ///
    /// Nor can an already-resolved [`OutCol`] re-enter as the unresolved
    /// dimension:
    ///
    /// ```compile_fail
    /// use cobre_sddp::indexer::{StateDim, StateSpace};
    ///
    /// let state = StateSpace::new(1, 0, 0, Vec::new(), 0, 0, Vec::new(), &[0]);
    /// let col = state.state_to_lp_column(StateDim::new(0));
    /// let _reentered = state.state_to_lp_column(col); // OutCol handed where StateDim is required
    /// ```
    #[inline]
    #[must_use]
    pub fn state_to_lp_column(&self, j: StateDim) -> OutCol {
        let (region, offset) = self.classify(j);

        OutCol::new(match region {
            StateRegion::Storage | StateRegion::Buckets | StateRegion::CommitmentHold => j.get(),
            StateRegion::Lag => {
                let n = self.hydro_count;
                let h = offset % n;
                let lag = offset / n;
                if lag == 0 {
                    self.z_inflow.start + h
                } else {
                    n + (lag - 1) * n + h
                }
            }
        })
    }

    /// Fill [`Self::state_to_lp_column_map`] by calling
    /// [`Self::state_to_lp_column`] for every `j ∈ [0, n_state)` — a pure cache
    /// of the resolver, never a reimplementation of its arithmetic.
    pub fn finalize_state_column_map(&mut self) {
        self.state_to_lp_column_map.clear();
        self.state_to_lp_column_map.reserve(self.n_state);
        for j in 0..self.n_state {
            self.state_to_lp_column_map
                .push(self.state_to_lp_column(StateDim::new(j)));
        }
        debug_assert_eq!(self.state_to_lp_column_map.len(), self.n_state);
    }

    /// Read the precomputed `state_to_lp_column(j)` from
    /// [`Self::state_to_lp_column_map`], which [`StateSpace::new`] always
    /// finalizes to `n_state` length (indexed read is in range for
    /// `j ∈ [0, n_state)`).
    #[inline]
    #[must_use]
    pub fn lp_column_for_state(&self, j: StateDim) -> OutCol {
        let j = j.get();
        debug_assert_eq!(
            self.state_to_lp_column_map.len(),
            self.n_state,
            "state_to_lp_column_map must be finalized to n_state length"
        );
        self.state_to_lp_column_map[j]
    }

    /// Map a state-vector index to its **incoming-state** LP column — the column
    /// pinned to `lb = ub = v` via `set_col_bounds` (never an equality/fixing
    /// row; the LP has no state-fixing row range). The returned columns are
    /// exactly those
    /// [`fill_col_state_patches`](crate::lp_builder::PatchBuffer::fill_col_state_patches)
    /// writes, in state-vector order; the backward pass reads
    /// `view.reduced_costs[col]` at them for the cut subgradient, one per
    /// component `j ∈ [0, n_state)`.
    ///
    /// The travel-time bucket range resolves through an explicit
    /// `transit_buckets_in` arm, not the trailing anticipated-state catch-all.
    ///
    /// Contrast [`state_to_lp_column`], which returns the **outgoing** column
    /// for forward-pass cut-row construction: for storage it returns `j`
    /// (outgoing), this returns `storage_in.start + j` (incoming). The two are
    /// related by the water-balance equality row — by KKT duality the incoming
    /// column's reduced cost equals the dual a row-based state-fixing
    /// formulation would produce, which is why column-bound pinning is exact.
    ///
    /// [`state_to_lp_column`]: Self::state_to_lp_column
    #[inline]
    #[must_use]
    pub fn state_to_lp_incoming_column(&self, j: StateDim) -> InCol {
        let (region, offset) = self.classify(j);
        InCol::new(self.incoming_block_start(region) + offset)
    }

    /// `region`'s incoming pinned-block column range.
    fn incoming_block_range(&self, region: StateRegion) -> Range<usize> {
        let start = self.incoming_block_start(region);
        start..start + self.state_dim_range(region).len()
    }

    /// Inverse of [`Self::state_to_lp_incoming_column`]: the region and
    /// in-region offset owning pinned column `c`. Total for the same reason
    /// as [`Self::classify`].
    #[must_use]
    pub(crate) fn classify_incoming_column(&self, c: InCol) -> (StateRegion, usize) {
        let c = c.get();
        let region = REGION_ORDER
            .into_iter()
            .find(|&region| self.incoming_block_range(region).contains(&c))
            .unwrap_or(StateRegion::CommitmentHold);
        (region, c - self.incoming_block_start(region))
    }

    /// Encode an in-study commitment-hold position: anticipated-local plant
    /// `plant`'s offset within [`Self::commit_out`]/[`Self::commit_in`] for
    /// delivery target `m`, keyed delivery-target-modular
    /// (`slot(m) = m mod k_max`), slot-major/plant-minor.
    #[must_use]
    pub(crate) fn commitment_hold_in_study_offset(&self, plant: usize, m: usize) -> usize {
        debug_assert!(
            plant < self.n_anticipated,
            "plant {plant} out of range (n_anticipated = {})",
            self.n_anticipated
        );
        debug_assert!(
            self.k_max > 0,
            "commitment_hold_in_study_offset requires k_max > 0"
        );
        (m % self.k_max) * self.n_anticipated + plant
    }

    /// Encode a post-horizon commitment-hold position: declared window
    /// `window`'s offset within [`Self::commit_out`]/[`Self::commit_in`],
    /// appended past the leading in-study block. The encoder counterpart to
    /// [`Self::commitment_hold_address`]'s decoder.
    // Rationale: no production caller needs the forward encode direction yet
    // (`commitment_hold_address` covers the decode direction the manifest
    // reads); kept alongside its decoder for the round-trip the tests pin.
    #[allow(dead_code)]
    #[must_use]
    pub(crate) fn commitment_hold_post_horizon_offset(&self, window: usize) -> usize {
        debug_assert!(
            window < self.n_commitment,
            "window {window} out of range (n_commitment = {})",
            self.n_commitment
        );
        self.n_anticipated * self.k_max + window
    }

    /// Decode a [`Self::commit_out`]/[`Self::commit_in`]-relative offset back
    /// to its addressing form — the exact inverse of
    /// [`Self::commitment_hold_in_study_offset`] and
    /// [`Self::commitment_hold_post_horizon_offset`]. `slot` (not `m`) is all
    /// an offset can recover: every delivery target congruent modulo `k_max`
    /// addresses the same in-study slot.
    #[must_use]
    pub(crate) fn commitment_hold_address(&self, offset: usize) -> CommitmentHoldAddress {
        let n_ant_state = self.n_anticipated * self.k_max;
        debug_assert!(
            offset < n_ant_state + self.n_commitment,
            "offset {offset} out of range (commit width = {})",
            n_ant_state + self.n_commitment
        );
        if offset < n_ant_state {
            CommitmentHoldAddress::InStudy {
                plant: offset % self.n_anticipated,
                slot: offset / self.n_anticipated,
            }
        } else {
            CommitmentHoldAddress::PostHorizon {
                window: offset - n_ant_state,
            }
        }
    }

    fn storage_state_dim(&self, h: usize) -> StateDim {
        debug_assert!(h < self.hydro_count);
        StateDim::new(self.state_dim_storage_range().start + h)
    }

    /// Incoming (stage-initial) storage column of hydro `h`.
    #[must_use]
    pub(crate) fn storage_incoming_col(&self, h: usize) -> InCol {
        self.state_to_lp_incoming_column(self.storage_state_dim(h))
    }

    /// Outgoing (stage-final) storage column of hydro `h`.
    #[must_use]
    pub(crate) fn storage_outgoing_col(&self, h: usize) -> OutCol {
        self.state_to_lp_column(self.storage_state_dim(h))
    }

    /// Incoming pinned lag column of hydro `h` at `lag` (lag-major block).
    #[must_use]
    pub(crate) fn lag_incoming_col(&self, lag: usize, h: usize) -> InCol {
        debug_assert!(lag < self.max_par_order && h < self.hydro_count);
        let start = self.state_dim_lag_range().start;
        self.state_to_lp_incoming_column(StateDim::new(start + lag * self.hydro_count + h))
    }

    /// Incoming pinned bucket column of bucket `b`
    /// ([`Self::transit_bucket_column_order`] order).
    #[must_use]
    pub(crate) fn bucket_incoming_col(&self, b: usize) -> InCol {
        debug_assert!(b < self.n_buckets);
        self.state_to_lp_incoming_column(StateDim::new(self.state_dim_bucket_range().start + b))
    }

    /// Outgoing bucket column of bucket `b`.
    #[must_use]
    pub(crate) fn bucket_outgoing_col(&self, b: usize) -> OutCol {
        debug_assert!(b < self.n_buckets);
        self.state_to_lp_column(StateDim::new(self.state_dim_bucket_range().start + b))
    }

    /// Compute and store [`Self::nonzero_state_indices`] from per-hydro
    /// lag-slot counts and per-plant anticipated lead-stage counts.
    ///
    /// `lag_counts` must have length `hydro_count`; `lag_counts[h]` is the count
    /// of lag slots that may carry non-zero cut coefficients for hydro `h`. It
    /// must be `PrecomputedPar::effective_lag_count(h)`, **not** `order(h)`: for
    /// PAR(p)-A hydros `effective_lag_count == max_par_order` so the `ψ̂/12`
    /// annual contributions on slots `order..max_par_order` reach the cut rows;
    /// `order(h)` would truncate them and produce over-estimating cuts (LB > UB
    /// at convergence). Storage `[0, N)` is always included.
    ///
    /// Every travel-time bucket slot is always included — bucket depth is
    /// already sized as the per-stage reachability union, so (unlike the
    /// commitment-hold region's in-study slots) there is no padding to
    /// exclude. Every commitment-hold post-horizon lane is likewise always
    /// included — a window either resolves to a lane or is dropped before
    /// reaching this layout, so (unlike the in-study slots) there is no
    /// padding to exclude there either.
    ///
    /// For the commitment-hold region's in-study slots, anticipated plant `i`
    /// only has slots `0..K_i` included; the trailing `k_max − K_i` are
    /// padding whose cut coefficients are structurally zero (no decision
    /// writes them). Including padding over-estimates cut hyperplanes — the
    /// same failure mode as the lag block above.
    ///
    /// The loop iterates lag/slot-first so the emitted indices stay strictly
    /// ascending (the sortedness the `debug_assert` enforces).
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
        let mut mask = Vec::with_capacity(
            self.hydro_count + n_lag_active + self.n_buckets + n_ant_active + self.n_commitment,
        );

        // REGION_ORDER fixes the walk order; storage and buckets have no
        // padding to exclude and extend their full range, lag keeps its own
        // per-region active-slot filter (padding stays excluded — see the doc
        // comment above), and commitment-hold applies that same filter to its
        // leading in-study slots before extending its trailing post-horizon
        // lanes unfiltered.
        for region in REGION_ORDER {
            match region {
                StateRegion::Storage | StateRegion::Buckets => {
                    mask.extend(self.state_dim_range(region).map(StateDim::new));
                }
                StateRegion::Lag => {
                    let start = self.state_dim_range(region).start;
                    for lag in 0..self.max_par_order {
                        for (h, &lag_count) in lag_counts.iter().enumerate() {
                            debug_assert!(lag_count <= self.max_par_order);
                            if lag < lag_count {
                                mask.push(StateDim::new(start + lag * self.hydro_count + h));
                            }
                        }
                    }
                }
                StateRegion::CommitmentHold => {
                    let start = self.state_dim_range(region).start;
                    for slot in 0..self.k_max {
                        for (plant, &k_i) in anticipated_lead_stages.iter().enumerate() {
                            debug_assert!(k_i <= self.k_max);
                            if slot < k_i {
                                mask.push(StateDim::new(start + slot * self.n_anticipated + plant));
                            }
                        }
                    }
                    let n_ant_state = self.n_anticipated * self.k_max;
                    mask.extend(
                        (start + n_ant_state..start + n_ant_state + self.n_commitment)
                            .map(StateDim::new),
                    );
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
    use super::{CommitmentHoldAddress, InCol, OutCol, StateDim, StateRegion, StateSpace};
    use cobre_core::EntityId;

    /// Build a [`StateSpace`] finalized the way production `resolve_state_layout`
    /// does: full `max_par_order` lag stride for every hydro (the coverage the
    /// dense path emits for test layouts without a PAR model) and the layout's
    /// own `anticipated_lead_stages`.
    fn finalized(
        hydro_count: usize,
        max_par_order: usize,
        n_anticipated: usize,
        k_max: usize,
        anticipated_lead_stages: Vec<usize>,
    ) -> StateSpace {
        finalized_with_transit_buckets(
            hydro_count,
            max_par_order,
            0,
            Vec::new(),
            n_anticipated,
            k_max,
            anticipated_lead_stages,
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
                idx.lp_column_for_state(StateDim::new(j)),
                idx.state_to_lp_column(StateDim::new(j)),
                "finalized map must match the resolver at j={j}"
            );
        }
    }

    /// `StateSpace::new` always finalizes `state_to_lp_column_map` to `n_state`
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
                assert_eq!(
                    idx.lp_column_for_state(StateDim::new(j)),
                    idx.state_to_lp_column_map[j]
                );
            }
        }
    }

    /// Storage-only (`max_par_order == 0`, no anticipated): the mask is exactly
    /// `[0, n_state)` ascending and `lp_column_for_state(j) == j` — the
    /// dense→sparse bit-identity premise for the unified cut-row loop.
    #[test]
    fn lp_column_map_storage_only_mask_is_full_range() {
        let idx = finalized(3, 0, 0, 0, vec![]);

        assert_eq!(
            idx.nonzero_state_indices,
            vec![StateDim::new(0), StateDim::new(1), StateDim::new(2)]
        );
        assert_eq!(idx.nonzero_state_indices.len(), idx.n_state);
        for j in 0..idx.n_state {
            assert_eq!(idx.lp_column_for_state(StateDim::new(j)), OutCol::new(j));
        }
    }

    // ── state_to_lp_column tests ──────────────────────────────────────────────

    /// `commit_out` indices resolve by identity when
    /// `max_par_order == 0 && n_anticipated > 0` — the ring transition
    /// (shift/deposit) is now an in-LP definition row, not a resolver-side
    /// remap. The `commit_out` branch runs before the
    /// `max_par_order == 0` lag-block guard; verify identity holds even when
    /// there are no inflow lags.
    #[test]
    fn state_to_lp_column_commit_out_is_identity_no_lag() {
        // N=1, L=0, n_anticipated=1, k_max=2, anticipated_lead_stages=[2].
        // n_state = 1*(1+0) + 1*2 = 3.
        // commit_out = [1, 3); slot 0 at j=1, slot 1 at j=2.
        let idx = finalized(1, 0, 1, 2, vec![2]);
        assert_eq!(idx.commit_out, 1..3);
        // Storage index: identity.
        assert_eq!(idx.state_to_lp_column(StateDim::new(0)), OutCol::new(0));
        // Both ring slots: identity.
        assert_eq!(idx.state_to_lp_column(StateDim::new(1)), OutCol::new(1));
        assert_eq!(idx.state_to_lp_column(StateDim::new(2)), OutCol::new(2));
    }

    /// `commit_out` indices resolve by identity when
    /// `max_par_order > 0 && n_anticipated > 0`, and the lag-remap branch for
    /// non-anticipated indices is unaffected. Fixture: N=1, L=1,
    /// `n_anticipated=1`, `k_max=2`, `anticipated_lead_stages=[2]`.
    #[test]
    fn state_to_lp_column_commit_out_is_identity_with_lag() {
        // N=1, L=1, n_anticipated=1, k_max=2, anticipated_lead_stages=[2].
        // n_state = 1*(1+1) + 1*2 = 4.
        // Layout: j=0 storage, j=1 lag-0, j=2 ant slot-0, j=3 ant slot-1.
        let idx = finalized(1, 1, 1, 2, vec![2]);
        assert_eq!(idx.commit_out, 2..4);
        // Storage: identity.
        assert_eq!(idx.state_to_lp_column(StateDim::new(0)), OutCol::new(0));
        // Lag block: remapped (unaffected by the anticipated-ring change).
        // j=1: offset=0, h=0, lag=0 → z_inflow.start + 0.
        assert_eq!(
            idx.state_to_lp_column(StateDim::new(1)),
            OutCol::new(idx.z_inflow.start)
        );
        // Both ring slots: identity.
        assert_eq!(idx.state_to_lp_column(StateDim::new(2)), OutCol::new(2));
        assert_eq!(idx.state_to_lp_column(StateDim::new(3)), OutCol::new(3));
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
        assert_eq!(idx.state_to_lp_column(StateDim::new(0)), OutCol::new(0));
        // Lag block j=1: offset=0, h=0, lag=0 → z_inflow.start + 0.
        // For N=1, L=0 anticipated: z_inflow = N*(1+L)..N*(2+L) = 2..3.
        assert_eq!(idx.z_inflow.start, 2);
        assert_eq!(idx.state_to_lp_column(StateDim::new(1)), OutCol::new(2));
    }

    /// Every anticipated ring slot resolves by identity, regardless of which
    /// slot is a plant's own newest (delivery-deposit) slot, a plain-shift
    /// interior slot, or a stage-invariant padding slot beyond that plant's own
    /// `K_p`: the in-LP definition rows (built in `lp/builder`) resolve the
    /// ring transition, so `state_to_lp_column` never special-cases a slot
    /// index the way the deleted constant-lead shift-map did.
    #[test]
    fn state_to_lp_column_commit_out_identity_multi_plant_heterogeneous_k() {
        // Two plants: plant 0 has K_p=1 (only slot 0 is in-use), plant 1 has
        // K_p=3 (slots 0, 1, 2 all in-use). k_max=3 so plant 0 has padding
        // at slots 1 and 2.
        let idx = finalized(0, 0, 2, 3, vec![1, 3]);
        assert_eq!(idx.commit_out, 0..6);
        for j in idx.commit_out.clone() {
            assert_eq!(
                idx.state_to_lp_column(StateDim::new(j)),
                OutCol::new(j),
                "anticipated ring slot {j} must resolve by identity"
            );
        }
    }

    /// The anticipated-ring identity resolution lands inside the **state
    /// region**: `commit_out` sits strictly between
    /// `transit_buckets_out` and `theta`, never inside the control region.
    #[test]
    fn state_to_lp_column_commit_out_resolves_into_state_region() {
        // N=3, L=2, A=2, k_max=3, uniform K_p = 3.
        let idx = finalized(3, 2, 2, 3, vec![3, 3]);
        for j in idx.commit_out.clone() {
            let col = idx.state_to_lp_column(StateDim::new(j)).get();
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
                idx.state_to_lp_incoming_column(StateDim::new(j)),
                InCol::new(idx.storage_in.start + j),
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
                idx.state_to_lp_incoming_column(StateDim::new(j)),
                InCol::new(idx.inflow_lags.start + (j - 3)),
                "j={j}: expected inflow_lags.start + {}",
                j - 3
            );
        }
    }

    /// Commitment-hold range: for a layout with `N=0, L=0, A=1, K=2`,
    /// `state_to_lp_incoming_column(j)` for `j ∈ [0, n_state)` returns
    /// `commit_in.start + j` (since `lag_end` = N*(1+L) = 0). With `N=0`
    /// every non-commitment-hold block collapses to `0..0`, so `commit_in`
    /// (the relocated incoming block, after `commit_out`/`z_inflow`/
    /// `storage_in`/`transit_buckets_in`) starts at `commit_out`'s own
    /// width (`A*K = 2`), not `0`.
    #[test]
    fn state_to_lp_incoming_column_anticipated_range() {
        // N=0, L=0, A=1, K=2: n_state = 0 + 1*2 = 2.
        let idx = finalized(0, 0, 1, 2, vec![2]);
        assert_eq!(idx.commit_in.start, 2);
        assert_eq!(idx.n_state, 2);
        for j in 0..2_usize {
            assert_eq!(
                idx.state_to_lp_incoming_column(StateDim::new(j)),
                InCol::new(idx.commit_in.start + j),
                "j={j}: expected commit_in.start + {j}"
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
        //     (storage + inflow_lags + commit_out + z_inflow).
        //   commit_in.start = storage_in.start + N = 17 (transit_buckets_in
        //     is empty; the relocated incoming block follows storage_in directly).
        //   lag_end = N*(1+L) = 9.
        let idx = finalized(3, 2, 1, 2, vec![2]);
        assert_eq!(idx.n_state, 11);
        // j=0: storage range → storage_in.start + 0.
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(0)),
            InCol::new(idx.storage_in.start),
            "j=0"
        );
        // j=2: storage range → storage_in.start + 2.
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(2)),
            InCol::new(idx.storage_in.start + 2),
            "j=2"
        );
        // j=3: first lag → inflow_lags.start + 0.
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(3)),
            InCol::new(idx.inflow_lags.start),
            "j=3"
        );
        // j=8: last lag → inflow_lags.start + 5.
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(8)),
            InCol::new(idx.inflow_lags.start + 5),
            "j=8"
        );
        // j=9: first anticipated-state → commit_in.start + 0.
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(9)),
            InCol::new(idx.commit_in.start),
            "j=9"
        );
        // j=10: last anticipated-state → commit_in.start + 1.
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(10)),
            InCol::new(idx.commit_in.start + 1),
            "j=10"
        );
        // All returned columns must be within the LP's column range.
        for j in 0..idx.n_state {
            let col = idx.state_to_lp_incoming_column(StateDim::new(j)).get();
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
            idx.state_to_lp_incoming_column(StateDim::new(0)).get(),
            idx.state_to_lp_column(StateDim::new(0)).get(),
            "storage range should differ: incoming={} outgoing={}",
            idx.state_to_lp_incoming_column(StateDim::new(0)).get(),
            idx.state_to_lp_column(StateDim::new(0)).get()
        );
        // j=0: incoming returns storage_in.start, outgoing returns 0.
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(0)),
            InCol::new(idx.storage_in.start)
        );
        assert_eq!(idx.state_to_lp_column(StateDim::new(0)), OutCol::new(0));

        // Lag range (j=2, j=3): incoming returns inflow_lags column;
        // outgoing returns z_inflow (lag=0) or lag-out (lag>=1) column.
        // j=2: lag 0, hydro 0. incoming = inflow_lags.start + 0.
        //                       outgoing = z_inflow.start + 0.
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(2)),
            InCol::new(idx.inflow_lags.start),
            "j=2 incoming should be inflow_lags.start"
        );
        assert_eq!(
            idx.state_to_lp_column(StateDim::new(2)),
            OutCol::new(idx.z_inflow.start),
            "j=2 outgoing should be z_inflow.start"
        );
        assert_ne!(
            idx.state_to_lp_incoming_column(StateDim::new(2)).get(),
            idx.state_to_lp_column(StateDim::new(2)).get(),
            "lag range should differ for j=2"
        );
        // j=3: lag 0, hydro 1. incoming = inflow_lags.start + 1.
        //                       outgoing = z_inflow.start + 1.
        assert_ne!(
            idx.state_to_lp_incoming_column(StateDim::new(3)).get(),
            idx.state_to_lp_column(StateDim::new(3)).get(),
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

        assert_eq!(
            &idx.nonzero_state_indices[..4],
            &[0, 1, 2, 3].map(StateDim::new)
        );
        assert_eq!(
            &idx.nonzero_state_indices[4..],
            &[5, 6, 7, 10, 11, 14, 15, 19, 23, 27].map(StateDim::new)
        );

        assert!(idx.nonzero_state_indices.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn nonzero_mask_zero_par_order() {
        // max_par_order=0: no lags, mask = storage only
        let mut idx = finalized(3, 0, 0, 0, vec![]);
        idx.set_nonzero_mask(&[0, 0, 0], &[]);
        assert_eq!(idx.nonzero_state_indices.len(), 3);
        assert_eq!(&idx.nonzero_state_indices, &[0, 1, 2].map(StateDim::new));
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
        assert_eq!(&idx.nonzero_state_indices[..2], &[0, 1].map(StateDim::new));

        // Hydro 0 (lag_count = 4): expect lag slots at indices
        //   inflow_lags.start + lag * hydro_count + h = 2 + lag*2 + 0 for lag in 0..4
        // → {2, 4, 6, 8}.
        // Hydro 1 (lag_count = 12): expect lag slots at indices
        //   2 + lag*2 + 1 for lag in 0..12 → {3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25}.
        // Mask is sorted globally. Confirm a few discriminating positions.
        assert!(
            idx.nonzero_state_indices.contains(&StateDim::new(25)),
            "lag-11 slot for hydro 1 (the trailing PAR-A annual slot) must be in the mask"
        );
        assert!(
            !idx.nonzero_state_indices.contains(&StateDim::new(10)),
            "lag-4 slot for hydro 0 (classical AR(4)) must NOT be in the mask"
        );
        // Sorted.
        assert!(idx.nonzero_state_indices.windows(2).all(|w| w[0] < w[1]));
    }

    // ── Commitment-hold in-study nonzero mask tests ────────────────────────

    /// Every anticipated plant uses every slot (`K_i == k_max`): all
    /// `n_anticipated * k_max` anticipated indices are included in the mask.
    #[test]
    fn nonzero_mask_commitment_hold_in_study_full_kmax() {
        // 2 anticipated plants, k_max = 3, no hydros, no lags.
        // commit_out.start = 0 (no storage, no lag block).
        // Layout: start + slot * n_anticipated + plant.
        let mut idx = finalized(0, 0, 2, 3, vec![3, 3]);

        assert_eq!(idx.commit_out.start, 0);
        assert_eq!(idx.n_anticipated, 2);
        assert_eq!(idx.k_max, 3);

        idx.set_nonzero_mask(&[], &[3, 3]);

        // Every slot is occupied, so all 6 indices appear.
        // slot=0: 0, 1; slot=1: 2, 3; slot=2: 4, 5.
        assert_eq!(
            idx.nonzero_state_indices,
            [0, 1, 2, 3, 4, 5].map(StateDim::new)
        );
    }

    /// `K_i < k_max` for some plants: padded slots are excluded.
    /// Configuration: `n_anticipated = 2`, `k_max = 3`,
    /// `anticipated_lead_stages = [3, 1]`.
    #[test]
    fn nonzero_mask_commitment_hold_in_study_partial_padding() {
        // 3 hydros, max_par_order = 2, 2 anticipated plants, k_max = 3.
        // inflow_lags = [3, 9), commit_out.start = 9.
        let mut idx = finalized(3, 2, 2, 3, vec![3, 1]);

        assert_eq!(idx.commit_out.start, 9);
        idx.set_nonzero_mask(&[2, 2, 2], &[3, 1]);

        // Storage [0, 1, 2] + lag (h0,h1,h2 full = 6 slots) +
        // anticipated: slot=0 plant=0 → 9, plant=1 → 10 (K_1=1 so slot 0 included);
        //              slot=1 plant=0 → 11 (K_0=3); plant=1 → padded (slot 1 >= 1).
        //              slot=2 plant=0 → 13 (K_0=3); plant=1 → padded.
        // The anticipated portion expected: [9, 10, 11, 13].
        let mask = &idx.nonzero_state_indices;
        let ant_portion: Vec<usize> = mask.iter().map(|d| d.get()).filter(|&i| i >= 9).collect();
        assert_eq!(ant_portion, vec![9, 10, 11, 13]);
        // Padded slots NOT present.
        assert!(!mask.contains(&StateDim::new(12)));
        assert!(!mask.contains(&StateDim::new(14)));
    }

    /// Anticipated-only: no hydros, no lags, only anticipated state.
    #[test]
    fn nonzero_mask_commitment_hold_in_study_only_no_hydros() {
        let mut idx = finalized(0, 0, 1, 2, vec![2]);

        assert_eq!(idx.hydro_count, 0);
        assert_eq!(idx.commit_out.start, 0);

        idx.set_nonzero_mask(&[], &[2]);

        // Only anticipated indices: slot=0 plant=0 → 0; slot=1 plant=0 → 1.
        assert_eq!(idx.nonzero_state_indices, [0, 1].map(StateDim::new));
    }

    /// Heterogeneous `K_i` across plants, including a plant with `K_i = k_max`
    /// and another with `K_i < k_max`.
    #[test]
    fn nonzero_mask_commitment_hold_in_study_mixed_k_values() {
        // n_anticipated = 3, k_max = 4. Lead stages = [4, 2, 1].
        // commit_out.start = 0 (no hydros).
        let mut idx = finalized(0, 0, 3, 4, vec![4, 2, 1]);

        idx.set_nonzero_mask(&[], &[4, 2, 1]);

        // slot=0 plant=0→0, plant=1→1, plant=2→2 (all K_i > 0).
        // slot=1 plant=0→3, plant=1→4 (K_1=2). plant=2 padded.
        // slot=2 plant=0→6 (K_0=4). plant=1 padded. plant=2 padded.
        // slot=3 plant=0→9 (K_0=4). Others padded.
        // Expected: [0, 1, 2, 3, 4, 6, 9].
        assert_eq!(
            idx.nonzero_state_indices,
            [0, 1, 2, 3, 4, 6, 9].map(StateDim::new)
        );
    }

    /// `n_anticipated == 0` reproduces the pre-anticipated behaviour exactly.
    #[test]
    fn nonzero_mask_commitment_hold_in_study_zero_anticipated_matches_existing() {
        let mut idx_with = finalized(4, 6, 0, 0, vec![]);
        idx_with.set_nonzero_mask(&[0, 1, 3, 6], &[]);

        // Same expected mask as `nonzero_mask_mixed_ar_orders`:
        // [0,1,2,3] (storage) + [5,6,7,10,11,14,15,19,23,27] (lags).
        assert_eq!(
            idx_with.nonzero_state_indices,
            [0, 1, 2, 3, 5, 6, 7, 10, 11, 14, 15, 19, 23, 27].map(StateDim::new)
        );
    }

    /// The extended mask is sorted ascending with no duplicates.
    #[test]
    fn nonzero_mask_commitment_hold_in_study_sorted_ascending() {
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
    fn nonzero_mask_commitment_hold_in_study_boundary_k_eq_kmax() {
        // 1 anticipated plant, K_0 = k_max = 3, no hydros.
        let mut idx = finalized(0, 0, 1, 3, vec![3]);

        idx.set_nonzero_mask(&[], &[3]);

        // All k_max slots included: slot 0,1,2 → indices 0,1,2.
        assert_eq!(idx.nonzero_state_indices, [0, 1, 2].map(StateDim::new));
    }

    /// `K_i == 0` excludes all slots for that plant (defensive — the parse
    /// layer rejects `K_i == 0`, but the helper must remain robust if
    /// invoked with zero).
    #[test]
    fn nonzero_mask_commitment_hold_in_study_boundary_k_zero_excluded() {
        // 2 anticipated plants, k_max = 2. Lead stages = [2, 0].
        let mut idx = finalized(0, 0, 2, 2, vec![2, 0]);

        idx.set_nonzero_mask(&[], &[2, 0]);

        // Plant 0 (K_0=2) emits slot=0→0, slot=1→2. Plant 1 (K_1=0) emits
        // nothing. Expected mask: [0, 2].
        assert_eq!(idx.nonzero_state_indices, [0, 2].map(StateDim::new));
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
                idx.state_to_lp_column(StateDim::new(j)),
                OutCol::new(j),
                "bucket state index {j} must map to its LP column by identity"
            );
        }
        // The last lag index (j=5, just below the bucket block) still resolves
        // via the lag remap, proving the bucket check does not swallow lag
        // indices.
        assert_eq!(idx.state_to_lp_column(StateDim::new(5)), OutCol::new(3));
    }

    /// Bucket-arm resolution for `state_to_lp_incoming_column`: bucket indices
    /// resolve to the pinned `transit_buckets_in` column via an explicit arm, not the
    /// anticipated arm — verified with anticipated state present so both arms
    /// are live and either could otherwise swallow the other's indices.
    #[test]
    fn state_to_lp_incoming_column_transit_bucket_arm_is_pinned_not_anticipated() {
        // N=2, L=1, B=2, A=1 (k_max=2, K=[2]).
        let idx = finalized_with_transit_buckets(2, 1, 2, vec![(0, 1), (0, 2)], 1, 2, vec![2]);

        assert_eq!(idx.transit_buckets_in, 12..14);
        assert_eq!(idx.commit_in.start, 14);

        // Bucket state indices: j=4, j=5 (lag_end = N*(1+L) = 4).
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(4)),
            InCol::new(idx.transit_buckets_in.start)
        );
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(5)),
            InCol::new(idx.transit_buckets_in.start + 1)
        );
        assert_ne!(
            idx.state_to_lp_incoming_column(StateDim::new(4)),
            InCol::new(idx.commit_in.start),
            "bucket index must not resolve to the anticipated catch-all"
        );

        // The first anticipated-state index (j=6) still resolves via its own
        // arm, immediately after the bucket range.
        assert_eq!(
            idx.state_to_lp_incoming_column(StateDim::new(6)),
            InCol::new(idx.commit_in.start)
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
            assert_eq!(
                idx.lp_column_for_state(StateDim::new(j)),
                OutCol::new(j),
                "bucket-only identity map"
            );
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
            [0, 1, 2, 3, 5, 6, 7].map(StateDim::new),
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
        assert_eq!(idx.commit_out, 9..13);
        assert_eq!(idx.z_inflow, 13..16);
        assert_eq!(idx.storage_in, 16..19);
        assert_eq!(idx.commit_in, 19..23);
        assert_eq!(idx.theta, 23);
        assert_eq!(idx.n_state, 13);
    }

    /// `A * k_max == 0` (no anticipated thermals) collapses `commit_out`/
    /// `commit_in` to `0..0` and reproduces the pre-anticipated-ring layout
    /// byte-for-byte (`N=3, L=2, B=2`).
    #[test]
    fn state_layout_a_zero_collapses_to_pre_anticipated_ring_layout() {
        let idx = finalized_with_transit_buckets(3, 2, 2, vec![(0, 1), (0, 2)], 0, 0, vec![]);

        assert_eq!(idx.n_anticipated, 0);
        assert_eq!(idx.k_max, 0);
        assert_eq!(idx.commit_out, 0..0);
        assert_eq!(idx.commit_in, 0..0);

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
                idx.lp_column_for_state(StateDim::new(j)),
                idx.state_to_lp_column(StateDim::new(j)),
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
        let zero_k_max = StateSpace::new(3, 2, 0, Vec::new(), 2, 0, vec![0, 0], &[2, 2, 2]);
        let no_plants = StateSpace::new(3, 2, 0, Vec::new(), 0, 0, vec![], &[2, 2, 2]);

        assert_eq!(zero_k_max.commit_out, 0..0);
        assert_eq!(zero_k_max.commit_in, 0..0);
        assert_eq!(zero_k_max.theta, no_plants.theta);
        assert_eq!(zero_k_max.n_state, no_plants.n_state);
        assert_eq!(
            zero_k_max.state_to_lp_column_map,
            no_plants.state_to_lp_column_map
        );
    }

    // ── Terminal commitment-hold (post-horizon lanes) ───────────────────────

    /// Sizing and column layout: `commit_out`/`commit_in` each span the
    /// merged in-study + post-horizon width (`n_anticipated*k_max +
    /// n_commitment`), inserted between the travel-time buckets and
    /// `z_inflow`/`theta` respectively — hand-computed against `N=3, L=2,
    /// B=0, A=2, k_max=2, W=2`.
    #[test]
    fn commitment_hold_sizing_and_column_layout() {
        let idx = StateSpace::new_with_commitment_hold_windows(
            3,
            2,
            0,
            Vec::new(),
            2,
            2,
            vec![1, 2],
            &[2, 2, 2],
            2,
            vec![0, 1],
            vec![EntityId(0), EntityId(0)],
        );

        assert_eq!(idx.n_commitment, 2);
        assert_eq!(idx.commitment_decider_stage, vec![0, 1]);
        assert_eq!(idx.storage, 0..3);
        assert_eq!(idx.inflow_lags, 3..9);
        assert_eq!(idx.commit_out, 9..15);
        assert_eq!(idx.z_inflow, 15..18);
        assert_eq!(idx.storage_in, 18..21);
        assert_eq!(idx.commit_in, 21..27);
        assert_eq!(idx.theta, 27);
        assert_eq!(idx.n_state, 15);
    }

    /// `state_to_lp_column` resolves every commitment-hold post-horizon lane
    /// by identity, mirroring storage/buckets/the in-study slots.
    #[test]
    fn commitment_hold_post_horizon_state_to_lp_column_is_identity() {
        let idx = StateSpace::new_with_commitment_hold_windows(
            3,
            2,
            0,
            Vec::new(),
            2,
            2,
            vec![1, 2],
            &[2, 2, 2],
            2,
            vec![0, 1],
            vec![EntityId(0), EntityId(0)],
        );
        for window in 0..idx.n_commitment {
            let j = idx.commit_out.start + idx.commitment_hold_post_horizon_offset(window);
            assert_eq!(
                idx.state_to_lp_column(StateDim::new(j)),
                OutCol::new(j),
                "commitment-hold post-horizon lane {j} must resolve by identity"
            );
            let (region, _) = idx.classify(StateDim::new(j));
            assert!(matches!(region, StateRegion::CommitmentHold));
        }
    }

    /// `classify_incoming_column` inverts `state_to_lp_incoming_column` for
    /// every commitment-hold post-horizon state dimension — the incoming
    /// (pinned) side of the classification the outgoing test above pins.
    #[test]
    fn commitment_hold_post_horizon_classify_incoming_column_inverts() {
        let idx = StateSpace::new_with_commitment_hold_windows(
            3,
            2,
            0,
            Vec::new(),
            2,
            2,
            vec![1, 2],
            &[2, 2, 2],
            2,
            vec![0, 1],
            vec![EntityId(0), EntityId(0)],
        );
        for window in 0..idx.n_commitment {
            let offset = idx.commitment_hold_post_horizon_offset(window);
            let j = idx.state_dim_commitment_hold_range().start + offset;
            let dim = StateDim::new(j);
            let (region, region_offset) =
                idx.classify_incoming_column(idx.state_to_lp_incoming_column(dim));
            assert!(matches!(region, StateRegion::CommitmentHold));
            assert_eq!(region_offset, offset);
            assert_eq!(
                idx.commitment_hold_address(region_offset),
                CommitmentHoldAddress::PostHorizon { window }
            );
        }
    }

    /// Every commitment-hold post-horizon lane enters `nonzero_state_indices`
    /// — unlike the in-study slots there is no padding concept (a window
    /// either resolves to a lane or is dropped upstream).
    #[test]
    fn commitment_hold_nonzero_mask_includes_every_window() {
        let idx = StateSpace::new_with_commitment_hold_windows(
            0,
            0,
            0,
            Vec::new(),
            0,
            0,
            vec![],
            &[],
            3,
            vec![0, 1, 2],
            vec![EntityId(0), EntityId(0), EntityId(0)],
        );
        assert_eq!(idx.commit_out, 0..3);
        assert_eq!(idx.nonzero_state_indices, [0, 1, 2].map(StateDim::new));
    }

    /// `n_commitment == 0` reproduces [`StateSpace::new`]'s layout byte-for-byte
    /// — the additive & inert contract: with no declared post-horizon windows,
    /// every existing case stays untouched.
    #[test]
    fn commitment_hold_zero_windows_is_byte_identical_to_pre_commitment_layout() {
        let with_zero_commitment = StateSpace::new_with_commitment_hold_windows(
            3,
            2,
            0,
            Vec::new(),
            2,
            2,
            vec![1, 2],
            &[2, 2, 2],
            0,
            Vec::new(),
            Vec::new(),
        );
        let baseline = StateSpace::new(3, 2, 0, Vec::new(), 2, 2, vec![1, 2], &[2, 2, 2]);

        assert_eq!(with_zero_commitment.n_commitment, 0);
        assert!(with_zero_commitment.commitment_decider_stage.is_empty());
        assert_eq!(with_zero_commitment.storage, baseline.storage);
        assert_eq!(with_zero_commitment.inflow_lags, baseline.inflow_lags);
        assert_eq!(
            with_zero_commitment.transit_buckets_out,
            baseline.transit_buckets_out
        );
        assert_eq!(with_zero_commitment.commit_out, baseline.commit_out);
        assert_eq!(with_zero_commitment.z_inflow, baseline.z_inflow);
        assert_eq!(with_zero_commitment.storage_in, baseline.storage_in);
        assert_eq!(
            with_zero_commitment.transit_buckets_in,
            baseline.transit_buckets_in
        );
        assert_eq!(with_zero_commitment.commit_in, baseline.commit_in);
        assert_eq!(with_zero_commitment.theta, baseline.theta);
        assert_eq!(with_zero_commitment.n_state, baseline.n_state);
        assert_eq!(
            with_zero_commitment.state_to_lp_column_map,
            baseline.state_to_lp_column_map
        );
        assert_eq!(
            with_zero_commitment.nonzero_state_indices,
            baseline.nonzero_state_indices
        );
    }

    // ── CommitmentHold addressing helper ────────────────────────────────────

    /// `commit_out`/`commit_in` always span exactly `n_anticipated*k_max +
    /// n_commitment` with equal width — the invariant
    /// [`StateSpace::new_with_commitment_hold_windows`]'s `debug_assert` pins,
    /// checked here across in-study-only, post-horizon-only, both, and
    /// neither.
    #[test]
    fn commit_out_in_width_matches_in_study_plus_post_horizon() {
        for idx in [
            StateSpace::new_with_commitment_hold_windows(
                3,
                2,
                0,
                Vec::new(),
                2,
                2,
                vec![1, 2],
                &[2, 2, 2],
                2,
                vec![0, 1],
                vec![EntityId(0), EntityId(0)],
            ),
            StateSpace::new_with_commitment_hold_windows(
                3,
                2,
                0,
                Vec::new(),
                0,
                0,
                vec![],
                &[2, 2, 2],
                2,
                vec![0, 1],
                vec![EntityId(0), EntityId(0)],
            ),
            StateSpace::new_with_commitment_hold_windows(
                3,
                2,
                0,
                Vec::new(),
                2,
                2,
                vec![1, 2],
                &[2, 2, 2],
                0,
                Vec::new(),
                Vec::new(),
            ),
            StateSpace::new(3, 2, 0, Vec::new(), 0, 0, vec![], &[2, 2, 2]),
        ] {
            let expected = idx.n_anticipated * idx.k_max + idx.n_commitment;
            assert_eq!(idx.commit_out.len(), expected);
            assert_eq!(idx.commit_in.len(), expected);
        }
    }

    /// `commitment_hold_in_study_offset` is delivery-target-modular: delivery
    /// targets congruent modulo `k_max` share the same plant's slot, and one
    /// full period of `m` bijects onto every `(plant, slot)` pair in the
    /// leading in-study block exactly once.
    #[test]
    fn commitment_hold_in_study_offset_is_delivery_target_modular_bijection() {
        let idx = finalized(0, 0, 3, 4, vec![4, 4, 4]);

        for plant in 0..idx.n_anticipated {
            for m in 0..idx.k_max {
                assert_eq!(
                    idx.commitment_hold_in_study_offset(plant, m),
                    idx.commitment_hold_in_study_offset(plant, m + idx.k_max * 3),
                    "delivery targets congruent mod k_max must share plant {plant}'s slot"
                );
            }
        }

        let mut offsets: Vec<usize> = Vec::new();
        for m in 0..idx.k_max {
            for plant in 0..idx.n_anticipated {
                offsets.push(idx.commitment_hold_in_study_offset(plant, m));
            }
        }
        offsets.sort_unstable();
        let expected: Vec<usize> = (0..idx.n_anticipated * idx.k_max).collect();
        assert_eq!(
            offsets, expected,
            "one period of m must biject onto the full leading in-study block"
        );
    }

    /// `commitment_hold_address` inverts the two encoders for every position
    /// in the merged region — in-study offsets decode back to their `(plant,
    /// slot)` pair, post-horizon offsets decode back to their `window`.
    #[test]
    fn commitment_hold_address_round_trips_encoders() {
        let idx = StateSpace::new_with_commitment_hold_windows(
            0,
            0,
            0,
            Vec::new(),
            2,
            3,
            vec![3, 3],
            &[],
            2,
            vec![0, 1],
            vec![EntityId(0), EntityId(0)],
        );

        for plant in 0..idx.n_anticipated {
            for slot in 0..idx.k_max {
                let offset = idx.commitment_hold_in_study_offset(plant, slot);
                assert_eq!(
                    idx.commitment_hold_address(offset),
                    CommitmentHoldAddress::InStudy { plant, slot }
                );
            }
        }
        for window in 0..idx.n_commitment {
            let offset = idx.commitment_hold_post_horizon_offset(window);
            assert_eq!(
                idx.commitment_hold_address(offset),
                CommitmentHoldAddress::PostHorizon { window }
            );
        }
    }

    /// `classify_incoming_column` composed with `commitment_hold_address`
    /// round-trips every commitment-hold state dimension back to its
    /// structured address, for a layout exercising the in-study and
    /// post-horizon sub-ranges together.
    #[test]
    fn commitment_hold_classify_incoming_column_round_trips_address() {
        let idx = StateSpace::new_with_commitment_hold_windows(
            0,
            0,
            0,
            Vec::new(),
            2,
            3,
            vec![3, 3],
            &[],
            2,
            vec![0, 1],
            vec![EntityId(0), EntityId(0)],
        );

        for j in idx.state_dim_commitment_hold_range() {
            let dim = StateDim::new(j);
            let (region, offset) =
                idx.classify_incoming_column(idx.state_to_lp_incoming_column(dim));
            assert!(matches!(region, StateRegion::CommitmentHold));
            let expected_offset = j - idx.state_dim_commitment_hold_range().start;
            assert_eq!(offset, expected_offset);
            match idx.commitment_hold_address(offset) {
                CommitmentHoldAddress::InStudy { plant, slot } => {
                    assert_eq!(offset, slot * idx.n_anticipated + plant);
                }
                CommitmentHoldAddress::PostHorizon { window } => {
                    assert_eq!(offset, idx.n_anticipated * idx.k_max + window);
                }
            }
        }
    }

    // ── State-dimension region accessors ────────────────────────────────────

    /// The four `state_dim_*_range` accessors partition `[0, n_state)`
    /// contiguously with no gap and no overlap — storage, lags, buckets, and
    /// the merged commitment-hold region all present — the direct pin for the
    /// region order `CutStateProjection::new` consumes.
    #[test]
    fn state_dim_ranges_partition_n_state_contiguously() {
        // N=3, L=2, B=2, A=2, k_max=2: every region non-empty.
        let idx = finalized_with_transit_buckets(3, 2, 2, vec![(0, 1), (0, 2)], 2, 2, vec![1, 2]);

        let storage = idx.state_dim_storage_range();
        let lag = idx.state_dim_lag_range();
        let bucket = idx.state_dim_bucket_range();
        let commitment_hold = idx.state_dim_commitment_hold_range();

        assert!(
            !storage.is_empty()
                && !lag.is_empty()
                && !bucket.is_empty()
                && !commitment_hold.is_empty(),
            "fixture must exercise every region non-empty"
        );

        assert_eq!(storage.start, 0, "storage must start at the state origin");
        assert_eq!(
            lag.start, storage.end,
            "lag region must start exactly where storage ends (no gap/overlap)"
        );
        assert_eq!(
            bucket.start, lag.end,
            "bucket region must start exactly where lag ends (no gap/overlap)"
        );
        assert_eq!(
            commitment_hold.start, bucket.end,
            "commitment-hold region must start exactly where bucket ends (no gap/overlap)"
        );
        assert_eq!(
            commitment_hold.end, idx.n_state,
            "commitment-hold region must end exactly at n_state (no trailing gap)"
        );
    }

    /// [`StateSpace::classify_incoming_column`] must invert
    /// [`StateSpace::state_to_lp_incoming_column`] for every state dimension,
    /// with every region non-empty.
    #[test]
    fn classify_incoming_column_inverts_incoming_resolver() {
        let idx = finalized_with_transit_buckets(3, 2, 2, vec![(0, 1), (0, 2)], 2, 2, vec![1, 2]);
        for j in 0..idx.n_state {
            let dim = StateDim::new(j);
            let (region, offset) =
                idx.classify_incoming_column(idx.state_to_lp_incoming_column(dim));
            assert_eq!(
                idx.state_dim_range(region).start + offset,
                j,
                "incoming classification must round-trip state dim {j}"
            );
        }
    }

    /// The purpose-named column accessors resolve to the same columns as the
    /// raw range arithmetic they replaced at the extraction and manifest
    /// seams — the byte-neutrality pin for that migration.
    #[test]
    fn typed_state_col_accessors_match_block_layout() {
        let idx = finalized_with_transit_buckets(3, 2, 2, vec![(0, 1), (0, 2)], 2, 2, vec![1, 2]);
        for h in 0..idx.hydro_count {
            assert_eq!(idx.storage_incoming_col(h).get(), idx.storage_in.start + h);
            assert_eq!(idx.storage_outgoing_col(h).get(), idx.storage.start + h);
            for lag in 0..idx.max_par_order {
                assert_eq!(
                    idx.lag_incoming_col(lag, h).get(),
                    idx.inflow_lags.start + lag * idx.hydro_count + h
                );
            }
        }
        for b in 0..idx.n_buckets {
            assert_eq!(
                idx.bucket_incoming_col(b).get(),
                idx.transit_buckets_in.start + b
            );
            assert_eq!(
                idx.bucket_outgoing_col(b).get(),
                idx.transit_buckets_out.start + b
            );
        }
    }
}
