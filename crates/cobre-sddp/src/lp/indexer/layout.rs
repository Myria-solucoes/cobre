//! The [`StageIndexer`] struct, its satellite layout types, the small layout
//! accessors, and the compile-time `Send + Sync` assertion.
//!
//! `StageIndexer` carries the state-pinning contract codified in
//! `.claude/rules/sddp.md`: the `storage_fixing`, `lag_fixing`, and
//! `anticipated_state_fixing` row ranges are permanent empty `0..0` sentinels —
//! state is pinned via column bounds resolved through
//! [`StageIndexer::state_to_lp_incoming_column`], never a fixing-row index. The
//! `0..0` sentinel field docs below are part of that contract and move verbatim.

use std::collections::HashMap;
use std::ops::Range;

/// Column and row indices for the evaporation constraint of one hydro.
///
/// Locates the three evaporation columns and one evaporation row assigned to
/// a single hydro within a stage LP.  Columns are stage-level (not per-block).
#[derive(Debug, Clone, Copy)]
pub struct EvaporationIndices {
    /// Column index of the stage-averaged evaporation-outflow variable (m³/s).
    pub evaporation_flow_col: usize,
    /// Column index of the positive violation slack `f_evap_plus_h` (m³/s).
    pub f_evap_plus_col: usize,
    /// Column index of the negative violation slack `f_evap_minus_h` (m³/s).
    pub f_evap_minus_col: usize,
    /// Row index of the evaporation equality constraint.
    pub evap_row: usize,
}

/// FPHA constraint row range for one hydro at one stage.
///
/// Locates the block of FPHA hyperplane rows assigned to a single FPHA hydro
/// within a stage LP. Rows for hydro `i` at block `k` and plane `p` are at:
/// `start + k * planes_per_block + p`.
#[derive(Debug, Clone, Copy)]
pub struct FphaRowRange {
    /// First row index of this hydro's FPHA constraints (for block 0, plane 0).
    pub start: usize,
    /// Number of hyperplanes per block.
    pub planes_per_block: usize,
}

/// Read-only LP layout index map for one SDDP stage subproblem.
///
/// Computed once from `hydro_count` (N) and `max_par_order` (L), then shared
/// read-only across all threads for the duration of training. Most fields are
/// plain `usize` or `Range<usize>`; FPHA fields use `Vec` for variable-length
/// hydro lists.
///
/// See the [module-level documentation](super) for the full column and row
/// layout, and [`StageIndexer::new`] for the construction formulas.
///
/// Equipment column ranges (`turbine`, `spillage`, `diversion`, `thermal`,
/// `line_fwd`, `line_rev`, `deficit`, `excess`) are populated only when constructed via
/// [`StageIndexer::with_equipment`]. When constructed via [`StageIndexer::new`]
/// or [`StageIndexer::from_stage_template`], those ranges are all empty (`0..0`)
/// and `n_blks`, `n_thermals`, `n_lines`, `n_buses` are zero.
///
/// FPHA fields (`generation`, `fpha_hydro_indices`, `fpha_rows`) are also
/// populated only by [`StageIndexer::with_equipment`] when FPHA hydros are
/// present. They are empty when built via [`StageIndexer::new`] or when no FPHA
/// hydros exist.
#[derive(Debug, Clone)]
pub struct StageIndexer {
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

    /// Column range `[N*(2+L), N*(3+L))` for incoming storage volumes.
    ///
    /// Pinned to the preceding stage's outgoing `storage` solution values via
    /// `set_col_bounds` on these columns — not via equality rows (the
    /// `storage_fixing` row range is a permanent empty sentinel). Resolve the
    /// column with [`StageIndexer::state_to_lp_incoming_column`].
    pub storage_in: Range<usize>,

    /// Column index `N*(3+L)` for the future cost variable (theta).
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
    /// anticipated thermals are present). State transfer copies
    /// `primal[0..n_transfer]` (all but the oldest lag row).
    ///
    /// ## Semantic distinction
    ///
    /// `n_state` is the state-vector **dimension** used by cut storage and
    /// broadcast payloads. It is **not** a valid LP row index. Do not slice
    /// the LP row buffer as `[0, n_state)` — no state-fixing rows exist.
    /// Use [`StageIndexer::state_to_lp_incoming_column`] to resolve the
    /// column index for state-pinning and cut-subgradient extraction.
    pub n_state: usize,

    /// Row range for storage-fixing constraints.
    ///
    /// Always empty (`0..0`): state pinning uses column bounds
    /// (`set_col_bounds` on the incoming-state columns) rather than equality
    /// rows. The field is retained as a permanent sentinel for API stability;
    /// downstream consumers will observe an empty range.
    /// Use [`StageIndexer::state_to_lp_incoming_column`] to resolve the
    /// column index for state pinning and cut-subgradient extraction.
    pub storage_fixing: Range<usize>,

    /// Row range for AR lag-fixing constraints.
    ///
    /// Always empty (`0..0`): state pinning uses column bounds rather than
    /// equality rows. Retained as a permanent sentinel for the same reason
    /// as `storage_fixing`.
    pub lag_fixing: Range<usize>,

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
    /// Empty (`0..0`) when `n_anticipated == 0` or when built via
    /// [`StageIndexer::new`].
    pub anticipated_state: Range<usize>,

    /// Row range for anticipated-state-fixing constraints.
    ///
    /// Always empty (`0..0`) regardless of `n_anticipated`: state pinning for
    /// anticipated thermals also uses column bounds. Retained as a permanent
    /// sentinel for API stability. The column range [`Self::anticipated_state`]
    /// carries the state-pinning semantics via `set_col_bounds`.
    pub anticipated_state_fixing: Range<usize>,

    /// Number of anticipated thermals (plants with
    /// `anticipated_config.is_some()`).
    ///
    /// Zero when no anticipated plants exist or when built via
    /// [`StageIndexer::new`].
    pub n_anticipated: usize,

    /// Maximum `lead_stages` across the anticipated thermals (`K_max`).
    ///
    /// Zero when `n_anticipated == 0` or when built via
    /// [`StageIndexer::new`].
    pub k_max: usize,

    /// Number of operating hydro plants (N).
    pub hydro_count: usize,

    /// Maximum PAR order across all operating hydros (L).
    ///
    /// All hydros use a uniform lag stride of `max_par_order`, enabling
    /// contiguous memory access and SIMD vectorisation over the lag dimension.
    pub max_par_order: usize,

    // ── Equipment column ranges ────────────────────────────────────────────
    // Populated only by `with_equipment`; empty (`0..0`) when built via `new`.
    /// Column range for turbined flow variables, one per (hydro, block) pair.
    ///
    /// Index for hydro `h`, block `b`: `turbine.start + h * n_blks + b`.
    /// Empty when built via [`StageIndexer::new`].
    pub turbine: Range<usize>,

    /// Column range for spillage variables, one per (hydro, block) pair.
    ///
    /// Index for hydro `h`, block `b`: `spillage.start + h * n_blks + b`.
    /// Empty when built via [`StageIndexer::new`].
    pub spillage: Range<usize>,

    /// Column range for diversion flow variables, one per (hydro, block) pair.
    ///
    /// Index for hydro `h`, block `b`: `diversion.start + h * n_blks + b`.
    /// Hydros without a diversion channel have bounds [0, 0]; the LP presolve
    /// eliminates them.
    /// Empty when built via [`StageIndexer::new`].
    pub diversion: Range<usize>,

    /// Column range for thermal generation variables, one per (thermal, block) pair.
    ///
    /// Index for thermal `t`, block `b`: `thermal.start + t * n_blks + b`.
    /// Empty when built via [`StageIndexer::new`].
    pub thermal: Range<usize>,

    /// Column range for anticipated-thermal commitment decisions, one stage-level
    /// column per anticipated plant active at stage 0.
    ///
    /// Decision variables are stage-level (NOT per-block): each column represents
    /// the MW the dispatcher commits for that plant at the current stage. The
    /// commitment is delivered `K_i` stages later via the fishing constraint.
    ///
    /// The `cobre-io` semantic validator enforces `K_i <= T` for every
    /// anticipated plant, so at stage 0 every anticipated plant is active and
    /// the range length equals [`Self::n_anticipated`].
    ///
    /// Per-stage gating is computed by
    /// [`anticipated_decision_active_at_stage`](Self::anticipated_decision_active_at_stage):
    /// at stage `t`, plant `i` is active iff `t + K_i <= T`. Inactive columns
    /// receive bounds `[0, 0]` in the LP build, matching the deficit-segment
    /// pattern.
    ///
    /// Empty (`0..0`) when `n_anticipated == 0` or when built via
    /// [`StageIndexer::new`].
    pub anticipated_decision: Range<usize>,

    /// Column range for the anticipated-thermal outgoing-state variables,
    /// one column per anticipated plant (stage-level, NOT per-block).
    ///
    /// Length: `n_anticipated`. Placed immediately after
    /// [`Self::anticipated_decision`] in the control region so that
    /// `anticipated_state_out.start == anticipated_decision.end`. Together
    /// with the `anticipated_state_out` definition row, this variable is
    /// pinned to the corresponding `anticipated_decision` column by an
    /// equality constraint, making it the correct target for cut-coefficient
    /// mapping via `state_to_lp_column`'s Equal branch.
    ///
    /// Empty (`0..0`) when `n_anticipated == 0` or when built via
    /// [`StageIndexer::new`].
    pub anticipated_state_out: Range<usize>,

    /// Per-plant `lead_stages` (`K_i`) for the anticipated thermals.
    ///
    /// Length [`Self::n_anticipated`]; indexed by anticipated-local position
    /// (0-indexed within the anticipated subset). Empty when built via
    /// [`StageIndexer::new`].
    pub anticipated_lead_stages: Vec<usize>,

    /// Mapping from anticipated-local position to global thermal index.
    ///
    /// Length [`Self::n_anticipated`]; `anticipated_thermal_indices[i]` is the
    /// position within `system.thermals[]` of the i-th anticipated plant.
    /// Parallel to [`Self::anticipated_lead_stages`]. Mirrors the FPHA
    /// `fpha_hydro_indices` pattern. Empty when built via
    /// [`StageIndexer::new`].
    pub anticipated_thermal_indices: Vec<usize>,

    /// Reverse map: global thermal position → anticipated-local index.
    ///
    /// Inverse of [`Self::anticipated_thermal_indices`]: given a system-level
    /// thermal position `sys_pos`, `anticipated_local_by_sys_pos[&sys_pos]`
    /// yields the anticipated-local index (0-indexed within the anticipated
    /// subset). Built once at construction for O(1) resolution in
    /// `resolve_anticipated_decision`. Empty when `n_anticipated == 0`.
    pub(crate) anticipated_local_by_sys_pos: HashMap<usize, usize>,

    /// Column range for forward line flow variables, one per (line, block) pair.
    ///
    /// Index for line `l`, block `b`: `line_fwd.start + l * n_blks + b`.
    /// Empty when built via [`StageIndexer::new`].
    pub line_fwd: Range<usize>,

    /// Column range for reverse line flow variables, one per (line, block) pair.
    ///
    /// Index for line `l`, block `b`: `line_rev.start + l * n_blks + b`.
    /// Empty when built via [`StageIndexer::new`].
    pub line_rev: Range<usize>,

    /// Column range for bus deficit variables, `B * S * K` columns total.
    ///
    /// S = `max_deficit_segments` (uniform stride across all buses).  For buses
    /// with fewer than S segments, the trailing segment slots have zero bounds
    /// and zero objective and are eliminated by the presolver.
    ///
    /// Index for bus `b_idx`, segment `s`, block `blk`:
    /// `deficit.start + b_idx * max_deficit_segments * n_blks + s * n_blks + blk`.
    ///
    /// Empty when built via [`StageIndexer::new`].
    pub deficit: Range<usize>,

    /// Maximum number of deficit segments across all buses (S).
    ///
    /// Used together with `deficit.start` to compute per-segment column indices.
    /// Set to `0` when built via [`StageIndexer::new`]; both equipment
    /// constructors ([`StageIndexer::with_equipment`] and
    /// [`StageIndexer::with_equipment_and_evaporation`]) pass the caller-supplied
    /// `counts.max_deficit_segments` through unchanged.
    pub max_deficit_segments: usize,

    /// Column range for bus excess variables, one per (bus, block) pair.
    ///
    /// Index for bus `b_idx`, block `blk`: `excess.start + b_idx * n_blks + blk`.
    /// Empty when built via [`StageIndexer::new`].
    pub excess: Range<usize>,

    /// Number of operating blocks per stage (K).
    ///
    /// Zero when built via [`StageIndexer::new`].
    pub n_blks: usize,

    /// Number of thermal units (T).
    ///
    /// Zero when built via [`StageIndexer::new`].
    pub n_thermals: usize,

    /// Number of transmission lines (`L_n`).
    ///
    /// Zero when built via [`StageIndexer::new`].
    pub n_lines: usize,

    /// Number of buses (B).
    ///
    /// Zero when built via [`StageIndexer::new`].
    pub n_buses: usize,

    /// Row range for water balance constraints, one per operating hydro.
    ///
    /// Index for hydro `h`: `water_balance.start + h`.
    /// The dual of this row gives the marginal value of water (water value).
    /// Empty when built via [`StageIndexer::new`].
    pub water_balance: Range<usize>,

    /// Row range for load balance constraints, one per (bus, block) pair.
    ///
    /// Index for bus `b_idx`, block `blk`: `load_balance.start + b_idx * n_blks + blk`.
    /// The RHS of these rows contains the load (MW) for each bus in each block.
    /// Empty when built via [`StageIndexer::new`].
    pub load_balance: Range<usize>,

    /// Column range for inflow non-negativity slack variables `sigma_inf_h`.
    ///
    /// One slack per operating hydro, appended after `excess` when the penalty
    /// method is active (`has_inflow_penalty == true`).  The slack is in m³/s;
    /// it absorbs negative inflow realisations and enters the water balance row
    /// with coefficient `+tau_total * M3S_TO_HM3`.
    ///
    /// Empty (`0..0`) when `has_inflow_penalty == false` or when built via
    /// [`StageIndexer::new`].
    pub inflow_slack: Range<usize>,

    /// Row range for inflow non-negativity constraint rows.
    ///
    /// Currently unused as a separate constraint block — the slack appears
    /// directly in the water balance row.  Reserved for future formulations
    /// that add an explicit `sigma_inf_h + a_h >= 0` row.
    ///
    /// Empty (`0..0`) in this implementation.
    pub inflow_slack_rows: Range<usize>,

    /// Whether inflow non-negativity penalty slack columns are present.
    ///
    /// `true` when `build_stage_templates` was called with an
    /// [`InflowNonNegativityMethod`](crate::inflow_method::InflowNonNegativityMethod)
    /// whose `has_slack_columns()` returns `true` and `n_hydros > 0`.
    /// `false` otherwise (including when built via [`StageIndexer::new`]).
    pub has_inflow_penalty: bool,

    // ── FPHA column and row ranges ─────────────────────────────────────────
    // Populated only by `with_equipment`; empty when built via `new`.
    /// Column range for FPHA generation variables, one per (`fpha_hydro`, block) pair.
    ///
    /// Index for FPHA hydro at local position `i`, block `b`:
    /// `generation.start + i * n_blks + b`.
    /// Empty when no FPHA hydros exist or when built via [`StageIndexer::new`].
    pub generation: Range<usize>,

    /// Number of FPHA hydros in this stage.
    ///
    /// Zero when built via [`StageIndexer::new`].
    pub n_fpha_hydros: usize,

    /// Mapping from FPHA local index to system hydro index.
    ///
    /// `fpha_hydro_indices[i]` is the system-level hydro position for FPHA hydro `i`.
    /// Empty when no FPHA hydros exist or when built via [`StageIndexer::new`].
    pub fpha_hydro_indices: Vec<usize>,

    /// FPHA constraint row ranges per FPHA hydro.
    ///
    /// `fpha_rows[i]` is the [`FphaRowRange`] for FPHA hydro at local position `i`.
    /// Empty when no FPHA hydros exist or when built via [`StageIndexer::new`].
    pub fpha_rows: Vec<FphaRowRange>,

    // ── Evaporation column and row indices ─────────────────────────────────
    // Populated only by `with_equipment`; empty when built via `new`.
    /// Number of hydros with linearized evaporation at this stage.
    ///
    /// Zero when built via [`StageIndexer::new`] or when no evaporation hydros exist.
    pub n_evap_hydros: usize,

    /// Mapping from evaporation local index to system hydro index.
    ///
    /// `evap_hydro_indices[i]` is the system-level hydro position for evaporation hydro `i`.
    /// Empty when no evaporation hydros exist or when built via [`StageIndexer::new`].
    pub evap_hydro_indices: Vec<usize>,

    /// Per-evaporation-hydro column and row indices.
    ///
    /// `evap_indices[i]` is the [`EvaporationIndices`] for evaporation hydro at local
    /// position `i`.  Empty when no evaporation hydros exist or when built via
    /// [`StageIndexer::new`].
    pub evap_indices: Vec<EvaporationIndices>,

    // ── Withdrawal slack column ranges ─────────────────────────────────────
    // Populated only by `with_equipment_and_evaporation`; empty when built via `new`.
    /// Column range for under-withdrawal slack (withdrew less than target).
    ///
    /// One slack per operating hydro, appended after the evaporation columns.
    /// Columns are stage-level (not per-block); the slack absorbs violations of
    /// the minimum water-withdrawal flow constraint.
    ///
    /// Allocated whenever `hydro_count > 0`, matching the `inflow_slack` pattern.
    /// Empty (`0..0`) when `hydro_count == 0` or when built via
    /// [`StageIndexer::new`].
    /// Layout: `withdrawal_slack_neg.start + h_idx`.
    pub withdrawal_slack_neg: Range<usize>,

    /// Column range for over-withdrawal slack (withdrew more than target).
    ///
    /// One slack per operating hydro, immediately following `withdrawal_slack_neg`.
    /// Layout: `withdrawal_slack_pos.start + h_idx`.
    pub withdrawal_slack_pos: Range<usize>,

    /// Whether withdrawal slack columns are present.
    ///
    /// `true` when `with_equipment_and_evaporation` was called with
    /// `hydro_count > 0`.  `false` otherwise (including when built via
    /// [`StageIndexer::new`]).
    pub has_withdrawal: bool,

    // ── Operational violation slack column ranges ─────────────────────────
    // Populated only by the full build path; empty (`0..0`) when built via `new`.
    /// Column range for outflow-below violation slacks, one per hydro per block.
    ///
    /// `outflow_below_slack.start + h * n_blks + blk` is the column for hydro `h`,
    /// block `blk`.  Empty (`0..0`) when built via [`StageIndexer::new`].
    pub outflow_below_slack: Range<usize>,

    /// Column range for outflow-above violation slacks, one per hydro per block.
    ///
    /// `outflow_above_slack.start + h * n_blks + blk` is the column for hydro `h`,
    /// block `blk`.  Empty (`0..0`) when built via [`StageIndexer::new`].
    pub outflow_above_slack: Range<usize>,

    /// Column range for turbine-below violation slacks, one per hydro per block.
    ///
    /// `turbine_below_slack.start + h * n_blks + blk` is the column for hydro `h`,
    /// block `blk`.  Empty (`0..0`) when built via [`StageIndexer::new`].
    pub turbine_below_slack: Range<usize>,

    /// Column range for generation-below violation slacks, one per hydro per block.
    ///
    /// `generation_below_slack.start + h * n_blks + blk` is the column for hydro `h`,
    /// block `blk`.  Empty (`0..0`) when built via [`StageIndexer::new`].
    pub generation_below_slack: Range<usize>,

    // ── Operational violation constraint row ranges ────────────────────────
    // Populated only by the full build path; empty (`0..0`) when built via `new`.
    /// Row range for min-outflow constraint rows, one per hydro per block.
    ///
    /// `min_outflow_rows.start + h * n_blks + blk` is the row for hydro `h`,
    /// block `blk`.  Empty (`0..0`) when built via [`StageIndexer::new`].
    pub min_outflow_rows: Range<usize>,

    /// Row range for max-outflow constraint rows, one per hydro per block.
    ///
    /// `max_outflow_rows.start + h * n_blks + blk` is the row for hydro `h`,
    /// block `blk`.  Empty (`0..0`) when built via [`StageIndexer::new`].
    pub max_outflow_rows: Range<usize>,

    /// Row range for min-turbine constraint rows, one per hydro per block.
    ///
    /// `min_turbine_rows.start + h * n_blks + blk` is the row for hydro `h`,
    /// block `blk`.  Empty (`0..0`) when built via [`StageIndexer::new`].
    pub min_turbine_rows: Range<usize>,

    /// Row range for min-generation constraint rows, one per hydro per block.
    ///
    /// `min_generation_rows.start + h * n_blks + blk` is the row for hydro `h`,
    /// block `blk`.  Empty (`0..0`) when built via [`StageIndexer::new`].
    pub min_generation_rows: Range<usize>,

    /// Row range for anticipated-thermal fishing constraints.
    ///
    /// Empty (`0..0`) in the canonical layout placeholder; per-stage active
    /// row indices are always `n_anticipated` under the always-active
    /// predicate and are accessed via
    /// [`anticipated_fishing_active_at_stage`](Self::anticipated_fishing_active_at_stage).
    ///
    /// The fishing constraint reads:
    /// `gt_i^(t) - anticipated_state[slot=0, plant=i] = 0`
    /// where the dual on this row carries the cut subgradient w.r.t.
    /// the slot read by the fishing constraint at delivery.
    pub anticipated_fishing: Range<usize>,

    /// First row index of the anticipated-fishing block.
    ///
    /// Equal to `min_generation_rows.end` when operational violations are
    /// active, or to `evap_rows_end` (= `fpha_row_cursor + n_evap_hydros`)
    /// when they are not. Zero when built via [`StageIndexer::new`].
    ///
    /// Per-stage fishing row indices are computed as
    /// `lp_row = anticipated_fishing_start + local_idx_at_stage`.
    pub anticipated_fishing_start: usize,

    /// Whether operational violation slack columns are present.
    ///
    /// `true` when the full build path was used with `hydro_count > 0`.
    /// `false` otherwise (including when built via [`StageIndexer::new`]).
    pub has_operational_violations: bool,

    // ── Generic constraint row and column ranges (permanent sentinels) ──────
    // CONTRACT: these three `StageIndexer` fields are never assigned a non-zero
    // value by any constructor or wiring step — they are permanent `0..0` / `0`
    // sentinels in this implementation, the same status as the `storage_fixing`
    // row range. The live per-stage generic-constraint layout lives on the
    // separate `StageLayout::generic_constraint_rows` field (a
    // `Vec<GenericConstraintRowEntry>`), which the matrix-build and simulation
    // extraction code consume; that vec is never written back into these
    // indexer fields. The sole read of the indexer field is
    // `generic_constraint_rows.start` in simulation extraction, where the
    // always-zero start feeds a discarded `_dual_value`; the offset extraction
    // actually uses is the per-entry loop counter, so the zero value is inert.
    // FORBIDDEN: do not "fix" the sentinel by wiring `StageLayout`'s vec back
    // into the indexer field, and do not treat the indexer field as the live
    // generic-row offset — both break the layout single-owner contract.
    /// Row range a generic-constraint block would occupy (one per active
    /// `(constraint, block)` pair), placed after evaporation rows.
    ///
    /// Always empty (`0..0`) here — see the permanent-sentinel contract above.
    /// The live per-stage layout is `StageLayout::generic_constraint_rows`.
    pub generic_constraint_rows: Range<usize>,

    /// Column range a generic-constraint slack block would occupy, placed after
    /// withdrawal slack columns.
    ///
    /// Always empty (`0..0`) here — see the permanent-sentinel contract above.
    pub generic_constraint_slack: Range<usize>,

    /// Count a generic-constraint block would contribute at this stage.
    ///
    /// Always `0` here — see the permanent-sentinel contract above.
    pub n_generic_constraints_active: usize,

    // ── NCS column range ──────────────────────────────────────────────────
    // OWNER: populated after construction by the NCS wiring in `setup`
    // (`build_wired_indexer`, which assigns `indexer.ncs_generation` from the LP
    // builder's stage-0 NCS column starts) — NOT by `StageLayout::new`, which
    // only reads indexer cursors. Left empty by every `StageIndexer` constructor
    // (`new`, `with_equipment`, `from_stage_template`); the wiring fills it in,
    // and the hot paths (forward, backward, simulation, lower-bound spec) read it.
    /// Column range for NCS generation variables, one per (ncs, block) pair.
    ///
    /// Index for NCS `r`, block `b`: `ncs_generation.start + r * n_blks + b`.
    /// Empty when built via [`StageIndexer::new`] or when no NCS entities are active.
    pub ncs_generation: Range<usize>,

    // ── Pumping-flow column range ─────────────────────────────────────────
    // Permanent `0..0` sentinel here, mirroring `ncs_generation`: every
    // `StageIndexer` constructor leaves it empty. The live per-stage pumping
    // column layout is owned by `StageLayout::col_pumping_start` / `n_pumping`;
    // this field exists so the indexer exposes a per-region range for pumping
    // the same way it does for NCS. The block-major layout is
    // `start + station_local_idx * n_blks + blk`.
    /// Column range for pumping-flow variables, one per (pumping, block) pair.
    ///
    /// Index for pumping station at local position `p`, block `b`:
    /// `pumping_flow.start + p * n_blks + b` (block-major).
    /// Always `0..0`: every constructor leaves it empty regardless of station
    /// count — the live column range is owned by `StageLayout::col_pumping_start`.
    pub pumping_flow: Range<usize>,

    // ── Z-inflow column and row ranges ────────────────────────────────────
    // Populated by all constructors.  The z_inflow columns are auxiliary
    // (NOT state variables); their primal values give the realized total
    // inflow Z_t per hydro after solving.
    /// Column range for realized-inflow variables `z_h`, one per hydro.
    ///
    /// These free columns (lower = -inf, upper = +inf, zero cost) represent the
    /// total natural inflow `Z_t_h` at each hydro, defined by the z-inflow
    /// equality constraints. After solving, `primal[z_inflow.start + h]` gives
    /// the realized inflow for hydro h.
    ///
    /// Empty when `hydro_count == 0`.
    pub z_inflow: Range<usize>,

    /// Row range for z-inflow definition constraints, one per hydro.
    ///
    /// Each row defines: `z_h - sum_l[psi_l * lag_in[h,l]] = base_h + sigma_h * eta_h`
    /// The RHS is noise-patched (Category 5 in `PatchBuffer`).
    ///
    /// Empty when `hydro_count == 0`.
    pub z_inflow_rows: Range<usize>,

    /// Row index of the first z-inflow definition constraint.
    ///
    /// Used by `PatchBuffer::fill_z_inflow_patches` as the base offset for
    /// Category 5 patches. Equal to `z_inflow_rows.start`.
    pub z_inflow_row_start: usize,

    /// Indices of state dimensions whose cut coefficients can be nonzero.
    ///
    /// Storage indices `[0, N)` are always included. Lag indices `[N, N*(1+L))`
    /// are included only when `lag < actual_ar_order[hydro]`. Hydros with AR
    /// order < `max_par_order` have padded lag slots whose duals are
    /// structurally zero.
    ///
    /// When empty (default), callers should treat all `n_state` indices as
    /// nonzero (dense path). Use [`set_nonzero_mask`](Self::set_nonzero_mask) to
    /// populate after construction when per-hydro AR orders are available.
    pub nonzero_state_indices: Vec<usize>,

    /// Precomputed `state_to_lp_column(j)` for every `j ∈ [0, n_state)`.
    ///
    /// Built once after the state layout is finalized; read on the
    /// forward-pass cut-row hot path. Empty until finalized.
    pub state_to_lp_column_map: Vec<usize>,
}

/// Equipment counts for constructing a [`StageIndexer`].
///
/// Groups the entity counts that determine the LP column layout for a single stage.
pub struct EquipmentCounts {
    /// Number of hydro plants.
    pub hydro_count: usize,
    /// Maximum PAR model order across all hydros.
    pub max_par_order: usize,
    /// Number of thermal units.
    pub n_thermals: usize,
    /// Number of transmission lines.
    pub n_lines: usize,
    /// Number of buses.
    pub n_buses: usize,
    /// Number of demand blocks in the stage.
    pub n_blks: usize,
    /// Whether to include inflow penalty slack columns.
    pub has_inflow_penalty: bool,
    /// Maximum number of deficit segments across all buses.
    pub max_deficit_segments: usize,
    /// Number of anticipated thermals (`anticipated_config.is_some()`).
    pub n_anticipated: usize,
    /// Maximum `lead_stages` across the anticipated thermals.
    pub k_max: usize,
    /// Number of pumping stations.
    ///
    /// Accepted for structural symmetry with the other entity counts but **not
    /// read** by [`StageIndexer::with_equipment_and_evaporation`]: the indexer's
    /// `pumping_flow` field is a permanent `0..0` sentinel. The real pumping-flow
    /// column block (`n_pumping * n_blks`, block-major, reserved between the NCS
    /// region and the generic-slack columns) is owned by `StageLayout`, which
    /// reads its station count from `ctx.n_pumping`, not from this field.
    pub n_pumping: usize,
    /// Per-plant `lead_stages` (`K_i`) for the anticipated thermals.
    ///
    /// Length must equal `n_anticipated`. The maximum entry (when non-empty)
    /// must equal `k_max`. Pass-through to
    /// [`StageIndexer::anticipated_lead_stages`].
    pub anticipated_lead_stages: Vec<usize>,
    /// Mapping from anticipated-local position to global thermal index.
    ///
    /// Length must equal `n_anticipated`. Parallel to `anticipated_lead_stages`.
    /// Pass-through to [`StageIndexer::anticipated_thermal_indices`].
    pub anticipated_thermal_indices: Vec<usize>,
}

/// FPHA (Piecewise-linear Hydro Approximation) column layout.
///
/// Groups the per-hydro FPHA data needed for column layout computation.
pub struct FphaColumnLayout {
    /// Indices of hydros using FPHA production models.
    pub hydro_indices: Vec<usize>,
    /// Number of FPHA planes for each hydro in `hydro_indices`.
    ///
    /// Must have the same length as `hydro_indices`.
    pub planes_per_hydro: Vec<usize>,
}

/// Evaporation configuration for hydro plants.
pub struct EvapConfig {
    /// Indices of hydros with evaporation modeling enabled.
    pub hydro_indices: Vec<usize>,
}

impl StageIndexer {
    /// Return the [`EvaporationIndices`] for the evaporation hydro at local position `local_idx`.
    ///
    /// `local_idx` is the position within the evaporation hydro list (0-indexed).
    /// Use `evap_hydro_indices[local_idx]` to map to the system-level hydro position.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `local_idx >= n_evap_hydros`.
    #[must_use]
    pub fn evap_indices(&self, local_idx: usize) -> &EvaporationIndices {
        debug_assert!(
            local_idx < self.n_evap_hydros,
            "evap local index {local_idx} out of bounds (n_evap_hydros = {})",
            self.n_evap_hydros
        );
        &self.evap_indices[local_idx]
    }

    /// First column of the FPHA generation block, even when that block is empty.
    ///
    /// The public [`Self::generation`] range is normalised to `0..0` when no FPHA
    /// hydros exist, so its `.start` is `0` rather than the real cursor. This
    /// returns the cursor (`inflow_slack.end` when the inflow penalty is active,
    /// otherwise `excess.end`), matching the `generation_start` derivation in the
    /// constructor — never `0` for a real LP.
    #[must_use]
    pub(crate) fn generation_col_start(&self) -> usize {
        if self.has_inflow_penalty {
            self.inflow_slack.end
        } else {
            self.excess.end
        }
    }

    /// First column of the evaporation block, even when that block is empty.
    ///
    /// The public [`Self::evap_indices`] list is empty when no evaporation hydros
    /// exist, so it exposes no cursor. This returns the cursor
    /// (`generation_col_start + n_fpha_hydros * n_blks`, i.e. the FPHA generation
    /// block end), matching the `evap_col_start` derivation in the constructor.
    #[must_use]
    pub(crate) fn evap_col_start(&self) -> usize {
        self.generation_col_start() + self.n_fpha_hydros * self.n_blks
    }

    /// First row after the FPHA constraint block, even when that block is empty.
    ///
    /// The public [`Self::fpha_rows`] list is empty when no FPHA hydros exist, so
    /// it exposes no end cursor. This returns the cursor
    /// (`load_balance.end + n_blks * sum(planes_per_block)`, i.e. the row at which
    /// evaporation rows begin), matching the `fpha_row_cursor` derivation in the
    /// constructor.
    #[must_use]
    pub(crate) fn fpha_rows_end(&self) -> usize {
        let total_planes: usize = self.fpha_rows.iter().map(|r| r.planes_per_block).sum();
        self.load_balance.end + self.n_blks * total_planes
    }
}

// StageIndexer contains only Send + Sync types (Range<usize>, usize, Vec<usize>,
// Vec<FphaRowRange>, Vec<EvaporationIndices>), so Send + Sync are automatically
// derived. The explicit bounds below serve as a compile-time assertion that the
// safety invariant holds.
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn check() {
        assert_send_sync::<StageIndexer>();
    }
    let _ = check;
};

#[cfg(test)]
mod tests {
    use super::{EvaporationIndices, FphaRowRange};
    use crate::indexer::StageIndexer;

    fn indexer_3_2() -> StageIndexer {
        StageIndexer::new(3, 2)
    }

    // Clone / Debug — structural sanity

    #[test]
    fn clone_and_debug() {
        let idx = indexer_3_2();
        let cloned = idx.clone();
        assert_eq!(cloned.theta, idx.theta);
        assert_eq!(cloned.n_state, idx.n_state);

        let debug_str = format!("{idx:?}");
        assert!(debug_str.contains("StageIndexer"));
    }

    // EvaporationIndices is Debug + Clone + Copy.
    #[test]
    fn evap_indices_debug_clone_copy() {
        let ei = EvaporationIndices {
            evaporation_flow_col: 10,
            f_evap_plus_col: 11,
            f_evap_minus_col: 12,
            evap_row: 5,
        };
        let cloned = ei;
        assert_eq!(cloned.evaporation_flow_col, 10);
        assert_eq!(cloned.evap_row, 5);
        let debug_str = format!("{ei:?}");
        assert!(debug_str.contains("EvaporationIndices"));
    }

    // FphaRowRange is Debug + Clone + Copy.
    #[test]
    fn fpha_row_range_debug_clone_copy() {
        let r = FphaRowRange {
            start: 42,
            planes_per_block: 5,
        };
        let cloned = r;
        assert_eq!(cloned.start, 42);
        assert_eq!(cloned.planes_per_block, 5);
        let debug_str = format!("{r:?}");
        assert!(debug_str.contains("FphaRowRange"));
    }
}
