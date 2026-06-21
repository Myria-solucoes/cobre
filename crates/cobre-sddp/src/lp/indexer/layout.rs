//! The [`StageIndexer`] role-(b) geometry descriptor, its satellite layout
//! types, the small layout accessors, and the compile-time `Send + Sync`
//! assertion.
//!
//! `StageIndexer` is the **role-(b) geometry descriptor**: it carries the
//! stage-0 equipment / slack / row ranges, the entity-count scalars that stride
//! them, the presence flags, and the FPHA / evaporation / anticipated identity
//! lists. The role-(a) state-vector concern — the state-region column ranges
//! (`storage`, `inflow_lags`, `storage_in`, `anticipated_state`,
//! `anticipated_state_out`, `z_inflow`, `theta`), `n_state`, the state-vector
//! dimension scalars, and the cut resolvers / mask — lives entirely on
//! [`StateLayout`](super::StateLayout). The cut path, state-fixing patch, and
//! simulation-extraction state reads resolve through that handle; this descriptor
//! never reimplements them.
//!
//! The equipment column **bases** here are strided by a stage-0-derived `n_blks`
//! (the single global block count); they are valid only at stages whose block
//! count equals stage 0's. The per-stage [`StageLayout`](crate::lp_builder)
//! recomputes each stage's own equipment bases for template construction and is
//! the authority where block counts vary.

use std::ops::Range;

use super::BlockGrid;

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

/// Read-only role-(b) LP geometry descriptor for one SDDP stage subproblem.
///
/// Carries the stage-0 equipment / slack column ranges, the constraint row
/// ranges, the entity-count scalars that stride them, the presence flags, and
/// the FPHA / evaporation / anticipated identity lists. Computed once in
/// `build_wired_indexer` and shared read-only across all threads for the
/// duration of training.
///
/// The role-(a) state-vector concern (`storage`, `inflow_lags`, `storage_in`,
/// `anticipated_state`, `anticipated_state_out`, `z_inflow` columns, `theta`,
/// `n_state`, the resolvers, the mask) lives on
/// [`StateLayout`](super::StateLayout); this descriptor carries none of it.
///
/// The equipment column ranges below are strided by a stage-0-derived `n_blks`
/// and are valid only at stages whose block count equals stage 0's; the
/// per-stage [`StageLayout`](crate::lp_builder) is the authority where block
/// counts vary.
// Rationale: the bool fields (`has_inflow_penalty`, `has_withdrawal`,
// `has_operational_violations`, `has_ncs`) are independent presence flags for
// optional column groups, not states of one machine; folding them into a
// two-variant enum or state machine — the lint's suggested refactor — would
// obscure that they vary independently. The slimmed descriptor still carries
// four such flags (above clippy's three-bool trigger), so the allow stays.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone)]
pub struct StageIndexer {
    // ── Equipment column ranges ────────────────────────────────────────────
    /// Column range for turbined flow variables, one per (hydro, block) pair.
    ///
    /// Index for hydro `h`, block `b`: `turbine.start + h * n_blks + b`.
    pub turbine: Range<usize>,

    /// Column range for spillage variables, one per (hydro, block) pair.
    ///
    /// Index for hydro `h`, block `b`: `spillage.start + h * n_blks + b`.
    pub spillage: Range<usize>,

    /// Column range for diversion flow variables, one per (hydro, block) pair.
    ///
    /// Index for hydro `h`, block `b`: `diversion.start + h * n_blks + b`.
    /// Hydros without a diversion channel have bounds [0, 0]; the LP presolve
    /// eliminates them.
    pub diversion: Range<usize>,

    /// Column range for thermal generation variables, one per (thermal, block) pair.
    ///
    /// Index for thermal `t`, block `b`: `thermal.start + t * n_blks + b`.
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
    /// the range length equals the anticipated-plant count.
    ///
    /// This is a **control-region** column: its base rides the n_blks-dependent
    /// `thermal.end`, so it is valid only at stages whose block count equals
    /// stage 0's. The matching cut-target `anticipated_state_out` column lives in
    /// the stage-invariant state region on [`StateLayout`](super::StateLayout).
    pub anticipated_decision: Range<usize>,

    /// Mapping from anticipated-local position to global thermal index.
    ///
    /// `anticipated_thermal_indices[i]` is the position within `system.thermals[]`
    /// of the i-th anticipated plant. Read by the simulation-extraction
    /// reverse-lookup builder and by `build_initial_state`. Empty when
    /// `n_anticipated == 0`.
    pub anticipated_thermal_indices: Vec<usize>,

    /// Column range for forward line flow variables, one per (line, block) pair.
    ///
    /// Index for line `l`, block `b`: `line_fwd.start + l * n_blks + b`.
    pub line_fwd: Range<usize>,

    /// Column range for reverse line flow variables, one per (line, block) pair.
    ///
    /// Index for line `l`, block `b`: `line_rev.start + l * n_blks + b`.
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
    pub deficit: Range<usize>,

    /// Maximum number of deficit segments across all buses (S).
    ///
    /// Used together with `deficit.start` to compute per-segment column indices.
    pub max_deficit_segments: usize,

    /// Column range for bus excess variables, one per (bus, block) pair.
    ///
    /// Index for bus `b_idx`, block `blk`: `excess.start + b_idx * n_blks + blk`.
    pub excess: Range<usize>,

    /// Number of operating blocks per stage (K).
    ///
    pub n_blks: usize,

    /// Number of thermal units (T).
    ///
    pub n_thermals: usize,

    /// Number of transmission lines (`L_n`).
    ///
    pub n_lines: usize,

    /// Number of buses (B).
    ///
    pub n_buses: usize,

    /// Row range for water balance constraints, one per operating hydro.
    ///
    /// Index for hydro `h`: `water_balance.start + h`.
    /// The dual of this row gives the marginal value of water (water value).
    pub water_balance: Range<usize>,

    /// Row range for load balance constraints, one per (bus, block) pair.
    ///
    /// Index for bus `b_idx`, block `blk`: `load_balance.start + b_idx * n_blks + blk`.
    /// The RHS of these rows contains the load (MW) for each bus in each block.
    pub load_balance: Range<usize>,

    /// Column range for inflow non-negativity slack variables `sigma_inf_h`.
    ///
    /// One slack per operating hydro, appended after `excess` when the penalty
    /// method is active (`has_inflow_penalty == true`).  The slack is in m³/s;
    /// it absorbs negative inflow realisations and enters the water balance row
    /// with coefficient `+tau_total * M3S_TO_HM3`.
    ///
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
    pub has_inflow_penalty: bool,

    // ── FPHA column and row ranges ─────────────────────────────────────────
    /// Column range for FPHA generation variables, one per (`fpha_hydro`, block) pair.
    ///
    /// Index for FPHA hydro at local position `i`, block `b`:
    /// `generation.start + i * n_blks + b`.
    pub generation: Range<usize>,

    /// Number of FPHA hydros in this stage.
    ///
    pub n_fpha_hydros: usize,

    /// Mapping from FPHA local index to system hydro index.
    ///
    /// `fpha_hydro_indices[i]` is the system-level hydro position for FPHA hydro `i`.
    pub fpha_hydro_indices: Vec<usize>,

    /// FPHA constraint row ranges per FPHA hydro.
    ///
    /// `fpha_rows[i]` is the [`FphaRowRange`] for FPHA hydro at local position `i`.
    pub fpha_rows: Vec<FphaRowRange>,

    // ── Evaporation column and row indices ─────────────────────────────────
    /// Number of hydros with linearized evaporation at this stage.
    ///
    pub n_evap_hydros: usize,

    /// Mapping from evaporation local index to system hydro index.
    ///
    /// `evap_hydro_indices[i]` is the system-level hydro position for evaporation hydro `i`.
    pub evap_hydro_indices: Vec<usize>,

    /// Per-evaporation-hydro column and row indices.
    ///
    /// `evap_indices[i]` is the [`EvaporationIndices`] for evaporation hydro at local
    /// position `i`..
    pub evap_indices: Vec<EvaporationIndices>,

    // ── Withdrawal slack column ranges ─────────────────────────────────────
    /// Column range for under-withdrawal slack (withdrew less than target).
    ///
    /// One slack per operating hydro, appended after the evaporation columns.
    /// Columns are stage-level (not per-block); the slack absorbs violations of
    /// the minimum water-withdrawal flow constraint.
    ///
    /// Allocated whenever `hydro_count > 0`, matching the `inflow_slack` pattern.
    /// Layout: `withdrawal_slack_neg.start + h_idx`.
    pub withdrawal_slack_neg: Range<usize>,

    /// Column range for over-withdrawal slack (withdrew more than target).
    ///
    /// One slack per operating hydro, immediately following `withdrawal_slack_neg`.
    /// Layout: `withdrawal_slack_pos.start + h_idx`.
    pub withdrawal_slack_pos: Range<usize>,

    /// Whether withdrawal slack columns are present.
    ///
    /// `hydro_count > 0`; `false` otherwise (zero hydros).
    pub has_withdrawal: bool,

    // ── Operational violation slack column ranges ─────────────────────────
    /// Column range for outflow-below violation slacks, one per hydro per block.
    ///
    /// `outflow_below_slack.start + h * n_blks + blk` is the column for hydro `h`,
    /// block `blk`.
    pub outflow_below_slack: Range<usize>,

    /// Column range for outflow-above violation slacks, one per hydro per block.
    ///
    /// `outflow_above_slack.start + h * n_blks + blk` is the column for hydro `h`,
    /// block `blk`.
    pub outflow_above_slack: Range<usize>,

    /// Column range for turbine-below violation slacks, one per hydro per block.
    ///
    /// `turbine_below_slack.start + h * n_blks + blk` is the column for hydro `h`,
    /// block `blk`.
    pub turbine_below_slack: Range<usize>,

    /// Column range for generation-below violation slacks, one per hydro per block.
    ///
    /// `generation_below_slack.start + h * n_blks + blk` is the column for hydro `h`,
    /// block `blk`.
    pub generation_below_slack: Range<usize>,

    // ── Operational violation constraint row ranges ────────────────────────
    /// Row range for min-outflow constraint rows, one per hydro per block.
    ///
    /// `min_outflow_rows.start + h * n_blks + blk` is the row for hydro `h`,
    /// block `blk`.
    pub min_outflow_rows: Range<usize>,

    /// Row range for max-outflow constraint rows, one per hydro per block.
    ///
    /// `max_outflow_rows.start + h * n_blks + blk` is the row for hydro `h`,
    /// block `blk`.
    pub max_outflow_rows: Range<usize>,

    /// Row range for min-turbine constraint rows, one per hydro per block.
    ///
    /// `min_turbine_rows.start + h * n_blks + blk` is the row for hydro `h`,
    /// block `blk`.
    pub min_turbine_rows: Range<usize>,

    /// Row range for min-generation constraint rows, one per hydro per block.
    ///
    /// `min_generation_rows.start + h * n_blks + blk` is the row for hydro `h`,
    /// block `blk`.
    pub min_generation_rows: Range<usize>,

    /// Row range for anticipated-thermal fishing constraints.
    ///
    /// Empty (`0..0`) in the canonical layout placeholder. The fishing
    /// constraint is always active for every anticipated plant, so a stage
    /// emits exactly `n_anticipated` rows at dense offsets
    /// `anticipated_fishing_start + local_idx` (see
    /// [`anticipated_fishing_start`](Self::anticipated_fishing_start)).
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
    /// when they are not.
    ///
    /// Per-stage fishing row indices are computed as
    /// `lp_row = anticipated_fishing_start + local_idx_at_stage`.
    pub anticipated_fishing_start: usize,

    /// Whether operational violation slack columns are present.
    ///
    /// `true` when the full build path was used with `hydro_count > 0`.
    pub has_operational_violations: bool,

    // ── NCS presence flag ─────────────────────────────────────────────────
    // OWNER: set after construction by the NCS wiring in `setup`
    // (`build_wired_indexer`). The constructor leaves it `false`; the wiring sets it.
    //
    // This is a presence flag, never a column base: NCS columns are addressed
    // per-stage from `StageContext::ncs_col_starts[stage]` (and
    // `LbEvalSpec::ncs_generation` for the stage-0 lower bound), because a source
    // that commissions mid-horizon or a stage with a differing block count shifts
    // the NCS column base per stage. A single global base would address the wrong
    // columns for such non-uniform geometries, so the descriptor stores only whether
    // NCS columns exist, not where they start.
    /// Whether the study has NCS generation columns.
    ///
    /// `true` when NCS entities are active; `false` when no NCS entities are
    /// present, in which case the forward, backward, and simulation NCS bound
    /// patches are guarded off.
    pub has_ncs: bool,

    // ── Z-inflow row range ────────────────────────────────────────────────
    // The z_inflow *columns* are role-(a) and live on `StateLayout`; the z-inflow
    // *rows* (the definition constraints, noise-patched per stage) are role-(b)
    // geometry and stay here.
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
    /// Category 5 patches. Equal to `z_inflow_rows.start` (row 0).
    pub z_inflow_row_start: usize,
}

/// Equipment counts for constructing a [`StageIndexer`].
///
/// Groups the entity counts that determine the LP column layout for a single stage.
///
/// `Default` yields all-zero / empty-`Vec` counts (every `usize` field `0`,
/// `has_inflow_penalty == false`, both anticipated vecs empty). It exists for
/// test-fixture ergonomics: production construction sites populate every field
/// explicitly. In particular `max_deficit_segments` defaults to `0`, whereas the
/// named test-fixture builders deliberately use `1`.
#[derive(Debug, Clone, Default)]
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
    /// read** by the geometry constructor: it reserves no pumping column block.
    /// The real pumping-flow column block (`n_pumping * n_blks`, block-major,
    /// reserved between the NCS region and the generic-slack columns) is owned by
    /// `StageLayout`, which reads its station count from `ctx.n_pumping`, not from
    /// this field.
    pub n_pumping: usize,
    /// Per-plant `lead_stages` (`K_i`) for the anticipated thermals.
    ///
    /// Length must equal `n_anticipated`. The maximum entry (when non-empty)
    /// must equal `k_max`. Threaded into
    /// [`StateLayout`](super::StateLayout)'s `anticipated_lead_stages`.
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
    /// Return the [`BlockGrid`] address primitive for this stage's LP.
    ///
    /// The grid carries this descriptor's `n_blks` and `max_deficit_segments` —
    /// the two stride constants the three block-stride shapes (flat block-major,
    /// FPHA-plane, deficit 3-term) need beyond their per-call args. It is a cheap
    /// `Copy` value. Sourcing both constants from this single owning descriptor is
    /// what keeps the grid from disagreeing with the LP it addresses; see
    /// [`BlockGrid`] for the per-shape contracts.
    ///
    /// This grid carries the descriptor's own `n_blks` (the global count wired
    /// once from stage 0). For a per-stage fill whose block count differs from
    /// stage 0 — the load-balance RHS patch — construct the grid from the
    /// per-stage count with [`BlockGrid::new`] instead, or the row stride
    /// addresses the wrong row.
    #[inline]
    #[must_use]
    pub fn block_grid(&self) -> BlockGrid {
        BlockGrid::new(self.n_blks, self.max_deficit_segments)
    }

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
    use crate::indexer::test_fixtures::{eq, fpha};

    fn indexer_3_2() -> StageIndexer {
        // N=3, L=2, T=1, Ln=1, B=2, K=1 — a representative role-(b) geometry.
        StageIndexer::with_equipment_and_evaporation(
            &eq(3, 2, 1, 1, 2, 1, false),
            &fpha(vec![], vec![]),
            &crate::indexer::test_fixtures::evap(vec![]),
        )
    }

    // EquipmentCounts::default() yields all-zero scalars and empty vecs.
    #[test]
    fn equipment_counts_default_is_all_zero() {
        let counts = crate::indexer::EquipmentCounts::default();
        assert_eq!(counts.hydro_count, 0);
        assert_eq!(counts.max_par_order, 0);
        assert_eq!(counts.n_thermals, 0);
        assert_eq!(counts.n_lines, 0);
        assert_eq!(counts.n_buses, 0);
        assert_eq!(counts.n_blks, 0);
        assert!(!counts.has_inflow_penalty);
        assert_eq!(counts.max_deficit_segments, 0);
        assert_eq!(counts.n_anticipated, 0);
        assert_eq!(counts.k_max, 0);
        assert_eq!(counts.n_pumping, 0);
        assert_eq!(counts.anticipated_lead_stages, Vec::<usize>::new());
        assert_eq!(counts.anticipated_thermal_indices, Vec::<usize>::new());
    }

    #[test]
    fn clone_and_debug() {
        let idx = indexer_3_2();
        let cloned = idx.clone();
        // Role-(b) geometry fields survive a clone.
        assert_eq!(cloned.turbine, idx.turbine);
        assert_eq!(cloned.thermal, idx.thermal);
        assert_eq!(cloned.n_blks, idx.n_blks);

        let debug_str = format!("{idx:?}");
        assert!(debug_str.contains("StageIndexer"));
    }

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
