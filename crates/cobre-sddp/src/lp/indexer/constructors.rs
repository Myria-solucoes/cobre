//! [`StageIndexer`] constructors and the private column/row range build helpers.
//!
//! Owns the three public constructors (`new`, `with_equipment`,
//! `with_equipment_and_evaporation`), the `from_stage_template` adapter, and the
//! private helpers that compute each region's column/row range. The
//! `storage_fixing: 0..0`, `lag_fixing: 0..0`, and `anticipated_state_fixing:
//! 0..0` field initialisers (the state-pinning sentinel contract) live in these
//! constructor bodies and move verbatim.

use std::collections::HashMap;
use std::ops::Range;

use cobre_solver::StageTemplate;

use crate::lp_builder::{
    EVAP_COLS_PER_HYDRO, EVAP_F_MINUS_OFFSET, EVAP_F_PLUS_OFFSET, EVAP_FLOW_OFFSET,
};

use super::layout::{
    EquipmentCounts, EvapConfig, EvaporationIndices, FphaColumnLayout, FphaRowRange, StageIndexer,
};

/// Build the inflow-penalty slack column range.
///
/// Returns `(range, active)` where `active` is `true` when the penalty method
/// applies. When inactive, the range is empty (`0..0`).
fn build_inflow_slack_range(
    has_inflow_penalty: bool,
    hydro_count: usize,
    excess_end: usize,
) -> (Range<usize>, bool) {
    if has_inflow_penalty && hydro_count > 0 {
        (excess_end..excess_end + hydro_count, true)
    } else {
        (0..0, false)
    }
}

/// Column and row ranges for the four operational-violation slack families.
///
/// Produced by [`build_oper_violation_ranges`] and consumed by
/// [`StageIndexer::with_equipment_and_evaporation`].
struct OperViolationRanges {
    outflow_below_slack: Range<usize>,
    outflow_above_slack: Range<usize>,
    turbine_below_slack: Range<usize>,
    generation_below_slack: Range<usize>,
    min_outflow_rows: Range<usize>,
    max_outflow_rows: Range<usize>,
    min_turbine_rows: Range<usize>,
    min_generation_rows: Range<usize>,
    has_operational_violations: bool,
}

/// Build operational-violation column and row ranges.
///
/// When `hydro_count == 0`, all ranges are empty (`0..0`) and
/// `has_operational_violations` is `false`. Otherwise the four slack column
/// families are laid out immediately after `ws_end` (the end of the withdrawal
/// slack columns), and the four constraint row families are laid out immediately
/// after `evap_rows_end`.
fn build_oper_violation_ranges(
    hydro_count: usize,
    n_blks: usize,
    ws_end: usize,
    evap_rows_end: usize,
) -> OperViolationRanges {
    if hydro_count == 0 {
        return OperViolationRanges {
            outflow_below_slack: 0..0,
            outflow_above_slack: 0..0,
            turbine_below_slack: 0..0,
            generation_below_slack: 0..0,
            min_outflow_rows: 0..0,
            max_outflow_rows: 0..0,
            min_turbine_rows: 0..0,
            min_generation_rows: 0..0,
            has_operational_violations: false,
        };
    }
    let n_op = hydro_count * n_blks;
    let ob = ws_end..ws_end + n_op;
    let oa = ob.end..ob.end + n_op;
    let tb = oa.end..oa.end + n_op;
    let gb = tb.end..tb.end + n_op;
    let r_min_out = evap_rows_end..evap_rows_end + n_op;
    let r_max_out = r_min_out.end..r_min_out.end + n_op;
    let r_min_turb = r_max_out.end..r_max_out.end + n_op;
    let r_min_gen = r_min_turb.end..r_min_turb.end + n_op;
    OperViolationRanges {
        outflow_below_slack: ob,
        outflow_above_slack: oa,
        turbine_below_slack: tb,
        generation_below_slack: gb,
        min_outflow_rows: r_min_out,
        max_outflow_rows: r_max_out,
        min_turbine_rows: r_min_turb,
        min_generation_rows: r_min_gen,
        has_operational_violations: true,
    }
}

impl StageIndexer {
    /// Construct a [`StageIndexer`] from `hydro_count` (N) and `max_par_order` (L).
    ///
    /// All index ranges are computed from N and L using the formulas in
    /// Solver Abstraction SS2.1–SS2.2. The constructor is infallible;
    /// validation of N and L is the caller's responsibility.
    ///
    /// Equipment column ranges (`turbine`, `spillage`, `thermal`, `line_fwd`,
    /// `line_rev`, `deficit`, `excess`) are all empty (`0..0`) and equipment
    /// counts (`n_blks`, `n_thermals`, `n_lines`, `n_buses`) are zero. Use
    /// [`StageIndexer::with_equipment`] to populate them.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_sddp::indexer::StageIndexer;
    ///
    /// // Worked example from spec SS5.5.3: N = 3, L = 2
    /// let idx = StageIndexer::new(3, 2);
    /// assert_eq!(idx.storage,   0..3);
    /// assert_eq!(idx.inflow_lags, 3..9);
    /// assert_eq!(idx.z_inflow,  9..12);
    /// assert_eq!(idx.storage_in, 12..15);
    /// assert_eq!(idx.theta,   15);
    /// assert_eq!(idx.n_state,  9);
    /// // State-fixing row ranges are permanent empty sentinels.
    /// assert_eq!(idx.storage_fixing, 0..0);
    /// assert_eq!(idx.lag_fixing, 0..0);
    /// // Equipment ranges are empty when built via `new`.
    /// assert!(idx.turbine.is_empty());
    /// assert_eq!(idx.n_blks, 0);
    /// // z_inflow rows start at row 0.
    /// assert_eq!(idx.z_inflow_rows, 0..3);
    /// assert_eq!(idx.z_inflow_row_start, 0);
    /// ```
    #[must_use]
    pub fn new(hydro_count: usize, max_par_order: usize) -> Self {
        let n = hydro_count;
        let l = max_par_order;

        let storage = 0..n;
        let inflow_lags = n..n * (1 + l);

        // z_inflow columns at fixed offset N*(1+L), immediately after lags
        // and before storage_in. This makes z_inflow stage-invariant.
        let z_inflow_start = n * (1 + l);
        let z_inflow = z_inflow_start..z_inflow_start + n;

        let storage_in = n * (2 + l)..n * (3 + l);
        let theta = n * (3 + l);
        let n_state = n * (1 + l);

        // State fixing uses column bounds; the row ranges are permanent empty
        // sentinels. The column ranges (storage, inflow_lags, anticipated_state)
        // carry the state-pinning semantics via set_col_bounds.
        let storage_fixing = 0..0;
        let lag_fixing = 0..0;

        // z_inflow rows start at row 0 — the first row block in the LP.
        let z_inflow_start_row = 0_usize;
        let z_inflow_rows = z_inflow_start_row..z_inflow_start_row + n;
        let z_inflow_row_start = z_inflow_start_row;

        Self {
            storage,
            inflow_lags,
            storage_in,
            theta,
            n_state,
            storage_fixing,
            lag_fixing,
            // Anticipated state block is empty when built via `new`;
            // callers that need it must use `with_equipment_and_evaporation`.
            anticipated_state: 0..0,
            anticipated_state_fixing: 0..0,
            n_anticipated: 0,
            k_max: 0,
            hydro_count,
            max_par_order,
            // Equipment ranges are empty until `with_equipment` is called.
            turbine: 0..0,
            spillage: 0..0,
            diversion: 0..0,
            thermal: 0..0,
            // Anticipated decision and state-out blocks are empty when built via `new`.
            anticipated_decision: 0..0,
            anticipated_state_out: 0..0,
            anticipated_lead_stages: Vec::new(),
            anticipated_thermal_indices: Vec::new(),
            anticipated_local_by_sys_pos: HashMap::new(),
            line_fwd: 0..0,
            line_rev: 0..0,
            deficit: 0..0,
            max_deficit_segments: 0,
            excess: 0..0,
            n_blks: 0,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            water_balance: 0..0,
            load_balance: 0..0,
            inflow_slack: 0..0,
            inflow_slack_rows: 0..0,
            has_inflow_penalty: false,
            generation: 0..0,
            n_fpha_hydros: 0,
            fpha_hydro_indices: Vec::new(),
            fpha_rows: Vec::new(),
            n_evap_hydros: 0,
            evap_hydro_indices: Vec::new(),
            evap_indices: Vec::new(),
            withdrawal_slack_neg: 0..0,
            withdrawal_slack_pos: 0..0,
            has_withdrawal: false,
            outflow_below_slack: 0..0,
            outflow_above_slack: 0..0,
            turbine_below_slack: 0..0,
            generation_below_slack: 0..0,
            min_outflow_rows: 0..0,
            max_outflow_rows: 0..0,
            min_turbine_rows: 0..0,
            min_generation_rows: 0..0,
            anticipated_fishing: 0..0,
            anticipated_fishing_start: 0,
            has_operational_violations: false,
            generic_constraint_rows: 0..0,
            generic_constraint_slack: 0..0,
            n_generic_constraints_active: 0,
            ncs_generation: 0..0,
            // Permanent `0..0` sentinel, mirroring `ncs_generation`: the live
            // per-stage pumping column layout is owned by `StageLayout`, not the
            // indexer. Inherited via `..base` by `with_equipment_and_evaporation`.
            pumping_flow: 0..0,
            z_inflow,
            z_inflow_rows,
            z_inflow_row_start,
            nonzero_state_indices: Vec::new(),
            state_to_lp_column_map: Vec::new(),
        }
    }

    /// Construct a [`StageIndexer`] with full equipment column ranges.
    ///
    /// Computes both the state-variable ranges (identical to [`StageIndexer::new`])
    /// and the equipment decision-variable ranges that follow `theta` in the LP.
    ///
    /// The equipment column layout matches `lp_builder.rs` exactly:
    ///
    /// ```text
    /// decision_start      = theta + 1
    /// turbine_start       = decision_start
    /// spillage_start      = turbine_start  + n_hydros * n_blks
    /// diversion_start     = spillage_start + n_hydros * n_blks
    /// thermal_start       = diversion_start + n_hydros * n_blks
    /// line_fwd_start      = thermal_start  + n_thermals * n_blks
    /// line_rev_start      = line_fwd_start + n_lines * n_blks
    /// deficit_start       = line_rev_start + n_lines * n_blks
    /// excess_start        = deficit_start  + n_buses * max_deficit_segments * n_blks
    /// inflow_slack_start  = excess_end  (only when has_inflow_penalty && hydro_count > 0)
    /// generation_start    = inflow_slack_end  (FPHA generation columns)
    /// evap_start          = generation_end  (3 columns per evaporation hydro, stage-level)
    /// ```
    ///
    /// FPHA generation columns come immediately after `inflow_slack` (or after
    /// `excess` when `has_inflow_penalty == false`), one column per FPHA hydro
    /// per block.  FPHA constraint rows are placed after `load_balance`.
    ///
    /// Evaporation columns (3 per evaporation hydro: evaporation outflow, `f_evap_plus`,
    /// `f_evap_minus`) are stage-level (not per-block) and come immediately after
    /// the FPHA generation columns.  Evaporation rows (1 per evaporation hydro)
    /// are placed after FPHA rows.
    ///
    /// # Notes
    ///
    /// This constructor assumes a **uniform block count across all stages**
    /// (i.e., all stages have the same `n_blks`). For the minimal viable
    /// solver this assumption holds; stages with heterogeneous block counts
    /// would require a per-stage indexer.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_sddp::indexer::StageIndexer;
    ///
    /// // N=1 hydro, L=0 lags, T=2 thermals, L_n=1 line, B=2 buses, K=1 block, no penalty
    /// // theta = N*(3+L) = 1*(3+0) = 3
    /// // decision_start = 4
    /// // turbine:    4..5   (1 hydro * 1 block)
    /// // spillage:   5..6   (1 hydro * 1 block)
    /// // diversion:  6..7   (1 hydro * 1 block)
    /// // thermal:    7..9   (2 thermals * 1 block)
    /// // line_fwd:   9..10  (1 line * 1 block)
    /// // line_rev:  10..11  (1 line * 1 block)
    /// // deficit:   11..13  (2 buses * 1 block)
    /// // excess:    13..15  (2 buses * 1 block)
    /// let counts = cobre_sddp::indexer::EquipmentCounts {
    ///     hydro_count: 1, max_par_order: 0, n_thermals: 2, n_lines: 1,
    ///     n_buses: 2, n_blks: 1, has_inflow_penalty: false, max_deficit_segments: 1,
    ///     n_anticipated: 0, k_max: 0, n_pumping: 0,
    ///     anticipated_lead_stages: vec![], anticipated_thermal_indices: vec![],
    /// };
    /// let fpha = cobre_sddp::indexer::FphaColumnLayout { hydro_indices: vec![], planes_per_hydro: vec![] };
    /// let idx = StageIndexer::with_equipment(&counts, &fpha);
    /// assert_eq!(idx.turbine,    4..5);
    /// assert_eq!(idx.spillage,   5..6);
    /// assert_eq!(idx.diversion,  6..7);
    /// assert_eq!(idx.thermal,    7..9);
    /// assert_eq!(idx.line_fwd,   9..10);
    /// assert_eq!(idx.line_rev,  10..11);
    /// assert_eq!(idx.deficit,   11..13);
    /// assert_eq!(idx.excess,    13..15);
    /// assert!(idx.inflow_slack.is_empty());
    /// assert!(idx.generation.is_empty());
    /// assert_eq!(idx.n_blks, 1);
    /// assert_eq!(idx.n_thermals, 2);
    /// assert_eq!(idx.n_lines, 1);
    /// assert_eq!(idx.n_buses, 2);
    /// ```
    #[must_use]
    pub fn with_equipment(counts: &EquipmentCounts, fpha: &FphaColumnLayout) -> Self {
        Self::with_equipment_and_evaporation(
            counts,
            fpha,
            &EvapConfig {
                hydro_indices: vec![],
            },
        )
    }

    /// Construct a [`StageIndexer`] with full equipment column ranges and evaporation.
    ///
    /// Extends [`StageIndexer::with_equipment`] with evaporation hydro indices.
    /// Evaporation columns (3 per evaporation hydro: evaporation outflow, `f_evap_plus`,
    /// `f_evap_minus`) are stage-level and placed after FPHA generation columns.
    /// Evaporation rows (1 per evaporation hydro) are placed after FPHA rows.
    ///
    /// # Arguments
    ///
    /// - `counts` — equipment counts grouped into [`EquipmentCounts`]
    /// - `fpha` — FPHA column layout grouped into [`FphaColumnLayout`]
    /// - `evap` — evaporation configuration grouped into [`EvapConfig`]
    ///
    /// When `evap.hydro_indices` is empty this produces the same result as
    /// [`StageIndexer::with_equipment`].
    #[must_use]
    // Rationale: single cohesive LP column/row layout constructor; every local binding
    // contributes to the `Self { .. }` literal that closes the function.  Splitting into
    // sub-helpers would scatter the field-initialization order and obscure the one-shot
    // build contract where each offset derives directly from the previous.
    #[allow(clippy::too_many_lines)]
    pub fn with_equipment_and_evaporation(
        counts: &EquipmentCounts,
        fpha: &FphaColumnLayout,
        evap: &EvapConfig,
    ) -> Self {
        debug_assert!(
            counts.n_anticipated == 0 || counts.k_max >= 1,
            "k_max must be >= 1 when n_anticipated > 0"
        );
        debug_assert_eq!(
            counts.anticipated_lead_stages.len(),
            counts.n_anticipated,
            "anticipated_lead_stages length must equal n_anticipated"
        );
        debug_assert_eq!(
            counts.anticipated_thermal_indices.len(),
            counts.n_anticipated,
            "anticipated_thermal_indices length must equal n_anticipated"
        );
        debug_assert!(
            counts.n_anticipated == 0
                || counts
                    .anticipated_lead_stages
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0)
                    == counts.k_max,
            "k_max must equal max(anticipated_lead_stages)"
        );

        let hydro_count = counts.hydro_count;
        let max_par_order = counts.max_par_order;
        let n_thermals = counts.n_thermals;
        let n_lines = counts.n_lines;
        let n_buses = counts.n_buses;
        let n_blks = counts.n_blks;
        let has_inflow_penalty = counts.has_inflow_penalty;
        let n_anticipated = counts.n_anticipated;
        let k_max = counts.k_max;
        let n_ant_state = n_anticipated * k_max;
        let fpha_hydro_indices = fpha.hydro_indices.clone();
        let evap_hydro_indices = evap.hydro_indices.clone();

        debug_assert_eq!(
            fpha_hydro_indices.len(),
            fpha.planes_per_hydro.len(),
            "fpha_hydro_indices and fpha_planes_per_hydro must have equal length"
        );

        let base = Self::new(hydro_count, max_par_order);

        // Shift state-block downstream offsets by n_ant_state to make room for
        // the anticipated_state block placed immediately after `inflow_lags`.
        // The new layout:
        //   storage           = 0..N
        //   inflow_lags       = N..N*(1+L)
        //   anticipated_state = N*(1+L)..N*(1+L) + n_ant_state
        //   z_inflow          = + N
        //   storage_in        = + N
        //   theta             = N*(3+L) + n_ant_state
        //   n_state           = N*(1+L) + n_ant_state
        let n_state = hydro_count * (1 + max_par_order) + n_ant_state;
        let anticipated_state_start = hydro_count * (1 + max_par_order);
        let anticipated_state_end = anticipated_state_start + n_ant_state;
        // When the block is empty, normalise the public range to `0..0` to
        // mirror the project-wide convention used by `inflow_slack`,
        // `withdrawal_slack`, and the operational-violation ranges. Use
        // `anticipated_state_end` (not `anticipated_state.end`) for
        // downstream layout arithmetic so the shift is preserved even when
        // the public range collapses to `0..0`.
        let anticipated_state = if n_ant_state > 0 {
            anticipated_state_start..anticipated_state_end
        } else {
            0..0
        };
        // Anticipated-state fixing uses column bounds; row range is a permanent empty sentinel.
        let anticipated_state_fixing = 0..0;
        let z_inflow_start = anticipated_state_end;
        let z_inflow = z_inflow_start..z_inflow_start + hydro_count;
        let storage_in_start = z_inflow.end;
        let storage_in = storage_in_start..storage_in_start + hydro_count;
        let theta = storage_in.end;
        // z_inflow rows start at row 0 (ROW start, not column start).
        let z_inflow_start_row = 0_usize;
        let z_inflow_rows = z_inflow_start_row..z_inflow_start_row + hydro_count;
        let z_inflow_row_start = z_inflow_start_row;

        let decision_start = theta + 1;

        let turbine_start = decision_start;
        let spillage_start = turbine_start + hydro_count * n_blks;
        let diversion_start = spillage_start + hydro_count * n_blks;
        let thermal_start = diversion_start + hydro_count * n_blks;
        let thermal_end = thermal_start + n_thermals * n_blks;
        // Anticipated-decision and anticipated-state-out columns sit between
        // `thermal` and `line_fwd`.  Per Decision 3, every anticipated plant
        // has K_i <= T, so at stage 0 (the canonical stage) all `n_anticipated`
        // columns are active.  Per-stage gating is bound-driven downstream via
        // `anticipated_decision_active_at_stage`; the column count is constant.
        //
        // Control-region layout (equipment side):
        //   anticipated_decision   = [thermal_end, thermal_end + A)
        //   anticipated_state_out  = [thermal_end + A, thermal_end + 2*A)
        //   line_fwd               = [thermal_end + 2*A, …)
        let anticipated_decision_start = thermal_end;
        let anticipated_decision_end = thermal_end + n_anticipated;
        let anticipated_decision = if n_anticipated > 0 {
            anticipated_decision_start..anticipated_decision_end
        } else {
            0..0
        };
        let anticipated_state_out_start = anticipated_decision_end;
        let anticipated_state_out_end = anticipated_state_out_start + n_anticipated;
        let anticipated_state_out = if n_anticipated > 0 {
            anticipated_state_out_start..anticipated_state_out_end
        } else {
            0..0
        };
        let line_fwd_start = anticipated_state_out_end;
        let line_rev_start = line_fwd_start + n_lines * n_blks;
        let deficit_start = line_rev_start + n_lines * n_blks;
        let max_deficit_segments = counts.max_deficit_segments;
        let excess_start = deficit_start + n_buses * max_deficit_segments * n_blks;
        let excess_end = excess_start + n_buses * n_blks;

        // Inflow slack columns are appended after excess when penalty is active.
        let (inflow_slack, active_penalty) =
            build_inflow_slack_range(has_inflow_penalty, hydro_count, excess_end);

        // FPHA generation columns: after inflow_slack (or excess when no penalty).
        let n_fpha_hydros = fpha_hydro_indices.len();
        let generation_start = if active_penalty {
            inflow_slack.end
        } else {
            excess_end
        };
        let generation_end = generation_start + n_fpha_hydros * n_blks;
        let generation = if n_fpha_hydros > 0 {
            generation_start..generation_end
        } else {
            0..0
        };

        // Evaporation columns: EVAP_COLS_PER_HYDRO per evap hydro (stage-level),
        // after FPHA generation columns. Within-hydro layout (see build_evap_indices):
        // evaporation_flow_col at EVAP_FLOW_OFFSET, f_evap_plus at EVAP_F_PLUS_OFFSET,
        // f_evap_minus at EVAP_F_MINUS_OFFSET.
        let n_evap_hydros = evap_hydro_indices.len();
        let evap_col_start = generation_end;

        // z_inflow rows start at row 0 (z_inflow_start_row declared above);
        // water_balance follows z_inflow at row hydro_count.
        let water_balance_start = z_inflow_start_row + hydro_count;
        let load_balance_start = water_balance_start + hydro_count;
        let load_balance_end = load_balance_start + n_buses * n_blks;

        let (fpha_rows, fpha_row_cursor) =
            Self::build_fpha_rows(&fpha.planes_per_hydro, n_blks, load_balance_end);

        let evap_indices_vec =
            Self::build_evap_indices(n_evap_hydros, evap_col_start, fpha_row_cursor);
        let evap_col_end = evap_col_start + n_evap_hydros * EVAP_COLS_PER_HYDRO;
        let (withdrawal_slack_neg, withdrawal_slack_pos, has_withdrawal) = if hydro_count > 0 {
            let neg = evap_col_end..evap_col_end + hydro_count;
            let pos = neg.end..neg.end + hydro_count;
            (neg, pos, true)
        } else {
            (0..0, 0..0, false)
        };

        // Operational violation slack columns: 4 families * (hydro_count * n_blks).
        // Columns are placed after withdrawal slack; rows after evaporation rows.
        let evap_rows_end = fpha_row_cursor + n_evap_hydros;
        let ws_end = withdrawal_slack_pos.end;
        let op = build_oper_violation_ranges(hydro_count, n_blks, ws_end, evap_rows_end);

        // Anticipated-fishing rows are placed after the operational-violation
        // rows when those are active, otherwise directly after the evaporation
        // rows. The stage-0 canonical layout stores a zero-length range; the
        // per-stage template populates
        // `anticipated_fishing_start + local_idx_at_stage`.
        let fishing_start = if op.has_operational_violations {
            op.min_generation_rows.end
        } else {
            evap_rows_end
        };

        // z_inflow / storage_in / theta / n_state are computed locally above
        // to absorb the anticipated_state shift; the remaining fields
        // (storage, inflow_lags, storage_fixing, lag_fixing, hydro_count,
        // max_par_order, nonzero_state_indices) are inherited from `base`.

        Self {
            turbine: turbine_start..spillage_start,
            spillage: spillage_start..diversion_start,
            diversion: diversion_start..thermal_start,
            thermal: thermal_start..thermal_end,
            anticipated_decision,
            anticipated_state_out,
            anticipated_lead_stages: counts.anticipated_lead_stages.clone(),
            anticipated_local_by_sys_pos: counts
                .anticipated_thermal_indices
                .iter()
                .enumerate()
                .map(|(local, &sys_pos)| (sys_pos, local))
                .collect(),
            anticipated_thermal_indices: counts.anticipated_thermal_indices.clone(),
            line_fwd: line_fwd_start..line_rev_start,
            line_rev: line_rev_start..deficit_start,
            deficit: deficit_start..excess_start,
            max_deficit_segments,
            excess: excess_start..excess_end,
            n_blks,
            n_thermals,
            n_lines,
            n_buses,
            water_balance: water_balance_start..water_balance_start + hydro_count,
            load_balance: load_balance_start..load_balance_end,
            inflow_slack,
            inflow_slack_rows: 0..0,
            has_inflow_penalty: active_penalty,
            generation,
            n_fpha_hydros,
            fpha_hydro_indices,
            fpha_rows,
            n_evap_hydros,
            evap_hydro_indices,
            evap_indices: evap_indices_vec,
            withdrawal_slack_neg,
            withdrawal_slack_pos,
            has_withdrawal,
            outflow_below_slack: op.outflow_below_slack,
            outflow_above_slack: op.outflow_above_slack,
            turbine_below_slack: op.turbine_below_slack,
            generation_below_slack: op.generation_below_slack,
            min_outflow_rows: op.min_outflow_rows,
            max_outflow_rows: op.max_outflow_rows,
            min_turbine_rows: op.min_turbine_rows,
            min_generation_rows: op.min_generation_rows,
            anticipated_fishing: fishing_start..fishing_start,
            anticipated_fishing_start: fishing_start,
            has_operational_violations: op.has_operational_violations,
            // Shifted state-block fields (recomputed to absorb n_ant_state).
            anticipated_state,
            anticipated_state_fixing,
            n_anticipated,
            k_max,
            n_state,
            storage_in,
            theta,
            z_inflow,
            z_inflow_rows,
            z_inflow_row_start,
            // Built empty here; filled by `finalize_state_column_map` once the
            // (shifted) state layout above is final.
            state_to_lp_column_map: Vec::new(),
            // Remaining state-block fields are unchanged; inherit from base.
            ..base
        }
    }

    /// Build FPHA constraint row ranges from per-hydro plane counts.
    fn build_fpha_rows(
        planes_per_hydro: &[usize],
        n_blks: usize,
        start_row: usize,
    ) -> (Vec<FphaRowRange>, usize) {
        let mut rows = Vec::with_capacity(planes_per_hydro.len());
        let mut cursor = start_row;
        for &planes in planes_per_hydro {
            rows.push(FphaRowRange {
                start: cursor,
                planes_per_block: planes,
            });
            cursor += planes * n_blks;
        }
        (rows, cursor)
    }

    /// Build evaporation column/row indices for each evaporation hydro.
    fn build_evap_indices(
        n_evap_hydros: usize,
        col_start: usize,
        row_start: usize,
    ) -> Vec<EvaporationIndices> {
        (0..n_evap_hydros)
            .map(|i| EvaporationIndices {
                evaporation_flow_col: col_start + i * EVAP_COLS_PER_HYDRO + EVAP_FLOW_OFFSET,
                f_evap_plus_col: col_start + i * EVAP_COLS_PER_HYDRO + EVAP_F_PLUS_OFFSET,
                f_evap_minus_col: col_start + i * EVAP_COLS_PER_HYDRO + EVAP_F_MINUS_OFFSET,
                evap_row: row_start + i,
            })
            .collect()
    }

    /// Construct a [`StageIndexer`] from a [`StageTemplate`].
    ///
    /// Extracts `n_hydro` and `max_par_order` from the template and delegates
    /// to [`StageIndexer::new`]. Produces identical results to calling
    /// `StageIndexer::new(template.n_hydro, template.max_par_order)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_sddp::indexer::StageIndexer;
    /// use cobre_solver::StageTemplate;
    ///
    /// let template = StageTemplate {
    ///     num_cols: 16,
    ///     num_rows: 12,
    ///     num_nz: 0,
    ///     col_starts: vec![0_i32; 17],
    ///     row_indices: vec![],
    ///     values: vec![],
    ///     col_lower: vec![0.0; 16],
    ///     col_upper: vec![f64::INFINITY; 16],
    ///     objective: vec![0.0; 16],
    ///     row_lower: vec![0.0; 12],
    ///     row_upper: vec![f64::INFINITY; 12],
    ///     n_state: 9,
    ///     n_transfer: 6,
    ///     n_dual_relevant: 9,
    ///     n_hydro: 3,
    ///     max_par_order: 2,
    ///     col_scale: vec![],
    ///     row_scale: vec![],
    /// };
    ///
    /// let idx = StageIndexer::from_stage_template(&template);
    /// assert_eq!(idx.storage, 0..3);
    /// assert_eq!(idx.theta,  15);
    /// ```
    #[must_use]
    pub fn from_stage_template(template: &StageTemplate) -> Self {
        Self::new(template.n_hydro, template.max_par_order)
    }
}

#[cfg(test)]
mod tests {
    use cobre_solver::StageTemplate;

    use crate::indexer::test_fixtures::{eq, eq_with_anticipated, evap, fpha};
    use crate::indexer::{EquipmentCounts, StageIndexer};

    // Worked example from spec SS5.5.3: N = 3, L = 2

    fn indexer_3_2() -> StageIndexer {
        StageIndexer::new(3, 2)
    }

    #[test]
    fn storage_range_3_2() {
        assert_eq!(indexer_3_2().storage, 0..3);
    }

    #[test]
    fn inflow_lags_range_3_2() {
        assert_eq!(indexer_3_2().inflow_lags, 3..9);
    }

    #[test]
    fn z_inflow_range_3_2() {
        assert_eq!(indexer_3_2().z_inflow, 9..12);
    }

    #[test]
    fn storage_in_range_3_2() {
        assert_eq!(indexer_3_2().storage_in, 12..15);
    }

    #[test]
    fn theta_index_3_2() {
        assert_eq!(indexer_3_2().theta, 15);
    }

    #[test]
    fn n_state_3_2() {
        assert_eq!(indexer_3_2().n_state, 9);
    }

    #[test]
    fn storage_fixing_range_3_2() {
        // storage_fixing is a permanent empty sentinel.
        assert_eq!(indexer_3_2().storage_fixing, 0..0);
    }

    #[test]
    fn lag_fixing_range_3_2() {
        // lag_fixing is a permanent empty sentinel.
        assert_eq!(indexer_3_2().lag_fixing, 0..0);
    }

    #[test]
    fn row_column_symmetry_3_2() {
        // State-fixing row ranges are permanent empty sentinels; column ranges are unchanged.
        let idx = indexer_3_2();
        assert_eq!(idx.storage_fixing, 0..0);
        assert_eq!(idx.lag_fixing, 0..0);
        // Column ranges are unchanged.
        assert_eq!(idx.storage, 0..3);
        assert_eq!(idx.inflow_lags, 3..9);
    }

    // Production scale: N = 160, L = 12

    fn indexer_160_12() -> StageIndexer {
        StageIndexer::new(160, 12)
    }

    #[test]
    fn n_state_production_scale() {
        assert_eq!(indexer_160_12().n_state, 2080);
    }

    #[test]
    fn theta_production_scale() {
        assert_eq!(indexer_160_12().theta, 2400);
    }

    #[test]
    fn row_column_symmetry_production_scale() {
        // State-fixing row ranges are permanent empty sentinels; column ranges are unchanged.
        let idx = indexer_160_12();
        assert_eq!(idx.storage_fixing, 0..0);
        assert_eq!(idx.lag_fixing, 0..0);
        // Column ranges are unchanged.
        assert_eq!(idx.storage, 0..160);
        assert_eq!(idx.inflow_lags, 160..160 * 13);
    }

    #[test]
    fn single_hydro_no_lags() {
        let idx = StageIndexer::new(1, 0);

        assert_eq!(idx.storage, 0..1);
        assert_eq!(idx.inflow_lags, 1..1);
        assert_eq!(idx.z_inflow, 1..2);
        assert_eq!(idx.storage_in, 2..3);
        assert_eq!(idx.theta, 3);
        assert_eq!(idx.n_state, 1);
        // storage_fixing and lag_fixing are permanent empty sentinels.
        assert_eq!(idx.storage_fixing, 0..0);
        assert_eq!(idx.lag_fixing, 0..0);
    }

    // Edge case: N = 0, L = 0 (degenerate — all ranges empty)

    #[test]
    fn degenerate_zero_hydros() {
        let idx = StageIndexer::new(0, 0);

        assert_eq!(idx.storage, 0..0);
        assert_eq!(idx.inflow_lags, 0..0);
        assert_eq!(idx.z_inflow, 0..0);
        assert_eq!(idx.storage_in, 0..0);
        assert_eq!(idx.theta, 0);
        assert_eq!(idx.n_state, 0);
        assert_eq!(idx.storage_fixing, 0..0);
        assert_eq!(idx.lag_fixing, 0..0);

        assert_eq!(idx.storage_fixing, idx.storage);
        assert_eq!(idx.lag_fixing, idx.inflow_lags);
    }

    // from_stage_template: must produce the same result as new()

    fn make_template(n_hydro: usize, max_par_order: usize) -> StageTemplate {
        let n_state = n_hydro * (1 + max_par_order);
        let n_transfer = n_hydro * max_par_order;
        // Minimal valid template (matrix contents are irrelevant for indexer)
        StageTemplate {
            num_cols: 0,
            num_rows: 0,
            num_nz: 0,
            col_starts: vec![0_i32],
            row_indices: vec![],
            values: vec![],
            col_lower: vec![],
            col_upper: vec![],
            objective: vec![],
            row_lower: vec![],
            row_upper: vec![],
            n_state,
            n_transfer,
            n_dual_relevant: n_state,
            n_hydro,
            max_par_order,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    #[test]
    fn from_stage_template_matches_new_3_2() {
        let tmpl = make_template(3, 2);
        let from_tmpl = StageIndexer::from_stage_template(&tmpl);
        let from_new = StageIndexer::new(3, 2);

        assert_eq!(from_tmpl.storage, from_new.storage);
        assert_eq!(from_tmpl.inflow_lags, from_new.inflow_lags);
        assert_eq!(from_tmpl.storage_in, from_new.storage_in);
        assert_eq!(from_tmpl.theta, from_new.theta);
        assert_eq!(from_tmpl.n_state, from_new.n_state);
        assert_eq!(from_tmpl.storage_fixing, from_new.storage_fixing);
        assert_eq!(from_tmpl.lag_fixing, from_new.lag_fixing);
        assert_eq!(from_tmpl.hydro_count, from_new.hydro_count);
        assert_eq!(from_tmpl.max_par_order, from_new.max_par_order);
    }

    #[test]
    fn from_stage_template_matches_new_160_12() {
        let tmpl = make_template(160, 12);
        let from_tmpl = StageIndexer::from_stage_template(&tmpl);
        let from_new = StageIndexer::new(160, 12);

        assert_eq!(from_tmpl.n_state, from_new.n_state);
        assert_eq!(from_tmpl.theta, from_new.theta);
        assert_eq!(from_tmpl.hydro_count, from_new.hydro_count);
        assert_eq!(from_tmpl.max_par_order, from_new.max_par_order);
    }

    #[test]
    fn from_stage_template_matches_new_edge_cases() {
        for (n, l) in [(0, 0), (1, 0), (1, 1)] {
            let tmpl = make_template(n, l);
            let from_tmpl = StageIndexer::from_stage_template(&tmpl);
            let from_new = StageIndexer::new(n, l);

            assert_eq!(from_tmpl.storage, from_new.storage, "N={n} L={l}");
            assert_eq!(from_tmpl.inflow_lags, from_new.inflow_lags, "N={n} L={l}");
            assert_eq!(from_tmpl.theta, from_new.theta, "N={n} L={l}");
            assert_eq!(from_tmpl.n_state, from_new.n_state, "N={n} L={l}");
        }
    }

    // new() produces empty equipment ranges
    #[test]
    fn new_equipment_ranges_are_empty() {
        let idx = StageIndexer::new(3, 2);
        assert!(idx.turbine.is_empty());
        assert!(idx.spillage.is_empty());
        assert!(idx.diversion.is_empty());
        assert!(idx.thermal.is_empty());
        assert!(idx.line_fwd.is_empty());
        assert!(idx.line_rev.is_empty());
        assert!(idx.deficit.is_empty());
        assert!(idx.excess.is_empty());
        assert_eq!(idx.n_blks, 0);
        assert_eq!(idx.n_thermals, 0);
        assert_eq!(idx.n_lines, 0);
        assert_eq!(idx.n_buses, 0);
    }

    // with_equipment: worked example from doc comment (N=1, L=0, T=2, Ln=1, B=2, K=1)
    //
    // theta = N*(3+L) = 1*(3+0) = 3
    // decision_start = 4
    // turbine:    [4, 4+1*1)  = 4..5
    // spillage:   [5, 5+1*1)  = 5..6
    // diversion:  [6, 6+1*1)  = 6..7
    // thermal:    [7, 7+2*1)  = 7..9
    // line_fwd:   [9, 9+1*1)  = 9..10
    // line_rev:  [10,10+1*1)  = 10..11
    // deficit:   [11,11+2*1)  = 11..13
    // excess:    [13,13+2*1)  = 13..15
    #[test]
    fn with_equipment_doctest_n1_l0_t2_l1_b2_k1() {
        let idx = StageIndexer::with_equipment(&eq(1, 0, 2, 1, 2, 1, false), &fpha(vec![], vec![]));

        // State ranges are identical to new(1, 0)
        assert_eq!(idx.storage, 0..1);
        assert_eq!(idx.inflow_lags, 1..1);
        assert_eq!(idx.z_inflow, 1..2);
        assert_eq!(idx.storage_in, 2..3);
        assert_eq!(idx.theta, 3);
        assert_eq!(idx.n_state, 1);

        // Equipment ranges
        assert_eq!(idx.turbine, 4..5);
        assert_eq!(idx.spillage, 5..6);
        assert_eq!(idx.diversion, 6..7);
        assert_eq!(idx.thermal, 7..9);
        assert_eq!(idx.line_fwd, 9..10);
        assert_eq!(idx.line_rev, 10..11);
        assert_eq!(idx.deficit, 11..13);
        assert_eq!(idx.excess, 13..15);

        // Equipment counts
        assert_eq!(idx.n_blks, 1);
        assert_eq!(idx.n_thermals, 2);
        assert_eq!(idx.n_lines, 1);
        assert_eq!(idx.n_buses, 2);
    }

    // with_equipment: N=2, L=1, T=3, Ln=2, B=4, K=2
    //
    // theta = N*(3+L) = 2*(3+1) = 8
    // decision_start = 9
    // turbine:    [9,  9+2*2)  = 9..13
    // spillage:  [13, 13+2*2)  = 13..17
    // diversion: [17, 17+2*2)  = 17..21
    // thermal:   [21, 21+3*2)  = 21..27
    // line_fwd:  [27, 27+2*2)  = 27..31
    // line_rev:  [31, 31+2*2)  = 31..35
    // deficit:   [35, 35+4*2)  = 35..43
    // excess:    [43, 43+4*2)  = 43..51
    #[test]
    fn with_equipment_n2_l1_t3_l2_b4_k2() {
        let idx = StageIndexer::with_equipment(&eq(2, 1, 3, 2, 4, 2, false), &fpha(vec![], vec![]));

        // State ranges identical to new(2, 1)
        assert_eq!(idx.theta, 8);
        assert_eq!(idx.n_state, 4); // N*(1+L) = 2*2 = 4

        // Equipment ranges
        assert_eq!(idx.turbine, 9..13);
        assert_eq!(idx.spillage, 13..17);
        assert_eq!(idx.diversion, 17..21);
        assert_eq!(idx.thermal, 21..27);
        assert_eq!(idx.line_fwd, 27..31);
        assert_eq!(idx.line_rev, 31..35);
        assert_eq!(idx.deficit, 35..43);
        assert_eq!(idx.excess, 43..51);
    }

    // with_equipment: no equipment (all counts zero), matches new() state layout
    #[test]
    fn with_equipment_all_counts_zero_matches_new() {
        let with_eq =
            StageIndexer::with_equipment(&eq(3, 2, 0, 0, 0, 0, false), &fpha(vec![], vec![]));
        let base = StageIndexer::new(3, 2);

        assert_eq!(with_eq.storage, base.storage);
        assert_eq!(with_eq.inflow_lags, base.inflow_lags);
        assert_eq!(with_eq.storage_in, base.storage_in);
        assert_eq!(with_eq.theta, base.theta);
        assert_eq!(with_eq.n_state, base.n_state);
        // All equipment ranges empty
        assert!(with_eq.turbine.is_empty());
        assert!(with_eq.spillage.is_empty());
        assert!(with_eq.diversion.is_empty());
        assert!(with_eq.thermal.is_empty());
        assert!(with_eq.line_fwd.is_empty());
        assert!(with_eq.line_rev.is_empty());
        assert!(with_eq.deficit.is_empty());
        assert!(with_eq.excess.is_empty());
    }

    // with_equipment: adjacency invariant — ranges must be contiguous and non-overlapping
    #[test]
    fn with_equipment_ranges_are_contiguous() {
        let idx = StageIndexer::with_equipment(&eq(2, 1, 3, 2, 4, 2, false), &fpha(vec![], vec![]));

        // turbine immediately follows theta
        assert_eq!(idx.turbine.start, idx.theta + 1);
        // each range starts where the previous ends
        assert_eq!(idx.spillage.start, idx.turbine.end);
        assert_eq!(idx.diversion.start, idx.spillage.end);
        assert_eq!(idx.thermal.start, idx.diversion.end);
        assert_eq!(idx.line_fwd.start, idx.thermal.end);
        assert_eq!(idx.line_rev.start, idx.line_fwd.end);
        assert_eq!(idx.deficit.start, idx.line_rev.end);
        assert_eq!(idx.excess.start, idx.deficit.end);
    }

    // Column index formula: turbine[h, b] = turbine.start + h * n_blks + b
    #[test]
    fn with_equipment_column_index_formulas() {
        let n_blks = 3_usize;
        let idx =
            StageIndexer::with_equipment(&eq(2, 1, 1, 1, 1, n_blks, false), &fpha(vec![], vec![]));

        // turbine[h=0, b=0] = turbine.start (no offset for h=0, b=0)
        assert_eq!(idx.turbine.start, idx.turbine.start);
        // turbine[h=1, b=2] = turbine.start + 1*3 + 2 = turbine.start + 5
        assert_eq!(idx.turbine.start + n_blks + 2, idx.turbine.start + 5);
        // deficit[b_idx=0, blk=1] = deficit.start + 1
        assert_eq!(idx.deficit.start + 1, idx.deficit.start + 1);
        // turbine[h=1, b=0] = turbine.start + n_blks
        assert_eq!(idx.turbine.start + n_blks, idx.turbine.start + 3);
    }

    // with_equipment: has_inflow_penalty=true appends N slack columns after excess
    //
    // N=2, L=1, T=1, Ln=1, B=1, K=1, penalty=true
    // theta = N*(3+L) = 2*(3+1) = 8
    // decision_start = 9
    // turbine:    [9,  11)
    // spillage:  [11, 13)
    // diversion: [13, 15)
    // thermal:   [15, 16)
    // line_fwd:  [16, 17)
    // line_rev:  [17, 18)
    // deficit:   [18, 19)
    // excess:    [19, 20)
    // inflow_slack: [20, 22)  <- excess_end..excess_end+N
    #[test]
    fn with_equipment_inflow_penalty_appends_slack() {
        let idx = StageIndexer::with_equipment(&eq(2, 1, 1, 1, 1, 1, true), &fpha(vec![], vec![]));

        assert!(idx.has_inflow_penalty, "has_inflow_penalty must be true");
        // inflow_slack must start exactly where excess ends
        assert_eq!(
            idx.inflow_slack.start, idx.excess.end,
            "inflow_slack.start must equal excess.end (contiguous)"
        );
        // inflow_slack must contain exactly hydro_count columns
        assert_eq!(
            idx.inflow_slack.len(),
            idx.hydro_count,
            "inflow_slack must contain exactly hydro_count columns"
        );
        assert_eq!(idx.inflow_slack, 20..22);
        // inflow_slack_rows stays empty in this implementation
        assert!(
            idx.inflow_slack_rows.is_empty(),
            "inflow_slack_rows must remain empty"
        );
        // without penalty the slack range is empty
        let no_penalty =
            StageIndexer::with_equipment(&eq(2, 1, 1, 1, 1, 1, false), &fpha(vec![], vec![]));
        assert!(!no_penalty.has_inflow_penalty);
        assert!(no_penalty.inflow_slack.is_empty());
    }

    // ── FPHA field tests ───────────────────────────────────────────────────

    // AC-4: no FPHA hydros → generation is empty, fpha_rows is empty.
    //
    // N=4, L=0, T=0, Ln=0, B=1, K=1, no penalty, no FPHA.
    // theta = N*(3+L) = 4*(3+0) = 12
    // decision_start = 13
    // turbine:  [13, 17)
    // spillage: [17, 21)
    // deficit:  [21, 22)
    // excess:   [22, 23)
    // generation: empty (no FPHA hydros)
    #[test]
    fn fpha_no_hydros_generation_is_empty() {
        let idx = StageIndexer::with_equipment(&eq(4, 0, 0, 0, 1, 1, false), &fpha(vec![], vec![]));

        assert!(
            idx.generation.is_empty(),
            "generation must be empty with no FPHA hydros"
        );
        assert_eq!(idx.n_fpha_hydros, 0);
        assert!(idx.fpha_hydro_indices.is_empty());
        assert!(idx.fpha_rows.is_empty());
    }

    // AC-1 + AC-2: 1 FPHA hydro, 1 block, 3 planes.
    //
    // N=2, L=0, T=1, Ln=0, B=1, K=1, no penalty.
    // theta = N*(3+L) = 2*(3+0) = 6
    // decision_start = 7
    // turbine:    [7, 9)   (2 hydros * 1 block)
    // spillage:   [9, 11)
    // diversion: [11, 13)  (2 hydros * 1 block)
    // thermal:   [13, 14)  (1 thermal * 1 block)
    // deficit:   [14, 15)  (1 bus * 1 block)
    // excess:    [15, 16)
    // generation: [16, 17) (1 FPHA hydro * 1 block)
    //
    // Row layout:
    // n_state = N*(1+L) = 2*(1+0) = 2
    // z_inflow rows = 2..4  (N*(1+L)..N*(2+L))
    // water_balance_start = N*(2+L) = 4
    // load_balance_start  = 4 + 2 = 6
    // load_balance_end    = 6 + 1*1 = 7
    // fpha_rows[0].start  = 7 (after load_balance.end)
    // fpha_rows[0].planes_per_block = 3
    #[test]
    fn fpha_one_hydro_one_block_three_planes() {
        let idx =
            StageIndexer::with_equipment(&eq(2, 0, 1, 0, 1, 1, false), &fpha(vec![0], vec![3]));

        // AC-1: generation spans 1 column (1 FPHA hydro * 1 block)
        assert_eq!(idx.generation.len(), 1, "generation must span 1 column");
        assert_eq!(idx.generation, 16..17);
        assert_eq!(idx.n_fpha_hydros, 1);
        assert_eq!(idx.fpha_hydro_indices, vec![0]);

        // AC-2: fpha_rows[0].start is after load_balance.end, planes_per_block == 3
        assert_eq!(idx.fpha_rows.len(), 1);
        assert_eq!(
            idx.fpha_rows[0].start, idx.load_balance.end,
            "fpha_rows[0].start must equal load_balance.end"
        );
        assert_eq!(idx.fpha_rows[0].planes_per_block, 3);
    }

    // AC-3: 2 FPHA hydros, 2 blocks, plane counts [5, 4].
    //
    // N=4, L=0, T=0, Ln=0, B=1, K=2, no penalty.
    // theta = N*(3+L) = 4*(3+0) = 12
    // decision_start = 13
    // turbine:    [13, 21)  (4 hydros * 2 blocks)
    // spillage:   [21, 29)
    // diversion:  [29, 37)  (4 hydros * 2 blocks)
    // deficit:    [37, 39)  (1 bus * 2 blocks)
    // excess:     [39, 41)
    // generation: [41, 45) (2 FPHA hydros * 2 blocks = 4 columns)
    #[test]
    fn fpha_two_hydros_two_blocks_different_planes() {
        let idx = StageIndexer::with_equipment(
            &eq(4, 0, 0, 0, 1, 2, false),
            &fpha(vec![1, 3], vec![5, 4]),
        );

        // AC-3: generation spans 4 columns (2 FPHA hydros * 2 blocks)
        assert_eq!(idx.generation.len(), 4, "generation must span 4 columns");
        assert_eq!(idx.n_fpha_hydros, 2);
        assert_eq!(idx.fpha_hydro_indices, vec![1, 3]);

        // fpha_rows: 2 entries with correct starts and plane counts
        assert_eq!(idx.fpha_rows.len(), 2);

        // fpha_rows[0]: hydro at local 0 (system hydro 1), 5 planes, 2 blocks
        // starts at load_balance.end
        assert_eq!(
            idx.fpha_rows[0].start, idx.load_balance.end,
            "fpha_rows[0].start must equal load_balance.end"
        );
        assert_eq!(idx.fpha_rows[0].planes_per_block, 5);

        // fpha_rows[1]: starts after fpha_rows[0]'s region (5 planes * 2 blocks = 10 rows)
        assert_eq!(
            idx.fpha_rows[1].start,
            idx.fpha_rows[0].start + 5 * 2,
            "fpha_rows[1].start must follow fpha_rows[0]'s 10-row region"
        );
        assert_eq!(idx.fpha_rows[1].planes_per_block, 4);
    }

    // FPHA generation columns are contiguous with the prior column region.
    //
    // No penalty: generation immediately follows excess.
    // With penalty: generation immediately follows inflow_slack.
    #[test]
    fn fpha_generation_contiguous_with_prior_region() {
        // No penalty case: generation.start == excess.end
        let no_penalty =
            StageIndexer::with_equipment(&eq(2, 0, 0, 0, 1, 1, false), &fpha(vec![0], vec![2]));
        assert_eq!(
            no_penalty.generation.start, no_penalty.excess.end,
            "generation.start must equal excess.end when no penalty"
        );

        // With penalty case: generation.start == inflow_slack.end
        let with_penalty =
            StageIndexer::with_equipment(&eq(2, 0, 0, 0, 1, 1, true), &fpha(vec![0], vec![2]));
        assert_eq!(
            with_penalty.generation.start, with_penalty.inflow_slack.end,
            "generation.start must equal inflow_slack.end when penalty active"
        );
    }

    // FPHA rows are contiguous with load_balance (start at load_balance.end).
    #[test]
    fn fpha_rows_contiguous_with_load_balance() {
        let idx = StageIndexer::with_equipment(
            &eq(3, 1, 2, 0, 2, 3, false),
            &fpha(vec![0, 2], vec![4, 6]),
        );

        // First FPHA hydro starts at load_balance.end
        assert_eq!(
            idx.fpha_rows[0].start, idx.load_balance.end,
            "fpha_rows[0] must start at load_balance.end"
        );

        // Each subsequent FPHA hydro starts after its predecessor's block
        // fpha_rows[0]: 4 planes * 3 blocks = 12 rows
        assert_eq!(
            idx.fpha_rows[1].start,
            idx.fpha_rows[0].start + 4 * 3,
            "fpha_rows[1] must start after fpha_rows[0]'s rows"
        );
        assert_eq!(idx.fpha_rows[1].planes_per_block, 6);
    }

    // ── Evaporation field tests ────────────────────────────────────────────

    // 0 evaporation hydros → evap_indices is empty.
    #[test]
    fn evap_no_hydros_indices_empty() {
        let idx = StageIndexer::with_equipment(&eq(3, 0, 1, 0, 1, 1, false), &fpha(vec![], vec![]));

        assert_eq!(idx.n_evap_hydros, 0);
        assert!(idx.evap_hydro_indices.is_empty());
        assert!(idx.evap_indices.is_empty());
    }

    // 1 evaporation hydro — verify column/row positions.
    //
    // N=2, L=0, T=0, Ln=0, B=1, K=1, no penalty, no FPHA, 1 evap hydro.
    // theta = N*(3+L) = 2*(3+0) = 6
    // decision_start = 7
    // turbine:    [7, 9)   (2 hydros * 1 block)
    // spillage:   [9, 11)
    // diversion: [11, 13)  (2 hydros * 1 block)
    // deficit:   [13, 14)  (1 bus * 1 block)
    // excess:    [14, 15)
    // generation: empty (no FPHA)
    // evap cols: [15, 18)  (3 columns: evaporation outflow, f_evap_plus, f_evap_minus)
    //
    // Row layout (Phase 1: state-fixing rows removed):
    // z_inflow rows = 0..2  (start at row 0)
    // water_balance_start = 2  (z_inflow rows end)
    // load_balance_start = 2 + 2 = 4
    // load_balance_end   = 4 + 1*1 = 5
    // evap_row[0] = 5
    #[test]
    fn evap_one_hydro_column_row_positions() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(2, 0, 0, 0, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![0]),
        );

        assert_eq!(idx.n_evap_hydros, 1);
        assert_eq!(idx.evap_hydro_indices, vec![0]);
        assert_eq!(idx.evap_indices.len(), 1);

        let ei = idx.evap_indices(0);
        // 3 columns placed after generation_end (which equals excess.end = 15)
        assert_eq!(ei.evaporation_flow_col, 15);
        assert_eq!(ei.f_evap_plus_col, 16);
        assert_eq!(ei.f_evap_minus_col, 17);
        // Row placed after load_balance.end = 5.
        assert_eq!(ei.evap_row, 5);
    }

    // 2 evaporation hydros — verify column/row ranges are
    // contiguous and non-overlapping with FPHA ranges.
    //
    // N=4, L=0, T=0, Ln=0, B=1, K=1, no penalty, 1 FPHA hydro (index 0, 3 planes),
    // 2 evap hydros (indices 1, 2).
    // theta = 4*(3+0) = 12
    // decision_start = 13
    // turbine:    [13, 17)  (4 hydros * 1 block)
    // spillage:   [17, 21)
    // diversion:  [21, 25)  (4 hydros * 1 block)
    // deficit:    [25, 26)  (1 bus * 1 block)
    // excess:     [26, 27)
    // generation: [27, 28) (1 FPHA hydro * 1 block)
    //
    // Row layout (Phase 1: state-fixing rows removed):
    // z_inflow rows = 0..4  (start at row 0)
    // water_balance_start = 4  (z_inflow rows end)
    // load_balance_start = 4 + 4 = 8
    // load_balance_end   = 8 + 1*1 = 9
    // fpha_rows[0].start = 9
    // fpha_row_cursor after FPHA = 9 + 3*1 = 12
    // evap cols: [28, 34)   (2 evap hydros * 3 = 6 columns)
    // evap_row[0] = 12, evap_row[1] = 13
    #[test]
    fn evap_two_hydros_with_fpha_contiguous() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(4, 0, 0, 0, 1, 1, false),
            &fpha(vec![0], vec![3]),
            &evap(vec![1, 2]),
        );

        assert_eq!(idx.n_evap_hydros, 2);
        assert_eq!(idx.evap_hydro_indices, vec![1, 2]);

        let ei0 = idx.evap_indices(0);
        let ei1 = idx.evap_indices(1);

        // Columns start at generation_end = 28
        assert_eq!(ei0.evaporation_flow_col, 28);
        assert_eq!(ei0.f_evap_plus_col, 29);
        assert_eq!(ei0.f_evap_minus_col, 30);

        assert_eq!(ei1.evaporation_flow_col, 31);
        assert_eq!(ei1.f_evap_plus_col, 32);
        assert_eq!(ei1.f_evap_minus_col, 33);

        // Rows placed after fpha_rows region: fpha_row_cursor = 9 + 3*1 = 12.
        assert_eq!(ei0.evap_row, 12);
        assert_eq!(ei1.evap_row, 13);

        // Evap rows do not overlap FPHA rows
        assert!(ei0.evap_row > idx.fpha_rows[0].start);
    }

    // new() produces empty evaporation fields.
    #[test]
    fn new_evap_ranges_are_empty() {
        let idx = StageIndexer::new(3, 2);
        assert_eq!(idx.n_evap_hydros, 0);
        assert!(idx.evap_hydro_indices.is_empty());
        assert!(idx.evap_indices.is_empty());
    }

    // ── Withdrawal slack field tests ───────────────────────────────────────

    // AC: with_equipment_and_evaporation, N=3 hydros, 1 evap hydro →
    // withdrawal_slack_neg starts at evap_col_end, withdrawal_slack_pos follows.
    //
    // N=3, L=0, T=0, Ln=0, B=1, K=1, no penalty, no FPHA, 1 evap hydro.
    // theta = N*(3+L) = 3*(3+0) = 9
    // decision_start = 10
    // turbine:    [10, 13)  (3 hydros * 1 block)
    // spillage:   [13, 16)
    // diversion:  [16, 19)  (3 hydros * 1 block)
    // deficit:    [19, 20)  (1 bus * 1 block)
    // excess:     [20, 21)
    // generation: empty (no FPHA)
    // evap cols:  [21, 24)  (1 evap hydro * 3 columns)
    // withdrawal_slack_neg: [24, 27)  (3 hydros)
    // withdrawal_slack_pos: [27, 30)  (3 hydros)
    #[test]
    fn withdrawal_slack_with_equipment_and_evaporation_n3_evap1() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(3, 0, 0, 0, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![0]),
        );

        assert!(idx.has_withdrawal);
        let evap_col_end = idx.evap_indices(0).f_evap_minus_col + 1;
        assert_eq!(
            idx.withdrawal_slack_neg.start, evap_col_end,
            "withdrawal_slack_neg.start must equal evap_col_end"
        );
        assert_eq!(idx.withdrawal_slack_neg.len(), 3);
        assert_eq!(idx.withdrawal_slack_neg, 24..27);
        assert_eq!(idx.withdrawal_slack_pos.start, idx.withdrawal_slack_neg.end);
        assert_eq!(idx.withdrawal_slack_pos.len(), 3);
        assert_eq!(idx.withdrawal_slack_pos, 27..30);
    }

    // AC: with_equipment_and_evaporation, N=0 → both withdrawal slacks are 0..0.
    #[test]
    fn withdrawal_slack_zero_hydros_is_empty() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(0, 0, 0, 0, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert!(!idx.has_withdrawal);
        assert_eq!(idx.withdrawal_slack_neg, 0..0);
        assert_eq!(idx.withdrawal_slack_pos, 0..0);
    }

    // AC: new() → both withdrawal slacks are 0..0.
    #[test]
    fn withdrawal_slack_from_new_is_empty() {
        let idx = StageIndexer::new(3, 2);
        assert!(!idx.has_withdrawal);
        assert_eq!(idx.withdrawal_slack_neg, 0..0);
        assert_eq!(idx.withdrawal_slack_pos, 0..0);
    }

    // AC: both withdrawal_slack ranges have length == hydro_count.
    #[test]
    fn withdrawal_slack_length_equals_hydro_count() {
        for n in [1_usize, 5] {
            let idx = StageIndexer::with_equipment_and_evaporation(
                &EquipmentCounts {
                    hydro_count: n,
                    max_par_order: 0,
                    n_thermals: 0,
                    n_lines: 0,
                    n_buses: 1,
                    n_blks: 1,
                    has_inflow_penalty: false,
                    max_deficit_segments: 1,
                    n_anticipated: 0,
                    k_max: 0,
                    anticipated_lead_stages: vec![],
                    anticipated_thermal_indices: vec![],
                    n_pumping: 0,
                },
                &fpha(vec![], vec![]),
                &evap(vec![]),
            );

            assert!(idx.has_withdrawal, "has_withdrawal must be true for n={n}");
            assert_eq!(
                idx.withdrawal_slack_neg.len(),
                n,
                "withdrawal_slack_neg length must equal hydro_count for n={n}"
            );
            assert_eq!(
                idx.withdrawal_slack_pos.len(),
                n,
                "withdrawal_slack_pos length must equal hydro_count for n={n}"
            );
        }
    }

    // AC: withdrawal_slack_neg.start == evap_col_end, pos starts at neg.end.
    //
    // N=2, L=0, no penalty, no FPHA, 1 evap hydro.
    // evap cols: [excess_end, excess_end+3) = [15, 18)
    // withdrawal_slack_neg: [18, 20)
    // withdrawal_slack_pos: [20, 22)
    #[test]
    fn withdrawal_slack_immediately_after_evap_columns() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(2, 0, 0, 0, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![0]),
        );

        assert_eq!(
            idx.withdrawal_slack_neg.start, 18,
            "withdrawal_slack_neg must start at evap_col_end=18"
        );
        assert_eq!(idx.withdrawal_slack_neg.len(), 2);
        assert_eq!(idx.withdrawal_slack_neg, 18..20);
        assert_eq!(idx.withdrawal_slack_pos, 20..22);
        // Operational slacks must start at pos.end
        assert_eq!(
            idx.outflow_below_slack.start, idx.withdrawal_slack_pos.end,
            "outflow_below_slack must start at withdrawal_slack_pos.end"
        );
    }

    // new() produces empty FPHA ranges.
    #[test]
    fn new_fpha_ranges_are_empty() {
        let idx = StageIndexer::new(3, 2);
        assert!(idx.generation.is_empty());
        assert_eq!(idx.n_fpha_hydros, 0);
        assert!(idx.fpha_hydro_indices.is_empty());
        assert!(idx.fpha_rows.is_empty());
    }

    // Adjacency invariant extended: generation immediately follows the prior region.
    #[test]
    fn extended_adjacency_invariant_with_fpha() {
        // N=2, L=1, T=1, Ln=1, B=1, K=1, no penalty, 1 FPHA hydro.
        // theta=8, decision_start=9
        // turbine:[9,11), spillage:[11,13), diversion:[13,15), thermal:[15,16),
        // line_fwd:[16,17), line_rev:[17,18), deficit:[18,19), excess:[19,20)
        // generation:[20,21) (1 FPHA * 1 block, after excess.end since no penalty)
        let idx =
            StageIndexer::with_equipment(&eq(2, 1, 1, 1, 1, 1, false), &fpha(vec![0], vec![3]));

        assert_eq!(idx.turbine.start, idx.theta + 1);
        assert_eq!(idx.spillage.start, idx.turbine.end);
        assert_eq!(idx.diversion.start, idx.spillage.end);
        assert_eq!(idx.thermal.start, idx.diversion.end);
        assert_eq!(idx.line_fwd.start, idx.thermal.end);
        assert_eq!(idx.line_rev.start, idx.line_fwd.end);
        assert_eq!(idx.deficit.start, idx.line_rev.end);
        assert_eq!(idx.excess.start, idx.deficit.end);
        // generation follows excess (no penalty)
        assert_eq!(idx.generation.start, idx.excess.end);
        assert_eq!(idx.generation.len(), 1);
    }

    // ── Diversion field tests ──────────────────────────────────────────────

    // Diversion range: N=3, K=2 → diversion.len() = 6, contiguous with spillage.
    #[test]
    fn test_diversion_range_n3_l0_k2() {
        let idx = StageIndexer::with_equipment(&eq(3, 0, 0, 0, 1, 2, false), &fpha(vec![], vec![]));

        assert_eq!(idx.diversion.start, idx.spillage.end);
        assert_eq!(idx.diversion.len(), 3 * 2);
        assert_eq!(idx.thermal.start, idx.diversion.end);
    }

    // Diversion range: N=0 → diversion is empty.
    #[test]
    fn test_diversion_zero_hydros() {
        let idx = StageIndexer::with_equipment(&eq(0, 0, 1, 0, 1, 1, false), &fpha(vec![], vec![]));

        assert!(idx.diversion.is_empty());
    }

    // ── z_inflow field tests ───────────────────────────────────────────────

    // z_inflow starts at N*(1+L) (fixed offset) and has length hydro_count.
    #[test]
    fn z_inflow_range_new_constructor() {
        let idx = StageIndexer::new(3, 2);
        // z_inflow at N*(1+L) = 9..12
        assert_eq!(idx.z_inflow, 9..12);
        assert_eq!(idx.z_inflow.len(), idx.hydro_count);
    }

    // z_inflow is empty when hydro_count == 0.
    #[test]
    fn z_inflow_range_zero_hydros() {
        let idx = StageIndexer::new(0, 0);
        assert!(idx.z_inflow.is_empty());
        assert!(idx.z_inflow_rows.is_empty());
        assert_eq!(idx.z_inflow_row_start, 0);
    }

    // z_inflow rows start at row 0.
    #[test]
    fn z_inflow_row_fields() {
        let idx = StageIndexer::new(5, 1);
        // z_inflow rows start at row 0, length = hydro_count = 5.
        assert_eq!(idx.z_inflow_rows, 0..5);
        assert_eq!(idx.z_inflow_row_start, 0);
        assert_eq!(idx.z_inflow.len(), 5);
    }

    // z_inflow has correct length and rows for with_equipment constructor.
    #[test]
    fn z_inflow_range_with_equipment() {
        let idx = StageIndexer::with_equipment(&eq(2, 1, 1, 1, 1, 1, false), &fpha(vec![], vec![]));
        // N*(1+L) = 2*(1+1) = 4 (column range unchanged)
        assert_eq!(idx.z_inflow, 4..6);
        assert_eq!(idx.z_inflow.len(), 2);
        // z_inflow rows start at row 0, length = hydro_count = 2.
        assert_eq!(idx.z_inflow_rows, 0..2);
        assert_eq!(idx.z_inflow_row_start, 0);
    }

    // z_inflow placed correctly for single hydro, no lags case.
    #[test]
    fn z_inflow_single_hydro_no_lags() {
        let idx = StageIndexer::new(1, 0);
        // N*(1+L) = 1*(1+0) = 1, z_inflow at 1..2
        assert_eq!(idx.z_inflow, 1..2);
        assert_eq!(idx.z_inflow.len(), 1);
    }

    // ── Anticipated thermal state tests ────────────────────────────────────

    /// `StageIndexer::new` always produces an empty anticipated block.
    #[test]
    fn anticipated_state_empty_for_new() {
        let idx = StageIndexer::new(3, 2);
        assert_eq!(idx.anticipated_state, 0..0);
        assert_eq!(idx.anticipated_state_fixing, 0..0);
        assert_eq!(idx.n_anticipated, 0);
        assert_eq!(idx.k_max, 0);
    }

    /// SS5.5.3-style example with two anticipated plants and `K_max = 3`:
    /// `anticipated_state` consumes 6 columns starting at `N*(1+L) = 9`,
    /// shifting `z_inflow`, `storage_in` and `theta` by 6.
    #[test]
    fn anticipated_state_layout_n3_l2_nant2_kmax3() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, 2, 3),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert_eq!(idx.anticipated_state, 9..15);
        // anticipated_state_fixing is a permanent empty sentinel; state fixing uses column bounds.
        assert_eq!(idx.anticipated_state_fixing, 0..0);
        assert_eq!(idx.z_inflow, 15..18);
        assert_eq!(idx.storage_in, 18..21);
        assert_eq!(idx.theta, 21);
        assert_eq!(idx.n_state, 15);
        assert_eq!(idx.n_anticipated, 2);
        assert_eq!(idx.k_max, 3);
    }

    /// Zero hydros but two anticipated plants: `anticipated_state` collapses
    /// to `0..6`, `z_inflow` and `storage_in` are empty, `theta == 6`.
    #[test]
    fn anticipated_state_layout_n0_nant2_kmax3() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(0, 0, 0, 0, 1, 1, false, 2, 3),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert_eq!(idx.anticipated_state, 0..6);
        // anticipated_state_fixing is a permanent empty sentinel; state fixing uses column bounds.
        assert_eq!(idx.anticipated_state_fixing, 0..0);
        assert_eq!(idx.n_state, 6);
        assert_eq!(idx.z_inflow, 6..6);
        assert_eq!(idx.storage_in, 6..6);
        assert_eq!(idx.theta, 6);
        assert_eq!(idx.n_anticipated, 2);
        assert_eq!(idx.k_max, 3);
    }

    /// When `n_anticipated == 0` the layout produced by
    /// `with_equipment_and_evaporation` must match the pre-anticipated layout
    /// field-for-field against `StageIndexer::new`.
    #[test]
    fn anticipated_state_no_thermals_matches_existing() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(3, 2, 0, 0, 0, 0, false),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let base = StageIndexer::new(3, 2);

        assert_eq!(idx.storage, base.storage);
        assert_eq!(idx.inflow_lags, base.inflow_lags);
        assert_eq!(idx.anticipated_state, 0..0);
        assert_eq!(idx.anticipated_state_fixing, 0..0);
        assert_eq!(idx.z_inflow, base.z_inflow);
        assert_eq!(idx.storage_in, base.storage_in);
        assert_eq!(idx.theta, base.theta);
        assert_eq!(idx.n_state, base.n_state);
        assert_eq!(idx.storage_fixing, base.storage_fixing);
        assert_eq!(idx.lag_fixing, base.lag_fixing);
        assert_eq!(idx.z_inflow_rows, base.z_inflow_rows);
        assert_eq!(idx.z_inflow_row_start, base.z_inflow_row_start);
        assert_eq!(idx.n_anticipated, 0);
        assert_eq!(idx.k_max, 0);
    }

    /// `anticipated_state_fixing` is always empty (`0..0`);
    /// state pinning uses column bounds, not rows.
    #[test]
    fn anticipated_state_fixing_mirrors_state() {
        for (n_anticipated, k_max) in [(0, 0), (1, 1), (1, 3), (2, 3), (5, 4)] {
            let idx = StageIndexer::with_equipment_and_evaporation(
                &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, n_anticipated, k_max),
                &fpha(vec![], vec![]),
                &evap(vec![]),
            );
            assert_eq!(
                idx.anticipated_state_fixing,
                0..0,
                "anticipated_state_fixing must be 0..0 (sentinel) for n_anticipated={n_anticipated} k_max={k_max}"
            );
        }
    }

    /// All three state-fixing row ranges are permanent empty sentinels in every constructor.
    ///
    /// Covers `StageIndexer::new`, `StageIndexer::with_equipment`, and
    /// `StageIndexer::with_equipment_and_evaporation` (with anticipated thermals).
    #[test]
    fn state_fixing_rows_collapsed_to_empty_in_all_constructors() {
        // new() constructor
        let idx_new = StageIndexer::new(3, 2);
        assert_eq!(
            idx_new.storage_fixing,
            0..0,
            "new: storage_fixing must be 0..0"
        );
        assert_eq!(idx_new.lag_fixing, 0..0, "new: lag_fixing must be 0..0");
        assert_eq!(
            idx_new.anticipated_state_fixing,
            0..0,
            "new: anticipated_state_fixing must be 0..0"
        );

        // with_equipment() constructor (delegates to with_equipment_and_evaporation)
        let idx_eq =
            StageIndexer::with_equipment(&eq(3, 2, 0, 0, 0, 0, false), &fpha(vec![], vec![]));
        assert_eq!(
            idx_eq.storage_fixing,
            0..0,
            "with_equipment: storage_fixing must be 0..0"
        );
        assert_eq!(
            idx_eq.lag_fixing,
            0..0,
            "with_equipment: lag_fixing must be 0..0"
        );
        assert_eq!(
            idx_eq.anticipated_state_fixing,
            0..0,
            "with_equipment: anticipated_state_fixing must be 0..0"
        );

        // with_equipment_and_evaporation() with anticipated thermals (n_anticipated=1, k_max=2)
        let idx_ant = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, 1, 2),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert_eq!(
            idx_ant.storage_fixing,
            0..0,
            "with_equipment_and_evaporation: storage_fixing must be 0..0"
        );
        assert_eq!(
            idx_ant.lag_fixing,
            0..0,
            "with_equipment_and_evaporation: lag_fixing must be 0..0"
        );
        assert_eq!(
            idx_ant.anticipated_state_fixing,
            0..0,
            "with_equipment_and_evaporation: anticipated_state_fixing must be 0..0"
        );
        // Column range anticipated_state is unchanged (N*(1+L) .. N*(1+L) + k_max).
        assert_eq!(
            idx_ant.anticipated_state,
            9..11,
            "anticipated_state column range must be N*(1+L)..N*(1+L)+n_ant*k_max = 9..11"
        );
        // water_balance starts immediately after z_inflow (hydro_count rows from row 0).
        assert_eq!(
            idx_ant.water_balance.start, idx_ant.z_inflow_rows.end,
            "water_balance.start must equal z_inflow_rows.end"
        );
    }

    /// `z_inflow.start` is exactly `N*(1+L) + n_anticipated * k_max`.
    #[test]
    fn anticipated_state_shifts_z_inflow() {
        // SS5.5.3-extended: N=3, L=2, n_anticipated=2, k_max=3.
        let idx_small = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, 2, 3),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert_eq!(idx_small.z_inflow.start, 3 * (1 + 2) + 2 * 3);

        // Production scale: N=160, L=12, n_anticipated=10, k_max=4.
        let idx_prod = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(160, 12, 0, 0, 1, 1, false, 10, 4),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert_eq!(idx_prod.z_inflow.start, 160 * (1 + 12) + 10 * 4);
    }

    /// Minimal non-empty anticipated block: `n_anticipated=1`, `k_max=1` → length 1.
    #[test]
    fn anticipated_state_boundary_kmax_one() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(2, 1, 0, 0, 1, 1, false, 1, 1),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        // N*(1+L) = 2*2 = 4
        assert_eq!(idx.anticipated_state, 4..5);
        assert_eq!(idx.anticipated_state.len(), 1);
        assert_eq!(idx.n_state, 5); // base 4 + 1
        assert_eq!(idx.z_inflow, 5..7);
        assert_eq!(idx.theta, 9); // 5 + 2*(storage_in width) = 5 + 2 + 2
    }

    /// Acceptance side of the `debug_assert`: (`n_anticipated=0`, `k_max=0`)
    /// produces a layout identical to the existing one.
    #[test]
    fn anticipated_state_boundary_n_anticipated_zero() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, 0, 0),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Layout must collapse back to the pre-anticipated state form.
        assert_eq!(idx.anticipated_state, 0..0);
        assert_eq!(idx.z_inflow, 9..12);
        assert_eq!(idx.storage_in, 12..15);
        assert_eq!(idx.theta, 15);
        assert_eq!(idx.n_state, 9);
    }

    /// Debug-only assertion: building with `n_anticipated > 0` but `k_max == 0`
    /// must trigger the documented `debug_assert!` failure.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "k_max must be >= 1 when n_anticipated > 0")]
    fn anticipated_state_debug_assert_kmax_zero_when_nant_nonzero() {
        let _ = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(1, 0, 0, 0, 1, 1, false, 1, 0),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
    }

    /// `n_state == N*(1+L) + n_anticipated * k_max` exactly.
    #[test]
    fn anticipated_state_n_state_formula() {
        for (n, l, n_anticipated, k_max) in [
            (3_usize, 2_usize, 0_usize, 0_usize),
            (3, 2, 2, 3),
            (160, 12, 10, 4),
            (1, 0, 1, 1),
        ] {
            let idx = StageIndexer::with_equipment_and_evaporation(
                &eq_with_anticipated(n, l, 0, 0, 1, 1, false, n_anticipated, k_max),
                &fpha(vec![], vec![]),
                &evap(vec![]),
            );
            assert_eq!(
                idx.n_state,
                n * (1 + l) + n_anticipated * k_max,
                "n_state mismatch for (N={n}, L={l}, n_anticipated={n_anticipated}, k_max={k_max})"
            );
        }
    }

    // ── Anticipated decision column tests ──────────────────────────────────

    /// `StageIndexer::new` produces an empty anticipated-decision block and
    /// empty per-plant metadata vectors.
    #[test]
    fn anticipated_decision_empty_for_new() {
        let idx = StageIndexer::new(3, 2);
        assert_eq!(idx.anticipated_decision, 0..0);
        assert!(idx.anticipated_lead_stages.is_empty());
        assert!(idx.anticipated_thermal_indices.is_empty());
    }

    /// Direct `EquipmentCounts` literal mirroring the AC-2 example:
    /// `n_anticipated=2`, `k_max=3`, `K_i = [3, 2]`, `thermal_idx = [0, 2]`.
    #[test]
    fn anticipated_decision_layout_basic() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 2,
                max_par_order: 1,
                n_thermals: 3,
                n_lines: 2,
                n_buses: 4,
                n_blks: 2,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 2,
                k_max: 3,
                anticipated_lead_stages: vec![3, 2],
                anticipated_thermal_indices: vec![0, 2],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        // anticipated_decision sits between thermal and anticipated_state_out, length 2.
        assert_eq!(idx.anticipated_decision.start, idx.thermal.end);
        assert_eq!(idx.anticipated_decision.len(), 2);
        // anticipated_state_out sits immediately after anticipated_decision, length 2.
        assert_eq!(
            idx.anticipated_state_out.start,
            idx.anticipated_decision.end
        );
        assert_eq!(idx.anticipated_state_out.len(), 2);
        // line_fwd starts after anticipated_state_out.
        assert_eq!(idx.line_fwd.start, idx.anticipated_state_out.end);
        // Per-plant metadata round-trips intact.
        assert_eq!(idx.anticipated_lead_stages, vec![3, 2]);
        assert_eq!(idx.anticipated_thermal_indices, vec![0, 2]);
    }

    /// When `n_anticipated == 0` the layout produced by
    /// `with_equipment_and_evaporation` must be byte-identical to a build with
    /// the legacy non-anticipated layout: `line_fwd.start == thermal.end`
    /// and no shift downstream.
    #[test]
    fn anticipated_decision_no_anticipated_matches_existing() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(2, 1, 3, 2, 4, 2, false),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert_eq!(idx.anticipated_decision, 0..0);
        assert_eq!(idx.line_fwd.start, idx.thermal.end);
        // Concrete offsets for the no-anticipated layout (decision_start = theta+1).
        // N=2, L=1, T=3, Ln=2, B=4, K=2, no penalty:
        // theta = 2*(3+1) = 8; decision_start = 9
        // turbine: 9..13 (2*2), spillage: 13..17, diversion: 17..21,
        // thermal: 21..27 (3*2), line_fwd: 27..31 (2*2), line_rev: 31..35,
        // deficit: 35..43 (4*1*2), excess: 43..51 (4*2)
        assert_eq!(idx.theta, 8);
        assert_eq!(idx.turbine, 9..13);
        assert_eq!(idx.thermal, 21..27);
        assert_eq!(idx.line_fwd, 27..31);
        assert_eq!(idx.line_rev, 31..35);
        assert_eq!(idx.deficit, 35..43);
        assert_eq!(idx.excess, 43..51);
    }

    /// `line_fwd.start == anticipated_state_out.end` for various
    /// `n_anticipated`. Sweep includes 0 (no shift), 1, 2, 5.
    ///
    /// `anticipated_state_out` is inserted between `anticipated_decision`
    /// and `line_fwd`, so the layout invariant is:
    ///   `anticipated_decision.end` == `anticipated_state_out.start`
    ///   `anticipated_state_out.end` == `line_fwd.start`
    #[test]
    fn anticipated_decision_shifts_line_fwd() {
        for n_ant in [0_usize, 1, 2, 5] {
            let lead = if n_ant == 0 { vec![] } else { vec![2; n_ant] };
            let thermal_idx = (0..n_ant).collect::<Vec<_>>();
            let k_max = if n_ant == 0 { 0 } else { 2 };
            let idx = StageIndexer::with_equipment_and_evaporation(
                &EquipmentCounts {
                    hydro_count: 1,
                    max_par_order: 0,
                    n_thermals: n_ant.max(1),
                    n_lines: 1,
                    n_buses: 1,
                    n_blks: 1,
                    has_inflow_penalty: false,
                    max_deficit_segments: 1,
                    n_anticipated: n_ant,
                    k_max,
                    anticipated_lead_stages: lead,
                    anticipated_thermal_indices: thermal_idx,
                    n_pumping: 0,
                },
                &fpha(vec![], vec![]),
                &evap(vec![]),
            );
            if n_ant > 0 {
                // When the block is non-empty:
                //   anticipated_decision.end == anticipated_state_out.start
                //   anticipated_state_out.end == line_fwd.start
                assert_eq!(
                    idx.anticipated_state_out.start, idx.anticipated_decision.end,
                    "anticipated_state_out.start must equal anticipated_decision.end for n_ant={n_ant}"
                );
                assert_eq!(
                    idx.line_fwd.start, idx.anticipated_state_out.end,
                    "line_fwd.start must equal anticipated_state_out.end for n_ant={n_ant}"
                );
                assert_eq!(
                    idx.anticipated_state_out.len(),
                    n_ant,
                    "anticipated_state_out.len() must equal n_anticipated for n_ant={n_ant}"
                );
            } else {
                // When both blocks collapse to `0..0`, `line_fwd.start`
                // falls directly on `thermal.end` (no shift).
                assert_eq!(
                    idx.anticipated_state_out,
                    0..0,
                    "anticipated_state_out must be 0..0 when n_ant=0"
                );
                assert_eq!(
                    idx.line_fwd.start, idx.thermal.end,
                    "line_fwd.start must equal thermal.end when n_ant=0"
                );
            }
            assert_eq!(
                idx.anticipated_decision.len(),
                n_ant,
                "anticipated_decision.len() must equal n_anticipated for n_ant={n_ant}"
            );
        }
    }

    /// `anticipated_decision.start == thermal.end` whenever `n_anticipated > 0`.
    #[test]
    fn anticipated_decision_contiguous_with_thermal() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 2,
                n_lines: 1,
                n_buses: 1,
                n_blks: 2,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 2,
                k_max: 3,
                anticipated_lead_stages: vec![3, 2],
                anticipated_thermal_indices: vec![0, 1],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert_eq!(idx.anticipated_decision.start, idx.thermal.end);
    }

    /// Per-plant `anticipated_thermal_indices` round-trips through the
    /// constructor exactly.
    #[test]
    fn anticipated_decision_thermal_indices_preserved() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 6,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 4,
                anticipated_lead_stages: vec![4, 2, 1],
                anticipated_thermal_indices: vec![0, 2, 5],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert_eq!(idx.anticipated_thermal_indices, vec![0, 2, 5]);
        assert_eq!(idx.anticipated_lead_stages, vec![4, 2, 1]);
    }

    // ── Anticipated fishing row tests (placement, via constructor) ───────────

    /// `StageIndexer::new` produces an empty fishing range and a zero start.
    #[test]
    fn anticipated_fishing_empty_for_new() {
        let idx = StageIndexer::new(3, 2);
        assert_eq!(idx.anticipated_fishing, 0..0);
        assert_eq!(idx.anticipated_fishing_start, 0);
    }

    /// When operational violations are active, `anticipated_fishing_start`
    /// equals `min_generation_rows.end`. AC-2 worked example: 4 operational
    /// row families of N*K = 2 rows each = 8 rows total beginning at `A`,
    /// so `anticipated_fishing_start == A + 8 == min_generation_rows.end`.
    #[test]
    fn anticipated_fishing_start_after_min_generation_rows() {
        // N=2 hydros, K=1 block, no FPHA, no evap, no penalty.
        // n_anticipated=2, k_max=3 — shifts state by 2*3 = 6.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 2,
                max_par_order: 0,
                n_thermals: 2,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 2,
                k_max: 3,
                anticipated_lead_stages: vec![3, 2],
                anticipated_thermal_indices: vec![0, 1],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Operational violations must be active (hydro_count > 0).
        assert!(idx.has_operational_violations);
        // The start offset is precisely the end of `min_generation_rows`.
        assert_eq!(idx.anticipated_fishing_start, idx.min_generation_rows.end);
        // Stage-0 canonical layout: zero-length range anchored at the start.
        assert_eq!(
            idx.anticipated_fishing,
            idx.anticipated_fishing_start..idx.anticipated_fishing_start,
        );
    }

    /// When operational violations are inactive (`hydro_count == 0`),
    /// `anticipated_fishing_start` equals `evap_rows_end`
    /// (= `fpha_row_cursor + n_evap_hydros`).
    #[test]
    fn anticipated_fishing_start_after_evap_when_no_op_violations() {
        // hydro_count = 0 → no operational violations, no withdrawal slack,
        // no FPHA, no evap. Row cursor sits at `load_balance.end` because
        // `fpha_row_cursor == load_balance.end` and `n_evap_hydros == 0`.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 0,
                max_par_order: 0,
                n_thermals: 1,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert!(!idx.has_operational_violations);
        // With no FPHA and no evap, `evap_rows_end == load_balance.end`.
        let evap_rows_end = idx.load_balance.end;
        assert_eq!(idx.anticipated_fishing_start, evap_rows_end);
        assert_eq!(
            idx.anticipated_fishing,
            idx.anticipated_fishing_start..idx.anticipated_fishing_start,
        );
    }

    /// `anticipated_state_out` range is placed immediately after
    /// `anticipated_decision` and `line_fwd` follows `anticipated_state_out`.
    #[test]
    fn test_anticipated_state_out_range_is_after_anticipated_decision() {
        let counts = EquipmentCounts {
            hydro_count: 2,
            max_par_order: 1,
            n_thermals: 3,
            n_lines: 1,
            n_buses: 2,
            n_blks: 1,
            has_inflow_penalty: false,
            max_deficit_segments: 1,
            n_anticipated: 2,
            k_max: 3,
            anticipated_lead_stages: vec![2, 3],
            anticipated_thermal_indices: vec![0, 1],
            n_pumping: 0,
        };
        let idx = StageIndexer::with_equipment(&counts, &fpha(vec![], vec![]));

        // Range adjacency.
        assert_eq!(
            idx.anticipated_state_out.start,
            idx.anticipated_decision.end
        );
        assert_eq!(
            idx.anticipated_state_out.end - idx.anticipated_state_out.start,
            2
        );

        // line_fwd starts immediately after the new column block.
        assert_eq!(idx.line_fwd.start, idx.anticipated_state_out.end);

        // n_state is unchanged: N*(1+L) + A*K_max = 2*2 + 2*3 = 10.
        assert_eq!(idx.n_state, 10);
    }

    /// `anticipated_state_out` is empty (`0..0`) when built via `StageIndexer::new`.
    #[test]
    fn test_anticipated_state_out_is_empty_when_no_anticipated() {
        let idx = StageIndexer::new(3, 2);
        assert_eq!(idx.anticipated_state_out, 0..0);
        assert_eq!(idx.n_state, 9);
    }

    // ---------------------------------------------------------------------------
    // Layout-invariance tests (indexer-layout-impact.md Q1)
    // ---------------------------------------------------------------------------

    /// Locks the `n_state` formula against the addition of the
    /// `anticipated_state_out` control-region column. Per
    /// `indexer-layout-impact.md` Q1, `anticipated_state_out` is not a state
    /// index and must not contribute to `n_state`.
    #[test]
    fn test_n_state_unchanged_with_anticipated_state_out_addition() {
        let fpha_empty = fpha(vec![], vec![]);

        // Case A: no anticipated thermals.
        // n_state = hydro_count * (1 + max_par_order) = 3 * (1 + 2) = 9.
        let counts_a = EquipmentCounts {
            hydro_count: 3,
            max_par_order: 2,
            n_thermals: 1,
            n_lines: 1,
            n_buses: 1,
            n_blks: 1,
            has_inflow_penalty: false,
            max_deficit_segments: 1,
            n_anticipated: 0,
            k_max: 0,
            anticipated_lead_stages: vec![],
            anticipated_thermal_indices: vec![],
            n_pumping: 0,
        };
        let idx_a = StageIndexer::with_equipment(&counts_a, &fpha_empty);
        assert_eq!(idx_a.n_state, 3 * (1 + 2), "n_state without anticipated");
        assert_eq!(idx_a.anticipated_state_out, 0..0);

        // Case B: with anticipated thermals.
        // n_state = hydro_count * (1 + max_par_order) + n_anticipated * k_max
        //         = 3 * (1 + 2) + 2 * 3 = 15.
        // anticipated_state_out has length n_anticipated = 2, but is a
        // control-region column and must NOT be counted in n_state.
        let counts_b = EquipmentCounts {
            hydro_count: 3,
            max_par_order: 2,
            n_thermals: 2,
            n_lines: 1,
            n_buses: 1,
            n_blks: 1,
            has_inflow_penalty: false,
            max_deficit_segments: 1,
            n_anticipated: 2,
            k_max: 3,
            anticipated_lead_stages: vec![2, 3],
            anticipated_thermal_indices: vec![0, 1],
            n_pumping: 0,
        };
        let idx_b = StageIndexer::with_equipment(&counts_b, &fpha_empty);
        assert_eq!(
            idx_b.n_state,
            3 * (1 + 2) + 2 * 3,
            "n_state formula must be N*(1+L) + A*K_max"
        );
        assert_eq!(
            idx_b.anticipated_state_out.end - idx_b.anticipated_state_out.start,
            2,
            "anticipated_state_out range length must equal n_anticipated"
        );
    }
}
