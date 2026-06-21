//! The [`StageIndexer`] role-(b) geometry constructor and the private
//! column/row range build helpers.
//!
//! Owns the single role-(b) constructor (`with_equipment_and_evaporation`) and
//! the private helpers that compute each region's column/row range. The control
//! region is anchored at `theta + 1`, where `theta = N*(3+L) + A*k_max + A` is a
//! pure function of the state-vector dimensions; the descriptor does not store
//! `theta` (it is role-(a), owned by [`StateLayout`](super::StateLayout)).

use std::ops::Range;

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
    // The four slack/row families exist iff there is at least one hydro, so the
    // flag is the hydro-count predicate, not an independent degree of freedom.
    let has_operational_violations = hydro_count != 0;
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
            has_operational_violations,
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
        has_operational_violations,
    }
}

impl StageIndexer {
    /// Construct the role-(b) geometry descriptor for one stage's LP.
    ///
    /// Computes the equipment decision-variable ranges that follow `theta` in the
    /// LP, the FPHA / evaporation / withdrawal / operational-violation slack
    /// ranges, the constraint row ranges, and the anticipated identity maps. The
    /// role-(a) state-vector region (`storage`, `inflow_lags`, `theta`, `n_state`,
    /// the resolvers, the mask) is **not** built here — it lives on
    /// [`StateLayout`](super::StateLayout).
    ///
    /// The equipment column layout matches the `build_single_stage_template`
    /// column layout in the [`lp::builder`](crate::lp::builder) `template` module
    /// exactly:
    ///
    /// ```text
    /// decision_start      = theta + 1
    /// turbine_start       = decision_start
    /// spillage_start      = turbine_start  + n_hydros * n_blks
    /// diversion_start     = spillage_start + n_hydros * n_blks
    /// thermal_start       = diversion_start + n_hydros * n_blks
    /// anticipated_decision = [thermal_end, thermal_end + A)
    /// line_fwd_start      = thermal_end + A
    /// line_rev_start      = line_fwd_start + n_lines * n_blks
    /// deficit_start       = line_rev_start + n_lines * n_blks
    /// excess_start        = deficit_start  + n_buses * max_deficit_segments * n_blks
    /// inflow_slack_start  = excess_end  (only when has_inflow_penalty && hydro_count > 0)
    /// generation_start    = inflow_slack_end  (FPHA generation columns)
    /// evap_start          = generation_end  (3 columns per evaporation hydro, stage-level)
    /// ```
    ///
    /// where `theta = N*(3+L) + A*k_max + A` is a pure function of the state-vector
    /// dimensions. FPHA generation columns come immediately after `inflow_slack`
    /// (or after `excess` when `has_inflow_penalty == false`), one column per FPHA
    /// hydro per block. FPHA constraint rows are placed after `load_balance`.
    ///
    /// Evaporation columns (3 per evaporation hydro: evaporation outflow,
    /// `f_evap_plus`, `f_evap_minus`) are stage-level (not per-block) and come
    /// immediately after the FPHA generation columns. Evaporation rows (1 per
    /// evaporation hydro) are placed after FPHA rows.
    ///
    /// # Notes
    ///
    /// This single global descriptor strides its equipment column ranges by a
    /// stage-0-derived `n_blks`, so those ranges are valid only at stages whose
    /// block count equals stage 0's; the per-stage `StageLayout` reads each
    /// stage's own `stage.blocks.len()` and is the authority where block counts
    /// vary.
    ///
    /// NCS generation columns are **not** addressed from this descriptor. NCS
    /// presence is carried by
    /// [`StudyDimensions::has_ncs`](crate::indexer::StudyDimensions); the
    /// per-(ncs, block) column base is read per stage from
    /// `StageContext::ncs_col_starts[stage]` (and `LbEvalSpec::ncs_generation` for
    /// the stage-0 lower bound).
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

        // theta is the role-(a) future-cost column, a pure function of the
        // state-vector dimensions: it lives after the storage / inflow-lag /
        // anticipated-state / anticipated-state-out / z-inflow / storage-in
        // blocks. It is owned by `StateLayout`; computed here only to anchor the
        // control region (`decision_start = theta + 1`) byte-identically.
        //   theta = N*(3+L) + A*k_max + A
        let theta = hydro_count * (3 + max_par_order) + n_ant_state + n_anticipated;
        let decision_start = theta + 1;

        // z_inflow rows start at row 0 (ROW start, not column start).
        let z_inflow_start_row = 0_usize;
        let z_inflow_rows = z_inflow_start_row..z_inflow_start_row + hydro_count;
        let z_inflow_row_start = z_inflow_start_row;

        let turbine_start = decision_start;
        let spillage_start = turbine_start + hydro_count * n_blks;
        let diversion_start = spillage_start + hydro_count * n_blks;
        let thermal_start = diversion_start + hydro_count * n_blks;
        let thermal_end = thermal_start + n_thermals * n_blks;
        // Anticipated-decision columns sit between `thermal` and `line_fwd`. Every
        // anticipated plant has K_i <= T, so at stage 0 (the canonical stage) all
        // `n_anticipated` columns are active. The matching `anticipated_state_out`
        // cut-target block does NOT live here — it is in the stage-invariant state
        // region on `StateLayout`; only the priced decision variable rides the
        // n_blks-dependent `thermal_end` offset.
        //
        // Control-region layout (equipment side):
        //   anticipated_decision = [thermal_end, thermal_end + A)
        //   line_fwd             = [thermal_end + A, …)
        let anticipated_decision_start = thermal_end;
        let anticipated_decision_end = thermal_end + n_anticipated;
        let anticipated_decision = if n_anticipated > 0 {
            anticipated_decision_start..anticipated_decision_end
        } else {
            0..0
        };
        let line_fwd_start = anticipated_decision_end;
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

        // z_inflow rows start at row 0; water_balance follows z_inflow at row
        // hydro_count.
        let water_balance_start = z_inflow_start_row + hydro_count;
        let load_balance_start = water_balance_start + hydro_count;
        let load_balance_end = load_balance_start + n_buses * n_blks;

        let (fpha_rows, fpha_row_cursor) =
            Self::build_fpha_rows(&fpha.planes_per_hydro, n_blks, load_balance_end);

        let evap_indices_vec =
            Self::build_evap_indices(n_evap_hydros, evap_col_start, fpha_row_cursor);
        let evap_col_end = evap_col_start + n_evap_hydros * EVAP_COLS_PER_HYDRO;
        let (withdrawal_slack_neg, withdrawal_slack_pos) = if hydro_count > 0 {
            let neg = evap_col_end..evap_col_end + hydro_count;
            let pos = neg.end..neg.end + hydro_count;
            (neg, pos)
        } else {
            (0..0, 0..0)
        };

        // Operational violation slack columns: 4 families * (hydro_count * n_blks).
        // Columns are placed after withdrawal slack; rows after evaporation rows.
        let evap_rows_end = fpha_row_cursor + n_evap_hydros;
        let ws_end = withdrawal_slack_pos.end;
        let op = build_oper_violation_ranges(hydro_count, n_blks, ws_end, evap_rows_end);

        // Anticipated-fishing rows are placed after the operational-violation rows
        // when those are active, otherwise directly after the evaporation rows. The
        // stage-0 canonical layout stores a zero-length range; the per-stage
        // template populates `anticipated_fishing_start + local_idx_at_stage`.
        let fishing_start = if op.has_operational_violations {
            op.min_generation_rows.end
        } else {
            evap_rows_end
        };

        Self {
            turbine: turbine_start..spillage_start,
            spillage: spillage_start..diversion_start,
            diversion: diversion_start..thermal_start,
            thermal: thermal_start..thermal_end,
            anticipated_decision,
            line_fwd: line_fwd_start..line_rev_start,
            line_rev: line_rev_start..deficit_start,
            deficit: deficit_start..excess_start,
            max_deficit_segments,
            excess: excess_start..excess_end,
            n_blks,
            water_balance: water_balance_start..water_balance_start + hydro_count,
            load_balance: load_balance_start..load_balance_end,
            inflow_slack,
            inflow_slack_rows: 0..0,
            generation,
            n_fpha_hydros,
            fpha_hydro_indices,
            fpha_rows,
            n_evap_hydros,
            evap_hydro_indices,
            evap_indices: evap_indices_vec,
            withdrawal_slack_neg,
            withdrawal_slack_pos,
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
            z_inflow_rows,
            z_inflow_row_start,
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
}

#[cfg(test)]
mod tests {
    use crate::indexer::StageIndexer;
    use crate::indexer::test_fixtures::{eq, evap, fpha};

    // Worked example: N=1, L=0, T=2, Ln=1, B=2, K=1, no penalty.
    //
    // theta = N*(3+L) = 1*(3+0) = 3
    // decision_start = 4
    // turbine:    [4, 5)   (1 hydro * 1 block)
    // spillage:   [5, 6)
    // diversion:  [6, 7)
    // thermal:    [7, 9)   (2 thermals * 1 block)
    // line_fwd:   [9, 10)
    // line_rev:  [10, 11)
    // deficit:   [11, 13)  (2 buses * 1 block)
    // excess:    [13, 15)
    #[test]
    fn with_equipment_doctest_n1_l0_t2_l1_b2_k1() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(1, 0, 2, 1, 2, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert_eq!(idx.turbine, 4..5);
        assert_eq!(idx.spillage, 5..6);
        assert_eq!(idx.diversion, 6..7);
        assert_eq!(idx.thermal, 7..9);
        assert_eq!(idx.line_fwd, 9..10);
        assert_eq!(idx.line_rev, 10..11);
        assert_eq!(idx.deficit, 11..13);
        assert_eq!(idx.excess, 13..15);

        assert_eq!(idx.n_blks, 1);
        // The non-state scalars (`n_thermals`/`n_lines`/`n_buses`) moved to
        // `StudyDimensions`; the equipment ranges above already prove the
        // constructor applied them (`thermal = 7..9` at `n_blks = 1` ⟹ T = 2,
        // `line_fwd = 9..10` ⟹ Ln = 1, `deficit = 11..13` at S = 1 ⟹ B = 2).
    }

    // N=2, L=1, T=3, Ln=2, B=4, K=2.
    //
    // theta = N*(3+L) = 2*(3+1) = 8 → decision_start = 9.
    #[test]
    fn with_equipment_n2_l1_t3_l2_b4_k2() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(2, 1, 3, 2, 4, 2, false),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert_eq!(idx.turbine, 9..13);
        assert_eq!(idx.spillage, 13..17);
        assert_eq!(idx.diversion, 17..21);
        assert_eq!(idx.thermal, 21..27);
        assert_eq!(idx.line_fwd, 27..31);
        assert_eq!(idx.line_rev, 31..35);
        assert_eq!(idx.deficit, 35..43);
        assert_eq!(idx.excess, 43..51);
    }

    // Adjacency invariant — ranges must be contiguous and non-overlapping.
    #[test]
    fn with_equipment_ranges_are_contiguous() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(2, 1, 3, 2, 4, 2, false),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert_eq!(idx.spillage.start, idx.turbine.end);
        assert_eq!(idx.diversion.start, idx.spillage.end);
        assert_eq!(idx.thermal.start, idx.diversion.end);
        assert_eq!(idx.line_fwd.start, idx.thermal.end);
        assert_eq!(idx.line_rev.start, idx.line_fwd.end);
        assert_eq!(idx.deficit.start, idx.line_rev.end);
        assert_eq!(idx.excess.start, idx.deficit.end);
    }

    // has_inflow_penalty=true appends N slack columns after excess.
    //
    // N=2, L=1, T=1, Ln=1, B=1, K=1, penalty=true. theta = 8 → decision_start = 9.
    // excess: [19, 20); inflow_slack: [20, 22).
    #[test]
    fn with_equipment_inflow_penalty_appends_slack() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(2, 1, 1, 1, 1, 1, true),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        // The `has_inflow_penalty` flag moved to `StudyDimensions`; the surviving
        // non-empty `inflow_slack` range is the role-(b) evidence the slack columns
        // were appended.
        assert!(
            !idx.inflow_slack.is_empty(),
            "inflow_slack columns must be present"
        );
        assert_eq!(
            idx.inflow_slack.start, idx.excess.end,
            "inflow_slack.start must equal excess.end (contiguous)"
        );
        assert_eq!(idx.inflow_slack, 20..22);
        assert!(
            idx.inflow_slack_rows.is_empty(),
            "inflow_slack_rows must remain empty"
        );

        let no_penalty = StageIndexer::with_equipment_and_evaporation(
            &eq(2, 1, 1, 1, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert!(no_penalty.inflow_slack.is_empty());
    }

    // ── FPHA field tests ───────────────────────────────────────────────────

    // No FPHA hydros → generation is empty, fpha_rows is empty.
    #[test]
    fn fpha_no_hydros_generation_is_empty() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(4, 0, 0, 0, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert!(
            idx.generation.is_empty(),
            "generation must be empty with no FPHA hydros"
        );
        assert_eq!(idx.n_fpha_hydros, 0);
        assert!(idx.fpha_hydro_indices.is_empty());
        assert!(idx.fpha_rows.is_empty());
    }

    // 1 FPHA hydro, 1 block, 3 planes. fpha_rows[0].start == load_balance.end.
    #[test]
    fn fpha_one_hydro_one_block_three_planes() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(2, 0, 1, 0, 1, 1, false),
            &fpha(vec![0], vec![3]),
            &evap(vec![]),
        );

        assert_eq!(idx.generation.len(), 1, "generation must span 1 column");
        assert_eq!(idx.n_fpha_hydros, 1);
        assert_eq!(idx.fpha_hydro_indices, vec![0]);

        assert_eq!(idx.fpha_rows.len(), 1);
        assert_eq!(
            idx.fpha_rows[0].start, idx.load_balance.end,
            "fpha_rows[0].start must equal load_balance.end"
        );
        assert_eq!(idx.fpha_rows[0].planes_per_block, 3);
    }

    // 2 FPHA hydros, 2 blocks, plane counts [5, 4].
    #[test]
    fn fpha_two_hydros_two_blocks_different_planes() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(4, 0, 0, 0, 1, 2, false),
            &fpha(vec![1, 3], vec![5, 4]),
            &evap(vec![]),
        );

        assert_eq!(idx.generation.len(), 4, "generation must span 4 columns");
        assert_eq!(idx.n_fpha_hydros, 2);
        assert_eq!(idx.fpha_hydro_indices, vec![1, 3]);

        assert_eq!(idx.fpha_rows.len(), 2);
        assert_eq!(
            idx.fpha_rows[0].start, idx.load_balance.end,
            "fpha_rows[0].start must equal load_balance.end"
        );
        assert_eq!(idx.fpha_rows[0].planes_per_block, 5);
        assert_eq!(
            idx.fpha_rows[1].start,
            idx.fpha_rows[0].start + 5 * 2,
            "fpha_rows[1].start must follow fpha_rows[0]'s 10-row region"
        );
        assert_eq!(idx.fpha_rows[1].planes_per_block, 4);
    }

    // FPHA generation columns are contiguous with the prior column region.
    #[test]
    fn fpha_generation_contiguous_with_prior_region() {
        let no_penalty = StageIndexer::with_equipment_and_evaporation(
            &eq(2, 0, 0, 0, 1, 1, false),
            &fpha(vec![0], vec![2]),
            &evap(vec![]),
        );
        assert_eq!(
            no_penalty.generation.start, no_penalty.excess.end,
            "generation.start must equal excess.end when no penalty"
        );

        let with_penalty = StageIndexer::with_equipment_and_evaporation(
            &eq(2, 0, 0, 0, 1, 1, true),
            &fpha(vec![0], vec![2]),
            &evap(vec![]),
        );
        assert_eq!(
            with_penalty.generation.start, with_penalty.inflow_slack.end,
            "generation.start must equal inflow_slack.end when penalty active"
        );
    }

    // ── Evaporation field tests ────────────────────────────────────────────

    // 0 evaporation hydros → evap_indices is empty.
    #[test]
    fn evap_no_hydros_indices_empty() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(3, 0, 1, 0, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert_eq!(idx.n_evap_hydros, 0);
        assert!(idx.evap_hydro_indices.is_empty());
        assert!(idx.evap_indices.is_empty());
    }

    // 1 evaporation hydro — verify column/row positions.
    //
    // N=2, L=0, T=0, Ln=0, B=1, K=1, no penalty, no FPHA, 1 evap hydro.
    // theta = 6 → decision_start = 7. excess: [14, 15); evap cols: [15, 18).
    // Rows: z_inflow [0,2); water_balance [2,4); load_balance [4,5); evap_row[0]=5.
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
        assert_eq!(ei.evaporation_flow_col, 15);
        assert_eq!(ei.f_evap_plus_col, 16);
        assert_eq!(ei.f_evap_minus_col, 17);
        assert_eq!(ei.evap_row, 5);
    }
}
