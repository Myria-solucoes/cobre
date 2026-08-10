//! Terminal commitment-state block: one state slot per declared post-horizon
//! delivery window, latched at its own in-study decider stage
//! ([`StateSpace::commitment_decider_stage`]) and carried by identity to the
//! terminal — the primal machinery [`crate::indexer::StateRegion::CommitmentBlock`]
//! sizes.
//!
//! Unlike the anticipated ring ([`super::delivery_ring`]), the block has no
//! depth axis: a window's state stays at the SAME slot for the whole horizon,
//! never shifting to a neighboring slot. So [`commitment_ring`] borrows
//! [`DeliveryRing`]'s addressing and [`DeliveryRing::emit_deposit`] for the
//! latch row, at `depth = 1`, but the carry-by-identity row
//! ([`fill_commitment_block_entries`]) is a bespoke same-slot `+1/−1` pair —
//! [`DeliveryRing::emit_shift_rows`] would target `in_col(slot + 1, lane)`,
//! which does not exist at `depth = 1` and silently drops the term.
//!
//! No consumption/fishing row exists for the block at all (contrast
//! `super::entries::fill_anticipated_fishing_entries`): every declared window
//! gets exactly one row per stage from [`fill_commitment_block_entries`] —
//! the latch at its own decider, the carry everywhere else — and nothing else
//! ever writes into a block column. The block's columns are never routed
//! through [`DeliveryRing::freeze_masked_columns`]: [`fill_commitment_block_columns`]
//! bounds every out/in/decision column open (`-inf, inf`) at every stage,
//! including the terminal.

use super::columns::ColumnBufs;
use super::delivery_ring::DeliveryRing;
use super::layout::StageLayout;
use crate::indexer::StateSpace;

/// The block's ring view: `n_lanes = state.n_commitment` (one lane per
/// declared window), `depth = 1` (no lead-stage axis — a window's state never
/// shifts slots). At `depth = 1`, `out_col(0, w)`/`in_col(0, w)` reduce to
/// `commitment_block_out.start + w`/`commitment_block_in.start + w`; borrowed
/// for the addressing and [`DeliveryRing::emit_deposit`] only, never
/// [`DeliveryRing::emit_shift_rows`]/[`DeliveryRing::freeze_masked_columns`].
pub(super) fn commitment_ring(state: &StateSpace) -> DeliveryRing {
    DeliveryRing::new(
        state.commitment_block_out.clone(),
        state.commitment_block_in.clone(),
        state.n_commitment,
        1,
    )
}

/// Global window indices whose decider is `stage_idx`, ascending — the sparse
/// decision-column family [`super::layout::CommitmentBlockLayout`] sizes.
/// Several windows may share one decider (a decider latching `N` windows
/// simply returns `N` entries here; there is no fan-out guard to trip).
pub(super) fn commitment_decision_windows_at(state: &StateSpace, stage_idx: usize) -> Vec<usize> {
    (0..state.n_commitment)
        .filter(|&w| state.commitment_decider_stage[w] == stage_idx)
        .collect()
}

/// Fill column bounds for the block's outgoing/incoming state columns (every
/// declared window, every stage) and this stage's decision-column family
/// (only the windows deciding here): all open `(-inf, inf)`, no objective —
/// no in-study cost is ever booked for a post-horizon commitment. Never calls
/// [`DeliveryRing::freeze_masked_columns`]: every commitment-block column is
/// genuine (no padding), so nothing is ever frozen `[0, 0]`, including at the
/// terminal stage.
pub(super) fn fill_commitment_block_columns(layout: &StageLayout, bufs: &mut ColumnBufs<'_>) {
    let state = layout.state;
    for col in state.commitment_block_out.clone() {
        bufs.col_lower[col] = f64::NEG_INFINITY;
        bufs.col_upper[col] = f64::INFINITY;
    }
    for col in state.commitment_block_in.clone() {
        bufs.col_lower[col] = f64::NEG_INFINITY;
        bufs.col_upper[col] = f64::INFINITY;
    }
    let decision_start = layout.commitment.col_commitment_decision_start;
    for local_idx in 0..layout.commitment.commitment_decision_windows.len() {
        let col = decision_start + local_idx;
        bufs.col_lower[col] = f64::NEG_INFINITY;
        bufs.col_upper[col] = f64::INFINITY;
    }
}

/// Fill row bounds for the block's per-window row family: `[0, 0]` equality
/// for every window every stage (both the latch and the carry row render
/// `[0, 0]` — the `+1`/`-1` structural coefficients in
/// [`fill_commitment_block_entries`] do the work, mirroring the anticipated
/// ring's definition rows).
pub(super) fn fill_commitment_block_rows(
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    let start = layout.commitment.row_commitment_start;
    for w in 0..layout.state.n_commitment {
        row_lower[start + w] = 0.0;
        row_upper[start + w] = 0.0;
    }
}

/// Fill CSC entries for the block's per-window row family: at window `w`'s
/// own decider stage, the latch-deposit row (`out_col(w) +1`, `decision_col
/// −1`, via [`DeliveryRing::emit_deposit`] — the `emit_deposit`/`emit_shift_rows`
/// choice in the module doc); every other stage, the carry-by-identity row
/// (`out_col(w) +1`, `in_col(w) −1`, a bespoke same-slot pair, NOT
/// [`DeliveryRing::emit_shift_rows`]'s next-slot shift). Neither row ever
/// references any column outside the block's own out/in/decision families —
/// no delivery/fishing row exists for the block at all.
pub(super) fn fill_commitment_block_entries(
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let state = layout.state;
    if state.n_commitment == 0 {
        return;
    }
    let ring = commitment_ring(state);
    let row_start = layout.commitment.row_commitment_start;
    let decision_start = layout.commitment.col_commitment_decision_start;
    let decision_windows = &layout.commitment.commitment_decision_windows;

    for w in 0..state.n_commitment {
        let row = row_start + w;
        if let Ok(local_idx) = decision_windows.binary_search(&w) {
            let decision_col = decision_start + local_idx;
            ring.emit_deposit(0, w, row, decision_col, col_entries);
        } else {
            col_entries[ring.out_col(0, w)].push((row, 1.0));
            col_entries[ring.in_col(0, w)].push((row, -1.0));
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, CascadeTopology, ContractBlockBounds, EntityId,
        HydroBlockBounds, HydroStageBounds, LineBlockBounds, PumpingBlockBounds, ResolvedBounds,
        ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds,
        ResolvedNcsFactors, ResolvedPenalties, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};
    use crate::indexer::{HydroCellIndex, StateSpace};
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::columns::{ColumnBufs, fill_stage_columns};
    use super::super::entries::build_stage_matrix_entries;
    use super::super::layout::{ResolvedTables, StageLayout, TemplateBuildCtx};
    use super::super::rows::fill_stage_rows;
    use super::super::test_support::two_block_stage;
    use super::{commitment_decision_windows_at, fill_commitment_block_columns};

    /// Owns data for a zero-entity context whose only feature is a terminal
    /// commitment block — mirrors `entries.rs`'s `zero_cost_tests::AntFixtures`
    /// for the anticipated ring.
    struct Fixtures {
        par_lp: PrecomputedPar,
        hydro_cell_index: HydroCellIndex,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
    }

    impl Fixtures {
        fn new(n_stages: usize) -> Self {
            Self {
                par_lp: PrecomputedPar::default(),
                hydro_cell_index: HydroCellIndex::build(&[]),
                cascade: CascadeTopology::build(&[]),
                bounds: ResolvedBounds::new(
                    &BoundsCountsSpec {
                        n_hydros: 0,
                        n_thermals: 0,
                        n_lines: 0,
                        n_pumping: 0,
                        n_contracts: 0,
                        n_stages,
                        k_max: 0,
                    },
                    &BoundsDefaults {
                        hydro: HydroStageBounds {
                            min_storage_hm3: 0.0,
                            max_storage_hm3: 0.0,
                            filling_min_rate_m3s: 0.0,
                            water_withdrawal_m3s: 0.0,
                        },
                        hydro_block: HydroBlockBounds::default(),
                        thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                        thermal_block: ThermalBlockBounds {
                            min_generation_mw: 0.0,
                            max_generation_mw: 0.0,
                        },
                        line_block: LineBlockBounds {
                            direct_mw: 0.0,
                            reverse_mw: 0.0,
                        },
                        pumping_block: PumpingBlockBounds {
                            min_flow_m3s: 0.0,
                            max_flow_m3s: 0.0,
                        },
                        contract_block: ContractBlockBounds {
                            min_mw: 0.0,
                            max_mw: 0.0,
                            price_per_mwh: 0.0,
                        },
                    },
                ),
                penalties: ResolvedPenalties::empty(),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                    cost_scale_factor: 1.0,
                },
                production_models: ProductionModelSet::new(vec![], 0, 1),
                evaporation_models: EvaporationModelSet::new(vec![]),
            }
        }

        fn make_ctx(&self, n_stages: usize) -> TemplateBuildCtx<'_> {
            TemplateBuildCtx {
                hydros: &[],
                thermals: &[],
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                hydro_cell_index: &self.hydro_cell_index,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_ncs_bounds: &self.resolved_ncs_bounds,
                    resolved_ncs_factors: &self.resolved_ncs_factors,
                    resolved_parameters: &self.resolved_parameters,
                },
                hydro_pos: BTreeMap::new(),
                thermal_pos: BTreeMap::new(),
                line_pos: BTreeMap::new(),
                bus_pos: BTreeMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &[],
                pumping_pos: BTreeMap::new(),
                n_pumping: 0,
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: Vec::new(),
                anticipated_thermal_indices: Vec::new(),
                anticipated_windows: Vec::new(),
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: (0..i32::try_from(n_stages).unwrap_or(0)).collect(),
                has_penalty: false,
                cumulative_discount_factors: vec![1.0; n_stages],
                total_hours_per_stage: vec![336.0; n_stages],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// AC5: a single decider latching `N > 1` windows returns `N` independent
    /// window indices — no fan-out guard, no rejection, unlike the in-study
    /// ring's `max_fanout` reject.
    #[test]
    fn commitment_decision_windows_at_returns_every_window_sharing_one_decider() {
        let state = StateSpace::new_with_commitment_block(
            0,
            0,
            0,
            Vec::new(),
            0,
            0,
            vec![],
            &[],
            3,
            vec![2, 2, 2],
            vec![EntityId(0), EntityId(0), EntityId(0)],
        );
        assert_eq!(commitment_decision_windows_at(&state, 2), vec![0, 1, 2]);
        assert!(commitment_decision_windows_at(&state, 0).is_empty());
    }

    /// A window not sharing another's decider stage is alone in its own
    /// per-stage decision-column family.
    #[test]
    fn commitment_decision_windows_at_is_per_stage_sparse() {
        let state = StateSpace::new_with_commitment_block(
            0,
            0,
            0,
            Vec::new(),
            0,
            0,
            vec![],
            &[],
            2,
            vec![0, 1],
            vec![EntityId(0), EntityId(0)],
        );
        assert_eq!(commitment_decision_windows_at(&state, 0), vec![0]);
        assert_eq!(commitment_decision_windows_at(&state, 1), vec![1]);
    }

    /// AC1 + AC2 + AC3, end to end through the real per-stage LP build
    /// (`fill_stage_columns` / `build_stage_matrix_entries` / `fill_stage_rows`,
    /// not the standalone functions): two windows, `w=0` deciding at stage 0,
    /// `w=1` carrying (its own decider is stage 1, beyond this stage). At
    /// stage 0: `w=0`'s row is the latch (`out +1`, decision `-1`); `w=1`'s
    /// row is the carry (`out +1`, `in -1`). No consumption row exists for
    /// either column (each column has exactly one CSC entry). Column bounds
    /// are open `(-inf, inf)`, never frozen.
    #[test]
    fn commitment_block_wires_latch_and_carry_through_the_full_stage_build() {
        let n_stages = 2;
        let state = StateSpace::new_with_commitment_block(
            0,
            0,
            0,
            Vec::new(),
            0,
            0,
            vec![],
            &[],
            2,
            vec![0, 1],
            vec![EntityId(0), EntityId(0)],
        );
        let fixtures = Fixtures::new(n_stages);
        let ctx = fixtures.make_ctx(n_stages);
        let stage = two_block_stage(0, [100.0, 100.0]);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        assert_eq!(layout.commitment.commitment_decision_windows, vec![0]);

        let col_entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
        let row_start = layout.commitment.row_commitment_start;
        let decision_col = layout.commitment.col_commitment_decision_start;
        let out0 = state.commitment_block_out.start;
        let out1 = state.commitment_block_out.start + 1;
        let in1 = state.commitment_block_in.start + 1;

        // w=0: latch at its own decider (stage 0).
        assert_eq!(col_entries[out0], vec![(row_start, 1.0)]);
        assert_eq!(col_entries[decision_col], vec![(row_start, -1.0)]);
        // w=1: carry (decider is stage 1, not this stage).
        assert_eq!(col_entries[out1], vec![(row_start + 1, 1.0)]);
        assert_eq!(col_entries[in1], vec![(row_start + 1, -1.0)]);

        let (row_lower, row_upper) = fill_stage_rows(&ctx, &stage, 0, &layout);
        assert_eq!(row_lower[row_start], 0.0);
        assert_eq!(row_upper[row_start], 0.0);
        assert_eq!(row_lower[row_start + 1], 0.0);
        assert_eq!(row_upper[row_start + 1], 0.0);

        let (col_lower, col_upper, objective) = fill_stage_columns(&ctx, &stage, 0, &layout);
        for &col in &[out0, out1, in1, decision_col] {
            assert_eq!(
                col_lower[col],
                f64::NEG_INFINITY,
                "col {col} lower must be open"
            );
            assert_eq!(
                col_upper[col],
                f64::INFINITY,
                "col {col} upper must be open"
            );
            assert_eq!(objective[col], 0.0, "col {col} carries no in-study cost");
        }
    }

    /// AC3: no row anywhere in the stage — not just the block's own rows —
    /// ever references a commitment-block column more than once (the
    /// no-fishing/no-consumption-row guarantee). Before its own decider (here
    /// stage 1, observed at stage 0), a window's carry row gives its out/in
    /// columns exactly one entry each, and no decision column exists at this
    /// stage at all — the latch case's degree-1 out/decision pair is pinned by
    /// `commitment_block_wires_latch_and_carry_through_the_full_stage_build`.
    #[test]
    fn commitment_block_columns_never_receive_a_second_entry() {
        let n_stages = 2;
        let state = StateSpace::new_with_commitment_block(
            0,
            0,
            0,
            Vec::new(),
            0,
            0,
            vec![],
            &[],
            1,
            vec![1],
            vec![EntityId(0)],
        );
        let fixtures = Fixtures::new(n_stages);
        let ctx = fixtures.make_ctx(n_stages);
        let stage = two_block_stage(0, [100.0, 100.0]);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let col_entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);

        assert!(layout.commitment.commitment_decision_windows.is_empty());
        assert_eq!(col_entries[state.commitment_block_out.start].len(), 1);
        assert_eq!(col_entries[state.commitment_block_in.start].len(), 1);
    }

    /// AC3 (terminal live): at the TERMINAL stage the block's out/in columns
    /// are still open `(-inf, inf)`, never frozen `[0, 0]` — confirms
    /// `fill_commitment_block_columns` never routes through
    /// `DeliveryRing::freeze_masked_columns`.
    #[test]
    fn commitment_block_terminal_columns_stay_live_not_frozen() {
        let n_stages = 2;
        let state = StateSpace::new_with_commitment_block(
            0,
            0,
            0,
            Vec::new(),
            0,
            0,
            vec![],
            &[],
            1,
            vec![0],
            vec![EntityId(0)],
        );
        let fixtures = Fixtures::new(n_stages);
        let ctx = fixtures.make_ctx(n_stages);
        let terminal_idx = n_stages - 1;
        let stage = two_block_stage(terminal_idx, [100.0, 100.0]);
        let layout = StageLayout::new(&ctx, &state, &stage, terminal_idx);

        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_commitment_block_columns(&layout, &mut bufs);

        for col in state.commitment_block_out.clone() {
            assert_eq!(col_lower[col], f64::NEG_INFINITY);
            assert_eq!(col_upper[col], f64::INFINITY);
        }
        for col in state.commitment_block_in.clone() {
            assert_eq!(col_lower[col], f64::NEG_INFINITY);
            assert_eq!(col_upper[col], f64::INFINITY);
        }
    }

    /// AC6 (additive & inert), at the LP-builder level: with `n_commitment ==
    /// 0` the built stage LP (`num_cols`, `num_rows`, and every CSC/bound
    /// array) is byte-identical to a layout built from a commitment-free
    /// `StateSpace` — complementing the `state_space.rs`-level layout pin.
    #[test]
    fn commitment_block_zero_windows_is_byte_identical_lp() {
        let n_stages = 1;
        let with_zero = StateSpace::new_with_commitment_block(
            0,
            0,
            0,
            Vec::new(),
            0,
            0,
            vec![],
            &[],
            0,
            Vec::new(),
            Vec::new(),
        );
        let baseline = StateSpace::new(0, 0, 0, Vec::new(), 0, 0, vec![], &[]);
        let fixtures = Fixtures::new(n_stages);
        let ctx = fixtures.make_ctx(n_stages);
        let stage = two_block_stage(0, [100.0, 100.0]);

        let layout_zero = StageLayout::new(&ctx, &with_zero, &stage, 0);
        let layout_baseline = StageLayout::new(&ctx, &baseline, &stage, 0);

        assert_eq!(layout_zero.num_cols, layout_baseline.num_cols);
        assert_eq!(layout_zero.rows.num_rows, layout_baseline.rows.num_rows);
        assert!(
            layout_zero
                .commitment
                .commitment_decision_windows
                .is_empty()
        );

        let (col_lower_zero, col_upper_zero, objective_zero) =
            fill_stage_columns(&ctx, &stage, 0, &layout_zero);
        let (col_lower_base, col_upper_base, objective_base) =
            fill_stage_columns(&ctx, &stage, 0, &layout_baseline);
        assert_eq!(col_lower_zero, col_lower_base);
        assert_eq!(col_upper_zero, col_upper_base);
        assert_eq!(objective_zero, objective_base);

        let entries_zero = build_stage_matrix_entries(&ctx, &stage, 0, &layout_zero);
        let entries_base = build_stage_matrix_entries(&ctx, &stage, 0, &layout_baseline);
        assert_eq!(entries_zero, entries_base);

        let (row_lower_zero, row_upper_zero) = fill_stage_rows(&ctx, &stage, 0, &layout_zero);
        let (row_lower_base, row_upper_base) = fill_stage_rows(&ctx, &stage, 0, &layout_baseline);
        assert_eq!(row_lower_zero, row_lower_base);
        assert_eq!(row_upper_zero, row_upper_base);
    }
}
