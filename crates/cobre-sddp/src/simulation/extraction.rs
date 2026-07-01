//! Scenario distribution and result extraction for the SDDP simulation phase.
//!
//! ## Column layout
//!
//! The state-region column layout is defined by [`StateLayout`]:
//!
//! ```text
//! [0, N)             storage      — outgoing storage volumes
//! [N, N*(1+L))       inflow_lags  — AR lag variables (hydro-major order)
//! [N*(1+L), N*(2+L)) z_inflow     — realized inflow (auxiliary, not state)
//! [N*(2+L), N*(3+L)) storage_in   — incoming storage volumes (fixed vars)
//! N*(3+L)            theta        — future cost variable
//! [theta+1, ...)     equipment    — turbine, spillage, thermal, lines, deficit, excess
//! ```
//!
//! The equipment column layout is defined per stage by `StageLayout`, threaded
//! into extraction as [`StageGeometry`](crate::lp_builder::StageGeometry).

use std::collections::HashMap;
use std::ops::Range;

use cobre_core::ConstraintSense;
use cobre_core::EntityId;

use crate::energy_conversion::EnergyConversionSet;
use crate::indexer::{BlockGrid, StateLayout, StudyDimensions};
use crate::lp_builder::{COST_SCALE_FACTOR, GenericConstraintRowEntry};
use crate::simulation::types::{
    ScenarioCategoryCosts, SimulationBusResult, SimulationContractResult, SimulationCostResult,
    SimulationExchangeResult, SimulationGenericViolationResult, SimulationHydroResult,
    SimulationInflowLagResult, SimulationNonControllableResult, SimulationPumpingResult,
    SimulationStageResult, SimulationThermalResult,
};

/// Reverse lookups from system hydro index to local FPHA/evaporation/filling-slack
/// slot, for **one stage**.
///
/// Membership is per-`(hydro, stage)`: a hydro can be FPHA at one stage and not
/// another, `σ_fill` exists only at a filling hydro's terminal Filling stage, and
/// `σ^{v-}` only at its Operating stages. A single global stage-0 list would
/// misclassify any stage whose membership differs. Each entry is `Some(slot)` /
/// `None`; the column is `geometry.<family>_col.start + slot`.
pub(crate) struct HydroReverseLookup {
    /// FPHA-local slot per hydro, `None` if not FPHA at this stage.
    pub(crate) fpha: Vec<Option<usize>>,
    /// Evaporation-local slot per hydro, `None` if no evaporation at this stage.
    pub(crate) evap: Vec<Option<usize>>,
    /// `σ_fill`-target slot per hydro, `None` if it owns no target column at this stage.
    pub(crate) filling_target: Vec<Option<usize>>,
    /// `σ^{v-}` operating-floor slot per hydro, `None` if it owns no floor column at this stage.
    pub(crate) filled_min_storage_floor: Vec<Option<usize>>,
}

impl HydroReverseLookup {
    /// Build the reverse lookup for one stage from its [`StageGeometry`].
    pub(crate) fn build(geometry: &crate::lp_builder::StageGeometry, n_hydros: usize) -> Self {
        let mut fpha = vec![None; n_hydros];
        for (local, &sys) in geometry.fpha_hydro_indices.iter().enumerate() {
            fpha[sys] = Some(local);
        }
        let mut evap = vec![None; n_hydros];
        for (local, &sys) in geometry.evap_hydro_indices.iter().enumerate() {
            evap[sys] = Some(local);
        }
        let mut filling_target = vec![None; n_hydros];
        for (local, &sys) in geometry.filling_target_hydro_indices.iter().enumerate() {
            filling_target[sys] = Some(local);
        }
        let mut filled_min_storage_floor = vec![None; n_hydros];
        for (local, &sys) in geometry
            .filled_min_storage_floor_hydro_indices
            .iter()
            .enumerate()
        {
            filled_min_storage_floor[sys] = Some(local);
        }
        Self {
            fpha,
            evap,
            filling_target,
            filled_min_storage_floor,
        }
    }

    /// Build one [`HydroReverseLookup`] per stage from the per-stage geometry table,
    /// once per simulation run so per-`(scenario, stage)` extraction never reallocates.
    pub(crate) fn build_per_stage(
        geometry_per_stage: &[crate::lp_builder::StageGeometry],
        n_hydros: usize,
    ) -> Vec<Self> {
        geometry_per_stage
            .iter()
            .map(|g| Self::build(g, n_hydros))
            .collect()
    }
}

/// Read the primal of a SPARSE per-hydro filling-slack column, or `0.0` when
/// `local_idx` is `None` (the hydro owns no column in that family at this stage).
#[inline]
fn read_filling_slack_primal(
    primal: &[f64],
    col_range: &Range<usize>,
    local_idx: Option<usize>,
) -> f64 {
    let Some(local) = local_idx else { return 0.0 };
    let col = col_range.start + local;
    debug_assert!(
        col < col_range.end && col < primal.len(),
        "filling-slack col {col} out of range {col_range:?} / primal len {}",
        primal.len(),
    );
    primal.get(col).copied().unwrap_or(0.0)
}

/// Reverse lookup from system thermal index to anticipated-local index. Depends
/// only on the study-invariant [`StudyDimensions`], so it is built once per run.
///
/// Entry `t` is `Some(local_anticipated_idx)` — the position of `t` within
/// `study_dims.anticipated_thermal_indices`, used to address anticipated-decision
/// columns — when thermal `t` is anticipated, `None` otherwise.
pub(crate) struct ThermalReverseLookup {
    /// Anticipated-local slot per thermal, `None` if not anticipated.
    pub(crate) thermal_is_anticipated: Vec<Option<usize>>,
}

impl ThermalReverseLookup {
    /// Build the reverse lookup table for anticipated thermal indices.
    pub(crate) fn build(study_dims: &StudyDimensions, n_thermals: usize) -> Self {
        let mut thermal_is_anticipated = vec![None; n_thermals];
        for (local, &sys) in study_dims.anticipated_thermal_indices.iter().enumerate() {
            debug_assert!(
                sys < n_thermals,
                "anticipated_thermal_indices entry {sys} >= n_thermals {n_thermals}"
            );
            thermal_is_anticipated[sys] = Some(local);
        }
        Self {
            thermal_is_anticipated,
        }
    }
}

/// Primal of a thermal's anticipated-decision column, or `None` when the thermal
/// is not anticipated or the decision is inactive at this stage.
///
/// Gates on [`StateLayout::is_anticipated_decision_active`] rather than reading
/// then checking: an inactive column is pinned to `[0, 0]` and the predicate is
/// the canonical single-owner test.
#[inline]
fn compute_anticipated_decision_mw(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    lookup: &ThermalReverseLookup,
    thermal_local: usize,
) -> Option<f64> {
    let local_idx = lookup.thermal_is_anticipated[thermal_local]?;
    if !spec.state.is_anticipated_decision_active(
        local_idx,
        spec.stage_index,
        spec.n_stages,
        spec.anticipated_windows,
        spec.study_stage_ids,
    ) {
        return None;
    }
    // Base is the per-stage `thermal.end` (n_blks-dependent), so use `spec.geometry`,
    // never the global stage-0 indexer — that addresses the wrong column off stage 0.
    let col = spec.geometry.anticipated_decision.start + local_idx;
    debug_assert!(
        col < view.primal.len(),
        "anticipated_decision col {col} out of primal bounds {}",
        view.primal.len(),
    );
    Some(view.primal[col])
}

/// Committed MW for an anticipated thermal, or `None` when not anticipated.
///
/// The committed scalar is slot 0 of the `anticipated_state` ring buffer
/// (`anticipated_state.start + local_idx`), NOT a per-block thermal generation
/// column — those differ when block hours or generations are non-uniform, and the
/// fishing constraint pins slot 0 to the block-hours-weighted average. The read
/// applies unconditionally for any anticipated plant.
#[inline]
fn compute_anticipated_committed_mw(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    lookup: &ThermalReverseLookup,
    thermal_local: usize,
) -> Option<f64> {
    let local_idx = lookup.thermal_is_anticipated[thermal_local]?;
    // Ring buffer lives in the stage-invariant state region, so the base is the
    // role-(a) `StateLayout`, not the geometry indexer. Slot 0 = start + local_idx.
    let col = spec.state.anticipated_state.start + local_idx;
    debug_assert!(
        col < view.primal.len(),
        "anticipated_state slot-0 col {col} out of primal bounds {}",
        view.primal.len(),
    );
    Some(view.primal[col])
}

/// System entity counts needed to populate per-entity result [`Vec`]s. Every ID
/// list is in canonical ID-sorted order. Entity types that contribute no columns
/// at a stage (e.g. contracts) still carry counts so stub zero-valued entries
/// preserve entity ordering for the output writer.
#[derive(Debug, Clone)]
pub struct EntityCounts {
    /// Operating hydro plant IDs.
    pub hydro_ids: Vec<i32>,
    /// Thermal unit IDs.
    pub thermal_ids: Vec<i32>,
    /// Transmission line IDs.
    pub line_ids: Vec<i32>,
    /// Bus IDs.
    pub bus_ids: Vec<i32>,
    /// Length must equal `indexer.hydro_count`. Values are unused — per-stage
    /// productivity is read through `StageExtractionSpec::hydro_productivities`;
    /// retained for the `debug_assert!` length invariant.
    pub hydro_productivities: Vec<f64>,
    /// Pumping station IDs (empty if none).
    pub pumping_station_ids: Vec<i32>,
    /// Contract IDs (empty if none).
    pub contract_ids: Vec<i32>,
    /// Non-controllable source IDs (empty if none).
    pub non_controllable_ids: Vec<i32>,
}

/// Return the 0-based scenario ID range assigned to `rank` out of `world_size` ranks.
///
/// Uses a two-level distribution: the first `n_scenarios % world_size` ranks
/// receive one extra scenario (the "fat" group), and the remaining ranks receive
/// the floor. This matches the distribution strategy from
/// simulation-architecture.md SS3.1.
///
/// The sum of all ranks' range lengths equals `n_scenarios`.
///
/// # Panics
///
/// Panics in debug builds when `world_size == 0`.
///
/// # Examples
///
/// ```
/// use cobre_sddp::simulation::extraction::assign_scenarios;
///
/// // 10 scenarios, 3 ranks:
/// //   10 % 3 = 1  → rank 0 gets ceil(10/3) = 4 scenarios
/// //   ranks 1-2 get floor(10/3) = 3 scenarios
/// assert_eq!(assign_scenarios(10, 0, 3), 0..4);
/// assert_eq!(assign_scenarios(10, 1, 3), 4..7);
/// assert_eq!(assign_scenarios(10, 2, 3), 7..10);
///
/// // Single rank: all scenarios assigned to rank 0.
/// assert_eq!(assign_scenarios(7, 0, 1), 0..7);
/// ```
#[must_use]
pub fn assign_scenarios(n_scenarios: u32, rank: usize, world_size: usize) -> Range<u32> {
    debug_assert!(world_size > 0, "world_size must be > 0");

    let n = n_scenarios as usize;
    let r = world_size;

    let fat_count = n % r;
    let fat_size = n / r + 1;
    let lean_size = n / r;

    let (start, size): (usize, usize) = if rank < fat_count {
        (rank * fat_size, fat_size)
    } else {
        (
            fat_count * fat_size + (rank - fat_count) * lean_size,
            lean_size,
        )
    };
    let end = start + size;

    #[allow(clippy::cast_possible_truncation)]
    {
        (start as u32)..(end as u32)
    }
}

/// LP solution view passed to result extraction helpers.
pub struct SolutionView<'a> {
    /// Primal variable values from the LP solve.
    pub primal: &'a [f64],
    /// Dual variable values (shadow prices) from the LP solve.
    pub dual: &'a [f64],
    /// LP objective value.
    pub objective: f64,
    /// Objective coefficient vector from the stage template.
    pub objective_coeffs: &'a [f64],
    /// Row lower bounds from the stage template (may be patched for load noise).
    pub row_lower: &'a [f64],
}

/// Conversion factor from `hm³ · MW/(m³/s)` to `MWh`.
///
/// Unit cancellation: `hm³ × 10⁶ m³/hm³ ÷ 3600 s/h × MW/(m³/s) = MWh`.
pub const ENERGY_FACTOR_MWH_PER_HM3_PER_MW_PER_M3S: f64 = 1.0e6 / 3600.0;

/// Extraction parameters bundled for a single stage.
///
/// **Per-stage geometry contract.** Every block-major equipment read must take its
/// base AND length from `n_blks` / `geometry` (this stage's values), never a global
/// stage-0 geometry: under a non-uniform block schedule (e.g. `[1, 3, 2]`) a stage-0
/// base/stride addresses the WRONG primal columns, silently misreporting equipment
/// and the cost breakdown. `state` (role-(a), pure function of `(N, L, A, k_max)`)
/// and `study_dims` (study-invariant non-state shape) instead resolve at every
/// stage. For uniform-block studies the per-stage and stage-0 reads coincide.
pub struct StageExtractionSpec<'a> {
    /// Role-(a) state layout: source of the state-region column reads (`storage`,
    /// `storage_in`, `inflow_lags`, `anticipated_state`, `max_par_order`).
    pub state: &'a StateLayout,
    /// Single owner of the study-invariant, non-state LP shape (entity counts and
    /// optional-column presence flags).
    pub study_dims: &'a StudyDimensions,
    /// Per-stage dispatch block count, sourced from `block_counts_per_stage[t]`.
    /// Strides every equipment-column family at this stage.
    pub n_blks: usize,
    /// Stage-correct equipment geometry, resolved per stage from `StageLayout`
    /// (via `StageTemplates::geometry_per_stage`).
    pub geometry: &'a crate::lp_builder::StageGeometry,
    /// Entity ID lists and productivities needed to build result records.
    pub entity_counts: &'a EntityCounts,
    /// Volumetric inflow per hydro (m³/s), one entry per hydro plant.
    pub inflow_m3s_per_hydro: &'a [f64],
    /// Block hours per dispatch block, used to convert duals to spot prices.
    pub block_hours: &'a [f64],
    /// Per-row metadata for active generic constraint rows at this stage.
    pub generic_constraint_entries: &'a [GenericConstraintRowEntry],
    /// First NCS generation column; NCS columns are `ncs_col_start + local_idx * n_blks + blk`.
    pub ncs_col_start: usize,
    /// Number of active NCS entities at this stage.
    pub n_ncs: usize,
    /// IDs of active NCS entities, in ID-sorted order. Length equals `n_ncs`.
    pub ncs_entity_ids: &'a [i32],
    /// Per-(ncs, block) column upper bounds, `available_gen * factor`. Same
    /// block-major layout as the NCS columns, length `n_ncs * n_blks`.
    pub ncs_col_upper: &'a [f64],
    /// First pumping-flow column. Dense over ALL system stations:
    /// `pumping_col_start + p_sys * n_blks + blk` (`p_sys` = SYSTEM index).
    pub pumping_col_start: usize,
    /// Full system station count (dense); a commissioning-dormant station keeps
    /// its column pinned to `[0, 0]`.
    pub n_pumping: usize,
    /// Per-station pumping power-consumption rate \[MW/(m³/s)\]. ID-sorted, indexed
    /// by SYSTEM station index — which under the dense layout IS the column-block
    /// position, so extraction reads it at the enumeration index.
    pub pumping_consumption_mw_per_m3s: &'a [f64],
    /// RESOLVED per-contract price \[$/`MWh`\] for THIS stage, ID-sorted parallel to
    /// `entity_counts.contract_ids`. The unscaled `contract_bounds(c, t).price_per_mwh`,
    /// NOT the `col_scale`-scaled LP objective; `total_cost = price * power * hours`.
    pub contract_prices: &'a [f64],
    /// Direction per contract, ID-sorted parallel to `entity_counts.contract_ids`:
    /// `true` = import (base `geometry.contract_import.start`), `false` = export
    /// (base `geometry.contract_export.start`). The running same-direction count
    /// gives the per-family slot — `c_sys` is the wrong grid stride.
    pub contract_is_import: &'a [bool],
    /// Map from target hydro ID to source hydro indices that divert to it.
    pub diversion_upstream: &'a HashMap<EntityId, Vec<usize>>,
    /// Per-hydro productivity at this stage. `0.0` for FPHA hydros (generation is
    /// read from the LP column instead). Length equals `indexer.hydro_count`.
    pub hydro_productivities: &'a [f64],
    /// Column scaling factors. Unscale per-variable cost: `c_orig = c_scaled / col_scale[j]`.
    pub col_scale: &'a [f64],
    /// Row scaling factors, to unscale row bounds at the extraction boundary.
    pub row_scale: &'a [f64],
    /// Product of one-step discount factors for transitions before this stage; `1.0` for stage 0.
    pub cumulative_discount_factor: f64,
    /// `ρ_eq` and `ρ_acum` scalars per `(hydro, stage)` via [`EnergyConversionSet`].
    pub energy_conversion: &'a EnergyConversionSet,
    /// `V_min` per hydro (hm³), in `entity_counts.hydro_ids` order. Feeds
    /// `stored_energy_mwh = (V - V_min) · ρ_acum · ENERGY_FACTOR`.
    pub hydro_min_storage_hm3: &'a [f64],
    /// Stage index within the planning horizon (0-based).
    pub stage_index: usize,
    /// Total study stages. Evaluates the horizon-boundary predicate `t + K_i <= n_stages`.
    pub n_stages: usize,
    /// Per-plant commissioning window `(entry_stage_id, exit_stage_id)` for
    /// anticipated thermals, by anticipated-local position. Gates the
    /// anticipated-decision read via [`StateLayout::is_anticipated_decision_active`]
    /// on the same predicate the LP builder used — reading when the gate is `false`
    /// reports a decision for a `[0, 0]`-pinned column. Empty when none.
    pub anticipated_windows: &'a [(Option<i32>, Option<i32>)],
    /// Study-stage commissioning id per stage index (`study_stage_ids[t] = stage.id`).
    /// The gate keys its operation-window clause on the DELIVERY stage's id
    /// (`t + K_i`). Length equals `n_stages`.
    pub study_stage_ids: &'a [i32],
}

impl StageExtractionSpec<'_> {
    /// Return the [`BlockGrid`] address primitive striding by this stage's `n_blks`.
    #[inline]
    fn block_grid(&self) -> BlockGrid {
        BlockGrid::new(self.n_blks, self.study_dims.max_deficit_segments)
    }

    /// `col_scale[col]` when in range and non-zero; `1.0` otherwise.
    #[inline]
    fn col_scale_factor(&self, col: usize) -> f64 {
        if col < self.col_scale.len() {
            let d = self.col_scale[col];
            if d == 0.0 { 1.0 } else { d }
        } else {
            1.0
        }
    }
}

/// Extract one hydro result for the no-turbine (stage-level aggregate) branch.
fn extract_hydro_no_turbine(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    lookup: &HydroReverseLookup,
    h: usize,
    hydro_id: i32,
    stage_id: u32,
) -> SimulationHydroResult {
    let study_dims = spec.study_dims;
    let state = spec.state;
    let incremental_inflow = if h < spec.inflow_m3s_per_hydro.len() {
        spec.inflow_m3s_per_hydro[h]
    } else if state.max_par_order > 0 {
        view.primal[state.inflow_lags.start + h]
    } else {
        0.0
    };
    let inflow_slack = if study_dims.has_inflow_penalty {
        view.primal[spec.geometry.inflow_slack.start + h]
    } else {
        0.0
    };
    let withdrawal_neg = if study_dims.has_withdrawal {
        view.primal[spec.geometry.withdrawal_slack_neg.start + h]
    } else {
        0.0
    };
    let withdrawal_pos = if study_dims.has_withdrawal {
        view.primal[spec.geometry.withdrawal_slack_pos.start + h]
    } else {
        0.0
    };
    let water_value = view
        .dual
        .get(spec.geometry.water_balance.start + h)
        .copied()
        .unwrap_or(0.0)
        * COST_SCALE_FACTOR;

    // Per-block slacks aggregated to stage-level as an hours-weighted average.
    let (turbined_slack, outflow_slack_below, outflow_slack_above, generation_slack) = if study_dims
        .has_operational_violations
    {
        let grid = spec.block_grid();
        let n_blks = spec.n_blks;
        let mut tb = 0.0_f64;
        let mut ob = 0.0_f64;
        let mut oa = 0.0_f64;
        let mut gb = 0.0_f64;
        let total_hours: f64 = spec.block_hours.iter().sum();
        for blk in 0..n_blks {
            let w = spec.block_hours[blk] / total_hours;
            tb += view.primal[grid.flat(spec.geometry.turbine_below_slack.start, h, blk)] * w;
            ob += view.primal[grid.flat(spec.geometry.outflow_below_slack.start, h, blk)] * w;
            oa += view.primal[grid.flat(spec.geometry.outflow_above_slack.start, h, blk)] * w;
            gb += view.primal[grid.flat(spec.geometry.generation_below_slack.start, h, blk)] * w;
        }
        (tb, ob, oa, gb)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    let (evaporation_m3s, evaporation_violation_neg_m3s, evaporation_violation_pos_m3s) =
        if let Some(local_evap_idx) = lookup.evap[h] {
            // `evap_indices` is block-major (`local * n_blks + blk`); block 0
            // preserves the current single-block read. Per-block evaporation
            // extraction is out of scope here.
            let ei = &spec.geometry.evap_indices[local_evap_idx * spec.geometry.n_blks];
            let evaporation_flow = view.primal[ei.evaporation_flow_col];
            let neg = view.primal[ei.f_evap_plus_col]; // f_evap_plus = under-evaporation
            let pos = view.primal[ei.f_evap_minus_col]; // f_evap_minus = over-evaporation
            (Some(evaporation_flow), neg, pos)
        } else {
            (Some(0.0), 0.0, 0.0)
        };

    let conv = spec.energy_conversion.conversion(h, spec.stage_index);
    let rho_acum = spec
        .energy_conversion
        .accumulated_productivity(h, spec.stage_index);
    let v_min = spec.hydro_min_storage_hm3.get(h).copied().unwrap_or(0.0);
    let storage_initial = view.primal[state.storage_in.start + h];
    let storage_final = view.primal[state.storage.start + h];

    let filling_target_violation = read_filling_slack_primal(
        view.primal,
        &spec.geometry.filling_target_col,
        lookup.filling_target[h],
    );
    let storage_violation_below = read_filling_slack_primal(
        view.primal,
        &spec.geometry.filled_min_storage_floor_col,
        lookup.filled_min_storage_floor[h],
    );

    SimulationHydroResult {
        stage_id,
        block_id: None,
        hydro_id,
        turbined_m3s: 0.0,
        spillage_m3s: 0.0,
        evaporation_m3s,
        diverted_inflow_m3s: Some(0.0),
        diverted_outflow_m3s: Some(0.0),
        incremental_inflow_m3s: incremental_inflow,
        inflow_m3s: incremental_inflow,
        storage_initial_hm3: storage_initial,
        storage_final_hm3: storage_final,
        generation_mw: 0.0,
        equivalent_productivity_mw_per_m3s: conv.equivalent_productivity_mw_per_m3s,
        accumulated_productivity_mw_per_m3s: rho_acum,
        incremental_inflow_energy_mw: rho_acum * incremental_inflow,
        stored_energy_initial_mwh: (storage_initial - v_min)
            * rho_acum
            * ENERGY_FACTOR_MWH_PER_HM3_PER_MW_PER_M3S,
        stored_energy_final_mwh: (storage_final - v_min)
            * rho_acum
            * ENERGY_FACTOR_MWH_PER_HM3_PER_MW_PER_M3S,
        spillage_cost: 0.0,
        water_value_per_hm3: water_value,
        storage_binding_code: 0,
        operative_state_code: 1,
        turbined_slack_m3s: turbined_slack,
        outflow_slack_below_m3s: outflow_slack_below,
        outflow_slack_above_m3s: outflow_slack_above,
        generation_slack_mw: generation_slack,
        storage_violation_below_hm3: storage_violation_below,
        filling_target_violation_hm3: filling_target_violation,
        evaporation_violation_pos_m3s,
        evaporation_violation_neg_m3s,
        inflow_nonnegativity_slack_m3s: inflow_slack,
        water_withdrawal_violation_pos_m3s: withdrawal_pos,
        water_withdrawal_violation_neg_m3s: withdrawal_neg,
    }
}

/// Stage-level (non-per-block) data extracted for one hydro plant.
///
/// Captures values that are constant across all blocks within a stage so that
/// the per-block closure in [`extract_hydro_per_block`] only needs to read
/// per-block columns.
struct HydroStageContext {
    storage_final: f64,
    storage_initial: f64,
    incremental_inflow: f64,
    inflow_slack: f64,
    withdrawal_neg: f64,
    withdrawal_pos: f64,
    water_value: f64,
    fpha_local: Option<usize>,
    equivalent_productivity_mw_per_m3s: f64,
    accumulated_productivity_mw_per_m3s: f64,
    incremental_inflow_energy_mw: f64,
    stored_energy_initial_mwh: f64,
    stored_energy_final_mwh: f64,
    evaporation_m3s: Option<f64>,
    evaporation_violation_neg_m3s: f64,
    evaporation_violation_pos_m3s: f64,
    /// `σ_fill` terminal-target slack (hm³); read once, repeated across per-block rows.
    filling_target_violation: f64,
    /// `σ^{v-}` operating-floor slack (hm³); read once, repeated across per-block rows.
    storage_violation_below: f64,
}

impl HydroStageContext {
    /// Read all stage-level scalars for hydro at system index `h`.
    fn new(
        view: &SolutionView<'_>,
        spec: &StageExtractionSpec<'_>,
        lookup: &HydroReverseLookup,
        h: usize,
    ) -> Self {
        let study_dims = spec.study_dims;
        let state = spec.state;
        let storage_final = view.primal[state.storage.start + h];
        let storage_initial = view.primal[state.storage_in.start + h];
        let incremental_inflow = if h < spec.inflow_m3s_per_hydro.len() {
            spec.inflow_m3s_per_hydro[h]
        } else if state.max_par_order > 0 {
            view.primal[state.inflow_lags.start + h]
        } else {
            0.0
        };
        let inflow_slack = if study_dims.has_inflow_penalty {
            view.primal[spec.geometry.inflow_slack.start + h]
        } else {
            0.0
        };
        let withdrawal_neg = if study_dims.has_withdrawal {
            view.primal[spec.geometry.withdrawal_slack_neg.start + h]
        } else {
            0.0
        };
        let withdrawal_pos = if study_dims.has_withdrawal {
            view.primal[spec.geometry.withdrawal_slack_pos.start + h]
        } else {
            0.0
        };
        let water_value = view
            .dual
            .get(spec.geometry.water_balance.start + h)
            .copied()
            .unwrap_or(0.0)
            * COST_SCALE_FACTOR;
        let fpha_local = lookup.fpha[h];
        let (evaporation_m3s, evaporation_violation_neg_m3s, evaporation_violation_pos_m3s) =
            if let Some(lei) = lookup.evap[h] {
                // Block-major `evap_indices`; block 0 preserves the single-block read
                // (per-block evaporation extraction is out of scope here).
                let ei = &spec.geometry.evap_indices[lei * spec.geometry.n_blks];
                let evaporation_flow = view.primal[ei.evaporation_flow_col];
                let neg = view.primal[ei.f_evap_plus_col]; // f_evap_plus = under-evaporation
                let pos = view.primal[ei.f_evap_minus_col]; // f_evap_minus = over-evaporation
                (Some(evaporation_flow), neg, pos)
            } else {
                (Some(0.0), 0.0, 0.0)
            };
        let filling_target_violation = read_filling_slack_primal(
            view.primal,
            &spec.geometry.filling_target_col,
            lookup.filling_target[h],
        );
        let storage_violation_below = read_filling_slack_primal(
            view.primal,
            &spec.geometry.filled_min_storage_floor_col,
            lookup.filled_min_storage_floor[h],
        );
        let conv = spec.energy_conversion.conversion(h, spec.stage_index);
        let rho_acum = spec
            .energy_conversion
            .accumulated_productivity(h, spec.stage_index);
        let v_min = spec.hydro_min_storage_hm3.get(h).copied().unwrap_or(0.0);
        Self {
            storage_final,
            storage_initial,
            incremental_inflow,
            inflow_slack,
            withdrawal_neg,
            withdrawal_pos,
            water_value,
            fpha_local,
            equivalent_productivity_mw_per_m3s: conv.equivalent_productivity_mw_per_m3s,
            accumulated_productivity_mw_per_m3s: rho_acum,
            incremental_inflow_energy_mw: rho_acum * incremental_inflow,
            stored_energy_initial_mwh: (storage_initial - v_min)
                * rho_acum
                * ENERGY_FACTOR_MWH_PER_HM3_PER_MW_PER_M3S,
            stored_energy_final_mwh: (storage_final - v_min)
                * rho_acum
                * ENERGY_FACTOR_MWH_PER_HM3_PER_MW_PER_M3S,
            evaporation_m3s,
            evaporation_violation_neg_m3s,
            evaporation_violation_pos_m3s,
            filling_target_violation,
            storage_violation_below,
        }
    }
}

/// Extract per-block hydro results for one hydro plant (turbined/spillage branch).
fn extract_hydro_per_block<'a>(
    view: &'a SolutionView<'a>,
    spec: &'a StageExtractionSpec<'a>,
    lookup: &'a HydroReverseLookup,
    h: usize,
    hydro_id: i32,
    stage_id: u32,
) -> impl Iterator<Item = SimulationHydroResult> + 'a {
    let study_dims = spec.study_dims;
    let n_blks = spec.n_blks;
    let grid = spec.block_grid();

    let ctx = HydroStageContext::new(view, spec, lookup, h);

    let hydro_entity_id = EntityId(hydro_id);
    let div_sources = spec.diversion_upstream.get(&hydro_entity_id);

    (0..n_blks).map(move |b| {
        let t_col = grid.flat(spec.geometry.turbine.start, h, b);
        let s_col = grid.flat(spec.geometry.spillage.start, h, b);
        let turbined = view.primal[t_col];
        let spillage = view.primal[s_col];

        let diverted_outflow = if spec.geometry.diversion.is_empty() {
            0.0
        } else {
            view.primal[grid.flat(spec.geometry.diversion.start, h, b)]
        };

        // Diversion columns are flat block-major over the source hydro index, so
        // address them with `flat`, not the 3-term deficit shape.
        let diverted_inflow = if let Some(sources) = div_sources {
            let mut total = 0.0;
            for &d_idx in sources {
                total += view.primal[grid.flat(spec.geometry.diversion.start, d_idx, b)];
            }
            total
        } else {
            0.0
        };

        // FPHA hydros read generation from the LP `g_{h,k}` column; constant-
        // productivity hydros compute it as turbined * productivity.
        let generation_mw = if let Some(local_fpha_idx) = ctx.fpha_local {
            view.primal[grid.flat(spec.geometry.generation.start, local_fpha_idx, b)]
        } else {
            turbined * spec.hydro_productivities[h]
        };

        let (turbined_slack, outflow_slack_below, outflow_slack_above, generation_slack) =
            if study_dims.has_operational_violations {
                (
                    view.primal[grid.flat(spec.geometry.turbine_below_slack.start, h, b)],
                    view.primal[grid.flat(spec.geometry.outflow_below_slack.start, h, b)],
                    view.primal[grid.flat(spec.geometry.outflow_above_slack.start, h, b)],
                    view.primal[grid.flat(spec.geometry.generation_below_slack.start, h, b)],
                )
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };

        #[allow(clippy::cast_possible_truncation)]
        SimulationHydroResult {
            stage_id,
            block_id: Some(b as u32),
            hydro_id,
            turbined_m3s: turbined,
            spillage_m3s: spillage,
            evaporation_m3s: ctx.evaporation_m3s,
            diverted_inflow_m3s: Some(diverted_inflow),
            diverted_outflow_m3s: Some(diverted_outflow),
            incremental_inflow_m3s: ctx.incremental_inflow,
            inflow_m3s: ctx.incremental_inflow,
            storage_initial_hm3: ctx.storage_initial,
            storage_final_hm3: ctx.storage_final,
            generation_mw,
            equivalent_productivity_mw_per_m3s: ctx.equivalent_productivity_mw_per_m3s,
            accumulated_productivity_mw_per_m3s: ctx.accumulated_productivity_mw_per_m3s,
            incremental_inflow_energy_mw: ctx.incremental_inflow_energy_mw,
            stored_energy_initial_mwh: ctx.stored_energy_initial_mwh,
            stored_energy_final_mwh: ctx.stored_energy_final_mwh,
            spillage_cost: spillage * view.objective_coeffs[s_col] / spec.col_scale_factor(s_col)
                * COST_SCALE_FACTOR,
            water_value_per_hm3: ctx.water_value,
            storage_binding_code: 0,
            operative_state_code: 1,
            turbined_slack_m3s: turbined_slack,
            outflow_slack_below_m3s: outflow_slack_below,
            outflow_slack_above_m3s: outflow_slack_above,
            generation_slack_mw: generation_slack,
            storage_violation_below_hm3: ctx.storage_violation_below,
            filling_target_violation_hm3: ctx.filling_target_violation,
            evaporation_violation_pos_m3s: ctx.evaporation_violation_pos_m3s,
            evaporation_violation_neg_m3s: ctx.evaporation_violation_neg_m3s,
            inflow_nonnegativity_slack_m3s: ctx.inflow_slack,
            water_withdrawal_violation_pos_m3s: ctx.withdrawal_pos,
            water_withdrawal_violation_neg_m3s: ctx.withdrawal_neg,
        }
    })
}

fn extract_hydros(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
    lookup: &HydroReverseLookup,
) -> Vec<SimulationHydroResult> {
    if spec.geometry.turbine.is_empty() || spec.n_blks == 0 {
        spec.entity_counts
            .hydro_ids
            .iter()
            .enumerate()
            .map(|(h, &hydro_id)| {
                extract_hydro_no_turbine(view, spec, lookup, h, hydro_id, stage_id)
            })
            .collect()
    } else {
        spec.entity_counts
            .hydro_ids
            .iter()
            .enumerate()
            .flat_map(|(h, &hydro_id)| {
                extract_hydro_per_block(view, spec, lookup, h, hydro_id, stage_id)
            })
            .collect()
    }
}

/// Extract thermal results from a raw LP solution view.
fn extract_thermals(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
    lookup: &ThermalReverseLookup,
) -> Vec<SimulationThermalResult> {
    let n_blks = spec.n_blks;
    if spec.geometry.thermal.is_empty() || n_blks == 0 {
        spec.entity_counts
            .thermal_ids
            .iter()
            .enumerate()
            .map(|(t, &thermal_id)| SimulationThermalResult {
                stage_id,
                block_id: None,
                thermal_id,
                generation_mw: 0.0,
                generation_cost: 0.0,
                is_anticipated: lookup.thermal_is_anticipated[t].is_some(),
                anticipated_committed_mw: compute_anticipated_committed_mw(view, spec, lookup, t),
                anticipated_decision_mw: compute_anticipated_decision_mw(view, spec, lookup, t),
                operative_state_code: 1,
            })
            .collect()
    } else {
        let grid = spec.block_grid();
        let mut results = Vec::with_capacity(spec.entity_counts.thermal_ids.len() * n_blks);
        for (t, &thermal_id) in spec.entity_counts.thermal_ids.iter().enumerate() {
            let is_anticipated = lookup.thermal_is_anticipated[t].is_some();
            let anticipated_decision_mw = compute_anticipated_decision_mw(view, spec, lookup, t);
            // Per-plant per-stage scalar; hoisted out of the per-block loop.
            let anticipated_committed_mw = compute_anticipated_committed_mw(view, spec, lookup, t);
            for b in 0..n_blks {
                let col = grid.flat(spec.geometry.thermal.start, t, b);
                let gen_mw = view.primal[col];
                #[allow(clippy::cast_possible_truncation)]
                results.push(SimulationThermalResult {
                    stage_id,
                    block_id: Some(b as u32),
                    thermal_id,
                    generation_mw: gen_mw,
                    generation_cost: gen_mw * view.objective_coeffs[col]
                        / spec.col_scale_factor(col)
                        * COST_SCALE_FACTOR,
                    is_anticipated,
                    anticipated_committed_mw,
                    anticipated_decision_mw,
                    operative_state_code: 1,
                });
            }
        }
        results
    }
}

/// Extract exchange (line flow) results from a raw LP solution view.
fn extract_exchanges(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> Vec<SimulationExchangeResult> {
    let n_blks = spec.n_blks;
    if spec.geometry.line_fwd.is_empty() || n_blks == 0 {
        spec.entity_counts
            .line_ids
            .iter()
            .map(|&line_id| SimulationExchangeResult {
                stage_id,
                block_id: None,
                line_id,
                direct_flow_mw: 0.0,
                reverse_flow_mw: 0.0,
                exchange_cost: 0.0,
                operative_state_code: 1,
            })
            .collect()
    } else {
        let grid = spec.block_grid();
        spec.entity_counts
            .line_ids
            .iter()
            .enumerate()
            .flat_map(move |(l, &line_id)| {
                (0..n_blks).map(move |b| {
                    let fwd_col = grid.flat(spec.geometry.line_fwd.start, l, b);
                    let rev_col = grid.flat(spec.geometry.line_rev.start, l, b);
                    let fwd = view.primal[fwd_col];
                    let rev = view.primal[rev_col];
                    #[allow(clippy::cast_possible_truncation)]
                    SimulationExchangeResult {
                        stage_id,
                        block_id: Some(b as u32),
                        line_id,
                        direct_flow_mw: fwd,
                        reverse_flow_mw: rev,
                        exchange_cost: (fwd * view.objective_coeffs[fwd_col]
                            / spec.col_scale_factor(fwd_col)
                            + rev * view.objective_coeffs[rev_col]
                                / spec.col_scale_factor(rev_col))
                            * COST_SCALE_FACTOR,
                        operative_state_code: 2,
                    }
                })
            })
            .collect()
    }
}

/// Extract bus results from a raw LP solution view.
fn extract_buses(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> Vec<SimulationBusResult> {
    let n_blks = spec.n_blks;
    if spec.geometry.deficit.is_empty() || n_blks == 0 {
        spec.entity_counts
            .bus_ids
            .iter()
            .map(|&bus_id| SimulationBusResult {
                stage_id,
                block_id: None,
                bus_id,
                load_mw: 0.0,
                deficit_mw: 0.0,
                excess_mw: 0.0,
                spot_price: 0.0,
            })
            .collect()
    } else {
        let grid = spec.block_grid();
        let max_segs = spec.study_dims.max_deficit_segments;
        spec.entity_counts
            .bus_ids
            .iter()
            .enumerate()
            .flat_map(move |(bus_idx, &bus_id)| {
                (0..n_blks).map(move |b| {
                    // Deficit is the 3-term bus-outer/segment-middle/block-inner
                    // shape, so address it with `deficit`, not `flat`.
                    let deficit_mw: f64 = (0..max_segs)
                        .map(|s| {
                            let col = grid.deficit(spec.geometry.deficit.start, bus_idx, s, b);
                            view.primal[col]
                        })
                        .sum();
                    let excess_col = grid.flat(spec.geometry.excess.start, bus_idx, b);
                    let load_row = grid.flat(spec.geometry.load_balance.start, bus_idx, b);
                    let raw_dual = view.dual.get(load_row).copied().unwrap_or(0.0);
                    let hrs = spec.block_hours.get(b).copied().unwrap_or(0.0);
                    #[allow(clippy::cast_possible_truncation)]
                    SimulationBusResult {
                        stage_id,
                        block_id: Some(b as u32),
                        bus_id,
                        load_mw: view.row_lower[load_row],
                        deficit_mw,
                        excess_mw: view.primal[excess_col],
                        spot_price: if hrs > 0.0 {
                            raw_dual * COST_SCALE_FACTOR / hrs
                        } else {
                            0.0
                        },
                    }
                })
            })
            .collect()
    }
}

/// Extract a [`SimulationStageResult`] from a raw LP solution at one stage.
///
/// Reads role-(b) equipment column values from `view.primal` using the ranges
/// stored in `spec.geometry` (the per-stage [`StageGeometry`](crate::lp_builder::StageGeometry));
/// role-(a) state columns resolve via `spec.state` ([`StateLayout`]). When a
/// family has zero entities its range is empty (`0..0`) and that result defaults
/// to zero.
///
/// The LP objective is split into `future_cost = primal[spec.state.theta]` and
/// `stage_cost = objective - future_cost`, following the same convention as the
/// training forward pass.
///
/// # Preconditions
///
/// - `view.primal.len() >= spec.state.theta + 1`
/// - `spec.entity_counts.hydro_ids.len() == spec.state.hydro_count`
/// - `spec.entity_counts.hydro_productivities.len() == spec.state.hydro_count`
/// - `view.objective_coeffs.len() >= view.primal.len()` when equipment ranges are non-empty
/// - `view.row_lower.len() >= spec.geometry.load_balance.end` when `load_balance` is non-empty
/// - `stage_id` is 0-based
///
/// Violations are caught by `debug_assert!` in debug builds.
///
/// # Performance
///
/// Builds the reverse-lookup tables on every call. On the hot path use
/// `extract_stage_result_with_lookups` with pre-built lookups instead.
#[must_use]
pub fn extract_stage_result(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> SimulationStageResult {
    let n_hydros = spec.entity_counts.hydro_ids.len();
    let n_thermals = spec.entity_counts.thermal_ids.len();
    let hydro_lookup = HydroReverseLookup::build(spec.geometry, n_hydros);
    let thermal_lookup = ThermalReverseLookup::build(spec.study_dims, n_thermals);
    extract_stage_result_with_lookups(view, spec, stage_id, &hydro_lookup, &thermal_lookup)
}

/// Extract a [`SimulationStageResult`] using pre-built reverse-lookup tables.
///
/// Identical to [`extract_stage_result`] but avoids building the
/// [`HydroReverseLookup`] and [`ThermalReverseLookup`] tables on every call.
///
/// `thermal_lookup` is study-invariant (one for the whole run); `hydro_lookup` is
/// the lookup for **this stage** (FPHA/evap membership is per-`(hydro, stage)`).
/// Build the thermal lookup and the per-stage hydro lookups once per simulation
/// run (or per worker thread) and pass the stage's entries by reference here to
/// eliminate per-`(scenario, stage)` allocations on the hot path.
///
/// # Preconditions
///
/// Same as [`extract_stage_result`] plus:
/// - `hydro_lookup` was built from this stage's [`StageGeometry`] and `n_hydros`.
/// - `thermal_lookup` was built from the same `(study_dims, n_thermals)` pair used here.
pub(crate) fn extract_stage_result_with_lookups(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
    hydro_lookup: &HydroReverseLookup,
    thermal_lookup: &ThermalReverseLookup,
) -> SimulationStageResult {
    let state = spec.state;
    debug_assert!(
        view.primal.len() > state.theta,
        "primal vector too short: len={}, need > theta={}",
        view.primal.len(),
        state.theta
    );
    debug_assert!(
        spec.entity_counts.hydro_ids.len() == state.hydro_count,
        "hydro_ids length {} does not match state.hydro_count {}",
        spec.entity_counts.hydro_ids.len(),
        state.hydro_count
    );
    // Bounds guard against the per-stage geometry's excess end, NOT the global
    // stage-0 `indexer.excess.end` (n_blks-dependent), so a non-uniform stage with
    // fewer blocks than stage 0 cannot spuriously trip this on a stale stage-0 end.
    debug_assert!(
        spec.geometry.excess.is_empty() || view.objective_coeffs.len() >= spec.geometry.excess.end,
        "objective_coeffs too short: len={}, need >= excess.end={}",
        view.objective_coeffs.len(),
        spec.geometry.excess.end
    );
    debug_assert!(
        spec.entity_counts.hydro_productivities.len() == state.hydro_count,
        "hydro_productivities length {} does not match state.hydro_count {}",
        spec.entity_counts.hydro_productivities.len(),
        state.hydro_count
    );
    // Bound is the per-stage row end `start + n_buses * n_blks`, not `load_balance.end`,
    // which is striped by stage 0's block count.
    let load_balance = &spec.geometry.load_balance;
    let load_balance_end = if load_balance.is_empty() {
        load_balance.end
    } else {
        load_balance.start + spec.entity_counts.bus_ids.len() * spec.n_blks
    };
    debug_assert!(
        load_balance.is_empty() || view.row_lower.len() >= load_balance_end,
        "row_lower too short: len={}, need >= load_balance_end={load_balance_end}",
        view.row_lower.len(),
    );

    let (generic_violations, generic_violation_cost) =
        extract_generic_violations(view, spec, stage_id);
    let (non_controllables, ncs_curtailment_cost) = extract_non_controllables(view, spec, stage_id);
    let costs = vec![compute_cost_result(
        view,
        spec.study_dims,
        spec.geometry,
        spec.state,
        spec.col_scale,
        generic_violation_cost,
        spec.cumulative_discount_factor,
        ncs_curtailment_cost,
        stage_id,
    )];
    let (inflow_lags, pumping_stations, contracts) = extract_stub_collections(view, spec, stage_id);

    SimulationStageResult {
        stage_id,
        costs,
        hydros: extract_hydros(view, spec, stage_id, hydro_lookup),
        thermals: extract_thermals(view, spec, stage_id, thermal_lookup),
        exchanges: extract_exchanges(view, spec, stage_id),
        buses: extract_buses(view, spec, stage_id),
        pumping_stations,
        contracts,
        non_controllables,
        inflow_lags,
        generic_violations,
    }
}

/// Per-constraint hydro violation costs extracted from a solution view.
struct HydroViolationCosts {
    evaporation: f64,
    withdrawal: f64,
    outflow_below: f64,
    outflow_above: f64,
    turbined: f64,
    generation: f64,
}

impl HydroViolationCosts {
    fn total(&self) -> f64 {
        self.evaporation
            + self.withdrawal
            + self.outflow_below
            + self.outflow_above
            + self.turbined
            + self.generation
    }
}

/// Compute the 6 per-constraint hydro violation costs from a solution view.
fn compute_hydro_violation_costs(
    study_dims: &StudyDimensions,
    equipment: &crate::lp_builder::StageGeometry,
    col_cost: impl Fn(usize) -> f64,
    range_sum: impl Fn(std::ops::Range<usize>) -> f64,
) -> HydroViolationCosts {
    let evaporation = equipment
        .evap_indices
        .iter()
        .map(|ei| col_cost(ei.f_evap_plus_col) + col_cost(ei.f_evap_minus_col))
        .sum::<f64>()
        * COST_SCALE_FACTOR;

    let withdrawal = if equipment.withdrawal_slack_neg.is_empty() {
        0.0
    } else {
        (range_sum(equipment.withdrawal_slack_neg.clone())
            + range_sum(equipment.withdrawal_slack_pos.clone()))
            * COST_SCALE_FACTOR
    };

    let (outflow_below, outflow_above, turbined, generation) =
        if study_dims.has_operational_violations {
            (
                range_sum(equipment.outflow_below_slack.clone()) * COST_SCALE_FACTOR,
                range_sum(equipment.outflow_above_slack.clone()) * COST_SCALE_FACTOR,
                range_sum(equipment.turbine_below_slack.clone()) * COST_SCALE_FACTOR,
                range_sum(equipment.generation_below_slack.clone()) * COST_SCALE_FACTOR,
            )
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

    HydroViolationCosts {
        evaporation,
        withdrawal,
        outflow_below,
        outflow_above,
        turbined,
        generation,
    }
}

/// Compute the single-stage cost breakdown from an LP solution view.
///
/// All cost fields are returned in original monetary units. The LP operates
/// in scaled cost space (objective coefficients divided by [`COST_SCALE_FACTOR`]);
/// this function multiplies back by [`COST_SCALE_FACTOR`] at the reporting
/// boundary to recover original units.
fn compute_cost_result(
    view: &SolutionView<'_>,
    study_dims: &StudyDimensions,
    equipment: &crate::lp_builder::StageGeometry,
    state: &StateLayout,
    col_scale: &[f64],
    generic_violation_cost: f64,
    cumulative_discount_factor: f64,
    ncs_curtailment_cost: f64,
    stage_id: u32,
) -> SimulationCostResult {
    let scale_factor = |col: usize| -> f64 {
        if col < col_scale.len() {
            let d = col_scale[col];
            if d == 0.0 { 1.0 } else { d }
        } else {
            1.0
        }
    };
    let col_cost = |col: usize| view.primal[col] * view.objective_coeffs[col] / scale_factor(col);
    let range_sum = |r: std::ops::Range<usize>| -> f64 { r.map(col_cost).sum() };

    let theta_obj_coeff = view
        .objective_coeffs
        .get(state.theta)
        .copied()
        .unwrap_or(1.0);
    let theta_contribution = view.primal[state.theta] * theta_obj_coeff;
    let future_cost = theta_contribution * COST_SCALE_FACTOR;
    let immediate_cost = (view.objective - theta_contribution) * COST_SCALE_FACTOR;

    // Every range summed below must sum the whole per-stage `equipment` family, not
    // just active columns: this is what keeps `Σ(breakdown) == immediate_cost`.
    let thermal_cost = if equipment.thermal.is_empty() {
        0.0
    } else {
        range_sum(equipment.thermal.clone()) * COST_SCALE_FACTOR
    };
    // Inactive anticipated-decision columns are `[0, 0]`-pinned (primal 0), so
    // summing the whole range books the fuel only where the decision is live —
    // matching `immediate_cost`.
    let anticipated_thermal_cost = if equipment.anticipated_decision.is_empty() {
        0.0
    } else {
        range_sum(equipment.anticipated_decision.clone()) * COST_SCALE_FACTOR
    };
    // Contract objective coeff is `price_per_mwh * block_hours`, so `col_cost`
    // sums `power * price * hours` with the stored sign (export price < 0 nets
    // negative). The objective term is in `immediate_cost`; booking it here keeps
    // `Σ(macro categories) == immediate_cost` when contracts are active.
    let contract_cost =
        if equipment.contract_import.is_empty() && equipment.contract_export.is_empty() {
            0.0
        } else {
            equipment
                .contract_import
                .clone()
                .chain(equipment.contract_export.clone())
                .map(col_cost)
                .sum::<f64>()
                * COST_SCALE_FACTOR
        };
    let spillage_cost = if equipment.spillage.is_empty() {
        0.0
    } else {
        range_sum(equipment.spillage.clone()) * COST_SCALE_FACTOR
    };
    let exchange_cost = if equipment.line_fwd.is_empty() {
        0.0
    } else {
        equipment
            .line_fwd
            .clone()
            .chain(equipment.line_rev.clone())
            .map(col_cost)
            .sum::<f64>()
            * COST_SCALE_FACTOR
    };
    let deficit_cost = if equipment.deficit.is_empty() {
        0.0
    } else {
        range_sum(equipment.deficit.clone()) * COST_SCALE_FACTOR
    };
    let excess_cost = if equipment.excess.is_empty() {
        0.0
    } else {
        range_sum(equipment.excess.clone()) * COST_SCALE_FACTOR
    };
    let turbined_cost = if equipment.turbine.is_empty() {
        0.0
    } else {
        range_sum(equipment.turbine.clone()) * COST_SCALE_FACTOR
    };
    let inflow_penalty_cost = if equipment.inflow_slack.is_empty() {
        0.0
    } else {
        range_sum(equipment.inflow_slack.clone()) * COST_SCALE_FACTOR
    };
    let diversion_cost = if equipment.diversion.is_empty() {
        0.0
    } else {
        range_sum(equipment.diversion.clone()) * COST_SCALE_FACTOR
    };

    let hv = compute_hydro_violation_costs(study_dims, equipment, col_cost, range_sum);

    SimulationCostResult {
        stage_id,
        block_id: None,
        total_cost: view.objective * COST_SCALE_FACTOR,
        immediate_cost,
        future_cost,
        discount_factor: cumulative_discount_factor,
        thermal_cost,
        anticipated_thermal_cost,
        contract_cost,
        deficit_cost,
        excess_cost,
        storage_violation_cost: 0.0,
        filling_target_cost: 0.0,
        hydro_violation_cost: hv.total(),
        outflow_violation_below_cost: hv.outflow_below,
        outflow_violation_above_cost: hv.outflow_above,
        turbined_violation_cost: hv.turbined,
        generation_violation_cost: hv.generation,
        evaporation_violation_cost: hv.evaporation,
        withdrawal_violation_cost: hv.withdrawal,
        inflow_penalty_cost,
        generic_violation_cost,
        spillage_cost: spillage_cost + diversion_cost,
        turbined_cost,
        curtailment_cost: ncs_curtailment_cost,
        exchange_cost,
        pumping_cost: 0.0,
    }
}

/// Extract generic constraint violation results from a solved LP.
///
/// For an `==` constraint (two slack columns) the reported `slack_value` is the
/// net violation `s_plus - s_minus`, while its cost charges both (`s_plus + s_minus`).
fn extract_generic_violations(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> (Vec<SimulationGenericViolationResult>, f64) {
    let entries = spec.generic_constraint_entries;
    if entries.is_empty() {
        return (Vec::new(), 0.0);
    }

    let mut results = Vec::with_capacity(entries.len());
    let mut total_cost = 0.0;

    for entry in entries {
        // A stage-level row is priced by total stage hours (matching the LP
        // objective in `fill_generic_constraint_entries`); a per-block row by its block's.
        let block_hours = if entry.is_stage_level {
            spec.block_hours.iter().sum()
        } else {
            spec.block_hours
                .get(entry.block_idx)
                .copied()
                .unwrap_or(0.0)
        };
        let (slack_value, slack_cost) = if entry.slack_enabled {
            match entry.sense {
                ConstraintSense::Equal => {
                    let s_plus = entry.slack_plus_col.map_or(0.0, |col| view.primal[col]);
                    let s_minus = entry.slack_minus_col.map_or(0.0, |col| view.primal[col]);
                    let net = s_plus - s_minus;
                    let cost = (s_plus + s_minus) * entry.slack_penalty * block_hours;
                    (net, cost)
                }
                ConstraintSense::LessEqual | ConstraintSense::GreaterEqual => {
                    let s = entry.slack_plus_col.map_or(0.0, |col| view.primal[col]);
                    let cost = s * entry.slack_penalty * block_hours;
                    (s, cost)
                }
            }
        } else {
            (0.0, 0.0)
        };

        total_cost += slack_cost;

        results.push(SimulationGenericViolationResult {
            stage_id,
            // SAFETY: block_idx is a stage block index, always < n_blocks which is << 2^32.
            #[allow(clippy::cast_possible_truncation)]
            block_id: if entry.is_stage_level {
                None
            } else {
                Some(entry.block_idx as u32)
            },
            constraint_id: entry.entity_id,
            slack_value,
            slack_cost,
        });
    }

    (results, total_cost)
}

/// Extract NCS generation results from a solved LP — dense, one row per system
/// NCS at every stage.
///
/// The total curtailment cost is negated so it is positive in the breakdown. A
/// commissioning-dormant NCS (`[0, 0]`-pinned column) emits a ZERO row rather than
/// being absent — uniform with how thermal/line report a zeroed entity.
fn extract_non_controllables(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> (Vec<SimulationNonControllableResult>, f64) {
    let n_ncs = spec.n_ncs;
    if n_ncs == 0 {
        return (Vec::new(), 0.0);
    }

    let n_blks = spec.n_blks;
    let grid = spec.block_grid();
    let col_start = spec.ncs_col_start;
    let mut results = Vec::with_capacity(n_ncs * n_blks);
    let mut total_curtailment_cost = 0.0;

    for (local_idx, &ncs_id) in spec.ncs_entity_ids.iter().enumerate() {
        for blk in 0..n_blks {
            let col = grid.flat(col_start, local_idx, blk);
            let generation_mw = view.primal[col];
            // `ncs_col_upper` is the same block-major layout zero-based, so `flat` from 0.
            let col_upper_offset = grid.flat(0, local_idx, blk);
            debug_assert!(
                col_upper_offset < spec.ncs_col_upper.len(),
                "NCS col_upper out of bounds: offset {col_upper_offset}, len {}",
                spec.ncs_col_upper.len()
            );
            let available_mw = spec.ncs_col_upper[col_upper_offset];
            let curtailment_mw = available_mw - generation_mw;
            // NCS obj coefficient is negative, so negate to report a positive cost.
            let col_cost = -(curtailment_mw * view.objective_coeffs[col]
                / spec.col_scale_factor(col))
                * COST_SCALE_FACTOR;
            total_curtailment_cost += col_cost;

            #[allow(clippy::cast_possible_truncation)]
            results.push(SimulationNonControllableResult {
                stage_id,
                block_id: Some(blk as u32),
                non_controllable_id: ncs_id,
                generation_mw,
                available_mw,
                curtailment_mw,
                curtailment_cost: col_cost,
                operative_state_code: 1,
            });
        }
    }

    (results, total_curtailment_cost)
}

/// Extract one [`SimulationPumpingResult`] per (station, block) from the solved
/// pumping-flow primals — dense, one row per system station at every stage.
///
/// The flow is NOT divided by `col_scale` — `view.primal` is already unscaled —
/// and `power_consumption_mw = pumped_flow_m3s * consumption[p_sys]` reuses the
/// same coefficient the `PumpingPower` resolver applies on the bus load-balance
/// row. Under the dense layout the enumeration index IS the SYSTEM index, so a
/// commissioning-dormant station emits a ZERO row rather than being absent.
/// `pumping_cost` is imputed `0.0` here, finalized by the output writer.
fn extract_pumping_stations(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> Vec<SimulationPumpingResult> {
    let n_pumping = spec.n_pumping;
    let n_blks = spec.n_blks;
    if n_pumping == 0 || n_blks == 0 {
        return Vec::new();
    }

    let col_start = spec.pumping_col_start;
    debug_assert!(
        view.primal.len() >= col_start + n_pumping * n_blks,
        "pumping primal out of bounds: need {}, have {}",
        col_start + n_pumping * n_blks,
        view.primal.len()
    );

    let grid = spec.block_grid();
    let mut results = Vec::with_capacity(n_pumping * n_blks);
    for p_sys in 0..n_pumping {
        debug_assert!(
            p_sys < spec.entity_counts.pumping_station_ids.len(),
            "pumping system index {p_sys} out of bounds for pumping_station_ids len {}",
            spec.entity_counts.pumping_station_ids.len()
        );
        let pumping_station_id = spec.entity_counts.pumping_station_ids[p_sys];
        let consumption = spec.pumping_consumption_mw_per_m3s[p_sys];
        for blk in 0..n_blks {
            let col = grid.flat(col_start, p_sys, blk);
            let pumped_flow_m3s = view.primal[col];
            #[allow(clippy::cast_possible_truncation)]
            results.push(SimulationPumpingResult {
                stage_id,
                block_id: Some(blk as u32),
                pumping_station_id,
                pumped_flow_m3s,
                power_consumption_mw: pumped_flow_m3s * consumption,
                pumping_cost: 0.0,
                operative_state_code: 1,
            });
        }
    }
    results
}

/// Extract one [`SimulationContractResult`] per (contract, block) from the solved
/// dispatch primals — dense, one row per system contract at every stage.
///
/// The family base is `geometry.contract_import.start` (import) or
/// `geometry.contract_export.start` (export); the per-family slot is the running
/// count of same-direction contracts preceding `c` in ID-sorted order — `c` itself
/// is the wrong grid stride (imports and exports share one ID-sorted list but
/// occupy separate column blocks). `power_mw` is read directly from `view.primal`
/// (already unscaled). `total_cost = price * power_mw * block_hours` uses the
/// RESOLVED price, not the `col_scale`-scaled LP objective. A dormant `[0, 0]`-pinned
/// contract emits a ZERO row with `operative_state_code = 1`.
fn extract_contracts(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> Vec<SimulationContractResult> {
    let n_contracts = spec.entity_counts.contract_ids.len();
    let n_blks = spec.n_blks;
    if n_contracts == 0 || n_blks == 0 {
        return Vec::new();
    }

    let import_base = spec.geometry.contract_import.start;
    let export_base = spec.geometry.contract_export.start;
    let import_end = spec.geometry.contract_import.end;
    let export_end = spec.geometry.contract_export.end;
    debug_assert!(
        view.primal.len() >= import_end && view.primal.len() >= export_end,
        "contract primal out of bounds: need import_end {import_end} / export_end {export_end}, have {}",
        view.primal.len()
    );

    let grid = spec.block_grid();
    let mut import_slot = 0_usize;
    let mut export_slot = 0_usize;
    let mut results = Vec::with_capacity(n_contracts * n_blks);
    for (c, &contract_id) in spec.entity_counts.contract_ids.iter().enumerate() {
        let is_import = spec.contract_is_import[c];
        let (base, family_slot) = if is_import {
            let slot = import_slot;
            import_slot += 1;
            (import_base, slot)
        } else {
            let slot = export_slot;
            export_slot += 1;
            (export_base, slot)
        };
        let price = spec.contract_prices[c];
        for blk in 0..n_blks {
            let col = grid.flat(base, family_slot, blk);
            let power_mw = view.primal[col];
            let dur = spec.block_hours[blk];
            let energy_mwh = power_mw * dur;
            let total_cost = price * energy_mwh;
            #[allow(clippy::cast_possible_truncation)]
            results.push(SimulationContractResult {
                stage_id,
                block_id: Some(blk as u32),
                contract_id,
                power_mw,
                price_per_mwh: price,
                total_cost,
                operative_state_code: 1,
            });
        }
    }
    results
}

/// Extract per-entity result collections grouped by their shared iteration pattern.
///
/// Inflow lags, pumping stations, and contracts read real primal values; the
/// pumping and contract reads are delegated to [`extract_pumping_stations`] and
/// [`extract_contracts`].
fn extract_stub_collections(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> (
    Vec<SimulationInflowLagResult>,
    Vec<SimulationPumpingResult>,
    Vec<SimulationContractResult>,
) {
    let state = spec.state;
    let inflow_lags: Vec<SimulationInflowLagResult> = spec
        .entity_counts
        .hydro_ids
        .iter()
        .enumerate()
        .flat_map(|(h, &hydro_id)| {
            (0..state.max_par_order).map(move |l| {
                #[allow(clippy::cast_possible_truncation)]
                SimulationInflowLagResult {
                    stage_id,
                    hydro_id,
                    lag_index: l as u32,
                    inflow_m3s: view.primal[state.inflow_lags.start + l * state.hydro_count + h],
                }
            })
        })
        .collect();
    let pumping_stations = extract_pumping_stations(view, spec, stage_id);
    let contracts = extract_contracts(view, spec, stage_id);
    (inflow_lags, pumping_stations, contracts)
}

/// Add one stage's cost breakdown into a running per-category accumulator.
///
/// The five categories follow the breakdown in `ScenarioCategoryCosts`:
///
/// | Field              | Sum expression                                        |
/// |--------------------|-------------------------------------------------------|
/// | `resource_cost`    | `thermal_cost + anticipated_thermal_cost + contract_cost` |
/// | `recourse_cost`    | `deficit_cost + excess_cost`                          |
/// | `violation_cost`   | `storage_violation_cost + filling_target_cost`        |
/// |                    | `+ hydro_violation_cost + inflow_penalty_cost`        |
/// |                    | `+ generic_violation_cost`                            |
/// | `regularization_cost` | `spillage_cost + turbined_cost`               |
/// |                    | `+ curtailment_cost + exchange_cost`                  |
/// | `imputed_cost`     | `pumping_cost`                                        |
///
/// # Examples
///
/// ```
/// use cobre_sddp::simulation::types::{ScenarioCategoryCosts, SimulationCostResult};
/// use cobre_sddp::simulation::extraction::accumulate_category_costs;
///
/// let cost = SimulationCostResult {
///     stage_id: 0,
///     block_id: None,
///     total_cost: 1000.0,
///     immediate_cost: 800.0,
///     future_cost: 200.0,
///     discount_factor: 1.0,
///     thermal_cost: 400.0,
///     anticipated_thermal_cost: 0.0,
///     contract_cost: 100.0,
///     deficit_cost: 50.0,
///     excess_cost: 10.0,
///     storage_violation_cost: 20.0,
///     filling_target_cost: 30.0,
///     hydro_violation_cost: 5.0,
///     outflow_violation_below_cost: 0.0,
///     outflow_violation_above_cost: 0.0,
///     turbined_violation_cost: 0.0,
///     generation_violation_cost: 0.0,
///     evaporation_violation_cost: 0.0,
///     withdrawal_violation_cost: 0.0,
///     inflow_penalty_cost: 3.0,
///     generic_violation_cost: 2.0,
///     spillage_cost: 1.0,
///     turbined_cost: 4.0,
///     curtailment_cost: 7.0,
///     exchange_cost: 8.0,
///     pumping_cost: 60.0,
/// };
///
/// let mut accum = ScenarioCategoryCosts {
///     resource_cost: 0.0,
///     recourse_cost: 0.0,
///     violation_cost: 0.0,
///     regularization_cost: 0.0,
///     imputed_cost: 0.0,
/// };
///
/// accumulate_category_costs(&cost, &mut accum);
/// assert_eq!(accum.resource_cost, 500.0);       // 400 + 0 + 100
/// assert_eq!(accum.recourse_cost, 60.0);         // 50 + 10
/// assert_eq!(accum.violation_cost, 60.0);        // 20 + 30 + 5 + 3 + 2
/// assert_eq!(accum.regularization_cost, 20.0);   // 1 + 4 + 7 + 8
/// assert_eq!(accum.imputed_cost, 60.0);          // 60
/// ```
pub fn accumulate_category_costs(cost: &SimulationCostResult, accum: &mut ScenarioCategoryCosts) {
    // Anticipated thermal fuel rolls up as a resource cost; this is what keeps
    // Σ(macro categories) == immediate_cost.
    accum.resource_cost += cost.thermal_cost + cost.anticipated_thermal_cost + cost.contract_cost;
    accum.recourse_cost += cost.deficit_cost + cost.excess_cost;
    accum.violation_cost += cost.storage_violation_cost
        + cost.filling_target_cost
        + cost.hydro_violation_cost
        + cost.inflow_penalty_cost
        + cost.generic_violation_cost;
    accum.regularization_cost +=
        cost.spillage_cost + cost.turbined_cost + cost.curtailment_cost + cost.exchange_cost;
    accum.imputed_cost += cost.pumping_cost;
}

#[cfg(test)]
mod tests;
