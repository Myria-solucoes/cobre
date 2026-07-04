use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use cobre_core::{BlockMode, ContractType, EntityId, Hydro, ResolvedBounds, Stage, System};
use cobre_solver::StageTemplate;
use cobre_stochastic::normal::precompute::PrecomputedNormal;
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::error::SddpError;
use crate::hydro_models::{EvaporationModelSet, ProductionModelSet, ResolvedProductionModel};
use crate::inflow_method::InflowNonNegativityMethod;
use crate::resolved_parameters::ResolvedParameters;
use crate::setup::template_postprocess::{
    compute_cumulative_discount_factors, compute_per_stage_discount_factors,
};

use super::layout::{ResolvedTables, StageLayout, TemplateBuildCtx};
use super::{
    COST_SCALE_FACTOR, GenericConstraintRowEntry, M3S_TO_HM3, columns, entries, rows, scaling,
};

/// Outcome of [`build_stage_templates`]: one [`StageTemplate`] per study stage
/// plus the per-stage offsets and counts the forward/backward/simulation passes
/// need. The per-stage `Vec`s are parallel — index `s` of each refers to stage `s`.
#[derive(Debug, Clone)]
pub struct StageTemplates {
    /// One structural LP template per study stage, in stage order.
    pub templates: Vec<StageTemplate>,
    /// Row index of the first water-balance constraint in each stage's LP.
    ///
    /// Length equals `templates.len()`.  Used by `PatchBuffer::fill_forward_patches`
    /// to locate the noise-injection rows.
    pub base_rows: Vec<usize>,
    /// Pre-computed noise scale `ζ_stage * σ_{stage,hydro}` for each (stage, hydro) pair.
    ///
    /// Flat array in stage-major layout: `noise_scale[stage * n_hydros + hydro]`.
    /// Length equals `n_study_stages * n_hydros`.
    ///
    /// Used by the forward pass to transform raw standard-normal noise `η` into
    /// the full noise term `ζ*σ*η` before patching the water-balance RHS.
    /// The complete patch value is `ζ*base + ζ*σ*η`, where `ζ*base` is encoded
    /// in the template's `row_lower`/`row_upper` and `ζ*σ*η` is computed by the
    /// caller at each stage using this pre-computed scale.
    pub noise_scale: Vec<f64>,
    /// Per-stage time-conversion factor `ζ = total_hours * M3S_TO_HM3`.
    ///
    /// Length equals `templates.len()`.  Used by the simulation pipeline to
    /// convert the water-balance RHS (in hm³) back to inflow in m³/s for
    /// output reporting: `inflow_m3s = rhs_hm3 / zeta_per_stage[stage]`.
    pub zeta_per_stage: Vec<f64>,
    /// Per-stage block durations in hours.
    ///
    /// `block_hours_per_stage[stage]` is a `Vec<f64>` of length `n_blocks` for
    /// that stage.  Used by the simulation pipeline to convert load-balance
    /// constraint duals from $/MW to $/`MWh`: `spot_price = dual / block_hours`.
    pub block_hours_per_stage: Vec<Vec<f64>>,
    /// Number of hydro plants (N) used to stride into `noise_scale`.
    pub n_hydros: usize,
    /// Per-stage row index of the first load-balance constraint.
    ///
    /// `load_balance_row_starts[s]` is `StageLayout::row_load_balance_start()`
    /// for stage `s` — NOT a hand-derived `row_water_balance_start + n_hydros`
    /// offset, which only held before the chronological per-block water rows
    /// and the travel-time bucket-definition rows existed between the two.
    /// Length equals `templates.len()`.  Used by the forward, backward, and
    /// simulation passes to locate load-balance rows for stochastic load
    /// patching.
    pub load_balance_row_starts: Vec<usize>,
    /// Number of buses with stochastic load noise (i.e. with `std_mw > 0`).
    ///
    /// Equals `normal_lp.n_entities()`.  Tells the forward and backward passes
    /// how many load-noise components to extract from the opening tree noise
    /// vector, which carries load noise in indices `[n_hydros, n_hydros + n_load_buses)`.
    pub n_load_buses: usize,
    /// Position in the `buses` slice for each stochastic load bus.
    ///
    /// Length equals `n_load_buses`.  Bus IDs are sorted by [`cobre_core::EntityId`] for
    /// declaration-order invariance.  The forward and backward passes use
    /// `load_bus_indices[i]` to compute the base row index of bus `i` in the
    /// load-balance region: `row = load_balance_row_start + load_bus_indices[i] * n_blks + blk`.
    pub load_bus_indices: Vec<usize>,
    /// Per-stage metadata for active generic constraint rows.
    ///
    /// `generic_constraint_row_entries[s]` contains one
    /// [`GenericConstraintRowEntry`] per active `(constraint, block)` pair at
    /// stage `s`.  Used by the simulation extraction pipeline to map LP
    /// row/column indices back to constraint identity and block.  Empty for
    /// stages with no active generic constraints.
    pub generic_constraint_row_entries: Vec<Vec<GenericConstraintRowEntry>>,
    /// Per-stage NCS column start indices.
    ///
    /// `ncs_col_starts[stage_idx]` is the column index of the first NCS generation
    /// variable for that stage. The base shifts per stage with `n_blks`, so it is
    /// legitimate per-stage geometry; the COUNT it strides is the scalar
    /// [`Self::n_ncs`].
    pub ncs_col_starts: Vec<usize>,
    /// NCS column count — the full system NCS count, identical at every stage.
    ///
    /// Under the dense layout every NCS keeps a column at every stage, so the count
    /// is a single scalar, not a per-stage Vec; a commissioning-dormant NCS keeps
    /// its column (pinned to `[0, 0]`).
    pub n_ncs: usize,
    /// Per-stage pumping-flow column start indices.
    ///
    /// `pumping_col_starts[stage_idx]` is the column index of the first
    /// pumping-flow variable for that stage, sourced from
    /// `StageLayout::col_pumping_start`, the sole owner of the pumping-flow column
    /// base. Pumping columns are block-major over ALL system stations (dense):
    /// `pumping_col_starts[stage_idx] + p_sys * n_blks + blk`, where `p_sys` is the
    /// SYSTEM station index. The base shifts per stage with `n_blks`, so it is
    /// legitimate per-stage geometry; the COUNT it strides is the scalar
    /// [`Self::n_pumping`]. A commissioning-dormant station keeps its column
    /// (pinned to `[0, 0]`).
    pub pumping_col_starts: Vec<usize>,
    /// Pumping-station column count — the full system station count, identical at
    /// every stage (dense). A commissioning-dormant station keeps its column.
    pub n_pumping: usize,
    /// Per-stage equipment geometry for simulation extraction.
    ///
    /// `geometry_per_stage[stage_idx]` holds the stage-correct column and row
    /// ranges for every block-major equipment family at that stage, sourced from
    /// the per-stage `StageLayout`. A single global stage-0 geometry would carry
    /// `n_blks`-striped bases/lengths that misread any stage with a differing
    /// block count. Length equals `templates.len()`. Threaded into
    /// `StageExtractionSpec` so the simulation read-path addresses the columns the
    /// solved primal occupies at the stage being extracted.
    pub geometry_per_stage: Vec<StageGeometry>,
    /// Mapping from target hydro ID to source hydro indices that divert to it.
    ///
    /// Used by the simulation extraction pipeline to compute `diverted_inflow_m3s`.
    /// Empty when no hydros have diversion.
    pub diversion_upstream: HashMap<EntityId, Vec<usize>>,
    /// Per-stage hydro productivities (MW per m³/s) for simulation extraction.
    ///
    /// `hydro_productivities_per_stage[stage][h]` is the productivity of hydro `h`
    /// at stage `stage`, accounting for per-stage overrides.  FPHA hydros have 0.0.
    pub hydro_productivities_per_stage: Vec<Vec<f64>>,
    /// Per-stage one-step discount factor for the transition departing stage `t`:
    /// `discount_factors[t] = 1 / (1 + r_t)^(Dt / 365.25)` (`r_t` the annual rate,
    /// `Dt` the stage duration in days); all `1.0` when the rate is `0.0` with no
    /// overrides. Applied to the theta objective coefficient.
    ///
    /// Private with a getter/setter: [`build_stage_templates`] leaves a
    /// `1.0`-placeholder until [`StageTemplates::set_discount_factors`]. A `pub`
    /// field would let a caller read the placeholder as the discounted value —
    /// silently yielding undiscounted NPV. Read via
    /// [`StageTemplates::discount_factors`].
    discount_factors: Vec<f64>,
    /// Cumulative discount factor for reporting: `cumulative[0] = 1.0`,
    /// `cumulative[t] = cumulative[t-1] * discount_factors[t-1]`. The present value
    /// of stage `t`'s immediate cost is `cumulative[t] * immediate_cost_t`.
    ///
    /// Private for the same reason as [`StageTemplates::discount_factors`] (shares
    /// the placeholder window). Read via
    /// [`StageTemplates::cumulative_discount_factors`].
    cumulative_discount_factors: Vec<f64>,
}

impl StageTemplates {
    /// All-empty [`StageTemplates`] for a study with zero stages. Only `n_hydros`
    /// (the stride into `noise_scale`) carries through; it is a system-level count
    /// well-defined even with no stages.
    #[must_use]
    pub(crate) fn empty(n_hydros: usize) -> Self {
        Self {
            templates: Vec::new(),
            base_rows: Vec::new(),
            noise_scale: Vec::new(),
            zeta_per_stage: Vec::new(),
            block_hours_per_stage: Vec::new(),
            n_hydros,
            load_balance_row_starts: Vec::new(),
            n_load_buses: 0,
            load_bus_indices: Vec::new(),
            generic_constraint_row_entries: Vec::new(),
            ncs_col_starts: Vec::new(),
            n_ncs: 0,
            pumping_col_starts: Vec::new(),
            n_pumping: 0,
            geometry_per_stage: Vec::new(),
            diversion_upstream: HashMap::new(),
            hydro_productivities_per_stage: Vec::new(),
            discount_factors: Vec::new(),
            cumulative_discount_factors: Vec::new(),
        }
    }

    /// Per-stage one-step discount factors (read access). A `1.0`-placeholder until
    /// [`StageTemplates::set_discount_factors`] runs.
    #[must_use]
    pub(crate) fn discount_factors(&self) -> &[f64] {
        &self.discount_factors
    }

    /// Cumulative discount factors for reporting (read access). A `1.0`-placeholder
    /// until [`StageTemplates::set_discount_factors`] runs.
    #[must_use]
    pub(crate) fn cumulative_discount_factors(&self) -> &[f64] {
        &self.cumulative_discount_factors
    }

    /// Install the per-stage discount factors and recompute the cumulative factors
    /// from them in one call, so the two slices cannot drift out of step (a caller
    /// cannot set per-stage factors while leaving a stale cumulative vector behind).
    pub(crate) fn set_discount_factors(&mut self, per_stage: Vec<f64>) {
        self.cumulative_discount_factors = compute_cumulative_discount_factors(&per_stage);
        self.discount_factors = per_stage;
    }
}

/// Per-stage equipment geometry for simulation extraction: the stage-correct
/// column/row `Range`s, identity lists, and block count for every block-major
/// family, each computed from **this** stage's `StageLayout`.
///
/// A single global stage-0 geometry is the bug this struct forbids: every family
/// after `turbine` has a base `turbine.start + Σ(prior)·n_blks` and length
/// `count·n_blks`, both striped by stage 0's block count, so at any stage with a
/// differing block count the stage-0 base/length addresses the WRONG primal
/// columns. The per-stage `n_blks` stride was already correct; this closes the
/// matching base/length gap. Uniform-block studies coincide with stage 0.
///
/// [`Default`] is the all-`0..0` geometry — every extraction read it gates returns
/// zero — the safe fallback when no per-stage geometry is available, matching the
/// sibling `ncs_col_starts` / `pumping_col_starts` empty-slice fallbacks.
#[derive(Debug, Clone, Default)]
pub struct StageGeometry {
    /// Future-cost epigraph (θ) column index — the authoritative value from
    /// `StageLayout::col_theta`, which accounts for the `anticipated_state_out`
    /// shift when `n_anticipated > 0`. Single source of truth for code that must
    /// address θ outside the builder (e.g. discount-factor postprocessing); do not
    /// re-derive the index from `n_state`/`n_hydros` by hand.
    pub theta_col: usize,
    /// Turbined-flow column range (one per hydro per block). `turbine.start` is
    /// `theta + 1` and stage-invariant, but `turbine.end` is `n_blks`-dependent,
    /// so the cost-breakdown `range_sum` still needs the per-stage range.
    pub turbine: Range<usize>,
    /// Spillage column range (one per hydro per block).
    pub spillage: Range<usize>,
    /// Diversion-flow column range (one per hydro per block).
    pub diversion: Range<usize>,
    /// Thermal-generation column range (one per thermal per block).
    pub thermal: Range<usize>,
    /// Anticipated-decision column range (one per anticipated thermal,
    /// stage-level). Starts at `thermal.end`, which is `n_blks`-dependent, so the
    /// cost-breakdown `range_sum` needs the per-stage base.
    pub anticipated_decision: Range<usize>,
    /// Forward line-flow column range (one per line per block).
    pub line_fwd: Range<usize>,
    /// Reverse line-flow column range (one per line per block).
    pub line_rev: Range<usize>,
    /// Bus-deficit column range (`B · S · K` columns).
    pub deficit: Range<usize>,
    /// Bus-excess column range (one per bus per block).
    pub excess: Range<usize>,
    /// FPHA-generation column range (one per FPHA hydro per block).
    pub generation: Range<usize>,
    /// Per-`(evaporation hydro, block)` column/row indices, block-major
    /// (`local_evap_idx * n_blks + blk`). Anchored at the `n_blks`-dependent
    /// FPHA-generation-block end, so they shift under a non-uniform schedule —
    /// this per-stage copy carries the stage-correct columns. A reader wanting a
    /// hydro's block 0 indexes `local_evap_idx * n_blks`.
    pub evap_indices: Vec<crate::indexer::EvaporationIndices>,
    /// Inflow non-negativity slack column range (one per hydro, stage-level).
    pub inflow_slack: Range<usize>,
    /// Under-withdrawal slack column range (one per hydro, stage-level).
    pub withdrawal_slack_neg: Range<usize>,
    /// Over-withdrawal slack column range (one per hydro, stage-level).
    pub withdrawal_slack_pos: Range<usize>,
    /// Outflow-below-minimum slack column range (one per hydro per block).
    pub outflow_below_slack: Range<usize>,
    /// Outflow-above-maximum slack column range (one per hydro per block).
    pub outflow_above_slack: Range<usize>,
    /// Turbine-below-minimum slack column range (one per hydro per block).
    pub turbine_below_slack: Range<usize>,
    /// Generation-below-minimum slack column range (one per hydro per block).
    pub generation_below_slack: Range<usize>,
    /// Import-contract column range (one per import contract per block); empty
    /// `start..start` (not `0..0`) at the pumping-end column when there are none.
    pub contract_import: Range<usize>,
    /// Export-contract column range (one per export contract per block); empty
    /// `start..start` at the import-end column when there are none.
    pub contract_export: Range<usize>,

    // ── Per-stage row ranges, identity lists, and block count ────────────────
    /// Water-balance row range (one row per hydro). Count is stage-invariant
    /// (`n_hydros`) but the base rides the per-stage block-major rows before it.
    pub water_balance: Range<usize>,
    /// Load-balance row range (one row per bus per block; `n_buses · n_blks`).
    pub load_balance: Range<usize>,
    /// Per-stage `σ_fill`-target row range (one row per Filling-phase hydro); empty
    /// `start..start` (not `0..0`) at every non-Filling stage.
    // Voice 4: no read site consumes this yet — it is the per-stage carrier that
    // keeps `StageGeometry` a faithful mirror of the row shape, the seam the sibling
    // `σ^{v-}` family extends. The `#[allow(dead_code)]` refires if the seam is
    // removed before a reader lands.
    #[allow(dead_code)]
    pub filling_target: Range<usize>,
    /// Per-stage `σ_fill`-target slack column range (one column per Filling-phase
    /// hydro); empty `start..start` at every non-Filling stage. Simulation
    /// extraction reads the `σ_fill` primal at `start + local_idx`, resolving
    /// `local_idx` via `filling_target_hydro_indices`.
    pub filling_target_col: Range<usize>,
    /// Soft `σ^{v-}` operating-floor row range (one row per Operating-phase filling
    /// hydro); empty `start..start` (not `0..0`) at every non-operating stage.
    // Voice 4: no read site consumes this yet — the per-stage row-shape carrier
    // mirroring the `filling_target` seam. The `#[allow(dead_code)]` refires if the
    // seam is removed before a reader lands.
    #[allow(dead_code)]
    pub filled_min_storage_floor: Range<usize>,
    /// Soft `σ^{v-}` operating-floor slack column range (one column per
    /// Operating-phase filling hydro); empty `start..start` at every non-operating
    /// stage. Simulation extraction reads the `σ^{v-}` primal at `start + local_idx`,
    /// resolving `local_idx` via `filled_min_storage_floor_hydro_indices`.
    pub filled_min_storage_floor_col: Range<usize>,
    /// Row index of the first z-inflow definition constraint. Always `0`: state
    /// pinning uses column bounds, so no state-fixing rows precede the z-inflow
    /// block. Carried per stage to mirror `StageLayout::z_inflow_row_start`.
    pub z_inflow_row_start: usize,
    /// Number of operating blocks (K) at this stage — the block-major stride for
    /// every equipment family.
    pub n_blks: usize,
    /// Control-region anchor for the interior storage boundaries `S¹ … Sᴷ⁻¹`,
    /// mirroring `StageLayout::storage_internal_start` (holds
    /// `StateLayout::control_region_start()`, not `0`); the within-family address is
    /// `storage_internal_start + h * (n_blks − 1) + (k − 1)` (stride `n_blks − 1`).
    /// Dead in parallel mode and when `n_blks ≤ 1`: the interior family is empty, so no
    /// `k` reaches the `_` arm of [`StageGeometry::block_storage_col`], which this feeds.
    pub storage_internal_start: usize,
    /// Block formulation mode at this stage. Selects per-block storage extraction
    /// (`Chronological` reads each block's own `(Sᵇ, Sᵇ⁺¹)` boundary) versus the
    /// stage-level `(S⁰, Sᴷ)` pair (`Parallel`); defaults to `Parallel`.
    pub block_mode: BlockMode,
    /// System hydro indices using FPHA at this stage, in slot order. FPHA
    /// membership is per `(hydro, stage)`, so this is the stage-correct list.
    pub fpha_hydro_indices: Vec<usize>,
    /// System hydro indices with linearized evaporation at this stage, in slot
    /// order. Parallel to `evap_indices`.
    pub evap_hydro_indices: Vec<usize>,
    /// System hydro indices owning a `σ_fill`-target slack column at this stage (the
    /// Filling-phase hydros), in slot order. Parallel to `filling_target_col` (slot
    /// `i` → `filling_target_col.start + i`). The family is SPARSE — one column per
    /// filling hydro — so extraction resolves a system hydro's column via this
    /// system→slot list, never by the dense system index `h`.
    pub filling_target_hydro_indices: Vec<usize>,
    /// System hydro indices owning a `σ^{v-}` operating-floor slack column at this
    /// stage (the Operating-phase filling hydros), in slot order. Parallel to
    /// `filled_min_storage_floor_col`; SPARSE like `filling_target_hydro_indices`,
    /// resolved the same way.
    pub filled_min_storage_floor_hydro_indices: Vec<usize>,
}

impl StageGeometry {
    /// Build the per-stage equipment geometry from this stage's `StageLayout`.
    ///
    /// This is the production source: every range is the stage-correct geometry
    /// the LP template was frozen with, so the simulation read-path addresses the
    /// columns the solved primal actually occupies at this stage. The empty-block
    /// `start` accessors (`col_generation_start`, the `col_*_slack` accessors)
    /// resolve the dedicated empty-block cursor rather than a bare `0` when the
    /// family collapses to `0..0`, matching the indexer convention.
    fn from_layout(layout: &StageLayout<'_>, block_mode: BlockMode) -> Self {
        // Most ranges are cloned from `StageLayout` own fields (already `0..0` when
        // empty). The `filling_*` families are built inline as
        // `start..start + indices.len()`, so an empty family is `start..start` (not
        // `0..0`) — both are `is_empty()`, and the read-path never dereferences an
        // empty range.
        let anticipated_decision = if layout.n_anticipated > 0 {
            let s = layout.anticipated.col_anticipated_decision_start;
            s..s + layout.n_anticipated
        } else {
            0..0
        };
        Self {
            theta_col: layout.col_theta(),
            turbine: layout.turbine.clone(),
            spillage: layout.spillage.clone(),
            diversion: layout.diversion.clone(),
            thermal: layout.thermal.clone(),
            anticipated_decision,
            line_fwd: layout.line_fwd.clone(),
            line_rev: layout.line_rev.clone(),
            deficit: layout.deficit.clone(),
            excess: layout.excess.clone(),
            generation: layout.generation.clone(),
            evap_indices: layout.evap_indices.clone(),
            inflow_slack: layout.inflow_slack.clone(),
            withdrawal_slack_neg: layout.withdrawal_slack_neg.clone(),
            withdrawal_slack_pos: layout.withdrawal_slack_pos.clone(),
            outflow_below_slack: layout.outflow_below_slack.clone(),
            outflow_above_slack: layout.outflow_above_slack.clone(),
            turbine_below_slack: layout.turbine_below_slack.clone(),
            generation_below_slack: layout.generation_below_slack.clone(),
            contract_import: layout.col_contract_import_start
                ..layout.col_contract_import_start + layout.n_contract_import * layout.n_blks,
            contract_export: layout.col_contract_export_start
                ..layout.col_contract_export_start + layout.n_contract_export * layout.n_blks,
            water_balance: layout.water_balance.clone(),
            load_balance: layout.load_balance.clone(),
            filling_target: layout.row_filling_target_start
                ..layout.row_filling_target_start + layout.filling_target_hydro_indices.len(),
            filling_target_col: layout.col_filling_target_start
                ..layout.col_filling_target_start + layout.filling_target_hydro_indices.len(),
            filled_min_storage_floor: layout.row_filled_min_storage_floor_start
                ..layout.row_filled_min_storage_floor_start
                    + layout.filled_min_storage_floor_hydro_indices.len(),
            filled_min_storage_floor_col: layout.col_filled_min_storage_floor_start
                ..layout.col_filled_min_storage_floor_start
                    + layout.filled_min_storage_floor_hydro_indices.len(),
            z_inflow_row_start: layout.z_inflow_row_start,
            n_blks: layout.n_blks,
            storage_internal_start: layout.storage_internal_start,
            block_mode,
            fpha_hydro_indices: layout.fpha_hydro_indices.clone(),
            evap_hydro_indices: layout.evap_hydro_indices.clone(),
            filling_target_hydro_indices: layout.filling_target_hydro_indices.clone(),
            filled_min_storage_floor_hydro_indices: layout
                .filled_min_storage_floor_hydro_indices
                .clone(),
        }
    }

    /// Storage column at chronological boundary `k ∈ 0..=K` (`K = self.n_blks`) for
    /// hydro `h`, mirroring `StageLayout::block_storage_col`
    /// so the simulation read-path resolves per-block boundaries without a
    /// `StageLayout`. `k = 0 → S⁰` (incoming state, base `storage_in_start`);
    /// `k = K → Sᴷ` (outgoing state, bare `h` because `state.storage.start == 0`);
    /// `k ∈ 1..K → storage_internal_start + h * (n_blks − 1) + (k − 1)` (interior
    /// CONTROL columns, stride `n_blks − 1`). The `k == self.n_blks` arm MUST precede
    /// the interior `_` arm, else `_` captures the outgoing endpoint and addresses an
    /// interior column past the family. The incoming-state base is passed in because
    /// the state region is owned by [`StateLayout`](crate::indexer::StateLayout), not
    /// `StageGeometry`.
    #[inline]
    #[must_use]
    pub fn block_storage_col(&self, h: usize, k: usize, storage_in_start: usize) -> usize {
        match k {
            0 => storage_in_start + h,
            k if k == self.n_blks => h,
            _ => self.storage_internal_start + h * (self.n_blks - 1) + (k - 1),
        }
    }
}

/// Per-stage outputs of [`build_single_stage_template`], transposed by
/// [`assemble_stage_templates_output`] into the parallel per-stage `Vec`s of
/// [`StageTemplates`]. Adding a per-stage datum is one field here plus one
/// transpose line in the assembler.
pub(super) struct StageBuildOutput {
    /// Structural LP template for the stage.
    pub template: StageTemplate,
    /// Row index of the first water-balance constraint (the `PatchBuffer`
    /// noise-injection `base_row`).
    pub stage_base_row: usize,
    /// Row index of the first load-balance constraint (load-noise patches).
    pub load_balance_row_start: usize,
    /// Active generic-constraint row metadata for the stage.
    pub gc_entries: Vec<GenericConstraintRowEntry>,
    /// Column index of the first NCS generation variable.
    pub ncs_col_start: usize,
    /// Number of NCS entities at the stage — the full system count (dense).
    pub ncs_count: usize,
    /// Column index of the first pumping-flow variable (sourced from
    /// `StageLayout::col_pumping_start`).
    pub pumping_col_start: usize,
    /// Number of pumping stations ACTIVE (contributing columns) at the stage
    /// (the commissioning-gated count, sourced from [`StageLayout::n_pumping`]).
    pub n_pumping: usize,
    /// Stage-correct equipment column ranges for simulation extraction, computed
    /// from this stage's [`StageLayout`].
    pub equipment_geometry: StageGeometry,
}

/// Construct the [`StageBuildOutput`] for a single study stage.
// Rationale: `clippy::similar_names` flags the `state` handle next to `stage`/`stage_idx`;
// both names are established (the `StageLayout`/`StageData` field is `state`, the per-stage
// inputs are `stage`/`stage_idx`), so renaming either to satisfy the heuristic would obscure
// intent rather than clarify it.
#[allow(clippy::similar_names)]
pub(super) fn build_single_stage_template(
    ctx: &TemplateBuildCtx<'_>,
    state: &crate::indexer::StateLayout,
    stage: &Stage,
    stage_idx: usize,
) -> StageBuildOutput {
    let layout = StageLayout::new(ctx, state, stage, stage_idx);
    let stage_base_row = layout.row_water_balance_start();
    let load_balance_row_start = layout.row_load_balance_start();

    let (col_lower, mut col_upper, mut objective) =
        columns::fill_stage_columns(ctx, stage, stage_idx, &layout);
    let (mut row_lower, mut row_upper) = rows::fill_stage_rows(ctx, stage, stage_idx, &layout);
    let mut col_entries = entries::build_stage_matrix_entries(ctx, stage, stage_idx, &layout);

    {
        let mut buffers = entries::LpMatrixBuffers {
            col_entries: &mut col_entries,
            col_upper: &mut col_upper,
            objective: &mut objective,
            row_lower: &mut row_lower,
            row_upper: &mut row_upper,
        };
        entries::fill_generic_constraint_entries(ctx, stage, stage_idx, &layout, &mut buffers);
    }

    // Scale every monetary objective coefficient by 1/K for numerical
    // conditioning; outputs are unscaled at the reporting boundary.
    //
    // Theta must NOT be divided: the Benders cuts already enforce
    // `theta >= Q_successor / K`, so theta holds the SCALED future cost. Dividing
    // it too would make the LP `stage_cost/K + (1/K)*theta`, which recovers
    // `stage_cost + future_cost/K` at the boundary — wrong. `layout.col_theta()`
    // reads the correct index even when `n_anticipated > 0` shifts theta.
    let theta_col = layout.col_theta();
    for (i, coeff) in objective.iter_mut().enumerate() {
        if i != theta_col {
            *coeff /= COST_SCALE_FACTOR;
        }
    }

    // Sort each column's entries by row index (CSC invariant).
    for col_entry_vec in &mut col_entries {
        col_entry_vec.sort_unstable_by_key(|&(row, _)| row);
    }

    let (col_starts, row_indices, values) = entries::assemble_csc(&col_entries);

    let n_transfer = ctx.n_hydros * ctx.max_par_order;

    let template = StageTemplate {
        num_cols: layout.num_cols,
        num_rows: layout.num_rows,
        num_nz: col_entries.iter().map(Vec::len).sum(),
        col_starts,
        row_indices,
        values,
        col_lower,
        col_upper,
        objective,
        row_lower,
        row_upper,
        n_state: layout.n_state(),
        n_transfer,
        n_dual_relevant: layout.n_dual_relevant,
        n_hydro: layout.n_h,
        max_par_order: layout.lag_order,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    // Snapshot the per-stage equipment geometry BEFORE moving `layout`'s owned
    // `generic_constraint_rows` Vec into the output: `from_layout` only borrows,
    // so it must run while `layout` is intact.
    let equipment_geometry = StageGeometry::from_layout(&layout, stage.block_mode);

    StageBuildOutput {
        template,
        stage_base_row,
        load_balance_row_start,
        gc_entries: layout.generic_constraint_rows,
        ncs_col_start: layout.col_ncs_start,
        ncs_count: layout.n_ncs,
        pumping_col_start: layout.col_pumping_start,
        n_pumping: layout.n_pumping,
        equipment_geometry,
    }
}

/// Collect the bus-slice positions of stochastic load buses.
///
/// Returns bus-position indices (into the buses slice) for every bus that has
/// `std_mw > 0` in any load model, sorted by `EntityId` for declaration-order
/// invariance.  Buses with duplicate IDs across stages are deduplicated.
fn collect_load_bus_indices(system: &System, bus_pos: &BTreeMap<EntityId, usize>) -> Vec<usize> {
    let mut ids: Vec<EntityId> = system
        .load_models()
        .iter()
        .filter(|m| m.std_mw > 0.0)
        .map(|m| m.bus_id)
        .collect();
    ids.sort_unstable_by_key(|id| id.0);
    ids.dedup();
    ids.iter()
        .filter_map(|id| bus_pos.get(id).copied())
        .collect()
}

/// Build one [`StageTemplate`] per study stage from a fully loaded [`System`].
///
/// The templates encode the complete structural LP for each SDDP subproblem
/// in CSC format, ready for bulk-loading via `SolverInterface::load_model`.
/// They are constructed once at solver initialisation and shared read-only
/// across all solver threads.
///
/// ## Column and row layout
///
/// See the module-level documentation for the full LP layout.
/// Key dimensions for a stage with N hydros, T thermals, Lines lines,
/// B buses, K blocks per stage, and F FPHA hydros each with M planes:
///
/// - `num_cols` and `num_rows` are computed by `layout::StageLayout` —
///   see `layout.rs` for the authoritative column and row counts
/// - `n_state  = N*(1+L)`
/// - `n_transfer = N*L`  (storage + all lags except the oldest)
/// - `n_dual_relevant = N*(1+L)`  (`z_inflow` definition, water balance, load balance, FPHA,
///   evaporation, operational violation, and generic constraint rows are all structural and
///   non-dual-relevant; only the state-fixing rows contribute to cut gradients)
///
/// ## PAR order and `max_par_order`
///
/// `max_par_order` is the maximum of (a) the maximum AR coefficient count
/// across all hydro inflow models and (b) `par_lp.max_order()`.  The latter
/// is non-classical only when an annual component is present, in which case
/// the precompute widens the lag stride to 12 and the LP must allocate
/// matching column and row slots.  All hydros use the same uniform lag stride
/// `max_par_order` to enable SIMD-friendly contiguous access.
///
/// ## Objective coefficients
///
/// Costs are expressed in `$/MWh` (thermal, deficit, excess, lines) multiplied
/// by the block duration in hours so they integrate to $/block.  Storage, lag,
/// incoming-storage, theta, turbine, and spillage columns carry zero or small
/// regularization costs drawn from the resolved penalty tables.
///
/// When the penalty method is active, each inflow slack column `sigma_inf_h`
/// carries objective coefficient `penalty_cost * total_stage_hours`.
///
/// FPHA generation columns carry objective coefficient 0.0 by default.
///
/// ## Inflow non-negativity
///
/// When `inflow_method.has_slack_columns()` is `true` (i.e., the `Penalty`
/// variant), `N` slack columns `sigma_inf_h >= 0`
/// are appended at the end of the column layout.  Each slack enters the water
/// balance row for hydro `h` with coefficient `+tau_total * M3S_TO_HM3`,
/// acting as virtual inflow that prevents infeasibility when the PAR(p) noise
/// is sufficiently negative.
///
/// ## FPHA hydros
///
/// For hydros whose resolved production model at a given stage is
/// [`ResolvedProductionModel::Fpha`], generation becomes a free variable
/// `g_{h,k} ∈ [0, max_generation_mw]` bounded by M hyperplane constraints:
///
/// ```text
/// g_{h,k} - gamma_v/2*v - gamma_v/2*v_in - gamma_q*q_{h,k} - gamma_s*s_{h,k} <= gamma_0
/// ```
///
/// The `v_in` contribution propagates through the LP via the matrix coefficient
/// `-gamma_v/2` on the incoming-storage column; when `v_in` is fixed by the
/// storage-fixing equality row its value automatically enters the FPHA constraint
/// right-hand side.  No changes to the backward pass or cut extraction are needed.
///
/// Returns `Ok` with empty templates for a system with zero stages.  All
/// entity counts may be zero (valid for degenerate test systems).
///
/// # Errors
///
/// Returns [`SddpError`] if the PAR precomputation data is inconsistent with
/// the system (e.g., a hydro in `par_lp` is not present in `system`), or if
/// the production model set has incompatible dimensions.
///
/// ## Evaporation hydros
///
/// For hydros whose evaporation model is
/// `EvaporationModel::Linearized`,
/// three stage-level columns are added per hydro (evaporation outflow,
/// `f_evap_plus`, `f_evap_minus`).  The evaporation-outflow column is bounded
/// symmetrically `[-q_max, +q_max]` so a negative value can absorb net rainfall
/// input on the lake surface; `f_evap_plus` and `f_evap_minus` are bounded
/// `[0, +inf)`.  The evaporation-outflow column carries objective coefficient
/// 0.0; the violation slacks carry the evaporation penalty.  One equality
/// constraint row is added per evaporation hydro with
/// `row_lower == row_upper == intercept_m3s`.
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
/// use cobre_sddp::InflowNonNegativityMethod;
/// use cobre_sddp::hydro_models::PrepareHydroModelsResult;
/// use cobre_sddp::lp_builder::build_stage_templates;
/// use cobre_sddp::resolved_parameters::ResolvedParameters;
/// use cobre_stochastic::par::precompute::PrecomputedPar;
///
/// let bus = Bus {
///     id: EntityId(1),
///     name: "B1".to_string(),
///     operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"),
///     deficit_segments: vec![DeficitSegment { depth_mw: None, cost_per_mwh: 1000.0 }],
///     excess_cost: 0.0,
/// };
/// let system = SystemBuilder::new().buses(vec![bus]).build().expect("valid");
/// let method = InflowNonNegativityMethod::None;
/// let par_lp = PrecomputedPar::build(&[], &[], &[], None).expect("empty ok");
/// let normal_lp = cobre_stochastic::normal::precompute::PrecomputedNormal::default();
/// let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
/// let resolved_parameters = ResolvedParameters::default();
/// // No stages → empty result.
/// let result = build_stage_templates(&system, method, &par_lp, &normal_lp,
///                                    &hydro_models.production, &hydro_models.evaporation,
///                                    &resolved_parameters)
///     .expect("empty system ok");
/// assert!(result.templates.is_empty());
/// ```
pub fn build_stage_templates(
    system: &System,
    inflow_method: InflowNonNegativityMethod,
    par_lp: &PrecomputedPar,
    normal_lp: &PrecomputedNormal,
    production_models: &ProductionModelSet,
    evaporation_models: &EvaporationModelSet,
    resolved_parameters: &ResolvedParameters,
) -> Result<StageTemplates, SddpError> {
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let n_hydros = system.hydros().len();

    if study_stages.is_empty() {
        return Ok(StageTemplates::empty(n_hydros));
    }

    let (ctx, load_bus_indices, diversion_upstream_output) = build_template_build_ctx(
        system,
        inflow_method,
        par_lp,
        production_models,
        evaporation_models,
        resolved_parameters,
    );
    let n_load_buses = load_bus_indices.len();
    debug_assert!(
        normal_lp.n_entities() == 0 || normal_lp.n_entities() == n_load_buses,
        "PrecomputedNormal has {} entities but system has {} stochastic load buses",
        normal_lp.n_entities(),
        n_load_buses
    );

    // One canonical role-(a) `StateLayout` shared by every `StageLayout`: the
    // column ranges and `state_to_lp_column_map` it reads are pure functions of the
    // state dimensions, so they match the `StateLayout` setup stores on
    // `StageData.state` regardless of the mask.
    //
    // `effective_lag_counts` feeds only the `nonzero_state_indices` mask (read off
    // `StageData.state` by the cut path), never the template build. Sized to
    // `ctx.n_hydros` per the `StateLayout::new` contract, falling back to the dense
    // `max_par_order` stride for hydros the PAR model omits — so a hydro-free
    // `PrecomputedPar` test still satisfies the length contract.
    let effective_lag_counts: Vec<usize> = if ctx.max_par_order > 0 {
        (0..ctx.n_hydros)
            .map(|h| {
                if h < par_lp.n_hydros() {
                    par_lp.effective_lag_count(h)
                } else {
                    ctx.max_par_order
                }
            })
            .collect()
    } else {
        vec![0; ctx.n_hydros]
    };
    // Recomputes the bucket topology (pure function of `system`) instead of
    // threading it in from the caller: keeps this role-(a) `StateLayout` in
    // agreement with the one `setup` stores on `StageData.state` without
    // widening this function's signature — the accepted redundant-but-
    // deterministic cost of a second call.
    let bucket_topology = crate::setup::bucket_topology::build_transit_bucket_topology(system);
    let state_layout = crate::indexer::StateLayout::new(
        ctx.n_hydros,
        ctx.max_par_order,
        bucket_topology.n_buckets,
        bucket_topology.column_order,
        ctx.n_anticipated,
        ctx.k_max,
        ctx.anticipated_lead_stages.clone(),
        &effective_lag_counts,
    );

    let n_study = study_stages.len();
    let mut stage_outputs = Vec::with_capacity(n_study);
    for (stage_idx, stage) in study_stages.iter().enumerate() {
        stage_outputs.push(build_single_stage_template(
            &ctx,
            &state_layout,
            stage,
            stage_idx,
        ));
    }

    Ok(assemble_stage_templates_output(
        stage_outputs,
        load_bus_indices,
        diversion_upstream_output,
        &study_stages,
        &ctx,
        par_lp,
        n_hydros,
        n_load_buses,
        n_study,
    ))
}

/// Precompute the per-stage minimum target-storage trajectory `V_target[t]` for
/// every filling hydro, keyed `(hydro_idx, stage_id) → V_target` \[hm³\].
///
/// Computed ONCE here, where the full per-stage ζ·rate schedule is available (a
/// per-stage row-fill helper sees one stage and cannot reconstruct the fold). With
/// `L = entry_stage_id − 1` the last Filling stage, anchored on the dead volume and
/// folded backward:
///
/// ```text
/// V_target[L] = min_storage_hm3                          (at L's stage_idx)
/// V_target[t] = min( V_target[t+1] − ζ_{t+1}·rate[t+1], min_storage_hm3 )
/// ```
///
/// `ζ_t = total_hours_per_stage[stage_idx]·M3S_TO_HM3`; `rate`/`min_storage` are
/// the RESOLVED per-stage bounds. The clip at `min_storage` enforces that no floor
/// exceeds the dead volume — dropping it would let an over-provisioned schedule
/// demand a floor ABOVE the dead volume (the design §3.1 forbidden alternative).
/// The fold runs on the UNCLIPPED running value, clipping each stored `V_target[t]`
/// independently to mirror the closed form.
///
/// Hydros are iterated in canonical slot order into a `BTreeMap`, so the result is
/// declaration-order-invariant; a non-filling system yields an empty map.
///
/// `pub(super)` so the sibling builder test modules can exercise it against
/// single-stage fixtures; production reaches it via `build_template_build_ctx`.
pub(super) fn build_filling_v_target(
    hydros: &[Hydro],
    bounds: &ResolvedBounds,
    total_hours_per_stage: &[f64],
    stage_id_to_idx: &BTreeMap<i32, usize>,
) -> BTreeMap<(usize, i32), f64> {
    let mut v_target: BTreeMap<(usize, i32), f64> = BTreeMap::new();
    for (h_idx, hydro) in hydros.iter().enumerate() {
        let (Some(filling), Some(entry)) = (hydro.filling.as_ref(), hydro.entry_stage_id) else {
            continue;
        };
        let start = filling.start_stage_id;
        let last = entry - 1; // L: the last Filling stage id.
        // Guard a hypothetical inverted config (`start < entry` is validated
        // upstream) into an empty trajectory rather than a malformed loop.
        if last < start {
            continue;
        }
        let Some(&last_idx) = stage_id_to_idx.get(&last) else {
            continue;
        };
        let min_storage_at_last = bounds.hydro_bounds(h_idx, last_idx).min_storage_hm3;
        v_target.insert((h_idx, last), min_storage_at_last);
        let mut running = min_storage_at_last;
        let mut t = last;
        while t > start {
            if let Some(&t_idx) = stage_id_to_idx.get(&t) {
                let zeta_t = total_hours_per_stage[t_idx] * M3S_TO_HM3;
                let rate_t = bounds.hydro_bounds(h_idx, t_idx).filling_min_rate_m3s;
                running -= zeta_t * rate_t;
            }
            let prev = t - 1;
            if let Some(&prev_idx) = stage_id_to_idx.get(&prev) {
                let min_storage_prev = bounds.hydro_bounds(h_idx, prev_idx).min_storage_hm3;
                v_target.insert((h_idx, prev), running.min(min_storage_prev));
            }
            t = prev;
        }
    }
    v_target
}

/// Build the [`TemplateBuildCtx`] and ancillary data needed by the stage loop.
///
/// Constructs position maps (hydro/thermal/line/bus), the diversion-upstream
/// map, and the `TemplateBuildCtx` that is shared across all per-stage builds.
/// Also returns `load_bus_indices` (the bus-slice positions of stochastic load
/// buses) and `diversion_upstream_output` (the clone of the diversion map
/// preserved for the final `StageTemplates` output field).
///
/// Called once per `build_stage_templates` invocation, after the early-return
/// guard for empty systems.
// Rationale: too_many_lines — one linear pass of per-entity prep blocks (position
// maps, anticipated metadata, contracts, discount factors) feeding a single
// `TemplateBuildCtx` literal; splitting it would scatter the construction the
// literal reads back, without removing any branching.
#[allow(clippy::too_many_lines)]
fn build_template_build_ctx<'a>(
    system: &'a System,
    inflow_method: InflowNonNegativityMethod,
    par_lp: &'a PrecomputedPar,
    production_models: &'a ProductionModelSet,
    evaporation_models: &'a EvaporationModelSet,
    resolved_parameters: &'a ResolvedParameters,
) -> (
    TemplateBuildCtx<'a>,
    Vec<usize>,
    HashMap<EntityId, Vec<usize>>,
) {
    let hydros = system.hydros();
    let buses = system.buses();
    let n_hydros = hydros.len();

    let hydro_pos: BTreeMap<EntityId, usize> =
        hydros.iter().enumerate().map(|(i, h)| (h.id, i)).collect();
    let thermal_pos: BTreeMap<EntityId, usize> = system
        .thermals()
        .iter()
        .enumerate()
        .map(|(i, t)| (t.id, i))
        .collect();
    let line_pos: BTreeMap<EntityId, usize> = system
        .lines()
        .iter()
        .enumerate()
        .map(|(i, l)| (l.id, i))
        .collect();
    let bus_pos: BTreeMap<EntityId, usize> =
        buses.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

    // Iterate the (ID-sorted) station slice in slot order, NOT declaration order,
    // to uphold the declaration-order bit-determinism rule.
    let pumping_stations = system.pumping_stations();
    let pumping_pos: BTreeMap<EntityId, usize> = pumping_stations
        .iter()
        .enumerate()
        .map(|(i, p)| (p.id, i))
        .collect();
    let n_pumping = pumping_stations.len();
    // Fail fast on a station-count divergence rather than silently reserving the
    // wrong number of pumping-flow columns.
    debug_assert_eq!(
        n_pumping,
        system.bounds().n_pumping(),
        "pumping_stations.len() ({}) != bounds.n_pumping() ({}): resolved-bounds \
         station count disagrees with the entity slice",
        n_pumping,
        system.bounds().n_pumping()
    );

    // One id-sorted slice for both directions; the import/export split is derived
    // here as counts (the dense per-direction column strides) by `contract_type`.
    let contracts = system.contracts();
    let contract_pos: BTreeMap<EntityId, usize> = contracts
        .iter()
        .enumerate()
        .map(|(i, c)| (c.id, i))
        .collect();
    let n_contract_import = contracts
        .iter()
        .filter(|c| c.contract_type == ContractType::Import)
        .count();
    let n_contract_export = contracts
        .iter()
        .filter(|c| c.contract_type == ContractType::Export)
        .count();
    // Fail fast on a contract-count divergence rather than silently reserving the
    // wrong number of contract columns.
    debug_assert_eq!(
        contracts.len(),
        system.bounds().n_contracts(),
        "contracts.len() ({}) != bounds.n_contracts() ({}): resolved-bounds \
         contract count disagrees with the entity slice",
        contracts.len(),
        system.bounds().n_contracts()
    );

    let load_bus_indices = collect_load_bus_indices(system, &bus_pos);

    let max_par_order: usize = system
        .inflow_models()
        .iter()
        .filter(|m| m.stage_id >= 0)
        .map(|m| m.ar_coefficients.len())
        .max()
        .unwrap_or(0)
        .max(par_lp.max_order());

    // Per anticipated thermal: global index, lead_stages (K_i), and commissioning
    // window. The window keys the decision gate's operation-window clause on the
    // delivery stage; `(None, None)` means active every delivery stage in horizon.
    let mut anticipated_thermal_indices: Vec<usize> = Vec::new();
    let mut anticipated_lead_stages: Vec<usize> = Vec::new();
    let mut anticipated_windows: Vec<(Option<i32>, Option<i32>)> = Vec::new();
    for (t_idx, thermal) in system.thermals().iter().enumerate() {
        if let Some(cfg) = thermal.anticipated_config.as_ref() {
            anticipated_thermal_indices.push(t_idx);
            // u32 always fits in usize on supported 32-bit and 64-bit targets.
            anticipated_lead_stages.push(cfg.lead_stages as usize);
            anticipated_windows.push((thermal.entry_stage_id, thermal.exit_stage_id));
        }
    }
    let n_anticipated = anticipated_thermal_indices.len();
    let k_max = anticipated_lead_stages.iter().copied().max().unwrap_or(0);

    // Target hydro ID -> source hydro indices that divert to it. Cloned so the map
    // serves both LP construction (ctx) and the simulation extraction output.
    let mut diversion_upstream: HashMap<EntityId, Vec<usize>> = HashMap::new();
    for (h_idx, hydro) in hydros.iter().enumerate() {
        if let Some(ref div) = hydro.diversion {
            diversion_upstream
                .entry(div.downstream_id)
                .or_default()
                .push(h_idx);
        }
    }
    let diversion_upstream_output = diversion_upstream.clone();

    // Computed before the per-stage loop so `fill_anticipated_columns` can read the
    // discount factors and stage hours from the ctx at LP build time (before
    // postprocess runs). Both arrays have length `n_study_stages`.
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let per_stage_discount =
        compute_per_stage_discount_factors(&study_stages, system.policy_graph());
    let cumulative_discount_factors = compute_cumulative_discount_factors(&per_stage_discount);
    let total_hours_per_stage: Vec<f64> = study_stages
        .iter()
        .map(|s| s.blocks.iter().map(|b| b.duration_hours).sum())
        .collect();

    debug_assert_eq!(
        cumulative_discount_factors.len(),
        study_stages.len(),
        "cumulative_discount_factors length must equal n_study_stages"
    );
    debug_assert_eq!(
        total_hours_per_stage.len(),
        study_stages.len(),
        "total_hours_per_stage length must equal n_study_stages"
    );

    // Study-stage ids by study stage index: the decision gate keys its
    // operation-window clause on the DELIVERY stage's `stage.id`, mapping the
    // delivery index `t + K_i` to its id through this slice.
    let study_stage_ids: Vec<i32> = study_stages.iter().map(|s| s.id).collect();

    // Inverse: study `stage.id` → study stage index. The filling-target fold reads
    // per-stage ζ and bounds at the INDEX but expresses the window in stage IDs.
    let stage_id_to_idx: BTreeMap<i32, usize> = study_stage_ids
        .iter()
        .enumerate()
        .map(|(idx, &id)| (id, idx))
        .collect();

    let filling_v_target = build_filling_v_target(
        hydros,
        system.bounds(),
        &total_hours_per_stage,
        &stage_id_to_idx,
    );

    // Resolved once here (SETUP time, never per stage-solve): `resolve_spread`
    // is O(declared arcs * n_stages), not called again per LP fill.
    let arc_spread_k = crate::setup::bucket_topology::build_arc_spread_k(system);
    let arc_spread_chrono = crate::setup::bucket_topology::build_arc_spread_chrono(system);
    // Recomputes the bucket topology (pure function of `system`) rather than
    // threading it in from the caller, mirroring `build_stage_templates`'s own
    // recomputation for `StateLayout` — the accepted redundant-but-deterministic
    // cost of a second call.
    let per_stage_mask =
        crate::setup::bucket_topology::build_transit_bucket_topology(system).per_stage_mask;

    let ctx = TemplateBuildCtx {
        hydros,
        thermals: system.thermals(),
        lines: system.lines(),
        buses,
        load_models: system.load_models(),
        cascade: system.cascade(),
        resolved: ResolvedTables {
            bounds: system.bounds(),
            penalties: system.penalties(),
            resolved_generic_bounds: system.resolved_generic_bounds(),
            resolved_load_factors: system.resolved_load_factors(),
            resolved_exchange_factors: system.resolved_exchange_factors(),
            resolved_ncs_bounds: system.resolved_ncs_bounds(),
            resolved_ncs_factors: system.resolved_ncs_factors(),
            resolved_parameters,
        },
        hydro_pos,
        thermal_pos,
        line_pos,
        bus_pos,
        par_lp,
        production_models,
        evaporation_models,
        generic_constraints: system.generic_constraints(),
        non_controllable_sources: system.non_controllable_sources(),
        pumping_stations,
        pumping_pos,
        n_pumping,
        contracts,
        contract_pos,
        n_contract_import,
        n_contract_export,
        diversion_upstream,
        n_hydros,
        n_thermals: system.thermals().len(),
        n_lines: system.lines().len(),
        n_buses: buses.len(),
        max_par_order,
        n_anticipated,
        k_max,
        anticipated_lead_stages,
        anticipated_thermal_indices,
        anticipated_windows,
        study_stage_ids,
        has_penalty: n_hydros > 0 && inflow_method.has_slack_columns(),
        cumulative_discount_factors,
        total_hours_per_stage,
        filling_v_target,
        arc_spread_k,
        arc_spread_chrono,
        per_stage_mask,
    };

    (ctx, load_bus_indices, diversion_upstream_output)
}

/// Transpose the per-stage `Vec<StageBuildOutput>` into the parallel per-stage
/// `Vec`s of [`StageTemplates`], computing the noise-scale, zeta, block-hour,
/// hydro-productivity, and discount arrays.
// Rationale: the args have distinct lifetimes and ownership (some borrowed, some
// owned), so bundling them into one struct would buy nothing on this single-call
// cold path while obscuring the transpose inputs.
#[allow(clippy::too_many_arguments)]
fn assemble_stage_templates_output(
    stage_outputs: Vec<StageBuildOutput>,
    load_bus_indices: Vec<usize>,
    diversion_upstream_output: HashMap<EntityId, Vec<usize>>,
    study_stages: &[&cobre_core::Stage],
    ctx: &TemplateBuildCtx<'_>,
    par_lp: &PrecomputedPar,
    n_hydros: usize,
    n_load_buses: usize,
    n_study: usize,
) -> StageTemplates {
    // Index `s` of every parallel Vec must refer to the same stage, so preserve the
    // per-stage push order.
    let mut templates = Vec::with_capacity(n_study);
    let mut base_rows = Vec::with_capacity(n_study);
    let mut load_balance_row_starts = Vec::with_capacity(n_study);
    let mut generic_constraint_row_entries = Vec::with_capacity(n_study);
    let mut ncs_col_starts = Vec::with_capacity(n_study);
    let mut pumping_col_starts = Vec::with_capacity(n_study);
    let mut geometry_per_stage = Vec::with_capacity(n_study);
    // The dense NCS/pumping counts are constant across stages, so they collapse to
    // scalars (column STARTS stay per-stage, riding `n_blks`): the first output
    // seeds the scalars, later outputs must agree.
    let mut n_ncs: usize = 0;
    let mut n_pumping: usize = 0;
    for (s, out) in stage_outputs.into_iter().enumerate() {
        templates.push(out.template);
        base_rows.push(out.stage_base_row);
        load_balance_row_starts.push(out.load_balance_row_start);
        generic_constraint_row_entries.push(out.gc_entries);
        ncs_col_starts.push(out.ncs_col_start);
        pumping_col_starts.push(out.pumping_col_start);
        if s == 0 {
            n_ncs = out.ncs_count;
            n_pumping = out.n_pumping;
        } else {
            debug_assert_eq!(
                out.ncs_count, n_ncs,
                "dense NCS count must be constant across stages",
            );
            debug_assert_eq!(
                out.n_pumping, n_pumping,
                "dense pumping count must be constant across stages",
            );
        }
        geometry_per_stage.push(out.equipment_geometry);
    }

    let (noise_scale, zeta_per_stage, block_hours_per_stage) =
        scaling::compute_noise_scale(study_stages, ctx.hydros, n_hydros, par_lp);

    let hydro_productivities_per_stage: Vec<Vec<f64>> = (0..n_study)
        .map(|s| {
            (0..n_hydros)
                .map(|h| match ctx.production_models.model(h, s) {
                    ResolvedProductionModel::ConstantProductivity { productivity } => *productivity,
                    ResolvedProductionModel::Fpha { .. } => 0.0,
                })
                .collect()
        })
        .collect();

    StageTemplates {
        templates,
        base_rows,
        noise_scale,
        zeta_per_stage,
        block_hours_per_stage,
        n_hydros,
        load_balance_row_starts,
        n_load_buses,
        load_bus_indices,
        generic_constraint_row_entries,
        ncs_col_starts,
        n_ncs,
        pumping_col_starts,
        n_pumping,
        geometry_per_stage,
        diversion_upstream: diversion_upstream_output,
        hydro_productivities_per_stage,
        // 1.0-placeholders until `StageTemplates::set_discount_factors`.
        discount_factors: vec![1.0; n_study],
        cumulative_discount_factors: vec![1.0; n_study],
    }
}

#[cfg(test)]
mod tests;
