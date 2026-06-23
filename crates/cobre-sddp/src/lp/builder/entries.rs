use cobre_core::{CoefficientRef, ConstraintSense, Stage};

use crate::generic_constraints::resolve_variable_ref;
use crate::hydro_models::{EvaporationModel, ResolvedProductionModel};

use super::M3S_TO_HM3;
use super::fpha_cursor::for_each_fpha_plane;
use super::layout::{StageLayout, TemplateBuildCtx};

/// Fill CSC matrix entries for anticipated-fishing equality constraints.
///
/// For each anticipated plant `i` (always-active predicate), writes:
/// - `(row, +block_hours[blk])` on each per-block thermal column
///   `col_thermal_start + thermal_idx * n_blks + blk` (`LHS`, `MWh`).
/// - `(row, -block_hours_total)` on the anticipated-state slot-0 column
///   `col_anticipated_state_start + local_idx` (`RHS` coupling: `MW` × h = `MWh`).
///
/// The constraint enforces that the total generated energy (sum over blocks) equals
/// the committed power level (slot 0 content) scaled to `MWh`.
///
/// When `n_anticipated == 0`, this function is a no-op.
pub(super) fn fill_anticipated_fishing_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;
    let grid = layout.block_grid();
    for local_idx in 0..ctx.n_anticipated {
        let row = layout.anticipated.row_anticipated_fishing_start + local_idx;
        let thermal_idx = ctx.anticipated_thermal_indices[local_idx];
        let mut block_hours_total: f64 = 0.0;
        for blk in 0..n_blks {
            let col_gen = grid.flat(layout.col_thermal_start(), thermal_idx, blk);
            let block_hours = stage.blocks[blk].duration_hours;
            col_entries[col_gen].push((row, block_hours));
            block_hours_total += block_hours;
        }
        let col_state = layout.col_anticipated_state_start() + local_idx;
        col_entries[col_state].push((row, -block_hours_total));
    }
    debug_assert_eq!(
        ctx.n_anticipated, layout.anticipated.n_anticipated_fishing_rows,
        "fill_anticipated_fishing_entries: row count must equal n_anticipated"
    );
}

/// Fill CSC entries for the `anticipated_state_out` definition equality rows.
///
/// For each active anticipated plant `i` (`stage_idx + K_i < n_stages`),
/// emits TWO CSC entries at the definition row:
/// - `(row, +1.0)` on `col_anticipated_state_out_start + i`
/// - `(row, -1.0)` on `col_anticipated_decision_start + i`
///
/// Encodes the equality `anticipated_state_out[i] − decision_col[i] = 0`.
/// Row bounds are filled by `super::rows::fill_anticipated_state_out_def_rows`. The
/// final CSC ordering is enforced by the per-column
/// `sort_unstable_by_key(|&(row, _)| row)` pass in
/// `build_single_stage_template`, so the relative order of the two pushes here
/// does not matter for correctness.
///
/// Inactive plants emit no entries.
///
/// No-op when `n_anticipated == 0` or when no plant is active.
pub(super) fn fill_anticipated_state_out_def_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_stages = ctx.resolved.bounds.n_stages();
    let mut active_pos: usize = 0;
    for local_idx in 0..ctx.n_anticipated {
        if !layout.is_anticipated_decision_active(
            local_idx,
            stage_idx,
            n_stages,
            &ctx.anticipated_windows,
            &ctx.study_stage_ids,
        ) {
            continue;
        }
        let row = layout.anticipated.row_anticipated_state_out_def_start + active_pos;
        let col_state_out = layout.anticipated.col_anticipated_state_out_start + local_idx;
        let col_decision = layout.anticipated.col_anticipated_decision_start + local_idx;
        col_entries[col_state_out].push((row, 1.0));
        col_entries[col_decision].push((row, -1.0));
        active_pos += 1;
    }
    debug_assert_eq!(
        active_pos, layout.anticipated.n_anticipated_state_out_def_rows,
        "fill_anticipated_state_out_def_entries: active_pos mismatch at stage {stage_idx}"
    );
}

/// Returns `true` when hydro `h_idx` is in the `PreFilling` phase at this stage.
///
/// Derived inline per `(stage, hydro)` from [`crate::lp_builder::filling_phase`]
/// — no cached per-stage mask (the dense-layout convention every gating site
/// follows). A hydro with no `FillingConfig` is `Operating` at every stage, so
/// this is `false`, and the water-balance fill keeps its standard form
/// (parity-neutral).
#[inline]
pub(super) fn is_prefilling(ctx: &TemplateBuildCtx<'_>, stage: &Stage, h_idx: usize) -> bool {
    let hydro = &ctx.hydros[h_idx];
    matches!(
        crate::lp_builder::filling_phase(hydro.filling.as_ref(), hydro.entry_stage_id, stage.id),
        crate::lp_builder::Phase::PreFilling
    )
}

/// Resolve the cascade target that an absent `PreFilling` hydro `h_idx` routes its
/// water onto: the FIRST downstream hydro that is NOT `PreFilling` at this stage.
///
/// Walks `h → downstream(h) → downstream(downstream(h)) → …`, skipping every
/// consecutive `PreFilling` hydro, and returns the system index of the first
/// hydro that is `Filling`, `Operating`, or non-filling — the first hydro whose
/// water-balance row is a NORMAL balance that can absorb the routed inflow,
/// upstream releases, diversion, and withdrawal. Returns `None` when the chain
/// reaches a terminal (`downstream == None`) or an unresolved id, or stays
/// `PreFilling` all the way down: in that SINK case `h`'s water exits the system,
/// exactly as a terminal hydro's outflow does.
///
/// The short-circuit target MUST be a non-`PreFilling` downstream: a `PreFilling`
/// downstream's row is the frozen identity `v_d − v_d_in = 0`, and routing inflow
/// / upstream / withdrawal terms onto it CORRUPTS that frozen constraint (it must
/// stay free of every such term — see [`fill_prefilling_shortcircuit`] and the
/// `super::entries::is_prefilling` frozen-row fill). The forbidden alternative is
/// routing to the immediate `downstream(h)` unconditionally (the prior behaviour):
/// when that immediate downstream is itself `PreFilling`, its frozen row is
/// corrupted and the cut is wrong yet still compiles.
///
/// Acyclicity: the cascade is validated acyclic by `check_cascade_acyclic`, so the
/// walk terminates; the `n_h`-bounded loop is defense-in-depth against a
/// pathological cycle reaching here via direct test construction — it returns
/// `None` (sink) rather than spinning, with no panic/unwrap.
pub(super) fn resolve_shortcircuit_target(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    h_idx: usize,
) -> Option<usize> {
    let mut current_id = ctx.hydros[h_idx].id;
    // Bound the walk by the hydro count: an acyclic cascade visits each node at
    // most once, so more than `n_h` hops can only mean a (validated-out) cycle.
    for _ in 0..ctx.hydros.len() {
        let down_id = ctx.cascade.downstream(current_id)?;
        let d_idx = *ctx.hydro_pos.get(&down_id)?;
        if !is_prefilling(ctx, stage, d_idx) {
            return Some(d_idx);
        }
        current_id = down_id;
    }
    None
}

/// Fill water-balance row entries into `col_entries`.
///
/// Writes entries for the water-balance rows (outgoing/incoming
/// storage, turbine, spillage, upstream cascade, and AR lag
/// dynamics). Incoming state is pinned via column bounds, so no
/// row-equality state-fixing diagonals are written here.
///
/// A hydro in the `PreFilling` phase is short-circuited: its own row collapses to
/// the frozen-storage identity `v_h − v_h_in = 0` and its water interactions
/// (incremental inflow, upstream releases, withdrawal) transfer to the first
/// non-`PreFilling` downstream hydro `d` (resolved by
/// [`resolve_shortcircuit_target`], cascading THROUGH any `PreFilling` downstream
/// whose row is itself a frozen identity). See [`fill_prefilling_shortcircuit`]
/// for the contract; the standard `Operating`/`Filling` fill below runs for every
/// non-`PreFilling` hydro unchanged.
pub(super) fn fill_state_and_water_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_h = layout.n_h;
    let n_blks = layout.n_blks;
    let lag_order = layout.lag_order;
    let zeta = layout.zeta;
    let row_water = layout.row_water_balance_start();
    let col_storage_in_start = layout.col_storage_in_start();
    let col_inflow_lags_start = layout.col_inflow_lags_start();

    // Water balance: outgoing storage (+1), incoming storage (-1),
    // turbine/spillage (+tau), upstream turbine/spillage (-tau),
    // and AR lag dynamics (-ζ*ψ).
    for h_idx in 0..n_h {
        let hydro = &ctx.hydros[h_idx];
        let row = row_water + h_idx;

        if is_prefilling(ctx, stage, h_idx) {
            // Frozen-storage identity `v_h − v_h_in = 0` for the absent reservoir.
            // The storage column keeps its dense system index `h_idx` (§4 trap 2:
            // NEVER omit or relocate the storage column — only the physics ON the
            // row changes); the incoming-storage column keeps its `−1.0`, pinned
            // per pass to `v̂_h` by `fill_col_state_patches`. Emitting ONLY these
            // two entries (no inflow/upstream/AR-lag/withdrawal/evaporation terms)
            // is the §4 trap-4 contract: with `h`'s row free of upstream/inflow
            // coupling, `v_h` is a dead variable, so perturbing `v̂_h` changes
            // nothing ⇒ `∂Q/∂v̂_h = 0` (a valid flat cut). Leaving ANY of `h`'s
            // upstream/inflow coupling on this row would make `β_h` stale-nonzero
            // — a wrong cut that still compiles. The frozen-identity RHS is `0`,
            // set by `super::rows::fill_water_balance_rows`, and the solve-time
            // noise patch is neutralized by zeroing this hydro's `noise_scale`
            // (see `super::scaling::compute_noise_scale`).
            col_entries[h_idx].push((row, 1.0));
            col_entries[col_storage_in_start + h_idx].push((row, -1.0));
            fill_prefilling_shortcircuit(ctx, stage, h_idx, layout, col_entries);
            continue;
        }

        col_entries[h_idx].push((row, 1.0));
        col_entries[col_storage_in_start + h_idx].push((row, -1.0));
        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col_turbine = layout.turbine_col(h_idx, blk);
            col_entries[col_turbine].push((row, tau_h));
            let col_spillage = layout.spillage_col(h_idx, blk);
            col_entries[col_spillage].push((row, tau_h));
            // Diversion outflow: this hydro's diversion column enters its own
            // water balance with +tau (outflow), same sign as turbine/spillage.
            let col_diversion = layout.diversion_col(h_idx, blk);
            col_entries[col_diversion].push((row, tau_h));
            // Cascade inflow: upstream turbine/spillage enter with -tau.
            for &up_id in ctx.cascade.upstream(hydro.id) {
                if let Some(&u_idx) = ctx.hydro_pos.get(&up_id) {
                    col_entries[layout.turbine_col(u_idx, blk)].push((row, -tau_h));
                    col_entries[layout.spillage_col(u_idx, blk)].push((row, -tau_h));
                }
            }
            // Diversion inflow: for each hydro that diverts TO this hydro, its
            // diversion column enters this hydro's water balance with -tau.
            if let Some(sources) = ctx.diversion_upstream.get(&hydro.id) {
                for &d_idx in sources {
                    let col_div = layout.diversion_col(d_idx, blk);
                    col_entries[col_div].push((row, -tau_h));
                }
            }
        }
        if ctx.par_lp.n_stages() > 0 && ctx.par_lp.n_hydros() == n_h {
            let psi = ctx.par_lp.psi_slice(stage_idx, h_idx);
            for (lag, &psi_val) in psi.iter().enumerate() {
                if psi_val != 0.0 && lag < lag_order {
                    let col = col_inflow_lags_start + lag * n_h + h_idx;
                    col_entries[col].push((row, -zeta * psi_val));
                }
            }
        }
    }

    // Inflow non-negativity slack: sigma_inf_h enters water balance with -ζ.
    // Skipped for a PreFilling hydro — its row is the frozen identity.
    if ctx.has_penalty {
        for h_idx in 0..n_h {
            if is_prefilling(ctx, stage, h_idx) {
                continue;
            }
            let col = layout.col_inflow_slack_start() + h_idx;
            let row = row_water + h_idx;
            col_entries[col].push((row, -zeta));
        }
    }

    // Evaporation: the per-hydro evaporation-outflow column enters water balance with +ζ.
    // Evaporation is an outflow (water leaving the reservoir), so its
    // coefficient matches the turbine/spillage sign convention (positive). A
    // PreFilling hydro is excluded from `evap_hydro_indices` upstream (no reservoir
    // surface, evaporation is zero and is NOT transferred to `d`), so this loop
    // never touches a frozen-identity row.
    for (local_idx, &h_idx) in layout.evap_hydro_indices.iter().enumerate() {
        let col_evaporation_flow = layout.evap_flow_col(local_idx);
        let row = row_water + h_idx;
        col_entries[col_evaporation_flow].push((row, zeta));
    }

    // Under-withdrawal slack (neg): adds water back to the balance.
    // When the reservoir cannot sustain the full scheduled withdrawal, the neg slack
    // absorbs the difference, reducing the effective withdrawal in that stage.
    // Skipped for a PreFilling hydro — `h`'s withdrawal demand is served by `d`'s
    // own withdrawal slacks (the demand transfers via `d`'s RHS in
    // `super::rows::fill_water_balance_rows`, not a slack on `h`'s frozen row).
    for h_idx in 0..n_h {
        if is_prefilling(ctx, stage, h_idx) {
            continue;
        }
        let col = layout.col_withdrawal_neg_start() + h_idx;
        let row = row_water + h_idx;
        col_entries[col].push((row, -zeta));
    }

    // Over-withdrawal slack (pos): removes additional water from the balance.
    // When the solver withdraws more than the target, the pos slack accounts for
    // the excess withdrawal at a penalty cost. Skipped for a PreFilling hydro (see
    // the neg-slack note above).
    for h_idx in 0..n_h {
        if is_prefilling(ctx, stage, h_idx) {
            continue;
        }
        let col = layout.col_withdrawal_pos_start() + h_idx;
        let row = row_water + h_idx;
        col_entries[col].push((row, zeta));
    }
}

/// Re-route an absent `PreFilling` hydro `h`'s water interactions onto the FIRST
/// non-`PreFilling` downstream hydro `d`'s water-balance row.
///
/// Before `start_stage_id` the dam at `h`'s site does not exist; the river flows
/// past it. `d` is NOT necessarily the immediate `downstream(h)`: when the
/// immediate downstream is itself `PreFilling`, its row is the frozen identity
/// `v_d − v_d_in = 0` and MUST NOT receive routed terms, so the water cascades
/// THROUGH it to the first existing reservoir downstream. The target is resolved by
/// [`resolve_shortcircuit_target`], which walks the chain skipping consecutive
/// `PreFilling` hydros; the forbidden alternative is routing to the immediate
/// `downstream(h)` unconditionally (which corrupts a `PreFilling` downstream's
/// frozen row — a wrong cut that still compiles).
///
/// Every water interaction of the absent `h` transfers to `d`, in the REAL
/// `ζ`/`τ_k` coefficients of the standard balance (NOT an approximation): by LP
/// duality the reduced cost of `d`'s pinned incoming-storage column is then the
/// TRUE sensitivity of the routed problem, so the harvested cut coefficient
/// `rc / col_scale` is a valid subgradient (§4, `PreFilling`-downstream). Synthesizing
/// a coefficient instead of routing the real columns would break that duality and
/// produce an invalid cut.
///
/// Conservation across a `PreFilling` chain (`h₁ → h₂ → … → d`, all of `h₁ … hₙ`
/// `PreFilling`, `d` the first non-`PreFilling`): each `PreFilling` hydro routes
/// its OWN incremental inflow + withdrawal to the SAME resolved `d` independently,
/// with no double-count. A skipped intermediate `PreFilling` hydro contributes ZERO
/// releases — its turbine/diversion are pinned `[0, 0]` and its spillage appears on
/// no row (a positive cost drives it to 0) — so referencing an intermediate's
/// release columns in a downstream's routing multiplies zero columns and adds
/// nothing. Each link's incremental inflow therefore reaches `d` exactly once.
///
/// Three pieces land on `d`'s row (`row_water + d_idx`):
///
/// - `h`'s incremental inflow, as the realized-inflow column `z_h` with `−ζ` (the
///   `z_h`→water coupling relocated from `h`'s row to `d`'s). `z_h` is pinned by
///   its own definition row to the realized scenario inflow `base_h + σ_h·η_h +
///   Σ_l ψ_l·lag_h`, and that machinery (the z-inflow definition row + its Category-5
///   patch) is left UNTOUCHED for `h`, so routing the column makes the inflow
///   scenario-exact with no new noise patch. The deterministic base is part of `z_h`,
///   so it must NOT also be added to `d`'s RHS (double-count).
/// - `h`'s upstream releases: for each `u ∈ upstream(h)`, per block, `u`'s
///   turbine/spillage with `−τ_h` — the cascade edge `upstream(h)→h→d` re-routed to
///   `upstream(h)→d` (upstream(h)'s releases feed `d` directly while `h` is absent).
///   Diversion-into-`h` columns are routed the same way (the "all water interactions
///   transfer to `d`" principle, §3.2), conserving water that would otherwise vanish.
///
/// `h`'s withdrawal DEMAND transfers to `d` via `d`'s RHS (served by `d`'s existing
/// withdrawal slacks); that is an RHS adjustment owned by
/// `super::rows::fill_water_balance_rows`, which MUST resolve the SAME `d` via
/// [`resolve_shortcircuit_target`] so the matrix routing and the RHS routing agree.
/// `h`'s evaporation is ZERO in `PreFilling` and is NOT transferred.
///
/// Sink case (no non-`PreFilling` downstream — terminal, unresolved id, or
/// `PreFilling` all the way down): `h`'s water exits the system, exactly as a
/// terminal hydro's outflow does today — nothing is routed, no panic.
fn fill_prefilling_shortcircuit(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    h_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let hydro = &ctx.hydros[h_idx];
    let Some(d_idx) = resolve_shortcircuit_target(ctx, stage, h_idx) else {
        // Sink: no non-PreFilling downstream row receives `h`'s water; it exits the
        // system (terminal, unresolved downstream id, or PreFilling all the way
        // down). Nothing routed rather than a panic.
        return;
    };
    let n_blks = layout.n_blks;
    let zeta = layout.zeta;
    let row_d = layout.row_water_balance_start() + d_idx;

    // `h`'s realized incremental inflow arrives at `d`: the `z_h` column with `−ζ`
    // (inflow sign), identical to how `fill_filling_retention_entries` references
    // the realized-inflow column. `z_h` is a stage-level column (not per-block).
    col_entries[layout.col_z_inflow_start() + h_idx].push((row_d, -zeta));

    for blk in 0..n_blks {
        let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
        // Upstream releases re-routed `upstream(h)→d` with `−τ_h` (inflow to `d`).
        for &up_id in ctx.cascade.upstream(hydro.id) {
            if let Some(&u_idx) = ctx.hydro_pos.get(&up_id) {
                col_entries[layout.turbine_col(u_idx, blk)].push((row_d, -tau_h));
                col_entries[layout.spillage_col(u_idx, blk)].push((row_d, -tau_h));
            }
        }
        // Diversion-into-`h` re-routed to `d` with `−τ_h`, same sign as upstream
        // releases — the absent `h`'s diversion inflow now reaches `d` directly.
        if let Some(sources) = ctx.diversion_upstream.get(&hydro.id) {
            for &src_idx in sources {
                col_entries[layout.diversion_col(src_idx, blk)].push((row_d, -tau_h));
            }
        }
    }
}

/// Fill impound-retention row entries into `col_entries`.
///
/// For each hydro in the Filling phase at this stage (the `filling_retention`
/// family enumerated by `layout.filling_retention_hydro_indices`), writes the LHS
/// of one `≥` retention row at `row_filling_retention_start() + local_idx`. The
/// row reads, in hm³:
///
/// ```text
/// Σ_blk τ_h·(release_h) − Σ_blk τ_h·(upstream release) − ζ·z_h ≥ −ζ·F_h
/// ```
///
/// equivalently `total release ≥ (incremental natural inflow ζ·z_h) − ζ·F_h`,
/// the §3.1 cap "total release ≥ natural inflow into the reservoir − `F_h`" with
/// upstream releases moved to the LHS as cascade columns. The release/cascade
/// columns and their `τ_h = block_hours · M3S_TO_HM3` coefficients are IDENTICAL
/// to the standard water-balance row's outflow/cascade terms (see
/// [`fill_state_and_water_entries`]):
///
/// - this hydro's own release (spillage, +turbine, +diversion) columns with
///   `+τ_h` — during Filling the turbine column is pinned `[0, 0]` and diversion
///   is `≡ 0`, so spillage carries the entire release, but the +τ entries are
///   written on all three for structural identity with the standard balance;
/// - upstream cascade releases (`q_u`, `s_u`) with `−τ_h`;
/// - diversion-into columns with `−τ_h`.
///
/// The incremental natural inflow enters as the `z_h` realized-inflow column with
/// coefficient `−ζ`, NOT as a deterministic RHS constant `ζ·base_h`. The `z_h`
/// column is pinned by its own definition row to the REALIZED scenario inflow
/// `base_h + σ_h·η_h + Σ_l ψ_l·lag_h` (the same realized inflow the standard
/// water-balance row consumes via its AR-lag columns + solve-time noise patch), so
/// referencing the column makes the cap scenario-exact with NO new noise patch.
/// The wrong-but-compiling alternative — a deterministic RHS `ζ·base_h − ζ·F_h` —
/// would cap against the mean inflow while the reservoir receives the realized
/// inflow, weakening the cap in wet scenarios (the reservoir could impound more
/// than `ζ·F_h` of the realized inflow) and making the policy depend on the noise
/// realization through an unintended channel. The retention RHS therefore carries
/// only the constant `−ζ·F_h` (set by
/// [`super::rows::fill_filling_retention_rows`]).
///
/// Contract: this row is a CAP, not an equality — storage rises by AT MOST
/// `ζ·F_h` per stage. Rising LESS (insufficient natural inflow) is permitted and
/// is caught terminally by the `σ_fill` slack, never here. The standard
/// water-balance EQUALITY row (with its PAR base RHS, noise patch, and AR-lag
/// terms) is kept UNCHANGED — this retention inequality is ADDITIONAL, never a
/// replacement. Folding the cap into the standard balance by editing its RHS would
/// drop the PAR/AR-lag coupling and break the inflow dynamics.
///
/// No-op when no hydro is in the Filling phase (the index list is empty).
pub(super) fn fill_filling_retention_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;
    let zeta = layout.zeta;
    let row_start = layout.row_filling_retention_start();
    let col_z_inflow_start = layout.col_z_inflow_start();
    for (local_idx, &h_idx) in layout.filling_retention_hydro_indices.iter().enumerate() {
        let hydro = &ctx.hydros[h_idx];
        let row = row_start + local_idx;
        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            // Own release: spillage (+turbine [0,0], +diversion ≡0 during Filling)
            // with +τ_h, matching the standard balance's outflow sign.
            col_entries[layout.turbine_col(h_idx, blk)].push((row, tau_h));
            col_entries[layout.spillage_col(h_idx, blk)].push((row, tau_h));
            col_entries[layout.diversion_col(h_idx, blk)].push((row, tau_h));
            // Upstream cascade releases enter with −τ_h (they are inflow to this
            // reservoir), identical to the standard balance.
            for &up_id in ctx.cascade.upstream(hydro.id) {
                if let Some(&u_idx) = ctx.hydro_pos.get(&up_id) {
                    col_entries[layout.turbine_col(u_idx, blk)].push((row, -tau_h));
                    col_entries[layout.spillage_col(u_idx, blk)].push((row, -tau_h));
                }
            }
            // Diversion-into columns enter with −τ_h, identical to the standard
            // balance.
            if let Some(sources) = ctx.diversion_upstream.get(&hydro.id) {
                for &d_idx in sources {
                    col_entries[layout.diversion_col(d_idx, blk)].push((row, -tau_h));
                }
            }
        }
        // Realized incremental inflow ζ·z_h enters with −ζ (it is inflow). z_h is a
        // stage-level column (not per-block), so this is one entry per row.
        col_entries[col_z_inflow_start + h_idx].push((row, -zeta));
    }
}

/// Fill terminal `σ_fill`-target row entries into `col_entries`.
///
/// For each filling hydro whose `entry_stage_id − 1 == stage.id` at this stage (the
/// `filling_target` family enumerated by `layout.filling_target_hydro_indices`),
/// writes the LHS of one `≥` target row at `row_filling_target_start() + local_idx`:
///
/// ```text
/// v_h + σ_fill ≥ min_storage_hm3
/// ```
///
/// - `+1.0` on the OUTGOING storage column `v_h`, which is the dense system index
///   `h_idx` — the same column `fill_state_and_water_entries` writes the
///   water-balance `+1.0` onto (`col_entries[h_idx]`). The storage column is never
///   omitted or relocated at any phase (§4 trap 2), so addressing it by `h_idx`
///   here is correct in every phase.
/// - `+1.0` on the `σ_fill` slack column at `col_filling_target_start() +
///   local_idx`, parallel to the row index.
///
/// The `≥` sense and the `min_storage_hm3` RHS live in the row bounds
/// (`row_lower = min_storage_hm3`, `row_upper = +∞`, set by
/// [`super::rows::fill_filling_target_rows`]); this fill writes only the two unit
/// coefficients.
///
/// Cut validity (§4 trap 3): this row couples to the storage state through the
/// constraint matrix, so LP duality folds the `σ_fill` soft-row dual into the
/// SINGLE reduced cost of the incoming-storage column that the cut already reads as
/// `rc / col_scale`. NEVER separately extract this row's dual and add it to the cut
/// coefficient by hand — that double-counts the soft floor. The dual-extraction
/// entry point (in `training/backward/duals_extraction.rs`) is correct as-is and is
/// not touched by this family — a guard test in this module asserts no `lp/builder`
/// file references it.
///
/// No-op at every non-terminal stage and for a non-filling system (the index list
/// is empty).
fn fill_filling_target_entries(layout: &StageLayout, col_entries: &mut [Vec<(usize, f64)>]) {
    let row_start = layout.row_filling_target_start();
    let col_start = layout.col_filling_target_start();
    for (local_idx, &h_idx) in layout.filling_target_hydro_indices.iter().enumerate() {
        let row = row_start + local_idx;
        // +1 on the outgoing storage column v_h (dense system index h_idx).
        col_entries[h_idx].push((row, 1.0));
        // +1 on the σ_fill slack column.
        col_entries[col_start + local_idx].push((row, 1.0));
    }
}

/// Fill soft `σ^{v-}` operating-floor row entries into `col_entries`.
///
/// For each filling hydro in the Operating phase at this stage (the
/// `filling_floor` family enumerated by `layout.filling_floor_hydro_indices`),
/// writes the LHS of one `≥` operating-floor row at
/// `row_filling_floor_start() + local_idx`:
///
/// ```text
/// v_h + σ^{v-} ≥ min_storage_hm3
/// ```
///
/// - `+1.0` on the OUTGOING storage column `v_h`, which is the dense system index
///   `h_idx` — the same column `fill_state_and_water_entries` writes the
///   water-balance `+1.0` onto (`col_entries[h_idx]`). The storage column is never
///   omitted or relocated at any phase (§4 trap 2), so addressing it by `h_idx`
///   here is correct in every phase.
/// - `+1.0` on the `σ^{v-}` slack column at `col_filling_floor_start() +
///   local_idx`, parallel to the row index.
///
/// The `≥` sense and the `min_storage_hm3` RHS live in the row bounds
/// (`row_lower = min_storage_hm3`, `row_upper = +∞`, set by
/// [`super::rows::fill_filling_floor_rows`]); this fill writes only the two unit
/// coefficients.
///
/// Cut validity (§4 trap 3): like the sibling `σ_fill` row, this row couples to the
/// storage state through the constraint matrix, so LP duality folds the `σ^{v-}`
/// soft-row dual into the SINGLE reduced cost of the incoming-storage column that
/// the cut already reads as `rc / col_scale`. NEVER separately extract this row's
/// dual and add it to the cut coefficient by hand — that double-counts the soft
/// floor. The dual-extraction entry point (in
/// `training/backward/duals_extraction.rs`) is correct as-is and is not touched by
/// this family — the `lp_builder_never_references_dual_extraction` guard test in
/// this module asserts no `lp/builder` file references it.
///
/// DISTINCT from `fill_filling_target_entries` (`σ_fill`, terminal Filling stage
/// only): same `v_h + slack ≥ min_storage` shape, but a different slack column,
/// different stage scope (every Operating stage), and a different cost — never
/// conflate them.
///
/// No-op at every non-operating stage of a filling hydro and for a non-filling
/// system (the index list is empty).
fn fill_filling_floor_entries(layout: &StageLayout, col_entries: &mut [Vec<(usize, f64)>]) {
    let row_start = layout.row_filling_floor_start();
    let col_start = layout.col_filling_floor_start();
    for (local_idx, &h_idx) in layout.filling_floor_hydro_indices.iter().enumerate() {
        let row = row_start + local_idx;
        // +1 on the outgoing storage column v_h (dense system index h_idx).
        col_entries[h_idx].push((row, 1.0));
        // +1 on the σ^{v-} slack column.
        col_entries[col_start + local_idx].push((row, 1.0));
    }
}

/// Fill pumping-flow water-balance row entries into `col_entries`.
///
/// A pumping station moves water from its SOURCE reservoir to its DESTINATION
/// reservoir. The pumped flow column therefore enters two water rows per block:
///
/// - the SOURCE hydro's water row with `+tau_h` (water LEAVES the source — the
///   same sign a turbine/spillage outflow carries in a plant's own row, see
///   [`fill_state_and_water_entries`]);
/// - the DESTINATION hydro's water row with `−tau_h` (water ARRIVES at the
///   destination — the same sign cascade-upstream inflow carries in the
///   downstream row, see [`fill_state_and_water_entries`]).
///
/// `tau_h` is the identical `stage.blocks[blk].duration_hours * M3S_TO_HM3`
/// expression used by turbine/spillage in [`fill_state_and_water_entries`] —
/// the same arithmetic on the same operands, so the coefficient is bit-identical
/// across the two sites; computing a differently-rounded τ would desynchronise
/// pumping from the cascade water terms.
///
/// Stations are iterated dense by SYSTEM index `p_sys` (the dense column
/// position), mirroring how the NCS column fill iterates NCS. The structural ±τ
/// entries are written for every station regardless of its commissioning window:
/// a dormant station's pumping column is forced to `[0, 0]` in
/// [`super::columns::fill_pumping_columns`], so its ±τ terms multiply a zero
/// column and contribute nothing (and presolve away). A source or destination
/// hydro id absent from `ctx.hydro_pos` skips only that side's entry — the
/// present side is still written, and no panic occurs (semantic validation of the
/// references is a separate concern).
pub(super) fn fill_pumping_water_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;
    let grid = layout.block_grid();
    let row_water = layout.row_water_balance_start();
    for (p_sys, station) in ctx.pumping_stations.iter().enumerate() {
        // `validate_pumping_station_refs` (run from `SystemBuilder::build()`)
        // guarantees both refs resolve on a validated `System`, so on the
        // production path both `Option`s are `Some`. The per-side `if let
        // Some(...)` guards below are defense-in-depth for direct-construction
        // test paths that bypass validation — never a production branch. A
        // one-sided resolve writes a feasible-but-wrong half water coupling, so
        // the guards must not be promoted to an unconditional index/expect.
        let source = ctx.hydro_pos.get(&station.source_hydro_id).copied();
        let destination = ctx.hydro_pos.get(&station.destination_hydro_id).copied();
        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col = grid.flat(layout.col_pumping_start, p_sys, blk);
            if let Some(s_idx) = source {
                col_entries[col].push((row_water + s_idx, tau_h));
            }
            if let Some(d_idx) = destination {
                col_entries[col].push((row_water + d_idx, -tau_h));
            }
        }
    }
}

/// Fill load-balance entries into `col_entries`.
///
/// Writes entries for hydro turbine generation, thermal generation,
/// line forward/reverse flows, pumping power consumption, and deficit/excess slacks.
///
/// For FPHA hydros the generation variable `g_{h,k}` (in `col_generation_start`)
/// enters the load balance with coefficient +1.0 instead of `rho * turbine_col`.
/// For constant-productivity hydros the original `rho * turbine_col` behavior is unchanged.
///
/// Pumping power `Eb = consumption_mw_per_m3s · Qb` is a negative injection on the
/// station's bus: the `pumping_flow` column enters the load-balance row with
/// `−consumption_mw_per_m3s` — the same negative sign a line carries into its
/// source bus (`col_fwd` → `−1.0` below), not the `+1.0` of generation. There is
/// no separate power column; the coefficient scales the SAME flow column, so a
/// positive coefficient (treating pumping as generation) would credit the bus for
/// power the station consumes.
pub(super) fn fill_load_balance_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;
    let grid = layout.block_grid();
    let row_load = layout.row_load_balance_start();

    for (h_idx, hydro) in ctx.hydros.iter().enumerate() {
        if let Some(local_idx) = layout.fpha_local_index[h_idx] {
            debug_assert!(
                matches!(
                    ctx.production_models.model(h_idx, stage_idx),
                    ResolvedProductionModel::Fpha { .. }
                ),
                "FPHA local-index table inconsistent with production model for hydro {h_idx}"
            );
            if let Some(&b_idx) = ctx.bus_pos.get(&hydro.bus_id) {
                for blk in 0..n_blks {
                    let row = grid.flat(row_load, b_idx, blk);
                    let col = layout.generation_col(local_idx, blk);
                    col_entries[col].push((row, 1.0));
                }
            }
        } else {
            let rho = match ctx.production_models.model(h_idx, stage_idx) {
                ResolvedProductionModel::ConstantProductivity { productivity } => *productivity,
                ResolvedProductionModel::Fpha { .. } => {
                    unreachable!(
                        "non-FPHA branch reached for FPHA resolved model at hydro {h_idx}"
                    );
                }
            };
            if let Some(&b_idx) = ctx.bus_pos.get(&hydro.bus_id) {
                for blk in 0..n_blks {
                    let row = grid.flat(row_load, b_idx, blk);
                    let col = layout.turbine_col(h_idx, blk);
                    col_entries[col].push((row, rho));
                }
            }
        }
    }

    for (t_idx, thermal) in ctx.thermals.iter().enumerate() {
        if let Some(&b_idx) = ctx.bus_pos.get(&thermal.bus_id) {
            for blk in 0..n_blks {
                let row = grid.flat(row_load, b_idx, blk);
                let col = grid.flat(layout.col_thermal_start(), t_idx, blk);
                col_entries[col].push((row, 1.0));
            }
        }
    }

    for (l_idx, line) in ctx.lines.iter().enumerate() {
        let src_idx = ctx.bus_pos.get(&line.source_bus_id).copied();
        let tgt_idx = ctx.bus_pos.get(&line.target_bus_id).copied();
        for blk in 0..n_blks {
            let col_fwd = layout.line_fwd_col(l_idx, blk);
            let col_rev = layout.line_rev_col(l_idx, blk);
            if let Some(tgt) = tgt_idx {
                let row = grid.flat(row_load, tgt, blk);
                col_entries[col_fwd].push((row, 1.0));
                col_entries[col_rev].push((row, -1.0));
            }
            if let Some(src) = src_idx {
                let row = grid.flat(row_load, src, blk);
                col_entries[col_fwd].push((row, -1.0));
                col_entries[col_rev].push((row, 1.0));
            }
        }
    }

    // Pumping power: negative injection on the station's bus. Iterate ALL
    // stations dense by SYSTEM index `p_sys` (the dense column position), the
    // same split the NCS fill uses. The structural entry is written for every
    // station regardless of its commissioning window: a dormant station's pumping
    // column is `[0, 0]`, so this `-consumption` term multiplies a zero column and
    // contributes no bus injection. A bus id absent from `bus_pos` skips that
    // station with no entry (semantic validation is a separate concern).
    for (p_sys, station) in ctx.pumping_stations.iter().enumerate() {
        if let Some(&b_idx) = ctx.bus_pos.get(&station.bus_id) {
            for blk in 0..n_blks {
                let row = grid.flat(row_load, b_idx, blk);
                let col = grid.flat(layout.col_pumping_start, p_sys, blk);
                col_entries[col].push((row, -station.consumption_mw_per_m3s));
            }
        }
    }

    for (b_idx, bus) in ctx.buses.iter().enumerate() {
        for blk in 0..n_blks {
            let row = grid.flat(row_load, b_idx, blk);
            for seg_idx in 0..bus.deficit_segments.len() {
                let col_def = layout.deficit_col(b_idx, seg_idx, blk);
                col_entries[col_def].push((row, 1.0));
            }
            let col_exc = grid.flat(layout.col_excess_start(), b_idx, blk);
            col_entries[col_exc].push((row, -1.0));
        }
    }
}

/// Fill FPHA hyperplane constraint entries into `col_entries`.
///
/// For each FPHA hydro `h` at this stage, for each block `k`, for each
/// hyperplane `m`, adds matrix entries to FPHA row `r(h,k,m)`:
///
/// ```text
/// g_{h,k}  column:  +1.0              (generation variable)
/// v        column:  -gamma_v/2         (outgoing storage)
/// v_in     column:  -gamma_v/2         (incoming storage; fixed by storage-fixing row)
/// q_{h,k}  column:  -gamma_q           (turbined flow)
/// s_{h,k}  column:  -gamma_s           (spillage)
/// ```
///
/// These entries implement `g - gamma_v/2*v - gamma_v/2*v_in - gamma_q*q - gamma_s*s <= gamma_0`,
/// where `gamma_0` is already encoded in the row upper bound set by `super::rows::fill_stage_rows`.
///
/// FPHA uses **average** storage `(V_in + V_out)/2`, so `-gamma_v/2` lands on
/// BOTH the outgoing-storage column (`v = h_idx`) and the incoming-storage column
/// (`v_in = col_storage_in_start + h_idx`). Putting `-gamma_v/2` on `v` (`V_out`)
/// alone compiles and passes single-plane single-hydro tests, but understates
/// generation by the `V_in` head term — the wrong-but-compiling alternative that
/// deterministic case D06 pins against.
///
/// Driven by [`for_each_fpha_plane`] so the matrix coefficients and the row
/// bounds set by [`super::rows::fill_fpha_rows`] share one row cursor.
pub(super) fn fill_fpha_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let col_storage_in_start = layout.col_storage_in_start();
    for_each_fpha_plane(
        ctx,
        stage_idx,
        layout,
        |local_idx, h_idx, blk, _p_idx, plane, row| {
            let col_v = h_idx; // outgoing storage column
            let col_v_in = col_storage_in_start + h_idx; // incoming storage column
            let col_q = layout.turbine_col(h_idx, blk);
            let col_s = layout.spillage_col(h_idx, blk);
            let col_g = layout.generation_col(local_idx, blk);

            // g_{h,k} column: +1.0
            col_entries[col_g].push((row, 1.0));
            // v (outgoing storage): -gamma_v/2 — average-storage term, also on v_in below.
            col_entries[col_v].push((row, -plane.gamma_v / 2.0));
            // v_in (incoming storage, fixed by storage-fixing row): -gamma_v/2.
            col_entries[col_v_in].push((row, -plane.gamma_v / 2.0));
            // q_{h,k} (turbine): -gamma_q
            col_entries[col_q].push((row, -plane.gamma_q));
            // s_{h,k} (spillage): -gamma_s
            col_entries[col_s].push((row, -plane.gamma_s));
        },
    );
}

/// Fill CSC matrix entries for the evaporation constraint rows.
///
/// For each evaporation hydro `h` at local position `local_idx`, the equality row
/// `row_evap_start + local_idx` encodes:
///
/// ```text
/// evaporation_flow column:  +1.0
/// v_h     column:  -volume_slope_m3s_per_hm3 / 2   (outgoing storage)
/// v_in_h  column:  -volume_slope_m3s_per_hm3 / 2   (incoming storage; fixed by storage-fixing row)
/// f_plus  column:  +1.0
/// f_minus column:  -1.0
/// ```
///
/// These entries implement
/// `evaporation_flow - volume_slope_m3s_per_hm3/2*v - volume_slope_m3s_per_hm3/2*v_in + f_plus - f_minus = intercept_m3s`,
/// where `intercept_m3s` is already encoded in the row bounds set by
/// `super::rows::fill_stage_rows`. When `v_in` is fixed to value `V`, the effective RHS
/// becomes `intercept_m3s + volume_slope_m3s_per_hm3/2 * V`, which matches the
/// linearized evaporation at the average volume `(v + V) / 2`.
pub(super) fn fill_evaporation_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let col_storage_in_start = layout.col_storage_in_start();

    for (local_idx, &h_idx) in layout.evap_hydro_indices.iter().enumerate() {
        let coeff = match ctx.evaporation_models.model(h_idx) {
            EvaporationModel::Linearized { coefficients, .. } => {
                debug_assert!(
                    stage_idx < coefficients.len(),
                    "evap_hydro_indices contains hydro {h_idx} but coefficients length {} \
                     is less than stage_idx {}",
                    coefficients.len(),
                    stage_idx
                );
                match coefficients.get(stage_idx) {
                    Some(c) => *c,
                    None => continue,
                }
            }
            EvaporationModel::None => {
                // Should never happen: evap_hydro_indices only contains linearized hydros.
                debug_assert!(
                    false,
                    "evap_hydro_indices contains hydro {h_idx} but model is None"
                );
                continue;
            }
        };

        let col_evaporation_flow = layout.evap_flow_col(local_idx);
        let col_f_plus = layout.evap_f_plus_col(local_idx);
        let col_f_minus = layout.evap_f_minus_col(local_idx);
        let col_v = h_idx; // outgoing storage column
        let col_v_in = col_storage_in_start + h_idx; // incoming storage column

        let row = layout.row_evap_start() + local_idx;

        col_entries[col_evaporation_flow].push((row, 1.0));
        col_entries[col_v].push((row, -coeff.volume_slope_m3s_per_hm3 / 2.0));
        col_entries[col_v_in].push((row, -coeff.volume_slope_m3s_per_hm3 / 2.0));
        col_entries[col_f_plus].push((row, 1.0));
        col_entries[col_f_minus].push((row, -1.0));
    }
}

/// Mutable LP matrix buffers for stage template construction.
///
/// Groups the column and row arrays that are filled during template building.
pub(super) struct LpMatrixBuffers<'a> {
    /// CSC column entries (column index -> list of (row, coefficient)).
    pub(super) col_entries: &'a mut [Vec<(usize, f64)>],
    /// Column upper bounds.
    pub(super) col_upper: &'a mut [f64],
    /// Objective function coefficients.
    pub(super) objective: &'a mut [f64],
    /// Row lower bounds.
    pub(super) row_lower: &'a mut [f64],
    /// Row upper bounds.
    pub(super) row_upper: &'a mut [f64],
}

/// Fill CSC matrix entries, row bounds, and slack column data for all active
/// generic constraint rows at this stage.
///
/// For each active `(constraint, block)` pair recorded in
/// `layout.generic_constraint_rows`:
///
/// 1. Sets `row_lower` / `row_upper` for the generic constraint row according
///    to the constraint sense:
///    - `<=`: `row_lower = -INF`, `row_upper = bound`
///    - `>=`: `row_lower = bound`, `row_upper = +INF`
///    - `==`: `row_lower = bound`, `row_upper = bound`
///
/// 2. Iterates over the constraint expression terms, calls
///    `resolve_variable_ref` for each `LinearTerm`, and pushes
///    `(row_index, coefficient * multiplier)` entries into `col_entries`.
///
/// 3. When `slack.enabled = true`, sets slack column bounds to `[0, +INF)` and
///    objective to `penalty * block_hours`:
///    - `<=`: one slack column `s_g` with CSC entry `(row, -1.0)`.
///    - `>=`: one slack column `s_g` with CSC entry `(row, +1.0)`.
///    - `==`: two slack columns `s_g_plus` and `s_g_minus` with CSC entries
///      `(row, +1.0)` and `(row, -1.0)` respectively.
///
/// Unknown entity IDs in variable refs produce zero contributions (the empty
/// vec returned by `resolve_variable_ref` is skipped), which is the
/// defense-in-depth fallback for referential validation gaps.
pub(super) fn fill_generic_constraint_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    buffers: &mut LpMatrixBuffers<'_>,
) {
    let col_entries = &mut *buffers.col_entries;
    let col_upper = &mut *buffers.col_upper;
    let objective = &mut *buffers.objective;
    let row_lower = &mut *buffers.row_lower;
    let row_upper = &mut *buffers.row_upper;
    if layout.n_generic_rows == 0 {
        return;
    }

    // Split the geometry the resolver reads by concern: role (a) — the
    // stage-invariant state region — through the layout's borrowed `StateLayout`
    // handle; role (b) — the per-stage equipment ranges, block-stride constants,
    // FPHA/evap local maps, and the anticipated-decision base + reverse map —
    // straight from the `StageLayout` being filled. The view borrows; it does not
    // clone the layout.
    let geom = crate::generic_constraints::GenericResolverGeom {
        state: layout.state,
        turbine: &layout.turbine,
        spillage: &layout.spillage,
        diversion: &layout.diversion,
        thermal: &layout.thermal,
        line_fwd: &layout.line_fwd,
        line_rev: &layout.line_rev,
        excess: &layout.excess,
        generation: &layout.generation,
        deficit: &layout.deficit,
        max_deficit_segments: layout.max_deficit_segments,
        n_blks: layout.n_blks,
        evap_indices: &layout.evap_indices,
        evap_hydro_indices: &layout.evap_hydro_indices,
        fpha_hydro_indices: &layout.fpha_hydro_indices,
        anticipated_decision_start: layout.anticipated.col_anticipated_decision_start,
        anticipated_local_by_sys_pos: &layout.anticipated_local_by_sys_pos,
    };
    let positions = crate::generic_constraints::EntityPositionMaps {
        hydro: &ctx.hydro_pos,
        thermal: &ctx.thermal_pos,
        bus: &ctx.bus_pos,
        line: &ctx.line_pos,
    };
    // Cascade topology + diversion-into map for the HydroInflow total-inflow
    // arm; both already borrowed on the build context.
    let cascade_refs = crate::generic_constraints::CascadeRefs {
        cascade: ctx.cascade,
        diversion_upstream: &ctx.diversion_upstream,
    };
    // Pumping column start + station data for the PumpingFlow/PumpingPower arms.
    // `col_pumping_start` is the real reserved range on the `StageLayout` being
    // built — the sole owner of the pumping-flow column base.
    let pumping_refs = crate::generic_constraints::PumpingRefs {
        col_pumping_start: layout.col_pumping_start,
        pumping_stations: ctx.pumping_stations,
        pumping_pos: &ctx.pumping_pos,
    };

    for (entry_idx, entry) in layout.generic_constraint_rows.iter().enumerate() {
        let row = layout.row_generic_start + entry_idx;
        let constraint = &ctx.generic_constraints[entry.constraint_idx];
        // A collapsed stage-level row is priced by the stage's total hours
        // (it stands in for one row per block); a per-block row by its own
        // block's hours. The total equals `penalty * Σ block_hours * D` either
        // way, so the collapse is penalty-conserving.
        let block_hours = if entry.is_stage_level {
            stage.blocks.iter().map(|b| b.duration_hours).sum()
        } else {
            stage.blocks[entry.block_idx].duration_hours
        };

        // 1. Set row bounds from sense and RHS bound value.
        match entry.sense {
            ConstraintSense::LessEqual => {
                row_lower[row] = f64::NEG_INFINITY;
                row_upper[row] = entry.bound;
            }
            ConstraintSense::GreaterEqual => {
                row_lower[row] = entry.bound;
                row_upper[row] = f64::INFINITY;
            }
            ConstraintSense::Equal => {
                row_lower[row] = entry.bound;
                row_upper[row] = entry.bound;
            }
        }

        // 2. Fill CSC matrix entries for each expression term.
        for term in &constraint.expression.terms {
            let pairs = resolve_variable_ref(
                &term.variable,
                entry.block_idx,
                stage_idx,
                &geom,
                ctx.production_models,
                &positions,
                &cascade_refs,
                &pumping_refs,
            );
            for (col, multiplier) in pairs {
                let coef = match term.coefficient {
                    CoefficientRef::Literal(v) => v,
                    CoefficientRef::Parameter(param_id) => {
                        ctx.resolved.resolved_parameters.get(param_id, stage_idx)
                    }
                };
                col_entries[col].push((row, coef * term.scale * multiplier));
            }
        }

        // 3. Set slack column bounds and CSC entries when slack is enabled.
        if let Some(plus_col) = entry.slack_plus_col {
            let penalty = constraint.slack.penalty.unwrap_or(0.0);
            let obj_coeff = penalty * block_hours;

            // plus slack: [0, +INF), penalised in objective.
            // col_lower is already 0.0 from vec initialisation.
            col_upper[plus_col] = f64::INFINITY;
            objective[plus_col] = obj_coeff;

            // CSC entry for plus slack depends on sense.
            match entry.sense {
                ConstraintSense::LessEqual => {
                    // LHS - s_g <= bound  →  slack enters with -1.0
                    col_entries[plus_col].push((row, -1.0));
                }
                ConstraintSense::GreaterEqual => {
                    // LHS + s_g >= bound  →  slack enters with +1.0
                    col_entries[plus_col].push((row, 1.0));
                }
                ConstraintSense::Equal => {
                    // LHS + s_g_plus - s_g_minus == bound  →  plus slack with +1.0
                    col_entries[plus_col].push((row, 1.0));
                }
            }

            // minus slack: only for equality constraints.
            if let Some(minus_col) = entry.slack_minus_col {
                // col_lower is already 0.0 from vec initialisation.
                col_upper[minus_col] = f64::INFINITY;
                objective[minus_col] = obj_coeff;
                // LHS + s_g_plus - s_g_minus == bound  →  minus slack with -1.0
                col_entries[minus_col].push((row, -1.0));
            }
        }
    }
}

/// Fill NCS generation entries into the load balance constraint rows.
///
/// For each NCS `r` at block `k`, injects `+1.0` at the load balance row of the
/// connected bus, identical to thermal generation injection. Iterates ALL NCS
/// dense by SYSTEM index (the dense column position). The structural `+1.0` is
/// written for every NCS regardless of its commissioning window: a dormant NCS's
/// generation column is forced to `[0, 0]`, so the injection multiplies a zero
/// column and contributes nothing.
pub(super) fn fill_ncs_load_balance_entries(
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let grid = layout.block_grid();
    for (ncs_sys_idx, ncs) in ctx.non_controllable_sources.iter().enumerate() {
        let Some(&bus_idx) = ctx.bus_pos.get(&ncs.bus_id) else {
            // Unknown bus — should not happen with valid data, but defensive skip.
            continue;
        };
        for blk in 0..layout.n_blks {
            let col = grid.flat(layout.col_ncs_start, ncs_sys_idx, blk);
            let row = grid.flat(layout.row_load_balance_start(), bus_idx, blk);
            col_entries[col].push((row, 1.0));
        }
    }
}

/// Fill z-inflow definition constraint entries into `col_entries`.
///
/// For each hydro h, the z-inflow constraint is:
///   `z_h - sum_l[psi_l * lag_in[h,l]] = base_h + sigma_h * eta_h`
///
/// Matrix entries:
/// - Column `z_h`: coefficient `+1.0` in row `row_z_inflow_start + h`
/// - For each lag l with nonzero `psi_l`: column `inflow_lags.start + lag * n_h + h`
///   gets coefficient `-psi_l` in row `row_z_inflow_start + h`
///
/// Note: the lag column layout matches the LP builder convention (lag-major):
/// the column at `inflow_lags.start + lag * n_h + h` stores lag `l` of hydro `h`.
pub(super) fn fill_z_inflow_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_h = layout.n_h;
    let lag_order = layout.lag_order;
    let col_inflow_lags_start = layout.col_inflow_lags_start();

    for h_idx in 0..n_h {
        let row = layout.row_z_inflow_start() + h_idx;

        // z_h column: coefficient +1.0
        let col_z = layout.col_z_inflow_start() + h_idx;
        col_entries[col_z].push((row, 1.0));

        // Lag columns: coefficient -psi_l for each nonzero psi.
        // Uses lag-major layout (lag * n_h + h) matching the water-balance
        // AR dynamics entries in fill_state_and_water_entries.
        if ctx.par_lp.n_stages() > 0 && ctx.par_lp.n_hydros() == n_h {
            let psi = ctx.par_lp.psi_slice(stage_idx, h_idx);
            for (lag, &psi_val) in psi.iter().enumerate() {
                if psi_val != 0.0 && lag < lag_order {
                    let col = col_inflow_lags_start + lag * n_h + h_idx;
                    col_entries[col].push((row, -psi_val));
                }
            }
        }
    }
}

/// Fill CSC matrix entries for the 4 operational violation constraint families.
///
/// Each constraint links decision variables (turbine, spillage, diversion, generation)
/// to their respective slack columns via the constraint rows allocated in
/// [`StageLayout`].
///
/// - **Min outflow** (`>=`): `sum_blk[tau * (q + s + d)] + sigma_below >= RHS`
/// - **Max outflow** (`<=`): `sum_blk[tau * (q + s + d)] - sigma_above <= RHS`
/// - **Min turbine** (`>=`): `sum_blk[tau * q] + sigma_below >= RHS`
/// - **Min generation** (`>=`): `sum_blk[coeff * var * hours] + sigma_below >= RHS`
///   where `coeff * var` is `rho * q` for constant-productivity hydros or `g` for FPHA.
pub(super) fn fill_operational_violation_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;
    let grid = layout.block_grid();

    for (h_idx, fpha_local_entry) in layout.fpha_local_index.iter().enumerate() {
        // Min outflow (per block): q + s + d + sigma >= min_outflow_m3s
        for blk in 0..n_blks {
            let row = grid.flat(layout.row_min_outflow_start(), h_idx, blk);
            let col_q = layout.turbine_col(h_idx, blk);
            col_entries[col_q].push((row, 1.0));
            let col_s = layout.spillage_col(h_idx, blk);
            col_entries[col_s].push((row, 1.0));
            let col_d = layout.diversion_col(h_idx, blk);
            col_entries[col_d].push((row, 1.0));
            let col_slack = layout.outflow_below_col(h_idx, blk);
            col_entries[col_slack].push((row, 1.0));
        }

        // Max outflow (per block): q + s + d - sigma <= max_outflow_m3s
        for blk in 0..n_blks {
            let row = grid.flat(layout.row_max_outflow_start(), h_idx, blk);
            let col_q = layout.turbine_col(h_idx, blk);
            col_entries[col_q].push((row, 1.0));
            let col_s = layout.spillage_col(h_idx, blk);
            col_entries[col_s].push((row, 1.0));
            let col_d = layout.diversion_col(h_idx, blk);
            col_entries[col_d].push((row, 1.0));
            let col_slack = layout.outflow_above_col(h_idx, blk);
            col_entries[col_slack].push((row, -1.0));
        }

        // Min turbine flow (per block): q + sigma >= min_turbined_m3s
        for blk in 0..n_blks {
            let row = grid.flat(layout.row_min_turbine_start(), h_idx, blk);
            let col_q = layout.turbine_col(h_idx, blk);
            col_entries[col_q].push((row, 1.0));
            let col_slack = layout.turbine_below_col(h_idx, blk);
            col_entries[col_slack].push((row, 1.0));
        }

        // Min generation (per block): g + sigma >= min_generation_mw
        if let Some(&local_fpha_idx) = fpha_local_entry.as_ref() {
            // FPHA: generation variable g_{h,blk} (already in MW).
            for blk in 0..n_blks {
                let row = grid.flat(layout.row_min_generation_start(), h_idx, blk);
                let col_g = layout.generation_col(local_fpha_idx, blk);
                col_entries[col_g].push((row, 1.0));
                let col_slack = layout.generation_below_col(h_idx, blk);
                col_entries[col_slack].push((row, 1.0));
            }
        } else {
            // Constant productivity: gen_k = rho * q_k (MW).
            // Always read rho from the resolved per-stage production model.
            let rho = match ctx.production_models.model(h_idx, stage_idx) {
                ResolvedProductionModel::ConstantProductivity { productivity } => *productivity,
                ResolvedProductionModel::Fpha { .. } => {
                    unreachable!(
                        "Fpha resolved model in ConstantProductivity LP path for hydro \
                         {h_idx}; validate production model assignment upstream"
                    );
                }
            };
            for blk in 0..n_blks {
                let row = grid.flat(layout.row_min_generation_start(), h_idx, blk);
                let col_q = layout.turbine_col(h_idx, blk);
                col_entries[col_q].push((row, rho));
                let col_slack = layout.generation_below_col(h_idx, blk);
                col_entries[col_slack].push((row, 1.0));
            }
        }
    }
}

/// Build the unsorted CSC matrix entries for one stage.
///
/// Returns one `Vec<(row, value)>` per column. Entries are in insertion
/// order; the caller is responsible for sorting by row index before
/// assembling the final CSC arrays (see `build_single_stage_template`).
pub(super) fn build_stage_matrix_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
) -> Vec<Vec<(usize, f64)>> {
    let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];

    fill_state_and_water_entries(ctx, stage, stage_idx, layout, &mut col_entries);
    fill_filling_retention_entries(ctx, stage, layout, &mut col_entries);
    fill_filling_target_entries(layout, &mut col_entries);
    fill_filling_floor_entries(layout, &mut col_entries);
    fill_pumping_water_entries(ctx, stage, layout, &mut col_entries);
    fill_anticipated_state_out_def_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_load_balance_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_ncs_load_balance_entries(ctx, layout, &mut col_entries);
    fill_fpha_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_evaporation_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_z_inflow_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_operational_violation_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_anticipated_fishing_entries(ctx, stage, layout, &mut col_entries);

    col_entries
}

/// Assemble CSC arrays from per-column entry lists.
///
/// Returns `(col_starts, row_indices, values)` in the format required by
/// `SolverInterface::load_model`.
pub(super) fn assemble_csc(col_entries: &[Vec<(usize, f64)>]) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    // Contract: within each column the (row, _) entries are sorted by row index
    // (non-decreasing). The caller (`build_single_stage_template` CSC assembly)
    // owns the `sort_unstable_by_key(|&(row, _)| row)`; `assemble_csc` emits each
    // column's rows in iteration order and does NOT sort. Passing unsorted entries
    // produces CSC `row_indices` out of order within a column, which HiGHS/CLP may
    // silently misfactorize — the assert surfaces a missing caller-side sort rather
    // than masking it with an internal re-sort. debug-only: the release/hot path
    // pays no scan.
    debug_assert!(
        col_entries
            .iter()
            .all(|c| c.windows(2).all(|w| w[0].0 <= w[1].0)),
        "assemble_csc: each column's entries must be row-sorted (caller-owned sort)"
    );
    let num_cols = col_entries.len();
    let total_nz: usize = col_entries.iter().map(Vec::len).sum();
    let mut col_starts = Vec::with_capacity(num_cols + 1);
    let mut row_indices = Vec::with_capacity(total_nz);
    let mut values = Vec::with_capacity(total_nz);

    let mut offset: i32 = 0;
    for entries in col_entries {
        col_starts.push(offset);
        for &(row, val) in entries {
            // Rationale: the HiGHS/CLP C API requires i32 row indices and column
            // offsets. The stage LP row count is bounded by O(entities^2 * blocks);
            // for any realistic SDDP problem this is far below i32::MAX, so the
            // usize -> i32 cast cannot truncate or wrap.
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            row_indices.push(row as i32);
            values.push(val);
        }
        // Rationale: the running CSC offset (total nonzeros so far) is bounded by
        // the stage LP nonzero count, far below i32::MAX for any realistic SDDP
        // problem; the i32 offset the solver C API demands cannot overflow.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        {
            offset += entries.len() as i32;
        }
    }
    col_starts.push(offset);

    (col_starts, row_indices, values)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod assemble_csc_tests {
    use super::assemble_csc;

    /// Sorted columns assemble into the expected CSC triple without panicking.
    /// Columns: c0 = [(0, 1.0), (2, 2.0)], c1 = [], c2 = [(1, 3.0)].
    #[test]
    fn test_assemble_csc_sorted_columns_assemble_expected_csc() {
        let col_entries: Vec<Vec<(usize, f64)>> =
            vec![vec![(0, 1.0), (2, 2.0)], vec![], vec![(1, 3.0)]];

        let (col_starts, row_indices, values) = assemble_csc(&col_entries);

        // col_starts has num_cols + 1 entries; the prefix sum of per-column nnz.
        assert_eq!(col_starts, vec![0, 2, 2, 3]);
        assert_eq!(row_indices, vec![0, 2, 1]);
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    /// A column whose `(row, _)` pairs are out of order trips the `debug_assert`,
    /// proving the caller-owned-sort precondition is guarded in debug/test builds.
    #[test]
    #[should_panic(expected = "each column's entries must be row-sorted")]
    fn test_assemble_csc_unsorted_column_panics() {
        // Column 0 has rows 2 then 1 — strictly decreasing, violating the contract.
        let col_entries: Vec<Vec<(usize, f64)>> = vec![vec![(2, 1.0), (1, 2.0)]];

        let _ = assemble_csc(&col_entries);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests: parameter resolution in the LP builder
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::too_many_lines,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
mod parameter_resolution_tests {
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties, ConstraintExpression,
        ConstraintSense, ContractStageBounds, DeficitSegment, EntityId, GenericConstraint,
        HydroStageBounds, HydroStagePenalties, LineStageBounds, LineStagePenalties,
        NcsStagePenalties, ParameterKind, PenaltiesCountsSpec, PenaltiesDefaults,
        PumpingStageBounds, ResolvedBounds, ResolvedGenericConstraintBounds, ResolvedPenalties,
        ScalarParameter, SlackConfig, SystemBuilder, ThermalStageBounds,
    };
    use cobre_core::{LinearTerm, VariableRef};
    use cobre_stochastic::normal::precompute::PrecomputedNormal;
    use cobre_stochastic::par::precompute::PrecomputedPar;
    use std::collections::HashMap;

    use crate::energy_conversion::{EnergyConversionSet, build_hydro_energy_productivity_override};
    use crate::hydro_models::PrepareHydroModelsResult;
    use crate::inflow_method::InflowNonNegativityMethod;
    use crate::resolved_parameters::build_resolved_parameters;

    /// Return all CSC values stored at `(col, row)` in the template.
    fn csc_entries_at(t: &cobre_solver::StageTemplate, col: usize, row: usize) -> Vec<f64> {
        let start = t.col_starts[col] as usize;
        let end = t.col_starts[col + 1] as usize;
        t.row_indices[start..end]
            .iter()
            .zip(t.values[start..end].iter())
            .filter_map(|(&r, &v)| if r as usize == row { Some(v) } else { None })
            .collect()
    }

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            max_diversion_m3s: None,
            filling_inflow_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn default_hydro_penalties() -> HydroStagePenalties {
        HydroStagePenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    /// Build a one-bus, one-thermal system with `n_stages` stages and one
    /// generic constraint. Each stage has a single block of 744 hours.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn one_thermal_n_stages(
        n_stages: usize,
        thermal_entity_id: EntityId,
        constraints: Vec<GenericConstraint>,
        bounds: ResolvedGenericConstraintBounds,
    ) -> cobre_core::System {
        use chrono::NaiveDate;
        use cobre_core::entities::thermal::Thermal;
        use cobre_core::scenario::LoadModel;
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let thermal = Thermal {
            id: thermal_entity_id,
            name: "T1".to_string(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            anticipated_config: None,
            entry_stage_id: None,
            exit_stage_id: None,
        };

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: Some(0),
                blocks: vec![Block {
                    index: 0,
                    name: "BLK0".to_string(),
                    duration_hours: 744.0,
                }],
                block_mode: BlockMode::Parallel,
                state_config: StageStateConfig {
                    storage: false,
                    inflow_lags: false,
                },
                risk_config: StageRiskConfig::Expectation,
                scenario_config: ScenarioSourceConfig {
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
            })
            .collect();

        let load_models: Vec<_> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        let resolved_bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                    cost_per_mwh: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal])
            .stages(stages)
            .load_models(load_models)
            .bounds(resolved_bounds)
            .penalties(penalties)
            .generic_constraints(constraints)
            .resolved_generic_bounds(bounds)
            .build()
            .expect("one_thermal_n_stages: valid system")
    }

    /// Build templates for the given system using the supplied `ResolvedParameters`.
    fn make_templates(
        system: &cobre_core::System,
        resolved_params: &crate::resolved_parameters::ResolvedParameters,
    ) -> Vec<cobre_solver::StageTemplate> {
        let production = PrepareHydroModelsResult::default_from_system(system).production;
        let evaporation = PrepareHydroModelsResult::default_from_system(system).evaporation;
        crate::lp_builder::build_stage_templates(
            system,
            InflowNonNegativityMethod::None,
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &evaporation,
            resolved_params,
        )
        .expect("make_templates: valid")
        .templates
    }

    /// Build an empty `ResolvedParameters` table (no parameters) with a given
    /// stage-to-season mapping.
    fn empty_resolved_params(n_stages: usize) -> crate::resolved_parameters::ResolvedParameters {
        let stage_to_season: Vec<i32> = vec![0; n_stages];
        let ec = EnergyConversionSet::new(vec![], vec![], 0, n_stages);
        let override_table =
            build_hydro_energy_productivity_override(&[]).expect("empty override table");
        build_resolved_parameters(&[], &ec, &override_table, &[], &stage_to_season, n_stages)
            .expect("empty_resolved_params: valid")
    }

    /// Build a `ResolvedParameters` table containing a single `Constant` parameter.
    fn constant_param_resolved(
        param_id: EntityId,
        value: f64,
        n_stages: usize,
    ) -> crate::resolved_parameters::ResolvedParameters {
        let stage_to_season: Vec<i32> = vec![0; n_stages];
        let ec = EnergyConversionSet::new(vec![], vec![], 0, n_stages);
        let override_table =
            build_hydro_energy_productivity_override(&[]).expect("empty override table");
        let params = vec![ScalarParameter {
            id: param_id,
            name: format!("p{}", param_id.0),
            kind: ParameterKind::Constant { value },
        }];
        build_resolved_parameters(
            &params,
            &ec,
            &override_table,
            &[],
            &stage_to_season,
            n_stages,
        )
        .expect("constant_param_resolved: valid")
    }

    /// Build a `ResolvedParameters` table containing a single `PerStage` parameter.
    fn per_stage_param_resolved(
        param_id: EntityId,
        values: Vec<f64>,
    ) -> crate::resolved_parameters::ResolvedParameters {
        let n_stages = values.len();
        let stage_to_season: Vec<i32> = vec![0; n_stages];
        let ec = EnergyConversionSet::new(vec![], vec![], 0, n_stages);
        let override_table =
            build_hydro_energy_productivity_override(&[]).expect("empty override table");
        let params = vec![ScalarParameter {
            id: param_id,
            name: format!("p{}", param_id.0),
            kind: ParameterKind::PerStage { values },
        }];
        build_resolved_parameters(
            &params,
            &ec,
            &override_table,
            &[],
            &stage_to_season,
            n_stages,
        )
        .expect("per_stage_param_resolved: valid")
    }

    /// Make a generic constraint with a single `Parameter` term over a thermal.
    fn parameter_constraint(
        constraint_id: EntityId,
        param_id: EntityId,
        scale: f64,
        thermal_id: EntityId,
    ) -> GenericConstraint {
        GenericConstraint {
            id: constraint_id,
            name: format!("gc_{}", constraint_id.0),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::parameter(
                    param_id,
                    scale,
                    VariableRef::ThermalGeneration {
                        thermal_id,
                        block_id: None,
                    },
                )],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        }
    }

    /// Make a generic constraint with a single `Literal` term over a thermal.
    fn literal_constraint(
        constraint_id: EntityId,
        coef: f64,
        scale: f64,
        thermal_id: EntityId,
    ) -> GenericConstraint {
        GenericConstraint {
            id: constraint_id,
            name: format!("gc_lit_{}", constraint_id.0),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm {
                    coefficient: cobre_core::CoefficientRef::Literal(coef),
                    scale,
                    variable: VariableRef::ThermalGeneration {
                        thermal_id,
                        block_id: None,
                    },
                }],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        }
    }

    /// Build a `ResolvedGenericConstraintBounds` for a single constraint active
    /// at all `n_stages` stages (bound value `50.0`).
    fn bounds_for_n_stages(
        constraint_id: EntityId,
        n_stages: usize,
    ) -> ResolvedGenericConstraintBounds {
        let id_map: HashMap<i32, usize> = [(constraint_id.0, 0)].into_iter().collect();
        let rows: Vec<(i32, i32, Option<i32>, f64)> = (0..n_stages)
            .map(|s| (constraint_id.0, s as i32, None, 50.0_f64))
            .collect();
        ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter())
    }

    // ── Layout constants for 1-bus, 1-thermal, 1-block, 0-hydro ──────────────
    //
    // theta=col 0, decision_start=col 1
    // thermal col 0 block 0 → col 1
    // load_balance row 0 → generic constraint row at row 1
    const THERMAL_COL: usize = 1;
    const GENERIC_ROW: usize = 1;

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1: single stage, constant parameter, coefficient = param * scale * 1
    // ─────────────────────────────────────────────────────────────────────────

    /// A `CoefficientRef::Parameter` term is resolved against `ResolvedParameters`
    /// at LP-build time.
    ///
    /// Fixture: EntityId(7) → 3.0 (constant), scale 2.0.
    /// Expected CSC value: 3.0 * 2.0 * 1.0 = 6.0.
    #[test]
    fn parameter_coefficient_is_resolved_at_lp_build() {
        let param_id = EntityId(7);
        let thermal_id = EntityId(2);
        let constraint_id = EntityId(10);
        let scale = 2.0_f64;
        let param_value = 3.0_f64;

        let constraint = parameter_constraint(constraint_id, param_id, scale, thermal_id);
        let generic_bounds = bounds_for_n_stages(constraint_id, 1);

        let system = one_thermal_n_stages(1, thermal_id, vec![constraint], generic_bounds);
        let resolved = constant_param_resolved(param_id, param_value, 1);
        let templates = make_templates(&system, &resolved);

        let t = &templates[0];
        let entries = csc_entries_at(t, THERMAL_COL, GENERIC_ROW);
        assert!(
            !entries.is_empty(),
            "no CSC entry at (col={THERMAL_COL}, row={GENERIC_ROW}) for parameter term"
        );
        let total: f64 = entries.iter().sum();
        let expected = param_value * scale; // multiplier=1.0
        assert_eq!(
            total.to_bits(),
            expected.to_bits(),
            "expected {expected} (bit-exact), got {total}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2: three stages, per-stage parameter, each stage gets its own value
    // ─────────────────────────────────────────────────────────────────────────

    /// When the parameter has per-stage values, each stage template receives
    /// the stage-specific resolved coefficient.
    ///
    /// Fixture: EntityId(7) → [3.0, 7.0, 11.0], scale 2.0.
    /// Expected CSC values: stage 0 → 6.0, stage 1 → 14.0, stage 2 → 22.0.
    #[test]
    fn parameter_coefficient_per_stage_changes_lp() {
        let param_id = EntityId(7);
        let thermal_id = EntityId(2);
        let constraint_id = EntityId(10);
        let scale = 2.0_f64;
        let per_stage_values = vec![3.0_f64, 7.0, 11.0];
        let n_stages = per_stage_values.len();

        let constraint = parameter_constraint(constraint_id, param_id, scale, thermal_id);
        let generic_bounds = bounds_for_n_stages(constraint_id, n_stages);

        let system = one_thermal_n_stages(n_stages, thermal_id, vec![constraint], generic_bounds);
        let resolved = per_stage_param_resolved(param_id, per_stage_values.clone());
        let templates = make_templates(&system, &resolved);

        for (stage_idx, &param_val) in per_stage_values.iter().enumerate() {
            let t = &templates[stage_idx];
            let entries = csc_entries_at(t, THERMAL_COL, GENERIC_ROW);
            assert!(
                !entries.is_empty(),
                "no CSC entry at (col={THERMAL_COL}, row={GENERIC_ROW}) for stage {stage_idx}"
            );
            let total: f64 = entries.iter().sum();
            let expected = param_val * scale; // multiplier=1.0
            assert_eq!(
                total.to_bits(),
                expected.to_bits(),
                "stage {stage_idx}: expected {expected} (bit-exact), got {total}"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3: regression — Literal coefficients still work after wiring
    // ─────────────────────────────────────────────────────────────────────────

    /// `CoefficientRef::Literal(5.0)` with scale 3.0 must produce a CSC entry
    /// of exactly `5.0 * 3.0 * 1.0 = 15.0`, unchanged by the parameter-resolution
    /// wiring.
    #[test]
    fn literal_coefficient_still_works_after_wiring() {
        let thermal_id = EntityId(2);
        let constraint_id = EntityId(10);
        let coef = 5.0_f64;
        let scale = 3.0_f64;

        let constraint = literal_constraint(constraint_id, coef, scale, thermal_id);
        let generic_bounds = bounds_for_n_stages(constraint_id, 1);

        let system = one_thermal_n_stages(1, thermal_id, vec![constraint], generic_bounds);
        let resolved = empty_resolved_params(1);
        let templates = make_templates(&system, &resolved);

        let t = &templates[0];
        let entries = csc_entries_at(t, THERMAL_COL, GENERIC_ROW);
        assert!(
            !entries.is_empty(),
            "no CSC entry at (col={THERMAL_COL}, row={GENERIC_ROW}) for literal term"
        );
        let total: f64 = entries.iter().sum();
        let expected = coef * scale; // multiplier=1.0
        assert_eq!(
            total.to_bits(),
            expected.to_bits(),
            "expected {expected} (bit-exact), got {total}"
        );
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod zero_cost_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, CascadeTopology, ContractStageBounds, HydroStageBounds,
        LineStageBounds, PumpingStageBounds, ResolvedBounds, ResolvedExchangeFactors,
        ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds,
        ResolvedNcsFactors, ResolvedPenalties, Stage, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::columns::{ColumnBufs, fill_anticipated_columns, fill_thermal_columns};
    use super::super::layout::{ResolvedTables, StageLayout, TemplateBuildCtx};
    use super::super::rows::{fill_anticipated_fishing_rows, fill_anticipated_state_out_def_rows};
    use super::super::test_support::{state_layout_for, two_block_stage};
    use super::{
        build_stage_matrix_entries, fill_anticipated_fishing_entries,
        fill_anticipated_state_out_def_entries,
    };

    /// Owns data for a context with anticipated thermals and zero other entities.
    struct AntFixtures {
        par_lp: PrecomputedPar,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_exchange_factors: ResolvedExchangeFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
    }

    impl AntFixtures {
        /// Build a minimal `ResolvedBounds` with the given thermal count,
        /// `n_stages`, and `k_max` (all other entity counts zero).
        ///
        /// `n_stages` must exceed the queried `stage_idx` so the anticipated
        /// layout the fixture builds places every plant inside the study horizon
        /// `[0, n_stages)`. `k_max` sizes the thermal-bounds stage axis: it must
        /// be large enough that every delivery stage `stage_idx + K_i` the test
        /// accesses falls within `[0, n_stages + k_max)` (callers whose
        /// deliveries all fall strictly inside `[0, n_stages)` pass `0`).
        /// `n_thermals` sizes the thermal slot axis: tests that read or mutate
        /// `thermal_bounds(t_idx, …)` for an active anticipated plant must size it
        /// to cover every `t_idx` they touch; pure state-out/row tests pass `0`.
        fn bounds_with_n_stages(
            n_stages: usize,
            k_max: usize,
            n_thermals: usize,
        ) -> ResolvedBounds {
            ResolvedBounds::new(
                &BoundsCountsSpec {
                    n_hydros: 0,
                    n_thermals,
                    n_lines: 0,
                    n_pumping: 0,
                    n_contracts: 0,
                    n_stages,
                    k_max,
                },
                &BoundsDefaults {
                    hydro: HydroStageBounds {
                        min_storage_hm3: 0.0,
                        max_storage_hm3: 0.0,
                        min_turbined_m3s: 0.0,
                        max_turbined_m3s: 0.0,
                        min_outflow_m3s: 0.0,
                        max_outflow_m3s: None,
                        min_generation_mw: 0.0,
                        max_generation_mw: 0.0,
                        max_diversion_m3s: None,
                        filling_inflow_m3s: 0.0,
                        water_withdrawal_m3s: 0.0,
                    },
                    thermal: ThermalStageBounds {
                        min_generation_mw: 0.0,
                        max_generation_mw: 0.0,
                        cost_per_mwh: 0.0,
                    },
                    line: LineStageBounds {
                        direct_mw: 0.0,
                        reverse_mw: 0.0,
                    },
                    pumping: PumpingStageBounds {
                        min_flow_m3s: 0.0,
                        max_flow_m3s: 0.0,
                    },
                    contract: ContractStageBounds {
                        min_mw: 0.0,
                        max_mw: 0.0,
                        price_per_mwh: 0.0,
                    },
                },
            )
        }

        fn new() -> Self {
            Self {
                par_lp: PrecomputedPar::default(),
                cascade: CascadeTopology::build(&[]),
                bounds: ResolvedBounds::empty(),
                penalties: ResolvedPenalties::empty(),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_exchange_factors: ResolvedExchangeFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                },
                production_models: ProductionModelSet::new(vec![], 0, 1),
                evaporation_models: EvaporationModelSet::new(vec![]),
            }
        }

        fn make_ctx(
            &self,
            n_anticipated: usize,
            k_max: usize,
            anticipated_lead_stages: Vec<usize>,
            anticipated_thermal_indices: Vec<usize>,
            n_thermals: usize,
        ) -> TemplateBuildCtx<'_> {
            TemplateBuildCtx {
                hydros: &[],
                thermals: &[],
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_exchange_factors: &self.resolved_exchange_factors,
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
                diversion_upstream: HashMap::new(),
                n_hydros: 0,
                n_thermals,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated,
                k_max,
                anticipated_lead_stages,
                anticipated_thermal_indices,
                // Windowless: one `(None, None)` per plant, so the decision gate
                // reduces to the strict horizon clause. `study_stage_ids` lists the
                // study-stage ids so the in-range delivery lookup is safe.
                anticipated_windows: vec![(None, None); n_anticipated],
                study_stage_ids: (0..i32::try_from(self.bounds.n_stages()).unwrap_or(0)).collect(),
                has_penalty: false,
                // Sized to cover every active plant's delivery stage
                // (`stage_idx + K_i < n_stages`); `fill_anticipated_columns`
                // indexes these by delivery stage when pricing the decision column.
                cumulative_discount_factors: vec![1.0; self.bounds.n_stages() + k_max],
                total_hours_per_stage: vec![744.0; self.bounds.n_stages() + k_max],
            }
        }
    }

    /// `fill_thermal_columns` skips the per-block objective for anticipated
    /// plants (leaving the `0.0` vec default) while still writing it for
    /// non-anticipated thermals — the order-independent replacement for the
    /// former write-then-zero pass. The anticipated plant's per-block **bounds**
    /// are still written.
    ///
    /// Fixture: two thermals, thermal 0 anticipated (`K=1`), thermal 1 standard,
    /// both with non-zero resolved `cost_per_mwh` so the skip is observable
    /// (a regression that priced anticipated plants would leave a non-zero
    /// objective on thermal 0).
    #[test]
    fn fill_thermal_columns_skips_objective_for_anticipated_plants() {
        use cobre_core::entities::thermal::AnticipatedConfig;
        use cobre_core::{EntityId, Thermal};

        const ANT_COST: f64 = 30.0;
        const STD_COST: f64 = 40.0;

        let thermals = vec![
            Thermal {
                id: EntityId(1),
                name: "T_ant".to_string(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: ANT_COST,
                anticipated_config: Some(AnticipatedConfig { lead_stages: 1 }),
                entry_stage_id: None,
                exit_stage_id: None,
            },
            Thermal {
                id: EntityId(2),
                name: "T_std".to_string(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: STD_COST,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
            },
        ];

        let mut fixtures = AntFixtures::new();
        // 10 stages, k_max=1; seed each thermal's resolved per-stage cost so the
        // objective write (when not skipped) is non-zero.
        fixtures.bounds = AntFixtures::bounds_with_n_stages(10, 1, 2);
        for stage in 0..10 {
            fixtures.bounds.thermal_bounds_mut(0, stage).cost_per_mwh = ANT_COST;
            fixtures
                .bounds
                .thermal_bounds_mut(0, stage)
                .max_generation_mw = 100.0;
            fixtures.bounds.thermal_bounds_mut(1, stage).cost_per_mwh = STD_COST;
            fixtures
                .bounds
                .thermal_bounds_mut(1, stage)
                .max_generation_mw = 100.0;
        }
        let mut ctx = fixtures.make_ctx(
            1,       // n_anticipated
            1,       // k_max
            vec![1], // anticipated_lead_stages: K_0 = 1
            vec![0], // anticipated_thermal_indices: thermal 0 is anticipated
            2,       // n_thermals
        );
        ctx.thermals = &thermals;

        let stage = two_block_stage(2, [372.0, 372.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 2);

        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };

        fill_thermal_columns(&ctx, &stage, 2, &layout, &mut bufs);

        let n_blks = layout.n_blks;
        // Anticipated thermal 0: objective skipped (stays 0.0), bounds still set.
        // Thermal 0's block columns start at col_thermal_start (t_idx 0 offset).
        for blk in 0..n_blks {
            let col = layout.col_thermal_start() + blk;
            assert_eq!(
                bufs.objective[col], 0.0,
                "anticipated thermal 0 objective must stay 0.0 at col {col}",
            );
            assert_eq!(
                bufs.col_upper[col], 100.0,
                "anticipated thermal 0 bounds must still be written at col {col}",
            );
        }
        // Standard thermal 1: objective priced as cost * block_hours.
        for blk in 0..n_blks {
            let col = layout.col_thermal_start() + n_blks + blk;
            let expected = STD_COST * stage.blocks[blk].duration_hours;
            assert_eq!(
                bufs.objective[col], expected,
                "standard thermal 1 objective must be priced at col {col}",
            );
        }
    }

    /// Fills rows for all anticipated plants (always-active predicate).
    /// At stage 2 with `K_i=[1, 5]` and `n_anticipated=2`: both plants are active,
    /// so exactly two fishing rows are written.
    #[test]
    fn fishing_rows_fill_all_plants() {
        let mut fixtures = AntFixtures::new();
        fixtures.bounds = AntFixtures::bounds_with_n_stages(10, 0, 0);
        let ctx = fixtures.make_ctx(2, 5, vec![1, 5], vec![0, 1], 2);
        let stage = two_block_stage(2, [372.0, 372.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 2);

        // Always-active: both plants active at stage 2 → two fishing rows.
        assert_eq!(
            layout.anticipated.n_anticipated_fishing_rows, 2,
            "expected n_anticipated_fishing_rows == 2, got {}",
            layout.anticipated.n_anticipated_fishing_rows
        );

        let mut row_lower = vec![f64::NAN; layout.num_rows];
        let mut row_upper = vec![f64::NAN; layout.num_rows];

        fill_anticipated_fishing_rows(&ctx, &layout, &mut row_lower, &mut row_upper);

        // Both plants write a row with (0.0, 0.0) bounds.
        for local_idx in 0..layout.anticipated.n_anticipated_fishing_rows {
            let row = layout.anticipated.row_anticipated_fishing_start + local_idx;
            assert_eq!(
                row_lower[row], 0.0,
                "row_lower[{row}] (local_idx={local_idx}) expected 0.0, got {}",
                row_lower[row]
            );
            assert_eq!(
                row_upper[row], 0.0,
                "row_upper[{row}] (local_idx={local_idx}) expected 0.0, got {}",
                row_upper[row]
            );
        }
    }

    /// Always-active at `stage_idx = 0`: with `K = [1, 5]` and `n_anticipated = 2`,
    /// both plants are active even before their lead time elapses.
    /// Asserts `layout.anticipated.n_anticipated_fishing_rows == 2`, that both rows
    /// are filled with `(0.0, 0.0)` bounds, and that the anticipated-state
    /// slot-0 column carries the `-block_hours_total` coupling for both plants.
    #[test]
    fn fishing_rows_always_active_stage_zero() {
        let mut fixtures = AntFixtures::new();
        fixtures.bounds = AntFixtures::bounds_with_n_stages(10, 0, 0);
        let ctx = fixtures.make_ctx(2, 5, vec![1, 5], vec![0, 1], 2);
        let stage = two_block_stage(0, [372.0, 372.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        // Always-active: both plants are active at stage 0 → two fishing rows.
        assert_eq!(
            layout.anticipated.n_anticipated_fishing_rows, 2,
            "expected n_anticipated_fishing_rows == 2 at stage 0, got {}",
            layout.anticipated.n_anticipated_fishing_rows
        );

        let mut row_lower = vec![f64::NAN; layout.num_rows];
        let mut row_upper = vec![f64::NAN; layout.num_rows];

        fill_anticipated_fishing_rows(&ctx, &layout, &mut row_lower, &mut row_upper);

        // Both plants write equality rows with (0.0, 0.0) bounds.
        for local_idx in 0..layout.anticipated.n_anticipated_fishing_rows {
            let row = layout.anticipated.row_anticipated_fishing_start + local_idx;
            assert_eq!(
                row_lower[row], 0.0,
                "row_lower[{row}] (local_idx={local_idx}) expected 0.0, got {}",
                row_lower[row]
            );
            assert_eq!(
                row_upper[row], 0.0,
                "row_upper[{row}] (local_idx={local_idx}) expected 0.0, got {}",
                row_upper[row]
            );
        }

        // CSC coupling: anticipated_state slot-0 column carries (row, -block_hours_total)
        // for each plant under the always-active predicate.
        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_anticipated_fishing_entries(&ctx, &stage, &layout, &mut col_entries);

        let block_hours_total: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();
        let expected_neg = -block_hours_total;
        for local_idx in 0..layout.anticipated.n_anticipated_fishing_rows {
            let row = layout.anticipated.row_anticipated_fishing_start + local_idx;
            let col_state = layout.col_anticipated_state_start() + local_idx;
            let state_couplings: Vec<&(usize, f64)> = col_entries[col_state]
                .iter()
                .filter(|(r, _)| *r == row)
                .collect();
            assert_eq!(
                state_couplings.len(),
                1,
                "anticipated_state col {col_state} must carry exactly 1 coupling \
                 at fishing row {row} (plant local_idx={local_idx})"
            );
            let (_, coeff) = state_couplings[0];
            assert!(
                (coeff - expected_neg).abs() < 1e-12,
                "anticipated_state col {col_state} fishing-row coupling: \
                 expected {expected_neg}, got {coeff} (plant local_idx={local_idx})"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fixture helper for anticipated state-out tests
    // ─────────────────────────────────────────────────────────────────────────

    /// Build a minimal fixture for anticipated state-out tests:
    /// `n_anticipated = 2`, `K = [2, 3]`, `n_stages = 6`.
    ///
    /// `ResolvedBounds` is constructed with the correct `n_stages` so that
    /// `ctx.resolved.bounds.n_stages()` returns 6, which is required by
    /// `fill_anticipated_columns`, `fill_anticipated_state_out_def_rows`,
    /// and `fill_anticipated_state_out_def_entries`.
    fn build_anticipated_ctx_n_stages_6() -> (AntFixtures, Stage) {
        let mut fixtures = AntFixtures::new();
        // Override bounds with a 6-stage, 2-thermal table; k_max = 3 matches the
        // anticipated config the state-out tests pass to `make_ctx`. The two
        // thermal slots back the active anticipated plants' delivery-stage bound
        // reads in `fill_anticipated_columns`; the row/def tests sharing this
        // fixture leave them unread.
        fixtures.bounds = AntFixtures::bounds_with_n_stages(6, 3, 2);
        let stage = two_block_stage(0, [372.0, 372.0]);
        (fixtures, stage)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tests for the anticipated state-out columns (fill_anticipated_columns)
    // ─────────────────────────────────────────────────────────────────────────

    /// Active plants (`stage_idx + K_i < n_stages`) get `[-INF, +INF]` bounds;
    /// inactive plants (`stage_idx + K_i >= n_stages`) get `[0, 0]` bounds.
    ///
    /// Fixture: `n_anticipated=2`, `K=[2, 3]`, `n_stages=6`.
    /// Stage 0: both plants active  (0+2=2 < 6, 0+3=3 < 6) → `[-INF, +INF]`.
    /// Stage 5: both plants inactive (5+2=7 >= 6, 5+3=8 >= 6) → `[0, 0]`.
    #[test]
    fn test_fill_anticipated_columns_state_out_active_and_inactive() {
        let (fixtures, _) = build_anticipated_ctx_n_stages_6();
        let ctx = fixtures.make_ctx(
            2,          // n_anticipated
            3,          // k_max
            vec![2, 3], // anticipated_lead_stages: K=[2,3]
            vec![0, 1], // anticipated_thermal_indices
            0,          // n_thermals
        );

        // Stage 0: both plants active.
        let stage0 = two_block_stage(0, [372.0, 372.0]);
        let state = state_layout_for(&ctx);
        let layout0 = StageLayout::new(&ctx, &state, &stage0, 0);
        let mut col_lower = vec![0.0_f64; layout0.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout0.num_cols];
        let mut objective = vec![0.0_f64; layout0.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_anticipated_columns(&ctx, 0, &layout0, &mut bufs);
        for i in 0..2 {
            let col = layout0.anticipated.col_anticipated_state_out_start + i;
            assert_eq!(
                col_lower[col],
                f64::NEG_INFINITY,
                "stage 0, plant {i}: col_lower expected -INF, got {}",
                col_lower[col]
            );
            assert_eq!(
                col_upper[col],
                f64::INFINITY,
                "stage 0, plant {i}: col_upper expected +INF, got {}",
                col_upper[col]
            );
        }

        // Stage 5: both plants inactive.
        let stage5 = two_block_stage(5, [372.0, 372.0]);
        let state = state_layout_for(&ctx);
        let layout5 = StageLayout::new(&ctx, &state, &stage5, 5);
        let mut col_lower5 = vec![0.0_f64; layout5.num_cols];
        let mut col_upper5 = vec![f64::INFINITY; layout5.num_cols];
        let mut objective5 = vec![0.0_f64; layout5.num_cols];
        let mut bufs5 = ColumnBufs {
            col_lower: &mut col_lower5,
            col_upper: &mut col_upper5,
            objective: &mut objective5,
        };
        fill_anticipated_columns(&ctx, 5, &layout5, &mut bufs5);
        assert_eq!(
            layout5.anticipated.n_anticipated_state_out_def_rows, 0,
            "stage 5 inactive: expected no def rows, got {}",
            layout5.anticipated.n_anticipated_state_out_def_rows,
        );
        for i in 0..2 {
            let col = layout5.anticipated.col_anticipated_state_out_start + i;
            assert_eq!(
                col_lower5[col], 0.0,
                "stage 5, plant {i}: col_lower expected 0.0, got {}",
                col_lower5[col]
            );
            assert_eq!(
                col_upper5[col], 0.0,
                "stage 5, plant {i}: col_upper expected 0.0, got {}",
                col_upper5[col]
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tests for fill_anticipated_state_out_def_rows
    // ─────────────────────────────────────────────────────────────────────────

    /// At stage 0 with `K=[2,3]` and `n_stages=6`, both plants are active
    /// (0+2 < 6, 0+3 < 6), so `n_anticipated_state_out_def_rows == 2` and
    /// both definition rows must have equality bounds `[0.0, 0.0]`.
    #[test]
    fn test_fill_anticipated_state_out_def_rows_two_active_plants() {
        let (fixtures, stage) = build_anticipated_ctx_n_stages_6();
        let ctx = fixtures.make_ctx(2, 3, vec![2, 3], vec![0, 1], 0);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        assert_eq!(
            layout.anticipated.n_anticipated_state_out_def_rows, 2,
            "expected n_anticipated_state_out_def_rows == 2, got {}",
            layout.anticipated.n_anticipated_state_out_def_rows
        );

        let mut row_lower = vec![f64::NEG_INFINITY; layout.num_rows];
        let mut row_upper = vec![f64::INFINITY; layout.num_rows];
        fill_anticipated_state_out_def_rows(&ctx, 0, &layout, &mut row_lower, &mut row_upper);

        for k in 0..2 {
            let row = layout.anticipated.row_anticipated_state_out_def_start + k;
            assert_eq!(
                row_lower[row], 0.0,
                "def row {k}: row_lower expected 0.0, got {}",
                row_lower[row]
            );
            assert_eq!(
                row_upper[row], 0.0,
                "def row {k}: row_upper expected 0.0, got {}",
                row_upper[row]
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tests for fill_anticipated_state_out_def_entries
    // ─────────────────────────────────────────────────────────────────────────

    /// At stage 0 with `K=[2,3]` and `n_stages=6`, both plants are active.
    /// For each active plant `i`, the CSC entry list must contain:
    /// - `(def_row_i, +1.0)` on `col_anticipated_state_out_start + i`
    /// - `(def_row_i, -1.0)` on `col_anticipated_decision_start + i`
    #[test]
    fn test_fill_anticipated_state_out_def_entries_two_active_plants() {
        let (fixtures, stage) = build_anticipated_ctx_n_stages_6();
        let ctx = fixtures.make_ctx(2, 3, vec![2, 3], vec![0, 1], 0);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_anticipated_state_out_def_entries(&ctx, 0, &layout, &mut col_entries);

        for k in 0..2 {
            let row = layout.anticipated.row_anticipated_state_out_def_start + k;
            let col_state_out = layout.anticipated.col_anticipated_state_out_start + k;
            let col_decision = layout.anticipated.col_anticipated_decision_start + k;

            assert!(
                col_entries[col_state_out]
                    .iter()
                    .any(|&(r, v)| r == row && (v - 1.0).abs() < 1e-15),
                "plant {k}: expected (+1.0) entry at (col_state_out={col_state_out}, row={row}), \
                 got {:?}",
                col_entries[col_state_out]
            );
            assert!(
                col_entries[col_decision]
                    .iter()
                    .any(|&(r, v)| r == row && (v + 1.0).abs() < 1e-15),
                "plant {k}: expected (-1.0) entry at (col_decision={col_decision}, row={row}), \
                 got {:?}",
                col_entries[col_decision]
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // State-fixing diagonals must be absent from the CSC output
    // ─────────────────────────────────────────────────────────────────────────

    /// Asserts `build_stage_matrix_entries` produces no state-fixing
    /// diagonals in the CSC output.
    ///
    /// Coverage strategy: storage-fixing and lag-fixing diagonals are
    /// guaranteed absent by structural deletion of their for-loops in
    /// `fill_state_and_water_entries` (verified by C1+C2 grep — the
    /// functions/loops emitting those entries no longer exist in the
    /// source). Anticipated-state-fixing diagonals are checked dynamically:
    /// the test builds a fixture with `n_anticipated = 2, k_max = 3` and
    /// asserts every `(slot, plant)` column at
    /// `col_anticipated_state_start + slot*A + plant` has no entry at
    /// row `slot*A + plant` (the diagonal entry that existed in the
    /// pre-cutover layout, before state pinning moved to column bounds).
    ///
    /// The storage/lag assertions are included as zero-iteration loops in
    /// this fixture (`n_hydros` = 0) so the test documents the intent and
    /// would catch a future regression in any fixture that adds hydros.
    #[test]
    fn state_fixing_diagonals_absent_from_csc() {
        let (fixtures, stage) = build_anticipated_ctx_n_stages_6();
        let ctx = fixtures.make_ctx(
            2,          // n_anticipated
            3,          // k_max
            vec![2, 3], // anticipated_lead_stages
            vec![0, 1], // anticipated_thermal_indices
            2,          // n_thermals: must cover thermal indices 0 and 1 so the
                        // fishing-row entry resolves to a real thermal column.
        );

        let state = state_layout_for(&ctx);

        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let col_entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);

        let a = ctx.n_anticipated;
        let k = ctx.k_max;
        for slot in 0..k {
            for plant in 0..a {
                let col = layout.col_anticipated_state_start() + slot * a + plant;
                let diag_row = slot * a + plant;
                let has_diag = col_entries[col]
                    .iter()
                    .any(|&(r, v)| r == diag_row && (v - 1.0).abs() < 1e-15);
                assert!(
                    !has_diag,
                    "anticipated-state-fixing diagonal (row={diag_row}, val=1.0) must be absent \
                     from col {col} (slot={slot}, plant={plant})"
                );
            }
        }

        // Storage-fixing and lag-fixing diagonal absence assertions. With
        // n_hydros = 0 in this fixture these loops execute zero iterations,
        // but the structure documents intent and the same assertion shape
        // catches a regression in any future fixture with non-zero hydros.
        let n_h = ctx.n_hydros;
        let lag_order = ctx.max_par_order;
        for h in 0..n_h {
            let col = layout.col_storage_in_start() + h;
            let has_diag = col_entries[col]
                .iter()
                .any(|&(r, v)| r == h && (v - 1.0).abs() < 1e-15);
            assert!(
                !has_diag,
                "storage-fixing diagonal (row={h}, val=1.0) must be absent from col {col}"
            );
        }
        for lag in 0..lag_order {
            for h in 0..n_h {
                let col = layout.col_inflow_lags_start() + lag * n_h + h;
                let diag_row = n_h + lag * n_h + h;
                let has_diag = col_entries[col]
                    .iter()
                    .any(|&(r, v)| r == diag_row && (v - 1.0).abs() < 1e-15);
                assert!(
                    !has_diag,
                    "lag-fixing diagonal (row={diag_row}, val=1.0) must be absent from col {col} \
                     (lag={lag}, h={h})"
                );
            }
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]
mod pumping_water_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties, CascadeTopology, CoefficientRef,
        ConstraintExpression, ConstraintSense, ContractStageBounds, DeficitSegment, EntityId,
        GenericConstraint, Hydro, HydroGenerationModel, HydroStageBounds, HydroStagePenalties,
        Line, LineStageBounds, LineStagePenalties, LinearTerm, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingStageBounds, PumpingStation, ResolvedBounds,
        ResolvedExchangeFactors, ResolvedGenericConstraintBounds, ResolvedLoadFactors,
        ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties, SlackConfig, Thermal,
        ThermalStageBounds, VariableRef,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{EvaporationModel, EvaporationModelSet, ProductionModelSet};
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::M3S_TO_HM3;
    use super::super::columns::{ColumnBufs, fill_pumping_columns};
    use super::super::layout::{ResolvedTables, StageLayout, TemplateBuildCtx};
    use super::super::test_support::{state_layout_for, two_block_stage, zero_hydro_penalties};
    use super::{
        LpMatrixBuffers, assemble_csc, build_stage_matrix_entries, fill_generic_constraint_entries,
        fill_load_balance_entries, fill_pumping_water_entries,
    };

    const N_STAGES: usize = 1;

    /// Minimal independent (no-downstream) constant-productivity hydro.
    fn fixture_hydro(id: i32) -> Hydro {
        fixture_hydro_ds(id, None)
    }

    /// `fixture_hydro` with a caller-chosen `downstream_id`, so a two-reservoir
    /// cascade can be built from the fixture hydros. `fixture_hydro` delegates
    /// here with `None`, keeping every existing caller's cascade empty.
    fn fixture_hydro_ds(id: i32, downstream_id: Option<i32>) -> Hydro {
        Hydro {
            id: EntityId(id),
            name: format!("H{id}"),
            bus_id: EntityId(1),
            downstream_id: downstream_id.map(EntityId),
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 45.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_hydro_penalties(),
        }
    }

    fn default_bounds_defaults() -> BoundsDefaults {
        BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 100.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 50.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 45.0,
                max_diversion_m3s: None,
                filling_inflow_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        }
    }

    /// A single bus with one unbounded deficit segment, on `EntityId(1)` (the bus
    /// the fixture hydros and `station` helper reference).
    fn fixture_bus(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("B{id}"),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 1.0,
        }
    }

    /// A pumping station on `EntityId(1)` with explicit source/destination/flow
    /// data and the default `consumption_mw_per_m3s` (0.5).
    fn station(
        id: i32,
        source: i32,
        destination: i32,
        min_flow: f64,
        max_flow: f64,
    ) -> PumpingStation {
        station_full(id, source, destination, min_flow, max_flow, 1, 0.5)
    }

    /// A pumping station with an explicit `bus_id` and consumption rate so the
    /// power-coupling tests can place a station on an unmapped bus and observe a
    /// distinct coefficient.
    fn station_full(
        id: i32,
        source: i32,
        destination: i32,
        min_flow: f64,
        max_flow: f64,
        bus_id: i32,
        consumption_mw_per_m3s: f64,
    ) -> PumpingStation {
        PumpingStation {
            id: EntityId(id),
            name: format!("P{id}"),
            bus_id: EntityId(bus_id),
            source_hydro_id: EntityId(source),
            destination_hydro_id: EntityId(destination),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s,
            min_flow_m3s: min_flow,
            max_flow_m3s: max_flow,
        }
    }

    /// A bus with a caller-chosen deficit-segment count and excess cost so the
    /// multi-entity permutation test can give each bus a distinct CSC footprint:
    /// `fill_load_balance_entries` emits one deficit column per
    /// `deficit_segments.len()`, so distinct segment counts make the deficit
    /// column block bus-position-dependent — a bus permutation that escaped the
    /// ID-sort would land the wrong segment count in the wrong column block.
    fn fixture_bus_with(id: i32, n_segments: usize, excess_cost: f64) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("B{id}"),
            deficit_segments: (0..n_segments)
                .map(|_| DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 1000.0,
                })
                .collect(),
            excess_cost,
        }
    }

    /// A thermal plant on `bus_id` with distinct generation bounds and cost so a
    /// permutation that mislabelled which thermal owns which column is observable.
    fn fixture_thermal(id: i32, bus_id: i32, min_gen: f64, max_gen: f64, cost: f64) -> Thermal {
        Thermal {
            id: EntityId(id),
            name: format!("T{id}"),
            bus_id: EntityId(bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: cost,
            min_generation_mw: min_gen,
            max_generation_mw: max_gen,
            anticipated_config: None,
        }
    }

    /// A transmission line between `source_bus`/`target_bus` with distinct
    /// capacities. Distinct bus pairs across lines make the load-balance fill
    /// (±1.0 into source/target bus rows) line-position- and bus-position-
    /// dependent, so a permutation that escaped either ID-sort changes the CSC.
    fn fixture_line(id: i32, source_bus: i32, target_bus: i32, direct: f64, reverse: f64) -> Line {
        Line {
            id: EntityId(id),
            name: format!("L{id}"),
            source_bus_id: EntityId(source_bus),
            target_bus_id: EntityId(target_bus),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: direct,
            reverse_capacity_mw: reverse,
            losses_percent: 0.0,
            exchange_cost: 0.0,
        }
    }

    /// Owns the data backing a two-hydro `TemplateBuildCtx` carrying pumping
    /// stations, thermals, and lines. Every entity slice is stored in canonical
    /// (ID-sorted) order; the position maps are derived from those sorted slices,
    /// exactly as `SystemBuilder::build` produces them in production. The ID-sort
    /// is what makes two declaration orders of the same entities converge to one
    /// ctx — the property the permutation tests assert.
    struct PumpFixtures {
        hydros: Vec<Hydro>,
        stations: Vec<PumpingStation>,
        buses: Vec<Bus>,
        thermals: Vec<Thermal>,
        lines: Vec<Line>,
        hydro_pos: BTreeMap<EntityId, usize>,
        pumping_pos: BTreeMap<EntityId, usize>,
        bus_pos: BTreeMap<EntityId, usize>,
        thermal_pos: BTreeMap<EntityId, usize>,
        line_pos: BTreeMap<EntityId, usize>,
        par_lp: PrecomputedPar,
        /// AR order the ctx exposes as `max_par_order`. Zero by default (no
        /// inflow-lag columns reserved); raised by [`PumpFixtures::with_par_lp`]
        /// to match the injected `par_lp` so the AR-lag water term can fire.
        max_par_order: usize,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_exchange_factors: ResolvedExchangeFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        /// Generic constraints whose expressions the LP builder resolves. Empty by
        /// default; the end-to-end test sets a pumping-referencing constraint so the
        /// `PumpingFlow`/`PumpingPower` resolver arms run through the real caller.
        generic_constraints: Vec<GenericConstraint>,
    }

    impl PumpFixtures {
        /// Build a fixture with a single bus (`EntityId(1)`) — the bus the fixture
        /// hydros and the `station` helper reference — so the load-balance row is
        /// present and the pumping-power coupling is exercised.
        fn new(hydros: Vec<Hydro>, stations: Vec<PumpingStation>) -> Self {
            Self::new_with_buses(hydros, stations, vec![fixture_bus(1)])
        }

        /// Build a fixture from hydros, stations, and buses supplied in arbitrary
        /// declaration order, with no thermals or lines. Delegates to
        /// [`PumpFixtures::new_full`], which sorts every slice by `id.0` (the
        /// canonical operation `SystemBuilder::build` performs) before deriving
        /// position maps, so the resulting ctx is declaration-order-invariant.
        fn new_with_buses(
            hydros: Vec<Hydro>,
            stations: Vec<PumpingStation>,
            buses: Vec<Bus>,
        ) -> Self {
            Self::new_full(hydros, stations, buses, Vec::new(), Vec::new())
        }

        /// Build a fixture carrying hydros, stations, buses, thermals, and lines in
        /// arbitrary declaration order. Every slice is sorted by `id.0` before
        /// deriving its position map and bounds table, mirroring
        /// `SystemBuilder::build`; this canonicalisation is what makes two
        /// declaration orders of the same entities converge to one ctx (the
        /// invariant the permutation tests assert). Thermal and line bounds are
        /// written per-entity from the sorted slices so a column/bound mislabel
        /// under permutation is observable.
        fn new_full(
            mut hydros: Vec<Hydro>,
            mut stations: Vec<PumpingStation>,
            mut buses: Vec<Bus>,
            mut thermals: Vec<Thermal>,
            mut lines: Vec<Line>,
        ) -> Self {
            hydros.sort_by_key(|h| h.id.0);
            stations.sort_by_key(|s| s.id.0);
            buses.sort_by_key(|b| b.id.0);
            thermals.sort_by_key(|t| t.id.0);
            lines.sort_by_key(|l| l.id.0);

            let hydro_pos: BTreeMap<EntityId, usize> =
                hydros.iter().enumerate().map(|(i, h)| (h.id, i)).collect();
            let pumping_pos: BTreeMap<EntityId, usize> = stations
                .iter()
                .enumerate()
                .map(|(i, s)| (s.id, i))
                .collect();
            let bus_pos: BTreeMap<EntityId, usize> =
                buses.iter().enumerate().map(|(i, b)| (b.id, i)).collect();
            let thermal_pos: BTreeMap<EntityId, usize> = thermals
                .iter()
                .enumerate()
                .map(|(i, t)| (t.id, i))
                .collect();
            let line_pos: BTreeMap<EntityId, usize> =
                lines.iter().enumerate().map(|(i, l)| (l.id, i)).collect();

            let mut bounds = ResolvedBounds::new(
                &BoundsCountsSpec {
                    n_hydros: hydros.len(),
                    n_thermals: thermals.len(),
                    n_lines: lines.len(),
                    n_pumping: stations.len(),
                    n_contracts: 0,
                    n_stages: N_STAGES,
                    k_max: 0,
                },
                &default_bounds_defaults(),
            );
            // Distinct per-station bounds so a column/bound mismatch is observable.
            for (p_idx, s) in stations.iter().enumerate() {
                for stage_idx in 0..N_STAGES {
                    *bounds.pumping_bounds_mut(p_idx, stage_idx) = PumpingStageBounds {
                        min_flow_m3s: s.min_flow_m3s,
                        max_flow_m3s: s.max_flow_m3s,
                    };
                }
            }
            // Distinct per-thermal generation bounds and cost, taken from the
            // sorted slice, so a permutation that mislabelled which thermal owns
            // which column would change the resolved bounds.
            for (t_idx, t) in thermals.iter().enumerate() {
                for stage_idx in 0..N_STAGES {
                    *bounds.thermal_bounds_mut(t_idx, stage_idx) = ThermalStageBounds {
                        min_generation_mw: t.min_generation_mw,
                        max_generation_mw: t.max_generation_mw,
                        cost_per_mwh: t.cost_per_mwh,
                    };
                }
            }
            // Distinct per-line capacities from the sorted slice, same rationale.
            for (l_idx, l) in lines.iter().enumerate() {
                for stage_idx in 0..N_STAGES {
                    *bounds.line_bounds_mut(l_idx, stage_idx) = LineStageBounds {
                        direct_mw: l.direct_capacity_mw,
                        reverse_mw: l.reverse_capacity_mw,
                    };
                }
            }

            let production_models = ProductionModelSet::new(
                vec![
                    vec![
                        crate::hydro_models::ResolvedProductionModel::ConstantProductivity {
                            productivity: 1.0,
                        };
                        N_STAGES
                    ];
                    hydros.len()
                ],
                hydros.len(),
                N_STAGES,
            );
            let evaporation_models =
                EvaporationModelSet::new(vec![EvaporationModel::None; hydros.len()]);

            // Built from the sorted hydros so a fixture carrying `downstream_id`
            // produces a real cascade; output-neutral for the `fixture_hydro`
            // callers, whose `downstream_id: None` yields the same empty cascade
            // `CascadeTopology::build(&[])` did.
            let cascade = CascadeTopology::build(&hydros);

            Self {
                hydros,
                stations,
                buses,
                thermals,
                lines,
                hydro_pos,
                pumping_pos,
                bus_pos,
                thermal_pos,
                line_pos,
                par_lp: PrecomputedPar::default(),
                cascade,
                max_par_order: 0,
                bounds,
                penalties: ResolvedPenalties::empty(),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_exchange_factors: ResolvedExchangeFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                },
                production_models,
                evaporation_models,
                generic_constraints: Vec::new(),
            }
        }

        /// Attach a generic constraint (and its active-at-stage-0 bound) so the
        /// LP builder resolves the constraint's expression against the pumping
        /// columns. Used by the end-to-end resolver-integration test.
        fn with_generic_constraint(mut self, constraint: GenericConstraint, bound: f64) -> Self {
            let constraint_id = constraint.id.0;
            let id_map: HashMap<i32, usize> = [(constraint_id, 0)].into_iter().collect();
            let rows = (0..N_STAGES).map(|s| {
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                (constraint_id, s as i32, None, bound)
            });
            self.resolved_generic_bounds =
                ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());
            self.generic_constraints = vec![constraint];
            self
        }

        /// Inject a `PrecomputedPar` and align `max_par_order` to its order, so
        /// the inflow-lag columns are reserved and the `−ζ·ψ` AR-lag water term
        /// fires. The default fixture carries `PrecomputedPar::default()` (no
        /// `psi`, `max_par_order: 0`), under which the AR-lag term is dormant.
        fn with_par_lp(mut self, par_lp: PrecomputedPar) -> Self {
            self.max_par_order = par_lp.max_order();
            self.par_lp = par_lp;
            self
        }

        /// Replace the default empty penalty table with a properly-sized all-zero
        /// (plus a small spillage cost) `ResolvedPenalties` so the FULL template
        /// build (`build_single_stage_template`, which reads per-hydro penalties)
        /// does not index an empty table. The matrix-only tests use
        /// `ResolvedPenalties::empty()` because `build_stage_matrix_entries` never
        /// reads penalties; the solver-backed duals test needs this.
        fn with_resolved_penalties(mut self) -> Self {
            let hydro = HydroStagePenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 0.0,
            };
            self.penalties = ResolvedPenalties::new(
                &PenaltiesCountsSpec {
                    n_hydros: self.hydros.len(),
                    n_buses: self.buses.len(),
                    n_lines: self.lines.len(),
                    n_ncs: 0,
                    n_stages: N_STAGES,
                },
                &PenaltiesDefaults {
                    hydro,
                    bus: BusStagePenalties { excess_cost: 0.0 },
                    line: LineStagePenalties { exchange_cost: 0.0 },
                    ncs: NcsStagePenalties {
                        curtailment_cost: 0.0,
                    },
                },
            );
            self
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            TemplateBuildCtx {
                hydros: &self.hydros,
                thermals: &self.thermals,
                lines: &self.lines,
                buses: &self.buses,
                load_models: &[],
                cascade: &self.cascade,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_exchange_factors: &self.resolved_exchange_factors,
                    resolved_ncs_bounds: &self.resolved_ncs_bounds,
                    resolved_ncs_factors: &self.resolved_ncs_factors,
                    resolved_parameters: &self.resolved_parameters,
                },
                hydro_pos: self.hydro_pos.clone(),
                thermal_pos: self.thermal_pos.clone(),
                line_pos: self.line_pos.clone(),
                bus_pos: self.bus_pos.clone(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &self.generic_constraints,
                non_controllable_sources: &[],
                pumping_stations: &self.stations,
                pumping_pos: self.pumping_pos.clone(),
                n_pumping: self.stations.len(),
                diversion_upstream: HashMap::new(),
                n_hydros: self.hydros.len(),
                n_thermals: self.thermals.len(),
                n_lines: self.lines.len(),
                n_buses: self.buses.len(),
                max_par_order: self.max_par_order,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                study_stage_ids: vec![],
                has_penalty: false,
                cumulative_discount_factors: vec![1.0; N_STAGES],
                total_hours_per_stage: vec![744.0; N_STAGES],
            }
        }
    }

    /// Column bounds = `[min_flow, max_flow]` and zero objective for every
    /// `(station, block)` pumping column.
    #[test]
    fn pumping_columns_get_flow_bounds_and_zero_cost() {
        let stations = vec![station(10, 1, 2, 5.0, 80.0), station(20, 2, 1, 0.0, 30.0)];
        let fixtures = PumpFixtures::new(vec![fixture_hydro(1), fixture_hydro(2)], stations);
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        // Lower/upper start at a NaN sentinel so any column the helper fails to
        // bound is visible; objective starts at the production default (0.0), so
        // the post-fill zero assertion proves the helper writes no cost.
        let mut col_lower = vec![f64::NAN; layout.num_cols];
        let mut col_upper = vec![f64::NAN; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };

        fill_pumping_columns(&ctx, &stage, 0, &layout, &mut bufs);

        let n_blks = layout.n_blks;
        for (p_idx, s) in ctx.pumping_stations.iter().enumerate() {
            for blk in 0..n_blks {
                let col = layout.col_pumping_start + p_idx * n_blks + blk;
                assert_eq!(
                    bufs.col_lower[col], s.min_flow_m3s,
                    "station {p_idx} blk {blk}: lower bound must be min_flow"
                );
                assert_eq!(
                    bufs.col_upper[col], s.max_flow_m3s,
                    "station {p_idx} blk {blk}: upper bound must be max_flow"
                );
                assert_eq!(
                    bufs.objective[col], 0.0,
                    "station {p_idx} blk {blk}: objective must be zero"
                );
            }
        }
    }

    /// Source water row gains `+tau_h`, destination water row gains `−tau_h`,
    /// with `tau_h == block.duration_hours * M3S_TO_HM3` per block.
    #[test]
    fn pumping_water_entries_source_plus_tau_destination_minus_tau() {
        // Station id 10: source hydro id 1 (pos 0), destination hydro id 2 (pos 1).
        let fixtures = PumpFixtures::new(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station(10, 1, 2, 0.0, 50.0)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_pumping_water_entries(&ctx, &stage, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        let source_pos = ctx.hydro_pos[&EntityId(1)];
        let dest_pos = ctx.hydro_pos[&EntityId(2)];
        let row_source = layout.row_water_balance_start() + source_pos;
        let row_dest = layout.row_water_balance_start() + dest_pos;

        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col = layout.col_pumping_start + blk;
            assert_eq!(
                col_entries[col],
                vec![(row_source, tau_h), (row_dest, -tau_h)],
                "blk {blk}: source +tau_h then destination -tau_h"
            );
        }
    }

    /// A station whose `source_hydro_id` is absent from `hydro_pos` skips only
    /// the source entry — the destination side is still written, no panic.
    #[test]
    fn pumping_water_entries_missing_source_skips_only_source() {
        // Source hydro id 99 does NOT exist; destination hydro id 2 (pos 1) does.
        let fixtures = PumpFixtures::new(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station(10, 99, 2, 0.0, 50.0)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_pumping_water_entries(&ctx, &stage, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        let dest_pos = ctx.hydro_pos[&EntityId(2)];
        let row_dest = layout.row_water_balance_start() + dest_pos;
        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col = layout.col_pumping_start + blk;
            assert_eq!(
                col_entries[col],
                vec![(row_dest, -tau_h)],
                "blk {blk}: only the destination -tau_h entry survives"
            );
        }
    }

    /// A station whose `destination_hydro_id` is absent skips only the
    /// destination entry — the source side is still written, no panic.
    #[test]
    fn pumping_water_entries_missing_destination_skips_only_destination() {
        // Source hydro id 1 (pos 0) exists; destination hydro id 99 does NOT.
        let fixtures = PumpFixtures::new(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station(10, 1, 99, 0.0, 50.0)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_pumping_water_entries(&ctx, &stage, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        let source_pos = ctx.hydro_pos[&EntityId(1)];
        let row_source = layout.row_water_balance_start() + source_pos;
        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col = layout.col_pumping_start + blk;
            assert_eq!(
                col_entries[col],
                vec![(row_source, tau_h)],
                "blk {blk}: only the source +tau_h entry survives"
            );
        }
    }

    /// The pumping flow column enters its bus load-balance row with
    /// `−consumption_mw_per_m3s` per block — a negative injection, the same sign a
    /// line carries into its source bus, NOT the `+1.0` of generation.
    #[test]
    fn pumping_power_enters_bus_row_with_negative_consumption() {
        // Station id 10 on bus id 1 (pos 0), consumption 0.75 MW per m³/s.
        let fixtures = PumpFixtures::new_with_buses(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station_full(10, 1, 2, 0.0, 50.0, 1, 0.75)],
            vec![fixture_bus(1)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        let b_idx = ctx.bus_pos[&EntityId(1)];
        for blk in 0..n_blks {
            let row = layout.row_load_balance_start() + b_idx * n_blks + blk;
            let col = layout.col_pumping_start + blk;
            assert!(
                col_entries[col].contains(&(row, -0.75)),
                "blk {blk}: pumping column {col} must carry (row {row}, -0.75); got {:?}",
                col_entries[col]
            );
            // The flow column carries the bus-power coupling and nothing else from
            // the load-balance fill (no positive generation-style entry).
            assert_eq!(
                col_entries[col],
                vec![(row, -0.75)],
                "blk {blk}: pumping column must carry only the negative-injection entry"
            );
        }
    }

    /// A station whose `bus_id` is absent from `bus_pos` writes no load-balance
    /// entry and does not panic.
    #[test]
    fn pumping_power_missing_bus_skips_without_panic() {
        // Station on bus id 99, which is NOT among the fixture buses (only id 1).
        let fixtures = PumpFixtures::new_with_buses(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station_full(10, 1, 2, 0.0, 50.0, 99, 0.5)],
            vec![fixture_bus(1)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        for blk in 0..n_blks {
            let col = layout.col_pumping_start + blk;
            assert!(
                col_entries[col].is_empty(),
                "blk {blk}: station on an unmapped bus must write no load-balance entry"
            );
        }
    }

    /// With no pumping stations, `fill_load_balance_entries` produces exactly the
    /// same entries it would without the pumping loop — the pumping path is inert.
    /// A 1-bus, 2-hydro system with one declared station is the baseline; removing
    /// the station must leave every column's load-balance entries unchanged.
    #[test]
    fn no_pumping_stations_leaves_load_balance_entries_identical() {
        let build = |stations: Vec<PumpingStation>| {
            let fixtures = PumpFixtures::new_with_buses(
                vec![fixture_hydro(1), fixture_hydro(2)],
                stations,
                vec![fixture_bus(1)],
            );
            let ctx = fixtures.make_ctx();
            let stage = two_block_stage(0, [300.0, 444.0]);
            let state = state_layout_for(&ctx);
            let layout = StageLayout::new(&ctx, &state, &stage, 0);
            let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
            fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);
            // Truncate to the non-pumping column region: with zero stations the
            // layout has no pumping columns, so compare the shared prefix that both
            // layouts share (generation/thermal/line/deficit/excess columns are all
            // indexed before the pumping block).
            (layout.col_pumping_start, col_entries)
        };

        let (pump_start_empty, entries_empty) = build(vec![]);
        let (_pump_start_one, entries_one) = build(vec![station(10, 1, 2, 0.0, 50.0)]);

        // Every column before the pumping block must carry identical load-balance
        // entries whether or not a station is present.
        for col in 0..pump_start_empty {
            assert_eq!(
                entries_empty[col], entries_one[col],
                "load-balance entries for column {col} must be pumping-independent"
            );
        }
    }

    /// Build the full CSC for a 2-reservoir + 1-bus system twice with the hydro,
    /// station, and bus declarations supplied in two DIFFERENT input orders, and
    /// assert the assembled CSC arrays are byte-identical. Determinism is a hard
    /// rule: the canonical ID-sort plus the per-column row-sort must erase all
    /// trace of the input declaration order.
    ///
    /// Two stations with opposite source/destination orientation are declared so
    /// the assertion is load-bearing on the pumping path: a single station would
    /// pass even if the pumping iteration were declaration-order-dependent (there
    /// is nothing to scramble), whereas permuting two stations exercises the
    /// per-column row-sort that decouples declaration order from CSC layout. Both
    /// stations sit on the single bus, so the `−consumption_mw_per_m3s` bus-power
    /// entries are part of the assembled CSC and the assertion covers them too.
    #[test]
    fn csc_byte_identical_under_permuted_declaration_order() {
        let assemble = |hydros: Vec<Hydro>, stations: Vec<PumpingStation>| {
            let fixtures = PumpFixtures::new(hydros, stations);
            let ctx = fixtures.make_ctx();
            let stage = two_block_stage(0, [300.0, 444.0]);
            let state = state_layout_for(&ctx);
            let layout = StageLayout::new(&ctx, &state, &stage, 0);
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            // Mirror the production per-column row-sort (see build_single_stage_template).
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };

        // Two reservoirs (ids 1, 2) and two stations moving water in opposite
        // directions (10: 1 → 2, 20: 2 → 1), both on bus 1 with DISTINCT
        // consumption rates so a permutation that mislabels which station's
        // `−consumption_mw_per_m3s` lands on which pumping column would be caught.
        // Order A declares both ascending.
        let csc_a = assemble(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![
                station_full(10, 1, 2, 5.0, 80.0, 1, 0.4),
                station_full(20, 2, 1, 0.0, 30.0, 1, 0.9),
            ],
        );
        // Order B declares the identical entities with both hydros and both
        // stations reversed.
        let csc_b = assemble(
            vec![fixture_hydro(2), fixture_hydro(1)],
            vec![
                station_full(20, 2, 1, 0.0, 30.0, 1, 0.9),
                station_full(10, 1, 2, 5.0, 80.0, 1, 0.4),
            ],
        );

        assert_eq!(csc_a.0, csc_b.0, "col_starts must be byte-identical");
        assert_eq!(csc_a.1, csc_b.1, "row_indices must be byte-identical");
        assert_eq!(csc_a.2, csc_b.2, "values must be byte-identical");
    }

    /// Build the full stage CSC for a multi-entity system (2 hydros, 3 buses, 2
    /// thermals on different buses, 2 lines on different bus pairs, 1 generic
    /// constraint over a thermal/line/bus triple) twice with EVERY entity family's
    /// declaration order reversed between the two builds, and assert the assembled
    /// CSC arrays are byte-identical. This is the determinism backstop for the
    /// bus/line/thermal/generic-constraint families the single-bus
    /// `csc_byte_identical_under_permuted_declaration_order` cannot scramble: a
    /// map-iterating fill that reintroduced declaration-order dependence on any of
    /// these families would diverge here.
    ///
    /// The constraint references a `ThermalGeneration`, a `LineExchange`, and a
    /// `BusDeficit` term so the resolver path through
    /// `thermal_pos`/`line_pos`/`bus_pos` participates in the asserted CSC, not
    /// only the load-balance fill. Distinct per-entity attributes (thermals on
    /// different buses with distinct bounds, lines on different bus pairs with
    /// distinct capacities, buses with distinct deficit-segment counts) make the
    /// assertion load-bearing: a permutation that mislabelled which entity owns
    /// which slot would change the CSC.
    #[test]
    fn csc_byte_identical_under_permuted_multi_entity_order() {
        // Generic constraint id 7:
        //   thermal_gen(10) + line_exchange(100) + bus_deficit(2) <= 100
        // All referenced ids ARE present in the position maps, so the resolver
        // contributes real entries rather than silently returning an empty vec.
        let make_constraint = || GenericConstraint {
            id: EntityId(7),
            name: "gc_multi".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![
                    LinearTerm {
                        coefficient: CoefficientRef::Literal(1.0),
                        scale: 1.0,
                        variable: VariableRef::ThermalGeneration {
                            thermal_id: EntityId(10),
                            block_id: None,
                        },
                    },
                    LinearTerm {
                        coefficient: CoefficientRef::Literal(1.0),
                        scale: 1.0,
                        variable: VariableRef::LineExchange {
                            line_id: EntityId(100),
                            block_id: None,
                        },
                    },
                    LinearTerm {
                        coefficient: CoefficientRef::Literal(1.0),
                        scale: 1.0,
                        variable: VariableRef::BusDeficit {
                            bus_id: EntityId(2),
                            block_id: None,
                        },
                    },
                ],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };

        // Build the fixture, run the PRODUCTION assemble sequence
        // (build_stage_matrix_entries -> fill_generic_constraint_entries via
        // LpMatrixBuffers -> per-column row-sort -> assemble_csc, matching
        // build_single_stage_template), and return the CSC triple plus the layout
        // so the caller can probe the generic-constraint row.
        // Run the production assemble sequence into a CSC for a fixture. Takes the
        // fixture by reference so the caller can keep it alive and build a layout
        // from it for the offset reads — the per-call `StageLayout` borrows the
        // function-local ctx/state and cannot escape the closure.
        let assemble = |fixtures: &PumpFixtures| {
            let ctx = fixtures.make_ctx();
            let stage = two_block_stage(0, [300.0, 444.0]);
            let state = state_layout_for(&ctx);
            let layout = StageLayout::new(&ctx, &state, &stage, 0);

            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            let mut col_upper = vec![f64::INFINITY; layout.num_cols];
            let mut objective = vec![0.0_f64; layout.num_cols];
            let mut row_lower = vec![f64::NEG_INFINITY; layout.num_rows];
            let mut row_upper = vec![f64::INFINITY; layout.num_rows];
            let mut buffers = LpMatrixBuffers {
                col_entries: &mut entries,
                col_upper: &mut col_upper,
                objective: &mut objective,
                row_lower: &mut row_lower,
                row_upper: &mut row_upper,
            };
            fill_generic_constraint_entries(&ctx, &stage, 0, &layout, &mut buffers);

            // Mirror the production per-column row-sort (see
            // build_single_stage_template) before assembling the CSC.
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };

        // Order A: every family declared ascending. The fixture is kept alive so
        // the order-A layout (for the offset reads below) is built from it.
        let fixtures_a = PumpFixtures::new_full(
            vec![fixture_hydro(1), fixture_hydro(2)],
            Vec::new(),
            vec![
                fixture_bus_with(1, 1, 1.0),
                fixture_bus_with(2, 2, 3.0),
                fixture_bus_with(3, 1, 5.0),
            ],
            vec![
                fixture_thermal(10, 1, 0.0, 30.0, 12.0),
                fixture_thermal(20, 3, 5.0, 45.0, 27.0),
            ],
            vec![
                fixture_line(100, 1, 2, 50.0, 40.0),
                fixture_line(200, 2, 3, 70.0, 60.0),
            ],
        )
        .with_generic_constraint(make_constraint(), 100.0);
        let csc_a = assemble(&fixtures_a);

        // Order B: the identical entities, every family declared in reverse.
        let fixtures_b = PumpFixtures::new_full(
            vec![fixture_hydro(2), fixture_hydro(1)],
            Vec::new(),
            vec![
                fixture_bus_with(3, 1, 5.0),
                fixture_bus_with(2, 2, 3.0),
                fixture_bus_with(1, 1, 1.0),
            ],
            vec![
                fixture_thermal(20, 3, 5.0, 45.0, 27.0),
                fixture_thermal(10, 1, 0.0, 30.0, 12.0),
            ],
            vec![
                fixture_line(200, 2, 3, 70.0, 60.0),
                fixture_line(100, 1, 2, 50.0, 40.0),
            ],
        )
        .with_generic_constraint(make_constraint(), 100.0);
        let csc_b = assemble(&fixtures_b);

        // Order-A layout (held by the test, owning its ctx/state) for the offset
        // reads below. The layout offsets are declaration-order-invariant, so this
        // matches the layout order A's CSC was assembled with.
        let ctx_a = fixtures_a.make_ctx();
        let stage_a = two_block_stage(0, [300.0, 444.0]);
        let state_a = state_layout_for(&ctx_a);
        let layout_a = StageLayout::new(&ctx_a, &state_a, &stage_a, 0);

        assert_eq!(csc_a.0, csc_b.0, "col_starts must be byte-identical");
        assert_eq!(csc_a.1, csc_b.1, "row_indices must be byte-identical");
        assert_eq!(csc_a.2, csc_b.2, "values must be byte-identical");

        // Criterion 3: prove the generic-constraint resolver path ran (not just
        // the load-balance fill) by reading the generic row's coefficients on the
        // resolved thermal/line/deficit columns from order A's CSC. The expression
        // is block-dependent (ThermalGeneration/LineExchange/BusDeficit), so it
        // expands to one generic row per block; probe every block.
        let n_blks = layout_a.n_blks;
        assert_eq!(
            layout_a.n_generic_rows, n_blks,
            "block-dependent generic constraint must expand to one row per block"
        );
        let grid = layout_a.block_grid();
        let t_pos = 0; // thermal id 10 sorts to position 0.
        let l_pos = 0; // line id 100 sorts to position 0.
        let b_pos = 1; // bus id 2 sorts to position 1 (buses 1,2,3).
        for blk in 0..n_blks {
            let row = i32::try_from(layout_a.row_generic_start + blk).unwrap();
            let coeff_at = |col: usize| -> f64 {
                let start = usize::try_from(csc_a.0[col]).unwrap();
                let end = usize::try_from(csc_a.0[col + 1]).unwrap();
                csc_a.1[start..end]
                    .iter()
                    .zip(&csc_a.2[start..end])
                    .filter(|&(&r, _)| r == row)
                    .map(|(_, &v)| v)
                    .sum()
            };

            // ThermalGeneration(10): +1.0 on thermal 10's column.
            let thermal_col = grid.flat(layout_a.col_thermal_start(), t_pos, blk);
            assert_eq!(
                coeff_at(thermal_col),
                1.0,
                "blk {blk}: generic row must carry +1.0 on thermal 10's column \
                 (resolver path through thermal_pos)"
            );
            // LineExchange(100): +1.0 on the forward column, -1.0 on the reverse.
            assert_eq!(
                coeff_at(layout_a.line_fwd_col(l_pos, blk)),
                1.0,
                "blk {blk}: generic row must carry +1.0 on line 100's forward column \
                 (resolver path through line_pos)"
            );
            assert_eq!(
                coeff_at(layout_a.line_rev_col(l_pos, blk)),
                -1.0,
                "blk {blk}: generic row must carry -1.0 on line 100's reverse column"
            );
            // BusDeficit(2): +1.0 on each of bus 2's two deficit-segment columns.
            for seg in 0..2 {
                assert_eq!(
                    coeff_at(layout_a.deficit_col(b_pos, seg, blk)),
                    1.0,
                    "blk {blk}: generic row must carry +1.0 on bus 2 deficit segment {seg} \
                     (resolver path through bus_pos)"
                );
            }
        }
    }

    /// Pin the two structural water-row coefficients `fill_state_and_water_entries`
    /// writes for a two-reservoir cascade: the cascade-upstream `−tau_h` and the
    /// AR-lag `−ζ·ψ`. Both are the weakest-backstopped coefficients in the water
    /// row — a sign flip on either silently mis-routes water and produces wrong
    /// bounds, yet outside this test they are exercised only by a slow parity
    /// D-case whose hash-mismatch failure mode does not localize to the water row.
    ///
    /// Cascade `H_up`(id 1) → `H_down`(id 2), both constant-productivity. On the
    /// downstream hydro's water-balance row, per block, the upstream turbine and
    /// spillage columns carry `−tau_h` (the upstream release arriving as inflow)
    /// while the downstream's own turbine carries `+tau_h` (its own outflow). The
    /// `+τ`/`−τ` sign *contrast* is asserted, not just one magnitude: a global
    /// flip negating both would otherwise pass. `tau_h` is computed from
    /// `M3S_TO_HM3` (never a literal) so the assertion cannot drift from the
    /// production constant.
    #[test]
    fn cascade_upstream_tau_and_ar_lag_land_on_downstream_water_row() {
        use cobre_core::scenario::InflowModel;

        // Assemble the production water-row fill into a CSC. The generic-constraint
        // fill is omitted (no generic constraints here); the water row is reached
        // through `build_stage_matrix_entries` alone.
        // H_up id 1 sorts to position 0, H_down id 2 to position 1.
        let up = 1;
        let down = 2;
        let cascade_fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );

        // Run the production water-row fill into a CSC. `ctx`/`state`/`layout` are
        // held by the test (borrowing `cascade_fixtures`, which outlives them), so
        // the layout offsets read below stay valid.
        let ctx = cascade_fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            // Mirror the production per-column row-sort (see
            // build_single_stage_template) before assembling the CSC.
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };

        // Sum the CSC values landing on `(col, row)` — mirrors the permutation
        // test's probe; multiple per-block pushes to one cell would otherwise hide.
        let coeff_at = |col: usize, row: i32| -> f64 {
            let start = usize::try_from(csc.0[col]).unwrap();
            let end = usize::try_from(csc.0[col + 1]).unwrap();
            csc.1[start..end]
                .iter()
                .zip(&csc.2[start..end])
                .filter(|&(&r, _)| r == row)
                .map(|(_, &v)| v)
                .sum()
        };

        let up_idx = 0; // H_up id 1.
        let down_idx = 1; // H_down id 2.
        let down_row = i32::try_from(layout.row_water_balance_start() + down_idx).unwrap();
        for blk in 0..layout.n_blks {
            // tau_h is the identical expression the production fill uses; the two
            // blocks carry distinct durations (300 vs 444), so a per-block divisor
            // confusion is observable.
            let tau_h = two_block_stage(0, [300.0, 444.0]).blocks[blk].duration_hours * M3S_TO_HM3;

            assert_eq!(
                coeff_at(layout.turbine_col(up_idx, blk), down_row),
                -tau_h,
                "blk {blk}: upstream turbine column must carry -tau_h on the \
                 downstream water row (cascade-upstream inflow)"
            );
            assert_eq!(
                coeff_at(layout.spillage_col(up_idx, blk), down_row),
                -tau_h,
                "blk {blk}: upstream spillage column must carry -tau_h on the \
                 downstream water row (cascade-upstream inflow)"
            );
            assert_eq!(
                coeff_at(layout.turbine_col(down_idx, blk), down_row),
                tau_h,
                "blk {blk}: downstream's OWN turbine column must carry +tau_h on \
                 its water row (self-outflow) — the +τ/−τ sign contrast"
            );
        }

        // AR-lag −ζ·ψ: self-contained block. An AR(1) PrecomputedPar carrying a
        // nonzero psi for the downstream hydro makes the AR-lag water term fire
        // (the default fixture has psi == 0, so the term is otherwise dormant).
        // psi[0] for the downstream hydro is constructed to equal phi exactly:
        // the classical conversion psi = phi * s_m / s_lag collapses to phi when
        // both the study stage and its pre-study lag stage carry the same std.
        let phi = 0.6_f64;
        let inflow_models = vec![
            // Upstream (id 1): white noise — psi stays 0 at every lag.
            InflowModel {
                hydro_id: EntityId(up),
                stage_id: 0,
                mean_m3s: 50.0,
                std_m3s: 1.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            },
            // Downstream (id 2) study stage: AR(1) with coefficient phi; std == 1.0
            // so the lag-stage ratio is exactly 1.0.
            InflowModel {
                hydro_id: EntityId(down),
                stage_id: 0,
                mean_m3s: 80.0,
                std_m3s: 1.0,
                ar_coefficients: vec![phi],
                residual_std_ratio: 1.0,
                annual: None,
            },
            // Downstream pre-study lag stage (stage_id -1) with the same std, so
            // the Tier-1 exact-match lag lookup yields s_lag == s_m and the
            // conversion collapses to psi[0] == phi bit-exactly.
            InflowModel {
                hydro_id: EntityId(down),
                stage_id: -1,
                mean_m3s: 80.0,
                std_m3s: 1.0,
                ar_coefficients: vec![phi],
                residual_std_ratio: 1.0,
                annual: None,
            },
        ];
        let par_stage = two_block_stage(0, [300.0, 444.0]);
        let par_lp = PrecomputedPar::build(
            &inflow_models,
            std::slice::from_ref(&par_stage),
            &[EntityId(up), EntityId(down)],
            None,
        )
        .expect("AR(1) PrecomputedPar build must succeed");
        let psi_val = par_lp.psi_slice(0, down_idx)[0];
        assert_eq!(psi_val, phi, "downstream psi[0] must equal phi exactly");

        let ar_fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        )
        .with_par_lp(par_lp);
        let ar_ctx = ar_fixtures.make_ctx();
        let ar_stage = two_block_stage(0, [300.0, 444.0]);
        let ar_state = state_layout_for(&ar_ctx);
        let ar_layout = StageLayout::new(&ar_ctx, &ar_state, &ar_stage, 0);
        let ar_csc = {
            let mut entries = build_stage_matrix_entries(&ar_ctx, &ar_stage, 0, &ar_layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let ar_coeff_at = |col: usize, row: i32| -> f64 {
            let start = usize::try_from(ar_csc.0[col]).unwrap();
            let end = usize::try_from(ar_csc.0[col + 1]).unwrap();
            ar_csc.1[start..end]
                .iter()
                .zip(&ar_csc.2[start..end])
                .filter(|&(&r, _)| r == row)
                .map(|(_, &v)| v)
                .sum()
        };
        // Lag column for (lag 0, downstream hydro): col_inflow_lags_start + 0*n_h + h.
        let lag_col = ar_layout.col_inflow_lags_start() + down_idx;
        let ar_row = i32::try_from(ar_layout.row_water_balance_start() + down_idx).unwrap();
        assert_eq!(
            ar_coeff_at(lag_col, ar_row),
            -(ar_layout.zeta * psi_val),
            "downstream inflow-lag column must carry -(zeta * psi) on its water row"
        );
    }

    /// End-to-end: a generic constraint referencing `pumping_flow` and
    /// `pumping_power` resolves to the REAL pumping column(s) through the
    /// resolver's sole caller (`fill_generic_constraint_entries`), and the
    /// constraint participates in the LP — its row carries CSC entries on the
    /// pumping columns. The `block_id = None` expression is block-dependent, so it
    /// expands to one generic row per block; each row's two terms (flow ×1.0 and
    /// power ×consumption) alias the SAME pumping column for that block, so the
    /// summed coefficient at `(pumping_col, generic_row)` is `1.0 + consumption`.
    #[test]
    fn b6b_generic_constraint_resolves_pumping_columns_in_lp() {
        let consumption = 0.5_f64;
        let constraint_id = EntityId(7);
        let station_id = EntityId(10);

        // pumping_flow(10) + pumping_power(10) <= 40 (block_id = None on both).
        let constraint = GenericConstraint {
            id: constraint_id,
            name: "gc_pump".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![
                    LinearTerm {
                        coefficient: CoefficientRef::Literal(1.0),
                        scale: 1.0,
                        variable: VariableRef::PumpingFlow {
                            station_id,
                            block_id: None,
                        },
                    },
                    LinearTerm {
                        coefficient: CoefficientRef::Literal(1.0),
                        scale: 1.0,
                        variable: VariableRef::PumpingPower {
                            station_id,
                            block_id: None,
                        },
                    },
                ],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };

        let fixtures = PumpFixtures::new(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station_full(station_id.0, 1, 2, 0.0, 50.0, 1, consumption)],
        )
        .with_generic_constraint(constraint, 40.0);
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        // Block-dependent expression with block_id = None expands to one generic
        // row per block, so the constraint participates as `n_blks` rows.
        let n_blks = layout.n_blks;
        assert_eq!(
            layout.n_generic_rows, n_blks,
            "block-dependent pumping constraint must expand to one row per block"
        );

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut row_lower = vec![f64::NEG_INFINITY; layout.num_rows];
        let mut row_upper = vec![f64::INFINITY; layout.num_rows];
        let mut buffers = LpMatrixBuffers {
            col_entries: &mut col_entries,
            col_upper: &mut col_upper,
            objective: &mut objective,
            row_lower: &mut row_lower,
            row_upper: &mut row_upper,
        };

        fill_generic_constraint_entries(&ctx, &stage, 0, &layout, &mut buffers);

        // Each generic row `blk` lands on the station's flow column for that block,
        // with the flow (1.0) and power (consumption) terms aliasing the SAME column.
        // p_idx = 0 (the only station), so col = col_pumping_start + blk.
        for blk in 0..n_blks {
            let row = layout.row_generic_start + blk;
            let col = layout.col_pumping_start + blk;
            let summed: f64 = col_entries[col]
                .iter()
                .filter(|&&(r, _)| r == row)
                .map(|&(_, v)| v)
                .sum();
            assert_eq!(
                summed,
                1.0 + consumption,
                "blk {blk}: pumping column {col} must carry flow(1.0) + power({consumption}) on generic row {row}"
            );
            // The row bound proves the constraint participates with the right sense.
            assert_eq!(row_upper[row], 40.0, "blk {blk}: <= row upper bound");
            assert_eq!(row_lower[row], f64::NEG_INFINITY, "blk {blk}: <= row lower");
        }
    }

    // ── Filling impound-retention row ────────────────────────────────────────

    use cobre_core::entities::hydro::FillingConfig;

    const RET_START_STAGE_ID: i32 = 2;
    const RET_ENTRY_STAGE_ID: i32 = 4;
    const RET_FILLING_ID: i32 = 3; // start <= 3 < entry  -> Filling
    const RET_OPERATING_ID: i32 = 4; // 4 >= entry         -> Operating

    /// A cascade hydro with caller-chosen `downstream_id`, `entry_stage_id`, and
    /// optional `FillingConfig`, so an upstream→downstream cascade with a filling
    /// downstream can be built. Constant productivity (no FPHA columns).
    fn ret_hydro(id: i32, downstream: Option<i32>, entry: Option<i32>, filling: bool) -> Hydro {
        Hydro {
            id: EntityId(id),
            name: format!("H{id}"),
            bus_id: EntityId(1),
            downstream_id: downstream.map(EntityId),
            entry_stage_id: entry,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 45.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: filling.then_some(FillingConfig {
                start_stage_id: RET_START_STAGE_ID,
                filling_inflow_m3s: 0.0,
            }),
            penalties: zero_hydro_penalties(),
        }
    }

    /// Build an `H1 → H2` cascade (H2 the filling hydro) at the given `stage_id`,
    /// with the resolved per-stage `filling_inflow_m3s` of H2 set to `f_h`. Returns
    /// the assembled CSC triple, the `(row_lower, row_upper)` vectors, the layout's
    /// `num_rows`, and the resolved offsets the assertions read.
    #[allow(clippy::type_complexity)]
    fn build_retention_case(
        stage_id: i32,
        f_h: f64,
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        usize,
        RetOffsets,
    ) {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
            ],
            Vec::new(),
        );
        // H2 (system index 1 after id-sort) carries the per-stage impound cap.
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .filling_inflow_m3s = f_h;
        let ctx = fixtures.make_ctx();
        let stage_index = usize::try_from(stage_id).expect("non-negative");
        let stage = two_block_stage(stage_index, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let h1_idx = fixtures.hydro_pos[&EntityId(1)];
        let offsets = RetOffsets {
            zeta: layout.zeta,
            n_blks: layout.n_blks,
            h2_idx,
            ret_row: layout.row_filling_retention_start(),
            n_ret_rows: layout.filling_retention_hydro_indices.len(),
            h2_spillage: (0..layout.n_blks)
                .map(|blk| layout.spillage_col(h2_idx, blk))
                .collect(),
            h2_turbine: (0..layout.n_blks)
                .map(|blk| layout.turbine_col(h2_idx, blk))
                .collect(),
            h1_spillage: (0..layout.n_blks)
                .map(|blk| layout.spillage_col(h1_idx, blk))
                .collect(),
            h1_turbine: (0..layout.n_blks)
                .map(|blk| layout.turbine_col(h1_idx, blk))
                .collect(),
            z_h2: layout.col_z_inflow_start() + h2_idx,
            water_row_h2: layout.row_water_balance_start() + h2_idx,
        };
        (csc, row_lower, row_upper, layout.num_rows, offsets)
    }

    struct RetOffsets {
        zeta: f64,
        n_blks: usize,
        h2_idx: usize,
        ret_row: usize,
        n_ret_rows: usize,
        h2_spillage: Vec<usize>,
        h2_turbine: Vec<usize>,
        h1_spillage: Vec<usize>,
        h1_turbine: Vec<usize>,
        z_h2: usize,
        water_row_h2: usize,
    }

    /// Sum the CSC values stored at `(col, row)`.
    fn csc_at(csc: &(Vec<i32>, Vec<i32>, Vec<f64>), col: usize, row: usize) -> f64 {
        let start = usize::try_from(csc.0[col]).unwrap();
        let end = usize::try_from(csc.0[col + 1]).unwrap();
        csc.1[start..end]
            .iter()
            .zip(&csc.2[start..end])
            .filter(|(r, _)| usize::try_from(**r).is_ok_and(|r| r == row))
            .map(|(_, &v)| v)
            .sum()
    }

    /// At a Filling stage the retention row exists with exactly the expected
    /// columns/coefficients: `+τ_h` on the filling hydro's own release (spillage +
    /// turbine), `−τ_h` on the upstream release (H1 spillage + turbine), `−ζ` on
    /// the realized-inflow `z_h` column, and `≥` sense with RHS `−ζ·F_h`. The cap
    /// references the SAME columns and coefficients as the standard balance's
    /// outflow/cascade terms.
    #[test]
    fn retention_row_columns_coefficients_and_rhs_at_filling_stage() {
        let f_h = 12.5_f64;
        let (csc, row_lower, row_upper, _num_rows, off) = build_retention_case(RET_FILLING_ID, f_h);
        assert_eq!(off.n_ret_rows, 1, "exactly one retention row (H2 filling)");
        let row = off.ret_row;

        for blk in 0..off.n_blks {
            let tau_h = [300.0_f64, 444.0][blk] * M3S_TO_HM3;
            assert_eq!(
                csc_at(&csc, off.h2_spillage[blk], row),
                tau_h,
                "blk {blk}: own spillage carries +τ_h on the retention row"
            );
            assert_eq!(
                csc_at(&csc, off.h2_turbine[blk], row),
                tau_h,
                "blk {blk}: own turbine carries +τ_h (structural identity with the balance)"
            );
            assert_eq!(
                csc_at(&csc, off.h1_spillage[blk], row),
                -tau_h,
                "blk {blk}: upstream H1 spillage carries −τ_h on the retention row"
            );
            assert_eq!(
                csc_at(&csc, off.h1_turbine[blk], row),
                -tau_h,
                "blk {blk}: upstream H1 turbine carries −τ_h on the retention row"
            );
        }
        // Realized incremental inflow z_h enters with −ζ (stage-level, one entry).
        assert_eq!(
            csc_at(&csc, off.z_h2, row),
            -off.zeta,
            "z_h column carries −ζ on the retention row (realized inflow)"
        );
        // Sense ≥ with RHS −ζ·F_h.
        assert_eq!(
            row_lower[row],
            -off.zeta * f_h,
            "retention row_lower == −ζ·F_h"
        );
        assert_eq!(
            row_upper[row],
            f64::INFINITY,
            "retention row is a ≥ inequality (upper = +∞)"
        );
    }

    /// `F_h = 0` ⇒ retention RHS is `0`, i.e. `total release − ζ·z_h ≥ 0`: the
    /// reservoir must pass the entire realized natural inflow downstream (a zero
    /// impound cap).
    #[test]
    fn retention_rhs_is_zero_when_filling_inflow_is_zero() {
        let (_csc, row_lower, row_upper, _num_rows, off) =
            build_retention_case(RET_FILLING_ID, 0.0);
        assert_eq!(
            row_lower[off.ret_row], 0.0,
            "F_h = 0 ⇒ retention RHS = −ζ·0 = 0 (pass all natural inflow)"
        );
        assert_eq!(row_upper[off.ret_row], f64::INFINITY, "still a ≥ row");
    }

    /// The standard water-balance EQUALITY row of the filling hydro is UNCHANGED by
    /// the retention family: its outflow/cascade/z-coupling coefficients and its
    /// equality `row_lower == row_upper` RHS match an equivalent OPERATING hydro
    /// built with the same cascade and bounds. The retention row is purely
    /// additional — it never edits the balance row.
    #[test]
    fn standard_balance_row_unchanged_versus_operating_hydro() {
        // Filling build at a Filling stage.
        let (csc_f, rl_f, ru_f, _nr_f, off_f) = build_retention_case(RET_FILLING_ID, 9.0);
        // Operating build: same topology, but H2 is past entry (Operating), so no
        // retention row and the hydro behaves as a normal plant.
        let (csc_o, rl_o, ru_o, _nr_o, off_o) = build_retention_case(RET_OPERATING_ID, 9.0);

        // The water-balance row index of H2 is identical in both builds (the
        // retention block is inserted AFTER the balance/load/fpha/evap/operational
        // rows, so it never shifts the balance rows).
        assert_eq!(
            off_f.water_row_h2, off_o.water_row_h2,
            "water-balance row index of H2 is unshifted by the retention family"
        );
        let wr = off_f.water_row_h2;

        // Outgoing/incoming storage, own release, upstream release, and z-coupling
        // coefficients on the balance row match between filling and operating.
        assert_eq!(
            csc_at(&csc_f, off_f.h2_idx, wr),
            csc_at(&csc_o, off_o.h2_idx, wr),
            "outgoing-storage coefficient on the balance row is unchanged"
        );
        for blk in 0..off_f.n_blks {
            assert_eq!(
                csc_at(&csc_f, off_f.h2_spillage[blk], wr),
                csc_at(&csc_o, off_o.h2_spillage[blk], wr),
                "blk {blk}: balance-row own-spillage coefficient unchanged"
            );
            assert_eq!(
                csc_at(&csc_f, off_f.h1_spillage[blk], wr),
                csc_at(&csc_o, off_o.h1_spillage[blk], wr),
                "blk {blk}: balance-row upstream-spillage coefficient unchanged"
            );
        }
        // Equality RHS (row_lower == row_upper) of the balance row matches.
        assert_eq!(rl_f[wr], ru_f[wr], "filling balance row is an equality");
        assert_eq!(rl_o[wr], ru_o[wr], "operating balance row is an equality");
        assert_eq!(
            rl_f[wr], rl_o[wr],
            "balance-row RHS unchanged between filling and operating"
        );
    }

    /// A filling H2 in the Operating phase emits NO retention row (retention is a
    /// Filling-phase-only family). Its `num_rows` differs from a non-filling control
    /// by EXACTLY the σ^{v-} operating-floor row that H2 legitimately gains in
    /// Operating — isolating the row-count delta to the soft floor, not the
    /// retention block. (Strict parity-neutrality of `num_rows` holds only for a
    /// genuinely non-filling hydro, asserted by `non_filling_system_emits_no_sigma_minus`.)
    #[test]
    fn non_filling_emits_no_retention_row_and_num_rows_unchanged() {
        // H2 in Operating: filling hydro but past entry ⇒ no retention row, but it
        // does gain the σ^{v-} operating floor.
        let (_csc_op, _rl, _ru, num_rows_op, off_op) = build_retention_case(RET_OPERATING_ID, 5.0);
        assert_eq!(off_op.n_ret_rows, 0, "no retention row in Operating");

        // A control build with NO filling hydro at all, same topology/stage.
        let control = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, None, false),
            ],
            Vec::new(),
        );
        let ctx = control.make_ctx();
        let stage = two_block_stage(usize::try_from(RET_OPERATING_ID).unwrap(), [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        assert_eq!(
            layout.filling_retention_hydro_indices.len(),
            0,
            "control system has no filling hydro ⇒ no retention row"
        );
        assert_eq!(
            layout.filling_floor_hydro_indices.len(),
            0,
            "control system has no filling hydro ⇒ no σ^{{v-}} floor row"
        );
        // The filling-H2 Operating build adds exactly one σ^{v-} row over the
        // non-filling control; the retention block is empty in both.
        assert_eq!(
            num_rows_op,
            layout.num_rows + 1,
            "filling H2 in Operating gains exactly the σ^{{v-}} floor row over the \
             non-filling control (retention empty in both)"
        );
    }

    /// The retention rows land STRICTLY BELOW `num_rows` (inside the structural
    /// region, ahead of the appended cut rows that start at `num_rows`). A
    /// retention row written at index `>= num_rows` would alias the cut rows.
    #[test]
    fn retention_rows_lie_below_num_rows() {
        let (_csc, _rl, _ru, num_rows, off) = build_retention_case(RET_FILLING_ID, 1.0);
        assert!(off.n_ret_rows > 0, "this case has a retention row");
        assert!(
            off.ret_row + off.n_ret_rows <= num_rows,
            "retention rows [{}, {}) must lie within [0, num_rows={})",
            off.ret_row,
            off.ret_row + off.n_ret_rows,
            num_rows
        );
    }

    // ── σ_fill terminal target row + column ──────────────────────────────────

    // The terminal Filling stage of the `ret_hydro` window: entry − 1 = 3.
    const TARGET_TERMINAL_ID: i32 = RET_ENTRY_STAGE_ID - 1;
    // A non-zero resolved dead volume so the σ_fill row RHS is observable.
    const TARGET_MIN_STORAGE_HM3: f64 = 37.5;

    struct TargetOffsets {
        n_target_rows: usize,
        target_row: usize,
        sigma_fill_col: usize,
        v_h_col: usize,
        num_rows: usize,
        num_cols: usize,
    }

    /// Build the `H1 → H2` cascade (H2 the filling hydro, `entry = 4`) at the given
    /// `stage_id`, with H2's resolved per-stage `min_storage_hm3` set to
    /// `TARGET_MIN_STORAGE_HM3`. Returns the assembled CSC triple, the
    /// `(row_lower, row_upper)` vectors, and the `σ_fill` offsets the assertions read.
    #[allow(clippy::type_complexity)]
    fn build_target_case(
        stage_id: i32,
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        TargetOffsets,
    ) {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures.bounds.hydro_bounds_mut(h2_idx, 0).min_storage_hm3 = TARGET_MIN_STORAGE_HM3;
        let ctx = fixtures.make_ctx();
        let stage_index = usize::try_from(stage_id).expect("non-negative");
        let stage = two_block_stage(stage_index, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let offsets = TargetOffsets {
            n_target_rows: layout.filling_target_hydro_indices.len(),
            target_row: layout.row_filling_target_start(),
            sigma_fill_col: layout.col_filling_target_start(),
            // The outgoing storage column v_h is the dense system index h2_idx.
            v_h_col: h2_idx,
            num_rows: layout.num_rows,
            num_cols: layout.num_cols,
        };
        (csc, row_lower, row_upper, offsets)
    }

    /// At the terminal Filling stage (id == entry − 1 == 3) the `σ_fill` row exists
    /// with exactly `+1` on the outgoing storage column `v_h`, `+1` on the `σ_fill`
    /// slack column, `≥` sense, and RHS `min_storage_hm3`. Exactly one such row +
    /// column (the single filling hydro H2).
    #[test]
    fn sigma_fill_row_columns_coefficients_and_rhs_at_terminal_stage() {
        let (csc, row_lower, row_upper, off) = build_target_case(TARGET_TERMINAL_ID);
        assert_eq!(off.n_target_rows, 1, "exactly one σ_fill row (H2 terminal)");
        let row = off.target_row;
        assert_eq!(
            csc_at(&csc, off.v_h_col, row),
            1.0,
            "+1 on the outgoing storage column v_h"
        );
        assert_eq!(
            csc_at(&csc, off.sigma_fill_col, row),
            1.0,
            "+1 on the σ_fill slack column"
        );
        assert_eq!(
            row_lower[row], TARGET_MIN_STORAGE_HM3,
            "σ_fill row_lower == min_storage_hm3 (≥ RHS)"
        );
        assert_eq!(
            row_upper[row],
            f64::INFINITY,
            "σ_fill row is a ≥ inequality (upper = +∞)"
        );
    }

    /// No `σ_fill` row or column is emitted at any stage id `≠ entry − 1` — a
    /// non-terminal Filling stage (id 2) or Operating (id 4). Terminal-only.
    ///
    /// At the `Filling` stage (id 2) the `σ^{v-}` family is also empty, so the empty
    /// `σ_fill` block's cursor coincides with `num_rows`/`num_cols`. At `Operating`
    /// (id 4) the `σ^{v-}` family legitimately adds exactly one row + column for the
    /// single filling hydro AFTER the empty `σ_fill` block, so the `σ_fill` cursor
    /// sits one short of `num_rows`/`num_cols` — the `σ_fill` block is still empty
    /// (`n_target_rows == 0`), it is simply no longer the last family.
    #[test]
    fn sigma_fill_absent_off_terminal_stage() {
        for stage_id in [RET_START_STAGE_ID, RET_OPERATING_ID] {
            let (_csc, _rl, _ru, off) = build_target_case(stage_id);
            assert_eq!(
                off.n_target_rows, 0,
                "no σ_fill row at id {stage_id} (only entry − 1 = 3 carries it)"
            );
            // σ^{v-} adds one row + column at Operating (the single filling hydro),
            // none at the Filling stage.
            let sigma_minus_width = usize::from(stage_id == RET_OPERATING_ID);
            assert_eq!(
                off.target_row + sigma_minus_width,
                off.num_rows,
                "empty σ_fill row block at id {stage_id} (σ^{{v-}} occupies the tail in Operating)"
            );
            assert_eq!(
                off.sigma_fill_col + sigma_minus_width,
                off.num_cols,
                "empty σ_fill column block at id {stage_id} (σ^{{v-}} occupies the tail in Operating)"
            );
        }
    }

    /// A non-filling system (no `FillingConfig` on any hydro) emits NO `σ_fill` row
    /// or column, and its `num_rows`/`num_cols` are unchanged at the would-be
    /// terminal stage — the parity-neutrality contract.
    #[test]
    fn non_filling_system_emits_no_sigma_fill() {
        let control = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, None, false),
            ],
            Vec::new(),
        );
        let ctx = control.make_ctx();
        let stage = two_block_stage(usize::try_from(TARGET_TERMINAL_ID).unwrap(), [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        assert_eq!(
            layout.filling_target_hydro_indices.len(),
            0,
            "control system has no filling hydro ⇒ no σ_fill row/column"
        );
        // The σ_fill row/column cursors degenerate to the structural bounds.
        assert_eq!(layout.row_filling_target_start(), layout.num_rows);
        assert_eq!(layout.col_filling_target_start(), layout.num_cols);
    }

    /// Cut-validity guard (§4 trap 3): the `σ_fill` soft row couples to the storage
    /// state through the constraint matrix, so LP duality folds its dual into the
    /// SINGLE reduced cost of the incoming-storage column the cut already reads as
    /// `rc / col_scale`. The `σ_fill` builder code therefore must NEVER reference the
    /// dual-extraction entry point — doing so would signal a hand-combination of the
    /// soft-row dual that double-counts the floor and corrupts the cut. This test
    /// reads every `lp/builder/` source file and asserts none mentions that symbol,
    /// so a future "simplification" that wires the soft dual in by hand fails here.
    /// The dual-extraction function itself lives in
    /// `training/backward/duals_extraction.rs`, untouched by this family.
    ///
    /// The forbidden symbol is assembled from chars so this guard's own source text
    /// does not contain the literal (which would make the test flag itself).
    #[test]
    fn lp_builder_never_references_dual_extraction() {
        use std::path::Path;
        // The dual-extraction symbol, assembled from fragments so the needle is
        // absent from this file's own bytes (else the guard would flag itself).
        let needle: String = ["extract", "_duals", "_from", "_view"].concat();
        let builder_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lp/builder");
        assert!(
            builder_dir.is_dir(),
            "lp/builder source dir must exist at {}",
            builder_dir.display()
        );
        let mut offenders: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(&builder_dir).expect("read lp/builder dir") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let src = std::fs::read_to_string(&path).expect("read builder source file");
            if src.contains(&needle) {
                offenders.push(path.display().to_string());
            }
        }
        assert!(
            offenders.is_empty(),
            "lp/builder must not reference the dual-extraction symbol (the σ_fill \
             soft dual is folded into the storage column reduced cost by LP duality, \
             never hand-combined); offending files: {offenders:?}"
        );
    }

    /// The `σ_fill` row lands STRICTLY BELOW `num_rows` (inside the structural
    /// region, ahead of the appended cut rows that start at `num_rows`), and the
    /// `σ_fill` column strictly below `num_cols`. A `σ_fill` row at index `>= num_rows`
    /// would alias a cut row and corrupt slot-identity warm-start reconstruction.
    #[test]
    fn sigma_fill_row_below_num_rows_and_col_below_num_cols() {
        let (_csc, _rl, _ru, off) = build_target_case(TARGET_TERMINAL_ID);
        assert!(off.n_target_rows > 0, "this case has a σ_fill row");
        assert!(
            off.target_row + off.n_target_rows <= off.num_rows,
            "σ_fill rows [{}, {}) must lie within [0, num_rows={})",
            off.target_row,
            off.target_row + off.n_target_rows,
            off.num_rows
        );
        assert!(
            off.sigma_fill_col + off.n_target_rows <= off.num_cols,
            "σ_fill cols [{}, {}) must lie within [0, num_cols={})",
            off.sigma_fill_col,
            off.sigma_fill_col + off.n_target_rows,
            off.num_cols
        );
    }

    // ── σ^{v-} operating-floor row + column ──────────────────────────────────

    struct FloorOffsets {
        n_floor_rows: usize,
        floor_row: usize,
        sigma_minus_col: usize,
        v_h_col: usize,
        num_rows: usize,
        num_cols: usize,
    }

    /// Build the `H1 → H2` cascade (H2 the filling hydro, `entry = 4`) at the given
    /// `stage_id`, with H2's resolved per-stage `min_storage_hm3` set to
    /// `TARGET_MIN_STORAGE_HM3`. Returns the assembled CSC triple, the
    /// `(row_lower, row_upper)` vectors, and the `σ^{v-}` offsets the assertions
    /// read. Mirrors `build_target_case` but reads the `filling_floor` family.
    #[allow(clippy::type_complexity)]
    fn build_floor_case(
        stage_id: i32,
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        FloorOffsets,
    ) {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures.bounds.hydro_bounds_mut(h2_idx, 0).min_storage_hm3 = TARGET_MIN_STORAGE_HM3;
        let ctx = fixtures.make_ctx();
        let stage_index = usize::try_from(stage_id).expect("non-negative");
        let stage = two_block_stage(stage_index, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let offsets = FloorOffsets {
            n_floor_rows: layout.filling_floor_hydro_indices.len(),
            floor_row: layout.row_filling_floor_start(),
            sigma_minus_col: layout.col_filling_floor_start(),
            // The outgoing storage column v_h is the dense system index h2_idx.
            v_h_col: h2_idx,
            num_rows: layout.num_rows,
            num_cols: layout.num_cols,
        };
        (csc, row_lower, row_upper, offsets)
    }

    /// At an Operating stage (id == entry == 4) the `σ^{v-}` row exists with exactly
    /// `+1` on the outgoing storage column `v_h`, `+1` on the `σ^{v-}` slack column,
    /// `≥` sense, and RHS `min_storage_hm3`. Exactly one such row + column (the
    /// single filling hydro H2). Same shape as the `σ_fill` row, different stage
    /// scope.
    #[test]
    fn sigma_minus_row_columns_coefficients_and_rhs_in_operating() {
        let (csc, row_lower, row_upper, off) = build_floor_case(RET_OPERATING_ID);
        assert_eq!(
            off.n_floor_rows, 1,
            "exactly one σ^{{v-}} row (H2 operating)"
        );
        let row = off.floor_row;
        assert_eq!(
            csc_at(&csc, off.v_h_col, row),
            1.0,
            "+1 on the outgoing storage column v_h"
        );
        assert_eq!(
            csc_at(&csc, off.sigma_minus_col, row),
            1.0,
            "+1 on the σ^{{v-}} slack column"
        );
        assert_eq!(
            row_lower[row], TARGET_MIN_STORAGE_HM3,
            "σ^{{v-}} row_lower == min_storage_hm3 (≥ RHS)"
        );
        assert_eq!(
            row_upper[row],
            f64::INFINITY,
            "σ^{{v-}} row is a ≥ inequality (upper = +∞)"
        );
    }

    /// No `σ^{v-}` row or column is emitted at a non-operating stage of a filling
    /// hydro — `PreFilling` (id 0) or a Filling stage (id 3). Operating-only.
    #[test]
    fn sigma_minus_absent_off_operating_stage() {
        for stage_id in [RET_PREFILLING_ID, TARGET_TERMINAL_ID] {
            let (_csc, _rl, _ru, off) = build_floor_case(stage_id);
            assert_eq!(
                off.n_floor_rows, 0,
                "no σ^{{v-}} row at id {stage_id} (only Operating carries it)"
            );
            assert_eq!(
                off.floor_row, off.num_rows,
                "empty σ^{{v-}} row block: cursor coincides with num_rows at id {stage_id}"
            );
            assert_eq!(
                off.sigma_minus_col, off.num_cols,
                "empty σ^{{v-}} column block: cursor coincides with num_cols at id {stage_id}"
            );
        }
    }

    /// At the terminal `Filling` stage (id == entry − 1 == 3) a filling hydro has
    /// the `σ_fill` family but NOT `σ^{v-}`; in `Operating` (id == entry == 4) it has
    /// `σ^{v-}` but NOT `σ_fill` — the two families are mutually exclusive by stage
    /// and must not be conflated.
    #[test]
    fn sigma_minus_and_sigma_fill_mutually_exclusive_by_stage() {
        let (_csc_t, _rl_t, _ru_t, target_at_terminal) = build_target_case(TARGET_TERMINAL_ID);
        let (_csc_tf, _rl_tf, _ru_tf, floor_at_terminal) = build_floor_case(TARGET_TERMINAL_ID);
        assert_eq!(
            target_at_terminal.n_target_rows, 1,
            "σ_fill present at terminal Filling stage"
        );
        assert_eq!(
            floor_at_terminal.n_floor_rows, 0,
            "σ^{{v-}} absent at terminal Filling stage"
        );

        let (_csc_o, _rl_o, _ru_o, target_in_op) = build_target_case(RET_OPERATING_ID);
        let (_csc_of, _rl_of, _ru_of, floor_in_op) = build_floor_case(RET_OPERATING_ID);
        assert_eq!(target_in_op.n_target_rows, 0, "σ_fill absent in Operating");
        assert_eq!(floor_in_op.n_floor_rows, 1, "σ^{{v-}} present in Operating");
    }

    /// A non-filling system (no `FillingConfig` on any hydro) emits NO `σ^{v-}` row
    /// or column, and its `num_rows`/`num_cols` are unchanged at the would-be
    /// Operating stage — the parity-neutrality contract.
    #[test]
    fn non_filling_system_emits_no_sigma_minus() {
        let control = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, None, false),
            ],
            Vec::new(),
        );
        let ctx = control.make_ctx();
        let stage = two_block_stage(usize::try_from(RET_OPERATING_ID).unwrap(), [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        assert_eq!(
            layout.filling_floor_hydro_indices.len(),
            0,
            "control system has no filling hydro ⇒ no σ^{{v-}} row/column"
        );
        // The σ^{v-} row/column cursors degenerate to the structural bounds.
        assert_eq!(layout.row_filling_floor_start(), layout.num_rows);
        assert_eq!(layout.col_filling_floor_start(), layout.num_cols);
    }

    /// The `σ^{v-}` row lands STRICTLY BELOW `num_rows` (inside the structural
    /// region, ahead of the appended cut rows that start at `num_rows`), and the
    /// `σ^{v-}` column strictly below `num_cols`. A `σ^{v-}` row at index
    /// `>= num_rows` would alias a cut row and corrupt slot-identity warm-start
    /// reconstruction.
    #[test]
    fn sigma_minus_row_below_num_rows_and_col_below_num_cols() {
        let (_csc, _rl, _ru, off) = build_floor_case(RET_OPERATING_ID);
        assert!(off.n_floor_rows > 0, "this case has a σ^{{v-}} row");
        assert!(
            off.floor_row + off.n_floor_rows <= off.num_rows,
            "σ^{{v-}} rows [{}, {}) must lie within [0, num_rows={})",
            off.floor_row,
            off.floor_row + off.n_floor_rows,
            off.num_rows
        );
        assert!(
            off.sigma_minus_col + off.n_floor_rows <= off.num_cols,
            "σ^{{v-}} cols [{}, {}) must lie within [0, num_cols={})",
            off.sigma_minus_col,
            off.sigma_minus_col + off.n_floor_rows,
            off.num_cols
        );
    }

    // ── PreFilling cascade short-circuit ─────────────────────────────────────

    // `RET_START_STAGE_ID = 2`, so a stage with id `< 2` is PreFilling for a
    // filling hydro built by `ret_hydro`. Stage id 0 is the canonical PreFilling
    // probe; the boundary stage id 2 is Filling (governed by the retention tests
    // above), confirming the short-circuit does NOT fire there.
    const RET_PREFILLING_ID: i32 = 0;

    /// Resolved offsets for the H1→H2→H3 routed short-circuit probe.
    struct ScOffsets {
        zeta: f64,
        n_blks: usize,
        h2_idx: usize,
        water_row_h2: usize,
        water_row_h3: usize,
        col_storage_in_h2: usize,
        z_h2: usize,
        h1_turbine: Vec<usize>,
        h1_spillage: Vec<usize>,
        h2_turbine: Vec<usize>,
        h2_spillage: Vec<usize>,
    }

    /// Build the cascade `H1(id 1) → H2(id 2, filling) → H3(id 3)` at `stage_id`,
    /// with H2's resolved per-stage withdrawal set to `withdrawal_h`. Returns the
    /// assembled CSC, the `(row_lower, row_upper)` vectors, and the offsets the
    /// short-circuit assertions read. H2 is the mid-cascade filling hydro, so the
    /// short-circuit routes to a REAL downstream (H3), not a sink.
    #[allow(clippy::type_complexity)]
    fn build_shortcircuit_case(
        stage_id: i32,
        withdrawal_h: f64,
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        ScOffsets,
    ) {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, Some(3), Some(RET_ENTRY_STAGE_ID), true),
                ret_hydro(3, None, None, false),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .water_withdrawal_m3s = withdrawal_h;
        let ctx = fixtures.make_ctx();
        let stage_index = usize::try_from(stage_id).expect("non-negative");
        let stage = two_block_stage(stage_index, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let h1_idx = fixtures.hydro_pos[&EntityId(1)];
        let h3_idx = fixtures.hydro_pos[&EntityId(3)];
        let offsets = ScOffsets {
            zeta: layout.zeta,
            n_blks: layout.n_blks,
            h2_idx,
            water_row_h2: layout.row_water_balance_start() + h2_idx,
            water_row_h3: layout.row_water_balance_start() + h3_idx,
            col_storage_in_h2: layout.col_storage_in_start() + h2_idx,
            z_h2: layout.col_z_inflow_start() + h2_idx,
            h1_turbine: (0..layout.n_blks)
                .map(|blk| layout.turbine_col(h1_idx, blk))
                .collect(),
            h1_spillage: (0..layout.n_blks)
                .map(|blk| layout.spillage_col(h1_idx, blk))
                .collect(),
            h2_turbine: (0..layout.n_blks)
                .map(|blk| layout.turbine_col(h2_idx, blk))
                .collect(),
            h2_spillage: (0..layout.n_blks)
                .map(|blk| layout.spillage_col(h2_idx, blk))
                .collect(),
        };
        (csc, row_lower, row_upper, offsets)
    }

    /// At a `PreFilling` stage the absent hydro `H2`'s four water interactions land
    /// on the DOWNSTREAM `H3`'s balance row in real `ζ`/`τ_k` coefficients:
    /// `H2`'s incremental inflow (`−ζ` on `z_{H2}`), `H2`'s upstream `H1` releases
    /// (`−τ` on `H1` turbine/spillage), and `H2`'s withdrawal demand (`−ζ·withdrawal`
    /// on `H3`'s RHS). `H2`'s evaporation is NOT transferred (no evap column exists
    /// — `PreFilling` is excluded upstream). The same columns carry NOTHING extra on
    /// `H2`'s own (frozen) row.
    #[test]
    fn prefilling_routes_inflow_upstream_releases_and_withdrawal_to_downstream() {
        let withdrawal_h = 17.5_f64;
        let (csc, _rl, row_upper, off) = build_shortcircuit_case(RET_PREFILLING_ID, withdrawal_h);
        let row_d = off.water_row_h3;
        let row_h = off.water_row_h2;

        // (1) Incremental inflow: z_{H2} carries −ζ on H3's row (relocated), and
        // NOTHING on H2's own frozen row.
        assert_eq!(
            csc_at(&csc, off.z_h2, row_d),
            -off.zeta,
            "z_{{H2}} carries −ζ on H3's water row (incremental inflow re-routed)"
        );
        assert_eq!(
            csc_at(&csc, off.z_h2, row_h),
            0.0,
            "z_{{H2}} carries nothing on H2's own frozen-identity row"
        );

        // (2) Upstream H1 releases re-routed upstream(H2)→H3 with −τ_h per block.
        // H1 is NOT a standard upstream of H3 (the standard cascade edge is
        // H1→H2→H3), so a nonzero −τ on H1's columns at H3's row can ONLY come from
        // the short-circuit re-route. (H2's own release columns DO appear on H3's
        // row via H3's standard upstream loop — H2 is structurally upstream of H3 —
        // but H2's turbine/spillage are pinned `[0,0]` in PreFilling, so that
        // standard coefficient multiplies a zero column and is harmless. The
        // re-route is what makes H1's water reach H3.)
        for blk in 0..off.n_blks {
            let tau_h = [300.0_f64, 444.0][blk] * M3S_TO_HM3;
            assert_eq!(
                csc_at(&csc, off.h1_turbine[blk], row_d),
                -tau_h,
                "blk {blk}: H1 turbine carries −τ_h on H3's row (cascade edge \
                 H1→H2→H3 re-routed to H1→H3 — H1 is NOT a standard upstream of H3)"
            );
            assert_eq!(
                csc_at(&csc, off.h1_spillage[blk], row_d),
                -tau_h,
                "blk {blk}: H1 spillage carries −τ_h on H3's row"
            );
        }

        // (3) Withdrawal demand transferred to H3's RHS: H3's row_upper drops by
        // ζ·withdrawal_h versus a no-withdrawal build (its own withdrawal is 0).
        let (_csc0, _rl0, row_upper0, off0) = build_shortcircuit_case(RET_PREFILLING_ID, 0.0);
        let delta = off.zeta * withdrawal_h;
        assert_eq!(
            row_upper0[off0.water_row_h3] - row_upper[row_d],
            delta,
            "H3's RHS drops by ζ·withdrawal_{{H2}} (the transferred withdrawal demand)"
        );
    }

    /// `H2`'s own (frozen) row is exactly `v_{H2} − v_{H2,in} = 0`: `+1.0` on its
    /// outgoing-storage column, `−1.0` on its incoming-storage column, and NOTHING
    /// else (no inflow/upstream/AR-lag/withdrawal/evaporation), with RHS 0. The
    /// storage column keeps its dense system index (the absent reservoir's column
    /// is never omitted or relocated — §4 trap 2).
    #[test]
    fn prefilling_h_row_is_frozen_identity_with_dense_storage_column() {
        let (csc, row_lower, row_upper, off) = build_shortcircuit_case(RET_PREFILLING_ID, 9.0);
        let row_h = off.water_row_h2;

        // Outgoing storage column index == H2's system hydro index (dense, unchanged
        // from an operating build — the storage column is never relocated).
        assert_eq!(
            csc_at(&csc, off.h2_idx, row_h),
            1.0,
            "outgoing storage v_{{H2}} carries +1.0 on the frozen-identity row, \
             at the dense column index == H2's system hydro index"
        );
        assert_eq!(
            csc_at(&csc, off.col_storage_in_h2, row_h),
            -1.0,
            "incoming storage v_{{H2,in}} carries −1.0 on the frozen-identity row"
        );

        // Nothing else on H2's row: own releases, z-coupling, withdrawal slacks.
        for blk in 0..off.n_blks {
            assert_eq!(
                csc_at(&csc, off.h2_turbine[blk], row_h),
                0.0,
                "blk {blk}: own turbine absent from the frozen row"
            );
            assert_eq!(
                csc_at(&csc, off.h2_spillage[blk], row_h),
                0.0,
                "blk {blk}: own spillage absent from the frozen row"
            );
            assert_eq!(
                csc_at(&csc, off.h1_turbine[blk], row_h),
                0.0,
                "blk {blk}: upstream H1 turbine absent from H2's frozen row \
                 (re-routed to H3, not left on H2 — §4 trap 4)"
            );
        }
        assert_eq!(
            csc_at(&csc, off.z_h2, row_h),
            0.0,
            "z_{{H2}} absent from the frozen row"
        );

        // RHS is exactly 0 (NOT ζ·(base − withdrawal)).
        assert_eq!(row_lower[row_h], 0.0, "frozen-identity row_lower == 0");
        assert_eq!(row_upper[row_h], 0.0, "frozen-identity row_upper == 0");
    }

    /// Sink case: a `PreFilling` hydro with `downstream_id == None` drops its water
    /// from the system (exactly as a terminal hydro's outflow today). No downstream
    /// row receives the inflow/upstream/withdrawal, and `H2`'s own row is still the
    /// frozen identity. Builds `H1 → H2(filling, terminal)`.
    #[test]
    fn prefilling_sink_drops_water_and_keeps_frozen_identity() {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .water_withdrawal_m3s = 13.0;
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(usize::try_from(RET_PREFILLING_ID).unwrap(), [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let h1_idx = fixtures.hydro_pos[&EntityId(1)];
        let row_h = layout.row_water_balance_start() + h2_idx;
        let z_h2 = layout.col_z_inflow_start() + h2_idx;

        // Frozen identity intact on H2's own row.
        assert_eq!(csc_at(&csc, h2_idx, row_h), 1.0, "v_{{H2}} +1.0");
        assert_eq!(
            csc_at(&csc, layout.col_storage_in_start() + h2_idx, row_h),
            -1.0,
            "v_{{H2,in}} −1.0"
        );
        assert_eq!(
            row_lower[row_h], 0.0,
            "frozen RHS 0 (no withdrawal folded in)"
        );
        assert_eq!(row_upper[row_h], 0.0, "frozen RHS 0");

        // H2's water exits the system: z_{H2} appears on NO water-balance row, and
        // H1's releases appear only on H1's own row (no downstream to feed).
        for h in 0..layout.n_h {
            let r = layout.row_water_balance_start() + h;
            assert_eq!(
                csc_at(&csc, z_h2, r),
                0.0,
                "z_{{H2}} must not land on any water row in the sink case (row {r})"
            );
        }
        let row_h1 = layout.row_water_balance_start() + h1_idx;
        for blk in 0..layout.n_blks {
            let tau_h = [300.0_f64, 444.0][blk] * M3S_TO_HM3;
            // H1's own +τ on its own row is unchanged; it lands on NO other water row.
            assert_eq!(
                csc_at(&csc, layout.turbine_col(h1_idx, blk), row_h1),
                tau_h,
                "blk {blk}: H1's own turbine still carries +τ on its own row"
            );
            assert_eq!(
                csc_at(&csc, layout.turbine_col(h1_idx, blk), row_h),
                0.0,
                "blk {blk}: H1's turbine does not feed the absent sink H2"
            );
        }
    }

    /// Duals test (real solver): at a `PreFilling` stage the incoming-storage column's
    /// reduced cost is 0 — the §4 trap-4 contract `∂Q/∂v̂_h = 0` (a valid flat cut).
    /// With `v_{H2,in}` relocated out of `H2`'s row (the row is the frozen identity
    /// and `v_{H2}` is a dead variable), perturbing the pinned `v̂_{H2}` changes
    /// neither the reduced cost nor the objective. A botched relocation that left
    /// `H2`'s row coupled to upstream would give a nonzero reduced cost
    /// (`β_{H2}` stale-nonzero).
    #[test]
    fn prefilling_incoming_storage_reduced_cost_is_zero() {
        use cobre_solver::{ActiveSolver, SolverInterface};

        let fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, Some(3), Some(RET_ENTRY_STAGE_ID), true),
                ret_hydro(3, None, None, false),
            ],
            Vec::new(),
        )
        .with_resolved_penalties();
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(usize::try_from(RET_PREFILLING_ID).unwrap(), [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let out = super::super::template::build_single_stage_template(&ctx, &state, &stage, 0);
        let template = out.template;

        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        let col_v_in = state.state_to_lp_incoming_column(h2_idx);

        // Solve twice with different pinned v̂_{H2}; assert the incoming-storage
        // reduced cost is 0 at both and the objective is unchanged. col_scale is
        // empty for this freshly built (unscaled) template, so the reduced cost is
        // already in original units (no /col_scale needed).
        let solve_at = |v_hat: f64| -> (f64, f64) {
            let mut solver = ActiveSolver::new().expect("ActiveSolver::new()");
            solver.load_model(&template);
            solver.set_col_bounds(&[col_v_in], &[v_hat], &[v_hat]);
            let view = solver.solve(None).expect("PreFilling LP must be feasible");
            (view.reduced_costs[col_v_in], view.objective)
        };

        let (rc_lo, obj_lo) = solve_at(10.0);
        let (rc_hi, obj_hi) = solve_at(70.0);

        assert!(
            rc_lo.abs() < 1e-9,
            "incoming-storage reduced cost must be 0 at v̂=10 (flat cut), got {rc_lo}"
        );
        assert!(
            rc_hi.abs() < 1e-9,
            "incoming-storage reduced cost must be 0 at v̂=70 (flat cut), got {rc_hi}"
        );
        assert!(
            (obj_lo - obj_hi).abs() < 1e-9,
            "objective must be invariant to v̂_{{H2}} (∂Q/∂v̂_h = 0): {obj_lo} vs {obj_hi}"
        );
    }

    /// Non-filling parity: building the SAME 3-hydro cascade with H2 NOT filling
    /// produces a bit-identical water-balance row for H2 (no short-circuit, no
    /// relocation) and identical `num_rows`. The `PreFilling` logic is a no-op for a
    /// hydro with no `FillingConfig`.
    #[test]
    fn non_filling_hydro_water_row_and_num_rows_bit_identical() {
        // Filling build at the PreFilling stage: H2 IS short-circuited.
        let (csc_f, rl_f, ru_f, off_f) = build_shortcircuit_case(RET_PREFILLING_ID, 0.0);

        // Control: same topology and stage, but H2 carries no filling ⇒ Operating
        // everywhere ⇒ standard balance row, no short-circuit.
        let control = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, Some(3), None, false),
                ret_hydro(3, None, None, false),
            ],
            Vec::new(),
        );
        let ctx = control.make_ctx();
        let stage = two_block_stage(usize::try_from(RET_PREFILLING_ID).unwrap(), [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let (rl_c, ru_c) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc_c = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let h2_idx_c = control.hydro_pos[&EntityId(2)];
        let row_h2_c = layout.row_water_balance_start() + h2_idx_c;

        // num_rows identical (no extra structural rows from the short-circuit; it
        // only moves coefficients, never adds rows).
        assert_eq!(
            layout.num_rows,
            ru_f.len(),
            "num_rows identical whether H2 is PreFilling or non-filling"
        );

        // H2's OWN water row differs between the builds (frozen vs standard) — that
        // is the short-circuit. The control's H2 row carries its own +τ releases;
        // assert at least the own-turbine coefficient is present in the control and
        // ABSENT in the filling build (the observable short-circuit).
        let row_h2_f = off_f.water_row_h2;
        for blk in 0..off_f.n_blks {
            let tau_h = [300.0_f64, 444.0][blk] * M3S_TO_HM3;
            assert_eq!(
                csc_at(&csc_c, layout.turbine_col(h2_idx_c, blk), row_h2_c),
                tau_h,
                "blk {blk}: control (non-filling) H2 carries +τ on its own row"
            );
            assert_eq!(
                csc_at(&csc_f, off_f.h2_turbine[blk], row_h2_f),
                0.0,
                "blk {blk}: filling (PreFilling) H2 row is frozen — no own +τ"
            );
        }

        // The non-filling control's balance row is a non-zero equality RHS
        // (ζ·base, here base=0 with no PAR ⇒ 0) and an equality (lower == upper).
        assert_eq!(
            rl_c[row_h2_c], ru_c[row_h2_c],
            "control H2 row is an equality"
        );
        // Sanity: the filling build's H2 row is the frozen identity (0).
        assert_eq!(rl_f[row_h2_f], 0.0, "filling H2 frozen RHS == 0");
        assert_eq!(ru_f[row_h2_f], 0.0, "filling H2 frozen RHS == 0");
    }

    /// Resolved offsets for a CHAINED short-circuit probe `H1 → H2 → H3` where H1
    /// AND H2 are BOTH `PreFilling` at the probe stage and H3 is operating.
    struct ChainOffsets {
        zeta: f64,
        h1_idx: usize,
        h2_idx: usize,
        water_row_h1: usize,
        water_row_h2: usize,
        water_row_h3: usize,
        col_storage_in_h1: usize,
        col_storage_in_h2: usize,
        z_h1: usize,
        z_h2: usize,
    }

    /// Build the cascade `H1(id 1, filling) → H2(id 2, filling) → H3(id 3,
    /// operating)` at `stage_id`, with H1's and H2's resolved per-stage withdrawals
    /// set to `withdrawal_h1` / `withdrawal_h2`. At a `PreFilling` `stage_id` both H1
    /// and H2 are `PreFilling` (frozen rows), so H1 must cascade THROUGH H2 to H3.
    /// Returns the CSC, `(row_lower, row_upper)`, and the chained offsets.
    #[allow(clippy::type_complexity)]
    fn build_chained_shortcircuit_case(
        stage_id: i32,
        withdrawal_h1: f64,
        withdrawal_h2: f64,
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        ChainOffsets,
    ) {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), Some(RET_ENTRY_STAGE_ID), true),
                ret_hydro(2, Some(3), Some(RET_ENTRY_STAGE_ID), true),
                ret_hydro(3, None, None, false),
            ],
            Vec::new(),
        );
        let h1_idx = fixtures.hydro_pos[&EntityId(1)];
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .bounds
            .hydro_bounds_mut(h1_idx, 0)
            .water_withdrawal_m3s = withdrawal_h1;
        fixtures
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .water_withdrawal_m3s = withdrawal_h2;
        let ctx = fixtures.make_ctx();
        let stage_index = usize::try_from(stage_id).expect("non-negative");
        let stage = two_block_stage(stage_index, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let h3_idx = fixtures.hydro_pos[&EntityId(3)];
        let offsets = ChainOffsets {
            zeta: layout.zeta,
            h1_idx,
            h2_idx,
            water_row_h1: layout.row_water_balance_start() + h1_idx,
            water_row_h2: layout.row_water_balance_start() + h2_idx,
            water_row_h3: layout.row_water_balance_start() + h3_idx,
            col_storage_in_h1: layout.col_storage_in_start() + h1_idx,
            col_storage_in_h2: layout.col_storage_in_start() + h2_idx,
            z_h1: layout.col_z_inflow_start() + h1_idx,
            z_h2: layout.col_z_inflow_start() + h2_idx,
        };
        (csc, row_lower, row_upper, offsets)
    }

    /// CHAINED short-circuit: cascade `H1 → H2 → H3` with H1 AND H2 BOTH `PreFilling`
    /// at the same stage and H3 operating. Both H1's water (z_{H1}, withdrawal_{H1})
    /// and H2's water (z_{H2}, withdrawal_{H2}) land on H3's row (the first
    /// non-`PreFilling` downstream) — H1 cascades THROUGH the absent H2, never onto
    /// H2's frozen row.
    #[test]
    fn chained_prefilling_routes_both_links_to_first_operating_downstream() {
        let (w_h1, w_h2) = (11.0_f64, 23.0_f64);
        let (csc, _rl, row_upper, off) =
            build_chained_shortcircuit_case(RET_PREFILLING_ID, w_h1, w_h2);

        // (1) Incremental inflow of BOTH links lands on H3's row (−ζ each), and on
        // NO frozen row. H1 routing THROUGH H2 is the chained behaviour: routing to
        // the immediate downstream H2 would corrupt H2's frozen identity.
        assert_eq!(
            csc_at(&csc, off.z_h1, off.water_row_h3),
            -off.zeta,
            "z_{{H1}} carries −ζ on H3's row (H1 cascades THROUGH the PreFilling H2)"
        );
        assert_eq!(
            csc_at(&csc, off.z_h2, off.water_row_h3),
            -off.zeta,
            "z_{{H2}} carries −ζ on H3's row (first non-PreFilling downstream)"
        );
        // Neither z column touches H2's frozen-identity row.
        assert_eq!(
            csc_at(&csc, off.z_h1, off.water_row_h2),
            0.0,
            "z_{{H1}} must NOT land on H2's frozen row (the corruption this guards against)"
        );
        assert_eq!(
            csc_at(&csc, off.z_h2, off.water_row_h2),
            0.0,
            "z_{{H2}} must NOT land on H2's own frozen row"
        );
        // Nor H1's frozen row.
        assert_eq!(
            csc_at(&csc, off.z_h1, off.water_row_h1),
            0.0,
            "z_{{H1}} must NOT land on H1's own frozen row"
        );

        // (2) Withdrawal demand of BOTH links lands on H3's RHS: H3's row_upper
        // drops by ζ·withdrawal_{H1} + ζ·withdrawal_{H2} versus a zero-withdrawal
        // build. Both transfers resolve to H3 (the same short-circuit target),
        // NEVER onto H2's frozen RHS. The expected delta mirrors the code's two
        // independent per-link `-=` subtractions (rows.rs), NOT a single
        // ζ·(w_h1 + w_h2) product, so the FP rounding matches bit-for-bit.
        let (_csc0, _rl0, row_upper0, off0) =
            build_chained_shortcircuit_case(RET_PREFILLING_ID, 0.0, 0.0);
        let delta = off.zeta * w_h1 + off.zeta * w_h2;
        assert_eq!(
            row_upper0[off0.water_row_h3] - row_upper[off.water_row_h3],
            delta,
            "H3's RHS drops by ζ·withdrawal_{{H1}} + ζ·withdrawal_{{H2}}"
        );
    }

    /// CHAINED short-circuit, clean frozen identities: with H1 AND H2 BOTH
    /// `PreFilling`, EACH of H1's and H2's own water rows is exactly the frozen
    /// identity (`+1 v_h`, `−1 v_h_in`, RHS 0) and exactly 2 matrix entries — NOT
    /// corrupted by the other link's routed inflow/upstream/withdrawal terms.
    #[test]
    fn chained_prefilling_keeps_both_frozen_rows_clean() {
        let (csc, row_lower, row_upper, off) =
            build_chained_shortcircuit_case(RET_PREFILLING_ID, 7.0, 13.0);

        // Helper: count the matrix entries on a given water row and assert it is the
        // clean frozen identity (exactly +1 on v_h, −1 on v_h_in, nothing else).
        let assert_clean_frozen = |row: usize, col_v: usize, col_v_in: usize, label: &str| {
            assert_eq!(
                csc_at(&csc, col_v, row),
                1.0,
                "{label}: outgoing storage v carries +1.0 on the frozen row"
            );
            assert_eq!(
                csc_at(&csc, col_v_in, row),
                -1.0,
                "{label}: incoming storage v_in carries −1.0 on the frozen row"
            );
            // Exactly 2 entries on the whole row: +1 v, −1 v_in, nothing else.
            let n_entries: usize = (0..csc.0.len().saturating_sub(1))
                .map(|col| {
                    let start = usize::try_from(csc.0[col]).unwrap();
                    let end = usize::try_from(csc.0[col + 1]).unwrap();
                    csc.1[start..end]
                        .iter()
                        .filter(|&&r| usize::try_from(r).is_ok_and(|r| r == row))
                        .count()
                })
                .sum();
            assert_eq!(
                n_entries, 2,
                "{label}: frozen-identity row must have EXACTLY 2 entries \
                 (+1 v, −1 v_in) — found {n_entries}, indicating corruption by a \
                 chained link's routed terms"
            );
            assert_eq!(row_lower[row], 0.0, "{label}: frozen RHS lower == 0");
            assert_eq!(row_upper[row], 0.0, "{label}: frozen RHS upper == 0");
        };

        // H2's row clean (NOT corrupted by H1's terms — the bug this fix removes).
        assert_clean_frozen(off.water_row_h2, off.h2_idx, off.col_storage_in_h2, "H2");
        // H1's row likewise clean.
        assert_clean_frozen(off.water_row_h1, off.h1_idx, off.col_storage_in_h1, "H1");
    }

    /// CHAIN-to-sink: a cascade `H1 → H2` where H1 AND H2 are BOTH `PreFilling` and
    /// H2 is terminal (no downstream). The whole chain is `PreFilling`, so the
    /// resolved target is None (SINK): H1's and H2's water exits the system, and
    /// NEITHER frozen row is corrupted.
    #[test]
    fn chained_prefilling_all_the_way_down_is_sink_and_keeps_frozen_rows_clean() {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), Some(RET_ENTRY_STAGE_ID), true),
                ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
            ],
            Vec::new(),
        );
        let h1_idx = fixtures.hydro_pos[&EntityId(1)];
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .bounds
            .hydro_bounds_mut(h1_idx, 0)
            .water_withdrawal_m3s = 8.0;
        fixtures
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .water_withdrawal_m3s = 19.0;
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(usize::try_from(RET_PREFILLING_ID).unwrap(), [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let z_h1 = layout.col_z_inflow_start() + h1_idx;
        let z_h2 = layout.col_z_inflow_start() + h2_idx;

        // Both links' inflow exits the system: neither z column lands on ANY water
        // row (no non-PreFilling downstream exists to receive it).
        for h in 0..layout.n_h {
            let r = layout.row_water_balance_start() + h;
            assert_eq!(
                csc_at(&csc, z_h1, r),
                0.0,
                "z_{{H1}} must not land on any water row in the all-PreFilling sink (row {r})"
            );
            assert_eq!(
                csc_at(&csc, z_h2, r),
                0.0,
                "z_{{H2}} must not land on any water row in the all-PreFilling sink (row {r})"
            );
        }

        // Both frozen rows are clean (exactly +1 v, −1 v_in, RHS 0) and no
        // withdrawal demand was folded onto any frozen RHS (the sink transfers
        // nothing).
        for (h_idx, label) in [(h1_idx, "H1"), (h2_idx, "H2")] {
            let row = layout.row_water_balance_start() + h_idx;
            assert_eq!(csc_at(&csc, h_idx, row), 1.0, "{label}: v +1.0");
            assert_eq!(
                csc_at(&csc, layout.col_storage_in_start() + h_idx, row),
                -1.0,
                "{label}: v_in −1.0"
            );
            assert_eq!(
                row_lower[row], 0.0,
                "{label}: frozen RHS 0 (no withdrawal folded in)"
            );
            assert_eq!(row_upper[row], 0.0, "{label}: frozen RHS 0");
        }
    }
}
