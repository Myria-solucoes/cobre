//! Shared noise transformation functions for the LP patching hot path.
//!
//! Both [`transform_inflow_noise`] and [`transform_load_noise`] convert raw
//! PAR(p) or normal noise samples into the patched RHS values that are written
//! into the stage LP before each solve.  Extracting them here eliminates the
//! class of bugs where one call site receives a fix and others are forgotten.

use cobre_core::temporal::StageLagTransition;
use cobre_stochastic::{StochasticContext, evaluate_par_batch, solve_par_noise_batch};

use crate::{
    InflowNonNegativityMethod,
    context::{StageContext, TrainingContext},
    workspace::ScratchBuffers,
};

/// Compute effective (possibly clamped) eta for each hydro.
///
/// When truncation is active and any PAR(p) inflow is negative, clamps each
/// negative hydro's eta upward to the floor that produces zero inflow.
/// Otherwise writes raw eta unchanged.
///
/// For non-truncation methods (`None`, `Penalty`), writes raw eta directly.
pub(crate) fn compute_effective_eta(
    raw_noise: &[f64],
    n_hydros: usize,
    inflow_method: InflowNonNegativityMethod,
    par_inflows: &[f64],
    eta_floor: &[f64],
    effective_eta: &mut Vec<f64>,
) {
    effective_eta.clear();

    match inflow_method {
        InflowNonNegativityMethod::Truncation
        | InflowNonNegativityMethod::TruncationWithPenalty => {
            let has_negative = par_inflows.iter().take(n_hydros).any(|&a| a < 0.0);
            for h in 0..n_hydros {
                let eta = raw_noise[h];
                let clamped = if has_negative && par_inflows[h] < 0.0 {
                    eta.max(eta_floor[h])
                } else {
                    eta
                };
                effective_eta.push(clamped);
            }
        }
        InflowNonNegativityMethod::None | InflowNonNegativityMethod::Penalty => {
            effective_eta.extend_from_slice(&raw_noise[..n_hydros]);
        }
    }
}

/// Transform raw inflow noise `η` into patched water-balance RHS values.
///
/// Writes `noise_buf[h] = base_rhs + noise_scale[stage * n_hydros + h] * η_effective[h]`
/// for each hydro, where `η_effective` is clamped when truncation is active
/// and negative inflow would occur.
///
/// No heap allocations; all scratch work uses pre-allocated buffers from `scratch`.
pub(crate) fn transform_inflow_noise(
    raw_noise: &[f64],
    stage: usize,
    current_state: &[f64],
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    scratch: &mut ScratchBuffers,
) {
    let n_hydros = ctx.n_hydros;
    let stage_offset = stage * n_hydros;
    let base_row = ctx.base_rows[stage];
    let template_row_lower = &ctx.templates[stage].row_lower;
    let noise_scale = ctx.noise_scale;
    let inflow_method = training_ctx.inflow_method;
    let stochastic = training_ctx.stochastic;
    let indexer = training_ctx.indexer;

    scratch.noise_buf.clear();
    scratch.z_inflow_rhs_buf.clear();

    // Pre-fetch PAR parameters for z-inflow RHS computation.
    let par_lp = stochastic.par();
    let has_par = par_lp.n_stages() > 0 && par_lp.n_hydros() == n_hydros;

    // Precompute PAR inflows and eta floor for truncation methods.
    // For None/Penalty, par_inflow_buf and eta_floor_buf are unused by
    // compute_effective_eta (it copies raw eta directly).
    match inflow_method {
        InflowNonNegativityMethod::Truncation
        | InflowNonNegativityMethod::TruncationWithPenalty => {
            let max_order = indexer.max_par_order;
            let lag_len = max_order * n_hydros;
            scratch.lag_matrix_buf.clear();
            scratch.lag_matrix_buf.resize(lag_len, 0.0);
            for h in 0..n_hydros {
                for l in 0..max_order {
                    scratch.lag_matrix_buf[l * n_hydros + h] =
                        current_state[indexer.inflow_lags.start + l * n_hydros + h];
                }
            }

            scratch.par_inflow_buf.clear();
            scratch.par_inflow_buf.resize(n_hydros, 0.0);
            // `evaluate_par_batch` operates on the n_hydros PAR series only;
            // `raw_noise` carries the full noise dimension (hydros + load buses
            // + NCS). Slice to the hydro prefix to match the contract, exactly
            // as the sibling `compute_effective_eta` call below does.
            evaluate_par_batch(
                par_lp,
                stage,
                &scratch.lag_matrix_buf,
                &raw_noise[..n_hydros],
                &mut scratch.par_inflow_buf,
            );

            let has_negative = scratch.par_inflow_buf.iter().any(|&a| a < 0.0);
            if has_negative {
                scratch.eta_floor_buf.clear();
                scratch.eta_floor_buf.resize(n_hydros, f64::NEG_INFINITY);
                let zero_targets = &scratch.zero_targets_buf[..n_hydros];
                solve_par_noise_batch(
                    par_lp,
                    stage,
                    &scratch.lag_matrix_buf,
                    zero_targets,
                    &mut scratch.eta_floor_buf,
                );
            }
        }
        InflowNonNegativityMethod::None | InflowNonNegativityMethod::Penalty => {}
    }

    // Unified: compute effective eta then build RHS for all methods.
    compute_effective_eta(
        raw_noise,
        n_hydros,
        *inflow_method,
        &scratch.par_inflow_buf,
        &scratch.eta_floor_buf,
        &mut scratch.effective_eta_buf,
    );

    for (h, &eta_eff) in scratch.effective_eta_buf.iter().enumerate() {
        let base_rhs = template_row_lower[base_row + h];
        scratch
            .noise_buf
            .push(base_rhs + noise_scale[stage_offset + h] * eta_eff);

        // Z-inflow RHS: base + sigma * eta_effective (m3/s, no zeta, no withdrawal).
        if has_par {
            let base = par_lp.deterministic_base(stage, h);
            let sigma = par_lp.sigma(stage, h);
            scratch.z_inflow_rhs_buf.push(base + sigma * eta_eff);
        } else {
            scratch.z_inflow_rhs_buf.push(0.0);
        }
    }
}

/// Shift the lag portion of the outgoing state vector using realized inflow.
///
/// Shifts older lags backward, with newest lag = realized inflow from LP primal.
/// No-op when `max_par_order == 0`. Zero heap allocations.
#[cfg(test)]
pub(crate) fn shift_lag_state(
    state: &mut [f64],
    incoming_lags: &[f64],
    unscaled_primal: &[f64],
    indexer: &crate::indexer::StageIndexer,
) {
    let n_h = indexer.hydro_count;
    let l_max = indexer.max_par_order;
    if l_max == 0 || n_h == 0 {
        return; // No lags to shift
    }
    let lag_start = indexer.inflow_lags.start;
    for h in 0..n_h {
        let z_t_h = unscaled_primal[indexer.z_inflow.start + h];
        // Shift older lags down (read from incoming_lags to avoid aliasing).
        // incoming_lags is in lag-major layout: incoming_lags[lag * n_h + h].
        for lag in (1..l_max).rev() {
            state[lag_start + lag * n_h + h] = incoming_lags[(lag - 1) * n_h + h];
        }
        // Newest lag = realized inflow from z_h primal.
        state[lag_start + h] = z_t_h;
    }
}

/// Shift the lag portion of the outgoing state vector using pre-computed monthly inflows.
///
/// Private helper used by [`accumulate_and_shift_lag_state`] when finalizing a lag
/// period. Takes a `monthly_inflows` slice of length `hydro_count` directly, avoiding
/// the need to read z-inflow offsets from a full primal buffer.
///
/// The caller guarantees `monthly_inflows.len() >= indexer.hydro_count`.
/// No heap allocations.
fn shift_lag_state_from_inflows(
    state: &mut [f64],
    incoming_lags: &[f64],
    monthly_inflows: &[f64],
    indexer: &crate::indexer::StageIndexer,
) {
    let n_h = indexer.hydro_count;
    let l_max = indexer.max_par_order;
    let lag_start = indexer.inflow_lags.start;
    for h in 0..n_h {
        // Shift older lags down (read from incoming_lags to avoid aliasing).
        // incoming_lags is in lag-major layout: incoming_lags[lag * n_h + h].
        for lag in (1..l_max).rev() {
            state[lag_start + lag * n_h + h] = incoming_lags[(lag - 1) * n_h + h];
        }
        // Newest lag = weighted-average monthly inflow for this period.
        state[lag_start + h] = monthly_inflows[h];
    }
}

/// Shift the anticipated-state ring buffer and write the new decision.
///
/// For each anticipated thermal `i` with lead time `K_i`, performs:
///
/// - **Slots `0..K_i-1`**: write `incoming_anticipated[(slot+1) * n_ant + i]`
///   into `state[anticipated_state.start + slot * n_ant + i]` (shift down by
///   one).  Slot 0 of the outgoing state holds what was slot 1 of the incoming
///   state (the second-oldest commitment horizon).  The incoming slot 0
///   (the oldest) falls out, consumed by the next-stage fishing constraint.
///   At stage 0, slot 0 of the incoming state may carry a non-zero pre-horizon
///   seed from `past_anticipated_commitments.values_mw[0]`; this is the normal
///   case when the study has pre-horizon commitments and the seeding block in
///   `setup/mod.rs` has populated the initial state.
/// - **Slot `K_i - 1`**: write `unscaled_primal[anticipated_decision.start + i]`
///   — the freshly-made commitment from this stage's solve, in MW.
/// - **Slots `K_i..k_max`**: write `0.0` (deterministic padding for plants
///   whose lead time is shorter than `k_max`).
///
/// No-op when `n_anticipated == 0` or `k_max == 0`.  Zero heap allocations.
///
/// ## Inactive plants (horizon boundary)
///
/// This function does not gate per-plant on the decision-active predicate
/// (`stage_idx + K_i < n_stages`, see [`StageIndexer::anticipated_decision_active_at_stage`]).
/// Inactive plants — those at stages where the LP did not emit a real
/// decision column — rely on the LP builder pinning their decision column
/// bounds to `[0.0, 0.0]` (matrix.rs:`fill_anticipated_decision_columns`).
/// The unscaled primal at that column is therefore `0.0`, and writing `0.0`
/// into slot `K_i - 1` is the correct ring-buffer transition: no new
/// commitment was placed, so slot `K_i - 1` should hold zero. If a future
/// change relaxes the `[0, 0]` invariant for inactive columns, this
/// function must be updated to gate on the activation predicate.
///
/// [`StageIndexer::anticipated_decision_active_at_stage`]: crate::indexer::StageIndexer::anticipated_decision_active_at_stage
///
/// # Panics (debug only)
///
/// Panics in debug builds if `incoming_anticipated.len() != n_ant * k_max`,
/// or if `unscaled_primal` is shorter than `anticipated_decision.end`, or if
/// any `K_i == 0`.
pub(crate) fn shift_anticipated_state(
    state: &mut [f64],
    incoming_anticipated: &[f64],
    unscaled_primal: &[f64],
    indexer: &crate::indexer::StageIndexer,
) {
    let n_ant = indexer.n_anticipated;
    let k_max = indexer.k_max;
    if n_ant == 0 || k_max == 0 {
        return;
    }
    debug_assert_eq!(
        incoming_anticipated.len(),
        n_ant * k_max,
        "incoming_anticipated length mismatch: expected {}, got {}",
        n_ant * k_max,
        incoming_anticipated.len(),
    );
    debug_assert!(
        unscaled_primal.len() >= indexer.anticipated_decision.end,
        "unscaled_primal too short for anticipated_decision range",
    );
    let state_start = indexer.anticipated_state.start;
    let decision_start = indexer.anticipated_decision.start;
    for (plant, &k_i) in indexer.anticipated_lead_stages.iter().enumerate() {
        debug_assert!(
            k_i >= 1,
            "K_i must be >= 1 (enforced by anticipation validation)"
        );
        // Shift: slot s ← incoming slot s+1, for s in 0..K_i - 1.
        for slot in 0..(k_i - 1) {
            let dst = state_start + slot * n_ant + plant;
            let src = (slot + 1) * n_ant + plant;
            state[dst] = incoming_anticipated[src];
        }
        // Write new decision at slot K_i - 1.
        let newest = state_start + (k_i - 1) * n_ant + plant;
        state[newest] = unscaled_primal[decision_start + plant];
        // Zero padding for slots k_i..k_max (plants with K_i < k_max).
        for slot in k_i..k_max {
            let dst = state_start + slot * n_ant + plant;
            state[dst] = 0.0;
        }
    }
}

/// Mutable primary lag-accumulation buffers threaded through the hot path.
///
/// Groups `lag_accumulator` and `lag_weight_accum` so that
/// [`accumulate_and_shift_lag_state`] stays within the 7-parameter budget.
pub(crate) struct LagAccumState<'a> {
    /// Weighted-sum buffer, length `>= hydro_count`.
    /// Holds the partial sum for the current primary lag period.
    pub accumulator: &'a mut [f64],
    /// Total weight accumulated so far in the current primary lag period.
    pub weight_accum: &'a mut f64,
}

/// Mutable downstream accumulation buffers threaded through the hot path.
///
/// Groups the five downstream parameters so that
/// [`accumulate_and_shift_lag_state`] stays within the 7-parameter budget.
///
/// For uniform-resolution studies pass `accumulator: &mut []` (empty slice);
/// all downstream code paths short-circuit on `accumulator.is_empty()`.
pub(crate) struct DownstreamAccumState<'a> {
    /// Weighted-sum accumulator buffer, length `>= hydro_count`.
    /// Empty slice when `par_order == 0` — all downstream paths skip on `is_empty()`.
    pub accumulator: &'a mut [f64],
    /// Accumulated weight for the current downstream lag period.
    pub weight_accum: &'a mut f64,
    /// Slot-major ring buffer storing completed downstream lags.
    /// Layout: `completed_lags[slot * n_h + h]`, slot 0 = oldest quarter.
    /// Length `n_h * par_order` when `par_order > 0`, or empty.
    pub completed_lags: &'a mut [f64],
    /// Number of completed downstream lags currently stored in the ring buffer
    /// (capped at `par_order`).
    pub n_completed: &'a mut usize,
    /// PAR order for the downstream (coarser) resolution.  `0` for
    /// uniform-resolution studies; all downstream code paths are skipped.
    pub par_order: usize,
}

/// Accumulate this stage's inflow and, when a lag period finalizes, shift the lag state.
///
/// Replaces the direct `shift_lag_state` call for multi-resolution studies where
/// stages may be shorter than the lag granularity (for example, weekly stages feeding
/// a monthly lag slot).  The three-step logic:
///
/// 1. **Accumulate**: add `z_inflow[h] * stage_lag.accumulate_weight` to
///    `lag_accumulator[h]` and `stage_lag.accumulate_weight` to `*lag_weight_accum`.
/// 2. **Finalize** (only when `stage_lag.finalize_period && *lag_weight_accum > 0.0`):
///    divide the accumulator by the total weight to get the weighted average, call
///    [`shift_lag_state_from_inflows`] with those averages, then reset the accumulator.
///    If `stage_lag.spillover_weight > 0.0`, seed the next period immediately.
/// 3. **Non-finalizing stages**: `state` is left untouched (lags frozen).
///
/// For the monthly identity case (`accumulate_weight=1.0, spillover_weight=0.0,
/// finalize_period=true`) the function produces bit-for-bit identical results to
/// `shift_lag_state`.
///
/// **Zero heap allocation.** All scratch work is performed in `lag.accumulator`,
/// which is overwritten with the monthly averages during finalization before being
/// reset.
///
/// # Panics (debug only)
///
/// Panics in debug builds if `lag.accumulator.len() < indexer.hydro_count`.
///
/// # Downstream accumulation (multi-resolution studies)
///
/// When `ds.accumulator` is non-empty (i.e., `ds.par_order > 0`), the function
/// additionally maintains a coarser-resolution ring buffer in parallel with the
/// primary accumulation. See [`DownstreamAccumState`] for field docs.
///
/// For uniform-resolution studies pass `ds.accumulator = &mut []` (empty slice).
/// All downstream code paths are skipped via a single `is_empty()` guard,
/// producing zero overhead.
///
/// # Anticipated-thermal state
///
/// This function does NOT shift the `anticipated_state` ring buffer.
/// Anticipated commitments advance once per LP stage regardless of
/// `StageLagTransition.finalize_period` or `accumulate_downstream`,
/// because the decision cadence equals the LP-stage cadence. The
/// companion function `shift_anticipated_state` performs the shift; the
/// stage driver (currently `forward.rs::run_forward_stage`) calls both
/// functions in sequence, once per stage.
pub(crate) fn accumulate_and_shift_lag_state(
    state: &mut [f64],
    incoming_lags: &[f64],
    unscaled_primal: &[f64],
    indexer: &crate::indexer::StageIndexer,
    stage_lag: &StageLagTransition,
    lag: &mut LagAccumState<'_>,
    ds: &mut DownstreamAccumState<'_>,
) {
    let n_h = indexer.hydro_count;
    let l_max = indexer.max_par_order;
    if l_max == 0 || n_h == 0 {
        return; // No lags to shift — identical early-return guard as shift_lag_state
    }

    debug_assert!(
        lag.accumulator.len() >= n_h,
        "lag_accumulator too short: {} < {n_h}",
        lag.accumulator.len()
    );

    let z_start = indexer.z_inflow.start;

    // ── Step 1: Primary accumulate ────────────────────────────────────────────
    // Must happen unconditionally before finalize check, so this stage's
    // contribution is included in the average.
    let w = stage_lag.accumulate_weight;
    for h in 0..n_h {
        lag.accumulator[h] += unscaled_primal[z_start + h] * w;
    }
    *lag.weight_accum += w;

    // ── Step 1b: Downstream accumulate (multi-resolution only) ───────────────
    // Guard on empty slice: zero overhead for uniform studies.
    if !ds.accumulator.is_empty() && stage_lag.accumulate_downstream {
        debug_assert!(
            ds.accumulator.len() >= n_h,
            "downstream_accumulator too short: {} < {n_h}",
            ds.accumulator.len()
        );
        debug_assert!(
            ds.par_order == 0 || ds.completed_lags.len() >= n_h * ds.par_order,
            "downstream_completed_lags too short: {} < {}",
            ds.completed_lags.len(),
            n_h * ds.par_order
        );

        let dw = stage_lag.downstream_accumulate_weight;
        for h in 0..n_h {
            ds.accumulator[h] += unscaled_primal[z_start + h] * dw;
        }
        *ds.weight_accum += dw;

        if stage_lag.downstream_finalize && *ds.weight_accum > 0.0 {
            let inv = 1.0 / *ds.weight_accum;
            for v in &mut ds.accumulator[..n_h] {
                *v *= inv;
            }

            // Push weighted average into the ring buffer.
            // Slot 0 = oldest completed quarter; slots fill in order.
            let slot = (*ds.n_completed).min(ds.par_order.saturating_sub(1));
            let offset = slot * n_h;
            ds.completed_lags[offset..offset + n_h].copy_from_slice(&ds.accumulator[..n_h]);
            *ds.n_completed = (*ds.n_completed + 1).min(ds.par_order);

            // Reset downstream accumulator, optionally seeding spillover.
            if stage_lag.downstream_spillover_weight > 0.0 {
                let dsw = stage_lag.downstream_spillover_weight;
                for h in 0..n_h {
                    ds.accumulator[h] = unscaled_primal[z_start + h] * dsw;
                }
                *ds.weight_accum = dsw;
            } else {
                ds.accumulator[..n_h].fill(0.0);
                *ds.weight_accum = 0.0;
            }
        }
    }

    // ── Rebuild from downstream (transition stage) ────────────────────────────
    // At the first quarterly stage, overwrite the primary lag state with the
    // completed quarterly lags from the downstream ring buffer, then return
    // (skipping the primary finalize which would overwrite with monthly data).
    if stage_lag.rebuild_from_downstream && *ds.n_completed > 0 {
        let n_fill = (*ds.n_completed).min(l_max);
        for lag_idx in 0..n_fill {
            // Ring buffer is slot-major. Slot 0 = oldest. Newest lag = slot n_completed-1.
            // state lag layout: state[lag_start + lag_idx * n_h + h]
            //   lag_idx=0 → newest (slot n_completed-1), lag_idx=1 → second newest, …
            let src_slot = *ds.n_completed - 1 - lag_idx;
            let src_offset = src_slot * n_h;
            let dst_offset = indexer.inflow_lags.start + lag_idx * n_h;
            state[dst_offset..dst_offset + n_h]
                .copy_from_slice(&ds.completed_lags[src_offset..src_offset + n_h]);
        }
        // Reset downstream state — all completed quarterly lags consumed.
        *ds.n_completed = 0;
        ds.completed_lags.fill(0.0);
        if !ds.accumulator.is_empty() {
            ds.accumulator[..n_h].fill(0.0);
        }
        *ds.weight_accum = 0.0;
        return; // Lag state fully rebuilt; skip primary finalize.
    }

    // ── Step 2: Primary finalize (if this stage closes a lag period) ──────────
    if stage_lag.finalize_period && *lag.weight_accum > 0.0 {
        // Overwrite lag.accumulator[h] with the weighted-average monthly inflow.
        // The original accumulated sum is not needed after this point.
        let inv = 1.0 / *lag.weight_accum;
        for v in &mut lag.accumulator[..n_h] {
            *v *= inv;
        }

        shift_lag_state_from_inflows(state, incoming_lags, lag.accumulator, indexer);

        // ── Reset accumulator, then optionally seed spillover ─────────────────
        // Spillover uses the RAW z_inflow (not the averaged value), because it
        // is this stage's contribution to the NEXT lag period.
        if stage_lag.spillover_weight > 0.0 {
            let sw = stage_lag.spillover_weight;
            for h in 0..n_h {
                lag.accumulator[h] = unscaled_primal[z_start + h] * sw;
            }
            *lag.weight_accum = sw;
        } else {
            lag.accumulator[..n_h].fill(0.0);
            *lag.weight_accum = 0.0;
        }
    }
    // ── Step 3: Non-finalizing stage ─────────────────────────────────────────
    // Lags frozen — state is not modified. Accumulation already applied above.
}

/// Transform raw load noise `η` into patched load-balance RHS values.
///
/// Writes `(mean + std * η).max(0.0) * block_factor` for each load bus and block.
/// Clamped to zero so load demand is never negative. No heap allocations.
pub(crate) fn transform_load_noise(
    raw_noise: &[f64],
    n_hydros: usize,
    n_load_buses: usize,
    stochastic: &StochasticContext,
    stage: usize,
    block_count: usize,
    load_rhs_buf: &mut Vec<f64>,
) {
    load_rhs_buf.clear();
    if n_load_buses == 0 {
        return;
    }
    let load_lp = stochastic.normal();
    for lb_idx in 0..n_load_buses {
        let eta = raw_noise[n_hydros + lb_idx];
        let mean = load_lp.mean(stage, lb_idx);
        let std = load_lp.std(stage, lb_idx);
        let realization = (mean + std * eta).max(0.0);
        for blk in 0..block_count {
            let factor = load_lp.block_factor(stage, lb_idx, blk);
            load_rhs_buf.push(realization * factor);
        }
    }
}

/// Noise-vector offsets needed to locate the NCS slice within the raw noise array.
///
/// The shared raw noise vector is laid out as `[hydro noise | load noise | NCS noise]`.
/// `n_hydros + n_load_buses` gives the start offset of the NCS section.
pub(crate) struct NcsNoiseOffsets {
    /// Number of hydro entries that precede the load slice.
    pub n_hydros: usize,
    /// Number of load-bus entries that precede the NCS slice.
    pub n_load_buses: usize,
}

/// Transform raw NCS noise into per-block column lower/upper bound values.
///
/// For each NCS entity computes a per-scenario availability ratio
/// `α = clamp(mean + std · η, 0, 1)`, then writes both column bounds:
///
/// - `ncs_col_upper_buf[blk] = max_gen · α · block_factor`
/// - `ncs_col_lower_buf[blk] = if allow_curtailment { 0 } else { upper }`
///
/// When `allow_curtailment[i] == false` the lower bound matches the upper
/// bound so the source is dispatched at exactly the realized availability
/// on every scenario — the **must-run** regime for non-simulated aggregate
/// generation pre-netted from load. When
/// `allow_curtailment[i] == true` the lower bound is zero and the LP can
/// curtail at `curtailment_cost` per `MWh`.
///
/// The `offsets` bundle encodes the raw-noise vector layout (`n_hydros +
/// n_load_buses` gives the start of the NCS slice).
///
/// # Panics
///
/// Panics in debug builds when `ncs_allow_curtailment.len() !=
/// ncs_max_gen.len()` or either slice is shorter than
/// `stochastic.n_stochastic_ncs()`.
pub(crate) fn transform_ncs_noise(
    raw_noise: &[f64],
    offsets: &NcsNoiseOffsets,
    stochastic: &StochasticContext,
    stage: usize,
    block_count: usize,
    ncs_max_gen: &[f64],
    ncs_allow_curtailment: &[bool],
    ncs_col_lower_buf: &mut Vec<f64>,
    ncs_col_upper_buf: &mut Vec<f64>,
) {
    let n_stochastic_ncs = stochastic.n_stochastic_ncs();
    ncs_col_upper_buf.clear();
    ncs_col_lower_buf.clear();
    if n_stochastic_ncs == 0 {
        return;
    }
    debug_assert_eq!(
        ncs_allow_curtailment.len(),
        ncs_max_gen.len(),
        "ncs_allow_curtailment and ncs_max_gen must have matching length",
    );
    let ncs_lp = stochastic.ncs_normal();
    let ncs_noise_start = offsets.n_hydros + offsets.n_load_buses;
    for ncs_idx in 0..n_stochastic_ncs {
        let eta = raw_noise[ncs_noise_start + ncs_idx];
        let mean = ncs_lp.mean(stage, ncs_idx);
        let std = ncs_lp.std(stage, ncs_idx);
        let max_gen = ncs_max_gen[ncs_idx];
        let availability_ratio = (mean + std * eta).clamp(0.0, 1.0);
        let realization = max_gen * availability_ratio;
        let allow_curtailment = ncs_allow_curtailment[ncs_idx];
        for blk in 0..block_count {
            let factor = ncs_lp.block_factor(stage, ncs_idx, blk);
            let upper = realization * factor;
            ncs_col_upper_buf.push(upper);
            ncs_col_lower_buf.push(if allow_curtailment { 0.0 } else { upper });
        }
    }
}

/// Rebuild the NCS column-index buffer for one stage's **active** NCS columns.
///
/// `slot_to_local[slot]` is `Some(ncs_local)` when the stochastic NCS at `slot`
/// (id-sorted `StochasticContext::ncs_entity_ids` order) is active at this stage,
/// `None` otherwise. The pushed column index strides by `ncs_local` — the slot's
/// position within `active_ncs_indices[stage]`, the order the LP allocated NCS
/// columns — never by the raw stochastic slot. Striding by the slot would address
/// `n_stochastic_ncs` columns; at a strict-subset stage the LP allocates only
/// `(active stochastic NCS) * block_count` NCS columns, so a slot-strided index
/// overruns the NCS column block and the solver rejects the set-bounds call. An
/// inactive slot (`None`) contributes no index.
///
/// The buffer is cleared then refilled; callers rebuild it lazily (only when the
/// expected active length changes — see the patch sites). No heap allocation
/// beyond the initial capacity growth.
pub(crate) fn build_active_ncs_col_indices(
    slot_to_local: &[Option<usize>],
    ncs_col_start: usize,
    block_count: usize,
    indices_out: &mut Vec<usize>,
) {
    indices_out.clear();
    for &maybe_local in slot_to_local {
        if let Some(ncs_local) = maybe_local {
            for blk in 0..block_count {
                indices_out.push(ncs_col_start + ncs_local * block_count + blk);
            }
        }
    }
}

/// Gather the **active** slots' NCS column bounds from the full slot-order buffers.
///
/// `lower_src` / `upper_src` are the verbatim `transform_ncs_noise` outputs — one
/// `block_count`-wide block per stochastic slot, in slot order, **including
/// inactive slots**. This copies only the active slots' blocks (those with
/// `Some(ncs_local)` in `slot_to_local`) into `lower_out` / `upper_out`, in slot
/// order, so the gathered buffers run parallel to the index buffer built by
/// [`build_active_ncs_col_indices`] and the set-bounds call receives index/lower/
/// upper buffers of equal length. Passing `lower_src` / `upper_src` verbatim on a
/// subset stage would feed inactive slots' bounds into the call and over-length
/// it. When every slot is active the gather reproduces the source buffers exactly
/// (hash-neutral for the all-active case). Cleared then refilled each call; the
/// bounds change every opening, so unlike the index buffer this cannot be cached.
pub(crate) fn gather_active_ncs_bounds(
    slot_to_local: &[Option<usize>],
    block_count: usize,
    lower_src: &[f64],
    upper_src: &[f64],
    lower_out: &mut Vec<f64>,
    upper_out: &mut Vec<f64>,
) {
    lower_out.clear();
    upper_out.clear();
    for (slot, &maybe_local) in slot_to_local.iter().enumerate() {
        if maybe_local.is_some() {
            let base = slot * block_count;
            lower_out.extend_from_slice(&lower_src[base..base + block_count]);
            upper_out.extend_from_slice(&upper_src[base..base + block_count]);
        }
    }
}

/// Count the active stochastic NCS slots at one stage.
///
/// The active-subset column/bound buffers have length `count * block_count`; the
/// patch sites key their lazy index-buffer rebuild on this length, not on
/// `n_stochastic_ncs`.
#[inline]
pub(crate) fn active_ncs_slot_count(slot_to_local: &[Option<usize>]) -> usize {
    slot_to_local.iter().filter(|s| s.is_some()).count()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::erasing_op,
    clippy::identity_op,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
        LoadModel, SamplingScheme,
    };
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
    use cobre_solver::StageTemplate;
    use cobre_stochastic::StochasticContext;
    use cobre_stochastic::context::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};
    use std::collections::BTreeMap;

    use crate::{
        context::{StageContext, TrainingContext},
        horizon_mode::HorizonMode,
        indexer::StageIndexer,
        inflow_method::InflowNonNegativityMethod,
        noise::{
            active_ncs_slot_count, build_active_ncs_col_indices, compute_effective_eta,
            gather_active_ncs_bounds, shift_anticipated_state, shift_lag_state,
            transform_inflow_noise, transform_load_noise,
        },
        workspace::ScratchBuffers,
    };

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Build a minimal `StageTemplate` with just `row_lower` populated.
    ///
    /// Only `row_lower` is accessed by `transform_inflow_noise`.  All other
    /// fields are set to their zero/empty defaults.
    fn make_minimal_template(row_lower: Vec<f64>) -> StageTemplate {
        let n = row_lower.len();
        StageTemplate {
            num_cols: 0,
            num_rows: n,
            num_nz: 0,
            col_starts: vec![0_i32],
            row_indices: vec![],
            values: vec![],
            col_lower: vec![],
            col_upper: vec![],
            objective: vec![],
            row_lower,
            row_upper: vec![0.0; n],
            n_transfer: 0,
            n_dual_relevant: 0,
            n_hydro: 0,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
            n_state: 0,
        }
    }

    /// Build a `ScratchBuffers` with the given pre-filled `zero_targets_buf`.
    fn make_scratch(n_hydros: usize) -> ScratchBuffers {
        ScratchBuffers {
            noise_buf: Vec::with_capacity(n_hydros),
            inflow_m3s_buf: Vec::new(),
            lag_matrix_buf: Vec::new(),
            par_inflow_buf: Vec::new(),
            eta_floor_buf: Vec::new(),
            zero_targets_buf: vec![0.0_f64; n_hydros],
            ncs_col_upper_buf: Vec::new(),
            ncs_col_lower_buf: Vec::new(),
            ncs_col_indices_buf: Vec::new(),
            ncs_col_lower_active_buf: Vec::new(),
            ncs_col_upper_active_buf: Vec::new(),
            ncs_col_upper_extract_buf: Vec::new(),
            load_rhs_buf: Vec::new(),
            row_lower_buf: Vec::new(),
            z_inflow_rhs_buf: Vec::new(),
            effective_eta_buf: Vec::with_capacity(n_hydros),
            unscaled_primal: Vec::new(),
            unscaled_dual: Vec::new(),
            lag_accumulator: Vec::new(),
            lag_weight_accum: 0.0,
            downstream_accumulator: Vec::new(),
            downstream_weight_accum: 0.0,
            downstream_completed_lags: Vec::new(),
            downstream_n_completed: 0,
            recon_slot_lookup: Vec::new(),
            trajectory_costs_buf: Vec::new(),
            raw_noise_buf: Vec::new(),
            perm_scratch: Vec::new(),
            anticipated_state_buf: Vec::new(),
        }
    }

    /// One-hydro, one-stage `StochasticContext` with AR(0) (white noise).
    ///
    /// PAR(0): inflow = `std_m3s` * eta (no autoregressive term).
    /// With `mean_m3s = 0.0` and `std_m3s = 1.0`, inflow = eta.
    #[allow(clippy::too_many_lines)]
    fn make_one_hydro_stochastic(n_stages: usize) -> StochasticContext {
        let bus = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let hydro = Hydro {
            id: EntityId(1),
            name: "H1".to_string(),
            bus_id: EntityId(0),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
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
            },
        };

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let make_stage = |idx: usize| Stage {
            index: idx,
            id: idx as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        };

        let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|idx| InflowModel {
                hydro_id: EntityId(1),
                stage_id: idx as i32,
                mean_m3s: 0.0,
                std_m3s: 1.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
                    entities: vec![CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(1),
                    }],
                    matrix: vec![vec![1.0]],
                }],
            },
        );
        let correlation = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(correlation)
            .build()
            .unwrap();

        build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap()
    }

    /// One-hydro, one-load-bus, n-stage `StochasticContext`.
    ///
    /// Load bus has `mean_mw` and `std_mw`, one block per stage.
    #[allow(clippy::too_many_lines)]
    fn make_stochastic_with_load(n_stages: usize, mean_mw: f64, std_mw: f64) -> StochasticContext {
        let bus0 = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let bus1 = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let hydro = Hydro {
            id: EntityId(10),
            name: "H10".to_string(),
            bus_id: EntityId(0),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
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
            },
        };

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let make_stage = |idx: usize| Stage {
            index: idx,
            id: idx as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        };

        let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|idx| InflowModel {
                hydro_id: EntityId(10),
                stage_id: idx as i32,
                mean_m3s: 0.0,
                std_m3s: 1.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|idx| LoadModel {
                bus_id: EntityId(1),
                stage_id: idx as i32,
                mean_mw,
                std_mw,
            })
            .collect();

        let correlation = CorrelationModel {
            method: "spectral".to_string(),
            profiles: BTreeMap::new(),
            schedule: vec![],
        };

        let system = SystemBuilder::new()
            .buses(vec![bus0, bus1])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .correlation(correlation)
            .build()
            .unwrap();

        build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap()
    }

    // ── transform_inflow_noise: None method ──────────────────────────────────

    /// None method: raw eta applied directly without clamping.
    #[test]
    fn test_transform_inflow_noise_none_method() {
        let stochastic = make_one_hydro_stochastic(1);
        // StageIndexer: 1 hydro, 0 PAR lags → n_state = 1
        let indexer = StageIndexer::new(1, 0);
        let current_state = vec![0.0; indexer.n_state];

        // noise_scale[0] = 1.0, base_rhs = 5.0, eta = -3.0
        // expected: 5.0 + 1.0 * (-3.0) = 2.0
        let raw_noise = vec![-3.0_f64];
        let noise_scale = vec![1.0_f64];
        // Template with row_lower = [0.0, 5.0]; base_row = 1.
        let template = make_minimal_template(vec![0.0, 5.0]);
        let templates = vec![template];
        let base_rows = vec![1_usize];
        let inflow_method = InflowNonNegativityMethod::None;
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &noise_scale,
            n_hydros: 1,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let training_ctx = TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            inflow_method: &inflow_method,
            stochastic: &stochastic,
            initial_state: &current_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            dcs: None,
            noise_key_diag: None,
        };
        let mut scratch = make_scratch(1);

        transform_inflow_noise(
            &raw_noise,
            0,
            &current_state,
            &ctx,
            &training_ctx,
            &mut scratch,
        );

        assert_eq!(scratch.noise_buf.len(), 1);
        assert!((scratch.noise_buf[0] - 2.0).abs() < 1e-12);
    }

    // ── transform_inflow_noise: Truncation ───────────────────────────────────

    /// Truncation: when the PAR inflow would be negative, eta is clamped.
    ///
    /// AR(0) model: inflow = sigma * eta.  With sigma=1.0 and lag=0:
    /// inflow = 1.0 * eta.  For eta = -5.0, inflow = -5.0 < 0 → clamp.
    #[test]
    fn test_transform_inflow_noise_truncation_clamps() {
        let stochastic = make_one_hydro_stochastic(1);
        // 1 hydro, 0 PAR lags
        let indexer = StageIndexer::new(1, 0);
        let current_state = vec![0.0; indexer.n_state];

        // Very negative eta guarantees negative inflow (AR(0) with sigma=1).
        let raw_noise = vec![-5.0_f64];
        let noise_scale = vec![1.0_f64];
        // Template with row_lower = [0.0]; base_row = 0.
        let template = make_minimal_template(vec![0.0]);
        let templates = vec![template];
        let base_rows = vec![0_usize];
        let inflow_method = InflowNonNegativityMethod::Truncation;
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &noise_scale,
            n_hydros: 1,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let training_ctx = TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            inflow_method: &inflow_method,
            stochastic: &stochastic,
            initial_state: &current_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            dcs: None,
            noise_key_diag: None,
        };
        let mut scratch = make_scratch(1);

        transform_inflow_noise(
            &raw_noise,
            0,
            &current_state,
            &ctx,
            &training_ctx,
            &mut scratch,
        );

        assert_eq!(scratch.noise_buf.len(), 1);
        // The patched RHS = base_rhs + noise_scale * clamped_eta.
        // After clamping, the inflow contribution must be >= 0: RHS >= base_rhs = 0.
        assert!(
            scratch.noise_buf[0] >= -1e-10,
            "truncation must yield non-negative RHS, got {}",
            scratch.noise_buf[0]
        );
    }

    /// Truncation passthrough: positive-inflow eta passes through unchanged.
    #[test]
    fn test_transform_inflow_noise_truncation_passthrough() {
        let stochastic = make_one_hydro_stochastic(1);
        let indexer = StageIndexer::new(1, 0);
        let current_state = vec![0.0; indexer.n_state];

        // eta = 3.0 → inflow = 1.0 * 3.0 = 3.0 > 0 → no clamping.
        let raw_noise = vec![3.0_f64];
        let noise_scale = vec![2.0_f64];
        // Template with row_lower = [5.0]; base_row = 0.
        let template = make_minimal_template(vec![5.0]);
        let templates = vec![template];
        let base_rows = vec![0_usize];
        let inflow_method = InflowNonNegativityMethod::Truncation;
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &noise_scale,
            n_hydros: 1,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1],
            ncs_col_starts: &[],
            ncs_active_slot_to_local: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let training_ctx = TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            inflow_method: &inflow_method,
            stochastic: &stochastic,
            initial_state: &current_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            dcs: None,
            noise_key_diag: None,
        };
        let mut scratch = make_scratch(1);

        transform_inflow_noise(
            &raw_noise,
            0,
            &current_state,
            &ctx,
            &training_ctx,
            &mut scratch,
        );

        assert_eq!(scratch.noise_buf.len(), 1);
        // Expected: 5.0 + 2.0 * 3.0 = 11.0 (no clamping).
        assert!(
            (scratch.noise_buf[0] - 11.0).abs() < 1e-12,
            "expected 11.0, got {}",
            scratch.noise_buf[0]
        );
    }

    // ── transform_load_noise ──────────────────────────────────────────────────

    /// Basic load noise: verify RHS computation matches expected values.
    ///
    /// 1 hydro + 1 load bus.  Load bus is at noise index 1.
    /// eta = 0.0 → realization = (mean + std * 0.0).max(0.0) = mean.
    #[test]
    fn test_transform_load_noise_basic() {
        let mean_mw = 5.0_f64;
        let std_mw = 1.0_f64;
        let stochastic = make_stochastic_with_load(1, mean_mw, std_mw);

        // n_hydros=1 (hydro noise at index 0), load bus noise at index 1.
        // eta_load = 0.0 → realization = 5.0; block_factor = 1.0 → rhs = 5.0.
        let raw_noise = vec![0.0_f64, 0.0_f64]; // [hydro_eta, load_eta]
        let mut load_rhs_buf = Vec::new();

        transform_load_noise(&raw_noise, 1, 1, &stochastic, 0, 1, &mut load_rhs_buf);

        assert_eq!(load_rhs_buf.len(), 1);
        // The block_factor for a single Parallel block is the block duration
        // divided by total stage hours; with one block it equals 1.0.
        // Expected: 5.0 * 1.0 = 5.0.
        assert!(
            (load_rhs_buf[0] - 5.0).abs() < 1e-10,
            "expected 5.0, got {}",
            load_rhs_buf[0]
        );
    }

    /// Negative realizations are clamped to zero.
    ///
    /// Very negative eta drives `mean + std * eta` below zero; must be clamped.
    #[test]
    fn test_transform_load_noise_clamped_non_negative() {
        let mean_mw = 2.0_f64;
        let std_mw = 1.0_f64;
        let stochastic = make_stochastic_with_load(1, mean_mw, std_mw);

        // eta_load = -10.0 → realization = (2.0 - 10.0).max(0.0) = 0.0.
        let raw_noise = vec![0.0_f64, -10.0_f64];
        let mut load_rhs_buf = Vec::new();

        transform_load_noise(&raw_noise, 1, 1, &stochastic, 0, 1, &mut load_rhs_buf);

        assert_eq!(load_rhs_buf.len(), 1);
        assert!(
            load_rhs_buf[0].abs() < 1e-12,
            "expected 0.0, got {}",
            load_rhs_buf[0]
        );
    }

    // ── shift_lag_state tests ────────────────────────────────────────────────

    #[test]
    fn shift_lag_state_par0_is_noop() {
        let indexer = StageIndexer::new(2, 0);
        let mut state = vec![100.0, 200.0]; // storage only, no lags
        let incoming_lags: Vec<f64> = vec![];
        let primal = vec![0.0; 10];
        shift_lag_state(&mut state, &incoming_lags, &primal, &indexer);
        assert_eq!(
            state,
            vec![100.0, 200.0],
            "state must be unchanged for PAR(0)"
        );
    }

    #[test]
    fn shift_lag_state_par1_single_hydro() {
        // N=1, L=1: state = [v_out, lag0], inflow_lags.start = 1
        let indexer = StageIndexer::new(1, 1);
        let mut state = vec![500.0, 99.0]; // v_out, stale lag
        let incoming_lags = vec![42.0]; // lag0 (lag-major: lag * n_h + h = 0*1+0 = 0)
        // z_inflow.start = N*(1+L) = 1*(1+1) = 2
        let mut primal = vec![0.0; 10];
        primal[indexer.z_inflow.start] = 77.0; // Z_t for hydro 0
        shift_lag_state(&mut state, &incoming_lags, &primal, &indexer);
        assert_eq!(state[1], 77.0, "lag[0] must be Z_t = 77.0");
    }

    #[test]
    fn shift_lag_state_par3_single_hydro() {
        // N=1, L=3: state = [v_out, lag0, lag1, lag2]
        let indexer = StageIndexer::new(1, 3);
        let mut state = vec![500.0, 0.0, 0.0, 0.0];
        // incoming_lags in lag-major: [lag0, lag1, lag2] = [10.0, 20.0, 30.0]
        let incoming_lags = vec![10.0, 20.0, 30.0];
        let mut primal = vec![0.0; 20];
        primal[indexer.z_inflow.start] = 55.0;
        shift_lag_state(&mut state, &incoming_lags, &primal, &indexer);
        // After shift: lag[0]=Z_t=55, lag[1]=incoming[0]=10, lag[2]=incoming[1]=20
        assert_eq!(state[1], 55.0, "lag[0] must be Z_t");
        assert_eq!(state[2], 10.0, "lag[1] must be incoming lag[0]");
        assert_eq!(state[3], 20.0, "lag[2] must be incoming lag[1]");
    }

    #[test]
    fn shift_lag_state_par1_two_hydros() {
        // N=2, L=1: state = [v0, v1, lag0_h0, lag0_h1]
        // inflow_lags.start = 2, lag-major: lag0 * 2 + 0 = 0, lag0 * 2 + 1 = 1
        let indexer = StageIndexer::new(2, 1);
        let mut state = vec![100.0, 200.0, 0.0, 0.0];
        let incoming_lags = vec![10.0, 20.0]; // lag0_h0=10, lag0_h1=20
        let mut primal = vec![0.0; 20];
        primal[indexer.z_inflow.start] = 33.0; // Z_t for hydro 0
        primal[indexer.z_inflow.start + 1] = 44.0; // Z_t for hydro 1
        shift_lag_state(&mut state, &incoming_lags, &primal, &indexer);
        assert_eq!(state[2], 33.0, "lag[0] for h0 must be Z_t_h0");
        assert_eq!(state[3], 44.0, "lag[0] for h1 must be Z_t_h1");
    }

    #[test]
    fn shift_lag_state_preserves_storage() {
        // Verify storage portion [0..N] is unchanged after shift.
        let indexer = StageIndexer::new(2, 2);
        let mut state = vec![100.0, 200.0, 0.0, 0.0, 0.0, 0.0];
        let incoming_lags = vec![1.0, 2.0, 3.0, 4.0];
        let mut primal = vec![0.0; 20];
        primal[indexer.z_inflow.start] = 50.0;
        primal[indexer.z_inflow.start + 1] = 60.0;
        shift_lag_state(&mut state, &incoming_lags, &primal, &indexer);
        assert_eq!(state[0], 100.0, "storage[0] must be preserved");
        assert_eq!(state[1], 200.0, "storage[1] must be preserved");
    }

    // ── shift_anticipated_state tests ────────────────────────────────────────

    /// Build a StageIndexer with anticipated thermals using with_equipment_and_evaporation.
    fn make_anticipated_indexer(
        n_anticipated: usize,
        k_max: usize,
        anticipated_lead_stages: Vec<usize>,
    ) -> StageIndexer {
        use crate::indexer::{EquipmentCounts, EvapConfig, FphaColumnLayout};
        StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 0,
                max_par_order: 0,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated,
                k_max,
                anticipated_lead_stages,
                anticipated_thermal_indices: (0..n_anticipated).collect(),
                n_pumping: 0,
            },
            &FphaColumnLayout {
                hydro_indices: vec![],
                planes_per_hydro: vec![],
            },
            &EvapConfig {
                hydro_indices: vec![],
            },
        )
    }

    /// AC-1: given n_anticipated == 0, shift_anticipated_state is a no-op.
    #[test]
    fn shift_anticipated_state_no_anticipated_is_noop() {
        let indexer = StageIndexer::new(1, 0);
        // state: [storage_0]
        let mut state = vec![42.0_f64; indexer.n_state.max(1)];
        let incoming = vec![];
        let primal = vec![0.0_f64; 10];
        let before = state.clone();
        shift_anticipated_state(&mut state, &incoming, &primal, &indexer);
        assert_eq!(
            state, before,
            "state must be unchanged when n_anticipated == 0"
        );
    }

    /// AC-2: n_anticipated=1, k_max=2, K_0=2.
    /// incoming = [10.0, 20.0] (slot 0 = 10.0, slot 1 = 20.0).
    /// primal[decision_start] = 99.0.
    /// Expected outgoing: slot 0 = 20.0 (shifted from incoming slot 1),
    ///                    slot 1 = 99.0 (new decision).
    #[test]
    fn shift_anticipated_state_single_plant_k2() {
        let indexer = make_anticipated_indexer(1, 2, vec![2]);
        let ant_start = indexer.anticipated_state.start;
        let dec_start = indexer.anticipated_decision.start;

        // n_state = 0*(1+0) + 1*2 = 2
        let mut state = vec![0.0_f64; indexer.n_state];
        let incoming = vec![10.0_f64, 20.0_f64]; // slot 0 = 10.0, slot 1 = 20.0
        let mut primal = vec![0.0_f64; indexer.anticipated_decision.end + 1];
        primal[dec_start] = 99.0;

        shift_anticipated_state(&mut state, &incoming, &primal, &indexer);

        assert_eq!(
            state[ant_start], 20.0,
            "slot 0 must be shifted from incoming slot 1"
        );
        assert_eq!(
            state[ant_start + 1],
            99.0,
            "slot K_i-1 must receive the new decision primal"
        );
    }

    /// K_i = 1 edge case: single-slot ring buffer.
    /// The shift loop `for slot in 0..(k_i - 1)` becomes `0..0` (empty), so
    /// only the "new decision at slot K_i - 1 = 0" write fires. Slot 0 is
    /// consumed by the fishing row of the current stage and immediately
    /// replaced by the new decision for next-stage delivery.
    #[test]
    fn shift_anticipated_state_single_plant_k1() {
        let indexer = make_anticipated_indexer(1, 1, vec![1]);
        let ant_start = indexer.anticipated_state.start;
        let dec_start = indexer.anticipated_decision.start;

        // n_state = 0*(1+0) + 1*1 = 1 single anticipated slot.
        let mut state = vec![0.0_f64; indexer.n_state];
        let incoming = vec![100.0_f64]; // slot 0 = 100.0 (will be overwritten)
        let mut primal = vec![0.0_f64; indexer.anticipated_decision.end + 1];
        primal[dec_start] = 42.0;

        shift_anticipated_state(&mut state, &incoming, &primal, &indexer);

        // Shift loop is empty (0..0); the new decision write at slot K_i - 1 = 0
        // is the only operation.
        assert_eq!(
            state[ant_start], 42.0,
            "K_i=1 single slot must receive the new decision",
        );
    }

    /// AC-3: n_anticipated=2, k_max=3, K_0=2, K_1=3.
    /// For plant 0 (K=2): slot 0 = incoming_slot_1_plant_0, slot 1 = decision[0], slot 2 = 0.0.
    /// For plant 1 (K=3): slot 0 = incoming_slot_1_plant_1, slot 1 = incoming_slot_2_plant_1,
    ///                    slot 2 = decision[1].
    #[test]
    fn shift_anticipated_state_two_plants_mixed_k() {
        // n_ant=2, k_max=3. Slot-major layout: slot * 2 + plant.
        // n_state = 0 + 2*3 = 6.
        let indexer = make_anticipated_indexer(2, 3, vec![2, 3]);
        let ant_start = indexer.anticipated_state.start;
        let dec_start = indexer.anticipated_decision.start;

        let mut state = vec![0.0_f64; indexer.n_state];
        // incoming[slot * 2 + plant]:
        // slot0_p0=1.0, slot0_p1=2.0, slot1_p0=3.0, slot1_p1=4.0, slot2_p0=5.0, slot2_p1=6.0
        let incoming = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut primal = vec![0.0_f64; indexer.anticipated_decision.end + 1];
        primal[dec_start] = 10.0; // decision for plant 0
        primal[dec_start + 1] = 20.0; // decision for plant 1

        shift_anticipated_state(&mut state, &incoming, &primal, &indexer);

        // Plant 0 (K=2): slot 0 = incoming[1*2+0]=3.0, slot 1 = decision[0]=10.0, slot 2 = 0.0
        assert_eq!(state[ant_start + 0 * 2 + 0], 3.0, "plant0 slot0");
        assert_eq!(
            state[ant_start + 1 * 2 + 0],
            10.0,
            "plant0 slot1 (decision)"
        );
        assert_eq!(state[ant_start + 2 * 2 + 0], 0.0, "plant0 slot2 (padding)");

        // Plant 1 (K=3): slot 0 = incoming[1*2+1]=4.0, slot 1 = incoming[2*2+1]=6.0,
        //                slot 2 = decision[1]=20.0
        assert_eq!(state[ant_start + 0 * 2 + 1], 4.0, "plant1 slot0");
        assert_eq!(state[ant_start + 1 * 2 + 1], 6.0, "plant1 slot1");
        assert_eq!(
            state[ant_start + 2 * 2 + 1],
            20.0,
            "plant1 slot2 (decision)"
        );
    }

    /// shift_anticipated_state only modifies the anticipated_state slice; storage
    /// and lag regions are preserved unchanged.
    #[test]
    fn shift_anticipated_state_preserves_storage_and_lag() {
        // Use StageIndexer::new for N=2, L=1 to get storage + lag slots,
        // then build a combined state vector by padding with anticipated slots.
        // We must use an indexer that has both hydro state AND anticipated state.
        use crate::indexer::{EquipmentCounts, EvapConfig, FphaColumnLayout};
        let indexer = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 2,
                max_par_order: 1,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 1,
                k_max: 2,
                anticipated_lead_stages: vec![2],
                anticipated_thermal_indices: vec![0],
                n_pumping: 0,
            },
            &FphaColumnLayout {
                hydro_indices: vec![],
                planes_per_hydro: vec![],
            },
            &EvapConfig {
                hydro_indices: vec![],
            },
        );
        // n_state = N*(1+L) + A*K = 2*(1+1) + 1*2 = 6
        // storage [0..2), lags [2..4), anticipated [4..6)
        let ant_start = indexer.anticipated_state.start;
        let dec_start = indexer.anticipated_decision.start;

        let mut state = vec![100.0, 200.0, 10.0, 20.0, 0.0, 0.0];
        let incoming = vec![7.0_f64, 11.0]; // A*K = 2 values
        let mut primal = vec![0.0_f64; dec_start + 2];
        primal[dec_start] = 55.0;

        shift_anticipated_state(&mut state, &incoming, &primal, &indexer);

        // Storage and lag must be untouched.
        assert_eq!(state[0], 100.0, "storage[0] must be preserved");
        assert_eq!(state[1], 200.0, "storage[1] must be preserved");
        assert_eq!(state[2], 10.0, "lag[0] must be preserved");
        assert_eq!(state[3], 20.0, "lag[1] must be preserved");

        // Anticipated state must be shifted.
        assert_eq!(state[ant_start], 11.0, "slot 0 = incoming slot 1");
        assert_eq!(state[ant_start + 1], 55.0, "slot 1 = decision");
    }

    /// Early-return guard: k_max == 0 means no anticipated state slots, no-op.
    #[test]
    fn shift_anticipated_state_zero_k_max_is_noop() {
        // Build a trivial indexer with n_anticipated=0 (which also makes k_max=0).
        let indexer = StageIndexer::new(1, 0);
        let mut state = vec![5.0_f64; 3];
        let incoming = vec![];
        let primal = vec![0.0_f64; 5];
        let before = state.clone();
        shift_anticipated_state(&mut state, &incoming, &primal, &indexer);
        assert_eq!(state, before, "state must be unchanged when k_max == 0");
    }

    /// AC-1: shift_anticipated_state produces bit-identical output regardless
    /// of StageLagTransition flags (finalize_period, accumulate_weight).
    ///
    /// Because shift_anticipated_state does not take a StageLagTransition
    /// parameter, calling it on a "finalizing" stage versus a "non-finalizing"
    /// stage with identical inputs produces identical outgoing anticipated state.
    /// This test documents that invariant explicitly.
    #[test]
    fn shift_anticipated_state_invariant_under_stage_lag_transition() {
        // One anticipated thermal, k_max=2, K=2.
        let indexer = make_anticipated_indexer(1, 2, vec![2]);

        let incoming_anticipated = vec![5.0_f64, 15.0];
        let dec_start = indexer.anticipated_decision.start;
        let mut primal = vec![0.0_f64; dec_start + 2];
        primal[dec_start] = 42.0;

        // state_a represents a finalizing stage (finalize_period = true,
        // accumulate_weight = 1.0 in the lag accumulator — irrelevant here).
        // state_b represents a non-finalizing stage (accumulate_weight = 0.25).
        // Neither flag is passed to shift_anticipated_state; both calls are
        // therefore identical and must yield bit-for-bit equal results.
        let mut state_a = vec![0.0_f64; indexer.n_state];
        let mut state_b = state_a.clone();

        shift_anticipated_state(&mut state_a, &incoming_anticipated, &primal, &indexer);
        shift_anticipated_state(&mut state_b, &incoming_anticipated, &primal, &indexer);

        let s = indexer.anticipated_state.start;
        let n_ant_state = indexer.n_anticipated * indexer.k_max;
        assert_eq!(
            &state_a[s..s + n_ant_state],
            &state_b[s..s + n_ant_state],
            "anticipated_state shift must be invariant under StageLagTransition"
        );
    }

    // ── compute_effective_eta tests ─────────────────────────────────────────

    #[test]
    fn test_compute_effective_eta_none_passes_through() {
        let raw_noise = [0.5, -1.0];
        let par_inflows = []; // unused for None
        let eta_floor = []; // unused for None
        let mut effective = Vec::new();
        compute_effective_eta(
            &raw_noise,
            2,
            InflowNonNegativityMethod::None,
            &par_inflows,
            &eta_floor,
            &mut effective,
        );
        assert_eq!(effective, vec![0.5, -1.0]);
    }

    #[test]
    fn test_compute_effective_eta_penalty_passes_through() {
        let raw_noise = [0.5, -1.0];
        let par_inflows = [];
        let eta_floor = [];
        let mut effective = Vec::new();
        compute_effective_eta(
            &raw_noise,
            2,
            InflowNonNegativityMethod::Penalty,
            &par_inflows,
            &eta_floor,
            &mut effective,
        );
        assert_eq!(effective, vec![0.5, -1.0]);
    }

    #[test]
    fn test_compute_effective_eta_truncation_clamps_negative() {
        // 2 hydros: par_inflows[0] < 0 -> clamp eta[0]; par_inflows[1] > 0 -> pass through.
        let raw_noise = [-2.0, 1.0];
        let par_inflows = [-5.0, 3.0];
        let eta_floor = [-1.0, -0.5]; // floor for hydro 0 is -1.0
        let mut effective = Vec::new();
        compute_effective_eta(
            &raw_noise,
            2,
            InflowNonNegativityMethod::Truncation,
            &par_inflows,
            &eta_floor,
            &mut effective,
        );
        // hydro 0: eta=-2.0, floor=-1.0 -> max(-2, -1) = -1.0
        // hydro 1: par_inflows[1]=3.0 >= 0 -> no clamp -> eta=1.0
        assert_eq!(effective, vec![-1.0, 1.0]);
    }

    #[test]
    fn test_compute_effective_eta_truncation_passes_positive() {
        // All PAR inflows positive -> no clamping at all.
        let raw_noise = [-2.0, 1.0];
        let par_inflows = [3.0, 5.0];
        let eta_floor = [-1.0, -0.5]; // floors are irrelevant when no negative inflow
        let mut effective = Vec::new();
        compute_effective_eta(
            &raw_noise,
            2,
            InflowNonNegativityMethod::Truncation,
            &par_inflows,
            &eta_floor,
            &mut effective,
        );
        assert_eq!(effective, vec![-2.0, 1.0]);
    }

    #[test]
    fn test_compute_effective_eta_truncation_with_penalty_clamps() {
        // TruncationWithPenalty behaves the same as Truncation for clamping.
        let raw_noise = [-2.0, 1.0];
        let par_inflows = [-5.0, 3.0];
        let eta_floor = [-1.0, -0.5];
        let mut effective = Vec::new();
        compute_effective_eta(
            &raw_noise,
            2,
            InflowNonNegativityMethod::TruncationWithPenalty,
            &par_inflows,
            &eta_floor,
            &mut effective,
        );
        assert_eq!(effective, vec![-1.0, 1.0]);
    }

    // ── accumulate_and_shift_lag_state tests ─────────────────────────────────

    use cobre_core::temporal::StageLagTransition;

    use crate::noise::{DownstreamAccumState, LagAccumState, accumulate_and_shift_lag_state};
    // Convenience helper: build a no-op DownstreamAccumState for tests that
    // exercise only primary accumulation (uniform-resolution path).
    fn noop_ds<'a>(
        accumulator: &'a mut Vec<f64>,
        weight_accum: &'a mut f64,
        completed_lags: &'a mut Vec<f64>,
        n_completed: &'a mut usize,
    ) -> DownstreamAccumState<'a> {
        DownstreamAccumState {
            accumulator: accumulator.as_mut_slice(),
            weight_accum,
            completed_lags: completed_lags.as_mut_slice(),
            n_completed,
            par_order: 0,
        }
    }

    /// Monthly identity: `accumulate_weight=1.0`, `spillover_weight=0.0`, `finalize_period=true`.
    ///
    /// With a single finalization stage the result must be bit-for-bit
    /// identical to `shift_lag_state`.
    #[test]
    fn test_accumulate_monthly_identity() {
        // N=1 hydro, L=1 lag order.
        let indexer = StageIndexer::new(1, 1);

        // Reference: shift_lag_state result.
        let mut state_ref = vec![500.0, 99.0];
        let incoming_lags = vec![42.0];
        let mut primal = vec![0.0; 10];
        primal[indexer.z_inflow.start] = 77.0;
        shift_lag_state(&mut state_ref, &incoming_lags, &primal, &indexer);

        // Accumulate: single stage with identity weights.
        let mut state_acc = vec![500.0, 99.0];
        let mut lag_accumulator = vec![0.0_f64; 1];
        let mut lag_weight_accum = 0.0_f64;
        let stage_lag = StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: true,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: false,
        };
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;
        accumulate_and_shift_lag_state(
            &mut state_acc,
            &incoming_lags,
            &primal,
            &indexer,
            &stage_lag,
            &mut LagAccumState {
                accumulator: &mut lag_accumulator,
                weight_accum: &mut lag_weight_accum,
            },
            &mut noop_ds(
                &mut ds_accum,
                &mut ds_weight,
                &mut ds_completed,
                &mut ds_n_completed,
            ),
        );

        assert_eq!(
            state_acc, state_ref,
            "monthly identity must produce identical result to shift_lag_state"
        );
        // Accumulator must be zeroed (clean for next period).
        assert_eq!(lag_accumulator[0], 0.0);
        assert_eq!(lag_weight_accum, 0.0);
    }

    /// Four weekly stages each contributing weight=0.25, finalize only on stage 3.
    ///
    /// After processing all four stages the lag[0] must equal the weighted
    /// average: (500 + 480 + 520 + 510) / 4 = 502.5.
    #[test]
    fn test_accumulate_four_weeks_then_finalize() {
        let indexer = StageIndexer::new(1, 1);
        let mut state = vec![500.0, 0.0]; // storage, lag0
        let incoming_lags = vec![0.0]; // lag-major: lag0 for hydro 0
        let mut lag_accumulator = vec![0.0_f64; 1];
        let mut lag_weight_accum = 0.0_f64;

        let z_inflows = [500.0_f64, 480.0, 520.0, 510.0];
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;

        for (week, &z) in z_inflows.iter().enumerate() {
            let finalize = week == 3;
            let stage_lag = StageLagTransition {
                accumulate_weight: 0.25,
                spillover_weight: 0.0,
                finalize_period: finalize,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            };
            let mut primal = vec![0.0; 10];
            primal[indexer.z_inflow.start] = z;
            accumulate_and_shift_lag_state(
                &mut state,
                &incoming_lags,
                &primal,
                &indexer,
                &stage_lag,
                &mut LagAccumState {
                    accumulator: &mut lag_accumulator,
                    weight_accum: &mut lag_weight_accum,
                },
                &mut noop_ds(
                    &mut ds_accum,
                    &mut ds_weight,
                    &mut ds_completed,
                    &mut ds_n_completed,
                ),
            );
        }

        // lag[0] is at inflow_lags.start = 1 (state index).
        let expected = (500.0 + 480.0 + 520.0 + 510.0) / 4.0;
        assert!(
            (state[indexer.inflow_lags.start] - expected).abs() < 1e-12,
            "lag[0] must equal weighted average {expected}, got {}",
            state[indexer.inflow_lags.start]
        );
        // Accumulator reset after finalization.
        assert_eq!(lag_accumulator[0], 0.0);
        assert_eq!(lag_weight_accum, 0.0);
    }

    /// Spillover seeds the next lag period with raw `z_inflow` * `spillover_weight`.
    #[test]
    fn test_accumulate_spillover_seeds_next_period() {
        let indexer = StageIndexer::new(1, 1);
        let mut state = vec![0.0, 0.0];
        let incoming_lags = vec![0.0];
        let mut lag_accumulator = vec![0.0_f64; 1];
        let mut lag_weight_accum = 0.0_f64;
        let mut primal = vec![0.0; 10];
        primal[indexer.z_inflow.start] = 200.0;

        let stage_lag = StageLagTransition {
            accumulate_weight: 0.968, // 1.0 - 0.032 = days in period / days in month
            spillover_weight: 0.032,
            finalize_period: true,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: false,
        };
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;
        accumulate_and_shift_lag_state(
            &mut state,
            &incoming_lags,
            &primal,
            &indexer,
            &stage_lag,
            &mut LagAccumState {
                accumulator: &mut lag_accumulator,
                weight_accum: &mut lag_weight_accum,
            },
            &mut noop_ds(
                &mut ds_accum,
                &mut ds_weight,
                &mut ds_completed,
                &mut ds_n_completed,
            ),
        );

        // After finalization, accumulator seeded with raw z_inflow * spillover_weight.
        let expected_seed = 200.0 * 0.032;
        assert!(
            (lag_accumulator[0] - expected_seed).abs() < 1e-12,
            "accumulator must be seeded with z_inflow * spillover_weight = {expected_seed}, got {}",
            lag_accumulator[0]
        );
        assert!(
            (lag_weight_accum - 0.032).abs() < 1e-12,
            "lag_weight_accum must equal spillover_weight = 0.032, got {lag_weight_accum}"
        );
    }

    /// `max_par_order == 0`: function must return immediately, nothing modified.
    #[test]
    fn test_accumulate_noop_for_par0() {
        let indexer = StageIndexer::new(2, 0); // no lag order
        let mut state = vec![100.0, 200.0];
        let incoming_lags: Vec<f64> = vec![];
        let primal = vec![0.0; 10];
        let mut lag_accumulator: Vec<f64> = vec![]; // empty — should never be accessed
        let mut lag_weight_accum = 0.0_f64;
        let stage_lag = StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: true,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: false,
        };
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;
        accumulate_and_shift_lag_state(
            &mut state,
            &incoming_lags,
            &primal,
            &indexer,
            &stage_lag,
            &mut LagAccumState {
                accumulator: &mut lag_accumulator,
                weight_accum: &mut lag_weight_accum,
            },
            &mut noop_ds(
                &mut ds_accum,
                &mut ds_weight,
                &mut ds_completed,
                &mut ds_n_completed,
            ),
        );
        assert_eq!(
            state,
            vec![100.0, 200.0],
            "state must be unchanged for PAR(0)"
        );
        assert_eq!(lag_weight_accum, 0.0, "weight must be unchanged for PAR(0)");
    }

    /// Storage region of state (indices 0..N) must not be touched by the shift.
    #[test]
    fn test_accumulate_preserves_storage() {
        // N=2 hydros, L=2 lag order: state = [v0, v1, lag0_h0, lag0_h1, lag1_h0, lag1_h1]
        let indexer = StageIndexer::new(2, 2);
        let mut state = vec![100.0, 200.0, 0.0, 0.0, 0.0, 0.0];
        let incoming_lags = vec![1.0, 2.0, 3.0, 4.0]; // lag-major: lag0 h0,h1; lag1 h0,h1
        let mut primal = vec![0.0; 20];
        primal[indexer.z_inflow.start] = 50.0;
        primal[indexer.z_inflow.start + 1] = 60.0;
        let mut lag_accumulator = vec![0.0_f64; 2];
        let mut lag_weight_accum = 0.0_f64;
        let stage_lag = StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: true,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: false,
        };
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;
        accumulate_and_shift_lag_state(
            &mut state,
            &incoming_lags,
            &primal,
            &indexer,
            &stage_lag,
            &mut LagAccumState {
                accumulator: &mut lag_accumulator,
                weight_accum: &mut lag_weight_accum,
            },
            &mut noop_ds(
                &mut ds_accum,
                &mut ds_weight,
                &mut ds_completed,
                &mut ds_n_completed,
            ),
        );
        assert_eq!(state[0], 100.0, "storage[0] must be preserved");
        assert_eq!(state[1], 200.0, "storage[1] must be preserved");
    }

    // ── downstream accumulation tests ────────────────────────────────────────
    //
    // These tests exercise the downstream (coarser-resolution) ring-buffer path
    // of `accumulate_and_shift_lag_state`.  They validate:
    //   • quarterly-average accumulation and ring-buffer storage
    //   • multi-lag PAR(2) fill ordering
    //   • post-rebuild state reset
    //   • downstream spillover seeding
    //   • multi-hydro independence

    /// Build a `StageLagTransition` for a standard monthly stage that also
    /// accumulates into the downstream (quarterly) ring buffer.
    ///
    /// weight = 1/3 per month (3 months per quarter, no spillover).
    fn monthly_with_downstream(
        finalize_primary: bool,
        downstream_finalize: bool,
    ) -> StageLagTransition {
        StageLagTransition {
            accumulate_weight: 1.0 / 3.0,
            spillover_weight: 0.0,
            finalize_period: finalize_primary,
            accumulate_downstream: true,
            downstream_accumulate_weight: 1.0 / 3.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize,
            rebuild_from_downstream: false,
        }
    }

    /// Drive one stage through `accumulate_and_shift_lag_state` with full
    /// downstream buffers.
    #[allow(clippy::too_many_arguments)]
    fn run_stage(
        state: &mut [f64],
        incoming_lags: &[f64],
        z_inflow: f64,
        indexer: &crate::indexer::StageIndexer,
        stage_lag: &StageLagTransition,
        lag: &mut LagAccumState<'_>,
        ds: &mut DownstreamAccumState<'_>,
    ) {
        let mut primal = vec![0.0; indexer.z_inflow.start + indexer.hydro_count + 4];
        primal[indexer.z_inflow.start] = z_inflow;
        accumulate_and_shift_lag_state(state, incoming_lags, &primal, indexer, stage_lag, lag, ds);
    }

    /// Drive one stage with two hydros.
    fn run_stage_2h(
        state: &mut [f64],
        incoming_lags: &[f64],
        z_inflows: [f64; 2],
        indexer: &crate::indexer::StageIndexer,
        stage_lag: &StageLagTransition,
        lag: &mut LagAccumState<'_>,
        ds: &mut DownstreamAccumState<'_>,
    ) {
        let n = indexer.z_inflow.start + indexer.hydro_count + 4;
        let mut primal = vec![0.0; n];
        primal[indexer.z_inflow.start] = z_inflows[0];
        primal[indexer.z_inflow.start + 1] = z_inflows[1];
        accumulate_and_shift_lag_state(state, incoming_lags, &primal, indexer, stage_lag, lag, ds);
    }

    /// Test 1: PAR(1) downstream accumulation with a 3-stage quarterly window.
    ///
    /// 3 monthly stages (each weight=1/3, no primary finalize, no primary
    /// spillover, `downstream_finalize` on last month) populate the downstream
    /// ring buffer.  After all 3 stages, `downstream_completed_lags[0]` must
    /// equal `(90.0 + 100.0 + 110.0) / 3.0 = 100.0`.  Then calling with
    /// `rebuild_from_downstream = true` on the first quarterly stage overwrites
    /// `state[lag_start]` with `100.0`.
    #[test]
    fn test_downstream_par1_accumulation_and_rebuild() {
        // N=1 hydro, L=1 lag (primary monthly PAR(1) order).
        let indexer = StageIndexer::new(1, 1);
        let lag_start = indexer.inflow_lags.start;

        // Primary state: storage=500, lag0=old_value_to_be_replaced.
        let mut state = vec![500.0, 42.0];
        let incoming_lags = vec![0.0];
        let mut lag_acc = vec![0.0_f64; 1];
        let mut lag_w = 0.0_f64;
        // downstream: par_order=1, ring buf capacity n_h * 1 = 1.
        let mut ds_acc = vec![0.0_f64; 1];
        let mut ds_w = 0.0_f64;
        let mut ds_completed = vec![0.0_f64; 1];
        let mut ds_n = 0_usize;

        let z_vals = [90.0_f64, 100.0, 110.0];
        for (i, &z) in z_vals.iter().enumerate() {
            let ds_finalize = i == 2; // last month of the quarter
            let stage_lag = monthly_with_downstream(false, ds_finalize);
            run_stage(
                &mut state,
                &incoming_lags,
                z,
                &indexer,
                &stage_lag,
                &mut LagAccumState {
                    accumulator: &mut lag_acc,
                    weight_accum: &mut lag_w,
                },
                &mut DownstreamAccumState {
                    accumulator: &mut ds_acc,
                    weight_accum: &mut ds_w,
                    completed_lags: &mut ds_completed,
                    n_completed: &mut ds_n,
                    par_order: 1,
                },
            );
        }

        // After 3 monthly stages the ring buffer should hold the quarterly average.
        let expected_avg = (90.0 + 100.0 + 110.0) / 3.0;
        assert!(
            (ds_completed[0] - expected_avg).abs() < 1e-12,
            "ring buf slot 0 should be {expected_avg}, got {}",
            ds_completed[0]
        );
        assert_eq!(ds_n, 1, "n_completed must be 1 after one quarter");

        // Now simulate the transition stage (first quarterly stage).
        // rebuild_from_downstream=true; primary accumulation is quarterly.
        let rebuild_lag = StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: false,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: true,
        };
        run_stage(
            &mut state,
            &incoming_lags,
            999.0, // z_inflow irrelevant — rebuild returns before primary finalize
            &indexer,
            &rebuild_lag,
            &mut LagAccumState {
                accumulator: &mut lag_acc,
                weight_accum: &mut lag_w,
            },
            &mut DownstreamAccumState {
                accumulator: &mut ds_acc,
                weight_accum: &mut ds_w,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n,
                par_order: 1,
            },
        );

        // lag[0] must be rebuilt to the quarterly average.
        assert!(
            (state[lag_start] - expected_avg).abs() < 1e-12,
            "state[lag_start] must be rebuilt to {expected_avg}, got {}",
            state[lag_start]
        );
        // Storage must be untouched.
        assert_eq!(state[0], 500.0, "storage must be untouched during rebuild");
        // Downstream state must be fully reset.
        assert_eq!(ds_n, 0, "n_completed must reset to 0 after rebuild");
        assert_eq!(
            ds_completed[0], 0.0,
            "completed_lags must be zeroed after rebuild"
        );
        assert_eq!(ds_w, 0.0, "downstream weight must reset after rebuild");
    }

    /// Test 2: PAR(2) downstream accumulation with two consecutive quarters.
    ///
    /// 6 monthly stages (Q3: stages 0-2, Q4: stages 3-5) with `downstream_par_order=2`.
    /// After Q3, `completed_lags[slot=0] == avg(60,70,80) == 70.0`, `n_completed==1`.
    /// After Q4, `completed_lags[slot=1] == avg(90,100,110) == 100.0`, `n_completed==2`.
    /// At rebuild: `state[lag_start] == 100.0` (newest Q4), `state[lag_start+1] == 70.0` (Q3).
    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_downstream_par2_two_quarters() {
        let indexer = StageIndexer::new(1, 2); // L=2 lag order
        let lag_start = indexer.inflow_lags.start;

        let mut state = vec![0.0; 1 + 2]; // storage + lag0 + lag1
        let incoming_lags = vec![0.0, 0.0]; // lag-major: lag0 h0, lag1 h0
        let mut lag_acc = vec![0.0_f64; 1];
        let mut lag_w = 0.0_f64;
        // par_order=2: ring buf capacity n_h * 2 = 2.
        let mut ds_acc = vec![0.0_f64; 1];
        let mut ds_w = 0.0_f64;
        let mut ds_completed = vec![0.0_f64; 2];
        let mut ds_n = 0_usize;

        // Q3: z_inflows 60, 70, 80 — no primary finalize, downstream finalize on month 3.
        let q3_vals = [60.0_f64, 70.0, 80.0];
        for (i, &z) in q3_vals.iter().enumerate() {
            let stage_lag = monthly_with_downstream(false, i == 2);
            run_stage(
                &mut state,
                &incoming_lags,
                z,
                &indexer,
                &stage_lag,
                &mut LagAccumState {
                    accumulator: &mut lag_acc,
                    weight_accum: &mut lag_w,
                },
                &mut DownstreamAccumState {
                    accumulator: &mut ds_acc,
                    weight_accum: &mut ds_w,
                    completed_lags: &mut ds_completed,
                    n_completed: &mut ds_n,
                    par_order: 2,
                },
            );
        }
        let q3_avg = (60.0 + 70.0 + 80.0) / 3.0;
        assert!(
            (ds_completed[0] - q3_avg).abs() < 1e-12,
            "slot 0 should be Q3 avg {q3_avg}, got {}",
            ds_completed[0]
        );
        assert_eq!(ds_n, 1);

        // Q4: z_inflows 90, 100, 110 — downstream finalize on month 3.
        let q4_vals = [90.0_f64, 100.0, 110.0];
        for (i, &z) in q4_vals.iter().enumerate() {
            let stage_lag = monthly_with_downstream(false, i == 2);
            run_stage(
                &mut state,
                &incoming_lags,
                z,
                &indexer,
                &stage_lag,
                &mut LagAccumState {
                    accumulator: &mut lag_acc,
                    weight_accum: &mut lag_w,
                },
                &mut DownstreamAccumState {
                    accumulator: &mut ds_acc,
                    weight_accum: &mut ds_w,
                    completed_lags: &mut ds_completed,
                    n_completed: &mut ds_n,
                    par_order: 2,
                },
            );
        }
        let q4_avg = (90.0 + 100.0 + 110.0) / 3.0;
        assert!(
            (ds_completed[1] - q4_avg).abs() < 1e-12,
            "slot 1 should be Q4 avg {q4_avg}, got {}",
            ds_completed[1]
        );
        assert_eq!(ds_n, 2);

        // Rebuild stage: lag[0] <- newest (Q4), lag[1] <- second-newest (Q3).
        let rebuild_lag = StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: false,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: true,
        };
        run_stage(
            &mut state,
            &incoming_lags,
            999.0,
            &indexer,
            &rebuild_lag,
            &mut LagAccumState {
                accumulator: &mut lag_acc,
                weight_accum: &mut lag_w,
            },
            &mut DownstreamAccumState {
                accumulator: &mut ds_acc,
                weight_accum: &mut ds_w,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n,
                par_order: 2,
            },
        );

        // lag[0] = newest = Q4 avg; lag[1] = Q3 avg.
        assert!(
            (state[lag_start] - q4_avg).abs() < 1e-12,
            "lag[0] should be newest Q4 avg {q4_avg}, got {}",
            state[lag_start]
        );
        assert!(
            (state[lag_start + 1] - q3_avg).abs() < 1e-12,
            "lag[1] should be Q3 avg {q3_avg}, got {}",
            state[lag_start + 1]
        );
    }

    /// Test 3: Uniform monthly study — empty downstream buffers, zero overhead.
    ///
    /// Calls `accumulate_and_shift_lag_state` with `downstream_accumulator = &mut []`.
    /// The function must produce exactly the same result,
    /// with no downstream fields accessed.
    #[test]
    fn test_no_downstream_for_uniform_monthly() {
        let indexer = StageIndexer::new(1, 1);
        let mut state_ds = vec![500.0, 0.0]; // with empty downstream
        let mut state_ref = vec![500.0, 0.0]; // with noop downstream
        let incoming_lags = vec![0.0];
        let z_inflows = [100.0_f64, 110.0, 120.0];

        let mut lag_acc_ref = vec![0.0_f64; 1];
        let mut lag_w_ref = 0.0_f64;
        let mut lag_acc_ds = vec![0.0_f64; 1];
        let mut lag_w_ds = 0.0_f64;

        for (i, &z) in z_inflows.iter().enumerate() {
            let finalize = i == 2;
            let stage_lag = StageLagTransition {
                accumulate_weight: 1.0 / 3.0,
                spillover_weight: 0.0,
                finalize_period: finalize,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            };

            // Reference: empty downstream (noop path).
            let mut ds_accum_ref: Vec<f64> = vec![];
            let mut ds_weight_ref = 0.0_f64;
            let mut ds_completed_ref: Vec<f64> = vec![];
            let mut ds_n_completed_ref = 0_usize;
            run_stage(
                &mut state_ref,
                &incoming_lags,
                z,
                &indexer,
                &stage_lag,
                &mut LagAccumState {
                    accumulator: &mut lag_acc_ref,
                    weight_accum: &mut lag_w_ref,
                },
                &mut noop_ds(
                    &mut ds_accum_ref,
                    &mut ds_weight_ref,
                    &mut ds_completed_ref,
                    &mut ds_n_completed_ref,
                ),
            );

            // Test: inline empty downstream (par_order=0).
            let mut ds_accum_ds: Vec<f64> = vec![];
            let mut ds_weight_ds = 0.0_f64;
            let mut ds_completed_ds: Vec<f64> = vec![];
            let mut ds_n_completed_ds = 0_usize;
            run_stage(
                &mut state_ds,
                &incoming_lags,
                z,
                &indexer,
                &stage_lag,
                &mut LagAccumState {
                    accumulator: &mut lag_acc_ds,
                    weight_accum: &mut lag_w_ds,
                },
                &mut DownstreamAccumState {
                    accumulator: &mut ds_accum_ds,
                    weight_accum: &mut ds_weight_ds,
                    completed_lags: &mut ds_completed_ds,
                    n_completed: &mut ds_n_completed_ds,
                    par_order: 0,
                },
            );
        }

        assert_eq!(
            state_ds, state_ref,
            "uniform monthly study must be identical with or without downstream buffers"
        );
    }

    /// Test 4: `rebuild_from_downstream` resets all downstream state.
    ///
    /// After rebuild, `n_completed == 0`, `completed_lags` all zero, and
    /// `downstream_weight_accum == 0.0`.
    #[test]
    fn test_rebuild_resets_downstream_state() {
        let indexer = StageIndexer::new(1, 1);
        let mut state = vec![0.0, 0.0];
        let incoming_lags = vec![0.0];
        let mut lag_acc = vec![0.0_f64; 1];
        let mut lag_w = 0.0_f64;
        let mut ds_acc = vec![0.0_f64; 1];
        let mut ds_w = 0.5_f64; // non-zero before rebuild
        let mut ds_completed = vec![77.0_f64; 1]; // non-zero before rebuild
        let mut ds_n = 1_usize; // pretend one quarter was completed

        let rebuild_lag = StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: false,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: true,
        };

        run_stage(
            &mut state,
            &incoming_lags,
            0.0,
            &indexer,
            &rebuild_lag,
            &mut LagAccumState {
                accumulator: &mut lag_acc,
                weight_accum: &mut lag_w,
            },
            &mut DownstreamAccumState {
                accumulator: &mut ds_acc,
                weight_accum: &mut ds_w,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n,
                par_order: 1,
            },
        );

        assert_eq!(ds_n, 0, "n_completed must reset to 0 after rebuild");
        assert_eq!(
            ds_completed[0], 0.0,
            "completed_lags must be zeroed after rebuild"
        );
        assert_eq!(
            ds_w, 0.0,
            "downstream weight_accum must reset after rebuild"
        );
        assert_eq!(
            ds_acc[0], 0.0,
            "downstream accumulator must be zeroed after rebuild"
        );
    }

    /// Test 5: Downstream spillover seeds the next quarterly accumulation.
    ///
    /// A monthly stage with `downstream_spillover_weight = 0.1` and
    /// `downstream_finalize = true` should: (a) finalize the current quarter,
    /// (b) seed the next quarter's accumulator with `z_inflow * 0.1`.
    #[test]
    fn test_downstream_spillover_seeds_next_quarter() {
        let indexer = StageIndexer::new(1, 1);
        let mut state = vec![0.0, 0.0];
        let incoming_lags = vec![0.0];
        let mut lag_acc = vec![0.0_f64; 1];
        let mut lag_w = 0.0_f64;
        let mut ds_acc = vec![0.0_f64; 1];
        let mut ds_completed = vec![0.0_f64; 1];
        let mut ds_n = 0_usize;

        // Single monthly stage that finalizes the quarter with spillover.
        // Pre-load the accumulator to simulate prior months already accumulated.
        ds_acc[0] = 200.0; // months 1+2 already accumulated
        let mut ds_w = 2.0 / 3.0_f64; // two months of weight 1/3 each

        let spillover_weight = 0.1;
        let stage_lag = StageLagTransition {
            accumulate_weight: 1.0 / 3.0,
            spillover_weight: 0.0,
            finalize_period: false,
            accumulate_downstream: true,
            downstream_accumulate_weight: 1.0 / 3.0,
            downstream_spillover_weight: spillover_weight,
            downstream_finalize: true,
            rebuild_from_downstream: false,
        };

        let z = 120.0_f64;
        run_stage(
            &mut state,
            &incoming_lags,
            z,
            &indexer,
            &stage_lag,
            &mut LagAccumState {
                accumulator: &mut lag_acc,
                weight_accum: &mut lag_w,
            },
            &mut DownstreamAccumState {
                accumulator: &mut ds_acc,
                weight_accum: &mut ds_w,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n,
                par_order: 1,
            },
        );

        // Quarter should be finalized and the ring buffer filled.
        assert_eq!(ds_n, 1, "one quarter must be finalized");
        // Downstream accumulator must be seeded with z * spillover_weight.
        let expected_seed = z * spillover_weight;
        assert!(
            (ds_acc[0] - expected_seed).abs() < 1e-12,
            "accumulator should be seeded with {expected_seed}, got {}",
            ds_acc[0]
        );
        assert!(
            (ds_w - spillover_weight).abs() < 1e-12,
            "weight_accum should be {spillover_weight}, got {ds_w}"
        );
    }

    /// Test 6: Multi-hydro downstream — 2 hydros, PAR(1).
    ///
    /// Each hydro has its own `z_inflow` values.  After 3 monthly stages,
    /// `downstream_completed_lags[0]` (hydro 0) and `[1]` (hydro 1) must
    /// each equal the independently computed quarterly average for that hydro.
    #[test]
    fn test_downstream_multi_hydro() {
        // N=2 hydros, L=1 lag order.
        let indexer = StageIndexer::new(2, 1);
        let lag_start = indexer.inflow_lags.start;

        let mut state = vec![0.0; 2 + 2]; // 2 storage + 2 lag entries (lag0 h0, lag0 h1)
        let incoming_lags = vec![0.0, 0.0]; // lag-major: lag0 h0, lag0 h1
        let mut lag_acc = vec![0.0_f64; 2];
        let mut lag_w = 0.0_f64;
        // ring buf capacity: n_h * par_order = 2 * 1 = 2
        let mut ds_acc = vec![0.0_f64; 2];
        let mut ds_w = 0.0_f64;
        let mut ds_completed = vec![0.0_f64; 2];
        let mut ds_n = 0_usize;

        // 3 monthly stages: hydro 0 inflows [10, 20, 30], hydro 1 inflows [40, 50, 60].
        let h0_vals = [10.0_f64, 20.0, 30.0];
        let h1_vals = [40.0_f64, 50.0, 60.0];

        for (i, (&z0, &z1)) in h0_vals.iter().zip(h1_vals.iter()).enumerate() {
            let stage_lag = monthly_with_downstream(false, i == 2);
            run_stage_2h(
                &mut state,
                &incoming_lags,
                [z0, z1],
                &indexer,
                &stage_lag,
                &mut LagAccumState {
                    accumulator: &mut lag_acc,
                    weight_accum: &mut lag_w,
                },
                &mut DownstreamAccumState {
                    accumulator: &mut ds_acc,
                    weight_accum: &mut ds_w,
                    completed_lags: &mut ds_completed,
                    n_completed: &mut ds_n,
                    par_order: 1,
                },
            );
        }

        let expected_h0 = (10.0 + 20.0 + 30.0) / 3.0;
        let expected_h1 = (40.0 + 50.0 + 60.0) / 3.0;
        assert!(
            (ds_completed[0] - expected_h0).abs() < 1e-12,
            "hydro 0 quarterly avg should be {expected_h0}, got {}",
            ds_completed[0]
        );
        assert!(
            (ds_completed[1] - expected_h1).abs() < 1e-12,
            "hydro 1 quarterly avg should be {expected_h1}, got {}",
            ds_completed[1]
        );
        assert_eq!(ds_n, 1);

        // Rebuild: both hydros rebuilt independently.
        let rebuild_lag = StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: false,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: true,
        };
        run_stage_2h(
            &mut state,
            &incoming_lags,
            [999.0, 888.0],
            &indexer,
            &rebuild_lag,
            &mut LagAccumState {
                accumulator: &mut lag_acc,
                weight_accum: &mut lag_w,
            },
            &mut DownstreamAccumState {
                accumulator: &mut ds_acc,
                weight_accum: &mut ds_w,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n,
                par_order: 1,
            },
        );

        // lag[0] for hydro 0 and hydro 1 must be rebuilt independently.
        assert!(
            (state[lag_start] - expected_h0).abs() < 1e-12,
            "rebuilt lag[0] hydro 0 should be {expected_h0}, got {}",
            state[lag_start]
        );
        assert!(
            (state[lag_start + 1] - expected_h1).abs() < 1e-12,
            "rebuilt lag[0] hydro 1 should be {expected_h1}, got {}",
            state[lag_start + 1]
        );
    }

    /// Three-stage evolution with pre-horizon seed: slot 1→0, decision→1 per stage.
    #[test]
    fn shift_anticipated_state_pre_horizon_seed_three_stage_evolution() {
        let indexer = make_anticipated_indexer(1, 2, vec![2]);
        let ant_start = indexer.anticipated_state.start;
        let dec_start = indexer.anticipated_decision.start;

        let mut state = vec![0.0_f64; indexer.n_state];
        state[ant_start] = 10.0;
        state[ant_start + 1] = 20.0;

        let mut incoming = vec![0.0_f64; 2];
        let mut primal = vec![0.0_f64; dec_start + 2];

        // Stage 0: incoming=[10.0, 20.0], decision=30.0 → slot 0=20.0, slot 1=30.0
        incoming.copy_from_slice(&state[ant_start..ant_start + 2]);
        primal[dec_start] = 30.0;
        shift_anticipated_state(&mut state, &incoming, &primal, &indexer);
        assert_eq!(state[ant_start], 20.0);
        assert_eq!(state[ant_start + 1], 30.0);

        // Stage 1: incoming=[20.0, 30.0], decision=40.0 → slot 0=30.0, slot 1=40.0
        incoming.copy_from_slice(&state[ant_start..ant_start + 2]);
        primal[dec_start] = 40.0;
        shift_anticipated_state(&mut state, &incoming, &primal, &indexer);
        assert_eq!(state[ant_start], 30.0);
        assert_eq!(state[ant_start + 1], 40.0);

        // Stage 2: incoming=[30.0, 40.0], decision=50.0 → slot 0=40.0, slot 1=50.0
        incoming.copy_from_slice(&state[ant_start..ant_start + 2]);
        primal[dec_start] = 50.0;
        shift_anticipated_state(&mut state, &incoming, &primal, &indexer);
        assert_eq!(state[ant_start], 40.0);
        assert_eq!(state[ant_start + 1], 50.0);
    }

    // ── active-subset NCS column/bound mapping ───────────────────────────────

    /// Strict-subset stage: slot 0 inactive (`None`), slots 1,2 active at local
    /// columns 0,1. The emitted index/bound buffers have length
    /// `active_stochastic_ncs * n_blks = 2 * 2 = 4`, NOT `n_stochastic_ncs *
    /// n_blks = 3 * 2 = 6` — the inactive slot contributes zero entries (AC #2).
    /// Indices stride by `ncs_local` from the per-stage base, and the gathered
    /// bounds copy only the active slots' blocks from the full slot-order buffers.
    #[test]
    fn active_ncs_subset_buffers_have_active_length() {
        let n_blks = 2_usize;
        let ncs_col_start = 100_usize;
        // slot 0: inactive; slot 1 → local 0; slot 2 → local 1.
        let slot_to_local = vec![None, Some(0_usize), Some(1_usize)];
        assert_eq!(active_ncs_slot_count(&slot_to_local), 2);

        let mut indices = Vec::new();
        build_active_ncs_col_indices(&slot_to_local, ncs_col_start, n_blks, &mut indices);
        // local 0 → cols 100,101; local 1 → cols 102,103. Inactive slot 0: nothing.
        assert_eq!(indices, vec![100, 101, 102, 103]);
        assert_eq!(indices.len(), 2 * n_blks, "index buffer is active*n_blks");

        // Full slot-order bounds: one 2-block stride per stochastic slot (3 slots).
        let lower_src = vec![0.0, 0.0, 10.0, 11.0, 20.0, 21.0];
        let upper_src = vec![1.0, 2.0, 13.0, 14.0, 23.0, 24.0];
        let mut lower_out = Vec::new();
        let mut upper_out = Vec::new();
        gather_active_ncs_bounds(
            &slot_to_local,
            n_blks,
            &lower_src,
            &upper_src,
            &mut lower_out,
            &mut upper_out,
        );
        // Only slots 1,2 are gathered (slot 0 skipped): blocks [10,11,20,21] /
        // [13,14,23,24].
        assert_eq!(lower_out, vec![10.0, 11.0, 20.0, 21.0]);
        assert_eq!(upper_out, vec![13.0, 14.0, 23.0, 24.0]);
        assert_eq!(lower_out.len(), 2 * n_blks);
        assert_eq!(upper_out.len(), indices.len(), "bounds parallel to indices");
    }

    /// All-active, slot order == active order: the gathered bounds and built
    /// indices are bit-identical to a verbatim full `n_stochastic_ncs` block —
    /// the hash-neutrality contract for every existing case.
    #[test]
    fn active_ncs_full_set_is_slot_order_identical() {
        let n_blks = 2_usize;
        let ncs_col_start = 0_usize;
        let slot_to_local = vec![Some(0_usize), Some(1_usize), Some(2_usize)];
        assert_eq!(active_ncs_slot_count(&slot_to_local), 3);

        let mut indices = Vec::new();
        build_active_ncs_col_indices(&slot_to_local, ncs_col_start, n_blks, &mut indices);
        // Identical to the old `for slot { for blk { start + slot*n_blks + blk }}`.
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5]);

        let lower_src = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let upper_src = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut lower_out = Vec::new();
        let mut upper_out = Vec::new();
        gather_active_ncs_bounds(
            &slot_to_local,
            n_blks,
            &lower_src,
            &upper_src,
            &mut lower_out,
            &mut upper_out,
        );
        // Verbatim copy of the full slot-order buffers.
        assert_eq!(lower_out, lower_src);
        assert_eq!(upper_out, upper_src);
    }
}
