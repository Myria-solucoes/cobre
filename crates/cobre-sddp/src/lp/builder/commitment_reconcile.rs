//! Reconcile a pinned anticipated commitment against the delivery-stage
//! generation bounds the fishing equality couples it to.
//!
//! # Why this is permanent
//!
//! A carried commitment is the solver's computed value for a **basic** ring-slot
//! column, so it is accurate only to the backend's `primal_feasibility_tolerance`
//! (`1e-9` on both `HiGHS` and CLP) — never to 1 ULP. Re-pinned as a hard equality at
//! the delivery stage and coupled by `Σ_b h_b·gen_b = H·commitment` to generation
//! columns that keep their own `[min_gen, max_gen]`, a commitment that drifted a
//! hair past the cap renders the LP infeasible over a physically meaningless
//! quantity.
//!
//! This cannot be fixed by construction. `col_scale = 1.0` on the ring columns
//! (`apply_anticipated_col_scale_unscale`) removes the *carry* drift and is
//! retained, but the deposit `slot_out − decision = 0` produces `slot_out` through
//! the basis factorization: exactness is the solver's to give, and it does not give
//! it. Deleting this reconciliation on the premise that unscaling made it redundant
//! reintroduces a hard `Infeasible` abort on any study whose commitment reaches its
//! generation cap.
//!
//! # What it refuses to do
//!
//! Absorb a real over-commitment. Drift beyond [`drift_margin`] is
//! [`SddpError::AnticipatedCommitmentOutOfBounds`], never relaxed — the margin is
//! the discrimination line between solver noise and a modelling error.

use cobre_solver::StageTemplate;

use crate::error::SddpError;
use crate::indexer::{AnticipatedLocal, StateSpace, anticipated_resolution_for};

use super::template::StageGeometry;

/// Relative headroom admitted around a drifted commitment. Deliberately two
/// orders above the backends' `primal_feasibility_tolerance` (`1e-9`): that
/// tolerance bounds the scaled residuals, while the carried value is a basic
/// variable whose error is the residual amplified by the basis conditioning —
/// severalfold past `1e-9` on production-scale LPs. Tightening this back to
/// the raw tolerance re-aborts real studies over sub-watt drift.
const COMMITMENT_DRIFT_REL: f64 = 1e-7;

/// Absolute headroom floor, in MW. Keeps the margin above the solver's own
/// feasibility tolerance once the generation column's `col_scale` divides it, which
/// a purely relative term fails to do for a commitment near zero.
const COMMITMENT_DRIFT_ABS: f64 = 1e-5;

/// Headroom bounding solver-tolerance drift on `commitment`.
#[must_use]
pub(crate) fn drift_margin(commitment: f64) -> f64 {
    commitment
        .abs()
        .mul_add(COMMITMENT_DRIFT_REL, COMMITMENT_DRIFT_ABS)
}

/// Outcome of reconciling one pinned commitment against one generation column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Reconciliation {
    /// Within the column's enforced bounds; emit no patch.
    InBounds,
    /// Outside by no more than [`drift_margin`]; relax the column's scaled bounds.
    Relaxed {
        /// Relaxed scaled lower bound.
        lower_scaled: f64,
        /// Relaxed scaled upper bound.
        upper_scaled: f64,
    },
    /// Outside by more than [`drift_margin`] — not solver drift.
    Violation {
        /// Enforced bound crossed, in physical units.
        bound: f64,
        /// Distance past `bound`; always positive.
        drift: f64,
    },
}

/// Classify `commitment` against a generation column's **enforced** bounds.
///
/// The enforced bound is `bound_scaled * scale`, not the template's raw
/// `[min_gen, max_gen]`: the scaled bound is what the solver actually applies, and
/// its own round-trip shifts it. Comparing against the raw value would leave a
/// residual mismatch this is meant to close.
#[must_use]
pub(crate) fn reconcile_commitment(
    commitment: f64,
    lower_scaled: f64,
    upper_scaled: f64,
    scale: f64,
) -> Reconciliation {
    let max_gen = upper_scaled * scale;
    let min_gen = lower_scaled * scale;

    let over = commitment - max_gen;
    let under = min_gen - commitment;
    if over <= 0.0 && under <= 0.0 {
        return Reconciliation::InBounds;
    }

    let margin = drift_margin(commitment);
    if over > 0.0 {
        if over > margin {
            return Reconciliation::Violation {
                bound: max_gen,
                drift: over,
            };
        }
        return Reconciliation::Relaxed {
            lower_scaled,
            upper_scaled: (commitment + margin) / scale,
        };
    }
    if under > margin {
        return Reconciliation::Violation {
            bound: min_gen,
            drift: under,
        };
    }
    Reconciliation::Relaxed {
        lower_scaled: (commitment - margin) / scale,
        upper_scaled,
    }
}

/// Reusable column-bound relaxations for one stage solve. Empty for every solve
/// whose commitments are in bounds, which is the overwhelming majority — an empty
/// set means the caller issues no `set_col_bounds` and numerics are untouched.
#[derive(Debug, Default, Clone)]
pub struct BoundRelaxations {
    /// Generation columns to relax.
    pub indices: Vec<usize>,
    /// Relaxed scaled lower bounds, parallel to [`Self::indices`].
    pub lower: Vec<f64>,
    /// Relaxed scaled upper bounds, parallel to [`Self::indices`].
    pub upper: Vec<f64>,
}

impl BoundRelaxations {
    /// Whether this solve needs no relaxation.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    fn clear(&mut self) {
        self.indices.clear();
        self.lower.clear();
        self.upper.clear();
    }

    fn push(&mut self, col: usize, lower_scaled: f64, upper_scaled: f64) {
        self.indices.push(col);
        self.lower.push(lower_scaled);
        self.upper.push(upper_scaled);
    }
}

/// The delivery-stage inputs [`fill_bound_relaxations`] reconciles.
pub(crate) struct DeliveryPins<'a> {
    /// Owns `anticipated_slots_out` and the per-plant resolution gate.
    pub state_layout: &'a StateSpace,
    /// State this solve pins; its slot-0 entries are the commitments.
    pub pinned_state: &'a [f64],
    /// This stage's template: the source of `col_scale` and the scaled bounds.
    pub template: &'a StageTemplate,
    /// This stage's column geometry, owning `thermal.start`.
    pub geometry: &'a StageGeometry,
    /// Anticipated-local → `system.thermals[]` position.
    pub anticipated_thermal_indices: &'a [usize],
    /// Blocks at this stage.
    pub n_blks: usize,
    /// Stage being solved.
    pub stage_idx: usize,
    /// Study stage count.
    pub n_stages: usize,
}

/// Fill `out` with the generation-column bound relaxations this stage's pinned
/// commitments require.
///
/// Gates each plant on `is_anticipated_at(stage_idx)` — the same gate the LP builder
/// uses to emit the fishing row. A stage with no fishing coupling leaves the plant's
/// generation column untouched.
///
/// # Errors
///
/// [`SddpError::AnticipatedCommitmentOutOfBounds`] when a commitment lies further
/// outside than [`drift_margin`] admits.
///
/// # Panics (debug builds only)
///
/// Panics if `pins.pinned_state.len() != pins.state_layout.n_state`.
pub(crate) fn fill_bound_relaxations(
    pins: &DeliveryPins<'_>,
    out: &mut BoundRelaxations,
) -> Result<(), SddpError> {
    out.clear();
    if pins.anticipated_thermal_indices.is_empty() {
        return Ok(());
    }
    debug_assert_eq!(
        pins.pinned_state.len(),
        pins.state_layout.n_state,
        "pinned state length {got} != n_state {expected}",
        got = pins.pinned_state.len(),
        expected = pins.state_layout.n_state,
    );

    let slot0_base = pins.state_layout.anticipated_slots_out.start;
    let col_scale = &pins.template.col_scale;

    for (local, &thermal_idx) in pins.anticipated_thermal_indices.iter().enumerate() {
        if !anticipated_resolution_for(
            pins.state_layout,
            AnticipatedLocal::new(local),
            pins.n_stages,
        )
        .is_anticipated_at(pins.stage_idx)
        {
            continue;
        }
        let commitment = pins.pinned_state[slot0_base + local];
        for blk in 0..pins.n_blks {
            let gen_col = pins.geometry.thermal.start + thermal_idx * pins.n_blks + blk;
            let scale = if col_scale.is_empty() {
                1.0
            } else {
                col_scale[gen_col]
            };
            match reconcile_commitment(
                commitment,
                pins.template.col_lower[gen_col],
                pins.template.col_upper[gen_col],
                scale,
            ) {
                Reconciliation::InBounds => {}
                Reconciliation::Relaxed {
                    lower_scaled,
                    upper_scaled,
                } => out.push(gen_col, lower_scaled, upper_scaled),
                Reconciliation::Violation { bound, drift } => {
                    return Err(SddpError::AnticipatedCommitmentOutOfBounds {
                        stage: pins.stage_idx,
                        thermal_index: thermal_idx,
                        block: blk,
                        commitment,
                        bound,
                        drift,
                    });
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
