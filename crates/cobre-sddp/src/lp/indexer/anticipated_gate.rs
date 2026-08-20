//! Anticipated-decision temporal gating: horizon and commissioning-window
//! predicates, and the delivery-anchored resolution lookup they key on.
//!
//! These are free functions, not [`StateSpace`] methods — the type owns
//! state-vector column geometry, not the horizon/commissioning gate that
//! decides whether an anticipated decision exists at a given stage. Each
//! function takes `state: &StateSpace` for uniformity with its siblings,
//! reading the two fields [`StateSpace::anticipated_lead_stages`] (`pub`)
//! and `anticipated_resolution` (`pub(crate)`) it already carries where
//! needed; nothing new is threaded.

use std::borrow::Cow;

use super::{AnticipatedLocal, StateSpace};
use crate::lead_time::LeadTime::Stages;
use crate::lead_time::PointResolution;
use crate::lead_time::{DeliveryAxis, resolve_point};

use cobre_core::commissioning::commissioning_active;

/// Whether anticipated plant `local_idx` emits a decision column and an
/// `anticipated_state_out_def` row at `stage_idx` — the single cross-module
/// owner of the anticipated-decision gate. Both clauses key on the
/// **delivery** stage `t + K_i`:
///
/// 1. **Strict extended-calendar bound** — `stage_idx + K_i < n_delivery`,
///    against the extended delivery calendar (study stages plus the
///    synthetic post-study continuation), not merely `n_stages`. The `<` is
///    strict: a `<=` would price a commitment delivered at `n_delivery`,
///    outside `[0, n_delivery)`, with no delivery LP.
/// 2. **Operation window** — the delivery stage is commissioning-active,
///    `commissioning_active(entry_i, exit_i, id(t + K_i))`, resolved through
///    `delivery_stage_ids` UNIFORMLY at every delivery stage, study or
///    post-study alike — no `delivery_stage < n_stages` conjunct exempts a
///    post-study delivery from this check: a plant whose `exit_stage_id`
///    lies inside the study is inactive for every delivery after it,
///    in-study or not (pinned by
///    `is_anticipated_decision_active_for_delivery_uniform_gate_rejects_post_study_delivery_for_exited_plant`).
///    Keying on the DELIVERY stage, not the
///    decision stage `t`, is separately load-bearing: a pre-entry decision
///    at `entry − K_i` legitimately delivers at `entry`, and keying on `t`
///    would invert which decisions are active.
///
/// `anticipated_windows` is indexed by anticipated-local position;
/// `delivery_stage_ids` by delivery stage index.
///
/// Test-only: every production call site resolves its delivery stage via
/// `anticipated_resolution_for` and gates through
/// [`is_anticipated_decision_active_for_delivery`] directly, never this
/// constant-lead wrapper.
#[cfg(test)]
#[inline]
#[must_use]
pub(crate) fn is_anticipated_decision_active(
    state: &StateSpace,
    local_idx: usize,
    stage_idx: usize,
    n_stages: usize,
    anticipated_windows: &[(Option<i32>, Option<i32>)],
    study_stage_ids: &[i32],
) -> bool {
    debug_assert!(
        local_idx < state.anticipated_lead_stages.len(),
        "local_idx {local_idx} out of bounds (n_anticipated = {})",
        state.anticipated_lead_stages.len(),
    );
    debug_assert_eq!(
        anticipated_windows.len(),
        state.anticipated_lead_stages.len(),
        "anticipated_windows must have one entry per anticipated plant",
    );
    let delivery_stage = stage_idx.saturating_add(state.anticipated_lead_stages[local_idx]);
    is_anticipated_decision_active_for_delivery(
        state,
        AnticipatedLocal::new(local_idx),
        delivery_stage,
        n_stages,
        anticipated_windows,
        study_stage_ids,
    )
}

/// Whether plant `local_idx`'s commitment maturing at an EXPLICIT
/// `delivery_stage` is active — the same horizon + commissioning gate as
/// `is_anticipated_decision_active` (test-only), for a decision whose
/// delivery stage comes from `PointResolution::genuine_decisions_at` rather
/// than a constant lead offset.
#[inline]
#[must_use]
pub(crate) fn is_anticipated_decision_active_for_delivery(
    _state: &StateSpace,
    local_idx: AnticipatedLocal,
    delivery_stage: usize,
    n_delivery: usize,
    anticipated_windows: &[(Option<i32>, Option<i32>)],
    delivery_stage_ids: &[i32],
) -> bool {
    let local_idx = local_idx.get();
    debug_assert!(
        local_idx < anticipated_windows.len(),
        "local_idx {local_idx} out of bounds (anticipated_windows.len() = {})",
        anticipated_windows.len(),
    );
    if delivery_stage >= n_delivery {
        return false;
    }
    debug_assert!(
        delivery_stage < delivery_stage_ids.len(),
        "delivery_stage {delivery_stage} out of bounds for delivery_stage_ids \
         (len {})",
        delivery_stage_ids.len(),
    );
    let (entry, exit) = anticipated_windows[local_idx];
    commissioning_active(entry, exit, delivery_stage_ids[delivery_stage])
}

/// Plant `local_idx`'s delivery-anchored resolution: the setup-threaded
/// [`crate::lead_time::PointResolution`] when
/// [`StateSpace::anticipated_resolution`] is attached (production always
/// attaches it), else an on-the-fly `LeadTime::Stages`-equivalent built from
/// the plant's constant [`StateSpace::anticipated_lead_stages`] — the
/// fixture fallback for constant-lead tests that thread no resolution.
#[must_use]
pub(crate) fn anticipated_resolution_for(
    state: &StateSpace,
    local_idx: AnticipatedLocal,
    n_stages: usize,
) -> Cow<'_, PointResolution> {
    let local_idx = local_idx.get();
    if !state.anticipated_resolution.per_plant.is_empty() {
        return Cow::Borrowed(&state.anticipated_resolution.per_plant[local_idx]);
    }
    let lead = u32::try_from(state.anticipated_lead_stages[local_idx]).unwrap_or(u32::MAX);
    Cow::Owned(resolve_point(
        Stages(lead),
        DeliveryAxis {
            stage_lengths_hours: &[],
            n_decision: n_stages,
            n_delivery: n_stages,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        AnticipatedLocal, StateSpace, is_anticipated_decision_active,
        is_anticipated_decision_active_for_delivery,
    };

    /// Build a [`StateSpace`] for the gating tests below — both fixtures use
    /// `hydro_count == 0`, matching `state_space.rs`'s `finalized` helper
    /// under that fixture shape.
    fn ant_layout(n_anticipated: usize, k_max: usize, leads: Vec<usize>) -> StateSpace {
        StateSpace::new(0, 0, 0, Vec::new(), n_anticipated, k_max, leads, &[])
    }

    /// The strict horizon clause is active iff `stage_idx + K_i < n_stages`:
    /// active strictly inside the horizon, inactive at the `==` boundary and
    /// beyond. The `==` boundary must be inactive (a `<=` gate would price a
    /// commitment whose delivery stage falls outside `[0, n_stages)`). With
    /// windowless plants `(None, None)` the operation-window clause is always
    /// `true`, so this isolates the horizon clause.
    #[test]
    fn is_anticipated_decision_active_strict_horizon_gate() {
        // Two plants: K_0 = 1, K_1 = 2; k_max = 2; n_stages = 5.
        let idx = ant_layout(2, 2, vec![1, 2]);
        let n_stages = 5;
        // Windowless: the operation-window clause is identically true.
        let windows = [(None, None); 2];
        let stage_ids = [0, 1, 2, 3, 4];

        // Plant 0 (K=1): active while stage_idx + 1 < 5, i.e. stage_idx <= 3.
        assert!(is_anticipated_decision_active(
            &idx, 0, 0, n_stages, &windows, &stage_ids
        ));
        assert!(is_anticipated_decision_active(
            &idx, 0, 3, n_stages, &windows, &stage_ids
        ));
        // == boundary (stage_idx + K_i == n_stages): inactive.
        assert!(!is_anticipated_decision_active(
            &idx, 0, 4, n_stages, &windows, &stage_ids
        ));
        // Beyond the boundary: inactive.
        assert!(!is_anticipated_decision_active(
            &idx, 0, 5, n_stages, &windows, &stage_ids
        ));

        // Plant 1 (K=2): active while stage_idx + 2 < 5, i.e. stage_idx <= 2.
        assert!(is_anticipated_decision_active(
            &idx, 1, 2, n_stages, &windows, &stage_ids
        ));
        // == boundary: inactive.
        assert!(!is_anticipated_decision_active(
            &idx, 1, 3, n_stages, &windows, &stage_ids
        ));
        // Beyond: inactive.
        assert!(!is_anticipated_decision_active(
            &idx, 1, 4, n_stages, &windows, &stage_ids
        ));
    }

    /// The operation-window clause gates on the DELIVERY stage's `stage.id`, not
    /// the decision stage. A single plant (K=2) with window `[entry=2, exit=4)`
    /// over a 6-stage horizon (stage ids `[0..6)`):
    ///
    /// - decision at `t=0` delivers at `id(2)=2` ∈ `[2,4)` → ACTIVE (pre-entry
    ///   decision: priced at `entry − K`),
    /// - decision at `t=1` delivers at `id(3)=3` ∈ `[2,4)` → ACTIVE,
    /// - decision at `t=2` delivers at `id(4)=4` ∉ `[2,4)` (half-open exit) →
    ///   INACTIVE (post-exit drain begins),
    /// - decision at `t=3` delivers at `id(5)=5` ∉ `[2,4)` → INACTIVE,
    /// - decision at `t < 0` not applicable; decision at `t=4` (`t+K=6 == n`) is
    ///   horizon-inactive regardless of window.
    ///
    /// The forbidden alternative — keying on the decision stage `t` — would make
    /// `t=2` (inside `[2,4)`) active and `t=0` (outside) inactive, the exact
    /// inversion the shift exists to prevent.
    #[test]
    fn is_anticipated_decision_active_delivery_stage_window_gate() {
        // One plant, K=2, k_max=2; n_stages=6; window [entry=2, exit=4).
        let idx = ant_layout(1, 2, vec![2]);
        let n_stages = 6;
        let windows = [(Some(2), Some(4))];
        let stage_ids = [0, 1, 2, 3, 4, 5];

        // Pre-entry decision (t=0): delivers at id 2 ∈ [2,4) → active.
        assert!(is_anticipated_decision_active(
            &idx, 0, 0, n_stages, &windows, &stage_ids
        ));
        // t=1: delivers at id 3 ∈ [2,4) → active.
        assert!(is_anticipated_decision_active(
            &idx, 0, 1, n_stages, &windows, &stage_ids
        ));
        // t=2: delivers at id 4 ∉ [2,4) (half-open exit) → inactive (drain).
        assert!(!is_anticipated_decision_active(
            &idx, 0, 2, n_stages, &windows, &stage_ids
        ));
        // t=3: delivers at id 5 ∉ [2,4) → inactive.
        assert!(!is_anticipated_decision_active(
            &idx, 0, 3, n_stages, &windows, &stage_ids
        ));
        // t=4: t+K=6 == n_stages → horizon-inactive (short-circuits before window).
        assert!(!is_anticipated_decision_active(
            &idx, 0, 4, n_stages, &windows, &stage_ids
        ));
    }

    /// The strict bound now gates against `n_delivery`, the extended
    /// delivery-calendar width, not `n_stages`. On a study-only axis
    /// (`n_delivery == n_stages`, as here) this is byte-identical to the
    /// pre-generalization bound: active strictly inside `[0, n_delivery)`,
    /// inactive at the `==` boundary and beyond. The `<` stays strict — a
    /// `<=` would price a commitment delivered at `n_delivery`, one past the
    /// last defined delivery stage.
    #[test]
    fn is_anticipated_decision_active_for_delivery_strict_extended_bound() {
        let idx = ant_layout(1, 1, vec![1]);
        let n_delivery = 5;
        let windows = [(None, None)];
        let delivery_stage_ids = [0, 1, 2, 3, 4];

        assert!(is_anticipated_decision_active_for_delivery(
            &idx,
            AnticipatedLocal::new(0),
            4,
            n_delivery,
            &windows,
            &delivery_stage_ids,
        ));
        assert!(!is_anticipated_decision_active_for_delivery(
            &idx,
            AnticipatedLocal::new(0),
            5,
            n_delivery,
            &windows,
            &delivery_stage_ids,
        ));
    }

    /// A post-study delivery (`delivery_stage >= n_stages`, still `<
    /// n_delivery`) is admissible on the extended axis: five study stages
    /// (`ids [0..5)`) followed by three synthetic post-study stages
    /// (`ids [5..8)`), a windowless plant, delivery at stage 6.
    #[test]
    fn is_anticipated_decision_active_for_delivery_post_study_delivery_admitted_for_windowless_plant()
     {
        let idx = ant_layout(1, 1, vec![1]);
        let n_delivery = 8;
        let windows = [(None, None)];
        let delivery_stage_ids = [0, 1, 2, 3, 4, 5, 6, 7];

        assert!(is_anticipated_decision_active_for_delivery(
            &idx,
            AnticipatedLocal::new(0),
            6,
            n_delivery,
            &windows,
            &delivery_stage_ids,
        ));
    }

    /// The uniform gate: a post-study delivery is commissioning-gated
    /// against the CONTINUED SYNTHETIC id, with no `delivery_stage <
    /// n_stages` exemption. A plant with `(entry, exit) == (Some(0),
    /// Some(5))` has exited before synthetic id `6`, so its post-study
    /// delivery at `delivery_stage == 6` is inactive; reintroducing a
    /// post-study commissioning exemption would have to delete this test.
    #[test]
    fn is_anticipated_decision_active_for_delivery_uniform_gate_rejects_post_study_delivery_for_exited_plant()
     {
        let idx = ant_layout(1, 1, vec![1]);
        let n_delivery = 8;
        let windows = [(Some(0), Some(5))];
        let delivery_stage_ids = [0, 1, 2, 3, 4, 5, 6, 7];

        assert!(!is_anticipated_decision_active_for_delivery(
            &idx,
            AnticipatedLocal::new(0),
            6,
            n_delivery,
            &windows,
            &delivery_stage_ids,
        ));
    }
}
