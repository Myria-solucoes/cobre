use super::{
    BoundRelaxations, DeliveryPins, Reconciliation, StageGeometry, drift_margin,
    fill_bound_relaxations, reconcile_commitment,
};
use crate::indexer::StateSpace;
use cobre_solver::StageTemplate;

/// A generation column's scale is well below 1, so its bound round-trips through
/// `raw / scale * scale`; any value here exercises that round-trip.
const GEN_SCALE: f64 = 8.9e-2;

/// Largest bound violation a backend may report as feasible
/// (`primal_feasibility_tolerance`, `1e-9` on `HiGHS` and CLP). A carried commitment is
/// a computed basic-variable value, so it can legitimately land this far out.
const SOLVER_PRIMAL_TOLERANCE: f64 = 1e-9;

#[test]
fn commitment_inside_bounds_needs_no_patch() {
    assert_eq!(
        reconcile_commitment(500.0, 0.0, 1000.0, 1.0),
        Reconciliation::InBounds,
        "an in-bounds commitment must emit no patch, so parity is untouched"
    );
}

#[test]
fn commitment_exactly_at_cap_needs_no_patch() {
    assert_eq!(
        reconcile_commitment(1000.0, 0.0, 1000.0, 1.0),
        Reconciliation::InBounds,
        "a commitment AT the cap is feasible unrelaxed; patching it would move parity"
    );
}

/// The regression this module exists for: a commitment carried a hair past a scaled
/// cap must still solve. Without the relaxation the delivery LP is infeasible over a
/// physically meaningless quantity, aborting training.
#[test]
fn solver_tolerance_drift_past_cap_is_absorbed() {
    let raw_cap = 1_358.651_007_84;
    let upper_scaled = raw_cap / GEN_SCALE;
    let enforced_cap = upper_scaled * GEN_SCALE;
    let commitment = enforced_cap + SOLVER_PRIMAL_TOLERANCE;

    match reconcile_commitment(commitment, 0.0, upper_scaled, GEN_SCALE) {
        Reconciliation::Relaxed { upper_scaled, .. } => assert!(
            upper_scaled * GEN_SCALE >= commitment,
            "the relaxed bound must admit the commitment after its own scale \
             round-trip: {} < {commitment}",
            upper_scaled * GEN_SCALE
        ),
        other => {
            panic!("drift within the solver's primal tolerance must be absorbed, got {other:?}")
        }
    }
}

#[test]
fn solver_tolerance_drift_below_floor_is_absorbed() {
    let commitment = 100.0 - SOLVER_PRIMAL_TOLERANCE;
    match reconcile_commitment(commitment, 100.0, 1000.0, 1.0) {
        Reconciliation::Relaxed { lower_scaled, .. } => assert!(
            lower_scaled <= commitment,
            "the relaxed floor must admit the drifted commitment"
        ),
        other => panic!("expected Relaxed, got {other:?}"),
    }
}

/// The margin discriminates solver noise from a modelling error. Absorbing any
/// overshoot silently admits a plant generating past its cap.
#[test]
fn genuine_over_commitment_is_a_violation_not_a_relaxation() {
    match reconcile_commitment(1500.0, 0.0, 1000.0, 1.0) {
        Reconciliation::Violation { bound, drift } => {
            assert!((bound - 1000.0).abs() < 1e-12, "bound must be the cap");
            assert!((drift - 500.0).abs() < 1e-12, "drift must be the overshoot");
        }
        other => panic!("500 MW over cap must be a Violation, got {other:?}"),
    }
}

#[test]
fn genuine_under_commitment_is_a_violation() {
    match reconcile_commitment(10.0, 100.0, 1000.0, 1.0) {
        Reconciliation::Violation { bound, drift } => {
            assert!((bound - 100.0).abs() < 1e-12);
            assert!((drift - 90.0).abs() < 1e-12);
        }
        other => panic!("90 MW under the floor must be a Violation, got {other:?}"),
    }
}

#[test]
fn margin_is_the_discrimination_line() {
    let cap = 1000.0;
    let margin = drift_margin(cap);

    let inside = reconcile_commitment(margin.mul_add(0.5, cap), 0.0, cap, 1.0);
    assert!(
        matches!(inside, Reconciliation::Relaxed { .. }),
        "drift within the margin must be absorbed, got {inside:?}"
    );

    let outside = reconcile_commitment(margin.mul_add(2.0, cap), 0.0, cap, 1.0);
    assert!(
        matches!(outside, Reconciliation::Violation { .. }),
        "drift past the margin must be refused, got {outside:?}"
    );
}

/// A basic-variable value drifts past the RESIDUAL tolerance by the basis
/// conditioning: this production-observed commitment landed `3.77e-6` MW over
/// a `1593.04` MW enforced cap (`2.4e-9` relative — noise, not a modelling
/// error) and must be absorbed, not refused. Pins the margin's headroom above
/// the raw solver tolerance.
#[test]
fn factorization_drift_severalfold_past_tolerance_is_absorbed() {
    let commitment = 1_593.040_683_768_330_6;
    let enforced_cap = 1_593.040_679_999_999_8;

    match reconcile_commitment(commitment, 0.0, enforced_cap, 1.0) {
        Reconciliation::Relaxed { upper_scaled, .. } => assert!(
            upper_scaled >= commitment,
            "the relaxed bound must admit the drifted commitment"
        ),
        other => {
            panic!("severalfold-tolerance factorization drift must be absorbed, got {other:?}")
        }
    }
}

/// The margin must cover what the backend may actually emit, or the reconciliation
/// rejects the very drift it exists to absorb.
#[test]
fn margin_covers_the_solver_primal_tolerance_at_every_magnitude() {
    for commitment in [0.0, 1.0, 1e3, 1e6] {
        assert!(
            drift_margin(commitment) >= SOLVER_PRIMAL_TOLERANCE,
            "margin at {commitment} must admit a bound violation the solver itself \
             reports as feasible"
        );
    }
}

/// A commitment near zero has a vanishing relative margin; the absolute floor is what
/// keeps the headroom above the solver's tolerance.
#[test]
fn absolute_floor_covers_a_near_zero_commitment() {
    let got = reconcile_commitment(SOLVER_PRIMAL_TOLERANCE, 0.0, 0.0, 1.0);
    assert!(
        matches!(got, Reconciliation::Relaxed { .. }),
        "sub-tolerance drift off a zero cap must be absorbed, got {got:?}"
    );
}

/// The enforced bound is `scaled * scale`, not the template's raw value; comparing
/// against the raw number would leave the residual this module closes.
#[test]
fn comparison_uses_the_scaled_round_tripped_bound() {
    let upper_scaled = 1_358.651_007_84 / GEN_SCALE;
    let enforced = upper_scaled * GEN_SCALE;

    assert_eq!(
        reconcile_commitment(enforced, 0.0, upper_scaled, GEN_SCALE),
        Reconciliation::InBounds,
        "a commitment at the ENFORCED bound is in bounds by definition"
    );
}

// =========================================================================
// fill_bound_relaxations / DeliveryPins: hold-slot re-homing
// =========================================================================

fn minimal_gen_template(gen_col_upper: f64) -> StageTemplate {
    StageTemplate {
        num_cols: 1,
        num_rows: 0,
        num_nz: 0,
        col_starts: vec![0, 0],
        row_indices: Vec::new(),
        values: Vec::new(),
        col_lower: vec![0.0],
        col_upper: vec![gen_col_upper],
        objective: vec![0.0],
        row_lower: Vec::new(),
        row_upper: Vec::new(),
        n_transfer: 0,
        n_dual_relevant: 0,
        n_hydro: 0,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
        n_state: 0,
    }
}

/// `A=1`, `k_max=2`: the in-study ring holds two DIFFERENT commitments, one per
/// slot. Delivery/decision stage `3` resolves to the modular slot `3 % 2 = 1`
/// (`StateSpace::commitment_hold_in_study_offset`) — never slot `0`, the
/// retired shift-ring's always-maturing-here convention. Slot `1` carries a
/// commitment drifted within [`drift_margin`] past the generation column's
/// cap; slot `0` carries an unrelated, deep-in-bounds value that would wrongly
/// short-circuit to [`Reconciliation::InBounds`] (no relaxation pushed) were
/// the retired slot-0 read still in place.
#[test]
fn fill_bound_relaxations_relaxes_the_stage_modular_hold_slot() {
    let state_layout = StateSpace::new(0, 0, 0, Vec::new(), 1, 2, vec![2], &[]);
    assert_eq!(state_layout.commit_out, 0..2);
    assert_eq!(state_layout.n_state, 2);

    let cap = 1_000.0_f64;
    let margin = drift_margin(cap);
    let drifted = margin.mul_add(0.5, cap);
    let pinned_state = vec![500.0_f64, drifted];

    let template = minimal_gen_template(cap);
    let geometry = StageGeometry {
        thermal: 0..1,
        ..StageGeometry::default()
    };
    let anticipated_thermal_indices = [0_usize];

    let pins = DeliveryPins {
        state_layout: &state_layout,
        pinned_state: &pinned_state,
        template: &template,
        geometry: &geometry,
        anticipated_thermal_indices: &anticipated_thermal_indices,
        n_blks: 1,
        stage_idx: 3,
        n_stages: 5,
    };

    let mut relaxations = BoundRelaxations::default();
    fill_bound_relaxations(&pins, &mut relaxations)
        .expect("drift within the margin must not be refused");

    assert_eq!(
        relaxations.indices,
        vec![0],
        "the thermal's single generation column must be relaxed, proving slot 1 \
         (not the retired always-slot-0 read) was reconciled"
    );
    assert!(
        relaxations.upper[0] >= drifted,
        "the relaxed bound must admit the drifted commitment: {} < {drifted}",
        relaxations.upper[0]
    );
}
