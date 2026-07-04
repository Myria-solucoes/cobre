use super::*;

const TOL: f64 = 1e-9;

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: got {actual:?}, expected {expected:?}"
    );
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() < TOL,
            "value mismatch: got {actual:?}, expected {expected:?}"
        );
    }
}

// Weekly anchor (168h stages), t_v=360h: stage_weights[0]=stage_weights[1]=0,
// stage_weights[2]=6/7, stage_weights[3]=1/7, depth 3.
#[test]
fn test_s1a_matches_decomp_fig_5_5b() {
    let stage_lengths_hours = [168.0; 6];
    let resolution = resolve_spread(360.0, 0, &stage_lengths_hours, None);

    assert_eq!(resolution.stage_reach, 3);
    assert_close(&resolution.stage_weights, &[0.0, 0.0, 6.0 / 7.0, 1.0 / 7.0]);
    assert!((resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < TOL);
}

#[test]
fn test_s1d_monthly_to_weekly_counterexample() {
    let stage_lengths_hours = [720.0, 168.0, 168.0, 168.0, 168.0];
    let resolution = resolve_spread(360.0, 0, &stage_lengths_hours, None);

    assert_eq!(resolution.stage_reach, 3);
    assert_close(
        &resolution.stage_weights,
        &[1.0 / 2.0, 7.0 / 30.0, 7.0 / 30.0, 1.0 / 30.0],
    );
    assert!((resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < TOL);

    // The closed-form ceiling `⌈360/720⌉ == 1` would drop the lag-2/lag-3
    // mass; only the general max-reach depth is correct here.
    assert_ne!(resolution.stage_reach, 1);
}

#[test]
fn test_example_iii_block_factors() {
    let stage_lengths_hours = [720.0, 720.0];
    let block_lengths_hours = [240.0, 240.0, 240.0];
    let resolution = resolve_spread(250.0, 0, &stage_lengths_hours, Some(&block_lengths_hours));

    assert_eq!(resolution.stage_reach, 1);
    assert_close(&resolution.stage_weights, &[470.0 / 720.0, 250.0 / 720.0]);

    assert_close(&resolution.block_deposits[0], &[1.0, 0.0]);
    assert_close(
        &resolution.block_deposits[1],
        &[230.0 / 240.0, 10.0 / 240.0],
    );
    assert_close(&resolution.block_deposits[2], &[0.0, 1.0]);

    // within_stage_routing[b] is self-inclusive: index 0 == block b routing
    // to itself, so the two nonzero downstream entries are
    // within_stage_routing[0][1..].
    assert_close(
        &resolution.within_stage_routing[0],
        &[0.0, 230.0 / 240.0, 10.0 / 240.0],
    );
    assert_close(&resolution.within_stage_routing[1], &[0.0, 230.0 / 240.0]);
    assert_close(&resolution.within_stage_routing[2], &[0.0]);

    assert_eq!(resolution.arrival_density.len(), 1);
    assert_close(
        &resolution.arrival_density[0],
        &[240.0 / 250.0, 10.0 / 250.0],
    );

    for (d, &k_d) in resolution.stage_weights.iter().enumerate() {
        let aggregated: f64 = resolution
            .block_deposits
            .iter()
            .zip(&block_lengths_hours)
            .map(|(row, &duration_b)| (duration_b / 720.0) * row[d])
            .sum();
        assert!((aggregated - k_d).abs() < TOL);
    }
}

// Monthly anchor, t_v=360h is exactly half the anchor length, so
// stage_weights[0]=stage_weights[1]=1/2 and depth is 1.
#[test]
fn test_s1b_monthly_half_split() {
    let stage_lengths_hours = [720.0, 720.0, 720.0];
    let resolution = resolve_spread(360.0, 0, &stage_lengths_hours, None);

    assert_eq!(resolution.stage_reach, 1);
    assert_close(&resolution.stage_weights, &[0.5, 0.5]);
    assert!((resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < TOL);
}

// Weekly-to-monthly transition: the same t_v=360h anchored at each of 4
// weekly stages then the first monthly stage gives depths (3, 3, 2, 1, 1) —
// week 3's window skips week 4 entirely and lands in the month
// (stage_weights[2] = 1, slot 1 transit-only) — and the global max over all
// anchors is depth 3, the reachability mask the per-stage state sizing needs.
#[test]
fn test_s1c_weekly_to_monthly_transition_depths() {
    let stage_lengths_hours = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0, 720.0];

    let week1 = resolve_spread(360.0, 0, &stage_lengths_hours, None);
    assert_eq!(week1.stage_reach, 3);
    assert_close(&week1.stage_weights, &[0.0, 0.0, 6.0 / 7.0, 1.0 / 7.0]);

    let week2 = resolve_spread(360.0, 1, &stage_lengths_hours, None);
    assert_eq!(week2.stage_reach, 3);
    assert_close(&week2.stage_weights, &[0.0, 0.0, 6.0 / 7.0, 1.0 / 7.0]);

    let week3 = resolve_spread(360.0, 2, &stage_lengths_hours, None);
    assert_eq!(week3.stage_reach, 2);
    assert_close(&week3.stage_weights, &[0.0, 0.0, 1.0]);

    let week4 = resolve_spread(360.0, 3, &stage_lengths_hours, None);
    assert_eq!(week4.stage_reach, 1);
    assert_close(&week4.stage_weights, &[0.0, 1.0]);

    let month1 = resolve_spread(360.0, 4, &stage_lengths_hours, None);
    assert_eq!(month1.stage_reach, 1);
    assert_close(&month1.stage_weights, &[0.5, 0.5]);

    let anchors = [&week1, &week2, &week3, &week4, &month1];
    let global_max_depth = anchors
        .iter()
        .map(|resolution| resolution.stage_reach)
        .max();
    assert_eq!(global_max_depth, Some(3));

    for resolution in anchors {
        assert!((resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < TOL);
    }
}

// Negligible same-stage-crossing mass: monthly anchor, t_v=6h — the
// setup-advisory case rather than a silent fold, since the bucket still
// carries the exact mass stage_weights[1] = 6/720.
#[test]
fn test_s2_monthly_negligible_mass() {
    let stage_lengths_hours = [720.0, 720.0];
    let resolution = resolve_spread(6.0, 0, &stage_lengths_hours, None);

    assert_eq!(resolution.stage_reach, 1);
    assert_close(&resolution.stage_weights, &[714.0 / 720.0, 6.0 / 720.0]);
    assert!((resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < TOL);
}

// Daily chronological block partition: t_v=6h against 24 hourly blocks gives
// stage-clock stage_weights[1] = 25%, and the chronological-adds boundary —
// blocks 0-17 route in-stage to block b+6 (block_deposits[b][0] = 1), blocks
// 18-23 cross the day boundary and deposit fully into lag 1
// (block_deposits[b][1] = 1).
#[test]
fn test_s3_daily_chronological_block_partition() {
    let stage_lengths_hours = [24.0, 24.0, 24.0];
    let block_lengths_hours = [1.0; 24];
    let resolution = resolve_spread(6.0, 0, &stage_lengths_hours, Some(&block_lengths_hours));

    assert_eq!(resolution.stage_reach, 1);
    assert_close(&resolution.stage_weights, &[18.0 / 24.0, 6.0 / 24.0]);
    assert!((resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < TOL);

    // Last in-stage block: routes fully to block 17+6=23 within the same day.
    assert_close(&resolution.block_deposits[17], &[1.0, 0.0]);
    assert_close(
        &resolution.within_stage_routing[17],
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    );

    // First cross-boundary block: the 6-hour delay lands entirely in the
    // next day, deposited fully rather than routed in-stage.
    assert_close(&resolution.block_deposits[18], &[0.0, 1.0]);

    for (d, &k_d) in resolution.stage_weights.iter().enumerate() {
        let aggregated: f64 = resolution
            .block_deposits
            .iter()
            .zip(&block_lengths_hours)
            .map(|(row, &duration_b)| (duration_b / 24.0) * row[d])
            .sum();
        assert!((aggregated - k_d).abs() < TOL);
    }
}

// Exact-multiple boundary: monthly anchor, t_v=720h is exactly one stage
// length, so the whole release crosses the boundary: stage_weights[0] = 0,
// stage_weights[1] = 1.
#[test]
fn test_s4_monthly_exact_multiple_boundary() {
    let stage_lengths_hours = [720.0, 720.0];
    let resolution = resolve_spread(720.0, 0, &stage_lengths_hours, None);

    assert_eq!(resolution.stage_reach, 1);
    assert_close(&resolution.stage_weights, &[0.0, 1.0]);
    assert!((resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < TOL);
}

// Transit-only slot: monthly anchor, t_v=1800h (75 d) — slot 1 carries zero
// deposit yet the ring shift still passes mass through it to slots 2 and 3,
// which split the release evenly; depth counts a reachable slot, not a
// nonzero-factor count.
#[test]
fn test_s5_monthly_transit_only_slot() {
    let stage_lengths_hours = [720.0; 5];
    let resolution = resolve_spread(1800.0, 0, &stage_lengths_hours, None);

    assert_eq!(resolution.stage_reach, 3);
    assert_close(&resolution.stage_weights, &[0.0, 0.0, 0.5, 0.5]);
    assert_eq!(resolution.stage_weights[1], 0.0);
    assert!((resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < TOL);
}

// Exercises the `Σ_d k_d == 1` conservation debug_assert directly, across a
// non-uniform calendar and a non-zero `anchor_stage` slice offset — distinct
// from the weekly (`test_s1a_matches_decomp_fig_5_5b`) and monthly-transition
// (`test_s1d_monthly_to_weekly_counterexample`) fixtures above.
#[test]
fn test_stage_level_conservation_debug_assert() {
    let stage_lengths_hours = [300.0, 200.0, 400.0, 150.0, 600.0, 250.0];

    for anchor_stage in 0..3 {
        for &travel_time_hours in &[10.0, 90.0, 275.0, 501.0, 725.0] {
            let resolution =
                resolve_spread(travel_time_hours, anchor_stage, &stage_lengths_hours, None);
            assert!(
                (resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < TOL,
                "conservation violated at anchor_stage={anchor_stage}, t_v={travel_time_hours}"
            );
            assert!(resolution.stage_reach >= 1);
            assert!(resolution.block_deposits.is_empty());
            assert!(resolution.within_stage_routing.is_empty());
        }
    }
}

// Exercises the `Σ_b w_b·χ_{b,d} == k_d` shared-density-consistency
// debug_assert directly (the shared-density aggregation identity), across a
// non-uniform block partition and non-uniform future stages, distinct from
// `test_example_iii_block_factors`.
#[test]
fn test_shared_density_consistency_debug_assert() {
    let stage_lengths_hours = [720.0, 500.0, 300.0, 300.0];
    let block_lengths_hours = [100.0, 200.0, 50.0, 370.0];

    for &travel_time_hours in &[50.0, 300.0, 690.0, 1000.0] {
        let resolution = resolve_spread(
            travel_time_hours,
            0,
            &stage_lengths_hours,
            Some(&block_lengths_hours),
        );

        for (d, &k_d) in resolution.stage_weights.iter().enumerate() {
            let aggregated: f64 = resolution
                .block_deposits
                .iter()
                .zip(&block_lengths_hours)
                .map(|(row, &duration_b)| (duration_b / 720.0) * row[d])
                .sum();
            assert!(
                (aggregated - k_d).abs() < TOL,
                "shared-density consistency violated at t_v={travel_time_hours}, lag={d}"
            );
        }
    }
}

// PMO calendar (4x168h weekly then 720h monthly), 30-day (720h) lead. The
// end-anchored decider computes directly against the calendar rather than
// reading a decision-anchored column, avoiding the many-to-one collision a
// decision-anchored scheme produces on this same calendar: weeks 0-3 cannot
// reach month 1 (end_4 - 720h = day -2, pre-study), so month 1 is IC; week 4
// (the stage immediately preceding month 1) is the unique decider, and each
// month thereafter decides the next at lag 1 (Δ == h_month exactly).
#[test]
fn test_pmo_end_anchored_delivery_resolution() {
    let stage_lengths_hours = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0, 720.0];
    let resolution = resolve_point(LeadTime::Time(720.0), &stage_lengths_hours, 7);

    assert_eq!(
        resolution.decider,
        vec![None, None, None, None, Some(3), Some(4), Some(5)]
    );
    assert_eq!(
        resolution.decision_sets,
        vec![vec![], vec![], vec![], vec![4], vec![5], vec![6], vec![]]
    );
    assert_eq!(resolution.depth, vec![0, 0, 0, 1, 1, 1, 0]);
}

// Sub-stage lead: a 700h stage then a 744h (31-day) month, 720h (30-day)
// lead. Δ < h_1, so end_1 - Δ falls inside stage 1's own window and c(1) ==
// 1 — the K=0 degeneracy, represented (not underflowed) with depth[1] == 0.
#[test]
fn test_sub_stage_lead_k0_degeneracy() {
    let stage_lengths_hours = [700.0, 744.0];
    let resolution = resolve_point(LeadTime::Time(720.0), &stage_lengths_hours, 2);

    assert_eq!(resolution.decider, vec![None, Some(1)]);
    assert_eq!(resolution.decision_sets, vec![vec![], vec![1]]);
    assert_eq!(resolution.depth, vec![0, 0]);
}

// Stage-count mode on a monthly calendar with unequal stage hours (672-744h):
// the hour clock is never consulted, so `LeadTime::Stages(2)` reproduces the
// shipped index shift identically regardless of the calendar values. Depth
// and fan-out are checked away from the array boundary, where K(t) == ℓ and
// |C(t)| == 1 hold without the natural edge truncation that bounds K(t) =
// |{m>t : c(m)<=t}| to the delivery stages that actually exist.
#[test]
fn test_stage_count_mode_unequal_monthly_hours() {
    let stage_lengths_hours = [672.0, 700.0, 744.0, 720.0, 672.0, 744.0, 700.0, 744.0];
    let resolution = resolve_point(LeadTime::Stages(2), &stage_lengths_hours, 8);

    for m in 2..8 {
        assert_eq!(
            resolution.decider[m],
            Some(m - 2),
            "decider mismatch at m={m}"
        );
    }
    for t in 1..=5 {
        assert_eq!(resolution.depth[t], 2, "depth mismatch at t={t}");
        assert_eq!(
            resolution.decision_sets[t].len(),
            1,
            "decision_sets length mismatch at t={t}"
        );
    }
    assert_eq!(resolution.decision_sets[0].len(), 1);
}

#[test]
fn test_ic_boundary_decider_is_none() {
    let stage_lengths_hours = [100.0];
    let resolution = resolve_point(LeadTime::Time(1000.0), &stage_lengths_hours, 1);

    assert_eq!(resolution.decider, vec![None]);
    assert_eq!(resolution.decision_sets, vec![Vec::<usize>::new()]);
    assert_eq!(resolution.depth, vec![0]);
}

// Fan-out: a coarse decision stage before a fine zone commits several
// delivery stages, |C(t)| > 1. A 720h month (stage 0) anchors four 168h
// weeks (stages 1-4); at a 750h lead each week's end_m - Δ lands inside the
// month's own window, so all four weeks share decider 0. The month's own
// delivery (Δ exceeds its own 720h length) precedes the horizon and is IC.
#[test]
fn test_coarse_decision_fans_out_over_fine_delivery_stages() {
    let stage_lengths_hours = [720.0, 168.0, 168.0, 168.0, 168.0];
    let resolution = resolve_point(LeadTime::Time(750.0), &stage_lengths_hours, 5);

    assert_eq!(
        resolution.decider,
        vec![None, Some(0), Some(0), Some(0), Some(0)]
    );
    assert_eq!(
        resolution.decision_sets,
        vec![vec![1, 2, 3, 4], vec![], vec![], vec![], vec![]]
    );
    assert_eq!(resolution.depth, vec![4, 3, 2, 1, 0]);
    assert!(resolution.decision_sets[0].len() > 1);
}

// AnticipatedResolution::resolve batches per-plant resolutions in anticipated-
// local order and derives k_max as the global max per-stage depth. Two plants on
// a uniform monthly calendar: LeadStages(2) has depth max 2, LeadStages(4) has
// depth max 4, so the derived ring depth is 4 (the deeper plant), not the sum.
#[test]
fn test_anticipated_resolution_k_max_is_global_depth_max() {
    let stage_lengths_hours = [720.0; 8];
    let leads = [LeadTime::Stages(2), LeadTime::Stages(4)];
    let resolution = AnticipatedResolution::resolve(&leads, &stage_lengths_hours, 8);

    assert_eq!(resolution.per_plant.len(), 2);
    assert_eq!(resolution.per_plant[0].depth.iter().copied().max(), Some(2));
    assert_eq!(resolution.per_plant[1].depth.iter().copied().max(), Some(4));
    assert_eq!(resolution.k_max, 4);
}

// An empty lead set (no anticipated plants) resolves to an empty batch with ring
// depth 0 — the zero-anticipated collapse.
#[test]
fn test_anticipated_resolution_empty_is_zero_depth() {
    let resolution = AnticipatedResolution::resolve(&[], &[720.0; 4], 4);
    assert!(resolution.per_plant.is_empty());
    assert_eq!(resolution.k_max, 0);
}
