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

// S1a stage-clock weights per temporal-lag-unification.md §3 (DECOMP Fig. 5.5b):
// k_0 = k_1 = 0, k_2 = 6/7, k_3 = 1/7, depth 3.
#[test]
fn test_s1a_matches_decomp_fig_5_5b() {
    let stage_lengths_hours = [168.0; 6];
    let resolution = resolve_spread(360.0, 0, &stage_lengths_hours, None);

    assert_eq!(resolution.depth, 3);
    assert_close(&resolution.k, &[0.0, 0.0, 6.0 / 7.0, 1.0 / 7.0]);
    assert!((resolution.k.iter().sum::<f64>() - 1.0).abs() < TOL);
}

#[test]
fn test_s1d_monthly_to_weekly_counterexample() {
    let stage_lengths_hours = [720.0, 168.0, 168.0, 168.0, 168.0];
    let resolution = resolve_spread(360.0, 0, &stage_lengths_hours, None);

    assert_eq!(resolution.depth, 3);
    assert_close(
        &resolution.k,
        &[1.0 / 2.0, 7.0 / 30.0, 7.0 / 30.0, 1.0 / 30.0],
    );
    assert!((resolution.k.iter().sum::<f64>() - 1.0).abs() < TOL);

    // The closed-form ceiling `⌈360/720⌉ == 1` would drop the lag-2/lag-3
    // mass; only the general max-reach depth is correct here.
    assert_ne!(resolution.depth, 1);
}

#[test]
fn test_example_iii_block_factors() {
    let stage_lengths_hours = [720.0, 720.0];
    let block_lengths_hours = [240.0, 240.0, 240.0];
    let resolution = resolve_spread(250.0, 0, &stage_lengths_hours, Some(&block_lengths_hours));

    assert_eq!(resolution.depth, 1);
    assert_close(&resolution.k, &[470.0 / 720.0, 250.0 / 720.0]);

    assert_close(&resolution.chi[0], &[1.0, 0.0]);
    assert_close(&resolution.chi[1], &[230.0 / 240.0, 10.0 / 240.0]);
    assert_close(&resolution.chi[2], &[0.0, 1.0]);

    // kappa[b] is self-inclusive: index 0 == block b routing to itself, so the
    // two nonzero downstream entries are kappa[0][1..].
    assert_close(&resolution.kappa[0], &[0.0, 230.0 / 240.0, 10.0 / 240.0]);
    assert_close(&resolution.kappa[1], &[0.0, 230.0 / 240.0]);
    assert_close(&resolution.kappa[2], &[0.0]);

    assert_eq!(resolution.delivery.len(), 1);
    assert_close(&resolution.delivery[0], &[240.0 / 250.0, 10.0 / 250.0]);

    for (d, &k_d) in resolution.k.iter().enumerate() {
        let aggregated: f64 = resolution
            .chi
            .iter()
            .zip(&block_lengths_hours)
            .map(|(row, &duration_b)| (duration_b / 720.0) * row[d])
            .sum();
        assert!((aggregated - k_d).abs() < TOL);
    }
}

// S1b stage-clock weights per temporal-lag-unification.md §3 (DECOMP Fig.
// 5.5c's (Δt-15)/Δt same-stage shape): monthly anchor, t_v=360h is exactly
// half the anchor length, so k_0 = k_1 = 1/2 and depth is 1.
#[test]
fn test_s1b_monthly_half_split() {
    let stage_lengths_hours = [720.0, 720.0, 720.0];
    let resolution = resolve_spread(360.0, 0, &stage_lengths_hours, None);

    assert_eq!(resolution.depth, 1);
    assert_close(&resolution.k, &[0.5, 0.5]);
    assert!((resolution.k.iter().sum::<f64>() - 1.0).abs() < TOL);
}

// S1c weekly-to-monthly transition per temporal-lag-unification.md §3: the
// same t_v=360h anchored at each of 4 weekly stages then the first monthly
// stage gives depths (3, 3, 2, 1, 1) — week 3's window skips week 4 entirely
// and lands in the month (k_2 = 1, slot 1 transit-only) — and the global max
// over all anchors is depth 3, the reachability mask the per-stage state
// sizing needs.
#[test]
fn test_s1c_weekly_to_monthly_transition_depths() {
    let stage_lengths_hours = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0, 720.0];

    let week1 = resolve_spread(360.0, 0, &stage_lengths_hours, None);
    assert_eq!(week1.depth, 3);
    assert_close(&week1.k, &[0.0, 0.0, 6.0 / 7.0, 1.0 / 7.0]);

    let week2 = resolve_spread(360.0, 1, &stage_lengths_hours, None);
    assert_eq!(week2.depth, 3);
    assert_close(&week2.k, &[0.0, 0.0, 6.0 / 7.0, 1.0 / 7.0]);

    let week3 = resolve_spread(360.0, 2, &stage_lengths_hours, None);
    assert_eq!(week3.depth, 2);
    assert_close(&week3.k, &[0.0, 0.0, 1.0]);

    let week4 = resolve_spread(360.0, 3, &stage_lengths_hours, None);
    assert_eq!(week4.depth, 1);
    assert_close(&week4.k, &[0.0, 1.0]);

    let month1 = resolve_spread(360.0, 4, &stage_lengths_hours, None);
    assert_eq!(month1.depth, 1);
    assert_close(&month1.k, &[0.5, 0.5]);

    let anchors = [&week1, &week2, &week3, &week4, &month1];
    let global_max_depth = anchors.iter().map(|resolution| resolution.depth).max();
    assert_eq!(global_max_depth, Some(3));

    for resolution in anchors {
        assert!((resolution.k.iter().sum::<f64>() - 1.0).abs() < TOL);
    }
}

// S2 negligible same-stage-crossing mass per temporal-lag-unification.md §3:
// monthly anchor, t_v=6h — the setup-advisory case rather than a silent
// fold, since the bucket still carries the exact mass k_1 = 6/720.
#[test]
fn test_s2_monthly_negligible_mass() {
    let stage_lengths_hours = [720.0, 720.0];
    let resolution = resolve_spread(6.0, 0, &stage_lengths_hours, None);

    assert_eq!(resolution.depth, 1);
    assert_close(&resolution.k, &[714.0 / 720.0, 6.0 / 720.0]);
    assert!((resolution.k.iter().sum::<f64>() - 1.0).abs() < TOL);
}

// S3 daily chronological block partition per temporal-lag-unification.md §3:
// t_v=6h against 24 hourly blocks gives stage-clock k_1 = 25%, and the
// chronological-adds boundary — blocks 0-17 route in-stage to block b+6
// (chi[b][0] = 1), blocks 18-23 cross the day boundary and deposit fully into
// lag 1 (chi[b][1] = 1).
#[test]
fn test_s3_daily_chronological_block_partition() {
    let stage_lengths_hours = [24.0, 24.0, 24.0];
    let block_lengths_hours = [1.0; 24];
    let resolution = resolve_spread(6.0, 0, &stage_lengths_hours, Some(&block_lengths_hours));

    assert_eq!(resolution.depth, 1);
    assert_close(&resolution.k, &[18.0 / 24.0, 6.0 / 24.0]);
    assert!((resolution.k.iter().sum::<f64>() - 1.0).abs() < TOL);

    // Last in-stage block: routes fully to block 17+6=23 within the same day.
    assert_close(&resolution.chi[17], &[1.0, 0.0]);
    assert_close(&resolution.kappa[17], &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);

    // First cross-boundary block: the 6-hour delay lands entirely in the
    // next day, deposited fully rather than routed in-stage.
    assert_close(&resolution.chi[18], &[0.0, 1.0]);

    for (d, &k_d) in resolution.k.iter().enumerate() {
        let aggregated: f64 = resolution
            .chi
            .iter()
            .zip(&block_lengths_hours)
            .map(|(row, &duration_b)| (duration_b / 24.0) * row[d])
            .sum();
        assert!((aggregated - k_d).abs() < TOL);
    }
}

// S4 exact-multiple boundary per temporal-lag-unification.md §3: monthly
// anchor, t_v=720h is exactly one stage length, so the whole release crosses
// the boundary: k_0 = 0, k_1 = 1.
#[test]
fn test_s4_monthly_exact_multiple_boundary() {
    let stage_lengths_hours = [720.0, 720.0];
    let resolution = resolve_spread(720.0, 0, &stage_lengths_hours, None);

    assert_eq!(resolution.depth, 1);
    assert_close(&resolution.k, &[0.0, 1.0]);
    assert!((resolution.k.iter().sum::<f64>() - 1.0).abs() < TOL);
}

// S5 transit-only slot per temporal-lag-unification.md §3: monthly anchor,
// t_v=1800h (75 d) — slot 1 carries zero deposit yet the ring shift still
// passes mass through it to slots 2 and 3, which split the release evenly;
// depth counts a reachable slot, not a nonzero-factor count.
#[test]
fn test_s5_monthly_transit_only_slot() {
    let stage_lengths_hours = [720.0; 5];
    let resolution = resolve_spread(1800.0, 0, &stage_lengths_hours, None);

    assert_eq!(resolution.depth, 3);
    assert_close(&resolution.k, &[0.0, 0.0, 0.5, 0.5]);
    assert_eq!(resolution.k[1], 0.0);
    assert!((resolution.k.iter().sum::<f64>() - 1.0).abs() < TOL);
}

// Exercises the `Σ_d k_d == 1` conservation debug_assert directly
// (water-travel-time-sddp-analysis.md §6.1), across a non-uniform calendar and
// a non-zero `anchor_stage` slice offset, distinct from S1a/S1d's fixtures.
#[test]
fn test_stage_level_conservation_debug_assert() {
    let stage_lengths_hours = [300.0, 200.0, 400.0, 150.0, 600.0, 250.0];

    for anchor_stage in 0..3 {
        for &travel_time_hours in &[10.0, 90.0, 275.0, 501.0, 725.0] {
            let resolution =
                resolve_spread(travel_time_hours, anchor_stage, &stage_lengths_hours, None);
            assert!(
                (resolution.k.iter().sum::<f64>() - 1.0).abs() < TOL,
                "conservation violated at anchor_stage={anchor_stage}, t_v={travel_time_hours}"
            );
            assert!(resolution.depth >= 1);
            assert!(resolution.chi.is_empty());
            assert!(resolution.kappa.is_empty());
        }
    }
}

// Exercises the `Σ_b w_b·χ_{b,d} == k_d` shared-density-consistency
// debug_assert directly (sub-contract 2), across a non-uniform block
// partition and non-uniform future stages, distinct from example (iii).
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

        for (d, &k_d) in resolution.k.iter().enumerate() {
            let aggregated: f64 = resolution
                .chi
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
// end-anchored decider (§4.3) inverts the memo §4.1 collision illustration
// directly against the calendar rather than reading its decision-anchored
// column: weeks 0-3 cannot reach month 1 (end_4 - 720h = day -2, pre-study),
// so month 1 is IC; week 4 (the stage immediately preceding month 1) is the
// unique decider, and each month thereafter decides the next at lag 1
// (Δ == h_month exactly).
#[test]
fn test_pmo_end_anchored_delivery_resolution() {
    let stage_lengths_hours = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0, 720.0];
    let resolution = resolve_point(Lag::Time(720.0), &stage_lengths_hours, 7);

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

// Sub-stage lead (§4.3): a 700h stage then a 744h (31-day) month, 720h
// (30-day) lead. Δ < h_1, so end_1 - Δ falls inside stage 1's own window and
// c(1) == 1 — the K=0 degeneracy, represented (not underflowed) with
// depth[1] == 0.
#[test]
fn test_sub_stage_lead_k0_degeneracy() {
    let stage_lengths_hours = [700.0, 744.0];
    let resolution = resolve_point(Lag::Time(720.0), &stage_lengths_hours, 2);

    assert_eq!(resolution.decider, vec![None, Some(1)]);
    assert_eq!(resolution.decision_sets, vec![vec![], vec![1]]);
    assert_eq!(resolution.depth, vec![0, 0]);
}

// Stage-count mode (§4.4) on a monthly calendar with unequal stage hours
// (672-744h): the hour clock is never consulted, so `Lag::Stages(2)`
// reproduces the shipped index shift identically regardless of the calendar
// values. Depth and fan-out are checked away from the array boundary, where
// K(t) == ℓ and |C(t)| == 1 hold without the natural edge truncation that
// bounds K(t) = |{m>t : c(m)<=t}| to the delivery stages that actually exist.
#[test]
fn test_stage_count_mode_unequal_monthly_hours() {
    let stage_lengths_hours = [672.0, 700.0, 744.0, 720.0, 672.0, 744.0, 700.0, 744.0];
    let resolution = resolve_point(Lag::Stages(2), &stage_lengths_hours, 8);

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
    let resolution = resolve_point(Lag::Time(1000.0), &stage_lengths_hours, 1);

    assert_eq!(resolution.decider, vec![None]);
    assert_eq!(resolution.decision_sets, vec![Vec::<usize>::new()]);
    assert_eq!(resolution.depth, vec![0]);
}

// Fan-out per temporal-lag-unification.md §4.3: a coarse decision stage
// before a fine zone commits several delivery stages, |C(t)| > 1. A 720h
// month (stage 0) anchors four 168h weeks (stages 1-4); at a 750h lead each
// week's end_m - Δ lands inside the month's own window, so all four weeks
// share decider 0. The month's own delivery (Δ exceeds its own 720h length)
// precedes the horizon and is IC.
#[test]
fn test_coarse_decision_fans_out_over_fine_delivery_stages() {
    let stage_lengths_hours = [720.0, 168.0, 168.0, 168.0, 168.0];
    let resolution = resolve_point(Lag::Time(750.0), &stage_lengths_hours, 5);

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
