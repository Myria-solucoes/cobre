//! Bucket topology: canonical column order, global bucket count, and
//! per-stage reachability mask for water travel-time in-transit buckets.
//!
//! Depths are resolved on the stage clock via [`resolve_spread`] (in-study
//! anchors) and [`window_period_overlaps`] (the pre-study IC anchor) —
//! `n_blks`/block mode never enter dimensioning. Every arc feeding one
//! downstream plant collapses into a single aggregated block (the arrival
//! schedule at a plant is a sufficient statistic over its upstreams), ordered
//! by the same canonical `(operational_start_date, id)` index every other
//! state block uses.

use std::{collections::HashMap, ops::Range};

use cobre_core::{BlockMode, EntityId, System, window_period_overlaps};

use crate::temporal_lag::{SpreadResolution, resolve_spread};

/// Canonical bucket ordering, global bucket count, and per-stage reachability
/// mask, stored on [`super::StudySetup`].
///
/// `b_total == 0` exactly when the system declares no travel-time arc
/// (`travel_time_hours` absent, `0.0`, or missing a `downstream_id`).
// Voice 4: no production read site consumes these fields yet — the state
// layout will read `b_total`/`column_order` to size and order the bucket
// block, and the per-stage LP fill will read `per_stage_mask` to gate which
// bucket rows it emits. The `#[allow(dead_code)]` refires once those readers
// land.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct BucketTopology {
    /// Global bucket count, `Σ_j per_plant_depth[j]`.
    pub(crate) b_total: usize,
    /// Aggregated depth `L_j` per downstream plant, in [`Self::column_order`]'s
    /// plant order.
    pub(crate) per_plant_depth: Vec<usize>,
    /// `(plant_canonical_idx, lag)` pairs, `lag = 1..=L_j`, plants sorted by
    /// canonical `(operational_start_date, id)` index (the position of the
    /// hydro in [`System::hydros`]).
    pub(crate) column_order: Vec<(usize, usize)>,
    /// `per_stage_mask[t]` holds one contiguous reachable lag range per
    /// declared downstream plant, in the same order as
    /// [`Self::per_plant_depth`], at study stage `t`.
    pub(crate) per_stage_mask: Vec<Vec<Range<usize>>>,
}

/// Study-stage (`id >= 0`) durations in canonical (ascending `id`) stage-index
/// order, each summed from its blocks.
pub(crate) fn study_stage_durations(system: &System) -> Vec<f64> {
    system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.blocks.iter().map(|b| b.duration_hours).sum())
        .collect()
}

/// Declared arcs' travel times grouped by downstream plant id. A hydro
/// declares an arc when `travel_time_hours` is `Some` and `> 0.0` (`0.0` is
/// undeclared) and `downstream_id` is `Some`.
fn declared_arcs(system: &System) -> HashMap<EntityId, Vec<f64>> {
    let mut arcs: HashMap<EntityId, Vec<f64>> = HashMap::new();
    for hydro in system.hydros() {
        let Some(t_v) = hydro.travel_time_hours.filter(|&t| t > 0.0) else {
            continue;
        };
        let Some(downstream_id) = hydro.downstream_id else {
            continue;
        };
        arcs.entry(downstream_id).or_default().push(t_v);
    }
    arcs
}

/// Extend the study calendar with copies of its trailing stage duration so
/// [`resolve_spread`] never sees a calendar too short to absorb the window —
/// its conservation check panics otherwise. The extension makes the depth
/// well-defined "as if the horizon continued"; capping it back to the true
/// remaining horizon is a separate, later step this function does not take.
fn extend_for_resolution(study_durations: &[f64], t_v: f64) -> Vec<f64> {
    let Some(&last) = study_durations.last() else {
        return study_durations.to_vec();
    };
    debug_assert!(last > 0.0, "every study stage duration must be > 0.0");

    let mut extended = study_durations.to_vec();
    let mut padded_hours = 0.0_f64;
    while padded_hours < t_v {
        extended.push(last);
        padded_hours += last;
    }
    extended
}

/// In-study depth at one stage anchor: [`resolve_spread`]'s overlap discards
/// its index-0 share (delivered same-stage on the water row, no bucket
/// needed).
fn in_study_depth(t_v: f64, stage: usize, extended_calendar: &[f64]) -> usize {
    resolve_spread(t_v, stage, extended_calendar, None).depth
}

/// Pre-study residual depth: the in-transit water arriving over the
/// study-clock window `[0, t_v)` has no same-stage share to discard, so this
/// is the raw overlap count — the one place this feature's depth arithmetic
/// diverges from [`in_study_depth`].
fn ic_only_depth(t_v: f64, study_durations: &[f64]) -> usize {
    window_period_overlaps(0.0, t_v, study_durations).len()
}

/// Caps a stage's active lag at `n_stages − stage − 1`, the deepest lag whose
/// target stage `stage + lag` still lands inside `[0, n_stages)` — the same
/// horizon bound `is_anticipated_decision_active` enforces as
/// `stage_idx + K_i < n_stages`. A lag beyond the cap has no receiving stage;
/// [`build_bucket_topology`] drops it from the mask here rather than
/// retaining and zeroing it downstream, the target-stage imprecision the
/// deferred terminal bucket credit (`V_eff`) would otherwise absorb under the
/// zero-terminal-value horizon [`crate::horizon_mode::HorizonMode::Finite`]
/// implements. Never caps [`BucketTopology::per_plant_depth`] or
/// [`BucketTopology::column_order`], which size from the global max over
/// every stage anchor and must retain what the earliest stages need.
fn horizon_cap_active(active: usize, stage: usize, n_stages: usize) -> usize {
    active.min(n_stages - 1 - stage)
}

/// Pre-study (`id < 0`) period durations in hours, most-recent-first (index 0
/// = the period immediately preceding stage 0) — the `past_inflows` /
/// `past_defluences` index convention, the reverse of the canonical
/// ascending-`id` order. Pre-study stages carry no blocks; duration comes
/// from the calendar dates.
// Rationale: pre-study period lengths are on the order of years, far under
// f64's exact-integer range; a checked conversion buys nothing.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn pre_study_period_durations_desc(system: &System) -> Vec<f64> {
    let mut pre_study: Vec<f64> = system
        .stages()
        .iter()
        .filter(|s| s.id < 0)
        .map(|s| (s.end_date - s.start_date).num_hours() as f64)
        .collect();
    pre_study.reverse();
    pre_study
}

/// The number of most-recent pre-study periods a declared arc's history must
/// supply so `[start_0 - t_v, start_0)` is fully covered — floored at 1 so an
/// empty pre-study calendar cannot vacuously report zero. Mirrors `cobre-io`'s
/// `validate_travel_time` row-5 gate; the bucket seed's source-selection
/// re-derives the same sufficiency check that validation already enforced.
pub(crate) fn required_history_periods(t_v: f64, pre_study_desc: &[f64]) -> usize {
    window_period_overlaps(0.0, t_v, pre_study_desc)
        .len()
        .max(1)
}

/// Fraction of pre-study period `m`'s release — spanning
/// `[t_v - cumulative_before - period_duration, t_v - cumulative_before)` in
/// real time before stage 0 — landing in study stage `d` (0-indexed, at
/// `result[d]`). The IC-anchor analogue of [`resolve_spread`]'s `k`: resolved
/// directly against the forward calendar rather than a concatenated local one
/// anchored at period `m` — equivalent, since [`window_period_overlaps`]
/// depends only on relative offsets, never the absolute origin.
pub(crate) fn ic_anchor_k(
    t_v: f64,
    cumulative_before: f64,
    period_duration: f64,
    study_durations: &[f64],
) -> Vec<f64> {
    let window_start = t_v - cumulative_before - period_duration;
    window_period_overlaps(window_start, period_duration, study_durations)
        .into_iter()
        .map(|overlap| overlap / period_duration)
        .collect()
}

/// Build the [`BucketTopology`] from the resolved system: group declared arcs
/// per downstream plant (confluence aggregates every contributing arc into
/// one block of depth `max_i L_i`, never one block per arc), size each
/// plant's depth as the max over every in-study stage anchor and the
/// pre-study IC anchor, and emit the canonical column order and per-stage
/// reachability mask in `(operational_start_date, id)` order.
pub(crate) fn build_bucket_topology(system: &System) -> BucketTopology {
    let study_durations = study_stage_durations(system);
    let n_stages = study_durations.len();
    let arcs_by_downstream = declared_arcs(system);

    let mut per_plant_depth = Vec::new();
    let mut column_order = Vec::new();
    let mut per_stage_mask: Vec<Vec<Range<usize>>> = vec![Vec::new(); n_stages];

    for (canonical_idx, hydro) in system.hydros().iter().enumerate() {
        let Some(t_vs) = arcs_by_downstream.get(&hydro.id) else {
            continue;
        };

        let mut own_release_by_stage = vec![0_usize; n_stages];
        let mut ic_depth = 0_usize;
        for &t_v in t_vs {
            let extended = extend_for_resolution(&study_durations, t_v);
            for (stage, slot) in own_release_by_stage.iter_mut().enumerate() {
                *slot = (*slot).max(in_study_depth(t_v, stage, &extended));
            }
            ic_depth = ic_depth.max(ic_only_depth(t_v, &study_durations));
        }

        let in_study_max = own_release_by_stage.iter().copied().max().unwrap_or(0);
        let depth = in_study_max.max(ic_depth);
        if depth == 0 {
            continue;
        }

        per_plant_depth.push(depth);
        for lag in 1..=depth {
            column_order.push((canonical_idx, lag));
        }
        for (stage, mask_row) in per_stage_mask.iter_mut().enumerate() {
            // Reachability, not a zero-deposit filter: a transit slot with no
            // net deposit at this stage still carries mass through the ring
            // shift and must stay in the active range.
            let active = own_release_by_stage[stage].max(ic_depth.saturating_sub(stage));
            let capped = horizon_cap_active(active, stage, n_stages);
            debug_assert!(
                stage + capped < n_stages,
                "capped active lag {capped} at stage {stage} must not target n_stages={n_stages} or beyond"
            );
            mask_row.push(1..(capped + 1));
        }
    }

    let b_total = column_order.len();
    debug_assert!(
        arcs_by_downstream.is_empty() == (b_total == 0),
        "b_total must be zero exactly when no arc is declared"
    );

    BucketTopology {
        b_total,
        per_plant_depth,
        column_order,
        per_stage_mask,
    }
}

/// Per-declared-arc resolved stage-clock weights for the PARALLEL-mode LP fill,
/// keyed by the arc's upstream hydro system index (a hydro declares at most one
/// arc — its own `travel_time_hours`/`downstream_id`). `k_by_stage[stage_idx]`
/// is [`resolve_spread`]'s stage-clock weight vector `k` anchored at that
/// in-study stage (`k[0]` the same-stage share); a hydro absent from the map
/// declares no arc — the LP fill's undeclared branch (full same-stage arrival,
/// no bucket deposit). The chronological-mode block-resolved `chi`/`kappa`
/// factors are threaded separately.
pub(crate) fn build_arc_spread_k(system: &System) -> HashMap<usize, Vec<Vec<f64>>> {
    let study_durations = study_stage_durations(system);
    let n_stages = study_durations.len();
    let mut arc_spread_k = HashMap::new();

    for (u_idx, hydro) in system.hydros().iter().enumerate() {
        let Some(t_v) = hydro.travel_time_hours.filter(|&t| t > 0.0) else {
            continue;
        };
        if hydro.downstream_id.is_none() {
            continue;
        }
        let extended = extend_for_resolution(&study_durations, t_v);
        let k_by_stage: Vec<Vec<f64>> = (0..n_stages)
            .map(|stage| resolve_spread(t_v, stage, &extended, None).k)
            .collect();
        arc_spread_k.insert(u_idx, k_by_stage);
    }

    arc_spread_k
}

/// Per-declared-arc, per-CHRONOLOGICAL-stage full [`SpreadResolution`] (`chi`,
/// `kappa`, `delivery`, plus the same `k`/`depth` [`build_arc_spread_k`]
/// stores), resolved with the sending stage's own block partition
/// (`resolve_spread(.., Some(blocks))`). Keyed like `build_arc_spread_k`;
/// `by_stage[stage_idx]` is `None` for a study stage whose own `block_mode` is
/// `Parallel` (no block-resolved routing to compute there — the parallel fill
/// reads `build_arc_spread_k` instead).
pub(crate) fn build_arc_spread_chrono(
    system: &System,
) -> HashMap<usize, Vec<Option<SpreadResolution>>> {
    let study_durations = study_stage_durations(system);
    let n_stages = study_durations.len();
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    debug_assert_eq!(study_stages.len(), n_stages);

    let mut arc_spread_chrono = HashMap::new();

    for (u_idx, hydro) in system.hydros().iter().enumerate() {
        let Some(t_v) = hydro.travel_time_hours.filter(|&t| t > 0.0) else {
            continue;
        };
        if hydro.downstream_id.is_none() {
            continue;
        }
        let extended = extend_for_resolution(&study_durations, t_v);
        let by_stage: Vec<Option<SpreadResolution>> = (0..n_stages)
            .map(|stage_idx| {
                if study_stages[stage_idx].block_mode != BlockMode::Chronological {
                    return None;
                }
                let blocks: Vec<f64> = study_stages[stage_idx]
                    .blocks
                    .iter()
                    .map(|b| b.duration_hours)
                    .collect();
                Some(resolve_spread(t_v, stage_idx, &extended, Some(&blocks)))
            })
            .collect();
        arc_spread_chrono.insert(u_idx, by_stage);
    }

    arc_spread_chrono
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use cobre_core::{
        Block, BlockMode, Bus, DeficitSegment, Hydro, HydroGenerationModel, HydroPenalties,
        NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig, SystemBuilder,
    };

    fn zero_penalties() -> HydroPenalties {
        HydroPenalties {
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
            inflow_nonnegativity_cost: 0.0,
        }
    }

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    fn hydro(id: i32, downstream_id: Option<i32>, travel_time_hours: Option<f64>) -> Hydro {
        Hydro {
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: date(2024, 1, 1),
            bus_id: EntityId(1),
            downstream_id: downstream_id.map(EntityId),
            travel_time_hours,
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
            penalties: zero_penalties(),
        }
    }

    fn stage_with_durations(id: i32, block_hours: &[f64]) -> Stage {
        Stage {
            index: usize::try_from(id).unwrap_or(0),
            id,
            start_date: date(2024, 1, 1),
            end_date: date(2024, 2, 1),
            season_id: None,
            blocks: block_hours
                .iter()
                .enumerate()
                .map(|(i, &h)| Block {
                    index: i,
                    name: format!("B{i}"),
                    duration_hours: h,
                })
                .collect(),
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
        }
    }

    fn chronological_stage_with_durations(id: i32, block_hours: &[f64]) -> Stage {
        Stage {
            block_mode: BlockMode::Chronological,
            ..stage_with_durations(id, block_hours)
        }
    }

    fn stages_with_durations(durations: &[f64]) -> Vec<Stage> {
        durations
            .iter()
            .enumerate()
            .map(|(i, &h)| stage_with_durations(i32::try_from(i).unwrap_or(0), &[h]))
            .collect()
    }

    fn uniform_stages(n: usize, hours: f64) -> Vec<Stage> {
        stages_with_durations(&vec![hours; n])
    }

    fn build_system(hydros: Vec<Hydro>, stages: Vec<Stage>) -> cobre_core::System {
        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            operational_start_date: date(2024, 1, 1),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(hydros)
            .stages(stages)
            .build()
            .expect("valid system")
    }

    #[test]
    fn test_b_total_zero_when_no_arc_declared() {
        let downstream = hydro(1, None, None);
        let system = build_system(vec![downstream], uniform_stages(3, 24.0));

        let topology = build_bucket_topology(&system);

        assert_eq!(topology.b_total, 0);
        assert!(topology.column_order.is_empty());
        assert!(topology.per_plant_depth.is_empty());
    }

    #[test]
    fn test_zero_travel_time_is_treated_as_undeclared() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(0.0));
        let system = build_system(vec![downstream, upstream], uniform_stages(3, 24.0));

        let topology = build_bucket_topology(&system);

        assert_eq!(topology.b_total, 0);
    }

    #[test]
    fn test_confluence_aggregates_to_single_block_of_max_depth() {
        let downstream = hydro(1, None, None);
        let upstream_a = hydro(2, Some(1), Some(24.0));
        let upstream_b = hydro(3, Some(1), Some(100.0));
        let system = build_system(
            vec![downstream, upstream_a, upstream_b],
            uniform_stages(10, 24.0),
        );

        let topology = build_bucket_topology(&system);

        assert_eq!(topology.per_plant_depth, vec![5]);
        assert_eq!(topology.b_total, 5);
        assert_eq!(
            topology.column_order,
            vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
        );
    }

    #[test]
    fn test_fine_first_coarse_next_ic_anchor_deepens_beyond_in_study_max() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(30.0));
        let durations = [24.0, 720.0, 720.0, 720.0];
        let system = build_system(
            vec![downstream, upstream],
            stages_with_durations(&durations),
        );

        let extended = extend_for_resolution(&durations, 30.0);
        let in_study_max = (0..durations.len())
            .map(|t| in_study_depth(30.0, t, &extended))
            .max()
            .unwrap_or(0);
        let ic_depth = ic_only_depth(30.0, &durations);
        assert_eq!(
            in_study_max, 1,
            "every in-study anchor must give L_arc == 1"
        );
        assert_eq!(ic_depth, 2, "the IC anchor must give L_arc(IC) == 2");

        let topology = build_bucket_topology(&system);

        assert_eq!(topology.per_plant_depth, vec![2]);
        assert_eq!(topology.b_total, 2);

        // The stage-0 mask reaches the IC-residual slot 2 (decaying reachability,
        // not a zero-deposit filter); it narrows to the own-release depth once the
        // residual has drained.
        assert_eq!(topology.per_stage_mask.len(), durations.len());
        assert_eq!(topology.per_stage_mask[0], vec![1..3]);
        assert_eq!(topology.per_stage_mask[1], vec![1..2]);
        assert_eq!(topology.per_stage_mask[2], vec![1..2]);
    }

    #[test]
    fn test_uniform_calendar_ic_anchor_does_not_deepen() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(24.0));
        let durations = vec![24.0; 6];
        let system = build_system(
            vec![downstream, upstream],
            stages_with_durations(&durations),
        );

        let extended = extend_for_resolution(&durations, 24.0);
        let in_study_max = (0..durations.len())
            .map(|t| in_study_depth(24.0, t, &extended))
            .max()
            .unwrap_or(0);
        let ic_depth = ic_only_depth(24.0, &durations);
        assert_eq!(ic_depth, in_study_max, "uniform calendar: no IC deepening");

        let topology = build_bucket_topology(&system);

        assert_eq!(topology.per_plant_depth, vec![in_study_max]);
    }

    #[test]
    fn test_horizon_cap_drops_lag_targeting_past_last_stage() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(72.0));
        let durations = [24.0, 24.0, 24.0];
        let system = build_system(
            vec![downstream, upstream],
            stages_with_durations(&durations),
        );

        let extended = extend_for_resolution(&durations, 72.0);
        let uncapped_active_by_stage: Vec<usize> = (0..durations.len())
            .map(|t| in_study_depth(72.0, t, &extended))
            .collect();
        let ic_depth = ic_only_depth(72.0, &durations);
        assert_eq!(
            uncapped_active_by_stage,
            vec![3, 3, 3],
            "every anchor's own-release depth must reach 3 stages ahead, past the 3-stage horizon"
        );
        assert_eq!(ic_depth, 3, "the IC anchor must also reach 3 stages ahead");

        let topology = build_bucket_topology(&system);

        assert_eq!(
            topology.per_plant_depth,
            vec![3],
            "global depth sizing is unaffected by the per-stage horizon cap"
        );
        assert_eq!(topology.b_total, 3);
        assert_eq!(topology.column_order, vec![(0, 1), (0, 2), (0, 3)]);

        assert_eq!(
            topology.per_stage_mask[0],
            vec![1..3],
            "cap = 3 - 1 - 0 = 2"
        );
        assert_eq!(
            topology.per_stage_mask[1],
            vec![1..2],
            "cap = 3 - 1 - 1 = 1"
        );
        assert_eq!(
            topology.per_stage_mask[2],
            vec![1..1],
            "cap = 3 - 1 - 2 = 0: the last stage targets nothing past T"
        );

        for (stage, mask_row) in topology.per_stage_mask.iter().enumerate() {
            for range in mask_row {
                let max_lag = range.end.saturating_sub(1);
                assert!(
                    stage + max_lag < durations.len(),
                    "stage {stage} lag {max_lag} must not target a stage at or past n_stages"
                );
            }
        }
    }

    #[test]
    fn test_column_order_is_declaration_order_invariant() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(48.0));

        let system_a = build_system(
            vec![downstream.clone(), upstream.clone()],
            uniform_stages(5, 24.0),
        );
        let system_b = build_system(vec![upstream, downstream], uniform_stages(5, 24.0));

        let topology_a = build_bucket_topology(&system_a);
        let topology_b = build_bucket_topology(&system_b);

        assert_eq!(topology_a.column_order, topology_b.column_order);
        assert_eq!(topology_a.per_plant_depth, topology_b.per_plant_depth);
        assert_eq!(topology_a.b_total, topology_b.b_total);
    }

    #[test]
    fn test_build_arc_spread_k_empty_when_no_arc_declared() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), None);
        let system = build_system(vec![downstream, upstream], uniform_stages(3, 24.0));

        let arc_spread_k = build_arc_spread_k(&system);

        assert!(arc_spread_k.is_empty());
    }

    #[test]
    fn test_build_arc_spread_k_conserves_and_matches_topology_depth() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(24.0));
        let system = build_system(vec![downstream, upstream], uniform_stages(10, 24.0));

        let topology = build_bucket_topology(&system);
        let arc_spread_k = build_arc_spread_k(&system);

        let upstream_idx = 1;
        let k_by_stage = arc_spread_k
            .get(&upstream_idx)
            .expect("declared arc must have an entry");
        assert_eq!(k_by_stage.len(), 10, "one k vector per in-study stage");
        for k in k_by_stage {
            let sum: f64 = k.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "k must conserve to 1.0, got {k:?}"
            );
        }
        let max_depth = k_by_stage.iter().map(|k| k.len() - 1).max().unwrap_or(0);
        assert_eq!(
            max_depth, topology.per_plant_depth[0],
            "the deepest in-study k vector must match the topology's per-plant depth"
        );
    }

    #[test]
    fn test_build_arc_spread_chrono_gates_on_stage_block_mode() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(250.0));
        let system = build_system(
            vec![downstream, upstream],
            vec![
                chronological_stage_with_durations(0, &[240.0, 240.0, 240.0]),
                stage_with_durations(1, &[720.0]),
            ],
        );

        let arc_spread_chrono = build_arc_spread_chrono(&system);
        let upstream_idx = 1;
        let by_stage = arc_spread_chrono
            .get(&upstream_idx)
            .expect("declared arc must have an entry");

        assert!(
            by_stage[0].is_some(),
            "chronological stage 0 must resolve chi/kappa/delivery"
        );
        assert!(
            by_stage[1].is_none(),
            "parallel stage 1 has no block-resolved routing to compute"
        );

        let resolution = by_stage[0].as_ref().expect("checked above");
        assert_eq!(resolution.kappa.len(), 3, "one kappa row per block");
        assert_eq!(resolution.chi.len(), 3, "one chi row per block");
        for (b, chi_b) in resolution.chi.iter().enumerate() {
            let kappa_sum: f64 = resolution.kappa[b].iter().sum();
            let chi_cross: f64 = chi_b[1..].iter().sum();
            assert!(
                (kappa_sum + chi_cross - 1.0).abs() < 1e-9,
                "block {b}: per-column conservation must hold"
            );
        }
    }
}
