//! Stage-0 incoming travel-time bucket seed from windowed `past_defluences`
//! releases.
//!
//! The caller splices the result into the same `initial_state` vector
//! `build_initial_state` populates, so the single `fill_col_state_patches` pin
//! path picks it up with no separate wiring.

use chrono::NaiveDate;
use cobre_core::System;

use crate::lp_builder::M3S_TO_HM3;

use super::bucket_topology::{TransitBucketTopology, ic_anchor_k, study_stage_durations};

/// Unroll every declared arc's `past_defluences` windows into the stage-0
/// incoming bucket seed, in [`TransitBucketTopology::column_order`] order.
///
/// Each window `[start_date, end_date)` for upstream hydro `i` contributes
/// `k_d · D_i`: `D_i` the window's period-duration-scaled volume, `k_d` the
/// fraction landing `d` study stages after stage 0 ([`ic_anchor_k`], anchored
/// at `e_off = start_0 − end_date`, width `end_date − start_date`).
///
/// Runs single-threaded in canonical [`TransitBucketTopology::column_order`]
/// order — never a rank-count-dependent parallel reduction.
///
/// `cobre-io`'s `validate_travel_time` coverage gate guarantees every declared
/// arc's windows cover `[start_0 − t_v, start_0)` before this runs; there is no
/// fallback for incomplete coverage.
#[must_use]
pub(crate) fn build_initial_transit_bucket_state(
    system: &System,
    topology: &TransitBucketTopology,
) -> Vec<f64> {
    let mut seed = vec![0.0_f64; topology.n_buckets];
    if topology.n_buckets == 0 {
        return seed;
    }

    let study_durations = study_stage_durations(system);
    let Some(start_0) = study_start_date(system) else {
        debug_assert!(
            false,
            "n_buckets > 0 implies build_transit_bucket_topology sized a depth from a non-empty \
             study calendar, so at least one study stage must exist here"
        );
        return seed;
    };
    let ic = system.initial_conditions();
    let hydros = system.hydros();

    let mut start = 0_usize;
    for &depth in &topology.per_plant_depth {
        let plant_id = hydros[topology.column_order[start].0].id;

        for upstream in hydros {
            let Some(t_v) = upstream.travel_time_hours.filter(|&t| t > 0.0) else {
                continue;
            };
            if upstream.downstream_id != Some(plant_id) {
                continue;
            }

            for window in ic
                .past_defluences
                .iter()
                .filter(|w| w.hydro_id == upstream.id)
            {
                debug_assert!(
                    window.end_date <= start_0,
                    "past_defluences window must end at or before start_0 ({start_0}); \
                     cobre-io's validate_travel_time row-5b gate guarantees this"
                );
                let e_off = hours_between(start_0, window.end_date);
                let width = hours_between(window.end_date, window.start_date);
                let volume = width * M3S_TO_HM3 * window.value_m3s;

                let k = ic_anchor_k(t_v, e_off, width, &study_durations);
                for (transit_bucket_offset, &k_val) in k.iter().enumerate().take(depth) {
                    if k_val != 0.0 {
                        seed[start + transit_bucket_offset] += k_val * volume;
                    }
                }
            }
        }

        start += depth;
    }

    debug_assert_eq!(seed.len(), topology.n_buckets);
    seed
}

/// The first study stage's (`id >= 0`, lowest `id`) start date — `start_0`,
/// the anchor every window's `(e_off, width)` measures against. `None` only
/// when the system declares no study stages.
fn study_start_date(system: &System) -> Option<NaiveDate> {
    system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .min_by_key(|s| s.id)
        .map(|s| s.start_date)
}

/// Hours of wall clock between `earlier` and `later` (`later − earlier`),
/// positive when `earlier` precedes `later`.
// Rationale: pre-study spans are on the order of years, far under f64's
// exact-integer range; a checked conversion buys nothing.
#[allow(clippy::cast_precision_loss)]
fn hours_between(later: NaiveDate, earlier: NaiveDate) -> f64 {
    (later - earlier).num_hours() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use cobre_core::{
        Block, BlockMode, Bus, DeficitSegment, EntityId, Hydro, HydroGenerationModel,
        HydroPastDefluence, HydroPenalties, InitialConditions, NoiseMethod, ScenarioSourceConfig,
        Stage, StageRiskConfig, StageStateConfig, SystemBuilder,
    };

    use crate::setup::bucket_topology::build_transit_bucket_topology;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

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

    fn hydro(id: i32, downstream_id: Option<i32>, travel_time_hours: Option<f64>) -> Hydro {
        Hydro {
            unit_groups: Vec::new(),
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

    /// `n` study stages (`id = 0..n`), each carrying a single `hours`-long
    /// block, all anchored at `start_0 = 2024-01-01`.
    fn study_stages(n: i32, hours: f64) -> Vec<Stage> {
        (0..n)
            .map(|id| Stage {
                index: usize::try_from(id).unwrap_or(0),
                id,
                start_date: date(2024, 1, 1),
                end_date: date(2024, 2, 1),
                season_id: None,
                blocks: vec![Block {
                    index: 0,
                    name: "FLAT".to_string(),
                    duration_hours: hours,
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
            })
            .collect()
    }

    fn build_system(
        hydros: Vec<Hydro>,
        stages: Vec<Stage>,
        past_defluences: Vec<HydroPastDefluence>,
    ) -> cobre_core::System {
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
            .initial_conditions(InitialConditions {
                past_defluences,
                ..InitialConditions::default()
            })
            .build()
            .expect("valid system")
    }

    /// `start_0 = 2024-01-01`. A single `past_defluences` window ending
    /// `start_0_minus_hours` before `start_0` and spanning `width_hours`, at
    /// rate `value` m³/s.
    fn defluence_window(
        hydro_id: i32,
        start_0_minus_hours: f64,
        width_hours: f64,
        value: f64,
    ) -> HydroPastDefluence {
        let start_0 = date(2024, 1, 1);
        let end_date = start_0 - Duration::hours(start_0_minus_hours as i64);
        let start_date = end_date - Duration::hours(width_hours as i64);
        HydroPastDefluence {
            hydro_id: EntityId(hydro_id),
            start_date,
            end_date,
            value_m3s: value,
        }
    }

    /// Single arc, `k = [1/2, 1/2]`, one window `[start_0 − 24h, start_0)` at
    /// 100 m³/s ⇒ `b_1 = k_1 · D = 1/2 · D` (`D` the width-scaled volume,
    /// mirroring how an in-study release is already volume-scaled by `τ`).
    #[test]
    fn test_single_arc_unroll_matches_ac1() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(24.0));
        let system = build_system(
            vec![downstream, upstream],
            study_stages(4, 12.0),
            vec![defluence_window(2, 0.0, 24.0, 100.0)],
        );

        let topology = build_transit_bucket_topology(&system);
        assert_eq!(topology.per_plant_depth, vec![2], "sanity: 2-bucket depth");

        let seed = build_initial_transit_bucket_state(&system, &topology);
        assert_eq!(seed.len(), topology.n_buckets);

        let volume = 24.0 * M3S_TO_HM3 * 100.0;
        assert!(
            (seed[0] - 0.5 * volume).abs() < 1e-9,
            "b_1 must equal 1/2 * volume, got {} vs expected {}",
            seed[0],
            0.5 * volume
        );
    }

    /// A mid-horizon upstream entrant (`entry_stage_id`
    /// mid-study) supplies a zero-valued `past_defluences` window -- the
    /// physically correct value, since the plant did not exist pre-study --
    /// and every stage-0 bucket the arc feeds comes out zero.
    /// [`build_initial_transit_bucket_state`] never reads `entry_stage_id`;
    /// conservation is forced by the input data, not a code branch.
    #[test]
    fn test_mid_horizon_entrant_zero_history_zero_seeds_stage_0_transit_buckets() {
        let downstream = hydro(1, None, None);
        let mut upstream = hydro(2, Some(1), Some(24.0));
        upstream.entry_stage_id = Some(2);
        let system = build_system(
            vec![downstream, upstream],
            study_stages(4, 12.0),
            vec![defluence_window(2, 0.0, 24.0, 0.0)],
        );

        let topology = build_transit_bucket_topology(&system);
        assert_eq!(topology.per_plant_depth, vec![2], "sanity: 2-bucket depth");

        let seed = build_initial_transit_bucket_state(&system, &topology);

        assert!(
            seed.iter().all(|&v| v.abs() < 1e-9),
            "a mid-horizon entrant's zero-valued pre-study history must zero-seed \
             every stage-0 bucket, got {seed:?}"
        );
    }

    /// Confluence: two upstreams with different `t_v` feeding one downstream
    /// plant sum their unrolled shares into the SAME per-plant bucket block.
    #[test]
    fn test_confluence_aggregates_two_upstreams_into_shared_transit_buckets() {
        let downstream = hydro(1, None, None);
        let upstream_a = hydro(2, Some(1), Some(24.0));
        let upstream_b = hydro(3, Some(1), Some(12.0));
        let system = build_system(
            vec![downstream, upstream_a, upstream_b],
            study_stages(4, 12.0),
            vec![
                defluence_window(2, 0.0, 24.0, 100.0),
                defluence_window(3, 0.0, 24.0, 50.0),
            ],
        );

        let topology = build_transit_bucket_topology(&system);
        assert_eq!(topology.per_plant_depth, vec![2], "sanity: 2-bucket depth");

        let seed = build_initial_transit_bucket_state(&system, &topology);

        let vol_a = 24.0 * M3S_TO_HM3 * 100.0;
        let vol_b = 24.0 * M3S_TO_HM3 * 50.0;
        let expected_b1 = 0.5 * vol_a + 0.5 * vol_b;
        let expected_b2 = 0.5 * vol_a;

        assert!(
            (seed[0] - expected_b1).abs() < 1e-9,
            "b_1 must sum both arcs' shares, got {} vs expected {expected_b1}",
            seed[0]
        );
        assert!(
            (seed[1] - expected_b2).abs() < 1e-9,
            "b_2 must carry only the deeper arc's share, got {} vs expected {expected_b2}",
            seed[1]
        );
    }

    /// Declaration-order invariance: swapping the hydro input order must not
    /// change the seed (canonical sort in `SystemBuilder::build` plus the
    /// canonical-index-driven aggregation loop).
    #[test]
    fn test_seed_is_declaration_order_invariant() {
        let downstream = hydro(1, None, None);
        let upstream_a = hydro(2, Some(1), Some(24.0));
        let upstream_b = hydro(3, Some(1), Some(12.0));
        let defluences = vec![
            defluence_window(2, 0.0, 24.0, 100.0),
            defluence_window(3, 0.0, 24.0, 50.0),
        ];

        let system_a = build_system(
            vec![downstream.clone(), upstream_a.clone(), upstream_b.clone()],
            study_stages(4, 12.0),
            defluences.clone(),
        );
        let system_b = build_system(
            vec![upstream_b, upstream_a, downstream],
            study_stages(4, 12.0),
            defluences,
        );

        let topology_a = build_transit_bucket_topology(&system_a);
        let topology_b = build_transit_bucket_topology(&system_b);
        let seed_a = build_initial_transit_bucket_state(&system_a, &topology_a);
        let seed_b = build_initial_transit_bucket_state(&system_b, &topology_b);

        assert_eq!(
            seed_a, seed_b,
            "seed must be bit-identical across input order"
        );
    }

    /// `seed.len() == B` for every declared topology, including when no arc
    /// is declared at all (`B == 0`).
    #[test]
    fn test_seed_len_matches_n_buckets() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(24.0));
        let system = build_system(
            vec![downstream, upstream],
            study_stages(4, 12.0),
            vec![defluence_window(2, 0.0, 24.0, 100.0)],
        );
        let topology = build_transit_bucket_topology(&system);
        let seed = build_initial_transit_bucket_state(&system, &topology);
        assert_eq!(seed.len(), topology.n_buckets);

        let no_arc_downstream = hydro(1, None, None);
        let no_arc_system = build_system(vec![no_arc_downstream], study_stages(3, 24.0), vec![]);
        let no_arc_topology = build_transit_bucket_topology(&no_arc_system);
        assert_eq!(no_arc_topology.n_buckets, 0);
        let no_arc_seed = build_initial_transit_bucket_state(&no_arc_system, &no_arc_topology);
        assert_eq!(no_arc_seed.len(), 0);
    }

    /// Two gapped (non-contiguous) windows for the same 72h arc land in
    /// DISJOINT bucket pairs: the recent window `[start_0 − 24h, start_0)`
    /// arrives at buckets 4-5 (`k = [0, 0, 0, 0, 1/2, 1/2]`), the older
    /// window `[start_0 − 72h, start_0 − 48h)` arrives at buckets 0-1
    /// (`k = [1/2, 1/2]`) -- a genuine 24h gap (`[start_0 − 48h,
    /// start_0 − 24h)`) separates the two release windows. Because the
    /// windows land in disjoint buckets, dropping the older one (the bug a
    /// `.find()` in place of `.filter()` would introduce) zeroes buckets 0-1
    /// and fails the assertion below.
    #[test]
    fn test_gapped_windows_contribute_additively() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(72.0));
        let system = build_system(
            vec![downstream, upstream],
            study_stages(6, 12.0),
            vec![
                defluence_window(2, 0.0, 24.0, 100.0),
                defluence_window(2, 48.0, 24.0, 40.0),
            ],
        );

        let topology = build_transit_bucket_topology(&system);
        let seed = build_initial_transit_bucket_state(&system, &topology);

        let vol_recent = 24.0 * M3S_TO_HM3 * 100.0;
        let vol_older = 24.0 * M3S_TO_HM3 * 40.0;
        let study_durations = study_stage_durations(&system);
        let k_recent = ic_anchor_k(72.0, 0.0, 24.0, &study_durations);
        let k_older = ic_anchor_k(72.0, 48.0, 24.0, &study_durations);

        let mut expected = vec![0.0_f64; topology.n_buckets];
        for (d, &k_val) in k_recent.iter().enumerate() {
            expected[d] += k_val * vol_recent;
        }
        for (d, &k_val) in k_older.iter().enumerate() {
            expected[d] += k_val * vol_older;
        }

        assert_eq!(seed.len(), expected.len());
        for (idx, (&got, &want)) in seed.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-9,
                "bucket {idx}: gapped windows must contribute additively, got {got} vs expected {want}"
            );
        }
    }
}
