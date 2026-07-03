//! Stage-0 incoming travel-time bucket seed from pre-study upstream releases.
//!
//! [`build_initial_bucket_state`] unrolls each declared arc's pre-study
//! defluence history through the IC-anchor overlap arithmetic (water memo
//! §4.1/§2.2) into the stage-0 incoming bucket state, aggregated per
//! downstream plant exactly as [`BucketTopology`] aggregates depth. The
//! caller splices the result into the same `initial_state` vector
//! `build_initial_state` populates, so the single `fill_col_state_patches`
//! pin path picks it up with no separate wiring.

use cobre_core::{EntityId, InitialConditions, System};

use crate::lp_builder::M3S_TO_HM3;

use super::bucket_topology::{
    BucketTopology, ic_anchor_k, pre_study_period_durations_desc, required_history_periods,
    study_stage_durations,
};

/// Unroll every declared arc's pre-study release history into the stage-0
/// incoming bucket seed, in [`BucketTopology::column_order`] order.
///
/// For downstream plant `j` with depth `L_j`, bucket `d` (`1..=L_j`) is
/// `Σ_{i∈up(j)} Σ_{m≥1} k_{d-1+m,i}·D_i^{-m}`: `D_i^{-m}` is the volume of
/// arc `i`'s `m`-th most-recent pre-study defluence (period-duration-scaled,
/// mirroring how an in-study release is already volume-scaled by `τ`), and
/// `k_{d-1+m,i}` is [`ic_anchor_k`]'s fraction of that period's release
/// landing `d-1` study stages after stage 0.
///
/// Runs single-threaded in [`BucketTopology::column_order`]'s canonical
/// (sorted) order — never a rank-count-dependent parallel reduction (water
/// memo §6 item 6).
///
/// Falls back to `past_inflows` when a declared arc's `past_defluences`
/// history is shorter than its required depth (D5) — the same choice
/// `cobre-io`'s `validate_travel_time` row-5 gate already enforced (and
/// already hard-errors / logs the caveat there); this only consumes whichever
/// history that validation accepted.
#[must_use]
pub(crate) fn build_initial_bucket_state(system: &System, topology: &BucketTopology) -> Vec<f64> {
    let mut seed = vec![0.0_f64; topology.b_total];
    if topology.b_total == 0 {
        return seed;
    }

    let study_durations = study_stage_durations(system);
    let pre_study_desc = pre_study_period_durations_desc(system);
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

            let required = required_history_periods(t_v, &pre_study_desc);
            let history = select_history(ic, upstream.id, required);

            for (offset, &raw_value) in history.iter().enumerate() {
                let m = offset + 1;
                if m > pre_study_desc.len() {
                    break;
                }
                let period_duration = pre_study_desc[m - 1];
                let cumulative_before: f64 = pre_study_desc[..m - 1].iter().sum();
                let volume = period_duration * M3S_TO_HM3 * raw_value;

                let k = ic_anchor_k(t_v, cumulative_before, period_duration, &study_durations);
                for (bucket_offset, &k_val) in k.iter().enumerate().take(depth) {
                    if k_val != 0.0 {
                        seed[start + bucket_offset] += k_val * volume;
                    }
                }
            }
        }

        start += depth;
    }

    debug_assert_eq!(seed.len(), topology.b_total);
    seed
}

/// Select the arc's pre-study release history: `past_defluences` when it
/// meets `required`, else the whole `past_inflows` proxy (D5) — the fallback
/// `cobre-io`'s row-5 validation already decided; neither being long enough
/// would already have hard-errored before setup runs this.
fn select_history(ic: &InitialConditions, hydro_id: EntityId, required: usize) -> &[f64] {
    ic.past_defluences
        .iter()
        .find(|e| e.hydro_id == hydro_id)
        .map(|e| e.values_m3s.as_slice())
        .filter(|values| values.len() >= required)
        .unwrap_or_else(|| {
            ic.past_inflows
                .iter()
                .find(|e| e.hydro_id == hydro_id)
                .map_or(&[], |e| e.values_m3s.as_slice())
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use cobre_core::{
        Block, BlockMode, Bus, DeficitSegment, EntityId, Hydro, HydroGenerationModel,
        HydroPastDefluences, HydroPenalties, NoiseMethod, ScenarioSourceConfig, Stage,
        StageRiskConfig, StageStateConfig, SystemBuilder,
    };

    use crate::setup::bucket_topology::build_bucket_topology;

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

    /// A single 24h pre-study period (`id = -1`) immediately before stage 0.
    fn pre_study_stage_24h() -> Stage {
        Stage {
            index: 0,
            id: -1,
            start_date: date(2023, 12, 31),
            end_date: date(2024, 1, 1),
            season_id: None,
            blocks: vec![],
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
        }
    }

    /// `n` study stages (`id = 0..n`), each carrying a single `hours`-long block.
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
        mut stages: Vec<Stage>,
        past_defluences: Vec<HydroPastDefluences>,
    ) -> cobre_core::System {
        stages.push(pre_study_stage_24h());
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
            .initial_conditions(cobre_core::InitialConditions {
                past_defluences,
                ..cobre_core::InitialConditions::default()
            })
            .build()
            .expect("valid system")
    }

    fn defluence(hydro_id: i32, values_m3s: Vec<f64>) -> HydroPastDefluences {
        HydroPastDefluences {
            hydro_id: EntityId(hydro_id),
            values_m3s,
            season_ids: None,
        }
    }

    /// Single arc, `k = [1/2, 1/2]`, one pre-study release `D^{-1}` ⇒
    /// `b_1 = k_1 · D^{-1} = 1/2 · D^{-1}` (`D^{-1}` the period-duration-scaled
    /// volume, mirroring the memo's already-volume-scaled `D_i^t`).
    #[test]
    fn test_single_arc_unroll_matches_ac1() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(24.0));
        let system = build_system(
            vec![downstream, upstream],
            study_stages(4, 12.0),
            vec![defluence(2, vec![100.0])],
        );

        let topology = build_bucket_topology(&system);
        assert_eq!(topology.per_plant_depth, vec![2], "sanity: 2-bucket depth");

        let seed = build_initial_bucket_state(&system, &topology);
        assert_eq!(seed.len(), topology.b_total);

        let d_minus_1 = 24.0 * M3S_TO_HM3 * 100.0;
        assert!(
            (seed[0] - 0.5 * d_minus_1).abs() < 1e-9,
            "b_1 must equal 1/2 * D^-1, got {} vs expected {}",
            seed[0],
            0.5 * d_minus_1
        );
    }

    /// A mid-horizon upstream entrant (`entry_stage_id`
    /// mid-study) supplies zero-valued `past_defluences` -- the physically
    /// correct value, since the plant did not exist pre-study -- and every
    /// stage-0 bucket the arc feeds comes out zero. [`build_initial_bucket_state`]
    /// never reads `entry_stage_id`; conservation is forced by the input
    /// data, not a code branch.
    #[test]
    fn test_mid_horizon_entrant_zero_history_zero_seeds_stage_0_buckets() {
        let downstream = hydro(1, None, None);
        let mut upstream = hydro(2, Some(1), Some(24.0));
        upstream.entry_stage_id = Some(2);
        let system = build_system(
            vec![downstream, upstream],
            study_stages(4, 12.0),
            vec![defluence(2, vec![0.0])],
        );

        let topology = build_bucket_topology(&system);
        assert_eq!(topology.per_plant_depth, vec![2], "sanity: 2-bucket depth");

        let seed = build_initial_bucket_state(&system, &topology);

        assert!(
            seed.iter().all(|&v| v.abs() < 1e-9),
            "a mid-horizon entrant's zero-valued pre-study history must zero-seed \
             every stage-0 bucket, got {seed:?}"
        );
    }

    /// Confluence: two upstreams with different `t_v` feeding one downstream
    /// plant sum their unrolled shares into the SAME per-plant bucket block.
    #[test]
    fn test_confluence_aggregates_two_upstreams_into_shared_buckets() {
        let downstream = hydro(1, None, None);
        let upstream_a = hydro(2, Some(1), Some(24.0));
        let upstream_b = hydro(3, Some(1), Some(12.0));
        let system = build_system(
            vec![downstream, upstream_a, upstream_b],
            study_stages(4, 12.0),
            vec![defluence(2, vec![100.0]), defluence(3, vec![50.0])],
        );

        let topology = build_bucket_topology(&system);
        assert_eq!(topology.per_plant_depth, vec![2], "sanity: 2-bucket depth");

        let seed = build_initial_bucket_state(&system, &topology);

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
        let defluences = vec![defluence(2, vec![100.0]), defluence(3, vec![50.0])];

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

        let topology_a = build_bucket_topology(&system_a);
        let topology_b = build_bucket_topology(&system_b);
        let seed_a = build_initial_bucket_state(&system_a, &topology_a);
        let seed_b = build_initial_bucket_state(&system_b, &topology_b);

        assert_eq!(
            seed_a, seed_b,
            "seed must be bit-identical across input order"
        );
    }

    /// `seed.len() == B` for every declared topology, including when no arc
    /// is declared at all (`B == 0`).
    #[test]
    fn test_seed_len_matches_b_total() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(24.0));
        let system = build_system(
            vec![downstream, upstream],
            study_stages(4, 12.0),
            vec![defluence(2, vec![100.0])],
        );
        let topology = build_bucket_topology(&system);
        let seed = build_initial_bucket_state(&system, &topology);
        assert_eq!(seed.len(), topology.b_total);

        let no_arc_downstream = hydro(1, None, None);
        let no_arc_system = build_system(vec![no_arc_downstream], study_stages(3, 24.0), vec![]);
        let no_arc_topology = build_bucket_topology(&no_arc_system);
        assert_eq!(no_arc_topology.b_total, 0);
        let no_arc_seed = build_initial_bucket_state(&no_arc_system, &no_arc_topology);
        assert_eq!(no_arc_seed.len(), 0);
    }
}
