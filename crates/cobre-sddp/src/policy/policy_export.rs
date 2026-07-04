//! Policy checkpoint export helpers.
//!
//! Shared conversion logic for extracting active cuts and basis data from a
//! trained [`FutureCostFunction`] and [`TrainingResult`] into the `cobre-io`
//! policy types needed by [`cobre_io::write_policy_checkpoint`].

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use cobre_core::System;
use cobre_io::output::policy::{
    EntitySlot, PolicyBasisRecord, PolicyCutRecord, StageCutsPayload, StageStatesPayload,
};

use crate::cut::FutureCostFunction;
use crate::indexer::{CutStateProjection, StateLayout};
use crate::lp_builder::{commissioning_active, hydro_operating_active};
use crate::training::TrainingResult;

/// `EntityType::HydroStorage` discriminant from `schemas/policy.fbs`.
const ENTITY_TYPE_HYDRO_STORAGE: u8 = 0;
/// `EntityType::HydroInflowLag` discriminant from `schemas/policy.fbs`.
const ENTITY_TYPE_HYDRO_INFLOW_LAG: u8 = 1;
/// `EntityType::AnticipatedThermalState` discriminant from `schemas/policy.fbs`.
const ENTITY_TYPE_ANTICIPATED_THERMAL_STATE: u8 = 2;
/// `EntityType::HydroTransitBucket` discriminant from `schemas/policy.fbs`.
const ENTITY_TYPE_HYDRO_TRANSIT_BUCKET: u8 = 3;

/// Build the per-slot entity-identity manifest for one stage's cut pool: one
/// [`EntitySlot`] per enabled cut-state dimension of `projection`.
///
/// Slots are emitted in `projection`'s storage → lag → buckets → anticipated order, so slot
/// `j` describes the entity owning positional coefficient `j` — the order a
/// consumer matches the manifest against the cut coefficients. Each slot is
/// classified by the global [`StateLayout`] region containing its incoming-state
/// column ([`CutStateProjection::state_to_lp_incoming_column`]), never by
/// re-deriving column arithmetic.
///
/// # Panics (debug builds only)
///
/// Panics if the built manifest length differs from `projection.n_state()`.
#[must_use]
pub fn build_stage_entity_manifest(
    system: &System,
    global_layout: &StateLayout,
    projection: &CutStateProjection,
    stage_id: i32,
) -> Vec<EntitySlot> {
    let n = global_layout.hydro_count;
    let n_anticipated = global_layout.n_anticipated;
    let hydros = system.hydros();
    let anticipated_thermals: Vec<&cobre_core::Thermal> = system
        .thermals()
        .iter()
        .filter(|t| t.anticipated_config.is_some())
        .collect();

    let mut manifest = Vec::with_capacity(projection.n_state());
    for j in 0..projection.n_state() {
        let col = projection.state_to_lp_incoming_column(j);
        let slot = if global_layout.storage_in.contains(&col) {
            let h = col - global_layout.storage_in.start;
            let hydro = &hydros[h];
            EntitySlot {
                entity_type: ENTITY_TYPE_HYDRO_STORAGE,
                entity_id: hydro.id.0,
                subindex: 0,
                was_active: hydro_operating_active(
                    hydro.filling.as_ref(),
                    hydro.entry_stage_id,
                    hydro.exit_stage_id,
                    stage_id,
                ),
            }
        } else if global_layout.inflow_lags.contains(&col) {
            let offset = col - global_layout.inflow_lags.start;
            let lag = offset / n;
            let h = offset % n;
            let hydro = &hydros[h];
            EntitySlot {
                entity_type: ENTITY_TYPE_HYDRO_INFLOW_LAG,
                entity_id: hydro.id.0,
                subindex: (lag + 1) as u32,
                was_active: hydro_operating_active(
                    hydro.filling.as_ref(),
                    hydro.entry_stage_id,
                    hydro.exit_stage_id,
                    stage_id,
                ),
            }
        } else if global_layout.transit_buckets_in.contains(&col) {
            let b = col - global_layout.transit_buckets_in.start;
            let (plant_idx, lag) = global_layout.transit_bucket_column_order[b];
            let hydro = &hydros[plant_idx];
            EntitySlot {
                entity_type: ENTITY_TYPE_HYDRO_TRANSIT_BUCKET,
                entity_id: hydro.id.0,
                subindex: lag as u32,
                was_active: hydro_operating_active(
                    hydro.filling.as_ref(),
                    hydro.entry_stage_id,
                    hydro.exit_stage_id,
                    stage_id,
                ),
            }
        } else {
            debug_assert!(
                global_layout.anticipated_state.contains(&col),
                "incoming column {col} must lie in storage_in, inflow_lags, transit_buckets_in, or \
                 anticipated_state"
            );
            let offset = col - global_layout.anticipated_state.start;
            let plant_pos = offset % n_anticipated;
            let slot_idx = offset / n_anticipated;
            let plant = anticipated_thermals[plant_pos];
            EntitySlot {
                entity_type: ENTITY_TYPE_ANTICIPATED_THERMAL_STATE,
                entity_id: plant.id.0,
                subindex: slot_idx as u32,
                was_active: commissioning_active(
                    plant.entry_stage_id,
                    plant.exit_stage_id,
                    stage_id,
                ),
            }
        };
        manifest.push(slot);
    }

    debug_assert_eq!(
        manifest.len(),
        projection.n_state(),
        "manifest length must equal projection.n_state()"
    );
    manifest
}

/// Build per-stage vectors of **all** populated [`PolicyCutRecord`]s from the FCF pools.
///
/// Both active and inactive cuts are included so the checkpoint preserves the
/// full training history. Use [`build_active_indices`] for the active subset.
#[must_use]
pub fn build_stage_cut_records(fcf: &FutureCostFunction) -> Vec<Vec<PolicyCutRecord<'_>>> {
    fcf.pools
        .iter()
        .map(|pool| {
            (0..pool.populated_count)
                .map(|i| {
                    let meta = &pool.metadata[i];
                    PolicyCutRecord {
                        cut_id: meta.iteration_generated * u64::from(pool.forward_passes)
                            + u64::from(meta.forward_pass_index),
                        slot_index: i as u32,
                        iteration: meta.iteration_generated as u32,
                        forward_pass_index: meta.forward_pass_index,
                        intercept: pool.intercepts[i],
                        coefficients: &pool.coefficients
                            [i * pool.state_dimension..(i + 1) * pool.state_dimension],
                        is_active: pool.active[i],
                    }
                })
                .collect()
        })
        .collect()
}

/// Build per-stage active cut index lists from the stage cut records.
#[must_use]
pub fn build_active_indices(stage_records: &[Vec<PolicyCutRecord<'_>>]) -> Vec<Vec<u32>> {
    stage_records
        .iter()
        .map(|records| {
            records
                .iter()
                .filter(|r| r.is_active)
                .map(|r| r.slot_index)
                .collect()
        })
        .collect()
}

/// Build [`StageCutsPayload`] references from pre-built records, indices, and
/// per-stage entity manifests.
///
/// `stage_records`, `stage_active_indices`, and `stage_manifests` must have been
/// built from the same `fcf` (via [`build_stage_cut_records`],
/// [`build_active_indices`], and [`build_stage_entity_manifest`] per pool), so
/// each is indexed by the same pool index. `stage_manifests[t]` carries one slot
/// per cut-state dimension of pool `t`.
#[must_use]
pub fn build_stage_cuts_payloads<'a>(
    fcf: &FutureCostFunction,
    stage_records: &'a [Vec<PolicyCutRecord<'a>>],
    stage_active_indices: &'a [Vec<u32>],
    stage_manifests: &'a [Vec<EntitySlot>],
) -> Vec<StageCutsPayload<'a>> {
    fcf.pools
        .iter()
        .enumerate()
        .map(|(stage_idx, pool)| StageCutsPayload {
            stage_id: stage_idx as u32,
            state_dimension: fcf.state_dimension as u32,
            capacity: pool.capacity as u32,
            warm_start_count: pool.warm_start_count,
            cuts: &stage_records[stage_idx],
            active_cut_indices: &stage_active_indices[stage_idx],
            populated_count: pool.populated_count as u32,
            entity_manifest: &stage_manifests[stage_idx],
        })
        .collect()
}

/// Convert the solver basis cache from i32 status codes to u8 byte vectors.
///
/// `HiGHS` status codes are in the range 0..=4, so the truncation is safe.
/// Returns `(col_status_bytes, row_status_bytes)`.
#[must_use]
pub fn convert_basis_cache(training_result: &TrainingResult) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let col = training_result
        .basis_cache
        .iter()
        .map(|opt| {
            opt.as_ref()
                .map(|cb| cb.basis.col_status.iter().map(|&v| v as u8).collect())
                .unwrap_or_default()
        })
        .collect();
    let row = training_result
        .basis_cache
        .iter()
        .map(|opt| {
            opt.as_ref()
                .map(|cb| cb.basis.row_status.iter().map(|&v| v as u8).collect())
                .unwrap_or_default()
        })
        .collect();
    (col, row)
}

/// Build per-stage [`PolicyBasisRecord`] references from pre-converted basis data.
#[must_use]
pub fn build_stage_basis_records<'a>(
    fcf: &FutureCostFunction,
    training_result: &TrainingResult,
    basis_col_u8: &'a [Vec<u8>],
    basis_row_u8: &'a [Vec<u8>],
) -> Vec<PolicyBasisRecord<'a>> {
    training_result
        .basis_cache
        .iter()
        .enumerate()
        .filter_map(|(stage_idx, opt)| {
            opt.as_ref().map(|_| {
                let num_cut_rows = fcf
                    .pools
                    .get(stage_idx)
                    .map_or(0, |pool| pool.populated_count.min(pool.capacity) as u32);
                PolicyBasisRecord {
                    stage_id: stage_idx as u32,
                    iteration: training_result.iterations as u32,
                    column_status: &basis_col_u8[stage_idx],
                    row_status: &basis_row_u8[stage_idx],
                    num_cut_rows,
                }
            })
        })
        .collect()
}

/// Build per-stage [`StageStatesPayload`]s from the visited states archive.
///
/// Returns an empty `Vec` if the archive is `None` (non-Dominated strategies).
/// `stage_manifests[t]` is the same per-pool entity manifest the cut payloads
/// carry, attached to stage `t`'s states payload.
#[must_use]
pub fn build_stage_states_payloads<'a>(
    archive: Option<&'a crate::visited_states::VisitedStatesArchive>,
    stage_manifests: &'a [Vec<EntitySlot>],
) -> Vec<StageStatesPayload<'a>> {
    let Some(archive) = archive else {
        return Vec::new();
    };
    (0..archive.num_stages())
        .map(|t| {
            let stage = archive.stage(t);
            StageStatesPayload {
                stage_id: t as u32,
                state_dimension: stage.state_dimension() as u32,
                count: stage.count() as u32,
                data: stage.states(),
                entity_manifest: &stage_manifests[t],
            }
        })
        .collect()
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
mod tests {
    use super::{
        ENTITY_TYPE_ANTICIPATED_THERMAL_STATE, ENTITY_TYPE_HYDRO_INFLOW_LAG,
        ENTITY_TYPE_HYDRO_STORAGE, ENTITY_TYPE_HYDRO_TRANSIT_BUCKET, build_stage_entity_manifest,
    };
    use crate::indexer::{CutStateProjection, StateLayout};
    use crate::lp_builder::hydro_operating_active;
    use crate::test_support;
    use cobre_core::temporal::StageStateConfig;
    use cobre_core::{
        AnticipatedConfig, Block, BlockMode, Bus, DeficitSegment, EntityId, Hydro,
        HydroGenerationModel, HydroPenalties, NoiseMethod, ScenarioSourceConfig, Stage,
        StageRiskConfig, System, SystemBuilder, Thermal,
        resolved::{
            BoundsCountsSpec, BoundsDefaults, ContractStageBounds, HydroStageBounds,
            LineStageBounds, PumpingStageBounds, ResolvedBounds, ThermalStageBounds,
        },
    };

    const ALL_ENABLED: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    const STORAGE_ONLY: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: false,
    };

    fn penalties_zero() -> HydroPenalties {
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
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    fn make_hydro(id: i32, entry: Option<i32>, exit: Option<i32>) -> Hydro {
        Hydro {
            id: EntityId(id),
            name: format!("Hydro{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: entry,
            exit_stage_id: exit,
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
            penalties: penalties_zero(),
        }
    }

    fn anticipated_thermal(id: i32, lead_stages: u32) -> Thermal {
        Thermal {
            id: EntityId(id),
            name: format!("Thermal{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: Some(AnticipatedConfig { lead_stages }),
        }
    }

    fn make_bus() -> Bus {
        Bus {
            id: EntityId(1),
            name: "Bus1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        }
    }

    fn make_stage() -> Stage {
        Stage {
            index: 0,
            id: 0,
            start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 720.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: ALL_ENABLED,
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// `System` with 2 hydros and 1 anticipated thermal (`lead_stages = 2`),
    /// matching the `N=2, L=2, A=1, k_max=2` layout fixture. `hydros` carry the
    /// supplied commissioning windows so `was_active` can be exercised.
    fn system_2h_1ant(
        h1_window: (Option<i32>, Option<i32>),
        h2_window: (Option<i32>, Option<i32>),
    ) -> System {
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
                k_max: 2,
            },
            &BoundsDefaults {
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
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                    cost_per_mwh: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 500.0,
                    reverse_mw: 500.0,
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
        SystemBuilder::new()
            .buses(vec![make_bus()])
            .hydros(vec![
                make_hydro(1, h1_window.0, h1_window.1),
                make_hydro(2, h2_window.0, h2_window.1),
            ])
            .thermals(vec![anticipated_thermal(1, 2)])
            .stages(vec![make_stage()])
            .bounds(bounds)
            .build()
            .expect("valid system")
    }

    /// The `N=2, L=2, A=1, k_max=2` global layout the fixture system maps onto.
    fn layout_2h_1ant() -> StateLayout {
        test_support::state_layout_full(2, 2, 1, 2, vec![2])
    }

    /// All-enabled projection: length 8 (2 storage + 4 lag + 2 anticipated), with
    /// storage slots typed 0 (subindex 0), lag slots typed 1 in lag-major order
    /// (hydro-interleaved: subindex 1,1,2,2 for hydros 1,2,1,2), and anticipated
    /// slots typed 2 (plant id 1, ring subindex 0,1).
    #[test]
    fn all_enabled_classification_identity_and_subindex() {
        let system = system_2h_1ant((None, None), (None, None));
        let global = layout_2h_1ant();
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        let manifest = build_stage_entity_manifest(&system, &global, &projection, 0);

        assert_eq!(manifest.len(), projection.n_state());
        assert_eq!(manifest.len(), 8);

        assert_eq!(manifest[0].entity_type, ENTITY_TYPE_HYDRO_STORAGE);
        assert_eq!(manifest[0].entity_id, 1);
        assert_eq!(manifest[0].subindex, 0);
        assert_eq!(manifest[1].entity_type, ENTITY_TYPE_HYDRO_STORAGE);
        assert_eq!(manifest[1].entity_id, 2);
        assert_eq!(manifest[1].subindex, 0);

        // Lag block, lag-major: (lag0,h0),(lag0,h1),(lag1,h0),(lag1,h1).
        for (slot, (expected_id, expected_lag)) in
            [(2, (1, 1)), (3, (2, 1)), (4, (1, 2)), (5, (2, 2))]
        {
            assert_eq!(
                manifest[slot].entity_type, ENTITY_TYPE_HYDRO_INFLOW_LAG,
                "slot {slot} must be an inflow-lag slot"
            );
            assert_eq!(
                manifest[slot].entity_id, expected_id,
                "slot {slot} hydro id"
            );
            assert_eq!(
                manifest[slot].subindex, expected_lag,
                "slot {slot} 1-based lag"
            );
        }

        // Anticipated block, slot-major (single plant, ring slots 0 and 1).
        assert_eq!(
            manifest[6].entity_type,
            ENTITY_TYPE_ANTICIPATED_THERMAL_STATE
        );
        assert_eq!(manifest[6].entity_id, 1);
        assert_eq!(manifest[6].subindex, 0);
        assert_eq!(
            manifest[7].entity_type,
            ENTITY_TYPE_ANTICIPATED_THERMAL_STATE
        );
        assert_eq!(manifest[7].entity_id, 1);
        assert_eq!(manifest[7].subindex, 1);
    }

    /// Bucket block classification (`N=2, L=2, B=2, A=1, k_max=2`): the two
    /// travel-time bucket slots sit between the lag block and the anticipated block,
    /// each carrying `entity_type == HydroTransitBucket`, `entity_id ==` the
    /// downstream hydro id (`transit_bucket_column_order[b].0` into `system.hydros()`), and
    /// `subindex ==` the maturity lag `d` (`transit_bucket_column_order[b].1`).
    #[test]
    fn bucket_slots_classify_as_transit_bucket_with_downstream_id_and_lag() {
        let system = system_2h_1ant((None, None), (None, None));
        let global = test_support::state_layout_with_transit_buckets(
            2,
            2,
            2,
            vec![(0, 1), (1, 2)],
            1,
            2,
            vec![2],
        );
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        let manifest = build_stage_entity_manifest(&system, &global, &projection, 0);

        assert_eq!(manifest.len(), projection.n_state());
        assert_eq!(
            manifest.len(),
            10,
            "2 storage + 4 lag + 2 buckets + 2 anticipated"
        );

        for (slot, (expected_id, expected_lag)) in [(6, (1, 1)), (7, (2, 2))] {
            assert_eq!(
                manifest[slot].entity_type, ENTITY_TYPE_HYDRO_TRANSIT_BUCKET,
                "slot {slot} must be a transit-bucket slot"
            );
            assert_eq!(
                manifest[slot].entity_id, expected_id,
                "slot {slot} downstream hydro id"
            );
            assert_eq!(
                manifest[slot].subindex, expected_lag,
                "slot {slot} maturity lag d"
            );
        }

        assert_eq!(
            manifest[5].entity_type, ENTITY_TYPE_HYDRO_INFLOW_LAG,
            "buckets must follow the lag block"
        );
        assert_eq!(
            manifest[8].entity_type, ENTITY_TYPE_ANTICIPATED_THERMAL_STATE,
            "buckets must precede the anticipated block"
        );
    }

    /// Storage-only projection (`inflow_lags: false`): the lag block is dropped, so
    /// the manifest is length 4 (2 storage + 2 anticipated) and carries NO type-1
    /// (`HydroInflowLag`) slot. Anticipated state is always included.
    #[test]
    fn storage_only_drops_lag_slots() {
        let system = system_2h_1ant((None, None), (None, None));
        let global = layout_2h_1ant();
        let projection = CutStateProjection::new(&global, STORAGE_ONLY);

        let manifest = build_stage_entity_manifest(&system, &global, &projection, 0);

        assert_eq!(manifest.len(), projection.n_state());
        assert_eq!(manifest.len(), 4);
        assert!(
            manifest
                .iter()
                .all(|s| s.entity_type != ENTITY_TYPE_HYDRO_INFLOW_LAG),
            "storage-only manifest must contain no HydroInflowLag slot"
        );
        assert_eq!(manifest[0].entity_type, ENTITY_TYPE_HYDRO_STORAGE);
        assert_eq!(manifest[1].entity_type, ENTITY_TYPE_HYDRO_STORAGE);
        assert_eq!(
            manifest[2].entity_type,
            ENTITY_TYPE_ANTICIPATED_THERMAL_STATE
        );
        assert_eq!(manifest[2].subindex, 0);
        assert_eq!(
            manifest[3].entity_type,
            ENTITY_TYPE_ANTICIPATED_THERMAL_STATE
        );
        assert_eq!(manifest[3].subindex, 1);
    }

    /// A hydro dormant at the slot's stage (commissioning window `[2, 5)` queried at
    /// stage 1) yields `was_active == false` on every slot it owns, and that value
    /// equals the single-owner `hydro_operating_active` predicate. The second hydro
    /// (no window) stays active, isolating the per-entity flag.
    #[test]
    fn was_active_matches_hydro_operating_active_for_dormant_window() {
        let h1_window = (Some(2), Some(5));
        let system = system_2h_1ant(h1_window, (None, None));
        let global = layout_2h_1ant();
        let projection = CutStateProjection::new(&global, ALL_ENABLED);
        let stage_id = 1;

        let manifest = build_stage_entity_manifest(&system, &global, &projection, stage_id);

        let expected_h1 = hydro_operating_active(None, h1_window.0, h1_window.1, stage_id);
        assert!(!expected_h1, "hydro 1 must be dormant at stage 1");

        // Hydro 1 owns storage slot 0 and lag slots 2, 4 (lag-major, h index 0).
        for slot in [0, 2, 4] {
            assert_eq!(manifest[slot].entity_id, 1, "slot {slot} must be hydro 1");
            assert_eq!(
                manifest[slot].was_active, expected_h1,
                "slot {slot} was_active must equal hydro_operating_active"
            );
        }
        // Hydro 2 has no window: active.
        for slot in [1, 3, 5] {
            assert_eq!(manifest[slot].entity_id, 2, "slot {slot} must be hydro 2");
            assert!(manifest[slot].was_active, "slot {slot} hydro 2 is active");
        }
    }
}
