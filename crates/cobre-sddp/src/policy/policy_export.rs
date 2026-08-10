//! Policy checkpoint export helpers.
//!
//! Shared conversion logic for extracting active cuts and basis data from a
//! trained [`FutureCostFunction`] and [`TrainingResult`] into the `cobre-io`
//! policy types needed by [`cobre_io::write_policy_checkpoint`].

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use crate::visited_states::VisitedStatesArchive;
use cobre_core::System;
use cobre_core::Thermal;
use cobre_core::commissioning::{commissioning_active, hydro_operating_active};
use cobre_io::output::policy::{
    ENTITY_SLOT_DELIVERY_DATE_SENTINEL, EntitySlot, GraphManifest, ManifestEdge, ManifestNode,
    OwnedPolicyCutRecord, PolicyBasisRecord, PolicyCutRecord, StageCutsPayload, StageStatesPayload,
};

use crate::cut::FutureCostFunction;
use crate::indexer::{CutSlot, CutStateProjection, StateRegion, StateSpace};
use crate::lp_builder::delivery_ring::DeliveryRing;
use crate::setup::{NodeGraph, NodePos};
use crate::training::TrainingResult;

/// `EntityType::HydroStorage` discriminant from `schemas/policy.fbs`.
const ENTITY_TYPE_HYDRO_STORAGE: u8 = 0;
/// `EntityType::HydroInflowLag` discriminant from `schemas/policy.fbs`.
pub(crate) const ENTITY_TYPE_HYDRO_INFLOW_LAG: u8 = 1;
/// `EntityType::AnticipatedThermalState` discriminant from `schemas/policy.fbs`.
const ENTITY_TYPE_ANTICIPATED_THERMAL_STATE: u8 = 2;
/// `EntityType::HydroTransitBucket` discriminant from `schemas/policy.fbs`.
const ENTITY_TYPE_HYDRO_TRANSIT_BUCKET: u8 = 3;

/// Canonical absolute delivery/arrival calendar date of a stage `start_date`,
/// encoded `year * 10000 + month * 100 + day` (`YYYYMMDD`). The day is pinned to
/// `01` so the anchor stays month-granular — the same calendar month maps to the
/// same date whether resolved from a weekly or a monthly stage.
fn year_month_day_anchor(date: chrono::NaiveDate) -> i32 {
    use chrono::Datelike;
    // `month()` is 1..=12, so the conversion never fails.
    date.year() * 10_000 + i32::try_from(date.month()).unwrap_or(1) * 100 + 1
}

/// Build the per-slot entity-identity manifest for one stage's cut pool: one
/// [`EntitySlot`] per enabled cut-state dimension of `projection`.
///
/// Slots are emitted in `projection`'s storage → lag → buckets → anticipated →
/// commitment-block order, so slot `j` describes the entity owning positional
/// coefficient `j` — the order a consumer matches the manifest against the cut
/// coefficients. Each slot is classified by the global [`StateSpace`] region
/// containing its incoming-state column ([`CutStateProjection::incoming_column`]),
/// never by re-deriving column arithmetic.
///
/// Each hydro-region slot's `was_active` comes from [`hydro_operating_active`]. A
/// commitment-block slot reuses `EntityType::AnticipatedThermalState` (no new
/// discriminant), keyed on its owning thermal's id with
/// `subindex = k_max + local_window_index` — offset past the ring's own
/// `[0, k_max)` domain so a thermal owning both ring slots and commitment
/// windows never collides on `(entity_type, entity_id, subindex)`; its
/// `delivery_date` is always the sentinel (real per-window dates are a
/// separate concern this manifest does not resolve).
///
/// # Panics (debug builds only)
///
/// Panics if the built manifest length differs from `projection.n_slots()`.
#[must_use]
pub fn build_stage_entity_manifest(
    system: &System,
    global_layout: &StateSpace,
    projection: &CutStateProjection,
    stage_id: i32,
) -> Vec<EntitySlot> {
    let n = global_layout.hydro_count;
    let n_anticipated = global_layout.n_anticipated;
    let hydros = system.hydros();
    let anticipated_thermals: Vec<&Thermal> = system
        .thermals()
        .iter()
        .filter(|t| t.anticipated_config.is_some())
        .collect();
    // Single owner of the anticipated ring's slot-major/plant-minor addressing
    // (see `lp_builder::delivery_ring`); only `slot_lane_at`'s reverse
    // decomposition is read here — the manifest never emits ring rows/columns.
    let anticipated_ring = DeliveryRing::new(
        global_layout.anticipated_slots_out.clone(),
        global_layout.anticipated_state.clone(),
        n_anticipated,
        global_layout.k_max,
    );

    // Study stages in canonical index order — the space `AnticipatedResolution`'s
    // decider/depth and the bucket topology both index.
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let current_stage_idx = study_stages.iter().position(|s| s.id == stage_id);
    // Delivery/arrival calendar anchor of the stage `current_stage_idx + offset`;
    // sentinel when this stage is unknown or the target lands past the horizon.
    let anchor_at = |offset: usize| -> i32 {
        current_stage_idx
            .and_then(|t| study_stages.get(t + offset))
            .map_or(ENTITY_SLOT_DELIVERY_DATE_SENTINEL, |s| {
                year_month_day_anchor(s.start_date)
            })
    };

    let anticipated_slot = |offset: usize| -> EntitySlot {
        let (slot_idx, plant_pos) = anticipated_ring.slot_lane_at(offset);
        let plant = anticipated_thermals[plant_pos];
        // The ring shifts one stage per transition, so slot `slot_idx` observed
        // at stage `t` matures at delivery stage `t + slot_idx` regardless of
        // lead mode (constant `LeadStages` or calendar-derived `LeadTime`).
        // Reachability uses the SAME per-plant bound the LP masking itself
        // uses (`anticipated_lead_stages[plant_pos]`, populated from
        // `AnticipatedResolution` for both lead modes by
        // `resolve_anticipated_commitments`): a slot beyond it is structural
        // padding (frozen `[0, 0]`), not a real commitment, even when
        // `t + slot_idx` itself still lands inside the horizon — the
        // multi-plant heterogeneous-lead case.
        let k_i = global_layout.anticipated_lead_stages[plant_pos];
        let delivery_date = if slot_idx < k_i {
            current_stage_idx.map_or(ENTITY_SLOT_DELIVERY_DATE_SENTINEL, |t| {
                let m = t + slot_idx;
                // Defensive cross-check against the resolver (the single owner
                // of c(m), from `resolve_point`): a within-study decider must
                // have already fired by `t`; a pre-study (IC-seeded) delivery
                // has no decider entry and is exempt.
                debug_assert!(
                    global_layout
                        .anticipated_resolution
                        .per_plant
                        .get(plant_pos)
                        .and_then(|resolution| resolution.decider.get(m))
                        .copied()
                        .flatten()
                        .is_none_or(|decided_at| decided_at <= t),
                    "anticipated delivery {m} observed at stage {t} was not yet decided"
                );
                anchor_at(slot_idx)
            })
        } else {
            ENTITY_SLOT_DELIVERY_DATE_SENTINEL
        };
        EntitySlot {
            entity_type: ENTITY_TYPE_ANTICIPATED_THERMAL_STATE,
            entity_id: plant.id.0,
            subindex: slot_idx as u32,
            was_active: commissioning_active(plant.entry_stage_id, plant.exit_stage_id, stage_id),
            delivery_date,
        }
    };

    // Reuses the anticipated-thermal family (no new EntityType discriminant):
    // `subindex = k_max + local_window_index`, where `local_window_index` is
    // the count of prior windows sharing this owner in
    // `commitment_window_thermal_id`'s canonical order. Offsetting past
    // `[0, k_max)` — the ring's own subindex domain — is what keeps a thermal
    // owning both ring slots and commitment windows collision-free on
    // `(entity_type, entity_id, subindex)`; dropping the `k_max` offset would
    // silently alias a commitment window onto a genuine ring slot.
    let commitment_slot = |w: usize| -> EntitySlot {
        let thermal_id = global_layout.commitment_window_thermal_id[w];
        let local_window_index = global_layout.commitment_window_thermal_id[..w]
            .iter()
            .filter(|&&id| id == thermal_id)
            .count();
        let subindex = global_layout.k_max + local_window_index;
        debug_assert!(
            subindex >= global_layout.k_max,
            "commitment-block subindex must land outside the ring's [0, k_max) domain"
        );
        let owning_thermal = anticipated_thermals.iter().find(|t| t.id == thermal_id);
        debug_assert!(
            owning_thermal.is_some(),
            "commitment window {w}'s owning thermal {thermal_id:?} must be an anticipated thermal"
        );
        EntitySlot {
            entity_type: ENTITY_TYPE_ANTICIPATED_THERMAL_STATE,
            entity_id: thermal_id.0,
            subindex: subindex as u32,
            was_active: owning_thermal
                .is_some_and(|t| commissioning_active(t.entry_stage_id, t.exit_stage_id, stage_id)),
            delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        }
    };

    let mut manifest = Vec::with_capacity(projection.n_slots());
    for j in 0..projection.n_slots() {
        let (region, offset) =
            global_layout.classify_incoming_column(projection.incoming_column(CutSlot::new(j)));
        let slot = match region {
            StateRegion::Storage => {
                let hydro = &hydros[offset];
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
                    delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
                }
            }
            StateRegion::Lag => {
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
                    delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
                }
            }
            StateRegion::Buckets => {
                let (plant_idx, lag) = global_layout.transit_bucket_column_order[offset];
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
                    // Water in this bucket arrives `lag` stages after this one.
                    delivery_date: anchor_at(lag),
                }
            }
            StateRegion::Anticipated => anticipated_slot(offset),
            StateRegion::CommitmentBlock => commitment_slot(offset),
        };
        manifest.push(slot);
    }

    debug_assert_eq!(
        manifest.len(),
        projection.n_slots(),
        "manifest length must equal projection.n_slots()"
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
            (0..pool.populated())
                .map(|i| {
                    let meta = pool.metadata(i);
                    PolicyCutRecord {
                        cut_id: meta.iteration_generated * u64::from(pool.visit_stride)
                            + u64::from(meta.forward_pass_index),
                        slot_index: i as u32,
                        iteration: meta.iteration_generated as u32,
                        forward_pass_index: meta.forward_pass_index,
                        intercept: pool.intercept(i),
                        coefficients: pool.coefficient_row(i),
                        is_active: pool.is_active(i),
                    }
                })
                .collect()
        })
        .collect()
}

/// Rescale [`build_stage_cut_records`]'s output from the writing study's
/// internal scaled cost space to canonical currency units at rest: every
/// `coefficients` entry and `intercept`, multiplied by
/// `cost_scale_factor`.
///
/// Returns owned records because [`PolicyCutRecord::coefficients`] borrows —
/// the writing study's [`FutureCostFunction`] pool storage cannot be mutated
/// in place (it stays live for further training/warm-start), so the scaled
/// values need a fresh, owned home. Pair with [`borrow_cut_records`] to
/// rebuild the `PolicyCutRecord` views [`build_stage_cuts_payloads`] needs.
#[must_use]
pub fn scale_cut_records_for_export(
    stage_records: &[Vec<PolicyCutRecord<'_>>],
    cost_scale_factor: f64,
) -> Vec<Vec<OwnedPolicyCutRecord>> {
    stage_records
        .iter()
        .map(|records| {
            records
                .iter()
                .map(|r| OwnedPolicyCutRecord {
                    cut_id: r.cut_id,
                    slot_index: r.slot_index,
                    iteration: r.iteration,
                    forward_pass_index: r.forward_pass_index,
                    intercept: r.intercept * cost_scale_factor,
                    coefficients: r
                        .coefficients
                        .iter()
                        .map(|c| c * cost_scale_factor)
                        .collect(),
                    is_active: r.is_active,
                })
                .collect()
        })
        .collect()
}

/// Rebuild borrowed [`PolicyCutRecord`] views over
/// [`scale_cut_records_for_export`]'s owned output, for
/// [`build_stage_cuts_payloads`] and [`build_active_indices`].
#[must_use]
pub fn borrow_cut_records(owned: &[Vec<OwnedPolicyCutRecord>]) -> Vec<Vec<PolicyCutRecord<'_>>> {
    owned
        .iter()
        .map(|records| {
            records
                .iter()
                .map(|r| PolicyCutRecord {
                    cut_id: r.cut_id,
                    slot_index: r.slot_index,
                    iteration: r.iteration,
                    forward_pass_index: r.forward_pass_index,
                    intercept: r.intercept,
                    coefficients: &r.coefficients,
                    is_active: r.is_active,
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
            populated_count: pool.populated() as u32,
            entity_manifest: &stage_manifests[stage_idx],
        })
        .collect()
}

/// Convert the solver basis cache to u8 byte vectors via `to_discriminant_code`.
///
/// The canonical discriminant space (`0..=6`) is a strict superset of the `HiGHS`
/// code space this format previously stored, so pre-existing checkpoints (bytes
/// `0..=4`) decode identically — while a CLP-captured `Superbasic`/`Fixed`, which
/// `to_highs_code` would fold, now survives reload. Mirrored on load by
/// `build_basis_cache_from_checkpoint`'s `from_discriminant_code`.
///
/// Returns `(col_status_bytes, row_status_bytes)`.
#[must_use]
pub fn convert_basis_cache(training_result: &TrainingResult) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let col = training_result
        .basis_cache
        .iter()
        .map(|opt| {
            opt.as_ref()
                .map(|cb| {
                    cb.basis
                        .col_status
                        .iter()
                        .map(|status| status.to_discriminant_code())
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect();
    let row = training_result
        .basis_cache
        .iter()
        .map(|opt| {
            opt.as_ref()
                .map(|cb| {
                    cb.basis
                        .row_status
                        .iter()
                        .map(|status| status.to_discriminant_code())
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect();
    (col, row)
}

/// Build per-node [`PolicyBasisRecord`] references from pre-converted basis data.
///
/// `training_result.basis_cache` is node-indexed (one slot per `node_graph`
/// position), so the enumeration index is the node ordinal, carried into
/// `stage_id`. `num_cut_rows` counts the node's OWN pool's affine rows —
/// resolved through `node_graph.nodes[node].pool_id`, never `fcf.pools[node]`,
/// which reads the wrong pool once `n_pools != n_nodes` on a branching graph.
#[must_use]
pub fn build_stage_basis_records<'a>(
    fcf: &FutureCostFunction,
    training_result: &TrainingResult,
    node_graph: &NodeGraph,
    basis_col_u8: &'a [Vec<u8>],
    basis_row_u8: &'a [Vec<u8>],
) -> Vec<PolicyBasisRecord<'a>> {
    training_result
        .basis_cache
        .iter()
        .enumerate()
        .filter_map(|(node, opt)| {
            opt.as_ref().map(|_| {
                let pool = node_graph.nodes[NodePos(node)].pool_id;
                let num_cut_rows = fcf
                    .pools
                    .get(pool)
                    .map_or(0, |pool| pool.populated().min(pool.capacity) as u32);
                PolicyBasisRecord {
                    stage_id: node as u32,
                    iteration: training_result.iterations as u32,
                    column_status: &basis_col_u8[node],
                    row_status: &basis_row_u8[node],
                    num_cut_rows,
                }
            })
        })
        .collect()
}

/// Build the value-function artifact's graph manifest from the runtime node
/// graph: one manifest node per canonical position carrying its declared id, its
/// stage id (resolved through `study_stage_ids`), and its pool id, plus the
/// flattened edge list. `pool_id` IS the node → pool map; leaf nodes sharing a
/// pool all name the same `pool_id`.
///
/// Single owner shared by the checkpoint writer and the full-FCF load-path
/// graph-identity check, so the two cannot derive a divergent manifest for the
/// same study.
#[must_use]
pub fn build_graph_manifest(node_graph: &NodeGraph, study_stage_ids: &[i32]) -> GraphManifest {
    let nodes = node_graph
        .nodes
        .iter_indexed()
        .map(|(pos, node)| ManifestNode {
            id: node_graph.node_ids[pos].0,
            stage_id: study_stage_ids.get(node.stage.0).copied().unwrap_or(-1),
            pool_id: node.pool_id as u32,
        })
        .collect();
    let edges = node_graph
        .successors
        .iter_indexed()
        .flat_map(|(pos, succs)| {
            let source_id = node_graph.node_ids[pos].0;
            succs.iter().map(move |succ| ManifestEdge {
                source_id,
                target_id: node_graph.node_ids[succ.child].0,
                probability: succ.probability,
            })
        })
        .collect();
    GraphManifest {
        n_pools: node_graph.n_pools as u32,
        nodes,
        edges,
    }
}

/// Build per-stage [`StageStatesPayload`]s from the visited states archive.
///
/// Returns an empty `Vec` if the archive is `None` (non-Dominated strategies).
/// `archive` is indexed per NODE (one `NodeStates` per `node_graph` position);
/// `stage_manifests` is indexed per POOL, one entry per `fcf.pools` slot —
/// the same array [`build_stage_cuts_payloads`] indexes. A branching graph has
/// strictly more nodes than pools (sibling leaves share one pool), so node
/// position `pos`'s manifest is `stage_manifests[node_graph.nodes[pos].pool_id]`
/// — never `stage_manifests[pos.0]`, which is only safe when `pool_id == pos.0`
/// (pool id is not itself a `NodePos`, so this substitution still type-checks).
///
/// `stage_id` carries the node's STUDY STAGE (`node_graph.nodes[pos].stage`);
/// `node_id` carries the node's own declared id (`node_graph.node_ids[pos]`) —
/// both distinct from the node's position `pos`, so a branching artifact's
/// per-node states are attributable to their stage, their declared id, and
/// their position independently.
#[must_use]
pub fn build_stage_states_payloads<'a>(
    archive: Option<&'a VisitedStatesArchive>,
    stage_manifests: &'a [Vec<EntitySlot>],
    node_graph: &NodeGraph,
) -> Vec<StageStatesPayload<'a>> {
    let Some(archive) = archive else {
        return Vec::new();
    };
    (0..archive.num_nodes())
        .map(|t| {
            let pos = NodePos(t);
            let node = archive.node(pos);
            let pool_id = node_graph.nodes[pos].pool_id;
            StageStatesPayload {
                stage_id: node_graph.nodes[pos].stage.0 as u32,
                node_id: node_graph.node_ids[pos].0,
                state_dimension: node.state_dimension() as u32,
                count: node.count() as u32,
                data: node.states(),
                entity_manifest: &stage_manifests[pool_id],
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
    clippy::cast_sign_loss,
    clippy::unreadable_literal
)]
mod tests {
    use super::{
        ENTITY_TYPE_ANTICIPATED_THERMAL_STATE, ENTITY_TYPE_HYDRO_INFLOW_LAG,
        ENTITY_TYPE_HYDRO_STORAGE, ENTITY_TYPE_HYDRO_TRANSIT_BUCKET, EntitySlot,
        build_stage_entity_manifest, build_stage_states_payloads,
    };
    use crate::indexer::{CutStateProjection, StateSpace};
    use crate::lead_time::{AnticipatedResolution, LeadTime};
    use crate::setup::{
        NodeGraph, NodeId, NodeOpenings, NodePos, NodeRuntime, NodeSuccessor, OpeningSource,
        StageIdx,
    };
    use crate::test_support;
    use crate::visited_states::VisitedStatesArchive;
    use cobre_core::commissioning::hydro_operating_active;
    use cobre_core::temporal::StageStateConfig;
    use cobre_core::{
        AnticipatedConfig, Block, BlockMode, Bus, DeficitSegment, EntityId, Hydro,
        HydroGenerationModel, HydroPenalties, NoiseMethod, ScenarioSourceConfig, Stage,
        StageRiskConfig, System, SystemBuilder, Thermal,
        resolved::{
            BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, HydroBlockBounds,
            HydroStageBounds, LineBlockBounds, PumpingBlockBounds, ResolvedBounds,
            ThermalBlockBounds, ThermalStageBounds,
        },
    };
    use cobre_io::ENTITY_SLOT_DELIVERY_DATE_SENTINEL;

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

    fn bounds_defaults() -> BoundsDefaults {
        BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 100.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 50.0,
                max_generation_mw: 45.0,
                ..Default::default()
            },
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
            },
            line_block: LineBlockBounds {
                direct_mw: 500.0,
                reverse_mw: 500.0,
            },
            pumping_block: PumpingBlockBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract_block: ContractBlockBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        }
    }

    fn make_hydro(id: i32, entry: Option<i32>, exit: Option<i32>) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("Hydro{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
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
        };
        hydro.declare_mirror_unit_group(EntityId(1));
        hydro
    }

    fn anticipated_thermal(id: i32, lead_stages: u32) -> Thermal {
        anticipated_thermal_cfg(id, AnticipatedConfig::LeadStages(lead_stages))
    }

    fn anticipated_thermal_cfg(id: i32, cfg: AnticipatedConfig) -> Thermal {
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
            anticipated_config: Some(cfg),
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
            &bounds_defaults(),
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
    fn layout_2h_1ant() -> StateSpace {
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

        assert_eq!(manifest.len(), projection.n_slots());
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

        assert_eq!(manifest.len(), projection.n_slots());
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

        assert_eq!(manifest.len(), projection.n_slots());
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

    /// A monthly stage (`index`/`id` shared) starting on the first of `year`/`month`.
    fn make_stage_ym(index: usize, id: i32, year: i32, month: u32) -> Stage {
        let start = chrono::NaiveDate::from_ymd_opt(year, month, 1).unwrap();
        let (end_year, end_month) = if month == 12 {
            (year + 1, 1)
        } else {
            (year, month + 1)
        };
        Stage {
            index,
            id,
            start_date: start,
            end_date: chrono::NaiveDate::from_ymd_opt(end_year, end_month, 1).unwrap(),
            season_id: Some((month - 1) as usize),
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

    /// `System` with 1 hydro and 1 anticipated thermal carrying `cfg` over three
    /// consecutive monthly stages 2024-04, 2024-05, 2024-06 (ids 0, 1, 2).
    fn system_1h_1ant_3monthly(cfg: AnticipatedConfig) -> System {
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 3,
                k_max: 2,
            },
            &bounds_defaults(),
        );
        SystemBuilder::new()
            .buses(vec![make_bus()])
            .hydros(vec![make_hydro(1, None, None)])
            .thermals(vec![anticipated_thermal_cfg(1, cfg)])
            .stages(vec![
                make_stage_ym(0, 0, 2024, 4),
                make_stage_ym(1, 1, 2024, 5),
                make_stage_ym(2, 2, 2024, 6),
            ])
            .bounds(bounds)
            .build()
            .expect("valid 3-stage system")
    }

    /// `System` with 1 hydro and two anticipated thermals of DIFFERENT
    /// `LeadStages` (id 1: ℓ=1, id 2: ℓ=2) over the same three monthly stages as
    /// [`system_1h_1ant_3monthly`], sharing one `k_max=2` ring.
    fn system_1h_2ant_3monthly() -> System {
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 3,
                k_max: 2,
            },
            &bounds_defaults(),
        );
        SystemBuilder::new()
            .buses(vec![make_bus()])
            .hydros(vec![make_hydro(1, None, None)])
            .thermals(vec![anticipated_thermal(1, 1), anticipated_thermal(2, 2)])
            .stages(vec![
                make_stage_ym(0, 0, 2024, 4),
                make_stage_ym(1, 1, 2024, 5),
                make_stage_ym(2, 2, 2024, 6),
            ])
            .bounds(bounds)
            .build()
            .expect("valid 3-stage 2-anticipated-plant system")
    }

    /// An `AnticipatedThermalState` ring slot's `delivery_date` is the `YYYYMMDD`
    /// of its delivery stage (`current index + ring slot`), resolved
    /// through the attached `AnticipatedResolution`; storage/lag slots stay at the
    /// sentinel. At stage index 1 (2024-05) with `lead_stages = 2`, ring slot 1
    /// delivers at index 2 (2024-06 → `20240601`).
    #[test]
    fn anticipated_slot_delivery_anchor_matches_delivery_stage_year_month() {
        let system = system_1h_1ant_3monthly(AnticipatedConfig::LeadStages(2));
        let mut global = test_support::state_layout_full(1, 1, 1, 2, vec![2]);
        global.set_anticipated_resolution(AnticipatedResolution::resolve(
            &[LeadTime::Stages(2)],
            &[720.0; 3],
            3,
        ));
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        // stage_id 1 is the middle stage (2024-05), study index 1.
        let manifest = build_stage_entity_manifest(&system, &global, &projection, 1);

        // Layout N=1, L=1, A=1, k_max=2: storage j=0, lag j=1, anticipated j=2,3.
        assert_eq!(manifest.len(), 4);
        assert_eq!(manifest[0].entity_type, ENTITY_TYPE_HYDRO_STORAGE);
        assert_eq!(
            manifest[0].delivery_date, ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
            "storage slot carries no delivery date"
        );
        assert_eq!(manifest[1].entity_type, ENTITY_TYPE_HYDRO_INFLOW_LAG);
        assert_eq!(
            manifest[1].delivery_date, ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
            "inflow-lag slot carries no delivery date"
        );

        assert_eq!(
            manifest[2].entity_type,
            ENTITY_TYPE_ANTICIPATED_THERMAL_STATE
        );
        assert_eq!(manifest[2].subindex, 0);
        assert_eq!(
            manifest[2].delivery_date, 20240501,
            "ring slot 0 delivers at index 1 (2024-05)"
        );

        assert_eq!(
            manifest[3].entity_type,
            ENTITY_TYPE_ANTICIPATED_THERMAL_STATE
        );
        assert_eq!(manifest[3].subindex, 1);
        assert_eq!(
            manifest[3].delivery_date, 20240601,
            "ring slot 1 delivers at index 2 (2024-06)"
        );
    }

    /// A ring slot whose delivery stage lands past the horizon reads the
    /// sentinel: at the last stage (index 2), ring slot 1 would deliver at index
    /// 3, which does not exist, so its date is the sentinel.
    #[test]
    fn anticipated_slot_delivery_anchor_past_horizon_is_sentinel() {
        let system = system_1h_1ant_3monthly(AnticipatedConfig::LeadStages(2));
        let mut global = test_support::state_layout_full(1, 1, 1, 2, vec![2]);
        global.set_anticipated_resolution(AnticipatedResolution::resolve(
            &[LeadTime::Stages(2)],
            &[720.0; 3],
            3,
        ));
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        // stage_id 2 is the terminal stage (2024-06), study index 2.
        let manifest = build_stage_entity_manifest(&system, &global, &projection, 2);

        // Ring slot 0 delivers at index 2 (2024-06); ring slot 1 at index 3 (gone).
        assert_eq!(manifest[2].subindex, 0);
        assert_eq!(manifest[2].delivery_date, 20240601);
        assert_eq!(manifest[3].subindex, 1);
        assert_eq!(
            manifest[3].delivery_date, ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
            "delivery past the horizon reads the sentinel"
        );
    }

    /// A `LeadTime`-mode anticipated plant yields a real anchor on every
    /// reachable ring slot, exactly like `LeadStages` (compare
    /// `anticipated_slot_delivery_anchor_matches_delivery_stage_year_month`):
    /// lead mode no longer gates the anchor, only reachability
    /// (`anticipated_lead_stages`) and horizon truncation do.
    #[test]
    fn anticipated_slot_leadtime_mode_yields_real_anchor() {
        let system = system_1h_1ant_3monthly(AnticipatedConfig::LeadTime(720.0));
        let mut global = test_support::state_layout_full(1, 1, 1, 2, vec![2]);
        global.set_anticipated_resolution(AnticipatedResolution::resolve(
            &[LeadTime::Time(720.0)],
            &[720.0; 3],
            3,
        ));
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        let manifest = build_stage_entity_manifest(&system, &global, &projection, 1);

        assert_eq!(manifest[2].subindex, 0);
        assert_eq!(
            manifest[2].delivery_date, 20240501,
            "ring slot 0 delivers at index 1 (2024-05)"
        );
        assert_eq!(manifest[3].subindex, 1);
        assert_eq!(
            manifest[3].delivery_date, 20240601,
            "ring slot 1 delivers at index 2 (2024-06)"
        );
    }

    /// A slot beyond a plant's OWN `anticipated_lead_stages` is structural
    /// padding (frozen `[0, 0]`), so it reads the sentinel even when
    /// `t + slot_idx` itself still lands inside the horizon: with plants
    /// ℓ=1 and ℓ=2 sharing one `k_max=2` ring, the ℓ=1 plant's slot 1 is
    /// padding at every stage, while the ℓ=2 plant's slot 1 is reachable.
    #[test]
    fn anticipated_slot_padding_beyond_own_lead_is_sentinel() {
        let system = system_1h_2ant_3monthly();
        let mut global = test_support::state_layout_full(1, 1, 2, 2, vec![1, 2]);
        global.set_anticipated_resolution(AnticipatedResolution::resolve(
            &[LeadTime::Stages(1), LeadTime::Stages(2)],
            &[720.0; 3],
            3,
        ));
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        // Stage index 0 (2024-04): plenty of horizon left for both plants.
        let manifest = build_stage_entity_manifest(&system, &global, &projection, 0);

        // Layout N=1, L=1, A=2, k_max=2: storage j=0, lag j=1, anticipated j=2..6
        // (slot-major, plant-minor: [slot0,plant0][slot0,plant1][slot1,plant0][slot1,plant1]).
        assert_eq!(manifest[2].entity_id, 1, "slot0/plant0 = ℓ=1 plant");
        assert_eq!(manifest[2].subindex, 0);
        assert_eq!(
            manifest[2].delivery_date, 20240401,
            "ℓ=1 plant slot 0 delivers at index 0 (2024-04)"
        );

        assert_eq!(manifest[3].entity_id, 2, "slot0/plant1 = ℓ=2 plant");
        assert_eq!(manifest[3].subindex, 0);
        assert_eq!(
            manifest[3].delivery_date, 20240401,
            "ℓ=2 plant slot 0 delivers at index 0 (2024-04)"
        );

        assert_eq!(manifest[4].entity_id, 1, "slot1/plant0 = ℓ=1 plant");
        assert_eq!(manifest[4].subindex, 1);
        assert_eq!(
            manifest[4].delivery_date, ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
            "ℓ=1 plant slot 1 is structural padding beyond its own lead"
        );

        assert_eq!(manifest[5].entity_id, 2, "slot1/plant1 = ℓ=2 plant");
        assert_eq!(manifest[5].subindex, 1);
        assert_eq!(
            manifest[5].delivery_date, 20240501,
            "ℓ=2 plant slot 1 delivers at index 1 (2024-05)"
        );
    }

    // -- Terminal commitment block: real manifest arm --

    /// A declared commitment block joins the manifest with a real
    /// `EntitySlot`: reused `AnticipatedThermalState` type, the owning
    /// thermal's id, and the sentinel delivery date (real dates are a
    /// separate concern this manifest does not resolve).
    #[test]
    fn commitment_block_slot_carries_real_identity_and_sentinel_date() {
        let system = system_2h_1ant((None, None), (None, None));
        let global = StateSpace::new_with_commitment_block(
            2,
            2,
            0,
            Vec::new(),
            1,
            2,
            vec![2],
            &[2, 2],
            1,
            vec![0],
            vec![EntityId(1)],
        );
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        let manifest = build_stage_entity_manifest(&system, &global, &projection, 0);

        // Layout N=2, L=2, A=1, k_max=2, W=1: storage 0..2, lag 2..6,
        // anticipated 6..8, commitment block at 8.
        assert_eq!(manifest.len(), 9);
        let slot = &manifest[8];
        assert_eq!(slot.entity_type, ENTITY_TYPE_ANTICIPATED_THERMAL_STATE);
        assert_eq!(slot.entity_id, 1, "commitment window owned by thermal 1");
        assert_eq!(
            slot.subindex, 2,
            "k_max (2) + local_window_index (0), the sole window for this thermal"
        );
        assert_eq!(slot.delivery_date, ENTITY_SLOT_DELIVERY_DATE_SENTINEL);
    }

    /// User-ratified subindex formula (`k_max + local_window_index`): a
    /// thermal owning BOTH ring slots (`[0, k_max)`) and commitment-block
    /// windows gets fully disjoint subindices on `(entity_type, entity_id)`
    /// — the collision the `k_max` offset exists to forbid. Two windows for
    /// the SAME thermal also get distinct subindices from each other.
    #[test]
    fn commitment_block_subindex_never_collides_with_ring_subindex() {
        let system = system_2h_1ant((None, None), (None, None));
        let global = StateSpace::new_with_commitment_block(
            2,
            2,
            0,
            Vec::new(),
            1,
            2,
            vec![2],
            &[2, 2],
            2,
            vec![0, 0],
            vec![EntityId(1), EntityId(1)],
        );
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        let manifest = build_stage_entity_manifest(&system, &global, &projection, 0);

        let same_family_subindices: Vec<u32> = manifest
            .iter()
            .filter(|s| s.entity_type == ENTITY_TYPE_ANTICIPATED_THERMAL_STATE && s.entity_id == 1)
            .map(|s| s.subindex)
            .collect();
        assert_eq!(
            same_family_subindices.len(),
            4,
            "thermal 1 must own 2 ring slots + 2 commitment windows"
        );

        let distinct: std::collections::BTreeSet<u32> =
            same_family_subindices.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            4,
            "every slot sharing (entity_type, entity_id) must carry a distinct subindex; \
             got {same_family_subindices:?}"
        );
        assert_eq!(
            distinct,
            std::collections::BTreeSet::from([0, 1, 2, 3]),
            "ring occupies [0, k_max), commitment windows occupy k_max.."
        );
    }

    // -- build_stage_states_payloads: node-vs-pool manifest indexing --

    /// A 7-node binary tree mirroring `setup::tests::
    /// node_native_binary_tree_loads_and_constructs_node_graph`'s fixture:
    /// nodes 0/1/2 are internal and each own their own pool (0/1/2); leaves
    /// 3/4/5/6 share pool 3. `n_pools == 4 < nodes.len() == 7`.
    fn binary_tree_node_graph() -> NodeGraph {
        let opening = NodeOpenings {
            source: OpeningSource::Generated,
            offset: 0,
            len: 1,
            q: 1.0,
        };
        let node = |stage: usize, pool_id: usize| NodeRuntime {
            stage: StageIdx(stage),
            pool_id,
            openings: opening,
        };
        let edge = |child: usize| NodeSuccessor {
            child: NodePos(child),
            probability: 0.5,
        };
        NodeGraph {
            node_ids: vec![
                NodeId(0),
                NodeId(1),
                NodeId(2),
                NodeId(3),
                NodeId(4),
                NodeId(5),
                NodeId(6),
            ]
            .into(),
            nodes: vec![
                node(0, 0),
                node(1, 1),
                node(1, 2),
                node(2, 3),
                node(2, 3),
                node(2, 3),
                node(2, 3),
            ]
            .into(),
            successors: vec![
                vec![edge(1), edge(2)],
                vec![edge(3), edge(4)],
                vec![edge(5), edge(6)],
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ]
            .into(),
            n_pools: 4,
            pool_stage: vec![StageIdx(0), StageIdx(1), StageIdx(1), StageIdx(2)],
        }
    }

    /// A one-slot manifest tagging `pool_id` into `entity_id`, so a payload's
    /// manifest can be traced back to the pool it came from.
    fn manifest_for_pool(pool_id: usize) -> Vec<EntitySlot> {
        vec![EntitySlot {
            entity_type: ENTITY_TYPE_HYDRO_STORAGE,
            entity_id: 100 + i32::try_from(pool_id).unwrap(),
            subindex: 0,
            was_active: true,
            delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        }]
    }

    /// On a branching graph, `archive.num_nodes() (7) > stage_manifests.len()
    /// (4)`: indexing the manifest directly by node position `t` is wrong — it
    /// panics out of bounds the moment `t >= 4`. Indexing through
    /// `node_graph.nodes[t].pool_id` stays in range and resolves every one of
    /// the 4 leaves (t = 3..=6) to the SAME shared pool-3 manifest.
    ///
    /// Also pins U13: `stage_id` carries `node_graph.nodes[t].stage` (the
    /// binary tree's stage 2 for every leaf, not the leaves' distinct node
    /// positions 3..=6) and `node_id` carries `node_graph.node_ids[t]`.
    #[test]
    fn tree_states_payload_resolves_manifest_through_node_pool_id() {
        let node_graph = binary_tree_node_graph();
        let stage_manifests: Vec<Vec<EntitySlot>> =
            (0..node_graph.n_pools).map(manifest_for_pool).collect();

        let mut archive = VisitedStatesArchive::new(node_graph.nodes.len(), 1, 1, 1);
        for t in 0..node_graph.nodes.len() {
            archive.archive_gathered_states(NodePos(t), &[0.0], 1);
        }

        let payloads = build_stage_states_payloads(Some(&archive), &stage_manifests, &node_graph);

        assert_eq!(payloads.len(), 7, "one payload per node, not per pool");
        for (t, payload) in payloads.iter().enumerate() {
            let pos = NodePos(t);
            let expected_pool = node_graph.nodes[pos].pool_id;
            assert_eq!(
                payload.entity_manifest.len(),
                1,
                "node {t} manifest must not be empty"
            );
            assert_eq!(
                payload.entity_manifest[0].entity_id,
                100 + i32::try_from(expected_pool).unwrap(),
                "node {t} must carry pool {expected_pool}'s manifest, not stage_manifests[{t}]"
            );
            assert_eq!(
                payload.stage_id, node_graph.nodes[pos].stage.0 as u32,
                "node {t}'s stage_id must be its STUDY STAGE, not its node position"
            );
            assert_eq!(
                payload.node_id, node_graph.node_ids[pos].0,
                "node {t}'s node_id must be its declared node id"
            );
        }

        // The 4 leaves all resolve to the one shared pool-3 manifest and the
        // one shared terminal stage, despite carrying 4 distinct node ids.
        for (t, payload) in payloads[3..=6].iter().enumerate() {
            assert_eq!(payload.entity_manifest[0].entity_id, 103);
            assert_eq!(payload.stage_id, 2, "every leaf sits at stage 2");
            assert_eq!(
                payload.node_id,
                i32::try_from(3 + t).unwrap(),
                "leaves keep their own distinct node ids"
            );
        }
    }

    /// On a branching (K-fan) graph, `build_stage_basis_records` is node-indexed
    /// (one record per `basis_cache` node slot, `stage_id == node ordinal`) and
    /// resolves each record's `num_cut_rows` through the node's OWN pool
    /// (`node_graph.nodes[node].pool_id`), never `fcf.pools[node_ordinal]` —
    /// which for leaves 4/5/6 (ordinal > `n_pools - 1`) reads no pool at all.
    /// The 4 leaves all report the shared pool-3 count.
    #[test]
    fn build_stage_basis_records_resolves_num_cut_rows_via_node_pool() {
        use super::{build_stage_basis_records, convert_basis_cache};
        use crate::TrainingResult;
        use crate::cut::FutureCostFunction;
        use crate::workspace::CapturedBasis;
        use cobre_solver::Basis;

        let node_graph = binary_tree_node_graph(); // 7 nodes; pools 0/1/2 + shared 3

        // 4 pools; pool 3 (shared leaf pool) gets 2 cuts, pools 0/1/2 get 1 each.
        let mut fcf = FutureCostFunction::new(4, 1, 1, 10, &[0; 4]);
        fcf.add_cut(NodeId(0), 0, 0, 0, 0.0, &[0.0]);
        fcf.add_cut(NodeId(0), 1, 0, 0, 0.0, &[0.0]);
        fcf.add_cut(NodeId(0), 2, 0, 0, 0.0, &[0.0]);
        fcf.add_cut(NodeId(0), 3, 0, 0, 0.0, &[0.0]);
        fcf.add_cut(NodeId(0), 3, 1, 0, 0.0, &[0.0]);

        let empty_basis = || CapturedBasis {
            basis: Basis {
                col_status: Vec::new(),
                row_status: Vec::new(),
            },
            base_row_count: 0,
            cut_row_slots: Vec::new(),
            state_at_capture: Vec::new(),
            node_id: NodeId(0),
        };
        let basis_cache: Vec<Option<CapturedBasis>> = (0..7).map(|_| Some(empty_basis())).collect();
        let training_result = TrainingResult::new(
            0.0,
            0.0,
            0.0,
            0.0,
            1,
            "t".to_string(),
            0,
            basis_cache,
            Vec::new(),
            None,
            None,
        );

        let (col, row) = convert_basis_cache(&training_result);
        let records = build_stage_basis_records(&fcf, &training_result, &node_graph, &col, &row);

        assert_eq!(records.len(), 7, "one basis record per node, not per pool");
        for (node, rec) in records.iter().enumerate() {
            assert_eq!(
                rec.stage_id as usize, node,
                "stage_id must carry the node ordinal"
            );
            let pool = node_graph.nodes[NodePos(node)].pool_id;
            let expected = if pool == 3 { 2 } else { 1 };
            assert_eq!(
                rec.num_cut_rows, expected,
                "node {node} num_cut_rows must reflect its own pool {pool}, not fcf.pools[{node}]"
            );
        }
    }
}
