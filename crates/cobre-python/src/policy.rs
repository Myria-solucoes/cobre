//! `cobre.write_policy_checkpoint` — writes a policy checkpoint from plain
//! Python dicts/sequences, single-sourcing the `FlatBuffers` byte layout in
//! `cobre_io`.
//!
//! Input dict shapes mirror what [`crate::results::load_policy`] emits, so a
//! loaded checkpoint round-trips: load -> edit -> write.

use std::collections::HashMap;
use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use cobre_io::{
    ENTITY_SLOT_DELIVERY_DATE_SENTINEL, EntitySlot, FORMAT_VERSION, GraphManifest, ManifestEdge,
    ManifestNode, PolicyBasisRecord, PolicyCheckpointMetadata, PolicyCutRecord, ProducerBlock,
    STAGE_STATES_NODE_ID_SENTINEL, StageCutsPayload, StageStatesPayload,
};
use cobre_sddp::{SddpError, reserve_boundary_inflow_lag_slots};

use crate::errors::{ErrorSource, convert_error};

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyEntitySlot {
    entity_type: u8,
    entity_id: i32,
    subindex: u32,
    was_active: bool,
    #[pyo3(default = ENTITY_SLOT_DELIVERY_DATE_SENTINEL)]
    delivery_date: i32,
}

impl From<&PyEntitySlot> for EntitySlot {
    fn from(slot: &PyEntitySlot) -> Self {
        Self {
            entity_type: slot.entity_type,
            entity_id: slot.entity_id,
            subindex: slot.subindex,
            was_active: slot.was_active,
            delivery_date: slot.delivery_date,
        }
    }
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyCutRecord {
    cut_id: u64,
    slot_index: u32,
    iteration: u32,
    forward_pass_index: u32,
    intercept: f64,
    coefficients: Vec<f64>,
    is_active: bool,
    /// Inflow-lag gradient terms keyed by hydro id (`hydro_id -> [coef_by_depth]`,
    /// index `0` = lag depth 1), separate from the storage-aligned
    /// `coefficients`. Consumed only when the top-level `inflow_lag_depth` is
    /// set; empty (the default) leaves the checkpoint byte-identical.
    #[pyo3(default)]
    inflow_lag_coefficients: HashMap<i32, Vec<f64>>,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyStageCutsPayload {
    stage_id: u32,
    state_dimension: u32,
    capacity: u32,
    #[pyo3(default)]
    warm_start_count: u32,
    cuts: Vec<PyCutRecord>,
    #[pyo3(default)]
    active_cut_indices: Vec<u32>,
    #[pyo3(default)]
    populated_count: Option<u32>,
    #[pyo3(default)]
    entity_manifest: Vec<PyEntitySlot>,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyBasisRecord {
    stage_id: u32,
    iteration: u32,
    column_status: Vec<u8>,
    row_status: Vec<u8>,
    num_cut_rows: u32,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyStageStatesPayload {
    stage_id: u32,
    #[pyo3(default = STAGE_STATES_NODE_ID_SENTINEL)]
    node_id: i32,
    state_dimension: u32,
    count: u32,
    data: Vec<f64>,
    #[pyo3(default)]
    entity_manifest: Vec<PyEntitySlot>,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyManifestNode {
    id: i32,
    stage_id: i32,
    pool_id: u32,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyManifestEdge {
    source_id: i32,
    target_id: i32,
    probability: f64,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyGraphManifest {
    n_pools: u32,
    nodes: Vec<PyManifestNode>,
    edges: Vec<PyManifestEdge>,
}

impl From<PyGraphManifest> for GraphManifest {
    fn from(g: PyGraphManifest) -> Self {
        Self {
            n_pools: g.n_pools,
            nodes: g
                .nodes
                .into_iter()
                .map(|n| ManifestNode {
                    id: n.id,
                    stage_id: n.stage_id,
                    pool_id: n.pool_id,
                })
                .collect(),
            edges: g
                .edges
                .into_iter()
                .map(|e| ManifestEdge {
                    source_id: e.source_id,
                    target_id: e.target_id,
                    probability: e.probability,
                })
                .collect(),
        }
    }
}

/// The producer-namespaced metadata block, mirroring what
/// [`crate::results::load_policy`] emits under `metadata["producer"]`.
#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyProducerBlock {
    completed_iterations: u32,
    final_lower_bound: f64,
    #[pyo3(default)]
    best_upper_bound: Option<f64>,
    max_iterations: u32,
    forward_passes: u32,
    warm_start_cuts: u32,
    #[pyo3(default)]
    warm_start_counts: Vec<u32>,
    rng_seed: u64,
    #[pyo3(default)]
    total_visited_states: u64,
    #[pyo3(default)]
    training_block_mode: String,
    #[pyo3(default)]
    training_block_mode_per_stage: Vec<String>,
    #[pyo3(default)]
    cost_scale_factor: Option<f64>,
}

impl From<PyProducerBlock> for ProducerBlock {
    fn from(p: PyProducerBlock) -> Self {
        Self {
            completed_iterations: p.completed_iterations,
            final_lower_bound: p.final_lower_bound,
            best_upper_bound: p.best_upper_bound,
            max_iterations: p.max_iterations,
            forward_passes: p.forward_passes,
            warm_start_cuts: p.warm_start_cuts,
            warm_start_counts: p.warm_start_counts,
            rng_seed: p.rng_seed,
            total_visited_states: p.total_visited_states,
            training_block_mode: p.training_block_mode,
            training_block_mode_per_stage: p.training_block_mode_per_stage,
            cost_scale_factor: p.cost_scale_factor,
        }
    }
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyPolicyCheckpointMetadata {
    /// Stamped to [`FORMAT_VERSION`] when omitted, so a checkpoint authored from
    /// raw records is always readable; a round-tripped one carries it back.
    #[pyo3(default = FORMAT_VERSION)]
    format_version: u32,
    cobre_version: String,
    created_at: String,
    num_stages: u32,
    #[pyo3(default)]
    graph_manifest: Option<PyGraphManifest>,
    producer: PyProducerBlock,
}

impl From<PyPolicyCheckpointMetadata> for PolicyCheckpointMetadata {
    fn from(m: PyPolicyCheckpointMetadata) -> Self {
        Self {
            format_version: m.format_version,
            cobre_version: m.cobre_version,
            created_at: m.created_at,
            num_stages: m.num_stages,
            graph_manifest: m
                .graph_manifest
                .map(GraphManifest::from)
                .unwrap_or_default(),
            producer: m.producer.into(),
        }
    }
}

/// Convert a length to `u32`, naming `what` in the overflow error.
fn checked_u32_len(len: usize, what: &str) -> PyResult<u32> {
    u32::try_from(len)
        .map_err(|_| PyValueError::new_err(format!("{what} has {len} entries, exceeding u32::MAX")))
}

/// Validate each stage's cut coefficient lengths against its `state_dimension`
/// and resolve the `populated_count` default (`cuts.len()`), naming the
/// offending stage/cut on failure.
fn resolve_populated_counts(stage_cuts: &[PyStageCutsPayload]) -> PyResult<Vec<u32>> {
    let mut resolved = Vec::with_capacity(stage_cuts.len());
    for sc in stage_cuts {
        for cut in &sc.cuts {
            if cut.coefficients.len() != sc.state_dimension as usize {
                return Err(PyValueError::new_err(format!(
                    "stage {} cut {}: coefficients has {} entries, expected \
                     state_dimension={}",
                    sc.stage_id,
                    cut.cut_id,
                    cut.coefficients.len(),
                    sc.state_dimension
                )));
            }
        }
        resolved.push(match sc.populated_count {
            Some(p) => p,
            None => checked_u32_len(sc.cuts.len(), "cuts")?,
        });
    }
    Ok(resolved)
}

/// Validate each stage's flat state-data length against `count * state_dimension`,
/// naming the offending stage on failure.
fn validate_stage_states(stage_states: &[PyStageStatesPayload]) -> PyResult<()> {
    for ss in stage_states {
        let expected = ss.count as usize * ss.state_dimension as usize;
        if ss.data.len() != expected {
            return Err(PyValueError::new_err(format!(
                "stage {}: states data has {} entries, expected count*state_dimension={}",
                ss.stage_id,
                ss.data.len(),
                expected
            )));
        }
    }
    Ok(())
}

/// Owned per-stage payload data: the entity manifest and each cut's coefficient
/// vector, either as supplied or widened by an inflow-lag reservation, plus the
/// resulting `state_dimension`. Owning these lets the borrowed
/// [`StageCutsPayload`]/[`PolicyCutRecord`] views reference the reserved
/// coefficients (which the writer must own — the caller's `coefficients` are
/// storage-only when a reservation runs).
struct StageCutsData {
    manifest: Vec<EntitySlot>,
    coefficients: Vec<Vec<f64>>,
    state_dimension: u32,
}

/// Convert one stage's manifest and coefficients to owned form, reserving the
/// canonical `HydroInflowLag` slots when `reserve_depth` is set.
fn build_stage_cuts_data(
    sc: &PyStageCutsPayload,
    reserve_depth: Option<u32>,
) -> PyResult<StageCutsData> {
    let manifest: Vec<EntitySlot> = sc.entity_manifest.iter().map(EntitySlot::from).collect();
    let coefficients: Vec<Vec<f64>> = sc.cuts.iter().map(|c| c.coefficients.clone()).collect();
    match reserve_depth {
        Some(depth) => {
            let cut_lag: Vec<HashMap<i32, Vec<f64>>> = sc
                .cuts
                .iter()
                .map(|c| c.inflow_lag_coefficients.clone())
                .collect();
            let reserved =
                reserve_boundary_inflow_lag_slots(&manifest, &coefficients, &cut_lag, depth)
                    .map_err(|e| match e {
                        SddpError::Validation(msg) => PyValueError::new_err(msg),
                        other => PyValueError::new_err(other.to_string()),
                    })?;
            Ok(StageCutsData {
                manifest: reserved.manifest,
                coefficients: reserved.coefficients,
                state_dimension: reserved.state_dimension,
            })
        }
        None => Ok(StageCutsData {
            manifest,
            coefficients,
            state_dimension: sc.state_dimension,
        }),
    }
}

/// Write a policy checkpoint from plain Python dicts/sequences to `path`.
///
/// `stage_cuts` and `metadata` mirror the `"stage_cuts"`/`"metadata"` keys
/// [`crate::results::load_policy`] returns; `stage_bases`/`stage_states`
/// default to empty when omitted (a checkpoint authored from raw cut data
/// carries neither).
///
/// `inflow_lag_depth`, when set to `N > 0`, has cobre reserve `N` canonical
/// `HydroInflowLag` state slots per storage hydro in every stage's manifest
/// (via [`reserve_boundary_inflow_lag_slots`]), placing each cut's
/// `inflow_lag_coefficients` at their `(hydro, depth)` positions. This is the
/// authoring path for a boundary policy of a case with no PAR model to infer
/// the depth from (a DECOMP-bridge bootstrap). Absent or `0`, the checkpoint is
/// byte-identical to one written without the argument.
///
/// # Errors
///
/// `ValueError` when a cut's `coefficients` length does not match its stage's
/// `state_dimension`, a stage's state data length does not match
/// `count * state_dimension`, or (under `inflow_lag_depth`) a manifest lacks a
/// leading storage block or an inflow-lag coefficient is unplaceable; otherwise
/// the `cobre.errors` leaf mapped from the underlying [`cobre_io::OutputError`].
#[pyfunction]
#[pyo3(signature = (path, stage_cuts, metadata, stage_bases=None, stage_states=None, inflow_lag_depth=None))]
#[allow(clippy::needless_pass_by_value)]
pub fn write_policy_checkpoint(
    py: Python<'_>,
    path: PathBuf,
    stage_cuts: Vec<PyStageCutsPayload>,
    metadata: PyPolicyCheckpointMetadata,
    stage_bases: Option<Vec<PyBasisRecord>>,
    stage_states: Option<Vec<PyStageStatesPayload>>,
    inflow_lag_depth: Option<u32>,
) -> PyResult<()> {
    let stage_bases = stage_bases.unwrap_or_default();
    let stage_states = stage_states.unwrap_or_default();

    let populated_counts = resolve_populated_counts(&stage_cuts)?;
    validate_stage_states(&stage_states)?;

    let reserve_depth = inflow_lag_depth.filter(|&n| n > 0);
    let stage_data: Vec<StageCutsData> = stage_cuts
        .iter()
        .map(|sc| build_stage_cuts_data(sc, reserve_depth))
        .collect::<PyResult<_>>()?;

    let metadata: PolicyCheckpointMetadata = metadata.into();

    py.detach(|| {
        let cut_records: Vec<Vec<PolicyCutRecord<'_>>> = stage_cuts
            .iter()
            .zip(&stage_data)
            .map(|(sc, data)| {
                sc.cuts
                    .iter()
                    .zip(&data.coefficients)
                    .map(|(c, coefficients)| PolicyCutRecord {
                        cut_id: c.cut_id,
                        slot_index: c.slot_index,
                        iteration: c.iteration,
                        forward_pass_index: c.forward_pass_index,
                        intercept: c.intercept,
                        coefficients,
                        is_active: c.is_active,
                    })
                    .collect()
            })
            .collect();

        let stage_cuts_payloads: Vec<StageCutsPayload<'_>> = stage_cuts
            .iter()
            .zip(&stage_data)
            .enumerate()
            .map(|(i, (sc, data))| StageCutsPayload {
                stage_id: sc.stage_id,
                state_dimension: data.state_dimension,
                capacity: sc.capacity,
                warm_start_count: sc.warm_start_count,
                cuts: &cut_records[i],
                active_cut_indices: &sc.active_cut_indices,
                populated_count: populated_counts[i],
                entity_manifest: &data.manifest,
            })
            .collect();

        let basis_records: Vec<PolicyBasisRecord<'_>> = stage_bases
            .iter()
            .map(|b| PolicyBasisRecord {
                stage_id: b.stage_id,
                iteration: b.iteration,
                column_status: &b.column_status,
                row_status: &b.row_status,
                num_cut_rows: b.num_cut_rows,
            })
            .collect();

        let state_manifests: Vec<Vec<EntitySlot>> = stage_states
            .iter()
            .map(|ss| ss.entity_manifest.iter().map(EntitySlot::from).collect())
            .collect();

        let state_payloads: Vec<StageStatesPayload<'_>> = stage_states
            .iter()
            .enumerate()
            .map(|(i, ss)| StageStatesPayload {
                stage_id: ss.stage_id,
                node_id: ss.node_id,
                state_dimension: ss.state_dimension,
                count: ss.count,
                data: &ss.data,
                entity_manifest: &state_manifests[i],
            })
            .collect();

        cobre_io::write_policy_checkpoint(
            &path,
            &stage_cuts_payloads,
            &basis_records,
            &metadata,
            &state_payloads,
        )
    })
    .map_err(|e| convert_error(ErrorSource::Output(&e)))
}
