//! Integration coverage for the embedded per-slot entity manifest written into
//! `policy/cuts/stage_NNN.bin` by the shared `write_checkpoint`.
//!
//! Trains a deterministic case to a policy checkpoint through the same
//! `write_checkpoint` both front ends call, then reads the cut files back and
//! asserts the manifest classification, identity, and per-stage length — including
//! the reduced-stage (`inflow_lags: false`) d43 case, whose pool drops its
//! inflow-lag slots.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::path::Path;

use cobre_core::scenario::ScenarioSource;
use cobre_sddp::{
    StudySetup,
    hydro_models::prepare_hydro_models,
    orchestration::{CheckpointParams, write_checkpoint},
    setup::prepare_stochastic,
};
use cobre_solver::ActiveSolver;

mod common;
use common::StubComm;

/// `EntityType::HydroInflowLag` discriminant from `schemas/policy.fbs`.
const ENTITY_TYPE_HYDRO_INFLOW_LAG: u8 = 1;
/// `EntityType::AnticipatedThermalState` discriminant from `schemas/policy.fbs`.
const ENTITY_TYPE_ANTICIPATED_THERMAL_STATE: u8 = 2;

fn case_dir(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/deterministic")
        .join(name)
}

/// Train a case to a policy checkpoint via the shared `write_checkpoint`, then read
/// it back. Returns `(checkpoint, per-pool cut_state_layout n_state)`.
fn train_and_read_checkpoint(name: &str) -> (cobre_io::PolicyCheckpoint, Vec<usize>) {
    let dir = case_dir(name);
    let config = cobre_io::parse_config(&dir.join("config.json")).expect("config must parse");
    let system = cobre_io::load_case(&dir).expect("load_case must succeed");

    let pr = prepare_stochastic(system, &dir, &config, 42, &ScenarioSource::default())
        .expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, &dir, false).expect("prepare_hydro_models must succeed");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");
    let result = outcome.result;

    let _training_output = setup.build_training_output(&result, &[]);

    // Each pool is sized to its stage's cut-state dimension at construction
    // (`pool_state_dimensions[t] == cut_state_layouts[t].n_state()`), so the pool's
    // own `state_dimension` is the authoritative per-stage manifest length.
    let pool_n_state: Vec<usize> = setup.fcf.pools.iter().map(|p| p.state_dimension).collect();

    let tmp = tempfile::tempdir().expect("tempdir must succeed");
    let policy_dir = tmp.path().join("policy");
    write_checkpoint(
        &policy_dir,
        &setup,
        &system,
        &result,
        &CheckpointParams {
            max_iterations: 100,
            forward_passes: 1,
            seed: 42,
            export_states: false,
        },
    )
    .expect("write_checkpoint must succeed");

    let checkpoint =
        cobre_io::read_policy_checkpoint(&policy_dir).expect("read_policy_checkpoint must succeed");
    (checkpoint, pool_n_state)
}

/// All-enabled study (d03, two-hydro cascade, storage state only): every stage's
/// manifest length equals that pool's cut-state dimension, and the storage slots
/// carry `entity_type == 0` with the hydro ids in `system.hydros()` order and
/// `subindex == 0`.
#[test]
fn all_enabled_manifest_storage_slots_carry_hydro_identity() {
    let (checkpoint, pool_n_state) = train_and_read_checkpoint("d03-two-hydro-cascade");

    assert!(
        !checkpoint.stage_cuts.is_empty(),
        "checkpoint must contain stage cut files"
    );
    for stage in &checkpoint.stage_cuts {
        let t = stage.stage_id as usize;
        assert_eq!(
            stage.entity_manifest.len(),
            pool_n_state[t],
            "stage {t} manifest length must equal the pool cut-state dimension"
        );
        // d03 is storage-only state (no inflow lags, no anticipated): every slot is
        // a storage slot, in system.hydros() order, subindex 0.
        for (i, slot) in stage.entity_manifest.iter().enumerate() {
            assert_eq!(
                slot.entity_type, 0,
                "stage {t} slot {i} must be HydroStorage"
            );
            assert_eq!(slot.subindex, 0, "stage {t} slot {i} storage subindex 0");
        }
        // The two cascade hydros have ids 0 and 1, in `system.hydros()` order.
        assert_eq!(stage.entity_manifest[0].entity_id, 0);
        assert_eq!(stage.entity_manifest[1].entity_id, 1);
    }
}

/// d43 reduced stage: the pool sized by a stage with `inflow_lags: false` produces
/// a manifest whose length equals that pool's reduced cut-state dimension and which
/// carries NO `HydroInflowLag` (type 1) slot, while the full-state pools do carry
/// lag slots. The reduced pool is located empirically by its `cut_state_layouts`
/// dimension (the pool-sizing off-by-one means the reduced projection is not at the
/// `inflow_lags: false` stage's own index).
#[test]
fn d43_reduced_stage_manifest_drops_inflow_lag_slots() {
    let (checkpoint, pool_n_state) = train_and_read_checkpoint("d43-storage-only-cut");

    // Per-stage manifest length always equals the pool's cut-state dimension.
    for stage in &checkpoint.stage_cuts {
        let t = stage.stage_id as usize;
        assert_eq!(
            stage.entity_manifest.len(),
            pool_n_state[t],
            "stage {t} manifest length must equal cut_state_layouts[{t}].n_state()"
        );
    }

    // d43 has one hydro; the full-state pools carry one storage + lag slots, the
    // reduced pool carries one storage slot only. So the reduced pool is the one
    // with the strictly smallest cut-state dimension, and exactly one pool is
    // reduced.
    let min_dim = *pool_n_state.iter().min().expect("at least one pool");
    let max_dim = *pool_n_state.iter().max().expect("at least one pool");
    assert!(
        min_dim < max_dim,
        "d43 must have a reduced pool (min {min_dim}) and a full pool (max {max_dim})"
    );

    let reduced_stage = checkpoint
        .stage_cuts
        .iter()
        .find(|s| s.entity_manifest.len() == min_dim)
        .expect("a reduced-dimension stage must exist");
    assert!(
        reduced_stage
            .entity_manifest
            .iter()
            .all(|s| s.entity_type != ENTITY_TYPE_HYDRO_INFLOW_LAG),
        "the reduced stage manifest must contain no HydroInflowLag slot"
    );

    // A full-state stage must carry at least one inflow-lag slot (d43 fits a PAR
    // model, so the full pools have lag dimensions).
    let full_stage = checkpoint
        .stage_cuts
        .iter()
        .find(|s| s.entity_manifest.len() == max_dim)
        .expect("a full-dimension stage must exist");
    assert!(
        full_stage
            .entity_manifest
            .iter()
            .any(|s| s.entity_type == ENTITY_TYPE_HYDRO_INFLOW_LAG),
        "a full-state stage manifest must contain an inflow-lag slot"
    );
}

/// Anticipated K=2 study (d37): the manifest carries `AnticipatedThermalState`
/// (type 2) slots whose `entity_id` is the anticipated plant (id 1) and whose
/// `subindex` ring slots cover `0..k_max` (`{0, 1}` for K=2).
#[test]
fn anticipated_k2_manifest_has_thermal_state_slots() {
    let (checkpoint, _pool_n_state) = train_and_read_checkpoint("d37-anticipated-commissioning");

    let stage0 = checkpoint
        .stage_cuts
        .iter()
        .find(|s| s.stage_id == 0)
        .expect("stage 0 cut file must exist");

    let anticipated_slots: Vec<_> = stage0
        .entity_manifest
        .iter()
        .filter(|s| s.entity_type == ENTITY_TYPE_ANTICIPATED_THERMAL_STATE)
        .collect();

    assert_eq!(
        anticipated_slots.len(),
        2,
        "K=2 yields two anticipated ring slots on the single anticipated plant"
    );
    for slot in &anticipated_slots {
        assert_eq!(slot.entity_id, 1, "anticipated slot must own plant id 1");
    }
    let ring: Vec<u32> = anticipated_slots.iter().map(|s| s.subindex).collect();
    assert_eq!(ring, vec![0, 1], "ring slots must cover 0..k_max");
}
