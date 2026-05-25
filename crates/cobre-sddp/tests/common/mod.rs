//! Shared test utilities for `cobre-sddp` integration tests.
//!
//! Provides [`build_setup_for_case`], a drop-in replacement for
//! `StudySetup::new` that drives the same construction pipeline as the CLI.
//! The parquet override flows through `hydro_models.productivity_override`,
//! which `prepare_hydro_models` populates from disk.
//!
//! Also exports [`StubComm`], a single-rank communicator stub used in
//! non-MPI integration tests.

#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

use std::path::Path;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_sddp::{StudySetup, setup::StudyParams};

pub mod anticipated_structural_assertions;
pub mod parity_hash;

/// Single-rank communicator stub for testing.
///
/// Implements `Communicator` for single-process test scenarios where no
/// inter-rank communication is needed. All collective operations are no-ops;
/// broadcasts and reductions copy data locally without communication.
pub struct StubComm;

impl Communicator for StubComm {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _counts: &[usize],
        _displs: &[usize],
    ) -> Result<(), CommError> {
        recv[..send.len()].clone_from_slice(send);
        Ok(())
    }

    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _op: ReduceOp,
    ) -> Result<(), CommError> {
        recv.clone_from_slice(send);
        Ok(())
    }

    fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
        Ok(())
    }

    fn barrier(&self) -> Result<(), CommError> {
        Ok(())
    }

    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        1
    }

    fn abort(&self, error_code: i32) -> ! {
        std::process::exit(error_code)
    }
}

/// Build a [`StudySetup`] for a case directory.
///
/// Reads scenario sources from `config`, derives `StudyParams`, and constructs
/// the setup via `StudySetup::from_broadcast_params`. The
/// `hydro_energy_productivity.parquet` override is already folded into
/// `hydro_models.productivity_override` by the caller's
/// `prepare_hydro_models` invocation, so this helper does no parquet I/O.
pub fn build_setup_for_case(
    _case_dir: &Path,
    config: &cobre_io::Config,
    system: &cobre_core::System,
    stochastic: cobre_stochastic::StochasticContext,
    hydro_models: cobre_sddp::PrepareHydroModelsResult,
) -> StudySetup {
    let sentinel = Path::new("config.json");
    let training_source = config
        .training_scenario_source(sentinel)
        .expect("training_scenario_source must parse");
    let simulation_source = config
        .simulation_scenario_source(sentinel)
        .expect("simulation_scenario_source must parse");

    let params = StudyParams::from_config(config).expect("StudyParams::from_config must succeed");
    let construction = params.into_construction_config();

    StudySetup::from_broadcast_params(
        system,
        stochastic,
        construction,
        hydro_models,
        &training_source,
        &simulation_source,
    )
    .expect("StudySetup::from_broadcast_params must build")
}
