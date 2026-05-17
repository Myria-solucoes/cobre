//! Shared test utilities for `cobre-sddp` integration tests.
//!
//! Provides [`build_setup_for_case`], a drop-in replacement for
//! `StudySetup::new` that also loads `system/hydro_energy_productivity.parquet`
//! when the file is present.  This is needed for FPHA cases (D05, D06, D07)
//! where the energy-conversion correctness gate requires an `ρ_eq` source.

#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

use std::path::Path;

use cobre_io::load_hydro_energy_productivity;
use cobre_sddp::{StudySetup, setup::StudyParams};

/// Build a [`StudySetup`] for a case directory, loading
/// `system/hydro_energy_productivity.parquet` when present.
///
/// This replaces direct calls to `StudySetup::new` in integration-test
/// helpers.  The key difference is that it loads `hydro_energy_productivity_rows`
/// from disk (if the file exists) and passes them to
/// `StudySetup::from_broadcast_params` so that FPHA hydros can satisfy the
/// energy-conversion correctness gate without VHA geometry.
pub fn build_setup_for_case(
    case_dir: &Path,
    config: &cobre_io::Config,
    system: &cobre_core::System,
    stochastic: cobre_stochastic::StochasticContext,
    hydro_models: cobre_sddp::hydro_models::PrepareHydroModelsResult,
) -> StudySetup {
    let productivity_path = case_dir
        .join("system")
        .join("hydro_energy_productivity.parquet");
    let productivity_rows =
        load_hydro_energy_productivity(productivity_path.exists().then_some(&productivity_path))
            .expect("hydro_energy_productivity load must succeed");

    let sentinel = Path::new("config.json");
    let training_source = config
        .training_scenario_source(sentinel)
        .expect("training_scenario_source must parse");
    let simulation_source = config
        .simulation_scenario_source(sentinel)
        .expect("simulation_scenario_source must parse");

    let params = StudyParams::from_config(config).expect("StudyParams::from_config must succeed");
    let mut construction = params.into_construction_config();
    construction.hydro_energy_productivity_rows = productivity_rows;

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
