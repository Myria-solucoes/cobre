//! B6a end-to-end parity test: a `>=` generic constraint on the total realized
//! inflow `hydro_inflow(h)` solves on both LP backends, and the B6a keyword is
//! inert for studies that do not use it.
//!
//! ## What this file proves (and what it deliberately does not)
//!
//! `hydro_inflow(h[,blk])` resolves to the **total** inflow column set:
//! `z_inflow(h)` plus, for each immediately-upstream plant `i`, the turbine and
//! spillage columns at the referenced block (plus any diversion-into columns),
//! each with coefficient `+1.0`. The exhaustive **resolver-level** assertion of
//! that full column set — that the row references `z_inflow` AND each upstream
//! plant's turbine+spillage (and diversion-into) at `+1.0`, NOT merely that
//! `z_inflow` appears — is a crate-internal property of `resolve_variable_ref`,
//! which is `pub(crate)` to `cobre-sddp` and therefore unreachable from an
//! integration test under `tests/`. That assertion lives in the crate-internal
//! unit tests in `lp::generic_constraints` (e.g.
//! `hydro_inflow_two_upstream_canonical_order`,
//! `hydro_inflow_diversion_into_appends_diversion_column`,
//! `hydro_inflow_headwater_resolves_to_z_inflow_only`,
//! `hydro_inflow_is_block_dependent`), which assert the exact `(col, +1.0)`
//! pair list for a multi-upstream cascade. This file relies on that coverage and
//! proves the complementary, genuinely-integration property the unit tests
//! cannot: the full public `load → train → simulate` pipeline materializes the
//! cascade `hydro_inflow` row and **solves** it on HiGHS and on CLP.
//!
//! ## Fixture
//!
//! `tests/fixtures/b6a_hydro_inflow_cascade/` is a two-hydro cascade (H0 → H1)
//! with a `hydro_inflow(1) >= 12.0` generic constraint. H1's standalone
//! incremental inflow is 10/5/2 m³/s by stage, so the `12.0` lower bound is
//! above the incremental inflow in every stage: a slack-free solution must count
//! the upstream H0 turbine+spillage release routed into H1, making the cascade
//! columns load-bearing. Slack is enabled (so the LP is never infeasible) and
//! the bound is well below the feasible total-inflow ceiling (~40 m³/s when H0
//! releases at capacity), so the row participates without forcing infeasibility.
//!
//! ## Inertness (no new baseline)
//!
//! B6a adds no LP columns and no rows for any study without a `hydro_inflow`
//! constraint, so the existing parity baselines are byte-identical and this file
//! adds **no** `.sha256` baseline and **no** `EXPECTED_HASHES` entry. The
//! `parity_baselines_unchanged::parity_baselines_have_not_changed` guard
//! verifies that property independently.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::doc_markdown
)]
#![cfg(any(feature = "highs", feature = "clp"))]

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use cobre_core::scenario::ScenarioSource;
use cobre_core::{ConstraintSense, EntityId, VariableRef};
use cobre_sddp::{
    aggregate_simulation, hydro_models::prepare_hydro_models, setup::prepare_stochastic,
};

mod common;

use crate::common::{StubComm, build_setup_for_case};

/// Path to the multi-plant cascade fixture carrying the `hydro_inflow(1)`
/// generic constraint.
fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/b6a_hydro_inflow_cascade")
}

/// Pin the fixture's intent at the parser boundary.
///
/// The cascade target is H1 (`EntityId(1)`), the downstream plant whose upstream
/// is H0 — so the resolved row references the total-inflow column set (z_inflow
/// plus H0's turbine+spillage), not the headwater z_inflow-only case. The
/// exhaustive `(col, +1.0)` resolver-level check is the crate-internal
/// responsibility of `lp::generic_constraints`'s unit tests (see this file's
/// module docs); this assertion only guards that the fixture references the
/// cascade target so the end-to-end solve genuinely exercises B6a.
fn assert_cascade_inflow_constraint(system: &cobre_core::System) {
    let constraints = system.generic_constraints();
    assert_eq!(
        constraints.len(),
        1,
        "fixture must declare exactly one generic constraint, got {}",
        constraints.len()
    );
    let gc = &constraints[0];
    assert_eq!(
        gc.sense,
        ConstraintSense::GreaterEqual,
        "fixture constraint must be a `>=` bound"
    );
    assert_eq!(
        gc.expression.terms.len(),
        1,
        "fixture constraint must be a single hydro_inflow term, got {} terms",
        gc.expression.terms.len()
    );
    let term = &gc.expression.terms[0];
    assert_eq!(
        term.variable,
        VariableRef::HydroInflow {
            hydro_id: EntityId(1),
            block_id: None,
        },
        "fixture constraint must reference hydro_inflow(1) (cascade target H1) with no explicit block"
    );
    assert_eq!(
        term.coefficient,
        cobre_core::CoefficientRef::Literal(1.0),
        "the hydro_inflow term coefficient must be the literal +1.0"
    );
    assert!(
        (term.scale - 1.0).abs() < f64::EPSILON,
        "the hydro_inflow term scale must be +1.0, got {}",
        term.scale
    );
}

/// Run the public `load → train → simulate` pipeline for the cascade
/// `hydro_inflow` fixture under the given LP solver, asserting the LP solves
/// without error end-to-end.
///
/// Generic over the solver so the HiGHS and CLP entry points share one body;
/// the closure `make_solver` supplies fresh worker solvers for the workspace
/// pool exactly as the production training/simulation paths do.
fn run_cascade_inflow_case<S, F>(make_solver: F)
where
    S: cobre_solver::SolverInterface<Profile = cobre_solver::ActiveProfile> + Send,
    F: Fn() -> Result<S, cobre_solver::SolverError> + Copy,
{
    let dir = fixture_dir();
    let config_path = dir.join("config.json");

    let config = cobre_io::parse_config(&config_path).expect("config must parse");
    let system = cobre_io::load_case(&dir).expect("load_case must succeed");

    assert_cascade_inflow_constraint(&system);

    let pr = prepare_stochastic(system, &dir, &config, 42, &ScenarioSource::default())
        .expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, &dir, false).expect("prepare_hydro_models must succeed");

    // Enable simulation so the cascade `hydro_inflow` row is exercised on the
    // simulation LP as well as the training LP, mirroring the D-case harness.
    let mut config_with_sim = config.clone();
    config_with_sim.simulation.enabled = true;
    config_with_sim.simulation.num_scenarios = 1;

    let mut setup = build_setup_for_case(&dir, &config_with_sim, &system, stochastic, hydro_models);

    let comm = StubComm;
    let mut solver = make_solver().expect("solver construction must succeed");

    let (event_tx, _event_rx) = mpsc::channel();
    let outcome = setup
        .train(&mut solver, &comm, 1, make_solver, Some(event_tx), None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "training must not surface a solver error for the cascade hydro_inflow row: {:?}",
        outcome.error
    );
    let result = outcome.result;

    let mut pool = setup
        .create_workspace_pool(&comm, 1, make_solver)
        .expect("simulation workspace pool must build");

    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

    let local_costs = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            result.baked_templates.as_deref(),
            &result.basis_cache,
        )
        .expect("simulate must return Ok");

    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");
    assert!(
        !scenario_results.is_empty(),
        "simulation must produce at least one scenario result"
    );

    let sim_config = setup.simulation_config();
    aggregate_simulation(&local_costs.costs, sim_config, &comm)
        .expect("aggregate_simulation must succeed");
}

/// The cascade `hydro_inflow(1) >= 12.0` constraint solves end-to-end on the
/// default HiGHS backend.
#[test]
#[cfg(feature = "highs")]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn b6a_hydro_inflow_cascade_solves_highs() {
    run_cascade_inflow_case(cobre_solver::highs::HighsSolver::new);
}

/// The same cascade `hydro_inflow(1) >= 12.0` constraint solves end-to-end
/// on the CLP backend (`--no-default-features --features "clp slow-tests"`).
#[test]
#[cfg(feature = "clp")]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn b6a_hydro_inflow_cascade_solves_clp() {
    run_cascade_inflow_case(cobre_solver::clp::ClpSolver::new);
}
