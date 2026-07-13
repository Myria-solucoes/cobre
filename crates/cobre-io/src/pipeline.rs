//! Loading pipeline: validation, resolution, scenario assembly, and
//! `SystemBuilder::build` for a case directory.
//!
//! Use [`run_pipeline_with_report`] when the caller needs the warnings collected
//! during validation (e.g., the `validate` CLI subcommand).

use std::collections::HashMap;

use cobre_core::{SystemBuilder, scenario::CorrelationModel};

use crate::{
    LoadError,
    extensions::load_tailrace_curves,
    report::{ValidationReport, generate_report},
    resolution::{
        BoundsEntitySlices, BoundsOverrides, PenaltiesEntitySlices, PenaltiesOverrides,
        resolve_bounds, resolve_exchange_factors, resolve_generic_constraint_bounds,
        resolve_load_factors, resolve_ncs_bounds, resolve_ncs_factors, resolve_penalties,
    },
    scenarios::assembly::{assemble_inflow_models, assemble_load_models},
    validation::{
        ValidationContext,
        dimensional::validate_dimensional_consistency,
        productivity_resolution::validate_productivity_resolution,
        referential::validate_referential_integrity,
        scalar_parameters::validate_scalar_parameters,
        schema::validate_schema,
        semantic::{validate_semantic_hydro_thermal, validate_semantic_stages_penalties_scenarios},
        structural::validate_structure,
    },
};

use crate::{CaseArtifacts, LoadedCase};
use cobre_core::System;
use std::path::Path;

/// Run the complete loading pipeline for a case directory, discarding warnings.
///
/// Use [`run_pipeline_with_report`] to retrieve the collected warnings.
///
/// # Errors
///
/// - [`LoadError::IoError`] / [`LoadError::ParseError`] — file read or JSON/Parquet
///   parse failure.
/// - [`LoadError::ConstraintError`] — collected validation errors or
///   `SystemBuilder::build` rejection.
/// - [`LoadError::SchemaError`] — AR coefficient count mismatch in scenario assembly.
pub(crate) fn run_pipeline(path: &Path) -> Result<System, LoadError> {
    run_pipeline_with_report(path).map(|(system, _report)| system)
}

/// Run the complete loading pipeline, returning the [`System`] and the
/// [`ValidationReport`] of collected warnings.
///
/// # Errors
///
/// Same error conditions as [`run_pipeline`].
pub(crate) fn run_pipeline_with_report(
    path: &Path,
) -> Result<(System, ValidationReport), LoadError> {
    run_pipeline_with_artifacts(path).map(|(loaded, report)| (loaded.system, report))
}

/// The canonical pipeline: returns the validated [`System`] in a [`LoadedCase`]
/// bundle with the [`CaseArtifacts`] rows and the [`ValidationReport`].
/// `run_pipeline` / `run_pipeline_with_report` delegate here.
///
/// # Errors
///
/// Same error conditions as [`run_pipeline`].
// Rationale: a single linear cascade where each stage's output feeds the next; splitting at layer
// boundaries would scatter the global ordering guarantee across call sites.
#[allow(clippy::too_many_lines)]
pub(crate) fn run_pipeline_with_artifacts(
    path: &Path,
) -> Result<(LoadedCase, ValidationReport), LoadError> {
    let mut ctx = ValidationContext::new();

    let manifest = validate_structure(path, &mut ctx);

    let data = validate_schema(path, &manifest, &mut ctx);

    let Some(data) = data else {
        return ctx.into_result().and(Err(LoadError::ConstraintError {
            description: "schema validation failed but no errors were collected".to_string(),
        }));
    };

    validate_referential_integrity(&data, &mut ctx);
    validate_dimensional_consistency(&data, &mut ctx);
    validate_semantic_hydro_thermal(&data, &mut ctx);
    validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

    validate_productivity_resolution(&data, &mut ctx);
    // n_stages counts study stages (id >= 0) only — must match what
    // `build_resolved_parameters` consumes downstream.
    let n_study_stages = data.stages.stages.iter().filter(|s| s.id >= 0).count();
    validate_scalar_parameters(
        &data.scalar_parameters,
        &data.hydros,
        n_study_stages,
        &mut ctx,
    );

    let report = generate_report(&ctx);
    ctx.into_result()?;

    // Study stages only (pre-study stages have negative IDs); stage_index maps
    // domain-level stage_id to a positional 0-based index.
    let study_stages: Vec<_> = data.stages.stages.iter().filter(|s| s.id >= 0).collect();
    let n_stages = study_stages.len();
    let stage_index: HashMap<i32, usize> = study_stages
        .iter()
        .enumerate()
        .map(|(idx, s)| (s.id, idx))
        .collect();

    let penalties = resolve_penalties(
        &PenaltiesEntitySlices {
            hydros: &data.hydros,
            buses: &data.buses,
            lines: &data.lines,
            ncs_sources: &data.non_controllable_sources,
        },
        n_stages,
        &stage_index,
        &PenaltiesOverrides {
            hydro: &data.penalty_overrides_hydro,
            bus: &data.penalty_overrides_bus,
            line: &data.penalty_overrides_line,
            ncs: &data.penalty_overrides_ncs,
        },
    );

    let k_max: usize = data
        .thermals
        .iter()
        .filter_map(|t| t.anticipated_config.as_ref())
        .map(|c| {
            // A LeadTime plant contributes 0 here: this crate cannot compute the
            // resolver-derived ring depth (that's downstream, solver-side state), and
            // the delivery-anchored ring only ever reads a plant's bounds at
            // delivery_stage < n_stages, so the padded [n_stages, n_stages + k_max)
            // region resolve_bounds sizes from this value is unread for anticipated
            // plants regardless of lead mode.
            usize::try_from(c.lead_stages().unwrap_or(0)).unwrap_or(usize::MAX)
        })
        .max()
        .unwrap_or(0);
    let bounds = resolve_bounds(
        &BoundsEntitySlices {
            hydros: &data.hydros,
            thermals: &data.thermals,
            lines: &data.lines,
            pumping_stations: &data.pumping_stations,
            contracts: &data.energy_contracts,
        },
        n_stages,
        k_max,
        &stage_index,
        &BoundsOverrides {
            hydro: &data.hydro_bounds,
            thermal: &data.thermal_bounds,
            line: &data.line_bounds,
            pumping: &data.pumping_bounds,
            contract: &data.contract_bounds,
        },
    );

    let resolved_generic_bounds = resolve_generic_constraint_bounds(
        &data.generic_constraints,
        &data.generic_constraint_bounds,
    );

    let resolved_load_factors =
        resolve_load_factors(&data.load_factors, &data.buses, &data.stages.stages);
    let resolved_exchange_factors =
        resolve_exchange_factors(&data.exchange_factors, &data.lines, &data.stages.stages);

    let resolved_ncs_bounds = resolve_ncs_bounds(
        &data.ncs_bounds,
        &data.non_controllable_sources,
        n_stages,
        &stage_index,
    );

    let resolved_ncs_factors = resolve_ncs_factors(
        &data.non_controllable_factors,
        &data.non_controllable_sources,
        &data.stages.stages,
    );

    let inflow_models = assemble_inflow_models(
        data.inflow_seasonal_stats,
        data.inflow_ar_coefficients,
        data.inflow_annual_components,
    )?;
    let load_models = assemble_load_models(data.load_seasonal_stats);

    // Load tailrace curves only when the manifest detected the file: without
    // them the fit collapses to the constant entity tailrace, which zeroes γ_S
    // and emits sub-ULP LP coefficients.
    let tailrace_curves = if manifest.system_tailrace_curves_parquet {
        let tailrace_path = path.join("system").join("tailrace_curves.parquet");
        load_tailrace_curves(Some(tailrace_path.as_path()))?
    } else {
        Vec::new()
    };

    let artifacts = CaseArtifacts {
        file_manifest: manifest,
        hydro_geometry: data.hydro_geometry,
        production_models: data.production_models,
        plane_reduction: data.plane_reduction,
        hydro_energy_productivity: data.hydro_energy_productivity_rows,
        fpha_hyperplanes: data.fpha_hyperplanes,
        scalar_parameters: data.scalar_parameters,
        tailrace_curves,
    };

    let system = SystemBuilder::new()
        .buses(data.buses)
        .lines(data.lines)
        .hydros(data.hydros)
        .thermals(data.thermals)
        .pumping_stations(data.pumping_stations)
        .contracts(data.energy_contracts)
        .non_controllable_sources(data.non_controllable_sources)
        .stages(data.stages.stages)
        .policy_graph(data.stages.policy_graph)
        .penalties(penalties)
        .bounds(bounds)
        .resolved_generic_bounds(resolved_generic_bounds)
        .resolved_load_factors(resolved_load_factors)
        .resolved_exchange_factors(resolved_exchange_factors)
        .resolved_ncs_bounds(resolved_ncs_bounds)
        .resolved_ncs_factors(resolved_ncs_factors)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .ncs_models(data.ncs_models)
        .correlation(data.correlation.unwrap_or_else(CorrelationModel::default))
        .initial_conditions(data.initial_conditions)
        .generic_constraints(data.generic_constraints)
        .inflow_history(data.inflow_history)
        .external_scenarios(data.external_scenarios)
        .external_load_scenarios(data.external_load_scenarios)
        .external_ncs_scenarios(data.external_ncs_scenarios)
        .build()
        .map_err(|errs| LoadError::ConstraintError {
            description: errs
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
        })?;

    Ok((LoadedCase { system, artifacts }, report))
}
