//! Layer 3 — Referential integrity validation.
//!
//! Verifies that every cross-entity reference in `ParsedData` resolves to an
//! existing entity in the corresponding registry. Every check runs regardless of
//! errors found in earlier checks — every dangling reference is collected before
//! returning.
//!
//! The primary entry point is `validate_referential_integrity`.

use std::collections::{HashMap, HashSet};

use super::{ErrorKind, ValidationContext, schema::ParsedData};

// ── validate_referential_integrity ───────────────────────────────────────────

/// Performs Layer 3 referential integrity validation on the parsed data.
///
/// Checks that every referenced entity ID exists in its target registry. Any
/// dangling reference adds one [`ErrorKind::InvalidReference`] entry to `ctx`
/// with the message:
///
/// ```text
/// "<source_type> <source_id> references non-existent <target_type> <target_id> via field '<field_name>'"
/// ```
///
/// Infallible — all errors are collected in `ctx`; optional data collections
/// (empty `Vec` or `None`) are silently skipped.
pub(crate) fn validate_referential_integrity(data: &ParsedData, ctx: &mut ValidationContext) {
    let ids = LookupSets {
        bus: data.buses.iter().map(|b| b.id.0).collect(),
        hydro: data.hydros.iter().map(|h| h.id.0).collect(),
        thermal: data.thermals.iter().map(|t| t.id.0).collect(),
        line: data.lines.iter().map(|l| l.id.0).collect(),
        pumping: data.pumping_stations.iter().map(|p| p.id.0).collect(),
        contract: data.energy_contracts.iter().map(|c| c.id.0).collect(),
        ncs: data
            .non_controllable_sources
            .iter()
            .map(|n| n.id.0)
            .collect(),
        generic_constraint: data.generic_constraints.iter().map(|g| g.id.0).collect(),
        hydro_unit_group: data
            .hydros
            .iter()
            .map(|h| (h.id.0, h.unit_groups.iter().map(|g| g.id.0).collect()))
            .collect(),
        hydro_group_bus: data
            .hydros
            .iter()
            .map(|h| (h.id.0, h.unit_groups.iter().map(|g| g.bus_id.0).collect()))
            .collect(),
    };

    check_line_references(data, ctx, &ids.bus);
    check_hydro_references(data, ctx, &ids.bus, &ids.hydro);
    check_thermal_references(data, ctx, &ids.bus);
    check_ncs_references(data, ctx, &ids.bus, &ids.ncs);
    check_pumping_references(data, ctx, &ids.bus, &ids.hydro);
    check_contract_references(data, ctx, &ids.bus);
    check_extension_references(data, ctx, &ids.hydro);
    check_scenario_references(data, ctx, &ids.bus, &ids.hydro, &ids.ncs);
    check_bounds_references(data, ctx, &ids);
    check_penalty_override_references(data, ctx, &ids.bus, &ids.hydro, &ids.line, &ids.ncs);
    check_load_factor_references(data, ctx, &ids.bus);
    check_generic_constraint_expression_references(data, ctx, &ids);
    check_generic_constraint_bounds_validity(data, ctx);
    check_ncs_bounds_and_factors(data, ctx, &ids.ncs);
}

/// O(1) lookup sets for all entity registries, built once and shared across helpers.
struct LookupSets {
    bus: HashSet<i32>,
    hydro: HashSet<i32>,
    thermal: HashSet<i32>,
    line: HashSet<i32>,
    pumping: HashSet<i32>,
    contract: HashSet<i32>,
    ncs: HashSet<i32>,
    generic_constraint: HashSet<i32>,
    // Plant id -> its own group ids; group ids are plant-scoped, not global, so
    // this stays keyed by plant rather than flattened into one set.
    hydro_unit_group: HashMap<i32, HashSet<i32>>,
    // Plant id -> the bus ids its unit groups sit on. A bus selector names a
    // cell, which is plant-scoped, even though a bus is a global entity — the
    // global `bus` set would wrongly accept a bus that exists but that this
    // plant has no group on, so membership is checked per plant here instead.
    hydro_group_bus: HashMap<i32, HashSet<i32>>,
}

// ── Per-entity-group helper functions ─────────────────────────────────────────

/// Line -> bus references (`source_bus_id`, `target_bus_id`).
fn check_line_references(data: &ParsedData, ctx: &mut ValidationContext, bus_ids: &HashSet<i32>) {
    for line in &data.lines {
        let entity_str = format!("Line {}", line.id.0);

        if !bus_ids.contains(&line.source_bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/lines.json",
                Some(&entity_str),
                format!(
                    "{entity_str} references non-existent Bus {} via field 'source_bus_id'",
                    line.source_bus_id.0
                ),
            );
        }

        if !bus_ids.contains(&line.target_bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/lines.json",
                Some(&entity_str),
                format!(
                    "{entity_str} references non-existent Bus {} via field 'target_bus_id'",
                    line.target_bus_id.0
                ),
            );
        }
    }
}

/// Hydro -> downstream hydro, diversion, and unit group bus references.
fn check_hydro_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
    hydro_ids: &HashSet<i32>,
) {
    for hydro in &data.hydros {
        let entity_str = format!("Hydro {}", hydro.id.0);

        if let Some(downstream_id) = hydro.downstream_id
            && !hydro_ids.contains(&downstream_id.0)
        {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str} references non-existent Hydro {} via field 'downstream_id'",
                    downstream_id.0
                ),
            );
        }

        if let Some(ref diversion) = hydro.diversion
            && !hydro_ids.contains(&diversion.downstream_id.0)
        {
            ctx.add_error(
                    ErrorKind::InvalidReference,
                    "system/hydros.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str} references non-existent Hydro {} via field 'diversion.downstream_id'",
                        diversion.downstream_id.0
                    ),
                );
        }

        for group in &hydro.unit_groups {
            if !bus_ids.contains(&group.bus_id.0) {
                let group_str = format!("{entity_str} unit group {}", group.id.0);
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    "system/hydros.json",
                    Some(&group_str),
                    format!(
                        "{group_str} references non-existent Bus {} via field 'bus_id'",
                        group.bus_id.0
                    ),
                );
            }
        }
    }
}

/// Thermal -> bus reference.
fn check_thermal_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
) {
    for thermal in &data.thermals {
        let entity_str = format!("Thermal {}", thermal.id.0);

        if !bus_ids.contains(&thermal.bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/thermals.json",
                Some(&entity_str),
                format!(
                    "{entity_str} references non-existent Bus {} via field 'bus_id'",
                    thermal.bus_id.0
                ),
            );
        }
    }
}

/// NCS -> bus reference and NCS model references.
fn check_ncs_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
    ncs_ids: &HashSet<i32>,
) {
    for ncs in &data.non_controllable_sources {
        let entity_str = format!("NonControllableSource {}", ncs.id.0);

        if !bus_ids.contains(&ncs.bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/non_controllable_sources.json",
                Some(&entity_str),
                format!(
                    "{entity_str} references non-existent Bus {} via field 'bus_id'",
                    ncs.bus_id.0
                ),
            );
        }
    }

    for (i, model) in data.ncs_models.iter().enumerate() {
        if !ncs_ids.contains(&model.ncs_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/non_controllable_stats.parquet",
                Some(format!("NcsModel[{i}]")),
                format!(
                    "NcsModel[{i}] references non-existent NonControllableSource {} via field 'ncs_id'",
                    model.ncs_id.0
                ),
            );
        }
    }
}

/// `PumpingStation` -> bus and hydro references.
fn check_pumping_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
    hydro_ids: &HashSet<i32>,
) {
    for station in &data.pumping_stations {
        let entity_str = format!("PumpingStation {}", station.id.0);

        if !bus_ids.contains(&station.bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/pumping_stations.json",
                Some(&entity_str),
                format!(
                    "{entity_str} references non-existent Bus {} via field 'bus_id'",
                    station.bus_id.0
                ),
            );
        }

        if !hydro_ids.contains(&station.source_hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/pumping_stations.json",
                Some(&entity_str),
                format!(
                    "{entity_str} references non-existent Hydro {} via field 'source_hydro_id'",
                    station.source_hydro_id.0
                ),
            );
        }

        if !hydro_ids.contains(&station.destination_hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/pumping_stations.json",
                Some(&entity_str),
                format!(
                    "{entity_str} references non-existent Hydro {} via field 'destination_hydro_id'",
                    station.destination_hydro_id.0
                ),
            );
        }
    }
}

/// `EnergyContract` -> bus reference.
fn check_contract_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
) {
    for contract in &data.energy_contracts {
        let entity_str = format!("EnergyContract {}", contract.id.0);

        if !bus_ids.contains(&contract.bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/energy_contracts.json",
                Some(&entity_str),
                format!(
                    "{entity_str} references non-existent Bus {} via field 'bus_id'",
                    contract.bus_id.0
                ),
            );
        }
    }
}

/// Extension data -> hydro references (geometry, production models, FPHA).
fn check_extension_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    hydro_ids: &HashSet<i32>,
) {
    for (i, row) in data.hydro_geometry.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/hydro_geometry.parquet",
                Some(format!("HydroGeometryRow[{i}]")),
                format!(
                    "HydroGeometryRow[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    row.hydro_id.0
                ),
            );
        }
    }

    for (i, model) in data.production_models.iter().enumerate() {
        if !hydro_ids.contains(&model.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/hydro_production_models.json",
                Some(format!("ProductionModelConfig[{i}]")),
                format!(
                    "ProductionModelConfig[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    model.hydro_id.0
                ),
            );
        }
    }

    for (i, row) in data.fpha_hyperplanes.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/fpha_hyperplanes.parquet",
                Some(format!("FphaHyperplaneRow[{i}]")),
                format!(
                    "FphaHyperplaneRow[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    row.hydro_id.0
                ),
            );
        }
    }
}

/// Scenario data references.
// Rationale: the scenario data sources are checked in one error-accumulating pass;
// splitting would force multiple passes over `ParsedData` or thread sub-results between
// helpers, obscuring that all checks share one accumulator and one return point.
#[allow(clippy::too_many_lines)]
fn check_scenario_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
    hydro_ids: &HashSet<i32>,
    ncs_ids: &HashSet<i32>,
) {
    for (i, row) in data.inflow_seasonal_stats.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/inflow_seasonal_stats.parquet",
                Some(format!("InflowSeasonalStatsRow[{i}]")),
                format!(
                    "InflowSeasonalStatsRow[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    row.hydro_id.0
                ),
            );
        }
    }

    for (i, row) in data.inflow_ar_coefficients.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/inflow_ar_coefficients.parquet",
                Some(format!("InflowArCoefficientRow[{i}]")),
                format!(
                    "InflowArCoefficientRow[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    row.hydro_id.0
                ),
            );
        }
    }

    for (i, row) in data.inflow_annual_components.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/inflow_annual_component.parquet",
                Some(format!("InflowAnnualComponentRow[{i}]")),
                format!(
                    "InflowAnnualComponentRow[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    row.hydro_id.0
                ),
            );
        }
    }

    for (i, row) in data.inflow_history.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/inflow_history.parquet",
                Some(format!("InflowHistoryRow[{i}]")),
                format!(
                    "InflowHistoryRow[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    row.hydro_id.0
                ),
            );
        }
    }

    for (i, row) in data.load_seasonal_stats.iter().enumerate() {
        if !bus_ids.contains(&row.bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/load_seasonal_stats.parquet",
                Some(format!("LoadSeasonalStatsRow[{i}]")),
                format!(
                    "LoadSeasonalStatsRow[{i}] references non-existent Bus {} via field 'bus_id'",
                    row.bus_id.0
                ),
            );
        }
    }

    if let Some(ref correlation) = data.correlation {
        for profile in correlation.profiles.values() {
            for group in &profile.groups {
                for entity in &group.entities {
                    let (valid, type_label, registry_label) = match entity.entity_type.as_str() {
                        "inflow" => (hydro_ids.contains(&entity.id.0), "inflow", "Hydro"),
                        "load" => (bus_ids.contains(&entity.id.0), "load", "Bus"),
                        "ncs" => (
                            ncs_ids.contains(&entity.id.0),
                            "ncs",
                            "NonControllableSource",
                        ),
                        other => {
                            ctx.add_error(
                                ErrorKind::InvalidReference,
                                "scenarios/correlation.json",
                                Some(format!("CorrelationEntity({other}, {})", entity.id.0)),
                                format!(
                                    "unknown entity_type '{other}'; valid types are: inflow, load, ncs"
                                ),
                            );
                            continue;
                        }
                    };
                    if !valid {
                        let entity_str =
                            format!("CorrelationEntity({type_label}, {})", entity.id.0);
                        ctx.add_error(
                            ErrorKind::InvalidReference,
                            "scenarios/correlation.json",
                            Some(&entity_str),
                            format!(
                                "{entity_str} references non-existent {registry_label} {} via field 'id'",
                                entity.id.0
                            ),
                        );
                    }
                }
            }
        }
    }

    for (i, row) in data.external_scenarios.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/external_inflow_scenarios.parquet",
                Some(format!("ExternalScenarioRow[{i}]")),
                format!(
                    "ExternalScenarioRow[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    row.hydro_id.0
                ),
            );
        }
    }

    for (i, row) in data.external_load_scenarios.iter().enumerate() {
        if !bus_ids.contains(&row.bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/external_load_scenarios.parquet",
                Some(format!("ExternalLoadRow[{i}]")),
                format!(
                    "ExternalLoadRow[{i}] references non-existent Bus {} via field 'bus_id'",
                    row.bus_id.0
                ),
            );
        }
    }

    for (i, row) in data.external_ncs_scenarios.iter().enumerate() {
        if !ncs_ids.contains(&row.ncs_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/external_ncs_scenarios.parquet",
                Some(format!("ExternalNcsRow[{i}]")),
                format!(
                    "ExternalNcsRow[{i}] references non-existent NonControllableSource {} via field 'ncs_id'",
                    row.ncs_id.0
                ),
            );
        }
    }
}

/// Bounds rows -> entity references.
fn check_bounds_references(data: &ParsedData, ctx: &mut ValidationContext, ids: &LookupSets) {
    for (i, row) in data.thermal_bounds.iter().enumerate() {
        if !ids.thermal.contains(&row.thermal_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/thermal_bounds.parquet",
                Some(format!("ThermalBoundsRow[{i}]")),
                format!(
                    "ThermalBoundsRow[{i}] references non-existent Thermal {} via field 'thermal_id'",
                    row.thermal_id.0
                ),
            );
        }
    }

    for (i, row) in data.hydro_bounds.iter().enumerate() {
        if !ids.hydro.contains(&row.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/hydro_bounds.parquet",
                Some(format!("HydroBoundsRow[{i}]")),
                format!(
                    "HydroBoundsRow[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    row.hydro_id.0
                ),
            );
        }
    }

    for (i, row) in data.hydro_unit_group_bounds.iter().enumerate() {
        // An unknown hydro_id makes the group check unanswerable, so it is an
        // else-if, not two independent ifs — one finding per row, never both.
        if !ids.hydro.contains(&row.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/hydro_unit_group_bounds.parquet",
                Some(format!("HydroUnitGroupBoundsRow[{i}]")),
                format!(
                    "HydroUnitGroupBoundsRow[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    row.hydro_id.0
                ),
            );
        } else if !ids
            .hydro_unit_group
            .get(&row.hydro_id.0)
            .is_some_and(|groups| groups.contains(&row.hydro_unit_group_id.0))
        {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/hydro_unit_group_bounds.parquet",
                Some(format!("HydroUnitGroupBoundsRow[{i}]")),
                format!(
                    "HydroUnitGroupBoundsRow[{i}] references non-existent unit group {} of Hydro {} via field 'hydro_unit_group_id'; unit group ids are unique within a plant, not globally",
                    row.hydro_unit_group_id.0, row.hydro_id.0
                ),
            );
        }
    }

    for (i, row) in data.line_bounds.iter().enumerate() {
        if !ids.line.contains(&row.line_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/line_bounds.parquet",
                Some(format!("LineBoundsRow[{i}]")),
                format!(
                    "LineBoundsRow[{i}] references non-existent Line {} via field 'line_id'",
                    row.line_id.0
                ),
            );
        }
    }

    for (i, row) in data.pumping_bounds.iter().enumerate() {
        if !ids.pumping.contains(&row.station_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/pumping_bounds.parquet",
                Some(format!("PumpingBoundsRow[{i}]")),
                format!(
                    "PumpingBoundsRow[{i}] references non-existent PumpingStation {} via field 'station_id'",
                    row.station_id.0
                ),
            );
        }
    }

    for (i, row) in data.contract_bounds.iter().enumerate() {
        if !ids.contract.contains(&row.contract_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/contract_bounds.parquet",
                Some(format!("ContractBoundsRow[{i}]")),
                format!(
                    "ContractBoundsRow[{i}] references non-existent EnergyContract {} via field 'contract_id'",
                    row.contract_id.0
                ),
            );
        }
    }

    for (i, row) in data.generic_constraint_bounds.iter().enumerate() {
        if !ids.generic_constraint.contains(&row.constraint_id) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/generic_constraint_bounds.parquet",
                Some(format!("GenericConstraintBoundsRow[{i}]")),
                format!(
                    "GenericConstraintBoundsRow[{i}] references non-existent GenericConstraint {} via field 'constraint_id'",
                    row.constraint_id
                ),
            );
        }
    }
}

/// Penalty override rows -> entity references.
fn check_penalty_override_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
    hydro_ids: &HashSet<i32>,
    line_ids: &HashSet<i32>,
    ncs_ids: &HashSet<i32>,
) {
    for (i, row) in data.penalty_overrides_bus.iter().enumerate() {
        if !bus_ids.contains(&row.bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/penalty_overrides_bus.parquet",
                Some(format!("BusPenaltyOverrideRow[{i}]")),
                format!(
                    "BusPenaltyOverrideRow[{i}] references non-existent Bus {} via field 'bus_id'",
                    row.bus_id.0
                ),
            );
        }
    }

    for (i, row) in data.penalty_overrides_line.iter().enumerate() {
        if !line_ids.contains(&row.line_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/penalty_overrides_line.parquet",
                Some(format!("LinePenaltyOverrideRow[{i}]")),
                format!(
                    "LinePenaltyOverrideRow[{i}] references non-existent Line {} via field 'line_id'",
                    row.line_id.0
                ),
            );
        }
    }

    for (i, row) in data.penalty_overrides_hydro.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/penalty_overrides_hydro.parquet",
                Some(format!("HydroPenaltyOverrideRow[{i}]")),
                format!(
                    "HydroPenaltyOverrideRow[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    row.hydro_id.0
                ),
            );
        }
    }

    for (i, row) in data.penalty_overrides_ncs.iter().enumerate() {
        if !ncs_ids.contains(&row.source_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/penalty_overrides_ncs.parquet",
                Some(format!("NcsPenaltyOverrideRow[{i}]")),
                format!(
                    "NcsPenaltyOverrideRow[{i}] references non-existent NonControllableSource {} via field 'source_id'",
                    row.source_id.0
                ),
            );
        }
    }
}

/// Study stage IDs; the negative-id pre-study stages are excluded.
fn collect_study_stage_ids(data: &ParsedData) -> HashSet<i32> {
    data.stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect()
}

/// `LoadFactorEntry` -> bus and stage references.
fn check_load_factor_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
) {
    let study_stage_ids = collect_study_stage_ids(data);

    for (i, entry) in data.load_factors.iter().enumerate() {
        if !bus_ids.contains(&entry.bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/load_factors.json",
                Some(format!("LoadFactorEntry[{i}]")),
                format!(
                    "LoadFactorEntry[{i}] references non-existent Bus {} via field 'bus_id'",
                    entry.bus_id.0
                ),
            );
        }

        if !study_stage_ids.contains(&entry.stage_id) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/load_factors.json",
                Some(format!("LoadFactorEntry[{i}]")),
                format!(
                    "LoadFactorEntry[{i}] references non-existent Stage {} via field 'stage_id'",
                    entry.stage_id
                ),
            );
        }
    }
}

/// `GenericConstraint` expression entity ID existence.
fn check_generic_constraint_expression_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    ids: &LookupSets,
) {
    for constraint in &data.generic_constraints {
        let gc_label = format!("GenericConstraint {}", constraint.id.0);
        for (term_idx, term) in constraint.expression.terms.iter().enumerate() {
            let label = format!("{gc_label} term[{term_idx}]");
            validate_variable_ref_entity(&term.variable, &label, ids, ctx);
        }
    }
}

/// `GenericConstraintBoundsRow` `block_id` validity and duplicate key detection.
fn check_generic_constraint_bounds_validity(data: &ParsedData, ctx: &mut ValidationContext) {
    let stage_block_counts: std::collections::HashMap<i32, usize> = data
        .stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| (s.id, s.blocks.len()))
        .collect();

    for (i, row) in data.generic_constraint_bounds.iter().enumerate() {
        if let Some(blk) = row.block_id
            && let Some(&n_blocks) = stage_block_counts.get(&row.stage_id)
        {
            #[allow(clippy::cast_sign_loss)]
            let blk_usize = blk as usize;
            if blk < 0 || blk_usize >= n_blocks {
                ctx.add_error(
                        ErrorKind::InvalidValue,
                        "constraints/generic_constraint_bounds.parquet",
                        Some(format!("GenericConstraintBoundsRow[{i}]")),
                        format!(
                            "GenericConstraintBoundsRow[{i}] has block_id={blk} but Stage {} has only {n_blocks} block(s) (valid range: 0..{n_blocks})",
                            row.stage_id
                        ),
                    );
            }
        }
    }

    let mut seen_keys: HashSet<(i32, i32, Option<i32>)> = HashSet::new();
    for (i, row) in data.generic_constraint_bounds.iter().enumerate() {
        let key = (row.constraint_id, row.stage_id, row.block_id);
        if !seen_keys.insert(key) {
            ctx.add_error(
                ErrorKind::DuplicateId,
                "constraints/generic_constraint_bounds.parquet",
                Some(format!("GenericConstraintBoundsRow[{i}]")),
                format!(
                    "Duplicate key (constraint_id={}, stage_id={}, block_id={:?}) in generic constraint bounds",
                    row.constraint_id, row.stage_id, row.block_id
                ),
            );
        }
    }
}

/// NCS bounds and NCS factor entry checks.
fn check_ncs_bounds_and_factors(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    ncs_ids: &HashSet<i32>,
) {
    let study_stage_ids = collect_study_stage_ids(data);

    for (i, row) in data.ncs_bounds.iter().enumerate() {
        if !ncs_ids.contains(&row.ncs_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/ncs_bounds.parquet",
                Some(format!("NcsBoundsRow[{i}]")),
                format!(
                    "NcsBoundsRow[{i}] references non-existent NonControllableSource {} via field 'ncs_id'",
                    row.ncs_id.0
                ),
            );
        }
        if !study_stage_ids.contains(&row.stage_id) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/ncs_bounds.parquet",
                Some(format!("NcsBoundsRow[{i}]")),
                format!(
                    "NcsBoundsRow[{i}] has invalid stage_id {} (not a valid study stage)",
                    row.stage_id
                ),
            );
        }
        if row.available_generation_mw < 0.0 {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "constraints/ncs_bounds.parquet",
                Some(format!("NcsBoundsRow[{i}]")),
                format!(
                    "NcsBoundsRow[{i}] has negative available_generation_mw: {}",
                    row.available_generation_mw
                ),
            );
        }
    }

    for (i, entry) in data.non_controllable_factors.iter().enumerate() {
        if !ncs_ids.contains(&entry.ncs_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/non_controllable_factors.json",
                Some(format!("NcsFactorEntry[{i}]")),
                format!(
                    "NcsFactorEntry[{i}] references non-existent NonControllableSource {} via field 'ncs_id'",
                    entry.ncs_id.0
                ),
            );
        }
        if !study_stage_ids.contains(&entry.stage_id) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/non_controllable_factors.json",
                Some(format!("NcsFactorEntry[{i}]")),
                format!(
                    "NcsFactorEntry[{i}] has invalid stage_id {} (not a valid study stage)",
                    entry.stage_id
                ),
            );
        }
        for (j, bf) in entry.block_factors.iter().enumerate() {
            if bf.factor < 0.0 {
                ctx.add_error(
                    ErrorKind::InvalidValue,
                    "scenarios/non_controllable_factors.json",
                    Some(format!("NcsFactorEntry[{i}].block_factors[{j}]")),
                    format!(
                        "NcsFactorEntry[{i}] block_factors[{j}] has negative factor: {}",
                        bf.factor
                    ),
                );
            }
        }
    }
}

/// Validate that a [`VariableRef`] references an existing entity.
///
/// A dangling reference is an [`ErrorKind::InvalidReference`] error for every
/// modeled entity type. `Contract` is the sole remaining stub (data-complete but
/// contributing no LP variables), so a dangling `Contract` reference is downgraded
/// to an [`ErrorKind::UnusedEntity`] warning, not an error.
fn validate_variable_ref_entity(
    var: &cobre_core::VariableRef,
    label: &str,
    ids: &LookupSets,
    ctx: &mut ValidationContext,
) {
    use cobre_core::VariableRef;

    let file = "system/generic_constraints.json";
    match var {
        VariableRef::HydroStorage { hydro_id, .. }
        | VariableRef::HydroEvaporation { hydro_id, .. }
        | VariableRef::HydroWithdrawal { hydro_id, .. }
        | VariableRef::HydroSpillage { hydro_id, .. }
        | VariableRef::HydroDiversion { hydro_id, .. }
        | VariableRef::HydroOutflow { hydro_id, .. }
        | VariableRef::HydroInflow { hydro_id, .. }
        | VariableRef::HydroStorageInitial { hydro_id, .. }
        | VariableRef::HydroStorageFinal { hydro_id, .. } => {
            if !ids.hydro.contains(&hydro_id.0) {
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    file,
                    Some(label.to_string()),
                    format!("{label} references non-existent Hydro {}", hydro_id.0),
                );
            }
        }
        VariableRef::HydroTurbined {
            hydro_id, bus_id, ..
        }
        | VariableRef::HydroGeneration {
            hydro_id, bus_id, ..
        } => {
            if !ids.hydro.contains(&hydro_id.0) {
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    file,
                    Some(label.to_string()),
                    format!("{label} references non-existent Hydro {}", hydro_id.0),
                );
            } else if let Some(b) = bus_id
                && !ids
                    .hydro_group_bus
                    .get(&hydro_id.0)
                    .is_some_and(|buses| buses.contains(&b.0))
            {
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    file,
                    Some(label.to_string()),
                    format!(
                        "{label} references bus {}, on which Hydro {} has no unit group, via field 'bus_id'; a bus selector names one side of a split plant, not any bus in the system",
                        b.0, hydro_id.0
                    ),
                );
            }
        }
        VariableRef::ThermalGeneration { thermal_id, .. }
        | VariableRef::AnticipatedDecision { thermal_id, .. } => {
            if !ids.thermal.contains(&thermal_id.0) {
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    file,
                    Some(label.to_string()),
                    format!("{label} references non-existent Thermal {}", thermal_id.0),
                );
            }
        }
        VariableRef::LineDirect { line_id, .. }
        | VariableRef::LineReverse { line_id, .. }
        | VariableRef::LineExchange { line_id, .. } => {
            if !ids.line.contains(&line_id.0) {
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    file,
                    Some(label.to_string()),
                    format!("{label} references non-existent Line {}", line_id.0),
                );
            }
        }
        VariableRef::BusDeficit { bus_id, .. } | VariableRef::BusExcess { bus_id, .. } => {
            if !ids.bus.contains(&bus_id.0) {
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    file,
                    Some(label.to_string()),
                    format!("{label} references non-existent Bus {}", bus_id.0),
                );
            }
        }
        VariableRef::PumpingFlow { station_id, .. }
        | VariableRef::PumpingPower { station_id, .. } => {
            if !ids.pumping.contains(&station_id.0) {
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    file,
                    Some(label.to_string()),
                    format!(
                        "{label} references non-existent PumpingStation {}",
                        station_id.0
                    ),
                );
            }
        }
        VariableRef::ContractImport { contract_id, .. }
        | VariableRef::ContractExport { contract_id, .. } => {
            if !ids.contract.contains(&contract_id.0) {
                ctx.add_warning(
                    ErrorKind::UnusedEntity,
                    file,
                    Some(label.to_string()),
                    format!(
                        "{label} references Contract {} which is a stub entity with no LP effect",
                        contract_id.0
                    ),
                );
            }
        }
        VariableRef::NonControllableGeneration { source_id, .. }
        | VariableRef::NonControllableCurtailment { source_id, .. } => {
            if !ids.ncs.contains(&source_id.0) {
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    file,
                    Some(label.to_string()),
                    format!(
                        "{label} references non-existent NonControllableSource {}",
                        source_id.0
                    ),
                );
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown
)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use cobre_core::{
        EntityId,
        entities::{
            Bus, DiversionChannel, Hydro, HydroGenerationModel, HydroPenalties, HydroUnitGroup,
            Line, NonControllableSource, PumpingStation, Thermal,
        },
        scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
    };
    use std::collections::BTreeMap;
    use std::fs;
    use tempfile::TempDir;

    use crate::{
        constraints::{
            BusPenaltyOverrideRow, GenericConstraintBoundsRow, HydroBoundsRow,
            HydroUnitGroupBoundsRow, LineBoundsRow, NcsBoundsRow, NcsPenaltyOverrideRow,
            ThermalBoundsRow,
        },
        extensions::HydroGeometryRow,
        scenarios::{
            BlockFactor, InflowSeasonalStatsRow, LoadFactorEntry, LoadSeasonalStatsRow,
            NcsFactorEntry,
        },
        validation::{
            schema::{ParsedData, validate_schema},
            structural::validate_structure,
        },
    };

    const VALID_CONFIG_JSON: &str = r#"{
        "training": {
            "selection": {"method": "sampled", "forward_passes": 10},
            "stopping_rules": [
                { "type": "iteration_limit", "limit": 100 }
            ]
        }
    }"#;

    const VALID_PENALTIES_JSON: &str = r#"{
        "bus": {
            "deficit_segments": [
                { "depth_mw": 500.0, "cost": 1000.0 },
                { "depth_mw": null,  "cost": 5000.0 }
            ],
            "excess_cost": 100.0
        },
        "line": { "exchange_cost": 2.0 },
        "hydro": {
            "spillage_cost": 0.01,
            "turbined_cost": 0.05,
            "diversion_cost": 0.1,
            "storage_violation_below_cost": 10000.0,
            "filling_target_violation_cost": 50000.0,
            "turbined_violation_below_cost": 500.0,
            "outflow_violation_below_cost": 500.0,
            "outflow_violation_above_cost": 500.0,
            "generation_violation_below_cost": 1000.0,
            "evaporation_violation_cost": 5000.0,
            "water_withdrawal_violation_cost": 1000.0
        },
        "non_controllable_source": { "curtailment_cost": 0.005 }
    }"#;

    const VALID_STAGES_JSON: &str = r#"{
        "policy_graph": {
            "type": "finite_horizon",
            "annual_discount_rate": 0.06,
            "transitions": []
        },
        "stages": [
            {
                "id": 0,
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
                "blocks": [{ "id": 0, "name": "FLAT", "hours": 744.0 }],
                "num_openings": 50
            }
        ]
    }"#;

    const VALID_INITIAL_CONDITIONS_JSON: &str = r#"{
        "storage": [],
        "filling_storage": []
    }"#;

    /// Write a string to a relative path under `root`, creating parent dirs.
    fn write_file(root: &std::path::Path, relative: &str, content: &str) {
        let full = root.join(relative);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&full, content).unwrap();
    }

    /// Build a minimal case directory with buses=[1], hydros=[], thermals=[], lines=[].
    fn make_minimal_case(dir: &TempDir) {
        let root = dir.path();
        write_file(root, "config.json", VALID_CONFIG_JSON);
        write_file(root, "penalties.json", VALID_PENALTIES_JSON);
        write_file(root, "stages.json", VALID_STAGES_JSON);
        write_file(
            root,
            "initial_conditions.json",
            VALID_INITIAL_CONDITIONS_JSON,
        );
        write_file(
            root,
            "system/buses.json",
            r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }] }"#,
        );
        write_file(root, "system/lines.json", r#"{ "lines": [] }"#);
        write_file(root, "system/hydros.json", r#"{ "hydros": [] }"#);
        write_file(root, "system/thermals.json", r#"{ "thermals": [] }"#);
    }

    /// Parse the case directory at `dir` and return `ParsedData`.
    /// Panics if validation fails — all test cases start from valid data.
    fn parse_case(dir: &TempDir) -> ParsedData {
        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);
        assert!(
            !ctx.has_errors(),
            "structural validation failed: {:?}",
            ctx.errors()
        );
        let data = validate_schema(dir.path(), &manifest, &mut ctx)
            .expect("schema validation should succeed for valid case");
        assert!(
            !ctx.has_errors(),
            "schema validation failed: {:?}",
            ctx.errors()
        );
        data
    }

    fn hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 1.0,
            diversion_cost: 1.0,
            turbined_cost: 1.0,
            storage_violation_below_cost: 1.0,
            filling_target_violation_cost: 1.0,
            turbined_violation_below_cost: 1.0,
            outflow_violation_below_cost: 1.0,
            outflow_violation_above_cost: 1.0,
            generation_violation_below_cost: 1.0,
            evaporation_violation_cost: 1.0,
            water_withdrawal_violation_cost: 1.0,
            water_withdrawal_violation_pos_cost: 1.0,
            water_withdrawal_violation_neg_cost: 1.0,
            evaporation_violation_pos_cost: 1.0,
            evaporation_violation_neg_cost: 1.0,
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    fn make_hydro(id: i32) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId::from(id),
            name: format!("Hydro_{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 1000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1000.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 1000.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: hydro_penalties(),
        };
        hydro.sort_unit_groups();
        hydro
    }

    fn make_line(id: i32, source_bus: i32, target_bus: i32) -> Line {
        Line {
            id: EntityId::from(id),
            name: format!("Line_{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            source_bus_id: EntityId::from(source_bus),
            target_bus_id: EntityId::from(target_bus),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: 100.0,
            reverse_capacity_mw: 100.0,
            losses_percent: 0.0,
            exchange_cost: 0.0,
        }
    }

    fn make_ncs(id: i32, bus_id: i32) -> NonControllableSource {
        NonControllableSource {
            id: EntityId::from(id),
            name: format!("Ncs_{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 50.0,
            allow_curtailment: true,
            curtailment_cost: 1.0,
        }
    }

    fn make_pumping(id: i32, bus_id: i32, src_hydro: i32, dst_hydro: i32) -> PumpingStation {
        PumpingStation {
            id: EntityId::from(id),
            name: format!("Pump_{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(bus_id),
            source_hydro_id: EntityId::from(src_hydro),
            destination_hydro_id: EntityId::from(dst_hydro),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s: 0.5,
            min_flow_m3s: 0.0,
            max_flow_m3s: 100.0,
        }
    }

    fn make_unit_group(
        id: i32,
        bus_id: i32,
        min_generation_mw: f64,
        max_generation_mw: f64,
        min_turbined_m3s: f64,
        max_turbined_m3s: f64,
    ) -> HydroUnitGroup {
        HydroUnitGroup {
            id: EntityId::from(id),
            name: format!("Group {id}"),
            bus_id: EntityId::from(bus_id),
            min_generation_mw,
            max_generation_mw,
            min_turbined_m3s,
            max_turbined_m3s,
        }
    }

    /// Given a `ParsedData` where all entity cross-references are valid,
    /// `validate_referential_integrity` adds no errors to `ctx`.
    #[test]
    fn test_all_valid_references_no_errors() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let data = parse_case(&dir);
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "expected no errors for valid data, got: {:?}",
            ctx.errors()
        );
    }

    /// Given a `ParsedData` where Line id=5 has `source_bus_id` referencing
    /// non-existent bus id=999, `validate_referential_integrity` adds exactly 1
    /// `InvalidReference` error mentioning `"Line 5"` and `"999"`.
    #[test]
    fn test_line_invalid_source_bus() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // bus 999 does not exist — only bus 1 was loaded
        data.lines = vec![make_line(5, 999, 1)];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors(), "expected errors for invalid line ref");
        let errors = ctx.errors();
        let inv_ref: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv_ref.len(),
            1,
            "expected exactly 1 InvalidReference error"
        );
        let msg = &inv_ref[0].message;
        assert!(
            msg.contains("Line 5"),
            "message should contain 'Line 5', got: {msg}"
        );
        assert!(
            msg.contains("999"),
            "message should contain '999', got: {msg}"
        );
    }

    /// Given a `ParsedData` where `Hydro` id=3 has `downstream_id = Some(EntityId(100))`
    /// and hydro 100 does not exist, `validate_referential_integrity` adds an
    /// `InvalidReference` error mentioning `"Hydro 3"` and `"downstream_id"`.
    #[test]
    fn test_hydro_invalid_downstream_id() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        let mut hydro = make_hydro(3);
        hydro.downstream_id = Some(EntityId::from(100)); // hydro 100 does not exist
        data.hydros = vec![hydro];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            ctx.has_errors(),
            "expected error for dangling downstream_id"
        );
        let errors = ctx.errors();
        let inv_ref: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert!(
            !inv_ref.is_empty(),
            "expected at least 1 InvalidReference error"
        );
        let msg = &inv_ref[0].message;
        assert!(
            msg.contains("Hydro 3"),
            "message should contain 'Hydro 3', got: {msg}"
        );
        assert!(
            msg.contains("downstream_id"),
            "message should contain 'downstream_id', got: {msg}"
        );
    }

    /// Given a `ParsedData` with empty `pumping_stations` and `energy_contracts`,
    /// `validate_referential_integrity` produces no errors for those rules.
    #[test]
    fn test_empty_optional_collections_no_errors() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.pumping_stations = vec![];
        data.energy_contracts = vec![];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "empty optional collections should not produce errors, got: {:?}",
            ctx.errors()
        );
    }

    /// Given a `ParsedData` with 2 invalid bus references (Line, Thermal)
    /// and 1 invalid hydro reference (HydroGeometryRow), all 3 are collected.
    #[test]
    fn test_multiple_invalid_references_all_collected() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // Line with bad target_bus_id (bus 999 does not exist)
        data.lines = vec![make_line(5, 1, 999)];
        // Thermal with bad bus_id (bus 777 does not exist)
        data.thermals = vec![Thermal {
            id: EntityId::from(20),
            name: "T20".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(777), // bad
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: None,
        }];
        // HydroGeometryRow referencing non-existent hydro (888)
        data.hydro_geometry = vec![HydroGeometryRow {
            hydro_id: EntityId::from(888),
            volume_hm3: 0.0,
            area_km2: 0.0,
            height_m: 0.0,
        }];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            ctx.has_errors(),
            "expected errors for multiple invalid refs"
        );
        let errors = ctx.errors();
        let inv_ref: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv_ref.len(),
            3,
            "expected exactly 3 InvalidReference errors, got {}: {:?}",
            inv_ref.len(),
            inv_ref.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
    }

    /// A hydro with no declared unit groups produces no referential error —
    /// the group-bus check has nothing to iterate.
    #[test]
    fn test_hydro_valid_bus_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.hydros = vec![make_hydro(10)];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    /// A hydro whose plant `bus_id` names a nonexistent bus is no longer
    /// checked at plant level: with a group on a valid bus, Layer 3 reports
    /// no error at all.
    #[test]
    fn test_hydro_invalid_plant_bus_with_valid_group_bus_produces_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        let mut hydro = make_hydro(10);
        hydro.unit_groups = vec![make_unit_group(0, 1, 0.0, 100.0, 0.0, 100.0)]; // group bus 1 exists
        data.hydros = vec![hydro];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    /// A hydro whose unit group's `bus_id` equals its own (also nonexistent)
    /// plant `bus_id` is rejected — the distinctness clause that used to
    /// suppress this case is gone. Exactly one `InvalidReference` is
    /// reported and it names the unit group, not the plant.
    #[test]
    fn test_hydro_group_bus_equals_invalid_plant_bus_is_rejected() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        let mut hydro = make_hydro(10);
        hydro.unit_groups = vec![make_unit_group(0, 999, 0.0, 100.0, 0.0, 100.0)]; // group bus also 999
        data.hydros = vec![hydro];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("Hydro 10 unit group 0"));
        assert!(!inv[0].message.contains("Hydro 10 references"));
        assert!(inv[0].message.contains("Bus 999"));
    }

    /// Hydro with `downstream_id = None` must not produce any error.
    #[test]
    fn test_hydro_downstream_id_none_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        let mut hydro = make_hydro(10);
        hydro.downstream_id = None;
        data.hydros = vec![hydro];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "downstream_id = None should not produce errors"
        );
    }

    /// Hydro with `diversion = None` must not produce any error.
    #[test]
    fn test_hydro_diversion_none_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        let mut hydro = make_hydro(10);
        hydro.diversion = None;
        data.hydros = vec![hydro];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "diversion = None should not produce errors"
        );
    }

    /// Hydro with a diversion referencing a non-existent downstream produces 1 error.
    #[test]
    fn test_hydro_diversion_invalid_downstream() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        let mut hydro = make_hydro(10);
        hydro.diversion = Some(DiversionChannel {
            downstream_id: EntityId::from(999), // does not exist
            max_flow_m3s: 100.0,
        });
        data.hydros = vec![hydro];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("diversion.downstream_id"));
        assert!(inv[0].message.contains("999"));
    }

    /// Given a two-hydro study with buses `{0, 1}` declared, where hydro 1's
    /// unit group sits on bus 0 (valid, distinct from its own bus 1) and hydro
    /// 2's unit group sits on bus 42 (nonexistent), exactly one
    /// `InvalidReference` is emitted naming hydro 2 and its group, and hydro 1
    /// produces no finding.
    #[test]
    fn test_unit_group_on_nonexistent_bus_is_rejected() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.buses.push(Bus {
            id: EntityId::from(0),
            name: "BUS_0".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![],
            excess_cost: 100.0,
        });

        let mut hydro1 = make_hydro(1);
        hydro1.unit_groups = vec![HydroUnitGroup {
            id: EntityId::from(4),
            name: "Group A".to_string(),
            bus_id: EntityId::from(0), // bus 0 exists, distinct from hydro1's own bus 1
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
        }];

        let mut hydro2 = make_hydro(2);
        hydro2.unit_groups = vec![HydroUnitGroup {
            id: EntityId::from(7),
            name: "Group B".to_string(),
            bus_id: EntityId::from(42), // does not exist
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
        }];

        data.hydros = vec![hydro1, hydro2];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
        assert!(
            inv[0].message.contains("Hydro 2 unit group 7"),
            "message should name Hydro 2 unit group 7, got: {}",
            inv[0].message
        );
        assert!(inv[0].message.contains("Bus 42"));
        assert!(inv[0].message.contains("bus_id"));
        assert!(
            !inv.iter().any(|e| e.message.contains("Hydro 1")),
            "hydro 1's valid-bus group must produce no finding, got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
    }

    /// PumpingStation with valid bus and hydro references produces no error.
    #[test]
    fn test_pumping_valid_refs() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.hydros = vec![make_hydro(10)];
        data.pumping_stations = vec![make_pumping(1, 1, 10, 10)]; // bus 1, hydros 10,10 all exist
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    /// PumpingStation referencing a non-existent source hydro produces 1 error.
    #[test]
    fn test_pumping_invalid_source_hydro() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.hydros = vec![make_hydro(10)];
        // source hydro 999 missing, destination hydro 10 exists
        data.pumping_stations = vec![make_pumping(1, 1, 999, 10)];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("source_hydro_id"));
        assert!(inv[0].message.contains("999"));
    }

    /// PumpingStation referencing a non-existent bus produces 1 error.
    #[test]
    fn test_pumping_invalid_bus() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.hydros = vec![make_hydro(10)];
        // bus 777 missing; source/destination hydro 10 exists
        data.pumping_stations = vec![make_pumping(1, 777, 10, 10)];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("bus_id"));
        assert!(inv[0].message.contains("777"));
    }

    /// PumpingStation referencing a non-existent destination hydro produces 1 error.
    #[test]
    fn test_pumping_invalid_destination_hydro() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.hydros = vec![make_hydro(10)];
        // source hydro 10 exists; destination hydro 999 missing
        data.pumping_stations = vec![make_pumping(1, 1, 10, 999)];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("destination_hydro_id"));
        assert!(inv[0].message.contains("999"));
    }

    /// `InflowSeasonalStatsRow` referencing non-existent hydro produces 1 error.
    #[test]
    fn test_inflow_seasonal_stats_invalid_hydro_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // hydro 999 does not exist
        data.inflow_seasonal_stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId::from(999),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 10.0,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("999"));
        assert!(inv[0].message.contains("hydro_id"));
    }

    /// `LoadSeasonalStatsRow` referencing non-existent bus produces 1 error.
    #[test]
    fn test_load_seasonal_stats_invalid_bus_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // bus 777 does not exist
        data.load_seasonal_stats = vec![LoadSeasonalStatsRow {
            bus_id: EntityId::from(777),
            stage_id: 0,
            mean_mw: 100.0,
            std_mw: 10.0,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("777"));
        assert!(inv[0].message.contains("bus_id"));
    }

    /// `CorrelationEntity` with a dangling inflow reference and one with an
    /// unknown `entity_type` each produce one `InvalidReference` error.
    #[test]
    fn test_correlation_entity_inflow_invalid_hydro() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "profile1".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "group1".to_string(),
                    entities: vec![
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId::from(999), // does not exist
                        },
                        CorrelationEntity {
                            entity_type: "unknown".to_string(),
                            id: EntityId::from(9999),
                        },
                    ],
                    matrix: vec![],
                }],
            },
        );
        data.correlation = Some(CorrelationModel {
            method: "pearson".to_string(),
            profiles,
            schedule: vec![],
        });

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv.len(),
            2,
            "expected errors for invalid hydro and unknown entity_type"
        );
        assert!(inv.iter().any(|e| e.message.contains("999")));
        assert!(
            inv.iter()
                .any(|e| e.message.contains("unknown entity_type"))
        );
    }

    /// `CorrelationEntity` with `entity_type == "inflow"` and a valid hydro id
    /// produces no error.
    #[test]
    fn test_correlation_entity_inflow_valid_hydro() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.hydros = vec![make_hydro(10)];
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "profile1".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "group1".to_string(),
                    entities: vec![CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId::from(10), // hydro 10 exists
                    }],
                    matrix: vec![],
                }],
            },
        );
        data.correlation = Some(CorrelationModel {
            method: "pearson".to_string(),
            profiles,
            schedule: vec![],
        });

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid inflow ref should not produce errors"
        );
    }

    /// `ThermalBoundsRow` referencing a non-existent thermal produces 1 error.
    #[test]
    fn test_thermal_bounds_invalid_thermal_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // thermal 999 does not exist
        data.thermal_bounds = vec![ThermalBoundsRow {
            thermal_id: EntityId::from(999),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: None,
            cost_per_mwh: None,
            block_id: None,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("999"));
        assert!(inv[0].message.contains("thermal_id"));
    }

    /// `HydroBoundsRow` referencing a non-existent hydro produces 1 error.
    #[test]
    fn test_hydro_bounds_invalid_hydro_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // hydro 555 does not exist
        data.hydro_bounds = vec![HydroBoundsRow {
            hydro_id: EntityId::from(555),
            stage_id: 0,
            min_turbined_m3s: None,
            max_turbined_m3s: None,
            min_storage_hm3: None,
            max_storage_hm3: None,
            min_outflow_m3s: None,
            max_outflow_m3s: None,
            min_generation_mw: None,
            max_generation_mw: None,
            max_diversion_m3s: None,
            filling_min_rate_m3s: None,
            water_withdrawal_m3s: None,
            block_id: None,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("555"));
        assert!(inv[0].message.contains("hydro_id"));
    }

    /// A `hydro_unit_group_bounds` row naming unit group 4 on Hydro 7 (which
    /// declares groups `{0, 3}`) is rejected even though group 4 exists on a
    /// different plant (Hydro 2) — group ids are plant-scoped, not global.
    #[test]
    fn test_hydro_unit_group_bounds_unknown_group_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        let mut hydro7 = make_hydro(7);
        hydro7.unit_groups = vec![
            make_unit_group(0, 1, 0.0, 100.0, 0.0, 100.0),
            make_unit_group(3, 1, 0.0, 100.0, 0.0, 100.0),
        ];
        let mut hydro2 = make_hydro(2);
        hydro2.unit_groups = vec![make_unit_group(4, 1, 0.0, 100.0, 0.0, 100.0)];
        data.hydros = vec![hydro7, hydro2];

        data.hydro_unit_group_bounds = vec![HydroUnitGroupBoundsRow {
            hydro_id: EntityId::from(7),
            hydro_unit_group_id: EntityId::from(4),
            stage_id: 9,
            min_turbined_m3s: None,
            max_turbined_m3s: None,
            min_generation_mw: None,
            max_generation_mw: Some(50.0),
            block_id: Some(1),
        }];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
        assert!(inv[0].file == std::path::Path::new("constraints/hydro_unit_group_bounds.parquet"));
        assert!(inv[0].message.contains("unit group 4"));
        assert!(inv[0].message.contains("Hydro 7"));
    }

    /// A `hydro_unit_group_bounds` row with `hydro_id = 99` (no such plant)
    /// emits exactly one finding — the dangling `hydro_id` — and no
    /// `hydro_unit_group_id` finding, even though group 4 does not exist on
    /// plant 99 either: the plant reference is unanswerable first.
    #[test]
    fn test_hydro_unit_group_bounds_unknown_hydro_ref_emits_one_finding() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        let mut hydro7 = make_hydro(7);
        hydro7.unit_groups = vec![
            make_unit_group(0, 1, 0.0, 100.0, 0.0, 100.0),
            make_unit_group(3, 1, 0.0, 100.0, 0.0, 100.0),
        ];
        data.hydros = vec![hydro7];

        data.hydro_unit_group_bounds = vec![HydroUnitGroupBoundsRow {
            hydro_id: EntityId::from(99),
            hydro_unit_group_id: EntityId::from(4),
            stage_id: 9,
            min_turbined_m3s: None,
            max_turbined_m3s: None,
            min_generation_mw: None,
            max_generation_mw: Some(50.0),
            block_id: Some(1),
        }];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference (hydro_id only), got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
        assert!(inv[0].message.contains("Hydro 99"));
        assert!(inv[0].message.contains("hydro_id"));
        assert!(!inv[0].message.contains("hydro_unit_group_id"));
    }

    /// Every `hydro_unit_group_bounds` row names a declared `(plant, group)`
    /// pair on plants whose group ids are neither `0` nor equal to their own
    /// position in `unit_groups` — no finding is produced against
    /// `constraints/hydro_unit_group_bounds.parquet`.
    #[test]
    fn test_hydro_unit_group_bounds_valid_refs_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        let mut hydro5 = make_hydro(5);
        hydro5.unit_groups = vec![
            make_unit_group(7, 1, 0.0, 100.0, 0.0, 100.0),
            make_unit_group(2, 1, 0.0, 100.0, 0.0, 100.0),
        ];
        let mut hydro6 = make_hydro(6);
        hydro6.unit_groups = vec![
            make_unit_group(10, 1, 0.0, 100.0, 0.0, 100.0),
            make_unit_group(20, 1, 0.0, 100.0, 0.0, 100.0),
        ];
        data.hydros = vec![hydro5, hydro6];

        data.hydro_unit_group_bounds = vec![
            HydroUnitGroupBoundsRow {
                hydro_id: EntityId::from(5),
                hydro_unit_group_id: EntityId::from(2),
                stage_id: 8,
                min_turbined_m3s: None,
                max_turbined_m3s: None,
                min_generation_mw: None,
                max_generation_mw: Some(50.0),
                block_id: Some(3),
            },
            HydroUnitGroupBoundsRow {
                hydro_id: EntityId::from(6),
                hydro_unit_group_id: EntityId::from(10),
                stage_id: 4,
                min_turbined_m3s: Some(1.0),
                max_turbined_m3s: None,
                min_generation_mw: None,
                max_generation_mw: None,
                block_id: None,
            },
        ];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .filter(|e| {
                e.file == std::path::Path::new("constraints/hydro_unit_group_bounds.parquet")
            })
            .collect();
        assert!(
            inv.is_empty(),
            "expected no hydro_unit_group_bounds errors, got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
    }

    /// `LineBoundsRow` referencing a non-existent line produces 1 error.
    #[test]
    fn test_line_bounds_invalid_line_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // line 333 does not exist
        data.line_bounds = vec![LineBoundsRow {
            line_id: EntityId::from(333),
            stage_id: 0,
            direct_mw: None,
            reverse_mw: None,
            block_id: None,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("333"));
        assert!(inv[0].message.contains("line_id"));
    }

    /// `GenericConstraintBoundsRow` referencing a non-existent constraint produces 1 error.
    #[test]
    fn test_generic_constraint_bounds_invalid_constraint_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // constraint 888 does not exist
        data.generic_constraint_bounds = vec![GenericConstraintBoundsRow {
            constraint_id: 888,
            stage_id: 0,
            block_id: None,
            bound: 1000.0,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("888"));
        assert!(inv[0].message.contains("constraint_id"));
    }

    /// `BusPenaltyOverrideRow` referencing a non-existent bus produces 1 error.
    #[test]
    fn test_bus_penalty_override_invalid_bus_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // bus 777 does not exist
        data.penalty_overrides_bus = vec![BusPenaltyOverrideRow {
            bus_id: EntityId::from(777),
            stage_id: 0,
            excess_cost: None,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("777"));
        assert!(inv[0].message.contains("bus_id"));
    }

    /// `NcsPenaltyOverrideRow` referencing a non-existent NCS source produces 1 error.
    #[test]
    fn test_ncs_penalty_override_invalid_ncs_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // NCS source 444 does not exist
        data.penalty_overrides_ncs = vec![NcsPenaltyOverrideRow {
            source_id: EntityId::from(444),
            stage_id: 0,
            curtailment_cost: None,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("444"));
        assert!(inv[0].message.contains("source_id"));
    }

    /// `NcsPenaltyOverrideRow` with a valid NCS source produces no error.
    #[test]
    fn test_ncs_penalty_override_valid_ncs_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.non_controllable_sources = vec![make_ncs(1, 1)];
        data.penalty_overrides_ncs = vec![NcsPenaltyOverrideRow {
            source_id: EntityId::from(1),
            stage_id: 0,
            curtailment_cost: None,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(!ctx.has_errors(), "valid NCS ref should not produce errors");
    }

    /// `LoadFactorEntry` with a non-existent `bus_id` produces 1
    /// `InvalidReference` error for `scenarios/load_factors.json`.
    #[test]
    fn test_load_factors_invalid_bus_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // bus 999 does not exist
        data.load_factors = vec![LoadFactorEntry {
            bus_id: EntityId::from(999),
            stage_id: 0,
            block_factors: vec![BlockFactor {
                block_id: 0,
                factor: 1.0,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("999"));
        assert!(inv[0].message.contains("bus_id"));
        assert!(
            inv[0]
                .entity
                .as_deref()
                .unwrap_or("")
                .contains("LoadFactorEntry")
        );
    }

    /// `LoadFactorEntry` with a non-existent `stage_id` produces 1
    /// `InvalidReference` error for `scenarios/load_factors.json`.
    #[test]
    fn test_load_factors_invalid_stage_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // stage 999 does not exist; bus 1 does exist (added by make_minimal_case)
        data.load_factors = vec![LoadFactorEntry {
            bus_id: EntityId::from(1),
            stage_id: 999,
            block_factors: vec![BlockFactor {
                block_id: 0,
                factor: 1.0,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("999"));
        assert!(inv[0].message.contains("stage_id"));
        assert!(
            inv[0]
                .entity
                .as_deref()
                .unwrap_or("")
                .contains("LoadFactorEntry")
        );
    }

    /// `LoadFactorEntry` with valid `bus_id` and `stage_id` produces no
    /// `InvalidReference` errors from the load-factors rules.
    #[test]
    fn test_load_factors_valid_refs_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // bus 1 and stage 0 both exist in the minimal case
        data.load_factors = vec![LoadFactorEntry {
            bus_id: EntityId::from(1),
            stage_id: 0,
            block_factors: vec![BlockFactor {
                block_id: 0,
                factor: 1.0,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid load_factors refs should produce no errors"
        );
    }

    /// Valid `NcsBoundsRow` with an existing NCS ID and valid stage produces no errors.
    #[test]
    fn test_ncs_bounds_valid_refs_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.non_controllable_sources = vec![make_ncs(1, 1)];
        data.ncs_bounds = vec![NcsBoundsRow {
            ncs_id: EntityId::from(1),
            stage_id: 0,
            available_generation_mw: 50.0,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid NCS bounds should produce no errors"
        );
    }

    /// `NcsBoundsRow` with a non-existent NCS ID produces `InvalidReference`.
    #[test]
    fn test_ncs_bounds_invalid_ncs_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.ncs_bounds = vec![NcsBoundsRow {
            ncs_id: EntityId::from(999),
            stage_id: 0,
            available_generation_mw: 50.0,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .filter(|e| e.file.to_str().unwrap_or("").contains("ncs_bounds"))
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("999"));
    }

    /// `NcsBoundsRow` with negative `available_generation_mw` produces `InvalidValue`.
    #[test]
    fn test_ncs_bounds_negative_available_generation() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.non_controllable_sources = vec![make_ncs(1, 1)];
        data.ncs_bounds = vec![NcsBoundsRow {
            ncs_id: EntityId::from(1),
            stage_id: 0,
            available_generation_mw: -10.0,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("negative"));
    }

    /// Valid `NcsFactorEntry` with an existing NCS ID and valid stage produces no errors.
    #[test]
    fn test_ncs_factors_valid_refs_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.non_controllable_sources = vec![make_ncs(1, 1)];
        data.non_controllable_factors = vec![NcsFactorEntry {
            ncs_id: EntityId::from(1),
            stage_id: 0,
            block_factors: vec![BlockFactor {
                block_id: 0,
                factor: 1.0,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid NCS factors should produce no errors"
        );
    }

    /// `NcsFactorEntry` with a non-existent NCS ID produces `InvalidReference`.
    #[test]
    fn test_ncs_factors_invalid_ncs_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.non_controllable_factors = vec![NcsFactorEntry {
            ncs_id: EntityId::from(999),
            stage_id: 0,
            block_factors: vec![BlockFactor {
                block_id: 0,
                factor: 1.0,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .filter(|e| {
                e.file
                    .to_str()
                    .unwrap_or("")
                    .contains("non_controllable_factors")
            })
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("999"));
    }

    /// `NcsFactorEntry` with an invalid `stage_id` produces `InvalidReference`.
    #[test]
    fn test_ncs_factors_invalid_stage_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.non_controllable_sources = vec![make_ncs(1, 1)];
        data.non_controllable_factors = vec![NcsFactorEntry {
            ncs_id: EntityId::from(1),
            stage_id: 999,
            block_factors: vec![BlockFactor {
                block_id: 0,
                factor: 1.0,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .filter(|e| {
                e.file
                    .to_str()
                    .unwrap_or("")
                    .contains("non_controllable_factors")
            })
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("999"));
    }

    /// `NcsFactorEntry` with a negative block factor produces `InvalidValue`.
    #[test]
    fn test_ncs_factors_negative_factor() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.non_controllable_sources = vec![make_ncs(1, 1)];
        data.non_controllable_factors = vec![NcsFactorEntry {
            ncs_id: EntityId::from(1),
            stage_id: 0,
            block_factors: vec![BlockFactor {
                block_id: 0,
                factor: -0.5,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("negative"));
    }

    // ── AnticipatedDecision referential validation ─────────────────────

    /// A constraint with `anticipated_decision(99)` where Thermal 99 does
    /// not exist produces `ErrorKind::InvalidReference` naming Thermal 99 and
    /// including the constraint id in the context.
    #[test]
    fn test_anticipated_decision_unknown_thermal_ref() {
        use cobre_core::{
            ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
            VariableRef,
        };

        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        // Build a generic constraint referencing Thermal 99, which does not exist.
        let gc = GenericConstraint {
            id: EntityId::from(1),
            name: "test_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::AnticipatedDecision {
                        thermal_id: EntityId::from(99),
                    },
                )],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };
        data.generic_constraints = vec![gc];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors(), "expected referential errors");

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {inv:?}"
        );
        assert!(
            inv[0].message.contains("99"),
            "error message must name Thermal 99, got: {}",
            inv[0].message
        );
        assert!(
            inv[0].message.contains("Thermal"),
            "error message must include 'Thermal', got: {}",
            inv[0].message
        );
    }

    #[test]
    fn test_hydro_inflow_unknown_hydro_ref() {
        use cobre_core::{
            ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
            VariableRef,
        };

        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        // Build a generic constraint referencing Hydro 99, which does not exist.
        let gc = GenericConstraint {
            id: EntityId::from(1),
            name: "test_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroInflow {
                        hydro_id: EntityId::from(99),
                        block_id: None,
                    },
                )],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };
        data.generic_constraints = vec![gc];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors(), "expected referential errors");

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {inv:?}"
        );
        assert!(
            inv[0].message.contains("non-existent Hydro 99"),
            "error message must name non-existent Hydro 99, got: {}",
            inv[0].message
        );
    }

    /// A block-qualified `hydro_inflow(99, 0)` referencing a non-existent hydro is also
    /// flagged: the hydro-bearing arm's `..` pattern absorbs `block_id`, so the
    /// `Some(_)` form validates identically to the `None` form.
    #[test]
    fn test_hydro_inflow_with_block_unknown_hydro_ref() {
        use cobre_core::{
            ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
            VariableRef,
        };

        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        let gc = GenericConstraint {
            id: EntityId::from(1),
            name: "test_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroInflow {
                        hydro_id: EntityId::from(99),
                        block_id: Some(0),
                    },
                )],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };
        data.generic_constraints = vec![gc];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors(), "expected referential errors");

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {inv:?}"
        );
        assert!(
            inv[0].message.contains("non-existent Hydro 99"),
            "error message must name non-existent Hydro 99, got: {}",
            inv[0].message
        );
    }

    /// A constraint referencing `hydro_storage_initial(99, 0)` where Hydro 99 does
    /// not exist produces exactly one `InvalidReference` naming Hydro 99. The
    /// hydro arm's `..` pattern absorbs `block_id`, matching `HydroStorage`.
    #[test]
    fn test_hydro_storage_initial_unknown_hydro_ref() {
        use cobre_core::{
            ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
            VariableRef,
        };

        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        let gc = GenericConstraint {
            id: EntityId::from(1),
            name: "test_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroStorageInitial {
                        hydro_id: EntityId::from(99),
                        block_id: Some(0),
                    },
                )],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };
        data.generic_constraints = vec![gc];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors(), "expected referential errors");

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {inv:?}"
        );
        assert!(
            inv[0].message.contains("non-existent Hydro 99"),
            "error message must name non-existent Hydro 99, got: {}",
            inv[0].message
        );
    }

    // ── `bus_id` selector referential validation ───────────────────────

    /// Hydro 7's two unit groups (ids 20, 21 at positions 0, 1) sit on buses 1
    /// and 4; Hydro 8 has one group (id 30) on bus 9. Buses 1, 4, and 9 are all
    /// declared; bus 777 is declared nowhere. Group ids, positions, and bus ids
    /// are mutually disjoint so a group-id or position lookup cannot pass as a
    /// bus lookup.
    fn make_split_plant_bus_selector_fixture(dir: &TempDir) -> ParsedData {
        let mut data = parse_case(dir);
        data.buses.push(Bus {
            id: EntityId::from(4),
            name: "BUS_4".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![],
            excess_cost: 100.0,
        });
        data.buses.push(Bus {
            id: EntityId::from(9),
            name: "BUS_9".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![],
            excess_cost: 100.0,
        });

        let mut hydro7 = make_hydro(7);
        hydro7.unit_groups = vec![
            make_unit_group(20, 1, 0.0, 100.0, 0.0, 100.0),
            make_unit_group(21, 4, 0.0, 100.0, 0.0, 100.0),
        ];

        let mut hydro8 = make_hydro(8);
        hydro8.unit_groups = vec![make_unit_group(30, 9, 0.0, 100.0, 0.0, 100.0)];

        data.hydros = vec![hydro7, hydro8];
        data
    }

    /// A `hydro_turbined(7, bus=9)` term names bus 9, which genuinely exists in
    /// the system (Hydro 8 has a group there) but on which Hydro 7 has no unit
    /// group. The per-plant check rejects it even though bus 9 is a declared bus.
    #[test]
    fn test_generic_constraint_unknown_bus_selector_rejected() {
        use cobre_core::{
            ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
            VariableRef,
        };

        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = make_split_plant_bus_selector_fixture(&dir);

        let gc = GenericConstraint {
            id: EntityId::from(1),
            name: "test_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroTurbined {
                        hydro_id: EntityId::from(7),
                        block_id: None,
                        bus_id: Some(EntityId::from(9)),
                    },
                )],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };
        data.generic_constraints = vec![gc];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .filter(|e| e.file == std::path::Path::new("system/generic_constraints.json"))
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
        assert!(
            inv[0].message.contains("bus 9"),
            "message must name bus 9, got: {}",
            inv[0].message
        );
        assert!(
            inv[0].message.contains("Hydro 7"),
            "message must name Hydro 7, got: {}",
            inv[0].message
        );
        assert!(
            inv[0].message.contains("GenericConstraint 1 term[0]"),
            "message must name the constraint's term label, got: {}",
            inv[0].message
        );
    }

    /// The same fixture with the term's `bus_id` changed to `Some(EntityId(4))`
    /// (a real bus of Hydro 7's second group) and a second constraint whose
    /// `HydroGeneration` term carries `bus_id: None` (the plant-wide reference)
    /// produce no finding: `None` stays silent, and resolving membership by
    /// bus rather than group id accepts the valid selector.
    #[test]
    fn test_generic_constraint_valid_bus_selector_and_none_accepted() {
        use cobre_core::{
            ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
            VariableRef,
        };

        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = make_split_plant_bus_selector_fixture(&dir);

        let gc_turbined = GenericConstraint {
            id: EntityId::from(1),
            name: "test_constraint_turbined".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroTurbined {
                        hydro_id: EntityId::from(7),
                        block_id: None,
                        bus_id: Some(EntityId::from(4)),
                    },
                )],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };
        let gc_generation = GenericConstraint {
            id: EntityId::from(2),
            name: "test_constraint_generation".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroGeneration {
                        hydro_id: EntityId::from(7),
                        block_id: None,
                        bus_id: None,
                    },
                )],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };
        data.generic_constraints = vec![gc_turbined, gc_generation];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .filter(|e| e.file == std::path::Path::new("system/generic_constraints.json"))
            .collect();
        assert!(
            inv.is_empty(),
            "expected no InvalidReference against generic_constraints.json, got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
    }

    /// A `hydro_generation(99, bus=1)` term where Hydro 99 does not exist emits
    /// only the `hydro_id` finding — the bus half is unanswerable without a
    /// plant and must not also fire.
    #[test]
    fn test_generic_constraint_unknown_hydro_with_selector_emits_one_finding() {
        use cobre_core::{
            ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
            VariableRef,
        };

        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        let gc = GenericConstraint {
            id: EntityId::from(1),
            name: "test_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroGeneration {
                        hydro_id: EntityId::from(99),
                        block_id: None,
                        bus_id: Some(EntityId::from(1)),
                    },
                )],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };
        data.generic_constraints = vec![gc];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .filter(|e| e.file == std::path::Path::new("system/generic_constraints.json"))
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
        assert!(
            inv[0].message.contains("Hydro 99"),
            "message must name Hydro 99, got: {}",
            inv[0].message
        );
        assert!(
            !inv[0].message.contains("bus"),
            "message must not carry a bus finding, got: {}",
            inv[0].message
        );
    }

    /// A bus that exists in no plant's group set and is not a declared bus at
    /// all is rejected with exactly one finding: the per-plant check subsumes
    /// the "bus does not exist" case, so no second finding is added.
    #[test]
    fn test_generic_constraint_nonexistent_bus_selector_emits_one_finding() {
        use cobre_core::{
            ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
            VariableRef,
        };

        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = make_split_plant_bus_selector_fixture(&dir);

        let gc = GenericConstraint {
            id: EntityId::from(1),
            name: "test_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroTurbined {
                        hydro_id: EntityId::from(7),
                        block_id: None,
                        bus_id: Some(EntityId::from(777)),
                    },
                )],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };
        data.generic_constraints = vec![gc];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .filter(|e| e.file == std::path::Path::new("system/generic_constraints.json"))
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
        assert!(
            inv[0].message.contains("bus 777"),
            "message must name bus 777, got: {}",
            inv[0].message
        );
        assert!(
            inv[0].message.contains("Hydro 7"),
            "message must name Hydro 7, got: {}",
            inv[0].message
        );
    }
}
