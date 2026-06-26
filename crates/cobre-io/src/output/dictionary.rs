//! [`write_dictionaries`] writes five self-documenting files to
//! `training/dictionaries/`:
//!
//! - `codes.json` — categorical code mappings (operative state, bound type, etc.)
//! - `entities.csv` — one row per entity with id, name, bus, and system columns
//! - `variables.csv` — one row per column across all output Parquet schemas
//! - `bounds.parquet` — per-entity, per-stage bound values
//! - `state_dictionary.json` — state space structure (storage + inflow lags)

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Builder, Int8Builder, Int32Builder, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use cobre_core::System;

use crate::Config;
use crate::output::atomic::{write_bytes_atomic, write_parquet_atomic};
use crate::output::error::OutputError;
use crate::output::parquet_config::ParquetWriterConfig;
use crate::output::schemas::{
    buses_schema, contracts_schema, convergence_schema, costs_schema, exchanges_schema,
    generic_violations_schema, hydro_energy_productivity_schema, hydros_schema, inflow_lags_schema,
    iteration_timing_schema, non_controllables_schema, pumping_stations_schema, rank_timing_schema,
    retry_histogram_schema, row_selection_schema, solver_iterations_schema, thermals_schema,
};

// ─── Entity type codes (SS3) ─────────────────────────────────────────────────

const ENTITY_TYPE_HYDRO: i8 = 0;
const ENTITY_TYPE_THERMAL: i8 = 1;
const ENTITY_TYPE_BUS: i8 = 2;
const ENTITY_TYPE_LINE: i8 = 3;
const ENTITY_TYPE_PUMPING_STATION: i8 = 4;
const ENTITY_TYPE_CONTRACT: i8 = 5;
const ENTITY_TYPE_NON_CONTROLLABLE: i8 = 7;

// ─── Bound type codes (SS3) ──────────────────────────────────────────────────

const BOUND_STORAGE_MIN: i8 = 0;
const BOUND_STORAGE_MAX: i8 = 1;
const BOUND_TURBINED_MIN: i8 = 2;
const BOUND_TURBINED_MAX: i8 = 3;
const BOUND_OUTFLOW_MIN: i8 = 4;
const BOUND_OUTFLOW_MAX: i8 = 5;
const BOUND_GENERATION_MIN: i8 = 6;
const BOUND_GENERATION_MAX: i8 = 7;
const BOUND_FLOW_MIN: i8 = 8;
const BOUND_FLOW_MAX: i8 = 9;

// ─── Public entry point ───────────────────────────────────────────────────────

/// Write all five dictionary files (see module doc) to `path`.
///
/// `path` must point to an already-created dictionaries directory (typically
/// `output_dir/training/dictionaries/`).
///
/// # Errors
///
/// - [`OutputError::IoError`] for file write failures.
/// - [`OutputError::SerializationError`] for Arrow `RecordBatch` construction
///   failures in `bounds.parquet`.
/// - [`OutputError::ManifestError`] for JSON serialization failures.
pub fn write_dictionaries(
    path: &Path,
    system: &System,
    _config: &Config,
) -> Result<(), OutputError> {
    write_codes_json(path)?;
    write_entities_csv(path, system)?;
    write_variables_csv(path)?;
    write_bounds_parquet(path, system, &ParquetWriterConfig::default())?;
    write_state_dictionary_json(path, system)?;
    Ok(())
}

// ─── codes.json ──────────────────────────────────────────────────────────────

/// Write `codes.json`: code-to-label mappings for every categorical integer
/// code used in the Parquet output files.
fn write_codes_json(path: &Path) -> Result<(), OutputError> {
    let generated_at = chrono::Utc::now().to_rfc3339();

    let content = serde_json::json!({
        "version": "1.0",
        "generated_at": generated_at,
        "operative_state": {
            "0": "deactivated",
            "1": "maintenance",
            "2": "operating",
            "3": "saturated"
        },
        "storage_binding": {
            "0": "none",
            "1": "below_minimum",
            "2": "above_maximum",
            "3": "both"
        },
        "contract_type": {
            "0": "import",
            "1": "export"
        },
        "entity_type": {
            "0": "hydro",
            "1": "thermal",
            "2": "bus",
            "3": "line",
            "4": "pumping_station",
            "5": "contract",
            "7": "non_controllable"
        },
        "bound_type": {
            "0": "storage_min",
            "1": "storage_max",
            "2": "turbined_min",
            "3": "turbined_max",
            "4": "outflow_min",
            "5": "outflow_max",
            "6": "generation_min",
            "7": "generation_max",
            "8": "flow_min",
            "9": "flow_max"
        }
    });

    let json_str =
        serde_json::to_string_pretty(&content).map_err(|e| OutputError::ManifestError {
            manifest_type: "codes.json".to_string(),
            message: e.to_string(),
        })?;

    write_bytes_atomic(path.join("codes.json").as_path(), json_str.as_bytes())
}

// ─── entities.csv ────────────────────────────────────────────────────────────

/// Write `entities.csv`, one row per entity, ordered by `entity_type_code`
/// ascending then by entity ID (canonical accessor order).
fn write_entities_csv(path: &Path, system: &System) -> Result<(), OutputError> {
    let file_path = path.join("entities.csv");
    let mut wtr = csv::Writer::from_path(&file_path)
        .map_err(|e| OutputError::io(&file_path, std::io::Error::other(e)))?;

    wtr.write_record([
        "entity_type_code",
        "entity_id",
        "name",
        "bus_id",
        "system_id",
    ])
    .map_err(|e| OutputError::io(&file_path, std::io::Error::other(e)))?;

    for h in system.hydros() {
        wtr.write_record(&[
            ENTITY_TYPE_HYDRO.to_string(),
            h.id.0.to_string(),
            h.name.clone(),
            h.bus_id.0.to_string(),
            "0".to_string(),
        ])
        .map_err(|e| OutputError::io(&file_path, std::io::Error::other(e)))?;
    }

    for t in system.thermals() {
        wtr.write_record(&[
            ENTITY_TYPE_THERMAL.to_string(),
            t.id.0.to_string(),
            t.name.clone(),
            t.bus_id.0.to_string(),
            "0".to_string(),
        ])
        .map_err(|e| OutputError::io(&file_path, std::io::Error::other(e)))?;
    }

    // Buses (code 2) — bus_id = own id
    for b in system.buses() {
        wtr.write_record(&[
            ENTITY_TYPE_BUS.to_string(),
            b.id.0.to_string(),
            b.name.clone(),
            b.id.0.to_string(),
            "0".to_string(),
        ])
        .map_err(|e| OutputError::io(&file_path, std::io::Error::other(e)))?;
    }

    // Lines (code 3) — bus_id = -1 (connects two buses)
    for l in system.lines() {
        wtr.write_record(&[
            ENTITY_TYPE_LINE.to_string(),
            l.id.0.to_string(),
            l.name.clone(),
            (-1_i32).to_string(),
            "0".to_string(),
        ])
        .map_err(|e| OutputError::io(&file_path, std::io::Error::other(e)))?;
    }

    for p in system.pumping_stations() {
        wtr.write_record(&[
            ENTITY_TYPE_PUMPING_STATION.to_string(),
            p.id.0.to_string(),
            p.name.clone(),
            p.bus_id.0.to_string(),
            "0".to_string(),
        ])
        .map_err(|e| OutputError::io(&file_path, std::io::Error::other(e)))?;
    }

    for c in system.contracts() {
        wtr.write_record(&[
            ENTITY_TYPE_CONTRACT.to_string(),
            c.id.0.to_string(),
            c.name.clone(),
            c.bus_id.0.to_string(),
            "0".to_string(),
        ])
        .map_err(|e| OutputError::io(&file_path, std::io::Error::other(e)))?;
    }

    for n in system.non_controllable_sources() {
        wtr.write_record(&[
            ENTITY_TYPE_NON_CONTROLLABLE.to_string(),
            n.id.0.to_string(),
            n.name.clone(),
            n.bus_id.0.to_string(),
            "0".to_string(),
        ])
        .map_err(|e| OutputError::io(&file_path, std::io::Error::other(e)))?;
    }

    wtr.flush().map_err(|e| OutputError::io(&file_path, e))?;

    Ok(())
}

// ─── variables.csv ───────────────────────────────────────────────────────────

/// Write `variables.csv`, one row per column across every output schema, grouped
/// by file and ordered by column position within each schema.
fn write_variables_csv(path: &Path) -> Result<(), OutputError> {
    let file_path = path.join("variables.csv");
    let mut wtr = csv::Writer::from_path(&file_path)
        .map_err(|e| OutputError::io(&file_path, std::io::Error::other(e)))?;

    wtr.write_record(["file", "column", "type", "unit", "description", "nullable"])
        .map_err(|e| OutputError::io(&file_path, std::io::Error::other(e)))?;

    let schemas: &[(&str, arrow::datatypes::Schema)] = &[
        ("costs", costs_schema()),
        ("hydros", hydros_schema()),
        ("thermals", thermals_schema()),
        ("exchanges", exchanges_schema()),
        ("buses", buses_schema()),
        ("pumping_stations", pumping_stations_schema()),
        ("contracts", contracts_schema()),
        ("non_controllables", non_controllables_schema()),
        ("inflow_lags", inflow_lags_schema()),
        ("generic_violations", generic_violations_schema()),
        ("convergence", convergence_schema()),
        ("iteration_timing", iteration_timing_schema()),
        ("rank_timing", rank_timing_schema()),
        ("cut_selection", row_selection_schema()),
        ("solver_iterations", solver_iterations_schema()),
        ("retry_histogram", retry_histogram_schema()),
        (
            "hydro_energy_productivity",
            hydro_energy_productivity_schema(),
        ),
    ];

    for (schema_name, schema) in schemas {
        for field in schema.fields() {
            let type_str = arrow_type_str(field.data_type());
            let unit = unit_for(schema_name, field.name());
            let description = description_for(schema_name, field.name());
            let nullable = if field.is_nullable() { "true" } else { "false" };

            wtr.write_record([
                *schema_name,
                field.name().as_str(),
                type_str,
                unit,
                description,
                nullable,
            ])
            .map_err(|e| OutputError::io(&file_path, std::io::Error::other(e)))?;
        }
    }

    wtr.flush().map_err(|e| OutputError::io(&file_path, e))?;

    Ok(())
}

/// Map an Arrow `DataType` to the string representation used in `variables.csv`.
fn arrow_type_str(dt: &DataType) -> &'static str {
    match dt {
        DataType::Int8 => "i8",
        DataType::Int32 => "i32",
        DataType::Int64 => "i64",
        DataType::UInt32 => "u32",
        DataType::UInt64 => "u64",
        DataType::Float64 => "f64",
        DataType::Boolean => "bool",
        DataType::Utf8 => "string",
        _ => "unknown",
    }
}

/// Return the physical unit string for a given (file, column) pair.
///
/// Returns `""` for dimensionless columns or columns without a defined unit.
// Rationale: one authoritative (file, column) → unit lookup table; identical
// arms are intentional (same unit recurs across schemas), so splitting or
// collapsing arms would degrade it as a catalog.
#[allow(clippy::too_many_lines, clippy::match_same_arms)]
fn unit_for(file: &str, column: &str) -> &'static str {
    match column {
        "stage_id"
        | "block_id"
        | "iteration"
        | "rank"
        | "forward_passes"
        | "scenarios_processed" => return "",
        "generation_mw"
        | "available_mw"
        | "curtailment_mw"
        | "direct_flow_mw"
        | "reverse_flow_mw"
        | "net_flow_mw"
        | "losses_mw"
        | "load_mw"
        | "deficit_mw"
        | "excess_mw"
        | "spot_price"
        | "pumped_flow_m3s"
        | "power_consumption_mw"
        | "anticipated_committed_mw"
        | "anticipated_decision_mw"
        | "power_mw" => return "MW",
        "generation_mwh"
        | "curtailment_mwh"
        | "net_flow_mwh"
        | "losses_mwh"
        | "load_mwh"
        | "deficit_mwh"
        | "excess_mwh"
        | "energy_consumption_mwh"
        | "energy_mwh" => return "MWh",
        "turbined_m3s"
        | "spillage_m3s"
        | "outflow_m3s"
        | "evaporation_m3s"
        | "diverted_inflow_m3s"
        | "diverted_outflow_m3s"
        | "incremental_inflow_m3s"
        | "inflow_m3s"
        | "turbined_slack_m3s"
        | "outflow_slack_below_m3s"
        | "outflow_slack_above_m3s"
        | "evaporation_violation_pos_m3s"
        | "evaporation_violation_neg_m3s"
        | "inflow_nonnegativity_slack_m3s"
        | "water_withdrawal_violation_pos_m3s"
        | "water_withdrawal_violation_neg_m3s"
        | "pumped_volume_hm3" => return "m3/s",
        "storage_initial_hm3"
        | "storage_final_hm3"
        | "storage_violation_below_hm3"
        | "filling_target_violation_hm3" => return "hm3",
        "total_cost"
        | "immediate_cost"
        | "future_cost"
        | "thermal_cost"
        | "anticipated_thermal_cost"
        | "contract_cost"
        | "deficit_cost"
        | "excess_cost"
        | "storage_violation_cost"
        | "filling_target_cost"
        | "hydro_violation_cost"
        | "outflow_violation_below_cost"
        | "outflow_violation_above_cost"
        | "turbined_violation_cost"
        | "generation_violation_cost"
        | "evaporation_violation_cost"
        | "withdrawal_violation_cost"
        | "inflow_penalty_cost"
        | "generic_violation_cost"
        | "spillage_cost"
        | "turbined_cost"
        | "curtailment_cost"
        | "exchange_cost"
        | "pumping_cost"
        | "generation_cost"
        | "total_cost_convergence"
        | "pumping_cost_csv"
        | "price_per_mwh"
        | "slack_cost" => return "$",
        "time_forward_ms"
        | "time_backward_ms"
        | "time_total_ms"
        | "forward_solve_ms"
        | "forward_sample_ms"
        | "backward_solve_ms"
        | "backward_cut_ms"
        | "cut_selection_ms"
        | "mpi_allreduce_ms"
        | "mpi_broadcast_ms"
        | "io_write_ms"
        | "state_exchange_ms"
        | "cut_batch_build_ms"
        | "bwd_setup_ms"
        | "bwd_load_imbalance_ms"
        | "bwd_scheduling_overhead_ms"
        | "fwd_setup_ms"
        | "fwd_load_imbalance_ms"
        | "fwd_scheduling_overhead_ms"
        | "overhead_ms"
        | "lazy_scoring_ms"
        | "forward_time_ms"
        | "backward_time_ms"
        | "communication_time_ms"
        | "idle_time_ms"
        | "solve_time_ms"
        | "load_model_time_ms"
        | "set_bounds_time_ms"
        | "basis_set_time_ms"
        | "selection_time_ms" => return "ms",
        _ => {}
    }
    match (file, column) {
        (_, "total_cost" | "pumping_cost" | "spillage_cost" | "exchange_cost") => "$",
        ("hydros", "water_value_per_hm3") => "$/hm3",
        ("hydros", "equivalent_productivity_mw_per_m3s") => "MW/(m3/s)",
        ("hydros", "accumulated_productivity_mw_per_m3s") => "MW/(m3/s)",
        ("hydros", "incremental_inflow_energy_mw") => "MW",
        ("hydros", "stored_energy_initial_mwh") => "MWh",
        ("hydros", "stored_energy_final_mwh") => "MWh",
        ("hydros", "generation_slack_mw") => "MW",
        _ => "",
    }
}

/// Return a short description for a given (file, column) pair.
///
/// Returns `""` for columns without a registered description.
// Rationale: one authoritative (file, column) → description lookup table;
// identical arms are intentional, so collapsing them would hide additions and
// per-schema divergence.
#[allow(clippy::too_many_lines, clippy::match_same_arms)]
fn description_for(file: &str, column: &str) -> &'static str {
    match (file, column) {
        ("costs", "stage_id") => "Stage index",
        ("costs", "block_id") => "Block index within stage (nullable)",
        ("costs", "total_cost") => "Total stage cost",
        ("costs", "immediate_cost") => "Immediate (operation) cost",
        ("costs", "future_cost") => "Expected future cost (envelope value)",
        ("costs", "discount_factor") => "Discount factor applied to this stage",
        ("costs", "thermal_cost") => "Total thermal generation cost",
        ("costs", "anticipated_thermal_cost") => {
            "Total anticipated (forward-committed) thermal generation cost"
        }
        ("costs", "contract_cost") => "Total contract cost",
        ("costs", "deficit_cost") => "Total load-deficit penalty cost",
        ("costs", "excess_cost") => "Total excess-generation cost",
        ("costs", "storage_violation_cost") => "Total storage violation penalty",
        ("costs", "filling_target_cost") => "Total filling-target violation cost",
        ("costs", "hydro_violation_cost") => "Total hydro constraint violation cost",
        ("costs", "outflow_violation_below_cost") => "Cost of minimum outflow violations",
        ("costs", "outflow_violation_above_cost") => "Cost of maximum outflow violations",
        ("costs", "turbined_violation_cost") => "Cost of minimum turbining violations",
        ("costs", "generation_violation_cost") => "Cost of minimum generation violations",
        ("costs", "evaporation_violation_cost") => "Cost of evaporation constraint violations",
        ("costs", "withdrawal_violation_cost") => "Cost of water withdrawal constraint violations",
        ("costs", "inflow_penalty_cost") => "Total inflow non-negativity penalty",
        ("costs", "generic_violation_cost") => "Total generic constraint violation cost",
        ("costs", "spillage_cost") => "Total spillage regularization cost",
        ("costs", "turbined_cost") => "Total turbined regularization cost",
        ("costs", "curtailment_cost") => "Total curtailment cost",
        ("costs", "exchange_cost") => "Total exchange regularization cost",
        ("costs", "pumping_cost") => "Total pumping cost",
        ("hydros", "stage_id") => "Stage index",
        ("hydros", "block_id") => "Block index within stage (nullable)",
        ("hydros", "hydro_id") => "Hydro plant identifier",
        ("hydros", "turbined_m3s") => "Turbined flow",
        ("hydros", "spillage_m3s") => "Spilled flow",
        ("hydros", "outflow_m3s") => "Total outflow (turbined + spilled)",
        ("hydros", "evaporation_m3s") => "Evaporation loss (nullable)",
        ("hydros", "diverted_inflow_m3s") => "Diverted inflow received (nullable)",
        ("hydros", "diverted_outflow_m3s") => "Diverted outflow sent (nullable)",
        ("hydros", "incremental_inflow_m3s") => "Incremental (local) inflow",
        ("hydros", "inflow_m3s") => "Total inflow including upstream contributions",
        ("hydros", "storage_initial_hm3") => "Reservoir storage at start of stage",
        ("hydros", "storage_final_hm3") => "Reservoir storage at end of stage",
        ("hydros", "generation_mw") => "Hydro generation",
        ("hydros", "generation_mwh") => "Hydro energy generated",
        ("hydros", "equivalent_productivity_mw_per_m3s") => {
            "Equivalent productivity `ρ_eq` (always populated)"
        }
        ("hydros", "accumulated_productivity_mw_per_m3s") => {
            "Accumulated productivity `ρ_acum` along downstream cascade"
        }
        ("hydros", "incremental_inflow_energy_mw") => {
            "Incremental natural energy inflow (`ρ_acum` · incremental inflow)"
        }
        ("hydros", "stored_energy_initial_mwh") => "Stored energy at start of block",
        ("hydros", "stored_energy_final_mwh") => "Stored energy at end of block",
        ("hydros", "spillage_cost") => "Spillage regularization cost",
        ("hydros", "water_value_per_hm3") => "Marginal water value",
        ("hydros", "storage_binding_code") => "Storage bound binding code",
        ("hydros", "operative_state_code") => "Operative state code",
        ("hydros", "turbined_slack_m3s") => "Turbined minimum slack",
        ("hydros", "outflow_slack_below_m3s") => "Outflow below-minimum slack",
        ("hydros", "outflow_slack_above_m3s") => "Outflow above-maximum slack",
        ("hydros", "generation_slack_mw") => "Generation minimum slack",
        ("hydros", "storage_violation_below_hm3") => "Storage below dead-volume violation",
        ("hydros", "filling_target_violation_hm3") => "Filling target violation",
        ("hydros", "evaporation_violation_pos_m3s") => "Over-evaporation constraint violation",
        ("hydros", "evaporation_violation_neg_m3s") => "Under-evaporation constraint violation",
        ("hydros", "inflow_nonnegativity_slack_m3s") => "Inflow non-negativity slack",
        ("hydros", "water_withdrawal_violation_pos_m3s") => "Over-withdrawal constraint violation",
        ("hydros", "water_withdrawal_violation_neg_m3s") => "Under-withdrawal constraint violation",
        ("thermals", "stage_id") => "Stage index",
        ("thermals", "block_id") => "Block index within stage (nullable)",
        ("thermals", "thermal_id") => "Thermal plant identifier",
        ("thermals", "generation_mw") => "Thermal generation",
        ("thermals", "generation_mwh") => "Thermal energy generated",
        ("thermals", "generation_cost") => "Thermal generation cost",
        ("thermals", "is_anticipated") => "Whether plant uses anticipated dispatch",
        ("thermals", "anticipated_committed_mw") => "Anticipated committed capacity (nullable)",
        ("thermals", "anticipated_decision_mw") => "Anticipated dispatch decision (nullable)",
        ("thermals", "operative_state_code") => "Operative state code",
        ("exchanges", "stage_id") => "Stage index",
        ("exchanges", "block_id") => "Block index within stage (nullable)",
        ("exchanges", "line_id") => "Transmission line identifier",
        ("exchanges", "direct_flow_mw") => "Flow in direct direction",
        ("exchanges", "reverse_flow_mw") => "Flow in reverse direction",
        ("exchanges", "net_flow_mw") => "Net flow (direct minus reverse)",
        ("exchanges", "net_flow_mwh") => "Net energy exchanged",
        ("exchanges", "losses_mw") => "Transmission losses",
        ("exchanges", "losses_mwh") => "Transmission energy losses",
        ("exchanges", "exchange_cost") => "Exchange regularization cost",
        ("exchanges", "operative_state_code") => "Operative state code",
        ("buses", "stage_id") => "Stage index",
        ("buses", "block_id") => "Block index within stage (nullable)",
        ("buses", "bus_id") => "Bus identifier",
        ("buses", "load_mw") => "Load demand",
        ("buses", "load_mwh") => "Load energy demand",
        ("buses", "deficit_mw") => "Unmet demand (deficit)",
        ("buses", "deficit_mwh") => "Unmet energy demand",
        ("buses", "excess_mw") => "Excess generation absorbed",
        ("buses", "excess_mwh") => "Excess energy absorbed",
        ("buses", "spot_price") => "Bus spot price (dual of balance constraint)",
        ("pumping_stations", "stage_id") => "Stage index",
        ("pumping_stations", "block_id") => "Block index within stage (nullable)",
        ("pumping_stations", "pumping_station_id") => "Pumping station identifier",
        ("pumping_stations", "pumped_flow_m3s") => "Pumped water flow",
        ("pumping_stations", "pumped_volume_hm3") => "Pumped water volume",
        ("pumping_stations", "power_consumption_mw") => "Electrical power consumed",
        ("pumping_stations", "energy_consumption_mwh") => "Electrical energy consumed",
        ("pumping_stations", "pumping_cost") => "Pumping operation cost",
        ("pumping_stations", "operative_state_code") => "Operative state code",
        ("contracts", "stage_id") => "Stage index",
        ("contracts", "block_id") => "Block index within stage (nullable)",
        ("contracts", "contract_id") => "Contract identifier",
        ("contracts", "power_mw") => "Contracted power",
        ("contracts", "energy_mwh") => "Contracted energy",
        ("contracts", "price_per_mwh") => "Effective contract price",
        ("contracts", "total_cost") => "Total contract cost",
        ("contracts", "operative_state_code") => "Operative state code",
        ("non_controllables", "stage_id") => "Stage index",
        ("non_controllables", "block_id") => "Block index within stage (nullable)",
        ("non_controllables", "non_controllable_id") => "Non-controllable source identifier",
        ("non_controllables", "generation_mw") => "Non-controllable generation dispatched",
        ("non_controllables", "generation_mwh") => "Non-controllable energy generated",
        ("non_controllables", "available_mw") => "Available generation capacity",
        ("non_controllables", "curtailment_mw") => "Curtailed generation",
        ("non_controllables", "curtailment_mwh") => "Curtailed energy",
        ("non_controllables", "curtailment_cost") => "Curtailment cost",
        ("non_controllables", "operative_state_code") => "Operative state code",
        ("inflow_lags", "stage_id") => "Stage index",
        ("inflow_lags", "hydro_id") => "Hydro plant identifier",
        ("inflow_lags", "lag_index") => "AR lag index (1-based)",
        ("inflow_lags", "inflow_m3s") => "Historical inflow for this lag",
        ("generic_violations", "stage_id") => "Stage index",
        ("generic_violations", "block_id") => "Block index within stage (nullable)",
        ("generic_violations", "constraint_id") => "Generic constraint identifier",
        ("generic_violations", "slack_value") => "Constraint slack value",
        ("generic_violations", "slack_cost") => "Constraint slack penalty cost",
        ("convergence", "iteration") => "Iteration number (1-based)",
        ("convergence", "lower_bound") => "Lower bound on the optimal value",
        ("convergence", "upper_bound_mean") => "Mean upper bound estimate (nullable)",
        ("convergence", "upper_bound_std") => "Std deviation of upper bound (nullable)",
        ("convergence", "gap_percent") => "Relative optimality gap in percent (nullable)",
        ("convergence", "cuts_added") => "Cuts added in this iteration",
        ("convergence", "cuts_removed") => "Cuts removed in this iteration",
        ("convergence", "cuts_active") => "Total active cuts after iteration",
        ("convergence", "time_forward_ms") => "Forward-pass wall-clock time",
        ("convergence", "time_backward_ms") => "Backward-pass wall-clock time",
        ("convergence", "time_total_ms") => "Total iteration wall-clock time",
        ("convergence", "forward_passes") => "Number of forward-pass scenarios",
        ("convergence", "lp_solves") => "Total LP solves in iteration",
        ("convergence", "mean_rows_in_lp") => {
            "Mean resident rows loaded per lazy-selection LP solve this iteration \
             (0 when no lazy selection ran)"
        }
        ("iteration_timing", "iteration") => "Iteration number (1-based)",
        ("iteration_timing", "rank") => {
            "MPI rank that produced this row. Always set; \
             single-rank runs use 0."
        }
        ("iteration_timing", "worker_id") => {
            "Worker thread index within the rank's pool. NULL on \
             rank-aggregated rows that carry rank-only timings (cut_selection, \
             mpi_allreduce, cut_sync, lower_bound, state_exchange, cut_batch_build, \
             load_imbalance / scheduling_overhead, overhead). Set on per-worker \
             rows that carry parallel-region timings (forward_wall, backward_wall, \
             fwd_setup, bwd_setup, lazy_scoring)."
        }
        ("iteration_timing", "forward_wall_ms") => "Forward pass wall-clock time",
        ("iteration_timing", "backward_wall_ms") => "Backward pass wall-clock time",
        ("iteration_timing", "cut_selection_ms") => "Row-selection time",
        ("iteration_timing", "mpi_allreduce_ms") => "MPI allreduce time",
        ("iteration_timing", "cut_sync_ms") => "Per-stage row-sync allgatherv time",
        ("iteration_timing", "lower_bound_ms") => "Lower bound evaluation time",
        ("iteration_timing", "state_exchange_ms") => "State exchange allgatherv time",
        ("iteration_timing", "cut_batch_build_ms") => "Row-batch assembly time",
        ("iteration_timing", "bwd_setup_ms") => "Thread-pool setup time before backward pass",
        ("iteration_timing", "bwd_load_imbalance_ms") => {
            "Estimated load imbalance across backward pass worker threads"
        }
        ("iteration_timing", "bwd_scheduling_overhead_ms") => {
            "Scheduling and synchronisation overhead in the backward pass"
        }
        ("iteration_timing", "fwd_setup_ms") => "Thread-pool setup time before forward pass",
        ("iteration_timing", "fwd_load_imbalance_ms") => {
            "Estimated load imbalance across forward pass worker threads"
        }
        ("iteration_timing", "fwd_scheduling_overhead_ms") => {
            "Scheduling and synchronisation overhead in the forward pass"
        }
        ("iteration_timing", "overhead_ms") => {
            "Residual iteration time not attributed to any phase"
        }
        ("iteration_timing", "lazy_scoring_ms") => {
            "Per-worker time spent in lazy candidate scoring inside the \
             lazy-selection solve; 0 when that solve path is not used. A \
             sub-component of the forward/backward phases."
        }
        ("rank_timing", "iteration") => "Iteration number (1-based)",
        ("rank_timing", "rank") => "MPI rank",
        ("rank_timing", "forward_time_ms") => "Forward-pass time for this rank",
        ("rank_timing", "backward_time_ms") => "Backward-pass time for this rank",
        ("rank_timing", "communication_time_ms") => "Communication time for this rank",
        ("rank_timing", "idle_time_ms") => "Idle time for this rank",
        ("rank_timing", "lp_solves") => "LP solves on this rank",
        ("rank_timing", "scenarios_processed") => "Scenarios processed by this rank",
        ("cut_selection", "iteration") => "Iteration number (1-based)",
        ("cut_selection", "stage") => "Stage index (0-based)",
        ("cut_selection", "cuts_populated") => "Total cuts ever generated at this stage",
        ("cut_selection", "cuts_active_before") => "Active cuts before selection ran",
        ("cut_selection", "cuts_deactivated") => "Cuts deactivated by selection",
        ("cut_selection", "cuts_reactivated") => {
            "Cuts reactivated (previously deactivated, re-entered LP)"
        }
        ("cut_selection", "cuts_active_after") => "Active cuts after selection",
        ("cut_selection", "selection_time_ms") => "Wall-clock time for selection at this stage",
        ("cut_selection", "budget_evicted") => {
            "Cuts evicted by budget enforcement (null when budget disabled)"
        }
        ("cut_selection", "active_after_budget") => {
            "Active cuts after budget enforcement (null when budget disabled)"
        }
        ("solver_iterations", "iteration") => "Iteration number (1-based) or scenario ID (0-based)",
        ("solver_iterations", "phase") => {
            "Solver phase (forward, backward, lower_bound, simulation)"
        }
        ("solver_iterations", "stage") => "Stage index (-1 for non-per-stage phases)",
        ("solver_iterations", "lp_solves") => "Number of LP solves",
        ("solver_iterations", "lp_successes") => "Solves that returned optimal",
        ("solver_iterations", "lp_retries") => "Solves requiring retry escalation",
        ("solver_iterations", "lp_failures") => "Solves that exhausted all retry levels",
        ("solver_iterations", "retry_attempts") => "Total retry attempts across all solves",
        ("solver_iterations", "basis_offered") => {
            "Number of warm-start solve calls (basis-offered)"
        }
        ("solver_iterations", "basis_consistency_failures") => {
            "Number of warm-start solve calls rejected because isBasisConsistent returned false"
        }
        ("solver_iterations", "simplex_iterations") => "Total simplex iterations",
        ("solver_iterations", "solve_time_ms") => "Cumulative solve wall-clock time",
        ("solver_iterations", "load_model_time_ms") => "Cumulative load_model call time",
        ("solver_iterations", "set_bounds_time_ms") => "Cumulative set_bounds call time",
        ("solver_iterations", "basis_set_time_ms") => "Cumulative set_basis call time",
        ("solver_iterations", "opening") => {
            "Opening (noise realization) index within the stage, for backward-pass \
             rows. NULL for forward, lower_bound, and simulation rows — these phases \
             do not have an opening dimension. Backward rows range 0..n_openings."
        }
        ("solver_iterations", "rank") => {
            "MPI rank that produced this row. NULL for rank-aggregated rows."
        }
        ("solver_iterations", "worker_id") => {
            "Worker thread index within the rank's pool that produced this row. \
             NULL for rank-aggregated rows."
        }
        ("retry_histogram", "iteration") => "Iteration number (1-based) or scenario ID (0-based)",
        ("retry_histogram", "phase") => "Solver phase (forward, backward, lower_bound, simulation)",
        ("retry_histogram", "stage") => "Stage index (-1 for non-per-stage phases)",
        ("retry_histogram", "retry_level") => "Retry escalation level (0-based)",
        ("retry_histogram", "count") => "Number of solves recovered at this level",
        _ => "",
    }
}

// ─── bounds.parquet ──────────────────────────────────────────────────────────

/// Write `bounds.parquet` with per-entity, per-stage resolved bounds: one row
/// per (entity, stage, `bound_type`), `block_id` always null.
// Rationale: all five entity classes share the pre-allocated Arrow column
// builders; per-entity helpers would force passing six builders through each or
// rebuilding per class, losing the single pre-estimated capacity allocation.
#[allow(clippy::too_many_lines)]
fn write_bounds_parquet(
    path: &Path,
    system: &System,
    config: &ParquetWriterConfig,
) -> Result<(), OutputError> {
    let schema = Arc::new(bounds_schema());
    let n_stages = system.bounds().n_stages();

    // Over-estimate: the optional max-outflow bound may be skipped per hydro.
    let capacity = (system.n_hydros() * 8
        + system.n_thermals() * 2
        + system.n_lines() * 2
        + system.n_pumping_stations() * 2
        + system.n_contracts() * 2)
        * n_stages;

    let mut entity_type_codes = Int8Builder::with_capacity(capacity);
    let mut entity_ids = Int32Builder::with_capacity(capacity);
    let mut stage_ids = Int32Builder::with_capacity(capacity);
    let mut block_ids = Int32Builder::with_capacity(capacity);
    let mut bound_type_codes = Int8Builder::with_capacity(capacity);
    let mut bound_values = Float64Builder::with_capacity(capacity);

    macro_rules! append_bound {
        ($entity_type:expr, $entity_id:expr, $stage_id:expr, $bound_type:expr, $value:expr) => {
            entity_type_codes.append_value($entity_type);
            entity_ids.append_value($entity_id);
            stage_ids.append_value($stage_id);
            block_ids.append_null(); // block_id always null (stage-level bounds)
            bound_type_codes.append_value($bound_type);
            bound_values.append_value($value);
        };
    }

    for (hydro_idx, hydro) in system.hydros().iter().enumerate() {
        let entity_id = hydro.id.0;
        for stage_idx in 0..n_stages {
            let stage_id = system.stages()[stage_idx].id;
            let b = system.bounds().hydro_bounds(hydro_idx, stage_idx);

            append_bound!(
                ENTITY_TYPE_HYDRO,
                entity_id,
                stage_id,
                BOUND_STORAGE_MIN,
                b.min_storage_hm3
            );
            append_bound!(
                ENTITY_TYPE_HYDRO,
                entity_id,
                stage_id,
                BOUND_STORAGE_MAX,
                b.max_storage_hm3
            );
            append_bound!(
                ENTITY_TYPE_HYDRO,
                entity_id,
                stage_id,
                BOUND_TURBINED_MIN,
                b.min_turbined_m3s
            );
            append_bound!(
                ENTITY_TYPE_HYDRO,
                entity_id,
                stage_id,
                BOUND_TURBINED_MAX,
                b.max_turbined_m3s
            );
            append_bound!(
                ENTITY_TYPE_HYDRO,
                entity_id,
                stage_id,
                BOUND_OUTFLOW_MIN,
                b.min_outflow_m3s
            );
            if let Some(max_outflow) = b.max_outflow_m3s {
                append_bound!(
                    ENTITY_TYPE_HYDRO,
                    entity_id,
                    stage_id,
                    BOUND_OUTFLOW_MAX,
                    max_outflow
                );
            }
            append_bound!(
                ENTITY_TYPE_HYDRO,
                entity_id,
                stage_id,
                BOUND_GENERATION_MIN,
                b.min_generation_mw
            );
            append_bound!(
                ENTITY_TYPE_HYDRO,
                entity_id,
                stage_id,
                BOUND_GENERATION_MAX,
                b.max_generation_mw
            );
        }
    }

    for (thermal_idx, thermal) in system.thermals().iter().enumerate() {
        let entity_id = thermal.id.0;
        for stage_idx in 0..n_stages {
            let stage_id = system.stages()[stage_idx].id;
            let b = system.bounds().thermal_bounds(thermal_idx, stage_idx);

            append_bound!(
                ENTITY_TYPE_THERMAL,
                entity_id,
                stage_id,
                BOUND_GENERATION_MIN,
                b.min_generation_mw
            );
            append_bound!(
                ENTITY_TYPE_THERMAL,
                entity_id,
                stage_id,
                BOUND_GENERATION_MAX,
                b.max_generation_mw
            );
        }
    }

    for (line_idx, line) in system.lines().iter().enumerate() {
        let entity_id = line.id.0;
        for stage_idx in 0..n_stages {
            let stage_id = system.stages()[stage_idx].id;
            let b = system.bounds().line_bounds(line_idx, stage_idx);

            append_bound!(
                ENTITY_TYPE_LINE,
                entity_id,
                stage_id,
                BOUND_FLOW_MIN,
                0.0_f64
            );
            append_bound!(
                ENTITY_TYPE_LINE,
                entity_id,
                stage_id,
                BOUND_FLOW_MAX,
                b.direct_mw
            );
        }
    }

    for (pumping_idx, pumping) in system.pumping_stations().iter().enumerate() {
        let entity_id = pumping.id.0;
        for stage_idx in 0..n_stages {
            let stage_id = system.stages()[stage_idx].id;
            let b = system.bounds().pumping_bounds(pumping_idx, stage_idx);

            append_bound!(
                ENTITY_TYPE_PUMPING_STATION,
                entity_id,
                stage_id,
                BOUND_FLOW_MIN,
                b.min_flow_m3s
            );
            append_bound!(
                ENTITY_TYPE_PUMPING_STATION,
                entity_id,
                stage_id,
                BOUND_FLOW_MAX,
                b.max_flow_m3s
            );
        }
    }

    for (contract_idx, contract) in system.contracts().iter().enumerate() {
        let entity_id = contract.id.0;
        for stage_idx in 0..n_stages {
            let stage_id = system.stages()[stage_idx].id;
            let b = system.bounds().contract_bounds(contract_idx, stage_idx);

            append_bound!(
                ENTITY_TYPE_CONTRACT,
                entity_id,
                stage_id,
                BOUND_FLOW_MIN,
                b.min_mw
            );
            append_bound!(
                ENTITY_TYPE_CONTRACT,
                entity_id,
                stage_id,
                BOUND_FLOW_MAX,
                b.max_mw
            );
        }
    }

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(entity_type_codes.finish()),
            Arc::new(entity_ids.finish()),
            Arc::new(stage_ids.finish()),
            Arc::new(block_ids.finish()),
            Arc::new(bound_type_codes.finish()),
            Arc::new(bound_values.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("bounds", e.to_string()))?;

    let parquet_path = path.join("bounds.parquet");
    write_parquet_atomic(&parquet_path, &batch, config)
}

/// Build the Arrow schema for `bounds.parquet`.
fn bounds_schema() -> Schema {
    Schema::new(vec![
        Field::new("entity_type_code", DataType::Int8, false),
        Field::new("entity_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("block_id", DataType::Int32, true),
        Field::new("bound_type_code", DataType::Int8, false),
        Field::new("bound_value", DataType::Float64, false),
    ])
}

// ─── state_dictionary.json ───────────────────────────────────────────────────

/// Write `state_dictionary.json`: one `"storage"` entry per hydro, an
/// `"inflow_lag"` entry per (hydro, lag) where AR order > 0, and an
/// `"anticipated_state"` entry per (slot, plant) when anticipated thermals exist.
fn write_state_dictionary_json(path: &Path, system: &System) -> Result<(), OutputError> {
    let mut state_variables = Vec::new();

    for hydro in system.hydros() {
        state_variables.push(serde_json::json!({
            "type": "storage",
            "entity_type": "hydro",
            "entity_id": hydro.id.0,
            "unit": "hm3"
        }));
    }

    for hydro_idx in 0..system.hydros().len() {
        let hydro_id = system.hydros()[hydro_idx].id.0;

        let max_order = system
            .inflow_models()
            .iter()
            .filter(|m| m.hydro_id.0 == hydro_id)
            .map(cobre_core::InflowModel::ar_order)
            .max()
            .unwrap_or(0);

        for lag_index in 1..=max_order {
            state_variables.push(serde_json::json!({
                "type": "inflow_lag",
                "entity_type": "hydro",
                "entity_id": hydro_id,
                "lag_index": lag_index,
                "unit": "m3/s"
            }));
        }
    }

    // Emit slot-major to match the LP layout
    // `anticipated_state.start + slot * n_anticipated + plant`. Slots
    // `lead_stages..k_max` are zero padding; the `lead_stages` field lets
    // consumers tell active slots (`slot_index < lead_stages`) from padding.
    let anticipated_thermals: Vec<&cobre_core::Thermal> = system
        .thermals()
        .iter()
        .filter(|t| t.anticipated_config.is_some())
        .collect();
    let k_max = anticipated_thermals
        .iter()
        .filter_map(|t| t.anticipated_config.map(|c| c.lead_stages as usize))
        .max()
        .unwrap_or(0);
    for slot in 0..k_max {
        for plant in &anticipated_thermals {
            let lead_stages = plant.anticipated_config.map_or(0_u32, |c| c.lead_stages);
            state_variables.push(serde_json::json!({
                "type": "anticipated_state",
                "entity_type": "thermal",
                "entity_id": plant.id.0,
                "slot_index": slot,
                "lead_stages": lead_stages,
                "unit": "MW"
            }));
        }
    }

    let content = serde_json::json!({
        "version": "1.0",
        "state_variables": state_variables
    });

    let json_str =
        serde_json::to_string_pretty(&content).map_err(|e| OutputError::ManifestError {
            manifest_type: "state_dictionary.json".to_string(),
            message: e.to_string(),
        })?;

    write_bytes_atomic(
        path.join("state_dictionary.json").as_path(),
        json_str.as_bytes(),
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use cobre_core::{
        AnticipatedConfig, Block, BlockMode, Bus, DeficitSegment, EntityId, Hydro,
        HydroGenerationModel, HydroPenalties, InflowModel, NoiseMethod, ScenarioSourceConfig,
        Stage, StageRiskConfig, StageStateConfig, SystemBuilder, Thermal,
        resolved::{
            BoundsCountsSpec, BoundsDefaults, ContractStageBounds, HydroStageBounds,
            LineStageBounds, PumpingStageBounds, ResolvedBounds, ThermalStageBounds,
        },
    };

    // ── Fixtures ─────────────────────────────────────────────────────────────

    fn hydro_penalties_zero() -> HydroPenalties {
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

    fn make_hydro(id: i32, name: &str, bus_id: i32) -> Hydro {
        Hydro {
            id: EntityId(id),
            name: name.to_string(),
            bus_id: EntityId(bus_id),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
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
            penalties: hydro_penalties_zero(),
        }
    }

    fn make_thermal(id: i32, name: &str, bus_id: i32) -> Thermal {
        Thermal {
            id: EntityId(id),
            name: name.to_string(),
            bus_id: EntityId(bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: None,
        }
    }

    fn make_bus(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("Bus{id}"),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        }
    }

    fn make_stage(id: i32) -> Stage {
        Stage {
            index: id.max(0) as usize,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 720.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 10,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// Build a `System` with 2 hydros and 1 thermal for standard tests.
    fn make_system_2h_1t() -> System {
        let bus = make_bus(1);
        let h1 = make_hydro(1, "Hydro1", 1);
        let h2 = make_hydro(2, "Hydro2", 1);
        let t1 = make_thermal(1, "Thermal1", 1);
        let stage = make_stage(0);

        let hydro_bounds_default = HydroStageBounds {
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
        };
        let thermal_bounds_default = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 0.0,
        };
        let line_default = LineStageBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pumping_default = PumpingStageBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 0.0,
        };
        let contract_default = ContractStageBounds {
            min_mw: 0.0,
            max_mw: 0.0,
            price_per_mwh: 0.0,
        };
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: hydro_bounds_default,
                thermal: thermal_bounds_default,
                line: line_default,
                pumping: pumping_default,
                contract: contract_default,
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![h1, h2])
            .thermals(vec![t1])
            .stages(vec![stage])
            .bounds(bounds)
            .build()
            .expect("valid system")
    }

    /// Build a `System` with 1 hydro, 2 stages, and custom bounds for bounds tests.
    fn make_system_1h_2stages(min_storage: f64, max_storage: f64) -> System {
        let bus = make_bus(1);
        let h1 = make_hydro(1, "Hydro1", 1);
        let stage0 = make_stage(0);
        let stage1 = make_stage(1);

        let hydro_bounds_default = HydroStageBounds {
            min_storage_hm3: min_storage,
            max_storage_hm3: max_storage,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            min_generation_mw: 0.0,
            max_generation_mw: 45.0,
            max_diversion_m3s: None,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        };
        let thermal_default = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 0.0,
        };
        let line_default = LineStageBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pumping_default = PumpingStageBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 0.0,
        };
        let contract_default = ContractStageBounds {
            min_mw: 0.0,
            max_mw: 0.0,
            price_per_mwh: 0.0,
        };
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 2,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: hydro_bounds_default,
                thermal: thermal_default,
                line: line_default,
                pumping: pumping_default,
                contract: contract_default,
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![h1])
            .stages(vec![stage0, stage1])
            .bounds(bounds)
            .build()
            .expect("valid system")
    }

    // ── codes.json ────────────────────────────────────────────────────────────

    #[test]
    fn codes_json_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        write_codes_json(tmp.path()).expect("write_codes_json must succeed");

        let raw = std::fs::read_to_string(tmp.path().join("codes.json")).unwrap();
        let val: serde_json::Value = serde_json::from_str(&raw).unwrap();

        assert_eq!(
            val["operative_state"]["2"],
            serde_json::json!("operating"),
            "operative_state[\"2\"] must equal \"operating\""
        );
        assert_eq!(
            val["storage_binding"]["1"],
            serde_json::json!("below_minimum"),
            "storage_binding[\"1\"] must equal \"below_minimum\""
        );
        assert_eq!(
            val["entity_type"]["0"],
            serde_json::json!("hydro"),
            "entity_type[\"0\"] must equal \"hydro\""
        );
        assert_eq!(
            val["bound_type"]["0"],
            serde_json::json!("storage_min"),
            "bound_type[\"0\"] must equal \"storage_min\""
        );
        assert!(
            val["generated_at"].is_string(),
            "generated_at must be a string"
        );
        assert_eq!(
            val["version"],
            serde_json::json!("1.0"),
            "version must be \"1.0\""
        );
    }

    // ── entities.csv ─────────────────────────────────────────────────────────

    #[test]
    fn entities_csv_correct_rows() {
        let system = make_system_2h_1t();
        let tmp = tempfile::tempdir().unwrap();
        write_entities_csv(tmp.path(), &system).expect("write_entities_csv must succeed");

        let content = std::fs::read_to_string(tmp.path().join("entities.csv")).unwrap();
        let mut rdr = csv::Reader::from_reader(content.as_bytes());

        let rows: Vec<Vec<String>> = rdr
            .records()
            .map(|r| r.unwrap().iter().map(ToString::to_string).collect())
            .collect();

        assert_eq!(
            rows.len(),
            4,
            "expected 4 data rows (2 hydros + 1 thermal + 1 bus)"
        );

        assert_eq!(rows[0][0], "0", "row 0: entity_type_code must be 0 (hydro)");
        assert_eq!(rows[0][1], "1", "row 0: entity_id must be 1");
        assert_eq!(rows[0][2], "Hydro1", "row 0: name must be Hydro1");

        assert_eq!(rows[1][0], "0", "row 1: entity_type_code must be 0 (hydro)");
        assert_eq!(rows[1][1], "2", "row 1: entity_id must be 2");
        assert_eq!(rows[1][2], "Hydro2", "row 1: name must be Hydro2");

        assert_eq!(
            rows[2][0], "1",
            "row 2: entity_type_code must be 1 (thermal)"
        );
        assert_eq!(rows[2][1], "1", "row 2: entity_id must be 1");
        assert_eq!(rows[2][2], "Thermal1", "row 2: name must be Thermal1");
    }

    #[test]
    fn entities_csv_entity_type_order() {
        let system = make_system_2h_1t();
        let tmp = tempfile::tempdir().unwrap();
        write_entities_csv(tmp.path(), &system).expect("write_entities_csv must succeed");

        let content = std::fs::read_to_string(tmp.path().join("entities.csv")).unwrap();
        let mut rdr = csv::Reader::from_reader(content.as_bytes());

        let type_codes: Vec<i8> = rdr
            .records()
            .map(|r| r.unwrap().get(0).unwrap().parse::<i8>().unwrap())
            .collect();

        for window in type_codes.windows(2) {
            assert!(
                window[0] <= window[1],
                "entity_type_codes must be non-decreasing, found {} followed by {}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn entities_csv_system_id_is_zero() {
        let system = make_system_2h_1t();
        let tmp = tempfile::tempdir().unwrap();
        write_entities_csv(tmp.path(), &system).expect("write_entities_csv must succeed");

        let content = std::fs::read_to_string(tmp.path().join("entities.csv")).unwrap();
        let mut rdr = csv::Reader::from_reader(content.as_bytes());

        for rec in rdr.records() {
            let row = rec.unwrap();
            assert_eq!(row.get(4).unwrap(), "0", "system_id must be 0 for all rows");
        }
    }

    // ── variables.csv ─────────────────────────────────────────────────────────

    #[test]
    fn variables_csv_total_columns() {
        let tmp = tempfile::tempdir().unwrap();
        write_variables_csv(tmp.path()).expect("write_variables_csv must succeed");

        let content = std::fs::read_to_string(tmp.path().join("variables.csv")).unwrap();
        let mut rdr = csv::Reader::from_reader(content.as_bytes());

        let row_count = rdr.records().count();
        assert_eq!(
            row_count, 209,
            "variables.csv must have exactly 209 data rows (one per column across all schemas)"
        );
    }

    #[test]
    fn variables_csv_has_required_columns_in_header() {
        let tmp = tempfile::tempdir().unwrap();
        write_variables_csv(tmp.path()).expect("write_variables_csv must succeed");

        let content = std::fs::read_to_string(tmp.path().join("variables.csv")).unwrap();
        let mut rdr = csv::Reader::from_reader(content.as_bytes());

        let headers: Vec<String> = rdr
            .headers()
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .collect();

        assert!(
            headers.contains(&"file".to_string()),
            "header must contain 'file'"
        );
        assert!(
            headers.contains(&"column".to_string()),
            "header must contain 'column'"
        );
        assert!(
            headers.contains(&"type".to_string()),
            "header must contain 'type'"
        );
        assert!(
            headers.contains(&"unit".to_string()),
            "header must contain 'unit'"
        );
        assert!(
            headers.contains(&"description".to_string()),
            "header must contain 'description'"
        );
        assert!(
            headers.contains(&"nullable".to_string()),
            "header must contain 'nullable'"
        );
    }

    #[test]
    fn variables_csv_nullable_reflects_schema() {
        let tmp = tempfile::tempdir().unwrap();
        write_variables_csv(tmp.path()).expect("write_variables_csv must succeed");

        let content = std::fs::read_to_string(tmp.path().join("variables.csv")).unwrap();
        let mut rdr = csv::Reader::from_reader(content.as_bytes());

        let block_id_nullable = rdr
            .records()
            .find(|r| {
                let row = r.as_ref().unwrap();
                row.get(0).unwrap() == "costs" && row.get(1).unwrap() == "block_id"
            })
            .map(|r| r.unwrap().get(5).unwrap().to_string());

        assert_eq!(
            block_id_nullable,
            Some("true".to_string()),
            "costs.block_id must have nullable=true"
        );
    }

    // ── bounds.parquet ────────────────────────────────────────────────────────

    #[test]
    fn bounds_parquet_roundtrip() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let system = make_system_1h_2stages(100.0, 500.0);
        let tmp = tempfile::tempdir().unwrap();
        let config = ParquetWriterConfig::default();

        write_bounds_parquet(tmp.path(), &system, &config)
            .expect("write_bounds_parquet must succeed");

        let path = tmp.path().join("bounds.parquet");
        assert!(path.exists(), "bounds.parquet must exist");

        let file = std::fs::File::open(&path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let mut reader = builder.build().unwrap();
        let batch = reader.next().expect("must have rows").expect("batch Ok");

        // 1 hydro, 2 stages, 7 bound types per stage (no max_outflow) = 14 rows
        assert_eq!(
            batch.num_rows(),
            14,
            "1 hydro × 2 stages × 7 bounds = 14 rows"
        );

        let entity_type_col = batch
            .column_by_name("entity_type_code")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int8Array>()
            .unwrap();
        let bound_type_col = batch
            .column_by_name("bound_type_code")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int8Array>()
            .unwrap();
        let bound_value_col = batch
            .column_by_name("bound_value")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        let block_id_col = batch.column_by_name("block_id").unwrap();

        for row in 0..batch.num_rows() {
            assert!(
                block_id_col.is_null(row),
                "block_id must be null at row {row}"
            );
        }

        let storage_min_row = (0..batch.num_rows()).find(|&i| {
            entity_type_col.value(i) == ENTITY_TYPE_HYDRO
                && bound_type_col.value(i) == BOUND_STORAGE_MIN
        });
        assert!(
            storage_min_row.is_some(),
            "must have a storage_min row for hydro"
        );
        let row = storage_min_row.unwrap();
        assert!(
            (bound_value_col.value(row) - 100.0).abs() < f64::EPSILON,
            "storage_min must be 100.0, got {}",
            bound_value_col.value(row)
        );

        let storage_max_row = (0..batch.num_rows()).find(|&i| {
            entity_type_col.value(i) == ENTITY_TYPE_HYDRO
                && bound_type_col.value(i) == BOUND_STORAGE_MAX
        });
        assert!(
            storage_max_row.is_some(),
            "must have a storage_max row for hydro"
        );
        let row = storage_max_row.unwrap();
        assert!(
            (bound_value_col.value(row) - 500.0).abs() < f64::EPSILON,
            "storage_max must be 500.0, got {}",
            bound_value_col.value(row)
        );
    }

    // ── state_dictionary.json ─────────────────────────────────────────────────

    #[test]
    fn state_dictionary_hydro_storage_entries() {
        let system = make_system_2h_1t();
        let tmp = tempfile::tempdir().unwrap();
        write_state_dictionary_json(tmp.path(), &system)
            .expect("write_state_dictionary_json must succeed");

        let raw = std::fs::read_to_string(tmp.path().join("state_dictionary.json")).unwrap();
        let val: serde_json::Value = serde_json::from_str(&raw).unwrap();

        let state_vars = val["state_variables"]
            .as_array()
            .expect("state_variables must be an array");

        let storage_entries: Vec<&serde_json::Value> = state_vars
            .iter()
            .filter(|v| v["type"] == serde_json::json!("storage"))
            .collect();

        assert_eq!(
            storage_entries.len(),
            2,
            "must have exactly 2 storage entries (one per hydro)"
        );

        for entry in &storage_entries {
            assert_eq!(
                entry["entity_type"],
                serde_json::json!("hydro"),
                "storage entry entity_type must be \"hydro\""
            );
            assert_eq!(
                entry["unit"],
                serde_json::json!("hm3"),
                "storage entry unit must be \"hm3\""
            );
        }
    }

    #[test]
    fn state_dictionary_version_field() {
        let system = make_system_2h_1t();
        let tmp = tempfile::tempdir().unwrap();
        write_state_dictionary_json(tmp.path(), &system)
            .expect("write_state_dictionary_json must succeed");

        let raw = std::fs::read_to_string(tmp.path().join("state_dictionary.json")).unwrap();
        let val: serde_json::Value = serde_json::from_str(&raw).unwrap();

        assert_eq!(
            val["version"],
            serde_json::json!("1.0"),
            "state_dictionary must have version \"1.0\""
        );
    }

    // ── dictionary unit/description tests ────────────────────────────────────

    #[test]
    fn new_energy_columns_have_units() {
        assert_eq!(
            unit_for("hydros", "equivalent_productivity_mw_per_m3s"),
            "MW/(m3/s)"
        );
        assert_eq!(
            unit_for("hydros", "accumulated_productivity_mw_per_m3s"),
            "MW/(m3/s)"
        );
        assert_eq!(unit_for("hydros", "incremental_inflow_energy_mw"), "MW");
        assert_eq!(unit_for("hydros", "stored_energy_initial_mwh"), "MWh");
        assert_eq!(unit_for("hydros", "stored_energy_final_mwh"), "MWh");
    }

    #[test]
    fn new_energy_columns_have_descriptions() {
        assert!(
            !description_for("hydros", "equivalent_productivity_mw_per_m3s").is_empty(),
            "equivalent_productivity_mw_per_m3s must have a description"
        );
        assert!(
            !description_for("hydros", "accumulated_productivity_mw_per_m3s").is_empty(),
            "accumulated_productivity_mw_per_m3s must have a description"
        );
        assert!(
            !description_for("hydros", "incremental_inflow_energy_mw").is_empty(),
            "incremental_inflow_energy_mw must have a description"
        );
        assert!(
            !description_for("hydros", "stored_energy_initial_mwh").is_empty(),
            "stored_energy_initial_mwh must have a description"
        );
        assert!(
            !description_for("hydros", "stored_energy_final_mwh").is_empty(),
            "stored_energy_final_mwh must have a description"
        );
    }

    #[test]
    fn old_productivity_field_returns_default_unit() {
        assert_eq!(
            unit_for("hydros", "productivity_mw_per_m3s"),
            "",
            "removed column must fall through to the default empty-string arm"
        );
    }

    #[test]
    fn every_hydros_schema_column_has_description() {
        let schema = hydros_schema();
        for field in schema.fields() {
            let desc = description_for("hydros", field.name());
            assert!(
                !desc.is_empty(),
                "hydros column '{}' has no description in description_for",
                field.name()
            );
        }
    }

    // ── state_dictionary anticipated_state tests ──────────────────────────────

    /// Build a `System` with no hydros and a list of thermals (each with optional
    /// `AnticipatedConfig`). Used by the anticipated-state dictionary tests.
    fn make_system_thermals_only(thermals: Vec<Thermal>) -> System {
        let bus = make_bus(1);
        let stage = make_stage(0);
        let n_thermals = thermals.len();

        let thermal_bounds_default = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 0.0,
        };
        let line_default = LineStageBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pumping_default = PumpingStageBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 0.0,
        };
        let contract_default = ContractStageBounds {
            min_mw: 0.0,
            max_mw: 0.0,
            price_per_mwh: 0.0,
        };
        // k_max is the maximum lead_stages across anticipated thermals (or 0).
        let k_max = thermals
            .iter()
            .filter_map(|t| t.anticipated_config.map(|c| c.lead_stages as usize))
            .max()
            .unwrap_or(0);
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
                k_max,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 0.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                    max_diversion_m3s: None,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                thermal: thermal_bounds_default,
                line: line_default,
                pumping: pumping_default,
                contract: contract_default,
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(thermals)
            .stages(vec![stage])
            .bounds(bounds)
            .build()
            .expect("valid system")
    }

    /// Helper: parse `state_dictionary.json` written to a temp dir and return the
    /// `state_variables` array.
    fn parse_state_variables(system: &System) -> Vec<serde_json::Value> {
        let tmp = tempfile::tempdir().unwrap();
        write_state_dictionary_json(tmp.path(), system)
            .expect("write_state_dictionary_json must succeed");
        let raw = std::fs::read_to_string(tmp.path().join("state_dictionary.json")).unwrap();
        let val: serde_json::Value = serde_json::from_str(&raw).unwrap();
        val["state_variables"]
            .as_array()
            .expect("state_variables must be an array")
            .clone()
    }

    #[test]
    fn state_dictionary_zero_anticipated_emits_no_anticipated_entries() {
        let system = make_system_2h_1t();
        let state_vars = parse_state_variables(&system);

        let anticipated: Vec<&serde_json::Value> = state_vars
            .iter()
            .filter(|v| v["type"] == serde_json::json!("anticipated_state"))
            .collect();

        assert_eq!(
            anticipated.len(),
            0,
            "expected zero anticipated_state entries when no anticipated thermals"
        );
    }

    #[test]
    fn state_dictionary_single_anticipated_k1_emits_one_entry() {
        let t = Thermal {
            anticipated_config: Some(AnticipatedConfig { lead_stages: 1 }),
            ..make_thermal(7, "Thermal7", 1)
        };
        let system = make_system_thermals_only(vec![t]);
        let state_vars = parse_state_variables(&system);

        let anticipated: Vec<&serde_json::Value> = state_vars
            .iter()
            .filter(|v| v["type"] == serde_json::json!("anticipated_state"))
            .collect();

        assert_eq!(
            anticipated.len(),
            1,
            "expected exactly one anticipated_state entry"
        );
        assert_eq!(
            anticipated[0]["entity_id"],
            serde_json::json!(7),
            "entity_id must be 7"
        );
        assert_eq!(
            anticipated[0]["slot_index"],
            serde_json::json!(0),
            "slot_index must be 0"
        );
        assert_eq!(
            anticipated[0]["unit"],
            serde_json::json!("MW"),
            "unit must be MW"
        );
        assert_eq!(
            anticipated[0]["entity_type"],
            serde_json::json!("thermal"),
            "entity_type must be thermal"
        );
    }

    #[test]
    fn state_dictionary_two_anticipated_k3_slot_major_ordering() {
        // 6 entries (A*K_max = 2*3) in slot-major order:
        // slot 0: [5, 8], slot 1: [5, 8], slot 2: [5, 8].
        let t5 = Thermal {
            anticipated_config: Some(AnticipatedConfig { lead_stages: 2 }),
            ..make_thermal(5, "Thermal5", 1)
        };
        let t8 = Thermal {
            anticipated_config: Some(AnticipatedConfig { lead_stages: 3 }),
            ..make_thermal(8, "Thermal8", 1)
        };
        // Build with t5 before t8; SystemBuilder sorts by id so order is [5, 8].
        let system = make_system_thermals_only(vec![t5, t8]);
        let state_vars = parse_state_variables(&system);

        let anticipated: Vec<&serde_json::Value> = state_vars
            .iter()
            .filter(|v| v["type"] == serde_json::json!("anticipated_state"))
            .collect();

        assert_eq!(anticipated.len(), 6, "expected 6 anticipated_state entries");

        let expected: Vec<(i64, i64)> = vec![(5, 0), (8, 0), (5, 1), (8, 1), (5, 2), (8, 2)];
        for (i, (exp_id, exp_slot)) in expected.iter().enumerate() {
            assert_eq!(
                anticipated[i]["entity_id"],
                serde_json::json!(exp_id),
                "entry {i}: entity_id must be {exp_id}"
            );
            assert_eq!(
                anticipated[i]["slot_index"],
                serde_json::json!(exp_slot),
                "entry {i}: slot_index must be {exp_slot}"
            );
        }
    }

    #[test]
    fn state_dictionary_total_length_equals_n_state() {
        // 2 hydros (ar_order 1 and 0), 1 anticipated thermal (K=2):
        // total = 2 (storage) + 1 (inflow_lag) + 1*2 (anticipated) = 5.
        let bus = make_bus(1);
        let h1 = make_hydro(1, "Hydro1", 1);
        let h2 = make_hydro(2, "Hydro2", 1);
        let t = Thermal {
            anticipated_config: Some(AnticipatedConfig { lead_stages: 2 }),
            ..make_thermal(1, "Thermal1", 1)
        };
        let stage = make_stage(0);

        // Inflow model for h1 with ar_order=1.
        let inflow_h1 = InflowModel {
            hydro_id: EntityId(1),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 20.0,
            ar_coefficients: vec![0.5],
            residual_std_ratio: 0.87,
            annual: None,
        };
        // Inflow model for h2 with ar_order=0 (white noise — empty coefficients).
        let inflow_h2 = InflowModel {
            hydro_id: EntityId(2),
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 15.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        };

        let hydro_bounds_default = HydroStageBounds {
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
        };
        let thermal_bounds_default = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 0.0,
        };
        let line_default = LineStageBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pumping_default = PumpingStageBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 0.0,
        };
        let contract_default = ContractStageBounds {
            min_mw: 0.0,
            max_mw: 0.0,
            price_per_mwh: 0.0,
        };
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
                hydro: hydro_bounds_default,
                thermal: thermal_bounds_default,
                line: line_default,
                pumping: pumping_default,
                contract: contract_default,
            },
        );

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![h1, h2])
            .thermals(vec![t])
            .stages(vec![stage])
            .bounds(bounds)
            .inflow_models(vec![inflow_h1, inflow_h2])
            .build()
            .expect("valid system");

        let state_vars = parse_state_variables(&system);

        assert_eq!(
            state_vars.len(),
            5,
            "total state_variables length must be 5 (2 storage + 1 inflow_lag + 2 anticipated)"
        );
    }
}
