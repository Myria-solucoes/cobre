//! Parsers for constraint files in the `constraints/` subdirectory.
//!
//! Constraint files provide stage-varying bound overrides for entity types and
//! user-defined generic linear constraints.
//! All files are optional — when absent, the `load_*` wrappers return
//! `Ok(Vec::new())` without touching the filesystem.
//!
//! ## Parsing convention
//!
//! Parquet parsers follow the canonical pattern:
//!
//! 1. Open the file with `std::fs::File::open`.
//! 2. Build a `ParquetRecordBatchReaderBuilder` and consume all record batches.
//! 3. Extract typed columns by name; return `SchemaError` for missing or wrong-type columns.
//! 4. Validate per-row constraints; return `SchemaError` on violation.
//! 5. Sort the output by the documented sort key and return.
//!
//! JSON parsers follow the 4-step pipeline:
//! `fs::read_to_string` → `serde_json::from_str` → `validate_raw` → `convert`.
//!
//! Cross-reference validation (checking that entity IDs exist in registries)
//! is deferred to Layer 3.
//! Semantic bound validation (e.g., min < max) is deferred.

pub mod bounds;
pub mod generic;
pub mod generic_bounds;
pub mod ncs_bounds;
pub mod penalty_overrides;

pub use bounds::{
    ContractBoundsRow, HydroBoundsRow, LineBoundsRow, PumpingBoundsRow, ThermalBoundsRow,
    parse_contract_bounds, parse_hydro_bounds, parse_line_bounds, parse_pumping_bounds,
    parse_thermal_bounds,
};
pub use generic::parse_generic_constraints;
pub use generic_bounds::{GenericConstraintBoundsRow, parse_generic_constraint_bounds};
pub use ncs_bounds::{NcsBoundsRow, parse_ncs_bounds};
pub use penalty_overrides::{
    BusPenaltyOverrideRow, HydroPenaltyOverrideRow, LinePenaltyOverrideRow, NcsPenaltyOverrideRow,
    parse_penalty_overrides_bus, parse_penalty_overrides_hydro, parse_penalty_overrides_line,
    parse_penalty_overrides_ncs,
};

use cobre_core::{EntityId, GenericConstraint};
use std::collections::HashMap;
use std::path::Path;

use crate::LoadError;

/// Load `constraints/thermal_bounds.parquet` when the path is known, or
/// return an empty `Vec` when the file is absent (optional file).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_thermal_bounds`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_thermal_bounds;
///
/// let rows = load_thermal_bounds(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_thermal_bounds(path: Option<&Path>) -> Result<Vec<ThermalBoundsRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_thermal_bounds(p),
    }
}

/// Load `constraints/hydro_bounds.parquet` when the path is known, or
/// return an empty `Vec` when the file is absent (optional file).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_hydro_bounds`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_hydro_bounds;
///
/// let rows = load_hydro_bounds(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_hydro_bounds(path: Option<&Path>) -> Result<Vec<HydroBoundsRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_hydro_bounds(p),
    }
}

/// Load `constraints/line_bounds.parquet` when the path is known, or
/// return an empty `Vec` when the file is absent (optional file).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_line_bounds`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_line_bounds;
///
/// let rows = load_line_bounds(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_line_bounds(path: Option<&Path>) -> Result<Vec<LineBoundsRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_line_bounds(p),
    }
}

/// Load `constraints/pumping_bounds.parquet` when the path is known, or
/// return an empty `Vec` when the file is absent (optional file).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_pumping_bounds`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_pumping_bounds;
///
/// let rows = load_pumping_bounds(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_pumping_bounds(path: Option<&Path>) -> Result<Vec<PumpingBoundsRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_pumping_bounds(p),
    }
}

/// Load `constraints/contract_bounds.parquet` when the path is known, or
/// return an empty `Vec` when the file is absent (optional file).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_contract_bounds`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_contract_bounds;
///
/// let rows = load_contract_bounds(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_contract_bounds(path: Option<&Path>) -> Result<Vec<ContractBoundsRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_contract_bounds(p),
    }
}

/// Load `constraints/penalty_overrides_bus.parquet` when the path is known, or
/// return an empty `Vec` when the file is absent (optional file).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_penalty_overrides_bus`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_penalty_overrides_bus;
///
/// let rows = load_penalty_overrides_bus(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_penalty_overrides_bus(
    path: Option<&Path>,
) -> Result<Vec<BusPenaltyOverrideRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_penalty_overrides_bus(p),
    }
}

/// Load `constraints/penalty_overrides_line.parquet` when the path is known, or
/// return an empty `Vec` when the file is absent (optional file).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_penalty_overrides_line`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_penalty_overrides_line;
///
/// let rows = load_penalty_overrides_line(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_penalty_overrides_line(
    path: Option<&Path>,
) -> Result<Vec<LinePenaltyOverrideRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_penalty_overrides_line(p),
    }
}

/// Load `constraints/penalty_overrides_hydro.parquet` when the path is known, or
/// return an empty `Vec` when the file is absent (optional file).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_penalty_overrides_hydro`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_penalty_overrides_hydro;
///
/// let rows = load_penalty_overrides_hydro(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_penalty_overrides_hydro(
    path: Option<&Path>,
) -> Result<Vec<HydroPenaltyOverrideRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_penalty_overrides_hydro(p),
    }
}

/// Load `constraints/penalty_overrides_ncs.parquet` when the path is known, or
/// return an empty `Vec` when the file is absent (optional file).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_penalty_overrides_ncs`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_penalty_overrides_ncs;
///
/// let rows = load_penalty_overrides_ncs(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_penalty_overrides_ncs(
    path: Option<&Path>,
) -> Result<Vec<NcsPenaltyOverrideRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_penalty_overrides_ncs(p),
    }
}

/// Load `constraints/generic_constraints.json` when the path is known, or
/// return an empty `Vec` when the file is absent (optional file).
///
/// `name_to_id` maps parameter definition names to their [`EntityId`].
/// Pass `&HashMap::new()` when no parameters have been loaded; expressions that
/// contain `@name` tokens will then fail with a schema error. The real mapping
/// is wired in by the caller once the parameter loader output is available.
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_generic_constraints`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_generic_constraints;
/// use std::collections::HashMap;
///
/// let constraints = load_generic_constraints(None, &HashMap::new()).expect("no file is fine");
/// assert!(constraints.is_empty());
/// ```
#[allow(clippy::implicit_hasher)]
pub fn load_generic_constraints(
    path: Option<&Path>,
    name_to_id: &HashMap<String, EntityId>,
) -> Result<Vec<GenericConstraint>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_generic_constraints(p, name_to_id),
    }
}

/// Load `constraints/generic_constraint_bounds.parquet` when the path is known, or
/// return an empty `Vec` when the file is absent (optional file).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_generic_constraint_bounds`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_generic_constraint_bounds;
///
/// let rows = load_generic_constraint_bounds(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_generic_constraint_bounds(
    path: Option<&Path>,
) -> Result<Vec<GenericConstraintBoundsRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_generic_constraint_bounds(p),
    }
}

/// Load `constraints/ncs_bounds.parquet` when the path is known, or
/// return an empty `Vec` when the file is absent (optional file).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_ncs_bounds`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_ncs_bounds;
///
/// let rows = load_ncs_bounds(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_ncs_bounds(path: Option<&Path>) -> Result<Vec<NcsBoundsRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_ncs_bounds(p),
    }
}
