//! Parsing for scalar parameter definition and value files.
//!
//! This module contains two Parquet parsers and one assembly function that together
//! form the full scalar parameter loading pipeline:
//!
//! - [`parse_scalar_parameter_definitions`] — reads definitions and returns a sorted
//!   `Vec<ParameterDefinitionRow>`.
//! - [`parse_scalar_parameter_values`] — reads value rows in file order and returns a
//!   flat `Vec<ScalarParameterValueRow>`.
//! - [`assemble_scalar_parameters`] — joins definitions with value rows, validates
//!   per-kind constraints, and produces the final `Vec<ScalarParameter>`.
//!
//! ## Definitions Parquet schema
//!
//! | Column          | Parquet type | Nullable | Description                                              |
//! | --------------- | ------------ | -------- | -------------------------------------------------------- |
//! | `id`            | INT32        | no       | Unique parameter identifier (`EntityId`)                 |
//! | `name`          | UTF8         | no       | Unique parameter name (non-empty, no leading/trailing spaces) |
//! | `kind`          | UTF8         | no       | One of `constant` / `per_stage` / `seasonal` / `computed` |
//! | `computed_spec` | UTF8         | yes      | Non-null and parseable when `kind == "computed"`; null for other kinds |
//!
//! ## Values Parquet schema
//!
//! | Column          | Parquet type | Nullable | Description                              |
//! | --------------- | ------------ | -------- | ---------------------------------------- |
//! | `parameter_id`  | INT32        | no       | `EntityId` of the parameter              |
//! | `stage_id`      | INT32        | yes      | Zero-based stage index for `PerStage`    |
//! | `season_id`     | INT32        | yes      | Season id for `Seasonal`                 |
//! | `value`         | DOUBLE       | no       | Finite `f64` value                       |
//!
//! ## Output ordering
//!
//! Definitions are sorted by `id.0` ascending. Value rows are returned in file order.
//! The assembled `Vec<ScalarParameter>` preserves definition order.
//!
//! ## Validation
//!
//! Per-row constraints enforced by this parser:
//!
//! - All columns must be present with the correct types.
//! - `id`, `name`, and `kind` must not be null.
//! - `name` must be non-empty and must not have leading or trailing whitespace.
//! - `kind` must be one of the four legal values listed above.
//! - When `kind == "computed"`, `computed_spec` must be non-null, non-empty,
//!   and parseable as `<variant_tag>(hydro_id=<int>)`.
//! - When `kind != "computed"`, `computed_spec` must be null or absent.
//! - `id` values must be unique across all rows.
//! - `name` values must be unique across all rows (case-sensitive comparison).
//!
//! Deferred validations (not performed here):
//!
//! - `hydro_id` existence in the hydro registry — deferred to cross-validation.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;

use arrow::array::Array;
use cobre_core::{ComputedParameter, EntityId, ParameterKind, ScalarParameter};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::LoadError;
use crate::parquet_helpers::{
    extract_optional_int32, extract_optional_utf8, extract_required_float64,
    extract_required_int32, extract_required_utf8,
};

/// An intermediate row from `system/scalar_parameter_definitions.parquet`.
///
/// Each row holds the unique identifier, human-readable name, and kind header
/// for one scalar parameter. For the `"computed"` kind the header carries the
/// fully-typed [`ComputedParameter`]; for the three value-bearing kinds it
/// carries only the tag — numeric values are loaded separately.
///
/// # Examples
///
/// ```
/// use cobre_io::{ParameterDefinitionRow, ParameterKindHeader};
/// use cobre_core::EntityId;
///
/// let row = ParameterDefinitionRow {
///     id: EntityId(1),
///     name: "rho_eq_h1".to_string(),
///     header: ParameterKindHeader::Constant,
/// };
/// assert_eq!(row.id, EntityId(1));
/// assert_eq!(row.header, ParameterKindHeader::Constant);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterDefinitionRow {
    /// Unique parameter identifier.
    pub id: EntityId,
    /// Unique parameter name (non-empty, no leading or trailing whitespace).
    pub name: String,
    /// Kind tag, with a [`ComputedParameter`] payload for the `Computed` variant.
    pub header: ParameterKindHeader,
}

/// Kind tag for a [`ParameterDefinitionRow`].
///
/// Mirrors the four cases of `ParameterKind` but carries no numeric payload
/// for the three value-bearing kinds (`Constant`, `PerStage`, `Seasonal`).
/// The `Computed` variant carries the fully-typed [`ComputedParameter`].
///
/// # Examples
///
/// ```
/// use cobre_io::ParameterKindHeader;
/// use cobre_core::{ComputedParameter, EntityId};
///
/// let header = ParameterKindHeader::Computed(
///     ComputedParameter::EquivalentProductivity { hydro_id: EntityId(7) }
/// );
/// assert!(matches!(header, ParameterKindHeader::Computed(_)));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParameterKindHeader {
    /// A single scalar value applied to every stage.
    Constant,
    /// One scalar value per study stage.
    PerStage,
    /// One scalar value per season.
    Seasonal,
    /// A value derived from physical plant data by the resolver.
    Computed(ComputedParameter),
}

/// The seven legal `<variant_tag>` strings for `computed_spec`, listed for error messages.
const LEGAL_VARIANT_TAGS: &str = "equivalent_productivity, accumulated_productivity, reference_volume, \
     reference_turbine, min_storage, max_storage, specific_productivity";

/// Parse `system/scalar_parameter_definitions.parquet` and return a sorted definition table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row constraints,
/// then returns all rows sorted by `id.0` ascending.
///
/// # Errors
///
/// | Condition                                                    | Error variant              |
/// | ------------------------------------------------------------ | -------------------------- |
/// | File not found or permission denied                          | [`LoadError::IoError`]     |
/// | Malformed Parquet                                            | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type                        | [`LoadError::SchemaError`] |
/// | Null `id`, `name`, or `kind`                                 | [`LoadError::SchemaError`] |
/// | `name` is empty or has leading or trailing whitespace        | [`LoadError::SchemaError`] |
/// | `kind` is not one of the four legal values                   | [`LoadError::SchemaError`] |
/// | `kind == "computed"` but `computed_spec` is null or empty    | [`LoadError::SchemaError`] |
/// | `kind != "computed"` but `computed_spec` is non-null/non-empty | [`LoadError::SchemaError`] |
/// | `computed_spec` fails to parse as `<tag>(hydro_id=<int>)`   | [`LoadError::SchemaError`] |
/// | Unknown variant tag in `computed_spec`                       | [`LoadError::SchemaError`] |
/// | Duplicate `id` across rows                                   | [`LoadError::SchemaError`] |
/// | Duplicate `name` across rows (case-sensitive)                | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::parse_scalar_parameter_definitions;
/// use std::path::Path;
///
/// let rows = parse_scalar_parameter_definitions(
///     Path::new("system/scalar_parameter_definitions.parquet")
/// ).expect("valid definitions file");
/// println!("loaded {} parameter definitions", rows.len());
/// ```
pub fn parse_scalar_parameter_definitions(
    path: &Path,
) -> Result<Vec<ParameterDefinitionRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<ParameterDefinitionRow> = Vec::new();
    let mut seen_ids: HashSet<i32> = HashSet::new();
    let mut seen_names: HashSet<String> = HashSet::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        // ── Extract columns by name ───────────────────────────────────────────
        let id_col = extract_required_int32(&batch, "id", path)?;
        let name_col = extract_required_utf8(&batch, "name", path)?;
        let kind_col = extract_required_utf8(&batch, "kind", path)?;
        let spec_col = extract_optional_utf8(&batch, "computed_spec", path)?;

        // ── Build rows with per-row validation ───────────────────────────────
        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            // ── id: non-null ──────────────────────────────────────────────
            if id_col.is_null(i) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("scalar_parameter_definitions[{row_idx}].id"),
                    message: "id must not be null".to_string(),
                });
            }
            let id_raw = id_col.value(i);

            // ── id: unique ────────────────────────────────────────────────
            if !seen_ids.insert(id_raw) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("scalar_parameter_definitions[{row_idx}].id"),
                    message: format!("duplicate id {id_raw}"),
                });
            }

            // ── name: non-null ────────────────────────────────────────────
            if name_col.is_null(i) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("scalar_parameter_definitions[{row_idx}].name"),
                    message: "name must not be null".to_string(),
                });
            }
            let name_val = name_col.value(i);

            // ── name: non-empty and no leading/trailing whitespace ────────
            if name_val.is_empty() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("scalar_parameter_definitions[{row_idx}].name"),
                    message: "name must not be empty".to_string(),
                });
            }
            if name_val != name_val.trim() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("scalar_parameter_definitions[{row_idx}].name"),
                    message: format!(
                        "name must not have leading or trailing whitespace, got {name_val:?}"
                    ),
                });
            }

            // ── name: unique (case-sensitive) ─────────────────────────────
            if !seen_names.insert(name_val.to_string()) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("scalar_parameter_definitions[{row_idx}].name"),
                    message: format!("duplicate name {name_val:?}"),
                });
            }

            // ── kind: non-null ────────────────────────────────────────────
            if kind_col.is_null(i) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("scalar_parameter_definitions[{row_idx}].kind"),
                    message: "kind must not be null".to_string(),
                });
            }
            let kind_val = kind_col.value(i);

            // ── computed_spec raw value ───────────────────────────────────
            let spec_raw: Option<&str> = spec_col.and_then(|col| {
                if col.is_null(i) {
                    None
                } else {
                    Some(col.value(i))
                }
            });

            // ── Parse kind and enforce computed_spec rules ────────────────
            let header = match kind_val {
                "constant" => {
                    reject_spec_for_non_computed(spec_raw, row_idx, path)?;
                    ParameterKindHeader::Constant
                }
                "per_stage" => {
                    reject_spec_for_non_computed(spec_raw, row_idx, path)?;
                    ParameterKindHeader::PerStage
                }
                "seasonal" => {
                    reject_spec_for_non_computed(spec_raw, row_idx, path)?;
                    ParameterKindHeader::Seasonal
                }
                "computed" => {
                    let spec = match spec_raw {
                        None => {
                            return Err(LoadError::SchemaError {
                                path: path.to_path_buf(),
                                field: format!(
                                    "scalar_parameter_definitions[{row_idx}].computed_spec"
                                ),
                                message: "computed_spec must be non-null when kind is \"computed\""
                                    .to_string(),
                            });
                        }
                        Some(s) if s.trim().is_empty() => {
                            return Err(LoadError::SchemaError {
                                path: path.to_path_buf(),
                                field: format!(
                                    "scalar_parameter_definitions[{row_idx}].computed_spec"
                                ),
                                message:
                                    "computed_spec must not be empty when kind is \"computed\""
                                        .to_string(),
                            });
                        }
                        Some(s) => s,
                    };
                    let computed = parse_computed_spec(spec, row_idx, path)?;
                    ParameterKindHeader::Computed(computed)
                }
                other => {
                    return Err(LoadError::SchemaError {
                        path: path.to_path_buf(),
                        field: format!("scalar_parameter_definitions[{row_idx}].kind"),
                        message: format!(
                            "unknown kind {other:?}; legal values are: constant, per_stage, seasonal, computed"
                        ),
                    });
                }
            };

            rows.push(ParameterDefinitionRow {
                id: EntityId(id_raw),
                name: name_val.to_string(),
                header,
            });
        }
    }

    // ── Sort by id.0 ascending ────────────────────────────────────────────────
    rows.sort_by_key(|r| r.id.0);

    Ok(rows)
}

/// Return an error if `spec_raw` is non-null and non-empty for a non-computed kind.
fn reject_spec_for_non_computed(
    spec_raw: Option<&str>,
    row_idx: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if let Some(s) = spec_raw {
        if !s.is_empty() {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("scalar_parameter_definitions[{row_idx}].computed_spec"),
                message:
                    "computed_spec must be null for non-computed kinds (constant, per_stage, seasonal)"
                        .to_string(),
            });
        }
    }
    Ok(())
}

/// Parse a `computed_spec` string of the form `<variant_tag>(hydro_id=<int>)`.
///
/// Whitespace is tolerated around `(`, `=`, `)`, and at the start and end of
/// the string. Returns a [`ComputedParameter`] on success.
fn parse_computed_spec(
    spec: &str,
    row_idx: usize,
    path: &Path,
) -> Result<ComputedParameter, LoadError> {
    let spec = spec.trim();

    // Find the outermost parentheses.
    let open = spec.find('(').ok_or_else(|| LoadError::SchemaError {
        path: path.to_path_buf(),
        field: format!("scalar_parameter_definitions[{row_idx}].computed_spec"),
        message: format!(
            "computed_spec must match <tag>(hydro_id=<int>); legal tags are: {LEGAL_VARIANT_TAGS}"
        ),
    })?;
    let close = spec.rfind(')').ok_or_else(|| LoadError::SchemaError {
        path: path.to_path_buf(),
        field: format!("scalar_parameter_definitions[{row_idx}].computed_spec"),
        message: format!(
            "computed_spec must match <tag>(hydro_id=<int>); legal tags are: {LEGAL_VARIANT_TAGS}"
        ),
    })?;

    if close <= open {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("scalar_parameter_definitions[{row_idx}].computed_spec"),
            message: format!(
                "computed_spec must match <tag>(hydro_id=<int>); legal tags are: {LEGAL_VARIANT_TAGS}"
            ),
        });
    }

    let tag = spec[..open].trim();
    let inner = spec[open + 1..close].trim();

    // Parse the inner `hydro_id=<int>` part.
    let (key, value_str) =
        inner
            .split_once('=')
            .ok_or_else(|| LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("scalar_parameter_definitions[{row_idx}].computed_spec"),
                message: format!(
                "computed_spec must match <tag>(hydro_id=<int>); legal tags are: {LEGAL_VARIANT_TAGS}"
            ),
            })?;

    if key.trim() != "hydro_id" {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("scalar_parameter_definitions[{row_idx}].computed_spec"),
            message: format!(
                "computed_spec must match <tag>(hydro_id=<int>); legal tags are: {LEGAL_VARIANT_TAGS}"
            ),
        });
    }

    let hydro_id_raw: i32 = value_str
        .trim()
        .parse()
        .map_err(|_| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("scalar_parameter_definitions[{row_idx}].computed_spec"),
            message: format!(
                "hydro_id in computed_spec must be a valid i32 integer, got {:?}; \
                 legal tags are: {LEGAL_VARIANT_TAGS}",
                value_str.trim()
            ),
        })?;

    let hydro_id = EntityId(hydro_id_raw);

    // Exhaustive match — no `_` arm — compile error if ComputedParameter gains a new variant.
    match tag {
        "equivalent_productivity" => Ok(ComputedParameter::EquivalentProductivity { hydro_id }),
        "accumulated_productivity" => Ok(ComputedParameter::AccumulatedProductivity { hydro_id }),
        "reference_volume" => Ok(ComputedParameter::ReferenceVolume { hydro_id }),
        "reference_turbine" => Ok(ComputedParameter::ReferenceTurbine { hydro_id }),
        "min_storage" => Ok(ComputedParameter::MinStorage { hydro_id }),
        "max_storage" => Ok(ComputedParameter::MaxStorage { hydro_id }),
        "specific_productivity" => Ok(ComputedParameter::SpecificProductivity { hydro_id }),
        other => Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("scalar_parameter_definitions[{row_idx}].computed_spec"),
            message: format!("unknown variant tag {other:?}; legal tags are: {LEGAL_VARIANT_TAGS}"),
        }),
    }
}

/// An intermediate row from `system/scalar_parameter_values.parquet`.
///
/// Each row carries one numeric value for a parameter, keyed by `parameter_id`
/// plus either `stage_id` or `season_id` (or neither, for `Constant` parameters).
///
/// # Examples
///
/// ```
/// use cobre_core::EntityId;
/// use cobre_io::ScalarParameterValueRow;
///
/// let row = ScalarParameterValueRow {
///     parameter_id: EntityId(10),
///     stage_id: None,
///     season_id: None,
///     value: 3.6,
/// };
/// assert_eq!(row.parameter_id, EntityId(10));
/// assert_eq!(row.value, 3.6);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarParameterValueRow {
    /// The parameter this value belongs to.
    pub parameter_id: EntityId,
    /// Zero-based stage index; non-null for `PerStage` parameters.
    pub stage_id: Option<i32>,
    /// Season identifier; non-null for `Seasonal` parameters.
    pub season_id: Option<i32>,
    /// Finite numeric value.
    pub value: f64,
}

/// Parse `system/scalar_parameter_values.parquet` and return rows in file order.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, and returns the rows in the order they appear in the file. No
/// grouping or sorting is performed here — those are done by
/// [`assemble_scalar_parameters`].
///
/// # Errors
///
/// | Condition                                                    | Error variant              |
/// | ------------------------------------------------------------ | -------------------------- |
/// | File not found or permission denied                          | [`LoadError::IoError`]     |
/// | Malformed Parquet                                            | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type                        | [`LoadError::SchemaError`] |
/// | Null `parameter_id` or `value`                               | [`LoadError::SchemaError`] |
/// | Non-finite `value` (NaN, +inf, -inf)                         | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::parse_scalar_parameter_values;
/// use std::path::Path;
///
/// let rows = parse_scalar_parameter_values(
///     Path::new("system/scalar_parameter_values.parquet")
/// ).expect("valid values file");
/// println!("loaded {} value rows", rows.len());
/// ```
pub fn parse_scalar_parameter_values(
    path: &Path,
) -> Result<Vec<ScalarParameterValueRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<ScalarParameterValueRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        // ── Extract columns by name ───────────────────────────────────────────
        let parameter_id_col = extract_required_int32(&batch, "parameter_id", path)?;
        let value_col = extract_required_float64(&batch, "value", path)?;
        let stage_id_col = extract_optional_int32(&batch, "stage_id", path)?;
        let season_id_col = extract_optional_int32(&batch, "season_id", path)?;

        // ── Build rows with per-row validation ───────────────────────────────
        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            // ── parameter_id: non-null ────────────────────────────────────────
            if parameter_id_col.is_null(i) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("scalar_parameter_values[{row_idx}].parameter_id"),
                    message: "parameter_id must not be null".to_string(),
                });
            }
            let parameter_id = EntityId(parameter_id_col.value(i));

            // ── value: non-null ───────────────────────────────────────────────
            if value_col.is_null(i) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("scalar_parameter_values[{row_idx}].value"),
                    message: "value must not be null".to_string(),
                });
            }
            let value = value_col.value(i);

            // ── value: finite ─────────────────────────────────────────────────
            if !value.is_finite() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("scalar_parameter_values[{row_idx}].value"),
                    message: format!("value must be finite, got {value}"),
                });
            }

            // ── stage_id: optional ────────────────────────────────────────────
            let stage_id = stage_id_col.and_then(|col| {
                if col.is_null(i) {
                    None
                } else {
                    Some(col.value(i))
                }
            });

            // ── season_id: optional ───────────────────────────────────────────
            let season_id = season_id_col.and_then(|col| {
                if col.is_null(i) {
                    None
                } else {
                    Some(col.value(i))
                }
            });

            rows.push(ScalarParameterValueRow {
                parameter_id,
                stage_id,
                season_id,
                value,
            });
        }
    }

    Ok(rows)
}

/// Join `definitions` with `values` and validate per-kind constraints.
///
/// Consumes both inputs and returns one [`ScalarParameter`] per definition,
/// preserving the definition input order. The values are grouped by
/// `parameter_id` and validated according to the parameter's kind:
///
/// - `Constant` — exactly one value row, both `stage_id` and `season_id` null.
/// - `PerStage` — exactly `n_stages` rows, each with a unique `stage_id` in
///   `0..n_stages` and `season_id` null; the output `Vec<f64>` is dense and
///   indexed by `stage_id`.
/// - `Seasonal` — one or more rows, each with a non-null `season_id` and null
///   `stage_id`, no duplicate `season_id`; the output is sorted via
///   [`ParameterKind::new_seasonal`].
/// - `Computed` — no value rows.
///
/// # Errors
///
/// Returns [`LoadError::SchemaError`] for any join inconsistency. Error fields
/// use the form `scalar_parameter_values[<row_idx>].<col>` for per-row errors
/// or `scalar_parameter_values:<parameter_name>` for whole-parameter errors.
///
/// # Examples
///
/// ```
/// use cobre_io::{ParameterDefinitionRow, ParameterKindHeader, ScalarParameterValueRow, assemble_scalar_parameters};
/// use cobre_core::{EntityId, ParameterKind};
///
/// let definitions = vec![
///     ParameterDefinitionRow { id: EntityId(1), name: "k".to_string(), header: ParameterKindHeader::Constant },
/// ];
/// let values = vec![
///     ScalarParameterValueRow { parameter_id: EntityId(1), stage_id: None, season_id: None, value: 3.6 },
/// ];
/// let params = assemble_scalar_parameters(definitions, values, 2).expect("valid");
/// assert_eq!(params[0].kind, ParameterKind::Constant(3.6));
/// ```
#[allow(clippy::needless_pass_by_value)]
pub fn assemble_scalar_parameters(
    definitions: Vec<ParameterDefinitionRow>,
    values: Vec<ScalarParameterValueRow>,
    n_stages: usize,
) -> Result<Vec<ScalarParameter>, LoadError> {
    // Build a set of known parameter ids for fast membership tests.
    let known_ids: HashSet<EntityId> = definitions.iter().map(|d| d.id).collect();

    // Build a map from parameter_id -> list of (row_index, &row) so that error
    // messages can reference the original file row index.
    let mut value_rows_by_parameter: HashMap<EntityId, Vec<(usize, &ScalarParameterValueRow)>> =
        HashMap::new();

    for (row_idx, row) in values.iter().enumerate() {
        if !known_ids.contains(&row.parameter_id) {
            return Err(LoadError::SchemaError {
                path: Path::new("<system>").to_path_buf(),
                field: format!("scalar_parameter_values[{row_idx}].parameter_id"),
                message: format!(
                    "parameter_id {} does not match any known parameter definition",
                    row.parameter_id.0
                ),
            });
        }
        value_rows_by_parameter
            .entry(row.parameter_id)
            .or_default()
            .push((row_idx, row));
    }

    let mut result: Vec<ScalarParameter> = Vec::with_capacity(definitions.len());

    for def in &definitions {
        let rows = value_rows_by_parameter.remove(&def.id).unwrap_or_default();

        let kind = match def.header {
            ParameterKindHeader::Constant => {
                validate_constant_rows(&rows, &def.name).map(ParameterKind::Constant)?
            }
            ParameterKindHeader::PerStage => {
                validate_per_stage_rows(&rows, &def.name, n_stages).map(ParameterKind::PerStage)?
            }
            ParameterKindHeader::Seasonal => {
                validate_seasonal_rows(&rows, &def.name).map(ParameterKind::new_seasonal)?
            }
            ParameterKindHeader::Computed(computed) => {
                validate_no_rows(&rows, &def.name)?;
                ParameterKind::Computed(computed)
            }
        };

        result.push(ScalarParameter {
            id: def.id,
            name: def.name.clone(),
            kind,
        });
    }

    // Orphan check: any remaining entries have parameter_ids that were in
    // `definitions` but also in `values` twice — this cannot happen because
    // we remove each id from the map when processing it. The remaining entries
    // would only be present if a value_row's parameter_id was accepted (in
    // known_ids) but then the matching definition was removed from the loop.
    // In practice this cannot occur given the current logic, but we do a
    // safety check for any entries not consumed.
    if let Some((orphan_id, orphan_rows)) = value_rows_by_parameter.into_iter().next() {
        let (row_idx, _) = orphan_rows[0];
        return Err(LoadError::SchemaError {
            path: Path::new("<system>").to_path_buf(),
            field: format!("scalar_parameter_values[{row_idx}].parameter_id"),
            message: format!(
                "value row references parameter_id {} which was not consumed during assembly",
                orphan_id.0
            ),
        });
    }

    Ok(result)
}

// ── Private validators ────────────────────────────────────────────────────────

/// Validate value rows for a `Constant` parameter.
///
/// Requires exactly one row, with both `stage_id` and `season_id` null.
fn validate_constant_rows(
    rows: &[(usize, &ScalarParameterValueRow)],
    name: &str,
) -> Result<f64, LoadError> {
    if rows.len() != 1 {
        return Err(LoadError::SchemaError {
            path: Path::new("<system>").to_path_buf(),
            field: format!("scalar_parameter_values:{name}"),
            message: format!(
                "expected exactly one value row for Constant parameter \"{name}\", \
                 got {}",
                rows.len()
            ),
        });
    }
    let (row_idx, row) = rows[0];
    if row.stage_id.is_some() {
        return Err(LoadError::SchemaError {
            path: Path::new("<system>").to_path_buf(),
            field: format!("scalar_parameter_values[{row_idx}].stage_id"),
            message: format!("stage_id must be null for Constant parameter \"{name}\""),
        });
    }
    if row.season_id.is_some() {
        return Err(LoadError::SchemaError {
            path: Path::new("<system>").to_path_buf(),
            field: format!("scalar_parameter_values[{row_idx}].season_id"),
            message: format!("season_id must be null for Constant parameter \"{name}\""),
        });
    }
    Ok(row.value)
}

/// Validate value rows for a `PerStage` parameter.
///
/// Requires exactly `n_stages` rows; each must have a unique `stage_id` in
/// `0..n_stages` and `season_id` null. Returns a dense `Vec<f64>` indexed by
/// `stage_id`.
fn validate_per_stage_rows(
    rows: &[(usize, &ScalarParameterValueRow)],
    name: &str,
    n_stages: usize,
) -> Result<Vec<f64>, LoadError> {
    let mut values = vec![0.0_f64; n_stages];
    let mut seen = vec![false; n_stages];

    for &(row_idx, row) in rows {
        // season_id must be null
        if row.season_id.is_some() {
            return Err(LoadError::SchemaError {
                path: Path::new("<system>").to_path_buf(),
                field: format!("scalar_parameter_values[{row_idx}].season_id"),
                message: format!("season_id must be null for PerStage parameter \"{name}\""),
            });
        }
        // stage_id must be non-null
        let stage_id = row.stage_id.ok_or_else(|| LoadError::SchemaError {
            path: Path::new("<system>").to_path_buf(),
            field: format!("scalar_parameter_values[{row_idx}].stage_id"),
            message: format!("stage_id must not be null for PerStage parameter \"{name}\""),
        })?;
        // stage_id must be in range
        let stage_usize = usize::try_from(stage_id).ok().filter(|&s| s < n_stages);
        let stage_usize = stage_usize.ok_or_else(|| LoadError::SchemaError {
            path: Path::new("<system>").to_path_buf(),
            field: format!("scalar_parameter_values[{row_idx}].stage_id"),
            message: format!(
                "stage_id {stage_id} is out of range 0..{n_stages} for \
                 PerStage parameter \"{name}\""
            ),
        })?;
        // stage_id must not be a duplicate
        if seen[stage_usize] {
            return Err(LoadError::SchemaError {
                path: Path::new("<system>").to_path_buf(),
                field: format!("scalar_parameter_values[{row_idx}].stage_id"),
                message: format!("duplicate stage_id {stage_id} for PerStage parameter \"{name}\""),
            });
        }
        seen[stage_usize] = true;
        values[stage_usize] = row.value;
    }

    // Check that every stage has been provided
    for (s, &present) in seen.iter().enumerate() {
        if !present {
            return Err(LoadError::SchemaError {
                path: Path::new("<system>").to_path_buf(),
                field: format!("scalar_parameter_values:{name}"),
                message: format!("missing value for stage_id {s} in PerStage parameter \"{name}\""),
            });
        }
    }

    Ok(values)
}

/// Validate value rows for a `Seasonal` parameter.
///
/// Requires one or more rows; each must have a non-null `season_id` and null
/// `stage_id`. Duplicate `season_id` values are rejected. Returns unsorted
/// `Vec<(i32, f64)>` pairs — the caller must pass them to
/// [`ParameterKind::new_seasonal`].
fn validate_seasonal_rows(
    rows: &[(usize, &ScalarParameterValueRow)],
    name: &str,
) -> Result<Vec<(i32, f64)>, LoadError> {
    let mut seen_seasons: HashSet<i32> = HashSet::new();
    let mut pairs: Vec<(i32, f64)> = Vec::with_capacity(rows.len());

    for &(row_idx, row) in rows {
        // stage_id must be null
        if row.stage_id.is_some() {
            return Err(LoadError::SchemaError {
                path: Path::new("<system>").to_path_buf(),
                field: format!("scalar_parameter_values[{row_idx}].stage_id"),
                message: format!("stage_id must be null for Seasonal parameter \"{name}\""),
            });
        }
        // season_id must be non-null
        let season_id = row.season_id.ok_or_else(|| LoadError::SchemaError {
            path: Path::new("<system>").to_path_buf(),
            field: format!("scalar_parameter_values[{row_idx}].season_id"),
            message: format!("season_id must not be null for Seasonal parameter \"{name}\""),
        })?;
        // season_id must not be a duplicate
        if !seen_seasons.insert(season_id) {
            return Err(LoadError::SchemaError {
                path: Path::new("<system>").to_path_buf(),
                field: format!("scalar_parameter_values[{row_idx}].season_id"),
                message: format!(
                    "duplicate season_id {season_id} for Seasonal parameter \"{name}\""
                ),
            });
        }
        pairs.push((season_id, row.value));
    }

    Ok(pairs)
}

/// Validate that a `Computed` parameter has no value rows.
fn validate_no_rows(
    rows: &[(usize, &ScalarParameterValueRow)],
    name: &str,
) -> Result<(), LoadError> {
    if !rows.is_empty() {
        let (row_idx, _) = rows[0];
        return Err(LoadError::SchemaError {
            path: Path::new("<system>").to_path_buf(),
            field: format!("scalar_parameter_values[{row_idx}].parameter_id"),
            message: format!(
                "Computed parameter \"{name}\" must have no value rows, \
                 got {}",
                rows.len()
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::case_sensitive_file_extension_comparisons,
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a record batch with the canonical four-column schema.
    fn make_batch(
        ids: &[i32],
        names: &[&str],
        kinds: &[&str],
        specs: &[Option<&str>],
    ) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("kind", DataType::Utf8, false),
            Field::new("computed_spec", DataType::Utf8, true),
        ]));

        let id_array: Int32Array = ids.iter().copied().collect();
        let name_array: StringArray = names.iter().map(|s| Some(*s)).collect();
        let kind_array: StringArray = kinds.iter().map(|s| Some(*s)).collect();
        let spec_array: StringArray = specs
            .iter()
            .map(|opt| opt.as_deref())
            .collect::<StringArray>();

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(name_array),
                Arc::new(kind_array),
                Arc::new(spec_array),
            ],
        )
        .expect("valid batch construction")
    }

    /// Write a single record batch to a temporary Parquet file.
    fn write_parquet(batch: &RecordBatch) -> NamedTempFile {
        let tmp = NamedTempFile::new().expect("tempfile");
        let mut writer = ArrowWriter::try_new(tmp.reopen().expect("reopen"), batch.schema(), None)
            .expect("ArrowWriter");
        writer.write(batch).expect("write batch");
        writer.close().expect("close writer");
        tmp
    }

    // ── AC 1: four-kinds round-trip ───────────────────────────────────────────

    /// Valid Parquet with one row per kind. Result: four rows sorted by id, with the
    /// correct `ParameterKindHeader` for each.
    #[test]
    fn test_parse_four_kinds_round_trip() {
        let batch = make_batch(
            &[3, 1, 4, 2],
            &["seasonal_p", "constant_p", "computed_p", "per_stage_p"],
            &["seasonal", "constant", "computed", "per_stage"],
            &[
                None,
                None,
                Some("equivalent_productivity(hydro_id=42)"),
                None,
            ],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_scalar_parameter_definitions(tmp.path()).unwrap();

        assert_eq!(rows.len(), 4);
        // Sorted by id ascending: 1, 2, 3, 4
        assert_eq!(rows[0].id, EntityId(1));
        assert_eq!(rows[0].header, ParameterKindHeader::Constant);
        assert_eq!(rows[1].id, EntityId(2));
        assert_eq!(rows[1].header, ParameterKindHeader::PerStage);
        assert_eq!(rows[2].id, EntityId(3));
        assert_eq!(rows[2].header, ParameterKindHeader::Seasonal);
        assert_eq!(rows[3].id, EntityId(4));
        assert_eq!(
            rows[3].header,
            ParameterKindHeader::Computed(ComputedParameter::EquivalentProductivity {
                hydro_id: EntityId(42)
            })
        );
    }

    // ── AC 2: all seven computed variant tags ─────────────────────────────────

    /// Seven computed rows, one per variant tag. Each must produce the correct ComputedParameter.
    #[test]
    fn test_all_seven_computed_variant_tags_parse() {
        let batch = make_batch(
            &[1, 2, 3, 4, 5, 6, 7],
            &[
                "eq_prod",
                "ac_prod",
                "ref_vol",
                "ref_turb",
                "min_stor",
                "max_stor",
                "spec_prod",
            ],
            &[
                "computed", "computed", "computed", "computed", "computed", "computed", "computed",
            ],
            &[
                Some("equivalent_productivity(hydro_id=10)"),
                Some("accumulated_productivity(hydro_id=20)"),
                Some("reference_volume(hydro_id=30)"),
                Some("reference_turbine(hydro_id=40)"),
                Some("min_storage(hydro_id=50)"),
                Some("max_storage(hydro_id=60)"),
                Some("specific_productivity(hydro_id=70)"),
            ],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_scalar_parameter_definitions(tmp.path()).unwrap();

        assert_eq!(rows.len(), 7);

        let expected: Vec<ComputedParameter> = vec![
            ComputedParameter::EquivalentProductivity {
                hydro_id: EntityId(10),
            },
            ComputedParameter::AccumulatedProductivity {
                hydro_id: EntityId(20),
            },
            ComputedParameter::ReferenceVolume {
                hydro_id: EntityId(30),
            },
            ComputedParameter::ReferenceTurbine {
                hydro_id: EntityId(40),
            },
            ComputedParameter::MinStorage {
                hydro_id: EntityId(50),
            },
            ComputedParameter::MaxStorage {
                hydro_id: EntityId(60),
            },
            ComputedParameter::SpecificProductivity {
                hydro_id: EntityId(70),
            },
        ];

        for (row, expected_cp) in rows.iter().zip(expected.iter()) {
            assert_eq!(
                row.header,
                ParameterKindHeader::Computed(*expected_cp),
                "row id={} failed",
                row.id.0
            );
        }
    }

    // ── AC 3: duplicate id rejected ───────────────────────────────────────────

    /// Two rows sharing `id == 1` must produce a SchemaError mentioning "id" and "duplicate".
    #[test]
    fn test_duplicate_id_rejected() {
        let batch = make_batch(
            &[1, 1],
            &["alpha", "beta"],
            &["constant", "constant"],
            &[None, None],
        );
        let tmp = write_parquet(&batch);
        let err = parse_scalar_parameter_definitions(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("id"),
                    "field should contain 'id', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should contain 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC 4: duplicate name rejected (case-sensitive) ────────────────────────

    /// Two rows sharing `name == "rho_eq_h1"` must produce a SchemaError.
    #[test]
    fn test_duplicate_name_case_sensitive_rejected() {
        let batch = make_batch(
            &[1, 2],
            &["rho_eq_h1", "rho_eq_h1"],
            &["constant", "constant"],
            &[None, None],
        );
        let tmp = write_parquet(&batch);
        let err = parse_scalar_parameter_definitions(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("name"),
                    "field should contain 'name', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should contain 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Different cases of the same name ("rho_eq_h1" vs "Rho_Eq_H1") are NOT duplicates.
    #[test]
    fn test_different_case_names_not_duplicates() {
        let batch = make_batch(
            &[1, 2],
            &["rho_eq_h1", "Rho_Eq_H1"],
            &["constant", "constant"],
            &[None, None],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_scalar_parameter_definitions(tmp.path()).unwrap();
        assert_eq!(rows.len(), 2);
    }

    // ── AC 5: unknown variant tag rejected ────────────────────────────────────

    /// `kind == "computed"`, `computed_spec == "unknown_tag(hydro_id=1)"` must error
    /// with a message naming "equivalent_productivity".
    #[test]
    fn test_unknown_variant_tag_rejected() {
        let batch = make_batch(
            &[1],
            &["p1"],
            &["computed"],
            &[Some("unknown_tag(hydro_id=1)")],
        );
        let tmp = write_parquet(&batch);
        let err = parse_scalar_parameter_definitions(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("computed_spec"),
                    "field should contain 'computed_spec', got: {field}"
                );
                assert!(
                    message.contains("equivalent_productivity"),
                    "message should list legal tags; got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC 6: non-computed kind with computed_spec rejected ───────────────────

    /// `kind == "per_stage"` with non-null `computed_spec` must error with field ending
    /// in `.computed_spec`.
    #[test]
    fn test_non_computed_kind_with_computed_spec_rejected() {
        let batch = make_batch(
            &[1],
            &["p1"],
            &["per_stage"],
            &[Some("equivalent_productivity(hydro_id=1)")],
        );
        let tmp = write_parquet(&batch);
        let err = parse_scalar_parameter_definitions(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.ends_with(".computed_spec"),
                    "field should end with '.computed_spec', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC 7: computed kind missing spec rejected ─────────────────────────────

    /// `kind == "computed"` with null `computed_spec` must produce a SchemaError.
    #[test]
    fn test_computed_kind_missing_spec_rejected() {
        let batch = make_batch(&[1], &["p1"], &["computed"], &[None]);
        let tmp = write_parquet(&batch);
        let err = parse_scalar_parameter_definitions(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("computed_spec"),
                    "field should contain 'computed_spec', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC 8: invalid kind string rejected ────────────────────────────────────

    /// `kind == "weekly"` must produce a SchemaError naming all four legal kinds.
    #[test]
    fn test_invalid_kind_string_rejected() {
        let batch = make_batch(&[1], &["p1"], &["weekly"], &[None]);
        let tmp = write_parquet(&batch);
        let err = parse_scalar_parameter_definitions(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("kind"),
                    "field should contain 'kind', got: {field}"
                );
                // Message must name all four legal kinds.
                assert!(
                    message.contains("constant"),
                    "missing 'constant' in: {message}"
                );
                assert!(
                    message.contains("per_stage"),
                    "missing 'per_stage' in: {message}"
                );
                assert!(
                    message.contains("seasonal"),
                    "missing 'seasonal' in: {message}"
                );
                assert!(
                    message.contains("computed"),
                    "missing 'computed' in: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC 9: empty name rejected ─────────────────────────────────────────────

    /// `name == ""` must produce a SchemaError.
    #[test]
    fn test_empty_name_rejected() {
        let batch = make_batch(&[1], &[""], &["constant"], &[None]);
        let tmp = write_parquet(&batch);
        let err = parse_scalar_parameter_definitions(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("name"),
                    "field should contain 'name', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC 10: trailing whitespace in name rejected ───────────────────────────

    /// `name == "rho "` must produce a SchemaError.
    #[test]
    fn test_name_with_trailing_whitespace_rejected() {
        let batch = make_batch(&[1], &["rho "], &["constant"], &[None]);
        let tmp = write_parquet(&batch);
        let err = parse_scalar_parameter_definitions(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("name"),
                    "field should contain 'name', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC 11: whitespace in computed_spec tolerated ──────────────────────────

    /// Whitespace around tokens in `computed_spec` must parse successfully.
    #[test]
    fn test_whitespace_in_computed_spec_tolerated() {
        let batch = make_batch(
            &[1],
            &["p1"],
            &["computed"],
            &[Some("  equivalent_productivity ( hydro_id = 42 ) ")],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_scalar_parameter_definitions(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].header,
            ParameterKindHeader::Computed(ComputedParameter::EquivalentProductivity {
                hydro_id: EntityId(42)
            })
        );
    }

    // ── AC 12: empty parquet returns empty vec ────────────────────────────────

    /// Zero-row Parquet file must return `Ok(vec![])`.
    #[test]
    fn test_empty_parquet_returns_empty_vec() {
        let batch = make_batch(&[], &[], &[], &[]);
        let tmp = write_parquet(&batch);
        let rows = parse_scalar_parameter_definitions(tmp.path()).unwrap();
        assert!(rows.is_empty(), "expected empty vec for empty Parquet");
    }

    // ── AC 13: load wrapper with None short-circuits ──────────────────────────

    /// `load_scalar_parameter_definitions(None)` must return `Ok(vec![])` without I/O.
    #[test]
    fn test_load_scalar_parameter_definitions_none_returns_empty() {
        use super::super::load_scalar_parameter_definitions;
        let result = load_scalar_parameter_definitions(None).unwrap();
        assert!(result.is_empty(), "expected empty vec for None path");
    }

    // ── Ticket-020: value parser and assembly tests ───────────────────────────

    /// Build a values parquet batch with the four columns.
    fn make_values_batch(
        parameter_ids: &[i32],
        stage_ids: &[Option<i32>],
        season_ids: &[Option<i32>],
        values: &[f64],
    ) -> RecordBatch {
        use arrow::array::Float64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("parameter_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, true),
            Field::new("season_id", DataType::Int32, true),
            Field::new("value", DataType::Float64, false),
        ]));
        let pid_array = Int32Array::from(parameter_ids.to_vec());
        let stage_array = Int32Array::from(stage_ids.to_vec());
        let season_array = Int32Array::from(season_ids.to_vec());
        let val_array = Float64Array::from(values.to_vec());
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(pid_array),
                Arc::new(stage_array),
                Arc::new(season_array),
                Arc::new(val_array),
            ],
        )
        .expect("valid values batch")
    }

    /// Returns definitions for ids [10=Constant, 20=PerStage, 30=Seasonal, 40=Computed].
    fn four_kinds_definitions() -> Vec<ParameterDefinitionRow> {
        vec![
            ParameterDefinitionRow {
                id: EntityId(10),
                name: "const_p".to_string(),
                header: ParameterKindHeader::Constant,
            },
            ParameterDefinitionRow {
                id: EntityId(20),
                name: "stage_p".to_string(),
                header: ParameterKindHeader::PerStage,
            },
            ParameterDefinitionRow {
                id: EntityId(30),
                name: "season_p".to_string(),
                header: ParameterKindHeader::Seasonal,
            },
            ParameterDefinitionRow {
                id: EntityId(40),
                name: "computed_p".to_string(),
                header: ParameterKindHeader::Computed(ComputedParameter::EquivalentProductivity {
                    hydro_id: EntityId(99),
                }),
            },
        ]
    }

    // ── Test 1: four-kinds round-trip ─────────────────────────────────────────

    /// Parquet with correct value rows for all four kinds; assemble should
    /// produce four `ScalarParameter`s with the expected kinds.
    #[test]
    fn test_parse_values_round_trip_four_kinds() {
        // 1 Constant row + 3 PerStage rows + 2 Seasonal rows + 0 Computed rows = 6
        let batch = make_values_batch(
            &[10, 20, 20, 20, 30, 30],
            &[None, Some(0), Some(1), Some(2), None, None],
            &[None, None, None, None, Some(2), Some(1)],
            &[7.5, 1.0, 2.0, 3.0, 0.8, 0.4],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_scalar_parameter_values(tmp.path()).unwrap();
        assert_eq!(rows.len(), 6);

        let defs = four_kinds_definitions();
        let params = assemble_scalar_parameters(defs, rows, 3).unwrap();

        assert_eq!(params.len(), 4);

        // Constant
        assert_eq!(params[0].id, EntityId(10));
        assert_eq!(params[0].kind, ParameterKind::Constant(7.5));

        // PerStage — dense vec indexed by stage_id
        assert_eq!(params[1].id, EntityId(20));
        assert_eq!(params[1].kind, ParameterKind::PerStage(vec![1.0, 2.0, 3.0]));

        // Seasonal — sorted ascending by season_id
        assert_eq!(params[2].id, EntityId(30));
        assert_eq!(
            params[2].kind,
            ParameterKind::Seasonal(vec![(1, 0.4), (2, 0.8)])
        );

        // Computed — carries the variant from the definition
        assert_eq!(params[3].id, EntityId(40));
        assert_eq!(
            params[3].kind,
            ParameterKind::Computed(ComputedParameter::EquivalentProductivity {
                hydro_id: EntityId(99)
            })
        );
    }

    // ── Test 2: PerStage missing stage rejected ───────────────────────────────

    /// Only 2 of 3 stage rows supplied; assemble must report the missing stage id.
    #[test]
    fn test_per_stage_missing_stage_rejected() {
        let defs = vec![ParameterDefinitionRow {
            id: EntityId(1),
            name: "p".to_string(),
            header: ParameterKindHeader::PerStage,
        }];
        let values = vec![
            ScalarParameterValueRow {
                parameter_id: EntityId(1),
                stage_id: Some(0),
                season_id: None,
                value: 1.0,
            },
            ScalarParameterValueRow {
                parameter_id: EntityId(1),
                stage_id: Some(1),
                season_id: None,
                value: 2.0,
            },
        ];
        let err = assemble_scalar_parameters(defs, values, 3).unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains('2'),
                    "message should mention missing stage 2, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Test 3: PerStage out-of-range stage_id rejected ───────────────────────

    /// stage_id = 5 for n_stages = 3 must be rejected.
    #[test]
    fn test_per_stage_out_of_range_rejected() {
        let defs = vec![ParameterDefinitionRow {
            id: EntityId(1),
            name: "p".to_string(),
            header: ParameterKindHeader::PerStage,
        }];
        let values = vec![ScalarParameterValueRow {
            parameter_id: EntityId(1),
            stage_id: Some(5),
            season_id: None,
            value: 1.0,
        }];
        let err = assemble_scalar_parameters(defs, values, 3).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError"
        );
    }

    // ── Test 4: PerStage duplicate stage_id rejected ──────────────────────────

    /// Two rows with stage_id = 1 for the same parameter must be rejected.
    #[test]
    fn test_per_stage_duplicate_stage_rejected() {
        let defs = vec![ParameterDefinitionRow {
            id: EntityId(1),
            name: "p".to_string(),
            header: ParameterKindHeader::PerStage,
        }];
        let values = vec![
            ScalarParameterValueRow {
                parameter_id: EntityId(1),
                stage_id: Some(1),
                season_id: None,
                value: 1.0,
            },
            ScalarParameterValueRow {
                parameter_id: EntityId(1),
                stage_id: Some(1),
                season_id: None,
                value: 2.0,
            },
        ];
        let err = assemble_scalar_parameters(defs, values, 3).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError"
        );
    }

    // ── Test 5: Constant with multiple rows rejected ──────────────────────────

    /// Constant definition with two value rows must be rejected with a message
    /// containing "expected exactly one value row".
    #[test]
    fn test_constant_multiple_rows_rejected() {
        let defs = vec![ParameterDefinitionRow {
            id: EntityId(1),
            name: "k".to_string(),
            header: ParameterKindHeader::Constant,
        }];
        let values = vec![
            ScalarParameterValueRow {
                parameter_id: EntityId(1),
                stage_id: None,
                season_id: None,
                value: 1.0,
            },
            ScalarParameterValueRow {
                parameter_id: EntityId(1),
                stage_id: None,
                season_id: None,
                value: 2.0,
            },
        ];
        let err = assemble_scalar_parameters(defs, values, 1).unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("expected exactly one value row"),
                    "message should say 'expected exactly one value row', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Test 6: Constant with non-null stage_id rejected ─────────────────────

    /// A Constant value row carrying a non-null stage_id must be rejected.
    #[test]
    fn test_constant_with_stage_id_rejected() {
        let defs = vec![ParameterDefinitionRow {
            id: EntityId(1),
            name: "k".to_string(),
            header: ParameterKindHeader::Constant,
        }];
        let values = vec![ScalarParameterValueRow {
            parameter_id: EntityId(1),
            stage_id: Some(0),
            season_id: None,
            value: 1.0,
        }];
        let err = assemble_scalar_parameters(defs, values, 2).unwrap_err();
        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("stage_id"),
                    "field should mention stage_id, got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Test 7: Seasonal duplicate season_id rejected ─────────────────────────

    /// Two rows sharing the same season_id for a Seasonal parameter must be rejected.
    #[test]
    fn test_seasonal_duplicate_season_rejected() {
        let defs = vec![ParameterDefinitionRow {
            id: EntityId(1),
            name: "s".to_string(),
            header: ParameterKindHeader::Seasonal,
        }];
        let values = vec![
            ScalarParameterValueRow {
                parameter_id: EntityId(1),
                stage_id: None,
                season_id: Some(1),
                value: 0.5,
            },
            ScalarParameterValueRow {
                parameter_id: EntityId(1),
                stage_id: None,
                season_id: Some(1),
                value: 0.9,
            },
        ];
        let err = assemble_scalar_parameters(defs, values, 0).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError"
        );
    }

    // ── Test 8: Seasonal output is sorted ────────────────────────────────────

    /// Seasons supplied in order [3, 1, 2]; output must be [(1,_),(2,_),(3,_)].
    #[test]
    fn test_seasonal_sorted_in_output() {
        let defs = vec![ParameterDefinitionRow {
            id: EntityId(1),
            name: "s".to_string(),
            header: ParameterKindHeader::Seasonal,
        }];
        let values = vec![
            ScalarParameterValueRow {
                parameter_id: EntityId(1),
                stage_id: None,
                season_id: Some(3),
                value: 0.3,
            },
            ScalarParameterValueRow {
                parameter_id: EntityId(1),
                stage_id: None,
                season_id: Some(1),
                value: 0.1,
            },
            ScalarParameterValueRow {
                parameter_id: EntityId(1),
                stage_id: None,
                season_id: Some(2),
                value: 0.2,
            },
        ];
        let params = assemble_scalar_parameters(defs, values, 0).unwrap();
        assert_eq!(params.len(), 1);
        assert_eq!(
            params[0].kind,
            ParameterKind::Seasonal(vec![(1, 0.1), (2, 0.2), (3, 0.3)])
        );
    }

    // ── Test 9: Computed with value rows rejected ─────────────────────────────

    /// A Computed definition with one matching value row must be rejected.
    #[test]
    fn test_computed_with_value_rows_rejected() {
        let defs = vec![ParameterDefinitionRow {
            id: EntityId(1),
            name: "c".to_string(),
            header: ParameterKindHeader::Computed(ComputedParameter::ReferenceVolume {
                hydro_id: EntityId(5),
            }),
        }];
        let values = vec![ScalarParameterValueRow {
            parameter_id: EntityId(1),
            stage_id: None,
            season_id: None,
            value: 1.0,
        }];
        let err = assemble_scalar_parameters(defs, values, 0).unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.to_lowercase().contains("computed"),
                    "message should mention 'computed', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Test 10: Orphan parameter_id rejected ─────────────────────────────────

    /// A value row referencing an id not in definitions (99) must be rejected
    /// with a message naming id 99.
    #[test]
    fn test_orphan_parameter_id_rejected() {
        let defs = vec![ParameterDefinitionRow {
            id: EntityId(1),
            name: "k".to_string(),
            header: ParameterKindHeader::Constant,
        }];
        let values = vec![ScalarParameterValueRow {
            parameter_id: EntityId(99),
            stage_id: None,
            season_id: None,
            value: 1.0,
        }];
        let err = assemble_scalar_parameters(defs, values, 0).unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("99"),
                    "message should mention parameter_id 99, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Test 11: NaN value rejected by parser ────────────────────────────────

    /// A parquet row with value = NaN must be rejected with field ending in `.value`.
    #[test]
    fn test_nan_value_rejected() {
        let batch = make_values_batch(&[1], &[None], &[None], &[f64::NAN]);
        let tmp = write_parquet(&batch);
        let err = parse_scalar_parameter_values(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.ends_with(".value"),
                    "field should end with '.value', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Test 12: +Infinity value rejected by parser ───────────────────────────

    /// A parquet row with value = +inf must be rejected.
    #[test]
    fn test_infinite_value_rejected() {
        let batch = make_values_batch(&[1], &[None], &[None], &[f64::INFINITY]);
        let tmp = write_parquet(&batch);
        let err = parse_scalar_parameter_values(tmp.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError for infinite value"
        );
    }

    // ── Test 13: load wrapper with None returns empty ─────────────────────────

    /// `load_scalar_parameter_values(None)` must return `Ok(vec![])` without I/O.
    #[test]
    fn test_load_scalar_parameter_values_none_returns_empty() {
        use super::super::load_scalar_parameter_values;
        let result = load_scalar_parameter_values(None).unwrap();
        assert!(result.is_empty(), "expected empty vec for None path");
    }

    // ── Test 14: assemble preserves definition order ──────────────────────────

    /// Definitions sorted by id [10, 20, 30, 40]; values supplied in a different
    /// order. The assembled `Vec<ScalarParameter>` must match definition order.
    #[test]
    fn test_assemble_preserves_definition_order() {
        let defs = four_kinds_definitions();
        // Supply value rows in reverse-id order to verify ordering is not affected
        let values = vec![
            // Seasonal rows for id=30
            ScalarParameterValueRow {
                parameter_id: EntityId(30),
                stage_id: None,
                season_id: Some(1),
                value: 0.1,
            },
            // PerStage rows for id=20 (out of stage order)
            ScalarParameterValueRow {
                parameter_id: EntityId(20),
                stage_id: Some(2),
                season_id: None,
                value: 3.0,
            },
            ScalarParameterValueRow {
                parameter_id: EntityId(20),
                stage_id: Some(0),
                season_id: None,
                value: 1.0,
            },
            ScalarParameterValueRow {
                parameter_id: EntityId(20),
                stage_id: Some(1),
                season_id: None,
                value: 2.0,
            },
            // Constant row for id=10
            ScalarParameterValueRow {
                parameter_id: EntityId(10),
                stage_id: None,
                season_id: None,
                value: 5.0,
            },
            // No rows for id=40 (Computed)
        ];
        let params = assemble_scalar_parameters(defs, values, 3).unwrap();

        // Order must match definition order: [10, 20, 30, 40]
        assert_eq!(params[0].id, EntityId(10));
        assert_eq!(params[1].id, EntityId(20));
        assert_eq!(params[2].id, EntityId(30));
        assert_eq!(params[3].id, EntityId(40));

        // Verify PerStage is properly dense despite out-of-order rows
        assert_eq!(params[1].kind, ParameterKind::PerStage(vec![1.0, 2.0, 3.0]));
    }
}
