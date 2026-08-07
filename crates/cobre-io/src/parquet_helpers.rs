//! Shared Parquet column extraction helpers centralising the typed-downcast
//! logic used by every Parquet parser in `cobre-io`.

use arrow::array::{Array, ArrayRef, Date32Array, Float64Array, Int32Array, UInt32Array};
use arrow::record_batch::RecordBatch;
use std::path::Path;

use crate::LoadError;

/// Extract a required column as [`Int32Array`] by name.
///
/// Returns `SchemaError` if the column is absent or has the wrong Arrow type.
pub(crate) fn extract_required_int32<'a>(
    batch: &'a RecordBatch,
    name: &str,
    path: &Path,
) -> Result<&'a Int32Array, LoadError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!("missing required column \"{name}\""),
        })?;
    col.as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!(
                "column \"{name}\" has type {} but Int32 is required",
                col.data_type()
            ),
        })
}

/// Extract a required column as [`Float64Array`] by name.
///
/// Returns `SchemaError` if the column is absent or has the wrong Arrow type.
pub(crate) fn extract_required_float64<'a>(
    batch: &'a RecordBatch,
    name: &str,
    path: &Path,
) -> Result<&'a Float64Array, LoadError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!("missing required column \"{name}\""),
        })?;
    col.as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!(
                "column \"{name}\" has type {} but Float64 is required",
                col.data_type()
            ),
        })
}

/// Locate the first present column among `names`, preferred spelling first.
///
/// Returns the matched index into `names` and the column. `SchemaError` names
/// the preferred (first) spelling when none is present — accepting a legacy
/// alias must never silently read the wrong column.
fn resolve_required_column<'a>(
    batch: &'a RecordBatch,
    names: &[&str],
    path: &Path,
) -> Result<(usize, &'a ArrayRef), LoadError> {
    for (idx, &name) in names.iter().enumerate() {
        if let Some(col) = batch.column_by_name(name) {
            return Ok((idx, col));
        }
    }
    let preferred = names.first().copied().unwrap_or_default();
    Err(LoadError::SchemaError {
        path: path.to_path_buf(),
        field: preferred.to_string(),
        message: format!("missing required column \"{preferred}\""),
    })
}

/// Extract a required column as [`Int32Array`], accepting any of `names`
/// (preferred spelling first, legacy aliases after).
///
/// Returns `SchemaError` naming the preferred spelling when no listed column is
/// present, or naming the matched column when it has the wrong Arrow type.
pub(crate) fn extract_required_int32_aliased<'a>(
    batch: &'a RecordBatch,
    names: &[&str],
    path: &Path,
) -> Result<&'a Int32Array, LoadError> {
    let (idx, col) = resolve_required_column(batch, names, path)?;
    let found = names[idx];
    col.as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: found.to_string(),
            message: format!(
                "column \"{found}\" has type {} but Int32 is required",
                col.data_type()
            ),
        })
}

/// Extract a required column as [`Float64Array`], accepting any of `names`
/// (preferred spelling first, legacy aliases after).
///
/// Returns `SchemaError` naming the preferred spelling when no listed column is
/// present, or naming the matched column when it has the wrong Arrow type.
pub(crate) fn extract_required_float64_aliased<'a>(
    batch: &'a RecordBatch,
    names: &[&str],
    path: &Path,
) -> Result<&'a Float64Array, LoadError> {
    let (idx, col) = resolve_required_column(batch, names, path)?;
    let found = names[idx];
    col.as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: found.to_string(),
            message: format!(
                "column \"{found}\" has type {} but Float64 is required",
                col.data_type()
            ),
        })
}

/// Extract an optional column as [`Int32Array`] by name, returning `None` if absent.
///
/// Returns `SchemaError` if the column exists but has the wrong Arrow type.
pub(crate) fn extract_optional_int32<'a>(
    batch: &'a RecordBatch,
    name: &str,
    path: &Path,
) -> Result<Option<&'a Int32Array>, LoadError> {
    let Some(col) = batch.column_by_name(name) else {
        return Ok(None);
    };
    let arr = col
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!(
                "column \"{name}\" has type {} but Int32 is required",
                col.data_type()
            ),
        })?;
    Ok(Some(arr))
}

/// Extract an optional column as [`Float64Array`] by name, returning `None` if absent.
///
/// Returns `SchemaError` if the column exists but has the wrong Arrow type.
pub(crate) fn extract_optional_float64<'a>(
    batch: &'a RecordBatch,
    name: &str,
    path: &Path,
) -> Result<Option<&'a Float64Array>, LoadError> {
    let Some(col) = batch.column_by_name(name) else {
        return Ok(None);
    };
    let arr =
        col.as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| LoadError::SchemaError {
                path: path.to_path_buf(),
                field: name.to_string(),
                message: format!(
                    "column \"{name}\" has type {} but Float64 is required",
                    col.data_type()
                ),
            })?;
    Ok(Some(arr))
}

/// Extract a required column as [`UInt32Array`] by name.
///
/// Returns `SchemaError` if the column is absent or has the wrong Arrow type.
pub(crate) fn extract_required_uint32<'a>(
    batch: &'a RecordBatch,
    name: &str,
    path: &Path,
) -> Result<&'a UInt32Array, LoadError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!("missing required column \"{name}\""),
        })?;
    col.as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!(
                "column \"{name}\" has type {} but UInt32 is required",
                col.data_type()
            ),
        })
}

/// Extract a required column as [`Date32Array`] by name.
///
/// Returns `SchemaError` if the column is absent or has the wrong Arrow type.
pub(crate) fn extract_required_date32<'a>(
    batch: &'a RecordBatch,
    name: &str,
    path: &Path,
) -> Result<&'a Date32Array, LoadError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!("missing required column \"{name}\""),
        })?;
    col.as_any()
        .downcast_ref::<Date32Array>()
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!(
                "column \"{name}\" has type {} but Date32 is required",
                col.data_type()
            ),
        })
}
