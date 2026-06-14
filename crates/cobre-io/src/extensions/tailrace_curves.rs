//! Parsing for `system/tailrace_curves.parquet` — piecewise-quartic
//! tailrace-level curves `h_jus(q_jus)`.
//!
//! [`parse_tailrace_curves`] reads `system/tailrace_curves.parquet` from the
//! case directory and returns a flat, sorted `Vec<TailraceCurveRow>`.
//!
//! ## Parquet schema
//!
//! | Column          | Parquet type    | Description                                          |
//! |---------------- |---------------- |------------------------------------------------------|
//! | `hydro_id`      | INT32           | Plant whose tailrace this describes                  |
//! | `family_id`     | INT32           | Family index within the plant (sequential)           |
//! | `href_jus_m`    | DOUBLE nullable | Downstream reference level keying the family (m)      |
//! | `segment_id`    | INT32           | Piece index within the family                         |
//! | `q_jus_inf_m3s` | DOUBLE          | Segment lower validity bound (m³/s)                  |
//! | `q_jus_sup_m3s` | DOUBLE          | Segment upper validity bound (m³/s)                  |
//! | `a_cf0`..`a_cf4`| DOUBLE          | Degree-4 polynomial coefficients (ascending power)   |
//!
//! ## Output ordering
//!
//! Rows are sorted by `(hydro_id, family_id, segment_id)` ascending using the
//! integer keys only. No floating-point column participates in the sort key, so
//! the order is exact and declaration-order-invariant.
//!
//! ## Validation
//!
//! Per-row constraints enforced by this parser:
//!
//! - All twelve columns must be present with the correct Arrow types
//!   (`href_jus_m` is nullable Float64; all other columns are non-nullable).
//! - `q_jus_inf_m3s` and `q_jus_sup_m3s` must be non-negative and finite.
//! - `q_jus_sup_m3s >= q_jus_inf_m3s` (a segment window is non-inverted).
//! - Each of `a_cf0`..`a_cf4` must be finite; any sign is accepted, because the
//!   higher-degree coefficients are routinely negative in the source data.
//! - `href_jus_m`, when non-null, must be non-negative and finite.
//!
//! Deferred validations (not performed here):
//!
//! - Cross-segment contiguity / C⁰-continuity within a family — a later layer.
//! - Family selection and downstream-level resolution — a later layer.
//! - `hydro_id` existence in the hydro registry — Layer 3.

use arrow::array::{Array, Float64Array, Int32Array};
use cobre_core::EntityId;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

use crate::LoadError;

/// A single row from `system/tailrace_curves.parquet`.
///
/// Each row is one segment of the piecewise-quartic tailrace curve for the plant
/// identified by `hydro_id`. A complete curve for one family consists of multiple
/// rows sharing `(hydro_id, family_id)`, sorted by ascending `segment_id`.
///
/// # Examples
///
/// ```
/// use cobre_io::extensions::TailraceCurveRow;
/// use cobre_core::EntityId;
///
/// let row = TailraceCurveRow {
///     hydro_id: EntityId::from(42),
///     family_id: 1,
///     href_jus_m: Some(885.3),
///     segment_id: 1,
///     q_jus_inf_m3s: 0.0,
///     q_jus_sup_m3s: 1500.0,
///     a_cf0: 320.0,
///     a_cf1: 1.0e-3,
///     a_cf2: -3.1521e-17,
///     a_cf3: 0.0,
///     a_cf4: 0.0,
/// };
/// assert_eq!(row.hydro_id, EntityId::from(42));
/// assert_eq!(row.segment_id, 1);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TailraceCurveRow {
    /// Plant this tailrace curve belongs to.
    pub hydro_id: EntityId,
    /// Family index within the plant (sequential grouping key).
    pub family_id: i32,
    /// Downstream reference level keying the family (m). `None` when the source
    /// carries no reference level (the plant has a single family).
    pub href_jus_m: Option<f64>,
    /// Piece index within the family.
    pub segment_id: i32,
    /// Segment lower validity bound (m³/s). Non-negative.
    pub q_jus_inf_m3s: f64,
    /// Segment upper validity bound (m³/s). Non-negative, `>= q_jus_inf_m3s`.
    pub q_jus_sup_m3s: f64,
    /// Degree-0 polynomial coefficient. Any sign.
    pub a_cf0: f64,
    /// Degree-1 polynomial coefficient. Any sign.
    pub a_cf1: f64,
    /// Degree-2 polynomial coefficient. Any sign.
    pub a_cf2: f64,
    /// Degree-3 polynomial coefficient. Any sign.
    pub a_cf3: f64,
    /// Degree-4 polynomial coefficient. Any sign.
    pub a_cf4: f64,
}

/// Parse `system/tailrace_curves.parquet` and return a sorted segment table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by
/// `(hydro_id, family_id, segment_id)` ascending.
///
/// # Errors
///
/// | Condition                                          | Error variant              |
/// |--------------------------------------------------- |--------------------------- |
/// | File not found or permission denied                | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)           | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type              | [`LoadError::SchemaError`] |
/// | Negative / non-finite bound, inverted window, or non-finite coefficient | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::extensions::parse_tailrace_curves;
/// use std::path::Path;
///
/// let rows = parse_tailrace_curves(Path::new("system/tailrace_curves.parquet"))
///     .expect("valid tailrace file");
/// println!("loaded {} tailrace segments", rows.len());
/// ```
pub fn parse_tailrace_curves(path: &Path) -> Result<Vec<TailraceCurveRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<TailraceCurveRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        // ── Extract columns by name ───────────────────────────────────────────
        let hydro_id_col = extract_int32_column(&batch, "hydro_id", path)?;
        let family_id_col = extract_int32_column(&batch, "family_id", path)?;
        let segment_id_col = extract_int32_column(&batch, "segment_id", path)?;
        let href_col = extract_float64_column(&batch, "href_jus_m", path)?;
        let q_inf_col = extract_float64_column(&batch, "q_jus_inf_m3s", path)?;
        let q_sup_col = extract_float64_column(&batch, "q_jus_sup_m3s", path)?;
        let a_cf0_col = extract_float64_column(&batch, "a_cf0", path)?;
        let a_cf1_col = extract_float64_column(&batch, "a_cf1", path)?;
        let a_cf2_col = extract_float64_column(&batch, "a_cf2", path)?;
        let a_cf3_col = extract_float64_column(&batch, "a_cf3", path)?;
        let a_cf4_col = extract_float64_column(&batch, "a_cf4", path)?;

        // ── Build rows with per-row validation ───────────────────────────────
        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let family_id = family_id_col.value(i);
            let segment_id = segment_id_col.value(i);

            // A null `href_jus_m` is a valid "single family / no reference level"
            // marker; a present value must clear the non-negative-finite gate.
            let href_jus_m = if href_col.is_null(i) {
                None
            } else {
                Some(validate_non_negative(
                    href_col.value(i),
                    row_idx,
                    "href_jus_m",
                    path,
                )?)
            };

            let q_jus_inf_m3s =
                validate_non_negative(q_inf_col.value(i), row_idx, "q_jus_inf_m3s", path)?;
            let q_jus_sup_m3s =
                validate_non_negative(q_sup_col.value(i), row_idx, "q_jus_sup_m3s", path)?;

            if q_jus_sup_m3s < q_jus_inf_m3s {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("tailrace_curves[{row_idx}].q_jus_sup_m3s"),
                    message: format!(
                        "upper bound must be >= lower bound, got q_jus_sup_m3s={q_jus_sup_m3s} < q_jus_inf_m3s={q_jus_inf_m3s}"
                    ),
                });
            }

            let a_cf0 = validate_finite(a_cf0_col.value(i), row_idx, "a_cf0", path)?;
            let a_cf1 = validate_finite(a_cf1_col.value(i), row_idx, "a_cf1", path)?;
            let a_cf2 = validate_finite(a_cf2_col.value(i), row_idx, "a_cf2", path)?;
            let a_cf3 = validate_finite(a_cf3_col.value(i), row_idx, "a_cf3", path)?;
            let a_cf4 = validate_finite(a_cf4_col.value(i), row_idx, "a_cf4", path)?;

            rows.push(TailraceCurveRow {
                hydro_id,
                family_id,
                href_jus_m,
                segment_id,
                q_jus_inf_m3s,
                q_jus_sup_m3s,
                a_cf0,
                a_cf1,
                a_cf2,
                a_cf3,
                a_cf4,
            });
        }
    }

    // ── Sort by (hydro_id, family_id, segment_id) ascending ──────────────────
    //
    // Integer keys only — no `total_cmp` on `href_jus_m` or a coefficient. A
    // float key would make tie-ordering depend on float values and break the
    // declaration-order-invariance contract downstream consumers rely on.
    rows.sort_by(|a, b| {
        a.hydro_id
            .0
            .cmp(&b.hydro_id.0)
            .then_with(|| a.family_id.cmp(&b.family_id))
            .then_with(|| a.segment_id.cmp(&b.segment_id))
    });

    Ok(rows)
}

/// Extract a column as [`Int32Array`] by name, returning [`LoadError::SchemaError`]
/// if the column is absent or has the wrong Arrow type.
fn extract_int32_column<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
    path: &Path,
) -> Result<&'a Int32Array, LoadError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!("missing column \"{name}\""),
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

/// Extract a column as [`Float64Array`] by name, returning [`LoadError::SchemaError`]
/// if the column is absent or has the wrong Arrow type.
fn extract_float64_column<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
    path: &Path,
) -> Result<&'a Float64Array, LoadError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!("missing column \"{name}\""),
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

/// Validate that a `f64` value is non-negative and finite, returning a
/// [`LoadError::SchemaError`] with the field path `"tailrace_curves[N].column_name"`
/// if the check fails.
fn validate_non_negative(
    value: f64,
    row_idx: usize,
    column: &str,
    path: &Path,
) -> Result<f64, LoadError> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("tailrace_curves[{row_idx}].{column}"),
            message: format!("value must be non-negative and finite, got {value}"),
        })
    }
}

/// Validate that a `f64` value is finite, returning a [`LoadError::SchemaError`]
/// with the field path `"tailrace_curves[N].column_name"` if the check fails.
///
/// Sign is unconstrained: the higher-degree tailrace coefficients are routinely
/// negative, so this gate rejects only NaN and the infinities.
fn validate_finite(
    value: f64,
    row_idx: usize,
    column: &str,
    path: &Path,
) -> Result<f64, LoadError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("tailrace_curves[{row_idx}].{column}"),
            message: format!("coefficient must be finite, got {value}"),
        })
    }
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// One column's worth of test data, packed positionally so `make_batch` can
    /// assemble the twelve-column schema from per-row tuples.
    struct RowSpec {
        hydro_id: i32,
        family_id: i32,
        href_jus_m: Option<f64>,
        segment_id: i32,
        q_jus_inf_m3s: f64,
        q_jus_sup_m3s: f64,
        a_cf: [f64; 5],
    }

    fn make_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("family_id", DataType::Int32, false),
            // The reference level is the only nullable column; a non-nullable
            // Float64 array cannot carry the NULL that "single family" needs.
            Field::new("href_jus_m", DataType::Float64, true),
            Field::new("segment_id", DataType::Int32, false),
            Field::new("q_jus_inf_m3s", DataType::Float64, false),
            Field::new("q_jus_sup_m3s", DataType::Float64, false),
            Field::new("a_cf0", DataType::Float64, false),
            Field::new("a_cf1", DataType::Float64, false),
            Field::new("a_cf2", DataType::Float64, false),
            Field::new("a_cf3", DataType::Float64, false),
            Field::new("a_cf4", DataType::Float64, false),
        ]))
    }

    fn make_batch(rows: &[RowSpec]) -> RecordBatch {
        let hydro_ids: Vec<i32> = rows.iter().map(|r| r.hydro_id).collect();
        let family_ids: Vec<i32> = rows.iter().map(|r| r.family_id).collect();
        let hrefs: Vec<Option<f64>> = rows.iter().map(|r| r.href_jus_m).collect();
        let segment_ids: Vec<i32> = rows.iter().map(|r| r.segment_id).collect();
        let q_infs: Vec<f64> = rows.iter().map(|r| r.q_jus_inf_m3s).collect();
        let q_sups: Vec<f64> = rows.iter().map(|r| r.q_jus_sup_m3s).collect();
        let a0: Vec<f64> = rows.iter().map(|r| r.a_cf[0]).collect();
        let a1: Vec<f64> = rows.iter().map(|r| r.a_cf[1]).collect();
        let a2: Vec<f64> = rows.iter().map(|r| r.a_cf[2]).collect();
        let a3: Vec<f64> = rows.iter().map(|r| r.a_cf[3]).collect();
        let a4: Vec<f64> = rows.iter().map(|r| r.a_cf[4]).collect();

        RecordBatch::try_new(
            make_schema(),
            vec![
                Arc::new(Int32Array::from(hydro_ids)),
                Arc::new(Int32Array::from(family_ids)),
                Arc::new(Float64Array::from(hrefs)),
                Arc::new(Int32Array::from(segment_ids)),
                Arc::new(Float64Array::from(q_infs)),
                Arc::new(Float64Array::from(q_sups)),
                Arc::new(Float64Array::from(a0)),
                Arc::new(Float64Array::from(a1)),
                Arc::new(Float64Array::from(a2)),
                Arc::new(Float64Array::from(a3)),
                Arc::new(Float64Array::from(a4)),
            ],
        )
        .expect("valid batch construction")
    }

    /// Compact constructor for a valid segment with a fixed coefficient set.
    fn seg(hydro_id: i32, family_id: i32, href: Option<f64>, segment_id: i32) -> RowSpec {
        RowSpec {
            hydro_id,
            family_id,
            href_jus_m: href,
            segment_id,
            q_jus_inf_m3s: 0.0,
            q_jus_sup_m3s: 1500.0,
            a_cf: [320.0, 1.0e-3, -3.1521e-17, 0.0, 0.0],
        }
    }

    fn write_parquet(batch: &RecordBatch) -> NamedTempFile {
        let tmp = NamedTempFile::new().expect("tempfile");
        let mut writer = ArrowWriter::try_new(tmp.reopen().expect("reopen"), batch.schema(), None)
            .expect("ArrowWriter");
        writer.write(batch).expect("write batch");
        writer.close().expect("close writer");
        tmp
    }

    fn write_parquet_batches(batches: &[RecordBatch]) -> NamedTempFile {
        assert!(!batches.is_empty(), "must provide at least one batch");
        let tmp = NamedTempFile::new().expect("tempfile");
        let mut writer =
            ArrowWriter::try_new(tmp.reopen().expect("reopen"), batches[0].schema(), None)
                .expect("ArrowWriter");
        for batch in batches {
            writer.write(batch).expect("write batch");
        }
        writer.close().expect("close writer");
        tmp
    }

    // ── AC: multi-segment family sorted ascending by segment_id ────────────────

    /// Two segments for hydro 1 / family 1 written in segment order `[2, 1]`
    /// come back sorted ascending: `rows[0].segment_id == 1`, `rows[1] == 2`.
    #[test]
    fn test_segments_sorted_ascending() {
        let batch = make_batch(&[seg(1, 1, Some(885.3), 2), seg(1, 1, Some(885.3), 1)]);
        let tmp = write_parquet(&batch);
        let rows = parse_tailrace_curves(tmp.path()).unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].segment_id, 1);
        assert_eq!(rows[1].segment_id, 2);
        assert_eq!(rows[0].hydro_id, EntityId::from(1));
        assert_eq!(rows[0].href_jus_m, Some(885.3));
    }

    // ── AC: negative quartic coefficient accepted ──────────────────────────────

    /// A negative `a_cf2` is preserved to bit precision (no sign constraint).
    #[test]
    fn test_negative_coefficient_accepted() {
        let mut spec = seg(1, 1, Some(885.3), 1);
        spec.a_cf[2] = -3.1521e-17;
        let batch = make_batch(&[spec]);
        let tmp = write_parquet(&batch);
        let rows = parse_tailrace_curves(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].a_cf2.to_bits(), (-3.1521e-17_f64).to_bits());
    }

    // ── AC: NULL href_jus_m maps to None ───────────────────────────────────────

    /// A NULL `href_jus_m` parses to `None` and the row is otherwise accepted.
    #[test]
    fn test_null_href_maps_to_none() {
        let batch = make_batch(&[seg(1, 1, None, 1)]);
        let tmp = write_parquet(&batch);
        let rows = parse_tailrace_curves(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].href_jus_m, None);
    }

    // ── AC: inverted window rejected ───────────────────────────────────────────

    /// `q_jus_sup_m3s < q_jus_inf_m3s` -> SchemaError naming the column and row.
    #[test]
    fn test_inverted_window_rejected() {
        let mut spec = seg(1, 1, Some(885.3), 1);
        spec.q_jus_inf_m3s = 1000.0;
        spec.q_jus_sup_m3s = 500.0;
        let batch = make_batch(&[spec]);
        let tmp = write_parquet(&batch);
        let err = parse_tailrace_curves(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("q_jus_sup_m3s"),
                    "field should name q_jus_sup_m3s, got: {field}"
                );
                assert!(
                    field.contains("[0]"),
                    "field should name row 0, got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: declaration-order invariance ───────────────────────────────────────

    /// Two clouds of identical rows supplied in different record-batch orders
    /// parse to element-for-element equal `Vec`s (integer-key sort is total).
    #[test]
    fn test_declaration_order_invariance() {
        let order_one = make_batch(&[
            seg(7, 2, Some(900.0), 3),
            seg(7, 1, Some(885.0), 1),
            seg(3, 1, None, 2),
        ]);
        let order_two = make_batch(&[
            seg(3, 1, None, 2),
            seg(7, 2, Some(900.0), 3),
            seg(7, 1, Some(885.0), 1),
        ]);

        let rows_one = parse_tailrace_curves(write_parquet(&order_one).path()).unwrap();
        let rows_two = parse_tailrace_curves(write_parquet(&order_two).path()).unwrap();

        assert_eq!(rows_one, rows_two);
        // Coefficient fields match to bit precision under reordering.
        for (x, y) in rows_one.iter().zip(rows_two.iter()) {
            assert_eq!(x.a_cf0.to_bits(), y.a_cf0.to_bits());
            assert_eq!(x.a_cf2.to_bits(), y.a_cf2.to_bits());
        }
    }

    // ── Missing column ─────────────────────────────────────────────────────────

    /// A Parquet file missing `a_cf4` -> SchemaError naming the column.
    #[test]
    fn test_missing_coefficient_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("family_id", DataType::Int32, false),
            Field::new("href_jus_m", DataType::Float64, true),
            Field::new("segment_id", DataType::Int32, false),
            Field::new("q_jus_inf_m3s", DataType::Float64, false),
            Field::new("q_jus_sup_m3s", DataType::Float64, false),
            Field::new("a_cf0", DataType::Float64, false),
            Field::new("a_cf1", DataType::Float64, false),
            Field::new("a_cf2", DataType::Float64, false),
            Field::new("a_cf3", DataType::Float64, false),
            // a_cf4 deliberately omitted
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Float64Array::from(vec![Some(885.3)])),
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(Float64Array::from(vec![1500.0])),
                Arc::new(Float64Array::from(vec![320.0])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(Float64Array::from(vec![0.0])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_tailrace_curves(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "a_cf4");
                assert!(message.contains("missing column"));
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Wrong column type ──────────────────────────────────────────────────────

    /// `segment_id` provided as Float64 instead of Int32 -> SchemaError.
    #[test]
    fn test_wrong_type_segment_id() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("family_id", DataType::Int32, false),
            Field::new("href_jus_m", DataType::Float64, true),
            Field::new("segment_id", DataType::Float64, false), // wrong type
            Field::new("q_jus_inf_m3s", DataType::Float64, false),
            Field::new("q_jus_sup_m3s", DataType::Float64, false),
            Field::new("a_cf0", DataType::Float64, false),
            Field::new("a_cf1", DataType::Float64, false),
            Field::new("a_cf2", DataType::Float64, false),
            Field::new("a_cf3", DataType::Float64, false),
            Field::new("a_cf4", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Float64Array::from(vec![Some(885.3)])),
                Arc::new(Float64Array::from(vec![1.0_f64])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(Float64Array::from(vec![1500.0])),
                Arc::new(Float64Array::from(vec![320.0])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(Float64Array::from(vec![0.0])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_tailrace_curves(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => assert_eq!(field, "segment_id"),
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Non-finite coefficient rejected ────────────────────────────────────────

    /// A NaN coefficient -> SchemaError naming the coefficient column.
    #[test]
    fn test_nan_coefficient_rejected() {
        let mut spec = seg(1, 1, Some(885.3), 1);
        spec.a_cf[3] = f64::NAN;
        let batch = make_batch(&[spec]);
        let tmp = write_parquet(&batch);
        let err = parse_tailrace_curves(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => assert!(field.contains("a_cf3")),
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// An infinite `q_jus_inf_m3s` -> SchemaError naming the bound column.
    #[test]
    fn test_infinite_bound_rejected() {
        let mut spec = seg(1, 1, Some(885.3), 1);
        spec.q_jus_inf_m3s = f64::INFINITY;
        spec.q_jus_sup_m3s = f64::INFINITY;
        let batch = make_batch(&[spec]);
        let tmp = write_parquet(&batch);
        let err = parse_tailrace_curves(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => assert!(field.contains("q_jus_inf_m3s")),
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A negative `q_jus_inf_m3s` -> SchemaError naming the bound column.
    #[test]
    fn test_negative_bound_rejected() {
        let mut spec = seg(1, 1, Some(885.3), 1);
        spec.q_jus_inf_m3s = -10.0;
        let batch = make_batch(&[spec]);
        let tmp = write_parquet(&batch);
        let err = parse_tailrace_curves(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => assert!(field.contains("q_jus_inf_m3s")),
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A negative non-null `href_jus_m` -> SchemaError naming the column.
    #[test]
    fn test_negative_href_rejected() {
        let batch = make_batch(&[seg(1, 1, Some(-5.0), 1)]);
        let tmp = write_parquet(&batch);
        let err = parse_tailrace_curves(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => assert!(field.contains("href_jus_m")),
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Empty / not-found ──────────────────────────────────────────────────────

    /// Empty Parquet (zero rows) -> Ok(Vec::new()).
    #[test]
    fn test_empty_parquet_returns_empty_vec() {
        let batch = make_batch(&[]);
        let tmp = write_parquet(&batch);
        let rows = parse_tailrace_curves(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    /// Non-existent path -> IoError with the matching path.
    #[test]
    fn test_file_not_found() {
        let path = Path::new("/nonexistent/path/tailrace_curves.parquet");
        let err = parse_tailrace_curves(path).unwrap_err();

        match err {
            LoadError::IoError { path: err_path, .. } => assert_eq!(err_path, path),
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    // ── Multi-batch read ───────────────────────────────────────────────────────

    /// A file with multiple record batches is fully read and globally sorted.
    #[test]
    fn test_multiple_record_batches() {
        let batch1 = make_batch(&[seg(5, 1, None, 2), seg(5, 1, None, 1)]);
        let batch2 = make_batch(&[seg(5, 1, None, 3)]);
        let tmp = write_parquet_batches(&[batch1, batch2]);
        let rows = parse_tailrace_curves(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].segment_id, 1);
        assert_eq!(rows[1].segment_id, 2);
        assert_eq!(rows[2].segment_id, 3);
    }

    /// Multiple families of one plant sort by `(family_id, segment_id)`.
    #[test]
    fn test_multiple_families_sorted() {
        let batch = make_batch(&[
            seg(1, 2, Some(900.0), 1),
            seg(1, 1, Some(885.0), 2),
            seg(1, 1, Some(885.0), 1),
        ]);
        let tmp = write_parquet(&batch);
        let rows = parse_tailrace_curves(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!((rows[0].family_id, rows[0].segment_id), (1, 1));
        assert_eq!((rows[1].family_id, rows[1].segment_id), (1, 2));
        assert_eq!((rows[2].family_id, rows[2].segment_id), (2, 1));
    }
}
