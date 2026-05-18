//! Parser for `system/hydro_energy_productivity.parquet` — per-plant per-stage
//! override values used by the energy-conversion preprocessing layer.
//!
//! ## Applicability
//!
//! The `equivalent_productivity_mw_per_m3s` column applies to **all** hydro
//! generation models. For FPHA hydros it overrides the `ρ_eq` value otherwise
//! derived from VHA geometry and `ρ_esp`. For non-FPHA hydros
//! (`constant_productivity`, `linearized_head`) it supplies `ρ_eq` directly
//! when `productivity_mw_per_m3s` is omitted from
//! `system/hydro_production_models.json`. Load-time validation enforces that
//! exactly one source supplies the value for each non-FPHA `(hydro, stage)`
//! pair — see [`crate::validation::productivity_resolution`].
//!
//! The other three override columns (`reference_volume_hm3`,
//! `reference_outflow_m3s`, `specific_productivity_mw_per_m3s_per_m`) keep
//! their existing per-row semantics independent of the generation model.
//!
//! ## Parquet schema
//!
//! | Column                                    | Parquet type | Nullable | Description                                     |
//! |-------------------------------------------|--------------|----------|-------------------------------------------------|
//! | `hydro_id`                                | INT32        | no       | Hydro plant identifier                          |
//! | `stage_id`                                | INT32        | yes      | Stage; NULL means "applies to all stages"       |
//! | `equivalent_productivity_mw_per_m3s`      | DOUBLE       | yes      | Direct `ρ_eq` override; finite and `>= 0.0`     |
//! | `reference_volume_hm3`                    | DOUBLE       | yes      | `V_ref` override; finite and `> 0.0`            |
//! | `reference_outflow_m3s`                   | DOUBLE       | yes      | `Q_ref` override; finite and `>= 0.0`           |
//! | `specific_productivity_mw_per_m3s_per_m`  | DOUBLE       | yes      | `ρ_esp` override; finite and `>= 0.0`           |
//!
//! ## Validation
//!
//! Per-row constraints enforced by [`parse_hydro_energy_productivity`]:
//!
//! - `hydro_id` must not be null.
//! - `equivalent_productivity_mw_per_m3s`, when set, must be finite and `>= 0.0`. A
//!   value of `0.0` is accepted (planned outage / zero-generation stage); the LP
//!   treats `ρ_eq` as a multiplier so zero produces zero generation without any
//!   division-by-zero hazard.
//! - `reference_volume_hm3`, when set, must be finite and `> 0.0`.
//! - `reference_outflow_m3s`, when set, must be finite and `>= 0.0`.
//! - `specific_productivity_mw_per_m3s_per_m`, when set, must be finite and `>= 0.0`. A
//!   value of `0.0` mirrors the planned-outage marker on
//!   `equivalent_productivity_mw_per_m3s`.
//! - A row where all four override columns are NULL is accepted (carries no
//!   information but is not an error).
//!
//! Duplicate `(hydro_id, stage_id)` detection is performed at build time by
//! the consumer that assembles the loaded rows into the override table.

use std::fs::File;
use std::path::Path;

use arrow::array::{Array, Float64Array, Int32Array};
use cobre_core::EntityId;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::LoadError;

/// A single row of the `system/hydro_energy_productivity.parquet` override table.
///
/// All four override columns are nullable. A row where all four are `None` is
/// accepted — it counts as a duplicate-detection key but carries no override
/// information.
#[derive(Debug, Clone, PartialEq)]
pub struct HydroEnergyProductivityRow {
    /// Hydro plant this override applies to.
    pub hydro_id: EntityId,
    /// Stage the override applies to. `None` means "applies to all stages for
    /// this hydro" (a per-hydro default).
    pub stage_id: Option<i32>,
    /// Direct `ρ_eq` override \[MW/(m³/s)\]. Finite and `>= 0.0` when set.
    /// `0.0` is accepted as a planned-outage marker.
    pub equivalent_productivity_mw_per_m3s: Option<f64>,
    /// `V_ref` override \[hm³\]. Finite and `> 0.0` when set.
    pub reference_volume_hm3: Option<f64>,
    /// `Q_ref` override \[m³/s\]. Finite and `>= 0.0` when set.
    pub reference_outflow_m3s: Option<f64>,
    /// `ρ_esp` override \[MW/(m³/s)/m\]. Finite and `>= 0.0` when set. `0.0`
    /// mirrors the planned-outage marker on `equivalent_productivity_mw_per_m3s`.
    pub specific_productivity_mw_per_m3s_per_m: Option<f64>,
}

/// Parse `system/hydro_energy_productivity.parquet`.
///
/// Rows are returned sorted by `(hydro_id.0, stage_id.unwrap_or(-1))` so that
/// NULL `stage_id` (per-hydro default) sorts before any concrete stage within
/// the same hydro.
///
/// # Errors
///
/// Returns [`LoadError::IoError`] when the file cannot be opened,
/// [`LoadError::ParseError`] for malformed Parquet, and
/// [`LoadError::SchemaError`] for missing/wrong-typed columns, null `hydro_id`,
/// or out-of-range override values.
pub fn parse_hydro_energy_productivity(
    path: &Path,
) -> Result<Vec<HydroEnergyProductivityRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;
    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<HydroEnergyProductivityRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let hydro_id_col = extract_int32_column(&batch, "hydro_id", path)?;
        let stage_id_col = extract_int32_column(&batch, "stage_id", path)?;
        let rho_eq_col =
            extract_float64_column(&batch, "equivalent_productivity_mw_per_m3s", path)?;
        let v_ref_col = extract_float64_column(&batch, "reference_volume_hm3", path)?;
        let q_ref_col = extract_float64_column(&batch, "reference_outflow_m3s", path)?;
        let rho_esp_col =
            extract_float64_column(&batch, "specific_productivity_mw_per_m3s_per_m", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            if hydro_id_col.is_null(i) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("hydro_energy_productivity[{row_idx}].hydro_id"),
                    message: "value must not be null".to_string(),
                });
            }

            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let stage_id = if stage_id_col.is_null(i) {
                None
            } else {
                Some(stage_id_col.value(i))
            };

            let equivalent_productivity_mw_per_m3s = if rho_eq_col.is_null(i) {
                None
            } else {
                Some(validate_nonnegative(
                    rho_eq_col.value(i),
                    row_idx,
                    "equivalent_productivity_mw_per_m3s",
                    path,
                )?)
            };

            let reference_volume_hm3 = if v_ref_col.is_null(i) {
                None
            } else {
                Some(validate_strictly_positive(
                    v_ref_col.value(i),
                    row_idx,
                    "reference_volume_hm3",
                    path,
                )?)
            };

            let reference_outflow_m3s = if q_ref_col.is_null(i) {
                None
            } else {
                Some(validate_nonnegative(
                    q_ref_col.value(i),
                    row_idx,
                    "reference_outflow_m3s",
                    path,
                )?)
            };

            let specific_productivity_mw_per_m3s_per_m = if rho_esp_col.is_null(i) {
                None
            } else {
                Some(validate_nonnegative(
                    rho_esp_col.value(i),
                    row_idx,
                    "specific_productivity_mw_per_m3s_per_m",
                    path,
                )?)
            };

            rows.push(HydroEnergyProductivityRow {
                hydro_id,
                stage_id,
                equivalent_productivity_mw_per_m3s,
                reference_volume_hm3,
                reference_outflow_m3s,
                specific_productivity_mw_per_m3s_per_m,
            });
        }
    }

    rows.sort_by_key(|r| (r.hydro_id.0, r.stage_id.unwrap_or(-1)));
    Ok(rows)
}

// ── column extraction helpers ──────────────────────────────────────────────────

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

// ── per-value validation helpers ───────────────────────────────────────────────

/// Validates that `value` is finite and strictly positive (`> 0.0`).
fn validate_strictly_positive(
    value: f64,
    row_idx: usize,
    column: &str,
    path: &Path,
) -> Result<f64, LoadError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydro_energy_productivity[{row_idx}].{column}"),
            message: format!("value must be finite and strictly positive (> 0.0), got {value}"),
        })
    }
}

/// Validates that `value` is finite and non-negative (`>= 0.0`).
fn validate_nonnegative(
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
            field: format!("hydro_energy_productivity[{row_idx}].{column}"),
            message: format!("value must be finite and non-negative (>= 0.0), got {value}"),
        })
    }
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::unwrap_used
)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use tempfile::NamedTempFile;

    use super::*;

    fn make_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, true),
            Field::new(
                "equivalent_productivity_mw_per_m3s",
                DataType::Float64,
                true,
            ),
            Field::new("reference_volume_hm3", DataType::Float64, true),
            Field::new("reference_outflow_m3s", DataType::Float64, true),
            Field::new(
                "specific_productivity_mw_per_m3s_per_m",
                DataType::Float64,
                true,
            ),
        ]))
    }

    fn make_batch(
        hydro_ids: &[i32],
        stage_ids: &[Option<i32>],
        rho_eqs: &[Option<f64>],
        v_refs: &[Option<f64>],
        q_refs: &[Option<f64>],
        rho_esps: &[Option<f64>],
    ) -> RecordBatch {
        let schema = make_schema();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(hydro_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(rho_eqs.to_vec())),
                Arc::new(Float64Array::from(v_refs.to_vec())),
                Arc::new(Float64Array::from(q_refs.to_vec())),
                Arc::new(Float64Array::from(rho_esps.to_vec())),
            ],
        )
        .expect("valid batch construction")
    }

    fn write_parquet(batch: &RecordBatch) -> NamedTempFile {
        let tmp = NamedTempFile::new().expect("tempfile");
        let mut writer = ArrowWriter::try_new(tmp.reopen().expect("reopen"), batch.schema(), None)
            .expect("ArrowWriter");
        writer.write(batch).expect("write batch");
        writer.close().expect("close writer");
        tmp
    }

    /// Round-trip: three rows matching the acceptance criterion fixture.
    ///
    /// Fixture:
    /// - row 0: `(hydro=1, stage=0, rho_eq=3.6, V_ref=NULL, Q_ref=NULL, rho_esp=NULL)`
    /// - row 1: `(hydro=1, stage=NULL, rho_eq=4.0, V_ref=120.0, Q_ref=NULL, rho_esp=0.009)`
    /// - row 2: `(hydro=2, stage=NULL, rho_eq=5.0, V_ref=NULL, Q_ref=200.0, rho_esp=NULL)`
    ///
    /// After sort the expected order is:
    /// `(hydro=1, NULL)` → `(hydro=1, stage=0)` → `(hydro=2, NULL)`.
    #[test]
    fn test_round_trip_three_rows() {
        // Write in non-sorted order to verify the parser sorts the output.
        let batch = make_batch(
            &[1, 1, 2],
            &[Some(0), None, None],
            &[Some(3.6), Some(4.0), Some(5.0)],
            &[None, Some(120.0), None],
            &[None, None, Some(200.0)],
            &[None, Some(0.009), None],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_energy_productivity(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3, "expected 3 rows");
        // Sorted: (hydro=1, NULL) → (hydro=1, stage=0) → (hydro=2, NULL)
        assert_eq!(rows[0].hydro_id, EntityId::from(1));
        assert_eq!(rows[0].stage_id, None);
        assert_eq!(rows[1].hydro_id, EntityId::from(1));
        assert_eq!(rows[1].stage_id, Some(0));
        assert_eq!(rows[2].hydro_id, EntityId::from(2));
        assert_eq!(rows[2].stage_id, None);
    }

    /// `equivalent_productivity_mw_per_m3s = 0.0` is accepted as a planned-outage marker.
    #[test]
    fn test_zero_rho_eq_accepted() {
        let batch = make_batch(&[1], &[Some(0)], &[Some(0.0)], &[None], &[None], &[None]);
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_energy_productivity(tmp.path())
            .expect("zero ρ_eq must be accepted as a planned-outage marker");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].equivalent_productivity_mw_per_m3s, Some(0.0));
    }

    /// Negative `equivalent_productivity_mw_per_m3s` is still rejected.
    #[test]
    fn test_negative_rho_eq_rejected() {
        let batch = make_batch(&[1], &[Some(0)], &[Some(-0.1)], &[None], &[None], &[None]);
        let tmp = write_parquet(&batch);
        let err = parse_hydro_energy_productivity(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("equivalent_productivity_mw_per_m3s"),
                    "field should name the column, got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `reference_volume_hm3 = -1.0` must be rejected.
    #[test]
    fn test_negative_v_ref_rejected() {
        let batch = make_batch(&[1], &[None], &[None], &[Some(-1.0)], &[None], &[None]);
        let tmp = write_parquet(&batch);
        let err = parse_hydro_energy_productivity(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("reference_volume_hm3"),
                    "field should name the column, got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `reference_outflow_m3s = NaN` must be rejected.
    #[test]
    fn test_nan_q_ref_rejected() {
        let batch = make_batch(&[1], &[None], &[None], &[None], &[Some(f64::NAN)], &[None]);
        let tmp = write_parquet(&batch);
        let err = parse_hydro_energy_productivity(tmp.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError, got: {err:?}"
        );
    }

    /// A row where all four override columns are NULL is accepted.
    #[test]
    fn test_all_overrides_null_accepted() {
        let batch = make_batch(&[1], &[Some(0)], &[None], &[None], &[None], &[None]);
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_energy_productivity(tmp.path()).unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.hydro_id, EntityId::from(1));
        assert_eq!(row.stage_id, Some(0));
        assert!(row.equivalent_productivity_mw_per_m3s.is_none());
        assert!(row.reference_volume_hm3.is_none());
        assert!(row.reference_outflow_m3s.is_none());
        assert!(row.specific_productivity_mw_per_m3s_per_m.is_none());
    }

    /// A null `hydro_id` must be rejected.
    #[test]
    fn test_null_hydro_id_rejected() {
        // Build a batch with a nullable hydro_id column that has a null value.
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, true), // nullable for this test
            Field::new("stage_id", DataType::Int32, true),
            Field::new(
                "equivalent_productivity_mw_per_m3s",
                DataType::Float64,
                true,
            ),
            Field::new("reference_volume_hm3", DataType::Float64, true),
            Field::new("reference_outflow_m3s", DataType::Float64, true),
            Field::new(
                "specific_productivity_mw_per_m3s_per_m",
                DataType::Float64,
                true,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![None::<i32>])),
                Arc::new(Int32Array::from(vec![None::<i32>])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
            ],
        )
        .expect("valid batch");
        let tmp = NamedTempFile::new().expect("tempfile");
        let mut writer = ArrowWriter::try_new(tmp.reopen().expect("reopen"), batch.schema(), None)
            .expect("ArrowWriter");
        writer.write(&batch).expect("write");
        writer.close().expect("close");

        let err = parse_hydro_energy_productivity(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("hydro_id"),
                    "field should mention hydro_id, got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── duplicate-key detection uses a HashSet ─────────────────────────────────
    // Duplicate detection is the consumer's responsibility (the builder that
    // assembles the loaded rows into the override table). The parser itself
    // does NOT detect duplicates — it only validates per-row values.

    /// `reference_outflow_m3s = 0.0` must be accepted (zero outflow is valid).
    #[test]
    fn test_zero_q_ref_accepted() {
        let batch = make_batch(&[1], &[None], &[None], &[None], &[Some(0.0)], &[None]);
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_energy_productivity(tmp.path()).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].reference_outflow_m3s, Some(0.0));
    }

    /// Duplicate `(hydro_id, stage_id)` — the parser does NOT detect this.
    /// Verification that two identical keys produce two distinct rows from the parser
    /// (the builder test covers the error case).
    #[test]
    fn test_duplicate_keys_not_rejected_by_parser() {
        let batch = make_batch(
            &[1, 1],
            &[Some(0), Some(0)],
            &[Some(3.6), Some(4.0)],
            &[None, None],
            &[None, None],
            &[None, None],
        );
        let tmp = write_parquet(&batch);
        // The parser accepts both rows; builder would reject them.
        let rows = parse_hydro_energy_productivity(tmp.path()).unwrap();
        assert_eq!(rows.len(), 2);
    }

    /// A `HashSet`-based note: the parser sorts by `(hydro_id, stage_id.unwrap_or(-1))`.
    #[test]
    fn test_sort_order_null_stage_before_concrete() {
        let batch = make_batch(
            &[2, 1, 1],
            &[None, Some(5), None],
            &[Some(1.0), Some(2.0), Some(3.0)],
            &[None, None, None],
            &[None, None, None],
            &[None, None, None],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_energy_productivity(tmp.path()).unwrap();

        // Expected order: (hydro=1, NULL) → (hydro=1, stage=5) → (hydro=2, NULL)
        assert_eq!(rows[0].hydro_id, EntityId::from(1));
        assert_eq!(rows[0].stage_id, None);
        assert_eq!(rows[1].hydro_id, EntityId::from(1));
        assert_eq!(rows[1].stage_id, Some(5));
        assert_eq!(rows[2].hydro_id, EntityId::from(2));
        assert_eq!(rows[2].stage_id, None);
    }
}
