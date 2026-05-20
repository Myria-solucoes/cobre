//! Parsing and resolution for
//! `system/hydro_reference_volume_fractions.parquet` — per-plant per-season
//! reference-volume-fraction overrides used by energy-conversion preprocessing.
//!
//! ## Parquet schema
//!
//! | Column      | Parquet type | Nullable | Description                                          |
//! |-------------|--------------|----------|------------------------------------------------------|
//! | `hydro_id`  | INT32        | no       | Hydro plant identifier                               |
//! | `season_id` | INT32        | yes      | Season id; NULL means "applies to all seasons"       |
//! | `fraction`  | DOUBLE       | no       | Reference-volume fraction in `(0.0, 1.0]`            |
//!
//! ## Validation
//!
//! Per-row constraints enforced by [`parse_hydro_reference_volume_fractions`]:
//!
//! - All three columns must be present with the correct types.
//! - `fraction` must be finite and in `(0.0, 1.0]` (exclusive zero, inclusive one).
//! - Duplicate `(hydro_id, season_id)` keys are rejected.
//!
//! Cross-reference validation against the hydro registry and the stage→season
//! mapping is performed by [`build_hydro_reference_volume_fractions`].
//!
//! ## Resolution precedence
//!
//! For a given `(hydro_id, stage_id)`, [`HydroReferenceVolumeFractions::get`]
//! returns:
//!
//! 1. `(hydro, season(stage))` row, if present.
//! 2. `(hydro, NULL)` cross-season row for the same hydro, if present.
//! 3. Case-level `EnergyConfig.reference_volume_fraction` default.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;

use arrow::array::{Array, Float64Array, Int32Array};
use cobre_core::{EntityId, Hydro};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::LoadError;

/// A single row of the reference-volume-fraction override table.
#[derive(Debug, Clone, PartialEq)]
pub struct HydroReferenceVolumeFractionRow {
    /// Hydro plant this override applies to.
    pub hydro_id: EntityId,
    /// Season the override applies to. `None` means "applies to all seasons
    /// for this hydro" (cross-season default for the hydro).
    pub season_id: Option<i32>,
    /// Reference-volume fraction in `(0.0, 1.0]`.
    pub fraction: f64,
}

/// Resolver returning the reference-volume fraction for a given
/// `(hydro_id, stage_id)` pair, applying the documented precedence chain.
#[derive(Debug, Clone)]
pub struct HydroReferenceVolumeFractions {
    overrides: HashMap<(EntityId, Option<i32>), f64>,
    stage_to_season: Vec<i32>,
    default_fraction: f64,
}

impl HydroReferenceVolumeFractions {
    /// Returns the resolved fraction for `(hydro, stage)`, applying the
    /// `(hydro, season)` → `(hydro, NULL)` → default precedence chain.
    ///
    /// In debug builds, an out-of-range `stage_id` triggers a `debug_assert!`.
    /// In release builds, an out-of-range `stage_id` falls back to the default.
    #[must_use]
    pub fn get(&self, hydro_id: EntityId, stage_id: usize) -> f64 {
        debug_assert!(
            stage_id < self.stage_to_season.len(),
            "stage_id {} out of range (stage_to_season has {} entries)",
            stage_id,
            self.stage_to_season.len(),
        );
        if let Some(&season) = self.stage_to_season.get(stage_id)
            && let Some(&v) = self.overrides.get(&(hydro_id, Some(season)))
        {
            return v;
        }
        if let Some(&v) = self.overrides.get(&(hydro_id, None)) {
            return v;
        }
        self.default_fraction
    }

    /// Case-level default fraction.
    #[must_use]
    pub fn default_fraction(&self) -> f64 {
        self.default_fraction
    }
}

/// Parse `system/hydro_reference_volume_fractions.parquet`.
///
/// # Errors
///
/// Returns [`LoadError::IoError`] when the file cannot be opened,
/// [`LoadError::ParseError`] for malformed Parquet, and
/// [`LoadError::SchemaError`] for missing/wrong-typed columns, out-of-range
/// `fraction` values, or duplicate `(hydro_id, season_id)` keys.
pub fn parse_hydro_reference_volume_fractions(
    path: &Path,
) -> Result<Vec<HydroReferenceVolumeFractionRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;
    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<HydroReferenceVolumeFractionRow> = Vec::new();
    let mut seen: HashSet<(EntityId, Option<i32>)> = HashSet::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let hydro_id_col = extract_int32_column(&batch, "hydro_id", path)?;
        let season_id_col = extract_int32_column(&batch, "season_id", path)?;
        let fraction_col = extract_float64_column(&batch, "fraction", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            if hydro_id_col.is_null(i) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("hydro_reference_volume_fractions[{row_idx}].hydro_id"),
                    message: "value must not be null".to_string(),
                });
            }
            if fraction_col.is_null(i) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("hydro_reference_volume_fractions[{row_idx}].fraction"),
                    message: "value must not be null".to_string(),
                });
            }

            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let season_id = if season_id_col.is_null(i) {
                None
            } else {
                Some(season_id_col.value(i))
            };
            let fraction = validate_fraction(fraction_col.value(i), row_idx, "fraction", path)?;

            let key = (hydro_id, season_id);
            if !seen.insert(key) {
                let season_label = season_id.map_or_else(|| "NULL".to_string(), |s| s.to_string());
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("hydro_reference_volume_fractions[{row_idx}]"),
                    message: format!(
                        "duplicate (hydro_id={}, season_id={}) key",
                        hydro_id.0, season_label
                    ),
                });
            }

            rows.push(HydroReferenceVolumeFractionRow {
                hydro_id,
                season_id,
                fraction,
            });
        }
    }

    Ok(rows)
}

/// Build a [`HydroReferenceVolumeFractions`] resolver from parsed rows.
///
/// Cross-validates that every `hydro_id` exists in `hydros` and every non-NULL
/// `season_id` appears at least once in `stage_to_season`. `stage_to_season[t]`
/// is the season id for stage `t`.
///
/// # Errors
///
/// Returns [`LoadError::SchemaError`] when a row references an unknown hydro
/// or season, or when `default_fraction` is not in `(0.0, 1.0]`.
pub fn build_hydro_reference_volume_fractions(
    rows: Vec<HydroReferenceVolumeFractionRow>,
    default_fraction: f64,
    hydros: &[Hydro],
    stage_to_season: &[i32],
) -> Result<HydroReferenceVolumeFractions, LoadError> {
    if !default_fraction.is_finite() || default_fraction <= 0.0 || default_fraction > 1.0 {
        return Err(LoadError::SchemaError {
            path: Path::new("<config>").to_path_buf(),
            field: "energy.reference_volume_fraction".to_string(),
            message: format!("default_fraction must be in (0.0, 1.0], got {default_fraction}"),
        });
    }

    let known_hydros: HashSet<EntityId> = hydros.iter().map(|h| h.id).collect();
    let known_seasons: HashSet<i32> = stage_to_season.iter().copied().collect();

    let mut overrides: HashMap<(EntityId, Option<i32>), f64> = HashMap::with_capacity(rows.len());

    for row in rows {
        if !known_hydros.contains(&row.hydro_id) {
            return Err(LoadError::SchemaError {
                path: Path::new("<system>").to_path_buf(),
                field: "hydro_reference_volume_fractions.hydro_id".to_string(),
                message: format!("unknown hydro_id {}", row.hydro_id.0),
            });
        }
        if let Some(season) = row.season_id
            && !known_seasons.contains(&season)
        {
            return Err(LoadError::SchemaError {
                path: Path::new("<system>").to_path_buf(),
                field: "hydro_reference_volume_fractions.season_id".to_string(),
                message: format!("unknown season_id {season}"),
            });
        }
        overrides.insert((row.hydro_id, row.season_id), row.fraction);
    }

    Ok(HydroReferenceVolumeFractions {
        overrides,
        stage_to_season: stage_to_season.to_vec(),
        default_fraction,
    })
}

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

fn validate_fraction(
    value: f64,
    row_idx: usize,
    column: &str,
    path: &Path,
) -> Result<f64, LoadError> {
    if value.is_finite() && value > 0.0 && value <= 1.0 {
        Ok(value)
    } else {
        Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydro_reference_volume_fractions[{row_idx}].{column}"),
            message: format!("fraction must be finite and in (0.0, 1.0], got {value}"),
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
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use cobre_core::{EntityId, Hydro, HydroGenerationModel, HydroPenalties};
    use parquet::arrow::ArrowWriter;
    use tempfile::NamedTempFile;

    use super::*;

    fn make_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("season_id", DataType::Int32, true),
            Field::new("fraction", DataType::Float64, false),
        ]))
    }

    fn make_batch(hydro_ids: &[i32], season_ids: &[Option<i32>], fractions: &[f64]) -> RecordBatch {
        let schema = make_schema();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(hydro_ids.to_vec())),
                Arc::new(Int32Array::from(season_ids.to_vec())),
                Arc::new(Float64Array::from(fractions.to_vec())),
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

    fn penalties_zero() -> HydroPenalties {
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

    fn make_hydro(id: i32) -> Hydro {
        Hydro {
            id: EntityId::from(id),
            name: format!("Hydro {id}"),
            bus_id: EntityId::from(1),
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
            penalties: penalties_zero(),
        }
    }

    #[test]
    fn parser_accepts_well_formed_rows() {
        let batch = make_batch(&[42, 42, 7], &[None, Some(1), Some(2)], &[0.50, 0.70, 0.90]);
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_reference_volume_fractions(tmp.path()).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].hydro_id, EntityId::from(42));
        assert_eq!(rows[0].season_id, None);
        assert_eq!(rows[0].fraction, 0.50);
        assert_eq!(rows[2].season_id, Some(2));
    }

    #[test]
    fn parser_rejects_zero_fraction() {
        let batch = make_batch(&[42], &[None], &[0.0]);
        let tmp = write_parquet(&batch);
        let err = parse_hydro_reference_volume_fractions(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { field, .. } => assert!(field.contains("fraction")),
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn parser_rejects_above_one() {
        let batch = make_batch(&[42], &[None], &[1.5]);
        let tmp = write_parquet(&batch);
        let err = parse_hydro_reference_volume_fractions(tmp.path()).unwrap_err();
        assert!(matches!(err, LoadError::SchemaError { .. }));
    }

    #[test]
    fn parser_rejects_negative() {
        let batch = make_batch(&[42], &[None], &[-0.1]);
        let tmp = write_parquet(&batch);
        let err = parse_hydro_reference_volume_fractions(tmp.path()).unwrap_err();
        assert!(matches!(err, LoadError::SchemaError { .. }));
    }

    #[test]
    fn parser_rejects_nan() {
        let batch = make_batch(&[42], &[None], &[f64::NAN]);
        let tmp = write_parquet(&batch);
        let err = parse_hydro_reference_volume_fractions(tmp.path()).unwrap_err();
        assert!(matches!(err, LoadError::SchemaError { .. }));
    }

    #[test]
    fn parser_rejects_duplicate_keys_null_season() {
        let batch = make_batch(&[42, 42], &[None, None], &[0.50, 0.55]);
        let tmp = write_parquet(&batch);
        let err = parse_hydro_reference_volume_fractions(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(message.contains("duplicate"));
                assert!(message.contains("hydro_id=42"));
                assert!(message.contains("NULL"));
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn parser_rejects_duplicate_keys_same_season() {
        let batch = make_batch(&[42, 42], &[Some(1), Some(1)], &[0.50, 0.55]);
        let tmp = write_parquet(&batch);
        let err = parse_hydro_reference_volume_fractions(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(message.contains("duplicate"));
                assert!(message.contains("season_id=1"));
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn resolver_falls_back_to_default_when_no_overrides() {
        let resolver = build_hydro_reference_volume_fractions(
            Vec::new(),
            0.65,
            &[make_hydro(42)],
            &[0, 1, 0, 1],
        )
        .unwrap();
        assert_eq!(resolver.get(EntityId(42), 0), 0.65);
        assert_eq!(resolver.get(EntityId(99), 0), 0.65);
    }

    #[test]
    fn resolver_uses_cross_season_override() {
        let rows = vec![HydroReferenceVolumeFractionRow {
            hydro_id: EntityId(42),
            season_id: None,
            fraction: 0.50,
        }];
        let resolver = build_hydro_reference_volume_fractions(
            rows,
            0.65,
            &[make_hydro(42), make_hydro(7)],
            &[0, 1, 0, 1],
        )
        .unwrap();
        // hydro 42 in any stage → 0.50 (cross-season override)
        assert_eq!(resolver.get(EntityId(42), 2), 0.50);
        // hydro 7 has no override → default 0.65
        assert_eq!(resolver.get(EntityId(7), 2), 0.65);
    }

    #[test]
    fn resolver_prefers_season_specific_over_cross_season() {
        let rows = vec![
            HydroReferenceVolumeFractionRow {
                hydro_id: EntityId(42),
                season_id: None,
                fraction: 0.50,
            },
            HydroReferenceVolumeFractionRow {
                hydro_id: EntityId(42),
                season_id: Some(1),
                fraction: 0.70,
            },
        ];
        let resolver =
            build_hydro_reference_volume_fractions(rows, 0.65, &[make_hydro(42)], &[0, 1, 0, 1])
                .unwrap();
        // stage 0 → season 0 → no specific override → falls back to (42, NULL) → 0.50
        assert_eq!(resolver.get(EntityId(42), 0), 0.50);
        // stage 1 → season 1 → specific (42, 1) → 0.70
        assert_eq!(resolver.get(EntityId(42), 1), 0.70);
        // stage 3 → season 1 → specific (42, 1) → 0.70
        assert_eq!(resolver.get(EntityId(42), 3), 0.70);
    }

    #[test]
    fn builder_rejects_unknown_hydro_id() {
        let rows = vec![HydroReferenceVolumeFractionRow {
            hydro_id: EntityId(999),
            season_id: None,
            fraction: 0.50,
        }];
        let err = build_hydro_reference_volume_fractions(rows, 0.65, &[make_hydro(42)], &[0, 1])
            .unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(message.contains("999"));
                assert!(message.contains("unknown"));
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn builder_rejects_unknown_season_id() {
        let rows = vec![HydroReferenceVolumeFractionRow {
            hydro_id: EntityId(42),
            season_id: Some(99),
            fraction: 0.50,
        }];
        let err = build_hydro_reference_volume_fractions(rows, 0.65, &[make_hydro(42)], &[0, 1])
            .unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(message.contains("99"));
                assert!(message.contains("unknown"));
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }
}
