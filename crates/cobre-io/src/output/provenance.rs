//! JSON writer for the model provenance report.
//!
//! The provenance report is a one-time diagnostic artifact produced after
//! stochastic preprocessing. It captures which data sources were used for
//! each role — seasonal statistics, AR coefficients, correlation, and the
//! opening scenario tree.

use std::path::Path;

use super::atomic::write_json_atomic;
use super::error::OutputError;

use serde::Serialize;
use serde::de::DeserializeOwned;

/// Write a model provenance report as pretty-printed JSON, atomically.
///
/// Generic over `Serialize` so the report struct stays in the calling algorithm
/// crate, keeping this crate algorithm-agnostic.
///
/// # Errors
///
/// Returns [`OutputError::IoError`] on filesystem failures, or
/// [`OutputError::SerializationError`] if JSON serialization fails.
pub fn write_provenance_report(path: &Path, report: &impl Serialize) -> Result<(), OutputError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| OutputError::io(parent, e))?;
    }

    write_json_atomic(path, report, "model_provenance")
}

/// Read a model provenance report from a JSON file.
///
/// Generic over `DeserializeOwned` so the report struct stays in the calling
/// algorithm crate, keeping this crate algorithm-agnostic.
///
/// # Errors
///
/// Returns [`OutputError::IoError`] if the file cannot be read — a missing file
/// surfaces as an `IoError` whose `source.kind()` is
/// [`std::io::ErrorKind::NotFound`], so callers can treat the section as absent
/// and degrade gracefully. Returns [`OutputError::ManifestError`] if the file
/// contains malformed JSON.
pub fn read_provenance_report<T: DeserializeOwned>(path: &Path) -> Result<T, OutputError> {
    let content = std::fs::read_to_string(path).map_err(|e| OutputError::io(path, e))?;
    serde_json::from_str(&content).map_err(|e| OutputError::ManifestError {
        manifest_type: "model_provenance".to_string(),
        message: e.to_string(),
    })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use tempfile::TempDir;

    #[derive(Serialize)]
    struct MockProvenanceReport {
        estimation_path: String,
        seasonal_stats_source: String,
        ar_coefficients_source: String,
        correlation_source: String,
        opening_tree_source: String,
        n_hydros: usize,
    }

    fn make_mock_report() -> MockProvenanceReport {
        MockProvenanceReport {
            estimation_path: "full_estimation".to_string(),
            seasonal_stats_source: "estimated".to_string(),
            ar_coefficients_source: "estimated".to_string(),
            correlation_source: "estimated".to_string(),
            opening_tree_source: "estimated".to_string(),
            n_hydros: 3,
        }
    }

    #[test]
    fn write_and_read_back_json() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("training/model_provenance.json");

        let report = make_mock_report();

        write_provenance_report(&path, &report).expect("write should succeed");

        let content = std::fs::read_to_string(&path).expect("read");
        assert!(content.contains("\"full_estimation\""));
        assert!(content.contains("\"n_hydros\": 3"));
    }

    #[test]
    fn round_trip_all_fields() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("model_provenance.json");

        let report = MockProvenanceReport {
            estimation_path: "deterministic".to_string(),
            seasonal_stats_source: "n/a".to_string(),
            ar_coefficients_source: "n/a".to_string(),
            correlation_source: "n/a".to_string(),
            opening_tree_source: "n/a".to_string(),
            n_hydros: 0,
        };

        write_provenance_report(&path, &report).expect("write should succeed");

        let content = std::fs::read_to_string(&path).expect("read");
        let value: serde_json::Value =
            serde_json::from_str(&content).expect("valid JSON after round-trip");

        assert_eq!(value["estimation_path"], "deterministic");
        assert_eq!(value["seasonal_stats_source"], "n/a");
        assert_eq!(value["ar_coefficients_source"], "n/a");
        assert_eq!(value["correlation_source"], "n/a");
        assert_eq!(value["opening_tree_source"], "n/a");
        assert_eq!(value["n_hydros"], 0);
    }

    #[test]
    fn tmp_file_is_cleaned_up() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("model_provenance.json");

        let report = make_mock_report();

        write_provenance_report(&path, &report).expect("write should succeed");

        let tmp_path = path.with_extension("json.tmp");
        assert!(
            !tmp_path.exists(),
            "tmp file should be removed after rename"
        );
        assert!(path.exists(), "final file should exist");
    }

    // ── Generic reader ───────────────────────────────────────────────────────

    /// Nested mock mirroring the cross-model report shape, defined locally so
    /// the reader test never depends on an algorithm crate (genericity rule).
    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct MockReport {
        inflow: MockSection,
        hydro_production: MockSection,
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct MockSection {
        source: String,
        count: usize,
    }

    fn make_nested_mock() -> MockReport {
        MockReport {
            inflow: MockSection {
                source: "estimated".to_string(),
                count: 5,
            },
            hydro_production: MockSection {
                source: "precomputed".to_string(),
                count: 12,
            },
        }
    }

    #[test]
    fn read_provenance_report_round_trips_mock() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("training/model_provenance.json");

        let report = make_nested_mock();
        write_provenance_report(&path, &report).expect("write should succeed");

        let decoded: MockReport = read_provenance_report(&path).expect("read should succeed");
        assert_eq!(decoded, report);
    }

    #[test]
    fn read_provenance_report_missing_file_is_not_found() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("does_not_exist.json");

        let result = read_provenance_report::<MockReport>(&path);

        assert!(
            matches!(
                &result,
                Err(OutputError::IoError { source, .. })
                    if source.kind() == std::io::ErrorKind::NotFound
            ),
            "missing file must return IoError with NotFound kind so callers \
             can degrade gracefully, got: {result:?}"
        );
    }

    #[test]
    fn read_provenance_report_malformed_json_errors() {
        use std::io::Write;
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("model_provenance.json");
        let mut file = std::fs::File::create(&path).expect("create file");
        writeln!(file, "{{not valid json at all").expect("write malformed json");

        let result = read_provenance_report::<MockReport>(&path);

        assert!(
            matches!(result, Err(OutputError::ManifestError { .. })),
            "malformed JSON must return the parse variant (not IoError), got: {result:?}"
        );
    }
}
