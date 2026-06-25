//! JSON Schema export helper for the `cobre.schema` sub-module.
//!
//! [`export`] wraps the same [`cobre_io::schema::generate_schemas`] generator as
//! the `cobre schema export` CLI, but returns the count of files written rather
//! than printing a confirmation line.

use std::path::PathBuf;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

/// Generate JSON Schema files for all case-directory input types and write them
/// to `output_dir` (a `str` or `pathlib.Path`), creating it if needed.
///
/// Existing schema files are overwritten without prompting — schemas are
/// generated, not hand-edited. Returns the number of files written.
///
/// # Raises
///
/// * `ValueError` — schema generation failed.
/// * `OSError` — the output directory could not be created or a file write
///   failed.
///
/// # Examples
///
/// ```python
/// import cobre.schema
/// count = cobre.schema.export("schemas")
/// print(f"wrote {count} schema files")
/// ```
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
#[pyo3(signature = (output_dir=PathBuf::from(".")))]
pub fn export(output_dir: PathBuf) -> PyResult<usize> {
    std::fs::create_dir_all(&output_dir).map_err(|e| {
        PyOSError::new_err(format!(
            "creating output directory '{}': {e}",
            output_dir.display()
        ))
    })?;

    let schemas = cobre_io::schema::generate_schemas()
        .map_err(|e| PyValueError::new_err(format!("schema generation failed: {e}")))?;

    let count = schemas.len();

    for (filename, value) in schemas {
        let dest = output_dir.join(&filename);
        let content = serde_json::to_string_pretty(&value).map_err(|e| {
            PyValueError::new_err(format!("failed to serialize schema '{filename}': {e}"))
        })?;
        std::fs::write(&dest, content)
            .map_err(|e| PyOSError::new_err(format!("{}: {e}", dest.display())))?;
    }

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_writes_all_schemas_as_valid_json() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().to_path_buf();

        let count = export(path.clone()).expect("export must succeed");

        let entries: Vec<_> = std::fs::read_dir(&path)
            .expect("read temp dir")
            .map(|e| e.expect("dir entry").path())
            .collect();

        assert_eq!(count, entries.len());
        assert!(count > 0, "at least one schema must be written");

        for entry in &entries {
            let content = std::fs::read_to_string(entry).expect("read schema file");
            let parsed: serde_json::Value =
                serde_json::from_str(&content).expect("schema file must be valid JSON");
            assert!(parsed.is_object(), "each schema must be a JSON object");
        }
    }

    #[test]
    fn test_export_creates_missing_directory() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let nested = dir.path().join("nested").join("schemas");

        let count = export(nested.clone()).expect("export must create nested dirs");

        assert!(nested.is_dir(), "nested output directory must be created");
        assert!(count > 0);
    }
}
