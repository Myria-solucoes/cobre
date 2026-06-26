//! Single owner of the write-side crash-safety contract: write to `{path}.tmp`,
//! flush **explicitly** (never via `Drop`), then `rename` onto `path`.
//!
//! The explicit flush before the rename is load-bearing: `BufWriter` flushes on
//! drop, but `Drop::drop` cannot return an error, so a drop-flush swallows an
//! `ENOSPC`/`EIO` on the buffered tail and the rename then installs a truncated
//! file with no error surfaced. Always flush via [`std::io::Write::flush`] and
//! propagate with `?` — never rely on drop-flush before a rename.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use super::error::OutputError;
use super::parquet_config::ParquetWriterConfig;

/// Temporary sibling path for an atomic write, preserving the original extension
/// as a prefix of `.tmp` (`foo.json` → `foo.json.tmp`) so the temp never
/// collides with a differently-typed sibling.
pub(crate) fn tmp_path(path: &Path) -> PathBuf {
    path.with_extension(path.extension().map_or_else(
        || "tmp".to_string(),
        |ext| format!("{}.tmp", ext.to_string_lossy()),
    ))
}

/// Write `bytes` to `path` atomically (write to `{path}.tmp`, flush, rename).
///
/// The parent directory must already exist. On any I/O error the target `path`
/// is left untouched; a partial `.tmp` may remain on disk.
///
/// # Errors
///
/// Returns [`OutputError::IoError`] if creating, writing, flushing, or renaming
/// the temporary file fails.
pub(crate) fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<(), OutputError> {
    let tmp = tmp_path(path);

    let file = std::fs::File::create(&tmp).map_err(|e| OutputError::io(&tmp, e))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(bytes)
        .map_err(|e| OutputError::io(&tmp, e))?;
    // Explicit flush before rename — see module doc (drop-flush swallows errors).
    writer.flush().map_err(|e| OutputError::io(&tmp, e))?;

    std::fs::rename(&tmp, path).map_err(|e| OutputError::io(path, e))?;
    Ok(())
}

/// Serialize `value` to pretty-printed JSON and write it to `path` atomically.
///
/// Byte content is identical to `serde_json::to_vec_pretty`. The parent
/// directory must already exist. `entity` labels any
/// [`OutputError::SerializationError`].
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] if JSON serialization fails, or
/// [`OutputError::IoError`] if creating, flushing, or renaming the temporary
/// file fails.
pub(crate) fn write_json_atomic(
    path: &Path,
    value: &impl serde::Serialize,
    entity: &str,
) -> Result<(), OutputError> {
    let tmp = tmp_path(path);

    let file = std::fs::File::create(&tmp).map_err(|e| OutputError::io(&tmp, e))?;
    serialize_json_then_flush(file, value, entity, &tmp)?;

    std::fs::rename(&tmp, path).map_err(|e| OutputError::io(path, e))?;
    Ok(())
}

/// Stream `value` as pretty JSON into a flushed `BufWriter` over `sink`.
///
/// Split from [`write_json_atomic`] so the serialize-then-explicit-flush step
/// can be driven with a failing writer in tests; the caller's rename runs only
/// on `Ok`, so a flush error can never install a target file.
fn serialize_json_then_flush<W: Write>(
    sink: W,
    value: &impl serde::Serialize,
    entity: &str,
    tmp: &Path,
) -> Result<(), OutputError> {
    let mut writer = BufWriter::new(sink);
    serde_json::to_writer_pretty(&mut writer, value)
        .map_err(|e| OutputError::serialization(entity, format!("JSON serialization: {e}")))?;
    // Explicit flush before the caller's rename — see module doc.
    writer.flush().map_err(|e| OutputError::io(tmp, e))?;
    Ok(())
}

/// Write a `RecordBatch` to `path` as a Parquet file, atomically.
///
/// Honors `config` (compression, row-group size, dictionary encoding). The
/// parent directory must already exist.
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] if the Parquet writer fails, or
/// [`OutputError::IoError`] if creating, flushing, or renaming the temporary
/// file fails.
pub(crate) fn write_parquet_atomic(
    path: &Path,
    batch: &RecordBatch,
    config: &ParquetWriterConfig,
) -> Result<(), OutputError> {
    let tmp = tmp_path(path);

    let props = WriterProperties::builder()
        .set_compression(config.compression)
        .set_max_row_group_row_count(Some(config.row_group_size))
        .set_dictionary_enabled(config.dictionary_encoding)
        .build();

    let file = std::fs::File::create(&tmp).map_err(|e| OutputError::io(&tmp, e))?;
    let buf = BufWriter::new(file);

    let mut writer = ArrowWriter::try_new(buf, batch.schema(), Some(props))
        .map_err(|e| OutputError::serialization("parquet_writer", e.to_string()))?;
    writer
        .write(batch)
        .map_err(|e| OutputError::serialization("parquet_writer", e.to_string()))?;

    // Redundant with `into_inner`'s flush today, but kept to surface an I/O
    // error before the rename rather than via drop-flush — do not remove without
    // re-verifying the parquet version in `Cargo.lock`.
    let mut buf = writer
        .into_inner()
        .map_err(|e| OutputError::serialization("parquet_writer", e.to_string()))?;
    buf.flush().map_err(|e| OutputError::io(&tmp, e))?;

    std::fs::rename(&tmp, path).map_err(|e| OutputError::io(path, e))?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::super::error::OutputError;
    use super::{serialize_json_then_flush, tmp_path, write_bytes_atomic, write_json_atomic};
    use serde::Serialize;
    use std::io;
    use std::path::{Path, PathBuf};
    use tempfile::TempDir;

    #[derive(Serialize)]
    struct Mock {
        a: i32,
        b: String,
    }

    #[test]
    fn tmp_path_preserves_extension() {
        assert_eq!(
            tmp_path(Path::new("/x/foo.parquet")),
            PathBuf::from("/x/foo.parquet.tmp")
        );
        assert_eq!(
            tmp_path(Path::new("/x/foo.json")),
            PathBuf::from("/x/foo.json.tmp")
        );
        assert_eq!(tmp_path(Path::new("/x/foo")), PathBuf::from("/x/foo.tmp"));
    }

    #[test]
    fn write_json_atomic_produces_exact_bytes_and_removes_tmp() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("out.json");

        let value = Mock {
            a: 7,
            b: "hi".to_string(),
        };
        write_json_atomic(&path, &value, "mock").expect("write should succeed");

        let expected = serde_json::to_vec_pretty(&value).expect("serialize");
        let actual = std::fs::read(&path).expect("read");
        assert_eq!(actual, expected, "written bytes must match pretty JSON");

        assert!(
            !tmp_path(&path).exists(),
            "tmp file must be removed after rename"
        );
    }

    /// A writer that fails on every `flush`, to drive the drop-flush error path.
    struct FlushFails;

    impl io::Write for FlushFails {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("flush failed"))
        }
    }

    #[test]
    fn flush_error_propagates_and_leaves_no_target_file() {
        let dir = TempDir::new().expect("temp dir");
        let target = dir.path().join("never_installed.json");
        let tmp = tmp_path(&target);

        let value = Mock {
            a: 1,
            b: "x".to_string(),
        };

        let result = serialize_json_then_flush(FlushFails, &value, "mock", &tmp);

        assert!(
            matches!(result, Err(OutputError::IoError { .. })),
            "flush failure must surface as IoError, got: {result:?}"
        );
        assert!(
            !target.exists(),
            "no file may be installed at the target path on flush failure"
        );
    }

    #[test]
    fn write_bytes_atomic_round_trips_and_removes_tmp() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("blob.bin");

        let bytes = b"some payload bytes";
        write_bytes_atomic(&path, bytes).expect("write should succeed");

        assert_eq!(std::fs::read(&path).expect("read"), bytes);
        assert!(
            !tmp_path(&path).exists(),
            "tmp file must be removed after rename"
        );
    }
}
