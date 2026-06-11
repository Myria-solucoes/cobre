//! Shared atomic-write helpers for output files.
//!
//! Every output artifact this crate writes must be crash-safe: a reader must
//! never observe a half-written file, and a write that fails mid-stream must
//! not install a truncated file in place of a previous good one. This module is
//! the single owner of that contract for the write side, mirroring the shared
//! read-side helpers.
//!
//! The mechanism is write-to-temp-then-rename: data is written to a sibling
//! `{path}.tmp`, the buffered writer is flushed **explicitly** (not via `Drop`),
//! and only then is the temporary file renamed onto `path` — `rename` within a
//! filesystem is atomic, so the target is either the old contents or the
//! complete new contents, never a partial one.
//!
//! The explicit flush is load-bearing: a `BufWriter` flushes on drop, but
//! `Drop::drop` cannot return an error, so a drop-flush **swallows** any
//! `ENOSPC`/`EIO` reported while emitting the buffered tail. The subsequent
//! rename would then install a truncated file with no error surfaced to the
//! caller. Always flush via [`std::io::Write::flush`] (here) and propagate the
//! error with `?` — never rely on drop-flush before a rename.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use super::error::OutputError;
use super::parquet_config::ParquetWriterConfig;

/// Compute the temporary sibling path used during an atomic write.
///
/// The original extension is preserved as a prefix of `.tmp`
/// (`foo.parquet` → `foo.parquet.tmp`, `foo.json` → `foo.json.tmp`), so the
/// temporary's extension still identifies the payload format and never
/// collides with a differently-typed sibling. Extension-less paths become
/// `foo.tmp`.
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
    // Explicit flush before rename: drop-flush would swallow a write error and
    // could leave a truncated file to be renamed into place.
    writer.flush().map_err(|e| OutputError::io(&tmp, e))?;

    std::fs::rename(&tmp, path).map_err(|e| OutputError::io(path, e))?;
    Ok(())
}

/// Serialize `value` to pretty-printed JSON and write it to `path` atomically.
///
/// Serialization streams directly into a flushed `BufWriter` over the
/// temporary file; the byte content is identical to `serde_json` pretty output
/// produced by `to_string_pretty`/`to_vec_pretty`. The parent directory must
/// already exist.
///
/// `entity` labels the [`OutputError::SerializationError`] raised if
/// serialization fails, so callers retain a descriptive error context.
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
/// Split out from [`write_json_atomic`] so the serialize-then-explicit-flush
/// step — the part that must propagate a flush error rather than swallow it on
/// drop — can be exercised with an injected failing writer in tests; the rename
/// in [`write_json_atomic`] only runs when this returns `Ok`, so a flush error
/// can never install a target file. `tmp` names the in-flight temporary file
/// solely for I/O-error context.
fn serialize_json_then_flush<W: Write>(
    sink: W,
    value: &impl serde::Serialize,
    entity: &str,
    tmp: &Path,
) -> Result<(), OutputError> {
    let mut writer = BufWriter::new(sink);
    serde_json::to_writer_pretty(&mut writer, value)
        .map_err(|e| OutputError::serialization(entity, format!("JSON serialization: {e}")))?;
    // Explicit flush before the caller's rename: drop-flush would swallow a
    // write error and could leave a truncated file to be renamed into place.
    writer.flush().map_err(|e| OutputError::io(tmp, e))?;
    Ok(())
}

/// Write a `RecordBatch` to `path` as a Parquet file, atomically.
///
/// Honors `config` (compression, row-group size, dictionary encoding); there is
/// no hard-coded codec. The parent directory must already exist.
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

    // `ArrowWriter::into_inner` finalizes the parquet footer into the
    // `BufWriter` but does NOT flush the `BufWriter` into the underlying `File`.
    // The explicit `flush` below guarantees every buffered byte reaches the file
    // before the atomic rename — relying on drop-flush would swallow I/O errors
    // and could truncate the file.
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
    use super::*;
    use serde::Serialize;
    use std::io;
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

        // Byte content must match serde_json pretty output exactly.
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
        // Drive `write_json_atomic`'s serialize-then-explicit-flush step with a
        // sink that errors on flush. It must surface an `IoError`; because the
        // rename in `write_json_atomic` runs only on `Ok`, the target path must
        // not exist afterward — exactly the drop-flush data-loss bug this
        // module exists to prevent.
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
