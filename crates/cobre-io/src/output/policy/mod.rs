//! `FlatBuffers` builder and reader types for policy checkpoint serialization.
//!
//! Types use generic names to maintain infrastructure crate genericity; conversion
//! from algorithm-specific types is the calling crate's responsibility.
//!
//! The canonical wire-format description is `schemas/policy.fbs` in this crate
//! (namespace `Cobre.IO.Policy`, tables `StageCuts`, `Cut`, `StageBasis`,
//! `StageStates`); the build hand-writes both the builder calls and the safe
//! raw-byte parser rather than consuming the schema. The `*_FIELD_*: u16` slot
//! constants in `codec` mirror the schema's `(id: N)` attributes via
//! `slot = (id + 2) * 2` and MUST stay in sync — the `flatc-conformance` feature
//! gates the round-trip test in `tests/flatbuffers_schema_conformance.rs` that
//! fails when they diverge. The wire layout itself is documented at `codec`.

pub mod checkpoint;
pub mod codec;
pub mod records;

pub use checkpoint::{read_policy_checkpoint, write_policy_checkpoint};
pub use codec::{deserialize_stage_basis, deserialize_stage_cuts, deserialize_stage_states};
pub use codec::{serialize_stage_basis, serialize_stage_cuts, serialize_stage_states};
pub use records::{
    OwnedPolicyBasisRecord, OwnedPolicyCutRecord, PolicyBasisRecord, PolicyCheckpoint,
    PolicyCheckpointMetadata, PolicyCutRecord, StageCutsPayload, StageCutsReadResult,
    StageStatesPayload, StageStatesReadResult,
};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::super::error::OutputError;
    use super::*;

    fn make_cut_record(
        cut_id: u64,
        slot_index: u32,
        iteration: u32,
        coefficients: &[f64],
    ) -> PolicyCutRecord<'_> {
        PolicyCutRecord {
            cut_id,
            slot_index,
            iteration,
            forward_pass_index: 0,
            intercept: 42.0,
            coefficients,
            is_active: true,
        }
    }

    // ── serialize_stage_cuts tests ────────────────────────────────────────────

    #[test]
    fn serialize_stage_cuts_single_cut_round_trip() {
        let coefficients = [1.0_f64, 2.0, 3.0];
        let cut = PolicyCutRecord {
            cut_id: 7,
            slot_index: 5,
            iteration: 3,
            forward_pass_index: 0,
            intercept: 42.0,
            coefficients: &coefficients,
            is_active: true,
        };

        let buf = serialize_stage_cuts(0, 3, 100, 0, &[cut], &[0], 1);

        assert!(!buf.is_empty(), "buffer must not be empty");
        assert!(buf.len() >= 4, "buffer must have at least 4 bytes");
        let root_offset = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        assert!(
            root_offset < buf.len(),
            "root offset must point inside the buffer"
        );
    }

    #[test]
    fn serialize_stage_cuts_empty_cuts_valid_buffer() {
        let buf = serialize_stage_cuts(0, 3, 100, 0, &[], &[], 0);

        assert!(!buf.is_empty(), "buffer must not be empty for empty cuts");
        assert!(
            buf.len() >= 4,
            "buffer must have at least 4 bytes even for empty cuts"
        );
        let root_offset = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        assert!(
            root_offset < buf.len(),
            "root offset must point inside the buffer"
        );
    }

    #[test]
    fn serialize_stage_cuts_multiple_cuts_deterministic() {
        let c0 = [1.0_f64, 2.0, 3.0];
        let c1 = [4.0_f64, 5.0, 6.0];
        let c2 = [7.0_f64, 8.0, 9.0];

        let cuts = [
            make_cut_record(1, 0, 1, &c0),
            make_cut_record(2, 1, 1, &c1),
            make_cut_record(3, 2, 1, &c2),
        ];

        let buf_a = serialize_stage_cuts(5, 3, 50, 0, &cuts, &[0, 1, 2], 3);
        let buf_b = serialize_stage_cuts(5, 3, 50, 0, &cuts, &[0, 1, 2], 3);

        assert_eq!(buf_a, buf_b, "output must be byte-identical for same input");
    }

    #[test]
    fn serialize_stage_cuts_non_empty_for_varying_state_dimensions() {
        for &dim in &[1u32, 10, 100, 1000] {
            let coefs: Vec<f64> = (0..dim).map(f64::from).collect();
            let cut = PolicyCutRecord {
                cut_id: 0,
                slot_index: 0,
                iteration: 1,
                forward_pass_index: 0,
                intercept: 0.0,
                coefficients: &coefs,
                is_active: true,
            };
            let buf = serialize_stage_cuts(0, dim, 10, 0, &[cut], &[0], 1);
            assert!(
                !buf.is_empty(),
                "buffer must not be empty for state_dimension={dim}"
            );
        }
    }

    // ── serialize_stage_basis tests ───────────────────────────────────────────

    #[test]
    fn serialize_stage_basis_round_trip() {
        let record = PolicyBasisRecord {
            stage_id: 0,
            iteration: 5,
            column_status: &[0, 1, 2],
            row_status: &[1, 1, 0, 0],
            num_cut_rows: 2,
        };

        let buf = serialize_stage_basis(&record);

        assert!(!buf.is_empty(), "buffer must not be empty");
        assert!(buf.len() >= 4, "buffer must have at least 4 bytes");
        let root_offset = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        assert!(
            root_offset < buf.len(),
            "root offset must point inside the buffer"
        );
    }

    #[test]
    fn serialize_stage_basis_empty_status_vectors() {
        let record = PolicyBasisRecord {
            stage_id: 1,
            iteration: 0,
            column_status: &[],
            row_status: &[],
            num_cut_rows: 0,
        };

        let buf = serialize_stage_basis(&record);

        assert!(
            !buf.is_empty(),
            "buffer must not be empty even with empty status vectors"
        );
        assert!(
            buf.len() >= 4,
            "buffer must have at least 4 bytes even with empty status vectors"
        );
    }

    #[test]
    fn serialize_stage_basis_deterministic() {
        let col = [0u8, 1, 2, 3];
        let row = [1u8, 0, 1, 0, 1];
        let record = PolicyBasisRecord {
            stage_id: 7,
            iteration: 12,
            column_status: &col,
            row_status: &row,
            num_cut_rows: 3,
        };

        let buf_a = serialize_stage_basis(&record);
        let buf_b = serialize_stage_basis(&record);

        assert_eq!(
            buf_a, buf_b,
            "basis output must be byte-identical for same input"
        );
    }

    // ── PolicyCheckpointMetadata tests ────────────────────────────────────────

    #[test]
    fn policy_checkpoint_metadata_serializes_to_json() {
        let meta = PolicyCheckpointMetadata {
            cobre_version: "0.0.1".to_string(),
            created_at: "2026-03-08T00:00:00Z".to_string(),
            completed_iterations: 50,
            final_lower_bound: 1234.56,
            best_upper_bound: Some(1300.0),
            state_dimension: 160,
            num_stages: 60,
            max_iterations: 200,
            forward_passes: 4,
            warm_start_cuts: 0,
            warm_start_counts: vec![],
            rng_seed: 42,
            total_visited_states: 0,
        };

        let json = serde_json::to_string_pretty(&meta)
            .expect("PolicyCheckpointMetadata must serialize to JSON without error");

        assert!(
            json.contains("completed_iterations"),
            "JSON must contain 'completed_iterations'"
        );
        assert!(
            json.contains("50"),
            "JSON must contain the completed_iterations value"
        );
        assert!(
            json.contains("final_lower_bound"),
            "JSON must contain 'final_lower_bound'"
        );
        assert!(
            json.contains("state_dimension"),
            "JSON must contain 'state_dimension'"
        );
        assert!(json.contains("rng_seed"), "JSON must contain 'rng_seed'");
        assert!(
            json.contains("best_upper_bound"),
            "JSON must contain 'best_upper_bound'"
        );
        assert!(
            json.contains("1300"),
            "JSON must contain the best_upper_bound value"
        );

        // Verify it round-trips through serde_json::Value.
        let value: serde_json::Value =
            serde_json::from_str(&json).expect("JSON output must be parseable");
        assert_eq!(
            value["completed_iterations"].as_u64(),
            Some(50),
            "completed_iterations must deserialize correctly"
        );
        assert_eq!(
            value["rng_seed"].as_u64(),
            Some(42),
            "rng_seed must deserialize correctly"
        );
    }

    #[test]
    fn policy_checkpoint_metadata_none_upper_bound_serializes_to_null() {
        let meta = PolicyCheckpointMetadata {
            cobre_version: "0.0.1".to_string(),
            created_at: "2026-03-08T00:00:00Z".to_string(),
            completed_iterations: 10,
            final_lower_bound: 0.0,
            best_upper_bound: None,
            state_dimension: 1,
            num_stages: 1,
            max_iterations: 10,
            forward_passes: 1,
            warm_start_cuts: 0,
            warm_start_counts: vec![],
            rng_seed: 0,
            total_visited_states: 0,
        };

        let json = serde_json::to_string_pretty(&meta)
            .expect("PolicyCheckpointMetadata must serialize to JSON");

        let value: serde_json::Value =
            serde_json::from_str(&json).expect("JSON output must be parseable");
        assert!(
            value["best_upper_bound"].is_null(),
            "best_upper_bound must serialize to null when None"
        );
    }

    // ── write_policy_checkpoint tests ─────────────────────────────────────────

    /// Build a minimal [`PolicyCheckpointMetadata`] for use in checkpoint tests.
    fn make_metadata(num_stages: u32, state_dimension: u32) -> PolicyCheckpointMetadata {
        PolicyCheckpointMetadata {
            cobre_version: "0.0.1".to_string(),
            created_at: "2026-03-08T00:00:00Z".to_string(),
            completed_iterations: 10,
            final_lower_bound: 999.0,
            best_upper_bound: Some(1050.0),
            state_dimension,
            num_stages,
            max_iterations: 100,
            forward_passes: 4,
            warm_start_cuts: 0,
            warm_start_counts: vec![0; num_stages as usize],
            rng_seed: 42,
            total_visited_states: 0,
        }
    }

    /// Build a [`StageCutsPayload`] with `n_cuts` cuts, all using the supplied
    /// `coefficients` slice (shared across cuts for test simplicity).
    fn make_stage_cuts_payload<'a>(
        stage_id: u32,
        cuts: &'a [PolicyCutRecord<'a>],
        active_cut_indices: &'a [u32],
        state_dimension: u32,
    ) -> StageCutsPayload<'a> {
        StageCutsPayload {
            stage_id,
            state_dimension,
            capacity: 100,
            warm_start_count: 0,
            cuts,
            active_cut_indices,
            populated_count: u32::try_from(cuts.len()).unwrap(),
        }
    }

    /// Build a [`PolicyBasisRecord`] for the given stage.
    fn make_basis_record(stage_id: u32) -> PolicyBasisRecord<'static> {
        PolicyBasisRecord {
            stage_id,
            iteration: 10,
            column_status: &[0, 1, 2, 3],
            row_status: &[1, 0, 1, 0, 1],
            num_cut_rows: 2,
        }
    }

    #[test]
    fn write_policy_checkpoint_creates_directory_structure() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64, 2.0, 3.0];
        let c1 = [4.0_f64, 5.0, 6.0];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0), make_cut_record(2, 1, 1, &c1)];
        let cuts_s1 = [make_cut_record(3, 0, 2, &c0)];
        let cuts_s2 = [make_cut_record(4, 0, 3, &c1)];

        let stage_cuts = [
            make_stage_cuts_payload(0, &cuts_s0, &[0, 1], 3),
            make_stage_cuts_payload(1, &cuts_s1, &[0], 3),
            make_stage_cuts_payload(2, &cuts_s2, &[0], 3),
        ];
        let basis_records = [
            make_basis_record(0),
            make_basis_record(1),
            make_basis_record(2),
        ];
        let metadata = make_metadata(3, 3);

        write_policy_checkpoint(tmp.path(), &stage_cuts, &basis_records, &metadata, &[])
            .expect("write_policy_checkpoint must succeed");

        // Directories must exist.
        assert!(tmp.path().join("cuts").is_dir(), "cuts/ must exist");
        assert!(tmp.path().join("basis").is_dir(), "basis/ must exist");

        // All cut files must exist.
        for i in 0..3u32 {
            let p = tmp.path().join(format!("cuts/stage_{i:03}.bin"));
            assert!(p.is_file(), "cuts/stage_{i:03}.bin must exist");
        }

        // All basis files must exist.
        for i in 0..3u32 {
            let p = tmp.path().join(format!("basis/stage_{i:03}.bin"));
            assert!(p.is_file(), "basis/stage_{i:03}.bin must exist");
        }

        // metadata.json must exist.
        assert!(
            tmp.path().join("metadata.json").is_file(),
            "metadata.json must exist"
        );
    }

    #[test]
    fn write_policy_checkpoint_metadata_json_valid() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64, 2.0, 3.0];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0)];
        let stage_cuts = [make_stage_cuts_payload(0, &cuts_s0, &[0], 3)];
        let metadata = make_metadata(1, 3);

        write_policy_checkpoint(tmp.path(), &stage_cuts, &[], &metadata, &[])
            .expect("write_policy_checkpoint must succeed");

        let content = std::fs::read_to_string(tmp.path().join("metadata.json")).unwrap();
        let value: serde_json::Value =
            serde_json::from_str(&content).expect("metadata.json must be valid JSON");

        for key in &[
            "cobre_version",
            "created_at",
            "completed_iterations",
            "final_lower_bound",
            "state_dimension",
            "num_stages",
        ] {
            assert!(
                value.get(key).is_some(),
                "metadata.json must contain key '{key}'"
            );
        }

        assert_eq!(
            value["completed_iterations"].as_u64(),
            Some(10),
            "completed_iterations must match"
        );
        assert_eq!(
            value["num_stages"].as_u64(),
            Some(1),
            "num_stages must match"
        );
        assert_eq!(
            value["state_dimension"].as_u64(),
            Some(3),
            "state_dimension must match"
        );
    }

    #[test]
    fn write_policy_checkpoint_cut_files_non_empty() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64, 2.0, 3.0];
        let c1 = [4.0_f64, 5.0, 6.0];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0), make_cut_record(2, 1, 1, &c1)];
        let cuts_s1 = [make_cut_record(3, 0, 2, &c0)];
        let cuts_s2 = [make_cut_record(4, 0, 3, &c1)];

        let stage_cuts = [
            make_stage_cuts_payload(0, &cuts_s0, &[0, 1], 3),
            make_stage_cuts_payload(1, &cuts_s1, &[0], 3),
            make_stage_cuts_payload(2, &cuts_s2, &[0], 3),
        ];
        let metadata = make_metadata(3, 3);

        write_policy_checkpoint(tmp.path(), &stage_cuts, &[], &metadata, &[])
            .expect("write_policy_checkpoint must succeed");

        for i in 0..3u32 {
            let p = tmp.path().join(format!("cuts/stage_{i:03}.bin"));
            let bytes = std::fs::read(&p).unwrap();
            assert!(!bytes.is_empty(), "cuts/stage_{i:03}.bin must not be empty");
            assert!(
                bytes.len() >= 4,
                "cuts/stage_{i:03}.bin must have >= 4 bytes"
            );
            let root_offset = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
            assert!(
                root_offset < bytes.len(),
                "cuts/stage_{i:03}.bin root offset must be in-range"
            );
        }
    }

    #[test]
    fn write_policy_checkpoint_basis_files_non_empty() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64, 2.0];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0)];
        let stage_cuts = [make_stage_cuts_payload(0, &cuts_s0, &[0], 2)];
        let basis_records = [make_basis_record(0)];
        let metadata = make_metadata(1, 2);

        write_policy_checkpoint(tmp.path(), &stage_cuts, &basis_records, &metadata, &[])
            .expect("write_policy_checkpoint must succeed");

        let p = tmp.path().join("basis/stage_000.bin");
        let bytes = std::fs::read(&p).unwrap();
        assert!(!bytes.is_empty(), "basis/stage_000.bin must not be empty");
        assert!(bytes.len() >= 4, "basis/stage_000.bin must have >= 4 bytes");
        let root_offset = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        assert!(
            root_offset < bytes.len(),
            "basis/stage_000.bin root offset must be in-range"
        );
    }

    #[test]
    fn write_policy_checkpoint_empty_bases_no_basis_files() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64, 2.0];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0)];
        let stage_cuts = [make_stage_cuts_payload(0, &cuts_s0, &[0], 2)];
        let metadata = make_metadata(1, 2);

        let result = write_policy_checkpoint(tmp.path(), &stage_cuts, &[], &metadata, &[]);

        assert!(
            result.is_ok(),
            "write_policy_checkpoint must return Ok(()) with empty stage_bases"
        );

        // basis/ directory must exist.
        assert!(
            tmp.path().join("basis").is_dir(),
            "basis/ directory must exist even with empty stage_bases"
        );

        // No .bin files inside basis/.
        let entries: Vec<_> = std::fs::read_dir(tmp.path().join("basis"))
            .unwrap()
            .filter_map(std::result::Result::ok)
            .collect();
        assert!(
            entries.is_empty(),
            "basis/ must contain no files when stage_bases is empty"
        );
    }

    /// Returns `true` when running as root (UID 0). Used to skip permission tests.
    #[cfg(unix)]
    fn is_root() -> bool {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("Uid:"))
                    .and_then(|l| l.split_whitespace().nth(2))
                    .and_then(|uid| uid.parse::<u32>().ok())
            })
            == Some(0)
    }

    #[cfg(not(unix))]
    fn is_root() -> bool {
        false
    }

    #[test]
    fn write_policy_checkpoint_error_on_readonly_dir() {
        // Skip this test on platforms where read-only enforcement is unreliable
        // (e.g., when running as root).
        if is_root() {
            return;
        }

        let tmp = tempfile::tempdir().unwrap();

        // Make the temp directory itself read-only so create_dir_all fails.
        let mut perms = std::fs::metadata(tmp.path()).unwrap().permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o555);
        std::fs::set_permissions(tmp.path(), perms).unwrap();

        let readonly_target = tmp.path().join("policy");

        let c0 = [1.0_f64];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0)];
        let stage_cuts = [make_stage_cuts_payload(0, &cuts_s0, &[0], 1)];
        let metadata = make_metadata(1, 1);

        let result = write_policy_checkpoint(&readonly_target, &stage_cuts, &[], &metadata, &[]);

        // Restore permissions so the tempdir can be cleaned up.
        let mut perms2 = std::fs::metadata(tmp.path()).unwrap().permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perms2, 0o755);
        std::fs::set_permissions(tmp.path(), perms2).unwrap();

        assert!(
            matches!(result, Err(OutputError::IoError { .. })),
            "write_policy_checkpoint must return Err(OutputError::IoError) on read-only dir, got: {result:?}"
        );
    }

    #[test]
    fn write_policy_checkpoint_stage_numbering_zero_padded() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64, 2.0];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0)];
        let cuts_s1 = [make_cut_record(2, 0, 1, &c0)];
        let cuts_s59 = [make_cut_record(3, 0, 1, &c0)];

        let stage_cuts = [
            make_stage_cuts_payload(0, &cuts_s0, &[0], 2),
            make_stage_cuts_payload(1, &cuts_s1, &[0], 2),
            make_stage_cuts_payload(59, &cuts_s59, &[0], 2),
        ];
        let basis_records_0 = PolicyBasisRecord {
            stage_id: 0,
            iteration: 1,
            column_status: &[0u8],
            row_status: &[1u8],
            num_cut_rows: 0,
        };
        let basis_records_1 = PolicyBasisRecord {
            stage_id: 1,
            iteration: 1,
            column_status: &[0u8],
            row_status: &[1u8],
            num_cut_rows: 0,
        };
        let basis_records_59 = PolicyBasisRecord {
            stage_id: 59,
            iteration: 1,
            column_status: &[0u8],
            row_status: &[1u8],
            num_cut_rows: 0,
        };
        let stage_bases = [basis_records_0, basis_records_1, basis_records_59];
        let metadata = make_metadata(3, 2);

        write_policy_checkpoint(tmp.path(), &stage_cuts, &stage_bases, &metadata, &[])
            .expect("write_policy_checkpoint must succeed");

        assert!(
            tmp.path().join("cuts/stage_000.bin").is_file(),
            "cuts/stage_000.bin must exist"
        );
        assert!(
            tmp.path().join("cuts/stage_001.bin").is_file(),
            "cuts/stage_001.bin must exist"
        );
        assert!(
            tmp.path().join("cuts/stage_059.bin").is_file(),
            "cuts/stage_059.bin must exist"
        );
        assert!(
            tmp.path().join("basis/stage_000.bin").is_file(),
            "basis/stage_000.bin must exist"
        );
        assert!(
            tmp.path().join("basis/stage_001.bin").is_file(),
            "basis/stage_001.bin must exist"
        );
        assert!(
            tmp.path().join("basis/stage_059.bin").is_file(),
            "basis/stage_059.bin must exist"
        );
    }

    // ── deserialize_stage_cuts tests ──────────────────────────────────────────

    #[test]
    fn deserialize_stage_cuts_single_cut_all_fields() {
        let coefficients = [1.0_f64, 2.0, 3.0];
        let cut = PolicyCutRecord {
            cut_id: 7,
            slot_index: 5,
            iteration: 3,
            forward_pass_index: 2,
            intercept: 42.0,
            coefficients: &coefficients,
            is_active: true,
        };

        let buf = serialize_stage_cuts(0, 3, 100, 0, &[cut], &[0], 1);
        let result = deserialize_stage_cuts(&buf).expect("deserialization must succeed");

        assert_eq!(result.stage_id, 0, "stage_id must round-trip");
        assert_eq!(result.state_dimension, 3, "state_dimension must round-trip");
        assert_eq!(result.capacity, 100, "capacity must round-trip");
        assert_eq!(
            result.warm_start_count, 0,
            "warm_start_count must round-trip"
        );
        assert_eq!(result.populated_count, 1, "populated_count must round-trip");
        assert_eq!(result.cuts.len(), 1, "one cut must be deserialized");

        let c = &result.cuts[0];
        assert_eq!(c.cut_id, 7, "cut_id must round-trip");
        assert_eq!(c.slot_index, 5, "slot_index must round-trip");
        assert_eq!(c.iteration, 3, "iteration must round-trip");
        assert_eq!(
            c.forward_pass_index, 2,
            "forward_pass_index must round-trip"
        );
        assert_eq!(c.intercept, 42.0, "intercept must round-trip");
        assert_eq!(
            c.coefficients,
            &[1.0, 2.0, 3.0],
            "coefficients must round-trip"
        );
        assert!(c.is_active, "is_active must round-trip");
    }

    #[test]
    fn deserialize_stage_cuts_three_cuts_all_match() {
        let c0 = [1.0_f64, 0.5];
        let c1 = [2.0_f64, 1.5];
        let c2 = [3.0_f64, 2.5];
        let cuts = [
            PolicyCutRecord {
                cut_id: 10,
                slot_index: 0,
                iteration: 1,
                forward_pass_index: 0,
                intercept: 100.0,
                coefficients: &c0,
                is_active: true,
            },
            PolicyCutRecord {
                cut_id: 20,
                slot_index: 1,
                iteration: 2,
                forward_pass_index: 1,
                intercept: 200.0,
                coefficients: &c1,
                is_active: false,
            },
            PolicyCutRecord {
                cut_id: 30,
                slot_index: 2,
                iteration: 3,
                forward_pass_index: 2,
                intercept: 300.0,
                coefficients: &c2,
                is_active: true,
            },
        ];

        let buf = serialize_stage_cuts(5, 2, 50, 1, &cuts, &[0, 2], 3);
        let result = deserialize_stage_cuts(&buf).expect("deserialization must succeed");

        assert_eq!(result.stage_id, 5);
        assert_eq!(result.state_dimension, 2);
        assert_eq!(result.capacity, 50);
        assert_eq!(result.warm_start_count, 1);
        assert_eq!(result.populated_count, 3);
        assert_eq!(result.cuts.len(), 3);

        let expected_cut_ids = [10u64, 20, 30];
        let expected_intercepts = [100.0f64, 200.0, 300.0];
        let expected_coefficients = [&c0[..], &c1[..], &c2[..]];
        let expected_active = [true, false, true];

        for (i, cut) in result.cuts.iter().enumerate() {
            assert_eq!(cut.cut_id, expected_cut_ids[i], "cut {i} cut_id");
            assert_eq!(cut.intercept, expected_intercepts[i], "cut {i} intercept");
            assert_eq!(
                cut.coefficients, expected_coefficients[i],
                "cut {i} coefficients"
            );
            assert_eq!(cut.is_active, expected_active[i], "cut {i} is_active");
        }
    }

    #[test]
    fn deserialize_stage_cuts_empty_cut_pool() {
        let buf = serialize_stage_cuts(2, 10, 200, 0, &[], &[], 0);
        let result =
            deserialize_stage_cuts(&buf).expect("deserialization of empty cut pool must succeed");

        assert_eq!(result.stage_id, 2);
        assert_eq!(result.capacity, 200);
        assert_eq!(result.populated_count, 0);
        assert!(
            result.cuts.is_empty(),
            "empty cut pool must produce zero cuts"
        );
    }

    #[test]
    fn deserialize_stage_cuts_zero_length_coefficients() {
        let cut = PolicyCutRecord {
            cut_id: 1,
            slot_index: 0,
            iteration: 1,
            forward_pass_index: 0,
            intercept: 5.0,
            coefficients: &[],
            is_active: true,
        };
        let buf = serialize_stage_cuts(0, 0, 10, 0, &[cut], &[0], 1);
        let result =
            deserialize_stage_cuts(&buf).expect("zero-length coefficients must deserialize");
        assert_eq!(result.cuts.len(), 1);
        assert!(
            result.cuts[0].coefficients.is_empty(),
            "empty coefficients must round-trip"
        );
    }

    #[test]
    fn deserialize_stage_cuts_large_coefficient_vector() {
        let dim = 1000u32;
        let coefs: Vec<f64> = (0..dim).map(f64::from).collect();
        let cut = PolicyCutRecord {
            cut_id: 42,
            slot_index: 0,
            iteration: 1,
            forward_pass_index: 0,
            intercept: -99.0,
            coefficients: &coefs,
            is_active: false,
        };
        let buf = serialize_stage_cuts(3, dim, 10, 0, &[cut], &[0], 1);
        let result =
            deserialize_stage_cuts(&buf).expect("large coefficient vector must deserialize");
        assert_eq!(result.cuts[0].coefficients.len(), dim as usize);
        assert_eq!(result.cuts[0].coefficients[999], 999.0);
        assert_eq!(result.cuts[0].intercept, -99.0);
    }

    #[test]
    fn deserialize_stage_cuts_truncated_buffer_returns_error() {
        let coefs = [1.0_f64, 2.0];
        let cut = make_cut_record(1, 0, 1, &coefs);
        let full_buf = serialize_stage_cuts(0, 2, 10, 0, &[cut], &[0], 1);
        // Truncate to 2 bytes — root offset itself is incomplete.
        let truncated = &full_buf[..2];
        let result = deserialize_stage_cuts(truncated);
        assert!(result.is_err(), "truncated buffer must return an error");
    }

    #[test]
    fn deserialize_stage_cuts_stage_id_nonzero() {
        let buf = serialize_stage_cuts(59, 4, 50, 0, &[], &[], 0);
        let result = deserialize_stage_cuts(&buf).expect("stage_id=59 must deserialize");
        assert_eq!(result.stage_id, 59, "stage_id=59 must round-trip");
    }

    // ── deserialize_stage_basis tests ─────────────────────────────────────────

    #[test]
    fn deserialize_stage_basis_all_fields() {
        let record = PolicyBasisRecord {
            stage_id: 3,
            iteration: 7,
            column_status: &[0, 1, 2, 3],
            row_status: &[1, 0, 1, 0, 1],
            num_cut_rows: 2,
        };

        let buf = serialize_stage_basis(&record);
        let owned = deserialize_stage_basis(&buf).expect("basis round-trip must succeed");

        assert_eq!(owned.stage_id, 3, "stage_id must round-trip");
        assert_eq!(owned.iteration, 7, "iteration must round-trip");
        assert_eq!(
            owned.column_status,
            &[0u8, 1, 2, 3],
            "column_status must round-trip"
        );
        assert_eq!(
            owned.row_status,
            &[1u8, 0, 1, 0, 1],
            "row_status must round-trip"
        );
        assert_eq!(owned.num_cut_rows, 2, "num_cut_rows must round-trip");
    }

    #[test]
    fn deserialize_stage_basis_empty_status_vectors() {
        let record = PolicyBasisRecord {
            stage_id: 0,
            iteration: 0,
            column_status: &[],
            row_status: &[],
            num_cut_rows: 0,
        };

        let buf = serialize_stage_basis(&record);
        let owned = deserialize_stage_basis(&buf).expect("empty basis must deserialize");

        assert!(
            owned.column_status.is_empty(),
            "empty column_status must round-trip"
        );
        assert!(
            owned.row_status.is_empty(),
            "empty row_status must round-trip"
        );
        assert_eq!(owned.num_cut_rows, 0);
    }

    #[test]
    fn deserialize_stage_basis_large_status_vectors() {
        let col: Vec<u8> = (0..200u8).collect();
        let row: Vec<u8> = (0..100u8).rev().collect();
        let record = PolicyBasisRecord {
            stage_id: 10,
            iteration: 99,
            column_status: &col,
            row_status: &row,
            num_cut_rows: 50,
        };

        let buf = serialize_stage_basis(&record);
        let owned = deserialize_stage_basis(&buf).expect("large basis must deserialize");

        assert_eq!(owned.column_status, col);
        assert_eq!(owned.row_status, row);
        assert_eq!(owned.num_cut_rows, 50);
    }

    #[test]
    fn deserialize_stage_basis_truncated_buffer_returns_error() {
        let record = PolicyBasisRecord {
            stage_id: 0,
            iteration: 1,
            column_status: &[0, 1],
            row_status: &[1, 0],
            num_cut_rows: 0,
        };
        let full_buf = serialize_stage_basis(&record);
        let truncated = &full_buf[..3];
        let result = deserialize_stage_basis(truncated);
        assert!(
            result.is_err(),
            "truncated basis buffer must return an error"
        );
    }

    // ── PolicyCheckpointMetadata deserialization tests ────────────────────────

    #[test]
    fn policy_checkpoint_metadata_deserializes_from_json() {
        let meta = PolicyCheckpointMetadata {
            cobre_version: "0.0.1".to_string(),
            created_at: "2026-03-08T00:00:00Z".to_string(),
            completed_iterations: 42,
            final_lower_bound: 9999.0,
            best_upper_bound: Some(10100.0),
            state_dimension: 5,
            num_stages: 3,
            max_iterations: 100,
            forward_passes: 4,
            warm_start_cuts: 10,
            warm_start_counts: vec![10, 10, 10],
            rng_seed: 12345,
            total_visited_states: 0,
        };

        let json = serde_json::to_string(&meta).expect("serialize must succeed");
        let back: PolicyCheckpointMetadata =
            serde_json::from_str(&json).expect("deserialize must succeed");

        assert_eq!(back.cobre_version, meta.cobre_version);
        assert_eq!(back.completed_iterations, meta.completed_iterations);
        assert_eq!(back.final_lower_bound, meta.final_lower_bound);
        assert_eq!(back.best_upper_bound, meta.best_upper_bound);
        assert_eq!(back.state_dimension, meta.state_dimension);
        assert_eq!(back.num_stages, meta.num_stages);
        assert_eq!(back.max_iterations, meta.max_iterations);
        assert_eq!(back.forward_passes, meta.forward_passes);
        assert_eq!(back.warm_start_cuts, meta.warm_start_cuts);
        assert_eq!(back.rng_seed, meta.rng_seed);
    }

    #[test]
    fn policy_checkpoint_metadata_deserializes_none_upper_bound() {
        let meta = PolicyCheckpointMetadata {
            cobre_version: "0.0.1".to_string(),
            created_at: "2026-03-08T00:00:00Z".to_string(),
            completed_iterations: 1,
            final_lower_bound: 0.0,
            best_upper_bound: None,
            state_dimension: 1,
            num_stages: 1,
            max_iterations: 10,
            forward_passes: 1,
            warm_start_cuts: 0,
            warm_start_counts: vec![],
            rng_seed: 0,
            total_visited_states: 0,
        };

        let json = serde_json::to_string(&meta).expect("serialize must succeed");
        let back: PolicyCheckpointMetadata =
            serde_json::from_str(&json).expect("deserialize must succeed");

        assert!(
            back.best_upper_bound.is_none(),
            "None upper bound must round-trip"
        );
    }

    // ── read_policy_checkpoint round-trip tests ───────────────────────────────

    #[test]
    fn read_policy_checkpoint_full_round_trip() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64, 2.0, 3.0];
        let c1 = [4.0_f64, 5.0, 6.0];
        let c2 = [7.0_f64, 8.0, 9.0];

        let cuts_s0 = [make_cut_record(1, 0, 1, &c0), make_cut_record(2, 1, 1, &c1)];
        let cuts_s1 = [make_cut_record(3, 0, 2, &c2)];

        let stage_cuts_payloads = [
            make_stage_cuts_payload(0, &cuts_s0, &[0, 1], 3),
            make_stage_cuts_payload(1, &cuts_s1, &[0], 3),
        ];
        let basis_records = [make_basis_record(0), make_basis_record(1)];
        let metadata = make_metadata(2, 3);

        write_policy_checkpoint(
            tmp.path(),
            &stage_cuts_payloads,
            &basis_records,
            &metadata,
            &[],
        )
        .expect("write must succeed");

        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");

        // Metadata fields.
        assert_eq!(checkpoint.metadata.completed_iterations, 10);
        assert_eq!(checkpoint.metadata.num_stages, 2);
        assert_eq!(checkpoint.metadata.state_dimension, 3);
        assert_eq!(checkpoint.metadata.rng_seed, 42);

        // Cuts: two stages, sorted by stage_id.
        assert_eq!(
            checkpoint.stage_cuts.len(),
            2,
            "must have two stage cut results"
        );
        assert_eq!(checkpoint.stage_cuts[0].stage_id, 0);
        assert_eq!(checkpoint.stage_cuts[1].stage_id, 1);
        assert_eq!(checkpoint.stage_cuts[0].cuts.len(), 2);
        assert_eq!(checkpoint.stage_cuts[1].cuts.len(), 1);

        // Stage 0 cut fields.
        let cut00 = &checkpoint.stage_cuts[0].cuts[0];
        assert_eq!(cut00.cut_id, 1);
        assert_eq!(cut00.coefficients, &[1.0f64, 2.0, 3.0]);
        assert_eq!(cut00.intercept, 42.0);
        assert!(cut00.is_active);

        let cut01 = &checkpoint.stage_cuts[0].cuts[1];
        assert_eq!(cut01.cut_id, 2);
        assert_eq!(cut01.coefficients, &[4.0f64, 5.0, 6.0]);

        // Stage 1 cut fields.
        let cut10 = &checkpoint.stage_cuts[1].cuts[0];
        assert_eq!(cut10.cut_id, 3);
        assert_eq!(cut10.coefficients, &[7.0f64, 8.0, 9.0]);

        // Bases: two stages, sorted by stage_id.
        assert_eq!(checkpoint.stage_bases.len(), 2, "must have two stage bases");
        assert_eq!(checkpoint.stage_bases[0].stage_id, 0);
        assert_eq!(checkpoint.stage_bases[1].stage_id, 1);
        assert_eq!(checkpoint.stage_bases[0].column_status, &[0u8, 1, 2, 3]);
        assert_eq!(checkpoint.stage_bases[0].row_status, &[1u8, 0, 1, 0, 1]);
        assert_eq!(checkpoint.stage_bases[0].num_cut_rows, 2);
    }

    #[test]
    fn read_policy_checkpoint_no_bases_empty_stage_bases() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0)];
        let stage_cuts_payloads = [make_stage_cuts_payload(0, &cuts_s0, &[0], 1)];
        let metadata = make_metadata(1, 1);

        write_policy_checkpoint(tmp.path(), &stage_cuts_payloads, &[], &metadata, &[])
            .expect("write must succeed");

        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");

        assert_eq!(checkpoint.stage_cuts.len(), 1);
        assert!(
            checkpoint.stage_bases.is_empty(),
            "no basis files must produce empty stage_bases"
        );
    }

    #[test]
    fn read_policy_checkpoint_missing_metadata_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        // Intentionally do NOT write metadata.json.
        let result = read_policy_checkpoint(tmp.path());
        assert!(
            result.is_err(),
            "missing metadata.json must return an error"
        );
        assert!(
            matches!(result, Err(OutputError::IoError { .. })),
            "error must be IoError for missing metadata.json"
        );
    }

    #[test]
    fn read_policy_checkpoint_stages_sorted_by_id() {
        let tmp = tempfile::tempdir().unwrap();

        // Write stages in non-ascending order — reader must sort.
        let c = [1.0_f64, 2.0];
        let cuts2 = [make_cut_record(1, 0, 1, &c)];
        let cuts0 = [make_cut_record(2, 0, 1, &c)];
        let cuts1 = [make_cut_record(3, 0, 1, &c)];

        let stage_cuts_payloads = [
            make_stage_cuts_payload(2, &cuts2, &[0], 2),
            make_stage_cuts_payload(0, &cuts0, &[0], 2),
            make_stage_cuts_payload(1, &cuts1, &[0], 2),
        ];
        let metadata = make_metadata(3, 2);

        write_policy_checkpoint(tmp.path(), &stage_cuts_payloads, &[], &metadata, &[])
            .expect("write must succeed");

        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");

        assert_eq!(checkpoint.stage_cuts.len(), 3);
        assert_eq!(
            checkpoint.stage_cuts[0].stage_id, 0,
            "first result must be stage 0"
        );
        assert_eq!(
            checkpoint.stage_cuts[1].stage_id, 1,
            "second result must be stage 1"
        );
        assert_eq!(
            checkpoint.stage_cuts[2].stage_id, 2,
            "third result must be stage 2"
        );
    }

    #[test]
    fn read_policy_checkpoint_metadata_json_field_by_field() {
        let tmp = tempfile::tempdir().unwrap();

        let meta_in = PolicyCheckpointMetadata {
            cobre_version: "0.0.1".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            completed_iterations: 77,
            final_lower_bound: 12345.678,
            best_upper_bound: Some(13000.0),
            state_dimension: 8,
            num_stages: 4,
            max_iterations: 500,
            forward_passes: 8,
            warm_start_cuts: 20,
            warm_start_counts: vec![20; 4],
            rng_seed: 99999,
            total_visited_states: 0,
        };

        let stage_cuts_payloads: [StageCutsPayload<'_>; 0] = [];
        write_policy_checkpoint(tmp.path(), &stage_cuts_payloads, &[], &meta_in, &[])
            .expect("write must succeed");

        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");
        let m = &checkpoint.metadata;

        assert_eq!(m.completed_iterations, 77);
        assert_eq!(m.final_lower_bound, 12345.678);
        assert_eq!(m.best_upper_bound, Some(13000.0));
        assert_eq!(m.state_dimension, 8);
        assert_eq!(m.num_stages, 4);
        assert_eq!(m.max_iterations, 500);
        assert_eq!(m.forward_passes, 8);
        assert_eq!(m.warm_start_cuts, 20);
        assert_eq!(m.warm_start_counts, vec![20u32; 4]);
        assert_eq!(m.rng_seed, 99999);
    }

    // ── warm_start_counts JSON serialization tests ────────────────────────────

    #[test]
    fn policy_checkpoint_metadata_warm_start_counts_round_trips() {
        let meta = PolicyCheckpointMetadata {
            cobre_version: "0.0.1".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            completed_iterations: 5,
            final_lower_bound: 0.0,
            best_upper_bound: None,
            state_dimension: 2,
            num_stages: 3,
            max_iterations: 10,
            forward_passes: 1,
            warm_start_cuts: 8,
            warm_start_counts: vec![10, 8, 6],
            rng_seed: 0,
            total_visited_states: 0,
        };

        let json = serde_json::to_string(&meta).expect("serialize must succeed");
        let back: PolicyCheckpointMetadata =
            serde_json::from_str(&json).expect("deserialize must succeed");

        assert_eq!(
            back.warm_start_counts,
            vec![10u32, 8, 6],
            "warm_start_counts must round-trip"
        );
    }

    #[test]
    fn policy_checkpoint_metadata_warm_start_counts_absent_defaults_to_empty() {
        let json = r#"{
            "cobre_version": "0.0.1",
            "created_at": "2026-01-01T00:00:00Z",
            "completed_iterations": 5,
            "final_lower_bound": 0.0,
            "best_upper_bound": null,
            "state_dimension": 2,
            "num_stages": 3,
            "max_iterations": 10,
            "forward_passes": 1,
            "warm_start_cuts": 5,
            "rng_seed": 0,
            "total_visited_states": 0
        }"#;

        let meta: PolicyCheckpointMetadata =
            serde_json::from_str(json).expect("old-format JSON must deserialize");

        assert!(
            meta.warm_start_counts.is_empty(),
            "absent warm_start_counts must default to empty vec"
        );
        assert_eq!(
            meta.warm_start_cuts, 5,
            "warm_start_cuts scalar must still be read"
        );
    }

    #[test]
    fn read_policy_checkpoint_warm_start_counts_in_metadata() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64, 2.0];
        let c1 = [3.0_f64, 4.0];
        let c2 = [5.0_f64, 6.0];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0), make_cut_record(2, 1, 1, &c0)];
        let cuts_s1 = [
            make_cut_record(3, 0, 2, &c1),
            make_cut_record(4, 1, 2, &c1),
            make_cut_record(5, 2, 2, &c1),
        ];
        let cuts_s2 = [
            make_cut_record(6, 0, 3, &c2),
            make_cut_record(7, 1, 3, &c2),
            make_cut_record(8, 2, 3, &c2),
            make_cut_record(9, 3, 3, &c2),
        ];

        let stage_cuts_payloads = [
            StageCutsPayload {
                stage_id: 0,
                state_dimension: 2,
                capacity: 100,
                warm_start_count: 10,
                cuts: &cuts_s0,
                active_cut_indices: &[0, 1],
                populated_count: 2,
            },
            StageCutsPayload {
                stage_id: 1,
                state_dimension: 2,
                capacity: 100,
                warm_start_count: 8,
                cuts: &cuts_s1,
                active_cut_indices: &[0, 1, 2],
                populated_count: 3,
            },
            StageCutsPayload {
                stage_id: 2,
                state_dimension: 2,
                capacity: 100,
                warm_start_count: 6,
                cuts: &cuts_s2,
                active_cut_indices: &[0, 1, 2, 3],
                populated_count: 4,
            },
        ];

        let metadata = PolicyCheckpointMetadata {
            cobre_version: "0.0.1".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            completed_iterations: 10,
            final_lower_bound: 100.0,
            best_upper_bound: None,
            state_dimension: 2,
            num_stages: 3,
            max_iterations: 50,
            forward_passes: 1,
            warm_start_cuts: 10,
            warm_start_counts: vec![10, 8, 6],
            rng_seed: 0,
            total_visited_states: 0,
        };

        write_policy_checkpoint(tmp.path(), &stage_cuts_payloads, &[], &metadata, &[])
            .expect("write must succeed");

        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");

        assert_eq!(
            checkpoint.metadata.warm_start_counts,
            vec![10u32, 8, 6],
            "warm_start_counts [10, 8, 6] must round-trip through metadata.json"
        );
    }
}
