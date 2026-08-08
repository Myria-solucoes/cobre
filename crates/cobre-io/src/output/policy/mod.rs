//! `FlatBuffers` builder and reader types for policy checkpoint serialization.
//!
//! Types use generic names to maintain infrastructure crate genericity; conversion
//! from algorithm-specific types is the calling crate's responsibility.
//!
//! The canonical wire-format description is `schemas/policy.fbs` in this crate
//! (namespace `Cobre.IO.Policy`, tables `StageCuts`, `AffinePiece`, `StageBasis`,
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
    ENTITY_SLOT_DELIVERY_DATE_SENTINEL, EntitySlot, FORMAT_VERSION, GraphManifest, ManifestEdge,
    ManifestNode, OwnedPolicyBasisRecord, OwnedPolicyCutRecord, PolicyBasisRecord,
    PolicyCheckpoint, PolicyCheckpointMetadata, PolicyCutRecord, ProducerBlock,
    STAGE_STATES_NODE_ID_SENTINEL, StageCutsPayload, StageCutsReadResult, StageStatesPayload,
    StageStatesReadResult,
};

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::unreadable_literal
)]
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

        let buf = serialize_stage_cuts(0, 3, 100, 0, &[cut], &[0], 1, &[]);

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
        let buf = serialize_stage_cuts(0, 3, 100, 0, &[], &[], 0, &[]);

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

        let buf_a = serialize_stage_cuts(5, 3, 50, 0, &cuts, &[0, 1, 2], 3, &[]);
        let buf_b = serialize_stage_cuts(5, 3, 50, 0, &cuts, &[0, 1, 2], 3, &[]);

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
            let buf = serialize_stage_cuts(0, dim, 10, 0, &[cut], &[0], 1, &[]);
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
        let mut meta = make_metadata(60, 160);
        meta.producer.completed_iterations = 50;
        meta.producer.final_lower_bound = 1234.56;
        meta.producer.best_upper_bound = Some(1300.0);
        meta.producer.max_iterations = 200;
        meta.producer.warm_start_counts = vec![];

        let json = serde_json::to_string_pretty(&meta)
            .expect("PolicyCheckpointMetadata must serialize to JSON without error");

        assert!(json.contains("producer"), "JSON must contain 'producer'");
        assert!(
            json.contains("format_version"),
            "JSON must contain 'format_version'"
        );
        assert!(
            json.contains("completed_iterations"),
            "JSON must contain 'completed_iterations'"
        );
        assert!(
            json.contains("final_lower_bound"),
            "JSON must contain 'final_lower_bound'"
        );
        assert!(json.contains("rng_seed"), "JSON must contain 'rng_seed'");
        assert!(
            json.contains("best_upper_bound"),
            "JSON must contain 'best_upper_bound'"
        );

        // Verify it round-trips through serde_json::Value; producer keys nest.
        let value: serde_json::Value =
            serde_json::from_str(&json).expect("JSON output must be parseable");
        assert_eq!(
            value["format_version"].as_u64(),
            Some(u64::from(FORMAT_VERSION)),
            "format_version must be the current marker"
        );
        assert_eq!(
            value["producer"]["completed_iterations"].as_u64(),
            Some(50),
            "completed_iterations must deserialize correctly under producer"
        );
        assert_eq!(
            value["producer"]["rng_seed"].as_u64(),
            Some(42),
            "rng_seed must deserialize correctly under producer"
        );
    }

    #[test]
    fn policy_checkpoint_metadata_none_upper_bound_serializes_to_null() {
        let mut meta = make_metadata(1, 1);
        meta.producer.best_upper_bound = None;

        let json = serde_json::to_string_pretty(&meta)
            .expect("PolicyCheckpointMetadata must serialize to JSON");

        let value: serde_json::Value =
            serde_json::from_str(&json).expect("JSON output must be parseable");
        assert!(
            value["producer"]["best_upper_bound"].is_null(),
            "best_upper_bound must serialize to null when None"
        );
    }

    // ── write_policy_checkpoint tests ─────────────────────────────────────────

    /// A trivial 1:1 chain graph manifest over `num_stages` nodes (node id ==
    /// stage id == pool id) — the shape a chain-degenerate study writes.
    fn chain_manifest(num_stages: u32) -> GraphManifest {
        let nodes = (0..num_stages)
            .map(|t| ManifestNode {
                id: i32::try_from(t).unwrap(),
                stage_id: i32::try_from(t).unwrap(),
                pool_id: t,
            })
            .collect();
        let edges = (0..num_stages.saturating_sub(1))
            .map(|t| ManifestEdge {
                source_id: i32::try_from(t).unwrap(),
                target_id: i32::try_from(t + 1).unwrap(),
                probability: 1.0,
            })
            .collect();
        GraphManifest {
            n_pools: num_stages,
            nodes,
            edges,
        }
    }

    /// Build a minimal [`PolicyCheckpointMetadata`] for use in artifact tests.
    /// `state_dimension` is retained for call-site clarity; it now lives per-pool
    /// on the payloads, not in the metadata core.
    fn make_metadata(num_stages: u32, _state_dimension: u32) -> PolicyCheckpointMetadata {
        PolicyCheckpointMetadata {
            format_version: FORMAT_VERSION,
            cobre_version: "0.0.1".to_string(),
            created_at: "2026-03-08T00:00:00Z".to_string(),
            num_stages,
            graph_manifest: chain_manifest(num_stages),
            producer: ProducerBlock {
                completed_iterations: 10,
                final_lower_bound: 999.0,
                best_upper_bound: Some(1050.0),
                max_iterations: 100,
                forward_passes: 4,
                warm_start_cuts: 0,
                warm_start_counts: vec![0; num_stages as usize],
                rng_seed: 42,
                total_visited_states: 0,
                training_block_mode: "parallel".to_string(),
                training_block_mode_per_stage: vec![],
                cost_scale_factor: None,
            },
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
            entity_manifest: &[],
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

        assert!(tmp.path().join("cuts").is_dir(), "cuts/ must exist");
        assert!(tmp.path().join("basis").is_dir(), "basis/ must exist");

        for i in 0..3u32 {
            let p = tmp.path().join(format!("cuts/{i:03}.bin"));
            assert!(p.is_file(), "cuts/{i:03}.bin must exist");
        }

        for i in 0..3u32 {
            let p = tmp.path().join(format!("basis/{i:03}.bin"));
            assert!(p.is_file(), "basis/{i:03}.bin must exist");
        }

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
            "format_version",
            "cobre_version",
            "created_at",
            "num_stages",
            "graph_manifest",
            "producer",
        ] {
            assert!(
                value.get(key).is_some(),
                "metadata.json must contain key '{key}'"
            );
        }

        assert_eq!(
            value["producer"]["completed_iterations"].as_u64(),
            Some(10),
            "completed_iterations must match under producer"
        );
        assert_eq!(
            value["num_stages"].as_u64(),
            Some(1),
            "num_stages must match"
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
            let p = tmp.path().join(format!("cuts/{i:03}.bin"));
            let bytes = std::fs::read(&p).unwrap();
            assert!(!bytes.is_empty(), "cuts/{i:03}.bin must not be empty");
            assert!(bytes.len() >= 4, "cuts/{i:03}.bin must have >= 4 bytes");
            let root_offset = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
            assert!(
                root_offset < bytes.len(),
                "cuts/{i:03}.bin root offset must be in-range"
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

        let p = tmp.path().join("basis/000.bin");
        let bytes = std::fs::read(&p).unwrap();
        assert!(!bytes.is_empty(), "basis/000.bin must not be empty");
        assert!(bytes.len() >= 4, "basis/000.bin must have >= 4 bytes");
        let root_offset = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        assert!(
            root_offset < bytes.len(),
            "basis/000.bin root offset must be in-range"
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

        assert!(
            tmp.path().join("basis").is_dir(),
            "basis/ directory must exist even with empty stage_bases"
        );

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
            tmp.path().join("cuts/000.bin").is_file(),
            "cuts/000.bin must exist"
        );
        assert!(
            tmp.path().join("cuts/001.bin").is_file(),
            "cuts/001.bin must exist"
        );
        assert!(
            tmp.path().join("cuts/059.bin").is_file(),
            "cuts/059.bin must exist"
        );
        assert!(
            tmp.path().join("basis/000.bin").is_file(),
            "basis/000.bin must exist"
        );
        assert!(
            tmp.path().join("basis/001.bin").is_file(),
            "basis/001.bin must exist"
        );
        assert!(
            tmp.path().join("basis/059.bin").is_file(),
            "basis/059.bin must exist"
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

        let buf = serialize_stage_cuts(0, 3, 100, 0, &[cut], &[0], 1, &[]);
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

        let buf = serialize_stage_cuts(5, 2, 50, 1, &cuts, &[0, 2], 3, &[]);
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
        let buf = serialize_stage_cuts(2, 10, 200, 0, &[], &[], 0, &[]);
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
        let buf = serialize_stage_cuts(0, 0, 10, 0, &[cut], &[0], 1, &[]);
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
        let buf = serialize_stage_cuts(3, dim, 10, 0, &[cut], &[0], 1, &[]);
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
        let full_buf = serialize_stage_cuts(0, 2, 10, 0, &[cut], &[0], 1, &[]);
        // Truncate to 2 bytes — root offset itself is incomplete.
        let truncated = &full_buf[..2];
        let result = deserialize_stage_cuts(truncated);
        assert!(result.is_err(), "truncated buffer must return an error");
    }

    #[test]
    fn deserialize_stage_cuts_stage_id_nonzero() {
        let buf = serialize_stage_cuts(59, 4, 50, 0, &[], &[], 0, &[]);
        let result = deserialize_stage_cuts(&buf).expect("stage_id=59 must deserialize");
        assert_eq!(result.stage_id, 59, "stage_id=59 must round-trip");
    }

    // ── entity_manifest round-trip tests ──────────────────────────────────────

    fn sample_manifest() -> Vec<EntitySlot> {
        vec![
            EntitySlot {
                entity_type: 0,
                entity_id: 12,
                subindex: 0,
                was_active: true,
                delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
            },
            EntitySlot {
                entity_type: 1,
                entity_id: -1,
                subindex: 3,
                was_active: false,
                delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
            },
            EntitySlot {
                entity_type: 2,
                entity_id: 7,
                subindex: 1,
                was_active: true,
                delivery_date: 20240501,
            },
        ]
    }

    fn assert_manifest_eq(actual: &[EntitySlot], expected: &[EntitySlot]) {
        assert_eq!(actual.len(), expected.len(), "manifest length must match");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(a.entity_type, e.entity_type, "slot {i} entity_type");
            assert_eq!(a.entity_id, e.entity_id, "slot {i} entity_id");
            assert_eq!(a.subindex, e.subindex, "slot {i} subindex");
            assert_eq!(a.was_active, e.was_active, "slot {i} was_active");
            assert_eq!(a.delivery_date, e.delivery_date, "slot {i} delivery_date");
        }
    }

    #[test]
    fn stage_cuts_entity_manifest_round_trip() {
        let coefficients = [1.0_f64, 2.0, 3.0];
        let cut = make_cut_record(7, 0, 1, &coefficients);
        let manifest = sample_manifest();

        let buf = serialize_stage_cuts(4, 3, 100, 0, &[cut], &[0], 1, &manifest);
        let result = deserialize_stage_cuts(&buf).expect("manifest round-trip must succeed");

        assert_eq!(result.cuts.len(), 1, "cuts must still round-trip");
        assert_manifest_eq(&result.entity_manifest, &manifest);
    }

    #[test]
    fn stage_states_entity_manifest_round_trip() {
        let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let manifest = sample_manifest();
        let payload = StageStatesPayload {
            stage_id: 2,
            node_id: 9,
            state_dimension: 3,
            count: 2,
            data: &data,
            entity_manifest: &manifest,
        };

        let buf = serialize_stage_states(&payload);
        let result = deserialize_stage_states(&buf).expect("manifest round-trip must succeed");

        assert_eq!(result.data, data.to_vec(), "data must still round-trip");
        assert_eq!(result.node_id, 9, "node_id must round-trip");
        assert_manifest_eq(&result.entity_manifest, &manifest);
    }

    #[test]
    fn stage_cuts_empty_manifest_deserializes_to_empty_vec() {
        let coefficients = [1.0_f64, 2.0];
        let cut = make_cut_record(1, 0, 1, &coefficients);
        let buf = serialize_stage_cuts(0, 2, 10, 0, &[cut], &[0], 1, &[]);
        let result = deserialize_stage_cuts(&buf).expect("empty manifest must deserialize");
        assert!(
            result.entity_manifest.is_empty(),
            "empty manifest must produce zero slots"
        );
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
        let mut meta = make_metadata(3, 5);
        meta.producer.completed_iterations = 42;
        meta.producer.final_lower_bound = 9999.0;
        meta.producer.best_upper_bound = Some(10100.0);
        meta.producer.warm_start_cuts = 10;
        meta.producer.warm_start_counts = vec![10, 10, 10];
        meta.producer.rng_seed = 12345;

        let json = serde_json::to_string(&meta).expect("serialize must succeed");
        let back: PolicyCheckpointMetadata =
            serde_json::from_str(&json).expect("deserialize must succeed");

        assert_eq!(back.format_version, meta.format_version);
        assert_eq!(back.cobre_version, meta.cobre_version);
        assert_eq!(
            back.producer.completed_iterations,
            meta.producer.completed_iterations
        );
        assert_eq!(
            back.producer.final_lower_bound,
            meta.producer.final_lower_bound
        );
        assert_eq!(
            back.producer.best_upper_bound,
            meta.producer.best_upper_bound
        );
        assert_eq!(back.num_stages, meta.num_stages);
        assert_eq!(back.producer.max_iterations, meta.producer.max_iterations);
        assert_eq!(back.producer.forward_passes, meta.producer.forward_passes);
        assert_eq!(back.producer.warm_start_cuts, meta.producer.warm_start_cuts);
        assert_eq!(back.producer.rng_seed, meta.producer.rng_seed);
    }

    #[test]
    fn policy_checkpoint_metadata_deserializes_none_upper_bound() {
        let mut meta = make_metadata(1, 1);
        meta.producer.best_upper_bound = None;

        let json = serde_json::to_string(&meta).expect("serialize must succeed");
        let back: PolicyCheckpointMetadata =
            serde_json::from_str(&json).expect("deserialize must succeed");

        assert!(
            back.producer.best_upper_bound.is_none(),
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

        assert_eq!(checkpoint.metadata.producer.completed_iterations, 10);
        assert_eq!(checkpoint.metadata.num_stages, 2);
        assert_eq!(checkpoint.metadata.producer.rng_seed, 42);

        assert_eq!(
            checkpoint.stage_cuts.len(),
            2,
            "must have two stage cut results"
        );
        assert_eq!(checkpoint.stage_cuts[0].stage_id, 0);
        assert_eq!(checkpoint.stage_cuts[1].stage_id, 1);
        assert_eq!(checkpoint.stage_cuts[0].cuts.len(), 2);
        assert_eq!(checkpoint.stage_cuts[1].cuts.len(), 1);

        let cut00 = &checkpoint.stage_cuts[0].cuts[0];
        assert_eq!(cut00.cut_id, 1);
        assert_eq!(cut00.coefficients, &[1.0f64, 2.0, 3.0]);
        assert_eq!(cut00.intercept, 42.0);
        assert!(cut00.is_active);

        let cut01 = &checkpoint.stage_cuts[0].cuts[1];
        assert_eq!(cut01.cut_id, 2);
        assert_eq!(cut01.coefficients, &[4.0f64, 5.0, 6.0]);

        let cut10 = &checkpoint.stage_cuts[1].cuts[0];
        assert_eq!(cut10.cut_id, 3);
        assert_eq!(cut10.coefficients, &[7.0f64, 8.0, 9.0]);

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

        let mut meta_in = make_metadata(4, 8);
        meta_in.created_at = "2026-01-01T00:00:00Z".to_string();
        meta_in.producer.completed_iterations = 77;
        meta_in.producer.final_lower_bound = 12345.678;
        meta_in.producer.best_upper_bound = Some(13000.0);
        meta_in.producer.max_iterations = 500;
        meta_in.producer.forward_passes = 8;
        meta_in.producer.warm_start_cuts = 20;
        meta_in.producer.warm_start_counts = vec![20; 4];
        meta_in.producer.rng_seed = 99999;

        let stage_cuts_payloads: [StageCutsPayload<'_>; 0] = [];
        write_policy_checkpoint(tmp.path(), &stage_cuts_payloads, &[], &meta_in, &[])
            .expect("write must succeed");

        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");
        let m = &checkpoint.metadata;

        assert_eq!(m.format_version, FORMAT_VERSION);
        assert_eq!(m.producer.completed_iterations, 77);
        assert_eq!(m.producer.final_lower_bound, 12345.678);
        assert_eq!(m.producer.best_upper_bound, Some(13000.0));
        assert_eq!(m.num_stages, 4);
        assert_eq!(m.producer.max_iterations, 500);
        assert_eq!(m.producer.forward_passes, 8);
        assert_eq!(m.producer.warm_start_cuts, 20);
        assert_eq!(m.producer.warm_start_counts, vec![20u32; 4]);
        assert_eq!(m.producer.rng_seed, 99999);
    }

    // ── producer-block JSON serialization tests ───────────────────────────────

    #[test]
    fn policy_checkpoint_metadata_warm_start_counts_round_trips() {
        let mut meta = make_metadata(3, 2);
        meta.producer.warm_start_cuts = 8;
        meta.producer.warm_start_counts = vec![10, 8, 6];

        let json = serde_json::to_string(&meta).expect("serialize must succeed");
        let back: PolicyCheckpointMetadata =
            serde_json::from_str(&json).expect("deserialize must succeed");

        assert_eq!(
            back.producer.warm_start_counts,
            vec![10u32, 8, 6],
            "warm_start_counts must round-trip under producer"
        );
    }

    /// The optional producer sub-fields default when absent from the JSON: a
    /// metadata carrying a `producer` block without `warm_start_counts` /
    /// `training_block_mode` / `cost_scale_factor` reads them as empty / None.
    #[test]
    fn policy_checkpoint_metadata_absent_producer_optionals_default() {
        let json = r#"{
            "format_version": 1,
            "cobre_version": "0.0.1",
            "created_at": "2026-01-01T00:00:00Z",
            "num_stages": 3,
            "producer": {
                "completed_iterations": 5,
                "final_lower_bound": 0.0,
                "best_upper_bound": null,
                "max_iterations": 10,
                "forward_passes": 1,
                "warm_start_cuts": 5,
                "rng_seed": 0
            }
        }"#;

        let meta: PolicyCheckpointMetadata =
            serde_json::from_str(json).expect("minimal-producer JSON must deserialize");

        assert!(
            meta.producer.warm_start_counts.is_empty(),
            "absent warm_start_counts must default to empty vec"
        );
        assert_eq!(
            meta.producer.warm_start_cuts, 5,
            "warm_start_cuts scalar must still be read"
        );
        assert!(
            meta.producer.training_block_mode.is_empty(),
            "absent training_block_mode must default to empty string"
        );
        assert!(
            meta.producer.training_block_mode_per_stage.is_empty(),
            "absent training_block_mode_per_stage must default to empty vec"
        );
        assert!(
            meta.producer.cost_scale_factor.is_none(),
            "absent cost_scale_factor must default to None (legacy interpretation)"
        );
        assert!(
            meta.graph_manifest.nodes.is_empty(),
            "absent graph_manifest must default to empty"
        );
    }

    #[test]
    fn policy_checkpoint_metadata_training_block_mode_round_trips_mixed() {
        let mut meta = make_metadata(3, 2);
        meta.producer.training_block_mode = "mixed".to_string();
        meta.producer.training_block_mode_per_stage = vec![
            "parallel".to_string(),
            "chronological".to_string(),
            "parallel".to_string(),
        ];

        let json = serde_json::to_string(&meta).expect("serialize must succeed");
        let back: PolicyCheckpointMetadata =
            serde_json::from_str(&json).expect("deserialize must succeed");

        assert_eq!(back.producer.training_block_mode, "mixed");
        assert_eq!(
            back.producer.training_block_mode_per_stage,
            vec!["parallel", "chronological", "parallel"]
        );
    }

    /// A written (marked) `cost_scale_factor` round-trips through JSON exactly.
    #[test]
    fn policy_checkpoint_metadata_cost_scale_factor_round_trips() {
        let mut meta = make_metadata(3, 2);
        meta.producer.cost_scale_factor = Some(2_500_000.0);

        let json = serde_json::to_string(&meta).expect("serialize must succeed");
        let back: PolicyCheckpointMetadata =
            serde_json::from_str(&json).expect("deserialize must succeed");

        assert_eq!(back.producer.cost_scale_factor, Some(2_500_000.0));
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
                entity_manifest: &[],
            },
            StageCutsPayload {
                stage_id: 1,
                state_dimension: 2,
                capacity: 100,
                warm_start_count: 8,
                cuts: &cuts_s1,
                active_cut_indices: &[0, 1, 2],
                populated_count: 3,
                entity_manifest: &[],
            },
            StageCutsPayload {
                stage_id: 2,
                state_dimension: 2,
                capacity: 100,
                warm_start_count: 6,
                cuts: &cuts_s2,
                active_cut_indices: &[0, 1, 2, 3],
                populated_count: 4,
                entity_manifest: &[],
            },
        ];

        let mut metadata = make_metadata(3, 2);
        metadata.producer.warm_start_counts = vec![10, 8, 6];

        write_policy_checkpoint(tmp.path(), &stage_cuts_payloads, &[], &metadata, &[])
            .expect("write must succeed");

        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");

        assert_eq!(
            checkpoint.metadata.producer.warm_start_counts,
            vec![10u32, 8, 6],
            "warm_start_counts [10, 8, 6] must round-trip through metadata.json"
        );
    }

    // ── 0.14 format-wave markers ──────────────────────────────────────────────

    /// A 0.13-era artifact (metadata.json without a `format_version` key) is
    /// rejected on the version marker BEFORE any payload is parsed: the reader
    /// names `format_version`, not the deliberately-corrupt `cuts/000.bin`.
    #[test]
    fn read_policy_checkpoint_rejects_absent_format_version_before_parsing_payloads() {
        let tmp = tempfile::tempdir().unwrap();
        let c0 = [1.0_f64];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0)];
        let stage_cuts = [make_stage_cuts_payload(0, &cuts_s0, &[0], 1)];
        write_policy_checkpoint(tmp.path(), &stage_cuts, &[], &make_metadata(1, 1), &[])
            .expect("write must succeed");

        // Strip the format_version key (a pre-marker artifact) and corrupt the payload.
        let meta_path = tmp.path().join("metadata.json");
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
        let mut obj = value.as_object().unwrap().clone();
        obj.remove("format_version");
        std::fs::write(
            &meta_path,
            serde_json::to_vec(&serde_json::Value::Object(obj)).unwrap(),
        )
        .unwrap();
        std::fs::write(tmp.path().join("cuts/000.bin"), b"garbage").unwrap();

        let err =
            read_policy_checkpoint(tmp.path()).expect_err("absent format_version must reject");
        let msg = err.to_string();
        assert!(
            msg.contains("format_version"),
            "must reject on the version marker, not the corrupt payload: {msg}"
        );
    }

    /// A genuine 0.13-era metadata.json — flat, no `producer`, no
    /// `format_version` — is rejected by the marker (naming `format_version`),
    /// not by an opaque missing-`producer` parse error: the marker check runs
    /// off a minimal probe before full deserialization.
    #[test]
    fn read_policy_checkpoint_rejects_flat_legacy_metadata_on_the_marker() {
        let tmp = tempfile::tempdir().unwrap();
        let legacy = r#"{
            "cobre_version": "0.13.0",
            "created_at": "2026-01-01T00:00:00Z",
            "completed_iterations": 5,
            "final_lower_bound": 0.0,
            "state_dimension": 2,
            "num_stages": 3
        }"#;
        std::fs::write(tmp.path().join("metadata.json"), legacy).unwrap();

        let err = read_policy_checkpoint(tmp.path()).expect_err("a flat 0.13 artifact must reject");
        let msg = err.to_string();
        assert!(
            msg.contains("format_version"),
            "must reject on the version marker, not a missing-field error: {msg}"
        );
    }

    /// Every top-level metadata key outside the namespaced `producer` block is a
    /// neutral-core key (format marker, provenance, or graph/stage descriptor).
    #[test]
    fn metadata_neutral_core_keys_are_format_provenance_and_graph_only() {
        let value = serde_json::to_value(make_metadata(2, 3)).expect("serialize must succeed");
        let obj = value.as_object().expect("metadata serializes to an object");
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "cobre_version",
                "created_at",
                "format_version",
                "graph_manifest",
                "num_stages",
                "producer",
            ],
            "every top-level key outside `producer` must be a neutral-core key"
        );
    }

    /// The positional `stage_NNN` filename convention is gone: no
    /// `format!("stage_{...}.bin", ...)` writer literal survives, and the reader
    /// derives identity from inside each buffer via `read_sorted_bin_files`, not
    /// from a filename. A surviving positional reader is the exact path that
    /// would read a 0.13 file as pool 0.
    #[test]
    fn no_stage_positional_filename_convention_remains() {
        let checkpoint_src = include_str!("checkpoint.rs");
        assert!(
            !checkpoint_src.contains("stage_{"),
            "no `format!(\"stage_{{...}}.bin\", ...)` writer literal may remain"
        );
        assert!(
            checkpoint_src.contains("read_sorted_bin_files"),
            "the reader must stay filename-agnostic (sorts by the in-buffer id)"
        );
    }

    /// K-fan (K = 4) whose leaves share one pool: the writer emits ONE payload
    /// for the shared leaf pool (not K), and the graph manifest maps all K leaf
    /// node ids to that one pool id.
    #[test]
    fn k_fan_shared_leaf_pool_serialized_once_and_manifest_maps_leaves() {
        let tmp = tempfile::tempdir().unwrap();
        let k = 4u32;

        // One payload per pool: pool 0 (root, stage 0) and pool 1 (shared leaf pool).
        let c = [1.0_f64];
        let cuts_p0 = [make_cut_record(1, 0, 1, &c)];
        let cuts_p1 = [make_cut_record(2, 0, 1, &c)];
        let stage_cuts = [
            make_stage_cuts_payload(0, &cuts_p0, &[0], 1),
            make_stage_cuts_payload(1, &cuts_p1, &[0], 1),
        ];

        let mut nodes = vec![ManifestNode {
            id: 0,
            stage_id: 0,
            pool_id: 0,
        }];
        for leaf in 1..=k {
            nodes.push(ManifestNode {
                id: i32::try_from(leaf).unwrap(),
                stage_id: 1,
                pool_id: 1,
            });
        }
        let edges = (1..=k)
            .map(|leaf| ManifestEdge {
                source_id: 0,
                target_id: i32::try_from(leaf).unwrap(),
                probability: 1.0 / f64::from(k),
            })
            .collect();
        let mut metadata = make_metadata(2, 1);
        metadata.graph_manifest = GraphManifest {
            n_pools: 2,
            nodes,
            edges,
        };

        write_policy_checkpoint(tmp.path(), &stage_cuts, &[], &metadata, &[])
            .expect("write must succeed");

        let bin_count = std::fs::read_dir(tmp.path().join("cuts"))
            .unwrap()
            .filter_map(std::result::Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "bin"))
            .count();
        assert_eq!(
            bin_count, 2,
            "the shared leaf pool is serialized once (2 pool files), not K times"
        );

        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");
        let leaf_pools: std::collections::BTreeSet<u32> = checkpoint
            .metadata
            .graph_manifest
            .nodes
            .iter()
            .filter(|n| n.stage_id == 1)
            .map(|n| n.pool_id)
            .collect();
        assert_eq!(
            leaf_pools.len(),
            1,
            "all K leaf node ids must map to one pool id: {leaf_pools:?}"
        );
        assert_eq!(*leaf_pools.iter().next().unwrap(), 1);
        assert_eq!(
            checkpoint
                .metadata
                .graph_manifest
                .nodes
                .iter()
                .filter(|n| n.stage_id == 1)
                .count(),
            k as usize,
            "all K leaf nodes must be present in the manifest"
        );
    }
}
