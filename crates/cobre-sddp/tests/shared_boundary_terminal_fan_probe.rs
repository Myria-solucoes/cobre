//! Guard test: a single-node boundary-policy source loads into a fanned
//! terminal target without rejecting on node count or graph identity, and
//! `inject_boundary_cuts` populates the ONE pool every terminal fan leaf's
//! `NodeRuntime::pool_id` resolves to — never a per-leaf distinct policy. A
//! multi-node source is still rejected.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use std::path::Path;

use cobre_io::{
    FORMAT_VERSION, GraphManifest, ManifestNode, PolicyCheckpointMetadata, PolicyCutRecord,
    ProducerBlock, StageCutsPayload, write_policy_checkpoint,
};
use cobre_sddp::setup::{NodeGraph, NodePos};
use cobre_sddp::test_support::k_fan_setup;
use cobre_sddp::{LEGACY_COST_SCALE_FACTOR, inject_boundary_cuts, load_boundary_cuts};

/// Write a synthetic single-pool policy checkpoint whose graph manifest
/// declares `graph_nodes` as `(node_id, stage_id, pool_id)` triples —
/// `load_boundary_cuts` resolves `source_stage` through this manifest, never
/// by a stage/pool coincidence. Carries no entity manifest (the identity
/// check is out of this probe's scope; covered by the boundary-injection
/// manifest-mismatch tests elsewhere).
fn write_synthetic_checkpoint(
    dir: &Path,
    graph_nodes: &[(i32, i32, u32)],
    pool_id: u32,
    intercepts: &[f64],
    state_dimension: u32,
) {
    let coefficients = vec![1.0_f64; state_dimension as usize];
    let cuts: Vec<PolicyCutRecord<'_>> = intercepts
        .iter()
        .enumerate()
        .map(|(i, &intercept)| PolicyCutRecord {
            cut_id: i as u64,
            slot_index: i as u32,
            iteration: 0,
            forward_pass_index: 0,
            intercept,
            coefficients: &coefficients,
            is_active: true,
        })
        .collect();
    let active_cut_indices: Vec<u32> = (0..intercepts.len() as u32).collect();
    let payload = StageCutsPayload {
        stage_id: pool_id,
        state_dimension,
        capacity: intercepts.len() as u32,
        warm_start_count: 0,
        cuts: &cuts,
        active_cut_indices: &active_cut_indices,
        populated_count: intercepts.len() as u32,
        entity_manifest: &[],
    };
    let nodes: Vec<ManifestNode> = graph_nodes
        .iter()
        .map(|&(id, stage_id, pool_id)| ManifestNode {
            id,
            stage_id,
            pool_id,
        })
        .collect();
    let n_pools = graph_nodes
        .iter()
        .map(|&(_, _, p)| p + 1)
        .max()
        .unwrap_or(0);
    let metadata = PolicyCheckpointMetadata {
        format_version: FORMAT_VERSION,
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: "2026-08-10T00:00:00Z".to_string(),
        num_stages: 1,
        graph_manifest: GraphManifest {
            n_pools,
            nodes,
            edges: vec![],
        },
        producer: ProducerBlock {
            completed_iterations: 0,
            final_lower_bound: 0.0,
            best_upper_bound: None,
            max_iterations: 0,
            forward_passes: 0,
            warm_start_cuts: 0,
            warm_start_counts: vec![],
            rng_seed: 0,
            total_visited_states: 0,
            training_block_mode: "parallel".to_string(),
            training_block_mode_per_stage: vec![],
            cost_scale_factor: None,
        },
    };
    write_policy_checkpoint(dir, &[payload], &[], &metadata, &[]).expect("write checkpoint");
}

/// Every leaf `NodePos` in `graph` — a node with no successors, the terminal
/// fan whose pool `inject_boundary_cuts` writes.
fn leaf_positions(graph: &NodeGraph) -> Vec<NodePos> {
    graph
        .nodes
        .iter_indexed()
        .filter(|&(pos, _)| graph.successors[pos].is_empty())
        .map(|(pos, _)| pos)
        .collect()
}

#[test]
fn single_node_source_injects_into_the_one_pool_every_terminal_fan_leaf_shares() {
    let fixture = k_fan_setup(3, 2, 5);
    let mut setup = fixture.setup;

    let leaves = leaf_positions(&setup.node_graph);
    assert_eq!(
        leaves.len(),
        fixture.k,
        "the fixture must declare a genuine multi-node terminal fan"
    );
    assert!(
        setup.node_graph.nodes.len() > setup.node_graph.n_pools,
        "the fan must have more nodes than pools, or leaf-pool-sharing has no work to do"
    );

    let terminal_idx = setup.fcf.pools.len() - 1;
    let leaf_pool_ids: Vec<usize> = leaves
        .iter()
        .map(|&pos| setup.node_graph.nodes[pos].pool_id)
        .collect();
    assert!(
        leaf_pool_ids.iter().all(|&p| p == terminal_idx),
        "every terminal fan leaf must resolve to the same pool inject_boundary_cuts writes: \
         {leaf_pool_ids:?} != {terminal_idx}"
    );

    let state_dimension = setup.fcf.state_dimension as u32;
    let tmp = tempfile::tempdir().expect("tempdir");
    let source_dir = tmp.path().join("single_node_source");
    // One declared node (id 100) at stage 0 -> pool 0: a single-node source,
    // structurally unrelated to the fanned target's own (larger) graph shape.
    write_synthetic_checkpoint(
        &source_dir,
        &[(100, 0, 0)],
        0,
        &[7.0, 11.0],
        state_dimension,
    );

    let mut warnings: Vec<String> = Vec::new();
    let boundary_cuts = load_boundary_cuts(
        &source_dir,
        0,
        state_dimension,
        &[],
        &[],
        None,
        LEGACY_COST_SCALE_FACTOR,
        &mut |msg| warnings.push(msg.to_string()),
    )
    .expect(
        "a single-node source boundary must load into a fanned terminal target, not reject on \
         node-count or graph identity",
    );
    assert_eq!(boundary_cuts.len(), 2);

    let n_pools_before = setup.fcf.pools.len();
    inject_boundary_cuts(&mut setup, &boundary_cuts);
    assert_eq!(
        setup.fcf.pools.len(),
        n_pools_before,
        "injection must not fan out into per-leaf pools"
    );

    for &pool_id in &leaf_pool_ids {
        let pool = &setup.fcf.pools[pool_id];
        assert_eq!(
            pool.warm_start_count as usize,
            boundary_cuts.len(),
            "every terminal fan leaf's own pool must carry the injected boundary cuts"
        );
        let loaded: Vec<(f64, Vec<f64>)> = pool
            .active_cuts()
            .map(|(_, intercept, coeffs)| (intercept, coeffs.to_vec()))
            .collect();
        let expected: Vec<(f64, Vec<f64>)> = boundary_cuts
            .iter()
            .map(|c| (c.intercept, c.coefficients.clone()))
            .collect();
        assert_eq!(
            loaded, expected,
            "the shared pool's cut content must match the loaded boundary cuts exactly — one \
             value function applied to every terminal fan node, not a per-leaf distinct policy"
        );
    }
}

#[test]
fn multi_node_source_stage_is_rejected() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let source_dir = tmp.path().join("multi_node_source");
    // Two declared nodes (ids 200, 201) both at stage 5: a multi-node source.
    write_synthetic_checkpoint(&source_dir, &[(200, 5, 0), (201, 5, 0)], 0, &[7.0, 11.0], 2);

    let result = load_boundary_cuts(
        &source_dir,
        5,
        2,
        &[],
        &[],
        None,
        LEGACY_COST_SCALE_FACTOR,
        &mut |_| {},
    );

    assert!(
        result.is_err(),
        "a multi-node source stage must be rejected, not silently resolved to one of its nodes"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("multi-node"),
        "rejection must name the stage as multi-node: {msg}"
    );
    assert!(
        msg.contains("source_stage 5"),
        "rejection must name the offending stage: {msg}"
    );
}
