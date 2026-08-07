//! MPI-integration-job gates for the cut exchange:
//!
//! - The single-pool `CutSyncBuffers::sync_cuts` uniform-count rejection: an
//!   `SddpError::Validation` when `local_cuts.len()` differs from
//!   `per_rank_cuts[my_rank]` (that guard is unchanged for the single-pool path).
//! - The real-MPI DECOMP K-fan branching rank-invariance gate: `final_lb` is
//!   bitwise identical across world sizes, driving the genuine per-`(rank, pool)`
//!   cut-count exchange under `mpiexec -n 2`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic
)]

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_sddp::{
    FutureCostFunction, SddpError, cut::wire::CutWireTuple, cut_sync::CutSyncBuffers,
};

/// Stub 2-rank cluster from rank 0's perspective. `allreduce` copies send to
/// recv — no aggregation is needed because `sync_cuts` validates the cut count
/// before reaching `allgatherv`.
struct StubComm2Rank;

impl Communicator for StubComm2Rank {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        counts: &[usize],
        displs: &[usize],
    ) -> Result<(), CommError> {
        for (&count, &displ) in counts.iter().zip(displs.iter()) {
            let src_len = count.min(send.len());
            recv[displ..displ + src_len].clone_from_slice(&send[..src_len]);
        }
        Ok(())
    }

    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _op: ReduceOp,
    ) -> Result<(), CommError> {
        recv.clone_from_slice(send);
        Ok(())
    }

    fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
        Ok(())
    }

    fn barrier(&self) -> Result<(), CommError> {
        Ok(())
    }

    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        2
    }

    fn abort(&self, error_code: i32) -> ! {
        std::process::exit(error_code)
    }
}

#[test]
fn sync_cuts_invariant_rejected_when_cut_count_mismatches() {
    let n_state = 2;
    let num_ranks = 2;
    let max_cuts_per_rank = 3;
    let total_forward_passes = 6;

    let mut bufs = CutSyncBuffers::with_distribution(
        n_state,
        max_cuts_per_rank,
        num_ranks,
        total_forward_passes,
    );

    let forward_passes = u32::try_from(max_cuts_per_rank).expect("max_cuts_per_rank fits in u32");
    let mut fcf = FutureCostFunction::new(1, n_state, forward_passes, 10, &[0; 1]);
    let comm = StubComm2Rank;

    // 2 cuts vs the expected per_rank_cuts[0] == 3 — the mismatch under test.
    let coeffs_a = [1.0_f64, 2.0_f64];
    let coeffs_b = [3.0_f64, 4.0_f64];
    let local_cuts: &[CutWireTuple<'_>] =
        &[(0, 0, 1, 0, 10.0, &coeffs_a), (0, 0, 1, 1, 20.0, &coeffs_b)];

    let result = bufs.sync_cuts(0, local_cuts, &mut fcf, &comm);

    match result {
        Err(SddpError::Validation(ref msg)) => {
            assert!(
                msg.contains("sync_cuts invariant violated"),
                "error must contain 'sync_cuts invariant violated'; got: {msg}"
            );
            assert!(
                msg.contains("rank 0 produced 2 cuts, expected 3"),
                "error must contain 'rank 0 produced 2 cuts, expected 3'; got: {msg}"
            );
        }
        other => panic!("expected Err(SddpError::Validation(_)), got: {other:?}"),
    }
}

/// Real-MPI DECOMP K-fan branching rank-invariance under the default by-scenario
/// scheduler. Launched under `mpiexec -n 2` (the MPI integration job) the real
/// communicator drives the genuine per-`(rank, pool)` cut-count exchange at the
/// fan stage's multi-node backward level; under plain `cargo test` the world is
/// size 1 and the comparison is the single-rank identity. Either way the
/// real-communicator training must match the `LocalBackend` single-rank reference
/// to the bit, and must not return the (now removed) multi-node rejection.
///
/// `create_communicator(BackendKind::Auto)` resolves to `LocalBackend` with no
/// launcher and to the MPI world under `mpiexec`, so the same source compiles and
/// runs in both the no-`mpi` and `mpi` builds without a per-crate feature gate;
/// its genuine 2-rank pass is verified by the real-MPI integration job.
#[cfg(feature = "test-support")]
mod k_fan_branching_rank_invariance {
    use cobre_comm::{BackendKind, Communicator, LocalBackend, create_communicator};
    use cobre_sddp::setup::NodePos;
    use cobre_sddp::test_support::k_fan_setup;
    use cobre_solver::ActiveSolver;

    const K: usize = 8;
    const FORWARD_PASSES: u32 = 6;
    const MAX_ITERATIONS: u32 = 3;

    /// Train a fresh K-fan fixture at world size `comm.size()`, single-threaded,
    /// under the default by-scenario scheduler; return the full converged result
    /// `(final_lb, final_ub, final_ub_std)` — the lower bound and the statistical
    /// upper bound's mean/std, all of which must be rank-shape-invariant.
    fn train_result<C: Communicator>(comm: &C) -> (f64, f64, f64) {
        let mut fixture = k_fan_setup(K, FORWARD_PASSES, MAX_ITERATIONS);
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = fixture
            .setup
            .train(&mut solver, comm, 1, ActiveSolver::new, None, None)
            .expect("k_fan training must return Ok (no multi-node rejection)");
        assert!(
            outcome.error.is_none(),
            "k_fan training must not error: {:?}",
            outcome.error
        );
        (
            outcome.result.final_lb,
            outcome.result.final_ub,
            outcome.result.final_ub_std,
        )
    }

    #[test]
    fn k_fan_final_lb_bitwise_invariant_across_world_size() {
        let world =
            create_communicator(BackendKind::Auto).expect("communicator construction must succeed");
        let world_size = world.size();

        // Power self-check: the fan level carries >= 2 cut-generating pools, and
        // at world size >= 2 a single rank's forward passes are fewer than those
        // pools — so at least one pool draws cuts from a strict subset of ranks,
        // the genuine per-(rank, pool) divergence the fix addresses.
        let probe = k_fan_setup(K, FORWARD_PASSES, MAX_ITERATIONS);
        let node_graph = &probe.setup.node_graph;
        let cut_generating = (0..node_graph.nodes.len())
            .filter(|&pos| !node_graph.successors[NodePos(pos)].is_empty())
            .count();
        let fan_nodes = cut_generating - 1;
        assert!(
            fan_nodes >= 2,
            "power: the K-fan must carry >= 2 cut-generating fan pools (got {fan_nodes})"
        );
        const { assert!(FORWARD_PASSES >= 2, "power: forward_passes must be >= 2") };
        if world_size >= 2 {
            let per_rank_max =
                FORWARD_PASSES.div_ceil(u32::try_from(world_size).expect("world size fits u32"));
            assert!(
                usize::try_from(per_rank_max).expect("per_rank_max fits usize") < fan_nodes,
                "power: a single rank's forward passes ({per_rank_max}) must be fewer than the \
                 fan pools ({fan_nodes}) so some pool draws cuts from a strict subset of ranks"
            );
        }

        let (lb_n, ub_n, ub_std_n) = train_result(&world);
        let (lb_1, ub_1, ub_std_1) = train_result(&LocalBackend);
        // The full converged result — lower bound, and the statistical upper
        // bound's mean and std — must not change with workload distribution: the
        // cut set, the canonical-order LB root evaluation, and the canonical-order
        // forward-cost merge are all rank-shape-invariant.
        assert_eq!(
            lb_n.to_bits(),
            lb_1.to_bits(),
            "final_lb at world size {world_size} (real MPI) must be bitwise identical to the \
             single-rank reference"
        );
        assert_eq!(
            ub_n.to_bits(),
            ub_1.to_bits(),
            "final_ub at world size {world_size} (real MPI) must be bitwise identical to the \
             single-rank reference"
        );
        assert_eq!(
            ub_std_n.to_bits(),
            ub_std_1.to_bits(),
            "final_ub_std at world size {world_size} (real MPI) must be bitwise identical to the \
             single-rank reference"
        );
    }
}
