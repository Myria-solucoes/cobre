//! `CutSyncBuffers::sync_cuts` rejects a local cut count that differs from the
//! expected per-rank count with `SddpError::Validation`.
//!
//! The separate `n_workers_local` uniformity handshake is covered by the unit
//! test `handshake_rejects_nonuniform_workers`.

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

/// Stub 2-rank cluster from rank 0's perspective. `allreduce` ignores `ReduceOp`
/// and copies send to recv — single-rank semantics suffice for the `sync_cuts`
/// invariant check, which validates before any aggregation matters.
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
fn sync_cuts_rejects_mismatched_local_cut_count() {
    let n_state = 2;
    let num_ranks = 2;
    let total_forward_passes = 6;
    let max_cuts_per_rank = 3;

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
                msg.contains("rank 0 produced 2 cuts"),
                "error must mention 'rank 0 produced 2 cuts'; got: {msg}"
            );
            assert!(
                msg.contains("expected 3"),
                "error must mention 'expected 3'; got: {msg}"
            );
        }
        other => panic!("expected Err(SddpError::Validation(_)), got: {other:?}"),
    }
}
