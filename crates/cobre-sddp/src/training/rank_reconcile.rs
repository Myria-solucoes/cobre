//! Deterministic cross-rank error reconciliation before lockstep collectives.

use cobre_comm::{CommError, Communicator, ReduceOp};

use crate::SddpError;

/// Reduce every rank's local outcome to one global outcome that all ranks return
/// identically, so they enter or skip the next collective in lockstep.
///
/// `ReduceOp::Max` over a 0/non-zero flag is order-independent, so the global
/// flag — hence the returned `Result` shape — is identical on every rank
/// regardless of rank count (the declaration-order/bit-for-bit invariant). A
/// failing rank keeps its own error; a healthy rank whose peer failed returns a
/// peer-failure `Communication` error. `scratch` is caller-owned so no heap
/// allocation occurs. A single-rank communicator returns `local_ok` unchanged
/// with no collective.
///
/// # Errors
///
/// Returns [`SddpError::Communication`] when the `allreduce` transport fails or
/// when a peer failed while this rank did not; returns this rank's own error when
/// it was the failing rank.
pub(crate) fn reconcile_error_flag<C: Communicator>(
    local_ok: Result<(), SddpError>,
    comm: &C,
    scratch: &mut [i32; 1],
) -> Result<(), SddpError> {
    if comm.size() == 1 {
        return local_ok;
    }

    scratch[0] = i32::from(local_ok.is_err());

    let mut reduced = [0_i32];
    comm.allreduce(&scratch[..], &mut reduced, ReduceOp::Max)
        .map_err(SddpError::from)?;

    if reduced[0] == 0 {
        Ok(())
    } else {
        Err(local_ok.err().unwrap_or_else(|| {
            SddpError::Communication(CommError::CollectiveFailed {
                operation: "reconcile_error_flag",
                mpi_error_code: 0,
                message: "a peer rank reported a local failure; failing this \
                          phase on every rank in lockstep"
                    .to_string(),
            })
        }))
    }
}

/// Reconcile a rank-local `Result<T, SddpError>` across ranks before a lockstep
/// collective: every rank returns `Ok(payload)` iff no rank failed, otherwise
/// every rank returns `Err` (its own error on a failing rank, a peer-failure
/// `Communication` error on a healthy rank whose peer failed). The `T` payload
/// rides on the stack and `scratch` is caller-owned, so no heap allocation occurs.
///
/// # Errors
///
/// Returns whatever [`reconcile_error_flag`] returns, plus a
/// [`SddpError::Communication`] on the branch (unreachable by the primitive's
/// contract) where the flag agreed but this rank held no payload.
pub(crate) fn reconcile_result<T, C: Communicator>(
    local: Result<T, SddpError>,
    comm: &C,
    scratch: &mut [i32; 1],
) -> Result<T, SddpError> {
    let (payload, local_ok) = match local {
        Ok(value) => (Some(value), Ok(())),
        Err(e) => (None, Err(e)),
    };
    reconcile_error_flag(local_ok, comm, scratch)?;
    // reconcile_error_flag returned Ok ⟹ no rank failed ⟹ this rank produced a payload.
    payload.ok_or_else(|| {
        SddpError::Communication(CommError::CollectiveFailed {
            operation: "reconcile_result",
            mpi_error_code: 0,
            message: "cross-rank reconcile agreed but this rank held no payload".to_string(),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::{reconcile_error_flag, reconcile_result};
    use crate::SddpError;
    use crate::forward::ForwardResult;
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};

    enum Mode {
        Reduce,
        Transport,
        Forbidden,
    }

    struct ReconcileStub {
        size: usize,
        peer_flag: i32,
        mode: Mode,
    }

    impl Communicator for ReconcileStub {
        fn allgatherv<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            unreachable!("reconcile_error_flag does not call allgatherv")
        }

        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            op: ReduceOp,
        ) -> Result<(), CommError> {
            match self.mode {
                Mode::Forbidden => {
                    unreachable!("single-rank fast path must not enter a collective")
                }
                Mode::Transport => Err(CommError::CollectiveFailed {
                    operation: "allreduce",
                    mpi_error_code: 7,
                    message: "simulated transport failure".to_string(),
                }),
                Mode::Reduce => {
                    assert_eq!(op, ReduceOp::Max, "reconcile must reduce with Max");
                    assert_eq!(send.len(), recv.len());
                    for (s, r) in send.iter().zip(recv.iter_mut()) {
                        *r = upcast_i32(downcast_i32(*s).max(self.peer_flag));
                    }
                    Ok(())
                }
            }
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            unreachable!("reconcile_error_flag does not call broadcast")
        }

        fn barrier(&self) -> Result<(), CommError> {
            unreachable!("reconcile_error_flag does not call barrier")
        }

        fn rank(&self) -> usize {
            0
        }

        fn size(&self) -> usize {
            self.size
        }

        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
    }

    // Box<dyn Any> is the only safe typed-dispatch path through the generic
    // `allreduce` signature in this forbid-unsafe crate; the stub only ever
    // reduces the i32 flag `reconcile_error_flag` sends.
    fn downcast_i32<T: CommData>(value: T) -> i32 {
        use std::any::Any;
        let boxed: Box<dyn Any> = Box::new(value);
        *boxed
            .downcast::<i32>()
            .expect("reconcile stub only reduces i32 flags")
    }

    fn upcast_i32<T: CommData>(value: i32) -> T {
        use std::any::Any;
        let boxed: Box<dyn Any> = Box::new(value);
        *boxed
            .downcast::<T>()
            .expect("reconcile stub only reduces i32 flags")
    }

    #[test]
    fn single_rank_ok_passes_through_without_collective() {
        let comm = ReconcileStub {
            size: 1,
            peer_flag: 0,
            mode: Mode::Forbidden,
        };
        let mut scratch = [0_i32];
        assert!(reconcile_error_flag(Ok(()), &comm, &mut scratch).is_ok());
    }

    #[test]
    fn single_rank_err_passes_through_without_collective() {
        let comm = ReconcileStub {
            size: 1,
            peer_flag: 0,
            mode: Mode::Forbidden,
        };
        let mut scratch = [0_i32];
        let out = reconcile_error_flag(
            Err(SddpError::Validation("local".to_string())),
            &comm,
            &mut scratch,
        );
        assert!(matches!(out, Err(SddpError::Validation(_))));
    }

    #[test]
    fn multi_rank_all_ok_returns_ok() {
        let comm = ReconcileStub {
            size: 4,
            peer_flag: 0,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        assert!(reconcile_error_flag(Ok(()), &comm, &mut scratch).is_ok());
    }

    #[test]
    fn multi_rank_failing_rank_returns_its_own_error() {
        let comm = ReconcileStub {
            size: 4,
            peer_flag: 0,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        let out = reconcile_error_flag(
            Err(SddpError::Infeasible {
                stage: 2,
                iteration: 3,
                scenario: 1,
            }),
            &comm,
            &mut scratch,
        );
        assert!(matches!(out, Err(SddpError::Infeasible { .. })));
    }

    #[test]
    fn multi_rank_healthy_peer_of_failing_rank_returns_err() {
        let comm = ReconcileStub {
            size: 4,
            peer_flag: 1,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        let out = reconcile_error_flag(Ok(()), &comm, &mut scratch);
        assert!(matches!(out, Err(SddpError::Communication(_))));
    }

    #[test]
    fn lockstep_agreement_every_rank_gets_err_when_one_fails() {
        let ranks: [(Result<(), SddpError>, i32); 3] = [
            (Ok(()), 1),
            (Err(SddpError::Validation("rank1 failed".to_string())), 0),
            (Ok(()), 1),
        ];
        for (local_ok, peer_flag) in ranks {
            let comm = ReconcileStub {
                size: 3,
                peer_flag,
                mode: Mode::Reduce,
            };
            let mut scratch = [0_i32];
            assert!(
                reconcile_error_flag(local_ok, &comm, &mut scratch).is_err(),
                "every rank must return Err in lockstep when one rank fails"
            );
        }
    }

    #[test]
    fn transport_failure_surfaces_as_communication_error() {
        let comm = ReconcileStub {
            size: 2,
            peer_flag: 0,
            mode: Mode::Transport,
        };
        let mut scratch = [0_i32];
        let out = reconcile_error_flag(Ok(()), &comm, &mut scratch);
        assert!(matches!(out, Err(SddpError::Communication(_))));
    }

    fn dummy_forward_result() -> ForwardResult {
        ForwardResult {
            scenario_costs: Vec::new(),
            elapsed_ms: 0,
            lp_solves: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
            stage_stats: Vec::new(),
        }
    }

    /// Forward-phase deadlock-freedom: with rank 1's forward solve returning
    /// `Infeasible` on a 2-rank comm, the reconcile between the solve loop and
    /// `sync_forward` makes BOTH ranks return `Err` (this test terminates rather
    /// than one rank blocking in `sync_forward`'s allgatherv).
    #[test]
    fn reconcile_result_fails_both_ranks_before_forward_collective() {
        // Rank 1: local forward solve failed; it keeps its own error.
        let failing = ReconcileStub {
            size: 2,
            peer_flag: 0,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        let rank1 = reconcile_result::<ForwardResult, _>(
            Err(SddpError::Infeasible {
                stage: 1,
                iteration: 3,
                scenario: 0,
            }),
            &failing,
            &mut scratch,
        );
        assert!(matches!(rank1, Err(SddpError::Infeasible { .. })));

        // Rank 0: local forward solve succeeded, but its peer failed; it must still
        // return Err (a peer-failure Communication error) so it never enters
        // sync_forward.
        let healthy = ReconcileStub {
            size: 2,
            peer_flag: 1,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        let rank0 = reconcile_result(Ok(dummy_forward_result()), &healthy, &mut scratch);
        assert!(matches!(rank0, Err(SddpError::Communication(_))));
    }

    /// Finalize deadlock-freedom: a rank that reached the clean `finalize` while a
    /// peer is erroring must return `Err` (and skip `broadcast_basis_cache`) rather
    /// than enter the broadcast alone, and the erroring rank (in
    /// `finalize_with_error`) keeps its own error and likewise skips the broadcast.
    #[test]
    fn finalize_reconcile_makes_all_ranks_skip_broadcast_in_lockstep() {
        let clean_peer_failed = ReconcileStub {
            size: 2,
            peer_flag: 1,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        assert!(matches!(
            reconcile_error_flag(Ok(()), &clean_peer_failed, &mut scratch),
            Err(SddpError::Communication(_))
        ));

        let erroring = ReconcileStub {
            size: 2,
            peer_flag: 0,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        assert!(matches!(
            reconcile_error_flag(
                Err(SddpError::Validation("boom".to_string())),
                &erroring,
                &mut scratch,
            ),
            Err(SddpError::Validation(_))
        ));
    }
}
