# MPI lockstep-collective deadlock audit

Static census (2026-07-09) of every `Communicator` collective call site in the
workspace, classifying each by whether an error path can diverge ranks before
the collective — the failure class where rank A returns early on an error
while rank B blocks forever inside a collective. The error-reconciliation
primitive (`training/rank_reconcile.rs`: order-independent i32 flag +
`ReduceOp::Max` allreduce) already protects the forward phase and the
finalize basis broadcast; this audit covers the rest of the surface.

Every collective dispatches through the `cobre-comm` trait and only the MPI
backend blocks, so every exposed site below requires `--features mpi` with
world size ≥ 2; single-rank runs take local fast paths everywhere.

## Verdict

**Four distinct exposed root causes (~8 physical call sites). Fixes are four
small standalone tickets reusing `reconcile_error_flag`/`reconcile_result` —
not a remediation-plan-sized effort.**

| Root cause                                                                                                                                                                                                                             | Sites                                                                                             | Severity                                                                                                                              | Minimal fix                                                                                                                                                             |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Backward-pass per-stage collectives entered after the per-rank parallel solve (`process_stage_backward`'s `worker_result?`): `sync_packed_records` allgatherv, `sync_stage_metadata` allreduce, `gather_stage_solver_stats` allgatherv | `cut/cut_sync.rs` + `training/backward_pass_state.rs` (fix point: before `sync_start`, ~line 795) | **High** — ranks solve disjoint trial points, so per-rank infeasibility/solver failure is genuinely divergent; hangs a production run | ONE reconcile allreduce per backward stage after the solve loop, before the first sync collective; also retroactively shields the post-backward `rows_in_lp` allreduces |
| Lower-bound broadcast: rank 0's `lb_evaluate_stage_0` can fail before `lb_aggregate_and_broadcast`'s broadcast while non-root ranks wait at their matching broadcast                                                                   | `training/lower_bound.rs` (~289 rank-0 / ~353 non-root)                                           | Medium-low (stage 0 designed feasible; genuine solver error still possible)                                                           | Reconcile rank-0's Result before the broadcast, or fold failure into the broadcast via a sentinel (the `broadcast_value` len-0 pattern)                                 |
| CLI simulation post-solve collectives: `simulate()` is collective-free per rank, but `sim_result?` returns divergently before `merge_simulation_metadata`'s allreduce and the aggregation collectives                                  | `cobre-cli/src/commands/run/simulation.rs` (divergence ~line 101, first collective ~239)          | Medium-low (sim solves are penalty-feasible by design)                                                                                | Reconcile `sim_result` across ranks before the first post-sim collective (thin helper exposed from `cobre-sddp`, or a status-flag allreduce in the CLI)                 |
| CLI setup post-export barrier: rank-0-only export writes (`write_hydro_model_summary`, provenance, `write_scaling_report`) return `?` before a barrier all ranks enter                                                                 | `cobre-cli/src/commands/run/setup.rs` (barrier ~line 643)                                         | Medium — disk-full/permissions is a realistic operational failure                                                                     | Reconcile the rank-0 write Result (or allreduce a status flag) before the barrier                                                                                       |

## Protected / safe sites (no action)

- Forward-phase stage-stats allreduce and `sync_forward` allgatherv — behind
  the forward `reconcile_result`; the forward pass itself is collective-free.
- Backward entry `n_workers` Min/Max allreduces — the min≠max check is itself
  a model reconcile (all ranks fail together).
- `exchange()` per-stage allgatherv — no divergent fallible op reaches it.
- `broadcast_basis_cache` ×4 — the shipped reconcile fix; all-or-none entry.
- System/config/resolved-parameters/tree `broadcast_value` — protected by the
  len-0 sentinel (rank-0 load failure propagates symmetrically). Residual:
  `postcard::to_allocvec` is a `?` before the length broadcast but does not
  fail for these concrete types.
- `aggregate_simulation`'s five collectives — lockstep once entered (entry is
  the CLI's responsibility, covered by the simulation row above).
- Training-phase post-`train` allreduce/barrier — `train` returns symmetrically
  by construction (internal finalize reconcile).
- `sync_cuts` — test-only caller; production uses `sync_packed_records`.
- `shared-memory`-gated fence/barrier surface — no production consumers.

## Open items requiring dynamic observation (not statically dischargeable)

1. Whether a production case actually produces backward infeasibility on a
   strict subset of ranks (vs. the false-infeasible retries the cold-solve
   escalation ladder absorbs). The structural fix is warranted regardless.
2. Asymmetric collective _transport_ failure: every protected sequence assumes
   a collective returns Ok on all ranks or Err on all ranks. Confirming the
   EFA/libfabric provider never returns one-sided Err is an observation task;
   MPI convention treats such failures as fatal/symmetric.
3. `postcard` serialization fallibility for the broadcast payload types —
   dischargeable only by a property/fuzz check over those Serialize impls.

The backward-pass fix is the only one touching hot-path code and warrants a
deadlock-freedom regression mirroring the existing
`reconcile_result_fails_both_ranks_before_forward_collective` test.
