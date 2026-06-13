//! Per-thread solver workspaces and the immutable hot-path context structs.
//!
//! This directory clusters the two read/write halves of what a stage solve
//! threads through its functions:
//!
//! - [`workspace`] — the mutable per-thread resources: [`SolverWorkspace`]
//!   bundles one thread's LP-solve scratch, [`WorkspacePool`] hands out one per
//!   worker, [`BasisStore`] holds per-scenario warm-start bases, and
//!   [`CapturedBasis`] is the slot-tracked basis whose
//!   [`to_broadcast_payload`](workspace::CapturedBasis::to_broadcast_payload) /
//!   [`try_from_broadcast_payload`](workspace::CapturedBasis::try_from_broadcast_payload)
//!   are the sole owners of the MPI basis-cache wire format.
//! - [`context`] — the immutable per-stage / per-training parameter bundles
//!   ([`StageContext`], [`TrainingContext`])
//!   passed by reference to keep hot-path argument counts down.

// Rationale: the workspace file keeps its established `workspace` basename
// inside this `workspace/` cluster, so the submodule shares its parent's name.
// Renaming the submodule would break the `workspace::workspace::{...}` re-export
// path the crate root uses for `CapturedBasis` / `BASIS_BROADCAST_WIRE_VERSION`
// for no behavioural gain.
pub mod context;
#[allow(clippy::module_inception)]
pub mod workspace;

// Re-export the symbols the in-crate `crate::workspace::Symbol` and
// `crate::context::Symbol` call sites reach for, so the directory-module move is
// transparent to them. `BackwardAccumulators` and `ScratchBuffers` are
// `pub(crate)`, so they are re-exported at the same visibility.
//
// Rationale: `BackwardAccumulators` is reached only from in-crate `#[cfg(test)]`
// modules via `crate::workspace::BackwardAccumulators`, so the re-export reads as
// unused in the non-test build; the `cfg(not(test))` allow scopes the suppression
// to that build only, keeping the warning live should a non-test caller drop.
pub use context::{StageContext, TrainingContext};
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use workspace::BackwardAccumulators;
pub(crate) use workspace::ScratchBuffers;
pub use workspace::{
    BASIS_BROADCAST_WIRE_VERSION, BasisStore, BasisStoreSliceMut, CapturedBasis, SolverWorkspace,
    WorkspacePool, WorkspaceSizing,
};
