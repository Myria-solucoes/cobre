//! Pre-resolved penalty and bound containers for O(1) solver lookup.
//!
//! During input loading, the three-tier cascade (global defaults → entity overrides
//! → stage overrides) is evaluated once and the results stored in these containers.
//! Solvers and LP builders then query resolved values in constant time via direct
//! array indexing — no re-evaluation of the cascade at solve time.
//!
//! The storage layout for both `ResolvedPenalties` and `ResolvedBounds` uses a
//! flat `Vec<T>` with manual 2D indexing:
//! `data[entity_idx * n_stages + stage_idx]`
//!
//! This gives cache-friendly sequential access when iterating over stages for a
//! fixed entity (the inner loop pattern used by LP builders).
//!
//! # Population
//!
//! These containers are populated by `cobre-io` during the penalty/bound resolution
//! step. They are never modified after construction.
//!
//! # Note on deficit segments
//!
//! Bus deficit segments are **not** stage-varying (see Penalty System spec SS3).
//! The piecewise structure is too complex for per-stage override. Therefore
//! `BusStagePenalties` contains only `excess_cost`.
//!
//! # Submodule layout
//!
//! - `penalties` — the four per-`(entity, stage)` penalty structs and the
//!   `ResolvedPenalties` table.
//! - `bounds` — the five per-`(entity, stage)` bound structs and the
//!   `ResolvedBounds` table (including the padded thermal stage axis).
//! - `generic` — `ResolvedGenericConstraintBounds`, the sparse RHS table for
//!   user-defined linear constraints, plus its private deterministic wire format.
//! - `factors` — the dense per-block factor tables and the
//!   non-controllable-source availability/factor family.
//!
//! Every public symbol is re-exported here so both the curated flat surface in
//! `lib.rs` and the `cobre_core::resolved::Symbol` module path resolve to the
//! same item regardless of which submodule owns it.

mod bounds;
mod factors;
mod generic;
mod penalties;

pub use bounds::{
    BoundsCountsSpec, BoundsDefaults, ContractStageBounds, HydroStageBounds, LineStageBounds,
    PumpingStageBounds, ResolvedBounds, ThermalStageBounds,
};
pub use factors::{
    ResolvedExchangeFactors, ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors,
};
pub use generic::ResolvedGenericConstraintBounds;
pub use penalties::{
    BusStagePenalties, HydroStagePenalties, LineStagePenalties, NcsStagePenalties,
    PenaltiesCountsSpec, PenaltiesDefaults, ResolvedPenalties,
};
