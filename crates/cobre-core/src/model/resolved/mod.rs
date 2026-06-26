//! Pre-resolved penalty and bound containers for O(1) solver lookup.
//!
//! The three-tier cascade (global defaults → entity overrides → stage overrides)
//! is evaluated once during input loading; these containers hold the result for
//! constant-time array-indexed lookup at solve time. Populated by `cobre-io`;
//! never modified after construction.
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
