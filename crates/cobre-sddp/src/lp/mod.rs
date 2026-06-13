//! LP-construction cluster: the column/row index map, the generic-constraint
//! lowering, and the stage-template builder that together turn a loaded
//! `System` into the structural stage LPs every pass solves.
//!
//! This directory module groups the three pieces that own the LP's structure,
//! kept together because each depends on the column layout the next encodes:
//!
//! - [`indexer`] — [`StageIndexer`] maps every entity to its LP column/row
//!   range. The `storage_fixing` / `lag_fixing` / `anticipated_state_fixing`
//!   ranges are permanent empty `0..0` sentinels: state is pinned via
//!   [`StageIndexer::state_to_lp_incoming_column`] column bounds, never a fixing
//!   row (`.claude/rules/sddp.md`, "State pinning uses column bounds").
//! - [`generic_constraints`] — lowers user-declared generic constraints onto the
//!   indexed column layout. Crate-private: it has no external raw-path consumer.
//! - [`builder`] — [`build_stage_templates`] assembles the CSC structural LP,
//!   bounds, and objective for each stage once at startup. The FPHA generation
//!   constraint carries the `−γᵥ/2` coefficient on **both** storage columns
//!   (`.claude/rules/sddp.md`, "FPHA uses average storage").

pub mod builder;
pub(crate) mod generic_constraints;
pub mod indexer;

// Re-exported here so the crate-root curated surface in `lib.rs`
// (`cobre_sddp::{EquipmentCounts, FphaColumnLayout, StageIndexer, StageTemplates,
// build_stage_templates}`) can re-point through this cluster after the move.
pub use builder::{StageTemplates, build_stage_templates};
pub use indexer::{EquipmentCounts, FphaColumnLayout, StageIndexer};
