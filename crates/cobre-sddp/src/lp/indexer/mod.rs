//! LP layout index map for SDDP stage subproblems.
//!
//! The state-vector column layout is owned by [`StateSpace`]; the per-stage
//! equipment column/row geometry is owned by
//! [`StageLayout`](crate::lp_builder)/[`StageGeometry`](crate::lp_builder::StageGeometry);
//! the non-state study shape is owned by [`StudyDimensions`]. The authoritative
//! ranges live on the owning types.
//!
//! ## Column layout (Solver Abstraction SS2.1)
//!
//! The stage-invariant state-vector column ranges (storage, AR lags,
//! travel-time buckets, the anticipated ring's outgoing and incoming blocks,
//! `z_inflow`, `theta`) are owned entirely by [`StateSpace`] — see its own
//! module doc for the authoritative diagram; this file does not re-derive it,
//! to avoid the two copies drifting apart.
//!
//! The equipment, slack, generic-constraint, and filling-phase column and row
//! ranges that follow `theta` — allocated in that equipment -> slack ->
//! generic -> filling family order — are owned entirely by
//! [`StageLayout`](crate::lp_builder).
//!
//! The `anticipated_decision` block is stage-level (one column per anticipated
//! plant, NOT per-block) and has length `A = n_anticipated`. The block collapses
//! to length 0 when `n_anticipated == 0`, leaving the rest of the layout
//! byte-identical to the pre-anticipated form. The control region runs
//! `anticipated_decision` then `line_fwd` directly — the anticipated ring's
//! outgoing slots (`StateSpace::anticipated_slots_out`) do NOT live here:
//! they sit in the stage-invariant state region owned by [`StateSpace`], so
//! their address never depends on `n_blks`. An `anticipated_state_out_def`
//! equality row pins the plant's own newest ring slot to its
//! `anticipated_decision` column.
//!
//! ## Row layout (Solver Abstraction SS2.2)
//!
//! State pinning uses column bounds (`set_col_bounds`) on the incoming-state
//! columns, so the LP has no state-fixing row range. z-inflow rows start at
//! row 0.
//!
//! The per-solve patch sequence layered on top of this geometry is documented in
//! [`crate::lp_builder`].
//!
//! # Submodule layout
//!
//! - `anticipated_gate` — the anticipated-decision temporal-gating free
//!   functions (`is_anticipated_decision_active_for_delivery`,
//!   `anticipated_resolution_for`, plus the test-only constant-lead wrapper
//!   `is_anticipated_decision_active`), taking `state: &`[`StateSpace`]
//!   rather than living as methods on it — column geometry and temporal
//!   gating are separate concerns.
//! - `layout` — the per-stage geometry satellite types [`EvaporationIndices`]
//!   and [`FphaRowRange`] (locating one hydro's evaporation columns/row and FPHA
//!   row block within a stage LP).
//! - `index` — the base typed vocabulary [`Col`]/[`Row`]/[`StateDim`] and the
//!   [`InCol`]/[`OutCol`] incoming/outgoing column-role split. Every
//!   `StateSpace`/`CutStateProjection` incoming and outgoing resolver, plus
//!   [`CutStateProjection::render_pairs`], resolves through it. [`BlockIdx`]
//!   is the block operand every [`BlockGrid`] shape method takes. [`Boundary`]
//!   is the chronological storage-boundary operand
//!   [`StorageBoundaryGrid::col`] takes, classified from a dynamic `k` via
//!   [`Boundary::from_index`].
//! - `block_grid` — the [`BlockGrid`] typed block-stride address primitive and
//!   its three shape methods ([`BlockGrid::flat`], [`BlockGrid::fpha_plane`],
//!   [`BlockGrid::deficit`]).
//! - `range_cursor` — the `RangeCursor` running column/row offset allocator
//!   shared by [`StageLayout`](crate::lp_builder)'s per-stage equipment chains
//!   and [`StateSpace`]'s stage-invariant state-vector chain.
//! - `storage_boundary_grid` — the [`StorageBoundaryGrid`] typed
//!   storage-boundary address primitive ([`StorageBoundaryGrid::col`]), the
//!   single owner of `block_storage_col`'s endpoint/interior split.
//! - `state_space` — the [`StateSpace`] type, the sole owner of the
//!   state-vector concern: the stage-invariant state-vector column ranges, the
//!   two layout-derived caches, and the resolver / mask methods
//!   ([`StateSpace::state_to_lp_column`],
//!   [`StateSpace::state_to_lp_incoming_column`],
//!   [`StateSpace::lp_column_for_state`], [`StateSpace::set_nonzero_mask`]). It
//!   finalizes both caches in its single constructor; downstream code threads a
//!   handle to it.
//! - `study_dimensions` — the [`StudyDimensions`] type, the single owner of the
//!   study-invariant non-state LP shape.
//! - `cut_state_projection` — the [`CutStateProjection`] type, a storage-scoped
//!   projection of [`StateSpace`] exposing only the cut-state dimensions a
//!   stage's `StageStateConfig` enables (anticipated state always included),
//!   delegating each column to [`StateSpace::state_to_lp_incoming_column`].
//! - `entity_index` — the entity system/local index vocabulary
//!   ([`HydroSys`]/[`ThermalSys`]/[`LineSys`], [`FphaLocal`]/[`EvapLocal`]/
//!   [`FillingTargetLocal`]/[`FloorLocal`]/[`AnticipatedLocal`],
//!   [`HydroCell`]/[`FphaCellLocal`]), distinguishing an entity's canonical
//!   system position from its position within a per-stage local identity list
//!   or within the study-scope hydro-cell partition.
//! - `hydro_cell` — the [`HydroCellIndex`] study-scope partition of each hydro
//!   plant's unit groups into `bus_id`-equivalence-class cells ([`HydroCell`]).
//!
//! Every public symbol is re-exported here so the `cobre_sddp::indexer::Symbol`
//! and `crate::indexer::Symbol` module paths resolve to the same item regardless
//! of which submodule owns it.

mod anticipated_gate;
mod block_grid;
mod cut_state_projection;
mod entity_index;
mod hydro_cell;
mod index;
mod layout;
mod range_cursor;
mod state_space;
mod storage_boundary_grid;
mod study_dimensions;

pub(crate) use anticipated_gate::{
    anticipated_resolution_for, is_anticipated_decision_active_for_delivery,
};
pub use block_grid::BlockGrid;
pub use cut_state_projection::CutStateProjection;
pub use entity_index::{
    AnticipatedLocal, EvapLocal, FillingTargetLocal, FloorLocal, FphaCellLocal, FphaLocal,
    HydroCell, HydroSys, LineSys, ThermalSys,
};
pub use hydro_cell::HydroCellIndex;
pub use index::{BlockIdx, Boundary, Col, CutSlot, InCol, OutCol, Row, StateDim};
pub use layout::{EvaporationIndices, FphaRowRange};
pub(crate) use range_cursor::RangeCursor;
pub use state_space::StateSpace;
pub(crate) use state_space::{REGION_ORDER, StateRegion};
pub use storage_boundary_grid::StorageBoundaryGrid;
pub use study_dimensions::StudyDimensions;
