//! LP layout index map for SDDP stage subproblems.
//!
//! The state-vector column layout is owned by [`StateLayout`]; the per-stage
//! equipment column/row geometry is owned by
//! [`StageLayout`](crate::lp_builder)/[`StageGeometry`](crate::lp_builder::StageGeometry);
//! the non-state study shape is owned by [`StudyDimensions`]. Together they
//! eliminate magic index numbers throughout the forward pass, backward pass, and
//! LP construction code. The full column/row layout is documented below.
//!
//! ## Column layout (Solver Abstraction SS2.1)
//!
//! ```text
//! [0, N)                                    storage           — outgoing storage volumes  (N = hydro_count)
//! [N, N*(1+L))                              inflow_lags       — AR lag variables (L lags per hydro)
//! [N*(1+L), N*(1+L) + A*K_max)              anticipated_state     — anticipated thermal commitment state slots (ring buffer)
//! [N*(1+L) + A*K_max, N*(1+L) + A*K_max + A) anticipated_state_out — relocated cut-target column; state region, stage-invariant (owned by `StateLayout`)
//! [N*(1+L) + A*K_max + A, N*(2+L) + A*K_max + A) z_inflow          — realized inflow (auxiliary, not state)
//! [N*(2+L) + A*K_max + A, N*(3+L) + A*K_max + A) storage_in        — incoming storage volumes
//! N*(3+L) + A*K_max + A                      theta                 — future cost variable (scalar)
//! ```
//!
//! where `A = n_anticipated` is the number of thermals with
//! `anticipated_config.is_some()` and `K_max` is the maximum `lead_stages`
//! across those plants. When `A == 0` the layout collapses to the
//! pre-anticipated form: `z_inflow` at `N*(1+L)`, `theta` at `N*(3+L)`.
//!
//! The following equipment columns follow immediately after `theta` (laid out
//! per stage by [`StageLayout`](crate::lp_builder)):
//!
//! ```text
//! [theta+1,                                  theta+1+H*K)                turbine                — turbined flow (m³/s)
//! [theta+1+H*K,                              theta+1+2*H*K)              spillage               — spilled flow (m³/s)
//! [theta+1+2*H*K,                            theta+1+3*H*K)              diversion              — diverted flow (m³/s)
//! [theta+1+3*H*K,                            theta+1+3*H*K+T*K)          thermal                — thermal generation (MW)
//! [theta+1+3*H*K+T*K,                        theta+1+3*H*K+T*K+A)        anticipated_decision   — A = n_anticipated columns
//! [theta+1+3*H*K+T*K+A,                      …+A+2*L_n*K)                line_fwd/rev           — line flows
//! [theta+1+3*H*K+T*K+A+2*L_n*K,             …+A+2*L_n*K+B*S*K)          deficit
//! [theta+1+3*H*K+T*K+A+2*L_n*K+B*S*K,       …+A+2*L_n*K+B*S*K+B*K)      excess
//! ```
//!
//! The `anticipated_decision` block is stage-level (one column per anticipated
//! plant, NOT per-block) and has length `A = n_anticipated`. The block collapses
//! to length 0 when `n_anticipated == 0`, leaving the rest of the layout
//! byte-identical to the pre-anticipated form. The control region runs
//! `anticipated_decision` then `line_fwd` directly — the cut-target
//! `anticipated_state_out` column does NOT live here: it was relocated into the
//! stage-invariant state region above (`[N*(1+L)+A*K_max, …+A)`, owned by
//! [`StateLayout`]), so its address never depends on `n_blks`. The
//! `anticipated_state_out_def` equality row still pins it to its
//! `anticipated_decision` column.
//!
//! When the inflow non-negativity penalty method is active (`has_inflow_penalty == true`),
//! `N` additional slack columns are appended after `excess`:
//!
//! ```text
//! [excess_end, excess_end+N)  inflow_slack — sigma_inf_h (m³/s), one per hydro
//! ```
//!
//! After FPHA generation and evaporation columns, `N` withdrawal slack columns are
//! appended when `hydro_count > 0`:
//!
//! ```text
//! [evap_end, evap_end+N)    withdrawal_slack_neg — under-withdrawal (m³/s), one per hydro
//! [evap_end+N, evap_end+2N) withdrawal_slack_pos — over-withdrawal (m³/s), one per hydro
//! ```
//!
//! After withdrawal slack columns, 4 operational violation slack column regions are
//! appended when `hydro_count > 0` (one column per hydro per block in each region):
//!
//! ```text
//! [ws_end,          ws_end+N*K)    outflow_below_slack    — per-block min-outflow violation
//! [ws_end+N*K,      ws_end+2*N*K)  outflow_above_slack    — per-block max-outflow violation
//! [ws_end+2*N*K,    ws_end+3*N*K)  turbine_below_slack    — per-block min-turbine violation
//! [ws_end+3*N*K,    ws_end+4*N*K)  generation_below_slack — per-block min-generation violation
//! ```
//!
//! where `ws_end` = `withdrawal_slack_pos.end`, H = `hydro_count`, K = `n_blks`,
//! T = `n_thermals`, Ln = `n_lines`, B = `n_buses`, S = `max_deficit_segments`.
//!
//! ## Row layout (Solver Abstraction SS2.2)
//!
//! State pinning uses column bounds (`set_col_bounds`) on the incoming-state
//! columns, so the LP has no state-fixing row range. z-inflow rows start at
//! row 0.
//!
//! ```text
//! [0, N)   z_inflow_rows — z-inflow definition rows
//! ```
//!
//! After evaporation rows, 4 operational violation constraint row regions are
//! appended when `hydro_count > 0` (one row per hydro per block in each region):
//!
//! ```text
//! [evap_end,          evap_end+N*K)    min_outflow_rows    — per-block min-outflow constraints
//! [evap_end+N*K,      evap_end+2*N*K)  max_outflow_rows    — per-block max-outflow constraints
//! [evap_end+2*N*K,    evap_end+3*N*K)  min_turbine_rows    — per-block min-turbine constraints
//! [evap_end+3*N*K,    evap_end+4*N*K)  min_generation_rows — per-block min-generation constraints
//! ```
//!
//! After the operational violation rows, the anticipated-thermal fishing rows
//! are placed. The stage-0 canonical layout stores a zero-length range; per-stage
//! row counts (`0..n_anticipated`) are produced downstream from the
//! `anticipated_fishing_start` offset:
//!
//! ```text
//! [min_generation_rows.end, +0)   anticipated_fishing — zero rows at stage 0
//! ```
//!
//! ## Worked example (SS5.5.3): N = 3, L = 2
//!
//! Without anticipated thermals:
//! ```text
//! storage = 0..3, inflow_lags = 3..9, z_inflow = 9..12, storage_in = 12..15,
//! theta = 15, n_state = 9
//! ```
//!
//! With 2 anticipated thermals (`K_max = 3`): `anticipated_state = 9..15` and the
//! relocated `anticipated_state_out = 15..17` insert before `z_inflow`, shifting
//! `z_inflow` to `17..20`, `storage_in` to `20..23`, and `theta` to `23`.
//!
//! The per-solve patch sequence layered on top of this geometry is documented in
//! [`crate::lp_builder`].
//!
//! # Submodule layout
//!
//! - `layout` — the per-stage geometry satellite types [`EvaporationIndices`]
//!   and [`FphaRowRange`] (locating one hydro's evaporation columns/row and FPHA
//!   row block within a stage LP).
//! - `block_grid` — the [`BlockGrid`] typed block-stride address primitive and
//!   its three shape methods ([`BlockGrid::flat`], [`BlockGrid::fpha_plane`],
//!   [`BlockGrid::deficit`]).
//! - `state_layout` — the [`StateLayout`] type, the sole owner of the role-(a)
//!   state-vector concern: the stage-invariant state-vector column ranges, the
//!   two layout-derived caches, and the resolver / mask methods
//!   ([`StateLayout::state_to_lp_column`],
//!   [`StateLayout::state_to_lp_incoming_column`],
//!   [`StateLayout::lp_column_for_state`], [`StateLayout::set_nonzero_mask`]). It
//!   finalizes both caches in its single constructor; downstream code threads a
//!   handle to it.
//! - `study_dimensions` — the [`StudyDimensions`] type, the single owner of the
//!   study-invariant non-state LP shape.
//! - `cut_state_projection` — the [`CutStateProjection`] type, a storage-scoped
//!   projection of [`StateLayout`] exposing only the cut-state dimensions a
//!   stage's `StageStateConfig` enables (anticipated state always included),
//!   delegating each column to [`StateLayout::state_to_lp_incoming_column`].
//!
//! Every public symbol is re-exported here so the `cobre_sddp::indexer::Symbol`
//! and `crate::indexer::Symbol` module paths resolve to the same item regardless
//! of which submodule owns it.

mod block_grid;
mod cut_state_projection;
mod layout;
mod state_layout;
mod study_dimensions;

pub use block_grid::BlockGrid;
pub use cut_state_projection::CutStateProjection;
pub use layout::{EvaporationIndices, FphaRowRange};
pub use state_layout::StateLayout;
pub use study_dimensions::StudyDimensions;
