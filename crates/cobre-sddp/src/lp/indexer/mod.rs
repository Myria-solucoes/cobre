//! LP layout index map for SDDP stage subproblems.
//!
//! [`StageIndexer`] centralises all column and row offset arithmetic for a
//! single-stage LP, eliminating magic index numbers throughout the forward
//! pass, backward pass, and LP construction code.
//!
//! ## Column layout (Solver Abstraction SS2.1)
//!
//! ```text
//! [0, N)                                    storage           — outgoing storage volumes  (N = hydro_count)
//! [N, N*(1+L))                              inflow_lags       — AR lag variables (L lags per hydro)
//! [N*(1+L), N*(1+L) + A*K_max)              anticipated_state — anticipated thermal commitment state slots
//! [N*(1+L) + A*K_max, N*(2+L) + A*K_max)    z_inflow          — realized inflow (auxiliary, not state)
//! [N*(2+L) + A*K_max, N*(3+L) + A*K_max)    storage_in        — incoming storage volumes
//! N*(3+L) + A*K_max                         theta             — future cost variable (scalar)
//! ```
//!
//! where `A = n_anticipated` is the number of thermals with
//! `anticipated_config.is_some()` and `K_max` is the maximum `lead_stages`
//! across those plants. When `A == 0` the layout collapses to the
//! pre-anticipated form: `z_inflow` at `N*(1+L)`, `theta` at `N*(3+L)`.
//!
//! When built with [`StageIndexer::with_equipment`], the following equipment
//! columns follow immediately after `theta`:
//!
//! ```text
//! [theta+1,                                  theta+1+H*K)                turbine                — turbined flow (m³/s)
//! [theta+1+H*K,                              theta+1+2*H*K)              spillage               — spilled flow (m³/s)
//! [theta+1+2*H*K,                            theta+1+3*H*K)              diversion              — diverted flow (m³/s)
//! [theta+1+3*H*K,                            theta+1+3*H*K+T*K)          thermal                — thermal generation (MW)
//! [theta+1+3*H*K+T*K,                        theta+1+3*H*K+T*K+A)        anticipated_decision   — A = n_anticipated columns
//! [theta+1+3*H*K+T*K+A,                      theta+1+3*H*K+T*K+2*A)      anticipated_state_out  — A = n_anticipated columns
//! [theta+1+3*H*K+T*K+2*A,                    …+2*A+2*L_n*K)              line_fwd/rev           — line flows
//! [theta+1+3*H*K+T*K+2*A+2*L_n*K,           …+2*A+2*L_n*K+B*S*K)        deficit
//! [theta+1+3*H*K+T*K+2*A+2*L_n*K+B*S*K,     …+2*A+2*L_n*K+B*S*K+B*K)    excess
//! ```
//!
//! The `anticipated_decision` block is stage-level (one column per anticipated
//! plant, NOT per-block) and has length `A = n_anticipated`. The block collapses
//! to length 0 when `n_anticipated == 0`, leaving the rest of the layout
//! byte-identical to the pre-anticipated form.
//!
//! The `anticipated_state_out` block immediately follows `anticipated_decision`
//! and has the same length `A`. It holds the outgoing state variable for the
//! cut-mapping definition row. Both blocks collapse to length 0
//! together when `n_anticipated == 0`.
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
//! columns, so the `storage_fixing`, `lag_fixing`, and `anticipated_state_fixing`
//! row ranges (grouped in the [`Sentinels`] sub-struct, reached as
//! `indexer.sentinels.*`) are always empty (`0..0`). z-inflow rows start at row 0.
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
//! With 2 anticipated thermals (`K_max = 3`): `anticipated_state = 9..15` inserts
//! before `z_inflow`, shifting it to `15..18` and `theta` to `21`.
//!
//! # Submodule layout
//!
//! - `layout` — the [`StageIndexer`] struct, the [`Sentinels`] sub-struct (which
//!   owns the `0..0` sentinel field docs), the satellite types
//!   ([`EvaporationIndices`], [`FphaRowRange`], [`EquipmentCounts`],
//!   [`FphaColumnLayout`], [`EvapConfig`]), the small layout accessors, and the
//!   compile-time `Send + Sync` assertion.
//! - `state_mapping` — the state-vector-to-LP-column resolvers
//!   ([`StageIndexer::state_to_lp_column`], [`StageIndexer::finalize_state_column_map`],
//!   [`StageIndexer::lp_column_for_state`], and the state-pinning entry point
//!   [`StageIndexer::state_to_lp_incoming_column`]).
//! - `anticipated` — the per-stage anticipated-thermal iterators and predicates.
//! - `sparse_state` — the nonzero-state mask builder `set_nonzero_mask`.
//! - `constructors` — the three constructors and the private column/row range
//!   build helpers (carrying the `0..0` sentinel initialisers).
//!
//! Every public symbol is re-exported here so both the curated flat surface in
//! `lib.rs` and the `cobre_sddp::indexer::Symbol` / `crate::indexer::Symbol`
//! module path resolve to the same item regardless of which submodule owns it.

mod anticipated;
mod constructors;
mod layout;
mod sparse_state;
mod state_mapping;
#[cfg(any(test, feature = "test-support"))]
pub mod test_fixtures;

pub use layout::{
    EquipmentCounts, EvapConfig, EvaporationIndices, FphaColumnLayout, FphaRowRange, Sentinels,
    StageIndexer,
};
