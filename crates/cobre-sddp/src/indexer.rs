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
//! row ranges are always empty (`0..0`). z-inflow rows start at row 0.
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

use std::collections::HashMap;
use std::ops::Range;

use cobre_solver::StageTemplate;

/// Column and row indices for the evaporation constraint of one hydro.
///
/// Locates the three evaporation columns and one evaporation row assigned to
/// a single hydro within a stage LP.  Columns are stage-level (not per-block).
#[derive(Debug, Clone, Copy)]
pub struct EvaporationIndices {
    /// Column index of the stage-averaged evaporation flow variable `Q_ev_h` (m³/s).
    pub q_ev_col: usize,
    /// Column index of the positive violation slack `f_evap_plus_h` (m³/s).
    pub f_evap_plus_col: usize,
    /// Column index of the negative violation slack `f_evap_minus_h` (m³/s).
    pub f_evap_minus_col: usize,
    /// Row index of the evaporation equality constraint.
    pub evap_row: usize,
}

/// FPHA constraint row range for one hydro at one stage.
///
/// Locates the block of FPHA hyperplane rows assigned to a single FPHA hydro
/// within a stage LP. Rows for hydro `i` at block `k` and plane `p` are at:
/// `start + k * planes_per_block + p`.
#[derive(Debug, Clone, Copy)]
pub struct FphaRowRange {
    /// First row index of this hydro's FPHA constraints (for block 0, plane 0).
    pub start: usize,
    /// Number of hyperplanes per block.
    pub planes_per_block: usize,
}

/// Build the inflow-penalty slack column range.
///
/// Returns `(range, active)` where `active` is `true` when the penalty method
/// applies. When inactive, the range is empty (`0..0`).
fn build_inflow_slack_range(
    has_inflow_penalty: bool,
    hydro_count: usize,
    excess_end: usize,
) -> (Range<usize>, bool) {
    if has_inflow_penalty && hydro_count > 0 {
        (excess_end..excess_end + hydro_count, true)
    } else {
        (0..0, false)
    }
}

/// Column and row ranges for the four operational-violation slack families.
///
/// Produced by [`build_oper_violation_ranges`] and consumed by
/// [`StageIndexer::with_equipment_and_evaporation`].
struct OperViolationRanges {
    outflow_below_slack: Range<usize>,
    outflow_above_slack: Range<usize>,
    turbine_below_slack: Range<usize>,
    generation_below_slack: Range<usize>,
    min_outflow_rows: Range<usize>,
    max_outflow_rows: Range<usize>,
    min_turbine_rows: Range<usize>,
    min_generation_rows: Range<usize>,
    has_operational_violations: bool,
}

/// Build operational-violation column and row ranges.
///
/// When `hydro_count == 0`, all ranges are empty (`0..0`) and
/// `has_operational_violations` is `false`. Otherwise the four slack column
/// families are laid out immediately after `ws_end` (the end of the withdrawal
/// slack columns), and the four constraint row families are laid out immediately
/// after `evap_rows_end`.
fn build_oper_violation_ranges(
    hydro_count: usize,
    n_blks: usize,
    ws_end: usize,
    evap_rows_end: usize,
) -> OperViolationRanges {
    if hydro_count == 0 {
        return OperViolationRanges {
            outflow_below_slack: 0..0,
            outflow_above_slack: 0..0,
            turbine_below_slack: 0..0,
            generation_below_slack: 0..0,
            min_outflow_rows: 0..0,
            max_outflow_rows: 0..0,
            min_turbine_rows: 0..0,
            min_generation_rows: 0..0,
            has_operational_violations: false,
        };
    }
    let n_op = hydro_count * n_blks;
    let ob = ws_end..ws_end + n_op;
    let oa = ob.end..ob.end + n_op;
    let tb = oa.end..oa.end + n_op;
    let gb = tb.end..tb.end + n_op;
    let r_min_out = evap_rows_end..evap_rows_end + n_op;
    let r_max_out = r_min_out.end..r_min_out.end + n_op;
    let r_min_turb = r_max_out.end..r_max_out.end + n_op;
    let r_min_gen = r_min_turb.end..r_min_turb.end + n_op;
    OperViolationRanges {
        outflow_below_slack: ob,
        outflow_above_slack: oa,
        turbine_below_slack: tb,
        generation_below_slack: gb,
        min_outflow_rows: r_min_out,
        max_outflow_rows: r_max_out,
        min_turbine_rows: r_min_turb,
        min_generation_rows: r_min_gen,
        has_operational_violations: true,
    }
}

/// Read-only LP layout index map for one SDDP stage subproblem.
///
/// Computed once from `hydro_count` (N) and `max_par_order` (L), then shared
/// read-only across all threads for the duration of training. Most fields are
/// plain `usize` or `Range<usize>`; FPHA fields use `Vec` for variable-length
/// hydro lists.
///
/// See the [module-level documentation](self) for the full column and row
/// layout, and [`StageIndexer::new`] for the construction formulas.
///
/// Equipment column ranges (`turbine`, `spillage`, `diversion`, `thermal`,
/// `line_fwd`, `line_rev`, `deficit`, `excess`) are populated only when constructed via
/// [`StageIndexer::with_equipment`]. When constructed via [`StageIndexer::new`]
/// or [`StageIndexer::from_stage_template`], those ranges are all empty (`0..0`)
/// and `n_blks`, `n_thermals`, `n_lines`, `n_buses` are zero.
///
/// FPHA fields (`generation`, `fpha_hydro_indices`, `fpha_rows`) are also
/// populated only by [`StageIndexer::with_equipment`] when FPHA hydros are
/// present. They are empty when built via [`StageIndexer::new`] or when no FPHA
/// hydros exist.
#[derive(Debug, Clone)]
pub struct StageIndexer {
    /// Column range `[0, N)` for outgoing storage volumes.
    ///
    /// Each entry `storage[h]` is the column index of hydro plant `h`'s
    /// outgoing storage volume.
    pub storage: Range<usize>,

    /// Column range `[N, N*(1+L))` for AR lag variables.
    ///
    /// Lag variables are stored in lag-major order: all hydros for lag 0,
    /// then all hydros for lag 1, etc. The column index for hydro `h` at
    /// lag `l` (0-indexed, lag 0 = most recent) is:
    /// `inflow_lags.start + l * hydro_count + h`.
    pub inflow_lags: Range<usize>,

    /// Column range `[N*(2+L), N*(3+L))` for incoming storage volumes.
    ///
    /// Fixed by the storage-fixing constraints; transferred from the preceding
    /// stage's `storage` solution values.
    pub storage_in: Range<usize>,

    /// Column index `N*(3+L)` for the future cost variable (theta).
    ///
    /// Scalar: there is exactly one theta variable per stage LP.
    pub theta: usize,

    /// State-vector dimension.
    ///
    /// Without anticipated thermals: `N*(1+L)`.
    /// With `A` anticipated thermals at `K_max` lead stages each:
    /// `N*(1+L) + A*K_max`.
    ///
    /// The state vector consists of the `N` outgoing storage volumes followed
    /// by the `N*L` lag variables (and `A*K_max` anticipated-state slots when
    /// anticipated thermals are present). State transfer copies
    /// `primal[0..n_transfer]` (all but the oldest lag row).
    ///
    /// ## Semantic distinction
    ///
    /// `n_state` is the state-vector **dimension** used by cut storage and
    /// broadcast payloads. It is **not** a valid LP row index. Do not slice
    /// the LP row buffer as `[0, n_state)` — no state-fixing rows exist.
    /// Use [`StageIndexer::state_to_lp_incoming_column`] to resolve the
    /// column index for state-pinning and cut-subgradient extraction.
    pub n_state: usize,

    /// Row range for storage-fixing constraints.
    ///
    /// Always empty (`0..0`): state pinning uses column bounds
    /// (`set_col_bounds` on the incoming-state columns) rather than equality
    /// rows. The field is retained as a permanent sentinel for API stability;
    /// downstream consumers will observe an empty range.
    /// Use [`StageIndexer::state_to_lp_incoming_column`] to resolve the
    /// column index for state pinning and cut-subgradient extraction.
    pub storage_fixing: Range<usize>,

    /// Row range for AR lag-fixing constraints.
    ///
    /// Always empty (`0..0`): state pinning uses column bounds rather than
    /// equality rows. Retained as a permanent sentinel for the same reason
    /// as `storage_fixing`.
    pub lag_fixing: Range<usize>,

    /// Column range `[N*(1+L), N*(1+L) + n_anticipated*K_max)` for
    /// anticipated thermal commitment state slots.
    ///
    /// Ring-buffer block mirroring the inflow-lag layout: slot
    /// `k = 0..K_max` for anticipated plant `i = 0..n_anticipated` lives at
    /// column `anticipated_state.start + k * n_anticipated + i` (slot-major,
    /// plant-minor). Slot 0 holds the commitment maturing at the current
    /// stage; slot `K_max - 1` holds the commitment that matures
    /// `K_max - 1` stages from now.
    ///
    /// Empty (`0..0`) when `n_anticipated == 0` or when built via
    /// [`StageIndexer::new`].
    pub anticipated_state: Range<usize>,

    /// Row range for anticipated-state-fixing constraints.
    ///
    /// Always empty (`0..0`) regardless of `n_anticipated`: state pinning for
    /// anticipated thermals also uses column bounds. Retained as a permanent
    /// sentinel for API stability. The column range [`Self::anticipated_state`]
    /// carries the state-pinning semantics via `set_col_bounds`.
    pub anticipated_state_fixing: Range<usize>,

    /// Number of anticipated thermals (plants with
    /// `anticipated_config.is_some()`).
    ///
    /// Zero when no anticipated plants exist or when built via
    /// [`StageIndexer::new`].
    pub n_anticipated: usize,

    /// Maximum `lead_stages` across the anticipated thermals (`K_max`).
    ///
    /// Zero when `n_anticipated == 0` or when built via
    /// [`StageIndexer::new`].
    pub k_max: usize,

    /// Number of operating hydro plants (N).
    pub hydro_count: usize,

    /// Maximum PAR order across all operating hydros (L).
    ///
    /// All hydros use a uniform lag stride of `max_par_order`, enabling
    /// contiguous memory access and SIMD vectorisation over the lag dimension.
    pub max_par_order: usize,

    // ── Equipment column ranges ────────────────────────────────────────────
    // Populated only by `with_equipment`; empty (`0..0`) when built via `new`.
    /// Column range for turbined flow variables, one per (hydro, block) pair.
    ///
    /// Index for hydro `h`, block `b`: `turbine.start + h * n_blks + b`.
    /// Empty when built via [`StageIndexer::new`].
    pub turbine: Range<usize>,

    /// Column range for spillage variables, one per (hydro, block) pair.
    ///
    /// Index for hydro `h`, block `b`: `spillage.start + h * n_blks + b`.
    /// Empty when built via [`StageIndexer::new`].
    pub spillage: Range<usize>,

    /// Column range for diversion flow variables, one per (hydro, block) pair.
    ///
    /// Index for hydro `h`, block `b`: `diversion.start + h * n_blks + b`.
    /// Hydros without a diversion channel have bounds [0, 0]; the LP presolve
    /// eliminates them.
    /// Empty when built via [`StageIndexer::new`].
    pub diversion: Range<usize>,

    /// Column range for thermal generation variables, one per (thermal, block) pair.
    ///
    /// Index for thermal `t`, block `b`: `thermal.start + t * n_blks + b`.
    /// Empty when built via [`StageIndexer::new`].
    pub thermal: Range<usize>,

    /// Column range for anticipated-thermal commitment decisions, one stage-level
    /// column per anticipated plant active at stage 0.
    ///
    /// Decision variables are stage-level (NOT per-block): each column represents
    /// the MW the dispatcher commits for that plant at the current stage. The
    /// commitment is delivered `K_i` stages later (fishing constraint, added by
    /// a subsequent ticket).
    ///
    /// The `cobre-io` semantic validator enforces `K_i <= T` for every
    /// anticipated plant, so at stage 0 every anticipated plant is active and
    /// the range length equals [`Self::n_anticipated`].
    ///
    /// Per-stage gating is computed by
    /// [`anticipated_decision_active_at_stage`](Self::anticipated_decision_active_at_stage):
    /// at stage `t`, plant `i` is active iff `t + K_i <= T`. Inactive columns
    /// receive bounds `[0, 0]` in the LP build, matching the deficit-segment
    /// pattern.
    ///
    /// Empty (`0..0`) when `n_anticipated == 0` or when built via
    /// [`StageIndexer::new`].
    pub anticipated_decision: Range<usize>,

    /// Column range for the anticipated-thermal outgoing-state variables,
    /// one column per anticipated plant (stage-level, NOT per-block).
    ///
    /// Length: `n_anticipated`. Placed immediately after
    /// [`Self::anticipated_decision`] in the control region so that
    /// `anticipated_state_out.start == anticipated_decision.end`. Together
    /// with the `anticipated_state_out` definition row, this variable is
    /// pinned to the corresponding `anticipated_decision` column by an
    /// equality constraint, making it the correct target for cut-coefficient
    /// mapping via `state_to_lp_column`'s Equal branch.
    ///
    /// Empty (`0..0`) when `n_anticipated == 0` or when built via
    /// [`StageIndexer::new`].
    pub anticipated_state_out: Range<usize>,

    /// Per-plant `lead_stages` (`K_i`) for the anticipated thermals.
    ///
    /// Length [`Self::n_anticipated`]; indexed by anticipated-local position
    /// (0-indexed within the anticipated subset). Empty when built via
    /// [`StageIndexer::new`].
    pub anticipated_lead_stages: Vec<usize>,

    /// Mapping from anticipated-local position to global thermal index.
    ///
    /// Length [`Self::n_anticipated`]; `anticipated_thermal_indices[i]` is the
    /// position within `system.thermals[]` of the i-th anticipated plant.
    /// Parallel to [`Self::anticipated_lead_stages`]. Mirrors the FPHA
    /// `fpha_hydro_indices` pattern. Empty when built via
    /// [`StageIndexer::new`].
    pub anticipated_thermal_indices: Vec<usize>,

    /// Reverse map: global thermal position → anticipated-local index.
    ///
    /// Inverse of [`Self::anticipated_thermal_indices`]: given a system-level
    /// thermal position `sys_pos`, `anticipated_local_by_sys_pos[&sys_pos]`
    /// yields the anticipated-local index (0-indexed within the anticipated
    /// subset). Built once at construction for O(1) resolution in
    /// `resolve_anticipated_decision`. Empty when `n_anticipated == 0`.
    pub(crate) anticipated_local_by_sys_pos: HashMap<usize, usize>,

    /// Column range for forward line flow variables, one per (line, block) pair.
    ///
    /// Index for line `l`, block `b`: `line_fwd.start + l * n_blks + b`.
    /// Empty when built via [`StageIndexer::new`].
    pub line_fwd: Range<usize>,

    /// Column range for reverse line flow variables, one per (line, block) pair.
    ///
    /// Index for line `l`, block `b`: `line_rev.start + l * n_blks + b`.
    /// Empty when built via [`StageIndexer::new`].
    pub line_rev: Range<usize>,

    /// Column range for bus deficit variables, `B * S * K` columns total.
    ///
    /// S = `max_deficit_segments` (uniform stride across all buses).  For buses
    /// with fewer than S segments, the trailing segment slots have zero bounds
    /// and zero objective and are eliminated by the presolver.
    ///
    /// Index for bus `b_idx`, segment `s`, block `blk`:
    /// `deficit.start + b_idx * max_deficit_segments * n_blks + s * n_blks + blk`.
    ///
    /// Empty when built via [`StageIndexer::new`].
    pub deficit: Range<usize>,

    /// Maximum number of deficit segments across all buses (S).
    ///
    /// Used together with `deficit.start` to compute per-segment column indices.
    /// Set to `0` when built via [`StageIndexer::new`], `1` when built via
    /// [`StageIndexer::with_equipment`] (backward-compatible single-segment mode),
    /// and the true maximum when built via [`StageIndexer::with_equipment_and_evaporation`].
    pub max_deficit_segments: usize,

    /// Column range for bus excess variables, one per (bus, block) pair.
    ///
    /// Index for bus `b_idx`, block `blk`: `excess.start + b_idx * n_blks + blk`.
    /// Empty when built via [`StageIndexer::new`].
    pub excess: Range<usize>,

    /// Number of operating blocks per stage (K).
    ///
    /// Zero when built via [`StageIndexer::new`].
    pub n_blks: usize,

    /// Number of thermal units (T).
    ///
    /// Zero when built via [`StageIndexer::new`].
    pub n_thermals: usize,

    /// Number of transmission lines (`L_n`).
    ///
    /// Zero when built via [`StageIndexer::new`].
    pub n_lines: usize,

    /// Number of buses (B).
    ///
    /// Zero when built via [`StageIndexer::new`].
    pub n_buses: usize,

    /// Row range for water balance constraints, one per operating hydro.
    ///
    /// Index for hydro `h`: `water_balance.start + h`.
    /// The dual of this row gives the marginal value of water (water value).
    /// Empty when built via [`StageIndexer::new`].
    pub water_balance: Range<usize>,

    /// Row range for load balance constraints, one per (bus, block) pair.
    ///
    /// Index for bus `b_idx`, block `blk`: `load_balance.start + b_idx * n_blks + blk`.
    /// The RHS of these rows contains the load (MW) for each bus in each block.
    /// Empty when built via [`StageIndexer::new`].
    pub load_balance: Range<usize>,

    /// Column range for inflow non-negativity slack variables `sigma_inf_h`.
    ///
    /// One slack per operating hydro, appended after `excess` when the penalty
    /// method is active (`has_inflow_penalty == true`).  The slack is in m³/s;
    /// it absorbs negative inflow realisations and enters the water balance row
    /// with coefficient `+tau_total * M3S_TO_HM3`.
    ///
    /// Empty (`0..0`) when `has_inflow_penalty == false` or when built via
    /// [`StageIndexer::new`].
    pub inflow_slack: Range<usize>,

    /// Row range for inflow non-negativity constraint rows.
    ///
    /// Currently unused as a separate constraint block — the slack appears
    /// directly in the water balance row.  Reserved for future formulations
    /// that add an explicit `sigma_inf_h + a_h >= 0` row.
    ///
    /// Empty (`0..0`) in this implementation.
    pub inflow_slack_rows: Range<usize>,

    /// Whether inflow non-negativity penalty slack columns are present.
    ///
    /// `true` when `build_stage_templates` was called with an
    /// [`InflowNonNegativityMethod`](crate::inflow_method::InflowNonNegativityMethod)
    /// whose `has_slack_columns()` returns `true` and `n_hydros > 0`.
    /// `false` otherwise (including when built via [`StageIndexer::new`]).
    pub has_inflow_penalty: bool,

    // ── FPHA column and row ranges ─────────────────────────────────────────
    // Populated only by `with_equipment`; empty when built via `new`.
    /// Column range for FPHA generation variables, one per (`fpha_hydro`, block) pair.
    ///
    /// Index for FPHA hydro at local position `i`, block `b`:
    /// `generation.start + i * n_blks + b`.
    /// Empty when no FPHA hydros exist or when built via [`StageIndexer::new`].
    pub generation: Range<usize>,

    /// Number of FPHA hydros in this stage.
    ///
    /// Zero when built via [`StageIndexer::new`].
    pub n_fpha_hydros: usize,

    /// Mapping from FPHA local index to system hydro index.
    ///
    /// `fpha_hydro_indices[i]` is the system-level hydro position for FPHA hydro `i`.
    /// Empty when no FPHA hydros exist or when built via [`StageIndexer::new`].
    pub fpha_hydro_indices: Vec<usize>,

    /// FPHA constraint row ranges per FPHA hydro.
    ///
    /// `fpha_rows[i]` is the [`FphaRowRange`] for FPHA hydro at local position `i`.
    /// Empty when no FPHA hydros exist or when built via [`StageIndexer::new`].
    pub fpha_rows: Vec<FphaRowRange>,

    // ── Evaporation column and row indices ─────────────────────────────────
    // Populated only by `with_equipment`; empty when built via `new`.
    /// Number of hydros with linearized evaporation at this stage.
    ///
    /// Zero when built via [`StageIndexer::new`] or when no evaporation hydros exist.
    pub n_evap_hydros: usize,

    /// Mapping from evaporation local index to system hydro index.
    ///
    /// `evap_hydro_indices[i]` is the system-level hydro position for evaporation hydro `i`.
    /// Empty when no evaporation hydros exist or when built via [`StageIndexer::new`].
    pub evap_hydro_indices: Vec<usize>,

    /// Per-evaporation-hydro column and row indices.
    ///
    /// `evap_indices[i]` is the [`EvaporationIndices`] for evaporation hydro at local
    /// position `i`.  Empty when no evaporation hydros exist or when built via
    /// [`StageIndexer::new`].
    pub evap_indices: Vec<EvaporationIndices>,

    // ── Withdrawal slack column ranges ─────────────────────────────────────
    // Populated only by `with_equipment_and_evaporation`; empty when built via `new`.
    /// Column range for under-withdrawal slack (withdrew less than target).
    ///
    /// One slack per operating hydro, appended after the evaporation columns.
    /// Columns are stage-level (not per-block); the slack absorbs violations of
    /// the minimum water-withdrawal flow constraint.
    ///
    /// Allocated whenever `hydro_count > 0`, matching the `inflow_slack` pattern.
    /// Empty (`0..0`) when `hydro_count == 0` or when built via
    /// [`StageIndexer::new`].
    /// Layout: `withdrawal_slack_neg.start + h_idx`.
    pub withdrawal_slack_neg: Range<usize>,

    /// Column range for over-withdrawal slack (withdrew more than target).
    ///
    /// One slack per operating hydro, immediately following `withdrawal_slack_neg`.
    /// Layout: `withdrawal_slack_pos.start + h_idx`.
    pub withdrawal_slack_pos: Range<usize>,

    /// Whether withdrawal slack columns are present.
    ///
    /// `true` when `with_equipment_and_evaporation` was called with
    /// `hydro_count > 0`.  `false` otherwise (including when built via
    /// [`StageIndexer::new`]).
    pub has_withdrawal: bool,

    // ── Operational violation slack column ranges ─────────────────────────
    // Populated only by the full build path; empty (`0..0`) when built via `new`.
    /// Column range for outflow-below violation slacks, one per hydro per block.
    ///
    /// `outflow_below_slack.start + h * n_blks + blk` is the column for hydro `h`,
    /// block `blk`.  Empty (`0..0`) when built via [`StageIndexer::new`].
    pub outflow_below_slack: Range<usize>,

    /// Column range for outflow-above violation slacks, one per hydro per block.
    ///
    /// `outflow_above_slack.start + h * n_blks + blk` is the column for hydro `h`,
    /// block `blk`.  Empty (`0..0`) when built via [`StageIndexer::new`].
    pub outflow_above_slack: Range<usize>,

    /// Column range for turbine-below violation slacks, one per hydro per block.
    ///
    /// `turbine_below_slack.start + h * n_blks + blk` is the column for hydro `h`,
    /// block `blk`.  Empty (`0..0`) when built via [`StageIndexer::new`].
    pub turbine_below_slack: Range<usize>,

    /// Column range for generation-below violation slacks, one per hydro per block.
    ///
    /// `generation_below_slack.start + h * n_blks + blk` is the column for hydro `h`,
    /// block `blk`.  Empty (`0..0`) when built via [`StageIndexer::new`].
    pub generation_below_slack: Range<usize>,

    // ── Operational violation constraint row ranges ────────────────────────
    // Populated only by the full build path; empty (`0..0`) when built via `new`.
    /// Row range for min-outflow constraint rows, one per hydro per block.
    ///
    /// `min_outflow_rows.start + h * n_blks + blk` is the row for hydro `h`,
    /// block `blk`.  Empty (`0..0`) when built via [`StageIndexer::new`].
    pub min_outflow_rows: Range<usize>,

    /// Row range for max-outflow constraint rows, one per hydro per block.
    ///
    /// `max_outflow_rows.start + h * n_blks + blk` is the row for hydro `h`,
    /// block `blk`.  Empty (`0..0`) when built via [`StageIndexer::new`].
    pub max_outflow_rows: Range<usize>,

    /// Row range for min-turbine constraint rows, one per hydro per block.
    ///
    /// `min_turbine_rows.start + h * n_blks + blk` is the row for hydro `h`,
    /// block `blk`.  Empty (`0..0`) when built via [`StageIndexer::new`].
    pub min_turbine_rows: Range<usize>,

    /// Row range for min-generation constraint rows, one per hydro per block.
    ///
    /// `min_generation_rows.start + h * n_blks + blk` is the row for hydro `h`,
    /// block `blk`.  Empty (`0..0`) when built via [`StageIndexer::new`].
    pub min_generation_rows: Range<usize>,

    /// Row range for anticipated-thermal fishing constraints.
    ///
    /// Empty (`0..0`) in the canonical layout placeholder; per-stage active
    /// row indices are always `n_anticipated` under the always-active
    /// predicate and are accessed via
    /// [`anticipated_fishing_active_at_stage`](Self::anticipated_fishing_active_at_stage).
    ///
    /// The fishing constraint reads:
    /// `gt_i^(t) - anticipated_state[slot=0, plant=i] = 0`
    /// where the dual on this row carries the cut subgradient w.r.t.
    /// the slot read by the fishing constraint at delivery.
    pub anticipated_fishing: Range<usize>,

    /// First row index of the anticipated-fishing block.
    ///
    /// Equal to `min_generation_rows.end` when operational violations are
    /// active, or to `evap_rows_end` (= `fpha_row_cursor + n_evap_hydros`)
    /// when they are not. Zero when built via [`StageIndexer::new`].
    ///
    /// Per-stage fishing row indices are computed as
    /// `lp_row = anticipated_fishing_start + local_idx_at_stage`.
    pub anticipated_fishing_start: usize,

    /// Whether operational violation slack columns are present.
    ///
    /// `true` when the full build path was used with `hydro_count > 0`.
    /// `false` otherwise (including when built via [`StageIndexer::new`]).
    pub has_operational_violations: bool,

    // ── Generic constraint row and column ranges ────────────────────────────
    // Populated only by `StageLayout::new` in lp_builder via the full build
    // path; empty (`0..0`, 0) when built via [`StageIndexer::new`].
    /// Row range for generic constraint rows (one per active `(constraint, block)` pair).
    ///
    /// Rows are placed after evaporation rows (the last row region before
    /// generic constraints).  Empty (`0..0`) when no generic constraints are
    /// active at this stage or when built via [`StageIndexer::new`].
    pub generic_constraint_rows: Range<usize>,

    /// Column range for generic constraint slack variables.
    ///
    /// Columns are placed after withdrawal slack columns (the last column region
    /// before generic constraint slacks).  The number of columns equals the number
    /// of active rows when `slack.enabled = true` and sense is `<=` or `>=`, or
    /// twice the number of active rows when sense is `==` (positive and negative
    /// violation slacks).  Empty (`0..0`) when no slack is needed or when built
    /// via [`StageIndexer::new`].
    pub generic_constraint_slack: Range<usize>,

    /// Number of active generic constraint rows contributed at this stage.
    ///
    /// Zero when no generic constraints are active or when built via
    /// [`StageIndexer::new`].
    pub n_generic_constraints_active: usize,

    // ── NCS column range ──────────────────────────────────────────────────
    // Populated only by `StageLayout::new` in lp_builder; empty when built
    // via `new`, `with_equipment`, or `from_stage_template`.
    /// Column range for NCS generation variables, one per (ncs, block) pair.
    ///
    /// Index for NCS `r`, block `b`: `ncs_generation.start + r * n_blks + b`.
    /// Empty when built via [`StageIndexer::new`] or when no NCS entities are active.
    pub ncs_generation: Range<usize>,

    // ── Z-inflow column and row ranges ────────────────────────────────────
    // Populated by all constructors.  The z_inflow columns are auxiliary
    // (NOT state variables); their primal values give the realized total
    // inflow Z_t per hydro after solving.
    /// Column range for realized-inflow variables `z_h`, one per hydro.
    ///
    /// These free columns (lower = -inf, upper = +inf, zero cost) represent the
    /// total natural inflow `Z_t_h` at each hydro, defined by the z-inflow
    /// equality constraints. After solving, `primal[z_inflow.start + h]` gives
    /// the realized inflow for hydro h.
    ///
    /// Empty when `hydro_count == 0`.
    pub z_inflow: Range<usize>,

    /// Row range for z-inflow definition constraints, one per hydro.
    ///
    /// Each row defines: `z_h - sum_l[psi_l * lag_in[h,l]] = base_h + sigma_h * eta_h`
    /// The RHS is noise-patched (Category 5 in `PatchBuffer`).
    ///
    /// Empty when `hydro_count == 0`.
    pub z_inflow_rows: Range<usize>,

    /// Row index of the first z-inflow definition constraint.
    ///
    /// Used by `PatchBuffer::fill_z_inflow_patches` as the base offset for
    /// Category 5 patches. Equal to `z_inflow_rows.start`.
    pub z_inflow_row_start: usize,

    /// Indices of state dimensions whose cut coefficients can be nonzero.
    ///
    /// Storage indices `[0, N)` are always included. Lag indices `[N, N*(1+L))`
    /// are included only when `lag < actual_ar_order[hydro]`. Hydros with AR
    /// order < `max_par_order` have padded lag slots whose duals are
    /// structurally zero.
    ///
    /// When empty (default), callers should treat all `n_state` indices as
    /// nonzero (dense path). Use [`set_nonzero_mask`](Self::set_nonzero_mask) to
    /// populate after construction when per-hydro AR orders are available.
    pub nonzero_state_indices: Vec<usize>,

    /// Precomputed `state_to_lp_column(j)` for every `j ∈ [0, n_state)`.
    ///
    /// Built once after the state layout is finalized; read on the
    /// forward-pass cut-row hot path. Empty until finalized.
    pub state_to_lp_column_map: Vec<usize>,
}

/// Equipment counts for constructing a [`StageIndexer`].
///
/// Groups the entity counts that determine the LP column layout for a single stage.
pub struct EquipmentCounts {
    /// Number of hydro plants.
    pub hydro_count: usize,
    /// Maximum PAR model order across all hydros.
    pub max_par_order: usize,
    /// Number of thermal units.
    pub n_thermals: usize,
    /// Number of transmission lines.
    pub n_lines: usize,
    /// Number of buses.
    pub n_buses: usize,
    /// Number of demand blocks in the stage.
    pub n_blks: usize,
    /// Whether to include inflow penalty slack columns.
    pub has_inflow_penalty: bool,
    /// Maximum number of deficit segments across all buses.
    pub max_deficit_segments: usize,
    /// Number of anticipated thermals (`anticipated_config.is_some()`).
    pub n_anticipated: usize,
    /// Maximum `lead_stages` across the anticipated thermals.
    pub k_max: usize,
    /// Per-plant `lead_stages` (`K_i`) for the anticipated thermals.
    ///
    /// Length must equal `n_anticipated`. The maximum entry (when non-empty)
    /// must equal `k_max`. Pass-through to
    /// [`StageIndexer::anticipated_lead_stages`].
    pub anticipated_lead_stages: Vec<usize>,
    /// Mapping from anticipated-local position to global thermal index.
    ///
    /// Length must equal `n_anticipated`. Parallel to `anticipated_lead_stages`.
    /// Pass-through to [`StageIndexer::anticipated_thermal_indices`].
    pub anticipated_thermal_indices: Vec<usize>,
}

/// FPHA (Piecewise-linear Hydro Approximation) column layout.
///
/// Groups the per-hydro FPHA data needed for column layout computation.
pub struct FphaColumnLayout {
    /// Indices of hydros using FPHA production models.
    pub hydro_indices: Vec<usize>,
    /// Number of FPHA planes for each hydro in `hydro_indices`.
    ///
    /// Must have the same length as `hydro_indices`.
    pub planes_per_hydro: Vec<usize>,
}

/// Evaporation configuration for hydro plants.
pub struct EvapConfig {
    /// Indices of hydros with evaporation modeling enabled.
    pub hydro_indices: Vec<usize>,
}

impl StageIndexer {
    /// Construct a [`StageIndexer`] from `hydro_count` (N) and `max_par_order` (L).
    ///
    /// All index ranges are computed from N and L using the formulas in
    /// Solver Abstraction SS2.1–SS2.2. The constructor is infallible;
    /// validation of N and L is the caller's responsibility.
    ///
    /// Equipment column ranges (`turbine`, `spillage`, `thermal`, `line_fwd`,
    /// `line_rev`, `deficit`, `excess`) are all empty (`0..0`) and equipment
    /// counts (`n_blks`, `n_thermals`, `n_lines`, `n_buses`) are zero. Use
    /// [`StageIndexer::with_equipment`] to populate them.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_sddp::indexer::StageIndexer;
    ///
    /// // Worked example from spec SS5.5.3: N = 3, L = 2
    /// let idx = StageIndexer::new(3, 2);
    /// assert_eq!(idx.storage,   0..3);
    /// assert_eq!(idx.inflow_lags, 3..9);
    /// assert_eq!(idx.z_inflow,  9..12);
    /// assert_eq!(idx.storage_in, 12..15);
    /// assert_eq!(idx.theta,   15);
    /// assert_eq!(idx.n_state,  9);
    /// // State-fixing row ranges are permanent empty sentinels.
    /// assert_eq!(idx.storage_fixing, 0..0);
    /// assert_eq!(idx.lag_fixing, 0..0);
    /// // Equipment ranges are empty when built via `new`.
    /// assert!(idx.turbine.is_empty());
    /// assert_eq!(idx.n_blks, 0);
    /// // z_inflow rows start at row 0.
    /// assert_eq!(idx.z_inflow_rows, 0..3);
    /// assert_eq!(idx.z_inflow_row_start, 0);
    /// ```
    #[must_use]
    pub fn new(hydro_count: usize, max_par_order: usize) -> Self {
        let n = hydro_count;
        let l = max_par_order;

        let storage = 0..n;
        let inflow_lags = n..n * (1 + l);

        // z_inflow columns at fixed offset N*(1+L), immediately after lags
        // and before storage_in. This makes z_inflow stage-invariant.
        let z_inflow_start = n * (1 + l);
        let z_inflow = z_inflow_start..z_inflow_start + n;

        let storage_in = n * (2 + l)..n * (3 + l);
        let theta = n * (3 + l);
        let n_state = n * (1 + l);

        // State fixing uses column bounds; the row ranges are permanent empty
        // sentinels. The column ranges (storage, inflow_lags, anticipated_state)
        // carry the state-pinning semantics via set_col_bounds.
        let storage_fixing = 0..0;
        let lag_fixing = 0..0;

        // z_inflow rows start at row 0 — the first row block in the LP.
        let z_inflow_start_row = 0_usize;
        let z_inflow_rows = z_inflow_start_row..z_inflow_start_row + n;
        let z_inflow_row_start = z_inflow_start_row;

        Self {
            storage,
            inflow_lags,
            storage_in,
            theta,
            n_state,
            storage_fixing,
            lag_fixing,
            // Anticipated state block is empty when built via `new`;
            // callers that need it must use `with_equipment_and_evaporation`.
            anticipated_state: 0..0,
            anticipated_state_fixing: 0..0,
            n_anticipated: 0,
            k_max: 0,
            hydro_count,
            max_par_order,
            // Equipment ranges are empty until `with_equipment` is called.
            turbine: 0..0,
            spillage: 0..0,
            diversion: 0..0,
            thermal: 0..0,
            // Anticipated decision and state-out blocks are empty when built via `new`.
            anticipated_decision: 0..0,
            anticipated_state_out: 0..0,
            anticipated_lead_stages: Vec::new(),
            anticipated_thermal_indices: Vec::new(),
            anticipated_local_by_sys_pos: HashMap::new(),
            line_fwd: 0..0,
            line_rev: 0..0,
            deficit: 0..0,
            max_deficit_segments: 0,
            excess: 0..0,
            n_blks: 0,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            water_balance: 0..0,
            load_balance: 0..0,
            inflow_slack: 0..0,
            inflow_slack_rows: 0..0,
            has_inflow_penalty: false,
            generation: 0..0,
            n_fpha_hydros: 0,
            fpha_hydro_indices: Vec::new(),
            fpha_rows: Vec::new(),
            n_evap_hydros: 0,
            evap_hydro_indices: Vec::new(),
            evap_indices: Vec::new(),
            withdrawal_slack_neg: 0..0,
            withdrawal_slack_pos: 0..0,
            has_withdrawal: false,
            outflow_below_slack: 0..0,
            outflow_above_slack: 0..0,
            turbine_below_slack: 0..0,
            generation_below_slack: 0..0,
            min_outflow_rows: 0..0,
            max_outflow_rows: 0..0,
            min_turbine_rows: 0..0,
            min_generation_rows: 0..0,
            anticipated_fishing: 0..0,
            anticipated_fishing_start: 0,
            has_operational_violations: false,
            generic_constraint_rows: 0..0,
            generic_constraint_slack: 0..0,
            n_generic_constraints_active: 0,
            ncs_generation: 0..0,
            z_inflow,
            z_inflow_rows,
            z_inflow_row_start,
            nonzero_state_indices: Vec::new(),
            state_to_lp_column_map: Vec::new(),
        }
    }

    /// Construct a [`StageIndexer`] with full equipment column ranges.
    ///
    /// Computes both the state-variable ranges (identical to [`StageIndexer::new`])
    /// and the equipment decision-variable ranges that follow `theta` in the LP.
    ///
    /// The equipment column layout matches `lp_builder.rs` exactly:
    ///
    /// ```text
    /// decision_start      = theta + 1
    /// turbine_start       = decision_start
    /// spillage_start      = turbine_start  + n_hydros * n_blks
    /// diversion_start     = spillage_start + n_hydros * n_blks
    /// thermal_start       = diversion_start + n_hydros * n_blks
    /// line_fwd_start      = thermal_start  + n_thermals * n_blks
    /// line_rev_start      = line_fwd_start + n_lines * n_blks
    /// deficit_start       = line_rev_start + n_lines * n_blks
    /// excess_start        = deficit_start  + n_buses * max_deficit_segments * n_blks
    /// inflow_slack_start  = excess_end  (only when has_inflow_penalty && hydro_count > 0)
    /// generation_start    = inflow_slack_end  (FPHA generation columns)
    /// evap_start          = generation_end  (3 columns per evaporation hydro, stage-level)
    /// ```
    ///
    /// FPHA generation columns come immediately after `inflow_slack` (or after
    /// `excess` when `has_inflow_penalty == false`), one column per FPHA hydro
    /// per block.  FPHA constraint rows are placed after `load_balance`.
    ///
    /// Evaporation columns (3 per evaporation hydro: `Q_ev`, `f_evap_plus`,
    /// `f_evap_minus`) are stage-level (not per-block) and come immediately after
    /// the FPHA generation columns.  Evaporation rows (1 per evaporation hydro)
    /// are placed after FPHA rows.
    ///
    /// # Notes
    ///
    /// This constructor assumes a **uniform block count across all stages**
    /// (i.e., all stages have the same `n_blks`). For the minimal viable
    /// solver this assumption holds; stages with heterogeneous block counts
    /// would require a per-stage indexer.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_sddp::indexer::StageIndexer;
    ///
    /// // N=1 hydro, L=0 lags, T=2 thermals, L_n=1 line, B=2 buses, K=1 block, no penalty
    /// // theta = N*(3+L) = 1*(3+0) = 3
    /// // decision_start = 4
    /// // turbine:    4..5   (1 hydro * 1 block)
    /// // spillage:   5..6   (1 hydro * 1 block)
    /// // diversion:  6..7   (1 hydro * 1 block)
    /// // thermal:    7..9   (2 thermals * 1 block)
    /// // line_fwd:   9..10  (1 line * 1 block)
    /// // line_rev:  10..11  (1 line * 1 block)
    /// // deficit:   11..13  (2 buses * 1 block)
    /// // excess:    13..15  (2 buses * 1 block)
    /// let counts = cobre_sddp::indexer::EquipmentCounts {
    ///     hydro_count: 1, max_par_order: 0, n_thermals: 2, n_lines: 1,
    ///     n_buses: 2, n_blks: 1, has_inflow_penalty: false, max_deficit_segments: 1,
    ///     n_anticipated: 0, k_max: 0,
    ///     anticipated_lead_stages: vec![], anticipated_thermal_indices: vec![],
    /// };
    /// let fpha = cobre_sddp::indexer::FphaColumnLayout { hydro_indices: vec![], planes_per_hydro: vec![] };
    /// let idx = StageIndexer::with_equipment(&counts, &fpha);
    /// assert_eq!(idx.turbine,    4..5);
    /// assert_eq!(idx.spillage,   5..6);
    /// assert_eq!(idx.diversion,  6..7);
    /// assert_eq!(idx.thermal,    7..9);
    /// assert_eq!(idx.line_fwd,   9..10);
    /// assert_eq!(idx.line_rev,  10..11);
    /// assert_eq!(idx.deficit,   11..13);
    /// assert_eq!(idx.excess,    13..15);
    /// assert!(idx.inflow_slack.is_empty());
    /// assert!(idx.generation.is_empty());
    /// assert_eq!(idx.n_blks, 1);
    /// assert_eq!(idx.n_thermals, 2);
    /// assert_eq!(idx.n_lines, 1);
    /// assert_eq!(idx.n_buses, 2);
    /// ```
    #[must_use]
    pub fn with_equipment(counts: &EquipmentCounts, fpha: &FphaColumnLayout) -> Self {
        Self::with_equipment_and_evaporation(
            counts,
            fpha,
            &EvapConfig {
                hydro_indices: vec![],
            },
        )
    }

    /// Construct a [`StageIndexer`] with full equipment column ranges and evaporation.
    ///
    /// Extends [`StageIndexer::with_equipment`] with evaporation hydro indices.
    /// Evaporation columns (3 per evaporation hydro: `Q_ev`, `f_evap_plus`,
    /// `f_evap_minus`) are stage-level and placed after FPHA generation columns.
    /// Evaporation rows (1 per evaporation hydro) are placed after FPHA rows.
    ///
    /// # Arguments
    ///
    /// - `counts` — equipment counts grouped into [`EquipmentCounts`]
    /// - `fpha` — FPHA column layout grouped into [`FphaColumnLayout`]
    /// - `evap` — evaporation configuration grouped into [`EvapConfig`]
    ///
    /// When `evap.hydro_indices` is empty this produces the same result as
    /// [`StageIndexer::with_equipment`].
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn with_equipment_and_evaporation(
        counts: &EquipmentCounts,
        fpha: &FphaColumnLayout,
        evap: &EvapConfig,
    ) -> Self {
        debug_assert!(
            counts.n_anticipated == 0 || counts.k_max >= 1,
            "k_max must be >= 1 when n_anticipated > 0"
        );
        debug_assert_eq!(
            counts.anticipated_lead_stages.len(),
            counts.n_anticipated,
            "anticipated_lead_stages length must equal n_anticipated"
        );
        debug_assert_eq!(
            counts.anticipated_thermal_indices.len(),
            counts.n_anticipated,
            "anticipated_thermal_indices length must equal n_anticipated"
        );
        debug_assert!(
            counts.n_anticipated == 0
                || counts
                    .anticipated_lead_stages
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0)
                    == counts.k_max,
            "k_max must equal max(anticipated_lead_stages)"
        );

        let hydro_count = counts.hydro_count;
        let max_par_order = counts.max_par_order;
        let n_thermals = counts.n_thermals;
        let n_lines = counts.n_lines;
        let n_buses = counts.n_buses;
        let n_blks = counts.n_blks;
        let has_inflow_penalty = counts.has_inflow_penalty;
        let n_anticipated = counts.n_anticipated;
        let k_max = counts.k_max;
        let n_ant_state = n_anticipated * k_max;
        let fpha_hydro_indices = fpha.hydro_indices.clone();
        let evap_hydro_indices = evap.hydro_indices.clone();

        debug_assert_eq!(
            fpha_hydro_indices.len(),
            fpha.planes_per_hydro.len(),
            "fpha_hydro_indices and fpha_planes_per_hydro must have equal length"
        );

        let base = Self::new(hydro_count, max_par_order);

        // Shift state-block downstream offsets by n_ant_state to make room for
        // the anticipated_state block placed immediately after `inflow_lags`.
        // The new layout:
        //   storage           = 0..N
        //   inflow_lags       = N..N*(1+L)
        //   anticipated_state = N*(1+L)..N*(1+L) + n_ant_state
        //   z_inflow          = + N
        //   storage_in        = + N
        //   theta             = N*(3+L) + n_ant_state
        //   n_state           = N*(1+L) + n_ant_state
        let n_state = hydro_count * (1 + max_par_order) + n_ant_state;
        let anticipated_state_start = hydro_count * (1 + max_par_order);
        let anticipated_state_end = anticipated_state_start + n_ant_state;
        // When the block is empty, normalise the public range to `0..0` to
        // mirror the project-wide convention used by `inflow_slack`,
        // `withdrawal_slack`, and the operational-violation ranges. Use
        // `anticipated_state_end` (not `anticipated_state.end`) for
        // downstream layout arithmetic so the shift is preserved even when
        // the public range collapses to `0..0`.
        let anticipated_state = if n_ant_state > 0 {
            anticipated_state_start..anticipated_state_end
        } else {
            0..0
        };
        // Anticipated-state fixing uses column bounds; row range is a permanent empty sentinel.
        let anticipated_state_fixing = 0..0;
        let z_inflow_start = anticipated_state_end;
        let z_inflow = z_inflow_start..z_inflow_start + hydro_count;
        let storage_in_start = z_inflow.end;
        let storage_in = storage_in_start..storage_in_start + hydro_count;
        let theta = storage_in.end;
        // z_inflow rows start at row 0 (ROW start, not column start).
        let z_inflow_start_row = 0_usize;
        let z_inflow_rows = z_inflow_start_row..z_inflow_start_row + hydro_count;
        let z_inflow_row_start = z_inflow_start_row;

        let decision_start = theta + 1;

        let turbine_start = decision_start;
        let spillage_start = turbine_start + hydro_count * n_blks;
        let diversion_start = spillage_start + hydro_count * n_blks;
        let thermal_start = diversion_start + hydro_count * n_blks;
        let thermal_end = thermal_start + n_thermals * n_blks;
        // Anticipated-decision and anticipated-state-out columns sit between
        // `thermal` and `line_fwd`.  Per Decision 3, every anticipated plant
        // has K_i <= T, so at stage 0 (the canonical stage) all `n_anticipated`
        // columns are active.  Per-stage gating is bound-driven downstream via
        // `anticipated_decision_active_at_stage`; the column count is constant.
        //
        // Control-region layout (equipment side):
        //   anticipated_decision   = [thermal_end, thermal_end + A)
        //   anticipated_state_out  = [thermal_end + A, thermal_end + 2*A)
        //   line_fwd               = [thermal_end + 2*A, …)
        let anticipated_decision_start = thermal_end;
        let anticipated_decision_end = thermal_end + n_anticipated;
        let anticipated_decision = if n_anticipated > 0 {
            anticipated_decision_start..anticipated_decision_end
        } else {
            0..0
        };
        let anticipated_state_out_start = anticipated_decision_end;
        let anticipated_state_out_end = anticipated_state_out_start + n_anticipated;
        let anticipated_state_out = if n_anticipated > 0 {
            anticipated_state_out_start..anticipated_state_out_end
        } else {
            0..0
        };
        let line_fwd_start = anticipated_state_out_end;
        let line_rev_start = line_fwd_start + n_lines * n_blks;
        let deficit_start = line_rev_start + n_lines * n_blks;
        let max_deficit_segments = counts.max_deficit_segments; // also assigned to Self
        let excess_start = deficit_start + n_buses * max_deficit_segments * n_blks;
        let excess_end = excess_start + n_buses * n_blks;

        // Inflow slack columns are appended after excess when penalty is active.
        let (inflow_slack, active_penalty) =
            build_inflow_slack_range(has_inflow_penalty, hydro_count, excess_end);

        // FPHA generation columns: after inflow_slack (or excess when no penalty).
        let n_fpha_hydros = fpha_hydro_indices.len();
        let generation_start = if active_penalty {
            inflow_slack.end
        } else {
            excess_end
        };
        let generation_end = generation_start + n_fpha_hydros * n_blks;
        let generation = if n_fpha_hydros > 0 {
            generation_start..generation_end
        } else {
            0..0
        };

        // Evaporation columns: 3 per evap hydro (stage-level), after FPHA generation columns.
        // Layout: Q_ev_col = evap_start+i*3, f_evap_plus = +1, f_evap_minus = +2.
        let n_evap_hydros = evap_hydro_indices.len();
        let evap_col_start = generation_end;

        // z_inflow rows start at row 0 (z_inflow_start_row declared above);
        // water_balance follows z_inflow at row hydro_count.
        let water_balance_start = z_inflow_start_row + hydro_count;
        let load_balance_start = water_balance_start + hydro_count;
        let load_balance_end = load_balance_start + n_buses * n_blks;

        let (fpha_rows, fpha_row_cursor) =
            Self::build_fpha_rows(&fpha.planes_per_hydro, n_blks, load_balance_end);

        let evap_indices_vec =
            Self::build_evap_indices(n_evap_hydros, evap_col_start, fpha_row_cursor);
        let evap_col_end = evap_col_start + n_evap_hydros * 3;
        let (withdrawal_slack_neg, withdrawal_slack_pos, has_withdrawal) = if hydro_count > 0 {
            let neg = evap_col_end..evap_col_end + hydro_count;
            let pos = neg.end..neg.end + hydro_count;
            (neg, pos, true)
        } else {
            (0..0, 0..0, false)
        };

        // Operational violation slack columns: 4 families * (hydro_count * n_blks).
        // Columns are placed after withdrawal slack; rows after evaporation rows.
        let evap_rows_end = fpha_row_cursor + n_evap_hydros;
        let ws_end = withdrawal_slack_pos.end;
        let op = build_oper_violation_ranges(hydro_count, n_blks, ws_end, evap_rows_end);

        // Anticipated-fishing rows are placed after the operational-violation
        // rows when those are active, otherwise directly after the evaporation
        // rows. The stage-0 canonical layout stores a zero-length range; the
        // per-stage template populates
        // `anticipated_fishing_start + local_idx_at_stage`.
        let fishing_start = if op.has_operational_violations {
            op.min_generation_rows.end
        } else {
            evap_rows_end
        };

        // z_inflow / storage_in / theta / n_state are computed locally above
        // to absorb the anticipated_state shift; the remaining fields
        // (storage, inflow_lags, storage_fixing, lag_fixing, hydro_count,
        // max_par_order, nonzero_state_indices) are inherited from `base`.

        Self {
            turbine: turbine_start..spillage_start,
            spillage: spillage_start..diversion_start,
            diversion: diversion_start..thermal_start,
            thermal: thermal_start..thermal_end,
            anticipated_decision,
            anticipated_state_out,
            anticipated_lead_stages: counts.anticipated_lead_stages.clone(),
            anticipated_local_by_sys_pos: counts
                .anticipated_thermal_indices
                .iter()
                .enumerate()
                .map(|(local, &sys_pos)| (sys_pos, local))
                .collect(),
            anticipated_thermal_indices: counts.anticipated_thermal_indices.clone(),
            line_fwd: line_fwd_start..line_rev_start,
            line_rev: line_rev_start..deficit_start,
            deficit: deficit_start..excess_start,
            max_deficit_segments,
            excess: excess_start..excess_end,
            n_blks,
            n_thermals,
            n_lines,
            n_buses,
            water_balance: water_balance_start..water_balance_start + hydro_count,
            load_balance: load_balance_start..load_balance_end,
            inflow_slack,
            inflow_slack_rows: 0..0,
            has_inflow_penalty: active_penalty,
            generation,
            n_fpha_hydros,
            fpha_hydro_indices,
            fpha_rows,
            n_evap_hydros,
            evap_hydro_indices,
            evap_indices: evap_indices_vec,
            withdrawal_slack_neg,
            withdrawal_slack_pos,
            has_withdrawal,
            outflow_below_slack: op.outflow_below_slack,
            outflow_above_slack: op.outflow_above_slack,
            turbine_below_slack: op.turbine_below_slack,
            generation_below_slack: op.generation_below_slack,
            min_outflow_rows: op.min_outflow_rows,
            max_outflow_rows: op.max_outflow_rows,
            min_turbine_rows: op.min_turbine_rows,
            min_generation_rows: op.min_generation_rows,
            anticipated_fishing: fishing_start..fishing_start,
            anticipated_fishing_start: fishing_start,
            has_operational_violations: op.has_operational_violations,
            // Shifted state-block fields (recomputed to absorb n_ant_state).
            anticipated_state,
            anticipated_state_fixing,
            n_anticipated,
            k_max,
            n_state,
            storage_in,
            theta,
            z_inflow,
            z_inflow_rows,
            z_inflow_row_start,
            // Built empty here; filled by `finalize_state_column_map` once the
            // (shifted) state layout above is final.
            state_to_lp_column_map: Vec::new(),
            // Remaining state-block fields are unchanged; inherit from base.
            ..base
        }
    }

    /// Return the [`EvaporationIndices`] for the evaporation hydro at local position `local_idx`.
    ///
    /// `local_idx` is the position within the evaporation hydro list (0-indexed).
    /// Use `evap_hydro_indices[local_idx]` to map to the system-level hydro position.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `local_idx >= n_evap_hydros`.
    #[must_use]
    pub fn evap_indices(&self, local_idx: usize) -> &EvaporationIndices {
        debug_assert!(
            local_idx < self.n_evap_hydros,
            "evap local index {local_idx} out of bounds (n_evap_hydros = {})",
            self.n_evap_hydros
        );
        &self.evap_indices[local_idx]
    }

    /// Build FPHA constraint row ranges from per-hydro plane counts.
    fn build_fpha_rows(
        planes_per_hydro: &[usize],
        n_blks: usize,
        start_row: usize,
    ) -> (Vec<FphaRowRange>, usize) {
        let mut rows = Vec::with_capacity(planes_per_hydro.len());
        let mut cursor = start_row;
        for &planes in planes_per_hydro {
            rows.push(FphaRowRange {
                start: cursor,
                planes_per_block: planes,
            });
            cursor += planes * n_blks;
        }
        (rows, cursor)
    }

    /// Build evaporation column/row indices for each evaporation hydro.
    fn build_evap_indices(
        n_evap_hydros: usize,
        col_start: usize,
        row_start: usize,
    ) -> Vec<EvaporationIndices> {
        (0..n_evap_hydros)
            .map(|i| EvaporationIndices {
                q_ev_col: col_start + i * 3,
                f_evap_plus_col: col_start + i * 3 + 1,
                f_evap_minus_col: col_start + i * 3 + 2,
                evap_row: row_start + i,
            })
            .collect()
    }

    /// Construct a [`StageIndexer`] from a [`StageTemplate`].
    ///
    /// Extracts `n_hydro` and `max_par_order` from the template and delegates
    /// to [`StageIndexer::new`]. Produces identical results to calling
    /// `StageIndexer::new(template.n_hydro, template.max_par_order)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_sddp::indexer::StageIndexer;
    /// use cobre_solver::StageTemplate;
    ///
    /// let template = StageTemplate {
    ///     num_cols: 16,
    ///     num_rows: 12,
    ///     num_nz: 0,
    ///     col_starts: vec![0_i32; 17],
    ///     row_indices: vec![],
    ///     values: vec![],
    ///     col_lower: vec![0.0; 16],
    ///     col_upper: vec![f64::INFINITY; 16],
    ///     objective: vec![0.0; 16],
    ///     row_lower: vec![0.0; 12],
    ///     row_upper: vec![f64::INFINITY; 12],
    ///     n_state: 9,
    ///     n_transfer: 6,
    ///     n_dual_relevant: 9,
    ///     n_hydro: 3,
    ///     max_par_order: 2,
    ///     col_scale: vec![],
    ///     row_scale: vec![],
    /// };
    ///
    /// let idx = StageIndexer::from_stage_template(&template);
    /// assert_eq!(idx.storage, 0..3);
    /// assert_eq!(idx.theta,  15);
    /// ```
    #[must_use]
    pub fn from_stage_template(template: &StageTemplate) -> Self {
        Self::new(template.n_hydro, template.max_par_order)
    }

    /// Map a state-vector index to the LP column it should reference in a cut.
    ///
    /// The outgoing state after `shift_lag_state` stores:
    /// - `[0, N)`: outgoing storage → LP column `j` (identity mapping)
    /// - `[N + 0·N + h]`: outgoing lag 0 for hydro `h` = realised inflow
    ///   → LP column `z_inflow.start + h`
    /// - `[N + l·N + h]` for `l ≥ 1`: outgoing lag `l` = incoming lag `l − 1`
    ///   → LP column `N + (l − 1)·N + h`
    /// - `[N*(1+L), N*(1+L) + n_anticipated*K_max)`: `anticipated_state` slots
    ///   → shift-aware mapping (mirrors the inflow-lag pattern structurally):
    ///   - `slot == K_p − 1` for plant `p`: the post-shift outgoing slot carries
    ///     the decision committed at stage `t`. The Equal branch returns
    ///     `anticipated_state_out.start + p`, which is pinned to
    ///     `decision_col[p]` by the `anticipated_state_out_def` equality row
    ///     (`anticipated_state_out[p] − decision_col[p] = 0`). The state-fixing
    ///     row at slot `K_p − 1` is PURE IDENTITY under this layout (no
    ///     decision-write coefficient).
    ///   - `slot < K_p − 1`: the successor's slot `i` = predecessor's incoming
    ///     slot `i + 1` (shift); returns
    ///     `anticipated_state.start + (slot + 1) * n_anticipated + p`.
    ///   - `slot > K_p − 1`: padding (unused for this plant, pinned to 0 by the
    ///     state-fixing row); returns `j` (identity; safe default).
    ///
    /// The `anticipated_state` branch is evaluated regardless of `max_par_order`
    /// because the shift semantics apply even when there are no inflow lags.
    /// The `max_par_order == 0` early-return only guards the lag-remap block
    /// (which has zero length when `max_par_order == 0`).
    #[inline]
    #[must_use]
    pub fn state_to_lp_column(&self, j: usize) -> usize {
        let n = self.hydro_count;
        if j < n {
            return j;
        }
        // Anticipated-state mapping (must check before early return on max_par_order == 0).
        if self.n_anticipated > 0 && j >= self.anticipated_state.start {
            let ant_block_size = self.n_anticipated * self.k_max;
            if j < self.anticipated_state.start + ant_block_size {
                let offset = j - self.anticipated_state.start;
                let slot = offset / self.n_anticipated;
                let plant = offset % self.n_anticipated;
                let k_p = self.anticipated_lead_stages[plant];
                return match (slot + 1).cmp(&k_p) {
                    std::cmp::Ordering::Equal => self.anticipated_state_out.start + plant,
                    std::cmp::Ordering::Less => {
                        self.anticipated_state.start + (slot + 1) * self.n_anticipated + plant
                    }
                    // INVARIANT: Padding slot — `slot >= k_p` means this ring-buffer
                    // entry belongs to plant `plant` but exceeds its lead time `K_i`.
                    // Padding slots exist because the ring buffer is sized to `k_max`
                    // slots (the system-wide maximum), but plant `plant` only uses
                    // `k_p = K_i` of them. These slots are safe to pass through as
                    // their own state-column index (not a decision-variable column)
                    // because the following 5-step chain guarantees their LP dual is 0:
                    //   1. `shift_anticipated_state` initialises padding slots to 0.0.
                    //   2. The corresponding state-fixing row has RHS 0 (from step 1).
                    //   3. The LP solver pins the slot value to 0 via the equality row.
                    //   4. A zero-valued variable at a zero-RHS equality has dual 0.
                    //   5. Zero duals produce zero cut coefficients, which are no-ops
                    //      in the cut row (neither pruned nor corrupted).
                    //
                    // Pre-horizon seeding is implemented (see `setup/mod.rs` —
                    // `setup_anticipated_state`). The padding-zero invariant is
                    // preserved because seeds populate slots `[0, K_i)` only;
                    // slots `[K_i, k_max)` remain zero (debug_assert! guards
                    // this in `setup/mod.rs`). The identity return is therefore
                    // safe for padding slots under always-active fishing.
                    std::cmp::Ordering::Greater => j,
                };
            }
            return j;
        }
        if self.max_par_order == 0 {
            return j;
        }
        // Lag block: slot-to-column mapping.
        let offset = j - n;
        let h = offset % n;
        let lag = offset / n;
        if lag == 0 {
            self.z_inflow.start + h
        } else {
            n + (lag - 1) * n + h
        }
    }

    /// Fill [`state_to_lp_column_map`](Self::state_to_lp_column_map) by calling
    /// [`state_to_lp_column`](Self::state_to_lp_column) for every
    /// `j ∈ [0, n_state)`.
    ///
    /// Call once after the state layout is finalized (e.g. after
    /// [`set_nonzero_mask`](Self::set_nonzero_mask) in study setup). The map is a
    /// pure cache of the resolver — it never reimplements the mapping arithmetic.
    pub fn finalize_state_column_map(&mut self) {
        self.state_to_lp_column_map.clear();
        self.state_to_lp_column_map.reserve(self.n_state);
        for j in 0..self.n_state {
            self.state_to_lp_column_map.push(self.state_to_lp_column(j));
        }
        debug_assert_eq!(self.state_to_lp_column_map.len(), self.n_state);
    }

    /// Read the precomputed `state_to_lp_column(j)`, falling back to the live
    /// resolver when the map has not been finalized.
    ///
    /// The fallback guarantees correctness for any indexer (e.g. test indexers
    /// that skip [`finalize_state_column_map`](Self::finalize_state_column_map))
    /// without editing every construction site.
    #[inline]
    #[must_use]
    pub fn lp_column_for_state(&self, j: usize) -> usize {
        self.state_to_lp_column_map
            .get(j)
            .copied()
            .unwrap_or_else(|| self.state_to_lp_column(j))
    }

    /// Map a state-vector index to the LP column pinned by
    /// [`fill_col_state_patches`](crate::lp_builder::PatchBuffer::fill_col_state_patches).
    ///
    /// This is the **incoming-state column** — the column whose bound is set to
    /// `lb = ub = v` when state-fixing is applied via `set_col_bounds`. The
    /// column indices returned here are exactly those written into
    /// `PatchBuffer::col_indices[..state_col_patch_count()]` by
    /// `fill_col_state_patches`, in state-vector order.
    ///
    /// Use this method in the backward pass to read `view.reduced_costs[col]`
    /// for the cut subgradient, one entry per state-vector component
    /// `j ∈ [0, n_state)`.
    ///
    /// ## Mapping by range
    ///
    /// - `j ∈ [0, N)` (storage): returns `self.storage_in.start + j`.
    /// - `j ∈ [N, N*(1+L))` (AR lags): returns
    ///   `self.inflow_lags.start + (j − N)`.
    /// - `j ∈ [N*(1+L), n_state)` (anticipated state): returns
    ///   `self.anticipated_state.start + (j − N*(1+L))`.
    ///
    /// ## Contrast with [`state_to_lp_column`]
    ///
    /// [`state_to_lp_column`] returns the **outgoing** column used for
    /// cut-row coefficient construction in the forward pass (`forward.rs`).
    /// For the storage range, `state_to_lp_column(j) = j` (the outgoing
    /// storage column), while this method returns `storage_in.start + j`
    /// (the incoming storage column). The two columns are related via the
    /// water-balance equality row, and by KKT duality the reduced cost on
    /// the incoming column equals the dual of the equivalent equality row
    /// that a row-based state-fixing formulation would produce.
    ///
    /// [`state_to_lp_column`]: Self::state_to_lp_column
    #[inline]
    #[must_use]
    pub fn state_to_lp_incoming_column(&self, j: usize) -> usize {
        let n = self.hydro_count;
        let lag_end = n * (1 + self.max_par_order);
        if j < n {
            // Storage range: incoming storage column.
            self.storage_in.start + j
        } else if j < lag_end {
            // AR lag range: incoming lag column.
            self.inflow_lags.start + (j - n)
        } else {
            // Anticipated-state range: anticipated state column.
            self.anticipated_state.start + (j - lag_end)
        }
    }

    /// Iterator over `(local_idx, lp_column)` for anticipated decisions active
    /// at `stage_idx`.
    ///
    /// A plant is active iff
    /// `stage_idx + anticipated_lead_stages[local_idx] < n_stages`. The boundary
    /// case `stage_idx + K_i == n_stages` is **excluded**: the commitment would
    /// mature at a delivery stage outside the study horizon `[0, n_stages)`, so
    /// no delivery LP is ever built for it. Inactive plants are skipped; the LP
    /// build applies `[0, 0]` bounds to their columns so the presolver
    /// eliminates them.
    ///
    /// All arithmetic uses `usize`; the upstream conversion from `u32 lead_stages`
    /// to `usize` happens when [`EquipmentCounts::anticipated_lead_stages`] is
    /// populated.
    pub fn anticipated_decision_active_at_stage(
        &self,
        stage_idx: usize,
        n_stages: usize,
    ) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.anticipated_lead_stages
            .iter()
            .enumerate()
            .filter_map(move |(i, &k_i)| {
                if stage_idx.saturating_add(k_i) < n_stages {
                    Some((i, self.anticipated_decision.start + i))
                } else {
                    None
                }
            })
    }

    /// Iterator over `(local_idx, lp_row)` for anticipated fishing
    /// constraints active at `stage_idx`. Active iff the plant exists
    /// (always-active predicate). Returns one row per anticipated plant in
    /// ascending `local_idx` order.
    ///
    /// Rows are assigned in ascending `local_idx` order: plant `i` gets row
    /// `anticipated_fishing_start + i`.
    ///
    /// The dual on this row carries the cut subgradient w.r.t. the
    /// currently-delivering anticipated-state slot (slot 0 after the ring-buffer
    /// shift) during backward-pass cut extraction.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `stage_idx > n_stages`.
    pub fn anticipated_fishing_active_at_stage(
        &self,
        stage_idx: usize,
        n_stages: usize,
    ) -> impl Iterator<Item = (usize, usize)> + '_ {
        debug_assert!(
            stage_idx <= n_stages,
            "stage_idx {stage_idx} exceeds n_stages {n_stages}"
        );
        self.anticipated_lead_stages
            .iter()
            .enumerate()
            .map(move |(local_idx, _)| (local_idx, self.anticipated_fishing_start + local_idx))
    }

    /// Return `true` when the anticipated-fishing equality is active for plant
    /// `local_idx` at `stage_idx`. Always returns `true` for valid inputs
    /// under the always-active rule.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `stage_idx > n_stages` or `local_idx >= self.n_anticipated()`.
    #[inline]
    #[must_use]
    pub fn is_anticipated_fishing_active(
        &self,
        local_idx: usize,
        stage_idx: usize,
        n_stages: usize,
    ) -> bool {
        debug_assert!(
            stage_idx <= n_stages,
            "stage_idx {stage_idx} must be <= n_stages {n_stages}",
        );
        debug_assert!(
            local_idx < self.anticipated_lead_stages.len(),
            "local_idx {local_idx} out of bounds (n_anticipated = {})",
            self.anticipated_lead_stages.len(),
        );
        true
    }

    /// Compute and store the nonzero state index mask from per-hydro
    /// lag-state-slot counts and per-plant anticipated lead-stage counts.
    ///
    /// `lag_counts` must have length `hydro_count`. Each entry is the number of
    /// lag-state slots that may carry non-zero cut coefficients for that hydro
    /// (0 means no AR lags). Indices `[0, N)` (storage) are always included.
    /// For each hydro `h`, lag indices
    /// `inflow_lags.start + l * hydro_count + h` are included for
    /// `l in 0..lag_counts[h]`.
    ///
    /// `anticipated_lead_stages` must have length `n_anticipated`. Each entry
    /// is the per-plant occupied-slot count `K_i` (`0..K_i` of the
    /// anticipated-state ring buffer at plant `i`). The trailing
    /// `k_max - K_i` slots are padding and are excluded from the mask: no
    /// decision variable writes to those columns, so their cut coefficients
    /// are structurally zero. Including padded slots would over-estimate cut
    /// hyperplanes (the direct analogue of the d0e4a42 PAR(p)-A bug, where
    /// padded lag slots were included and shifted the cut above the LP value
    /// at the visited state).
    ///
    /// Layout for the anticipated block:
    /// `anticipated_state.start + slot * n_anticipated + plant`. The loop
    /// iterates slot-first, plant-second so the emitted indices stay
    /// monotonically increasing.
    ///
    /// The correct value for `lag_counts[h]` is
    /// `PrecomputedPar::effective_lag_count(h)`, **not** `PrecomputedPar::order(h)`.
    /// For PAR(p)-A hydros, `effective_lag_count` equals `max_order` (= 12) so
    /// that the `ψ̂/12` annual contributions on lag slots `order..max_order` are
    /// included in cut rows. Using `order(h)` would truncate those slots and
    /// produce over-estimating cuts (the same failure mode as anticipated
    /// padding, but on the lag block).
    ///
    /// After calling, `nonzero_state_indices` is sorted in ascending order and
    /// has no duplicates. If `max_par_order == 0` or all hydros use their full
    /// `max_par_order` slots, the mask covers all lag indices; if
    /// `n_anticipated == 0` no anticipated entries are appended.
    ///
    /// # Worked example
    ///
    /// One anticipated plant, `K_0 = 2`, `k_max = 3`, `n_anticipated = 1`,
    /// `hydro_count = 0`, `max_par_order = 0`,
    /// `anticipated_state.start = X` (for the corresponding indexer
    /// configuration). With `lag_counts = &[]` and
    /// `anticipated_lead_stages = &[2]`, the mask is:
    ///
    /// - Storage: empty (no hydros).
    /// - Lag: empty (no AR lags).
    /// - Anticipated: slot 0 emits `X + 0 * 1 + 0 = X`; slot 1 emits
    ///   `X + 1 * 1 + 0 = X + 1`; slot 2 is padding (`slot >= K_0`) and is
    ///   excluded.
    ///
    /// Resulting mask: `[X, X + 1]`.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `lag_counts.len() != hydro_count`,
    /// `anticipated_lead_stages.len() != n_anticipated`, any
    /// `lag_counts[h] > max_par_order`, or any
    /// `anticipated_lead_stages[p] > k_max`.
    pub fn set_nonzero_mask(&mut self, lag_counts: &[usize], anticipated_lead_stages: &[usize]) {
        debug_assert_eq!(lag_counts.len(), self.hydro_count);
        debug_assert_eq!(anticipated_lead_stages.len(), self.n_anticipated);

        let n_lag_active: usize = lag_counts.iter().copied().sum();
        let n_ant_active: usize = anticipated_lead_stages.iter().copied().sum();
        let mut mask = Vec::with_capacity(self.hydro_count + n_lag_active + n_ant_active);

        // Storage indices.
        for h in 0..self.hydro_count {
            mask.push(h);
        }

        // Lag indices: lag-major layout, iterate lag-first to produce sorted indices.
        for lag in 0..self.max_par_order {
            for (h, &lag_count) in lag_counts.iter().enumerate() {
                debug_assert!(lag_count <= self.max_par_order);
                if lag < lag_count {
                    mask.push(self.inflow_lags.start + lag * self.hydro_count + h);
                }
            }
        }

        // Anticipated state: slots `0..K_i` for plant `i`.
        // Layout: anticipated_state.start + slot * n_anticipated + plant.
        for slot in 0..self.k_max {
            for (plant, &k_i) in anticipated_lead_stages.iter().enumerate() {
                debug_assert!(k_i <= self.k_max);
                if slot < k_i {
                    mask.push(self.anticipated_state.start + slot * self.n_anticipated + plant);
                }
            }
        }

        debug_assert!(
            mask.windows(2).all(|w| w[0] < w[1]),
            "nonzero_state_indices must be sorted and unique"
        );

        self.nonzero_state_indices = mask;
    }

    /// Finalize a test-constructed indexer the way production
    /// [`build_wired_indexer`](crate::setup) does: populate the nonzero-state
    /// mask, then the `state_to_lp_column` precompute map.
    ///
    /// Production studies build their mask from per-hydro
    /// `PrecomputedPar::effective_lag_count`, but test indexers built via
    /// [`StageIndexer::new`] / [`StageIndexer::with_equipment_and_evaporation`]
    /// have no PAR model. This helper uses the full `max_par_order` for every
    /// hydro (so the mask covers every lag slot — the same coverage the old
    /// dense path emitted) and the indexer's own `anticipated_lead_stages`. For
    /// a storage-only indexer this yields mask `[0, n_state)` ascending,
    /// reproducing the pre-unification dense output byte-for-byte.
    ///
    /// Use this at any test site that feeds the indexer into the forward-pass
    /// cut-row hot path (`build_cut_row_batch_into`), which is mask-driven and
    /// requires a finalized mask.
    #[cfg(test)]
    pub fn finalize_for_test(&mut self) {
        let lag_counts = vec![self.max_par_order; self.hydro_count];
        let anticipated_k = self.anticipated_lead_stages.clone();
        self.set_nonzero_mask(&lag_counts, &anticipated_k);
        self.finalize_state_column_map();
    }
}

// StageIndexer contains only Send + Sync types (Range<usize>, usize, Vec<usize>,
// Vec<FphaRowRange>, Vec<EvaporationIndices>), so Send + Sync are automatically
// derived. The explicit bounds below serve as a compile-time assertion that the
// safety invariant holds.
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn check() {
        assert_send_sync::<StageIndexer>();
    }
    let _ = check;
};

#[cfg(test)]
mod tests {
    use cobre_solver::StageTemplate;

    use super::{EquipmentCounts, EvapConfig, FphaColumnLayout, FphaRowRange, StageIndexer};

    fn eq(
        hydro_count: usize,
        max_par_order: usize,
        n_thermals: usize,
        n_lines: usize,
        n_buses: usize,
        n_blks: usize,
        has_inflow_penalty: bool,
    ) -> EquipmentCounts {
        EquipmentCounts {
            hydro_count,
            max_par_order,
            n_thermals,
            n_lines,
            n_buses,
            n_blks,
            has_inflow_penalty,
            max_deficit_segments: 1,
            n_anticipated: 0,
            k_max: 0,
            anticipated_lead_stages: vec![],
            anticipated_thermal_indices: vec![],
        }
    }

    fn fpha(hydro_indices: Vec<usize>, planes_per_hydro: Vec<usize>) -> FphaColumnLayout {
        FphaColumnLayout {
            hydro_indices,
            planes_per_hydro,
        }
    }

    fn evap(hydro_indices: Vec<usize>) -> EvapConfig {
        EvapConfig { hydro_indices }
    }

    // Worked example from spec SS5.5.3: N = 3, L = 2

    fn indexer_3_2() -> StageIndexer {
        StageIndexer::new(3, 2)
    }

    #[test]
    fn storage_range_3_2() {
        assert_eq!(indexer_3_2().storage, 0..3);
    }

    #[test]
    fn inflow_lags_range_3_2() {
        assert_eq!(indexer_3_2().inflow_lags, 3..9);
    }

    #[test]
    fn z_inflow_range_3_2() {
        assert_eq!(indexer_3_2().z_inflow, 9..12);
    }

    #[test]
    fn storage_in_range_3_2() {
        assert_eq!(indexer_3_2().storage_in, 12..15);
    }

    #[test]
    fn theta_index_3_2() {
        assert_eq!(indexer_3_2().theta, 15);
    }

    #[test]
    fn n_state_3_2() {
        assert_eq!(indexer_3_2().n_state, 9);
    }

    #[test]
    fn storage_fixing_range_3_2() {
        // storage_fixing is a permanent empty sentinel.
        assert_eq!(indexer_3_2().storage_fixing, 0..0);
    }

    #[test]
    fn lag_fixing_range_3_2() {
        // lag_fixing is a permanent empty sentinel.
        assert_eq!(indexer_3_2().lag_fixing, 0..0);
    }

    #[test]
    fn row_column_symmetry_3_2() {
        // State-fixing row ranges are permanent empty sentinels; column ranges are unchanged.
        let idx = indexer_3_2();
        assert_eq!(idx.storage_fixing, 0..0);
        assert_eq!(idx.lag_fixing, 0..0);
        // Column ranges are unchanged.
        assert_eq!(idx.storage, 0..3);
        assert_eq!(idx.inflow_lags, 3..9);
    }

    // Production scale: N = 160, L = 12

    fn indexer_160_12() -> StageIndexer {
        StageIndexer::new(160, 12)
    }

    #[test]
    fn n_state_production_scale() {
        assert_eq!(indexer_160_12().n_state, 2080);
    }

    #[test]
    fn theta_production_scale() {
        assert_eq!(indexer_160_12().theta, 2400);
    }

    #[test]
    fn row_column_symmetry_production_scale() {
        // State-fixing row ranges are permanent empty sentinels; column ranges are unchanged.
        let idx = indexer_160_12();
        assert_eq!(idx.storage_fixing, 0..0);
        assert_eq!(idx.lag_fixing, 0..0);
        // Column ranges are unchanged.
        assert_eq!(idx.storage, 0..160);
        assert_eq!(idx.inflow_lags, 160..160 * 13);
    }

    #[test]
    fn single_hydro_no_lags() {
        let idx = StageIndexer::new(1, 0);

        assert_eq!(idx.storage, 0..1);
        assert_eq!(idx.inflow_lags, 1..1);
        assert_eq!(idx.z_inflow, 1..2);
        assert_eq!(idx.storage_in, 2..3);
        assert_eq!(idx.theta, 3);
        assert_eq!(idx.n_state, 1);
        // storage_fixing and lag_fixing are permanent empty sentinels.
        assert_eq!(idx.storage_fixing, 0..0);
        assert_eq!(idx.lag_fixing, 0..0);
    }

    // Edge case: N = 0, L = 0 (degenerate — all ranges empty)

    #[test]
    fn degenerate_zero_hydros() {
        let idx = StageIndexer::new(0, 0);

        assert_eq!(idx.storage, 0..0);
        assert_eq!(idx.inflow_lags, 0..0);
        assert_eq!(idx.z_inflow, 0..0);
        assert_eq!(idx.storage_in, 0..0);
        assert_eq!(idx.theta, 0);
        assert_eq!(idx.n_state, 0);
        assert_eq!(idx.storage_fixing, 0..0);
        assert_eq!(idx.lag_fixing, 0..0);

        assert_eq!(idx.storage_fixing, idx.storage);
        assert_eq!(idx.lag_fixing, idx.inflow_lags);
    }

    // from_stage_template: must produce the same result as new()

    fn make_template(n_hydro: usize, max_par_order: usize) -> StageTemplate {
        let n_state = n_hydro * (1 + max_par_order);
        let n_transfer = n_hydro * max_par_order;
        // Minimal valid template (matrix contents are irrelevant for indexer)
        StageTemplate {
            num_cols: 0,
            num_rows: 0,
            num_nz: 0,
            col_starts: vec![0_i32],
            row_indices: vec![],
            values: vec![],
            col_lower: vec![],
            col_upper: vec![],
            objective: vec![],
            row_lower: vec![],
            row_upper: vec![],
            n_state,
            n_transfer,
            n_dual_relevant: n_state,
            n_hydro,
            max_par_order,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    #[test]
    fn from_stage_template_matches_new_3_2() {
        let tmpl = make_template(3, 2);
        let from_tmpl = StageIndexer::from_stage_template(&tmpl);
        let from_new = StageIndexer::new(3, 2);

        assert_eq!(from_tmpl.storage, from_new.storage);
        assert_eq!(from_tmpl.inflow_lags, from_new.inflow_lags);
        assert_eq!(from_tmpl.storage_in, from_new.storage_in);
        assert_eq!(from_tmpl.theta, from_new.theta);
        assert_eq!(from_tmpl.n_state, from_new.n_state);
        assert_eq!(from_tmpl.storage_fixing, from_new.storage_fixing);
        assert_eq!(from_tmpl.lag_fixing, from_new.lag_fixing);
        assert_eq!(from_tmpl.hydro_count, from_new.hydro_count);
        assert_eq!(from_tmpl.max_par_order, from_new.max_par_order);
    }

    #[test]
    fn from_stage_template_matches_new_160_12() {
        let tmpl = make_template(160, 12);
        let from_tmpl = StageIndexer::from_stage_template(&tmpl);
        let from_new = StageIndexer::new(160, 12);

        assert_eq!(from_tmpl.n_state, from_new.n_state);
        assert_eq!(from_tmpl.theta, from_new.theta);
        assert_eq!(from_tmpl.hydro_count, from_new.hydro_count);
        assert_eq!(from_tmpl.max_par_order, from_new.max_par_order);
    }

    #[test]
    fn from_stage_template_matches_new_edge_cases() {
        for (n, l) in [(0, 0), (1, 0), (1, 1)] {
            let tmpl = make_template(n, l);
            let from_tmpl = StageIndexer::from_stage_template(&tmpl);
            let from_new = StageIndexer::new(n, l);

            assert_eq!(from_tmpl.storage, from_new.storage, "N={n} L={l}");
            assert_eq!(from_tmpl.inflow_lags, from_new.inflow_lags, "N={n} L={l}");
            assert_eq!(from_tmpl.theta, from_new.theta, "N={n} L={l}");
            assert_eq!(from_tmpl.n_state, from_new.n_state, "N={n} L={l}");
        }
    }

    // Clone / Debug — structural sanity

    #[test]
    fn clone_and_debug() {
        let idx = indexer_3_2();
        let cloned = idx.clone();
        assert_eq!(cloned.theta, idx.theta);
        assert_eq!(cloned.n_state, idx.n_state);

        let debug_str = format!("{idx:?}");
        assert!(debug_str.contains("StageIndexer"));
    }

    // new() produces empty equipment ranges
    #[test]
    fn new_equipment_ranges_are_empty() {
        let idx = StageIndexer::new(3, 2);
        assert!(idx.turbine.is_empty());
        assert!(idx.spillage.is_empty());
        assert!(idx.diversion.is_empty());
        assert!(idx.thermal.is_empty());
        assert!(idx.line_fwd.is_empty());
        assert!(idx.line_rev.is_empty());
        assert!(idx.deficit.is_empty());
        assert!(idx.excess.is_empty());
        assert_eq!(idx.n_blks, 0);
        assert_eq!(idx.n_thermals, 0);
        assert_eq!(idx.n_lines, 0);
        assert_eq!(idx.n_buses, 0);
    }

    // with_equipment: worked example from doc comment (N=1, L=0, T=2, Ln=1, B=2, K=1)
    //
    // theta = N*(3+L) = 1*(3+0) = 3
    // decision_start = 4
    // turbine:    [4, 4+1*1)  = 4..5
    // spillage:   [5, 5+1*1)  = 5..6
    // diversion:  [6, 6+1*1)  = 6..7
    // thermal:    [7, 7+2*1)  = 7..9
    // line_fwd:   [9, 9+1*1)  = 9..10
    // line_rev:  [10,10+1*1)  = 10..11
    // deficit:   [11,11+2*1)  = 11..13
    // excess:    [13,13+2*1)  = 13..15
    #[test]
    fn with_equipment_doctest_n1_l0_t2_l1_b2_k1() {
        let idx = StageIndexer::with_equipment(&eq(1, 0, 2, 1, 2, 1, false), &fpha(vec![], vec![]));

        // State ranges are identical to new(1, 0)
        assert_eq!(idx.storage, 0..1);
        assert_eq!(idx.inflow_lags, 1..1);
        assert_eq!(idx.z_inflow, 1..2);
        assert_eq!(idx.storage_in, 2..3);
        assert_eq!(idx.theta, 3);
        assert_eq!(idx.n_state, 1);

        // Equipment ranges
        assert_eq!(idx.turbine, 4..5);
        assert_eq!(idx.spillage, 5..6);
        assert_eq!(idx.diversion, 6..7);
        assert_eq!(idx.thermal, 7..9);
        assert_eq!(idx.line_fwd, 9..10);
        assert_eq!(idx.line_rev, 10..11);
        assert_eq!(idx.deficit, 11..13);
        assert_eq!(idx.excess, 13..15);

        // Equipment counts
        assert_eq!(idx.n_blks, 1);
        assert_eq!(idx.n_thermals, 2);
        assert_eq!(idx.n_lines, 1);
        assert_eq!(idx.n_buses, 2);
    }

    // with_equipment: N=2, L=1, T=3, Ln=2, B=4, K=2
    //
    // theta = N*(3+L) = 2*(3+1) = 8
    // decision_start = 9
    // turbine:    [9,  9+2*2)  = 9..13
    // spillage:  [13, 13+2*2)  = 13..17
    // diversion: [17, 17+2*2)  = 17..21
    // thermal:   [21, 21+3*2)  = 21..27
    // line_fwd:  [27, 27+2*2)  = 27..31
    // line_rev:  [31, 31+2*2)  = 31..35
    // deficit:   [35, 35+4*2)  = 35..43
    // excess:    [43, 43+4*2)  = 43..51
    #[test]
    fn with_equipment_n2_l1_t3_l2_b4_k2() {
        let idx = StageIndexer::with_equipment(&eq(2, 1, 3, 2, 4, 2, false), &fpha(vec![], vec![]));

        // State ranges identical to new(2, 1)
        assert_eq!(idx.theta, 8);
        assert_eq!(idx.n_state, 4); // N*(1+L) = 2*2 = 4

        // Equipment ranges
        assert_eq!(idx.turbine, 9..13);
        assert_eq!(idx.spillage, 13..17);
        assert_eq!(idx.diversion, 17..21);
        assert_eq!(idx.thermal, 21..27);
        assert_eq!(idx.line_fwd, 27..31);
        assert_eq!(idx.line_rev, 31..35);
        assert_eq!(idx.deficit, 35..43);
        assert_eq!(idx.excess, 43..51);
    }

    // with_equipment: no equipment (all counts zero), matches new() state layout
    #[test]
    fn with_equipment_all_counts_zero_matches_new() {
        let with_eq =
            StageIndexer::with_equipment(&eq(3, 2, 0, 0, 0, 0, false), &fpha(vec![], vec![]));
        let base = StageIndexer::new(3, 2);

        assert_eq!(with_eq.storage, base.storage);
        assert_eq!(with_eq.inflow_lags, base.inflow_lags);
        assert_eq!(with_eq.storage_in, base.storage_in);
        assert_eq!(with_eq.theta, base.theta);
        assert_eq!(with_eq.n_state, base.n_state);
        // All equipment ranges empty
        assert!(with_eq.turbine.is_empty());
        assert!(with_eq.spillage.is_empty());
        assert!(with_eq.diversion.is_empty());
        assert!(with_eq.thermal.is_empty());
        assert!(with_eq.line_fwd.is_empty());
        assert!(with_eq.line_rev.is_empty());
        assert!(with_eq.deficit.is_empty());
        assert!(with_eq.excess.is_empty());
    }

    // with_equipment: adjacency invariant — ranges must be contiguous and non-overlapping
    #[test]
    fn with_equipment_ranges_are_contiguous() {
        let idx = StageIndexer::with_equipment(&eq(2, 1, 3, 2, 4, 2, false), &fpha(vec![], vec![]));

        // turbine immediately follows theta
        assert_eq!(idx.turbine.start, idx.theta + 1);
        // each range starts where the previous ends
        assert_eq!(idx.spillage.start, idx.turbine.end);
        assert_eq!(idx.diversion.start, idx.spillage.end);
        assert_eq!(idx.thermal.start, idx.diversion.end);
        assert_eq!(idx.line_fwd.start, idx.thermal.end);
        assert_eq!(idx.line_rev.start, idx.line_fwd.end);
        assert_eq!(idx.deficit.start, idx.line_rev.end);
        assert_eq!(idx.excess.start, idx.deficit.end);
    }

    // Column index formula: turbine[h, b] = turbine.start + h * n_blks + b
    #[test]
    fn with_equipment_column_index_formulas() {
        let n_blks = 3_usize;
        let idx =
            StageIndexer::with_equipment(&eq(2, 1, 1, 1, 1, n_blks, false), &fpha(vec![], vec![]));

        // turbine[h=0, b=0] = turbine.start (no offset for h=0, b=0)
        assert_eq!(idx.turbine.start, idx.turbine.start);
        // turbine[h=1, b=2] = turbine.start + 1*3 + 2 = turbine.start + 5
        assert_eq!(idx.turbine.start + n_blks + 2, idx.turbine.start + 5);
        // deficit[b_idx=0, blk=1] = deficit.start + 1
        assert_eq!(idx.deficit.start + 1, idx.deficit.start + 1);
        // turbine[h=1, b=0] = turbine.start + n_blks
        assert_eq!(idx.turbine.start + n_blks, idx.turbine.start + 3);
    }

    // with_equipment: has_inflow_penalty=true appends N slack columns after excess
    //
    // N=2, L=1, T=1, Ln=1, B=1, K=1, penalty=true
    // theta = N*(3+L) = 2*(3+1) = 8
    // decision_start = 9
    // turbine:    [9,  11)
    // spillage:  [11, 13)
    // diversion: [13, 15)
    // thermal:   [15, 16)
    // line_fwd:  [16, 17)
    // line_rev:  [17, 18)
    // deficit:   [18, 19)
    // excess:    [19, 20)
    // inflow_slack: [20, 22)  <- excess_end..excess_end+N
    #[test]
    fn with_equipment_inflow_penalty_appends_slack() {
        let idx = StageIndexer::with_equipment(&eq(2, 1, 1, 1, 1, 1, true), &fpha(vec![], vec![]));

        assert!(idx.has_inflow_penalty, "has_inflow_penalty must be true");
        // inflow_slack must start exactly where excess ends
        assert_eq!(
            idx.inflow_slack.start, idx.excess.end,
            "inflow_slack.start must equal excess.end (contiguous)"
        );
        // inflow_slack must contain exactly hydro_count columns
        assert_eq!(
            idx.inflow_slack.len(),
            idx.hydro_count,
            "inflow_slack must contain exactly hydro_count columns"
        );
        assert_eq!(idx.inflow_slack, 20..22);
        // inflow_slack_rows stays empty in this implementation
        assert!(
            idx.inflow_slack_rows.is_empty(),
            "inflow_slack_rows must remain empty"
        );
        // without penalty the slack range is empty
        let no_penalty =
            StageIndexer::with_equipment(&eq(2, 1, 1, 1, 1, 1, false), &fpha(vec![], vec![]));
        assert!(!no_penalty.has_inflow_penalty);
        assert!(no_penalty.inflow_slack.is_empty());
    }

    // ── FPHA field tests ───────────────────────────────────────────────────

    // AC-4: no FPHA hydros → generation is empty, fpha_rows is empty.
    //
    // N=4, L=0, T=0, Ln=0, B=1, K=1, no penalty, no FPHA.
    // theta = N*(3+L) = 4*(3+0) = 12
    // decision_start = 13
    // turbine:  [13, 17)
    // spillage: [17, 21)
    // deficit:  [21, 22)
    // excess:   [22, 23)
    // generation: empty (no FPHA hydros)
    #[test]
    fn fpha_no_hydros_generation_is_empty() {
        let idx = StageIndexer::with_equipment(&eq(4, 0, 0, 0, 1, 1, false), &fpha(vec![], vec![]));

        assert!(
            idx.generation.is_empty(),
            "generation must be empty with no FPHA hydros"
        );
        assert_eq!(idx.n_fpha_hydros, 0);
        assert!(idx.fpha_hydro_indices.is_empty());
        assert!(idx.fpha_rows.is_empty());
    }

    // AC-1 + AC-2: 1 FPHA hydro, 1 block, 3 planes.
    //
    // N=2, L=0, T=1, Ln=0, B=1, K=1, no penalty.
    // theta = N*(3+L) = 2*(3+0) = 6
    // decision_start = 7
    // turbine:    [7, 9)   (2 hydros * 1 block)
    // spillage:   [9, 11)
    // diversion: [11, 13)  (2 hydros * 1 block)
    // thermal:   [13, 14)  (1 thermal * 1 block)
    // deficit:   [14, 15)  (1 bus * 1 block)
    // excess:    [15, 16)
    // generation: [16, 17) (1 FPHA hydro * 1 block)
    //
    // Row layout:
    // n_state = N*(1+L) = 2*(1+0) = 2
    // z_inflow rows = 2..4  (N*(1+L)..N*(2+L))
    // water_balance_start = N*(2+L) = 4
    // load_balance_start  = 4 + 2 = 6
    // load_balance_end    = 6 + 1*1 = 7
    // fpha_rows[0].start  = 7 (after load_balance.end)
    // fpha_rows[0].planes_per_block = 3
    #[test]
    fn fpha_one_hydro_one_block_three_planes() {
        let idx =
            StageIndexer::with_equipment(&eq(2, 0, 1, 0, 1, 1, false), &fpha(vec![0], vec![3]));

        // AC-1: generation spans 1 column (1 FPHA hydro * 1 block)
        assert_eq!(idx.generation.len(), 1, "generation must span 1 column");
        assert_eq!(idx.generation, 16..17);
        assert_eq!(idx.n_fpha_hydros, 1);
        assert_eq!(idx.fpha_hydro_indices, vec![0]);

        // AC-2: fpha_rows[0].start is after load_balance.end, planes_per_block == 3
        assert_eq!(idx.fpha_rows.len(), 1);
        assert_eq!(
            idx.fpha_rows[0].start, idx.load_balance.end,
            "fpha_rows[0].start must equal load_balance.end"
        );
        assert_eq!(idx.fpha_rows[0].planes_per_block, 3);
    }

    // AC-3: 2 FPHA hydros, 2 blocks, plane counts [5, 4].
    //
    // N=4, L=0, T=0, Ln=0, B=1, K=2, no penalty.
    // theta = N*(3+L) = 4*(3+0) = 12
    // decision_start = 13
    // turbine:    [13, 21)  (4 hydros * 2 blocks)
    // spillage:   [21, 29)
    // diversion:  [29, 37)  (4 hydros * 2 blocks)
    // deficit:    [37, 39)  (1 bus * 2 blocks)
    // excess:     [39, 41)
    // generation: [41, 45) (2 FPHA hydros * 2 blocks = 4 columns)
    #[test]
    fn fpha_two_hydros_two_blocks_different_planes() {
        let idx = StageIndexer::with_equipment(
            &eq(4, 0, 0, 0, 1, 2, false),
            &fpha(vec![1, 3], vec![5, 4]),
        );

        // AC-3: generation spans 4 columns (2 FPHA hydros * 2 blocks)
        assert_eq!(idx.generation.len(), 4, "generation must span 4 columns");
        assert_eq!(idx.n_fpha_hydros, 2);
        assert_eq!(idx.fpha_hydro_indices, vec![1, 3]);

        // fpha_rows: 2 entries with correct starts and plane counts
        assert_eq!(idx.fpha_rows.len(), 2);

        // fpha_rows[0]: hydro at local 0 (system hydro 1), 5 planes, 2 blocks
        // starts at load_balance.end
        assert_eq!(
            idx.fpha_rows[0].start, idx.load_balance.end,
            "fpha_rows[0].start must equal load_balance.end"
        );
        assert_eq!(idx.fpha_rows[0].planes_per_block, 5);

        // fpha_rows[1]: starts after fpha_rows[0]'s region (5 planes * 2 blocks = 10 rows)
        assert_eq!(
            idx.fpha_rows[1].start,
            idx.fpha_rows[0].start + 5 * 2,
            "fpha_rows[1].start must follow fpha_rows[0]'s 10-row region"
        );
        assert_eq!(idx.fpha_rows[1].planes_per_block, 4);
    }

    // FPHA generation columns are contiguous with the prior column region.
    //
    // No penalty: generation immediately follows excess.
    // With penalty: generation immediately follows inflow_slack.
    #[test]
    fn fpha_generation_contiguous_with_prior_region() {
        // No penalty case: generation.start == excess.end
        let no_penalty =
            StageIndexer::with_equipment(&eq(2, 0, 0, 0, 1, 1, false), &fpha(vec![0], vec![2]));
        assert_eq!(
            no_penalty.generation.start, no_penalty.excess.end,
            "generation.start must equal excess.end when no penalty"
        );

        // With penalty case: generation.start == inflow_slack.end
        let with_penalty =
            StageIndexer::with_equipment(&eq(2, 0, 0, 0, 1, 1, true), &fpha(vec![0], vec![2]));
        assert_eq!(
            with_penalty.generation.start, with_penalty.inflow_slack.end,
            "generation.start must equal inflow_slack.end when penalty active"
        );
    }

    // FPHA rows are contiguous with load_balance (start at load_balance.end).
    #[test]
    fn fpha_rows_contiguous_with_load_balance() {
        let idx = StageIndexer::with_equipment(
            &eq(3, 1, 2, 0, 2, 3, false),
            &fpha(vec![0, 2], vec![4, 6]),
        );

        // First FPHA hydro starts at load_balance.end
        assert_eq!(
            idx.fpha_rows[0].start, idx.load_balance.end,
            "fpha_rows[0] must start at load_balance.end"
        );

        // Each subsequent FPHA hydro starts after its predecessor's block
        // fpha_rows[0]: 4 planes * 3 blocks = 12 rows
        assert_eq!(
            idx.fpha_rows[1].start,
            idx.fpha_rows[0].start + 4 * 3,
            "fpha_rows[1] must start after fpha_rows[0]'s rows"
        );
        assert_eq!(idx.fpha_rows[1].planes_per_block, 6);
    }

    // ── Evaporation field tests ────────────────────────────────────────────

    // 0 evaporation hydros → evap_indices is empty.
    #[test]
    fn evap_no_hydros_indices_empty() {
        let idx = StageIndexer::with_equipment(&eq(3, 0, 1, 0, 1, 1, false), &fpha(vec![], vec![]));

        assert_eq!(idx.n_evap_hydros, 0);
        assert!(idx.evap_hydro_indices.is_empty());
        assert!(idx.evap_indices.is_empty());
    }

    // 1 evaporation hydro — verify column/row positions.
    //
    // N=2, L=0, T=0, Ln=0, B=1, K=1, no penalty, no FPHA, 1 evap hydro.
    // theta = N*(3+L) = 2*(3+0) = 6
    // decision_start = 7
    // turbine:    [7, 9)   (2 hydros * 1 block)
    // spillage:   [9, 11)
    // diversion: [11, 13)  (2 hydros * 1 block)
    // deficit:   [13, 14)  (1 bus * 1 block)
    // excess:    [14, 15)
    // generation: empty (no FPHA)
    // evap cols: [15, 18)  (3 columns: Q_ev, f_evap_plus, f_evap_minus)
    //
    // Row layout (Phase 1: state-fixing rows removed):
    // z_inflow rows = 0..2  (start at row 0)
    // water_balance_start = 2  (z_inflow rows end)
    // load_balance_start = 2 + 2 = 4
    // load_balance_end   = 4 + 1*1 = 5
    // evap_row[0] = 5
    #[test]
    fn evap_one_hydro_column_row_positions() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(2, 0, 0, 0, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![0]),
        );

        assert_eq!(idx.n_evap_hydros, 1);
        assert_eq!(idx.evap_hydro_indices, vec![0]);
        assert_eq!(idx.evap_indices.len(), 1);

        let ei = idx.evap_indices(0);
        // 3 columns placed after generation_end (which equals excess.end = 15)
        assert_eq!(ei.q_ev_col, 15);
        assert_eq!(ei.f_evap_plus_col, 16);
        assert_eq!(ei.f_evap_minus_col, 17);
        // Row placed after load_balance.end = 5.
        assert_eq!(ei.evap_row, 5);
    }

    // 2 evaporation hydros — verify column/row ranges are
    // contiguous and non-overlapping with FPHA ranges.
    //
    // N=4, L=0, T=0, Ln=0, B=1, K=1, no penalty, 1 FPHA hydro (index 0, 3 planes),
    // 2 evap hydros (indices 1, 2).
    // theta = 4*(3+0) = 12
    // decision_start = 13
    // turbine:    [13, 17)  (4 hydros * 1 block)
    // spillage:   [17, 21)
    // diversion:  [21, 25)  (4 hydros * 1 block)
    // deficit:    [25, 26)  (1 bus * 1 block)
    // excess:     [26, 27)
    // generation: [27, 28) (1 FPHA hydro * 1 block)
    //
    // Row layout (Phase 1: state-fixing rows removed):
    // z_inflow rows = 0..4  (start at row 0)
    // water_balance_start = 4  (z_inflow rows end)
    // load_balance_start = 4 + 4 = 8
    // load_balance_end   = 8 + 1*1 = 9
    // fpha_rows[0].start = 9
    // fpha_row_cursor after FPHA = 9 + 3*1 = 12
    // evap cols: [28, 34)   (2 evap hydros * 3 = 6 columns)
    // evap_row[0] = 12, evap_row[1] = 13
    #[test]
    fn evap_two_hydros_with_fpha_contiguous() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(4, 0, 0, 0, 1, 1, false),
            &fpha(vec![0], vec![3]),
            &evap(vec![1, 2]),
        );

        assert_eq!(idx.n_evap_hydros, 2);
        assert_eq!(idx.evap_hydro_indices, vec![1, 2]);

        let ei0 = idx.evap_indices(0);
        let ei1 = idx.evap_indices(1);

        // Columns start at generation_end = 28
        assert_eq!(ei0.q_ev_col, 28);
        assert_eq!(ei0.f_evap_plus_col, 29);
        assert_eq!(ei0.f_evap_minus_col, 30);

        assert_eq!(ei1.q_ev_col, 31);
        assert_eq!(ei1.f_evap_plus_col, 32);
        assert_eq!(ei1.f_evap_minus_col, 33);

        // Rows placed after fpha_rows region: fpha_row_cursor = 9 + 3*1 = 12.
        assert_eq!(ei0.evap_row, 12);
        assert_eq!(ei1.evap_row, 13);

        // Evap rows do not overlap FPHA rows
        assert!(ei0.evap_row > idx.fpha_rows[0].start);
    }

    // new() produces empty evaporation fields.
    #[test]
    fn new_evap_ranges_are_empty() {
        let idx = StageIndexer::new(3, 2);
        assert_eq!(idx.n_evap_hydros, 0);
        assert!(idx.evap_hydro_indices.is_empty());
        assert!(idx.evap_indices.is_empty());
    }

    // ── Withdrawal slack field tests ───────────────────────────────────────

    // AC: with_equipment_and_evaporation, N=3 hydros, 1 evap hydro →
    // withdrawal_slack_neg starts at evap_col_end, withdrawal_slack_pos follows.
    //
    // N=3, L=0, T=0, Ln=0, B=1, K=1, no penalty, no FPHA, 1 evap hydro.
    // theta = N*(3+L) = 3*(3+0) = 9
    // decision_start = 10
    // turbine:    [10, 13)  (3 hydros * 1 block)
    // spillage:   [13, 16)
    // diversion:  [16, 19)  (3 hydros * 1 block)
    // deficit:    [19, 20)  (1 bus * 1 block)
    // excess:     [20, 21)
    // generation: empty (no FPHA)
    // evap cols:  [21, 24)  (1 evap hydro * 3 columns)
    // withdrawal_slack_neg: [24, 27)  (3 hydros)
    // withdrawal_slack_pos: [27, 30)  (3 hydros)
    #[test]
    fn withdrawal_slack_with_equipment_and_evaporation_n3_evap1() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(3, 0, 0, 0, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![0]),
        );

        assert!(idx.has_withdrawal);
        let evap_col_end = idx.evap_indices(0).f_evap_minus_col + 1;
        assert_eq!(
            idx.withdrawal_slack_neg.start, evap_col_end,
            "withdrawal_slack_neg.start must equal evap_col_end"
        );
        assert_eq!(idx.withdrawal_slack_neg.len(), 3);
        assert_eq!(idx.withdrawal_slack_neg, 24..27);
        assert_eq!(idx.withdrawal_slack_pos.start, idx.withdrawal_slack_neg.end);
        assert_eq!(idx.withdrawal_slack_pos.len(), 3);
        assert_eq!(idx.withdrawal_slack_pos, 27..30);
    }

    // AC: with_equipment_and_evaporation, N=0 → both withdrawal slacks are 0..0.
    #[test]
    fn withdrawal_slack_zero_hydros_is_empty() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(0, 0, 0, 0, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert!(!idx.has_withdrawal);
        assert_eq!(idx.withdrawal_slack_neg, 0..0);
        assert_eq!(idx.withdrawal_slack_pos, 0..0);
    }

    // AC: new() → both withdrawal slacks are 0..0.
    #[test]
    fn withdrawal_slack_from_new_is_empty() {
        let idx = StageIndexer::new(3, 2);
        assert!(!idx.has_withdrawal);
        assert_eq!(idx.withdrawal_slack_neg, 0..0);
        assert_eq!(idx.withdrawal_slack_pos, 0..0);
    }

    // AC: both withdrawal_slack ranges have length == hydro_count.
    #[test]
    fn withdrawal_slack_length_equals_hydro_count() {
        for n in [1_usize, 5] {
            let idx = StageIndexer::with_equipment_and_evaporation(
                &EquipmentCounts {
                    hydro_count: n,
                    max_par_order: 0,
                    n_thermals: 0,
                    n_lines: 0,
                    n_buses: 1,
                    n_blks: 1,
                    has_inflow_penalty: false,
                    max_deficit_segments: 1,
                    n_anticipated: 0,
                    k_max: 0,
                    anticipated_lead_stages: vec![],
                    anticipated_thermal_indices: vec![],
                },
                &fpha(vec![], vec![]),
                &evap(vec![]),
            );

            assert!(idx.has_withdrawal, "has_withdrawal must be true for n={n}");
            assert_eq!(
                idx.withdrawal_slack_neg.len(),
                n,
                "withdrawal_slack_neg length must equal hydro_count for n={n}"
            );
            assert_eq!(
                idx.withdrawal_slack_pos.len(),
                n,
                "withdrawal_slack_pos length must equal hydro_count for n={n}"
            );
        }
    }

    // AC: withdrawal_slack_neg.start == evap_col_end, pos starts at neg.end.
    //
    // N=2, L=0, no penalty, no FPHA, 1 evap hydro.
    // evap cols: [excess_end, excess_end+3) = [15, 18)
    // withdrawal_slack_neg: [18, 20)
    // withdrawal_slack_pos: [20, 22)
    #[test]
    fn withdrawal_slack_immediately_after_evap_columns() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(2, 0, 0, 0, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![0]),
        );

        assert_eq!(
            idx.withdrawal_slack_neg.start, 18,
            "withdrawal_slack_neg must start at evap_col_end=18"
        );
        assert_eq!(idx.withdrawal_slack_neg.len(), 2);
        assert_eq!(idx.withdrawal_slack_neg, 18..20);
        assert_eq!(idx.withdrawal_slack_pos, 20..22);
        // Operational slacks must start at pos.end
        assert_eq!(
            idx.outflow_below_slack.start, idx.withdrawal_slack_pos.end,
            "outflow_below_slack must start at withdrawal_slack_pos.end"
        );
    }

    // EvaporationIndices is Debug + Clone + Copy.
    #[test]
    fn evap_indices_debug_clone_copy() {
        use super::EvaporationIndices;
        let ei = EvaporationIndices {
            q_ev_col: 10,
            f_evap_plus_col: 11,
            f_evap_minus_col: 12,
            evap_row: 5,
        };
        let cloned = ei;
        assert_eq!(cloned.q_ev_col, 10);
        assert_eq!(cloned.evap_row, 5);
        let debug_str = format!("{ei:?}");
        assert!(debug_str.contains("EvaporationIndices"));
    }

    // FphaRowRange is Debug + Clone + Copy.
    #[test]
    fn fpha_row_range_debug_clone_copy() {
        let r = FphaRowRange {
            start: 42,
            planes_per_block: 5,
        };
        let cloned = r;
        assert_eq!(cloned.start, 42);
        assert_eq!(cloned.planes_per_block, 5);
        let debug_str = format!("{r:?}");
        assert!(debug_str.contains("FphaRowRange"));
    }

    // new() produces empty FPHA ranges.
    #[test]
    fn new_fpha_ranges_are_empty() {
        let idx = StageIndexer::new(3, 2);
        assert!(idx.generation.is_empty());
        assert_eq!(idx.n_fpha_hydros, 0);
        assert!(idx.fpha_hydro_indices.is_empty());
        assert!(idx.fpha_rows.is_empty());
    }

    // Adjacency invariant extended: generation immediately follows the prior region.
    #[test]
    fn extended_adjacency_invariant_with_fpha() {
        // N=2, L=1, T=1, Ln=1, B=1, K=1, no penalty, 1 FPHA hydro.
        // theta=8, decision_start=9
        // turbine:[9,11), spillage:[11,13), diversion:[13,15), thermal:[15,16),
        // line_fwd:[16,17), line_rev:[17,18), deficit:[18,19), excess:[19,20)
        // generation:[20,21) (1 FPHA * 1 block, after excess.end since no penalty)
        let idx =
            StageIndexer::with_equipment(&eq(2, 1, 1, 1, 1, 1, false), &fpha(vec![0], vec![3]));

        assert_eq!(idx.turbine.start, idx.theta + 1);
        assert_eq!(idx.spillage.start, idx.turbine.end);
        assert_eq!(idx.diversion.start, idx.spillage.end);
        assert_eq!(idx.thermal.start, idx.diversion.end);
        assert_eq!(idx.line_fwd.start, idx.thermal.end);
        assert_eq!(idx.line_rev.start, idx.line_fwd.end);
        assert_eq!(idx.deficit.start, idx.line_rev.end);
        assert_eq!(idx.excess.start, idx.deficit.end);
        // generation follows excess (no penalty)
        assert_eq!(idx.generation.start, idx.excess.end);
        assert_eq!(idx.generation.len(), 1);
    }

    // ── Diversion field tests ──────────────────────────────────────────────

    // Diversion range: N=3, K=2 → diversion.len() = 6, contiguous with spillage.
    #[test]
    fn test_diversion_range_n3_l0_k2() {
        let idx = StageIndexer::with_equipment(&eq(3, 0, 0, 0, 1, 2, false), &fpha(vec![], vec![]));

        assert_eq!(idx.diversion.start, idx.spillage.end);
        assert_eq!(idx.diversion.len(), 3 * 2);
        assert_eq!(idx.thermal.start, idx.diversion.end);
    }

    // Diversion range: N=0 → diversion is empty.
    #[test]
    fn test_diversion_zero_hydros() {
        let idx = StageIndexer::with_equipment(&eq(0, 0, 1, 0, 1, 1, false), &fpha(vec![], vec![]));

        assert!(idx.diversion.is_empty());
    }

    // ── z_inflow field tests ───────────────────────────────────────────────

    // z_inflow starts at N*(1+L) (fixed offset) and has length hydro_count.
    #[test]
    fn z_inflow_range_new_constructor() {
        let idx = StageIndexer::new(3, 2);
        // z_inflow at N*(1+L) = 9..12
        assert_eq!(idx.z_inflow, 9..12);
        assert_eq!(idx.z_inflow.len(), idx.hydro_count);
    }

    // z_inflow is empty when hydro_count == 0.
    #[test]
    fn z_inflow_range_zero_hydros() {
        let idx = StageIndexer::new(0, 0);
        assert!(idx.z_inflow.is_empty());
        assert!(idx.z_inflow_rows.is_empty());
        assert_eq!(idx.z_inflow_row_start, 0);
    }

    // z_inflow rows start at row 0.
    #[test]
    fn z_inflow_row_fields() {
        let idx = StageIndexer::new(5, 1);
        // z_inflow rows start at row 0, length = hydro_count = 5.
        assert_eq!(idx.z_inflow_rows, 0..5);
        assert_eq!(idx.z_inflow_row_start, 0);
        assert_eq!(idx.z_inflow.len(), 5);
    }

    // z_inflow has correct length and rows for with_equipment constructor.
    #[test]
    fn z_inflow_range_with_equipment() {
        let idx = StageIndexer::with_equipment(&eq(2, 1, 1, 1, 1, 1, false), &fpha(vec![], vec![]));
        // N*(1+L) = 2*(1+1) = 4 (column range unchanged)
        assert_eq!(idx.z_inflow, 4..6);
        assert_eq!(idx.z_inflow.len(), 2);
        // z_inflow rows start at row 0, length = hydro_count = 2.
        assert_eq!(idx.z_inflow_rows, 0..2);
        assert_eq!(idx.z_inflow_row_start, 0);
    }

    // z_inflow placed correctly for single hydro, no lags case.
    #[test]
    fn z_inflow_single_hydro_no_lags() {
        let idx = StageIndexer::new(1, 0);
        // N*(1+L) = 1*(1+0) = 1, z_inflow at 1..2
        assert_eq!(idx.z_inflow, 1..2);
        assert_eq!(idx.z_inflow.len(), 1);
    }

    // ── Nonzero state mask tests ───────────────────────────────────────────

    #[test]
    fn nonzero_mask_default_is_empty() {
        let idx = StageIndexer::new(4, 6);
        assert!(
            idx.nonzero_state_indices.is_empty(),
            "default mask must be empty (dense path)"
        );
    }

    #[test]
    fn nonzero_mask_mixed_ar_orders() {
        // 4 hydros (N=4), max_par_order=6 (L=6), ar_orders=[0, 1, 3, 6]
        // inflow_lags.start = N = 4
        // Lag-major layout: slot = 4 + lag * N + h
        let mut idx = StageIndexer::new(4, 6);
        idx.set_nonzero_mask(&[0, 1, 3, 6], &[]);

        // Storage: [0, 1, 2, 3]
        // lag0: h1→4+0*4+1=5, h2→6, h3→7
        // lag1: h2→4+1*4+2=10, h3→11
        // lag2: h2→4+2*4+2=14, h3→15
        // lag3: h3→4+3*4+3=19
        // lag4: h3→4+4*4+3=23
        // lag5: h3→4+5*4+3=27
        // Total: 4 + 0 + 1 + 3 + 6 = 14
        assert_eq!(
            idx.nonzero_state_indices.len(),
            14,
            "mask length: 4 storage + 0 + 1 + 3 + 6 = 14"
        );

        assert_eq!(&idx.nonzero_state_indices[..4], &[0, 1, 2, 3]);
        assert_eq!(
            &idx.nonzero_state_indices[4..],
            &[5, 6, 7, 10, 11, 14, 15, 19, 23, 27]
        );

        assert!(idx.nonzero_state_indices.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn nonzero_mask_zero_par_order() {
        // max_par_order=0: no lags, mask = storage only
        let mut idx = StageIndexer::new(3, 0);
        idx.set_nonzero_mask(&[0, 0, 0], &[]);
        assert_eq!(idx.nonzero_state_indices.len(), 3);
        assert_eq!(&idx.nonzero_state_indices, &[0, 1, 2]);
    }

    // ── state_to_lp_column precompute tests ─────────────────────────────────

    /// A finalized indexer carrying storage + AR lags + anticipated thermals
    /// (every `state_to_lp_column` branch) must have
    /// `lp_column_for_state(j) == state_to_lp_column(j)` for every state index.
    #[test]
    fn lp_column_map_matches_resolver_with_lags_and_anticipated() {
        // hydro_count=3, max_par_order=2, n_anticipated=2 (K = [1, 2], k_max=2).
        let mut idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 3,
                max_par_order: 2,
                n_thermals: 2,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 2,
                k_max: 2,
                anticipated_lead_stages: vec![1, 2],
                anticipated_thermal_indices: vec![0, 1],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Finalize both layout-derived caches, as study setup does.
        let lag_counts = vec![2_usize; idx.hydro_count];
        let anticipated_k = idx.anticipated_lead_stages.clone();
        idx.set_nonzero_mask(&lag_counts, &anticipated_k);
        idx.finalize_state_column_map();

        assert_eq!(idx.state_to_lp_column_map.len(), idx.n_state);
        for j in 0..idx.n_state {
            assert_eq!(
                idx.lp_column_for_state(j),
                idx.state_to_lp_column(j),
                "finalized map must match the resolver at j={j}"
            );
        }

        // Un-finalized clone: the fallback must still match the resolver.
        let mut unfinalized = idx.clone();
        unfinalized.state_to_lp_column_map.clear();
        for j in 0..unfinalized.n_state {
            assert_eq!(
                unfinalized.lp_column_for_state(j),
                unfinalized.state_to_lp_column(j),
                "fallback must match the resolver at j={j} when un-finalized"
            );
        }
    }

    /// Storage-only (`max_par_order == 0`, no anticipated): the mask is exactly
    /// `[0, n_state)` ascending and `lp_column_for_state(j) == j` — the
    /// dense→sparse bit-identity premise for the unified cut-row loop.
    #[test]
    fn lp_column_map_storage_only_mask_is_full_range() {
        let mut idx = StageIndexer::new(3, 0);
        idx.set_nonzero_mask(&[0, 0, 0], &[]);
        idx.finalize_state_column_map();

        assert_eq!(idx.nonzero_state_indices, vec![0, 1, 2]);
        assert_eq!(idx.nonzero_state_indices.len(), idx.n_state);
        for j in 0..idx.n_state {
            assert_eq!(idx.lp_column_for_state(j), j);
        }
    }

    #[test]
    fn nonzero_mask_all_full_order() {
        // All hydros at max AR order: mask covers all n_state indices
        let mut idx = StageIndexer::new(2, 3);
        idx.set_nonzero_mask(&[3, 3], &[]);
        // n_state = 2*(1+3) = 8, mask should have 2 + 2*3 = 8
        assert_eq!(idx.nonzero_state_indices.len(), 8);
        assert_eq!(idx.nonzero_state_indices.len(), idx.n_state);
    }

    /// Regression test for the PAR(p)-A cut sparse-mask bug.
    ///
    /// `lag_counts` (formerly `ar_orders`) is the per-hydro count of lag-state
    /// slots that may carry non-zero cut coefficients — equal to
    /// `PrecomputedPar::effective_lag_count(h)`. When PAR(p)-A annual is active
    /// on a hydro this is `max_par_order` (= 12) even though the classical AR
    /// order is smaller, because `ψ̂/12` fills the trailing lag slots.
    ///
    /// Before the fix at setup/mod.rs (which passed `par.order(h)` here), the
    /// cut row omitted state coefficients on slots `order..max_par_order`,
    /// producing over-estimating cuts (LB > UB at convergence).
    #[test]
    fn nonzero_mask_par_a_includes_full_psi_stride() {
        // Two hydros: hydro 0 has classical AR(4); hydro 1 has PAR(4)-A and
        // therefore uses all 12 lag slots. max_par_order = 12 (widened by
        // PrecomputedPar when any model has an annual component).
        let mut idx = StageIndexer::new(2, 12);
        idx.set_nonzero_mask(&[4, 12], &[]);

        // n_state = 2 * (1 + 12) = 26.
        // Mask = [storage 0..2] + [lag * 2 + h for lag in 0..lag_count[h]]
        //      = [0, 1] + [hydro 0 lags 0..4] + [hydro 1 lags 0..12]
        //      = 2 + 4 + 12 = 18 entries.
        assert_eq!(
            idx.nonzero_state_indices.len(),
            18,
            "PAR-A hydro must contribute all 12 lag slots to the cut mask; \
             omitting slots 4..12 (where ψ̂/12 lives) shifts the cut hyperplane \
             above the LP value at the visited state (over-estimating cuts)."
        );

        // Storage indices.
        assert_eq!(&idx.nonzero_state_indices[..2], &[0, 1]);

        // Hydro 0 (lag_count = 4): expect lag slots at indices
        //   inflow_lags.start + lag * hydro_count + h = 2 + lag*2 + 0 for lag in 0..4
        // → {2, 4, 6, 8}.
        // Hydro 1 (lag_count = 12): expect lag slots at indices
        //   2 + lag*2 + 1 for lag in 0..12 → {3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25}.
        // Mask is sorted globally. Confirm a few discriminating positions.
        assert!(
            idx.nonzero_state_indices.contains(&25),
            "lag-11 slot for hydro 1 (the trailing PAR-A annual slot) must be in the mask"
        );
        assert!(
            !idx.nonzero_state_indices.contains(&10),
            "lag-4 slot for hydro 0 (classical AR(4)) must NOT be in the mask"
        );
        // Sorted.
        assert!(idx.nonzero_state_indices.windows(2).all(|w| w[0] < w[1]));
    }

    // ── Anticipated-state nonzero mask tests ───────────────────────────────

    /// Every anticipated plant uses every slot (`K_i == k_max`): all
    /// `n_anticipated * k_max` anticipated indices are included in the mask.
    #[test]
    fn nonzero_mask_anticipated_state_full_kmax() {
        // 2 anticipated plants, k_max = 3, no hydros, no lags.
        // anticipated_state.start = 0 (no storage, no lag block).
        // Layout: start + slot * n_anticipated + plant.
        let mut idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(0, 0, 0, 0, 1, 1, false, 2, 3),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert_eq!(idx.anticipated_state.start, 0);
        assert_eq!(idx.n_anticipated, 2);
        assert_eq!(idx.k_max, 3);

        idx.set_nonzero_mask(&[], &[3, 3]);

        // Every slot is occupied, so all 6 indices appear.
        // slot=0: 0, 1; slot=1: 2, 3; slot=2: 4, 5.
        assert_eq!(idx.nonzero_state_indices, vec![0, 1, 2, 3, 4, 5]);
    }

    /// `K_i < k_max` for some plants: padded slots are excluded.
    /// Uses the configuration from acceptance criterion 2: `n_anticipated = 2`,
    /// `k_max = 3`, `anticipated_lead_stages = [3, 1]`.
    #[test]
    fn nonzero_mask_anticipated_state_partial_padding() {
        // 3 hydros, max_par_order = 2, 2 anticipated plants, k_max = 3.
        // inflow_lags = [3, 9), anticipated_state.start = 9.
        let mut idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, 2, 3),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert_eq!(idx.anticipated_state.start, 9);
        idx.set_nonzero_mask(&[2, 2, 2], &[3, 1]);

        // Storage [0, 1, 2] + lag (h0,h1,h2 full = 6 slots) +
        // anticipated: slot=0 plant=0 → 9, plant=1 → 10 (K_1=1 so slot 0 included);
        //              slot=1 plant=0 → 11 (K_0=3); plant=1 → padded (slot 1 >= 1).
        //              slot=2 plant=0 → 13 (K_0=3); plant=1 → padded.
        // The anticipated portion expected: [9, 10, 11, 13].
        let mask = &idx.nonzero_state_indices;
        let ant_portion: Vec<usize> = mask.iter().copied().filter(|&i| i >= 9).collect();
        assert_eq!(ant_portion, vec![9, 10, 11, 13]);
        // Padded slots NOT present.
        assert!(!mask.contains(&12));
        assert!(!mask.contains(&14));
    }

    /// Anticipated-only: no hydros, no lags, only anticipated state.
    #[test]
    fn nonzero_mask_anticipated_state_only_no_hydros() {
        let mut idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(0, 0, 0, 0, 1, 1, false, 1, 2),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert_eq!(idx.hydro_count, 0);
        assert_eq!(idx.anticipated_state.start, 0);

        idx.set_nonzero_mask(&[], &[2]);

        // Only anticipated indices: slot=0 plant=0 → 0; slot=1 plant=0 → 1.
        assert_eq!(idx.nonzero_state_indices, vec![0, 1]);
    }

    /// Heterogeneous `K_i` across plants, including a plant with `K_i = k_max`
    /// and another with `K_i < k_max`.
    #[test]
    fn nonzero_mask_anticipated_state_mixed_k_values() {
        // n_anticipated = 3, k_max = 4. Lead stages = [4, 2, 1].
        // anticipated_state.start = 0 (no hydros).
        // eq_with_anticipated defaults to K_i = k_max for all plants, but we
        // need [4, 2, 1]. Build `EquipmentCounts` directly to override the
        // anticipated_lead_stages field.
        let counts = EquipmentCounts {
            hydro_count: 0,
            max_par_order: 0,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 1,
            n_blks: 1,
            has_inflow_penalty: false,
            max_deficit_segments: 1,
            n_anticipated: 3,
            k_max: 4,
            anticipated_lead_stages: vec![4, 2, 1],
            anticipated_thermal_indices: vec![0, 1, 2],
        };
        let mut idx = StageIndexer::with_equipment_and_evaporation(
            &counts,
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        idx.set_nonzero_mask(&[], &[4, 2, 1]);

        // slot=0 plant=0→0, plant=1→1, plant=2→2 (all K_i > 0).
        // slot=1 plant=0→3, plant=1→4 (K_1=2). plant=2 padded.
        // slot=2 plant=0→6 (K_0=4). plant=1 padded. plant=2 padded.
        // slot=3 plant=0→9 (K_0=4). Others padded.
        // Expected: [0, 1, 2, 3, 4, 6, 9].
        assert_eq!(idx.nonzero_state_indices, vec![0, 1, 2, 3, 4, 6, 9]);
    }

    /// `n_anticipated == 0` reproduces the pre-ticket behaviour exactly.
    #[test]
    fn nonzero_mask_anticipated_state_zero_anticipated_matches_existing() {
        let mut idx_with = StageIndexer::new(4, 6);
        idx_with.set_nonzero_mask(&[0, 1, 3, 6], &[]);

        // Same expected mask as the existing `nonzero_mask_mixed_ar_orders`
        // test: [0,1,2,3] (storage) + [5,6,7,10,11,14,15,19,23,27] (lags).
        assert_eq!(
            idx_with.nonzero_state_indices,
            vec![0, 1, 2, 3, 5, 6, 7, 10, 11, 14, 15, 19, 23, 27]
        );
    }

    /// The extended mask is sorted ascending with no duplicates.
    #[test]
    fn nonzero_mask_anticipated_state_sorted_ascending() {
        // Mixed configuration: 3 hydros with mixed lag_counts + 2 anticipated
        // plants with mixed K_i. The slot-major iteration over anticipated
        // must keep the global mask sorted.
        let mut idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, 2, 3),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        idx.set_nonzero_mask(&[1, 2, 0], &[2, 3]);

        assert!(
            idx.nonzero_state_indices.windows(2).all(|w| w[0] < w[1]),
            "mask must be strictly ascending with no duplicates: {:?}",
            idx.nonzero_state_indices
        );
    }

    /// Plant with `K_i == k_max` (boundary, no padding): all its slots are
    /// included.
    #[test]
    fn nonzero_mask_anticipated_state_boundary_k_eq_kmax() {
        // 1 anticipated plant, K_0 = k_max = 3, no hydros.
        let mut idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(0, 0, 0, 0, 1, 1, false, 1, 3),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        idx.set_nonzero_mask(&[], &[3]);

        // All k_max slots included: slot 0,1,2 → indices 0,1,2.
        assert_eq!(idx.nonzero_state_indices, vec![0, 1, 2]);
    }

    /// `K_i == 0` excludes all slots for that plant (defensive — the parse
    /// layer rejects `K_i == 0`, but the helper must remain robust if
    /// invoked with zero).
    #[test]
    fn nonzero_mask_anticipated_state_boundary_k_zero_excluded() {
        // 2 anticipated plants, k_max = 2. Lead stages = [2, 0].
        let counts = EquipmentCounts {
            hydro_count: 0,
            max_par_order: 0,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 1,
            n_blks: 1,
            has_inflow_penalty: false,
            max_deficit_segments: 1,
            n_anticipated: 2,
            k_max: 2,
            anticipated_lead_stages: vec![2, 0],
            anticipated_thermal_indices: vec![0, 1],
        };
        let mut idx = StageIndexer::with_equipment_and_evaporation(
            &counts,
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        idx.set_nonzero_mask(&[], &[2, 0]);

        // Plant 0 (K_0=2) emits slot=0→0, slot=1→2. Plant 1 (K_1=0) emits
        // nothing. Expected mask: [0, 2].
        assert_eq!(idx.nonzero_state_indices, vec![0, 2]);
    }

    // ── Anticipated thermal state tests ────────────────────────────────────

    /// Test helper: build `EquipmentCounts` with explicit anticipated thermal
    /// fields.
    fn eq_with_anticipated(
        hydro_count: usize,
        max_par_order: usize,
        n_thermals: usize,
        n_lines: usize,
        n_buses: usize,
        n_blks: usize,
        has_inflow_penalty: bool,
        n_anticipated: usize,
        k_max: usize,
    ) -> EquipmentCounts {
        // Default the per-plant K_i array to a uniform `k_max` of length
        // `n_anticipated` so debug asserts on per-plant lead-stage
        // bookkeeping hold. Tests that need a mixed K_i array must construct
        // `EquipmentCounts` directly.
        let anticipated_lead_stages = if n_anticipated == 0 {
            vec![]
        } else {
            vec![k_max; n_anticipated]
        };
        let anticipated_thermal_indices = if n_anticipated == 0 {
            vec![]
        } else {
            (0..n_anticipated).collect()
        };
        EquipmentCounts {
            hydro_count,
            max_par_order,
            n_thermals,
            n_lines,
            n_buses,
            n_blks,
            has_inflow_penalty,
            max_deficit_segments: 1,
            n_anticipated,
            k_max,
            anticipated_lead_stages,
            anticipated_thermal_indices,
        }
    }

    /// `StageIndexer::new` always produces an empty anticipated block.
    #[test]
    fn anticipated_state_empty_for_new() {
        let idx = StageIndexer::new(3, 2);
        assert_eq!(idx.anticipated_state, 0..0);
        assert_eq!(idx.anticipated_state_fixing, 0..0);
        assert_eq!(idx.n_anticipated, 0);
        assert_eq!(idx.k_max, 0);
    }

    /// SS5.5.3-style example with two anticipated plants and `K_max = 3`:
    /// `anticipated_state` consumes 6 columns starting at `N*(1+L) = 9`,
    /// shifting `z_inflow`, `storage_in` and `theta` by 6.
    #[test]
    fn anticipated_state_layout_n3_l2_nant2_kmax3() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, 2, 3),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert_eq!(idx.anticipated_state, 9..15);
        // anticipated_state_fixing is a permanent empty sentinel; state fixing uses column bounds.
        assert_eq!(idx.anticipated_state_fixing, 0..0);
        assert_eq!(idx.z_inflow, 15..18);
        assert_eq!(idx.storage_in, 18..21);
        assert_eq!(idx.theta, 21);
        assert_eq!(idx.n_state, 15);
        assert_eq!(idx.n_anticipated, 2);
        assert_eq!(idx.k_max, 3);
    }

    /// Zero hydros but two anticipated plants: `anticipated_state` collapses
    /// to `0..6`, `z_inflow` and `storage_in` are empty, `theta == 6`.
    #[test]
    fn anticipated_state_layout_n0_nant2_kmax3() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(0, 0, 0, 0, 1, 1, false, 2, 3),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert_eq!(idx.anticipated_state, 0..6);
        // anticipated_state_fixing is a permanent empty sentinel; state fixing uses column bounds.
        assert_eq!(idx.anticipated_state_fixing, 0..0);
        assert_eq!(idx.n_state, 6);
        assert_eq!(idx.z_inflow, 6..6);
        assert_eq!(idx.storage_in, 6..6);
        assert_eq!(idx.theta, 6);
        assert_eq!(idx.n_anticipated, 2);
        assert_eq!(idx.k_max, 3);
    }

    /// When `n_anticipated == 0` the layout produced by
    /// `with_equipment_and_evaporation` must match the pre-ticket layout
    /// field-for-field against `StageIndexer::new`.
    #[test]
    fn anticipated_state_no_thermals_matches_existing() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(3, 2, 0, 0, 0, 0, false),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let base = StageIndexer::new(3, 2);

        assert_eq!(idx.storage, base.storage);
        assert_eq!(idx.inflow_lags, base.inflow_lags);
        assert_eq!(idx.anticipated_state, 0..0);
        assert_eq!(idx.anticipated_state_fixing, 0..0);
        assert_eq!(idx.z_inflow, base.z_inflow);
        assert_eq!(idx.storage_in, base.storage_in);
        assert_eq!(idx.theta, base.theta);
        assert_eq!(idx.n_state, base.n_state);
        assert_eq!(idx.storage_fixing, base.storage_fixing);
        assert_eq!(idx.lag_fixing, base.lag_fixing);
        assert_eq!(idx.z_inflow_rows, base.z_inflow_rows);
        assert_eq!(idx.z_inflow_row_start, base.z_inflow_row_start);
        assert_eq!(idx.n_anticipated, 0);
        assert_eq!(idx.k_max, 0);
    }

    /// `anticipated_state_fixing` is always empty (`0..0`);
    /// state pinning uses column bounds, not rows.
    #[test]
    fn anticipated_state_fixing_mirrors_state() {
        for (n_anticipated, k_max) in [(0, 0), (1, 1), (1, 3), (2, 3), (5, 4)] {
            let idx = StageIndexer::with_equipment_and_evaporation(
                &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, n_anticipated, k_max),
                &fpha(vec![], vec![]),
                &evap(vec![]),
            );
            assert_eq!(
                idx.anticipated_state_fixing,
                0..0,
                "anticipated_state_fixing must be 0..0 (sentinel) for n_anticipated={n_anticipated} k_max={k_max}"
            );
        }
    }

    /// All three state-fixing row ranges are permanent empty sentinels in every constructor.
    ///
    /// Covers `StageIndexer::new`, `StageIndexer::with_equipment`, and
    /// `StageIndexer::with_equipment_and_evaporation` (with anticipated thermals).
    #[test]
    fn state_fixing_rows_collapsed_to_empty_in_all_constructors() {
        // new() constructor
        let idx_new = StageIndexer::new(3, 2);
        assert_eq!(
            idx_new.storage_fixing,
            0..0,
            "new: storage_fixing must be 0..0"
        );
        assert_eq!(idx_new.lag_fixing, 0..0, "new: lag_fixing must be 0..0");
        assert_eq!(
            idx_new.anticipated_state_fixing,
            0..0,
            "new: anticipated_state_fixing must be 0..0"
        );

        // with_equipment() constructor (delegates to with_equipment_and_evaporation)
        let idx_eq =
            StageIndexer::with_equipment(&eq(3, 2, 0, 0, 0, 0, false), &fpha(vec![], vec![]));
        assert_eq!(
            idx_eq.storage_fixing,
            0..0,
            "with_equipment: storage_fixing must be 0..0"
        );
        assert_eq!(
            idx_eq.lag_fixing,
            0..0,
            "with_equipment: lag_fixing must be 0..0"
        );
        assert_eq!(
            idx_eq.anticipated_state_fixing,
            0..0,
            "with_equipment: anticipated_state_fixing must be 0..0"
        );

        // with_equipment_and_evaporation() with anticipated thermals (n_anticipated=1, k_max=2)
        let idx_ant = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, 1, 2),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert_eq!(
            idx_ant.storage_fixing,
            0..0,
            "with_equipment_and_evaporation: storage_fixing must be 0..0"
        );
        assert_eq!(
            idx_ant.lag_fixing,
            0..0,
            "with_equipment_and_evaporation: lag_fixing must be 0..0"
        );
        assert_eq!(
            idx_ant.anticipated_state_fixing,
            0..0,
            "with_equipment_and_evaporation: anticipated_state_fixing must be 0..0"
        );
        // Column range anticipated_state is unchanged (N*(1+L) .. N*(1+L) + k_max).
        assert_eq!(
            idx_ant.anticipated_state,
            9..11,
            "anticipated_state column range must be N*(1+L)..N*(1+L)+n_ant*k_max = 9..11"
        );
        // water_balance starts immediately after z_inflow (hydro_count rows from row 0).
        assert_eq!(
            idx_ant.water_balance.start, idx_ant.z_inflow_rows.end,
            "water_balance.start must equal z_inflow_rows.end"
        );
    }

    /// `z_inflow.start` is exactly `N*(1+L) + n_anticipated * k_max`.
    #[test]
    fn anticipated_state_shifts_z_inflow() {
        // SS5.5.3-extended: N=3, L=2, n_anticipated=2, k_max=3.
        let idx_small = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, 2, 3),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert_eq!(idx_small.z_inflow.start, 3 * (1 + 2) + 2 * 3);

        // Production scale: N=160, L=12, n_anticipated=10, k_max=4.
        let idx_prod = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(160, 12, 0, 0, 1, 1, false, 10, 4),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert_eq!(idx_prod.z_inflow.start, 160 * (1 + 12) + 10 * 4);
    }

    /// Minimal non-empty anticipated block: `n_anticipated=1`, `k_max=1` → length 1.
    #[test]
    fn anticipated_state_boundary_kmax_one() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(2, 1, 0, 0, 1, 1, false, 1, 1),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        // N*(1+L) = 2*2 = 4
        assert_eq!(idx.anticipated_state, 4..5);
        assert_eq!(idx.anticipated_state.len(), 1);
        assert_eq!(idx.n_state, 5); // base 4 + 1
        assert_eq!(idx.z_inflow, 5..7);
        assert_eq!(idx.theta, 9); // 5 + 2*(storage_in width) = 5 + 2 + 2
    }

    /// Acceptance side of the `debug_assert`: (`n_anticipated=0`, `k_max=0`)
    /// produces a layout identical to the existing one.
    #[test]
    fn anticipated_state_boundary_n_anticipated_zero() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, 0, 0),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Layout must collapse back to the pre-anticipated state form.
        assert_eq!(idx.anticipated_state, 0..0);
        assert_eq!(idx.z_inflow, 9..12);
        assert_eq!(idx.storage_in, 12..15);
        assert_eq!(idx.theta, 15);
        assert_eq!(idx.n_state, 9);
    }

    /// Debug-only assertion: building with `n_anticipated > 0` but `k_max == 0`
    /// must trigger the documented `debug_assert!` failure.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "k_max must be >= 1 when n_anticipated > 0")]
    fn anticipated_state_debug_assert_kmax_zero_when_nant_nonzero() {
        let _ = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(1, 0, 0, 0, 1, 1, false, 1, 0),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
    }

    /// `n_state == N*(1+L) + n_anticipated * k_max` exactly.
    #[test]
    fn anticipated_state_n_state_formula() {
        for (n, l, n_anticipated, k_max) in [
            (3_usize, 2_usize, 0_usize, 0_usize),
            (3, 2, 2, 3),
            (160, 12, 10, 4),
            (1, 0, 1, 1),
        ] {
            let idx = StageIndexer::with_equipment_and_evaporation(
                &eq_with_anticipated(n, l, 0, 0, 1, 1, false, n_anticipated, k_max),
                &fpha(vec![], vec![]),
                &evap(vec![]),
            );
            assert_eq!(
                idx.n_state,
                n * (1 + l) + n_anticipated * k_max,
                "n_state mismatch for (N={n}, L={l}, n_anticipated={n_anticipated}, k_max={k_max})"
            );
        }
    }

    // ── Anticipated decision column tests ──────────────────────────────────

    /// `StageIndexer::new` produces an empty anticipated-decision block and
    /// empty per-plant metadata vectors.
    #[test]
    fn anticipated_decision_empty_for_new() {
        let idx = StageIndexer::new(3, 2);
        assert_eq!(idx.anticipated_decision, 0..0);
        assert!(idx.anticipated_lead_stages.is_empty());
        assert!(idx.anticipated_thermal_indices.is_empty());
    }

    /// Direct `EquipmentCounts` literal mirroring the AC-2 example:
    /// `n_anticipated=2`, `k_max=3`, `K_i = [3, 2]`, `thermal_idx = [0, 2]`.
    #[test]
    fn anticipated_decision_layout_basic() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 2,
                max_par_order: 1,
                n_thermals: 3,
                n_lines: 2,
                n_buses: 4,
                n_blks: 2,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 2,
                k_max: 3,
                anticipated_lead_stages: vec![3, 2],
                anticipated_thermal_indices: vec![0, 2],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        // anticipated_decision sits between thermal and anticipated_state_out, length 2.
        assert_eq!(idx.anticipated_decision.start, idx.thermal.end);
        assert_eq!(idx.anticipated_decision.len(), 2);
        // anticipated_state_out sits immediately after anticipated_decision, length 2.
        assert_eq!(
            idx.anticipated_state_out.start,
            idx.anticipated_decision.end
        );
        assert_eq!(idx.anticipated_state_out.len(), 2);
        // line_fwd starts after anticipated_state_out.
        assert_eq!(idx.line_fwd.start, idx.anticipated_state_out.end);
        // Per-plant metadata round-trips intact.
        assert_eq!(idx.anticipated_lead_stages, vec![3, 2]);
        assert_eq!(idx.anticipated_thermal_indices, vec![0, 2]);
    }

    /// When `n_anticipated == 0` the layout produced by
    /// `with_equipment_and_evaporation` must be byte-identical to a build with
    /// the legacy non-anticipated layout: `line_fwd.start == thermal.end`
    /// and no shift downstream.
    #[test]
    fn anticipated_decision_no_anticipated_matches_existing() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(2, 1, 3, 2, 4, 2, false),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        assert_eq!(idx.anticipated_decision, 0..0);
        assert_eq!(idx.line_fwd.start, idx.thermal.end);
        // Concrete offsets for the no-anticipated layout (decision_start = theta+1).
        // N=2, L=1, T=3, Ln=2, B=4, K=2, no penalty:
        // theta = 2*(3+1) = 8; decision_start = 9
        // turbine: 9..13 (2*2), spillage: 13..17, diversion: 17..21,
        // thermal: 21..27 (3*2), line_fwd: 27..31 (2*2), line_rev: 31..35,
        // deficit: 35..43 (4*1*2), excess: 43..51 (4*2)
        assert_eq!(idx.theta, 8);
        assert_eq!(idx.turbine, 9..13);
        assert_eq!(idx.thermal, 21..27);
        assert_eq!(idx.line_fwd, 27..31);
        assert_eq!(idx.line_rev, 31..35);
        assert_eq!(idx.deficit, 35..43);
        assert_eq!(idx.excess, 43..51);
    }

    /// Strict-predicate acceptance: `stage_idx + K_i < n_stages` accepts.
    /// `K_i = 3`, `stage_idx = 2`, `n_stages = 6` → `delivery_stage` = 5 < 6 → active.
    #[test]
    fn anticipated_decision_active_acceptance_strict_interior() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 1,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 1,
                k_max: 3,
                anticipated_lead_stages: vec![3],
                anticipated_thermal_indices: vec![0],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let active: Vec<_> = idx.anticipated_decision_active_at_stage(2, 6).collect();
        assert_eq!(active, vec![(0, idx.anticipated_decision.start)]);
    }

    /// F2-002 strict-predicate boundary rejection: `stage_idx + K_i == n_stages`
    /// is REJECTED. `K_i = 3`, `stage_idx = 3`, `n_stages = 6` → `delivery_stage` = 6
    /// would fall outside the study horizon `[0, 6)`. The strict predicate
    /// `stage_idx + K_i < n_stages` excludes this case so the LP never builds
    /// a priced-but-not-delivered commitment.
    #[test]
    fn anticipated_decision_active_rejection_at_n_stages_boundary() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 1,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 1,
                k_max: 3,
                anticipated_lead_stages: vec![3],
                anticipated_thermal_indices: vec![0],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let active: Vec<_> = idx.anticipated_decision_active_at_stage(3, 6).collect();
        assert!(
            active.is_empty(),
            "stage_idx + K_i == n_stages must be excluded under the strict predicate"
        );
    }

    /// Strict-predicate one-past-boundary rejection: `stage_idx + K_i == n_stages + 1`
    /// also rejects (the inclusive-predicate "one-past-boundary" case).
    /// `K_i = 3`, `stage_idx = 4`, `n_stages = 6` → plant is NOT active.
    #[test]
    fn anticipated_decision_active_rejection_one_past_boundary() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 1,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 1,
                k_max: 3,
                anticipated_lead_stages: vec![3],
                anticipated_thermal_indices: vec![0],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let active: Vec<_> = idx.anticipated_decision_active_at_stage(4, 6).collect();
        assert!(active.is_empty());
    }

    /// At `stage_idx == 0`, every plant with `K_i < n_stages` is active under
    /// the strict predicate. Plants with `K_i == n_stages` are excluded
    /// (delivery would land at `n_stages`, outside the study horizon).
    #[test]
    fn anticipated_decision_active_all_at_stage_zero() {
        // K = [1, 3, 6], n_stages = 6 → plants 0 (K=1) and 1 (K=3) accept
        // (0+1=1 < 6, 0+3=3 < 6). Plant 2 (K=6) rejects (0+6=6 NOT < 6).
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 6,
                anticipated_lead_stages: vec![1, 3, 6],
                anticipated_thermal_indices: vec![0, 1, 2],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_decision.start;
        let active: Vec<_> = idx.anticipated_decision_active_at_stage(0, 6).collect();
        assert_eq!(active, vec![(0, start), (1, start + 1)]);
    }

    /// When `anticipated_lead_stages` is empty, the iterator yields nothing
    /// for any `(stage_idx, n_stages)` pair.
    #[test]
    fn anticipated_decision_active_empty_iterator_when_no_plants() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(1, 0, 0, 0, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        for (s, t) in [(0_usize, 0_usize), (0, 12), (5, 12), (12, 12), (20, 12)] {
            let active: Vec<_> = idx.anticipated_decision_active_at_stage(s, t).collect();
            assert!(
                active.is_empty(),
                "expected empty iterator at (stage_idx={s}, n_stages={t})"
            );
        }
    }

    /// AC example `K_i = [3, 2, 5]` with `n_stages = 6`: verify each
    /// `stage_idx` selects the expected subset under the strict predicate
    /// `stage_idx + K_i < n_stages`.
    #[test]
    fn anticipated_decision_active_mixed_k_values() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 5,
                anticipated_lead_stages: vec![3, 2, 5],
                anticipated_thermal_indices: vec![0, 1, 2],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_decision.start;

        // stage_idx = 3: plant 0 rejects (3+3=6 NOT < 6), plant 1 accepts
        // (3+2=5 < 6), plant 2 rejects (3+5=8 NOT < 6).
        let active3: Vec<_> = idx.anticipated_decision_active_at_stage(3, 6).collect();
        assert_eq!(active3, vec![(1, start + 1)]);

        // stage_idx = 1: plant 0 accepts (1+3=4 < 6), plant 1 accepts
        // (1+2=3 < 6), plant 2 rejects (1+5=6 NOT < 6).
        let active1: Vec<_> = idx.anticipated_decision_active_at_stage(1, 6).collect();
        assert_eq!(
            active1,
            vec![(0, start), (1, start + 1)],
            "at stage_idx=1 under strict predicate: plants 0,1 active (1+3=4 < 6, 1+2=3 < 6); plant 2 excluded (1+5=6 NOT < 6)"
        );

        // stage_idx = 0: all three plants accept (0+3=3 < 6, 0+2=2 < 6,
        // 0+5=5 < 6).
        let active0: Vec<_> = idx.anticipated_decision_active_at_stage(0, 6).collect();
        assert_eq!(
            active0,
            vec![(0, start), (1, start + 1), (2, start + 2)],
            "at stage_idx=0 under strict predicate: all plants active (0+K < 6 for K in {{3,2,5}})"
        );

        // stage_idx = 4: all plants reject under strict predicate
        // (4+3=7, 4+2=6, 4+5=9; none < 6).
        let active4: Vec<_> = idx.anticipated_decision_active_at_stage(4, 6).collect();
        assert!(
            active4.is_empty(),
            "at stage_idx=4 under strict predicate, no plant satisfies stage_idx + K < 6"
        );
    }

    /// At horizon tail `stage_idx == n_stages` and every plant has `K_i > 0`,
    /// no plant should be active (`stage_idx + K_i > n_stages`).
    #[test]
    fn anticipated_decision_active_no_plants_at_horizon_tail() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 5,
                anticipated_lead_stages: vec![1, 3, 5],
                anticipated_thermal_indices: vec![0, 1, 2],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let active: Vec<_> = idx.anticipated_decision_active_at_stage(6, 6).collect();
        assert!(active.is_empty());
    }

    /// `line_fwd.start == anticipated_state_out.end` for various
    /// `n_anticipated`. Sweep includes 0 (no shift), 1, 2, 5.
    ///
    /// `anticipated_state_out` is inserted between `anticipated_decision`
    /// and `line_fwd`, so the layout invariant is:
    ///   `anticipated_decision.end` == `anticipated_state_out.start`
    ///   `anticipated_state_out.end` == `line_fwd.start`
    #[test]
    fn anticipated_decision_shifts_line_fwd() {
        for n_ant in [0_usize, 1, 2, 5] {
            let lead = if n_ant == 0 { vec![] } else { vec![2; n_ant] };
            let thermal_idx = (0..n_ant).collect::<Vec<_>>();
            let k_max = if n_ant == 0 { 0 } else { 2 };
            let idx = StageIndexer::with_equipment_and_evaporation(
                &EquipmentCounts {
                    hydro_count: 1,
                    max_par_order: 0,
                    n_thermals: n_ant.max(1),
                    n_lines: 1,
                    n_buses: 1,
                    n_blks: 1,
                    has_inflow_penalty: false,
                    max_deficit_segments: 1,
                    n_anticipated: n_ant,
                    k_max,
                    anticipated_lead_stages: lead,
                    anticipated_thermal_indices: thermal_idx,
                },
                &fpha(vec![], vec![]),
                &evap(vec![]),
            );
            if n_ant > 0 {
                // When the block is non-empty:
                //   anticipated_decision.end == anticipated_state_out.start
                //   anticipated_state_out.end == line_fwd.start
                assert_eq!(
                    idx.anticipated_state_out.start, idx.anticipated_decision.end,
                    "anticipated_state_out.start must equal anticipated_decision.end for n_ant={n_ant}"
                );
                assert_eq!(
                    idx.line_fwd.start, idx.anticipated_state_out.end,
                    "line_fwd.start must equal anticipated_state_out.end for n_ant={n_ant}"
                );
                assert_eq!(
                    idx.anticipated_state_out.len(),
                    n_ant,
                    "anticipated_state_out.len() must equal n_anticipated for n_ant={n_ant}"
                );
            } else {
                // When both blocks collapse to `0..0`, `line_fwd.start`
                // falls directly on `thermal.end` (no shift).
                assert_eq!(
                    idx.anticipated_state_out,
                    0..0,
                    "anticipated_state_out must be 0..0 when n_ant=0"
                );
                assert_eq!(
                    idx.line_fwd.start, idx.thermal.end,
                    "line_fwd.start must equal thermal.end when n_ant=0"
                );
            }
            assert_eq!(
                idx.anticipated_decision.len(),
                n_ant,
                "anticipated_decision.len() must equal n_anticipated for n_ant={n_ant}"
            );
        }
    }

    /// `anticipated_decision.start == thermal.end` whenever `n_anticipated > 0`.
    #[test]
    fn anticipated_decision_contiguous_with_thermal() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 2,
                n_lines: 1,
                n_buses: 1,
                n_blks: 2,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 2,
                k_max: 3,
                anticipated_lead_stages: vec![3, 2],
                anticipated_thermal_indices: vec![0, 1],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert_eq!(idx.anticipated_decision.start, idx.thermal.end);
    }

    /// Per-plant `anticipated_thermal_indices` round-trips through the
    /// constructor exactly.
    #[test]
    fn anticipated_decision_thermal_indices_preserved() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 6,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 4,
                anticipated_lead_stages: vec![4, 2, 1],
                anticipated_thermal_indices: vec![0, 2, 5],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert_eq!(idx.anticipated_thermal_indices, vec![0, 2, 5]);
        assert_eq!(idx.anticipated_lead_stages, vec![4, 2, 1]);
    }

    // ── Anticipated fishing row tests ──────────────────────────────────────

    /// `StageIndexer::new` produces an empty fishing range and a zero start.
    #[test]
    fn anticipated_fishing_empty_for_new() {
        let idx = StageIndexer::new(3, 2);
        assert_eq!(idx.anticipated_fishing, 0..0);
        assert_eq!(idx.anticipated_fishing_start, 0);
    }

    /// When operational violations are active, `anticipated_fishing_start`
    /// equals `min_generation_rows.end`. AC-2 worked example: 4 operational
    /// row families of N*K = 2 rows each = 8 rows total beginning at `A`,
    /// so `anticipated_fishing_start == A + 8 == min_generation_rows.end`.
    #[test]
    fn anticipated_fishing_start_after_min_generation_rows() {
        // N=2 hydros, K=1 block, no FPHA, no evap, no penalty.
        // n_anticipated=2, k_max=3 — shifts state by 2*3 = 6.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 2,
                max_par_order: 0,
                n_thermals: 2,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 2,
                k_max: 3,
                anticipated_lead_stages: vec![3, 2],
                anticipated_thermal_indices: vec![0, 1],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Operational violations must be active (hydro_count > 0).
        assert!(idx.has_operational_violations);
        // The start offset is precisely the end of `min_generation_rows`.
        assert_eq!(idx.anticipated_fishing_start, idx.min_generation_rows.end);
        // Stage-0 canonical layout: zero-length range anchored at the start.
        assert_eq!(
            idx.anticipated_fishing,
            idx.anticipated_fishing_start..idx.anticipated_fishing_start,
        );
    }

    /// When operational violations are inactive (`hydro_count == 0`),
    /// `anticipated_fishing_start` equals `evap_rows_end`
    /// (= `fpha_row_cursor + n_evap_hydros`).
    #[test]
    fn anticipated_fishing_start_after_evap_when_no_op_violations() {
        // hydro_count = 0 → no operational violations, no withdrawal slack,
        // no FPHA, no evap. Row cursor sits at `load_balance.end` because
        // `fpha_row_cursor == load_balance.end` and `n_evap_hydros == 0`.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 0,
                max_par_order: 0,
                n_thermals: 1,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert!(!idx.has_operational_violations);
        // With no FPHA and no evap, `evap_rows_end == load_balance.end`.
        let evap_rows_end = idx.load_balance.end;
        assert_eq!(idx.anticipated_fishing_start, evap_rows_end);
        assert_eq!(
            idx.anticipated_fishing,
            idx.anticipated_fishing_start..idx.anticipated_fishing_start,
        );
    }

    /// At `stage_idx == 0` all anticipated plants are active (always-active
    /// predicate): the iterator yields one entry per plant.
    #[test]
    fn anticipated_fishing_active_at_stage_zero() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 3,
                anticipated_lead_stages: vec![1, 2, 3],
                anticipated_thermal_indices: vec![0, 1, 2],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_fishing_start;
        let active: Vec<_> = idx.anticipated_fishing_active_at_stage(0, 5).collect();
        assert_eq!(active, vec![(0, start), (1, start + 1), (2, start + 2)]);
    }

    /// Always-active: `stage_idx == 0` with `K = [1, 2, 3]` yields the full
    /// set `[(0, start), (1, start+1), (2, start+2)]`.
    #[test]
    fn anticipated_fishing_active_always_active_stage_zero() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 3,
                anticipated_lead_stages: vec![1, 2, 3],
                anticipated_thermal_indices: vec![0, 1, 2],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_fishing_start;
        let active: Vec<_> = idx.anticipated_fishing_active_at_stage(0, 5).collect();
        assert_eq!(active, vec![(0, start), (1, start + 1), (2, start + 2)]);
    }

    /// Always-active: at any `stage_idx`, all plants are returned regardless of
    /// their `K_i` value.
    #[test]
    fn anticipated_fishing_active_acceptance_boundary() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 3,
                anticipated_lead_stages: vec![1, 2, 3],
                anticipated_thermal_indices: vec![0, 1, 2],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_fishing_start;
        let active: Vec<_> = idx.anticipated_fishing_active_at_stage(1, 5).collect();
        // All three plants are active (always-active predicate).
        assert_eq!(active, vec![(0, start), (1, start + 1), (2, start + 2)]);
    }

    /// Always-active: `K_i = 5`, `stage_idx = 4` — plant is still active since
    /// the always-active predicate ignores `K_i`.
    #[test]
    fn anticipated_fishing_active_rejection_boundary() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 1,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 1,
                k_max: 5,
                anticipated_lead_stages: vec![5],
                anticipated_thermal_indices: vec![0],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_fishing_start;
        let active: Vec<_> = idx.anticipated_fishing_active_at_stage(4, 6).collect();
        // Always-active predicate: plant is returned regardless of K_i vs stage_idx.
        assert_eq!(active, vec![(0, start)]);
    }

    /// All plants active at `stage_idx == 3` when `K_i = [1, 2, 3]`. Rows are
    /// assigned ascending: `(0, start+0), (1, start+1), (2, start+2)`.
    #[test]
    fn anticipated_fishing_active_all_plants() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 3,
                anticipated_lead_stages: vec![1, 2, 3],
                anticipated_thermal_indices: vec![0, 1, 2],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_fishing_start;
        let active: Vec<_> = idx.anticipated_fishing_active_at_stage(3, 5).collect();
        assert_eq!(active, vec![(0, start), (1, start + 1), (2, start + 2)]);
    }

    /// When `n_anticipated == 0`, the iterator yields nothing for any
    /// `(stage_idx, n_stages)` pair.
    #[test]
    fn anticipated_fishing_active_no_plants() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(1, 0, 0, 0, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        for (s, t) in [(0_usize, 0_usize), (0, 5), (3, 5), (5, 5)] {
            let active: Vec<_> = idx.anticipated_fishing_active_at_stage(s, t).collect();
            assert!(
                active.is_empty(),
                "expected empty iterator at (stage_idx={s}, n_stages={t})"
            );
        }
    }

    /// Ascending `local_idx` order is preserved even when ties cause the
    /// active subset to be the full set: `K_i = [2, 2, 2]`, `stage_idx = 2`
    /// yields `(0, start+0), (1, start+1), (2, start+2)`.
    #[test]
    fn anticipated_fishing_active_preserves_local_idx_order() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 2,
                anticipated_lead_stages: vec![2, 2, 2],
                anticipated_thermal_indices: vec![0, 1, 2],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_fishing_start;
        let active: Vec<_> = idx.anticipated_fishing_active_at_stage(2, 4).collect();
        assert_eq!(active, vec![(0, start), (1, start + 1), (2, start + 2)]);
    }

    /// Successive yields have strictly increasing `lp_row` values. Under
    /// always-active, `local_idx == active_pos` so row index equals
    /// `start + local_idx` for every plant.
    #[test]
    fn anticipated_fishing_active_row_indices_monotonic() {
        // K_i = [1, 4, 2, 3] with stage_idx = 3 → all 4 plants active.
        // Row indices: start+0, start+1, start+2, start+3 (contiguous).
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 4,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 4,
                k_max: 4,
                anticipated_lead_stages: vec![1, 4, 2, 3],
                anticipated_thermal_indices: vec![0, 1, 2, 3],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_fishing_start;
        let active: Vec<_> = idx.anticipated_fishing_active_at_stage(3, 5).collect();
        assert_eq!(
            active,
            vec![(0, start), (1, start + 1), (2, start + 2), (3, start + 3)]
        );
        // Row indices are strictly monotonic.
        let rows: Vec<_> = active.iter().map(|(_, row)| *row).collect();
        assert!(rows.windows(2).all(|w| w[0] < w[1]));
    }

    /// `stage_idx == n_stages` is an acceptance boundary (delivery happens at
    /// the horizon end and the LP at stage `T` still solves).
    #[test]
    fn anticipated_fishing_active_at_n_stages_boundary() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 2,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 2,
                k_max: 5,
                anticipated_lead_stages: vec![3, 5],
                anticipated_thermal_indices: vec![0, 1],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_fishing_start;
        let active: Vec<_> = idx.anticipated_fishing_active_at_stage(5, 5).collect();
        // Both plants active (always-active predicate; `stage_idx == n_stages` accepted by debug guard).
        assert_eq!(active, vec![(0, start), (1, start + 1)]);
    }

    /// `is_anticipated_fishing_active` returns `true` for every valid
    /// `(local_idx, stage_idx, n_stages)` triple across a parameter sweep.
    ///
    /// Fixtures: `n_anticipated ∈ {0, 1, 3}`, `anticipated_lead_stages`
    /// covering `k_max ∈ {0, 1, 5}`, `n_stages = 5`, `stage_idx ∈ 0..5`,
    /// `local_idx ∈ 0..n_anticipated`.
    #[test]
    fn is_anticipated_fishing_active_returns_true_everywhere() {
        // Fixture 1: n_anticipated = 0 (no plants → no iterations, vacuously ok)
        let idx0 = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 0,
                max_par_order: 0,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let n_stages = 5;
        // n_anticipated=0: no plants, so the predicate is vacuously active
        // for all (local_idx, stage_idx) pairs — nothing to iterate over.
        let _ = (&idx0, n_stages);

        // Fixture 2: n_anticipated = 1, k_max = 1 (k_max >= 1 required when n_anticipated > 0)
        let idx1 = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 0,
                max_par_order: 0,
                n_thermals: 1,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 1,
                k_max: 1,
                anticipated_lead_stages: vec![1],
                anticipated_thermal_indices: vec![0],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        for stage_idx in 0..n_stages {
            assert!(
                idx1.is_anticipated_fishing_active(0, stage_idx, n_stages),
                "fixture n_anticipated=1 k_max=1: expected true at (0, {stage_idx})"
            );
        }

        // Fixture 3: n_anticipated = 3, k_max = 5 (covers k_max = 5)
        let idx3 = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 0,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 5,
                anticipated_lead_stages: vec![1, 3, 5],
                anticipated_thermal_indices: vec![0, 1, 2],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        for stage_idx in 0..n_stages {
            for local_idx in 0..3_usize {
                assert!(
                    idx3.is_anticipated_fishing_active(local_idx, stage_idx, n_stages),
                    "fixture n_anticipated=3: expected true at ({local_idx}, {stage_idx})"
                );
            }
        }
    }

    /// Lock-step parity: `is_anticipated_fishing_active(p, s, n_stages)` matches
    /// membership of `p` in the iterator `anticipated_fishing_active_at_stage(s, n_stages)`
    /// for every `(p, s)` cell in `[0, n_anticipated) × [0, n_stages)`.
    ///
    /// Fixture: `n_anticipated = 3`, `K = [1, 2, 3]`, `n_stages = 5`.
    #[test]
    fn fishing_predicate_lockstep_with_iterator() {
        use std::collections::HashSet;

        let n_stages: usize = 5;
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 0,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 3,
                anticipated_lead_stages: vec![1, 2, 3],
                anticipated_thermal_indices: vec![0, 1, 2],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        for stage_idx in 0..n_stages {
            let iter_set: HashSet<usize> = idx
                .anticipated_fishing_active_at_stage(stage_idx, n_stages)
                .map(|(local, _)| local)
                .collect();
            for local_idx in 0..3_usize {
                let predicate_result =
                    idx.is_anticipated_fishing_active(local_idx, stage_idx, n_stages);
                let iter_result = iter_set.contains(&local_idx);
                assert_eq!(
                    predicate_result, iter_result,
                    "lock-step parity failed at (local={local_idx}, stage={stage_idx}): \
                     is_anticipated_fishing_active={predicate_result}, \
                     iterator membership={iter_result}"
                );
            }
        }
    }

    // ── state_to_lp_column tests ──────────────────────────────────────────────

    /// R4.a: `anticipated_state` indices use the shift-aware mapping when
    /// `max_par_order == 0 && n_anticipated > 0`. The `anticipated_state` branch
    /// runs before the `max_par_order == 0` lag-block guard; verify the shift
    /// semantics apply even when there are no inflow lags.
    ///
    /// Assertions reflect the shift-aware mapping (the anticipated-state
    /// branch in `state_to_lp_column` runs before the `max_par_order==0`
    /// early-return).
    #[test]
    fn state_to_lp_column_anticipated_identity_no_lag() {
        // N=1, L=0, n_anticipated=1, k_max=2, anticipated_lead_stages=[2].
        // n_state = 1*(1+0) + 1*2 = 3.
        // anticipated_state = [1, 3); slot 0 at j=1, slot 1 at j=2.
        // anticipated_state.start = N*(1+L) = 1.
        // anticipated_decision.start = theta+1 = 6 (no hydro/thermal cols).
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(1, 0, 0, 0, 1, 1, false, 1, 2),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Storage index: identity.
        assert_eq!(idx.state_to_lp_column(0), 0);
        // Anticipated-state slot 0 (j=1): slot+1=1 < k_p=2 → shift to slot 1.
        // Returns anticipated_state.start + 1*n_anticipated + 0 = 1+1 = 2.
        assert_eq!(idx.state_to_lp_column(1), 2);
        // Anticipated-state slot 1 (j=2): slot+1=2 == k_p=2 → state-out channel.
        // Returns anticipated_state_out.start + 0.
        assert_eq!(idx.state_to_lp_column(2), idx.anticipated_state_out.start);
    }

    /// R4.b: `anticipated_state` indices use the shift-aware mapping when
    /// `max_par_order > 0 && n_anticipated > 0`.  The fixture uses N=1, L=1,
    /// `n_anticipated=1`, `K_max=2`, `anticipated_lead_stages=[2]`.
    ///
    /// Assertions reflect the shift-aware mapping (identity is wrong for
    /// `K_max >= 2` because the decision-write slot `K_p - 1` and the shifted
    /// slots map to different LP columns than the incoming state's own).
    #[test]
    fn state_to_lp_column_anticipated_identity_with_lag() {
        // N=1, L=1, n_anticipated=1, k_max=2, anticipated_lead_stages=[2].
        // n_state = 1*(1+1) + 1*2 = 4.
        // Layout: j=0 storage, j=1 lag-0, j=2 ant slot-0, j=3 ant slot-1.
        // anticipated_state.start = N*(1+L) = 2.
        // z_inflow.start = anticipated_state_end = 2 + 2 = 4.
        // anticipated_decision.start = theta + 1 = 7 (no hydro/thermal columns).
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(1, 1, 0, 0, 1, 1, false, 1, 2),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Storage: identity.
        assert_eq!(idx.state_to_lp_column(0), 0);
        // Lag block: remapped (existing behaviour preserved).
        // j=1: offset=0, h=0, lag=0 → z_inflow.start + 0 = 4.
        assert_eq!(idx.state_to_lp_column(1), idx.z_inflow.start);
        // Anticipated-state slot 0 (j=2): slot+1=1 < k_p=2 → shift to slot 1.
        // Returns anticipated_state.start + 1*n_anticipated + 0 = 2+1 = 3.
        assert_eq!(idx.state_to_lp_column(2), 3);
        // Anticipated-state slot 1 (j=3): slot+1=2 == k_p=2 → state-out channel.
        // Returns anticipated_state_out.start + 0.
        assert_eq!(idx.state_to_lp_column(3), idx.anticipated_state_out.start);
    }

    /// R4.c: lag-remap branch is preserved when `n_anticipated == 0` and
    /// `max_par_order > 0`.  The new anticipated-state guard must not fire
    /// when there are no anticipated thermals.
    #[test]
    fn state_to_lp_column_lag_remap_preserved_no_anticipated() {
        // N=1, L=1, n_anticipated=0 — classic PAR(p) case.
        // n_state = 1*(1+1) = 2. Layout: j=0 storage, j=1 lag-0.
        let idx = StageIndexer::new(1, 1);
        assert_eq!(idx.n_anticipated, 0);
        // Storage: identity.
        assert_eq!(idx.state_to_lp_column(0), 0);
        // Lag block j=1: offset=0, h=0, lag=0 → z_inflow.start + 0.
        // For StageIndexer::new(1,1): z_inflow = N*(1+L)..N*(2+L) = 2..3.
        assert_eq!(idx.z_inflow.start, 2);
        assert_eq!(idx.state_to_lp_column(1), 2);
    }

    /// State-out channel branch: slot `K_p - 1` of an anticipated plant maps to
    /// the `anticipated_state_out` column for that plant (not `anticipated_decision`
    /// directly). The `anticipated_state_out` variable is pinned to the decision
    /// column by the `anticipated_state_out_def` equality row, so cut coefficients
    /// on the state-out column correctly express the Benders subgradient.
    #[test]
    fn state_to_lp_column_anticipated_decision_channel() {
        // N=0, L=0, n_anticipated=1, k_max=2, anticipated_lead_stages=[2].
        // n_state = 0*(1+0) + 1*2 = 2.
        // anticipated_state = [0, 2); slot 0 at j=0, slot 1 at j=1.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(0, 0, 0, 0, 1, 1, false, 1, 2),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Slot K_p - 1 = 1 (the highest slot for plant 0) → anticipated_state_out column.
        let slot_k_minus_1 = idx.anticipated_state.start + (idx.k_max - 1) * idx.n_anticipated;
        assert_eq!(
            idx.state_to_lp_column(slot_k_minus_1),
            idx.anticipated_state_out.start,
        );
    }

    /// Shift branch: an `anticipated_state` slot `i < K_p - 1` maps to the
    /// predecessor stage's `anticipated_state` column at slot `i + 1` (the
    /// shift). Successor's slot `i` comes from predecessor's incoming slot
    /// `i + 1` after `shift_anticipated_state` runs.
    #[test]
    fn state_to_lp_column_anticipated_shift() {
        // Same fixture as R5.a: single plant, K=2.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(0, 0, 0, 0, 1, 1, false, 1, 2),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Slot 0 → shift to slot 1's column.
        let slot_0 = idx.anticipated_state.start;
        let slot_1 = idx.anticipated_state.start + idx.n_anticipated;
        assert_eq!(idx.state_to_lp_column(slot_0), slot_1);
    }

    /// Padding branch: an `anticipated_state` slot `i > K_p - 1` (padding
    /// for a plant with `K_p < K_max`) maps to identity `j`. Padding slots
    /// are pinned to 0 by the state-fixing row, so the identity mapping is a
    /// safe default that does not introduce wrong cuts.
    #[test]
    fn state_to_lp_column_anticipated_padding_slot_identity() {
        // Two plants: plant 0 has K_p=1 (only slot 0 is in-use), plant 1 has
        // K_p=3 (slots 0, 1, 2 all in-use). k_max=3 so plant 0 has padding
        // at slots 1 and 2.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 0,
                max_par_order: 0,
                n_thermals: 2,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 2,
                k_max: 3,
                anticipated_lead_stages: vec![1, 3],
                anticipated_thermal_indices: vec![0, 1],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Plant 0 padding: slot 1 at j = ant_start + 1*2 + 0, slot 2 at j = ant_start + 2*2 + 0.
        let pad_slot_1_plant_0 = idx.anticipated_state.start + idx.n_anticipated;
        let pad_slot_2_plant_0 = idx.anticipated_state.start + 2 * idx.n_anticipated;
        assert_eq!(
            idx.state_to_lp_column(pad_slot_1_plant_0),
            pad_slot_1_plant_0
        );
        assert_eq!(
            idx.state_to_lp_column(pad_slot_2_plant_0),
            pad_slot_2_plant_0
        );
    }

    /// Multi-plant layout: correct routing for all `(slot, plant)`
    /// combinations in a two-plant K=2 fixture.
    #[test]
    fn state_to_lp_column_anticipated_multi_plant_layout() {
        // Two plants, both with K_p=2; k_max=2.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 0,
                max_par_order: 0,
                n_thermals: 2,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 2,
                k_max: 2,
                anticipated_lead_stages: vec![2, 2],
                anticipated_thermal_indices: vec![0, 1],
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Layout: slot * n_anticipated + plant. n_anticipated=2.
        //   j=ant_start+0 → slot 0, plant 0; shift → ant_start + 1*2 + 0 = ant_start + 2
        //   j=ant_start+1 → slot 0, plant 1; shift → ant_start + 1*2 + 1 = ant_start + 3
        //   j=ant_start+2 → slot 1, plant 0; state-out → anticipated_state_out.start + 0
        //   j=ant_start+3 → slot 1, plant 1; state-out → anticipated_state_out.start + 1
        let s = idx.anticipated_state.start;
        let so = idx.anticipated_state_out.start;
        assert_eq!(idx.state_to_lp_column(s), s + 2);
        assert_eq!(idx.state_to_lp_column(s + 1), s + 3);
        assert_eq!(idx.state_to_lp_column(s + 2), so);
        assert_eq!(idx.state_to_lp_column(s + 3), so + 1);
    }

    /// `anticipated_state_out` range is placed immediately after
    /// `anticipated_decision` and `line_fwd` follows `anticipated_state_out`.
    #[test]
    fn test_anticipated_state_out_range_is_after_anticipated_decision() {
        let counts = EquipmentCounts {
            hydro_count: 2,
            max_par_order: 1,
            n_thermals: 3,
            n_lines: 1,
            n_buses: 2,
            n_blks: 1,
            has_inflow_penalty: false,
            max_deficit_segments: 1,
            n_anticipated: 2,
            k_max: 3,
            anticipated_lead_stages: vec![2, 3],
            anticipated_thermal_indices: vec![0, 1],
        };
        let idx = StageIndexer::with_equipment(&counts, &fpha(vec![], vec![]));

        // Range adjacency.
        assert_eq!(
            idx.anticipated_state_out.start,
            idx.anticipated_decision.end
        );
        assert_eq!(
            idx.anticipated_state_out.end - idx.anticipated_state_out.start,
            2
        );

        // line_fwd starts immediately after the new column block.
        assert_eq!(idx.line_fwd.start, idx.anticipated_state_out.end);

        // n_state is unchanged: N*(1+L) + A*K_max = 2*2 + 2*3 = 10.
        assert_eq!(idx.n_state, 10);
    }

    /// `anticipated_state_out` is empty (`0..0`) when built via `StageIndexer::new`.
    #[test]
    fn test_anticipated_state_out_is_empty_when_no_anticipated() {
        let idx = StageIndexer::new(3, 2);
        assert_eq!(idx.anticipated_state_out, 0..0);
        assert_eq!(idx.n_state, 9);
    }

    // ── state_to_lp_incoming_column tests ────────────────────────────────────

    /// Storage range: for a `with_equipment` indexer with `N=3, L=2, A=0`,
    /// `state_to_lp_incoming_column(j)` for `j ∈ [0, N)` returns
    /// `storage_in.start + j`.
    #[test]
    fn state_to_lp_incoming_column_storage_range() {
        // N=3, L=2: storage_in.start = N*(2+L) = 3*4 = 12.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, 0, 0),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // storage_in.start should be N*(2+L) = 12.
        assert_eq!(idx.storage_in.start, 12);
        for j in 0..3_usize {
            assert_eq!(
                idx.state_to_lp_incoming_column(j),
                idx.storage_in.start + j,
                "j={j}: expected storage_in.start + {j}"
            );
        }
    }

    /// AR lag range: for a `with_equipment` indexer with `N=3, L=2, A=0`,
    /// `state_to_lp_incoming_column(j)` for `j ∈ [N, N*(1+L))` returns
    /// `inflow_lags.start + (j − N)`.
    #[test]
    fn state_to_lp_incoming_column_lag_range() {
        // N=3, L=2: inflow_lags = 3..9.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, 0, 0),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert_eq!(idx.inflow_lags.start, 3);
        for j in 3..9_usize {
            assert_eq!(
                idx.state_to_lp_incoming_column(j),
                idx.inflow_lags.start + (j - 3),
                "j={j}: expected inflow_lags.start + {}",
                j - 3
            );
        }
    }

    /// Anticipated-state range: for an indexer with `N=0, L=0, A=1, K=2`,
    /// `state_to_lp_incoming_column(j)` for `j ∈ [0, n_state)` returns
    /// `anticipated_state.start + j` (since `lag_end` = N*(1+L) = 0).
    #[test]
    fn state_to_lp_incoming_column_anticipated_range() {
        // N=0, L=0, A=1, K=2: n_state = 0 + 1*2 = 2.
        // anticipated_state.start = N*(1+L) = 0.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(0, 0, 0, 0, 1, 1, false, 1, 2),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert_eq!(idx.anticipated_state.start, 0);
        assert_eq!(idx.n_state, 2);
        for j in 0..2_usize {
            assert_eq!(
                idx.state_to_lp_incoming_column(j),
                idx.anticipated_state.start + j,
                "j={j}: expected anticipated_state.start + {j}"
            );
        }
    }

    /// Combined boundary-case test: `N=3, L=2, A=1, K=2`.
    /// Checks j = 0, 2, 3, 8, 9, 10 (the boundary points from the spec).
    #[test]
    fn state_to_lp_incoming_column_combined_with_equipment_indexer() {
        // N=3, L=2, A=1, K=2:
        //   n_state = N*(1+L) + A*K = 3*3 + 1*2 = 11.
        //   storage_in.start = N*(2+L) + A*K_max = 3*4 + 1*2 = 14.
        //   inflow_lags.start = N = 3.
        //   anticipated_state.start = N*(1+L) = 9.
        //   lag_end = N*(1+L) = 9.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(3, 2, 0, 0, 1, 1, false, 1, 2),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        assert_eq!(idx.n_state, 11);
        // j=0: storage range → storage_in.start + 0.
        assert_eq!(
            idx.state_to_lp_incoming_column(0),
            idx.storage_in.start,
            "j=0"
        );
        // j=2: storage range → storage_in.start + 2.
        assert_eq!(
            idx.state_to_lp_incoming_column(2),
            idx.storage_in.start + 2,
            "j=2"
        );
        // j=3: first lag → inflow_lags.start + 0.
        assert_eq!(
            idx.state_to_lp_incoming_column(3),
            idx.inflow_lags.start,
            "j=3"
        );
        // j=8: last lag → inflow_lags.start + 5.
        assert_eq!(
            idx.state_to_lp_incoming_column(8),
            idx.inflow_lags.start + 5,
            "j=8"
        );
        // j=9: first anticipated-state → anticipated_state.start + 0.
        assert_eq!(
            idx.state_to_lp_incoming_column(9),
            idx.anticipated_state.start,
            "j=9"
        );
        // j=10: last anticipated-state → anticipated_state.start + 1.
        assert_eq!(
            idx.state_to_lp_incoming_column(10),
            idx.anticipated_state.start + 1,
            "j=10"
        );
        // All returned columns must be within the LP's column range.
        for j in 0..idx.n_state {
            let col = idx.state_to_lp_incoming_column(j);
            assert!(
                col < idx.theta + 1,
                "j={j}: column {col} out of range (theta={})",
                idx.theta
            );
        }
    }

    /// For the lag range, `state_to_lp_incoming_column` and `state_to_lp_column`
    /// return different values under `with_equipment` (the former returns the
    /// incoming-lag column, the latter returns the `z_inflow` or lag-out column).
    /// For the storage range under `with_equipment`, the results differ because
    /// `storage_in.start > 0` while `state_to_lp_column` returns `j` (outgoing,
    /// which starts at column 0).
    #[test]
    fn state_to_lp_incoming_column_differs_from_state_to_lp_column_for_lag() {
        // N=2, L=1: storage_in.start = N*(2+L) = 2*3 = 6.
        // state_to_lp_column(0) = 0 (outgoing storage).
        // state_to_lp_incoming_column(0) = storage_in.start + 0 = 6.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq_with_anticipated(2, 1, 0, 0, 1, 1, false, 0, 0),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        // Storage range: incoming ≠ outgoing under with_equipment.
        assert_ne!(
            idx.state_to_lp_incoming_column(0),
            idx.state_to_lp_column(0),
            "storage range should differ: incoming={} outgoing={}",
            idx.state_to_lp_incoming_column(0),
            idx.state_to_lp_column(0)
        );
        // j=0: incoming returns storage_in.start, outgoing returns 0.
        assert_eq!(idx.state_to_lp_incoming_column(0), idx.storage_in.start);
        assert_eq!(idx.state_to_lp_column(0), 0);

        // Lag range (j=2, j=3): incoming returns inflow_lags column;
        // outgoing returns z_inflow (lag=0) or lag-out (lag>=1) column.
        // j=2: lag 0, hydro 0. incoming = inflow_lags.start + 0.
        //                       outgoing = z_inflow.start + 0.
        assert_eq!(
            idx.state_to_lp_incoming_column(2),
            idx.inflow_lags.start,
            "j=2 incoming should be inflow_lags.start"
        );
        assert_eq!(
            idx.state_to_lp_column(2),
            idx.z_inflow.start,
            "j=2 outgoing should be z_inflow.start"
        );
        assert_ne!(
            idx.state_to_lp_incoming_column(2),
            idx.state_to_lp_column(2),
            "lag range should differ for j=2"
        );
        // j=3: lag 0, hydro 1. incoming = inflow_lags.start + 1.
        //                       outgoing = z_inflow.start + 1.
        assert_ne!(
            idx.state_to_lp_incoming_column(3),
            idx.state_to_lp_column(3),
            "lag range should differ for j=3"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // state_to_lp_column branch tests
    // ─────────────────────────────────────────────────────────────────────────

    /// K=1, single plant: slot 0 hits the Equal branch and must route to
    /// `anticipated_state_out`, NOT to `anticipated_decision`.
    #[test]
    fn test_state_to_lp_column_equal_branch_routes_to_state_out() {
        let counts = EquipmentCounts {
            hydro_count: 1,
            max_par_order: 0,
            n_thermals: 1,
            n_lines: 0,
            n_buses: 1,
            n_blks: 1,
            has_inflow_penalty: false,
            max_deficit_segments: 1,
            n_anticipated: 1,
            k_max: 1,
            anticipated_lead_stages: vec![1],
            anticipated_thermal_indices: vec![0],
        };
        let fpha_layout = FphaColumnLayout {
            hydro_indices: vec![],
            planes_per_hydro: vec![],
        };
        let idx = StageIndexer::with_equipment(&counts, &fpha_layout);

        let j_slot0 = idx.anticipated_state.start; // slot 0, plant 0
        assert_eq!(
            idx.state_to_lp_column(j_slot0),
            idx.anticipated_state_out.start,
            "Equal branch must route to anticipated_state_out, not anticipated_decision"
        );
        assert_ne!(
            idx.state_to_lp_column(j_slot0),
            idx.anticipated_decision.start,
            "Equal branch must NOT route to anticipated_decision (the buggy mapping)"
        );
    }

    /// K=2, single plant: slot 0 hits the Less branch.
    /// Less branch must return `anticipated_state.start + (slot+1)*A + plant`.
    #[test]
    fn test_state_to_lp_column_less_branch_unchanged() {
        let counts = EquipmentCounts {
            hydro_count: 1,
            max_par_order: 0,
            n_thermals: 1,
            n_lines: 0,
            n_buses: 1,
            n_blks: 1,
            has_inflow_penalty: false,
            max_deficit_segments: 1,
            n_anticipated: 1,
            k_max: 2,
            anticipated_lead_stages: vec![2],
            anticipated_thermal_indices: vec![0],
        };
        let fpha_layout = FphaColumnLayout {
            hydro_indices: vec![],
            planes_per_hydro: vec![],
        };
        let idx = StageIndexer::with_equipment(&counts, &fpha_layout);

        let j_slot0 = idx.anticipated_state.start; // slot 0, plant 0
        assert_eq!(
            idx.state_to_lp_column(j_slot0),
            idx.anticipated_state.start + 1,
            "Less branch must return anticipated_state.start + (slot+1)*A + plant"
        );
    }

    /// K=2 plant in a `k_max=3` layout: slot 2 hits the Greater branch (padding).
    /// Greater branch must return identity (`j` unchanged).
    ///
    /// Uses two anticipated plants with `k_p`=[2,3] so that `k_max=max(k_p)=3`
    /// satisfies the indexer invariant, while plant 0 (`k_p=2`) has a genuine
    /// padding slot at slot index 2.
    #[test]
    fn test_state_to_lp_column_greater_branch_unchanged() {
        let counts = EquipmentCounts {
            hydro_count: 1,
            max_par_order: 0,
            n_thermals: 2,
            n_lines: 0,
            n_buses: 1,
            n_blks: 1,
            has_inflow_penalty: false,
            max_deficit_segments: 1,
            n_anticipated: 2,
            k_max: 3,
            anticipated_lead_stages: vec![2, 3],
            anticipated_thermal_indices: vec![0, 1],
        };
        let fpha_layout = FphaColumnLayout {
            hydro_indices: vec![],
            planes_per_hydro: vec![],
        };
        let idx = StageIndexer::with_equipment(&counts, &fpha_layout);

        // Slot 2, plant 0: slot+1=3 > k_p=2 → Greater branch (padding).
        // n_anticipated=2, so j = ant_start + 2*2 + 0.
        let j_slot2_plant0 = idx.anticipated_state.start + 2 * idx.n_anticipated;
        assert_eq!(
            idx.state_to_lp_column(j_slot2_plant0),
            j_slot2_plant0,
            "Greater branch must return identity (padding-slot invariant)"
        );
    }

    /// Structural-invariant sweep tests for the anticipated-thermal indexer
    /// extensions added by tickets 012-015.
    ///
    /// Each test in this sub-module iterates the same hand-written parameter
    /// grid (see [`parameter_grid`]) and asserts a single structural invariant
    /// (I1-I13) on every grid point. The grid is finite (~1900 configurations
    /// after pruning) and each iteration performs only range arithmetic, so
    /// the full sweep completes in well under 1 second on a standard laptop.
    ///
    /// No new dev-dependency is introduced: the sweep is hand-written, not
    /// generated by `proptest`/`quickcheck`. This matches the project-wide
    /// convention (no existing usage of property-test crates anywhere in
    /// the workspace).
    mod anticipated_invariants {
        use std::ops::Range;

        use super::{EquipmentCounts, EvapConfig, FphaColumnLayout, StageIndexer};

        /// Grid point describing one configuration to test.
        #[derive(Clone, Copy, Debug)]
        struct SweepParams {
            n_hyd: usize,
            l: usize,
            n_ant: usize,
            k_max: usize,
            n_t: usize,
            n_l: usize,
            n_b: usize,
            n_blk: usize,
            pen: bool,
        }

        /// Number of stages used for active-stage queries (I10, I11).
        ///
        /// Must be `>= k_max` so the `K_i <= T` invariant holds for every
        /// generated `lead_stages` scheme.
        const N_STAGES: usize = 8;

        /// Iterator over the full sweep grid, post-pruning.
        ///
        /// Pruning rules:
        /// - `n_ant == 0` requires `k_max == 0` (no plants → no ring buffer);
        /// - `n_ant > 0` requires `k_max >= 1` (debug-asserted by the
        ///   constructor).
        fn parameter_grid() -> impl Iterator<Item = SweepParams> {
            let ns = [0_usize, 1, 3, 5];
            let ls = [0_usize, 1, 2, 4];
            let n_ants = [0_usize, 1, 2, 4];
            let k_maxes = [0_usize, 1, 2, 3];
            let n_therms = [0_usize, 3];
            let n_lns = [0_usize, 2];
            let n_buses_a = [1_usize, 2];
            let n_blks_a = [1_usize, 2, 3];
            let penalties = [false, true];

            ns.into_iter()
                .flat_map(move |n_hyd| {
                    ls.into_iter().flat_map(move |l| {
                        n_ants.into_iter().flat_map(move |n_ant| {
                            k_maxes.into_iter().flat_map(move |k_max| {
                                n_therms.into_iter().flat_map(move |n_t| {
                                    n_lns.into_iter().flat_map(move |n_l| {
                                        n_buses_a.into_iter().flat_map(move |n_b| {
                                            n_blks_a.into_iter().flat_map(move |n_blk| {
                                                penalties.into_iter().map(move |pen| SweepParams {
                                                    n_hyd,
                                                    l,
                                                    n_ant,
                                                    k_max,
                                                    n_t,
                                                    n_l,
                                                    n_b,
                                                    n_blk,
                                                    pen,
                                                })
                                            })
                                        })
                                    })
                                })
                            })
                        })
                    })
                })
                .filter(|p| (p.n_ant > 0 && p.k_max >= 1) || (p.n_ant == 0 && p.k_max == 0))
        }

        /// Build a deterministic `lead_stages` vector for a given grid point.
        ///
        /// The scheme cycles `1..=k_max` so we exercise both `K_i < k_max`
        /// (padding) and `K_i == k_max` (no padding) within a single
        /// configuration. The maximum entry equals `k_max`, which the
        /// constructor debug-asserts.
        fn lead_stages_for(p: &SweepParams) -> Vec<usize> {
            if p.n_ant == 0 {
                return Vec::new();
            }
            let mut out = Vec::with_capacity(p.n_ant);
            // Plant 0 carries k_max to satisfy the `max == k_max` debug_assert.
            out.push(p.k_max);
            for i in 1..p.n_ant {
                out.push(1 + (i % p.k_max));
            }
            out
        }

        /// Build a `StageIndexer` for a grid point with empty FPHA/evap.
        fn build_indexer(p: &SweepParams) -> (StageIndexer, Vec<usize>) {
            let lead_stages = lead_stages_for(p);
            let thermal_indices: Vec<usize> = (0..p.n_ant).collect();
            let counts = EquipmentCounts {
                hydro_count: p.n_hyd,
                max_par_order: p.l,
                n_thermals: p.n_t,
                n_lines: p.n_l,
                n_buses: p.n_b,
                n_blks: p.n_blk,
                has_inflow_penalty: p.pen,
                max_deficit_segments: 1,
                n_anticipated: p.n_ant,
                k_max: p.k_max,
                anticipated_lead_stages: lead_stages.clone(),
                anticipated_thermal_indices: thermal_indices,
            };
            let idx = StageIndexer::with_equipment_and_evaporation(
                &counts,
                &FphaColumnLayout {
                    hydro_indices: vec![],
                    planes_per_hydro: vec![],
                },
                &EvapConfig {
                    hydro_indices: vec![],
                },
            );
            (idx, lead_stages)
        }

        /// Append `r` to `v` iff `r` is non-empty.
        fn push_nonempty(v: &mut Vec<Range<usize>>, r: Range<usize>) {
            if !r.is_empty() {
                v.push(r);
            }
        }

        /// Collect the non-empty column ranges in canonical layout order.
        ///
        /// `theta` is a single column index represented as the unit range
        /// `theta..theta + 1` so it slots into the same non-overlap check.
        fn collect_active_column_ranges(idx: &StageIndexer) -> Vec<Range<usize>> {
            let mut v = Vec::new();
            push_nonempty(&mut v, idx.storage.clone());
            push_nonempty(&mut v, idx.inflow_lags.clone());
            push_nonempty(&mut v, idx.anticipated_state.clone());
            push_nonempty(&mut v, idx.z_inflow.clone());
            push_nonempty(&mut v, idx.storage_in.clone());
            v.push(idx.theta..idx.theta + 1);
            push_nonempty(&mut v, idx.turbine.clone());
            push_nonempty(&mut v, idx.spillage.clone());
            push_nonempty(&mut v, idx.diversion.clone());
            push_nonempty(&mut v, idx.thermal.clone());
            push_nonempty(&mut v, idx.anticipated_decision.clone());
            push_nonempty(&mut v, idx.line_fwd.clone());
            push_nonempty(&mut v, idx.line_rev.clone());
            push_nonempty(&mut v, idx.deficit.clone());
            push_nonempty(&mut v, idx.excess.clone());
            push_nonempty(&mut v, idx.inflow_slack.clone());
            push_nonempty(&mut v, idx.generation.clone());
            push_nonempty(&mut v, idx.withdrawal_slack_neg.clone());
            push_nonempty(&mut v, idx.withdrawal_slack_pos.clone());
            push_nonempty(&mut v, idx.outflow_below_slack.clone());
            push_nonempty(&mut v, idx.outflow_above_slack.clone());
            push_nonempty(&mut v, idx.turbine_below_slack.clone());
            push_nonempty(&mut v, idx.generation_below_slack.clone());
            v
        }

        /// Collect the non-empty row ranges in canonical layout order.
        fn collect_active_row_ranges(idx: &StageIndexer) -> Vec<Range<usize>> {
            let mut v = Vec::new();
            push_nonempty(&mut v, idx.storage_fixing.clone());
            push_nonempty(&mut v, idx.lag_fixing.clone());
            push_nonempty(&mut v, idx.anticipated_state_fixing.clone());
            push_nonempty(&mut v, idx.z_inflow_rows.clone());
            push_nonempty(&mut v, idx.water_balance.clone());
            push_nonempty(&mut v, idx.load_balance.clone());
            push_nonempty(&mut v, idx.min_outflow_rows.clone());
            push_nonempty(&mut v, idx.max_outflow_rows.clone());
            push_nonempty(&mut v, idx.min_turbine_rows.clone());
            push_nonempty(&mut v, idx.min_generation_rows.clone());
            // `anticipated_fishing` is structurally `fishing_start..fishing_start`
            // at stage 0; it carries no rows but pins the start offset for I9.
            // Empty ranges are excluded from the contiguity check.
            v
        }

        /// I1: column ranges are pairwise non-overlapping and ascending.
        #[test]
        fn i1_column_ranges_non_overlapping_ascending() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                let ranges = collect_active_column_ranges(&idx);
                for win in ranges.windows(2) {
                    assert!(
                        win[0].end <= win[1].start,
                        "I1 failed at {p:?}: {:?} overlaps {:?}",
                        win[0],
                        win[1]
                    );
                }
            }
        }

        /// I2: row ranges are pairwise non-overlapping and ascending.
        #[test]
        fn i2_row_ranges_non_overlapping_ascending() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                let ranges = collect_active_row_ranges(&idx);
                for win in ranges.windows(2) {
                    assert!(
                        win[0].end <= win[1].start,
                        "I2 failed at {p:?}: {:?} overlaps {:?}",
                        win[0],
                        win[1]
                    );
                }
            }
        }

        /// I3: state-block dimension formula `n_state == N*(1+L) + n_ant*k_max`.
        #[test]
        fn i3_n_state_formula() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                let expected = p.n_hyd * (1 + p.l) + p.n_ant * p.k_max;
                assert_eq!(
                    idx.n_state, expected,
                    "I3 failed at {p:?}: n_state {} != expected {}",
                    idx.n_state, expected
                );
            }
        }

        /// I4 (Phase 1): all three state-fixing row ranges are empty sentinel `0..0`.
        ///
        /// State fixing has moved to column bounds; the row-side ranges are
        /// no longer expected to mirror the column ranges.
        #[test]
        fn i4_state_row_symmetry() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                assert_eq!(
                    idx.storage_fixing,
                    0..0,
                    "I4 storage_fixing must be 0..0 (sentinel) at {p:?}"
                );
                assert_eq!(
                    idx.lag_fixing,
                    0..0,
                    "I4 lag_fixing must be 0..0 (sentinel) at {p:?}"
                );
                assert_eq!(
                    idx.anticipated_state_fixing,
                    0..0,
                    "I4 anticipated_state_fixing must be 0..0 (sentinel) at {p:?}"
                );
            }
        }

        /// I5: theta placement `theta == storage_in.end == N*(3+L) + n_ant*k_max`.
        #[test]
        fn i5_theta_placement() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                let expected = p.n_hyd * (3 + p.l) + p.n_ant * p.k_max;
                assert_eq!(
                    idx.theta, idx.storage_in.end,
                    "I5 storage_in.end mismatch at {p:?}"
                );
                assert_eq!(
                    idx.theta, expected,
                    "I5 formula mismatch at {p:?}: theta {} != {}",
                    idx.theta, expected
                );
            }
        }

        /// I6: `anticipated_decision` and `anticipated_state_out` are contiguous
        /// between `thermal` and `line_fwd`.
        ///
        /// The layout is: `thermal → anticipated_decision → anticipated_state_out → line_fwd`.
        /// When both blocks collapse to `0..0` (no anticipated plants), the
        /// contiguity property reduces to `line_fwd.start == thermal.end`.
        /// Check that branch explicitly so the public `0..0` normalisation does
        /// not silently break the layout.
        #[test]
        fn i6_decision_contiguity() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                if p.n_ant > 0 {
                    assert_eq!(
                        idx.anticipated_decision.start, idx.thermal.end,
                        "I6 decision.start != thermal.end at {p:?}"
                    );
                    assert_eq!(
                        idx.anticipated_state_out.start, idx.anticipated_decision.end,
                        "I6 state_out.start != decision.end at {p:?}"
                    );
                    assert_eq!(
                        idx.line_fwd.start, idx.anticipated_state_out.end,
                        "I6 line_fwd.start != state_out.end at {p:?}"
                    );
                } else {
                    assert_eq!(
                        idx.anticipated_state_out,
                        0..0,
                        "I6 state_out must be 0..0 when n_ant=0 at {p:?}"
                    );
                    assert_eq!(
                        idx.line_fwd.start, idx.thermal.end,
                        "I6 zero-anticipated line_fwd.start != thermal.end at {p:?}"
                    );
                }
            }
        }

        /// I7: `anticipated_decision` column count equals `n_anticipated`.
        #[test]
        fn i7_decision_column_count() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                assert_eq!(
                    idx.anticipated_decision.len(),
                    p.n_ant,
                    "I7 failed at {p:?}: decision len {} != n_ant {}",
                    idx.anticipated_decision.len(),
                    p.n_ant
                );
            }
        }

        /// I8: ring-buffer state dimension `anticipated_state.len() == n_ant * k_max`.
        #[test]
        fn i8_anticipated_state_ring_buffer_dimension() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                let expected = p.n_ant * p.k_max;
                assert_eq!(
                    idx.anticipated_state.len(),
                    expected,
                    "I8 failed at {p:?}: anticipated_state len {} != {}",
                    idx.anticipated_state.len(),
                    expected
                );
            }
        }

        /// I9: `fishing_start` placement.
        ///
        /// When operational violations are active (`hydro_count > 0`), the
        /// fishing block sits right after `min_generation_rows`. When they are
        /// inactive (`hydro_count == 0`), and FPHA/evap are empty as built by
        /// this sweep, `fishing_start` collapses to `load_balance.end`.
        #[test]
        fn i9_fishing_start_placement() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                if p.n_hyd > 0 {
                    assert!(
                        idx.has_operational_violations,
                        "I9 expected operational violations active at {p:?}"
                    );
                    assert_eq!(
                        idx.anticipated_fishing_start, idx.min_generation_rows.end,
                        "I9 fishing_start != min_generation_rows.end at {p:?}"
                    );
                } else {
                    assert!(
                        !idx.has_operational_violations,
                        "I9 expected no operational violations at {p:?}"
                    );
                    assert_eq!(
                        idx.anticipated_fishing_start, idx.load_balance.end,
                        "I9 zero-hydro fishing_start != load_balance.end at {p:?}"
                    );
                }
                // Stage-0 fishing range is empty and pinned at fishing_start.
                assert_eq!(
                    idx.anticipated_fishing.start, idx.anticipated_fishing_start,
                    "I9 fishing.start != fishing_start at {p:?}"
                );
                assert!(
                    idx.anticipated_fishing.is_empty(),
                    "I9 fishing range non-empty at stage 0 for {p:?}"
                );
            }
        }

        /// I10: at stage 0, every anticipated plant is active.
        #[test]
        fn i10_decision_active_at_stage_zero() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                let count = idx
                    .anticipated_decision_active_at_stage(0, N_STAGES)
                    .count();
                assert_eq!(
                    count, p.n_ant,
                    "I10 failed at {p:?}: active decisions {} != n_ant {}",
                    count, p.n_ant
                );
            }
        }

        /// I11: at stage 0, all anticipated plants are active (always-active predicate).
        #[test]
        fn i11_fishing_active_at_stage_zero_is_empty() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                let count = idx.anticipated_fishing_active_at_stage(0, N_STAGES).count();
                assert_eq!(
                    count, p.n_ant,
                    "I11 failed at {p:?}: fishing active count {count} != n_ant {}",
                    p.n_ant
                );
            }
        }

        /// I12: nonzero state mask is sorted ascending with no duplicates.
        #[test]
        fn i12_mask_sorted_unique() {
            for p in parameter_grid() {
                let (mut idx, lead_stages) = build_indexer(&p);
                let lag_counts = vec![p.l; p.n_hyd];
                idx.set_nonzero_mask(&lag_counts, &lead_stages);
                assert!(
                    idx.nonzero_state_indices.windows(2).all(|w| w[0] < w[1]),
                    "I12 failed at {p:?}: mask {:?} is not strictly ascending",
                    idx.nonzero_state_indices
                );
            }
        }

        /// I13: mask length equals `N + sum(lag_counts) + sum(anticipated_lead_stages)`.
        #[test]
        fn i13_mask_length_formula() {
            for p in parameter_grid() {
                let (mut idx, lead_stages) = build_indexer(&p);
                let lag_counts = vec![p.l; p.n_hyd];
                idx.set_nonzero_mask(&lag_counts, &lead_stages);
                let expected =
                    p.n_hyd + lag_counts.iter().sum::<usize>() + lead_stages.iter().sum::<usize>();
                assert_eq!(
                    idx.nonzero_state_indices.len(),
                    expected,
                    "I13 failed at {p:?}: mask len {} != {}",
                    idx.nonzero_state_indices.len(),
                    expected
                );
            }
        }

        /// Sweep coverage assertion: the pruned grid must exercise at least 500
        /// distinct configurations to give meaningful coverage of the joint
        /// parameter space.
        #[test]
        fn sweep_coverage_at_least_500() {
            let count = parameter_grid().count();
            assert!(
                count >= 500,
                "sweep coverage {count} is below the 500-config minimum"
            );
        }
    }

    // ---------------------------------------------------------------------------
    // Layout-invariance tests (indexer-layout-impact.md Q1)
    // ---------------------------------------------------------------------------

    /// Locks the `n_state` formula against the addition of the
    /// `anticipated_state_out` control-region column. Per
    /// `indexer-layout-impact.md` Q1, `anticipated_state_out` is not a state
    /// index and must not contribute to `n_state`.
    #[test]
    fn test_n_state_unchanged_with_anticipated_state_out_addition() {
        let fpha_empty = fpha(vec![], vec![]);

        // Case A: no anticipated thermals.
        // n_state = hydro_count * (1 + max_par_order) = 3 * (1 + 2) = 9.
        let counts_a = EquipmentCounts {
            hydro_count: 3,
            max_par_order: 2,
            n_thermals: 1,
            n_lines: 1,
            n_buses: 1,
            n_blks: 1,
            has_inflow_penalty: false,
            max_deficit_segments: 1,
            n_anticipated: 0,
            k_max: 0,
            anticipated_lead_stages: vec![],
            anticipated_thermal_indices: vec![],
        };
        let idx_a = StageIndexer::with_equipment(&counts_a, &fpha_empty);
        assert_eq!(idx_a.n_state, 3 * (1 + 2), "n_state without anticipated");
        assert_eq!(idx_a.anticipated_state_out, 0..0);

        // Case B: with anticipated thermals.
        // n_state = hydro_count * (1 + max_par_order) + n_anticipated * k_max
        //         = 3 * (1 + 2) + 2 * 3 = 15.
        // anticipated_state_out has length n_anticipated = 2, but is a
        // control-region column and must NOT be counted in n_state.
        let counts_b = EquipmentCounts {
            hydro_count: 3,
            max_par_order: 2,
            n_thermals: 2,
            n_lines: 1,
            n_buses: 1,
            n_blks: 1,
            has_inflow_penalty: false,
            max_deficit_segments: 1,
            n_anticipated: 2,
            k_max: 3,
            anticipated_lead_stages: vec![2, 3],
            anticipated_thermal_indices: vec![0, 1],
        };
        let idx_b = StageIndexer::with_equipment(&counts_b, &fpha_empty);
        assert_eq!(
            idx_b.n_state,
            3 * (1 + 2) + 2 * 3,
            "n_state formula must be N*(1+L) + A*K_max"
        );
        assert_eq!(
            idx_b.anticipated_state_out.end - idx_b.anticipated_state_out.start,
            2,
            "anticipated_state_out range length must equal n_anticipated"
        );
    }
}
