//! Sentinel constants for LP-solver profile iteration limits.
//!
//! These constants are referenced by both `HighsProfile` and the individual
//! `SolverInterface` setter methods that accept raw iteration counts.

/// Sentinel `u32` value indicating "use the historical heuristic
/// `num_cols * 50 max 100_000`" for the simplex iteration limit.
///
/// When [`crate::highs::HighsProfile::simplex_iteration_limit`] equals this
/// value, solver implementations MUST fall back to their per-call heuristic
/// rather than applying a flat iteration cap. Any non-zero value is applied
/// verbatim as the cap.
///
/// `0` is chosen as the sentinel because zero iterations is nonsensical for
/// any LP solver; legitimate iteration caps are always positive.
pub const DEFAULT_PROFILE_HEURISTIC_SENTINEL: u32 = 0;

/// Sentinel `u32` value indicating "unbounded" for the IPM iteration limit.
///
/// When [`crate::highs::HighsProfile::ipm_iteration_limit`] equals this value,
/// solver implementations MUST apply no cap (i.e., pass `i32::MAX` to the
/// underlying solver). Any positive value is applied verbatim as the cap.
///
/// `0` is chosen as the sentinel because zero IPM iterations is nonsensical;
/// legitimate iteration caps are always positive.
pub const DEFAULT_PROFILE_IPM_UNBOUNDED_SENTINEL: u32 = 0;
