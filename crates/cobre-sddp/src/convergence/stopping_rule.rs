//! Stopping rules for the SDDP training loop.
//!
//! Defines stopping rule variants, composition logic, and convergence state.
//! Rules use enum dispatch; [`StoppingRuleSet`] composes them with AND/OR logic.
//!
//! ## Usage
//!
//! ```rust
//! use cobre_sddp::stopping_rule::{
//!     MonitorState, StoppingMode, StoppingRule, StoppingRuleSet,
//! };
//!
//! let state = MonitorState {
//!     iteration: 10,
//!     wall_time_seconds: 50.0,
//!     lower_bound: 100.0,
//!     upper_bound: 110.0,
//!     lower_bound_history: vec![90.0, 95.0, 98.0, 99.0, 100.0,
//!                              100.0, 100.0, 100.0, 100.0, 100.0],
//!     shutdown_requested: false,
//! };
//!
//! let rule = StoppingRule::IterationLimit { limit: 10 };
//! let result = rule.evaluate(&state);
//! assert!(result.triggered);
//! assert_eq!(result.rule_name, "iteration_limit");
//! ```

use std::borrow::Cow;

use cobre_core::StoppingRuleResult;

/// Rule name for the iteration limit stopping rule.
pub const RULE_ITERATION_LIMIT: &str = "iteration_limit";
/// Rule name for the wall-clock time limit stopping rule.
pub const RULE_TIME_LIMIT: &str = "time_limit";
/// Rule name for the lower-bound stalling stopping rule.
pub const RULE_BOUND_STALLING: &str = "bound_stalling";
/// Rule name for the exact-upper-bound gap stopping rule.
pub const RULE_GAP: &str = "gap";
/// Rule name for the graceful-shutdown stopping rule.
pub const RULE_GRACEFUL_SHUTDOWN: &str = "graceful_shutdown";

/// Guarded denominator for the RELATIVE gap: the LOWER bound's magnitude, floored
/// at `1.0`. Both the reported gap ([`crate::ConvergenceMonitor::gap`]) and the
/// [`StoppingRule::Gap`] relative arm divide by this, so the two never disagree on
/// what "relative gap" means — it is normalized by the lower bound, never the
/// upper. The `1.0` floor bounds the ratio when the lower bound is near zero
/// (startup, or a zero-cost study); the canonical-R$ lower bound is far larger
/// than `1.0` once gap-checking matters, so the floor only guards that degenerate
/// case.
pub(crate) fn relative_gap_denominator(lower_bound: f64) -> f64 {
    lower_bound.abs().max(1.0_f64)
}

// ---------------------------------------------------------------------------
// MonitorState
// ---------------------------------------------------------------------------

/// Read-only snapshot of convergence-monitor quantities consumed by
/// [`StoppingRuleSet::evaluate`].
#[derive(Debug, Clone)]
pub struct MonitorState {
    /// Current iteration index (1-based).
    pub iteration: u64,

    /// Cumulative wall-clock time since training start, in seconds.
    pub wall_time_seconds: f64,

    /// Current lower bound (stage-1 LP objective value).
    pub lower_bound: f64,

    /// Current upper bound, canonical R$; the exact `Σ w·c` bound under
    /// enumerated forwards that [`StoppingRule::Gap`] compares against.
    pub upper_bound: f64,

    /// Lower bounds from past iterations, chronological: `[i]` is iteration `i + 1`.
    pub lower_bound_history: Vec<f64>,

    /// Whether an external shutdown signal (SIGTERM / SIGINT) has been received.
    pub shutdown_requested: bool,
}

// ---------------------------------------------------------------------------
// StoppingMode
// ---------------------------------------------------------------------------

/// Combination mode for [`StoppingRuleSet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoppingMode {
    /// Stop when **any** configured rule triggers (OR logic). `GracefulShutdown`
    /// takes precedence regardless of mode.
    Any,

    /// Stop when **all** configured rules trigger at the same iteration (AND
    /// logic). `GracefulShutdown` takes precedence regardless of mode.
    All,
}

// ---------------------------------------------------------------------------
// StoppingRule
// ---------------------------------------------------------------------------

/// Individual stopping rule for the SDDP training loop, composed into a
/// [`StoppingRuleSet`]. [`StoppingRule::GracefulShutdown`] is always evaluated
/// first and bypasses composition; every set must contain at least one
/// `IterationLimit` (the safety bound against infinite loops, validated at
/// config load).
#[derive(Debug, Clone)]
pub enum StoppingRule {
    /// Terminate when the iteration count reaches a fixed limit.
    IterationLimit {
        /// Maximum iteration count. Training stops when `iteration >= limit`.
        limit: u64,
    },

    /// Terminate when cumulative wall-clock time exceeds a threshold.
    TimeLimit {
        /// Maximum wall-clock time in seconds. Training stops when
        /// `wall_time_seconds >= seconds`.
        seconds: f64,
    },

    /// Terminate when the lower bound improvement over a sliding window
    /// falls below a relative tolerance.
    ///
    /// Uses the formula:
    /// `Δ = (lb_current - lb_window_start) / max(1.0, |lb_current|)`.
    /// Triggers when `|Δ| < tolerance`.
    BoundStalling {
        /// Relative improvement tolerance. Triggers when relative improvement
        /// over the window is below this value.
        tolerance: f64,

        /// Number of past iterations over which to measure improvement (τ).
        iterations: u64,
    },

    /// Terminate once the clamped canonical-R$ gap `UB_exact − LB` satisfies the
    /// disjunction of the configured tolerance arms. Admissible only under
    /// enumerated forwards + an expectation measure (enforced by the setup
    /// admission gate); at least one tolerance arm is required (enforced at
    /// config mapping).
    Gap {
        /// Absolute gap tolerance, canonical R$.
        tolerance: Option<f64>,

        /// Relative gap tolerance in PERCENT: stops when
        /// `100·gap / max(1, |LB|) ≤ relative_tolerance`. A value of `0.01` means
        /// 0.01%, directly comparable to the reported `gap_percent`.
        relative_tolerance: Option<f64>,
    },

    /// Terminate when an external shutdown signal (SIGTERM / SIGINT) is received.
    /// Not JSON-configured — always implicitly present and evaluated before the
    /// composition logic.
    GracefulShutdown,
}

impl StoppingRule {
    /// Evaluate this rule against the current monitor state (pure; reads `state`
    /// only).
    #[must_use]
    pub fn evaluate(&self, state: &MonitorState) -> StoppingRuleResult {
        match self {
            Self::IterationLimit { limit } => {
                let triggered = state.iteration >= *limit;
                StoppingRuleResult {
                    rule_name: RULE_ITERATION_LIMIT,
                    triggered,
                    detail: Cow::Owned(format!("iteration {}/{}", state.iteration, limit)),
                }
            }

            Self::TimeLimit { seconds } => {
                let triggered = state.wall_time_seconds >= *seconds;
                StoppingRuleResult {
                    rule_name: RULE_TIME_LIMIT,
                    triggered,
                    detail: Cow::Owned(format!(
                        "elapsed {:.1}s / {:.1}s limit",
                        state.wall_time_seconds, seconds
                    )),
                }
            }

            Self::BoundStalling {
                tolerance,
                iterations,
            } => Self::evaluate_bound_stalling(state, *tolerance, *iterations),

            Self::Gap {
                tolerance,
                relative_tolerance,
            } => Self::evaluate_gap(state, *tolerance, *relative_tolerance),

            Self::GracefulShutdown => {
                let triggered = state.shutdown_requested;
                StoppingRuleResult {
                    rule_name: RULE_GRACEFUL_SHUTDOWN,
                    triggered,
                    detail: if triggered {
                        Cow::Borrowed("shutdown signal received")
                    } else {
                        Cow::Borrowed("no shutdown signal")
                    },
                }
            }
        }
    }

    /// Evaluate the [`StoppingRule::BoundStalling`] condition.
    fn evaluate_bound_stalling(
        state: &MonitorState,
        tolerance: f64,
        iterations: u64,
    ) -> StoppingRuleResult {
        // `iterations` is config-validated <= u32::MAX, so the cast cannot truncate.
        #[allow(clippy::cast_possible_truncation)]
        let window = iterations as usize;
        if state.lower_bound_history.len() < window {
            return StoppingRuleResult {
                rule_name: RULE_BOUND_STALLING,
                triggered: false,
                detail: Cow::Owned(format!(
                    "insufficient history: {}/{} iterations",
                    state.lower_bound_history.len(),
                    window
                )),
            };
        }

        let history_len = state.lower_bound_history.len();
        let lb_window_start = state.lower_bound_history[history_len - window];
        let lb_current = state.lower_bound;

        let denominator = lb_current.abs().max(1.0_f64);
        let delta = (lb_current - lb_window_start) / denominator;

        let triggered = delta.abs() < tolerance;
        StoppingRuleResult {
            rule_name: RULE_BOUND_STALLING,
            triggered,
            detail: Cow::Owned(format!(
                "relative improvement {:.6} / tolerance {:.6} over {} iterations",
                delta.abs(),
                tolerance,
                window
            )),
        }
    }

    /// Evaluate the [`StoppingRule::Gap`] condition: the clamped canonical-R$ gap
    /// `UB_exact − LB` against the disjunction of the configured tolerance arms
    /// (`gap ≤ tolerance` OR `100·gap / max(1, |LB|) ≤ relative_tolerance`). The
    /// relative arm normalizes by the LOWER bound
    /// ([`relative_gap_denominator`]) and is expressed in percent, the same
    /// convention the reported `gap_percent` uses. A small negative gap (float
    /// noise at a closed gap) is clamped to `0` before comparing.
    fn evaluate_gap(
        state: &MonitorState,
        tolerance: Option<f64>,
        relative_tolerance: Option<f64>,
    ) -> StoppingRuleResult {
        let gap = (state.upper_bound - state.lower_bound).max(0.0);
        let absolute_hit = tolerance.is_some_and(|t| gap <= t);
        let relative_hit = relative_tolerance
            .is_some_and(|r| 100.0 * gap / relative_gap_denominator(state.lower_bound) <= r);
        StoppingRuleResult {
            rule_name: RULE_GAP,
            triggered: absolute_hit || relative_hit,
            detail: Cow::Owned(format!(
                "gap {gap:.6} (abs tol {tolerance:?}, rel tol {relative_tolerance:?})"
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// StoppingRuleSet
// ---------------------------------------------------------------------------

/// Composed set of [`StoppingRule`] variants combined under a [`StoppingMode`].
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::stopping_rule::{
///     MonitorState, StoppingMode, StoppingRule, StoppingRuleSet,
/// };
///
/// let state = MonitorState {
///     iteration: 100,
///     wall_time_seconds: 1000.0,
///     lower_bound: 100.0,
///     upper_bound: 110.0,
///     lower_bound_history: vec![],
///     shutdown_requested: false,
/// };
///
/// let rule_set = StoppingRuleSet {
///     rules: vec![
///         StoppingRule::IterationLimit { limit: 100 },
///         StoppingRule::TimeLimit { seconds: 3600.0 },
///     ],
///     mode: StoppingMode::Any,
/// };
///
/// let (should_stop, results) = rule_set.evaluate(&state);
/// assert!(should_stop);
/// assert_eq!(results.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct StoppingRuleSet {
    /// The individual stopping rules. Must contain at least one
    /// [`StoppingRule::IterationLimit`] (validated at config load);
    /// [`StoppingRule::GracefulShutdown`] is evaluated unconditionally regardless
    /// of its position here.
    pub rules: Vec<StoppingRule>,

    /// Combination mode for the rules.
    pub mode: StoppingMode,
}

impl StoppingRuleSet {
    /// Evaluate all stopping rules, returning `(should_stop, all_results)`.
    ///
    /// A set shutdown flag returns `(true, results)` immediately, regardless of
    /// `mode`. Otherwise [`StoppingMode::Any`] stops if any rule triggered,
    /// [`StoppingMode::All`] stops only if all did.
    #[must_use]
    pub fn evaluate(&self, state: &MonitorState) -> (bool, Vec<StoppingRuleResult>) {
        let results: Vec<StoppingRuleResult> =
            self.rules.iter().map(|r| r.evaluate(state)).collect();

        if state.shutdown_requested {
            return (true, results);
        }

        // GracefulShutdown is excluded here — already handled above.
        let non_shutdown_triggered: Vec<bool> = self
            .rules
            .iter()
            .zip(results.iter())
            .filter(|(rule, _)| !matches!(rule, StoppingRule::GracefulShutdown))
            .map(|(_, result)| result.triggered)
            .collect();

        let should_stop = match self.mode {
            StoppingMode::Any => non_shutdown_triggered.iter().any(|&t| t),
            StoppingMode::All => {
                !non_shutdown_triggered.is_empty() && non_shutdown_triggered.iter().all(|&t| t)
            }
        };

        (should_stop, results)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{MonitorState, StoppingMode, StoppingRule, StoppingRuleSet};

    fn make_state(iteration: u64, wall_time: f64, lb: f64, history: Vec<f64>) -> MonitorState {
        MonitorState {
            iteration,
            wall_time_seconds: wall_time,
            lower_bound: lb,
            upper_bound: 0.0,
            lower_bound_history: history,
            shutdown_requested: false,
        }
    }

    fn gap_state(lb: f64, ub: f64) -> MonitorState {
        MonitorState {
            iteration: 1,
            wall_time_seconds: 0.0,
            lower_bound: lb,
            upper_bound: ub,
            lower_bound_history: vec![],
            shutdown_requested: false,
        }
    }

    #[test]
    fn iteration_limit_triggered_at_limit() {
        let rule = StoppingRule::IterationLimit { limit: 10 };
        let state = make_state(10, 0.0, 0.0, vec![]);
        let result = rule.evaluate(&state);
        assert!(result.triggered);
        assert_eq!(result.rule_name, "iteration_limit");
    }

    #[test]
    fn iteration_limit_triggered_above_limit() {
        let rule = StoppingRule::IterationLimit { limit: 10 };
        let state = make_state(15, 0.0, 0.0, vec![]);
        let result = rule.evaluate(&state);
        assert!(result.triggered);
    }

    #[test]
    fn iteration_limit_not_triggered_below_limit() {
        let rule = StoppingRule::IterationLimit { limit: 10 };
        let state = make_state(9, 0.0, 0.0, vec![]);
        let result = rule.evaluate(&state);
        assert!(!result.triggered);
    }

    #[test]
    fn time_limit_triggered_at_threshold() {
        let rule = StoppingRule::TimeLimit { seconds: 3600.0 };
        let state = make_state(1, 3600.0, 0.0, vec![]);
        let result = rule.evaluate(&state);
        assert!(result.triggered);
        assert_eq!(result.rule_name, "time_limit");
    }

    #[test]
    fn time_limit_triggered_above_threshold() {
        let rule = StoppingRule::TimeLimit { seconds: 3600.0 };
        let state = make_state(1, 3700.0, 0.0, vec![]);
        let result = rule.evaluate(&state);
        assert!(result.triggered);
    }

    #[test]
    fn time_limit_not_triggered_below_threshold() {
        let rule = StoppingRule::TimeLimit { seconds: 3600.0 };
        let state = make_state(1, 1000.0, 0.0, vec![]);
        let result = rule.evaluate(&state);
        assert!(!result.triggered);
    }

    #[test]
    fn bound_stalling_not_triggered_with_insufficient_history() {
        let rule = StoppingRule::BoundStalling {
            tolerance: 0.01,
            iterations: 5,
        };
        let state = make_state(3, 0.0, 100.0, vec![90.0, 95.0, 100.0]);
        let result = rule.evaluate(&state);
        assert!(!result.triggered);
        assert_eq!(result.rule_name, "bound_stalling");
    }

    #[test]
    fn bound_stalling_triggered_when_lb_stable() {
        let rule = StoppingRule::BoundStalling {
            tolerance: 0.011,
            iterations: 5,
        };
        let history = vec![80.0, 99.0, 99.5, 99.8, 99.9, 100.0];
        let state = make_state(6, 0.0, 100.0, history);
        let result = rule.evaluate(&state);
        assert!(result.triggered);
    }

    #[test]
    fn bound_stalling_not_triggered_when_lb_improving() {
        let rule = StoppingRule::BoundStalling {
            tolerance: 0.01,
            iterations: 5,
        };
        let history = vec![50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let state = make_state(6, 0.0, 100.0, history);
        let result = rule.evaluate(&state);
        assert!(!result.triggered);
    }

    #[test]
    fn bound_stalling_near_zero_lb_uses_max_guard() {
        let rule = StoppingRule::BoundStalling {
            tolerance: 0.01,
            iterations: 3,
        };
        let history = vec![0.0, 0.0, 0.0, 0.001];
        let state = make_state(4, 0.0, 0.001, history);
        let result = rule.evaluate(&state);
        assert!(result.triggered);
    }

    #[test]
    fn gap_absolute_arm_stops_within_tolerance() {
        let rule = StoppingRule::Gap {
            tolerance: Some(10.0),
            relative_tolerance: None,
        };
        // gap = 105 - 100 = 5 <= 10 → stop; 120 - 100 = 20 > 10 → no stop.
        assert!(rule.evaluate(&gap_state(100.0, 105.0)).triggered);
        assert!(!rule.evaluate(&gap_state(100.0, 120.0)).triggered);
        assert_eq!(rule.evaluate(&gap_state(100.0, 105.0)).rule_name, "gap");
    }

    #[test]
    fn gap_relative_arm_stops_within_relative_tolerance() {
        let rule = StoppingRule::Gap {
            tolerance: None,
            relative_tolerance: Some(10.0),
        };
        // percent gap = 100·5/100 = 5% <= 10% → stop; 100·20/100 = 20% > 10% → no stop.
        assert!(rule.evaluate(&gap_state(100.0, 105.0)).triggered);
        assert!(!rule.evaluate(&gap_state(100.0, 120.0)).triggered);
    }

    #[test]
    fn gap_disjunction_stops_when_only_relative_arm_holds() {
        // abs tol 1.0 NOT met (gap 5 > 1); rel tol 10% met (5% <= 10%) → OR stops.
        let rule = StoppingRule::Gap {
            tolerance: Some(1.0),
            relative_tolerance: Some(10.0),
        };
        assert!(rule.evaluate(&gap_state(100.0, 105.0)).triggered);
    }

    #[test]
    fn gap_disjunction_stops_when_only_absolute_arm_holds() {
        // Large |LB| starves the relative arm (percent gap 100·50/1e6 = 5e-3% >
        // 1e-9%); abs tol 100 met (gap 50 <= 100) → OR stops.
        let rule = StoppingRule::Gap {
            tolerance: Some(100.0),
            relative_tolerance: Some(1e-9),
        };
        assert!(
            rule.evaluate(&gap_state(1_000_000.0, 1_000_050.0))
                .triggered
        );
    }

    #[test]
    fn gap_negative_float_noise_clamps_to_zero_and_converges() {
        // UB a hair below LB (closed-gap float noise): clamped to 0, reported
        // non-negative, and counts as converged.
        let rule = StoppingRule::Gap {
            tolerance: Some(0.0),
            relative_tolerance: None,
        };
        let result = rule.evaluate(&gap_state(100.0, 100.0 - 1e-9));
        assert!(
            result.triggered,
            "a small negative gap must clamp to 0 and count as converged"
        );
        assert!(
            !result.detail.contains("-0.000000"),
            "the reported gap must be non-negative after clamping: {}",
            result.detail
        );
    }

    #[test]
    fn graceful_shutdown_triggered_when_requested() {
        let rule = StoppingRule::GracefulShutdown;
        let mut state = make_state(1, 0.0, 0.0, vec![]);
        state.shutdown_requested = true;
        let result = rule.evaluate(&state);
        assert!(result.triggered);
        assert_eq!(result.rule_name, "graceful_shutdown");
    }

    #[test]
    fn graceful_shutdown_not_triggered_when_not_requested() {
        let rule = StoppingRule::GracefulShutdown;
        let state = make_state(1, 0.0, 0.0, vec![]);
        let result = rule.evaluate(&state);
        assert!(!result.triggered);
    }

    #[test]
    fn rule_set_any_mode_stops_on_first_triggered_rule() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 100 },
                StoppingRule::TimeLimit { seconds: 3600.0 },
            ],
            mode: StoppingMode::Any,
        };
        let state = make_state(100, 1000.0, 0.0, vec![]);
        let (should_stop, results) = rule_set.evaluate(&state);
        assert!(should_stop);
        assert_eq!(results.len(), 2);
        assert!(results[0].triggered);
        assert!(!results[1].triggered);
    }

    #[test]
    fn rule_set_any_mode_does_not_stop_when_no_rules_trigger() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 100 },
                StoppingRule::TimeLimit { seconds: 3600.0 },
            ],
            mode: StoppingMode::Any,
        };
        let state = make_state(50, 1000.0, 0.0, vec![]);
        let (should_stop, _) = rule_set.evaluate(&state);
        assert!(!should_stop);
    }

    #[test]
    fn rule_set_all_mode_stops_only_when_all_rules_trigger() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 100 },
                StoppingRule::TimeLimit { seconds: 3600.0 },
            ],
            mode: StoppingMode::All,
        };
        let state = make_state(100, 4000.0, 0.0, vec![]);
        let (should_stop, results) = rule_set.evaluate(&state);
        assert!(should_stop);
        assert!(results[0].triggered);
        assert!(results[1].triggered);
    }

    #[test]
    fn rule_set_all_mode_does_not_stop_when_only_one_triggers() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 100 },
                StoppingRule::TimeLimit { seconds: 3600.0 },
            ],
            mode: StoppingMode::All,
        };
        let state = make_state(100, 1000.0, 0.0, vec![]);
        let (should_stop, _) = rule_set.evaluate(&state);
        assert!(!should_stop);
    }

    #[test]
    fn rule_set_graceful_shutdown_bypasses_all_mode() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 100 },
                StoppingRule::GracefulShutdown,
            ],
            mode: StoppingMode::All,
        };
        let mut state = make_state(1, 0.0, 0.0, vec![]);
        state.shutdown_requested = true;
        let (should_stop, _) = rule_set.evaluate(&state);
        assert!(should_stop);
    }

    #[test]
    fn rule_set_graceful_shutdown_bypasses_any_mode() {
        let rule_set = StoppingRuleSet {
            rules: vec![StoppingRule::GracefulShutdown],
            mode: StoppingMode::Any,
        };
        let mut state = make_state(1, 0.0, 0.0, vec![]);
        state.shutdown_requested = true;
        let (should_stop, _) = rule_set.evaluate(&state);
        assert!(should_stop);
    }

    #[test]
    fn rule_set_returns_all_results_regardless_of_mode() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 10 },
                StoppingRule::TimeLimit { seconds: 3600.0 },
                StoppingRule::GracefulShutdown,
            ],
            mode: StoppingMode::Any,
        };
        let state = make_state(10, 100.0, 0.0, vec![]);
        let (_, results) = rule_set.evaluate(&state);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn ac_iteration_limit_triggered_at_10() {
        let rule = StoppingRule::IterationLimit { limit: 10 };
        let state = make_state(10, 0.0, 0.0, vec![]);
        let result = rule.evaluate(&state);
        assert!(result.triggered);
        assert_eq!(result.rule_name, "iteration_limit");
    }

    #[test]
    fn ac_bound_stalling_with_6_history_entries() {
        let rule = StoppingRule::BoundStalling {
            tolerance: 0.01,
            iterations: 5,
        };
        let history = vec![80.0, 99.1, 99.4, 99.7, 99.9, 100.0];
        let state = make_state(6, 0.0, 100.0, history);
        let result = rule.evaluate(&state);
        assert!(result.triggered);
    }

    #[test]
    fn ac_rule_set_any_mode_stops_at_iteration_100() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 100 },
                StoppingRule::TimeLimit { seconds: 3600.0 },
            ],
            mode: StoppingMode::Any,
        };
        let state = make_state(100, 1000.0, 0.0, vec![]);
        let (should_stop, _) = rule_set.evaluate(&state);
        assert!(should_stop);
    }
}
