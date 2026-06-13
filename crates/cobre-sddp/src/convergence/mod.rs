//! Convergence monitoring for the SDDP training loop.
//!
//! This directory clusters the three components that decide *when* training
//! stops and *how* per-iteration bounds are aggregated:
//!
//! - [`convergence`] — [`ConvergenceMonitor`](convergence::ConvergenceMonitor):
//!   tracks the lower bound, upper bound, gap, and per-iteration history, and
//!   evaluates the stopping rules. Its [`upper_bound`](convergence::ConvergenceMonitor::upper_bound)
//!   returns the raw per-iteration upper bound with **no** exponential smoothing
//!   — a contract, not an oversight.
//! - [`stopping_rule`] — stopping-rule variants ([`StoppingRule`](stopping_rule::StoppingRule)),
//!   their AND/OR composition ([`StoppingRuleSet`](stopping_rule::StoppingRuleSet)),
//!   and the convergence state they read ([`MonitorState`](stopping_rule::MonitorState)).
//! - [`risk_measure`] — [`RiskMeasure`](risk_measure::RiskMeasure) cut aggregation
//!   and the backward-pass outcome it consumes ([`BackwardOutcome`](risk_measure::BackwardOutcome)).

// Rationale: the convergence-monitor file keeps its established `convergence`
// basename inside this `convergence/` cluster, so the submodule shares its
// parent's name. Renaming the submodule (e.g. to `monitor`) would break the
// `cobre_sddp::convergence::convergence::ConvergenceMonitor` re-export path for
// no behavioural gain.
#[allow(clippy::module_inception)]
pub mod convergence;
pub mod risk_measure;
pub mod stopping_rule;
