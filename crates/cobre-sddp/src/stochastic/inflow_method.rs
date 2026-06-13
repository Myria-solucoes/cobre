//! Inflow non-negativity treatment method for SDDP subproblems.
//!
//! [`InflowNonNegativityMethod`] is a flat enum representing the strategies
//! for handling negative PAR(p) inflow realisations in the LP subproblems.
//! It is dispatched via `match` when constructing LP templates and extracting
//! simulation results. Enum dispatch is used because the variant set is closed
//! (enum dispatch for closed variant sets).

/// Inflow non-negativity treatment method.
///
/// Determines whether slack columns are added to the LP and what objective
/// coefficient they carry.  The variant must be the same across all stages
/// (set once at solver initialisation from the loaded case config).
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::InflowNonNegativityMethod;
///
/// let penalty = InflowNonNegativityMethod::Penalty;
/// assert!(penalty.has_slack_columns());
///
/// let none = InflowNonNegativityMethod::None;
/// assert!(!none.has_slack_columns());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum InflowNonNegativityMethod {
    /// No inflow non-negativity enforcement.
    ///
    /// The LP does not include slack columns.  Negative inflow noise may cause
    /// LP infeasibility if the PAR(p) noise realisation is sufficiently negative.
    None,

    /// Truncation-based enforcement: clamps negative PAR(p) inflows to zero
    /// by modifying the noise vector before LP patching.
    ///
    /// Does not add slack columns to the LP.  Instead, the PAR(p) model is
    /// evaluated outside the LP to obtain the full inflow value; if the result
    /// is negative, the noise component is adjusted so that the inflow is zero.
    /// This prevents LP infeasibility without perturbing the objective function.
    Truncation,

    /// Penalty-based enforcement.
    ///
    /// Appends `N` slack columns (`sigma_inf_h >= 0`) to the LP.  Each slack
    /// enters the water balance row for hydro `h` with coefficient
    /// `tau_total * M3S_TO_HM3`, where `tau_total` is the total stage duration
    /// in hours.  The objective coefficient is sourced from
    /// `penalties.json → hydro.inflow_nonnegativity_cost` (default 1000.0).
    Penalty,

    /// Combined truncation and penalty enforcement.
    ///
    /// The PAR(p) noise is clamped (identical to `Truncation`) so that the
    /// mean + noise inflow is never negative. In addition, penalty slack
    /// columns are added (identical to `Penalty`) so the solver can "undo"
    /// part of the clamping if cost-effective. This matches `SPTcpp`'s
    /// `truncamento_penalizacao` mode.
    TruncationWithPenalty,
}

impl InflowNonNegativityMethod {
    /// Returns `true` when slack columns are appended to the LP.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::InflowNonNegativityMethod;
    ///
    /// assert!(!InflowNonNegativityMethod::None.has_slack_columns());
    /// assert!(InflowNonNegativityMethod::Penalty.has_slack_columns());
    /// ```
    #[must_use]
    pub fn has_slack_columns(&self) -> bool {
        matches!(
            self,
            InflowNonNegativityMethod::Penalty | InflowNonNegativityMethod::TruncationWithPenalty
        )
    }
}

impl From<&cobre_io::config::InflowNonNegativityConfig> for InflowNonNegativityMethod {
    /// Convert from the cobre-io config type.
    ///
    /// Recognised method values are `None`, `Truncation`, `Penalty`,
    /// and `TruncationWithPenalty` (the typed enum — typos are rejected at parse
    /// time before this conversion runs).
    fn from(cfg: &cobre_io::config::InflowNonNegativityConfig) -> Self {
        match cfg.method {
            cobre_io::config::InflowNonNegativityMethod::None => InflowNonNegativityMethod::None,
            cobre_io::config::InflowNonNegativityMethod::Truncation => {
                InflowNonNegativityMethod::Truncation
            }
            cobre_io::config::InflowNonNegativityMethod::Penalty => {
                InflowNonNegativityMethod::Penalty
            }
            cobre_io::config::InflowNonNegativityMethod::TruncationWithPenalty => {
                InflowNonNegativityMethod::TruncationWithPenalty
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::InflowNonNegativityMethod;
    use cobre_io::config::{InflowNonNegativityConfig, InflowNonNegativityMethod as CfgMethod};

    // ── has_slack_columns ────────────────────────────────────────────────────

    #[test]
    fn none_has_no_slack_columns() {
        assert!(!InflowNonNegativityMethod::None.has_slack_columns());
    }

    #[test]
    fn truncation_has_no_slack_columns() {
        assert!(!InflowNonNegativityMethod::Truncation.has_slack_columns());
    }

    #[test]
    fn penalty_has_slack_columns() {
        assert!(InflowNonNegativityMethod::Penalty.has_slack_columns());
    }

    // ── conversion from config ───────────────────────────────────────────────

    #[test]
    fn test_inflow_method_conversion_none() {
        let cfg = InflowNonNegativityConfig {
            method: CfgMethod::None,
        };
        assert_eq!(
            InflowNonNegativityMethod::from(&cfg),
            InflowNonNegativityMethod::None
        );
    }

    #[test]
    fn test_inflow_method_conversion_penalty() {
        let cfg = InflowNonNegativityConfig {
            method: CfgMethod::Penalty,
        };
        assert_eq!(
            InflowNonNegativityMethod::from(&cfg),
            InflowNonNegativityMethod::Penalty
        );
    }

    #[test]
    fn test_inflow_method_conversion_truncation() {
        let cfg = InflowNonNegativityConfig {
            method: CfgMethod::Truncation,
        };
        assert_eq!(
            InflowNonNegativityMethod::from(&cfg),
            InflowNonNegativityMethod::Truncation
        );
    }

    // ── TruncationWithPenalty ───────────────────────────────────────────────

    #[test]
    fn truncation_with_penalty_has_slack_columns() {
        assert!(InflowNonNegativityMethod::TruncationWithPenalty.has_slack_columns());
    }

    #[test]
    fn test_inflow_method_conversion_truncation_with_penalty() {
        let cfg = InflowNonNegativityConfig {
            method: CfgMethod::TruncationWithPenalty,
        };
        assert_eq!(
            InflowNonNegativityMethod::from(&cfg),
            InflowNonNegativityMethod::TruncationWithPenalty
        );
    }
}
