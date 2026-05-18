//! Modeling option types for `config.json → modeling`.

use serde::{Deserialize, Serialize};

/// Modeling options (`config.json → modeling`).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ModelingConfig {
    /// Strategy for handling non-negative inflow constraints.
    #[serde(default)]
    pub inflow_non_negativity: InflowNonNegativityConfig,
}

/// Inflow non-negativity treatment settings.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct InflowNonNegativityConfig {
    /// Method: `"none"`, `"penalty"`, or `"truncation"`.
    pub method: String,

    /// Penalty coefficient $c^{inf}$ applied when `method` is `"penalty"`.
    ///
    /// **Deprecated:** Use `penalties.json` -> `hydro.inflow_nonnegativity_cost`
    /// instead. When both are specified, the penalty cascade takes precedence.
    /// This field is retained for backward compatibility with existing cases
    /// that do not yet have `inflow_nonnegativity_cost` in their `penalties.json`.
    pub penalty_cost: f64,
}

impl Default for InflowNonNegativityConfig {
    fn default() -> Self {
        Self {
            method: "penalty".to_string(),
            penalty_cost: 1000.0,
        }
    }
}
