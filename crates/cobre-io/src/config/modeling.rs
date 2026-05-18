//! Modeling option types for `config.json → modeling`.

use serde::{Deserialize, Serialize};

/// Method string for inflow non-negativity enforcement.
///
/// Accepted values in `config.json → modeling.inflow_non_negativity.method`:
///
/// - `"none"` — no enforcement; PAR(p) inflows may be negative.
/// - `"truncation"` — clamp negative PAR(p) inflows to zero before LP patching.
/// - `"penalty"` — add slack columns with `inflow_nonnegativity_cost` objective
///   coefficient (sourced from `penalties.json → hydro.inflow_nonnegativity_cost`).
/// - `"truncation_with_penalty"` — combine both: clamp noise *and* add slack columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum InflowNonNegativityMethod {
    /// No inflow non-negativity enforcement.
    None,
    /// Truncation-based enforcement only (no slack columns).
    Truncation,
    /// Penalty-based enforcement via slack columns.
    ///
    /// Objective coefficient is `penalties.json → hydro.inflow_nonnegativity_cost`.
    #[default]
    Penalty,
    /// Combined truncation and penalty enforcement.
    TruncationWithPenalty,
}

/// Modeling options (`config.json → modeling`).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ModelingConfig {
    /// Strategy for handling non-negative inflow constraints.
    #[serde(default)]
    pub inflow_non_negativity: InflowNonNegativityConfig,
}

/// Inflow non-negativity treatment settings.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct InflowNonNegativityConfig {
    /// Method: `"none"`, `"truncation"`, `"penalty"`, or `"truncation_with_penalty"`.
    ///
    /// Default: `"penalty"`. The penalty objective coefficient is always sourced from
    /// `penalties.json → hydro.inflow_nonnegativity_cost` (default 1000.0 when absent).
    pub method: InflowNonNegativityMethod,
}

impl Default for InflowNonNegativityConfig {
    fn default() -> Self {
        Self {
            method: InflowNonNegativityMethod::Penalty,
        }
    }
}
