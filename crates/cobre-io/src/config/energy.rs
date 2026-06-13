//! Energy conversion settings for `config.json → energy`.

use serde::{Deserialize, Serialize};

/// Energy conversion settings (`config.json → energy`).
///
/// Controls reservoir reference-volume computation for FPHA hydros.
/// `V_ref = V_min + fraction · (V_max − V_min)` is the reference storage
/// used to evaluate the equivalent head `h_eq` (and thereby `ρ_eq`).
/// Per-plant per-season overrides are loaded from
/// `system/hydro_reference_volume_fractions.parquet`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct EnergyConfig {
    /// Case-wide default fraction in `(0.0, 1.0]` used when no per-`(hydro,
    /// season)` override applies. `V_ref = V_min + fraction · (V_max − V_min)`.
    /// Default 0.65 (conventional long-term reference-volume fraction).
    pub reference_volume_fraction: f64,
}

impl Default for EnergyConfig {
    fn default() -> Self {
        Self {
            reference_volume_fraction: 0.65,
        }
    }
}
