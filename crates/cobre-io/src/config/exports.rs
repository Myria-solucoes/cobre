//! Export flag configuration for `config.json → exports`.

use serde::{Deserialize, Serialize};

/// Export flags controlling which outputs are written to disk
/// (`config.json → exports`).
///
/// Only the two flags with active consumers are retained. Keys for removed
/// fields (`training`, `cuts`, `vertices`, `simulation`, `forward_detail`,
/// `backward_detail`, `compression`) are silently ignored when present in
/// legacy `config.json` files.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ExportsConfig {
    /// Export visited forward-pass trial points to the policy checkpoint.
    pub states: bool,

    /// Export stochastic preprocessing artifacts to `output/stochastic/`.
    pub stochastic: bool,
}
