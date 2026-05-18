//! Export flag configuration for `config.json → exports`.

use serde::{Deserialize, Serialize};

/// Export flags controlling which outputs are written to disk
/// (`config.json → exports`).
///
/// Only the two active fields are accepted. Legacy keys (`training`, `cuts`,
/// `vertices`, `simulation`, `forward_detail`, `backward_detail`,
/// `compression`) must be removed from existing `config.json` files before
/// loading — they are now rejected as unknown fields.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ExportsConfig {
    /// Export visited forward-pass trial points to the policy checkpoint.
    pub states: bool,

    /// Export stochastic preprocessing artifacts to `output/stochastic/`.
    pub stochastic: bool,
}
