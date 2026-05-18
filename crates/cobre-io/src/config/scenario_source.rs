//! Raw serde types for scenario source configuration in `config.json`.
//!
//! These are intermediate deserialization types. Conversion to the canonical
//! [`cobre_core::scenario::ScenarioSource`] is performed by the helpers in
//! `config/mod.rs`.

use serde::{Deserialize, Serialize};

/// Intermediate serde type for per-class scenario source configuration in `config.json`.
///
/// Scoped to `config.json` fields (`training.scenario_source` /
/// `simulation.scenario_source`).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RawScenarioSourceConfig {
    /// Optional random seed for reproducible scenario generation.
    #[serde(default)]
    pub seed: Option<i64>,

    /// Historical year pool. Absent means `None` (auto-discover at validation time).
    #[serde(default)]
    pub historical_years: Option<RawHistoricalYearsConfig>,

    /// Inflow class scenario config. Absent defaults to `in_sample`.
    #[serde(default)]
    pub inflow: Option<RawClassConfigEntry>,

    /// Load class scenario config. Absent defaults to `in_sample`.
    #[serde(default)]
    pub load: Option<RawClassConfigEntry>,

    /// NCS class scenario config. Absent defaults to `in_sample`.
    #[serde(default)]
    pub ncs: Option<RawClassConfigEntry>,
}

/// Intermediate serde type for a single per-class scenario scheme in `config.json`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RawClassConfigEntry {
    /// Scheme string: `"in_sample"`, `"out_of_sample"`, `"external"`, or `"historical"`.
    pub scheme: String,
}

/// Intermediate serde type for `historical_years` in `config.json`.
///
/// Handles two JSON representations via `#[serde(untagged)]`:
/// - Array: `[1940, 1953, 1971]` → [`RawHistoricalYearsConfig::List`]
/// - Object: `{"from": 1940, "to": 2010}` → [`RawHistoricalYearsConfig::Range`]
///
/// The `List` variant must be declared first so serde tries it before `Range`
/// (an integer array is tried before an object).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum RawHistoricalYearsConfig {
    /// Explicit list of year integers.
    List(Vec<i32>),
    /// Inclusive range shorthand.
    Range {
        /// First year (inclusive).
        from: i32,
        /// Last year (inclusive).
        to: i32,
    },
}
