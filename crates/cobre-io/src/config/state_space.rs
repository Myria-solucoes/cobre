//! State-space option types for `config.json → state_space`.

use serde::{Deserialize, Serialize};

/// State-space options (`config.json → state_space`).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct StateSpaceConfig {
    /// Number of active inflow-lag state slots, decoupled from the fitted AR
    /// order. Absent leaves the depth undeclared; a present value must be `>= 1`.
    /// Optional override: when a boundary policy (`policy.boundary`) is loaded, the
    /// required depth is inferred from its cuts and widens this value, so the field
    /// need not be declared for a boundary-loaded study.
    pub inflow_lag_depth: Option<u32>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::StateSpaceConfig;

    #[test]
    fn absent_inflow_lag_depth_defaults_to_none() {
        let cfg: StateSpaceConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.inflow_lag_depth, None);
    }

    #[test]
    fn present_inflow_lag_depth_parses_to_some() {
        let cfg: StateSpaceConfig = serde_json::from_str(r#"{"inflow_lag_depth": 12}"#).unwrap();
        assert_eq!(cfg.inflow_lag_depth, Some(12));
    }

    #[test]
    fn unknown_nested_key_rejected() {
        let err = serde_json::from_str::<StateSpaceConfig>(r#"{"bogus": 1}"#).unwrap_err();
        assert!(
            err.to_string().contains("unknown field"),
            "expected an unknown-field error, got: {err}"
        );
    }
}
