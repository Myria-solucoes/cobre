//! Integration tests for config/stages defaults cascade.
#![allow(clippy::unwrap_used, clippy::panic, clippy::doc_markdown)]

use cobre_io::config::{InflowNonNegativityMethod, parse_config};
use std::io::Write;
use tempfile::NamedTempFile;

fn write_json(content: &str) -> NamedTempFile {
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(content.as_bytes()).unwrap();
    f
}

#[test]
fn test_minimal_config_all_defaults() {
    let f = write_json(
        r#"{
          "training": {
            "forward_passes": 50,
            "stopping_rules": [{"type": "iteration_limit", "limit": 10}]
          }
        }"#,
    );
    let cfg = parse_config(f.path()).unwrap();

    assert_eq!(
        cfg.modeling.inflow_non_negativity.method,
        InflowNonNegativityMethod::Penalty,
        "inflow_non_negativity.method should default to Penalty"
    );

    assert!(
        cfg.training.enabled,
        "training.enabled should default to true"
    );
    assert_eq!(
        cfg.training.stopping_mode, "any",
        "training.stopping_mode should default to 'any'"
    );
    assert!(
        cfg.training.tree_seed.is_none(),
        "training.tree_seed should default to None when absent"
    );

    assert!(
        !cfg.simulation.enabled,
        "simulation.enabled should default to false"
    );
    assert_eq!(
        cfg.simulation.num_scenarios, 2000,
        "simulation.num_scenarios should default to 2000"
    );

    assert_eq!(
        cfg.policy.mode,
        cobre_io::PolicyMode::Fresh,
        "policy.mode should default to 'fresh'"
    );
    assert_eq!(
        cfg.policy.path, "./policy",
        "policy.path should default to './policy'"
    );

    assert!(
        !cfg.exports.states,
        "exports.states should default to false"
    );
    assert!(
        !cfg.exports.stochastic,
        "exports.stochastic should default to false"
    );
}

#[test]
fn test_config_explicit_seed_preserved() {
    let f = write_json(
        r#"{
          "training": {
            "tree_seed": 99,
            "forward_passes": 50,
            "stopping_rules": [{"type": "iteration_limit", "limit": 10}]
          }
        }"#,
    );
    let cfg = parse_config(f.path()).unwrap();

    assert_eq!(
        cfg.training.tree_seed,
        Some(99),
        "training.tree_seed should be Some(99) when explicitly set"
    );
}

#[test]
fn test_config_absent_seed_is_none() {
    let f = write_json(
        r#"{
          "training": {
            "forward_passes": 50,
            "stopping_rules": [{"type": "iteration_limit", "limit": 10}]
          }
        }"#,
    );
    let cfg = parse_config(f.path()).unwrap();

    assert!(
        cfg.training.tree_seed.is_none(),
        "training.tree_seed must be None when not present in JSON"
    );
}

#[test]
fn test_config_all_sections_explicit_no_defaults_applied() {
    let f = write_json(
        r#"{
          "modeling": {
            "inflow_non_negativity": {
              "method": "truncation"
            }
          },
          "training": {
            "enabled": false,
            "tree_seed": 7,
            "forward_passes": 192,
            "stopping_rules": [{"type": "iteration_limit", "limit": 200}],
            "stopping_mode": "all"
          },
          "simulation": {
            "enabled": true,
            "num_scenarios": 500
          },
          "policy": {
            "path": "./my_policy",
            "mode": "warm_start",
            "validate_compatibility": false
          },
          "exports": {
            "states": true,
            "stochastic": true
          }
        }"#,
    );
    let cfg = parse_config(f.path()).unwrap();

    assert_eq!(
        cfg.modeling.inflow_non_negativity.method,
        InflowNonNegativityMethod::Truncation
    );

    assert!(!cfg.training.enabled, "enabled: false should be preserved");
    assert_eq!(cfg.training.tree_seed, Some(7));
    assert_eq!(cfg.training.forward_passes, Some(192));
    assert_eq!(cfg.training.stopping_mode, "all");

    assert!(
        cfg.simulation.enabled,
        "simulation.enabled: true should be preserved"
    );
    assert_eq!(cfg.simulation.num_scenarios, 500);

    assert_eq!(cfg.policy.path, "./my_policy");
    assert_eq!(cfg.policy.mode, cobre_io::PolicyMode::WarmStart);
    assert!(!cfg.policy.validate_compatibility);

    assert!(cfg.exports.states);
    assert!(cfg.exports.stochastic);
}

#[test]
fn test_config_absent_modeling_uses_defaults() {
    let f = write_json(
        r#"{
          "training": {
            "forward_passes": 10,
            "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
          }
        }"#,
    );
    let cfg = parse_config(f.path()).unwrap();

    assert_eq!(
        cfg.modeling.inflow_non_negativity.method,
        InflowNonNegativityMethod::Penalty,
        "absent modeling section must default method to Penalty"
    );
}

#[test]
fn test_config_absent_simulation_uses_defaults() {
    let f = write_json(
        r#"{
          "training": {
            "forward_passes": 10,
            "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
          }
        }"#,
    );
    let cfg = parse_config(f.path()).unwrap();

    assert!(
        !cfg.simulation.enabled,
        "absent simulation section must default enabled to false"
    );
    assert_eq!(
        cfg.simulation.num_scenarios, 2000,
        "absent simulation section must default num_scenarios to 2000"
    );
}

#[test]
fn test_config_absent_exports_uses_defaults() {
    let f = write_json(
        r#"{
          "training": {
            "forward_passes": 10,
            "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
          }
        }"#,
    );
    let cfg = parse_config(f.path()).unwrap();

    assert!(
        !cfg.exports.states,
        "absent exports section must default exports.states to false"
    );
    assert!(
        !cfg.exports.stochastic,
        "absent exports section must default exports.stochastic to false"
    );
}
