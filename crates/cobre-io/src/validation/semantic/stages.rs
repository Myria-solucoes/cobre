//! Layer 5b — stage-structure semantic validation.
//!
//! Validates policy-graph transitions (`source_id`/`target_id`
//! existence), outgoing-transition probability-sum invariant,
//! cyclic-graph discount rate, block durations, and `CVaR`
//! parameters.

use std::collections::{HashMap, HashSet};

use cobre_core::temporal::{PolicyGraphType, StageRiskConfig};

use super::super::{ErrorKind, ValidationContext, schema::ParsedData};
use super::PROB_TOLERANCE;

/// Validates policy graph transitions, block durations, and `CVaR` parameters.
pub(super) fn check_stage_structure(data: &ParsedData, ctx: &mut ValidationContext) {
    let graph = &data.stages.policy_graph;
    let stages = &data.stages.stages;

    // Build a set of all valid stage IDs for fast membership tests.
    let stage_ids: HashSet<i32> = stages.iter().map(|s| s.id).collect();

    // Rule 1: Every source_id and target_id in transitions must be a valid stage ID.
    for transition in &graph.transitions {
        if !stage_ids.contains(&transition.source_id) {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "stages.json",
                None::<&str>,
                format!(
                    "transition source_id {} does not refer to a valid stage ID",
                    transition.source_id
                ),
            );
        }
        if !stage_ids.contains(&transition.target_id) {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "stages.json",
                None::<&str>,
                format!(
                    "transition target_id {} does not refer to a valid stage ID",
                    transition.target_id
                ),
            );
        }
    }

    // Rule 2: For each unique source_id, outgoing probability sum must be ≈ 1.0.
    // Group transitions by source_id and sum probabilities.
    let mut prob_sums: HashMap<i32, f64> = HashMap::new();
    for transition in &graph.transitions {
        *prob_sums.entry(transition.source_id).or_insert(0.0) += transition.probability;
    }
    let mut sorted_sources: Vec<i32> = prob_sums.keys().copied().collect();
    sorted_sources.sort_unstable();
    for source_id in sorted_sources {
        let total = prob_sums[&source_id];
        if (total - 1.0).abs() > PROB_TOLERANCE {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "stages.json",
                None::<&str>,
                format!(
                    "outgoing transition probabilities from stage {source_id} sum to {total:.8} \
                     (expected 1.0 ±{PROB_TOLERANCE}); probability must sum to 1.0"
                ),
            );
        }
    }

    // Rule 3: Cyclic graphs require annual_discount_rate > 0.0.
    if graph.graph_type == PolicyGraphType::Cyclic && graph.annual_discount_rate <= 0.0 {
        ctx.add_error(
            ErrorKind::InvalidValue,
            "stages.json",
            None::<&str>,
            format!(
                "cyclic policy graph requires annual_discount_rate > 0.0 for convergence, \
                 got {}",
                graph.annual_discount_rate
            ),
        );
    }

    // Rule 4: Every Block.duration_hours must be > 0.0.
    for stage in stages {
        for block in &stage.blocks {
            if block.duration_hours <= 0.0 {
                ctx.add_error(
                    ErrorKind::InvalidValue,
                    "stages.json",
                    Some(format!("Stage {}", stage.id)),
                    format!(
                        "Stage {}: block has duration_hours {} which is not > 0.0; \
                         block duration must be positive",
                        stage.id, block.duration_hours
                    ),
                );
            }
        }
    }

    // Rule 5: CVaR alpha must be in (0, 1] and lambda must be in [0, 1].
    for stage in stages {
        if let StageRiskConfig::CVaR { alpha, lambda } = stage.risk_config {
            if alpha <= 0.0 || alpha > 1.0 {
                ctx.add_error(
                    ErrorKind::InvalidValue,
                    "stages.json",
                    Some(format!("Stage {}", stage.id)),
                    format!(
                        "Stage {}: CVaR alpha ({alpha}) must be in (0, 1]; \
                         alpha must be a valid tail probability",
                        stage.id
                    ),
                );
            }
            if !(0.0..=1.0).contains(&lambda) {
                ctx.add_error(
                    ErrorKind::InvalidValue,
                    "stages.json",
                    Some(format!("Stage {}", stage.id)),
                    format!(
                        "Stage {}: CVaR lambda ({lambda}) must be in [0, 1]; \
                         lambda is the CVaR mixing weight",
                        stage.id
                    ),
                );
            }
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
mod tests {
    use super::super::test_support::*;
    use super::super::validate_semantic_stages_penalties_scenarios;
    use cobre_core::temporal::{Block, PolicyGraphType, StageRiskConfig, Transition};

    use crate::validation::{ErrorKind, ValidationContext};

    // ── Rule 1: Transition stage validity ─────────────────────────────────────

    /// Transition referencing a non-existent source_id produces InvalidValue error.
    #[test]
    fn test_5b_transition_invalid_source_id() {
        let mut stages = make_stages_5b(vec![0, 1]);
        stages.policy_graph.transitions = vec![Transition {
            source_id: 99, // does not exist
            target_id: 1,
            probability: 1.0,
            annual_discount_rate_override: None,
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::InvalidValue),
            "should have InvalidValue for invalid source_id"
        );
    }

    /// Transition referencing a non-existent target_id produces InvalidValue error.
    #[test]
    fn test_5b_transition_invalid_target_id() {
        let mut stages = make_stages_5b(vec![0, 1]);
        stages.policy_graph.transitions = vec![Transition {
            source_id: 0,
            target_id: 99, // does not exist
            probability: 1.0,
            annual_discount_rate_override: None,
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidValue),
            "should have InvalidValue for invalid target_id"
        );
    }

    // ── Rule 2: Transition probability sums ───────────────────────────────────

    /// Transitions from stage 0 with probability sum 0.5 produce one InvalidValue
    /// error with "probability" and "stage 0" in the message.
    #[test]
    fn test_5b_transition_probability_sum_wrong() {
        let mut stages = make_stages_5b(vec![0, 1]);
        stages.policy_graph.transitions = vec![Transition {
            source_id: 0,
            target_id: 1,
            probability: 0.5, // should sum to 1.0
            annual_discount_rate_override: None,
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(relevant.len(), 1, "exactly 1 InvalidValue error expected");
        let msg = &relevant[0].message;
        assert!(
            msg.contains("probability"),
            "message should contain 'probability', got: {msg}"
        );
        assert!(
            msg.contains("stage 0"),
            "message should contain 'stage 0', got: {msg}"
        );
    }

    /// Transitions from stage 0 summing exactly 1.0 produce no probability error.
    #[test]
    fn test_5b_transition_probability_sum_valid() {
        let mut stages = make_stages_5b(vec![0, 1, 2]);
        stages.policy_graph.transitions = vec![
            Transition {
                source_id: 0,
                target_id: 1,
                probability: 0.6,
                annual_discount_rate_override: None,
            },
            Transition {
                source_id: 0,
                target_id: 2,
                probability: 0.4,
                annual_discount_rate_override: None,
            },
        ];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let prob_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert!(
            prob_errors.is_empty(),
            "valid probability sum should produce no InvalidValue errors, got: {prob_errors:?}"
        );
    }

    // ── Rule 3: Cyclic discount rate ──────────────────────────────────────────

    /// Cyclic graph with annual_discount_rate = 0.0 produces InvalidValue error.
    #[test]
    fn test_5b_cyclic_zero_discount_rate() {
        let mut stages = make_stages_5b(vec![0]);
        stages.policy_graph.graph_type = PolicyGraphType::Cyclic;
        stages.policy_graph.annual_discount_rate = 0.0;
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidValue),
            "cyclic with 0 discount rate should produce InvalidValue"
        );
    }

    /// Cyclic graph with annual_discount_rate > 0.0 produces no discount rate error.
    #[test]
    fn test_5b_cyclic_positive_discount_rate_valid() {
        let mut stages = make_stages_5b(vec![0]);
        stages.policy_graph.graph_type = PolicyGraphType::Cyclic;
        stages.policy_graph.annual_discount_rate = 0.06;
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let discount_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert!(
            discount_errors.is_empty(),
            "cyclic with positive discount rate should produce no error, got: {discount_errors:?}"
        );
    }

    // ── Rule 4: Block duration positivity ─────────────────────────────────────

    /// A block with duration_hours = 0.0 produces an InvalidValue error.
    #[test]
    fn test_5b_block_zero_duration() {
        let mut stages = make_stages_5b(vec![0]);
        stages.stages[0].blocks = vec![Block {
            index: 0,
            name: "Peak".to_string(),
            duration_hours: 0.0, // invalid
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidValue),
            "zero duration block should produce InvalidValue"
        );
    }

    /// A block with positive duration_hours produces no block duration error.
    #[test]
    fn test_5b_block_positive_duration_valid() {
        let mut stages = make_stages_5b(vec![0]);
        stages.stages[0].blocks = vec![Block {
            index: 0,
            name: "Peak".to_string(),
            duration_hours: 168.0,
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert!(
            errors.is_empty(),
            "positive block duration should produce no error, got: {errors:?}"
        );
    }

    // ── Rule 5: CVaR parameter validity ───────────────────────────────────────

    /// CVaR alpha = 0.0 (invalid, must be in (0, 1]) produces InvalidValue.
    #[test]
    fn test_5b_cvar_alpha_zero_invalid() {
        let mut stages = make_stages_5b(vec![0]);
        stages.stages[0].risk_config = StageRiskConfig::CVaR {
            alpha: 0.0, // invalid: must be in (0, 1]
            lambda: 0.5,
        };
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidValue),
            "CVaR alpha=0.0 should produce InvalidValue"
        );
    }

    /// CVaR lambda = -0.1 (invalid, must be in [0, 1]) produces InvalidValue.
    #[test]
    fn test_5b_cvar_lambda_out_of_range() {
        let mut stages = make_stages_5b(vec![0]);
        stages.stages[0].risk_config = StageRiskConfig::CVaR {
            alpha: 0.95,
            lambda: -0.1, // invalid: must be in [0, 1]
        };
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidValue),
            "CVaR lambda=-0.1 should produce InvalidValue"
        );
    }
}
