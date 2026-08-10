//! Template post-processing: discount factors, LP scaling, and noise pre-scaling.

use cobre_core::{HorizonGraph, Stage, System};

use crate::indexer::StateSpace;
use crate::scaling_report::ScalingReport;
use crate::scaling_report::{
    LpDimensions, StageScalingReport, build_scaling_report, compute_coefficient_range,
    summarize_scale_factors,
};
use crate::{lp_builder, lp_builder::StageTemplates};

/// Compute per-stage one-step discount factors from study stages and a policy graph.
///
/// `discount_factors[t] = 1 / (1 + r_t)^(Dt / 365.25)` where `r_t` is the annual
/// discount rate for stage `t` (`HorizonGraph::stage_discount_rate_overrides` keyed
/// by `Stage::id`, else the global `annual_discount_rate`) and `Dt` is the stage
/// duration in days. When `rate == 0.0`, the factor is `1.0` (no discounting).
pub(crate) fn compute_per_stage_discount_factors(
    study_stages: &[&Stage],
    pg: &HorizonGraph,
) -> Vec<f64> {
    study_stages
        .iter()
        .map(|stage| {
            let rate = pg
                .stage_discount_rate_overrides
                .get(&stage.id)
                .copied()
                .unwrap_or(pg.annual_discount_rate);
            if rate == 0.0 {
                1.0
            } else {
                let dt_days = f64::from(
                    i32::try_from((stage.end_date - stage.start_date).num_days())
                        .unwrap_or(i32::MAX),
                );
                1.0 / (1.0 + rate).powf(dt_days / 365.25)
            }
        })
        .collect()
}

/// Compute cumulative discount factors from per-stage one-step factors.
///
/// Length is `per_stage.len()` exactly: the strict anticipated-decision predicate
/// (`stage_idx + K_i < n_stages`) keeps every delivery lookup within
/// `[0, n_stages)`, so no boundary-stage entry is needed.
pub(crate) fn compute_cumulative_discount_factors(per_stage: &[f64]) -> Vec<f64> {
    let n = per_stage.len();
    let mut cumulative = vec![1.0; n];
    for t in 1..n {
        cumulative[t] = cumulative[t - 1] * per_stage[t - 1];
    }
    cumulative
}

/// Apply discount factors, LP scaling, and noise pre-scaling to stage templates.
///
/// Returns a [`ScalingReport`] with pre/post coefficient ranges.
pub(crate) fn postprocess_templates(
    stage_templates: &mut StageTemplates,
    system: &System,
    state_layout: &StateSpace,
    cost_scale_factor: f64,
) -> ScalingReport {
    // The setter derives cumulative factors in the same call, so the two slices
    // cannot drift.
    {
        let pg = system.policy_graph();
        let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
        stage_templates.set_discount_factors(compute_per_stage_discount_factors(&study_stages, pg));
    }

    debug_assert_eq!(
        stage_templates.cumulative_discount_factors().len(),
        stage_templates.templates.len(),
        "cumulative_discount_factors must have length n_stages after postprocess"
    );

    // Discount theta before column/row scaling: cost scaling divides c_i by K but
    // leaves theta untouched, so the two must not be folded together.
    //
    // Use the per-stage `StageGeometry::theta_col` (= authoritative
    // `StageLayout::col_theta()`), NOT a re-derivation from `n_state`/`n_hydros`:
    // that hand arithmetic omits the anticipated ring's `anticipated_slots_out`/
    // `anticipated_state` blocks and, with anticipated thermals, lands on a
    // zero-cost `storage_in` column, silently disabling discounting (`0 * d = 0`).
    {
        let theta_cols: Vec<usize> = stage_templates
            .geometry_per_stage
            .iter()
            .map(|g| g.theta_col)
            .collect();
        let discount_factors = stage_templates.discount_factors().to_vec();
        debug_assert_eq!(
            theta_cols.len(),
            stage_templates.templates.len(),
            "geometry_per_stage must be populated and aligned with templates",
        );
        for (s_idx, tmpl) in stage_templates.templates.iter_mut().enumerate() {
            tmpl.objective[theta_cols[s_idx]] *= discount_factors[s_idx];
        }
    }

    // Column scaling then row scaling (D_r * A * D_c). Scale factors are stored on
    // the template for unscaling primal/dual solutions in the forward/backward passes.
    let mut stage_scaling_reports = Vec::with_capacity(stage_templates.templates.len());

    for (stage_id, tmpl) in stage_templates.templates.iter_mut().enumerate() {
        let pre_scaling = compute_coefficient_range(tmpl);

        let mut col_scale =
            lp_builder::compute_col_scale(tmpl.num_cols, &tmpl.col_starts, &tmpl.values);
        lp_builder::apply_commitment_hold_col_scale_unscale(&mut col_scale, state_layout);
        lp_builder::apply_col_scale(tmpl, &col_scale);
        tmpl.col_scale.clone_from(&col_scale);
        let row_scale = lp_builder::compute_row_scale(
            tmpl.num_rows,
            tmpl.num_cols,
            &tmpl.col_starts,
            &tmpl.row_indices,
            &tmpl.values,
        );
        lp_builder::apply_row_scale(tmpl, &row_scale);
        tmpl.row_scale.clone_from(&row_scale);

        let post_scaling = compute_coefficient_range(tmpl);

        stage_scaling_reports.push(StageScalingReport {
            stage_id,
            dimensions: LpDimensions {
                num_cols: tmpl.num_cols,
                num_rows: tmpl.num_rows,
                num_nz: tmpl.num_nz,
            },
            pre_scaling,
            post_scaling,
            col_scale: summarize_scale_factors(&col_scale),
            row_scale: summarize_scale_factors(&row_scale),
        });
    }

    let scaling_report = build_scaling_report(cost_scale_factor, stage_scaling_reports);

    // Pre-scale noise_scale by row_scale so the perturbation (noise_scale * eta)
    // shares the units of the already-row-scaled row bounds; otherwise
    // transform_inflow_noise produces a mixed-scale RHS (scaled base + unscaled perturbation).
    let n_hydros_noise = stage_templates.n_hydros;
    for (s_idx, tmpl) in stage_templates.templates.iter().enumerate() {
        if !tmpl.row_scale.is_empty() {
            let base_row = stage_templates.base_rows[s_idx];
            for h in 0..n_hydros_noise {
                stage_templates.noise_scale[s_idx * n_hydros_noise + h] *=
                    tmpl.row_scale[base_row + h];
            }
        }
    }

    scaling_report
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
mod tests {
    use super::{compute_cumulative_discount_factors, compute_per_stage_discount_factors};
    use chrono::NaiveDate;
    use cobre_core::HorizonGraph;
    use cobre_core::temporal::{
        BlockMode, NoiseMethod, PolicyGraphType, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use std::collections::HashMap;

    fn one_year_stage(id: i32) -> Stage {
        Stage {
            index: 0,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            season_id: None,
            blocks: vec![],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// A `stages[].annual_discount_rate_override` (carried on
    /// `HorizonGraph::stage_discount_rate_overrides`) sets that stage's rate,
    /// overriding the global `annual_discount_rate` (B2).
    #[test]
    fn stage_discount_override_is_read_off_the_stage() {
        let stage = one_year_stage(0);
        let days = f64::from((stage.end_date - stage.start_date).num_days() as i32);
        let mut overrides = HashMap::new();
        overrides.insert(0, 0.10);
        let pg = HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.06,
            transitions: vec![],
            nodes: vec![],
            stage_discount_rate_overrides: overrides,
            season_map: None,
        };

        let factors = compute_per_stage_discount_factors(&[&stage], &pg);
        let expected = 1.0 / (1.0_f64 + 0.10).powf(days / 365.25);
        assert!(
            (factors[0] - expected).abs() < 1e-12,
            "stage override 0.10 must set the rate, got {}",
            factors[0]
        );
        let global = 1.0 / (1.0_f64 + 0.06).powf(days / 365.25);
        assert!(
            (factors[0] - global).abs() > 1e-6,
            "override must differ from the global-rate factor"
        );
    }

    /// A stage with no override falls back to the global `annual_discount_rate`.
    #[test]
    fn stage_without_override_uses_global_rate() {
        let stage = one_year_stage(0);
        let days = f64::from((stage.end_date - stage.start_date).num_days() as i32);
        let pg = HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.06,
            transitions: vec![],
            nodes: vec![],
            stage_discount_rate_overrides: HashMap::new(),
            season_map: None,
        };

        let factors = compute_per_stage_discount_factors(&[&stage], &pg);
        let expected = 1.0 / (1.0_f64 + 0.06).powf(days / 365.25);
        assert!(
            (factors[0] - expected).abs() < 1e-12,
            "absent override must fall back to the global rate, got {}",
            factors[0]
        );
    }

    #[test]
    fn cumulative_discount_factors_length_matches_n_stages() {
        let n_stages = 4_usize;
        let per_stage = vec![0.95_f64; n_stages];
        let cumulative = compute_cumulative_discount_factors(&per_stage);

        assert_eq!(
            cumulative.len(),
            n_stages,
            "cumulative_discount_factors length must equal n_stages = {n_stages}"
        );

        assert_eq!(cumulative[0], 1.0, "cumulative[0] == 1.0 (present value)");
        assert_eq!(
            cumulative[1], 0.95,
            "cumulative[1] == 0.95 = 1.0 * per_stage[0]"
        );
        // Approximate: repeated multiplication may differ from powi(3) by a ULP
        // (floating-point associativity).
        assert!(
            (cumulative[n_stages - 1] - 0.95_f64.powi(3)).abs() < 1e-15,
            "cumulative[n_stages-1] must be within 1e-15 of 0.95^(n_stages-1) (got {})",
            cumulative[n_stages - 1]
        );
    }

    #[test]
    fn cumulative_discount_factors_all_ones_when_rate_zero() {
        let per_stage = vec![1.0_f64; 3];
        let cumulative = compute_cumulative_discount_factors(&per_stage);
        assert_eq!(cumulative.len(), 3);
        for (i, &v) in cumulative.iter().enumerate() {
            assert_eq!(
                v, 1.0,
                "cumulative[{i}] must be 1.0 when per-stage factor is 1.0"
            );
        }
    }
}
