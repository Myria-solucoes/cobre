//! Shared `#[cfg(test)]` fixtures for the split builder representation modules.

use chrono::NaiveDate;
use cobre_core::{
    Block, BlockMode, HydroPenalties, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
    StageStateConfig,
};

use crate::indexer::StateLayout;

use super::layout::TemplateBuildCtx;

/// Build the canonical role-(a) [`StateLayout`] a `StageLayout` borrows, from a
/// test [`TemplateBuildCtx`].
///
/// Mirrors `template.rs::build_stage_templates` — same state dimensions and
/// PAR-derived effective lag counts — so the handle a test passes to
/// `StageLayout::new` is byte-identical to production's.
pub(super) fn state_layout_for(ctx: &TemplateBuildCtx<'_>) -> StateLayout {
    let effective_lag_counts: Vec<usize> = if ctx.max_par_order > 0 {
        (0..ctx.n_hydros)
            .map(|h| {
                if h < ctx.par_lp.n_hydros() {
                    ctx.par_lp.effective_lag_count(h)
                } else {
                    ctx.max_par_order
                }
            })
            .collect()
    } else {
        vec![0; ctx.n_hydros]
    };
    StateLayout::new(
        ctx.n_hydros,
        ctx.max_par_order,
        0,
        Vec::new(),
        ctx.n_anticipated,
        ctx.k_max,
        ctx.anticipated_lead_stages.clone(),
        &effective_lag_counts,
    )
}

/// All-zero `HydroPenalties` so no fixture-side penalty cost contaminates the
/// column/objective assertions in the builder tests.
pub(super) fn zero_hydro_penalties() -> HydroPenalties {
    HydroPenalties {
        spillage_cost: 0.0,
        diversion_cost: 0.0,
        turbined_cost: 0.0,
        storage_violation_below_cost: 0.0,
        filling_target_violation_cost: 0.0,
        turbined_violation_below_cost: 0.0,
        outflow_violation_below_cost: 0.0,
        outflow_violation_above_cost: 0.0,
        generation_violation_below_cost: 0.0,
        evaporation_violation_cost: 0.0,
        water_withdrawal_violation_cost: 0.0,
        water_withdrawal_violation_pos_cost: 0.0,
        water_withdrawal_violation_neg_cost: 0.0,
        evaporation_violation_pos_cost: 0.0,
        evaporation_violation_neg_cost: 0.0,
        inflow_nonnegativity_cost: 0.0,
    }
}

/// Two-block `Stage` at `index` with per-block `durations`.
///
/// The two durations are passed by the caller rather than fixed because several
/// callers deliberately use unequal blocks (e.g. `[300.0, 444.0]`): equal
/// durations would mask a per-block divisor confusion in the code under test.
pub(super) fn two_block_stage(index: usize, durations: [f64; 2]) -> Stage {
    Stage {
        index,
        id: index as i32,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks: vec![
            Block {
                index: 0,
                name: "BLK0".to_string(),
                duration_hours: durations[0],
            },
            Block {
                index: 1,
                name: "BLK1".to_string(),
                duration_hours: durations[1],
            },
        ],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: false,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    }
}
