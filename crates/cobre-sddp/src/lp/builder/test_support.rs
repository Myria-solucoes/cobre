//! Shared `#[cfg(test)]` fixtures for the split builder representation modules.
//!
//! `zero_hydro_penalties` and `two_block_stage` live here once; every user
//! imports them by explicit name via `use super::super::test_support::…` (the
//! test modules nest one level below this module) — never `use super::*`.

use chrono::NaiveDate;
use cobre_core::{
    Block, BlockMode, HydroPenalties, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
    StageStateConfig,
};

/// All-zero hydro penalties: every one of the 16 `HydroPenalties` cost fields is
/// `0.0`, so no fixture-side penalty cost contaminates the column/objective
/// assertions in the builder tests.
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
