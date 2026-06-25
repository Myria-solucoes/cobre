//! Shared parity-hash computation for the integration test harnesses — the sole
//! owner of the hash whitelist and its byte layout.
//!
//! The hash is deterministic: every field is little-endian, iterations and stages
//! ascending, scenarios sorted by `scenario_id`, hydro/thermal records by
//! `(block_id, id)`. Fields 5–7 are not redundant with storage/dual: each is an
//! `n_blks`-dependent read kept specifically to surface an extraction/cost/base
//! bug a uniform-block case cannot detect — do not drop them as duplicate.
//!
//! ## Hash whitelist (in fixed order)
//!
//! 1. Per-iteration convergence: `iteration_u64_le || lower_bound_f64_le
//!    || upper_bound_f64_le || upper_bound_std_f64_le || gap_f64_le`
//! 2. Per-stage, per-cut: `stage_u32_le || intercept_f64_le ||
//!    coefficient_count_u32_le || coefficient_f64_le[]`
//! 3. Primal trajectory (`storage_final_hm3`) per scenario per stage.
//! 4. Dual trajectory (`water_value_per_hm3`) per scenario per stage.
//! 5. Per-block equipment (`spillage_m3s`) — base shifts off stage 0's block
//!    width under a non-uniform schedule (the simulation-extraction base bug).
//! 6. Cost breakdown (`spillage_cost`) — a `range_sum` whose base AND length
//!    shift under a non-uniform schedule (the cost-breakdown bug).
//! 7. Anticipated decision (`anticipated_decision_mw`) — base is the per-stage
//!    `thermal.end` (`n_blks`-dependent). `None` hashes as a 0-flag, so only
//!    anticipated cases (D34) move this field.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    dead_code
)]

use cobre_sddp::{SimulationScenarioResult, StudySetup};
use sha2::{Digest, Sha256};

/// Compute the SHA-256 parity hash over the module-doc whitelist.
pub fn compute_parity_hash(
    convergence_updates: &[(u64, f64, f64, f64, f64)],
    setup: &StudySetup,
    mut scenario_results: Vec<SimulationScenarioResult>,
) -> String {
    let mut hasher = Sha256::new();

    for &(iteration, lb, ub, ub_std, gap) in convergence_updates {
        hasher.update(iteration.to_le_bytes());
        hasher.update(lb.to_le_bytes());
        hasher.update(ub.to_le_bytes());
        hasher.update(ub_std.to_le_bytes());
        hasher.update(gap.to_le_bytes());
    }

    // Active cuts in ascending stage order, then active_cuts() slot order — fixed
    // iteration order is what makes the cut digest declaration-order-stable.
    let fcf = &setup.fcf;
    let num_stages = fcf.pools.len();
    for stage in 0..num_stages {
        for (_slot, intercept, coefficients) in fcf.active_cuts(stage) {
            hasher.update((stage as u32).to_le_bytes());
            hasher.update(intercept.to_le_bytes());
            hasher.update((coefficients.len() as u32).to_le_bytes());
            for &c in coefficients {
                hasher.update(c.to_le_bytes());
            }
        }
    }

    // Sort into canonical order so the digest is independent of how the pipeline
    // emitted scenarios/stages/records.
    scenario_results.sort_by_key(|r| r.scenario_id);

    for scenario in &mut scenario_results {
        scenario.stages.sort_by_key(|s| s.stage_id);

        for stage in &mut scenario.stages {
            stage.hydros.sort_by_key(|h| (h.block_id, h.hydro_id));

            let num_primals = stage.hydros.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_primals.to_le_bytes());
            for h in &stage.hydros {
                hasher.update(h.storage_final_hm3.to_le_bytes());
            }

            let num_duals = stage.hydros.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_duals.to_le_bytes());
            for h in &stage.hydros {
                hasher.update(h.water_value_per_hm3.to_le_bytes());
            }

            let num_equipment = stage.hydros.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_equipment.to_le_bytes());
            for h in &stage.hydros {
                hasher.update(h.spillage_m3s.to_le_bytes());
            }

            let num_costs = stage.costs.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_costs.to_le_bytes());
            for c in &stage.costs {
                hasher.update(c.spillage_cost.to_le_bytes());
            }

            stage.thermals.sort_by_key(|t| (t.block_id, t.thermal_id));
            let num_thermals = stage.thermals.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_thermals.to_le_bytes());
            for t in &stage.thermals {
                // Hash a presence flag + value so `None` maps to a fixed (0, 0.0):
                // dropping the flag would collide `None` with `Some(0.0)` and break
                // the encoding's injectivity.
                let (flag, value) = t.anticipated_decision_mw.map_or((0u8, 0.0), |v| (1u8, v));
                hasher.update(flag.to_le_bytes());
                hasher.update(value.to_le_bytes());
            }
        }
    }

    format!("{:x}", hasher.finalize())
}
