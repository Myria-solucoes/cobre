//! Shared parity-hash computation for integration test harnesses.
//!
//! The hash is deterministic: every field is encoded as little-endian bytes,
//! the iteration order is ascending, stages are ascending within cuts, and
//! scenarios are sorted ascending by `scenario_id`.
//!
//! ## Hash whitelist (in fixed order)
//!
//! 1. Per-iteration convergence data: `iteration_u64_le || lower_bound_f64_le
//!    || upper_bound_f64_le || upper_bound_std_f64_le || gap_f64_le`
//! 2. Per-stage, per-cut: `stage_u32_le || intercept_f64_le ||
//!    coefficient_count_u32_le || coefficient_f64_le[]`
//! 3. Simulation primal trajectory per scenario per stage.
//! 4. Simulation dual trajectory per scenario per stage.
//! 5. Per-block equipment trajectory (`spillage_m3s`) per scenario per stage —
//!    an `n_blks`-dependent column read whose base shifts under a non-uniform
//!    block schedule, so it makes the simulation-extraction base bug visible.
//! 6. Cost-breakdown trajectory (`spillage_cost`) per scenario per stage — a
//!    `range_sum` over an `n_blks`-dependent range whose base AND length shift
//!    under a non-uniform schedule, so it makes the cost-breakdown bug visible.
//! 7. Anticipated-decision trajectory (`anticipated_decision_mw`) per scenario
//!    per stage — a column read whose base is the per-stage `thermal.end`
//!    (`n_blks`-dependent), so a stage-0-based read reports the wrong MW at a
//!    non-uniform stage. `None` hashes as a 0-flag, so only anticipated cases
//!    (D34) move this field.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    dead_code
)]

use cobre_sddp::{SimulationScenarioResult, StudySetup};
use sha2::{Digest, Sha256};

/// Compute a SHA-256 parity hash over the semantic whitelist.
///
/// The hash is deterministic: every field is encoded as little-endian bytes,
/// the iteration order is ascending, stages are ascending within cuts, and
/// scenarios are sorted ascending by `scenario_id`.
///
/// # Field translation
///
/// - Per-iteration convergence data comes from the collected
///   `ConvergenceUpdate` events (timing-free variant of `IterationSummary`).
/// - Cut data comes from `setup.fcf().active_cuts(stage)`.
/// - Primal trajectory uses `SimulationHydroResult::storage_final_hm3`.
/// - Dual trajectory uses `SimulationHydroResult::water_value_per_hm3`.
///   Both are sorted by `(block_id, hydro_id)` within each stage.
/// - Per-block equipment trajectory uses `SimulationHydroResult::spillage_m3s`,
///   sorted by `(block_id, hydro_id)` — a per-block column whose base shifts off
///   stage 0's block width under a non-uniform schedule.
/// - Cost-breakdown trajectory uses `SimulationCostResult::spillage_cost` from
///   `SimulationStageResult::costs` — a `range_sum` whose base AND length shift
///   under a non-uniform schedule.
/// - Anticipated-decision trajectory uses
///   `SimulationThermalResult::anticipated_decision_mw`, sorted by
///   `(block_id, thermal_id)` — a column whose base is the per-stage
///   `thermal.end` (`n_blks`-dependent). `None` hashes as a 0-flag.
pub fn compute_parity_hash(
    convergence_updates: &[(u64, f64, f64, f64, f64)],
    setup: &StudySetup,
    mut scenario_results: Vec<SimulationScenarioResult>,
) -> String {
    let mut hasher = Sha256::new();

    // ------------------------------------------------------------------
    // Section 1: Per-iteration convergence data
    // ------------------------------------------------------------------
    for &(iteration, lb, ub, ub_std, gap) in convergence_updates {
        hasher.update(iteration.to_le_bytes());
        hasher.update(lb.to_le_bytes());
        hasher.update(ub.to_le_bytes());
        hasher.update(ub_std.to_le_bytes());
        hasher.update(gap.to_le_bytes());
    }

    // ------------------------------------------------------------------
    // Section 2: Active cuts per stage (ascending stage order, then slot
    //            order within each stage as reported by active_cuts())
    // ------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // Section 3 & 4: Simulation primal and dual trajectories
    //
    // Sort scenarios ascending by scenario_id for determinism.
    // ------------------------------------------------------------------
    scenario_results.sort_by_key(|r| r.scenario_id);

    for scenario in &mut scenario_results {
        // Sort stages ascending by stage_id (pipeline already stage-ordered,
        // but sort defensively). `SimulationStageResult` does not derive Clone,
        // so we sort in-place using the owned Vec.
        scenario.stages.sort_by_key(|s| s.stage_id);

        for stage in &mut scenario.stages {
            // Sort hydro records by (block_id, hydro_id) for determinism.
            // `SimulationHydroResult` does not derive Clone; sort in-place.
            stage.hydros.sort_by_key(|h| (h.block_id, h.hydro_id));

            // Primal trajectory: storage_final_hm3 per hydro record.
            let num_primals = stage.hydros.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_primals.to_le_bytes());
            for h in &stage.hydros {
                hasher.update(h.storage_final_hm3.to_le_bytes());
            }

            // Dual trajectory: water_value_per_hm3 per hydro record.
            let num_duals = stage.hydros.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_duals.to_le_bytes());
            for h in &stage.hydros {
                hasher.update(h.water_value_per_hm3.to_le_bytes());
            }

            // Per-block equipment trajectory: spillage_m3s per hydro record
            // (same (block_id, hydro_id) order). A per-block column read whose
            // base shifts off stage 0's block width under a non-uniform schedule,
            // so it surfaces the extraction base bug the cut/storage fields miss.
            let num_equipment = stage.hydros.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_equipment.to_le_bytes());
            for h in &stage.hydros {
                hasher.update(h.spillage_m3s.to_le_bytes());
            }

            // Cost-breakdown trajectory: spillage_cost per cost record. Sums an
            // n_blks-dependent range, so a wrong base/length at a non-uniform
            // stage misbooks it — the second defect sub-class the storage/dual
            // fields cannot detect.
            let num_costs = stage.costs.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_costs.to_le_bytes());
            for c in &stage.costs {
                hasher.update(c.spillage_cost.to_le_bytes());
            }

            // Anticipated-decision trajectory: anticipated_decision_mw per thermal
            // record, sorted by (block_id, thermal_id) for determinism. The
            // anticipated-decision column base is the per-stage `thermal.end`
            // (n_blks-dependent), so a stage-0-based read reports the WRONG MW at a
            // non-uniform stage. `None` (non-anticipated / inactive) hashes as a
            // presence flag of 0; `Some(v)` as flag 1 followed by the value — so
            // pure-thermal and non-anticipated cases contribute a fixed all-zero
            // stream and only anticipated cases (D34) move this field.
            stage.thermals.sort_by_key(|t| (t.block_id, t.thermal_id));
            let num_thermals = stage.thermals.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_thermals.to_le_bytes());
            for t in &stage.thermals {
                // Presence flag + value, so `None` (non-anticipated / inactive)
                // hashes as a fixed (0, 0.0) pair and only `Some` cases move the
                // field; keeps the encoding injective and declaration-order-stable.
                let (flag, value) = t.anticipated_decision_mw.map_or((0u8, 0.0), |v| (1u8, v));
                hasher.update(flag.to_le_bytes());
                hasher.update(value.to_le_bytes());
            }
        }
    }

    format!("{:x}", hasher.finalize())
}
