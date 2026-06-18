//! Per-stage anticipated-thermal iterators and predicates for [`StageIndexer`].
//!
//! Owns the stage-level gating of the anticipated-decision columns and the
//! anticipated-fishing rows: which plants are active at a given stage and what
//! LP column/row each maps to.

use super::layout::StageIndexer;

impl StageIndexer {
    /// Iterator over `(local_idx, lp_column)` for anticipated decisions active
    /// at `stage_idx`.
    ///
    /// A plant is active iff
    /// `stage_idx + anticipated_lead_stages[local_idx] < n_stages`. The boundary
    /// case `stage_idx + K_i == n_stages` is **excluded**: the commitment would
    /// mature at a delivery stage outside the study horizon `[0, n_stages)`, so
    /// no delivery LP is ever built for it. Inactive plants are skipped; the LP
    /// build applies `[0, 0]` bounds to their columns so the presolver
    /// eliminates them.
    ///
    /// All arithmetic uses `usize`; the upstream conversion from `u32 lead_stages`
    /// to `usize` happens when [`EquipmentCounts::anticipated_lead_stages`] is
    /// populated.
    ///
    /// [`EquipmentCounts::anticipated_lead_stages`]: super::layout::EquipmentCounts::anticipated_lead_stages
    pub fn anticipated_decision_active_at_stage(
        &self,
        stage_idx: usize,
        n_stages: usize,
    ) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.anticipated_lead_stages
            .iter()
            .enumerate()
            .filter_map(move |(i, &k_i)| {
                if stage_idx.saturating_add(k_i) < n_stages {
                    Some((i, self.anticipated_decision.start + i))
                } else {
                    None
                }
            })
    }

    /// Iterator over `(local_idx, lp_row)` for anticipated fishing
    /// constraints active at `stage_idx`. Active iff the plant exists
    /// (always-active predicate). Returns one row per anticipated plant in
    /// ascending `local_idx` order.
    ///
    /// Rows are assigned in ascending `local_idx` order: plant `i` gets row
    /// `anticipated_fishing_start + i`.
    ///
    /// The dual on this row carries the cut subgradient w.r.t. the
    /// currently-delivering anticipated-state slot (slot 0 after the ring-buffer
    /// shift) during backward-pass cut extraction.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `stage_idx > n_stages`.
    pub fn anticipated_fishing_active_at_stage(
        &self,
        stage_idx: usize,
        n_stages: usize,
    ) -> impl Iterator<Item = (usize, usize)> + '_ {
        debug_assert!(
            stage_idx <= n_stages,
            "stage_idx {stage_idx} exceeds n_stages {n_stages}"
        );
        self.anticipated_lead_stages
            .iter()
            .enumerate()
            .map(move |(local_idx, _)| (local_idx, self.anticipated_fishing_start + local_idx))
    }

    /// Return `true` when the anticipated-fishing equality is active for plant
    /// `local_idx` at `stage_idx`. Always returns `true` for valid inputs
    /// under the always-active rule.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `stage_idx > n_stages` or `local_idx >= self.n_anticipated()`.
    #[inline]
    #[must_use]
    pub fn is_anticipated_fishing_active(
        &self,
        local_idx: usize,
        stage_idx: usize,
        n_stages: usize,
    ) -> bool {
        debug_assert!(
            stage_idx <= n_stages,
            "stage_idx {stage_idx} must be <= n_stages {n_stages}",
        );
        debug_assert!(
            local_idx < self.anticipated_lead_stages.len(),
            "local_idx {local_idx} out of bounds (n_anticipated = {})",
            self.anticipated_lead_stages.len(),
        );
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::indexer::{EquipmentCounts, EvapConfig, FphaColumnLayout, StageIndexer};

    fn fpha(hydro_indices: Vec<usize>, planes_per_hydro: Vec<usize>) -> FphaColumnLayout {
        FphaColumnLayout {
            hydro_indices,
            planes_per_hydro,
        }
    }

    fn evap(hydro_indices: Vec<usize>) -> EvapConfig {
        EvapConfig { hydro_indices }
    }

    fn eq(
        hydro_count: usize,
        max_par_order: usize,
        n_thermals: usize,
        n_lines: usize,
        n_buses: usize,
        n_blks: usize,
        has_inflow_penalty: bool,
    ) -> EquipmentCounts {
        EquipmentCounts {
            hydro_count,
            max_par_order,
            n_thermals,
            n_lines,
            n_buses,
            n_blks,
            has_inflow_penalty,
            max_deficit_segments: 1,
            n_anticipated: 0,
            k_max: 0,
            anticipated_lead_stages: vec![],
            anticipated_thermal_indices: vec![],
            n_pumping: 0,
        }
    }

    /// Strict-predicate acceptance: `stage_idx + K_i < n_stages` accepts.
    /// `K_i = 3`, `stage_idx = 2`, `n_stages = 6` → `delivery_stage` = 5 < 6 → active.
    #[test]
    fn anticipated_decision_active_acceptance_strict_interior() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 1,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 1,
                k_max: 3,
                anticipated_lead_stages: vec![3],
                anticipated_thermal_indices: vec![0],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let active: Vec<_> = idx.anticipated_decision_active_at_stage(2, 6).collect();
        assert_eq!(active, vec![(0, idx.anticipated_decision.start)]);
    }

    /// Strict-predicate boundary rejection: `stage_idx + K_i == n_stages`
    /// is REJECTED. `K_i = 3`, `stage_idx = 3`, `n_stages = 6` → `delivery_stage` = 6
    /// would fall outside the study horizon `[0, 6)`. The strict predicate
    /// `stage_idx + K_i < n_stages` excludes this case so the LP never builds
    /// a priced-but-not-delivered commitment.
    #[test]
    fn anticipated_decision_active_rejection_at_n_stages_boundary() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 1,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 1,
                k_max: 3,
                anticipated_lead_stages: vec![3],
                anticipated_thermal_indices: vec![0],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let active: Vec<_> = idx.anticipated_decision_active_at_stage(3, 6).collect();
        assert!(
            active.is_empty(),
            "stage_idx + K_i == n_stages must be excluded under the strict predicate"
        );
    }

    /// Strict-predicate one-past-boundary rejection: `stage_idx + K_i == n_stages + 1`
    /// also rejects (the inclusive-predicate "one-past-boundary" case).
    /// `K_i = 3`, `stage_idx = 4`, `n_stages = 6` → plant is NOT active.
    #[test]
    fn anticipated_decision_active_rejection_one_past_boundary() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 1,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 1,
                k_max: 3,
                anticipated_lead_stages: vec![3],
                anticipated_thermal_indices: vec![0],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let active: Vec<_> = idx.anticipated_decision_active_at_stage(4, 6).collect();
        assert!(active.is_empty());
    }

    /// At `stage_idx == 0`, every plant with `K_i < n_stages` is active under
    /// the strict predicate. Plants with `K_i == n_stages` are excluded
    /// (delivery would land at `n_stages`, outside the study horizon).
    #[test]
    fn anticipated_decision_active_all_at_stage_zero() {
        // K = [1, 3, 6], n_stages = 6 → plants 0 (K=1) and 1 (K=3) accept
        // (0+1=1 < 6, 0+3=3 < 6). Plant 2 (K=6) rejects (0+6=6 NOT < 6).
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 6,
                anticipated_lead_stages: vec![1, 3, 6],
                anticipated_thermal_indices: vec![0, 1, 2],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_decision.start;
        let active: Vec<_> = idx.anticipated_decision_active_at_stage(0, 6).collect();
        assert_eq!(active, vec![(0, start), (1, start + 1)]);
    }

    /// When `anticipated_lead_stages` is empty, the iterator yields nothing
    /// for any `(stage_idx, n_stages)` pair.
    #[test]
    fn anticipated_decision_active_empty_iterator_when_no_plants() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(1, 0, 0, 0, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        for (s, t) in [(0_usize, 0_usize), (0, 12), (5, 12), (12, 12), (20, 12)] {
            let active: Vec<_> = idx.anticipated_decision_active_at_stage(s, t).collect();
            assert!(
                active.is_empty(),
                "expected empty iterator at (stage_idx={s}, n_stages={t})"
            );
        }
    }

    /// AC example `K_i = [3, 2, 5]` with `n_stages = 6`: verify each
    /// `stage_idx` selects the expected subset under the strict predicate
    /// `stage_idx + K_i < n_stages`.
    #[test]
    fn anticipated_decision_active_mixed_k_values() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 5,
                anticipated_lead_stages: vec![3, 2, 5],
                anticipated_thermal_indices: vec![0, 1, 2],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_decision.start;

        // stage_idx = 3: plant 0 rejects (3+3=6 NOT < 6), plant 1 accepts
        // (3+2=5 < 6), plant 2 rejects (3+5=8 NOT < 6).
        let active3: Vec<_> = idx.anticipated_decision_active_at_stage(3, 6).collect();
        assert_eq!(active3, vec![(1, start + 1)]);

        // stage_idx = 1: plant 0 accepts (1+3=4 < 6), plant 1 accepts
        // (1+2=3 < 6), plant 2 rejects (1+5=6 NOT < 6).
        let active1: Vec<_> = idx.anticipated_decision_active_at_stage(1, 6).collect();
        assert_eq!(
            active1,
            vec![(0, start), (1, start + 1)],
            "at stage_idx=1 under strict predicate: plants 0,1 active (1+3=4 < 6, 1+2=3 < 6); plant 2 excluded (1+5=6 NOT < 6)"
        );

        // stage_idx = 0: all three plants accept (0+3=3 < 6, 0+2=2 < 6,
        // 0+5=5 < 6).
        let active0: Vec<_> = idx.anticipated_decision_active_at_stage(0, 6).collect();
        assert_eq!(
            active0,
            vec![(0, start), (1, start + 1), (2, start + 2)],
            "at stage_idx=0 under strict predicate: all plants active (0+K < 6 for K in {{3,2,5}})"
        );

        // stage_idx = 4: all plants reject under strict predicate
        // (4+3=7, 4+2=6, 4+5=9; none < 6).
        let active4: Vec<_> = idx.anticipated_decision_active_at_stage(4, 6).collect();
        assert!(
            active4.is_empty(),
            "at stage_idx=4 under strict predicate, no plant satisfies stage_idx + K < 6"
        );
    }

    /// At horizon tail `stage_idx == n_stages` and every plant has `K_i > 0`,
    /// no plant should be active (`stage_idx + K_i > n_stages`).
    #[test]
    fn anticipated_decision_active_no_plants_at_horizon_tail() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 5,
                anticipated_lead_stages: vec![1, 3, 5],
                anticipated_thermal_indices: vec![0, 1, 2],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let active: Vec<_> = idx.anticipated_decision_active_at_stage(6, 6).collect();
        assert!(active.is_empty());
    }

    /// At `stage_idx == 0` all anticipated plants are active (always-active
    /// predicate): the iterator yields one entry per plant.
    #[test]
    fn anticipated_fishing_active_at_stage_zero() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 3,
                anticipated_lead_stages: vec![1, 2, 3],
                anticipated_thermal_indices: vec![0, 1, 2],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_fishing_start;
        let active: Vec<_> = idx.anticipated_fishing_active_at_stage(0, 5).collect();
        assert_eq!(active, vec![(0, start), (1, start + 1), (2, start + 2)]);
    }

    /// Always-active: `stage_idx == 0` with `K = [1, 2, 3]` yields the full
    /// set `[(0, start), (1, start+1), (2, start+2)]`.
    #[test]
    fn anticipated_fishing_active_always_active_stage_zero() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 3,
                anticipated_lead_stages: vec![1, 2, 3],
                anticipated_thermal_indices: vec![0, 1, 2],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_fishing_start;
        let active: Vec<_> = idx.anticipated_fishing_active_at_stage(0, 5).collect();
        assert_eq!(active, vec![(0, start), (1, start + 1), (2, start + 2)]);
    }

    /// Always-active: at any `stage_idx`, all plants are returned regardless of
    /// their `K_i` value.
    #[test]
    fn anticipated_fishing_active_acceptance_boundary() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 3,
                anticipated_lead_stages: vec![1, 2, 3],
                anticipated_thermal_indices: vec![0, 1, 2],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_fishing_start;
        let active: Vec<_> = idx.anticipated_fishing_active_at_stage(1, 5).collect();
        // All three plants are active (always-active predicate).
        assert_eq!(active, vec![(0, start), (1, start + 1), (2, start + 2)]);
    }

    /// Always-active: `K_i = 5`, `stage_idx = 4` — plant is still active since
    /// the always-active predicate ignores `K_i`.
    #[test]
    fn anticipated_fishing_active_rejection_boundary() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 1,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 1,
                k_max: 5,
                anticipated_lead_stages: vec![5],
                anticipated_thermal_indices: vec![0],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_fishing_start;
        let active: Vec<_> = idx.anticipated_fishing_active_at_stage(4, 6).collect();
        // Always-active predicate: plant is returned regardless of K_i vs stage_idx.
        assert_eq!(active, vec![(0, start)]);
    }

    /// All plants active at `stage_idx == 3` when `K_i = [1, 2, 3]`. Rows are
    /// assigned ascending: `(0, start+0), (1, start+1), (2, start+2)`.
    #[test]
    fn anticipated_fishing_active_all_plants() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 3,
                anticipated_lead_stages: vec![1, 2, 3],
                anticipated_thermal_indices: vec![0, 1, 2],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_fishing_start;
        let active: Vec<_> = idx.anticipated_fishing_active_at_stage(3, 5).collect();
        assert_eq!(active, vec![(0, start), (1, start + 1), (2, start + 2)]);
    }

    /// When `n_anticipated == 0`, the iterator yields nothing for any
    /// `(stage_idx, n_stages)` pair.
    #[test]
    fn anticipated_fishing_active_no_plants() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &eq(1, 0, 0, 0, 1, 1, false),
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        for (s, t) in [(0_usize, 0_usize), (0, 5), (3, 5), (5, 5)] {
            let active: Vec<_> = idx.anticipated_fishing_active_at_stage(s, t).collect();
            assert!(
                active.is_empty(),
                "expected empty iterator at (stage_idx={s}, n_stages={t})"
            );
        }
    }

    /// Ascending `local_idx` order is preserved even when ties cause the
    /// active subset to be the full set: `K_i = [2, 2, 2]`, `stage_idx = 2`
    /// yields `(0, start+0), (1, start+1), (2, start+2)`.
    #[test]
    fn anticipated_fishing_active_preserves_local_idx_order() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 2,
                anticipated_lead_stages: vec![2, 2, 2],
                anticipated_thermal_indices: vec![0, 1, 2],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_fishing_start;
        let active: Vec<_> = idx.anticipated_fishing_active_at_stage(2, 4).collect();
        assert_eq!(active, vec![(0, start), (1, start + 1), (2, start + 2)]);
    }

    /// Successive yields have strictly increasing `lp_row` values. Under
    /// always-active, `local_idx == active_pos` so row index equals
    /// `start + local_idx` for every plant.
    #[test]
    fn anticipated_fishing_active_row_indices_monotonic() {
        // K_i = [1, 4, 2, 3] with stage_idx = 3 → all 4 plants active.
        // Row indices: start+0, start+1, start+2, start+3 (contiguous).
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 4,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 4,
                k_max: 4,
                anticipated_lead_stages: vec![1, 4, 2, 3],
                anticipated_thermal_indices: vec![0, 1, 2, 3],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_fishing_start;
        let active: Vec<_> = idx.anticipated_fishing_active_at_stage(3, 5).collect();
        assert_eq!(
            active,
            vec![(0, start), (1, start + 1), (2, start + 2), (3, start + 3)]
        );
        // Row indices are strictly monotonic.
        let rows: Vec<_> = active.iter().map(|(_, row)| *row).collect();
        assert!(rows.windows(2).all(|w| w[0] < w[1]));
    }

    /// `stage_idx == n_stages` is an acceptance boundary (delivery happens at
    /// the horizon end and the LP at stage `T` still solves).
    #[test]
    fn anticipated_fishing_active_at_n_stages_boundary() {
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 1,
                max_par_order: 0,
                n_thermals: 2,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 2,
                k_max: 5,
                anticipated_lead_stages: vec![3, 5],
                anticipated_thermal_indices: vec![0, 1],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let start = idx.anticipated_fishing_start;
        let active: Vec<_> = idx.anticipated_fishing_active_at_stage(5, 5).collect();
        // Both plants active (always-active predicate; `stage_idx == n_stages` accepted by debug guard).
        assert_eq!(active, vec![(0, start), (1, start + 1)]);
    }

    /// `is_anticipated_fishing_active` returns `true` for every valid
    /// `(local_idx, stage_idx, n_stages)` triple across a parameter sweep.
    ///
    /// Fixtures: `n_anticipated ∈ {0, 1, 3}`, `anticipated_lead_stages`
    /// covering `k_max ∈ {0, 1, 5}`, `n_stages = 5`, `stage_idx ∈ 0..5`,
    /// `local_idx ∈ 0..n_anticipated`.
    #[test]
    fn is_anticipated_fishing_active_returns_true_everywhere() {
        // Fixture 1: n_anticipated = 0 (no plants → no iterations, vacuously ok)
        let idx0 = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 0,
                max_par_order: 0,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        let n_stages = 5;
        // n_anticipated=0: no plants, so the predicate is vacuously active
        // for all (local_idx, stage_idx) pairs — nothing to iterate over.
        let _ = (&idx0, n_stages);

        // Fixture 2: n_anticipated = 1, k_max = 1 (k_max >= 1 required when n_anticipated > 0)
        let idx1 = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 0,
                max_par_order: 0,
                n_thermals: 1,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 1,
                k_max: 1,
                anticipated_lead_stages: vec![1],
                anticipated_thermal_indices: vec![0],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        for stage_idx in 0..n_stages {
            assert!(
                idx1.is_anticipated_fishing_active(0, stage_idx, n_stages),
                "fixture n_anticipated=1 k_max=1: expected true at (0, {stage_idx})"
            );
        }

        // Fixture 3: n_anticipated = 3, k_max = 5 (covers k_max = 5)
        let idx3 = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 0,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 5,
                anticipated_lead_stages: vec![1, 3, 5],
                anticipated_thermal_indices: vec![0, 1, 2],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );
        for stage_idx in 0..n_stages {
            for local_idx in 0..3_usize {
                assert!(
                    idx3.is_anticipated_fishing_active(local_idx, stage_idx, n_stages),
                    "fixture n_anticipated=3: expected true at ({local_idx}, {stage_idx})"
                );
            }
        }
    }

    /// Lock-step parity: `is_anticipated_fishing_active(p, s, n_stages)` matches
    /// membership of `p` in the iterator `anticipated_fishing_active_at_stage(s, n_stages)`
    /// for every `(p, s)` cell in `[0, n_anticipated) × [0, n_stages)`.
    ///
    /// Fixture: `n_anticipated = 3`, `K = [1, 2, 3]`, `n_stages = 5`.
    #[test]
    fn fishing_predicate_lockstep_with_iterator() {
        use std::collections::HashSet;

        let n_stages: usize = 5;
        let idx = StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 0,
                max_par_order: 0,
                n_thermals: 3,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 3,
                k_max: 3,
                anticipated_lead_stages: vec![1, 2, 3],
                anticipated_thermal_indices: vec![0, 1, 2],
                n_pumping: 0,
            },
            &fpha(vec![], vec![]),
            &evap(vec![]),
        );

        for stage_idx in 0..n_stages {
            let iter_set: HashSet<usize> = idx
                .anticipated_fishing_active_at_stage(stage_idx, n_stages)
                .map(|(local, _)| local)
                .collect();
            for local_idx in 0..3_usize {
                let predicate_result =
                    idx.is_anticipated_fishing_active(local_idx, stage_idx, n_stages);
                let iter_result = iter_set.contains(&local_idx);
                assert_eq!(
                    predicate_result, iter_result,
                    "lock-step parity failed at (local={local_idx}, stage={stage_idx}): \
                     is_anticipated_fishing_active={predicate_result}, \
                     iterator membership={iter_result}"
                );
            }
        }
    }

    /// Structural-invariant sweep tests for the anticipated-thermal indexer
    /// extensions.
    ///
    /// Each test in this sub-module iterates the same hand-written parameter
    /// grid (see [`parameter_grid`]) and asserts a single structural invariant
    /// (I1-I13) on every grid point. The grid is finite (~1900 configurations
    /// after pruning) and each iteration performs only range arithmetic, so
    /// the full sweep completes in well under 1 second on a standard laptop.
    ///
    /// No new dev-dependency is introduced: the sweep is hand-written, not
    /// generated by `proptest`/`quickcheck`. This matches the project-wide
    /// convention (no existing usage of property-test crates anywhere in
    /// the workspace).
    mod anticipated_invariants {
        use std::ops::Range;

        use crate::indexer::{EquipmentCounts, EvapConfig, FphaColumnLayout, StageIndexer};

        /// Grid point describing one configuration to test.
        #[derive(Clone, Copy, Debug)]
        struct SweepParams {
            n_hyd: usize,
            l: usize,
            n_ant: usize,
            k_max: usize,
            n_t: usize,
            n_l: usize,
            n_b: usize,
            n_blk: usize,
            pen: bool,
        }

        /// Number of stages used for active-stage queries (I10, I11).
        ///
        /// Must be `>= k_max` so the `K_i <= T` invariant holds for every
        /// generated `lead_stages` scheme.
        const N_STAGES: usize = 8;

        /// Iterator over the full sweep grid, post-pruning.
        ///
        /// Pruning rules:
        /// - `n_ant == 0` requires `k_max == 0` (no plants → no ring buffer);
        /// - `n_ant > 0` requires `k_max >= 1` (debug-asserted by the
        ///   constructor).
        fn parameter_grid() -> impl Iterator<Item = SweepParams> {
            let ns = [0_usize, 1, 3, 5];
            let ls = [0_usize, 1, 2, 4];
            let n_ants = [0_usize, 1, 2, 4];
            let k_maxes = [0_usize, 1, 2, 3];
            let n_therms = [0_usize, 3];
            let n_lns = [0_usize, 2];
            let n_buses_a = [1_usize, 2];
            let n_blks_a = [1_usize, 2, 3];
            let penalties = [false, true];

            ns.into_iter()
                .flat_map(move |n_hyd| {
                    ls.into_iter().flat_map(move |l| {
                        n_ants.into_iter().flat_map(move |n_ant| {
                            k_maxes.into_iter().flat_map(move |k_max| {
                                n_therms.into_iter().flat_map(move |n_t| {
                                    n_lns.into_iter().flat_map(move |n_l| {
                                        n_buses_a.into_iter().flat_map(move |n_b| {
                                            n_blks_a.into_iter().flat_map(move |n_blk| {
                                                penalties.into_iter().map(move |pen| SweepParams {
                                                    n_hyd,
                                                    l,
                                                    n_ant,
                                                    k_max,
                                                    n_t,
                                                    n_l,
                                                    n_b,
                                                    n_blk,
                                                    pen,
                                                })
                                            })
                                        })
                                    })
                                })
                            })
                        })
                    })
                })
                .filter(|p| (p.n_ant > 0 && p.k_max >= 1) || (p.n_ant == 0 && p.k_max == 0))
        }

        /// Build a deterministic `lead_stages` vector for a given grid point.
        ///
        /// The scheme cycles `1..=k_max` so we exercise both `K_i < k_max`
        /// (padding) and `K_i == k_max` (no padding) within a single
        /// configuration. The maximum entry equals `k_max`, which the
        /// constructor debug-asserts.
        fn lead_stages_for(p: &SweepParams) -> Vec<usize> {
            if p.n_ant == 0 {
                return Vec::new();
            }
            let mut out = Vec::with_capacity(p.n_ant);
            // Plant 0 carries k_max to satisfy the `max == k_max` debug_assert.
            out.push(p.k_max);
            for i in 1..p.n_ant {
                out.push(1 + (i % p.k_max));
            }
            out
        }

        /// Build a `StageIndexer` for a grid point with empty FPHA/evap.
        fn build_indexer(p: &SweepParams) -> (StageIndexer, Vec<usize>) {
            let lead_stages = lead_stages_for(p);
            let thermal_indices: Vec<usize> = (0..p.n_ant).collect();
            let counts = EquipmentCounts {
                hydro_count: p.n_hyd,
                max_par_order: p.l,
                n_thermals: p.n_t,
                n_lines: p.n_l,
                n_buses: p.n_b,
                n_blks: p.n_blk,
                has_inflow_penalty: p.pen,
                max_deficit_segments: 1,
                n_anticipated: p.n_ant,
                k_max: p.k_max,
                anticipated_lead_stages: lead_stages.clone(),
                anticipated_thermal_indices: thermal_indices,
                n_pumping: 0,
            };
            let idx = StageIndexer::with_equipment_and_evaporation(
                &counts,
                &FphaColumnLayout {
                    hydro_indices: vec![],
                    planes_per_hydro: vec![],
                },
                &EvapConfig {
                    hydro_indices: vec![],
                },
            );
            (idx, lead_stages)
        }

        /// Append `r` to `v` iff `r` is non-empty.
        fn push_nonempty(v: &mut Vec<Range<usize>>, r: Range<usize>) {
            if !r.is_empty() {
                v.push(r);
            }
        }

        /// Collect the non-empty column ranges in canonical layout order.
        ///
        /// `theta` is a single column index represented as the unit range
        /// `theta..theta + 1` so it slots into the same non-overlap check.
        fn collect_active_column_ranges(idx: &StageIndexer) -> Vec<Range<usize>> {
            let mut v = Vec::new();
            push_nonempty(&mut v, idx.storage.clone());
            push_nonempty(&mut v, idx.inflow_lags.clone());
            push_nonempty(&mut v, idx.anticipated_state.clone());
            push_nonempty(&mut v, idx.z_inflow.clone());
            push_nonempty(&mut v, idx.storage_in.clone());
            v.push(idx.theta..idx.theta + 1);
            push_nonempty(&mut v, idx.turbine.clone());
            push_nonempty(&mut v, idx.spillage.clone());
            push_nonempty(&mut v, idx.diversion.clone());
            push_nonempty(&mut v, idx.thermal.clone());
            push_nonempty(&mut v, idx.anticipated_decision.clone());
            push_nonempty(&mut v, idx.line_fwd.clone());
            push_nonempty(&mut v, idx.line_rev.clone());
            push_nonempty(&mut v, idx.deficit.clone());
            push_nonempty(&mut v, idx.excess.clone());
            push_nonempty(&mut v, idx.inflow_slack.clone());
            push_nonempty(&mut v, idx.generation.clone());
            push_nonempty(&mut v, idx.withdrawal_slack_neg.clone());
            push_nonempty(&mut v, idx.withdrawal_slack_pos.clone());
            push_nonempty(&mut v, idx.outflow_below_slack.clone());
            push_nonempty(&mut v, idx.outflow_above_slack.clone());
            push_nonempty(&mut v, idx.turbine_below_slack.clone());
            push_nonempty(&mut v, idx.generation_below_slack.clone());
            v
        }

        /// Collect the non-empty row ranges in canonical layout order.
        fn collect_active_row_ranges(idx: &StageIndexer) -> Vec<Range<usize>> {
            let mut v = Vec::new();
            push_nonempty(&mut v, idx.storage_fixing.clone());
            push_nonempty(&mut v, idx.lag_fixing.clone());
            push_nonempty(&mut v, idx.anticipated_state_fixing.clone());
            push_nonempty(&mut v, idx.z_inflow_rows.clone());
            push_nonempty(&mut v, idx.water_balance.clone());
            push_nonempty(&mut v, idx.load_balance.clone());
            push_nonempty(&mut v, idx.min_outflow_rows.clone());
            push_nonempty(&mut v, idx.max_outflow_rows.clone());
            push_nonempty(&mut v, idx.min_turbine_rows.clone());
            push_nonempty(&mut v, idx.min_generation_rows.clone());
            // `anticipated_fishing` is structurally `fishing_start..fishing_start`
            // at stage 0; it carries no rows but pins the start offset for I9.
            // Empty ranges are excluded from the contiguity check.
            v
        }

        /// I1: column ranges are pairwise non-overlapping and ascending.
        #[test]
        fn i1_column_ranges_non_overlapping_ascending() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                let ranges = collect_active_column_ranges(&idx);
                for win in ranges.windows(2) {
                    assert!(
                        win[0].end <= win[1].start,
                        "I1 failed at {p:?}: {:?} overlaps {:?}",
                        win[0],
                        win[1]
                    );
                }
            }
        }

        /// I2: row ranges are pairwise non-overlapping and ascending.
        #[test]
        fn i2_row_ranges_non_overlapping_ascending() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                let ranges = collect_active_row_ranges(&idx);
                for win in ranges.windows(2) {
                    assert!(
                        win[0].end <= win[1].start,
                        "I2 failed at {p:?}: {:?} overlaps {:?}",
                        win[0],
                        win[1]
                    );
                }
            }
        }

        /// I3: state-block dimension formula `n_state == N*(1+L) + n_ant*k_max`.
        #[test]
        fn i3_n_state_formula() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                let expected = p.n_hyd * (1 + p.l) + p.n_ant * p.k_max;
                assert_eq!(
                    idx.n_state, expected,
                    "I3 failed at {p:?}: n_state {} != expected {}",
                    idx.n_state, expected
                );
            }
        }

        /// I4 (Phase 1): all three state-fixing row ranges are empty sentinel `0..0`.
        ///
        /// State fixing has moved to column bounds; the row-side ranges are
        /// no longer expected to mirror the column ranges.
        #[test]
        fn i4_state_row_symmetry() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                assert_eq!(
                    idx.storage_fixing,
                    0..0,
                    "I4 storage_fixing must be 0..0 (sentinel) at {p:?}"
                );
                assert_eq!(
                    idx.lag_fixing,
                    0..0,
                    "I4 lag_fixing must be 0..0 (sentinel) at {p:?}"
                );
                assert_eq!(
                    idx.anticipated_state_fixing,
                    0..0,
                    "I4 anticipated_state_fixing must be 0..0 (sentinel) at {p:?}"
                );
            }
        }

        /// I5: theta placement `theta == storage_in.end == N*(3+L) + n_ant*k_max`.
        #[test]
        fn i5_theta_placement() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                let expected = p.n_hyd * (3 + p.l) + p.n_ant * p.k_max;
                assert_eq!(
                    idx.theta, idx.storage_in.end,
                    "I5 storage_in.end mismatch at {p:?}"
                );
                assert_eq!(
                    idx.theta, expected,
                    "I5 formula mismatch at {p:?}: theta {} != {}",
                    idx.theta, expected
                );
            }
        }

        /// I6: `anticipated_decision` and `anticipated_state_out` are contiguous
        /// between `thermal` and `line_fwd`.
        ///
        /// The layout is: `thermal → anticipated_decision → anticipated_state_out → line_fwd`.
        /// When both blocks collapse to `0..0` (no anticipated plants), the
        /// contiguity property reduces to `line_fwd.start == thermal.end`.
        /// Check that branch explicitly so the public `0..0` normalisation does
        /// not silently break the layout.
        #[test]
        fn i6_decision_contiguity() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                if p.n_ant > 0 {
                    assert_eq!(
                        idx.anticipated_decision.start, idx.thermal.end,
                        "I6 decision.start != thermal.end at {p:?}"
                    );
                    assert_eq!(
                        idx.anticipated_state_out.start, idx.anticipated_decision.end,
                        "I6 state_out.start != decision.end at {p:?}"
                    );
                    assert_eq!(
                        idx.line_fwd.start, idx.anticipated_state_out.end,
                        "I6 line_fwd.start != state_out.end at {p:?}"
                    );
                } else {
                    assert_eq!(
                        idx.anticipated_state_out,
                        0..0,
                        "I6 state_out must be 0..0 when n_ant=0 at {p:?}"
                    );
                    assert_eq!(
                        idx.line_fwd.start, idx.thermal.end,
                        "I6 zero-anticipated line_fwd.start != thermal.end at {p:?}"
                    );
                }
            }
        }

        /// I7: `anticipated_decision` column count equals `n_anticipated`.
        #[test]
        fn i7_decision_column_count() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                assert_eq!(
                    idx.anticipated_decision.len(),
                    p.n_ant,
                    "I7 failed at {p:?}: decision len {} != n_ant {}",
                    idx.anticipated_decision.len(),
                    p.n_ant
                );
            }
        }

        /// I8: ring-buffer state dimension `anticipated_state.len() == n_ant * k_max`.
        #[test]
        fn i8_anticipated_state_ring_buffer_dimension() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                let expected = p.n_ant * p.k_max;
                assert_eq!(
                    idx.anticipated_state.len(),
                    expected,
                    "I8 failed at {p:?}: anticipated_state len {} != {}",
                    idx.anticipated_state.len(),
                    expected
                );
            }
        }

        /// I9: `fishing_start` placement.
        ///
        /// When operational violations are active (`hydro_count > 0`), the
        /// fishing block sits right after `min_generation_rows`. When they are
        /// inactive (`hydro_count == 0`), and FPHA/evap are empty as built by
        /// this sweep, `fishing_start` collapses to `load_balance.end`.
        #[test]
        fn i9_fishing_start_placement() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                if p.n_hyd > 0 {
                    assert!(
                        idx.has_operational_violations,
                        "I9 expected operational violations active at {p:?}"
                    );
                    assert_eq!(
                        idx.anticipated_fishing_start, idx.min_generation_rows.end,
                        "I9 fishing_start != min_generation_rows.end at {p:?}"
                    );
                } else {
                    assert!(
                        !idx.has_operational_violations,
                        "I9 expected no operational violations at {p:?}"
                    );
                    assert_eq!(
                        idx.anticipated_fishing_start, idx.load_balance.end,
                        "I9 zero-hydro fishing_start != load_balance.end at {p:?}"
                    );
                }
                // Stage-0 fishing range is empty and pinned at fishing_start.
                assert_eq!(
                    idx.anticipated_fishing.start, idx.anticipated_fishing_start,
                    "I9 fishing.start != fishing_start at {p:?}"
                );
                assert!(
                    idx.anticipated_fishing.is_empty(),
                    "I9 fishing range non-empty at stage 0 for {p:?}"
                );
            }
        }

        /// I10: at stage 0, every anticipated plant is active.
        #[test]
        fn i10_decision_active_at_stage_zero() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                let count = idx
                    .anticipated_decision_active_at_stage(0, N_STAGES)
                    .count();
                assert_eq!(
                    count, p.n_ant,
                    "I10 failed at {p:?}: active decisions {} != n_ant {}",
                    count, p.n_ant
                );
            }
        }

        /// I11: at stage 0, all anticipated plants are active (always-active predicate).
        #[test]
        fn i11_fishing_active_at_stage_zero_is_empty() {
            for p in parameter_grid() {
                let (idx, _) = build_indexer(&p);
                let count = idx.anticipated_fishing_active_at_stage(0, N_STAGES).count();
                assert_eq!(
                    count, p.n_ant,
                    "I11 failed at {p:?}: fishing active count {count} != n_ant {}",
                    p.n_ant
                );
            }
        }

        /// I12: nonzero state mask is sorted ascending with no duplicates.
        #[test]
        fn i12_mask_sorted_unique() {
            for p in parameter_grid() {
                let (mut idx, lead_stages) = build_indexer(&p);
                let lag_counts = vec![p.l; p.n_hyd];
                idx.set_nonzero_mask(&lag_counts, &lead_stages);
                assert!(
                    idx.nonzero_state_indices.windows(2).all(|w| w[0] < w[1]),
                    "I12 failed at {p:?}: mask {:?} is not strictly ascending",
                    idx.nonzero_state_indices
                );
            }
        }

        /// I13: mask length equals `N + sum(lag_counts) + sum(anticipated_lead_stages)`.
        #[test]
        fn i13_mask_length_formula() {
            for p in parameter_grid() {
                let (mut idx, lead_stages) = build_indexer(&p);
                let lag_counts = vec![p.l; p.n_hyd];
                idx.set_nonzero_mask(&lag_counts, &lead_stages);
                let expected =
                    p.n_hyd + lag_counts.iter().sum::<usize>() + lead_stages.iter().sum::<usize>();
                assert_eq!(
                    idx.nonzero_state_indices.len(),
                    expected,
                    "I13 failed at {p:?}: mask len {} != {}",
                    idx.nonzero_state_indices.len(),
                    expected
                );
            }
        }

        /// Sweep coverage assertion: the pruned grid must exercise at least 500
        /// distinct configurations to give meaningful coverage of the joint
        /// parameter space.
        #[test]
        fn sweep_coverage_at_least_500() {
            let count = parameter_grid().count();
            assert!(
                count >= 500,
                "sweep coverage {count} is below the 500-config minimum"
            );
        }
    }
}
