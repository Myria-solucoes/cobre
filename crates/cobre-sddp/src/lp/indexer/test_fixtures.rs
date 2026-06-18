//! Shared `EquipmentCounts` / `FphaColumnLayout` / `EvapConfig` builders for the
//! sibling indexer unit tests.
//!
//! Single owner of the `eq` / `eq_with_anticipated` / `fpha` / `evap` fixtures
//! that the `constructors`, `anticipated`, `state_mapping`, and `sparse_state`
//! test modules consume. The named `eq` / `eq_with_anticipated` builders set
//! `max_deficit_segments: 1` (a non-degenerate deficit stride), which is **not**
//! the `EquipmentCounts::default()` value `0`; use `..Default::default()`
//! directly when an all-zero count is wanted instead.
//!
//! Compiled under `#[cfg(any(test, feature = "test-support"))]` so plain
//! `cargo test` and downstream integration tests (via `test-support`) both
//! reach the same builders.

use super::layout::{EquipmentCounts, EvapConfig, FphaColumnLayout};

/// Build `EquipmentCounts` with the seven scalar entity counts set and no
/// anticipated thermals.
#[must_use]
pub fn eq(
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

/// Build an `FphaColumnLayout` from per-hydro indices and plane counts.
#[must_use]
pub fn fpha(hydro_indices: Vec<usize>, planes_per_hydro: Vec<usize>) -> FphaColumnLayout {
    FphaColumnLayout {
        hydro_indices,
        planes_per_hydro,
    }
}

/// Build an `EvapConfig` covering the given hydro indices.
#[must_use]
pub fn evap(hydro_indices: Vec<usize>) -> EvapConfig {
    EvapConfig { hydro_indices }
}

/// Test helper: build `EquipmentCounts` with explicit anticipated thermal
/// fields.
#[must_use]
pub fn eq_with_anticipated(
    hydro_count: usize,
    max_par_order: usize,
    n_thermals: usize,
    n_lines: usize,
    n_buses: usize,
    n_blks: usize,
    has_inflow_penalty: bool,
    n_anticipated: usize,
    k_max: usize,
) -> EquipmentCounts {
    // Default the per-plant K_i array to a uniform `k_max` of length
    // `n_anticipated` so debug asserts on per-plant lead-stage
    // bookkeeping hold. Tests that need a mixed K_i array must construct
    // `EquipmentCounts` directly.
    let anticipated_lead_stages = if n_anticipated == 0 {
        vec![]
    } else {
        vec![k_max; n_anticipated]
    };
    let anticipated_thermal_indices = if n_anticipated == 0 {
        vec![]
    } else {
        (0..n_anticipated).collect()
    };
    EquipmentCounts {
        hydro_count,
        max_par_order,
        n_thermals,
        n_lines,
        n_buses,
        n_blks,
        has_inflow_penalty,
        max_deficit_segments: 1,
        n_anticipated,
        k_max,
        anticipated_lead_stages,
        anticipated_thermal_indices,
        n_pumping: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::eq_with_anticipated;

    /// The shared `eq_with_anticipated` builder pins `max_deficit_segments == 1`
    /// (not the `Default` value `0`) and produces empty anticipated vecs when
    /// `n_anticipated == 0`, reproducing the field set the per-module copies
    /// previously hard-coded.
    #[test]
    fn shared_eq_with_anticipated_matches_legacy_fixture_shape() {
        let counts = eq_with_anticipated(0, 0, 0, 0, 0, 0, false, 0, 0);
        assert_eq!(counts.hydro_count, 0);
        assert_eq!(counts.max_par_order, 0);
        assert_eq!(counts.n_thermals, 0);
        assert_eq!(counts.n_lines, 0);
        assert_eq!(counts.n_buses, 0);
        assert_eq!(counts.n_blks, 0);
        assert!(!counts.has_inflow_penalty);
        // Named builder uses 1, deliberately diverging from Default's 0.
        assert_eq!(counts.max_deficit_segments, 1);
        assert_eq!(counts.n_anticipated, 0);
        assert_eq!(counts.k_max, 0);
        assert_eq!(counts.n_pumping, 0);
        assert_eq!(counts.anticipated_lead_stages, Vec::<usize>::new());
        assert_eq!(counts.anticipated_thermal_indices, Vec::<usize>::new());
    }
}
