//! Declaration-order invariance regression test for [`ResolvedParameters`].
//!
//! Asserts that calling [`build_resolved_parameters`] with the same set of
//! [`ScalarParameter`] entries in two different orderings produces bit-for-bit
//! identical resolved values for every `(EntityId, stage_idx)` pair and that
//! the sorted `id_to_slot` key sequence is canonicalized to `[10, 20, 30]`
//! regardless of authored order.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use cobre_core::{EntityId, ParameterKind, ScalarParameter};
use cobre_sddp::build_resolved_parameters;
use cobre_sddp::energy_conversion::{EnergyConversionSet, HydroEnergyProductivityOverride};

/// Return three [`ScalarParameter`] entries in the order specified by `order`.
///
/// - slot 0 → `id=10`, `name="alpha"`, `Constant { value: 3.6 }`
/// - slot 1 → `id=20`, `name="beta"`,  `PerStage { values: [1.0, 1.5, 2.0, 2.5] }`
/// - slot 2 → `id=30`, `name="gamma"`, `Seasonal { values: [(0, 0.9), (1, 1.1)] }`
fn make_params(order: &[usize]) -> Vec<ScalarParameter> {
    let alpha = ScalarParameter {
        id: EntityId(10),
        name: "alpha".to_string(),
        kind: ParameterKind::Constant { value: 3.6 },
    };
    let beta = ScalarParameter {
        id: EntityId(20),
        name: "beta".to_string(),
        kind: ParameterKind::PerStage {
            values: vec![1.0, 1.5, 2.0, 2.5],
        },
    };
    let gamma = ScalarParameter {
        id: EntityId(30),
        name: "gamma".to_string(),
        kind: ParameterKind::Seasonal {
            values: vec![(0, 0.9), (1, 1.1)],
        },
    };
    let pool = [alpha, beta, gamma];
    order.iter().map(|&i| pool[i].clone()).collect()
}

#[test]
fn scalar_parameters_resolution_is_declaration_order_invariant() {
    let stage_to_season: [i32; 4] = [0, 1, 0, 1];
    let n_stages = 4;
    let ec = EnergyConversionSet::new(vec![], vec![], 0, n_stages);
    let overrides = HydroEnergyProductivityOverride::default();
    let hydros: Vec<cobre_core::Hydro> = Vec::new();

    let order_a = make_params(&[0, 1, 2]); // alpha, beta, gamma
    let order_b = make_params(&[2, 0, 1]); // gamma, alpha, beta

    let resolved_a = build_resolved_parameters(
        &order_a,
        &ec,
        &overrides,
        &hydros,
        &stage_to_season,
        n_stages,
    )
    .expect("ResolvedParameters builds for order_a");
    let resolved_b = build_resolved_parameters(
        &order_b,
        &ec,
        &overrides,
        &hydros,
        &stage_to_season,
        n_stages,
    )
    .expect("ResolvedParameters builds for order_b");

    // Bit-exact value equality for every (id, stage) pair.
    for id_raw in [10_i32, 20, 30] {
        for stage_idx in 0..n_stages {
            let va = resolved_a.get(EntityId(id_raw), stage_idx);
            let vb = resolved_b.get(EntityId(id_raw), stage_idx);
            assert_eq!(
                va.to_bits(),
                vb.to_bits(),
                "mismatch at id={id_raw}, stage={stage_idx}: a={va} b={vb}",
            );
        }
    }

    // The sorted key sequence must be canonicalized to [10, 20, 30]
    // regardless of authored order.
    assert_eq!(resolved_a.id_to_slot.len(), 3);
    assert_eq!(resolved_b.id_to_slot.len(), 3);

    let keys_a: Vec<i32> = resolved_a.id_to_slot.iter().map(|(k, _)| *k).collect();
    let keys_b: Vec<i32> = resolved_b.id_to_slot.iter().map(|(k, _)| *k).collect();
    assert_eq!(keys_a, vec![10, 20, 30]);
    assert_eq!(keys_b, vec![10, 20, 30]);
}
