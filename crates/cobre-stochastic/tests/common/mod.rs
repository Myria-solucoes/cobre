//! Shared correlation-model and entity-order fixtures for the Sobol/Halton/LHS
//! integration-test suite.

use std::collections::BTreeMap;

use cobre_core::{
    EntityId,
    scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
};
use cobre_stochastic::{ClassDimensions, correlation::resolve::DecomposedCorrelation};

fn correlation_model(entity_ids: &[i32], rho: f64) -> CorrelationModel {
    let n = entity_ids.len();
    let matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
        .collect();
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: entity_ids
                    .iter()
                    .map(|&id| CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(id),
                    })
                    .collect(),
                matrix,
            }],
        },
    );
    CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    }
}

fn inflow_entity_order_and_dims(entity_ids: &[i32]) -> (Vec<EntityId>, ClassDimensions) {
    (
        entity_ids.iter().map(|&id| EntityId(id)).collect(),
        ClassDimensions {
            n_hydros: entity_ids.len(),
            n_load_buses: 0,
            n_ncs: 0,
        },
    )
}

pub fn identity_correlation_model(entity_ids: &[i32]) -> CorrelationModel {
    correlation_model(entity_ids, 0.0)
}

pub fn identity_correlation(entity_ids: &[i32]) -> DecomposedCorrelation {
    let (entity_order, dims) = inflow_entity_order_and_dims(entity_ids);
    DecomposedCorrelation::build(&identity_correlation_model(entity_ids), &entity_order, dims)
        .unwrap()
}

pub fn correlated_correlation(entity_ids: &[i32], rho: f64) -> DecomposedCorrelation {
    let (entity_order, dims) = inflow_entity_order_and_dims(entity_ids);
    DecomposedCorrelation::build(&correlation_model(entity_ids, rho), &entity_order, dims).unwrap()
}
