//! Spectral decomposition and runtime application of spatial correlation.
//!
//! Decomposes each correlation profile's groups into spectral factors and builds
//! a stage-to-profile schedule for runtime lookup, then transforms independent
//! standard-normal noise into spatially correlated noise.
//!
//! Non-positive-definite matrices do not error; negative eigenvalues are clipped
//! to 0.0 (nearest PSD approximation).

use std::collections::{BTreeMap, HashMap};

use cobre_core::{CorrelationModel, EntityId};

use crate::{
    StochasticError,
    correlation::spectral::{NEGLIGIBLE_NEGATIVE_EIGENVALUE, SpectralFactor},
};

/// Group dimension at or below which `apply_correlation` buffers are stack-allocated;
/// larger groups fall back to heap allocation.
const MAX_STACK_DIM: usize = 64;

/// A single correlation group's spectral factor with entity ID mapping.
#[derive(Debug)]
pub struct GroupFactor {
    /// Spectral factor for this group.
    pub factor: SpectralFactor,
    /// Entity IDs in the order matching the factor rows/columns.
    pub entity_ids: Vec<EntityId>,
    /// Entity type shared by all entities in this group (`"inflow"`, `"load"`, `"ncs"`).
    pub entity_type: String,
    /// Positions within the canonical full entity order; when `Some`,
    /// `apply_correlation` skips the per-call linear scan.
    positions: Option<Box<[usize]>>,
    /// Positions within the per-class entity order; when `Some`,
    /// `apply_correlation_for_class` skips the per-call linear scan.
    class_positions: Option<Box<[usize]>>,
}

/// Pre-decomposed correlation data for all profiles, with stage-to-profile mapping.
#[derive(Debug)]
pub struct DecomposedCorrelation {
    /// Spectral factors keyed by profile name; `BTreeMap` for deterministic
    /// iteration order (upholds declaration-order invariance).
    factors: BTreeMap<String, Vec<GroupFactor>>,

    /// Stage-to-profile mapping; stages absent here use the default profile.
    schedule: HashMap<i32, String>,

    /// Name of the default profile.
    default_profile: String,
}

impl DecomposedCorrelation {
    /// Constructs an empty `DecomposedCorrelation` for systems with no stochastic
    /// entities. [`apply_correlation`] is then a no-op; [`profile_for_stage`]
    /// returns an empty string.
    ///
    /// [`apply_correlation`]: Self::apply_correlation
    /// [`profile_for_stage`]: Self::profile_for_stage
    #[must_use]
    pub fn empty() -> Self {
        Self {
            factors: BTreeMap::new(),
            schedule: HashMap::new(),
            default_profile: String::new(),
        }
    }

    /// Builds a `DecomposedCorrelation` from a [`CorrelationModel`].
    ///
    /// # Errors
    ///
    /// - [`StochasticError::InvalidCorrelation`] if no `"default"` profile exists
    ///   and there is more than one profile (ambiguous default).
    /// - [`StochasticError::InvalidCorrelation`] if the model has no profiles.
    /// - [`StochasticError::InvalidCorrelation`] if a correlation matrix is not
    ///   square or not symmetric.
    ///
    /// Non-positive-definite matrices do not cause an error; negative eigenvalues
    /// are clipped to 0.0 to produce the nearest positive-semidefinite approximation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    /// use cobre_core::{EntityId, scenario::{
    ///     CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
    /// }};
    /// use cobre_stochastic::correlation::resolve::DecomposedCorrelation;
    ///
    /// let mut profiles = BTreeMap::new();
    /// profiles.insert("default".to_string(), CorrelationProfile {
    ///     groups: vec![CorrelationGroup {
    ///         name: "g1".to_string(),
    ///         entities: vec![
    ///             CorrelationEntity { entity_type: "inflow".to_string(), id: EntityId(1) },
    ///             CorrelationEntity { entity_type: "inflow".to_string(), id: EntityId(2) },
    ///         ],
    ///         matrix: vec![vec![1.0, 0.8], vec![0.8, 1.0]],
    ///     }],
    /// });
    /// let model = CorrelationModel { method: "spectral".to_string(), profiles, schedule: vec![] };
    /// let dc = DecomposedCorrelation::build(&model).unwrap();
    /// ```
    pub fn build(model: &CorrelationModel) -> Result<Self, StochasticError> {
        if model.profiles.is_empty() {
            return Err(StochasticError::InvalidCorrelation {
                profile_name: String::new(),
                reason: "correlation model contains no profiles".into(),
            });
        }

        let default_profile = if model.profiles.contains_key("default") {
            "default".to_string()
        } else if model.profiles.len() == 1 {
            model.profiles.keys().next().cloned().unwrap_or_default()
        } else {
            return Err(StochasticError::InvalidCorrelation {
                profile_name: String::new(),
                reason: format!(
                    "no 'default' profile found and {} profiles exist; \
                     add a profile named 'default' or reduce to a single profile",
                    model.profiles.len()
                ),
            });
        };

        let mut factors: BTreeMap<String, Vec<GroupFactor>> = BTreeMap::new();

        // Accumulate clipping across all matrices to emit one summary, not one line per matrix.
        let mut clip_matrices_total = 0_usize;
        let mut clip_matrices_affected = 0_usize;
        let mut clip_eigenvalues_total = 0_usize;
        let mut clip_largest_magnitude = 0.0_f64;

        for (profile_name, profile) in &model.profiles {
            for group in &profile.groups {
                if group.entities.len() > 1 {
                    let first_type = &group.entities[0].entity_type;
                    if let Some(mixed) =
                        group.entities.iter().find(|e| e.entity_type != *first_type)
                    {
                        return Err(StochasticError::InvalidCorrelation {
                            profile_name: profile_name.clone(),
                            reason: format!(
                                "correlation group '{}' contains mixed entity types: \
                                 found '{}' and '{}'; \
                                 all entities in a group must share the same entity_type",
                                group.name, first_type, mixed.entity_type,
                            ),
                        });
                    }
                }
            }

            // Overlapping groups (an entity ID in more than one group) produce incorrect covariance.
            {
                let mut seen: std::collections::HashSet<EntityId> =
                    std::collections::HashSet::new();
                for group in &profile.groups {
                    for entity in &group.entities {
                        if !seen.insert(entity.id) {
                            return Err(StochasticError::InvalidCorrelation {
                                profile_name: profile_name.clone(),
                                reason: format!(
                                    "entity ID {} appears in more than one correlation group \
                                     within profile '{}'; groups must be disjoint",
                                    entity.id.0, profile_name,
                                ),
                            });
                        }
                    }
                }
            }

            let mut group_factors: Vec<GroupFactor> = Vec::with_capacity(profile.groups.len());
            for group in &profile.groups {
                let (factor, clip) = SpectralFactor::decompose_with_diagnostics(&group.matrix)
                    .map_err(|e| match e {
                        StochasticError::InvalidCorrelation { reason, .. } => {
                            StochasticError::InvalidCorrelation {
                                profile_name: profile_name.clone(),
                                reason,
                            }
                        }
                        other => other,
                    })?;
                clip_matrices_total += 1;
                if clip.clipped_count > 0 {
                    clip_matrices_affected += 1;
                    clip_eigenvalues_total += clip.clipped_count;
                    clip_largest_magnitude = clip_largest_magnitude.max(clip.largest_magnitude);
                }

                let entity_ids: Vec<EntityId> = group.entities.iter().map(|e| e.id).collect();
                let entity_type = group
                    .entities
                    .first()
                    .ok_or_else(|| StochasticError::InvalidCorrelation {
                        profile_name: profile_name.clone(),
                        reason: format!("correlation group '{}' has no entities", group.name),
                    })?
                    .entity_type
                    .clone();

                group_factors.push(GroupFactor {
                    factor,
                    entity_ids,
                    entity_type,
                    positions: None,
                    class_positions: None,
                });
            }
            factors.insert(profile_name.clone(), group_factors);
        }

        // Clipping above NEGLIGIBLE_NEGATIVE_EIGENVALUE signals a genuinely indefinite
        // input (warn); round-off-scale clipping is harmless PSD projection (debug).
        if clip_matrices_affected > 0 {
            if clip_largest_magnitude > NEGLIGIBLE_NEGATIVE_EIGENVALUE {
                tracing::warn!(
                    matrices_affected = clip_matrices_affected,
                    matrices_total = clip_matrices_total,
                    clipped_eigenvalues = clip_eigenvalues_total,
                    largest_negative_magnitude = clip_largest_magnitude,
                    "spectral decomposition clipped significant negative eigenvalues; \
                     one or more correlation matrices may be indefinite rather than \
                     merely rank-deficient — verify the correlation inputs"
                );
            } else {
                tracing::debug!(
                    matrices_affected = clip_matrices_affected,
                    matrices_total = clip_matrices_total,
                    clipped_eigenvalues = clip_eigenvalues_total,
                    largest_negative_magnitude = clip_largest_magnitude,
                    "spectral decomposition clipped near-zero negative eigenvalues to 0.0 \
                     (nearest PSD approximation); correlation matrices are rank-deficient \
                     but the projection is numerically negligible"
                );
            }
        }

        let schedule: HashMap<i32, String> = model
            .schedule
            .iter()
            .map(|entry| (entry.stage_id, entry.profile_name.clone()))
            .collect();

        Ok(Self {
            factors,
            schedule,
            default_profile,
        })
    }

    /// Returns the profile name active for `stage_id`, falling back to the default profile.
    pub fn profile_for_stage(&self, stage_id: i32) -> &str {
        self.schedule
            .get(&stage_id)
            .map_or(self.default_profile.as_str(), String::as_str)
    }

    /// Pre-computes each group's entity positions within `entity_order`. Call once
    /// before the [`Self::apply_correlation`] hot loop to eliminate its per-call
    /// O(n) linear scan. Groups absent from `entity_order` get empty positions and
    /// are skipped during application.
    pub fn resolve_positions(&mut self, entity_order: &[EntityId]) {
        let id_to_pos: HashMap<EntityId, usize> = entity_order
            .iter()
            .enumerate()
            .map(|(i, &eid)| (eid, i))
            .collect();

        for group_factors in self.factors.values_mut() {
            for gf in group_factors.iter_mut() {
                let positions: Vec<usize> = gf
                    .entity_ids
                    .iter()
                    .filter_map(|eid| id_to_pos.get(eid).copied())
                    .collect();
                gf.positions = Some(positions.into_boxed_slice());
            }
        }
    }

    /// Per-class counterpart to [`Self::resolve_positions`]: `class_entity_order`
    /// holds only one class's entity IDs, and only groups matching `entity_type` are
    /// updated. Call once per class before the [`Self::apply_correlation_for_class`]
    /// hot loop. Groups absent from `class_entity_order` get empty positions and are
    /// skipped during application.
    pub fn resolve_class_positions(&mut self, class_entity_order: &[EntityId], entity_type: &str) {
        let id_to_pos: HashMap<EntityId, usize> = class_entity_order
            .iter()
            .enumerate()
            .map(|(i, &eid)| (eid, i))
            .collect();

        for group_factors in self.factors.values_mut() {
            for gf in group_factors.iter_mut() {
                if gf.entity_type != entity_type {
                    continue;
                }
                let positions: Vec<usize> = gf
                    .entity_ids
                    .iter()
                    .filter_map(|eid| id_to_pos.get(eid).copied())
                    .collect();
                gf.class_positions = Some(positions.into_boxed_slice());
            }
        }
    }

    /// Applies spatial correlation to `independent_noise` for `stage_id`. Entities
    /// in no correlation group keep their independent values.
    ///
    /// Call [`resolve_positions`] once before the hot loop; with positions
    /// pre-computed this performs zero heap allocations for groups of up to
    /// `MAX_STACK_DIM` entities.
    ///
    /// [`resolve_positions`]: Self::resolve_positions
    pub fn apply_correlation(
        &self,
        stage_id: i32,
        independent_noise: &mut [f64],
        entity_order: &[EntityId],
    ) {
        let profile_name = self.profile_for_stage(stage_id);
        let Some(group_factors) = self.factors.get(profile_name) else {
            // Missing profile leaves noise unchanged rather than erroring.
            return;
        };

        for gf in group_factors {
            if let Some(ref precomputed) = gf.positions {
                Self::apply_group_precomputed(&gf.factor, precomputed, independent_noise);
            } else {
                Self::apply_group_scan(&gf.factor, &gf.entity_ids, independent_noise, entity_order);
            }
        }
    }

    /// Per-class counterpart to [`apply_correlation`], applying only groups whose
    /// `entity_type` matches. `class_noise` and `class_entity_order` hold only that
    /// class's entities, so positions are relative to the class segment, NOT the
    /// full noise vector.
    ///
    /// Call [`resolve_class_positions`] once per class before the hot loop for the
    /// same allocation-free fast path as [`apply_correlation`].
    ///
    /// [`apply_correlation`]: Self::apply_correlation
    /// [`resolve_class_positions`]: Self::resolve_class_positions
    pub fn apply_correlation_for_class(
        &self,
        stage_id: i32,
        class_noise: &mut [f64],
        class_entity_order: &[EntityId],
        entity_type: &str,
    ) {
        let profile_name = self.profile_for_stage(stage_id);
        let Some(group_factors) = self.factors.get(profile_name) else {
            // Missing profile leaves noise unchanged rather than erroring.
            return;
        };

        for gf in group_factors {
            if gf.entity_type != entity_type {
                continue;
            }
            if let Some(ref precomputed) = gf.class_positions {
                Self::apply_group_precomputed(&gf.factor, precomputed, class_noise);
            } else {
                Self::apply_group_scan(&gf.factor, &gf.entity_ids, class_noise, class_entity_order);
            }
        }
    }

    fn apply_group_precomputed(factor: &SpectralFactor, positions: &[usize], noise: &mut [f64]) {
        let n = positions.len();
        if n == 0 || n != factor.dim() {
            return;
        }

        if n <= MAX_STACK_DIM {
            let mut gathered = [0.0_f64; MAX_STACK_DIM];
            let mut correlated = [0.0_f64; MAX_STACK_DIM];
            for (i, &pos) in positions.iter().enumerate() {
                gathered[i] = noise[pos];
            }
            factor.transform(&gathered[..n], &mut correlated[..n]);
            for (i, &pos) in positions.iter().enumerate() {
                noise[pos] = correlated[i];
            }
        } else {
            let gathered: Vec<f64> = positions.iter().map(|&pos| noise[pos]).collect();
            let mut correlated = vec![0.0_f64; n];
            factor.transform(&gathered, &mut correlated);
            for (i, &pos) in positions.iter().enumerate() {
                noise[pos] = correlated[i];
            }
        }
    }

    /// Linear-scan fallback for callers that did not pre-compute positions.
    fn apply_group_scan(
        factor: &SpectralFactor,
        entity_ids: &[EntityId],
        noise: &mut [f64],
        entity_order: &[EntityId],
    ) {
        let entity_count = entity_ids.len();
        if entity_count == 0 {
            return;
        }

        if entity_count <= MAX_STACK_DIM {
            let mut positions = [0_usize; MAX_STACK_DIM];
            let mut gathered = [0.0_f64; MAX_STACK_DIM];
            let mut correlated = [0.0_f64; MAX_STACK_DIM];

            let mut n = 0;
            for eid in entity_ids {
                if let Some(pos) = entity_order.iter().position(|e| e == eid) {
                    positions[n] = pos;
                    n += 1;
                }
            }

            if n == 0 || n != factor.dim() {
                return;
            }

            for i in 0..n {
                gathered[i] = noise[positions[i]];
            }
            factor.transform(&gathered[..n], &mut correlated[..n]);
            for i in 0..n {
                noise[positions[i]] = correlated[i];
            }
        } else {
            let positions: Vec<usize> = entity_ids
                .iter()
                .filter_map(|eid| entity_order.iter().position(|e| e == eid))
                .collect();

            let n = positions.len();
            if n == 0 || n != factor.dim() {
                return;
            }

            let gathered: Vec<f64> = positions.iter().map(|&pos| noise[pos]).collect();
            let mut correlated = vec![0.0_f64; n];
            factor.transform(&gathered, &mut correlated);
            for (i, &pos) in positions.iter().enumerate() {
                noise[pos] = correlated[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use cobre_core::{
        EntityId,
        scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
            CorrelationScheduleEntry,
        },
    };

    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_entity(id: i32) -> CorrelationEntity {
        CorrelationEntity {
            entity_type: "inflow".to_string(),
            id: EntityId(id),
        }
    }

    fn make_entity_of_type(id: i32, entity_type: &str) -> CorrelationEntity {
        CorrelationEntity {
            entity_type: entity_type.to_string(),
            id: EntityId(id),
        }
    }

    fn identity_group(name: &str, entity_ids: &[i32]) -> CorrelationGroup {
        let n = entity_ids.len();
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        CorrelationGroup {
            name: name.to_string(),
            entities: entity_ids.iter().copied().map(make_entity).collect(),
            matrix,
        }
    }

    fn correlated_group(name: &str, entity_ids: &[i32], rho: f64) -> CorrelationGroup {
        let n = entity_ids.len();
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
            .collect();
        CorrelationGroup {
            name: name.to_string(),
            entities: entity_ids.iter().copied().map(make_entity).collect(),
            matrix,
        }
    }

    fn make_group_with_type(
        name: &str,
        entity_ids: &[i32],
        rho: f64,
        entity_type: &str,
    ) -> CorrelationGroup {
        let n = entity_ids.len();
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
            .collect();
        CorrelationGroup {
            name: name.to_string(),
            entities: entity_ids
                .iter()
                .copied()
                .map(|id| make_entity_of_type(id, entity_type))
                .collect(),
            matrix,
        }
    }

    fn single_profile_model(profile_name: &str, groups: Vec<CorrelationGroup>) -> CorrelationModel {
        let mut profiles = BTreeMap::new();
        profiles.insert(profile_name.to_string(), CorrelationProfile { groups });
        CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        }
    }

    // -----------------------------------------------------------------------
    // Build tests
    // -----------------------------------------------------------------------

    #[test]
    fn build_single_default_profile() {
        let model = single_profile_model("default", vec![identity_group("g1", &[1, 2])]);
        let dc = DecomposedCorrelation::build(&model).unwrap();
        assert_eq!(dc.default_profile, "default");
        assert!(dc.factors.contains_key("default"));
    }

    #[test]
    fn build_single_non_default_profile_used_as_default() {
        // Exactly one profile named "wet" — should become the implicit default.
        let model = single_profile_model("wet", vec![identity_group("g1", &[1])]);
        let dc = DecomposedCorrelation::build(&model).unwrap();
        assert_eq!(dc.default_profile, "wet");
    }

    #[test]
    fn build_fails_with_no_profiles() {
        let model = CorrelationModel {
            method: "spectral".to_string(),
            profiles: BTreeMap::new(),
            schedule: vec![],
        };
        let result = DecomposedCorrelation::build(&model);
        assert!(
            matches!(result, Err(StochasticError::InvalidCorrelation { .. })),
            "Expected InvalidCorrelation, got: {result:?}"
        );
    }

    #[test]
    fn build_fails_with_multiple_profiles_and_no_default() {
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "wet".to_string(),
            CorrelationProfile {
                groups: vec![identity_group("g1", &[1])],
            },
        );
        profiles.insert(
            "dry".to_string(),
            CorrelationProfile {
                groups: vec![identity_group("g1", &[1])],
            },
        );
        let model = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };
        let result = DecomposedCorrelation::build(&model);
        assert!(
            matches!(result, Err(StochasticError::InvalidCorrelation { .. })),
            "Expected InvalidCorrelation, got: {result:?}"
        );
    }

    #[test]
    fn build_with_schedule_mapping() {
        // Acceptance criterion: profiles "default" and "wet", schedule maps stage 0 -> "wet".
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![identity_group("g1", &[1, 2])],
            },
        );
        profiles.insert(
            "wet".to_string(),
            CorrelationProfile {
                groups: vec![correlated_group("g1", &[1, 2], 0.8)],
            },
        );
        let model = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![CorrelationScheduleEntry {
                stage_id: 0,
                profile_name: "wet".to_string(),
            }],
        };
        let dc = DecomposedCorrelation::build(&model).unwrap();

        // Stage 0 should use "wet".
        assert_eq!(dc.profile_for_stage(0), "wet");
        // Stage 1 (not in schedule) should use "default".
        assert_eq!(dc.profile_for_stage(1), "default");
    }

    // -----------------------------------------------------------------------
    // apply_correlation tests
    // -----------------------------------------------------------------------

    #[test]
    fn apply_correlation_with_identity_factor_leaves_noise_unchanged() {
        let model = single_profile_model("default", vec![identity_group("g1", &[1, 2])]);
        let dc = DecomposedCorrelation::build(&model).unwrap();

        let entity_order = [EntityId(1), EntityId(2)];
        let mut noise = [3.0_f64, 5.0];
        dc.apply_correlation(0, &mut noise, &entity_order);

        assert!((noise[0] - 3.0).abs() < 1e-12, "noise[0]={}", noise[0]);
        assert!((noise[1] - 5.0).abs() < 1e-12, "noise[1]={}", noise[1]);
    }

    #[test]
    fn apply_correlation_with_known_factor() {
        // Use [[1, 0.8],[0.8, 1]]. Eigenvalues: lambda_1=1.8, lambda_2=0.2.
        // Eigenvectors: v1=[1,1]/sqrt(2), v2=[1,-1]/sqrt(2).
        // D = V * diag(sqrt(1.8), sqrt(0.2)) * V^T:
        //   D[0][0] = D[1][1] = (sqrt(1.8) + sqrt(0.2)) / 2 ~= 0.894427190999916
        //   D[0][1] = D[1][0] = (sqrt(1.8) - sqrt(0.2)) / 2 ~= 0.447213595499958
        // z=[1.0, 0.0] => result = [D[0][0], D[1][0]] ~= [0.894427, 0.447214].
        let group = CorrelationGroup {
            name: "g1".to_string(),
            entities: vec![make_entity(1), make_entity(2)],
            matrix: vec![vec![1.0, 0.8], vec![0.8, 1.0]],
        };
        let model = single_profile_model("default", vec![group]);
        let dc = DecomposedCorrelation::build(&model).unwrap();

        let entity_order = [EntityId(1), EntityId(2)];
        let mut noise = [1.0_f64, 0.0];
        dc.apply_correlation(0, &mut noise, &entity_order);

        let d00 = f64::midpoint(f64::sqrt(1.8), f64::sqrt(0.2));
        let d10 = (f64::sqrt(1.8) - f64::sqrt(0.2)) / 2.0;
        assert!((noise[0] - d00).abs() < 1e-8, "noise[0]={}", noise[0]);
        assert!((noise[1] - d10).abs() < 1e-8, "noise[1]={}", noise[1]);
    }

    #[test]
    fn apply_correlation_leaves_unmatched_entities_unchanged() {
        // entity_order has entities 1, 2, 3; group only covers 1 and 2.
        let model = single_profile_model("default", vec![identity_group("g1", &[1, 2])]);
        let dc = DecomposedCorrelation::build(&model).unwrap();

        let entity_order = [EntityId(1), EntityId(2), EntityId(3)];
        let mut noise = [1.0_f64, 2.0, 99.0];
        dc.apply_correlation(0, &mut noise, &entity_order);

        // Entity 3 (index 2) must be untouched.
        assert!(
            (noise[2] - 99.0).abs() < 1e-12,
            "unmatched entity changed: {}",
            noise[2]
        );
    }

    #[test]
    fn apply_correlation_with_reordered_entities() {
        // The canonical order is [2, 1] but the group defines entities as [1, 2].
        // Independent samples: noise[0] (entity 2) = 0.5, noise[1] (entity 1) = 1.0.
        // Group matrix [[1,0.8],[0.8,1]] -- entities in group order [1, 2]:
        //   entity 1 is at position 1 in entity_order
        //   entity 2 is at position 0 in entity_order
        // gathered = [noise[pos(1)]=noise[1]=1.0, noise[pos(2)]=noise[0]=0.5]
        // Spectral factor: D[0][0]=D[1][1]=(sqrt(1.8)+sqrt(0.2))/2, D[0][1]=D[1][0]=(sqrt(1.8)-sqrt(0.2))/2
        // correlated = D * [1.0, 0.5]:
        //   correlated[0] = D[0][0]*1.0 + D[0][1]*0.5 ~= 1.118033988749895
        //   correlated[1] = D[1][0]*1.0 + D[1][1]*0.5 ~= 0.894427190999916
        // scattered: noise[1] = correlated[0] ~= 1.118034, noise[0] = correlated[1] ~= 0.894427
        let group = CorrelationGroup {
            name: "g1".to_string(),
            entities: vec![make_entity(1), make_entity(2)],
            matrix: vec![vec![1.0, 0.8], vec![0.8, 1.0]],
        };
        let model = single_profile_model("default", vec![group]);
        let dc = DecomposedCorrelation::build(&model).unwrap();

        // Canonical order: entity 2 first, entity 1 second.
        let entity_order = [EntityId(2), EntityId(1)];
        let mut noise = [0.5_f64, 1.0]; // noise[0]=entity2, noise[1]=entity1
        dc.apply_correlation(0, &mut noise, &entity_order);

        let d00 = f64::midpoint(f64::sqrt(1.8), f64::sqrt(0.2));
        let d01 = (f64::sqrt(1.8) - f64::sqrt(0.2)) / 2.0;
        // correlated[0] = D[0][0]*1.0 + D[0][1]*0.5; scattered to noise[1]
        let expected_noise1 = d00 * 1.0 + d01 * 0.5;
        // correlated[1] = D[1][0]*1.0 + D[1][1]*0.5; scattered to noise[0]
        let expected_noise0 = d01 * 1.0 + d00 * 0.5;
        assert!(
            (noise[1] - expected_noise1).abs() < 1e-8,
            "noise[1]={} expected {}",
            noise[1],
            expected_noise1
        );
        assert!(
            (noise[0] - expected_noise0).abs() < 1e-8,
            "noise[0]={} expected {}",
            noise[0],
            expected_noise0
        );
    }

    #[test]
    fn apply_correlation_uses_correct_profile_for_stage() {
        // Stage 0 -> "wet" (rho=0.8), stage 1 -> "default" (identity).
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![identity_group("g1", &[1, 2])],
            },
        );
        profiles.insert(
            "wet".to_string(),
            CorrelationProfile {
                groups: vec![correlated_group("g1", &[1, 2], 0.8)],
            },
        );
        let model = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![CorrelationScheduleEntry {
                stage_id: 0,
                profile_name: "wet".to_string(),
            }],
        };
        let dc = DecomposedCorrelation::build(&model).unwrap();

        let entity_order = [EntityId(1), EntityId(2)];

        // Stage 0 uses "wet": z=[1,0] -> spectral D*[1,0]=[D[0][0], D[1][0]].
        // For [[1,0.8],[0.8,1]]: D[0][0]=(sqrt(1.8)+sqrt(0.2))/2, D[1][0]=(sqrt(1.8)-sqrt(0.2))/2.
        let d00 = f64::midpoint(f64::sqrt(1.8), f64::sqrt(0.2));
        let d10 = (f64::sqrt(1.8) - f64::sqrt(0.2)) / 2.0;
        let mut noise0 = [1.0_f64, 0.0];
        dc.apply_correlation(0, &mut noise0, &entity_order);
        assert!(
            (noise0[0] - d00).abs() < 1e-8,
            "stage0 noise0[0]={}",
            noise0[0]
        );
        assert!(
            (noise0[1] - d10).abs() < 1e-8,
            "stage0 noise0[1]={}",
            noise0[1]
        );

        // Stage 1 uses "default" (identity): z=[1,0] -> [1.0, 0.0].
        let mut noise1 = [1.0_f64, 0.0];
        dc.apply_correlation(1, &mut noise1, &entity_order);
        assert!(
            (noise1[0] - 1.0).abs() < 1e-12,
            "stage1 noise1[0]={}",
            noise1[0]
        );
        assert!(
            (noise1[1] - 0.0).abs() < 1e-12,
            "stage1 noise1[1]={}",
            noise1[1]
        );
    }

    // -----------------------------------------------------------------------
    // Same-type entity validation tests
    // -----------------------------------------------------------------------

    fn mixed_type_group(name: &str) -> CorrelationGroup {
        // One "inflow" entity and one "load" entity — intentionally invalid.
        CorrelationGroup {
            name: name.to_string(),
            entities: vec![
                CorrelationEntity {
                    entity_type: "inflow".to_string(),
                    id: EntityId(1),
                },
                CorrelationEntity {
                    entity_type: "load".to_string(),
                    id: EntityId(2),
                },
            ],
            matrix: vec![vec![1.0, 0.5], vec![0.5, 1.0]],
        }
    }

    #[test]
    fn test_build_rejects_mixed_entity_types() {
        let model = single_profile_model("default", vec![mixed_type_group("mixed_group")]);
        let result = DecomposedCorrelation::build(&model);
        match result {
            Err(StochasticError::InvalidCorrelation { reason, .. }) => {
                assert!(
                    reason.contains("mixed entity types"),
                    "expected 'mixed entity types' in reason, got: {reason}"
                );
            }
            other => panic!("expected InvalidCorrelation, got: {other:?}"),
        }
    }

    #[test]
    fn test_build_accepts_same_type_entities() {
        // All "inflow" — must succeed without error.
        let model = single_profile_model("default", vec![identity_group("g1", &[1, 2, 3])]);
        assert!(
            DecomposedCorrelation::build(&model).is_ok(),
            "same-type group should be accepted"
        );
    }

    #[test]
    fn test_build_accepts_single_entity_group() {
        // A single-entity group is trivially homogeneous.
        let model = single_profile_model("default", vec![identity_group("g1", &[1])]);
        assert!(
            DecomposedCorrelation::build(&model).is_ok(),
            "single-entity group should be accepted"
        );
    }

    #[test]
    fn test_build_mixed_type_error_includes_group_name() {
        let model = single_profile_model("default", vec![mixed_type_group("mixed_group")]);
        let result = DecomposedCorrelation::build(&model);
        match result {
            Err(StochasticError::InvalidCorrelation {
                profile_name,
                reason,
            }) => {
                assert_eq!(
                    profile_name, "default",
                    "expected profile_name 'default', got: {profile_name}"
                );
                assert!(
                    reason.contains("mixed_group"),
                    "expected group name 'mixed_group' in reason, got: {reason}"
                );
            }
            other => panic!("expected InvalidCorrelation, got: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // apply_correlation_for_class tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_correlation_for_class_inflow_only() {
        // Inflow group [EntityId(1), EntityId(2)] with rho=0.8 and a single-entity
        // load group [EntityId(3)].
        // Spectral factor of [[1,0.8],[0.8,1]]:
        //   D[0][0] = D[1][1] = (sqrt(1.8) + sqrt(0.2)) / 2 ~= 0.894427
        //   D[0][1] = D[1][0] = (sqrt(1.8) - sqrt(0.2)) / 2 ~= 0.447214
        // z=[1.0, 0.0] => result=[D[0][0], D[1][0]] ~= [0.894427, 0.447214].
        let inflow_group = make_group_with_type("inflow_g", &[1, 2], 0.8, "inflow");
        let load_group = make_group_with_type("load_g", &[3], 0.0, "load");
        let model = single_profile_model("default", vec![inflow_group, load_group]);
        let dc = DecomposedCorrelation::build(&model).unwrap();

        let class_order = [EntityId(1), EntityId(2)];
        let mut inflow_noise = [1.0_f64, 0.0];
        dc.apply_correlation_for_class(0, &mut inflow_noise, &class_order, "inflow");

        let d00 = f64::midpoint(f64::sqrt(1.8), f64::sqrt(0.2));
        let d10 = (f64::sqrt(1.8) - f64::sqrt(0.2)) / 2.0;
        assert!(
            (inflow_noise[0] - d00).abs() < 1e-8,
            "inflow_noise[0]={} (expected {d00})",
            inflow_noise[0]
        );
        assert!(
            (inflow_noise[1] - d10).abs() < 1e-8,
            "inflow_noise[1]={} (expected {d10})",
            inflow_noise[1]
        );
    }

    #[test]
    fn test_apply_correlation_for_class_skips_other_types() {
        // Same setup as above: inflow group [1,2] with rho=0.8, load group [3].
        // Calling with entity_type="load" and load_noise=[5.0] should leave noise
        // unchanged (the single-entity identity group is a no-op AND this also
        // verifies that the inflow group is skipped entirely).
        let inflow_group = make_group_with_type("inflow_g", &[1, 2], 0.8, "inflow");
        let load_group = make_group_with_type("load_g", &[3], 0.0, "load");
        let model = single_profile_model("default", vec![inflow_group, load_group]);
        let dc = DecomposedCorrelation::build(&model).unwrap();

        let class_order = [EntityId(3)];
        let mut load_noise = [5.0_f64];
        dc.apply_correlation_for_class(0, &mut load_noise, &class_order, "load");

        // Identity group on a single entity leaves noise unchanged.
        assert!(
            (load_noise[0] - 5.0).abs() < 1e-12,
            "load_noise[0]={} (expected 5.0)",
            load_noise[0]
        );
    }

    #[test]
    fn test_apply_correlation_for_class_no_matching_groups() {
        // Only inflow groups exist; calling with entity_type="ncs" must be a no-op.
        let inflow_group = make_group_with_type("inflow_g", &[1, 2], 0.8, "inflow");

        let model = single_profile_model("default", vec![inflow_group]);
        let dc = DecomposedCorrelation::build(&model).unwrap();

        let class_order: [EntityId; 0] = [];
        let mut noise: [f64; 0] = [];
        dc.apply_correlation_for_class(0, &mut noise, &class_order, "ncs");
        // No panic, no modification — test passes if we reach here.
    }

    #[test]
    fn test_group_factor_stores_entity_type() {
        // Verify that each GroupFactor carries the correct entity_type string from
        // the input CorrelationGroup.
        let inflow_group = make_group_with_type("g_inflow", &[1, 2], 0.0, "inflow");
        let load_group = make_group_with_type("g_load", &[3], 0.0, "load");
        let ncs_group = make_group_with_type("g_ncs", &[4], 0.0, "ncs");
        let model = single_profile_model("default", vec![inflow_group, load_group, ncs_group]);
        let dc = DecomposedCorrelation::build(&model).unwrap();

        let group_factors = dc.factors.get("default").unwrap();
        assert_eq!(group_factors.len(), 3);

        // BTreeMap preserves insertion order of the profile vec — the groups
        // appear in the order they were pushed during build().
        let types: Vec<&str> = group_factors
            .iter()
            .map(|gf| gf.entity_type.as_str())
            .collect();
        assert!(
            types.contains(&"inflow"),
            "expected 'inflow' group factor, got: {types:?}"
        );
        assert!(
            types.contains(&"load"),
            "expected 'load' group factor, got: {types:?}"
        );
        assert!(
            types.contains(&"ncs"),
            "expected 'ncs' group factor, got: {types:?}"
        );
    }
}
