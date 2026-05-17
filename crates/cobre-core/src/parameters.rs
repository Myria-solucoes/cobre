//! Scalar parameter types for user-defined and computed coefficients.
//!
//! This module defines the in-memory representation of scalar parameters that
//! users or the system can attach to entities and reference from constraint
//! expressions. A [`ScalarParameter`] bundles a unique identifier, a human-readable
//! name, and a [`ParameterKind`] that describes how the numeric value is
//! determined at solve time.
//!
//! The parameter system is deliberately stratified:
//!
//! - **Resolution** (mapping kinds to concrete `f64` values) belongs to a
//!   dedicated resolver layer in the solver crate.
//! - **Loading** (reading parquet files, validating IDs and lengths) belongs to
//!   the I/O layer.
//! - **Consumption** (substituting resolved values into LP rows) belongs to the
//!   LP builder in the solver crate.
//!
//! This module defines only the structural types — it contains no I/O, no
//! resolution logic, and no LP wiring.
//!
//! # Future work
//!
//! The [`ComputedParameter`] enum currently supports seven hydro-indexed
//! quantities. Two duration-based variants — `Computed(BlockDuration)` and
//! `Computed(StageDuration)` — are deferred. When added, they will carry no
//! entity ID (they are study-level scalars), so a new variant shape will be
//! required.

use crate::EntityId;

/// Reference to a scalar coefficient in a linear term.
///
/// Allows constraint coefficients to be either a literal `f64` value known at
/// input time, or a named parameter whose value is resolved later. The `Parameter`
/// variant carries the [`EntityId`] of a [`ScalarParameter`] stored in the
/// study's parameter collection.
///
/// # Examples
///
/// ```
/// use cobre_core::{CoefficientRef, EntityId};
///
/// // A literal coefficient of 3.6:
/// let literal = CoefficientRef::Literal(3.6);
///
/// // A parameter-backed coefficient referencing parameter ID 42:
/// let param = CoefficientRef::Parameter(EntityId(42));
///
/// // CoefficientRef is Copy, so it can be used after moving:
/// let a = CoefficientRef::Literal(1.0);
/// let b = a;
/// assert_eq!(a, b);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CoefficientRef {
    /// A constant scalar value known at input parse time.
    Literal(f64),
    /// An indirect reference to a [`ScalarParameter`] by its entity ID.
    Parameter(EntityId),
}

/// A Cobre-computed quantity indexed by hydro plant.
///
/// Each variant names one of the seven scalar quantities that the resolver
/// derives from hydro geometry and operational data. All variants carry a
/// single `hydro_id` field identifying the hydro plant.
///
/// # Examples
///
/// ```
/// use cobre_core::{ComputedParameter, EntityId};
///
/// let param = ComputedParameter::EquivalentProductivity {
///     hydro_id: EntityId(1),
/// };
///
/// // ComputedParameter is Copy:
/// let copy = param;
/// assert_eq!(param, copy);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ComputedParameter {
    /// Equivalent productivity coefficient (`ρ_eq`).
    EquivalentProductivity {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Accumulated productivity coefficient (`ρ_acum`).
    AccumulatedProductivity {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Reference reservoir volume (`V_ref`).
    ReferenceVolume {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Reference turbine flow (`q_ref`).
    ReferenceTurbine {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Minimum operational storage (`V_min`).
    MinStorage {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Maximum operational storage (`V_max`).
    MaxStorage {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Specific productivity (`ρ_esp`).
    SpecificProductivity {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
}

/// How the numeric value of a [`ScalarParameter`] is determined at solve time.
///
/// The four variants cover the full range from compile-time constants to
/// values computed from physical plant data:
///
/// - [`Constant`](ParameterKind::Constant) — one value for all stages.
/// - [`PerStage`](ParameterKind::PerStage) — one value per study stage;
///   the resolver validates `len() == n_stages`.
/// - [`Seasonal`](ParameterKind::Seasonal) — one value per season, keyed by
///   `season_id` (`i32`). Use [`ParameterKind::new_seasonal`] to construct
///   this variant with the sort-and-dedup invariant enforced.
/// - [`Computed`](ParameterKind::Computed) — derived by the resolver from
///   hydro geometry data; no explicit user value is required.
///
/// # Examples
///
/// ```
/// use cobre_core::{ComputedParameter, EntityId, ParameterKind};
///
/// let constant = ParameterKind::Constant(3.6);
/// let per_stage = ParameterKind::PerStage(vec![1.0, 2.0, 3.0]);
/// let seasonal = ParameterKind::new_seasonal(vec![(2, 1.5), (1, 0.5)]);
/// let computed = ParameterKind::Computed(ComputedParameter::EquivalentProductivity {
///     hydro_id: EntityId(1),
/// });
///
/// // Seasonal entries are sorted ascending by season_id:
/// assert_eq!(
///     seasonal,
///     ParameterKind::Seasonal(vec![(1, 0.5), (2, 1.5)])
/// );
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParameterKind {
    /// A single scalar value applied to every stage.
    Constant(f64),
    /// One scalar value per study stage; length must equal `n_stages`.
    PerStage(Vec<f64>),
    /// One scalar value per season, keyed by `season_id` (`i32`).
    ///
    /// Entries are stored sorted ascending by `season_id` with unique keys.
    /// Construct via [`ParameterKind::new_seasonal`] to enforce this invariant,
    /// or supply a pre-sorted, deduplicated vector directly.
    Seasonal(Vec<(i32, f64)>),
    /// A value derived from physical plant data by the resolver.
    Computed(ComputedParameter),
}

impl ParameterKind {
    /// Construct a [`ParameterKind::Seasonal`] from an unsorted, possibly
    /// duplicate list of `(season_id, value)` pairs.
    ///
    /// The constructor sorts the pairs ascending by `season_id` and removes
    /// duplicate keys, keeping the **first** occurrence of each key. Downstream
    /// loaders are responsible for reporting duplicate-key errors to the user;
    /// this constructor only guarantees the invariant that the internal
    /// `Vec<(i32, f64)>` is sorted and contains unique keys.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::ParameterKind;
    ///
    /// // Duplicates: key 1 appears twice; first occurrence (0.5) is kept.
    /// let seasonal = ParameterKind::new_seasonal(vec![(3, 1.5), (1, 0.5), (1, 0.9), (2, 1.0)]);
    /// assert_eq!(
    ///     seasonal,
    ///     ParameterKind::Seasonal(vec![(1, 0.5), (2, 1.0), (3, 1.5)])
    /// );
    /// ```
    #[must_use]
    pub fn new_seasonal(mut pairs: Vec<(i32, f64)>) -> Self {
        pairs.sort_by_key(|(k, _)| *k);
        pairs.dedup_by_key(|(k, _)| *k);
        Self::Seasonal(pairs)
    }
}

/// A named scalar parameter whose value is resolved before LP construction.
///
/// Parameters are identified by a unique [`EntityId`] and a human-readable
/// `name`. The `kind` field describes how the numeric value is determined at
/// solve time (see [`ParameterKind`]).
///
/// Parameters are loaded from parquet files by the I/O layer and resolved to
/// concrete `f64` values by the resolver before the LP builder consumes them.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, ParameterKind, ScalarParameter};
///
/// let param = ScalarParameter {
///     id: EntityId(1),
///     name: "rho_eq_h1".to_string(),
///     kind: ParameterKind::Constant(3.6),
/// };
///
/// assert_eq!(param.id, EntityId(1));
/// assert_eq!(param.name, "rho_eq_h1");
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScalarParameter {
    /// Unique parameter identifier.
    pub id: EntityId,
    /// Short name used in reports and log output.
    pub name: String,
    /// How the numeric value of this parameter is determined at solve time.
    pub kind: ParameterKind,
}

#[cfg(test)]
mod tests {
    use super::{CoefficientRef, ComputedParameter, EntityId, ParameterKind, ScalarParameter};

    #[test]
    fn seven_computed_parameter_variants() {
        let variants = [
            ComputedParameter::EquivalentProductivity {
                hydro_id: EntityId(1),
            },
            ComputedParameter::AccumulatedProductivity {
                hydro_id: EntityId(2),
            },
            ComputedParameter::ReferenceVolume {
                hydro_id: EntityId(3),
            },
            ComputedParameter::ReferenceTurbine {
                hydro_id: EntityId(4),
            },
            ComputedParameter::MinStorage {
                hydro_id: EntityId(5),
            },
            ComputedParameter::MaxStorage {
                hydro_id: EntityId(6),
            },
            ComputedParameter::SpecificProductivity {
                hydro_id: EntityId(7),
            },
        ];

        assert_eq!(
            variants.len(),
            7,
            "ComputedParameter must have exactly 7 variants"
        );

        // Exhaustive match — no _ arm — compile error if a variant is added without updating here.
        for variant in &variants {
            let _name = match variant {
                ComputedParameter::EquivalentProductivity { .. } => "EquivalentProductivity",
                ComputedParameter::AccumulatedProductivity { .. } => "AccumulatedProductivity",
                ComputedParameter::ReferenceVolume { .. } => "ReferenceVolume",
                ComputedParameter::ReferenceTurbine { .. } => "ReferenceTurbine",
                ComputedParameter::MinStorage { .. } => "MinStorage",
                ComputedParameter::MaxStorage { .. } => "MaxStorage",
                ComputedParameter::SpecificProductivity { .. } => "SpecificProductivity",
            };
        }
    }

    #[test]
    fn parameter_kind_four_variants() {
        let variants = [
            ParameterKind::Constant(1.0),
            ParameterKind::PerStage(vec![1.0, 2.0]),
            ParameterKind::Seasonal(vec![(1, 0.5)]),
            ParameterKind::Computed(ComputedParameter::EquivalentProductivity {
                hydro_id: EntityId(1),
            }),
        ];

        // Exhaustive match — no _ arm — compile error if a variant is added without updating here.
        for variant in &variants {
            let _name = match variant {
                ParameterKind::Constant(_) => "Constant",
                ParameterKind::PerStage(_) => "PerStage",
                ParameterKind::Seasonal(_) => "Seasonal",
                ParameterKind::Computed(_) => "Computed",
            };
        }
    }

    #[test]
    fn seasonal_constructor_sorts_and_dedups() {
        // Key 1 appears twice; first occurrence (0.5) must be kept.
        let input = vec![(3, 1.5), (1, 0.5), (1, 0.9), (2, 1.0)];
        let result = ParameterKind::new_seasonal(input);
        assert_eq!(
            result,
            ParameterKind::Seasonal(vec![(1, 0.5), (2, 1.0), (3, 1.5)])
        );
    }

    #[test]
    fn coefficient_ref_copy_semantics() {
        let a = CoefficientRef::Literal(1.0);
        let b = a;
        assert_eq!(a, b);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn scalar_parameter_seasonal_serde_roundtrip() {
        let param = ScalarParameter {
            id: EntityId(7),
            name: "rho_acum_h1".to_string(),
            kind: ParameterKind::Seasonal(vec![(1, 0.5), (2, 1.0)]),
        };

        let json = serde_json::to_string(&param).unwrap();
        let deserialized: ScalarParameter = serde_json::from_str(&json).unwrap();
        assert_eq!(param, deserialized);
    }
}
