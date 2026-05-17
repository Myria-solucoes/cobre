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
//! - **Loading** (reading the JSON file, validating IDs and lengths) belongs to
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
#[cfg_attr(feature = "serde", serde(tag = "tag", rename_all = "snake_case"))]
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
/// # JSON schema
///
/// Serialization uses an internally-tagged JSON form. Each variant produces a
/// `"kind"` discriminant field alongside its payload fields:
///
/// ```
/// # #[cfg(feature = "serde")] {
/// use cobre_core::{ComputedParameter, EntityId, ParameterKind};
///
/// // {"kind":"constant","value":3.6}
/// let c = ParameterKind::Constant { value: 3.6 };
/// assert_eq!(
///     serde_json::to_string(&c).unwrap(),
///     r#"{"kind":"constant","value":3.6}"#
/// );
///
/// // {"kind":"per_stage","values":[[0,1.0],[1,1.1],[2,0.9]]}
/// let ps = ParameterKind::PerStage { values: vec![1.0, 1.1, 0.9] };
/// assert_eq!(
///     serde_json::to_string(&ps).unwrap(),
///     r#"{"kind":"per_stage","values":[[0,1.0],[1,1.1],[2,0.9]]}"#
/// );
///
/// // {"kind":"seasonal","values":[[1,0.5],[2,1.5]]}
/// let s = ParameterKind::Seasonal { values: vec![(1, 0.5), (2, 1.5)] };
/// assert_eq!(
///     serde_json::to_string(&s).unwrap(),
///     r#"{"kind":"seasonal","values":[[1,0.5],[2,1.5]]}"#
/// );
///
/// // {"kind":"computed","computed_spec":{"tag":"equivalent_productivity","hydro_id":7}}
/// let comp = ParameterKind::Computed {
///     computed_spec: ComputedParameter::EquivalentProductivity { hydro_id: EntityId(7) },
/// };
/// assert_eq!(
///     serde_json::to_string(&comp).unwrap(),
///     r#"{"kind":"computed","computed_spec":{"tag":"equivalent_productivity","hydro_id":7}}"#
/// );
/// # }
/// ```
///
/// All four variants round-trip through `serde_json::from_str` back to the same
/// in-memory value.
///
/// # Examples
///
/// ```
/// use cobre_core::{ComputedParameter, EntityId, ParameterKind};
///
/// let constant = ParameterKind::Constant { value: 3.6 };
/// let per_stage = ParameterKind::PerStage { values: vec![1.0, 2.0, 3.0] };
/// let seasonal = ParameterKind::new_seasonal(vec![(2, 1.5), (1, 0.5)]);
/// let computed = ParameterKind::Computed {
///     computed_spec: ComputedParameter::EquivalentProductivity {
///         hydro_id: EntityId(1),
///     },
/// };
///
/// // Seasonal entries are sorted ascending by season_id:
/// assert_eq!(
///     seasonal,
///     ParameterKind::Seasonal { values: vec![(1, 0.5), (2, 1.5)] }
/// );
/// ```
// Serialization: derive Serialize with the `into` shim so the JSON wire format
// uses `ParameterKindJson` (which carries `PerStage` as `[[stage_id, value], ...]`).
// Deserialization: NOT derived here; a manual `impl Deserialize` below applies
// the PerStage contiguity-from-0 validation before accepting the value.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(into = "ParameterKindJson"))]
pub enum ParameterKind {
    /// A single scalar value applied to every stage.
    Constant {
        /// The scalar value for all stages.
        value: f64,
    },
    /// One scalar value per study stage; length must equal `n_stages`.
    PerStage {
        /// Dense array of values indexed by stage (0-based).
        values: Vec<f64>,
    },
    /// One scalar value per season, keyed by `season_id` (`i32`).
    ///
    /// Entries are stored sorted ascending by `season_id` with unique keys.
    /// Construct via [`ParameterKind::new_seasonal`] to enforce this invariant,
    /// or supply a pre-sorted, deduplicated vector directly.
    Seasonal {
        /// Sorted, deduplicated `(season_id, value)` pairs.
        values: Vec<(i32, f64)>,
    },
    /// A value derived from physical plant data by the resolver.
    Computed {
        /// The computed quantity specification.
        computed_spec: ComputedParameter,
    },
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
    ///     ParameterKind::Seasonal { values: vec![(1, 0.5), (2, 1.0), (3, 1.5)] }
    /// );
    /// ```
    #[must_use]
    pub fn new_seasonal(mut pairs: Vec<(i32, f64)>) -> Self {
        pairs.sort_by_key(|(k, _)| *k);
        pairs.dedup_by_key(|(k, _)| *k);
        Self::Seasonal { values: pairs }
    }
}

/// A named scalar parameter whose value is resolved before LP construction.
///
/// Parameters are identified by a unique [`EntityId`] and a human-readable
/// `name`. The `kind` field describes how the numeric value is determined at
/// solve time (see [`ParameterKind`]).
///
/// Parameters are loaded from the JSON case file by the I/O layer and resolved
/// to concrete `f64` values by the resolver before the LP builder consumes them.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, ParameterKind, ScalarParameter};
///
/// let param = ScalarParameter {
///     id: EntityId(1),
///     name: "rho_eq_h1".to_string(),
///     kind: ParameterKind::Constant { value: 3.6 },
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

// ── Serde shim: ParameterKindJson ─────────────────────────────────────────────
//
// `ParameterKind::PerStage` stores a dense `Vec<f64>` in memory but the JSON
// form uses `[[stage_id, value], ...]` pairs.  The other three variants have
// identical in-memory and JSON shapes.  `ParameterKindJson` is a private
// serde-only intermediate that carries `PerStage` as `Vec<(i32, f64)>`.
//
// The `#[serde(into = "ParameterKindJson", from = "ParameterKindJson")]`
// attribute on `ParameterKind` routes all (de)serialization through this shim
// so the public type is never exposed in serde output.

#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ParameterKindJson {
    Constant { value: f64 },
    PerStage { values: Vec<(i32, f64)> },
    Seasonal { values: Vec<(i32, f64)> },
    Computed { computed_spec: ComputedParameter },
}

#[cfg(feature = "serde")]
impl From<ParameterKind> for ParameterKindJson {
    fn from(kind: ParameterKind) -> Self {
        match kind {
            ParameterKind::Constant { value } => ParameterKindJson::Constant { value },
            ParameterKind::PerStage { values } => ParameterKindJson::PerStage {
                values: values
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| {
                        // Stage counts are always < i32::MAX in practice;
                        // saturate rather than panic on a theoretically impossible overflow.
                        (i32::try_from(i).unwrap_or(i32::MAX), v)
                    })
                    .collect(),
            },
            ParameterKind::Seasonal { values } => ParameterKindJson::Seasonal { values },
            ParameterKind::Computed { computed_spec } => {
                ParameterKindJson::Computed { computed_spec }
            }
        }
    }
}

// ── Manual Deserialize impl for ParameterKind ────────────────────────────────
//
// We cannot use `#[serde(from = "ParameterKindJson")]` for deserialization
// because `From<ParameterKindJson> for ParameterKind` cannot return a serde
// error — `From` is infallible.  The PerStage contiguity validation requires
// surfacing an error during deserialization.
//
// The manual impl:
//   1. Deserializes `ParameterKindJson` (which handles the JSON tag + shape).
//   2. Validates the PerStage pairs (sort, check no duplicate keys, check
//      contiguity from 0, then convert to dense Vec<f64>).
//   3. Constructs the appropriate ParameterKind variant.

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for ParameterKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let json = ParameterKindJson::deserialize(deserializer)?;
        match json {
            ParameterKindJson::Constant { value } => Ok(ParameterKind::Constant { value }),
            ParameterKindJson::PerStage { mut values } => {
                // Sort ascending by stage_id for deterministic validation.
                values.sort_by_key(|(k, _)| *k);

                // Check for duplicate stage_ids.
                for window in values.windows(2) {
                    if window[0].0 == window[1].0 {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate stage_id {} in per_stage values",
                            window[0].0
                        )));
                    }
                }

                // Check that stage_ids form a contiguous range starting at 0.
                for (expected, (actual, _)) in values.iter().enumerate() {
                    let expected_i32 = i32::try_from(expected).unwrap_or(i32::MAX);
                    if *actual != expected_i32 {
                        return Err(serde::de::Error::custom(format!(
                            "per_stage values must have contiguous stage_ids starting at 0; \
                             expected stage_id {expected_i32} but got {actual}"
                        )));
                    }
                }

                // Convert to dense Vec<f64>.
                let dense: Vec<f64> = values.into_iter().map(|(_, v)| v).collect();
                Ok(ParameterKind::PerStage { values: dense })
            }
            ParameterKindJson::Seasonal { values } => {
                // Reject duplicate season_ids explicitly before calling new_seasonal,
                // which silently dedups.  Duplicates in authored JSON are errors.
                let mut seen: std::collections::HashSet<i32> = std::collections::HashSet::new();
                for &(season_id, _) in &values {
                    if !seen.insert(season_id) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate season_id {season_id} in seasonal values"
                        )));
                    }
                }
                Ok(ParameterKind::new_seasonal(values))
            }
            ParameterKindJson::Computed { computed_spec } => {
                Ok(ParameterKind::Computed { computed_spec })
            }
        }
    }
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
            ParameterKind::Constant { value: 1.0 },
            ParameterKind::PerStage {
                values: vec![1.0, 2.0],
            },
            ParameterKind::Seasonal {
                values: vec![(1, 0.5)],
            },
            ParameterKind::Computed {
                computed_spec: ComputedParameter::EquivalentProductivity {
                    hydro_id: EntityId(1),
                },
            },
        ];

        // Exhaustive match — no _ arm — compile error if a variant is added without updating here.
        for variant in &variants {
            let _name = match variant {
                ParameterKind::Constant { .. } => "Constant",
                ParameterKind::PerStage { .. } => "PerStage",
                ParameterKind::Seasonal { .. } => "Seasonal",
                ParameterKind::Computed { .. } => "Computed",
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
            ParameterKind::Seasonal {
                values: vec![(1, 0.5), (2, 1.0), (3, 1.5)]
            }
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
            kind: ParameterKind::Seasonal {
                values: vec![(1, 0.5), (2, 1.0)],
            },
        };

        let json = serde_json::to_string(&param).unwrap();
        let deserialized: ScalarParameter = serde_json::from_str(&json).unwrap();
        assert_eq!(param, deserialized);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn parameter_kind_serde_constant_form() {
        let kind = ParameterKind::Constant { value: 3.6 };
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(json, r#"{"kind":"constant","value":3.6}"#);
        let roundtrip: ParameterKind = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip, kind);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn parameter_kind_serde_per_stage_form() {
        let kind = ParameterKind::PerStage {
            values: vec![1.0, 1.1, 0.9],
        };
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(
            json,
            r#"{"kind":"per_stage","values":[[0,1.0],[1,1.1],[2,0.9]]}"#
        );
        let roundtrip: ParameterKind = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip, kind);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn parameter_kind_serde_seasonal_form() {
        let kind = ParameterKind::Seasonal {
            values: vec![(1, 0.5), (2, 1.5)],
        };
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(json, r#"{"kind":"seasonal","values":[[1,0.5],[2,1.5]]}"#);
        let roundtrip: ParameterKind = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip, kind);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn parameter_kind_serde_computed_form() {
        let kind = ParameterKind::Computed {
            computed_spec: ComputedParameter::EquivalentProductivity {
                hydro_id: EntityId(7),
            },
        };
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(
            json,
            r#"{"kind":"computed","computed_spec":{"tag":"equivalent_productivity","hydro_id":7}}"#
        );
        let roundtrip: ParameterKind = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip, kind);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn parameter_kind_per_stage_rejects_non_contiguous_stage_ids() {
        // stage_id 1 is missing — the range [0, 2] is not contiguous.
        let json = r#"{"kind":"per_stage","values":[[0,1.0],[2,0.9]]}"#;
        let result: Result<ParameterKind, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "expected an error for non-contiguous stage_ids, got: {result:?}"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn parameter_kind_seasonal_rejects_duplicate_season_id_via_serde() {
        // season_id 1 appears twice — must be rejected with "duplicate season_id 1".
        let json = r#"{"kind":"seasonal","values":[[1,0.5],[1,0.9],[2,1.0]]}"#;
        let result: Result<ParameterKind, _> = serde_json::from_str(json);
        let err = result.expect_err("expected an error for duplicate season_id");
        assert!(
            err.to_string().contains("duplicate season_id 1"),
            "error message must mention the duplicate id; got: {err}"
        );
    }
}
