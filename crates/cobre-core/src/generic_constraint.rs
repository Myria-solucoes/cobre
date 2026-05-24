//! User-defined generic linear constraints.
//!
//! This module defines the in-memory representation of generic constraints
//! that users can specify to add custom linear relationships between LP
//! variables. The expression parser (string → [`ConstraintExpression`])
//! lives in `cobre-io`, not here. This module contains only the output types.
//!
//! See `internal-structures.md §15` and `input-constraints.md §3` for the
//! full specification, grammar, and validation rules.
//!
//! # Variable Reference Catalog
//!
//! [`VariableRef`] covers all 20 LP variable types defined in the spec (SS15).
//! Each variant carries the entity ID and, for block-specific variables, an
//! optional block ID (`None` = sum over all blocks, `Some(i)` = block `i`).
//!
//! # Examples
//!
//! ```
//! use cobre_core::{
//!     EntityId, GenericConstraint, ConstraintExpression, ConstraintSense,
//!     LinearTerm, SlackConfig, VariableRef,
//! };
//!
//! // Represents: hydro_generation(10) + hydro_generation(11)
//! let expr = ConstraintExpression {
//!     terms: vec![
//!         LinearTerm::literal(1.0, VariableRef::HydroGeneration {
//!             hydro_id: EntityId(10),
//!             block_id: None,
//!         }),
//!         LinearTerm::literal(1.0, VariableRef::HydroGeneration {
//!             hydro_id: EntityId(11),
//!             block_id: None,
//!         }),
//!     ],
//! };
//!
//! assert_eq!(expr.terms.len(), 2);
//!
//! let gc = GenericConstraint {
//!     id: EntityId(0),
//!     name: "min_southeast_hydro".to_string(),
//!     description: Some("Minimum hydro generation in Southeast region".to_string()),
//!     expression: expr,
//!     sense: ConstraintSense::GreaterEqual,
//!     slack: SlackConfig { enabled: true, penalty: Some(5_000.0) },
//! };
//!
//! assert_eq!(gc.expression.terms.len(), 2);
//! ```

use crate::CoefficientRef;
use crate::EntityId;

/// Reference to a single LP variable in a generic constraint expression.
///
/// Each variant names the variable type and carries the entity ID. For
/// block-specific variables, `block_id` is `None` to sum over all blocks or
/// `Some(i)` to reference block `i` specifically.
///
/// The 20 variants cover the full variable catalog defined in
/// `internal-structures.md §15` (table in the "Variable References" section).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VariableRef {
    /// Reservoir storage level for a hydro plant (stage-level, not block-specific).
    HydroStorage {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Turbined water flow for a hydro plant (m³/s).
    HydroTurbined {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Spillage flow for a hydro plant (m³/s).
    HydroSpillage {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Diversion flow for a hydro plant (m³/s). Only valid for hydros with diversion.
    HydroDiversion {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Total outflow (turbined + spillage) for a hydro plant (m³/s).
    ///
    /// Currently an alias for turbined + spillage. Future CEPEL formulations
    /// may turn this into an independent variable.
    HydroOutflow {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Electrical generation from a hydro plant (MW).
    HydroGeneration {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Signed evaporation flow from a hydro reservoir (m³/s). Stage-level, not
    /// block-specific. Positive values represent net evaporative outflow;
    /// negative values represent net rainfall input absorbed by the reservoir.
    HydroEvaporation {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Water withdrawal from a hydro reservoir (m³/s). Stage-level, not block-specific.
    HydroWithdrawal {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Electrical generation from a thermal unit (MW).
    ThermalGeneration {
        /// Thermal unit identifier.
        thermal_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Direct (forward) power flow on a transmission line (MW).
    LineDirect {
        /// Transmission line identifier.
        line_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Reverse power flow on a transmission line (MW).
    LineReverse {
        /// Transmission line identifier.
        line_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Net exchange flow on a transmission line (direct - reverse) (MW).
    ///
    /// This is a derived variable: the resolver maps it to two LP columns
    /// (forward flow with +1.0 and reverse flow with -1.0), representing
    /// net flow in the source-to-target direction.
    LineExchange {
        /// Transmission line identifier.
        line_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Load deficit (unserved energy) at a bus (MW).
    BusDeficit {
        /// Bus identifier.
        bus_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Load excess (over-generation) at a bus (MW).
    BusExcess {
        /// Bus identifier.
        bus_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Pumped water flow at a pumping station (m³/s).
    PumpingFlow {
        /// Pumping station identifier.
        station_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Electrical power consumed by a pumping station (MW).
    PumpingPower {
        /// Pumping station identifier.
        station_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Energy imported via a contract (MW).
    ContractImport {
        /// Energy contract identifier.
        contract_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Energy exported via a contract (MW).
    ContractExport {
        /// Energy contract identifier.
        contract_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Generation from a non-controllable source (wind, solar, etc.) (MW).
    NonControllableGeneration {
        /// Non-controllable source identifier.
        source_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Curtailment of a non-controllable source (MW).
    NonControllableCurtailment {
        /// Non-controllable source identifier.
        source_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Forward-commitment decision MW for an anticipated thermal unit (MW).
    ///
    /// References the commitment placed at the current stage `t` for delivery
    /// at stage `t + lead_stages`. This is a per-plant per-stage scalar — it has
    /// **no `block_id`** because the commitment is uniform across blocks.
    ///
    /// The column exists in the LP only for plants whose `anticipated_config`
    /// is `Some(_)`. Referencing this variant for a non-anticipated thermal is
    /// a referential-validation error (see
    /// `cobre-io::validation::referential::validate_variable_ref_entity`).
    ///
    /// The column also has `[0.0, 0.0]` bounds at boundary stages where
    /// `t + K_i >= n_stages` (the F2-002 strict predicate); a constraint
    /// referencing the column at the boundary is structurally a no-op.
    ///
    /// **Postcard wire-format note**: this variant is appended at the END of
    /// the enum to preserve the discriminant indices of all existing variants.
    /// Postcard encodes enum variants by declaration order; inserting in the
    /// middle would break existing serialized policies.
    AnticipatedDecision {
        /// Thermal unit identifier. Must satisfy `anticipated_config: Some(_)`.
        thermal_id: EntityId,
    },
}

/// One term in a linear constraint expression: `coefficient * scale * variable`.
///
/// The LP coefficient is `resolve(coefficient, stage) * scale`, where `resolve`
/// returns the parameter's stage-resolved scalar for
/// `CoefficientRef::Parameter(id)` or the literal value for
/// `CoefficientRef::Literal(v)`. Resolution for `Parameter` variants is
/// implemented by the resolver layer; until then only `Literal` is produced by
/// the parser.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearTerm {
    /// Coefficient reference. `Literal(f64)` carries an inline constant;
    /// `Parameter(EntityId)` references a `ScalarParameter` resolved per
    /// stage at LP build time.
    pub coefficient: CoefficientRef,
    /// Multiplicative scale applied after the coefficient is resolved.
    /// Defaults to `1.0` via `LinearTerm::literal(_, _)` and is set
    /// explicitly when constructing `LinearTerm` literals directly.
    pub scale: f64,
    /// The LP variable being referenced.
    pub variable: VariableRef,
}

impl LinearTerm {
    /// Construct a `LinearTerm` whose coefficient is the literal `coef`,
    /// with `scale = 1.0`. The common case during expression parsing.
    #[must_use]
    pub fn literal(coef: f64, variable: VariableRef) -> Self {
        Self {
            coefficient: CoefficientRef::Literal(coef),
            scale: 1.0,
            variable,
        }
    }

    /// Construct a `LinearTerm` whose coefficient is a named parameter reference.
    ///
    /// The `scale` argument carries the literal multiplier from the expression
    /// (e.g. `2.5` for `"2.5 * @rho_eq * x"`, or `sign` for `"@rho_eq * x"`).
    /// Resolution of the parameter to a concrete `f64` per stage happens at
    /// LP-build time.
    #[must_use]
    pub fn parameter(id: crate::EntityId, scale: f64, variable: VariableRef) -> Self {
        Self {
            coefficient: CoefficientRef::Parameter(id),
            scale,
            variable,
        }
    }
}

/// Parsed linear constraint expression.
///
/// Represents the left-hand side of a generic constraint as a list of weighted
/// variable references. An empty `terms` vector is valid (constant-only
/// expression, unusual but not rejected at this layer).
///
/// The expression parser (string → `ConstraintExpression`) lives in `cobre-io`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConstraintExpression {
    /// Ordered list of linear terms that form the left-hand side of the constraint.
    pub terms: Vec<LinearTerm>,
}

/// Comparison sense for a generic constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ConstraintSense {
    /// The expression must be greater than or equal to the bound (`>=`).
    GreaterEqual,
    /// The expression must be less than or equal to the bound (`<=`).
    LessEqual,
    /// The expression must be exactly equal to the bound (`==`).
    Equal,
}

/// Slack variable configuration for a generic constraint.
///
/// When `enabled` is `true`, a slack variable is added to the LP so that the
/// constraint can be violated at a cost. This prevents infeasibility when
/// bounds are tight or conflicting. The penalty cost enters the LP objective
/// function.
///
/// `penalty` must be `Some(value)` with a positive value when `enabled` is
/// `true`, and `None` when `enabled` is `false`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SlackConfig {
    /// Whether a slack variable is added to allow soft violation of the constraint.
    pub enabled: bool,
    /// Penalty cost per unit of constraint violation. `None` when `enabled` is `false`.
    pub penalty: Option<f64>,
}

/// A user-defined generic linear constraint.
///
/// Stored in [`crate::System::generic_constraints`] after loading and
/// validation. Constraints are sorted by `id` after loading to satisfy the
/// declaration-order invariance requirement.
///
/// The expression parser, referential validation (entity IDs exist), and
/// bounds loading (from `generic_constraint_bounds.parquet`) are all
/// performed by `cobre-io`, not here.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GenericConstraint {
    /// Unique constraint identifier.
    pub id: EntityId,
    /// Short name used in reports and log output.
    pub name: String,
    /// Optional human-readable description.
    pub description: Option<String>,
    /// Parsed left-hand-side expression of the constraint.
    pub expression: ConstraintExpression,
    /// Comparison sense (`>=`, `<=`, or `==`).
    pub sense: ConstraintSense,
    /// Slack variable configuration.
    pub slack: SlackConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_ref_variants() {
        let variants: &[(&str, VariableRef)] = &[
            (
                "HydroStorage",
                VariableRef::HydroStorage {
                    hydro_id: EntityId(0),
                },
            ),
            (
                "HydroTurbined",
                VariableRef::HydroTurbined {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "HydroSpillage",
                VariableRef::HydroSpillage {
                    hydro_id: EntityId(0),
                    block_id: Some(1),
                },
            ),
            (
                "HydroDiversion",
                VariableRef::HydroDiversion {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "HydroOutflow",
                VariableRef::HydroOutflow {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "HydroGeneration",
                VariableRef::HydroGeneration {
                    hydro_id: EntityId(0),
                    block_id: Some(0),
                },
            ),
            (
                "HydroEvaporation",
                VariableRef::HydroEvaporation {
                    hydro_id: EntityId(0),
                },
            ),
            (
                "HydroWithdrawal",
                VariableRef::HydroWithdrawal {
                    hydro_id: EntityId(0),
                },
            ),
            (
                "ThermalGeneration",
                VariableRef::ThermalGeneration {
                    thermal_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "LineDirect",
                VariableRef::LineDirect {
                    line_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "LineReverse",
                VariableRef::LineReverse {
                    line_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "LineExchange",
                VariableRef::LineExchange {
                    line_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "BusDeficit",
                VariableRef::BusDeficit {
                    bus_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "BusExcess",
                VariableRef::BusExcess {
                    bus_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "PumpingFlow",
                VariableRef::PumpingFlow {
                    station_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "PumpingPower",
                VariableRef::PumpingPower {
                    station_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "ContractImport",
                VariableRef::ContractImport {
                    contract_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "ContractExport",
                VariableRef::ContractExport {
                    contract_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "NonControllableGeneration",
                VariableRef::NonControllableGeneration {
                    source_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "NonControllableCurtailment",
                VariableRef::NonControllableCurtailment {
                    source_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "AnticipatedDecision",
                VariableRef::AnticipatedDecision {
                    thermal_id: EntityId(0),
                },
            ),
        ];

        assert_eq!(
            variants.len(),
            21,
            "VariableRef must have exactly 21 variants"
        );

        for (name, variant) in variants {
            let debug_str = format!("{variant:?}");
            assert!(
                debug_str.contains(name),
                "Debug output for {name} does not contain the variant name: {debug_str}"
            );
        }
    }

    #[test]
    fn test_generic_constraint_construction() {
        let expr = ConstraintExpression {
            terms: vec![
                LinearTerm::literal(
                    1.0,
                    VariableRef::HydroGeneration {
                        hydro_id: EntityId(10),
                        block_id: None,
                    },
                ),
                LinearTerm::literal(
                    1.0,
                    VariableRef::HydroGeneration {
                        hydro_id: EntityId(11),
                        block_id: None,
                    },
                ),
            ],
        };

        let gc = GenericConstraint {
            id: EntityId(0),
            name: "min_southeast_hydro".to_string(),
            description: Some("Minimum hydro generation in Southeast region".to_string()),
            expression: expr,
            sense: ConstraintSense::GreaterEqual,
            slack: SlackConfig {
                enabled: true,
                penalty: Some(5_000.0),
            },
        };

        assert_eq!(gc.expression.terms.len(), 2);
        assert_eq!(gc.id, EntityId(0));
        assert_eq!(gc.name, "min_southeast_hydro");
        assert!(gc.description.is_some());
        assert_eq!(gc.sense, ConstraintSense::GreaterEqual);
        assert!(gc.slack.enabled);
        assert_eq!(gc.slack.penalty, Some(5_000.0));
    }

    #[test]
    fn test_slack_config_disabled_has_no_penalty() {
        let slack = SlackConfig {
            enabled: false,
            penalty: None,
        };
        assert!(!slack.enabled);
        assert!(slack.penalty.is_none());
    }

    #[test]
    fn test_constraint_sense_variants() {
        assert_ne!(ConstraintSense::GreaterEqual, ConstraintSense::LessEqual);
        assert_ne!(ConstraintSense::GreaterEqual, ConstraintSense::Equal);
        assert_ne!(ConstraintSense::LessEqual, ConstraintSense::Equal);
    }

    fn lit(term: &LinearTerm) -> f64 {
        match term.coefficient {
            CoefficientRef::Literal(v) => v,
            CoefficientRef::Parameter(_) => panic!("expected literal"),
        }
    }

    #[test]
    fn test_linear_term_with_coefficient() {
        let term = LinearTerm::literal(
            2.5,
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(5),
                block_id: None,
            },
        );
        assert!((lit(&term) - 2.5).abs() < f64::EPSILON);
        let debug = format!("{:?}", term.variable);
        assert!(debug.contains("ThermalGeneration"));
    }

    #[test]
    fn linear_term_literal_constructor() {
        let term = LinearTerm::literal(
            3.0,
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(1),
                block_id: None,
            },
        );
        assert_eq!(term.coefficient, CoefficientRef::Literal(3.0));
        assert!((term.scale - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            term.variable,
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(1),
                block_id: None,
            }
        );
    }

    #[test]
    fn linear_term_explicit_scale() {
        let term = LinearTerm {
            coefficient: CoefficientRef::Literal(2.0),
            scale: 0.5,
            variable: VariableRef::HydroStorage {
                hydro_id: EntityId(1),
            },
        };
        assert!((term.scale - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_variable_ref_block_none_vs_some() {
        let all_blocks = VariableRef::HydroTurbined {
            hydro_id: EntityId(3),
            block_id: None,
        };
        let specific_block = VariableRef::HydroTurbined {
            hydro_id: EntityId(3),
            block_id: Some(0),
        };
        assert_ne!(all_blocks, specific_block);
    }

    // ── AC-1: AnticipatedDecision construction ────────────────────────────────

    /// AC-1: `VariableRef::AnticipatedDecision` constructs successfully.
    #[test]
    fn anticipated_decision_constructs() {
        let v = VariableRef::AnticipatedDecision {
            thermal_id: EntityId(5),
        };
        let debug_str = format!("{v:?}");
        assert!(
            debug_str.contains("AnticipatedDecision"),
            "Debug output should contain variant name: {debug_str}"
        );
        assert!(
            debug_str.contains("thermal_id"),
            "Debug output should contain field name: {debug_str}"
        );
    }

    /// AC-1 (structural): `AnticipatedDecision` is `Copy` and `PartialEq`.
    #[test]
    fn anticipated_decision_copy_and_eq() {
        let v1 = VariableRef::AnticipatedDecision {
            thermal_id: EntityId(5),
        };
        let v2 = v1; // copy
        assert_eq!(v1, v2);

        let v3 = VariableRef::AnticipatedDecision {
            thermal_id: EntityId(9),
        };
        assert_ne!(v1, v3);
    }

    // ── AC-2: Postcard round-trip ─────────────────────────────────────────────

    /// AC-2: Postcard round-trip preserves `AnticipatedDecision` and its `thermal_id`.
    ///
    /// Also pins the discriminant byte so that future variants cannot be
    /// inserted before `AnticipatedDecision` without breaking this assertion
    /// (postcard encodes enum variants by declaration order).
    #[cfg(feature = "serde")]
    #[test]
    fn anticipated_decision_postcard_roundtrip() {
        let original = VariableRef::AnticipatedDecision {
            thermal_id: EntityId(5),
        };
        let bytes = postcard::to_allocvec(&original).expect("serialize");
        let recovered: VariableRef = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(
            original, recovered,
            "postcard round-trip must preserve the variant"
        );

        // Pin the discriminant: AnticipatedDecision is variant 20 (0-indexed).
        // Postcard encodes enum variants as a varint; for index 20 that is 0x14.
        assert_eq!(
            bytes[0], 20,
            "AnticipatedDecision must be discriminant 20 (end-of-enum); \
             got {}. Did you insert a variant before it?",
            bytes[0]
        );
    }

    /// AC-2 (regression): existing variants retain their expected discriminants.
    ///
    /// Smoke-tests that no insertion before `NonControllableCurtailment` shifted
    /// the discriminant of the 20th variant (index 19).
    #[cfg(feature = "serde")]
    #[test]
    fn non_controllable_curtailment_discriminant_is_19() {
        let v = VariableRef::NonControllableCurtailment {
            source_id: EntityId(0),
            block_id: None,
        };
        let bytes = postcard::to_allocvec(&v).expect("serialize");
        assert_eq!(
            bytes[0], 19,
            "NonControllableCurtailment must remain discriminant 19; \
             got {}. A variant was inserted before it.",
            bytes[0]
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_generic_constraint_serde_roundtrip() {
        let gc = GenericConstraint {
            id: EntityId(0),
            name: "test".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![
                    LinearTerm::literal(
                        1.0,
                        VariableRef::HydroGeneration {
                            hydro_id: EntityId(10),
                            block_id: None,
                        },
                    ),
                    LinearTerm::literal(
                        1.0,
                        VariableRef::HydroGeneration {
                            hydro_id: EntityId(11),
                            block_id: None,
                        },
                    ),
                ],
            },
            sense: ConstraintSense::GreaterEqual,
            slack: SlackConfig {
                enabled: true,
                penalty: Some(5_000.0),
            },
        };

        let json = serde_json::to_string(&gc).unwrap();
        let deserialized: GenericConstraint = serde_json::from_str(&json).unwrap();
        assert_eq!(gc, deserialized);
        assert_eq!(deserialized.expression.terms.len(), 2);
    }
}
