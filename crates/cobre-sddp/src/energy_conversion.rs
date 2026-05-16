//! Energy-conversion preprocessing: derives the per-`(hydro, stage)` scalars
//! `ρ_eq` (equivalent productivity, MW per m³/s), `V_ref` (reference reservoir
//! volume, hm³), `Q_ref` (reference turbined flow, m³/s), and `ρ_acum`
//! (accumulated cascade productivity, MW per m³/s).
//!
//! The output [`EnergyConversionSet`] is consumed by simulation extraction
//! and by the energy-balance constraints that compute ENA and EARM.
//!
//! ## Ownership of derivation logic
//!
//! This module owns the data layout and the placeholder builder. The actual
//! derivation lives in sibling tickets:
//!
//! - non-FPHA `ρ_eq` derivation populates `equivalent_productivity_mw_per_m3s`
//!   from the model's stored scalar.
//! - FPHA `ρ_eq` derivation evaluates `ρ_esp · h_eq(V_ref, Q_ref)` against the
//!   reservoir VHA curve.
//! - `ρ_acum` derivation walks the cascade in topological order summing
//!   downstream equivalent productivities.

use std::collections::HashMap;

use cobre_core::{CascadeTopology, EntityId, Hydro, HydroGenerationModel};
use cobre_io::{HydroGeometryRow, HydroReferenceVolumeFractions};
use thiserror::Error;

use crate::fpha_fitting::{ForebayTable, evaluate_losses, evaluate_tailrace};

/// Per-`(hydro, stage)` scalars used for ENA / EARM accounting.
///
/// All three values are scalar reductions of the underlying production model
/// at a representative operating point (`V_ref`, `Q_ref`). The triple is
/// `Copy` so it can be sliced cheaply on hot paths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyConversion {
    /// Equivalent productivity `ρ_eq` \[MW / (m³/s)\]. Aggregates head and
    /// efficiency into a single scalar at the reference operating point.
    pub equivalent_productivity_mw_per_m3s: f64,
    /// Reference reservoir storage `V_ref` \[hm³\] used to evaluate
    /// `h_eq(V_ref, Q_ref)`. Typically `V_min + fraction · (V_max − V_min)`
    /// with `fraction` resolved per-`(hydro, season)`.
    pub reference_volume_hm3: f64,
    /// Reference turbined flow `Q_ref` \[m³/s\] used to evaluate `h_eq`.
    /// Typically the installed turbine capacity.
    pub reference_outflow_m3s: f64,
}

/// Indexed grid of `(EnergyConversion, ρ_acum)` per `(hydro, stage)`.
///
/// Indexing convention mirrors [`ProductionModelSet`](crate::hydro_models::ProductionModelSet):
/// the outer dimension is the hydro plant, the inner dimension is the stage.
/// All accesses are `O(1)`.
#[derive(Debug, Clone)]
pub struct EnergyConversionSet {
    per_hydro_stage: Vec<Vec<EnergyConversion>>,
    accumulated: Vec<Vec<f64>>,
    n_hydros: usize,
    n_stages: usize,
}

impl EnergyConversionSet {
    /// Build an [`EnergyConversionSet`] from raw grids.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if the outer dimensions do not match
    /// `n_hydros` or any inner row has length `!= n_stages`.
    #[must_use]
    pub fn new(
        per_hydro_stage: Vec<Vec<EnergyConversion>>,
        accumulated: Vec<Vec<f64>>,
        n_hydros: usize,
        n_stages: usize,
    ) -> Self {
        debug_assert_eq!(
            per_hydro_stage.len(),
            n_hydros,
            "per_hydro_stage outer length must equal n_hydros"
        );
        debug_assert_eq!(
            accumulated.len(),
            n_hydros,
            "accumulated outer length must equal n_hydros"
        );
        debug_assert!(
            per_hydro_stage.iter().all(|row| row.len() == n_stages),
            "each per_hydro_stage row must have length n_stages"
        );
        debug_assert!(
            accumulated.iter().all(|row| row.len() == n_stages),
            "each accumulated row must have length n_stages"
        );
        Self {
            per_hydro_stage,
            accumulated,
            n_hydros,
            n_stages,
        }
    }

    /// Return the [`EnergyConversion`] for `(hydro, stage)`.
    ///
    /// # Panics
    ///
    /// In debug builds, panics on out-of-range indices.
    #[must_use]
    pub fn conversion(&self, hydro: usize, stage: usize) -> &EnergyConversion {
        debug_assert!(
            hydro < self.n_hydros,
            "hydro index {hydro} out of bounds (n_hydros = {})",
            self.n_hydros
        );
        debug_assert!(
            stage < self.n_stages,
            "stage index {stage} out of bounds (n_stages = {})",
            self.n_stages
        );
        &self.per_hydro_stage[hydro][stage]
    }

    /// Return the accumulated productivity `ρ_acum` for `(hydro, stage)`.
    ///
    /// # Panics
    ///
    /// In debug builds, panics on out-of-range indices.
    #[must_use]
    pub fn accumulated_productivity(&self, hydro: usize, stage: usize) -> f64 {
        debug_assert!(
            hydro < self.n_hydros,
            "hydro index {hydro} out of bounds (n_hydros = {})",
            self.n_hydros
        );
        debug_assert!(
            stage < self.n_stages,
            "stage index {stage} out of bounds (n_stages = {})",
            self.n_stages
        );
        self.accumulated[hydro][stage]
    }

    /// Number of hydro plants (outer grid dimension).
    #[must_use]
    pub fn n_hydros(&self) -> usize {
        self.n_hydros
    }

    /// Number of stages (inner grid dimension).
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }
}

/// Errors raised by [`build_energy_conversion_set`] and successor derivations.
#[derive(Debug, Error)]
pub enum EnergyConversionError {
    /// Dimensions of input data did not match each other.
    #[error("energy-conversion dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Description of the expected dimension.
        expected: String,
        /// Description of the actual dimension.
        got: String,
    },
    /// `max_storage_hm3 < min_storage_hm3` for a hydro plant.
    #[error(
        "hydro {hydro_id:?} has invalid storage range: max_storage_hm3={v_max} < min_storage_hm3={v_min}"
    )]
    InvalidStorageRange {
        /// Identifier of the offending hydro.
        hydro_id: EntityId,
        /// Stored minimum reservoir volume \[hm³\].
        v_min: f64,
        /// Stored maximum reservoir volume \[hm³\].
        v_max: f64,
    },
    /// `max_turbined_m3s` is negative.
    #[error("hydro {hydro_id:?} has negative max_turbined_m3s={q_max}")]
    NegativeMaxTurbined {
        /// Identifier of the offending hydro.
        hydro_id: EntityId,
        /// Stored maximum turbined flow \[m³/s\].
        q_max: f64,
    },
    /// FPHA equivalent head `h_eq = h_fore − h_tail − h_loss` is non-positive,
    /// which would yield a non-physical `ρ_eq`.
    #[error("hydro {hydro_id:?} has non-positive equivalent head h_eq={h_eq}")]
    NonPositiveEquivalentHead {
        /// Identifier of the offending hydro.
        hydro_id: EntityId,
        /// Computed equivalent head \[m\].
        h_eq: f64,
    },
    /// Building a `ForebayTable` from the VHA rows failed validation.
    #[error("hydro {hydro_id:?} forebay table construction failed: {message}")]
    ForebayTableInvalid {
        /// Identifier of the offending hydro.
        hydro_id: EntityId,
        /// Human-readable description of the underlying FPHA-fitting error.
        /// The concrete error type is intentionally kept crate-private.
        message: String,
    },
}

/// Build the [`EnergyConversionSet`] for the case.
///
/// Populates `equivalent_productivity_mw_per_m3s`, `reference_volume_hm3`, and
/// `reference_outflow_m3s` for hydros whose generation model is
/// [`HydroGenerationModel::ConstantProductivity`] or
/// [`HydroGenerationModel::LinearizedHead`]. FPHA hydros receive populated
/// `V_ref` / `Q_ref` but `ρ_eq = 0.0` here; the FPHA-specific derivation lands
/// in a later step. `accumulated` (`ρ_acum`) is zero-initialized and filled by
/// the cascade walk in a later step.
///
/// # Errors
///
/// Returns [`EnergyConversionError::InvalidStorageRange`] when a hydro has
/// `max_storage_hm3 < min_storage_hm3`, [`EnergyConversionError::NegativeMaxTurbined`]
/// when `max_turbined_m3s < 0`, [`EnergyConversionError::ForebayTableInvalid`]
/// when VHA rows for an FPHA hydro do not form a valid forebay table, and
/// [`EnergyConversionError::NonPositiveEquivalentHead`] when the derived
/// `h_eq` for an FPHA hydro is non-positive.
#[allow(clippy::missing_errors_doc, unused_variables)]
pub fn build_energy_conversion_set<S: std::hash::BuildHasher>(
    hydros: &[Hydro],
    n_stages: usize,
    cascade: &CascadeTopology,
    reference_volume_fractions: &HydroReferenceVolumeFractions,
    vha_rows_by_hydro: &HashMap<EntityId, Vec<HydroGeometryRow>, S>,
) -> Result<EnergyConversionSet, EnergyConversionError> {
    let n_hydros = hydros.len();

    let mut per_hydro_stage: Vec<Vec<EnergyConversion>> = Vec::with_capacity(n_hydros);

    for hydro in hydros {
        let v_min = hydro.min_storage_hm3;
        let v_max = hydro.max_storage_hm3;
        let q_max = hydro.max_turbined_m3s;

        if v_max < v_min {
            return Err(EnergyConversionError::InvalidStorageRange {
                hydro_id: hydro.id,
                v_min,
                v_max,
            });
        }
        if q_max < 0.0 {
            return Err(EnergyConversionError::NegativeMaxTurbined {
                hydro_id: hydro.id,
                q_max,
            });
        }

        // Build the ForebayTable once per hydro (independent of stage) so the
        // per-stage loop only does lookups.
        let forebay_table = if matches!(hydro.generation_model, HydroGenerationModel::Fpha) {
            match (
                vha_rows_by_hydro.get(&hydro.id),
                hydro.specific_productivity_mw_per_m3s_per_m,
            ) {
                (Some(rows), Some(_)) => {
                    Some(ForebayTable::new(rows, &hydro.name).map_err(|e| {
                        EnergyConversionError::ForebayTableInvalid {
                            hydro_id: hydro.id,
                            message: e.to_string(),
                        }
                    })?)
                }
                _ => None,
            }
        } else {
            None
        };

        let mut row: Vec<EnergyConversion> = Vec::with_capacity(n_stages);
        for stage in 0..n_stages {
            let fraction = reference_volume_fractions.get(hydro.id, stage);
            let mut conversion = derive_conversion_for_hydro(hydro, fraction);
            if let (HydroGenerationModel::Fpha, Some(table), Some(rho_esp)) = (
                &hydro.generation_model,
                forebay_table.as_ref(),
                hydro.specific_productivity_mw_per_m3s_per_m,
            ) {
                let h_eq = fpha_equivalent_head(
                    hydro,
                    conversion.reference_volume_hm3,
                    conversion.reference_outflow_m3s,
                    table,
                )?;
                conversion.equivalent_productivity_mw_per_m3s = rho_esp * h_eq;
            }
            row.push(conversion);
        }
        per_hydro_stage.push(row);
    }

    let accumulated = vec![vec![0.0_f64; n_stages]; n_hydros];
    Ok(EnergyConversionSet::new(
        per_hydro_stage,
        accumulated,
        n_hydros,
        n_stages,
    ))
}

/// Derive the per-`(hydro, fraction)` [`EnergyConversion`] cell.
///
/// `fraction` is the resolved reference-volume fraction for the
/// `(hydro, stage)` of interest, already obtained from the resolver. For FPHA
/// hydros, `equivalent_productivity_mw_per_m3s` is left at `0.0` here — it is
/// filled in by the FPHA-specific derivation in `build_energy_conversion_set`.
fn derive_conversion_for_hydro(hydro: &Hydro, fraction: f64) -> EnergyConversion {
    let v_min = hydro.min_storage_hm3;
    let v_max = hydro.max_storage_hm3;
    let reference_volume_hm3 = v_min + fraction * (v_max - v_min);
    let reference_outflow_m3s = hydro.max_turbined_m3s;
    let equivalent_productivity_mw_per_m3s = match hydro.generation_model {
        HydroGenerationModel::ConstantProductivity {
            productivity_mw_per_m3s,
        }
        | HydroGenerationModel::LinearizedHead {
            productivity_mw_per_m3s,
        } => productivity_mw_per_m3s,
        HydroGenerationModel::Fpha => 0.0,
    };
    EnergyConversion {
        equivalent_productivity_mw_per_m3s,
        reference_volume_hm3,
        reference_outflow_m3s,
    }
}

/// Compute the FPHA equivalent head `h_eq = h_fore(V_ref) − h_tail(Q_ref) − h_loss`.
///
/// `h_tail` is taken from `hydro.tailrace` when present (else 0.0), and
/// `h_loss` from `hydro.hydraulic_losses` (matched explicitly so new variants
/// trip the compiler). Returns [`EnergyConversionError::NonPositiveEquivalentHead`]
/// when the result is `<= 0.0` (which would yield a non-physical `ρ_eq`).
fn fpha_equivalent_head(
    hydro: &Hydro,
    v_ref: f64,
    q_ref: f64,
    table: &ForebayTable,
) -> Result<f64, EnergyConversionError> {
    let h_fore = table.height(v_ref);
    let h_tail = hydro
        .tailrace
        .as_ref()
        .map_or(0.0, |t| evaluate_tailrace(t, q_ref));
    let gross_head = h_fore - h_tail;
    let h_loss = hydro
        .hydraulic_losses
        .as_ref()
        .map_or(0.0, |m| evaluate_losses(m, gross_head, q_ref));
    let h_eq = h_fore - h_tail - h_loss;
    if h_eq <= 0.0 {
        return Err(EnergyConversionError::NonPositiveEquivalentHead {
            hydro_id: hydro.id,
            h_eq,
        });
    }
    Ok(h_eq)
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::unwrap_used
)]
mod tests {
    use cobre_core::{
        CascadeTopology, EntityId, HydraulicLossesModel, Hydro, HydroGenerationModel,
        HydroPenalties,
    };
    use cobre_io::{
        HydroGeometryRow, HydroReferenceVolumeFractions, build_hydro_reference_volume_fractions,
    };

    use super::*;

    fn penalties_zero() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            fpha_turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    fn make_hydro(id: i32, downstream: Option<i32>) -> Hydro {
        Hydro {
            id: EntityId::from(id),
            name: format!("Hydro {id}"),
            bus_id: EntityId::from(1),
            downstream_id: downstream.map(EntityId::from),
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 1.0,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 45.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: penalties_zero(),
        }
    }

    fn make_resolver(hydros: &[Hydro]) -> HydroReferenceVolumeFractions {
        build_hydro_reference_volume_fractions(Vec::new(), 0.65, hydros, &[0, 1])
            .expect("resolver builds")
    }

    #[test]
    fn new_round_trips_grid_dimensions() {
        let grid = vec![
            vec![
                EnergyConversion {
                    equivalent_productivity_mw_per_m3s: 0.5,
                    reference_volume_hm3: 100.0,
                    reference_outflow_m3s: 50.0,
                },
                EnergyConversion {
                    equivalent_productivity_mw_per_m3s: 0.6,
                    reference_volume_hm3: 110.0,
                    reference_outflow_m3s: 55.0,
                },
                EnergyConversion {
                    equivalent_productivity_mw_per_m3s: 0.7,
                    reference_volume_hm3: 120.0,
                    reference_outflow_m3s: 60.0,
                },
            ],
            vec![
                EnergyConversion {
                    equivalent_productivity_mw_per_m3s: 1.0,
                    reference_volume_hm3: 200.0,
                    reference_outflow_m3s: 80.0,
                },
                EnergyConversion {
                    equivalent_productivity_mw_per_m3s: 1.1,
                    reference_volume_hm3: 210.0,
                    reference_outflow_m3s: 85.0,
                },
                EnergyConversion {
                    equivalent_productivity_mw_per_m3s: 1.2,
                    reference_volume_hm3: 220.0,
                    reference_outflow_m3s: 90.0,
                },
            ],
        ];
        let acc = vec![vec![10.0, 11.0, 12.0], vec![20.0, 21.0, 22.0]];
        let set = EnergyConversionSet::new(grid.clone(), acc.clone(), 2, 3);

        assert_eq!(set.n_hydros(), 2);
        assert_eq!(set.n_stages(), 3);
        for (h, row) in grid.iter().enumerate() {
            for (s, expected) in row.iter().enumerate() {
                assert_eq!(set.conversion(h, s), expected);
            }
        }
    }

    #[test]
    fn accessors_return_correct_cell() {
        let grid = vec![
            vec![EnergyConversion {
                equivalent_productivity_mw_per_m3s: 0.5,
                reference_volume_hm3: 100.0,
                reference_outflow_m3s: 50.0,
            }],
            vec![EnergyConversion {
                equivalent_productivity_mw_per_m3s: 0.9,
                reference_volume_hm3: 180.0,
                reference_outflow_m3s: 70.0,
            }],
        ];
        let acc = vec![vec![3.5_f64], vec![2.5_f64]];
        let set = EnergyConversionSet::new(grid, acc, 2, 1);

        assert_eq!(set.conversion(0, 0).equivalent_productivity_mw_per_m3s, 0.5);
        assert_eq!(set.conversion(1, 0).reference_outflow_m3s, 70.0);
        assert_eq!(set.accumulated_productivity(0, 0), 3.5);
        assert_eq!(set.accumulated_productivity(1, 0), 2.5);
    }

    #[test]
    fn builder_returns_grid_with_expected_dimensions() {
        let hydros = vec![make_hydro(1, Some(2)), make_hydro(2, None)];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = make_resolver(&hydros);

        let set = build_energy_conversion_set(&hydros, 2, &cascade, &resolver, &HashMap::new())
            .expect("builder succeeds");

        assert_eq!(set.n_hydros(), 2);
        assert_eq!(set.n_stages(), 2);
        // accumulated remains zero-initialized until the cascade-walk step.
        for h in 0..2 {
            for s in 0..2 {
                assert_eq!(set.accumulated_productivity(h, s), 0.0);
            }
        }
    }

    fn make_hydro_with(
        id: i32,
        model: HydroGenerationModel,
        v_min: f64,
        v_max: f64,
        q_max: f64,
        specific: Option<f64>,
    ) -> Hydro {
        let mut h = make_hydro(id, None);
        h.generation_model = model;
        h.min_storage_hm3 = v_min;
        h.max_storage_hm3 = v_max;
        h.max_turbined_m3s = q_max;
        h.specific_productivity_mw_per_m3s_per_m = specific;
        h
    }

    fn constant_resolver(
        hydros: &[Hydro],
        fraction: f64,
        n_stages: usize,
    ) -> HydroReferenceVolumeFractions {
        let rows = hydros
            .iter()
            .map(|h| cobre_io::HydroReferenceVolumeFractionRow {
                hydro_id: h.id,
                season_id: None,
                fraction,
            })
            .collect();
        build_hydro_reference_volume_fractions(rows, 0.65, hydros, &vec![0_i32; n_stages])
            .expect("resolver builds")
    }

    #[test]
    fn constant_productivity_yields_input_scalar() {
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.9,
            },
            100.0,
            200.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 2);

        let set = build_energy_conversion_set(&hydros, 2, &cascade, &resolver, &HashMap::new())
            .expect("builder succeeds");

        let c = set.conversion(0, 0);
        assert_eq!(c.equivalent_productivity_mw_per_m3s, 0.9);
        assert_eq!(c.reference_volume_hm3, 100.0 + 0.65 * (200.0 - 100.0));
        assert_eq!(c.reference_outflow_m3s, 50.0);
    }

    #[test]
    fn linearized_head_yields_input_scalar() {
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::LinearizedHead {
                productivity_mw_per_m3s: 1.2,
            },
            100.0,
            200.0,
            40.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.5, 1);

        let set = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &HashMap::new())
            .expect("builder succeeds");

        assert_eq!(set.conversion(0, 0).equivalent_productivity_mw_per_m3s, 1.2);
    }

    #[test]
    fn reference_volume_uses_fraction() {
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 1.0,
            },
            100.0,
            200.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);

        for f in [0.1_f64, 0.5, 1.0] {
            let resolver = constant_resolver(&hydros, f, 1);
            let set = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &HashMap::new())
                .expect("builder succeeds");
            let expected = 100.0 + f * (200.0 - 100.0);
            assert!(
                (set.conversion(0, 0).reference_volume_hm3 - expected).abs() < f64::EPSILON,
                "fraction {f}: V_ref expected {expected}, got {}",
                set.conversion(0, 0).reference_volume_hm3
            );
        }
    }

    #[test]
    fn per_season_override_produces_different_v_ref_per_stage() {
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.9,
            },
            100.0,
            200.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        // 4 stages alternating between season 0 and season 1.
        let stage_to_season = vec![0_i32, 1, 0, 1];
        let rows = vec![
            cobre_io::HydroReferenceVolumeFractionRow {
                hydro_id: hydros[0].id,
                season_id: Some(0),
                fraction: 0.50,
            },
            cobre_io::HydroReferenceVolumeFractionRow {
                hydro_id: hydros[0].id,
                season_id: Some(1),
                fraction: 0.70,
            },
        ];
        let resolver =
            build_hydro_reference_volume_fractions(rows, 0.65, &hydros, &stage_to_season)
                .expect("resolver builds");
        let set = build_energy_conversion_set(&hydros, 4, &cascade, &resolver, &HashMap::new())
            .expect("builder succeeds");

        let expected = [150.0_f64, 170.0, 150.0, 170.0];
        for (s, want) in expected.iter().enumerate() {
            assert!(
                (set.conversion(0, s).reference_volume_hm3 - want).abs() < f64::EPSILON,
                "stage {s}: expected V_ref {want}, got {}",
                set.conversion(0, s).reference_volume_hm3
            );
            assert_eq!(set.conversion(0, s).equivalent_productivity_mw_per_m3s, 0.9);
        }
    }

    #[test]
    fn invalid_storage_range_is_rejected() {
        let hydros = vec![make_hydro_with(
            42,
            HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 1.0,
            },
            200.0,
            100.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);

        let err = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &HashMap::new())
            .unwrap_err();
        match err {
            EnergyConversionError::InvalidStorageRange { hydro_id, .. } => {
                assert_eq!(hydro_id, hydros[0].id);
            }
            other => panic!("expected InvalidStorageRange, got: {other:?}"),
        }
    }

    #[test]
    fn negative_max_turbined_is_rejected() {
        let hydros = vec![make_hydro_with(
            42,
            HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 1.0,
            },
            100.0,
            200.0,
            -1.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);

        let err = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &HashMap::new())
            .unwrap_err();
        match err {
            EnergyConversionError::NegativeMaxTurbined { hydro_id, q_max } => {
                assert_eq!(hydro_id, hydros[0].id);
                assert_eq!(q_max, -1.0);
            }
            other => panic!("expected NegativeMaxTurbined, got: {other:?}"),
        }
    }

    #[test]
    fn fpha_hydro_leaves_rho_eq_zero_but_populates_v_ref() {
        let hydros = vec![make_hydro_with(
            5,
            HydroGenerationModel::Fpha,
            100.0,
            200.0,
            50.0,
            Some(0.01),
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.5, 1);
        let set = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &HashMap::new())
            .expect("builder succeeds for FPHA");

        let c = set.conversion(0, 0);
        assert_eq!(c.equivalent_productivity_mw_per_m3s, 0.0);
        assert_eq!(c.reference_volume_hm3, 150.0);
        assert_eq!(c.reference_outflow_m3s, 50.0);
    }

    // ── ticket-005 FPHA derivation tests ───────────────────────────────────

    /// A VHA table designed so that `table.height(V_ref) = expected_h_fore`
    /// for the V_ref produced by `make_hydro_with(...) + fraction=0.65`. The
    /// linear interpolation between (V_min=100, h=300) and (V_max=200, h=465)
    /// yields h(165) = 300 + 0.65 * (465 - 300) = 300 + 107.25 = 407.25, but
    /// we want exactly 400.0 at V=165, so use:
    /// (100, 360) and (200, 440) -> h(165) = 360 + 0.65 * 80 = 412.0 — also
    /// not exact. To get exactly 400 at v=165, set h(100)=295 and h(200)=460:
    /// h(165) = 295 + 0.65 * 165 = 295 + 107.25 = 402.25 — still off.
    /// Easiest: use a flat table h(v) = 400 for all v (two points with same height).
    fn vha_constant_height(hydro_id: EntityId, height: f64) -> (EntityId, Vec<HydroGeometryRow>) {
        let rows = vec![
            HydroGeometryRow {
                hydro_id,
                volume_hm3: 0.0,
                height_m: height,
                area_km2: 1.0,
            },
            HydroGeometryRow {
                hydro_id,
                volume_hm3: 1000.0,
                height_m: height,
                area_km2: 1.0,
            },
        ];
        (hydro_id, rows)
    }

    fn fpha_hydro_for_tests(id: i32) -> Hydro {
        make_hydro_with(
            id,
            HydroGenerationModel::Fpha,
            100.0,
            200.0,
            50.0,
            Some(0.0090),
        )
    }

    #[test]
    fn fpha_rho_eq_from_vha_no_tailrace_no_losses() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let (id, rows) = vha_constant_height(hydros[0].id, 400.0);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let set = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &map)
            .expect("builder succeeds");

        let got = set.conversion(0, 0).equivalent_productivity_mw_per_m3s;
        let expected = 0.0090 * 400.0;
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn fpha_rho_eq_with_factor_losses() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.tailrace = None;
        hydro.hydraulic_losses = Some(HydraulicLossesModel::Factor { value: 0.05 });
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let (id, rows) = vha_constant_height(hydros[0].id, 400.0);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let set = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &map)
            .expect("builder succeeds");

        // h_eq = 400 - 0 - 0.05 * 400 = 380; rho_eq = 0.009 * 380 = 3.42
        let got = set.conversion(0, 0).equivalent_productivity_mw_per_m3s;
        let expected = 0.0090 * 380.0;
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn fpha_rho_eq_with_constant_losses() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.tailrace = None;
        hydro.hydraulic_losses = Some(HydraulicLossesModel::Constant { value_m: 5.0 });
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let (id, rows) = vha_constant_height(hydros[0].id, 400.0);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let set = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &map)
            .expect("builder succeeds");

        // h_eq = 400 - 0 - 5 = 395; rho_eq = 0.009 * 395 = 3.555
        let got = set.conversion(0, 0).equivalent_productivity_mw_per_m3s;
        let expected = 0.0090 * 395.0;
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn fpha_missing_rho_esp_yields_zero() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.specific_productivity_mw_per_m3s_per_m = None;
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let (id, rows) = vha_constant_height(hydros[0].id, 400.0);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let set = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &map)
            .expect("builder succeeds");

        assert_eq!(set.conversion(0, 0).equivalent_productivity_mw_per_m3s, 0.0);
    }

    #[test]
    fn fpha_missing_vha_yields_zero() {
        let hydro = fpha_hydro_for_tests(7); // has rho_esp = Some(0.009)
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);

        let set = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &HashMap::new())
            .expect("builder succeeds");

        assert_eq!(set.conversion(0, 0).equivalent_productivity_mw_per_m3s, 0.0);
    }

    #[test]
    fn fpha_rejects_non_positive_h_eq() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.tailrace = None;
        // Constant loss equal to h_fore -> h_eq = 0 -> non-positive -> Err.
        hydro.hydraulic_losses = Some(HydraulicLossesModel::Constant { value_m: 400.0 });
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let (id, rows) = vha_constant_height(hydros[0].id, 400.0);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let err = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &map).unwrap_err();
        match err {
            EnergyConversionError::NonPositiveEquivalentHead { hydro_id, h_eq } => {
                assert_eq!(hydro_id, hydros[0].id);
                assert!(h_eq <= 0.0);
            }
            other => panic!("expected NonPositiveEquivalentHead, got: {other:?}"),
        }
    }

    #[test]
    fn fpha_propagates_forebay_table_error() {
        // 1-row VHA fails ForebayTable::new (InsufficientPoints).
        let hydro = fpha_hydro_for_tests(7);
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let rows = vec![HydroGeometryRow {
            hydro_id: hydros[0].id,
            volume_hm3: 0.0,
            height_m: 400.0,
            area_km2: 1.0,
        }];
        let mut map = HashMap::new();
        map.insert(hydros[0].id, rows);

        let err = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &map).unwrap_err();
        match err {
            EnergyConversionError::ForebayTableInvalid { hydro_id, .. } => {
                assert_eq!(hydro_id, hydros[0].id);
            }
            other => panic!("expected ForebayTableInvalid, got: {other:?}"),
        }
    }
}
