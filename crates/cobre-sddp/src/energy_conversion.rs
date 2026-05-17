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

use std::collections::{HashMap, HashSet};

use cobre_core::{CascadeTopology, EntityId, Hydro, HydroGenerationModel};
use cobre_io::{
    HydroEnergyProductivityRow, HydroGeometryRow, HydroReferenceVolumeFractions, LoadError,
};
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

    /// Check whether any non-FPHA hydro's stored `ρ_eq_stored` is inconsistent
    /// with its supplied `ρ_esp` and VHA geometry.
    ///
    /// For each `(hydro, stage)` where:
    /// - `hydro.generation_model` is not [`HydroGenerationModel::Fpha`],
    /// - `hydro.specific_productivity_mw_per_m3s_per_m` is `Some(ρ_esp)`,
    /// - `vha_rows_by_hydro` contains VHA rows for the hydro, and
    /// - the computed `h_eq(V_ref, Q_ref) > 0.0`,
    ///
    /// this method computes `implied_rho_esp = ρ_eq_stored / h_eq` and flags a
    /// [`ConsistencyWarning::ImpliedSpecificProductivityMismatch`] when the
    /// relative divergence `|supplied − implied| / implied.max(1e-30)` exceeds
    /// [`SPECIFIC_PRODUCTIVITY_CONSISTENCY_TOLERANCE`].
    ///
    /// The method is pure (no I/O, no logging) and infallible. Warnings are
    /// returned in declaration order of `hydros`, then ascending stage index.
    ///
    /// # Numerical edge cases
    ///
    /// - `h_eq <= 0.0`: silently skipped (no error — non-FPHA hydros do not
    ///   hard-error on non-positive head during setup).
    /// - NaN `delta`: the `delta > tolerance` comparison is `false` for NaN,
    ///   so NaN propagates no spurious warning.
    #[must_use]
    pub fn consistency_warnings<S: std::hash::BuildHasher>(
        &self,
        hydros: &[Hydro],
        vha_rows_by_hydro: &HashMap<EntityId, Vec<HydroGeometryRow>, S>,
    ) -> Vec<ConsistencyWarning> {
        let mut warnings = Vec::new();

        for (h_idx, hydro) in hydros.iter().enumerate() {
            // Skip FPHA hydros — they have no scalar `ρ_eq` to compare against.
            if matches!(hydro.generation_model, HydroGenerationModel::Fpha) {
                continue;
            }

            // Skip when no supplied ρ_esp.
            let Some(supplied_rho_esp) = hydro.specific_productivity_mw_per_m3s_per_m else {
                continue;
            };

            // Skip when no VHA geometry for this hydro.
            let Some(vha_rows) = vha_rows_by_hydro.get(&hydro.id) else {
                continue;
            };

            // Build the ForebayTable once per hydro; skip if construction fails.
            let Ok(table) = ForebayTable::new(vha_rows, &hydro.name) else {
                continue;
            };

            for stage in 0..self.n_stages {
                let conv = self.conversion(h_idx, stage);
                let v_ref = conv.reference_volume_hm3;
                let q_ref = conv.reference_outflow_m3s;

                // Read ρ_eq from the already-resolved conversion set rather
                // than from the entity model (which no longer carries a scalar).
                let rho_eq_stored = conv.equivalent_productivity_mw_per_m3s;

                // Skip when h_eq <= 0.0 (non-positive head is informational
                // for non-FPHA hydros; no error is raised here).
                let Some(h_eq) = equivalent_head(hydro, &table, v_ref, q_ref) else {
                    continue;
                };

                let implied_rho_esp = rho_eq_stored / h_eq;
                let delta =
                    (supplied_rho_esp - implied_rho_esp).abs() / implied_rho_esp.max(1e-30_f64);

                if delta > SPECIFIC_PRODUCTIVITY_CONSISTENCY_TOLERANCE {
                    let stage_id = i32::try_from(stage).unwrap_or(i32::MAX);
                    warnings.push(ConsistencyWarning::ImpliedSpecificProductivityMismatch {
                        hydro_id: hydro.id,
                        stage_id,
                        supplied_rho_esp,
                        implied_rho_esp,
                        tolerance: SPECIFIC_PRODUCTIVITY_CONSISTENCY_TOLERANCE,
                    });
                }
            }
        }

        warnings
    }
}

/// Per-`(hydro, stage)` override table loaded from
/// `system/hydro_energy_productivity.parquet`.
///
/// Each of the four override columns (`ρ_eq`, `V_ref`, `Q_ref`, `ρ_esp`) is
/// stored in two parallel lookup tables: one for stage-specific rows and one
/// for per-hydro defaults (rows whose `stage_id` is NULL in the source file).
///
/// Accessors apply a per-stage → per-hydro-default → `None` fallback chain.
/// `Default` yields an override where every accessor returns `None`.
#[derive(Debug, Default, Clone)]
pub struct HydroEnergyProductivityOverride {
    rho_eq_per_hydro_stage: HashMap<(EntityId, i32), f64>,
    rho_eq_per_hydro_default: HashMap<EntityId, f64>,
    v_ref_per_hydro_stage: HashMap<(EntityId, i32), f64>,
    v_ref_per_hydro_default: HashMap<EntityId, f64>,
    q_ref_per_hydro_stage: HashMap<(EntityId, i32), f64>,
    q_ref_per_hydro_default: HashMap<EntityId, f64>,
    rho_esp_per_hydro_stage: HashMap<(EntityId, i32), f64>,
    rho_esp_per_hydro_default: HashMap<EntityId, f64>,
}

impl HydroEnergyProductivityOverride {
    /// Returns the user-supplied `ρ_eq` for `(hydro, stage)` if any.
    ///
    /// Looks up the stage-specific entry first, then falls back to the
    /// per-hydro default (NULL `stage_id` rows), then returns `None`.
    /// An out-of-range `stage` (larger than `i32::MAX`) always returns `None`.
    #[must_use]
    pub fn equivalent_productivity(&self, hydro: EntityId, stage: usize) -> Option<f64> {
        let s = i32::try_from(stage).ok()?;
        if let Some(&v) = self.rho_eq_per_hydro_stage.get(&(hydro, s)) {
            return Some(v);
        }
        self.rho_eq_per_hydro_default.get(&hydro).copied()
    }

    /// Returns the user-supplied `V_ref` \[hm³\] for `(hydro, stage)` if any.
    ///
    /// Applies the same per-stage → per-hydro-default → `None` fallback as
    /// [`equivalent_productivity`](Self::equivalent_productivity).
    #[must_use]
    pub fn reference_volume(&self, hydro: EntityId, stage: usize) -> Option<f64> {
        let s = i32::try_from(stage).ok()?;
        if let Some(&v) = self.v_ref_per_hydro_stage.get(&(hydro, s)) {
            return Some(v);
        }
        self.v_ref_per_hydro_default.get(&hydro).copied()
    }

    /// Returns the user-supplied `Q_ref` \[m³/s\] for `(hydro, stage)` if any.
    ///
    /// Applies the same per-stage → per-hydro-default → `None` fallback as
    /// [`equivalent_productivity`](Self::equivalent_productivity).
    #[must_use]
    pub fn reference_outflow(&self, hydro: EntityId, stage: usize) -> Option<f64> {
        let s = i32::try_from(stage).ok()?;
        if let Some(&v) = self.q_ref_per_hydro_stage.get(&(hydro, s)) {
            return Some(v);
        }
        self.q_ref_per_hydro_default.get(&hydro).copied()
    }

    /// Returns the user-supplied `ρ_esp` \[MW/(m³/s)/m\] for `(hydro, stage)` if any.
    ///
    /// Applies the same per-stage → per-hydro-default → `None` fallback as
    /// [`equivalent_productivity`](Self::equivalent_productivity).
    #[must_use]
    pub fn specific_productivity(&self, hydro: EntityId, stage: usize) -> Option<f64> {
        let s = i32::try_from(stage).ok()?;
        if let Some(&v) = self.rho_esp_per_hydro_stage.get(&(hydro, s)) {
            return Some(v);
        }
        self.rho_esp_per_hydro_default.get(&hydro).copied()
    }
}

/// Build a [`HydroEnergyProductivityOverride`] from parsed rows.
///
/// Each non-`None` override column is scattered into the matching per-stage or
/// per-hydro-default lookup table. Duplicate `(hydro_id, stage_id)` pairs
/// (with NULL `stage_id` distinct from any concrete value) are rejected.
///
/// # Errors
///
/// Returns [`LoadError::SchemaError`] with
/// `field = "hydro_energy_productivity.duplicate_entry"` when the same
/// `(hydro_id, stage_id)` key appears more than once in `rows`.
pub fn build_hydro_energy_productivity_override(
    rows: Vec<HydroEnergyProductivityRow>,
) -> Result<HydroEnergyProductivityOverride, LoadError> {
    let mut seen: HashSet<(EntityId, Option<i32>)> = HashSet::with_capacity(rows.len());
    let mut out = HydroEnergyProductivityOverride {
        rho_eq_per_hydro_stage: HashMap::new(),
        rho_eq_per_hydro_default: HashMap::new(),
        v_ref_per_hydro_stage: HashMap::new(),
        v_ref_per_hydro_default: HashMap::new(),
        q_ref_per_hydro_stage: HashMap::new(),
        q_ref_per_hydro_default: HashMap::new(),
        rho_esp_per_hydro_stage: HashMap::new(),
        rho_esp_per_hydro_default: HashMap::new(),
    };

    for row in rows {
        let key = (row.hydro_id, row.stage_id);
        if !seen.insert(key) {
            let stage_label = row
                .stage_id
                .map_or_else(|| "NULL".to_string(), |s| s.to_string());
            return Err(LoadError::SchemaError {
                path: std::path::PathBuf::from("<hydro_energy_productivity>"),
                field: "hydro_energy_productivity.duplicate_entry".to_string(),
                message: format!(
                    "duplicate (hydro_id={}, stage_id={}) key",
                    row.hydro_id.0, stage_label
                ),
            });
        }

        if let Some(s) = row.stage_id {
            if let Some(v) = row.equivalent_productivity_mw_per_m3s {
                out.rho_eq_per_hydro_stage.insert((row.hydro_id, s), v);
            }
            if let Some(v) = row.reference_volume_hm3 {
                out.v_ref_per_hydro_stage.insert((row.hydro_id, s), v);
            }
            if let Some(v) = row.reference_outflow_m3s {
                out.q_ref_per_hydro_stage.insert((row.hydro_id, s), v);
            }
            if let Some(v) = row.specific_productivity_mw_per_m3s_per_m {
                out.rho_esp_per_hydro_stage.insert((row.hydro_id, s), v);
            }
        } else {
            if let Some(v) = row.equivalent_productivity_mw_per_m3s {
                out.rho_eq_per_hydro_default.insert(row.hydro_id, v);
            }
            if let Some(v) = row.reference_volume_hm3 {
                out.v_ref_per_hydro_default.insert(row.hydro_id, v);
            }
            if let Some(v) = row.reference_outflow_m3s {
                out.q_ref_per_hydro_default.insert(row.hydro_id, v);
            }
            if let Some(v) = row.specific_productivity_mw_per_m3s_per_m {
                out.rho_esp_per_hydro_default.insert(row.hydro_id, v);
            }
        }
    }

    Ok(out)
}

/// Relative tolerance for the implied-vs-supplied `ρ_esp` consistency check.
///
/// 5% covers normal numerical noise from VHA interpolation while still
/// catching the order-of-magnitude divergences that real data-entry bugs
/// produce.
pub const SPECIFIC_PRODUCTIVITY_CONSISTENCY_TOLERANCE: f64 = 0.05;

/// A diagnostic warning produced by [`EnergyConversionSet::consistency_warnings`].
///
/// Warnings are purely informational — the case still loads and runs when any
/// of these are present. They flag data-entry discrepancies that are impossible
/// to miss on stderr.
///
/// # Future resolution
///
/// A future user-facing suppression mechanism (e.g. an
/// `expect_inconsistent_productivity_layers: bool` field on the
/// `hydro_energy_productivity.parquet` row schema) is planned but not yet
/// implemented. Warnings are currently emitted unconditionally.
#[derive(Debug, Clone, PartialEq)]
pub enum ConsistencyWarning {
    /// The `ρ_esp` supplied on the `Hydro` field diverges from the value
    /// implied by `ρ_eq_stored / h_eq` (for non-FPHA hydros) by more than
    /// [`SPECIFIC_PRODUCTIVITY_CONSISTENCY_TOLERANCE`].
    ImpliedSpecificProductivityMismatch {
        /// Identifier of the hydro whose supplied and implied `ρ_esp` diverge.
        hydro_id: EntityId,
        /// Stage index (0-based) for which the divergence was detected.
        stage_id: i32,
        /// The `ρ_esp` value supplied on `Hydro.specific_productivity_mw_per_m3s_per_m`.
        supplied_rho_esp: f64,
        /// The `ρ_esp` implied by `ρ_eq_stored / h_eq(V_ref, Q_ref)`.
        implied_rho_esp: f64,
        /// The relative tolerance threshold that was exceeded.
        tolerance: f64,
    },
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
    /// The cascade topology contains a different number of entries than the
    /// `hydros` slice, indicating that the topology was built from a different
    /// set of hydros.
    #[error("cascade topological order length {got} does not match hydros slice length {expected}")]
    CascadeIndexMismatch {
        /// Number of hydros in the `hydros` slice (expected).
        expected: usize,
        /// Number of entries in `cascade.topological_order()` (actual).
        got: usize,
    },
    /// A downstream reference in the cascade points to an `EntityId` that is
    /// not present in the `hydros` slice.
    ///
    /// This indicates a malformed cascade built from a hydro set that does not
    /// match the one passed to `build_energy_conversion_set`.
    #[error(
        "cascade has dangling downstream reference: hydro {hydro_id:?} points to {downstream_id:?} which is not in the hydros slice"
    )]
    DanglingDownstream {
        /// The hydro whose `downstream_id` is dangling.
        hydro_id: EntityId,
        /// The downstream `EntityId` that has no matching entry.
        downstream_id: EntityId,
    },
    /// An FPHA hydro has no way to derive `ρ_eq`: no VHA geometry, no
    /// `ρ_esp`, and no entry in the override table.
    ///
    /// To fix: supply VHA geometry rows and `ρ_esp` for the hydro,
    /// or add an entry in `system/hydro_energy_productivity.parquet`,
    /// or change the hydro's `generation_model` away from FPHA.
    #[error(
        "FPHA hydro '{hydro_name}' ({hydro_id:?}) cannot derive ρ_eq for stage {stage}: \
        no VHA geometry + ρ_esp pair is present and no override entry exists. \
        Remediation: (1) supply VHA geometry rows and specific_productivity (ρ_esp) for this hydro, \
        (2) add an entry in system/hydro_energy_productivity.parquet, \
        or (3) change the hydro's generation_model away from FPHA."
    )]
    FphaMissingEquivalentProductivity {
        /// Identifier of the offending FPHA hydro.
        hydro_id: EntityId,
        /// Name of the offending FPHA hydro, for user-facing diagnostics.
        hydro_name: String,
        /// First stage index for which `ρ_eq` could not be derived.
        stage: usize,
    },
}

/// Build the [`EnergyConversionSet`] for the case.
///
/// Populates `equivalent_productivity_mw_per_m3s`, `reference_volume_hm3`, and
/// `reference_outflow_m3s` for every hydro and stage. For FPHA hydros, `ρ_eq`
/// is resolved from three sources in priority order:
///
/// 1. `override_table` — a per-`(hydro, stage)` user-supplied value from the
///    optional `system/hydro_energy_productivity.parquet` rows.
/// 2. VHA geometry + `ρ_esp` — derived via `ρ_esp · h_eq(V_ref, Q_ref)`.
/// 3. Neither — returns [`EnergyConversionError::FphaMissingEquivalentProductivity`].
///
/// For non-FPHA hydros, `ρ_eq` is read from `production_models` (the per-stage
/// resolved production model, populated from `hydro_production_models.json`).
/// Pass `None` to use `0.0` as a placeholder (only appropriate in tests that do
/// not require accurate `equivalent_productivity_mw_per_m3s` values).
///
/// `accumulated` (`ρ_acum`) is zero-initialized and then filled by the cascade
/// walk before returning.
///
/// # Errors
///
/// - [`EnergyConversionError::InvalidStorageRange`] — `max_storage_hm3 < min_storage_hm3`.
/// - [`EnergyConversionError::NegativeMaxTurbined`] — `max_turbined_m3s < 0`.
/// - [`EnergyConversionError::ForebayTableInvalid`] — VHA rows fail forebay-table validation.
/// - [`EnergyConversionError::NonPositiveEquivalentHead`] — derived `h_eq ≤ 0`.
/// - [`EnergyConversionError::FphaMissingEquivalentProductivity`] — FPHA hydro has no
///   usable `ρ_eq` source for a given stage and no override entry.
/// - [`EnergyConversionError::CascadeIndexMismatch`] — cascade built from different hydro set.
/// - [`EnergyConversionError::DanglingDownstream`] — dangling downstream reference.
#[allow(clippy::missing_errors_doc)]
pub fn build_energy_conversion_set<S: std::hash::BuildHasher>(
    hydros: &[Hydro],
    n_stages: usize,
    cascade: &CascadeTopology,
    reference_volume_fractions: &HydroReferenceVolumeFractions,
    vha_rows_by_hydro: &HashMap<EntityId, Vec<HydroGeometryRow>, S>,
    override_table: Option<&HydroEnergyProductivityOverride>,
    production_models: Option<&crate::hydro_models::ProductionModelSet>,
) -> Result<EnergyConversionSet, EnergyConversionError> {
    let n_hydros = hydros.len();

    let mut per_hydro_stage: Vec<Vec<EnergyConversion>> = Vec::with_capacity(n_hydros);

    for (h_idx, hydro) in hydros.iter().enumerate() {
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

        // For FPHA hydros, try to derive ρ_eq from VHA + ρ_esp once per hydro
        // (independent of stage) so the per-stage loop only does lookups.
        //
        // rho_eq_opt is Some(value) only when both VHA rows and ρ_esp are
        // present and the forebay table validates. For non-FPHA hydros this
        // path is skipped entirely.
        let fpha_derivation = if matches!(hydro.generation_model, HydroGenerationModel::Fpha) {
            match (
                vha_rows_by_hydro.get(&hydro.id),
                hydro.specific_productivity_mw_per_m3s_per_m,
            ) {
                (Some(rows), Some(rho_esp)) => {
                    let table = ForebayTable::new(rows, &hydro.name).map_err(|e| {
                        EnergyConversionError::ForebayTableInvalid {
                            hydro_id: hydro.id,
                            message: e.to_string(),
                        }
                    })?;
                    Some((table, rho_esp))
                }
                _ => None,
            }
        } else {
            None
        };

        let mut row: Vec<EnergyConversion> = Vec::with_capacity(n_stages);
        for stage in 0..n_stages {
            let fraction = reference_volume_fractions.get(hydro.id, stage);
            // For non-FPHA hydros, read ρ_eq from the resolved per-stage
            // production model. For FPHA hydros, use 0.0 as a placeholder
            // (the FPHA branch below overwrites it).
            let productivity = if matches!(hydro.generation_model, HydroGenerationModel::Fpha) {
                0.0
            } else {
                production_models.map_or(0.0, |pm| match pm.model(h_idx, stage) {
                    crate::hydro_models::ResolvedProductionModel::ConstantProductivity {
                        productivity,
                    } => *productivity,
                    crate::hydro_models::ResolvedProductionModel::Fpha { .. } => 0.0,
                })
            };
            let mut conversion = derive_conversion_for_hydro(hydro, fraction, productivity);

            if matches!(hydro.generation_model, HydroGenerationModel::Fpha) {
                // Try override first, then VHA derivation, then error.
                let rho_eq = if let Some(value) =
                    override_table.and_then(|o| o.equivalent_productivity(hydro.id, stage))
                {
                    value
                } else if let Some((ref table, rho_esp)) = fpha_derivation {
                    let h_eq = fpha_equivalent_head(
                        hydro,
                        conversion.reference_volume_hm3,
                        conversion.reference_outflow_m3s,
                        table,
                    )?;
                    rho_esp * h_eq
                } else {
                    return Err(EnergyConversionError::FphaMissingEquivalentProductivity {
                        hydro_id: hydro.id,
                        hydro_name: hydro.name.clone(),
                        stage,
                    });
                };
                conversion.equivalent_productivity_mw_per_m3s = rho_eq;
            }

            row.push(conversion);
        }
        per_hydro_stage.push(row);
    }

    // Validate that the cascade topology was built from the same hydro set.
    // topological_order() contains every hydro reachable from the graph; if a
    // hydro was declared with a downstream_id that points outside the slice the
    // topology length will differ from n_hydros.
    let topo_len = cascade.topological_order().len();
    if topo_len != n_hydros {
        return Err(EnergyConversionError::CascadeIndexMismatch {
            expected: n_hydros,
            got: topo_len,
        });
    }

    // Build a lookup from EntityId to slice index once, outside the stage loop,
    // so the inner loop is O(1) per entry.
    let mut id_to_index: HashMap<EntityId, usize> = HashMap::with_capacity(n_hydros);
    for (idx, h) in hydros.iter().enumerate() {
        id_to_index.insert(h.id, idx);
    }

    // Walk the topological order in reverse (downstream before upstream) so
    // that when we process hydro i, the accumulated value for its downstream
    // plant is already fully computed.
    let mut accumulated = vec![vec![0.0_f64; n_stages]; n_hydros];
    for t in 0..n_stages {
        for id in cascade.topological_order().iter().rev() {
            let h_idx =
                *id_to_index
                    .get(id)
                    .ok_or(EnergyConversionError::CascadeIndexMismatch {
                        expected: n_hydros,
                        got: topo_len,
                    })?;
            let rho_eq = per_hydro_stage[h_idx][t].equivalent_productivity_mw_per_m3s;
            let downstream_contrib = if let Some(ds_id) = cascade.downstream(*id) {
                let ds_idx =
                    *id_to_index
                        .get(&ds_id)
                        .ok_or(EnergyConversionError::DanglingDownstream {
                            hydro_id: *id,
                            downstream_id: ds_id,
                        })?;
                accumulated[ds_idx][t]
            } else {
                0.0
            };
            accumulated[h_idx][t] = rho_eq + downstream_contrib;
        }
    }

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
/// hydros, `productivity` should be `0.0` — it is filled in by the
/// FPHA-specific derivation in `build_energy_conversion_set`. For non-FPHA
/// hydros, `productivity` is the per-stage value from the resolved production
/// model (i.e., from `hydro_production_models.json`).
fn derive_conversion_for_hydro(
    hydro: &Hydro,
    fraction: f64,
    productivity: f64,
) -> EnergyConversion {
    let v_min = hydro.min_storage_hm3;
    let v_max = hydro.max_storage_hm3;
    let reference_volume_hm3 = v_min + fraction * (v_max - v_min);
    let reference_outflow_m3s = hydro.max_turbined_m3s;
    EnergyConversion {
        equivalent_productivity_mw_per_m3s: productivity,
        reference_volume_hm3,
        reference_outflow_m3s,
    }
}

/// Compute `h_eq = h_fore(V_ref) − h_tail(Q_ref) − h_loss` for a hydro plant.
///
/// Returns `None` when `h_eq <= 0.0` so callers that treat non-positive head as
/// a silent skip (e.g., [`EnergyConversionSet::consistency_warnings`]) do not need
/// separate guard logic. Callers that must surface the non-positive case as an
/// error use [`fpha_equivalent_head`] instead.
fn equivalent_head(hydro: &Hydro, table: &ForebayTable, v_ref: f64, q_ref: f64) -> Option<f64> {
    let h_fore = table.height(v_ref);
    let h_tail = hydro
        .tailrace
        .as_ref()
        .map_or(0.0, |t| evaluate_tailrace(t, q_ref));
    let h_loss = hydro
        .hydraulic_losses
        .as_ref()
        .map_or(0.0, |m| evaluate_losses(m, h_fore - h_tail, q_ref));
    let h_eq = h_fore - h_tail - h_loss;
    (h_eq > 0.0).then_some(h_eq)
}

/// Compute the FPHA equivalent head `h_eq = h_fore(V_ref) − h_tail(Q_ref) − h_loss`.
///
/// Returns [`EnergyConversionError::NonPositiveEquivalentHead`] when the result
/// is `<= 0.0` (which would yield a non-physical `ρ_eq`).
fn fpha_equivalent_head(
    hydro: &Hydro,
    v_ref: f64,
    q_ref: f64,
    table: &ForebayTable,
) -> Result<f64, EnergyConversionError> {
    if let Some(h_eq) = equivalent_head(hydro, table, v_ref, q_ref) {
        Ok(h_eq)
    } else {
        let h_fore = table.height(v_ref);
        let h_tail = hydro
            .tailrace
            .as_ref()
            .map_or(0.0, |t| evaluate_tailrace(t, q_ref));
        let h_loss = hydro
            .hydraulic_losses
            .as_ref()
            .map_or(0.0, |m| evaluate_losses(m, h_fore - h_tail, q_ref));
        Err(EnergyConversionError::NonPositiveEquivalentHead {
            hydro_id: hydro.id,
            h_eq: h_fore - h_tail - h_loss,
        })
    }
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

    use crate::hydro_models::{ProductionModelSet, ResolvedProductionModel};

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
            generation_model: HydroGenerationModel::ConstantProductivity,
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
        // hydro id=1 (downstream=2) and hydro id=2 (terminal), both ρ_eq=1.0.
        // After the cascade walk:
        //   ρ_acum(id=2) = 1.0            (terminal, no downstream contrib)
        //   ρ_acum(id=1) = 1.0 + 1.0 = 2.0
        let n_stages = 2;
        let hydros = vec![make_hydro(1, Some(2)), make_hydro(2, None)];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = make_resolver(&hydros);
        let pm = production_set(&[1.0, 1.0], n_stages);

        let set = build_energy_conversion_set(
            &hydros,
            n_stages,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        assert_eq!(set.n_hydros(), 2);
        assert_eq!(set.n_stages(), n_stages);
        // hydro index 0 = id=1 (upstream), index 1 = id=2 (terminal).
        for s in 0..n_stages {
            assert_eq!(set.accumulated_productivity(1, s), 1.0); // id=2, terminal
            assert_eq!(set.accumulated_productivity(0, s), 2.0); // id=1, upstream of id=2
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

    /// Build a `ProductionModelSet` where every (hydro, stage) cell uses
    /// `ConstantProductivity` with the given per-hydro productivity values.
    ///
    /// `productivities[h]` is the productivity for hydro at declaration index `h`.
    fn production_set(productivities: &[f64], n_stages: usize) -> ProductionModelSet {
        let n_hydros = productivities.len();
        let models = productivities
            .iter()
            .map(|&p| {
                vec![ResolvedProductionModel::ConstantProductivity { productivity: p }; n_stages]
            })
            .collect();
        ProductionModelSet::new(models, n_hydros, n_stages)
    }

    #[test]
    fn constant_productivity_yields_input_scalar() {
        let n_stages = 2;
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity,
            100.0,
            200.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, n_stages);
        let pm = production_set(&[0.9], n_stages);

        let set = build_energy_conversion_set(
            &hydros,
            n_stages,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        let c = set.conversion(0, 0);
        assert_eq!(c.equivalent_productivity_mw_per_m3s, 0.9);
        assert_eq!(c.reference_volume_hm3, 100.0 + 0.65 * (200.0 - 100.0));
        assert_eq!(c.reference_outflow_m3s, 50.0);
    }

    #[test]
    fn linearized_head_yields_input_scalar() {
        let n_stages = 1;
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::LinearizedHead,
            100.0,
            200.0,
            40.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.5, n_stages);
        let pm = production_set(&[1.2], n_stages);

        let set = build_energy_conversion_set(
            &hydros,
            n_stages,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        assert_eq!(set.conversion(0, 0).equivalent_productivity_mw_per_m3s, 1.2);
    }

    #[test]
    fn reference_volume_uses_fraction() {
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity,
            100.0,
            200.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);

        for f in [0.1_f64, 0.5, 1.0] {
            let resolver = constant_resolver(&hydros, f, 1);
            let set = build_energy_conversion_set(
                &hydros,
                1,
                &cascade,
                &resolver,
                &HashMap::new(),
                None,
                None,
            )
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
        let n_stages = 4;
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity,
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
        let pm = production_set(&[0.9], n_stages);
        let set = build_energy_conversion_set(
            &hydros,
            n_stages,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
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
            HydroGenerationModel::ConstantProductivity,
            200.0,
            100.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);

        let err = build_energy_conversion_set(
            &hydros,
            1,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
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
            HydroGenerationModel::ConstantProductivity,
            100.0,
            200.0,
            -1.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);

        let err = build_energy_conversion_set(
            &hydros,
            1,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::NegativeMaxTurbined { hydro_id, q_max } => {
                assert_eq!(hydro_id, hydros[0].id);
                assert_eq!(q_max, -1.0);
            }
            other => panic!("expected NegativeMaxTurbined, got: {other:?}"),
        }
    }

    /// Prior to ticket-007, an FPHA hydro with ρ_esp but no VHA silently
    /// yielded ρ_eq=0.0. After ticket-007, the correctness gate rejects it with
    /// `FphaMissingEquivalentProductivity`.
    #[test]
    fn fpha_hydro_missing_vha_is_rejected_with_actionable_error() {
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
        let err = build_energy_conversion_set(
            &hydros,
            1,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::FphaMissingEquivalentProductivity {
                hydro_id,
                ref hydro_name,
                stage,
            } => {
                assert_eq!(hydro_id, hydros[0].id);
                assert!(
                    hydro_name.contains("Hydro 5"),
                    "error should mention the hydro name, got: {hydro_name}"
                );
                assert_eq!(stage, 0);
            }
            other => panic!("expected FphaMissingEquivalentProductivity, got: {other:?}"),
        }
    }

    // ── ticket-005 FPHA derivation tests ───────────────────────────────────

    /// Create a flat VHA table (constant height) for testing.
    fn vha_constant_height(hydro_id: EntityId, height: f64) -> (EntityId, Vec<HydroGeometryRow>) {
        (
            hydro_id,
            vec![
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
            ],
        )
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

        let set = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &map, None, None)
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

        let set = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &map, None, None)
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

        let set = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &map, None, None)
            .expect("builder succeeds");

        // h_eq = 400 - 0 - 5 = 395; rho_eq = 0.009 * 395 = 3.555
        let got = set.conversion(0, 0).equivalent_productivity_mw_per_m3s;
        let expected = 0.0090 * 395.0;
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }

    /// Prior to ticket-007, an FPHA hydro with VHA but no ρ_esp silently
    /// yielded ρ_eq=0.0. After ticket-007 the correctness gate rejects it.
    #[test]
    fn fpha_missing_rho_esp_is_rejected() {
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

        let err = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &map, None, None)
            .unwrap_err();
        match err {
            EnergyConversionError::FphaMissingEquivalentProductivity {
                hydro_id,
                ref hydro_name,
                stage: _,
            } => {
                assert_eq!(hydro_id, hydros[0].id);
                assert!(
                    hydro_name.contains("Hydro 7"),
                    "error should mention hydro name, got: {hydro_name}"
                );
            }
            other => panic!("expected FphaMissingEquivalentProductivity, got: {other:?}"),
        }
    }

    /// Prior to ticket-007, an FPHA hydro with ρ_esp but no VHA silently
    /// yielded ρ_eq=0.0. After ticket-007 the correctness gate rejects it.
    #[test]
    fn fpha_missing_vha_is_rejected() {
        let hydro = fpha_hydro_for_tests(7); // has rho_esp = Some(0.009)
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);

        let err = build_energy_conversion_set(
            &hydros,
            1,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::FphaMissingEquivalentProductivity {
                hydro_id,
                ref hydro_name,
                stage: _,
            } => {
                assert_eq!(hydro_id, hydros[0].id);
                assert!(
                    hydro_name.contains("Hydro 7"),
                    "error should mention hydro name, got: {hydro_name}"
                );
            }
            other => panic!("expected FphaMissingEquivalentProductivity, got: {other:?}"),
        }
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

        let err = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &map, None, None)
            .unwrap_err();
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

        let err = build_energy_conversion_set(&hydros, 1, &cascade, &resolver, &map, None, None)
            .unwrap_err();
        match err {
            EnergyConversionError::ForebayTableInvalid { hydro_id, .. } => {
                assert_eq!(hydro_id, hydros[0].id);
            }
            other => panic!("expected ForebayTableInvalid, got: {other:?}"),
        }
    }

    // ── ticket-006 cascade accumulator tests ──────────────────────────────────

    /// A->B->C linear cascade. ρ_eq values: A=2.0, B=3.0, C=5.0 at stage 0.
    /// Expected ρ_acum: C=5.0, B=3+5=8.0, A=2+8=10.0.
    #[test]
    fn linear_cascade_accumulates_three_levels() {
        // A(id=0)->B(id=1)->C(id=2, terminal). Each hydro has ConstantProductivity.
        // The ρ_eq values are supplied via ProductionModelSet (declaration order: A=0, B=1, C=2).
        let mut a = make_hydro(0, Some(1));
        a.generation_model = HydroGenerationModel::ConstantProductivity;
        let mut b = make_hydro(1, Some(2));
        b.generation_model = HydroGenerationModel::ConstantProductivity;
        let mut c = make_hydro(2, None);
        c.generation_model = HydroGenerationModel::ConstantProductivity;
        let hydros = vec![a, b, c];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        // Declaration order: A(idx=0)=2.0, B(idx=1)=3.0, C(idx=2)=5.0.
        let pm = production_set(&[2.0, 3.0, 5.0], 1);

        let set = build_energy_conversion_set(
            &hydros,
            1,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        // Index by declaration order: A=0, B=1, C=2.
        assert_eq!(set.accumulated_productivity(0, 0), 10.0); // A
        assert_eq!(set.accumulated_productivity(1, 0), 8.0); // B
        assert_eq!(set.accumulated_productivity(2, 0), 5.0); // C
    }

    /// A->C and B->C branching cascade. ρ_eq: A=1.0, B=2.0, C=4.0.
    /// ρ_acum: C=4.0, A=1+4=5.0, B=2+4=6.0.
    #[test]
    fn branching_cascade_accumulates_correctly() {
        let mut a = make_hydro(0, Some(2));
        a.generation_model = HydroGenerationModel::ConstantProductivity;
        let mut b = make_hydro(1, Some(2));
        b.generation_model = HydroGenerationModel::ConstantProductivity;
        let mut c = make_hydro(2, None);
        c.generation_model = HydroGenerationModel::ConstantProductivity;
        let hydros = vec![a, b, c];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        // Declaration order: A(idx=0)=1.0, B(idx=1)=2.0, C(idx=2)=4.0.
        let pm = production_set(&[1.0, 2.0, 4.0], 1);

        let set = build_energy_conversion_set(
            &hydros,
            1,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        // A(idx=0), B(idx=1), C(idx=2).
        assert_eq!(set.accumulated_productivity(2, 0), 4.0); // C terminal
        assert_eq!(set.accumulated_productivity(0, 0), 5.0); // A = 1 + 4
        assert_eq!(set.accumulated_productivity(1, 0), 6.0); // B = 2 + 4
    }

    /// Build the same A->B->C cascade with two different declaration orders and
    /// confirm that ρ_acum indexed by EntityId is bit-for-bit identical.
    #[test]
    fn declaration_order_invariance() {
        let make_linear = |order: &[i32]| {
            // Map entity ID to downstream: 0->1->2 (terminal).
            let downstream = |id: i32| match id {
                0 => Some(1),
                1 => Some(2),
                _ => None,
            };
            let hydros: Vec<Hydro> = order
                .iter()
                .map(|&id| {
                    let mut h = make_hydro(id, downstream(id));
                    h.generation_model = HydroGenerationModel::ConstantProductivity;
                    h
                })
                .collect();
            hydros
        };

        let hydros_abc = make_linear(&[0, 1, 2]);
        let hydros_cab = make_linear(&[2, 0, 1]);

        let cascade_abc = CascadeTopology::build(&hydros_abc);
        let cascade_cab = CascadeTopology::build(&hydros_cab);

        let resolver_abc = constant_resolver(&hydros_abc, 0.65, 1);
        let resolver_cab = constant_resolver(&hydros_cab, 0.65, 1);

        let set_abc = build_energy_conversion_set(
            &hydros_abc,
            1,
            &cascade_abc,
            &resolver_abc,
            &HashMap::new(),
            None,
            None,
        )
        .expect("abc order");
        let set_cab = build_energy_conversion_set(
            &hydros_cab,
            1,
            &cascade_cab,
            &resolver_cab,
            &HashMap::new(),
            None,
            None,
        )
        .expect("cab order");

        // Build id->index maps for each ordering.
        let idx_abc: HashMap<i32, usize> = hydros_abc
            .iter()
            .enumerate()
            .map(|(i, h)| (h.id.0, i))
            .collect();
        let idx_cab: HashMap<i32, usize> = hydros_cab
            .iter()
            .enumerate()
            .map(|(i, h)| (h.id.0, i))
            .collect();

        for entity_id in [0_i32, 1, 2] {
            let val_abc = set_abc.accumulated_productivity(idx_abc[&entity_id], 0);
            let val_cab = set_cab.accumulated_productivity(idx_cab[&entity_id], 0);
            assert_eq!(
                val_abc.to_bits(),
                val_cab.to_bits(),
                "entity {entity_id}: abc={val_abc}, cab={val_cab}"
            );
        }
    }

    /// A hydro whose downstream_id points to an EntityId not in the hydros slice
    /// must return DanglingDownstream.
    #[test]
    fn dangling_downstream_is_rejected() {
        // Hydro 0 has downstream_id=99, but hydro 99 is not in the slice.
        // We must build the cascade manually to inject the dangling edge,
        // because CascadeTopology::build silently stores the dangling entry and
        // topological_order() ends up shorter than hydros.len(), triggering
        // CascadeIndexMismatch instead.  We therefore build a topology where
        // the downstream entry 99 IS in the map but not in id_to_index.

        // Use two hydros: id=0 (downstream=99) and id=1 (terminal).
        // The cascade built from these two will have topological_order length 2
        // (only hydros present in in_degree are included in the topo sort).
        // Because hydro 99 is absent from in_degree, length == 2 == hydros.len().
        // The DanglingDownstream error fires when the inner loop tries to look
        // up ds_id=99 in id_to_index.
        let mut h0 = make_hydro(0, Some(99));
        h0.generation_model = HydroGenerationModel::ConstantProductivity;
        let mut h1 = make_hydro(1, None);
        h1.generation_model = HydroGenerationModel::ConstantProductivity;
        let hydros = vec![h0, h1];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);

        let err = build_energy_conversion_set(
            &hydros,
            1,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::DanglingDownstream {
                hydro_id,
                downstream_id,
            } => {
                assert_eq!(hydro_id, EntityId::from(0));
                assert_eq!(downstream_id, EntityId::from(99));
            }
            other => panic!("expected DanglingDownstream, got: {other:?}"),
        }
    }

    /// When cascade.topological_order().len() != hydros.len(), return
    /// CascadeIndexMismatch.
    #[test]
    fn cascade_index_mismatch_is_rejected() {
        // Three hydros but a cascade for only two of them (built from a
        // different slice).  We create a cascade from a 2-hydro slice and then
        // call the builder with a 3-hydro slice.
        let h0 = make_hydro(0, Some(1));
        let h1 = make_hydro(1, None);
        let h2 = make_hydro(2, None);

        // Build the cascade from only two hydros so topo order has length 2.
        let short_cascade = CascadeTopology::build(&[h0.clone(), h1.clone()]);

        let hydros_three = vec![h0, h1, h2];
        let resolver = constant_resolver(&hydros_three, 0.65, 1);

        let err = build_energy_conversion_set(
            &hydros_three,
            1,
            &short_cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::CascadeIndexMismatch { expected, got } => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            other => panic!("expected CascadeIndexMismatch, got: {other:?}"),
        }
    }

    /// FPHA hydro with no VHA and no rho_esp but an override that returns a
    /// constant value for every stage succeeds, and the resolved
    /// equivalent_productivity equals the override value.
    #[test]
    fn fpha_with_override_only_succeeds() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.specific_productivity_mw_per_m3s_per_m = None;
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 3);
        let override_table =
            build_hydro_energy_productivity_override(vec![HydroEnergyProductivityRow {
                hydro_id: hydros[0].id,
                stage_id: None,
                equivalent_productivity_mw_per_m3s: Some(2.5),
                reference_volume_hm3: None,
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            }])
            .expect("override builds");

        let set = build_energy_conversion_set(
            &hydros,
            3,
            &cascade,
            &resolver,
            &HashMap::new(),
            Some(&override_table),
            None,
        )
        .expect("builder succeeds with override only");

        for s in 0..3 {
            let cell = set.conversion(0, s);
            assert!(
                (cell.equivalent_productivity_mw_per_m3s - 2.5).abs() < 1e-12,
                "stage {s}: expected 2.5, got {}",
                cell.equivalent_productivity_mw_per_m3s
            );
        }
    }

    /// A ConstantProductivity hydro with no VHA, no rho_esp, and no override
    /// must succeed — the FPHA correctness gate must not apply to non-FPHA
    /// generation models. The productivity is supplied via `ProductionModelSet`;
    /// the test verifies the value is passed through to the conversion cell.
    #[test]
    fn constant_productivity_bypasses_gate() {
        let mut hydro = make_hydro(0, None);
        hydro.generation_model = HydroGenerationModel::ConstantProductivity;
        hydro.specific_productivity_mw_per_m3s_per_m = None;
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let pm = production_set(&[1.5], 1);

        let set = build_energy_conversion_set(
            &hydros,
            1,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("non-FPHA hydro succeeds despite missing FPHA inputs");

        let cell = set.conversion(0, 0);
        assert!((cell.equivalent_productivity_mw_per_m3s - 1.5).abs() < 1e-12);
    }

    // ── ticket-008 consistency_warnings tests ─────────────────────────────────

    /// Build a 1-hydro EnergyConversionSet with `ConstantProductivity` and the
    /// given `rho_eq`, `v_ref`, and `specific_productivity`.
    ///
    /// `rho_eq` is supplied via a `ProductionModelSet` so that
    /// `equivalent_productivity_mw_per_m3s` in the returned set equals `rho_eq`.
    ///
    /// Uses fraction=0.0 so `V_ref = v_min + 0 * (v_max - v_min) = v_min`,
    /// giving a deterministic `reference_volume_hm3` value.
    fn make_set_for_warning_tests(
        id: i32,
        rho_eq: f64,
        v_ref: f64,
        specific_productivity: Option<f64>,
    ) -> (EnergyConversionSet, Hydro) {
        // v_min = v_ref, v_max = v_ref + 100 so fraction=0.0 yields exactly v_ref.
        let mut hydro = make_hydro_with(
            id,
            HydroGenerationModel::ConstantProductivity,
            v_ref,
            v_ref + 100.0,
            50.0,
            specific_productivity,
        );
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro.clone()];
        let cascade = CascadeTopology::build(&hydros);
        // fraction=0.0 -> V_ref = v_min = v_ref exactly.
        let resolver = constant_resolver(&hydros, 0.0, 1);
        let pm = production_set(&[rho_eq], 1);
        let set = build_energy_conversion_set(
            &hydros,
            1,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("set builds");
        (set, hydro)
    }

    /// Given a hydro with `ConstantProductivity { productivity_mw_per_m3s: 0.85 }`,
    /// `specific_productivity = Some(0.0085)`, flat VHA `h_fore = 100.0` (no
    /// tailrace, no losses), the implied `rho_esp = 0.85 / 100.0 = 0.0085`
    /// exactly matches the supplied value => no warning.
    #[test]
    fn consistent_rho_esp_yields_no_warning() {
        let (set, hydro) = make_set_for_warning_tests(1, 0.85, 0.0, Some(0.0085));
        let hydros = vec![hydro.clone()];
        let (id, rows) = vha_constant_height(hydro.id, 100.0);
        let mut vha_map = HashMap::new();
        vha_map.insert(id, rows);

        let warnings = set.consistency_warnings(&hydros, &vha_map);
        assert!(
            warnings.is_empty(),
            "expected no warnings for consistent rho_esp, got: {warnings:?}"
        );
    }

    /// Given the same hydro changed to `ConstantProductivity {
    /// productivity_mw_per_m3s: 0.50 }`, implied `rho_esp = 0.50 / 100.0 = 0.005`,
    /// divergence ≈ 41% from supplied `0.0085` => one warning.
    #[test]
    fn divergent_rho_esp_yields_one_warning() {
        let (set, hydro) = make_set_for_warning_tests(1, 0.50, 0.0, Some(0.0085));
        let hydros = vec![hydro.clone()];
        let (id, rows) = vha_constant_height(hydro.id, 100.0);
        let mut vha_map = HashMap::new();
        vha_map.insert(id, rows);

        let warnings = set.consistency_warnings(&hydros, &vha_map);
        assert_eq!(
            warnings.len(),
            1,
            "expected exactly 1 warning, got: {warnings:?}"
        );
        match &warnings[0] {
            ConsistencyWarning::ImpliedSpecificProductivityMismatch {
                hydro_id,
                stage_id,
                supplied_rho_esp,
                implied_rho_esp,
                tolerance,
            } => {
                assert_eq!(*hydro_id, hydro.id);
                assert_eq!(*stage_id, 0);
                assert!(
                    (supplied_rho_esp - 0.0085).abs() < 1e-12,
                    "supplied_rho_esp={supplied_rho_esp}"
                );
                assert!(
                    (implied_rho_esp - 0.005).abs() < 1e-12,
                    "implied_rho_esp={implied_rho_esp}"
                );
                assert!(
                    (tolerance - SPECIFIC_PRODUCTIVITY_CONSISTENCY_TOLERANCE).abs() < 1e-12,
                    "tolerance={tolerance}"
                );
            }
        }
    }

    /// Given a hydro with no VHA geometry in the map, `consistency_warnings`
    /// returns an empty vec for that hydro (silent skip).
    #[test]
    fn missing_vha_skips_silently() {
        let (set, hydro) = make_set_for_warning_tests(1, 0.50, 0.0, Some(0.0085));
        let hydros = vec![hydro];
        // Empty map — no VHA rows.
        let vha_map: HashMap<EntityId, Vec<HydroGeometryRow>> = HashMap::new();

        let warnings = set.consistency_warnings(&hydros, &vha_map);
        assert!(
            warnings.is_empty(),
            "expected no warnings when VHA is absent, got: {warnings:?}"
        );
    }

    /// Given a hydro with `specific_productivity_mw_per_m3s_per_m = None`,
    /// `consistency_warnings` returns an empty vec (silent skip).
    #[test]
    fn missing_rho_esp_skips_silently() {
        let (set, mut hydro) = make_set_for_warning_tests(1, 0.50, 0.0, None);
        hydro.specific_productivity_mw_per_m3s_per_m = None;
        let hydros = vec![hydro.clone()];
        let (id, rows) = vha_constant_height(hydro.id, 100.0);
        let mut vha_map = HashMap::new();
        vha_map.insert(id, rows);

        let warnings = set.consistency_warnings(&hydros, &vha_map);
        assert!(
            warnings.is_empty(),
            "expected no warnings when rho_esp is None, got: {warnings:?}"
        );
    }

    /// Given a `HydroGenerationModel::Fpha` hydro, `consistency_warnings`
    /// returns an empty vec (FPHA has no `ρ_eq_stored` to compare against).
    #[test]
    fn fpha_hydro_skips_silently() {
        // Build a set for an FPHA hydro using the override mechanism so the
        // builder succeeds without VHA data.
        let mut hydro = make_hydro_with(
            1,
            HydroGenerationModel::Fpha,
            0.0,
            100.0,
            50.0,
            Some(0.0085),
        );
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro.clone()];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.0, 1);
        let override_table =
            build_hydro_energy_productivity_override(vec![HydroEnergyProductivityRow {
                hydro_id: hydros[0].id,
                stage_id: None,
                equivalent_productivity_mw_per_m3s: Some(0.85),
                reference_volume_hm3: None,
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            }])
            .expect("override builds");
        let set = build_energy_conversion_set(
            &hydros,
            1,
            &cascade,
            &resolver,
            &HashMap::new(),
            Some(&override_table),
            None,
        )
        .expect("FPHA set builds with override");

        let (id, rows) = vha_constant_height(hydro.id, 100.0);
        let mut vha_map = HashMap::new();
        vha_map.insert(id, rows);

        let warnings = set.consistency_warnings(&hydros, &vha_map);
        assert!(
            warnings.is_empty(),
            "expected no warnings for FPHA hydro, got: {warnings:?}"
        );
    }

    /// Given a 1-hydro set with 3 stages where the hydro is divergent, all 3
    /// stages produce a warning with `stage_id` values 0, 1, 2 in order.
    #[test]
    fn multi_stage_divergence_emits_one_warning_per_stage() {
        let n_stages = 3;
        let mut hydro = make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity,
            0.0,
            100.0,
            50.0,
            Some(0.0085),
        );
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro.clone()];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.0, n_stages);
        let set = build_energy_conversion_set(
            &hydros,
            n_stages,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
        .expect("set builds");

        let (id, rows) = vha_constant_height(hydro.id, 100.0);
        let mut vha_map = HashMap::new();
        vha_map.insert(id, rows);

        let warnings = set.consistency_warnings(&hydros, &vha_map);
        assert_eq!(
            warnings.len(),
            n_stages,
            "expected {n_stages} warnings, got: {warnings:?}"
        );
        for (i, w) in warnings.iter().enumerate() {
            match w {
                ConsistencyWarning::ImpliedSpecificProductivityMismatch { stage_id, .. } => {
                    assert_eq!(
                        *stage_id, i as i32,
                        "stage {i}: expected stage_id={i}, got {stage_id}"
                    );
                }
            }
        }
    }

    /// Build a 2-hydro case where hydro 0 is consistent and hydro 1 is
    /// divergent across all 3 stages. Returned warnings are for hydro 1
    /// only, with `stage_id` 0, 1, 2 in order.
    #[test]
    fn warnings_ordered_by_hydro_declaration_then_stage() {
        let n_stages = 3;

        // Hydro 0 (index 0): rho_eq=0.85, supplied=0.0085, h_fore=100 → implied=0.0085 (consistent).
        let mut h0 = make_hydro_with(
            10,
            HydroGenerationModel::ConstantProductivity,
            0.0,
            100.0,
            50.0,
            Some(0.0085),
        );
        h0.tailrace = None;
        h0.hydraulic_losses = None;

        // Hydro 1 (index 1): rho_eq=0.50, supplied=0.0085, h_fore=100 → implied=0.005 (divergent).
        let mut h1 = make_hydro_with(
            20,
            HydroGenerationModel::ConstantProductivity,
            0.0,
            100.0,
            50.0,
            Some(0.0085),
        );
        h1.tailrace = None;
        h1.hydraulic_losses = None;

        let hydros = vec![h0.clone(), h1.clone()];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.0, n_stages);
        // h0(idx=0) ρ_eq=0.85 (consistent), h1(idx=1) ρ_eq=0.50 (divergent).
        let pm = production_set(&[0.85, 0.50], n_stages);
        let set = build_energy_conversion_set(
            &hydros,
            n_stages,
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("set builds");

        let (id0, rows0) = vha_constant_height(h0.id, 100.0);
        let (id1, rows1) = vha_constant_height(h1.id, 100.0);
        let mut vha_map = HashMap::new();
        vha_map.insert(id0, rows0);
        vha_map.insert(id1, rows1);

        let warnings = set.consistency_warnings(&hydros, &vha_map);

        // Only hydro 1 diverges; 3 stages → 3 warnings.
        assert_eq!(warnings.len(), 3, "expected 3 warnings, got: {warnings:?}");
        for (i, w) in warnings.iter().enumerate() {
            match w {
                ConsistencyWarning::ImpliedSpecificProductivityMismatch {
                    hydro_id,
                    stage_id,
                    ..
                } => {
                    assert_eq!(
                        *hydro_id, h1.id,
                        "warning {i}: expected hydro_id={:?}, got {hydro_id:?}",
                        h1.id
                    );
                    assert_eq!(
                        *stage_id, i as i32,
                        "warning {i}: expected stage_id={i}, got {stage_id}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_override_four_column_lookup_precedence() {
        let rows = vec![
            HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: Some(0),
                equivalent_productivity_mw_per_m3s: Some(3.6),
                reference_volume_hm3: None,
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            },
            HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: None,
                equivalent_productivity_mw_per_m3s: Some(4.0),
                reference_volume_hm3: Some(120.0),
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: Some(0.009),
            },
            HydroEnergyProductivityRow {
                hydro_id: EntityId(2),
                stage_id: None,
                equivalent_productivity_mw_per_m3s: Some(5.0),
                reference_volume_hm3: None,
                reference_outflow_m3s: Some(200.0),
                specific_productivity_mw_per_m3s_per_m: None,
            },
        ];
        let o = build_hydro_energy_productivity_override(rows).expect("override builds");
        assert_eq!(o.equivalent_productivity(EntityId(1), 0), Some(3.6));
        assert_eq!(o.equivalent_productivity(EntityId(1), 1), Some(4.0));
        assert_eq!(o.equivalent_productivity(EntityId(2), 0), Some(5.0));
        assert_eq!(o.equivalent_productivity(EntityId(3), 0), None);
        assert_eq!(o.reference_volume(EntityId(1), 0), Some(120.0));
        assert_eq!(o.reference_volume(EntityId(1), 1), Some(120.0));
        assert_eq!(o.reference_volume(EntityId(2), 0), None);
        assert_eq!(o.reference_outflow(EntityId(2), 0), Some(200.0));
        assert_eq!(o.reference_outflow(EntityId(1), 0), None);
        assert_eq!(o.specific_productivity(EntityId(1), 0), Some(0.009));
        assert_eq!(o.specific_productivity(EntityId(1), 1), Some(0.009));
        assert_eq!(o.specific_productivity(EntityId(2), 0), None);
    }

    #[test]
    fn test_build_override_rejects_duplicate_hydro_stage() {
        let rows = vec![
            HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: Some(0),
                equivalent_productivity_mw_per_m3s: Some(3.6),
                reference_volume_hm3: None,
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            },
            HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: Some(0),
                equivalent_productivity_mw_per_m3s: None,
                reference_volume_hm3: Some(120.0),
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            },
        ];
        let err = build_hydro_energy_productivity_override(rows).unwrap_err();
        match err {
            cobre_io::LoadError::SchemaError { field, .. } => {
                assert_eq!(field, "hydro_energy_productivity.duplicate_entry");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_build_override_distinguishes_null_and_concrete_stages() {
        let rows = vec![
            HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: None,
                equivalent_productivity_mw_per_m3s: Some(2.0),
                reference_volume_hm3: None,
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            },
            HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: Some(0),
                equivalent_productivity_mw_per_m3s: Some(3.0),
                reference_volume_hm3: None,
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            },
        ];
        let o = build_hydro_energy_productivity_override(rows).expect("override builds");
        assert_eq!(o.equivalent_productivity(EntityId(1), 0), Some(3.0));
        assert_eq!(o.equivalent_productivity(EntityId(1), 1), Some(2.0));
    }

    #[test]
    fn test_default_override_returns_none_for_every_accessor() {
        let o = HydroEnergyProductivityOverride::default();
        assert_eq!(o.equivalent_productivity(EntityId(1), 0), None);
        assert_eq!(o.reference_volume(EntityId(1), 0), None);
        assert_eq!(o.reference_outflow(EntityId(1), 0), None);
        assert_eq!(o.specific_productivity(EntityId(1), 0), None);
    }
}
