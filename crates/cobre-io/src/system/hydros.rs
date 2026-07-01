//! Parsing for `system/hydros.json` — hydro plant entity registry.
//!
//! [`parse_hydros`] reads `system/hydros.json` from the case directory and
//! returns a fully-validated, sorted `Vec<Hydro>`.
//!
//! ## JSON structure
//!
//! ```json
//! {
//!   "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/hydros.schema.json",
//!   "hydros": [{
//!     "id": 0, "name": "FURNAS", "operational_start_date": "2030-01-01", "bus_id": 0,
//!     "downstream_id": 2,
//!     "entry_stage_id": null, "exit_stage_id": null,
//!     "filling": null,
//!     "diversion": null,
//!     "reservoir": { "min_storage_hm3": 5733.0, "max_storage_hm3": 22950.0 },
//!     "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
//!     "generation": {
//!       "model": "constant_productivity",
//!       "min_turbined_m3s": 0.0, "max_turbined_m3s": 1692.0,
//!       "min_generation_mw": 0.0, "max_generation_mw": 1312.0
//!     },
//!     "tailrace": { "type": "polynomial", "coefficients": [326.0, 0.0032, -1.2e-7] },
//!     "hydraulic_losses": { "type": "factor", "value": 0.03 },
//!     "efficiency": { "type": "constant", "value": 0.92 },
//!     "evaporation": { "coefficients_mm": [150, 130, 120, 90, 60, 40, 30, 40, 70, 100, 130, 150] },
//!     "penalties": { "spillage_cost": 0.05 }
//!   }]
//! }
//! ```
//!
//! ## Validation
//!
//! After deserializing, the following invariants are checked before conversion:
//!
//! 1. No two hydros share the same `id`.
//! 2. `min_storage_hm3 >= 0`, `max_storage_hm3 >= 0`, `min_storage_hm3 <= max_storage_hm3`.
//! 3. `min_outflow_m3s >= 0`.
//! 4. `min_turbined_m3s >= 0`, `max_turbined_m3s >= 0`, `min_turbined_m3s <= max_turbined_m3s`.
//! 5. `min_generation_mw <= max_generation_mw`.
//! 6. Evaporation array, if present, must have exactly 12 elements.
//!
//! Cross-reference validation (`bus_id`, `downstream_id`, `diversion.downstream_id`)
//! is deferred to Layer 3.

use cobre_core::{
    EntityId,
    entities::{
        DiversionChannel, EfficiencyModel, FillingConfig, HydraulicLossesModel, Hydro,
        HydroGenerationModel, TailraceModel, TailracePoint,
    },
    penalty::{GlobalPenaltyDefaults, HydroPenaltyOverrides, resolve_hydro_penalties},
};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use super::parse_operational_start_date;
use crate::LoadError;

// ── Intermediate serde types ──────────────────────────────────────────────────

/// Top-level intermediate type for `hydros.json`.
///
/// Private — only used during deserialization. Not re-exported.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawHydroFile {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Array of hydro plant entries.
    hydros: Vec<RawHydro>,
}

/// Intermediate type for a single hydro plant entry.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawHydro {
    /// Hydro plant identifier. Must be unique within the file.
    id: i32,
    /// Human-readable plant name.
    name: String,
    /// Date the entity enters service (ISO 8601 `YYYY-MM-DD`).
    operational_start_date: String,
    /// Bus to which this plant's generation is injected.
    bus_id: i32,
    /// Downstream hydro plant in the cascade. `null` = no downstream.
    downstream_id: Option<i32>,
    /// Stage index when the plant enters service. Absent or null = always exists.
    #[serde(default)]
    entry_stage_id: Option<i32>,
    /// Stage index when the plant is decommissioned. Absent or null = never.
    #[serde(default)]
    exit_stage_id: Option<i32>,
    /// Reservoir storage bounds.
    reservoir: RawReservoir,
    /// Total outflow bounds.
    outflow: RawOutflow,
    /// Generation model configuration (tagged union on `model` field).
    generation: RawGeneration,
    /// Tailrace elevation model. Absent or null = no tailrace.
    #[serde(default)]
    tailrace: Option<RawTailrace>,
    /// Hydraulic loss model. Absent or null = lossless penstock.
    #[serde(default)]
    hydraulic_losses: Option<RawHydraulicLosses>,
    /// Turbine efficiency model. Absent or null = 100% efficiency.
    #[serde(default)]
    efficiency: Option<RawEfficiency>,
    /// Monthly evaporation coefficients. Absent or null = no evaporation.
    #[serde(default)]
    evaporation: Option<RawEvaporation>,
    /// Diversion channel configuration. Absent or null = no diversion channel.
    #[serde(default)]
    diversion: Option<RawDiversionChannel>,
    /// Reservoir filling configuration. Absent or null = no filling operation.
    #[serde(default)]
    filling: Option<RawFillingConfig>,
    /// Specific productivity `ρ_esp` \[MW / ((m³/s) · m)\].
    ///
    /// **Resolution cascade** (first source that supplies a non-`null` value wins):
    ///
    /// 1. Per-`(hydro, stage)` row in `system/hydro_energy_productivity.parquet`
    ///    (loaded at study setup time by the solver).
    /// 2. This field — a single value applied uniformly across all stages.
    ///
    /// Absent or `null` means this fallback level is skipped.  If the cascade
    /// finds no value for a hydro whose generation model requires one
    /// (`constant_productivity` or `linearized_head`), study setup fails with an
    /// explicit error.
    #[serde(default)]
    specific_productivity_mw_per_m3s_per_m: Option<f64>,
    /// Entity-level penalty overrides. Absent = all penalties use global defaults.
    #[serde(default)]
    penalties: Option<RawHydroPenaltyOverrides>,
}

/// Intermediate type for the `reservoir` sub-object.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawReservoir {
    /// Minimum operational storage (dead volume) [hm³].
    min_storage_hm3: f64,
    /// Maximum operational storage [hm³].
    max_storage_hm3: f64,
}

/// Intermediate type for the `outflow` sub-object.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawOutflow {
    /// Minimum total outflow [m³/s].
    min_outflow_m3s: f64,
    /// Maximum total outflow [m³/s]. `null` = no upper bound.
    max_outflow_m3s: Option<f64>,
}

/// Tagged-union intermediate type for the `generation` sub-object.
///
/// Uses `#[serde(tag = "model")]` (internally-tagged) to dispatch on the
/// `"model"` field value. Each variant carries only the fields relevant to
/// that model. The `productivity_mw_per_m3s` field is NOT accepted here —
/// productivity coefficients are read from `system/hydro_production_models.json`
/// and are associated per `(hydro, stage)` outside this file.
/// A `hydros.json` input that includes `productivity_mw_per_m3s` in its
/// `generation` block will be rejected with a parse error.
#[derive(Deserialize)]
#[serde(tag = "model", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) enum RawGeneration {
    /// Constant productivity: `power = productivity * turbined_m3s`.
    ///
    /// The productivity coefficient is read from
    /// `system/hydro_production_models.json` per `(hydro, stage)`. It is not
    /// accepted in `hydros.json`; supplying `productivity_mw_per_m3s` here
    /// produces a hard parse error.
    ConstantProductivity {
        /// Minimum turbined flow [m³/s].
        min_turbined_m3s: f64,
        /// Maximum turbined flow [m³/s].
        max_turbined_m3s: f64,
        /// Minimum electrical generation [MW].
        min_generation_mw: f64,
        /// Maximum electrical generation [MW].
        max_generation_mw: f64,
    },
    /// Head-dependent productivity linearized around an operating point.
    ///
    /// The productivity coefficient is read from
    /// `system/hydro_production_models.json` per `(hydro, stage)`. It is not
    /// accepted in `hydros.json`; supplying `productivity_mw_per_m3s` here
    /// produces a hard parse error.
    LinearizedHead {
        /// Minimum turbined flow [m³/s].
        min_turbined_m3s: f64,
        /// Maximum turbined flow [m³/s].
        max_turbined_m3s: f64,
        /// Minimum electrical generation [MW].
        min_generation_mw: f64,
        /// Maximum electrical generation [MW].
        max_generation_mw: f64,
    },
    /// Full production function with head-area-productivity tables (FPHA model).
    ///
    /// Productivity is derived from the FPHA tables in
    /// `system/hydro_production_models.json`. The `productivity_mw_per_m3s`
    /// field is not accepted here or in any other `generation` variant.
    Fpha {
        /// Minimum turbined flow [m³/s].
        min_turbined_m3s: f64,
        /// Maximum turbined flow [m³/s].
        max_turbined_m3s: f64,
        /// Minimum electrical generation [MW].
        min_generation_mw: f64,
        /// Maximum electrical generation [MW].
        max_generation_mw: f64,
    },
}

impl RawGeneration {
    /// Extract the turbine and generation bounds shared across all variants.
    fn bounds(&self) -> (f64, f64, f64, f64) {
        match self {
            Self::ConstantProductivity {
                min_turbined_m3s,
                max_turbined_m3s,
                min_generation_mw,
                max_generation_mw,
                ..
            }
            | Self::LinearizedHead {
                min_turbined_m3s,
                max_turbined_m3s,
                min_generation_mw,
                max_generation_mw,
                ..
            }
            | Self::Fpha {
                min_turbined_m3s,
                max_turbined_m3s,
                min_generation_mw,
                max_generation_mw,
            } => (
                *min_turbined_m3s,
                *max_turbined_m3s,
                *min_generation_mw,
                *max_generation_mw,
            ),
        }
    }
}

/// Tagged-union intermediate type for the `tailrace` sub-object.
///
/// Uses `#[serde(tag = "type")]` internally-tagged on the `"type"` field.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) enum RawTailrace {
    /// Polynomial tailrace curve.
    Polynomial {
        /// Polynomial coefficients in ascending power order.
        coefficients: Vec<f64>,
    },
    /// Piecewise-linear tailrace curve.
    Piecewise {
        /// Breakpoints defining the piecewise-linear curve.
        points: Vec<RawTailracePoint>,
    },
}

/// Intermediate type for a single piecewise tailrace breakpoint.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawTailracePoint {
    /// Total outflow at this point [m³/s].
    outflow_m3s: f64,
    /// Downstream water level (tailrace height) at this outflow [m].
    height_m: f64,
}

/// Tagged-union intermediate type for the `hydraulic_losses` sub-object.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) enum RawHydraulicLosses {
    /// Losses as a fraction of net head.
    Factor {
        /// Dimensionless loss factor.
        value: f64,
    },
    /// Constant head loss independent of flow or head.
    Constant {
        /// Fixed head loss [m].
        value_m: f64,
    },
}

/// Tagged-union intermediate type for the `efficiency` sub-object.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) enum RawEfficiency {
    /// Constant efficiency across all operating points.
    Constant {
        /// Turbine efficiency as a fraction in (0, 1].
        value: f64,
    },
}

/// Intermediate type for the `evaporation` sub-object.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawEvaporation {
    /// Monthly evaporation coefficients [mm/month], one per calendar month.
    /// Index 0 = January, index 11 = December.
    coefficients_mm: Vec<f64>,
    /// Monthly reservoir reference volumes [hm³] used as the linearization
    /// reference point for evaporation, one per calendar month.
    /// Index 0 = January, index 11 = December.
    /// Absent = no reference volume override; the calling algorithm uses its
    /// own default (e.g., mid-point of the storage range).
    #[serde(default)]
    reference_volumes_hm3: Option<Vec<f64>>,
}

/// Intermediate type for the `diversion` sub-object.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawDiversionChannel {
    /// Identifier of the downstream hydro plant receiving diverted water.
    downstream_id: i32,
    /// Maximum diversion flow capacity [m³/s].
    max_flow_m3s: f64,
}

/// Intermediate type for the `filling` sub-object.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawFillingConfig {
    /// Stage index at which filling begins (inclusive).
    start_stage_id: i32,
    /// Minimum accumulation rate applied during filling [m³/s].
    /// Absent = passive filling (no minimum rate, defaults to 0.0 per spec).
    #[serde(default)]
    filling_min_rate_m3s: f64,
}

/// Intermediate type for entity-level hydro penalty overrides.
///
/// All 11 fields are `Option<f64>`. Absent fields default to `None`,
/// meaning the global default for that penalty is used.
///
/// JSON field names mirror `HydroPenalties` and `HydroPenaltyOverrides` field names.
#[allow(clippy::struct_field_names)]
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawHydroPenaltyOverrides {
    #[serde(default)]
    spillage_cost: Option<f64>,
    #[serde(default)]
    diversion_cost: Option<f64>,
    #[serde(default)]
    turbined_cost: Option<f64>,
    #[serde(default)]
    storage_violation_below_cost: Option<f64>,
    #[serde(default)]
    filling_target_violation_cost: Option<f64>,
    #[serde(default)]
    turbined_violation_below_cost: Option<f64>,
    #[serde(default)]
    outflow_violation_below_cost: Option<f64>,
    #[serde(default)]
    outflow_violation_above_cost: Option<f64>,
    #[serde(default)]
    generation_violation_below_cost: Option<f64>,
    #[serde(default)]
    evaporation_violation_cost: Option<f64>,
    #[serde(default)]
    water_withdrawal_violation_cost: Option<f64>,
    #[serde(default)]
    inflow_nonnegativity_cost: Option<f64>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load and validate `system/hydros.json` from `path`.
///
/// Reads the JSON file, deserializes it through intermediate serde types,
/// performs post-deserialization validation, then converts to `Vec<Hydro>` using
/// the three-tier penalty resolution cascade (global → entity). The result is
/// sorted by `id` ascending, so parser output is deterministic regardless of file
/// row order (declaration-order invariance); the builder applies the same id as
/// its `(operational_start_date, id)` canonical tiebreak.
///
/// Cross-reference validation (`bus_id`, `downstream_id`,
/// `diversion.downstream_id`) is deferred to Layer 3.
///
/// # Errors
///
/// | Condition                                           | Error variant              |
/// | --------------------------------------------------- | -------------------------- |
/// | File not found / read failure                       | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or missing required field       | [`LoadError::ParseError`]  |
/// | Unknown `generation.model` variant                  | [`LoadError::SchemaError`] |
/// | Duplicate `id` within the hydros array              | [`LoadError::SchemaError`] |
/// | `min_storage_hm3 < 0` or `max_storage_hm3 < 0`    | [`LoadError::SchemaError`] |
/// | `min_storage_hm3 > max_storage_hm3`                | [`LoadError::SchemaError`] |
/// | `min_outflow_m3s < 0`                              | [`LoadError::SchemaError`] |
/// | `min_turbined_m3s < 0` or `max_turbined_m3s < 0`  | [`LoadError::SchemaError`] |
/// | `max_turbined_m3s < min_turbined_m3s`              | [`LoadError::SchemaError`] |
/// | `max_generation_mw < min_generation_mw`            | [`LoadError::SchemaError`] |
/// | Evaporation array not exactly 12 elements          | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::system::parse_hydros;
/// use cobre_core::penalty::GlobalPenaltyDefaults;
/// use std::path::Path;
///
/// # fn make_global() -> GlobalPenaltyDefaults { unimplemented!() }
/// let global = make_global();
/// let hydros = parse_hydros(Path::new("case/system/hydros.json"), &global).unwrap();
/// assert!(!hydros.is_empty());
/// ```
pub fn parse_hydros(
    path: &Path,
    global_penalties: &GlobalPenaltyDefaults,
) -> Result<Vec<Hydro>, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawHydroFile = serde_json::from_str(&raw_text).map_err(|e| {
        let msg = e.to_string();
        // Reclassify unknown-variant / missing-field as SchemaError, not ParseError,
        // so callers distinguish bad JSON syntax from unknown enum discriminants.
        if msg.contains("unknown variant") || msg.contains("missing field") {
            LoadError::SchemaError {
                path: path.to_path_buf(),
                field: extract_field_from_serde_msg(&msg),
                message: msg,
            }
        } else {
            LoadError::parse(path, msg)
        }
    })?;

    validate_raw_hydros(&raw, path)?;

    convert_hydros(raw, global_penalties, path)
}

// ── Validation ────────────────────────────────────────────────────────────────

fn validate_raw_hydros(raw: &RawHydroFile, path: &Path) -> Result<(), LoadError> {
    validate_no_duplicate_hydro_ids(&raw.hydros, path)?;
    for (i, hydro) in raw.hydros.iter().enumerate() {
        validate_reservoir(&hydro.reservoir, i, path)?;
        validate_outflow(&hydro.outflow, i, path)?;
        validate_generation(&hydro.generation, i, path)?;
        if let Some(evap) = &hydro.evaporation {
            validate_evaporation(
                evap,
                i,
                path,
                hydro.reservoir.min_storage_hm3,
                hydro.reservoir.max_storage_hm3,
            )?;
        }
    }
    Ok(())
}

fn validate_no_duplicate_hydro_ids(hydros: &[RawHydro], path: &Path) -> Result<(), LoadError> {
    let mut seen: HashSet<i32> = HashSet::new();
    for (i, hydro) in hydros.iter().enumerate() {
        if !seen.insert(hydro.id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("hydros[{i}].id"),
                message: format!("duplicate id {} in hydros array", hydro.id),
            });
        }
    }
    Ok(())
}

fn validate_reservoir(
    reservoir: &RawReservoir,
    hydro_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if reservoir.min_storage_hm3 < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{hydro_index}].reservoir.min_storage_hm3"),
            message: format!(
                "min_storage_hm3 must be >= 0, got {}",
                reservoir.min_storage_hm3
            ),
        });
    }
    if reservoir.max_storage_hm3 < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{hydro_index}].reservoir.max_storage_hm3"),
            message: format!(
                "max_storage_hm3 must be >= 0, got {}",
                reservoir.max_storage_hm3
            ),
        });
    }
    if reservoir.min_storage_hm3 > reservoir.max_storage_hm3 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{hydro_index}].reservoir"),
            message: format!(
                "min_storage_hm3 ({}) must be <= max_storage_hm3 ({})",
                reservoir.min_storage_hm3, reservoir.max_storage_hm3
            ),
        });
    }
    Ok(())
}

fn validate_outflow(
    outflow: &RawOutflow,
    hydro_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if outflow.min_outflow_m3s < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{hydro_index}].outflow.min_outflow_m3s"),
            message: format!(
                "min_outflow_m3s must be >= 0, got {}",
                outflow.min_outflow_m3s
            ),
        });
    }
    Ok(())
}

fn validate_generation(
    generation: &RawGeneration,
    hydro_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    let (min_turbined, max_turbined, min_gen, max_gen) = generation.bounds();

    if min_turbined < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{hydro_index}].generation.min_turbined_m3s"),
            message: format!("min_turbined_m3s must be >= 0, got {min_turbined}"),
        });
    }
    if max_turbined < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{hydro_index}].generation.max_turbined_m3s"),
            message: format!("max_turbined_m3s must be >= 0, got {max_turbined}"),
        });
    }
    if max_turbined < min_turbined {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{hydro_index}].generation.max_turbined_m3s"),
            message: format!(
                "max_turbined_m3s ({max_turbined}) must be >= min_turbined_m3s ({min_turbined})"
            ),
        });
    }
    if max_gen < min_gen {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{hydro_index}].generation.max_generation_mw"),
            message: format!(
                "max_generation_mw ({max_gen}) must be >= min_generation_mw ({min_gen})"
            ),
        });
    }
    Ok(())
}

fn validate_evaporation(
    evaporation: &RawEvaporation,
    hydro_index: usize,
    path: &Path,
    min_storage_hm3: f64,
    max_storage_hm3: f64,
) -> Result<(), LoadError> {
    let len = evaporation.coefficients_mm.len();
    if len != 12 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{hydro_index}].evaporation.coefficients_mm"),
            message: format!(
                "evaporation coefficients_mm must have exactly 12 elements (one per calendar month), got {len}"
            ),
        });
    }

    if let Some(ref_vols) = &evaporation.reference_volumes_hm3 {
        let ref_len = ref_vols.len();
        if ref_len != 12 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("hydros[{hydro_index}].evaporation.reference_volumes_hm3"),
                message: format!(
                    "evaporation reference_volumes_hm3 must have exactly 12 elements (one per calendar month), got {ref_len}"
                ),
            });
        }
        for (month, &vol) in ref_vols.iter().enumerate() {
            if !vol.is_finite() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("hydros[{hydro_index}].evaporation.reference_volumes_hm3"),
                    message: format!(
                        "evaporation reference_volumes_hm3[{month}] must be finite, got {vol}"
                    ),
                });
            }
            if vol < min_storage_hm3 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("hydros[{hydro_index}].evaporation.reference_volumes_hm3"),
                    message: format!(
                        "evaporation reference_volumes_hm3[{month}] ({vol}) must be >= min_storage_hm3 ({min_storage_hm3})"
                    ),
                });
            }
            if vol > max_storage_hm3 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("hydros[{hydro_index}].evaporation.reference_volumes_hm3"),
                    message: format!(
                        "evaporation reference_volumes_hm3[{month}] ({vol}) must be <= max_storage_hm3 ({max_storage_hm3})"
                    ),
                });
            }
        }
    }

    Ok(())
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Precondition: [`validate_raw_hydros`] has returned `Ok(())` for this data.
fn convert_hydros(
    raw: RawHydroFile,
    global: &GlobalPenaltyDefaults,
    path: &Path,
) -> Result<Vec<Hydro>, LoadError> {
    let mut hydros: Vec<Hydro> = raw
        .hydros
        .into_iter()
        .enumerate()
        .map(|(i, raw_hydro)| {
            let operational_start_date = parse_operational_start_date(
                &raw_hydro.operational_start_date,
                path,
                &format!("hydros[{i}].operational_start_date"),
            )?;

            let (
                generation_model,
                min_turbined_m3s,
                max_turbined_m3s,
                min_generation_mw,
                max_generation_mw,
            ) = convert_generation(raw_hydro.generation);

            let tailrace = raw_hydro.tailrace.map(convert_tailrace);

            let hydraulic_losses = raw_hydro.hydraulic_losses.map(convert_hydraulic_losses);

            let efficiency = raw_hydro.efficiency.map(convert_efficiency);

            let (evaporation_coefficients_mm, evaporation_reference_volumes_hm3) =
                match raw_hydro.evaporation {
                    None => (None, None),
                    Some(evap) => {
                        let coeffs: [f64; 12] =
                            evap.coefficients_mm.try_into().unwrap_or_else(|_| {
                                unreachable!("evaporation length validated to be 12")
                            });
                        let ref_vols: Option<[f64; 12]> = evap.reference_volumes_hm3.map(|v| {
                            v.try_into().unwrap_or_else(|_| {
                                unreachable!("reference_volumes_hm3 length validated to be 12")
                            })
                        });
                        (Some(coeffs), ref_vols)
                    }
                };

            let diversion = raw_hydro.diversion.map(|d| DiversionChannel {
                downstream_id: EntityId(d.downstream_id),
                max_flow_m3s: d.max_flow_m3s,
            });

            let filling = raw_hydro.filling.map(|f| FillingConfig {
                start_stage_id: f.start_stage_id,
                filling_min_rate_m3s: f.filling_min_rate_m3s,
            });

            let entity_overrides: Option<HydroPenaltyOverrides> =
                raw_hydro.penalties.map(convert_penalty_overrides);
            let penalties = resolve_hydro_penalties(&entity_overrides, global);

            Ok(Hydro {
                id: EntityId(raw_hydro.id),
                name: raw_hydro.name,
                operational_start_date,
                bus_id: EntityId(raw_hydro.bus_id),
                downstream_id: raw_hydro.downstream_id.map(EntityId),
                entry_stage_id: raw_hydro.entry_stage_id,
                exit_stage_id: raw_hydro.exit_stage_id,
                min_storage_hm3: raw_hydro.reservoir.min_storage_hm3,
                max_storage_hm3: raw_hydro.reservoir.max_storage_hm3,
                min_outflow_m3s: raw_hydro.outflow.min_outflow_m3s,
                max_outflow_m3s: raw_hydro.outflow.max_outflow_m3s,
                generation_model,
                min_turbined_m3s,
                max_turbined_m3s,
                specific_productivity_mw_per_m3s_per_m: raw_hydro
                    .specific_productivity_mw_per_m3s_per_m,
                min_generation_mw,
                max_generation_mw,
                tailrace,
                hydraulic_losses,
                efficiency,
                evaporation_coefficients_mm,
                evaporation_reference_volumes_hm3,
                diversion,
                filling,
                penalties,
            })
        })
        .collect::<Result<_, LoadError>>()?;

    // Sort by id so this parser's output is deterministic regardless of file row
    // order (declaration-order invariance); id is the builder's canonical tiebreak.
    hydros.sort_by_key(|h| h.id.0);
    Ok(hydros)
}

/// Returns `(model, min_turbined_m3s, max_turbined_m3s, min_generation_mw, max_generation_mw)`.
// Clippy flags this as needless_pass_by_value, but the function consumes its
// argument by destructuring (moving fields out). Taking &RawGeneration would
// require cloning the copied f64 fields, which is no improvement.
#[allow(clippy::needless_pass_by_value)]
fn convert_generation(raw: RawGeneration) -> (HydroGenerationModel, f64, f64, f64, f64) {
    match raw {
        RawGeneration::ConstantProductivity {
            min_turbined_m3s,
            max_turbined_m3s,
            min_generation_mw,
            max_generation_mw,
            ..
        } => (
            HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s,
            max_turbined_m3s,
            min_generation_mw,
            max_generation_mw,
        ),
        RawGeneration::LinearizedHead {
            min_turbined_m3s,
            max_turbined_m3s,
            min_generation_mw,
            max_generation_mw,
            ..
        } => (
            HydroGenerationModel::LinearizedHead,
            min_turbined_m3s,
            max_turbined_m3s,
            min_generation_mw,
            max_generation_mw,
        ),
        RawGeneration::Fpha {
            min_turbined_m3s,
            max_turbined_m3s,
            min_generation_mw,
            max_generation_mw,
        } => (
            HydroGenerationModel::Fpha,
            min_turbined_m3s,
            max_turbined_m3s,
            min_generation_mw,
            max_generation_mw,
        ),
    }
}

fn convert_tailrace(raw: RawTailrace) -> TailraceModel {
    match raw {
        RawTailrace::Polynomial { coefficients } => TailraceModel::Polynomial { coefficients },
        RawTailrace::Piecewise { points } => TailraceModel::Piecewise {
            points: points
                .into_iter()
                .map(|p| TailracePoint {
                    outflow_m3s: p.outflow_m3s,
                    height_m: p.height_m,
                })
                .collect(),
        },
    }
}

// Clippy flags this as needless_pass_by_value; the function consumes its argument
// by destructuring (no heap allocation involved). Allow here since the by-value
// API correctly signals ownership transfer.
#[allow(clippy::needless_pass_by_value)]
fn convert_hydraulic_losses(raw: RawHydraulicLosses) -> HydraulicLossesModel {
    match raw {
        RawHydraulicLosses::Factor { value } => HydraulicLossesModel::Factor { value },
        RawHydraulicLosses::Constant { value_m } => HydraulicLossesModel::Constant { value_m },
    }
}

// Clippy flags this as needless_pass_by_value; same rationale as convert_hydraulic_losses.
#[allow(clippy::needless_pass_by_value)]
fn convert_efficiency(raw: RawEfficiency) -> EfficiencyModel {
    match raw {
        RawEfficiency::Constant { value } => EfficiencyModel::Constant { value },
    }
}

// Clippy flags this as needless_pass_by_value; the struct fields (Option<f64>) are
// all Copy, but taking by reference would require dereferencing every field.
// By-value is idiomatic for this conversion pattern.
#[allow(clippy::needless_pass_by_value)]
fn convert_penalty_overrides(raw: RawHydroPenaltyOverrides) -> HydroPenaltyOverrides {
    HydroPenaltyOverrides {
        spillage_cost: raw.spillage_cost,
        diversion_cost: raw.diversion_cost,
        turbined_cost: raw.turbined_cost,
        storage_violation_below_cost: raw.storage_violation_below_cost,
        filling_target_violation_cost: raw.filling_target_violation_cost,
        turbined_violation_below_cost: raw.turbined_violation_below_cost,
        outflow_violation_below_cost: raw.outflow_violation_below_cost,
        outflow_violation_above_cost: raw.outflow_violation_above_cost,
        generation_violation_below_cost: raw.generation_violation_below_cost,
        evaporation_violation_cost: raw.evaporation_violation_cost,
        water_withdrawal_violation_cost: raw.water_withdrawal_violation_cost,
        water_withdrawal_violation_pos_cost: raw.water_withdrawal_violation_cost,
        water_withdrawal_violation_neg_cost: raw.water_withdrawal_violation_cost,
        evaporation_violation_pos_cost: raw.evaporation_violation_cost,
        evaporation_violation_neg_cost: raw.evaporation_violation_cost,
        inflow_nonnegativity_cost: raw.inflow_nonnegativity_cost,
    }
}

/// Extract a field name hint from a `serde_json` error message.
///
/// Mirrors the implementation in `config.rs`. `serde_json` error messages follow
/// patterns such as:
/// - `"unknown variant 'foo', expected one of …"`
/// - `"missing field 'xyz' at line 1 column 2"`
///
/// This helper extracts the identifier between backticks, returning a best-effort
/// field name or `"<unknown>"` when no match is found.
fn extract_field_from_serde_msg(msg: &str) -> String {
    if let Some(start) = msg.find('`')
        && let Some(end) = msg[start + 1..].find('`')
    {
        return msg[start + 1..start + 1 + end].to_string();
    }
    "<unknown>".to_string()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::expect_used
)]
mod tests {
    use super::*;
    use cobre_core::entities::{DeficitSegment, HydroPenalties};
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Write a string to a temp file and return the file handle (keeps it alive).
    fn write_json(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    /// Build a canonical `GlobalPenaltyDefaults` for test use.
    fn make_global() -> GlobalPenaltyDefaults {
        GlobalPenaltyDefaults {
            bus_deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(500.0),
                    cost_per_mwh: 1000.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            bus_excess_cost: 100.0,
            line_exchange_cost: 2.0,
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                turbined_cost: 0.05,
                diversion_cost: 0.1,
                storage_violation_below_cost: 10_000.0,
                filling_target_violation_cost: 50_000.0,
                turbined_violation_below_cost: 500.0,
                outflow_violation_below_cost: 500.0,
                outflow_violation_above_cost: 500.0,
                generation_violation_below_cost: 1_000.0,
                evaporation_violation_cost: 5_000.0,
                water_withdrawal_violation_cost: 1_000.0,
                water_withdrawal_violation_pos_cost: 1_000.0,
                water_withdrawal_violation_neg_cost: 1_000.0,
                evaporation_violation_pos_cost: 5_000.0,
                evaporation_violation_neg_cost: 5_000.0,
                inflow_nonnegativity_cost: 1000.0,
            },
            ncs_curtailment_cost: 0.005,
        }
    }

    /// Minimal hydro entry (`constant_productivity`, no optional fields).
    const MINIMAL_HYDRO_JSON: &str = r#"{
      "id": 1,
      "name": "Minimal",
      "operational_start_date": "2024-01-01",
      "bus_id": 0,
      "downstream_id": null,
      "reservoir": { "min_storage_hm3": 100.0, "max_storage_hm3": 2000.0 },
      "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
      "generation": {
        "model": "constant_productivity",
        "min_turbined_m3s": 0.0,
        "max_turbined_m3s": 1000.0,
        "min_generation_mw": 0.0,
        "max_generation_mw": 750.0
      }
    }"#;

    /// Full hydro entry (all optional fields populated, polynomial tailrace).
    const FULL_HYDRO_JSON: &str = r#"{
      "id": 0,
      "name": "FURNAS",
      "operational_start_date": "2024-01-01",
      "bus_id": 0,
      "downstream_id": 2,
      "entry_stage_id": 1,
      "exit_stage_id": 600,
      "reservoir": { "min_storage_hm3": 5733.0, "max_storage_hm3": 22950.0 },
      "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": 4000.0 },
      "generation": {
        "model": "constant_productivity",
        "min_turbined_m3s": 0.0,
        "max_turbined_m3s": 1692.0,
        "min_generation_mw": 0.0,
        "max_generation_mw": 1312.0
      },
      "tailrace": { "type": "polynomial", "coefficients": [326.0, 0.0032, -1.2e-7] },
      "hydraulic_losses": { "type": "factor", "value": 0.03 },
      "efficiency": { "type": "constant", "value": 0.92 },
      "evaporation": { "coefficients_mm": [150, 130, 120, 90, 60, 40, 30, 40, 70, 100, 130, 150] },
      "diversion": { "downstream_id": 3, "max_flow_m3s": 200.0 },
      "filling": { "start_stage_id": 48, "filling_min_rate_m3s": 100.0 },
      "penalties": { "spillage_cost": 0.05 }
    }"#;

    // ── AC: parse valid hydros — full and minimal ──────────────────────────────

    /// Given a valid `hydros.json` with 2 hydros (one full, one minimal), `parse_hydros`
    /// returns `Ok(vec)` sorted by `id`; the full hydro has all optional fields mapped.
    #[test]
    fn test_parse_valid_full_and_minimal() {
        let json = format!(r#"{{ "hydros": [{FULL_HYDRO_JSON}, {MINIMAL_HYDRO_JSON}] }}"#);
        let f = write_json(&json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        assert_eq!(hydros.len(), 2);

        let h0 = &hydros[0];
        assert_eq!(h0.id, EntityId(0));
        assert_eq!(h0.name, "FURNAS");
        assert_eq!(h0.bus_id, EntityId(0));
        assert_eq!(h0.downstream_id, Some(EntityId(2)));
        assert_eq!(h0.entry_stage_id, Some(1));
        assert_eq!(h0.exit_stage_id, Some(600));
        assert!((h0.min_storage_hm3 - 5733.0).abs() < f64::EPSILON);
        assert!((h0.max_storage_hm3 - 22950.0).abs() < f64::EPSILON);
        assert!((h0.min_outflow_m3s - 0.0).abs() < f64::EPSILON);
        assert_eq!(h0.max_outflow_m3s, Some(4000.0));
        assert!(
            matches!(
                h0.generation_model,
                HydroGenerationModel::ConstantProductivity
            ),
            "expected ConstantProductivity generation model"
        );
        assert!((h0.min_turbined_m3s - 0.0).abs() < f64::EPSILON);
        assert!((h0.max_turbined_m3s - 1692.0).abs() < f64::EPSILON);
        assert!((h0.min_generation_mw - 0.0).abs() < f64::EPSILON);
        assert!((h0.max_generation_mw - 1312.0).abs() < f64::EPSILON);
        assert!(
            matches!(&h0.tailrace, Some(TailraceModel::Polynomial { coefficients }) if coefficients.len() == 3),
            "expected Polynomial tailrace with 3 coefficients"
        );
        assert!(matches!(
            h0.hydraulic_losses,
            Some(HydraulicLossesModel::Factor { value }) if (value - 0.03).abs() < f64::EPSILON
        ));
        assert!(matches!(
            h0.efficiency,
            Some(EfficiencyModel::Constant { value }) if (value - 0.92).abs() < f64::EPSILON
        ));
        assert!(h0.evaporation_coefficients_mm.is_some());
        assert_eq!(h0.evaporation_coefficients_mm.map(|a| a.len()), Some(12));
        assert!(matches!(
            &h0.diversion,
            Some(DiversionChannel { downstream_id, max_flow_m3s })
            if *downstream_id == EntityId(3) && (max_flow_m3s - 200.0).abs() < f64::EPSILON
        ));
        assert!(matches!(
            &h0.filling,
            Some(FillingConfig { start_stage_id: 48, filling_min_rate_m3s })
            if (filling_min_rate_m3s - 100.0).abs() < f64::EPSILON
        ));
        assert!((h0.penalties.spillage_cost - 0.05).abs() < f64::EPSILON);
        assert!((h0.penalties.diversion_cost - 0.1).abs() < f64::EPSILON);

        let h1 = &hydros[1];
        assert_eq!(h1.id, EntityId(1));
        assert_eq!(h1.name, "Minimal");
        assert_eq!(h1.downstream_id, None);
        assert_eq!(h1.entry_stage_id, None);
        assert_eq!(h1.exit_stage_id, None);
        assert_eq!(h1.max_outflow_m3s, None);
        assert!(h1.tailrace.is_none());
        assert!(h1.hydraulic_losses.is_none());
        assert!(h1.efficiency.is_none());
        assert!(h1.evaporation_coefficients_mm.is_none());
        assert!(h1.diversion.is_none());
        assert!(h1.filling.is_none());
        assert!((h1.penalties.spillage_cost - 0.01).abs() < f64::EPSILON);
    }

    // ── AC: FPHA generation model ──────────────────────────────────────────────

    /// Given a hydro with `generation.model: "fpha"`, the resulting `Hydro` has
    /// `generation_model: HydroGenerationModel::Fpha` (no `productivity_mw_per_m3s`).
    #[test]
    fn test_parse_fpha_generation_model() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "FPHA Plant", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 100.0, "max_storage_hm3": 5000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "fpha",
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 2000.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 8000.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        assert_eq!(hydros.len(), 1);
        assert_eq!(hydros[0].generation_model, HydroGenerationModel::Fpha);
        assert!((hydros[0].min_turbined_m3s - 0.0).abs() < f64::EPSILON);
        assert!((hydros[0].max_turbined_m3s - 2000.0).abs() < f64::EPSILON);
    }

    // ── AC: linearized_head generation model ─────────────────────────────────

    /// Given a hydro with `generation.model: "linearized_head"`, it parses correctly.
    #[test]
    fn test_parse_linearized_head_generation_model() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "LH Plant", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "linearized_head",
              "min_turbined_m3s": 100.0,
              "max_turbined_m3s": 3000.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 1950.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        assert_eq!(hydros.len(), 1);
        assert!(
            matches!(
                &hydros[0].generation_model,
                HydroGenerationModel::LinearizedHead
            ),
            "expected LinearizedHead generation model"
        );
    }

    // ── AC: tailrace piecewise variant ────────────────────────────────────────

    /// Tailrace with `"type": "piecewise"` is parsed correctly.
    #[test]
    fn test_parse_tailrace_piecewise() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Piecewise", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 250.0
            },
            "tailrace": {
              "type": "piecewise",
              "points": [
                { "outflow_m3s": 0.0, "height_m": 3.0 },
                { "outflow_m3s": 5000.0, "height_m": 4.5 }
              ]
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        assert!(
            matches!(
                &hydros[0].tailrace,
                Some(TailraceModel::Piecewise { points }) if points.len() == 2
            ),
            "expected Piecewise tailrace with 2 points"
        );
    }

    // ── AC: hydraulic losses constant variant ─────────────────────────────────

    /// Hydraulic losses with `"type": "constant"` is parsed correctly.
    #[test]
    fn test_parse_hydraulic_losses_constant() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "ConstantLoss", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 500.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 250.0
            },
            "hydraulic_losses": { "type": "constant", "value_m": 2.5 }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        assert!(
            matches!(
                hydros[0].hydraulic_losses,
                Some(HydraulicLossesModel::Constant { value_m }) if (value_m - 2.5).abs() < f64::EPSILON
            ),
            "expected Constant hydraulic losses with value_m = 2.5"
        );
    }

    // ── AC: entity-level penalty partial override ─────────────────────────────

    /// Entity-level penalty partial override: overridden fields use entity value,
    /// non-overridden fields use global default.
    #[test]
    fn test_entity_level_penalty_partial_override() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Override", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 250.0
            },
            "penalties": { "spillage_cost": 0.05 }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        assert!(
            (hydros[0].penalties.spillage_cost - 0.05).abs() < f64::EPSILON,
            "spillage_cost should be 0.05 (entity override)"
        );
        assert!(
            (hydros[0].penalties.diversion_cost - 0.1).abs() < f64::EPSILON,
            "diversion_cost should be 0.1 (global default)"
        );
        assert!(
            (hydros[0].penalties.storage_violation_below_cost - 10_000.0).abs() < f64::EPSILON,
            "storage_violation_below_cost should be 10_000.0 (global default)"
        );
    }

    // ── AC: entity-level penalty all-default (no penalties block) ─────────────

    /// No `penalties` block in JSON → all hydro penalties use global defaults.
    #[test]
    fn test_entity_level_penalty_all_global_defaults() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "NoOverride", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 250.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        let g = &global.hydro;
        let p = &hydros[0].penalties;
        assert!((p.spillage_cost - g.spillage_cost).abs() < f64::EPSILON);
        assert!((p.diversion_cost - g.diversion_cost).abs() < f64::EPSILON);
        assert!((p.turbined_cost - g.turbined_cost).abs() < f64::EPSILON);
        assert!(
            (p.storage_violation_below_cost - g.storage_violation_below_cost).abs() < f64::EPSILON
        );
        assert!(
            (p.filling_target_violation_cost - g.filling_target_violation_cost).abs()
                < f64::EPSILON
        );
        assert!(
            (p.turbined_violation_below_cost - g.turbined_violation_below_cost).abs()
                < f64::EPSILON
        );
        assert!(
            (p.outflow_violation_below_cost - g.outflow_violation_below_cost).abs() < f64::EPSILON
        );
        assert!(
            (p.outflow_violation_above_cost - g.outflow_violation_above_cost).abs() < f64::EPSILON
        );
        assert!(
            (p.generation_violation_below_cost - g.generation_violation_below_cost).abs()
                < f64::EPSILON
        );
        assert!((p.evaporation_violation_cost - g.evaporation_violation_cost).abs() < f64::EPSILON);
        assert!(
            (p.water_withdrawal_violation_cost - g.water_withdrawal_violation_cost).abs()
                < f64::EPSILON
        );
    }

    // ── AC: filling config with absent filling_min_rate_m3s ───────────────────

    /// Filling config with no `filling_min_rate_m3s` field defaults to 0.0 (passive filling).
    #[test]
    fn test_filling_min_rate_defaults_to_zero() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Fill", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 250.0
            },
            "filling": { "start_stage_id": 10 }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        assert!(matches!(
            &hydros[0].filling,
            Some(FillingConfig {
                start_stage_id: 10,
                filling_min_rate_m3s,
            }) if (*filling_min_rate_m3s - 0.0).abs() < f64::EPSILON
        ));
    }

    // ── AC: duplicate ID detection ─────────────────────────────────────────────

    /// Given `hydros.json` with duplicate `id` values, `parse_hydros` returns
    /// `Err(LoadError::SchemaError)` with field containing `"hydros[N].id"` and
    /// message containing `"duplicate"`.
    #[test]
    fn test_duplicate_hydro_id() {
        let entry = r#"{
          "id": 5, "name": "Alpha", "bus_id": 0,
          "operational_start_date": "2024-01-01",
          "downstream_id": null,
          "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
          "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
          "generation": {
            "model": "constant_productivity",
            "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
            "min_generation_mw": 0.0, "max_generation_mw": 250.0
          }
        }"#;
        let json = format!(r#"{{ "hydros": [{entry}, {entry}] }}"#);
        let f = write_json(&json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("hydros[1].id"),
                    "field should contain 'hydros[1].id', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should contain 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: reservoir validation ───────────────────────────────────────────────

    /// `min_storage_hm3 > max_storage_hm3` → `SchemaError` with field containing
    /// `"reservoir"` and message containing `"min_storage_hm3"`.
    #[test]
    fn test_invalid_reservoir_bounds_min_gt_max() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Bad", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 5000.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0, "max_generation_mw": 250.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("reservoir"),
                    "field should contain 'reservoir', got: {field}"
                );
                assert!(
                    message.contains("min_storage_hm3"),
                    "message should contain 'min_storage_hm3', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Negative `min_storage_hm3` → `SchemaError`.
    #[test]
    fn test_invalid_reservoir_negative_min() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Bad", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": -1.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0, "max_generation_mw": 250.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        assert!(
            matches!(&err, LoadError::SchemaError { field, .. } if field.contains("min_storage_hm3")),
            "expected SchemaError for negative min_storage_hm3, got: {err:?}"
        );
    }

    /// Negative `max_storage_hm3` → `SchemaError`.
    #[test]
    fn test_invalid_reservoir_negative_max() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Bad", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": -100.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0, "max_generation_mw": 250.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        assert!(
            matches!(&err, LoadError::SchemaError { field, .. } if field.contains("max_storage_hm3")),
            "expected SchemaError for negative max_storage_hm3, got: {err:?}"
        );
    }

    // ── AC: generation bound validation ───────────────────────────────────────

    /// `max_generation_mw < min_generation_mw` → `SchemaError`.
    #[test]
    fn test_invalid_generation_bounds_max_lt_min() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Bad", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
              "min_generation_mw": 500.0, "max_generation_mw": 100.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("max_generation_mw"),
                    "field should contain 'max_generation_mw', got: {field}"
                );
                assert!(
                    message.contains("min_generation_mw"),
                    "message should reference min_generation_mw, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `max_turbined_m3s < min_turbined_m3s` → `SchemaError`.
    #[test]
    fn test_invalid_turbined_bounds_max_lt_min() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Bad", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 600.0, "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0, "max_generation_mw": 250.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        assert!(
            matches!(&err, LoadError::SchemaError { field, .. } if field.contains("max_turbined_m3s")),
            "expected SchemaError for max_turbined < min_turbined, got: {err:?}"
        );
    }

    /// Negative `min_outflow_m3s` → `SchemaError`.
    #[test]
    fn test_invalid_outflow_negative_min() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Bad", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": -10.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0, "max_generation_mw": 250.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        assert!(
            matches!(&err, LoadError::SchemaError { field, .. } if field.contains("min_outflow_m3s")),
            "expected SchemaError for negative min_outflow_m3s, got: {err:?}"
        );
    }

    // ── AC: evaporation array length validation ────────────────────────────────

    /// Evaporation array with wrong element count → `SchemaError`.
    #[test]
    fn test_invalid_evaporation_wrong_length() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Bad", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0, "max_generation_mw": 250.0
            },
            "evaporation": { "coefficients_mm": [10.0, 20.0] }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("coefficients_mm"),
                    "field should contain 'coefficients_mm', got: {field}"
                );
                assert!(
                    message.contains("12"),
                    "message should mention 12 elements, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: unknown generation model → SchemaError ────────────────────────────

    /// Unknown `generation.model` value → `SchemaError`.
    #[test]
    fn test_unknown_generation_model() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Bad", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "unknown_model_xyz",
              "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0, "max_generation_mw": 250.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "unknown generation model should produce SchemaError, got: {err:?}"
        );
    }

    // ── AC: declaration-order invariance ──────────────────────────────────────

    /// Given hydros in reverse ID order in JSON, `parse_hydros` returns a
    /// `Vec<Hydro>` sorted by ascending `id`.
    #[test]
    fn test_declaration_order_invariance() {
        let entry_a = r#"{
          "id": 0, "name": "Alpha", "bus_id": 0,
          "operational_start_date": "2024-01-01",
          "downstream_id": null,
          "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 500.0 },
          "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
          "generation": {
            "model": "constant_productivity",
            "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
            "min_generation_mw": 0.0, "max_generation_mw": 250.0
          }
        }"#;
        let entry_b = r#"{
          "id": 1, "name": "Beta", "bus_id": 0,
          "operational_start_date": "2024-01-01",
          "downstream_id": null,
          "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
          "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
          "generation": {
            "model": "constant_productivity",
            "min_turbined_m3s": 0.0, "max_turbined_m3s": 1000.0,
            "min_generation_mw": 0.0, "max_generation_mw": 800.0
          }
        }"#;

        let json_forward = format!(r#"{{ "hydros": [{entry_a}, {entry_b}] }}"#);
        let json_reversed = format!(r#"{{ "hydros": [{entry_b}, {entry_a}] }}"#);
        let global = make_global();

        let f1 = write_json(&json_forward);
        let f2 = write_json(&json_reversed);
        let hydros1 = parse_hydros(f1.path(), &global).unwrap();
        let hydros2 = parse_hydros(f2.path(), &global).unwrap();

        assert_eq!(
            hydros1, hydros2,
            "results must be identical regardless of input ordering"
        );
        assert_eq!(hydros1[0].id, EntityId(0));
        assert_eq!(hydros1[1].id, EntityId(1));
    }

    // ── AC: file not found → IoError ─────────────────────────────────────────

    /// Given a nonexistent path, `parse_hydros` returns `Err(LoadError::IoError)`.
    #[test]
    fn test_file_not_found() {
        let path = Path::new("/nonexistent/system/hydros.json");
        let global = make_global();
        let err = parse_hydros(path, &global).unwrap_err();
        match &err {
            LoadError::IoError { path: p, .. } => {
                assert_eq!(p, path);
            }
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    // ── AC: invalid JSON → ParseError ─────────────────────────────────────────

    /// Given invalid JSON, `parse_hydros` returns `Err(LoadError::ParseError)`.
    #[test]
    fn test_invalid_json() {
        let f = write_json(r#"{"hydros": [not valid json}}"#);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for invalid JSON, got: {err:?}"
        );
    }

    // ── Additional edge cases ─────────────────────────────────────────────────

    /// Empty `hydros` array is valid — returns an empty Vec.
    #[test]
    fn test_empty_hydros_array() {
        let json = r#"{ "hydros": [] }"#;
        let f = write_json(json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();
        assert!(hydros.is_empty());
    }

    /// `min_storage_hm3 == max_storage_hm3` (degenerate reservoir) is valid.
    #[test]
    fn test_reservoir_min_equals_max_is_valid() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Deg", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 500.0, "max_storage_hm3": 500.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0, "max_generation_mw": 250.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let result = parse_hydros(f.path(), &global);
        assert!(
            result.is_ok(),
            "min_storage == max_storage should be valid, got: {result:?}"
        );
    }

    // ── AC: evaporation reference_volumes_hm3 ─────────────────────────────────

    /// Helper: reservoir with min=1000 hm³, max=20000 hm³ and evaporation coefficients.
    const HYDRO_WITH_EVAP_BASE: &str = r#"{
      "id": 0, "name": "ReservoirEvap", "bus_id": 0,
      "operational_start_date": "2024-01-01",
      "downstream_id": null,
      "reservoir": { "min_storage_hm3": 1000.0, "max_storage_hm3": 20000.0 },
      "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
      "generation": {
        "model": "constant_productivity",
        "min_turbined_m3s": 0.0,
        "max_turbined_m3s": 1000.0,
        "min_generation_mw": 0.0,
        "max_generation_mw": 750.0
      }
    }"#;

    /// Given a `hydros.json` with `evaporation.reference_volumes_hm3` containing 12
    /// valid values within reservoir bounds, the returned `Hydro` has
    /// `evaporation_reference_volumes_hm3 == Some([f64; 12])`.
    #[test]
    fn test_evaporation_reference_volumes_happy_path() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "ReservoirEvap", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 1000.0, "max_storage_hm3": 20000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 1000.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 750.0
            },
            "evaporation": {
              "coefficients_mm": [150, 130, 120, 90, 60, 40, 30, 40, 70, 100, 130, 150],
              "reference_volumes_hm3": [15000, 12000, 10000, 8000, 6000, 5000, 5500, 7000, 9000, 11000, 13000, 14500]
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        let ref_vols = hydros[0]
            .evaporation_reference_volumes_hm3
            .expect("reference_volumes_hm3 should be Some");
        assert_eq!(ref_vols.len(), 12);
        assert!((ref_vols[0] - 15000.0).abs() < f64::EPSILON);
        assert!((ref_vols[5] - 5000.0).abs() < f64::EPSILON);
        assert!((ref_vols[11] - 14500.0).abs() < f64::EPSILON);
    }

    /// Given a `hydros.json` where the evaporation block has no
    /// `reference_volumes_hm3` key, the returned `Hydro` has
    /// `evaporation_reference_volumes_hm3 == None` (backward compatible).
    #[test]
    fn test_evaporation_reference_volumes_absent_is_none() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "ReservoirEvap", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 1000.0, "max_storage_hm3": 20000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 1000.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 750.0
            },
            "evaporation": { "coefficients_mm": [150, 130, 120, 90, 60, 40, 30, 40, 70, 100, 130, 150] }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        assert!(
            hydros[0].evaporation_reference_volumes_hm3.is_none(),
            "reference_volumes_hm3 should be None when key is absent from JSON"
        );
    }

    /// Given `reference_volumes_hm3` with 11 elements (wrong length), `parse_hydros`
    /// returns `LoadError::SchemaError` with a message containing "exactly 12 elements".
    #[test]
    fn test_evaporation_reference_volumes_wrong_length() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "ReservoirEvap", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 1000.0, "max_storage_hm3": 20000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 1000.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 750.0
            },
            "evaporation": {
              "coefficients_mm": [150, 130, 120, 90, 60, 40, 30, 40, 70, 100, 130, 150],
              "reference_volumes_hm3": [10000, 9000, 8000, 7000, 6000, 5000, 5500, 6500, 7500, 8500, 9500]
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("reference_volumes_hm3"),
                    "field should contain 'reference_volumes_hm3', got: {field}"
                );
                assert!(
                    message.contains("exactly 12 elements"),
                    "message should contain 'exactly 12 elements', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Given `reference_volumes_hm3` with a NaN value, `parse_hydros` returns
    /// `LoadError::SchemaError`.
    #[test]
    fn test_evaporation_reference_volumes_nan_value() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "ReservoirEvap", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 1000.0, "max_storage_hm3": 20000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 1000.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 750.0
            },
            "evaporation": {
              "coefficients_mm": [150, 130, 120, 90, 60, 40, 30, 40, 70, 100, 130, 150],
              "reference_volumes_hm3": [null, 9000, 8000, 7000, 6000, 5000, 5500, 6500, 7500, 8500, 9500, 10000]
            }
          }]
        }"#;
        // NaN is injected via RawEvaporation directly — JSON has no NaN literal.
        let _ = json;

        let evap = RawEvaporation {
            coefficients_mm: vec![0.0; 12],
            reference_volumes_hm3: Some({
                let mut v = vec![5000.0; 12];
                v[3] = f64::NAN;
                v
            }),
        };
        let path = Path::new("hydros.json");
        let err = validate_evaporation(&evap, 0, path, 1000.0, 20000.0).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("reference_volumes_hm3"),
                    "field should contain 'reference_volumes_hm3', got: {field}"
                );
                assert!(
                    message.contains("finite"),
                    "message should contain 'finite', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Given `reference_volumes_hm3` with a value exceeding `max_storage_hm3`,
    /// `parse_hydros` returns `LoadError::SchemaError` with a message containing
    /// "`max_storage_hm3`".
    #[test]
    fn test_evaporation_reference_volumes_exceeds_max_storage() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "ReservoirEvap", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 1000.0, "max_storage_hm3": 20000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 1000.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 750.0
            },
            "evaporation": {
              "coefficients_mm": [150, 130, 120, 90, 60, 40, 30, 40, 70, 100, 130, 150],
              "reference_volumes_hm3": [25000, 12000, 10000, 8000, 6000, 5000, 5500, 7000, 9000, 11000, 13000, 14500]
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("reference_volumes_hm3"),
                    "field should contain 'reference_volumes_hm3', got: {field}"
                );
                assert!(
                    message.contains("max_storage_hm3"),
                    "message should contain 'max_storage_hm3', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Given `reference_volumes_hm3` with a value below `min_storage_hm3`,
    /// `parse_hydros` returns `LoadError::SchemaError` with a message containing
    /// "`min_storage_hm3`".
    #[test]
    fn test_evaporation_reference_volumes_below_min_storage() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "ReservoirEvap", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 1000.0, "max_storage_hm3": 20000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 1000.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 750.0
            },
            "evaporation": {
              "coefficients_mm": [150, 130, 120, 90, 60, 40, 30, 40, 70, 100, 130, 150],
              "reference_volumes_hm3": [500, 12000, 10000, 8000, 6000, 5000, 5500, 7000, 9000, 11000, 13000, 14500]
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("reference_volumes_hm3"),
                    "field should contain 'reference_volumes_hm3', got: {field}"
                );
                assert!(
                    message.contains("min_storage_hm3"),
                    "message should contain 'min_storage_hm3', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Additional edge case: no evaporation block → reference volumes also None ─

    /// When the entire `evaporation` block is absent, both evaporation fields are
    /// `None` (backward compatible).
    #[test]
    fn test_no_evaporation_block_both_fields_none() {
        let json = format!(r#"{{ "hydros": [{HYDRO_WITH_EVAP_BASE}] }}"#);
        let f = write_json(&json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        assert!(hydros[0].evaporation_coefficients_mm.is_none());
        assert!(hydros[0].evaporation_reference_volumes_hm3.is_none());
    }

    // ── AC: legacy productivity_mw_per_m3s is rejected ───────────────────────

    /// Given a `hydros.json` whose `generation` block contains the legacy
    /// `productivity_mw_per_m3s` field, `parse_hydros` returns an `Err` whose
    /// message contains the substring `"productivity_mw_per_m3s"`.
    ///
    /// This guards against silent acceptance of stale case files: the field is
    /// no longer a member of any `RawGeneration` variant, and
    /// `#[serde(deny_unknown_fields)]` ensures the presence of the legacy key
    /// surfaces as a hard parse error rather than being silently ignored.
    #[test]
    fn hydros_json_rejects_legacy_inline_productivity() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "LegacyPlant", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 1.0,
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 100.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 100.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("productivity_mw_per_m3s"),
            "error message should name the rejected field 'productivity_mw_per_m3s', got: {msg}"
        );
    }

    /// `$schema` field is accepted and ignored.
    #[test]
    fn test_schema_field_is_ignored() {
        let json = r#"{
          "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/hydros.schema.json",
          "hydros": [{
            "id": 0, "name": "H", "bus_id": 0,
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 100.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "min_turbined_m3s": 0.0, "max_turbined_m3s": 100.0,
              "min_generation_mw": 0.0, "max_generation_mw": 50.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let result = parse_hydros(f.path(), &global);
        assert!(
            result.is_ok(),
            "$schema field should be ignored, got: {result:?}"
        );
    }
}
