//! Production model resolution: per-`(hydro, stage)` constant productivity or FPHA.
//!
//! Resolves each hydro's production function from the case directory: constant
//! productivity from the entity definition / parquet override, precomputed FPHA
//! hyperplanes, or FPHA hyperplanes fitted from reservoir geometry via the
//! `crate::fpha_fitting` pipeline. Produces the `ProductionModelSet`, the
//! per-hydro `ProductionModelSource` provenance, the `ρ_eq` override carried for
//! energy-conversion derivation, low-kappa warnings, and the computed-FPHA
//! export rows.

use std::collections::HashMap;
use std::path::Path;

use cobre_core::{EntityId, System, entities::hydro::HydroGenerationModel};
use cobre_io::extensions::{
    FphaColumnLayout, FphaHyperplaneRow, HydroGeometryRow, ProductionModelConfig, SelectionMode,
};

use super::load_artifacts_for_hydro_models;
use super::types::{FphaPlane, ProductionModelSet, ProductionModelSource, ResolvedProductionModel};
use crate::SddpError;
use crate::fpha_fitting::{FphaFitResult, fit_fpha_planes};
// ── FPHA production model resolution ─────────────────────────────────────────

/// Return type for [`resolve_production_models`]: the model set, provenance vector,
/// low-kappa warnings, and computed-FPHA export rows.
///
/// The export rows are non-empty only when at least one hydro uses
/// `source: "computed"`.  The write site is the calling entry point;
/// `resolve_production_models` never performs any I/O.
type ResolveProductionResult = (
    ProductionModelSet,
    crate::energy_conversion::HydroEnergyProductivityOverride,
    Vec<(EntityId, ProductionModelSource)>,
    Vec<(String, f64)>,
    Vec<cobre_io::FphaHyperplaneRow>,
);

/// Resolve per-hydro per-stage production models from the case directory.
///
/// Reads `system/hydro_production_models.json` when present (optional file).
/// If absent, all hydros fall back to the [`HydroGenerationModel`] from their
/// entity definition in `system/hydros.json`. When any hydro is configured as
/// FPHA with `source: "precomputed"`, also loads `system/fpha_hyperplanes.parquet`.
/// When any hydro is configured as FPHA with `source: "computed"`, also loads
/// `system/hydro_geometry.parquet` and runs the FPHA fitting pipeline.
///
/// Returns `(ProductionModelSet, provenance_vec, kappa_warnings)` where the
/// provenance vector records the [`ProductionModelSource`] for each hydro in
/// canonical ID order, and `kappa_warnings` contains `(name, kappa)` pairs for
/// any computed FPHA hydro whose fitted envelope had kappa < 0.95.
///
/// # Model resolution per hydro
///
/// For each hydro in `system.hydros()` (canonical ID order):
///
/// 1. If `hydro_production_models.json` has an entry for this hydro:
///    - `source: "precomputed"` → load hyperplanes from `fpha_hyperplanes.parquet`,
///      scale `gamma_0` by `kappa`, record [`ProductionModelSource::PrecomputedHyperplanes`].
///    - `source: "computed"` → fit hyperplanes from `hydro_geometry.parquet` via the
///      FPHA fitting pipeline, record [`ProductionModelSource::ComputedFromGeometry`].
/// 2. Otherwise, use the [`HydroGenerationModel`] from the entity definition:
///    - [`HydroGenerationModel::ConstantProductivity`] →
///      [`ResolvedProductionModel::ConstantProductivity`].
///    - [`HydroGenerationModel::LinearizedHead`] →
///      [`ResolvedProductionModel::ConstantProductivity`] (uses the productivity field).
///    - [`HydroGenerationModel::Fpha`] without a config entry →
///      [`SddpError::Validation`] (no hyperplane source specified).
///
/// # Errors
///
/// | Condition                                                       | Error variant              |
/// | --------------------------------------------------------------- | -------------------------- |
/// | `Fpha` entity model with no config entry                        | [`SddpError::Validation`]  |
/// | `source: "computed"` with missing tailrace/losses/efficiency    | [`SddpError::Validation`]  |
/// | `source: "computed"` with no geometry rows for the hydro        | [`SddpError::Validation`]  |
/// | FPHA fitting pipeline error                                     | [`SddpError::Validation`]  |
/// | `gamma_v <= 0` for any precomputed hyperplane                   | [`SddpError::Validation`]  |
/// | `gamma_s > 0` for any precomputed hyperplane                    | [`SddpError::Validation`]  |
/// | `gamma_q <= 0` for any precomputed hyperplane                   | [`SddpError::Validation`]  |
/// | `kappa` not in `(0, 1]` for precomputed hyperplane              | [`SddpError::Validation`]  |
/// | Zero hyperplanes for an FPHA hydro at any stage                 | [`SddpError::Validation`]  |
/// | I/O failure loading JSON or Parquet                             | [`SddpError::Io`]          |
pub fn resolve_production_models(
    system: &System,
    case_dir: &Path,
) -> Result<ResolveProductionResult, SddpError> {
    let artifacts = load_artifacts_for_hydro_models(case_dir)?;
    resolve_production_models_from_artifacts(system, &artifacts)
}

/// Variant of [`resolve_production_models`] that consumes a pre-parsed
/// [`cobre_io::CaseArtifacts`] bundle.
///
/// # Errors
///
/// Same conditions as [`resolve_production_models`].
pub fn resolve_production_models_from_artifacts(
    system: &System,
    artifacts: &cobre_io::CaseArtifacts,
) -> Result<ResolveProductionResult, SddpError> {
    let override_table = crate::energy_conversion::build_hydro_energy_productivity_override(
        &artifacts.hydro_energy_productivity,
    )
    .map_err(|e| SddpError::Validation(e.to_string()))?;

    // Borrow from the artifacts bundle; no clone of the config rows is
    // needed because the per-hydro maps below only need references.
    let prod_configs: &[ProductionModelConfig] = &artifacts.production_models;

    let config_map: HashMap<EntityId, &ProductionModelConfig> =
        prod_configs.iter().map(|c| (c.hydro_id, c)).collect();

    let mut hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
        HashMap::new();
    if prod_configs.iter().any(config_uses_precomputed_fpha) {
        for row in &artifacts.fpha_hyperplanes {
            hyperplane_map
                .entry((row.hydro_id, row.stage_id))
                .or_default()
                .push(row);
        }
    }

    let geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> =
        if prod_configs.iter().any(config_uses_computed_fpha) {
            build_geometry_map(&artifacts.hydro_geometry)
        } else {
            HashMap::new()
        };

    // Study stages only (id >= 0), in canonical order.
    let study_stages: Vec<&cobre_core::temporal::Stage> =
        system.stages().iter().filter(|s| s.id >= 0).collect();
    let n_stages = study_stages.len();
    let n_hydros = system.hydros().len();

    let mut all_models: Vec<Vec<ResolvedProductionModel>> = Vec::with_capacity(n_hydros);
    let mut provenance: Vec<(EntityId, ProductionModelSource)> = Vec::with_capacity(n_hydros);
    let mut export_rows: Vec<cobre_io::FphaHyperplaneRow> = Vec::new();
    let mut kappa_warnings: Vec<(String, f64)> = Vec::new();

    for hydro in system.hydros() {
        let config_entry = config_map.get(&hydro.id).copied();

        let source = determine_source(hydro, config_entry)?;
        provenance.push((hydro.id, source));

        // Fit computed-source planes once per hydro, reuse for each stage.
        let cached_computed_planes: Option<Vec<FphaPlane>> =
            if source == ProductionModelSource::ComputedFromGeometry {
                let fit_result =
                    fit_planes_for_hydro(hydro, config_entry, &geometry_map, &study_stages)?;
                if let Some(kappa) = fit_result.low_kappa_warning {
                    kappa_warnings.push((hydro.name.clone(), kappa));
                }
                for (plane_id, plane) in fit_result.planes.iter().enumerate() {
                    let raw_gamma_0 = plane.intercept / fit_result.kappa;
                    // Rationale: plane_id comes from enumerate() over the fitting
                    // result; plane counts are bounded by max_planes_per_hydro
                    // (default <= 30), far below i32::MAX, so truncation and wrap
                    // are unreachable.
                    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                    export_rows.push(cobre_io::FphaHyperplaneRow {
                        hydro_id: hydro.id,
                        stage_id: None,
                        plane_id: plane_id as i32,
                        gamma_0: raw_gamma_0,
                        gamma_v: plane.gamma_v,
                        gamma_q: plane.gamma_q,
                        gamma_s: plane.gamma_s,
                        kappa: fit_result.kappa,
                        valid_v_min_hm3: None,
                        valid_v_max_hm3: None,
                        valid_q_max_m3s: None,
                    });
                }
                Some(fit_result.planes)
            } else {
                None
            };

        let mut stage_models: Vec<ResolvedProductionModel> = Vec::with_capacity(n_stages);
        for stage in &study_stages {
            let model = resolve_stage_model(
                hydro,
                stage,
                config_entry,
                source,
                &hyperplane_map,
                cached_computed_planes.as_deref(),
                Some(&override_table),
            )?;
            stage_models.push(model);
        }

        all_models.push(stage_models);
    }

    let set = ProductionModelSet::new(all_models, n_hydros, n_stages);
    Ok((set, override_table, provenance, kappa_warnings, export_rows))
}

/// Build an `O(1)` geometry map: `hydro_id → sorted geometry row references`.
fn build_geometry_map(
    geometry_rows: &[HydroGeometryRow],
) -> HashMap<EntityId, Vec<&HydroGeometryRow>> {
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    for row in geometry_rows {
        geometry_map.entry(row.hydro_id).or_default().push(row);
    }
    for rows in geometry_map.values_mut() {
        rows.sort_by(|a, b| a.volume_hm3.total_cmp(&b.volume_hm3));
    }
    geometry_map
}

/// Fit FPHA planes from geometry for a computed-source hydro.
/// Validates prerequisites (tailrace, losses, efficiency present), then calls
/// `fit_fpha_planes`. Returns planes, kappa, and warnings for caching per hydro.
fn fit_planes_for_hydro(
    hydro: &cobre_core::entities::hydro::Hydro,
    config_entry: Option<&ProductionModelConfig>,
    geometry_map: &HashMap<EntityId, Vec<&HydroGeometryRow>>,
    study_stages: &[&cobre_core::temporal::Stage],
) -> Result<FphaFitResult, SddpError> {
    validate_computed_prerequisites(hydro, geometry_map)?;

    // Use the first study stage as representative for FphaColumnLayout lookup.
    // In the MVP, the fitting result is stage-independent.
    let representative_stage = study_stages.first().ok_or_else(|| {
        SddpError::Validation(format!(
            "hydro {} (id={}) has source: \"computed\" but the system has no study stages",
            hydro.name, hydro.id.0
        ))
    })?;

    let config = config_entry
        .and_then(|c| find_fpha_config_for_stage(c, representative_stage))
        .ok_or_else(|| {
            SddpError::Validation(format!(
                "hydro {} (id={}) has source: \"computed\" but no FphaColumnLayout \
                 was found in hydro_production_models.json",
                hydro.name, hydro.id.0
            ))
        })?;

    // Clone geometry rows from map to satisfy fit_fpha_planes signature.
    let geo_rows_owned: Vec<HydroGeometryRow> = geometry_map
        .get(&hydro.id)
        .map_or(&[][..], Vec::as_slice)
        .iter()
        .map(|r| (*r).clone())
        .collect();

    Ok(fit_fpha_planes(&geo_rows_owned, hydro, config)?)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Return `true` if the config entry uses `source: "precomputed"` FPHA in any
/// stage range or season entry.
fn config_uses_precomputed_fpha(config: &ProductionModelConfig) -> bool {
    match &config.selection_mode {
        SelectionMode::StageRanges { ranges } => ranges.iter().any(|r| {
            r.fpha_config
                .as_ref()
                .is_some_and(|f| f.source == "precomputed")
        }),
        SelectionMode::Seasonal { seasons, .. } => seasons.iter().any(|s| {
            s.fpha_config
                .as_ref()
                .is_some_and(|f| f.source == "precomputed")
        }),
    }
}

/// Return `true` if the config entry uses `source: "computed"` FPHA in any
/// stage range or season entry.
fn config_uses_computed_fpha(config: &ProductionModelConfig) -> bool {
    match &config.selection_mode {
        SelectionMode::StageRanges { ranges } => ranges.iter().any(|r| {
            r.fpha_config
                .as_ref()
                .is_some_and(|f| f.source == "computed")
        }),
        SelectionMode::Seasonal { seasons, .. } => seasons.iter().any(|s| {
            s.fpha_config
                .as_ref()
                .is_some_and(|f| f.source == "computed")
        }),
    }
}

/// Extract the [`FphaColumnLayout`] that applies to a given stage from a [`ProductionModelConfig`].
///
/// Returns `None` when no stage range or season entry covers the stage, or when
/// the matched entry has no `fpha_config` field.
fn find_fpha_config_for_stage<'a>(
    config: &'a ProductionModelConfig,
    stage: &cobre_core::temporal::Stage,
) -> Option<&'a FphaColumnLayout> {
    match &config.selection_mode {
        SelectionMode::StageRanges { ranges } => {
            for range in ranges {
                let after_start = stage.id >= range.start_stage_id;
                let before_end = range.end_stage_id.is_none_or(|end| stage.id <= end);
                if after_start && before_end {
                    return range.fpha_config.as_ref();
                }
            }
            None
        }
        SelectionMode::Seasonal {
            default_model: _,
            seasons,
        } => {
            if let Some(season_id) = stage.season_id {
                for season in seasons {
                    if i32::try_from(season_id).is_ok_and(|sid| sid == season.season_id) {
                        return season.fpha_config.as_ref();
                    }
                }
            }
            None
        }
    }
}

/// Validate that a hydro with `source: "computed"` has all required model fields and geometry.
///
/// Checks that `tailrace`, `hydraulic_losses`, and `efficiency` are all `Some`, and
/// that at least one geometry row exists for this hydro in the geometry map.
///
/// # Policy rationale
///
/// Although the production function math can handle `None` for each of these
/// fields (zero tailrace, lossless penstock, 100% efficiency as defaults),
/// requiring all three as `Some` ensures the reservoir geometry was **fully
/// characterized** before committing to the computed FPHA path.  Accepting
/// partial geometry risks producing envelopes that are physically inconsistent
/// with the operator's intent and hard to diagnose after the fact.  Any hydro
/// that genuinely has no tailrace, lossless penstock, or a perfect turbine
/// must declare this explicitly by providing the respective model with an
/// appropriate constant or polynomial value.
///
/// # Errors
///
/// Returns `SddpError::Validation` listing the first missing prerequisite found,
/// including the hydro name and id.
fn validate_computed_prerequisites(
    hydro: &cobre_core::entities::hydro::Hydro,
    geometry_map: &HashMap<EntityId, Vec<&HydroGeometryRow>>,
) -> Result<(), SddpError> {
    let missing = if hydro.tailrace.is_none() {
        Some("tailrace")
    } else if hydro.hydraulic_losses.is_none() {
        Some("hydraulic_losses")
    } else if hydro.efficiency.is_none() {
        Some("efficiency")
    } else if geometry_map.get(&hydro.id).is_none_or(Vec::is_empty) {
        Some("geometry data")
    } else {
        None
    };

    if let Some(missing_item) = missing {
        return Err(SddpError::Validation(format!(
            "hydro {} (id={}) has source: \"computed\" but is missing {}. \
             Computed FPHA fitting requires tailrace, hydraulic_losses, \
             efficiency, and geometry data.",
            hydro.name, hydro.id.0, missing_item
        )));
    }

    Ok(())
}

/// Determine the [`ProductionModelSource`] for one hydro.
///
/// This checks only the high-level source classification without building the
/// per-stage model data; it is called once per hydro before the per-stage loop.
///
/// The function also rejects unsupported cases early to give clear errors before
/// any expensive Parquet loading occurs.
fn determine_source(
    hydro: &cobre_core::entities::hydro::Hydro,
    config_entry: Option<&ProductionModelConfig>,
) -> Result<ProductionModelSource, SddpError> {
    if let Some(config) = config_entry {
        // A "computed" source short-circuits to ComputedFromGeometry, so no
        // further range/season scan for "precomputed" is needed below.
        let computed_range = match &config.selection_mode {
            SelectionMode::StageRanges { ranges } => ranges
                .iter()
                .find(|r| {
                    r.fpha_config
                        .as_ref()
                        .is_some_and(|f| f.source == "computed")
                })
                .map(|r| r.model.clone()),
            SelectionMode::Seasonal { seasons, .. } => seasons
                .iter()
                .find(|s| {
                    s.fpha_config
                        .as_ref()
                        .is_some_and(|f| f.source == "computed")
                })
                .map(|s| s.model.clone()),
        };
        if computed_range.is_some() {
            return Ok(ProductionModelSource::ComputedFromGeometry);
        }
        // Only "precomputed" FPHA entries remain.
        let has_fpha = match &config.selection_mode {
            SelectionMode::StageRanges { ranges } => ranges.iter().any(|r| r.model == "fpha"),
            SelectionMode::Seasonal { seasons, .. } => seasons.iter().any(|s| s.model == "fpha"),
        };
        Ok(if has_fpha {
            ProductionModelSource::PrecomputedHyperplanes
        } else {
            ProductionModelSource::DefaultConstant
        })
    } else {
        // No config entry: use HydroGenerationModel from entity.
        match &hydro.generation_model {
            HydroGenerationModel::ConstantProductivity | HydroGenerationModel::LinearizedHead => {
                Ok(ProductionModelSource::DefaultConstant)
            }
            HydroGenerationModel::Fpha => Err(SddpError::Validation(format!(
                "hydro {} (id={}) has generation_model: \"fpha\" in hydros.json \
                 but no entry in hydro_production_models.json. \
                 Add an entry with source: \"precomputed\" to specify the hyperplane source.",
                hydro.name, hydro.id.0
            ))),
        }
    }
}

/// Resolve the production model for one (hydro, stage) pair.
///
/// `cached_computed_planes` carries planes already fitted by the outer loop
/// when `source == ComputedFromGeometry`. Passing pre-fitted planes avoids
/// running the fitting pipeline once per stage; the outer loop fits once per
/// hydro and clones for every stage via this parameter.
fn resolve_stage_model(
    hydro: &cobre_core::entities::hydro::Hydro,
    stage: &cobre_core::temporal::Stage,
    config_entry: Option<&ProductionModelConfig>,
    source: ProductionModelSource,
    hyperplane_map: &HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>>,
    cached_computed_planes: Option<&[FphaPlane]>,
    productivity_override: Option<&crate::energy_conversion::HydroEnergyProductivityOverride>,
) -> Result<ResolvedProductionModel, SddpError> {
    // Look up the parquet override for non-FPHA (hydro, stage) productivity.
    // Cross-file resolution (cobre_io::validation::productivity_resolution)
    // rejects the case where both JSON and parquet supply a value, so this
    // lookup never silently masks a JSON-supplied value at load time.
    let stage_idx = usize::try_from(stage.id.max(0)).unwrap_or(0);
    let parquet_productivity =
        productivity_override.and_then(|o| o.equivalent_productivity(hydro.id, stage_idx));

    if let Some(config) = config_entry {
        let model_info = find_model_for_stage(config, stage);

        if model_info.as_ref().map(|(name, _)| name.as_str()) == Some("fpha") {
            if source == ProductionModelSource::ComputedFromGeometry {
                // Use the pre-fitted planes from the outer loop cache.
                let planes = cached_computed_planes
                    .ok_or_else(|| {
                        SddpError::Validation(format!(
                            "hydro {} (id={}) is ComputedFromGeometry but no cached planes \
                             were provided to resolve_stage_model",
                            hydro.name, hydro.id.0
                        ))
                    })?
                    .to_vec();
                Ok(ResolvedProductionModel::Fpha { planes })
            } else {
                build_fpha_model(hydro, stage, source, hyperplane_map)
            }
        } else {
            // "constant_productivity" or "linearized_head" from config.
            //
            // Resolution order (matches build_energy_conversion_set): parquet
            // override first, JSON productivity fallback. Load-time validation
            // in cobre_io::validation::productivity_resolution guarantees that
            // exactly one source supplies the value, so a None outcome here
            // would indicate a validator gap.
            let productivity = parquet_productivity
                .or_else(|| model_info.and_then(|(_, p)| p))
                .unwrap_or_else(|| {
                    debug_assert!(
                        false,
                        "non-FPHA {}/{} reached resolve_stage_model with productivity=None; \
                         see cobre_io::validation::productivity_resolution",
                        hydro.name, stage.id
                    );
                    0.0
                });
            Ok(ResolvedProductionModel::ConstantProductivity { productivity })
        }
    } else {
        // No JSON config entry at all for this hydro. Use the parquet override
        // when present; otherwise fall through to the sentinel (validator
        // already rejected this case at load time).
        let productivity = parquet_productivity.unwrap_or_else(|| {
            debug_assert!(
                false,
                "non-FPHA {}/{} reached resolve_stage_model with productivity=None; \
                 see cobre_io::validation::productivity_resolution",
                hydro.name, stage.id
            );
            0.0
        });
        Ok(ResolvedProductionModel::ConstantProductivity { productivity })
    }
}

/// Find the model name and optional productivity override for a given stage.
///
/// Returns `None` when the config has no entry covering the given stage (gap in coverage).
/// For `StageRanges`, the match is `start_stage_id <= stage.id <= end_stage_id`.
/// For `Seasonal`, the match is by `season_id == stage.season_id`.
fn find_model_for_stage(
    config: &ProductionModelConfig,
    stage: &cobre_core::temporal::Stage,
) -> Option<(String, Option<f64>)> {
    match &config.selection_mode {
        SelectionMode::StageRanges { ranges } => {
            for range in ranges {
                let after_start = stage.id >= range.start_stage_id;
                let before_end = range.end_stage_id.is_none_or(|end| stage.id <= end);
                if after_start && before_end {
                    return Some((range.model.clone(), range.productivity_mw_per_m3s));
                }
            }
            None
        }
        SelectionMode::Seasonal {
            default_model,
            seasons,
        } => {
            if let Some(season_id) = stage.season_id {
                for season in seasons {
                    // season.season_id is i32; stage.season_id is usize.
                    // Convert usize to i32 for comparison to avoid cast_sign_loss.
                    if i32::try_from(season_id).is_ok_and(|sid| sid == season.season_id) {
                        return Some((season.model.clone(), season.productivity_mw_per_m3s));
                    }
                }
            }
            // Fall back to default model when no season matches (or no season_id on stage).
            // Default model has no override.
            Some((default_model.clone(), None))
        }
    }
}

/// Build an `Fpha` variant `ResolvedProductionModel` for one (hydro, stage) pair.
///
/// Looks up hyperplanes for `(hydro_id, Some(stage.id))` first; falls back to
/// `(hydro_id, None)` when no stage-specific rows exist (global all-stage rows).
/// Validates each hyperplane's coefficients and `kappa`, then constructs
/// [`FphaPlane`] with the pre-scaled intercept `gamma_0 * kappa`.
fn build_fpha_model(
    hydro: &cobre_core::entities::hydro::Hydro,
    stage: &cobre_core::temporal::Stage,
    _source: ProductionModelSource,
    hyperplane_map: &HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>>,
) -> Result<ResolvedProductionModel, SddpError> {
    // Prefer stage-specific rows; fall back to global (stage_id: None) rows.
    let rows: &[&FphaHyperplaneRow] = hyperplane_map
        .get(&(hydro.id, Some(stage.id)))
        .or_else(|| hyperplane_map.get(&(hydro.id, None)))
        .ok_or_else(|| {
            SddpError::Validation(format!(
                "hydro {} (id={}) is configured as FPHA but has no hyperplane rows \
             in fpha_hyperplanes.parquet for stage {} (and no global all-stage rows).",
                hydro.name, hydro.id.0, stage.id
            ))
        })?;

    if rows.is_empty() {
        return Err(SddpError::Validation(format!(
            "hydro {} (id={}) has zero hyperplane rows for stage {}.",
            hydro.name, hydro.id.0, stage.id
        )));
    }

    let mut planes: Vec<FphaPlane> = Vec::with_capacity(rows.len());
    for row in rows {
        validate_hyperplane_row(hydro, stage, row)?;
        planes.push(FphaPlane {
            intercept: row.gamma_0 * row.kappa,
            gamma_v: row.gamma_v,
            gamma_q: row.gamma_q,
            gamma_s: row.gamma_s,
        });
    }

    Ok(ResolvedProductionModel::Fpha { planes })
}

/// Validate the physical constraints for one `FphaHyperplaneRow`.
///
/// Returns `Err(SddpError::Validation(...))` when any constraint is violated.
///
/// Constraints:
///
/// - `gamma_v >= 0` — higher storage must not decrease generation; zero is valid
///   for constant-head plants where head does not depend on volume
/// - `gamma_s <= 0` — spillage reduces generation
/// - `gamma_q > 0` — more turbined flow → more generation
/// - `kappa ∈ (0, 1]` — correction factor range
fn validate_hyperplane_row(
    hydro: &cobre_core::entities::hydro::Hydro,
    stage: &cobre_core::temporal::Stage,
    row: &FphaHyperplaneRow,
) -> Result<(), SddpError> {
    let ctx = format!(
        "hydro {} (id={}) plane {} stage {}",
        hydro.name, hydro.id.0, row.plane_id, stage.id
    );

    if row.gamma_v < 0.0 {
        return Err(SddpError::Validation(format!(
            "{ctx}: gamma_v must be >= 0 (higher storage must not decrease generation; \
             zero is valid for constant-head plants), got gamma_v = {}",
            row.gamma_v
        )));
    }

    if row.gamma_s > 0.0 {
        return Err(SddpError::Validation(format!(
            "{ctx}: gamma_s must be <= 0 (spillage reduces generation), \
             got gamma_s = {}",
            row.gamma_s
        )));
    }

    if row.gamma_q <= 0.0 {
        return Err(SddpError::Validation(format!(
            "{ctx}: gamma_q must be > 0 (more turbined flow → more generation), \
             got gamma_q = {}",
            row.gamma_q
        )));
    }

    if row.kappa <= 0.0 || row.kappa > 1.0 {
        return Err(SddpError::Validation(format!(
            "{ctx}: kappa must be in (0, 1] (correction factor range), \
             got kappa = {}",
            row.kappa
        )));
    }

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::match_wildcard_for_single_variants,
    clippy::cast_precision_loss,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic
)]
mod tests {
    use std::collections::HashMap;

    use chrono::NaiveDate;
    use cobre_core::{
        EfficiencyModel, EntityId, HydraulicLossesModel, TailraceModel,
        entities::hydro::{HydroGenerationModel, HydroPenalties},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };
    use cobre_io::extensions::{
        FphaColumnLayout, FphaHyperplaneRow, HydroGeometryRow, ProductionModelConfig, SeasonConfig,
        SelectionMode, StageRange,
    };

    use super::*;

    // ── Test helpers ──────────────────────────────────────────────────────────

    fn make_stage(id: i32) -> Stage {
        Stage {
            index: usize::try_from(id.max(0)).unwrap_or(0),
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap_or_default(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap_or_default(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 50,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn zero_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
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

    fn make_hydro(id: i32, model: HydroGenerationModel) -> cobre_core::entities::hydro::Hydro {
        cobre_core::entities::hydro::Hydro {
            id: EntityId::from(id),
            name: format!("Hydro{id}"),
            bus_id: EntityId::from(10),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 100.0,
            max_storage_hm3: 2000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: model,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 500.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 1000.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_penalties(),
        }
    }

    fn valid_row(hydro_id: i32, stage_id: Option<i32>, plane_id: i32) -> FphaHyperplaneRow {
        FphaHyperplaneRow {
            hydro_id: EntityId::from(hydro_id),
            stage_id,
            plane_id,
            gamma_0: 1000.0,
            gamma_v: 0.002,
            gamma_q: 0.85,
            gamma_s: -0.01,
            kappa: 1.0,
            valid_v_min_hm3: None,
            valid_v_max_hm3: None,
            valid_q_max_m3s: None,
        }
    }

    fn precomputed_fpha_config(hydro_id: i32) -> ProductionModelConfig {
        ProductionModelConfig {
            hydro_id: EntityId::from(hydro_id),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 0,
                    end_stage_id: None,
                    model: "fpha".to_string(),
                    fpha_config: Some(FphaColumnLayout {
                        source: "precomputed".to_string(),
                        volume_discretization_points: None,
                        turbine_discretization_points: None,
                        spillage_discretization_points: None,
                        max_planes_per_hydro: None,
                        fitting_window: None,
                    }),
                    productivity_mw_per_m3s: None,
                }],
            },
        }
    }

    fn computed_fpha_config(hydro_id: i32) -> ProductionModelConfig {
        ProductionModelConfig {
            hydro_id: EntityId::from(hydro_id),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 0,
                    end_stage_id: None,
                    model: "fpha".to_string(),
                    fpha_config: Some(FphaColumnLayout {
                        source: "computed".to_string(),
                        volume_discretization_points: None,
                        turbine_discretization_points: None,
                        spillage_discretization_points: None,
                        max_planes_per_hydro: None,
                        fitting_window: None,
                    }),
                    productivity_mw_per_m3s: None,
                }],
            },
        }
    }

    // ── resolve_production_models unit tests (in-memory, no disk I/O) ─────────

    /// Non-FPHA hydros without any config entry produce
    /// `DefaultConstant` provenance from `determine_source`. The
    /// downstream sentinel behaviour in `resolve_stage_model` is exercised
    /// by `test_resolve_stage_model_returns_sentinel_when_no_config_entry`.
    #[test]
    fn all_constant_no_config_returns_default_constant_provenance() {
        let hydro0 = make_hydro(0, HydroGenerationModel::ConstantProductivity);
        let hydro1 = make_hydro(1, HydroGenerationModel::ConstantProductivity);

        let src0 = determine_source(&hydro0, None).expect("should succeed");
        let src1 = determine_source(&hydro1, None).expect("should succeed");
        assert_eq!(src0, ProductionModelSource::DefaultConstant);
        assert_eq!(src1, ProductionModelSource::DefaultConstant);
    }

    /// `LinearizedHead` entities without a config entry produce
    /// `DefaultConstant` provenance from `determine_source`. The downstream
    /// sentinel behaviour in `resolve_stage_model` is exercised by
    /// `test_resolve_stage_model_returns_sentinel_when_no_config_entry`.
    #[test]
    fn linearized_head_entity_resolves_to_constant_productivity() {
        let hydro = make_hydro(0, HydroGenerationModel::LinearizedHead);

        let src = determine_source(&hydro, None).expect("should succeed");
        assert_eq!(src, ProductionModelSource::DefaultConstant);
    }

    /// Fpha entity model without config → validation error.
    #[test]
    fn fpha_entity_without_config_entry_returns_validation_error() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let err = determine_source(&hydro, None).expect_err("should fail");
        assert!(
            matches!(err, crate::SddpError::Validation(ref msg) if
                msg.contains("fpha") || msg.contains("no entry") || msg.contains("hydro_production_models")),
            "expected Validation error mentioning missing config entry, got {err:?}"
        );
    }

    /// source: "computed" in config → returns `ComputedFromGeometry` (fitting is now supported).
    #[test]
    fn computed_source_returns_computed_from_geometry() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let config = computed_fpha_config(0);

        let source = determine_source(&hydro, Some(&config)).expect("should succeed");
        assert_eq!(
            source,
            ProductionModelSource::ComputedFromGeometry,
            "expected ComputedFromGeometry, got {source:?}"
        );
    }

    /// Helper: build a minimal hydro with all computed-source prerequisites set.
    fn make_computed_hydro(id: i32) -> cobre_core::entities::hydro::Hydro {
        let mut hydro = make_hydro(id, HydroGenerationModel::Fpha);
        hydro.tailrace = Some(TailraceModel::Polynomial {
            coefficients: vec![300.0],
        });
        hydro.hydraulic_losses = Some(HydraulicLossesModel::Factor { value: 0.02 });
        hydro.efficiency = Some(EfficiencyModel::Constant { value: 0.92 });
        hydro
    }

    /// Helper: build a two-point VHA geometry row vector for a hydro.
    fn make_geometry_rows(hydro_id: i32) -> Vec<HydroGeometryRow> {
        vec![
            HydroGeometryRow {
                hydro_id: EntityId::from(hydro_id),
                volume_hm3: 100.0,
                height_m: 400.0,
                area_km2: 10.0,
            },
            HydroGeometryRow {
                hydro_id: EntityId::from(hydro_id),
                volume_hm3: 2000.0,
                height_m: 450.0,
                area_km2: 50.0,
            },
        ]
    }

    /// validate_computed_prerequisites: missing tailrace → Validation error with "tailrace" and hydro name.
    #[test]
    fn computed_source_missing_tailrace_returns_validation_error() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        // tailrace is None in make_hydro
        let rows = make_geometry_rows(0);
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        let row_refs: Vec<&HydroGeometryRow> = rows.iter().collect();
        geometry_map.insert(EntityId::from(0), row_refs);

        let err = validate_computed_prerequisites(&hydro, &geometry_map)
            .expect_err("should fail when tailrace is None");
        let msg = err.to_string();
        assert!(
            msg.contains("tailrace"),
            "error must mention 'tailrace', got: {msg}"
        );
        assert!(
            msg.contains(&hydro.name),
            "error must include hydro name '{}', got: {msg}",
            hydro.name
        );
    }

    /// validate_computed_prerequisites: missing geometry rows → Validation error with "geometry" and hydro name.
    #[test]
    fn computed_source_missing_geometry_returns_validation_error() {
        let hydro = make_computed_hydro(0);
        let empty_geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();

        let err = validate_computed_prerequisites(&hydro, &empty_geometry_map)
            .expect_err("should fail when geometry rows are absent");
        let msg = err.to_string();
        assert!(
            msg.contains("geometry"),
            "error must mention 'geometry', got: {msg}"
        );
        assert!(
            msg.contains(&hydro.name),
            "error must include hydro name '{}', got: {msg}",
            hydro.name
        );
    }

    /// find_fpha_config_for_stage: returns Some(&FphaColumnLayout) when stage is in the range.
    #[test]
    fn find_fpha_config_for_stage_returns_config_in_range() {
        let config = computed_fpha_config(0);
        let stage = make_stage(5);

        let result = find_fpha_config_for_stage(&config, &stage);
        assert!(
            result.is_some(),
            "expected Some(FphaColumnLayout) for stage 5, got None"
        );
        assert_eq!(
            result.expect("just checked is_some").source,
            "computed",
            "expected source 'computed'"
        );
    }

    /// find_fpha_config_for_stage: returns None when no range covers the stage.
    #[test]
    fn find_fpha_config_for_stage_returns_none_outside_range() {
        // Create a config with range [5, 10].
        let config = ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 5,
                    end_stage_id: Some(10),
                    model: "fpha".to_string(),
                    fpha_config: Some(FphaColumnLayout {
                        source: "computed".to_string(),
                        volume_discretization_points: None,
                        turbine_discretization_points: None,
                        spillage_discretization_points: None,
                        max_planes_per_hydro: None,
                        fitting_window: None,
                    }),
                    productivity_mw_per_m3s: None,
                }],
            },
        };

        // Stage 0 is before the range [5, 10].
        let stage = make_stage(0);
        let result = find_fpha_config_for_stage(&config, &stage);
        assert!(
            result.is_none(),
            "expected None for stage 0 (outside range [5,10]), got {result:?}"
        );
    }

    /// kappa = 0.95 → intercept is gamma_0 * kappa.
    #[test]
    fn gamma_0_is_scaled_by_kappa() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(0);

        let row = FphaHyperplaneRow {
            hydro_id: EntityId::from(0),
            stage_id: None,
            plane_id: 0,
            gamma_0: 1000.0,
            gamma_v: 0.002,
            gamma_q: 0.85,
            gamma_s: -0.01,
            kappa: 0.95,
            valid_v_min_hm3: None,
            valid_v_max_hm3: None,
            valid_q_max_m3s: None,
        };

        let mut map = std::collections::HashMap::new();
        map.insert(
            (EntityId::from(0), None::<i32>),
            vec![&row as &FphaHyperplaneRow],
        );

        let model = build_fpha_model(
            &hydro,
            &stage,
            ProductionModelSource::PrecomputedHyperplanes,
            &map,
        )
        .expect("should build FPHA model");

        match model {
            ResolvedProductionModel::Fpha { planes, .. } => {
                assert_eq!(planes.len(), 1);
                let expected = 1000.0 * 0.95;
                assert!(
                    (planes[0].intercept - expected).abs() < 1e-10,
                    "intercept must be gamma_0 * kappa = {expected}, got {}",
                    planes[0].intercept
                );
            }
            other => panic!("expected Fpha variant, got {other:?}"),
        }
    }

    /// validate_hyperplane_row rejects negative gamma_v.
    #[test]
    fn validation_rejects_gamma_v_negative() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(0);

        let mut row = valid_row(0, None, 0);
        row.gamma_v = -0.1; // invalid: must be >= 0

        let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("gamma_v"),
            "error must mention gamma_v, got: {msg}"
        );
    }

    /// validate_hyperplane_row accepts gamma_v == 0.0 (constant-head plant).
    #[test]
    fn validation_accepts_gamma_v_zero() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(0);

        let mut row = valid_row(0, None, 0);
        row.gamma_v = 0.0; // valid: constant-head plant

        validate_hyperplane_row(&hydro, &stage, &row)
            .expect("gamma_v = 0.0 must be valid for constant-head plants");
    }

    /// validate_hyperplane_row rejects gamma_s > 0.
    #[test]
    fn validation_rejects_gamma_s_positive() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(0);

        let mut row = valid_row(0, None, 0);
        row.gamma_s = 0.01;

        let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("gamma_s"),
            "error must mention gamma_s, got: {msg}"
        );
    }

    /// validate_hyperplane_row rejects gamma_q <= 0.
    #[test]
    fn validation_rejects_gamma_q_nonpositive() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(0);

        let mut row = valid_row(0, None, 0);
        row.gamma_q = 0.0;

        let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("gamma_q"),
            "error must mention gamma_q, got: {msg}"
        );
    }

    /// validate_hyperplane_row rejects kappa = 0 (must be > 0).
    #[test]
    fn validation_rejects_kappa_zero() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(0);

        let mut row = valid_row(0, None, 0);
        row.kappa = 0.0;

        let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("kappa"),
            "error must mention kappa, got: {msg}"
        );
    }

    /// validate_hyperplane_row rejects kappa = 1.5 (must be <= 1).
    #[test]
    fn validation_rejects_kappa_above_one() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(0);

        let mut row = valid_row(0, None, 0);
        row.kappa = 1.5;

        let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("kappa"),
            "error must mention kappa, got: {msg}"
        );
    }

    /// Stage-specific hyperplanes (Some(stage_id)) override all-stage (None) rows.
    #[test]
    fn stage_specific_hyperplanes_override_all_stage() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(0);

        let global_row = FphaHyperplaneRow {
            hydro_id: EntityId::from(0),
            stage_id: None,
            plane_id: 0,
            gamma_0: 500.0, // distinct intercept to identify
            gamma_v: 0.001,
            gamma_q: 0.80,
            gamma_s: -0.005,
            kappa: 1.0,
            valid_v_min_hm3: None,
            valid_v_max_hm3: None,
            valid_q_max_m3s: None,
        };
        let stage_row = FphaHyperplaneRow {
            hydro_id: EntityId::from(0),
            stage_id: Some(0),
            plane_id: 0,
            gamma_0: 900.0, // distinct intercept to identify
            gamma_v: 0.002,
            gamma_q: 0.85,
            gamma_s: -0.01,
            kappa: 1.0,
            valid_v_min_hm3: None,
            valid_v_max_hm3: None,
            valid_q_max_m3s: None,
        };

        let mut map = std::collections::HashMap::new();
        map.insert(
            (EntityId::from(0), None::<i32>),
            vec![&global_row as &FphaHyperplaneRow],
        );
        map.insert(
            (EntityId::from(0), Some(0i32)),
            vec![&stage_row as &FphaHyperplaneRow],
        );

        let model = build_fpha_model(
            &hydro,
            &stage,
            ProductionModelSource::PrecomputedHyperplanes,
            &map,
        )
        .expect("should succeed");

        match model {
            ResolvedProductionModel::Fpha { planes, .. } => {
                // Stage-specific row has gamma_0 = 900, global has 500; stage-specific wins.
                assert!(
                    (planes[0].intercept - 900.0).abs() < 1e-10,
                    "stage-specific intercept 900 should override global 500, got {}",
                    planes[0].intercept
                );
            }
            other => panic!("expected Fpha variant, got {other:?}"),
        }
    }

    /// All-stage hyperplanes (stage_id: None) are used when no stage-specific rows exist.
    #[test]
    fn all_stage_hyperplanes_used_when_no_stage_specific_rows() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(5); // stage id 5, no stage-specific rows for it

        let global_row = FphaHyperplaneRow {
            hydro_id: EntityId::from(0),
            stage_id: None,
            plane_id: 0,
            gamma_0: 700.0,
            gamma_v: 0.002,
            gamma_q: 0.85,
            gamma_s: -0.01,
            kappa: 1.0,
            valid_v_min_hm3: None,
            valid_v_max_hm3: None,
            valid_q_max_m3s: None,
        };

        let mut map = std::collections::HashMap::new();
        map.insert(
            (EntityId::from(0), None::<i32>),
            vec![&global_row as &FphaHyperplaneRow],
        );

        let model = build_fpha_model(
            &hydro,
            &stage,
            ProductionModelSource::PrecomputedHyperplanes,
            &map,
        )
        .expect("should succeed using global rows");

        match model {
            ResolvedProductionModel::Fpha { planes, .. } => {
                assert!(
                    (planes[0].intercept - 700.0).abs() < 1e-10,
                    "expected global intercept 700, got {}",
                    planes[0].intercept
                );
            }
            other => panic!("expected Fpha, got {other:?}"),
        }
    }

    /// Zero hyperplanes for a stage (empty rows) → validation error.
    #[test]
    fn zero_hyperplanes_for_stage_returns_validation_error() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(0);

        // Map has the key but an empty rows vec.
        let mut map = std::collections::HashMap::new();
        map.insert(
            (EntityId::from(0), None::<i32>),
            Vec::<&FphaHyperplaneRow>::new(),
        );

        let err = build_fpha_model(
            &hydro,
            &stage,
            ProductionModelSource::PrecomputedHyperplanes,
            &map,
        )
        .expect_err("should fail with zero hyperplanes");

        assert!(
            matches!(err, crate::SddpError::Validation(_)),
            "expected Validation error, got {err:?}"
        );
    }

    /// find_model_for_stage: stage_id in range returns fpha model name.
    #[test]
    fn find_model_for_stage_returns_correct_model_name_in_range() {
        let config = precomputed_fpha_config(0);
        let stage = make_stage(3);
        let result = find_model_for_stage(&config, &stage);
        assert_eq!(result.as_ref().map(|(name, _)| name.as_str()), Some("fpha"));
    }

    /// find_model_for_stage: stage_id before start of range returns None.
    #[test]
    fn find_model_for_stage_returns_none_when_before_range_start() {
        let config = ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 5,
                    end_stage_id: Some(10),
                    model: "fpha".to_string(),
                    fpha_config: None,
                    productivity_mw_per_m3s: None,
                }],
            },
        };
        let stage = make_stage(3); // id 3, before start_stage_id 5
        let result = find_model_for_stage(&config, &stage);
        assert!(
            result.is_none(),
            "stage 3 is before range [5, 10], expected None"
        );
    }

    /// find_model_for_stage: end_stage_id = None covers all stages from start.
    #[test]
    fn find_model_for_stage_open_ended_range_covers_all_stages() {
        let config = ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 0,
                    end_stage_id: None,
                    model: "constant_productivity".to_string(),
                    fpha_config: None,
                    productivity_mw_per_m3s: None,
                }],
            },
        };
        for stage_id in [0, 5, 11, 100] {
            let stage = make_stage(stage_id);
            let result = find_model_for_stage(&config, &stage);
            assert_eq!(
                result.as_ref().map(|(name, _)| name.as_str()),
                Some("constant_productivity"),
                "open-ended range must cover stage {stage_id}"
            );
        }
    }

    // ── Productivity override tests ─────────────────────────────────────────

    /// resolve_stage_model uses productivity_override when present.
    #[test]
    fn resolve_stage_model_uses_productivity_override() {
        let hydro = make_hydro(0, HydroGenerationModel::ConstantProductivity);
        let stage = make_stage(0);
        let config = ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 0,
                    end_stage_id: None,
                    model: "constant_productivity".to_string(),
                    fpha_config: None,
                    productivity_mw_per_m3s: Some(0.55),
                }],
            },
        };
        let empty_map = std::collections::HashMap::new();
        let model = super::resolve_stage_model(
            &hydro,
            &stage,
            Some(&config),
            ProductionModelSource::DefaultConstant,
            &empty_map,
            None,
            None,
        )
        .expect("should succeed");
        assert!(
            matches!(model, ResolvedProductionModel::ConstantProductivity { productivity }
                if (productivity - 0.55).abs() < f64::EPSILON),
            "expected ConstantProductivity 0.55 (override), got {model:?}"
        );
    }

    /// When the JSON config entry exists but its `productivity_mw_per_m3s`
    /// field is `None`, `resolve_stage_model` returns a sentinel
    /// `ConstantProductivity { productivity: 0.0 }`. The build site in
    /// `build_energy_conversion_set` consults the parquet override and
    /// overwrites the sentinel with the user-supplied value.
    ///
    /// In debug builds the sentinel path is gated by a `debug_assert!` that
    /// catches mis-configured cases that escape load-time validation in
    /// `cobre_io::validation::productivity_resolution`. In release builds the
    /// `debug_assert!` is compiled out and the function returns the sentinel
    /// directly.
    #[test]
    fn test_resolve_stage_model_returns_sentinel_when_json_lacks_productivity() {
        let hydro = make_hydro(0, HydroGenerationModel::ConstantProductivity);
        let stage = make_stage(0);
        // Config entry exists but productivity is missing — the parquet
        // override is the supplier in production.
        let config = ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 0,
                    end_stage_id: None,
                    model: "constant_productivity".to_string(),
                    fpha_config: None,
                    productivity_mw_per_m3s: None,
                }],
            },
        };
        let empty_map = std::collections::HashMap::new();

        #[cfg(debug_assertions)]
        {
            // Debug build: the `debug_assert!` fires and the call panics.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                super::resolve_stage_model(
                    &hydro,
                    &stage,
                    Some(&config),
                    ProductionModelSource::DefaultConstant,
                    &empty_map,
                    None,
                    None,
                )
            }));
            let panic_payload = result
                .expect_err("debug build must panic via debug_assert! when productivity is None");
            let msg = panic_payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| {
                    panic_payload
                        .downcast_ref::<&str>()
                        .map(|s| (*s).to_owned())
                })
                .unwrap_or_default();
            assert!(
                msg.contains("Hydro0") && msg.contains("validation::productivity_resolution"),
                "panic message must name the hydro and the validator; got: {msg}"
            );
        }

        #[cfg(not(debug_assertions))]
        {
            // Release build: the assert is compiled out and the function
            // returns the sentinel directly.
            let model = super::resolve_stage_model(
                &hydro,
                &stage,
                Some(&config),
                ProductionModelSource::DefaultConstant,
                &empty_map,
                None,
                None,
            )
            .expect("release build returns sentinel");
            assert!(
                matches!(
                    model,
                    ResolvedProductionModel::ConstantProductivity { productivity }
                    if productivity == 0.0
                ),
                "release build must return ConstantProductivity {{ productivity: 0.0 }}; got {model:?}"
            );
        }
    }

    /// When the JSON has no productivity and the parquet override supplies a
    /// value, `resolve_stage_model` returns that value as the resolved
    /// `ConstantProductivity { productivity }`. This is the path the LP
    /// coefficient flows through for non-FPHA hydros authored entirely via the
    /// parquet.
    #[test]
    fn test_resolve_stage_model_uses_parquet_override_when_json_omits_productivity() {
        let hydro = make_hydro(0, HydroGenerationModel::ConstantProductivity);
        let stage = make_stage(0);
        let config = ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 0,
                    end_stage_id: None,
                    model: "constant_productivity".to_string(),
                    fpha_config: None,
                    productivity_mw_per_m3s: None,
                }],
            },
        };
        let empty_map = std::collections::HashMap::new();
        let override_table = crate::energy_conversion::build_hydro_energy_productivity_override(&[
            cobre_io::HydroEnergyProductivityRow {
                hydro_id: EntityId::from(0),
                stage_id: Some(0),
                equivalent_productivity_mw_per_m3s: Some(0.42),
                reference_volume_hm3: None,
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            },
        ])
        .expect("override builds");

        let model = super::resolve_stage_model(
            &hydro,
            &stage,
            Some(&config),
            ProductionModelSource::DefaultConstant,
            &empty_map,
            None,
            Some(&override_table),
        )
        .expect("resolve succeeds");
        assert!(
            matches!(
                model,
                ResolvedProductionModel::ConstantProductivity { productivity }
                if (productivity - 0.42).abs() < 1e-12
            ),
            "override value must reach the resolved model; got {model:?}"
        );
    }

    /// When no JSON config entry exists at all for a non-FPHA hydro,
    /// `resolve_stage_model` returns the same sentinel. Load-time validation
    /// in `cobre_io::validation::productivity_resolution` is responsible for
    /// catching missing entries; this branch trusts that invariant in release
    /// builds and is guarded by a `debug_assert!` in debug builds.
    #[test]
    fn test_resolve_stage_model_returns_sentinel_when_no_config_entry() {
        let hydro = make_hydro(7, HydroGenerationModel::ConstantProductivity);
        let stage = make_stage(3);
        let empty_map = std::collections::HashMap::new();

        #[cfg(debug_assertions)]
        {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                super::resolve_stage_model(
                    &hydro,
                    &stage,
                    None,
                    ProductionModelSource::DefaultConstant,
                    &empty_map,
                    None,
                    None,
                )
            }));
            let panic_payload = result
                .expect_err("debug build must panic via debug_assert! when no config entry exists");
            let msg = panic_payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| {
                    panic_payload
                        .downcast_ref::<&str>()
                        .map(|s| (*s).to_owned())
                })
                .unwrap_or_default();
            assert!(
                msg.contains("Hydro7") && msg.contains("validation::productivity_resolution"),
                "panic message must name the hydro and the validator; got: {msg}"
            );
        }

        #[cfg(not(debug_assertions))]
        {
            let model = super::resolve_stage_model(
                &hydro,
                &stage,
                None,
                ProductionModelSource::DefaultConstant,
                &empty_map,
                None,
                None,
            )
            .expect("release build returns sentinel");
            assert!(
                matches!(
                    model,
                    ResolvedProductionModel::ConstantProductivity { productivity }
                    if productivity == 0.0
                ),
                "release build must return ConstantProductivity {{ productivity: 0.0 }}; got {model:?}"
            );
        }
    }

    /// find_model_for_stage returns override in tuple.
    #[test]
    fn find_model_for_stage_returns_override_in_tuple() {
        let config = ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 0,
                    end_stage_id: None,
                    model: "constant_productivity".to_string(),
                    fpha_config: None,
                    productivity_mw_per_m3s: Some(0.75),
                }],
            },
        };
        let stage = make_stage(0);
        let result = find_model_for_stage(&config, &stage);
        assert_eq!(
            result,
            Some(("constant_productivity".to_string(), Some(0.75)))
        );
    }

    /// Seasonal mode: find_model_for_stage returns override for matching season
    /// and None for default model.
    #[test]
    fn find_model_for_stage_seasonal_with_override() {
        let config = ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::Seasonal {
                default_model: "constant_productivity".to_string(),
                seasons: vec![SeasonConfig {
                    season_id: 1,
                    model: "constant_productivity".to_string(),
                    fpha_config: None,
                    productivity_mw_per_m3s: Some(0.60),
                }],
            },
        };
        // Stage with matching season_id = 1
        let mut stage_match = make_stage(0);
        stage_match.season_id = Some(1);
        let result = find_model_for_stage(&config, &stage_match);
        assert_eq!(
            result,
            Some(("constant_productivity".to_string(), Some(0.60)))
        );

        // Stage with non-matching season_id → default model, no override
        let mut stage_default = make_stage(0);
        stage_default.season_id = Some(99);
        let result = find_model_for_stage(&config, &stage_default);
        assert_eq!(result, Some(("constant_productivity".to_string(), None)));
    }

    /// precomputed config returns PrecomputedHyperplanes source.
    #[test]
    fn precomputed_config_returns_precomputed_source() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let config = precomputed_fpha_config(0);
        let src = determine_source(&hydro, Some(&config)).expect("should succeed");
        assert_eq!(src, ProductionModelSource::PrecomputedHyperplanes);
    }

    // ── Computed-source integration tests ─────────────────────────────────────

    /// Sobradinho-style hydro with all computed prerequisites, matching the known-valid
    /// fixture from `fpha_fitting.rs`. Used for end-to-end computed-source tests.
    fn make_sobradinho_computed_hydro(id: i32) -> cobre_core::entities::hydro::Hydro {
        let mut hydro = make_hydro(id, HydroGenerationModel::Fpha);
        hydro.name = format!("Sobradinho{id}");
        hydro.min_storage_hm3 = 100.0;
        hydro.max_storage_hm3 = 20_000.0;
        hydro.max_turbined_m3s = 500.0;
        hydro.tailrace = Some(TailraceModel::Polynomial {
            coefficients: vec![0.0, 0.001_f64],
        });
        hydro.hydraulic_losses = Some(HydraulicLossesModel::Constant { value_m: 2.0 });
        hydro.efficiency = Some(EfficiencyModel::Constant { value: 0.92 });
        hydro
    }

    /// Four-point VHA geometry rows spanning volumes 100.0 to 20_000.0 hm³ and heights
    /// 386.5 to 400.0 m. Mirrors the Sobradinho-style fixture used in `fpha_fitting.rs`.
    fn make_sobradinho_geometry_rows(hydro_id: i32) -> Vec<HydroGeometryRow> {
        vec![
            HydroGeometryRow {
                hydro_id: EntityId::from(hydro_id),
                volume_hm3: 100.0,
                height_m: 386.5,
                area_km2: 500.0,
            },
            HydroGeometryRow {
                hydro_id: EntityId::from(hydro_id),
                volume_hm3: 5_000.0,
                height_m: 392.0,
                area_km2: 800.0,
            },
            HydroGeometryRow {
                hydro_id: EntityId::from(hydro_id),
                volume_hm3: 12_000.0,
                height_m: 396.5,
                area_km2: 1_100.0,
            },
            HydroGeometryRow {
                hydro_id: EntityId::from(hydro_id),
                volume_hm3: 20_000.0,
                height_m: 400.0,
                area_km2: 1_400.0,
            },
        ]
    }

    /// Computed-source end-to-end: a hydro with all prerequisites and Sobradinho-style geometry
    /// produces a valid `Fpha` model with 3–10 planes and correct coefficient signs.
    ///
    /// Tests `fit_planes_for_hydro` + `resolve_stage_model` together.
    #[test]
    fn computed_source_end_to_end_produces_valid_fpha_planes() {
        let hydro = make_sobradinho_computed_hydro(0);
        let config = computed_fpha_config(0);
        let stage = make_stage(0);

        let geo_rows = make_sobradinho_geometry_rows(0);
        let geo_refs: Vec<&HydroGeometryRow> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        let study_stages = [stage.clone()];
        let stage_refs: Vec<&cobre_core::temporal::Stage> = study_stages.iter().collect();

        // Fit planes once (simulating the outer loop in resolve_production_models).
        let fit_result =
            super::fit_planes_for_hydro(&hydro, Some(&config), &geometry_map, &stage_refs)
                .expect("fit_planes_for_hydro must succeed for valid Sobradinho-style input");
        let planes = &fit_result.planes;

        // Plane count must be within the expected range for default FphaColumnLayout.
        assert!(
            (3..=10).contains(&planes.len()),
            "expected 3–10 planes, got {}",
            planes.len()
        );

        // Coefficient signs must satisfy physical constraints.
        for (idx, plane) in planes.iter().enumerate() {
            assert!(
                plane.gamma_v > 0.0,
                "plane {idx}: gamma_v={} must be > 0",
                plane.gamma_v
            );
            assert!(
                plane.gamma_q > 0.0,
                "plane {idx}: gamma_q={} must be > 0",
                plane.gamma_q
            );
            assert!(
                plane.gamma_s <= 0.0,
                "plane {idx}: gamma_s={} must be <= 0",
                plane.gamma_s
            );
        }

        // Verify resolve_stage_model correctly wraps the cached planes.
        let empty_hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
            HashMap::new();
        let model = super::resolve_stage_model(
            &hydro,
            &stage,
            Some(&config),
            ProductionModelSource::ComputedFromGeometry,
            &empty_hyperplane_map,
            Some(planes),
            None,
        )
        .expect("resolve_stage_model must succeed for ComputedFromGeometry with cached planes");

        match model {
            ResolvedProductionModel::Fpha {
                planes: out_planes, ..
            } => {
                assert_eq!(
                    out_planes.len(),
                    planes.len(),
                    "stage model must have the same plane count as the fitted planes"
                );
            }
            other => panic!("expected Fpha variant, got {other:?}"),
        }
    }

    /// Mixed precomputed + computed sources: both hydros resolve to valid `Fpha` models and
    /// provenance is correctly differentiated by source.
    ///
    /// Hydro 0: `source: "precomputed"` with 3 manually-constructed hyperplane rows.
    /// Hydro 1: `source: "computed"` with Sobradinho-style geometry.
    #[test]
    fn mixed_precomputed_and_computed_sources_resolve_correctly() {
        // Hydro 0: precomputed FPHA.
        let hydro0 = make_hydro(0, HydroGenerationModel::Fpha);
        let config0 = precomputed_fpha_config(0);

        let precomp_row_a = valid_row(0, None, 0);
        let precomp_row_b = valid_row(0, None, 1);
        let precomp_row_c = valid_row(0, None, 2);
        let mut hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
            HashMap::new();
        hyperplane_map.insert(
            (EntityId::from(0), None),
            vec![&precomp_row_a, &precomp_row_b, &precomp_row_c],
        );

        // Hydro 1: computed FPHA.
        let hydro1 = make_sobradinho_computed_hydro(1);
        let config1 = computed_fpha_config(1);

        let geo_rows = make_sobradinho_geometry_rows(1);
        let geo_refs: Vec<&HydroGeometryRow> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        geometry_map.insert(EntityId::from(1), geo_refs);

        let stage = make_stage(0);
        let study_stages = [stage.clone()];
        let stage_refs: Vec<&cobre_core::temporal::Stage> = study_stages.iter().collect();

        // Determine sources.
        let src0 = determine_source(&hydro0, Some(&config0)).expect("hydro0 source");
        let src1 = determine_source(&hydro1, Some(&config1)).expect("hydro1 source");
        assert_eq!(
            src0,
            ProductionModelSource::PrecomputedHyperplanes,
            "hydro 0 must be PrecomputedHyperplanes"
        );
        assert_eq!(
            src1,
            ProductionModelSource::ComputedFromGeometry,
            "hydro 1 must be ComputedFromGeometry"
        );

        // Fit computed planes for hydro 1.
        let computed_fit =
            super::fit_planes_for_hydro(&hydro1, Some(&config1), &geometry_map, &stage_refs)
                .expect("fit_planes_for_hydro must succeed for hydro 1");

        // Resolve stage model for hydro 0 (precomputed path).
        let model0 = super::resolve_stage_model(
            &hydro0,
            &stage,
            Some(&config0),
            src0,
            &hyperplane_map,
            None,
            None,
        )
        .expect("resolve_stage_model must succeed for hydro 0 (precomputed)");

        // Resolve stage model for hydro 1 (computed path, cached planes).
        let empty_hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
            HashMap::new();
        let model1 = super::resolve_stage_model(
            &hydro1,
            &stage,
            Some(&config1),
            src1,
            &empty_hyperplane_map,
            Some(&computed_fit.planes),
            None,
        )
        .expect("resolve_stage_model must succeed for hydro 1 (computed)");

        // Both models must be Fpha.
        assert!(
            matches!(model0, ResolvedProductionModel::Fpha { .. }),
            "hydro 0 must resolve to Fpha, got {model0:?}"
        );
        assert!(
            matches!(model1, ResolvedProductionModel::Fpha { .. }),
            "hydro 1 must resolve to Fpha, got {model1:?}"
        );

        // Provenance in canonical id-sorted order: [(id=0, Precomputed), (id=1, Computed)].
        assert_eq!(
            src0,
            ProductionModelSource::PrecomputedHyperplanes,
            "provenance[0] must be PrecomputedHyperplanes"
        );
        assert_eq!(
            src1,
            ProductionModelSource::ComputedFromGeometry,
            "provenance[1] must be ComputedFromGeometry"
        );
    }

    /// Computed-source all-stages-same: three stages all receive plane vectors with identical
    /// coefficients, confirming that the outer loop fits once and clones for every stage.
    #[test]
    fn computed_source_all_stages_produce_identical_planes() {
        let hydro = make_sobradinho_computed_hydro(0);
        let config = computed_fpha_config(0);

        let geo_rows = make_sobradinho_geometry_rows(0);
        let geo_refs: Vec<&HydroGeometryRow> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        // Three study stages.
        let stages = [make_stage(0), make_stage(1), make_stage(2)];
        let stage_refs: Vec<&cobre_core::temporal::Stage> = stages.iter().collect();

        // Fit once.
        let cached_fit =
            super::fit_planes_for_hydro(&hydro, Some(&config), &geometry_map, &stage_refs)
                .expect("fit_planes_for_hydro must succeed");

        let empty_hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
            HashMap::new();

        // Resolve for each stage and collect planes.
        let stage_planes: Vec<Vec<FphaPlane>> = stages
            .iter()
            .map(|stage| {
                let model = super::resolve_stage_model(
                    &hydro,
                    stage,
                    Some(&config),
                    ProductionModelSource::ComputedFromGeometry,
                    &empty_hyperplane_map,
                    Some(&cached_fit.planes),
                    None,
                )
                .expect("resolve_stage_model must succeed");
                match model {
                    ResolvedProductionModel::Fpha { planes, .. } => planes,
                    other => panic!("expected Fpha, got {other:?}"),
                }
            })
            .collect();

        assert_eq!(
            stage_planes.len(),
            3,
            "must have plane vectors for 3 stages"
        );

        // All stages must have the same plane count.
        let expected_count = stage_planes[0].len();
        for (s, planes) in stage_planes.iter().enumerate() {
            assert_eq!(
                planes.len(),
                expected_count,
                "stage {s}: plane count must be {expected_count}, got {}",
                planes.len()
            );
        }

        // Planes must be bitwise-identical across stages (cloned from the same source).
        for (s, planes) in stage_planes.iter().enumerate().skip(1) {
            for (p, plane) in planes.iter().enumerate() {
                assert_eq!(
                    *plane, stage_planes[0][p],
                    "stage {s} plane {p}: must be identical to stage 0 plane {p}"
                );
            }
        }
    }

    /// `validate_computed_prerequisites`: missing `efficiency` returns `SddpError::Validation`
    /// with a message containing both "efficiency" and the hydro name "TestHydro".
    #[test]
    fn computed_source_missing_efficiency_returns_validation_error() {
        let mut hydro = make_computed_hydro(0);
        hydro.name = "TestHydro".to_string();
        hydro.efficiency = None; // remove efficiency to trigger prerequisite failure

        let rows = make_geometry_rows(0);
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        let row_refs: Vec<&HydroGeometryRow> = rows.iter().collect();
        geometry_map.insert(EntityId::from(0), row_refs);

        let err = validate_computed_prerequisites(&hydro, &geometry_map)
            .expect_err("should fail when efficiency is None");
        let msg = err.to_string();
        assert!(
            msg.contains("efficiency"),
            "error must mention 'efficiency', got: {msg}"
        );
        assert!(
            msg.contains("TestHydro"),
            "error must include hydro name 'TestHydro', got: {msg}"
        );
    }

    /// `validate_computed_prerequisites`: missing `hydraulic_losses` returns `SddpError::Validation`
    /// with a message containing "hydraulic_losses" and the hydro name.
    #[test]
    fn computed_source_missing_losses_returns_validation_error() {
        let mut hydro = make_computed_hydro(0);
        hydro.hydraulic_losses = None; // remove losses to trigger prerequisite failure

        let rows = make_geometry_rows(0);
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        let row_refs: Vec<&HydroGeometryRow> = rows.iter().collect();
        geometry_map.insert(EntityId::from(0), row_refs);

        let err = validate_computed_prerequisites(&hydro, &geometry_map)
            .expect_err("should fail when hydraulic_losses is None");
        let msg = err.to_string();
        assert!(
            msg.contains("hydraulic_losses"),
            "error must mention 'hydraulic_losses', got: {msg}"
        );
        assert!(
            msg.contains(&hydro.name),
            "error must include hydro name '{}', got: {msg}",
            hydro.name
        );
    }
}
