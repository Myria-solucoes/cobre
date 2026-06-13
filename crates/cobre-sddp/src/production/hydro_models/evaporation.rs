//! Evaporation model resolution: per-hydro linearized evaporation from geometry.
//!
//! Resolves the per-`(hydro, stage)` linearized evaporation coefficients from
//! reservoir Volume-Height-Area geometry, plus the supporting area
//! interpolation and finite-difference derivative helpers. The resolved
//! coefficients feed the water-balance row in the LP builder.

use std::collections::HashMap;
use std::path::Path;

use cobre_core::{EntityId, System};
use cobre_io::extensions::HydroGeometryRow;

use super::load_artifacts_for_hydro_models;
use super::types::{
    EvaporationModel, EvaporationModelSet, EvaporationReferenceSource, EvaporationSource,
    LinearizedEvaporation,
};
use crate::SddpError;
// ── Evaporation model resolution ──────────────────────────────────────────────

/// Resolve per-hydro linearized evaporation models from reservoir geometry.
///
/// Scans `system.hydros()` for plants with `evaporation_coefficients_mm` set.
/// When none are found, returns `EvaporationModel::None` for every hydro
/// without touching the filesystem. When any hydros have evaporation
/// coefficients, loads `system/hydro_geometry.parquet` and computes a
/// first-order Taylor linearization around the operating midpoint.
///
/// The linearized evaporation model for stage `t` is:
///
/// ```text
/// Q_ev = k_evap0 + k_evap_v * v
/// ```
///
/// where:
///
/// ```text
/// zeta  = 1.0 / (3.6 * stage_hours)         -- mm·km² → m³/s
/// k_evap_v = zeta * c_ev[month] * dA/dv|_{v_ref}
/// k_evap0  = zeta * c_ev[month] * A(v_ref) - k_evap_v * v_ref
/// ```
///
/// `v_ref = (v_min + v_max) / 2` is the linearization reference volume.
/// `stage_hours` is the sum of all block durations in the stage.
/// `month` is the 0-based calendar month from `stage.season_id` (0 = January).
///
/// # Errors
///
/// | Condition                                                        | Error variant             |
/// | ---------------------------------------------------------------- | ------------------------- |
/// | Hydro has evaporation coefficients but no geometry rows          | [`SddpError::Validation`] |
/// | All geometry `area_km2` values are zero                          | [`SddpError::Validation`] |
/// | Computed `k_evap_v` or `k_evap0` is NaN or infinite             | [`SddpError::Validation`] |
/// | Stage has no `season_id` (cannot map to a month)                 | [`SddpError::Validation`] |
/// | I/O failure loading geometry Parquet                             | [`SddpError::Io`]         |
// Rationale: the return tuple carries three independently typed outputs of the
// evaporation resolution step — the model set, the per-hydro source provenance,
// and the per-hydro reference-source provenance; a type alias would obscure the
// concrete types that callers destructure immediately at every call site.
#[allow(clippy::type_complexity)]
pub fn resolve_evaporation_models(
    system: &System,
    case_dir: &Path,
) -> Result<
    (
        EvaporationModelSet,
        Vec<(EntityId, EvaporationSource)>,
        Vec<(EntityId, EvaporationReferenceSource)>,
    ),
    SddpError,
> {
    let artifacts = load_artifacts_for_hydro_models(case_dir)?;
    resolve_evaporation_models_from_artifacts(system, &artifacts)
}

/// Variant of [`resolve_evaporation_models`] that consumes a pre-parsed
/// [`cobre_io::CaseArtifacts`] bundle.
///
/// # Errors
///
/// Same conditions as [`resolve_evaporation_models`].
// Rationale: the return tuple names the three independently typed outputs of the
// evaporation resolution step — the model set, the per-hydro source provenance, and
// the per-hydro reference-source provenance; a type alias would hide the concrete types
// that callers destructure immediately, making the three-way split less readable at
// the call sites.
#[allow(clippy::type_complexity)]
pub fn resolve_evaporation_models_from_artifacts(
    system: &System,
    artifacts: &cobre_io::CaseArtifacts,
) -> Result<
    (
        EvaporationModelSet,
        Vec<(EntityId, EvaporationSource)>,
        Vec<(EntityId, EvaporationReferenceSource)>,
    ),
    SddpError,
> {
    let any_evaporation = system
        .hydros()
        .iter()
        .any(|h| h.evaporation_coefficients_mm.is_some());

    if !any_evaporation {
        let models = system
            .hydros()
            .iter()
            .map(|_| EvaporationModel::None)
            .collect();
        let provenance = system
            .hydros()
            .iter()
            .map(|h| (h.id, EvaporationSource::NotModeled))
            .collect();
        let reference_sources = system
            .hydros()
            .iter()
            .map(|h| (h.id, EvaporationReferenceSource::DefaultMidpoint))
            .collect();
        return Ok((
            EvaporationModelSet::new(models),
            provenance,
            reference_sources,
        ));
    }

    let geometry_rows: &[HydroGeometryRow] = &artifacts.hydro_geometry;

    let mut geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
        HashMap::new();
    for row in geometry_rows {
        geometry_map.entry(row.hydro_id).or_default().push(row);
    }
    // Sort each hydro's rows by volume_hm3 ascending (should already be sorted,
    // but guarantee it here since we rely on sorted order for interpolation).
    for rows in geometry_map.values_mut() {
        rows.sort_by(|a, b| a.volume_hm3.total_cmp(&b.volume_hm3));
    }

    let study_stages: Vec<&cobre_core::temporal::Stage> =
        system.stages().iter().filter(|s| s.id >= 0).collect();

    resolve_evaporation_core(system.hydros(), &geometry_map, &study_stages)
}

/// Core evaporation linearization logic, operating on pre-loaded data.
///
/// Separated from [`resolve_evaporation_models`] so that unit tests can exercise
/// the resolution logic without loading files from disk. Takes pre-loaded, grouped
/// geometry rows and the ordered set of study stages.
///
/// # Errors
///
/// Same error conditions as [`resolve_evaporation_models`].
// Rationale: the function produces three independently typed outputs (same
// three-tuple as [`resolve_evaporation_models`]); a type alias would obscure the
// concrete types that callers destructure immediately.  The length is necessary
// because the per-stage linearization loop must handle two interleaved
// reference-volume paths (user-supplied vs. computed midpoint) together with
// geometry interpolation, unit-conversion, and finite-value validation; splitting
// would require threading several computed intermediates across helper boundaries.
#[allow(clippy::type_complexity, clippy::too_many_lines)]
fn resolve_evaporation_core(
    hydros: &[cobre_core::entities::hydro::Hydro],
    geometry_map: &HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>>,
    study_stages: &[&cobre_core::temporal::Stage],
) -> Result<
    (
        EvaporationModelSet,
        Vec<(EntityId, EvaporationSource)>,
        Vec<(EntityId, EvaporationReferenceSource)>,
    ),
    SddpError,
> {
    let n_stages = study_stages.len();
    let mut all_models: Vec<EvaporationModel> = Vec::with_capacity(hydros.len());
    let mut provenance: Vec<(EntityId, EvaporationSource)> = Vec::with_capacity(hydros.len());
    let mut reference_provenance: Vec<(EntityId, EvaporationReferenceSource)> =
        Vec::with_capacity(hydros.len());

    for hydro in hydros {
        let Some(coefficients_mm) = hydro.evaporation_coefficients_mm else {
            all_models.push(EvaporationModel::None);
            provenance.push((hydro.id, EvaporationSource::NotModeled));
            reference_provenance.push((hydro.id, EvaporationReferenceSource::DefaultMidpoint));
            continue;
        };

        let geo_rows: &[&cobre_io::extensions::HydroGeometryRow] =
            geometry_map.get(&hydro.id).map_or(&[], Vec::as_slice);

        if geo_rows.is_empty() {
            return Err(SddpError::Validation(format!(
                "hydro {} (id={}) has evaporation_coefficients_mm but no geometry data \
                 in hydro_geometry.parquet. Evaporation linearization requires \
                 area-volume curve data.",
                hydro.name, hydro.id.0
            )));
        }

        let all_zero_area = geo_rows.iter().all(|r| r.area_km2 == 0.0);
        if all_zero_area {
            return Err(SddpError::Validation(format!(
                "hydro {} (id={}) has evaporation_coefficients_mm but all area_km2 \
                 values in hydro_geometry.parquet are zero. \
                 Evaporation linearization requires non-zero surface area data.",
                hydro.name, hydro.id.0
            )));
        }

        // When the hydro supplies per-season reference volumes, use them per
        // stage (compute A and dA/dv inside the loop). Otherwise fall back to
        // the midpoint once outside the loop.
        let ref_source = if hydro.evaporation_reference_volumes_hm3.is_some() {
            EvaporationReferenceSource::UserSupplied
        } else {
            EvaporationReferenceSource::DefaultMidpoint
        };

        // For the midpoint path, pre-compute v_ref / A(v_ref) / dA/dv once.
        // These are only accessed when evaporation_reference_volumes_hm3 is None,
        // so the values are always initialised before use.
        let midpoint_v = f64::midpoint(hydro.min_storage_hm3, hydro.max_storage_hm3);
        let midpoint_area = if hydro.evaporation_reference_volumes_hm3.is_none() {
            interpolate_area(geo_rows, midpoint_v)
        } else {
            0.0 // unused; per-season path computes per stage
        };
        let midpoint_slope = if hydro.evaporation_reference_volumes_hm3.is_none() {
            area_derivative(geo_rows, midpoint_v)
        } else {
            0.0 // unused; per-season path computes per stage
        };

        let mut stage_coefficients: Vec<LinearizedEvaporation> = Vec::with_capacity(n_stages);
        let mut stage_ref_volumes: Vec<f64> = Vec::with_capacity(n_stages);

        for stage in study_stages {
            let month_index = stage.season_id.ok_or_else(|| {
                SddpError::Validation(format!(
                    "stage {} has no season_id and cannot be mapped to a calendar month \
                     for evaporation coefficient lookup (hydro {} id={}). \
                     All study stages must have a season_id for evaporation modeling.",
                    stage.id, hydro.name, hydro.id.0
                ))
            })?;

            if month_index >= 12 {
                return Err(SddpError::Validation(format!(
                    "stage {} has season_id={month_index} which is outside [0, 11]. \
                     Evaporation coefficient arrays have 12 entries (one per calendar month). \
                     (hydro {} id={})",
                    stage.id, hydro.name, hydro.id.0
                )));
            }

            let c_ev = coefficients_mm[month_index];

            // Resolve v_ref, a_ref, da_dv for this stage.
            let (v_ref, a_ref, da_dv) =
                if let Some(ref_vols) = hydro.evaporation_reference_volumes_hm3 {
                    // Per-season path: look up the reference volume for this month.
                    let v = ref_vols[month_index];
                    (
                        v,
                        interpolate_area(geo_rows, v),
                        area_derivative(geo_rows, v),
                    )
                } else {
                    // Midpoint path: use the values pre-computed outside the loop.
                    (midpoint_v, midpoint_area, midpoint_slope)
                };

            // Total stage duration in hours (sum of all block durations).
            let stage_hours: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();

            let zeta_evap = 1.0 / (3.6 * stage_hours);

            let k_evap_v = zeta_evap * c_ev * da_dv;
            let k_evap0 = zeta_evap * c_ev * a_ref - k_evap_v * v_ref;

            // Validate finiteness (catches degenerate geometry or zero-duration stages).
            if !k_evap_v.is_finite() {
                return Err(SddpError::Validation(format!(
                    "hydro {} (id={}) stage {}: computed k_evap_v = {k_evap_v} is not \
                     finite. Check geometry data for degenerate area-volume curve points.",
                    hydro.name, hydro.id.0, stage.id
                )));
            }
            if !k_evap0.is_finite() {
                return Err(SddpError::Validation(format!(
                    "hydro {} (id={}) stage {}: computed k_evap0 = {k_evap0} is not \
                     finite. Check geometry data for degenerate area-volume curve points.",
                    hydro.name, hydro.id.0, stage.id
                )));
            }

            stage_coefficients.push(LinearizedEvaporation { k_evap0, k_evap_v });
            stage_ref_volumes.push(v_ref);
        }

        all_models.push(EvaporationModel::Linearized {
            coefficients: stage_coefficients,
            reference_volumes_hm3: stage_ref_volumes,
        });
        provenance.push((hydro.id, EvaporationSource::LinearizedFromGeometry));
        reference_provenance.push((hydro.id, ref_source));
    }

    Ok((
        EvaporationModelSet::new(all_models),
        provenance,
        reference_provenance,
    ))
}

// ── Evaporation geometry helpers ──────────────────────────────────────────────

/// Linearly interpolate reservoir surface area at volume `v` from the sorted
/// geometry table.
///
/// When `v` is below the first point or above the last point, returns the
/// area at the first or last point respectively (clamping, not extrapolation).
/// When `v` falls exactly on a geometry point, returns that point's area.
/// Between two points, performs linear interpolation.
///
/// Assumes `geometry` is sorted by `volume_hm3` ascending and non-empty.
/// Returns `0.0` for an empty geometry slice.
fn interpolate_area(geometry: &[&cobre_io::extensions::HydroGeometryRow], v: f64) -> f64 {
    if geometry.is_empty() {
        return 0.0;
    }

    let n = geometry.len();

    // Clamp below the first point.
    if v <= geometry[0].volume_hm3 {
        return geometry[0].area_km2;
    }

    // Clamp above the last point.
    if v >= geometry[n - 1].volume_hm3 {
        return geometry[n - 1].area_km2;
    }

    // Binary search for the interval [i, i+1] that straddles v.
    // We know v is strictly between geometry[0] and geometry[n-1].
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = lo.midpoint(hi);
        if geometry[mid].volume_hm3 <= v {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let v0 = geometry[lo].volume_hm3;
    let v1 = geometry[hi].volume_hm3;
    let a0 = geometry[lo].area_km2;
    let a1 = geometry[hi].area_km2;

    // Linear interpolation: A(v) = A0 + (A1 - A0) * (v - v0) / (v1 - v0).
    // Guard against identical volume points (degenerate geometry).
    let dv = v1 - v0;
    if dv == 0.0 {
        return a0;
    }

    a0 + (a1 - a0) * (v - v0) / dv
}

/// Compute the finite-difference derivative `dA/dv` at volume `v` from the
/// sorted geometry table.
///
/// Uses the slope of the enclosing interval `[i, i+1]`. When `v` is at or
/// below the first point, uses the slope between the first and second points.
/// When `v` is at or above the last point, uses the slope between the last two
/// points. For a single-point geometry, returns `0.0` (no gradient information).
///
/// Assumes `geometry` is sorted by `volume_hm3` ascending.
fn area_derivative(geometry: &[&cobre_io::extensions::HydroGeometryRow], v: f64) -> f64 {
    let n = geometry.len();

    if n < 2 {
        // Single-point or empty geometry: no gradient information.
        return 0.0;
    }

    // Determine the interval to use for the finite difference.
    let (lo, hi) = if v <= geometry[0].volume_hm3 {
        // At or below the first point: use the first interval.
        (0, 1)
    } else if v >= geometry[n - 1].volume_hm3 {
        // At or above the last point: use the last interval.
        (n - 2, n - 1)
    } else {
        // Binary search for the enclosing interval.
        let mut l = 0usize;
        let mut r = n - 1;
        while r - l > 1 {
            let mid = l.midpoint(r);
            if geometry[mid].volume_hm3 <= v {
                l = mid;
            } else {
                r = mid;
            }
        }
        (l, r)
    };

    let dv = geometry[hi].volume_hm3 - geometry[lo].volume_hm3;
    let da = geometry[hi].area_km2 - geometry[lo].area_km2;

    // Guard against identical volume points (degenerate geometry).
    if dv == 0.0 {
        return 0.0;
    }

    da / dv
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
        EntityId,
        entities::hydro::{HydroGenerationModel, HydroPenalties},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };

    use super::*;

    // ── Test helpers ──────────────────────────────────────────────────────────

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

    /// Helper: build a slice of HydroGeometryRow references for interpolation tests.
    fn make_geo_rows(volume_area: &[(f64, f64)]) -> Vec<cobre_io::extensions::HydroGeometryRow> {
        volume_area
            .iter()
            .map(|&(v, a)| cobre_io::extensions::HydroGeometryRow {
                hydro_id: EntityId::from(1),
                volume_hm3: v,
                height_m: 0.0,
                area_km2: a,
            })
            .collect()
    }

    /// Helper: build a Stage with the given id and season_id (month index).
    fn make_stage_with_month(id: i32, month: usize) -> Stage {
        Stage {
            index: usize::try_from(id.max(0)).unwrap_or(0),
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap_or_default(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap_or_default(),
            season_id: Some(month),
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

    /// Helper: build a Hydro with the given id and evaporation coefficients.
    fn make_hydro_with_evaporation(
        id: i32,
        min_storage: f64,
        max_storage: f64,
        evap_mm: Option<[f64; 12]>,
    ) -> cobre_core::entities::hydro::Hydro {
        cobre_core::entities::hydro::Hydro {
            id: EntityId::from(id),
            name: format!("Hydro{id}"),
            bus_id: EntityId::from(10),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: min_storage,
            max_storage_hm3: max_storage,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 500.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 1000.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: evap_mm,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_penalties(),
        }
    }

    /// interpolate_area: exact match on the first geometry point returns that area.
    #[test]
    fn interpolate_area_exact_first_point() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::interpolate_area(&refs, 100.0);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "exact first point: expected 1.0, got {result}"
        );
    }

    /// interpolate_area: exact match on the last geometry point returns that area.
    #[test]
    fn interpolate_area_exact_last_point() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::interpolate_area(&refs, 300.0);
        assert!(
            (result - 2.0).abs() < 1e-10,
            "exact last point: expected 2.0, got {result}"
        );
    }

    /// interpolate_area: exact match on a middle geometry point returns that area.
    #[test]
    fn interpolate_area_exact_middle_point() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::interpolate_area(&refs, 200.0);
        assert!(
            (result - 1.5).abs() < 1e-10,
            "exact middle point: expected 1.5, got {result}"
        );
    }

    /// interpolate_area: midpoint between two geometry points is linearly interpolated.
    ///
    /// Geometry: volumes [100, 200, 300, 400, 500], areas [1.0, 1.5, 2.0, 2.5, 3.0].
    /// At v=300, A(300) = 2.0 (exact match). At v=250, A(250) = 1.75 (midpoint of [1.5, 2.0]).
    #[test]
    fn interpolate_area_midpoint_between_two_points() {
        let rows = make_geo_rows(&[
            (100.0, 1.0),
            (200.0, 1.5),
            (300.0, 2.0),
            (400.0, 2.5),
            (500.0, 3.0),
        ]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::interpolate_area(&refs, 250.0);
        // Midpoint between (200, 1.5) and (300, 2.0): 1.5 + 0.5 * (2.0 - 1.5) = 1.75
        assert!(
            (result - 1.75).abs() < 1e-10,
            "midpoint: expected 1.75, got {result}"
        );
    }

    /// interpolate_area: volume below first point clamps to first area.
    #[test]
    fn interpolate_area_clamps_below_first_point() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::interpolate_area(&refs, 50.0);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "below first point: expected clamped area 1.0, got {result}"
        );
    }

    /// interpolate_area: volume above last point clamps to last area.
    #[test]
    fn interpolate_area_clamps_above_last_point() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::interpolate_area(&refs, 400.0);
        assert!(
            (result - 2.0).abs() < 1e-10,
            "above last point: expected clamped area 2.0, got {result}"
        );
    }

    // ── area_derivative unit tests ────────────────────────────────────────────

    /// area_derivative: correct finite difference between two points spanning v.
    ///
    /// Geometry: volumes [100, 200, 300, 400, 500], areas [1.0, 1.5, 2.0, 2.5, 3.0].
    /// dA/dv at v=300 uses the interval [200, 300]: (2.0 - 1.5) / (300 - 200) = 0.005.
    #[test]
    fn area_derivative_correct_finite_difference() {
        let rows = make_geo_rows(&[
            (100.0, 1.0),
            (200.0, 1.5),
            (300.0, 2.0),
            (400.0, 2.5),
            (500.0, 3.0),
        ]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::area_derivative(&refs, 300.0);
        // Interval [200, 300]: (2.0 - 1.5) / (300 - 200) = 0.005
        assert!(
            (result - 0.005).abs() < 1e-10,
            "dA/dv at 300: expected 0.005, got {result}"
        );
    }

    /// area_derivative: single-point geometry returns 0.0.
    #[test]
    fn area_derivative_single_point_returns_zero() {
        let rows = make_geo_rows(&[(200.0, 1.5)]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::area_derivative(&refs, 200.0);
        assert!(
            result.abs() < 1e-10,
            "single-point geometry: expected dA/dv = 0.0, got {result}"
        );
    }

    /// area_derivative: at or below the first point uses the first interval.
    #[test]
    fn area_derivative_at_or_below_first_point_uses_first_interval() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        // First interval: (1.5 - 1.0) / (200 - 100) = 0.005
        let result = super::area_derivative(&refs, 50.0);
        assert!(
            (result - 0.005).abs() < 1e-10,
            "below first point: expected first-interval slope 0.005, got {result}"
        );
    }

    /// area_derivative: at or above the last point uses the last interval.
    #[test]
    fn area_derivative_at_or_above_last_point_uses_last_interval() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        // Last interval: (2.0 - 1.5) / (300 - 200) = 0.005
        let result = super::area_derivative(&refs, 400.0);
        assert!(
            (result - 0.005).abs() < 1e-10,
            "above last point: expected last-interval slope 0.005, got {result}"
        );
    }

    /// resolve_evaporation_models core logic: all-no-evaporation system returns all None
    /// without geometry lookup.
    ///
    /// This test calls the internal core logic directly without loading from disk by
    /// using an empty geometry map.
    #[test]
    fn resolve_evaporation_all_none_when_no_hydro_has_coefficients() {
        let hydros = vec![
            make_hydro_with_evaporation(0, 100.0, 500.0, None),
            make_hydro_with_evaporation(1, 200.0, 1000.0, None),
        ];

        // Build the geometry map (empty, since no hydro needs it).
        let geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
            HashMap::new();
        let study_stages = [make_stage_with_month(0, 0)];
        let stage_refs: Vec<_> = study_stages.iter().collect();

        let (models, provenance, _ref_provenance) =
            super::resolve_evaporation_core(&hydros, &geometry_map, &stage_refs)
                .expect("should succeed for all-no-evaporation");

        assert_eq!(models.n_hydros(), 2);
        assert!(
            matches!(models.model(0), EvaporationModel::None),
            "hydro 0 must be None"
        );
        assert!(
            matches!(models.model(1), EvaporationModel::None),
            "hydro 1 must be None"
        );
        assert!(!models.has_evaporation(), "has_evaporation() must be false");
        assert_eq!(provenance.len(), 2);
        assert!(
            provenance
                .iter()
                .all(|(_, src)| *src == EvaporationSource::NotModeled)
        );
    }

    /// resolve_evaporation_models core logic: known geometry + coefficient gives correct k_evap0 and k_evap_v.
    ///
    /// Spec (acceptance criterion 2):
    ///   hydro: v_min=100, v_max=500, evaporation_coefficients_mm=[5.0; 12]
    ///   geometry: volumes [100, 200, 300, 400, 500], areas [1.0, 1.5, 2.0, 2.5, 3.0]
    ///   v_ref = (100 + 500) / 2 = 300
    ///   A(300) = 2.0
    ///   dA/dv|_300 = (2.0 - 1.5) / (300 - 200) = 0.005
    ///   stage: season_id=0 (January), duration=744h
    ///   zeta = 1 / (3.6 * 744) = 1 / 2678.4
    ///   c_ev = 5.0
    ///   k_evap_v = zeta * 5.0 * 0.005
    ///   k_evap0  = zeta * 5.0 * 2.0 - k_evap_v * 300
    #[test]
    fn resolve_evaporation_known_geometry_produces_correct_coefficients() {
        let evap_mm = [5.0f64; 12];
        let hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some(evap_mm));

        let geo_rows = make_geo_rows(&[
            (100.0, 1.0),
            (200.0, 1.5),
            (300.0, 2.0),
            (400.0, 2.5),
            (500.0, 3.0),
        ]);
        let geo_refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
            HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        let study_stages = [make_stage_with_month(0, 0)]; // January
        let stage_refs: Vec<_> = study_stages.iter().collect();

        let (models, provenance, _ref_provenance) =
            super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
                .expect("should succeed");

        assert_eq!(models.n_hydros(), 1);
        assert_eq!(provenance.len(), 1);
        assert_eq!(provenance[0].1, EvaporationSource::LinearizedFromGeometry);

        match models.model(0) {
            EvaporationModel::Linearized {
                coefficients,
                reference_volumes_hm3,
            } => {
                assert_eq!(
                    reference_volumes_hm3.len(),
                    1,
                    "must have one ref volume per stage"
                );
                assert!(
                    (reference_volumes_hm3[0] - 300.0).abs() < 1e-10,
                    "v_ref must be (100+500)/2 = 300, got {}",
                    reference_volumes_hm3[0]
                );
                assert_eq!(coefficients.len(), 1);

                let v_ref = 300.0_f64;
                let a_ref = 2.0_f64;
                let da_dv = 0.005_f64;
                let c_ev = 5.0_f64;
                let stage_hours = 744.0_f64;
                let zeta = 1.0 / (3.6 * stage_hours);

                let expected_k_evap_v = zeta * c_ev * da_dv;
                let expected_k_evap0 = zeta * c_ev * a_ref - expected_k_evap_v * v_ref;

                let coeff = &coefficients[0];
                assert!(
                    (coeff.k_evap_v - expected_k_evap_v).abs() < 1e-10,
                    "k_evap_v: expected {expected_k_evap_v}, got {}",
                    coeff.k_evap_v
                );
                assert!(
                    (coeff.k_evap0 - expected_k_evap0).abs() < 1e-10,
                    "k_evap0: expected {expected_k_evap0}, got {}",
                    coeff.k_evap0
                );
            }
            other => panic!("expected Linearized, got {other:?}"),
        }
    }

    /// resolve_evaporation_models core logic: negative evaporation coefficients produce valid results.
    ///
    /// Net precipitation (negative c_ev) is physically valid. k_evap_v can be negative.
    #[test]
    fn resolve_evaporation_negative_coefficient_produces_valid_results() {
        let mut evap_mm = [0.0f64; 12];
        evap_mm[0] = -3.0; // net precipitation in January
        let hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some(evap_mm));

        let geo_rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let geo_refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
            HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        let study_stages = [make_stage_with_month(0, 0)]; // January
        let stage_refs: Vec<_> = study_stages.iter().collect();

        let (models, provenance, _ref_provenance) =
            super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
                .expect("negative evaporation must succeed");

        assert_eq!(provenance[0].1, EvaporationSource::LinearizedFromGeometry);

        match models.model(0) {
            EvaporationModel::Linearized { coefficients, .. } => {
                let coeff = &coefficients[0];
                assert!(
                    coeff.k_evap_v.is_finite(),
                    "k_evap_v must be finite for negative c_ev"
                );
                assert!(
                    coeff.k_evap0.is_finite(),
                    "k_evap0 must be finite for negative c_ev"
                );
                // Negative c_ev with positive dA/dv → negative k_evap_v.
                assert!(
                    coeff.k_evap_v < 0.0,
                    "k_evap_v must be negative for net precipitation scenario"
                );
            }
            other => panic!("expected Linearized, got {other:?}"),
        }
    }

    /// resolve_evaporation_models core logic: hydro with evaporation but no geometry rows
    /// returns SddpError::Validation containing "geometry".
    #[test]
    fn resolve_evaporation_missing_geometry_returns_validation_error() {
        let evap_mm = [5.0f64; 12];
        let hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some(evap_mm));

        // Geometry map has no entry for hydro 0.
        let geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
            HashMap::new();

        let study_stages = [make_stage_with_month(0, 0)];
        let stage_refs: Vec<_> = study_stages.iter().collect();

        let err = super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
            .expect_err("missing geometry must return an error");

        match err {
            crate::SddpError::Validation(msg) => {
                assert!(
                    msg.to_lowercase().contains("geometry"),
                    "error message must mention 'geometry', got: {msg}"
                );
            }
            other => panic!("expected Validation error, got {other:?}"),
        }
    }

    /// resolve_evaporation_models core logic: 4 hydros where 2 have evaporation and 2 do not.
    ///
    /// Acceptance criterion 1: returns 2 Linearized and 2 None models.
    #[test]
    fn resolve_evaporation_mixed_system_returns_correct_model_mix() {
        let evap_mm = [5.0f64; 12];
        let hydros = vec![
            make_hydro_with_evaporation(0, 100.0, 500.0, Some(evap_mm)),
            make_hydro_with_evaporation(1, 200.0, 1000.0, None),
            make_hydro_with_evaporation(2, 50.0, 300.0, Some(evap_mm)),
            make_hydro_with_evaporation(3, 300.0, 2000.0, None),
        ];

        let geo_rows_h0 = make_geo_rows(&[(100.0, 1.0), (300.0, 2.0), (500.0, 3.0)]);
        let geo_rows_h2 = make_geo_rows(&[(50.0, 0.5), (175.0, 1.0), (300.0, 1.5)]);

        let refs_h0: Vec<_> = geo_rows_h0.iter().collect();
        let refs_h2: Vec<_> = geo_rows_h2.iter().collect();

        let mut geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
            HashMap::new();
        geometry_map.insert(EntityId::from(0), refs_h0);
        geometry_map.insert(EntityId::from(2), refs_h2);

        let study_stages = [make_stage_with_month(0, 0)];
        let stage_refs: Vec<_> = study_stages.iter().collect();

        let (models, provenance, _ref_provenance) =
            super::resolve_evaporation_core(&hydros, &geometry_map, &stage_refs)
                .expect("should succeed");

        assert_eq!(models.n_hydros(), 4);
        assert!(
            matches!(models.model(0), EvaporationModel::Linearized { .. }),
            "hydro 0 must be Linearized"
        );
        assert!(
            matches!(models.model(1), EvaporationModel::None),
            "hydro 1 must be None"
        );
        assert!(
            matches!(models.model(2), EvaporationModel::Linearized { .. }),
            "hydro 2 must be Linearized"
        );
        assert!(
            matches!(models.model(3), EvaporationModel::None),
            "hydro 3 must be None"
        );

        // 2 Linearized, 2 NotModeled in provenance.
        let n_linearized = provenance
            .iter()
            .filter(|(_, s)| *s == EvaporationSource::LinearizedFromGeometry)
            .count();
        let n_not_modeled = provenance
            .iter()
            .filter(|(_, s)| *s == EvaporationSource::NotModeled)
            .count();
        assert_eq!(n_linearized, 2, "expected 2 LinearizedFromGeometry");
        assert_eq!(n_not_modeled, 2, "expected 2 NotModeled");
    }

    /// resolve_evaporation_models core logic: NaN/Inf detection from degenerate geometry
    /// (two identical volume points) returns a validation error.
    #[test]
    fn resolve_evaporation_degenerate_geometry_nan_detected() {
        let evap_mm = [5.0f64; 12];
        let hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some(evap_mm));

        // Two rows with same volume but different areas: this is degenerate.
        // With dv=0, area_derivative returns 0.0, so no NaN there.
        // To force NaN we need the scenario where stage_hours=0 (zeta → Inf).
        // Test that scenario instead.
        let mut stage_zero_duration = make_stage_with_month(0, 0);
        stage_zero_duration.blocks = vec![Block {
            index: 0,
            name: "ZERO".to_string(),
            duration_hours: 0.0, // zero duration → zeta = Inf → k_evap_v = Inf
        }];

        let geo_rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let geo_refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
            HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        let stage_refs = vec![&stage_zero_duration];

        let err = super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
            .expect_err("degenerate geometry (zero duration) must return an error");

        assert!(
            matches!(err, crate::SddpError::Validation(_)),
            "expected Validation error for non-finite coefficients, got {err:?}"
        );
    }

    // ── Per-season reference volume tests ────────────────────────────────────

    /// resolve_evaporation_core: user-supplied per-season reference volumes produce
    /// stage coefficients derived from the month-specific v_ref.
    ///
    /// Geometry: volumes [100, 200, 300, 400, 500], areas [1.0, 1.5, 2.0, 2.5, 3.0].
    /// ref_vols[0] = 200 (January), ref_vols[1] = 400 (February).
    /// Hydro: v_min=100, v_max=500.
    /// Stage 0: season_id=0, 744h. Stage 1: season_id=1, 672h.
    ///
    /// For stage 0 (v_ref=200): A(200)=1.5, dA/dv=(2.0-1.5)/(300-200)=0.005
    /// For stage 1 (v_ref=400): A(400)=2.5, dA/dv=(3.0-2.5)/(500-400)=0.005
    #[test]
    fn resolve_evaporation_per_season_ref_vols_produces_per_stage_coefficients() {
        let mut ref_vols = [0.0f64; 12];
        ref_vols[0] = 200.0; // January
        ref_vols[1] = 400.0; // February

        let mut hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some([5.0f64; 12]));
        hydro.evaporation_reference_volumes_hm3 = Some(ref_vols);

        let geo_rows = make_geo_rows(&[
            (100.0, 1.0),
            (200.0, 1.5),
            (300.0, 2.0),
            (400.0, 2.5),
            (500.0, 3.0),
        ]);
        let geo_refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
            HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        // Two stages: January (744h) and February (672h).
        let stage_jan = make_stage_with_month(0, 0);
        let mut stage_feb = make_stage_with_month(1, 1);
        stage_feb.blocks = vec![Block {
            index: 0,
            name: "FEB".to_string(),
            duration_hours: 672.0,
        }];
        let stage_refs = vec![&stage_jan, &stage_feb];

        let (models, evap_provenance, ref_provenance) =
            super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
                .expect("should succeed");

        assert_eq!(models.n_hydros(), 1);
        assert_eq!(
            evap_provenance[0].1,
            EvaporationSource::LinearizedFromGeometry
        );
        assert_eq!(
            ref_provenance[0].1,
            EvaporationReferenceSource::UserSupplied,
            "user-supplied volumes must produce UserSupplied provenance"
        );

        match models.model(0) {
            EvaporationModel::Linearized {
                coefficients,
                reference_volumes_hm3,
            } => {
                assert_eq!(coefficients.len(), 2, "must have 2 stage coefficients");
                assert_eq!(reference_volumes_hm3.len(), 2, "must have 2 ref volumes");

                // Stage 0: v_ref=200
                assert!(
                    (reference_volumes_hm3[0] - 200.0).abs() < 1e-10,
                    "stage 0 ref vol must be 200, got {}",
                    reference_volumes_hm3[0]
                );

                // Stage 1: v_ref=400
                assert!(
                    (reference_volumes_hm3[1] - 400.0).abs() < 1e-10,
                    "stage 1 ref vol must be 400, got {}",
                    reference_volumes_hm3[1]
                );

                // Verify stage 0 coefficients using v_ref=200.
                let c_ev = 5.0_f64;
                let da_dv = 0.005_f64; // same slope in both segments

                let zeta_jan = 1.0 / (3.6 * 744.0);
                let a_jan = 1.5_f64;
                let v_ref_jan = 200.0_f64;
                let expected_k_evap_v_jan = zeta_jan * c_ev * da_dv;
                let expected_k_evap0_jan =
                    zeta_jan * c_ev * a_jan - expected_k_evap_v_jan * v_ref_jan;
                assert!(
                    (coefficients[0].k_evap_v - expected_k_evap_v_jan).abs() < 1e-10,
                    "stage 0 k_evap_v: expected {expected_k_evap_v_jan}, got {}",
                    coefficients[0].k_evap_v
                );
                assert!(
                    (coefficients[0].k_evap0 - expected_k_evap0_jan).abs() < 1e-10,
                    "stage 0 k_evap0: expected {expected_k_evap0_jan}, got {}",
                    coefficients[0].k_evap0
                );

                // Verify stage 1 coefficients using v_ref=400.
                let zeta_feb = 1.0 / (3.6 * 672.0);
                let a_feb = 2.5_f64;
                let v_ref_feb = 400.0_f64;
                let expected_k_evap_v_feb = zeta_feb * c_ev * da_dv;
                let expected_k_evap0_feb =
                    zeta_feb * c_ev * a_feb - expected_k_evap_v_feb * v_ref_feb;
                assert!(
                    (coefficients[1].k_evap_v - expected_k_evap_v_feb).abs() < 1e-10,
                    "stage 1 k_evap_v: expected {expected_k_evap_v_feb}, got {}",
                    coefficients[1].k_evap_v
                );
                assert!(
                    (coefficients[1].k_evap0 - expected_k_evap0_feb).abs() < 1e-10,
                    "stage 1 k_evap0: expected {expected_k_evap0_feb}, got {}",
                    coefficients[1].k_evap0
                );
            }
            other => panic!("expected Linearized, got {other:?}"),
        }
    }

    /// resolve_evaporation_core: None reference volumes produce DefaultMidpoint provenance and
    /// all reference_volumes_hm3 entries equal (v_min + v_max) / 2.
    #[test]
    fn resolve_evaporation_none_ref_vols_produces_default_midpoint_provenance() {
        // `make_hydro_with_evaporation` already sets evaporation_reference_volumes_hm3 = None.
        let hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some([5.0f64; 12]));

        let geo_rows = make_geo_rows(&[
            (100.0, 1.0),
            (200.0, 1.5),
            (300.0, 2.0),
            (400.0, 2.5),
            (500.0, 3.0),
        ]);
        let geo_refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
            HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        // Two stages with different months (January = 0, June = 5).
        let stage_january = make_stage_with_month(0, 0);
        let stage_june = make_stage_with_month(1, 5);
        let stage_refs = vec![&stage_january, &stage_june];

        let (models, evap_provenance, ref_provenance) =
            super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
                .expect("should succeed");

        assert_eq!(
            ref_provenance[0].1,
            EvaporationReferenceSource::DefaultMidpoint,
            "None reference volumes must produce DefaultMidpoint provenance"
        );
        assert_eq!(
            evap_provenance[0].1,
            EvaporationSource::LinearizedFromGeometry
        );

        let expected_v_ref = f64::midpoint(100.0, 500.0); // 300.0

        match models.model(0) {
            EvaporationModel::Linearized {
                reference_volumes_hm3,
                ..
            } => {
                assert_eq!(
                    reference_volumes_hm3.len(),
                    2,
                    "must have 2 ref volumes (one per stage)"
                );
                for (s, &v) in reference_volumes_hm3.iter().enumerate() {
                    assert!(
                        (v - expected_v_ref).abs() < 1e-10,
                        "stage {s} ref vol must be midpoint {expected_v_ref}, got {v}"
                    );
                }
            }
            other => panic!("expected Linearized, got {other:?}"),
        }
    }

    /// resolve_evaporation_core: mixed hydro set (one with user-supplied, one without)
    /// produces correct per-hydro provenance.
    #[test]
    fn resolve_evaporation_mixed_ref_vol_provenance() {
        let mut ref_vols = [300.0f64; 12];
        ref_vols[0] = 200.0;

        let mut hydro_with = make_hydro_with_evaporation(0, 100.0, 500.0, Some([5.0f64; 12]));
        hydro_with.evaporation_reference_volumes_hm3 = Some(ref_vols);

        let hydro_without = make_hydro_with_evaporation(1, 100.0, 500.0, Some([5.0f64; 12]));
        // hydro_without.evaporation_reference_volumes_hm3 is already None.

        let geo_rows = make_geo_rows(&[(100.0, 1.0), (300.0, 2.0), (500.0, 3.0)]);
        let refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
            HashMap::new();
        geometry_map.insert(EntityId::from(0), refs.clone());
        geometry_map.insert(EntityId::from(1), refs);

        let stage = make_stage_with_month(0, 0);
        let stage_refs = vec![&stage];

        let (_, _, ref_provenance) = super::resolve_evaporation_core(
            &[hydro_with, hydro_without],
            &geometry_map,
            &stage_refs,
        )
        .expect("should succeed");

        assert_eq!(ref_provenance.len(), 2);
        assert_eq!(
            ref_provenance[0].1,
            EvaporationReferenceSource::UserSupplied,
            "hydro with ref vols must be UserSupplied"
        );
        assert_eq!(
            ref_provenance[1].1,
            EvaporationReferenceSource::DefaultMidpoint,
            "hydro without ref vols must be DefaultMidpoint"
        );
    }
}
