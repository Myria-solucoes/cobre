//! Hydro model preprocessing pipeline for the production function and evaporation.
//!
//! Resolves per-`(hydro, stage)` production models (constant productivity or FPHA
//! hyperplanes) and per-hydro evaporation models from the case directory, bundles
//! them into a `PrepareHydroModelsResult`, and produces a display summary.
//!
//! These types and functions live in `cobre-sddp` because they are
//! algorithm-specific (FPHA hyperplane approximation is an SDDP concept). They
//! must not be placed in `cobre-core`.
//!
//! # Submodule layout
//!
//! - `types` — the runtime output types (`ResolvedProductionModel`,
//!   `ProductionModelSet`, `EvaporationModel`, `EvaporationModelSet`,
//!   `HydroModelProvenance`, `HydroModelSummary`, `PrepareHydroModelsResult`,
//!   and the source/provenance enums).
//! - `production` — per-`(hydro, stage)` production-model resolution: constant
//!   productivity, precomputed FPHA hyperplanes, and computed FPHA fitting via
//!   `crate::fpha_fitting`.
//! - `evaporation` — per-hydro linearized evaporation resolution from reservoir
//!   geometry, plus the area interpolation and derivative helpers.
//! - `summary` — the `build_hydro_model_summary` display-summary builder.
//! - `export` — the `build_evaporation_model_rows` export-row builder that maps
//!   the resolved evaporation models into `cobre_io::EvaporationModelRow`s.
//!
//! The orchestration entry points (`prepare_hydro_models`,
//! `prepare_hydro_models_from_artifacts`) and the private
//! `load_artifacts_for_hydro_models` reader live here in `mod`.
//!
//! Every public symbol is re-exported here so both the curated flat surface in
//! `lib.rs` and the `cobre_sddp::hydro_models::Symbol` module path resolve to
//! the same item regardless of which submodule owns it.

use std::path::Path;

use cobre_core::System;

use crate::SddpError;

mod evaporation;
mod export;
mod production;
mod summary;
mod types;

pub use evaporation::{resolve_evaporation_models, resolve_evaporation_models_from_artifacts};
pub use export::build_evaporation_model_rows;
pub use production::{resolve_production_models, resolve_production_models_from_artifacts};
pub use summary::build_hydro_model_summary;

/// Wall-clock split of the two hydro-model fitting steps run by
/// [`prepare_hydro_models_from_artifacts`].
///
/// Production fitting (constant-productivity/FPHA resolution) and evaporation
/// fitting are timed independently because an operator wants each surfaced on
/// its own. Populated only when a caller passes a `&mut` borrow to
/// [`prepare_hydro_models_from_artifacts`]; callers that pass `None` pay no
/// measurement and the resolver logic is identical in both cases.
#[derive(Debug, Clone, Copy, Default)]
pub struct HydroFitTimings {
    /// Wall-clock seconds spent in [`resolve_production_models_from_artifacts`].
    pub production_fit_seconds: f64,
    /// Wall-clock seconds spent in [`resolve_evaporation_models_from_artifacts`].
    pub evaporation_fit_seconds: f64,
}
pub use types::{
    EvaporationModel, EvaporationModelSet, EvaporationReferenceSource, EvaporationSource,
    FphaHydroDetail, FphaPlane, HydroModelProvenance, HydroModelSummary, LinearizedEvaporation,
    PrepareHydroModelsResult, ProductionModelSet, ProductionModelSource, ResolvedProductionModel,
};
// ── Top-level pipeline function ───────────────────────────────────────────────

/// Run the full hydro model preprocessing pipeline for a case directory.
///
/// Composes [`resolve_production_models`] and [`resolve_evaporation_models`]
/// and returns a [`PrepareHydroModelsResult`] bundling all pipeline outputs.
///
/// Called once per entry point (CLI, Python) before constructing `StudySetup`.
/// On MPI setups, this function runs on all ranks independently (each rank has
/// the system via broadcast and can load the optional files from a shared
/// filesystem).
///
/// # Errors
///
/// Propagates errors from [`resolve_production_models`] and
/// [`resolve_evaporation_models`]. See their individual documentation for the
/// full error table.
pub fn prepare_hydro_models(
    system: &System,
    case_dir: &Path,
) -> Result<PrepareHydroModelsResult, SddpError> {
    let artifacts = load_artifacts_for_hydro_models(case_dir)?;
    prepare_hydro_models_from_artifacts(system, &artifacts, None)
}

/// Variant of [`prepare_hydro_models`] that consumes a pre-parsed
/// [`cobre_io::CaseArtifacts`] bundle instead of re-reading the case
/// directory from disk.
///
/// Use this from any pipeline that has already called
/// [`cobre_io::load_case_with_artifacts`]; it avoids the duplicate parsing
/// and parallel validation paths.
///
/// When `timings` is `Some`, the wall time of each of the two fitting steps is
/// recorded into the supplied [`HydroFitTimings`]. Passing `None` skips the
/// measurement; the resolver calls and their results are identical either way.
///
/// # Errors
///
/// Same conditions as [`prepare_hydro_models`].
pub fn prepare_hydro_models_from_artifacts(
    system: &System,
    artifacts: &cobre_io::CaseArtifacts,
    timings: Option<&mut HydroFitTimings>,
) -> Result<PrepareHydroModelsResult, SddpError> {
    let production_start = std::time::Instant::now();
    let (
        production,
        productivity_override,
        production_sources,
        fpha_export_rows,
        reference_volumes_hm3,
    ) = resolve_production_models_from_artifacts(system, artifacts)?;
    let production_fit_seconds = production_start.elapsed().as_secs_f64();

    let evaporation_start = std::time::Instant::now();
    let (evaporation, evaporation_sources, evaporation_reference_sources) =
        resolve_evaporation_models_from_artifacts(system, artifacts)?;
    let evaporation_fit_seconds = evaporation_start.elapsed().as_secs_f64();

    if let Some(timings) = timings {
        timings.production_fit_seconds = production_fit_seconds;
        timings.evaporation_fit_seconds = evaporation_fit_seconds;
    }

    // Group the VHA geometry by plant (sorted by ascending volume, as
    // `ForebayTable::new` expects) so the energy-conversion build can derive ρ_eq
    // from VHA + ρ_esp for any FPHA plant lacking a parquet override. Cloned out
    // of `artifacts`, which the caller drops once preprocessing returns.
    let mut vha_geometry_by_hydro: std::collections::HashMap<
        cobre_core::EntityId,
        Vec<cobre_io::HydroGeometryRow>,
    > = std::collections::HashMap::new();
    for row in &artifacts.hydro_geometry {
        vha_geometry_by_hydro
            .entry(row.hydro_id)
            .or_default()
            .push(row.clone());
    }
    for rows in vha_geometry_by_hydro.values_mut() {
        rows.sort_by(|a, b| a.volume_hm3.total_cmp(&b.volume_hm3));
    }

    Ok(PrepareHydroModelsResult {
        production,
        productivity_override,
        evaporation,
        provenance: HydroModelProvenance {
            production_sources,
            evaporation_sources,
            evaporation_reference_sources,
        },
        fpha_export_rows,
        reference_volumes_hm3,
        vha_geometry_by_hydro,
    })
}

/// Build a [`cobre_io::CaseArtifacts`] containing the rows
/// [`prepare_hydro_models_from_artifacts`] needs, by reading the case
/// directory directly.
///
/// Used to back the legacy [`prepare_hydro_models`] signature; production
/// pipelines should call [`cobre_io::load_case_with_artifacts`] instead so
/// the full validation runs once.
fn load_artifacts_for_hydro_models(case_dir: &Path) -> Result<cobre_io::CaseArtifacts, SddpError> {
    let mut ctx = cobre_io::ValidationContext::new();
    let manifest = cobre_io::validate_structure(case_dir, &mut ctx);
    // Propagate structural validation errors before attempting any file loads;
    // a malformed layout must fail here rather than surface as a confusing
    // parse error (or silent default) downstream.
    ctx.into_result().map_err(SddpError::from)?;

    let prod_path = if manifest.system_hydro_production_models_json {
        Some(case_dir.join("system").join("hydro_production_models.json"))
    } else {
        None
    };
    let geom_path = if manifest.system_hydro_geometry_parquet {
        Some(case_dir.join("system").join("hydro_geometry.parquet"))
    } else {
        None
    };
    let fpha_path = if manifest.system_fpha_hyperplanes_parquet {
        Some(case_dir.join("system").join("fpha_hyperplanes.parquet"))
    } else {
        None
    };
    let prod_eff_path = case_dir
        .join("system")
        .join("hydro_energy_productivity.parquet");
    let prod_eff_path_opt = if prod_eff_path.exists() {
        Some(prod_eff_path.as_path())
    } else {
        None
    };
    let tailrace_path = if manifest.system_tailrace_curves_parquet {
        Some(case_dir.join("system").join("tailrace_curves.parquet"))
    } else {
        None
    };

    let production_file = cobre_io::extensions::load_production_models(prod_path.as_deref())?;

    Ok(cobre_io::CaseArtifacts {
        file_manifest: manifest,
        hydro_geometry: cobre_io::extensions::load_hydro_geometry(geom_path.as_deref())?,
        production_models: production_file.configs,
        plane_reduction: production_file.plane_reduction,
        hydro_energy_productivity: cobre_io::load_hydro_energy_productivity(prod_eff_path_opt)?,
        fpha_hyperplanes: cobre_io::extensions::load_fpha_hyperplanes(fpha_path.as_deref())?,
        scalar_parameters: Vec::new(),
        tailrace_curves: cobre_io::extensions::load_tailrace_curves(tailrace_path.as_deref())?,
    })
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
    // ── 2-rank parity test ────────────────────────────────────────────────────

    /// Simulates two independent MPI ranks both calling `prepare_hydro_models` on
    /// a computed-FPHA case (d07-fpha-computed). Asserts that `fpha_export_rows` is
    /// non-empty and bit-identical between the two calls, confirming that the
    /// preprocessing is deterministic and rank-independent.
    ///
    /// No real MPI is used. The test simply calls `prepare_hydro_models` twice from
    /// the same source data and compares the results.
    #[test]
    fn prepare_hydro_models_fpha_export_rows_are_identical_across_ranks() {
        let case_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("cobre-sddp parent dir must exist")
            .parent()
            .expect("crates parent dir must exist")
            .join("examples/deterministic/d07-fpha-computed");

        let system =
            cobre_io::load_case(&case_dir).expect("d07-fpha-computed must load successfully");

        // Simulated rank 0: call prepare_hydro_models and capture rows.
        let result_rank0 = super::prepare_hydro_models(&system, &case_dir)
            .expect("prepare_hydro_models must succeed for rank 0");

        // Simulated rank 1: independent call with the same inputs.
        let result_rank1 = super::prepare_hydro_models(&system, &case_dir)
            .expect("prepare_hydro_models must succeed for rank 1");

        // Post-condition: computed-FPHA rows must be present.
        assert!(
            !result_rank0.fpha_export_rows.is_empty(),
            "rank 0: fpha_export_rows must be non-empty for a computed-FPHA case"
        );

        // Parity: both ranks must produce bit-identical rows.
        assert_eq!(
            result_rank0.fpha_export_rows, result_rank1.fpha_export_rows,
            "fpha_export_rows must be bit-identical across ranks (deterministic preprocessing)"
        );
    }

    /// `prepare_hydro_models` carries the per-hydro VHA geometry on the result,
    /// grouped by plant and sorted by ascending volume, so the energy-conversion
    /// build can derive ρ_eq from VHA + ρ_esp for an FPHA plant with no parquet
    /// override. A computed-FPHA case ships geometry, so the map must be populated.
    #[test]
    fn prepare_hydro_models_carries_sorted_vha_geometry() {
        let case_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("cobre-sddp parent dir must exist")
            .parent()
            .expect("crates parent dir must exist")
            .join("examples/deterministic/d07-fpha-computed");

        let system =
            cobre_io::load_case(&case_dir).expect("d07-fpha-computed must load successfully");
        let result = super::prepare_hydro_models(&system, &case_dir)
            .expect("prepare_hydro_models must succeed");

        assert!(
            !result.vha_geometry_by_hydro.is_empty(),
            "a computed-FPHA case ships VHA geometry; the map must be populated"
        );
        for rows in result.vha_geometry_by_hydro.values() {
            assert!(!rows.is_empty(), "no plant entry may be empty");
            assert!(
                rows.windows(2).all(|w| w[0].volume_hm3 <= w[1].volume_hm3),
                "each plant's VHA rows must be sorted by ascending volume"
            );
        }
    }

    // ── thread-count determinism gate ─────────────────────────────────────────

    /// The per-hydro FPHA fit loop in `resolve_production_models_from_artifacts`
    /// is a `par_iter().collect()`, so its output must not depend on the rayon
    /// pool size. This gate fits the computed-FPHA case d07-fpha-computed under
    /// pools of 1, 2, and 4 threads and asserts the resolved `fpha_export_rows`
    /// are `to_bits`-identical across all three — the `(hydro_id, stage_id,
    /// plane_id)` ordering and every gamma coefficient must match bit-for-bit
    /// regardless of thread scheduling. A regression that collected into a shared
    /// `Mutex<Vec>` or pushed rows from worker threads would reorder the stream
    /// and fail here.
    #[test]
    fn fit_is_thread_count_invariant() {
        let case_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("cobre-sddp parent dir must exist")
            .parent()
            .expect("crates parent dir must exist")
            .join("examples/deterministic/d07-fpha-computed");

        let system =
            cobre_io::load_case(&case_dir).expect("d07-fpha-computed must load successfully");

        // Resolve the FPHA export rows inside a fixed-size rayon pool so the fit
        // loop runs under exactly `n` worker threads.
        let resolve_under_pool = |n: usize| -> Vec<cobre_io::FphaHyperplaneRow> {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("rayon pool must build")
                .install(|| {
                    super::prepare_hydro_models(&system, &case_dir)
                        .expect("prepare_hydro_models must succeed")
                        .fpha_export_rows
                })
        };

        let thread_counts = [1usize, 2, 4];
        let rows: Vec<Vec<cobre_io::FphaHyperplaneRow>> = thread_counts
            .iter()
            .map(|&n| resolve_under_pool(n))
            .collect();

        assert!(
            !rows[0].is_empty(),
            "the computed-FPHA case must export at least one plane for the fit path to run"
        );

        // Bit-exact equality across pool sizes: ids, stage/plane ordering, and the
        // gamma coefficients (compared via `to_bits`, not float `==`).
        let assert_bit_identical = |a: &[cobre_io::FphaHyperplaneRow],
                                    b: &[cobre_io::FphaHyperplaneRow],
                                    threads_a: usize,
                                    threads_b: usize| {
            assert_eq!(
                a.len(),
                b.len(),
                "row count must match across {threads_a}- and {threads_b}-thread pools"
            );
            for (ra, rb) in a.iter().zip(b) {
                assert_eq!(ra.hydro_id, rb.hydro_id, "hydro_id must match");
                assert_eq!(ra.stage_id, rb.stage_id, "stage_id must match");
                assert_eq!(ra.plane_id, rb.plane_id, "plane_id must match");
                assert_eq!(
                    ra.gamma_0.to_bits(),
                    rb.gamma_0.to_bits(),
                    "gamma_0 must be bit-identical across pool sizes"
                );
                assert_eq!(
                    ra.gamma_v.to_bits(),
                    rb.gamma_v.to_bits(),
                    "gamma_v must be bit-identical across pool sizes"
                );
                assert_eq!(
                    ra.gamma_q.to_bits(),
                    rb.gamma_q.to_bits(),
                    "gamma_q must be bit-identical across pool sizes"
                );
                assert_eq!(
                    ra.gamma_s.to_bits(),
                    rb.gamma_s.to_bits(),
                    "gamma_s must be bit-identical across pool sizes"
                );
                assert_eq!(
                    ra.kappa.to_bits(),
                    rb.kappa.to_bits(),
                    "kappa must be bit-identical across pool sizes"
                );
            }
        };

        // Compare every pool size against the single-thread baseline.
        for (rows_n, &n) in rows.iter().zip(&thread_counts).skip(1) {
            assert_bit_identical(&rows[0], rows_n, thread_counts[0], n);
        }
    }
}
