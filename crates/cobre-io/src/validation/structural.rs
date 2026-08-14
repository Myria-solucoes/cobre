//! Layer 1 — Structural validation.
//!
//! Checks that required files exist in the case directory and records whether
//! optional files are present.  This layer does **not** parse any file content;
//! it only tests for the existence of paths on disk. It also rejects any file
//! whose input contract has been withdrawn but is still present
//! ([`ErrorKind::BusinessRuleViolation`]).
//!
//! Call [`validate_structure`] with a path to the case root and a mutable
//! [`ValidationContext`].  It returns a [`FileManifest`] with one field per
//! `FILE_ENTRIES` row, zipped positionally.  Missing required files produce
//! [`ErrorKind::FileNotFound`] entries in the context.  Missing optional files
//! leave the corresponding manifest field `false` without adding any error.
//!
//! # Examples
//!
//! ```no_run
//! use std::path::Path;
//! use cobre_io::validation::{ValidationContext, structural::validate_structure};
//!
//! let mut ctx = ValidationContext::new();
//! let manifest = validate_structure(Path::new("/path/to/case"), &mut ctx);
//! assert!(!ctx.has_errors());
//! assert!(manifest.config_json);
//! ```

use std::path::Path;

use super::{ErrorKind, ValidationContext};

// ── FileManifest ─────────────────────────────────────────────────────────────

/// Records whether each input file (one field per `FILE_ENTRIES` row) is
/// present in the case directory.
///
/// Fields default to `false`; [`validate_structure`] sets each to `true` if the
/// corresponding file was found on disk.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Default)]
pub struct FileManifest {
    /// `config.json` — required
    pub config_json: bool,
    /// `penalties.json` — required
    pub penalties_json: bool,
    /// `stages.json` — required
    pub stages_json: bool,
    /// `initial_conditions.json` — required
    pub initial_conditions_json: bool,
    /// `post_study_stages.json` — optional
    pub post_study_stages_json: bool,

    /// `system/buses.json` — required
    pub system_buses_json: bool,
    /// `system/lines.json` — required
    pub system_lines_json: bool,
    /// `system/hydros.json` — required
    pub system_hydros_json: bool,
    /// `system/thermals.json` — required
    pub system_thermals_json: bool,
    /// `system/non_controllable_sources.json` — optional
    pub system_non_controllable_sources_json: bool,
    /// `system/pumping_stations.json` — optional
    pub system_pumping_stations_json: bool,
    /// `system/energy_contracts.json` — optional
    pub system_energy_contracts_json: bool,
    /// `system/hydro_geometry.parquet` — optional
    pub system_hydro_geometry_parquet: bool,
    /// `system/hydro_production_models.json` — optional
    pub system_hydro_production_models_json: bool,
    /// `system/fpha_hyperplanes.parquet` — optional
    pub system_fpha_hyperplanes_parquet: bool,
    /// `system/hydro_energy_productivity.parquet` — optional
    pub system_hydro_energy_productivity_parquet: bool,
    /// `system/tailrace_curves.parquet` — optional
    pub system_tailrace_curves_parquet: bool,

    /// `scenarios/inflow_history.parquet` — optional
    pub scenarios_inflow_history_parquet: bool,
    /// `scenarios/inflow_seasonal_stats.parquet` — optional
    pub scenarios_inflow_seasonal_stats_parquet: bool,
    /// `scenarios/inflow_ar_coefficients.parquet` — optional
    pub scenarios_inflow_ar_coefficients_parquet: bool,
    /// `scenarios/inflow_annual_component.parquet` — optional
    pub scenarios_inflow_annual_component_parquet: bool,
    /// `scenarios/external_inflow_scenarios.parquet` — optional
    pub scenarios_external_inflow_scenarios_parquet: bool,
    /// `scenarios/external_load_scenarios.parquet` — optional
    pub scenarios_external_load_scenarios_parquet: bool,
    /// `scenarios/external_ncs_scenarios.parquet` — optional
    pub scenarios_external_ncs_scenarios_parquet: bool,
    /// `scenarios/load_seasonal_stats.parquet` — optional
    pub scenarios_load_seasonal_stats_parquet: bool,
    /// `scenarios/load_factors.json` — optional
    pub scenarios_load_factors_json: bool,
    /// `scenarios/correlation.json` — optional
    pub scenarios_correlation_json: bool,
    /// `scenarios/non_controllable_factors.json` — optional
    pub scenarios_non_controllable_factors_json: bool,
    /// `scenarios/non_controllable_stats.parquet` — optional
    pub scenarios_non_controllable_stats_parquet: bool,

    /// `constraints/thermal_bounds.parquet` — optional
    pub constraints_thermal_bounds_parquet: bool,
    /// `constraints/hydro_bounds.parquet` — optional
    pub constraints_hydro_bounds_parquet: bool,
    /// `constraints/line_bounds.parquet` — optional
    pub constraints_line_bounds_parquet: bool,
    /// `constraints/pumping_bounds.parquet` — optional
    pub constraints_pumping_bounds_parquet: bool,
    /// `constraints/contract_bounds.parquet` — optional
    pub constraints_contract_bounds_parquet: bool,
    /// `constraints/generic_constraints.json` — optional
    pub constraints_generic_constraints_json: bool,
    /// `constraints/generic_constraint_bounds.parquet` — optional
    pub constraints_generic_constraint_bounds_parquet: bool,
    /// `constraints/generic_parameters.json` — optional
    pub constraints_generic_parameters_json: bool,
    /// `constraints/penalty_overrides_bus.parquet` — optional
    pub constraints_penalty_overrides_bus_parquet: bool,
    /// `constraints/penalty_overrides_line.parquet` — optional
    pub constraints_penalty_overrides_line_parquet: bool,
    /// `constraints/penalty_overrides_hydro.parquet` — optional
    pub constraints_penalty_overrides_hydro_parquet: bool,
    /// `constraints/penalty_overrides_ncs.parquet` — optional
    pub constraints_penalty_overrides_ncs_parquet: bool,
    /// `constraints/ncs_bounds.parquet` — optional
    pub constraints_ncs_bounds_parquet: bool,
    /// `constraints/hydro_unit_group_bounds.parquet` — optional
    pub constraints_hydro_unit_group_bounds_parquet: bool,
}

// ── validate_structure ────────────────────────────────────────────────────────

/// Describes a single file entry for the structural check.
struct FileEntry {
    /// Path relative to the case root.
    relative: &'static str,
    required: bool,
}

/// Every input file in canonical order — one row per [`FileManifest`] field,
/// zipped positionally by [`manifest_fields_mut`].
const FILE_ENTRIES: &[FileEntry] = &[
    // Root-level — required
    FileEntry {
        relative: "config.json",
        required: true,
    },
    FileEntry {
        relative: "penalties.json",
        required: true,
    },
    FileEntry {
        relative: "stages.json",
        required: true,
    },
    FileEntry {
        relative: "initial_conditions.json",
        required: true,
    },
    // Root-level — optional
    FileEntry {
        relative: "post_study_stages.json",
        required: false,
    },
    // system/ — required
    FileEntry {
        relative: "system/buses.json",
        required: true,
    },
    FileEntry {
        relative: "system/lines.json",
        required: true,
    },
    FileEntry {
        relative: "system/hydros.json",
        required: true,
    },
    FileEntry {
        relative: "system/thermals.json",
        required: true,
    },
    // system/ — optional
    FileEntry {
        relative: "system/non_controllable_sources.json",
        required: false,
    },
    FileEntry {
        relative: "system/pumping_stations.json",
        required: false,
    },
    FileEntry {
        relative: "system/energy_contracts.json",
        required: false,
    },
    FileEntry {
        relative: "system/hydro_geometry.parquet",
        required: false,
    },
    FileEntry {
        relative: "system/hydro_production_models.json",
        required: false,
    },
    FileEntry {
        relative: "system/fpha_hyperplanes.parquet",
        required: false,
    },
    FileEntry {
        relative: "system/hydro_energy_productivity.parquet",
        required: false,
    },
    FileEntry {
        relative: "system/tailrace_curves.parquet",
        required: false,
    },
    // scenarios/ — optional
    FileEntry {
        relative: "scenarios/inflow_history.parquet",
        required: false,
    },
    FileEntry {
        relative: "scenarios/inflow_seasonal_stats.parquet",
        required: false,
    },
    FileEntry {
        relative: "scenarios/inflow_ar_coefficients.parquet",
        required: false,
    },
    FileEntry {
        relative: "scenarios/inflow_annual_component.parquet",
        required: false,
    },
    FileEntry {
        relative: "scenarios/external_inflow_scenarios.parquet",
        required: false,
    },
    FileEntry {
        relative: "scenarios/external_load_scenarios.parquet",
        required: false,
    },
    FileEntry {
        relative: "scenarios/external_ncs_scenarios.parquet",
        required: false,
    },
    FileEntry {
        relative: "scenarios/load_seasonal_stats.parquet",
        required: false,
    },
    FileEntry {
        relative: "scenarios/load_factors.json",
        required: false,
    },
    FileEntry {
        relative: "scenarios/correlation.json",
        required: false,
    },
    FileEntry {
        relative: "scenarios/non_controllable_factors.json",
        required: false,
    },
    FileEntry {
        relative: "scenarios/non_controllable_stats.parquet",
        required: false,
    },
    // constraints/ — optional
    FileEntry {
        relative: "constraints/thermal_bounds.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/hydro_bounds.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/line_bounds.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/pumping_bounds.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/contract_bounds.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/generic_constraints.json",
        required: false,
    },
    FileEntry {
        relative: "constraints/generic_constraint_bounds.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/generic_parameters.json",
        required: false,
    },
    FileEntry {
        relative: "constraints/penalty_overrides_bus.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/penalty_overrides_line.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/penalty_overrides_hydro.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/penalty_overrides_ncs.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/ncs_bounds.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/hydro_unit_group_bounds.parquet",
        required: false,
    },
];

/// Describes an input file no longer read; `replacement` names its migration
/// in the rejection message.
struct RemovedFile {
    /// Path relative to the case root.
    relative: &'static str,
    replacement: &'static str,
}

/// An input file no longer read by any loader is rejected when present —
/// never accepted and silently ignored.
const REMOVED_FILES: &[RemovedFile] = &[
    RemovedFile {
        relative: "constraints/exchange_factors.json",
        replacement: "per-block line capacity is now declared as absolute MW in \
            constraints/line_bounds.parquet via direct_mw / reverse_mw rows \
            carrying a block_id (use base_capacity * factor for each block). \
            Remove the file.",
    },
    RemovedFile {
        relative: "system/scalar_parameters.json",
        replacement: "scalar parameters are now read from \
            constraints/generic_parameters.json, beside the constraints that \
            reference them by @name. Move the file (its contents are unchanged).",
    },
];

/// Performs Layer 1 structural validation on the case directory at `case_root`,
/// returning a [`FileManifest`] of which files are present.
///
/// A present file sets its manifest field to `true`. An absent **required** file
/// adds an [`ErrorKind::FileNotFound`] error; an absent **optional** file leaves
/// its field `false` with no error. A present file listed in `REMOVED_FILES`
/// adds an [`ErrorKind::BusinessRuleViolation`] error naming its replacement.
/// This function does **not** read or parse any file content.
#[must_use]
pub fn validate_structure(case_root: &Path, ctx: &mut ValidationContext) -> FileManifest {
    let mut manifest = FileManifest::default();

    for removed in REMOVED_FILES {
        if case_root.join(removed.relative).exists() {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                removed.relative,
                None::<&str>,
                format!(
                    "{} is no longer read; {}",
                    removed.relative, removed.replacement
                ),
            );
        }
    }

    for (entry, present) in FILE_ENTRIES.iter().zip(manifest_fields_mut(&mut manifest)) {
        if case_root.join(entry.relative).exists() {
            *present = true;
        } else if entry.required {
            ctx.add_error(
                ErrorKind::FileNotFound,
                entry.relative,
                None::<&str>,
                format!(
                    "required file '{}' not found in case directory",
                    entry.relative
                ),
            );
        }
    }

    manifest
}

/// Returns mutable references to every `bool` field of [`FileManifest`] in the
/// same order as [`FILE_ENTRIES`] — `validate_structure` zips the two positionally,
/// so a divergence here silently misassigns presence flags.
fn manifest_fields_mut(m: &mut FileManifest) -> [&mut bool; 43] {
    [
        // Root
        &mut m.config_json,
        &mut m.penalties_json,
        &mut m.stages_json,
        &mut m.initial_conditions_json,
        &mut m.post_study_stages_json,
        // system/ required
        &mut m.system_buses_json,
        &mut m.system_lines_json,
        &mut m.system_hydros_json,
        &mut m.system_thermals_json,
        // system/ optional
        &mut m.system_non_controllable_sources_json,
        &mut m.system_pumping_stations_json,
        &mut m.system_energy_contracts_json,
        &mut m.system_hydro_geometry_parquet,
        &mut m.system_hydro_production_models_json,
        &mut m.system_fpha_hyperplanes_parquet,
        &mut m.system_hydro_energy_productivity_parquet,
        &mut m.system_tailrace_curves_parquet,
        // scenarios/
        &mut m.scenarios_inflow_history_parquet,
        &mut m.scenarios_inflow_seasonal_stats_parquet,
        &mut m.scenarios_inflow_ar_coefficients_parquet,
        &mut m.scenarios_inflow_annual_component_parquet,
        &mut m.scenarios_external_inflow_scenarios_parquet,
        &mut m.scenarios_external_load_scenarios_parquet,
        &mut m.scenarios_external_ncs_scenarios_parquet,
        &mut m.scenarios_load_seasonal_stats_parquet,
        &mut m.scenarios_load_factors_json,
        &mut m.scenarios_correlation_json,
        &mut m.scenarios_non_controllable_factors_json,
        &mut m.scenarios_non_controllable_stats_parquet,
        // constraints/
        &mut m.constraints_thermal_bounds_parquet,
        &mut m.constraints_hydro_bounds_parquet,
        &mut m.constraints_line_bounds_parquet,
        &mut m.constraints_pumping_bounds_parquet,
        &mut m.constraints_contract_bounds_parquet,
        &mut m.constraints_generic_constraints_json,
        &mut m.constraints_generic_constraint_bounds_parquet,
        &mut m.constraints_generic_parameters_json,
        &mut m.constraints_penalty_overrides_bus_parquet,
        &mut m.constraints_penalty_overrides_line_parquet,
        &mut m.constraints_penalty_overrides_hydro_parquet,
        &mut m.constraints_penalty_overrides_ncs_parquet,
        &mut m.constraints_ncs_bounds_parquet,
        &mut m.constraints_hydro_unit_group_bounds_parquet,
    ]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Create a temporary case directory containing all 8 required files.
    fn make_case_with_required(dir: &TempDir) {
        let root = dir.path();
        fs::create_dir_all(root.join("system")).unwrap();
        fs::write(root.join("config.json"), b"{}").unwrap();
        fs::write(root.join("penalties.json"), b"{}").unwrap();
        fs::write(root.join("stages.json"), b"{}").unwrap();
        fs::write(root.join("initial_conditions.json"), b"{}").unwrap();
        fs::write(root.join("system/buses.json"), b"{}").unwrap();
        fs::write(root.join("system/lines.json"), b"{}").unwrap();
        fs::write(root.join("system/hydros.json"), b"{}").unwrap();
        fs::write(root.join("system/thermals.json"), b"{}").unwrap();
    }

    #[test]
    fn test_structural_all_required_present() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "should have 0 errors when all required files present, got: {:?}",
            ctx.errors()
        );

        assert!(manifest.config_json, "config.json should be present");
        assert!(manifest.penalties_json, "penalties.json should be present");
        assert!(manifest.stages_json, "stages.json should be present");
        assert!(
            manifest.initial_conditions_json,
            "initial_conditions.json should be present"
        );
        assert!(
            manifest.system_buses_json,
            "system/buses.json should be present"
        );
        assert!(
            manifest.system_lines_json,
            "system/lines.json should be present"
        );
        assert!(
            manifest.system_hydros_json,
            "system/hydros.json should be present"
        );
        assert!(
            manifest.system_thermals_json,
            "system/thermals.json should be present"
        );
    }

    #[test]
    fn test_structural_missing_required_hydros() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        fs::remove_file(dir.path().join("system/hydros.json")).unwrap();

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            ctx.has_errors(),
            "should have at least 1 error when system/hydros.json is missing"
        );
        assert_eq!(ctx.errors().len(), 1, "should have exactly 1 error");
        let entry = &ctx.errors()[0];
        assert_eq!(
            entry.kind,
            ErrorKind::FileNotFound,
            "error kind should be FileNotFound"
        );
        assert!(
            entry.file.to_string_lossy().contains("hydros.json"),
            "error file should reference hydros.json, got: {}",
            entry.file.display()
        );
        assert!(
            !manifest.system_hydros_json,
            "manifest.system_hydros_json should be false"
        );
    }

    #[test]
    fn test_structural_optional_absent_no_error() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        // No optional files are created

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "absent optional files should not produce errors"
        );

        // Verify representative optional files are false
        assert!(!manifest.system_non_controllable_sources_json);
        assert!(!manifest.system_hydro_geometry_parquet);
        assert!(!manifest.scenarios_inflow_history_parquet);
        assert!(!manifest.constraints_thermal_bounds_parquet);
        assert!(!manifest.scenarios_correlation_json);
    }

    #[test]
    fn test_structural_optional_present_in_manifest() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        let scenarios_dir = dir.path().join("scenarios");
        fs::create_dir_all(&scenarios_dir).unwrap();
        fs::write(scenarios_dir.join("correlation.json"), b"{}").unwrap();

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(!ctx.has_errors());
        assert!(
            manifest.scenarios_correlation_json,
            "present optional file should be marked true in manifest"
        );
    }

    #[test]
    fn test_structural_multiple_missing_required() {
        let dir = TempDir::new().unwrap();
        // Create only the system/ subdirectory but no files
        fs::create_dir_all(dir.path().join("system")).unwrap();

        let mut ctx = ValidationContext::new();
        let _manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            ctx.has_errors(),
            "should have errors for all 8 missing required files"
        );
        assert_eq!(
            ctx.errors().len(),
            8,
            "should have exactly 8 errors (one per required file), got: {}",
            ctx.errors().len()
        );
        for entry in ctx.errors() {
            assert_eq!(
                entry.kind,
                ErrorKind::FileNotFound,
                "all errors should be FileNotFound"
            );
        }
    }

    #[test]
    fn test_structural_manifest_fields_count() {
        // validate_structure zips FILE_ENTRIES with manifest_fields_mut positionally;
        // the invariant is that the two stay the same length, not any specific count.
        let mut manifest = FileManifest::default();
        let fields = manifest_fields_mut(&mut manifest);
        assert_eq!(
            FILE_ENTRIES.len(),
            fields.len(),
            "FILE_ENTRIES and manifest_fields_mut must return the same length"
        );
    }

    #[test]
    fn test_scalar_parameters_json_and_hydro_energy_productivity_present() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        fs::create_dir_all(dir.path().join("constraints")).unwrap();
        fs::write(
            dir.path().join("constraints/generic_parameters.json"),
            b"{\"scalar_parameters\":[]}",
        )
        .unwrap();
        fs::write(
            dir.path().join("system/hydro_energy_productivity.parquet"),
            b"",
        )
        .unwrap();

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "no errors expected when all required files present"
        );
        assert!(
            manifest.constraints_generic_parameters_json,
            "constraints_generic_parameters_json should be true"
        );
        assert!(
            manifest.system_hydro_energy_productivity_parquet,
            "system_hydro_energy_productivity_parquet should be true"
        );
    }

    #[test]
    fn test_scalar_parameters_json_and_hydro_energy_productivity_absent() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        // The optional files are deliberately not created

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "absent optional files must not produce errors"
        );
        assert!(
            !manifest.constraints_generic_parameters_json,
            "constraints_generic_parameters_json should be false when file is absent"
        );
        assert!(
            !manifest.system_hydro_energy_productivity_parquet,
            "system_hydro_energy_productivity_parquet should be false when file is absent"
        );
    }

    /// AC: a case directory containing `constraints/generic_parameters.json` must set
    /// `manifest.constraints_generic_parameters_json == true`.
    #[test]
    fn manifest_detects_scalar_parameters_json_when_present() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        fs::create_dir_all(dir.path().join("constraints")).unwrap();
        fs::write(
            dir.path().join("constraints/generic_parameters.json"),
            b"{\"scalar_parameters\":[]}",
        )
        .unwrap();

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "no errors expected when all required files are present"
        );
        assert!(
            manifest.constraints_generic_parameters_json,
            "constraints_generic_parameters_json must be true when constraints/generic_parameters.json exists"
        );
    }

    /// AC: a case directory with no scalar parameter file must report
    /// `manifest.constraints_generic_parameters_json == false` without producing an error.
    #[test]
    fn manifest_reports_absent_when_no_parameter_file() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        // constraints/generic_parameters.json is deliberately absent.

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "absent optional file must not produce errors"
        );
        assert!(
            !manifest.constraints_generic_parameters_json,
            "constraints_generic_parameters_json must be false when constraints/generic_parameters.json is absent"
        );
    }

    #[test]
    fn test_manifest_tailrace_curves_present() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        fs::write(dir.path().join("system/tailrace_curves.parquet"), b"").unwrap();

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "present optional file should not produce errors"
        );
        assert!(
            manifest.system_tailrace_curves_parquet,
            "system_tailrace_curves_parquet should be true when file is present"
        );
    }

    #[test]
    fn test_manifest_tailrace_curves_absent() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        // system/tailrace_curves.parquet deliberately absent.

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "absent optional file should not produce errors"
        );
        assert!(
            !manifest.system_tailrace_curves_parquet,
            "system_tailrace_curves_parquet should be false when file is absent"
        );
    }

    /// AC #5: `scenarios/inflow_annual_component.parquet` present → manifest flag `true`.
    #[test]
    fn test_manifest_detects_inflow_annual_component_present() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        let scenarios_dir = dir.path().join("scenarios");
        fs::create_dir_all(&scenarios_dir).unwrap();
        fs::write(scenarios_dir.join("inflow_annual_component.parquet"), b"").unwrap();

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "present optional file should not produce errors"
        );
        assert!(
            manifest.scenarios_inflow_annual_component_parquet,
            "scenarios_inflow_annual_component_parquet should be true when file is present"
        );
    }

    /// AC #6: `scenarios/inflow_annual_component.parquet` absent → manifest flag `false`, no error.
    #[test]
    fn test_manifest_inflow_annual_component_absent() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        // Do not create the optional annual component file.

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "absent optional file should not produce errors"
        );
        assert!(
            !manifest.scenarios_inflow_annual_component_parquet,
            "scenarios_inflow_annual_component_parquet should be false when file is absent"
        );
    }

    /// AC: `constraints/hydro_unit_group_bounds.parquet` present -> manifest flag
    /// `true`; absent (a separate case directory) -> `false`, no error either way.
    #[test]
    fn test_manifest_hydro_unit_group_bounds() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        let constraints_dir = dir.path().join("constraints");
        fs::create_dir_all(&constraints_dir).unwrap();
        fs::write(constraints_dir.join("hydro_unit_group_bounds.parquet"), b"").unwrap();

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "present optional file should not produce errors"
        );
        assert!(
            manifest.constraints_hydro_unit_group_bounds_parquet,
            "constraints_hydro_unit_group_bounds_parquet should be true when file is present"
        );

        let absent_dir = TempDir::new().unwrap();
        make_case_with_required(&absent_dir);

        let mut absent_ctx = ValidationContext::new();
        let absent_manifest = validate_structure(absent_dir.path(), &mut absent_ctx);

        assert!(
            !absent_ctx.has_errors(),
            "absent optional file should not produce errors"
        );
        assert!(
            !absent_manifest.constraints_hydro_unit_group_bounds_parquet,
            "constraints_hydro_unit_group_bounds_parquet should be false when file is absent"
        );
    }

    #[test]
    fn removed_exchange_factors_file_is_rejected() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        let constraints_dir = dir.path().join("constraints");
        fs::create_dir_all(&constraints_dir).unwrap();
        // An empty factor list, not garbage bytes: Layer 1 rejects on presence
        // alone and never parses, so even a well-formed empty file is refused.
        fs::write(
            constraints_dir.join("exchange_factors.json"),
            b"{\"exchange_factors\": []}",
        )
        .unwrap();

        let mut ctx = ValidationContext::new();
        let _manifest = validate_structure(dir.path(), &mut ctx);

        assert_eq!(
            ctx.errors().len(),
            1,
            "should have exactly 1 error when constraints/exchange_factors.json is present, got: {:?}",
            ctx.errors()
        );
        let entry = ctx.errors()[0];
        assert_eq!(
            entry.kind,
            ErrorKind::BusinessRuleViolation,
            "removed-file rejection should carry BusinessRuleViolation"
        );
        assert!(
            entry
                .file
                .to_string_lossy()
                .contains("constraints/exchange_factors.json"),
            "error file should reference constraints/exchange_factors.json, got: {}",
            entry.file.display()
        );
        assert!(
            entry.message.contains("constraints/exchange_factors.json"),
            "message should name the removed file, got: {}",
            entry.message
        );
        assert!(
            entry.message.contains("constraints/line_bounds.parquet"),
            "message should name the replacement file, got: {}",
            entry.message
        );
    }

    #[test]
    fn removed_scalar_parameters_file_is_rejected() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        // The relocated parameters file at its old system/ path: Layer 1 rejects
        // on presence alone and never parses, so a well-formed empty list is
        // still refused rather than silently read.
        fs::write(
            dir.path().join("system/scalar_parameters.json"),
            b"{\"scalar_parameters\": []}",
        )
        .unwrap();

        let mut ctx = ValidationContext::new();
        let _manifest = validate_structure(dir.path(), &mut ctx);

        assert_eq!(
            ctx.errors().len(),
            1,
            "should have exactly 1 error when system/scalar_parameters.json is present, got: {:?}",
            ctx.errors()
        );
        let entry = ctx.errors()[0];
        assert_eq!(
            entry.kind,
            ErrorKind::BusinessRuleViolation,
            "removed-file rejection should carry BusinessRuleViolation"
        );
        assert!(
            entry
                .file
                .to_string_lossy()
                .contains("system/scalar_parameters.json"),
            "error file should reference the withdrawn path, got: {}",
            entry.file.display()
        );
        assert!(
            entry
                .message
                .contains("constraints/generic_parameters.json"),
            "message should name the new path so the old file is rejected loudly, got: {}",
            entry.message
        );
    }

    #[test]
    fn absent_removed_file_produces_no_finding() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        // constraints/exchange_factors.json deliberately absent.

        let mut ctx = ValidationContext::new();
        let _manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "absent removed file should not produce any error, got: {:?}",
            ctx.errors()
        );
    }

    #[test]
    fn removed_file_check_does_not_disturb_manifest_flags() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        let constraints_dir = dir.path().join("constraints");
        fs::create_dir_all(&constraints_dir).unwrap();
        fs::write(constraints_dir.join("exchange_factors.json"), b"{}").unwrap();
        fs::write(constraints_dir.join("line_bounds.parquet"), b"").unwrap();
        fs::write(constraints_dir.join("hydro_bounds.parquet"), b"").unwrap();
        let scenarios_dir = dir.path().join("scenarios");
        fs::create_dir_all(&scenarios_dir).unwrap();
        fs::write(scenarios_dir.join("load_factors.json"), b"{}").unwrap();

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        // The three named optional files stay tracked in the manifest; the
        // removed file itself carries no manifest field — REMOVED_FILES is a
        // separate rejection loop, not a FILE_ENTRIES row.
        assert!(manifest.constraints_line_bounds_parquet);
        assert!(manifest.constraints_hydro_bounds_parquet);
        assert!(manifest.scenarios_load_factors_json);

        // Every other optional flag stays false — pins that the removed-file
        // loop did not shift the FILE_ENTRIES / manifest_fields_mut positional zip.
        assert!(!manifest.system_non_controllable_sources_json);
        assert!(!manifest.system_pumping_stations_json);
        assert!(!manifest.system_energy_contracts_json);
        assert!(!manifest.system_hydro_geometry_parquet);
        assert!(!manifest.system_hydro_production_models_json);
        assert!(!manifest.system_fpha_hyperplanes_parquet);
        assert!(!manifest.system_hydro_energy_productivity_parquet);
        assert!(!manifest.system_tailrace_curves_parquet);

        assert!(!manifest.scenarios_inflow_history_parquet);
        assert!(!manifest.scenarios_inflow_seasonal_stats_parquet);
        assert!(!manifest.scenarios_inflow_ar_coefficients_parquet);
        assert!(!manifest.scenarios_inflow_annual_component_parquet);
        assert!(!manifest.scenarios_external_inflow_scenarios_parquet);
        assert!(!manifest.scenarios_external_load_scenarios_parquet);
        assert!(!manifest.scenarios_external_ncs_scenarios_parquet);
        assert!(!manifest.scenarios_load_seasonal_stats_parquet);
        assert!(!manifest.scenarios_correlation_json);
        assert!(!manifest.scenarios_non_controllable_factors_json);
        assert!(!manifest.scenarios_non_controllable_stats_parquet);

        assert!(!manifest.constraints_thermal_bounds_parquet);
        assert!(!manifest.constraints_pumping_bounds_parquet);
        assert!(!manifest.constraints_contract_bounds_parquet);
        assert!(!manifest.constraints_generic_constraints_json);
        assert!(!manifest.constraints_generic_constraint_bounds_parquet);
        assert!(!manifest.constraints_generic_parameters_json);
        assert!(!manifest.constraints_penalty_overrides_bus_parquet);
        assert!(!manifest.constraints_penalty_overrides_line_parquet);
        assert!(!manifest.constraints_penalty_overrides_hydro_parquet);
        assert!(!manifest.constraints_penalty_overrides_ncs_parquet);
        assert!(!manifest.constraints_ncs_bounds_parquet);
        assert!(!manifest.constraints_hydro_unit_group_bounds_parquet);
    }

    #[test]
    fn similarly_named_file_alongside_removed_file_is_not_rejected() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        let constraints_dir = dir.path().join("constraints");
        fs::create_dir_all(&constraints_dir).unwrap();
        // Substring "exchange", not the exact REMOVED_FILES path: pins that the
        // check is exact-path equality, not a substring/prefix match that would
        // also catch this file.
        fs::write(constraints_dir.join("exchange_factors_backup.json"), b"{}").unwrap();

        let mut ctx = ValidationContext::new();
        let _manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "a similarly-named file must not trigger the removed-file rejection, got: {:?}",
            ctx.errors()
        );
    }
}
