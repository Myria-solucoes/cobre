//! End-to-end proof that a downstream plant's declared `reference_volume`
//! reaches the upstream plant's backwater family selection.
//!
//! The case `d31-backwater-reference-volume` wires an upstream computed-FPHA
//! plant U (`hydro_id = 0`) discharging into a downstream plant D
//! (`hydro_id = 1`). U's tailrace carries two backwater families keyed at
//! distinct downstream levels (`downstream_reference_level_m`), so the resolved downstream
//! level — D's forebay surface evaluated at D's reference operating volume —
//! moves the interpolated tailrace elevation and therefore U's fitted FPHA
//! planes.
//!
//! D declares `reference_volume: { percentile: 0.95 }`. The two tests below
//! prove (a) clearing that declaration (falling back to the 0.65 default) shifts
//! U's exported planes, so the JSON value genuinely reaches
//! `resolve_downstream_level`, and (b) reversing the auxiliary-row declaration
//! order leaves U's planes bit-identical (the declaration-order-invariance
//! contract the case must uphold).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;

use cobre_io::FphaHyperplaneRow;
use cobre_io::extensions::SelectionMode;
use cobre_sddp::hydro_models::prepare_hydro_models_from_artifacts;

/// Map a D-case label to its fixture directory.
fn case_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/deterministic")
        .join("d31-backwater-reference-volume")
}

/// Load D31's system + artifacts straight from disk.
///
/// `load_case_with_artifacts` does not yet parse `tailrace_curves.parquet` into
/// the bundle (it defaults the field to empty); the production pipeline fills it
/// from disk inside `prepare_hydro_models`. This helper mirrors that step so the
/// `prepare_hydro_models_from_artifacts` call below sees the same backwater
/// families the CLI/training path does.
fn load_d31() -> (cobre_core::System, cobre_io::CaseArtifacts) {
    let dir = case_dir();
    let loaded = cobre_io::load_case_with_artifacts(&dir).expect("d31 must load");
    let mut artifacts = loaded.artifacts;
    let tailrace_path = dir.join("system").join("tailrace_curves.parquet");
    artifacts.tailrace_curves = cobre_io::extensions::load_tailrace_curves(Some(&tailrace_path))
        .expect("tailrace curves must load");
    (loaded.system, artifacts)
}

/// The FPHA export rows produced for `system` from `artifacts`, restricted to
/// the upstream plant U (`hydro_id = 0`) and sorted into a stable order so two
/// runs compare element-by-element.
fn upstream_planes(
    system: &cobre_core::System,
    artifacts: &cobre_io::CaseArtifacts,
) -> Vec<FphaHyperplaneRow> {
    let prepared = prepare_hydro_models_from_artifacts(system, artifacts, false, None)
        .expect("hydro models must prepare");
    let mut rows: Vec<FphaHyperplaneRow> = prepared
        .fpha_export_rows
        .into_iter()
        .filter(|r| r.hydro_id == cobre_core::EntityId::from(0))
        .collect();
    rows.sort_by(|a, b| {
        (a.stage_id, a.plane_id)
            .cmp(&(b.stage_id, b.plane_id))
            .then(a.gamma_0.total_cmp(&b.gamma_0))
    });
    rows
}

/// Clear the downstream plant D's (`hydro_id = 1`) declared `reference_volume`
/// in place, so the resolver falls back to the 0.65 default fraction. U's plane
/// fit is the only observable that may change.
fn clear_downstream_reference_volume(artifacts: &mut cobre_io::CaseArtifacts) {
    for config in &mut artifacts.production_models {
        if config.hydro_id != cobre_core::EntityId::from(1) {
            continue;
        }
        match &mut config.selection_mode {
            SelectionMode::StageRanges { ranges } => {
                for range in ranges {
                    range.reference_volume = None;
                }
            }
            SelectionMode::Seasonal { seasons, .. } => {
                for season in seasons {
                    season.reference_volume = None;
                }
            }
        }
    }
}

/// Declaring `reference_volume` on the downstream plant D measurably shifts the
/// upstream plant U's exported FPHA planes versus the 0.65-default fallback —
/// proving the JSON value reaches `resolve_downstream_level`. A value that left
/// U's planes unchanged would prove nothing (the backwater family bracket was
/// not crossed), so the test asserts an actual bit-level difference.
#[test]
fn backwater_reference_volume_shifts_upstream_planes() {
    let (system, declared_artifacts) = load_d31();

    let with_reference_volume = upstream_planes(&system, &declared_artifacts);

    let mut default_artifacts = declared_artifacts.clone();
    clear_downstream_reference_volume(&mut default_artifacts);
    let without_reference_volume = upstream_planes(&system, &default_artifacts);

    assert!(
        !with_reference_volume.is_empty(),
        "the upstream computed-FPHA plant must export at least one plane"
    );
    assert_eq!(
        with_reference_volume.len(),
        without_reference_volume.len(),
        "the plane count must not depend on the downstream reference volume"
    );

    // At least one gamma coefficient must differ at the bit level: the declared
    // reference volume and the default fraction resolve different downstream
    // forebay levels, which the multi-family tailrace turns into different
    // interpolated elevations and therefore different fitted planes.
    let differs = with_reference_volume
        .iter()
        .zip(&without_reference_volume)
        .any(|(declared, default)| {
            declared.gamma_0.to_bits() != default.gamma_0.to_bits()
                || declared.gamma_v.to_bits() != default.gamma_v.to_bits()
                || declared.gamma_q.to_bits() != default.gamma_q.to_bits()
                || declared.gamma_s.to_bits() != default.gamma_s.to_bits()
        });
    assert!(
        differs,
        "U's planes must differ between the declared reference volume and the \
         0.65 default — the backwater family bracket was not crossed:\n\
         declared = {with_reference_volume:?}\n\
         default  = {without_reference_volume:?}"
    );
}

/// Reversing the declaration order of the auxiliary rows (geometry, tailrace,
/// and production-model configs) leaves U's exported planes bit-identical. The
/// resolve path re-sorts geometry by volume, groups tailrace families by plant,
/// and keys production configs by `hydro_id`, so input ordering must not perturb
/// the fitted planes — the declaration-order-invariance hard rule.
#[test]
fn backwater_upstream_planes_are_declaration_order_invariant() {
    let (system, forward_artifacts) = load_d31();
    let forward = upstream_planes(&system, &forward_artifacts);

    let mut reversed_artifacts = forward_artifacts.clone();
    reversed_artifacts.hydro_geometry.reverse();
    reversed_artifacts.tailrace_curves.reverse();
    reversed_artifacts.production_models.reverse();
    let reversed = upstream_planes(&system, &reversed_artifacts);

    assert_eq!(
        forward.len(),
        reversed.len(),
        "plane counts must match across input orderings"
    );
    for (a, b) in forward.iter().zip(&reversed) {
        assert_eq!(a.stage_id, b.stage_id);
        assert_eq!(a.plane_id, b.plane_id);
        assert_eq!(
            a.gamma_0.to_bits(),
            b.gamma_0.to_bits(),
            "gamma_0 must be bit-identical across input orderings"
        );
        assert_eq!(a.gamma_v.to_bits(), b.gamma_v.to_bits());
        assert_eq!(a.gamma_q.to_bits(), b.gamma_q.to_bits());
        assert_eq!(a.gamma_s.to_bits(), b.gamma_s.to_bits());
    }
}
