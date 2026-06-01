//! Layer 5a — hydro-domain semantic validation.
//!
//! Cascade acyclicity, hydro bounds, lifecycle consistency,
//! filling config, geometry monotonicity, evaporation geometry
//! coverage, FPHA constraint shape.

use std::collections::{HashMap, HashSet};

use super::super::{ErrorKind, ValidationContext, schema::ParsedData};

pub(super) fn check_cascade_acyclic(data: &ParsedData, ctx: &mut ValidationContext) {
    if data.hydros.is_empty() {
        return;
    }

    let all_ids: Vec<i32> = data.hydros.iter().map(|h| h.id.0).collect();
    let downstream_set: HashSet<i32> = all_ids.iter().copied().collect();

    let mut adjacency: HashMap<i32, Vec<i32>> =
        all_ids.iter().copied().map(|id| (id, Vec::new())).collect();
    for hydro in &data.hydros {
        if let Some(ds) = hydro.downstream_id
            && downstream_set.contains(&ds.0)
        {
            adjacency.entry(hydro.id.0).or_default().push(ds.0);
        }
    }

    let mut in_degree: HashMap<i32, usize> = all_ids.iter().copied().map(|id| (id, 0)).collect();
    for hydro in &data.hydros {
        if let Some(ds) = hydro.downstream_id
            && downstream_set.contains(&ds.0)
        {
            *in_degree.entry(ds.0).or_insert(0) += 1;
        }
    }

    let mut queue: std::collections::VecDeque<i32> = in_degree
        .iter()
        .filter(|&(_, deg)| *deg == 0)
        .map(|(&id, _)| id)
        .collect();

    let mut visited_count: usize = 0;

    while let Some(node) = queue.pop_front() {
        visited_count += 1;
        if let Some(neighbors) = adjacency.get(&node) {
            for &neighbor in neighbors {
                let deg = in_degree.entry(neighbor).or_insert(0);
                if *deg > 0 {
                    *deg -= 1;
                }
                if *deg == 0 {
                    queue.push_back(neighbor);
                }
            }
        }
    }

    if visited_count < all_ids.len() {
        let mut cycle_participants: Vec<i32> = in_degree
            .iter()
            .filter(|&(_, deg)| *deg > 0)
            .map(|(&id, _)| id)
            .collect();
        cycle_participants.sort_unstable();

        ctx.add_error(
            ErrorKind::CycleDetected,
            "system/hydros.json",
            None::<&str>,
            format!(
                "hydro cascade contains a cycle involving hydro IDs: [{}]",
                cycle_participants
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        );
    }
}

pub(super) fn check_hydro_bounds(data: &ParsedData, ctx: &mut ValidationContext) {
    for hydro in &data.hydros {
        let entity_str = format!("Hydro {}", hydro.id.0);

        if hydro.min_storage_hm3 > hydro.max_storage_hm3 {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: min_storage_hm3 ({}) > max_storage_hm3 ({}); storage bounds are inconsistent",
                    hydro.min_storage_hm3, hydro.max_storage_hm3
                ),
            );
        }

        if hydro.min_turbined_m3s > hydro.max_turbined_m3s {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: min_turbined_m3s ({}) > max_turbined_m3s ({}); turbine bounds are inconsistent",
                    hydro.min_turbined_m3s, hydro.max_turbined_m3s
                ),
            );
        }

        if let Some(max_outflow) = hydro.max_outflow_m3s
            && hydro.min_outflow_m3s > max_outflow
        {
            ctx.add_error(
                    ErrorKind::InvalidValue,
                    "system/hydros.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: min_outflow_m3s ({}) > max_outflow_m3s ({}); outflow bounds are inconsistent",
                        hydro.min_outflow_m3s, max_outflow
                    ),
                );
        }

        if hydro.min_generation_mw > hydro.max_generation_mw {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: min_generation_mw ({}) > max_generation_mw ({}); generation bounds are inconsistent",
                    hydro.min_generation_mw, hydro.max_generation_mw
                ),
            );
        }
    }
}

pub(super) fn check_lifecycle_consistency(data: &ParsedData, ctx: &mut ValidationContext) {
    for hydro in &data.hydros {
        if let (Some(entry), Some(exit)) = (hydro.entry_stage_id, hydro.exit_stage_id)
            && entry >= exit
        {
            let entity_str = format!("Hydro {}", hydro.id.0);
            ctx.add_error(
                    ErrorKind::InvalidValue,
                    "system/hydros.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: entry_stage_id ({entry}) >= exit_stage_id ({exit}); entry must precede exit"
                    ),
                );
        }
    }

    for line in &data.lines {
        if let (Some(entry), Some(exit)) = (line.entry_stage_id, line.exit_stage_id)
            && entry >= exit
        {
            let entity_str = format!("Line {}", line.id.0);
            ctx.add_error(
                    ErrorKind::InvalidValue,
                    "system/lines.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: entry_stage_id ({entry}) >= exit_stage_id ({exit}); entry must precede exit"
                    ),
                );
        }
    }

    for thermal in &data.thermals {
        if let (Some(entry), Some(exit)) = (thermal.entry_stage_id, thermal.exit_stage_id)
            && entry >= exit
        {
            let entity_str = format!("Thermal {}", thermal.id.0);
            ctx.add_error(
                    ErrorKind::InvalidValue,
                    "system/thermals.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: entry_stage_id ({entry}) >= exit_stage_id ({exit}); entry must precede exit"
                    ),
                );
        }
    }
}

pub(super) fn check_filling_config(data: &ParsedData, ctx: &mut ValidationContext) {
    let study_stage_ids: HashSet<i32> = data
        .stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect();

    for hydro in &data.hydros {
        if let Some(filling) = &hydro.filling
            && !study_stage_ids.contains(&filling.start_stage_id)
        {
            let entity_str = format!("Hydro {}", hydro.id.0);
            ctx.add_error(
                ErrorKind::InvalidValue,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: filling.start_stage_id ({}) is not a valid study stage ID",
                    filling.start_stage_id
                ),
            );
        }
    }
}

pub(super) fn check_geometry_monotonicity(data: &ParsedData, ctx: &mut ValidationContext) {
    if data.hydro_geometry.is_empty() {
        return;
    }

    let mut i = 0;
    let rows = &data.hydro_geometry;

    while i < rows.len() {
        let current_hydro_id = rows[i].hydro_id.0;
        let group_start = i;

        // Find end of this hydro's group (rows are sorted by hydro_id then volume_hm3).
        while i < rows.len() && rows[i].hydro_id.0 == current_hydro_id {
            i += 1;
        }
        let group = &rows[group_start..i];

        for pair in group.windows(2) {
            let prev = &pair[0];
            let curr = &pair[1];
            let entity_str = format!("Hydro {current_hydro_id}");

            if curr.volume_hm3 <= prev.volume_hm3 {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "system/hydro_geometry.parquet",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: volume_hm3 values are not strictly increasing ({} then {}); geometry curve must have strictly increasing volume",
                        prev.volume_hm3, curr.volume_hm3
                    ),
                );
            }

            if curr.height_m < prev.height_m {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "system/hydro_geometry.parquet",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: height_m values are not non-decreasing ({} then {}); geometry curve must have non-decreasing height with volume",
                        prev.height_m, curr.height_m
                    ),
                );
            }

            if curr.area_km2 < prev.area_km2 {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "system/hydro_geometry.parquet",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: area_km2 values are not non-decreasing ({} then {}); geometry curve must have non-decreasing area with volume",
                        prev.area_km2, curr.area_km2
                    ),
                );
            }
        }
    }
}

/// Hydros with `evaporation_coefficients_mm` require geometry rows in
/// `hydro_geometry.parquet` (area-volume curve for linearization).
pub(super) fn check_evaporation_geometry_coverage(data: &ParsedData, ctx: &mut ValidationContext) {
    let geometry_hydro_ids: HashSet<i32> =
        data.hydro_geometry.iter().map(|r| r.hydro_id.0).collect();

    for hydro in &data.hydros {
        if hydro.evaporation_coefficients_mm.is_some() && !geometry_hydro_ids.contains(&hydro.id.0)
        {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "system/hydros.json",
                Some(format!("Hydro {} (id={})", hydro.name, hydro.id.0)),
                format!(
                    "hydro {} (id={}) has evaporation_coefficients_mm but no geometry data \
                     in hydro_geometry.parquet; evaporation linearization requires \
                     area-volume curve data",
                    hydro.name, hydro.id.0
                ),
            );
        }
    }
}

pub(super) fn check_fpha_constraints(data: &ParsedData, ctx: &mut ValidationContext) {
    if data.fpha_hyperplanes.is_empty() {
        return;
    }

    for row in &data.fpha_hyperplanes {
        let entity_str = format!("Hydro {}", row.hydro_id.0);

        if row.gamma_v < 0.0 {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "system/fpha_hyperplanes.parquet",
                Some(&entity_str),
                format!(
                    "{entity_str} (stage={}, plane={}): gamma_v ({}) must be non-negative (>= 0); \
                     power must not decrease with volume/head (zero is valid for constant-head plants)",
                    row.stage_id.map_or_else(|| "all".to_string(), |s| s.to_string()),
                    row.plane_id,
                    row.gamma_v
                ),
            );
        }

        if row.gamma_s > 0.0 {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "system/fpha_hyperplanes.parquet",
                Some(&entity_str),
                format!(
                    "{entity_str} (stage={}, plane={}): gamma_s ({}) must be non-positive (<= 0); power must not increase with spillage",
                    row.stage_id.map_or_else(|| "all".to_string(), |s| s.to_string()),
                    row.plane_id,
                    row.gamma_s
                ),
            );
        }
    }

    let rows = &data.fpha_hyperplanes;
    let mut i = 0;

    while i < rows.len() {
        let current_hydro_id = rows[i].hydro_id.0;
        let current_stage_id = rows[i].stage_id;
        let group_start = i;

        while i < rows.len()
            && rows[i].hydro_id.0 == current_hydro_id
            && rows[i].stage_id == current_stage_id
        {
            i += 1;
        }

        let plane_count = i - group_start;

        if plane_count < 1 {
            let entity_str = format!("Hydro {current_hydro_id}");
            let stage_label = current_stage_id.map_or_else(|| "all".to_string(), |s| s.to_string());
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "system/fpha_hyperplanes.parquet",
                Some(&entity_str),
                format!(
                    "{entity_str} (stage={stage_label}): no FPHA planes defined; \
                     at least 1 plane is required"
                ),
            );
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
mod tests {
    use super::super::test_support::*;
    use super::super::validate_semantic_hydro_thermal;
    use crate::validation::{ErrorKind, ValidationContext};

    // ── Cascade acyclicity tests ───────────────────────────────────────────────

    /// Given an acyclic cascade A -> B -> C (all have downstream_id pointing to next),
    /// no errors are produced.
    #[test]
    fn test_cascade_acyclic_valid() {
        let hydros = vec![
            make_hydro(1, Some(2)), // 1 -> 2
            make_hydro(2, Some(3)), // 2 -> 3
            make_hydro(3, None),    // root (no downstream)
        ];
        let data = make_data(hydros, vec![], vec![], make_stages(vec![0]), vec![], vec![]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid acyclic cascade should produce no errors, got: {:?}",
            ctx.errors()
        );
    }

    /// Given a cycle A -> B -> C -> A, exactly one CycleDetected error is produced.
    #[test]
    fn test_cascade_cycle_detected() {
        let hydros = vec![
            make_hydro(1, Some(2)), // 1 -> 2
            make_hydro(2, Some(3)), // 2 -> 3
            make_hydro(3, Some(1)), // 3 -> 1 (cycle!)
        ];
        let data = make_data(hydros, vec![], vec![], make_stages(vec![0]), vec![], vec![]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors(), "cycle should produce errors");
        let cycle_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::CycleDetected)
            .collect();
        assert!(
            !cycle_errors.is_empty(),
            "should have at least one CycleDetected error"
        );
    }

    /// Empty hydro list produces no cascade errors.
    #[test]
    fn test_cascade_empty_hydros() {
        let data = make_data(vec![], vec![], vec![], make_stages(vec![0]), vec![], vec![]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    // ── Hydro storage bounds tests ────────────────────────────────────────────

    /// min_storage > max_storage produces one InvalidValue error with "Hydro 5"
    /// and "storage" in the message.
    #[test]
    fn test_hydro_storage_min_greater_than_max() {
        let mut hydro = make_hydro(5, None);
        hydro.min_storage_hm3 = 200.0;
        hydro.max_storage_hm3 = 100.0;
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(relevant.len(), 1, "exactly 1 InvalidValue error expected");
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Hydro 5"),
            "message should contain 'Hydro 5', got: {msg}"
        );
        assert!(
            msg.contains("storage"),
            "message should contain 'storage', got: {msg}"
        );
    }

    /// min_storage == max_storage (run-of-river) produces no error.
    #[test]
    fn test_hydro_storage_equal_bounds_valid() {
        let mut hydro = make_hydro(1, None);
        hydro.min_storage_hm3 = 500.0;
        hydro.max_storage_hm3 = 500.0;
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "equal storage bounds should be valid, got: {:?}",
            ctx.errors()
        );
    }

    // ── Hydro turbine bounds tests ────────────────────────────────────────────

    /// min_turbined > max_turbined produces one InvalidValue error.
    #[test]
    fn test_hydro_turbine_min_greater_than_max() {
        let mut hydro = make_hydro(2, None);
        hydro.min_turbined_m3s = 500.0;
        hydro.max_turbined_m3s = 100.0;
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let turbine_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert!(!turbine_errors.is_empty());
    }

    // ── Hydro outflow bounds tests ────────────────────────────────────────────

    /// When max_outflow_m3s is None, no outflow bound error is produced even if
    /// min_outflow_m3s has any value.
    #[test]
    fn test_hydro_outflow_no_max_no_error() {
        let mut hydro = make_hydro(3, None);
        hydro.min_outflow_m3s = 999.0;
        hydro.max_outflow_m3s = None;
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    /// When max_outflow_m3s is Some but min > max, one InvalidValue error is produced.
    #[test]
    fn test_hydro_outflow_min_greater_than_max() {
        let mut hydro = make_hydro(4, None);
        hydro.min_outflow_m3s = 500.0;
        hydro.max_outflow_m3s = Some(300.0);
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
    }

    // ── Lifecycle consistency tests ───────────────────────────────────────────

    /// Hydro with entry >= exit produces one InvalidValue error.
    #[test]
    fn test_hydro_lifecycle_entry_gte_exit() {
        let mut hydro = make_hydro(7, None);
        hydro.entry_stage_id = Some(10);
        hydro.exit_stage_id = Some(5);
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::InvalidValue),
            "should have InvalidValue error for lifecycle"
        );
    }

    /// Hydro with only entry_stage_id set (no exit) produces no lifecycle error.
    #[test]
    fn test_hydro_lifecycle_only_entry_no_error() {
        let mut hydro = make_hydro(8, None);
        hydro.entry_stage_id = Some(5);
        hydro.exit_stage_id = None;
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "only entry_stage_id set should produce no error, got: {:?}",
            ctx.errors()
        );
    }

    /// Hydro with valid entry < exit produces no lifecycle error.
    #[test]
    fn test_hydro_lifecycle_valid() {
        let mut hydro = make_hydro(9, None);
        hydro.entry_stage_id = Some(0);
        hydro.exit_stage_id = Some(10);
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    // ── Geometry monotonicity tests ───────────────────────────────────────────

    /// Empty geometry slice produces no errors.
    #[test]
    fn test_geometry_empty_no_error() {
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    /// Strictly increasing volume, non-decreasing height and area produces no error.
    #[test]
    fn test_geometry_valid_monotonic() {
        let geometry = vec![
            make_geom_row(1, 10.0, 100.0, 1.0),
            make_geom_row(1, 20.0, 110.0, 1.5),
            make_geom_row(1, 30.0, 120.0, 2.0),
        ];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            geometry,
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid monotonic geometry should produce no errors, got: {:?}",
            ctx.errors()
        );
    }

    /// Non-monotonic volume produces BusinessRuleViolation with "Hydro 3" and "volume".
    #[test]
    fn test_geometry_non_monotonic_volume() {
        // Volume sequence [10.0, 20.0, 15.0] has a decrease at index 2.
        // Note: rows pre-sorted by (hydro_id, volume_hm3), but we construct
        // the violation by using the same volume values — the parser would have
        // sorted them, so [10, 15, 20] after sort. To test the actual validation,
        // we craft a case where sorted order still violates (equal volumes).
        // Use equal volumes to trigger the "not strictly increasing" check.
        let geometry = vec![
            make_geom_row(3, 10.0, 100.0, 1.0),
            make_geom_row(3, 20.0, 110.0, 1.5),
            make_geom_row(3, 20.0, 115.0, 1.6), // duplicate volume — not strictly increasing
        ];
        let data = make_data(
            vec![make_hydro(3, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            geometry,
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(!relevant.is_empty(), "should have BusinessRuleViolation");
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Hydro 3"),
            "message should contain 'Hydro 3', got: {msg}"
        );
        assert!(
            msg.contains("volume"),
            "message should contain 'volume', got: {msg}"
        );
    }

    /// Non-monotonic height produces BusinessRuleViolation with "height" in message.
    #[test]
    fn test_geometry_non_monotonic_height() {
        let geometry = vec![
            make_geom_row(2, 10.0, 100.0, 1.0),
            make_geom_row(2, 20.0, 90.0, 1.5), // height decreased — violation
            make_geom_row(2, 30.0, 110.0, 2.0),
        ];
        let data = make_data(
            vec![make_hydro(2, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            geometry,
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(!relevant.is_empty());
        let msg = &relevant[0].message;
        assert!(
            msg.contains("height"),
            "message should mention 'height', got: {msg}"
        );
    }

    // ── FPHA minimum planes tests ─────────────────────────────────────────────

    /// 1 plane for (hydro, stage) is valid — minimum count is 1.
    #[test]
    fn test_fpha_one_plane_valid() {
        let rows = vec![make_fpha_row(1, Some(0), 0)];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "1 plane should be valid (minimum is 1), got: {:?}",
            ctx.errors()
        );
    }

    /// 2 planes for (hydro, stage) is valid — minimum count is 1.
    #[test]
    fn test_fpha_two_planes_valid() {
        let rows = vec![make_fpha_row(1, Some(0), 0), make_fpha_row(1, Some(0), 1)];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "2 planes should be valid (minimum is 1), got: {:?}",
            ctx.errors()
        );
    }

    /// 3 planes for (hydro, stage) produces no minimum-count error.
    #[test]
    fn test_fpha_minimum_planes_valid() {
        let rows = vec![
            make_fpha_row(1, Some(0), 0),
            make_fpha_row(1, Some(0), 1),
            make_fpha_row(1, Some(0), 2),
        ];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "3 planes should be valid, got: {:?}",
            ctx.errors()
        );
    }

    // ── FPHA gamma sign tests ─────────────────────────────────────────────────

    /// Negative gamma_v produces BusinessRuleViolation.
    #[test]
    fn test_fpha_negative_gamma_v() {
        let mut row = make_fpha_row(1, None, 0);
        row.gamma_v = -0.5; // invalid: must be >= 0
        let rows = vec![row, make_fpha_row(1, None, 1), make_fpha_row(1, None, 2)];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::BusinessRuleViolation),
            "negative gamma_v should produce BusinessRuleViolation"
        );
    }

    /// Positive gamma_s produces BusinessRuleViolation.
    #[test]
    fn test_fpha_positive_gamma_s() {
        let mut row = make_fpha_row(1, None, 0);
        row.gamma_s = 0.1; // invalid: must be <= 0
        let rows = vec![row, make_fpha_row(1, None, 1), make_fpha_row(1, None, 2)];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::BusinessRuleViolation),
            "positive gamma_s should produce BusinessRuleViolation"
        );
    }

    /// gamma_s == 0.0 is valid (non-positive).
    #[test]
    fn test_fpha_gamma_s_zero_valid() {
        let rows: Vec<crate::extensions::FphaHyperplaneRow> = (0..3)
            .map(|i| {
                let mut r = make_fpha_row(1, None, i);
                r.gamma_s = 0.0;
                r
            })
            .collect();
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "gamma_s == 0 should be valid, got: {:?}",
            ctx.errors()
        );
    }

    /// gamma_v == 0.0 is valid (constant-head plant: zero storage coefficient).
    #[test]
    fn test_fpha_gamma_v_zero_valid() {
        let mut row = make_fpha_row(1, None, 0);
        row.gamma_v = 0.0; // valid: >= 0 (constant-head)
        let rows = vec![row];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "gamma_v == 0 should be valid for constant-head plants, got: {:?}",
            ctx.errors()
        );
    }

    /// Empty FPHA slice produces no errors (rules 11-12 are skipped).
    #[test]
    fn test_fpha_empty_no_error() {
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    // ── All-rules-checked test (no short-circuit) ─────────────────────────────

    /// Given two hydros each with bound violations, both errors are collected
    /// (all rules checked, no early exit).
    #[test]
    fn test_all_rules_checked_no_short_circuit() {
        let mut h1 = make_hydro(1, None);
        h1.min_storage_hm3 = 200.0;
        h1.max_storage_hm3 = 100.0; // violation

        let mut h2 = make_hydro(2, None);
        h2.min_generation_mw = 500.0;
        h2.max_generation_mw = 100.0; // violation

        let data = make_data(
            vec![h1, h2],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            ctx.errors().len() >= 2,
            "both violations should be collected; got {} errors",
            ctx.errors().len()
        );
    }

    // ── Acceptance criteria tests ─────────────────────────────────────────────

    /// AC 1: Valid data produces no errors.
    #[test]
    fn test_ac1_valid_data_no_errors() {
        let geometry = vec![
            make_geom_row(1, 10.0, 100.0, 1.0),
            make_geom_row(1, 20.0, 110.0, 2.0),
            make_geom_row(1, 30.0, 120.0, 3.0),
        ];
        let fpha: Vec<crate::extensions::FphaHyperplaneRow> =
            (0..3).map(|i| make_fpha_row(1, Some(0), i)).collect();
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![make_thermal(1, 0.0, 500.0)],
            vec![],
            make_stages(vec![0]),
            geometry,
            fpha,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid data should produce no errors, got: {:?}",
            ctx.errors()
        );
    }

    /// AC 2: Hydro id=5 with inverted storage bounds produces exactly 1 InvalidValue
    /// entry whose message contains "Hydro 5" and "storage".
    #[test]
    fn test_ac2_hydro_storage_bounds_error() {
        let mut hydro = make_hydro(5, None);
        hydro.min_storage_hm3 = 200.0;
        hydro.max_storage_hm3 = 100.0;
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(relevant.len(), 1);
        let msg = &relevant[0].message;
        assert!(msg.contains("Hydro 5"), "message must contain 'Hydro 5'");
        assert!(msg.contains("storage"), "message must contain 'storage'");
    }

    /// AC 3: Cycle A->B->C->A produces at least 1 CycleDetected error.
    #[test]
    fn test_ac3_cycle_detected() {
        let hydros = vec![
            make_hydro(1, Some(2)),
            make_hydro(2, Some(3)),
            make_hydro(3, Some(1)),
        ];
        let data = make_data(hydros, vec![], vec![], make_stages(vec![0]), vec![], vec![]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::CycleDetected),
            "should have CycleDetected error"
        );
    }

    /// AC 4: Non-monotonic volume for hydro id=3 produces BusinessRuleViolation
    /// with "Hydro 3" and "volume" in the message.
    #[test]
    fn test_ac4_geometry_non_monotonic_volume_error() {
        // Use equal volumes (10.0, 20.0, 20.0) to trigger the strict-increase check.
        let geometry = vec![
            make_geom_row(3, 10.0, 100.0, 1.0),
            make_geom_row(3, 20.0, 110.0, 1.5),
            make_geom_row(3, 20.0, 115.0, 1.6),
        ];
        let data = make_data(
            vec![make_hydro(3, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            geometry,
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(!relevant.is_empty(), "should have BusinessRuleViolation");
        let msg = &relevant[0].message;
        assert!(msg.contains("Hydro 3"), "must contain 'Hydro 3': {msg}");
        assert!(msg.contains("volume"), "must contain 'volume': {msg}");
    }

    /// AC 5: Empty geometry and FPHA produce no errors from rules 8-12.
    #[test]
    fn test_ac5_empty_geometry_and_fpha_no_false_positives() {
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![], // empty geometry
            vec![], // empty FPHA
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "empty geometry and FPHA should produce no errors, got: {:?}",
            ctx.errors()
        );
    }
}
