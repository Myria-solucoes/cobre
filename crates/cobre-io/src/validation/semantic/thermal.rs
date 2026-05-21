//! Layer 5a — thermal-domain semantic validation.
//!
//! Thermal generation bounds (`min_generation_mw <= max_generation_mw`) and
//! anticipated-thermal cross-field invariants.

use std::collections::HashMap;

use cobre_core::{AnticipatedCommitmentHistory, EntityId};

use super::super::{ErrorKind, ValidationContext, schema::ParsedData};

pub(super) fn check_thermal_generation_bounds(data: &ParsedData, ctx: &mut ValidationContext) {
    for thermal in &data.thermals {
        if thermal.min_generation_mw > thermal.max_generation_mw {
            let entity_str = format!("Thermal {}", thermal.id.0);
            ctx.add_error(
                ErrorKind::InvalidValue,
                "system/thermals.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: min_generation_mw ({}) > max_generation_mw ({}); generation bounds are inconsistent",
                    thermal.min_generation_mw, thermal.max_generation_mw
                ),
            );
        }
    }
}

/// Checks cross-field invariants for anticipated thermal plants.
///
/// Enforces:
///
/// 1. **Per-plant horizon window** — for each thermal with
///    `anticipated_config: Some(AnticipatedConfig { lead_stages: K })`:
///    - `K == 0` is rejected (defence in depth; parse-time also rejects this).
///    - `K > n_stages` — the plant can never deliver within the study horizon.
///    - `entry_stage_id = Some(e)` and `e + K > n_stages` — delivery window
///      starts after the study horizon ends (hard rejection, not a warning).
///    - `exit_stage_id = Some(x)` and `x < K` — the plant exits before its
///      earliest possible delivery at stage `K`.
///
/// 2. **Past-commitments registry correspondence** — validates the bijection
///    between anticipated thermals and `ic.past_anticipated_commitments`:
///    - Every anticipated thermal must have exactly one matching history entry.
///    - Every history entry must reference a thermal whose `anticipated_config`
///      is `Some`.
///    - Each matched `(thermal, history)` pair must satisfy
///      `history.values_mw.len() == lead_stages as usize` exactly.
pub(super) fn check_anticipated_thermals(data: &ParsedData, ctx: &mut ValidationContext) {
    // Study stages only: pre-study stages have negative IDs and are never
    // delivery targets for anticipated commitments.
    let n_stages = data.stages.stages.iter().filter(|s| s.id >= 0).count();

    // ── 1. Per-plant horizon window ───────────────────────────────────────────

    for thermal in &data.thermals {
        let Some(ref cfg) = thermal.anticipated_config else {
            continue;
        };
        let k = cfg.lead_stages;
        let thermal_id = thermal.id.0;
        let entity_str = format!("thermals[id={thermal_id}].anticipated_config.lead_stages");

        // K == 0: defence in depth (parse-time should also reject)
        if k == 0 {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "system/thermals.json",
                Some(&entity_str),
                format!("Thermal {thermal_id}: anticipated_config.lead_stages must be >= 1, got 0"),
            );
            continue;
        }

        let k_u = k as usize;

        // K > n_stages: plant can never deliver within the study horizon
        if k_u > n_stages {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "system/thermals.json",
                Some(&entity_str),
                format!(
                    "Thermal {thermal_id}: lead_stages exceeds study horizon \
                     (lead_stages={k}, n_stages={n_stages}); \
                     the plant can never deliver within the study horizon"
                ),
            );
        }

        // entry_stage_id + lead_stages > n_stages
        if let Some(e) = thermal.entry_stage_id {
            let e_i = i64::from(e);
            let k_i = i64::from(k);
            let n_i = i64::try_from(n_stages).unwrap_or(i64::MAX);
            let sum = e_i + k_i;
            if sum > n_i {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "system/thermals.json",
                    Some(&entity_str),
                    format!(
                        "Thermal {thermal_id}: entry_stage_id + lead_stages > n_stages \
                         ({e} + {k} = {sum} > {n_stages}); \
                         the anticipated configuration cannot produce a valid delivery \
                         within the study horizon"
                    ),
                );
            }
        }

        // exit_stage_id < lead_stages
        if let Some(x) = thermal.exit_stage_id {
            if i64::from(x) < i64::from(k) {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "system/thermals.json",
                    Some(&entity_str),
                    format!(
                        "Thermal {thermal_id}: exit_stage_id ({x}) < lead_stages ({k}); \
                         the plant exits before its earliest possible delivery at stage {k}, \
                         so the anticipated configuration cannot produce a valid delivery"
                    ),
                );
            }
        }
    }

    // ── 2. Past-commitments registry correspondence ───────────────────────────

    // Build a map from thermal_id -> history entry for the IC entries.
    let ic = &data.initial_conditions;
    let mut history_by_id: HashMap<EntityId, &AnticipatedCommitmentHistory> = HashMap::new();
    for history in &ic.past_anticipated_commitments {
        history_by_id.insert(history.thermal_id, history);
    }

    // Build a set of thermal IDs that have anticipated_config.
    let mut anticipated_thermal_ids: std::collections::HashSet<EntityId> =
        std::collections::HashSet::new();
    for thermal in &data.thermals {
        if thermal.anticipated_config.is_some() {
            anticipated_thermal_ids.insert(thermal.id);
        }
    }

    // Check: every anticipated thermal must have exactly one matching history entry.
    for thermal in &data.thermals {
        let Some(ref cfg) = thermal.anticipated_config else {
            continue;
        };
        let k = cfg.lead_stages;
        let thermal_id = thermal.id;

        match history_by_id.get(&thermal_id) {
            None => {
                // Missing entry
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "initial_conditions.json",
                    Some("initial_conditions.past_anticipated_commitments"),
                    format!(
                        "Thermal {}: missing entry in initial_conditions.past_anticipated_commitments; \
                         every anticipated thermal must have a corresponding history entry",
                        thermal_id.0
                    ),
                );
            }
            Some(history) => {
                // Check length exactly
                let expected = k as usize;
                let actual = history.values_mw.len();
                if actual != expected {
                    let entity_str = format!(
                        "thermals[id={}].anticipated_config.lead_stages",
                        thermal_id.0
                    );
                    ctx.add_error(
                        ErrorKind::BusinessRuleViolation,
                        "initial_conditions.json",
                        Some(&entity_str),
                        format!(
                            "Thermal {}: past_anticipated_commitments.values_mw has wrong length: \
                             expected {expected} values, got {actual}",
                            thermal_id.0
                        ),
                    );
                }
            }
        }
    }

    // Check: every history entry must reference an anticipated thermal.
    for history in &ic.past_anticipated_commitments {
        if !anticipated_thermal_ids.contains(&history.thermal_id) {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "initial_conditions.json",
                Some(format!(
                    "initial_conditions.past_anticipated_commitments[thermal_id={}]",
                    history.thermal_id.0
                )),
                format!(
                    "Thermal {}: referenced in past_anticipated_commitments \
                     but is not an anticipated thermal (anticipated_config is None or thermal does not exist)",
                    history.thermal_id.0
                ),
            );
        }
    }
}

/// Layer 5a — rejects per-stage thermal bound overrides whose `stage_id` is
/// outside the study horizon `[0, n_stages)`.
///
/// Per epic-04 Decision 7, the thermal-bounds resolution table is padded
/// with each plant's base entity values for stages `[n_stages, n_stages + K_max)`
/// to support anticipated-delivery lookups. Overrides in that padded region
/// would be silently dropped by `resolve_bounds` (the `stage_index` only
/// covers study stages); this validator surfaces them as a user-visible error.
pub(super) fn check_thermal_bounds_override_stage_range(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    // `filter(id >= 0)` counts study stages only, matching the resolver's
    // `stage_index` (cobre-io/src/pipeline.rs builds it from study stages
    // only). Pre-study stages with negative IDs are excluded from the
    // override-table horizon.
    let n_stages = data.stages.stages.iter().filter(|s| s.id >= 0).count();
    let n_stages_i = i64::try_from(n_stages).unwrap_or(i64::MAX);
    for row in &data.thermal_bounds {
        let s = i64::from(row.stage_id);
        if s < 0 || s >= n_stages_i {
            let entity_str = format!("thermal_id={}, stage_id={}", row.thermal_id.0, row.stage_id);
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "constraints/thermal_bounds.parquet",
                Some(&entity_str),
                format!(
                    "Thermal {}: thermal_bounds override at stage_id={} is \
                     outside the study horizon [0, {}); per-stage thermal \
                     overrides past the horizon are not allowed",
                    row.thermal_id.0, row.stage_id, n_stages
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
    use cobre_core::{AnticipatedCommitmentHistory, EntityId, entities::AnticipatedConfig};

    use super::super::test_support::*;
    use super::super::validate_semantic_hydro_thermal;
    use crate::validation::{ErrorKind, ValidationContext};

    // ── Helper: build a thermal with anticipated_config ───────────────────────

    fn make_anticipated_thermal(
        id: i32,
        lead_stages: u32,
        entry_stage_id: Option<i32>,
        exit_stage_id: Option<i32>,
    ) -> cobre_core::entities::Thermal {
        cobre_core::entities::Thermal {
            anticipated_config: Some(AnticipatedConfig { lead_stages }),
            entry_stage_id,
            exit_stage_id,
            ..make_thermal(id, 0.0, 500.0)
        }
    }

    /// Build `ParsedData` for anticipated-thermal tests.
    ///
    /// Supplies `n_stages` stages with IDs `0..n_stages`, one or more thermals,
    /// and places the given `past_anticipated_commitments` in `initial_conditions`.
    fn make_data_anticipated(
        thermals: Vec<cobre_core::entities::Thermal>,
        n_stages: usize,
        past_anticipated_commitments: Vec<AnticipatedCommitmentHistory>,
    ) -> crate::validation::schema::ParsedData {
        let stage_ids: Vec<i32> = (0..n_stages as i32).collect();
        let mut data = make_data(
            vec![],
            thermals,
            vec![],
            make_stages(stage_ids),
            vec![],
            vec![],
        );
        data.initial_conditions.past_anticipated_commitments = past_anticipated_commitments;
        data
    }

    // ── AC-1: valid anticipated thermal — returns Ok(()) ─────────────────────

    /// Given a valid anticipated thermal (lead_stages=2, n_stages=5, no entry/exit,
    /// exactly 2 values_mw in past_anticipated_commitments), validate_anticipated_thermals
    /// returns no errors.
    #[test]
    fn test_valid_anticipated_thermal_ok() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(1),
            values_mw: vec![100.0, 200.0],
        };
        let data = make_data_anticipated(vec![thermal], 5, vec![history]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "expected no errors, got: {:?}",
            ctx.errors()
        );
    }

    // ── AC-2: missing history entry ───────────────────────────────────────────

    /// Given an anticipated thermal with no matching entry in
    /// past_anticipated_commitments, returns Err with thermal_id and "missing".
    #[test]
    fn test_missing_history_entry_error() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let data = make_data_anticipated(vec![thermal], 5, vec![]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            !relevant.is_empty(),
            "expected BusinessRuleViolation, got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Thermal 1"),
            "message should contain 'Thermal 1', got: {msg}"
        );
        assert!(
            msg.contains("missing"),
            "message should contain 'missing', got: {msg}"
        );
        let file = relevant[0].file.to_string_lossy();
        assert!(
            file.contains("initial_conditions"),
            "file path should reference initial_conditions, got: {file}"
        );
        let entity = relevant[0].entity.as_deref().unwrap_or("");
        assert!(
            entity.contains("initial_conditions.past_anticipated_commitments"),
            "entity should contain 'initial_conditions.past_anticipated_commitments', got: {entity}"
        );
    }

    // ── AC-3: history entry for non-anticipated thermal ───────────────────────

    /// Given a history entry whose thermal_id references a thermal with
    /// anticipated_config == None, returns Err with "not an anticipated thermal".
    #[test]
    fn test_history_entry_for_non_anticipated_thermal_error() {
        // thermal 1 is NOT anticipated
        let thermal = make_thermal(1, 0.0, 500.0);
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(1),
            values_mw: vec![100.0, 200.0],
        };
        let data = make_data_anticipated(vec![thermal], 5, vec![history]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            !relevant.is_empty(),
            "expected BusinessRuleViolation, got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("not an anticipated thermal"),
            "message should contain 'not an anticipated thermal', got: {msg}"
        );
    }

    // ── AC-4: over-length values_mw (len > lead_stages) ──────────────────────

    /// Given lead_stages=2 but values_mw.len()==3, returns Err with
    /// "expected 2 values, got 3".
    #[test]
    fn test_over_length_values_mw_error() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(1),
            values_mw: vec![100.0, 200.0, 300.0], // 3 instead of 2
        };
        let data = make_data_anticipated(vec![thermal], 5, vec![history]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            !relevant.is_empty(),
            "expected BusinessRuleViolation, got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("expected 2 values, got 3"),
            "message should contain 'expected 2 values, got 3', got: {msg}"
        );
    }

    // ── AC-5: under-length values_mw (len < lead_stages) ─────────────────────

    /// Given lead_stages=2 but values_mw.len()==1, returns Err with
    /// "expected 2 values, got 1".
    #[test]
    fn test_under_length_values_mw_error() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(1),
            values_mw: vec![100.0], // 1 instead of 2
        };
        let data = make_data_anticipated(vec![thermal], 5, vec![history]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            !relevant.is_empty(),
            "expected BusinessRuleViolation, got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("expected 2 values, got 1"),
            "message should contain 'expected 2 values, got 1', got: {msg}"
        );
    }

    // ── AC-6: lead_stages exceeds study horizon ───────────────────────────────

    /// Given lead_stages=10 and n_stages=5, returns Err with
    /// "lead_stages exceeds study horizon".
    #[test]
    fn test_lead_stages_exceeds_study_horizon_error() {
        let thermal = make_anticipated_thermal(1, 10, None, None);
        let data = make_data_anticipated(vec![thermal], 5, vec![]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("lead_stages exceeds study horizon")
            })
            .collect();
        assert!(
            !relevant.is_empty(),
            "expected BusinessRuleViolation with 'lead_stages exceeds study horizon', got: {errors:?}"
        );
    }

    // ── Boundary: lead_stages == n_stages is accepted (strict-greater check) ──

    /// Boundary case for AC-6: lead_stages == n_stages must NOT error.
    /// The horizon check is `lead_stages > n_stages` (strict). A plant whose
    /// first delivery lands on the final study stage is a valid configuration.
    #[test]
    fn test_lead_stages_equal_n_stages_ok() {
        let thermal = make_anticipated_thermal(1, 5, None, None);
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(1),
            values_mw: vec![100.0, 110.0, 120.0, 130.0, 140.0],
        };
        let data = make_data_anticipated(vec![thermal], 5, vec![history]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "lead_stages == n_stages must be accepted, got: {:?}",
            ctx.errors()
        );
    }

    // ── AC-7: entry_stage_id + lead_stages > n_stages ────────────────────────

    /// Given lead_stages=3, entry_stage_id=Some(4), n_stages=5:
    /// 4 + 3 = 7 > 5 — hard rejection with message containing
    /// "entry_stage_id + lead_stages > n_stages".
    #[test]
    fn test_entry_plus_lead_exceeds_horizon_error() {
        let thermal = make_anticipated_thermal(1, 3, Some(4), None);
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(1),
            values_mw: vec![100.0, 200.0, 300.0],
        };
        let data = make_data_anticipated(vec![thermal], 5, vec![history]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message
                        .contains("entry_stage_id + lead_stages > n_stages")
            })
            .collect();
        assert!(
            !relevant.is_empty(),
            "expected BusinessRuleViolation with 'entry_stage_id + lead_stages > n_stages', got: {errors:?}"
        );
        assert_eq!(
            relevant[0].severity,
            crate::validation::Severity::Error,
            "must be an error"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("4 + 3") && msg.contains('7') && msg.contains('5'),
            "message should show '4 + 3 = 7 > 5', got: {msg}"
        );
    }

    // ── AC-8: exit_stage_id < lead_stages ────────────────────────────────────

    /// Given lead_stages=3 and exit_stage_id=Some(2): the plant exits before
    /// its earliest possible delivery at stage 3. Returns Err with "exit_stage_id".
    #[test]
    fn test_exit_before_earliest_delivery_error() {
        let thermal = make_anticipated_thermal(1, 3, None, Some(2));
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(1),
            values_mw: vec![100.0, 200.0, 300.0],
        };
        let data = make_data_anticipated(vec![thermal], 5, vec![history]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation && e.message.contains("exit_stage_id")
            })
            .collect();
        assert!(
            !relevant.is_empty(),
            "expected a BusinessRuleViolation mentioning 'exit_stage_id', got: {errors:?}"
        );
    }

    // ── Existing bounds test (unchanged) ─────────────────────────────────────

    /// min_generation_mw > max_generation_mw produces InvalidValue with "Thermal <id>".
    #[test]
    fn test_thermal_generation_min_greater_than_max() {
        let thermal = make_thermal(10, 500.0, 100.0); // min > max — violation
        let data = make_data(
            vec![],
            vec![thermal],
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
            msg.contains("Thermal 10"),
            "message should contain 'Thermal 10', got: {msg}"
        );
    }

    /// min_generation_mw == max_generation_mw produces no error.
    #[test]
    fn test_thermal_generation_equal_bounds_valid() {
        let thermal = make_thermal(11, 200.0, 200.0);
        let data = make_data(
            vec![],
            vec![thermal],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    // ── Layer 5a rule 16: thermal_bounds override stage_id within [0, n_stages)

    /// Build a `ThermalBoundsRow` with the given thermal_id and stage_id and
    /// no override values set.
    fn make_thermal_bounds_row(thermal_id: i32, stage_id: i32) -> crate::ThermalBoundsRow {
        crate::ThermalBoundsRow {
            thermal_id: EntityId::from(thermal_id),
            stage_id,
            min_generation_mw: None,
            max_generation_mw: None,
            cost_per_mwh: None,
            block_id: None,
        }
    }

    /// Build a `ParsedData` with `n_stages` study stages, one thermal,
    /// and the given `thermal_bounds` rows.
    fn make_data_thermal_bounds(
        n_stages: usize,
        rows: Vec<crate::ThermalBoundsRow>,
    ) -> crate::validation::schema::ParsedData {
        let thermal = make_thermal(1, 0.0, 100.0);
        let stage_ids: Vec<i32> = (0..n_stages as i32).collect();
        let mut data = make_data(
            vec![],
            vec![thermal],
            vec![],
            make_stages(stage_ids),
            vec![],
            vec![],
        );
        data.thermal_bounds = rows;
        data
    }

    /// `n_stages = 5`, row `stage_id = 4` is within `[0, 5)` — accepted.
    #[test]
    fn test_thermal_bounds_override_stage_within_horizon_accepted() {
        let data = make_data_thermal_bounds(5, vec![make_thermal_bounds_row(1, 4)]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.file
                    .to_string_lossy()
                    .contains("constraints/thermal_bounds.parquet")
            })
            .collect();
        assert!(
            relevant.is_empty(),
            "expected no thermal_bounds.parquet errors, got: {relevant:?}"
        );
    }

    /// `n_stages = 5`, row `stage_id = 5` is the first invalid index — rejected.
    #[test]
    fn test_thermal_bounds_override_stage_equals_n_rejected() {
        let data = make_data_thermal_bounds(5, vec![make_thermal_bounds_row(1, 5)]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file
                        .to_string_lossy()
                        .contains("constraints/thermal_bounds.parquet")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one BusinessRuleViolation, got: {relevant:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("stage_id=5"),
            "message should contain 'stage_id=5', got: {msg}"
        );
        assert!(
            msg.contains("[0, 5)"),
            "message should contain '[0, 5)', got: {msg}"
        );
        assert!(
            msg.contains("not allowed"),
            "message should contain 'not allowed', got: {msg}"
        );
    }

    // Acceptance-boundary case is covered by
    // `test_thermal_bounds_override_stage_within_horizon_accepted` above and
    // by `boundary_tests::override_at_t_minus_1_acceptance_boundary` below.

    /// `stage_id = -1` (pre-study stage) — rejected.
    #[test]
    fn test_thermal_bounds_override_stage_negative_rejected() {
        let data = make_data_thermal_bounds(5, vec![make_thermal_bounds_row(1, -1)]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file
                        .to_string_lossy()
                        .contains("constraints/thermal_bounds.parquet")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one BusinessRuleViolation for stage_id=-1, got: {relevant:?}"
        );
    }

    /// Three rows, two offending (`stage_id == n_stages` and
    /// `stage_id > n_stages`), one valid. Exactly two errors are emitted.
    #[test]
    fn test_thermal_bounds_override_multiple_offending_rows() {
        let rows = vec![
            make_thermal_bounds_row(1, 0), // valid
            make_thermal_bounds_row(1, 5), // invalid: equals n_stages
            make_thermal_bounds_row(1, 9), // invalid: past n_stages
        ];
        let data = make_data_thermal_bounds(5, rows);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file
                        .to_string_lossy()
                        .contains("constraints/thermal_bounds.parquet")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            2,
            "expected exactly two BusinessRuleViolations, got: {relevant:?}"
        );
    }

    /// `n_stages = 0`: the half-open interval `[0, 0)` is empty, so any row
    /// at any non-negative stage_id is rejected.
    #[test]
    fn test_thermal_bounds_override_zero_n_stages_all_rejected() {
        let data = make_data_thermal_bounds(0, vec![make_thermal_bounds_row(1, 0)]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file
                        .to_string_lossy()
                        .contains("constraints/thermal_bounds.parquet")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one BusinessRuleViolation when n_stages=0, got: {relevant:?}"
        );
    }

    // ── ticket-019: boundary_tests sub-module ─────────────────────────────────
    //
    // Re-organizes the ticket-018 strict-inequality boundary tests into a
    // discoverable named sub-module using the
    // `..._acceptance_boundary` / `..._rejection_boundary` suffix convention
    // established by epic-03. Each test below mirrors a ticket-018 test (the
    // originals are preserved above to keep the existing test surface stable
    // — see ticket-019 §"DO NOT delete the existing ticket-018 tests").
    //
    // The predicate under test is the strict-inequality guard `stage_id < 0
    // || stage_id >= n_stages` in `validate_semantic_hydro_thermal`. The
    // boundaries are:
    //
    //   * acceptance at `stage_id == n_stages - 1` (the last in-horizon row)
    //   * rejection  at `stage_id == n_stages`     (first out-of-horizon row)
    //   * rejection  at `stage_id == n_stages + 1` (interior rejection)
    //   * rejection  at `stage_id < 0`             (negative pre-study)
    mod boundary_tests {
        use super::*;

        /// `n_stages = 5`, `stage_id = 4` is within `[0, 5)` — accepted.
        #[test]
        fn override_at_t_minus_1_acceptance_boundary() {
            let data = make_data_thermal_bounds(5, vec![make_thermal_bounds_row(1, 4)]);
            let mut ctx = ValidationContext::new();
            validate_semantic_hydro_thermal(&data, &mut ctx);
            let errors = ctx.errors();
            let relevant: Vec<_> = errors
                .iter()
                .filter(|e| {
                    e.file
                        .to_string_lossy()
                        .contains("constraints/thermal_bounds.parquet")
                })
                .collect();
            assert!(
                relevant.is_empty(),
                "stage_id=4 with n_stages=5 must be accepted, got: {relevant:?}"
            );
        }

        /// `n_stages = 5`, `stage_id = 5` is the first invalid index — rejected.
        #[test]
        fn override_at_t_rejection_boundary() {
            let data = make_data_thermal_bounds(5, vec![make_thermal_bounds_row(1, 5)]);
            let mut ctx = ValidationContext::new();
            validate_semantic_hydro_thermal(&data, &mut ctx);
            let errors = ctx.errors();
            let relevant: Vec<_> = errors
                .iter()
                .filter(|e| {
                    e.kind == ErrorKind::BusinessRuleViolation
                        && e.file
                            .to_string_lossy()
                            .contains("constraints/thermal_bounds.parquet")
                })
                .collect();
            assert_eq!(
                relevant.len(),
                1,
                "expected exactly one BusinessRuleViolation at stage_id=5, got: {relevant:?}"
            );
        }

        /// `n_stages = 5`, `stage_id = 6` (one past the rejection boundary)
        /// — rejected. Interior rejection to complement the boundary
        /// rejection at `stage_id == n_stages`.
        #[test]
        fn override_at_t_plus_one_rejection() {
            let data = make_data_thermal_bounds(5, vec![make_thermal_bounds_row(1, 6)]);
            let mut ctx = ValidationContext::new();
            validate_semantic_hydro_thermal(&data, &mut ctx);
            let errors = ctx.errors();
            let relevant: Vec<_> = errors
                .iter()
                .filter(|e| {
                    e.kind == ErrorKind::BusinessRuleViolation
                        && e.file
                            .to_string_lossy()
                            .contains("constraints/thermal_bounds.parquet")
                })
                .collect();
            assert_eq!(
                relevant.len(),
                1,
                "expected exactly one BusinessRuleViolation at stage_id=6, got: {relevant:?}"
            );
        }

        /// `stage_id = -1` (pre-study) — rejected.
        #[test]
        fn override_negative_stage_rejection() {
            let data = make_data_thermal_bounds(5, vec![make_thermal_bounds_row(1, -1)]);
            let mut ctx = ValidationContext::new();
            validate_semantic_hydro_thermal(&data, &mut ctx);
            let errors = ctx.errors();
            let relevant: Vec<_> = errors
                .iter()
                .filter(|e| {
                    e.kind == ErrorKind::BusinessRuleViolation
                        && e.file
                            .to_string_lossy()
                            .contains("constraints/thermal_bounds.parquet")
                })
                .collect();
            assert_eq!(
                relevant.len(),
                1,
                "expected exactly one BusinessRuleViolation at stage_id=-1, got: {relevant:?}"
            );
        }
    }
}
