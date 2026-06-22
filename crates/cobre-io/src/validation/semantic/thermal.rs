//! Layer 5a — thermal-domain semantic validation.
//!
//! Thermal generation bounds (`min_generation_mw <= max_generation_mw`) and
//! anticipated-thermal cross-field invariants.

use std::collections::{HashMap, HashSet};

use cobre_core::{AnticipatedCommitmentHistory, EntityId, VariableRef};

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
/// 1. **Per-plant lead-stage horizon** — for each thermal with
///    `anticipated_config: Some(AnticipatedConfig { lead_stages: K })`:
///    - `K == 0` is rejected (defence in depth; parse-time also rejects this).
///    - `K > n_stages` — the plant can never deliver within the study horizon.
///    - **Commissioning window rejection** — an anticipated thermal that sets
///      `entry_stage_id` or `exit_stage_id` is rejected. Anticipated thermals
///      carry the `A·K` anticipated-state column block, a ring buffer of past
///      commitments, and a `K`-stage delivery lookahead; a commissioning window
///      on them needs a separate design and is out of scope. This blanket
///      rejection SUPERSEDES the earlier horizon-edge window checks
///      (`entry + K > n_stages` and `exit < K`): both only fired when a window
///      was set, which is now rejected outright, so they were unreachable and
///      are removed. The dispatch-side zeroing of dormant thermal columns
///      therefore never sees an anticipated thermal.
///
/// 2. **Past-commitments registry correspondence** — validates the bijection
///    between anticipated thermals and `ic.past_anticipated_commitments`:
///    - Every anticipated thermal must have exactly one matching history entry.
///    - Every history entry must reference a thermal whose `anticipated_config`
///      is `Some`.
///    - Each matched `(thermal, history)` pair must satisfy
///      `history.values_mw.len() == lead_stages as usize` exactly.
///
/// 3. **Committed-value generation bounds** — for each matched
///    `(thermal, history)` pair whose length is correct, every
///    `history.values_mw[j]` must lie within
///    `[thermal.min_generation_mw, thermal.max_generation_mw]`.
///    A value outside this range seeds the ring buffer with a commitment that
///    cannot be satisfied by the per-block generation bounds in the LP,
///    causing an infeasible problem at the delivery stage.
pub(super) fn check_anticipated_thermals(data: &ParsedData, ctx: &mut ValidationContext) {
    // Study stages only: pre-study stages have negative IDs and are never
    // delivery targets for anticipated commitments.
    let n_stages = data.stages.stages.iter().filter(|s| s.id >= 0).count();

    // ── 1. Per-plant lead-stage horizon ───────────────────────────────────────

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

        // A commissioning window on an anticipated thermal is rejected outright:
        // the anticipated-state column block, the past-commitment ring buffer, and
        // the K-stage delivery lookahead are not designed to interact with a window,
        // so dispatch never zeroes a dormant anticipated thermal. This supersedes
        // the former entry+K>n_stages / exit<K horizon-edge window checks.
        if thermal.entry_stage_id.is_some() || thermal.exit_stage_id.is_some() {
            let window_entity = format!("thermals[id={thermal_id}]");
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "system/thermals.json",
                Some(&window_entity),
                format!(
                    "Thermal {thermal_id}: commissioning windows \
                     (entry_stage_id/exit_stage_id) are not supported on anticipated \
                     thermals; remove the window or the anticipated_config"
                ),
            );
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
                let expected = k as usize;
                let actual = history.values_mw.len();
                if actual == expected {
                    check_committed_value_bounds(thermal, thermal_id, &history.values_mw, ctx);
                } else {
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

/// For an anticipated thermal whose history has the correct length, check each
/// `values_mw[j]` for LP feasibility.
///
/// **Generation bounds check** — every `values_mw[j]` must lie within
/// `[thermal.min_generation_mw, thermal.max_generation_mw]`.
///
/// Out-of-bounds entries would make the LP infeasible at the corresponding
/// stage's fishing equality.
fn check_committed_value_bounds(
    thermal: &cobre_core::entities::Thermal,
    thermal_id: EntityId,
    values_mw: &[f64],
    ctx: &mut ValidationContext,
) {
    let min_mw = thermal.min_generation_mw;
    let max_mw = thermal.max_generation_mw;
    let entity_str = format!("thermals[id={}].anticipated_config", thermal_id.0);
    for (j, &v) in values_mw.iter().enumerate() {
        if v < min_mw || v > max_mw {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "initial_conditions.json",
                Some(&entity_str),
                format!(
                    "Thermal {}: past_anticipated_commitments.values_mw[{j}] = {v} \
                     is outside the plant's generation bounds [{min_mw}, {max_mw}]; \
                     the LP delivery equality at the corresponding stage cannot be \
                     satisfied and the LP will be infeasible",
                    thermal_id.0
                ),
            );
        }
    }
}

/// Layer 5a — rejects use of `anticipated_decision(N)` in a generic constraint
/// when thermal `N` does not have `anticipated_config: Some(_)`.
///
/// `anticipated_decision` is an LP column that only exists for plants committed
/// in advance. Referencing a non-anticipated thermal via this variant is always
/// a model error: the column does not appear in the LP and the constraint would
/// silently become an equality 0 = bound, which is either trivially satisfied
/// or immediately infeasible.
pub(super) fn check_anticipated_decision_target_is_anticipated(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    // Build the set of thermal IDs that are anticipated.
    let anticipated_ids: HashSet<EntityId> = data
        .thermals
        .iter()
        .filter(|t| t.anticipated_config.is_some())
        .map(|t| t.id)
        .collect();

    for constraint in &data.generic_constraints {
        for term in &constraint.expression.terms {
            if let VariableRef::AnticipatedDecision { thermal_id } = term.variable
                && !anticipated_ids.contains(&thermal_id)
            {
                let entity_str = format!("constraint[id={}]", constraint.id.0);
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "constraints/generic_constraints.json",
                    Some(&entity_str),
                    format!(
                        "Constraint \"{}\": anticipated_decision({}) references Thermal {} \
                             which is not an anticipated thermal (anticipated_config is None). \
                             The anticipated_decision column only exists for plants with \
                             anticipated_config set.",
                        constraint.name, thermal_id.0, thermal_id.0,
                    ),
                );
            }
        }
    }
}

/// Layer 5a — warns when `thermal_generation(N)` is used in a generic
/// constraint and thermal `N` is anticipated.
///
/// `thermal_generation` for an anticipated thermal references the per-block
/// generation at the *delivery* stage (the stage when the commitment matures),
/// not the commitment made at the current stage. This is valid but frequently
/// surprising: users who want to constrain the *commitment* should use
/// `anticipated_decision(N)` instead. The stage at which the commitment is
/// expressed differs from the stage at which generation is observed.
///
/// This validator emits a `SemanticAmbiguity` warning so the model author
/// can confirm the intent.
pub(super) fn warn_thermal_generation_on_anticipated_thermal(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    // Build the set of thermal IDs that are anticipated.
    let anticipated_ids: HashSet<EntityId> = data
        .thermals
        .iter()
        .filter(|t| t.anticipated_config.is_some())
        .map(|t| t.id)
        .collect();

    if anticipated_ids.is_empty() {
        return;
    }

    for constraint in &data.generic_constraints {
        for term in &constraint.expression.terms {
            if let VariableRef::ThermalGeneration { thermal_id, .. } = term.variable
                && anticipated_ids.contains(&thermal_id)
            {
                let entity_str = format!("constraint[id={}]", constraint.id.0);
                ctx.add_warning(
                    ErrorKind::SemanticAmbiguity,
                    "constraints/generic_constraints.json",
                    Some(&entity_str),
                    format!(
                        "Constraint \"{}\": thermal_generation({id}) references an \
                             anticipated thermal. thermal_generation refers to the \
                             per-block generation at the delivery stage, not the \
                             forward commitment. If you intend to constrain the \
                             commitment itself, use anticipated_decision({id}) instead.",
                        constraint.name,
                        id = thermal_id.0,
                    ),
                );
            }
        }
    }
}

/// Layer 5a — rejects per-stage thermal bound overrides whose `stage_id` is
/// outside the study horizon `[0, n_stages)`.
///
/// The thermal-bounds resolution table is padded with each plant's base entity
/// values for stages `[n_stages, n_stages + K_max)` to support
/// anticipated-delivery lookups. Overrides in that padded region would be
/// silently dropped by `resolve_bounds` (the `stage_index` only covers study
/// stages); this validator surfaces them as a user-visible error.
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
    /// exactly 2 values_mw all zero in past_anticipated_commitments),
    /// validate_anticipated_thermals returns no errors.
    #[test]
    fn test_valid_anticipated_thermal_ok() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(1),
            values_mw: vec![0.0, 0.0],
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
            values_mw: vec![0.0, 0.0, 0.0, 0.0, 0.0],
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

    // ── AC-7: entry_stage_id window on an anticipated thermal is rejected ─────

    /// An anticipated thermal that sets `entry_stage_id` is rejected with the
    /// blanket "commissioning windows ... are not supported on anticipated
    /// thermals" message. This supersedes the former `entry + K > n_stages`
    /// horizon-edge check: any window on an anticipated thermal is now rejected
    /// outright, regardless of where its edge lands relative to the horizon.
    #[test]
    fn test_entry_window_on_anticipated_thermal_rejected() {
        let thermal = make_anticipated_thermal(1, 3, Some(4), None);
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(1),
            values_mw: vec![0.0, 0.0, 0.0],
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
                    && e.message.contains("not supported on anticipated thermals")
            })
            .collect();
        assert!(
            !relevant.is_empty(),
            "expected BusinessRuleViolation rejecting the window, got: {errors:?}"
        );
        assert_eq!(
            relevant[0].severity,
            crate::validation::Severity::Error,
            "must be an error"
        );
        // The former entry+K>n_stages arithmetic check is removed: no error
        // should carry that wording any more.
        assert!(
            !errors.iter().any(|e| e
                .message
                .contains("entry_stage_id + lead_stages > n_stages")),
            "the superseded entry+K>n_stages check must not fire, got: {errors:?}"
        );
    }

    // ── AC-8: exit_stage_id window on an anticipated thermal is rejected ──────

    /// An anticipated thermal that sets `exit_stage_id` is rejected with the
    /// blanket window-rejection message. This supersedes the former `exit < K`
    /// horizon-edge check.
    #[test]
    fn test_exit_window_on_anticipated_thermal_rejected() {
        let thermal = make_anticipated_thermal(1, 3, None, Some(2));
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(1),
            values_mw: vec![0.0, 0.0, 0.0],
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
                    && e.message.contains("not supported on anticipated thermals")
            })
            .collect();
        assert!(
            !relevant.is_empty(),
            "expected a BusinessRuleViolation rejecting the window, got: {errors:?}"
        );
    }

    // ── Commissioning window on a PLAIN thermal is accepted (applied, not warned) ─

    /// A non-anticipated thermal that sets a commissioning window is accepted:
    /// the window is now applied at the LP fill site (dormant column pinned to
    /// `[0, 0]`), so the anticipated-only rejection must NOT fire and no error
    /// is emitted.
    #[test]
    fn test_window_on_plain_thermal_accepted() {
        let thermal = cobre_core::entities::Thermal {
            entry_stage_id: Some(1),
            exit_stage_id: Some(2),
            ..make_thermal(1, 0.0, 100.0)
        };
        let data = make_data(
            vec![],
            vec![thermal],
            vec![],
            make_stages(vec![0, 1, 2]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "a window on a plain (non-anticipated) thermal must not error, got: {:?}",
            ctx.errors()
        );
        assert!(
            !ctx.errors()
                .iter()
                .any(|e| e.message.contains("not supported on anticipated thermals")),
            "the anticipated-only window rejection must not fire for a plain thermal"
        );
    }

    /// An anticipated thermal with NO commissioning window is accepted: the
    /// rejection is gated on a window being set, so a windowless anticipated
    /// plant passes (regression guard against a blanket reject).
    #[test]
    fn test_anticipated_thermal_without_window_accepted() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(1),
            values_mw: vec![0.0, 0.0],
        };
        let data = make_data_anticipated(vec![thermal], 5, vec![history]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.errors()
                .iter()
                .any(|e| e.message.contains("not supported on anticipated thermals")),
            "a windowless anticipated thermal must not be rejected, got: {:?}",
            ctx.errors()
        );
    }

    // ── AC-9: values_mw[0] outside bounds → bounds error, in-bounds slot passes ─

    /// Given values_mw = [600.0, 200.0] against a plant with max_generation_mw =
    /// 500.0, the validator emits exactly one bounds-violation error for slot 0
    /// (600.0 > 500.0). Slot 1 (200.0) is within [0.0, 500.0] and passes
    /// silently.
    ///
    /// Expected: one BusinessRuleViolation whose message contains
    /// "outside the plant's generation bounds", names "Thermal 3", and
    /// identifies "values_mw[0]".
    #[test]
    fn test_committed_value_out_of_bounds_error() {
        let thermal = make_anticipated_thermal(3, 2, None, None); // min=0.0, max=500.0
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(3),
            values_mw: vec![600.0, 200.0], // slot 0 out-of-bounds, slot 1 in-bounds
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
                    && e.message.contains("outside the plant's generation bounds")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one bounds-violation error (slot 0 only), got: {errors:?}"
        );
        let msg0 = &relevant[0].message;
        assert!(
            msg0.contains("Thermal 3"),
            "message should contain 'Thermal 3', got: {msg0}"
        );
        assert!(
            msg0.contains("values_mw[0]"),
            "message should identify the index [0], got: {msg0}"
        );
        assert!(
            msg0.contains("600"),
            "message should contain the offending value 600, got: {msg0}"
        );
        let file = relevant[0].file.to_string_lossy();
        assert!(
            file.contains("initial_conditions"),
            "file path should reference initial_conditions, got: {file}"
        );
        let entity = relevant[0].entity.as_deref().unwrap_or("");
        assert!(
            entity.contains("thermals[id=3].anticipated_config"),
            "entity should reference the thermal anticipated_config, got: {entity}"
        );
    }

    // ── AC-10: values_mw below min_gen → bounds error for that slot ──────────

    /// Given values_mw = [200.0, 50.0] against a plant with min_generation_mw =
    /// 100.0 and max_generation_mw = 500.0, the validator emits exactly one
    /// bounds-violation error for slot 1 (50.0 < 100.0). Slot 0 (200.0) is
    /// within [100.0, 500.0] and passes silently.
    ///
    /// Expected: one BusinessRuleViolation whose message contains
    /// "outside the plant's generation bounds", names "Thermal 5", and
    /// identifies "values_mw[1]".
    #[test]
    fn test_committed_value_below_min_gen_bounds_error() {
        // Build a thermal with min_mw=100.0.
        let thermal = cobre_core::entities::Thermal {
            anticipated_config: Some(cobre_core::entities::AnticipatedConfig { lead_stages: 2 }),
            ..make_thermal(5, 100.0, 500.0)
        };
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(5),
            values_mw: vec![200.0, 50.0], // slot 0 in-bounds, slot 1 below min
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
                    && e.message.contains("outside the plant's generation bounds")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one bounds-violation error (slot 1 only), got: {errors:?}"
        );
        let msg1 = &relevant[0].message;
        assert!(
            msg1.contains("Thermal 5"),
            "message should contain 'Thermal 5', got: {msg1}"
        );
        assert!(
            msg1.contains("values_mw[1]"),
            "message should identify the index [1], got: {msg1}"
        );
        assert!(
            msg1.contains("50"),
            "message should contain the offending value 50, got: {msg1}"
        );
    }

    // ── AC-11: all values zero — passes (within [min_gen, max_gen]) ──────────

    /// Given values_mw all zero and min_gen = 0.0, no bounds error is emitted
    /// (0.0 is within [0.0, 400.0]).
    #[test]
    fn test_committed_values_all_zero_ok() {
        let thermal = cobre_core::entities::Thermal {
            anticipated_config: Some(cobre_core::entities::AnticipatedConfig { lead_stages: 3 }),
            ..make_thermal(7, 0.0, 400.0)
        };
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(7),
            values_mw: vec![0.0, 0.0, 0.0],
        };
        let data = make_data_anticipated(vec![thermal], 5, vec![history]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "expected no errors for all-zero values_mw, got: {:?}",
            ctx.errors()
        );
    }

    // ── AC-12: boundary — value == min_gen (0.0) is accepted ─────────────────

    /// Given values_mw[0] == 0.0 and min_gen == 0.0, no bounds error is emitted
    /// (the minimum is inclusive).
    #[test]
    fn test_committed_value_zero_accepted() {
        let thermal = cobre_core::entities::Thermal {
            anticipated_config: Some(cobre_core::entities::AnticipatedConfig { lead_stages: 1 }),
            ..make_thermal(9, 0.0, 400.0)
        };
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(9),
            values_mw: vec![0.0],
        };
        let data = make_data_anticipated(vec![thermal], 5, vec![history]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "zero value must be accepted, got: {:?}",
            ctx.errors()
        );
    }

    // ── AC-13: single value above max_generation_mw → bounds error ───────────

    /// Given values_mw[0] = 400.0 against max_generation_mw = 350.0, the
    /// validator emits exactly one bounds-violation error because
    /// 400.0 > 350.0 is outside [100.0, 350.0].
    ///
    /// Expected: one BusinessRuleViolation whose message contains
    /// "outside the plant's generation bounds", names "Thermal 11", and
    /// identifies "values_mw[0]" with value 400.
    #[test]
    fn test_committed_value_above_max_bounds_error() {
        let thermal = cobre_core::entities::Thermal {
            anticipated_config: Some(cobre_core::entities::AnticipatedConfig { lead_stages: 1 }),
            ..make_thermal(11, 100.0, 350.0)
        };
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(11),
            values_mw: vec![400.0], // 400.0 > 350.0 — outside generation bounds
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
                    && e.message.contains("outside the plant's generation bounds")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one bounds-violation error, got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Thermal 11"),
            "message should contain 'Thermal 11', got: {msg}"
        );
        assert!(
            msg.contains("values_mw[0]"),
            "message should identify the index [0], got: {msg}"
        );
        assert!(
            msg.contains("400"),
            "message should contain the offending value 400, got: {msg}"
        );
    }

    // ── AC-14: non-zero in-bounds seed K=1 — accepted ────────────────────────

    /// K=1 case: `values_mw = [204.5647]` against `max_generation_mw = 350.0`.
    /// The value lies within [0.0, 350.0], so the bounds check passes and the
    /// validator emits no errors.
    ///
    /// Expected: ctx.errors() is empty.
    #[test]
    fn test_f3_002_nonzero_values_mw_in_bounds_accepted_k1() {
        let thermal = cobre_core::entities::Thermal {
            anticipated_config: Some(cobre_core::entities::AnticipatedConfig { lead_stages: 1 }),
            ..make_thermal(2, 0.0, 350.0)
        };
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(2),
            values_mw: vec![204.5647], // within [0.0, 350.0] — must be accepted
        };
        let data = make_data_anticipated(vec![thermal], 5, vec![history]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            ctx.errors().is_empty(),
            "expected no errors for in-bounds values_mw, got: {:?}",
            ctx.errors()
        );
    }

    /// K=2 acceptance: two non-zero slots, both within
    /// `[min_generation_mw, max_generation_mw]`, are accepted with zero errors.
    /// The new pre-horizon-seeding contract delivers `values_mw[k]` MW at study
    /// stage `k` as a sunk-cost commitment; in-bounds seeds carry no validator
    /// errors.
    #[test]
    fn test_k2_two_in_bounds_nonzero_values_accepted() {
        let thermal = make_anticipated_thermal(5, 2, None, None); // min=0.0, max=500.0
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(5),
            values_mw: vec![50.0, 30.0], // both within [0, 500]
        };
        let data = make_data_anticipated(vec![thermal], 5, vec![history]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            ctx.errors().is_empty(),
            "expected no errors for in-bounds K=2 values_mw, got: {:?}",
            ctx.errors()
        );
    }

    /// Regression lock: `check_committed_value_bounds` must not emit any
    /// `SemanticAmbiguity` warning for an in-bounds non-zero seed.
    ///
    /// Verifies that the obsolete "non-zero seeds will load but produce the same
    /// dispatch as all-zero seeds" warning has been permanently removed and cannot
    /// be reintroduced silently.
    #[test]
    fn test_nonzero_in_bounds_seed_emits_no_semantic_ambiguity_warning() {
        let thermal = cobre_core::entities::Thermal {
            anticipated_config: Some(cobre_core::entities::AnticipatedConfig { lead_stages: 2 }),
            ..make_thermal(3, 0.0, 350.0)
        };
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(3),
            values_mw: vec![100.0, 200.0], // both within [0.0, 350.0]
        };
        let data = make_data_anticipated(vec![thermal], 5, vec![history]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let all_warnings = ctx.warnings();
        let ambiguity_warnings: Vec<_> = all_warnings
            .iter()
            .filter(|w| {
                w.kind == ErrorKind::SemanticAmbiguity
                    && w.file.to_string_lossy().contains("initial_conditions.json")
            })
            .collect();
        assert!(
            ambiguity_warnings.is_empty(),
            "expected no SemanticAmbiguity warning from initial_conditions.json \
             for an in-bounds non-zero seed, got: {ambiguity_warnings:?}"
        );
    }

    /// K=2 acceptance: mixed in-bounds and out-of-bounds slots. Only the
    /// out-of-bounds slot produces an error; the in-bounds slot (including a
    /// zero slot) passes silently. Verifies that the bounds check is per-slot,
    /// not per-history.
    #[test]
    fn test_k2_mixed_in_bounds_and_out_of_bounds_only_oob_reported() {
        let thermal = make_anticipated_thermal(7, 2, None, None); // min=0.0, max=500.0
        let history = AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(7),
            values_mw: vec![0.0, 600.0], // slot 0 zero (OK); slot 1 above max (OOB)
        };
        let data = make_data_anticipated(vec![thermal], 5, vec![history]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("outside the plant's generation bounds")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one bounds error (slot 1 only), got: {errors:?}"
        );
        assert!(
            relevant[0].message.contains("values_mw[1]"),
            "error must identify slot [1], got: {}",
            relevant[0].message
        );
        assert!(
            relevant[0].message.contains("600"),
            "error must contain value 600, got: {}",
            relevant[0].message
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

    // ── boundary_tests sub-module ─────────────────────────────────────────────
    //
    // Re-organizes the strict-inequality boundary tests into a discoverable
    // named sub-module using the
    // `..._acceptance_boundary` / `..._rejection_boundary` suffix convention.
    // Each test below mirrors a corresponding strict-inequality boundary test
    // (the originals are preserved above to keep the existing test surface
    // stable — DO NOT delete the strict-inequality boundary tests above).
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

    // ── AD-1: anticipated_decision on non-anticipated thermal → hard error ─────

    /// AD-1: A generic constraint that references `anticipated_decision(N)` where
    /// thermal `N` is NOT anticipated produces a `BusinessRuleViolation` naming
    /// the constraint and the thermal ID.
    #[test]
    fn test_anticipated_decision_on_non_anticipated_thermal_error() {
        use cobre_core::{
            ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
            VariableRef,
        };

        let thermal = make_thermal(7, 0.0, 500.0); // NOT anticipated
        let constraint = GenericConstraint {
            id: EntityId::from(1),
            name: "bad_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::AnticipatedDecision {
                        thermal_id: EntityId::from(7),
                    },
                )],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };
        let stage_ids: Vec<i32> = (0..5).collect();
        let mut data = make_data(
            vec![],
            vec![thermal],
            vec![],
            make_stages(stage_ids),
            vec![],
            vec![],
        );
        data.generic_constraints = vec![constraint];

        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            ctx.has_errors(),
            "expected error for non-anticipated thermal"
        );
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
            msg.contains("bad_constraint"),
            "message should contain constraint name, got: {msg}"
        );
        assert!(
            msg.contains('7'),
            "message should contain thermal id 7, got: {msg}"
        );
        assert!(
            msg.contains("not an anticipated thermal"),
            "message should explain the rule, got: {msg}"
        );
        let file = relevant[0].file.to_string_lossy();
        assert!(
            file.contains("generic_constraints.json"),
            "file should reference generic_constraints.json, got: {file}"
        );
    }

    // ── AD-2: anticipated_decision on anticipated thermal → no error ───────────

    /// AD-2: A generic constraint that references `anticipated_decision(N)` where
    /// thermal `N` IS anticipated produces no `BusinessRuleViolation` from the
    /// new validator.
    #[test]
    fn test_anticipated_decision_on_anticipated_thermal_ok() {
        use cobre_core::{
            ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
            VariableRef, entities::AnticipatedConfig,
        };

        let thermal = cobre_core::entities::Thermal {
            anticipated_config: Some(AnticipatedConfig { lead_stages: 2 }),
            ..make_thermal(3, 0.0, 500.0)
        };
        let history = cobre_core::AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(3),
            values_mw: vec![0.0, 0.0],
        };
        let constraint = GenericConstraint {
            id: EntityId::from(10),
            name: "valid_anticipated_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::AnticipatedDecision {
                        thermal_id: EntityId::from(3),
                    },
                )],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };
        let stage_ids: Vec<i32> = (0..5).collect();
        let mut data = make_data(
            vec![],
            vec![thermal],
            vec![],
            make_stages(stage_ids),
            vec![],
            vec![],
        );
        data.initial_conditions.past_anticipated_commitments = vec![history];
        data.generic_constraints = vec![constraint];

        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file
                        .to_string_lossy()
                        .contains("generic_constraints.json")
            })
            .collect();
        assert!(
            relevant.is_empty(),
            "anticipated_decision on an anticipated thermal must not produce a BusinessRuleViolation, got: {relevant:?}"
        );
    }

    // ── TG-1: thermal_generation on anticipated thermal → SemanticAmbiguity ───

    /// TG-1: A generic constraint that references `thermal_generation(N)` where
    /// thermal `N` IS anticipated produces a `SemanticAmbiguity` warning naming
    /// the constraint and the thermal ID with a hint to use `anticipated_decision`.
    #[test]
    fn test_thermal_generation_on_anticipated_thermal_warns() {
        use cobre_core::{
            ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
            VariableRef, entities::AnticipatedConfig,
        };

        let thermal = cobre_core::entities::Thermal {
            anticipated_config: Some(AnticipatedConfig { lead_stages: 1 }),
            ..make_thermal(5, 0.0, 300.0)
        };
        let history = cobre_core::AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(5),
            values_mw: vec![0.0],
        };
        let constraint = GenericConstraint {
            id: EntityId::from(20),
            name: "ambiguous_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::ThermalGeneration {
                        thermal_id: EntityId::from(5),
                        block_id: None,
                    },
                )],
            },
            sense: ConstraintSense::GreaterEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };
        let stage_ids: Vec<i32> = (0..5).collect();
        let mut data = make_data(
            vec![],
            vec![thermal],
            vec![],
            make_stages(stage_ids),
            vec![],
            vec![],
        );
        data.initial_conditions.past_anticipated_commitments = vec![history];
        data.generic_constraints = vec![constraint];

        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);

        // No hard errors from the new validator.
        let errors = ctx.errors();
        let hard: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.file
                    .to_string_lossy()
                    .contains("generic_constraints.json")
            })
            .collect();
        assert!(
            hard.is_empty(),
            "thermal_generation on anticipated thermal must not produce a hard error, got: {hard:?}"
        );

        // Exactly one SemanticAmbiguity warning.
        let warnings = ctx.warnings();
        let relevant: Vec<_> = warnings
            .iter()
            .filter(|w| w.kind == ErrorKind::SemanticAmbiguity)
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one SemanticAmbiguity warning, got: {warnings:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("ambiguous_constraint"),
            "warning should name the constraint, got: {msg}"
        );
        assert!(
            msg.contains('5'),
            "warning should mention thermal id 5, got: {msg}"
        );
        assert!(
            msg.contains("anticipated_decision"),
            "warning should suggest anticipated_decision, got: {msg}"
        );
        let file = relevant[0].file.to_string_lossy();
        assert!(
            file.contains("generic_constraints.json"),
            "file should reference generic_constraints.json, got: {file}"
        );
    }

    // ── TG-2: thermal_generation on non-anticipated thermal → no warning ──────

    /// TG-2: A generic constraint that references `thermal_generation(N)` where
    /// thermal `N` is NOT anticipated produces no `SemanticAmbiguity` warning.
    #[test]
    fn test_thermal_generation_on_non_anticipated_thermal_no_warn() {
        use cobre_core::{
            ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
            VariableRef,
        };

        let thermal = make_thermal(9, 0.0, 200.0); // NOT anticipated
        let constraint = GenericConstraint {
            id: EntityId::from(30),
            name: "plain_thermal_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::ThermalGeneration {
                        thermal_id: EntityId::from(9),
                        block_id: None,
                    },
                )],
            },
            sense: ConstraintSense::GreaterEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };
        let stage_ids: Vec<i32> = (0..5).collect();
        let mut data = make_data(
            vec![],
            vec![thermal],
            vec![],
            make_stages(stage_ids),
            vec![],
            vec![],
        );
        data.generic_constraints = vec![constraint];

        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);

        let warnings = ctx.warnings();
        let relevant: Vec<_> = warnings
            .iter()
            .filter(|w| w.kind == ErrorKind::SemanticAmbiguity)
            .collect();
        assert!(
            relevant.is_empty(),
            "thermal_generation on a non-anticipated thermal must not emit SemanticAmbiguity, got: {relevant:?}"
        );
    }
}
