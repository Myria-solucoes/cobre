//! Layer 5a — thermal-domain semantic validation.
//!
//! Thermal generation bounds (`min_generation_mw <= max_generation_mw`) and
//! anticipated-thermal cross-field invariants.

use std::collections::{HashMap, HashSet};

use cobre_core::commissioning::commissioning_active;
use cobre_core::temporal::Stage;
use cobre_core::{AnticipatedCommitmentHistory, AnticipatedConfig, EntityId, Thermal, VariableRef};
use cobre_stochastic::season_cast::{DatedWindow, StageCalendar};

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
/// 1. **Per-plant lead horizon** — `LeadStages` rejects `K == 0` (defence in
///    depth; parse-time also rejects it) and `K > n_stages`; `LeadTime` rejects
///    `delta_hours` exceeding the summed study-stage durations (strict `>`, so
///    a delivery landing exactly on the final stage is accepted). Either way,
///    the plant can never deliver within the study horizon. A commissioning
///    window IS supported and composes with the lookahead via the shifted
///    decision gate; these checks validate the LEAD itself, independent of any
///    window.
/// 2. **Past-commitments registry bijection** with
///    `ic.past_anticipated_commitments`: each anticipated thermal has at least
///    one commitment window, each window references an anticipated thermal, and
///    the plant's windows tile its leading `lead_delivery_stage_count` delivery
///    stages exactly (coverage `1.0`, no gap, no overlap, none beyond the
///    horizon) via the shared [`StageCalendar`] resolver — a hard gate, no
///    fallback (the "Pre-study anticipated commitments: calendar-derived
///    coverage" contract).
/// 3. **Committed-value generation bounds** — see `check_committed_value_bounds`.
/// 4. **Seed-vs-window consistency** — see `check_seed_within_window`.
pub(super) fn check_anticipated_thermals(data: &ParsedData, ctx: &mut ValidationContext) {
    // Study stages (id >= 0) are the contiguous suffix of the id-sorted stage
    // list; pre-study stages (negative IDs) are never delivery targets.
    let study_stages: &[Stage] = match data.stages.stages.iter().position(|s| s.id >= 0) {
        Some(idx) => &data.stages.stages[idx..],
        None => &[],
    };
    let study_stage_ids: Vec<i32> = study_stages.iter().map(|s| s.id).collect();
    let n_stages = study_stage_ids.len();
    let study_durations = study_stage_durations(data);

    for thermal in &data.thermals {
        let Some(ref cfg) = thermal.anticipated_config else {
            continue;
        };
        let thermal_id = thermal.id.0;

        if let AnticipatedConfig::LeadTime(delta_hours) = *cfg {
            let total_horizon_hours: f64 = study_durations.iter().sum();
            if delta_hours > total_horizon_hours {
                let entity_str = format!("thermals[id={thermal_id}].anticipated_config.lead_time");
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "system/thermals.json",
                    Some(&entity_str),
                    format!(
                        "Thermal {thermal_id}: lead_time exceeds study horizon \
                         (lead_time={delta_hours}, total_horizon_hours={total_horizon_hours}); \
                         the plant can never deliver within the study horizon"
                    ),
                );
            }
        }

        let Some(k) = cfg.lead_stages() else {
            continue;
        };
        let entity_str = format!("thermals[id={thermal_id}].anticipated_config.lead_stages");

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
    }

    let ic = &data.initial_conditions;
    let mut windows_by_id: HashMap<EntityId, Vec<&AnticipatedCommitmentHistory>> = HashMap::new();
    for history in &ic.past_anticipated_commitments {
        windows_by_id
            .entry(history.thermal_id)
            .or_default()
            .push(history);
    }

    // The resolver is only consulted for a thermal that carries commitment
    // windows; skip building it (and its ordered-calendar precondition) when
    // there are none to validate.
    let calendar = (!windows_by_id.is_empty()).then(|| StageCalendar::new(study_stages));

    let anticipated_thermal_ids: HashSet<EntityId> = data
        .thermals
        .iter()
        .filter(|t| t.anticipated_config.is_some())
        .map(|t| t.id)
        .collect();

    for thermal in &data.thermals {
        let Some(ref cfg) = thermal.anticipated_config else {
            continue;
        };
        let thermal_id = thermal.id;

        match windows_by_id.get(&thermal_id) {
            None => {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "initial_conditions.json",
                    Some("initial_conditions.past_anticipated_commitments"),
                    format!(
                        "Thermal {}: missing entry in initial_conditions.past_anticipated_commitments; \
                         every anticipated thermal must have at least one commitment window",
                        thermal_id.0
                    ),
                );
            }
            Some(records) => {
                let Some(calendar) = calendar.as_ref() else {
                    continue;
                };
                let k_i = lead_delivery_stage_count(*cfg, &study_durations, n_stages);
                if check_commitment_coverage(
                    thermal_id,
                    records,
                    calendar,
                    k_i,
                    &study_stage_ids,
                    ctx,
                ) {
                    check_committed_value_bounds(thermal, thermal_id, records, ctx);
                    check_seed_within_window(
                        thermal,
                        thermal_id,
                        records,
                        calendar,
                        &study_stage_ids,
                        ctx,
                    );
                }
            }
        }
    }

    let mut reported: HashSet<EntityId> = HashSet::new();
    for history in &ic.past_anticipated_commitments {
        if !anticipated_thermal_ids.contains(&history.thermal_id)
            && reported.insert(history.thermal_id)
        {
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

/// Advisory (`ModelQuality`): a `lead_stages`-configured thermal whose active
/// window — decision stage `t` through delivery `t + lead_stages`, `t` ranging
/// over the plant's commissioning window and the delivery side clamped to the
/// study horizon — spans a pair of adjacent study stages with differing
/// durations. A fixed stage-count lead delivers a different physical lead on
/// each side of such a cadence change; `anticipated_config.lead_time` anchors
/// the lead to physical hours instead and is immune to this. Never a hard error.
pub(super) fn check_anticipated_cadence_transition(data: &ParsedData, ctx: &mut ValidationContext) {
    let study_durations = study_stage_durations(data);
    let n_stages = study_durations.len();
    if n_stages < 2 {
        return;
    }

    for thermal in &data.thermals {
        let Some(cfg) = thermal.anticipated_config else {
            continue;
        };
        let Some(k) = cfg.lead_stages() else {
            continue;
        };
        let thermal_id = thermal.id.0;
        let k_u = usize::try_from(k).unwrap_or(usize::MAX);

        let decision_start = usize::try_from(thermal.entry_stage_id.unwrap_or(0).max(0))
            .unwrap_or(0)
            .min(n_stages - 1);
        let decision_end_id = thermal.exit_stage_id.map_or_else(
            || i32::try_from(n_stages - 1).unwrap_or(i32::MAX),
            |exit| exit - 1,
        );
        if decision_end_id < 0 {
            continue;
        }
        let decision_end = usize::try_from(decision_end_id)
            .unwrap_or(0)
            .min(n_stages - 1);
        if decision_start > decision_end {
            continue;
        }
        let window_end = decision_end.saturating_add(k_u).min(n_stages - 1);

        let entity_str = format!("thermals[id={thermal_id}].anticipated_config.lead_stages");
        for i in decision_start..window_end {
            let (prev, next) = (study_durations[i], study_durations[i + 1]);
            if (prev - next).abs() > 1e-9 {
                ctx.add_warning(
                    ErrorKind::ModelQuality,
                    "system/thermals.json",
                    Some(&entity_str),
                    format!(
                        "Thermal {thermal_id}: anticipated_config.lead_stages={k} active window \
                         spans a stage-cadence transition between stage {i} ({prev}h) and stage \
                         {} ({next}h); a fixed stage-count lead delivers a different physical \
                         lead on each side of the transition. Consider anticipated_config.lead_time, \
                         which anchors the lead to physical hours instead of a stage count.",
                        i + 1
                    ),
                );
                break;
            }
        }
    }
}

/// Study-stage (`id >= 0`) durations in canonical (ascending `id`) order, each
/// summed from its blocks. Computed independently of
/// `travel_time::study_stage_durations` (same shape, own walk) rather than
/// shared across files.
fn study_stage_durations(data: &ParsedData) -> Vec<f64> {
    data.stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.blocks.iter().map(|b| b.duration_hours).sum())
        .collect()
}

/// Calendar-derived count of leading pre-study-committed delivery stages:
/// `LeadStages(l)` clamps `l` to `n_stages`; `LeadTime(delta)` counts the
/// leading study stages whose stage-end cumulative hours are `<= delta`
/// (tie-inclusive). Both deciders are monotonic in stage-end cumulative hours,
/// so the pre-study run is always the leading prefix `0..k`. Computed
/// independently of the solver crate's point-commitment resolver (cobre-io is
/// upstream and cannot depend on it), mirroring `check_defluence_coverage`'s
/// own calendar walk (`validation/semantic/travel_time.rs`). Do not shortcut
/// `LeadTime` to a stage count from a window length — on a non-uniform calendar
/// the cumulative-hours walk and a bare length diverge.
fn lead_delivery_stage_count(
    mode: AnticipatedConfig,
    study_durations: &[f64],
    n_stages: usize,
) -> usize {
    match mode {
        AnticipatedConfig::LeadStages(lead_stages) => (lead_stages as usize).min(n_stages),
        AnticipatedConfig::LeadTime(delta_hours) => {
            let mut cumulative_hours = 0.0_f64;
            let mut count = 0usize;
            for &duration in study_durations.iter().take(n_stages) {
                cumulative_hours += duration;
                if cumulative_hours > delta_hours {
                    break;
                }
                count += 1;
            }
            count
        }
    }
}

/// A thermal's commitment windows must tile its leading `k_i` delivery stages
/// exactly: every leading stage covered at fraction `1.0` (via
/// [`StageCalendar::covers_exactly`]), and no window reaching any stage at or
/// beyond `k_i`. Emits a named `BusinessRuleViolation` for an uncovered leading
/// stage (gap) or a stage covered beyond the horizon (over-coverage); overlap
/// is rejected earlier by the shared windowed-record validator. Returns whether
/// coverage is exact, gating the per-window bounds/commissioning checks.
#[allow(clippy::float_cmp)] // whole-day-hours coverage keeps the per-stage tiling fraction bit-exact (mirrors covers_exactly)
fn check_commitment_coverage(
    thermal_id: EntityId,
    records: &[&AnticipatedCommitmentHistory],
    calendar: &StageCalendar,
    k_i: usize,
    study_stage_ids: &[i32],
    ctx: &mut ValidationContext,
) -> bool {
    let windows: Vec<DatedWindow> = records
        .iter()
        .map(|r| DatedWindow {
            start_date: r.start_date,
            end_date: r.end_date,
        })
        .collect();

    let mut per_stage = vec![0.0_f64; study_stage_ids.len()];
    for window in &windows {
        for (total, fraction) in per_stage.iter_mut().zip(calendar.coverage(window)) {
            *total += fraction;
        }
    }

    let entity_str = format!("thermals[id={}].anticipated_config", thermal_id.0);
    let mut valid = true;

    if !calendar.covers_exactly(&windows, k_i) {
        let uncovered: Vec<i32> = (0..k_i)
            .filter(|&i| per_stage[i] != 1.0)
            .map(|i| study_stage_ids[i])
            .collect();
        ctx.add_error(
            ErrorKind::BusinessRuleViolation,
            "initial_conditions.json",
            Some(&entity_str),
            format!(
                "Thermal {}: past_anticipated_commitments do not tile the leading {k_i} \
                 delivery stage(s) at coverage 1.0; study stage id(s) {uncovered:?} are not \
                 covered exactly once. Write a commitment window (a committed 0 MW is explicit) \
                 for every leading delivery stage.",
                thermal_id.0
            ),
        );
        valid = false;
    }

    let over_covered: Vec<i32> = (k_i..study_stage_ids.len())
        .filter(|&i| per_stage[i] != 0.0)
        .map(|i| study_stage_ids[i])
        .collect();
    if !over_covered.is_empty() {
        ctx.add_error(
            ErrorKind::BusinessRuleViolation,
            "initial_conditions.json",
            Some(&entity_str),
            format!(
                "Thermal {}: past_anticipated_commitments cover study stage id(s) {over_covered:?} \
                 beyond the leading {k_i} calendar-derived delivery stage(s); commitment windows \
                 must not extend past the pre-study delivery horizon.",
                thermal_id.0
            ),
        );
        valid = false;
    }

    valid
}

/// Every window's `value_mw` must lie within
/// `[min_generation_mw, max_generation_mw]`; an out-of-bounds rate makes the LP
/// infeasible at every covered stage's fishing equality.
fn check_committed_value_bounds(
    thermal: &Thermal,
    thermal_id: EntityId,
    records: &[&AnticipatedCommitmentHistory],
    ctx: &mut ValidationContext,
) {
    let min_mw = thermal.min_generation_mw;
    let max_mw = thermal.max_generation_mw;
    let entity_str = format!("thermals[id={}].anticipated_config", thermal_id.0);
    for record in records {
        let v = record.value_mw;
        if v < min_mw || v > max_mw {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "initial_conditions.json",
                Some(&entity_str),
                format!(
                    "Thermal {}: past_anticipated_commitments window [{}, {}) value_mw = {v} \
                     is outside the plant's generation bounds [{min_mw}, {max_mw}]; \
                     the LP delivery equality on the covered stage(s) cannot be \
                     satisfied and the LP will be infeasible",
                    thermal_id.0, record.start_date, record.end_date
                ),
            );
        }
    }
}

/// For a windowed anticipated thermal, reject any nonzero `value_mw` maturing at
/// a covered study stage outside the operation window: the matured generation
/// column is pinned to `[0, 0]` there, so the always-active fishing equality
/// reads `0 == seed` — an infeasible LP. A zero rate is consistent at any stage
/// and allowed.
///
/// The window predicate is the LP builder's own
/// `cobre_core::commissioning::commissioning_active`, so validation and the
/// builder cannot drift on what an infeasible seed is. Each window's covered
/// study stages are resolved through the shared [`StageCalendar`], so a window
/// straddling the commissioning boundary is checked stage-by-stage.
#[allow(clippy::float_cmp)] // day-aligned coverage is exactly 0 or nonzero per stage; the zero-rate skip is a bit-exact test
fn check_seed_within_window(
    thermal: &Thermal,
    thermal_id: EntityId,
    records: &[&AnticipatedCommitmentHistory],
    calendar: &StageCalendar,
    study_stage_ids: &[i32],
    ctx: &mut ValidationContext,
) {
    let entry = thermal.entry_stage_id;
    let exit = thermal.exit_stage_id;
    if entry.is_none() && exit.is_none() {
        return;
    }
    let entity_str = format!("thermals[id={}].anticipated_config", thermal_id.0);
    for record in records {
        let v = record.value_mw;
        if v == 0.0 {
            continue;
        }
        let window = DatedWindow {
            start_date: record.start_date,
            end_date: record.end_date,
        };
        for (i, fraction) in calendar.coverage(&window).into_iter().enumerate() {
            if fraction == 0.0 {
                continue;
            }
            let stage_id = study_stage_ids[i];
            if !commissioning_active(entry, exit, stage_id) {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "initial_conditions.json",
                    Some(&entity_str),
                    format!(
                        "Thermal {}: past_anticipated_commitments window [{}, {}) value_mw = {v} \
                         matures at study stage id {stage_id}, which is outside the \
                         plant's commissioning window [entry={entry:?}, exit={exit:?}); \
                         the matured generation column is pinned to [0, 0] there, so the \
                         fishing equality reads 0 == {v} and the LP is infeasible. \
                         Commit a zero rate at this stage or widen the window.",
                        thermal_id.0, record.start_date, record.end_date
                    ),
                );
            }
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
/// generation at the *delivery* stage, not the *commitment* made at the current
/// stage. This is valid but surprising; to constrain the commitment use
/// `anticipated_decision(N)` instead. Emits a `SemanticAmbiguity` warning.
pub(super) fn warn_thermal_generation_on_anticipated_thermal(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
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
    // Study stages only (id >= 0), matching the resolver's `stage_index`; a
    // mismatch here would reject overrides the resolver actually applies.
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
    use cobre_core::temporal::{Block, PolicyGraphType, Stage};
    use cobre_core::{
        AnticipatedCommitmentHistory, AnticipatedConfig, EntityId, HorizonGraph, Thermal,
    };

    use chrono::{NaiveDate, TimeDelta};

    use super::super::test_support::*;
    use super::super::validate_semantic_hydro_thermal;
    use super::lead_delivery_stage_count;
    use crate::stages::StagesData;
    use crate::validation::schema::ParsedData;
    use crate::validation::{ErrorKind, ValidationContext};

    // ── Helper: build a thermal with anticipated_config ───────────────────────

    fn make_anticipated_thermal(
        id: i32,
        lead_stages: u32,
        entry_stage_id: Option<i32>,
        exit_stage_id: Option<i32>,
    ) -> Thermal {
        Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(lead_stages)),
            entry_stage_id,
            exit_stage_id,
            ..make_thermal(id, 0.0, 500.0)
        }
    }

    /// Build a `LeadTime`-configured thermal.
    fn make_lead_time_anticipated_thermal(
        id: i32,
        lead_time_hours: f64,
        entry_stage_id: Option<i32>,
        exit_stage_id: Option<i32>,
    ) -> Thermal {
        Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadTime(lead_time_hours)),
            entry_stage_id,
            exit_stage_id,
            ..make_thermal(id, 0.0, 500.0)
        }
    }

    /// Anchor of every anticipated-thermal test calendar; stage 0 starts here.
    fn calendar_anchor() -> NaiveDate {
        NaiveDate::from_ymd_opt(2024, 1, 1).unwrap()
    }

    /// Contiguous stage boundary dates for a calendar whose stage `i` spans
    /// `durations_hours[i] / 24` whole days from [`calendar_anchor`]; length is
    /// `durations_hours.len() + 1`. Every fixture duration is a whole-day
    /// multiple so the date span and block hours agree, the alignment
    /// [`StageCalendar`] coverage needs.
    fn stage_boundaries(durations_hours: &[f64]) -> Vec<NaiveDate> {
        let mut cursor = calendar_anchor();
        let mut dates = vec![cursor];
        for &hours in durations_hours {
            cursor += TimeDelta::days((hours / 24.0) as i64);
            dates.push(cursor);
        }
        dates
    }

    /// Contiguous study stages (ids `0..durations_hours.len()`), stage `i` a
    /// single block of `durations_hours[i]` over `[boundary[i], boundary[i+1])`.
    fn contiguous_stages(durations_hours: &[f64]) -> Vec<Stage> {
        let boundaries = stage_boundaries(durations_hours);
        durations_hours
            .iter()
            .enumerate()
            .map(|(i, &hours)| {
                let mut stage = make_stage(i as i32);
                stage.index = i;
                stage.start_date = boundaries[i];
                stage.end_date = boundaries[i + 1];
                stage.blocks = vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: hours,
                }];
                stage
            })
            .collect()
    }

    /// One commitment window per value, tiling the leading `values.len()` stages
    /// of the contiguous calendar defined by `durations_hours`. `values.len()`
    /// may differ from `durations_hours.len()` (fewer → gap; more → beyond the
    /// horizon) to build the coverage-failure fixtures.
    fn commitments_on(
        thermal_id: i32,
        values: &[f64],
        durations_hours: &[f64],
    ) -> Vec<AnticipatedCommitmentHistory> {
        let boundaries = stage_boundaries(durations_hours);
        values
            .iter()
            .enumerate()
            .map(|(i, &value_mw)| AnticipatedCommitmentHistory {
                thermal_id: EntityId::from(thermal_id),
                start_date: boundaries[i],
                end_date: boundaries[i + 1],
                value_mw,
            })
            .collect()
    }

    /// One commitment window per value on the uniform 30-day calendar
    /// [`make_data_anticipated`] builds — value `i` delivers over study stage
    /// `i`.
    fn commitments(thermal_id: i32, values: &[f64]) -> Vec<AnticipatedCommitmentHistory> {
        commitments_on(thermal_id, values, &vec![720.0; values.len().max(1)])
    }

    /// Assemble `ParsedData` from a contiguous stage calendar and commitments.
    fn anticipated_data(
        thermals: Vec<Thermal>,
        durations_hours: &[f64],
        past_anticipated_commitments: Vec<AnticipatedCommitmentHistory>,
    ) -> ParsedData {
        let stages_data = StagesData {
            openings_declared: std::collections::HashSet::new(),
            stages: contiguous_stages(durations_hours),
            policy_graph: HorizonGraph {
                stage_discount_rate_overrides: std::collections::HashMap::new(),
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                nodes: Vec::new(),
                season_map: None,
            },
        };
        let mut data = make_data(vec![], thermals, vec![], stages_data, vec![], vec![]);
        data.initial_conditions.past_anticipated_commitments = past_anticipated_commitments;
        data
    }

    /// Build `ParsedData` for anticipated-thermal tests on a non-uniform
    /// per-stage calendar (`durations_hours`, one entry per study stage,
    /// `id = 0..durations_hours.len()`).
    fn make_data_anticipated_with_durations(
        thermals: Vec<Thermal>,
        durations_hours: &[f64],
        past_anticipated_commitments: Vec<AnticipatedCommitmentHistory>,
    ) -> ParsedData {
        anticipated_data(thermals, durations_hours, past_anticipated_commitments)
    }

    /// Build `ParsedData` for anticipated-thermal tests on a uniform 30-day
    /// (720 h) calendar of `n_stages` stages with IDs `0..n_stages`.
    fn make_data_anticipated(
        thermals: Vec<Thermal>,
        n_stages: usize,
        past_anticipated_commitments: Vec<AnticipatedCommitmentHistory>,
    ) -> ParsedData {
        anticipated_data(
            thermals,
            &vec![720.0; n_stages],
            past_anticipated_commitments,
        )
    }

    // ── AC-1: valid anticipated thermal — returns Ok(()) ─────────────────────

    /// Given a valid anticipated thermal (lead_stages=2, n_stages=5, no entry/exit,
    /// two all-zero commitment windows tiling stages 0..2), semantic validation
    /// returns no errors.
    #[test]
    fn test_valid_anticipated_thermal_ok() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[0.0, 0.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "expected no errors, got: {:?}",
            ctx.errors()
        );
    }

    // ── LeadTime: no longer gated `NotImplemented` ────────────────────────────

    /// Given an otherwise-valid `LeadTime`-configured thermal on the
    /// weekly-then-monthly PMO calendar (calendar-covered commitments), when
    /// semantic validation runs, then no `NotImplemented` error is produced and
    /// the study passes validation entirely: `LeadTime` falls through to the
    /// shared coverage/bounds/window checks unchanged.
    #[test]
    fn test_lead_time_thermal_accepted() {
        let thermal = make_lead_time_anticipated_thermal(1, 720.0, None, None);
        let durations = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0];
        let data = make_data_anticipated_with_durations(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0, 0.0, 0.0, 0.0], &durations),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::NotImplemented),
            "LeadTime must no longer be rejected as NotImplemented, got: {:?}",
            ctx.errors()
        );
        assert!(
            !ctx.has_errors(),
            "a valid LeadTime thermal with calendar-covered commitments must pass \
             validation entirely, got: {:?}",
            ctx.errors()
        );
    }

    /// Given a valid single-decider `LeadTime(744.0)` thermal on a uniform
    /// 3×744h calendar with a matching one-entry `past_anticipated_commitments`
    /// history, when semantic validation runs, then the study passes validation
    /// with no errors — the first `LeadTime` config the gate admits through to
    /// setup.
    #[test]
    fn lead_time_single_decider_passes_validation() {
        let thermal = make_lead_time_anticipated_thermal(1, 744.0, None, None);
        let durations = [744.0, 744.0, 744.0];
        let data = make_data_anticipated_with_durations(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0], &durations),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "a valid single-decider LeadTime thermal must pass validation, got: {:?}",
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
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[100.0, 200.0]));
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

    // ── AC-4: windows beyond the lead horizon (over-coverage) ────────────────

    /// Given lead_stages=2 but three tiling windows (covering stages 0..3), the
    /// window on stage 2 lies beyond the leading K=2 delivery horizon and is
    /// rejected with a named over-coverage error.
    #[test]
    fn test_over_coverage_beyond_lead_horizon_rejected() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[100.0, 200.0, 300.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("beyond the leading")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one over-coverage error, got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains('2'),
            "message should name over-covered study stage id 2, got: {msg}"
        );
    }

    // ── AC-5: gap in the lead horizon (under-coverage) ───────────────────────

    /// Given lead_stages=2 but only one tiling window (covering stage 0), study
    /// stage 1 is uncovered and rejected with a named under-coverage error.
    #[test]
    fn test_under_coverage_gap_in_lead_horizon_rejected() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[100.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("do not tile the leading")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one under-coverage error, got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains('1'),
            "message should name the uncovered study stage id 1, got: {msg}"
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
        let data =
            make_data_anticipated(vec![thermal], 5, commitments(1, &[0.0, 0.0, 0.0, 0.0, 0.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "lead_stages == n_stages must be accepted, got: {:?}",
            ctx.errors()
        );
    }

    // ── lead_time exceeds study horizon ───────────────────────────────────────

    /// Given a `LeadTime(delta_hours)` thermal whose `delta_hours` strictly
    /// exceeds the summed study-stage durations, when semantic validation runs,
    /// then exactly one `BusinessRuleViolation` is appended naming the thermal
    /// id, `delta_hours`, and the total horizon hours.
    #[test]
    fn test_lead_time_exceeds_study_horizon_error() {
        let thermal = make_lead_time_anticipated_thermal(1, 3000.0, None, None);
        let data = make_data_anticipated_with_durations(
            vec![thermal],
            &[168.0, 168.0, 168.0, 168.0, 720.0, 720.0],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("lead_time exceeds study horizon")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one BusinessRuleViolation with 'lead_time exceeds study horizon', got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Thermal 1"),
            "message should contain 'Thermal 1', got: {msg}"
        );
        assert!(
            msg.contains("3000"),
            "message should contain delta_hours 3000, got: {msg}"
        );
        assert!(
            msg.contains("2112"),
            "message should contain the total horizon hours 2112, got: {msg}"
        );
    }

    // ── Boundary: lead_time == total horizon is accepted (strict-greater check) ──

    /// Boundary case: `delta_hours == total_horizon_hours` must NOT error. The
    /// horizon check is `delta_hours > total_horizon_hours` (strict), mirroring
    /// the `LeadStages` boundary — a delivery landing exactly on the final study
    /// stage is a valid configuration.
    #[test]
    fn test_lead_time_equal_total_horizon_ok() {
        let thermal = make_lead_time_anticipated_thermal(1, 2112.0, None, None);
        let durations = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0];
        let data = make_data_anticipated_with_durations(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &durations),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "delta_hours == total_horizon_hours must be accepted, got: {:?}",
            ctx.errors()
        );
    }

    // ── Cadence-transition advisory (check_anticipated_cadence_transition) ────

    /// Given a `lead_stages=2` thermal whose active window spans a
    /// weekly (168h) -> monthly (744h) stage-cadence transition, when
    /// semantic validation runs, then exactly one advisory is produced naming
    /// the thermal and citing `lead_time`, and validation still succeeds.
    #[test]
    fn lead_stages_cadence_transition_emits_advisory() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let durations = [168.0, 168.0, 168.0, 744.0, 744.0];
        let data = make_data_anticipated_with_durations(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0, 0.0], &durations),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "a cadence transition is advisory-only, never an error; got: {:?}",
            ctx.errors()
        );
        let relevant: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| w.kind == ErrorKind::ModelQuality && w.message.contains("Thermal 1"))
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one cadence-transition advisory, got: {:?}",
            ctx.warnings()
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("lead_time"),
            "advisory must cite lead_time as the physically-anchored alternative, got: {msg}"
        );
        assert!(
            msg.contains("stage 2") && msg.contains("stage 3"),
            "advisory must cite the transition stage pair, got: {msg}"
        );
    }

    /// Given a `lead_stages=2` thermal whose active window spans only
    /// equal-duration (uniform) stages, when semantic validation runs, then no
    /// cadence-transition advisory is produced.
    #[test]
    fn lead_stages_uniform_calendar_no_advisory() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let durations = [744.0, 744.0, 744.0, 744.0, 744.0];
        let data = make_data_anticipated_with_durations(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0, 0.0], &durations),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "a uniform calendar is valid, got: {:?}",
            ctx.errors()
        );
        assert!(
            !ctx.warnings()
                .iter()
                .any(|w| w.kind == ErrorKind::ModelQuality && w.message.contains("cadence")),
            "a uniform calendar must not trigger a cadence-transition advisory, got: {:?}",
            ctx.warnings()
        );
    }

    // ── lead_delivery_stage_count: direct unit tests ─────────────────────────

    /// Weekly-then-monthly PMO calendar `[168,168,168,168,720,720]` h with
    /// `LeadTime(720.0)`: the leading count is 4 (`S_{m+1} <= 720` holds for
    /// `m = 0..=3` only; `S_5 = 1392 > 720`).
    #[test]
    fn test_lead_delivery_stage_count_lead_time_pmo_calendar() {
        let durations = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0];
        let count = lead_delivery_stage_count(
            AnticipatedConfig::LeadTime(720.0),
            &durations,
            durations.len(),
        );
        assert_eq!(count, 4);
    }

    /// Uniform 5x720h calendar with `LeadTime(1440.0)`: the count is 2
    /// (`S_{m+1} <= 1440` holds for `m = 0, 1` only; tie-inclusive at `m = 1`).
    #[test]
    fn test_lead_delivery_stage_count_lead_time_uniform_calendar() {
        let durations = [720.0; 5];
        let count = lead_delivery_stage_count(
            AnticipatedConfig::LeadTime(1440.0),
            &durations,
            durations.len(),
        );
        assert_eq!(count, 2);
    }

    /// `LeadStages` ignores durations entirely and clamps to `n_stages`.
    #[test]
    fn test_lead_delivery_stage_count_lead_stages_clamped() {
        let count = lead_delivery_stage_count(AnticipatedConfig::LeadStages(2), &[720.0; 5], 5);
        assert_eq!(count, 2);
    }

    // ── Calendar-derived coverage: LeadTime on a non-uniform calendar ────────

    /// Given a `LeadTime(720.0)` plant on the weekly-then-monthly PMO calendar
    /// `[168,168,168,168,720,720]` h whose four windows tile the leading four
    /// delivery stages exactly, no coverage error fires.
    #[test]
    fn test_anticipated_lead_time_coverage_pmo_calendar() {
        let thermal = make_lead_time_anticipated_thermal(1, 720.0, None, None);
        let durations = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0];
        let data = make_data_anticipated_with_durations(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0, 0.0, 0.0, 0.0], &durations),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let coverage_errors: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && (e.message.contains("do not tile the leading")
                        || e.message.contains("beyond the leading"))
            })
            .collect();
        assert!(
            coverage_errors.is_empty(),
            "expected no coverage error, got: {errors:?}"
        );
    }

    /// Given the same PMO calendar and `LeadTime(720.0)` plant but windows
    /// tiling only the leading two of the four calendar-derived delivery stages,
    /// exactly one `BusinessRuleViolation` fires naming the uncovered stages.
    #[test]
    fn test_anticipated_lead_time_coverage_pmo_calendar_under_coverage_rejected() {
        let thermal = make_lead_time_anticipated_thermal(1, 720.0, None, None);
        let durations = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0];
        let data = make_data_anticipated_with_durations(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0, 0.0], &durations),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let coverage_errors: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("do not tile the leading")
            })
            .collect();
        assert_eq!(
            coverage_errors.len(),
            1,
            "expected exactly one under-coverage error, got: {errors:?}"
        );
        let msg = &coverage_errors[0].message;
        assert!(
            msg.contains('2') && msg.contains('3'),
            "message should name the uncovered study stage ids 2 and 3, got: {msg}"
        );
    }

    // ── AC-7: entry_stage_id window on an anticipated thermal is ACCEPTED ─────

    /// An anticipated thermal that sets `entry_stage_id` is accepted: the window
    /// composes with the K-stage delivery lookahead via the shifted decision gate.
    /// With all-zero seeds the seed-vs-window rule does not fire (a zero seed is
    /// consistent with the `[0, 0]` pin at a dormant stage).
    #[test]
    fn test_entry_window_on_anticipated_thermal_accepted() {
        let thermal = make_anticipated_thermal(1, 3, Some(4), None);
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[0.0, 0.0, 0.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "an entry window on an anticipated thermal must be accepted, got: {:?}",
            ctx.errors()
        );
        assert!(
            !ctx.errors()
                .iter()
                .any(|e| e.message.contains("not supported on anticipated thermals")),
            "the removed window rejection must not fire"
        );
    }

    // ── AC-8: exit_stage_id window on an anticipated thermal is ACCEPTED ──────

    /// An anticipated thermal that sets `exit_stage_id` is accepted. With all-zero
    /// seeds the seed-vs-window rule does not fire.
    #[test]
    fn test_exit_window_on_anticipated_thermal_accepted() {
        let thermal = make_anticipated_thermal(1, 3, None, Some(2));
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[0.0, 0.0, 0.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "an exit window on an anticipated thermal must be accepted, got: {:?}",
            ctx.errors()
        );
        assert!(
            !ctx.errors()
                .iter()
                .any(|e| e.message.contains("not supported on anticipated thermals")),
            "the removed window rejection must not fire"
        );
    }

    // ── Seed-vs-window: nonzero seed maturing outside the window is rejected ──

    /// A windowed anticipated thermal whose nonzero seed matures OUTSIDE the
    /// operation window is rejected. K=2, n_stages=5, commissioning window
    /// `[entry=2, exit=4)`, commitment windows `[100.0, 0.0]` over study stages
    /// 0 and 1: the 100 MW window delivers at study stage id 0, before
    /// `entry=2`, so `commissioning_active(2, 4, 0) == false` and the nonzero
    /// rate forces an infeasible `0 == 100` fishing equality. The 0 MW window is
    /// consistent at any stage and emits no error.
    #[test]
    fn test_nonzero_seed_outside_window_rejected() {
        let thermal = make_anticipated_thermal(1, 2, Some(2), Some(4));
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[100.0, 0.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message
                        .contains("outside the plant's commissioning window")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one seed-vs-window error (stage 0 only), got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("study stage id 0") && msg.contains("Thermal 1"),
            "error must identify study stage 0 of Thermal 1, got: {msg}"
        );
    }

    /// A windowed anticipated thermal whose nonzero seeds all mature INSIDE the
    /// window is accepted. K=1, n_stages=5, commissioning window
    /// `[entry=0, exit=5)`, a single 50 MW window over study stage id 0 ∈
    /// `[0, 5)`, so `commissioning_active(0, 5, 0) == true` and no seed-vs-window
    /// error fires.
    #[test]
    fn test_nonzero_seed_inside_window_accepted() {
        let thermal = make_anticipated_thermal(1, 1, Some(0), Some(5));
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[50.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.errors().iter().any(|e| e
                .message
                .contains("outside the plant's commissioning window")),
            "an in-window nonzero seed must not be rejected, got: {:?}",
            ctx.errors()
        );
    }

    // ── Commissioning window on a PLAIN thermal is accepted (applied, not warned) ─

    /// A non-anticipated thermal that sets a commissioning window is accepted:
    /// the window is applied at the LP fill site (dormant column pinned to
    /// `[0, 0]`), so the anticipated-only rejection must NOT fire and no error
    /// is emitted.
    #[test]
    fn test_window_on_plain_thermal_accepted() {
        let thermal = Thermal {
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
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[0.0, 0.0]));
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

    // ── AC-9: a window value above bounds → bounds error, in-bounds window passes ─

    /// Given windows `[600.0, 200.0]` against a plant with max_generation_mw =
    /// 500.0, the validator emits exactly one bounds-violation error for the
    /// 600 MW window (600.0 > 500.0). The 200 MW window is within [0.0, 500.0]
    /// and passes silently.
    ///
    /// Expected: one BusinessRuleViolation whose message contains
    /// "outside the plant's generation bounds", names "Thermal 3", and the
    /// offending window.
    #[test]
    fn test_committed_value_out_of_bounds_error() {
        let thermal = make_anticipated_thermal(3, 2, None, None); // min=0.0, max=500.0
        let data = make_data_anticipated(vec![thermal], 5, commitments(3, &[600.0, 200.0]));
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
            "expected exactly one bounds-violation error (600 MW window only), got: {errors:?}"
        );
        let msg0 = &relevant[0].message;
        assert!(
            msg0.contains("Thermal 3"),
            "message should contain 'Thermal 3', got: {msg0}"
        );
        assert!(
            msg0.contains("2024-01-01"),
            "message should identify the offending window by its start date, got: {msg0}"
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

    // ── AC-10: a window value below min_gen → bounds error for that window ────

    /// Given windows `[200.0, 50.0]` against a plant with min_generation_mw =
    /// 100.0 and max_generation_mw = 500.0, the validator emits exactly one
    /// bounds-violation error for the 50 MW window (50.0 < 100.0). The 200 MW
    /// window is within [100.0, 500.0] and passes silently.
    #[test]
    fn test_committed_value_below_min_gen_bounds_error() {
        // Build a thermal with min_mw=100.0.
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
            ..make_thermal(5, 100.0, 500.0)
        };
        let data = make_data_anticipated(vec![thermal], 5, commitments(5, &[200.0, 50.0]));
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
            "expected exactly one bounds-violation error (50 MW window only), got: {errors:?}"
        );
        let msg1 = &relevant[0].message;
        assert!(
            msg1.contains("Thermal 5"),
            "message should contain 'Thermal 5', got: {msg1}"
        );
        assert!(
            msg1.contains("50"),
            "message should contain the offending value 50, got: {msg1}"
        );
    }

    // ── AC-11: all values zero — passes (within [min_gen, max_gen]) ──────────

    /// Given all-zero window values and min_gen = 0.0, no bounds error is
    /// emitted (0.0 is within [0.0, 400.0]).
    #[test]
    fn test_committed_values_all_zero_ok() {
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(3)),
            ..make_thermal(7, 0.0, 400.0)
        };
        let data = make_data_anticipated(vec![thermal], 5, commitments(7, &[0.0, 0.0, 0.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "expected no errors for all-zero window values, got: {:?}",
            ctx.errors()
        );
    }

    // ── AC-12: boundary — value == min_gen (0.0) is accepted ─────────────────

    /// Given a single window value == 0.0 and min_gen == 0.0, no bounds error is
    /// emitted (the minimum is inclusive).
    #[test]
    fn test_committed_value_zero_accepted() {
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            ..make_thermal(9, 0.0, 400.0)
        };
        let data = make_data_anticipated(vec![thermal], 5, commitments(9, &[0.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "zero value must be accepted, got: {:?}",
            ctx.errors()
        );
    }

    // ── AC-13: single value above max_generation_mw → bounds error ───────────

    /// Given a single window value 400.0 against max_generation_mw = 350.0, the
    /// validator emits exactly one bounds-violation error because
    /// 400.0 > 350.0 is outside [100.0, 350.0].
    #[test]
    fn test_committed_value_above_max_bounds_error() {
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            ..make_thermal(11, 100.0, 350.0)
        };
        let data = make_data_anticipated(vec![thermal], 5, commitments(11, &[400.0]));
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
            msg.contains("400"),
            "message should contain the offending value 400, got: {msg}"
        );
    }

    // ── AC-14: non-zero in-bounds window value K=1 — accepted ────────────────

    /// K=1 case: a single window value 204.5647 against `max_generation_mw =
    /// 350.0`. The value lies within [0.0, 350.0], so the bounds check passes
    /// and the validator emits no errors.
    #[test]
    fn test_nonzero_value_in_bounds_accepted_k1() {
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            ..make_thermal(2, 0.0, 350.0)
        };
        let data = make_data_anticipated(vec![thermal], 5, commitments(2, &[204.5647]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            ctx.errors().is_empty(),
            "expected no errors for an in-bounds window value, got: {:?}",
            ctx.errors()
        );
    }

    /// K=2 acceptance: two non-zero windows, both within
    /// `[min_generation_mw, max_generation_mw]`, are accepted with zero errors.
    /// The pre-horizon-seeding contract delivers each window's rate over its
    /// covered study stage as a sunk-cost commitment; in-bounds seeds carry no
    /// validator errors.
    #[test]
    fn test_k2_two_in_bounds_nonzero_values_accepted() {
        let thermal = make_anticipated_thermal(5, 2, None, None); // min=0.0, max=500.0
        let data = make_data_anticipated(vec![thermal], 5, commitments(5, &[50.0, 30.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            ctx.errors().is_empty(),
            "expected no errors for in-bounds K=2 windows, got: {:?}",
            ctx.errors()
        );
    }

    /// Regression lock: `check_committed_value_bounds` must not emit any
    /// `SemanticAmbiguity` warning for an in-bounds non-zero seed — guards against
    /// reintroducing a same-dispatch-as-zero-seed warning.
    #[test]
    fn test_nonzero_in_bounds_seed_emits_no_semantic_ambiguity_warning() {
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
            ..make_thermal(3, 0.0, 350.0)
        };
        let data = make_data_anticipated(vec![thermal], 5, commitments(3, &[100.0, 200.0]));
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

    /// K=2 acceptance: mixed in-bounds and out-of-bounds windows. Only the
    /// out-of-bounds window produces an error; the in-bounds window (a zero
    /// value) passes silently. Verifies that the bounds check is per-window, not
    /// per-history.
    #[test]
    fn test_k2_mixed_in_bounds_and_out_of_bounds_only_oob_reported() {
        let thermal = make_anticipated_thermal(7, 2, None, None); // min=0.0, max=500.0
        let data = make_data_anticipated(vec![thermal], 5, commitments(7, &[0.0, 600.0]));
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
            "expected exactly one bounds error (600 MW window only), got: {errors:?}"
        );
        assert!(
            relevant[0].message.contains("600"),
            "error must contain value 600, got: {}",
            relevant[0].message
        );
    }

    // ── Bounds test ───────────────────────────────────────────────────────────

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
    fn make_data_thermal_bounds(n_stages: usize, rows: Vec<crate::ThermalBoundsRow>) -> ParsedData {
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
    // These mirror the strict-inequality boundary tests above for the guard
    // `stage_id < 0 || stage_id >= n_stages`; the originals above are intentional
    // duplicate coverage — do not delete them.
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
            ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
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
            ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
            entities::AnticipatedConfig,
        };

        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
            ..make_thermal(3, 0.0, 500.0)
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
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };
        let mut data = make_data_anticipated(vec![thermal], 5, commitments(3, &[0.0, 0.0]));
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
            ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
            entities::AnticipatedConfig,
        };

        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            ..make_thermal(5, 0.0, 300.0)
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
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };
        let mut data = make_data_anticipated(vec![thermal], 5, commitments(5, &[0.0]));
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
            ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
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
