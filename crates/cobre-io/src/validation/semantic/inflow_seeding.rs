//! Layer 5a — inflow lag-slot seeding validation (load-time coverage gates).
//!
//! Validates the record/conditioning coverage the load-time PAR seed
//! derivation ([`cobre_stochastic::derive_inflow_seeds`]) needs to fill the
//! lag chain and the mid-period accumulator, plus the annual-component
//! monthly-exclusive restriction.
//!
//! | # | Rule                                                                    | `ErrorKind`               |
//! |---|--------------------------------------------------------------------------|---------------------------|
//! | 1 | Lag slots `1..=max_AR_order` require full record coverage (`coverage == 1.0`) | `BusinessRuleViolation` |
//! | 2 | Lag slots `max_AR_order < s <= L_state - n_fin` require the same full coverage; slots `s > L_state - n_fin` are provably never read (advisory only) | `BusinessRuleViolation` / `ModelQuality` (warning) |
//! | 3 | A `recent_observations` conditioning window extends past the study start, into the solved study itself | `InvalidValue` |
//! | 4 | The in-progress period `[period_start, study_start)` is covered strictly between 0 and 1 | `ModelQuality` (warning) |
//! | 5 | The first study stage's season is unresolvable while PAR seeding is active (`L_state > 0`) | `ModelQuality` (warning) |
//! | 6 | Study supplies an inflow annual component (`inflow_annual_components` non-empty) while `season_map.cycle_type` is not `Monthly` | `BusinessRuleViolation` |
//! | 7 | A realized inflow record (`inflow_history` or `recent_observations`) is negative — accepted, since incremental inflow is a difference | `ModelQuality` (warning) |

use std::collections::HashMap;

use cobre_core::{EntityId, SeasonCycleType, Stage};
use cobre_stochastic::par::precompute_stage_lag_transitions;
use cobre_stochastic::season_cast::{
    RealizedWindow, cast, merge_layered_windows, nth_previous_occurrence, season_period_window,
};

use super::super::{ErrorKind, ValidationContext, schema::ParsedData};

pub(super) fn validate_inflow_seeding(data: &ParsedData, ctx: &mut ValidationContext) {
    warn_unresolvable_first_stage_season(data, ctx);
    check_conditioning_window_bound(data, ctx);
    check_inprogress_partial_coverage(data, ctx);
    check_slot_coverage(data, ctx);
    report_negative_realized_inflows(data, ctx);
    check_annual_component_monthly_only(data, ctx);
}

/// Row 7: reports realized inflows below zero. One warning per file, naming the
/// count and the worst offender — a handful of lossy reaches is legitimate
/// incremental inflow, a sign-flipped series is not, and only the magnitude
/// separates them.
fn report_negative_realized_inflows(data: &ParsedData, ctx: &mut ValidationContext) {
    let mut report = |negatives: Vec<(i32, f64)>, file: &str, field: &str| {
        let Some((worst_hydro, worst_value)) =
            negatives.iter().copied().min_by(|a, b| a.1.total_cmp(&b.1))
        else {
            return;
        };
        let count = negatives.len();
        ctx.add_warning(
            ErrorKind::ModelQuality,
            file,
            None::<String>,
            format!(
                "{count} {field} value(s) are negative (most negative {worst_value} m³/s at \
                 hydro {worst_hydro}); accepted as incremental inflow, which the LP prices \
                 through the inflow non-negativity slack"
            ),
        );
    };

    report(
        data.inflow_history
            .iter()
            .filter(|row| row.value_m3s < 0.0)
            .map(|row| (row.hydro_id.0, row.value_m3s))
            .collect(),
        "scenarios/inflow_history.parquet",
        "inflow_history",
    );
    report(
        data.initial_conditions
            .recent_observations
            .iter()
            .filter(|entry| entry.value_m3s < 0.0)
            .map(|entry| (entry.hydro_id.0, entry.value_m3s))
            .collect(),
        "initial_conditions.json",
        "recent_observations",
    );
}

/// Row 6: rejects a study supplying an inflow annual component
/// (`inflow_annual_components` non-empty) under a non-`Monthly` season cycle.
/// PAR(p)-A is monthly-exclusive by design — a permanent restriction.
fn check_annual_component_monthly_only(data: &ParsedData, ctx: &mut ValidationContext) {
    if data.inflow_annual_components.is_empty() {
        return;
    }

    let Some(season_map) = data.stages.policy_graph.season_map.as_ref() else {
        return;
    };

    let cycle = match season_map.cycle_type {
        SeasonCycleType::Weekly => "Weekly",
        SeasonCycleType::Custom => "Custom",
        SeasonCycleType::Monthly => return,
    };

    ctx.add_error(
        ErrorKind::BusinessRuleViolation,
        "scenarios/inflow_annual_component.parquet",
        None::<String>,
        format!(
            "the study supplies {} inflow annual component row(s) under a {cycle} season \
             cycle; PAR(p)-A (the annual/long-memory extension) is monthly-exclusive by design \
             — declare a Monthly season cycle or remove the annual component",
            data.inflow_annual_components.len()
        ),
    );
}

/// The first study stage (`id >= 0`, lowest `id`; stages are canonical-sorted
/// ascending). `None` when the study declares no study stages.
fn first_study_stage(data: &ParsedData) -> Option<&Stage> {
    data.stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .min_by_key(|s| s.id)
}

/// The classical AR order per (hydro, stage): the row count of
/// `inflow_ar_coefficients` sharing that key (mirrors `InflowModel::ar_order`,
/// which is `ar_coefficients.len()`). Zero without any rows.
fn classical_max_ar_order(data: &ParsedData) -> usize {
    let mut counts: HashMap<(EntityId, i32), usize> = HashMap::new();
    for row in &data.inflow_ar_coefficients {
        *counts.entry((row.hydro_id, row.stage_id)).or_insert(0) += 1;
    }
    counts.values().copied().max().unwrap_or(0)
}

/// `L_state` for the study in `data` — see [`seed_lag_state_depth`] for the
/// formula; the declared depth is read off `data.config.state_space`.
fn max_seed_lag_depth(data: &ParsedData) -> usize {
    seed_lag_state_depth(
        classical_max_ar_order(data),
        !data.inflow_annual_components.is_empty(),
        data.config.state_space.inflow_lag_depth,
    )
}

/// `L_state`: the classical AR order, floored at 12 when an annual component is
/// present (mirrors `PrecomputedPar::max_order`'s `max(classical, 12·annual)`),
/// then widened to a declared `state_space.inflow_lag_depth` via
/// `max(computed, declared)`; `None` leaves the computed depth unchanged.
///
/// The state-layout resolution applies the identical `max(computed, declared)`
/// widening, so the load-time seed depth and the runtime state layout never
/// disagree — kept identical by a cross-crate coherence regression.
#[must_use]
pub fn seed_lag_state_depth(
    classical_ar_order: usize,
    has_annual_component: bool,
    declared_inflow_lag_depth: Option<u32>,
) -> usize {
    let computed = if has_annual_component {
        classical_ar_order.max(12)
    } else {
        classical_ar_order
    };
    declared_inflow_lag_depth.map_or(computed, |declared| computed.max(declared as usize))
}

/// `n_fin`: the number of season-period occurrences that finalize (complete)
/// within the study horizon — the count of `finalize_period` transitions
/// among study stages (`id >= 0`). Computed over the full canonical stage set
/// (pre-study stages included) so `finalize_period`'s own stage lookahead
/// resolves correctly; zero without a `season_map`.
fn finalizing_period_count(data: &ParsedData) -> usize {
    let Some(season_map) = &data.stages.policy_graph.season_map else {
        return 0;
    };
    let transitions = precompute_stage_lag_transitions(&data.stages.stages, season_map, 0);
    data.stages
        .stages
        .iter()
        .zip(&transitions)
        .filter(|(stage, transition)| stage.id >= 0 && transition.finalize_period)
        .count()
}

/// `hydro_id`'s layered windows: `inflow_history` (record) shadowed day-wise
/// by `recent_observations` (conditioning) — the same construction
/// [`cobre_stochastic::derive_inflow_seeds`] uses per hydro.
fn merged_windows_for_hydro(data: &ParsedData, hydro_id: EntityId) -> Vec<RealizedWindow> {
    let record: Vec<RealizedWindow> = data
        .inflow_history
        .iter()
        .filter(|row| row.hydro_id == hydro_id)
        .map(|row| RealizedWindow {
            start_date: row.start_date,
            end_date: row.end_date,
            value_m3s: row.value_m3s,
        })
        .collect();
    let conditioning: Vec<RealizedWindow> = data
        .initial_conditions
        .recent_observations
        .iter()
        .filter(|obs| obs.hydro_id == hydro_id)
        .map(|obs| RealizedWindow {
            start_date: obs.start_date,
            end_date: obs.end_date,
            value_m3s: obs.value_m3s,
        })
        .collect();
    merge_layered_windows(&record, &conditioning)
}

/// Rows 1-2: PAR lag-slot coverage. Slots `1..=max(max_AR_order, L_state -
/// n_fin)` must have `coverage == 1.0` (a gap errors, naming the slot and the
/// affected hydros); slots beyond that are provably never read by the
/// terminal boundary, so a gap there is advisory only.
///
/// Skipped on the pure estimation path (`inflow_ar_coefficients` empty, PD-2)
/// — that path has no load-time-knowable AR order and is already guarded by
/// the record-coverage rules gating estimation itself.
#[allow(clippy::float_cmp)] // cast's whole-day-hours arithmetic keeps a full-coverage ratio bit-exact
fn check_slot_coverage(data: &ParsedData, ctx: &mut ValidationContext) {
    if data.inflow_ar_coefficients.is_empty() {
        return;
    }
    let Some(season_map) = &data.stages.policy_graph.season_map else {
        return;
    };
    let Some(first_stage) = first_study_stage(data) else {
        return;
    };
    let Some(season_id) = first_stage.season_id else {
        return;
    };
    let Some(season_def) = season_map.seasons.iter().find(|s| s.id == season_id) else {
        return;
    };

    let l_state = max_seed_lag_depth(data);
    if l_state == 0 {
        return;
    }
    let max_ar_order = classical_max_ar_order(data);
    let n_fin = finalizing_period_count(data);
    let full_coverage_upper = max_ar_order.max(l_state.saturating_sub(n_fin));

    let anchor = season_period_window(season_map, season_def, first_stage);
    let merged_by_hydro: Vec<(i32, Vec<RealizedWindow>)> = data
        .hydros
        .iter()
        .map(|h| (h.id.0, merged_windows_for_hydro(data, h.id)))
        .collect();

    for k in 1..=l_state {
        let Some(occurrence) = nth_previous_occurrence(season_map, season_def, &anchor, k) else {
            continue;
        };
        let gapped: Vec<i32> = merged_by_hydro
            .iter()
            .filter(|(_, windows)| cast(windows, &occurrence).coverage != 1.0)
            .map(|(id, _)| *id)
            .collect();
        if gapped.is_empty() {
            continue;
        }
        let names = gapped
            .iter()
            .map(|id| format!("Hydro {id}"))
            .collect::<Vec<_>>()
            .join(", ");

        if k <= full_coverage_upper {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "scenarios/inflow_history.parquet",
                None::<String>,
                format!(
                    "slot {k} lag seed requires full coverage (coverage == 1.0) at load time but \
                     is not fully covered for: {names}; supply inflow_history/recent_observations \
                     windows spanning slot {k}'s season occurrence"
                ),
            );
        } else {
            ctx.add_warning(
                ErrorKind::ModelQuality,
                "scenarios/inflow_history.parquet",
                None::<String>,
                format!(
                    "slot {k} lag seed is not fully covered for: {names}; slot {k} exceeds \
                     max(max_AR_order, L_state - n_fin) and is never read at the terminal \
                     boundary, so the partial or zero seed there is harmless"
                ),
            );
        }
    }
}

/// Row 3: a `recent_observations` window extending past the study start would
/// overlap the solved study itself — the same future-dating ban
/// `travel_time.rs`'s defluence rule enforces on `past_defluences`, mirrored
/// here on the same side: a conditioning window must stay entirely pre-study
/// (lag-slot and in-progress seeding both read only pre-study days).
fn check_conditioning_window_bound(data: &ParsedData, ctx: &mut ValidationContext) {
    let Some(study_start) = first_study_stage(data).map(|s| s.start_date) else {
        return;
    };
    for obs in &data.initial_conditions.recent_observations {
        if obs.end_date <= study_start {
            continue;
        }
        ctx.add_error(
            ErrorKind::InvalidValue,
            "initial_conditions.json",
            Some(format!("Hydro {}", obs.hydro_id.0)),
            format!(
                "Hydro {}: recent_observations window [{}, {}) extends past the study start \
                 {study_start}, into the solved study itself; a conditioning window must stay \
                 entirely pre-study",
                obs.hydro_id.0, obs.start_date, obs.end_date
            ),
        );
    }
}

/// Row 4: the in-progress period `[period_start, study_start)` covered
/// strictly between 0 and 1 is legitimate (that is the accumulator's
/// purpose) but worth a per-hydro advisory naming the fraction; full or zero
/// coverage is silent.
fn check_inprogress_partial_coverage(data: &ParsedData, ctx: &mut ValidationContext) {
    let Some(season_map) = &data.stages.policy_graph.season_map else {
        return;
    };
    let Some(first_stage) = first_study_stage(data) else {
        return;
    };
    let Some(season_id) = first_stage.season_id else {
        return;
    };
    let Some(season_def) = season_map.seasons.iter().find(|s| s.id == season_id) else {
        return;
    };

    let in_progress = season_period_window(season_map, season_def, first_stage);

    for hydro in &data.hydros {
        let merged = merged_windows_for_hydro(data, hydro.id);
        let projection = cast(&merged, &in_progress);
        if !(projection.coverage > 0.0 && projection.coverage < 1.0) {
            continue;
        }
        let hydro_id = hydro.id.0;
        ctx.add_warning(
            ErrorKind::ModelQuality,
            "scenarios/inflow_history.parquet",
            Some(format!("Hydro {hydro_id}")),
            format!(
                "Hydro {hydro_id}: the in-progress period [{}, {}) is covered only a fraction \
                 {} of the way by inflow_history/recent_observations; the accumulator seed \
                 reflects this partial period",
                in_progress.start, first_stage.start_date, projection.coverage
            ),
        );
    }
}

/// Row 5 (PD-1): warns when the first study stage's season cannot resolve —
/// no `season_map`, no `season_id` on the first study stage, or an unmatched
/// `season_id` — the same three guards `derive_inflow_seeds` returns a zero
/// seed under. Still fires alongside Layer 5b's season-id-consistency rule
/// (which already hard-errors an unmatched id on any stage): the two
/// diagnostics answer different questions — schema validity vs. seed
/// derivability.
fn warn_unresolvable_first_stage_season(data: &ParsedData, ctx: &mut ValidationContext) {
    if max_seed_lag_depth(data) == 0 || data.hydros.is_empty() {
        return;
    }
    let Some(first_stage) = first_study_stage(data) else {
        return;
    };

    let unresolvable = match &data.stages.policy_graph.season_map {
        None => true,
        Some(season_map) => match first_stage.season_id {
            None => true,
            Some(season_id) => !season_map.seasons.iter().any(|s| s.id == season_id),
        },
    };
    if !unresolvable {
        return;
    }

    let names = data
        .hydros
        .iter()
        .map(|h| format!("hydro {}", h.id.0))
        .collect::<Vec<_>>()
        .join(", ");
    ctx.add_warning(
        ErrorKind::ModelQuality,
        "initial_conditions.json",
        None::<String>,
        format!(
            "{names}: initial inflow lags seed to 0 — no resolvable season; provide \
             season_map/season_id or recent_observations"
        ),
    );
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]
mod tests {
    use super::super::test_support::*;
    use super::*;
    use crate::scenarios::InflowAnnualComponentRow;
    use cobre_core::{RecentObservation, SeasonCycleType, SeasonMap};

    fn d(y: i32, m: u32, day: u32) -> chrono::NaiveDate {
        chrono::NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn history_row(
        hydro_id: i32,
        start: chrono::NaiveDate,
        end: chrono::NaiveDate,
    ) -> crate::InflowHistoryRow {
        crate::InflowHistoryRow {
            hydro_id: EntityId::from(hydro_id),
            start_date: start,
            end_date: end,
            value_m3s: 500.0,
        }
    }

    // ── Rows 1-2: slot coverage ────────────────────────────────────────────

    /// PD-2: a pure estimation-path case (`inflow_ar_coefficients` absent,
    /// only an annual component) must skip slot-coverage entirely, even with
    /// a badly under-covered record that would otherwise error on every slot.
    #[test]
    fn test_estimation_path_without_ar_coefficients_skips_slot_coverage() {
        let mut stages = make_stages_with_seasons(3, true);
        stages.stages[0].season_id = Some(0);
        let mut data = make_data_estimation(
            vec![make_hydro(1, None)],
            stages,
            vec![history_row(1, d(1999, 12, 20), d(2000, 1, 1))],
        );
        data.inflow_annual_components = vec![annual_component_row(1)];

        let mut ctx = ValidationContext::new();
        validate_inflow_seeding(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "the estimation path (inflow_ar_coefficients empty) must skip \
             slot-coverage checks entirely, got: {:?}",
            ctx.errors()
        );
    }

    #[test]
    fn test_sixty_stage_monthly_order_six_full_coverage_no_error() {
        let mut stages = make_stages_with_seasons(60, true);
        stages.stages[0].season_id = Some(0);
        let mut data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            stages,
            vec![],
            vec![],
        );
        data.inflow_ar_coefficients = (1..=6).map(|lag| make_ar_row(1, 0, lag)).collect();
        data.inflow_history = vec![history_row(1, d(1999, 7, 1), d(2000, 1, 1))];

        let mut ctx = ValidationContext::new();
        validate_inflow_seeding(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "full coverage of slots 1..=6 must not error, got: {:?}",
            ctx.errors()
        );
    }

    #[test]
    fn test_ar_slot_gap_errors_naming_slot_and_hydro() {
        let mut stages = make_stages_with_seasons(60, true);
        stages.stages[0].season_id = Some(0);
        let mut data = make_data(
            vec![make_hydro(1, None), make_hydro(7, None)],
            vec![],
            vec![],
            stages,
            vec![],
            vec![],
        );
        data.inflow_ar_coefficients = (1..=6).map(|lag| make_ar_row(1, 0, lag)).collect();
        data.inflow_history = vec![
            history_row(1, d(1999, 7, 1), d(2000, 1, 1)),
            history_row(7, d(1999, 7, 1), d(1999, 10, 1)),
            history_row(7, d(1999, 11, 1), d(2000, 1, 1)),
        ];

        let mut ctx = ValidationContext::new();
        validate_inflow_seeding(&data, &mut ctx);

        let errors = ctx.errors();
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("slot 3")
                    && e.message.contains("Hydro 7")),
            "expected a slot-3 gap naming Hydro 7, got: {errors:?}"
        );
        assert!(
            !errors
                .iter()
                .any(|e| e.message.contains("slot") && e.message.contains("Hydro 1")),
            "hydro 1's full coverage must not error, got: {errors:?}"
        );
    }

    /// The six-stage weekly-into-monthly layout (April via W1-W5, May via M2)
    /// finalizes exactly 2 season periods (`n_fin = 2`); with a classical AR
    /// order dominated by `L_state - n_fin` and an annual component present,
    /// `L_state = 12`.
    fn weekly_monthly_two_month_stages() -> crate::stages::StagesData {
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, PolicyGraph, PolicyGraphType, ScenarioSourceConfig,
            SeasonCycleType, SeasonDefinition, SeasonMap, Stage, StageRiskConfig, StageStateConfig,
        };

        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Monthly,
            seasons: (0..12u32)
                .map(|m| SeasonDefinition {
                    id: m as usize,
                    label: format!("Month{}", m + 1),
                    month_start: m + 1,
                    day_start: None,
                    month_end: None,
                    day_end: None,
                })
                .collect(),
        };

        let make =
            |index: usize, start: chrono::NaiveDate, end: chrono::NaiveDate, season_id: usize| {
                Stage {
                    index,
                    id: i32::try_from(index).unwrap(),
                    start_date: start,
                    end_date: end,
                    season_id: Some(season_id),
                    blocks: vec![Block {
                        index: 0,
                        name: "SINGLE".to_string(),
                        duration_hours: (end - start).num_days() as f64 * 24.0,
                    }],
                    block_mode: BlockMode::Parallel,
                    state_config: StageStateConfig {
                        storage: true,
                        inflow_lags: false,
                    },
                    risk_config: StageRiskConfig::Expectation,
                    scenario_config: ScenarioSourceConfig {
                        branching_factor: 1,
                        noise_method: NoiseMethod::Saa,
                    },
                }
            };

        let stages = vec![
            make(0, d(2026, 3, 28), d(2026, 4, 4), 3),
            make(1, d(2026, 4, 4), d(2026, 4, 11), 3),
            make(2, d(2026, 4, 11), d(2026, 4, 18), 3),
            make(3, d(2026, 4, 18), d(2026, 4, 25), 3),
            make(4, d(2026, 4, 25), d(2026, 5, 2), 3),
            make(5, d(2026, 5, 2), d(2026, 6, 1), 4),
        ];

        crate::stages::StagesData {
            openings_declared: std::collections::HashSet::new(),
            stages,
            policy_graph: PolicyGraph {
                stage_discount_rate_overrides: std::collections::HashMap::new(),
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                nodes: Vec::new(),
                season_map: Some(season_map),
            },
        }
    }

    fn annual_component_row(hydro_id: i32) -> InflowAnnualComponentRow {
        InflowAnnualComponentRow {
            hydro_id: EntityId::from(hydro_id),
            stage_id: 0,
            annual_coefficient: -0.25,
            annual_mean_m3s: 1500.0,
            annual_std_m3s: 300.0,
        }
    }

    #[test]
    fn test_eleven_month_record_l_state_12_n_fin_2_slot12_info_no_error() {
        let stages = weekly_monthly_two_month_stages();
        let mut data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            stages,
            vec![],
            vec![],
        );
        data.inflow_ar_coefficients = vec![make_ar_row(1, 0, 1)];
        data.inflow_annual_components = vec![annual_component_row(1)];
        data.inflow_history = vec![history_row(1, d(2025, 5, 1), d(2026, 4, 1))];

        let mut ctx = ValidationContext::new();
        validate_inflow_seeding(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "an 11-month record covering slots 1..=11 must not error, got: {:?}",
            ctx.errors()
        );
        assert!(
            ctx.warnings()
                .iter()
                .any(|w| w.kind == ErrorKind::ModelQuality && w.message.contains("slot 12")),
            "expected a slot-12 zero-fill info diagnostic, got: {:?}",
            ctx.warnings()
        );
    }

    #[test]
    fn test_nine_month_record_errors_naming_slot_10_and_hydros() {
        let stages = weekly_monthly_two_month_stages();
        let mut data = make_data(
            vec![make_hydro(1, None), make_hydro(2, None)],
            vec![],
            vec![],
            stages,
            vec![],
            vec![],
        );
        data.inflow_ar_coefficients = vec![make_ar_row(1, 0, 1), make_ar_row(2, 0, 1)];
        data.inflow_annual_components = vec![annual_component_row(1)];
        data.inflow_history = vec![
            history_row(1, d(2025, 7, 1), d(2026, 4, 1)),
            history_row(2, d(2025, 7, 1), d(2026, 4, 1)),
        ];

        let mut ctx = ValidationContext::new();
        validate_inflow_seeding(&data, &mut ctx);

        let errors = ctx.errors();
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("slot 10")
                    && e.message.contains("Hydro 1")
                    && e.message.contains("Hydro 2")),
            "expected a slot-10 gap naming both hydros, got: {errors:?}"
        );
    }

    // ── Row 3: conditioning-window bound ───────────────────────────────────

    #[test]
    fn test_conditioning_window_extending_past_study_start_errors() {
        let data_stages = make_stages(vec![0, 1, 2]);
        let mut data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            data_stages,
            vec![],
            vec![],
        );
        data.initial_conditions.recent_observations = vec![RecentObservation {
            hydro_id: EntityId::from(1),
            start_date: d(2023, 12, 20),
            end_date: d(2024, 1, 10),
            value_m3s: 200.0,
        }];

        let mut ctx = ValidationContext::new();
        validate_inflow_seeding(&data, &mut ctx);

        let errors = ctx.errors();
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::InvalidValue
                && e.message.contains("Hydro 1")
                && e.message.contains("study")),
            "a conditioning window extending past study_start must error, got: {errors:?}"
        );
    }

    #[test]
    fn test_conditioning_window_ending_at_study_start_no_error() {
        let data_stages = make_stages(vec![0, 1, 2]);
        let mut data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            data_stages,
            vec![],
            vec![],
        );
        data.initial_conditions.recent_observations = vec![RecentObservation {
            hydro_id: EntityId::from(1),
            start_date: d(2023, 12, 1),
            end_date: d(2024, 1, 1),
            value_m3s: 200.0,
        }];

        let mut ctx = ValidationContext::new();
        validate_inflow_seeding(&data, &mut ctx);

        assert!(
            !ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidValue),
            "a fully pre-study window ending exactly at study_start must not error, got: {:?}",
            ctx.errors()
        );
    }

    // ── Row 4: in-progress partial-coverage advisory ───────────────────────

    #[test]
    fn test_partial_inprogress_coverage_emits_single_warning_with_fraction() {
        let mut stages = make_stages(vec![0, 1, 2]);
        stages.stages[0].start_date = d(2026, 4, 11);
        stages.stages[0].end_date = d(2026, 5, 2);
        stages.stages[0].season_id = Some(3);
        stages.policy_graph.season_map = Some(make_monthly_season_map());
        let mut data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            stages,
            vec![],
            vec![],
        );
        data.inflow_history = vec![history_row(1, d(2026, 4, 1), d(2026, 4, 11))];

        let mut ctx = ValidationContext::new();
        validate_inflow_seeding(&data, &mut ctx);

        let expected_fraction = 10.0 * 24.0 / (30.0 * 24.0);
        let matches: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| {
                w.kind == ErrorKind::ModelQuality
                    && w.entity.as_deref() == Some("Hydro 1")
                    && w.message.contains(&format!("{expected_fraction}"))
            })
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected exactly one in-progress partial-coverage warning naming the fraction, got: {:?}",
            ctx.warnings()
        );
    }

    #[test]
    fn test_full_coverage_record_no_diagnostics() {
        let stages = make_stages_with_seasons(3, true);
        let mut data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            stages,
            vec![],
            vec![],
        );
        data.inflow_ar_coefficients = vec![make_ar_row(1, 0, 1), make_ar_row(1, 0, 2)];
        data.inflow_history = vec![history_row(1, d(1999, 11, 1), d(2000, 1, 1))];

        let mut ctx = ValidationContext::new();
        validate_inflow_seeding(&data, &mut ctx);

        assert!(!ctx.has_errors(), "got: {:?}", ctx.errors());
        assert!(ctx.warnings().is_empty(), "got: {:?}", ctx.warnings());
    }

    // ── Row 5 (PD-1): unresolvable first-stage season ──────────────────────

    #[test]
    fn test_unresolvable_first_stage_season_emits_zero_fill_warning() {
        let stages = make_stages(vec![0, 1, 2]); // no season_map, no season_id
        let mut data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            stages,
            vec![],
            vec![],
        );
        data.inflow_ar_coefficients = vec![make_ar_row(1, 0, 1)];

        let mut ctx = ValidationContext::new();
        validate_inflow_seeding(&data, &mut ctx);

        assert!(
            ctx.warnings()
                .iter()
                .any(|w| w.kind == ErrorKind::ModelQuality
                    && w.message.contains("no resolvable season")
                    && w.message.contains("hydro 1")),
            "expected the unresolvable-season zero-fill warning, got: {:?}",
            ctx.warnings()
        );
    }

    // ── Row 6: non-monthly annual component reject ─────────────────────────

    fn season_map_with_cycle(cycle: SeasonCycleType) -> SeasonMap {
        use cobre_core::temporal::SeasonDefinition;
        SeasonMap {
            cycle_type: cycle,
            seasons: vec![SeasonDefinition {
                id: 0,
                label: "S0".to_string(),
                month_start: 1,
                day_start: None,
                month_end: None,
                day_end: None,
            }],
        }
    }

    fn par_cycle_data(season_map: SeasonMap) -> ParsedData {
        let mut data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0, 1, 2]),
            vec![],
            vec![],
        );
        data.stages.stages[0].season_id = Some(0);
        data.stages.policy_graph.season_map = Some(season_map);
        data
    }

    #[test]
    fn test_annual_component_non_monthly_still_rejects_from_inflow_seeding() {
        let mut data = par_cycle_data(season_map_with_cycle(SeasonCycleType::Weekly));
        data.inflow_annual_components = vec![annual_component_row(1)];

        let mut ctx = ValidationContext::new();
        check_annual_component_monthly_only(&data, &mut ctx);

        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("monthly-exclusive")),
            "expected a monthly-exclusive reject from inflow_seeding, got: {:?}",
            ctx.errors()
        );
    }

    #[test]
    fn test_weekly_cycle_with_annual_component_rejects_naming_monthly_exclusive() {
        let mut data = par_cycle_data(season_map_with_cycle(SeasonCycleType::Weekly));
        data.inflow_annual_components = vec![annual_component_row(1)];

        let mut ctx = ValidationContext::new();
        check_annual_component_monthly_only(&data, &mut ctx);

        let errors = ctx.errors();
        assert_eq!(errors.len(), 1, "exactly one reject, got: {errors:?}");
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("Weekly")
                    && e.message.contains("monthly-exclusive")),
            "expected a Weekly PAR(p)-A monthly-exclusive reject, got: {errors:?}"
        );
    }

    #[test]
    fn test_monthly_cycle_with_annual_component_no_reject() {
        let mut data = par_cycle_data(make_monthly_season_map());
        data.inflow_annual_components = vec![annual_component_row(1)];

        let mut ctx = ValidationContext::new();
        check_annual_component_monthly_only(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "a Monthly-cycle study with an annual component must not reject, got: {:?}",
            ctx.errors()
        );
    }

    #[test]
    fn test_weekly_cycle_without_annual_component_no_reject() {
        let data = par_cycle_data(season_map_with_cycle(SeasonCycleType::Weekly));

        let mut ctx = ValidationContext::new();
        check_annual_component_monthly_only(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "a Weekly-cycle study with no annual component must not reject, got: {:?}",
            ctx.errors()
        );
    }

    // ── Row 7: negative realized inflows ──────────────────────────────────

    fn negative_inflow_data() -> crate::validation::schema::ParsedData {
        let mut data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages_with_seasons(3, true),
            vec![],
            vec![],
        );
        data.inflow_history = vec![
            crate::InflowHistoryRow {
                hydro_id: EntityId::from(1),
                start_date: d(1999, 11, 1),
                end_date: d(1999, 12, 1),
                value_m3s: -12.0,
            },
            crate::InflowHistoryRow {
                hydro_id: EntityId::from(2),
                start_date: d(1999, 12, 1),
                end_date: d(2000, 1, 1),
                value_m3s: -30.0,
            },
        ];
        data.initial_conditions.recent_observations = vec![RecentObservation {
            hydro_id: EntityId::from(1),
            start_date: d(2000, 1, 1),
            end_date: d(2000, 1, 8),
            value_m3s: -4.0,
        }];
        data
    }

    #[test]
    fn test_negative_realized_inflows_warn_once_per_file_naming_the_worst() {
        let data = negative_inflow_data();

        let mut ctx = ValidationContext::new();
        report_negative_realized_inflows(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "negative incremental inflow must warn, never reject, got: {:?}",
            ctx.errors()
        );
        let warnings = ctx.warnings();
        assert_eq!(
            warnings.len(),
            2,
            "one warning per file, not per row, got: {warnings:?}"
        );
        assert!(
            warnings.iter().any(|w| w.kind == ErrorKind::ModelQuality
                && w.file.ends_with("inflow_history.parquet")
                && w.message.contains('2')
                && w.message.contains("-30")
                && w.message.contains("hydro 2")),
            "expected the history warning to name the count and the worst offender, got: {warnings:?}"
        );
        assert!(
            warnings.iter().any(|w| w.kind == ErrorKind::ModelQuality
                && w.file.ends_with("initial_conditions.json")
                && w.message.contains("-4")
                && w.message.contains("hydro 1")),
            "expected the observation warning to name its own worst offender, got: {warnings:?}"
        );
    }

    #[test]
    fn test_non_negative_realized_inflows_produce_no_warning() {
        let mut data = negative_inflow_data();
        for row in &mut data.inflow_history {
            row.value_m3s = 500.0;
        }
        data.initial_conditions.recent_observations[0].value_m3s = 0.0;

        let mut ctx = ValidationContext::new();
        report_negative_realized_inflows(&data, &mut ctx);

        assert!(
            ctx.warnings().is_empty(),
            "no negative value must produce no warning, got: {:?}",
            ctx.warnings()
        );
    }

    // ── max_seed_lag_depth / seed_lag_state_depth: declared-depth widening ──

    fn order_six_data(declared: Option<u32>) -> ParsedData {
        let stages = make_stages_with_seasons(60, true);
        let mut data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            stages,
            vec![],
            vec![],
        );
        data.inflow_ar_coefficients = (1..=6).map(|lag| make_ar_row(1, 0, lag)).collect();
        data.config.state_space.inflow_lag_depth = declared;
        data
    }

    #[test]
    fn test_declared_depth_widens_seed_lag_depth_above_ar_order() {
        assert_eq!(max_seed_lag_depth(&order_six_data(Some(24))), 24);
    }

    #[test]
    fn test_declared_depth_below_ar_order_floors_at_ar_order() {
        assert_eq!(max_seed_lag_depth(&order_six_data(Some(3))), 6);
    }

    #[test]
    fn test_absent_declared_depth_is_byte_neutral() {
        let data = order_six_data(None);
        assert_eq!(data.config.state_space.inflow_lag_depth, None);
        assert_eq!(max_seed_lag_depth(&data), 6);
    }

    #[test]
    fn test_seed_lag_state_depth_formula_floors_and_widens() {
        assert_eq!(seed_lag_state_depth(6, false, Some(24)), 24);
        assert_eq!(seed_lag_state_depth(6, false, Some(3)), 6);
        assert_eq!(seed_lag_state_depth(6, false, None), 6);
        assert_eq!(seed_lag_state_depth(1, true, Some(8)), 12);
        assert_eq!(seed_lag_state_depth(1, true, Some(24)), 24);
        assert_eq!(seed_lag_state_depth(1, true, None), 12);
    }
}
