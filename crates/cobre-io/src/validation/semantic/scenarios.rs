//! Layer 5b — scenario, penalty, and probability-data validation.

use std::collections::{HashMap, HashSet};

use super::super::{ErrorKind, ValidationContext, schema::ParsedData};

// ── Rules 6-10: Penalty ordering ──────────────────────────────────────────────

/// Checks the penalty hierarchy ordering across all hydros and buses.
///
/// Emits one `ModelQuality` warning per violated ordering check, aggregating
/// all violating entities into a single warning with the count and worst-case ID.
// Rationale: five independent ordering rules, each a full entity pass with its
// own worst-case aggregation; per-rule helpers would not cut the line count.
#[allow(clippy::too_many_lines)]
pub(super) fn check_penalty_ordering(data: &ParsedData, ctx: &mut ValidationContext) {
    let max_deficit_cost: f64 = data
        .buses
        .iter()
        .flat_map(|b| b.deficit_segments.iter().map(|s| s.cost_per_mwh))
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0);

    // Check 6: filling must not be as hard as load shedding. Skipped with no
    // deficit segments (max == 0.0) — there is then no comparand.
    if max_deficit_cost > 0.0 {
        let mut violations: Vec<(i32, f64)> = Vec::new(); // (id, filling_target_cost)
        for hydro in &data.hydros {
            let filling = hydro.penalties.filling_target_violation_cost;
            if filling >= max_deficit_cost {
                violations.push((hydro.id.0, filling));
            }
        }
        if let Some(worst) = violations
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            let count = violations.len();
            ctx.add_warning(
                ErrorKind::ModelQuality,
                "penalties.json",
                None::<&str>,
                format!(
                    "Penalty ordering violation: filling_target_violation_cost ({}) should be < \
                     deficit_cost ({max_deficit_cost}) so filling is not as hard as load shedding \
                     -- {count} hydro(s) affected, worst-case hydro {}",
                    worst.1, worst.0
                ),
            );
        }
    }

    // Check 7.
    {
        let mut violations: Vec<(i32, f64)> = Vec::new(); // (id, storage_violation_cost)
        for hydro in &data.hydros {
            let higher = hydro.penalties.storage_violation_below_cost;
            if higher <= max_deficit_cost {
                violations.push((hydro.id.0, higher));
            }
        }
        if let Some(worst) = violations
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            let count = violations.len();
            ctx.add_warning(
                ErrorKind::ModelQuality,
                "penalties.json",
                None::<&str>,
                format!(
                    "Penalty ordering violation: storage_violation_below_cost ({}) should be > \
                     max(deficit_segment_costs) ({max_deficit_cost}) -- {count} hydro(s) affected, \
                     worst case: Hydro {}",
                    worst.1, worst.0
                ),
            );
        }
    }

    // Check 8.
    {
        let max_cv = |h: &cobre_core::entities::Hydro| {
            let p = &h.penalties;
            p.turbined_violation_below_cost
                .max(p.outflow_violation_below_cost)
                .max(p.outflow_violation_above_cost)
                .max(p.generation_violation_below_cost)
                .max(p.evaporation_violation_cost)
                .max(p.water_withdrawal_violation_cost)
        };

        let max_constraint_cost: f64 = data
            .hydros
            .iter()
            .map(max_cv)
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.0);

        if !data.hydros.is_empty() && max_deficit_cost <= max_constraint_cost {
            // worst offender: highest constraint-violation cost
            if let Some(worst_hydro) = data.hydros.iter().max_by(|a, b| {
                max_cv(a)
                    .partial_cmp(&max_cv(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                ctx.add_warning(
                    ErrorKind::ModelQuality,
                    "penalties.json",
                    None::<&str>,
                    format!(
                        "Penalty ordering violation: max(deficit_segment_costs) \
                         ({max_deficit_cost}) should be > max(constraint_violation_costs) \
                         ({max_constraint_cost}) -- 1 hydro(s) affected, worst case: Hydro {}",
                        worst_hydro.id.0
                    ),
                );
            }
        }
    }

    // Check 9.
    {
        if !data.hydros.is_empty() {
            let min_cv = |h: &cobre_core::entities::Hydro| {
                let p = &h.penalties;
                p.turbined_violation_below_cost
                    .min(p.outflow_violation_below_cost)
                    .min(p.outflow_violation_above_cost)
                    .min(p.generation_violation_below_cost)
                    .min(p.evaporation_violation_cost)
                    .min(p.water_withdrawal_violation_cost)
            };

            let min_constraint_cost: f64 =
                data.hydros.iter().map(min_cv).fold(f64::INFINITY, f64::min);

            let max_resource_cost: f64 = data
                .hydros
                .iter()
                .map(|h| h.penalties.spillage_cost.max(h.penalties.diversion_cost))
                .fold(f64::NEG_INFINITY, f64::max)
                .max(0.0);

            if min_constraint_cost <= max_resource_cost {
                // worst offender: lowest constraint-violation cost
                if let Some(worst_hydro) = data.hydros.iter().min_by(|a, b| {
                    min_cv(a)
                        .partial_cmp(&min_cv(b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    ctx.add_warning(
                        ErrorKind::ModelQuality,
                        "penalties.json",
                        None::<&str>,
                        format!(
                            "Penalty ordering violation: min(constraint_violation_costs) \
                             ({min_constraint_cost}) should be > max(resource_costs) \
                             ({max_resource_cost}) -- 1 hydro(s) affected, worst case: Hydro {}",
                            worst_hydro.id.0
                        ),
                    );
                }
            }
        }
    }

    // Check 10.
    {
        let mut violations: Vec<(i32, f64)> = Vec::new(); // (id, min_resource_cost)
        for hydro in &data.hydros {
            let min_resource = hydro
                .penalties
                .spillage_cost
                .min(hydro.penalties.diversion_cost);
            if min_resource <= 0.0 {
                violations.push((hydro.id.0, min_resource));
            }
        }
        if let Some(worst) = violations
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            let count = violations.len();
            ctx.add_warning(
                ErrorKind::ModelQuality,
                "penalties.json",
                None::<&str>,
                format!(
                    "Penalty ordering violation: min(resource_costs) ({}) should be > 0 \
                     (regularization costs must be positive to prevent LP degeneracy) -- \
                     {count} hydro(s) affected, worst case: Hydro {}",
                    worst.1, worst.0
                ),
            );
        }
    }
}

// ── Rule 11: FPHA penalty rule ─────────────────────────────────────────────────

/// Checks that FPHA hydros have `turbined_cost >= 0`.
///
/// A zero cost is valid for constant-head plants (e.g., `gamma_v = 0`) where the
/// LP has no incentive to spill rather than turbine. Negative values are rejected
/// because they would make turbining artificially profitable and distort dispatch.
pub(super) fn check_fpha_penalty_rule(data: &ParsedData, ctx: &mut ValidationContext) {
    use cobre_core::entities::HydroGenerationModel;
    for hydro in &data.hydros {
        if hydro.generation_model == HydroGenerationModel::Fpha {
            let fpha_cost = hydro.penalties.turbined_cost;
            if fpha_cost < 0.0 {
                let entity_str = format!("Hydro {}", hydro.id.0);
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "penalties.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: turbined_cost ({fpha_cost}) must be non-negative (>= 0) \
                         for FPHA hydros; negative values distort LP dispatch"
                    ),
                );
            }
        }
    }
}

// ── Rules 12-13: Scenario model rules ─────────────────────────────────────────

/// Validates inflow model standard deviation and AR coefficient count consistency.
pub(super) fn check_scenario_models(data: &ParsedData, ctx: &mut ValidationContext) {
    // Rule 12: the parser rejects std_m3s < 0; this layer only warns on == 0.0
    // (valid but unusual deterministic inflow).
    for row in &data.inflow_seasonal_stats {
        if row.std_m3s == 0.0 {
            ctx.add_warning(
                ErrorKind::ModelQuality,
                "scenarios/inflow_seasonal_stats.parquet",
                Some(format!("Hydro {}", row.hydro_id.0)),
                format!(
                    "Hydro {} stage {}: std_m3s is 0.0, indicating deterministic inflow \
                     (no stochastic component); verify this is intentional",
                    row.hydro_id.0, row.stage_id
                ),
            );
        }
    }

    // Rule 13 (V-AR-4): all lag rows of a (hydro_id, stage_id) group must share
    // one residual_std_ratio. The parser does range validation; this checks only
    // cross-row consistency within a group.
    {
        let mut ratio_by_group: HashMap<(i32, i32), f64> = HashMap::new();
        for row in &data.inflow_ar_coefficients {
            let key = (row.hydro_id.0, row.stage_id);
            match ratio_by_group.entry(key) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(row.residual_std_ratio);
                }
                std::collections::hash_map::Entry::Occupied(e) => {
                    if (*e.get() - row.residual_std_ratio).abs() > f64::EPSILON {
                        ctx.add_error(
                            ErrorKind::InvalidValue,
                            "scenarios/inflow_ar_coefficients.parquet",
                            Some(format!("Hydro {}", row.hydro_id.0)),
                            format!(
                                "Hydro {} stage {}: inconsistent residual_std_ratio across \
                                 lag rows (first={}, current={}); all lags must share the \
                                 same ratio",
                                row.hydro_id.0,
                                row.stage_id,
                                e.get(),
                                row.residual_std_ratio,
                            ),
                        );
                    }
                }
            }
        }
    }
}

// ── External scheme requires external scenario files ─────────────────────────

/// Validates that when a class uses the `External` sampling scheme, the
/// corresponding external scenario file data is non-empty.
pub(super) fn check_external_scheme_has_files(data: &ParsedData, ctx: &mut ValidationContext) {
    use cobre_core::scenario::SamplingScheme;
    use std::path::Path;

    // Config is Layer-2-validated, so these reads do not fail in practice.
    let Ok(training_source) = data
        .config
        .training_scenario_source(Path::new("config.json"))
    else {
        return;
    };
    let Ok(simulation_source) = data
        .config
        .simulation_scenario_source(Path::new("config.json"))
    else {
        return;
    };

    // Check simulation only when it defines its own scenario_source; otherwise it
    // falls back to training (already checked) and would duplicate errors.
    let sources: &[(&str, &_)] = if data.config.simulation.scenario_source.is_some() {
        &[
            ("training", &training_source),
            ("simulation", &simulation_source),
        ]
    } else {
        &[("training", &training_source)]
    };

    let mut check_external =
        |section: &str, scheme: SamplingScheme, class_name: &str, is_empty: bool| {
            if scheme == SamplingScheme::External && is_empty {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "config.json",
                    Some(format!("{section}.scenario_source.{class_name}")),
                    format!(
                        "{class_name} class uses 'external' scheme but no \
                     external_{class_name}_scenarios.parquet data was found; \
                     external scheme requires corresponding scenario file"
                    ),
                );
            }
        };

    for (section, source) in sources {
        check_external(
            section,
            source.inflow_scheme,
            "inflow",
            data.external_scenarios.is_empty(),
        );
        check_external(
            section,
            source.load_scheme,
            "load",
            data.external_load_scenarios.is_empty(),
        );
        check_external(
            section,
            source.ncs_scheme,
            "ncs",
            data.external_ncs_scenarios.is_empty(),
        );
    }
}

// ── Rules 17-18: Load factor consistency ─────────────────────────────────────

/// Validates cross-file consistency between `load_factors.json` and
/// `load_seasonal_stats.parquet`.
///
/// Rule 17: For every `LoadFactorEntry`, each `block_factors[j].block_id` must
/// match a `Block.index` in the corresponding stage's `blocks` array.
///
/// Rule 18: A `LoadFactorEntry` for a `(bus_id, stage_id)` pair where
/// `load_seasonal_stats` has `std_mw == 0.0` (deterministic load) produces a
/// `ModelQuality` warning because block factors have no effect on deterministic
/// loads.
///
/// Silently skips when `data.load_factors` is empty.
pub(super) fn check_load_factor_consistency(data: &ParsedData, ctx: &mut ValidationContext) {
    if data.load_factors.is_empty() {
        return;
    }

    let stage_block_indices: HashMap<i32, HashSet<usize>> = data
        .stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| {
            let indices: HashSet<usize> = s.blocks.iter().map(|b| b.index).collect();
            (s.id, indices)
        })
        .collect();

    let load_std: HashMap<(i32, i32), f64> = data
        .load_seasonal_stats
        .iter()
        .map(|row| ((row.bus_id.0, row.stage_id), row.std_mw))
        .collect();

    for (i, entry) in data.load_factors.iter().enumerate() {
        // Rule 17.
        if let Some(valid_indices) = stage_block_indices.get(&entry.stage_id) {
            for bf in &entry.block_factors {
                let block_idx = usize::try_from(bf.block_id).unwrap_or(usize::MAX);
                if !valid_indices.contains(&block_idx) {
                    let sorted: Vec<usize> = {
                        let mut v: Vec<usize> = valid_indices.iter().copied().collect();
                        v.sort_unstable();
                        v
                    };
                    ctx.add_error(
                        ErrorKind::BusinessRuleViolation,
                        "scenarios/load_factors.json",
                        Some(format!("LoadFactorEntry[{i}]")),
                        format!(
                            "LoadFactorEntry[{i}] has block_id {} which is not in the block set \
                             {sorted:?} for stage {}",
                            bf.block_id, entry.stage_id
                        ),
                    );
                }
            }
        }

        // Rule 18.
        let key = (entry.bus_id.0, entry.stage_id);
        if let Some(&std_mw) = load_std.get(&key)
            && std_mw == 0.0
        {
            ctx.add_warning(
                ErrorKind::ModelQuality,
                "scenarios/load_factors.json",
                Some(format!("LoadFactorEntry[{i}]")),
                format!(
                    "LoadFactorEntry[{i}] (bus {}, stage {}) references a deterministic load \
                         (std_mw == 0.0); block factors have no effect on deterministic loads",
                    entry.bus_id.0, entry.stage_id
                ),
            );
        }
    }
}

// ── Rules 19-21: Estimation prerequisites ─────────────────────────────────────

/// Validates prerequisites for the history-based PAR(p) estimation path.
///
/// Runs only when `inflow_history.parquet` is present and
/// `inflow_seasonal_stats.parquet` is absent — i.e., when the estimation path
/// will be triggered (same condition used by the estimation pipeline).
///
/// Rule 19: `season_definitions` must be present in `stages.json` so that
/// observations can be grouped by season.
///
/// Rule 20: Each `(hydro_id, season_id)` group must have at least
/// `config.estimation.min_observations_per_season` observations.
///
/// Rule 21: Every hydro in the system must have at least one observation in
/// `inflow_history.parquet`; missing hydros cannot be estimated.
pub(super) fn check_estimation_prerequisites(data: &ParsedData, ctx: &mut ValidationContext) {
    // Mirror the runtime's estimation trigger: it skips only when BOTH stats and
    // AR coefficients are present, so estimation is active otherwise (history present).
    let has_history = !data.inflow_history.is_empty();
    let has_stats = !data.inflow_seasonal_stats.is_empty();
    let has_ar_coefficients = !data.inflow_ar_coefficients.is_empty();
    let estimation_active = has_history && !(has_stats && has_ar_coefficients);

    if !estimation_active {
        return;
    }

    // Rule 19.
    if data.stages.policy_graph.season_map.is_none() {
        ctx.add_error(
            ErrorKind::BusinessRuleViolation,
            "scenarios/inflow_history.parquet",
            None::<&str>,
            "season_definitions is required in stages.json when estimating from \
             inflow_history.parquet; add a season_definitions section to stages.json",
        );
    }

    // Rule 21.
    let hydro_ids_in_history: HashSet<i32> =
        data.inflow_history.iter().map(|r| r.hydro_id.0).collect();
    let mut missing_hydros: Vec<i32> = data
        .hydros
        .iter()
        .filter(|h| !hydro_ids_in_history.contains(&h.id.0))
        .map(|h| h.id.0)
        .collect();
    missing_hydros.sort_unstable();
    for id in missing_hydros {
        ctx.add_error(
            ErrorKind::BusinessRuleViolation,
            "scenarios/inflow_history.parquet",
            Some(format!("Hydro {id}")),
            format!(
                "hydro {id} has no observations in inflow_history.parquet but estimation \
                 is required; add historical inflow data for this hydro"
            ),
        );
    }

    // Rule 20. Skipped when season_map is None — Rule 19 already errored, and
    // running here would cascade a confusing second diagnostic.
    if let Some(_season_map) = &data.stages.policy_graph.season_map {
        let min_obs = data.config.estimation.min_observations_per_season as usize;

        // Stages are sorted by id, which matches date order — partition_point relies on it.
        let stage_index: Vec<(chrono::NaiveDate, chrono::NaiveDate, usize)> = data
            .stages
            .stages
            .iter()
            .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, sid)))
            .collect();

        let mut counts: HashMap<(i32, usize), usize> = HashMap::new();
        for row in &data.inflow_history {
            let pos = stage_index.partition_point(|(start, _, _)| *start <= row.date);
            let season_id = if pos > 0 {
                let (_, end_date, sid) = stage_index[pos - 1];
                if row.date < end_date { Some(sid) } else { None }
            } else {
                None
            };

            if let Some(sid) = season_id {
                *counts.entry((row.hydro_id.0, sid)).or_insert(0) += 1;
            }
        }

        let mut violations: Vec<(i32, usize, usize)> = counts
            .iter()
            .filter(|&(_, n)| *n < min_obs)
            .map(|(&(hid, sid), &n)| (hid, sid, n))
            .collect();
        // Sort for deterministic output order.
        violations.sort_unstable_by_key(|&(hid, sid, _)| (hid, sid));
        for (hid, sid, n) in violations {
            ctx.add_warning(
                ErrorKind::ModelQuality,
                "scenarios/inflow_history.parquet",
                Some(format!("Hydro {hid}")),
                format!(
                    "hydro {hid} season {sid} has {n} observations \
                     (minimum recommended: {min_obs}); estimation accuracy may be \
                     insufficient with so few observations"
                ),
            );
        }
    }
}

// ── Rules 22-24: Past inflows coverage ────────────────────────────────────────

/// Validates that `initial_conditions.json` provides sufficient `past_inflows`
/// entries for lag initialization when `inflow_lags: true` and PAR order > 0.
///
/// Runs only when at least one study stage has `state_config.inflow_lags: true`
/// AND `inflow_ar_coefficients` is non-empty with maximum PAR order > 0.
///
/// Rule 22: `past_inflows` must be non-empty when lag initialization is needed.
///
/// Rule 23: For each hydro with per-hydro PAR order `p` (max lag across all its
/// `(hydro_id, stage_id)` groups), `past_inflows` must contain an entry for that
/// hydro with `values_m3s.len() >= p`.
///
/// Rule 24: Every hydro ID present in `past_inflows` must exist in the hydro
/// registry.
pub(super) fn check_past_inflows_coverage(data: &ParsedData, ctx: &mut ValidationContext) {
    let lags_enabled = data
        .stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .any(|s| s.state_config.inflow_lags);
    if !lags_enabled {
        return;
    }

    // Lags are 1-based, so the max `lag` value is the PAR order p.
    let max_order_overall: i32 = data
        .inflow_ar_coefficients
        .iter()
        .map(|c| c.lag)
        .max()
        .unwrap_or(0);
    if max_order_overall == 0 {
        return;
    }

    let past_inflows = &data.initial_conditions.past_inflows;

    // Rule 22.
    if past_inflows.is_empty() {
        ctx.add_error(
            ErrorKind::BusinessRuleViolation,
            "initial_conditions.json",
            None::<&str>,
            "inflow_lags is enabled with PAR order > 0 but              initial_conditions.json has no past_inflows entries;              lag initialization requires past inflow values",
        );
        return; // rules 23-24 require non-empty past_inflows
    }

    let mut max_order_per_hydro: HashMap<i32, i32> = HashMap::new();
    for row in &data.inflow_ar_coefficients {
        let entry = max_order_per_hydro.entry(row.hydro_id.0).or_insert(0);
        if row.lag > *entry {
            *entry = row.lag;
        }
    }

    let past_inflows_len: HashMap<i32, usize> = past_inflows
        .iter()
        .map(|pi| (pi.hydro_id.0, pi.values_m3s.len()))
        .collect();

    // Rule 23.
    {
        let mut coverage_violations: Vec<(i32, i32, usize)> = Vec::new(); // (hydro_id, order, provided)
        for (&hydro_id, &order) in &max_order_per_hydro {
            if order == 0 {
                continue;
            }
            let required = usize::try_from(order).unwrap_or(usize::MAX);
            let provided = past_inflows_len.get(&hydro_id).copied().unwrap_or(0);
            if provided < required {
                coverage_violations.push((hydro_id, order, provided));
            }
        }

        // Sort for deterministic output order.
        coverage_violations.sort_unstable_by_key(|&(hid, _, _)| hid);
        for (hydro_id, order, provided) in coverage_violations {
            let entity_str = format!("Hydro {hydro_id}");
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "initial_conditions.json",
                Some(&entity_str),
                format!(
                    "Hydro {hydro_id}: insufficient past_inflows for lag initialization; \
                     PAR order is {order} but initial_conditions.json provides only \
                     {provided} value(s) in past_inflows (need at least {order})"
                ),
            );
        }
    }

    // Rule 24.
    {
        let hydro_registry: HashSet<i32> = data.hydros.iter().map(|h| h.id.0).collect();
        let past_inflow_ids: HashSet<i32> = past_inflows.iter().map(|pi| pi.hydro_id.0).collect();
        let mut unknown_ids: Vec<i32> = past_inflow_ids
            .difference(&hydro_registry)
            .copied()
            .collect();
        unknown_ids.sort_unstable();
        for id in unknown_ids {
            let entity_str = format!("Hydro {id}");
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "initial_conditions.json",
                Some(&entity_str),
                format!(
                    "Hydro {id} appears in past_inflows but does not exist \
                     in the hydro registry (system/hydros.json); \
                     remove the unknown hydro or add it to the registry"
                ),
            );
        }
    }
}

// ── Rule 32: past_inflows season_ids against SeasonMap ───────────────────────

/// Rule 32: when `past_inflows[i].season_ids` is `Some` and the hydro has
/// PAR order > 0, each `season_id` value must exist in the `SeasonMap`.
///
/// Skips the check when `season_map` is `None` — the semantic layer cannot
/// validate season IDs without a `SeasonMap`. Schema-layer length validation
/// (matching `season_ids.len() == values_m3s.len()`) is handled in
/// `cobre-io/src/initial_conditions.rs`.
pub(super) fn check_past_inflows_season_ids(data: &ParsedData, ctx: &mut ValidationContext) {
    let Some(season_map) = &data.stages.policy_graph.season_map else {
        return;
    };

    let mut max_order_per_hydro: HashMap<i32, i32> = HashMap::new();
    for row in &data.inflow_ar_coefficients {
        let entry = max_order_per_hydro.entry(row.hydro_id.0).or_insert(0);
        if row.lag > *entry {
            *entry = row.lag;
        }
    }

    let valid_ids: HashSet<usize> = season_map.seasons.iter().map(|s| s.id).collect();
    let mut sorted_valid_ids: Vec<usize> = valid_ids.iter().copied().collect();
    sorted_valid_ids.sort_unstable();

    for pi in &data.initial_conditions.past_inflows {
        let par_order = max_order_per_hydro
            .get(&pi.hydro_id.0)
            .copied()
            .unwrap_or(0);
        if par_order == 0 {
            continue;
        }

        let Some(season_ids) = &pi.season_ids else {
            continue;
        };

        for &sid in season_ids {
            let sid_usize = sid as usize;
            if !valid_ids.contains(&sid_usize) {
                let entity_str = format!("Hydro {}", pi.hydro_id.0);
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "initial_conditions.json",
                    Some(&entity_str),
                    format!(
                        "Hydro {}: past_inflows.season_ids contains season_id {} which is \
                         not defined in season_definitions; valid season IDs are {:?}",
                        pi.hydro_id.0, sid, sorted_valid_ids,
                    ),
                );
            }
        }
    }
}

// ── Filling-schedule sufficiency ──────────────────────────────────────────────

/// m³/s → hm³ per stage-hour: `3600 s/h ÷ 1e6 m³/hm³`. Duplicated from the
/// solver-side copy on purpose — a shared dependency would break cobre-io's
/// infrastructure-genericity rule; this is redundancy-with-purpose, not drift.
const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;

/// Checks that each filling hydro's minimum accumulation schedule can reach its
/// dead volume before the entry stage:
///
/// ```text
/// Σ_{s ∈ [start_stage_id, entry)} ζ_s · rate_s  ≥  (min_storage_hm3 − seed)
/// ```
///
/// where `ζ_s = (Σ_b blocks[s].duration_hours) · M3S_TO_HM3` and `rate_s` is the
/// per-stage `filling_min_rate_m3s` override from `hydro_bounds` when present,
/// else the entity-level rate.
///
/// One-sided: only under-provisioning is rejected (strict `capacity < required`,
/// never float equality) — surplus capacity merely relaxes the earliest floors to
/// slack, so a two-sided / exact-equality test would reject valid schedules.
///
/// Stage ids index `data.stages.stages` by `Stage::id`, not array position: the
/// override key (`HydroBoundsRow.stage_id`) and `start_stage_id`/`entry_stage_id`
/// are all domain identifiers.
pub(super) fn check_filling_sufficiency(data: &ParsedData, ctx: &mut ValidationContext) {
    let zeta_by_stage: HashMap<i32, f64> = data
        .stages
        .stages
        .iter()
        .map(|s| {
            let duration_hours: f64 = s.blocks.iter().map(|b| b.duration_hours).sum();
            (s.id, duration_hours * M3S_TO_HM3)
        })
        .collect();

    let rate_override: HashMap<(i32, i32), f64> = data
        .hydro_bounds
        .iter()
        .filter_map(|row| {
            row.filling_min_rate_m3s
                .map(|rate| ((row.hydro_id.0, row.stage_id), rate))
        })
        .collect();

    let seed_by_hydro: HashMap<i32, f64> = data
        .initial_conditions
        .filling_storage
        .iter()
        .map(|s| (s.hydro_id.0, s.value_hm3))
        .collect();

    for hydro in &data.hydros {
        let (Some(filling), Some(entry)) = (hydro.filling.as_ref(), hydro.entry_stage_id) else {
            continue;
        };

        let mut capacity = 0.0;
        for stage_id in filling.start_stage_id..entry {
            let Some(&zeta) = zeta_by_stage.get(&stage_id) else {
                continue;
            };
            let rate = rate_override
                .get(&(hydro.id.0, stage_id))
                .copied()
                .unwrap_or(filling.filling_min_rate_m3s);
            capacity += zeta * rate;
        }

        let seed = seed_by_hydro.get(&hydro.id.0).copied().unwrap_or(0.0);
        let required = hydro.min_storage_hm3 - seed;

        if capacity < required {
            let entity_str = format!("Hydro {}", hydro.id.0);
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: filling schedule is insufficient to reach the dead volume \
                     before stage {entry}; cumulative minimum-rate capacity over stages \
                     [{}, {entry}) is {capacity} hm3 but {required} hm3 \
                     (min_storage {} - seed {seed}) is required",
                    filling.start_stage_id, hydro.min_storage_hm3
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
    use super::super::validate_semantic_stages_penalties_scenarios;
    use super::*;
    use crate::{
        scenarios::{
            BlockFactor, InflowArCoefficientRow, InflowSeasonalStatsRow, LoadFactorEntry,
            LoadSeasonalStatsRow,
        },
        stages::StagesData,
        validation::{ErrorKind, ValidationContext},
    };
    use cobre_core::{
        EntityId,
        entities::HydroGenerationModel,
        temporal::{Block, PolicyGraph, PolicyGraphType},
    };

    // ── Local helpers ─────────────────────────────────────────────────────────

    /// Build a `StagesData` with one stage that has one block (index = 0).
    fn make_stages_with_block(stage_id: i32) -> StagesData {
        let mut stage = make_stage(stage_id);
        stage.blocks = vec![Block {
            index: 0,
            name: "FLAT".to_string(),
            duration_hours: 744.0,
        }];
        StagesData {
            stages: vec![stage],
            policy_graph: PolicyGraph {
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                season_map: None,
            },
        }
    }

    // ── Rules 6-7: Penalty ordering ───────────────────────────────────────────

    /// Check 6: `filling_target_violation_cost` (100) >= `max_deficit_cost` (50)
    /// produces a `ModelQuality` warning that filling is not below load deficit.
    #[test]
    fn test_5b_penalty_ordering_filling_not_below_deficit_warns() {
        let mut hydro = make_hydro_ordered_penalties(7);
        hydro.penalties.filling_target_violation_cost = 100.0;
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 50.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "penalty-ordering checks are non-blocking warnings, not errors"
        );
        let warnings = ctx.warnings();
        let check6: Vec<_> = warnings
            .iter()
            .filter(|w| {
                w.kind == ErrorKind::ModelQuality && w.message.contains("should be < deficit_cost")
            })
            .collect();
        assert_eq!(
            check6.len(),
            1,
            "exactly 1 Check-6 ModelQuality warning expected"
        );
        let msg = &check6[0].message;
        assert!(
            msg.contains("filling_target_violation_cost"),
            "message should contain 'filling_target_violation_cost', got: {msg}"
        );
        assert!(
            msg.contains("not as hard as load shedding"),
            "message should contain 'not as hard as load shedding', got: {msg}"
        );
    }

    /// Check 6: `filling_target_violation_cost` (50) < `max_deficit_cost` (100)
    /// emits no warning -- the fill schedule is softer than load shedding.
    #[test]
    fn test_5b_penalty_ordering_filling_below_deficit_no_warn() {
        let mut hydro = make_hydro_ordered_penalties(7);
        hydro.penalties.filling_target_violation_cost = 50.0;
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 100.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let warnings = ctx.warnings();
        let check6 = warnings
            .iter()
            .filter(|w| w.message.contains("should be < deficit_cost"))
            .count();
        assert_eq!(check6, 0, "no Check-6 warning expected, got {check6}");
    }

    /// Check 6: with no bus deficit segments (`max_deficit_cost == 0.0`) the check is
    /// skipped -- there is no deficit comparand even though filling cost is high.
    #[test]
    fn test_5b_penalty_ordering_no_deficit_segment_skips_check6() {
        let mut hydro = make_hydro_ordered_penalties(7);
        hydro.penalties.filling_target_violation_cost = 1000.0;
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let warnings = ctx.warnings();
        let check6 = warnings
            .iter()
            .filter(|w| w.message.contains("should be < deficit_cost"))
            .count();
        assert_eq!(
            check6, 0,
            "Check 6 must be skipped when max_deficit_cost == 0.0, got {check6}"
        );
    }

    /// Check 7: `storage_violation_below_cost` (5) <= `max_deficit_cost` (10)
    /// produces a `ModelQuality` warning that storage-below should outrank load deficit.
    #[test]
    fn test_5b_penalty_storage_below_deficit_warns() {
        let mut hydro = make_hydro_ordered_penalties(7);
        hydro.penalties.storage_violation_below_cost = 5.0;
        // Keep filling below deficit so this isolates the Check-7 warning.
        hydro.penalties.filling_target_violation_cost = 5.0;
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "penalty-ordering checks are non-blocking warnings, not errors"
        );
        let warnings = ctx.warnings();
        let check7: Vec<_> = warnings
            .iter()
            .filter(|w| {
                w.kind == ErrorKind::ModelQuality
                    && w.message.contains("storage_violation_below_cost")
                    && w.message.contains("max(deficit_segment_costs)")
            })
            .collect();
        assert_eq!(
            check7.len(),
            1,
            "exactly 1 Check-7 ModelQuality warning expected"
        );
    }

    // ── Rule 11: FPHA penalty rule ────────────────────────────────────────────

    /// Hydro 3 with Fpha model, `turbined_cost = -0.01` produces a
    /// `BusinessRuleViolation` error with "Hydro 3" and "turbined_cost".
    #[test]
    fn test_5b_fpha_penalty_violated() {
        let mut hydro = make_hydro_ordered_penalties(3);
        hydro.generation_model = HydroGenerationModel::Fpha;
        hydro.penalties.turbined_cost = -0.01; // invalid: must be >= 0
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "exactly 1 BusinessRuleViolation expected"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Hydro 3"),
            "message should contain 'Hydro 3', got: {msg}"
        );
        assert!(
            msg.contains("turbined_cost"),
            "message should contain 'turbined_cost', got: {msg}"
        );
    }

    /// FPHA hydro with `turbined_cost == 0.0` (constant-head) produces no error.
    #[test]
    fn test_5b_fpha_penalty_zero_valid() {
        let mut hydro = make_hydro_ordered_penalties(3);
        hydro.generation_model = HydroGenerationModel::Fpha;
        hydro.penalties.turbined_cost = 0.0; // valid: constant-head plant
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            errors.is_empty(),
            "turbined_cost == 0.0 should be valid for constant-head plants, \
             got: {errors:?}"
        );
    }

    /// FPHA hydro with `turbined_cost == spillage_cost` produces no error.
    #[test]
    fn test_5b_fpha_penalty_equal_spillage_valid() {
        let mut hydro = make_hydro_ordered_penalties(3);
        hydro.generation_model = HydroGenerationModel::Fpha;
        hydro.penalties.turbined_cost = 1.0;
        hydro.penalties.spillage_cost = 1.0; // turbined_cost == spillage_cost is valid
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            errors.is_empty(),
            "turbined_cost == spillage_cost should be valid, got: {errors:?}"
        );
    }

    /// FPHA hydro with `turbined_cost > spillage_cost` produces no error.
    #[test]
    fn test_5b_fpha_penalty_valid() {
        let mut hydro = make_hydro_ordered_penalties(4);
        hydro.generation_model = HydroGenerationModel::Fpha;
        hydro.penalties.turbined_cost = 2.0;
        hydro.penalties.spillage_cost = 1.0;
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            errors.is_empty(),
            "valid FPHA penalty ordering should produce no BusinessRuleViolation, \
             got: {errors:?}"
        );
    }

    // ── Rule 12: Inflow std_m3s = 0.0 warning ────────────────────────────────

    /// `std_m3s = 0.0` produces a `ModelQuality` warning (deterministic inflow).
    #[test]
    fn test_5b_inflow_std_zero_warning() {
        let stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 0.0, // triggers ModelQuality warning
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            stats,
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "std_m3s=0.0 should produce warning, not error, got: {:?}",
            ctx.errors()
        );
        let warnings = ctx.warnings();
        assert!(
            !warnings.is_empty(),
            "std_m3s=0.0 should produce at least 1 ModelQuality warning"
        );
        assert!(
            warnings.iter().any(|w| w.kind == ErrorKind::ModelQuality),
            "should have ModelQuality warning"
        );
    }

    // ── Rule 13: residual_std_ratio consistency ───────────────────────────────

    /// Two AR coefficient rows for the same `(hydro, stage)` with identical
    /// `residual_std_ratio` values produce no `InvalidValue` error.
    #[test]
    fn test_5b_residual_std_ratio_consistent_no_error() {
        let ar_rows = vec![
            InflowArCoefficientRow {
                hydro_id: EntityId::from(1),
                stage_id: 0,
                lag: 1,
                coefficient: 0.5,
                residual_std_ratio: 0.85,
            },
            InflowArCoefficientRow {
                hydro_id: EntityId::from(1),
                stage_id: 0,
                lag: 2,
                coefficient: 0.3,
                residual_std_ratio: 0.85, // same ratio as lag 1
            },
        ];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            ar_rows,
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let errors = ctx.errors();
        let invalid_value_errors: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::InvalidValue && e.message.contains("residual_std_ratio")
            })
            .collect();
        assert!(
            invalid_value_errors.is_empty(),
            "consistent residual_std_ratio should produce no InvalidValue errors, \
             got: {invalid_value_errors:?}"
        );
    }

    /// Two AR coefficient rows for the same `(hydro, stage)` with different
    /// `residual_std_ratio` values produce an `InvalidValue` error whose message
    /// contains "residual_std_ratio" and "inconsistent".
    #[test]
    fn test_5b_residual_std_ratio_inconsistent_error() {
        let ar_rows = vec![
            InflowArCoefficientRow {
                hydro_id: EntityId::from(1),
                stage_id: 0,
                lag: 1,
                coefficient: 0.5,
                residual_std_ratio: 0.85,
            },
            InflowArCoefficientRow {
                hydro_id: EntityId::from(1),
                stage_id: 0,
                lag: 2,
                coefficient: 0.3,
                residual_std_ratio: 0.90, // different ratio — triggers V-AR-4
            },
        ];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            ar_rows,
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let errors = ctx.errors();
        let invalid_value_errors: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert!(
            !invalid_value_errors.is_empty(),
            "inconsistent residual_std_ratio should produce at least one InvalidValue error"
        );
        let ratio_error = invalid_value_errors.iter().find(|e| {
            e.message.contains("residual_std_ratio") && e.message.contains("inconsistent")
        });
        assert!(
            ratio_error.is_some(),
            "InvalidValue error message should contain 'residual_std_ratio' and \
             'inconsistent', got: {invalid_value_errors:?}"
        );
    }

    // ── Rules 17-18: Load factor consistency ─────────────────────────────────

    /// `LoadFactorEntry` with a `block_id` not present in the stage's blocks
    /// produces 1 `BusinessRuleViolation` error.
    #[test]
    fn test_5b_load_factors_invalid_block_id() {
        let mut data = make_data_5b(
            vec![],
            make_stages_with_block(0),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        // Stage 0 has block index 0 only; block_id=99 is invalid.
        data.load_factors = vec![LoadFactorEntry {
            bus_id: EntityId::from(1),
            stage_id: 0,
            block_factors: vec![BlockFactor {
                block_id: 99,
                factor: 1.0,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert_eq!(
            errors.len(),
            1,
            "expected 1 BusinessRuleViolation, got: {errors:?}"
        );
        assert!(
            errors[0].file.to_string_lossy().contains("load_factors"),
            "error should reference load_factors.json"
        );
        assert!(
            errors[0].message.contains("99"),
            "message should mention invalid block_id 99"
        );
    }

    /// `LoadFactorEntry` for a `(bus_id, stage_id)` where `load_seasonal_stats`
    /// has `std_mw == 0.0` produces 1 `ModelQuality` warning.
    #[test]
    fn test_5b_load_factors_deterministic_bus_warning() {
        let mut data = make_data_5b(
            vec![],
            make_stages_with_block(0),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        // Bus 1, stage 0 with std_mw == 0.0 (deterministic load).
        data.load_seasonal_stats = vec![LoadSeasonalStatsRow {
            bus_id: EntityId::from(1),
            stage_id: 0,
            mean_mw: 100.0,
            std_mw: 0.0,
        }];
        data.load_factors = vec![LoadFactorEntry {
            bus_id: EntityId::from(1),
            stage_id: 0,
            block_factors: vec![BlockFactor {
                block_id: 0,
                factor: 1.0,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "deterministic load warning should not produce an error, got: {:?}",
            ctx.errors()
        );
        let warnings = ctx.warnings();
        let relevant: Vec<_> = warnings
            .iter()
            .filter(|w| w.kind == ErrorKind::ModelQuality)
            .filter(|w| w.file.to_string_lossy().contains("load_factors"))
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected 1 ModelQuality warning for load_factors.json, got: {warnings:?}"
        );
    }

    /// Empty `load_factors` produces zero load-related diagnostics.
    #[test]
    fn test_5b_load_factors_empty_no_errors() {
        let data = make_data_5b(
            vec![],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        // load_factors is already empty in make_data_5b
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let load_factor_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.file.to_string_lossy().contains("load_factors"))
            .collect();
        let load_factor_warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| w.file.to_string_lossy().contains("load_factors"))
            .collect();
        assert!(
            load_factor_errors.is_empty() && load_factor_warnings.is_empty(),
            "empty load_factors should produce no load-related diagnostics; \
             errors: {load_factor_errors:?}, warnings: {load_factor_warnings:?}"
        );
    }

    // ── Estimation prerequisites (Rules 19-21) ────────────────────────────────

    /// Given `inflow_history` present, `inflow_seasonal_stats` absent, and
    /// `stages.json` WITHOUT `season_definitions`, validation produces a
    /// `BusinessRuleViolation` mentioning "season_definitions is required".
    #[test]
    fn test_estimation_requires_season_definitions() {
        let history = make_history_rows(1, 12);
        let stages = make_stages_with_seasons(12, /*with_season_map=*/ false);
        let data = make_data_estimation(vec![make_hydro(1, None)], stages, history);

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let matching: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("season_definitions is required")
            })
            .collect();
        assert!(
            !matching.is_empty(),
            "expected a BusinessRuleViolation about season_definitions, got errors: {:?}",
            ctx.errors()
        );
    }

    /// Given `inflow_history` with only 3 observations for one `(hydro, season)`,
    /// validation produces a `ModelQuality` warning containing "has 3 observations".
    #[test]
    fn test_estimation_warns_low_observations() {
        // 3 observations for hydro 1: one per January (season 0) over 3 years.
        let history: Vec<crate::scenarios::InflowHistoryRow> = (0..3)
            .map(|y| crate::scenarios::InflowHistoryRow {
                hydro_id: EntityId::from(1),
                date: chrono::NaiveDate::from_ymd_opt(2000 + y, 1, 15).unwrap(),
                value_m3s: 100.0,
            })
            .collect();

        // 3 years × 12 months = 36 stages, with season_map present.
        let stages = make_stages_with_seasons(36, true);
        let data = make_data_estimation(vec![make_hydro(1, None)], stages, history);

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let matching: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| {
                w.kind == ErrorKind::ModelQuality && w.message.contains("has 3 observations")
            })
            .collect();
        assert!(
            !matching.is_empty(),
            "expected a ModelQuality warning about 3 observations, got warnings: {:?}",
            ctx.warnings()
        );
    }

    /// Given `inflow_history` with observations for hydro 1 only, but `hydros`
    /// containing hydro 1 and hydro 2, validation produces a
    /// `BusinessRuleViolation` for hydro 2.
    #[test]
    fn test_estimation_error_missing_hydro() {
        let history = make_history_rows(1, 36); // only hydro 1
        let stages = make_stages_with_seasons(36, true);
        let hydros = vec![make_hydro(1, None), make_hydro(2, None)];
        let data = make_data_estimation(hydros, stages, history);

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let matching: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("hydro 2 has no observations")
            })
            .collect();
        assert!(
            !matching.is_empty(),
            "expected a BusinessRuleViolation for hydro 2, got errors: {:?}",
            ctx.errors()
        );
    }

    /// When BOTH `inflow_seasonal_stats` and `inflow_ar_coefficients` are
    /// non-empty, no estimation-related errors or warnings are produced.
    #[test]
    fn test_no_estimation_when_stats_and_coefficients_present() {
        let history = make_history_rows(1, 12);
        let stages = make_stages_with_seasons(12, false); // no season_map

        // Provide both stats AND AR coefficients to fully deactivate estimation.
        let stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            mean_m3s: 500.0,
            std_m3s: 50.0,
        }];
        let ar_coefficients = vec![make_ar_row(1, 0, 1)];

        let mut data = make_data_estimation(vec![make_hydro(1, None)], stages, history);
        data.inflow_seasonal_stats = stats;
        data.inflow_ar_coefficients = ar_coefficients;

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        // No estimation errors — the estimation path is not active.
        let estimation_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.file.to_string_lossy().contains("inflow_history.parquet"))
            .collect();
        let estimation_warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| w.file.to_string_lossy().contains("inflow_history.parquet"))
            .collect();
        assert!(
            estimation_errors.is_empty() && estimation_warnings.is_empty(),
            "stats+coefficients present should disable estimation checks; \
             errors: {estimation_errors:?}, warnings: {estimation_warnings:?}"
        );
    }

    /// When `inflow_seasonal_stats` is present but `inflow_ar_coefficients`
    /// is absent, estimation IS active and season_definitions is required.
    #[test]
    fn test_estimation_active_when_stats_present_but_coefficients_absent() {
        let history = make_history_rows(1, 12);
        let stages = make_stages_with_seasons(12, false); // no season_map

        let stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            mean_m3s: 500.0,
            std_m3s: 50.0,
        }];

        let mut data = make_data_estimation(vec![make_hydro(1, None)], stages, history);
        data.inflow_seasonal_stats = stats;
        // AR coefficients NOT provided — estimation should be active.

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        // Should produce a season_definitions error (Rule 19).
        let estimation_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.file.to_string_lossy().contains("inflow_history.parquet")
                    && e.message.contains("season_definitions")
            })
            .collect();
        assert!(
            !estimation_errors.is_empty(),
            "stats present without coefficients should trigger estimation checks; \
             got errors: {:?}",
            ctx.errors()
        );
    }

    // ── Rules 22-24: Past inflows coverage ───────────────────────────────────

    /// Rule 22: inflow_lags true, PAR order 3, empty past_inflows → one
    /// `BusinessRuleViolation` mentioning "inflow_lags is enabled" and
    /// "initial_conditions.json".
    #[test]
    fn test_rule22_lags_enabled_no_past_inflows_errors() {
        let ar_rows = vec![
            make_ar_row(1, 0, 1),
            make_ar_row(1, 0, 2),
            make_ar_row(1, 0, 3),
        ];
        let data = make_data_past_inflows(
            vec![make_hydro(1, None)],
            true,
            vec![], // empty past_inflows
            ar_rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let matching: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("inflow_lags is enabled")
            })
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "expected exactly one rule-22 BusinessRuleViolation, got: {:?}",
            ctx.errors()
        );
        assert!(
            matching[0]
                .file
                .to_string_lossy()
                .contains("initial_conditions.json"),
            "error file should reference initial_conditions.json"
        );
    }

    /// Rule 23: inflow_lags true, hydro 1 PAR order 3, past_inflows has 3 values
    /// → no rule-22/23/24 violations.
    #[test]
    fn test_rule23_sufficient_past_inflows_no_error() {
        let ar_rows = vec![
            make_ar_row(1, 0, 1),
            make_ar_row(1, 0, 2),
            make_ar_row(1, 0, 3),
        ];
        let past = vec![cobre_core::HydroPastInflows {
            hydro_id: EntityId::from(1),
            values_m3s: vec![300.0, 200.0, 100.0], // 3 values >= PAR order 3
            season_ids: None,
        }];
        let data = make_data_past_inflows(vec![make_hydro(1, None)], true, past, ar_rows);
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let lag_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file.to_string_lossy().contains("initial_conditions.json")
            })
            .collect();
        assert!(
            lag_errors.is_empty(),
            "sufficient past_inflows should produce no errors, got: {lag_errors:?}"
        );
    }

    /// Rule 23: inflow_lags true, hydro 1 PAR order 3, past_inflows has only 2
    /// values → one `BusinessRuleViolation` for hydro 1 mentioning
    /// "insufficient past_inflows".
    #[test]
    fn test_rule23_insufficient_past_inflows_errors() {
        let ar_rows = vec![
            make_ar_row(1, 0, 1),
            make_ar_row(1, 0, 2),
            make_ar_row(1, 0, 3),
        ];
        let past = vec![cobre_core::HydroPastInflows {
            hydro_id: EntityId::from(1),
            values_m3s: vec![200.0, 100.0], // only 2 values, need 3
            season_ids: None,
        }];
        let data = make_data_past_inflows(vec![make_hydro(1, None)], true, past, ar_rows);
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let coverage_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("Hydro 1")
                    && e.message.contains("insufficient past_inflows")
            })
            .collect();
        assert!(
            !coverage_errors.is_empty(),
            "insufficient past_inflows should produce a BusinessRuleViolation for Hydro 1; \
             got errors: {:?}",
            ctx.errors()
        );
    }

    /// Rules 22-24 are skipped when no stage has `inflow_lags: true`.
    #[test]
    fn test_rules_skip_when_lags_disabled() {
        let ar_rows = vec![make_ar_row(1, 0, 1), make_ar_row(1, 0, 2)];
        let data = make_data_past_inflows(
            vec![make_hydro(1, None)],
            false,  // lags disabled
            vec![], // empty past_inflows — would trigger rule 22 if lags enabled
            ar_rows,
        );
        let mut ctx = ValidationContext::new();
        check_past_inflows_coverage(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "lags disabled should produce no rule-22/23/24 errors; got: {:?}",
            ctx.errors()
        );
    }

    /// Rules 22-24 are skipped when `inflow_ar_coefficients` is empty.
    #[test]
    fn test_rules_skip_when_par_order_zero() {
        let data = make_data_past_inflows(
            vec![make_hydro(1, None)],
            true,   // lags enabled
            vec![], // empty past_inflows — would trigger rule 22 if AR coefficients present
            vec![], // no AR coefficients -> max_order == 0, early return
        );
        let mut ctx = ValidationContext::new();
        check_past_inflows_coverage(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "no AR coefficients should produce no rule-22/23/24 errors; got: {:?}",
            ctx.errors()
        );
    }

    /// Rule 24: hydro ID in past_inflows that does not exist in the hydro registry
    /// produces a `BusinessRuleViolation` mentioning the unknown hydro ID.
    #[test]
    fn test_rule24_unknown_hydro_in_past_inflows_errors() {
        // past_inflows contains hydro 99, which is not in the registry.
        let past = vec![
            cobre_core::HydroPastInflows {
                hydro_id: EntityId::from(1),
                values_m3s: vec![100.0],
                season_ids: None,
            },
            cobre_core::HydroPastInflows {
                hydro_id: EntityId::from(99), // unknown
                values_m3s: vec![50.0],
                season_ids: None,
            },
        ];
        // Provide enough AR rows so rule 22 and 23 are satisfied for hydro 1.
        let ar_rows = vec![make_ar_row(1, 0, 1)];
        let data = make_data_past_inflows(
            vec![make_hydro(1, None)], // only hydro 1 in registry
            true,
            past,
            ar_rows,
        );
        let mut ctx = ValidationContext::new();
        check_past_inflows_coverage(&data, &mut ctx);

        let rule24_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation && e.message.contains("Hydro 99")
            })
            .collect();
        assert!(
            !rule24_errors.is_empty(),
            "unknown hydro 99 in past_inflows should produce a BusinessRuleViolation; \
             got errors: {:?}",
            ctx.errors()
        );
    }

    // ── Rule 32: past_inflows season_ids against SeasonMap ───────────────────

    /// Rule 32: `past_inflows[i].season_ids` contains an ID not in the `SeasonMap`
    /// → `BusinessRuleViolation` mentioning the invalid season ID.
    #[test]
    fn test_past_inflows_season_ids_invalid_season() {
        let past = vec![cobre_core::HydroPastInflows {
            hydro_id: EntityId::from(1),
            values_m3s: vec![300.0, 200.0],
            season_ids: Some(vec![0, 99]), // season_id 99 is invalid (only 0..4 exist)
        }];
        let ar_rows = vec![make_ar_row(1, 0, 1), make_ar_row(1, 0, 2)];
        let data = make_data_past_inflows_with_season_map(
            vec![make_hydro(1, None)],
            past,
            ar_rows,
            5, // seasons 0..4 exist
        );
        let mut ctx = ValidationContext::new();
        check_past_inflows_season_ids(&data, &mut ctx);

        let rule32_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("season_id")
                    && e.message.contains("99")
            })
            .collect();
        assert!(
            !rule32_errors.is_empty(),
            "invalid season_id 99 should produce a BusinessRuleViolation; \
             got errors: {:?}",
            ctx.errors()
        );
    }

    /// Rule 32: all `season_ids` are valid → no `BusinessRuleViolation`.
    #[test]
    fn test_past_inflows_season_ids_valid() {
        let past = vec![cobre_core::HydroPastInflows {
            hydro_id: EntityId::from(1),
            values_m3s: vec![300.0, 200.0],
            season_ids: Some(vec![3, 2]), // both exist in seasons 0..4
        }];
        let ar_rows = vec![make_ar_row(1, 0, 1), make_ar_row(1, 0, 2)];
        let data = make_data_past_inflows_with_season_map(
            vec![make_hydro(1, None)],
            past,
            ar_rows,
            5, // seasons 0..4 exist
        );
        let mut ctx = ValidationContext::new();
        check_past_inflows_season_ids(&data, &mut ctx);

        let rule32_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file.to_string_lossy().contains("initial_conditions.json")
                    && e.message.contains("season_id")
            })
            .collect();
        assert!(
            rule32_errors.is_empty(),
            "valid season_ids should produce no rule-32 errors; got: {:?}",
            ctx.errors()
        );
    }

    /// Rule 32 is skipped when `season_map` is `None`.
    #[test]
    fn test_past_inflows_season_ids_no_season_map_skipped() {
        let past = vec![cobre_core::HydroPastInflows {
            hydro_id: EntityId::from(1),
            values_m3s: vec![300.0],
            season_ids: Some(vec![999]), // would be invalid if season_map were present
        }];
        let ar_rows = vec![make_ar_row(1, 0, 1)];
        // make_data_past_inflows uses season_map: None
        let data = make_data_past_inflows(vec![make_hydro(1, None)], true, past, ar_rows);
        let mut ctx = ValidationContext::new();
        check_past_inflows_season_ids(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "no season_map means rule 32 should be skipped; got: {:?}",
            ctx.errors()
        );
    }

    // ── F2-002: External scheme requires external scenario files ──────────────

    /// `config.training.scenario_source.inflow.scheme = "external"` with no
    /// `external_scenarios` data produces an error referencing `"config.json"` and
    /// field `"training.scenario_source.inflow"`.
    #[test]
    fn test_training_external_inflow_without_file_is_error() {
        let mut data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 75.0)],
            vec![],
            vec![],
            None,
        );
        data.config = config_with_training_external_inflow();
        data.external_scenarios = vec![]; // no external inflow file

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let errors = ctx.errors();
        let matching: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file == std::path::Path::new("config.json")
                    && e.entity
                        .as_deref()
                        .is_some_and(|f| f.contains("training.scenario_source.inflow"))
            })
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "expected 1 error for missing external inflow file (training), \
             got: {errors:?}"
        );
    }

    /// `config.simulation.scenario_source.load.scheme = "external"` with no
    /// `external_load_scenarios` data produces an error referencing `"config.json"`
    /// and field `"simulation.scenario_source.load"`.
    #[test]
    fn test_simulation_external_load_without_file_is_error() {
        let mut data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 75.0)],
            vec![],
            vec![],
            None,
        );
        data.config = config_with_simulation_external_load();
        data.external_load_scenarios = vec![]; // no external load file

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let errors = ctx.errors();
        let matching: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file == std::path::Path::new("config.json")
                    && e.entity
                        .as_deref()
                        .is_some_and(|f| f.contains("simulation.scenario_source.load"))
            })
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "expected 1 error for missing external load file (simulation), \
             got: {errors:?}"
        );
    }

    /// Training uses External inflow and the external file is present: no error.
    #[test]
    fn test_training_external_inflow_with_file_is_ok() {
        use cobre_core::scenario::ExternalScenarioRow;
        let mut data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 75.0)],
            vec![],
            vec![],
            None,
        );
        data.config = config_with_training_external_inflow();
        // Provide at least one row so the file is considered non-empty.
        data.external_scenarios = vec![ExternalScenarioRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            scenario_id: 1,
            value_m3s: 10.0,
        }];

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let errors = ctx.errors();
        let external_errors: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file == std::path::Path::new("config.json")
            })
            .collect();
        assert!(
            external_errors.is_empty(),
            "no external-file errors expected when file is present, \
             got: {external_errors:?}"
        );
    }

    // ── Rule 33: Filling-schedule sufficiency ─────────────────────────────────

    /// Build a `StagesData` whose stages have ids `ids`, each carrying a single
    /// block of `duration_hours`. The ζ for every stage is therefore
    /// `duration_hours · M3S_TO_HM3`.
    fn make_stages_with_block_duration(ids: &[i32], duration_hours: f64) -> StagesData {
        let stages = ids
            .iter()
            .map(|&id| {
                let mut stage = make_stage(id);
                stage.blocks = vec![Block {
                    index: 0,
                    name: "FLAT".to_string(),
                    duration_hours,
                }];
                stage
            })
            .collect();
        StagesData {
            stages,
            policy_graph: PolicyGraph {
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                season_map: None,
            },
        }
    }

    /// Build a filling hydro with dead volume `min_storage_hm3`, filling window
    /// `[start_stage_id, entry_stage_id)`, and entity-level `filling_min_rate_m3s`.
    fn make_filling_hydro(
        id: i32,
        min_storage_hm3: f64,
        start_stage_id: i32,
        entry_stage_id: i32,
        filling_min_rate_m3s: f64,
    ) -> cobre_core::entities::Hydro {
        use cobre_core::entities::FillingConfig;
        let mut h = make_hydro_ordered_penalties(id);
        h.min_storage_hm3 = min_storage_hm3;
        h.entry_stage_id = Some(entry_stage_id);
        h.filling = Some(FillingConfig {
            start_stage_id,
            filling_min_rate_m3s,
        });
        h
    }

    /// Under-provisioned: two filling stages (ζ = 0.0036·720 = 2.592 each) at
    /// rate 1.0 give capacity 5.184 < 60.0 → one `BusinessRuleViolation`.
    #[test]
    fn test_filling_sufficiency_underprovisioned_errors() {
        let hydro = make_filling_hydro(7, 60.0, 2, 4, 1.0);
        let data = make_data_5b(
            vec![hydro],
            make_stages_with_block_duration(&[2, 3], 720.0),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let relevant: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file == std::path::Path::new("system/hydros.json")
                    && e.message.contains("filling schedule is insufficient")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly 1 sufficiency error, got: {:?}",
            ctx.errors()
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Hydro 7"),
            "message should name Hydro 7, got: {msg}"
        );
        assert!(
            msg.contains("5.184"),
            "message should contain the capacity 5.184, got: {msg}"
        );
        assert!(
            msg.contains("60"),
            "message should contain the required 60 shortfall, got: {msg}"
        );
    }

    /// Sufficient: same two stages at rate 20.0 give capacity 103.68 >= 60.0 →
    /// no sufficiency error.
    #[test]
    fn test_filling_sufficiency_sufficient_no_error() {
        let hydro = make_filling_hydro(7, 60.0, 2, 4, 20.0);
        let data = make_data_5b(
            vec![hydro],
            make_stages_with_block_duration(&[2, 3], 720.0),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let relevant: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.message.contains("filling schedule is insufficient"))
            .collect();
        assert!(
            relevant.is_empty(),
            "capacity 103.68 >= 60.0 should emit no sufficiency error, got: {relevant:?}"
        );
    }

    /// Over-provisioned: capacity strictly greater than `min_storage − seed`
    /// must not be rejected (one-sided check).
    #[test]
    fn test_filling_sufficiency_overprovisioned_no_error() {
        // Capacity = 2·2.592·100 = 518.4, far above min_storage 60.0.
        let hydro = make_filling_hydro(7, 60.0, 2, 4, 100.0);
        let data = make_data_5b(
            vec![hydro],
            make_stages_with_block_duration(&[2, 3], 720.0),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let relevant: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.message.contains("filling schedule is insufficient"))
            .collect();
        assert!(
            relevant.is_empty(),
            "over-provisioning must never be rejected (one-sided), got: {relevant:?}"
        );
        assert!(
            !ctx.has_errors(),
            "an over-provisioned filling schedule should produce no errors at all, got: {:?}",
            ctx.errors()
        );
    }

    /// A non-filling hydro (`filling: None`) is ignored by the sufficiency check
    /// even when its `min_storage_hm3` is large.
    #[test]
    fn test_filling_sufficiency_ignores_non_filling_hydro() {
        let mut hydro = make_hydro_ordered_penalties(7);
        // Large dead volume, but no filling config: the check must skip it.
        hydro.min_storage_hm3 = 60.0;
        hydro.entry_stage_id = Some(4);
        assert!(hydro.filling.is_none());
        let data = make_data_5b(
            vec![hydro],
            make_stages_with_block_duration(&[2, 3], 720.0),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let relevant: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.message.contains("filling schedule is insufficient"))
            .collect();
        assert!(
            relevant.is_empty(),
            "non-filling hydro must emit no sufficiency diagnostic, got: {relevant:?}"
        );
    }
}
