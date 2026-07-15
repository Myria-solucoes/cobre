//! Layer 5b — scenario, penalty, and probability-data validation.

use std::collections::{BTreeMap, HashMap, HashSet};

use cobre_core::Hydro;
use cobre_stochastic::par::{
    AnnualParams, ClosureRejection, check_stationarity, check_stationarity_annual,
};

use super::super::{ErrorKind, ValidationContext, schema::ParsedData};

// ── Rules 6-10: Penalty ordering ──────────────────────────────────────────────

/// Checks the penalty hierarchy ordering across all hydros and buses.
///
/// Emits one `ModelQuality` warning per violated ordering check, aggregating
/// all violating entities into a single warning with the count and worst-case ID.
// Rationale: five independent ordering rules, each a full entity pass with its
// own worst-case aggregation; per-rule helpers would not reduce the line count.
#[allow(clippy::too_many_lines)]
pub(super) fn check_penalty_ordering(data: &ParsedData, ctx: &mut ValidationContext) {
    let max_deficit_cost: f64 = data
        .buses
        .iter()
        .flat_map(|b| b.deficit_segments.iter().map(|s| s.cost_per_mwh))
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0);

    // Skipped with no deficit segments (max == 0.0): there is then no comparand.
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

    {
        let max_cv = |h: &Hydro| {
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

        if !data.hydros.is_empty()
            && max_deficit_cost <= max_constraint_cost
            && let Some(worst_hydro) = data.hydros.iter().max_by(|a, b| {
                max_cv(a)
                    .partial_cmp(&max_cv(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
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

    {
        if !data.hydros.is_empty() {
            let min_cv = |h: &Hydro| {
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

            if min_constraint_cost <= max_resource_cost
                && let Some(worst_hydro) = data.hydros.iter().min_by(|a, b| {
                    min_cv(a)
                        .partial_cmp(&min_cv(b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            {
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

// ── Rule 12: Scenario model rules ───────────────────────────────────────────
//
// Rule 13 is retired; the number is never reused — rules 14-35 are referenced
// by number elsewhere in this module and in the crate-level rule catalogue
// (`validation/semantic/mod.rs`).

/// Validates inflow model standard deviation.
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
}

// ── Rule 35: Hard stationarity gate on user-supplied AR coefficients ─────────

/// Gates user-supplied `inflow_ar_coefficients.parquet` rows for stationarity
/// via the periodic-ACF closure (`cobre_stochastic::par::closure`).
///
/// Runs only when `data.inflow_ar_coefficients` is non-empty -- the
/// external-input path. The internal fitting/estimation path (triggered from
/// `inflow_history.parquet` when no coefficient rows are supplied) keeps its
/// own fallbacks and is not gated here.
///
/// For each hydro present in `inflow_ar_coefficients` (ascending `hydro_id`),
/// resolves seasons via [`crate::scenarios::resolve_stage_seasons`] --
/// including its no-`season_map` fallback (dense-ranked distinct per-stage
/// `season_id`s) -- then groups lag rows by season: one representative stage
/// per season, mirroring
/// [`crate::scenarios::populate_derived_residual_ratios`]'s grouping, since
/// stages sharing a season carry identical `ψ*`. Calls
/// [`check_stationarity_annual`] when any season carries an annual component
/// (`inflow_annual_components.parquet`), else [`check_stationarity`]. Every
/// [`ClosureRejection`] becomes an `InvalidValue` error naming the offending
/// season/lag and the failing quantity.
///
/// A hydro whose order-bearing coefficients reference a stage with no
/// resolvable season (no `season_map` AND no usable per-stage `season_id` --
/// the same condition under which
/// [`crate::scenarios::populate_derived_residual_ratios`] itself errors) gets
/// a `BusinessRuleViolation` naming the unresolved stage(s), and is skipped
/// for the stationarity check (other hydros still run). Bare per-stage
/// `season_id`s with no `season_map` (the fallback) resolve cleanly and ARE
/// gated. Never silently skipped.
pub(super) fn check_par_stationarity(data: &ParsedData, ctx: &mut ValidationContext) {
    if data.inflow_ar_coefficients.is_empty() {
        return;
    }

    let (stage_to_season, n_seasons) = crate::scenarios::resolve_stage_seasons(
        &data.stages.stages,
        data.stages.policy_graph.season_map.as_ref(),
    );

    let seasonal_std_by_key: HashMap<(i32, i32), f64> = data
        .inflow_seasonal_stats
        .iter()
        .map(|row| ((row.hydro_id.0, row.stage_id), row.std_m3s))
        .collect();
    let annual_by_key: HashMap<(i32, i32), AnnualParams> = data
        .inflow_annual_components
        .iter()
        .map(|row| {
            (
                (row.hydro_id.0, row.stage_id),
                AnnualParams {
                    coefficient: row.annual_coefficient,
                    sigma_a: row.annual_std_m3s,
                },
            )
        })
        .collect();

    let mut psi_by_hydro_stage: BTreeMap<i32, BTreeMap<i32, Vec<f64>>> = BTreeMap::new();
    for row in &data.inflow_ar_coefficients {
        psi_by_hydro_stage
            .entry(row.hydro_id.0)
            .or_default()
            .entry(row.stage_id)
            .or_default()
            .push(row.coefficient);
    }

    for (hydro_id, stage_psi) in psi_by_hydro_stage {
        let unresolved: Vec<i32> = stage_psi
            .keys()
            .filter(|stage_id| !stage_to_season.contains_key(stage_id))
            .copied()
            .collect();
        if !unresolved.is_empty() {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "scenarios/inflow_ar_coefficients.parquet",
                Some(format!("Hydro {hydro_id}")),
                format!(
                    "Hydro {hydro_id}: PAR stationarity gate cannot resolve a season \
                     for stage(s) {unresolved:?} with order-bearing AR coefficients \
                     (no season_definitions and no per-stage season_id); add \
                     season_definitions to stages.json or a season_id on the \
                     affected stage(s)"
                ),
            );
            continue;
        }

        let mut orders = vec![0_usize; n_seasons];
        let mut psi_by_season = vec![Vec::new(); n_seasons];
        let mut seasonal_std = vec![0.0_f64; n_seasons];
        let mut annual: Vec<Option<AnnualParams>> = vec![None; n_seasons];
        let mut season_seen = vec![false; n_seasons];

        for (stage_id, psi) in stage_psi {
            let Some(&season) = stage_to_season.get(&stage_id) else {
                continue;
            };
            if season_seen[season] {
                continue;
            }
            season_seen[season] = true;
            orders[season] = psi.len();
            seasonal_std[season] = seasonal_std_by_key
                .get(&(hydro_id, stage_id))
                .copied()
                .unwrap_or(0.0);
            annual[season] = annual_by_key.get(&(hydro_id, stage_id)).copied();
            psi_by_season[season] = psi;
        }

        let has_annual = annual.iter().any(Option::is_some);
        let result = if has_annual {
            check_stationarity_annual(&psi_by_season, &orders, &annual, &seasonal_std, n_seasons)
        } else {
            check_stationarity(&psi_by_season, &orders, n_seasons)
        };

        if let Err(rejection) = result {
            let entity_str = format!("Hydro {hydro_id}");
            ctx.add_error(
                ErrorKind::InvalidValue,
                "scenarios/inflow_ar_coefficients.parquet",
                Some(&entity_str),
                describe_par_rejection(hydro_id, &rejection),
            );
        }
    }
}

/// Formats a [`ClosureRejection`] into a message naming the offending
/// season/lag and the failing quantity, for [`check_par_stationarity`].
fn describe_par_rejection(hydro_id: i32, rejection: &ClosureRejection) -> String {
    match rejection {
        ClosureRejection::SingularClosure => format!(
            "Hydro {hydro_id}: PAR stationarity gate found the periodic-ACF closure \
             singular for the AR coefficients in inflow_ar_coefficients.parquet"
        ),
        ClosureRejection::AutocorrelationOutOfRange { season, lag, rho } => format!(
            "Hydro {hydro_id} season {season}: PAR stationarity gate rejected \
             inflow_ar_coefficients.parquet -- implied autocorrelation ρ(lag {lag}) \
             = {rho} is outside [-1, 1]"
        ),
        ClosureRejection::NonPositiveResidualVariance { season, r_squared } => format!(
            "Hydro {hydro_id} season {season}: PAR stationarity gate rejected \
             inflow_ar_coefficients.parquet -- implied residual variance r² \
             = {r_squared} is at or below the numerical floor"
        ),
        ClosureRejection::NonStationaryMonodromy { spectral_radius } => format!(
            "Hydro {hydro_id}: PAR stationarity gate rejected \
             inflow_ar_coefficients.parquet -- periodic monodromy spectral radius \
             = {spectral_radius} is at or above 1"
        ),
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

    if data.stages.policy_graph.season_map.is_none() {
        ctx.add_error(
            ErrorKind::BusinessRuleViolation,
            "scenarios/inflow_history.parquet",
            None::<&str>,
            "season_definitions is required in stages.json when estimating from \
             inflow_history.parquet; add a season_definitions section to stages.json",
        );
    }

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

    // Skipped when season_map is None: Rule 19 already errored, and running here
    // would cascade a confusing second diagnostic.
    if data.stages.policy_graph.season_map.is_some() {
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
            BlockFactor, InflowAnnualComponentRow, InflowArCoefficientRow, InflowHistoryRow,
            InflowSeasonalStatsRow, LoadFactorEntry, LoadSeasonalStatsRow,
        },
        stages::StagesData,
        validation::{ErrorKind, ValidationContext},
    };
    use cobre_core::{
        EntityId, Hydro,
        entities::HydroGenerationModel,
        temporal::{
            Block, PolicyGraph, PolicyGraphType, SeasonCycleType, SeasonDefinition, SeasonMap,
        },
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

    // ── Rule 35: PAR stationarity gate ────────────────────────────────────────

    /// Build a `SeasonMap` with exactly `n` seasons, ids `0..n`.
    fn season_map_n(n: usize) -> SeasonMap {
        SeasonMap {
            cycle_type: SeasonCycleType::Monthly,
            seasons: (0..n)
                .map(|i| SeasonDefinition {
                    id: i,
                    label: format!("Season{i}"),
                    month_start: (i % 12 + 1) as u32,
                    day_start: None,
                    month_end: None,
                    day_end: None,
                })
                .collect(),
        }
    }

    /// Build a `StagesData` with `n` stages (ids `0..n`), each pinned 1:1 to
    /// its own season (`stage.id == season_id`), and a matching `n`-season
    /// `SeasonMap`.
    fn stages_one_stage_per_season(n: usize) -> StagesData {
        let stages = (0..n as i32)
            .map(|id| {
                let mut stage = make_stage(id);
                stage.season_id = Some(id as usize);
                stage
            })
            .collect();
        StagesData {
            stages,
            policy_graph: PolicyGraph {
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                season_map: Some(season_map_n(n)),
            },
        }
    }

    /// An explosive AR(1) (`ψ* = 1.2`) on a single-season hydro produces an
    /// `InvalidValue` error naming the hydro, season 0, and the offending
    /// autocorrelation.
    #[test]
    fn par_gate_rejects_explosive() {
        let ar_rows = vec![InflowArCoefficientRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            lag: 1,
            coefficient: 1.2,
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages_one_stage_per_season(1),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            ar_rows,
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let rejections: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue && e.message.contains("stationarity"))
            .collect();
        assert_eq!(
            rejections.len(),
            1,
            "expected exactly 1 stationarity rejection, got: {:?}",
            ctx.errors()
        );
        assert!(
            rejections[0].message.contains("Hydro 1") && rejections[0].message.contains("season 0"),
            "message should name the hydro and season, got: {}",
            rejections[0].message
        );
    }

    /// A stationary mixed-order (`orders = [3, 1, 2, 1]`) hydro -- the same
    /// fixture Epic 01's `t2_mixed_orders_gate_passes` verified stationary --
    /// produces no stationarity rejection.
    #[test]
    fn par_gate_accepts_stationary() {
        let psi: [Vec<f64>; 4] = [
            vec![
                0.398_915_659_532_620_8,
                0.062_802_463_704_355_48,
                -0.014_634_714_422_504_587,
            ],
            vec![0.35],
            vec![0.294_017_094_017_094, 0.017_094_017_094_017_092],
            vec![0.38],
        ];
        let mut ar_rows = Vec::new();
        for (season, coeffs) in psi.iter().enumerate() {
            for (i, &coefficient) in coeffs.iter().enumerate() {
                ar_rows.push(InflowArCoefficientRow {
                    hydro_id: EntityId::from(1),
                    stage_id: season as i32,
                    lag: (i + 1) as i32,
                    coefficient,
                });
            }
        }
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages_one_stage_per_season(4),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            ar_rows,
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let rejections: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.message.contains("stationarity"))
            .collect();
        assert!(
            rejections.is_empty(),
            "stationary mixed-order coefficients should produce no stationarity \
             rejection, got: {rejections:?}"
        );
    }

    /// Empty `inflow_ar_coefficients` (history-only / estimation path) skips
    /// the gate even when unrelated history data is present.
    #[test]
    fn par_gate_skips_when_no_coefficients() {
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages_one_stage_per_season(1),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![], // no AR coefficients -- estimation/history path
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let rejections: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.message.contains("stationarity"))
            .collect();
        assert!(
            rejections.is_empty(),
            "empty inflow_ar_coefficients should skip the gate entirely, \
             got: {rejections:?}"
        );
    }

    /// Order-bearing AR coefficients whose stage carries NO resolvable season
    /// context (no `season_map` AND no per-stage `season_id` -- genuinely
    /// unresolvable, the same condition under which
    /// `populate_derived_residual_ratios` itself errors) produce a
    /// `BusinessRuleViolation` naming the unresolved stage, instead of
    /// silently skipping the gate.
    #[test]
    fn par_gate_requires_season_map() {
        let ar_rows = vec![InflowArCoefficientRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            lag: 1,
            coefficient: 0.5,
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]), // stage.season_id: None, no season_map
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            ar_rows,
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let matching: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("stationarity")
                    && e.message.contains("season_definitions")
            })
            .collect();
        assert!(
            !matching.is_empty(),
            "genuinely unresolvable season context with order-bearing \
             coefficients should error, got: {:?}",
            ctx.errors()
        );
    }

    /// Bare per-stage `season_id`s with no `season_map` resolve via
    /// `resolve_stage_seasons`'s fallback (amended 2026-07-14): no
    /// missing-season-context error is added, and the stationarity check
    /// actually runs -- proven here by an explosive AR(1) still being
    /// rejected on this path, not merely "no error".
    #[test]
    fn par_gate_uses_season_id_fallback() {
        let mut stages = make_stages_5b(vec![0]);
        stages.stages[0].season_id = Some(0); // bare season_id, no season_map
        let ar_rows = vec![InflowArCoefficientRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            lag: 1,
            coefficient: 1.2, // explosive -- proves the check actually ran
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            ar_rows,
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let missing_context: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation && e.message.contains("stationarity")
            })
            .collect();
        assert!(
            missing_context.is_empty(),
            "a bare per-stage season_id should resolve via the fallback, \
             got: {missing_context:?}"
        );

        let rejections: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue && e.message.contains("stationarity"))
            .collect();
        assert_eq!(
            rejections.len(),
            1,
            "expected the fallback-resolved gate to run and reject the \
             explosive AR(1), got: {:?}",
            ctx.errors()
        );
    }

    /// A PAR-A hydro whose classical part is stationary but whose effective
    /// 12-lag system (widened by the annual term) is explosive -- the same
    /// parameterization as Epic 01's `par_a_explosive_effective_rejected` --
    /// produces a stationarity rejection (the annual gate fired, not merely
    /// a singular closure).
    #[test]
    fn par_gate_rejects_explosive_annual() {
        let ar_rows = vec![InflowArCoefficientRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            lag: 1,
            coefficient: 0.3,
        }];
        let stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 20.0,
        }];
        let mut data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages_one_stage_per_season(1),
            vec![make_bus_with_deficit(1, 10.0)],
            stats,
            ar_rows,
            None,
        );
        data.inflow_annual_components = vec![InflowAnnualComponentRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            annual_coefficient: 5.0,
            annual_mean_m3s: 100.0,
            annual_std_m3s: 1.0,
        }];

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let rejections: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue && e.message.contains("stationarity"))
            .collect();
        assert_eq!(
            rejections.len(),
            1,
            "expected exactly 1 stationarity rejection from the annual gate, got: {:?}",
            ctx.errors()
        );
        assert!(
            !rejections[0].message.contains("singular"),
            "the explosive effective annual system should be a typed rejection, \
             not a singular closure, got: {}",
            rejections[0].message
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
        let history: Vec<InflowHistoryRow> = (0..3)
            .map(|y| InflowHistoryRow {
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
    ) -> Hydro {
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
