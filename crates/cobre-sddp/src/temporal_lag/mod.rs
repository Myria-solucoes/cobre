//! Resolution of calendar-anchored lags into per-stage and per-block
//! factors, built on `cobre_core`'s [`window_period_overlaps`]
//! interval-overlap primitive. Two entry points share the overlap engine
//! (`docs/design/temporal-lag-unification.md` §2.1): [`resolve_spread`]
//! resolves a spreadable quantity's stage-clock and block-clock weights;
//! [`resolve_point`] resolves a point commitment (anticipated dispatch) into
//! a single decider per delivery stage.
//!
//! [`resolve_spread`] is the sole resolver dependency for the water
//! travel-time feature (`docs/design/water-travel-time-sddp-analysis.md`
//! §2.5, `docs/design/temporal-lag-unification.md` §3). Every factor it
//! returns — stage-clock `k`, block-resolved `chi`/`kappa`, and
//! per-arrival-stage `delivery` — overlaps `arrival_window`'s ONE shared
//! uniform arrival density against nested calendar partitions (stage clock
//! ⊃ block clock), so the aggregation-consistency identity
//! `Σ_b w_b·χ_{b,d} == k_d` holds by construction rather than by convention.

use cobre_core::window_period_overlaps;

/// Resolved spread of one arc's travel-time arrival density, anchored at a
/// single stage `t`.
#[derive(Debug, Clone)]
pub struct SpreadResolution {
    /// Deepest future stage the arrival window reaches — the max index
    /// reached, never a count (`window_period_overlaps`'s contiguity
    /// contract).
    pub depth: usize,
    /// Stage-clock weight `k_d` for `d = 0..=depth`; `k_0` is the same-stage
    /// share and `Σ_d k_d == 1`.
    pub k: Vec<f64>,
    /// Per-source-block deposit, `chi[b][d]`: index `0` is block `b`'s
    /// same-stage retained share, `1..=depth` are its bucket-`d` deposits.
    /// Empty when the anchor has no block partition.
    pub chi: Vec<Vec<f64>>,
    /// Per-source-block within-stage routing, `kappa[b][j]` to target block
    /// `b + j` (`j == 0` is block `b` routing to itself). Empty when the
    /// anchor has no block partition.
    pub kappa: Vec<Vec<f64>>,
    /// Per-arrival-stage delivery density for lag `d = 1..=depth` at index
    /// `d - 1`; an empty row where `k_d == 0` (nothing arrives that lag).
    pub delivery: Vec<Vec<f64>>,
}

/// The one shared arrival-density window: a uniform release over the anchor
/// stage `[0, h_anchor)` delayed by `travel_time_hours` arrives over
/// `[travel_time_hours, travel_time_hours + h_anchor)`. `k`, `chi`, `kappa`,
/// and `delivery` all read overlaps against this same window.
fn arrival_window(travel_time_hours: f64, h_anchor: f64) -> (f64, f64) {
    (travel_time_hours, travel_time_hours + h_anchor)
}

/// Resolve a scalar travel time into stage-clock weights and, for a
/// chronological anchor, block-resolved deposit/routing/delivery factors.
///
/// `stage_lengths_hours` is the full per-stage calendar; `anchor_stage`
/// selects stage `t`, and the arrival window is overlapped against
/// `stage_lengths_hours[anchor_stage..]`. `block_lengths_hours` is the
/// anchor's own block partition for a chronological anchor (`None` for a
/// parallel anchor) and must sum to `stage_lengths_hours[anchor_stage]`; v1
/// reuses it as every reached arrival stage's own partition too (a future
/// per-arrival-stage partition is a config-only extension, `chi`/`kappa`/`k`
/// are unaffected either way).
///
/// # Panics
///
/// Debug builds panic if `anchor_stage` is out of bounds, if
/// `travel_time_hours` is not finite and positive, or if the
/// conservation / aggregation-consistency identities do not hold.
#[must_use]
pub fn resolve_spread(
    travel_time_hours: f64,
    anchor_stage: usize,
    stage_lengths_hours: &[f64],
    block_lengths_hours: Option<&[f64]>,
) -> SpreadResolution {
    debug_assert!(
        travel_time_hours.is_finite() && travel_time_hours > 0.0,
        "travel_time_hours must be finite and > 0.0"
    );
    debug_assert!(
        anchor_stage < stage_lengths_hours.len(),
        "anchor_stage must index into stage_lengths_hours"
    );

    let future_calendar = &stage_lengths_hours[anchor_stage..];
    let h_anchor = future_calendar[0];
    let (window_start, window_end) = arrival_window(travel_time_hours, h_anchor);

    let k_overlaps = window_period_overlaps(window_start, h_anchor, future_calendar);
    let k: Vec<f64> = k_overlaps
        .iter()
        .map(|&overlap| overlap / h_anchor)
        .collect();
    let depth = k.len().saturating_sub(1);

    debug_assert!(
        (k.iter().sum::<f64>() - 1.0).abs() < 1e-9,
        "stage-clock weights must sum to 1.0 (conservation)"
    );

    let blocks = block_lengths_hours.filter(|blocks| !blocks.is_empty());
    if let Some(blocks) = blocks {
        debug_assert!(
            (blocks.iter().sum::<f64>() - h_anchor).abs() < 1e-9,
            "block_lengths_hours must sum to the anchor stage length"
        );
    }

    let (chi, kappa) = blocks.map_or_else(
        || (Vec::new(), Vec::new()),
        |blocks| resolve_block_factors(travel_time_hours, blocks, future_calendar, depth),
    );

    if let Some(blocks) = blocks {
        for (d, &k_d) in k.iter().enumerate() {
            let aggregated: f64 = chi
                .iter()
                .zip(blocks)
                .map(|(row, &duration_b)| (duration_b / h_anchor) * row[d])
                .sum();
            debug_assert!(
                (aggregated - k_d).abs() < 1e-9,
                "block deposits must aggregate to the stage-level k_d (sub-contract 2)"
            );
        }
    }

    let delivery = resolve_delivery(window_start, window_end, future_calendar, blocks, depth);

    SpreadResolution {
        depth,
        k,
        chi,
        kappa,
        delivery,
    }
}

/// Per-source-block `chi`/`kappa`: reads the same arrival window as `k`,
/// restricted to each block's own local origin, so the aggregation
/// identity in [`resolve_spread`] holds structurally rather than needing a
/// separate density.
fn resolve_block_factors(
    travel_time_hours: f64,
    blocks: &[f64],
    future_calendar: &[f64],
    depth: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n_blocks = blocks.len();
    let mut chi = vec![vec![0.0_f64; depth + 1]; n_blocks];
    let mut kappa = Vec::with_capacity(n_blocks);

    for (b, &duration_b) in blocks.iter().enumerate() {
        let mut target = Vec::with_capacity(n_blocks - b + future_calendar.len() - 1);
        target.extend_from_slice(&blocks[b..]);
        target.extend_from_slice(&future_calendar[1..]);

        let combined = window_period_overlaps(travel_time_hours, duration_b, &target);
        let kappa_len = (n_blocks - b).min(combined.len());

        let mut kappa_b: Vec<f64> = combined[..kappa_len]
            .iter()
            .map(|&overlap| overlap / duration_b)
            .collect();
        kappa_b.resize(n_blocks - b, 0.0);

        chi[b][0] = kappa_b.iter().sum();
        for (offset, &overlap) in combined[kappa_len..].iter().enumerate() {
            let d = offset + 1;
            if d <= depth {
                chi[b][d] = overlap / duration_b;
            }
        }

        kappa.push(kappa_b);
    }

    (chi, kappa)
}

/// Per-arrival-stage delivery density for lag `d = 1..=depth`: the same
/// `[window_start, window_end)` window restricted to stage `t+d`'s own
/// local clock and split across `blocks` (`None` delivers onto a single
/// parallel row).
fn resolve_delivery(
    window_start: f64,
    window_end: f64,
    future_calendar: &[f64],
    blocks: Option<&[f64]>,
    depth: usize,
) -> Vec<Vec<f64>> {
    let mut delivery = Vec::with_capacity(depth);
    let mut stage_start = 0.0_f64;

    for (d, &stage_len) in future_calendar[..=depth].iter().enumerate() {
        let stage_end = stage_start + stage_len;
        if d >= 1 {
            let overlap_start = window_start.max(stage_start);
            let overlap_end = window_end.min(stage_end);
            let width = (overlap_end - overlap_start).max(0.0);

            let row = if width > 0.0 {
                let local_start = overlap_start - stage_start;
                blocks.map_or_else(
                    || vec![1.0],
                    |blocks| {
                        window_period_overlaps(local_start, width, blocks)
                            .iter()
                            .map(|&overlap| overlap / width)
                            .collect()
                    },
                )
            } else {
                Vec::new()
            };
            delivery.push(row);
        }
        stage_start = stage_end;
    }

    delivery
}

/// A calendar-anchored point-commitment lag: a physical lead time on the
/// hour clock, or a first-class stage-count shift that never reads the
/// calendar (`docs/design/temporal-lag-unification.md` §4.4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lag {
    /// Physical lead time in hours, delivery-anchored (§4.3).
    Time(f64),
    /// Stage-count shift; the calendar is never consulted (§4.4).
    Stages(u32),
}

/// Resolved point-commitment lag: the delivery-anchored decider, the
/// per-decision-stage outgoing commitment sets, and the per-decision-stage
/// depths (`docs/design/temporal-lag-unification.md` §4.3).
#[derive(Debug, Clone)]
pub struct PointResolution {
    /// Decision stage `c(m)` for each delivery stage `m`; `None` is a
    /// pre-study (initial-conditions) decider.
    pub decider: Vec<Option<usize>>,
    /// `C(t) = { m : c(m) = t }`, indexed by decision stage `t`.
    pub decision_sets: Vec<Vec<usize>>,
    /// `K(t) = |{ m > t : c(m) <= t }|`, indexed by decision stage `t`.
    pub depth: Vec<usize>,
}

/// Resolve a point-commitment lag into the delivery-anchored decider, the
/// per-decision-stage outgoing commitment sets, and the per-decision-stage
/// depths (§4.3 physical mode, §4.4 stage-count mode).
///
/// `stage_lengths_hours` must have length `n_stages`; [`Lag::Stages`] never
/// reads it.
///
/// # Panics
///
/// Debug builds panic if `stage_lengths_hours.len() != n_stages` in
/// [`Lag::Time`] mode, if a stage length or the lead time is not finite and
/// positive, or if a non-IC delivery stage fails to appear in its own
/// decision set.
#[must_use]
pub fn resolve_point(lag: Lag, stage_lengths_hours: &[f64], n_stages: usize) -> PointResolution {
    let decider = match lag {
        Lag::Time(delta_hours) => {
            resolve_decider_physical(delta_hours, stage_lengths_hours, n_stages)
        }
        Lag::Stages(lead_stages) => resolve_decider_stage_count(lead_stages, n_stages),
    };
    let (decision_sets, depth) = build_decision_sets_and_depth(&decider, n_stages);

    PointResolution {
        decider,
        decision_sets,
        depth,
    }
}

/// Cumulative stage-end boundaries `S_0 = 0, S_1, .., S_n`, the hour-clock
/// primitive shared with [`resolve_spread`].
fn cumulative_stage_boundaries(stage_lengths_hours: &[f64]) -> Vec<f64> {
    let mut boundaries = Vec::with_capacity(stage_lengths_hours.len() + 1);
    let mut cumulative = 0.0_f64;
    boundaries.push(cumulative);
    for &length in stage_lengths_hours {
        debug_assert!(
            length.is_finite() && length > 0.0,
            "every stage length must be finite and > 0.0"
        );
        cumulative += length;
        boundaries.push(cumulative);
    }
    boundaries
}

/// `c(m)` = the stage containing `end_m − Δ`, anchored at the delivery
/// stage's end with boundary ties resolving to the earlier stage — a
/// sub-stage lead (`Δ < h_m`) gives `c(m) = m`; a start-anchored `start_m −
/// Δ` could never reach that (§4.3). `None` when the target precedes the
/// horizon start.
fn resolve_decider_physical(
    delta_hours: f64,
    stage_lengths_hours: &[f64],
    n_stages: usize,
) -> Vec<Option<usize>> {
    debug_assert!(
        delta_hours.is_finite() && delta_hours > 0.0,
        "delta_hours must be finite and > 0.0"
    );
    debug_assert_eq!(
        stage_lengths_hours.len(),
        n_stages,
        "stage_lengths_hours must cover every delivery stage in physical mode"
    );

    let boundaries = cumulative_stage_boundaries(stage_lengths_hours);

    (0..n_stages)
        .map(|m| {
            let target = boundaries[m + 1] - delta_hours;
            let before_target = boundaries.partition_point(|&boundary| boundary < target);
            before_target.checked_sub(1)
        })
        .collect()
}

/// `c(m) = m − ℓ` (or `None` if negative); the calendar is never read (§4.4)
/// — enforced by construction, since this arm takes no stage-length
/// parameter.
fn resolve_decider_stage_count(lead_stages: u32, n_stages: usize) -> Vec<Option<usize>> {
    let lead = usize::try_from(lead_stages).unwrap_or(usize::MAX);
    (0..n_stages).map(|m| m.checked_sub(lead)).collect()
}

/// Builds `C(t)` and `K(t)` from `decider` in one forward pass: each
/// delivery `m` with decider `d` both joins `decision_sets[d]` and widens a
/// sweep interval `[d, m)` (`d <= m` always, since `c(m)` never exceeds
/// `m`), so a difference-array prefix sum yields `K(t) = |{m>t : c(m)<=t}|`
/// without an O(n²) rescan.
fn build_decision_sets_and_depth(
    decider: &[Option<usize>],
    n_stages: usize,
) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut decision_sets: Vec<Vec<usize>> = vec![Vec::new(); n_stages];
    let mut depth_delta = vec![0_isize; n_stages];

    for (m, &decided_at) in decider.iter().enumerate() {
        if let Some(t) = decided_at {
            decision_sets[t].push(m);
            depth_delta[t] += 1;
            depth_delta[m] -= 1;
        }
    }

    let mut running = 0_isize;
    let depth: Vec<usize> = depth_delta
        .iter()
        .map(|&delta| {
            running += delta;
            debug_assert!(
                running >= 0,
                "depth sweep invariant violated: every decider must satisfy c(m) <= m"
            );
            usize::try_from(running).unwrap_or(0)
        })
        .collect();

    for (m, &decided_at) in decider.iter().enumerate() {
        debug_assert!(
            decided_at.is_none_or(|t| decision_sets[t].contains(&m)),
            "every non-IC delivery stage must appear in exactly one decision_set"
        );
    }

    (decision_sets, depth)
}

#[cfg(test)]
mod tests;
