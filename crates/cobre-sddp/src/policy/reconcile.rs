//! Boundary-policy coefficient reconciliation: per-target-slot rebind
//! operations that map a source cut's coefficients onto a target manifest of
//! possibly different shape than `source`.
//!
//! [`build_rebind`] resolves one [`RebindOp`] per target slot, dispatched by
//! `entity_type`; [`rebind_cut`] applies the resolved ops to one source cut's
//! coefficients. Storage and inflow-lag are the state's must-correspond core:
//! a target slot of either family with no source counterpart is
//! [`RebindOp::Reject`] — the entity (`entity_type`/`entity_id`) is never
//! relaxed, only matched exactly. Transit buckets resolve by the same
//! identity match; a miss defaults to [`RebindOp::Zero`] instead of
//! rejecting — a NEWAVE-shaped source carries no transit slots at all, so a
//! miss is the expected case, not a boundary the current study is
//! incompatible with. Anticipated slots dispatch on `delivery_date` and the
//! caller-supplied `target_delivery_intervals`: `SENTINEL` (pre-fan-out
//! padding) always defaults to `Zero`; a live, dated post-study-targeted ring
//! slot (dated WITH a resolved interval) fans out against the source's own anticipated
//! months by calendar overlap — full coverage by priced source months yields
//! [`RebindOp::Blend`] (the `÷H_M` distribute), a boundary-edge slot straddling
//! into unpriced time yields [`RebindOp::Renormalize`] (anti-deflation over the
//! covered span), no covered month yields `Zero`; a live, dated slot with NO
//! resolved interval is an in-study ring slot (a within-horizon delivery the
//! terminal boundary does not price) and also yields `Zero`. Every remaining
//! family falls back to the identity-reject default, pending its own arm. When every
//! storage/lag/transit/unclassified target slot has a same-identity source
//! counterpart and every anticipated slot is still sentinel-dated — the shape
//! an already target-aligned, pre-fan-out boundary policy has — the rebind
//! reproduces the source cut's own coefficients bit-for-bit: `Copy` at the
//! matching position for every identity-resolved family, and `Zero` for the
//! sentinel-anticipated slot, whose source coefficient there is itself always
//! `0.0` (a masked state dimension never holds a value). This is the
//! strict-superset guarantee.

use std::collections::HashMap;

use chrono::Months;
use chrono::NaiveDate;
use cobre_core::AnticipatedCommitmentHistory;
use cobre_io::ENTITY_SLOT_DELIVERY_DATE_SENTINEL;
use cobre_io::EntitySlot;
use cobre_io::OwnedPolicyCutRecord;
use cobre_io::StateFamily;
use serde::Serialize;

use crate::SddpError;

/// Tolerance (hours) for treating a target slot's covered-month overlap as
/// exactly `H_w` — the `Blend`-vs-`Renormalize` dividing line. Calendar-day
/// arithmetic converted to hours is exact in `f64` for any realistic
/// horizon, so this only absorbs a multi-term summation's rounding, never
/// masks a genuine gap.
const COVERAGE_TOLERANCE_HOURS: f64 = 1e-6;

/// One reconciliation operation, producing one target-manifest slot's
/// coefficient from a source cut's coefficients.
///
/// [`build_rebind`] assigns exactly one op per target slot; [`rebind_cut`]
/// applies the assignment to a source cut.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RebindOp {
    /// Take the source coefficient at this position verbatim.
    Copy(usize),
    /// No source counterpart contributes; the coefficient is `0.0`.
    Zero,
    /// An hours-weighted blend of source positions: [`rebind_cut`] applies
    /// `Σ cut.coefficients[p] · w` over the `(source_position, weight)`
    /// terms. [`build_rebind`] constructs this for a live, dated
    /// `AnticipatedThermalState` target slot fully covered by priced source
    /// months, `weight = overlap(w, M) / H_M` per covered month `M` — the
    /// `÷H_M` distribute. A monthly target fully inside one source month
    /// yields a single `1.0` term (copy-equivalent).
    Blend(Vec<(usize, f64)>),
    /// A [`Self::Blend`] re-normalized over its covered overlap span:
    /// [`rebind_cut`] applies the identical weighted sum, but each term's
    /// weight is additionally scaled by `H_w / Σ_covered overlap(w, M)` so
    /// the covered months' density replicates across the target slot's
    /// uncovered span instead of deflating it with an implicit `0.0` term.
    /// [`build_rebind`] constructs this for a target slot whose interval
    /// straddles a priced month and an unpriced one.
    Renormalize(Vec<(usize, f64)>),
    /// The target slot cannot be resolved from `source`: either no
    /// same-identity counterpart exists under a family that requires one
    /// (storage, inflow-lag, the identity fallback — the entity is never
    /// relaxed), or a dated `AnticipatedThermalState` slot has no resolved
    /// delivery interval (an invariant violation, never expected on real
    /// input). A sentinel: [`build_rebind`] converts this into an
    /// [`SddpError::Validation`] rather than returning it, so it never
    /// appears in a successfully built op vector.
    Reject {
        /// Human-readable rejection reason.
        reason: String,
    },
}

/// `(entity_type, entity_id, subindex)` — the identity a reconciliation join
/// keys on.
type SlotKey = (u8, i32, u32);

fn slot_key(slot: &EntitySlot) -> SlotKey {
    (slot.entity_type, slot.entity_id, slot.subindex)
}

/// One anticipated source slot's decoded calendar month: `source_pos` is its
/// position in `source`; `[month_start, month_end)` and `h_m` come from
/// [`decode_month_anchor`].
struct MonthSource {
    source_pos: usize,
    month_start: NaiveDate,
    month_end: NaiveDate,
    h_m: f64,
}

/// `(entity_type, entity_id)` — the anticipated fan-out join key. A source
/// anticipated slot's `entity_type` is always
/// [`StateFamily::AnticipatedThermalState`], so the first component never varies;
/// kept for symmetry with [`SlotKey`],
/// the identity families' join key.
type MonthKey = (u8, i32);

/// Positive `[start, end)` span in hours; `0.0` for a degenerate or
/// backwards interval (`end <= start`), never a negative value.
fn positive_hours(start: NaiveDate, end: NaiveDate) -> f64 {
    if end <= start {
        return 0.0;
    }
    let days = u32::try_from((end - start).num_days()).unwrap_or(0);
    f64::from(days) * 24.0
}

/// Hours of overlap between two `[start, end)` calendar intervals; `0.0` when
/// they do not intersect.
pub(crate) fn overlap_hours(a: (NaiveDate, NaiveDate), b: (NaiveDate, NaiveDate)) -> f64 {
    positive_hours(a.0.max(b.0), a.1.min(b.1))
}

/// Decode a day-01 `YYYYMM01` anchor into `[month_start, month_end)` and
/// `H_M = days_in_month · 24` hours — the exact inverse of
/// `year_month_day_anchor` (`setup/mod.rs`). Shared with
/// [`crate::policy::policy_load::resolve_boundary_source_stage`], which
/// decodes the same per-pool anticipated anchors to auto-resolve
/// `policy.boundary.source_stage`.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if `delivery_date` does not decode to a
/// real calendar month.
pub(crate) fn decode_month_anchor(
    delivery_date: i32,
) -> Result<(NaiveDate, NaiveDate, f64), SddpError> {
    let year = delivery_date / 10_000;
    let month = u32::try_from(delivery_date / 100 % 100).map_err(|_| {
        SddpError::Validation(format!(
            "boundary policy source anticipated slot has a malformed delivery_date anchor \
             {delivery_date}"
        ))
    })?;
    let month_start = NaiveDate::from_ymd_opt(year, month, 1).ok_or_else(|| {
        SddpError::Validation(format!(
            "boundary policy source anticipated slot has an invalid delivery_date anchor \
             {delivery_date}"
        ))
    })?;
    let month_end = month_start
        .checked_add_months(Months::new(1))
        .ok_or_else(|| {
            SddpError::Validation(format!(
                "boundary policy source anticipated slot's delivery_date anchor {delivery_date} \
             overflows the calendar"
            ))
        })?;
    let h_m = positive_hours(month_start, month_end);
    Ok((month_start, month_end, h_m))
}

/// Index every LIVE (non-sentinel) anticipated source slot by its owning
/// [`MonthKey`], decoding each slot's day-01 `delivery_date` anchor into the
/// calendar month [`resolve_anticipated`] computes `overlap(w, M)` against.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if a live anticipated source slot's
/// `delivery_date` fails to decode ([`decode_month_anchor`]).
fn build_by_month_index(
    source: &[EntitySlot],
) -> Result<HashMap<MonthKey, Vec<MonthSource>>, SddpError> {
    let mut by_month: HashMap<MonthKey, Vec<MonthSource>> = HashMap::new();
    for (pos, slot) in source.iter().enumerate() {
        if slot.entity_type != StateFamily::AnticipatedThermalState.code()
            || slot.delivery_date == ENTITY_SLOT_DELIVERY_DATE_SENTINEL
        {
            continue;
        }
        let (month_start, month_end, h_m) = decode_month_anchor(slot.delivery_date)?;
        by_month
            .entry((slot.entity_type, slot.entity_id))
            .or_default()
            .push(MonthSource {
                source_pos: pos,
                month_start,
                month_end,
                h_m,
            });
    }
    Ok(by_month)
}

/// The boundary intercept-fold vector `(source_pos, factor)` pricing a study's
/// fixed post-horizon commitments, reusing [`resolve_anticipated`]'s per-month
/// `÷H_M` distribute (`overlap_hours(w, M) / H_M`) applied to each fixed
/// window's declared MW instead of a state dimension. A window overlapping no
/// source month contributes nothing (mirrors [`RebindOp::Zero`], never an
/// error), a `value_mw == 0.0` window is skipped, and only non-zero factors are
/// emitted — an empty vector is the byte-neutral no-contribution case the
/// intercept fold relies on, so zero-factor entries must never be emitted.
///
/// Determinism (D5): accumulates into a `source_pos`-indexed `Vec`, iterating
/// `fixed_windows` and each plant's [`build_by_month_index`] records in given
/// order; the sole map use is the `by_month` lookup, never a `HashMap`
/// iteration.
///
/// # Errors
///
/// Propagates [`SddpError::Validation`] from [`build_by_month_index`] when a
/// live anticipated `source` slot's `delivery_date` fails to decode.
pub(crate) fn build_boundary_fold(
    source: &[EntitySlot],
    fixed_windows: &[AnticipatedCommitmentHistory],
) -> Result<Vec<(usize, f64)>, SddpError> {
    let by_month = build_by_month_index(source)?;
    let mut factor = vec![0.0_f64; source.len()];
    for window in fixed_windows {
        if window.value_mw == 0.0 {
            continue;
        }
        let key = (
            StateFamily::AnticipatedThermalState.code(),
            window.thermal_id.0,
        );
        let Some(months) = by_month.get(&key) else {
            continue;
        };
        for m in months {
            let overlap = overlap_hours(
                (window.start_date, window.end_date),
                (m.month_start, m.month_end),
            );
            if overlap > 0.0 {
                factor[m.source_pos] += (overlap / m.h_m) * window.value_mw;
            }
        }
    }
    Ok(factor
        .into_iter()
        .enumerate()
        .filter(|&(_, f)| f != 0.0)
        .collect())
}

/// Build one [`RebindOp`] per `target` slot, dispatched per target slot's
/// `entity_type` to [`resolve_storage`], [`resolve_inflow_lag`],
/// [`resolve_transit_bucket`], [`resolve_anticipated`], or the identity
/// fallback [`resolve_by_identity`]. `target_delivery_intervals` is aligned
/// 1:1 with `target` (`Some((start, end))` for a live, dated
/// `AnticipatedThermalState` target slot, `None` elsewhere) — the calendar
/// span [`resolve_anticipated`] resolves `overlap(w, M)` against `source`'s
/// own anticipated months, reconstructed from their `delivery_date` day-01
/// anchors ([`build_by_month_index`]; no new source input).
///
/// For a `target` whose every storage/lag/transit/unclassified slot has a
/// same-identity `source` counterpart, and whose every anticipated slot is
/// still sentinel-dated (pre-fan-out) — the shape a still target-aligned
/// boundary policy has — the rebind reproduces the `source` cut's own
/// coefficients bit-for-bit: `Copy` at the matching position for every
/// identity-resolved family, and `Zero` for the sentinel-anticipated slot
/// (see the module doc for why that still matches `source`).
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if a storage, inflow-lag, or
/// unclassified `target` slot has no `source` counterpart under identity
/// resolution, or a live anticipated `source` slot's `delivery_date` fails to
/// decode to a real calendar month. A dated `AnticipatedThermalState` `target`
/// slot with no resolved `target_delivery_intervals` entry is an in-study ring
/// slot, resolved to [`RebindOp::Zero`] rather than rejected (see
/// [`resolve_anticipated`]).
pub(crate) fn build_rebind(
    source: &[EntitySlot],
    target: &[EntitySlot],
    target_delivery_intervals: &[Option<(NaiveDate, NaiveDate)>],
) -> Result<Vec<RebindOp>, SddpError> {
    debug_assert_eq!(
        target_delivery_intervals.len(),
        target.len(),
        "target_delivery_intervals must be aligned 1:1 with target"
    );

    let mut by_identity: HashMap<SlotKey, usize> = HashMap::with_capacity(source.len());
    for (pos, slot) in source.iter().enumerate() {
        by_identity.insert(slot_key(slot), pos);
    }
    let by_month = build_by_month_index(source)?;

    let mut ops = Vec::with_capacity(target.len());
    for (i, slot) in target.iter().enumerate() {
        let interval = target_delivery_intervals.get(i).copied().flatten();
        match resolve_target_slot(i, slot, &by_identity, &by_month, interval) {
            RebindOp::Reject { reason } => return Err(SddpError::Validation(reason)),
            op => ops.push(op),
        }
    }
    Ok(ops)
}

/// Dispatch one target slot to its family's resolution rule, by
/// `entity_type`. Storage, inflow-lag, and transit-bucket are each indexed by
/// their own identity shape (storage ignores `subindex`, always `0`; lag and
/// transit-bucket include it); all three share the one `by_identity` map,
/// since it already keys on the full `(entity_type, entity_id, subindex)`
/// triple. Anticipated slots never consult `by_identity` — dispatch is on
/// `delivery_date` and `target_interval`, joined against `by_month`.
fn resolve_target_slot(
    i: usize,
    slot: &EntitySlot,
    by_identity: &HashMap<SlotKey, usize>,
    by_month: &HashMap<MonthKey, Vec<MonthSource>>,
    target_interval: Option<(NaiveDate, NaiveDate)>,
) -> RebindOp {
    match slot.family() {
        Some(StateFamily::HydroStorage) => resolve_storage(slot, by_identity),
        Some(StateFamily::HydroInflowLag) => resolve_inflow_lag(slot, by_identity),
        Some(StateFamily::HydroTransitBucket) => resolve_transit_bucket(slot, by_identity),
        Some(StateFamily::AnticipatedThermalState) => {
            resolve_anticipated(slot, target_interval, by_month)
        }
        None => resolve_by_identity(i, slot, by_identity),
    }
}

/// `HydroStorage` resolution: matched by `(entity_type, entity_id)` identity;
/// unmatched rejects naming the hydro the boundary policy does not price.
fn resolve_storage(slot: &EntitySlot, by_identity: &HashMap<SlotKey, usize>) -> RebindOp {
    match by_identity.get(&slot_key(slot)) {
        Some(&pos) => RebindOp::Copy(pos),
        None => RebindOp::Reject {
            reason: format!(
                "boundary policy does not price hydro {}; it was trained on a different set of \
                 plants",
                slot.entity_id
            ),
        },
    }
}

/// `HydroInflowLag` resolution: matched by `(entity_type, entity_id,
/// subindex)` identity; unmatched rejects as a lag-depth incompatibility,
/// naming the offending hydro and lag depth.
fn resolve_inflow_lag(slot: &EntitySlot, by_identity: &HashMap<SlotKey, usize>) -> RebindOp {
    match by_identity.get(&slot_key(slot)) {
        Some(&pos) => RebindOp::Copy(pos),
        None => RebindOp::Reject {
            reason: format!(
                "boundary policy has no inflow-lag coefficient for hydro {} at lag depth {}: the \
                 boundary is lag-depth-incompatible with the current study",
                slot.entity_id, slot.subindex
            ),
        },
    }
}

/// `HydroTransitBucket` resolution: matched by `(entity_type, entity_id,
/// subindex)` identity, mirroring inflow-lag's shape; unmatched defaults to
/// `Zero` rather than rejecting — a NEWAVE-shaped source carries no transit
/// slots at all, so a miss is the expected case, not a boundary the current
/// study is incompatible with (unlike storage/lag's must-correspond core). A
/// Cobre-to-Cobre boundary with a matching transit arc hits instead and
/// copies the source's own coefficient.
fn resolve_transit_bucket(slot: &EntitySlot, by_identity: &HashMap<SlotKey, usize>) -> RebindOp {
    match by_identity.get(&slot_key(slot)) {
        Some(&pos) => RebindOp::Copy(pos),
        None => RebindOp::Zero,
    }
}

/// `AnticipatedThermalState` resolution. `slot.delivery_date == SENTINEL`
/// (pre-fan-out padding) always resolves to `Zero`, regardless of
/// `target_interval`. A live, dated slot with no resolved `target_interval`
/// is an IN-STUDY ring slot — a commitment delivered WITHIN the current
/// horizon (e.g. a matured commitment fished at the terminal stage, or a
/// `K = 0` sub-stage-lead delivery self-delivered there) — and resolves to
/// `Zero`: the terminal boundary FCF prices only post-study obligations, so
/// a within-horizon delivery, already discharged inside the study, contributes
/// nothing. A post-study-targeted ring slot derives its `delivery_date` and its
/// `target_interval` from the SAME modular delivery stage
/// ([`build_stage_entity_delivery_intervals`] mirrors
/// [`build_stage_entity_manifest`], each dating the slot at its modular delivery
/// stage), so a post-study-targeted ring slot is always
/// `dated ⟺ Some(interval)`; a `None` interval on a dated slot therefore marks
/// the in-study ring uniquely, never a failed post-study resolution. Two
/// wrong-but-compiling alternatives: [`RebindOp::Reject`] here aborts a
/// legitimate boundary load the moment any anticipated thermal delivers
/// in-horizon (a sub-stage lead at the terminal stage — the K=0 case); fanning
/// an in-study slot out (resolving it an in-horizon interval) would wrongly
/// [`RebindOp::Blend`] a within-horizon delivery against the source's months.
/// A live, dated slot WITH a resolved interval fans out against `by_month`'s
/// calendar-overlap-weighted source months: no covered month yields `Zero`;
/// full coverage yields [`RebindOp::Blend`] (`weight = overlap(w, M) / H_M`,
/// the `÷H_M` distribute); partial coverage (a boundary-edge slot straddling
/// into unpriced time) yields [`RebindOp::Renormalize`], scaling the covered
/// months' density up to the full slot instead of an implicit `0.0`
/// deflation term.
///
/// [`build_stage_entity_delivery_intervals`]: crate::policy_export::build_stage_entity_delivery_intervals
/// [`build_stage_entity_manifest`]: crate::policy_export::build_stage_entity_manifest
fn resolve_anticipated(
    slot: &EntitySlot,
    target_interval: Option<(NaiveDate, NaiveDate)>,
    by_month: &HashMap<MonthKey, Vec<MonthSource>>,
) -> RebindOp {
    if slot.delivery_date == ENTITY_SLOT_DELIVERY_DATE_SENTINEL {
        return RebindOp::Zero;
    }
    let Some((start_w, end_w)) = target_interval else {
        return RebindOp::Zero;
    };

    let h_w = positive_hours(start_w, end_w);
    if h_w <= 0.0 {
        return RebindOp::Zero;
    }
    let Some(months) = by_month.get(&(slot.entity_type, slot.entity_id)) else {
        return RebindOp::Zero;
    };

    let mut terms = Vec::new();
    let mut covered = 0.0;
    for m in months {
        let overlap = overlap_hours((start_w, end_w), (m.month_start, m.month_end));
        if overlap > 0.0 {
            terms.push((m.source_pos, overlap, m.h_m));
            covered += overlap;
        }
    }
    if terms.is_empty() {
        return RebindOp::Zero;
    }

    if (h_w - covered).abs() <= COVERAGE_TOLERANCE_HOURS {
        RebindOp::Blend(
            terms
                .into_iter()
                .map(|(pos, overlap, h_m)| (pos, overlap / h_m))
                .collect(),
        )
    } else {
        let scale = h_w / covered;
        RebindOp::Renormalize(
            terms
                .into_iter()
                .map(|(pos, overlap, h_m)| (pos, (overlap / h_m) * scale))
                .collect(),
        )
    }
}

/// Generic identity resolution: the default for any family without its own
/// arm. Unmatched rejects naming the target's own position and identity.
fn resolve_by_identity(
    i: usize,
    slot: &EntitySlot,
    by_identity: &HashMap<SlotKey, usize>,
) -> RebindOp {
    match by_identity.get(&slot_key(slot)) {
        Some(&pos) => RebindOp::Copy(pos),
        None => RebindOp::Reject {
            reason: format!(
                "target slot {i} (entity_type={}, entity_id={}, subindex={}) has no source \
                 counterpart",
                slot.entity_type, slot.entity_id, slot.subindex
            ),
        },
    }
}

/// Produce `rebind`'s target-aligned coefficient vector from one source cut.
///
/// `Blend` and `Renormalize` both apply their precomputed `(source_position,
/// weight)` terms identically — `Σ cut.coefficients[p] · w` — the
/// distribute-vs-renormalize distinction lives entirely in how
/// [`build_rebind`] computed the weights, never in this application.
///
/// # Panics
///
/// Panics if `rebind` contains a [`RebindOp::Reject`] — a [`build_rebind`]
/// postcondition violation: `build_rebind` converts every `Reject` into an
/// error before returning.
pub(crate) fn rebind_cut(cut: &OwnedPolicyCutRecord, rebind: &[RebindOp]) -> Vec<f64> {
    rebind
        .iter()
        .map(|op| match op {
            RebindOp::Copy(pos) => cut.coefficients[*pos],
            RebindOp::Zero => 0.0,
            RebindOp::Blend(terms) | RebindOp::Renormalize(terms) => terms
                .iter()
                .map(|&(pos, w)| cut.coefficients[pos] * w)
                .sum(),
            RebindOp::Reject { reason } => unreachable!(
                "build_rebind must convert Reject into an error before rebind_cut sees it: \
                 {reason}"
            ),
        })
        .collect()
}

/// `source` positions no [`RebindOp::Copy`] or `Blend`/`Renormalize` term in
/// `rebind` ever references.
pub(crate) fn dropped_source_positions(source_len: usize, rebind: &[RebindOp]) -> Vec<usize> {
    let mut referenced = vec![false; source_len];
    for op in rebind {
        match op {
            RebindOp::Copy(pos) => {
                if *pos < source_len {
                    referenced[*pos] = true;
                }
            }
            RebindOp::Blend(terms) | RebindOp::Renormalize(terms) => {
                for &(pos, _) in terms {
                    if pos < source_len {
                        referenced[pos] = true;
                    }
                }
            }
            RebindOp::Zero | RebindOp::Reject { .. } => {}
        }
    }
    referenced
        .into_iter()
        .enumerate()
        .filter_map(|(pos, was_referenced)| (!was_referenced).then_some(pos))
        .collect()
}

/// Per-family tally of one [`BoundaryReconciliationReport`]'s per-operation
/// classification: `copy` ([`RebindOp::Copy`]), `fan_out`
/// (`Blend` + `Renormalize` target slots), `straddling` (the `Renormalize`
/// subset of `fan_out` — the boundary-edge sub-case), `default_zero`
/// (target-only `Zero`, excluding a sentinel-anticipated pad), and
/// `dropped_source` (this family's own [`dropped_source_positions`]).
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct FamilyTally {
    /// Target slots resolved by [`RebindOp::Copy`].
    pub copy: usize,
    /// Target slots resolved by [`RebindOp::Blend`] or [`RebindOp::Renormalize`].
    pub fan_out: usize,
    /// The [`RebindOp::Renormalize`] subset of `fan_out`.
    pub straddling: usize,
    /// Target-only [`RebindOp::Zero`] slots (excludes a sentinel-anticipated pad).
    pub default_zero: usize,
    /// This family's source positions no op references.
    pub dropped_source: usize,
}

/// Anticipated-family fan-out coverage: the source's own priced
/// delivery-month span and the target's live delivery-interval span.
/// Paired with the sibling [`BoundaryReconciliationReport::anticipated`]
/// tally's `fan_out`/`straddling`/`default_zero`, this is everything
/// [`BoundaryReconciliationReport::detail_lines`] needs to render the
/// coverage line.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct AnticipatedCoverage {
    /// Live, dated anticipated source slots contributing a decoded calendar month.
    pub source_month_count: usize,
    /// `[earliest month_start, latest month_end)` across those source months.
    pub source_span: Option<(NaiveDate, NaiveDate)>,
    /// `[earliest start, latest end)` across live, dated anticipated target intervals.
    pub target_span: Option<(NaiveDate, NaiveDate)>,
}

/// The "which boundary policy we have + what got reconciled" diagnostic:
/// [`build_reconciliation_report`]'s pure tally of one
/// `load_boundary_cuts` reconciliation, by family. `reconciled` is `false`
/// only on the empty-manifest / dimension-only skip path, where every tally
/// stays at its zero [`Default`].
#[derive(Debug, Clone, Default, Serialize)]
pub struct BoundaryReconciliationReport {
    /// `false` on the skipped (dimension-only) load path; `true` when rebind ran.
    pub reconciled: bool,
    /// `HydroStorage` tally.
    pub storage: FamilyTally,
    /// `HydroInflowLag` tally.
    pub inflow_lag: FamilyTally,
    /// `HydroTransitBucket` tally.
    pub transit_bucket: FamilyTally,
    /// `AnticipatedThermalState` tally.
    pub anticipated: FamilyTally,
    /// The anticipated family's fan-out coverage summary.
    pub anticipated_coverage: AnticipatedCoverage,
    /// Every family without its own reconcile arm (the identity-reject default).
    pub other_identity: FamilyTally,
}

impl BoundaryReconciliationReport {
    fn tally_mut(&mut self, family: Option<StateFamily>) -> &mut FamilyTally {
        match family {
            Some(StateFamily::HydroStorage) => &mut self.storage,
            Some(StateFamily::HydroInflowLag) => &mut self.inflow_lag,
            Some(StateFamily::HydroTransitBucket) => &mut self.transit_bucket,
            Some(StateFamily::AnticipatedThermalState) => &mut self.anticipated,
            None => &mut self.other_identity,
        }
    }

    fn families(&self) -> [(&'static str, FamilyTally); 5] {
        [
            ("storage", self.storage),
            ("inflow-lag", self.inflow_lag),
            ("transit-bucket", self.transit_bucket),
            ("anticipated", self.anticipated),
            ("other-identity", self.other_identity),
        ]
    }

    /// The four aggregate tallies (copy, fan-out, default-zero, dropped-source)
    /// summed across every family — shared by [`Self::tally_clause`]'s wording
    /// and the CLI's own compact rendering of the same totals.
    #[must_use]
    pub fn tally_totals(&self) -> (usize, usize, usize, usize) {
        let families = self.families();
        let copy: usize = families.iter().map(|(_, t)| t.copy).sum();
        let fan_out: usize = families.iter().map(|(_, t)| t.fan_out).sum();
        let default_zero: usize = families.iter().map(|(_, t)| t.default_zero).sum();
        let dropped: usize = families.iter().map(|(_, t)| t.dropped_source).sum();
        (copy, fan_out, default_zero, dropped)
    }

    /// The four-total tally clause, with no leading "boundary reconciliation: "
    /// prefix — [`Self::summary_line`]'s payload. Reconciled totals only; the
    /// dimension-only notice is [`Self::summary_line`]'s own early return.
    #[must_use]
    pub fn tally_clause(&self) -> String {
        let (total_copy, total_fan_out, total_default_zero, total_dropped) = self.tally_totals();
        format!(
            "{total_copy} copied, {total_fan_out} fanned out, {total_default_zero} defaulted \
             to 0.0, {total_dropped} source slots dropped"
        )
    }

    /// One-line reconciliation summary: the dimension-only notice on the skip
    /// path, otherwise the totals across every family. The per-family breakdown
    /// lives in [`Self::detail_lines`].
    #[must_use]
    pub fn summary_line(&self) -> String {
        if !self.reconciled {
            return "boundary reconciliation: dimension-only load (entity manifest absent); no \
                    per-family fan-out tally"
                .to_string();
        }
        format!("boundary reconciliation: {}", self.tally_clause())
    }

    /// Per-family reconciliation breakdown (one COPY / FAN-OUT / DEFAULT-0.0 /
    /// DROP line per family, then the anticipated coverage line) — the verbose
    /// detail behind [`Self::summary_line`]. Empty on the dimension-only skip path.
    #[must_use]
    pub fn detail_lines(&self) -> Vec<String> {
        if !self.reconciled {
            return Vec::new();
        }
        let families = self.families();
        let mut lines = Vec::with_capacity(families.len() + 1);
        for (name, tally) in families {
            lines.push(format!(
                "{name}: COPY={}, FAN-OUT=({}, rule = distribute), DEFAULT-0.0={}, DROP={}",
                tally.copy, tally.fan_out, tally.default_zero, tally.dropped_source
            ));
        }
        lines.push(format!(
            "anticipated: {} source months fanned to {} target slots ({} straddling, \
             overlap-blended), {} months defaulted",
            self.anticipated_coverage.source_month_count,
            self.anticipated.fan_out,
            self.anticipated.straddling,
            self.anticipated.default_zero
        ));
        lines
    }
}

/// Classify one target slot's `(family, op)` into `tally`: `Copy` → COPY;
/// `Blend`/`Renormalize` → FAN-OUT (`Renormalize` also STRADDLING); `Zero` on
/// a sentinel-anticipated slot is a structural pad, excluded from every
/// tally; every other `Zero` → DEFAULT-0.0.
fn classify_op(
    op: &RebindOp,
    family: Option<StateFamily>,
    slot: &EntitySlot,
    tally: &mut FamilyTally,
) {
    match op {
        RebindOp::Copy(_) => tally.copy += 1,
        RebindOp::Blend(_) => tally.fan_out += 1,
        RebindOp::Renormalize(_) => {
            tally.fan_out += 1;
            tally.straddling += 1;
        }
        RebindOp::Zero => {
            let sentinel_anticipated_pad = family == Some(StateFamily::AnticipatedThermalState)
                && slot.delivery_date == ENTITY_SLOT_DELIVERY_DATE_SENTINEL;
            if !sentinel_anticipated_pad {
                tally.default_zero += 1;
            }
        }
        RebindOp::Reject { reason } => unreachable!(
            "build_rebind must convert Reject into an error before build_reconciliation_report \
             sees it: {reason}"
        ),
    }
}

/// Widen `span` to also cover `interval`, when present; a no-op for `None`
/// (a sentinel or otherwise interval-less slot).
fn fold_span(span: &mut Option<(NaiveDate, NaiveDate)>, interval: Option<(NaiveDate, NaiveDate)>) {
    let Some((start, end)) = interval else {
        return;
    };
    *span = Some(match *span {
        Some((cur_start, cur_end)) => (cur_start.min(start), cur_end.max(end)),
        None => (start, end),
    });
}

/// Build a [`BoundaryReconciliationReport`] over one `load_boundary_cuts`
/// reconciliation: a pure pass over the aligned `target`/`rebind` vectors,
/// `source` (for [`dropped_source_positions`] and the source month span), and
/// `target_delivery_intervals` (for the target interval span). No I/O, no
/// mutation, not on any hot path.
///
/// # Panics
///
/// Panics if `rebind` contains a [`RebindOp::Reject`] — the same
/// `build_rebind` postcondition violation [`rebind_cut`] guards against.
pub(crate) fn build_reconciliation_report(
    source: &[EntitySlot],
    target: &[EntitySlot],
    target_delivery_intervals: &[Option<(NaiveDate, NaiveDate)>],
    rebind: &[RebindOp],
) -> BoundaryReconciliationReport {
    let mut report = BoundaryReconciliationReport {
        reconciled: true,
        ..BoundaryReconciliationReport::default()
    };

    let mut target_span = None;
    for (i, (slot, op)) in target.iter().zip(rebind).enumerate() {
        let family = slot.family();
        let interval = target_delivery_intervals.get(i).copied().flatten();
        fold_span(&mut target_span, interval);
        classify_op(op, family, slot, report.tally_mut(family));
    }
    report.anticipated_coverage.target_span = target_span;

    for pos in dropped_source_positions(source.len(), rebind) {
        if let Some(slot) = source.get(pos) {
            report.tally_mut(slot.family()).dropped_source += 1;
        }
    }

    let mut source_span = None;
    let mut source_month_count = 0;
    for slot in source {
        if slot.entity_type != StateFamily::AnticipatedThermalState.code()
            || slot.delivery_date == ENTITY_SLOT_DELIVERY_DATE_SENTINEL
        {
            continue;
        }
        let Ok((start, end, _)) = decode_month_anchor(slot.delivery_date) else {
            continue;
        };
        source_month_count += 1;
        fold_span(&mut source_span, Some((start, end)));
    }
    report.anticipated_coverage.source_month_count = source_month_count;
    report.anticipated_coverage.source_span = source_span;

    report
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;

    use super::{
        BoundaryReconciliationReport, FamilyTally, RebindOp, StateFamily, build_boundary_fold,
        build_rebind, build_reconciliation_report, decode_month_anchor, dropped_source_positions,
        rebind_cut,
    };
    use crate::SddpError;
    use cobre_core::{AnticipatedCommitmentHistory, EntityId};
    use cobre_io::{ENTITY_SLOT_DELIVERY_DATE_SENTINEL, EntitySlot, OwnedPolicyCutRecord};

    fn storage_slot(id: i32) -> EntitySlot {
        EntitySlot {
            entity_type: 0,
            entity_id: id,
            subindex: 0,
            was_active: true,
            delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        }
    }

    fn inflow_lag_slot(id: i32, lag_depth: u32) -> EntitySlot {
        EntitySlot {
            entity_type: StateFamily::HydroInflowLag.code(),
            entity_id: id,
            subindex: lag_depth,
            was_active: true,
            delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        }
    }

    fn transit_bucket_slot(downstream_hydro_id: i32, lag: u32) -> EntitySlot {
        EntitySlot {
            entity_type: StateFamily::HydroTransitBucket.code(),
            entity_id: downstream_hydro_id,
            subindex: lag,
            was_active: true,
            delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        }
    }

    fn anticipated_sentinel_slot(thermal_id: i32, ring_slot: u32) -> EntitySlot {
        EntitySlot {
            entity_type: StateFamily::AnticipatedThermalState.code(),
            entity_id: thermal_id,
            subindex: ring_slot,
            was_active: true,
            delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        }
    }

    fn anticipated_dated_slot(thermal_id: i32, ring_slot: u32, delivery_date: i32) -> EntitySlot {
        EntitySlot {
            entity_type: StateFamily::AnticipatedThermalState.code(),
            entity_id: thermal_id,
            subindex: ring_slot,
            was_active: true,
            delivery_date,
        }
    }

    fn owned_cut(coefficients: Vec<f64>) -> OwnedPolicyCutRecord {
        OwnedPolicyCutRecord {
            cut_id: 1,
            slot_index: 0,
            iteration: 0,
            forward_pass_index: 0,
            intercept: 0.0,
            coefficients,
            is_active: true,
        }
    }

    /// All-`None` intervals aligned to `target.len()` — for tests exercising
    /// only the identity families, which never consult
    /// `target_delivery_intervals`.
    fn no_intervals(target: &[EntitySlot]) -> Vec<Option<(NaiveDate, NaiveDate)>> {
        vec![None; target.len()]
    }

    fn ymd(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).expect("valid calendar date")
    }

    fn fixed_window(
        thermal_id: i32,
        start: NaiveDate,
        end: NaiveDate,
        value_mw: f64,
    ) -> AnticipatedCommitmentHistory {
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(thermal_id),
            start_date: start,
            end_date: end,
            value_mw,
        }
    }

    /// Given a `source` and `target` manifest of equal shape (all storage),
    /// when `build_rebind` runs, then it returns one `Copy` per slot at the
    /// matching source position.
    #[test]
    fn build_rebind_equal_shape_all_storage_yields_identity_copy() {
        let source = vec![storage_slot(1), storage_slot(2), storage_slot(3)];
        let target = source.clone();

        let rebind = build_rebind(&source, &target, &no_intervals(&target)).unwrap();

        assert_eq!(
            rebind,
            vec![RebindOp::Copy(0), RebindOp::Copy(1), RebindOp::Copy(2)]
        );
    }

    /// Given a target manifest mixing storage and inflow-lag slots that both
    /// match the source by identity, when `build_rebind` runs, then every slot
    /// resolves to `Copy` at its identity-matched source position — order
    /// independent, since matching is by identity, not position.
    #[test]
    fn build_rebind_storage_and_lag_identity_match_yields_copy() {
        let source = vec![
            storage_slot(1),
            inflow_lag_slot(1, 1),
            storage_slot(2),
            inflow_lag_slot(2, 1),
        ];
        let target = vec![storage_slot(2), inflow_lag_slot(1, 1)];

        let rebind = build_rebind(&source, &target, &no_intervals(&target)).unwrap();

        assert_eq!(rebind, vec![RebindOp::Copy(2), RebindOp::Copy(1)]);
    }

    /// Given a target storage slot for a hydro absent from the source, when
    /// `build_rebind` runs, then it rejects, naming the unpriced hydro.
    #[test]
    fn build_rebind_storage_miss_rejects_naming_hydro() {
        let source = vec![storage_slot(1)];
        let target = vec![storage_slot(1), storage_slot(42)];

        let err = build_rebind(&source, &target, &no_intervals(&target)).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("42"), "must name the unpriced hydro: {msg}");
    }

    /// Given a target inflow-lag slot whose `(entity_id, subindex)` is absent
    /// from the source, when `build_rebind` runs, then it rejects as a
    /// lag-depth incompatibility, naming the offending hydro and lag depth.
    #[test]
    fn build_rebind_lag_miss_rejects_naming_lag_depth() {
        let source = vec![storage_slot(1), inflow_lag_slot(1, 1)];
        let target = vec![storage_slot(1), inflow_lag_slot(1, 2)];

        let err = build_rebind(&source, &target, &no_intervals(&target)).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains('1'), "must name hydro 1: {msg}");
        assert!(msg.contains('2'), "must name lag depth 2: {msg}");
    }

    /// A target storage slot never matches a source slot sharing its
    /// `entity_id` under a different family (inflow-lag): the entity is
    /// identified by `(entity_type, entity_id[, subindex])` jointly, never
    /// `entity_id` alone — the entity is never relaxed across families.
    #[test]
    fn build_rebind_entity_never_crosses_family() {
        let source = vec![inflow_lag_slot(5, 1)];
        let target = vec![storage_slot(5)];

        let err = build_rebind(&source, &target, &no_intervals(&target)).unwrap_err();

        assert!(matches!(err, SddpError::Validation(_)));
    }

    /// Given a rebind of all `Copy` ops and a source cut, when `rebind_cut`
    /// runs, then the returned coefficient vector equals the source cut's
    /// coefficients verbatim.
    #[test]
    fn rebind_cut_all_copy_matches_source_coefficients_verbatim() {
        let source = vec![storage_slot(1), storage_slot(2)];
        let target = source.clone();
        let rebind = build_rebind(&source, &target, &no_intervals(&target)).unwrap();
        let cut = owned_cut(vec![10.0, -5.0]);

        let coefficients = rebind_cut(&cut, &rebind);

        assert_eq!(coefficients, cut.coefficients);
    }

    /// Given a rebind containing a `Zero` op at position `j`, when
    /// `rebind_cut` runs, then `coefficients[j] == 0.0`.
    #[test]
    fn rebind_cut_zero_op_yields_zero_coefficient() {
        let rebind = vec![RebindOp::Copy(0), RebindOp::Zero];
        let cut = owned_cut(vec![7.0]);

        let coefficients = rebind_cut(&cut, &rebind);

        assert_eq!(coefficients, vec![7.0, 0.0]);
    }

    /// Given a `Blend` op with two weighted source positions, when
    /// `rebind_cut` runs, then it returns their weighted sum — the mechanical
    /// application `build_rebind`'s weight computation feeds.
    #[test]
    fn rebind_cut_blend_op_sums_weighted_source_positions() {
        let rebind = vec![RebindOp::Blend(vec![(0, 0.25), (1, 0.75)])];
        let cut = owned_cut(vec![100.0, 200.0]);

        let coefficients = rebind_cut(&cut, &rebind);

        assert_eq!(coefficients, vec![175.0]);
    }

    /// Given a `Renormalize` op, when `rebind_cut` runs, then it applies the
    /// identical weighted-sum mechanics as `Blend` — the distinction is only
    /// in the weight value, never in how `rebind_cut` uses it.
    #[test]
    fn rebind_cut_renormalize_op_sums_weighted_source_positions() {
        let rebind = vec![RebindOp::Renormalize(vec![(0, 0.5)])];
        let cut = owned_cut(vec![40.0]);

        let coefficients = rebind_cut(&cut, &rebind);

        assert_eq!(coefficients, vec![20.0]);
    }

    #[test]
    fn build_rebind_rejects_target_slot_with_no_source_counterpart() {
        let source = vec![storage_slot(1)];
        let target = vec![storage_slot(1), storage_slot(2)];

        let err = build_rebind(&source, &target, &no_intervals(&target)).unwrap_err();

        assert!(matches!(err, SddpError::Validation(_)));
    }

    #[test]
    fn dropped_source_positions_reports_unreferenced_source_slots() {
        let rebind = vec![RebindOp::Copy(1)];

        let dropped = dropped_source_positions(3, &rebind);

        assert_eq!(dropped, vec![0, 2]);
    }

    /// Given a target transit-bucket slot, when `build_rebind` runs, then it
    /// resolves to `Copy` at the source's identity-matched position when the
    /// source carries the same `(entity_id, subindex)` transit slot, and to
    /// `Zero` when it does not (the NEWAVE-shaped-source default) — transit
    /// resolves by identity like storage/lag, never unconditionally.
    #[test]
    fn build_rebind_transit_bucket_copies_on_identity_match_else_zero() {
        let source = vec![storage_slot(1), transit_bucket_slot(2, 1)];
        let matching_target = vec![storage_slot(1), transit_bucket_slot(2, 1)];

        let rebind =
            build_rebind(&source, &matching_target, &no_intervals(&matching_target)).unwrap();

        assert_eq!(rebind, vec![RebindOp::Copy(0), RebindOp::Copy(1)]);
        let cut = owned_cut(vec![10.0, 99.0]);
        assert_eq!(rebind_cut(&cut, &rebind), vec![10.0, 99.0]);

        let missing_target = vec![storage_slot(1), transit_bucket_slot(3, 1)];

        let rebind =
            build_rebind(&source, &missing_target, &no_intervals(&missing_target)).unwrap();

        assert_eq!(rebind, vec![RebindOp::Copy(0), RebindOp::Zero]);
        assert_eq!(rebind_cut(&cut, &rebind), vec![10.0, 0.0]);
    }

    /// Given a target anticipated slot with `delivery_date == SENTINEL`, when
    /// `build_rebind` runs, then it resolves to `Zero` — even when the source
    /// carries an identical sentinel-dated slot at the same identity, proving
    /// the sentinel case is dispatched purely on `delivery_date`.
    #[test]
    fn build_rebind_sentinel_anticipated_target_is_always_zero() {
        let source = vec![storage_slot(1), anticipated_sentinel_slot(9, 0)];
        let target = vec![storage_slot(1), anticipated_sentinel_slot(9, 0)];

        let rebind = build_rebind(&source, &target, &no_intervals(&target)).unwrap();

        assert_eq!(rebind, vec![RebindOp::Copy(0), RebindOp::Zero]);
    }

    /// A `YYYYMM01` source anchor decodes to the correct `[month_start,
    /// month_end)` interval and `H_M = days_in_month · 24` hours.
    #[test]
    fn decode_month_anchor_yields_correct_interval_and_hours() {
        let (start, end, h_m) = decode_month_anchor(20_260_301).unwrap();

        assert_eq!(start, ymd(2026, 3, 1));
        assert_eq!(end, ymd(2026, 4, 1));
        assert_eq!(h_m, 31.0 * 24.0, "March 2026 has 31 days");
    }

    /// Given a source manifest with one monthly anticipated slot and a
    /// target ring slot whose interval lies fully inside that month, when
    /// `build_rebind` runs, then it resolves to `Blend` with a single term
    /// weighted `overlap/H_M`.
    #[test]
    fn build_rebind_dated_anticipated_target_fully_covered_yields_blend() {
        let source = vec![anticipated_dated_slot(9, 0, 20_260_301)];
        let target = vec![anticipated_dated_slot(9, 100, 20_260_301)];
        let target_interval = (ymd(2026, 3, 1), ymd(2026, 4, 1));
        let intervals = vec![Some(target_interval)];

        let rebind = build_rebind(&source, &target, &intervals).unwrap();

        assert_eq!(
            rebind,
            vec![RebindOp::Blend(vec![(0, 1.0)])],
            "a monthly target fully inside one source month yields a single unit-weight term"
        );

        let cut = owned_cut(vec![42.5]);
        let coefficients = rebind_cut(&cut, &rebind);
        assert_eq!(
            coefficients[0].to_bits(),
            cut.coefficients[0].to_bits(),
            "a single unit-weight Blend term must reproduce the source coefficient \
             bit-for-bit, copy-equivalent"
        );
    }

    /// Given a target ring slot spanning one week fully inside a priced
    /// source month, when `build_rebind` runs, then it resolves to `Blend`
    /// with a fractional `overlap/H_M` weight.
    #[test]
    fn build_rebind_anticipated_partial_month_yields_blend_fractional_weight() {
        let source = vec![storage_slot(1), anticipated_dated_slot(9, 0, 20_260_401)];
        let target = vec![storage_slot(1), anticipated_dated_slot(9, 100, 20_260_401)];
        let target_interval = (ymd(2026, 4, 8), ymd(2026, 4, 15));
        let intervals = vec![None, Some(target_interval)];

        let rebind = build_rebind(&source, &target, &intervals).unwrap();

        match &rebind[1] {
            RebindOp::Blend(terms) => {
                assert_eq!(terms.len(), 1);
                assert_eq!(terms[0].0, 1);
                let expected_weight = (7.0 * 24.0) / (30.0 * 24.0);
                assert!(
                    (terms[0].1 - expected_weight).abs() < expected_weight * 1e-9,
                    "weight {} != expected {expected_weight}",
                    terms[0].1
                );
            }
            other => panic!("expected Blend, got {other:?}"),
        }
    }

    /// Given a target ring slot whose interval straddles a priced month and
    /// an unpriced one, when `build_rebind` runs, then it resolves to
    /// `Renormalize` with a single covered term scaled to the full slot —
    /// never an implicit `0.0` deflation term for the uncovered days.
    #[test]
    fn build_rebind_anticipated_straddle_into_unpriced_yields_renormalize_no_zero_term() {
        let source_coeff = 300.0;
        let source = vec![anticipated_dated_slot(9, 0, 20_260_301)];
        let target = vec![anticipated_dated_slot(9, 100, 20_260_301)];
        let start_w = ymd(2026, 2, 26);
        let end_w = ymd(2026, 3, 5);
        let intervals = vec![Some((start_w, end_w))];

        let rebind = build_rebind(&source, &target, &intervals).unwrap();

        let h_w = f64::from(u32::try_from((end_w - start_w).num_days()).unwrap()) * 24.0;
        let h_m = 31.0 * 24.0;
        match &rebind[0] {
            RebindOp::Renormalize(terms) => {
                assert_eq!(
                    terms.len(),
                    1,
                    "no 0.0 deflation term for the uncovered (unpriced February) days"
                );
                let (pos, weight) = terms[0];
                assert_eq!(pos, 0);
                let expected_weight = h_w / h_m;
                assert!(
                    (weight - expected_weight).abs() < expected_weight * 1e-9,
                    "weight {weight} != expected {expected_weight}"
                );
            }
            other => panic!("expected Renormalize, got {other:?}"),
        }

        let cut = owned_cut(vec![source_coeff]);
        let coefficients = rebind_cut(&cut, &rebind);
        let expected_coeff = source_coeff * h_w / h_m;
        assert!(
            (coefficients[0] - expected_coeff).abs() < expected_coeff.abs() * 1e-9,
            "renormalized coefficient {} != expected {expected_coeff} (source · H_w/H_priced)",
            coefficients[0]
        );
    }

    /// Given a target ring slot whose interval falls in a month the source
    /// carries no anticipated slot for, when `build_rebind` runs, then it
    /// resolves to `Zero` — no covered month, nothing to reconcile to.
    #[test]
    fn build_rebind_anticipated_no_covered_month_yields_zero() {
        let source = vec![anticipated_dated_slot(9, 0, 20_260_301)];
        let target = vec![anticipated_dated_slot(9, 100, 20_260_301)];
        let intervals = vec![Some((ymd(2026, 5, 1), ymd(2026, 5, 8)))];

        let rebind = build_rebind(&source, &target, &intervals).unwrap();

        assert_eq!(rebind, vec![RebindOp::Zero]);
    }

    /// Given a live, dated anticipated target slot whose
    /// `target_delivery_intervals` entry is `None` — the shape of an IN-STUDY
    /// ring slot (a within-horizon delivery, e.g. a `K = 0` sub-stage-lead
    /// thermal maturing at the terminal stage), never a post-study-targeted ring
    /// slot (those are always dated ⟺ interval) — when `build_rebind` runs, then it
    /// resolves to `Zero`: the terminal boundary prices no within-horizon
    /// delivery. It must never reject (which would abort a legitimate load) and
    /// never fan out against the source months (which would wrongly `Blend`).
    #[test]
    fn build_rebind_dated_in_study_ring_slot_with_no_interval_yields_zero() {
        // A source month for thermal 9 that WOULD overlap the in-study slot's
        // April date if it were (wrongly) fanned out — proving the `Zero` is
        // the in-study classification, not an accidental no-covered-month miss.
        let source = vec![storage_slot(1), anticipated_dated_slot(9, 0, 20_260_401)];
        let target = vec![storage_slot(1), anticipated_dated_slot(9, 0, 20_260_401)];
        let intervals = vec![None, None];

        let rebind = build_rebind(&source, &target, &intervals).unwrap();

        assert_eq!(rebind, vec![RebindOp::Copy(0), RebindOp::Zero]);
    }

    /// Given a source already shaped identically to the target (storage, lag,
    /// and a sentinel-dated anticipated slot), when `build_rebind` and
    /// `rebind_cut` run over the source cut, then every resulting coefficient
    /// is bit-identical to the source's own coefficient (`f64::to_bits`) — the
    /// superset property: reconciling a target-shaped source never regresses
    /// today's exact-match load. The sentinel-anticipated position stays
    /// `0.0` on both sides — a masked state dimension never holds a value —
    /// so the forced `Zero` output equals `source`'s own coefficient there.
    #[test]
    fn superset_target_shaped_source_reconciles_bit_identically() {
        let source = vec![
            storage_slot(1),
            inflow_lag_slot(1, 1),
            anticipated_sentinel_slot(9, 0),
        ];
        let target = source.clone();
        let rebind = build_rebind(&source, &target, &no_intervals(&target)).unwrap();
        let cut = owned_cut(vec![10.5, -3.25, 0.0]);

        let coefficients = rebind_cut(&cut, &rebind);

        assert_eq!(coefficients.len(), cut.coefficients.len());
        for (actual, expected) in coefficients.iter().zip(cut.coefficients.iter()) {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "reconciling a target-shaped source must reproduce its coefficients \
                 bit-for-bit: {actual} != {expected}"
            );
        }
    }

    /// Given a source carrying a transit-bucket slot and an anticipated slot
    /// that no target rebind entry references (the target has neither
    /// family), when `dropped_source_positions` runs, then both positions are
    /// reported dropped.
    #[test]
    fn dropped_source_positions_reports_unreferenced_transit_and_anticipated_source_slots() {
        let source = vec![
            storage_slot(1),
            transit_bucket_slot(2, 1),
            anticipated_sentinel_slot(9, 0),
        ];
        let target = vec![storage_slot(1)];
        let rebind = build_rebind(&source, &target, &no_intervals(&target)).unwrap();

        let dropped = dropped_source_positions(source.len(), &rebind);

        assert_eq!(dropped, vec![1, 2]);
    }

    /// A source position referenced only by a `Blend` or `Renormalize` term
    /// (never a `Copy`) is not reported dropped — `dropped_source_positions`
    /// counts every referencing op, not `Copy` alone; an unreferenced pad
    /// still is.
    #[test]
    fn dropped_source_positions_blend_and_renormalize_referenced_slots_are_not_dropped() {
        let rebind = vec![
            RebindOp::Blend(vec![(0, 0.5), (1, 0.5)]),
            RebindOp::Renormalize(vec![(2, 1.0)]),
        ];

        let dropped = dropped_source_positions(4, &rebind);

        assert_eq!(
            dropped,
            vec![3],
            "positions 0/1 (Blend terms) and 2 (Renormalize term) are referenced; only 3 is \
             dropped"
        );
    }

    /// Given a source with one monthly anticipated slot and a target with two
    /// ring slots exactly tiling that month, when `build_reconciliation_report`
    /// runs over the resulting `Blend` ops, then the anticipated family's
    /// `fan_out` equals the target slot count, `straddling`/`default_zero` are
    /// `0`, and the coverage line renders the expected shape.
    #[test]
    fn build_reconciliation_report_full_coverage_fan_out_matches_target_slot_count() {
        let source = vec![anticipated_dated_slot(9, 0, 20_260_401)];
        let target = vec![
            anticipated_dated_slot(9, 100, 20_260_401),
            anticipated_dated_slot(9, 101, 20_260_401),
        ];
        let intervals = vec![
            Some((ymd(2026, 4, 1), ymd(2026, 4, 16))),
            Some((ymd(2026, 4, 16), ymd(2026, 5, 1))),
        ];
        let rebind = build_rebind(&source, &target, &intervals).unwrap();

        let report = build_reconciliation_report(&source, &target, &intervals, &rebind);

        assert_eq!(
            report.anticipated.fan_out, 2,
            "N target slots -> fan_out == N"
        );
        assert_eq!(
            report.anticipated.straddling, 0,
            "full coverage: no straddle"
        );
        assert_eq!(report.anticipated.default_zero, 0);
        assert_eq!(
            report.anticipated_coverage.source_month_count, 1,
            "one source month (K = 1)"
        );

        let lines = report.detail_lines();
        assert!(
            lines.iter().any(|l| l
                == "anticipated: 1 source months fanned to 2 target slots (0 straddling, \
                    overlap-blended), 0 months defaulted"),
            "coverage line must match the expected shape: {lines:?}"
        );
    }

    /// Given a target ring slot straddling a priced month and an unpriced one
    /// (a `Renormalize` op), when `build_reconciliation_report` runs, then the
    /// anticipated family's `straddling` includes that slot and it is also
    /// counted in `fan_out`.
    #[test]
    fn build_reconciliation_report_renormalize_counts_in_fan_out_and_straddling() {
        let source = vec![anticipated_dated_slot(9, 0, 20_260_301)];
        let target = vec![anticipated_dated_slot(9, 100, 20_260_301)];
        let intervals = vec![Some((ymd(2026, 2, 26), ymd(2026, 3, 5)))];
        let rebind = build_rebind(&source, &target, &intervals).unwrap();
        assert!(
            matches!(rebind[0], RebindOp::Renormalize(_)),
            "fixture must straddle into unpriced time"
        );

        let report = build_reconciliation_report(&source, &target, &intervals, &rebind);

        assert_eq!(
            report.anticipated.fan_out, 1,
            "Renormalize counts toward fan_out"
        );
        assert_eq!(report.anticipated.straddling, 1, "and toward straddling");
    }

    /// Given a target dated anticipated slot with no covered source month
    /// (`Zero`) and a sentinel anticipated slot (also `Zero`), when
    /// `build_reconciliation_report` runs, then the dated slot counts as
    /// `default_zero` while the sentinel slot is excluded — not a default.
    #[test]
    fn build_reconciliation_report_dated_zero_defaults_sentinel_zero_excluded() {
        let source = vec![anticipated_dated_slot(9, 0, 20_260_301)];
        let target = vec![
            anticipated_dated_slot(9, 100, 20_260_301),
            anticipated_sentinel_slot(9, 101),
        ];
        let intervals = vec![Some((ymd(2026, 5, 1), ymd(2026, 5, 8))), None];
        let rebind = build_rebind(&source, &target, &intervals).unwrap();
        assert_eq!(rebind, vec![RebindOp::Zero, RebindOp::Zero]);

        let report = build_reconciliation_report(&source, &target, &intervals, &rebind);

        assert_eq!(
            report.anticipated.default_zero, 1,
            "only the dated no-covered-month Zero counts as a default"
        );
        assert_eq!(report.anticipated.fan_out, 0);
        assert_eq!(report.anticipated.copy, 0);
    }

    /// Given a target-shaped source (storage, lag, sentinel anticipated) that
    /// loads bit-identically, when `build_reconciliation_report` runs, then
    /// every family reports only `copy` (the sentinel-anticipated slot is
    /// excluded per its own classification, not counted as `copy` or
    /// `default_zero`), and `fan_out == 0` everywhere.
    #[test]
    fn build_reconciliation_report_target_shaped_superset_reports_copy_only() {
        let source = vec![
            storage_slot(1),
            inflow_lag_slot(1, 1),
            anticipated_sentinel_slot(9, 0),
        ];
        let target = source.clone();
        let intervals = no_intervals(&target);
        let rebind = build_rebind(&source, &target, &intervals).unwrap();

        let report = build_reconciliation_report(&source, &target, &intervals, &rebind);

        assert_eq!(report.storage.copy, 1);
        assert_eq!(report.inflow_lag.copy, 1);
        assert_eq!(
            report.anticipated.copy, 0,
            "sentinel Zero is excluded, not copy"
        );
        assert_eq!(
            report.anticipated.default_zero, 0,
            "sentinel Zero is excluded, not a default"
        );
        assert_eq!(report.anticipated.fan_out, 0);
        assert_eq!(report.transit_bucket.fan_out, 0);
        assert_eq!(report.other_identity.fan_out, 0);
    }

    /// The default report (the shape `load_boundary_cuts` stores on its
    /// empty-manifest / dimension-only skip path) is `reconciled == false`,
    /// renders a dimension-only summary line, and carries no per-family detail.
    #[test]
    fn boundary_reconciliation_report_default_is_unreconciled_with_dimension_only_render() {
        let report = BoundaryReconciliationReport::default();

        assert!(!report.reconciled);
        assert!(
            report.summary_line().contains("dimension-only"),
            "must state a dimension-only load: {}",
            report.summary_line()
        );
        assert!(
            report.detail_lines().is_empty(),
            "the skip path carries no per-family detail"
        );
    }

    /// `summary_line()`'s reconciled-path wording is a correctness contract for
    /// `cobre validate` (asserted byte-exact by `cli_validate.rs`) — the
    /// `tally_clause()` extraction must not change a single byte of it, and
    /// `tally_clause()` itself is exactly that wording minus the leading
    /// "boundary reconciliation: " prefix.
    #[test]
    fn summary_line_reconciled_wording_is_byte_identical_after_tally_clause_extraction() {
        let report = BoundaryReconciliationReport {
            reconciled: true,
            storage: FamilyTally {
                copy: 2184,
                ..FamilyTally::default()
            },
            inflow_lag: FamilyTally {
                fan_out: 1,
                ..FamilyTally::default()
            },
            transit_bucket: FamilyTally {
                dropped_source: 2,
                ..FamilyTally::default()
            },
            anticipated: FamilyTally {
                default_zero: 1,
                ..FamilyTally::default()
            },
            ..BoundaryReconciliationReport::default()
        };

        assert_eq!(
            report.summary_line(),
            "boundary reconciliation: 2184 copied, 1 fanned out, 1 defaulted to 0.0, 2 source \
             slots dropped"
        );
        assert_eq!(
            report.tally_clause(),
            "2184 copied, 1 fanned out, 1 defaulted to 0.0, 2 source slots dropped"
        );
        assert_eq!(
            report.summary_line(),
            format!("boundary reconciliation: {}", report.tally_clause())
        );
    }

    /// A single fixed window one whole week fully inside a priced source month
    /// (April, `H_M = 30·24`) folds to the hand-computed `(overlap/H_M)·value`
    /// at the source month's own position.
    #[test]
    fn build_boundary_fold_single_window_yields_hand_computed_factor() {
        let source = vec![storage_slot(1), anticipated_dated_slot(9, 0, 20_260_401)];
        let k = 1;
        let value = 50.0;
        let windows = vec![fixed_window(9, ymd(2026, 4, 8), ymd(2026, 4, 15), value)];

        let fold = build_boundary_fold(&source, &windows).unwrap();

        assert_eq!(fold.len(), 1);
        assert_eq!(fold[0].0, k);
        let expected = (7.0 * 24.0 / (30.0 * 24.0)) * value;
        assert!(
            (fold[0].1 - expected).abs() < expected * 1e-9,
            "factor {} != expected {expected}",
            fold[0].1
        );
    }

    /// A fixed window in a month the source carries no anticipated slot for
    /// contributes nothing — mirrors `RebindOp::Zero`, an empty fold.
    #[test]
    fn build_boundary_fold_no_overlapping_source_month_is_empty() {
        let source = vec![anticipated_dated_slot(9, 0, 20_260_401)];
        let windows = vec![fixed_window(9, ymd(2026, 5, 1), ymd(2026, 5, 8), 50.0)];

        let fold = build_boundary_fold(&source, &windows).unwrap();

        assert_eq!(fold, vec![]);
    }

    /// A `value_mw == 0.0` window overlapping a source month contributes
    /// nothing, keeping the fold empty for the all-zero horizon-end stub.
    #[test]
    fn build_boundary_fold_zero_value_window_is_empty() {
        let source = vec![anticipated_dated_slot(9, 0, 20_260_401)];
        let windows = vec![fixed_window(9, ymd(2026, 4, 8), ymd(2026, 4, 15), 0.0)];

        let fold = build_boundary_fold(&source, &windows).unwrap();

        assert_eq!(fold, vec![]);
    }

    /// Two windows overlapping the one source month accumulate into a single
    /// emitted term at that source position — a per-position accumulation.
    #[test]
    fn build_boundary_fold_accumulates_windows_at_one_source_position() {
        let source = vec![storage_slot(1), anticipated_dated_slot(9, 0, 20_260_401)];
        let k = 1;
        let value_a = 30.0;
        let value_b = 60.0;
        let windows = vec![
            fixed_window(9, ymd(2026, 4, 1), ymd(2026, 4, 8), value_a),
            fixed_window(9, ymd(2026, 4, 20), ymd(2026, 4, 27), value_b),
        ];

        let fold = build_boundary_fold(&source, &windows).unwrap();

        assert_eq!(fold.len(), 1, "one source position -> one emitted term");
        assert_eq!(fold[0].0, k);
        let w = 7.0 * 24.0 / (30.0 * 24.0);
        let expected = w * value_a + w * value_b;
        assert!(
            (fold[0].1 - expected).abs() < expected * 1e-9,
            "factor {} != expected sum {expected}",
            fold[0].1
        );
    }

    /// Reordering the fixed windows (here across two plants at distinct source
    /// positions) never changes the emitted fold — the sole map use is a
    /// lookup, the accumulation a `source_pos`-indexed `Vec`.
    #[test]
    fn build_boundary_fold_is_order_invariant() {
        let source = vec![
            anticipated_dated_slot(9, 0, 20_260_401),
            anticipated_dated_slot(7, 0, 20_260_401),
        ];
        let w9 = fixed_window(9, ymd(2026, 4, 8), ymd(2026, 4, 15), 50.0);
        let w7 = fixed_window(7, ymd(2026, 4, 1), ymd(2026, 4, 8), 20.0);

        let fold_ab = build_boundary_fold(&source, &[w9.clone(), w7.clone()]).unwrap();
        let fold_ba = build_boundary_fold(&source, &[w7, w9]).unwrap();

        assert_eq!(fold_ab, fold_ba);
        assert_eq!(fold_ab.len(), 2);
        assert!(
            fold_ab[0].0 < fold_ab[1].0,
            "the fold is emitted ascending by source position"
        );
    }
}
