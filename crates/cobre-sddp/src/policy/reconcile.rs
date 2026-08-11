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
//! incompatible with. Anticipated slots dispatch on `delivery_date`:
//! `SENTINEL` (pre-fan-out padding) defaults to `Zero`; a live, dated slot is
//! [`RebindOp::Reject`] — the dated fan-out reconciliation is not yet
//! implemented, so [`RebindOp::Blend`]/[`RebindOp::Renormalize`] stay an
//! unconstructed seam for that future work, never reached from real input.
//! Every remaining family falls back to the identity-reject default, pending
//! its own arm. When every storage/lag/transit/unclassified target slot has a
//! same-identity source counterpart and every anticipated slot is still
//! sentinel-dated — the shape an already target-aligned, pre-fan-out boundary
//! policy has — the rebind reproduces the source cut's own coefficients
//! bit-for-bit: `Copy` at the matching position for every identity-resolved
//! family, and `Zero` for the sentinel-anticipated slot, whose source
//! coefficient there is itself always `0.0` (a masked state dimension never
//! holds a value). This is the strict-superset guarantee.

use std::collections::HashMap;

use cobre_io::ENTITY_SLOT_DELIVERY_DATE_SENTINEL;
use cobre_io::EntitySlot;
use cobre_io::OwnedPolicyCutRecord;

use crate::SddpError;
use crate::policy::policy_export::{
    ENTITY_TYPE_ANTICIPATED_THERMAL_STATE, ENTITY_TYPE_HYDRO_INFLOW_LAG, ENTITY_TYPE_HYDRO_STORAGE,
    ENTITY_TYPE_HYDRO_TRANSIT_BUCKET,
};

/// One reconciliation operation, producing one target-manifest slot's
/// coefficient from a source cut's coefficients.
///
/// [`build_rebind`] assigns exactly one op per target slot; [`rebind_cut`]
/// applies the assignment to a source cut.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RebindOp {
    /// Take the source coefficient at this position verbatim.
    Copy(usize),
    /// No source counterpart contributes; the coefficient is `0.0`.
    Zero,
    /// A calendar-overlap-weighted blend of source positions.
    ///
    /// [`build_rebind`] never constructs this from real input today — a
    /// dated `AnticipatedThermalState` target slot rejects instead, pending
    /// the dated fan-out reconciliation — so reaching it in [`rebind_cut`] is
    /// unreachable, a [`build_rebind`] postcondition.
    // Rationale: constructed only by this module's tests until the dated
    // fan-out reconciliation lands.
    #[allow(dead_code)]
    Blend,
    /// A [`Self::Blend`] re-normalized over its covered overlap span.
    ///
    /// [`build_rebind`] never constructs this from real input today, for the
    /// same reason as [`Self::Blend`]; reaching it in [`rebind_cut`] is
    /// likewise unreachable.
    // Rationale: constructed only by this module's tests until the dated
    // fan-out reconciliation lands.
    #[allow(dead_code)]
    Renormalize,
    /// The target slot cannot be resolved from `source`: either no
    /// same-identity counterpart exists under a family that requires one
    /// (storage, inflow-lag, the identity fallback — the entity is never
    /// relaxed), or the family's reconciliation is not yet implemented (a
    /// dated `AnticipatedThermalState` slot). A sentinel: [`build_rebind`]
    /// converts this into an [`SddpError::Validation`] rather than returning
    /// it, so it never appears in a successfully built op vector.
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

/// Build one [`RebindOp`] per `target` slot, dispatched per target slot's
/// `entity_type` to [`resolve_storage`], [`resolve_inflow_lag`],
/// [`resolve_transit_bucket`], [`resolve_anticipated`], or the identity
/// fallback [`resolve_by_identity`].
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
/// resolution, or if a `target` slot is a dated (non-sentinel)
/// `AnticipatedThermalState` slot — the dated fan-out reconciliation is not
/// yet implemented.
pub(crate) fn build_rebind(
    source: &[EntitySlot],
    target: &[EntitySlot],
) -> Result<Vec<RebindOp>, SddpError> {
    let mut by_identity: HashMap<SlotKey, usize> = HashMap::with_capacity(source.len());
    for (pos, slot) in source.iter().enumerate() {
        by_identity.insert(slot_key(slot), pos);
    }

    let mut ops = Vec::with_capacity(target.len());
    for (i, slot) in target.iter().enumerate() {
        match resolve_target_slot(i, slot, &by_identity) {
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
/// `delivery_date` alone.
fn resolve_target_slot(
    i: usize,
    slot: &EntitySlot,
    by_identity: &HashMap<SlotKey, usize>,
) -> RebindOp {
    match slot.entity_type {
        ENTITY_TYPE_HYDRO_STORAGE => resolve_storage(slot, by_identity),
        ENTITY_TYPE_HYDRO_INFLOW_LAG => resolve_inflow_lag(slot, by_identity),
        ENTITY_TYPE_HYDRO_TRANSIT_BUCKET => resolve_transit_bucket(slot, by_identity),
        ENTITY_TYPE_ANTICIPATED_THERMAL_STATE => resolve_anticipated(slot),
        _ => resolve_by_identity(i, slot, by_identity),
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

/// `AnticipatedThermalState` resolution, dispatched on `delivery_date` alone
/// (never `by_identity`): a slot padding beyond its own reachable lead
/// (`delivery_date == SENTINEL`) never held a value, so it defaults to
/// `Zero`. A live, dated slot rejects — the dated fan-out reconciliation is
/// not yet implemented — never silently zeroed.
fn resolve_anticipated(slot: &EntitySlot) -> RebindOp {
    if slot.delivery_date == ENTITY_SLOT_DELIVERY_DATE_SENTINEL {
        RebindOp::Zero
    } else {
        RebindOp::Reject {
            reason: format!(
                "boundary policy target has a dated anticipated-thermal-state slot \
                 (entity_id={}, subindex={}, delivery_date={}): post-horizon anticipated date \
                 reconciliation is not yet supported (the dated fan-out reconciliation is not \
                 yet implemented)",
                slot.entity_id, slot.subindex, slot.delivery_date
            ),
        }
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
/// # Panics
///
/// Panics if `rebind` contains a [`RebindOp::Blend`], [`RebindOp::Renormalize`],
/// or [`RebindOp::Reject`] — each is a [`build_rebind`] postcondition
/// violation: `build_rebind` converts every `Reject` into an error before
/// returning, and never constructs `Blend`/`Renormalize` from real input (a
/// dated `AnticipatedThermalState` target slot rejects instead, pending the
/// dated fan-out reconciliation).
pub(crate) fn rebind_cut(cut: &OwnedPolicyCutRecord, rebind: &[RebindOp]) -> Vec<f64> {
    rebind
        .iter()
        .map(|op| match op {
            RebindOp::Copy(pos) => cut.coefficients[*pos],
            RebindOp::Zero => 0.0,
            RebindOp::Blend | RebindOp::Renormalize => unreachable!(
                "{op:?} is a build_rebind postcondition violation: build_rebind never \
                 constructs this from real input"
            ),
            RebindOp::Reject { reason } => unreachable!(
                "build_rebind must convert Reject into an error before rebind_cut sees it: \
                 {reason}"
            ),
        })
        .collect()
}

/// `source` positions no [`RebindOp::Copy`] in `rebind` ever references.
// Rationale: reserved for a future reconciliation-report surface (which
// source slots a load silently dropped); not yet called outside tests.
#[allow(dead_code)]
pub(crate) fn dropped_source_positions(source_len: usize, rebind: &[RebindOp]) -> Vec<usize> {
    let mut referenced = vec![false; source_len];
    for op in rebind {
        if let RebindOp::Copy(pos) = op
            && *pos < source_len
        {
            referenced[*pos] = true;
        }
    }
    referenced
        .into_iter()
        .enumerate()
        .filter_map(|(pos, was_referenced)| (!was_referenced).then_some(pos))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        ENTITY_TYPE_ANTICIPATED_THERMAL_STATE, ENTITY_TYPE_HYDRO_INFLOW_LAG,
        ENTITY_TYPE_HYDRO_TRANSIT_BUCKET, RebindOp, build_rebind, dropped_source_positions,
        rebind_cut,
    };
    use crate::SddpError;
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
            entity_type: ENTITY_TYPE_HYDRO_INFLOW_LAG,
            entity_id: id,
            subindex: lag_depth,
            was_active: true,
            delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        }
    }

    fn transit_bucket_slot(downstream_hydro_id: i32, lag: u32) -> EntitySlot {
        EntitySlot {
            entity_type: ENTITY_TYPE_HYDRO_TRANSIT_BUCKET,
            entity_id: downstream_hydro_id,
            subindex: lag,
            was_active: true,
            delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        }
    }

    fn anticipated_sentinel_slot(thermal_id: i32, ring_slot: u32) -> EntitySlot {
        EntitySlot {
            entity_type: ENTITY_TYPE_ANTICIPATED_THERMAL_STATE,
            entity_id: thermal_id,
            subindex: ring_slot,
            was_active: true,
            delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        }
    }

    fn anticipated_dated_slot(thermal_id: i32, ring_slot: u32, delivery_date: i32) -> EntitySlot {
        EntitySlot {
            entity_type: ENTITY_TYPE_ANTICIPATED_THERMAL_STATE,
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

    /// Given a `source` and `target` manifest of equal shape (all storage),
    /// when `build_rebind` runs, then it returns one `Copy` per slot at the
    /// matching source position.
    #[test]
    fn build_rebind_equal_shape_all_storage_yields_identity_copy() {
        let source = vec![storage_slot(1), storage_slot(2), storage_slot(3)];
        let target = source.clone();

        let rebind = build_rebind(&source, &target).unwrap();

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

        let rebind = build_rebind(&source, &target).unwrap();

        assert_eq!(rebind, vec![RebindOp::Copy(2), RebindOp::Copy(1)]);
    }

    /// Given a target storage slot for a hydro absent from the source, when
    /// `build_rebind` runs, then it rejects, naming the unpriced hydro.
    #[test]
    fn build_rebind_storage_miss_rejects_naming_hydro() {
        let source = vec![storage_slot(1)];
        let target = vec![storage_slot(1), storage_slot(42)];

        let err = build_rebind(&source, &target).unwrap_err();

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

        let err = build_rebind(&source, &target).unwrap_err();

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

        let err = build_rebind(&source, &target).unwrap_err();

        assert!(matches!(err, SddpError::Validation(_)));
    }

    /// Given a rebind of all `Copy` ops and a source cut, when `rebind_cut`
    /// runs, then the returned coefficient vector equals the source cut's
    /// coefficients verbatim.
    #[test]
    fn rebind_cut_all_copy_matches_source_coefficients_verbatim() {
        let source = vec![storage_slot(1), storage_slot(2)];
        let target = source.clone();
        let rebind = build_rebind(&source, &target).unwrap();
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

    /// Given a `Blend` op reached in violation of `build_rebind`'s
    /// postcondition (it never constructs one from real input), when
    /// `rebind_cut` runs, then it panics loudly (never silently returns
    /// `0.0`).
    #[test]
    #[should_panic(expected = "postcondition violation")]
    fn rebind_cut_blend_op_panics_loudly() {
        let rebind = vec![RebindOp::Blend];
        let cut = owned_cut(vec![1.0]);

        let _ = rebind_cut(&cut, &rebind);
    }

    /// Given a `Renormalize` op reached in violation of `build_rebind`'s
    /// postcondition, when `rebind_cut` runs, then it panics loudly (never
    /// silently returns `0.0`).
    #[test]
    #[should_panic(expected = "postcondition violation")]
    fn rebind_cut_renormalize_op_panics_loudly() {
        let rebind = vec![RebindOp::Renormalize];
        let cut = owned_cut(vec![1.0]);

        let _ = rebind_cut(&cut, &rebind);
    }

    #[test]
    fn build_rebind_rejects_target_slot_with_no_source_counterpart() {
        let source = vec![storage_slot(1)];
        let target = vec![storage_slot(1), storage_slot(2)];

        let err = build_rebind(&source, &target).unwrap_err();

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

        let rebind = build_rebind(&source, &matching_target).unwrap();

        assert_eq!(rebind, vec![RebindOp::Copy(0), RebindOp::Copy(1)]);
        let cut = owned_cut(vec![10.0, 99.0]);
        assert_eq!(rebind_cut(&cut, &rebind), vec![10.0, 99.0]);

        let missing_target = vec![storage_slot(1), transit_bucket_slot(3, 1)];

        let rebind = build_rebind(&source, &missing_target).unwrap();

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

        let rebind = build_rebind(&source, &target).unwrap();

        assert_eq!(rebind, vec![RebindOp::Copy(0), RebindOp::Zero]);
    }

    /// Given a target anticipated slot with a live (non-sentinel)
    /// `delivery_date`, when `build_rebind` runs, then it returns a graceful
    /// `Err(SddpError::Validation)` naming the not-yet-supported dated
    /// fan-out reconciliation — never a panic, and never a silent `Zero`
    /// over a real dated slot.
    #[test]
    fn build_rebind_dated_anticipated_target_rejects_gracefully() {
        let source = vec![storage_slot(1)];
        let target = vec![storage_slot(1), anticipated_dated_slot(9, 0, 20_260_301)];

        let err = build_rebind(&source, &target).unwrap_err();

        assert!(matches!(err, SddpError::Validation(_)));
        let msg = err.to_string();
        assert!(
            msg.contains("post-horizon anticipated date reconciliation is not yet supported"),
            "must name the not-yet-supported dated fan-out reconciliation: {msg}"
        );
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
        let rebind = build_rebind(&source, &target).unwrap();
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
        let rebind = build_rebind(&source, &target).unwrap();

        let dropped = dropped_source_positions(source.len(), &rebind);

        assert_eq!(dropped, vec![1, 2]);
    }
}
