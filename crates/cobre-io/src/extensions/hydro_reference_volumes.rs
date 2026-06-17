//! Per-`(plant, stage)` reference operating volume resolution.
//!
//! [`HydroReferenceVolumeFractions::get`] returns the resolved reference operating
//! volume in absolute storage units (hm³) for a given `(hydro_id, stage_id)`. The
//! value is sourced from the declared per-`(plant, stage)` reference volume,
//! already resolved against the plant's `[v_min, v_max]` operating band at the
//! caller. A plant/stage with no declared value falls back to the case default.
//!
//! ## Construction
//!
//! [`build_hydro_reference_volumes_resolved`] is the constructor: it takes
//! already-resolved absolute hm³ values for each `(hydro_id, stage_id)` pair and a
//! scalar default, producing a resolver whose `get` returns those values directly.
//! The resolution of the declared input (absolute, percentile, or the default) to
//! absolute hm³ — including the band against which a percentile is resolved — is
//! performed by the caller before construction, so this module stays free of any
//! band assumption.

use std::collections::HashMap;

use cobre_core::EntityId;

/// Resolver returning the reference operating volume for a given
/// `(hydro_id, stage_id)` pair.
///
/// Built via [`build_hydro_reference_volumes_resolved`], `get` returns an absolute
/// storage volume (hm³) sourced from the dense per-`(plant, stage)` resolved store;
/// a `(hydro, stage)` the caller did not populate falls back to the scalar default.
#[derive(Debug, Clone)]
pub struct HydroReferenceVolumeFractions {
    // Dense per-`(plant, stage)` resolved absolute hm³ store, populated by
    // `build_hydro_reference_volumes_resolved`. When a `(hydro, stage)` key is
    // present, `get` returns its value directly — no formula is applied here, the
    // caller already resolved the declared input against the plant band.
    resolved_hm3: HashMap<(EntityId, usize), f64>,
    default_value: f64,
}

impl HydroReferenceVolumeFractions {
    /// Returns the resolved reference operating volume for `(hydro, stage)`.
    ///
    /// A `(hydro, stage)` present in the resolved-hm³ store returns its value
    /// directly; any key the caller did not populate falls back to the scalar
    /// default.
    #[must_use]
    pub fn get(&self, hydro_id: EntityId, stage_id: usize) -> f64 {
        if let Some(&v) = self.resolved_hm3.get(&(hydro_id, stage_id)) {
            return v;
        }
        self.default_value
    }
}

/// Build a resolver from already-resolved per-`(plant, stage)` reference volumes.
///
/// `resolved_hm3` carries one entry per `(hydro_id, stage_id)` whose reference
/// operating volume has been resolved to an absolute storage value (hm³) by the
/// caller — including the case default for plants/stages with no declared value,
/// and the band against which any percentile was resolved. [`HydroReferenceVolumeFractions::get`]
/// returns the stored value directly for every populated key, and `default_hm3`
/// for any key the caller did not populate (a defensive fallback; callers
/// populate every `(plant, stage)` they will query).
///
/// The construction is order-independent: `resolved_hm3` is folded into a lookup
/// map whose contents do not depend on iteration order, so declaration order of
/// the input pairs does not affect any later `get`.
#[must_use]
pub fn build_hydro_reference_volumes_resolved(
    resolved_hm3: &[(EntityId, usize, f64)],
    default_hm3: f64,
) -> HydroReferenceVolumeFractions {
    let mut map: HashMap<(EntityId, usize), f64> = HashMap::with_capacity(resolved_hm3.len());
    for &(hydro_id, stage_id, value) in resolved_hm3 {
        map.insert((hydro_id, stage_id), value);
    }
    HydroReferenceVolumeFractions {
        resolved_hm3: map,
        default_value: default_hm3,
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use cobre_core::EntityId;

    use super::build_hydro_reference_volumes_resolved;

    #[test]
    fn resolved_constructor_returns_stored_value_directly() {
        // The resolved store returns the absolute hm³ verbatim — no formula.
        let resolver = build_hydro_reference_volumes_resolved(
            &[
                (EntityId(42), 0, 800.0),
                (EntityId(42), 1, 150.0),
                (EntityId(7), 0, 1234.5),
            ],
            999.0,
        );
        assert_eq!(resolver.get(EntityId(42), 0), 800.0);
        assert_eq!(resolver.get(EntityId(42), 1), 150.0);
        assert_eq!(resolver.get(EntityId(7), 0), 1234.5);
    }

    #[test]
    fn resolved_constructor_falls_back_to_default_for_unpopulated_key() {
        let resolver = build_hydro_reference_volumes_resolved(&[(EntityId(42), 0, 800.0)], 999.0);
        // A key the caller did not populate returns the scalar default.
        assert_eq!(resolver.get(EntityId(42), 5), 999.0);
        assert_eq!(resolver.get(EntityId(99), 0), 999.0);
    }

    #[test]
    fn resolved_constructor_is_order_independent() {
        let forward = build_hydro_reference_volumes_resolved(
            &[(EntityId(1), 0, 10.0), (EntityId(2), 0, 20.0)],
            0.0,
        );
        let reverse = build_hydro_reference_volumes_resolved(
            &[(EntityId(2), 0, 20.0), (EntityId(1), 0, 10.0)],
            0.0,
        );
        assert_eq!(
            forward.get(EntityId(1), 0).to_bits(),
            reverse.get(EntityId(1), 0).to_bits()
        );
        assert_eq!(
            forward.get(EntityId(2), 0).to_bits(),
            reverse.get(EntityId(2), 0).to_bits()
        );
    }
}
