//! Deterministic seed derivation via SipHash-1-3.
//!
//! Every function returns the same `u64` for identical inputs regardless of MPI
//! rank, thread ID, or process restart — the basis for communication-free noise.
//!
//! Domain separation: SipHash-1-3 folds message length into its state, so the
//! variants' differing wire-format lengths (20 / 16 / 12 bytes, plus the
//! grouped variant's `0x01` prefix) keep their outputs distinct even when the
//! numeric arguments overlap. Changing a function's byte layout breaks this and
//! the golden-value tests.

use siphasher::sip::SipHasher13;
use std::hash::Hasher;

/// Derive a deterministic seed for forward-pass noise from
/// `(base_seed, iteration, scenario, stage)`.
#[must_use]
pub fn derive_forward_seed(base_seed: u64, iteration: u32, scenario: u32, stage: u32) -> u64 {
    let mut hasher = SipHasher13::new();
    hasher.write(&base_seed.to_le_bytes());
    hasher.write(&iteration.to_le_bytes());
    hasher.write(&scenario.to_le_bytes());
    hasher.write(&stage.to_le_bytes());
    hasher.finish()
}

/// Derive a deterministic seed for stage-level batch generation from
/// `(base_seed, stage_id)`.
///
/// Intended for batch noise methods (LHS, QMC) that need all openings at a stage
/// simultaneously.
#[must_use]
pub fn derive_stage_seed(base_seed: u64, stage_id: u32) -> u64 {
    let mut hasher = SipHasher13::new();
    hasher.write(&base_seed.to_le_bytes());
    hasher.write(&stage_id.to_le_bytes());
    hasher.finish()
}

/// Derive a deterministic seed for opening-tree generation from
/// `(base_seed, opening_index, stage)`.
#[must_use]
pub fn derive_opening_seed(base_seed: u64, opening_index: u32, stage: u32) -> u64 {
    let mut hasher = SipHasher13::new();
    hasher.write(&base_seed.to_le_bytes());
    hasher.write(&opening_index.to_le_bytes());
    hasher.write(&stage.to_le_bytes());
    hasher.finish()
}

/// Derive a deterministic seed for grouped forward-pass noise from
/// `(base_seed, iteration, scenario, group_id)`.
///
/// `group_id` replaces the per-stage ID so all stages in one noise group share a
/// seed — e.g. weekly stages in a `(season_id, year)` bucket sharing monthly PAR
/// noise. The `0x01` prefix byte keeps its output distinct from
/// [`derive_forward_seed`] even when every numeric argument matches.
#[must_use]
pub fn derive_forward_seed_grouped(
    base_seed: u64,
    iteration: u32,
    scenario: u32,
    group_id: u32,
) -> u64 {
    let mut hasher = SipHasher13::new();
    hasher.write(&[0x01]); // domain separator — absent in derive_forward_seed
    hasher.write(&base_seed.to_le_bytes());
    hasher.write(&iteration.to_le_bytes());
    hasher.write(&scenario.to_le_bytes());
    hasher.write(&group_id.to_le_bytes());
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::{
        derive_forward_seed, derive_forward_seed_grouped, derive_opening_seed, derive_stage_seed,
    };

    // -------------------------------------------------------------------------
    // derive_forward_seed: determinism
    // -------------------------------------------------------------------------

    #[test]
    fn forward_seed_is_deterministic() {
        assert_eq!(
            derive_forward_seed(42, 0, 0, 0),
            derive_forward_seed(42, 0, 0, 0),
        );
    }

    #[test]
    fn forward_seed_varies_with_stage() {
        assert_ne!(
            derive_forward_seed(42, 0, 0, 0),
            derive_forward_seed(42, 0, 0, 1),
        );
    }

    #[test]
    fn forward_seed_varies_with_scenario() {
        assert_ne!(
            derive_forward_seed(42, 0, 0, 0),
            derive_forward_seed(42, 0, 1, 0),
        );
    }

    #[test]
    fn forward_seed_varies_with_iteration() {
        assert_ne!(
            derive_forward_seed(42, 0, 0, 0),
            derive_forward_seed(42, 1, 0, 0),
        );
    }

    #[test]
    fn forward_seed_varies_with_base_seed() {
        assert_ne!(
            derive_forward_seed(42, 0, 0, 0),
            derive_forward_seed(43, 0, 0, 0),
        );
    }

    // -------------------------------------------------------------------------
    // derive_opening_seed: determinism
    // -------------------------------------------------------------------------

    #[test]
    fn opening_seed_is_deterministic() {
        assert_eq!(derive_opening_seed(42, 0, 0), derive_opening_seed(42, 0, 0),);
    }

    #[test]
    fn opening_seed_varies_with_stage() {
        assert_ne!(derive_opening_seed(42, 0, 0), derive_opening_seed(42, 0, 1),);
    }

    #[test]
    fn opening_seed_varies_with_opening_index() {
        assert_ne!(derive_opening_seed(42, 0, 0), derive_opening_seed(42, 1, 0),);
    }

    #[test]
    fn opening_seed_varies_with_base_seed() {
        assert_ne!(derive_opening_seed(42, 0, 0), derive_opening_seed(43, 0, 0),);
    }

    // -------------------------------------------------------------------------
    // Cross-function differentiation: 16-byte vs 20-byte wire format
    // -------------------------------------------------------------------------

    /// 16-byte (opening) vs 20-byte (forward) inputs must differ — see the
    /// length-based domain-separation contract in the module doc.
    #[test]
    fn forward_and_opening_seeds_differ_for_same_partial_inputs() {
        assert_ne!(
            derive_opening_seed(42, 0, 0),
            derive_forward_seed(42, 0, 0, 0),
        );
    }

    // -------------------------------------------------------------------------
    // Golden value regression: pin the output to catch algorithm changes
    // -------------------------------------------------------------------------

    /// Failure means the SipHash-1-3 wire format or the `siphasher` crate version
    /// changed in a breaking way.
    #[test]
    fn forward_seed_golden_value() {
        let seed = derive_forward_seed(42, 0, 0, 0);
        // Golden value recorded from siphasher 1.0.2 with zero key.
        assert_eq!(seed, 4_418_977_803_187_233_897_u64);
    }

    // -------------------------------------------------------------------------
    // derive_stage_seed: determinism
    // -------------------------------------------------------------------------

    #[test]
    fn stage_seed_is_deterministic() {
        assert_eq!(derive_stage_seed(42, 0), derive_stage_seed(42, 0));
    }

    #[test]
    fn stage_seed_varies_with_stage() {
        assert_ne!(derive_stage_seed(42, 0), derive_stage_seed(42, 1));
    }

    #[test]
    fn stage_seed_varies_with_base_seed() {
        assert_ne!(derive_stage_seed(42, 0), derive_stage_seed(43, 0));
    }

    // -------------------------------------------------------------------------
    // Cross-function differentiation: 12-byte vs 16-byte and 20-byte wire formats
    // -------------------------------------------------------------------------

    /// 12-byte (stage) vs 16-byte (opening) inputs must differ — length-based
    /// domain separation (module doc).
    #[test]
    fn stage_seed_differs_from_opening_seed() {
        assert_ne!(derive_stage_seed(42, 0), derive_opening_seed(42, 0, 0));
    }

    /// 12-byte (stage) vs 20-byte (forward) inputs must differ — length-based
    /// domain separation (module doc).
    #[test]
    fn stage_seed_differs_from_forward_seed() {
        assert_ne!(derive_stage_seed(42, 0), derive_forward_seed(42, 0, 0, 0));
    }

    // -------------------------------------------------------------------------
    // Golden value regression: pin derive_stage_seed output
    // -------------------------------------------------------------------------

    /// Failure means the SipHash-1-3 wire format or the `siphasher` crate version
    /// changed in a breaking way.
    #[test]
    fn stage_seed_golden_value() {
        let seed = derive_stage_seed(42, 0);
        // Golden value recorded from siphasher 1.0.2 with zero key.
        assert_eq!(seed, 983_776_962_555_776_753_u64);
    }

    // -------------------------------------------------------------------------
    // derive_forward_seed_grouped: determinism and domain separation
    // -------------------------------------------------------------------------

    #[test]
    fn test_derive_forward_seed_grouped_deterministic() {
        assert_eq!(
            derive_forward_seed_grouped(42, 3, 7, 5),
            derive_forward_seed_grouped(42, 3, 7, 5),
        );
    }

    /// The `0x01` prefix must keep the grouped variant distinct from
    /// `derive_forward_seed` even with identical numeric arguments.
    #[test]
    fn test_derive_forward_seed_grouped_differs_from_forward() {
        assert_ne!(
            derive_forward_seed_grouped(42, 0, 0, 5),
            derive_forward_seed(42, 0, 0, 5),
        );
    }
}
