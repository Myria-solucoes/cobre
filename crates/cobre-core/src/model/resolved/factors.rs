//! Pre-resolved per-block factor and NCS-availability lookup tables.
//!
//! Holds the dense factor tables consumed on the LP-building hot path:
//! `ResolvedLoadFactors` and `ResolvedExchangeFactors` (per-`(entity, stage,
//! block)` scaling), plus the non-controllable-source family `ResolvedNcsBounds`
//! (per-`(ncs, stage)` available generation) and `ResolvedNcsFactors`
//! (per-`(ncs, stage, block)` scaling). Absent entries return the no-scaling
//! identity (`1.0`, or `(1.0, 1.0)` for exchange) and absent NCS availability
//! returns `0.0`. Populated by `cobre-io`; never modified after construction.

// ─── Block factor lookup tables ──────────────────────────────────────────────

/// Pre-resolved per-block load scaling factors.
///
/// Provides O(1) lookup of load block factors by `(bus_index, stage_index,
/// block_index)`. Returns `1.0` for absent entries (no scaling). Populated
/// by `cobre-io` during the resolution step and stored in [`crate::System`].
///
/// Uses dense 3D storage (`n_buses * n_stages * max_blocks`) initialized to
/// `1.0`. The total size is small (typically < 10K entries) and the lookup is
/// on the LP-building hot path.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::ResolvedLoadFactors;
///
/// let empty = ResolvedLoadFactors::empty();
/// assert!((empty.factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolvedLoadFactors {
    /// Dense 3D array stored flat: `[bus_idx][stage_idx][block_idx]`.
    /// Dimensions: `n_buses * n_stages * max_blocks`.
    factors: Vec<f64>,
    /// Number of stages.
    n_stages: usize,
    /// Maximum number of blocks across all stages.
    max_blocks: usize,
}

impl ResolvedLoadFactors {
    /// Create an empty load factors table. All lookups return `1.0`.
    ///
    /// Used as the default when no `load_factors.json` exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::resolved::ResolvedLoadFactors;
    ///
    /// let t = ResolvedLoadFactors::empty();
    /// assert!((t.factor(5, 3, 2) - 1.0).abs() < f64::EPSILON);
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            factors: Vec::new(),
            n_stages: 0,
            max_blocks: 0,
        }
    }

    /// Create a new load factors table with the given dimensions.
    ///
    /// All entries are initialized to `1.0` (no scaling). Use [`set`] to
    /// populate individual entries.
    ///
    /// [`set`]: Self::set
    #[must_use]
    pub fn new(n_buses: usize, n_stages: usize, max_blocks: usize) -> Self {
        Self {
            factors: vec![1.0; n_buses * n_stages * max_blocks],
            n_stages,
            max_blocks,
        }
    }

    /// Set the load factor for a specific `(bus_idx, stage_idx, block_idx)` triple.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of bounds.
    pub fn set(&mut self, bus_idx: usize, stage_idx: usize, block_idx: usize, value: f64) {
        let idx = (bus_idx * self.n_stages + stage_idx) * self.max_blocks + block_idx;
        self.factors[idx] = value;
    }

    /// Look up the load factor for a `(bus_idx, stage_idx, block_idx)` triple.
    ///
    /// Returns `1.0` when the table is empty or the computed flat index falls
    /// past `Vec::len`.
    ///
    /// Contract: the `1.0` identity fallback is only guaranteed when the flat
    /// index `(bus_idx * n_stages + stage_idx) * max_blocks + block_idx` lands
    /// past the end of the backing `Vec`. A per-dimension overflow that stays
    /// within `Vec::len` — e.g. `block_idx >= max_blocks` while `bus_idx` is
    /// small — aliases into a neighbouring cell rather than returning `1.0`.
    /// Callers (`lp/builder/matrix.rs`) only pass in-range dimensions, so this
    /// is unreachable in practice; do not rely on the fallback for arbitrary
    /// out-of-range dimension combinations.
    #[inline]
    #[must_use]
    pub fn factor(&self, bus_idx: usize, stage_idx: usize, block_idx: usize) -> f64 {
        if self.factors.is_empty() {
            return 1.0;
        }
        let idx = (bus_idx * self.n_stages + stage_idx) * self.max_blocks + block_idx;
        self.factors.get(idx).copied().unwrap_or(1.0)
    }
}

/// Pre-resolved per-block exchange capacity factors.
///
/// Provides O(1) lookup of exchange factors by `(line_index, stage_index,
/// block_index)` returning `(direct_factor, reverse_factor)`. Returns
/// `(1.0, 1.0)` for absent entries. Populated by `cobre-io` during the
/// resolution step and stored in [`crate::System`].
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::ResolvedExchangeFactors;
///
/// let empty = ResolvedExchangeFactors::empty();
/// assert_eq!(empty.factors(0, 0, 0), (1.0, 1.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolvedExchangeFactors {
    /// Dense 3D array stored flat: `[line_idx][stage_idx][block_idx]`.
    /// Each entry stores `(direct_factor, reverse_factor)`.
    data: Vec<(f64, f64)>,
    /// Number of stages.
    n_stages: usize,
    /// Maximum number of blocks across all stages.
    max_blocks: usize,
}

impl ResolvedExchangeFactors {
    /// Create an empty exchange factors table. All lookups return `(1.0, 1.0)`.
    ///
    /// Used as the default when no `exchange_factors.json` exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::resolved::ResolvedExchangeFactors;
    ///
    /// let t = ResolvedExchangeFactors::empty();
    /// assert_eq!(t.factors(5, 3, 2), (1.0, 1.0));
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            n_stages: 0,
            max_blocks: 0,
        }
    }

    /// Create a new exchange factors table with the given dimensions.
    ///
    /// All entries are initialized to `(1.0, 1.0)` (no scaling). Use [`set`]
    /// to populate individual entries.
    ///
    /// [`set`]: Self::set
    #[must_use]
    pub fn new(n_lines: usize, n_stages: usize, max_blocks: usize) -> Self {
        Self {
            data: vec![(1.0, 1.0); n_lines * n_stages * max_blocks],
            n_stages,
            max_blocks,
        }
    }

    /// Set the exchange factors for a specific `(line_idx, stage_idx, block_idx)` triple.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of bounds.
    pub fn set(
        &mut self,
        line_idx: usize,
        stage_idx: usize,
        block_idx: usize,
        direct_factor: f64,
        reverse_factor: f64,
    ) {
        let idx = (line_idx * self.n_stages + stage_idx) * self.max_blocks + block_idx;
        self.data[idx] = (direct_factor, reverse_factor);
    }

    /// Look up the exchange factors for a `(line_idx, stage_idx, block_idx)` triple.
    ///
    /// Returns `(direct_factor, reverse_factor)`. Returns `(1.0, 1.0)` when the
    /// table is empty or the computed flat index falls past `Vec::len`.
    ///
    /// Contract: the `(1.0, 1.0)` fallback is only guaranteed when the flat
    /// index lands past the end of the backing `Vec`. A per-dimension overflow
    /// that stays within `Vec::len` aliases into a neighbouring cell rather than
    /// returning the identity. Callers only pass in-range dimensions, so this is
    /// unreachable in practice.
    #[inline]
    #[must_use]
    pub fn factors(&self, line_idx: usize, stage_idx: usize, block_idx: usize) -> (f64, f64) {
        if self.data.is_empty() {
            return (1.0, 1.0);
        }
        let idx = (line_idx * self.n_stages + stage_idx) * self.max_blocks + block_idx;
        self.data.get(idx).copied().unwrap_or((1.0, 1.0))
    }
}

/// Pre-resolved per-stage NCS available generation bounds.
///
/// Provides O(1) lookup of `available_generation_mw` by `(ncs_index, stage_index)`.
/// Returns `0.0` for out-of-bounds access. Populated by `cobre-io` during the
/// resolution step and stored in [`crate::System`].
///
/// Uses dense 2D storage (`n_ncs * n_stages`) initialized with each NCS entity's
/// installed capacity (`max_generation_mw`). Stage-varying overrides from
/// `constraints/ncs_bounds.parquet` replace individual entries.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::ResolvedNcsBounds;
///
/// let empty = ResolvedNcsBounds::empty();
/// assert!(empty.is_empty());
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolvedNcsBounds {
    /// Dense 2D array: `[ncs_idx * n_stages + stage_idx]`.
    data: Vec<f64>,
    /// Number of stages.
    n_stages: usize,
}

impl ResolvedNcsBounds {
    /// Create an empty NCS bounds table.
    ///
    /// Used as the default when no NCS entities exist or no bounds file is provided.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::resolved::ResolvedNcsBounds;
    ///
    /// let t = ResolvedNcsBounds::empty();
    /// assert!(t.is_empty());
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            n_stages: 0,
        }
    }

    /// Create a new NCS bounds table with per-entity defaults.
    ///
    /// All stages for NCS entity `i` are initialized to `default_mw[i]`
    /// (the installed capacity). Use [`set`] to apply stage-varying overrides.
    ///
    /// [`set`]: Self::set
    ///
    /// # Panics
    ///
    /// Panics if `default_mw.len() != n_ncs`.
    #[must_use]
    pub fn new(n_ncs: usize, n_stages: usize, default_mw: &[f64]) -> Self {
        assert!(
            default_mw.len() == n_ncs,
            "default_mw length ({}) must equal n_ncs ({n_ncs})",
            default_mw.len()
        );
        let mut data = vec![0.0; n_ncs * n_stages];
        for (ncs_idx, &mw) in default_mw.iter().enumerate() {
            for stage_idx in 0..n_stages {
                data[ncs_idx * n_stages + stage_idx] = mw;
            }
        }
        Self { data, n_stages }
    }

    /// Set the available generation for a specific `(ncs_idx, stage_idx)` pair.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of bounds.
    pub fn set(&mut self, ncs_idx: usize, stage_idx: usize, value: f64) {
        let idx = ncs_idx * self.n_stages + stage_idx;
        self.data[idx] = value;
    }

    /// Look up the available generation (MW) for a `(ncs_idx, stage_idx)` pair.
    ///
    /// Returns `0.0` when the index is out of bounds or the table is empty.
    #[inline]
    #[must_use]
    pub fn available_generation(&self, ncs_idx: usize, stage_idx: usize) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        let idx = ncs_idx * self.n_stages + stage_idx;
        self.data.get(idx).copied().unwrap_or(0.0)
    }

    /// Returns `true` when the table has no data.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Pre-resolved per-block NCS generation scaling factors.
///
/// Provides O(1) lookup of the generation factor by `(ncs_index, stage_index,
/// block_index)`. Returns `1.0` for absent entries (no scaling). Populated
/// by `cobre-io` during the resolution step and stored in [`crate::System`].
///
/// Uses dense 3D storage (`n_ncs * n_stages * max_blocks`) initialized to
/// `1.0`. The total size is small (typically < 10K entries) and the lookup is
/// on the LP-building hot path.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::ResolvedNcsFactors;
///
/// let empty = ResolvedNcsFactors::empty();
/// assert!((empty.factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolvedNcsFactors {
    /// Dense 3D array stored flat: `[ncs_idx][stage_idx][block_idx]`.
    /// Dimensions: `n_ncs * n_stages * max_blocks`.
    factors: Vec<f64>,
    /// Number of stages.
    n_stages: usize,
    /// Maximum number of blocks across all stages.
    max_blocks: usize,
}

impl ResolvedNcsFactors {
    /// Create an empty NCS factors table. All lookups return `1.0`.
    ///
    /// Used as the default when no `non_controllable_factors.json` exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::resolved::ResolvedNcsFactors;
    ///
    /// let t = ResolvedNcsFactors::empty();
    /// assert!((t.factor(5, 3, 2) - 1.0).abs() < f64::EPSILON);
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            factors: Vec::new(),
            n_stages: 0,
            max_blocks: 0,
        }
    }

    /// Create a new NCS factors table with the given dimensions.
    ///
    /// All entries are initialized to `1.0` (no scaling). Use [`set`] to
    /// populate individual entries.
    ///
    /// [`set`]: Self::set
    #[must_use]
    pub fn new(n_ncs: usize, n_stages: usize, max_blocks: usize) -> Self {
        Self {
            factors: vec![1.0; n_ncs * n_stages * max_blocks],
            n_stages,
            max_blocks,
        }
    }

    /// Set the NCS factor for a specific `(ncs_idx, stage_idx, block_idx)` triple.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of bounds.
    pub fn set(&mut self, ncs_idx: usize, stage_idx: usize, block_idx: usize, value: f64) {
        let idx = (ncs_idx * self.n_stages + stage_idx) * self.max_blocks + block_idx;
        self.factors[idx] = value;
    }

    /// Look up the NCS factor for a `(ncs_idx, stage_idx, block_idx)` triple.
    ///
    /// Returns `1.0` when the table is empty or the computed flat index falls
    /// past `Vec::len`.
    ///
    /// Contract: the `1.0` identity fallback is only guaranteed when the flat
    /// index lands past the end of the backing `Vec`. A per-dimension overflow
    /// that stays within `Vec::len` aliases into a neighbouring cell rather than
    /// returning `1.0`. Callers only pass in-range dimensions, so this is
    /// unreachable in practice.
    #[inline]
    #[must_use]
    pub fn factor(&self, ncs_idx: usize, stage_idx: usize, block_idx: usize) -> f64 {
        if self.factors.is_empty() {
            return 1.0;
        }
        let idx = (ncs_idx * self.n_stages + stage_idx) * self.max_blocks + block_idx;
        self.factors.get(idx).copied().unwrap_or(1.0)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── ResolvedLoadFactors tests ─────────────────────────────────────────────

    #[test]
    fn test_load_factors_empty_returns_one() {
        let t = ResolvedLoadFactors::empty();
        assert!((t.factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((t.factor(5, 3, 2) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_factors_new_default_is_one() {
        let t = ResolvedLoadFactors::new(2, 1, 3);
        for bus in 0..2 {
            for blk in 0..3 {
                assert!(
                    (t.factor(bus, 0, blk) - 1.0).abs() < f64::EPSILON,
                    "expected 1.0 at ({bus}, 0, {blk})"
                );
            }
        }
    }

    #[test]
    fn test_load_factors_set_and_get() {
        let mut t = ResolvedLoadFactors::new(2, 1, 3);
        t.set(0, 0, 0, 0.85);
        t.set(0, 0, 1, 1.15);
        assert!((t.factor(0, 0, 0) - 0.85).abs() < 1e-10);
        assert!((t.factor(0, 0, 1) - 1.15).abs() < 1e-10);
        assert!((t.factor(0, 0, 2) - 1.0).abs() < f64::EPSILON);
        // Bus 1 untouched.
        assert!((t.factor(1, 0, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_factors_out_of_bounds_returns_one() {
        let t = ResolvedLoadFactors::new(1, 1, 2);
        // Out of bounds on bus index.
        assert!((t.factor(5, 0, 0) - 1.0).abs() < f64::EPSILON);
        // Out of bounds on block index.
        assert!((t.factor(0, 0, 99) - 1.0).abs() < f64::EPSILON);
    }

    // ─── ResolvedExchangeFactors tests ─────────────────────────────────────────

    #[test]
    fn test_exchange_factors_empty_returns_one_one() {
        let t = ResolvedExchangeFactors::empty();
        assert_eq!(t.factors(0, 0, 0), (1.0, 1.0));
        assert_eq!(t.factors(5, 3, 2), (1.0, 1.0));
    }

    #[test]
    fn test_exchange_factors_new_default_is_one_one() {
        let t = ResolvedExchangeFactors::new(1, 1, 2);
        assert_eq!(t.factors(0, 0, 0), (1.0, 1.0));
        assert_eq!(t.factors(0, 0, 1), (1.0, 1.0));
    }

    #[test]
    fn test_exchange_factors_set_and_get() {
        let mut t = ResolvedExchangeFactors::new(1, 1, 2);
        t.set(0, 0, 0, 0.9, 0.85);
        assert_eq!(t.factors(0, 0, 0), (0.9, 0.85));
        assert_eq!(t.factors(0, 0, 1), (1.0, 1.0));
    }

    #[test]
    fn test_exchange_factors_out_of_bounds_returns_default() {
        let t = ResolvedExchangeFactors::new(1, 1, 1);
        assert_eq!(t.factors(5, 0, 0), (1.0, 1.0));
    }

    // ─── ResolvedNcsBounds tests ──────────────────────────────────────────────

    #[test]
    fn test_ncs_bounds_empty_is_empty() {
        let t = ResolvedNcsBounds::empty();
        assert!(t.is_empty());
        assert!((t.available_generation(0, 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ncs_bounds_new_uses_defaults() {
        let t = ResolvedNcsBounds::new(2, 3, &[100.0, 200.0]);
        assert!(!t.is_empty());
        assert!((t.available_generation(0, 0) - 100.0).abs() < f64::EPSILON);
        assert!((t.available_generation(0, 2) - 100.0).abs() < f64::EPSILON);
        assert!((t.available_generation(1, 0) - 200.0).abs() < f64::EPSILON);
        assert!((t.available_generation(1, 2) - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ncs_bounds_set_and_get() {
        let mut t = ResolvedNcsBounds::new(2, 3, &[100.0, 200.0]);
        t.set(0, 1, 50.0);
        assert!((t.available_generation(0, 1) - 50.0).abs() < f64::EPSILON);
        // Other entries unchanged.
        assert!((t.available_generation(0, 0) - 100.0).abs() < f64::EPSILON);
        assert!((t.available_generation(1, 0) - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ncs_bounds_out_of_bounds_returns_zero() {
        let t = ResolvedNcsBounds::new(1, 1, &[100.0]);
        assert!((t.available_generation(5, 0) - 0.0).abs() < f64::EPSILON);
        assert!((t.available_generation(0, 99) - 0.0).abs() < f64::EPSILON);
    }

    // ─── ResolvedNcsFactors tests ─────────────────────────────────────────────

    #[test]
    fn test_ncs_factors_empty_returns_one() {
        let t = ResolvedNcsFactors::empty();
        assert!((t.factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((t.factor(5, 3, 2) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ncs_factors_new_default_is_one() {
        let t = ResolvedNcsFactors::new(2, 1, 3);
        for ncs in 0..2 {
            for blk in 0..3 {
                assert!(
                    (t.factor(ncs, 0, blk) - 1.0).abs() < f64::EPSILON,
                    "factor({ncs}, 0, {blk}) should be 1.0"
                );
            }
        }
    }

    #[test]
    fn test_ncs_factors_set_and_get() {
        let mut t = ResolvedNcsFactors::new(2, 1, 3);
        t.set(0, 0, 1, 0.8);
        assert!((t.factor(0, 0, 1) - 0.8).abs() < 1e-10);
        assert!((t.factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((t.factor(1, 0, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ncs_factors_out_of_bounds_returns_one() {
        let t = ResolvedNcsFactors::new(1, 1, 2);
        assert!((t.factor(5, 0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((t.factor(0, 0, 99) - 1.0).abs() < f64::EPSILON);
    }
}
