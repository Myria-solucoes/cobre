//! Per-stage geometry satellite types: [`EvaporationIndices`] and
//! [`FphaRowRange`].
//!
//! Each locates a single hydro's evaporation columns/row or FPHA row block
//! within one stage LP. They are produced by the per-stage
//! [`StageLayout`](crate::lp_builder) and carried on its
//! [`StageGeometry`](crate::lp_builder::StageGeometry) snapshot; the role-(a)
//! state-vector concern lives on [`StateLayout`](super::StateLayout) and the
//! non-state study shape on [`StudyDimensions`](super::StudyDimensions).

/// Column and row indices for the evaporation constraint of one hydro.
///
/// Locates the three evaporation columns and one evaporation row assigned to
/// a single hydro within a stage LP.  Columns are stage-level (not per-block).
#[derive(Debug, Clone, Copy)]
pub struct EvaporationIndices {
    /// Column index of the stage-averaged evaporation-outflow variable (m³/s).
    pub evaporation_flow_col: usize,
    /// Column index of the positive violation slack `f_evap_plus_h` (m³/s).
    pub f_evap_plus_col: usize,
    /// Column index of the negative violation slack `f_evap_minus_h` (m³/s).
    pub f_evap_minus_col: usize,
    /// Row index of the evaporation equality constraint.
    pub evap_row: usize,
}

/// FPHA constraint row range for one hydro at one stage.
///
/// Locates the block of FPHA hyperplane rows assigned to a single FPHA hydro
/// within a stage LP. Rows for hydro `i` at block `k` and plane `p` are at:
/// `start + k * planes_per_block + p`.
#[derive(Debug, Clone, Copy)]
pub struct FphaRowRange {
    /// First row index of this hydro's FPHA constraints (for block 0, plane 0).
    pub start: usize,
    /// Number of hyperplanes per block.
    pub planes_per_block: usize,
}

#[cfg(test)]
mod tests {
    use super::{EvaporationIndices, FphaRowRange};

    #[test]
    fn evap_indices_debug_clone_copy() {
        let ei = EvaporationIndices {
            evaporation_flow_col: 10,
            f_evap_plus_col: 11,
            f_evap_minus_col: 12,
            evap_row: 5,
        };
        let cloned = ei;
        assert_eq!(cloned.evaporation_flow_col, 10);
        assert_eq!(cloned.evap_row, 5);
        let debug_str = format!("{ei:?}");
        assert!(debug_str.contains("EvaporationIndices"));
    }

    #[test]
    fn fpha_row_range_debug_clone_copy() {
        let r = FphaRowRange {
            start: 42,
            planes_per_block: 5,
        };
        let cloned = r;
        assert_eq!(cloned.start, 42);
        assert_eq!(cloned.planes_per_block, 5);
        let debug_str = format!("{r:?}");
        assert!(debug_str.contains("FphaRowRange"));
    }
}
