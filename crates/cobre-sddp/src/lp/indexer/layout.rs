//! Per-stage geometry satellite type: [`EvaporationIndices`].
//!
//! [`EvaporationIndices`] locates a single hydro's evaporation columns and row
//! within one stage LP; the per-stage [`StageLayout`](crate::lp::builder)
//! produces it and carries it on its
//! [`StageGeometry`](crate::lp::builder::StageGeometry) snapshot as
//! `evap_indices`. The role-(a) state-vector concern lives on
//! [`StateSpace`](super::StateSpace) and the non-state study shape on
//! [`StudyDimensions`](super::StudyDimensions).

/// Column and row indices for one evaporation constraint.
///
/// Locates the three evaporation columns and one evaporation row of a single
/// `(hydro, block)` pair within a stage LP. The producing layout holds one entry
/// per `(evap hydro, block)`, block-major.
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

#[cfg(test)]
mod tests {
    use super::EvaporationIndices;

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
}
