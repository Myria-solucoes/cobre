//! Base typed vocabulary for LP addresses and state-vector dimensions.
//!
//! A bare `usize` does not distinguish an LP column from an LP row, a
//! state-vector dimension from an LP column, or an incoming-state column from
//! an outgoing-state one — any of those mismatches still compiles and silently
//! addresses the wrong cell. [`Col`], [`Row`], and [`StateDim`] make each
//! concern its own type; [`InCol`]/[`OutCol`] further split the column role the
//! state-path resolvers return, so a render call site handed an incoming
//! column (or vice versa) is a compile error, not a silently wrong cut
//! coefficient. None of the five carries arithmetic: offset formulas stay with
//! the owning value type ([`BlockGrid`](super::BlockGrid),
//! [`StorageBoundaryGrid`](super::StorageBoundaryGrid), `RangeCursor`,
//! `DeliveryRing`) — these types only gate which `usize` crosses which
//! boundary. Every type is `#[repr(transparent)]` and extracts its raw
//! `usize` via `get()` for the solver-FFI seam, which stays `usize`-only per
//! the infra-genericity rule.

/// A generic LP column index — the solver-FFI seam type (`set_col_bounds`,
/// CSC/CSR column indices) when no incoming/outgoing role applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Col(usize);

impl Col {
    /// Wrap a raw LP column index.
    #[inline]
    #[must_use]
    pub fn new(col: usize) -> Self {
        Self(col)
    }

    /// Extract the raw LP column index for the solver-FFI seam.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// A generic LP row index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Row(usize);

impl Row {
    /// Wrap a raw LP row index.
    #[inline]
    #[must_use]
    pub fn new(row: usize) -> Self {
        Self(row)
    }

    /// Extract the raw LP row index for the solver-FFI seam.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// A state-vector dimension index `j ∈ [0, n_state)` — not an LP address;
/// resolve it through the owning state layout's resolver before indexing an
/// LP buffer.
///
/// A raw `StateDim` cannot stand in for the [`OutCol`] a resolver returns:
///
/// ```compile_fail
/// use cobre_sddp::indexer::{OutCol, StateDim};
///
/// let dim = StateDim::new(0);
/// let _unresolved: OutCol = dim; // StateDim substituted for a resolved OutCol
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct StateDim(usize);

impl StateDim {
    /// Wrap a raw state-vector dimension index.
    #[inline]
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self(dim)
    }

    /// Extract the raw state-vector dimension index.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// An incoming-state LP column — the pin target (`set_col_bounds`) and the
/// dual-extraction read (`reduced_costs[col]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct InCol(usize);

impl InCol {
    /// Wrap a raw incoming-state LP column index.
    #[inline]
    #[must_use]
    pub fn new(col: usize) -> Self {
        Self(col)
    }

    /// Extract the raw LP column index for the solver-FFI seam.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// An outgoing-state LP column — the forward-pass/DCS cut-row render target.
///
/// An [`InCol`] cannot stand in for it: the render path (e.g.
/// `cut::row::push_scaled_coefficient`'s `col: OutCol` parameter) rejects the
/// incoming-column role the pin/dual-extraction path uses.
///
/// ```compile_fail
/// use cobre_sddp::indexer::{InCol, OutCol};
///
/// fn render(_outgoing: OutCol) {}
///
/// render(InCol::new(3)); // render path handed an InCol, not the OutCol it requires
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct OutCol(usize);

impl OutCol {
    /// Wrap a raw outgoing-state LP column index.
    #[inline]
    #[must_use]
    pub fn new(col: usize) -> Self {
        Self(col)
    }

    /// Extract the raw LP column index for the solver-FFI seam.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::{Col, InCol, OutCol, Row, StateDim};

    #[test]
    fn col_is_zero_cost_and_round_trips() {
        assert_eq!(std::mem::size_of::<Col>(), std::mem::size_of::<usize>());
        assert_eq!(Col::new(42).get(), 42);
    }

    #[test]
    fn row_is_zero_cost_and_round_trips() {
        assert_eq!(std::mem::size_of::<Row>(), std::mem::size_of::<usize>());
        assert_eq!(Row::new(7).get(), 7);
    }

    #[test]
    fn state_dim_is_zero_cost_and_round_trips() {
        assert_eq!(
            std::mem::size_of::<StateDim>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(StateDim::new(13).get(), 13);
    }

    #[test]
    fn in_col_is_zero_cost_and_round_trips() {
        assert_eq!(std::mem::size_of::<InCol>(), std::mem::size_of::<usize>());
        assert_eq!(InCol::new(5).get(), 5);
    }

    #[test]
    fn out_col_is_zero_cost_and_round_trips() {
        assert_eq!(std::mem::size_of::<OutCol>(), std::mem::size_of::<usize>());
        assert_eq!(OutCol::new(9).get(), 9);
    }

    #[test]
    fn col_equality_is_value_based_not_identity() {
        assert_eq!(Col::new(3), Col::new(3));
        assert_ne!(Col::new(3), Col::new(4));
    }

    /// `InCol`/`OutCol` are distinct types even when constructed from the same
    /// raw value — the compile-time guard the in/out role split exists for
    /// cannot be demonstrated by a runtime assertion, so this test pins only
    /// what runtime code CAN observe: each wraps and returns its own value
    /// independently of the other's existence.
    #[test]
    fn in_col_and_out_col_round_trip_independently() {
        assert_eq!(InCol::new(7).get(), 7);
        assert_eq!(OutCol::new(7).get(), 7);
    }
}
