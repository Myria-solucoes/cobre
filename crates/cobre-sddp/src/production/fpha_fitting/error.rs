//! FPHA fitting error type.
//!
//! Owns [`FphaFittingError`], the validation-error enum returned by every
//! fallible step of the fitting pipeline (geometry-table construction, bounds
//! resolution, kappa computation, and coefficient-sign validation).

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that arise during FPHA fitting geometry validation or evaluation.
///
/// Returned by [`ForebayTable::new`](super::geometry::ForebayTable::new) when the
/// supplied VHA curve data does not satisfy the invariants required for linear
/// interpolation.
#[derive(Debug)]
pub(crate) enum FphaFittingError {
    /// Fewer than 2 VHA curve points were provided for the named hydro plant.
    ///
    /// Linear interpolation requires at least 2 breakpoints. A single point
    /// defines only a trivial (constant) function and cannot represent the
    /// full volume-height relationship.
    InsufficientPoints {
        /// Name of the hydro plant whose curve was rejected.
        hydro_name: String,
        /// Number of points actually provided.
        count: usize,
    },

    /// The `volume_hm3` values are not strictly increasing between consecutive rows.
    ///
    /// Strict monotonicity is required so that each volume maps to a unique
    /// interpolation interval. Duplicate volumes produce a zero-length segment
    /// and undefined derivatives.
    NonMonotonicVolume {
        /// Name of the hydro plant whose curve was rejected.
        hydro_name: String,
        /// Zero-based index of the row whose volume is not strictly greater than
        /// the previous row's volume.
        index: usize,
        /// Volume at the previous row (hm³).
        v_prev: f64,
        /// Volume at the current row (hm³), which must satisfy `v_curr > v_prev`.
        v_curr: f64,
    },

    /// The `height_m` values decrease between consecutive rows.
    ///
    /// Heights must be monotonically non-decreasing because greater reservoir
    /// volume always corresponds to a higher or equal water surface elevation.
    NonMonotonicHeight {
        /// Name of the hydro plant whose curve was rejected.
        hydro_name: String,
        /// Zero-based index of the row whose height is strictly less than the
        /// previous row's height.
        index: usize,
        /// Height at the previous row (m).
        h_prev: f64,
        /// Height at the current row (m), which must satisfy `h_curr >= h_prev`.
        h_curr: f64,
    },

    /// Both absolute and percentile bounds were specified for the same dimension.
    ///
    /// `volume_min_hm3` and `volume_min_percentile` are mutually exclusive, as
    /// are `volume_max_hm3` and `volume_max_percentile`. Setting both for the
    /// same bound is ambiguous and is always rejected.
    ConflictingFittingWindow {
        /// Name of the hydro plant whose configuration was rejected.
        hydro_name: String,
        /// Human-readable description of the conflict.
        detail: String,
    },

    /// The resolved volume range is empty (`v_min >= v_max`).
    ///
    /// After applying the fitting window configuration, the lower bound was
    /// not strictly less than the upper bound. This can happen when absolute
    /// bounds are inverted, when percentile bounds yield a zero-width range,
    /// or when clamping collapses the window to a single point.
    EmptyFittingWindow {
        /// Name of the hydro plant whose configuration was rejected.
        hydro_name: String,
        /// Resolved lower bound (hm³).
        v_min: f64,
        /// Resolved upper bound (hm³).
        v_max: f64,
    },

    /// A discretization count was too small to define a valid grid interval.
    ///
    /// All three dimension counts (`n_volume_points`, `n_flow_points`,
    /// `n_spillage_points`) must be >= 2. `max_planes_per_hydro` must be >= 1.
    InsufficientDiscretization {
        /// Name of the hydro plant whose configuration was rejected.
        hydro_name: String,
        /// Which dimension was too small: `"volume"`, `"turbine"`, `"spillage"`,
        /// or `"max_planes_per_hydro"`.
        dimension: String,
        /// The value that was provided (< 2 for grid dimensions, < 1 for max planes).
        value: usize,
    },

    /// The computed kappa correction factor is outside the valid range `(0, 1]`.
    ///
    /// Kappa must be strictly positive (zero production everywhere is degenerate)
    /// and at most 1.0 (a kappa > 1.0 would mean the envelope underestimates phi,
    /// which violates the outer-approximation guarantee).
    InvalidKappa {
        /// Name of the hydro plant whose fitting was rejected.
        hydro_name: String,
        /// The kappa value that was computed.
        kappa: f64,
    },

    /// The fitting pipeline produced zero valid hyperplanes.
    ///
    /// This can occur when every sampled grid point has zero or negative production
    /// (e.g., net head ≤ 0 everywhere), so no tangent planes can be constructed.
    NoHyperplanesProduced {
        /// Name of the hydro plant for which no hyperplanes were produced.
        hydro_name: String,
    },

    /// A fitted hyperplane has a coefficient with the wrong sign.
    ///
    /// Valid physical hyperplanes satisfy `gamma_v > 0` (more storage → more head →
    /// more power), `gamma_q > 0` (turbining produces power), and `gamma_s <= 0`
    /// (spillage raises tailrace, reducing net head). A coefficient outside these
    /// bounds indicates a numerical problem during fitting.
    InvalidCoefficient {
        /// Name of the hydro plant whose fitting was rejected.
        hydro_name: String,
        /// Zero-based index of the offending hyperplane in the selected set.
        plane_index: usize,
        /// Human-readable description of which coefficient failed and its value.
        detail: String,
    },
}

impl std::fmt::Display for FphaFittingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientPoints { hydro_name, count } => write!(
                f,
                "hydro '{hydro_name}': VHA curve has {count} point(s); \
                 at least 2 are required for interpolation"
            ),
            Self::NonMonotonicVolume {
                hydro_name,
                index,
                v_prev,
                v_curr,
            } => write!(
                f,
                "hydro '{hydro_name}': volume is not strictly increasing at index {index}: \
                 v[{index}]={v_curr} is not greater than v[{}]={v_prev}",
                index - 1
            ),
            Self::NonMonotonicHeight {
                hydro_name,
                index,
                h_prev,
                h_curr,
            } => write!(
                f,
                "hydro '{hydro_name}': height decreases at index {index}: \
                 h[{index}]={h_curr} < h[{}]={h_prev}",
                index - 1
            ),
            Self::ConflictingFittingWindow { hydro_name, detail } => write!(
                f,
                "hydro '{hydro_name}': conflicting fitting window configuration: {detail}"
            ),
            Self::EmptyFittingWindow {
                hydro_name,
                v_min,
                v_max,
            } => write!(
                f,
                "hydro '{hydro_name}': fitting window is empty after resolution: \
                 v_min={v_min} >= v_max={v_max}"
            ),
            Self::InsufficientDiscretization {
                hydro_name,
                dimension,
                value,
            } => write!(
                f,
                "hydro '{hydro_name}': discretization count for '{dimension}' is {value}, \
                 which is below the minimum required"
            ),
            Self::InvalidKappa { hydro_name, kappa } => write!(
                f,
                "hydro '{hydro_name}': computed kappa {kappa} is outside the valid range (0, 1]; \
                 kappa must be strictly positive and at most 1.0"
            ),
            Self::NoHyperplanesProduced { hydro_name } => write!(
                f,
                "hydro '{hydro_name}': fitting pipeline produced zero valid hyperplanes; \
                 check that net head is positive over the fitting grid"
            ),
            Self::InvalidCoefficient {
                hydro_name,
                plane_index,
                detail,
            } => write!(
                f,
                "hydro '{hydro_name}': hyperplane {plane_index} has an invalid coefficient: \
                 {detail}"
            ),
        }
    }
}

impl std::error::Error for FphaFittingError {}
