//! FPHA fitting error type.
//!
//! Owns [`FphaFittingError`], the validation-error enum returned by every
//! fallible step of the fitting pipeline (geometry-table construction, bounds
//! resolution, and the `α_FPHA > 0` / coefficient-sign validation).

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that arise during FPHA fitting geometry validation or evaluation.
///
/// Returned by [`ForebayTable::new`](super::geometry::ForebayTable::new) when the
/// supplied VHA curve data does not satisfy the invariants required for linear
/// interpolation.
#[derive(Debug)]
pub(crate) enum FphaFittingError {
    /// No VHA curve points were provided for the named hydro plant.
    ///
    /// An empty table has no curve to evaluate. A single point IS accepted — it
    /// defines a constant (run-of-river) forebay; the single-volume FPHA fit then
    /// yields `γ_V = 0`. Only the zero-row case reaches this variant.
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

    /// The resolved volume range is inverted (`v_max < v_min`).
    ///
    /// After applying the fitting window configuration, the upper bound was
    /// strictly below the lower bound — only inverted absolute or percentile
    /// bounds reach this variant. A range that collapses to a single point
    /// (`v_min == v_max`) is NOT an error: it is a run-of-river plant and is
    /// rerouted through the single-volume fitting path
    /// (see [`resolve_fitting_bounds`](super::geometry::resolve_fitting_bounds)).
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
    /// The flow and spillage counts (`n_flow_points`, `n_spillage_points`) must
    /// be >= 2, and `max_planes_per_hydro` must be >= 1. `n_volume_points` must
    /// be >= 2 on the multi-volume path only — the run-of-river single-volume
    /// path synthesizes its own two samples and is exempt (see
    /// [`resolve_fitting_bounds`](super::geometry::resolve_fitting_bounds)).
    InsufficientDiscretization {
        /// Name of the hydro plant whose configuration was rejected.
        hydro_name: String,
        /// Which dimension was too small: `"volume"`, `"turbine"`, `"spillage"`,
        /// or `"max_planes_per_hydro"`.
        dimension: String,
        /// The value that was provided (< 2 for grid dimensions, < 1 for max planes).
        value: usize,
    },

    /// The least-squares `α_FPHA` correction factor is not strictly positive.
    ///
    /// `α_FPHA` balances the raw hull envelope against the exact production
    /// function; a non-positive `α` would flip every coefficient sign or collapse
    /// the envelope to zero, both physically invalid.
    NonPositiveAlpha {
        /// Name of the hydro plant whose fitting was rejected.
        hydro_name: String,
        /// The `α_FPHA` value that was computed.
        alpha: f64,
    },

    /// The fitting pipeline produced zero valid hyperplanes.
    ///
    /// This can occur when every grid point has zero or negative production
    /// (e.g., net head ≤ 0 everywhere), so the hull yields no upper-envelope facet.
    NoHyperplanesProduced {
        /// Name of the hydro plant for which no hyperplanes were produced.
        hydro_name: String,
    },

    /// The 3-D production cloud was too degenerate for a convex-hull fit.
    ///
    /// The hull primitive needs at least four affinely-independent points to
    /// build a full-dimensional 3-D hull. A production function whose `(V, Q, generation)`
    /// cloud collapses onto a single line or plane (e.g. a constant net head with
    /// no V- or Q-dependence, so every grid point and the closing point are
    /// collinear) cannot yield even one upper-envelope facet from the hull. The
    /// caller maps the hull's degenerate status to this variant rather than
    /// panicking, so one pathological hydro does not abort the whole fitting loop.
    DegenerateProductionCloud {
        /// Name of the hydro plant whose production cloud was degenerate.
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

    /// Consecutive tailrace segments do not tile the outflow domain.
    ///
    /// A family's quartic segments must partition `outflow` without gaps or
    /// overlaps: each segment's lower bound must meet the previous segment's
    /// upper bound. A gap leaves `outflow` values with no owning segment; an
    /// overlap makes the owning segment ambiguous. Both are rejected here.
    TailraceGap {
        /// Name of the hydro plant whose tailrace family was rejected.
        hydro_name: String,
        /// Upper bound of the lower-indexed segment (m³/s).
        outflow_max_prev: f64,
        /// Lower bound of the higher-indexed segment (m³/s), which must equal
        /// `outflow_max_prev` within tolerance.
        outflow_min_curr: f64,
    },

    /// Consecutive tailrace segments disagree at their shared boundary.
    ///
    /// The piecewise quartic must be continuous (C0): the lower-indexed segment
    /// and the higher-indexed segment must evaluate to the same tailrace
    /// elevation at the boundary they share. A jump indicates miscalibrated
    /// coefficients that would make the within-family evaluator discontinuous.
    TailraceDiscontinuity {
        /// Name of the hydro plant whose tailrace family was rejected.
        hydro_name: String,
        /// Shared boundary outflow at which the two segments are evaluated (m³/s).
        boundary: f64,
        /// Tailrace elevation from the lower-indexed segment at `boundary` (m).
        h_left: f64,
        /// Tailrace elevation from the higher-indexed segment at `boundary` (m).
        h_right: f64,
    },

    /// A multi-family tailrace table has a family with no downstream reference level.
    ///
    /// When a plant has more than one tailrace family, every family must carry a
    /// downstream reference level so the families can be ordered and bracketed by
    /// that level. A `None` level is only admissible for a plant with exactly one
    /// family (the level argument is then ignored). A multi-family table with any
    /// keyless family is ambiguous — it cannot be bracketed — and is rejected
    /// here rather than silently picking one family. The owning check is
    /// [`TailraceFamilies::from_rows`](super::tailrace::TailraceFamilies::from_rows).
    TailraceFamilyKeyMissing {
        /// Name of the hydro plant whose tailrace family table was rejected.
        hydro_name: String,
        /// Number of families found for the plant (> 1 when this fires).
        family_count: usize,
    },
}

impl std::fmt::Display for FphaFittingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientPoints { hydro_name, count } => write!(
                f,
                "hydro '{hydro_name}': VHA curve has {count} point(s); \
                 at least 1 is required (a single point defines a constant forebay)"
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
            Self::NonPositiveAlpha { hydro_name, alpha } => write!(
                f,
                "hydro '{hydro_name}': least-squares alpha_FPHA {alpha} is not strictly positive; \
                 alpha_FPHA must be > 0"
            ),
            Self::DegenerateProductionCloud { hydro_name } => write!(
                f,
                "hydro '{hydro_name}': production cloud is degenerate (collinear or coplanar); \
                 a 3-D convex hull needs at least 4 affinely-independent points"
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
            Self::TailraceGap {
                hydro_name,
                outflow_max_prev,
                outflow_min_curr,
            } => write!(
                f,
                "hydro '{hydro_name}': tailrace segments leave a gap or overlap at the \
                 boundary: previous outflow_max={outflow_max_prev} does not meet next outflow_min={outflow_min_curr}"
            ),
            Self::TailraceDiscontinuity {
                hydro_name,
                boundary,
                h_left,
                h_right,
            } => write!(
                f,
                "hydro '{hydro_name}': tailrace segments are discontinuous at q={boundary}: \
                 left tailrace_level={h_left} != right tailrace_level={h_right}"
            ),
            Self::TailraceFamilyKeyMissing {
                hydro_name,
                family_count,
            } => write!(
                f,
                "hydro '{hydro_name}': tailrace table has {family_count} families but at least \
                 one carries no downstream reference level (downstream_reference_level_m); a keyless family is \
                 only valid for a single-family plant"
            ),
        }
    }
}

impl std::error::Error for FphaFittingError {}
