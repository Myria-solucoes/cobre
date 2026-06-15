//! Reservoir-geometry layer of the FPHA fitting pipeline.
//!
//! Owns the resolved fitting bounds ([`FittingBounds`] + [`resolve_fitting_bounds`]),
//! the forebay interpolation table ([`ForebayTable`]), and the tailrace / hydraulic-loss
//! evaluators ([`evaluate_tailrace`], [`evaluate_losses`]). The `production` submodule
//! reads these to assemble the complete production function.

use cobre_core::{HydraulicLossesModel, Hydro, TailraceModel};
use cobre_io::extensions::{FphaColumnLayout, HydroGeometryRow};

use super::error::FphaFittingError;

// ── FittingBounds ─────────────────────────────────────────────────────────────

/// Resolved volume range and discretization counts for the FPHA fitting grid.
///
/// Produced by [`resolve_fitting_bounds`] from an [`FphaColumnLayout`] and the hydro
/// plant entity. Consumed by the FPHA fitting grid construction step.
#[derive(Debug)]
pub(crate) struct FittingBounds {
    /// Resolved lower bound of the fitting volume range (hm³).
    pub v_min: f64,
    /// Resolved upper bound of the fitting volume range (hm³).
    pub v_max: f64,
    /// Number of volume grid points (>= 2).
    pub n_volume_points: usize,
    /// Number of turbined-flow grid points (>= 2).
    pub n_flow_points: usize,
    /// Number of spillage grid points (>= 2). Validated but not a fitting axis:
    /// the cloud and α regression fix `s = 0` and the lateral secant sweeps its
    /// own `S_max` sample, so no step builds a spillage grid axis.
    // Intent/Seam: resolved and range-validated in `resolve_fitting_bounds` so the
    // `spillage_discretization_points` config field round-trips, but read by no
    // fitting step now that spillage is not a grid axis. The allow re-fires if a
    // spillage-axis consumer is reintroduced.
    #[allow(dead_code)]
    pub n_spillage_points: usize,
    /// Parsed plane-count ceiling (>= 1). No longer drives fitting: the hull fitter
    /// emits exactly the upper-envelope facets and there is no greedy selector to
    /// cap. Plane trimming is a separate post-fit pass; the field is resolved and
    /// validated so the config still deserializes and a later trimming pass can
    /// repurpose it.
    // Intent/Seam: resolved from the config and range-validated in
    // `resolve_fitting_bounds`, but read by no fitting step now that plane
    // selection is retired. Kept so the config field round-trips and a future
    // trimming pass can consume it; the allow re-fires when that reader lands.
    #[allow(dead_code)]
    pub max_planes_per_hydro: usize,
    /// Run-of-river marker: the requested volume range collapsed to a single
    /// point, so `[v_min, v_max]` are the two synthesized tangent-in-volume
    /// samples (see [`resolve_fitting_bounds`]) rather than a genuine storage
    /// window. The hull fitter reads this to zero `γ_V` exactly on the emitted
    /// planes — generation does not vary with storage for a run-of-river plant.
    pub single_volume: bool,
}

/// Absolute single-point detection tolerance for the volume range (hm³).
///
/// A resolved range narrower than this is treated as a single point (a
/// run-of-river plant with no useful storage) and rerouted through the
/// single-volume fitting path, NOT rejected as an empty window. A genuinely
/// inverted range (`v_max < v_min`) is still an error: the reroute fires only
/// for `0 <= v_max - v_min <= V_EPS`.
const V_EPS: f64 = 1e-9;

/// Absolute fallback half-width for the synthesized volume samples (hm³).
///
/// When the entity's useful storage `ΔV_useful` is itself zero (the run-of-river
/// case), the `0.005·ΔV_useful` tangent half-width collapses to nothing, so the
/// two synthesized samples would coincide and the cloud would stay coplanar in
/// `V`. This absolute floor opens the `V` axis just enough for a non-degenerate
/// 3-D hull; `γ_V` is zeroed afterward, so its exact magnitude does not reach
/// the emitted planes.
const V_SYNTH_ABS: f64 = 1.0;

/// Resolve the fitting volume range and discretization counts from configuration.
///
/// Combines the [`FphaColumnLayout`] fitting window (if any), the hydro entity's
/// operating limits, and the forebay table's interpolation range to produce
/// a concrete [`FittingBounds`] for grid construction.
///
/// # Volume range resolution
///
/// The volume range is resolved in three mutually exclusive modes:
///
/// 1. **No fitting window** (`config.fitting_window` is `None`): use the full
///    forebay table range `[forebay.v_min(), forebay.v_max()]`.
/// 2. **Absolute bounds**: use `volume_min_hm3` / `volume_max_hm3` directly,
///    clamping to the forebay table range.
/// 3. **Percentile bounds**: compute
///    `v = entity_v_min + p * (entity_v_max - entity_v_min)` for each bound,
///    then clamp to the forebay table range.
///
/// Mixed modes (absolute min, percentile max or vice versa) are accepted as long
/// as neither the min bound nor the max bound has both absolute and percentile set.
///
/// # Run-of-river reroute
///
/// A resolved range that collapses to a single point
/// (`0 <= v_max - v_min <= V_EPS`), or a config requesting `NPTV = 1`, is a
/// run-of-river plant with no useful storage. Rather than rejecting it, the
/// function synthesizes two tangent-in-volume samples `1%` of the entity's
/// useful storage apart (clamped to the forebay table range; an absolute floor
/// when the useful storage is itself 0), sets `n_volume_points = 2`, and flags
/// `single_volume = true` so the hull fitter zeros `γ_V` exactly. The
/// multi-volume discretization-count guard on `n_volume_points` does NOT apply
/// to this path.
///
/// # Errors
///
/// | Condition | Error variant |
/// |-----------|---------------|
/// | Both absolute and percentile set for the same bound (min or max) | [`FphaFittingError::ConflictingFittingWindow`] |
/// | Resolved `v_max < v_min` (inverted range, not single-point) | [`FphaFittingError::EmptyFittingWindow`] |
/// | A turbine/spillage count < 2, `max_planes_per_hydro` < 1, or (multi-volume only) `n_volume_points` < 2 | [`FphaFittingError::InsufficientDiscretization`] |
pub(crate) fn resolve_fitting_bounds(
    config: &FphaColumnLayout,
    hydro: &Hydro,
    forebay: &ForebayTable,
) -> Result<FittingBounds, FphaFittingError> {
    let hydro_name = &hydro.name;

    // ── Step 1: Resolve volume range ─────────────────────────────────────────

    let (v_min, v_max) = match &config.fitting_window {
        None => (forebay.v_min(), forebay.v_max()),
        Some(fw) => {
            // Check for conflicts on the min bound.
            if fw.volume_min_hm3.is_some() && fw.volume_min_percentile.is_some() {
                return Err(FphaFittingError::ConflictingFittingWindow {
                    hydro_name: hydro_name.clone(),
                    detail: "volume_min_hm3 and volume_min_percentile cannot both be set; \
                             use absolute bounds OR percentiles, not both for the same bound"
                        .to_owned(),
                });
            }
            // Check for conflicts on the max bound.
            if fw.volume_max_hm3.is_some() && fw.volume_max_percentile.is_some() {
                return Err(FphaFittingError::ConflictingFittingWindow {
                    hydro_name: hydro_name.clone(),
                    detail: "volume_max_hm3 and volume_max_percentile cannot both be set; \
                             use absolute bounds OR percentiles, not both for the same bound"
                        .to_owned(),
                });
            }

            let entity_v_min = hydro.min_storage_hm3;
            let entity_v_max = hydro.max_storage_hm3;
            let entity_range = entity_v_max - entity_v_min;

            // Resolve lower bound.
            let v_min_raw = if let Some(abs) = fw.volume_min_hm3 {
                abs
            } else if let Some(pct) = fw.volume_min_percentile {
                entity_v_min + pct * entity_range
            } else {
                forebay.v_min()
            };

            // Resolve upper bound.
            let v_max_raw = if let Some(abs) = fw.volume_max_hm3 {
                abs
            } else if let Some(pct) = fw.volume_max_percentile {
                entity_v_min + pct * entity_range
            } else {
                forebay.v_max()
            };

            // Clamp to forebay table range.
            let v_min = v_min_raw.clamp(forebay.v_min(), forebay.v_max());
            let v_max = v_max_raw.clamp(forebay.v_min(), forebay.v_max());

            (v_min, v_max)
        }
    };

    // ── Step 2: Validate volume range ────────────────────────────────────────

    // A genuinely inverted range is always an error. The single-point case
    // (`0 <= v_max - v_min <= V_EPS`) is NOT inverted — it is rerouted below,
    // so this guard fires only for a strictly negative range.
    if v_max < v_min {
        return Err(FphaFittingError::EmptyFittingWindow {
            hydro_name: hydro_name.clone(),
            v_min,
            v_max,
        });
    }

    // ── Step 3: Resolve discretization counts ────────────────────────────────

    #[allow(clippy::cast_sign_loss)]
    let requested_volume_points = config.volume_discretization_points.unwrap_or(5) as usize;
    #[allow(clippy::cast_sign_loss)]
    let n_flow_points = config.turbine_discretization_points.unwrap_or(5) as usize;
    #[allow(clippy::cast_sign_loss)]
    let n_spillage_points = config.spillage_discretization_points.unwrap_or(5) as usize;
    #[allow(clippy::cast_sign_loss)]
    let max_planes = config.max_planes_per_hydro.unwrap_or(10) as usize;

    // ── Step 4: Detect and reroute the single-volume (run-of-river) case ──────
    //
    // A run-of-river plant has no useful storage: its resolved volume range
    // collapses to a single point (`v_max - v_min <= V_EPS`), or the config
    // explicitly requests `NPTV = 1`. Either way the production cloud is
    // coplanar in `V`, so the 3-D hull cannot be posed and the multi-volume
    // discretization-count guards (`n_volume_points >= 2`) are inapplicable.
    // Instead of erroring, synthesize two tangent-in-volume samples `1%` of the
    // entity's useful storage apart, clamped to the forebay table range, and
    // flag the bounds so the hull fitter zeros `γ_V` exactly on the result.
    let single_volume = (v_max - v_min) <= V_EPS || requested_volume_points == 1;
    if single_volume {
        let v0 = v_min;
        // ΔV_useful is the entity's storage band; for a true run-of-river plant
        // it is itself 0, so the proportional half-width collapses and the
        // absolute floor `V_SYNTH_ABS` takes over.
        let useful = (hydro.max_storage_hm3 - hydro.min_storage_hm3).max(0.0);
        let half_width = (0.005 * useful).max(V_SYNTH_ABS);
        // Spread two tangent-in-volume samples around `v0`. When the forebay table
        // itself spans a range, clamp to it. But when the table is a SINGLE POINT
        // (`v_min == v_max`, the genuine run-of-river case), clamping to `[v0, v0]`
        // would collapse `v_lo == v_hi == v0` — a coplanar cloud the hull rejects
        // as degenerate. `height()` clamps queries internally, so spreading beyond
        // the degenerate forebay range is safe: both samples evaluate to the same
        // constant elevation, giving a GH cloud that is flat in V and yields
        // `γ_V = 0` exactly. Only clamp when the forebay actually spans a range.
        let (v_lo, v_hi) = if (forebay.v_max() - forebay.v_min()) <= V_EPS {
            (v0 - half_width, v0 + half_width)
        } else {
            (
                (v0 - half_width).clamp(forebay.v_min(), forebay.v_max()),
                (v0 + half_width).clamp(forebay.v_min(), forebay.v_max()),
            )
        };

        if n_flow_points < 2 {
            return Err(FphaFittingError::InsufficientDiscretization {
                hydro_name: hydro_name.clone(),
                dimension: "turbine".to_owned(),
                value: n_flow_points,
            });
        }
        if n_spillage_points < 2 {
            return Err(FphaFittingError::InsufficientDiscretization {
                hydro_name: hydro_name.clone(),
                dimension: "spillage".to_owned(),
                value: n_spillage_points,
            });
        }
        if max_planes < 1 {
            return Err(FphaFittingError::InsufficientDiscretization {
                hydro_name: hydro_name.clone(),
                dimension: "max_planes_per_hydro".to_owned(),
                value: max_planes,
            });
        }

        return Ok(FittingBounds {
            v_min: v_lo,
            v_max: v_hi,
            n_volume_points: 2,
            n_flow_points,
            n_spillage_points,
            max_planes_per_hydro: max_planes,
            single_volume: true,
        });
    }

    let n_volume_points = requested_volume_points;
    if n_volume_points < 2 {
        return Err(FphaFittingError::InsufficientDiscretization {
            hydro_name: hydro_name.clone(),
            dimension: "volume".to_owned(),
            value: n_volume_points,
        });
    }
    if n_flow_points < 2 {
        return Err(FphaFittingError::InsufficientDiscretization {
            hydro_name: hydro_name.clone(),
            dimension: "turbine".to_owned(),
            value: n_flow_points,
        });
    }
    if n_spillage_points < 2 {
        return Err(FphaFittingError::InsufficientDiscretization {
            hydro_name: hydro_name.clone(),
            dimension: "spillage".to_owned(),
            value: n_spillage_points,
        });
    }
    if max_planes < 1 {
        return Err(FphaFittingError::InsufficientDiscretization {
            hydro_name: hydro_name.clone(),
            dimension: "max_planes_per_hydro".to_owned(),
            value: max_planes,
        });
    }

    Ok(FittingBounds {
        v_min,
        v_max,
        n_volume_points,
        n_flow_points,
        n_spillage_points,
        max_planes_per_hydro: max_planes,
        single_volume: false,
    })
}

// ── ForebayTable ──────────────────────────────────────────────────────────────

/// Linear interpolation table for forebay height `h_fore(v)`.
///
/// Stores the VHA curve for a single hydro plant as two parallel sorted vectors
/// of volume breakpoints (`volumes`, hm³) and corresponding surface elevations
/// (`heights`, m). All queries are clamped to `[v_min, v_max]`, so the table
/// never extrapolates and every method is infallible after construction.
///
/// # Construction
///
/// Build from a slice of [`HydroGeometryRow`] values (all rows for one hydro,
/// already sorted by ascending `volume_hm3` by the parser):
///
/// ```no_run
/// use cobre_io::extensions::HydroGeometryRow;
/// use cobre_core::EntityId;
///
/// // (ForebayTable is pub(crate); this example is for illustration only.)
/// let rows = vec![
///     HydroGeometryRow { hydro_id: EntityId::from(1), volume_hm3: 0.0,    height_m: 386.5, area_km2: 2.5 },
///     HydroGeometryRow { hydro_id: EntityId::from(1), volume_hm3: 2000.0, height_m: 390.0, area_km2: 3.1 },
/// ];
/// ```
#[derive(Debug, Clone)]
pub(crate) struct ForebayTable {
    /// Volume breakpoints (hm³), strictly increasing.
    volumes: Vec<f64>,
    /// Surface elevation breakpoints (m), monotonically non-decreasing.
    heights: Vec<f64>,
}

impl ForebayTable {
    /// Build a [`ForebayTable`] from a slice of VHA curve rows for one hydro plant.
    ///
    /// # Parameters
    ///
    /// - `rows` — all [`HydroGeometryRow`] entries for the hydro plant, sorted
    ///   by ascending `volume_hm3` (as returned by `cobre_io::extensions::parse_hydro_geometry`).
    /// - `hydro_name` — human-readable plant name used in error messages.
    ///
    /// A single row is accepted: a run-of-river plant has one operating volume
    /// (`v_min == v_max`), so its VHA curve is one row defining a CONSTANT forebay
    /// (`height()` returns that row's elevation for any query). Rejecting it here
    /// would make the single-volume FPHA fit (`resolve_fitting_bounds` →
    /// `single_volume`) unreachable end-to-end. Only an empty table is an error.
    ///
    /// # Errors
    ///
    /// | Condition | Error variant |
    /// |-----------|---------------|
    /// | Empty (zero rows) | [`FphaFittingError::InsufficientPoints`] |
    /// | `volume_hm3` not strictly increasing | [`FphaFittingError::NonMonotonicVolume`] |
    /// | `height_m` decreasing | [`FphaFittingError::NonMonotonicHeight`] |
    pub(crate) fn new(
        rows: &[HydroGeometryRow],
        hydro_name: &str,
    ) -> Result<Self, FphaFittingError> {
        // A single point defines a trivial (constant) forebay — valid for a
        // run-of-river plant. Only the empty table has no curve to evaluate.
        if rows.is_empty() {
            return Err(FphaFittingError::InsufficientPoints {
                hydro_name: hydro_name.to_owned(),
                count: rows.len(),
            });
        }

        let mut volumes = Vec::with_capacity(rows.len());
        let mut heights = Vec::with_capacity(rows.len());

        volumes.push(rows[0].volume_hm3);
        heights.push(rows[0].height_m);

        for i in 1..rows.len() {
            let v_prev = rows[i - 1].volume_hm3;
            let v_curr = rows[i].volume_hm3;
            let h_prev = rows[i - 1].height_m;
            let h_curr = rows[i].height_m;

            // Volumes must be strictly increasing.
            if v_curr <= v_prev {
                return Err(FphaFittingError::NonMonotonicVolume {
                    hydro_name: hydro_name.to_owned(),
                    index: i,
                    v_prev,
                    v_curr,
                });
            }

            // Heights must be non-decreasing.
            if h_curr < h_prev {
                return Err(FphaFittingError::NonMonotonicHeight {
                    hydro_name: hydro_name.to_owned(),
                    index: i,
                    h_prev,
                    h_curr,
                });
            }

            volumes.push(v_curr);
            heights.push(h_curr);
        }

        Ok(Self { volumes, heights })
    }

    /// Minimum volume in the table (hm³).
    #[inline]
    pub(crate) fn v_min(&self) -> f64 {
        // INVARIANT: `volumes` is non-empty (enforced by `new`). For a single-row
        // run-of-river table `v_min == v_max == volumes[0]`.
        self.volumes[0]
    }

    /// Maximum volume in the table (hm³).
    #[inline]
    pub(crate) fn v_max(&self) -> f64 {
        // INVARIANT: `volumes` is non-empty (enforced by `new`). For a single-row
        // run-of-river table `v_max == v_min == volumes[0]`.
        self.volumes[self.volumes.len() - 1]
    }

    /// Interpolated forebay surface elevation at `volume_hm3` (m).
    ///
    /// The query volume is clamped to `[v_min, v_max]` before interpolation,
    /// so this method is infallible and never extrapolates. Values at exact
    /// breakpoints are returned without rounding error.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cobre_io::extensions::HydroGeometryRow;
    /// use cobre_core::EntityId;
    ///
    /// // (ForebayTable is pub(crate); this example is for illustration only.)
    /// // let table = ForebayTable::new(&rows, "Sobradinho").unwrap();
    /// // assert!((table.height(1000.0) - 388.25).abs() < 1e-10);
    /// ```
    pub(crate) fn height(&self, volume_hm3: f64) -> f64 {
        // Single-row (run-of-river) table: a constant forebay with no segment to
        // interpolate. Returning `heights[0]` directly is REQUIRED — `locate`
        // computes `n - 2`, which underflows for `n == 1` and then indexes
        // `volumes[i + 1]` out of bounds.
        if self.volumes.len() == 1 {
            return self.heights[0];
        }
        let v = volume_hm3.clamp(self.v_min(), self.v_max());
        let (i, t) = self.locate(v);
        self.heights[i] + t * (self.heights[i + 1] - self.heights[i])
    }

    /// Find the segment index `i` and the fractional position `t` within it.
    ///
    /// Returns `(i, t)` such that:
    /// - `self.volumes[i] <= v <= self.volumes[i + 1]`
    /// - `t = (v - self.volumes[i]) / (self.volumes[i + 1] - self.volumes[i])`
    ///
    /// The caller must ensure `v` is already clamped to `[v_min, v_max]`.
    ///
    /// Uses `partition_point` (binary search) for O(log n) lookup.
    ///
    /// At `v_max` the last segment is returned (index `n - 2`) to avoid an
    /// out-of-bounds access on `i + 1`.
    fn locate(&self, v: f64) -> (usize, f64) {
        let n = self.volumes.len();
        // `partition_point` returns the first index where `volumes[idx] > v`.
        // Subtract 1 to get the left bracket. Saturate at `n - 2` so that
        // `i + 1` is always a valid index (handles the v == v_max case).
        let idx = self.volumes.partition_point(|&vk| vk <= v);
        let i = idx.saturating_sub(1).min(n - 2);
        let dv = self.volumes[i + 1] - self.volumes[i];
        let t = (v - self.volumes[i]) / dv;
        (i, t)
    }
}

// ── Tailrace and hydraulic loss evaluation ────────────────────────────────────

/// Tailrace elevation `h_tail(q_out)` for a total outflow of `outflow_m3s` (m).
///
/// - `Polynomial`: evaluates `c[0] + c[1]·q + c[2]·q² + …` via Horner's method.
/// - `Piecewise`: linearly interpolates between adjacent [`cobre_core::TailracePoint`]
///   breakpoints; the outflow is clamped to the table's range before lookup.
///
/// The function is infallible — the model invariants (≥ 1 coefficient; ≥ 2 points
/// sorted ascending) are enforced by the `cobre-io` parsing layer.
pub(crate) fn evaluate_tailrace(model: &TailraceModel, outflow_m3s: f64) -> f64 {
    match model {
        TailraceModel::Polynomial { coefficients } => {
            // Horner's method: evaluate from the highest-degree coefficient down.
            // For an empty slice (should not happen after IO validation) return 0.
            coefficients
                .iter()
                .rev()
                .fold(0.0_f64, |acc, c| acc * outflow_m3s + c)
        }
        TailraceModel::Piecewise { points } => {
            let n = points.len();
            if n == 0 {
                return 0.0;
            }
            if n == 1 {
                return points[0].height_m;
            }
            // Clamp outflow to [q_min, q_max].
            let q = outflow_m3s.clamp(points[0].outflow_m3s, points[n - 1].outflow_m3s);
            let (i, t) = locate_tailrace(points, q);
            points[i].height_m + t * (points[i + 1].height_m - points[i].height_m)
        }
    }
}

/// Head loss `h_loss` (m) for the given `gross_head` (m) and `turbined_m3s` (m³/s).
///
/// - `Factor { value }`: returns `value * gross_head` (fraction of gross head).
/// - `Constant { value_m }`: returns the fixed head loss; `gross_head` and
///   `turbined_m3s` are unused.
///
/// The `turbined_m3s` parameter is reserved for future flow-dependent loss variants
/// and is intentionally ignored for both current variants.
pub(crate) fn evaluate_losses(
    model: &HydraulicLossesModel,
    gross_head: f64,
    _turbined_m3s: f64,
) -> f64 {
    match model {
        HydraulicLossesModel::Factor { value } => value * gross_head,
        HydraulicLossesModel::Constant { value_m } => *value_m,
    }
}

/// Find segment index `i` and fractional position `t` in a piecewise tailrace table.
///
/// Returns `(i, t)` such that:
/// - `points[i].outflow_m3s <= q <= points[i+1].outflow_m3s`
/// - `t = (q - outflow[i]) / (outflow[i+1] - outflow[i])`
///
/// The caller must ensure `q` is already clamped to `[q_min, q_max]`. Uses
/// `partition_point` for O(log n) binary search; saturates at `n - 2` to keep
/// `i + 1` in bounds at `q == q_max`.
fn locate_tailrace(points: &[cobre_core::TailracePoint], q: f64) -> (usize, f64) {
    let n = points.len();
    let idx = points.partition_point(|p| p.outflow_m3s <= q);
    let i = idx.saturating_sub(1).min(n - 2);
    let dq = points[i + 1].outflow_m3s - points[i].outflow_m3s;
    let t = (q - points[i].outflow_m3s) / dq;
    (i, t)
}
