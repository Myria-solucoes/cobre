//! Production-function model for the FPHA fitting pipeline.
//!
//! Owns the head-conversion constant [`K`] and [`ProductionFunction`], the
//! evaluable bundle of forebay table + tailrace + hydraulic-loss + efficiency that
//! computes `phi(v, q, s)`. The `hull_fit`, `grid`, `alpha`, and `secant`
//! submodules evaluate this model over the fitting grid.

use cobre_core::{EfficiencyModel, HydraulicLossesModel, TailraceModel};

use super::geometry::{ForebayTable, evaluate_losses, evaluate_tailrace};
use super::tailrace::TailraceFamilies;

// ── Tailrace source ───────────────────────────────────────────────────────────

/// Source of the tailrace elevation `h_jus(Q_jus)` the production function reads.
///
/// Enum dispatch (no `Box<dyn>`) over a closed two-variant set, so `net_head`
/// branches on the variant without a vtable. Built once per fit and held by
/// [`ProductionFunction`]; the grid loop never reconstructs it, keeping
/// `net_head` allocation-free.
///
/// # Contract — the `Entity` arm is the inert fallback (Voice 1)
///
/// A plant WITHOUT a `tailrace_curves` table resolves to [`TailraceSource::Entity`],
/// which reproduces the pre-families behavior EXACTLY: `net_head` evaluates the
/// same `evaluate_tailrace(model, q + s)` it always has. This bit-for-bit
/// equivalence is the load-bearing invariant — a plant with no table must fit to
/// the same planes as before the families path existed. Routing a table-less plant
/// through the `Families` arm (e.g. with an empty/synthetic family) would silently
/// perturb its fitted γ values and is forbidden.
#[derive(Debug, Clone)]
pub(crate) enum TailraceSource {
    /// The entity-level [`TailraceModel`]; `None` means constant zero tailrace.
    /// Preserves the exact pre-families `net_head` value for plants with no table.
    Entity(Option<TailraceModel>),
    /// The exact piecewise-quartic backwater-coupled families for a plant that has a
    /// `tailrace_curves` table. `downstream_level_m` is the resolved backwater
    /// level (`None` when unresolved — the evaluator then falls back to the
    /// lowest-keyed family).
    Families {
        /// The plant's ordered, validated tailrace families (built once per fit).
        families: TailraceFamilies,
        /// Resolved downstream reference level (m) keying the backwater coupling.
        downstream_level_m: Option<f64>,
    },
}

// ── ProductionFunction ────────────────────────────────────────────────────────

/// Gravity times water density over unit conversion: g·ρ / 1000.
///
/// Used in the production function `phi = K * eta * q * h_net` to convert
/// from hydraulic power (W) to megawatts. The factor is `9.81 * 1000 / 1e6 = 9.81e-3`.
const K: f64 = 9.81 / 1000.0;

/// Complete hydro production function `phi(v, q, s)` with analytical derivatives.
///
/// Bundles the forebay interpolation table with the optional tailrace and hydraulic
/// loss models and a constant turbine efficiency into a single evaluable object.
/// Evaluation produces power output in MW; the FPHA hull fitter samples it on the
/// `(V, Q)` grid to build the production cloud.
///
/// # Construction
///
/// Build from a validated [`ForebayTable`] and the optional model fields taken
/// directly from a hydro plant entity. All validation is done upstream; `new` is
/// infallible.
///
/// # Evaluation
///
/// Both public methods (`net_head`, `evaluate`) accept `(v, q, s)` where:
/// - `v` — reservoir volume \[hm³\]
/// - `q` — turbined flow \[m³/s\]
/// - `s` — spillage flow \[m³/s\]
#[derive(Debug, Clone)]
pub(crate) struct ProductionFunction {
    /// Forebay height interpolation table.
    forebay: ForebayTable,
    /// Tailrace elevation source. [`TailraceSource::Entity`] preserves the
    /// pre-families behavior; [`TailraceSource::Families`] couples the exact
    /// piecewise-quartic backwater families to a resolved downstream level.
    tailrace: TailraceSource,
    /// Hydraulic losses model. `None` means lossless penstock.
    hydraulic_losses: Option<HydraulicLossesModel>,
    /// Turbine efficiency (dimensionless, in `(0, 1]`). Defaults to `1.0` when the
    /// hydro entity has no `EfficiencyModel`.
    efficiency: f64,
    /// Maximum turbined flow \[m³/s\], carried for grid construction in the fitting
    /// algorithm.
    pub(crate) max_turbined_m3s: f64,
    /// Human-readable plant name for error messages.
    ///
    /// Retained for diagnostic use in integration tests.
    // Rationale: the field is populated and read in integration tests for diagnostic error
    // messages; production code constructs `ProductionFunction` with a name but never reads
    // it back — removing it would eliminate useful context from test failure output.
    #[allow(dead_code)]
    pub(crate) hydro_name: String,
}

impl ProductionFunction {
    /// Build a [`ProductionFunction`] from component models.
    ///
    /// # Parameters
    ///
    /// - `forebay` — pre-validated [`ForebayTable`] for this plant.
    /// - `tailrace` — the resolved [`TailraceSource`]. [`TailraceSource::Entity`]
    ///   carries the plant's [`TailraceModel`] (`None` = constant zero tailrace) and
    ///   preserves the pre-families behavior; [`TailraceSource::Families`] carries
    ///   the exact backwater families and the resolved downstream level.
    /// - `hydraulic_losses` — optional reference to the plant's [`HydraulicLossesModel`];
    ///   copied into the struct. `None` = lossless.
    /// - `efficiency` — optional reference to the plant's [`EfficiencyModel`]; only
    ///   [`EfficiencyModel::Constant`] is supported. `None` = 1.0 (100% efficiency).
    /// - `max_turbined_m3s` — maximum turbined flow from the hydro entity \[m³/s\].
    /// - `hydro_name` — plant name used in diagnostic messages.
    pub(crate) fn new(
        forebay: ForebayTable,
        tailrace: TailraceSource,
        hydraulic_losses: Option<&HydraulicLossesModel>,
        efficiency: Option<&EfficiencyModel>,
        max_turbined_m3s: f64,
        hydro_name: String,
    ) -> Self {
        let efficiency_value = match efficiency {
            Some(EfficiencyModel::Constant { value }) => *value,
            None => 1.0,
        };
        Self {
            forebay,
            tailrace,
            hydraulic_losses: hydraulic_losses.copied(),
            efficiency: efficiency_value,
            max_turbined_m3s,
            hydro_name,
        }
    }

    /// Net head available at the turbine \[m\].
    ///
    /// Computes `h_net = h_fore(v) - h_tail(q+s) - h_loss(gross_head, q)`, where:
    /// - `h_fore` is the interpolated forebay surface elevation,
    /// - `h_tail` is the tailrace elevation at total outflow `q + s` (0 if no model),
    /// - `h_loss` is the hydraulic head loss (0 if no model).
    ///
    /// For the [`HydraulicLossesModel::Factor`] variant, losses are proportional to
    /// gross head, which simplifies to `h_net = (1 - k) * (h_fore - h_tail)`.
    ///
    /// The result is clamped to `max(0.0, h_net)` — negative net head is physically
    /// impossible and arises only at out-of-range operating points.
    ///
    /// # Parameters
    ///
    /// - `v` — reservoir volume \[hm³\]
    /// - `q` — turbined flow \[m³/s\]
    /// - `s` — spillage flow \[m³/s\]
    pub(crate) fn net_head(&self, v: f64, q: f64, s: f64) -> f64 {
        let h_fore = self.forebay.height(v);
        // `q_out = q + s = Q_jus` is the total downstream flow the tailrace sees;
        // the secant's `s = Q_lat` sweep flows in through `s` unchanged, so both
        // source variants observe the identical `Q_jus`. The `Entity` arm MUST
        // call `evaluate_tailrace(model, q_out)` verbatim — the inert-fallback
        // contract on [`TailraceSource`] — so a table-less plant's net head is
        // bit-for-bit the pre-families value.
        let q_out = q + s;
        let h_tail = match &self.tailrace {
            TailraceSource::Entity(model) => {
                model.as_ref().map_or(0.0, |m| evaluate_tailrace(m, q_out))
            }
            TailraceSource::Families {
                families,
                downstream_level_m,
            } => families.evaluate(q_out, *downstream_level_m),
        };
        let gross_head = h_fore - h_tail;
        let h_loss = self
            .hydraulic_losses
            .as_ref()
            .map_or(0.0, |m| evaluate_losses(m, gross_head, q));
        let h_net = gross_head - h_loss;
        h_net.max(0.0)
    }

    /// Power output from the production function \[MW\].
    ///
    /// Evaluates `phi(v, q, s) = K * eta * q * h_net(v, q, s)` where
    /// `K = 9.81 / 1000` and `eta` is the turbine efficiency.
    ///
    /// The result is always non-negative because `q >= 0` and `h_net >= 0`.
    ///
    /// # Parameters
    ///
    /// - `v` — reservoir volume \[hm³\]
    /// - `q` — turbined flow \[m³/s\]
    /// - `s` — spillage flow \[m³/s\]
    pub(crate) fn evaluate(&self, v: f64, q: f64, s: f64) -> f64 {
        let h_net = self.net_head(v, q, s);
        K * self.efficiency * q * h_net
    }
}
