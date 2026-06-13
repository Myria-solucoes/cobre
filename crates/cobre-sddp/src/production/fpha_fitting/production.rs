//! Production-function model for the FPHA fitting pipeline.
//!
//! Owns the head-conversion constant [`K`] and [`ProductionFunction`], the
//! evaluable bundle of forebay table + tailrace + hydraulic-loss + efficiency that
//! computes `phi(v, q, s)` and its analytical partial derivatives. The `tangent`,
//! `grid`, and `selection` submodules evaluate this model over the fitting grid.

use cobre_core::{EfficiencyModel, HydraulicLossesModel, TailraceModel};

use super::geometry::{
    ForebayTable, evaluate_losses, evaluate_tailrace, evaluate_tailrace_derivative,
};

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
/// Evaluation produces power output in MW; derivatives are used by the FPHA fitting
/// algorithm to compute tangent hyperplanes.
///
/// # Construction
///
/// Build from a validated [`ForebayTable`] and the optional model fields taken
/// directly from a hydro plant entity. All validation is done upstream; `new` is
/// infallible.
///
/// # Evaluation
///
/// All three public methods (`net_head`, `evaluate`, `partial_derivatives`) accept
/// `(v, q, s)` where:
/// - `v` — reservoir volume \[hm³\]
/// - `q` — turbined flow \[m³/s\]
/// - `s` — spillage flow \[m³/s\]
#[derive(Debug, Clone)]
pub(crate) struct ProductionFunction {
    /// Forebay height interpolation table.
    forebay: ForebayTable,
    /// Tailrace elevation model. `None` means zero tailrace height for all outflows.
    tailrace: Option<TailraceModel>,
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
    /// - `tailrace` — optional reference to the plant's [`TailraceModel`]; cloned
    ///   into the struct. `None` = constant zero tailrace.
    /// - `hydraulic_losses` — optional reference to the plant's [`HydraulicLossesModel`];
    ///   copied into the struct. `None` = lossless.
    /// - `efficiency` — optional reference to the plant's [`EfficiencyModel`]; only
    ///   [`EfficiencyModel::Constant`] is supported. `None` = 1.0 (100% efficiency).
    /// - `max_turbined_m3s` — maximum turbined flow from the hydro entity \[m³/s\].
    /// - `hydro_name` — plant name used in diagnostic messages.
    pub(crate) fn new(
        forebay: ForebayTable,
        tailrace: Option<&TailraceModel>,
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
            tailrace: tailrace.cloned(),
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
        let q_out = q + s;
        let h_tail = self
            .tailrace
            .as_ref()
            .map_or(0.0, |m| evaluate_tailrace(m, q_out));
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

    /// Analytical partial derivatives of the production function.
    ///
    /// Returns `(d_phi/dv, d_phi/dq, d_phi/ds)` evaluated at `(v, q, s)`.
    ///
    /// The derivative formulas depend on the loss model:
    ///
    /// **Constant losses or no losses** (`h_net = h_fore - h_tail - c`):
    /// ```text
    /// d_phi/dv = K·eta·q·dh_fore/dv
    /// d_phi/dq = K·eta·(h_net - q·dh_tail/dq_out)
    /// d_phi/ds = -K·eta·q·dh_tail/dq_out
    /// ```
    ///
    /// **Factor losses** (`h_net = (1-k)·(h_fore - h_tail)`):
    /// ```text
    /// d_phi/dv = K·eta·q·(1-k)·dh_fore/dv
    /// d_phi/dq = K·eta·(h_net - q·(1-k)·dh_tail/dq_out)
    /// d_phi/ds = -K·eta·q·(1-k)·dh_tail/dq_out
    /// ```
    ///
    /// # Sign conventions
    ///
    /// - `d_phi/dv > 0`: more storage raises forebay, increasing net head and power.
    /// - `d_phi/dq > 0` when net head is positive (turbining produces power).
    /// - `d_phi/ds <= 0`: spillage raises tailrace, reducing net head.
    ///   Equals zero when there is no tailrace model.
    ///
    /// # Parameters
    ///
    /// - `v` — reservoir volume \[hm³\]
    /// - `q` — turbined flow \[m³/s\]
    /// - `s` — spillage flow \[m³/s\]
    #[allow(clippy::similar_names)] // d_phi_dv / d_phi_dq / d_phi_ds are standard PDE notation
    pub(crate) fn partial_derivatives(&self, v: f64, q: f64, s: f64) -> (f64, f64, f64) {
        let h_fore = self.forebay.height(v);
        let dh_fore_dv = self.forebay.height_derivative(v);
        let q_out = q + s;

        let h_tail = self
            .tailrace
            .as_ref()
            .map_or(0.0, |m| evaluate_tailrace(m, q_out));
        let dh_tail_dq_out = self
            .tailrace
            .as_ref()
            .map_or(0.0, |m| evaluate_tailrace_derivative(m, q_out));

        let ke = K * self.efficiency;

        match self.hydraulic_losses {
            Some(HydraulicLossesModel::Factor { value: k_loss }) => {
                // h_net = (1 - k_loss) * (h_fore - h_tail)
                let one_minus_k = 1.0 - k_loss;
                let h_net = (one_minus_k * (h_fore - h_tail)).max(0.0);
                let d_phi_dv = ke * q * one_minus_k * dh_fore_dv;
                let d_phi_dq = ke * (h_net - q * one_minus_k * dh_tail_dq_out);
                let d_phi_ds = -ke * q * one_minus_k * dh_tail_dq_out;
                (d_phi_dv, d_phi_dq, d_phi_ds)
            }
            Some(HydraulicLossesModel::Constant { .. }) | None => {
                // h_net = h_fore - h_tail - h_loss_const   (h_loss_const may be 0)
                let h_net = self.net_head(v, q, s);
                let d_phi_dv = ke * q * dh_fore_dv;
                let d_phi_dq = ke * (h_net - q * dh_tail_dq_out);
                let d_phi_ds = -ke * q * dh_tail_dq_out;
                (d_phi_dv, d_phi_dq, d_phi_ds)
            }
        }
    }
}
