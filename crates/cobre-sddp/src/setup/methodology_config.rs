//! [`MethodologyConfig`] — stochastic numerical methodology parameters.

use crate::{horizon_mode::HorizonMode, inflow_method::InflowNonNegativityMethod};

/// Stochastic numerical methodology parameters stored on [`super::StudySetup`].
///
/// Explicit construction only — no `Default` impl, to prevent silent
/// misconfiguration.
#[derive(Debug)]
pub(crate) struct MethodologyConfig {
    /// Study horizon mode (finite vs. infinite-horizon approximation).
    pub(crate) horizon: HorizonMode,
    /// Inflow non-negativity enforcement method.
    pub(crate) inflow_method: InflowNonNegativityMethod,
}
