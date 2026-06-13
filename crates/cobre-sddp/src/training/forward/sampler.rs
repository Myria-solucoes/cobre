//! Forward-pass scenario sampler construction.
//!
//! Owns `build_sampler_from_ctx`: builds the `ForwardSampler` that drives
//! deterministic scenario assignment, extracted so the training loop can
//! construct it once and reuse it across iterations without re-allocation.

use cobre_stochastic::context::ClassSchemes;
use cobre_stochastic::{
    ClassDimensions, ForwardSampler, ForwardSamplerConfig, build_forward_sampler,
};

use crate::context::TrainingContext;
use crate::error::SddpError;

/// Build a [`ForwardSampler`] from the sampler-related fields of a
/// [`TrainingContext`].
///
/// Extracted so callers (e.g. the training loop in `training.rs`) can
/// construct the sampler once before the iteration loop and reuse it across
/// all iterations without repeated heap allocation.
///
/// # Errors
///
/// Propagates any error from [`build_forward_sampler`], such as a missing
/// `OutOfSample` seed or an incompatible library shape.
pub fn build_sampler_from_ctx<'a>(
    ctx: &'a TrainingContext<'a>,
) -> Result<ForwardSampler<'a>, SddpError> {
    let stochastic = ctx.stochastic;
    build_forward_sampler(ForwardSamplerConfig {
        class_schemes: ClassSchemes {
            inflow: Some(ctx.inflow_scheme),
            load: Some(ctx.load_scheme),
            ncs: Some(ctx.ncs_scheme),
        },
        ctx: stochastic,
        stages: ctx.stages,
        dims: ClassDimensions {
            n_hydros: stochastic.n_hydros(),
            n_load_buses: stochastic.n_load_buses(),
            n_ncs: stochastic.n_stochastic_ncs(),
        },
        historical_library: ctx.historical_library,
        external_inflow_library: ctx.external_inflow_library,
        external_load_library: ctx.external_load_library,
        external_ncs_library: ctx.external_ncs_library,
    })
    .map_err(SddpError::Stochastic)
}
