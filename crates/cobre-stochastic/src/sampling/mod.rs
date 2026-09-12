//! Scenario sampling schemes — strategies that select which scenarios are
//! simulated each iteration.
//!
//! [`ForwardSampler`] is the composite entry point: it holds three
//! [`ClassSampler`] instances (one per entity class) and applies per-class
//! spectral correlation only for `OutOfSample`. Build one with
//! [`build_forward_sampler`].
//!
//! ```
//! use cobre_core::scenario::SamplingScheme;
//! use cobre_stochastic::sampling::{ForwardSampler, build_forward_sampler};
//! ```

use crate::context::ClassSchemes;

use std::fmt;
pub mod class_sampler;
mod eta_inversion;
pub mod external;
pub mod historical;
pub mod insample;
pub mod tables;
pub mod window;

pub use class_sampler::{ClassSampleRequest, ClassSampler, select_transition_child};
pub use external::{
    ExternalScenarioLibrary, derive_external_sample_moments, pad_library_to_uniform,
    standardize_external_inflow, standardize_external_load, standardize_external_ncs,
    validate_external_library,
};
pub use historical::{
    HistoricalScenarioLibrary, standardize_historical_windows, validate_historical_library,
};
pub use tables::{ClassNoiseTables, ForwardNoiseTables, NoiseTable};
pub use window::discover_historical_windows;
pub(crate) mod out_of_sample;

use cobre_core::{EntityId, scenario::SamplingScheme, temporal::NoiseMethod, temporal::Stage};

use crate::noise::seed::derive_class_forward_seed;
use crate::{
    OpeningTreeView, StochasticError, context::StochasticContext,
    correlation::resolve::DecomposedCorrelation, tree::generate::ClassDimensions,
};

// ---------------------------------------------------------------------------
// ForwardNoise
// ---------------------------------------------------------------------------

/// Noise payload returned by [`ForwardSampler::sample`].
#[derive(Debug)]
pub struct ForwardNoise<'b>(&'b [f64]);

impl<'b> ForwardNoise<'b> {
    /// Create a new `ForwardNoise` from a borrowed slice.
    #[must_use]
    pub fn new(data: &'b [f64]) -> Self {
        Self(data)
    }

    /// Return the underlying noise slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        self.0
    }
}

// ---------------------------------------------------------------------------
// CorrelationRef
// ---------------------------------------------------------------------------

/// Pre-decomposed correlation matrix and entity ordering for one entity class.
///
/// Present (`Some`) only for `OutOfSample` samplers. `InSample`, `Historical`,
/// and `External` samplers produce pre-correlated noise and must not apply it again.
#[derive(Debug)]
pub struct CorrelationRef<'a> {
    /// Pre-decomposed spectral factors for this entity class.
    pub decomposed: &'a DecomposedCorrelation,
    /// Canonical entity ID ordering for the class segment.
    pub entity_order: &'a [EntityId],
}

// ---------------------------------------------------------------------------
// ForwardSampler
// ---------------------------------------------------------------------------

/// Composite forward-pass sampler holding one [`ClassSampler`] per entity class.
///
/// Built once per run via [`build_forward_sampler`] and reused across all
/// `(iteration, scenario, stage)` calls without per-call allocation. The
/// `*_correlation` fields are `Some` only for `OutOfSample`; pre-correlated
/// sources leave them `None`.
pub struct ForwardSampler<'a> {
    /// Class sampler for inflow (hydro) entities.
    inflow: ClassSampler<'a>,
    /// Class sampler for stochastic load bus entities.
    load: ClassSampler<'a>,
    /// Class sampler for NCS entities.
    ncs: ClassSampler<'a>,
    /// Per-class entity counts that define the buffer split.
    dims: ClassDimensions,
    /// Correlation ref for the inflow class.
    inflow_correlation: Option<CorrelationRef<'a>>,
    /// Correlation ref for the load class.
    load_correlation: Option<CorrelationRef<'a>>,
    /// Correlation ref for the NCS class.
    ncs_correlation: Option<CorrelationRef<'a>>,
}

impl<'a> ForwardSampler<'a> {
    /// Construct a [`ForwardSampler`] from its constituent parts.
    pub(crate) fn new(
        inflow: ClassSampler<'a>,
        load: ClassSampler<'a>,
        ncs: ClassSampler<'a>,
        dims: ClassDimensions,
        inflow_correlation: Option<CorrelationRef<'a>>,
        load_correlation: Option<CorrelationRef<'a>>,
        ncs_correlation: Option<CorrelationRef<'a>>,
    ) -> Self {
        Self {
            inflow,
            load,
            ncs,
            dims,
            inflow_correlation,
            load_correlation,
            ncs_correlation,
        }
    }
}

impl std::fmt::Debug for ForwardSampler<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ForwardSampler")
            .field("dims", &self.dims)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// SampleRequest
// ---------------------------------------------------------------------------

/// Per-call arguments for [`ForwardSampler::sample`].
pub struct SampleRequest<'b> {
    /// Training iteration counter (0-based).
    pub iteration: u32,
    /// Global scenario index (includes MPI offset).
    pub scenario: u32,
    /// Stage domain ID used for seed derivation.
    pub stage: u32,
    /// Stage array index used for tree/method lookup.
    pub stage_idx: usize,
    /// Caller-owned buffer for fresh noise output.
    pub noise_buf: &'b mut [f64],
    /// Caller-owned scratch for LHS permutation generation.
    pub perm_scratch: &'b mut [usize],
    /// Per-iteration scenario-invariant tables the driver owns, rebuilt once
    /// via [`ForwardSampler::rebuild_noise_tables`].
    pub tables: &'b ForwardNoiseTables,
    /// Total scenario count across all ranks (for LHS stratification).
    pub total_scenarios: u32,
    /// Seed-derivation identifier: stages sharing a `(season_id, year)` bucket
    /// share a `noise_group_id` so their noise draws are identical.
    pub noise_group_id: u32,
    /// Sampled node's Ω sub-range — see [`ClassSampleRequest::node_opening_offset`].
    pub node_opening_offset: usize,
    /// See [`ClassSampleRequest::node_opening_offset`].
    pub node_opening_len: usize,
    /// See [`ClassSampleRequest::pinned_scenario`].
    pub pinned_scenario: Option<usize>,
}

impl ForwardSampler<'_> {
    /// Per-class initial-state hook before the stage-0 solve; currently a no-op
    /// for every class (see [`ClassSampler::apply_initial_state`]) so that
    /// forward, backward, and lower-bound paths consume the same `x_0`.
    ///
    /// `lag_offset` is an absolute index into `state` computed by the caller from
    /// the `StageIndexer`.
    pub fn apply_initial_state(
        &self,
        req: &ClassSampleRequest,
        state: &mut [f64],
        lag_offset: usize,
    ) {
        self.inflow.apply_initial_state(req, state, lag_offset);
        self.load.apply_initial_state(req, state, 0);
        self.ncs.apply_initial_state(req, state, 0);
    }

    /// Draw noise for a single `(iteration, scenario, stage)` triple into the
    /// per-class segments `[hydros | load_buses | ncs]` of `req.noise_buf`.
    ///
    /// # Errors
    ///
    /// - [`StochasticError::InsufficientData`] — when `stage_idx` is out of
    ///   bounds for any per-stage noise methods.
    //
    // Passing SampleRequest by value is intentional: we need owned access
    // to write into req.noise_buf and return a slice borrowing from it.
    #[allow(clippy::needless_pass_by_value)]
    pub fn sample<'b>(&self, req: SampleRequest<'b>) -> Result<ForwardNoise<'b>, StochasticError> {
        let total_dim = self.dims.n_hydros + self.dims.n_load_buses + self.dims.n_ncs;

        let (inflow_buf, rest) = req.noise_buf.split_at_mut(self.dims.n_hydros);
        let (load_buf, ncs_buf) = rest.split_at_mut(self.dims.n_load_buses);

        let class_req = ClassSampleRequest {
            iteration: req.iteration,
            scenario: req.scenario,
            stage: req.stage,
            stage_idx: req.stage_idx,
            total_scenarios: req.total_scenarios,
            noise_group_id: req.noise_group_id,
            node_opening_offset: req.node_opening_offset,
            node_opening_len: req.node_opening_len,
            pinned_scenario: req.pinned_scenario,
        };

        self.inflow
            .fill(&class_req, req.tables.inflow(), inflow_buf)?;
        self.load.fill(&class_req, req.tables.load(), load_buf)?;
        self.ncs.fill(&class_req, req.tables.ncs(), ncs_buf)?;

        // Correlation is applied only where a ref is set (OutOfSample); applying
        // it to a pre-correlated source would double-correlate.
        #[allow(clippy::cast_possible_wrap)]
        if let Some(ref corr) = self.inflow_correlation {
            corr.decomposed.apply_correlation_for_class(
                req.stage as i32,
                inflow_buf,
                corr.entity_order,
                "inflow",
            );
        }
        #[allow(clippy::cast_possible_wrap)]
        if let Some(ref corr) = self.load_correlation {
            corr.decomposed.apply_correlation_for_class(
                req.stage as i32,
                load_buf,
                corr.entity_order,
                "load",
            );
        }
        #[allow(clippy::cast_possible_wrap)]
        if let Some(ref corr) = self.ncs_correlation {
            corr.decomposed.apply_correlation_for_class(
                req.stage as i32,
                ncs_buf,
                corr.entity_order,
                "ncs",
            );
        }

        Ok(ForwardNoise::new(&req.noise_buf[..total_dim]))
    }

    /// Rebuild every scenario-invariant table in `out` for one training
    /// iteration's `(iteration, total_scenarios, noise_group_ids)`, reusing
    /// its buffer capacity across iterations. A class not sampled out of
    /// sample is cleared, and its `table_at` calls return `None`.
    ///
    /// # Errors
    ///
    /// Returns [`StochasticError::DimensionExceedsCapacity`] when an
    /// `OutOfSample` class uses `QmcSobol` with `dim > MAX_SOBOL_DIM`.
    pub fn rebuild_noise_tables(
        &self,
        iteration: u32,
        total_scenarios: u32,
        noise_group_ids: &[u32],
        out: &mut ForwardNoiseTables,
    ) -> Result<(), StochasticError> {
        rebuild_class_tables(
            &self.inflow,
            iteration,
            total_scenarios,
            noise_group_ids,
            &mut out.inflow,
        )?;
        rebuild_class_tables(
            &self.load,
            iteration,
            total_scenarios,
            noise_group_ids,
            &mut out.load,
        )?;
        rebuild_class_tables(
            &self.ncs,
            iteration,
            total_scenarios,
            noise_group_ids,
            &mut out.ncs,
        )?;
        Ok(())
    }
}

/// Refill one class's noise tables from its [`ClassSampler`], clearing them
/// when the class is not sampled out of sample.
///
/// # Errors
///
/// Propagates [`ClassNoiseTables::refill`]'s error.
fn rebuild_class_tables(
    sampler: &ClassSampler<'_>,
    iteration: u32,
    total_scenarios: u32,
    noise_group_ids: &[u32],
    out: &mut ClassNoiseTables,
) -> Result<(), StochasticError> {
    if let ClassSampler::OutOfSample {
        forward_seed,
        dim,
        noise_methods,
    } = sampler
    {
        out.refill(
            *forward_seed,
            *dim,
            iteration,
            total_scenarios,
            noise_group_ids,
            noise_methods,
        )
    } else {
        out.clear();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ForwardSamplerConfig
// ---------------------------------------------------------------------------

/// All parameters needed by [`build_forward_sampler`].
#[derive(Debug, Clone, Copy)]
pub struct ForwardSamplerConfig<'a> {
    /// Per-class sampling scheme selections.
    pub class_schemes: ClassSchemes,
    /// Stochastic context providing tree, seeds, correlation, and entity order.
    pub ctx: &'a StochasticContext,
    /// Study stages in index order; required by `OutOfSample` to read per-stage
    /// noise methods.
    pub stages: &'a [Stage],
    /// Per-class entity counts for noise buffer splitting.
    pub dims: ClassDimensions,
    /// Pre-standardized historical inflow windows library.
    ///
    /// Required when `class_schemes.inflow == Some(Historical)`.
    pub historical_library: Option<&'a HistoricalScenarioLibrary>,
    /// Pre-standardized external inflow scenario library.
    ///
    /// Required when `class_schemes.inflow == Some(External)`.
    pub external_inflow_library: Option<&'a ExternalScenarioLibrary>,
    /// Pre-standardized external load scenario library.
    ///
    /// Required when `class_schemes.load == Some(External)`.
    pub external_load_library: Option<&'a ExternalScenarioLibrary>,
    /// Pre-standardized external NCS scenario library.
    ///
    /// Required when `class_schemes.ncs == Some(External)`.
    pub external_ncs_library: Option<&'a ExternalScenarioLibrary>,
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Inputs to [`build_class_sampler`] for one entity class.
struct ClassSamplerParams<'a, 'b> {
    class_name: &'b str,
    scheme: SamplingScheme,
    offset: usize,
    len: usize,
    forward_seed: Option<u64>,
    noise_methods: &'b [NoiseMethod],
    tree: Option<OpeningTreeView<'a>>,
    base_seed: u64,
    historical_library: Option<&'a HistoricalScenarioLibrary>,
    external_library: Option<&'a ExternalScenarioLibrary>,
}

/// Build a [`ClassSampler`] for one entity class. `class_name` is used only in
/// error messages.
///
/// # Errors
///
/// Returns [`StochasticError::MissingScenarioSource`] when `OutOfSample` lacks a
/// `forward_seed`, when `Historical`/`External` lacks its library, or when
/// `Historical` is requested for a class other than `"inflow"`.
fn build_class_sampler<'a>(
    p: ClassSamplerParams<'a, '_>,
) -> Result<ClassSampler<'a>, StochasticError> {
    let ClassSamplerParams {
        class_name,
        scheme,
        offset,
        len,
        forward_seed,
        noise_methods,
        tree,
        base_seed,
        historical_library,
        external_library,
    } = p;
    match scheme {
        SamplingScheme::InSample => {
            let tree = tree.ok_or_else(|| StochasticError::MissingScenarioSource {
                scheme: "in_sample".to_string(),
                reason: "opening tree not available for InSample class sampler".to_string(),
            })?;
            Ok(ClassSampler::InSample {
                tree,
                base_seed,
                offset,
                len,
            })
        }
        SamplingScheme::OutOfSample => {
            let forward_seed =
                forward_seed.ok_or_else(|| StochasticError::MissingScenarioSource {
                    scheme: "out_of_sample".to_string(),
                    reason: "no forward_seed configured; set a seed in stages.json for \
                             out-of-sample forward pass noise generation"
                        .to_string(),
                })?;
            Ok(ClassSampler::OutOfSample {
                forward_seed,
                dim: len,
                noise_methods: noise_methods.into(),
            })
        }
        SamplingScheme::Historical => {
            if class_name != "inflow" {
                return Err(StochasticError::MissingScenarioSource {
                    scheme: format!("historical_{class_name}"),
                    reason: format!(
                        "historical replay is only supported for the inflow class; \
                         requested for class '{class_name}'"
                    ),
                });
            }
            let library =
                historical_library.ok_or_else(|| StochasticError::MissingScenarioSource {
                    scheme: "historical".to_string(),
                    reason: "historical replay scheme selected but no historical library \
                             was loaded; provide historical_windows in the study config"
                        .to_string(),
                })?;
            Ok(ClassSampler::Historical { library })
        }
        SamplingScheme::External => {
            let library =
                external_library.ok_or_else(|| StochasticError::MissingScenarioSource {
                    scheme: format!("external_{class_name}"),
                    reason: format!(
                        "external scenario scheme selected for class '{class_name}' but no \
                     external library was loaded; provide the external scenario file"
                    ),
                })?;
            Ok(ClassSampler::External { library })
        }
    }
}

/// Build a composite [`ForwardSampler`] from a [`ForwardSamplerConfig`].
///
/// A `None` scheme in `config.class_schemes` defaults to `InSample`.
///
/// # Errors
///
/// Returns [`StochasticError::MissingScenarioSource`] when:
/// - `OutOfSample` scheme lacks a configured `forward_seed` in `ctx`.
/// - `Historical` is selected for inflow but `historical_library` is `None`.
/// - `Historical` is selected for load or NCS (not supported).
/// - `External` is selected but the corresponding library is `None`.
pub fn build_forward_sampler(
    config: ForwardSamplerConfig<'_>,
) -> Result<ForwardSampler<'_>, StochasticError> {
    let ForwardSamplerConfig {
        class_schemes,
        ctx,
        stages,
        dims,
        historical_library,
        external_inflow_library,
        external_load_library,
        external_ncs_library,
    } = config;

    let inflow_scheme = class_schemes.inflow.unwrap_or(SamplingScheme::InSample);
    let load_scheme = class_schemes.load.unwrap_or(SamplingScheme::InSample);
    let ncs_scheme = class_schemes.ncs.unwrap_or(SamplingScheme::InSample);

    // Inflow keeps the root seed: deriving it too would change every shipped
    // inflow-only deck.
    let inflow_forward_seed = ctx.forward_seed();
    let load_forward_seed = inflow_forward_seed.map(|s| derive_class_forward_seed(s, "load"));
    let ncs_forward_seed = inflow_forward_seed.map(|s| derive_class_forward_seed(s, "ncs"));
    let base_seed = ctx.base_seed();

    let noise_methods: Box<[NoiseMethod]> = stages
        .iter()
        .map(|s| s.scenario_config.noise_method)
        .collect();

    let entity_order = ctx.entity_order();
    let inflow_order = &entity_order[..dims.n_hydros];
    let load_order = &entity_order[dims.n_hydros..dims.n_hydros + dims.n_load_buses];
    let ncs_order = &entity_order[dims.n_hydros + dims.n_load_buses..];

    let correlation = ctx.correlation();

    let inflow = build_class_sampler(ClassSamplerParams {
        class_name: "inflow",
        scheme: inflow_scheme,
        offset: 0,
        len: dims.n_hydros,
        forward_seed: inflow_forward_seed,
        noise_methods: &noise_methods,
        tree: Some(ctx.tree_view()),
        base_seed,
        historical_library,
        external_library: external_inflow_library,
    })?;

    let load = build_class_sampler(ClassSamplerParams {
        class_name: "load",
        scheme: load_scheme,
        offset: dims.n_hydros,
        len: dims.n_load_buses,
        forward_seed: load_forward_seed,
        noise_methods: &noise_methods,
        tree: Some(ctx.tree_view()),
        base_seed,
        historical_library: None,
        external_library: external_load_library,
    })?;

    let ncs = build_class_sampler(ClassSamplerParams {
        class_name: "ncs",
        scheme: ncs_scheme,
        offset: dims.n_hydros + dims.n_load_buses,
        len: dims.n_ncs,
        forward_seed: ncs_forward_seed,
        noise_methods: &noise_methods,
        tree: Some(ctx.tree_view()),
        base_seed,
        historical_library: None,
        external_library: external_ncs_library,
    })?;

    // Correlation refs are set only for OutOfSample; pre-correlated sources must
    // not be correlated again.
    let inflow_correlation = if matches!(inflow_scheme, SamplingScheme::OutOfSample) {
        Some(CorrelationRef {
            decomposed: correlation,
            entity_order: inflow_order,
        })
    } else {
        None
    };
    let load_correlation = if matches!(load_scheme, SamplingScheme::OutOfSample) {
        Some(CorrelationRef {
            decomposed: correlation,
            entity_order: load_order,
        })
    } else {
        None
    };
    let ncs_correlation = if matches!(ncs_scheme, SamplingScheme::OutOfSample) {
        Some(CorrelationRef {
            decomposed: correlation,
            entity_order: ncs_order,
        })
    } else {
        None
    };

    Ok(ForwardSampler::new(
        inflow,
        load,
        ncs,
        dims,
        inflow_correlation,
        load_correlation,
        ncs_correlation,
    ))
}

// ---------------------------------------------------------------------------
// Shared helper
// ---------------------------------------------------------------------------

/// Build the full observation sequence as `(year_offset, season_id)` pairs.
///
/// Returns `max_order + stages.len()` entries in chronological order:
/// - Indices `0..max_order`: pre-study lag seasons (oldest first)
/// - Indices `max_order..max_order + stages.len()`: study seasons
///
/// The year offset relative to the window starting year increments whenever
/// the season sequence wraps from `n_seasons - 1` back to `0`.
///
/// When `n_seasons == 1` (annual data), every entry advances exactly one year
/// (wrap-detection is suppressed to avoid self-referential offsets).
pub(crate) fn build_observation_sequence(
    stages: &[Stage],
    max_order: usize,
    n_seasons: usize,
) -> Vec<(i32, usize)> {
    if stages.is_empty() {
        return Vec::new();
    }

    let study_seasons: Vec<usize> = stages.iter().filter_map(|s| s.season_id).collect();
    if study_seasons.is_empty() {
        return Vec::new();
    }

    // Lag seasons, oldest first: step backwards from study_seasons[0].
    let first_study_season = study_seasons[0];
    let lag_seasons: Vec<usize> = (1..=max_order)
        .rev()
        .map(|k| {
            // k seasons before first_study_season, wrapping modularly.
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            let n = n_seasons as i32;
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            let s = first_study_season as i32;
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            let k_i32 = k as i32;
            #[allow(clippy::cast_sign_loss)]
            let season = ((s - k_i32 % n + n) % n) as usize;
            season
        })
        .collect();

    let full_seasons: Vec<usize> = lag_seasons.into_iter().chain(study_seasons).collect();

    // For n_seasons == 1 (annual) the wrap test `season < prev_season` is always
    // false (`0 < 0`), so each entry must advance a year by explicit arithmetic
    // instead of wrap detection.
    let mut result = Vec::with_capacity(full_seasons.len());
    if n_seasons == 1 {
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        for (i, &season) in full_seasons.iter().enumerate() {
            result.push((i as i32, season));
        }
    } else {
        // Detect season wraps (Dec→Jan) to advance the year offset.
        let mut year_offset: i32 = 0;
        let mut prev_season = full_seasons[0];
        for (i, &season) in full_seasons.iter().enumerate() {
            if i > 0 && season < prev_season {
                year_offset += 1;
            }
            result.push((year_offset, season));
            prev_season = season;
        }
        // Normalize so the first study stage (index max_order) has year_offset 0
        // — `window_year` is the first study observation's year; lags go negative.
        let study_base = result[max_order].0;
        if study_base != 0 {
            for entry in &mut result {
                entry.0 -= study_base;
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
mod tests {
    use std::collections::BTreeMap;

    use chrono::NaiveDate;
    use cobre_core::{
        Bus, DeficitSegment, EntityId, SystemBuilder,
        entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
        scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
            LoadModel, NcsModel, SamplingScheme,
        },
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };

    use super::{
        ClassNoiseTables, ClassSampleRequest, ClassSampler, ForwardNoise, ForwardNoiseTables,
        ForwardSampler, ForwardSamplerConfig, NoiseTable, SampleRequest, build_forward_sampler,
        rebuild_class_tables,
    };
    use crate::{
        NoisePointSpec, StochasticContext, StochasticError,
        context::{ClassSchemes, OpeningTreeInputs, build_stochastic_context},
        sample_forward,
        tree::generate::ClassDimensions,
        tree::lhs::{sample_lhs_point, sample_lhs_point_reference},
        tree::opening_tree::OpeningTree,
    };

    fn make_bus(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("Bus{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        }
    }

    fn make_stage(index: usize, id: i32, bf: usize) -> Stage {
        Stage {
            index,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: bf,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn make_stage_with_method(index: usize, id: i32, bf: usize, method: NoiseMethod) -> Stage {
        Stage {
            index,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: bf,
                noise_method: method,
            },
        }
    }

    fn make_hydro(id: i32) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
        };
        hydro.declare_mirror_unit_group(EntityId(0));
        hydro
    }

    fn make_inflow_model(hydro_id: i32, stage_id: i32) -> InflowModel {
        InflowModel {
            hydro_id: EntityId(hydro_id),
            stage_id,
            mean_m3s: 100.0,
            std_m3s: 30.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        }
    }

    fn identity_correlation(entity_ids: &[i32]) -> CorrelationModel {
        let n = entity_ids.len();
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
                    entities: entity_ids
                        .iter()
                        .map(|&id| CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId(id),
                        })
                        .collect(),
                    matrix,
                }],
            },
        );
        CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        }
    }

    fn build_test_ctx(forward_seed: Option<u64>) -> (StochasticContext, Vec<Stage>) {
        let hydros = vec![make_hydro(1)];
        let stages = vec![make_stage(0, 0, 5), make_stage(1, 1, 5)];
        let inflow_models = vec![make_inflow_model(1, 0), make_inflow_model(1, 1)];
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(hydros)
            .stages(stages.clone())
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();
        let ctx = build_stochastic_context(
            &system,
            42,
            forward_seed,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();
        (ctx, stages)
    }

    /// Build a uniform opening tree for testing.
    fn uniform_tree(n_stages: usize, openings: usize, dim: usize) -> OpeningTree {
        let total = n_stages * openings * dim;
        let data: Vec<f64> = (0_u32..u32::try_from(total).unwrap())
            .map(f64::from)
            .collect();
        OpeningTree::from_parts(data, vec![openings; n_stages], dim)
    }

    // -----------------------------------------------------------------------
    // Factory helper
    // -----------------------------------------------------------------------

    /// Build a `ForwardSamplerConfig` with all three classes set to `scheme`.
    fn all_classes_config<'a>(
        scheme: SamplingScheme,
        ctx: &'a StochasticContext,
        stages: &'a [Stage],
    ) -> super::ForwardSamplerConfig<'a> {
        let n_hydros = ctx.dim() - ctx.n_load_buses() - ctx.n_stochastic_ncs();
        let dims = ClassDimensions {
            n_hydros,
            n_load_buses: ctx.n_load_buses(),
            n_ncs: ctx.n_stochastic_ncs(),
        };
        super::ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(scheme),
                load: Some(scheme),
                ncs: Some(scheme),
            },
            ctx,
            stages,
            dims,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
        }
    }

    /// Rebuild `sampler`'s noise tables for one `(iteration, total,
    /// groups)` triple — the `SampleRequest.tables` every `sample()` test
    /// call needs.
    fn tables_for(
        sampler: &ForwardSampler<'_>,
        iteration: u32,
        total: u32,
        groups: &[u32],
    ) -> ForwardNoiseTables {
        let mut tables = ForwardNoiseTables::default();
        sampler
            .rebuild_noise_tables(iteration, total, groups, &mut tables)
            .expect("test fixtures never exceed the Sobol dimension cap");
        tables
    }

    // -----------------------------------------------------------------------
    // Factory tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_all_in_sample() {
        let (ctx, stages) = build_test_ctx(None);
        let config = all_classes_config(SamplingScheme::InSample, &ctx, &stages);
        let result = build_forward_sampler(config);
        assert!(
            result.is_ok(),
            "expected Ok for all-InSample but got: {result:?}"
        );
    }

    #[test]
    fn test_build_out_of_sample_missing_seed() {
        let (ctx, stages) = build_test_ctx(None);
        let config = all_classes_config(SamplingScheme::OutOfSample, &ctx, &stages);
        let result = build_forward_sampler(config);
        match result {
            Err(StochasticError::MissingScenarioSource { scheme, .. }) => {
                assert!(
                    scheme.contains("out_of_sample"),
                    "expected scheme to contain 'out_of_sample', got: {scheme}"
                );
            }
            other => panic!("expected Err(MissingScenarioSource), got: {other:?}"),
        }
    }

    #[test]
    fn test_build_out_of_sample_with_seed() {
        let (ctx, stages) = build_test_ctx(Some(99));
        let config = all_classes_config(SamplingScheme::OutOfSample, &ctx, &stages);
        let result = build_forward_sampler(config);
        assert!(
            result.is_ok(),
            "expected Ok for OutOfSample with seed but got: {result:?}"
        );
    }

    #[test]
    fn test_build_historical_with_library() {
        use super::HistoricalScenarioLibrary;
        let (ctx, stages) = build_test_ctx(None);
        let n_hydros = ctx.dim() - ctx.n_load_buses() - ctx.n_stochastic_ncs();
        let dims = ClassDimensions {
            n_hydros,
            n_load_buses: ctx.n_load_buses(),
            n_ncs: ctx.n_stochastic_ncs(),
        };
        // 3 windows, 2 stages, 1 hydro, max_order=1.
        let lib =
            HistoricalScenarioLibrary::new(3, stages.len(), n_hydros, 1, vec![2000, 2001, 2002]);
        let config = super::ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::Historical),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx: &ctx,
            stages: &stages,
            dims,
            historical_library: Some(&lib),
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
        };
        let result = build_forward_sampler(config);
        assert!(
            result.is_ok(),
            "expected Ok for Historical inflow with library, got: {result:?}"
        );
    }

    #[test]
    fn test_build_historical_missing_library() {
        let (ctx, stages) = build_test_ctx(None);
        let n_hydros = ctx.dim() - ctx.n_load_buses() - ctx.n_stochastic_ncs();
        let dims = ClassDimensions {
            n_hydros,
            n_load_buses: ctx.n_load_buses(),
            n_ncs: ctx.n_stochastic_ncs(),
        };
        let config = super::ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::Historical),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx: &ctx,
            stages: &stages,
            dims,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
        };
        let result = build_forward_sampler(config);
        match result {
            Err(StochasticError::MissingScenarioSource { scheme, .. }) => {
                assert!(
                    scheme.contains("historical"),
                    "expected scheme to contain 'historical', got: {scheme}"
                );
            }
            other => panic!("expected Err(MissingScenarioSource), got: {other:?}"),
        }
    }

    #[test]
    fn test_build_external_with_library() {
        use super::ExternalScenarioLibrary;
        let (ctx, stages) = build_test_ctx(None);
        let n_hydros = ctx.dim() - ctx.n_load_buses() - ctx.n_stochastic_ncs();
        let dims = ClassDimensions {
            n_hydros,
            n_load_buses: ctx.n_load_buses(),
            n_ncs: ctx.n_stochastic_ncs(),
        };
        let lib = ExternalScenarioLibrary::new(
            stages.len(),
            10,
            n_hydros,
            "inflow",
            vec![10usize; stages.len()],
        );
        let config = super::ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::External),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx: &ctx,
            stages: &stages,
            dims,
            historical_library: None,
            external_inflow_library: Some(&lib),
            external_load_library: None,
            external_ncs_library: None,
        };
        let result = build_forward_sampler(config);
        assert!(
            result.is_ok(),
            "expected Ok for External inflow with library, got: {result:?}"
        );
    }

    #[test]
    fn test_build_historical_load_unsupported() {
        let (ctx, stages) = build_test_ctx(None);
        let n_hydros = ctx.dim() - ctx.n_load_buses() - ctx.n_stochastic_ncs();
        let dims = ClassDimensions {
            n_hydros,
            n_load_buses: ctx.n_load_buses(),
            n_ncs: ctx.n_stochastic_ncs(),
        };
        let config = super::ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::Historical),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx: &ctx,
            stages: &stages,
            dims,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
        };
        let result = build_forward_sampler(config);
        match result {
            Err(StochasticError::MissingScenarioSource { scheme, .. }) => {
                assert_eq!(
                    scheme, "historical_load",
                    "expected scheme 'historical_load', got: {scheme}"
                );
            }
            other => panic!("expected Err(MissingScenarioSource), got: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // ForwardNoise newtype tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_noise_as_slice_newtype() {
        let data = [1.0f64, 2.0, 3.0];
        let noise = ForwardNoise::new(&data);
        assert_eq!(noise.as_slice(), &data);
    }

    #[test]
    fn test_forward_noise_as_slice() {
        let buf = [4.0f64, 5.0];
        let noise = ForwardNoise::new(&buf);
        assert_eq!(noise.as_slice(), &buf);
    }

    // -----------------------------------------------------------------------
    // Composite InSample sample() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_in_sample_sample_returns_noise() {
        let (ctx, stages) = build_test_ctx(None);
        let sampler =
            build_forward_sampler(all_classes_config(SamplingScheme::InSample, &ctx, &stages))
                .unwrap();
        let dim = ctx.dim();

        let mut noise_buf = vec![0.0f64; dim];
        let mut perm_scratch = vec![0usize; dim];
        let tables = tables_for(&sampler, 0, 5, &[]);

        let result = sampler.sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut noise_buf,
            perm_scratch: &mut perm_scratch,
            total_scenarios: 5,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: ctx.tree_view().n_openings(0),
            pinned_scenario: None,
            tables: &tables,
        });
        let noise = result.expect("expected Ok from InSample sample()");
        assert_eq!(
            noise.as_slice().len(),
            dim,
            "noise slice length {} != dim {dim}",
            noise.as_slice().len()
        );
    }

    #[test]
    fn test_in_sample_sample_is_deterministic() {
        let (ctx, stages) = build_test_ctx(None);
        let sampler =
            build_forward_sampler(all_classes_config(SamplingScheme::InSample, &ctx, &stages))
                .unwrap();
        let dim = ctx.dim();

        let mut buf_a = vec![0.0f64; dim];
        let mut buf_b = vec![0.0f64; dim];
        let mut perm_a = vec![0usize; dim];
        let mut perm_b = vec![0usize; dim];
        let tables = tables_for(&sampler, 1, 5, &[]);

        let a = sampler
            .sample(SampleRequest {
                iteration: 1,
                scenario: 2,
                stage: 0,
                stage_idx: 0,
                noise_buf: &mut buf_a,
                perm_scratch: &mut perm_a,
                total_scenarios: 5,
                noise_group_id: 0,
                node_opening_offset: 0,
                node_opening_len: ctx.tree_view().n_openings(0),
                pinned_scenario: None,
                tables: &tables,
            })
            .unwrap();
        let b = sampler
            .sample(SampleRequest {
                iteration: 1,
                scenario: 2,
                stage: 0,
                stage_idx: 0,
                noise_buf: &mut buf_b,
                perm_scratch: &mut perm_b,
                total_scenarios: 5,
                noise_group_id: 0,
                node_opening_offset: 0,
                node_opening_len: ctx.tree_view().n_openings(0),
                pinned_scenario: None,
                tables: &tables,
            })
            .unwrap();

        assert_eq!(a.as_slice(), b.as_slice());
    }

    #[test]
    fn test_composite_in_sample_fills_correct_segments() {
        // dim=5 split as 2 hydros + 2 load + 1 ncs.
        let tree = uniform_tree(1, 3, 5);
        let view = tree.view();
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 2,
            n_ncs: 1,
        };

        let sampler = ForwardSampler::new(
            ClassSampler::InSample {
                tree: view,
                base_seed: 42,
                offset: 0,
                len: 2,
            },
            ClassSampler::InSample {
                tree: tree.view(),
                base_seed: 42,
                offset: 2,
                len: 2,
            },
            ClassSampler::InSample {
                tree: tree.view(),
                base_seed: 42,
                offset: 4,
                len: 1,
            },
            dims,
            None,
            None,
            None,
        );

        let mut noise_buf = vec![0.0f64; 5];
        let mut perm_scratch = vec![0usize; 10];
        let tables = tables_for(&sampler, 0, 3, &[]);

        let result = sampler.sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut noise_buf,
            perm_scratch: &mut perm_scratch,
            total_scenarios: 3,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: tree.view().n_openings(0),
            pinned_scenario: None,
            tables: &tables,
        });

        let noise = result.expect("expected Ok from composite InSample sample()");
        assert_eq!(
            noise.as_slice().len(),
            5,
            "total noise length must equal total_dim"
        );

        let (_, full_slice) =
            sample_forward(&tree.view(), 42, 0, 0, 0, 0, 0, tree.view().n_openings(0));
        assert_eq!(
            noise.as_slice(),
            full_slice,
            "composite InSample must reproduce the full tree slice"
        );
    }

    #[test]
    fn test_composite_out_of_sample_applies_per_class_correlation() {
        let (ctx, stages) = build_test_ctx(Some(99));
        let sampler = build_forward_sampler(all_classes_config(
            SamplingScheme::OutOfSample,
            &ctx,
            &stages,
        ))
        .unwrap();
        let dim = ctx.dim();

        let mut noise_buf = vec![0.0f64; dim];
        let mut perm_scratch = vec![0usize; 5];
        let tables = tables_for(&sampler, 0, 5, &[]);

        let result = sampler.sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut noise_buf,
            perm_scratch: &mut perm_scratch,
            total_scenarios: 5,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
            tables: &tables,
        });

        let noise = result.expect("expected Ok from OutOfSample sample()");
        for (i, &v) in noise.as_slice().iter().enumerate() {
            assert!(v.is_finite(), "element[{i}] is not finite: {v}");
        }
        assert_eq!(noise.as_slice().len(), dim);
        let _ = ctx;
        let _ = stages;
    }

    #[test]
    fn test_composite_sample_deterministic() {
        let (ctx, stages) = build_test_ctx(Some(77));
        let sampler = build_forward_sampler(all_classes_config(
            SamplingScheme::OutOfSample,
            &ctx,
            &stages,
        ))
        .unwrap();
        let dim = ctx.dim();

        let mut buf_a = vec![0.0f64; dim];
        let mut buf_b = vec![0.0f64; dim];
        let mut perm_a = vec![0usize; 5];
        let mut perm_b = vec![0usize; 5];
        let tables = tables_for(&sampler, 3, 5, &[]);

        let a = sampler
            .sample(SampleRequest {
                iteration: 3,
                scenario: 7,
                stage: 1,
                stage_idx: 1,
                noise_buf: &mut buf_a,
                perm_scratch: &mut perm_a,
                total_scenarios: 5,
                noise_group_id: 1,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
                tables: &tables,
            })
            .unwrap();
        let b = sampler
            .sample(SampleRequest {
                iteration: 3,
                scenario: 7,
                stage: 1,
                stage_idx: 1,
                noise_buf: &mut buf_b,
                perm_scratch: &mut perm_b,
                total_scenarios: 5,
                noise_group_id: 1,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
                tables: &tables,
            })
            .unwrap();

        assert_eq!(
            a.as_slice(),
            b.as_slice(),
            "composite sample() must be deterministic for same inputs"
        );
    }

    #[test]
    fn test_sample_request_propagates_noise_group_id() {
        let (ctx, stages) = build_test_ctx(Some(42));
        let sampler = build_forward_sampler(all_classes_config(
            SamplingScheme::OutOfSample,
            &ctx,
            &stages,
        ))
        .unwrap();
        let dim = ctx.dim();

        let mut buf_a = vec![0.0f64; dim];
        let mut buf_b = vec![0.0f64; dim];
        let mut perm_a = vec![0usize; 5];
        let mut perm_b = vec![0usize; 5];
        let tables_group7 = tables_for(&sampler, 2, 5, &[7]);

        let a = sampler
            .sample(SampleRequest {
                iteration: 2,
                scenario: 3,
                stage: 0,
                stage_idx: 0,
                noise_buf: &mut buf_a,
                perm_scratch: &mut perm_a,
                total_scenarios: 5,
                noise_group_id: 7,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
                tables: &tables_group7,
            })
            .unwrap();
        let b = sampler
            .sample(SampleRequest {
                iteration: 2,
                scenario: 3,
                stage: 1,
                stage_idx: 0,
                noise_buf: &mut buf_b,
                perm_scratch: &mut perm_b,
                total_scenarios: 5,
                noise_group_id: 7,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
                tables: &tables_group7,
            })
            .unwrap();
        assert_eq!(
            a.as_slice(),
            b.as_slice(),
            "same noise_group_id with different stage must produce identical OutOfSample noise"
        );

        let mut buf_c = vec![0.0f64; dim];
        let mut perm_c = vec![0usize; 5];
        let tables_group8 = tables_for(&sampler, 2, 5, &[8]);
        let c = sampler
            .sample(SampleRequest {
                iteration: 2,
                scenario: 3,
                stage: 0,
                stage_idx: 0,
                noise_buf: &mut buf_c,
                perm_scratch: &mut perm_c,
                total_scenarios: 5,
                noise_group_id: 8,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
                tables: &tables_group8,
            })
            .unwrap();
        let any_differ = a.as_slice().iter().zip(c.as_slice()).any(|(x, y)| x != y);
        assert!(
            any_differ,
            "different noise_group_id must produce different OutOfSample noise"
        );
    }

    /// Classes sampled out of sample must not share a noise stream: with one
    /// seed for every class, the load and NCS slots repeated the first inflow
    /// slots bit-for-bit under every noise method. The inflow slot is pinned to
    /// its pre-fix value: inflow keeps the root seed.
    #[test]
    fn test_out_of_sample_classes_draw_distinct_streams() {
        let stages = vec![make_stage(0, 0, 5), make_stage(1, 1, 5)];
        let load_model = |stage_id: i32| LoadModel {
            bus_id: EntityId(0),
            stage_id,
            mean_mw: 100.0,
            std_mw: 10.0,
        };
        let ncs_model = |stage_id: i32| NcsModel {
            ncs_id: EntityId(20),
            stage_id,
            mean: 0.7,
            std: 0.1,
        };
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(vec![make_hydro(1), make_hydro(2)])
            .stages(stages.clone())
            .inflow_models(vec![
                make_inflow_model(1, 0),
                make_inflow_model(1, 1),
                make_inflow_model(2, 0),
                make_inflow_model(2, 1),
            ])
            .load_models(vec![load_model(0), load_model(1)])
            .ncs_models(vec![ncs_model(0), ncs_model(1)])
            .correlation(identity_correlation(&[1, 2]))
            .build()
            .unwrap();
        let ctx = build_stochastic_context(
            &system,
            42,
            Some(99),
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::OutOfSample),
                load: Some(SamplingScheme::OutOfSample),
                ncs: Some(SamplingScheme::OutOfSample),
            },
        )
        .unwrap();
        assert_eq!(ctx.n_load_buses(), 1);
        assert_eq!(ctx.n_stochastic_ncs(), 1);
        let sampler = build_forward_sampler(ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::OutOfSample),
                load: Some(SamplingScheme::OutOfSample),
                ncs: Some(SamplingScheme::OutOfSample),
            },
            ctx: &ctx,
            stages: &stages,
            dims: ClassDimensions {
                n_hydros: 2,
                n_load_buses: 1,
                n_ncs: 1,
            },
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
        })
        .unwrap();

        let mut buf = vec![0.0f64; ctx.dim()];
        let mut perm = vec![0usize; 5];
        let tables = tables_for(&sampler, 1, 5, &[]);
        let noise = sampler
            .sample(SampleRequest {
                iteration: 1,
                scenario: 2,
                stage: 0,
                stage_idx: 0,
                noise_buf: &mut buf,
                perm_scratch: &mut perm,
                total_scenarios: 5,
                noise_group_id: 0,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
                tables: &tables,
            })
            .unwrap();
        let s = noise.as_slice();
        assert_ne!(
            s[2], s[0],
            "load slot must not repeat the first inflow slot"
        );
        assert_ne!(s[2], s[1]);
        assert_ne!(s[3], s[0], "NCS slot must not repeat the first inflow slot");
        assert_ne!(s[3], s[1]);
        assert_ne!(s[3], s[2], "NCS slot must not repeat the load slot");
        assert_eq!(
            s[0].to_bits(),
            4_608_014_355_120_151_153_u64,
            "inflow draw must keep its root-seed value"
        );
    }

    // -----------------------------------------------------------------------
    // rebuild_noise_tables
    // -----------------------------------------------------------------------

    /// Build a single-hydro, three-stage study whose inflow class is sampled
    /// out of sample with the given per-stage methods and forward seed. Load
    /// and NCS default to `InSample` — this fixture exercises only inflow.
    fn build_inflow_oos_test_ctx(
        methods: [NoiseMethod; 3],
        forward_seed: u64,
    ) -> (StochasticContext, Vec<Stage>) {
        let hydros = vec![make_hydro(1)];
        let stages = vec![
            make_stage_with_method(0, 0, 5, methods[0]),
            make_stage_with_method(1, 1, 5, methods[1]),
            make_stage_with_method(2, 2, 5, methods[2]),
        ];
        let inflow_models = vec![
            make_inflow_model(1, 0),
            make_inflow_model(1, 1),
            make_inflow_model(1, 2),
        ];
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(hydros)
            .stages(stages.clone())
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();
        let ctx = build_stochastic_context(
            &system,
            42,
            Some(forward_seed),
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::OutOfSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();
        (ctx, stages)
    }

    fn build_oos_inflow_config<'a>(
        ctx: &'a StochasticContext,
        stages: &'a [Stage],
    ) -> ForwardSamplerConfig<'a> {
        let n_hydros = ctx.dim() - ctx.n_load_buses() - ctx.n_stochastic_ncs();
        ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::OutOfSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx,
            stages,
            dims: ClassDimensions {
                n_hydros,
                n_load_buses: ctx.n_load_buses(),
                n_ncs: ctx.n_stochastic_ncs(),
            },
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
        }
    }

    #[test]
    fn test_rebuild_noise_tables_variant_per_method() {
        let (ctx, stages) = build_inflow_oos_test_ctx(
            [
                NoiseMethod::QmcSobol,
                NoiseMethod::QmcHalton,
                NoiseMethod::Lhs,
            ],
            99,
        );
        let sampler = build_forward_sampler(build_oos_inflow_config(&ctx, &stages)).unwrap();
        let mut tables = ForwardNoiseTables::default();

        sampler
            .rebuild_noise_tables(0, 8, &[0, 1, 2], &mut tables)
            .expect("single-hydro dim never exceeds the Sobol dimension cap");

        assert!(matches!(
            tables.inflow().table_at(0),
            Some(NoiseTable::Sobol(_))
        ));
        assert!(matches!(
            tables.inflow().table_at(1),
            Some(NoiseTable::Halton(_))
        ));
        assert!(matches!(
            tables.inflow().table_at(2),
            Some(NoiseTable::Lhs(_))
        ));
    }

    #[test]
    fn test_rebuild_noise_tables_dedups_same_group_and_method() {
        let (ctx, stages) =
            build_inflow_oos_test_ctx([NoiseMethod::Lhs, NoiseMethod::Lhs, NoiseMethod::Lhs], 99);
        let sampler = build_forward_sampler(build_oos_inflow_config(&ctx, &stages)).unwrap();
        let mut tables = ForwardNoiseTables::default();

        sampler
            .rebuild_noise_tables(0, 8, &[0, 0, 1], &mut tables)
            .expect("Lhs never exceeds the Sobol dimension cap");

        let t0 = tables.inflow().table_at(0).expect("stage 0 has a table");
        let t1 = tables.inflow().table_at(1).expect("stage 1 has a table");
        let t2 = tables.inflow().table_at(2).expect("stage 2 has a table");
        assert!(
            std::ptr::eq(t0, t1),
            "stages sharing a (group, method) pair must resolve to the same table"
        );
        assert!(
            !std::ptr::eq(t0, t2),
            "a different noise group must resolve to a distinct table"
        );
    }

    #[test]
    fn test_rebuild_noise_tables_keys_on_group_and_method_pair() {
        let (ctx, stages) = build_inflow_oos_test_ctx(
            [NoiseMethod::Lhs, NoiseMethod::QmcSobol, NoiseMethod::Saa],
            99,
        );
        let sampler = build_forward_sampler(build_oos_inflow_config(&ctx, &stages)).unwrap();
        let mut tables = ForwardNoiseTables::default();

        sampler
            .rebuild_noise_tables(0, 8, &[0, 0, 0], &mut tables)
            .expect("single-hydro dim never exceeds the Sobol dimension cap");

        assert!(matches!(
            tables.inflow().table_at(0),
            Some(NoiseTable::Lhs(_))
        ));
        assert!(matches!(
            tables.inflow().table_at(1),
            Some(NoiseTable::Sobol(_))
        ));
        assert!(matches!(
            tables.inflow().table_at(2),
            Some(NoiseTable::Direct)
        ));
    }

    #[test]
    fn test_rebuild_noise_tables_lhs_matches_direct_point_function() {
        let (ctx, stages) =
            build_inflow_oos_test_ctx([NoiseMethod::Lhs, NoiseMethod::Lhs, NoiseMethod::Lhs], 99);
        let sampler = build_forward_sampler(build_oos_inflow_config(&ctx, &stages)).unwrap();
        let mut tables = ForwardNoiseTables::default();

        sampler
            .rebuild_noise_tables(0, 8, &[0, 0, 1], &mut tables)
            .expect("Lhs never exceeds the Sobol dimension cap");

        let Some(NoiseTable::Lhs(lhs_ctx)) = tables.inflow().table_at(2) else {
            panic!("expected NoiseTable::Lhs for stage 2");
        };

        let forward_seed = 99; // inflow keeps the root seed unchanged
        for scenario in 0..8u32 {
            let spec = NoisePointSpec {
                sampling_seed: forward_seed,
                iteration: 0,
                scenario,
                stage_id: 1,
                total_scenarios: 8,
                dim: 1,
            };
            let mut precomputed_out = [0.0f64];
            sample_lhs_point(&spec, lhs_ctx, &mut precomputed_out);

            let mut perm_scratch = vec![0usize; 8];
            let mut direct_out = [0.0f64];
            sample_lhs_point_reference(&spec, &mut direct_out, &mut perm_scratch);

            assert_eq!(
                precomputed_out, direct_out,
                "scenario {scenario}: precomputed LHS table must match the direct point function"
            );
        }
    }

    #[test]
    fn rebuild_class_tables_rejects_an_oversized_sobol_class_and_clears_non_out_of_sample() {
        let mut out = ClassNoiseTables::default();

        let oversized = ClassSampler::OutOfSample {
            forward_seed: 1,
            dim: 21_202, // one above the crate's Sobol dimension cap (21_201)
            noise_methods: vec![NoiseMethod::QmcSobol].into(),
        };
        match rebuild_class_tables(&oversized, 0, 1, &[0], &mut out) {
            Err(StochasticError::DimensionExceedsCapacity {
                dim,
                max_dim,
                method,
            }) => {
                assert_eq!(dim, 21_202, "dim field");
                assert_eq!(max_dim, 21_201, "max_dim field");
                assert!(
                    method.contains("sobol"),
                    "method must contain 'sobol', got: {method}"
                );
            }
            other => panic!("expected Err(DimensionExceedsCapacity), got {other:?}"),
        }
        assert!(out.table_at(0).is_none());

        let tree = uniform_tree(1, 2, 3);
        let in_sample = ClassSampler::InSample {
            tree: tree.view(),
            base_seed: 42,
            offset: 0,
            len: 2,
        };
        let result = rebuild_class_tables(&in_sample, 0, 1, &[], &mut out);
        assert!(
            result.is_ok(),
            "expected Ok for a class not sampled out of sample, got: {result:?}"
        );
        assert!(out.table_at(0).is_none());
    }

    // -----------------------------------------------------------------------
    // OutOfSample::fill stamp assertions
    // -----------------------------------------------------------------------

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "tables built for")]
    fn out_of_sample_fill_panics_when_tables_stale_for_iteration() {
        let sampler = ClassSampler::OutOfSample {
            forward_seed: 1,
            dim: 2,
            noise_methods: vec![NoiseMethod::Saa].into_boxed_slice(),
        };
        let mut tables = ClassNoiseTables::default();
        tables
            .refill(1, 2, 0, 4, &[0], &[NoiseMethod::Saa])
            .expect("Saa never exceeds the Sobol dimension cap");

        let req = ClassSampleRequest {
            iteration: 1,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            total_scenarios: 4,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
        };
        let mut output = vec![0.0f64; 2];
        let _ = sampler.fill(&req, &tables, &mut output);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "table built for group")]
    fn out_of_sample_fill_panics_when_stage_table_built_for_a_different_group() {
        let methods = [NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa];
        let sampler = ClassSampler::OutOfSample {
            forward_seed: 1,
            dim: 2,
            noise_methods: methods.into(),
        };
        let mut tables = ClassNoiseTables::default();
        tables
            .refill(1, 2, 0, 4, &[0, 0, 1], &methods)
            .expect("Saa never exceeds the Sobol dimension cap");

        let req = ClassSampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 2,
            total_scenarios: 4,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
        };
        let mut output = vec![0.0f64; 2];
        let _ = sampler.fill(&req, &tables, &mut output);
    }
}
