//! Pre-resolved per-(entity, stage) bound containers for O(1) solver lookup.
//!
//! Most entity tables use the flat layout `data[entity_idx * n_stages + stage_idx]`;
//! the thermal table's extended stride is documented on [`ResolvedBounds`].
//! Populated by `cobre-io` after base bounds are overlaid with stage-specific
//! overrides; never modified after construction.

use super::ResolvedBlockBounds;

/// All hydro bound values for a given (hydro, stage) pair.
///
/// Resolved from `hydros.json` overlaid with optional per-stage overrides from
/// `constraints/hydro_bounds.parquet`. Rows mirror the spec SS11 hydro bounds table.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::HydroStageBounds;
///
/// let b = HydroStageBounds {
///     min_storage_hm3: 10.0,
///     max_storage_hm3: 200.0,
///     min_turbined_m3s: 0.0,
///     max_turbined_m3s: 500.0,
///     min_outflow_m3s: 5.0,
///     max_outflow_m3s: None,
///     min_generation_mw: 0.0,
///     max_generation_mw: 100.0,
///     max_diversion_m3s: None,
///     filling_min_rate_m3s: 0.0,
///     water_withdrawal_m3s: 0.0,
/// };
/// assert!((b.min_storage_hm3 - 10.0).abs() < f64::EPSILON);
/// assert!(b.max_outflow_m3s.is_none());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HydroStageBounds {
    /// Dead volume \[hm³\]. Soft lower bound; slack `storage_violation_below`.
    pub min_storage_hm3: f64,
    /// Physical capacity \[hm³\]. Hard upper bound.
    pub max_storage_hm3: f64,
    /// Minimum turbined flow \[m³/s\]. Soft lower bound; slack `turbined_violation_below`.
    pub min_turbined_m3s: f64,
    /// Maximum turbined flow \[m³/s\]. Hard upper bound.
    pub max_turbined_m3s: f64,
    /// Environmental flow requirement \[m³/s\]. Soft lower bound; slack `outflow_violation_below`.
    pub min_outflow_m3s: f64,
    /// Flood-control limit \[m³/s\]. Soft upper bound; slack `outflow_violation_above`. `None` = unbounded.
    pub max_outflow_m3s: Option<f64>,
    /// Minimum generation \[MW\]. Soft lower bound; slack `generation_violation_below`.
    pub min_generation_mw: f64,
    /// Maximum generation \[MW\]. Hard upper bound.
    pub max_generation_mw: f64,
    /// Maximum diversion flow \[m³/s\]. Hard upper bound. `None` = no diversion channel.
    pub max_diversion_m3s: Option<f64>,
    /// Minimum dead-volume filling rate \[m³/s\], anchoring a per-stage minimum
    /// target-storage trajectory on `min_storage_hm3`. Not an inflow and not a cap. Default `0.0`.
    pub filling_min_rate_m3s: f64,
    /// Water withdrawal per stage \[m³/s\]. Positive = removed; negative = added. Default `0.0`.
    pub water_withdrawal_m3s: f64,
}

/// Thermal bound values for a given (thermal, stage) pair.
///
/// Resolved from `thermals.json` overlaid with `constraints/thermal_bounds.parquet`.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::ThermalStageBounds;
///
/// let b = ThermalStageBounds { min_generation_mw: 50.0, max_generation_mw: 400.0, cost_per_mwh: 120.0 };
/// let c = b; // Copy
/// assert!((c.max_generation_mw - 400.0).abs() < f64::EPSILON);
/// assert!((c.cost_per_mwh - 120.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThermalStageBounds {
    /// Minimum stable generation \[MW\]. Hard lower bound.
    pub min_generation_mw: f64,
    /// Maximum generation capacity \[MW\]. Hard upper bound.
    pub max_generation_mw: f64,
    /// Dispatch cost override (`$/MWh`). Resolved from `Thermal.cost_per_mwh` with optional
    /// per-stage override from `constraints/thermal_bounds.parquet` (null `block_id` rows only).
    pub cost_per_mwh: f64,
}

/// Transmission line bound values for a given (line, stage) pair.
///
/// Resolved from `lines.json` overlaid with `constraints/line_bounds.parquet`.
/// Per-block exchange factors are stored separately
/// ([`ResolvedExchangeFactors`](crate::ResolvedExchangeFactors)) and applied on top
/// of these stage-level bounds at LP construction time.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::LineStageBounds;
///
/// let b = LineStageBounds { direct_mw: 1000.0, reverse_mw: 800.0 };
/// let c = b; // Copy
/// assert!((c.direct_mw - 1000.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LineStageBounds {
    /// Maximum direct flow capacity \[MW\]. Hard upper bound.
    pub direct_mw: f64,
    /// Maximum reverse flow capacity \[MW\]. Hard upper bound.
    pub reverse_mw: f64,
}

/// Pumping station bound values for a given (pumping, stage) pair.
///
/// Resolved from `pumping_stations.json` overlaid with `constraints/pumping_bounds.parquet`.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::PumpingStageBounds;
///
/// let b = PumpingStageBounds { min_flow_m3s: 0.0, max_flow_m3s: 50.0 };
/// let c = b; // Copy
/// assert!((c.max_flow_m3s - 50.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PumpingStageBounds {
    /// Minimum pumped flow \[m³/s\]. Hard lower bound.
    pub min_flow_m3s: f64,
    /// Maximum pumped flow \[m³/s\]. Hard upper bound.
    pub max_flow_m3s: f64,
}

/// Energy contract bound values for a given (contract, stage) pair.
///
/// Resolved from `energy_contracts.json` overlaid with `constraints/contract_bounds.parquet`.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::ContractStageBounds;
///
/// let b = ContractStageBounds { min_mw: 0.0, max_mw: 200.0, price_per_mwh: 80.0 };
/// let c = b; // Copy
/// assert!((c.max_mw - 200.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContractStageBounds {
    /// Minimum contract usage \[MW\]. Hard lower bound.
    pub min_mw: f64,
    /// Maximum contract usage \[MW\]. Hard upper bound.
    pub max_mw: f64,
    /// Effective contract price \[$/`MWh`\]. May differ from base when a stage override
    /// supplies a per-stage price.
    pub price_per_mwh: f64,
}

// ─── Pre-resolved containers ──────────────────────────────────────────────────

/// Pre-resolved bound table for all entities across all stages.
///
/// Most tables index `data[entity_idx * n_stages + stage_idx]`; the `thermal`
/// table uses an extended `n_stages + k_max` stride — see
/// [`thermal_stage_axis_len`](Self::thermal_stage_axis_len).
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::{
///     BoundsCountsSpec, BoundsDefaults, ContractStageBounds, HydroStageBounds,
///     LineStageBounds, PumpingStageBounds, ResolvedBounds, ThermalStageBounds,
/// };
///
/// let hydro_default = HydroStageBounds {
///     min_storage_hm3: 0.0, max_storage_hm3: 100.0,
///     min_turbined_m3s: 0.0, max_turbined_m3s: 50.0,
///     min_outflow_m3s: 0.0, max_outflow_m3s: None,
///     min_generation_mw: 0.0, max_generation_mw: 30.0,
///     max_diversion_m3s: None,
///     filling_min_rate_m3s: 0.0, water_withdrawal_m3s: 0.0,
/// };
/// let thermal_default = ThermalStageBounds { min_generation_mw: 0.0, max_generation_mw: 100.0, cost_per_mwh: 50.0 };
/// let line_default = LineStageBounds { direct_mw: 500.0, reverse_mw: 500.0 };
/// let pumping_default = PumpingStageBounds { min_flow_m3s: 0.0, max_flow_m3s: 20.0 };
/// let contract_default = ContractStageBounds { min_mw: 0.0, max_mw: 50.0, price_per_mwh: 80.0 };
///
/// let table = ResolvedBounds::new(
///     &BoundsCountsSpec { n_hydros: 2, n_thermals: 1, n_lines: 1, n_pumping: 1, n_contracts: 1, n_stages: 3, k_max: 0 },
///     &BoundsDefaults { hydro: hydro_default, thermal: thermal_default, line: line_default, pumping: pumping_default, contract: contract_default },
/// );
///
/// let b = table.hydro_bounds(0, 2);
/// assert!((b.max_storage_hm3 - 100.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "ResolvedBoundsWire"))]
pub struct ResolvedBounds {
    /// Stride for every entity table except `thermal`: `data[entity_idx * n_stages + stage_idx]`.
    n_stages: usize,
    /// Stride for the `thermal` Vec; equals `n_stages + k_max`. Required on the
    /// wire and never defaulted: a missing or zero stride (with `thermal`
    /// non-empty) is rejected by [`ResolvedBoundsWire`]'s `TryFrom`, because
    /// defaulting to `0` would alias every thermal to thermal 0's stage block and
    /// silently return wrong bounds.
    thermal_stage_axis_len: usize,
    hydro: Vec<HydroStageBounds>,
    /// Indexed `[thermal_idx * thermal_stage_axis_len + stage_idx]`. The stage axis
    /// is extended by `k_max` cells per thermal: `[0, n_stages)` is the study
    /// horizon, `[n_stages, n_stages + k_max)` the padding for delivery-stage
    /// lookups by anticipated-decision columns.
    thermal: Vec<ThermalStageBounds>,
    line: Vec<LineStageBounds>,
    pumping: Vec<PumpingStageBounds>,
    contract: Vec<ContractStageBounds>,
    block: ResolvedBlockBounds,
}

/// Deserialization shadow for [`ResolvedBounds`].
///
/// Has no `serde(default)` on `thermal_stage_axis_len`, so a missing field is
/// rejected rather than aliasing every thermal to thermal 0; the `TryFrom` below
/// also rejects a present-but-zero stride with a non-empty thermal table.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct ResolvedBoundsWire {
    n_stages: usize,
    thermal_stage_axis_len: usize,
    hydro: Vec<HydroStageBounds>,
    thermal: Vec<ThermalStageBounds>,
    line: Vec<LineStageBounds>,
    pumping: Vec<PumpingStageBounds>,
    contract: Vec<ContractStageBounds>,
    #[serde(default)]
    block: ResolvedBlockBounds,
}

#[cfg(feature = "serde")]
impl TryFrom<ResolvedBoundsWire> for ResolvedBounds {
    type Error = String;

    fn try_from(wire: ResolvedBoundsWire) -> Result<Self, Self::Error> {
        if !wire.thermal.is_empty() && wire.thermal_stage_axis_len == 0 {
            return Err(
                "thermal_stage_axis_len must be > 0 when the thermal table is non-empty; \
                 a zero stride aliases every thermal to thermal 0"
                    .to_string(),
            );
        }
        Ok(Self {
            n_stages: wire.n_stages,
            thermal_stage_axis_len: wire.thermal_stage_axis_len,
            hydro: wire.hydro,
            thermal: wire.thermal,
            line: wire.line,
            pumping: wire.pumping,
            contract: wire.contract,
            block: wire.block,
        })
    }
}

/// Entity counts for constructing a [`ResolvedBounds`] table.
#[derive(Debug, Clone)]
pub struct BoundsCountsSpec {
    /// Number of hydro plants.
    pub n_hydros: usize,
    /// Number of thermal units.
    pub n_thermals: usize,
    /// Number of transmission lines.
    pub n_lines: usize,
    /// Number of pumping stations.
    pub n_pumping: usize,
    /// Number of energy contracts.
    pub n_contracts: usize,
    /// Number of time stages.
    pub n_stages: usize,
    /// Maximum lead-stages `K_max` across anticipated thermals; the thermal
    /// Vec stage axis is sized `n_stages + k_max`. Zero means no padding.
    pub k_max: usize,
}

/// Default per-stage bound values for each entity type.
#[derive(Debug, Clone)]
pub struct BoundsDefaults {
    /// Default hydro bounds for all (hydro, stage) cells.
    pub hydro: HydroStageBounds,
    /// Default thermal bounds for all (thermal, stage) cells.
    pub thermal: ThermalStageBounds,
    /// Default line bounds for all (line, stage) cells.
    pub line: LineStageBounds,
    /// Default pumping bounds for all (pumping, stage) cells.
    pub pumping: PumpingStageBounds,
    /// Default contract bounds for all (contract, stage) cells.
    pub contract: ContractStageBounds,
}

impl ResolvedBounds {
    /// Return an empty bounds table with zero entities and zero stages.
    ///
    /// The default value in [`System`](crate::System) before bound resolution.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::ResolvedBounds;
    ///
    /// let empty = ResolvedBounds::empty();
    /// assert_eq!(empty.n_stages(), 0);
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            n_stages: 0,
            thermal_stage_axis_len: 0,
            hydro: Vec::new(),
            thermal: Vec::new(),
            line: Vec::new(),
            pumping: Vec::new(),
            contract: Vec::new(),
            block: ResolvedBlockBounds::empty(),
        }
    }

    /// Allocate a new resolved-bounds table filled with the given defaults.
    ///
    /// `counts.n_stages` must be `> 0`. Entity counts may be `0`.
    #[must_use]
    pub fn new(counts: &BoundsCountsSpec, defaults: &BoundsDefaults) -> Self {
        debug_assert!(
            counts.n_stages > 0,
            "ResolvedBounds::new: n_stages must be > 0 (got 0)"
        );
        let thermal_axis = counts.n_stages + counts.k_max;
        Self {
            n_stages: counts.n_stages,
            thermal_stage_axis_len: thermal_axis,
            hydro: vec![defaults.hydro; counts.n_hydros * counts.n_stages],
            thermal: vec![defaults.thermal; counts.n_thermals * thermal_axis],
            line: vec![defaults.line; counts.n_lines * counts.n_stages],
            pumping: vec![defaults.pumping; counts.n_pumping * counts.n_stages],
            contract: vec![defaults.contract; counts.n_contracts * counts.n_stages],
            block: ResolvedBlockBounds::empty(),
        }
    }

    /// Return the resolved bounds for a hydro plant at a specific stage.
    ///
    /// Returns a reference rather than a copy to avoid copying the struct on hot paths.
    #[inline]
    #[must_use]
    pub fn hydro_bounds(&self, hydro_index: usize, stage_index: usize) -> &HydroStageBounds {
        &self.hydro[hydro_index * self.n_stages + stage_index]
    }

    /// Return the resolved bounds for a thermal unit at a specific stage.
    ///
    /// `stage_index` is valid in `[0, thermal_stage_axis_len())`; indices
    /// `>= n_stages()` access the padded delivery-stage region.
    #[inline]
    #[must_use]
    pub fn thermal_bounds(&self, thermal_index: usize, stage_index: usize) -> ThermalStageBounds {
        debug_assert!(
            self.thermal.is_empty() || self.thermal_stage_axis_len > 0,
            "thermal_stage_axis_len must be > 0 when the thermal table is non-empty"
        );
        self.thermal[thermal_index * self.thermal_stage_axis_len + stage_index]
    }

    /// Return the resolved bounds for a transmission line at a specific stage.
    #[inline]
    #[must_use]
    pub fn line_bounds(&self, line_index: usize, stage_index: usize) -> LineStageBounds {
        self.line[line_index * self.n_stages + stage_index]
    }

    /// Return the resolved bounds for a pumping station at a specific stage.
    #[inline]
    #[must_use]
    pub fn pumping_bounds(&self, pumping_index: usize, stage_index: usize) -> PumpingStageBounds {
        self.pumping[pumping_index * self.n_stages + stage_index]
    }

    /// Return the resolved bounds for an energy contract at a specific stage.
    #[inline]
    #[must_use]
    pub fn contract_bounds(
        &self,
        contract_index: usize,
        stage_index: usize,
    ) -> ContractStageBounds {
        self.contract[contract_index * self.n_stages + stage_index]
    }

    /// Return a mutable reference to the hydro bounds cell for in-place update.
    #[inline]
    pub fn hydro_bounds_mut(
        &mut self,
        hydro_index: usize,
        stage_index: usize,
    ) -> &mut HydroStageBounds {
        &mut self.hydro[hydro_index * self.n_stages + stage_index]
    }

    /// Return a mutable reference to the thermal bounds cell for in-place update.
    ///
    /// `stage_index` is valid in `[0, thermal_stage_axis_len())`; indices
    /// `>= n_stages()` write into the padded delivery-stage region.
    #[inline]
    pub fn thermal_bounds_mut(
        &mut self,
        thermal_index: usize,
        stage_index: usize,
    ) -> &mut ThermalStageBounds {
        debug_assert!(
            self.thermal.is_empty() || self.thermal_stage_axis_len > 0,
            "thermal_stage_axis_len must be > 0 when the thermal table is non-empty"
        );
        &mut self.thermal[thermal_index * self.thermal_stage_axis_len + stage_index]
    }

    /// Return a mutable reference to the line bounds cell for in-place update.
    #[inline]
    pub fn line_bounds_mut(
        &mut self,
        line_index: usize,
        stage_index: usize,
    ) -> &mut LineStageBounds {
        &mut self.line[line_index * self.n_stages + stage_index]
    }

    /// Return a mutable reference to the pumping bounds cell for in-place update.
    #[inline]
    pub fn pumping_bounds_mut(
        &mut self,
        pumping_index: usize,
        stage_index: usize,
    ) -> &mut PumpingStageBounds {
        &mut self.pumping[pumping_index * self.n_stages + stage_index]
    }

    /// Return a mutable reference to the contract bounds cell for in-place update.
    #[inline]
    pub fn contract_bounds_mut(
        &mut self,
        contract_index: usize,
        stage_index: usize,
    ) -> &mut ContractStageBounds {
        &mut self.contract[contract_index * self.n_stages + stage_index]
    }

    /// Install the per-block override overlay (bound-precedence layer 1).
    ///
    /// The overlay stays [`ResolvedBlockBounds::empty`] — every `*_bounds_at_block`
    /// call falls through to the stage cell — until this is called.
    pub fn set_block_overlay(&mut self, block: ResolvedBlockBounds) {
        self.block = block;
    }

    /// Return the installed per-block override overlay.
    #[inline]
    #[must_use]
    pub fn block_overlay(&self) -> &ResolvedBlockBounds {
        &self.block
    }

    /// Return a mutable handle to the per-block override overlay.
    #[inline]
    pub fn block_overlay_mut(&mut self) -> &mut ResolvedBlockBounds {
        &mut self.block
    }

    /// Return the resolved hydro bounds for `(hydro_index, stage_index, block_index)`,
    /// applying the block overlay over the stage cell from
    /// [`hydro_bounds`](Self::hydro_bounds): each block-eligible column takes the
    /// overlay's value when `Some`, otherwise falls through to the stage cell.
    /// With an empty overlay this returns a value bit-identical to `hydro_bounds`
    /// for every `block_index` — never special-case the empty-overlay path.
    #[inline]
    #[must_use]
    pub fn hydro_bounds_at_block(
        &self,
        hydro_index: usize,
        stage_index: usize,
        block_index: usize,
    ) -> HydroStageBounds {
        let cell = *self.hydro_bounds(hydro_index, stage_index);
        let over = self
            .block
            .hydro_override(hydro_index, stage_index, block_index);
        HydroStageBounds {
            min_storage_hm3: cell.min_storage_hm3,
            max_storage_hm3: cell.max_storage_hm3,
            min_turbined_m3s: over.min_turbined_m3s.unwrap_or(cell.min_turbined_m3s),
            max_turbined_m3s: over.max_turbined_m3s.unwrap_or(cell.max_turbined_m3s),
            min_outflow_m3s: over.min_outflow_m3s.unwrap_or(cell.min_outflow_m3s),
            max_outflow_m3s: over.max_outflow_m3s.or(cell.max_outflow_m3s),
            min_generation_mw: over.min_generation_mw.unwrap_or(cell.min_generation_mw),
            max_generation_mw: over.max_generation_mw.unwrap_or(cell.max_generation_mw),
            max_diversion_m3s: over.max_diversion_m3s.or(cell.max_diversion_m3s),
            filling_min_rate_m3s: cell.filling_min_rate_m3s,
            water_withdrawal_m3s: cell.water_withdrawal_m3s,
        }
    }

    /// Return the resolved thermal bounds for a specific block; see
    /// [`hydro_bounds_at_block`](Self::hydro_bounds_at_block) for the overlay
    /// contract. `cost_per_mwh` has no overlay column and always comes from
    /// [`thermal_bounds`](Self::thermal_bounds).
    #[inline]
    #[must_use]
    pub fn thermal_bounds_at_block(
        &self,
        thermal_index: usize,
        stage_index: usize,
        block_index: usize,
    ) -> ThermalStageBounds {
        let cell = self.thermal_bounds(thermal_index, stage_index);
        let over = self
            .block
            .thermal_override(thermal_index, stage_index, block_index);
        ThermalStageBounds {
            min_generation_mw: over.min_generation_mw.unwrap_or(cell.min_generation_mw),
            max_generation_mw: over.max_generation_mw.unwrap_or(cell.max_generation_mw),
            cost_per_mwh: cell.cost_per_mwh,
        }
    }

    /// Return the resolved line bounds for a specific block; see
    /// [`hydro_bounds_at_block`](Self::hydro_bounds_at_block) for the overlay contract.
    #[inline]
    #[must_use]
    pub fn line_bounds_at_block(
        &self,
        line_index: usize,
        stage_index: usize,
        block_index: usize,
    ) -> LineStageBounds {
        let cell = self.line_bounds(line_index, stage_index);
        let over = self
            .block
            .line_override(line_index, stage_index, block_index);
        LineStageBounds {
            direct_mw: over.direct_mw.unwrap_or(cell.direct_mw),
            reverse_mw: over.reverse_mw.unwrap_or(cell.reverse_mw),
        }
    }

    /// Return the resolved pumping bounds for a specific block; see
    /// [`hydro_bounds_at_block`](Self::hydro_bounds_at_block) for the overlay contract.
    #[inline]
    #[must_use]
    pub fn pumping_bounds_at_block(
        &self,
        pumping_index: usize,
        stage_index: usize,
        block_index: usize,
    ) -> PumpingStageBounds {
        let cell = self.pumping_bounds(pumping_index, stage_index);
        let over = self
            .block
            .pumping_override(pumping_index, stage_index, block_index);
        PumpingStageBounds {
            min_flow_m3s: over.min_flow_m3s.unwrap_or(cell.min_flow_m3s),
            max_flow_m3s: over.max_flow_m3s.unwrap_or(cell.max_flow_m3s),
        }
    }

    /// Return the resolved contract bounds for a specific block; see
    /// [`hydro_bounds_at_block`](Self::hydro_bounds_at_block) for the overlay
    /// contract. `price_per_mwh` IS block-eligible, deliberately asymmetric
    /// with [`thermal_bounds_at_block`](Self::thermal_bounds_at_block)'s
    /// `cost_per_mwh`.
    #[inline]
    #[must_use]
    pub fn contract_bounds_at_block(
        &self,
        contract_index: usize,
        stage_index: usize,
        block_index: usize,
    ) -> ContractStageBounds {
        let cell = self.contract_bounds(contract_index, stage_index);
        let over = self
            .block
            .contract_override(contract_index, stage_index, block_index);
        ContractStageBounds {
            min_mw: over.min_mw.unwrap_or(cell.min_mw),
            max_mw: over.max_mw.unwrap_or(cell.max_mw),
            price_per_mwh: over.price_per_mwh.unwrap_or(cell.price_per_mwh),
        }
    }

    /// Return the number of stages in this table.
    #[inline]
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }

    /// Return the number of pumping stations.
    ///
    /// Derived from the `pumping` Vec length and `n_stages` rather than a stored
    /// count, since `n_pumping` is never serialized. The `n_stages == 0` guard
    /// avoids divide-by-zero on [`ResolvedBounds::empty`].
    #[inline]
    #[must_use]
    pub fn n_pumping(&self) -> usize {
        if self.n_stages == 0 {
            0
        } else {
            self.pumping.len() / self.n_stages
        }
    }

    /// Return the number of energy contracts.
    ///
    /// Derived from the `contract` Vec length and `n_stages` rather than a stored
    /// count, since `n_contracts` is never serialized. The `n_stages == 0` guard
    /// avoids divide-by-zero on [`ResolvedBounds::empty`].
    #[inline]
    #[must_use]
    pub fn n_contracts(&self) -> usize {
        if self.n_stages == 0 {
            0
        } else {
            debug_assert_eq!(
                self.contract.len() % self.n_stages,
                0,
                "contract Vec length must be a multiple of n_stages"
            );
            self.contract.len() / self.n_stages
        }
    }

    /// Return the stride used to index the thermal Vec; equals `n_stages() + k_max`.
    ///
    /// `k_max` is the maximum lead-stages across anticipated thermals. The thermal
    /// table reserves indices `[n_stages(), thermal_stage_axis_len())` for
    /// delivery-stage lookups by anticipated-decision columns.
    #[inline]
    #[must_use]
    pub fn thermal_stage_axis_len(&self) -> usize {
        self.thermal_stage_axis_len
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::BlockBoundsCountsSpec;
    use super::{
        BoundsCountsSpec, BoundsDefaults, ContractStageBounds, HydroStageBounds, LineStageBounds,
        PumpingStageBounds, ResolvedBlockBounds, ResolvedBounds, ThermalStageBounds,
    };

    fn make_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 10.0,
            max_storage_hm3: 200.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 500.0,
            min_outflow_m3s: 5.0,
            max_outflow_m3s: None,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            max_diversion_m3s: None,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    #[test]
    fn test_all_bound_structs_are_copy() {
        let hb = make_hydro_bounds();
        let tb = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
        };
        let lb = LineStageBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pb = PumpingStageBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 20.0,
        };
        let cb = ContractStageBounds {
            min_mw: 0.0,
            max_mw: 50.0,
            price_per_mwh: 80.0,
        };

        let hb2 = hb;
        let tb2 = tb;
        let lb2 = lb;
        let pb2 = pb;
        let cb2 = cb;
        assert_eq!(hb, hb2);
        assert_eq!(tb, tb2);
        assert_eq!(lb, lb2);
        assert_eq!(pb, pb2);
        assert_eq!(cb, cb2);
    }

    #[test]
    fn test_resolved_bounds_construction() {
        let hb = make_hydro_bounds();
        let tb = ThermalStageBounds {
            min_generation_mw: 50.0,
            max_generation_mw: 400.0,
            cost_per_mwh: 0.0,
        };
        let lb = LineStageBounds {
            direct_mw: 1000.0,
            reverse_mw: 800.0,
        };
        let pb = PumpingStageBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 20.0,
        };
        let cb = ContractStageBounds {
            min_mw: 0.0,
            max_mw: 100.0,
            price_per_mwh: 80.0,
        };

        let table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 2,
                n_lines: 1,
                n_pumping: 1,
                n_contracts: 1,
                n_stages: 3,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: hb,
                thermal: tb,
                line: lb,
                pumping: pb,
                contract: cb,
            },
        );

        let b = table.hydro_bounds(0, 2);
        assert!((b.min_storage_hm3 - 10.0).abs() < f64::EPSILON);
        assert!((b.max_storage_hm3 - 200.0).abs() < f64::EPSILON);
        assert!(b.max_outflow_m3s.is_none());
        assert!(b.max_diversion_m3s.is_none());

        let t0 = table.thermal_bounds(0, 0);
        let t1 = table.thermal_bounds(1, 2);
        assert!((t0.max_generation_mw - 400.0).abs() < f64::EPSILON);
        assert!((t1.min_generation_mw - 50.0).abs() < f64::EPSILON);

        assert!((table.line_bounds(0, 1).direct_mw - 1000.0).abs() < f64::EPSILON);
        assert!((table.pumping_bounds(0, 0).max_flow_m3s - 20.0).abs() < f64::EPSILON);
        assert!((table.contract_bounds(0, 2).price_per_mwh - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolved_bounds_mutable_update() {
        let hb = make_hydro_bounds();
        let tb = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 200.0,
            cost_per_mwh: 0.0,
        };
        let lb = LineStageBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pb = PumpingStageBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 30.0,
        };
        let cb = ContractStageBounds {
            min_mw: 0.0,
            max_mw: 50.0,
            price_per_mwh: 60.0,
        };

        let mut table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 1,
                n_lines: 1,
                n_pumping: 1,
                n_contracts: 1,
                n_stages: 3,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: hb,
                thermal: tb,
                line: lb,
                pumping: pb,
                contract: cb,
            },
        );

        let cell = table.hydro_bounds_mut(1, 0);
        cell.min_storage_hm3 = 25.0;
        cell.max_outflow_m3s = Some(1000.0);

        assert!((table.hydro_bounds(1, 0).min_storage_hm3 - 25.0).abs() < f64::EPSILON);
        assert_eq!(table.hydro_bounds(1, 0).max_outflow_m3s, Some(1000.0));
        assert!((table.hydro_bounds(0, 0).min_storage_hm3 - 10.0).abs() < f64::EPSILON);
        assert!(table.hydro_bounds(1, 1).max_outflow_m3s.is_none());

        table.thermal_bounds_mut(0, 2).max_generation_mw = 150.0;
        assert!((table.thermal_bounds(0, 2).max_generation_mw - 150.0).abs() < f64::EPSILON);
        assert!((table.thermal_bounds(0, 0).max_generation_mw - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thermal_stage_axis_extends_with_k_max() {
        let tb = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 0.0,
        };
        let table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 3,
                k_max: 2,
            },
            &BoundsDefaults {
                thermal: tb,
                ..zero_defaults()
            },
        );
        assert_eq!(table.thermal_stage_axis_len(), 5);
        // Padded region inherits the default ThermalStageBounds.
        let padded = table.thermal_bounds(1, 4);
        assert!((padded.max_generation_mw - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thermal_stage_axis_zero_k_max_unchanged() {
        let tb = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 50.0,
            cost_per_mwh: 0.0,
        };
        let table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 4,
                k_max: 0,
            },
            &BoundsDefaults {
                thermal: tb,
                ..zero_defaults()
            },
        );
        assert_eq!(table.thermal_stage_axis_len(), table.n_stages());
        // Last valid horizon stage still works.
        let last = table.thermal_bounds(0, 3);
        assert!((last.max_generation_mw - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_bounds_has_zero_thermal_axis() {
        let empty = ResolvedBounds::empty();
        assert_eq!(empty.thermal_stage_axis_len(), 0);
        assert_eq!(empty.n_stages(), 0);
    }

    #[test]
    fn test_n_pumping_recovers_station_count() {
        let table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 2,
                n_contracts: 0,
                n_stages: 3,
                k_max: 0,
            },
            &zero_defaults(),
        );
        assert_eq!(table.n_pumping(), 2);
    }

    #[test]
    fn test_n_pumping_zero_when_no_stations() {
        let table = make_bounds_for_boundary_tests(4, 0);
        assert_eq!(table.n_pumping(), 0);
    }

    #[test]
    fn test_n_pumping_empty_table_is_zero() {
        assert_eq!(ResolvedBounds::empty().n_pumping(), 0);
    }

    // ─── Thermal-bounds padding boundary tests ───────────────────────────────
    //
    // This module verifies only the uniform `BoundsDefaults.thermal` fill; the
    // per-thermal base-fill semantics are owned by `cobre-io`'s resolution tests,
    // which construct `Thermal` entities.

    /// Sentinel default used by the thermal-padding boundary tests. Values are
    /// picked so an off-by-one read returns a value that does not collide with
    /// any plausible production default.
    const T_DEFAULT: ThermalStageBounds = ThermalStageBounds {
        min_generation_mw: 7.0,
        max_generation_mw: 77.0,
        cost_per_mwh: 7.7,
    };

    /// Construct a `ResolvedBounds` with one thermal entity, the given
    /// `n_stages` / `k_max`, and `T_DEFAULT` as the thermal default. Other
    /// entity types are zero-sized.
    fn make_bounds_for_boundary_tests(n_stages: usize, k_max: usize) -> ResolvedBounds {
        ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max,
            },
            &BoundsDefaults {
                thermal: T_DEFAULT,
                ..zero_defaults()
            },
        )
    }

    /// `T - 1`: writing a distinctive value via `thermal_bounds_mut` at the
    /// last study stage and reading it back via `thermal_bounds` must return
    /// the written value — the padding region must not shadow study stages.
    #[test]
    fn test_thermal_bounds_at_last_study_stage() {
        let mut table = make_bounds_for_boundary_tests(5, 3);
        let written = ThermalStageBounds {
            min_generation_mw: 11.0,
            max_generation_mw: 111.0,
            cost_per_mwh: 1.1,
        };
        *table.thermal_bounds_mut(0, 4) = written;
        let read = table.thermal_bounds(0, 4);
        assert!((read.min_generation_mw - 11.0).abs() < f64::EPSILON);
        assert!((read.max_generation_mw - 111.0).abs() < f64::EPSILON);
        assert!((read.cost_per_mwh - 1.1).abs() < f64::EPSILON);
    }

    /// `T`: the first padded stage must contain the uniform thermal default
    /// after `ResolvedBounds::new` — no spillover from any non-existent prior
    /// override and no zero-initialization regression.
    #[test]
    fn test_thermal_bounds_at_first_padded_stage() {
        let table = make_bounds_for_boundary_tests(5, 3);
        let padded = table.thermal_bounds(0, 5);
        assert!((padded.min_generation_mw - T_DEFAULT.min_generation_mw).abs() < f64::EPSILON);
        assert!((padded.max_generation_mw - T_DEFAULT.max_generation_mw).abs() < f64::EPSILON);
        assert!((padded.cost_per_mwh - T_DEFAULT.cost_per_mwh).abs() < f64::EPSILON);
    }

    /// `T + K_max - 1`: the last padded stage must still return the uniform
    /// thermal default — the padded region is contiguous and uniform.
    #[test]
    fn test_thermal_bounds_at_last_padded_stage() {
        let table = make_bounds_for_boundary_tests(5, 3);
        let padded = table.thermal_bounds(0, 7);
        assert!((padded.min_generation_mw - T_DEFAULT.min_generation_mw).abs() < f64::EPSILON);
        assert!((padded.max_generation_mw - T_DEFAULT.max_generation_mw).abs() < f64::EPSILON);
        assert!((padded.cost_per_mwh - T_DEFAULT.cost_per_mwh).abs() < f64::EPSILON);
    }

    /// `T + K_max`: one past the padding region must panic in debug builds.
    /// Gated by `#[cfg(debug_assertions)]` because release builds may silently
    /// read adjacent memory via `Vec` indexing (see `thermal_bounds` docs).
    #[test]
    #[cfg(debug_assertions)]
    fn test_thermal_bounds_out_of_range_panics_in_debug() {
        let table = make_bounds_for_boundary_tests(5, 3);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = table.thermal_bounds(0, 8);
        }));
        assert!(
            result.is_err(),
            "thermal_bounds(0, 8) must panic in debug builds when n_stages=5, k_max=3"
        );
    }

    /// `n_stages()` returns the *study horizon* length, not the padded axis.
    /// The padded region is internal to the thermal storage; consumers that
    /// iterate the study horizon (forward/backward passes, simulation) must
    /// continue to see `n_stages() == 5`.
    #[test]
    fn test_n_stages_unchanged_with_padding() {
        let table = make_bounds_for_boundary_tests(5, 3);
        assert_eq!(table.n_stages(), 5);
    }

    /// `thermal_stage_axis_len()` returns `n_stages + k_max`. This is the
    /// public accessor anticipated-decision consumers use to validate that
    /// `t + K_i` lookups remain in-range.
    #[test]
    fn test_thermal_stage_axis_len_equals_n_plus_k_max() {
        let table = make_bounds_for_boundary_tests(5, 3);
        assert_eq!(table.thermal_stage_axis_len(), 8);
    }

    /// Asserts `thermal_stage_axis_len() == n_stages + k_max` across a sweep of
    /// `(n_stages, k_max, n_thermals)` configurations.
    mod bounds_padding_invariants {
        use super::{BoundsCountsSpec, BoundsDefaults, ResolvedBounds, T_DEFAULT, zero_defaults};

        #[test]
        fn axis_len_matches_n_plus_k_max() {
            // n_stages starts at 1: ResolvedBounds::new debug-asserts n_stages > 0,
            // so the 0 case is exercised separately by
            // new_with_zero_n_stages_panics_in_debug.
            let n_stages_grid = [1_usize, 5, 12];
            let k_max_grid = [0_usize, 1, 3, 10];
            let n_thermals_grid = [0_usize, 1, 5];

            let mut count: usize = 0;
            for &n_stages in &n_stages_grid {
                for &k_max in &k_max_grid {
                    for &n_thermals in &n_thermals_grid {
                        let table = ResolvedBounds::new(
                            &BoundsCountsSpec {
                                n_hydros: 0,
                                n_thermals,
                                n_lines: 0,
                                n_pumping: 0,
                                n_contracts: 0,
                                n_stages,
                                k_max,
                            },
                            &BoundsDefaults {
                                thermal: T_DEFAULT,
                                ..zero_defaults()
                            },
                        );
                        assert_eq!(
                            table.thermal_stage_axis_len(),
                            n_stages + k_max,
                            "axis_len mismatch at (n_stages={n_stages}, k_max={k_max}, n_thermals={n_thermals})"
                        );
                        assert_eq!(
                            table.n_stages(),
                            n_stages,
                            "n_stages mismatch at (n_stages={n_stages}, k_max={k_max}, n_thermals={n_thermals})"
                        );
                        count += 1;
                    }
                }
            }
            // Guards against accidental loop truncation if the grids are edited.
            assert!(
                count >= 27,
                "expected at least 27 sweep combinations, got {count}"
            );
        }
    }

    /// Zero-valued defaults for every entity family; a test overrides only the
    /// families it exercises via struct-update syntax.
    fn zero_defaults() -> BoundsDefaults {
        BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 0.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 0.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                max_diversion_m3s: None,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        }
    }

    #[test]
    fn test_hydro_stage_bounds_has_eleven_fields() {
        let b = HydroStageBounds {
            min_storage_hm3: 1.0,
            max_storage_hm3: 2.0,
            min_turbined_m3s: 3.0,
            max_turbined_m3s: 4.0,
            min_outflow_m3s: 5.0,
            max_outflow_m3s: Some(6.0),
            min_generation_mw: 7.0,
            max_generation_mw: 8.0,
            max_diversion_m3s: Some(9.0),
            filling_min_rate_m3s: 10.0,
            water_withdrawal_m3s: 11.0,
        };
        assert!((b.min_storage_hm3 - 1.0).abs() < f64::EPSILON);
        assert!((b.water_withdrawal_m3s - 11.0).abs() < f64::EPSILON);
        assert_eq!(b.max_outflow_m3s, Some(6.0));
        assert_eq!(b.max_diversion_m3s, Some(9.0));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_resolved_bounds_serde_roundtrip() {
        let hb = make_hydro_bounds();
        let tb = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 0.0,
        };
        let lb = LineStageBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pb = PumpingStageBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 20.0,
        };
        let cb = ContractStageBounds {
            min_mw: 0.0,
            max_mw: 50.0,
            price_per_mwh: 80.0,
        };

        let original = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 1,
                n_lines: 1,
                n_pumping: 1,
                n_contracts: 1,
                n_stages: 3,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: hb,
                thermal: tb,
                line: lb,
                pumping: pb,
                contract: cb,
            },
        );
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: ResolvedBounds = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, restored);
    }

    /// Roundtrip with a non-zero `k_max`: guards against silent data loss in
    /// the `thermal_stage_axis_len` field. With `serde(default)` on that
    /// field, an absent JSON key would deserialize back to `0`, aliasing all
    /// thermals to thermal 0's cells. This test ensures the field is actually
    /// serialized.
    #[cfg(feature = "serde")]
    #[test]
    fn test_resolved_bounds_serde_roundtrip_with_padding() {
        let hb = make_hydro_bounds();
        let tb = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 200.0,
            cost_per_mwh: 60.0,
        };
        let lb = LineStageBounds {
            direct_mw: 50.0,
            reverse_mw: 50.0,
        };
        let pb = PumpingStageBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 20.0,
        };
        let cb = ContractStageBounds {
            min_mw: 0.0,
            max_mw: 50.0,
            price_per_mwh: 80.0,
        };

        let original = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 2,
                n_lines: 1,
                n_pumping: 1,
                n_contracts: 1,
                n_stages: 3,
                k_max: 2,
            },
            &BoundsDefaults {
                hydro: hb,
                thermal: tb,
                line: lb,
                pumping: pb,
                contract: cb,
            },
        );
        assert_eq!(original.thermal_stage_axis_len(), 5);
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: ResolvedBounds = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            restored.thermal_stage_axis_len(),
            original.thermal_stage_axis_len(),
            "thermal_stage_axis_len must survive serde roundtrip"
        );
        assert_eq!(original, restored);
    }

    /// A JSON payload that omits `thermal_stage_axis_len` while the thermal
    /// table is non-empty must be **rejected**, not silently defaulted to `0`.
    /// A zero stride would alias every thermal to thermal 0's stage block; the
    /// `serde(try_from = "ResolvedBoundsWire")` path errors instead.
    #[cfg(feature = "serde")]
    #[test]
    fn deserialize_missing_thermal_axis_len_with_thermals_is_rejected() {
        // One thermal, one stage: the thermal table is non-empty, so the
        // absent stride must trigger a deserialization error.
        let json = r#"{
            "n_stages": 1,
            "hydro": [],
            "thermal": [{"min_generation_mw": 0.0, "max_generation_mw": 100.0, "cost_per_mwh": 50.0}],
            "line": [],
            "pumping": [],
            "contract": []
        }"#;
        let result: Result<ResolvedBounds, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "deserializing a non-empty thermal table without thermal_stage_axis_len \
             must error, got Ok"
        );
    }

    /// A present-but-zero `thermal_stage_axis_len` with a non-empty thermal
    /// table is also rejected by the `TryFrom` cross-field check.
    #[cfg(feature = "serde")]
    #[test]
    fn deserialize_zero_thermal_axis_len_with_thermals_is_rejected() {
        let json = r#"{
            "n_stages": 1,
            "thermal_stage_axis_len": 0,
            "hydro": [],
            "thermal": [{"min_generation_mw": 0.0, "max_generation_mw": 100.0, "cost_per_mwh": 50.0}],
            "line": [],
            "pumping": [],
            "contract": []
        }"#;
        let result: Result<ResolvedBounds, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "deserializing a non-empty thermal table with thermal_stage_axis_len=0 \
             must error, got Ok"
        );
    }

    /// `ResolvedBounds::new` documents `n_stages > 0` as a precondition and
    /// enforces it with a `debug_assert!`. Verify the debug-build panic.
    #[test]
    #[cfg(debug_assertions)]
    fn new_with_zero_n_stages_panics_in_debug() {
        let result = std::panic::catch_unwind(|| {
            ResolvedBounds::new(
                &BoundsCountsSpec {
                    n_hydros: 1,
                    n_thermals: 1,
                    n_lines: 1,
                    n_pumping: 1,
                    n_contracts: 1,
                    n_stages: 0,
                    k_max: 0,
                },
                &zero_defaults(),
            )
        });
        assert!(
            result.is_err(),
            "ResolvedBounds::new(n_stages=0) must panic in debug builds"
        );
    }

    // ─── Block overlay tests (bound-precedence layer 1) ─────────────────────

    fn opt_f64_bits_eq(a: Option<f64>, b: Option<f64>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(x), Some(y)) => x.to_bits() == y.to_bits(),
            _ => false,
        }
    }

    fn hydro_bounds_bits_eq(a: &HydroStageBounds, b: &HydroStageBounds) -> bool {
        a.min_storage_hm3.to_bits() == b.min_storage_hm3.to_bits()
            && a.max_storage_hm3.to_bits() == b.max_storage_hm3.to_bits()
            && a.min_turbined_m3s.to_bits() == b.min_turbined_m3s.to_bits()
            && a.max_turbined_m3s.to_bits() == b.max_turbined_m3s.to_bits()
            && a.min_outflow_m3s.to_bits() == b.min_outflow_m3s.to_bits()
            && opt_f64_bits_eq(a.max_outflow_m3s, b.max_outflow_m3s)
            && a.min_generation_mw.to_bits() == b.min_generation_mw.to_bits()
            && a.max_generation_mw.to_bits() == b.max_generation_mw.to_bits()
            && opt_f64_bits_eq(a.max_diversion_m3s, b.max_diversion_m3s)
            && a.filling_min_rate_m3s.to_bits() == b.filling_min_rate_m3s.to_bits()
            && a.water_withdrawal_m3s.to_bits() == b.water_withdrawal_m3s.to_bits()
    }

    fn thermal_bounds_bits_eq(a: &ThermalStageBounds, b: &ThermalStageBounds) -> bool {
        a.min_generation_mw.to_bits() == b.min_generation_mw.to_bits()
            && a.max_generation_mw.to_bits() == b.max_generation_mw.to_bits()
            && a.cost_per_mwh.to_bits() == b.cost_per_mwh.to_bits()
    }

    fn line_bounds_bits_eq(a: &LineStageBounds, b: &LineStageBounds) -> bool {
        a.direct_mw.to_bits() == b.direct_mw.to_bits()
            && a.reverse_mw.to_bits() == b.reverse_mw.to_bits()
    }

    fn pumping_bounds_bits_eq(a: &PumpingStageBounds, b: &PumpingStageBounds) -> bool {
        a.min_flow_m3s.to_bits() == b.min_flow_m3s.to_bits()
            && a.max_flow_m3s.to_bits() == b.max_flow_m3s.to_bits()
    }

    fn contract_bounds_bits_eq(a: &ContractStageBounds, b: &ContractStageBounds) -> bool {
        a.min_mw.to_bits() == b.min_mw.to_bits()
            && a.max_mw.to_bits() == b.max_mw.to_bits()
            && a.price_per_mwh.to_bits() == b.price_per_mwh.to_bits()
    }

    /// Builds a table with distinct per-(entity, stage) values for every family
    /// so an indexing/stride bug in an `*_bounds_at_block` accessor surfaces as
    /// a bit mismatch rather than a coincidental match.
    #[allow(clippy::cast_precision_loss)] // entity/stage indices stay well within f64's exact-integer range
    fn make_distinct_bounds_table(n_entities: usize, n_stages: usize) -> ResolvedBounds {
        let mut table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: n_entities,
                n_thermals: n_entities,
                n_lines: n_entities,
                n_pumping: n_entities,
                n_contracts: n_entities,
                n_stages,
                k_max: 0,
            },
            &zero_defaults(),
        );
        for e in 0..n_entities {
            for s in 0..n_stages {
                let base = (e * 1000 + s) as f64;
                *table.hydro_bounds_mut(e, s) = HydroStageBounds {
                    min_storage_hm3: base + 1.0,
                    max_storage_hm3: base + 2.0,
                    min_turbined_m3s: base + 3.0,
                    max_turbined_m3s: base + 4.0,
                    min_outflow_m3s: base + 5.0,
                    max_outflow_m3s: if (e + s) % 2 == 0 {
                        Some(base + 6.0)
                    } else {
                        None
                    },
                    min_generation_mw: base + 7.0,
                    max_generation_mw: base + 8.0,
                    max_diversion_m3s: if (e + s) % 2 == 0 {
                        None
                    } else {
                        Some(base + 9.0)
                    },
                    filling_min_rate_m3s: base + 10.0,
                    water_withdrawal_m3s: base + 11.0,
                };
                *table.thermal_bounds_mut(e, s) = ThermalStageBounds {
                    min_generation_mw: base + 1.0,
                    max_generation_mw: base + 2.0,
                    cost_per_mwh: base + 3.0,
                };
                *table.line_bounds_mut(e, s) = LineStageBounds {
                    direct_mw: base + 1.0,
                    reverse_mw: base + 2.0,
                };
                *table.pumping_bounds_mut(e, s) = PumpingStageBounds {
                    min_flow_m3s: base + 1.0,
                    max_flow_m3s: base + 2.0,
                };
                *table.contract_bounds_mut(e, s) = ContractStageBounds {
                    min_mw: base + 1.0,
                    max_mw: base + 2.0,
                    price_per_mwh: base + 3.0,
                };
            }
        }
        table
    }

    #[test]
    fn test_empty_overlay_block_accessor_is_bit_identical_to_stage_accessor() {
        let n_entities = 2;
        let n_stages = 3;
        let table = make_distinct_bounds_table(n_entities, n_stages);

        for e in 0..n_entities {
            for s in 0..n_stages {
                let hydro_expected = *table.hydro_bounds(e, s);
                let thermal_expected = table.thermal_bounds(e, s);
                let line_expected = table.line_bounds(e, s);
                let pumping_expected = table.pumping_bounds(e, s);
                let contract_expected = table.contract_bounds(e, s);
                for b in 0..5 {
                    assert!(
                        hydro_bounds_bits_eq(
                            &hydro_expected,
                            &table.hydro_bounds_at_block(e, s, b)
                        ),
                        "hydro mismatch at (e={e}, s={s}, b={b})"
                    );
                    assert!(
                        thermal_bounds_bits_eq(
                            &thermal_expected,
                            &table.thermal_bounds_at_block(e, s, b)
                        ),
                        "thermal mismatch at (e={e}, s={s}, b={b})"
                    );
                    assert!(
                        line_bounds_bits_eq(&line_expected, &table.line_bounds_at_block(e, s, b)),
                        "line mismatch at (e={e}, s={s}, b={b})"
                    );
                    assert!(
                        pumping_bounds_bits_eq(
                            &pumping_expected,
                            &table.pumping_bounds_at_block(e, s, b)
                        ),
                        "pumping mismatch at (e={e}, s={s}, b={b})"
                    );
                    assert!(
                        contract_bounds_bits_eq(
                            &contract_expected,
                            &table.contract_bounds_at_block(e, s, b)
                        ),
                        "contract mismatch at (e={e}, s={s}, b={b})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_block_override_replaces_only_its_own_column_and_block() {
        let tb = ThermalStageBounds {
            min_generation_mw: 10.0,
            max_generation_mw: 50.0,
            cost_per_mwh: 30.0,
        };
        let mut table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 2,
                k_max: 0,
            },
            &BoundsDefaults {
                thermal: tb,
                ..zero_defaults()
            },
        );

        let mut block = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
            max_blocks: 3,
        });
        block
            .thermal_override_mut(0, 1, 2)
            .expect("in-range override cell")
            .max_generation_mw = Some(100.0);
        table.set_block_overlay(block);

        let overridden = table.thermal_bounds_at_block(0, 1, 2);
        assert!((overridden.max_generation_mw - 100.0).abs() < f64::EPSILON);
        assert!((overridden.min_generation_mw - tb.min_generation_mw).abs() < f64::EPSILON);
        assert!((overridden.cost_per_mwh - tb.cost_per_mwh).abs() < f64::EPSILON);

        let other_block = table.thermal_bounds_at_block(0, 1, 0);
        let stage_cell = table.thermal_bounds(0, 1);
        assert!(
            (other_block.min_generation_mw - stage_cell.min_generation_mw).abs() < f64::EPSILON
        );
        assert!(
            (other_block.max_generation_mw - stage_cell.max_generation_mw).abs() < f64::EPSILON
        );
        assert!((other_block.cost_per_mwh - stage_cell.cost_per_mwh).abs() < f64::EPSILON);
    }

    #[test]
    fn test_optional_column_override_replaces_and_falls_through() {
        let mut hydro_default = make_hydro_bounds();
        hydro_default.max_outflow_m3s = None;

        let mut table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 2,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: hydro_default,
                ..zero_defaults()
            },
        );

        let mut block = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
            max_blocks: 2,
        });
        block
            .hydro_override_mut(0, 0, 0)
            .expect("in-range override cell")
            .max_outflow_m3s = Some(250.0);
        table.set_block_overlay(block);

        assert_eq!(
            table.hydro_bounds_at_block(0, 0, 0).max_outflow_m3s,
            Some(250.0)
        );
        assert_eq!(table.hydro_bounds_at_block(0, 0, 1).max_outflow_m3s, None);

        // `over.X.or(cell.X)` also type-checks with the operands swapped
        // (`cell.X.or(over.X)`, stage-wide beating the block override); with at
        // most one side `Some` above, the two are indistinguishable. Pin it
        // with both sides `Some` and distinct.
        table.hydro_bounds_mut(0, 1).max_outflow_m3s = Some(400.0);
        table.hydro_bounds_mut(0, 1).max_diversion_m3s = Some(40.0);
        table
            .block_overlay_mut()
            .hydro_override_mut(0, 1, 0)
            .expect("in-range override cell")
            .max_outflow_m3s = Some(650.0);
        table
            .block_overlay_mut()
            .hydro_override_mut(0, 1, 1)
            .expect("in-range override cell")
            .max_diversion_m3s = Some(80.0);

        assert_eq!(
            table.hydro_bounds_at_block(0, 1, 0).max_outflow_m3s,
            Some(650.0)
        );
        assert_eq!(
            table.hydro_bounds_at_block(0, 1, 1).max_diversion_m3s,
            Some(80.0)
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_resolved_bounds_wire_round_trip_with_overlay() {
        let hb = make_hydro_bounds();
        let tb = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 20.0,
        };
        let lb = LineStageBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pb = PumpingStageBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 20.0,
        };
        let cb = ContractStageBounds {
            min_mw: 0.0,
            max_mw: 50.0,
            price_per_mwh: 80.0,
        };

        let mut original = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 1,
                n_lines: 1,
                n_pumping: 1,
                n_contracts: 1,
                n_stages: 2,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: hb,
                thermal: tb,
                line: lb,
                pumping: pb,
                contract: cb,
            },
        );

        let mut block = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 1,
            n_pumping: 1,
            n_contracts: 1,
            n_stages: 2,
            max_blocks: 2,
        });
        block
            .thermal_override_mut(0, 0, 0)
            .expect("in-range override cell")
            .max_generation_mw = Some(75.0);
        original.set_block_overlay(block);

        let json = serde_json::to_string(&original).expect("serialize json");
        let restored_json: ResolvedBounds = serde_json::from_str(&json).expect("deserialize json");
        assert_eq!(original, restored_json);

        let bytes = postcard::to_allocvec(&original).expect("serialize postcard");
        let restored_postcard: ResolvedBounds =
            postcard::from_bytes(&bytes).expect("deserialize postcard");
        assert_eq!(original, restored_postcard);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_resolved_bounds_wire_absent_overlay_defaults_to_empty() {
        let json = r#"{
            "n_stages": 1,
            "thermal_stage_axis_len": 1,
            "hydro": [],
            "thermal": [],
            "line": [],
            "pumping": [],
            "contract": []
        }"#;
        let restored: ResolvedBounds = serde_json::from_str(json).expect("deserialize");
        assert!(restored.block_overlay().is_empty());
    }
}
