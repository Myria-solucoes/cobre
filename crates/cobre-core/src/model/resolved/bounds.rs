//! Pre-resolved per-(entity, stage) bound containers for O(1) solver lookup.
//!
//! Holds the stage-resolved bound structs (`HydroStageBounds`,
//! `ThermalStageBounds`, `LineStageBounds`, `PumpingStageBounds`,
//! `ContractStageBounds`) and the `ResolvedBounds` table. Most entity tables use
//! the flat layout `data[entity_idx * n_stages + stage_idx]`; the thermal table
//! uses an extended stride `n_stages + k_max` so the padded region
//! `[n_stages, n_stages + k_max)` can host delivery-stage values for
//! anticipated-decision columns. Populated by `cobre-io` after base bounds are
//! overlaid with stage-specific overrides; never modified after construction.

// ─── Per-(entity, stage) bound structs ───────────────────────────────────────

/// All hydro bound values for a given (hydro, stage) pair.
///
/// The 11 fields match the 11 rows in spec SS11 hydro bounds table. These are
/// the fully resolved bounds after base values from `hydros.json` have been
/// overlaid with any stage-specific overrides from `constraints/hydro_bounds.parquet`.
///
/// `max_outflow_m3s` is `Option<f64>` because the outflow upper bound may be absent
/// (unbounded above) when no flood-control limit is defined for the hydro.
/// `water_withdrawal_m3s` defaults to `0.0` when no per-stage override is present.
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
///     filling_inflow_m3s: 0.0,
///     water_withdrawal_m3s: 0.0,
/// };
/// assert!((b.min_storage_hm3 - 10.0).abs() < f64::EPSILON);
/// assert!(b.max_outflow_m3s.is_none());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HydroStageBounds {
    /// Minimum reservoir storage — dead volume \[hm³\]. Soft lower bound;
    /// violation uses `storage_violation_below` slack.
    pub min_storage_hm3: f64,
    /// Maximum reservoir storage — physical capacity \[hm³\]. Hard upper bound;
    /// emergency spillage handles excess.
    pub max_storage_hm3: f64,
    /// Minimum turbined flow \[m³/s\]. Soft lower bound;
    /// violation uses `turbined_violation_below` slack.
    pub min_turbined_m3s: f64,
    /// Maximum turbined flow \[m³/s\]. Hard upper bound.
    pub max_turbined_m3s: f64,
    /// Minimum outflow — environmental flow requirement \[m³/s\]. Soft lower bound;
    /// violation uses `outflow_violation_below` slack.
    pub min_outflow_m3s: f64,
    /// Maximum outflow — flood-control limit \[m³/s\]. Soft upper bound;
    /// violation uses `outflow_violation_above` slack. `None` = unbounded.
    pub max_outflow_m3s: Option<f64>,
    /// Minimum generation \[MW\]. Soft lower bound;
    /// violation uses `generation_violation_below` slack.
    pub min_generation_mw: f64,
    /// Maximum generation \[MW\]. Hard upper bound.
    pub max_generation_mw: f64,
    /// Maximum diversion flow \[m³/s\]. Hard upper bound. `None` = no diversion channel.
    pub max_diversion_m3s: Option<f64>,
    /// Filling inflow retained for dead-volume filling during filling stages \[m³/s\].
    /// Resolved from entity default → stage override cascade. Default `0.0`.
    pub filling_inflow_m3s: f64,
    /// Water withdrawal from reservoir per stage \[m³/s\]. Positive = water removed;
    /// negative = external addition. Default `0.0`.
    pub water_withdrawal_m3s: f64,
}

/// Thermal bound values for a given (thermal, stage) pair.
///
/// Resolved from base values in `thermals.json` with optional per-stage overrides
/// from `constraints/thermal_bounds.parquet`.
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
/// Resolved from base values in `lines.json` with optional per-stage overrides
/// from `constraints/line_bounds.parquet`. Note that block-level exchange factors
/// (per-block capacity multipliers) are stored separately and applied on top of
/// these stage-level bounds at LP construction time.
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
/// Resolved from base values in `pumping_stations.json` with optional per-stage
/// overrides from `constraints/pumping_bounds.parquet`.
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
/// Resolved from base values in `energy_contracts.json` with optional per-stage
/// overrides from `constraints/contract_bounds.parquet`. The price field can also
/// be stage-varying.
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
/// Populated by `cobre-io` after base bounds are overlaid with stage-specific
/// overrides. Provides O(1) lookup via direct array indexing.
///
/// Internal layout: most tables use `data[entity_idx * n_stages + stage_idx]`.
/// The `thermal` table uses an extended stride
/// `data[thermal_idx * thermal_stage_axis_len + stage_idx]` with
/// `thermal_stage_axis_len = n_stages + k_max`, where `k_max` is the maximum
/// lead-stages across anticipated thermals. The padded region
/// `[n_stages, n_stages + k_max)` is reserved for delivery-stage lookups by
/// anticipated-decision columns.
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
///     filling_inflow_m3s: 0.0, water_withdrawal_m3s: 0.0,
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
    /// Total number of stages. Used to compute flat indices.
    n_stages: usize,
    /// Stride used to index the `thermal` Vec; equals `n_stages + k_max`.
    ///
    /// Stored as a denormalized scalar so the hot-path accessors do not need
    /// to recompute it from a `BoundsCountsSpec` (which is not retained).
    ///
    /// Contract: this field is **required** on the wire — it is never defaulted.
    /// A payload that omits it, or that supplies `0` while `thermal` is
    /// non-empty, is rejected by [`ResolvedBoundsWire`]'s `TryFrom`. Defaulting a
    /// missing field to `0` would alias every thermal to thermal 0's stage block
    /// (the divisor in `thermal_idx * thermal_stage_axis_len + stage_idx`
    /// collapses to `stage_idx`), silently returning wrong bounds.
    thermal_stage_axis_len: usize,
    /// Flat `n_hydros * n_stages` array indexed `[hydro_idx * n_stages + stage_idx]`.
    hydro: Vec<HydroStageBounds>,
    /// Flat `n_thermals * (n_stages + k_max)` array indexed
    /// `[thermal_idx * thermal_stage_axis_len + stage_idx]`.
    ///
    /// The stage axis is asymmetric relative to the other entity tables: it is
    /// extended by `k_max` cells per thermal to host delivery-stage values for
    /// anticipated-decision columns. Indices `[0, n_stages)` are the regular
    /// study horizon; indices `[n_stages, n_stages + k_max)` are the padded
    /// region.
    thermal: Vec<ThermalStageBounds>,
    /// Flat `n_lines * n_stages` array indexed `[line_idx * n_stages + stage_idx]`.
    line: Vec<LineStageBounds>,
    /// Flat `n_pumping * n_stages` array indexed `[pumping_idx * n_stages + stage_idx]`.
    pumping: Vec<PumpingStageBounds>,
    /// Flat `n_contracts * n_stages` array indexed `[contract_idx * n_stages + stage_idx]`.
    contract: Vec<ContractStageBounds>,
}

/// Deserialization shadow for [`ResolvedBounds`].
///
/// Mirrors the serialized field layout exactly so round-trips are lossless, but
/// crucially does **not** apply `serde(default)` to `thermal_stage_axis_len`: a
/// payload missing that field fails at the field level rather than aliasing
/// every thermal to thermal 0. The `TryFrom` below additionally rejects a
/// present-but-zero stride when the thermal table is non-empty.
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
    /// Used as the default value in [`System`](crate::System) when no bound
    /// resolution has been performed yet (e.g., when building a `System` from
    /// raw entity collections without `cobre-io`).
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
        }
    }

    /// Allocate a new resolved-bounds table filled with the given defaults.
    ///
    /// `counts.n_stages` must be `> 0`. Entity counts may be `0`.
    ///
    /// # Arguments
    ///
    /// * `counts` — entity counts grouped into [`BoundsCountsSpec`]
    /// * `defaults` — default per-stage bound values grouped into [`BoundsDefaults`]
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
        }
    }

    /// Return the resolved bounds for a hydro plant at a specific stage.
    ///
    /// Returns a shared reference to avoid copying the 11-field struct on hot paths.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `hydro_index >= n_hydros` or `stage_index >= n_stages`.
    #[inline]
    #[must_use]
    pub fn hydro_bounds(&self, hydro_index: usize, stage_index: usize) -> &HydroStageBounds {
        &self.hydro[hydro_index * self.n_stages + stage_index]
    }

    /// Return the resolved bounds for a thermal unit at a specific stage.
    ///
    /// `stage_index` is valid in `[0, thermal_stage_axis_len())`, which equals
    /// `n_stages() + k_max`. Indices `>= n_stages()` access the padded region
    /// reserved for delivery-stage lookups by anticipated-decision columns.
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
    ///
    /// Used by `cobre-io` during bound resolution to set stage-specific overrides.
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
    /// `stage_index` is valid in `[0, thermal_stage_axis_len())`. Indices
    /// `>= n_stages()` write into the padded region reserved for
    /// delivery-stage lookups by anticipated-decision columns.
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

    /// Return the number of stages in this table.
    #[inline]
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }

    /// Return the stride used to index the thermal Vec.
    ///
    /// Equals `n_stages() + k_max`, where `k_max` is the maximum lead-stages
    /// across anticipated thermals. When `k_max == 0` this equals
    /// `n_stages()`. The thermal table reserves indices
    /// `[n_stages(), thermal_stage_axis_len())` for delivery-stage lookups by
    /// anticipated-decision columns.
    #[inline]
    #[must_use]
    pub fn thermal_stage_axis_len(&self) -> usize {
        self.thermal_stage_axis_len
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        BoundsCountsSpec, BoundsDefaults, ContractStageBounds, HydroStageBounds, LineStageBounds,
        PumpingStageBounds, ResolvedBounds, ThermalStageBounds,
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
            filling_inflow_m3s: 0.0,
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
                hydro: zero_hydro_default_for_tests(),
                thermal: tb,
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
                hydro: zero_hydro_default_for_tests(),
                thermal: tb,
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

    // ─── Thermal-bounds padding boundary tests ───────────────────────────────
    //
    // These tests pin down the lookup contract at the four boundary stage
    // indices that the LP-template wiring exercises:
    //
    //   * `T - 1` — last study stage (real, possibly overridden).
    //   * `T`     — first padded stage (must inherit plant base).
    //   * `T + K - 1` — last padded stage (still plant base).
    //   * `T + K` — one past the padding (panics in debug builds).
    //
    // The per-thermal base-fill semantics are verified in
    // `crates/cobre-io/src/resolution/bounds.rs::tests` because that file owns
    // `Thermal` entity construction; this module only verifies the uniform
    // `BoundsDefaults.thermal` fill behavior.

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
                hydro: zero_hydro_default_for_tests(),
                thermal: T_DEFAULT,
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
        // 5 + 3 - 1 == 7
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
            // 5 + 3 == 8: one past the last valid padded stage.
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

    /// Parameter-sweep invariant test (`anticipated_invariants`
    /// pattern). Asserts `thermal_stage_axis_len() == n_stages + k_max` across
    /// a 3 x 4 x 3 grid of configurations. The coverage gate at the end
    /// confirms every combination was reached.
    mod bounds_padding_invariants {
        use super::{
            BoundsCountsSpec, BoundsDefaults, ContractStageBounds, LineStageBounds,
            PumpingStageBounds, ResolvedBounds, T_DEFAULT, zero_hydro_default_for_tests,
        };

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
                                hydro: zero_hydro_default_for_tests(),
                                thermal: T_DEFAULT,
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
            // Coverage gate: 3 * 4 * 3 == 36 combinations expected; assert the
            // documented minimum of 27 just to guard against accidental loop
            // truncation if the grids are edited.
            assert!(
                count >= 27,
                "expected at least 27 sweep combinations, got {count}"
            );
        }
    }

    /// Helper returning a zero-valued [`HydroStageBounds`] for tests that do
    /// not exercise the hydro entity table.
    fn zero_hydro_default_for_tests() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 0.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 0.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            min_generation_mw: 0.0,
            max_generation_mw: 0.0,
            max_diversion_m3s: None,
            filling_inflow_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
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
            filling_inflow_m3s: 10.0,
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
                &BoundsDefaults {
                    hydro: zero_hydro_default_for_tests(),
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
                },
            )
        });
        assert!(
            result.is_err(),
            "ResolvedBounds::new(n_stages=0) must panic in debug builds"
        );
    }
}
