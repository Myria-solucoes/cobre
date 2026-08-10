//! Top-level system struct and builder.
//!
//! All entity collections in `System` are stored in canonical ID-sorted order to ensure
//! declaration-order invariance: results are bit-for-bit identical regardless of input
//! entity ordering.

use std::collections::HashMap;

use crate::{
    Bus, CascadeTopology, CorrelationModel, EnergyContract, EntityId, ExternalLoadRow,
    ExternalNcsRow, ExternalScenarioRow, GenericConstraint, HorizonGraph, Hydro, InflowHistoryRow,
    InflowModel, InitialConditions, Line, LoadModel, NcsModel, NetworkTopology,
    NonControllableSource, PumpingStation, ResolvedBounds, ResolvedGenericConstraintBounds,
    ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties, Stage, Thermal,
};

mod builder;
mod validate;

pub use builder::SystemBuilder;

#[cfg(feature = "serde")]
use validate::{build_index, build_stage_index};

/// Top-level system representation: immutable and thread-safe after construction.
///
/// Entity collections are in canonical order (sorted by [`EntityId`]'s inner `i32`).
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
///
/// let bus = Bus {
///     id: EntityId(1),
///     name: "Main Bus".to_string(),
///     operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
///     deficit_segments: vec![],
///     excess_cost: 0.0,
/// };
///
/// let system = SystemBuilder::new()
///     .buses(vec![bus])
///     .build()
///     .expect("valid system");
///
/// assert_eq!(system.n_buses(), 1);
/// assert!(system.bus(EntityId(1)).is_some());
/// ```
#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(from = "SystemRepr"))]
pub struct System {
    buses: Vec<Bus>,
    lines: Vec<Line>,
    hydros: Vec<Hydro>,
    thermals: Vec<Thermal>,
    pumping_stations: Vec<PumpingStation>,
    contracts: Vec<EnergyContract>,
    non_controllable_sources: Vec<NonControllableSource>,

    // Not serialized: `HashMap` iteration order is unstable, so serializing an
    // index would make the wire payload non-reproducible for identical content.
    // `serde(from = "SystemRepr")` above is `Deserialize`'s sole entry point and
    // rebuilds them unconditionally — without it every lookup on a deserialized
    // `System` silently returns `None`.
    #[cfg_attr(feature = "serde", serde(skip))]
    bus_index: HashMap<EntityId, usize>,
    #[cfg_attr(feature = "serde", serde(skip))]
    line_index: HashMap<EntityId, usize>,
    #[cfg_attr(feature = "serde", serde(skip))]
    hydro_index: HashMap<EntityId, usize>,
    #[cfg_attr(feature = "serde", serde(skip))]
    thermal_index: HashMap<EntityId, usize>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pumping_station_index: HashMap<EntityId, usize>,
    #[cfg_attr(feature = "serde", serde(skip))]
    contract_index: HashMap<EntityId, usize>,
    #[cfg_attr(feature = "serde", serde(skip))]
    non_controllable_source_index: HashMap<EntityId, usize>,

    /// Resolved hydro cascade graph.
    cascade: CascadeTopology,
    /// Resolved transmission network topology.
    network: NetworkTopology,

    /// Ordered list of stages (study + pre-study), sorted by `id` (canonical order).
    stages: Vec<Stage>,
    /// Policy graph defining stage transitions, horizon type, and discount rate.
    policy_graph: HorizonGraph,

    #[cfg_attr(feature = "serde", serde(skip))]
    stage_index: HashMap<i32, usize>,

    /// Pre-resolved penalty values for all entities across all stages.
    penalties: ResolvedPenalties,
    /// Pre-resolved bound values for all entities across all stages.
    bounds: ResolvedBounds,
    /// Pre-resolved RHS bound table for user-defined generic linear constraints.
    resolved_generic_bounds: ResolvedGenericConstraintBounds,
    /// Pre-resolved per-block load scaling factors.
    resolved_load_factors: ResolvedLoadFactors,
    /// Pre-resolved per-stage NCS available generation bounds.
    resolved_ncs_bounds: ResolvedNcsBounds,
    /// Pre-resolved per-block NCS generation scaling factors.
    resolved_ncs_factors: ResolvedNcsFactors,

    /// PAR(p) inflow model parameters, one entry per (hydro, stage) pair.
    inflow_models: Vec<InflowModel>,
    /// Seasonal load statistics, one entry per (bus, stage) pair.
    load_models: Vec<LoadModel>,
    /// NCS availability noise model parameters, one entry per (ncs, stage) pair.
    ncs_models: Vec<NcsModel>,
    /// Correlation model for stochastic inflow/load generation.
    correlation: CorrelationModel,

    /// Initial reservoir storage levels at the start of the study.
    initial_conditions: InitialConditions,
    /// User-defined generic linear constraints, sorted by `id`.
    generic_constraints: Vec<GenericConstraint>,

    /// Raw historical inflow observations, sorted by `(hydro_id, start_date)` ascending.
    inflow_history: Vec<InflowHistoryRow>,
    /// Raw external inflow scenario rows, sorted by `(stage_id, scenario_id, hydro_id)` ascending.
    external_scenarios: Vec<ExternalScenarioRow>,
    /// Raw external load scenario rows, sorted by `(stage_id, scenario_id, bus_id)` ascending.
    external_load_scenarios: Vec<ExternalLoadRow>,
    /// Raw external NCS scenario rows, sorted by `(stage_id, scenario_id, ncs_id)` ascending.
    external_ncs_scenarios: Vec<ExternalNcsRow>,
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<System>();
};

/// Deserialize-only mirror of [`System`] without the derived indices. Field
/// order must match `System`'s non-skipped fields exactly — postcard is
/// non-self-describing, so a reorder silently decodes into the wrong fields.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct SystemRepr {
    buses: Vec<Bus>,
    lines: Vec<Line>,
    hydros: Vec<Hydro>,
    thermals: Vec<Thermal>,
    pumping_stations: Vec<PumpingStation>,
    contracts: Vec<EnergyContract>,
    non_controllable_sources: Vec<NonControllableSource>,
    cascade: CascadeTopology,
    network: NetworkTopology,
    stages: Vec<Stage>,
    policy_graph: HorizonGraph,
    penalties: ResolvedPenalties,
    bounds: ResolvedBounds,
    resolved_generic_bounds: ResolvedGenericConstraintBounds,
    resolved_load_factors: ResolvedLoadFactors,
    resolved_ncs_bounds: ResolvedNcsBounds,
    resolved_ncs_factors: ResolvedNcsFactors,
    inflow_models: Vec<InflowModel>,
    load_models: Vec<LoadModel>,
    ncs_models: Vec<NcsModel>,
    correlation: CorrelationModel,
    initial_conditions: InitialConditions,
    generic_constraints: Vec<GenericConstraint>,
    inflow_history: Vec<InflowHistoryRow>,
    external_scenarios: Vec<ExternalScenarioRow>,
    external_load_scenarios: Vec<ExternalLoadRow>,
    external_ncs_scenarios: Vec<ExternalNcsRow>,
}

#[cfg(feature = "serde")]
impl From<SystemRepr> for System {
    fn from(repr: SystemRepr) -> Self {
        let mut system = System {
            buses: repr.buses,
            lines: repr.lines,
            hydros: repr.hydros,
            thermals: repr.thermals,
            pumping_stations: repr.pumping_stations,
            contracts: repr.contracts,
            non_controllable_sources: repr.non_controllable_sources,
            bus_index: HashMap::new(),
            line_index: HashMap::new(),
            hydro_index: HashMap::new(),
            thermal_index: HashMap::new(),
            pumping_station_index: HashMap::new(),
            contract_index: HashMap::new(),
            non_controllable_source_index: HashMap::new(),
            cascade: repr.cascade,
            network: repr.network,
            stages: repr.stages,
            policy_graph: repr.policy_graph,
            stage_index: HashMap::new(),
            penalties: repr.penalties,
            bounds: repr.bounds,
            resolved_generic_bounds: repr.resolved_generic_bounds,
            resolved_load_factors: repr.resolved_load_factors,
            resolved_ncs_bounds: repr.resolved_ncs_bounds,
            resolved_ncs_factors: repr.resolved_ncs_factors,
            inflow_models: repr.inflow_models,
            load_models: repr.load_models,
            ncs_models: repr.ncs_models,
            correlation: repr.correlation,
            initial_conditions: repr.initial_conditions,
            generic_constraints: repr.generic_constraints,
            inflow_history: repr.inflow_history,
            external_scenarios: repr.external_scenarios,
            external_load_scenarios: repr.external_load_scenarios,
            external_ncs_scenarios: repr.external_ncs_scenarios,
        };
        system.rebuild_indices();
        system
    }
}

impl System {
    /// Returns all buses in canonical ID order.
    #[must_use]
    pub fn buses(&self) -> &[Bus] {
        &self.buses
    }

    /// Returns all lines in canonical ID order.
    #[must_use]
    pub fn lines(&self) -> &[Line] {
        &self.lines
    }

    /// Returns all hydro plants in canonical ID order.
    #[must_use]
    pub fn hydros(&self) -> &[Hydro] {
        &self.hydros
    }

    /// Returns all thermal plants in canonical ID order.
    #[must_use]
    pub fn thermals(&self) -> &[Thermal] {
        &self.thermals
    }

    /// Returns all pumping stations in canonical ID order.
    #[must_use]
    pub fn pumping_stations(&self) -> &[PumpingStation] {
        &self.pumping_stations
    }

    /// Returns all energy contracts in canonical ID order.
    #[must_use]
    pub fn contracts(&self) -> &[EnergyContract] {
        &self.contracts
    }

    /// Returns all non-controllable sources in canonical ID order.
    #[must_use]
    pub fn non_controllable_sources(&self) -> &[NonControllableSource] {
        &self.non_controllable_sources
    }

    /// Returns the number of buses in the system.
    #[must_use]
    pub fn n_buses(&self) -> usize {
        self.buses.len()
    }

    /// Returns the number of lines in the system.
    #[must_use]
    pub fn n_lines(&self) -> usize {
        self.lines.len()
    }

    /// Returns the number of hydro plants in the system.
    #[must_use]
    pub fn n_hydros(&self) -> usize {
        self.hydros.len()
    }

    /// Returns the number of thermal plants in the system.
    #[must_use]
    pub fn n_thermals(&self) -> usize {
        self.thermals.len()
    }

    /// Returns the number of pumping stations in the system.
    #[must_use]
    pub fn n_pumping_stations(&self) -> usize {
        self.pumping_stations.len()
    }

    /// Returns the number of energy contracts in the system.
    #[must_use]
    pub fn n_contracts(&self) -> usize {
        self.contracts.len()
    }

    /// Returns the number of non-controllable sources in the system.
    #[must_use]
    pub fn n_non_controllable_sources(&self) -> usize {
        self.non_controllable_sources.len()
    }

    /// Returns the bus with the given ID, or `None` if not found.
    #[must_use]
    pub fn bus(&self, id: EntityId) -> Option<&Bus> {
        self.bus_index.get(&id).map(|&i| &self.buses[i])
    }

    /// Returns the line with the given ID, or `None` if not found.
    #[must_use]
    pub fn line(&self, id: EntityId) -> Option<&Line> {
        self.line_index.get(&id).map(|&i| &self.lines[i])
    }

    /// Returns the hydro plant with the given ID, or `None` if not found.
    #[must_use]
    pub fn hydro(&self, id: EntityId) -> Option<&Hydro> {
        self.hydro_index.get(&id).map(|&i| &self.hydros[i])
    }

    /// Returns the thermal plant with the given ID, or `None` if not found.
    #[must_use]
    pub fn thermal(&self, id: EntityId) -> Option<&Thermal> {
        self.thermal_index.get(&id).map(|&i| &self.thermals[i])
    }

    /// Returns the pumping station with the given ID, or `None` if not found.
    #[must_use]
    pub fn pumping_station(&self, id: EntityId) -> Option<&PumpingStation> {
        self.pumping_station_index
            .get(&id)
            .map(|&i| &self.pumping_stations[i])
    }

    /// Returns the energy contract with the given ID, or `None` if not found.
    #[must_use]
    pub fn contract(&self, id: EntityId) -> Option<&EnergyContract> {
        self.contract_index.get(&id).map(|&i| &self.contracts[i])
    }

    /// Returns the non-controllable source with the given ID, or `None` if not found.
    #[must_use]
    pub fn non_controllable_source(&self, id: EntityId) -> Option<&NonControllableSource> {
        self.non_controllable_source_index
            .get(&id)
            .map(|&i| &self.non_controllable_sources[i])
    }

    /// Returns a reference to the hydro cascade topology.
    #[must_use]
    pub fn cascade(&self) -> &CascadeTopology {
        &self.cascade
    }

    /// Returns a reference to the transmission network topology.
    #[must_use]
    pub fn network(&self) -> &NetworkTopology {
        &self.network
    }

    /// Returns all stages in canonical ID order (study and pre-study stages).
    #[must_use]
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// Returns the number of stages (study and pre-study) in the system.
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.stages.len()
    }

    /// Returns the stage with the given stage ID, or `None` if not found.
    ///
    /// Stage IDs are `i32`. Study stages have non-negative IDs; pre-study
    /// stages (used only for PAR model lag initialization) have negative IDs.
    #[must_use]
    pub fn stage(&self, id: i32) -> Option<&Stage> {
        self.stage_index.get(&id).map(|&i| &self.stages[i])
    }

    /// Returns a reference to the policy graph.
    #[must_use]
    pub fn policy_graph(&self) -> &HorizonGraph {
        &self.policy_graph
    }

    /// Returns a reference to the pre-resolved penalty table.
    #[must_use]
    pub fn penalties(&self) -> &ResolvedPenalties {
        &self.penalties
    }

    /// Returns a reference to the pre-resolved bounds table.
    #[must_use]
    pub fn bounds(&self) -> &ResolvedBounds {
        &self.bounds
    }

    /// Returns a reference to the pre-resolved generic constraint RHS bound table.
    #[must_use]
    pub fn resolved_generic_bounds(&self) -> &ResolvedGenericConstraintBounds {
        &self.resolved_generic_bounds
    }

    /// Returns a reference to the pre-resolved per-block load scaling factors.
    #[must_use]
    pub fn resolved_load_factors(&self) -> &ResolvedLoadFactors {
        &self.resolved_load_factors
    }

    /// Returns a reference to the pre-resolved per-stage NCS available generation bounds.
    #[must_use]
    pub fn resolved_ncs_bounds(&self) -> &ResolvedNcsBounds {
        &self.resolved_ncs_bounds
    }

    /// Returns a reference to the pre-resolved per-block NCS generation scaling factors.
    #[must_use]
    pub fn resolved_ncs_factors(&self) -> &ResolvedNcsFactors {
        &self.resolved_ncs_factors
    }

    /// Returns all PAR(p) inflow models in canonical order (by hydro ID, then stage ID).
    #[must_use]
    pub fn inflow_models(&self) -> &[InflowModel] {
        &self.inflow_models
    }

    /// Returns all load models in canonical order (by bus ID, then stage ID).
    #[must_use]
    pub fn load_models(&self) -> &[LoadModel] {
        &self.load_models
    }

    /// Returns all NCS availability noise models in canonical order (by NCS ID, then stage ID).
    #[must_use]
    pub fn ncs_models(&self) -> &[NcsModel] {
        &self.ncs_models
    }

    /// Returns a reference to the correlation model.
    #[must_use]
    pub fn correlation(&self) -> &CorrelationModel {
        &self.correlation
    }

    /// Returns a reference to the initial conditions.
    #[must_use]
    pub fn initial_conditions(&self) -> &InitialConditions {
        &self.initial_conditions
    }

    /// Returns all generic constraints in canonical ID order.
    #[must_use]
    pub fn generic_constraints(&self) -> &[GenericConstraint] {
        &self.generic_constraints
    }

    /// Returns the raw historical inflow observations, sorted by `(hydro_id, start_date)`.
    ///
    /// Returns an empty slice when `scenarios/inflow_history.parquet` was absent
    /// at case-load time.
    #[must_use]
    pub fn inflow_history(&self) -> &[InflowHistoryRow] {
        &self.inflow_history
    }

    /// Returns the raw external inflow scenario rows, sorted by `(stage_id, scenario_id, hydro_id)`.
    ///
    /// Returns an empty slice when no external inflow scenario file was present at case-load time.
    #[must_use]
    pub fn external_scenarios(&self) -> &[ExternalScenarioRow] {
        &self.external_scenarios
    }

    /// Returns the raw external load scenario rows, sorted by `(stage_id, scenario_id, bus_id)`.
    ///
    /// Returns an empty slice when no external load scenario file was present at case-load time.
    #[must_use]
    pub fn external_load_scenarios(&self) -> &[ExternalLoadRow] {
        &self.external_load_scenarios
    }

    /// Returns the raw external NCS scenario rows, sorted by `(stage_id, scenario_id, ncs_id)`.
    ///
    /// Returns an empty slice when no external NCS scenario file was present at case-load time.
    #[must_use]
    pub fn external_ncs_scenarios(&self) -> &[ExternalNcsRow] {
        &self.external_ncs_scenarios
    }

    /// Replace `inflow_models` and `correlation`, returning the `System` with all
    /// other fields preserved. The only supported post-construction update path for
    /// these fields, which are not public outside this crate.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::{EntityId, SystemBuilder};
    /// use cobre_core::scenario::{InflowModel, CorrelationModel};
    ///
    /// let system = SystemBuilder::new().build().expect("valid system");
    /// let model = InflowModel {
    ///     hydro_id: EntityId(1),
    ///     stage_id: 0,
    ///     mean_m3s: 100.0,
    ///     std_m3s: 10.0,
    ///     ar_coefficients: vec![],
    ///     residual_std_ratio: 1.0,
    ///     annual: None,
    /// };
    /// let updated = system.with_scenario_models(vec![model], CorrelationModel::default());
    /// assert_eq!(updated.inflow_models().len(), 1);
    /// ```
    #[must_use]
    pub fn with_scenario_models(
        mut self,
        inflow_models: Vec<InflowModel>,
        correlation: CorrelationModel,
    ) -> Self {
        self.inflow_models = inflow_models;
        self.correlation = correlation;
        self
    }

    /// Rebuild all lookup indices from the entity collections.
    ///
    /// Sole caller: `From<SystemRepr>` (`Deserialize`'s entry point).
    /// `SystemBuilder::build` needs the same maps earlier, for cross-reference
    /// validation, so it builds them inline instead of calling this.
    #[cfg(feature = "serde")]
    pub(crate) fn rebuild_indices(&mut self) {
        self.bus_index = build_index(&self.buses);
        self.line_index = build_index(&self.lines);
        self.hydro_index = build_index(&self.hydros);
        self.thermal_index = build_index(&self.thermals);
        self.pumping_station_index = build_index(&self.pumping_stations);
        self.contract_index = build_index(&self.contracts);
        self.non_controllable_source_index = build_index(&self.non_controllable_sources);
        self.stage_index = build_stage_index(&self.stages);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ValidationError;
    #[cfg(feature = "serde")]
    use crate::entities::HydroUnitGroup;
    use crate::entities::{ContractType, FillingConfig, HydroGenerationModel, HydroPenalties};
    use chrono::NaiveDate;

    fn make_bus(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("bus-{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![],
            excess_cost: 0.0,
        }
    }

    fn make_line(id: i32, source_bus_id: i32, target_bus_id: i32) -> Line {
        Line {
            id: EntityId(id),
            name: format!("line-{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            source_bus_id: EntityId(source_bus_id),
            target_bus_id: EntityId(target_bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: 100.0,
            reverse_capacity_mw: 100.0,
            losses_percent: 0.0,
            exchange_cost: 0.0,
        }
    }

    fn make_hydro_on_bus(id: i32, bus_id: i32) -> Hydro {
        let zero_penalties = HydroPenalties {
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
        };
        let mut hydro = Hydro {
            id: EntityId(id),
            name: format!("hydro-{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 1.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 1.0,
            unit_groups: Vec::new(),
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_penalties,
        };
        hydro.declare_mirror_unit_group(EntityId(bus_id));
        hydro
    }

    /// Creates a hydro on bus 0. Caller must supply `make_bus(0)`.
    fn make_hydro(id: i32) -> Hydro {
        make_hydro_on_bus(id, 0)
    }

    fn make_thermal_on_bus(id: i32, bus_id: i32) -> Thermal {
        Thermal {
            id: EntityId(id),
            name: format!("thermal-{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: None,
        }
    }

    /// Creates a thermal on bus 0. Caller must supply `make_bus(0)`.
    fn make_thermal(id: i32) -> Thermal {
        make_thermal_on_bus(id, 0)
    }

    fn make_pumping_station_full(
        id: i32,
        bus_id: i32,
        source_hydro_id: i32,
        destination_hydro_id: i32,
    ) -> PumpingStation {
        PumpingStation {
            id: EntityId(id),
            name: format!("ps-{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(bus_id),
            source_hydro_id: EntityId(source_hydro_id),
            destination_hydro_id: EntityId(destination_hydro_id),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s: 0.5,
            min_flow_m3s: 0.0,
            max_flow_m3s: 10.0,
        }
    }

    fn make_pumping_station(id: i32) -> PumpingStation {
        make_pumping_station_full(id, 0, 0, 1)
    }

    fn make_contract_on_bus(id: i32, bus_id: i32) -> EnergyContract {
        EnergyContract {
            id: EntityId(id),
            name: format!("contract-{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(bus_id),
            contract_type: ContractType::Import,
            entry_stage_id: None,
            exit_stage_id: None,
            price_per_mwh: 0.0,
            min_mw: 0.0,
            max_mw: 100.0,
        }
    }

    fn make_contract(id: i32) -> EnergyContract {
        make_contract_on_bus(id, 0)
    }

    fn make_ncs_on_bus(id: i32, bus_id: i32) -> NonControllableSource {
        NonControllableSource {
            id: EntityId(id),
            name: format!("ncs-{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 50.0,
            allow_curtailment: true,
            curtailment_cost: 0.0,
        }
    }

    fn make_ncs(id: i32) -> NonControllableSource {
        make_ncs_on_bus(id, 0)
    }

    #[test]
    fn test_empty_system() {
        let system = SystemBuilder::new().build().expect("empty system is valid");
        assert_eq!(system.n_buses(), 0);
        assert_eq!(system.n_lines(), 0);
        assert_eq!(system.n_hydros(), 0);
        assert_eq!(system.n_thermals(), 0);
        assert_eq!(system.n_pumping_stations(), 0);
        assert_eq!(system.n_contracts(), 0);
        assert_eq!(system.n_non_controllable_sources(), 0);
        assert!(system.buses().is_empty());
        assert!(system.cascade().is_empty());
    }

    #[test]
    fn test_canonical_ordering() {
        let system = SystemBuilder::new()
            .buses(vec![make_bus(2), make_bus(1), make_bus(0)])
            .build()
            .expect("valid system");

        assert_eq!(system.buses()[0].id, EntityId(0));
        assert_eq!(system.buses()[1].id, EntityId(1));
        assert_eq!(system.buses()[2].id, EntityId(2));
    }

    #[test]
    fn test_lookup_by_id() {
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(vec![make_hydro(10), make_hydro(5), make_hydro(20)])
            .build()
            .expect("valid system");

        assert_eq!(system.hydro(EntityId(5)).map(|h| h.id), Some(EntityId(5)));
        assert_eq!(system.hydro(EntityId(10)).map(|h| h.id), Some(EntityId(10)));
        assert_eq!(system.hydro(EntityId(20)).map(|h| h.id), Some(EntityId(20)));
    }

    #[test]
    fn test_lookup_missing_id() {
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(vec![make_hydro(1), make_hydro(2)])
            .build()
            .expect("valid system");

        assert!(system.hydro(EntityId(999)).is_none());
    }

    #[test]
    fn test_count_queries() {
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(1)])
            .lines(vec![make_line(0, 0, 1)])
            .hydros(vec![make_hydro(0), make_hydro(1), make_hydro(2)])
            .thermals(vec![make_thermal(0)])
            .pumping_stations(vec![make_pumping_station(0)])
            .contracts(vec![make_contract(0), make_contract(1)])
            .non_controllable_sources(vec![make_ncs(0)])
            .build()
            .expect("valid system");

        assert_eq!(system.n_buses(), 2);
        assert_eq!(system.n_lines(), 1);
        assert_eq!(system.n_hydros(), 3);
        assert_eq!(system.n_thermals(), 1);
        assert_eq!(system.n_pumping_stations(), 1);
        assert_eq!(system.n_contracts(), 2);
        assert_eq!(system.n_non_controllable_sources(), 1);
    }

    #[test]
    fn test_slice_accessors() {
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(1), make_bus(2)])
            .build()
            .expect("valid system");

        let buses = system.buses();
        assert_eq!(buses.len(), 3);
        assert_eq!(buses[0].id, EntityId(0));
        assert_eq!(buses[1].id, EntityId(1));
        assert_eq!(buses[2].id, EntityId(2));
    }

    #[test]
    fn test_duplicate_id_error() {
        let result = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(0)])
            .build();

        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::DuplicateId {
                entity_type: "Bus",
                id: EntityId(0),
            }
        )));
    }

    #[test]
    fn test_duplicate_stage_id_error() {
        // Without this check build_stage_index silently overwrites the colliding stage.
        let result = SystemBuilder::new()
            .stages(vec![make_stage(0), make_stage(0)])
            .build();

        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::DuplicateId {
                entity_type: "Stage",
                id: EntityId(0),
            }
        )));
    }

    #[test]
    fn test_multiple_duplicate_errors() {
        // Both duplicates must be reported (no short-circuiting on the first).
        let result = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(0)])
            .thermals(vec![make_thermal(5), make_thermal(5)])
            .build();

        assert!(result.is_err());
        let errors = result.unwrap_err();

        let has_bus_dup = errors.iter().any(|e| {
            matches!(
                e,
                ValidationError::DuplicateId {
                    entity_type: "Bus",
                    ..
                }
            )
        });
        let has_thermal_dup = errors.iter().any(|e| {
            matches!(
                e,
                ValidationError::DuplicateId {
                    entity_type: "Thermal",
                    ..
                }
            )
        });
        assert!(has_bus_dup, "expected Bus duplicate error");
        assert!(has_thermal_dup, "expected Thermal duplicate error");
    }

    #[test]
    fn test_send_sync() {
        fn require_send_sync<T: Send + Sync>(_: T) {}
        let system = SystemBuilder::new().build().expect("valid system");
        require_send_sync(system);
    }

    #[test]
    fn test_cascade_accessible() {
        let mut h0 = make_hydro_on_bus(0, 0);
        h0.downstream_id = Some(EntityId(1));
        let mut h1 = make_hydro_on_bus(1, 0);
        h1.downstream_id = Some(EntityId(2));
        let h2 = make_hydro_on_bus(2, 0);

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(vec![h0, h1, h2])
            .build()
            .expect("valid system");

        let order = system.cascade().topological_order();
        assert!(!order.is_empty(), "topological order must be non-empty");
        let pos_0 = order
            .iter()
            .position(|&id| id == EntityId(0))
            .expect("EntityId(0) must be in topological order");
        let pos_2 = order
            .iter()
            .position(|&id| id == EntityId(2))
            .expect("EntityId(2) must be in topological order");
        assert!(pos_0 < pos_2, "EntityId(0) must precede EntityId(2)");
    }

    #[test]
    fn test_network_accessible() {
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(1)])
            .lines(vec![make_line(0, 0, 1)])
            .build()
            .expect("valid system");

        let connections = system.network().bus_lines(EntityId(0));
        assert!(!connections.is_empty(), "bus 0 must have connections");
        assert_eq!(connections[0].line_id, EntityId(0));
    }

    #[test]
    fn test_all_entity_lookups() {
        // Hydros 0 and 1 exist for the pumping station's source/destination refs;
        // hydro 3 is the lookup target.
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(1)])
            .lines(vec![make_line(2, 0, 1)])
            .hydros(vec![
                make_hydro_on_bus(0, 0),
                make_hydro_on_bus(1, 0),
                make_hydro_on_bus(3, 0),
            ])
            .thermals(vec![make_thermal(4)])
            .pumping_stations(vec![make_pumping_station(5)])
            .contracts(vec![make_contract(6)])
            .non_controllable_sources(vec![make_ncs(7)])
            .build()
            .expect("valid system");

        assert!(system.bus(EntityId(1)).is_some());
        assert!(system.line(EntityId(2)).is_some());
        assert!(system.hydro(EntityId(3)).is_some());
        assert!(system.thermal(EntityId(4)).is_some());
        assert!(system.pumping_station(EntityId(5)).is_some());
        assert!(system.contract(EntityId(6)).is_some());
        assert!(system.non_controllable_source(EntityId(7)).is_some());

        assert!(system.bus(EntityId(999)).is_none());
        assert!(system.line(EntityId(999)).is_none());
        assert!(system.hydro(EntityId(999)).is_none());
        assert!(system.thermal(EntityId(999)).is_none());
        assert!(system.pumping_station(EntityId(999)).is_none());
        assert!(system.contract(EntityId(999)).is_none());
        assert!(system.non_controllable_source(EntityId(999)).is_none());
    }

    #[test]
    fn test_default_builder() {
        let system = SystemBuilder::default()
            .build()
            .expect("default builder produces valid empty system");
        assert_eq!(system.n_buses(), 0);
    }

    // ---- Cross-reference validation tests -----------------------------------

    #[test]
    fn test_invalid_downstream_reference() {
        let bus = make_bus(0);
        let mut hydro = make_hydro(1);
        hydro.downstream_id = Some(EntityId(50));

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .build();

        assert!(
            result.is_err(),
            "expected Err for missing downstream reference"
        );
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::InvalidReference {
                    source_entity_type: "Hydro",
                    source_id: EntityId(1),
                    field_name: "downstream_id",
                    referenced_id: EntityId(50),
                    expected_type: "Hydro",
                }
            )),
            "expected InvalidReference for Hydro downstream_id=50, got: {errors:?}"
        );
    }

    #[test]
    fn test_invalid_pumping_station_hydro_refs() {
        let bus = make_bus(0);
        let dest_hydro = make_hydro(1);
        let ps = make_pumping_station_full(10, 0, 77, 1);

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![dest_hydro])
            .pumping_stations(vec![ps])
            .build();

        assert!(
            result.is_err(),
            "expected Err for missing source_hydro_id reference"
        );
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::InvalidReference {
                    source_entity_type: "PumpingStation",
                    source_id: EntityId(10),
                    field_name: "source_hydro_id",
                    referenced_id: EntityId(77),
                    expected_type: "Hydro",
                }
            )),
            "expected InvalidReference for PumpingStation source_hydro_id=77, got: {errors:?}"
        );
    }

    #[test]
    fn test_multiple_invalid_references_collected() {
        // Both errors must be reported (no short-circuiting on the first).
        let line = make_line(1, 99, 0);
        let thermal = make_thermal_on_bus(2, 88);

        let result = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .lines(vec![line])
            .thermals(vec![thermal])
            .build();

        assert!(
            result.is_err(),
            "expected Err for multiple invalid references"
        );
        let errors = result.unwrap_err();

        let has_line_error = errors.iter().any(|e| {
            matches!(
                e,
                ValidationError::InvalidReference {
                    source_entity_type: "Line",
                    field_name: "source_bus_id",
                    referenced_id: EntityId(99),
                    ..
                }
            )
        });
        let has_thermal_error = errors.iter().any(|e| {
            matches!(
                e,
                ValidationError::InvalidReference {
                    source_entity_type: "Thermal",
                    field_name: "bus_id",
                    referenced_id: EntityId(88),
                    ..
                }
            )
        });

        assert!(
            has_line_error,
            "expected Line source_bus_id=99 error, got: {errors:?}"
        );
        assert!(
            has_thermal_error,
            "expected Thermal bus_id=88 error, got: {errors:?}"
        );
        assert!(
            errors.len() >= 2,
            "expected at least 2 errors, got {}: {errors:?}",
            errors.len()
        );
    }

    #[test]
    fn test_valid_cross_references_pass() {
        let bus_0 = make_bus(0);
        let bus_1 = make_bus(1);
        let h0 = make_hydro_on_bus(0, 0);
        let h1 = make_hydro_on_bus(1, 1);
        let mut h2 = make_hydro_on_bus(2, 0);
        h2.downstream_id = Some(EntityId(1));
        let line = make_line(10, 0, 1);
        let thermal = make_thermal_on_bus(20, 0);
        let ps = make_pumping_station_full(30, 0, 0, 1);
        let contract = make_contract_on_bus(40, 1);
        let ncs = make_ncs_on_bus(50, 0);

        let result = SystemBuilder::new()
            .buses(vec![bus_0, bus_1])
            .lines(vec![line])
            .hydros(vec![h0, h1, h2])
            .thermals(vec![thermal])
            .pumping_stations(vec![ps])
            .contracts(vec![contract])
            .non_controllable_sources(vec![ncs])
            .build();

        assert!(
            result.is_ok(),
            "expected Ok for all valid cross-references, got: {:?}",
            result.unwrap_err()
        );
        let system = result.unwrap_or_else(|_| unreachable!());
        assert_eq!(system.n_buses(), 2);
        assert_eq!(system.n_hydros(), 3);
        assert_eq!(system.n_lines(), 1);
        assert_eq!(system.n_thermals(), 1);
        assert_eq!(system.n_pumping_stations(), 1);
        assert_eq!(system.n_contracts(), 1);
        assert_eq!(system.n_non_controllable_sources(), 1);
    }

    // ---- Cascade cycle detection tests --------------------------------------

    #[test]
    fn test_cascade_cycle_detected() {
        let bus = make_bus(0);
        let mut h0 = make_hydro(0);
        h0.downstream_id = Some(EntityId(1));
        let mut h1 = make_hydro(1);
        h1.downstream_id = Some(EntityId(2));
        let mut h2 = make_hydro(2);
        h2.downstream_id = Some(EntityId(0));

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![h0, h1, h2])
            .build();

        assert!(result.is_err(), "expected Err for 3-node cycle");
        let errors = result.unwrap_err();
        let cycle_error = errors
            .iter()
            .find(|e| matches!(e, ValidationError::CascadeCycle { .. }));
        assert!(
            cycle_error.is_some(),
            "expected CascadeCycle error, got: {errors:?}"
        );
        let ValidationError::CascadeCycle { cycle_ids } = cycle_error.unwrap() else {
            unreachable!()
        };
        assert_eq!(
            cycle_ids,
            &[EntityId(0), EntityId(1), EntityId(2)],
            "cycle_ids must be sorted ascending, got: {cycle_ids:?}"
        );
    }

    #[test]
    fn test_cascade_self_loop_detected() {
        let bus = make_bus(0);
        let mut h0 = make_hydro(0);
        h0.downstream_id = Some(EntityId(0));

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![h0])
            .build();

        assert!(result.is_err(), "expected Err for self-loop");
        let errors = result.unwrap_err();
        let has_cycle = errors
            .iter()
            .any(|e| matches!(e, ValidationError::CascadeCycle { cycle_ids } if cycle_ids.contains(&EntityId(0))));
        assert!(
            has_cycle,
            "expected CascadeCycle containing EntityId(0), got: {errors:?}"
        );
    }

    #[test]
    fn test_valid_acyclic_cascade_passes() {
        let bus = make_bus(0);
        let mut h0 = make_hydro(0);
        h0.downstream_id = Some(EntityId(1));
        let mut h1 = make_hydro(1);
        h1.downstream_id = Some(EntityId(2));
        let h2 = make_hydro(2);

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![h0, h1, h2])
            .build();

        assert!(
            result.is_ok(),
            "expected Ok for acyclic cascade, got: {:?}",
            result.unwrap_err()
        );
        let system = result.unwrap_or_else(|_| unreachable!());
        assert_eq!(
            system.cascade().topological_order().len(),
            system.n_hydros(),
            "topological_order must contain all hydros"
        );
    }

    // ---- Filling config validation tests ------------------------------------

    #[test]
    fn test_filling_without_entry_stage() {
        let bus = make_bus(0);
        let mut hydro = make_hydro(1);
        hydro.entry_stage_id = None;
        hydro.filling = Some(FillingConfig {
            start_stage_id: 10,
            filling_min_rate_m3s: 100.0,
        });

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .build();

        assert!(
            result.is_err(),
            "expected Err for filling without entry_stage_id"
        );
        let errors = result.unwrap_err();
        let has_error = errors.iter().any(|e| match e {
            ValidationError::InvalidFillingConfig { hydro_id, reason } => {
                *hydro_id == EntityId(1) && reason.contains("entry_stage_id")
            }
            _ => false,
        });
        assert!(
            has_error,
            "expected InvalidFillingConfig with entry_stage_id reason, got: {errors:?}"
        );
    }

    #[test]
    fn test_filling_negative_rate() {
        // Only a negative rate is rejected; zero is valid (test_filling_zero_rate_accepted).
        let bus = make_bus(0);
        let mut hydro = make_hydro(1);
        hydro.entry_stage_id = Some(10);
        hydro.filling = Some(FillingConfig {
            start_stage_id: 10,
            filling_min_rate_m3s: -5.0,
        });

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .build();

        assert!(
            result.is_err(),
            "expected Err for negative filling_min_rate_m3s"
        );
        let errors = result.unwrap_err();
        let has_error = errors.iter().any(|e| match e {
            ValidationError::InvalidFillingConfig { hydro_id, reason } => {
                *hydro_id == EntityId(1)
                    && reason.contains("filling_min_rate_m3s must be non-negative")
            }
            _ => false,
        });
        assert!(
            has_error,
            "expected InvalidFillingConfig with non-negative rate reason, got: {errors:?}"
        );
    }

    #[test]
    fn test_filling_zero_rate_accepted() {
        // A zero rate is valid: no minimum accumulation is required this stage.
        let bus = make_bus(0);
        let mut hydro = make_hydro(1);
        hydro.entry_stage_id = Some(10);
        hydro.filling = Some(FillingConfig {
            start_stage_id: 9,
            filling_min_rate_m3s: 0.0,
        });

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .build();

        assert!(
            result.is_ok(),
            "expected Ok for zero filling_min_rate_m3s, got: {:?}",
            result.unwrap_err()
        );
    }

    #[test]
    fn test_valid_filling_config_passes() {
        let bus = make_bus(0);
        let mut hydro = make_hydro(1);
        hydro.entry_stage_id = Some(10);
        hydro.filling = Some(FillingConfig {
            start_stage_id: 9,
            filling_min_rate_m3s: 100.0,
        });

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .build();

        assert!(
            result.is_ok(),
            "expected Ok for valid filling config, got: {:?}",
            result.unwrap_err()
        );
    }

    #[test]
    fn test_filling_start_not_before_entry_rejected() {
        // SystemBuilder rejects start_stage_id >= entry_stage_id even when cobre-io
        // is bypassed; an inverted ordering otherwise mis-phases the reservoir.
        let bus = make_bus(0);
        let mut hydro = make_hydro(1);
        hydro.entry_stage_id = Some(5);
        hydro.filling = Some(FillingConfig {
            start_stage_id: 5,
            filling_min_rate_m3s: 100.0,
        });

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .build();

        assert!(
            result.is_err(),
            "expected Err for start_stage_id >= entry_stage_id"
        );
        let errors = result.unwrap_err();
        let has_error = errors.iter().any(|e| match e {
            ValidationError::InvalidFillingConfig { hydro_id, reason } => {
                *hydro_id == EntityId(1) && reason.contains("less than entry_stage_id")
            }
            _ => false,
        });
        assert!(
            has_error,
            "expected InvalidFillingConfig with ordering reason, got: {errors:?}"
        );
    }

    #[test]
    fn test_cascade_cycle_and_invalid_filling_both_reported() {
        let bus = make_bus(0);

        let mut h0 = make_hydro(0);
        h0.downstream_id = Some(EntityId(0));

        let mut h1 = make_hydro(1);
        h1.entry_stage_id = None;
        h1.filling = Some(FillingConfig {
            start_stage_id: 5,
            filling_min_rate_m3s: 50.0,
        });

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![h0, h1])
            .build();

        assert!(result.is_err(), "expected Err for cycle + invalid filling");
        let errors = result.unwrap_err();
        let has_cycle = errors
            .iter()
            .any(|e| matches!(e, ValidationError::CascadeCycle { .. }));
        let has_filling = errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidFillingConfig { .. }));
        assert!(has_cycle, "expected CascadeCycle error, got: {errors:?}");
        assert!(
            has_filling,
            "expected InvalidFillingConfig error, got: {errors:?}"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_system_serde_roundtrip() {
        let bus_a = make_bus(1);
        let bus_b = make_bus(2);
        let hydro = make_hydro_on_bus(10, 1);
        let thermal = make_thermal_on_bus(20, 2);
        let line = make_line(1, 1, 2);

        let system = SystemBuilder::new()
            .buses(vec![bus_a, bus_b])
            .hydros(vec![hydro])
            .thermals(vec![thermal])
            .lines(vec![line])
            .build()
            .expect("valid system");

        let json = serde_json::to_string(&system).unwrap();

        let deserialized: System = serde_json::from_str(&json).unwrap();

        assert_eq!(system.buses(), deserialized.buses());
        assert_eq!(system.hydros(), deserialized.hydros());
        assert_eq!(system.thermals(), deserialized.thermals());
        assert_eq!(system.lines(), deserialized.lines());

        assert_eq!(
            deserialized.bus(EntityId(1)).map(|b| b.id),
            Some(EntityId(1))
        );
        assert_eq!(
            deserialized.hydro(EntityId(10)).map(|h| h.id),
            Some(EntityId(10))
        );
        assert_eq!(
            deserialized.thermal(EntityId(20)).map(|t| t.id),
            Some(EntityId(20))
        );
        assert_eq!(
            deserialized.line(EntityId(1)).map(|l| l.id),
            Some(EntityId(1))
        );
    }

    // ---- Extended System tests ----------------------------------------------

    fn make_stage(id: i32) -> Stage {
        use crate::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
        };
        Stage {
            index: usize::try_from(id.max(0)).unwrap_or(0),
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
                branching_factor: 50,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    #[test]
    fn test_system_backward_compat() {
        let system = SystemBuilder::new().build().expect("empty system is valid");
        assert_eq!(system.n_buses(), 0);
        assert_eq!(system.n_hydros(), 0);
        assert_eq!(system.n_stages(), 0);
        assert!(system.stages().is_empty());
        assert!(system.initial_conditions().storage.is_empty());
        assert!(system.generic_constraints().is_empty());
        assert!(system.inflow_models().is_empty());
        assert!(system.load_models().is_empty());
        assert_eq!(system.penalties().n_stages(), 0);
        assert_eq!(system.bounds().n_stages(), 0);
        assert!(!system.resolved_generic_bounds().is_active(0, 0));
        assert!(
            system
                .resolved_generic_bounds()
                .bounds_for_stage(0, 0)
                .is_empty()
        );
    }

    #[test]
    fn test_system_resolved_generic_bounds_accessor() {
        use crate::model::resolved::GenericConstraintBoundEntry;

        let id_map: HashMap<i32, usize> = [(0, 0), (1, 1)].into_iter().collect();
        let rows = vec![(0i32, 0i32, None::<i32>, Some(100.0f64), None::<f64>)];
        let table = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        let system = SystemBuilder::new()
            .resolved_generic_bounds(table)
            .build()
            .expect("valid system");

        assert!(system.resolved_generic_bounds().is_active(0, 0));
        assert!(!system.resolved_generic_bounds().is_active(1, 0));
        let slice = system.resolved_generic_bounds().bounds_for_stage(0, 0);
        assert_eq!(slice.len(), 1);
        assert_eq!(
            slice[0],
            GenericConstraintBoundEntry {
                block_id: None,
                bound_lower: Some(100.0),
                bound_upper: None,
            }
        );
    }

    #[test]
    fn test_system_with_stages() {
        let s0 = make_stage(0);
        let s1 = make_stage(1);

        let system = SystemBuilder::new()
            .stages(vec![s1.clone(), s0.clone()])
            .build()
            .expect("valid system");

        assert_eq!(system.n_stages(), 2);
        assert_eq!(system.stages()[0].id, 0);
        assert_eq!(system.stages()[1].id, 1);

        let found = system.stage(0).expect("stage 0 must be found");
        assert_eq!(found.id, s0.id);

        let found1 = system.stage(1).expect("stage 1 must be found");
        assert_eq!(found1.id, s1.id);

        assert!(system.stage(99).is_none());
    }

    #[test]
    fn test_system_stage_lookup_by_id() {
        let stages: Vec<Stage> = [0i32, 1, 2].iter().map(|&id| make_stage(id)).collect();

        let system = SystemBuilder::new()
            .stages(stages)
            .build()
            .expect("valid system");

        assert_eq!(system.stage(1).map(|s| s.id), Some(1));
        assert!(system.stage(99).is_none());
    }

    #[test]
    fn test_system_with_initial_conditions() {
        let ic = InitialConditions {
            storage: vec![crate::HydroStorage {
                hydro_id: EntityId(0),
                value_hm3: 15_000.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
            future_anticipated_deliveries: vec![],
        };

        let system = SystemBuilder::new()
            .initial_conditions(ic)
            .build()
            .expect("valid system");

        assert_eq!(system.initial_conditions().storage.len(), 1);
        assert_eq!(system.initial_conditions().storage[0].hydro_id, EntityId(0));
        assert!((system.initial_conditions().storage[0].value_hm3 - 15_000.0).abs() < f64::EPSILON);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_system_serde_roundtrip_with_stages() {
        use crate::temporal::PolicyGraphType;

        let stages = vec![make_stage(0), make_stage(1)];
        let policy_graph = HorizonGraph {
            stage_discount_rate_overrides: std::collections::HashMap::new(),
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: vec![],
            nodes: Vec::new(),
            season_map: None,
        };

        let system = SystemBuilder::new()
            .stages(stages)
            .policy_graph(policy_graph)
            .build()
            .expect("valid system");

        let json = serde_json::to_string(&system).unwrap();
        let deserialized: System = serde_json::from_str(&json).unwrap();

        assert_eq!(system.n_stages(), deserialized.n_stages());
        assert_eq!(system.stages()[0].id, deserialized.stages()[0].id);
        assert_eq!(system.stages()[1].id, deserialized.stages()[1].id);

        assert_eq!(deserialized.stage(0).map(|s| s.id), Some(0));
        assert_eq!(deserialized.stage(1).map(|s| s.id), Some(1));
        assert!(deserialized.stage(99).is_none());

        assert_eq!(
            deserialized.policy_graph().graph_type,
            system.policy_graph().graph_type
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn deserialized_system_lookups_work_without_manual_rebuild() {
        let bus = make_bus(1);
        let hydro = make_hydro_on_bus(10, 1);
        let thermal = make_thermal_on_bus(20, 1);

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .thermals(vec![thermal])
            .build()
            .expect("valid system");

        let bytes = postcard::to_allocvec(&system).unwrap();
        let deserialized: System = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(
            deserialized.bus(EntityId(1)).map(|b| b.id),
            Some(EntityId(1))
        );
        assert_eq!(
            deserialized.hydro(EntityId(10)).map(|h| h.id),
            Some(EntityId(10))
        );
        assert_eq!(
            deserialized.thermal(EntityId(20)).map(|t| t.id),
            Some(EntityId(20))
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_system_postcard_roundtrip_preserves_unit_groups() {
        let bus0 = make_bus(0);
        let bus4 = make_bus(4);
        let bus9 = make_bus(9);

        let no_groups_hydro = make_hydro_on_bus(1, 0);
        let mut two_groups_hydro = make_hydro_on_bus(2, 0);
        two_groups_hydro.unit_groups = vec![
            HydroUnitGroup {
                id: EntityId(3),
                name: "Group A".to_string(),
                bus_id: EntityId(4),
                min_generation_mw: 10.0,
                max_generation_mw: 20.0,
                min_turbined_m3s: 30.0,
                max_turbined_m3s: 40.0,
            },
            HydroUnitGroup {
                id: EntityId(7),
                name: "Group B".to_string(),
                bus_id: EntityId(9),
                min_generation_mw: 50.0,
                max_generation_mw: 60.0,
                min_turbined_m3s: 70.0,
                max_turbined_m3s: 80.0,
            },
        ];

        let system = SystemBuilder::new()
            .buses(vec![bus0, bus4, bus9])
            .hydros(vec![no_groups_hydro, two_groups_hydro.clone()])
            .build()
            .expect("valid system");

        let bytes = postcard::to_allocvec(&system).unwrap();
        let deserialized: System = postcard::from_bytes(&bytes).unwrap();

        let decoded_no_groups = deserialized
            .hydro(EntityId(1))
            .expect("hydro 1 must round-trip");
        assert_eq!(decoded_no_groups.unit_groups.len(), 1);
        assert_eq!(decoded_no_groups.unit_groups[0].id, EntityId(0));
        assert_eq!(decoded_no_groups.unit_groups[0].bus_id, EntityId(0));

        let decoded_two_groups = deserialized
            .hydro(EntityId(2))
            .expect("hydro 2 must round-trip");
        assert_eq!(decoded_two_groups.unit_groups, two_groups_hydro.unit_groups);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn fully_populated_system_survives_postcard_roundtrip_intact() {
        use crate::{
            AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
            ConstraintExpression, ContractBlockBounds, CorrelationEntity, CorrelationGroup,
            CorrelationProfile, CorrelationScheduleEntry, DeficitSegment, HydroBlockBounds,
            HydroPastDefluence, HydroStageBounds, HydroStagePenalties, HydroStorage,
            LineBlockBounds, LineStagePenalties, LinearTerm, NcsStagePenalties,
            PenaltiesCountsSpec, PenaltiesDefaults, PolicyGraphType, PumpingBlockBounds,
            RecentObservation, SlackConfig, ThermalBlockBounds, ThermalStageBounds, Transition,
            VariableRef,
        };

        let bus1 = {
            let mut b = make_bus(1);
            b.deficit_segments = vec![DeficitSegment {
                depth_mw: Some(50.0),
                cost_per_mwh: 3000.0,
            }];
            b.excess_cost = 12.5;
            b
        };
        let bus2 = {
            let mut b = make_bus(2);
            b.deficit_segments = vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 5000.0,
            }];
            b.excess_cost = 7.25;
            b
        };

        let mut hydro1 = make_hydro_on_bus(1, 1);
        hydro1.downstream_id = Some(EntityId(2));
        hydro1.travel_time_hours = Some(6.0);
        hydro1.entry_stage_id = Some(0);
        hydro1.unit_groups = vec![
            HydroUnitGroup {
                id: EntityId(10),
                name: "Group A".to_string(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 0.4,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 0.4,
            },
            HydroUnitGroup {
                id: EntityId(20),
                name: "Group B".to_string(),
                bus_id: EntityId(2),
                min_generation_mw: 0.0,
                max_generation_mw: 0.6,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 0.6,
            },
        ];
        let hydro2 = make_hydro_on_bus(2, 2);

        let thermal1 = make_thermal_on_bus(1, 1);
        let line1 = make_line(1, 1, 2);
        let pump1 = make_pumping_station_full(1, 1, 1, 2);
        let contract1 = make_contract_on_bus(1, 2);
        let ncs1 = make_ncs_on_bus(1, 2);

        let stage0 = make_stage(0);
        let stage1 = make_stage(1);

        let policy_graph = HorizonGraph {
            stage_discount_rate_overrides: std::collections::HashMap::new(),
            graph_type: PolicyGraphType::Cyclic,
            annual_discount_rate: 0.08,
            transitions: vec![
                Transition {
                    source_id: 0,
                    target_id: 1,
                    probability: 1.0,
                    annual_discount_rate_override: None,
                },
                Transition {
                    source_id: 1,
                    target_id: 0,
                    probability: 1.0,
                    annual_discount_rate_override: Some(0.05),
                },
            ],
            nodes: Vec::new(),
            season_map: None,
        };

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 2,
                n_buses: 2,
                n_lines: 1,
                n_ncs: 1,
                n_stages: 2,
            },
            &PenaltiesDefaults {
                hydro: HydroStagePenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.02,
                    turbined_cost: 0.03,
                    storage_violation_below_cost: 1000.0,
                    filling_target_violation_cost: 5000.0,
                    turbined_violation_below_cost: 500.0,
                    outflow_violation_below_cost: 400.0,
                    outflow_violation_above_cost: 300.0,
                    generation_violation_below_cost: 200.0,
                    evaporation_violation_cost: 150.0,
                    water_withdrawal_violation_cost: 100.0,
                    water_withdrawal_violation_pos_cost: 100.0,
                    water_withdrawal_violation_neg_cost: 100.0,
                    evaporation_violation_pos_cost: 150.0,
                    evaporation_violation_neg_cost: 150.0,
                    inflow_nonnegativity_cost: 1000.0,
                },
                bus: BusStagePenalties { excess_cost: 250.0 },
                line: LineStagePenalties {
                    exchange_cost: 12.5,
                },
                ncs: NcsStagePenalties {
                    curtailment_cost: 33.0,
                },
            },
        );

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 1,
                n_lines: 1,
                n_pumping: 1,
                n_contracts: 1,
                n_stages: 2,
                k_max: 1,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 10.0,
                    max_storage_hm3: 500.0,
                    filling_min_rate_m3s: 3.0,
                    water_withdrawal_m3s: 1.5,
                },
                hydro_block: HydroBlockBounds {
                    min_turbined_m3s: 1.0,
                    max_turbined_m3s: 300.0,
                    min_outflow_m3s: 2.0,
                    max_outflow_m3s: Some(600.0),
                    min_generation_mw: 5.0,
                    max_generation_mw: 200.0,
                    max_diversion_m3s: Some(20.0),
                    ..Default::default()
                },
                thermal: ThermalStageBounds { cost_per_mwh: 85.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 10.0,
                    max_generation_mw: 150.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 300.0,
                    reverse_mw: 250.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.5,
                    max_flow_m3s: 40.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 90.0,
                    price_per_mwh: 95.0,
                },
            },
        );

        let resolved_generic_bounds = ResolvedGenericConstraintBounds::new(
            &std::collections::HashMap::from([(1i32, 0usize)]),
            vec![(1i32, 0i32, None::<i32>, Some(777.0f64), None::<f64>)].into_iter(),
        );

        let mut resolved_load_factors = ResolvedLoadFactors::new(2, 2, 1);
        resolved_load_factors.set(0, 0, 0, 0.92);
        resolved_load_factors.set(1, 1, 0, 1.08);

        let resolved_ncs_bounds = ResolvedNcsBounds::new(1, 2, &[45.0]);

        let mut resolved_ncs_factors = ResolvedNcsFactors::new(1, 2, 1);
        resolved_ncs_factors.set(0, 0, 0, 0.77);

        let inflow_models = vec![
            InflowModel {
                hydro_id: EntityId(1),
                stage_id: 0,
                mean_m3s: 150.0,
                std_m3s: 30.0,
                ar_coefficients: vec![0.45, 0.22],
                residual_std_ratio: 0.85,
                annual: None,
            },
            InflowModel {
                hydro_id: EntityId(2),
                stage_id: 1,
                mean_m3s: 90.0,
                std_m3s: 15.0,
                ar_coefficients: vec![0.3],
                residual_std_ratio: 0.7,
                annual: None,
            },
        ];

        let load_models = vec![
            LoadModel {
                bus_id: EntityId(1),
                stage_id: 0,
                mean_mw: 320.5,
                std_mw: 45.0,
            },
            LoadModel {
                bus_id: EntityId(2),
                stage_id: 1,
                mean_mw: 210.0,
                std_mw: 30.0,
            },
        ];

        let ncs_models = vec![NcsModel {
            ncs_id: EntityId(1),
            stage_id: 0,
            mean: 0.5,
            std: 0.1,
        }];

        let correlation = {
            let mut profiles = std::collections::BTreeMap::new();
            profiles.insert(
                "default".to_string(),
                CorrelationProfile {
                    groups: vec![CorrelationGroup {
                        name: "All".to_string(),
                        entities: vec![
                            CorrelationEntity {
                                entity_type: "inflow".to_string(),
                                id: EntityId(1),
                            },
                            CorrelationEntity {
                                entity_type: "inflow".to_string(),
                                id: EntityId(2),
                            },
                        ],
                        matrix: vec![vec![1.0, 0.3], vec![0.3, 1.0]],
                    }],
                },
            );
            CorrelationModel {
                method: "spectral".to_string(),
                profiles,
                schedule: vec![CorrelationScheduleEntry {
                    stage_id: 0,
                    profile_name: "default".to_string(),
                }],
            }
        };

        let initial_conditions = InitialConditions {
            storage: vec![
                HydroStorage {
                    hydro_id: EntityId(1),
                    value_hm3: 12_000.0,
                },
                HydroStorage {
                    hydro_id: EntityId(2),
                    value_hm3: 8_500.0,
                },
            ],
            filling_storage: vec![HydroStorage {
                hydro_id: EntityId(1),
                value_hm3: 50.0,
            }],
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: EntityId(1),
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                value_mw: 100.0,
            }],
            recent_observations: vec![RecentObservation {
                hydro_id: EntityId(1),
                start_date: NaiveDate::from_ymd_opt(2023, 12, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2023, 12, 15).unwrap(),
                value_m3s: 480.0,
            }],
            past_defluences: vec![HydroPastDefluence {
                hydro_id: EntityId(1),
                start_date: NaiveDate::from_ymd_opt(2023, 11, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2023, 12, 1).unwrap(),
                value_m3s: 320.0,
            }],
            future_anticipated_deliveries: vec![],
        };

        let generic_constraints = vec![GenericConstraint {
            id: EntityId(1),
            name: "gc-full".to_string(),
            description: Some("full population coverage".to_string()),
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroGeneration {
                        hydro_id: EntityId(1),
                        block_id: None,
                        bus_id: None,
                    },
                )],
            },
            slack: SlackConfig {
                enabled: true,
                penalty: Some(2500.0),
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        }];

        let inflow_history = vec![
            InflowHistoryRow {
                hydro_id: EntityId(1),
                start_date: NaiveDate::from_ymd_opt(2000, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2000, 2, 1).unwrap(),
                value_m3s: 500.0,
            },
            InflowHistoryRow {
                hydro_id: EntityId(2),
                start_date: NaiveDate::from_ymd_opt(2000, 2, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2000, 3, 1).unwrap(),
                value_m3s: 420.0,
            },
        ];

        let external_scenarios = vec![ExternalScenarioRow {
            stage_id: 0,
            scenario_id: 2,
            hydro_id: EntityId(1),
            value_m3s: 320.5,
        }];

        let external_load_scenarios = vec![ExternalLoadRow {
            stage_id: 0,
            scenario_id: 2,
            bus_id: EntityId(1),
            value_mw: 150.0,
        }];

        let external_ncs_scenarios = vec![ExternalNcsRow {
            stage_id: 1,
            scenario_id: 0,
            ncs_id: EntityId(1),
            value: 0.85,
        }];

        let system = SystemBuilder::new()
            .buses(vec![bus1, bus2])
            .lines(vec![line1])
            .hydros(vec![hydro1, hydro2])
            .thermals(vec![thermal1])
            .pumping_stations(vec![pump1])
            .contracts(vec![contract1])
            .non_controllable_sources(vec![ncs1])
            .stages(vec![stage0, stage1])
            .policy_graph(policy_graph)
            .penalties(penalties)
            .bounds(bounds)
            .resolved_generic_bounds(resolved_generic_bounds)
            .resolved_load_factors(resolved_load_factors)
            .resolved_ncs_bounds(resolved_ncs_bounds)
            .resolved_ncs_factors(resolved_ncs_factors)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .ncs_models(ncs_models)
            .correlation(correlation)
            .initial_conditions(initial_conditions)
            .generic_constraints(generic_constraints)
            .inflow_history(inflow_history)
            .external_scenarios(external_scenarios)
            .external_load_scenarios(external_load_scenarios)
            .external_ncs_scenarios(external_ncs_scenarios)
            .build()
            .expect("fully populated, cross-reference-consistent system must be valid");

        let bytes = postcard::to_allocvec(&system).unwrap();
        let deserialized: System = postcard::from_bytes(&bytes).unwrap();

        // Failure means SystemRepr drifted from System: field order or field set.
        assert_eq!(system, deserialized);
    }

    // ---- inflow_history and external_scenarios field tests ------------------

    #[test]
    fn test_system_inflow_history_defaults_empty() {
        let system = SystemBuilder::new().build().expect("valid system");
        assert!(
            system.inflow_history().is_empty(),
            "inflow_history must default to empty"
        );
    }

    #[test]
    fn test_system_inflow_history_stores_rows() {
        let row1 = InflowHistoryRow {
            hydro_id: EntityId(1),
            start_date: NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date"),
            end_date: NaiveDate::from_ymd_opt(2000, 2, 1).expect("valid date"),
            value_m3s: 500.0,
        };
        let row2 = InflowHistoryRow {
            hydro_id: EntityId(1),
            start_date: NaiveDate::from_ymd_opt(2000, 2, 1).expect("valid date"),
            end_date: NaiveDate::from_ymd_opt(2000, 3, 1).expect("valid date"),
            value_m3s: 420.0,
        };

        let system = SystemBuilder::new()
            .inflow_history(vec![row1.clone(), row2.clone()])
            .build()
            .expect("valid system");

        assert_eq!(system.inflow_history().len(), 2);
        assert_eq!(system.inflow_history()[0], row1);
        assert_eq!(system.inflow_history()[1], row2);
    }

    #[test]
    fn test_system_external_scenarios_defaults_empty() {
        let system = SystemBuilder::new().build().expect("valid system");
        assert!(
            system.external_scenarios().is_empty(),
            "external_scenarios must default to empty"
        );
    }

    #[test]
    fn test_system_external_scenarios_stores_rows() {
        let row = ExternalScenarioRow {
            stage_id: 0,
            scenario_id: 2,
            hydro_id: EntityId(5),
            value_m3s: 320.5,
        };

        let system = SystemBuilder::new()
            .external_scenarios(vec![row.clone()])
            .build()
            .expect("valid system");

        assert_eq!(system.external_scenarios().len(), 1);
        assert_eq!(system.external_scenarios()[0], row);
    }
}
