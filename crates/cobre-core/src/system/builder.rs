//! `SystemBuilder` — canonical-order assembly and validation of a [`System`].

use std::collections::HashSet;

use super::System;
use super::validate::{
    CrossRefEntities, build_index, build_stage_index, check_duplicate_stages, check_duplicates,
    validate_cross_references, validate_filling_configs,
};
use crate::{
    Bus, CascadeTopology, CorrelationModel, EnergyContract, EntityId, ExternalLoadRow,
    ExternalNcsRow, ExternalScenarioRow, GenericConstraint, Hydro, InflowHistoryRow, InflowModel,
    InitialConditions, Line, LoadModel, NcsModel, NetworkTopology, NonControllableSource,
    PolicyGraph, PumpingStation, ResolvedBounds, ResolvedExchangeFactors,
    ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors,
    ResolvedPenalties, Stage, Thermal, ValidationError,
};

/// Builder for constructing a validated, immutable [`System`].
///
/// All entity collections default to empty; supply only the ones you need.
///
/// # Examples
///
/// ```
/// use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
///
/// let system = SystemBuilder::new()
///     .buses(vec![
///         Bus { id: EntityId(2), name: "B".to_string(), deficit_segments: vec![], excess_cost: 0.0 },
///         Bus { id: EntityId(1), name: "A".to_string(), deficit_segments: vec![], excess_cost: 0.0 },
///     ])
///     .build()
///     .expect("valid system");
///
/// // Canonical ordering: id=1 comes before id=2.
/// assert_eq!(system.buses()[0].id, EntityId(1));
/// assert_eq!(system.buses()[1].id, EntityId(2));
/// ```
pub struct SystemBuilder {
    buses: Vec<Bus>,
    lines: Vec<Line>,
    hydros: Vec<Hydro>,
    thermals: Vec<Thermal>,
    pumping_stations: Vec<PumpingStation>,
    contracts: Vec<EnergyContract>,
    non_controllable_sources: Vec<NonControllableSource>,
    stages: Vec<Stage>,
    policy_graph: PolicyGraph,
    penalties: ResolvedPenalties,
    bounds: ResolvedBounds,
    resolved_generic_bounds: ResolvedGenericConstraintBounds,
    resolved_load_factors: ResolvedLoadFactors,
    resolved_exchange_factors: ResolvedExchangeFactors,
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

impl Default for SystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemBuilder {
    /// Create a new builder with every collection empty and every field at its default.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buses: Vec::new(),
            lines: Vec::new(),
            hydros: Vec::new(),
            thermals: Vec::new(),
            pumping_stations: Vec::new(),
            contracts: Vec::new(),
            non_controllable_sources: Vec::new(),
            stages: Vec::new(),
            policy_graph: PolicyGraph::default(),
            penalties: ResolvedPenalties::empty(),
            bounds: ResolvedBounds::empty(),
            resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
            resolved_load_factors: ResolvedLoadFactors::empty(),
            resolved_exchange_factors: ResolvedExchangeFactors::empty(),
            resolved_ncs_bounds: ResolvedNcsBounds::empty(),
            resolved_ncs_factors: ResolvedNcsFactors::empty(),
            inflow_models: Vec::new(),
            load_models: Vec::new(),
            ncs_models: Vec::new(),
            correlation: CorrelationModel::default(),
            initial_conditions: InitialConditions::default(),
            generic_constraints: Vec::new(),
            inflow_history: Vec::new(),
            external_scenarios: Vec::new(),
            external_load_scenarios: Vec::new(),
            external_ncs_scenarios: Vec::new(),
        }
    }

    /// Set the bus collection.
    #[must_use]
    pub fn buses(mut self, buses: Vec<Bus>) -> Self {
        self.buses = buses;
        self
    }

    /// Set the line collection.
    #[must_use]
    pub fn lines(mut self, lines: Vec<Line>) -> Self {
        self.lines = lines;
        self
    }

    /// Set the hydro plant collection.
    #[must_use]
    pub fn hydros(mut self, hydros: Vec<Hydro>) -> Self {
        self.hydros = hydros;
        self
    }

    /// Set the thermal plant collection.
    #[must_use]
    pub fn thermals(mut self, thermals: Vec<Thermal>) -> Self {
        self.thermals = thermals;
        self
    }

    /// Set the pumping station collection.
    #[must_use]
    pub fn pumping_stations(mut self, stations: Vec<PumpingStation>) -> Self {
        self.pumping_stations = stations;
        self
    }

    /// Set the energy contract collection.
    #[must_use]
    pub fn contracts(mut self, contracts: Vec<EnergyContract>) -> Self {
        self.contracts = contracts;
        self
    }

    /// Set the non-controllable source collection.
    #[must_use]
    pub fn non_controllable_sources(mut self, sources: Vec<NonControllableSource>) -> Self {
        self.non_controllable_sources = sources;
        self
    }

    /// Set the stage collection (study and pre-study stages).
    ///
    /// Stages are sorted by `id` in [`build`](Self::build) to canonical order.
    #[must_use]
    pub fn stages(mut self, stages: Vec<Stage>) -> Self {
        self.stages = stages;
        self
    }

    /// Set the policy graph.
    #[must_use]
    pub fn policy_graph(mut self, policy_graph: PolicyGraph) -> Self {
        self.policy_graph = policy_graph;
        self
    }

    /// Set the pre-resolved penalty table.
    #[must_use]
    pub fn penalties(mut self, penalties: ResolvedPenalties) -> Self {
        self.penalties = penalties;
        self
    }

    /// Set the pre-resolved bounds table.
    #[must_use]
    pub fn bounds(mut self, bounds: ResolvedBounds) -> Self {
        self.bounds = bounds;
        self
    }

    /// Set the pre-resolved generic constraint RHS bound table.
    #[must_use]
    pub fn resolved_generic_bounds(
        mut self,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
    ) -> Self {
        self.resolved_generic_bounds = resolved_generic_bounds;
        self
    }

    /// Set the pre-resolved per-block load scaling factors.
    #[must_use]
    pub fn resolved_load_factors(mut self, resolved_load_factors: ResolvedLoadFactors) -> Self {
        self.resolved_load_factors = resolved_load_factors;
        self
    }

    /// Set the pre-resolved per-block exchange capacity factors.
    #[must_use]
    pub fn resolved_exchange_factors(
        mut self,
        resolved_exchange_factors: ResolvedExchangeFactors,
    ) -> Self {
        self.resolved_exchange_factors = resolved_exchange_factors;
        self
    }

    /// Set the pre-resolved per-stage NCS available generation bounds.
    #[must_use]
    pub fn resolved_ncs_bounds(mut self, resolved_ncs_bounds: ResolvedNcsBounds) -> Self {
        self.resolved_ncs_bounds = resolved_ncs_bounds;
        self
    }

    /// Set the pre-resolved per-block NCS generation scaling factors.
    #[must_use]
    pub fn resolved_ncs_factors(mut self, resolved_ncs_factors: ResolvedNcsFactors) -> Self {
        self.resolved_ncs_factors = resolved_ncs_factors;
        self
    }

    /// Set the PAR(p) inflow model collection.
    #[must_use]
    pub fn inflow_models(mut self, inflow_models: Vec<InflowModel>) -> Self {
        self.inflow_models = inflow_models;
        self
    }

    /// Set the load model collection.
    #[must_use]
    pub fn load_models(mut self, load_models: Vec<LoadModel>) -> Self {
        self.load_models = load_models;
        self
    }

    /// Set the NCS availability noise model collection.
    #[must_use]
    pub fn ncs_models(mut self, ncs_models: Vec<NcsModel>) -> Self {
        self.ncs_models = ncs_models;
        self
    }

    /// Set the correlation model.
    #[must_use]
    pub fn correlation(mut self, correlation: CorrelationModel) -> Self {
        self.correlation = correlation;
        self
    }

    /// Set the initial conditions.
    #[must_use]
    pub fn initial_conditions(mut self, initial_conditions: InitialConditions) -> Self {
        self.initial_conditions = initial_conditions;
        self
    }

    /// Set the generic constraint collection.
    ///
    /// Constraints are sorted by `id` in [`build`](Self::build) to canonical order.
    #[must_use]
    pub fn generic_constraints(mut self, generic_constraints: Vec<GenericConstraint>) -> Self {
        self.generic_constraints = generic_constraints;
        self
    }

    /// Set the raw historical inflow observations; rows must be sorted by
    /// `(hydro_id, date)` ascending.
    #[must_use]
    pub fn inflow_history(mut self, rows: Vec<InflowHistoryRow>) -> Self {
        self.inflow_history = rows;
        self
    }

    /// Set the raw external inflow scenario rows; rows must be sorted by
    /// `(stage_id, scenario_id, hydro_id)` ascending.
    #[must_use]
    pub fn external_scenarios(mut self, rows: Vec<ExternalScenarioRow>) -> Self {
        self.external_scenarios = rows;
        self
    }

    /// Set the raw external load scenario rows; rows must be sorted by
    /// `(stage_id, scenario_id, bus_id)` ascending.
    #[must_use]
    pub fn external_load_scenarios(mut self, rows: Vec<ExternalLoadRow>) -> Self {
        self.external_load_scenarios = rows;
        self
    }

    /// Set the raw external NCS scenario rows; rows must be sorted by
    /// `(stage_id, scenario_id, ncs_id)` ascending.
    #[must_use]
    pub fn external_ncs_scenarios(mut self, rows: Vec<ExternalNcsRow>) -> Self {
        self.external_ncs_scenarios = rows;
        self
    }

    /// Sort every collection into canonical [`EntityId`] order, validate, and assemble
    /// the immutable [`System`]. All validation errors are collected before returning —
    /// no short-circuiting on the first error.
    ///
    /// # Errors
    ///
    /// Returns `Err(Vec<ValidationError>)` if:
    /// - Duplicate IDs are detected in any entity collection or in the stage collection.
    /// - Any cross-reference field refers to an entity ID that does not exist.
    /// - The hydro cascade graph contains a cycle.
    /// - Any hydro filling configuration is invalid (non-positive inflow or missing
    ///   `entry_stage_id`).
    // Rationale: sort, duplicate/cross-ref/cycle checks, and `System` assembly share
    // one `errors` accumulator and the intermediate index maps; splitting them would
    // thread those through every call and lose the fail-fast-on-duplicates short-circuit.
    #[allow(clippy::too_many_lines)]
    pub fn build(mut self) -> Result<System, Vec<ValidationError>> {
        self.buses.sort_by_key(|e| e.id.0);
        self.lines.sort_by_key(|e| e.id.0);
        self.hydros.sort_by_key(|e| e.id.0);
        self.thermals.sort_by_key(|e| e.id.0);
        self.pumping_stations.sort_by_key(|e| e.id.0);
        self.contracts.sort_by_key(|e| e.id.0);
        self.non_controllable_sources.sort_by_key(|e| e.id.0);
        self.stages.sort_by_key(|s| s.id);
        self.generic_constraints.sort_by_key(|c| c.id.0);

        let mut errors: Vec<ValidationError> = Vec::new();
        check_duplicates(&self.buses, "Bus", &mut errors);
        check_duplicates(&self.lines, "Line", &mut errors);
        check_duplicates(&self.hydros, "Hydro", &mut errors);
        check_duplicates(&self.thermals, "Thermal", &mut errors);
        check_duplicates(&self.pumping_stations, "PumpingStation", &mut errors);
        check_duplicates(&self.contracts, "EnergyContract", &mut errors);
        check_duplicates(
            &self.non_controllable_sources,
            "NonControllableSource",
            &mut errors,
        );
        check_duplicate_stages(&self.stages, &mut errors);

        if !errors.is_empty() {
            return Err(errors);
        }

        let bus_index = build_index(&self.buses);
        let line_index = build_index(&self.lines);
        let hydro_index = build_index(&self.hydros);
        let thermal_index = build_index(&self.thermals);
        let pumping_station_index = build_index(&self.pumping_stations);
        let contract_index = build_index(&self.contracts);
        let non_controllable_source_index = build_index(&self.non_controllable_sources);

        validate_cross_references(
            &CrossRefEntities {
                lines: &self.lines,
                hydros: &self.hydros,
                thermals: &self.thermals,
                pumping_stations: &self.pumping_stations,
                contracts: &self.contracts,
                non_controllable_sources: &self.non_controllable_sources,
            },
            &bus_index,
            &hydro_index,
            &mut errors,
        );

        if !errors.is_empty() {
            return Err(errors);
        }

        let cascade = CascadeTopology::build(&self.hydros);

        if cascade.topological_order().len() < self.hydros.len() {
            let in_topo: HashSet<EntityId> = cascade.topological_order().iter().copied().collect();
            let mut cycle_ids: Vec<EntityId> = self
                .hydros
                .iter()
                .map(|h| h.id)
                .filter(|id| !in_topo.contains(id))
                .collect();
            cycle_ids.sort_by_key(|id| id.0);
            errors.push(ValidationError::CascadeCycle { cycle_ids });
        }

        validate_filling_configs(&self.hydros, &mut errors);

        if !errors.is_empty() {
            return Err(errors);
        }

        let network = NetworkTopology::build(
            &self.buses,
            &self.lines,
            &self.hydros,
            &self.thermals,
            &self.non_controllable_sources,
            &self.contracts,
            &self.pumping_stations,
        );

        let stage_index = build_stage_index(&self.stages);

        Ok(System {
            buses: self.buses,
            lines: self.lines,
            hydros: self.hydros,
            thermals: self.thermals,
            pumping_stations: self.pumping_stations,
            contracts: self.contracts,
            non_controllable_sources: self.non_controllable_sources,
            bus_index,
            line_index,
            hydro_index,
            thermal_index,
            pumping_station_index,
            contract_index,
            non_controllable_source_index,
            cascade,
            network,
            stages: self.stages,
            policy_graph: self.policy_graph,
            stage_index,
            penalties: self.penalties,
            bounds: self.bounds,
            resolved_generic_bounds: self.resolved_generic_bounds,
            resolved_load_factors: self.resolved_load_factors,
            resolved_exchange_factors: self.resolved_exchange_factors,
            resolved_ncs_bounds: self.resolved_ncs_bounds,
            resolved_ncs_factors: self.resolved_ncs_factors,
            inflow_models: self.inflow_models,
            load_models: self.load_models,
            ncs_models: self.ncs_models,
            correlation: self.correlation,
            initial_conditions: self.initial_conditions,
            generic_constraints: self.generic_constraints,
            inflow_history: self.inflow_history,
            external_scenarios: self.external_scenarios,
            external_load_scenarios: self.external_load_scenarios,
            external_ncs_scenarios: self.external_ncs_scenarios,
        })
    }
}
