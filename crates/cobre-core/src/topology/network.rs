//! Resolved electrical transmission network topology.
//!
//! `NetworkTopology` holds the validated adjacency structure derived from the
//! `Line` entity collection. It is built during case loading after all `Bus` and
//! `Line` entities have been validated and their cross-references verified.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::entities::Bus;
use crate::{
    EnergyContract, EntityId, Hydro, Line, NonControllableSource, PumpingStation, Thermal,
};

/// A line connection from a bus perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BusLineConnection {
    /// The line's entity ID.
    pub line_id: EntityId,
    /// True when this bus is the line's source (direct flow); false at the target (reverse flow).
    pub is_source: bool,
}

/// Generator entity IDs at a bus. All lists are in canonical ascending-`i32`
/// order for determinism.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BusGenerators {
    /// IDs of hydros with at least one unit group at this bus. A plant with
    /// groups on several buses is listed under each; a plant with several
    /// groups on this bus is listed once.
    pub hydro_ids: Vec<EntityId>,
    /// Thermal plant IDs.
    pub thermal_ids: Vec<EntityId>,
    /// Non-controllable source IDs.
    pub ncs_ids: Vec<EntityId>,
}

/// Load/demand entity IDs at a bus. All lists are in canonical ascending-`i32`
/// order for determinism.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BusLoads {
    /// Energy contract IDs.
    pub contract_ids: Vec<EntityId>,
    /// Pumping station IDs.
    pub pumping_station_ids: Vec<EntityId>,
}

/// Resolved transmission network topology for buses and lines.
///
/// Built from entity collections during System construction and immutable thereafter.
// Rationale: the `bus_` prefix groups fields keyed by bus identity and mirrors the
// public accessor names.
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NetworkTopology {
    bus_lines: HashMap<EntityId, Vec<BusLineConnection>>,

    bus_generators: HashMap<EntityId, BusGenerators>,

    bus_loads: HashMap<EntityId, BusLoads>,
}

static DEFAULT_BUS_GENERATORS: OnceLock<BusGenerators> = OnceLock::new();

static DEFAULT_BUS_LOADS: OnceLock<BusLoads> = OnceLock::new();

impl NetworkTopology {
    /// Build network topology from entity collections.
    ///
    /// All entity slices are assumed to be in canonical ID order. Does not
    /// validate bus existence; validation is a separate layer.
    #[must_use]
    pub fn build(
        buses: &[Bus],
        lines: &[Line],
        hydros: &[Hydro],
        thermals: &[Thermal],
        non_controllable_sources: &[NonControllableSource],
        contracts: &[EnergyContract],
        pumping_stations: &[PumpingStation],
    ) -> Self {
        let mut bus_lines: HashMap<EntityId, Vec<BusLineConnection>> = HashMap::new();
        let mut bus_generators: HashMap<EntityId, BusGenerators> = HashMap::new();
        let mut bus_loads: HashMap<EntityId, BusLoads> = HashMap::new();

        // TODO: use `buses` for disconnected-bus validation (ValidationError::DisconnectedBus)
        let _ = buses;

        for line in lines {
            bus_lines
                .entry(line.source_bus_id)
                .or_default()
                .push(BusLineConnection {
                    line_id: line.id,
                    is_source: true,
                });
            bus_lines
                .entry(line.target_bus_id)
                .or_default()
                .push(BusLineConnection {
                    line_id: line.id,
                    is_source: false,
                });
        }
        for connections in bus_lines.values_mut() {
            connections.sort_by_key(|c| c.line_id.0);
        }

        for hydro in hydros {
            for (i, group) in hydro.unit_groups.iter().enumerate() {
                if hydro.unit_groups[..i]
                    .iter()
                    .any(|g| g.bus_id == group.bus_id)
                {
                    continue;
                }
                bus_generators
                    .entry(group.bus_id)
                    .or_default()
                    .hydro_ids
                    .push(hydro.id);
            }
        }

        for thermal in thermals {
            bus_generators
                .entry(thermal.bus_id)
                .or_default()
                .thermal_ids
                .push(thermal.id);
        }

        for ncs in non_controllable_sources {
            bus_generators
                .entry(ncs.bus_id)
                .or_default()
                .ncs_ids
                .push(ncs.id);
        }

        for generators in bus_generators.values_mut() {
            generators.hydro_ids.sort_by_key(|id| id.0);
            generators.thermal_ids.sort_by_key(|id| id.0);
            generators.ncs_ids.sort_by_key(|id| id.0);
        }

        for contract in contracts {
            bus_loads
                .entry(contract.bus_id)
                .or_default()
                .contract_ids
                .push(contract.id);
        }

        for station in pumping_stations {
            bus_loads
                .entry(station.bus_id)
                .or_default()
                .pumping_station_ids
                .push(station.id);
        }

        for loads in bus_loads.values_mut() {
            loads.contract_ids.sort_by_key(|id| id.0);
            loads.pumping_station_ids.sort_by_key(|id| id.0);
        }

        Self {
            bus_lines,
            bus_generators,
            bus_loads,
        }
    }

    /// Returns the lines connected to a bus, or an empty slice if none.
    #[must_use]
    pub fn bus_lines(&self, bus_id: EntityId) -> &[BusLineConnection] {
        self.bus_lines.get(&bus_id).map_or(&[], Vec::as_slice)
    }

    /// Returns the generators connected to a bus, or an empty `BusGenerators` if none.
    #[must_use]
    pub fn bus_generators(&self, bus_id: EntityId) -> &BusGenerators {
        self.bus_generators
            .get(&bus_id)
            .unwrap_or_else(|| DEFAULT_BUS_GENERATORS.get_or_init(BusGenerators::default))
    }

    /// Returns the loads connected to a bus, or an empty `BusLoads` if none.
    #[must_use]
    pub fn bus_loads(&self, bus_id: EntityId) -> &BusLoads {
        self.bus_loads
            .get(&bus_id)
            .unwrap_or_else(|| DEFAULT_BUS_LOADS.get_or_init(BusLoads::default))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        BusSpec, ContractSpec, HydroSpec, LineSpec, NcsSpec, PumpingSpec, ThermalSpec,
        UnitGroupSpec, make_bus, make_contract, make_hydro, make_line, make_ncs,
        make_pumping_station, make_thermal, make_unit_group,
    };

    #[test]
    fn test_empty_network() {
        let topo = NetworkTopology::build(&[], &[], &[], &[], &[], &[], &[]);

        assert_eq!(topo.bus_lines(EntityId(0)), &[]);
        assert!(topo.bus_generators(EntityId(0)).hydro_ids.is_empty());
        assert!(topo.bus_generators(EntityId(0)).thermal_ids.is_empty());
        assert!(topo.bus_generators(EntityId(0)).ncs_ids.is_empty());
        assert!(topo.bus_loads(EntityId(0)).contract_ids.is_empty());
        assert!(topo.bus_loads(EntityId(0)).pumping_station_ids.is_empty());
    }

    #[test]
    fn test_single_line() {
        let buses = vec![
            make_bus(BusSpec {
                id: 0,
                ..Default::default()
            }),
            make_bus(BusSpec {
                id: 1,
                ..Default::default()
            }),
        ];
        let lines = vec![make_line(LineSpec {
            id: 0,
            source_bus_id: 0,
            target_bus_id: 1,
            ..Default::default()
        })];
        let topo = NetworkTopology::build(&buses, &lines, &[], &[], &[], &[], &[]);

        let conns_0 = topo.bus_lines(EntityId(0));
        assert_eq!(conns_0.len(), 1);
        assert_eq!(conns_0[0].line_id, EntityId(0));
        assert!(conns_0[0].is_source);

        let conns_1 = topo.bus_lines(EntityId(1));
        assert_eq!(conns_1.len(), 1);
        assert_eq!(conns_1[0].line_id, EntityId(0));
        assert!(!conns_1[0].is_source);
    }

    #[test]
    fn test_multiple_lines_same_bus() {
        let buses = vec![
            make_bus(BusSpec {
                id: 0,
                ..Default::default()
            }),
            make_bus(BusSpec {
                id: 1,
                ..Default::default()
            }),
            make_bus(BusSpec {
                id: 2,
                ..Default::default()
            }),
            make_bus(BusSpec {
                id: 3,
                ..Default::default()
            }),
        ];
        let lines = vec![
            make_line(LineSpec {
                id: 0,
                source_bus_id: 0,
                target_bus_id: 1,
                ..Default::default()
            }),
            make_line(LineSpec {
                id: 1,
                source_bus_id: 0,
                target_bus_id: 2,
                ..Default::default()
            }),
            make_line(LineSpec {
                id: 2,
                source_bus_id: 0,
                target_bus_id: 3,
                ..Default::default()
            }),
        ];
        let topo = NetworkTopology::build(&buses, &lines, &[], &[], &[], &[], &[]);

        let conns = topo.bus_lines(EntityId(0));
        assert_eq!(conns.len(), 3);
        assert!(conns.iter().all(|c| c.is_source));
        assert_eq!(conns[0].line_id, EntityId(0));
        assert_eq!(conns[1].line_id, EntityId(1));
        assert_eq!(conns[2].line_id, EntityId(2));
    }

    #[test]
    fn test_generators_per_bus() {
        let buses = vec![
            make_bus(BusSpec {
                id: 0,
                ..Default::default()
            }),
            make_bus(BusSpec {
                id: 1,
                ..Default::default()
            }),
        ];
        let hydros = vec![
            make_hydro(HydroSpec {
                id: 0,
                bus_id: 0,
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 1,
                bus_id: 0,
                ..Default::default()
            }),
        ];
        let thermals = vec![make_thermal(ThermalSpec {
            id: 0,
            bus_id: 0,
            ..Default::default()
        })];
        let ncs = vec![make_ncs(NcsSpec {
            id: 0,
            bus_id: 1,
            ..Default::default()
        })];
        let topo = NetworkTopology::build(&buses, &[], &hydros, &thermals, &ncs, &[], &[]);

        let gen0 = topo.bus_generators(EntityId(0));
        assert_eq!(gen0.hydro_ids.len(), 2);
        assert_eq!(gen0.thermal_ids.len(), 1);
        assert!(gen0.ncs_ids.is_empty());

        let gen1 = topo.bus_generators(EntityId(1));
        assert!(gen1.hydro_ids.is_empty());
        assert!(gen1.thermal_ids.is_empty());
        assert_eq!(gen1.ncs_ids.len(), 1);
        assert_eq!(gen1.ncs_ids[0], EntityId(0));
    }

    #[test]
    fn test_loads_per_bus() {
        let buses = vec![make_bus(BusSpec {
            id: 0,
            ..Default::default()
        })];
        let contracts = vec![make_contract(ContractSpec {
            id: 0,
            bus_id: 0,
            ..Default::default()
        })];
        let stations = vec![make_pumping_station(PumpingSpec {
            id: 0,
            bus_id: 0,
            ..Default::default()
        })];
        let topo = NetworkTopology::build(&buses, &[], &[], &[], &[], &contracts, &stations);

        let loads0 = topo.bus_loads(EntityId(0));
        assert_eq!(loads0.contract_ids.len(), 1);
        assert_eq!(loads0.contract_ids[0], EntityId(0));
        assert_eq!(loads0.pumping_station_ids.len(), 1);
        assert_eq!(loads0.pumping_station_ids[0], EntityId(0));
    }

    #[test]
    fn test_bus_no_connections() {
        let buses = vec![make_bus(BusSpec {
            id: 0,
            ..Default::default()
        })];
        let topo = NetworkTopology::build(&buses, &[], &[], &[], &[], &[], &[]);

        assert_eq!(topo.bus_lines(EntityId(0)), &[]);
        let generators = topo.bus_generators(EntityId(0));
        assert!(generators.hydro_ids.is_empty());
        assert!(generators.thermal_ids.is_empty());
        assert!(generators.ncs_ids.is_empty());
        let loads = topo.bus_loads(EntityId(0));
        assert!(loads.contract_ids.is_empty());
        assert!(loads.pumping_station_ids.is_empty());
    }

    #[test]
    fn test_deterministic_ordering() {
        let buses = vec![make_bus(BusSpec {
            id: 0,
            ..Default::default()
        })];
        let hydros = vec![
            make_hydro(HydroSpec {
                id: 5,
                bus_id: 0,
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 3,
                bus_id: 0,
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 1,
                bus_id: 0,
                ..Default::default()
            }),
        ];
        let thermals = vec![
            make_thermal(ThermalSpec {
                id: 4,
                bus_id: 0,
                ..Default::default()
            }),
            make_thermal(ThermalSpec {
                id: 2,
                bus_id: 0,
                ..Default::default()
            }),
        ];
        let contracts = vec![
            make_contract(ContractSpec {
                id: 10,
                bus_id: 0,
                ..Default::default()
            }),
            make_contract(ContractSpec {
                id: 7,
                bus_id: 0,
                ..Default::default()
            }),
        ];
        let topo = NetworkTopology::build(&buses, &[], &hydros, &thermals, &[], &contracts, &[]);

        let generators = topo.bus_generators(EntityId(0));
        assert_eq!(
            generators.hydro_ids,
            vec![EntityId(1), EntityId(3), EntityId(5)]
        );
        assert_eq!(generators.thermal_ids, vec![EntityId(2), EntityId(4)]);

        let loads = topo.bus_loads(EntityId(0));
        assert_eq!(loads.contract_ids, vec![EntityId(7), EntityId(10)]);
    }

    #[test]
    fn test_bus_generators_same_bus_groups_collapse_to_one_hydro_id() {
        let buses = vec![make_bus(BusSpec {
            id: 0,
            ..Default::default()
        })];
        let mut hydro = make_hydro(HydroSpec {
            id: 10,
            bus_id: 0,
            ..Default::default()
        });
        hydro.unit_groups = vec![
            make_unit_group(UnitGroupSpec {
                id: 0,
                bus_id: 0,
                ..Default::default()
            }),
            make_unit_group(UnitGroupSpec {
                id: 1,
                bus_id: 0,
                ..Default::default()
            }),
            make_unit_group(UnitGroupSpec {
                id: 2,
                bus_id: 0,
                ..Default::default()
            }),
        ];
        let topo = NetworkTopology::build(&buses, &[], &[hydro], &[], &[], &[], &[]);

        assert_eq!(
            topo.bus_generators(EntityId(0)).hydro_ids,
            vec![EntityId(10)]
        );
    }

    #[test]
    fn test_bus_generators_group_bus_listing_is_declaration_order_invariant() {
        let buses = vec![
            make_bus(BusSpec {
                id: 1,
                ..Default::default()
            }),
            make_bus(BusSpec {
                id: 2,
                ..Default::default()
            }),
        ];

        let mut hydro_fwd = make_hydro(HydroSpec {
            id: 10,
            bus_id: 1,
            ..Default::default()
        });
        hydro_fwd.unit_groups = vec![
            make_unit_group(UnitGroupSpec {
                id: 2,
                bus_id: 1,
                ..Default::default()
            }),
            make_unit_group(UnitGroupSpec {
                id: 5,
                bus_id: 2,
                ..Default::default()
            }),
        ];
        let topo_fwd = NetworkTopology::build(&buses, &[], &[hydro_fwd], &[], &[], &[], &[]);

        let mut hydro_rev = make_hydro(HydroSpec {
            id: 10,
            bus_id: 1,
            ..Default::default()
        });
        hydro_rev.unit_groups = vec![
            make_unit_group(UnitGroupSpec {
                id: 5,
                bus_id: 2,
                ..Default::default()
            }),
            make_unit_group(UnitGroupSpec {
                id: 2,
                bus_id: 1,
                ..Default::default()
            }),
        ];
        let topo_rev = NetworkTopology::build(&buses, &[], &[hydro_rev], &[], &[], &[], &[]);

        assert_eq!(
            topo_fwd.bus_generators(EntityId(1)),
            topo_rev.bus_generators(EntityId(1))
        );
        assert_eq!(
            topo_fwd.bus_generators(EntityId(2)),
            topo_rev.bus_generators(EntityId(2))
        );
        assert_eq!(
            topo_fwd.bus_generators(EntityId(1)).hydro_ids,
            vec![EntityId(10)]
        );
        assert_eq!(
            topo_fwd.bus_generators(EntityId(2)).hydro_ids,
            vec![EntityId(10)]
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_topology_serde_roundtrip_network() {
        let buses = vec![
            make_bus(BusSpec {
                id: 0,
                ..Default::default()
            }),
            make_bus(BusSpec {
                id: 1,
                ..Default::default()
            }),
        ];
        let lines = vec![make_line(LineSpec {
            id: 0,
            source_bus_id: 0,
            target_bus_id: 1,
            ..Default::default()
        })];
        let hydros = vec![make_hydro(HydroSpec {
            id: 0,
            bus_id: 0,
            ..Default::default()
        })];
        let thermals = vec![make_thermal(ThermalSpec {
            id: 0,
            bus_id: 1,
            ..Default::default()
        })];
        let topo = NetworkTopology::build(&buses, &lines, &hydros, &thermals, &[], &[], &[]);
        let json = serde_json::to_string(&topo).unwrap();
        let deserialized: NetworkTopology = serde_json::from_str(&json).unwrap();
        assert_eq!(topo, deserialized);
    }
}
