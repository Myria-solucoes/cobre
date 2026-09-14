//! Integration tests exercising the full `SystemBuilder::build()` pipeline
//! through the public API only.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::too_many_lines
)]

use chrono::NaiveDate;
use cobre_core::test_support::{
    BusSpec, ContractSpec, HydroSpec, LineSpec, NcsSpec, PumpingSpec, ThermalSpec, UnitGroupSpec,
    make_bus, make_contract, make_hydro, make_line, make_ncs, make_pumping_station, make_thermal,
    make_unit_group,
};
use cobre_core::{
    DeficitSegment, DiversionChannel, EntityId, FillingConfig, SystemBuilder, ValidationError,
};

#[test]
fn test_declaration_order_invariance() {
    let buses_fwd = vec![
        make_bus(BusSpec {
            id: 1,
            name: format!("bus-{}", 1),
            deficit_segments: vec![DeficitSegment {
                depth_mw: Some(100.0),
                cost_per_mwh: 500.0,
            }],
            ..Default::default()
        }),
        make_bus(BusSpec {
            id: 2,
            name: format!("bus-{}", 2),
            deficit_segments: vec![DeficitSegment {
                depth_mw: Some(100.0),
                cost_per_mwh: 500.0,
            }],
            ..Default::default()
        }),
    ];
    let lines_fwd = vec![make_line(LineSpec {
        id: 10,
        name: format!("line-{}", 10),
        source_bus_id: 1,
        target_bus_id: 2,
        direct_capacity_mw: 200.0,
        reverse_capacity_mw: 200.0,
        ..Default::default()
    })];
    let hydros_fwd = vec![
        make_hydro(HydroSpec {
            id: 20,
            name: format!("hydro-{}", 20),
            bus_id: 1,
            downstream_id: Some(21),
            max_storage_hm3: 100.0,
            max_turbined_m3s: 500.0,
            max_generation_mw: 450.0,
            ..Default::default()
        }),
        make_hydro(HydroSpec {
            id: 21,
            name: format!("hydro-{}", 21),
            bus_id: 1,
            max_storage_hm3: 100.0,
            max_turbined_m3s: 500.0,
            max_generation_mw: 450.0,
            ..Default::default()
        }),
    ];
    let thermals_fwd = vec![make_thermal(ThermalSpec {
        id: 30,
        name: format!("thermal-{}", 30),
        bus_id: 2,
        cost_per_mwh: 80.0,
        max_generation_mw: 300.0,
        ..Default::default()
    })];
    let pumping_fwd = vec![make_pumping_station(PumpingSpec {
        id: 40,
        name: format!("ps-{}", 40),
        bus_id: 2,
        source_hydro_id: 20,
        destination_hydro_id: 21,
        max_flow_m3s: 20.0,
        ..Default::default()
    })];
    let contracts_fwd = vec![make_contract(ContractSpec {
        id: 50,
        name: format!("contract-{}", 50),
        bus_id: 1,
        price_per_mwh: 50.0,
        max_mw: 150.0,
        ..Default::default()
    })];
    let ncs_fwd = vec![make_ncs(NcsSpec {
        id: 60,
        name: format!("ncs-{}", 60),
        bus_id: 2,
        max_generation_mw: 80.0,
        curtailment_cost: 5.0,
        ..Default::default()
    })];

    let system_fwd = SystemBuilder::new()
        .buses(buses_fwd)
        .lines(lines_fwd)
        .hydros(hydros_fwd)
        .thermals(thermals_fwd)
        .pumping_stations(pumping_fwd)
        .contracts(contracts_fwd)
        .non_controllable_sources(ncs_fwd)
        .build()
        .expect("forward-order system must be valid");

    let buses_rev = vec![
        make_bus(BusSpec {
            id: 2,
            name: format!("bus-{}", 2),
            deficit_segments: vec![DeficitSegment {
                depth_mw: Some(100.0),
                cost_per_mwh: 500.0,
            }],
            ..Default::default()
        }),
        make_bus(BusSpec {
            id: 1,
            name: format!("bus-{}", 1),
            deficit_segments: vec![DeficitSegment {
                depth_mw: Some(100.0),
                cost_per_mwh: 500.0,
            }],
            ..Default::default()
        }),
    ];
    let lines_rev = vec![make_line(LineSpec {
        id: 10,
        name: format!("line-{}", 10),
        source_bus_id: 1,
        target_bus_id: 2,
        direct_capacity_mw: 200.0,
        reverse_capacity_mw: 200.0,
        ..Default::default()
    })];
    let hydros_rev = vec![
        make_hydro(HydroSpec {
            id: 21,
            name: format!("hydro-{}", 21),
            bus_id: 1,
            max_storage_hm3: 100.0,
            max_turbined_m3s: 500.0,
            max_generation_mw: 450.0,
            ..Default::default()
        }),
        make_hydro(HydroSpec {
            id: 20,
            name: format!("hydro-{}", 20),
            bus_id: 1,
            downstream_id: Some(21),
            max_storage_hm3: 100.0,
            max_turbined_m3s: 500.0,
            max_generation_mw: 450.0,
            ..Default::default()
        }),
    ];
    let thermals_rev = vec![make_thermal(ThermalSpec {
        id: 30,
        name: format!("thermal-{}", 30),
        bus_id: 2,
        cost_per_mwh: 80.0,
        max_generation_mw: 300.0,
        ..Default::default()
    })];
    let pumping_rev = vec![make_pumping_station(PumpingSpec {
        id: 40,
        name: format!("ps-{}", 40),
        bus_id: 2,
        source_hydro_id: 20,
        destination_hydro_id: 21,
        max_flow_m3s: 20.0,
        ..Default::default()
    })];
    let contracts_rev = vec![make_contract(ContractSpec {
        id: 50,
        name: format!("contract-{}", 50),
        bus_id: 1,
        price_per_mwh: 50.0,
        max_mw: 150.0,
        ..Default::default()
    })];
    let ncs_rev = vec![make_ncs(NcsSpec {
        id: 60,
        name: format!("ncs-{}", 60),
        bus_id: 2,
        max_generation_mw: 80.0,
        curtailment_cost: 5.0,
        ..Default::default()
    })];

    let system_rev = SystemBuilder::new()
        .buses(buses_rev)
        .lines(lines_rev)
        .hydros(hydros_rev)
        .thermals(thermals_rev)
        .pumping_stations(pumping_rev)
        .contracts(contracts_rev)
        .non_controllable_sources(ncs_rev)
        .build()
        .expect("reverse-order system must be valid");

    assert_eq!(
        system_fwd, system_rev,
        "System must be identical regardless of input entity ordering"
    );
}

#[test]
fn test_realistic_multi_entity_system() {
    let mut hydro_10 = make_hydro(HydroSpec {
        id: 10,
        name: format!("hydro-{}", 10),
        bus_id: 1,
        downstream_id: Some(12),
        max_storage_hm3: 100.0,
        max_turbined_m3s: 500.0,
        max_generation_mw: 450.0,
        ..Default::default()
    });
    let mut hydro_11 = make_hydro(HydroSpec {
        id: 11,
        name: format!("hydro-{}", 11),
        bus_id: 2,
        downstream_id: Some(12),
        max_storage_hm3: 100.0,
        max_turbined_m3s: 500.0,
        max_generation_mw: 450.0,
        ..Default::default()
    });
    let hydro_12 = make_hydro(HydroSpec {
        id: 12,
        name: format!("hydro-{}", 12),
        bus_id: 3,
        max_storage_hm3: 100.0,
        max_turbined_m3s: 500.0,
        max_generation_mw: 450.0,
        ..Default::default()
    });

    hydro_10.name = "upstream-A".to_string();
    hydro_11.name = "upstream-B".to_string();

    let system = SystemBuilder::new()
        .buses(vec![
            make_bus(BusSpec {
                id: 1,
                name: format!("bus-{}", 1),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: Some(100.0),
                    cost_per_mwh: 500.0,
                }],
                ..Default::default()
            }),
            make_bus(BusSpec {
                id: 2,
                name: format!("bus-{}", 2),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: Some(100.0),
                    cost_per_mwh: 500.0,
                }],
                ..Default::default()
            }),
            make_bus(BusSpec {
                id: 3,
                name: format!("bus-{}", 3),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: Some(100.0),
                    cost_per_mwh: 500.0,
                }],
                ..Default::default()
            }),
        ])
        .lines(vec![
            make_line(LineSpec {
                id: 100,
                name: format!("line-{}", 100),
                source_bus_id: 1,
                target_bus_id: 2,
                direct_capacity_mw: 200.0,
                reverse_capacity_mw: 200.0,
                ..Default::default()
            }),
            make_line(LineSpec {
                id: 101,
                name: format!("line-{}", 101),
                source_bus_id: 2,
                target_bus_id: 3,
                direct_capacity_mw: 200.0,
                reverse_capacity_mw: 200.0,
                ..Default::default()
            }),
        ])
        .hydros(vec![hydro_10, hydro_11, hydro_12])
        .thermals(vec![
            make_thermal(ThermalSpec {
                id: 20,
                name: format!("thermal-{}", 20),
                bus_id: 1,
                cost_per_mwh: 80.0,
                max_generation_mw: 300.0,
                ..Default::default()
            }),
            make_thermal(ThermalSpec {
                id: 21,
                name: format!("thermal-{}", 21),
                bus_id: 3,
                cost_per_mwh: 80.0,
                max_generation_mw: 300.0,
                ..Default::default()
            }),
        ])
        .pumping_stations(vec![make_pumping_station(PumpingSpec {
            id: 30,
            name: format!("ps-{}", 30),
            bus_id: 2,
            source_hydro_id: 10,
            destination_hydro_id: 12,
            max_flow_m3s: 20.0,
            ..Default::default()
        })])
        .contracts(vec![make_contract(ContractSpec {
            id: 40,
            name: format!("contract-{}", 40),
            bus_id: 1,
            price_per_mwh: 50.0,
            max_mw: 150.0,
            ..Default::default()
        })])
        .non_controllable_sources(vec![make_ncs(NcsSpec {
            id: 50,
            name: format!("ncs-{}", 50),
            bus_id: 3,
            max_generation_mw: 80.0,
            curtailment_cost: 5.0,
            ..Default::default()
        })])
        .build()
        .expect("realistic multi-entity system must be valid");

    assert_eq!(system.n_buses(), 3);
    assert_eq!(system.n_lines(), 2);
    assert_eq!(system.n_hydros(), 3);
    assert_eq!(system.n_thermals(), 2);
    assert_eq!(system.n_pumping_stations(), 1);
    assert_eq!(system.n_contracts(), 1);
    assert_eq!(system.n_non_controllable_sources(), 1);

    assert!(system.bus(EntityId(1)).is_some());
    assert!(system.bus(EntityId(2)).is_some());
    assert!(system.bus(EntityId(3)).is_some());
    assert!(system.bus(EntityId(999)).is_none());

    assert!(system.line(EntityId(100)).is_some());
    assert!(system.line(EntityId(101)).is_some());

    let h10 = system.hydro(EntityId(10)).expect("hydro 10 must exist");
    assert_eq!(h10.name, "upstream-A");
    assert!(system.hydro(EntityId(11)).is_some());
    assert!(system.hydro(EntityId(12)).is_some());
    assert!(system.hydro(EntityId(999)).is_none());

    assert!(system.thermal(EntityId(20)).is_some());
    assert!(system.thermal(EntityId(21)).is_some());

    assert!(system.pumping_station(EntityId(30)).is_some());
    assert!(system.contract(EntityId(40)).is_some());
    assert!(system.non_controllable_source(EntityId(50)).is_some());

    let buses = system.buses();
    assert_eq!(buses[0].id, EntityId(1));
    assert_eq!(buses[1].id, EntityId(2));
    assert_eq!(buses[2].id, EntityId(3));

    // Equal dates sort by id ascending.
    let hydros = system.hydros();
    assert_eq!(hydros[0].id, EntityId(10));
    assert_eq!(hydros[1].id, EntityId(11));
    assert_eq!(hydros[2].id, EntityId(12));

    let cascade = system.cascade();
    assert_eq!(cascade.len(), 3);

    assert_eq!(cascade.downstream(EntityId(10)), Some(EntityId(12)));
    assert_eq!(cascade.downstream(EntityId(11)), Some(EntityId(12)));
    assert_eq!(cascade.downstream(EntityId(12)), None);
    let upstream_12 = cascade.upstream(EntityId(12));
    assert_eq!(upstream_12.len(), 2);
    assert_eq!(upstream_12[0], EntityId(10));
    assert_eq!(upstream_12[1], EntityId(11));
    assert!(cascade.is_headwater(EntityId(10)));
    assert!(cascade.is_headwater(EntityId(11)));
    assert!(!cascade.is_headwater(EntityId(12)));
    assert!(!cascade.is_terminal(EntityId(10)));
    assert!(!cascade.is_terminal(EntityId(11)));
    assert!(cascade.is_terminal(EntityId(12)));

    let topo = cascade.topological_order();
    let pos_10 = topo
        .iter()
        .position(|&id| id == EntityId(10))
        .expect("10 in topo");
    let pos_11 = topo
        .iter()
        .position(|&id| id == EntityId(11))
        .expect("11 in topo");
    let pos_12 = topo
        .iter()
        .position(|&id| id == EntityId(12))
        .expect("12 in topo");
    assert!(
        pos_10 < pos_12,
        "hydro 10 must precede hydro 12 in topo order"
    );
    assert!(
        pos_11 < pos_12,
        "hydro 11 must precede hydro 12 in topo order"
    );

    let network = system.network();

    let conns_bus1 = network.bus_lines(EntityId(1));
    assert_eq!(conns_bus1.len(), 1);
    assert_eq!(conns_bus1[0].line_id, EntityId(100));
    assert!(conns_bus1[0].is_source);

    let conns_bus2 = network.bus_lines(EntityId(2));
    assert_eq!(conns_bus2.len(), 2);
    assert_eq!(conns_bus2[0].line_id, EntityId(100));
    assert!(!conns_bus2[0].is_source);
    assert_eq!(conns_bus2[1].line_id, EntityId(101));
    assert!(conns_bus2[1].is_source);

    let conns_bus3 = network.bus_lines(EntityId(3));
    assert_eq!(conns_bus3.len(), 1);
    assert_eq!(conns_bus3[0].line_id, EntityId(101));
    assert!(!conns_bus3[0].is_source);

    let gen1 = network.bus_generators(EntityId(1));
    assert_eq!(gen1.hydro_ids, vec![EntityId(10)]);
    assert_eq!(gen1.thermal_ids, vec![EntityId(20)]);
    assert!(gen1.ncs_ids.is_empty());

    let gen2 = network.bus_generators(EntityId(2));
    assert_eq!(gen2.hydro_ids, vec![EntityId(11)]);
    assert!(gen2.thermal_ids.is_empty());
    assert!(gen2.ncs_ids.is_empty());

    let gen3 = network.bus_generators(EntityId(3));
    assert_eq!(gen3.hydro_ids, vec![EntityId(12)]);
    assert_eq!(gen3.thermal_ids, vec![EntityId(21)]);
    assert_eq!(gen3.ncs_ids, vec![EntityId(50)]);

    let loads1 = network.bus_loads(EntityId(1));
    assert_eq!(loads1.contract_ids, vec![EntityId(40)]);
    assert!(loads1.pumping_station_ids.is_empty());

    let loads2 = network.bus_loads(EntityId(2));
    assert!(loads2.contract_ids.is_empty());
    assert_eq!(loads2.pumping_station_ids, vec![EntityId(30)]);
}

#[test]
fn test_hydro_with_groups_on_multiple_buses_lists_under_each() {
    let mut hydro = make_hydro(HydroSpec {
        id: 10,
        name: format!("hydro-{}", 10),
        bus_id: 1,
        max_storage_hm3: 100.0,
        max_turbined_m3s: 500.0,
        max_generation_mw: 450.0,
        ..Default::default()
    });
    hydro.unit_groups = vec![
        make_unit_group(UnitGroupSpec {
            id: 0,
            name: format!("group-{}", 0),
            bus_id: 1,
            max_generation_mw: 450.0,
            max_turbined_m3s: 500.0,
            ..Default::default()
        }),
        make_unit_group(UnitGroupSpec {
            id: 1,
            name: format!("group-{}", 1),
            bus_id: 2,
            max_generation_mw: 450.0,
            max_turbined_m3s: 500.0,
            ..Default::default()
        }),
    ];

    let system = SystemBuilder::new()
        .buses(vec![
            make_bus(BusSpec {
                id: 1,
                name: format!("bus-{}", 1),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: Some(100.0),
                    cost_per_mwh: 500.0,
                }],
                ..Default::default()
            }),
            make_bus(BusSpec {
                id: 2,
                name: format!("bus-{}", 2),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: Some(100.0),
                    cost_per_mwh: 500.0,
                }],
                ..Default::default()
            }),
        ])
        .hydros(vec![hydro])
        .build()
        .expect("hydro with groups on two buses must be valid");

    let network = system.network();
    assert_eq!(
        network.bus_generators(EntityId(1)).hydro_ids,
        vec![EntityId(10)]
    );
    assert_eq!(
        network.bus_generators(EntityId(2)).hydro_ids,
        vec![EntityId(10)]
    );
}

#[test]
fn test_hydro_with_same_bus_groups_appears_once() {
    let mut hydro = make_hydro(HydroSpec {
        id: 10,
        name: format!("hydro-{}", 10),
        bus_id: 1,
        max_storage_hm3: 100.0,
        max_turbined_m3s: 500.0,
        max_generation_mw: 450.0,
        ..Default::default()
    });
    hydro.unit_groups = vec![
        make_unit_group(UnitGroupSpec {
            id: 0,
            name: format!("group-{}", 0),
            bus_id: 1,
            max_generation_mw: 450.0,
            max_turbined_m3s: 500.0,
            ..Default::default()
        }),
        make_unit_group(UnitGroupSpec {
            id: 1,
            name: format!("group-{}", 1),
            bus_id: 1,
            max_generation_mw: 450.0,
            max_turbined_m3s: 500.0,
            ..Default::default()
        }),
        make_unit_group(UnitGroupSpec {
            id: 2,
            name: format!("group-{}", 2),
            bus_id: 1,
            max_generation_mw: 450.0,
            max_turbined_m3s: 500.0,
            ..Default::default()
        }),
    ];

    let system = SystemBuilder::new()
        .buses(vec![make_bus(BusSpec {
            id: 1,
            name: format!("bus-{}", 1),
            deficit_segments: vec![DeficitSegment {
                depth_mw: Some(100.0),
                cost_per_mwh: 500.0,
            }],
            ..Default::default()
        })])
        .hydros(vec![hydro])
        .build()
        .expect("hydro with three same-bus groups must be valid");

    let network = system.network();
    assert_eq!(
        network.bus_generators(EntityId(1)).hydro_ids,
        vec![EntityId(10)]
    );
}

#[test]
fn test_hydro_on_unknown_bus_is_accepted_groups_own_the_bus() {
    let mut hydro = make_hydro(HydroSpec {
        id: 1,
        name: format!("hydro-{}", 1),
        bus_id: 999,
        max_storage_hm3: 100.0,
        max_turbined_m3s: 500.0,
        max_generation_mw: 450.0,
        ..Default::default()
    });
    hydro.unit_groups = vec![make_unit_group(UnitGroupSpec {
        id: 0,
        name: format!("group-{}", 0),
        bus_id: 1,
        max_generation_mw: 450.0,
        max_turbined_m3s: 500.0,
        ..Default::default()
    })];

    let result = SystemBuilder::new()
        .buses(vec![make_bus(BusSpec {
            id: 1,
            name: format!("bus-{}", 1),
            deficit_segments: vec![DeficitSegment {
                depth_mw: Some(100.0),
                cost_per_mwh: 500.0,
            }],
            ..Default::default()
        })])
        .hydros(vec![hydro])
        .build();

    assert!(
        result.is_ok(),
        "a plant's own bus argument naming a nonexistent bus must not fail \
         validation — only its groups' own buses are validated; got: {:?}",
        result.err()
    );
}

#[test]
fn test_cascade_cycle_rejected() {
    let hydro_1 = make_hydro(HydroSpec {
        id: 1,
        name: format!("hydro-{}", 1),
        bus_id: 1,
        downstream_id: Some(2),
        max_storage_hm3: 100.0,
        max_turbined_m3s: 500.0,
        max_generation_mw: 450.0,
        ..Default::default()
    });
    let hydro_2 = make_hydro(HydroSpec {
        id: 2,
        name: format!("hydro-{}", 2),
        bus_id: 1,
        downstream_id: Some(1),
        max_storage_hm3: 100.0,
        max_turbined_m3s: 500.0,
        max_generation_mw: 450.0,
        ..Default::default()
    });

    let result = SystemBuilder::new()
        .buses(vec![make_bus(BusSpec {
            id: 1,
            name: format!("bus-{}", 1),
            deficit_segments: vec![DeficitSegment {
                depth_mw: Some(100.0),
                cost_per_mwh: 500.0,
            }],
            ..Default::default()
        })])
        .hydros(vec![hydro_1, hydro_2])
        .build();

    assert!(result.is_err(), "cyclic cascade must fail validation");

    let errors = result.unwrap_err();
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, ValidationError::CascadeCycle { .. })),
        "expected CascadeCycle error; got: {errors:?}"
    );
}

#[test]
fn test_large_order_invariance() {
    let make_system = |bus_order: Vec<i32>, hydro_order: Vec<(i32, Option<i32>)>| {
        let buses = bus_order
            .into_iter()
            .map(|id| {
                make_bus(BusSpec {
                    id,
                    name: format!("bus-{id}"),
                    deficit_segments: vec![DeficitSegment {
                        depth_mw: Some(100.0),
                        cost_per_mwh: 500.0,
                    }],
                    ..Default::default()
                })
            })
            .collect();
        let hydros = hydro_order
            .into_iter()
            .map(|(id, ds)| {
                make_hydro(HydroSpec {
                    id,
                    name: format!("hydro-{id}"),
                    bus_id: 1,
                    downstream_id: ds,
                    max_storage_hm3: 100.0,
                    max_turbined_m3s: 500.0,
                    max_generation_mw: 450.0,
                    ..Default::default()
                })
            })
            .collect();

        SystemBuilder::new()
            .buses(buses)
            .lines(vec![
                make_line(LineSpec {
                    id: 10,
                    name: format!("line-{}", 10),
                    source_bus_id: 1,
                    target_bus_id: 2,
                    direct_capacity_mw: 200.0,
                    reverse_capacity_mw: 200.0,
                    ..Default::default()
                }),
                make_line(LineSpec {
                    id: 11,
                    name: format!("line-{}", 11),
                    source_bus_id: 2,
                    target_bus_id: 3,
                    direct_capacity_mw: 200.0,
                    reverse_capacity_mw: 200.0,
                    ..Default::default()
                }),
            ])
            .hydros(hydros)
            .thermals(vec![
                make_thermal(ThermalSpec {
                    id: 20,
                    name: format!("thermal-{}", 20),
                    bus_id: 2,
                    cost_per_mwh: 80.0,
                    max_generation_mw: 300.0,
                    ..Default::default()
                }),
                make_thermal(ThermalSpec {
                    id: 21,
                    name: format!("thermal-{}", 21),
                    bus_id: 3,
                    cost_per_mwh: 80.0,
                    max_generation_mw: 300.0,
                    ..Default::default()
                }),
            ])
            .pumping_stations(vec![make_pumping_station(PumpingSpec {
                id: 30,
                name: format!("ps-{}", 30),
                bus_id: 2,
                source_hydro_id: 1,
                destination_hydro_id: 3,
                max_flow_m3s: 20.0,
                ..Default::default()
            })])
            .contracts(vec![make_contract(ContractSpec {
                id: 40,
                name: format!("contract-{}", 40),
                bus_id: 1,
                price_per_mwh: 50.0,
                max_mw: 150.0,
                ..Default::default()
            })])
            .non_controllable_sources(vec![make_ncs(NcsSpec {
                id: 50,
                name: format!("ncs-{}", 50),
                bus_id: 3,
                max_generation_mw: 80.0,
                curtailment_cost: 5.0,
                ..Default::default()
            })])
            .build()
            .expect("system must be valid")
    };

    let system_asc = make_system(vec![1, 2, 3], vec![(1, Some(3)), (2, Some(3)), (3, None)]);
    let system_desc = make_system(vec![3, 2, 1], vec![(3, None), (2, Some(3)), (1, Some(3))]);

    assert_eq!(
        system_asc, system_desc,
        "System must be identical regardless of input ordering (large test)"
    );
}

#[test]
fn test_invalid_filling_config_rejected() {
    let mut hydro = make_hydro(HydroSpec {
        id: 1,
        name: format!("hydro-{}", 1),
        bus_id: 1,
        max_storage_hm3: 100.0,
        max_turbined_m3s: 500.0,
        max_generation_mw: 450.0,
        ..Default::default()
    });
    hydro.entry_stage_id = Some(0);
    hydro.filling = Some(FillingConfig {
        start_stage_id: 0,
        filling_min_rate_m3s: -5.0,
    });

    let result = SystemBuilder::new()
        .buses(vec![make_bus(BusSpec {
            id: 1,
            name: format!("bus-{}", 1),
            deficit_segments: vec![DeficitSegment {
                depth_mw: Some(100.0),
                cost_per_mwh: 500.0,
            }],
            ..Default::default()
        })])
        .hydros(vec![hydro])
        .build();

    assert!(
        result.is_err(),
        "invalid filling config must fail validation"
    );

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| matches!(
            e,
            ValidationError::InvalidFillingConfig {
                hydro_id: EntityId(1),
                ..
            }
        )),
        "expected InvalidFillingConfig for hydro 1; got: {errors:?}"
    );
}

#[test]
fn test_diversion_invalid_reference_rejected() {
    let mut hydro = make_hydro(HydroSpec {
        id: 1,
        name: format!("hydro-{}", 1),
        bus_id: 1,
        max_storage_hm3: 100.0,
        max_turbined_m3s: 500.0,
        max_generation_mw: 450.0,
        ..Default::default()
    });
    hydro.diversion = Some(DiversionChannel {
        downstream_id: EntityId(999),
        max_flow_m3s: 10.0,
    });

    let result = SystemBuilder::new()
        .buses(vec![make_bus(BusSpec {
            id: 1,
            name: format!("bus-{}", 1),
            deficit_segments: vec![DeficitSegment {
                depth_mw: Some(100.0),
                cost_per_mwh: 500.0,
            }],
            ..Default::default()
        })])
        .hydros(vec![hydro])
        .build();

    assert!(
        result.is_err(),
        "hydro with bad diversion.downstream_id must fail validation"
    );

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| matches!(
            e,
            ValidationError::InvalidReference {
                source_entity_type: "Hydro",
                field_name: "diversion.downstream_id",
                referenced_id: EntityId(999),
                ..
            }
        )),
        "expected InvalidReference for Hydro.diversion.downstream_id -> Hydro 999; got: {errors:?}"
    );
}

#[test]
fn test_canonical_order_stable_under_name_changes() {
    // Canonical order is (operational_start_date, id): renaming entities with ids
    // and dates held constant must not change processing order — names are
    // user-chosen and must not influence the layout.
    let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();

    let mut hydro_a = make_hydro(HydroSpec {
        id: 1,
        name: format!("hydro-{}", 1),
        bus_id: 1,
        max_storage_hm3: 100.0,
        max_turbined_m3s: 500.0,
        max_generation_mw: 450.0,
        ..Default::default()
    });
    hydro_a.name = "alpha".to_string();
    hydro_a.operational_start_date = date;
    let mut hydro_b = make_hydro(HydroSpec {
        id: 2,
        name: format!("hydro-{}", 2),
        bus_id: 1,
        max_storage_hm3: 100.0,
        max_turbined_m3s: 500.0,
        max_generation_mw: 450.0,
        ..Default::default()
    });
    hydro_b.name = "bravo".to_string();
    hydro_b.operational_start_date = date;

    let system_a = SystemBuilder::new()
        .buses(vec![make_bus(BusSpec {
            id: 1,
            name: format!("bus-{}", 1),
            deficit_segments: vec![DeficitSegment {
                depth_mw: Some(100.0),
                cost_per_mwh: 500.0,
            }],
            ..Default::default()
        })])
        .hydros(vec![hydro_a.clone(), hydro_b.clone()])
        .build()
        .expect("system must be valid");

    let mut a_renamed = hydro_a;
    a_renamed.name = "zulu".to_string();
    let mut b_renamed = hydro_b;
    b_renamed.name = "alfa".to_string();

    let system_b = SystemBuilder::new()
        .buses(vec![make_bus(BusSpec {
            id: 1,
            name: format!("bus-{}", 1),
            deficit_segments: vec![DeficitSegment {
                depth_mw: Some(100.0),
                cost_per_mwh: 500.0,
            }],
            ..Default::default()
        })])
        .hydros(vec![b_renamed, a_renamed])
        .build()
        .expect("system must be valid");

    let ids_a: Vec<i32> = system_a.hydros().iter().map(|h| h.id.0).collect();
    let ids_b: Vec<i32> = system_b.hydros().iter().map(|h| h.id.0).collect();
    assert_eq!(
        ids_a, ids_b,
        "renaming with (date, id) held constant must not change processing order"
    );
    assert_eq!(
        ids_a,
        vec![1, 2],
        "same-date entities order by id ascending"
    );
}

#[test]
fn test_canonical_order_sorts_by_distinct_date() {
    let early = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let late = NaiveDate::from_ymd_opt(2024, 6, 1).unwrap();

    let mut bus_late = make_bus(BusSpec {
        id: 1,
        name: format!("bus-{}", 1),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(100.0),
            cost_per_mwh: 500.0,
        }],
        ..Default::default()
    });
    bus_late.operational_start_date = late;
    let mut bus_early = make_bus(BusSpec {
        id: 2,
        name: format!("bus-{}", 2),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(100.0),
            cost_per_mwh: 500.0,
        }],
        ..Default::default()
    });
    bus_early.operational_start_date = early;

    let system = SystemBuilder::new()
        .buses(vec![bus_late, bus_early])
        .build()
        .expect("system must be valid");

    assert!(
        system.buses()[0].operational_start_date < system.buses()[1].operational_start_date,
        "distinct dates supplied in reverse must come out date-ascending"
    );
}

#[test]
fn test_canonical_order_id_tiebreak_on_equal_date() {
    let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();

    // id 1 has the name that sorts LAST ("B"); id 2 the name that sorts first ("A").
    // The id tiebreak must win, so id 1 comes first regardless of name.
    let mut bus_b = make_bus(BusSpec {
        id: 1,
        name: format!("bus-{}", 1),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(100.0),
            cost_per_mwh: 500.0,
        }],
        ..Default::default()
    });
    bus_b.name = "B".to_string();
    bus_b.operational_start_date = date;
    let mut bus_a = make_bus(BusSpec {
        id: 2,
        name: format!("bus-{}", 2),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(100.0),
            cost_per_mwh: 500.0,
        }],
        ..Default::default()
    });
    bus_a.name = "A".to_string();
    bus_a.operational_start_date = date;

    let system = SystemBuilder::new()
        .buses(vec![bus_a, bus_b])
        .build()
        .expect("system must be valid");

    assert_eq!(system.buses()[0].id, EntityId(1));
    assert_eq!(system.buses()[1].id, EntityId(2));
}
