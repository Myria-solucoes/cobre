//! Integration tests exercising the full `SystemBuilder::build()` pipeline
//! through the public API only.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::too_many_lines
)]

use cobre_core::{
    ContractType, DeficitSegment, DiversionChannel, EnergyContract, EntityId, FillingConfig, Hydro,
    HydroGenerationModel, HydroPenalties, Line, NonControllableSource, PumpingStation,
    SystemBuilder, Thermal, ValidationError,
};

fn zero_hydro_penalties() -> HydroPenalties {
    HydroPenalties {
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
    }
}

fn make_bus(id: i32) -> cobre_core::Bus {
    cobre_core::Bus {
        id: EntityId(id),
        name: format!("bus-{id}"),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(100.0),
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    }
}

fn make_line(id: i32, source_bus_id: i32, target_bus_id: i32) -> Line {
    Line {
        id: EntityId(id),
        name: format!("line-{id}").to_string(),
        source_bus_id: EntityId(source_bus_id),
        target_bus_id: EntityId(target_bus_id),
        entry_stage_id: None,
        exit_stage_id: None,
        direct_capacity_mw: 200.0,
        reverse_capacity_mw: 200.0,
        losses_percent: 0.0,
        exchange_cost: 0.0,
    }
}

fn make_hydro(id: i32, bus_id: i32, downstream_id: Option<i32>) -> Hydro {
    Hydro {
        id: EntityId(id),
        name: format!("hydro-{id}").to_string(),
        bus_id: EntityId(bus_id),
        downstream_id: downstream_id.map(EntityId),
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 100.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 500.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 450.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: zero_hydro_penalties(),
    }
}

fn make_thermal(id: i32, bus_id: i32) -> Thermal {
    Thermal {
        id: EntityId(id),
        name: format!("thermal-{id}").to_string(),
        bus_id: EntityId(bus_id),
        entry_stage_id: None,
        exit_stage_id: None,
        cost_per_mwh: 80.0,
        min_generation_mw: 0.0,
        max_generation_mw: 300.0,
        anticipated_config: None,
    }
}

fn make_pumping_station(
    id: i32,
    bus_id: i32,
    source_hydro_id: i32,
    destination_hydro_id: i32,
) -> PumpingStation {
    PumpingStation {
        id: EntityId(id),
        name: format!("ps-{id}").to_string(),
        bus_id: EntityId(bus_id),
        source_hydro_id: EntityId(source_hydro_id),
        destination_hydro_id: EntityId(destination_hydro_id),
        entry_stage_id: None,
        exit_stage_id: None,
        consumption_mw_per_m3s: 0.5,
        min_flow_m3s: 0.0,
        max_flow_m3s: 20.0,
    }
}

fn make_contract(id: i32, bus_id: i32) -> EnergyContract {
    EnergyContract {
        id: EntityId(id),
        name: format!("contract-{id}").to_string(),
        bus_id: EntityId(bus_id),
        contract_type: ContractType::Import,
        entry_stage_id: None,
        exit_stage_id: None,
        price_per_mwh: 50.0,
        min_mw: 0.0,
        max_mw: 150.0,
    }
}

fn make_ncs(id: i32, bus_id: i32) -> NonControllableSource {
    NonControllableSource {
        id: EntityId(id),
        name: format!("ncs-{id}").to_string(),
        bus_id: EntityId(bus_id),
        entry_stage_id: None,
        exit_stage_id: None,
        max_generation_mw: 80.0,
        allow_curtailment: true,
        curtailment_cost: 5.0,
    }
}

#[test]
fn test_declaration_order_invariance() {
    let buses_fwd = vec![make_bus(1), make_bus(2)];
    let lines_fwd = vec![make_line(10, 1, 2)];
    let hydros_fwd = vec![make_hydro(20, 1, Some(21)), make_hydro(21, 1, None)];
    let thermals_fwd = vec![make_thermal(30, 2)];
    let pumping_fwd = vec![make_pumping_station(40, 2, 20, 21)];
    let contracts_fwd = vec![make_contract(50, 1)];
    let ncs_fwd = vec![make_ncs(60, 2)];

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

    let buses_rev = vec![make_bus(2), make_bus(1)];
    let lines_rev = vec![make_line(10, 1, 2)];
    let hydros_rev = vec![make_hydro(21, 1, None), make_hydro(20, 1, Some(21))];
    let thermals_rev = vec![make_thermal(30, 2)];
    let pumping_rev = vec![make_pumping_station(40, 2, 20, 21)];
    let contracts_rev = vec![make_contract(50, 1)];
    let ncs_rev = vec![make_ncs(60, 2)];

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
    let mut hydro_10 = make_hydro(10, 1, Some(12));
    let mut hydro_11 = make_hydro(11, 2, Some(12));
    let hydro_12 = make_hydro(12, 3, None);

    hydro_10.name = "upstream-A".to_string();
    hydro_11.name = "upstream-B".to_string();

    let system = SystemBuilder::new()
        .buses(vec![make_bus(1), make_bus(2), make_bus(3)])
        .lines(vec![make_line(100, 1, 2), make_line(101, 2, 3)])
        .hydros(vec![hydro_10, hydro_11, hydro_12])
        .thermals(vec![make_thermal(20, 1), make_thermal(21, 3)])
        .pumping_stations(vec![make_pumping_station(30, 2, 10, 12)])
        .contracts(vec![make_contract(40, 1)])
        .non_controllable_sources(vec![make_ncs(50, 3)])
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
fn test_invalid_cross_reference_rejected() {
    let bad_hydro = make_hydro(1, 999, None);

    let result = SystemBuilder::new()
        .buses(vec![make_bus(1)])
        .hydros(vec![bad_hydro])
        .build();

    assert!(
        result.is_err(),
        "system with bad bus_id must fail validation"
    );

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| matches!(
            e,
            ValidationError::InvalidReference {
                source_entity_type: "Hydro",
                field_name: "bus_id",
                referenced_id: EntityId(999),
                ..
            }
        )),
        "expected InvalidReference for Hydro.bus_id -> Bus 999; got: {errors:?}"
    );
}

#[test]
fn test_cascade_cycle_rejected() {
    let hydro_1 = make_hydro(1, 1, Some(2));
    let hydro_2 = make_hydro(2, 1, Some(1));

    let result = SystemBuilder::new()
        .buses(vec![make_bus(1)])
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
        let buses = bus_order.into_iter().map(make_bus).collect();
        let hydros = hydro_order
            .into_iter()
            .map(|(id, ds)| make_hydro(id, 1, ds))
            .collect();

        SystemBuilder::new()
            .buses(buses)
            .lines(vec![make_line(10, 1, 2), make_line(11, 2, 3)])
            .hydros(hydros)
            .thermals(vec![make_thermal(20, 2), make_thermal(21, 3)])
            .pumping_stations(vec![make_pumping_station(30, 2, 1, 3)])
            .contracts(vec![make_contract(40, 1)])
            .non_controllable_sources(vec![make_ncs(50, 3)])
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
    let mut hydro = make_hydro(1, 1, None);
    hydro.entry_stage_id = Some(0);
    hydro.filling = Some(FillingConfig {
        start_stage_id: 0,
        filling_min_rate_m3s: -5.0,
    });

    let result = SystemBuilder::new()
        .buses(vec![make_bus(1)])
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
    let mut hydro = make_hydro(1, 1, None);
    hydro.diversion = Some(DiversionChannel {
        downstream_id: EntityId(999),
        max_flow_m3s: 10.0,
    });

    let result = SystemBuilder::new()
        .buses(vec![make_bus(1)])
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
