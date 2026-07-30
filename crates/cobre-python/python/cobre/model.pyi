from typing import Any

class Bus:
    id: Any
    name: str
    deficit_segments: Any
    excess_cost: Any

class Line:
    id: Any
    name: str
    source_bus_id: Any
    target_bus_id: Any
    direct_capacity_mw: Any
    reverse_capacity_mw: Any
    losses_percent: Any
    exchange_cost: Any

class Thermal:
    id: Any
    name: str
    bus_id: Any
    min_generation_mw: Any
    max_generation_mw: Any
    cost_per_mwh: Any

class Hydro:
    id: Any
    name: str
    downstream_id: Any
    min_storage_hm3: Any
    max_storage_hm3: Any
    min_turbined_m3s: Any
    max_turbined_m3s: Any
    productivity_mw_per_m3s: Any

class EnergyContract:
    id: Any
    name: str

class PumpingStation:
    id: Any
    name: str

class NonControllableSource:
    id: Any
    name: str

class System:
    n_buses: int
    n_hydros: int
    n_lines: int
    n_stages: int
    n_thermals: int
    buses: list[Bus]
    hydros: list[Hydro]
    lines: list[Line]
    thermals: list[Thermal]
    contracts: list[EnergyContract]
    pumping_stations: list[PumpingStation]
    non_controllable_sources: list[NonControllableSource]
