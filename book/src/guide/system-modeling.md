# System Modeling

A Cobre case describes a power system as a collection of **entities**. Each entity
represents a physical component — a bus, a generator, a transmission line — or a
contractual obligation. Together, they form the complete model that the solver turns
into a sequence of LP sub-problems, one per stage per scenario trajectory.

The fundamental organizing principle: every generator and every load
connects to a **bus**. A bus is an electrical node at which the power balance
constraint must hold. At each stage and each load block, the LP enforces that the
total power injected into a bus equals the total power withdrawn from it. When the
constraint cannot be satisfied by physical generation alone, deficit slack variables
absorb the gap at a penalty cost, ensuring the LP always has a feasible solution.

Entities are grouped by type and stored in a `System` object. The `System` is built
from the case directory by `load_case`, which runs a layered validation pipeline
before handing the model to the solver. Within the `System`, all entity collections
are kept in canonical ID-sorted order. This ordering is an invariant: it guarantees
that simulation results are bit-for-bit identical regardless of the order entities
appear in the input files.

---

## Entity Types

Every modeled entity type contributes LP variables and constraints in optimization
and simulation.

| Entity Type      | Status | JSON File                              | Description                                                                                                                                                                                                |
| ---------------- | ------ | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Bus              | Full   | `system/buses.json`                    | Electrical node. Power balance constraint per stage per block. See [Network Topology](./network-topology.md).                                                                                              |
| Line             | Full   | `system/lines.json`                    | Transmission interconnection between two buses with flow limits and losses. See [Network Topology](./network-topology.md).                                                                                 |
| Hydro            | Full   | `system/hydros.json`                   | Reservoir-turbine-spillway system with cascade linkage. See [Hydro Plants](./hydro-plants.md).                                                                                                             |
| Thermal          | Full   | `system/thermals.json`                 | Dispatchable generator with piecewise-linear cost curve. See [Thermal Units](./thermal-units.md).                                                                                                          |
| Pumping Station  | Full   | `system/pumping_stations.json`         | Pumped-storage or water-transfer station. Contributes a per-block pumped-flow variable; withdraws water from a source reservoir and injects it into a destination reservoir, consuming power from its bus. |
| Non-Controllable | Full   | `system/non_controllable_sources.json` | Variable renewable source (wind, solar, run-of-river). Generation variable bounded by available capacity × block factor, with curtailment penalty.                                                         |
| Contract         | Full   | `system/energy_contracts.json`         | Bilateral energy purchase or sale obligation. Contributes one LP column per block per direction (import or export), bounded by `[min_mw, max_mw]`, with a signed injection into the bus power balance.     |

---

## Non-Controllable Sources

A non-controllable source (NCS) represents a variable renewable generator whose
output is externally specified rather than optimized by the solver. Typical
examples include wind farms, utility-scale solar arrays, and run-of-river hydro
units without significant storage. The solver dispatches the NCS at its full
available capacity unless doing so would oversupply the bus, in which case
curtailment occurs and the solver pays a curtailment penalty.

Each NCS contributes one generation LP variable per block, bounded by:

```
0 <= generation_mw <= available_generation_mw * block_factor
```

where `available_generation_mw` comes from `constraints/ncs_bounds.parquet`
(with `system/non_controllable_sources.json` providing the base value) and
`block_factor` from `scenarios/non_controllable_factors.json` (default 1.0).

When `scenarios/non_controllable_stats.parquet` is present, NCS availability
becomes stochastic: each forward and backward pass scenario draws a random
availability factor and the LP column upper bound varies per scenario. See
[Stochastic Modeling](./stochastic-modeling.md#stochastic-ncs-availability)
for details.

The objective coefficient is `-curtailment_cost * block_hours`, making it
cheaper to generate than to curtail. The NCS generation variable injects +1.0 MW
at its connected bus in the power balance constraint, identical to a thermal plant.

Simulation output is written to `simulation/non_controllables/` with columns
for `generation_mw`, `available_mw`, `curtailment_mw`, and `curtailment_cost`
per (stage, block, source) triplet. See the
[Output Format Reference](../reference/output-format.md) for the complete schema.

---

## Pumping Stations

A pumping station represents a pumped-storage or water-transfer installation that
moves water from a source hydro reservoir uphill to a destination hydro reservoir,
consuming electrical power in the process.

Each pumping station contributes one per-block pumped-flow decision variable,
bounded by `[min_m3s, max_m3s]`. The pumped flow appears with opposite signs in
the two reservoir water-balance rows: it is subtracted from the source reservoir
and added to the destination reservoir. The power drawn from the station's bus is:

```
power_consumed_mw = consumption_mw_per_m3s × flow_m3s
```

This power appears as a load on the bus power-balance row, identical in structure
to a bus load demand. Simulation output is written to `simulation/pumping_stations/`
and the associated cost is reported in the `pumping_cost` column.

Pumping stations support the same commissioning window available on other entity
types: when `entry_stage_id` and `exit_stage_id` are set, the station contributes
LP variables only at stages in `[entry_stage_id, exit_stage_id)`. Outside that
window the station contributes no columns. A worked example is available at
`examples/deterministic/d32-reversible-plant`.

---

## Energy Contracts

An energy contract represents a bilateral purchase or sale obligation with a
counterparty outside the modeled system. Each contract contributes one LP column
per block per direction on its `bus_id`. An import contract injects power into the
bus (`+1.0` coefficient in the power-balance row); an export contract withdraws
power from the bus (`−1.0` coefficient). The column is bounded by:

```
min_mw <= power_mw <= max_mw
```

The price sign follows the economic convention: a positive `price_per_mwh`
represents a cost (the system pays for imported energy), and a negative
`price_per_mwh` represents revenue (the system earns from exported energy).

Contracts support the same commissioning window used by other entity types:
when `entry_stage_id` and `exit_stage_id` are set, the contract is active only
at stages in `[entry_stage_id, exit_stage_id)`. At dormant stages the column
bounds are pinned to `[0, 0]`, and the output row is emitted with `power_mw = 0`
and `operative_state_code = 1` — the row is never absent.

Stage-varying bounds and prices are supplied via `constraints/contract_bounds.parquet`,
which accepts sparse `(contract_id, stage_id)` rows carrying any combination of
`min_mw`, `max_mw`, and `price_per_mwh`. Absent rows use the base entity values.
A non-zero `min_mw` at a given stage acts as a take-or-pay floor: the LP must
dispatch at least that quantity at the contract price.

Contract dispatch is stateless: contracts carry no state variable and do not
contribute to Benders cuts. All contract cost is booked inside `resource_cost`
in the cost breakdown. Simulation output is written to `simulation/contracts/`
with columns for `stage_id`, `block_id`, `contract_id`, `power_mw`,
`energy_mwh`, `price_per_mwh`, `total_cost`, and `operative_state_code`. See
the [Output Format Reference](../reference/output-format.md) for the complete
schema.

### Worked example — `examples/deterministic/d41-energy-contracts`

The D41 case has two contracts on a single bus, with three stages of 730 h each.

**Contract 0 — import, always active:**

```json
{
  "id": 0,
  "type": "import",
  "price_per_mwh": 200.0,
  "limits": { "min_mw": 0.0, "max_mw": 50.0 }
}
```

At stage 0 the import dispatches (`power_mw > 0`): the LP draws up to 50 MW
of purchased energy at $200/MWh to balance the bus.

**Contract 1 — export, commissioned at stage 1 only:**

```json
{
  "id": 1,
  "type": "export",
  "entry_stage_id": 1,
  "exit_stage_id": 2,
  "price_per_mwh": -150.0,
  "limits": { "min_mw": 0.0, "max_mw": 30.0 }
}
```

At stage 0 the export is dormant (`operative_state_code = 1`, `power_mw = 0`).
At stage 1 the export is active: the LP can dispatch up to 30 MW of sold energy,
earning $150/MWh (`total_cost < 0`).

**Stage-2 override on contract 0 via `constraints/contract_bounds.parquet`:**

| `contract_id` | `stage_id` | `min_mw` | `price_per_mwh` |
| ------------- | ---------- | -------- | --------------- |
| 0             | 2          | 10.0     | 999.0           |

At stage 2 the import is pinned to its `min_mw = 10.0` take-or-pay floor and
priced at $999/MWh. The LP must dispatch at least 10 MW regardless of the thermal
cost, because the floor is a hard column lower bound in the LP.

---

## How Entities Connect

The network is **bus-centric**. Every entity that produces or consumes power is
attached to a bus via a `bus_id` field:

```
   Hydro ──┐
           │ inject
  Thermal ─┤
           ├──> Bus <──── Line ────> Bus
  NCS ─────┘
  Import ──┘
                │
               load
                │
           Export
         Pumping Station
```

At each stage and load block, the LP enforces the bus balance constraint:

```
  sum(generation at bus) + sum(imports from lines) + deficit
    = load_demand + sum(exports to lines) + excess
```

Deficit and excess slack variables absorb imbalance at a penalty cost, ensuring
the LP is always feasible. When the deficit penalty is high enough relative to
the cost of available generation, the solver will prefer to generate rather than
incur deficit.

**Cascade topology** governs hydro plant interactions. A hydro plant with a non-null
`downstream_id` sends all of its outflow — turbined flow plus spillage — into the
downstream plant's reservoir at the same stage. The cascade forms a directed forest:
multiple upstream plants may flow into a single downstream plant, but no cycles
are allowed. Water balance is computed in topological order — upstream plants first,
downstream plants last — in a single pass per stage.

---

## Declaration-Order Invariance

The order in which entities appear in the JSON input files does not affect results.
Cobre reads all entities from their files, then sorts each collection by entity ID
before building the `System`. Every function that processes entity collections
operates on this canonical sorted order.

This invariant has a practical consequence: you can rearrange entries in
`buses.json`, `hydros.json`, or any other entity file without changing the
simulation output. You can also add new entities with lower IDs than existing ones
without disturbing results for the existing entities.

---

## Penalties and Soft Constraints

LP solvers require feasible problems. Physical constraints — minimum outflow, minimum
turbined flow, reservoir bounds — can become infeasible under extreme stochastic
scenarios (very low inflow, very high load). Cobre handles this by making nearly
every physical constraint **soft**: instead of a hard infeasibility, the solver pays
a penalty cost to violate the constraint by a small amount.

Penalties are set at three levels, resolved from most specific to most general:

1. **Stage-level override** — penalty files for individual stages, when present
2. **Entity-level override** — a `penalties` block inside the entity's JSON object
3. **Global default** — the top-level `penalties.json` file in the case directory

This three-tier cascade lets you set a strict global spillage penalty and relax
it for a specific plant that is known to spill frequently in wet years. For details on the penalty fields for each entity type,
see the [Configuration](./configuration.md) guide and the
[Case Format Reference](../reference/case-format.md).

The bus deficit segments are the most important penalty to configure correctly.
A deficit cost that is too low makes the solver prefer deficit over building
generation capacity; a cost that is too high (or an unbounded segment that is
absent) can cause numerical instability. The final deficit segment must always
have `depth_mw: null` (unbounded) to guarantee LP feasibility.

---

## Entity Lifecycle

Entities can enter service or be decommissioned at specified stages using
`entry_stage_id` and `exit_stage_id` fields:

| Field            | Type            | Meaning                                                                                                                                                                                              |
| ---------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `entry_stage_id` | integer or null | Stage index at which the entity enters service (inclusive). `null` = available from stage 0                                                                                                          |
| `exit_stage_id`  | integer or null | Stage index from which the entity is decommissioned — inactive at this stage and after, so the active window is the half-open range `[entry_stage_id, exit_stage_id)`. `null` = never decommissioned |

These fields are available on `Hydro`, `Thermal`, `Line`, `NonControllableSource`,
`PumpingStation`, and `EnergyContract` entities. When a plant has `entry_stage_id: 12`,
the LP does not include any variables for that plant in stages 0 through 11. From
stage 12 onward, the plant appears in every sub-problem as normal.

Lifecycle fields are useful for planning studies that span commissioning or retirement
events: new thermal plants coming online mid-horizon, or aging hydro units being
decommissioned. Each lifecycle event is validated to ensure that `entry_stage_id`
falls within the stage range defined in `stages.json`.

---

## Related Pages

- [Hydro Plants](./hydro-plants.md) — complete field reference for `system/hydros.json`
- [Thermal Units](./thermal-units.md) — complete field reference for `system/thermals.json`
- [Network Topology](./network-topology.md) — buses, lines, deficit modeling, and transmission
- [Anatomy of a Case](../tutorial/anatomy-of-a-case.md) — walkthrough of every file in the `1dtoy` example
- [Case Format Reference](../reference/case-format.md) — complete JSON schema for all input files
