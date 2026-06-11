# The 4ree Example

The `4ree` case ships in `examples/4ree/` in the Cobre repository. It models the
four-region Brazilian interconnected power system — SUDESTE, SUL, NORDESTE, and
NORTE — with hydro and thermal generation over a 12-month planning horizon
(January–December 2015). The source data is the `4ree` example from the
[sddp-lab](https://github.com/rjmalves/sddp-lab) reference implementation.

This case is larger and more structurally complex than the `1dtoy` example. It
exercises the multi-bus power balance, bidirectional transmission line constraints,
and independent hydro cascades. It is intended for structural validation of the LP
formulation against a real-world system topology, not for producing physically
meaningful dispatch results (see [Known Limitations](#known-limitations)).

---

## System Description

| Element      | Count | Details                                                                         |
| ------------ | ----- | ------------------------------------------------------------------------------- |
| Buses        | 5     | SUDESTE (0), SUL (1), NORDESTE (2), NORTE (3), NOFICT1 (4)                      |
| Hydro plants | 4     | One per real region, independent cascades, constant productivity                |
| Thermals     | 126   | All original sddp-lab thermals, remapped to 4 real buses                        |
| Lines        | 5     | SUDESTE-SUL, SUDESTE-NORDESTE, SUDESTE-NOFICT1, NORDESTE-NOFICT1, NORTE-NOFICT1 |
| Stages       | 12    | Monthly, January 2015 – December 2015, 1 block per stage                        |
| Simulation   | 100   | Post-training evaluation over 100 independently sampled scenarios               |

The system has four independent hydro cascades, each with a single reservoir
serving its own real region. NOFICT1 is a fictitious aggregation node with zero
load that acts as a transit hub connecting NORTE, NORDESTE, and SUDESTE. All five
transmission lines are bidirectional with asymmetric capacity.

Initial reservoir storage values come directly from the sddp-lab source data:

| Hydro plant | Region   | Initial storage (hm³) |
| ----------- | -------- | --------------------- |
| 0           | SUDESTE  | 38343.9               |
| 1           | SUL      | 10068.8               |
| 2           | NORDESTE | 9030.2                |
| 3           | NORTE    | 5161.9                |

---

## Network Topology

NOFICT1 serves as a hub node through which NORTE, NORDESTE, and SUDESTE exchange
energy. SUL connects directly to SUDESTE. The topology is:

```
 SUL ──────────── SUDESTE ──────────── NORDESTE
                     │                    │
                     └────── NOFICT1 ─────┘
                                 │
                               NORTE
```

Line capacities (direct / reverse MW):

| Line             | Source   | Target   | Direct (MW) | Reverse (MW) |
| ---------------- | -------- | -------- | ----------- | ------------ |
| SUDESTE_SUL      | SUDESTE  | SUL      | 7500        | 5470         |
| SUDESTE_NORDESTE | SUDESTE  | NORDESTE | 1000        | 600          |
| SUDESTE_NOFICT1  | SUDESTE  | NOFICT1  | 4000        | 2940         |
| NORDESTE_NOFICT1 | NORDESTE | NOFICT1  | 3500        | 3300         |
| NORTE_NOFICT1    | NORTE    | NOFICT1  | 10000       | 4407         |

The `direct` direction is defined as from the lower bus ID to the higher bus ID
(e.g., SUDESTE→SUL, SUDESTE→NOFICT1). All five lines are represented as single
bidirectional entries using Cobre's `capacity.direct_mw` / `capacity.reverse_mw`
fields.

---

## Input Files

### `config.json`

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/config.schema.json",
  "training": {
    "forward_passes": 4,
    "stopping_rules": [
      {
        "type": "iteration_limit",
        "limit": 256
      }
    ],
    "scenario_source": {
      "seed": 42,
      "inflow": { "scheme": "in_sample" },
      "load": { "scheme": "in_sample" },
      "ncs": { "scheme": "in_sample" }
    }
  },
  "simulation": {
    "enabled": true,
    "num_scenarios": 100
  },
  "modeling": {
    "inflow_non_negativity": {
      "method": "none"
    }
  }
}
```

`forward_passes: 4` draws four scenario trajectories per training iteration (multi-cut
SDDP). The iteration limit is 256 — higher than the 1dtoy case to
allow more cuts to accumulate across the 12-stage horizon. No convergence-based
stopping rule is configured; the iteration limit acts as the sole termination
criterion.

The `scenario_source` block configures per-class scenario sampling. All three
entity classes use `in_sample` with `seed: 42` for deterministic forward-pass
noise.

`modeling.inflow_non_negativity.method: "none"` allows the PAR(p) noise model to
produce negative samples without truncation. This setting has no practical effect
here because the seasonal statistics have non-negative means that dominate the
noise.

---

### `stages.json` (excerpt — Stages 0 and 1)

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/stages.schema.json",
  "policy_graph": {
    "type": "finite_horizon",
    "annual_discount_rate": 0.0
  },
  "stages": [
    {
      "id": 0,
      "start_date": "2015-01-01",
      "end_date": "2015-02-01",
      "blocks": [{ "id": 0, "name": "SINGLE", "hours": 744 }],
      "num_scenarios": 10
    },
    {
      "id": 1,
      "start_date": "2015-02-01",
      "end_date": "2015-03-01",
      "blocks": [{ "id": 0, "name": "SINGLE", "hours": 672 }],
      "num_scenarios": 10
    }
  ]
}
```

The remaining ten stages follow the same pattern covering March 2015 through
December 2015. Each stage has one load block (`SINGLE`) whose `hours` value
matches the calendar month length.

`annual_discount_rate: 0.0` matches the sddp-lab source data, which used zero
discount on all policy graph edges. The 1dtoy case uses 12% annual discount;
this case uses 0%, so costs are summed directly across stages without discounting.

---

## Usage

Validate the case (checks all five validation layers):

```sh
cobre validate examples/4ree
```

Run training and simulation:

```sh
cobre run examples/4ree
```

To write output to an explicit directory:

```sh
cobre run examples/4ree --output output
```

The run produces the same output directory structure as the 1dtoy case:
`output/training/`, `output/simulation/`, and `output/policy/`. See
[Output Structure](./1dtoy.md#output-structure) in the 1dtoy page for the
full file listing.

With 12 stages and 126 thermals the LP is substantially larger than 1dtoy. On a
modern laptop the run typically completes within a few minutes for 256 iterations.

---

## Conversion Decisions

The 4ree case was converted from the sddp-lab reference implementation. Several
structural decisions were made during the conversion; understanding them is
necessary for correctly interpreting the results.

### Bus ID remapping

sddp-lab uses 1-indexed bus IDs; Cobre uses 0-indexed IDs. The mapping is:

| sddp-lab ID | sddp-lab name | Cobre ID | Cobre name |
| ----------- | ------------- | -------- | ---------- |
| 1           | SUDESTE       | 0        | SUDESTE    |
| 2           | SUL           | 1        | SUL        |
| 3           | NORDESTE      | 2        | NORDESTE   |
| 4           | NORTE         | 3        | NORTE      |
| 5           | NOFICT1       | 4        | NOFICT1    |

All `bus_id` references in hydros, thermals, and lines are remapped accordingly.
Thermal IDs are also remapped from 1-indexed (sddp-lab) to 0-indexed (Cobre).

### NOFICT1 as a transit hub

sddp-lab includes a fictitious aggregation node NOFICT1 (sddp-lab id=5) with zero
load that acts as an intermediate hub connecting northern generation to southern
load centers. In this conversion NOFICT1 is retained as bus id=4 because three of
the five modeled transmission lines use it as an endpoint.

All 126 thermals in sddp-lab connect to real buses 1–4; none were attached to
bus 5, so no thermal reassignment was needed. No hydro plant is assigned to
NOFICT1 — the four hydro cascades remain tied to the four real regions.

### Line merging

The original sddp-lab model used paired unidirectional lines to represent
asymmetric capacity. Cobre's `capacity.direct_mw` and `capacity.reverse_mw` fields
encode both directions in a single line entry. Ten sddp-lab lines collapse to five
Cobre lines:

| Cobre line name  | direct_mw | reverse_mw |
| ---------------- | --------- | ---------- |
| SUDESTE_SUL      | 7500      | 5470       |
| SUDESTE_NORDESTE | 1000      | 600        |
| SUDESTE_NOFICT1  | 4000      | 2940       |
| NORDESTE_NOFICT1 | 3500      | 3300       |
| NORTE_NOFICT1    | 10000     | 4407       |

The `direct` direction is defined as from the lower bus ID to the higher bus ID
(SUDESTE→SUL, SUDESTE→NORDESTE, SUDESTE→NOFICT1, NORDESTE→NOFICT1, NORTE→NOFICT1).

### Inflow model

sddp-lab uses per-season LogNormal marginal distributions with independent hydros
for its 4ree inflow scenarios. Cobre uses PAR(p) with additive normal noise.
Converting LogNormal(mu, sigma) parameters to PAR(0) normal parameters requires
moment-matching, but the resulting distributions have fundamentally different tail
shapes, making convergence bound comparisons unreliable.

Decision: provide seasonal statistics via the `scenarios/` directory and run with
stochastic inflows using PAR(p). The `scenarios/inflow_seasonal_stats.parquet`
file supplies per-season means and standard deviations derived from the sddp-lab
LogNormal parameters via moment-matching. The resulting distributions differ from
the original LogNormal tails, so convergence bounds remain incomparable with
sddp-lab, but the model produces physically plausible hydro dispatch.

### Risk measure

The sddp-lab 4ree case uses CVaR (alpha=0.5, lambda=0.5). Cobre supports both
Expectation (risk-neutral) and CVaR risk measures via `stages.json`. However, this
example currently runs with the default Expectation risk measure to keep the case
simple. To match sddp-lab's objective, configure CVaR in the stage definitions
with `{"cvar": {"alpha": 0.5, "lambda": 0.5}}`. Even with matching risk measures,
numerical results may differ due to the deterministic-inflow simplification.

### Discount rate

sddp-lab's policy graph edges all carry `discount_rate: 0.0`. The `stages.json`
`annual_discount_rate: 0.0` field matches this, so costs are accumulated without
discounting across the 12-month horizon.

### Spillage penalty

The sddp-lab `hydros.csv` lists `spillage_penalty = 1` ($/hm³) for all hydros.
The global spillage penalty in `penalties.json` is set to 1.0 $/hm³ to match.

---

## Known Limitations

**Results are not comparable to sddp-lab.** Structural differences make objective
values and dispatch patterns incomparable: PAR(p) normal versus lognormal inflow
distributions (different tail shapes despite moment-matching), default Expectation
versus CVaR risk measure (configurable — see [Risk measure](#risk-measure)), and
differences in how the NOFICT1 hub lines are modeled. Use this case for LP
structural validation and for verifying that stochastic inflow sampling behaves
correctly.

**NOFICT1 carries no load and no generation.** As a fictitious hub node, NOFICT1
has a zero-load balance constraint. Energy may flow through it in transit between
NORTE, NORDESTE, and SUDESTE, but there is no generator or consumer attached
directly to it.
