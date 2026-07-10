# Hydro Plants

Hydroelectric power plants are the central dispatchable resource in Cobre's system
model. Unlike thermal units, which convert fuel into electricity at a cost,
hydro plants manage a reservoir — a state variable that persists between stages and
couples the dispatch decisions of today to the feasibility of tomorrow. This
intertemporal coupling is precisely why hydrothermal scheduling requires stochastic
dynamic programming rather than a simple merit-order dispatch.

A hydro plant in Cobre is composed of three physical components: a **reservoir**
that stores water between stages, a **turbine** that converts water flow into
electrical generation, and a **spillway** that releases excess water without
producing power. Each stage's LP sub-problem contains one water balance constraint
per plant: inflow plus beginning storage equals turbined flow plus spillage plus
ending storage. The solver decides how much to turbine and how much to store,
trading off present-stage generation against future-stage optionality.

Plants can be linked into a **cascade** via the `downstream_id` field. When plant A
has `downstream_id` pointing to plant B, all water released from A (turbined flow
plus spillage) enters B's reservoir at the same stage. Cascade topology is validated
to be acyclic — no chain of downstream references may loop back to an earlier plant.

For a step-by-step introduction to writing `hydros.json`, see
[Building a System](../tutorial/building-a-system.md) and
[Anatomy of a Case](../tutorial/anatomy-of-a-case.md). This page provides the
complete field reference with all optional fields documented.

> **Theory reference**: For the mathematical formulation of hydro modeling and the
> SDDP algorithm that drives dispatch decisions, see
> [Hydro Production Function Models](https://methodology.cobre-rs.dev/math/hydro-production-models/)
> in the methodology reference.

---

## JSON Schema

Hydro plants are defined in `system/hydros.json`. The top-level object has a single
key `"hydros"` containing an array of plant objects. The following example shows
all fields — required and optional — for a single plant:

```json
{
  "hydros": [
    {
      "id": 1,
      "name": "UHE Tucuruí",
      "bus_id": 0,
      "downstream_id": null,
      "entry_stage_id": null,
      "exit_stage_id": null,
      "reservoir": {
        "min_storage_hm3": 50.0,
        "max_storage_hm3": 45000.0
      },
      "outflow": {
        "min_outflow_m3s": 1000.0,
        "max_outflow_m3s": 100000.0
      },
      "generation": {
        "model": "constant_productivity",
        "min_turbined_m3s": 500.0,
        "max_turbined_m3s": 22500.0,
        "min_generation_mw": 0.0,
        "max_generation_mw": 8370.0
      },
      "tailrace": {
        "type": "polynomial",
        "coefficients": [5.0, 0.001]
      },
      "hydraulic_losses": {
        "type": "factor",
        "value": 0.03
      },
      "efficiency": {
        "type": "constant",
        "value": 0.93
      },
      "evaporation": {
        "coefficients_mm": [
          80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0
        ]
      },
      "diversion": {
        "downstream_id": 2,
        "max_flow_m3s": 200.0
      },
      "filling": {
        "start_stage_id": 48,
        "filling_min_rate_m3s": 100.0
      },
      "penalties": {
        "spillage_cost": 0.01,
        "diversion_cost": 0.1,
        "turbined_cost": 0.05,
        "storage_violation_below_cost": 10000.0,
        "filling_target_violation_cost": 6000.0,
        "turbined_violation_below_cost": 500.0,
        "outflow_violation_below_cost": 500.0,
        "outflow_violation_above_cost": 500.0,
        "generation_violation_below_cost": 1000.0,
        "evaporation_violation_cost": 5000.0,
        "water_withdrawal_violation_cost": 1000.0
      }
    }
  ]
}
```

The `1dtoy` template uses a minimal hydro definition that omits all optional fields.
Only `id`, `name`, `bus_id`, `downstream_id`, `reservoir`, `outflow`, and `generation`
are required. All other top-level keys (`tailrace`, `hydraulic_losses`, `efficiency`,
`evaporation`, `diversion`, `filling`, `penalties`, `travel_time_hours`) are optional and
default to off when absent.

---

## Core Fields

These fields appear at the top level of each hydro plant object.

| Field            | Type            | Required | Description                                                                                                                                                      |
| ---------------- | --------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`             | integer         | Yes      | Unique non-negative integer identifier. Must be unique across all hydro plants. Referenced by `initial_conditions.json` and by other plants via `downstream_id`. |
| `name`           | string          | Yes      | Human-readable plant name. Used in output files, validation messages, and log output.                                                                            |
| `bus_id`         | integer         | Yes      | Identifier of the electrical bus to which this plant's generation is injected. Must match an `id` in `buses.json`.                                               |
| `downstream_id`  | integer or null | Yes      | Identifier of the plant that receives this plant's outflow. `null` means the plant is at the bottom of its cascade — outflow leaves the system.                  |
| `travel_time_hours` | number or null | No    | Delay, in hours, on the cascade arc to `downstream_id`. `null` or `0` means the arc is instantaneous (today's default). Declared only on the main cascade arc — diversion and pumping flows are unaffected. See [Water Travel Time](./water-travel-time.md). |
| `entry_stage_id` | integer or null | No       | Stage index at which the plant enters service (inclusive). `null` means the plant is available from stage 0.                                                     |
| `exit_stage_id`  | integer or null | No       | Stage index at which the plant is decommissioned (inclusive). `null` means the plant is never decommissioned.                                                    |

---

## Reservoir

The `reservoir` block defines the operational storage bounds for the plant. Storage
is tracked in **hm³** (cubic hectometres; 1 hm³ = 10⁶ m³). The beginning-of-stage
storage is the state variable that links consecutive stages in the LP.

```json
"reservoir": {
  "min_storage_hm3": 0.0,
  "max_storage_hm3": 1000.0
}
```

| Field             | Type   | Description                                                                                                                                                                   |
| ----------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `min_storage_hm3` | number | Minimum operational storage (dead volume). Water below this level cannot reach the turbine intakes. For plants that can empty completely, use `0.0`.                          |
| `max_storage_hm3` | number | Maximum operational storage (flood control level). When the reservoir reaches this level, all excess inflow must be spilled. Must be strictly greater than `min_storage_hm3`. |

Setting `min_storage_hm3` to the dead volume of your reservoir is important for
correctly computing the usable storage range. A reservoir with 500 hm³ total
physical capacity but 100 hm³ below the turbine intakes should be modeled as
`min_storage_hm3: 100.0, max_storage_hm3: 500.0`.

---

## Outflow Constraints

The `outflow` block constrains total outflow from the plant. Total outflow equals
turbined flow plus spillage. These constraints are enforced by soft penalties
when they cannot be satisfied due to extreme scenario conditions.

```json
"outflow": {
  "min_outflow_m3s": 0.0,
  "max_outflow_m3s": 50.0
}
```

| Field             | Type           | Description                                                                                                                                                         |
| ----------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `min_outflow_m3s` | number         | Minimum total outflow required at all times [m³/s]. Set to the ecological flow requirement or minimum riparian right. Use `0.0` if there is no minimum requirement. |
| `max_outflow_m3s` | number or null | Maximum total outflow [m³/s]. Models the physical capacity of the river channel below the dam. `null` means no upper bound on outflow.                              |

Minimum outflow is a hard lower bound on the sum of turbined flow and spillage.
When the solver cannot meet this bound (for example, because the reservoir is
nearly empty and inflow is very low), a violation slack variable is added to the
LP at the cost specified by `outflow_violation_below_cost` in the penalties block.

---

## Generation Models

The `generation` block configures the turbine model for dispatch purposes. It
provides the default production function used when no `hydro_production_models.json`
file is present, or for any plant not listed there. All variants share the core
turbine bounds (`min_turbined_m3s`, `max_turbined_m3s`) and generation bounds
(`min_generation_mw`, `max_generation_mw`). The `model` key selects which
production function converts flow to power.

```json
"generation": {
  "model": "constant_productivity",
  "min_turbined_m3s": 0.0,
  "max_turbined_m3s": 50.0,
  "min_generation_mw": 0.0,
  "max_generation_mw": 50.0
}
```

| Field               | Type   | Description                                                                             |
| ------------------- | ------ | --------------------------------------------------------------------------------------- |
| `model`             | string | Production function variant. See the model table below.                                 |
| `min_turbined_m3s`  | number | Minimum turbined flow [m³/s]. Non-zero values model a minimum stable turbine operation. |
| `max_turbined_m3s`  | number | Maximum turbined flow (installed turbine capacity) [m³/s].                              |
| `min_generation_mw` | number | Minimum electrical generation [MW].                                                     |
| `max_generation_mw` | number | Maximum electrical generation (installed capacity) [MW].                                |

### Available Production Function Models

| Model                 | `model` value             | Status            | Description                                                                                                                                                                  |
| --------------------- | ------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Constant productivity | `"constant_productivity"` | Available         | `power = productivity * turbined_flow`. Independent of reservoir head. Productivity coefficient supplied per stage range or season in `system/hydro_production_models.json`. |
| FPHA                  | `"fpha"`                  | Available         | Piecewise-linear envelope of the nonlinear production function. Head-dependent. Configured via `hydro_production_models.json`. See below.                                    |
| Linearized head       | `"linearized_head"`       | Not yet available | Head-dependent productivity linearized around an operating point at each stage. Will be documented when released.                                                            |

For the `1dtoy` example and for most initial studies, `constant_productivity` is
the correct choice. The productivity coefficient encodes the plant's average
efficiency and net head, and is supplied in `system/hydro_production_models.json`.
For a plant with 80 m net head and 90% efficiency, the theoretical productivity is
approximately `9.81 × 80 × 0.90 / 1000 ≈ 0.706` MW/(m³/s).

---

## FPHA Production Model

The FPHA (Função de Produção Hidroelétrica Aproximada) model represents the nonlinear
relationship between reservoir volume, turbined flow, spillage, and electrical
generation as a piecewise-linear envelope. It captures the head
dependence of hydro production — plants with high reservoir levels generate more
power for the same turbined flow.

FPHA is configured per plant and per stage via `system/hydro_production_models.json`.
A plant not listed in that file uses the `model` specified in its `generation` block
in `hydros.json`.

### Configuration File

`system/hydro_production_models.json` maps each hydro plant to a production model
selection strategy. The file is optional; when absent, all plants use their
`generation.model` from `hydros.json`.

Two selection strategies are supported:

**`stage_ranges`** — assigns a model to each contiguous stage interval:

```json
{
  "$schema": "../schemas/production_models.schema.json",
  "production_models": [
    {
      "hydro_id": 1,
      "selection_mode": "stage_ranges",
      "stage_ranges": [
        {
          "start_stage_id": 0,
          "end_stage_id": null,
          "model": "fpha",
          "fpha_config": {
            "source": "precomputed"
          }
        }
      ]
    }
  ]
}
```

Each stage range and season entry for a `constant_productivity` or
`linearized_head` plant must supply its productivity coefficient through
exactly one source: either an inline `productivity_mw_per_m3s` field on the
entry, or a matching `(hydro, stage)` row in
`system/hydro_energy_productivity.parquet`
(see [Per-Range and Per-Season Productivity](#per-range-and-per-season-productivity) below).

**`seasonal`** — assigns a model based on season index, with a fallback for seasons
not explicitly listed:

```json
{
  "$schema": "../schemas/production_models.schema.json",
  "production_models": [
    {
      "hydro_id": 1,
      "selection_mode": "seasonal",
      "default_model": "constant_productivity",
      "seasons": [
        {
          "season_id": 0,
          "model": "fpha",
          "fpha_config": {
            "source": "computed",
            "volume_discretization_points": 7,
            "turbine_discretization_points": 7
          }
        }
      ]
    }
  ]
}
```

Season indices are 0-based and match the season map defined in `stages.json`.

#### `reference_volume`

Each stage range and season entry may carry an optional `reference_volume`
sibling of `fpha_config`, declaring the reference operating volume the
computed-FPHA fit and the equivalent-productivity derivation consume. Set
exactly one of two mutually-exclusive forms:

- `volume_hm3` — an absolute storage value in hm³ (finite and `> 0.0`).
- `percentile` — a fraction in `[0.0, 1.0]` of the plant's operating range.

```json
"reference_volume": { "percentile": 0.65 }
```

This is the single source of truth for the reference volume; it replaces the
retired `reference_volume_hm3` column of `system/hydro_energy_productivity.parquet`.

### Hyperplane Sources

When a plant is configured with `model: "fpha"`, the `fpha_config.source` field
selects where the hyperplane coefficients come from.

#### `source: "precomputed"`

Hyperplanes are loaded directly from `system/fpha_hyperplanes.parquet`. Use this
source when you have pre-fitted hyperplanes from a previous run or from an external
tool.

```json
"fpha_config": {
  "source": "precomputed"
}
```

The `fpha_config` block for `"precomputed"` requires no additional fields. The
discretization and fitting options are ignored — the hyperplanes are used as-is.

The Parquet file must be present at `system/fpha_hyperplanes.parquet`. Its schema is:

| Column            | Type    | Required | Description                                            |
| ----------------- | ------- | -------- | ------------------------------------------------------ |
| `hydro_id`        | INT32   | Yes      | Hydro plant identifier                                 |
| `stage_id`        | INT32?  | No       | Stage the plane applies to (`null` = all stages)       |
| `plane_id`        | INT32   | Yes      | Plane index within this hydro                          |
| `gamma_0`         | DOUBLE  | Yes      | Intercept coefficient (MW)                             |
| `gamma_v`         | DOUBLE  | Yes      | Volume coefficient (MW/hm³). Must be positive.         |
| `gamma_q`         | DOUBLE  | Yes      | Turbined flow coefficient (MW per m³/s)                |
| `gamma_s`         | DOUBLE  | Yes      | Spillage coefficient (MW per m³/s). Must be ≤ 0.       |
| `kappa`           | DOUBLE? | No       | Correction factor (default: 1.0)                       |
| `valid_v_min_hm3` | DOUBLE? | No       | Minimum volume where this plane is valid (hm³)         |
| `valid_v_max_hm3` | DOUBLE? | No       | Maximum volume where this plane is valid (hm³)         |
| `valid_q_max_m3s` | DOUBLE? | No       | Maximum turbined flow where this plane is valid (m³/s) |

Each `(hydro_id, stage_id)` group must have at least 1 plane. Rows are sorted by
`(hydro_id, stage_id, plane_id)` ascending; null `stage_id` sorts before any
non-null value.

#### `source: "computed"`

Hyperplanes are fitted at runtime from the plant's physical geometry. Cobre
evaluates the production function `phi(v, q)` (at spillage = 0) on a
(volume, turbined-flow) grid, takes the 3-D convex hull of the resulting
cloud using vendored qhull, applies a least-squares α correction to the
intercept, and then fits a per-plane lateral/spillage secant. Fits are
resolved independently per stage (one fit per season or stage range), so
plants whose head-efficiency characteristics change across seasons get
stage-specific plane sets. Run-of-river plants with a single operating
volume (constant forebay) are supported: the volume dimension collapses and
the fit produces a valid single-volume hyperplane set.

This source requires:

1. The hydro plant must have `tailrace`, `hydraulic_losses`, and `efficiency` models
   defined in `hydros.json`.
2. `system/hydro_geometry.parquet` must contain at least 1 row for the plant.
   A single row is valid for run-of-river plants with a constant forebay (γ_V = 0).
   `linearized_head` still requires at least 2 rows because it fits a head slope
   in volume; that constraint does not apply to FPHA.

```json
"fpha_config": {
  "source": "computed",
  "volume_discretization_points": 5,
  "turbine_discretization_points": 5,
  "spillage_discretization_points": 5,
  "max_planes_per_hydro": 10,
  "fitting_window": null
}
```

All fields except `source` are optional:

| Field                            | Default | Description                                                                                                            |
| -------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------- |
| `volume_discretization_points`   | 5       | Number of volume grid points for fitting. Must be >= 2.                                                                |
| `turbine_discretization_points`  | 5       | Number of turbined-flow grid points. Must be >= 2.                                                                     |
| `spillage_discretization_points` | 5       | Number of spillage grid points. Must be >= 2.                                                                          |
| `max_planes_per_hydro`           | 10      | Maximum planes to retain per (hydro, stage) after the convex-hull fit. Must be >= 1.                                   |
| `fitting_window`                 | null    | Optional volume range for fitting. When absent, the full operating range `[min_storage_hm3, max_storage_hm3]` is used. |

The `fitting_window` field restricts which portion of the operating range is used to
construct the grid. Use it when the plant rarely operates near one extreme and you
want the planes to be tighter in the operating region. Two bound variants are
supported per dimension, and they are mutually exclusive:

```json
"fitting_window": {
  "volume_min_hm3": 1000.0,
  "volume_max_hm3": 40000.0
}
```

```json
"fitting_window": {
  "volume_min_percentile": 5.0,
  "volume_max_percentile": 95.0
}
```

Do not mix absolute (`_hm3`) and percentile (`_percentile`) bounds for the same
limit — the validator will reject the configuration.

### Fit-Quality Warning

After fitting, Cobre evaluates every fitted plane set against the exact
production function on the spillage = 0 grid (the V/Q envelope). When the
relative mean absolute deviation between the fitted envelope and the exact
function exceeds 5 %, a warning is logged naming the plant and stage:

```
Warning: hydro 'UHE Example' stage 3 FPHA fit deviation = 6.2 % (> 5 %).
Consider increasing discretization points or narrowing the fitting window.
```

The warning is informational — the run continues with the fitted planes. The
threshold of 5 % is assessed on the spillage = 0 grid and reflects how well
the V/Q envelope was captured; the spillage secant correction is applied
separately and is not included in this check.

For `source: "precomputed"`, the `kappa` column in `system/fpha_hyperplanes.parquet`
is read as a back-compat correction factor applied to each plane's intercept.
When the column is absent or null, kappa defaults to 1.0 (the stored intercepts are
used unchanged). The kappa derivation and warning are removed from the computed path;
they apply only to precomputed inputs that carry an explicit kappa value.

### Parquet Export for Round-Trip Use

When hyperplanes are fitted at runtime (`source: "computed"`), the fitted
coefficients are automatically written to:

```
output/hydro_models/fpha_hyperplanes.parquet
```

This file uses the same 11-column schema as the input `system/fpha_hyperplanes.parquet`.
To switch from computed to precomputed fitting on a subsequent run, copy this file
to `system/fpha_hyperplanes.parquet` and change `source` to `"precomputed"` in
`hydro_production_models.json`.

### Plane Reduction (`fpha_plane_reduction`)

The optional file-level `fpha_plane_reduction` block in
`system/hydro_production_models.json` merges near-parallel or near-coincident FPHA
planes after fitting, reducing the LP column count without changing the fitted
approximation significantly. It is off by default (absent = no reduction) and is
applied uniformly to every plant in the file.

Two mutually exclusive methods are supported, selected by the `method` field:

**Angle method** — merges planes whose normal vectors are within `tolerance_deg`
degrees of each other:

```json
"fpha_plane_reduction": {
  "method": "angle",
  "tolerance_deg": 2.0
}
```

| Field           | Required | Description                                                                  |
| --------------- | -------- | ---------------------------------------------------------------------------- |
| `method`        | Yes      | Must be `"angle"`.                                                           |
| `tolerance_deg` | Yes      | Maximum angle between plane normals to merge them. Finite, in `[0.0, 90.0]`. |

**Distance method** — merges planes whose sampled mean-squared distance stays within
`tolerance_pct` of each other, using `n_samples` sample points:

```json
"fpha_plane_reduction": {
  "method": "distance",
  "tolerance_pct": 0.01,
  "n_samples": 200
}
```

| Field           | Required | Description                                                                                 |
| --------------- | -------- | ------------------------------------------------------------------------------------------- |
| `method`        | Yes      | Must be `"distance"`.                                                                       |
| `tolerance_pct` | Yes      | Maximum relative MSE distance (fraction) to treat two planes as coincident. Finite, >= 0.0. |
| `n_samples`     | Yes      | Number of sample points used to estimate the distance. Must be >= 1.                        |

Supplying a field that belongs to the other method is a load-time error
(`deny_unknown_fields`). The origin plane (zero generation at zero turbining) is
never merged into another plane. The distance method is deterministically seeded:
its sample draws are bit-identical across input ordering and rank count.

### Per-Range and Per-Season Productivity

For `constant_productivity` and `linearized_head` hydros, the equivalent
productivity ρ_eq [MW/(m³/s)] for each `(hydro, stage)` pair must be supplied
by exactly one of two sources:

- **`system/hydro_production_models.json`** — set
  `productivity_mw_per_m3s` directly on a `stage_range` or seasonal
  entry. Use this when productivity is constant across a range of stages
  or repeats with the season cycle.
- **`system/hydro_energy_productivity.parquet`** — supply a row in the
  `equivalent_productivity_mw_per_m3s` column. A row with `stage_id` set
  refines a single stage; a row with `stage_id = NULL` is a per-hydro
  default that covers any stage not refined by a stage-specific row.
  Use this for per-stage numerical refinement of an otherwise declarative
  JSON configuration.

Resolution order at load time:

1. Parquet stage-specific row (exact `stage_id` match).
2. Parquet per-hydro default row (`stage_id = NULL`).
3. JSON `productivity_mw_per_m3s` on the matching stage range or season.

If neither source supplies a value for a `(hydro, stage)` pair, loading
fails with a clear schema error naming both files. Supplying a value
from both files for the same `(hydro, stage)` is also rejected — pick
exactly one source per pair.

```json
{
  "start_stage_id": 12,
  "end_stage_id": 24,
  "model": "constant_productivity",
  "productivity_mw_per_m3s": 0.72
}
```

| Field                     | Type   | Required            | Description                                                                                                                                                                  |
| ------------------------- | ------ | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `productivity_mw_per_m3s` | number | Optional (non-FPHA) | Productivity coefficient [MW/(m³/s)]. Finite and non-negative when present (`>= 0.0`); `0.0` marks a planned-outage stage. Omit to supply via the parquet. Rejected on FPHA. |

**Validation rules:**

- `productivity_mw_per_m3s` must be finite and non-negative (`>= 0.0`) when present.
  `0.0` is accepted as a planned-outage marker.
- `productivity_mw_per_m3s` is rejected when `model` is `"fpha"` (FPHA derives
  productivity from VHA geometry and ρ_esp, not a scalar coefficient).
- For `constant_productivity` and `linearized_head`, the JSON value may be
  omitted (or set to `null`) when the parquet override supplies the value
  for the same `(hydro, stage)`.

For FPHA hydros, ρ_eq is derived from VHA geometry and ρ_esp. The parquet
column `equivalent_productivity_mw_per_m3s` may still supply an override
that replaces the derivation when present.

---

## Cascade Topology

The `downstream_id` field creates a directed chain of hydro plants. Water released
from an upstream plant — whether turbined or spilled — enters the downstream
plant's reservoir in the same stage.

To model a three-plant cascade where plant 0 flows into plant 1, which flows into
plant 2:

```json
{ "id": 0, "downstream_id": 1, ... }
{ "id": 1, "downstream_id": 2, ... }
{ "id": 2, "downstream_id": null, ... }
```

Cobre validates that the downstream graph is **acyclic**: no chain of
`downstream_id` references may return to a plant already in the chain. A cycle
would make the water balance equation unsolvable. The validator reports the cycle
as a topology error with the full chain of plant IDs.

Plants with `downstream_id: null` are **tailwater plants** — their outflow leaves
the basin. Each connected component of the cascade graph must have exactly one
tailwater plant (the chain's end node). A cascade component with no tailwater plant
would be a cycle, which the validator rejects.

---

## Advanced Fields

The following fields enable more detailed physical modeling. They are all optional.
For most system planning studies, these fields can be omitted; they become relevant
when calibrating a model against historical dispatch data or when the head variation
at a plant is significant.

### Tailrace Model

The `tailrace` block models the downstream water level as a function of total
outflow. The tailrace elevation affects the net hydraulic head and is used by the
`linearized_head` and `fpha` generation models. When absent, tailrace elevation
is treated as zero.

Two variants are supported:

**Polynomial** — `height = a₀ + a₁·Q + a₂·Q² + …`

```json
"tailrace": {
  "type": "polynomial",
  "coefficients": [5.0, 0.001]
}
```

`coefficients` is an array of polynomial coefficients in ascending power order.
`coefficients[0]` is the constant term (height at zero outflow in metres),
`coefficients[1]` is the coefficient for Q¹, and so on.

**Piecewise** — linearly interpolated between (outflow, height) breakpoints.

```json
"tailrace": {
  "type": "piecewise",
  "points": [
    { "outflow_m3s": 0.0, "height_m": 3.0 },
    { "outflow_m3s": 5000.0, "height_m": 4.5 },
    { "outflow_m3s": 15000.0, "height_m": 6.2 }
  ]
}
```

Points must be sorted in ascending `outflow_m3s` order. The solver interpolates
linearly between adjacent points.

### Hydraulic Losses

The `hydraulic_losses` block models head loss in the penstock and draft tube.
Hydraulic losses reduce the effective head available at the turbine. When absent,
the penstock is modeled as lossless.

**Factor** — loss as a fraction of net head:

```json
"hydraulic_losses": { "type": "factor", "value": 0.03 }
```

`value` is a dimensionless fraction (e.g., `0.03` = 3% of net head).

**Constant** — fixed head loss regardless of flow:

```json
"hydraulic_losses": { "type": "constant", "value_m": 2.5 }
```

`value_m` is the fixed head loss in metres.

### Efficiency Model

The `efficiency` block scales the power output from the hydraulic power available.
When absent, 100% efficiency is assumed.

Currently only the `"constant"` variant is supported:

```json
"efficiency": { "type": "constant", "value": 0.93 }
```

`value` is a dimensionless fraction in the range (0, 1]. A value of `0.93` means
the turbine converts 93% of available hydraulic power to electrical output.

### Evaporation

The `evaporation` block models the net water flux at the reservoir surface.
When absent, no evaporation is modeled. Coefficients are **signed**: positive
values represent net evaporative loss, negative values represent net rainfall
input on the lake surface (precipitation on the reservoir exceeds open-water
evaporation, common in wet months of tropical and subtropical basins).

```json
"evaporation": {
  "coefficients_mm": [
    80.0, 75.0, 70.0, 65.0, 60.0, 55.0,
    60.0, 65.0, 70.0, 75.0, 80.0, 85.0
  ],
  "reference_volumes_hm3": [
    15000, 12000, 10000, 8000, 6000, 5000,
    5500, 7000, 9000, 11000, 13000, 14500
  ]
}
```

| Field                   | Type  | Required | Description                                                                                                                                                                                                                                  |
| ----------------------- | ----- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coefficients_mm`       | array | Yes      | Exactly 12 values, one per calendar month (index 0 = January, index 11 = December). Values are in mm/month and may be negative (net rainfall on the lake surface). The net flux is computed from reservoir area.                             |
| `reference_volumes_hm3` | array | No       | Exactly 12 reference volumes [hm³] used as linearization points for evaporation, one per month. Must be within `[min_storage_hm3, max_storage_hm3]`. When absent, the algorithm uses its own default (e.g., mid-point of the storage range). |

### Diversion Channel

The `diversion` block models a water diversion channel that routes flow directly
from this plant's reservoir to a downstream plant's reservoir, bypassing turbines
and spillways. When absent, no diversion is modeled.

```json
"diversion": {
  "downstream_id": 2,
  "max_flow_m3s": 200.0
}
```

| Field           | Description                                                         |
| --------------- | ------------------------------------------------------------------- |
| `downstream_id` | Identifier of the plant whose reservoir receives the diverted flow. |
| `max_flow_m3s`  | Maximum diversion flow capacity [m³/s].                             |

### Filling Configuration

The `filling` block enables a filling operation mode, where the reservoir is
intentionally filled from an external, fixed inflow source (such as a diversion
works from an unrelated basin) during a defined stage window. When absent, no
filling operation is active.

```json
"filling": {
  "start_stage_id": 48,
  "filling_min_rate_m3s": 100.0
}
```

| Field                  | Description                                                                                                                                                                 |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `start_stage_id`       | Stage index at which filling begins (inclusive).                                                                                                                            |
| `filling_min_rate_m3s` | Per-stage minimum accumulation rate during filling [m³/s]: anchors a per-stage minimum target-storage trajectory on `min_storage_hm3`. Not an applied inflow and not a cap. |

---

## Penalties

The `penalties` block inside a hydro plant definition overrides the global defaults
from `penalties.json` for that specific plant. When the block is absent, all penalty
values fall back to the global defaults. When it is present, it must contain all penalty
fields.

Penalty costs are added to the LP objective when soft constraint violations occur.
They do not represent physical costs — they are optimization weights that guide the
solver to avoid infeasible or undesirable operating states.

```json
"penalties": {
  "spillage_cost": 0.01,
  "diversion_cost": 0.1,
  "turbined_cost": 0.05,
  "storage_violation_below_cost": 10000.0,
  "filling_target_violation_cost": 6000.0,
  "turbined_violation_below_cost": 500.0,
  "outflow_violation_below_cost": 500.0,
  "outflow_violation_above_cost": 500.0,
  "generation_violation_below_cost": 1000.0,
  "evaporation_violation_cost": 5000.0,
  "water_withdrawal_violation_cost": 1000.0,
  "water_withdrawal_violation_pos_cost": 1200.0,
  "water_withdrawal_violation_neg_cost": 800.0,
  "evaporation_violation_pos_cost": 5000.0,
  "evaporation_violation_neg_cost": 5000.0,
  "inflow_nonnegativity_cost": 1000.0
}
```

| Field                                 | Unit   | Description                                                                                                                                                                                        |
| ------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `spillage_cost`                       | $/m³/s | Penalty per m³/s of water spilled. Setting this low (e.g., 0.01) makes spillage the least-cost way to relieve a flood situation. Setting it high penalizes wasted water in water-scarce scenarios. |
| `diversion_cost`                      | $/m³/s | Penalty per m³/s of diverted flow exceeding the diversion channel capacity.                                                                                                                        |
| `turbined_cost`                       | $/MWh  | Regularization cost per MWh of turbined generation; applied to every hydro's turbine column regardless of production model.                                                                        |
| `storage_violation_below_cost`        | $/hm³  | Penalty per hm³ of storage below `min_storage_hm3`. Should be set high (thousands) to make violations a last resort.                                                                               |
| `filling_target_violation_cost`       | $/hm³  | Penalty per hm³ of storage below the filling target. Only active when a `filling` block is present.                                                                                                |
| `turbined_violation_below_cost`       | $/m³/s | Penalty per m³/s of turbined flow below `min_turbined_m3s`. Applied per block.                                                                                                                     |
| `outflow_violation_below_cost`        | $/m³/s | Penalty per m³/s of total outflow below `min_outflow_m3s`. Set high to enforce ecological flow requirements. Applied per block.                                                                    |
| `outflow_violation_above_cost`        | $/m³/s | Penalty per m³/s of total outflow above `max_outflow_m3s`. Set high to enforce flood channel capacity limits. Applied per block.                                                                   |
| `generation_violation_below_cost`     | $/MW   | Penalty per MW of generation below `min_generation_mw`. Applied per block.                                                                                                                         |
| `evaporation_violation_cost`          | $/mm   | Symmetric evaporation violation penalty. Applies to both directions unless overridden by directional fields.                                                                                       |
| `water_withdrawal_violation_cost`     | $/m³/s | Symmetric water withdrawal violation penalty. Applies to both directions unless overridden by directional fields.                                                                                  |
| `evaporation_violation_pos_cost`      | $/mm   | Over-evaporation violation penalty. Overrides `evaporation_violation_cost` for the positive direction.                                                                                             |
| `evaporation_violation_neg_cost`      | $/mm   | Under-evaporation violation penalty. Overrides `evaporation_violation_cost` for the negative direction.                                                                                            |
| `water_withdrawal_violation_pos_cost` | $/m³/s | Over-withdrawal violation penalty. Overrides `water_withdrawal_violation_cost` for the positive direction.                                                                                         |
| `water_withdrawal_violation_neg_cost` | $/m³/s | Under-withdrawal violation penalty. Overrides `water_withdrawal_violation_cost` for the negative direction.                                                                                        |
| `inflow_nonnegativity_cost`           | $/m³/s | Per-plant override for the global inflow non-negativity penalty. Only active when `modeling.inflow_non_negativity.method` is `"penalty"` or `"truncation_with_penalty"`.                           |

The `evaporation_violation_cost` and `water_withdrawal_violation_cost` fields act as
symmetric defaults: the same penalty applies whether the violation is positive
(over-evaporation or over-withdrawal) or negative (under-evaporation or
under-withdrawal). When the directional fields are present
(`evaporation_violation_pos_cost`, `evaporation_violation_neg_cost`,
`water_withdrawal_violation_pos_cost`, `water_withdrawal_violation_neg_cost`),
they override the symmetric default for their respective direction, allowing
asymmetric penalty weights. The `turbined_violation_below_cost`,
`outflow_violation_below_cost`, `outflow_violation_above_cost`, and
`generation_violation_below_cost` penalties are applied independently to each
dispatch block within a stage.

### Three-Tier Resolution Cascade

Penalty values are resolved from the most specific to the most general source:

1. **Stage-level override** (defined in stage-specific penalty files, when present)
2. **Entity-level override** (the `penalties` block inside the plant's JSON object)
3. **Global default** (the `hydro` section of `penalties.json`)

The `penalties` block on a plant replaces the global default for that plant alone.
All plants that do not have a `penalties` block use the global values from
`penalties.json`. The global `penalties.json` file must always be present and must
contain all hydro penalty fields.

---

## Validation Rules

Cobre's layered validation pipeline checks the following conditions on hydro
plants. Violations are reported as error messages with the failing plant's `id`
and the nature of the problem.

| Rule                            | Error Class          | Description                                                                                                                                                                                                              |
| ------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Bus reference integrity         | Reference error      | Every `bus_id` must match an `id` in `buses.json`.                                                                                                                                                                       |
| Downstream reference integrity  | Reference error      | Every non-null `downstream_id` must match an `id` in `hydros.json`.                                                                                                                                                      |
| Cascade acyclicity              | Topology error       | The directed graph of `downstream_id` links must be acyclic.                                                                                                                                                             |
| Storage bounds ordering         | Physical feasibility | `min_storage_hm3` must be less than `max_storage_hm3`.                                                                                                                                                                   |
| Outflow bounds ordering         | Physical feasibility | When `max_outflow_m3s` is present, it must be greater than or equal to `min_outflow_m3s`.                                                                                                                                |
| Turbine bounds ordering         | Physical feasibility | `min_turbined_m3s` must be less than or equal to `max_turbined_m3s`.                                                                                                                                                     |
| Generation bounds consistency   | Physical feasibility | `min_generation_mw` must be less than or equal to `max_generation_mw`.                                                                                                                                                   |
| Initial conditions completeness | Reference error      | Every hydro plant must have exactly one entry in `initial_conditions.json` (either in `storage` or `filling_storage`, not both).                                                                                         |
| Evaporation array length        | Schema error         | When `evaporation` is present, `coefficients_mm` must have exactly 12 values. `reference_volumes_hm3`, when present, must also have exactly 12 values within `[min_storage_hm3, max_storage_hm3]`.                       |
| FPHA geometry coverage          | Dimensional error    | Every plant configured with `fpha` must have at least 1 row in `system/hydro_geometry.parquet` (a single row is valid for run-of-river plants); every plant configured with `linearized_head` must have at least 2 rows. |
| FPHA plane coverage             | Dimensional error    | Every `(hydro_id, stage_id)` group in `system/fpha_hyperplanes.parquet` must have at least 1 plane.                                                                                                                      |
| FPHA coefficient signs          | Semantic error       | `gamma_v` must be positive; `gamma_s` must be non-positive.                                                                                                                                                              |
| Geometry monotonicity           | Semantic error       | `volume_hm3` must be strictly increasing; `height_m` and `area_km2` must be non-decreasing.                                                                                                                              |

---

## Related Pages

- [Anatomy of a Case](../tutorial/anatomy-of-a-case.md) — walks through the complete `1dtoy` hydro definition
- [Building a System](../tutorial/building-a-system.md) — step-by-step guide to writing `hydros.json` from scratch
- [System Modeling](./system-modeling.md) — overview of all entity types and how they interact
- [Case Format Reference](../reference/case-format.md) — complete JSON schema for all input files
