# Case Format Reference

A Cobre case directory is a self-contained folder that holds all input data
for a single power system study. `load_case` reads this directory and produces
a fully-validated `System` ready for the solver.

For a description of how these files are parsed and validated, see
[cobre-io](../crates/io.md).

> JSON Schema files for all JSON input types are available on the
> [Schemas](./schemas.md) page. Download them for use with your editor's JSON
> Schema validation feature.

## Directory layout

```
my_case/
├── config.json                              # Solver configuration (required)
├── penalties.json                           # Global penalty defaults (required)
├── stages.json                              # Stage sequence and policy graph (required)
├── initial_conditions.json                  # Reservoir storage at study start (required)
├── system/
│   ├── buses.json                           # Electrical buses (required)
│   ├── lines.json                           # Transmission lines (required)
│   ├── hydros.json                          # Hydro plants (required)
│   ├── thermals.json                        # Thermal plants (required)
│   ├── non_controllable_sources.json        # Intermittent sources (optional)
│   ├── pumping_stations.json                # Pumping stations (optional)
│   ├── energy_contracts.json                # Bilateral contracts (optional)
│   ├── hydro_geometry.parquet               # Reservoir geometry tables (optional)
│   ├── hydro_production_models.json         # FPHA production function configs (optional)
    ├── hydro_energy_productivity.parquet    # Per-plant, per-stage energy-conversion overrides (optional)
│   ├── fpha_hyperplanes.parquet             # FPHA hyperplane coefficients (optional)
│   ├── tailrace_curves.parquet              # Piecewise-quartic tailrace curves (optional)
│   └── scalar_parameters.json              # Scalar parameters for constraint expressions (optional)
├── scenarios/
│   ├── inflow_history.parquet               # Historical inflow series (optional)
│   ├── inflow_seasonal_stats.parquet        # PAR model seasonal statistics (optional)
│   ├── inflow_ar_coefficients.parquet       # PAR autoregressive coefficients (optional)
│   ├── external_inflow_scenarios.parquet    # External inflow scenarios (optional)
│   ├── external_load_scenarios.parquet      # External load scenarios (optional)
│   ├── external_ncs_scenarios.parquet       # External NCS scenarios (optional)
│   ├── load_seasonal_stats.parquet          # Load model seasonal statistics (optional)
│   ├── load_factors.json                    # Load scaling factors (optional)
│   ├── non_controllable_factors.json        # NCS block scaling factors (optional)
│   ├── non_controllable_stats.parquet      # NCS stochastic availability (optional)
│   ├── correlation.json                     # Cross-series correlation model (optional)
│   └── noise_openings.parquet              # User-supplied backward-pass opening tree (optional)
└── constraints/
    ├── thermal_bounds.parquet               # Stage-varying thermal bounds (optional)
    ├── hydro_bounds.parquet                 # Stage-varying hydro bounds (optional)
    ├── line_bounds.parquet                  # Stage-varying line bounds (optional)
    ├── pumping_bounds.parquet               # Stage-varying pumping bounds (optional)
    ├── contract_bounds.parquet              # Stage-varying contract bounds (optional)
    ├── ncs_bounds.parquet                   # Stage-varying NCS available generation bounds (optional)
    ├── exchange_factors.json                # Block exchange factors (optional)
    ├── generic_constraints.json             # User-defined LP constraints (optional)
    ├── generic_constraint_bounds.parquet    # Bounds for generic constraints (optional)
    ├── penalty_overrides_bus.parquet        # Stage-varying bus penalty overrides (optional)
    ├── penalty_overrides_line.parquet       # Stage-varying line penalty overrides (optional)
    ├── penalty_overrides_hydro.parquet      # Stage-varying hydro penalty overrides (optional)
    └── penalty_overrides_ncs.parquet        # Stage-varying NCS penalty overrides (optional)
```

## File summary

| File                                            | Format  | Required | Description                                                                        |
| ----------------------------------------------- | ------- | -------- | ---------------------------------------------------------------------------------- |
| `config.json`                                   | JSON    | Yes      | Solver configuration                                                               |
| `penalties.json`                                | JSON    | Yes      | Global penalty defaults                                                            |
| `stages.json`                                   | JSON    | Yes      | Stage sequence and policy graph                                                    |
| `initial_conditions.json`                       | JSON    | Yes      | Initial reservoir storage                                                          |
| `system/buses.json`                             | JSON    | Yes      | Electrical bus registry                                                            |
| `system/lines.json`                             | JSON    | Yes      | Transmission line registry                                                         |
| `system/hydros.json`                            | JSON    | Yes      | Hydro plant registry                                                               |
| `system/thermals.json`                          | JSON    | Yes      | Thermal plant registry                                                             |
| `system/non_controllable_sources.json`          | JSON    | No       | Intermittent source registry                                                       |
| `system/pumping_stations.json`                  | JSON    | No       | Pumping station registry                                                           |
| `system/energy_contracts.json`                  | JSON    | No       | Bilateral energy contract registry                                                 |
| `system/hydro_geometry.parquet`                 | Parquet | No       | Reservoir geometry elevation tables                                                |
| `system/hydro_production_models.json`           | JSON    | No       | FPHA production function configs                                                   |
| `system/fpha_hyperplanes.parquet`               | Parquet | No       | FPHA hyperplane coefficients                                                       |
| `system/hydro_energy_productivity.parquet`      | Parquet | No       | Per-plant, per-stage energy-conversion overrides                                   |
| `system/tailrace_curves.parquet`                | Parquet | No       | Piecewise-quartic tailrace curves with backwater families                          |
| `system/scalar_parameters.json`                 | JSON    | No       | Scalar parameters for constraint expressions                                       |
| `scenarios/inflow_history.parquet`              | Parquet | No       | Historical inflow time series                                                      |
| `scenarios/inflow_seasonal_stats.parquet`       | Parquet | No       | PAR model seasonal statistics                                                      |
| `scenarios/inflow_ar_coefficients.parquet`      | Parquet | No       | PAR autoregressive coefficients                                                    |
| `scenarios/external_inflow_scenarios.parquet`   | Parquet | No       | External inflow scenario realizations (hydro_id, stage_id, scenario_id, value_m3s) |
| `scenarios/external_load_scenarios.parquet`     | Parquet | No       | External load scenario realizations (bus_id, stage_id, scenario_id, value_mw)      |
| `scenarios/external_ncs_scenarios.parquet`      | Parquet | No       | External NCS scenario realizations (ncs_id, stage_id, scenario_id, value)          |
| `scenarios/load_seasonal_stats.parquet`         | Parquet | No       | Load model seasonal statistics                                                     |
| `scenarios/load_factors.json`                   | JSON    | No       | Load scaling factors per bus/stage                                                 |
| `scenarios/non_controllable_factors.json`       | JSON    | No       | NCS block scaling factors per source/stage                                         |
| `scenarios/non_controllable_stats.parquet`      | Parquet | No       | NCS stochastic availability factors                                                |
| `scenarios/correlation.json`                    | JSON    | No       | Cross-series correlation model                                                     |
| `scenarios/noise_openings.parquet`              | Parquet | No       | User-supplied backward-pass opening tree                                           |
| `constraints/thermal_bounds.parquet`            | Parquet | No       | Stage-varying thermal generation bounds                                            |
| `constraints/hydro_bounds.parquet`              | Parquet | No       | Stage-varying hydro operational bounds                                             |
| `constraints/line_bounds.parquet`               | Parquet | No       | Stage-varying line flow capacity                                                   |
| `constraints/pumping_bounds.parquet`            | Parquet | No       | Stage-varying pumping flow bounds                                                  |
| `constraints/contract_bounds.parquet`           | Parquet | No       | Stage-varying contract power bounds                                                |
| `constraints/ncs_bounds.parquet`                | Parquet | No       | Stage-varying NCS available generation bounds                                      |
| `constraints/exchange_factors.json`             | JSON    | No       | Block exchange factors                                                             |
| `constraints/generic_constraints.json`          | JSON    | No       | User-defined LP constraints                                                        |
| `constraints/generic_constraint_bounds.parquet` | Parquet | No       | Generic constraint RHS bounds                                                      |
| `constraints/penalty_overrides_bus.parquet`     | Parquet | No       | Stage-varying bus excess cost                                                      |
| `constraints/penalty_overrides_line.parquet`    | Parquet | No       | Stage-varying line exchange cost                                                   |
| `constraints/penalty_overrides_hydro.parquet`   | Parquet | No       | Stage-varying hydro penalty costs                                                  |
| `constraints/penalty_overrides_ncs.parquet`     | Parquet | No       | Stage-varying NCS curtailment cost                                                 |

---

## Root-level files

### `config.json`

Controls all solver parameters. The `training` section is required; all other
sections are optional and fall back to documented defaults when absent.

**Top-level sections:**

| Section                  | Type   | Default      | Purpose                                                           |
| ------------------------ | ------ | ------------ | ----------------------------------------------------------------- |
| `$schema`                | string | `null`       | JSON Schema URI for editor validation (ignored during processing) |
| `modeling`               | object | `{}`         | Inflow non-negativity treatment                                   |
| `training`               | object | **required** | Iteration count, stopping rules, cut selection                    |
| `estimation`             | object | `{}`         | PAR(p) model estimation settings (max order, selection criterion) |
| `upper_bound_evaluation` | object | `{}`         | Inner approximation upper-bound settings                          |
| `policy`                 | object | fresh mode   | Policy directory path and warm-start mode                         |
| `simulation`             | object | disabled     | Post-training simulation settings                                 |
| `exports`                | object | all enabled  | Output file selection flags                                       |

**`modeling` section:**

| Field                                   | Type   | Default     | Description                                                                                                        |
| --------------------------------------- | ------ | ----------- | ------------------------------------------------------------------------------------------------------------------ |
| `modeling.inflow_non_negativity.method` | string | `"penalty"` | How to handle negative modelled inflows. One of `"none"`, `"penalty"`, `"truncation"`, `"truncation_with_penalty"` |

The per-hydro penalty coefficient applied to the inflow slack column is
authored in `penalties.json::hydro.inflow_nonnegativity_cost`.

**`training` section (mandatory fields):**

| Field                      | Type            | Default      | Description                                                                                     |
| -------------------------- | --------------- | ------------ | ----------------------------------------------------------------------------------------------- |
| `training.forward_passes`  | integer         | **required** | Number of scenario trajectories per iteration (>= 1)                                            |
| `training.stopping_rules`  | array           | **required** | At least one stopping rule entry; must include an `iteration_limit` rule                        |
| `training.stopping_mode`   | string          | `"any"`      | How multiple rules combine: `"any"` (stop when any triggers) or `"all"` (stop when all trigger) |
| `training.enabled`         | boolean         | `true`       | When `false`, skip training and proceed directly to simulation                                  |
| `training.tree_seed`       | integer or null | `null`       | Random seed for reproducible noise generation (see [Seed resolution](#seed-resolution))         |
| `training.scenario_source` | object or null  | `null`       | Per-class sampling scheme for the training forward pass (see below)                             |

**`training.scenario_source` sub-section:**

Configures which scenario sampling scheme is used for each entity class during training.
When absent, all classes default to `InSample` (PAR-based noise generation).

| Field                                       | Type            | Default       | Description                                                                                            |
| ------------------------------------------- | --------------- | ------------- | ------------------------------------------------------------------------------------------------------ |
| `training.scenario_source.inflow.scheme`    | string          | `"in_sample"` | Inflow sampling scheme: `"in_sample"`, `"historical"`, `"external"`, or `"out_of_sample"`              |
| `training.scenario_source.load.scheme`      | string          | `"in_sample"` | Load sampling scheme: `"in_sample"`, `"historical"`, `"external"`, or `"out_of_sample"`                |
| `training.scenario_source.ncs.scheme`       | string          | `"in_sample"` | NCS sampling scheme: `"in_sample"`, `"historical"`, `"external"`, or `"out_of_sample"`                 |
| `training.scenario_source.historical_years` | array or object | `null`        | Years eligible as inflow replay windows. List (`[2010, 2015]`) or range (`{"from": 2010, "to": 2023}`) |

#### Seed resolution

`training.tree_seed` in `config.json` is the **only** seed that controls noise generation
at runtime. It governs both the training forward pass and the post-training simulation.

- When `training.tree_seed` is a non-null integer, the CLI uses `|seed|` (unsigned absolute
  value) as the base seed for deterministic SipHash-1-3 noise generation. Results are
  bit-for-bit reproducible across runs with the same seed.
- When `training.tree_seed` is absent or `null`, the CLI applies a **default seed of 42**
  and prints a warning to stderr:

  ```
  warning: no random seed specified in config.json (training.tree_seed); using default seed 42. Set training.tree_seed for reproducible results.
  ```

  Runs will be reproducible (same output every time) but the seed value is arbitrary.
  Set `training.tree_seed` explicitly to make the choice intentional and visible to other
  users of the case directory.

**`training.stopping_rules` entries:**

Each entry has a `"type"` discriminator. Valid types:

| Type              | Required fields                                                       | Stops when                                                               |
| ----------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `iteration_limit` | `limit: integer`                                                      | Iteration count reaches `limit`                                          |
| `time_limit`      | `seconds: number`                                                     | Wall-clock time exceeds `seconds`                                        |
| `bound_stalling`  | `iterations: integer`, `tolerance: number`                            | Lower bound improvement falls below `tolerance` over `iterations` window |
| `simulation`      | `replications`, `period`, `bound_window`, `distance_tol`, `bound_tol` | Both policy cost and bound have stabilized                               |

**`training.cut_selection` sub-section:**

Two always-on knobs plus a tagged `selection` object that chooses the method
and carries only that method's parameters. Omitting `selection` disables row
selection. See the
[Configuration guide](../guide/configuration.md#cut_selection) for the full
per-method field tables.

| Field                    | Type    | Default | Description                                                                                            |
| ------------------------ | ------- | ------- | ------------------------------------------------------------------------------------------------------ |
| `row_activity_tolerance` | number  | `0.0`   | Minimum dual multiplier for a row to count as binding                                                  |
| `max_active_per_stage`   | integer | `null`  | Hard cap on active rows per stage; `null` = no cap                                                     |
| `selection`              | object  | `null`  | Active method and its parameters; `method` is one of `"level1"`, `"lml1"`, `"domination"`, `"dynamic"` |

**`upper_bound_evaluation` section:**

| Field                      | Type    | Default | Description                                   |
| -------------------------- | ------- | ------- | --------------------------------------------- |
| `enabled`                  | boolean | `null`  | Enable vertex-based inner approximation       |
| `initial_iteration`        | integer | `null`  | First iteration to compute the upper bound    |
| `interval_iterations`      | integer | `null`  | Iterations between upper-bound evaluations    |
| `lipschitz.mode`           | string  | `null`  | Lipschitz constant computation mode: `"auto"` |
| `lipschitz.fallback_value` | number  | `null`  | Fallback when automatic computation fails     |
| `lipschitz.scale_factor`   | number  | `null`  | Multiplicative safety margin                  |

**`policy` section:**

| Field                               | Type           | Default      | Description                                                            |
| ----------------------------------- | -------------- | ------------ | ---------------------------------------------------------------------- |
| `path`                              | string         | `"./policy"` | Directory for policy data (cuts, states, vertices, basis)              |
| `mode`                              | string         | `"fresh"`    | Initialization mode: `"fresh"`, `"warm_start"`, or `"resume"`          |
| `validate_compatibility`            | boolean        | `true`       | Verify entity and dimension compatibility when loading a stored policy |
| `boundary`                          | object or null | `null`       | Terminal boundary cut config: `path` (string) + `source_stage` (int)   |
| `checkpointing.enabled`             | boolean        | `null`       | Enable periodic checkpointing                                          |
| `checkpointing.initial_iteration`   | integer        | `null`       | First iteration to write a checkpoint                                  |
| `checkpointing.interval_iterations` | integer        | `null`       | Iterations between checkpoints                                         |
| `checkpointing.store_basis`         | boolean        | `null`       | Include LP basis in checkpoints                                        |
| `checkpointing.compress`            | boolean        | `null`       | Compress checkpoint files                                              |

**`simulation` section:**

| Field                                         | Type            | Default       | Description                                                                                            |
| --------------------------------------------- | --------------- | ------------- | ------------------------------------------------------------------------------------------------------ |
| `enabled`                                     | boolean         | `false`       | Enable post-training simulation                                                                        |
| `num_scenarios`                               | integer         | `2000`        | Number of simulation scenarios                                                                         |
| `io_channel_capacity`                         | integer         | `64`          | Channel capacity between simulation and I/O writer threads                                             |
| `simulation.scenario_source`                  | object or null  | `null`        | Per-class sampling scheme for the simulation pass (see below)                                          |
| `simulation.scenario_source.inflow.scheme`    | string          | `"in_sample"` | Inflow sampling scheme: `"in_sample"`, `"historical"`, `"external"`, or `"out_of_sample"`              |
| `simulation.scenario_source.load.scheme`      | string          | `"in_sample"` | Load sampling scheme: `"in_sample"`, `"historical"`, `"external"`, or `"out_of_sample"`                |
| `simulation.scenario_source.ncs.scheme`       | string          | `"in_sample"` | NCS sampling scheme: `"in_sample"`, `"historical"`, `"external"`, or `"out_of_sample"`                 |
| `simulation.scenario_source.historical_years` | array or object | `null`        | Years eligible as inflow replay windows. List (`[2010, 2015]`) or range (`{"from": 2010, "to": 2023}`) |

**`exports` section:**

| Field        | Type    | Default | Description                                                       |
| ------------ | ------- | ------- | ----------------------------------------------------------------- |
| `states`     | boolean | `false` | Export visited forward-pass trial points to the policy checkpoint |
| `stochastic` | boolean | `false` | Export stochastic preprocessing artifacts to `output/stochastic/` |

**Minimal valid example:**

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/config.schema.json",
  "training": {
    "forward_passes": 192,
    "stopping_rules": [{ "type": "iteration_limit", "limit": 200 }]
  }
}
```

---

### `penalties.json`

Global penalty cost defaults used when no entity-level override is present.
All four sections are required. Every scalar cost must be strictly positive (> 0.0).
Deficit segment costs must be monotonically increasing and the last segment must
have `depth_mw: null` (unbounded).

| Section                   | Field                                 | Type           | Description                                                                                                                            |
| ------------------------- | ------------------------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `bus`                     | `deficit_segments`                    | array          | Piecewise-linear deficit cost tiers                                                                                                    |
| `bus`                     | `deficit_segments[].depth_mw`         | number or null | Segment depth (MW); `null` for the final unbounded segment                                                                             |
| `bus`                     | `deficit_segments[].cost`             | number         | Cost per MWh of deficit in this tier (USD/MWh)                                                                                         |
| `bus`                     | `excess_cost`                         | number         | Cost per MWh of excess injection (USD/MWh)                                                                                             |
| `line`                    | `exchange_cost`                       | number         | Cost per MWh of inter-bus exchange flow (USD/MWh)                                                                                      |
| `hydro`                   | `spillage_cost`                       | number         | Spillage penalty                                                                                                                       |
| `hydro`                   | `turbined_cost`                       | number         | Turbined flow regularization cost (applied to every hydro)                                                                             |
| `hydro`                   | `diversion_cost`                      | number         | Diversion flow penalty                                                                                                                 |
| `hydro`                   | `storage_violation_below_cost`        | number         | Storage below-minimum violation penalty                                                                                                |
| `hydro`                   | `filling_target_violation_cost`       | number         | Filling target violation penalty                                                                                                       |
| `hydro`                   | `turbined_violation_below_cost`       | number         | Turbined flow below-minimum violation penalty                                                                                          |
| `hydro`                   | `outflow_violation_below_cost`        | number         | Total outflow below-minimum violation penalty                                                                                          |
| `hydro`                   | `outflow_violation_above_cost`        | number         | Total outflow above-maximum violation penalty                                                                                          |
| `hydro`                   | `generation_violation_below_cost`     | number         | Generation below-minimum violation penalty                                                                                             |
| `hydro`                   | `evaporation_violation_cost`          | number         | Symmetric evaporation violation penalty                                                                                                |
| `hydro`                   | `evaporation_violation_pos_cost`      | number or null | Optional over-evaporation override; supersedes `evaporation_violation_cost` for the positive direction. Omitted = symmetric value      |
| `hydro`                   | `evaporation_violation_neg_cost`      | number or null | Optional under-evaporation override; supersedes `evaporation_violation_cost` for the negative direction. Omitted = symmetric value     |
| `hydro`                   | `water_withdrawal_violation_cost`     | number         | Symmetric water withdrawal violation penalty                                                                                           |
| `hydro`                   | `water_withdrawal_violation_pos_cost` | number or null | Optional over-withdrawal override; supersedes `water_withdrawal_violation_cost` for the positive direction. Omitted = symmetric value  |
| `hydro`                   | `water_withdrawal_violation_neg_cost` | number or null | Optional under-withdrawal override; supersedes `water_withdrawal_violation_cost` for the negative direction. Omitted = symmetric value |
| `hydro`                   | `inflow_nonnegativity_cost`           | number or null | Optional inflow non-negativity penalty. Omitted = default `1000.0`                                                                     |
| `non_controllable_source` | `curtailment_cost`                    | number         | Curtailment penalty (USD/MWh)                                                                                                          |

**Example:**

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/penalties.schema.json",
  "bus": {
    "deficit_segments": [
      { "depth_mw": 500.0, "cost": 1000.0 },
      { "depth_mw": null, "cost": 5000.0 }
    ],
    "excess_cost": 100.0
  },
  "line": { "exchange_cost": 2.0 },
  "hydro": {
    "spillage_cost": 0.01,
    "turbined_cost": 0.05,
    "diversion_cost": 0.1,
    "storage_violation_below_cost": 10000.0,
    "filling_target_violation_cost": 50000.0,
    "turbined_violation_below_cost": 500.0,
    "outflow_violation_below_cost": 500.0,
    "outflow_violation_above_cost": 500.0,
    "generation_violation_below_cost": 1000.0,
    "evaporation_violation_cost": 5000.0,
    "water_withdrawal_violation_cost": 1000.0
  },
  "non_controllable_source": { "curtailment_cost": 0.005 }
}
```

---

### `stages.json`

Defines the temporal structure of the study: stage sequence, block decomposition,
and policy graph horizon type.

**Top-level fields:**

| Field                | Required | Description                                                                    |
| -------------------- | -------- | ------------------------------------------------------------------------------ |
| `policy_graph`       | Yes      | Horizon type (`"finite_horizon"`), annual discount rate, and stage transitions |
| `stages`             | Yes      | Array of study stage definitions                                               |
| `season_definitions` | No       | Season labeling for seasonal model alignment                                   |
| `pre_study_stages`   | No       | Pre-study stages for AR model warm-up (negative IDs)                           |

> **Migration note (v0.4.0):** `scenario_source` has moved from `stages.json` to
> `config.json`. Training and simulation now carry independent `scenario_source`
> sub-objects under `training.scenario_source` and `simulation.scenario_source`
> respectively. A `scenario_source` key at the top level of `stages.json` is no
> longer read; move it to `config.json` and split it per-pass as needed.

**`stages[]` entry fields:**

| Field             | Required | Description                                                       |
| ----------------- | -------- | ----------------------------------------------------------------- |
| `id`              | Yes      | Stage identifier (non-negative integer, unique)                   |
| `start_date`      | Yes      | ISO 8601 date (e.g., `"2024-01-01"`)                              |
| `end_date`        | Yes      | ISO 8601 date; must be after `start_date`                         |
| `blocks`          | Yes      | Array of load blocks (`id`, `name`, `hours`)                      |
| `num_scenarios`   | Yes      | Number of forward-pass scenarios for this stage (>= 1)            |
| `season_id`       | No       | Reference to a season in `season_definitions`                     |
| `block_mode`      | No       | Block execution mode: `"parallel"` (default) or `"chronological"` |
| `state_variables` | No       | Which state variables are active: `storage`, `inflow_lags`        |
| `risk_measure`    | No       | Per-stage risk measure: `"expectation"` or CVaR config            |
| `sampling_method` | No       | Noise method: `"saa"` or other variants                           |

**`season_definitions` sub-object:**

The optional `season_definitions` object maps season IDs to calendar periods for the PAR model.
When absent, Cobre infers 12 monthly seasons from stage dates. When present, it controls
how `season_id` values on stages translate to stochastic parameters.

| Field        | Required | Description                            |
| ------------ | -------- | -------------------------------------- |
| `cycle_type` | Yes      | `"monthly"`, `"weekly"`, or `"custom"` |
| `seasons`    | Yes      | Array of season entries (see below)    |

**`season_definitions.seasons[]` entry fields:**

| Field         | Required    | Description                                                                    |
| ------------- | ----------- | ------------------------------------------------------------------------------ |
| `id`          | Yes         | Season identifier (0-based integer, unique within the season map)              |
| `label`       | Yes         | Human-readable label (e.g., `"January"`, `"Q1"`, `"Wet Season"`)               |
| `month_start` | Yes         | Calendar month where the season starts (1–12)                                  |
| `day_start`   | Custom only | Calendar day where the season starts (1–31). Required for `custom` cycle type. |
| `month_end`   | Custom only | Calendar month where the season ends (1–12). Required for `custom` cycle type. |
| `day_end`     | Custom only | Calendar day where the season ends (1–31). Required for `custom` cycle type.   |

**Cycle types:**

- `"monthly"` — seasons map to calendar months (12 seasons, 0 = January, ..., 11 = December).
  Only `id`, `label`, and `month_start` are needed per entry.
- `"weekly"` — seasons map to ISO calendar weeks (52 seasons). Only `id`, `label`,
  and `month_start` are needed per entry.
- `"custom"` — user-defined date ranges with explicit `month_start`/`day_start`/`month_end`/`day_end`.
  All four boundary fields are required. Use this cycle type for mixed-resolution studies
  where some stages are monthly (IDs 0–11) and others are quarterly (IDs 12–15).

**Example — Custom cycle type with monthly and quarterly seasons:**

```json
{
  "season_definitions": {
    "cycle_type": "custom",
    "seasons": [
      {
        "id": 0,
        "label": "January",
        "month_start": 1,
        "day_start": 1,
        "month_end": 2,
        "day_end": 1
      },
      {
        "id": 1,
        "label": "February",
        "month_start": 2,
        "day_start": 1,
        "month_end": 3,
        "day_end": 1
      },
      {
        "id": 11,
        "label": "December",
        "month_start": 12,
        "day_start": 1,
        "month_end": 1,
        "day_end": 1
      },
      {
        "id": 12,
        "label": "Q1",
        "month_start": 1,
        "day_start": 1,
        "month_end": 4,
        "day_end": 1
      },
      {
        "id": 13,
        "label": "Q2",
        "month_start": 4,
        "day_start": 1,
        "month_end": 7,
        "day_end": 1
      },
      {
        "id": 14,
        "label": "Q3",
        "month_start": 7,
        "day_start": 1,
        "month_end": 10,
        "day_end": 1
      },
      {
        "id": 15,
        "label": "Q4",
        "month_start": 10,
        "day_start": 1,
        "month_end": 1,
        "day_end": 1
      }
    ]
  }
}
```

In this example, seasons 0–11 cover monthly PAR models for the near-term phase and seasons 12–15
cover quarterly PAR models for the long-term phase. Each monthly stage assigns a `season_id` of 0–11;
each quarterly stage assigns a `season_id` of 12–15. Rule 29 enforces that stages sharing the same
`season_id` must have similar durations (within 7 days), so monthly and quarterly stages must use
distinct season IDs.

---

### `initial_conditions.json`

Initial reservoir storage, past inflow lags, and recent observations at the
start of the study.

| Field                 | Required | Description                                                                                                       |
| --------------------- | -------- | ----------------------------------------------------------------------------------------------------------------- |
| `storage`             | Yes      | Array of `{ "hydro_id": integer, "value_hm3": number }` entries for operating hydros                              |
| `filling_storage`     | Yes      | Array of `{ "hydro_id": integer, "value_hm3": number }` entries for filling hydros                                |
| `past_inflows`        | No       | Array of `{ "hydro_id": integer, "values_m3s": [number], "season_ids": [integer] }` for PAR(p) lag initialization |
| `recent_observations` | No       | Array of observed inflow entries for mid-season study starts (see below)                                          |

Each `hydro_id` must be unique within its array and must not appear in both
`storage` and `filling_storage`. All `value_hm3` values must be non-negative.

**`past_inflows`** provides the most-recent inflow history for PAR(p) lag
initialization. For each hydro, `values_m3s[0]` is the most recent past inflow
(lag 1) and `values_m3s[p-1]` is the oldest (lag p). The array length must be

> = the hydro's PAR order. Optional; defaults to an empty array when absent.

Each `past_inflows` entry supports an optional `season_ids` field:

| Field        | Type             | Description                                                                                                                                                                                                                            |
| ------------ | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hydro_id`   | integer          | Hydro plant identifier                                                                                                                                                                                                                 |
| `values_m3s` | array of number  | Past inflow values [m³/s], most recent first                                                                                                                                                                                           |
| `season_ids` | array of integer | Optional. Season IDs corresponding to each lag entry. When present, length must equal `values_m3s.length`. Each value must reference a valid season ID from `season_definitions`. Absent from legacy JSON files (backward compatible). |

When `season_ids` is present and a season ID is not defined in `season_definitions`, a
`BusinessRuleViolation` is emitted during semantic validation (Rule 32) when the
hydro has PAR order > 0 and a `SeasonMap` is available.

**`recent_observations`** provides observed inflow data for partial periods
before the study start. Used to seed the lag accumulator when a study begins
mid-season (e.g., a coupled study starting on January 5 needs observed inflow
for January 1--4). Each entry has:

| Field        | Type    | Description                                                      |
| ------------ | ------- | ---------------------------------------------------------------- |
| `hydro_id`   | integer | Hydro plant identifier                                           |
| `start_date` | string  | Start of the observation period (inclusive), ISO 8601 YYYY-MM-DD |
| `end_date`   | string  | End of the observation period (exclusive), ISO 8601 YYYY-MM-DD   |
| `value_m3s`  | number  | Average inflow observed during the period, in m³/s               |

Date ranges for the same hydro must not overlap; adjacent ranges
(`start_date == previous end_date`) are accepted. Values must be finite and
non-negative. Optional; defaults to an empty array when absent. Existing cases
without this field are unaffected.

Example:

```json
{
  "storage": [{ "hydro_id": 0, "value_hm3": 15000.0 }],
  "filling_storage": [],
  "past_inflows": [{ "hydro_id": 0, "values_m3s": [600.0, 500.0] }],
  "recent_observations": [
    {
      "hydro_id": 0,
      "start_date": "2026-04-01",
      "end_date": "2026-04-04",
      "value_m3s": 500.0
    },
    {
      "hydro_id": 0,
      "start_date": "2026-04-04",
      "end_date": "2026-04-11",
      "value_m3s": 480.0
    }
  ]
}
```

---

## `system/` files

### `system/buses.json`

Electrical bus registry. Buses are the nodes of the transmission network.

| Field                                 | Required | Description                                                                               |
| ------------------------------------- | -------- | ----------------------------------------------------------------------------------------- |
| `buses[].id`                          | Yes      | Bus identifier (integer, unique)                                                          |
| `buses[].name`                        | Yes      | Human-readable bus name (string)                                                          |
| `buses[].deficit_segments`            | No       | Entity-level deficit cost tiers; when absent, global defaults from `penalties.json` apply |
| `buses[].deficit_segments[].depth_mw` | No       | Segment MW depth; `null` for the final unbounded segment                                  |
| `buses[].deficit_segments[].cost`     | No       | Cost per MWh of deficit in this tier (USD/MWh)                                            |

---

### `system/lines.json`

Transmission line registry. Lines connect buses and carry power flows.

| Field                         | Required | Description                                                          |
| ----------------------------- | -------- | -------------------------------------------------------------------- |
| `lines[].id`                  | Yes      | Line identifier (integer, unique)                                    |
| `lines[].name`                | Yes      | Human-readable line name (string)                                    |
| `lines[].source_bus_id`       | Yes      | Sending-end bus ID                                                   |
| `lines[].target_bus_id`       | Yes      | Receiving-end bus ID                                                 |
| `lines[].entry_stage_id`      | No       | Stage when line enters service; `null` = always exists               |
| `lines[].exit_stage_id`       | No       | Stage when line is decommissioned; `null` = never                    |
| `lines[].capacity.direct_mw`  | Yes      | Maximum power flow in the direct direction (MW)                      |
| `lines[].capacity.reverse_mw` | Yes      | Maximum power flow in the reverse direction (MW)                     |
| `lines[].exchange_cost`       | No       | Entity-level exchange cost override ($/MWh); absent = global default |
| `lines[].losses_percent`      | No       | Transmission losses as percentage (default: 0.0)                     |

---

### `system/hydros.json`

Hydro plant registry. Each entry defines a complete hydro plant with reservoir,
turbine, and optional cascade linkage.

Key fields:

| Field                                             | Required | Description                                                                                                   |
| ------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------- |
| `hydros[].id`                                     | Yes      | Plant identifier (integer, unique)                                                                            |
| `hydros[].name`                                   | Yes      | Human-readable plant name                                                                                     |
| `hydros[].bus_id`                                 | Yes      | Bus where generation is injected                                                                              |
| `hydros[].downstream_id`                          | No       | Downstream plant ID in the cascade; `null` = tailwater                                                        |
| `hydros[].entry_stage_id`                         | No       | Stage when plant enters service; `null` = always exists                                                       |
| `hydros[].exit_stage_id`                          | No       | Stage when plant is decommissioned; `null` = never                                                            |
| `hydros[].reservoir`                              | Yes      | `min_storage_hm3` and `max_storage_hm3` (both >= 0)                                                           |
| `hydros[].outflow`                                | Yes      | `min_outflow_m3s` and `max_outflow_m3s` total outflow bounds                                                  |
| `hydros[].generation`                             | Yes      | Generation model: `model`, turbine flow bounds, generation MW bounds                                          |
| `hydros[].generation.model`                       | Yes      | `"constant_productivity"`, `"linearized_head"`, or `"fpha"`                                                   |
| `hydros[].specific_productivity_mw_per_m3s_per_m` | No       | Specific productivity ρ_esp [MW/(m³/s)/m]. Required for FPHA hydros that rely on VHA geometry to derive ρ_eq. |
| `hydros[].tailrace`                               | No       | Tailrace model: `"polynomial"` or `"piecewise"`                                                               |
| `hydros[].hydraulic_losses`                       | No       | Head loss model: `"factor"` or `"constant"`                                                                   |
| `hydros[].efficiency`                             | No       | Turbine efficiency model: `"constant"`                                                                        |
| `hydros[].evaporation`                            | No       | Evaporation config: `coefficients_mm` (12 values) and optional `reference_volumes_hm3`                        |
| `hydros[].diversion`                              | No       | Diversion channel: `downstream_id` and `max_flow_m3s`                                                         |
| `hydros[].filling`                                | No       | Filling config: `start_stage_id` and `filling_min_rate_m3s`                                                   |
| `hydros[].penalties`                              | No       | Entity-level hydro penalty overrides (all fields optional, fall back to global)                               |

All fields within `hydros[].penalties` are optional. When a field is absent the
global default from `penalties.json` is used. The following fields are supported:

| Field within `penalties`              | Optional | Description                                                                                                                               |
| ------------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `spillage_cost`                       | Yes      | Spillage penalty ($/m³/s).                                                                                                                |
| `turbined_cost`                       | Yes      | Turbined flow regularization cost; applied to every hydro's turbine column in the LP objective.                                           |
| `diversion_cost`                      | Yes      | Diversion flow penalty.                                                                                                                   |
| `storage_violation_below_cost`        | Yes      | Storage below-minimum violation penalty.                                                                                                  |
| `filling_target_violation_cost`       | Yes      | Filling target violation penalty.                                                                                                         |
| `turbined_violation_below_cost`       | Yes      | Turbined flow below-minimum violation penalty.                                                                                            |
| `outflow_violation_below_cost`        | Yes      | Total outflow below-minimum violation penalty.                                                                                            |
| `outflow_violation_above_cost`        | Yes      | Total outflow above-maximum violation penalty.                                                                                            |
| `generation_violation_below_cost`     | Yes      | Generation below-minimum violation penalty.                                                                                               |
| `evaporation_violation_cost`          | Yes      | Symmetric evaporation violation penalty (applies to both directions when directional fields are absent).                                  |
| `water_withdrawal_violation_cost`     | Yes      | Symmetric water withdrawal violation penalty (applies to both directions when directional fields are absent).                             |
| `water_withdrawal_violation_pos_cost` | Yes      | Override cost for over-withdrawal violations (actual > target). Supersedes `water_withdrawal_violation_cost` for the positive direction.  |
| `water_withdrawal_violation_neg_cost` | Yes      | Override cost for under-withdrawal violations (actual < target). Supersedes `water_withdrawal_violation_cost` for the negative direction. |
| `evaporation_violation_pos_cost`      | Yes      | Override cost for over-evaporation violations (actual > modelled). Supersedes `evaporation_violation_cost` for the positive direction.    |
| `evaporation_violation_neg_cost`      | Yes      | Override cost for under-evaporation violations (actual < modelled). Supersedes `evaporation_violation_cost` for the negative direction.   |
| `inflow_nonnegativity_cost`           | Yes      | Override global inflow non-negativity penalty cost for this plant ($/m³/s).                                                               |

---

### `system/thermals.json`

Thermal plant registry. Each entry defines a dispatchable generation unit.

| Field                           | Required | Description                                                        |
| ------------------------------- | -------- | ------------------------------------------------------------------ |
| `thermals[].id`                 | Yes      | Plant identifier (integer, unique)                                 |
| `thermals[].name`               | Yes      | Human-readable plant name                                          |
| `thermals[].bus_id`             | Yes      | Bus where generation is injected                                   |
| `thermals[].generation`         | Yes      | Dispatch-bounds object with `min_mw` and `max_mw`                  |
| `thermals[].generation.min_mw`  | Yes      | Minimum dispatch level (MW)                                        |
| `thermals[].generation.max_mw`  | Yes      | Maximum dispatch level (MW)                                        |
| `thermals[].cost_per_mwh`       | Yes      | Linear generation cost (USD/MWh)                                   |
| `thermals[].entry_stage_id`     | No       | Stage when the unit enters service (`null` = present from stage 0) |
| `thermals[].exit_stage_id`      | No       | Stage when the unit is decommissioned (`null` = never)             |
| `thermals[].anticipated_config` | No       | Anticipated-dispatch config (object with `lead_stages` ≥ 1)        |

---

### `system/pumping_stations.json`

Pumping station registry. Each entry defines a pumped-storage or water-transfer
installation that withdraws water from a source hydro reservoir, injects it into a
destination hydro reservoir, and consumes electrical power from a bus. The file is
optional; when absent, no pumping stations are modeled.

| Field                                       | Required | Description                                                                     |
| ------------------------------------------- | -------- | ------------------------------------------------------------------------------- |
| `pumping_stations[].id`                     | Yes      | Station identifier (integer, unique)                                            |
| `pumping_stations[].name`                   | Yes      | Human-readable station name (string)                                            |
| `pumping_stations[].bus_id`                 | Yes      | Bus from which electrical power is consumed                                     |
| `pumping_stations[].source_hydro_id`        | Yes      | Hydro plant from whose reservoir water is extracted                             |
| `pumping_stations[].destination_hydro_id`   | Yes      | Hydro plant into whose reservoir water is injected                              |
| `pumping_stations[].consumption_mw_per_m3s` | Yes      | Power drawn per unit of pumped flow [MW/(m³/s)]; must be >= 0                   |
| `pumping_stations[].entry_stage_id`         | No       | Stage when the station enters service; `null` or absent = present from stage 0  |
| `pumping_stations[].exit_stage_id`          | No       | Stage when the station is decommissioned; `null` or absent = never              |
| `pumping_stations[].flow`                   | Yes      | Nested object with `min_m3s` and `max_m3s` (see below)                          |
| `pumping_stations[].flow.min_m3s`           | Yes      | Minimum pumped flow [m³/s]; must be >= 0                                        |
| `pumping_stations[].flow.max_m3s`           | Yes      | Maximum pumped flow (installed pump capacity) [m³/s]; must be >= `flow.min_m3s` |

The pumped flow variable is bounded by `[flow.min_m3s, flow.max_m3s]` in the LP.
At each stage within `[entry_stage_id, exit_stage_id)`, the flow appears with
a negative sign in the source reservoir water-balance row and a positive sign in
the destination reservoir water-balance row. Power consumed equals
`consumption_mw_per_m3s × flow_m3s` and is charged as load on the station's bus.
Stage-varying flow bounds can be overridden via `constraints/pumping_bounds.parquet`.

**Minimal valid example:**

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/pumping_stations.schema.json",
  "pumping_stations": [
    {
      "id": 0,
      "name": "Bombeamento Serra da Mesa",
      "bus_id": 10,
      "source_hydro_id": 3,
      "destination_hydro_id": 5,
      "consumption_mw_per_m3s": 0.5,
      "flow": { "min_m3s": 0.0, "max_m3s": 150.0 }
    }
  ]
}
```

---

### `system/energy_contracts.json`

Energy contract registry. Each entry defines a bilateral energy purchase or sale
obligation with a counterparty outside the modeled system. The file is optional;
when absent, no contracts are modeled.

| Field                        | Required | Description                                                                                |
| ---------------------------- | -------- | ------------------------------------------------------------------------------------------ |
| `contracts[].id`             | Yes      | Contract identifier (integer, unique)                                                      |
| `contracts[].name`           | Yes      | Human-readable contract name (string)                                                      |
| `contracts[].bus_id`         | Yes      | Bus where power is injected (import) or withdrawn (export)                                 |
| `contracts[].type`           | Yes      | Energy flow direction: `"import"` or `"export"`                                            |
| `contracts[].price_per_mwh`  | Yes      | Contract price [monetary units/MWh]. Positive = cost (import); negative = revenue (export) |
| `contracts[].limits.min_mw`  | Yes      | Minimum dispatch level [MW]; use `0.0` unless a take-or-pay floor applies                  |
| `contracts[].limits.max_mw`  | Yes      | Maximum dispatch level [MW]; must be >= `limits.min_mw`                                    |
| `contracts[].entry_stage_id` | No       | Stage when the contract enters service; `null` or absent = present from stage 0            |
| `contracts[].exit_stage_id`  | No       | Stage when the contract is decommissioned; `null` or absent = never                        |

At each active stage within `[entry_stage_id, exit_stage_id)`, the LP adds one
column per block per direction bounded by `[limits.min_mw, limits.max_mw]`. An
import column injects `+1.0` MW into the bus power-balance row; an export column
withdraws `−1.0` MW. At dormant stages the column bounds are pinned to `[0, 0]`
and the output row is emitted with `power_mw = 0`. Stage-varying bounds and prices
can be overridden via `constraints/contract_bounds.parquet`.

**Minimal valid example:**

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/energy_contracts.schema.json",
  "contracts": [
    {
      "id": 0,
      "name": "Import base load",
      "bus_id": 0,
      "type": "import",
      "price_per_mwh": 200.0,
      "limits": { "min_mw": 0.0, "max_mw": 50.0 }
    },
    {
      "id": 1,
      "name": "Export revenue (stage 1 only)",
      "bus_id": 0,
      "type": "export",
      "entry_stage_id": 1,
      "exit_stage_id": 2,
      "price_per_mwh": -150.0,
      "limits": { "min_mw": 0.0, "max_mw": 30.0 }
    }
  ]
}
```

---

### `system/hydro_geometry.parquet`

Volume-Height-Area (VHA) curves for hydro reservoirs. Required when any hydro is
configured with a computed FPHA production model (`source: "computed"`) or with
evaporation linearization. When absent, FPHA computation and evaporation
linearization are unavailable for all plants.

4 columns, all non-nullable. Rows are sorted by `(hydro_id, volume_hm3)` ascending.
Multiple rows per `hydro_id` together constitute the VHA curve for that plant.

| Column       | Type   | Required | Description                                                              |
| ------------ | ------ | -------- | ------------------------------------------------------------------------ |
| `hydro_id`   | INT32  | Yes      | Hydro plant ID                                                           |
| `volume_hm3` | DOUBLE | Yes      | Total reservoir volume at this point (hm³). Non-negative and finite.     |
| `height_m`   | DOUBLE | Yes      | Reservoir surface elevation at this volume (m). Non-negative and finite. |
| `area_km2`   | DOUBLE | Yes      | Water surface area at this volume (km²). Non-negative and finite.        |

**Validation:** all four columns must be present with the correct types. `volume_hm3`,
`height_m`, and `area_km2` must be non-negative and finite. Monotonicity of
`volume_hm3` within each hydro is enforced during Layer 5 semantic validation.

---

### `system/hydro_production_models.json`

Per-hydro production function assignment. The file is required whenever
the case contains at least one non-FPHA hydro: each non-FPHA plant must
have a matching entry that supplies either an inline
`productivity_mw_per_m3s` per stage range / season, or defers to
`system/hydro_energy_productivity.parquet` for that `(hydro, stage)`
coefficient.

The file contains a `"production_models"` array. Each entry configures one hydro
plant and is identified by a unique `hydro_id`. Results are loaded in
`hydro_id`-ascending order regardless of declaration order.

**Top-level structure:**

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/production_models.schema.json",
  "production_models": [ ... ]
}
```

**Per-hydro entry fields:**

| Field            | Required | Description                                                                 |
| ---------------- | -------- | --------------------------------------------------------------------------- |
| `hydro_id`       | Yes      | Hydro plant ID. Must be unique within the file.                             |
| `selection_mode` | Yes      | How the model variant is chosen per stage: `"stage_ranges"` or `"seasonal"` |

**`stage_ranges` mode.** The model for each stage is determined by the first
matching `[start_stage_id, end_stage_id]` range. `end_stage_id` may be `null`
to mean "until end of horizon".

| Field within each range   | Required | Description                                                                                                                                                                                                                                                                                                                                                    |
| ------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `start_stage_id`          | Yes      | First stage (inclusive) to which this entry applies                                                                                                                                                                                                                                                                                                            |
| `end_stage_id`            | Yes      | Last stage (inclusive); `null` means open-ended                                                                                                                                                                                                                                                                                                                |
| `model`                   | Yes      | Model name: `"constant_productivity"`, `"linearized_head"`, or `"fpha"`                                                                                                                                                                                                                                                                                        |
| `fpha_config`             | No       | Required when `model` is `"fpha"`. See FPHA config fields below.                                                                                                                                                                                                                                                                                               |
| `reference_volume`        | No       | Reference operating volume V_ref, a sibling of `fpha_config` (not nested). Set exactly one of `volume_hm3` (absolute, hm³, `> 0.0`) or `percentile` (a fraction of the operating range, `[0.0, 1.0]`); both or neither is rejected. Absent ⇒ the case-wide default fraction. Applies to any plant in either selection mode. See reference-volume fields below. |
| `productivity_mw_per_m3s` | No       | Positive when present; rejected on `"fpha"`. Optional for `constant_productivity` and `linearized_head` — when omitted, supply the value via `system/hydro_energy_productivity.parquet`. Exactly one source per `(hydro, stage)` is required; both is rejected at load time.                                                                                   |

**`seasonal` mode.** The model for a stage is determined by its `season_id`.
Stages whose season is not listed use `default_model`.

| Field           | Required | Description                                                                                                            |
| --------------- | -------- | ---------------------------------------------------------------------------------------------------------------------- |
| `default_model` | Yes      | Fallback model name for unlisted seasons                                                                               |
| `seasons`       | Yes      | Array of season overrides: `season_id`, `model`, optional `fpha_config`, `reference_volume`, `productivity_mw_per_m3s` |

**`reference_volume` fields (optional sibling of `fpha_config`):**

| Field        | Required | Description                                                                                                                    |
| ------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `volume_hm3` | No       | Absolute reference volume [hm³]; finite and `> 0.0`. Mutually exclusive with `percentile`.                                     |
| `percentile` | No       | Reference volume as a fraction of the `[V_min, V_max]` band; finite and in `[0.0, 1.0]`. Mutually exclusive with `volume_hm3`. |

The reference operating volume V_ref feeds the FPHA backwater (downstream
forebay) level and the energy-equivalent productivity ρ_eq. It is the single
source of truth for V_ref: when absent, the case-wide default fraction is used.

**`fpha_config` fields (required when `model` is `"fpha"`):**

| Field                            | Required | Default        | Description                                                   |
| -------------------------------- | -------- | -------------- | ------------------------------------------------------------- |
| `source`                         | Yes      | —              | `"precomputed"` or `"computed"`                               |
| `volume_discretization_points`   | No       | solver default | Number of volume grid points for hyperplane computation       |
| `turbine_discretization_points`  | No       | solver default | Number of turbine-flow grid points for hyperplane computation |
| `spillage_discretization_points` | No       | solver default | Number of spillage grid points for hyperplane computation     |
| `max_planes_per_hydro`           | No       | solver default | Maximum hyperplanes per plant after selection heuristic       |
| `fitting_window`                 | No       | full range     | Volume range restriction for hyperplane computation           |

`source: "precomputed"` means the hyperplanes are loaded from
`system/fpha_hyperplanes.parquet`. `source: "computed"` means Cobre derives
them from `system/hydro_geometry.parquet`; in this case `hydro_geometry.parquet`
must be present and the computed planes are automatically written to
`output/hydro_models/fpha_hyperplanes.parquet`.

**`fitting_window` fields.** Absolute bounds (`volume_min_hm3`, `volume_max_hm3`)
and percentile bounds (`volume_min_percentile`, `volume_max_percentile`) are
mutually exclusive — set one pair or the other, not both.

| Field                   | Type   | Description                                          |
| ----------------------- | ------ | ---------------------------------------------------- |
| `volume_min_hm3`        | number | Explicit minimum volume for fitting (hm³)            |
| `volume_max_hm3`        | number | Explicit maximum volume for fitting (hm³)            |
| `volume_min_percentile` | number | Minimum as a percentile of the operating range (0–1) |
| `volume_max_percentile` | number | Maximum as a percentile of the operating range (0–1) |

**Example — hydro 0 uses computed FPHA for stages 0–24, then constant productivity:**

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/production_models.schema.json",
  "production_models": [
    {
      "hydro_id": 0,
      "selection_mode": "stage_ranges",
      "stage_ranges": [
        {
          "start_stage_id": 0,
          "end_stage_id": 24,
          "model": "fpha",
          "fpha_config": {
            "source": "computed",
            "volume_discretization_points": 7,
            "turbine_discretization_points": 15
          }
        },
        {
          "start_stage_id": 25,
          "end_stage_id": null,
          "model": "constant_productivity",
          "productivity_mw_per_m3s": 0.72
        }
      ]
    }
  ]
}
```

**Example — hydro 5 uses FPHA in season 0, linearized_head in all other seasons:**

```json
{
  "production_models": [
    {
      "hydro_id": 5,
      "selection_mode": "seasonal",
      "default_model": "linearized_head",
      "seasons": [
        {
          "season_id": 0,
          "model": "fpha",
          "fpha_config": { "source": "precomputed" }
        }
      ]
    }
  ]
}
```

---

### `system/fpha_hyperplanes.parquet`

Pre-computed FPHA hyperplane coefficients for hydros configured with
`fpha_config.source: "precomputed"`. When absent, only `"computed"` source is
available.

11 columns. Rows are sorted by `(hydro_id, stage_id, plane_id)` ascending.
Null `stage_id` sorts before any non-null stage and means the plane is valid for
all stages of that hydro. One row per hyperplane; at least 3 planes are required
per `(hydro_id, stage_id)` group.

| Column            | Type   | Nullable | Description                                                 |
| ----------------- | ------ | -------- | ----------------------------------------------------------- |
| `hydro_id`        | INT32  | No       | Hydro plant ID                                              |
| `stage_id`        | INT32  | Yes      | Stage the plane applies to. `null` = valid for all stages   |
| `plane_id`        | INT32  | No       | Plane index within this hydro (and stage)                   |
| `gamma_0`         | DOUBLE | No       | Intercept coefficient (MW)                                  |
| `gamma_v`         | DOUBLE | No       | Volume coefficient (MW/hm³). Positive.                      |
| `gamma_q`         | DOUBLE | No       | Turbined flow coefficient (MW per m³/s)                     |
| `gamma_s`         | DOUBLE | No       | Spillage coefficient (MW per m³/s). Typically non-positive. |
| `kappa`           | DOUBLE | Yes      | Correction factor. Defaults to `1.0` when absent or null.   |
| `valid_v_min_hm3` | DOUBLE | Yes      | Volume range minimum where this plane is valid (hm³)        |
| `valid_v_max_hm3` | DOUBLE | Yes      | Volume range maximum where this plane is valid (hm³)        |
| `valid_q_max_m3s` | DOUBLE | Yes      | Maximum turbined flow where this plane is valid (m³/s)      |

**Validation:** required columns (`hydro_id`, `plane_id`, `gamma_0`, `gamma_v`,
`gamma_q`, `gamma_s`) must be present with the correct types. Optional columns
that are present must also have the correct types. Minimum planes per
`(hydro_id, stage_id)` group and sign constraints on `gamma_v` and `gamma_s`
are enforced during Layer 5 semantic validation.

The file produced by `output/hydro_models/fpha_hyperplanes.parquet` (written when
`source: "computed"` is used) has this exact same 11-column schema and is
suitable for use as a future precomputed input.

---

### `system/hydro_energy_productivity.parquet`

Optional per-plant, per-stage overrides for the energy-conversion preprocessing
layer. When present, any non-null column in a matching row replaces the value
that would otherwise be derived from VHA geometry or plant defaults. Rows with
`stage_id = NULL` act as per-hydro defaults and apply to all stages not covered
by a stage-specific row.

| Column                                   | Parquet type | Nullable | Description                                                                              |
| ---------------------------------------- | ------------ | -------- | ---------------------------------------------------------------------------------------- |
| `hydro_id`                               | INT32        | no       | Hydro plant identifier                                                                   |
| `stage_id`                               | INT32        | yes      | Stage; NULL means "applies to all stages"                                                |
| `equivalent_productivity_mw_per_m3s`     | DOUBLE       | yes      | Direct ρ_eq override [MW/(m³/s)]; finite and >= 0.0 (`0.0` marks a planned-outage stage) |
| `reference_outflow_m3s`                  | DOUBLE       | yes      | Q_ref override [m³/s]; finite and >= 0.0                                                 |
| `specific_productivity_mw_per_m3s_per_m` | DOUBLE       | yes      | ρ_esp override [MW/(m³/s)/m]; finite and > 0.0                                           |

**Validation:**

- `hydro_id` must not be null.
- `equivalent_productivity_mw_per_m3s`, when set, must be finite and >= 0.0;
  `0.0` is accepted as a planned-outage marker.
- `reference_outflow_m3s`, when set, must be finite and >= 0.0.
- `specific_productivity_mw_per_m3s_per_m`, when set, must be finite and >= 0.0;
  `0.0` mirrors the `equivalent_productivity_mw_per_m3s` planned-outage marker.
- A row where all three override columns are NULL is accepted.
- Duplicate `(hydro_id, stage_id)` pairs are rejected during case build.
- The reference operating volume V_ref is no longer an override column here; it
  is declared per `(plant, stage)` via `reference_volume` in
  `system/hydro_production_models.json`. A legacy `reference_volume_hm3` column,
  if still present, is ignored (a one-time warning is emitted).

---

### `system/tailrace_curves.parquet`

Optional piecewise-quartic tailrace-level curves that replace the entity-level
tailrace model for any plant that has rows in this file. When a plant has rows
here, the computed-FPHA pipeline evaluates its tailrace level from these
piecewise-quartic curves — selecting the segment by downstream flow and
interpolating between backwater families at the downstream plant's stage
reference level — instead of the tailrace model declared in `hydros.json`.
Plants without a row in this file keep their existing tailrace model; the file
is inert (silently skipped) when absent from the case directory.

Rows are sorted by `(hydro_id, family_id, segment_id)` ascending. A complete
curve for one backwater family consists of multiple rows sharing
`(hydro_id, family_id)`.

| Column                         | Type   | Nullable | Description                                                                                                                         |
| ------------------------------ | ------ | -------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `hydro_id`                     | INT32  | No       | Plant whose tailrace this describes                                                                                                 |
| `family_id`                    | INT32  | No       | Family index within the plant (sequential grouping key)                                                                             |
| `downstream_reference_level_m` | DOUBLE | Yes      | Downstream reservoir reference level keying this family (m). `null` when the plant has a single family and no backwater dependency. |
| `segment_id`                   | INT32  | No       | Piece index within the family                                                                                                       |
| `outflow_min_m3s`              | DOUBLE | No       | Segment lower validity bound (m³/s). Non-negative.                                                                                  |
| `outflow_max_m3s`              | DOUBLE | No       | Segment upper validity bound (m³/s). Non-negative, >= `outflow_min_m3s`.                                                            |
| `coefficient_0`                | DOUBLE | No       | Degree-0 polynomial coefficient. Any sign.                                                                                          |
| `coefficient_1`                | DOUBLE | No       | Degree-1 polynomial coefficient. Any sign.                                                                                          |
| `coefficient_2`                | DOUBLE | No       | Degree-2 polynomial coefficient. Any sign.                                                                                          |
| `coefficient_3`                | DOUBLE | No       | Degree-3 polynomial coefficient. Any sign.                                                                                          |
| `coefficient_4`                | DOUBLE | No       | Degree-4 polynomial coefficient. Any sign.                                                                                          |

The quartic is evaluated as `coefficient_0 + coefficient_1*x + coefficient_2*x² + coefficient_3*x³ + coefficient_4*x⁴` where `x` is the downstream outflow in m³/s. Higher-degree coefficients are routinely negative in source data; all signs are accepted.

**Validation rules:**

- All eleven columns must be present with the correct Arrow types.
- `outflow_min_m3s` and `outflow_max_m3s` must be non-negative and finite.
- `outflow_max_m3s >= outflow_min_m3s` (segments are non-inverted).
- `coefficient_0` through `coefficient_4` must be finite.
- `downstream_reference_level_m`, when non-null, must be non-negative and finite.

---

### `system/scalar_parameters.json`

Named scalar parameters that can be referenced from generic-constraint coefficient
expressions using the `@name` sigil. The file is optional; when absent, no
parameters are loaded and any `@name` token in a constraint expression causes a
load error.

**Top-level structure:**

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/scalar_parameters.schema.json",
  "scalar_parameters": [
    {
      "id": 1,
      "name": "rho_eq_h1",
      "kind": "computed",
      "computed_spec": { "tag": "equivalent_productivity", "hydro_id": 1 }
    }
  ]
}
```

**Per-entry fields:**

| Field           | Type    | Required | Description                                                               |
| --------------- | ------- | -------- | ------------------------------------------------------------------------- |
| `id`            | integer | Yes      | Unique parameter identifier (int32)                                       |
| `name`          | string  | Yes      | Unique parameter name (non-empty, no leading/trailing whitespace)         |
| `kind`          | string  | Yes      | One of `constant`, `per_stage`, `seasonal`, `computed`                    |
| `value`         | number  | kind dep | Finite f64 value. Required for `constant`. Absent otherwise.              |
| `values`        | array   | kind dep | Array of `[index, value]` pairs. Required for `per_stage` and `seasonal`. |
| `computed_spec` | object  | kind dep | `{"tag": "<variant>", "hydro_id": <int>}`. Required for `computed`.       |

**`computed_spec` tag values:**

| `tag`                      | Description                             |
| -------------------------- | --------------------------------------- |
| `equivalent_productivity`  | Equivalent productivity ρ_eq            |
| `accumulated_productivity` | Accumulated cascade productivity ρ_acum |
| `reference_volume`         | Reference reservoir volume V_ref        |
| `reference_turbine`        | Reference turbined flow Q_ref           |
| `min_storage`              | Minimum operational storage V_min       |
| `max_storage`              | Maximum operational storage V_max       |
| `specific_productivity`    | Specific productivity ρ_esp             |

**Validation:**

- `id` values must be unique across all entries.
- `name` values must be unique (case-sensitive), non-empty, and have no leading
  or trailing whitespace.
- `kind` must be exactly one of the four legal values.
- For `per_stage`: `values` pairs must have contiguous `stage_id` keys starting
  at 0; duplicates and gaps are rejected.
- For `seasonal`: `season_id` keys within an entry must be unique; duplicates
  are rejected.
- For `computed`: `computed_spec` must be present with a valid `tag` and integer
  `hydro_id`. The referenced hydro must exist in `hydros.json`.
- Unknown JSON fields on any entry are rejected immediately.

See [Scalar Parameters](../guide/scalar-parameters.md) for usage examples.

---

## `scenarios/` files (Parquet)

### `scenarios/inflow_seasonal_stats.parquet`

PAR(p) model seasonal statistics for each (hydro plant, stage) pair.

| Column     | Type   | Required | Description                                                 |
| ---------- | ------ | -------- | ----------------------------------------------------------- |
| `hydro_id` | INT32  | Yes      | Hydro plant ID                                              |
| `stage_id` | INT32  | Yes      | Stage ID                                                    |
| `mean_m3s` | DOUBLE | Yes      | Seasonal mean inflow (m³/s); must be finite                 |
| `std_m3s`  | DOUBLE | Yes      | Seasonal standard deviation (m³/s); must be >= 0 and finite |

---

### `scenarios/inflow_ar_coefficients.parquet`

Autoregressive coefficients for the PAR(p) inflow model.

| Column        | Type   | Required | Description                                 |
| ------------- | ------ | -------- | ------------------------------------------- |
| `hydro_id`    | INT32  | Yes      | Hydro plant ID                              |
| `stage_id`    | INT32  | Yes      | Stage ID                                    |
| `lag`         | INT32  | Yes      | Lag index (1-based)                         |
| `coefficient` | DOUBLE | Yes      | AR coefficient for this (hydro, stage, lag) |

---

### `scenarios/noise_openings.parquet`

User-supplied backward-pass opening tree. When present, Cobre loads the opening
tree directly from this file instead of generating it internally via
`generate_opening_tree()`. This enables cross-tool comparison, sensitivity
analysis, and round-trip replay of a previously exported opening tree.

| Column          | Type   | Required | Description                                                                   |
| --------------- | ------ | -------- | ----------------------------------------------------------------------------- |
| `stage_id`      | INT32  | Yes      | Zero-based stage index (0 to n_stages − 1)                                    |
| `opening_index` | UINT32 | Yes      | Zero-based opening index within the stage (0 to openings_per_stage − 1)       |
| `entity_index`  | UINT32 | Yes      | Zero-based entity index in system dimension order (see entity ordering below) |
| `value`         | DOUBLE | Yes      | Noise realization for this (stage, opening, entity) triple                    |

**Entity ordering.** The `entity_index` column follows the system dimension
convention: hydro entities first (sorted by canonical ID), then load buses
(sorted by canonical ID), matching the ordering used by the internal opening
tree generator. Violating this convention causes silent value misassignment
because the file stores indices only, not entity identifiers.

**Validation rules.** The loader checks three conditions and raises a hard error
on failure:

- **Dimension mismatch** — the number of distinct `entity_index` values must
  equal `n_hydros + n_load_buses`.
- **Stage count mismatch** — the number of distinct `stage_id` values must
  equal the configured number of study stages.
- **Missing opening indices** — for each stage, every opening index from 0 to
  `openings_per_stage − 1` must be present for every entity. Gaps are not
  permitted; partial-stage override is not supported.

The total row count must equal `n_stages × openings_per_stage × (n_hydros +
n_load_buses)`.

See the `noise_openings.rs` module for the full schema and validation
rules, and [User-Supplied Opening Trees](../guide/stochastic-modeling.md#user-supplied-opening-trees)
in the Stochastic Modeling guide for usage instructions.

---

## `scenarios/` files (JSON)

### `scenarios/load_factors.json`

Per-bus, per-stage, per-block load scaling factors. When present, each factor
multiplies the stochastic load demand realization at the specified bus for the
specified block. This allows you to model time-of-day or seasonal patterns in
load shape without changing the underlying statistical model.

When this file is absent, all load factors default to 1.0. When a
`(bus_id, stage_id)` pair is absent from the file, its factors also default
to 1.0 for every block.

**JSON structure:**

```json
{
  "load_factors": [
    {
      "bus_id": 0,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 0, "factor": 0.8 },
        { "block_id": 1, "factor": 1.2 }
      ]
    }
  ]
}
```

**Fields per entry:**

| Field           | Type    | Description                                                        |
| --------------- | ------- | ------------------------------------------------------------------ |
| `bus_id`        | integer | Bus entity ID. Must refer to a bus defined in `system/buses.json`. |
| `stage_id`      | integer | Study stage index. Must be a valid stage ID from `stages.json`.    |
| `block_factors` | array   | Array of `{ block_id, factor }` pairs for each load block.         |

**`block_factors` entry fields:**

| Field      | Type    | Constraints                     | Description                                                                       |
| ---------- | ------- | ------------------------------- | --------------------------------------------------------------------------------- |
| `block_id` | integer | Must be a valid block for stage | Zero-based block index within the stage.                                          |
| `factor`   | number  | > 0, finite                     | Multiplier applied to the stochastic load realization (MW) at this bus and block. |

**Effect:** `load_rhs = mean_mw * stochastic_noise_factor * block_factor`.
A factor of 1.0 leaves the load unchanged. Values less than 1.0 reduce load;
values greater than 1.0 increase it.

---

### `scenarios/non_controllable_factors.json`

Per-NCS, per-stage, per-block scaling factors for non-controllable source (NCS)
available generation. When present, each factor multiplies the available
generation bound from `constraints/ncs_bounds.parquet` for the specified block.
This allows modeling of intra-stage availability patterns such as diurnal solar
irradiance profiles or wind speed variations across load blocks.

When this file is absent, all NCS block factors default to 1.0. When a
`(ncs_id, stage_id)` pair is absent from the file, its factors default to 1.0
for every block.

**JSON structure:**

```json
{
  "non_controllable_factors": [
    {
      "ncs_id": 0,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 0, "factor": 0.3 },
        { "block_id": 1, "factor": 0.8 }
      ]
    }
  ]
}
```

**Fields per entry:**

| Field           | Type    | Description                                                                      |
| --------------- | ------- | -------------------------------------------------------------------------------- |
| `ncs_id`        | integer | NCS entity ID. Must refer to a source in `system/non_controllable_sources.json`. |
| `stage_id`      | integer | Study stage index. Must be a valid stage ID from `stages.json`.                  |
| `block_factors` | array   | Array of `{ block_id, factor }` pairs for each load block.                       |

**`block_factors` entry fields:**

| Field      | Type    | Constraints                     | Description                                                                |
| ---------- | ------- | ------------------------------- | -------------------------------------------------------------------------- |
| `block_id` | integer | Must be a valid block for stage | Zero-based block index within the stage.                                   |
| `factor`   | number  | >= 0, finite                    | Multiplier applied to the stage available generation bound for this block. |

**Effect:** `available_mw_block = available_generation_mw * block_factor`.
A factor of 1.0 leaves the bound unchanged. A factor of 0.0 sets availability
to zero for that block (complete generation unavailability).

---

### `scenarios/non_controllable_stats.parquet`

Per-NCS, per-stage stochastic availability model. Each row provides the mean
and standard deviation of the availability factor for one NCS entity at one
stage. The noise transform produces: `A_r = max_gen × clamp(mean + std × η, 0, 1)`.

| Column     | Type   | Required | Description                                      |
| ---------- | ------ | -------- | ------------------------------------------------ |
| `ncs_id`   | INT32  | Yes      | Non-controllable source ID                       |
| `stage_id` | INT32  | Yes      | Stage ID (0-based)                               |
| `mean`     | DOUBLE | Yes      | Mean availability factor in [0, 1]               |
| `std`      | DOUBLE | Yes      | Standard deviation of availability factor (>= 0) |

When absent, NCS availability is deterministic from `constraints/ncs_bounds.parquet`
or the entity's `max_generation_mw`.

---

## `constraints/` files (Parquet)

All bounds Parquet files use sparse storage: only `(entity_id, stage_id)` pairs
that differ from the base entity-level value need rows. Absent rows use the
entity-level value unchanged.

### `constraints/thermal_bounds.parquet`

Stage-varying generation bound overrides for thermal plants.

| Column              | Type   | Required | Description                      |
| ------------------- | ------ | -------- | -------------------------------- |
| `thermal_id`        | INT32  | Yes      | Thermal plant ID                 |
| `stage_id`          | INT32  | Yes      | Stage ID                         |
| `min_generation_mw` | DOUBLE | No       | Minimum generation override (MW) |
| `max_generation_mw` | DOUBLE | No       | Maximum generation override (MW) |

---

### `constraints/hydro_bounds.parquet`

Stage-varying operational bound overrides for hydro plants.

| Column                 | Type   | Required | Description                          |
| ---------------------- | ------ | -------- | ------------------------------------ |
| `hydro_id`             | INT32  | Yes      | Hydro plant ID                       |
| `stage_id`             | INT32  | Yes      | Stage ID                             |
| `min_turbined_m3s`     | DOUBLE | No       | Minimum turbined flow (m³/s)         |
| `max_turbined_m3s`     | DOUBLE | No       | Maximum turbined flow (m³/s)         |
| `min_storage_hm3`      | DOUBLE | No       | Minimum reservoir storage (hm³)      |
| `max_storage_hm3`      | DOUBLE | No       | Maximum reservoir storage (hm³)      |
| `min_outflow_m3s`      | DOUBLE | No       | Minimum total outflow (m³/s)         |
| `max_outflow_m3s`      | DOUBLE | No       | Maximum total outflow (m³/s)         |
| `min_generation_mw`    | DOUBLE | No       | Minimum generation (MW)              |
| `max_generation_mw`    | DOUBLE | No       | Maximum generation (MW)              |
| `max_diversion_m3s`    | DOUBLE | No       | Maximum diversion flow (m³/s)        |
| `filling_min_rate_m3s` | DOUBLE | No       | Filling minimum-rate override (m³/s) |
| `water_withdrawal_m3s` | DOUBLE | No       | Water withdrawal (m³/s)              |

---

### `constraints/line_bounds.parquet`

Stage-varying flow capacity overrides for transmission lines.

| Column       | Type   | Required | Description                         |
| ------------ | ------ | -------- | ----------------------------------- |
| `line_id`    | INT32  | Yes      | Transmission line ID                |
| `stage_id`   | INT32  | Yes      | Stage ID                            |
| `direct_mw`  | DOUBLE | No       | Direct-flow capacity override (MW)  |
| `reverse_mw` | DOUBLE | No       | Reverse-flow capacity override (MW) |

---

### `constraints/pumping_bounds.parquet`

Stage-varying flow bounds for pumping stations.

| Column       | Type   | Required | Description                 |
| ------------ | ------ | -------- | --------------------------- |
| `station_id` | INT32  | Yes      | Pumping station ID          |
| `stage_id`   | INT32  | Yes      | Stage ID                    |
| `min_m3s`    | DOUBLE | No       | Minimum pumping flow (m³/s) |
| `max_m3s`    | DOUBLE | No       | Maximum pumping flow (m³/s) |

---

### `constraints/contract_bounds.parquet`

Stage-varying power and price overrides for energy contracts.

| Column          | Type   | Required | Description              |
| --------------- | ------ | -------- | ------------------------ |
| `contract_id`   | INT32  | Yes      | Energy contract ID       |
| `stage_id`      | INT32  | Yes      | Stage ID                 |
| `min_mw`        | DOUBLE | No       | Minimum power (MW)       |
| `max_mw`        | DOUBLE | No       | Maximum power (MW)       |
| `price_per_mwh` | DOUBLE | No       | Price override (USD/MWh) |

---

### `constraints/ncs_bounds.parquet`

Stage-varying available generation bounds for non-controllable sources. Uses
sparse storage: only `(ncs_id, stage_id)` pairs that differ from the base
entity-level value need rows. Absent rows keep the entity's declared
`available_generation_mw` unchanged.

| Column                    | Type   | Required | Description                                                     |
| ------------------------- | ------ | -------- | --------------------------------------------------------------- |
| `ncs_id`                  | INT32  | Yes      | Non-controllable source ID                                      |
| `stage_id`                | INT32  | Yes      | Stage ID                                                        |
| `available_generation_mw` | DOUBLE | Yes      | Maximum available generation for this stage (MW). Must be >= 0. |

The per-block available generation bound in the LP is:
`available_mw_block = available_generation_mw * block_factor`, where
`block_factor` comes from `scenarios/non_controllable_factors.json`
(default 1.0 when absent).

---

### `constraints/exchange_factors.json`

Per-line, per-stage, per-block scaling factors for transmission line capacity
bounds. When present, each factor multiplies the line's direct or reverse
capacity for the specified block. This allows modeling of planned outages,
seasonal de-rating, or time-of-day capacity constraints without replacing the
base entity bounds.

When this file is absent, all exchange factors default to (1.0, 1.0). When a
`(line_id, stage_id)` pair is absent, its factors default to (1.0, 1.0) for
every block.

**JSON structure:**

```json
{
  "exchange_factors": [
    {
      "line_id": 0,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 0, "direct_factor": 0.9, "reverse_factor": 1.0 }
      ]
    }
  ]
}
```

**Fields per entry:**

| Field           | Type    | Description                                                          |
| --------------- | ------- | -------------------------------------------------------------------- |
| `line_id`       | integer | Line entity ID. Must refer to a line defined in `system/lines.json`. |
| `stage_id`      | integer | Study stage index. Must be a valid stage ID from `stages.json`.      |
| `block_factors` | array   | Array of `{ block_id, direct_factor, reverse_factor }` pairs.        |

**`block_factors` entry fields:**

| Field            | Type    | Constraints                     | Description                                                        |
| ---------------- | ------- | ------------------------------- | ------------------------------------------------------------------ |
| `block_id`       | integer | Must be a valid block for stage | Zero-based block index within the stage.                           |
| `direct_factor`  | number  | >= 0, finite                    | Multiplier for the direct-direction flow capacity (`direct_mw`).   |
| `reverse_factor` | number  | >= 0, finite                    | Multiplier for the reverse-direction flow capacity (`reverse_mw`). |

**Effect:** `col_upper_fwd = direct_mw * direct_factor`,
`col_upper_rev = reverse_mw * reverse_factor`. A factor of 1.0 leaves the
capacity unchanged. A factor of 0.0 fully blocks flow in that direction for
the block.

---

## Penalty override files

All penalty override files use sparse storage. Only rows for `(entity_id, stage_id)`
pairs where the penalty differs from the entity-level or global default are required.
All penalty values must be strictly positive (> 0.0) and finite.

### `constraints/penalty_overrides_bus.parquet`

| Column        | Type   | Required | Description                              |
| ------------- | ------ | -------- | ---------------------------------------- |
| `bus_id`      | INT32  | Yes      | Bus ID                                   |
| `stage_id`    | INT32  | Yes      | Stage ID                                 |
| `excess_cost` | DOUBLE | No       | Excess injection cost override (USD/MWh) |

Note: Bus deficit segments are not stage-varying. Only `excess_cost` can be
overridden per stage for buses.

---

### `constraints/penalty_overrides_line.parquet`

| Column          | Type   | Required | Description                           |
| --------------- | ------ | -------- | ------------------------------------- |
| `line_id`       | INT32  | Yes      | Transmission line ID                  |
| `stage_id`      | INT32  | Yes      | Stage ID                              |
| `exchange_cost` | DOUBLE | No       | Exchange flow cost override (USD/MWh) |

---

### `constraints/penalty_overrides_hydro.parquet`

| Column                            | Type   | Required | Description                                 |
| --------------------------------- | ------ | -------- | ------------------------------------------- |
| `hydro_id`                        | INT32  | Yes      | Hydro plant ID                              |
| `stage_id`                        | INT32  | Yes      | Stage ID                                    |
| `spillage_cost`                   | DOUBLE | No       | Spillage penalty override                   |
| `turbined_cost`                   | DOUBLE | No       | Turbined cost override                      |
| `diversion_cost`                  | DOUBLE | No       | Diversion penalty override                  |
| `storage_violation_below_cost`    | DOUBLE | No       | Storage below-minimum violation override    |
| `filling_target_violation_cost`   | DOUBLE | No       | Filling target violation override           |
| `turbined_violation_below_cost`   | DOUBLE | No       | Turbined below-minimum violation override   |
| `outflow_violation_below_cost`    | DOUBLE | No       | Outflow below-minimum violation override    |
| `outflow_violation_above_cost`    | DOUBLE | No       | Outflow above-maximum violation override    |
| `generation_violation_below_cost` | DOUBLE | No       | Generation below-minimum violation override |
| `evaporation_violation_cost`      | DOUBLE | No       | Evaporation violation override              |
| `water_withdrawal_violation_cost` | DOUBLE | No       | Water withdrawal violation override         |

---

### `constraints/penalty_overrides_ncs.parquet`

| Column             | Type   | Required | Description                            |
| ------------------ | ------ | -------- | -------------------------------------- |
| `source_id`        | INT32  | Yes      | Non-controllable source ID             |
| `stage_id`         | INT32  | Yes      | Stage ID                               |
| `curtailment_cost` | DOUBLE | No       | Curtailment penalty override (USD/MWh) |
