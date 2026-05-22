# Thermal Units

Thermal power plants are the dispatchable generation assets that complement hydro
in Cobre's system model. The term "thermal" covers any generator whose output is
bounded by installed capacity and whose dispatch incurs an explicit cost per MWh:
combustion turbines, combined-cycle plants, coal-fired units, nuclear plants, and
diesel generators all map onto the same Cobre `Thermal` entity type.

Unlike hydro plants, thermal units carry no state between stages. Each stage's
LP sub-problem treats a thermal unit as a simple bounded generation variable with
a marginal cost. The solver dispatches thermal units in merit order — from cheapest
to most expensive — to meet any residual demand not covered by hydro generation.
In a hydrothermal system, the long-run value of stored water is compared against
the short-run cost of thermal dispatch at each stage, which is the fundamental
trade-off the SDDP algorithm optimizes.

The cost structure of a thermal unit is modeled with a **scalar marginal cost**
(`cost_per_mwh`). The LP dispatches the unit at any level between `min_mw` and
`max_mw`, with the generation cost equal to `dispatched_mw * hours_in_block * cost_per_mwh`.

For an introductory walkthrough of writing `thermals.json`, see
[Building a System](../tutorial/building-a-system.md) and
[Anatomy of a Case](../tutorial/anatomy-of-a-case.md). This page provides the
complete field reference, including anticipated dispatch configuration.

---

## JSON Schema

Thermal units are defined in `system/thermals.json`. The top-level object has a
single key `"thermals"` containing an array of unit objects. The following example
shows all fields, including the optional `entry_stage_id`, `exit_stage_id`, and
`anticipated_config`:

```json
{
  "thermals": [
    {
      "id": 0,
      "name": "UTE1",
      "bus_id": 0,
      "cost_per_mwh": 5.0,
      "generation": {
        "min_mw": 0.0,
        "max_mw": 15.0
      }
    },
    {
      "id": 1,
      "name": "Angra 1",
      "bus_id": 0,
      "entry_stage_id": null,
      "exit_stage_id": null,
      "cost_per_mwh": 50.0,
      "generation": {
        "min_mw": 0.0,
        "max_mw": 657.0
      },
      "anticipated_config": {
        "lead_stages": 2
      }
    }
  ]
}
```

The first plant (`UTE1`) matches the `1dtoy` template format: a cost per MWh with
no optional fields. The second plant (`Angra 1`) shows the complete schema with
anticipated dispatch. The fields `entry_stage_id`, `exit_stage_id`, and
`anticipated_config` are optional and can be omitted.

---

## Core Fields

These fields appear at the top level of each thermal unit object.

| Field            | Type            | Required | Description                                                                                                       |
| ---------------- | --------------- | -------- | ----------------------------------------------------------------------------------------------------------------- |
| `id`             | integer         | Yes      | Unique non-negative integer identifier. Must be unique across all thermal units.                                  |
| `name`           | string          | Yes      | Human-readable plant name. Used in output files, validation messages, and log output.                             |
| `bus_id`         | integer         | Yes      | Identifier of the electrical bus to which this unit's generation is injected. Must match an `id` in `buses.json`. |
| `cost_per_mwh`   | number          | Yes      | Marginal cost of generation [$/MWh]. Must be ≥ 0.0.                                                               |
| `entry_stage_id` | integer or null | No       | Stage index at which the unit enters service (inclusive). `null` means the unit is available from stage 0.        |
| `exit_stage_id`  | integer or null | No       | Stage index at which the unit is decommissioned (inclusive). `null` means the unit is never decommissioned.       |

---

## Generation Bounds

The `generation` block sets the output limits for the unit (stored internally as
`min_generation_mw` and `max_generation_mw` on the `Thermal` struct). These are
enforced as hard bounds on the generation variable in each stage LP.

```json
"generation": {
  "min_mw": 0.0,
  "max_mw": 657.0
}
```

| Field    | Type   | Description                                                                                                                                                                                                    |
| -------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `min_mw` | number | Minimum electrical generation (minimum stable load) [MW]. A non-zero value represents a must-run commitment: the solver is required to dispatch at least this much generation whenever the unit is in service. |
| `max_mw` | number | Maximum electrical generation (installed capacity) [MW].                                                                                                                                                       |

A `min_mw` of `0.0` means the unit can be turned off completely — it is treated as
an interruptible resource. A non-zero `min_mw` (for example, `100.0` for a plant
whose turbine must spin continuously for mechanical reasons) means the LP must
always dispatch at least that amount whenever the plant is active.

---

## Anticipated Dispatch Configuration

The optional `anticipated_config` block enables anticipated dispatch for thermal
units that require advance scheduling over multiple stages due to commitment lead
times — for example, a plant that must be booked several weeks before the dispatch
occurs.

```json
"anticipated_config": {
  "lead_stages": 2
}
```

| Field         | Type    | Description                                                                                                                               |
| ------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `lead_stages` | integer | Number of stages of dispatch anticipation. A value of `2` means the generation commitment for stage `t` must be decided at stage `t - 2`. |

### How anticipated dispatch works

When a thermal unit has `lead_stages = K`, its dispatch commitment is split across
two roles that appear at different stages:

- **Decision stage** (`t`): the LP at stage `t` sets the generation level that will
  be delivered `K` stages later. This decision variable is carried forward as state.
- **Delivery stage** (`t + K`): the LP at stage `t + K` receives the committed MW
  value as a fixed bound, reflecting that the generation level was locked in earlier.

Consider a 3-stage finite-horizon study with one anticipated thermal unit configured
as `"lead_stages": 2`:

| Stage | Role for this unit                                          | `anticipated_decision_mw`                            | `anticipated_committed_mw`                                 |
| ----- | ----------------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------- |
| 0     | Decision                                                    | non-null (commitment placed for delivery at stage 2) | `null` (no matured delivery yet)                           |
| 1     | Decision (horizon boundary: stage 1 + 2 = 3 = total stages) | non-null                                             | `null` (delivery requires K ≤ stage index; 2 ≤ 1 is false) |
| 2     | Delivery                                                    | `null` (stage 2 + 2 = 4 exceeds the horizon)         | non-null (matured commitment from stage 0)                 |

The `null` values in this table are not errors — they reflect the position of a
stage within the horizon. At the first stages the commitment is being placed but
has not yet matured; at the last stage the commitment has matured but there are no
more future stages to place new decisions into.

For a `lead_stages = 1` configuration on a 2-stage study, the coupling is simpler:
the decision placed at stage 0 matures at stage 1. Stage 0 shows a non-null
`anticipated_decision_mw` and null `anticipated_committed_mw`; stage 1 shows the
reverse.

### Pairing with initial_conditions.json

Because anticipated dispatch carries state across stages, every anticipated thermal
unit must have a corresponding entry in `past_anticipated_commitments` in
`initial_conditions.json`:

```json
{
  "storage": [],
  "filling_storage": [],
  "past_anticipated_commitments": [
    {
      "thermal_id": 2,
      "values_mw": [0.0, 0.0]
    }
  ]
}
```

The `values_mw` array must have exactly `lead_stages` entries. The values are
ordered chronologically from oldest to most recent: `values_mw[0]` corresponds to
the oldest pending slot and `values_mw[lead_stages - 1]` to the most recent.
For the example above with `lead_stages = 2`, the array has length 2. Supplying an
array of a different length is a validation error.

**Current limitation: every entry in `values_mw` must be `0.0`.** Pre-horizon
commitments (generation dispatched outside the study horizon that delivers during
the study) cannot be expressed in the current version. The semantic validator rejects
any non-zero `values_mw` entry with an explicit error message naming the thermal id
and the offending slot index. Set all entries to `0.0` when constructing
`initial_conditions.json` for studies with anticipated thermal units.

Support for non-zero pre-horizon commitments is planned for a future release.

The `past_anticipated_commitments` key is optional in the JSON file and defaults to
an empty list for studies that have no anticipated thermal units.

### Reading the outputs

After a simulation run, three additional columns appear in
`simulation/thermals/scenario_id=NNNN/data.parquet` for every thermal unit. See
[Output Format Reference](../reference/output-format.md) for the full column schema.
The anticipated-dispatch columns are:

| Column                     | Type    | Nullable | Meaning                                                                                                                                                                                   |
| -------------------------- | ------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `is_anticipated`           | Boolean | No       | `true` for units configured with `anticipated_config`; `false` for all others.                                                                                                            |
| `anticipated_committed_mw` | Float64 | Yes      | The committed MW value that matures and is delivered at this stage. `null` at early stages before any commitment has matured, and always `null` for non-anticipated units.                |
| `anticipated_decision_mw`  | Float64 | Yes      | The commitment placed at this stage for delivery `K` stages later. `null` when no forward decision is available (e.g., at the final stages of the horizon, or for non-anticipated units). |

Regular (non-anticipated) thermal units always have `is_anticipated = false` and
both optional columns set to `null`. Rows for anticipated units have
`is_anticipated = true`; the two nullable columns are populated according to each
stage's position relative to the decision and delivery windows described above.

Training output also records anticipated-dispatch state in
`training/dictionaries/state_dictionary.json`. For each anticipated thermal unit,
the dictionary contains one entry per slot index from `0` to `lead_stages - 1`,
in slot-major order. Each entry has the following shape:

```json
{
  "type": "anticipated_state",
  "entity_type": "thermal",
  "entity_id": 2,
  "slot_index": 0,
  "unit": "MW"
}
```

For a study with a single anticipated thermal unit (`id = 2`) configured as
`lead_stages = 2`, the state dictionary will contain exactly two such entries:
one with `slot_index = 0` and one with `slot_index = 1`. The slot index identifies
which pending commitment the state variable tracks: slot 0 holds the oldest
still-pending commitment and slot `lead_stages - 1` holds the most recent.

---

## Validation Rules

Cobre's five-layer validation pipeline checks the following conditions on thermal
units. Violations are reported as error messages with the failing unit's `id`.

| Rule                       | Error Class          | Description                                                                         |
| -------------------------- | -------------------- | ----------------------------------------------------------------------------------- |
| Bus reference integrity    | Reference error      | Every `bus_id` must match an `id` in `buses.json`.                                  |
| Non-negative cost          | Schema error         | `cost_per_mwh` must be ≥ 0.0.                                                       |
| Generation bounds ordering | Physical feasibility | `min_mw` must be less than or equal to `max_mw`.                                    |
| Anticipated lead validity  | Physical feasibility | When `anticipated_config` is present, `lead_stages` must be a non-negative integer. |

---

## Related Pages

- [Anatomy of a Case](../tutorial/anatomy-of-a-case.md) — walks through the complete `1dtoy` thermal definitions
- [Building a System](../tutorial/building-a-system.md) — step-by-step guide to writing `thermals.json` from scratch
- [System Modeling](./system-modeling.md) — overview of all entity types and how they interact
- [Case Format Reference](../reference/case-format.md) — complete JSON schema for all input files
