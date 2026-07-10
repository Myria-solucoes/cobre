# Thermal Units

Thermal power plants are the dispatchable generation assets that complement hydro
in Cobre's system model. The term "thermal" covers any generator whose output is
bounded by installed capacity and whose dispatch incurs an explicit cost per MWh:
combustion turbines, combined-cycle plants, coal-fired units, nuclear plants, and
diesel generators all map onto the same Cobre `Thermal` entity type.

Unlike hydro plants, thermal units carry no state between stages. Each stage's
LP sub-problem treats a thermal unit as a bounded generation variable with
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
      "operational_start_date": "2024-01-01",
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
      "operational_start_date": "2024-01-01",
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
occurs. Two lead modes are available (below): a stage count, or a physical lead
time in hours.

```json
"anticipated_config": {
  "lead_stages": 2
}
```

| Field         | Type    | Description                                                                                                                               |
| ------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `lead_stages` | integer | Number of stages of dispatch anticipation. A value of `2` means the generation commitment for stage `t` must be decided at stage `t - 2`. |

### Two lead modes: `lead_stages` and `lead_time_hours`

`anticipated_config` carries exactly one of two lead modes — never both, never
neither:

- **`lead_stages`** (above) — a stage count; the calendar is never consulted.
- **`lead_time_hours`** — a physical lead time in hours; delivery-anchored and
  resolved against the study calendar.

```json
"anticipated_config": {
  "lead_time_hours": 720.0
}
```

| Field             | Type   | Description                                               |
| ----------------- | ------ | --------------------------------------------------------- |
| `lead_time_hours` | number | Physical lead time, in hours. Must be finite and `> 0.0`. |

Setting both fields, or setting neither, is rejected while `system/thermals.json`
is being loaded: `anticipated_config` is parsed as one of these two shapes, so a
JSON object naming both fields, or naming neither, matches neither shape and the
case fails to load.

Rather than fixing a stage count, `lead_time_hours` fixes a duration on the hour
clock. For every delivery stage `m`, Cobre finds the study stage that was current
`lead_time_hours` hours before `m` ends (ties at a stage boundary resolve to the
earlier stage) and makes that stage `m`'s decision stage. On a calendar where
every stage has the same length, this produces exactly the same decision/delivery
pairing as the equivalent `lead_stages` value; the two modes only diverge once the
calendar's stage lengths change partway through the plant's active window — which
is exactly where a physical commitment needs to stay anchored to hours rather than
to a stage count (worked example below).

### Which lead mode to choose

Use `lead_stages` when the plant's active window sits entirely within a single
stage-cadence regime — the study calendar does not change resolution across it —
or when the commitment genuinely tracks a fixed number of dispatch cycles rather
than a real-world duration. Use `lead_time_hours` when the commitment is anchored
to a duration that should stay constant in hours regardless of how finely the
study calendar resolves that period — a fuel-nomination deadline, an
outage-booking window, a regulatory advance-notice period — especially when the
plant's active window crosses a stage-cadence change.

### How anticipated dispatch works

The decision/delivery split below is described for `lead_stages`, where every
decision stage delivers to exactly one delivery stage `K` stages later.
`lead_time_hours` reuses the same decision/delivery roles and the same
`anticipated_decision_mw`/`anticipated_committed_mw` output columns; only the
rule that maps a delivery stage to its decision stage differs (the study
calendar, rather than a constant `K`).

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

### Worked example: a fixed stage count and a physical lead diverge

Suppose a thermal plant's fuel-nomination process requires the operator to commit
to a delivery `720` hours (30 days) before it is dispatched, and the study's
calendar begins with four weekly stages (168 hours each, stages 0–3) before
switching to monthly stages (720 hours each, stage 4 onward).

With `lead_stages`, the only lever is a constant stage count, so the same setting
applies on both sides of the cadence change. The table below reports, for each
delivery stage, the decision stage each configuration resolves to and the
resulting physical lead — the number of hours between the decision stage's end
and the delivery stage's end:

| Delivery stage (`m`) | Duration (h) | `lead_stages = 1` | `lead_stages = 4` | `lead_time_hours = 720.0` |
| -------------------- | ------------ | ----------------- | ----------------- | ------------------------- |
| 0 (week 1)           | 168          | pre-study         | pre-study         | pre-study                 |
| 1 (week 2)           | 168          | stage 0 → 168 h   | pre-study         | pre-study                 |
| 2 (week 3)           | 168          | stage 1 → 168 h   | pre-study         | pre-study                 |
| 3 (week 4)           | 168          | stage 2 → 168 h   | pre-study         | pre-study                 |
| 4 (month 1)          | 720          | stage 3 → 720 h   | stage 0 → 1224 h  | stage 3 → 720 h           |
| 5 (month 2)          | 720          | stage 4 → 720 h   | stage 1 → 1776 h  | stage 4 → 720 h           |

`lead_stages = 1` under-notices every weekly delivery — 168 hours instead of the
720 the process needs — and only reaches 720 hours at the monthly deliveries,
where it happens to coincide with the delivery stage's own length.
`lead_stages = 4` needs the same four pre-study seed entries the correct physical
lead needs (below), but then over-notices the monthly deliveries, and by a margin
that grows every stage (1224, then 1776 hours, against a 720-hour target). No
single `lead_stages` value is correct on both sides of the transition; both
values above also trigger the cadence-transition advisory described below, at
the same stage-3/stage-4 boundary.

`lead_time_hours = 720.0` resolves to exactly 720 hours at every delivery stage
that has an in-study decision, on both sides of the transition, because it
derives the decision stage from the actual calendar rather than from a fixed
stage count.

The first four delivery stages resolve to a decision before the study begins —
720 hours of lead reaches back past the study's first stage while the calendar
is still weekly — so those four deliveries are seeded from
`past_anticipated_commitments` rather than decided by an in-study LP stage:

```json
{
  "thermals": [
    {
      "id": 5,
      "name": "UTE2",
      "operational_start_date": "2024-01-01",
      "bus_id": 0,
      "cost_per_mwh": 90.0,
      "generation": {
        "min_mw": 0.0,
        "max_mw": 200.0
      },
      "anticipated_config": {
        "lead_time_hours": 720.0
      }
    }
  ]
}
```

```json
{
  "past_anticipated_commitments": [
    {
      "thermal_id": 5,
      "values_mw": [0.0, 0.0, 150.0, 180.0]
    }
  ]
}
```

The required seed length here — four entries — is calendar-derived: it is the
count of study stages whose stage-end cumulative hours are still `<= 720`, not
`lead_stages` (this plant has none). See
[Pairing with initial_conditions.json](#pairing-with-initial_conditionsjson)
below for the full rule, including why `150.0` and `180.0` are accepted values.

### Cadence-transition advisory

A `lead_stages` plant whose decision-to-delivery window spans a stage-cadence
change — the situation the worked example above shows for both `lead_stages = 1`
and `lead_stages = 4` — triggers an advisory, not a hard error. Validation scans
every pair of consecutive stage durations the plant's active window covers, and
on the first pair with differing durations, emits a warning naming the thermal,
the two stage indices, and `lead_time_hours` as the alternative that stays
correct across the transition. The study still validates successfully; a cadence
transition never blocks a study on its own. A `lead_time_hours` plant never
triggers this advisory, since its decider already accounts for the actual
calendar on both sides of any transition.

### Sub-stage leads: the K=0 exclusion

A `lead_time_hours` plant can resolve to a lead shorter than the delivery
stage's own duration — for example, `lead_time_hours: 100.0` against a 168-hour
weekly stage. When that happens, the commitment would be decided inside its own
delivery stage, so no anticipation actually binds. Cobre excludes that stage
from the anticipated-commitment state entirely and dispatches the plant's
generation there as ordinary, unconstrained thermal output — exactly as if
`anticipated_config` were absent for that one stage. This is never a hard error.
At setup time Cobre logs one diagnostic per affected stage, naming the plant,
the stage, and that the effective lead at that stage is equivalent to
`lead_stages = 0`. A `lead_stages` plant never triggers this: a positive stage
count is never zero stages by construction.

### The fan-out limitation

On a calendar that coarsens as the horizon proceeds, a single `lead_time_hours`
decision stage can anchor more than one delivery stage — a coarse stage deciding
several finer stages that follow it. This is structurally impossible under
`lead_stages`, where a constant stage-count lead always maps one decision stage
to exactly one delivery stage. Cobre's setup rejects any `lead_time_hours`
configuration that fans out this way: the study fails to load with a validation
error naming the plant, rather than silently dropping the later deliveries'
committed-MW simulation output. Per-delivery-stage simulation output for a
fanned-out configuration is not yet implemented; until it is, keep a
`lead_time_hours` plant's calendar from coarsening across its active window, or
use `lead_stages` where a constant stage-count lead is an acceptable
approximation.

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

`values_mw`'s required length is calendar-derived, not a fixed count: it equals
the number of study stages whose commitment is decided before the study begins —
the pre-study-committed delivery stages the worked example above walks through.
For a `lead_stages` plant on a study at least `lead_stages` stages long, that
count equals `lead_stages` exactly (as in the example above, length 2); for a
`lead_time_hours` plant it depends on the calendar, as the worked example's
four-entry array shows. The values are ordered chronologically from oldest to
most recent: `values_mw[0]` corresponds to the oldest pre-study-committed
delivery stage. Supplying an array of a different length than the
calendar-derived count is a validation error naming both the expected and
actual counts.

Each `values_mw[j]` is the MW the plant delivers at that pre-study-committed
delivery stage; its cost is sunk and does not enter the study's objective. A
value is accepted as long as it lies within the plant's own
`[min_generation_mw, max_generation_mw]` bounds — an out-of-bounds value is
rejected, since the LP would otherwise need to dispatch outside the plant's
physical capacity to honor it. A plant that also sets
`entry_stage_id`/`exit_stage_id` additionally requires every nonzero
`values_mw[j]` to mature inside that commissioning window; a nonzero value
maturing outside it is rejected, since the plant's generation column is pinned
to `[0, 0]` there.

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

The anticipated-commitment state is part of the policy's state vector, so each
anticipated thermal unit contributes ring-buffer slots to the per-slot entity
manifest embedded in every `policy/cuts/stage_NNN.bin` and
`policy/states/stage_NNN.bin` (see
[Policy Checkpoint](../reference/output-format.md#policy-checkpoint)). For an
anticipated unit each slot's manifest entry has `entity_type` set to the
anticipated-thermal-state class, `entity_id` set to the unit's id, and `subindex`
set to the ring-buffer slot — slot 0 tracks the oldest still-pending commitment
and the highest slot the most recent.

The ring buffer is sized to the study-wide `K_max` — the deepest per-stage
in-flight depth across every anticipated thermal. For a `lead_stages` plant this
equals its constant `lead_stages` value; for a `lead_time_hours` plant it is the
deepest depth the calendar resolution produces at any decision stage (the worked
example above reaches a depth of 1). For a unit whose own reach is shallower
than `K_max` (mixed studies), the surplus high slots are structural padding
aligning the buffer to a uniform stride; they are deterministically zero. Each
slot's `was_active` flag records whether the owning unit was operationally
active at that stage, encoding the active/padding distinction directly.

For a study with a single anticipated thermal unit (`id = 2`) configured as
`lead_stages = 2`, the manifest carries exactly two such slots — `subindex = 0`
and `subindex = 1` — both active, since `K_i = K_max = 2`.

---

## Constraining commitments via generic constraints

The anticipated-commitment decision variable can be referenced directly in a
generic constraint using the `anticipated_decision(N)` expression syntax, where
`N` is the thermal unit's `id`. This lets you cap, floor, or couple the MW level
committed at each decision stage across multiple anticipated thermals.

```json
{
  "constraints": [
    {
      "id": 1,
      "name": "cap_ant_t1",
      "expression": "anticipated_decision(2)",
      "sense": "<=",
      "slack": { "enabled": false }
    }
  ]
}
```

With a matching bound row in `constraints/generic_constraint_bounds.parquet`
that sets `bound = 20.0` at stage 0, the constraint limits the commitment placed
at stage 0 for delivery 2 stages later to at most 20 MW.

Two semantic rules apply:

- `anticipated_decision(N)` must reference a thermal that carries an
  `anticipated_config` block. Referencing a non-anticipated thermal is a hard
  error (`BusinessRuleViolation`).
- `thermal_generation(N)` referencing an anticipated thermal emits a
  `SemanticAmbiguity` warning, because the variable is the per-block generation
  at the current stage and does not represent the forward commitment. Use
  `anticipated_decision(N)` when the intent is to constrain the commitment level.

For context on the constraint file format see
[Generic Constraints](../reference/case-format.md).

---

## Validation Rules

Cobre's layered validation pipeline checks the following conditions on thermal
units. Violations are reported as error messages with the failing unit's `id`.

| Rule                                  | Error Class          | Description                                                                                                                     |
| ------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Bus reference integrity               | Reference error      | Every `bus_id` must match an `id` in `buses.json`.                                                                              |
| Non-negative cost                     | Schema error         | `cost_per_mwh` must be ≥ 0.0.                                                                                                   |
| Generation bounds ordering            | Physical feasibility | `min_mw` must be less than or equal to `max_mw`.                                                                                |
| Anticipated lead exclusivity          | Schema error         | `anticipated_config` must set exactly one of `lead_stages` or `lead_time_hours`; setting both or setting neither fails to load. |
| Anticipated stage-count lead validity | Physical feasibility | When `anticipated_config.lead_stages` is present, it must be a positive integer (`>= 1`).                                       |
| Anticipated physical lead validity    | Physical feasibility | When `anticipated_config.lead_time_hours` is present, it must be finite and `> 0.0`.                                            |

The cadence-transition advisory and the K=0 sub-stage-lead diagnostic (above) are
non-blocking advisories, not entries in this table — they are logged but never
fail validation or setup.

---

## Related Pages

- [Anatomy of a Case](../tutorial/anatomy-of-a-case.md) — walks through the complete `1dtoy` thermal definitions
- [Building a System](../tutorial/building-a-system.md) — step-by-step guide to writing `thermals.json` from scratch
- [System Modeling](./system-modeling.md) — overview of all entity types and how they interact
- [Case Format Reference](../reference/case-format.md) — complete JSON schema for all input files
