# Scalar Parameters

A **scalar parameter** is a named, typed value that can be referenced by name
from generic-constraint coefficient expressions. Instead of hard-coding a
coefficient in the constraint expression, you declare the parameter once in an
input file and reference it with the `@name` sigil. The solver resolves each
parameter to a concrete `f64` value before building the LP for each stage.

Parameters are useful when:

- The same physical quantity (e.g. a plant's equivalent productivity) appears in
  multiple constraints and should stay consistent automatically.
- A coefficient varies by stage or season and you want a single place to
  maintain those values rather than editing multiple constraint expressions.
- The coefficient is derived from hydro geometry data and should be kept in sync
  with the model automatically.

---

## Input Files

Scalar parameters are loaded from a single JSON file:

### `system/scalar_parameters.json`

The file is optional. When absent, no parameters are loaded and any `@name`
token in a constraint expression causes a load error.

**Top-level object shape:**

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/scalar_parameters.schema.json",
  "scalar_parameters": [
    { "id": 1, "name": "discount_rate", "kind": "constant", "value": 0.05 },
    {
      "id": 2,
      "name": "demand",
      "kind": "per_stage",
      "values": [
        [0, 100.0],
        [1, 110.0],
        [2, 105.0]
      ]
    },
    {
      "id": 3,
      "name": "wet_season_factor",
      "kind": "seasonal",
      "values": [
        [0, 1.2],
        [1, 0.8]
      ]
    },
    {
      "id": 4,
      "name": "hydro_prod",
      "kind": "computed",
      "computed_spec": { "tag": "equivalent_productivity", "hydro_id": 7 }
    }
  ]
}
```

**Per-entry fields present on every parameter:**

| Field  | Type    | Description                                                             |
| ------ | ------- | ----------------------------------------------------------------------- |
| `id`   | integer | Unique parameter identifier (int32). Must be unique across all entries. |
| `name` | string  | Unique parameter name. Non-empty, no leading or trailing whitespace.    |
| `kind` | string  | One of `constant`, `per_stage`, `seasonal`, `computed`.                 |

**Kind-specific payload fields** (present only for the matching `kind`):

| `kind`      | Extra field(s)                                                            |
| ----------- | ------------------------------------------------------------------------- |
| `constant`  | `"value": <f64>` — one finite value for all stages                        |
| `per_stage` | `"values": [[stage_id, value], ...]` — contiguous from 0, all finite      |
| `seasonal`  | `"values": [[season_id, value], ...]` — unique season indices, all finite |
| `computed`  | `"computed_spec": { "tag": "<variant>", "hydro_id": <int> }`              |

Unknown fields on any entry are rejected at parse time.

---

## Parameter Kinds

### `constant`

One value applied to every stage.

```json
{ "id": 1, "name": "demand_scale", "kind": "constant", "value": 1.05 }
```

### `per_stage`

One value per study stage. The `values` array contains `[stage_id, value]`
pairs. Stage indices must form a contiguous range starting at 0 (i.e.
`[0, 1, 2, …, N-1]`). Duplicate indices and gaps are both rejected.

```json
{
  "id": 2,
  "name": "hydro_limit_factor",
  "kind": "per_stage",
  "values": [
    [0, 0.9],
    [1, 0.85],
    [2, 0.8]
  ]
}
```

### `seasonal`

One value per season, keyed by `season_id`. The value for a given stage is
looked up by the stage's season. Season indices need not be contiguous but must
be unique within the entry.

```json
{
  "id": 3,
  "name": "wet_season_weight",
  "kind": "seasonal",
  "values": [
    [0, 1.2],
    [1, 0.95],
    [2, 0.8],
    [3, 1.1]
  ]
}
```

### `computed`

The value is derived from hydro geometry data by the solver — no numeric values
are needed. The `computed_spec` object carries the variant tag and plant
reference:

```json
{
  "id": 4,
  "name": "rho_eq_h1",
  "kind": "computed",
  "computed_spec": { "tag": "equivalent_productivity", "hydro_id": 1 }
}
```

---

## Computed Parameter Catalog

Seven hydro-indexed quantities are available as computed parameters:

| `tag`                      | Symbol | Unit        | Description                                    |
| -------------------------- | ------ | ----------- | ---------------------------------------------- |
| `equivalent_productivity`  | ρ_eq   | MW/(m³/s)   | Equivalent productivity at the reference point |
| `accumulated_productivity` | ρ_acum | MW/(m³/s)   | Accumulated cascade productivity               |
| `reference_volume`         | V_ref  | hm³         | Reference reservoir volume                     |
| `reference_turbine`        | Q_ref  | m³/s        | Reference turbined flow                        |
| `min_storage`              | V_min  | hm³         | Minimum operational reservoir storage          |
| `max_storage`              | V_max  | hm³         | Maximum operational reservoir storage          |
| `specific_productivity`    | ρ_esp  | MW/(m³/s)/m | Specific productivity from `hydros.json`       |

All seven are stage-resolved: the value provided to the LP builder is the scalar
for the stage currently being built.

---

## Referencing a Parameter in a Constraint

Generic constraints in `constraints/generic_constraints.json` carry a free-form
`expression` string. Normally a coefficient is a literal number:

```json
{
  "id": 0,
  "name": "min_cascade_energy",
  "expression": "3.6 * hydro_generation(1) + 3.6 * hydro_generation(2)",
  "sense": ">=",
  "slack": { "enabled": true, "penalty": 5000.0 }
}
```

Replace literal coefficients with `@name` to reference a parameter. The
expression parser recognises three term shapes involving `@`:

```text
@name * variable(...)              — parameter coefficient, implicit scale 1.0
literal * @name * variable(...)    — literal scale multiplied by parameter coefficient
```

Using a computed parameter instead:

```json
{
  "id": 0,
  "name": "min_cascade_energy",
  "expression": "@rho_eq_h1 * hydro_generation(1) + @rho_eq_h2 * hydro_generation(2)",
  "sense": ">=",
  "slack": { "enabled": true, "penalty": 5000.0 }
}
```

With the definitions above (`rho_eq_h1` resolved from the VHA geometry for hydro 1,
`rho_eq_h2` for hydro 2), the LP coefficient is updated automatically each stage
as the equivalent productivity changes.

If `@name` is used but no parameter with that name has been loaded, the case
fails with a schema error during load.

---

## Validation Rules

- `id` values must be unique across all entries.
- `name` values must be unique (case-sensitive), non-empty, and have no leading
  or trailing whitespace.
- `kind` must be exactly one of `constant`, `per_stage`, `seasonal`, or
  `computed`.
- For `constant`: `value` must be present and finite.
- For `per_stage`: `values` must be present and non-empty; the `stage_id`
  integers must form a contiguous range starting at 0; all values must be
  finite.
- For `seasonal`: `values` must be present and non-empty; `season_id` values
  must be unique within the entry; all values must be finite.
- For `computed`: `computed_spec` must be present with a valid `tag` (one of
  the seven listed above) and a `hydro_id` integer. Existence of the referenced
  hydro is validated during cross-reference checks after all entity files are
  loaded.
- Unknown JSON fields on any entry are rejected immediately at parse time.
