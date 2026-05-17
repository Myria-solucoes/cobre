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

Scalar parameters are loaded from two parquet files:

### `system/scalar_parameter_definitions.parquet`

One row per parameter. Defines the identity and kind of each parameter.

| Column          | Parquet type | Nullable | Description                                                            |
| --------------- | ------------ | -------- | ---------------------------------------------------------------------- |
| `id`            | INT32        | no       | Unique parameter identifier (`EntityId`)                               |
| `name`          | UTF8         | no       | Unique parameter name (non-empty, no leading/trailing spaces)          |
| `kind`          | UTF8         | no       | One of `constant` / `per_stage` / `seasonal` / `computed`              |
| `computed_spec` | UTF8         | yes      | Non-null and parseable when `kind == "computed"`; null for other kinds |

### `system/scalar_parameter_values.parquet`

One row per (parameter, stage or season). Supplies the numeric values for
`constant`, `per_stage`, and `seasonal` parameters. Not used for `computed`
parameters.

| Column         | Parquet type | Nullable | Description                            |
| -------------- | ------------ | -------- | -------------------------------------- |
| `parameter_id` | INT32        | no       | `EntityId` of the parameter            |
| `stage_id`     | INT32        | yes      | Zero-based stage index for `per_stage` |
| `season_id`    | INT32        | yes      | Season id for `seasonal`               |
| `value`        | DOUBLE       | no       | Finite `f64` value                     |

Both files are optional. When absent, no parameters are loaded and any `@name`
token in a constraint expression causes a load error.

---

## Parameter Kinds

### `constant`

One value applied to every stage. Requires exactly one value row with both
`stage_id` and `season_id` null.

```text
Definitions row:
  id=1  name="demand_scale"  kind="constant"  computed_spec=null

Values row:
  parameter_id=1  stage_id=null  season_id=null  value=1.05
```

### `per_stage`

One value per study stage. Requires exactly `n_stages` value rows, one for each
stage from `0` to `n_stages − 1`. Both `season_id` must be null.

```text
Definitions row:
  id=2  name="hydro_limit_factor"  kind="per_stage"  computed_spec=null

Values rows (for a 3-stage study):
  parameter_id=2  stage_id=0  season_id=null  value=0.90
  parameter_id=2  stage_id=1  season_id=null  value=0.85
  parameter_id=2  stage_id=2  season_id=null  value=0.80
```

### `seasonal`

One value per season, keyed by `season_id`. The value for a given stage is
looked up by the stage's season. Requires at least one value row; `stage_id`
must be null on all rows.

```text
Definitions row:
  id=3  name="wet_season_weight"  kind="seasonal"  computed_spec=null

Values rows:
  parameter_id=3  stage_id=null  season_id=0  value=1.20
  parameter_id=3  stage_id=null  season_id=1  value=0.95
  parameter_id=3  stage_id=null  season_id=2  value=0.80
  parameter_id=3  stage_id=null  season_id=3  value=1.10
```

### `computed`

The value is derived from hydro geometry data by the solver — no value rows are
needed. `computed_spec` carries the variant and plant reference in the form:

```text
<variant_tag>(hydro_id=<int>)
```

```text
Definitions row:
  id=4  name="rho_eq_h1"  kind="computed"  computed_spec="EquivalentProductivity(hydro_id=1)"

No values rows needed for this parameter.
```

---

## Computed Parameter Catalog

Seven hydro-indexed quantities are available as computed parameters:

| Variant tag               | Symbol | Unit        | Description                                    |
| ------------------------- | ------ | ----------- | ---------------------------------------------- |
| `EquivalentProductivity`  | ρ_eq   | MW/(m³/s)   | Equivalent productivity at the reference point |
| `AccumulatedProductivity` | ρ_acum | MW/(m³/s)   | Accumulated cascade productivity               |
| `ReferenceVolume`         | V_ref  | hm³         | Reference reservoir volume                     |
| `ReferenceTurbine`        | Q_ref  | m³/s        | Reference turbined flow                        |
| `MinStorage`              | V_min  | hm³         | Minimum operational reservoir storage          |
| `MaxStorage`              | V_max  | hm³         | Maximum operational reservoir storage          |
| `SpecificProductivity`    | ρ_esp  | MW/(m³/s)/m | Specific productivity from `hydros.json`       |

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

- `id` values must be unique across all definition rows.
- `name` values must be unique, non-empty, and have no leading or trailing whitespace.
- `kind` must be exactly one of `constant`, `per_stage`, `seasonal`, or `computed`.
- When `kind` is `computed`, `computed_spec` must be non-null, non-empty, and parse
  as `<variant_tag>(hydro_id=<int>)` using one of the seven variant tags above.
- When `kind` is not `computed`, `computed_spec` must be null or absent.
- Existence of the referenced `hydro_id` in the hydro registry is validated
  during cross-reference checks after all entity files are loaded.
