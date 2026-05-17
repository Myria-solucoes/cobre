# Energy Variables

Cobre computes five energy-related quantities for each hydro plant at every stage
and writes them to `simulation/hydros/`. These quantities are derived from
productivity coefficients that summarise how efficiently each plant — and its
downstream cascade — converts water volume into electrical energy. This page
explains what those coefficients are, how they are derived, and what the five
output columns mean.

---

## Equivalent Productivity (ρ_eq)

The **equivalent productivity** `ρ_eq` [MW/(m³/s)] is a single scalar that
represents the power yield per unit of turbined flow at a specific operating
point `(V_ref, Q_ref)`. It collapses the head, tailrace, and hydraulic loss
effects into one number for a given stage.

For the two fixed-productivity models (`constant_productivity` and
`linearized_head`), `ρ_eq` is taken directly from the plant's
`productivity_mw_per_m3s` field in `hydros.json` (or its stage-range override
from `hydro_production_models.json`). For FPHA plants the head is variable, so
`ρ_eq` is computed at a reference operating point:

```text
ρ_eq = ρ_esp × h_eq(V_ref, Q_ref)

where:
  ρ_esp  = specific productivity [MW/(m³/s)/m]
  h_eq   = h_fore(V_ref) − h_tail(Q_ref) − h_loss  [m]
  h_fore = forebay elevation interpolated from the VHA curve at V_ref
  h_tail = tailrace elevation at Q_ref (0 if no tailrace model)
  h_loss = hydraulic head loss at Q_ref (0 if no loss model)
```

The reference operating point defaults to:

```text
V_ref = V_min + fraction × (V_max − V_min)
Q_ref = max_turbined_m3s
```

where `fraction` is a per-`(hydro, season)` value resolved from the reference
volume configuration.

### Derivation Precedence for FPHA Plants

Cobre resolves `ρ_eq` for each FPHA hydro in the following priority order at
each stage:

1. **Override table** — an explicit `equivalent_productivity_mw_per_m3s` entry in
   `system/hydro_energy_productivity.parquet` for the `(hydro_id, stage_id)` pair
   (or a per-hydro default row with `stage_id = NULL`).
2. **VHA geometry + ρ_esp** — `ρ_esp` from the plant's
   `specific_productivity_mw_per_m3s_per_m` field in `hydros.json` and VHA rows
   from `system/hydro_geometry.parquet`, evaluated at `(V_ref, Q_ref)`.
3. **Error** — if neither source is available, `StudySetup::new` returns:

```text
FPHA hydro '<name>' (<id>) cannot derive ρ_eq for stage <N>:
no VHA geometry + ρ_esp pair is present and no override entry exists.
Remediation: (1) supply VHA geometry rows and specific_productivity (ρ_esp)
for this hydro, (2) add an entry in system/hydro_energy_productivity.parquet,
or (3) change the hydro's generation_model away from FPHA.
```

Non-FPHA plants always use path (1) from the override table or their stored
`productivity_mw_per_m3s` scalar; they are never blocked by the FPHA gate.

---

## Accumulated Productivity (ρ_acum)

The **accumulated productivity** `ρ_acum` [MW/(m³/s)] sums the equivalent
productivities along the cascade from the plant itself down to the last plant
before the sea (or tail of the river). A unit of water flowing through the
entire downstream chain generates `ρ_acum` megawatts in aggregate.

```text
ρ_acum(hydro) = ρ_eq(hydro) + ρ_acum(downstream hydro)
```

For the plant at the tail of the cascade (no downstream neighbour):

```text
ρ_acum(tail) = ρ_eq(tail)
```

### Two-Plant Cascade Example

Consider two plants, A and B, where A discharges into B:

```
River → [Reservoir A] → turbine A → [Reservoir B] → turbine B → tailwater
```

Suppose at a given stage:

```text
ρ_eq(A) = 2.50 MW/(m³/s)
ρ_eq(B) = 1.80 MW/(m³/s)
```

Then:

```text
ρ_acum(B) = ρ_eq(B)              = 1.80 MW/(m³/s)
ρ_acum(A) = ρ_eq(A) + ρ_acum(B) = 2.50 + 1.80 = 4.30 MW/(m³/s)
```

Water released by A eventually passes through both turbines; its energy value is
4.30 MW per m³/s of turbined flow.

---

## The Five Output Columns

All five columns appear in every row of `simulation/hydros/`. The schema
position is after `generation_mwh` and before `spillage_cost`.

### `equivalent_productivity_mw_per_m3s`

The `ρ_eq` value for this plant at this stage, in MW/(m³/s). Never null.

Derived as described above: override table first, then VHA geometry, then stored
scalar for non-FPHA models.

### `accumulated_productivity_mw_per_m3s`

The `ρ_acum` value for this plant at this stage, in MW/(m³/s). Never null.

For a tail plant, equals `equivalent_productivity_mw_per_m3s`. For a headwater
plant in a long cascade, may be several times larger.

### `incremental_inflow_energy_mw`

The power equivalent of the natural incremental inflow to this plant at this
stage, expressed as an average MW over the stage:

```text
incremental_inflow_energy_mw = ρ_acum × incremental_inflow_m3s
```

This is the ENA (Energia Natural Afluente) contribution of this plant's
incremental inflow in MW. It measures how much firm energy the incoming water
represents considering the full cascade downstream.

Using the two-plant cascade above with an incremental inflow to A of
200 m³/s:

```text
incremental_inflow_energy_mw(A) = 4.30 × 200 = 860 MW
```

### `stored_energy_initial_mwh`

The energy content of the water stored in the reservoir at the beginning of the
stage, expressed in MWh:

```text
stored_energy_initial_mwh = (storage_initial_hm3 − V_min) × ρ_acum × 1e6 / 3600
```

The factor `1e6 / 3600` converts hm³ to m³ and then seconds to hours (1 hm³ =
1×10⁶ m³; 1 MWh = 3600 MWs = 3600 MW·s). Only the usable storage above the
minimum operational volume `V_min` is counted.

Using the cascade example with V_min(A) = 50 hm³ and storage_initial(A) = 200 hm³:

```text
stored_energy_initial_mwh(A) = (200 − 50) × 4.30 × 1e6 / 3600 ≈ 179,167 MWh
```

### `stored_energy_final_mwh`

Same formula as `stored_energy_initial_mwh`, applied to `storage_final_hm3`:

```text
stored_energy_final_mwh = (storage_final_hm3 − V_min) × ρ_acum × 1e6 / 3600
```

This column is the EARM (Energia Armazenada Reservatório Montante) at the end
of the stage in MWh.

---

## Productivity Override File

`system/hydro_energy_productivity.parquet` is an optional file that allows you
to override any of the four scalars (`ρ_eq`, `V_ref`, `Q_ref`, `ρ_esp`) on a
per-`(hydro, stage)` basis. Rows with `stage_id = NULL` serve as a per-hydro
default that applies to all stages not covered by a stage-specific row.

See the [Case Directory Format](../reference/case-format.md#systemhydro_energy_productivityparquet)
reference for the full column table and validation rules.

---

## Diversion Channels

Plants with a diversion channel are treated as standard cascade members for
energy-variable purposes. The plant's `ρ_eq` and `ρ_acum` are derived from its
own production model and its position in the main cascade topology. Diverted
flow is accounted for in `incremental_inflow_m3s` through the normal water
balance; the energy variables reflect the declared topology without special
diversion-specific adjustments.
