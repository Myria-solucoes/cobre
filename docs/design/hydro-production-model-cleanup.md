# Hydro production model — data-model cleanup

**Status:** Draft / design loop (no code changes yet)
**Scope:** `cobre-core`, `cobre-io`, `cobre-sddp`, `cobre-python`, `book/`, schemas, `CHANGELOG.md`
**Author:** design review, 2026-06-03

## Summary

A review of the hydro production-function data model surfaced five issues. This
spec proposes how to resolve all five. The headline change is the **removal of
the `LinearizedHead` generation model**, which is declared, documented, and
validated as head-dependent but is never implemented — it silently resolves to
plain constant productivity. Three further changes are correctness/clarity
hardening: a public-type name collision, a stringly-typed model field, and a
silent `productivity = 0.0` fallback on the release path. The fifth — more
substantive — change **removes `EfficiencyModel` and unifies the computed-FPHA
fitting constant on `ρ_esp`** (specific productivity), eliminating a redundant,
unenforced double-specification of turbine efficiency.

No edits are made by this document. It scopes the work so an implementation plan
can follow.

## Background — the three layers today

The production model flows through three layers:

| Layer                  | Crate / file                                 | Type(s)                                                                                                                                                                                                  |
| ---------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Config / wire**      | `cobre-io` `extensions/production_models.rs` | `ProductionModelConfig` → `SelectionMode::{StageRanges, Seasonal}` → `StageRange` / `SeasonConfig` (each `model: String`, `fpha_config`, `productivity_mw_per_m3s`); `FphaColumnLayout`; `FittingWindow` |
| **Entity selector**    | `cobre-core` `entities/hydro.rs`             | `HydroGenerationModel::{ConstantProductivity, LinearizedHead, Fpha}` (pure selector)                                                                                                                     |
| **Resolved / runtime** | `cobre-sddp` `hydro_models.rs`               | `ResolvedProductionModel::{ConstantProductivity{productivity}, Fpha{planes}}`; `FphaPlane`; `ProductionModelSet`; `ProductionModelSource`                                                                |

Resolution path: `hydro_production_models.json` config (or, when absent, the
entity's `HydroGenerationModel` from `hydros.json`) →
`resolve_production_models` → per-`(hydro, stage)` `ResolvedProductionModel` →
LP builder. A per-`(hydro, stage)` override table
(`HydroEnergyProductivityOverride`, from `hydro_energy_productivity.parquet`)
supplies non-FPHA productivity and FPHA ρ_eq inputs.

## Goals

- Remove the `LinearizedHead` model end-to-end; reject it at parse time (via the
  standard serde unknown-variant error).
- Eliminate the `FphaColumnLayout` name collision between `cobre-io` and
  `cobre-sddp` by renaming the `cobre-io` type to `FphaConfig`.
- Replace the stringly-typed `model` field in `hydro_production_models.json`
  parsing with the existing `HydroGenerationModel` enum.
- Convert the silent `productivity = 0.0` fallback in the resolution path into a
  hard, all-builds error.
- Remove `EfficiencyModel`; source the computed-FPHA fitting constant from the
  entity's `ρ_esp` (`specific_productivity_mw_per_m3s_per_m`), the same field
  energy conversion already uses.

## Non-goals

- Implementing a _real_ head-dependent linearized model (the opposite — we are
  removing the non-functional stand-in). A genuine head-linearization model can
  be proposed separately later if desired.
- Changing the FPHA hyperplane **structure** (the `g ≤ γ₀ + (γᵥ/2)(V_in+V_out) +
γ_q·q + γ_s·s` row, or the average-storage contract) or the override-table
  semantics. Change 5 changes only how the fitting _constant_ is sourced (η → ρ_esp),
  which rescales γ values; it does not touch the LP row layout.
- Restructuring `hydros.json`'s `generation` block (already a properly-tagged
  enum) beyond dropping one variant.

---

## Change 1 — Remove `LinearizedHead`

### Problem

`HydroGenerationModel::LinearizedHead` (`cobre-core/.../hydro.rs:107`) is
documented as "head-dependent productivity linearized around an operating
point … computed from the current head at the start of each time step," and
dimensional validation (`cobre-io/.../validation/dimensional.rs:246,275`)
_requires_ such a hydro to ship ≥ 2 `hydro_geometry` rows, exactly like FPHA.
But at resolution it is indistinguishable from constant productivity:

- `determine_source` maps `ConstantProductivity | LinearizedHead → DefaultConstant`
  (`hydro_models.rs:1122`).
- `resolve_stage_model` returns `ResolvedProductionModel::ConstantProductivity`
  for anything that is not `"fpha"`.
- `ResolvedProductionModel` has **no** `LinearizedHead` variant; the test
  `linearized_head_yields_input_scalar` (`energy_conversion.rs:934`) _asserts_
  it returns the plain input scalar.

A user who selects `linearized_head` is forced to supply geometry that is never
read, and silently receives constant productivity.

### Proposed change

Remove the variant and its handling everywhere; reject the string at parse time.

**Code:**

- `cobre-core/src/entities/hydro.rs` — drop the `LinearizedHead` enum variant
  (enum becomes 2 variants). Update the `specific_productivity_*` doc comment
  that references it.
- `cobre-io/src/system/hydros.rs` — drop `RawGeneration::LinearizedHead`
  (`:197`), its arm in `bounds()` (`:235`) and `convert_generation` (`:800,807`).
  `#[serde(tag = "model", deny_unknown_fields)]` then makes
  `"model": "linearized_head"` a serde _unknown variant_ error.
- `cobre-io/src/validation/dimensional.rs` — remove the `LinearizedHead` arm
  (`:246`) and the `|| ... == "linearized_head"` clauses (`:275,281,284`). After
  this, **only FPHA** requires `hydro_geometry` rows.
- `cobre-sddp/src/hydro_models.rs` — `determine_source` no-config arm becomes
  `ConstantProductivity` only (`:1122`); update the doc table (`:756`).
- `cobre-sddp/src/simulation/types.rs:126` — drop `LinearizedHead` from the doc.
- `cobre-python/src/model.rs:285` — collapse the match to 2 arms (Python parity
  preserved: getter still returns `None`).

**Parse-error UX (Decision A — resolved):** rely on the **standard serde
unknown-variant error** (`unknown variant 'linearized_head', expected one of
…`). No custom interception. In `production_models.rs` this already routes
through the existing unknown-variant → `SchemaError` mapping; in `hydros.rs`
the internally-tagged `RawGeneration` surfaces it as a serde/`ParseError`.

**Tests:** update/remove constructors and assertions at
`hydro.rs:360,653–657` (serde round-trip), `energy_conversion.rs:934`
(`linearized_head_yields_input_scalar`), `productivity_resolution.rs:567`,
`dimensional.rs:1011–1021`, `hydro_models.rs:1948–1954`, `hydros.rs:1142,1158`.
Add a new test asserting `linearized_head` is **rejected** as a parse/schema
error in both files.

**Docs:** update 7 `book/` files (`guide/hydro-plants.md`,
`guide/energy-variables.md`, `tutorial/building-a-system.md`,
`reference/case-format.md`, `reference/error-codes.md`, and the two schema
files); regenerate `book/src/schemas/{hydros,production_models}.schema.json`
(`--features schema`); add a `CHANGELOG.md` **Removed** entry. (Historical
CHANGELOG entries at `:655,:706,:1197` are an immutable record of past releases
— leave them; do not rewrite history.)

### Migration & compatibility

- **Input format (breaking):** cases that use `linearized_head` are now
  rejected. Migration is a one-token rename to `constant_productivity` — the
  resolved behavior is byte-identical (productivity already came from
  `hydro_production_models.json` / `hydro_energy_productivity.parquet`, not the
  variant). Documented in CHANGELOG + book.
- **Risk:** low. Removes a path that never did what it claimed.

---

## Change 2 — Resolve the `FphaColumnLayout` name collision

### Problem

Two unrelated public types share the name `FphaColumnLayout`:

- `cobre-io::extensions::FphaColumnLayout` — the JSON **FPHA configuration**
  block (`source`, discretization points, `fitting_window`). The name is a
  **misnomer**: it is not a column layout.
- `cobre-sddp::indexer::FphaColumnLayout` (`indexer.rs:818`) — the **LP column
  layout** (`hydro_indices`, `planes_per_hydro`). Accurately named.

`hydro_models.rs` imports the `cobre-io` type and its error strings say "no
FphaColumnLayout was found," which reads ambiguously next to the indexer type.

### Proposed change

Rename the **`cobre-io`** type to **`FphaConfig`** (Decision B — resolved;
aligns with the `fpha_config` JSON field). The indexer type keeps its correct
name, so the collision and the misnomer are fixed in one move.

- Rename the struct in `production_models.rs`, its re-export in
  `extensions/mod.rs`, and all references in `production_models.rs` and
  `hydro_models.rs` (imports, `find_fpha_config_for_stage` return type, error
  strings).
- JSON field name `fpha_config` is **unchanged** — this is a Rust-API rename
  only.

### Compatibility

- **Rust API (breaking):** `cobre-io` public type renamed. Pre-1.0; acceptable.
  Note in CHANGELOG. **No input-format change.**

---

## Change 3 — Replace the stringly-typed `model` field with `HydroGenerationModel`

### Problem

`StageRange.model`, `SeasonConfig.model`, and `Seasonal.default_model` are
`String`, validated by scattered string comparisons (`r.model == "fpha"`, etc.)
across `hydro_models.rs`, `dimensional.rs`, and `productivity_resolution.rs`
(~10 sites). `cobre-core` already exposes `HydroGenerationModel` with
`#[serde(rename_all = "snake_case")]`, which deserializes from exactly those
strings, and `cobre-io` already imports it.

### Proposed change (minimal swap — Decision C resolved)

- In `production_models.rs`, change the public `StageRange.model` /
  `SeasonConfig.model` / `Seasonal.default_model` from `String` to
  `HydroGenerationModel`, and the matching raw-type fields. Serde then accepts
  `constant_productivity` / `fpha` and rejects anything else **at parse time** —
  which makes Change 1's rejection of `linearized_head` automatic.
- Replace every `model == "fpha"` / `== "constant_productivity"` comparison with
  enum equality / `matches!`. `find_model_for_stage` returns
  `(HydroGenerationModel, Option<f64>)`; the
  `… == Some("fpha")` check becomes `… == Some(HydroGenerationModel::Fpha)`.
- Remove now-dead string-validity checks in `production_models.rs` (invalid
  names are now parse errors). **Keep** the field-combination cross-checks
  (FPHA requires `fpha_config`; non-FPHA rejects `productivity` only for FPHA;
  productivity sign/finiteness) — those are about field _combinations_, not name
  validity.

### Genericity-rule check

`HydroGenerationModel` is a domain entity already living in `cobre-core` and is
not algorithm-specific (no SDDP/Benders reference), so using it in `cobre-io`
satisfies the infrastructure-crate genericity rule.

### Compatibility

- **Rust API (breaking):** `StageRange` / `SeasonConfig` field types change.
  Pre-1.0; acceptable. **No input-format change** (same JSON strings).
- **Coupling:** do Change 1 and Change 3 together — Change 3's enum is what makes
  Change 1's `linearized_head` rejection a clean parse error.

---

## Change 4 — Sentinel `productivity = 0.0` → hard error

### Problem

`resolve_stage_model` (`hydro_models.rs:1187,1202`) falls back to
`productivity = 0.0` behind a `debug_assert!(false, …)` when no productivity
source is found, trusting `cobre_io::validation::productivity_resolution` to
guarantee exactly one source. In a **release** build a validator gap silently
produces a plant that generates nothing, rather than failing.

### Proposed change

- Replace the two `unwrap_or_else(|| { debug_assert!(false, …); 0.0 })` sites
  with `ok_or_else(|| SddpError::Validation(…))?` (the function already returns
  `Result`). The error names the hydro/stage and points at the
  `productivity_resolution` validator.
- **Leave `default_from_system` (`:439`) unchanged.** Its `0.0` is a _legitimate_
  placeholder for non-root MPI ranks (which overwrite it from the broadcast
  payload) and tests; it is never the from-scratch resolution path. Add a clarifying
  comment distinguishing the two.

### Compatibility

- **Behavioral:** a genuinely-absent productivity now errors in all builds
  (was silent-zero in release). This is the intent; risk is only that a latent
  bad case previously masked by silent-zero now fails loudly. Low and desirable.
- Determinism unaffected (errors are deterministic).

---

## Change 5 — Remove `EfficiencyModel`; unify the FPHA fitting constant on `ρ_esp`

### Problem

`EfficiencyModel` (`cobre-core/.../hydro.rs:176`) has a single `Constant { value }`
variant and is consumed **only** by computed-FPHA fitting. Unlike `LinearizedHead`,
η is **not** dead: the production function is `P = K·η·q·h_net` with
`K = 9.81e-3` (gravity + unit conversion, _no_ efficiency, `fpha_fitting.rs:711,838`),
so η scales every fitted hyperplane coefficient. The `kappa` envelope-tightness
factor does **not** reabsorb it (kappa = `min(φ / max_plane)` is invariant to a
uniform η scale). Dropping η naively (→ 1.0) would inflate FPHA generation by
`1/η` — a silent correctness regression, not a cleanup.

The real defect is **redundant, unenforced double-specification** of efficiency
for a computed-FPHA hydro:

| Path                                      | Formula               | Efficiency enters as                                                                         |
| ----------------------------------------- | --------------------- | -------------------------------------------------------------------------------------------- |
| **Fitting** → LP generation hyperplanes   | `K·η·q·h_net`         | explicit `EfficiencyModel.value`                                                             |
| **Energy conversion** → stored/inflow MWh | `ρ_eq = ρ_esp · h_eq` | embedded in `ρ_esp` (the Brazilian _produtibilidade específica_ already bundles `9.81e-3·η`) |

Nothing enforces that the explicit η and the η baked into `ρ_esp` agree, and the
enum is over-structured for a single scalar.

### Proposed change (Decision D — resolved: Option A, "unify on `ρ_esp`")

`ρ_esp` has the **same units** as `K·η` (`MW/((m³/s)·m)`) and _is_ "`K·η`
bundled". Route the fitting onto it:

```text
P = ρ_esp · q · h_net          (drop K and the separate η)
```

`ρ_esp` becomes the single source of specific productivity, the fitted
hyperplanes are consistent-by-construction with `ρ_eq = ρ_esp · h_eq` (both use
`ρ_esp`), and the double-specification disappears. Energy conversion already
sources `ρ_esp` from the **entity field** `specific_productivity_mw_per_m3s_per_m`
(`energy_conversion.rs:473`), so the fitting reads the _same field_ — no override-
table threading.

**Code:**

- `cobre-core/src/entities/hydro.rs` — remove the `EfficiencyModel` enum
  (`:176`) and the `Hydro.efficiency` field (`:244`); update the `ρ_esp` field
  doc (`:227`); drop the re-exports (`lib.rs:56`, `entities/mod.rs:17`).
- `cobre-io/src/system/hydros.rs` — remove `RawEfficiency` (`:304–311`), the
  raw `efficiency` field (`:109–111`), its conversion (`:703`), and the import
  (`:50`). With `deny_unknown_fields`, an `"efficiency"` key in `hydros.json`
  becomes a parse error.
- `cobre-sddp/src/fpha_fitting.rs` — `ProductionFunction.efficiency: f64` →
  `rho_esp: f64`; constructor takes `ρ_esp` instead of `Option<&EfficiencyModel>`
  (`:769–784`); `evaluate` (`:838`) and the tangent-plane derivative (`:888`) use
  `self.rho_esp` (no `K`); **remove the now-dead `K` constant** (`:711`);
  `fit_fpha_planes` (`:1595`) reads `hydro.specific_productivity_mw_per_m3s_per_m`;
  drop the `EfficiencyModel` import (`:14`). Update the many tests that pass
  `Some(&EfficiencyModel::Constant { … })` to pass a `ρ_esp` scalar.
- `cobre-sddp/src/hydro_models.rs` — in `validate_computed_prerequisites`
  (`:1045`) replace the `efficiency.is_none()` check with
  `specific_productivity_mw_per_m3s_per_m.is_none()` (`ρ_esp` is now required for
  computed FPHA); update the policy-rationale doc and error message.
- `cobre-python` — no efficiency surface today (parity unaffected).
- Docs: update `book/` (`guide/hydro-plants.md`, `reference/case-format.md`,
  `crates/core.md`, `guide/performance-accelerators.md`, `schemas/hydros.schema.json`);
  regenerate the schema; add a CHANGELOG **Removed** entry.

### Migration & compatibility

- **Input format (breaking):** an `"efficiency"` block in `hydros.json` is now
  rejected, and computed-FPHA hydros must carry
  `specific_productivity_mw_per_m3s_per_m`. Mechanical migration:
  `ρ_esp = 9.81e-3 · η_old`. A hydro that relied on a parquet `ρ_eq` override and
  had no entity `ρ_esp` must now supply it (same formula).
- **Behavioral:** for any computed-FPHA hydro whose entity `ρ_esp` ≠ `K·η_old`
  (or that lacked entity `ρ_esp`), the fitted γ values change. This is the
  intended unification (the two sources should always have agreed), but it means
  expected-output files for affected FPHA cases must be **re-blessed**.
- **Orthogonal to the FPHA average-storage contract** (`.claude/rules/sddp.md`):
  that governs row construction (`γᵥ/2` on both storage columns), which is
  untouched — only γ magnitudes change.
- **Risk:** medium — this is the one change that alters FPHA numerics. Isolate it
  and validate against the deterministic FPHA cases + the slow-tests
  plane-selection suite.

---

## Resolved decisions

| #     | Decision                                       | Resolution                                                                                                                                                                                                                                                       |
| ----- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A** | Parse-error UX for rejected `linearized_head`  | **Standard serde unknown-variant error** — no custom interception.                                                                                                                                                                                               |
| **B** | New name for the `cobre-io` `FphaColumnLayout` | **`FphaConfig`** (aligns with the `fpha_config` JSON field).                                                                                                                                                                                                     |
| **C** | Scope of Change 3                              | **Minimal swap**: `String` → `HydroGenerationModel`, keep separate `fpha_config` / `productivity` fields + their cross-checks. (A larger data-carrying enum that makes illegal states unrepresentable is possible but changes the parsing structure — deferred.) |
| **D** | `EfficiencyModel`                              | **Option A — unify on `ρ_esp`**: remove the enum, route the computed-FPHA fitting constant through the entity `ρ_esp` (`P = ρ_esp · q · h_net`), make `ρ_esp` required for computed FPHA. (Naive η→1 removal rejected — it silently inflates generation.)        |

## Proposed sequencing

1. **Change 2** (rename) — independent, mechanical.
2. **Change 4** (hard error) — independent, small.
3. **Changes 1 + 3 together** (remove `LinearizedHead` + enum-typed `model`) —
   the substantive parsing change set; the enum makes the rejection automatic.
4. **Change 5** (remove `EfficiencyModel` → `ρ_esp`) — **last and isolated**; it
   is the only change that alters FPHA numerics, so land it on its own and
   re-bless affected FPHA expected-outputs before merging.
5. **Docs/schemas/CHANGELOG** — fold into each change; final schema regeneration
   and book pass at the end.

Each step keeps `cargo build/test --workspace --all-features` green, clippy
pedantic at zero warnings, and `cargo fmt` clean.

## Validation & testing

- **Unit:** `linearized_head` rejected as a parse/schema error in both files;
  `HydroGenerationModel` round-trips `constant_productivity` / `fpha`; FPHA still
  requires `fpha_config`; productivity sign/finiteness preserved;
  productivity-absent now errors in `resolve_stage_model`. For Change 5: an
  `"efficiency"` block in `hydros.json` is rejected; computed FPHA without entity
  `ρ_esp` errors in `validate_computed_prerequisites`; a fitting unit test
  asserts `P = ρ_esp · q · h_net` (the `K·η` path is gone).
- **Determinism:** declaration-order invariance preserved (sort key unchanged);
  bit-identical results for unaffected cases.
- **Integration:** deterministic d-cases still pass; no shipped case uses
  `linearized_head` (confirmed — only `book/`, `CHANGELOG`, schemas reference it).
  For Change 5, **re-bless** any computed-FPHA deterministic case (and the
  slow-tests plane-selection suite) whose γ values shift because entity `ρ_esp`
  ≠ `K·η_old`; migrate any case-fixture `efficiency` block to `ρ_esp = 9.81e-3·η`.
- **Parity:** `cobre-python` builds; the `productivity_mw_per_m3s` getter still
  returns `None`.
- **Docs/schema:** regenerate both JSON schemas; sweep the book files (7 for
  `linearized_head`, 5 for `efficiency`).

## Out of scope / future

- A genuine head-dependent linearized production model (distinct from the removed
  stand-in).
- A data-carrying `model` enum that fuses `model` + `fpha_config` + `productivity`
  so illegal field combinations are unrepresentable (Decision C's larger
  alternative).
- A `Factor`/`Piecewise`-style richer efficiency or hydraulic-loss model — out of
  scope; Change 5 collapses efficiency into `ρ_esp` rather than enriching it.
- Letting the `FittingWindow` percentile track a temporal/seasonal operating band:
  today it is a fraction of the **static** `[min_storage_hm3, max_storage_hm3]`
  range (the computed-FPHA fit is stage-independent), ignoring seasonal reference
  volumes and the per-`(hydro,stage)` `V_ref` override. A temporal fit window
  would require stage-dependent fitting — a separate effort.

## Next step

Once Open Decisions A–C are confirmed, generate the implementation plan
(epics/tickets) into `plans/` via `/plan`, then execute via `/implement-plan`.
