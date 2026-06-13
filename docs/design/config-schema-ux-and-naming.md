# Config Schema UX & Competitor-Mention Remediation

**Date:** 2026-06-13 · **Baseline:** `main` @ `6333bffd` (v0.8.1)
**Scope:** `training.cut_selection` schema naming/UX + workspace-wide config consistency + external-software mention sweep
**Method:** 4 parallel deep-analysis agents over the config crate, the SDDP parse/consume layer, the generated `config.schema.json`, the book, and a repo-wide mention inventory (65 mention sites catalogued).

> Internal investigation artifact. Nothing here is committed or changed. The schema redesign in
> Part 2 is a **breaking change** under `deny_unknown_fields` and needs owner sign-off before any edit.

---

## TL;DR

Your instinct is right on both counts.

1. **`cut_selection` is the worst-designed block in the config**, and it's a _symptom_ of a workspace-wide
   pattern, not a one-off. It's a flat 15-field bag where **8 of 11 functional fields are method-conditional**
   ("Ignored unless `method = …`"), the field names range from opaque (`nadic`) to leaking paper/competitor
   symbols (`active_window` = `k2`, documented against NEWAVE's `selcor.dat`), the tolerance/window naming
   families are inconsistent, and three deprecated fields are _silently ignored_. Meanwhile, the **same file**
   already contains the cure: `StoppingRuleConfig` is a clean internally-tagged enum where wrong-field-for-method
   is a parse-time error.

2. **The competitor mentions split into three buckets**, and only one should be scrubbed:
   - ~16 **rationale comments in shipped code** justify defaults/behavior by naming NEWAVE — and one of them
     ships verbatim into the user-facing `config.schema.json`. **Rephrase to domain-neutral terms** (preserve the
     rationale, drop the proper noun). This is the real work and aligns with the existing genericity philosophy.
   - **`cobre-bridge` interop** (`cobre-bridge convert newave`, `convert_newave_case`, the `inewave` dependency)
     are real external tool/API/package names. **Keep** — renaming them would misdocument a real feature.
   - **Bibliographic citations** (`de Matos 2015`, `Guigues 2017/2019`, `Diniz et al. CEPEL 2020`) are the
     explicit keep-carve-out in `.claude/rules/comments.md`. **Keep** (or relocate methodology framing to `cobre-docs`).

---

# Part 1 — Diagnosis: why `cut_selection` is confusing

`RowSelectionConfig` lives in `crates/cobre-io/src/config/training.rs:64-186`; parse/validation in
`crates/cobre-sddp/src/cut/cut_selection.rs:629-768`; consumed in `crates/cobre-sddp/src/cut/dcs.rs`.

| #   | Problem                                                                                                                                                                                                                           | Severity (UX)                         | Evidence                                                                                 |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------- |
| 1   | **Flat bag of method-conditional fields.** 8 of 11 functional fields are `Ignored unless method = X`. A user can't tell from structure which knobs apply.                                                                         | **High**                              | `training.rs:68-186`; per-field "Ignored unless"/"Not used by"/"Required when" docs      |
| 2   | **Opaque / jargon names.** `nadic` (means "max cuts added per round"); `active_window` (=`k2`), `candidate_window` (=`k1`) — paper symbols, and `k2` is documented against NEWAVE `selcor.dat`.                                   | **High**                              | `training.rs:142` (`nadic`), `:130` (`active_window`/k2), `:137` (`candidate_window`/k1) |
| 3   | **`method` is a stringly-typed discriminator** validated only at solver-setup time. A typo (`"dynmic"`) passes serde, passes the schema (no `enum` constraint), and surfaces late.                                                | **High**                              | `training.rs:75` `method: Option<String>`; error only at `cut_selection.rs:766`          |
| 4   | **3 deprecated fields silently ignored.** `threshold`, `memory_window` give _zero_ feedback when set; only `basis_activity_window` warns. All three still pollute the published schema.                                           | **High** (silent two)                 | `training.rs:84,93,172`; warn only at `setup/params.rs:144-157`                          |
| 5   | **Inconsistent naming families.** Tolerances: `tie_tolerance` / `domination_epsilon` / `violation_tolerance` / `cut_activity_tolerance` (one is "\_epsilon", same math as a "\_tolerance"). Windows: 2 live + 2 dead, 3 meanings. | Med                                   | `training.rs:102,109,148,157`                                                            |
| 6   | **`enabled` vs `method` ambiguity.** `method:"level1"` alone is a silent no-op unless `enabled:true`; `enabled:true` without `method` is a runtime error. Redundant fields, non-obvious truth table.                              | Med                                   | truth table at `cut_selection.rs:665-673`                                                |
| 7   | **Hidden cross-field coupling.** Under `dynamic`, `violation_tolerance` silently falls back to `tie_tolerance` (a level1-named field) — undocumented in the schema description.                                                   | Med                                   | `cut_selection.rs:743-746`                                                               |
| 8   | **Struct/key/genericity split.** Type is `RowSelectionConfig` ("row"), JSON key is `cut_selection`, field is `cut_activity_tolerance` ("cut"), docs mix "row"/"cut". Migration is half-done — neither noun is canonical.          | Med (UX) / **High** (genericity rule) | `training.rs:64,39,157`; `config.schema.json:417,462`                                    |

**One correctness-adjacent note:** `candidate_window` (k1), when set, _deliberately makes the policy inexact_
(`dcs.rs:264-269` "deliberately NOT exact") — a knob that silently trades away an exactness guarantee should say
so loudly in its user-facing description; today it just says "Ignored unless `method = dynamic`."

---

# Part 2 — Proposed redesign (recommended)

Mirror the `StoppingRuleConfig` tagged-enum that already works in the same file. Lift the two genuinely
always-on knobs to the parent; gate method-specific knobs inside a tagged `selection` block so wrong-method
fields become a **parse-time** `unknown field` error and `method` typos become an `unknown variant` error with
the valid set listed.

### Before / after — a `"dynamic"` config

```jsonc
// BEFORE — flat bag; every method's fields share one namespace; enabled separate from method
"cut_selection": {
  "enabled": true,
  "method": "dynamic",
  "start_iteration": 5,
  "active_window": 0,          // k2 seed window
  "candidate_window": 20,      // k1 recency; null = ∞
  "nadic": 3,                  // max added per round
  "violation_tolerance": 1e-9,
  "cut_activity_tolerance": 1e-6,
  "max_active_per_stage": 4000
  // threshold, memory_window, tie_tolerance, domination_epsilon,
  // check_frequency, basis_activity_window ALSO accepted here, silently ignored
}
```

```jsonc
// AFTER — always-on knobs at parent; method-specific knobs tagged; wrong-method field = parse error
"cut_selection": {
  "row_activity_tolerance": 1e-6,   // always-on (was cut_activity_tolerance)
  "max_active_per_stage": 4000,     // always-on
  "selection": {
    "method": "dynamic",
    "start_iteration": 5,
    "seed_window": 0,               // was active_window / k2
    "candidate_recency": 20,        // was candidate_window / k1; null = unbounded
    "max_added_per_round": 3,       // was nadic
    "violation_tolerance": 1e-9
  }
}
```

Disabling becomes structural: omit `selection` (or `"selection": null`). The `enabled=true, method=None` error
state disappears; `"method":"level1"` can no longer carry a dead `nadic`.

### Rename table

| Old                                                   | New / location                      | Why                                                                                |
| ----------------------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------- |
| `enabled: Option<bool>`                               | **removed**                         | Presence of `selection` _is_ enabled; deletes the runtime error path               |
| `method: Option<String>`                              | `selection.method` **as enum tag**  | `#[serde(tag="method")]` → `oneOf` + `const` in schema → typo caught at parse time |
| `nadic`                                               | `max_added_per_round` (Dynamic)     | states the effect; drops the n-adic jargon                                         |
| `active_window` (k2)                                  | `seed_window` (Dynamic)             | "active" collided with 2 other fields; `0` stays valid                             |
| `candidate_window` (k1)                               | `candidate_recency` (Dynamic)       | reads as a pair with `seed_window`; surface the `null=∞` + inexactness caveat      |
| `violation_tolerance`                                 | same (Dynamic)                      | **drop the `.or(tie_tolerance)` fallback** — default `1e-10` directly              |
| `tie_tolerance`                                       | same (Level1/Lml1)                  | now physically inaccessible from Dynamic                                           |
| `domination_epsilon`                                  | `domination_tolerance` (Domination) | one suffix; same math as `tie_tolerance`                                           |
| `check_frequency`                                     | same (Level1/Lml1/Domination)       | now only appears where used                                                        |
| `start_iteration`                                     | same (Dynamic)                      | Dynamic-only by construction                                                       |
| `cut_activity_tolerance`                              | `row_activity_tolerance` (parent)   | genericity scrub in `cobre-io`; serde `alias` for back-compat                      |
| `max_active_per_stage`                                | same (parent)                       | genuinely method-independent                                                       |
| `threshold`, `memory_window`, `basis_activity_window` | **removed**                         | dead; handled by the migration shim                                                |

### Rust structure — tagged enum (recommended over nested sub-structs)

```rust
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RowSelectionConfig {
    #[serde(default, alias = "cut_activity_tolerance")]
    pub row_activity_tolerance: Option<f64>,
    #[serde(default)]
    pub max_active_per_stage: Option<u32>,
    #[serde(default)]
    pub selection: Option<SelectionMethod>,   // None / null = disabled
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum SelectionMethod {
    Level1     { #[serde(default = "..")] check_frequency: u32, #[serde(default = "..")] tie_tolerance: f64 },
    Lml1       { #[serde(default = "..")] check_frequency: u32, #[serde(default = "..")] tie_tolerance: f64 },
    Domination { domination_tolerance: f64, #[serde(default = "..")] check_frequency: u32 },   // tol required
    Dynamic    { #[serde(default = "..")] start_iteration: u32,
                 #[serde(default = "..")] seed_window: u32,
                 #[serde(default)]        candidate_recency: Option<u32>,
                 #[serde(default = "..")] max_added_per_round: u32,
                 #[serde(default = "..")] violation_tolerance: f64 },
}
```

Why tagged enum, not `{ "dynamic": {...}, "level1": {...} }` nesting: it's the proven in-file pattern,
`schemars` renders it as a discriminated `oneOf` (editor autocomplete + typo detection), `deny_unknown_fields`
gates wrong-method fields for free, and per-variant `#[serde(default = "..")]` puts the authoritative defaults
**on the type** so `schemars` can publish them (fixes the "schema shows `default: null`, real default in a third
crate" drift). The `violation_tolerance ← tie_tolerance` coupling vanishes structurally.

### Migration / back-compat (the breaking-change decision)

`deny_unknown_fields` means any key rename or restructure hard-errors on every existing config. The flagship
consumer is ONS, whose production configs are long-lived. Three options:

| Option                                                                          | Gains                                                                                                                                               | Costs                                                                                                                                                |
| ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. Custom `Deserialize` dual-read shim + deprecation warnings (recommended)** | No flag day; old flat configs keep parsing with a precise per-key migration `warn!`; new configs get full schema validation; one deprecation window | Most code; the shim itself needs round-trip + reject-old tests (dual-owned-format discipline); custom-deserialize × `deny_unknown_fields` needs care |
| **B. Serde aliases only (no restructure)**                                      | Cheapest; renames `nadic→max_added_per_round` etc. in place                                                                                         | Does **not** fix the flat-bag / method-conditional / typo problems — the core UX defect survives                                                     |
| **C. Clean break with a clear error**                                           | Simplest final code                                                                                                                                 | Breaks every existing config on upgrade; needs an out-of-band migrator + major-version bump                                                          |

**Recommendation: A**, gated behind a minor-version bump, with shim removal (clean break) scheduled for the
following minor. Pair it with a `cobre schema validate config.json` subcommand — the `SchemaCommand` enum already
reserves room (`crates/cobre-cli/src/commands/schema.rs:26`) — so config mistakes surface at validate-time in
`cobre-io`, not at solver setup (genericity-safe, since the enum lives in `cobre-io`).

### The `row` vs `cut` genericity decision

The migration is half-done (type is "row", key/field/docs are "cut"). Recommended coherent end-state:

- **Inside `cobre-io` (types, fields, docs): all "row"** — the genericity hard rule binds identifiers and doc
  comments here. So `cut_activity_tolerance → row_activity_tolerance`, scrub "cut" from the struct's doc comments.
- **User-facing JSON key: keep `cut_selection`** (with a `row_selection` serde alias). "Cut selection" is the term
  every SDDP/ONS practitioner uses; the genericity rule binds _code identifiers and comments_, not the serialized
  string a domain user types. Add a one-line Voice-2 rationale comment recording this deliberate carve-out so a
  future reader doesn't "fix" the apparent mismatch.
- **`cobre-sddp` strategy layer stays "cut"** (`CutSelectionStrategy`) — it's the algorithm crate, not infra.

---

# Part 3 — This is symptomatic: workspace-wide config consistency

`cut_selection` is the worst offender but the same anti-patterns recur. The codebase already has the _correct_
templates (`StoppingRuleConfig`, `PolicyMode`, `InflowNonNegativityMethod`), so these are "apply the house
pattern" fixes, not new design.

| Gap                                                          | Severity               | Detail                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------ | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Stringly-typed enums `deny_unknown_fields` can't protect** | **High (correctness)** | `TrainingConfig::stopping_mode: String` (`training.rs:35`) — consumer at `setup/params.rs:105` does `eq_ignore_ascii_case("all") else Any`. A typo (`"al"`, `"and"`) **silently → `Any`**, changing when training stops. Model as `enum StoppingMode { Any, All }`. Same class: `method`, `LipschitzConfig::mode`.                         |
| **A whole config section is dead**                           | **High (honesty)**     | `upper_bound_evaluation` + nested `LipschitzConfig` (`training.rs:267-301`) are parsed, schema-documented, unit-tested — but **zero** reads in `cobre-sddp`. A user setting `upper_bound_evaluation.enabled:true` gets a silent no-op. Wire it, remove it, or give it a Voice-4 Intent/Seam justification.                                 |
| **`enabled` typed 3 ways**                                   | Med                    | `bool` in `Simulation`/`Training`, `Option<bool>` in `RowSelection`/`UpperBound`/`Checkpointing`. All mean "on/off, default off." Prefer concrete `bool` + `Default` (keeps the authoritative default in `cobre-io` where `schemars` publishes it); reserve `Option<T>` for "absence ≠ any value" (`max_active_per_stage: None` = no cap). |
| **Defaults invisible in schema**                             | Med                    | All `Option<T>` fields show `"default": null`; real defaults live in `cut_selection.rs` parse constants. The `enabled` schema default (`null`) even contradicts the documented `false`. Concrete+`Default` or per-variant `#[serde(default=..)]` fixes this.                                                                               |
| **Units/ranges documented unevenly**                         | Med                    | Good: `*_seconds`, `reference_volume_fraction` with `(0,1]` + runtime check. Weak: `io_channel_capacity` (no units, `0` silently → 64), the tolerance family. Add `#[schemars(range(..))]` + `validate_config` checks.                                                                                                                     |
| **Mandatory-but-`Option` fields**                            | Med (defensible)       | `forward_passes`/`stopping_rules` are `Option` only for friendlier errors. Keep the pattern but add one explaining comment so it isn't "simplified" away.                                                                                                                                                                                  |

---

# Part 4 — Competitor-mention remediation

**65 sites catalogued. Counts:** NEWAVE dominant, DECOMP ~9 (proper noun), DESSEM 1, CEPEL (design docs only),
plus NEWAVE artifact names (`selcor.dat`, `vazpast.dat`, `geracao_usinas_nao_simuladas`, `TENDENCIA HIDROLOGICA`).
Zero PSR/SUISHI/GEVAZP. (Note: the raw 76 "NEWAVE" / 263 "decomp" grep counts collapse once "decomposition"
false-positives and duplicate book/design clusters are removed.)

### Tier 1 — Scrub: rationale comments in shipped code (~16 sites)

These justify a default or behavior by naming NEWAVE and **ship into the binary / `config.schema.json`**.
Rephrase to domain-neutral terms that preserve the rationale. Representative sites + proposed replacements:

| File:line                                                     | Current                                                                              | Proposed neutral                                                                                                                 |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `cobre-io/src/config/energy.rs:18`                            | "Default 0.65 (NEWAVE convention)."                                                  | "Default 0.65 (conventional long-term reference-volume fraction)."                                                               |
| `cobre-io/src/config/training.rs:126-127` _(ships to schema)_ | "matching NEWAVE's `selcor.dat` `TAMANHO DA JANELA DE CORTES ATIVOS (k2)=0`."        | "`0` is valid and meaningful — it seeds only the current iteration's rows (no historical seeding)."                              |
| `cobre-sddp/src/cut/cut_selection.rs:2696`                    | "(valid, NEWAVE-style), and absent ⇒ default `k2=5`."                                | "(valid — seeds only the current iteration), and absent ⇒ default `5`."                                                          |
| `cobre-core/src/entities/non_controllable.rs:22,49,106`       | "mirrors NEWAVE's …", "Models NEWAVE's `geracao_usinas_nao_simuladas` (PCH, PCT, …)" | "non-curtailable must-run regime", "models non-simulated aggregate generation (small hydro, biomass, etc.) pre-netted from load" |
| `cobre-sddp/src/stochastic/noise.rs:571`                      | "must-run regime that mirrors NEWAVE's …"                                            | "must-run regime (non-curtailable aggregate generation)"                                                                         |
| `cobre-stochastic/src/sampling/historical.rs:49`              | "Per NEWAVE's `TENDENCIA HIDROLOGICA` …"                                             | "hydrological-tendency initial condition (recent inflow trend in the state vector)"                                              |
| `cobre-stochastic/src/sampling/mod.rs:191`                    | "analogous to NEWAVE's `vazpast.dat`"                                                | "past-inflow initial conditions"                                                                                                 |
| `cobre-stochastic/src/sampling/class_sampler.rs:177,747`      | "matching NEWAVE's `TENDENCIA HIDROLOGICA` convention"                               | "hydrological-tendency convention (recent-inflow trend)"                                                                         |
| `cobre-sddp/src/lp/builder/matrix.rs:168,747`                 | "matches NEWAVE behavior", "must-run, matching NEWAVE's …"                           | "single turbine column regardless of production model", "must-run (non-curtailable) level on every stage"                        |
| `cobre-sddp/src/setup/mod.rs:110`                             | "matching NEWAVE's `geracao_usinas_nao_simuladas` pre-netting"                       | "non-simulated aggregate generation pre-netted from load"                                                                        |
| `cobre-core/src/constraints/generic_constraint.rs:95`         | "Future CEPEL formulations"                                                          | "future alternative formulations"                                                                                                |

After editing `training.rs:126`, regenerate `config.schema.json` (`cobre schema export`) — the rustdoc is the
single source and the schema regenerates from it.

### Tier 2 — Keep (legitimate)

- **`cobre-bridge` interop** — `cobre-bridge convert newave` / `compare newave` (`book/src/guide/cobre-bridge.md:32,108`),
  `convert_newave_case` Python symbol (`:131`), the `inewave` dependency (`:184`). These are real external
  CLI/API/package names; renaming them would misdocument a working conversion tool. Keep. _(Open question below if
  you'd rather the main book not advertise NEWAVE-format conversion at all.)_
- **Bibliographic citations** — `de Matos (2015)`, `Guigues (2017/2019)`, `Diniz et al. (CEPEL, 2020)`. `Author(YYYY)`
  is the explicit keep-carve-out in `.claude/rules/comments.md`. Keep in-tree, or relocate the methodology framing
  to the `cobre-docs` repo (the declared source-of-truth root).

### Tier 3 — Owner judgment

- **CHANGELOG history** (~14 sites, e.g. `CHANGELOG.md:419,1566,1591`). Ironically one entry already records
  "Removed explicit references to external software." History may legitimately name past artifacts; recommend
  neutralizing _new_ entries going forward and lightly neutralizing the load-bearing ones, leaving pure history.
- **Design docs** (`docs/design/dev-strategy.md`, `fpha-tailrace-modeling.md`, `dynamic-cut-selection-design.md`).
  `dev-strategy.md` _positions_ cobre against NEWAVE/DECOMP/DESSEM — competitor names are intrinsic to that doc's
  purpose. These are internal (not shipped product). Recommend: keep as internal scratch, or relocate competitive
  positioning + methodology citations to `cobre-docs`. Lowest priority.

---

# Part 5 — Decisions needed & sequencing

**Decisions required before I touch code:**

1. **Competitor scrub scope** — Tier 1 only (shipped code + schema, low-risk, clearly desired), or also Tier 3
   (CHANGELOG/design-doc sweep)?
2. **`cobre-bridge` in the book** — keep documenting NEWAVE-format conversion (Tier 2 keep), or remove that page
   from the main book and point to the `cobre-bridge` repo instead?
3. **Schema redesign** — proceed with the tagged-enum redesign (Part 2)? If yes, confirm migration strategy
   **A (shim, recommended)** vs **C (clean break + major bump)**.

**Recommended sequencing:**

1. **Tier-1 competitor scrub + schema regen** (½ day, no behavior change, no breaking change). Do this first — it's
   independent of the redesign and clearly in scope.
2. **Low-risk consistency fixes** (1 day): warn on the two silent deprecated fields; enumerate valid methods in the
   `method` error; unify `enabled: bool`; add the `cut_selection.method` typo to `validate_config` (or move it into
   a `cobre-io` enum). Each has an in-repo exemplar.
3. **`stopping_mode` → enum + dead `upper_bound_evaluation` decision** (½ day) — the two correctness/honesty gaps
   from Part 3.
4. **The `cut_selection` redesign** (Part 2) — the big one; needs the migration-strategy decision and a version bump.
   Build the shim with round-trip + reject-old tests, regenerate the schema, update `configuration.md`.
