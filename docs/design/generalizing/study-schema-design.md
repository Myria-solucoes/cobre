# Study & boundary-condition schema — working design

**Started:** 2026-07-23 · **Status:** working draft under active refinement
(pre-plan). Companion to `beyond-sddp-generalization.md` (III.3, IV.4, V.1,
V.9, D15). Every "exists today" claim below was verified against `develop` at
v0.12.0 on the start date.

## 1. Constraints the schema must satisfy (fixed)

- **Absent `study` block ⇒ exactly today's behavior** (SDDP train + simulate;
  every existing config valid unedited) — the V.9 stable spine.
- **Strict rejection**: every new block is `deny_unknown_fields`; retired keys
  hard-fail (the house regime since the unknown-key sweep).
- **Selection is data** (D9): the engine is chosen at run time; an illegal
  (study × engine × backend) combination is rejected by a typed admission gate
  with a named error. Live in-tree pattern to generalize: the CLP backend's
  named rejection of per-phase solver-profile overrides at setup.
- **Schema-exported and `$schema`-stamped**: new blocks enter
  `config.schema.json`; `cobre init` pins the versioned URL.
- **Broadcast-aware** (D14): rank 0 parses; the engine tag must broadcast
  before the heavy payloads so non-root ranks can skip setup entirely for
  rank-0-only engines. Postcard note: the MPI snapshot cannot reuse
  internally-tagged serde enums — broadcast mirror types (the
  `BroadcastBackwardScheduler` pattern) are the house workaround.
- **Presets select structure, never tuning numbers.** The named solver-profile
  presets were removed by owner decision in favor of explicit per-field
  overrides; a _study_ preset survives that precedent only because it selects
  problem structure (template + engine + output family), carries no numeric
  tuning, and its expansion is documented and inspectable.

## 2. The `study` block (new top-level `config.json` section)

```jsonc
{
  "$schema": "…/config.schema.json",
  "study": {
    "preset": "economic_dispatch", // closed enum; see below
    "boundary": {
      "kind": "target_storage", // §3 — the D15 axis
      "targets": [{ "hydro_id": 0, "value_hm3": 83.222 }],
      "violation_cost": 5000.0,
    },
    "solver": {
      // §4 — Direct single-solve profile
      "dual_edge_weight": "steepest_edge",
    }, //     (optional, all fields optional)
  },
}
```

- `preset` — closed enum, snake_case per house style: `operation_planning`
  (the implicit default when the block is absent) and `economic_dispatch`.
  The enum grows per phase (OPF, UC presets later); it never carries unbuilt
  names.
- **No `engine` field in Phase 0a.** Each 0a preset has exactly one legal
  engine (`operation_planning → sddp`, `economic_dispatch → direct`), so an
  override would be dead config. D9's "defaulted by the preset, overridable
  in config" activates when the first problem with two legal engines arrives
  (expansion: monolithic-direct vs composed, Phase 4) — the field is additive
  then. _(Open question 3, §7.)_
- Expansion is a pure data mapping (preset → template features + engine +
  output family), deterministic, with the mapping table in the config
  reference docs.

## 3. The boundary-condition axis (D15) — concrete schema

Internally-tagged on `"kind"` (house tagged-enum style: `type` /
`method` / `scheme` precedents):

```jsonc
// Reach a stated storage at end of horizon (soft, penalized shortfall):
"boundary": {
  "kind": "target_storage",
  "targets": [ { "hydro_id": 0, "value_hm3": 83.222 } ],
  "violation_cost": 5000.0                  // $/hm³ below target, global
}

// Explicit myopic run (declared, warned at setup, recorded in run summary):
"boundary": { "kind": "zero_value" }

// RESERVED — documented but rejected by every 0a engine; unified with
// SDDP's `policy.boundary` at the Phase-1 relocation (see below):
"boundary": { "kind": "value_function", "path": "…", "source_stage": 11 }
```

Design points, in order of load-bearing-ness:

- **Soft, never hard-equality.** A hard terminal target is an infeasibility
  trap (inflows may be insufficient). Shortfall below target is penalized at
  `violation_cost`; storage above target is free. A hard kind can be added
  later if pulled.
- **The lowering seam exists today.** `GenericConstraint` already carries
  `SlackConfig { enabled, penalty }` (soft violation at a cost) and the
  variable catalog already has `HydroStorageFinal` — `target_storage` lowers
  onto the same penalized-slack row machinery. New schema, existing math.
- **Units mirror `initial_conditions.json`**: entries are
  `{ "hydro_id", "value_hm3" }`, id-sorted canonical order, duplicate or
  unknown `hydro_id` rejected at load. A hydro without a target is
  unconstrained (documented).
- **Examination finding (closes the D15 "examine first" instruction):**
  `Hydro.filling_target_violation_cost` is **commissioning** vocabulary — the
  filling target is the _dead volume_ a newly-built reservoir fills before
  operating (`FillingConfig`, `commissioning.rs`). Semantically distinct from
  an end-of-horizon target: **do not reuse**; `target_storage` gets its own
  fields as above. The reserved commissioning penalties stay untouched.
- **`value_function` unification is a Phase-1 relocation, not a 0a rename.**
  SDDP's existing `policy.boundary { path, source_stage }` _is_ the
  `value_function` kind in today's spelling. 0a leaves it where it is
  (existing configs valid unedited); the Phase-1 field-relocation map moves it
  to `study.boundary { kind: value_function }` under the v1 shim. Until then
  the gate rejects `kind: value_function` with an error naming
  `policy.boundary` as the current spelling — one concept, one live spelling
  at a time.
- **Required under `economic_dispatch`** _(recommendation — open question 2)_:
  a missing `boundary` under the ED preset is a named admission error, not a
  silent `zero_value` default — "myopic by declaration, never by accident"
  (D15). `operation_planning` takes no `study.boundary` in 0a (SDDP builds
  its own terminal value; `policy.boundary` covers injection).

## 4. Per-engine solver profiles

- Engines declare their profile-slot set: `sddp → {training.solver.backward,
training.solver.forward, simulation.solver}` (existing spellings untouched);
  `direct → study.solver` (one single-solve profile; same
  `PhaseSolverProfileConfig` field set — the 12 fields are phase-agnostic).
- Admission is symmetric and named: `training.solver.*` under `direct` is
  rejected ("engine `direct` has no backward/forward phases"); `study.solver`
  under `sddp` is rejected ("engine `sddp` takes per-phase profiles"). Exactly
  the CLP-rejection error shape.
- Phase-1 direction (not 0a): when SDDP config relocates into the engine's own
  section (III.7), profiles become engine-owned maps keyed by engine-declared
  phase names; the 0a shape above is forward-compatible with that.

## 5. Engine-scoped section admissibility

`training` is structurally **required** today (`Config.training` has no
`#[serde(default)]`). Under 0a it becomes structurally optional, with
admissibility enforced per engine at validation:

| Section                  | `operation_planning` (sddp)   | `economic_dispatch` (direct)           |
| ------------------------ | ----------------------------- | -------------------------------------- |
| `training`               | **required** (as today)       | **rejected** (named error)             |
| `simulation`             | consumed (as today)           | **rejected**                           |
| `upper_bound_evaluation` | consumed                      | **rejected**                           |
| `estimation`             | consumed                      | **rejected** *(pending Q1 — a mean     |
|                          |                               | path may need fitted models)*          |
| `policy`                 | consumed (incl. `boundary`)   | **rejected** (no policy directory)     |
| `modeling`               | consumed                      | **consumed** (`cost_scale_factor`,     |
|                          |                               | inflow non-negativity apply to any LP) |
| `exports`                | consumed                      | consumed subset (flags without an ED   |
|                          |                               | output are inert-but-valid — TBD)      |
| `study`                  | optional (defaults preserved) | required entry point                   |

Rejected-not-ignored keeps the strict-config regime coherent: switching a case
from SDDP to ED is an explicit config edit, and every error names the section
and the engine. A config that is invalid today (missing `training`) stays
invalid — only the error's shape moves from parse to admission.

## 6. Broadcast & MPI shape

- A tiny fixed-size **engine header** broadcasts first. `direct` ⇒ non-roots
  skip case broadcast, stochastic reconstruction
  (`reconstruct_stochastic_context_non_root`,
  `rebuild_historical_library_non_root`), and all setup; they join only the
  final barrier (D14). `sddp` ⇒ today's path, unchanged bytes.
- `BroadcastConfig` (18 fields, postcard) remains SDDP-only and is simply not
  built for `direct`. New broadcast-crossing enums get mirror types (the
  postcard tagged-enum trap).
- The run summary writes `ranks_participated` (existing manifest field) = 1
  for `direct` under `mpirun -n > 1`, plus the idle-rank warning.

## 7. Open questions (owner input wanted)

1. **ED deterministic inputs — the biggest modeling call.** Demand is
   `LoadModel.mean_mw` (std = 0 annihilation verified in 0a). Hydro inflows
   need a deterministic source; candidates, using the existing per-quantity
   `SamplingScheme` vocabulary (`in_sample | out_of_sample | external |
historical`, per inflow/load/ncs):
   (a) **zero-noise conditional mean** — the PAR(p) trajectory from the
   case's initial lags with noise = 0; the literal SDDP forward degenerate;
   needs a small new scheme or an ED-internal path _(recommended)_;
   (b) **external with exactly 1 scenario** — no new vocabulary, but demands
   the user supply the series;
   (c) **historical window** — replay a chosen year.
   The choice decides whether `estimation`/fitted models are consumed by ED
   (§5 table).
2. **`boundary` required vs defaulted under `economic_dispatch`** —
   recommendation: required (§3); alternative: default `zero_value` + warning.
3. **Omit `study.engine` in 0a** — recommendation: omit (dead config until a
   two-engine problem exists); alternative: ship it now to match D9's letter.
4. **`exports` under ED** — inert-but-valid flags vs rejected unknown-output
   flags (§5 table, last row).

## 8. Worked examples

Existing v1 config, untouched, still the operation-planning study — valid
with no `study` block (the stable spine). Minimal ED config:

```jsonc
{
  "$schema": "…/config.schema.json",
  "study": {
    "preset": "economic_dispatch",
    "boundary": {
      "kind": "target_storage",
      "targets": [{ "hydro_id": 0, "value_hm3": 83.222 }],
      "violation_cost": 5000.0,
    },
  },
  "modeling": { "inflow_non_negativity": { "method": "none" } },
}
```

## 9. Reconciliation with the DECOMP program (2026-07-23)

Binding inputs from `~/git/cobre-bridge/plans/` (see
`decomp-program-reconciliation.md`, including the `bridge-D<n>` vs roadmap
`D<n>` namespace warning). Impact on this design:

- **The `study` block is structurally unaffected.** A DECOMP-like study is
  the `operation_planning` preset on the SDDP engine; everything
  DECOMP-specific is SDDP-engine-owned config — exactly the ownership
  boundary this design draws.
- **New engine-owned config surfaces are incoming** (Rung 1 / bridge-D8, all
  default-path byte-neutral): `scenario_source.*.openings =
"generated" | "external"`, `scenario_source.selection =
"hash" | "enumerate"`, `scenarios/scenario_probabilities.parquet`
  `{stage_id, scenario_id, probability}`, `state_space.inflow_lag_depth`,
  and `stopping_rules: [{ "type": "gap", "tolerance": … }]` — the gap rule
  admission-rejected under sampled forwards, one more customer of the §2/§4
  gate machinery.
- **The boundary axis is validated from outside**: the FCF importer authors
  synthetic checkpoints consumed via `policy.boundary`, so keeping the SDDP
  spelling live until Phase 1 (§3) was correct. The Phase-1 unification into
  `study.boundary { kind: value_function }` must carry (i) the
  manifest/state-dimension validation semantics unchanged and (ii) a source
  selector that generalizes stage → node (Rung 2).
- **Output-schema growth is scheduled** (simulation probabilities,
  `node_id`, `unit_group_id` columns) — a further argument for landing the
  shared output-orchestration entry point early in 0a, so each lands as a
  one-list change mirrored in Python by construction.
- **The stable-spine baseline moves at v0.13.0** (2026-07-24): the windowed
  inflow epic breaks case inputs (`initial_conditions.past_inflows` removed;
  `inflow_history` re-laid as dated windows — see
  `decomp-program-reconciliation.md` §7). The §1 constraint's mechanism is
  untouched (it governs the `study` block in `config.json`), but "every
  existing config valid unedited" reads against the current major input
  format: pre-0.13 cases migrate their inflow inputs first.
