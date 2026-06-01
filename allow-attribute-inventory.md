# `#[allow(...)]` Inventory & Remediation Plan

Date: 2026-05-31
Scope: all production source under `crates/*/src` (excludes `tests/`, `benches/`)
Purpose: catalog every production lint-suppression so we can plan how to retire
or justify each, rather than land a blanket CI gate that breaks the build.

## Totals (production `crates/*/src`)

| Bucket                                                                                                         | Count | Track           | Disposition                                                                                                                          |
| -------------------------------------------------------------------------------------------------------------- | ----: | --------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `clippy::cast_*` (truncation/wrap/precision/sign)                                                              |   641 | Numeric hygiene | Separate track — pragmatic in HPC numeric code; not over-engineering. Leave for now; optionally tighten later via typed conversions. |
| `unwrap_used` / `expect_used` / `panic` / `float_cmp`                                                          |   415 | Test relaxation | Mostly inline `#[cfg(test)] mod tests` + `#![cfg_attr(test, allow(...))]` in `lib.rs`. Expected pattern — **not** production debt.   |
| `clippy::too_many_lines`                                                                                       |    79 | **Actionable**  | Plan below (§1).                                                                                                                     |
| `dead_code`                                                                                                    |    28 | **Actionable**  | Plan below (§2) — highest priority; masks genuinely-unused items.                                                                    |
| `needless_pass_by_value`                                                                                       |    27 | Minor hygiene   | Audit opportunistically; low value.                                                                                                  |
| `too_many_arguments`                                                                                           |    22 | Minor hygiene   | Often signals a missing params struct; address when touching the fn.                                                                 |
| `struct_field_names`                                                                                           |    13 | Cosmetic        | Leave; renaming churns the data model for no behavioral gain.                                                                        |
| `deprecated`                                                                                                   |    14 | **Actionable**  | Investigate — internal deprecations should be removed, not suppressed (§3).                                                          |
| `type_complexity` (6), `struct_excessive_bools` (2), `unnecessary_wraps` (2), `unused_imports` (2), misc (≈10) |   ~22 | Minor           | Case-by-case when touching the code.                                                                                                 |

Only **2 of 79** `too_many_lines` and **0 of 28** `dead_code` carry a trailing
justification comment. That ratio is the core problem: the suppressions are
silent, so they can't be distinguished from rot during review.

---

## §1 — `too_many_lines` (79 occurrences)

Concentrated in the SDDP hot path; the long tail is one-per-file boilerplate.

| File                                                                                                                                                                                                                                                       |                         Count | Character                  | Recommended disposition                                                                                                                                                       |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------: | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cobre-sddp/src/backward.rs`                                                                                                                                                                                                                               |                            15 | Hot path                   | **Keep, but justify each.** Splitting hot-path solve loops risks readability/locality. Add a one-line rationale comment per allow; revisit only the outliers.                 |
| `cobre-sddp/src/forward.rs`                                                                                                                                                                                                                                |                            12 | Hot path                   | Same as backward.rs.                                                                                                                                                          |
| `cobre-sddp/src/noise.rs`                                                                                                                                                                                                                                  |                             3 | Stochastic                 | Review for extraction; likely justifiable.                                                                                                                                    |
| `cobre-io/src/output/dictionary.rs`                                                                                                                                                                                                                        |                             3 | Serialization              | "N entity types × M lines" pattern — keep + comment.                                                                                                                          |
| `cobre-io` writers/validators/system loaders                                                                                                                                                                                                               | ~22 (1 each across ~22 files) | Serialization / validation | Inherent breadth (per-entity-type match arms). **Keep + add the `// N entity types × ~M lines each` justification** (the pattern already used at `simulation_writer.rs:531`). |
| `cobre-sddp/src/setup/mod.rs`                                                                                                                                                                                                                              |                             1 | **ARCH-03**                | The 513-line `from_broadcast_params` god-constructor. **Decompose** (tracked as ARCH-03) and remove the allow.                                                                |
| `cobre-core/src/system.rs:864`, `estimation.rs` (2), `hydro_models.rs` (2), `lp_builder/layout.rs`, `indexer.rs`, `lower_bound.rs`, `simulation/{pipeline,extraction}.rs` (2 each), `cobre-cli/commands/run.rs` (2), `cobre-python/{run,results}.rs`, etc. |                           ~30 | Mixed                      | Review individually; most are domain-wide match/dispatch (keep + comment), a few may be decomposable.                                                                         |

**Only 2 currently justified:** `simulation_writer.rs:531`, `baking.rs:55`.

**§1 plan:** (a) ARCH-03 removes the one true god-constructor allow. (b) For the
rest, adopt a **"justify-or-remove"** rule: every retained `too_many_lines` gets a
trailing `// <reason>` comment. This is mechanical, reviewable, and converts the
79 silent suppressions into 79 explicit decisions — at which point a CI gate
requiring the comment becomes enforceable without a mass refactor.

---

## §2 — `dead_code` (28 occurrences) — highest priority

These mask genuinely-unused items and directly defeat the workspace dead-code
lint. Classified by disposition:

### 2a — Remove (true dead/vestigial; includes findings ARCH-05/06)

| Location                                          | Item                                                                      | Note                                                                        |
| ------------------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `cobre-sddp/src/setup/mod.rs:194`                 | `resolved_parameters` field                                               | **ARCH-05** — stored, never read; demote to construction-local.             |
| `cobre-sddp/src/stage_solve.rs:60`                | `StageInputs` struct                                                      | **ARCH-06** — 3 unread fields; remove with the Phase/StageOutcome collapse. |
| `cobre-sddp/src/stage_solve.rs:108`               | `StageOutcome` enum                                                       | **ARCH-06** — single-valued-per-site; collapse.                             |
| `cobre-io/src/output/policy/codec.rs:400`         | `read_f32_vector_as_f64`                                                  | Verify no caller; remove if dead.                                           |
| `cobre-sddp/src/fpha_fitting.rs:679,750,1217`     | `evaluate_losses_factor`, `hydro_name`, `compute_max_approximation_error` | Verify; remove or wire up.                                                  |
| `cobre-stochastic/src/tree/qmc_halton/mod.rs:137` | `radical_inverse`                                                         | Verify; remove if superseded by Sobol path.                                 |
| `cobre-io/src/stages.rs:264`                      | `Expectation(String)` variant                                             | Verify; unused enum variant.                                                |

### 2b — Keep but convert to scoped form (legitimate, mis-suppressed)

| Location                                               | Item                             | Better form                                                                                              |
| ------------------------------------------------------ | -------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `cobre-comm/src/factory.rs:261`                        | `mpi_launch_detected`            | Already `#[cfg_attr(not(feature="mpi"), allow(dead_code))]` — **correct**; this is the model to imitate. |
| `cobre-comm/src/ferrompi.rs:60`                        | `mpi` field                      | Feature-conditional; scope to `cfg(not(feature=...))` if truly only used under a feature.                |
| `cobre-comm/src/factory.rs:99`                         | `_assert_comm_backend_send_sync` | Compile-time assertion fn — rename pattern or `#[cfg(test)]`-gate.                                       |
| `cobre-io/src/validation/semantic/test_support.rs:809` | `_assert_helpers_present`        | Test-support shim — gate under `cfg(test)`.                                                              |

### 2c — Investigate (likely "reserved for upcoming integration" — the ARCH-05 smell)

`validation/schema.rs:77,114,157` (penalties / scalar_parameters / exchange_factors
fields), `workspace.rs:581` (anticipated-state buffer), `lp_builder/layout.rs:131,183,288`
(anticipated-state counts/rows), `training_session/{rank_distribution.rs:20,runtime.rs:25}`
(my_rank / export_states), `cobre-python/src/study.rs:83,87,91` (3 summary fields),
`policy_load.rs:35` (`resolve_warm_start_counts`), `state_exchange.rs:140` (`ExchangeBuffers` impl).

→ For each: either there is a near-term reader (add a tracking note + keep) or
there isn't (remove). Same defect class as ARCH-05 — **don't let "retained for
future X" justify a silent `dead_code`.**

**§2 plan:** triage all 28 in one pass. ARCH-05/06 are already scheduled. The 2b
set converts to `cfg`-scoped allows (the `factory.rs:261` model). The 2c set is
the real audit — each is a yes/no on "does a reader land this quarter?"

---

## §3 — `deprecated` (14 occurrences)

Internal `#[allow(deprecated)]` usually means we're calling our own deprecated
API. Either the deprecation is premature (un-deprecate) or real (migrate callers
and delete). **Action:** enumerate the 14, identify whether the deprecated target
is internal or external (e.g. a dependency). Internal ones should be retired.

---

## Recommended sequencing

1. **§2 dead_code triage (28)** — highest signal, includes ARCH-05/06; mostly
   removals + a few `cfg`-scoping. Do first.
2. **ARCH-03** — decompose `from_broadcast_params`, removes its `too_many_lines`.
3. **§1 justify-or-remove sweep** — add rationale comments to the ~22 cobre-io
   serialization allows and the hot-path forward/backward allows; decompose only
   the few genuine outliers.
4. **§3 deprecated audit (14)**.
5. **Then** introduce the CI gate, in its final enforceable shape:
   - **Forbid bare `#[allow(dead_code)]`** outside `#[cfg(test)]` (allow the
     `cfg_attr(not(feature=...))` scoped form).
   - **Require a trailing `// reason` on every `too_many_lines`** (not a ban).
   - Leave `cast_*` and the test-relaxation lints out of the gate.

The gate goes in **last**, once the tree already satisfies it — so it locks in a
clean state instead of failing CI on day one.

---

## Out of scope (deliberately)

- **641 `cast_*` allows** — numeric-conversion lints, ubiquitous and pragmatic in
  HPC code. A separate, lower-priority hygiene track if desired (typed conversion
  helpers), not part of this plan.
- **415 test-relaxation allows** — the sanctioned `#![cfg_attr(test, allow(...))]`
  pattern; not production debt.
- **The large files themselves** (`estimation.rs` 6123, `par/fitting.rs` 7039,
  `indexer.rs` 5309, lp_builder/cut/simulation decompositions) — the round-2/3
  adversarial sweep found these are large because the domain is large, not
  over-abstracted. Do **not** split them for line-count's sake.
