# Architecture Audit — cobre-io and cobre-sddp

**Date**: 2026-05-18
**Scope**: deep scrutiny of IO and SDDP crates after ~50 commits of feature accretion (scalar parameters, energy variables, productivity decoupling, PAR(p)-A NEWAVE-parity, productivity resolution refactor).
**Method**: read-only static analysis combining mechanical sweeps and three parallel adversarial-attacker agent audits (cobre-io smells, cobre-sddp smells, cross-crate boundaries + Python parity).

---

## Headlines

- **3 Critical issues** including one outright correctness bug (`validate_scalar_parameters` is exported as a public validator but **never invoked by the load pipeline**) and one hot-path allocation that bypasses the project's "never allocate on hot paths" hard rule.
- **The productivity refactor (commits ff96e8a → 755592c) is a symptom**, not a root cause. The underlying disease is that `load_case` returns only `System`, so any artifact `System` does not carry — productivity overrides, scalar parameters, manifests, FPHA rows, geometry rows — is **re-parsed downstream** with weaker (or no) validation. Closing this gap eliminates four findings at once.
- The codebase is otherwise in **strong shape**: zero algorithm-name leaks in infrastructure crates, zero production `.unwrap()`, no `Box<dyn Trait>` in production paths, almost no `TODO/FIXME` debt (1 instance found), CLI ↔ Python file parity intact (every output written by one is written by the other).

---

## CRITICAL

### CR-1 — `validate_scalar_parameters` is dead code in production (correctness bug)

**Where**: `crates/cobre-io/src/lib.rs:129` exports it; `crates/cobre-io/src/validation/scalar_parameters.rs:55` defines it; `crates/cobre-io/src/pipeline.rs:67-99` never calls it. Only consumers are its own tests.

**Impact**: A `system/scalar_parameters.json` with a dangling `hydro_id`, wrong `per_stage` length, or duplicate names passes `load_case` and explodes inside `build_resolved_parameters` (or worse, silently produces wrong values).

**Fix**: Add a call inside `pipeline::run_pipeline_with_report` after Layer 5, threading the assembled `ParsedData.scalar_parameters`. Two lines.

### CR-2 — Hot-path allocation in lower-bound evaluation

**Where**: `crates/cobre-sddp/src/lower_bound.rs:472`

```rust
let mut lb = risk_measure.evaluate_risk(objectives, &vec![uniform_prob; objectives.len()])
```

**Impact**: Called once per training iteration; `vec![…]` allocates a fresh `Vec<f64>` per call. Hard-rule violation ("never allocate on hot paths"). CI clippy will not flag this.

**Fix**: Add a `uniform_prob: Vec<f64>` field to `LbEvalScratchBundle`, fill once via `resize`, reuse. Trivial.

### CR-3 — Data is parsed twice on every run; the validated copy is discarded

**Where**:

- `cobre-io` parses `hydro_energy_productivity.parquet`, `hydro_production_models.json`, `fpha_hyperplanes.parquet`, `hydro_geometry.parquet`, `scalar_parameters.json` into `ParsedData` (`crates/cobre-io/src/validation/schema.rs:107, 362-365, 700-714`).
- `ParsedData` is consumed only by `validate_productivity_resolution` and then dropped — **`pipeline.rs:182-216` never threads any of these into `SystemBuilder`**.
- `cobre-sddp` re-opens the same files from disk: `crates/cobre-sddp/src/hydro_models.rs:727, 737, 843, 864, 1422`; CLI re-opens `scalar_parameters.json` at `crates/cobre-cli/src/commands/run.rs:135-141`; Python re-opens it at `crates/cobre-python/src/run.rs:535-545`.

**Impact**: This is the architectural fault line that produced the entire productivity-refactor commit chain. The validated `ParsedData` is the single source of truth — but it never reaches the consumer. Every productivity rule had to be re-thought in two places, with `ConstructionConfig.scalar_parameters` becoming a workaround field. CR-1 is also a symptom of this gap.

**Fix**: Extend the return type of `load_case` (or add a sibling `load_case_with_artifacts`) so that `ParsedData`'s already-parsed-and-validated rows flow into `cobre-sddp::prepare_hydro_models` and into `StudySetup::from_broadcast_params` as data, not as filesystem paths. This closes CR-1, CR-3, and HI-3 in one pass.

---

## HIGH

### HI-1 — Public API over-exposure: ~35 `pub mod` declarations should be `pub(crate)`

**Where**: `crates/cobre-sddp/src/lib.rs:20-63` declares 37 modules `pub`; external consumers (cobre-cli, cobre-python) reach into only 2 of them by qualified path (`policy_export`, `setup`). The other 35 are used only through the curated re-exports at the top of `lib.rs`. Same pattern in `crates/cobre-io/src/output/mod.rs:11-23` — sub-modules are `pub` _and_ re-exported, and downstream crates use both spellings inconsistently (e.g., `crates/cobre-sddp/src/conversion.rs:3` imports `cobre_io::output::simulation_writer::...`; `crates/cobre-cli/src/commands/run.rs:27` imports `cobre_io::output::...`).

**Impact**: The semver API surface includes thousands of identifiers nobody intended to expose. Any internal restructuring of `forward.rs`, `backward.rs`, `workspace.rs`, `cut_sync.rs`, `lp_builder/`, `setup/scenario_libraries`, `output/simulation_writer`, etc. is technically a breaking change.

**Fix**: Single sweep: change `pub mod` → `pub(crate) mod` in both crates' `lib.rs`, except for the two genuinely-external modules (`policy_export`, `setup`). Add a brief `lib.rs` doc stating the contract. Compile, then promote items that no longer reach where they're needed. Likely 5-15 items to promote individually.

### HI-2 — CLI ↔ Python orchestration duplication: `write_policy_checkpoint` + `export_stochastic_artifacts`

**Where**:

- `crates/cobre-cli/src/policy_io.rs:1-138` and `crates/cobre-python/src/run.rs:78-139` — near-line-for-line duplicate of `write_policy_checkpoint`.
- `crates/cobre-cli/src/commands/run.rs:1820-1944` (125 lines) and `crates/cobre-python/src/run.rs:141-243` (88 lines) — `export_stochastic_artifacts` duplicated.
- `crates/cobre-cli/src/commands/run.rs:1681-1772` and `crates/cobre-python/src/run.rs:327-408` — `write_training_outputs`/`write_training_artifacts`; FPHA write is _inside_ the bundle on the CLI path but _outside_ on the Python path (line 699). A latent ordering asymmetry that will bite a future maintainer.

**Impact**: Python parity is "non-negotiable" per CLAUDE.md; this duplication is the most likely source of silent drift in the next 12 months.

**Fix**: Lift both functions into `cobre-sddp` (a new `cobre_sddp::orchestration` module). CLI and Python both call it. MD-13 (the implicit `TrainingOutput` ownership split between cobre-io and cobre-sddp) is the same theme; sharing one constructor closes both.

### HI-3 — `extensions::scalar_parameters` and `validation::scalar_parameters` are an island

**Where**: parsed at `crates/cobre-io/src/validation/schema.rs:571` for sigil resolution only; the assembled `Vec<ScalarParameter>` is stored in `ParsedData.scalar_parameters` (`schema.rs:115`) but **not forwarded into `SystemBuilder`** (`pipeline.rs:182-216`). CLI re-reads from disk at `crates/cobre-cli/src/commands/run.rs:135-141`; Python re-reads at `crates/cobre-python/src/run.rs:535-545`. Combined with CR-1, the production validators are bypassed entirely.

**Impact**: Same architectural fault as CR-3, with the added correctness consequence of CR-1.

**Fix**: Resolved jointly by CR-3's resolution.

### HI-4 — `validation/semantic/mod.rs`: incomplete decomposition (4593 lines, ~120 of which are production)

**Where**: `crates/cobre-io/src/validation/semantic/mod.rs:1-145` has two 9-line dispatchers; `mod tests` at line 145 contains **99 tests** that exercise sibling modules (`hydro.rs`, `thermal.rs`, `scenarios.rs`, `season.rs`, `correlation.rs`, `sobol.rs`, `stages.rs`). The siblings contain zero tests.

**Impact**: A developer modifying `semantic/scenarios.rs` has to find that file's tests 2,500 lines away in `mod.rs`. The decomposition that motivated the original commit is half-done. `crates/cobre-io/src/validation/semantic/shared.rs` is also literally a 6-line placeholder file ("Reserved for cross-domain semantic helpers ... a placeholder").

**Fix**: Move each test next to the module it exercises (mechanical refactor; tests follow shared `make_data` helper to `semantic/test_support.rs` or similar). Delete `shared.rs` until something cross-domain actually shows up.

### HI-5 — Per-stage `Vec::new()` allocations in the backward / forward sweeps

**Where**:

- `crates/cobre-sddp/src/backward_pass_state.rs:389` — `let mut stage_stats: Vec<...> = Vec::new();` grown per backward stage.
- `crates/cobre-sddp/src/backward_pass_state.rs:646` — `gather_stage_solver_stats` returns a fresh `Vec` by value per stage per iteration.
- `crates/cobre-sddp/src/forward_pass_state.rs:799` — `per_stage_stats.to_vec()` per worker per forward pass.

**Impact**: With ~12 stages and 100s of iterations, ~hundreds of thousands of heap allocations per training run. Same class as CR-2 but smaller per-call cost.

**Fix**: Move `stage_stats` and the gather buffer onto `BackwardPassState` (clear-and-reuse). Make `per_stage_stats` a `Vec` on the worker context so `mem::take` replaces `to_vec()` (the comment block at `forward_pass_state.rs:568-577` already explains this pattern for `stage_stats` and `scenario_costs` — extend it).

### HI-6 — Massive inline test bloat in two files (`lp_builder/template.rs`, `simulation/pipeline.rs`)

**Where**:

- `crates/cobre-sddp/src/lp_builder/template.rs`: 655 production lines, **9061 inline test lines** (14:1 ratio; production exposes 1 struct + 1 fn).
- `crates/cobre-sddp/src/simulation/pipeline.rs`: 953 production lines, ~3964 inline test lines.

**Impact**: Every change to template.rs / pipeline.rs triggers LLVM to recompile ~9000 / ~4000 lines of tests. Tests that are integration-style (system-builder, MockSolver, StubComm) belong in `crates/cobre-sddp/tests/` (which already houses 27 integration test files, so the convention exists).

**Fix**: Move integration-style tests to `tests/`. Keep only narrow unit tests of `build_stage_templates` / `simulate` shims inline. Expect significant CI compile-time win.

### HI-7 — Layer 6 hidden behind a "five-layer" documented pipeline

**Where**: `crates/cobre-io/src/validation/productivity_resolution.rs:1` self-identifies as "Layer 6"; `crates/cobre-io/src/pipeline.rs:93-94` calls it after the documented five layers; `crates/cobre-io/src/lib.rs:15-24` and `pipeline.rs:1-7` document only five layers.

**Impact**: Doc drift. The next single-field cross-file rule (CR-1 included) will become "Layer 7". The check `productivity_resolution` performs is structurally identical to Layer 4 (dimensional coverage) and Layer 5 semantic rules.

**Fix**: Either rename to a real "Layer 6 — cross-file resolution" and document it (will absorb the scalar-parameter validator once wired), or fold the check into `dimensional.rs` + `semantic/hydro.rs`.

### HI-8 — Stub implementations sitting in production code

**Where**:

- `crates/cobre-sddp/src/cut_selection.rs:185-198, 683-694` — `CutSelectionStrategy::Dominated::select` always returns the empty set; the `threshold` field is documented as "Ignored by the stub implementation". A configurable production variant that is silently a no-op is a worse failure mode than no variant at all.
- `crates/cobre-sddp/src/stopping_rule.rs:325-356` — `StoppingRule::SimulationBased` is documented as "stub correct" but the actual two-snapshot comparison is "deferred". It computes distance against a zero baseline (never triggers on first evaluation).
- `crates/cobre-io/src/output/training_writer.rs:9` — per-phase iteration timing is **placeholder zeros**.
- `crates/cobre-io/src/constraints/penalty_overrides.rs:670-687` — five directional override fields are hardcoded to `None` because "stage-level directional overrides not yet exposed in Parquet schema".

**Impact**: User-visible features that silently don't do the documented thing. Each is a footgun.

**Fix**: For each: either implement, or gate behind a feature flag, or fail at config-parse with "not implemented" — never silently no-op.

### HI-9 — `cobre-io` infrastructure-crate genericity drift in `RowSelectionConfig`

**Where**: `crates/cobre-io/src/config.rs:184-249` exposes solver-internal knobs as user-facing JSON fields: `cut_activity_tolerance`, `basis_activity_window` (documented as "basis-reconstruction classifier, Scheme 1 sort popcount"), `max_active_per_stage` ("row-selection pipeline stage 2"). The CLAUDE.md hard rule says infrastructure crates must have zero algorithm-specific references in types, functions, **or doc comments**.

**Impact**: Long-running soft violation. Adds friction to the "swap solver out" story.

**Fix**: Re-doc with generic terminology (e.g., "constraint pruning activity threshold", "basis observation window"). The implementation can keep solver-specific algorithms; only the cobre-io types need to read as algorithm-agnostic.

---

## MEDIUM

### MD-1 — `StudySetup::resolved_parameters` is dead post-construction; `#[allow(dead_code)]` and doc are stale

**Where**: `crates/cobre-sddp/src/setup/mod.rs:188-191`. The doc claims "Retained for MPI broadcast (upcoming broadcast integration)" — but `build_resolved_parameters` is called at line 317, threaded into `build_stage_templates`, baked into `StageTemplate`, then the field on `StudySetup` is never read again (verified by grep). The MPI broadcast already happens at the `BroadcastScalarParameter`/`ConstructionConfig` level.

**Fix**: Remove the field and the `#[allow(dead_code)]`. Build `ResolvedParameters` as a local in `from_broadcast_params`, pass it to `build_stage_templates`, drop.

### MD-2 — `setup/mod.rs::from_broadcast_params` is 490 lines with `#[allow(too_many_lines)]`

**Where**: `crates/cobre-sddp/src/setup/mod.rs:262-754`. Seven distinct phases (stage_to_season → reference volumes → energy conversion → resolved parameters → templates → entity counts → block layout) inlined. The 8-submodule decomposition is clean (no back-references) but the orchestration mass stayed put.

**Fix**: Extract phase 1 (stage→season + reference volumes) and phase 2 (energy conversion + resolved parameters) into the existing `setup/orchestration.rs` (currently only 10K). Each phase has clear inputs/outputs.

### MD-3 — `HydroEnergyProductivityOverride` is a post-refactor scar threaded through three call sites

**Where**: After commit `c140b55` made `ProductionModelSet` authoritative for non-FPHA productivity, the override table is still threaded through `crates/cobre-sddp/src/hydro_models.rs:388`, `setup/mod.rs:313`, and `resolved_parameters.rs:344-415`. It is needed only by the FPHA VHA derivation now.

**Fix**: Audit non-FPHA call sites and remove the parameter where unused. The doc comment at `hydro_models.rs:380-388` already admits the limited role; the type can shrink accordingly.

### MD-4 — `config.rs` (cobre-io) has 21 config object families in 2171 lines

**Where**: `crates/cobre-io/src/config.rs:38-732`. `Config`, `ModelingConfig`, `TrainingConfig`, `ForwardPassConfig`, `RowSelectionConfig`, `TrainingSolverConfig`, `RawScenarioSourceConfig`, …, `EstimationConfig`, `ExportsConfig` — all in one file.

**Fix**: Split into `config/training.rs`, `config/policy.rs`, `config/simulation.rs`, `config/estimation.rs`, `config/exports.rs`. `Config` stays in `config/mod.rs`. Each file becomes editable in isolation.

### MD-5 — `output/policy.rs` (2570 lines) conflates 4 distinct concerns

**Where**: `crates/cobre-io/src/output/policy.rs`. Mixes record types (`PolicyCutRecord`, `PolicyBasisRecord`), FlatBuffers serializers (`serialize_stage_cuts`, etc.), safe raw-byte wire helpers (`resolve_root`, `read_u32_le`), and file-level read/write.

**Fix**: Split into `policy/records.rs`, `policy/codec.rs`, `policy/checkpoint.rs`.

### MD-6 — `constraints/bounds.rs` and `constraints/penalty_overrides.rs`: 5 near-identical parquet parsers each

**Where**: `crates/cobre-io/src/constraints/bounds.rs:123-855` (5 `*BoundsRow` + 5 parsers) and `constraints/penalty_overrides.rs:107-727` (4 + 4). Same shape; no shared row-trait or column-extraction helper.

**Fix**: Define a small `ParquetRowParser` helper (column-by-name + per-row validate + sort). Reuse across the 9 parsers. Roadmap features (batteries, GNL thermals) will need a 10th and 11th.

### MD-7 — `cobre-comm/Cargo.toml` manually mirrors workspace lint table

**Where**: `crates/cobre-comm/Cargo.toml:24-35`. Because Cargo cannot combine `lints.workspace = true` with `unsafe_code` override, the full lint set is duplicated.

**Impact**: Any workspace-lint tightening silently skips `cobre-comm` (and similarly `cobre-solver`/`cobre-python`).

**Fix**: Add a tracking comment in workspace `Cargo.toml` ("if you change [workspace.lints], also update overrides in cobre-comm, cobre-solver, cobre-python Cargo.toml") and consider a small CI check.

### MD-8 — `SolverWorkspace`, `StageIndexer` fields are `pub` when `pub(crate)` suffices

**Where**: `crates/cobre-sddp/src/workspace.rs:592-634` (`pub rank, worker_id, solver, patch_buf, current_state, worker_timing_buf`), `crates/cobre-sddp/src/indexer.rs:99-376` (78 pub fields/ranges). Confirmed by grep: no external consumer reads any of these.

**Fix**: Sweep `pub` → `pub(crate)`. Risk-free.

### MD-9 — Test mocks (`MockSolver`, `StubComm`) duplicated across 5+ files

**Where**: `forward.rs:1271`, `backward.rs:917`, `lower_bound.rs:585`, `simulation/pipeline.rs:1083`, `training.rs:492` — ~80-150 line near-duplicates each.

**Fix**: Single `test_support` module behind `#[cfg(test)]`, or `crates/cobre-sddp/tests/common/test_support.rs` if tests move to integration suites (HI-6). Eliminates ~600 lines.

### MD-10 — `estimation.rs` (6456 lines): path-dispatch, fitting, validation, reporting in one module

**Where**: `crates/cobre-sddp/src/estimation.rs`. Two `#[cfg(test)]` blocks (lines 2199 and 2550) hint at structural issues. Public surface is 2 functions, but a 7-path matrix is inlined.

**Fix**: Split into `estimation/{dispatch, full, partial, user_ar, report}.rs`. Lower urgency than HI-6 because the production logic genuinely is concentrated here.

### MD-11 — `BroadcastScalarParameter`/`BroadcastParameterKind`/`BroadcastComputedParameter` mirror `cobre-core` types

**Where**: `crates/cobre-io/src/broadcast.rs:52-90` plus `From` impls. Exists because postcard does not support internally-tagged enums, and `cobre_core` types use that tagging for JSON.

**Fix**: Move internally-tagged JSON representation into `extensions/scalar_parameters.rs` (it already uses a `Raw*` intermediate), let the canonical in-memory type be postcard-friendly. Removes ~40 lines of conversion code and a parallel type hierarchy.

### MD-12 — CLI imports `setup::` submodule helpers (`build_ncs_factor_entries`, `load_load_factors_for_stochastic`)

**Where**: `crates/cobre-cli/src/commands/run.rs:33-39`. These look like internal helpers; the facade is `train`/`simulate`/`StudySetup`.

**Fix**: Either promote to top-level re-exports if they're genuinely facade-level, or refactor the CLI so they're not needed at the orchestration layer.

### MD-13 — `cobre-io::TrainingOutput` lives in cobre-io but its constructor lives in cobre-sddp

**Where**: `crates/cobre-io/src/output/mod.rs:266` (struct); `crates/cobre-sddp/src/training_output.rs:316` (constructor). The struct is structurally `pub` for every field because of this split.

**Impact**: No guardrail prevents a future maintainer from adding an SDDP-specific field (e.g., a Benders-specific metric) to `cobre-io::TrainingOutput` — that would violate infrastructure-crate genericity.

**Fix**: Add a CLAUDE.md note (or a doc-test) documenting the asymmetry. Lower priority because HI-2's lift-into-cobre-sddp would close this naturally.

### MD-14 — Broken rustdoc link in `cobre-io/broadcast.rs`

**Where**: `crates/cobre-io/src/broadcast.rs:20` links to `../../cobre_cli/commands/broadcast/struct.BroadcastConfig.html` — `cobre-io` does not depend on `cobre-cli`. Dead on docs.rs.

**Fix**: Remove the link or replace with prose. Trivial.

---

## LOW

### LO-1 — `extensions/mod.rs` `load_*` wrappers are pure boilerplate (5 functions × 8 lines + 12 lines of doc)

**Where**: `crates/cobre-io/src/extensions/mod.rs:71-178`. Pattern: `None => Ok(vec![]); Some(p) => parse_*(p)`. Same pattern open-coded again in `validation/schema.rs:357-368`.

**Fix**: One generic helper `optional_parse<T, F>(path: Option<&Path>, parse: F)` or just inline at call sites.

### LO-2 — `cobre-sddp::hydro_models` re-runs `validate_structure` and re-builds `ValidationContext` only to discard the errors

**Where**: `crates/cobre-sddp/src/hydro_models.rs:714, 1416` (the `FileManifest` is taken, errors are thrown away).

**Fix**: Expose `FileManifest` from `load_case` so cobre-sddp can take it without redoing validation.

### LO-3 — `cobre-core::PolicyGraph` carries SDDP heritage in its docstring

**Where**: `crates/cobre-core/src/temporal.rs:561` ("`PolicyGraph` (SS12.10)"). The field name is generic enough; only the documentation reference is borderline.

**Fix**: Reword to remove the SDDP.jl-specific reference; or accept as methodology terminology.

### LO-4 — `broadcast_basis_cache` allocates two Vecs per call (finalize, not hot)

**Where**: `crates/cobre-sddp/src/training.rs:255-256`. Once per training run.

**Fix**: Pre-allocate on `WorkspacePool` if you want full consistency with the pre-allocate-and-reuse discipline. Low impact.

### LO-5 — `validation/semantic/shared.rs` is a literal placeholder file

**Where**: 6 lines documenting that future cross-domain semantic helpers would go here. Nothing currently uses it.

**Fix**: Delete; add the file back when there's an actual cross-domain helper.

### LO-6 — `pipeline::run_pipeline_with_report` has `#[allow(clippy::too_many_lines)]` at two annotations

**Where**: `crates/cobre-io/src/pipeline.rs:52, 66`. 154 lines of layered orchestration.

**Fix**: Three private helpers (`validate_all`, `resolve_all`, `build_system`). Cosmetic.

---

## Recommended Sequence

Ordered for ROI (lowest-effort highest-impact first):

1. **CR-1** + **CR-2** + **HI-5** — three trivial-to-medium fixes that each close a real correctness or hard-rule violation. ~half a day each.
2. **HI-1** — `pub mod` → `pub(crate) mod` sweep. One sitting. Massive future-restructuring leverage.
3. **CR-3** + **HI-3** — Joint fix: extend `load_case` return so `ParsedData` artifacts flow downstream. Removes ~50 lines of duplicate file-loading in `hydro_models.rs`/CLI/Python. Most architecturally important change.
4. **HI-2** — Lift `write_policy_checkpoint` + `export_stochastic_artifacts` into `cobre-sddp::orchestration`. Closes MD-13 implicitly.
5. **HI-6** — Move tests out of `lp_builder/template.rs` and `simulation/pipeline.rs` into `tests/`. Compile-time win.
6. **HI-4** + **HI-7** — Move semantic tests next to their domain modules; rename Layer 6 to be honest.
7. **HI-8** — Resolve the three stubs (Dominated select / SimulationBased / per-phase timing / directional overrides) one by one — for each, either implement or remove the user-facing knob.
8. **HI-9** + **MD-4** + **MD-5** + **MD-6** — IO-side cleanup pass (RowSelectionConfig docs, config.rs split, output/policy.rs split, parquet parser DRY).
9. The remaining MD/LO items: clean up opportunistically.

---

## What is in good shape

These were inspected and found compliant — no action needed:

- Algorithm-name leak in infrastructure crates (`cobre-core`, `cobre-io`, `cobre-solver`, `cobre-stochastic`, `cobre-comm`): zero hits ✓
- Production `.unwrap()`: zero (all hits are in doc-comments or under `cfg(test)`) ✓
- `Box<dyn Trait>` in production paths: zero (enum dispatch used throughout) ✓
- `bincode` usage: zero ✓
- Wire-format ownership discipline: `CapturedBasis::{to,try_from}_broadcast_payload`, `cut::wire::*`, `state_exchange::*` are correctly the sole owners with explicit doc contracts ✓
- Context-struct discipline: `ForwardPassInputs`, `BackwardPassInputs`, `SimulationInputs`, `LbEvalScratchBundle` all consolidate hot-path data correctly ✓
- Setup submodule cohesion: no back-references between sibling submodules ✓
- `cobre-sddp` ferrompi independence: depends only on `cobre-comm` ✓
- CLI ↔ Python file parity (which files are written): intact ✓
- `TODO`/`FIXME` markers in production: 1 found ✓

---

## Audit Method

This audit combined four parallel investigations:

1. **Mechanical sweeps** (grep/find based) — algorithm-name leak detection, `.unwrap()` counts split by `#[cfg(test)]` boundary, `Box<dyn>` occurrences, lint-suppression patterns (`#[allow(...)]`), TODO/FIXME debt, file-size and prod-vs-test ratios, public-API enumeration, allocation-pattern grep in hot-path files.
2. **cobre-io adversarial agent** — focused on SRP violations, single-source-of-truth, layering, validation discipline, public-API leaks, and recent feature accretion in `cobre-io`.
3. **cobre-sddp adversarial agent** — focused on hot-path discipline, context-struct usage, lp_builder anomaly, fat files, wire-format ownership, and recent refactor scars in `cobre-sddp`.
4. **Cross-crate adversarial agent** — verified CLI↔Python parity, cobre-core/cobre-comm boundary integrity, and output type duplication.

Findings from all four sources were verified against the codebase before inclusion. Severity is ranked by impact × likelihood; recommendations are ordered by ROI within each tier.
