# Test-Binary Consolidation

**Status:** Proposed (not yet implemented)
**Scope:** build-cost reduction for the integration test suite; coverage-neutral
**Related:** `.claude/rules/testing.md` ("Cost discipline — Cobre links a solver
into every test binary"), the global Rust rule ("Integration test binaries are
expensive by default"). This is the structural half of a disk/CI-cost effort
whose hygiene half (artifact pruning, `[profile.dev]` debug-info tuning) is
already applied — see `CONTRIBUTING.md` → "Reclaiming disk space".

---

## 1. Problem

Every crate that depends on `cobre-solver` statically links the vendored
HiGHS/CLP/CoinUtils/qhull C++ into **each** `tests/*.rs` file, because Cargo
compiles one executable per integration-test file. A single `cobre-sddp`
integration binary is on the order of hundreds of MB in a debug build (solver
object code + debug info + the crate rlib), and each one pays a full C++ static
link (several seconds) at build time.

At authoring time the workspace has these integration-test file counts (the
solver-linking crates are the expensive ones):

| Crate              | `tests/*.rs` files | Statically links solver? |
| ------------------ | -----------------: | :----------------------: |
| `cobre-sddp`       |                 35 |         **yes**          |
| `cobre-cli`        |                 16 |         **yes**          |
| `cobre-io`         |                 13 |            no            |
| `cobre-solver`     |                  8 |         **yes**          |
| `cobre-stochastic` |                  8 |            no            |
| `cobre-comm`       |                  2 |            no            |
| `cobre-core`       |                  2 |            no            |

So ~59 of the ~84 integration binaries embed and re-link the solver. This is the
dominant per-build (post-`cargo clean`) contributor to `target/` size and a large
share of CI link wall-time. It also **violates the repo's own testing rule**,
which mandates grouping related tests into one file with `mod` submodules and
treats a new test _binary_ as needing justification proportional to its link
cost — the suite has drifted from that rule as files accumulated one-per-feature.

Note on cause separation: the multi-hundred-GB `target/` that motivated this work
was ~85% _stale accumulated artifacts_ (addressed by the hygiene half). This spec
addresses the **steady-state** cost that remains after a clean build: binary
count × solver-link size/time.

## 2. Goals and non-goals

**Goals**

- Reduce the number of solver-linking integration binaries ~3–4× by grouping
  related test files into domain binaries via `mod` submodules.
- Cut a clean debug build's test-binary disk footprint and cumulative link time
  proportionally.
- Bring the suite back into compliance with `.claude/rules/testing.md`.

**Non-goals / hard invariants (this is a refactor, not a coverage change)**

- **No test is deleted, skipped, renamed, or weakened.** Test-function count is
  identical before and after, verified mechanically (§6).
- **No test logic changes.** Bodies, assertions, fixtures, and the tier a test
  belongs to are untouched.
- **No change to feature gating or slow-gating.** `#[cfg(feature = "mpi")]`,
  `#[cfg_attr(not(feature = "slow-tests"), ignore = …)]`, and every other gate are
  preserved verbatim.
- **The shared `tests/common/` harness stays the single source** of fixture
  builders (`StubComm`, `build_setup_*`, `make_*`, `parity_hash`, `permute`); it
  is not duplicated or forked per domain binary.
- Determinism gates keep their power (e.g. `mpi_wire.rs`'s self-checked
  thresholds) — grouping must not reorder or share mutable global state that a
  gate depends on. Integration submodules do not share process state today, and
  must not start.

## 3. Consolidation mechanics

Each current `tests/<file>.rs` becomes a **submodule** of a domain binary,
included by path so the file contents move with minimal edits:

```
tests/
  <domain>.rs                # the new binary root
  <domain>/
    <file_a>.rs              # was tests/<file_a>.rs
    <file_b>.rs
  common/                    # unchanged, shared harness
```

`tests/<domain>.rs`:

```rust
mod common;                              // declared ONCE per binary, at the root
#[path = "<domain>/file_a.rs"] mod file_a;
#[path = "<domain>/file_b.rs"] mod file_b;
```

Per-file edits when moving `tests/file_a.rs` → `tests/<domain>/file_a.rs`:

1. **`mod common;` → remove.** The domain root owns the single `mod common;`.
   Submodule references change `use common::…` / bare `common::…` to
   `use crate::common::…` (Rust resolves `common` at the crate root, not in the
   submodule's own namespace).
2. **Crate-inner attributes stay, as module-inner attributes.** A file's leading
   `#![allow(clippy::…)]` / `#![allow(unused)]` is valid unchanged inside
   `mod file_a { #![allow(…)] … }`. With the `#[path]` include the file _is_ the
   module body, so its `#![…]` inner attributes remain legal at the top of that
   body — no rewrite needed. (This is the reason to prefer `#[path]` includes over
   physically concatenating files, which would collide the inner attributes.)
3. **Symbol collisions.** Two merged files may each define a free item with the
   same leaf name (`fn setup`, `const CASE`, a local `struct Fixture`). Because
   each file is its own `mod`, these do **not** collide — a submodule's items are
   namespaced under it. Collisions only arise for items a file declares at crate
   scope (rare in these files); audit per merge.
4. **Feature/slow gates ride along** on the individual `#[test]`/`mod`; no change.
   Files that are entirely `#[cfg(feature = "mpi")]` are grouped with each other
   (§4) so the whole domain binary is coherently gated.

This is the exact pattern `.claude/rules/testing.md` prescribes ("group related
tests with `mod`"); the `#[path]` subdir layout is the low-diff way to apply it to
existing large files without rewriting their contents.

## 4. Proposed target layout

Priority order = the solver-linking crates. Groupings are by domain so a
contributor still finds tests by subject. Exact membership is a starting
proposal, refine during migration.

### `cobre-sddp` (35 → ~9 binaries)

| Domain binary      | Absorbs                                                                                                                                                                                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `deterministic.rs` | `deterministic.rs` (stays standalone — already one large domain file)                                                                                                                                                                |
| `parity.rs`        | `parity.rs` (stays standalone — owns the slow-gated `parity_regen` ignored tests and golden baselines)                                                                                                                               |
| `anticipated.rs`   | `anticipated_core.rs`, `anticipated_scenarios.rs`, `commitment_hold_wiring_probe.rs`, `hold_k1_byte_stability_probe.rs`                                                                                                              |
| `boundary.rs`      | `boundary_reconcile_defaults.rs`, `right_boundary_cost_semantics.rs`, `right_boundary_pricing.rs`, `right_boundary_output.rs`, `right_boundary_validation.rs`, `right_boundary_e2e_deck.rs`, `shared_boundary_terminal_fan_probe.rs` |
| `simulation.rs`    | `simulation_integration.rs`, `simulation_pipeline_integration.rs`, `hydro_sim.rs`                                                                                                                                                    |
| `cut_backward.rs`  | `cut_basis.rs`, `basis_trajectory_probe.rs`, `node_native_backward_gate.rs`, `branching_value_oracle.rs`, `extensive_form_oracle.rs`                                                                                                 |
| `lp_structural.rs` | `lp_builder.rs`, `template_integration.rs`, `par_a_lag12_lp_coefficient.rs`, `scalar_parameters_declaration_order.rs`, `inflow_nonnegativity.rs`                                                                                     |
| `pipeline_io.rs`   | `forward_sampler_integration.rs`, `estimation_integration.rs`, `load_integration.rs`, `integration.rs`, `conformance.rs`, `filling_commissioning.rs`                                                                                 |
| `mpi.rs`           | `mpi_wire.rs`, `test_mpi_allgatherv_nonuniform_workers.rs`, `test_mpi_sync_cuts_invariant.rs` (coherently `mpi`-gated)                                                                                                               |

### `cobre-cli` (16 → ~5 binaries)

| Domain binary      | Absorbs                                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `cli_run.rs`       | `cli_run.rs`, `cli_run_anticipated.rs`, `cli_run_anticipated_k2.rs`, `cli_run_evaporation.rs`, `cli_run_generic_echo.rs` |
| `cli_validate.rs`  | `cli_validate.rs`, `cli_schema.rs`                                                                                       |
| `cli_reporting.rs` | `cli_summary.rs`, `cli_report.rs`, `cli_e2e_summary_report.rs`                                                           |
| `cli_metadata.rs`  | `output_metadata_active_backend.rs`, `setup_timings_metadata.rs`, `python_parity_check.rs`                               |
| `cli_basics.rs`    | `cli_smoke.rs`, `cli_color.rs`, `init.rs`                                                                                |

### `cobre-solver` (8 → ~2–3 binaries)

Group by concern (backend FFI, warm-start/basis, determinism). Lower absolute win
than `cobre-sddp`/`cobre-cli` but each of its 8 binaries links the solver, so it
is worth folding into ~2–3.

Non-solver crates (`cobre-io` 13, `cobre-stochastic` 8, `cobre-comm`,
`cobre-core`) are **out of scope**: their binaries do not link the solver, so the
per-binary cost is small. Consolidate opportunistically only, never as part of
this effort's critical path.

Expected outcome: ~59 solver-linking integration binaries → ~16–18. Clean-build
test-binary disk and cumulative link time drop roughly proportionally.

## 5. Migration plan (incremental, one domain at a time)

Do **not** move all files at once. Per domain binary:

1. Create `tests/<domain>.rs` and `tests/<domain>/`, `git mv` each member file in,
   apply the §3 per-file edits.
2. `cargo build --tests -p <crate>` (both `highs` and, where relevant, `clp`
   features) — fix `use crate::common::…` paths and any crate-scope collisions.
3. Run the new binary and confirm every moved test still runs and passes.
4. Verify **test-count parity** for the crate (§6) before moving to the next
   domain. A drop means a `#[test]` was lost in the move — stop and fix.
5. Commit per domain (`refactor(test): consolidate <domain> integration tests`),
   so a regression bisects to one domain.

Order within `cobre-sddp`: start with a small, self-contained family
(`boundary.rs` — its files are already grouped by subject and were recently
touched) to validate the mechanics end-to-end, then the larger families.

## 6. Verification / acceptance criteria

- **Test-count parity (mechanical, the primary gate).** `cargo nextest list`
  emits every discovered test; the per-crate count must be identical before and
  after each domain move:

  ```bash
  cargo nextest list -p cobre-sddp --features test-support | wc -l   # capture baseline first
  ```

  A per-domain diff of the sorted test-name list (`nextest list` names are stable)
  must show only the module-path prefix changing (`deterministic::foo` →
  `<domain>::deterministic::foo`), never an added or removed test.

- All moved tests pass under `highs` and (for the CLP-relevant binaries) `clp`.
- The `mpi`-gated domain binary compiles and runs under `--features mpi` in the
  MPI SLURM CI job and compiles (tests gated out) without it.
- Binary count for the crate drops to the target; confirm by counting executables
  Cargo emits for `--tests`.
- CI link wall-time for the crate's test build measurably drops (before/after the
  crate is fully migrated). Optional but the headline payoff — capture it once.

Golden parity baselines (`tests/fixtures/parity_baselines*`) are **not** touched;
`parity.rs` stays a standalone binary specifically so its slow-gated regen and
baseline wiring are unchanged.

## 7. Risks and mitigations

| Risk                                                          | Mitigation                                                                                                                                                                                                 |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A `#[test]` silently dropped during a move                    | Test-count parity gate (§6) per domain, before proceeding                                                                                                                                                  |
| `common` path breakage (`common::` unresolved in a submodule) | Mechanical: `use crate::common::…`; caught immediately at `cargo build --tests`                                                                                                                            |
| Crate-scope symbol collision between two merged files         | Each file is its own `mod`, so only genuine crate-scope items collide; audit + qualify per merge                                                                                                           |
| A determinism gate that relied on process isolation           | Integration submodules share no mutable global state today; grouping keeps them independent processes-per-binary is unchanged (one binary, still separate test threads); no shared `static mut` introduced |
| Feature-gated files mixed with ungated ones in one binary     | Group all-`mpi` files into the `mpi.rs` domain; keep per-`#[test]` gates intact elsewhere                                                                                                                  |
| Larger diffs harder to review                                 | One commit per domain binary; `git mv` preserves blame                                                                                                                                                     |

## 8. Alternatives considered

- **Leave as-is.** Rejected: violates `testing.md`, and the steady-state cost
  persists after every `cargo clean`.
- **Physically concatenate files into one `.rs`.** Rejected: collides the
  per-file `#![allow]` inner attributes and free-item names, and destroys `git
mv` blame. The `#[path]`-submodule approach avoids both.
- **Delete/trim tests to shrink the suite.** Rejected outright: the cost is
  per-_binary_, not per-_test_ (deleting tests within a binary removes neither the
  binary nor its solver link), and it would sacrifice the suite's correctness
  coverage — the opposite of the project's testing discipline.
- **Dynamically link the solver for dev/test builds** (static for `dist`). A
  larger, complementary lever that would shrink every test binary regardless of
  count, but it changes the vendored-static-reproducible build contract and adds
  an `LD_LIBRARY_PATH` runtime dependency. Tracked separately; it does not block
  this refactor and the two compose.
