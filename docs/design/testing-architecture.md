# Uniform Testing Architecture

**Status:** Proposed (analysis + target standard; not yet implemented)
**Scope:** the whole workspace's test strategy — layering, per-crate structure,
shared fixtures, comparators, runner, CI tiering, and correctness hardening.
**Relationship to other docs:**

- `test-binary-consolidation.md` is **Layer 1 (binary structure)** of this
  standard — this document subsumes and generalizes it, and does not restate its
  file-by-file mapping.
- `.claude/rules/testing.md` is the authoritative tier taxonomy; this document
  **formalizes and extends** it (adds property / compile-fail / hardening tiers,
  a decision rule, and a comparator standard) without weakening any existing
  contract.
- Several items here close entries in
  `reserved-seams-and-deferred-debt.md` (oracle-harness duplication, Python-Rust
  tests invisible to CI, mega-file/inline-test asymmetry).

The measured numbers below are a point-in-time snapshot for calibration; the
standard is stated as invariants, not counts.

---

## 1. Problem statement

The suite is **strong on correctness and weak on uniformity/sustainability.**
The advanced machinery — the four test tiers, golden baselines, the determinism
gate binary, the shared fixture harness — lives almost entirely in `cobre-sddp`;
every other crate tests ad hoc. The workspace links **84 separate integration
binaries** (59 of them statically linking the C++ solver), homes unit tests
inconsistently (giant inline modules next to extracted `tests.rs` siblings _in
the same directory_), runs the entire slow suite on every PR, and has no uniform
cross-crate fixture-sharing convention, no nextest configuration, and no
correctness-hardening tier (miri / sanitizers / fuzz). None of this is broken; all of it scales badly and
resists being taught to a new crate or contributor. The objective is a **single,
uniform, per-crate testing standard** that preserves the correctness depth while
making the suite cheap to build, consistent to navigate, and safe to extend.

---

## 2. Current state (measured)

### 2.1 Shape

| Metric                                   | Value                                                                   |
| ---------------------------------------- | ----------------------------------------------------------------------- |
| Integration-test binaries (`tests/*.rs`) | **84** (sddp 35, cli 16, io 13, solver 8, stochastic 8, comm 2, core 2) |
| …that statically link the solver         | **59** (all sddp + cli + solver binaries)                               |
| Unit tests (`#[test]` in `src/`)         | ~5340                                                                   |
| Integration tests                        | ~984                                                                    |
| Doctests                                 | ~345                                                                    |
| pytest (cobre-python)                    | 31 files / ~189 tests                                                   |
| Golden bit-exact cases                   | 5 (D06/D15/D30/D34/D41) × 2 backends                                    |
| `to_bits`/ULP determinism assertions     | ~917 across ~70 files                                                   |
| `proptest!` sites                        | 5                                                                       |
| Slow-gated (`slow-tests`) attributes     | ~94, entirely in `cobre-sddp`                                           |

The unit:integration ratio (~5.4:1) is a healthy pyramid; the cost is not test
_count_ but binary _count_ × static-solver-link.

### 2.2 Where the sophistication lives

`cobre-sddp` is the sole home of: all four test tiers, the only `tests/common/`
harness (1591 LOC — `StubComm`/`Rank0Of2`, `build_setup_*`, `make_*` builders,
`parity_hash`, `permute`), the only `tests/fixtures/` (dual golden baselines +
deterministic case decks), the `mpi_wire.rs` determinism-gate binary (70 tests,
power self-checks), and the only `benches/`. Every other crate is unit +
behavioral only, constructing fixtures per-file.

### 2.3 Tooling & CI topology

- **Runner is inconsistent**: the HiGHS `Test` job uses `cargo test`; the CLP
  job and the shuffle-matrix job use `cargo nextest`. There is **no
  `.config/nextest.toml`** — no profiles, retries, partitioning, JUnit, or
  archive.
- **No CI cadence tiering**: `NON_SOLVER_FEATURES` (used by `Test`, `CLP`, and
  `Coverage`) **includes `slow-tests`**, so the full slow suite runs on **every
  PR**. The `slow-tests` cargo feature therefore functions only as a _local-dev_
  convenience, not as a CI tier. The one lighter job is `Check`.
- **Order-invariance shuffle matrix** (`invariance-shuffle.yml`) is
  **`workflow_dispatch`-only** — its nightly cron is commented out — so a
  _hard-rule_ determinism guarantee is exercised in automation only on manual
  dispatch.
- **Coverage** via `cargo-llvm-cov` → codecov (HiGHS backend). Good.
- **Real multi-rank MPI** via a SLURM Docker cluster on `examples/4ree`
  (`mpi-slurm.yml`). Good. In-process MPI via `StubComm`/`Rank0Of2` in 25 files.
- **Absent**: miri, sanitizers/valgrind, fuzzing, snapshot tooling
  (`insta`/`expect-test`), a uniform cross-crate fixture convention (the
  `test-support` feature exists but is applied inconsistently), and any
  `cargo test`/doctest run for the workspace-excluded `cobre-python` (its ~21
  Rust unit tests and doctests never compile in CI).

---

## 3. Assessment (test-engineer lens)

### 3.1 What is genuinely strong — keep and standardize, do not disturb

1. **An explicit four-tier taxonomy, including an analytical-derivation tier.**
   Golden-bit-exact / behavioral / LP-structural / analytical closed-form. The
   analytical tier is the Rust analog of the scientific-computing **Method of
   Manufactured Solutions** — proving _correctness against a derived truth_, not
   "same as last time." Most Rust projects have nothing like it; this is ahead of
   the field and must be preserved and named.
2. **Contract-pinning.** Every invariant in `.claude/rules/sddp.md` is tied to a
   named regression test. This is the single best property of the suite.
3. **Determinism discipline at HPC grade.** ~917 bit-exact assertions, the
   `mpi_wire` gates with _power self-checks_ (a gate proves it exercises the
   condition it guards), dual HiGHS/CLP golden baselines, an order-invariance
   shuffle matrix, and real multi-rank MPI. This matches the strict end of the
   parallel-numerics reproducibility spectrum.
4. **O(1) fixture field-add builder** (`tests/common/builders.rs`).
5. **Coverage measured; compile-fail tested (via doctests); property tests for
   ordering.**

### 3.2 Sustainability & uniformity problems (ranked by leverage)

1. **Binary sharding — the dominant structural cost.** Rust links one binary per
   `tests/*.rs`; with `n` binaries and `m` libraries the linker does `m·n` work,
   and Cargo runs integration binaries _sequentially_. Cobre amplifies this by
   statically linking a C++ LP solver into all 59 solver-linked binaries. (When
   Cargo itself moved to a single integration binary, compile time dropped **3×**
   and artifacts **5×** — matklad.) This is Layer 1 / the consolidation doc.
2. **Per-crate non-uniformity.** There is no crate testing _standard_: tiers,
   the shared harness, fixtures, and golden management exist only in `cobre-sddp`.
   A contributor to `cobre-io` or `cobre-stochastic` has no template to follow.
   This is the user's central concern and the reason the suite "isn't
   sustainable" — it cannot be taught, only imitated from one crate.
3. **Fixture / test-support fragmentation.** The cross-crate mechanism is a
   `test-support` _cargo feature_ that gates internal symbols behind
   `#[cfg(any(test, feature = "test-support"))]`; the reusable builders live in
   `cobre-sddp/tests/common/`; `cobre-python` (workspace-excluded) shares nothing
   and its Rust tests never run in CI. Three different sharing mechanisms, none
   uniform.
4. **Inline-giant vs extracted-sibling asymmetry — intra-directory.**
   `lp/builder/entries.rs` (8081 inline test LOC) and `columns.rs` (7612) keep
   tests inline while their siblings `template.rs`/`layout.rs` use extracted
   `tests.rs`. The homing rule is a coin-flip.
5. **No CI cadence tiering.** Everything (including the entire slow suite) runs
   on every PR, while the order-invariance shuffle — a hard-rule guarantee —
   runs only on manual dispatch. This is backwards: the cheap high-signal gate is
   under-run and the expensive sweep over-runs. It also contradicts the project's
   own recorded intent that slow tests not burden routine CI.
6. **Runner inconsistency + zero nextest configuration.** `cargo test` vs
   `nextest` across jobs; no profiles, no per-test process isolation guarantees
   documented, no partitioning/sharding, no JUnit dashboard feed, and — most
   costly for an HPC project — **no use of `nextest archive`** to build on a CI
   node and run on Slurm/MPI compute nodes.
7. **Correctness-hardening gaps for an FFI/HPC solver.** No miri on the pure-Rust
   `unsafe` (the isolated `gemm` kernel, raw-buffer reuse); no sanitizer/valgrind
   job for the C++ solver / MPI / PyO3 FFI; no fuzzing of the parser-heavy
   `cobre-io`; property testing at only 5 sites despite pervasive
   sort/canonicalization/reduction invariants that are its ideal target.
8. **Ad hoc golden and matrix management.** Golden re-baselining is prose
   discipline; the HiGHS×CLP × mpi/numa/shared-memory × slow feature matrix is
   uncodified (the CLP suite can silently rot, per the register's
   backend-scoped-parity caveat); MPI tests carry no rank-weight, so concurrent
   multi-rank tests can oversubscribe cores.

---

## 4. How mature projects test (benchmark)

### 4.1 Large Rust projects

- **One integration binary.** `tests/it/main.rs` with everything else a `mod`
  submodule; libtest still parallelizes the `#[test]`s within it. Adopted by
  Cargo and rust-analyzer (matklad, _Delete Cargo Integration Tests_ /
  _Fast Rust Builds_). The `m·n` link argument is exactly cobre's amplified case.
- **cargo-nextest.** Process-per-test isolation (a solver segfault or MPI
  deadlock kills one process, not the binary's whole result set), `hash:`
  partitioning for sharding, retries with flaky-marking, JUnit XML, and
  **`archive`** (build on one node, ship a `.tar.zst`, run on compute nodes with
  no toolchain — the standout HPC feature).
- **Snapshot testing.** `insta` (file + inline, `cargo insta review`) and
  rust-analyzer's `expect-test` (`UPDATE_EXPECT=1` patches expectations inline) —
  for verbose structured output (report tables, cut-pool dumps, LP structure).
- **Property + compile-fail.** `proptest` (regression seeds committed to VCS) for
  invariants; `trybuild` for API-misuse diagnostics (only if proc-macros ship).
- **Hardening.** tokio runs dedicated `miri`, `loom`, `valgrind`, and
  `-Zsanitizer=address` jobs; miri via `cargo miri nextest` (isolation disabled,
  strict provenance).
- **Centralized shared fixtures — a `*-test-support` crate _or_ a `test-support`
  feature.** tokio-test, rust-analyzer `test-utils`, polars-testing, datafusion
  `test_util` are dedicated crates; the lighter, equally-common variant is a
  `test-support` cargo feature on existing crates. Either centralizes
  fixtures/comparators workspace-wide; cobre uses the feature variant (§5.2).
- **CI tiering.** tokio/DataFusion/polars all split PR-fast from an extended
  main-branch/nightly tier (the full edge-case sweep runs off the PR path).

### 4.2 HPC / scientific computing (C++, Julia)

- **CTest labels + timeouts + `PROCESSORS` weights** (Trilinos/TriBITS, deal.II,
  PETSc): one test registry sliced at _run time_ by label regex into
  fast/slow/perf; per-test timeouts kill hung MPI/solver tests; each MPI test
  declares a rank weight so the scheduler never oversubscribes cores, and a
  max-rank budget auto-excludes over-budget tests so the same suite runs
  laptop→cluster.
- **Tolerance-aware golden diff with "alt" outputs** (deal.II `numdiff`, PETSc
  `petscdiff` + `alt` files): scientific codes _cannot_ demand byte-equality
  across machines, so they diff with FP tolerance and allow _multiple legitimate
  outputs_. Cobre sits at the strict end (bit-exact within a mode); the PETSc
  `alt`-file idea is the clean precedent for encoding "different-but-valid
  optimal vertex" (the hot≠cold case cobre already exempts).
- **Verification & Validation.** The **Method of Manufactured Solutions** and
  **order-of-accuracy** tests (Sandia SAND2000-1444; deal.II `ConvergenceTable`)
  prove correctness against a _derived_ truth — exactly cobre's analytical tier,
  and the strongest correctness signal a numerical code can have.
- **Nightly dashboards + promotion gates** (CDash; Trilinos requires a clean
  nightly before `develop`→`master`) — maps onto cobre's `develop`→`main`
  discipline; nextest JUnit is the Rust-native dashboard feed.
- **Julia optimization (closest domain).** SDDP.jl seeds `Random.seed!(12345)` in
  its test runner and runs `docs/examples` as tests; JuMP.jl runs `Aqua.jl`
  package-hygiene as a dedicated gate over an explicit toolchain matrix.
- **Reproducibility comparator split** (ReproBLAS, Intel MKL CNR, Collange et
  al.): exact bit/hash equality for within-mode + cross-rank-count invariance
  (seeded RNG, fixed reduction order); ULP tolerance for cross-mode equivalence.
  This is _precisely_ cobre's stated contract.

---

## 5. Proposed standard — uniform per-crate testing architecture

Every crate adopts the same layout, the same sharing mechanism, the same runner
config, and the same tier vocabulary. The standard is the following eleven rules.

### 5.1 Canonical per-crate layout

```
crates/<crate>/
  src/<module>.rs            # unit tests INLINE (#[cfg(test)] mod tests) …
  src/<module>/tests.rs      # … OR an extracted sibling above the threshold
  tests/it.rs                # THE single integration binary (mod-declares below)
  tests/it/<domain>.rs       # one submodule per domain (boundary, cut, sim, …)
  benches/                   # criterion — only where a hot path is measured
```

- **One integration binary per crate** (`tests/it.rs` or `tests/it/main.rs`),
  submodules by domain via `#[path]`-free `mod`. Generalizes the consolidation
  doc from sddp/cli to **every** crate. Kills the `m·n` re-link.
- **One deterministic unit-test homing rule**: inline when the test module is
  below a fixed threshold (proposal: ~500 test-LOC or ~40 test fns), extracted to
  a sibling `tests.rs` above it. No more intra-directory coin-flip. (Pick the
  threshold once here; it is a lint, not a judgment call.)

### 5.2 A uniform `test-support` feature convention (not a dedicated test crate)

Keep the `test-support` cargo-feature mechanism the repo already uses
(`cobre-core`, `cobre-solver`, `cobre-sddp`) and make it uniform and complete —
do **not** introduce a dedicated test crate. Cross-crate test sharing in Rust is
either a crate or a `#[cfg(any(test, feature = "test-support"))]` feature; a
separate crate adds a workspace member, a version-bump and crates.io-publish
surface, and a dev-dependency cycle (`cobre-io` → support → `cobre-sddp` →
`cobre-io`), and forces an artificial layering split to respect the
infra-genericity hard rule — for no capability the feature does not already
provide. The `#[cfg(...)]` gate compiles to nothing in a normal build, so the
"test feature in a production `Cargo.toml`" cost is cosmetic.

The convention:

- **Helpers live with the type they build** — a cleaner ownership model than a
  catch-all crate. Generic scaffolding (the golden-SHA helper, the two-tier
  comparator of §5.4, the `permute` order-shuffle helper, RNG-seed and `TempDir`
  conventions) lives in `cobre-core` (the universal base dependency) behind its
  `test-support` feature; `StubComm`/`Rank0Of2` live in `cobre-comm` (they impl
  its `Communicator`); the entity/`StudySetup` builders stay in `cobre-sddp`.
- **Every crate exposes its shareable fixtures the same way** — a `test-support`
  feature gating them behind `#[cfg(any(test, feature = "test-support"))]`, enabled
  by consumers as a dev-dependency feature. `cobre-sddp/tests/common/` collapses
  into `cobre-sddp`'s `test-support` surface so `cobre-cli` and `cobre-python`
  reach it, not just sddp's own `tests/`.
- **`cobre-python` dev-depends on `cobre-sddp`'s `test-support`**, which (with the
  §5.11 CI wiring) lets its Rust tests share the same fixtures.

A dedicated crate wins in exactly one case — heavy test-only dependencies you
want kept out of every production crate's dev-graph, or a pristine
crates.io-published surface — neither of which applies today; revisit only if the
helper surface grows its own heavy deps. This convention also resolves the
register's oracle-harness-duplication item (hoist the `close`/tolerance helpers
into `cobre-core`'s `test-support` surface).

### 5.3 The tier taxonomy, formalized with a decision rule

Keep the four tiers from `testing.md`; name two more that already exist implicitly
(property, compile-fail) and one to add (hardening). The **decision rule** a new
test follows, cheapest-sufficient-tier first:

| Tier                               | Proves                                                                         | Use when                                             | Cost         |
| ---------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------- | ------------ |
| Unit                               | a function's local contract                                                    | always, by default                                   | cheap        |
| Property                           | an invariant over generated inputs (order-invariance, reduction commutativity) | a claim quantifies over "all orderings/inputs"       | cheap        |
| LP-structural                      | column/row/coefficient wiring without a solve                                  | a wrong count would be numerically smooth            | cheap        |
| Analytical (MMS/order-of-accuracy) | correctness vs a **hand-derived** value/cut/rate                               | a _new_ correctness claim                            | medium       |
| Behavioral                         | LB==UB / known cost to tolerance, backend-agnostic                             | a deterministic case's end result                    | medium       |
| Golden bit-exact                   | byte-stability of the final artifact                                           | the small, deliberate cross-feature set only         | high         |
| Compile-fail (doctest/`trybuild`)  | misuse fails to compile                                                        | a type-state contract (e.g. `ValidatedBoundaryCuts`) | cheap        |
| Hardening (miri/sanitizer/fuzz)    | absence of UB / crash on adversarial input                                     | code with `unsafe`/FFI or untrusted parsing          | high, off-PR |

Rule of promotion: **write the analytical test before adding a behavioral case,
and a behavioral case before promoting anything into the golden set** (golden
membership needs justification — it is the only tier that breaks on a faster path
to the same answer).

### 5.4 The two-tier comparator standard

Encode the determinism contract in shared comparators (`cobre-core`'s
`test-support` surface, §5.2), not ad hoc per test:

- **Exact (`to_bits`/SHA)** for the _within-mode reproducibility_ and
  _cross-rank/thread invariance_ tier — the guarantee cobre actually makes.
- **ULP-tolerance** for _cross-mode / cross-algorithm equivalence_ (objective
  agreement where duals legitimately differ). A `assert_equivalent_vertex`
  comparator asserts equal objective + primals within ULP and _permits_ different
  duals — the PETSc `alt`-output idea, typed. This stops contributors from either
  over-asserting bit-equality across modes or under-asserting with a loose `abs`
  tolerance.

### 5.5 Runner standard — nextest everywhere, configured

Adopt `cargo-nextest` as the sole runner (keep a separate `cargo test --doc` for
doctests, which nextest does not run) and commit a `.config/nextest.toml`:

- `[profile.ci]` with `retries` (exponential backoff) + flaky-marking, and
  `[profile.ci.junit]` for the dashboard feed.
- `[test-groups]` + overrides to bound MPI/solver-heavy tests (the rank-weight /
  oversubscription control, §5.8), and per-test `slow-timeout` to kill hung
  MPI/solve tests.
- Standardize CI shards on `--partition hash:m/n` (deterministic pinning).
- **`nextest archive`** for the SLURM/MPI path: build the archive on the CI node,
  ship the `.tar.zst` to compute nodes, run without a toolchain there.

### 5.6 CI cadence tiering

Split the monolithic per-PR run into cadence tiers (this is the change that most
improves sustainability):

| Tier          | Contents                                                                                                                                                    | Cadence                      |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| 0 — smoke     | fmt, clippy, `Check`, quality-script gates                                                                                                                  | every push                   |
| 1 — PR-fast   | unit + property + structural + **determinism gates (`mpi_wire`)** + doctests, both backends (check/clippy), coverage                                        | every PR                     |
| 2 — extended  | `slow-tests` sweep (D-case, FPHA, forward-sampler convergence), golden parity sweep, **order-invariance shuffle matrix**, real multi-rank MPI (`mpi-slurm`) | merge-to-`develop` + nightly |
| 3 — hardening | miri (pure-Rust unsafe), valgrind/ASan (FFI), fuzz smoke                                                                                                    | nightly / weekly             |

The load-bearing moves: (a) **remove `slow-tests` from the per-PR feature set**
(`NON_SOLVER_FEATURES`) and run it on the extended tier — PR CI gets dramatically
faster and the `slow-tests` gate finally _means_ a CI tier, not just a local
convenience; (b) **put the shuffle matrix on the nightly cadence** (uncomment the
cron) so the hard order-invariance rule is actually exercised in automation. This
is a policy change with a tradeoff — a slow-test regression is caught at
merge/nightly rather than on the PR — and should be ratified explicitly; it
matches DataFusion/tokio norms and the project's own recorded intent that slow
tests not gate routine CI.

### 5.7 Feature/backend matrix policy

Declare the `{backend} × {feature set} × {CI job}` matrix in one place (a table in
this doc + a CI matrix), so the CLP path cannot silently rot: HiGHS is the golden

- primary tier; CLP re-runs behavioral + structural + its own golden baseline;
  `mpi`/`numa`/`shared-memory` gate only the tests that need them. State the
  backend-scoping of the parity roster (the register's open caveat) inline in the
  roster.

### 5.8 MPI & determinism testing standard

- Keep `StubComm`/`Rank0Of2` (in `cobre-comm`'s `test-support` surface) as the
  in-process rank-shape harness for PR-tier determinism gates; keep the SLURM job for real
  multi-rank on the extended tier.
- Give every multi-rank test an explicit **rank weight** (nextest test-group
  `max-threads`) and a **max-rank budget**, the CTest `PROCESSORS` /
  `MPI_EXEC_MAX_NUMPROCS` idea, so concurrent multi-rank tests don't oversubscribe
  and the suite runs laptop→cluster unchanged.
- Frame the forward-sampler convergence test explicitly as an
  **order-of-accuracy** verification (observed vs theoretical rate).

### 5.9 Correctness-hardening additions (right-sized)

Add Tier-3, off the PR path, scoped to where it pays:

- **miri** on the pure-Rust `unsafe` only (the isolated `gemm` kernel, raw-buffer
  reuse). Miri **cannot** cross the C++ FFI boundary — do not expect it to cover
  the solver.
- **valgrind / `-Zsanitizer=address`** for the C++ solver + MPI + PyO3 FFI, where
  miri cannot reach (tokio runs both as dedicated jobs).
- **One `cargo-fuzz` target for the `cobre-io` parsers** (config/Parquet/JSON
  ingest is the untrusted-input surface); commit the corpus.
- **Expand `proptest`** from 5 sites to cover the declaration-order-invariance and
  reduction-order invariants directly (permute → assert identical bits), with
  `proptest-regressions/` committed to VCS.

### 5.10 Golden / snapshot standard

- Keep SHA-bit-exact for the small deliberate golden set (unchanged).
- Adopt **`insta` (or `expect-test`)** for the _verbose text_ goldens that are
  currently hand-maintained strings — the reconciliation diagnostic block, CLI
  report tables, LP-structure dumps, exported schemas — so regeneration is
  `cargo insta review` / `UPDATE_EXPECT=1`, not manual editing.
- One `just`/script regen entrypoint per golden family; the "goldens are the final
  artifact, never the trajectory" rule (already in `testing.md`) stays.

### 5.11 cobre-python parity

Wire the crate's Rust tests into CI (a `cargo test -p cobre-python` / `cargo check
--tests` step), closing the register's "Python-Rust tests invisible to CI" item.
Have it dev-depend on `cobre-sddp`'s `test-support` feature so its Rust fixtures
match the CLI's by construction; keep the pytest output-parity suite unchanged.

---

## 6. Migration path (incremental, coverage-neutral)

Each phase is behavior-neutral and gated on `cargo nextest list` count parity
(no test added, removed, or renamed except the module-path prefix). Order by
leverage:

1. **Layer 1 — binary consolidation** (`test-binary-consolidation.md`): sddp
   35→~9, cli 16→~5, solver 8→~2. Biggest single win; already specced.
2. **`.config/nextest.toml` + nextest as the sole runner** (+ `--doc` step). Cheap,
   immediate CI-time and observability win; unlocks archive/partitioning.
3. **Standardize the `test-support` feature** (§5.2): collapse `tests/common/`
   into `cobre-sddp`'s `test-support` surface, move the generic comparators into
   `cobre-core`'s, and dev-depend `cobre-python` on it — no new crate.
4. **CI cadence tiering** (§5.6): move `slow-tests` off the PR feature set;
   enable the shuffle-matrix nightly. Ratify the tradeoff first.
5. **Unit-test homing lint** (§5.1 threshold) + resolve the inline-giant
   asymmetry as files are touched.
6. **Tier-3 hardening** (miri / sanitizer / fuzz jobs) — additive, off-PR.
7. **`insta` for text goldens** — opportunistic, as each hand-maintained golden is
   next touched.

Phases 1–3 are the sustainability core; 4 is the policy decision; 5–7 are
steady-state hygiene.

## 7. Non-goals / what NOT to over-adopt

- **`loom`** — only if hand-rolled lock-free/atomic code appears; MPI
  message-passing is out of its scope. Not warranted today.
- **`trybuild`** — the doctest `compile_fail` blocks already cover the type-state
  contracts; adopt only if a proc-macro/DSL ships.
- **Blanket miri** — the `unsafe` surface is mostly FFI, which miri can't enter;
  scope it to pure-Rust unsafe and lean on sanitizers/valgrind for the rest.
- **Reducing coverage to shrink the suite** — the cost is per-_binary_ and
  per-_feature-combo_, never per-_test_; deleting tests saves nothing structural
  and forfeits the suite's best asset.

## 8. References

- matklad, _Delete Cargo Integration Tests_ — https://matklad.github.io/2021/02/27/delete-cargo-integration-tests.html
- matklad, _Fast Rust Builds_ (the `m·n` link argument) — https://matklad.github.io/2021/09/04/fast-rust-builds.html
- cargo-nextest: archiving — https://nexte.st/docs/ci-features/archiving/ · partitioning — https://nexte.st/docs/ci-features/partitioning/ · JUnit — https://nexte.st/docs/machine-readable/junit/
- `insta` — https://insta.rs/docs/ · `expect-test` — https://github.com/rust-analyzer/expect-test
- `proptest` failure persistence — https://altsysrq.github.io/proptest-book/proptest/failure-persistence.html
- miri — https://github.com/rust-lang/miri · tokio miri→nextest — https://github.com/tokio-rs/tokio/pull/6885
- CMake CTest `PROCESSORS` — https://cmake.org/cmake/help/latest/prop_test/PROCESSORS.html · `FindMPI` — https://cmake.org/cmake/help/latest/module/FindMPI.html · LABELS — https://cmake.org/cmake/help/latest/prop_test/LABELS.html
- deal.II testsuite — https://dealii.org/current/developers/testsuite.html · PETSc testing — https://petsc.org/release/developers/testing/
- Method of Manufactured Solutions, Sandia SAND2000-1444 — https://www.osti.gov/biblio/759450 · deal.II `ConvergenceTable` (order of accuracy) — https://dealii.org/current/doxygen/deal.II/step_7.html
- SDDP.jl tests — https://github.com/odow/SDDP.jl/tree/master/test · Aqua.jl — https://github.com/JuliaTesting/Aqua.jl
- ReproBLAS — http://bebop.cs.berkeley.edu/reproblas/ · Intel MKL CNR — https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2024-2/numerical-reproducibility.html
- DataFusion testing (PR vs extended) — https://datafusion.apache.org/contributor-guide/testing.html · tokio CI — https://github.com/tokio-rs/tokio/blob/master/.github/workflows/ci.yml
