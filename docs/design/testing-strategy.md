# Cobre testing strategy — simplification & philosophy

> **Status**: Proposal / recommendation. No test change applied yet. Scope: the
> whole test suite, not only the deterministic baselines. §2 records the
> already-approved deterministic-baseline decision; §3 inventories the suite-wide
> smells with evidence; §4 proposes remediations and a sequencing; §5 states a
> testing philosophy (Cobre-specific pillars here; the generic Rust pillars are
> proposed for the cross-project rule, placement to be decided). Effort and
> speed-up figures below are estimates to be measured, not commitments.

## 1. Why this exists

The suite grew by accretion: each new feature added a deterministic case **and** an
integration test **and** fixtures **and** bit-exact baselines, with no
consolidation pass. The result, measured on the current branch:

- **114 integration-test binaries** (each top-level `crates/*/tests/*.rs` links its
  own executable): cobre-sddp 71, cobre-cli 15, cobre-io 8, cobre-solver 8,
  cobre-stochastic 8, cobre-comm 2, cobre-core 2. cobre-solver statically links the
  HiGHS/CLP C++ into every dependent binary.
- **4,747 `#[test]` functions**, ~344 doctests, ~66k LOC under `crates/*/tests/`.
- **1,123 committed `*.parquet` fixtures** across 43 `examples/deterministic/d*`
  cases and 4 `crates/*/tests/fixtures` dirs (these also churn with metadata-only
  rewrites when tests run).
- **Shared-helper duplication**: `StubComm` defined ~59×, `build_setup_for_case`
  ~26×, `make_stage` 44×, `make_hydro` 40×, `make_bus` 23× — despite a canonical
  `tests/common/mod.rs` that only ~6 of 71 cobre-sddp test files use.
- **Deterministic parity baselines**: 29 cases hashed, each pinned in **3 locations**
  (`parity_baselines/*.sha256`, `parity_baselines_clp/*.sha256`, the `EXPECTED_HASHES`
  array in `parity_baselines_unchanged.rs`) and across **2 backends**.

The growth is linear and unbounded; the maintenance cost is the multiplicative
product of (number of cases) × (pin locations) × (backends) × (per-file helper
copies). The fix is to attack the multipliers, not to delete coverage.

## 2. Deterministic baselines — the approved decision (recorded)

The lever is **bit-exact pin count and its duplication**, not study count. Decisions:

- **Reserve bit-exact "golden" parity hashes for ~5 cases.** Choose them to be
  deliberately **feature-combined** (e.g. hydro cascade + anticipated thermal + NCS
  - discount + pumping/contracts; a multi-resolution case) so they exercise the
    shared LP-build / scaling / cut / basis machinery most refactors touch **and**
    catch the cross-feature dormant-bug class (the v0.9.1 discount×anticipated bug
    hid precisely because no shipped case combined those features).
- **Demote the remaining hashed cases to backend-agnostic behavioral assertions** —
  `LB == UB` to tolerance, known-optimum cost, water-balance closure, feature-specific
  dispatch values (`operative_state_code`, `energy_mwh`, …). One such assertion
  covers HiGHS **and** CLP and survives benign refactors with no re-baseline. 14 of
  43 cases already work this way (`deterministic.rs`); this generalizes the pattern.
- **Keep every study.** Do **not** merge or delete the deterministic cases — only
  retire most golden _hashes_. Localization and feature coverage are preserved.
- **Single source of truth for baselines.** Retire the redundant `EXPECTED_HASHES`
  mirror (a guard-on-a-guard: it only checks the `.sha256` file matches a hand-copied
  hash; git diff already records changes and the slow end-to-end test verifies the
  computation). The committed `.sha256` is the baseline.
- **Determinism is guarded separately.** Declaration-order invariance / determinism
  have dedicated tests (`determinism.rs`, `declaration_order_invariance_*.rs`), so
  the 29 parity hashes are _pure regression tripwires_ and can be thinned safely —
  see §3-G for the caveat that those DOI tests are currently weak.

## 3. Suite-wide smell inventory (prioritized)

### A — Integration-binary explosion (HIGH)

114 link units; 71 in cobre-sddp, each statically linking the solver C++. The
dominant test-build cost is the linker, not execution, and any change to
`cobre-sddp/src/` re-links all 71. Every new `tests/*.rs` multiplies it. Structural,
not behavioral — fixable with zero coverage change by grouping related tests into
fewer binaries (the `tests/<domain>.rs` + `mod` pattern; `deterministic.rs` and
`template_integration.rs` already model this).

### B — Shared-helper duplication + `common/` adoption failure (HIGH)

`StubComm`, `build_setup_for_case`, and entity builders are copy-pasted across dozens
of files; the canonical `tests/common/mod.rs` exists but is used by ~6 of 71 files.
This is _why_ the `operational_start_date` field rippled across ~95 files. Field
additions are O(files-that-construct-the-type) instead of O(1). Root cause is a
convention not enforced, not a missing abstraction.

### C — Parity hash couples regression detection to algorithmic path (HIGH)

`tests/common/parity_hash.rs::compute_parity_hash` digests a field set that includes
the **per-iteration convergence trajectory** (`compute_parity_hash(&convergence_updates,
…)`). A warm-start improvement that reaches the _same optimum in fewer iterations_
changes the hash for every affected case — all golden hashes fail though the solver
produced identical correct final output. This taxes exactly the optimization work the
project wants to encourage. The hash should cover **final numerical output** (cut pool,
simulation primal/dual/equipment/cost), not how the solver got there.

### D — 3-location × 2-backend pin / guard-on-guard (HIGH, already decided)

See §2. `parity_baselines_unchanged.rs` adds no coverage the primary parity tests lack;
it only adds a third location to keep in sync (which has already rotted once).

### E — No documented test-tier hierarchy (HIGH, preventive)

Five assertion styles coexist (bit-exact hash; tolerance LB; LP structural counts;
file-existence; dispatch-value) with no written rule for which applies when. New tests
default to copying the nearest example — often the parity hash, the most brittle tier.
Codifying tiers (§5) stops future smell generation at the source.

### F — K-parameterized near-duplicate binaries (MEDIUM)

`anticipated_backward_cut_k{1,2,3}.rs`, `anticipated_pre_horizon_seed_delivery_k{1,2,3}.rs`,
`anticipated_d_t_saturation_k{2,3}.rs` — 8 binaries differing only in a lead-stage count
K, each re-defining its own `StubComm`/setup. One binary with a K table replaces all 8.

### G — DOI tests that are tautologies (MEDIUM)

`declaration_order_invariance_anticipated.rs`'s own module doc admits it is "structurally
a tautology": because `SystemBuilder::build()` sorts entities before any downstream layer,
both orderings present identical canonical input, so the full-SDDP run cannot detect an
ordering bug even in principle — while consuming slow-test budget. Declaration-order
invariance is a **sort contract**: it should be a unit test on `build()` comparing the
sorted vectors, not a training run. (This matters for §2: the determinism guard the golden
hashes lean on must be made real, not tautological.)

### H — Raw `#[ignore]` benchmark-as-test (LOW)

`cut_selection_kernel_perf.rs` uses bare `#[ignore]` (not the `slow-tests` gate), so it is
invisible to both CI modes — a perf benchmark parked as a test. Belongs in `benches/` or
under the uniform slow-test gate.

## 4. Remediation & sequencing

Principle: biggest maintenance-cost reduction at lowest risk first; coverage never
decreases; every phase leaves the full feature-matrix test run green before the next.

| Phase | Smells  | Action                                                                                                                                                                                     | Risk    | Gain                                                                                        |
| ----- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- | ------------------------------------------------------------------------------------------- |
| 1     | D, H, E | Retire `EXPECTED_HASHES`/`parity_baselines_unchanged.rs`; fix the raw `#[ignore]`; write the tier hierarchy (§5) into `tests/common/` docs                                                 | none    | foundation; guard-on-guard gone; convention documented                                      |
| 2     | C       | Drop the convergence trajectory from `compute_parity_hash`; add `assert_cost(final_lb, …)` behavioral coverage for the affected cases; one-time re-baseline; designate the ~5 golden cases | low     | algorithmic improvements stop failing the golden hashes; backend-agnostic LB coverage added |
| 3     | F       | Merge the 8 K-parameterized anticipated binaries into 3 (K-table), adopting `common/` helpers                                                                                              | low     | −5 binaries; forces `common/` adoption in that group                                        |
| 4     | B       | Centralize `StubComm` + builders + `build_setup_for_case` in `tests/common/`; delete per-file copies; `use common::…` everywhere                                                           | low–med | field-addition blast radius ~65 → ~1                                                        |
| 5     | G       | Replace the tautological DOI training tests with a unit test on `SystemBuilder::build()` over permuted input                                                                               | low     | real DOI coverage; slow-test budget recovered                                               |
| 6     | A       | Consolidate remaining cobre-sddp `tests/*.rs` into ~grouped binaries (target ~20–25 from 71)                                                                                               | low     | link steps (the dominant test-build cost) cut substantially                                 |

The main risk across phases 3–4–6 is entity-ID collision when merging test functions
that previously lived in isolated binaries — the K-parameterized tests already use
disjoint ID ranges, so verify ID disjointness per merge batch. Phases 1–2 deliver the
bulk of the felt maintenance relief and are independently shippable.

## 5. Testing philosophy

### 5a. Cobre-specific pillars

- **C1 — The parity hash guards final output, not the algorithmic path.** It covers
  what the user observes (cut pool, simulation primal/dual/equipment/cost), never the
  iteration count or convergence trajectory. Convergence-speed improvements do not
  re-baseline.
- **C2 — Five golden cases; the rest behavioral.** ~5 deliberately feature-combined
  cases carry the bit-exact hash; all others assert `LB≈UB`, known cost (to documented
  tolerance), balance, and feature dispatch values — backend-agnostically.
- **C3 — LP structural tests are a first-class tier.** A wrong column/row count can
  produce numerically smooth but wrong bounds a behavioral tolerance may miss;
  `template_integration.rs`-style structural assertions catch builder regressions
  early.
- **C4 — Analytical-derivation tests are the gold standard for new correctness claims.**
  A new cut type or constraint starts with a closed-form expected coefficient
  (`anticipated_backward_cut`-style), before any deterministic case.
- **C5 — Feature interactions need explicit coverage.** Each feature ships at least one
  test exercising it alongside its nearest neighbours in the LP column layout. Dormant
  bugs come from untested feature combinations (discount×anticipated; nonuniform-block
  extraction).
- **C6 — Declaration-order invariance is a sort contract, not a convergence contract.**
  Test `SystemBuilder::build()` over permuted input and assert identical canonical
  order — a unit test, not a full training run.
- **C7 — Backend parity at the behavioral tier; bit-exact only for golden cases.**
  HiGHS and CLP are mathematically equivalent but differ at the FP-trajectory level;
  tolerance assertions cover both with one check, the bit-exact hash is necessarily
  one-backend.
- **C8 — Test output is `TempDir`, never the fixture tree.** No test writes into
  `examples/deterministic/*/output/`; generated artifacts self-delete.

### 5b. Generic Rust pillars (proposed for the cross-project rule)

These are not Cobre-specific and are candidates for the global Rust testing rule
(placement to be decided by the owner — see "Open decision" below):

- **G1 — Unit tests live in the module** (`#[cfg(test)] mod tests`); `tests/` is for
  genuinely crate-external behavior. Module tests also build faster and add no binary.
- **G2 — Integration test binaries are expensive by default.** Each `tests/*.rs` links
  its own executable (worse when it links C/C++); group related tests into one file.
  A new test _binary_ needs justification proportional to its link cost.
- **G3 — Shared test infrastructure lives in one `tests/common/` (or a test-support
  crate) and is used universally.** Adding a field to a shared type must be O(1), not
  O(files).
- **G4 — Property-based tests for ordering/commutativity claims** (`proptest`): generate
  random orderings, assert the canonicalization is invariant — executable documentation,
  not a tautology.
- **G5 — Benchmarks are not tests.** Performance work lives in `benches/` (`criterion`),
  never a bare `#[ignore]` test the CI never runs.
- **G6 — Slow tests are gated uniformly** (`#[cfg_attr(not(feature = "slow-tests"), ignore = …)]`).
  Exactly two CI modes (fast, full); a test outside both is invisible and therefore dead.

## 6. Non-negotiables for the migration

- Coverage never decreases — every step is coverage-neutral or coverage-improving;
  structural reorganizations change no assertion.
- Each phase ends green on the full feature matrix
  (`cargo test --workspace --features "mpi numa shared-memory serde schema slow-tests flatc-conformance test-support"`)
  before the next begins. Partial states with ID collisions or missing `mod common;`
  are not acceptable checkpoints.

## Rule placement (resolved)

- **§5a Cobre pillars → project rule** `.claude/rules/testing.md` (path-scoped to
  `crates/**/tests/**/*.rs` and `examples/deterministic/**`).
- **§5b generic pillars → global rule** `~/.claude/rules/rust.md` Testing section,
  so every project inherits the pyramid / one-builder / few-binaries discipline.

`.claude/rules/testing.md` points at the global rule for the generic pillars and
carries only the Cobre-specific contract.
