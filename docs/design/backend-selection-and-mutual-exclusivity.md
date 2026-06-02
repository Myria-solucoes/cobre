# Solver Backend Selection & Feature Mutual Exclusivity

> **Status**: Draft / Proposed (2026-06-01). Not yet implemented. This document
> scopes the follow-up work to the now-landed CLP backend: making the `highs`
> and `clp` solver features **mutually exclusive** (one solver per binary,
> enabling `clp` ignores `highs`) via **compile-time selection**, wiring backend
> **selection** through the workspace, and **completing the CLP capability
> surface** (C++ shim, dual-vs-primal, incremental mutation). It is a design
> starting point, not a plan — it records the current state, the obstacles, the
> settled decisions, and an explicit pending-work checklist.
>
> **Settled decisions (2026-06-01):**
>
> 1. Selection mechanism = **Option A (compile-time type alias)**; one solver per
>    binary, `clp` overrides `highs`. Options B/C below are retained only as
>    reference for the `cobre-comm` sibling.
> 2. The CLP capability items (C++ shim, dual-vs-primal, incremental mutation)
>    are **IN SCOPE** for this follow-up — they MUST be addressed, not deferred.
> 3. Generalizing the pattern to `cobre-comm` backends is a **sibling plan**, not
>    part of this follow-up; the design here is kept clean so its ideas transfer.

## Summary

`cobre-solver` now ships two LP backends:

- **HiGHS** behind the `highs` feature (in `default`, on by default) — `HighsSolver` / `HighsProfile`.
- **CLP (COIN-OR)** behind the `clp` feature (off by default) — `ClpSolver` / `ClpProfile`.

Today they **coexist additively**: both can be enabled at once, `cargo
--all-features` builds both, and the rest of the workspace always links HiGHS.
`ClpSolver` is a complete, conformance-validated `SolverInterface` drop-in — it
reproduces HiGHS's results exactly on the shared fixtures (objective `100`,
primals `(6, 0, 2)`, row duals `[-100, 50]`, `add_rows → 162`, warm-start
roundtrip → `100`).

The desired end state is the opposite of additive: a build selects **exactly
one** LP backend, `clp` overrides `highs`, and the choice flows through to
training/simulation, the CLI, the Python bindings, and the output metadata. This
could not be done while landing CLP because it requires the whole workspace to
stop hard-referencing HiGHS — which is only safe once a second backend exists
and is proven equivalent (both now true).

## Foundational requirement: per-backend profile richness

This requirement is the reason `HighsProfile` is a deliberate, HiGHS-specific
struct (not a shared/generic profile), and it constrains every option below. It
**must be preserved** by the selection design.

**Why per-backend profiles exist.** A solver profile's purpose is to let the
algorithm developer (here, the author of `cobre-sddp`) set _any_ attribute the
underlying solver exposes, so they can tune the LP backend for maximum SDDP
performance. The top priorities are **performance and flexibility**, not
uniformity. Different backend solvers expose **different parameters** to the
end user:

- They differ in **which parameters exist** (HiGHS's `simplex_dual_edge_weight_strategy`,
  `dual_simplex_cost_perturbation_multiplier`, `rebuild_refactor_solution_error_tolerance`,
  … have no CLP equivalent; CLP's perturbation/scaling/pricing-object knobs are
  shaped differently).
- They differ in **value** (the same conceptual setting takes different
  ranges/enumerations per solver).
- They differ in **datatype and how the value is passed** — HiGHS is configured
  through name-keyed option setters of distinct types (a value may be a string
  `"1"`, an integer `1`, a `bool`, or a `double`, depending on the option), while
  CLP uses typed C++ setters (`setPerturbation(int)`, `scaling(int)`,
  `setPrimalTolerance(double)`, the `ClpDualRowSteepest` pricing object, …). The
  same logical "1" is a string for one option and an integer for another.

Because of this, `HighsProfile` is hand-authored so the `cobre-sddp` developer
can pick the **exact** set of HiGHS options (and their per-phase values) that
best tune the algorithm on the HiGHS backend. `ClpProfile` is independently
authored for CLP and may carry a **different set of parameters** — different in
value, and even in _which_ parameters are present. This is precisely why
`SolverInterface::Profile` is an **associated type**, not a shared concrete
struct or a lowest-common-denominator option bag: the associated-type design
gives each backend its **full** native parameter surface with zero loss.

**Hard requirements (the follow-up must honor all of these):**

- **R-P1 — Full native surface, per backend.** Each backend defines its own
  `Profile` struct exposing its native parameter set. There is no requirement
  that profiles share fields, cardinality, datatypes, or semantics across
  backends.
- **R-P2 — No lowest-common-denominator collapse.** The chosen selection design
  (Option A) MUST NOT merge the profiles into a shared/unified profile or a
  common subset. `ActiveProfile` resolves, per build, to the _concrete_ backend
  profile at full richness. (If a runtime enum were ever used — Option B, for
  the `cobre-comm` sibling — each variant must carry the full native profile, not
  a flattened intersection.)
- **R-P3 — Per-phase tuning is itself backend-specific.** The forward / backward
  / simulation profile **constants** are defined per backend, of that backend's
  profile type, with independently chosen parameters and values. The _only_
  thing abstracted across backends is the **phase identity** (which phase is
  running) — never the parameter content. There is no shared "tuning" type that
  flattens values.
- **R-P4 — Datatype fidelity.** A profile field carries its value in the datatype
  the solver's setter expects, and `apply_profile` issues the correctly-typed
  call (HiGHS: the right string/int/bool/double option setter by name; CLP: the
  right typed C++/FFI setter). The abstraction must not force a single datatype.
- **R-P5 — Independent evolution.** Adding, removing, or retyping a parameter in
  one backend's profile must not require touching the other backend's profile or
  any shared type. Backends grow their parameter surfaces independently.

**How this shapes the chosen design (Option A).** `ActiveProfile` is the concrete
backend `Profile` (full surface). The per-phase constants live in a
backend-specific location — each backend's module supplies its own
`FORWARD_PROFILE` / `BACKWARD_PROFILE` / `SIMULATION_PROFILE` of _its_ profile
type — selected at compile time alongside `ActiveProfile`. The backend-agnostic
piece is only the `Phase` enum and a `Phase::profile() -> ActiveProfile` lookup
(e.g. a small `PhaseProfiles` trait the profile types implement with
`fn forward() -> Self` / `backward()` / `simulation()`, returning that backend's
own full profile). `solver_phase.rs` references `ActiveProfile` and the active
backend's phase constants; it never names a field set. See O3 and the per-phase
open question below.

## Why this is its own effort

Two prerequisites had to be met first; both are now satisfied:

1. **A complete, proven second backend.** HiGHS cannot be made removable until
   something can replace it. `ClpSolver` is now a full `SolverInterface`
   implementor and is conformance-validated against the HiGHS fixtures.
2. **Confidence that swapping backends is behaviorally sound.** The dual-sign
   probe confirmed CLP needs no normalization; CLP and HiGHS agree on the
   reference LPs.

What remains is the _decoupling_ work — removing the workspace's hard
dependency on the concrete HiGHS types — plus the _enforcement_ and _wiring_.
That work is invasive (it touches `cobre-sddp`'s generic bounds, the CLI, the
Python bindings, the test suite, and CI) and deserves its own plan.

## Current state (what the CLP plan delivered)

- `cobre-solver`: `SolverInterface` trait with an associated `type Profile`;
  two concrete backends (`HighsSolver`/`HighsProfile`, `ClpSolver`/`ClpProfile`),
  each gated by its feature; `ProfiledSolver<S>` delta-tracks `S::Profile`.
- `default = ["highs"]`; `highs = []`; `clp = []` (additive; both may be on).
- `ClpProfile` is C-API-only and **dual-only** this round (no primal/IPM
  selection — `cobre_clp_primal` was descoped; perturbation defaults to `102`
  = off for SDDP safety).
- CLP mutations (`add_rows` / `set_*_bounds`) use a **Rust-retained model + full
  `Clp_loadProblem` reload** (Option A), because the vendored CLP C API exposes
  no incremental row/bound mutation. Warm-start across bound patches is
  therefore weakened for CLP versus HiGHS.
- Licensing: HiGHS = MIT, Clp + CoinUtils = EPL-2.0 (documented in
  `THIRD_PARTY_NOTICES.md`); EPL deps build only under `clp` (off by default).

## The obstacles to resolve

### O1. Cargo features are additive; CI runs `--all-features`

Cargo's feature unification means `highs` and `clp` can always be requested
together, and the project convention runs `cargo … --all-features` for
check/test/clippy/docs everywhere. True mutual exclusivity needs either a
`compile_error!` when both are enabled (which then makes `--all-features` fail)
or a **priority rule** (`clp` wins, `highs` auto-disabled). Whichever is chosen,
the `--all-features` convention must be reconciled (e.g. per-feature CI jobs for
`cobre-solver`, or a workspace-level "pick one solver" matrix).

### O2. The workspace hard-references the concrete HiGHS types

The decoupling surface (discovered while landing CLP):

- **`cobre-sddp`** bounds many generics on `S: SolverInterface<Profile = cobre_solver::HighsProfile>`
  — in `training.rs`, `forward.rs`, `backward.rs`, `forward_pass_state.rs`,
  `backward_pass_state.rs`, `workspace.rs`, `simulation/pipeline.rs`,
  `lower_bound.rs`, `setup/orchestration.rs`.
- **`solver_phase.rs`** defines the per-phase profiles `FORWARD_PROFILE`,
  `BACKWARD_PROFILE`, `SIMULATION_PROFILE` as concrete `HighsProfile` constants,
  and `Phase::profile()` returns `HighsProfile`.
- **`cobre-cli`** (`commands/run.rs`) constructs `HighsSolver::new` directly for
  both the training factory and the simulation workspace pool, and writes a
  hardcoded `OutputContext.solver = "highs"` + `cobre_solver::highs_version()`.
- **`cobre-python`** (`src/run.rs`) does the same (constructs `HighsSolver::new`,
  hardcodes `"highs"` metadata).
- **`cobre-solver` tests + `baking.rs` MockSolver** reference `HighsSolver` /
  `HighsProfile` unconditionally (see O4).

### O3. Per-phase profiles are backend-specific

`HighsProfile` and `ClpProfile` are distinct structs with different fields, and
per the foundational requirement (R-P1/R-P3) they stay that way — the per-phase
tuning is genuinely backend-specific in both values and parameter set. The SDDP
per-phase constants must therefore be parameterized over the _active_ profile
type at compile time, with each backend supplying its own
`FORWARD/BACKWARD/SIMULATION_PROFILE` of its own profile type. The "cross-backend
tuning intent" is **not** flattened into a shared value representation (that
would violate R-P2/R-P3); the only backend-agnostic abstraction is _phase
identity_ — e.g. a `PhaseProfiles` trait with `forward()/backward()/simulation()`
returning `Self`, implemented independently by each profile type. See
"Foundational requirement: per-backend profile richness".

### O4. Test suite + MockSolver hard-reference HiGHS (known limitation)

`cargo test --no-default-features --features clp` (HiGHS off) does **not**
compile today: `conformance.rs`, the HiGHS-specific probe/smoke tests, and
`baking.rs`'s `MockSolver` (`type Profile = crate::HighsProfile`) reference HiGHS
types ungated. The library builds clp-only fine; only the test suite assumes
HiGHS. This is a direct symptom of O2 and is the smallest concrete decoupling
task — a good first step.

## Design options for the selection mechanism

**Decision: Option A (compile-time type alias) is chosen** — one solver per
binary, enabling `clp` ignores `highs`. Options B and C are recorded below as
the alternatives considered; they are **not** used for the LP solver backend,
but their runtime-dispatch shape is deliberately kept here because the
`cobre-comm` sibling plan (see "Out of scope") may adopt that model for the
communication backends.

### Option A — Compile-time type alias — **CHOSEN**

`cobre-solver` exposes `pub type ActiveSolver = …;` and `pub type ActiveProfile = …;`
selected by feature (`highs` → HiGHS, `clp` overrides → CLP). `cobre-sddp`,
`cobre-cli`, `cobre-python` reference the aliases instead of the concrete HiGHS
types. Per-phase profiles become associated constants/functions on a trait the
profile types implement, so `solver_phase.rs` is generic over `ActiveProfile`.

- **Pros**: true one-solver-per-binary; zero runtime dispatch; smallest runtime
  surface; matches the intent.
- **Cons**: cannot ship a single binary offering both; CI must build a
  per-solver matrix; phase-profile constants must be re-expressed generically.

### Option B — Runtime enum dispatch (alternative; reference for the `cobre-comm` sibling)

A `SolverBackend` enum (`Highs(..)` | `Clp(..)`, variants feature-gated) with a
unified `SolverProfile` enum; a `--solver` flag / env var picks among
compiled-in backends, exactly as `cobre-comm`'s `CommBackend` + `create_communicator()`
do for MPI vs local.

- **Pros**: one binary can offer both; runtime choice; precedent in the repo.
- **Cons**: this is _not_ "mutual exclusivity" — both are compiled in; adds an
  enum-dispatch layer; the unified profile enum must carry both backends'
  fields. Conflicts with the stated "clp ignores highs" intent.

### Option C — Hybrid (alternative; feature gates availability, runtime picks among available)

`cobre-comm`'s actual model: the feature decides whether a backend is _compiled
in_; a runtime selector picks among those present. Mutual exclusivity is then a
_build profile_ choice, not a hard `compile_error!`.

**What choosing Option A entails** (the committed shape for this follow-up):

- `cobre-solver` exposes `ActiveSolver` / `ActiveProfile` type aliases resolved
  by feature (`highs` → HiGHS; `clp` overrides → CLP).
- `cobre-sddp`, `cobre-cli`, `cobre-python` reference the aliases, never the
  concrete HiGHS types.
- Per-phase profiles (`solver_phase.rs`) are re-expressed generically over the
  active profile type via a small `PhaseProfiles`-style trait that abstracts only
  **phase identity** (`forward()/backward()/simulation()` returning `Self`); each
  backend supplies its own full-richness constants (R-P3), so the values/fields
  are never flattened across backends.
- Mutual exclusivity is enforced at compile time, and the `--all-features` CI
  convention is reconciled with a per-solver CI matrix (see O1).

Options B and C remain documented because the **`cobre-comm` sibling plan**
(communication-backend selection) is a better fit for runtime dispatch — the
local backend is always present and MPI is detected at launch — so its design
will likely draw on B/C rather than A. Keeping all three here lets the sibling
reuse this analysis.

## Pending work checklist (raw material for the plan's epics)

1. **Decouple `cobre-sddp` from `HighsProfile`** — replace the
   `Profile = cobre_solver::HighsProfile` bounds with the active/abstract
   profile across the nine listed sites; make `solver_phase.rs` phase profiles
   generic over the active profile (O2, O3).
2. **Introduce the selection mechanism in `cobre-solver`** — `ActiveSolver`/
   `ActiveProfile` aliases (Option A) or the `SolverBackend`/`SolverProfile`
   enums (Option B), feature-driven.
3. **Enforce mutual exclusivity** — `compile_error!` on `highs`+`clp` together,
   or `clp`-overrides-`highs` priority; and **reconcile the `--all-features` CI
   convention** (per-solver CI matrix / per-feature jobs for `cobre-solver`).
4. **Fix the test suite + MockSolver** (O4) — gate the HiGHS-specific tests
   behind `feature = "highs"`; give `MockSolver` a backend-agnostic `Profile`.
   This is the recommended first, self-contained step.
5. **Wire selection through the CLI** — a `--solver {highs|clp}` flag (or the
   build feature), routed to the solver factory and the workspace pool; make
   `OutputContext.solver` / `solver_version` reflect the _active_ backend
   (currently hardcoded `"highs"`) — honoring the Python-parity rule.
6. **Wire selection through `cobre-python`** — expose the solver choice and
   write the matching metadata; keep CLI/Python output parity.
7. **CLP-specific parity baselines** — switching solvers legitimately changes
   numerical results (different simplex), so CLP needs its own deterministic
   parity baselines (the D-case `parity_hash` family), not HiGHS's.
8. **CLP C++ shim for the rich knobs** — a `clp_wrapper_cpp.cpp` object (mirroring
   the existing `highs_wrapper_cpp.cpp`) exposing `ClpDualRowSteepest` pricing
   modes, factorization frequency, and `markHotStart`/`solveFromHotStart`; extend
   `ClpProfile` + the FFI to drive them.
9. **CLP dual-vs-primal selection** — add `cobre_clp_primal` to the C-wrapper
   triple (`clp_wrapper.{h,c}` + `clp_ffi.rs`) and a `ClpProfile` algorithm field;
   `solve` dispatches to dual or primal accordingly.
10. **CLP incremental mutation** — replace the retained-model full `Clp_loadProblem`
    reload in `add_rows` / `set_*_bounds` with native incremental calls
    (`Clp_addRows`, `Clp_chgRowLower/Upper`, `Clp_chgColumnLower/Upper`) so CLP
    retains its factorization/basis across bound patches (warm-start parity with
    HiGHS). Re-validate the conformance/warm-start tests against the new path.

(The `cobre-comm` backend generalization is **out of scope** for this follow-up —
see below.)

## Required CLP capability work (IN SCOPE for this follow-up)

These were deliberately left out of the CLP coexistence round and **must be
completed in this follow-up** (checklist items 8–10). They are the detail and
rationale behind those items:

- **C++ shim for the rich CLP knobs** — the plain `Clp_C_Interface.h` surface
  cannot reach the pricing-object API or the hot-start loop, so this needs a
  `clp_wrapper_cpp.cpp` compiled as a separate C++17 object (exactly the pattern
  `highs_wrapper_cpp.cpp` already uses in `build.rs`). It exposes
  `ClpDualRowSteepest` pricing modes (the SDDP-relevant "full DSE" pin),
  factorization frequency, and `markHotStart`/`solveFromHotStart` for the
  per-opening backward-pass loop. `ClpProfile` and the FFI grow to drive them.
  See `ideas/clp-solver-options-sddp.md` for the option semantics and the
  recommended SDDP-tuned configuration.
- **Dual-vs-primal selection** — add `cobre_clp_primal` to the C-wrapper triple
  (`clp_wrapper.{h,c}` + `clp_ffi.rs`) and a `ClpProfile` algorithm field
  (currently dual-only); `solve` dispatches to `cobre_clp_dual` or
  `cobre_clp_primal`. This completes the "C-API-coverable" profile surface that
  was trimmed when the wrapper shipped dual-only.
- **Incremental CLP mutation** — replace the retained-model full-reload
  `add_rows` / `set_*_bounds` (the "Option A" stopgap from the CLP plan) with
  native incremental calls (`Clp_addRows`, `Clp_chgRowLower/Upper`,
  `Clp_chgColumnLower/Upper`). This restores warm-start fidelity across bound
  patches — the SDDP backward pass patches state bounds per opening, so the
  full-reload path discards the factorization on every patch, which is a real
  per-opening cost. The retained-model buffers can be kept as the source of
  truth and reconciled, or dropped once incremental mutation is native.

## Out of scope — `cobre-comm` backend generalization (sibling plan)

Applying the same mutually-exclusive feature-backend pattern to the
communication backends (`mpi` / `local` / `numa` / `shared-memory`) is a
**separate sibling plan**, not part of this follow-up. It is called out here
because the two efforts share a shape, and the analysis in this document is kept
clean so the sibling can reuse it:

- The selection-mechanism trade-offs (Options A/B/C above) transfer directly;
  the sibling will likely lean toward **runtime dispatch (Option B/C)** because
  `cobre-comm` already has that exact model (`CommBackend` enum +
  `create_communicator()` picking local vs MPI at launch), whereas the LP solver
  chose compile-time (Option A).
- The `--all-features`-vs-mutual-exclusivity reconciliation (O1) and the
  "decouple the workspace from one concrete backend type" pattern (O2/O3) are the
  same problems in a different crate.

When the sibling is planned, start from this document's Options section and O1.

## Open design questions

Resolved (see "Settled decisions" in the header): selection mechanism = Option A
(compile-time); CLP capability items = in scope; `cobre-comm` = sibling plan.
Still open for the design phase:

- **Mutual-exclusivity enforcement + `--all-features`.** With Option A, how do
  `highs` and `clp` reject being enabled together — a hard `compile_error!`
  (then `--all-features` must be dropped for `cobre-solver` in favor of a
  per-solver CI matrix), or a `clp`-overrides-`highs` priority (then
  `--all-features` builds CLP-only and HiGHS is silently off)? This drives the CI
  layout for the whole workspace.
- **Hot-start vs the persistent-instance pattern.** Does CLP's
  `markHotStart`/`solveFromHotStart` (item 8) compose with the per-(worker,
  stage) persistent solver instances and the slot-tracked basis reuse the SDDP
  backward pass relies on — and does it preserve declaration-order /
  bit-for-bit determinism? Needs a determinism check before adoption.
- **Incremental mutation & determinism.** When `add_rows` / `set_*_bounds`
  switch to native incremental calls (item 10), do the retained-model buffers
  stay as the source of truth (reconciled each call) or get dropped? Either way,
  the result must remain declaration-order-invariant and match the conformance
  fixtures.
- **Primal exposure (item 9).** Is primal simplex ever _selected_ by the SDDP
  phases, or only exposed on `ClpProfile` for completeness? (Default stays dual.)
- **Per-phase profile abstraction — exact shape.** The principle is settled
  (R-P3: a `PhaseProfiles` trait abstracts only phase identity, returning each
  backend's own full profile; no value flattening). Open only at the mechanics
  level: assoc-consts vs methods on the trait, and where each backend's
  `FORWARD/BACKWARD/SIMULATION` constants live so they are selected with
  `ActiveProfile` at compile time.

## Prerequisites already satisfied

- `ClpSolver` is a complete `SolverInterface` drop-in, conformance-validated
  against the HiGHS fixtures.
- CLP dual-sign convention resolved (no negation; matches the canonical
  convention).
- EPL-2.0 vendoring documented and signed off.
- `highs` already exists as a (default-on) feature with the build gated behind
  it — half of the "select one backend" machinery is in place.

## References

- LP-solver option/cost-model research: `ideas/clp-solver-options-sddp.md`,
  `ideas/highs-solver-options-sddp.md`.
- The communication-backend selection precedent to mirror for Option B/C:
  `crates/cobre-comm/src/factory.rs` (`CommBackend`, `create_communicator`),
  `crates/cobre-comm/src/lib.rs` (`#[cfg(feature = "mpi")]` module gating).
- The `SolverInterface` contract and the two backends:
  `crates/cobre-solver/src/trait_def.rs`, `src/highs.rs`, `src/clp.rs`,
  `src/profiled.rs`.
