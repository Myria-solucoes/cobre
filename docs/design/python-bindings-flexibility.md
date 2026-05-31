# Python Bindings — Flexibility & Programmatic Control

> **Status**: Draft / Proposed (2026-05-30). Not yet implemented. This
> document proposes a set of additive features for the `cobre-python` crate
> that move it from a thin CLI clone toward a first-class programmatic API for
> single-node, interactive, and parametric workflows (scripts, notebooks,
> orchestration frameworks). No code has been written; this is a design for
> review.

## Summary

Today `cobre-python` exposes four submodules (`io`, `model`, `run`, `results`)
that closely mirror a subset of the CLI. Computation runs with the GIL released
and the crate independently re-populates every metadata carrier the CLI writes,
satisfying the **Python-parity hard rule** (every output file the CLI writes is
also written by the bindings). That foundation is solid.

The crate's ceiling is structural, not cosmetic: the API treats Python as a
front-end that _invokes_ Cobre, never as a host that _drives_ it. Four limits
follow from that framing:

1. **`run()` is monolithic and disk-bound** — load → train → simulate → write is
   one opaque call; you cannot train, inspect the policy in memory, then
   simulate without a disk round-trip and a redundant `prepare_stochastic`.
2. **Configuration is disk-only** — every knob lives in `config.json`, so a
   parameter sweep or seed study requires writing N case directories.
3. **The run is silent** — nothing surfaces to Python until `run()` returns; no
   convergence callback, no progress hook, no cooperative early stop, no Ctrl-C
   until an iteration boundary.
4. **Results are stringly-typed `dict`s** — no typed objects, no policy
   evaluation, no programmatic introspection of cuts or stochastics.

This document proposes seven tracks (A–G) that lift those limits while
preserving the three invariants that make the current crate correct:
single-process only (no MPI), GIL released during computation, and CLI/Python
output parity.

## Design principles (invariants every track must hold)

- **P1 — Single-process only.** No track may initialize MPI or depend on
  `ferrompi`. Distributed execution remains `mpiexec cobre` as a subprocess.
- **P2 — GIL released during computation.** All Rust solve work runs inside
  `py.detach(...)`. Any Python interaction (callbacks) happens at iteration
  boundaries via a GIL reacquisition, never inside the hot LP loop.
- **P3 — Output parity is the anchor, in-memory modes are additive.** The
  file-writing path stays bit-for-bit identical to the CLI. New "don't write,
  return in memory" modes are _additions_; they never become the only way to
  get an output, and they never silently drop a file that a writing run would
  have produced.
- **P4 — No `Box<dyn Trait>`, no hot-path allocation, no `.unwrap()` in library
  code** — the workspace hard rules apply to binding code too (the crate already
  sets `unwrap_used = "deny"`).
- **P5 — Determinism preserved.** Programmatic config overrides and in-memory
  lifecycles must produce identical results to the equivalent on-disk config;
  declaration-order invariance and seed handling are unchanged.

## Ground-truth references

This design is anchored on the following existing types (verified against the
tree as of 2026-05-30):

- `cobre_io::Config` (`crates/cobre-io/src/config/mod.rs`) — `#[derive(Clone,
Deserialize, Serialize)]`, `#[serde(deny_unknown_fields)]`, nested sections
  `modeling`, `training`, `upper_bound_evaluation`, `policy`, `simulation`,
  `exports`, `estimation`, `energy`, each with `#[serde(default)]`.
- `cobre_core::TrainingEvent` (`crates/cobre-core/src/training_event.rs`) — the
  enum already sent on `StudySetup::train`'s `Option<Sender<TrainingEvent>>`.
  Relevant variants: `IterationSummary`, `ConvergenceUpdate`,
  `ForwardPassComplete`, `BackwardPassComplete`, `TrainingStarted/Finished`,
  `SimulationProgress`, `SimulationStarted/Finished`.
- `StudySetup::{train, simulate, create_workspace_pool, from_broadcast_params}`
  (`crates/cobre-sddp/src/setup/orchestration.rs`, `setup/mod.rs`) —
  `StudySetup` is `Send`, holds **no** solver handles (solvers arrive via a
  `solver_factory` closure), and is therefore safe to store in a long-lived
  `#[pyclass]` across calls. `train` takes `&mut self` and a
  `shutdown_flag: Option<&Arc<AtomicBool>>`.
- Error enums: `cobre_io::LoadError` (6 variants), `cobre_sddp::SddpError` (8
  variants incl. `Infeasible`, `Validation`, `Simulation`), `cobre_io::OutputError`
  (4 variants).

These types already expose everything tracks A–G need; the work is binding
plumbing, not new core algorithms.

---

## Track A — In-memory configuration overrides

**Goal.** Run a case with config knobs changed at call time, with no disk writes
to the case directory. Unlocks parameter sweeps, seed studies, and convergence
experiments — the dominant scientific-Python workflow.

### API

```python
cobre.run.run(
    case_dir,
    output_dir=None,
    threads=None,
    skip_simulation=None,
    config_overrides=None,   # NEW: Mapping[str, Any]
)
```

`config_overrides` is a flat dotted-key map applied on top of the parsed
`config.json`:

```python
cobre.run.run(
    "examples/1dtoy",
    output_dir="out/seed7",
    config_overrides={
        "training.max_iterations": 50,     # via stopping_rules, see note
        "training.tree_seed": 7,
        "simulation.num_scenarios": 500,
        "simulation.enabled": True,
    },
)
```

The same parameter is also accepted by `cobre.io.validate(...)` and by the
`Study` constructor in Track B, so overrides are validated identically wherever
they enter.

### Mechanics

`Config` is `Deserialize + Serialize`, so the override path is a JSON merge:

1. Parse `config.json` → `serde_json::Value` (not yet into `Config`).
2. For each dotted key, set the value into the `Value` tree, creating
   intermediate objects as needed (deep merge — `policy.checkpointing.compress`
   must not clobber siblings).
3. Re-deserialize the merged `Value` into `Config`. Because `Config` uses
   `#[serde(deny_unknown_fields)]`, a misspelled override key
   (`trainning.max_iterations`) fails loudly at this step rather than being
   silently ignored — a deliberate, valuable property.
4. Proceed through the existing `run_inner` path unchanged.

A new helper in `cobre-io` (e.g. `Config::with_overrides(value: &Value,
overrides: &Map) -> Result<Config, LoadError>`) keeps the merge logic in the
crate that owns the schema, so it tracks schema changes automatically and is
reusable by the CLI later (`cobre run --set training.tree_seed=7`).

**Note on `max_iterations`.** Iteration count is expressed through
`training.stopping_rules`, not a flat field. The override map must therefore
support setting array/object values (`"training.stopping_rules":
[{"max_iterations": 50}]`), not only scalars. The dotted-key setter assigns any
JSON value, so this falls out naturally; the docs will show the canonical
stopping-rule override.

### Parity & determinism

- No case-directory writes — overrides live only in the in-memory `Config`.
- The written `training/metadata.json` reflects the _effective_ config (post
  override), exactly as if the user had edited `config.json`. This keeps
  `cobre summary`/`report` honest about what actually ran.
- Determinism (P5) holds: same effective config ⇒ same results, regardless of
  whether the config came from disk or from a merge.

### Trade-offs

- **Chosen: flat dotted-key map.** Simple, JSON-serializable, sweep-friendly,
  and `deny_unknown_fields` catches typos. _Cost:_ array/nested edits are
  slightly verbose.
- _Rejected: a typed `RunConfig` Python class mirroring every field._ Highest
  discoverability, but it duplicates the entire config schema in PyO3 and must
  be kept in lockstep with `Config` forever — a standing parity liability. A
  `TypedDict` in the stubs (Track D) gives most of the discoverability at none
  of the maintenance cost.
- _Rejected: accept a full `dict` replacing the config._ Loses the base
  `config.json` and forces callers to restate mandatory fields.

---

## Track B — Decoupled lifecycle (`Study` / `Policy` objects)

**Goal.** Separate load, train, and simulate into composable steps that share
in-memory state, eliminating disk round-trips and enabling repeated simulation
against one trained policy.

### API

```python
study = cobre.Study("examples/1dtoy", config_overrides={...}, threads=4)

report = study.validate()          # same dict as cobre.io.validate
policy = study.train(on_iteration=cb)   # returns a Policy handle; writes training/* (parity)
results = study.simulate(policy, output_dir="out/")   # writes simulation/* (parity)

# Repeated simulation against the same policy, no retrain:
for seed in range(10):
    study.simulate(policy, output_dir=f"out/{seed}",
                   config_overrides={"simulation.tree_seed": seed})
```

`Study`, `Policy`, and `Results` are `#[pyclass]` handles. `Study` owns the live
`StudySetup` (and the parsed `System`/`Config`); `Policy` wraps the trained
`FutureCostFunction` + basis cache (or a path to a checkpoint); `Results` wraps
an output directory with lazy accessors (Track D).

### Mechanics

- `Study.__new__` runs the front half of today's `run_inner` (load_case,
  prepare_stochastic, hydro models, `from_broadcast_params`) once and stores the
  `StudySetup`. Because `StudySetup` is `Send` and holds no solver handle, it
  lives safely in the pyclass across calls.
- `Study.train(&mut self, ...)` calls `StudySetup::train` with a freshly built
  `HighsSolver` + `solver_factory`, releasing the GIL via `py.detach`. It writes
  the training artifacts (parity) and returns a `Policy` referencing the now
  trained `self.fcf` / basis cache held in the study.
- `Study.simulate(&self, policy, ...)` builds a workspace pool
  (`create_workspace_pool`) and runs `StudySetup::simulate`, reusing the
  in-memory policy with no checkpoint reload. Writes simulation artifacts
  (parity).
- The existing monolithic `run.run()` is reimplemented on top of `Study`
  (`Study(...).train(); .simulate()`), so there is exactly one execution path to
  maintain and parity is proven once.

### Parity & determinism (the hard part)

This is where P3 earns its keep. Two risks:

1. **In-memory simulate without re-reading the checkpoint** must produce
   identical results to the CLI's load-from-disk simulate. The CLI's
   simulation-only path reconstructs the FCF + basis cache from the checkpoint;
   the in-memory path reuses the live objects. These must be equivalent. A
   regression test will train + simulate in-memory and compare metadata against
   the CLI golden values (extending the existing
   `python_run_1dtoy_metadata_matches_cli_golden_values`).
2. **`Policy` as a path vs. as live state.** To avoid two code paths, `Policy`
   always carries enough to drive `simulate` directly from memory after a
   `train`; a `Policy.load(output_dir)` constructor reuses the _existing_ CLI
   checkpoint-reconstruction code (`read_policy_checkpoint` +
   `FutureCostFunction::from_deserialized` + `build_basis_cache_from_checkpoint`)
   so loaded and trained policies converge on the same simulate entry point.

### Trade-offs

- **Chosen: stateful `Study` object holding `StudySetup`.** Matches the
  scikit-learn `fit`/`predict` mental model, kills the disk round-trip, and is
  the single highest-value unlock. _Cost:_ largest new API surface; lifetime and
  `&mut`/`&` discipline at the PyO3 boundary; must guarantee the in-memory
  simulate equals the on-disk simulate (mitigated by the parity test above).
- _Rejected: free functions `train(setup)` / `simulate(setup, policy)` passing an
  opaque setup handle._ Less idiomatic in Python; same lifetime concerns without
  the ergonomic payoff.
- **Escalation flag.** Whether `train`/`simulate` should offer a
  `write=False` in-memory-only mode (returning results without touching disk) is
  an open question (see Open Questions) — it tensions against P3 and needs an
  explicit decision before implementation.

---

## Track C — Convergence / progress callback

**Goal.** Surface iteration-level progress to Python for live plots, `tqdm`
bars, logging, and cooperative early stopping — using the event channel that
already exists internally.

### API

```python
def on_iteration(event: dict) -> bool | None:
    # event = {"kind": "iteration", "iteration": 12, "lower_bound": ...,
    #          "upper_bound": ..., "gap": ..., "wall_time_ms": ...}
    print(event["iteration"], event["gap"])
    return False   # return True to request a graceful early stop

cobre.run.run("case", on_iteration=on_iteration)   # also on Study.train
```

The callback receives a dict (a `TypedDict` in stubs, Track D) per
`IterationSummary`/`ConvergenceUpdate` event. A truthy return requests a
cooperative stop at the next iteration boundary.

### Mechanics

`StudySetup::train` already accepts `Option<Sender<TrainingEvent>>` and
`shutdown_flag: Option<&Arc<AtomicBool>>`. Today the Python path drains the
channel into a `Vec` _after_ training returns. To stream live:

1. Create the `mpsc::channel::<TrainingEvent>()` and an
   `Arc<AtomicBool>` shutdown flag before the `py.detach` block.
2. Spawn a **drain thread** (std thread, not rayon) that loops on
   `event_rx.recv()`. For each event of interest, it reacquires the GIL via
   `Python::attach(|py| { ... })`, converts the event to a dict, and invokes the
   Python callback. If the callback returns truthy, it sets the shutdown flag.
3. Run `train` inside `py.detach` with the sender + `&shutdown_flag`. The solver
   loop checks the flag at iteration boundaries and exits gracefully (this is
   the same flag the CLI uses for SIGINT).
4. On `train` return, drop the sender, join the drain thread, then (as today)
   collect any remaining events for the convergence-record write so parity is
   unaffected.

Only boundary events (`IterationSummary`) cross into Python; high-frequency
`WorkerTiming`/`SimulationProgress` events are filtered unless explicitly
requested, keeping GIL reacquisition rare (P2).

### Ctrl-C / SIGINT

Because the GIL is released, Python's signal machinery cannot run during the
solve. The drain thread can periodically call `py.check_signals()` (under the
GIL it already holds for callbacks); on `KeyboardInterrupt` it sets the shutdown
flag, turning Ctrl-C into a graceful iteration-boundary stop instead of a
"queued until return" no-op. This directly fixes the limitation documented in
`run.rs`'s module docs.

### Trade-offs

- **Chosen: separate drain thread reacquiring the GIL at boundaries.** Keeps the
  hot loop GIL-free (P2) and reuses the existing event + shutdown plumbing.
  _Cost:_ a callback that raises must be handled (propagate as the run's error,
  after a clean shutdown); ordering between the drain thread and the final
  collection must be sequenced (drop sender → join → collect).
- _Rejected: callback invoked directly from the solver thread._ Would require
  acquiring the GIL inside the hot region — violates P2 and serializes workers.
- _Rejected: no early-stop, progress-only._ Leaves the most-requested
  interactive feature (stop a diverging run) on the table when the flag is
  already wired.

---

## Track D — Typed ergonomic layer & structured exceptions

**Goal.** Replace stringly-typed dicts and generic exceptions with discoverable,
checkable types, and add a thin pure-Python convenience layer.

### Packaging decision

The crate is currently a pure cdylib with a `stubs/` typing layer and no Python
source. This track introduces a **small pure-Python package layer** (a
`cobre/` source tree shipped alongside the compiled `_cobre` extension via
maturin's mixed Rust/Python layout). The compiled module is renamed to a private
`cobre._native` (or similar); the public `cobre` package re-exports it and adds
ergonomic wrappers. This is the standard maturin pattern and keeps the Rust
boundary minimal.

> **Parity implication.** P3 now spans three layers (Rust write path, Rust
> bindings, Python wrappers). The wrappers must be _pure presentation/typing_ —
> they may not own any output-writing logic, so the parity rule continues to be
> enforced entirely in Rust. The CLAUDE.md note for `write_outputs`/`run_inner`
> stays the single source of truth.

### Typed results

```python
results = cobre.run.run(...)          # returns RunResult (TypedDict today; dataclass later)
results.converged                      # typed attribute access
df = results.convergence_df()          # -> polars/pandas via the Arrow path
sim = results.simulation("hydros")     # -> pyarrow.Table
```

`RunResult`, `TrainingSummary`, `SimulationSummary`, `StochasticSummary`,
`ProvenanceReport` are declared as `TypedDict`s in the stubs first (zero runtime
cost, immediate IDE/mypy value), optionally promoted to dataclasses with helper
methods in the Python layer. The stale `run.pyi` (`-> dict[str, Any]`) and the
out-of-date `load_simulation` entity-type docstring are corrected here.

### Structured exception hierarchy

A `cobre.errors` module (Python-visible exception classes registered from Rust)
replaces today's flat `OSError`/`ValueError`/`RuntimeError` collapse:

```
CobreError(Exception)
├── ValidationError      # LoadError::{Schema,CrossReference,Constraint,Parse}, StudyParams config errors
├── CaseIoError          # LoadError::IoError, OutputError::IoError  (also subclasses OSError)
├── PolicyIncompatibleError  # LoadError::PolicyIncompatible, validate_policy_compatibility
├── SolverError          # SddpError::{Solver, Infeasible}
├── SimulationError      # SddpError::Simulation
└── OutputError          # OutputError::{Serialization, Schema, Manifest}
```

Mapping happens in one place (a `convert_error` matching `SddpError`/`LoadError`/
`OutputError` variants). `CaseIoError` multiply-inherits `OSError` so existing
`except OSError` code keeps working — backward compatible. `Infeasible { stage,
iteration, scenario }` carries its fields onto the Python exception so callers
can react programmatically (today it's an opaque `RuntimeError` string).

### Trade-offs

- **Chosen: TypedDicts now, optional dataclass/Python layer next.** Incremental:
  typing value lands without a runtime layer; the pure-Python package is added
  only where it pays (DataFrame helpers). _Cost:_ introduces Python source to
  maintain and a `_native` rename (one-time packaging change).
- _Rejected: dataclasses returned directly from Rust._ PyO3 can emit classes,
  but DataFrame/helper ergonomics are far cheaper to write in Python.
- **Backward-compat risk:** changing the exception types for existing calls is
  technically a breaking change. Mitigated by subclassing the current builtins
  (`CaseIoError(OSError)`, `ValidationError(ValueError)`) so `except`-clauses on
  the old types still catch. The crate is `Development Status :: 3 - Alpha`,
  so the window for this change is now.

---

## Track E — CLI-parity utilities

**Goal.** Close the remaining CLI/Python surface gap with low-effort, mostly
mechanical bindings.

### API

```python
cobre.schema.export(output_dir=".")        # cobre_io::schema::generate_schemas
cobre.templates.list() -> list[str]         # templates::available_templates
cobre.templates.scaffold(name, dest, force=False)
cobre.version_info() -> dict                # version, solver, comm, zstd, arch, build
cobre.results.report(output_dir) -> dict    # the `cobre report` JSON shape
cobre.results.summary(output_dir) -> str    # rendered `cobre summary` text
```

### Mechanics

- `schema.export`, `templates.*`, `version_info` wrap existing library calls /
  embedded data directly — trivial.
- `report` reuses `read_training_metadata` + `read_simulation_metadata` and
  assembles the exact JSON shape `cobre report` emits (hoisted `bounds`/`cost`
  keys), returned as a Python dict rather than printed.
- `summary` is the one with a wrinkle: the CLI's renderers
  (`print_training_summary`, etc.) live in `cobre-cli` and write to a terminal.
  Two options: (i) lift the pure formatting functions into a shared crate so
  both CLI and Python render identically, or (ii) have Python return the
  structured dict and leave rendering to the Python layer (Track D). Option (i)
  guarantees identical text; option (ii) is less coupling. **Recommendation:**
  return structured data from Rust (`report`) and render in the Python layer,
  treating the CLI's text as presentation that need not be byte-identical.

### Trade-offs

- Mostly mechanical; the only real decision is the `summary` rendering location
  above. Low risk, high "feels complete" value.

---

## Track F — Policy & stochastic introspection

**Goal.** Let Python analyze the trained policy and the stochastic model
quantitatively, not just dump raw cut dicts.

### API

```python
policy = cobre.results.load_policy("out/")   # already exists (raw dicts)

# NEW:
policy.evaluate(stage=3, state=[v0, v1, ...]) -> float        # FCF value at a state
policy.cut_matrix(stage=3) -> (intercepts: np.ndarray, coeffs: np.ndarray)  # (n_cuts,), (n_cuts, dim)

stoch = cobre.results.load_stochastic("out/")  # PAR coefficients, correlations as arrays
stoch.par_coefficients() -> np.ndarray
stoch.opening_tree(stage=3) -> np.ndarray
```

### Mechanics

- `policy.evaluate` computes `max_k(intercept_k + coeffs_k · state)` over the
  stage's active cuts — a pure read over the already-loaded FCF. (Respect the
  documented cut convention: stored `coefficients = dual`; the FCF value uses
  them directly as the gradient — see the project's Benders-convention note.)
- `cut_matrix` returns NumPy arrays (zero-copy where possible) instead of nested
  Python lists, for vectorized analysis/plotting.
- `load_stochastic` reads the stochastic export artifacts (already written when
  `exports.stochastic` is set) and surfaces PAR coefficients, seasonal stats,
  and opening trees as arrays.

### Trade-offs

- **Chosen: read-only analytical accessors over existing artifacts.** No new
  outputs, no new core algorithms — pure projection of data already on disk / in
  memory. _Cost:_ adds a NumPy dependency surface (optional, like the existing
  pyarrow soft-dependency: raise `ImportError` only when an array accessor is
  called without NumPy).
- _Deferred:_ writing _new_ policies from Python (constructing cuts) — out of
  scope; introspection is read-only for now.

---

## Track G — `init_rayon` per-call thread correctness

**Goal.** Honor the `threads` argument on every `run()` call within a process.

### Problem

`init_rayon` calls `rayon::ThreadPoolBuilder::build_global()`, which succeeds
**once per process**. In a Jupyter session the first `run(threads=k)` wins;
every later `run(threads=j)` silently falls back to the first pool. The crate's
own `init_rayon_falls_back_to_actual_count` test documents this. For an
interactive tool this is a surprising footgun.

### Fix

Replace the global pool with a **scoped pool** created per call:

```rust
let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build()?;
pool.install(|| run_inner(...))   // all rayon work inside run_inner uses this pool
```

`pool.install` confines the call's rayon work to the scoped pool, so each
`run()`/`Study.train()` respects its own `threads`. The pool is dropped at the
end of the call. This composes cleanly with Tracks B and C (each `Study.train`
gets its own pool).

### Trade-offs

- **Chosen: scoped `install` per call.** Correct per-call semantics, no global
  state. _Cost:_ a small pool-construction cost per call (negligible vs. a solve).
- _Note:_ fold this into whichever track ships first (it touches the same
  `run_inner` entry point).

---

## Cross-cutting concerns

### Parity test strategy

The existing `python_run_1dtoy_metadata_matches_cli_golden_values` is the
template. New parity guards:

- **A:** override path produces the same metadata as an equivalent edited
  `config.json`.
- **B:** in-memory `Study.train().simulate()` produces metadata bit-identical to
  the CLI golden values, _and_ to the monolithic `run.run()` (which is now built
  on `Study`).
- **B:** `Policy.load(dir)` then `simulate` equals train-then-simulate-in-memory.
- **C:** events streamed via callback match the rows written to
  `convergence.parquet` (count and values).

### Versioning / ABI

The crate uses `abi3-py312`. Adding submodules, functions, classes, and
exception types is ABI-compatible. The `_native` rename (Track D) is the only
packaging-visible change and is internal (public `import cobre` is unchanged).

### Dependency posture

Per the workspace "no unnecessary dependencies" rule: `pyarrow` and `numpy` stay
**soft** dependencies — imported lazily inside the accessor that needs them,
raising `ImportError` with a clear message if absent. No new mandatory runtime
dependency is introduced.

### Documentation

`book/` user docs gain a "Python API" chapter covering the `Study` lifecycle,
overrides, and callbacks. Per the no-narrative-docs rule, docs describe behavior,
not the track structure of this plan.

---

## Phasing & dependencies

```
G  ─┐ (fold into first shipped track; touches run_inner)
A  ─┼─► independent, highest value/lowest risk — ship first
E  ─┘   mechanical, parallelizable any time

C  ───► depends on nothing structural; reuses event channel — ship after A
B  ───► largest; reimplements run.run on top of Study; absorbs A (overrides in ctor) and C (callback in train)
D  ───► TypedDicts can land with A; pure-Python layer + exceptions land with/after B
F  ───► independent read-only layer; ship any time after results exist
```

Recommended order: **A + G + E (foundation)** → **C (callback)** → **B (lifecycle,
absorbing A/C)** → **D (typing + exceptions, now that the API shape is stable)** →
**F (introspection)**.

## Resolved decisions (settled 2026-05-30, prior to planning)

1. **In-memory-only modes (`write=False`) — DEFERRED.** `Study.train/simulate`
   always write their artifacts (parity-as-anchor, P3). A non-writing mode is out
   of scope for this plan; it will be revisited only if a concrete workflow
   demands it. Rationale: keeps exactly one execution path, so parity is proven
   once and never silently bypassed.
2. **`summary` rendering location — PYTHON LAYER.** Rust exposes structured data
   (`cobre.results.report(dir) -> dict`); human-readable rendering lives in the
   Python layer (Track D). The CLI's terminal text is treated as presentation and
   need not be byte-identical. Rationale: avoids coupling the CLI's `console`-based
   formatters into a shared crate for a non-essential cosmetic guarantee.
3. **Exception hierarchy — INTRODUCE NOW, SUBCLASS BUILTINS.** The new
   `cobre.errors` classes subclass the current builtins (`CaseIoError(OSError)`,
   `ValidationError(ValueError)`) so existing `except OSError`/`except ValueError`
   code keeps catching. No deprecation shim. Rationale: the crate is alpha
   (`Development Status :: 3 - Alpha`); subclassing preserves backward
   compatibility, so the change is non-breaking in practice and the window is now.
4. **Override key surface — FULL `Config` SURFACE.** `config_overrides` may target
   any field; `#[serde(deny_unknown_fields)]` rejects typos at merge time. No
   allow-list. Rationale: simpler, and every "load-bearing" field
   (`policy.mode`, `tree_seed`, stopping rules) is a legitimate sweep target;
   restricting them would cripple the primary use case. Structural safety is the
   re-deserialization step's job, not an allow-list's — an override that produces
   an invalid `Config` fails loudly with a `ValidationError`.
