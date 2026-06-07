# CLP factorization reuse in the backward opening loop

> **Status**: Proposed / investigation (2026-06-07). Not yet implemented. This
> note records a traced finding about an asymmetry between the HiGHS and CLP
> backends in the backward-pass opening loop, and proposes wiring CLP's
> already-delivered hot-start capability into the solve path to close (and
> possibly reverse) the measured HiGHS-faster gap. The hot-start trio it relies
> on is already implemented and determinism-verified in isolation — the work is
> integration, not new capability.
>
> **Companions:**
>
> - `docs/design/clp-capability-determinism.md` — the determinism contract and
>   the delivered (but unwired) CLP hot-start trio this plan depends on.
> - `docs/design/solver-tuning-results.md` — the tuning campaign that found
>   HiGHS ~11–15% faster and more robust than CLP.
> - `docs/design/backward-pass-performance-analysis.md` — backward-pass cost
>   structure.
> - `docs/design/backward-node-parallelism.md` — the larger per-(worker,stage)
>   persistent-instance direction this plan is a contained first step toward.

## TL;DR

In the backward pass, every trial point solves its `|Ω|` opening subproblems on
**one resident LP** with only RHS/bounds changing between openings, in fixed
order, on a single worker — the textbook "fixed matrix, change bounds, re-solve"
pattern.

- **HiGHS already exploits this.** Opening `ω=0` installs a canonical basis;
  openings `ω>0` call `solve(None)`, which installs no basis and lets HiGHS
  continue from its **retained internal factorization** (INVERT). No
  refactorization per opening.
- **CLP does not.** CLP's `solve()` always calls `Clp_dual(handle, 0)`, which
  runs `ClpSimplex::dual(ifValuesPass, startFinishOptions = 0)` — a full
  `createRim → factorize → solve → deleteRim` cycle **on every opening**. CLP
  warm-starts the basis _status_ but cold-starts the _factorization_ each solve.

For these LPs (thousands of rows, very sparse, warm basis ⇒ few dual pivots), the
factorization is likely the dominant per-opening cost, and the pivots are cheap.
So CLP performs `O×` the factorization work HiGHS does in the inner loop. This is
a prime suspect for the bulk of the measured HiGHS-faster verdict.

The fix already exists in the wrapper but is orphaned: the
`markHotStart`/`solveFromHotStart`/`unmarkHotStart` trio
(`clp.rs:644+`, `clp_wrapper_cpp.cpp:71-102`) is delivered and
determinism-verified (`crates/cobre-solver/tests/clp_determinism.rs`), but the
`SolverInterface` trait (`trait_def.rs:59`) has no hot-start lifecycle, so the
generic backward loop cannot reach it. **Wiring it is the proposed investment.**

## 1. What the backward opening loop actually does

The LP is loaded **once per trial point**, then openings are solved on the
resident model:

- `load_backward_lp` (`backward.rs:278-286`) calls `ws.solver.load_model(...)`
  once and, on the DCS path, `add_rows(...)` for the resident cut set.
- `solve_opening_baked` (`backward.rs:703-776`) iterates openings. Basis handling
  is gated by opening index (`backward.rs:727-731`):

  ```rust
  let stored_basis = if omega == 0 {
      resolve_backward_basis(basis_slice, m, s)   // canonical stored basis
  } else {
      None                                         // subsequent openings: no install
  };
  let view = run_stage_solve(ws, &inputs)?;
  ```

- `run_stage_solve` (`stage_solve.rs:91-160`): with `Some(basis)` it reconstructs
  by slot identity and calls `solve(Some(basis))`; with `None` it calls
  `solve(None)` and the solver starts from its **current internal basis**.
- `save_basis_at_omega_zero` (`backward.rs:643-648`) captures the basis only at
  `ω=0`, documenting why: _"writes at ω>0 are forbidden because the retained LU
  factorization would be overwritten by subsequent opening solves."_

This was made deliberate in commit `212c9608` ("epic-03 quick wins P3b"):
subsequent openings switched from `solve(Some(&working_basis))` to `solve(None)`
to use the solver's internal hot-start. The test
`multi_opening_subsequent_openings_use_internal_hotstart` (`backward.rs:2950`)
asserts `warm_start_calls == 0` for `ω>0` (`backward.rs:3034-3042`).

The DCS path (`solve_opening_dcs`, the production-default cut-selection path) does
the same warm-carry across openings (`backward.rs:778-819`), differing only in
that the resident cut set grows lazily.

## 2. The asymmetry — root of the gap

### HiGHS retains the factorization

HiGHS keeps its `HEkk` simplex state (basis + INVERT) across `Highs_run()` calls
on a resident model. The default hot-path config has `presolve = "off"`
(`highs.rs:171`), so warm-start is not defeated by presolve, and the basis is
installed via the non-alien path (`highs_wrapper_cpp.cpp:21-48`). The cost of
_not_ retaining this state is on record: removing the (now-absent) per-solve
`clear_solver` recovered a **4.7× regression** (`highs.rs:1038-1043`). So for
`ω>0`, `solve(None)` ⇒ no basis install, no refactorization, a handful of
incremental dual-simplex pivots.

### CLP refactorizes every opening

`ClpSolver::solve` always dispatches to `Clp_dual(handle, 0)` /
`Clp_primal(handle, 0)` (`clp.rs:~1310`), with the comment:

> "CLP retains its internal basis across consecutive simplex calls, so both
> `None` … and `Some` … fall through to the same cold solve below."

That is correct about the _basis status_ and incomplete about the
_factorization_. The C interface `Clp_dual(model, ifValuesPass)` calls
`ClpSimplex::dual(ifValuesPass, startFinishOptions = 0)`. With
`startFinishOptions = 0`, each `dual()` does the full `createRim → factorize →
solve → deleteRim` cycle. The incremental bound patch (`Clp_chgColumnLower/Upper`,
etc.) keeps the model and basis valid across the patch — but the subsequent
solve still rebuilds the rim and factorization from scratch.

> **Net:** per opening, HiGHS reuses the INVERT; CLP rebuilds it. Over
> `|Ω|` openings × trial points × stages, CLP pays roughly `|Ω|×` the
> factorization work HiGHS pays.

### How to confirm the mechanism (Phase 0)

The hypothesis is directly testable from existing instrumentation
(`solver_stats.rs`): run the local case once per backend at matched config and
compare, per opening, **simplex iteration count** vs **solve time**. The
prediction: CLP shows low iteration counts (warm basis) but high solve time
(factorization-dominated), while HiGHS shows low on both. If solve time tracks
factorization rather than pivots, the mechanism is confirmed.

## 3. Why it is unexploited (and why that is good news)

Per `clp-capability-determinism.md`, the hot-start trio was added in commit
`55de2ccf` ("complete the CLP capability surface") to round out the wrapper, and
is delivered as **inherent `ClpSolver` methods, separate from the shared `solve`
entry point** (`clp.rs:644+`). The same doc records the _intended_ composition
(§"The three class-only knobs"):

> "per-(worker, stage) persistent solver instances, each solved once to leave the
> simplex rim and factorization alive, then snapshotting once with `markHotStart`,
> re-solving each bound-patched model with `solveFromHotStart`, and releasing with
> `unmarkHotStart` on teardown."

That is exactly the backward opening loop's shape — but it was **never wired into
the algorithm**, because the `SolverInterface` trait (`trait_def.rs:89-303`) has
no hot-start lifecycle and the generic backward loop only calls trait methods.
Crucially: this is _untried in the algorithm_, not _tried-and-abandoned_. No
commit shows a hot-start integration that regressed. (An earlier SIGSEGV scare
was root-caused to a C++ shim cast bug and fixed; see
`clp-capability-determinism.md` §"Root-cause correction".) The capability works
and is determinism-verified; only the integration is missing.

## 4. The fix

### 4.1 A hot-start lifecycle on `SolverInterface`

Add three methods with HiGHS-neutral defaults:

```rust
/// Snapshot the current factorization/rim for repeated bound-only re-solves.
/// Default: no-op (HiGHS already retains its INVERT across solves).
fn begin_hot_start(&mut self) {}

/// Re-solve the resident model after a bound patch, reusing the snapshot.
/// Default impl forwards to `solve(None)` (the current HiGHS warm path).
fn solve_hot(&mut self) -> Result<SolutionView<'_>, SolverError> { self.solve(None) }

/// Release the snapshot. Default: no-op.
fn end_hot_start(&mut self) {}
```

- **HiGHS impl**: defaults. Behavior is byte-identical to today (the `ω>0`
  `solve(None)` path is unchanged).
- **CLP impl**: `begin_hot_start` → `mark_hot_start`; `solve_hot` →
  `solve_from_hot_start`; `end_hot_start` → `unmark_hot_start`. These already
  exist on `ClpSolver`.

### 4.2 Wire it into the opening loop

In `solve_opening_baked` (and the DCS analog), per trial point:

```
load_backward_lp                       // once
patch ω=0 bounds; solve(Some(canonical))   // establishes basis + factorization
begin_hot_start()                          // snapshot (CLP: markHotStart; HiGHS: no-op)
for ω in 1..|Ω|:
    patch ω bounds
    solve_hot()                            // CLP: solveFromHotStart; HiGHS: solve(None)
end_hot_start()                            // CLP: unmarkHotStart; HiGHS: no-op
```

This preserves the exact HiGHS path while giving CLP factorization reuse.

### 4.3 Two CLP variants by cut-selection mode

`markHotStart` snapshots a **fixed** factorization; it is invalid across row
additions. So:

- **Baked path (`cut_selection = none`)** — matrix is fixed across all openings of
  a trial point ⇒ `markHotStart`/`solveFromHotStart` applies directly. Biggest,
  cleanest win.
- **DCS path (cuts grow lazily per/across openings)** — `markHotStart` cannot span
  row growth. Use the more general **`startFinishOptions` persistence** lever
  instead: call `ClpSimplex::dual` with `startFinishOptions` set to keep the rim
  and factorization alive across the `chg-bounds` + `addRows` + re-solve sequence,
  letting CLP perform incremental factor updates (governed by the already-tuned
  `factorization_frequency = 200`) rather than a full rebuild per call. This is
  reachable through the existing C++ shim (which already reaches `->model_`) and
  needs a new shim entry exposing the `startFinishOptions` argument.

Both variants flow through the same trait lifecycle; `solve_hot`'s CLP impl picks
the mechanism based on whether rows may grow (i.e. cut-selection mode).

## 5. Determinism

Most of the determinism burden is already discharged:

- The hot-start trio is verified bit-for-bit (run-to-run, cross-instance,
  same-instance-reused) plus declaration-order invariant by
  `crates/cobre-solver/tests/clp_determinism.rs`
  (`clp-capability-determinism.md` §"What is wired").
- Openings are processed in fixed order `0..|Ω|` by a single worker, seeded from
  `ω=0`'s canonical basis. The chain is independent of trial-point partitioning,
  so it is bit-identical across rank counts — the same argument HiGHS already
  relies on. `solveFromHotStart` is in fact _stronger_: every opening restarts
  from the identical `ω=0` snapshot rather than chaining.
- Perturbation off (`102`) and scaling off remain preconditions and are already
  the CLP defaults.

Two checks remain and are **non-negotiable gates**:

1. **Cross-rank-count bit-determinism at the SDDP level** for the CLP backend
   under the new path (not just the isolated solver harness): identical
   cuts/lower bound across rank counts. The DCS `startFinishOptions` variant in
   particular relies on CLP's incremental factor-update path being deterministic.
2. **CLP-internal re-baseline.** Switching CLP's `ω>0` openings from chained
   `solve(None)` to snapshot-restart `solveFromHotStart` changes which valid
   optimal dual vertex CLP reports, so CLP's end-to-end results shift (still
   self-consistent, per the contract — a hot path need not match a cold one).
   CLP-vs-HiGHS cross-mode LB drift is already expected and accepted; this is
   intra-CLP re-baselining of golden values.

If either gate fails as designed, stop and escalate per the escalation contract
in `clp-capability-determinism.md` — do not relax bit-for-bit to "close enough".

## 6. Scope boundary

This plan covers **only the backward opening loop**, where repeated solves of the
same matrix are _consecutive_ on one worker and fit the current one-solver-per-
worker model.

It does **not** cover the forward pass or cross-stage reuse. In the forward pass,
stage `t` for scenario `s` is solved once; the next solve of stage `t` (scenario
`s+1`) is separated by solves of other stages on the same solver, so holding
stage `t`'s factorization hot requires **per-(worker, stage) persistent solver
instances** — the larger architecture sketched in
`clp-capability-determinism.md` and `backward-node-parallelism.md`. That is a
deliberate follow-up, out of scope here. This note is the contained first step:
prove factorization reuse pays on the backward openings before committing to the
per-stage-instance rework.

## 7. Implementation plan (robustness-first)

> **Re-sequenced 2026-06-07 per owner decision:** address CLP's retry/robustness
> deficit (§10 #2, §11) **before** the hot-start work (§4). Rationale: robustness
> helps every phase (the simulation phase is +70 % purely from retries, with no
> hot-start lever at all), and a quieter backward pass lets the later hot-start
> A/B read cleanly.
>
> **Measurement harness — note on what changed.** The first-try-vs-retried
> backward-solve-time split is **not** the entry point: that instrumentation was
> attempted before and did not pan out (retries are interleaved across workers and
> bundled inside the solver's `solve()` call, so the per-attempt timing is hard to
> separate cleanly). Instead, **the simulation phase is the robustness harness** —
> it is forward-only and retry-dominated, so the **retry count** (CLP 390 vs HiGHS 0) and **simulation wall-time** isolate the robustness signal directly from
> existing metadata, with no new per-solve instrumentation. Training retry count
> (1 491 vs 21) is the secondary signal.

### Track A — CLP robustness (do first)

| Phase                      | Work                                                                                                                                                                                                                                               | Gate                                                                             |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **A1. Diagnose**           | Use the simulation phase as the reproducer (10.6 % false-infeasible rate). Capture the failing LPs; determine whether a **warm** re-solve recovers them or only the current **cold all-slack** restart does, and at what basis state they arise.   | False-infeasibility characterized; cheapest recovering action identified.        |
| **A2. Cheaper first rung** | Replace the escalation ladder's cold all-slack first rung (`clp.rs:520-528`) with a cheaper warm recovery (e.g. perturbation-on warm re-solve / single refactor-and-retry **without** discarding the basis). Keep the full ladder as the fallback. | Sim retry-cost drops; all retries still recovered (0 failed); determinism holds. |
| **A3. Parameterization**   | Bounded sweep of perturbation / feasibility-tolerance / `factorization_frequency` / pricing to cut the **trigger frequency**. Time-boxed — do not rabbit-hole into CLP dual-simplex internals.                                                     | Training + sim retry count drops toward HiGHS; effort bounded.                   |
| **A4. Re-benchmark**       | Full train+sim A/B vs HiGHS on `cobre_tuning`/`cobre_tuning_clp` (identical config). Cross-rank-count bit-determinism check.                                                                                                                       | Sim gap and training retry rate near HiGHS; determinism passes.                  |

### Track B — Hot-start (after Track A, on a quiet backward pass)

| Phase                   | Work                                                                                                                                                                                                  | Gate                                                 |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| **B1. Trait lifecycle** | Add `begin_hot_start` / `solve_hot` / `end_hot_start` to `SolverInterface` with HiGHS-neutral defaults. CLP impl maps to the existing `mark_hot_start` / `solve_from_hot_start` / `unmark_hot_start`. | HiGHS path byte-identical (existing tests green).    |
| **B2. Wire baked path** | Use the lifecycle around the opening loop in `solve_opening_baked` (and LB/simulation analogs if they share the pattern). `cut_selection = none` only.                                                | Builds; backward unit tests green for both backends. |
| **B3. A/B (baked)**     | `cobre_tuning_clp`, `cut_selection = none`, before/after — now readable because retries are quiet (Track A). Cross-rank-count determinism check; CLP golden re-baseline.                              | Backward wall drops; determinism gate passes.        |
| **B4. DCS variant**     | Add a `startFinishOptions`-persistence shim entry; route `solve_hot` to it when cut rows may grow. Wire into `solve_opening_dcs` / `lazy_solve_preloaded`.                                            | Builds; determinism gate passes on DCS path.         |
| **B5. A/B (DCS)**       | Same case with cut-selection enabled, before/after.                                                                                                                                                   | Backward wall drops; determinism gate passes.        |

### Track C — Production validation & default decision

| Phase                  | Work                                                                                                                    | Gate            |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- | --------------- |
| **C1. Production A/B** | Operator-run convertido-scale A/B (manual, outside `/implement-plan`), per usual flow. Decide per-case default backend. | Owner sign-off. |

## 8. Validation & the conditional default switch

The objective stated by the owner: _make CLP the default backend for these cases
if it produces better results._ The deficits are independent (§10), so the
decision rule follows the two tracks:

1. **Track A (robustness)** must bring CLP's simulation gap and training retry rate
   near HiGHS's (~0 % retries). This is the larger total-wall lever and is
   prerequisite to a clean hot-start read.
2. **Track B (hot-start)** must then show CLP backward wall dropping materially on
   the now-quiet backward pass, with both determinism gates (§5) passing.
3. **Track C** confirms the combined win at production scale.

If all three hold, promote CLP to the default backend **for the validated case
class** (compile-time backend selection already exists, see
`backend-selection-and-mutual-exclusivity.md`); otherwise keep HiGHS default and
record the negative result here. Track A is independently worthwhile even if
Track B is later deferred — it improves the CLP backend in every phase.

## 9. Risks & open questions

- **Snapshot-restart vs chained-warm pivot count.** `solveFromHotStart` restarts
  every opening from the `ω=0` snapshot rather than chaining from the previous
  opening. For openings whose RHS is far from `ω=0`, this could need more dual
  pivots than chaining. Net effect (saved factorization vs extra pivots) is
  empirical — Phase B3 resolves it. If chaining proves better, the
  `startFinishOptions` variant (which chains while persisting the factorization)
  is the fallback for the baked path too.
- **`startFinishOptions` + `addRows` determinism (DCS).** CLP's incremental
  factor-update path must be bit-reproducible across rank counts. The existing
  harness covers single-instance reuse; Phase B4 must add an SDDP-level
  cross-rank-count assertion.
- **Refactorization cadence interaction.** The backward profile already pins
  `factorization_frequency = 200`. With a persistent rim, this cadence now governs
  drift-control refactors; confirm it is still appropriate (or tune) once the rim
  is actually held across solves.
- **Orphaned-code maturity.** The trio is determinism-verified but has never run
  under the algorithm's load (long opening chains, large cut sets). Phases B3/B5
  are its first real exercise; watch for numerical edge cases the isolated fixtures
  did not cover.

## 10. Empirical results — `cobre_tuning` (HiGHS) vs `cobre_tuning_clp` (CLP), 2026-06-07

First like-for-like A/B on a production-class case. **Identical `config.json`** (verified
byte-diff), same machine (`ip-10-211-36-166`), same case: 115 stages, 159 hydros, 106
thermals, 5 buses, 6 lines; 10 iterations, 5 forward passes, parallelism 5, single rank,
`cut_selection.method = "dynamic"` (DCS, not `none`). HiGHS 1.13.1 vs CLP 1.17.11. This is a
10-iteration **timing** run (final gap ~90%, not converged) — not a solution-quality
comparison.

| Metric                     | HiGHS              | CLP                | Δ (CLP vs HiGHS)         |
| -------------------------- | ------------------ | ------------------ | ------------------------ |
| **Training wall**          | 655.8 s            | 698.1 s            | **+42.3 s (+6.4 %)**     |
| — Forward wall (Σ iters)   | 59.2 s             | 51.1 s             | **−8.1 s (−13.7 %)** ✅  |
| — Backward wall (Σ iters)  | 592.6 s            | 642.8 s            | **+50.3 s (+8.5 %)** ❌  |
| **Simulation wall**        | 68.5 s             | 116.5 s            | **+48.0 s (+70.0 %)** ❌ |
| **Train+sim wall**         | 724.3 s            | 814.5 s            | **+90.2 s (+12.5 %)**    |
| Fwd solver-s (Σ×5 workers) | 271.7 s            | 228.8 s            | −42.9 s (−15.8 %)        |
| Bwd solver-s (Σ×5 workers) | 2022.4 s           | 2184.0 s           | +161.6 s (+8.0 %)        |
| Training retries           | 21 (0.02 %)        | 1 491 (1.14 %)     | **71× more**             |
| Simulation retries         | 0                  | 390 (10.6 %)       | **CLP only**             |
| LP solves (train)          | 130 323            | 130 437            | +114                     |
| Cuts active (final)        | 5 700              | 5 700              | identical                |
| Final LB / gap             | 4.357e10 / 90.71 % | 4.218e10 / 90.12 % | −3.2 % LB / ~same gap    |

### What the data shows

1. **Forward/backward split is exactly the predicted signature.** CLP is _faster_ in the
   forward pass at every iteration and _slower_ in the backward pass at every iteration. The
   entire +42 s training deficit is backward (+50 s), partly offset by CLP's faster forward
   (−8 s). Where each solve factorizes once anyway (forward), CLP's raw simplex wins; only the
   repeated-solve-on-a-resident-matrix backward openings expose HiGHS's INVERT-retention
   advantage — precisely the loop this plan targets.
2. **The backward gap broadly grows with cut-pool size** (refactorization signature): ~+3 %
   at iteration 1 (570 cuts) widening to +10–13 % at iterations 5/8/9 (2 850–5 130 cuts),
   though noisy. LB trajectories are near-identical for iters 1–3 then diverge (ordinary
   degenerate-vertex drift; not a quality verdict at 90 % gap).
3. **A confound: retries.** CLP fired its escalation ladder **71× more** in training and the
   simulation phase isolates the cost cleanly — simulation is _forward-only_ (no opening loop,
   so **zero** refactorization-per-opening to recover), yet CLP is **+70 % slower** there,
   driven entirely by a **10.6 % false-infeasibility retry rate** (390 vs 0). Each retry
   cold-solves a 5 700-cut LP up to 5×.

### Consequence for the plan

CLP has **two distinct deficits**, and the aggregate metadata cannot fully separate them in
the backward pass (both grow with cut count):

- **#1 Per-opening refactorization** — the hot-start thesis. Supported by the
  forward-fast/backward-slow asymmetry. Addressed by §4.
- **#2 False-infeasibility retries** — a robustness gap, proven independently by the
  simulation result. **Not** addressed by hot-start (and the simulation phase, where hot-start
  does not even apply, is +70 % purely from this). See §11 for why CLP escalates so much more.

Therefore **Phase 0 must be sharpened** to split _first-try_ vs _retried_ backward solve time,
so we can attribute the +8.5 % backward gap between #1 (hot-start-addressable) and #2
(robustness). And the CLP retry rate is its own tracked work item — likely the larger lever
for making CLP viable as default, orthogonal to this plan. The forward result shows the
ceiling is favorable: CLP's per-solve engine is competitive-to-faster, so if #1 is closed by
hot-start **and** #2 is brought near HiGHS's ~0 % retry rate, CLP could become the faster
backend overall.

**Caveats:** single case, reps=1, 10-iteration timing run. Forward/backward solver-seconds
are summed across 5 workers; the per-iteration **wall** figures (from `convergence.parquet`,
which sum to the reported wall durations) are the trustworthy split.

## 11. Solver defaults & escalation-ladder comparison

### 11.1 Steady-state (floor) parameters

At the hot-path floor the two backends are configured almost identically — both dual simplex,
no scaling (cobre pre-scales), no perturbation, `1e-9` feasibility tolerances:

| Concern                | HiGHS floor (`highs.rs:155-210`)                                                                                        | CLP floor (`clp.rs:144-155`)                                                     |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Algorithm              | dual simplex (`simplex_strategy=1`)                                                                                     | dual simplex (`algorithm=Dual`)                                                  |
| Presolve               | **off**                                                                                                                 | not applied (no `ClpSolve` presolve in path)                                     |
| Scaling                | off (`simplex_scale_strategy=0`)                                                                                        | off (`scaling=0`)                                                                |
| Perturbation           | off (`dual_simplex_cost_perturbation_multiplier=0.0`)                                                                   | off (`perturbation=102`, not CLP's `100`=auto)                                   |
| Pricing                | Devex (`simplex_dual_edge_weight_strategy=1`)                                                                           | CLP `ClpDualRowSteepest` ctor default (`dual_pricing_mode=3` sentinel ⇒ no call) |
| Primal/dual feas. tol. | `1e-9` / `1e-9`                                                                                                         | `1e-9` / `1e-9`                                                                  |
| Iteration limit        | `max(100 000, 50·cols)`                                                                                                 | `max(100 000, 50·cols)`                                                          |
| Other                  | `simplex_price_strategy=1` (Row); `simplex_initial_condition_check=0`; `rebuild_refactor_solution_error_tolerance=1e-6` | log level 0; `factorization_frequency=0` sentinel ⇒ CLP internal default         |

**Per-phase ("deep-cut-pool") profile** (applied identically for forward / backward /
simulation, `solver_phase.rs`): each backend overrides only what's native to it —

- **HiGHS**: `simplex_price_strategy` 1 → **2** (`RowHyperSparse`). One field.
- **CLP**: `dual_pricing_mode` 3 → **1** (full dual steepest-edge) and
  `factorization_frequency` 0 → **200**. Two fields.

These are independently chosen for each solver's option surface, not translations of each
other.

### 11.2 Retry escalation ladders — structurally different

| Aspect              | HiGHS (`highs.rs:708-834`, `retry_escalation`)                                         | CLP (`clp.rs:494-572`, `escalate_solve`)                            |
| ------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Trigger             | initial solve non-optimal (infeasible / unbounded / iteration-limit)                   | `PRIMAL_INFEASIBLE` or `STOPPED`                                    |
| Length              | **12 levels**, two-phase                                                               | **5 rungs**                                                         |
| Warm state on retry | L0 cold-restarts (`clear_solver`)                                                      | **every** rung resets to a cold all-slack basis                     |
| Time budget         | 15 s/level (Phase 1), 30 s/level (Phase 2), 120 s overall                              | none — runs the 5 rungs                                             |
| Levers available    | presolve, dual simplex, **tolerance relaxation**, **IPM**, **objective/bound scaling** | algorithm switch (dual↔primal), **perturbation on**, **scaling on** |

**HiGHS ladder** — Phase 1 (L0–L4, cumulative): L0 cold restart + perturbation on → L1
+presolve → L2 +dual simplex → L3 +relaxed feasibility tolerances → L4 +IPM. Phase 2 (L5–L11,
each restarted fresh from defaults with presolve on): combinations of objective scaling
(`user_objective_scale` −10/−13), bound scaling (`user_bound_scale` −5/−8), further relaxed
tolerances (~`1e-7`), dual simplex, and IPM. Defaults restored unconditionally after.

**CLP ladder** — 5 rungs, first OPTIMAL wins, floor restored after:

1. switch to **primal** simplex (perturbation/scaling stay off)
2. **perturbation on** (`50`), dual
3. perturbation on, primal
4. perturbation on + **scaling on** (`1`), dual
5. perturbation on + scaling on, primal

Note CLP has **no IPM fallback** (simplex-only) and **no tolerance-relaxation rung**.

### 11.3 Why this explains the retry results

CLP's warm-started dual simplex reports **false `PRIMAL_INFEASIBLE`** far more often than
HiGHS does at the floor (a known issue — LP is feasible, HiGHS solves it; see
`project_clp_floor_infeasible` and the `cec20748` ladder fix). That fires the 5-rung ladder
**71× more** in training and **10.6 %** of the time in simulation. HiGHS had an analogous
warm-start false-infeasible problem, fixed by a cold-retry (`f428754e`), leaving a 0.02 %
residual. CLP's ladder recovers every case (0 failed) but the underlying fragility is much
more frequent, so the recovery cost is large — and because each CLP rung discards the warm
basis and cold-solves a cut-laden LP, those recoveries are individually expensive.

The strategic read: **hot-start (§4) targets refactorization, not this.** Reducing CLP's
false-infeasibility/retry rate is a separate, and on this evidence possibly larger, lever for
making CLP competitive — especially in the simulation phase, where the opening-loop hot-start
gives no benefit at all.

## 12. A1 diagnosis (executed 2026-06-07): the retry deficit is a dual-simplex pathology — primal fixes it

Track A / Phase A1 was run against the simulation phase (the retry-dense reproducer: a
sim-only run via `training.enabled = false` reusing the trained `cobre_tuning_clp` policy,
which re-bakes all 5,700 cuts → ~390 false-infeasibilities, ~117 s; same box
`ip-10-211-36-166`). A throwaway env-gated diagnostic (`COBRE_CLP_DIAG`) probed cheaper
recoveries before the cold ladder; a sequence of single-variable A/B runs followed. Results:

1. **The cold all-slack reset is unnecessary.** 375/375 false-infeasibilities recovered with a
   warm re-solve that does **not** reset the basis: **78 % via primal**, **22 % via simply
   re-running dual** (fresh factorization, same basis). Zero needed the cold reset.

2. **…but removing the reset saves no wall time.** Same-box A/B: warm-recovery **118.6 s** vs
   cold-reset **116.9 s**. The recovery _solve_ costs ~1,200–1,400 pivots (rec_iters median
   1165 dual / 1433 primal) on these ~2,494-row LPs; the per-element reset is negligible
   against that. **→ The originally-planned A2 ("cheaper recovery first rung") is refuted as a
   performance measure** — keep it, if at all, only as a code simplification.

3. **The false-infeasibility is parameter-robust.** Retry count stayed at **exactly ~390**
   across `factorization_frequency` (200→50), `dual_pricing_mode` (1→3), perturbation
   (off→on), scaling, and feasibility tolerance. None of the obvious solver knobs move it.
   (The `fail_iters≈200` clustering near the refactor cadence was coincidental — `ff=50`
   changed nothing.)

4. **Latent config gap found.** Simulation **never applies a solver profile**:
   `ProfiledSolver::new` stores `current_profile` as a Rust value without pushing it to FFI,
   and the sim pipeline (unlike `forward_pass_state` / `backward_pass_state`) never calls
   `set_profile`. So sim runs CLP on its **native** defaults (auto-perturbation, native
   scaling, `1e-7` tolerances) — diverging from training and carrying a determinism smell
   (auto-perturbation). Almost certainly an oversight.

5. **The fix — primal as the primary simplex in simulation.** Applying a profile in sim **and**
   setting `algorithm = Primal`:

   | sim config (same box, same policy)         | retries | duration   |
   | ------------------------------------------ | ------- | ---------- |
   | dual-primary (baseline / native)           | 390     | 116.9 s    |
   | dual-primary + cobre floor profile applied | 401     | 110.5 s    |
   | **primal-primary**                         | **0**   | **78.6 s** |
   | (HiGHS reference)                          | 0       | 68.5 s     |

   Cleanly isolated: same setup with **dual** = 401 retries, with **primal** = **0**. CLP's
   _dual_ simplex is the sole cause — it falsely declares these warm-started cut-laden LPs
   infeasible; CLP's **primal** solves them directly. Primal-primary **eliminates all retries
   and cuts sim wall-time by ~33 %** (116.9 → 78.6 s), shrinking the gap to HiGHS from **+70 %
   to +14.8 %**. Primal simplex is deterministic, so the fix is determinism-compatible.

### Consequence — Track A redirected

- **A2 is no longer "cheaper recovery rung."** It is: **(a) make the simulation pipeline apply
  a solver profile** (fixing the latent gap in §12.4), and **(b) use primal as the primary
  algorithm for the simulation phase.** Small, determinism-safe, ~33 % sim speedup, zero
  retries.
- **A3 (parameterization)** is largely closed: the obvious knobs are null (§12.3). The
  remaining question is whether primal-primary also helps the **training** passes (forward
  retries, and the dual-favourable backward openings), which must be tested separately — do
  **not** assume; dual is theoretically preferred for the bound-change re-solves of the
  backward opening loop, and training's retry rate (1.14 %) is already far below sim's.
- **Open gates for the clean A2 implementation:** verify primal-primary sim is bit-reproducible
  across rank counts (primal is deterministic, but assert it at the SDDP level); decide the fix
  scope (narrow: sim pipeline applies a `SIMULATION` profile with `algorithm = Primal` — safer
  than broadening `ProfiledSolver::new`); re-baseline CLP simulation golden values (primal may
  report a different valid optimal vertex than dual).

All diagnostic edits were reverted; the measurements above were produced with throwaway,
env-gated instrumentation and single-variable profile tweaks, none committed.
