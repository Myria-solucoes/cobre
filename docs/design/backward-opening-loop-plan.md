# Backward opening loop — unified performance & CLP-backend plan

> **Status**: Plan / sequencing authority (2026-06-08). This is the **single
> implementation plan** that unifies three previously-separate design notes plus
> two findings made while executing them. It owns the **current-state analysis,
> the workstream interactions, and the sequencing**; each workstream's detailed
> design continues to live in its source note. Where this plan and a source note
> disagree on a code fact, **this plan is authoritative** (it was written against
> a verified read of the tree on 2026-06-08).
>
> **Source notes (each remains the detailed design for its workstream):**
>
> - `docs/design/clp-hot-start-backward-pass.md` — CLP robustness (A2/A2b, done)
>   - CLP factorization-reuse hot-start (W3).
> - `docs/design/backward-opening-ordering.md` — warm-start-friendly opening
>   ordering (W1).
> - `docs/design/dcs-architecture-debt.md` — the dual-path `StageOpeningSolver`
>   seam (the consolidation refactor both W1 and W3 want).
> - `docs/design/backward-pass-performance-analysis.md`,
>   `docs/design/backward-node-parallelism.md` — adjacent levers (out of scope
>   here, noted in §8).
> - `.claude/rules/sddp.md` — the determinism / cut-sign-scale contracts every
>   workstream must preserve.

---

## 0. Why one plan

Five lines of work all operate on the **same hot path** — the backward pass's
per-trial-point opening loop and the CLP backend that runs it:

| ID     | Workstream                             | Cuts what                                       | Backend  | Source            | Status            |
| ------ | -------------------------------------- | ----------------------------------------------- | -------- | ----------------- | ----------------- |
| **W0** | CLP simulation robustness (A2 + A2b)   | sim retries + sim determinism                   | CLP      | clp-hot-start     | **done**          |
| **W1** | Warm-start-friendly opening ordering   | **pivots** per warm re-solve (~88% of backward) | **both** | backward-ordering | proposed          |
| **W2** | CLP forward/training determinism       | thread/rank-count variance in training          | CLP      | this plan (§1.3)  | follow-up         |
| **W3** | CLP factorization-reuse hot-start      | **refactorizations** per opening                | CLP      | clp-hot-start §4  | proposed          |
| **W4** | Backward false-infeasibility retries   | 1491 dual retries (100% backward)               | CLP      | this plan (§1.4)  | measure-first     |
| **R**  | `StageOpeningSolver` seam + DCS config | dual-path comprehension debt                    | n/a      | dcs-debt §7       | enabling refactor |

They are not independent: W1 and W3 attack different terms of the _same_
per-opening cost and **stack**; W1 changes the `ω==0` logic that R wants to
unify; W1 likely _absorbs_ part of W4; and W2 shares A2b's root cause. Planning
them separately would mean three uncoordinated edits to the same ~18-conditional
hot path. This note sequences them so each lands once, on a baseline the next one
measures against.

---

## 1. Current state of the codebase (verified 2026-06-08)

### 1.1 The backward opening loop as it exists today

`process_trial_point_backward` (`backward.rs:945`) loads the stage LP **once per
trial point**, then iterates openings and dispatches to one of two drifted solve
functions:

```
dcs_params = training_ctx.dcs.filter(|p| p.is_active(iteration))   // backward.rs:982
if Some(params) = dcs_params { load cut-free core; build_initial_resident_set }
for omega in 0..succ.probabilities.len() {                          // canonical ω
    if dcs { solve_opening_dcs(.., continue_carry = omega != 0) }   // backward.rs:1008
    else   { solve_opening_baked(..) }                              // backward.rs:703 / 824
}
aggregate StagedCut from per-opening outcomes                       // canonical ω
```

**The reduction that makes reordering possible (verified).** The cut is a
fixed canonical-ω reduction, independent of solve order:

- Per-opening results are written **indexed by canonical ω**:
  `outcomes[omega]` (`backward.rs:628`) and `per_opening_stats[omega]`
  (`backward.rs:621`) — never by solve position.
- Aggregation iterates `outcomes[0..n]` zipped with `probabilities[0..n]`
  (`risk_measure.rs:423-437`, called at `backward.rs:1033-1039`); probabilities
  are uniform, canonical (`backward_pass_state.rs:773-778`).
- Per-opening noise comes from the canonical tree: `tree_view.opening(s, omega)`
  (`backward.rs:993`), and every RHS patch (`transform_inflow_noise` /
  `_load_noise` / `_ncs_noise`) reads that opening's `raw_noise`
  (`patch_opening_bounds`, `backward.rs:307-333`).

⇒ **Solving the openings in a permuted order, while aggregating in canonical ω
order, yields bit-identical cuts** — _provided the per-opening values are
unchanged_. They are unchanged only if the three sites below stop keying off
literal `ω==0`.

**Three `ω==0` couplings (must become "first-solved" for W1):**

1. **Baked basis capture** — `save_basis_at_omega_zero` guarded by `omega == 0`
   (`backward.rs:771-772`).
2. **Baked warm-start load** — `resolve_backward_basis` only at `omega == 0`
   (`backward.rs:727-731`).
3. **DCS warm-carry** — `continue_carry = (omega != 0)` (`backward.rs:1008`); the
   `continue_carry == false` branch **resets** `row_map` (`dcs.rs:680-688`), so if
   the canonical ω=0 is _not_ solved first it would wipe the resident cut set the
   earlier-solved openings accumulated.

Because the W1 ordering key is **run-constant** (§1.5), the first-solved opening
is the same every iteration, so the per-(m,s) basis store (`save`→`resolve`)
stays self-consistent once these guards key off "first-solved" instead of `ω==0`.
The DCS path captures **no** ω=0 basis by design (`backward.rs:920-922`), so only
its `continue_carry` flag changes.

### 1.2 Warm-start and solver state across solves

- **HiGHS** rebuilds its full solver state on every `load_model` (`Highs_passLp`),
  so each solve is order-independent — HiGHS is already thread-count invariant.
- **CLP**'s `Clp_loadProblem` swaps model data but **does not heal the ClpSimplex
  rim/pricing** (documented in `clp.rs` `load_model`), so stale steepest-edge
  weights persist across solves on a worker. This is the root cause behind W0/W2.
- Within a trial point, `ω>0` warm-continues via `solve(None)` (no basis install),
  reusing the solver's retained state from the previous opening.

### 1.3 CLP backend status

- **A2 (done, `a973f720`)**: simulation now applies the `SIMULATION` profile and
  `ClpProfile::SIMULATION.algorithm = Primal`; CLP sim went 390→0 retries, −32%.
- **A2b (done, `652fc537`)**: `SolverInterface::reset_solver_state()` (no-op
  default; CLP recreates the `ClpSimplex` handle + re-applies the cached profile),
  called at the sim scenario boundary; CLP sim is now thread-count invariant
  (threads 1/3/5 bit-identical) at no measurable cost.
- **W2 gap (open)**: the forward (training) pass shares A2b's root cause. It is
  **stage-synchronous** (`for t { for m }`, `forward_pass_state.rs:712/720`),
  per-scenario reset at `:729`; CLP training is therefore _likely_ thread-count
  variant on large cases — a correctness prerequisite for CLP-as-default, not yet
  verified or fixed.
- **Hot-start surface for W3 (verified present-but-dormant)**: the
  `mark_hot_start` / `solve_from_hot_start` / `unmark_hot_start` trio exists on
  `ClpSolver` (`clp.rs:644-785`) and is **called zero times** in `cobre-sddp`.
  `SolverInterface` (`trait_def.rs`) has **no hot-start lifecycle** (only the
  `reset_solver_state` and `record_reconstruction_stats` no-op defaults).
  `startFinishOptions` is **not** exposed in the FFI/shim (`clp_ffi.rs`,
  `csrc/clp_wrapper*`) — the DCS-compatible hot-start variant needs a new shim
  entry.

### 1.4 Benchmark baseline — `cobre_tuning` (identical config, same box)

| metric             | HiGHS   | CLP (pre-W0)      | CLP (post-A2/A2b) |
| ------------------ | ------- | ----------------- | ----------------- |
| training wall      | 655.8 s | 698.1 s           | (unchanged)       |
| simulation wall    | 68.5 s  | 116.9 s           | **76.4 s**        |
| simulation retries | 0       | 390 (10.6 %)      | **0**             |
| training retries   | 21      | **1491 (1.24 %)** | (unchanged)       |

Two structural facts drive W1 and W4:

- **W1 target**: backward warm re-solves (`ω≥1`) are **~88 % of backward solve
  time at ~220 pivots each** (vs ~705 for the `ω=0` cold head), measured on
  HiGHS — i.e. the residual inner-loop cost on the _default_ backend is pivots,
  exactly what ordering attacks.
- **W4 target**: the 1491 CLP training retries are **100 % in the backward pass**
  (forward 0, lower-bound 0; from `output/training/solver/iterations.parquet`),
  spread broadly across stages, peaking at iteration 2 then declining.

### 1.5 The ordering key is run-constant (W1 precompute feasibility — verified)

- The `OpeningTree` is built **once at setup**, owned by `StochasticContext`,
  rank-invariant; runtime access is `OpeningTree::opening(stage, ω)`
  (`cobre-stochastic/.../opening_tree.rs:34-103`).
- The per-opening **noise vector** `raw_noise` is the single pre-image of every
  per-opening RHS change, and the σ scales (`InflowModel.std_m3s`,
  `cobre-core/src/scenario.rs:225-240`) are available at setup (in `System`),
  though **not yet pre-aggregated** into the stochastic context — a one-pass
  precompute adds them.
- ⇒ A per-stage permutation `solve_order[s]` is a **pure, rank-invariant function
  of the synced tree + fixed σ**: precompute once at setup (a few KB), index in
  the hot path. Natural owner: `OpeningTree` (add `solve_order(stage) -> &[u32]`)
  or `StochasticContext`.

### 1.6 Architecture debt the perf work must navigate (verified, dcs-debt §2)

DCS was added as a **parallel strategy** rather than behind a seam, leaving
**≈18 conditionals** and **two drifted opening-solve functions**:

| concern              | `solve_opening_baked` (`backward.rs:703`) | `solve_opening_dcs` (`backward.rs:824`)     |
| -------------------- | ----------------------------------------- | ------------------------------------------- |
| solve                | `run_stage_solve` (baked all-cuts LP)     | `lazy_solve_preloaded` (lazy resident set)  |
| warm start (ω=0)     | `resolve_backward_basis` (cross-iter)     | `stored_basis = None` → cold                |
| dual extraction      | `extract_duals_from_view`                 | `extract_state_duals_only` + binding counts |
| basis capture (ω=0)  | `save_basis_at_omega_zero`                | none                                        |
| outcome accumulation | `accumulate_opening_outcome`              | `write_opening_outcome`                     |

Plus driver-level `dcs_active` load-skips (`backward_pass_state.rs:1004/1017/1068`).
**No correctness bug** — but W1 (which edits the `ω==0` logic in _both_ twins) and
W3 (which adds hot-start to _both_) would each have to touch both drifted paths.
The debt note's remediation — an enum-dispatched `StageOpeningSolver`
(`Baked`/`Lazy`) — is the consolidation point, and its §8 explicitly says it is
"best attempted **after** opening ordering, since the 'first solved vs ω==0'
change naturally wants the same seam." That ordering is adopted below.

### 1.7 Determinism contract (non-negotiable, `.claude/rules/sddp.md`)

Results must be **bit-identical within a backend across MPI rank / thread counts**
and **invariant to input declaration order**. Cut aggregation is the canonical-ω
reduction in §1.1; the append-only cut pool matches bases by slot identity. Every
workstream below is gated on this contract, and the existing
`tests/determinism.rs` (training + simulation thread-count) + `parity_hash_d01_d15`
(golden output) are the regression guards.

---

## 2. The workstreams, reconciled against the code

### W1 — Warm-start-friendly opening ordering (lead)

Solve a trial point's openings in an order that minimizes inter-opening RHS
distance (sort by a σ-weighted aggregate of each opening's `raw_noise`), so each
warm re-solve is a small hop. **Backend-agnostic** (the 220-pivot figure is on
HiGHS), **no FFI**, **determinism-safe** given §1.1. Verified-correct
implementation shape:

1. **Setup precompute** (once): for each stage, `solve_order[s]` = `0..|Ω_s|`
   sorted by the σ-weighted aggregate noise key (descending default, see source
   §3.2), tie-broken by ω. Store on the `OpeningTree`; expose `solve_order(s)`.
   Add the σ-per-noise-component aggregation at setup (§1.5).
2. **Hot-path** (`process_trial_point_backward`): iterate `solve_order(s)` instead
   of `0..|Ω|`. Change the three `ω==0` couplings (§1.1) to "first entry of
   `solve_order(s)`". Keep results written to `outcomes[ω]`/`per_opening_stats[ω]`
   by canonical ω and the aggregation loop untouched.
3. **Diagnostic first** (cheap, predicts the ceiling): per-opening
   `(noise_key, simplex_iterations)` is _already_ logged in
   `solver/iterations.parquet` (it has `phase, stage, opening, simplex_iterations`)
   — only the `noise_key` column needs adding. Confirm pivots correlate with
   consecutive-opening noise distance before implementing.

### W2 — CLP forward/training determinism (correctness prerequisite)

Extend `reset_solver_state()` (already in the trait, no-op for HiGHS) into the
forward worker. Because forward is stage-synchronous (`for t { for m }`), the
reset must fire per scenario _within each stage_ — placement and cost
(handle-recreate frequency) need care, and the gain must be **verified by a CLP
training threads-1-vs-5 determinism check** before committing the placement.
Independent of W1/W3; required before CLP can be a determinism-compliant default
for training.

### W3 — CLP factorization-reuse hot-start

Add a `SolverInterface` hot-start lifecycle (`begin_hot_start` / `solve_hot` /
`end_hot_start`, HiGHS no-op default since it already retains its INVERT; CLP
maps to the dormant trio for the fixed-matrix baked path, and to a new
`startFinishOptions` shim entry for the row-growing DCS path). Reduces CLP's
per-opening refactorizations. **Measured after W1**, because ordering changes the
baseline and may shrink the CLP-specific deficit enough that the FFI complexity is
not worth it.

### W4 — Backward false-infeasibility retries

100 % backward, 1.24 %, CLP dual-simplex false-infeasibilities. **Primal is off
the table** in backward (dual is correct for the bound-change warm-re-solve loop).
Re-measure after W1: smaller inter-opening hops should make the dual far less
likely to hit the spurious infeasibility, so W1 may absorb much of W4. Residual is
addressed by backward-profile param tuning (validly testable in backward, unlike
sim — the backward profile _is_ applied), not by an algorithm switch.

### R — `StageOpeningSolver` seam (enabling refactor)

Collapse the two drifted `solve_opening_*` functions + the driver `dcs_active`
skips into one enum-dispatched seam (variants `Baked`/`Lazy`), so the W1
"first-solved" logic and W3 hot-start lifecycle live in **one** place per variant.
Sequenced after W1 per dcs-debt §8. The independent config refactor (dcs-debt §7
item 2: first-class DCS config, stop honoring deprecated `threshold`/
`memory_window`) can go any time.

---

## 3. How they interact (why the order matters)

- **W1 ⟂ W3, and they stack.** Ordering cuts _pivots_; hot-start cuts
  _refactorizations_ — different terms of the per-opening cost. W1 also benefits
  HiGHS (the default), so it pays off regardless of the CLP decision.
- **W1 likely absorbs W4.** Smaller consecutive RHS hops keep the dual near its
  prior optimum, where spurious infeasibility (a large-jump / numerically-delicate
  artifact) is far less likely. Measure W4 _after_ W1.
- **W1 and W3 both want the seam (R).** Both edit the `ω==0` / opening-solve logic
  in both drifted twins. Doing W1 first introduces the "first-solved" concept;
  then R consolidates it and gives W3 a single landing site. Doing R _before_ W1
  would refactor a moving target; doing W3 _before_ R means touching two drifted
  paths.
- **W2 is independent** of all of the above (it's a per-scenario solver reset, not
  an opening-loop change), but is a prerequisite for trusting any CLP-default
  decision.
- **W0 is the precedent**: A2b proved the `reset_solver_state` hook works and is
  free; W2 reuses it; W3 adds a second (hot-start) lifecycle hook in the same
  trait-extension style.

---

## 4. Sequencing (the definitive order)

```
Phase A  W1  Opening ordering            ── lead: backend-agnostic, low-risk, helps HiGHS
            A1 diagnostic (noise_key vs pivots)         [gate: correlation confirms ceiling]
            A2 prototype behind default-off flag        [gate: determinism + parity green, bit-identical cuts]
            A3 measure both sweep directions; set default
            A4 default-on
Phase B  W2  CLP forward determinism      ── parallel with A; correctness prerequisite
            [gate: CLP training threads-1-vs-5 bit-identical]
Phase C  ── RE-BENCHMARK CLP vs HiGHS (train+sim) on the W1 baseline ──
            decides whether W3 is still worth the FFI complexity
Phase D  R   StageOpeningSolver seam       ── now motivated by W1's first-solved + incoming W3
Phase E  W3  CLP hot-start                 ── lands on the seam; baked=markHotStart, DCS=startFinishOptions
            [gate: determinism across rank counts; CLP golden re-baseline]
Phase F  W4  retry re-measure + decision   ── after W1; param tuning only, no algorithm switch
Phase G  ── per-case default-backend decision (the owner goal) ──
(anytime) R2 first-class DCS config (dcs-debt §7.2) — independent, low-risk
```

**Rationale in one line each:** A leads because it is the only change that helps
the _default_ backend and resets the baseline; B is independent correctness; C is
the decision gate that prevents over-investing in D/E; D unblocks clean landing of
E; E is measured against the post-W1 baseline; F is gated on A's side effect; G is
the owner's objective, made on real post-A/B numbers.

---

## 5. Per-phase detail & acceptance criteria

### Phase A — W1 opening ordering

| Step   | Work                                                                                                                                                                                                                                     | Gate                                                                                                                                                                     |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **A1** | Add `noise_key` (σ-weighted aggregate of `raw_noise`) to `solver/iterations.parquet`; on `cobre_tuning`, regress per-opening `simplex_iterations` on consecutive-opening noise distance (natural vs sorted).                             | Pivots correlate with hop distance; natural-order jumps large → ordering will pay.                                                                                       |
| **A2** | Setup precompute of `solve_order[s]` on `OpeningTree` (+ σ aggregation); behind a default-off flag, iterate it in `process_trial_point_backward`; change the **three** `ω==0` couplings (§1.1) to "first-solved". Aggregation untouched. | Builds; `determinism.rs` + `parity_hash_d01_d15` **green unchanged** under both backends (cuts bit-identical — the strongest correctness proof the decoupling is right). |
| **A3** | Run prototype ascending vs descending (and median-anchor if both tails costly); compare backward warm-pivots/wall.                                                                                                                       | Default `sweep_direction` fixed from data.                                                                                                                               |
| **A4** | Flip default-on if material, robust pivot reduction with bit-identical cuts.                                                                                                                                                             | Backward warm-solve pivots materially < 220; backward wall down.                                                                                                         |

### Phase B — W2 CLP forward determinism

| Step   | Work                                                                                                                                                                                               | Gate                                                                                             |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **B1** | Verify CLP training is thread-count variant: 5-iter, no-sim CLP training, threads 1 vs 5, compare cut/LB hashes.                                                                                   | Confirms (or refutes) the need before touching the hot path.                                     |
| **B2** | If variant: call `reset_solver_state()` at the forward per-scenario boundary (`forward_pass_state.rs` `run_forward_worker`); choose placement (per (t,m)) to balance determinism vs recreate cost. | CLP training bit-identical across threads 1/3/5; HiGHS unaffected; forward wall cost acceptable. |

### Phase D — R `StageOpeningSolver` seam

Collapse `solve_opening_baked`/`solve_opening_dcs` + `dcs_active` load-skips into
an enum-dispatched seam (variants own dual extraction, basis capture, outcome
accumulation, and the W1 "first-solved" logic). **Gate**: zero behavior change —
determinism + parity green; the ≈18 conditionals reduce to one construction + one
call site per pass.

### Phase E — W3 CLP hot-start

Add the `begin/solve_hot/end` trait lifecycle (HiGHS no-op). CLP: `markHotStart`
for `cut_selection = none` (fixed matrix); a new `startFinishOptions` shim entry
for the DCS path (row growth). Wire through the Phase-D seam. **Gate**: backward
wall drops on the post-W1 baseline; determinism across rank counts; CLP golden
re-baseline (primal/hot path may report a different valid optimal vertex).

### Phase F — W4 retries

Re-read `solver/iterations.parquet` backward `lp_retries` after W1. If materially
reduced → done. Else, bounded backward-profile param sweep (perturbation /
factorization cadence / pricing — all validly applied in backward). **No primal
switch in backward.**

---

## 6. Cross-cutting correctness gates

1. **Bit-identical cuts (W1)** — `parity_hash_d01_d15` (slow-tests) unchanged
   under both backends; the canonical-ω aggregation loop must remain untouched
   (the one regression that would silently break determinism).
2. **Thread/rank-count invariance (W2, W3, W1)** — `determinism.rs`
   (`test_{training,simulation}_determinism_across_thread_counts`,
   `test_canonical_ub_determinism_across_rank_counts`) green; plus the real-case
   threads-1-vs-5 hash checks used for A2b.
3. **Cut sign/scale + slot-identity basis** (`.claude/rules/sddp.md`) — untouched
   by any workstream (all operate on solve _order_ / solver _state_, not on cut
   construction).
4. **HiGHS neutrality** — every CLP-targeted change (W2/W3) is a no-op on the
   HiGHS path by trait default; verified by the HiGHS suite staying green.

---

## 7. Risks & open questions

- **W1 gain is data-dependent.** Tightly-clustered inflows ⇒ small gain; the A1
  diagnostic predicts this before implementation. The strong spatial inflow
  correlation in `cobre_tuning` (the indefinite-correlation warning) favors a 1-D
  key here.
- **W1 basis-store consistency** rests on `solve_order` being run-constant (§1.5).
  If a future change made the key per-iteration, the per-(m,s) basis capture/load
  would desync — keep the key setup-time and run-constant.
- **W1 σ alignment**: the σ weights are indexed by (hydro, stage); the opening
  noise vector layout must be matched at setup. A wiring detail, not a blocker.
- **W2 cost**: per-(t,m) handle recreate in the training hot path; verify the CLP
  forward wall cost is acceptable (A2b's per-scenario sim reset was free, but
  forward fires far more often).
- **W3 FFI + determinism**: `startFinishOptions` persistence must be
  bit-reproducible across rank counts (the existing harness covers single-instance
  reuse; an SDDP-level assertion is required). `markHotStart` is incompatible with
  the DCS row-growth path — baked-only.
- **R timing**: do not start the seam refactor while W1 is mid-flight in the same
  files (dcs-debt §8); land W1 first.
- **Aggregation-order regression** is the single highest-consequence W1 risk: the
  entire determinism guarantee rests on keeping aggregation canonical-ω while only
  the solve loop reorders. The green parity suite is the guard.

---

## 8. Scope boundary

In scope: the backward opening loop (W1, W3, W4, R), the CLP solver-state resets
(W0 done, W2), and the per-case default-backend decision (G).

Out of scope (separate, orthogonal efforts): forward/simulation single-solve
structure beyond the determinism resets; load imbalance / per-worker persistence
(`backward-node-parallelism.md`); cross-iteration ω-head basis capture; the FCF /
cut-pool internals; any change to cut construction, sign, or scale.

---

## 9. Done so far (W0)

- **A2** (`a973f720`): CLP simulation uses the primal simplex via an applied
  `SIMULATION` profile — 390→0 sim retries, −32 % sim wall; HiGHS byte-stable.
- **A2b** (`652fc537`): `SolverInterface::reset_solver_state()` recreates the CLP
  `ClpSimplex` at the sim scenario boundary — CLP sim thread-count invariant, free;
  HiGHS no-op. Establishes the trait-extension pattern W2 and W3 reuse.
  </content>
