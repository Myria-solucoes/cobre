---
title: "Dynamic Cut Selection (NEWAVE SC) — implementation design"
date: 2026-06-05
status: design (approved for implementation)
tags:
  [
    cobre,
    sddp,
    cut-selection,
    dynamic,
    lazy-constraints,
    row-generation,
    warm-start,
  ]
references: ["ideas/newave-cut-selection.pdf"]
companion:
  [
    "docs/design/accelerator-effectiveness-research.md",
    "docs/design/solver-tuning-results.md",
  ]
---

# Dynamic Cut Selection (NEWAVE SC) — implementation design

A design for implementing **Dynamic Cut Selection** (DCS) — NEWAVE's _Estratégia
SC_, Diniz et al. 2020 — in Cobre. This is **not** a spike: DCS is a lazy
constraint-generation loop _inside_ each stage LP solve with several
hyperparameters and correctness-critical details. This document specifies the
algorithm, the hyperparameters, the interactions with Cobre's architecture, and
a phased implementation plan.

DCS is an established production-grade SDDP technique (NEWAVE ships it), so the
implementation is **approved without a separate go/no-go measurement** — there is
no Phase-0 opportunity probe. The remaining decisions, recorded below, are about
_how_ DCS meets Cobre's baked-template architecture, not _whether_ to build it.

## 1. What DCS is (and is not)

### Two independent axes

Cobre's LP-solve strategy is a point in a 2-axis space, and keeping the axes
separate is what makes this design tractable:

- **Construction** — _baked_ (`load_model` a template with every active cut as a
  structural row) vs _incremental_ (`load_model` the **core** template, then
  `add_rows` the cuts).
- **Resident cut set** — _all active cuts_ (today's "TC") vs _a violated subset
  grown lazily_ (the paper's "SC" = DCS).

Cobre today sits at **(baked, all-cuts)**. DCS is the **resident-set = lazy**
axis; **incremental construction is its prerequisite**. The two are implemented
together but reasoned about separately.

### DCS is one cut-selection _method_, mutually exclusive with the others

Cobre's existing cut _selection_ (Level1, Dominated) **deactivates** cuts at the
**pool** level between iterations (post-backward): a deactivated cut is removed
from candidacy and never baked again, permanently shrinking `active_cuts()`. DCS
is a different mechanism — **lazy row generation within a single LP solve** — that
keeps the pool append-only with **every cut a candidate** and manages cost
per-solve by residency.

They are exposed as values of the **same** `training.cut_selection.method` enum
(`none | level1 | dominated | dynamic`), so they are **one-of by construction**:
selecting `dynamic` means Level1/Dominated do **not** run. This is not incidental
— it is required for correctness. DCS's exactness guarantee (below) rests on
`k1 = ∞` (every pool cut remains a candidate); a concurrent Level1 that removed
cuts from candidacy would break that guarantee (DCS would be exact only relative
to the Level1-pruned pool), and the two LP-shaping paths (deactivate-and-rebake
vs core-plus-incremental-residency) manipulate the LP in contradictory ways. DCS
therefore _replaces_ deactivation-based selection: it achieves the same cut-load
reduction _provably exactly_ rather than heuristically.

(A theoretically-coherent composition — Level1 prunes the candidate _pool_ to
bound DCS's scoring cost/memory while DCS manages residency — is deliberately
**out of scope**; see §6.)

### The algorithm

NEWAVE SC (paper §3.3), per stage/scenario/opening LP:

1. **Initial set** — solve the LP with only a small subset of cuts: those that
   were _active_ at this state in the last `k2` iterations, plus (backward pass)
   all cuts built at stage `t+1` in the current iteration.
2. **Solve** → optimum `f*`, FCF argument state `x*` (the state variables + `θ`).
3. **Score candidates** — for each cut `i` _not_ in the LP, compute its value at
   `x*`: `α_i = intercept_i + π_i · x*`. A candidate with `α_i > θ*` is violated
   (the current LP under-estimates the future cost there).
4. **Add** the `nadic` most-violated candidates (those with largest `α_i > θ*`),
   **warm re-solve** from the previous basis, `k ← k+1`, go to 2.
5. **Stop** when no candidate is violated (`nadic_added = 0`). The solution is
   then **exact**: every omitted cut is satisfied, so the optimum equals the
   all-cuts optimum (paper §3.5, §4.4.1).

Key property (paper §3.4): each inner re-solve adds only a few rows, so
**basis recovery (warm-start) makes it nearly free** — basis recovery is called
_indispensable_. This is why DCS and Cobre's warm-start are **complementary**,
not in tension.

## 2. Scope: both passes, one unified LP-solving approach

DCS is applied to **both the forward and backward passes**, and it is a hard
requirement that the two passes use the **same** LP-solving approach (same
construction, same basis handling, same lazy loop). This mirrors the per-phase
solver-profile unification already shipped for forward/backward/simulation.

The lazy loop — _score omitted cuts at `x*` → `add_rows(violated ≤ nadic)` →
`solve(None)` → repeat to no-violation_ — is **pass-agnostic** and lives in one
shared routine. Only the _post-solve extraction_ differs, and that already lives
in separate per-pass code:

- **Backward** consumes the final-LP **duals** to build a Benders cut.
- **Forward** consumes the final-LP **primal** (the next state + stage cost).

**Implementation order: backward first** (it is ≈93% of wall and carries the
harder correctness contract), then forward — but the shared routine is written
pass-agnostic from the start so forward is a drop-in, not a re-implementation.
**Simulation** (policy evaluation against the full pool) is the third consumer of
the same routine, run to exactness like forward; see §5.7. Lower-bound evaluation
is treated separately (§5.7).

## 3. Hyperparameters

| Param           | Meaning                                                                            | NEWAVE setting / finding                                                                               |
| --------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `k1`            | Window of past iterations whose cuts are _candidates_ at all                       | `k1 = max_iters` (all cuts) → guarantees equivalence with all-cuts (TC); never silently drop old cuts  |
| `k2`            | Window for the _initial_ active set (cuts active at this state in last `k2` iters) | tested {5,10,15,20,45}; **`k2 = 5` best**                                                              |
| `nadic`         | Max cuts added per inner iteration                                                 | tested {5,10,15}; ≥5 fine, slightly better at 15; _"determination above 5 barely affects performance"_ |
| start iteration | First SDDP iteration to apply DCS (need cuts to exist)                             | from iteration 2                                                                                       |

Cobre defaults to propose: `k1 = ∞` (all cuts candidate), `k2 = 5`, `nadic = 10`,
start at iteration 2. All exposed as `training.cut_selection` config (the new
`method = "dynamic"` variant) so the tuning harness can sweep them. (There is no
backward-only/forward-only `phase` knob: per §2 DCS applies to both passes
uniformly.)

## 4. Correctness contracts

- **Exactness (backward) — dual from the final LP.** The stop condition (no
  violated candidate) guarantees the LP optimum equals the all-cuts optimum. The
  dual `π` extracted for cut construction must come from the _final_
  (all-satisfied) LP, so cut gradients are unaffected. Extract after the inner
  loop converges, never mid-loop. (Respects the Benders-cut sign / subgradient
  contract in `.claude/rules/sddp.md`.)
- **Exactness (forward) — must run to no-violation.** Forward duals are _not_
  used to build cuts, so forward DCS is correctness-_simpler_ — it only needs the
  correct primal. But forward DCS **must run to exactness** (loop until no cut is
  violated): at exactness the primal optimum equals the all-cuts optimum, so the
  sampled trajectory is unchanged. An early stop in forward would shift the
  states the backward pass then visits → policy drift. No early-stop in forward.
- **Scoring space and sign.** Cut coefficients are stored as **raw** `∂Q/∂x`;
  the LP solves in **scaled** space, so `x*`/`θ*` read from the `SolutionView`
  must be **unscaled by `col_scale`** before scoring (the existing
  `cut_selection.rs` already scores in raw space, but over already-unscaled
  archived states — DCS reads the scaled live solution and must convert). The
  cut row is `−∇·x + θ ≥ intercept` (`sddp.md`), so the cut's `θ`-floor at `x*`
  is `intercept + ∇·x*_raw` and the candidate is **violated iff
  `θ*_raw < intercept + ∇·x*_raw − ε_viol`**. A sign or scaling slip here
  compiles, runs, and silently emits wrong cuts — treat it as a contract.
- **Violation tolerance `ε_viol`.** Exactness is a floating-point claim, so it
  needs an explicit tolerance. Too tight → over-add / cycle; **too loose → stop
  with a genuinely-violated cut resident-absent → duals are off the all-cuts
  optimum → wrong cut.** `ε_viol` must be `≥` the LP dual-feasibility tolerance
  and on the scale of the existing selection `tie_tolerance` (`1e-10`); expose it
  rather than hard-coding.
- **Bounded inner loop with TC fallback.** Cap the inner iterations; if the cap
  is hit (numerical noise re-finding "violations"), add **all** remaining
  violated candidates and solve once — degrading to TC for that solve, which
  preserves exactness. The loop must never be able to spin unbounded.
- **Policy equivalence, not bit-identity (DCS vs TC).** SC and TC reach
  equivalent-quality policies but may differ in alternate-optima multipliers, so
  bounds drift slightly across the run (paper §4.4). This is the same cross-mode
  drift Cobre already documents; the correctness gate compares converged bound +
  cost within tolerance, not bit-for-bit.
- **Determinism — DCS-vs-DCS AND across MPI rank counts.** Within DCS mode results
  must be bit-identical _and_ rank-count-invariant (Cobre's hard rule). The
  candidate scoring, violated-set selection, and ties must be deterministic and
  declaration-order invariant: score by slot, break ties by ascending slot id,
  select a stable top-`nadic`; no `Date`/random. **The initial resident set must
  not depend on per-worker solve order** — derive it from the synchronized
  cut-pool metadata, not from a per-worker trace (see §5.3, §5.5). Reuse the
  bit-deterministic `gemm_block` kernel for scoring (`gemm.rs`).
- **Append-only pool preserved — and never physically removed.** DCS never
  removes cuts from the pool; it only controls which are _resident in the LP at
  solve time_, and the solver FFI has **no row-removal primitive** (verified), so
  in-LP residency is **append-only too**: it can only grow within a stage-visit
  (see §5.3). Slot identity is retained.

## 5. Interactions with Cobre's architecture

### 5.1 Construction: core + incremental, no new FFI

Today `bake_rows_into_template(base, rows, out)` merges a **cut-free base
`StageTemplate`** with the cut `RowBatch` and the solver solves the result once.
The **"core" DCS loads is exactly that `base` template** (it already exists as the
bake input; confirm it is materialized and reachable per-stage at solve time, not
just transiently inside baking). DCS instead: `load_model(core)` →
`add_rows(initial subset)` → solve → loop[ score → `add_rows(violated)` → warm
re-solve ]. This needs an _incremental add-rows + warm re-solve_ path on the
loaded model, not a rebake per inner step.

**Confirmed: both backends already support this with the existing FFI — no new
wrapper functions are required.**

- HiGHS `add_rows` (`Highs_addRows`) does not clear the solver or reset the
  basis; HiGHS appends the new rows' logical slacks as basic and keeps the basis
  valid. A following `solve(None)` is a warm dual-simplex restart.
- CLP `add_rows` (`Clp_addRows`) likewise preserves the factorization/basis ("no
  full reload"); `Clp_dual` warm-restarts from the retained basis.

HiGHS is the leaner backend for the inner loop: CLP's `add_rows` additionally
rebuilds a full Rust-side CSC mirror on every call, and CLP installs an explicit
basis per-element (HiGHS uses a single bulk `set_basis_non_alien`). CLP's OSI
hot-start (`markHotStart`/`solveFromHotStart`) is **not** usable here — it is for
fixed-matrix bound-change re-solves and cannot add rows. Optional CLP-only
micro-optimizations (drop the CSC mirror; add a bulk basis setter) are deferred
until measured (§6).

The one option that matters is **presolve on the inner re-solves**: with a valid
warm basis present HiGHS skips presolve and warm-starts (evidenced by the
production warm-start at presolve=on: basis-reject 0%, ~15× fewer iterations on
repeat solves), but the predictable inner-loop config is **presolve off**, set via
the per-phase solver profile — not new FFI.

### 5.2 Basis: reconstruct the core, mark every added cut BASIC

The warm-start basis for the _initial_ solve of each (stage, solve) is built by
`reconstruct_basis`, with a **simplification** DCS enables:

- The **core** rows/columns are positionally stable across iterations, so their
  statuses are copied **by index** (this is already how `reconstruct_basis`
  handles the template region).
- Every **cut** row is assigned `BASIC` (slack basic = non-binding). Because cut
  rows are uniformly BASIC, their LP position carries no information — so the
  per-cut **slot-identity tracking** (`slot_lookup`, `cut_row_slots`, the
  "preserved status" path) that existed only to keep surviving cuts aligned under
  churn becomes **unnecessary and is dropped**. We do **not** try to guess which
  specific cuts will bind.

This is not _purely_ "all BASIC", and the exception is important: HiGHS rejects
an inconsistent basis (`set_basis_non_alien` requires
`col_basic + row_basic == num_row`). When `k` cuts bound in the captured solve,
the core statuses carry `k` excess basics, so `enforce_basic_count_invariant`
(demote `k` trailing BASIC cut rows → LOWER) **stays** — it is the consistency
repair, derived from arithmetic, not a guess about cut identity. Cost vs the old
slot-preserved path: the handful of genuinely-binding survivors start BASIC and
get pivoted back — a few warm pivots, never a correctness issue.

**Sequence** (the core sub-basis cannot be installed on the core-only LP — it is
inconsistent when cuts bind — so cuts are added first):

```
load_model(core)                          // fixed core structure, positionally stable
add_rows(initial resident subset)         // k2 history + this-iter new cuts
reconstruct(core copied + cuts BASIC + repair)
solve(Some(full_basis))                   // single solve via the existing warm path
   └─ inner loop:  score omitted cuts at x*
                   add_rows(violated ≤ nadic)   // HiGHS auto-extends slacks BASIC
                   solve(None)                  // warm restart from retained internal basis
                   … until none violated
```

The inner-loop adds never go through `reconstruct_basis` — the solver extends the
basis itself. `reconstruct_basis` seeds only the initial solve per (stage, solve).

**MPI wire-format consequence.** Dropping `cut_row_slots` shrinks `CapturedBasis`
to a core-only basis — but that struct is broadcast across ranks with a versioned
layout. Per the CLAUDE.md rule, the change must **bump
`BASIS_BROADCAST_WIRE_VERSION` and update `to_broadcast_payload` /
`try_from_broadcast_payload` together** (`workspace.rs`). Upside: a smaller,
core-only broadcast payload. (Keeping `cut_row_slots` populated-but-ignored is a
safe interim if we want to defer the wire-format change.)

### 5.3 Residency policy — forced monotonic growth, rank-deterministic seeding

Two hard constraints, discovered in review, shape this — it is **not** a free
choice between "reset" vs "persist":

**(a) No row removal → in-LP residency can only grow within a stage-visit.** The
solver FFI exposes `add_rows` but **no delete/remove** (verified across both
backends). So "reset to the initial set per solve" is not cheaply achievable:
a full `load_model` reload per solve would kill the cross-solve re-pin fast path,
and deactivating a cut by `±∞` bounds leaves its row in the factorization (no real
shrink). Residency is therefore **append-only within a stage-visit** and resets
only at the **`load_model` that already happens when the worker (re)enters a stage
each iteration**. This maps directly onto the existing append-only `CutRowMap`
(`append_new_cuts_to_lp`), which already tracks slot→LP-row and never removes —
**no new teardown machinery, and no `remove_rows` FFI to add.** Practical
consequence: within one stage-visit the resident set converges to the **union of
cuts that bind across that visit's solves**; DCS wins iff that union stays small
(high cross-solve binding-set overlap — the bet we take by skipping measurement,
and the regime production single-cut SDDP validates; cobre's multi-cut pool makes
it worth watching, see §5.4).

**(b) The initial set must be rank-count-deterministic.** Cobre requires
bit-identical results across MPI rank counts (§4). The initial set is only a
warm-start hint — by exactness the optimum is independent of it — **but** under
degenerate duals a different resident set can select a different (valid) cut, so a
per-worker _solve-trace_ history would leak rank-count into the generated cuts.
**Fix: seed the initial set from the cut pool's synchronized per-slot metadata,
not from per-worker traces.** `cut_selection.rs` already maintains `active_count`
and `last_active_iter` per cut, updated deterministically each iteration from the
**MPI-gathered** visited states — exactly the "active in the last `k2` iterations"
signal, already rank-independent. `x*` itself is scenario-deterministic (a
function of the LP, not the rank), so scoring + a metadata-seeded initial set is
fully rank-deterministic.

With those fixed, the policy is the paper's: **per-solve lazy with a `k2`
initial set**, applied uniformly to both passes, where the initial set _is_ the
warm-start mechanism that compensates for the relaxed fixed-shape (a solve seeded
with the recently-active cuts already contains most of what will bind → few inner
iterations, cheap `reconstruct_basis`).

Initial-set granularity: start **per-stage** (cuts with
`current_iter − last_active_iter ≤ k2` at this stage), generalize to per-visited-
state later if warranted. If per-solve residency churn ever erodes the cross-solve
warm-start more than the `k2` seed recovers, the fallback is a **per-(stage,
iteration) fixed resident subset** shared by all that stage's solves (chosen up
front from the same synchronized metadata, fully preserving the re-pin fast path
and staying rank-deterministic). Tuning-time decision, not a v1 blocker.

### 5.4 Candidate scoring

The pool already stores `(intercept, coefficients)` per slot; `x*`/`θ*` are read
from the `SolutionView`. **Reuse the existing `gemm_block` kernel** (`gemm.rs`,
`V = coef · stateᵀ`) — it is the same operation `cut_selection.rs` already runs to
evaluate cuts at states, it is bit-deterministic, and reusing it keeps DCS scoring
consistent with selection. Per §4, **unscale `x*`/`θ*` by `col_scale`** before
scoring (the kernel and stored coefficients are in raw space; the live solution is
scaled), and apply the violation test `θ*_raw < intercept + ∇·x*_raw − ε_viol`.
Scoring **must not allocate on the hot path** (reuse a scratch buffer, as
`cut_selection.rs` does).

**Cost is the dominant perf risk, and we have opted out of measuring it.** Scoring
is `O(num_candidates × state_dim)` per inner iteration. Cobre is **multi-cut** —
one cut per (forward pass, stage, iteration) — so the candidate pool reaches
~60k active cuts at production scale, roughly **96× a single-cut NEWAVE pool**.
Scoring tens of thousands of candidates every inner iteration could swamp the
LP-solve savings. Mitigations, in order: (1) the `gemm_block` batched kernel; (2)
score only **non-resident** candidates; (3) if it still dominates, bound DCS's own
`k1` candidate window (§7) — **never** enable Level1 concurrently. This is the one
place skipping a measurement carries real risk; if Phase-1 wall regresses, instrument
the scoring/solve split before tuning blindly.

### 5.5 Parallel/MPI

NEWAVE notes SC adds inter-processor communication for active-cut info. Cobre needs
**no new collective**: the initial set is seeded from the cut pool's already-
synchronized per-slot metadata (`active_count`/`last_active_iter`, updated from the
MPI-gathered visited states), so it is rank-count-deterministic **without** sharing
a per-worker trace (see §5.3b — the per-worker-local framing of an earlier draft
would have broken the cross-rank determinism rule and is rejected). Per-worker
scratch (scoring buffers, the `CutRowMap`) stays local and carries no cross-rank
state.

### 5.6 Interaction with the solver retry / escalation ladders

An inner `solve(None)` that fails routes through the backend's recovery path —
HiGHS's 12-level ladder (which calls `clear_solver` at some levels) or CLP's
escalation (`reset_cold_basis`). Either **discards the warm basis**, so the
remaining inner re-solves in that loop run cold (slower) until the loop ends and
the next initial solve reseeds. This is rare (~0.15% of solves at production scale)
and **still correct** — but the inner loop must not _assume_ a warm basis survives
every re-solve; treat a cold mid-loop re-solve as normal. No special handling
beyond not depending on warm-state invariants inside the loop.

### 5.7 Other cut-laden LP-solve sites (simulation, lower-bound eval)

Forward and backward are not the only places cobre solves cut-laden LPs:

- **Simulation** evaluates the converged policy against the full cut pool. Under
  the unified-LP-solving priority it should **also** use DCS, run to **exactness**
  (no cut generation, so — like forward — it only needs the correct primal; an
  early stop would mis-state realized cost). Include it in scope alongside forward.
- **Lower-bound evaluation** (`lower_bound.rs`, already incremental via
  `append_new_cuts_to_lp`) **must be exact** if DCS is applied: under-resident
  cuts would let `θ` fall → **underestimate the lower bound** (an invalid, if
  conservative, bound). Cheapest-safe choice for v1: leave LB eval at all-cuts (it
  runs once per iteration, not on the hot path) and revisit only if it shows up in
  profiles.

## 6. Phased implementation plan

- **Phase 1 — shared pass-agnostic lazy loop, backward first.** Initial set
  seeded from synchronized cut metadata (§5.3b); core (= base template, §5.1) +
  incremental construction; uniform-BASIC reconstruction (§5.2); raw-space scoring
  via `gemm_block` with `col_scale` unscaling and `ε_viol` (§4, §5.4); bounded
  inner loop + TC fallback (§4); monotonic in-LP residency via `CutRowMap` (§5.3a).
  Gate behind `training.cut_selection.method = "dynamic"` (mutually exclusive with
  `level1`/`dominated`). **Validate:** exactness (same optimum as all-cuts) on
  deterministic cases; warm restart survives `add_rows` (small inner-solve
  iteration counts — else flip the inner profile to presolve-off); and
  **bit-identical results across MPI rank counts** (§4) on a small case. Decide the
  `CapturedBasis` wire-format change vs the populated-but-ignored interim (§5.2).
- **Phase 2 — forward + simulation + hyperparameters.** Wire the same routine into
  the forward and simulation passes (run-to-exactness, §4/§5.7); add the `k2`
  window history; expose `k1`/`k2`/`nadic`/`ε_viol`/start-iteration; sweep on a
  production-scale case against all-cuts and Level1/Dominated, **instrumenting the
  scoring-vs-solve time split** (§5.4) since scoring cost is the chief unknown.
- **Phase 3 — refinements (if warranted).** Per-visited-state initial-set
  granularity; per-(stage, iteration) fixed-subset residency fallback (§5.3);
  bounded-`k1` candidate window if scoring dominates (§5.4, §7); CLP inner-loop
  micro-optimizations (§5.1); lower-bound-eval DCS (§5.7); `CapturedBasis`
  wire-format shrink if deferred from Phase 1.

## 7. Open questions (deferred, not blocking)

- **Level1 ⊕ DCS composition.** Running Level1 to prune the candidate _pool_
  (bounding DCS's `O(candidates)` scoring cost and pool memory) while DCS manages
  residency is coherent but sacrifices `k1 = ∞` exactness and adds interference
  complexity. **Out of scope for v1.** If candidate scoring dominates at scale,
  the in-scope lever is bounding DCS's own `k1` candidate window, not enabling
  Level1 concurrently.
- **CLP inner-loop micro-optimizations.** Whether to drop CLP's per-`add_rows`
  CSC mirror rebuild and/or add a bulk basis setter — decide only if CLP+DCS is
  measured to be inner-loop-bound (HiGHS is the default and leaner backend).
- **Residency fallback.** Whether the per-`k2` initial set sufficiently preserves
  the cross-solve warm-start, or a per-(stage, iteration) fixed subset is needed
  (§5.3) — a tuning-time call.
