---
title: "Dynamic Cut Selection (NEWAVE SC) — implementation design"
date: 2026-06-02
status: design (deferred implementation)
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
companion: ["docs/design/accelerator-effectiveness-research.md"]
---

# Dynamic Cut Selection (NEWAVE SC) — implementation design

A design for implementing **Dynamic Cut Selection** (DCS) — NEWAVE's _Estratégia
SC_, Diniz et al. 2020 — in Cobre. This is **not** a spike: DCS is a lazy
constraint-generation loop _inside_ each stage LP solve with several
hyperparameters and correctness-critical details. This document specifies the
algorithm, the hyperparameters, the interactions with Cobre's architecture, and
a phased implementation plan, so it can be built deliberately in a near-future
effort.

## 1. What DCS is (and is not)

Cobre today **bakes every active cut as a structural row** and solves each stage
LP once (the all-cuts "TC" strategy). Its existing cut _selection_ (Level1,
Dominated) **deactivates** cuts between iterations (post-backward), shrinking the
baked template. DCS is different: it is **lazy row generation within a single LP
solve**.

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

## 2. Hyperparameters

| Param           | Meaning                                                                            | NEWAVE setting / finding                                                                               |
| --------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `k1`            | Window of past iterations whose cuts are _candidates_ at all                       | `k1 = max_iters` (all cuts) → guarantees equivalence with all-cuts (TC); never silently drop old cuts  |
| `k2`            | Window for the _initial_ active set (cuts active at this state in last `k2` iters) | tested {5,10,15,20,45}; **`k2 = 5` best**                                                              |
| `nadic`         | Max cuts added per inner iteration                                                 | tested {5,10,15}; ≥5 fine, slightly better at 15; _"determination above 5 barely affects performance"_ |
| start iteration | First SDDP iteration to apply DCS (need cuts to exist)                             | from iteration 2                                                                                       |
| phase           | Apply to backward only, or forward+backward                                        | NEWAVE v23.1.3 applied **backward only** (the bulk of LP-solve cost)                                   |

Cobre defaults to propose: `k1 = ∞` (all cuts candidate), `k2 = 5`, `nadic = 10`,
start at iteration 2, **backward-only first** (matches where Cobre is
solver-bound). All exposed as `training.cut_selection` config (a new
`method = "dynamic"` variant) so the tuning harness can sweep them.

## 3. Correctness contracts

- **Exactness** — the stop condition (no violated candidate) guarantees the LP
  optimum equals the all-cuts optimum. The dual `π` extracted for cut
  construction must come from the _final_ (all-satisfied) LP, so cut gradients are
  unaffected. (Respects the Benders-cut sign / subgradient contract in
  `.claude/rules/sddp.md`.)
- **Policy equivalence, not bit-identity** — SC and TC reach equivalent-quality
  policies but may differ in alternate-optima multipliers, so bounds drift
  slightly across the run (paper §4.4). This is the same cross-mode drift Cobre
  already documents; the correctness gate compares converged bound + cost within
  tolerance, not bit-for-bit.
- **Determinism** — the candidate scoring, the violated-set selection, and ties
  must be deterministic and declaration-order invariant: score by slot, break
  ties by ascending slot id, select a stable top-`nadic`. No `Date`/random input.
- **Append-only pool preserved** — DCS never removes cuts from the pool; it only
  controls which are _resident in the LP at solve time_. Slot identity is retained.

## 4. Interactions with Cobre's architecture (why it's not a spike)

1. **Per-solve LP construction changes shape.** Today `bake_rows_into_template`
   builds one LP with all active cuts; the solver solves once. DCS needs:
   load core + initial cut subset → solve → **incrementally add rows** → warm
   re-solve → loop. This requires an _incremental add-rows + warm re-solve_ path
   on the loaded model, not a full rebake per inner step.
   - Confirm the `SolverInterface` supports incremental row addition with
     warm-start for **both** backends (CLP `add_rows` exists; the HiGHS wrapper's
     incremental-add + warm path must be verified or added). This is the single
     biggest implementation dependency.
2. **Candidate scoring needs the omitted cuts' coefficients and `x*`.** The pool
   already stores `(intercept, coefficients)` per slot; `x*` is read from the
   solution view. Scoring is `O(num_candidates × state_dim)` per inner iteration
   — vectorizable, but with thousands of candidates it is itself non-trivial and
   must not allocate on the hot path (reuse a scratch buffer).
3. **Initial-set selection needs per-state active-cut history.** "Cuts active at
   this state in the last `k2` iterations" requires tracking, per stage (and
   ideally per visited state), which slots were active recently. This is new
   bookkeeping; a simpler first cut: initial set = cuts active at the _previous
   iteration's_ solve at this stage (a 1-deep history), generalize to `k2` later.
4. **Warm-start is the enabler, not the victim.** Each inner re-solve adds rows;
   the natural basis status for an added cut row is BASIC (slack basic) — exactly
   the `reconstruct_basis` "new cut → BASIC" path and the `core` warm-start mode.
   DCS should reuse that machinery. (This is precisely why the
   accelerator-effectiveness study front-runs the warm-start question.)
5. **Parallel/MPI.** NEWAVE notes SC adds inter-processor communication for active-
   cut info. Cobre's backward is parallel-by-scenario per worker; the initial-set
   history is per-worker/per-stage local, so the first implementation needs no new
   collective. Revisit if initial-set sharing across workers is wanted.

## 5. Phased implementation plan

- **Phase 0 — measure the opportunity.** Extend the `probe_k_disaggregated`
  example to report, per stage, how many resident cuts actually _bind_ at the
  solution vs how many are carried. If only a handful bind out of hundreds, DCS
  has headroom; if most bind, it does not. Decide go/no-go from this.
- **Phase 1 — minimal lazy loop, backward-only.** Initial set = previous-solve
  active cuts (1-deep history); incremental add + warm re-solve; `nadic`, stop on
  no-violation. Gate behind `training.cut_selection.method = "dynamic"`. Validate
  exactness (same optimum as all-cuts) on deterministic cases.
- **Phase 2 — hyperparameters + tuning.** Add `k2` window history; expose
  `k2`/`nadic`; sweep on a production-scale benchmark case against all-cuts and
  Level1/Dominated (cf. `solver-tuning-results.md`).
- **Phase 3 — forward pass + (if warranted) cross-worker initial-set sharing.**

## 6. Open questions

- Does the HiGHS wrapper support incremental row addition with basis warm-start
  mid-solve, or must each inner step rebake (which would erode DCS's benefit)?
  This gates the whole approach and should be answered in Phase 0.
- Initial-set history granularity: per-stage (cheap) vs per-visited-state
  (closer to NEWAVE, more state). Start per-stage.
- Where DCS sits relative to existing deactivation-based selection (Level1 /
  Dominated): orthogonal (DCS controls LP-residency per solve; selection controls
  pool activity across iterations) or mutually exclusive in practice? Decide once
  Phase 1 data exists.
