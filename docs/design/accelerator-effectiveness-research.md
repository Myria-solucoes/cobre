---
title: "Accelerator effectiveness research — cut selection & basis reconstruction"
date: 2026-06-02
status: design
tags: [cobre, sddp, cut-selection, basis-reconstruction, warm-start, benchmark]
companion:
  [
    "docs/design/solver-parameter-tuning.md",
    "docs/design/solver-tuning-benchmark-case.md",
    "docs/design/dynamic-cut-selection-design.md",
  ]
references:
  ["ideas/cut-selection.pdf", "ideas/newave-cut-selection.pdf", "~/git/SPTcpp"]
---

# Accelerator effectiveness research

Extends the solver-parameter tuning (see `solver-parameter-tuning.md`) to a
second, higher-leverage question: **do our algorithmic performance accelerators
actually earn their complexity?** The two accelerators under study are **cut
selection** and **basis reconstruction / warm-start**. Both are evaluated on the
same benchmark case (`solver-tuning-benchmark-case.md`), per backend, local
threads, no MPI.

## 1. What the references say

### `cut-selection.pdf` — de Matos, Philpott, Finardi (2015), _Improving the performance of SDDP_

The academic source for the methods Cobre **already implements**:

- **Cut selection** reduces the number of cuts carried in each stage LP. The
  paper's variants: _Last cuts_ (window), **Level 1** (keep any cut within
  tolerance of the per-state max at any visited state — convergence-preserving),
  and **dominated-cut** elimination.
- Best result: increasing forward scenarios per iteration **+** cut selection →
  _"an order of magnitude decrease in computation time with little change in
  solution quality."_

Cobre's `cut_selection.rs` implements `Level1` (de Matos 2015), `Lml1`
(Guigues 2019), and `Dominated` (Tekaya 2012), runtime-configurable via
`training.cut_selection`. They default **off**; post-backward selection is kept,
in-backward selection was evaluated and gated off.

### `newave-cut-selection.pdf` — Diniz et al. (CEPEL, 2020), _Estratégia de Seleção de Cortes_ (the "Dynamic Cut Selection")

NEWAVE's **SC strategy** — the method Cobre does **not** implement — is
**lazy/iterative cut insertion** within each LP solve:

1. Start the LP with a small initial cut set (cuts that were _active_ at this
   state in recent iterations, plus same-iteration backward cuts).
2. Solve → optimum `f*`, state `x*`.
3. For each cut not yet in the LP, compute its value at `x*`; the cuts whose
   value exceeds `f*` are violated (the FCF is under-estimated).
4. Add the `nadic` most-violated cuts; re-solve; repeat until none are violated.

The final solution is **exact** (any omitted cut is inactive). Key findings:

- **§3.4 basis recovery is "indispensable"**: each re-solve adds only a few rows,
  so warm-starting from the previous basis makes the re-solve nearly free. SC
  _relies on_ warm-start.
- **§4.3.3**: because NEWAVE _already_ recovers the basis, the all-cuts (TC) LP is
  _"already very efficient,"_ which **limits** SC's extra benefit to ~50% on
  LP-solve time and less on total time (diluted by problem assembly, cut
  construction, and inter-processor communication; SC itself adds comms).
- **§4.4 / §5**: SC and TC yield **equivalent-quality policies** (same final
  bounds), with small numerical differences from alternate optima — the same
  cross-mode drift Cobre already documents.

## 2. Reframing the hypotheses

The starting intuition was: _"Dynamic cut selection mutates the LP, which might
hurt warm-start, which makes us doubt whether the complex basis reconstruction
helps at all."_ The references sharpen this:

- **LP mutation vs warm-start** — NEWAVE's SC mutates the LP by _adding_ a few
  rows per inner solve and **depends on** warm-start to stay cheap. Cobre's cut
  selection mutates by _deactivating_ cut rows; the baked template shrinks while
  the pool stays append-only, and `reconstruct_basis` re-aligns the stored basis
  by **slot identity** across that churn. Neither mechanism is hostile to
  warm-start; both are _designed around_ it.
- **Is basis reconstruction worth its complexity?** This is the decisive
  question. NEWAVE's evidence says basis recovery is the _dominant_ accelerator
  (it is what makes even the all-cuts LP efficient). But Cobre's reconstruction
  is more complex than a plain warm-start because it must survive cut-set churn
  by slot identity. The complexity is only justified if (a) warm-start itself
  pays off **and** (b) cut-set churn actually occurs in the regimes we run. If we
  run with selection **off** (append-only, no churn), a simpler length-keyed
  warm-start would suffice — the slot-identity machinery earns its keep only when
  selection is **on**.

These become testable claims (§4).

## 3. Current implementation (the seams)

From a read of the cut-selection and warm-start code:

- **Cut selection**: `crates/cobre-sddp/src/cut_selection.rs`
  (`CutSelectionStrategy::{Level1, Lml1, Dominated}`), config
  `training.cut_selection` (`enabled`, `method`, `check_frequency`,
  `tie_tolerance`, `domination_epsilon`, `max_active_per_stage`, …). Selection is
  **post-backward**; deactivation is a flag + sentinel bounds (append-only pool),
  and the per-solve **baked template shrinks to active cuts**
  (`cut/pool.rs::active_cuts`, `bake_rows_into_template`).
- **Warm-start / basis reconstruction**: `basis_reconstruct.rs::reconstruct_basis`
  (slot-identity match), applied at the single hot-path entry
  `stage_solve.rs::run_stage_solve` when `inputs.stored_basis.is_some()`.
  `policy.checkpointing.store_basis` gates **disk** persistence only; in-run
  warm-start always happens when a stored basis is present.
- **Warm-start toggle (implemented)**: `COBRE_TUNE_WARMSTART` selects, once, at
  the single `run_stage_solve` basis branch — inert/default `full`:
  - `full` — slot-identity reconstruction (`reconstruct_basis`), current default.
  - `core` — warm-start the LP core only; **every cut row starts BASIC**
    (non-binding) via `reconstruct_basis_core`, skipping slot-identity matching.
    The simplex pivots in whichever cuts actually bind.
  - `off` — no warm-start; every solve cold-starts.
    All three are correctness-neutral (verified: identical converged bound on a
    deterministic case); they change only the solve path.
- **Dynamic Cut Selection (NEWAVE SC)** is **not implemented**. Designed for
  near-future implementation in `dynamic-cut-selection-design.md` — a new in-solve
  lazy-row-generation loop interacting with the rebake and warm-start.

## 3a. Cut-residency strategies (the design space)

"How many cuts are resident in the solved LP, how is that managed, and is the
basis reused" — the live strategies:

| Strategy                                     | Resident cut rows               | Warm-start                              | Status                                      |
| -------------------------------------------- | ------------------------------- | --------------------------------------- | ------------------------------------------- |
| **Shrink-bake + full reconstruct** (current) | active only                     | slot-identity (`full`) to survive churn | implemented                                 |
| **Shrink-bake + core / off**                 | active only                     | core-only / none (`core`, `off`)        | implemented (toggle)                        |
| **SPTcpp-style: shrink + cold**              | active only                     | **none** (cold-solve every LP)          | the `off` × selection-on cell — no new code |
| **Dynamic (NEWAVE SC)**                      | minimal, grown lazily per solve | indispensable, across inner adds        | designed, deferred                          |

### SPTcpp-style: shrink + cold-start, no warm-start (the alternative we will test)

Settled from a direct read of `~/git/SPTcpp` (`C_Solver.h` + the cut paths),
SPTcpp's design is **"no warm-start + aggressive cut deletion → a small,
cold-solved LP"**:

- LP solver **CLP** (`ClpSimplex` / `OsiClpSolverInterface`; optional Gurobi via
  `#ifdef GRB`).
- **No warm-start of any kind** — every solve is a cold `initialSolve` /
  `initialDualSolve` (`C_Solver.h:1732`), perturbation and scaling on. A full-tree
  search of `src` finds no `CoinWarmStart`/`setBasis`/`getWarmStart`/`resolve`.
- Cut **deselection physically deletes the row** (`remRestricao` → CLP
  `deleteRows`, `C_Solver.h:2959`), keeping the LP small. `setRHSRestricao`
  (`:2745`) only sets a row's one-sided bound. Rich selection (dominance level
  0–10, selected/deselected sets, NEWAVE-window flag).
- The design is internally consistent: once you do not warm-start, structural
  stability is worthless, so deleting deselected rows to keep the LP small is the
  right move.

**Cobre already covers this with no new code.** Cobre's pool is append-only, but
the per-solve **baked template includes only `active_cuts()`**, so with cut
selection on the _solved_ LP shrinks to the active set — the same solve cost as
SPTcpp's physical deletion (the append-only pool is a memory/bookkeeping
difference, not a solve-cost one). Combined with `COBRE_TUNE_WARMSTART=off`, this
**is** the SPTcpp-style configuration: cold-solve a small LP. It is the
`warm-start off × selection on` cell of the matrix below and needs no new
instrument. SPTcpp performing well in production is a real-world data point that
**this cell may be competitive** — exactly the suspicion driving this study.

> **Parked — static loose-RHS LP.** Keeping all cuts resident and deselecting by
> loosening the row RHS (so the LP is structurally static and warm-start is
> length-keyed) was considered and **dropped from the test matrix**: SPTcpp does
> not do it, a permanently large LP is suspected ineffective, and a prior attempt
> saw HiGHS struggle to price efficiently over the many loose rows. Recorded here
> only as a not-pursued alternative.

## 4. Experiment design

Three accelerator axes, crossed, on the deep-pool Mode-C probe (primary) with
Mode-A correctness gating throughout:

| Axis              | Levels                                                                 |
| ----------------- | ---------------------------------------------------------------------- |
| **Warm-start**    | `full` (slot-identity) · `core` (core-only, cuts BASIC) · `off` (cold) |
| **Cut selection** | none (append-only) · Level1 · Dominated                                |
| **Solver params** | the per-backend grid from `solver-parameter-tuning.md`                 |

The `core` warm-start level is the direct test of the user's hypothesis: if
`core` ≈ `full` under cut-set churn, the slot-identity reconstruction complexity
is not paying off and can be replaced by the far simpler core-only warm-start.
The `off` × selection-on corner is the **SPTcpp-style alternative** (cold-solve a
small LP, no warm-start; see §3a) — a production-proven point that needs no new
code, just `COBRE_TUNE_WARMSTART=off` plus cut selection enabled. (DCS is a fourth
residency point, deferred — see `dynamic-cut-selection-design.md`.)

The decisive sub-experiment is the **warm-start × cut-selection 2×N interaction**:

|                    | selection off (big LP) | selection on (small LP)                    |
| ------------------ | ---------------------- | ------------------------------------------ |
| **warm-start on**  | baseline               | current design's bet (churn + reconstruct) |
| **warm-start off** | cold, big LP           | cold, small LP                             |

Readout:

- If **warm-start on / selection off** ≫ **warm-start off / selection off**, then
  warm-start is a large win (expected from NEWAVE) — basis reuse is justified.
- If **warm-start off / selection on** (small cold LPs — the **SPTcpp-style**
  configuration) ≈ or beats **warm-start on / selection off** (big warm LPs), then
  aggressive selection + cold-start is competitive and **the slot-identity
  reconstruction complexity is not paying off** — the core suspicion, validated.
- The **selection on / warm-start on** cell tests whether reconstruction survives
  churn _and_ helps; comparing it to **selection on / warm-start off** isolates
  the value of reconstruction _specifically under churn_ (the only regime where
  its complexity is exercised).
- **DCS** is the most aggressive selection: if it keeps LPs small enough that
  cold-start is fine, basis reconstruction may be unnecessary; if DCS + warm-start
  wins (NEWAVE's result), reconstruction is vindicated.

**Metrics & guard**: as in `solver-parameter-tuning.md` — `backward_solve_seconds`

- per-iteration `backward_wall_ms` primary; final LB + first-stage cost within
  tolerance as the correctness gate (selection and DCS must preserve policy
  quality, per NEWAVE §4.4); watch `solve_stats.retried`.

## 5. Instruments

1. **Warm-start toggle** — **built**. Env-gated `COBRE_TUNE_WARMSTART`
   (`full`|`core`|`off`), resolved once at the single `run_stage_solve` basis
   branch, inert by default. The `core` mode (`reconstruct_basis_core`) warm-starts
   the LP core and starts every cut row BASIC; `off` cold-starts. Verified
   correctness-neutral on a deterministic case.
2. **Cut-selection variants** — no code needed; driven by `training.cut_selection`
   config, templated by the harness.
3. **SPTcpp-style (shrink + cold)** — **no new code**. The `warm-start off ×
selection on` configuration (`COBRE_TUNE_WARMSTART=off` + cut selection
   enabled) already reproduces it (small cold-solved LP); see §3a.
4. **Dynamic Cut Selection (NEWAVE SC)** — **deferred, designed separately** in
   `dynamic-cut-selection-design.md`. It has real hyperparameters and tricks
   (initial-set window, candidate scoring, `nadic`, basis-recovery dependence,
   incremental row addition) and is not a spike; it will be implemented in the
   near future per that design.

## 6. Open decisions

- **SPTcpp — settled** (see §3a): it does **row deletion + cold-start, no
  warm-start** (CLP `initialSolve` + `deleteRows`), i.e. the warm-start `off` ×
  selection-on cell — runnable today with no new code. Static loose-RHS is parked.
- **DCS**: design first (done — see `dynamic-cut-selection-design.md`), implement
  in a near-future dedicated effort. The warm-start × existing-selection
  interaction already answers the basis-reconstruction question and runs now.
- Confirm the correctness tolerance and the deep-pool depth `K` (shared with the
  solver-parameter study).
