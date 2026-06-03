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

"How many cuts are resident in the solved LP, and how is that managed" is a
spectrum. Each point trades LP size against mutation and warm-start complexity:

| Strategy                     | Resident cut rows               | LP mutation                                           | Warm-start need                                        | Status                      |
| ---------------------------- | ------------------------------- | ----------------------------------------------------- | ------------------------------------------------------ | --------------------------- |
| **Shrink-bake** (current)    | active only                     | rows added/removed/reordered across iters             | slot-identity reconstruction (`full`) to survive churn | implemented                 |
| **Shrink-bake + core / off** | active only                     | same                                                  | core-only / none (`core`, `off`)                       | implemented (toggle)        |
| **Static loose-RHS**         | **all** (inactive at ±∞ RHS)    | **none** structurally (append-only; only RHS toggles) | trivial / length-keyed (rows never move)               | **proposed (this section)** |
| **Dynamic (NEWAVE SC)**      | minimal, grown lazily per solve | rows _added_ within a solve                           | indispensable, across inner adds                       | designed, deferred          |

### Static loose-RHS LP (the alternative to mutating the LP)

Instead of shrinking the baked template to active cuts, **bake every populated
cut as a row and "deselect" a cut by setting its row bound/RHS to a loose value
(±∞ sentinel)** so it is trivially satisfied and never binds. The LP structure is
then **static** (append-only as cuts are built; rows are never removed or
reordered), which has a real upside and a real cost:

- **Upside**: warm-start becomes trivial — the stored basis aligns by position
  (length-keyed), so the slot-identity reconstruction machinery is **unnecessary
  by construction**. No churn, no slot bookkeeping. This is the cleanest possible
  answer to "is the complex reconstruction worth it?": if static loose-RHS is
  competitive, the reconstruction can be retired.
- **Cost**: the LP is **always large** (carries every cut, active or not), so the
  solver must efficiently **ignore the many loose, slack-basic rows**. This is
  pricing-strategy- and presolve-sensitive: a solver that prices over all rows
  pays for the loose ones every iteration.

Cobre's pool already represents an inactive cut with a ±∞ sentinel RHS
(`.claude/rules/sddp.md`), but the per-solve **bake currently excludes inactive
cuts** (`active_cuts()`), so the solved LP shrinks. The static variant would bake
**all** populated cuts (inactive ones at the sentinel RHS) — a change to the
cut-row-batch build / bake path, after which length-keyed warm-start suffices.

**Coupling to solver parameters (the key link).** Past experience: a static
loose-RHS LP was tried, and **HiGHS struggled to handle the many loose rows
efficiently with the solver parameters in use at the time**. This makes the
strategy's viability a **joint** question with the solver-parameter study — the
price strategy (row vs column, hyper-sparse), presolve, and how the backend
treats trivially-satisfied rows likely decide whether static loose-RHS is
competitive. It must therefore be evaluated **crossed with the solver-param
grid**, not in isolation.

**Reference — SPTcpp** (`~/git/SPTcpp`). Investigated as a possible precedent.
What the read actually found (to be confirmed): it uses **CLP**, pre-allocates
cut rows as RHS=0 skeletons and activates them by setting the RHS, but
**deselects cuts by physically deleting rows** (`deleteRows()`) and **cold-starts
every solve** (`initialSolve`, no basis reuse). That is closer to row-mutation +
cold-start than to a clean static-loose-RHS + warm-start design, so SPTcpp does
not cleanly confirm the recollection — flagged for confirmation. The static
loose-RHS strategy is worth evaluating on its own merits regardless.

## 4. Experiment design

Three accelerator axes, crossed, on the deep-pool Mode-C probe (primary) with
Mode-A correctness gating throughout:

| Axis              | Levels                                                                            |
| ----------------- | --------------------------------------------------------------------------------- |
| **Cut residency** | shrink-bake (current) · **static loose-RHS** (all cuts resident) · DCS (deferred) |
| **Warm-start**    | `full` (slot-identity) · `core` (core-only, cuts BASIC) · `off` (cold)            |
| **Cut selection** | none (append-only) · Level1 · Dominated                                           |
| **Solver params** | the per-backend grid from `solver-parameter-tuning.md`                            |

The `core` warm-start level is the direct test of the user's hypothesis: if
`core` ≈ `full` under cut-set churn, the slot-identity reconstruction complexity
is not paying off and can be replaced by the far simpler core-only warm-start.
The **static loose-RHS** residency level is the complementary test: it removes
churn entirely (so length-keyed warm-start suffices and reconstruction is
unnecessary) at the cost of a permanently large LP — and is **only meaningful
crossed with the solver-param grid**, since its viability hinges on how the
backend prices over many loose, slack-basic rows (the regime where HiGHS
previously struggled).

The decisive sub-experiment is the **warm-start × cut-selection 2×N interaction**:

|                    | selection off (big LP) | selection on (small LP)                    |
| ------------------ | ---------------------- | ------------------------------------------ |
| **warm-start on**  | baseline               | current design's bet (churn + reconstruct) |
| **warm-start off** | cold, big LP           | cold, small LP                             |

Readout:

- If **warm-start on / selection off** ≫ **warm-start off / selection off**, then
  warm-start is a large win (expected from NEWAVE) — basis reuse is justified.
- If **warm-start off / selection on** (small cold LPs) ≈ or beats **warm-start on
  / selection off** (big warm LPs), then aggressive selection + cold-start is
  competitive and **the slot-identity reconstruction complexity is not paying
  off** — the user's core suspicion, validated.
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
3. **Static loose-RHS residency** — **proposed, not yet built**. A bake-path mode
   that bakes **all** populated cuts (inactive ones at the ±∞ sentinel RHS) so the
   solved LP is structurally static and warm-start is length-keyed. Seam: the
   cut-row-batch build / bake path (iterate all populated cuts instead of
   `active_cuts()`; set inactive rows loose), plus a length-keyed (or `core`)
   warm-start since slot-identity reconstruction is then moot. Mid-complexity —
   smaller than DCS, larger than the basis toggle; gate behind a
   `COBRE_TUNE_CUT_RESIDENCY=shrink|static` env toggle to mirror the existing
   seams. Must be evaluated crossed with the solver-param grid.
4. **Dynamic Cut Selection (NEWAVE SC)** — **deferred, designed separately** in
   `dynamic-cut-selection-design.md`. It has real hyperparameters and tricks
   (initial-set window, candidate scoring, `nadic`, basis-recovery dependence,
   incremental row addition) and is not a spike; it will be implemented in the
   near future per that design.

## 6. Open decisions

- **Static loose-RHS toggle**: build the `COBRE_TUNE_CUT_RESIDENCY=static` bake
  mode now (so the cut-residency axis runs alongside warm-start × selection), or
  defer until the warm-start × selection results are in? It is the cleanest test
  of whether the slot-identity reconstruction can be retired outright.
- **Confirm SPTcpp**: verify what `~/git/SPTcpp` actually does (row deletion +
  cold-start vs static loose-RHS + warm-start) — the investigation suggested the
  former, contradicting the recollection.
- **DCS**: design first (done — see `dynamic-cut-selection-design.md`), implement
  in a near-future dedicated effort. The warm-start × existing-selection
  interaction already answers the basis-reconstruction question and runs now.
- Confirm the correctness tolerance and the deep-pool depth `K` (shared with the
  solver-parameter study).
