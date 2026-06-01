# Backward-Pass Node Parallelism (Parallel-by-Node within Rank)

> **Status**: Draft / Proposed (2026-05-29). Not yet implemented. This
> document proposes adding a _parallel-by-node_ dimension to the backward
> pass so that worker utilization is decoupled from the forward-pass count.

## Summary

Cobre's backward pass is parallelized **purely by scenario** (trial point):
`process_stage_backward` distributes the rank's trial points across rayon
workers, and each worker solves its trial points' `|Ω_t|` opening subproblems
**sequentially** (`backward.rs:599`). Worker utilization is therefore capped at
`min(trial_points_per_rank, n_workers)`. To use all workers a rank must process
at least `n_workers` trial points, i.e. the total forward count must satisfy
`F ≥ n_ranks × n_workers` (production: `F ≥ 2 × 96 = 192`).

This forces operators onto the one scaling lever cobre exposes — **raising the
forward count** — which is independently harmful for two reasons established by
this project's investigation and the reference literature (see
[References](#references)):

1. **Pool-driven LP cost** (measured): more forwards/iteration grows the cut
   pool faster, so more solves run against a large pool; per-solve wall rises
   with pool size.
2. **Myopic early cuts** (Ávila et al. 2022): more forward samples generate more
   low-quality cuts in early iterations, wasting work and degrading
   convergence-per-unit-time.

This proposal adds **parallel-by-node within a rank (PN)**: when a rank has
fewer trial points than workers, distribute each trial point's `|Ω_t|` opening
subproblems across the surplus workers. This **decouples the forward count from
the worker count**, so `F` can be chosen for algorithmic reasons (the true
optimum, often `F ≪ n_ranks × n_workers`) instead of being floored by a
hardware-utilization constraint.

The scheme is, by construction:

- **Result-identical** to the current backward pass (same per-opening LP solves,
  same deterministic probability-weighted reduction) — it changes _who_ computes
  each opening, not _what_ is computed. No convergence or correctness risk.
- **MPI-transparent**: the rank boundary, cut sync, and cross-rank `allgatherv`
  are unchanged. Cross-rank bit-determinism is preserved trivially.
- **Determinism-preserving**: the per-cut reduction reads opening outcomes in
  `ω`-order regardless of which worker produced each, so cuts are bit-identical.

Production has **`|Ω_t| = 20` openings**, so PN provides up to a **20× wider**
parallel work pool per trial point.

## Motivation

### The forward-count bind

Let `F` = forward passes, `R` = MPI ranks, `W` = rayon workers/rank, and
`|Ω|` = openings/stage. Per stage the backward pass has `(F/R) × |Ω|`
independent subproblem solves available on each rank, but cobre parallelizes
only the `F/R` trial-point dimension.

|                   | full-utilization condition | production threshold (`R=2, W=96, | Ω    | =20`)      |
| ----------------- | -------------------------- | --------------------------------- | ---- | ---------- | --- | ------- |
| **PS (today)**    | `F/R ≥ W`                  | `F ≥ 192`                         |
| **PN (proposed)** | `(F/R)·                    | Ω                                 | ≥ W` | `F ≥ ⌈R·W/ | Ω   | ⌉ = 10` |

PN saturates 96 workers/rank at `F ≈ 10–20` instead of `F ≥ 192` — a ~10–20×
reduction in the forward count required for full utilization.

### Why a lower forward count is desirable

At equal total work (`F × iterations` constant), a lower `F` with more
iterations:

- keeps the per-iteration cut pool smaller → cheaper LP solves (pool-cost
  finding);
- generates fewer, higher-quality cuts per iteration when value functions are
  still myopic → better convergence-per-iteration (Ávila et al. 2022).

PN does **not** assert that `F = 20` is optimal. It removes the _artificial
floor_ `F ≥ R·W` so that the operator can tune `F` to the algorithmic optimum
`F*` (which balances pool cost, cut quality, and per-iteration barrier overhead;
see [Tradeoffs](#tradeoffs-and-the-gain-ceiling)) rather than to the hardware.

### Underutilization today

The cliff is real and measured: a constant-product local sweep showed
`forwards = 2` with 4 workers (2 idle) ran **+60%** slower than
`forwards = 4`. `partition(local_work, n_workers, w)` hands empty ranges to
surplus workers (`backward_pass_state.rs:1039`).

## Background: current backward architecture

Hot path, per backward stage `t` (descending), on each rank:

1. `process_stage_backward` (`backward_pass_state.rs:986`):
   `workspaces.par_iter_mut()` over the rank's `n_workers` `SolverWorkspace`
   instances. Each worker is assigned a contiguous block of trial points via
   `partition(local_work, n_workers, w)` (`:1039`).
2. For each assigned trial point `m`, the worker calls `load_backward_lp` to
   reset the LP to the baked template + active cuts (`:1049`), then
   `process_trial_point_backward(…, m)` (`backward.rs:575`).
3. Inside `process_trial_point_backward`, `for omega in 0..|Ω_t|`
   (`backward.rs:599`): `patch_opening_bounds` sets the realization's RHS, then
   `run_stage_solve(Phase::Backward)`; duals are extracted and
   `accumulate_opening_outcome` stores `(intercept, coefficients, objective)`
   into `ws.backward_accum.outcomes[omega]`.
4. After the opening loop, `risk_measures[t].aggregate_cut_into(outcomes[..|Ω|],
probabilities, …)` (`backward.rs:677`) reduces the per-opening outcomes into
   one cut → `StagedCut { trial_point_idx: m, intercept, coefficients, … }`.
5. Cuts from all workers are collected and **sorted by `trial_point_idx`** after
   the parallel region for deterministic FCF insertion (`backward.rs:82-84`).

Two structural facts that constrain the design:

- **Warm-start chaining.** The stored warm-start basis is resolved/saved **only
  at `ω = 0`** (`backward.rs:610-614`, `:668-669`), keyed per trial point.
  Openings `ω = 1..|Ω|-1` pass `stored_basis = None` and warm-start off the
  solver's _retained_ basis from the previous opening (only the RHS changed via
  `patch_opening_bounds`). So a trial point's openings form an efficient
  **dual-simplex warm-start chain** behind a single `load_backward_lp` reload.
- **The per-trial-point reload.** `load_backward_lp` (`:1049`) is the dominant
  fixed per-trial-point cost (≈ the ~220 ms base measured in the local sweep:
  HiGHS model reset + template/cut load + factorization).

The MPI layer already carries per-`(rank, worker, opening)` granularity
(`bwd_max_openings`, `per_opening_stats`, the stats `allgatherv` at
`backward_pass_state.rs:617-672`), so the opening is already a first-class index
in the data model.

## Proposed design

### The two-dimensional decomposition

Keep **PS across ranks** (each rank owns a disjoint subset of trial points;
unchanged). Add **PN within a rank**: the unit of work distributed across a
rank's workers becomes a _contiguous opening-block of a single trial point_
rather than a whole trial point.

Per stage, on a rank with `P = local_work` trial points and `W` workers:

- **If `P ≥ W`** → behave exactly as today (assign trial points to workers;
  each worker reloads once per trial point and chains its `|Ω|` openings). PN is
  inactive; zero overhead.
- **If `P < W`** → give each trial point `⌊W/P⌋..⌈W/P⌉` workers; each such
  worker takes a **contiguous block** of that trial point's openings. Every
  worker therefore touches **exactly one trial point → exactly one
  `load_backward_lp` reload**, and chains the openings within its block.

This "one trial point per worker" invariant is the crux: it keeps the reload
count at **1 per worker** (same as PS), so PN parallelizes the _solve_ work
without multiplying the expensive reload.

### Opening-outcome gather and deterministic reduction

When a trial point's openings span multiple workers, its
`outcomes[0..|Ω|]` must be **gathered** before `aggregate_cut_into`:

1. Each worker writes its computed openings into the shared per-trial-point
   `outcomes` array at the correct `ω` slots (no contention: disjoint `ω`).
2. After the parallel region, for each trial point `m`, reduce `outcomes[0..|Ω|]`
   in `ω`-order via the existing `aggregate_cut_into`.

Because the reduction reads outcomes in `ω`-order and each per-opening outcome
is bit-identical regardless of which worker solved it, **the resulting cut is
bit-identical to today's**. Determinism (and cross-rank invariance) is
therefore preserved by construction; the StagedCut `trial_point_idx` sort is
unchanged.

### Warm-start handling under opening-blocking

Splitting a trial point's openings across workers breaks the single
intra-trial-point chain into one chain _per worker block_. Mitigations:

- Each worker chains _within_ its contiguous opening block (only the block's
  first solve pays a non-chained start).
- The `ω = 0` worker computes and publishes the stored basis as today; other
  blocks seed their first solve from that shared `ω = 0` basis (read-only
  broadcast within the rank's shared memory). Since openings differ only in RHS,
  the `ω = 0` basis is a strong warm start for any block's first solve.

Net warm-start penalty ≈ `(workers_per_trial_point − 1)` extra non-chained
solves per trial point, bounded and small relative to the parallelism gained.

### Data-structure changes (intra-rank only)

- Promote the per-opening `outcomes` buffer from per-worker
  (`ws.backward_accum.outcomes`) to a per-`(trial_point, opening)` view the
  participating workers write into (e.g. a `n_local_trial_points × |Ω|` arena
  sliced disjointly by `ω`).
- A work-assignment helper mapping `worker → (trial_point, ω_start..ω_end)` for
  the `P < W` case (generalizing `partition`).
- Cut aggregation (`aggregate_cut_into`) and the StagedCut sort move to a
  post-parallel reduction over the gathered arena. No change to cut content.

No changes to: the MPI protocol, cut sync, FCF pool, cut selection, risk
measures, or the forward pass.

## Tradeoffs and the gain ceiling

Per-stage wall in the underutilized regime (`P < W`), with `r` = reload cost and
`s` = per-opening solve cost:

- **PS today**: `≈ r + |Ω|·s` (one active worker does a full trial point; `W−P`
  idle).
- **PN**: `≈ r + (|Ω|·P/W)·s` (each worker does one reload + its opening block).

PN is faster whenever `P < W`, and the speedup of the _solve_ portion is
`W/P`. **But the `r` (reload) term is paid once per worker in both schemes and
is not parallelized away** — so the achievable gain is bounded by the
reload-to-solve ratio:

- If solves dominate (`|Ω|·s ≫ r`): PN approaches a `W/P×` backward speedup.
- If the reload dominates (`r ≫ |Ω|·s`): both schemes are reload-bound and PN's
  gain is small.

**Implication:** PN should be pursued _together with_ reducing the
per-trial-point reload. The current `load_backward_lp` fully resets HiGHS per
trial point; if that can be reduced to an incoming-state RHS patch (as openings
already do via `patch_opening_bounds`) the base cost drops for **both** PS and
PN and lifts PN's ceiling. This is tracked as complementary work, not a
prerequisite.

A second tradeoff: enabling low `F` shifts work toward **more (sequential)
iterations**, each with a synchronization barrier (cut sync, LB eval, MPI
allreduce). The per-iteration fixed overhead is multiplied by the iteration
count. The optimum `F*` balances pool cost + cut quality (favor low `F`) against
per-iteration barrier overhead (favor high `F`). PN makes the low-`F` side
_reachable_; it does not by itself determine `F*`.

## Granularity alternatives considered

1. **Flat `(trial_point, opening)` work pool** (max parallelism / work-stealing).
   Rejected as the default: workers hop between trial points → multiple
   `load_backward_lp` reloads per worker → reload cost multiplies. Also
   complicates per-trial-point basis ownership.
2. **Opening-blocked, one trial point per worker** (recommended). Preserves the
   1-reload-per-worker invariant and most warm-start chaining; activates only
   when `P < W`.
3. **Adaptive** (recommended wrapper): use PS when `P ≥ W`, switch to (2) when
   `P < W`. Zero overhead in the already-saturated regime.

## Correctness and determinism

- **Cut validity**: each cut is the same probability-weighted (risk-adjusted)
  aggregate of the same per-opening dual outcomes as today; only the assignment
  of openings to workers changes. Cuts are valid by the same argument as the
  current scheme (no stale-cut approximation as in async PN of Ávila et al.).
- **Bit-determinism within a mode**: the reduction is over `ω`-ordered outcomes;
  the FCF insertion order is the existing `trial_point_idx` sort. Both are
  independent of worker scheduling.
- **Cross-rank invariance**: the rank boundary and all MPI collectives are
  unchanged, so results are identical across rank counts (the project's
  declaration-order-invariance contract holds unchanged).

These properties make PN-within-rank a low-risk, _result-preserving_
optimization — in contrast to async schemes, which trade reproducibility and
exact results for latency hiding.

## Validation plan

1. **Reload/solve ratio probe** (decides the gain ceiling _before_ building):
   instrument one backward sweep to separate `load_backward_lp` time from
   per-opening solve time on the production case. If solves dominate, PN's
   ceiling is high.
2. **Prototype** opening-blocked PN behind a toggle; on the local case at low
   `F` (`F < W`), confirm worker utilization rises and per-stage wall drops
   toward `r + (|Ω|·P/W)·s`.
3. **Determinism tests**: assert the prototype produces **bit-identical cuts and
   final lower bound** vs the current scheme (same `F`, same seed), and
   bit-identical across rank counts. This is the acceptance gate — PN must be a
   no-op on results.
4. **Constant-work A/B**: compare `F ≈ 20` (PN) vs `F = 192` (PS) at equal
   `F × iterations` on the production case (user-owned per the project's
   manual-benchmark policy); measure wall and convergence to locate `F*`.

## Phasing

- **Phase 0** — reload/solve probe (step 1) + opening-width audit across
  production cases (`|Ω|` per stage; confirmed 20 for the primary case).
- **Phase 1** — adaptive opening-blocked PN behind a default-off toggle; the
  determinism gate (step 3) must pass before it is anything but experimental.
- **Phase 2** — optional: reduce the per-trial-point reload to an RHS patch
  (lifts the PN ceiling and speeds up PS too).
- **Phase 3** — make PN the default for the `P < W` regime once validated.

## Open questions

- Magnitude of the reload-vs-solve ratio at production scale (Phase 0 decides).
- Is `|Ω_t|` uniform across stages, or do some stages have far fewer openings
  (limiting PN width there)? Affects per-stage utilization.
- Interaction with risk-averse `aggregate_cut_into` (CVaR reweighting): the
  reduction still reads `outcomes[0..|Ω|]`, so it is unaffected, but confirm no
  per-opening ordering assumption beyond `ω`-index.
- Whether the shared `ω = 0` basis broadcast within a rank is worth the
  coordination vs. letting each block's first solve warm-start cold.

## References

- Ávila, Papavasiliou, Löhndorf, "Parallel and distributed computing for
  stochastic dual dynamic programming," _Computational Management Science_
  19:199–226 (2022). Local copy: `sddp-parallel.pdf`. Defines the
  parallel-scenario / parallel-node taxonomy; shows increasing forward samples
  harms performance and that a parallel-node strategy scales better but was
  limited to shared memory — the gap this design targets via PN-within-rank ×
  PS-across-ranks.
- Project investigation (2026-05-29): pool-driven LP-cost root cause of the
  more-forwards/fewer-iterations slowdown; the forwards-vs-workers
  underutilization cliff.
