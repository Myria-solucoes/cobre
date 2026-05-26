# HiGHS LP Solver Tuning for the SDDP Backward Pass

**Status**: Investigation report (in progress). Documents the optimization
journey from the active-only bake landing through to the current
`BACKWARD_PROFILE` state, the variance hypothesis it surfaced, and the
mitigation strategies that are still on the table.

**Companion**: `docs/design/backward-pass-performance-analysis.md` provides
the upstream architectural map (timing sample taxonomy, SPTcpp comparison,
wall-clock attribution). This document focuses specifically on the LP
solver-level tuning surface — what we changed inside `HighsProfile` and why.

## 1. The particular challenge: cobre's backward solve loop

Understanding the variance starts with the precise shape of the backward
solve loop. It is not a series of independent LPs.

### 1.1 Per-stage structure

At each backward stage `t`, for each trial point `x_t` collected during the
forward pass:

1. **Load** the stage-`t+1` baked template
   (`load_backward_lp`, `backward.rs:264-272`). This is a full
   `load_model` FFI call per trial point — non-negotiable for
   reproducibility.
2. **Install** the warm-start basis captured from the forward pass at
   stage `t+1`. The basis encodes the optimal vertex for the realized
   uncertainty `ω_{t+1}^*` that the forward sampler drew.
3. **For each opening `k = 1 .. N`** (the backward sample of uncertainty
   at stage `t+1`):
   - Patch row bounds with opening `k`'s inflow / load noise.
   - Patch column bounds with opening `k`'s NCS (non-controllable source)
     realisation.
   - Patch column bounds with the trial-point state `x_t` on the
     incoming-state columns.
   - **Solve**.
   - Extract `reduced_costs[col] * col_scale[col]` as the cut subgradient.
   - **Do not reset the basis.** The next opening reuses whatever basis
     the simplex left in HiGHS after this solve.
4. Aggregate the `N` cut contributions into a single Benders cut and
   insert into the FCF (future cost function).

The cut subgradient extraction reads `view.reduced_costs[col]` (per
`cobre-sddp/src/lp_builder/mod.rs:85`). Cut deactivation is local-only
(by RHS toggle), not row removal.

### 1.2 The warm-start chain

The basis lifecycle within one trial point at one stage is:

```
forward basis @ stage t+1, uncertainty ω*
        │
        ▼  (install once)
        ─ patch(x_t, ω_1)  → solve  → reduced_costs  ─┐
                                                      │ basis preserved
        ─ patch(x_t, ω_2)  → solve  → reduced_costs  ─┤
                                                      │
        ─ patch(x_t, ω_3)  → solve  → reduced_costs  ─┤
        ...                                           │
        ─ patch(x_t, ω_N)  → solve  → reduced_costs  ─┘
```

Each `solve` does its work starting from wherever simplex finished on
the previous opening. This is the canonical setting for **sequential
parametric warm-started simplex**.

What changes between consecutive openings:

- A subset of row bounds (typically `n_hydros + n_loads + n_ncs`).
- A subset of column bounds (NCS columns when applicable).

What stays fixed:

- The matrix `A` (cuts and structural rows are constant within a stage).
- The objective `c`.
- The number of rows and columns.

So the parametric perturbation between openings is bound-only, never
structural. In LP-perturbation terms, this is a **right-hand-side
parametric** problem with periodic column-bound updates.

### 1.3 Why this matters for variance

Three properties of this loop concentrate the variance:

**(a) Opening 1 is the closest warm-start, opening N is the farthest.**
The captured basis is for `(x_{t+1}^*, ω_{t+1}^*)`. Opening 1's LP differs
only in the uncertainty realisation. By opening `k`, the basis has been
re-pivoted `(k-1)` times to absorb prior openings' perturbations; it is
no longer "the forward basis" but rather "whatever simplex converged to
for opening `k-1`". The warm-start advantage isn't uniform — it's
strongest on opening 1, weakest on opening `N` in expectation, but the
actual difficulty depends on how close opening `k-1`'s vertex is to
opening `k`'s.

**(b) The basis can drift through degenerate vertices.**
The cut rows are sparse but numerous (16k+ active cuts by iteration 4
on the production case). Many cuts can be simultaneously tight at the
optimal vertex, so the dual simplex finds itself on highly-degenerate
faces. With cost perturbation disabled
(`dual_simplex_cost_perturbation_multiplier = 0`) — the right choice
for the bulk of warm-started solves — the simplex can stall on these
faces and need a long sequence of zero-cost-improvement pivots to
escape.

**(c) The LU factorisation accumulates drift over the opening chain.**
Each pivot updates the LU factorisation in-place; periodic
re-factorisations bound the numerical error but cost ~O(m²) each. The
trigger is `rebuild_refactor_solution_error_tolerance` (set to `1e-6`,
loosened from the HiGHS default `1e-8`). A long warm-start chain that
goes many pivots between refactorisations carries more accumulated
LU error into each opening, which can in turn produce more correction
pivots — a slow positive feedback.

These three effects compound. The wall-time variance is broad rather
than long-tail-spiky: `p99/p50 ≈ 3.6×` on the production case, with
the top 10% of solves accounting for only 23% of total time. There is
no single outlier kind to target — the variance is structural to the
loop.

## 2. Production variance signature

From `plans/benchmark-outputs/run_1` (192 workers, 967 680 backward
solves across 4 SDDP iterations on the production case, with HiGHS
defaults):

| Statistic       | Backward solve_time_ms |
| --------------- | ---------------------: |
| mean            |                72.6 ms |
| std             |                42.5 ms |
| p50             |                62.2 ms |
| p90             |               127.3 ms |
| p99             |               222.6 ms |
| p99.9           |               325.2 ms |
| max             |               764.3 ms |
| ratio p99 / p50 |                   3.6× |

| Statistic | simplex_iterations |
| --------- | -----------------: |
| mean      |                255 |
| p99       |                924 |
| max       |              2 222 |

Average per-pivot cost: **284 µs/pivot** (mean wall ÷ mean iterations).

Backward dominates the iteration: per-iter max-worker backward wall is
60–135 s vs forward 2–7 s. Cut sync and lower-bound evaluation are <10 s
combined. Backward load imbalance (slack between fastest and slowest
worker per stage) is 21–32 s per iteration — itself a function of the
solve-time variance we're trying to characterise.

The retry ladder fires on **0.0155% of solves** (150 out of 967 680).
There is no headroom to bound the bulk by tightening the iteration cap —
the cap of `max(100k, 50·num_cols)` is orders of magnitude above the
observed `p99` of 924 iterations.

## 3. Hypothesised causes of variance

Working hypothesis tree, ranked by likely contribution:

### 3.1 Stage-position effects (high confidence)

Late stages have larger LPs. As SDDP iterations progress, the active
cut count grows. By iteration 4 on the production case, stages 50–62
mean 100–130 ms per solve vs early stages around 45 ms. The variance
within a single stage's solves at iter 4 is `std ≈ 50–80 ms`.

Variance source: LP size grows monotonically with stage index (because
cuts accumulate at each stage) and with SDDP iteration. Per-pivot cost
scales linearly with row count for row-pricing strategies.

### 3.2 Cut degeneracy at the optimum (high confidence)

Multiple cuts active simultaneously at the optimal vertex creates a
degenerate LP. Dual simplex on degenerate faces:

- Either cycles (handled by anti-cycling rules but at iteration cost), or
- Takes many zero-improvement pivots searching the face.

Cost perturbation (`dual_simplex_cost_perturbation_multiplier`) breaks
degeneracy by adding small random shifts to the objective. We have it
disabled because it costs ~5–10% on non-degenerate solves through
cleanup pivots after the perturbed solve converges. The retry ladder
re-enables it on retry level 0. With retries at 0.0155%, the
perturbation safety net catches genuine cycling but doesn't help the
bulk-degenerate solves that finish "eventually" without escalating.

Variance source: openings whose optimum vertex is on a wide degenerate
face take many more pivots than openings whose optimum is non-degenerate.

### 3.3 Warm-start chain quality (medium confidence)

The basis from opening `k-1` may be a good or poor starting point for
opening `k` depending on the uncertainty distance. Extreme-tail inflow
or load realisations can produce LPs whose optimum is far from the
previous opening's. The pivot work to bridge that distance is
opening-pair-specific.

Variance source: opening ordering is fixed at scenario generation;
some adjacent opening pairs are inherently close in optimum, others
far. The variance is structural to the opening sequence.

### 3.4 LU factorisation drift (medium confidence)

Long warm-start chains can carry significant LU update error before
the next refactor. The currently-loosened tolerance (`1e-6` vs HiGHS's
`1e-8` default) is tuned for the bulk-easy case. Hard chains may
benefit from a tighter tolerance forcing earlier refactor.

Variance source: refactor schedule is error-driven; openings that
trigger more pivots also produce more numerical drift, but the next
opening only sees the consequence when its solve hits the slower
near-singular factorisation.

### 3.5 Pricing-strategy variance (now confirmed marginal, see §4)

We hypothesised that pricing strategy choice could materially shift
mean and tail. The empirical sweep below shows the effect is real but
small — Devex / Dantzig / SteepestEdge all land within ±5% on mean
wall.

## 4. The optimization journey

What follows is a chronological account of what was tried, in what
order, and what the measurements showed. The full benchmark artifacts
are in `plans/benchmark-outputs/run_*/`.

### 4.1 Phase 0 — baseline (run_1)

Branch state: `cb2fd661` + Run 1 surgical reverts (active-only bake
restored, all other HiGHS options at defaults).

`HighsProfile` did not exist yet; HiGHS configuration was the
`default_options()` table only.

Headline: 437.1 s total wall, 406.8 s backward wall, 72.6 ms mean
backward solve, 284 µs/pivot.

### 4.2 Phase 1 — restore the per-phase profile mechanism

Tickets 004 + 005 of `plans/lp-bake-cleanup-and-backward-tuning/`.

- Restored `SolveProfile` (from reverted commit `818001b8`) and
  `ProfiledSolver<S>` wrapper. Pivoted from a single global
  `SolveProfile` to an associated-type-per-solver design
  (`SolverInterface::Profile` → `HighsProfile`) to keep the solver-side
  surface honest about which knobs are HiGHS-specific while still
  allowing future CLP / Gurobi / etc. profiles to coexist.
- Added a re-apply-before-solve mechanism in `ProfiledSolver::solve`
  to defeat HiGHS's internal option resets between solves (a
  production bug observed where profile values reverted after the
  first solve).
- Wired `set_profile(Phase::Backward.profile())` per worker before
  each backward parallel region.

This phase did not change solver behaviour by itself
(`FORWARD_PROFILE = BACKWARD_PROFILE = SIMULATION_PROFILE =
HighsProfile::default()` initially). It established the surface to
experiment on.

### 4.3 Phase 2 — initial `BACKWARD_PROFILE` overrides (run_5_post_revert)

Set `BACKWARD_PROFILE`:

- `simplex_dual_edge_weight_strategy = 0` (Dantzig)
- `simplex_price_strategy = 2` (RowHyperSparse)

The choice was driven by `plans/benchmark-outputs/analysis-report.md`
§7 Tier 2, which suggested these two options for backward.

A third candidate from the analysis report — `simplex_scale_strategy =
0` (off) — was initially applied based on the docstring claim that
"the cobre prescaler already normalizes matrix entries". On
investigating before measuring, the prescaler at
`crates/cobre-sddp/src/lp_builder/scaling.rs` was found to be guarded
by `#![allow(dead_code)]` and bypassed by
`setup/template_postprocess.rs`. **The cobre prescaler is not in the
production data path.** HiGHS's internal equilibration is the only
scaling layer running.

`simplex_scale_strategy` was reverted to `4` (Equilibration) for all
three profiles. This is the only honest baseline; the docstring was
updated to match reality.

The measurement (run_5_post_revert, current branch state up to this
point):

| Metric           | run_1 (default) | run_5 (Dantzig+HS+Eq) |         Δ |
| ---------------- | --------------: | --------------------: | --------: |
| mean ms          |            72.6 |                  70.8 |     −2.5% |
| p99 ms           |           222.6 |                 207.2 |     −6.9% |
| p99.9 ms         |           325.2 |                 337.2 | **+3.7%** |
| max ms           |             764 |              **1230** |  **+61%** |
| mean iters       |             255 |               **317** |  **+24%** |
| max iters        |           2 222 |             **4 243** |  **+91%** |
| µs/pivot         |             284 |               **223** |  **−22%** |
| total backward s |          70 232 |                68 507 |     −2.5% |
| total wall s     |           437.1 |                 433.8 |     −0.7% |

Reading the result:

- Dantzig made the **per-pivot cost** 22% cheaper. The price-strategy
  - edge-weight combination genuinely reduces the work each pivot does.
- But Dantzig pricing needs **24% more pivots** to converge — Dantzig
  is theoretically less iteration-efficient than Devex on degenerate
  LPs. The two effects nearly cancelled.
- The **tail got materially worse** — max solve jumped from 764 to
  1230 ms (+61%), max iterations nearly doubled. Dantzig's
  iteration-inefficiency hit hardest on the hard LPs.
- Net backward wall: **−1.8%**. Net total wall: **−0.7%**.

The hypothesised "Tier 2: 5–15% backward win" did not materialize. The
report's headline number assumed three contributors — pricing,
price-strategy, and scaling-off — and the scaling-off third was based
on a false premise.

### 4.4 Phase 3 — SteepestEdge (run_6)

Hypothesis: SteepestEdge picks better pivot directions on degenerate
LPs than either Devex or Dantzig — typically fewer iterations than
either, at moderate per-pivot cost. Expected: iteration count drops
back near Devex levels, per-pivot cost rises modestly, tail bounded.

Set `BACKWARD_PROFILE.simplex_dual_edge_weight_strategy = 2`. Single
field change.

Measurement (run_6):

| Metric           | run_1 Devex+Row | run_5 Dantzig+HS | run_6 SteepestEdge+HS |
| ---------------- | --------------: | ---------------: | --------------------: |
| mean ms          |            72.6 |             70.8 |              **76.2** |
| p50 ms           |            62.2 |             62.2 |                  62.4 |
| p99 ms           |           222.6 |            207.2 |             **269.3** |
| p99.9 ms         |           325.2 |            337.2 |             **405.9** |
| max ms           |             764 |            1 230 |                   823 |
| std ms           |            42.5 |             40.6 |              **51.1** |
| mean iters       |             255 |              317 |               **248** |
| max iters        |           2 222 |            4 243 |                 2 511 |
| µs/pivot         |             284 |              223 |               **308** |
| total backward s |          70 232 |           68 507 |            **73 723** |
| total wall s     |           437.1 |            433.8 |             **463.6** |

Reading the result:

- The iteration hypothesis was **confirmed**: SteepestEdge produces the
  lowest mean iterations of all three (248), max iterations dropped
  from Dantzig's 4 243 to 2 511.
- But the **per-pivot cost is the highest** of the three at 308 µs/pivot
  — 38% more than Dantzig, 8% more than Devex. SteepestEdge's
  expensive edge-weight maintenance ate the iteration savings.
- The **tail also regressed on percentile measures**: p99 +30% over
  Dantzig, p99.9 +20% over Dantzig. Only the absolute max improved
  (because Dantzig's max was specifically a runaway-iteration LP that
  SteepestEdge converged faster). std grew to 51 ms.
- Net backward wall: **+7.6%**. Total wall: **+6.9%**.

SteepestEdge is **strictly dominated** by both alternatives on our
workload. The expected mid-point on the pricing/cost trade-off curve
is not on this objective surface.

### 4.5 Phase 4 — isolate the price-strategy contribution (run_7, planned)

Phases 2 and 3 conflated two changes (edge-weight + price-strategy).
We do not know how much of run_5's modest win came from RowHyperSparse
vs how much from Dantzig.

Set `BACKWARD_PROFILE.simplex_dual_edge_weight_strategy = 1` (Devex,
matching `HighsProfile::default`). Keep only the price-strategy
override.

This is the experiment whose result is pending at the time of writing.
The result will close the edge-weight axis and either justify keeping
the RowHyperSparse override or motivate dropping it.

## 5. Summary table

All four runs, same case, same workers, same SDDP configuration. Only
`BACKWARD_PROFILE` changes.

| Run   | Profile (vs default)                | mean ms | p99 ms | max ms | µs/pivot | mean iters | total wall s | Convergence (cuts_active iter 4) |
| ----- | ----------------------------------- | ------: | -----: | -----: | -------: | ---------: | -----------: | -------------------------------: |
| run_1 | (none — defaults)                   |    72.6 |  222.6 |    764 |      284 |        255 |        437.1 |                           16 453 |
| run_5 | Dantzig + RowHyperSparse            |    70.8 |  207.2 |  1 230 |      223 |        317 |        433.8 |                           16 607 |
| run_6 | SteepestEdge + RowHyperSparse       |    76.2 |  269.3 |    823 |      308 |        248 |        463.6 |                           17 623 |
| run_7 | RowHyperSparse only (Devex default) |     TBD |    TBD |    TBD |      TBD |        TBD |          TBD |                              TBD |

Cut counts at iteration 4 vary by <8% across runs, but the
lower-bound trajectories match within 0.05%. The algorithmic
convergence is essentially identical; the solver tuning is moving
LP-vertex choice on degenerate faces without changing the policy.

## 6. Conclusions on HighsProfile-level tuning

Three datapoints across the edge-weight axis tell a clear story.

1. **The available win is small.** The best edge-weight choice
   (Dantzig, run_5) buys 0.7% total wall and 2.5% backward wall over
   defaults — and even that win is paired with a tail regression
   (+61% max solve time). The expected mid-point (SteepestEdge) is
   strictly worse.

2. **The Tier 2 headline was scaling-dependent.** The analysis
   report's 5–15% backward win estimate combined pricing changes with
   scaling-off. Scaling-off was based on the false-premise dead
   prescaler. With scaling honestly at Equilibration the available
   pricing win is in the 0–5% range.

3. **The variance is not in HighsProfile's reach.** The broad
   `p99/p50 ≈ 3.6` distribution is shaped by stage-position effects,
   cut degeneracy, warm-start chain quality, and LU drift — none of
   which the edge-weight or price-strategy knobs significantly affect.
   The 50th percentile is 62 ms in all three runs (Devex, Dantzig,
   SteepestEdge). The bulk LP solve time is set by per-pivot row work
   times iteration count, and those are bounded by LP size which is
   bounded by cut count.

## 7. Mitigation strategies that remain on the table

Grouped by where the change lives.

### 7.1 Algorithmic (outside HighsProfile)

**A1. Cut aging / pool compaction.** Hard-delete cut slots that have
been inactive for `K` SDDP iterations. Bounds LP size below the
"every cut ever generated stays forever" floor. Architectural change:
slot identifiers must handle compaction or holes; basis reconstruction
must be aware. Estimated win from the analysis report: meaningful
(Tier 3). Effort: high.

**A2. Drop backward warm-start basis.** The forward warm-start gains
~2× per solve; the backward warm-start buys only ~10% per the
benchmark sweep (run_3 → run_4). Without backward warm-start, presolve
runs (HiGHS skips presolve when a basis is supplied), which removes
free rows and shrinks the active LP. The basis-reconstruction
machinery (`basis_reconstruct.rs`, `enforce_basic_count_invariant`,
the basis broadcast cache, `CapturedBasis` allocation pool) can be
deleted. Cost: +7% backward wall. Benefit: ~1 000 LOC removed,
tail-latency stabilization. Effort: medium.

**A3. Opening ordering.** Process backward openings in an order
designed to keep adjacent openings close in optimum-vertex space.
E.g. sort by aggregate inflow level, or cluster openings before the
solve loop. The basis chain stays warmer; tail variance from
opening-pair distance shrinks. Effort: medium; needs ordering
metric that doesn't itself dominate the solve cost.

**A4. Periodic basis refresh.** Every `K` openings, restore the
captured forward basis and discard accumulated pivot state. Bounds
LU drift. Trade-off: lose accumulated pivot progress when the chain
is benign. Effort: low; one new knob.

**A5. IPM for hard stages.** Switch from simplex to IPM at stage
indices where active cut count exceeds a threshold. IPM is not
warm-start-sensitive so the LU drift mechanism doesn't apply. Need
reproducibility validation across solver paths and a clear switch
criterion. Effort: medium-high.

### 7.2 Solver tuning (additional HighsProfile fields)

**S1. Cost perturbation.** Currently
`dual_simplex_cost_perturbation_multiplier = 0`. Re-enabling at a
small value (e.g. `0.1`) trades a small bulk cost (cleanup pivots) for
breaking out of degenerate stalls on the variance shoulder. Could
add the field to `HighsProfile` and override for backward only. Risk:
non-trivial because the basis we extract reduced costs from now
reflects the perturbed LP. Effort: low.

**S2. Refactorization frequency.** Tighten
`rebuild_refactor_solution_error_tolerance` from `1e-6` back toward
`1e-7` or `1e-8` for backward only. More frequent refactor bounds LU
drift per opening at the cost of per-stage refactor overhead. Effort:
low; one new HighsProfile field.

**S3. Initial condition check.** Currently
`simplex_initial_condition_check = false`. Enabling for backward
(with `simplex_initial_condition_tolerance` set permissively) would
detect when the warm-start basis is too far from optimal and trigger
a cold-start cheaper than a long iteration chain. Could materially
shorten tail solves. Effort: low; two new HighsProfile fields.

**S4. Dual SIP (`simplex_strategy = 2`).** Different pivot
co-ordination scheme inside dual simplex than Dual Plain. Designed
for cases where degeneracy is the dominant cost. Single-threaded
variant of PAMI. Effort: low; one new HighsProfile field.

### 7.3 Hybrid

**H1. Adaptive iteration cap with retry escalation.** Track a moving
average of recent backward solve times; if a solve exceeds 2–3× the
median, bail to retry level 0 (clear solver, perturbation on). Uses
the existing retry ladder more aggressively than its current 0.0155%
trigger rate. Provides empirical tail bounding without a static cap.
Effort: medium; requires adaptive state across solves.

## 8. What we are not pursuing and why

- **`simplex_unscaled_solution_strategy = 0`** (skip post-solve
  refinement). Originally proposed when we believed scaling was off
  end-to-end. With HiGHS equilibration active, the refinement step
  re-solves the unscaled LP using the scaled basis as a warm-start.
  Reduced costs we read from `view.reduced_costs` come from the
  refined (unscaled) solution. Skipping refinement would give us
  scaled reduced costs that need manual de-scaling — and we have not
  audited the cut-subgradient extraction path for that.

- **Loosen primal/dual feasibility tolerance.** The retry ladder is
  built on `applied = max(level_default, profile_value)`, so a looser
  profile silently survives across retries — there is no path back to
  strict tolerances within a solve sequence. The cut subgradient is
  tolerance-sensitive at the SDDP convergence level.

- **Iteration cap as a tuning lever.** Run_1's max iter is 2 222 and
  p99 is 924 — no realistic cap above ~1 000 helps. Setting it below
  1 000 would catch genuine work and force costly retry escalations
  that aren't necessary for those LPs.

- **HiGHS PAMI / parallel-within-solve.** The training loop already
  parallelises 192 LPs across rayon workers; per-LP threading would
  oversubscribe and regress.

## 9. Open questions

- **Cut-sync wall growth.** Cut-sync wall grew monotonically across
  runs 1 → 5 → 6 (5.8 → 10.8 → 16.5 s). Cut payload is identical;
  this should be a constant. Worth a controlled re-run to see if it
  reproduces or was MPI variance. If it persists, investigate.

- **Are stages 50–62 systematically harder, or coincidentally
  hard?** The mean solve time at late stages is ~1.5× early stages.
  Is this purely cut count, or is the LP structure (e.g. end-of-horizon
  storage value coefficients) materially different? Profile-per-stage
  might be warranted if the answer is structural.

- **Would a measured "hard-opening predictor" help?** Some openings
  are reliably slow. If a cheap predictor exists (e.g. inflow
  z-score on tail), we could pre-trigger cost perturbation for those
  openings rather than waiting for the retry ladder.

## 10. Next planned step

Run 7: `BACKWARD_PROFILE.simplex_price_strategy = 2`, all other fields
match `HighsProfile::default`. Isolates the price-strategy contribution
from the edge-weight contribution. The result will:

- Confirm whether RowHyperSparse alone justifies its 1-field override
  (i.e. delivers a measurable mean / p99 improvement over Devex+Row),
- or motivate dropping `BACKWARD_PROFILE` overrides entirely and
  moving to the algorithmic mitigations in §7.1.

After run 7 lands, the natural next decision point is whether to
invest in §7.1 (architectural changes, larger expected win) or §7.2
(more profile fields, marginal expected win). The data so far
strongly suggests §7.1 — variance reduction needs an architectural
attack, not a tuning attack.

## References

- `plans/benchmark-outputs/analysis-report.md` — 4-run benchmark
  sweep on the Phase 2 regression. Tier 2 / Tier 3 estimates.
- `plans/benchmark-outputs/run_{1,5_post_revert,6}/` — raw artifacts
  for the runs analysed here.
- `docs/design/backward-pass-performance-analysis.md` — timing
  taxonomy, wall-clock attribution, SPTcpp comparison.
- `crates/cobre-solver/src/highs.rs` — `default_options()` table,
  `HighsProfile`, retry ladder.
- `crates/cobre-sddp/src/solver_phase.rs` — `BACKWARD_PROFILE`,
  drift guards.
- `crates/cobre-sddp/src/lp_builder/scaling.rs` — dead prescaler
  helpers (kept for re-wire option).
