---
title: "Solver tuning — production results and default recommendations"
date: 2026-06-05
status: results
companion: ["accelerator-effectiveness-research.md"]
---

# Solver tuning — production results

Empirical results of the staged solver/accelerator tuning campaign run on a
production-scale case, and the resulting default recommendations. The campaign
used a benchmark-only `COBRE_TUNE_*` override seam and a SLURM harness
(`scripts/solver-tuning/`) that were **removed once the campaign concluded** (this
document is the durable record; see git history for the procedure and harness).
`accelerator-effectiveness-research.md` retains the accelerator hypotheses.

## Setup

|              |                                                                                                 |
| ------------ | ----------------------------------------------------------------------------------------------- |
| Case         | 64 stages · 155 hydros · 112 thermals · 5 buses                                                 |
| Regime       | 96 forward passes · 10 iterations · single node · 96 threads · no MPI exchange (`world_size=1`) |
| Risk measure | CVaR on every stage, `α=0.15`, `λ=0.4` (risk-averse)                                            |
| Backends     | HiGHS and CLP, tuned **independently** on the **identical** case                                |
| Repeats      | 1 per cell (see Caveats)                                                                        |

Two stages per backend: **Stage 1** sweeps solver parameters (one-factor-at-a-time
on top of a correctness floor); **Stage 2** sweeps the accelerator matrix
(warm-start `{full,core,off}` × cut-selection `{none,level1,dominated}`) on top of
the Stage-1 winning solver environment.

Timings below are **total wall-clock** (`duration_seconds`). The backward pass
dominates (≈93% of wall), so wall and backward-solve rankings agree.

> **Correctness under CVaR.** With a risk-averse objective the lower bound
> converges to the risk-adjusted optimum while the forward-pass "upper bound"
> estimates the mean cost, so `LB > UB` is **expected** and is not a failure. Cut
> validity is therefore checked via **LB monotonicity** (the lower bound is
> non-decreasing as cuts accumulate; a decrease beyond `1e-4` relative signals an
> invalid cut). Every retained cell passed (max LB decrease ≈ 0). The two backends'
> floor lower bounds agree to **1.7%** (alternate optima at a non-converged cap).

## Results

### HiGHS — Stage 1 (solver parameters)

Reference = `floor` (presolve off). All levers are within single-rep noise of the
floor **except scaling, which is strongly harmful**.

| cell          | wall   | Δ%        | note                                  |
| ------------- | ------ | --------- | ------------------------------------- |
| presolve-on   | 1842 s | **−2.7%** | presolve ON (cobre's shipped default) |
| edge-steepest | 1876 s | −0.9%     |                                       |
| edge-choose   | 1892 s | −0.1%     |                                       |
| floor         | 1893 s | 0.0%      | presolve off                          |
| scale-equil   | 2598 s | +37.3%    | HiGHS scaling — harmful               |

The `price-rsc` (price strategy 3) fragility probes from the local shakedown were
re-run here: plain `price-rsc` and `price-rsc.scale` were neutral-to-harmful
(+0.3% / +52%); none beat the floor. **Conclusion: presolve `on` (already the
shipped default) is marginally best at scale, warm-start stays healthy
(basis-reject 0%), and no other HiGHS solver lever helps — keep the current
defaults; do not enable HiGHS scaling.**

### HiGHS — Stage 2 (accelerators, on presolve=on)

Reference = `ws-full.sel-none`.

| cell                    | wall   | Δ%         |
| ----------------------- | ------ | ---------- |
| ws-full · sel-dominated | 1513 s | **−19.2%** |
| ws-full · sel-level1    | 1548 s | −17.3%     |
| ws-core · sel-dominated | 1554 s | −17.0%     |
| ws-off · sel-level1     | 1666 s | −11.0%     |
| ws-full · sel-none      | 1872 s | 0.0%       |
| ws-off · sel-none       | 2113 s | +12.9%     |

### CLP — Stage 1 (solver parameters)

Reference = `floor`. `pricing_mode=0` (uninitialized DSE weights) **failed**
(errored at iteration 0, exit 4) — unsafe at production scale.

| cell           | wall   | Δ%        | note                         |
| -------------- | ------ | --------- | ---------------------------- |
| scaling-equil  | 2167 s | **−8.4%** | CLP scaling=1                |
| floor          | 2365 s | 0.0%      |                              |
| pricing-full   | 2430 s | +2.7%     |                              |
| factor-100     | 2504 s | +5.9%     | factorization freq — harmful |
| pricing-uninit | —      | FAIL      | `pricing_mode=0` errored     |

### CLP — Stage 2 (accelerators, on scaling=1)

Reference = `ws-full.sel-none`.

| cell                    | wall   | Δ%         |
| ----------------------- | ------ | ---------- |
| ws-full · sel-level1    | 1695 s | **−23.2%** |
| ws-full · sel-dominated | 1714 s | −22.3%     |
| ws-core · sel-level1    | 1907 s | −13.6%     |
| ws-off · sel-dominated  | 2061 s | −6.6%      |
| ws-full · sel-none      | 2208 s | 0.0%       |
| ws-off · sel-none       | 2837 s | +28.5%     |

## Cross-backend comparison (best config per backend)

| config                                 | wall       | bwd (worker-s) | retried | failures | final LB  | active cuts |
| -------------------------------------- | ---------- | -------------- | ------- | -------- | --------- | ----------- |
| HiGHS floor (ws-full.sel-none)         | 1872 s     | 101 307        | 1 991   | 0        | 1.7096e12 | 60 480      |
| **HiGHS best** (ws-full.sel-dominated) | **1513 s** | 81 093         | 1 799   | 0        | 1.7086e12 | 22 028      |
| CLP floor (ws-full.sel-none)           | 2208 s     | 116 095        | 40      | 0        | 1.6804e12 | 60 480      |
| **CLP best** (ws-full.sel-level1)      | **1695 s** | 87 350         | 34      | 0        | 1.6817e12 | 22 097      |

- **Speed:** HiGHS is faster at both floor (+15.2% vs CLP) and best config (+10.8%).
- **Robustness:** HiGHS retries ~1800–2100 solves (≈0.15% of 1.27M), zero failures,
  no special handling. CLP on the _warm_ path is clean (~35 retries) but on **cold**
  solves escalates heavily (~7600 retries via the cold-solve escalation ladder), and
  one Stage-1 config failed outright. CLP also required a backend fix (cold-solve
  escalation) to run this case at all.
- **Cut-selection is safe and large:** it cut the active pool 60 480 → ~22 000 while
  moving the final LB by **<0.1%** — i.e. it removed redundant cuts, preserving the
  policy, for a ~20–25% backward-pass win on both backends.

## Recommendations

The campaign confirmed that cobre's shipped defaults are already at (or near) the
optimum, so **no compiled default was changed**:

1. **Default backend: HiGHS** — faster and more robust, equal correctness. Already
   cobre's default feature; CLP remains the opt-in alternative. _(No change.)_
2. **HiGHS solver defaults: no change** — `presolve=on` is best at scale; do not
   enable HiGHS scaling; edge/price levers are within noise.
3. **CLP solver defaults: no change** — `scaling=1` looked marginally faster (−8.4%,
   single-rep) but is not worth flipping a default for the non-default backend;
   **never use `pricing_mode=0`** (fails at scale).
4. **Warm-start: `full`** — already the default and clearly best (cold solves are
   worst: HiGHS +13%, CLP +28%). The evaluation-only `core`/`off` modes were removed.

The one **opt-in** worth adopting per case:

5. **Enable cut-selection** (`training.cut_selection`, method `dominated` or `level1`
   — near-tie; `check_frequency=5`) — the headline ~20–25% backward-pass win on both
   backends, LB-invariant. It remains **off by default** (deliberate, and pending the
   out-of-sample certification below); enable it explicitly in a production
   `config.json` for the speedup.

## Caveats

- **Single rep.** The large effects are solid (backend gap; cut-selection ≈ −20–25%;
  warm-`off` worst; HiGHS scaling and CLP `pricing_mode=0` harmful). The fine
  orderings (HiGHS S1 within ±3%, CLP `scaling=1` −8.4%, `level1` vs `dominated`) are
  soft; confirm with `--reps 3` on each backend's floor + best before treating them as
  exact.
- **No simulation.** Policy quality is inferred from the LB-invariance under
  cut-selection (<0.1%), not measured out-of-sample. Recommend one `simulate`
  comparing cut-selection vs none on the winner (matching realized mean/CVaR) before
  enabling cut-selection in production by default.
