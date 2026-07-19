# LPT claim ordering in the opening-block backward scheduler — local A/B

> **Status**: Implemented (always-on under the `opening_block` backward
> scheduler). This note records a local, single-box A/B measuring the
> realized value of the claim ordering and states the keep/drop recommendation.
> The binding keep/drop decision is deferred to a cluster-scale confirmation
> arm; this local result is one input to it, not the verdict.
>
> **Companion:** `docs/design/backward-opening-ordering.md` — a separate
> warm-start lever on the same backward pass (it reorders the _openings solved_
> within a trial point; this reorders the _claim sequence_ across worker
> threads). Its "Measured results — A/B" section is the house-style precedent
> for the hedged, matched-epoch register used below.

## What LPT is, and why it is result-neutral

Under the opening-block scheduler, the backward pass distributes
`(trial point, opening-block)` work units to worker threads from a shared atomic
claim counter. LPT ("longest processing time first") orders the claim sequence
hardest-`(stage, block)`-first, keyed by the **previous** iteration's per-
`(stage, block)` mean `simplex_iterations`. Packing the longest blocks first
keeps a worker from being left finishing a long block after its peers have
drained the queue — the classic makespan argument for greedy longest-first
scheduling.

The ordering key is per-`(stage, block)`, not per-`(trial point,
block)`: resampled trial points make per-trial-point hardness resampling noise,
while the opening-block component is iteration-stable, so a previous-iteration
pivot count is a usable predictor only at block granularity.

LPT is **result-neutral**. Every claim unit is hermetic — LPT changes only
_which worker_ runs a unit and _when_, never _what_ the unit computes. Each
opening's outcome is written into a per-`(m, ω)` arena and aggregated per trial
point over canonical `ω` in ascending `m`, unchanged by the claim order. The
generated cut set and `final_lb` are therefore bit-identical to canonical-order
opening-block scheduling. This is pinned by `lpt_claim_order_is_result_neutral`
(LPT-on vs LPT-off, bitwise `final_lb`) in
`crates/cobre-sddp/tests/mpi_wire.rs`, alongside the scheduler's determinism
gates `pn_scheduler_determinism_expectation` and `pn_scheduler_determinism_cvar`
(bitwise `final_lb` across thread and rank shapes). The claim-order contract is
also recorded in `.claude/rules/sddp.md`.

Because cuts are unchanged, LPT's only realizable payoff is **makespan / load
imbalance** at moderate worker subscription — not pivots, which it leaves
bit-identical. An offline schedule simulation with exact per-unit costs
projected a mid-teens-percent makespan reduction; the shipped variant keys only
on the iteration-stable per-`(stage, block)` component, so it is expected to
realize a fraction of that.

## Protocol

**Arms (two builds, not a toggle).** LPT is not exposed as a config, CLI, or
environment switch, so the two arms are two binaries:

- **LPT-on** — this repository revision (the always-on default under
  `opening_block`).
- **LPT-off** — a build of the immediate parent revision, identical except the
  LPT claim-order kernel. It runs the same opening-block scheduler in canonical
  (identity) claim order.

Both arms set `training.backward_scheduler = "opening_block"`.

**Cases.**

- **`cobre_rodada`** (owner-local; the load-bearing case) —
  `opening_block_size = 10`, 32 forward passes, 3 iterations, simulation off,
  `--threads 14` of 20 logical cores. Fourteen threads is the moderate-
  subscription regime where per-stage claim-loop imbalance registers; this is
  the only case here whose backward pass is large enough to accumulate a
  measurable imbalance. Not distributed with the repository.
- **`examples/1dtoy`, `examples/4ree`** (in-repo) — `opening_block` forced,
  small (single-reservoir toy; four-reservoir multi-opening stochastic). Their
  backward pass is sub-millisecond, so they carry **no** makespan signal by
  construction; they serve only as byte-neutrality / smoke cross-checks on cases
  any reader can reproduce.

**Metrics.**

- **Primary — `bwd_load_imbalance_ms`** (`training/timing/iterations.parquet`,
  per iteration): the per-stage backward load-imbalance estimate accumulated
  over all stages — the quantity LPT directly attacks. Corroborated by the
  **matched-epoch backward wall** (the `bwd:` field of the per-iteration
  progress line, summed over iterations).
- **Invariance cross-check — total backward `simplex_iterations`**
  (`training/solver/iterations.parquet`, `phase == backward`): expected
  bit-identical under LPT. A non-trivial delta here would mean the arms were
  misconfigured (different schedulers, cases, or iteration counts), not an LPT
  effect.
- **Byte-neutrality cross-check — final lower bound**
  (`training/convergence.parquet`): expected bitwise identical between arms.

The comparison is automated by the local investigation script
`plans/backward-perf-investigation/ab_compare.py` (progress-log wall + backward
pivots + lower bound), plus a direct read of the two timing columns above. That
script is an operator-local harness, not a committed or CI-run benchmark; the
committed, always-reproducible references are the parquet columns and the gates
named above.

**Matched-epoch rule and noise caveat.** Desktop wall time swings run-to-run at
bit-identical work. On this box the _same_ LPT-on build measured a total backward
wall of 526,314 ms and 444,063 ms on two separate runs — a ~16% epoch swing for
identical work. Only within-pair deltas (arms run back-to-back, alternating on /
off, on an otherwise-idle box) are quotable; cross-epoch wall comparisons are
not. The imbalance estimate, a within-run accumulation, is the cleaner primary
metric.

## Measured results — local A/B

Measured 2026-07-19 on a single development box (i7-12700KF, 20 logical cores),
HiGHS backend, single run per arm, back-to-back matched epochs. Every number
below is single-run and box-specific; none is a standing benchmark. Δ is
`(on − off) / off`; negative means LPT-on is lower (the improvement direction).

**`cobre_rodada` — primary signal (totals over the 3 iterations, per matched
pair):**

| matched pair | `bwd_load_imbalance_ms` on | off    | Δ     | backward wall on | off        | Δ     |
| ------------ | -------------------------- | ------ | ----- | ---------------- | ---------- | ----- |
| pair 1       | 53,678                     | 54,574 | −1.6% | 526,314 ms       | 526,970 ms | −0.1% |
| pair 2       | 46,022                     | 49,394 | −6.8% | 444,063 ms       | 471,982 ms | −5.9% |

LPT-on is lower in both matched pairs on both metrics, and regresses neither
matched total. The reduction concentrates in the last iteration, where the
accumulated cut pool is largest and the per-block cost spread widest — pair 2,
iteration 3: imbalance 18,695 vs 22,036 ms (−15.2%), backward wall 176,819 vs
205,197 ms (−13.8%). Earlier iterations, with smaller pools, are within noise and
occasionally flip sign.

**Invariance cross-check — total backward `simplex_iterations`:**

| case           | LPT-on     | LPT-off    | Δ                       |
| -------------- | ---------- | ---------- | ----------------------- |
| `cobre_rodada` | 67,038,554 | 67,038,554 | +0.000% (bit-identical) |
| `1dtoy`        | 108        | 108        | +0.000%                 |
| `4ree`         | 6,684      | 6,684      | +0.000%                 |

Bit-identical, not merely close — the arms ran the same scheduler, case, and
iteration count, differing only in claim order. This is the expected reading and
a correctness cross-check, not a null value signal.

**Byte-neutrality cross-check — final lower bound, bitwise identity (LPT-on vs
LPT-off):**

| case           | LB bitwise identical |
| -------------- | -------------------- |
| `cobre_rodada` | yes                  |
| `1dtoy`        | yes                  |
| `4ree`         | yes                  |

The A/B compares equal-work builds: LPT changed no cut, corroborating
`lpt_claim_order_is_result_neutral`.

## Recommendation

The two cross-checks pass unambiguously. Total backward pivots are bit-identical
between arms (+0.000%) and the final lower bound is bitwise identical on every
case — LPT carries zero correctness risk here.

The primary signal is favorable in direction but small in magnitude. LPT-on
reduced `bwd_load_imbalance_ms` in both matched pairs (−1.6%, −6.8%) and reduced
or tied the matched-epoch backward wall in both (−0.1%, −5.9%), with the effect
concentrated at the largest cut pool. But the magnitude sits within the ~16%
desktop wall-noise band, and a single box at 14 threads is below the worker count
and pool sizes where the offline projection expected the makespan reduction to
materialize.

**Keep LPT in the tree pending the cluster-scale confirmation.** It is
result-neutral (cross-checked above), adds no local regression, and its
direction matches the projection. **Defer the keep/drop decision to the cluster
arm.** The local measurement rules out a regression and confirms byte-neutrality,
but cannot by itself size the makespan payoff or justify dropping the
claim-ordering kernel. The cluster arm — more workers, larger pools, matched
epochs — is the binding input; this local result is one data point feeding it.
