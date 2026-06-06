# Dynamic lazy cut-selection: production sweep results

> **Status: skeleton — operator fills in the data tables below.**
>
> This document records the outcome of the operator-run production-scale sweep
> that characterises the scoring-versus-solve split of the dynamic lazy
> cut-selection path and informs whether a bounded candidate window is worth
> building. The instrumentation and the sweep harness
> (`scripts/dcs_production_sweep.sh`) are in place; the run and the verdict are
> manual operator steps. The empty tables and the recommendation section are
> intentionally left for the operator to complete after running the sweep.

## Background

The lazy selection loop scores every non-resident candidate cut once per inner
round (`score_violated_candidates`). At production cut-pool scale (tens of
thousands of cuts), the open question is whether that scoring cost swamps the
LP-solve savings the lazy path is meant to buy. If scoring dominates, bounding
the candidate window (scoring only a recency- or count-limited subset) is
warranted. If it does not, the exact, unbounded default is kept.

The harness measures the split directly. Each lazy solve accumulates its
scoring wall time into a per-worker counter; that counter is folded — by
snapshot delta at each phase boundary — into the per-worker timing emitted for
the forward and backward passes, and surfaced as the `lazy_scoring_ms` column
of `training/timing/iterations.parquet`. The scoring fraction is
`scoring / (scoring + solve)` summed across workers.

## Build command

Cluster build (MPI feature enabled):

```
cargo build --release --features mpi
```

Dual-backend builds, as needed for backend comparison:

```
cargo build --release --features mpi,highs          # default backend
cargo build --release --no-default-features --features mpi,clp
```

## Case-config knobs

The sweep is driven by the case `config.json`; the harness reads
`training.cut_selection.method` to learn which variant the case runs. Knobs the
operator varies between runs:

- **`training.stopping_rules` / `max_iterations`** — the iteration budget is a
  case-config knob. For the all-cuts baseline and for variants that disable
  candidate selection, cap the iteration count so the comparison is bounded.
- **`--threads`** — intra-node worker count (default 5 in the harness). Use the
  same value for the lazy and all-cuts runs so the split is comparable.
- **`--reps 1`** — on quiet, exclusively-allocated nodes a single repetition is
  representative; node exclusivity removes the cross-run noise that would
  otherwise motivate averaging. Increase `--reps` only on shared nodes.
- **SLURM submission gotcha** — submit the `.sbatch` job **from the harness
  directory**. The job's working directory and the `BASH_SOURCE`-derived script
  path both resolve relative to the submission directory under the SLURM
  spool-dir model; submitting from elsewhere makes the harness fail to locate
  the built binary and the case directory.

## Hyperparameter grid

The lazy path exposes five hyperparameters (each settable in the case
`config.json` under `training.cut_selection`). The operator sweeps these against
the all-cuts baseline:

| Hyperparameter        | Config field          | Role                                                        |
| --------------------- | --------------------- | ----------------------------------------------------------- |
| candidate window (k1) | `candidate_window`    | Recency window bounding the scored candidate subset.        |
| check cadence (k2)    | `check_frequency`     | How often the resident set is re-seeded from pool metadata. |
| per-round add (nadic) | `nadic`               | How many top-violated slots are added per inner round.      |
| violation tolerance   | `violation_tolerance` | The `epsilon_viol` threshold a candidate must exceed.       |
| activation iteration  | `start_iteration`     | First training iteration the lazy loop activates.           |

## Scoring-versus-solve split

> **Operator fills in.** One row per swept configuration. Read `scoring_ms` and
> the wall columns from `training/timing/iterations.parquet` (the harness writes
> the per-rep split to `sweep_report.txt` in `--out-dir`); read the converged
> bound from `training/metadata.json`.

| Config (k1 / k2 / nadic / eps / start) | Threads | Wall (s) | Scoring (ms) | Solve (ms) | Scoring fraction | Converged LB |
| -------------------------------------- | ------- | -------- | ------------ | ---------- | ---------------- | ------------ |
|                                        |         |          |              |            |                  |              |
|                                        |         |          |              |            |                  |              |
|                                        |         |          |              |            |                  |              |

All-cuts baseline (for cross-mode converged-bound agreement):

| Case | Threads | Wall (s) | Converged LB | Converged UB |
| ---- | ------- | -------- | ------------ | ------------ |
|      |         |          |              |              |

## Bounded candidate-window recommendation

> **Operator fills in.** State, from the split above, whether scoring dominates
> solve at production cut-pool scale, and therefore whether the bounded
> candidate-window lever is worth building or the unbounded exact default is
> kept. Note the threshold cut-pool size (if any) above which the verdict flips,
> and any backend-specific differences observed.

_(verdict to be recorded after the sweep run)_
