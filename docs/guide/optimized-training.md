# Optimized training runtime

The optimized runtime is a separate build. Install it in a version-specific
location and select it explicitly; do not replace an existing runtime executable
or its current-version manifest. Keep input and output directories separate when
comparing policies.

## Exact cut selection

Dynamic cut selection retains its existing configuration surface. Candidate
scoring uses the pool's coefficient matrix directly when most rows are candidates,
and gathers a compact matrix otherwise. Only the selected prefix of violated
rows is sorted, with the same violation/slot ordering. At the inner-loop limit,
the fallback loads every eligible missing row before solving: a row that is not
violated at the previous solution may bind after reoptimization.

```json
{
  "training": {
    "selection": { "method": "sampled", "forward_passes": 24 },
    "stopping_rules": [{ "type": "iteration_limit", "limit": 12 }],
    "cut_selection": {
      "selection": {
        "method": "dynamic",
        "candidate_recency": null,
        "seed_window": 2,
        "max_added_per_round": 20
      }
    }
  }
}
```

A finite `candidate_recency` excludes older rows and remains an approximation.
Dynamic selection uses the by-scenario scheduler; this build does not combine it
with by-node scheduling. Smaller resident LPs can require more solves, so measure
the whole pass rather than assuming that enabling dynamic selection helps.

## Experimental point budget

`training.backward_selection` is opt-in. Its absence processes every visited
point as before. The option supports sampled training with one MPI rank and
multiple worker threads. Enumerated training and multiple ranks are rejected.

```json
{
  "training": {
    "selection": { "method": "sampled", "forward_passes": 24 },
    "stopping_rules": [{ "type": "iteration_limit", "limit": 12 }],
    "backward_selection": {
      "initial_points": 6,
      "exploration_points": 2,
      "full_every": 4,
      "full_from_iteration": 8
    },
    "cut_selection": {
      "selection": { "method": "lml1", "check_frequency": 1 }
    }
  }
}
```

All four point-budget fields are positive integers. The diverse-point budget
increases linearly from `initial_points` toward full coverage at
`full_from_iteration`, separately at each node. Deterministic farthest-point
selection uses the complete state vector, with each coordinate normalized by
its observed range. `exploration_points` adds pseudo-random remaining states;
its ordering depends on iteration and original scenario identity, not worker
scheduling. Every `full_every` iterations, and from `full_from_iteration` onward,
every point is processed. The refinement iteration must fit within the configured
iteration limit.

Every selected point still integrates all successor openings with the configured
risk measure. Forward trajectories and their statistics are not subsampled.
Periodic complete passes provide coverage audits; there is no automatic
non-inferiority certificate or feedback controller that proves omitted cuts were
unnecessary. Time limits, shutdown, or another stopping rule can interrupt before
refinement. Inspect completion before treating a policy as refined.

Unused cut slots stay unoccupied, are excluded from row selection, and are not
exported as artificial zero-valued cuts. Checkpoints retain the generated active
and inactive cuts. Basis export falls back to no cached basis for sparse pools,
since policy reload compacts slot identities. Resume therefore preserves valid
cuts but can follow a different numerical trajectory.

This option changes the training trajectory and is experimental. Compare policy
quality out of sample on common scenarios and multiple training seeds. Include
cost, deficit, storage and an evaluation compatible with nested conditional risk;
the CVaR of total simulated cost alone is not that nested objective. A stable lower
bound is not a quality certificate.

## Reproducible compact comparisons

Run benchmarks serially on an otherwise idle machine. Both executables should
come from the same toolchain, solver and build profile. The harness copies the
case, freezes configuration, uses a separate simulation seed, alternates arm
order, applies a timeout and records binary hashes, commands, exit codes and wall
time. It never edits the source case or an installed runtime.

```sh
python3 scripts/benchmarks/training.py \
  --baseline /absolute/path/to/cobre-baseline \
  --candidate /absolute/path/to/cobre-candidate \
  --case examples/4ree \
  --output /absolute/path/to/new-comparison-directory \
  --iterations 12 --forwards 24 --threads 4 --repeats 3
```

The default comparison uses LML1 for baseline, candidate with exhaustive points,
and candidate with experimental point selection. Inspect the simulation and
training outputs as well as `results.json`; short runs establish regressions and
measurement plumbing, not production-scale non-inferiority. Scheduler/profile
sweeps and dynamic-selection comparisons should use matched case configurations.


## Preview artifacts

The `Release Myria runtime` workflow accepts `prerelease=true` when publishing an
opt-in runtime. It creates a prerelease without changing GitHub's latest release.
The runtime tag must be new. Keep the platform's recommended runtime unchanged
when registering the preview's CLI and Python artifacts.

For independent training repetitions, change `--seed` (timing `--repeats` keeps
that seed fixed). Use `--simulation-scheme out_of_sample` to generate inflows
outside the training opening set. Keep `--simulation-seed` fixed across arms.
`--scheduler by_node --block-size 10` supports matched scheduler experiments;
DCS still uses its by-scenario fallback.

With `pyarrow` available, summarize an output directory with
`python3 scripts/benchmarks/summarize.py /absolute/path/to/comparison`.
The paired cost interval is conditional on one pair of trained policies; it does
not cover variation between training seeds or certify nested CVaR quality.
