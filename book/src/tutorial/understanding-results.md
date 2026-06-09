# Understanding Results

After `cobre run` completes, the output directory contains three categories of
artifacts: training convergence data, a saved policy checkpoint, and simulation
dispatch results. This page explains how to read each category and how to query
the results programmatically using `cobre report`.

If you have not yet run the quickstart, complete [Quickstart](./quickstart.md)
first — this page references the `my_first_study/results/` directory produced
by that walkthrough.

---

## The Post-Run Summary

When `cobre run` finishes, it prints a summary block to stderr. The 1dtoy run
from the quickstart produces output similar to:

```
Training complete in 3.2s (128 iterations, iteration_limit)
  Lower bound:  142.3 $/stage
  Upper bound:  143.1 +/- 1.2 $/stage
  Gap:          0.6%
  Cuts:         384 active / 384 generated
  LP solves:    512

Simulation complete (100 scenarios)
  Completed: 100  Failed: 0

Output written to my_first_study/results/
```

Exact numerical values vary across runs because scenario sampling is stochastic.
The values below are representative of the 1dtoy example; your run will differ
slightly.

| Line                                                          | What it means                                                                                                                                                                      |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Training complete in 3.2s (128 iterations, iteration_limit)` | Training ran for 128 iterations (the limit set in `config.json`) and stopped because the iteration limit was reached, not because a convergence criterion was met.                 |
| `Lower bound:  142.3 $/stage`                                 | The optimizer's best proven lower bound on the minimum expected cost per stage. As training progresses this value rises and stabilizes.                                            |
| `Upper bound:  143.1 +/- 1.2 $/stage`                         | A statistical estimate of the true expected cost, computed from the forward-pass scenarios in the final iteration. The `+/- 1.2` is the standard deviation across those scenarios. |
| `Gap:  0.6%`                                                  | The relative distance between the lower and upper bounds expressed as a percentage. A gap of 0.6% means the policy cost is within 0.6% of the best possible. Smaller is better.    |
| `Cuts:  384 active / 384 generated`                           | The total number of optimality cuts in the policy pool. All 384 are currently active; none were deactivated (the 1dtoy config does not enable cut selection).                     |
| `LP solves:  512`                                             | Total number of linear programs solved across all stages and iterations.                                                                                                           |
| `Simulation complete (100 scenarios)`                         | The post-training simulation evaluated the trained policy over 100 independently sampled scenarios.                                                                                |
| `Completed: 100  Failed: 0`                                   | All 100 scenarios completed without solver errors.                                                                                                                                 |
| `Output written to my_first_study/results/`                   | Root path of the output directory.                                                                                                                                                 |

**Lower bound vs. upper bound.** The lower bound is the optimizer's proven best
estimate of the minimum achievable cost. The upper bound is the average cost
observed when running the current policy over sampled scenarios. When the gap is
small, the policy is near-optimal. When the gap is large, running more iterations
will typically narrow it further.

**Termination reasons.** The parenthetical after the iteration count explains why
training stopped:

- `iteration_limit` — the maximum iteration count was reached (the 1dtoy default).
- `converged at iter N` — a convergence criterion was met at iteration N and training
  stopped early. This appears when you configure a `bound_stalling` or similar rule
  in `config.json`.

> **Theory reference**: For the mathematical definition of lower and upper bounds,
> optimality gap, and stopping criteria, see
> [Convergence](https://cobre-rs.github.io/cobre-docs/theory/convergence.html)
> in the methodology reference.

---

## Output Directory Structure

All artifacts are written under the results directory you specified with `--output`.
The 1dtoy run produces:

```
my_first_study/results/
  training/
    metadata.json           Run metadata: configuration, convergence, row-pool, bounds, solve stats
    convergence.parquet     Per-iteration convergence metrics (lower bound, upper bound, gap)
    dictionaries/
      codes.json            Integer-to-string code mappings for entity categories
      state_dictionary.json State variable definitions and units
      entities.csv          Entity registry (id, name, type)
      variables.csv         LP variable registry
      bounds.parquet        LP variable bound definitions
    timing/
      iterations.parquet    Per-iteration wall-clock timing broken down by phase
  policy/
    cuts/
      stage_000.bin         FlatBuffers-encoded optimality cuts for stage 0
      stage_001.bin         ... stage 1
      stage_002.bin         ... stage 2
      stage_003.bin         ... stage 3
    basis/
      stage_000.bin         LP basis checkpoints for warm-starting
      stage_001.bin
      stage_002.bin
      stage_003.bin
    metadata.json           Policy metadata: stage count, cut counts per stage
  simulation/
    metadata.json           Run metadata: scenario counts, cost statistics, solve stats
    buses/
      scenario_id=0000/data.parquet
      scenario_id=0001/data.parquet
      ...                   One partition per scenario
    costs/
      scenario_id=0000/data.parquet
      ...
    hydros/
      scenario_id=0000/data.parquet
      ...
    thermals/
      scenario_id=0000/data.parquet
      ...
    inflow_lags/            Inflow lag state data used to initialize scenario chains
```

The three top-level subdirectories have distinct roles:

- `training/` — everything produced during the training loop: convergence history,
  timing, and the dictionaries needed to interpret LP variable indices.
- `policy/` — the trained policy checkpoint. These binary files encode the optimality
  cuts built during training. They can be used to resume or extend a study.
- `simulation/` — the dispatch results from evaluating the trained policy over
  100 simulation scenarios.

---

## Training Results

### Reading `training/metadata.json`

The training metadata file is the canonical record of what happened during
training. The 1dtoy run produces:

```json
{
  "cobre_version": "0.8.0",
  "hostname": "fedora",
  "solver": "highs",
  "solver_version": "1.13.1",
  "started_at": "2026-06-09T14:09:50Z",
  "completed_at": "2026-06-09T14:09:50Z",
  "duration_seconds": 0.15,
  "status": "complete",
  "configuration": {
    "seed": null,
    "max_iterations": 128,
    "forward_passes": 1,
    "stopping_mode": "any",
    "policy_mode": "fresh"
  },
  "problem_dimensions": {
    "num_stages": 4,
    "num_hydros": 1,
    "num_thermals": 2,
    "num_buses": 1,
    "num_lines": 0
  },
  "iterations": {
    "completed": 128,
    "converged_at": null
  },
  "convergence": {
    "achieved": false,
    "final_gap_percent": -2590.77437875556,
    "termination_reason": "iteration_limit"
  },
  "row_pool": {
    "total_generated": 384,
    "total_active": 384,
    "peak_active": 384,
    "cuts_active": 384
  },
  "bounds": {
    "final_lower_bound": 15595518.381798675,
    "final_upper_bound": 579592.1986224408,
    "final_upper_bound_std": 0.0
  },
  "solve_stats": {
    "total_lp_solves": 5632,
    "first_try": 5632,
    "retried": 0,
    "failed": 0,
    "forward_solve_seconds": 0.016273423999999915,
    "backward_solve_seconds": 0.07880944500000037,
    "parallelism": 1
  },
  "distribution": {
    "backend": "local",
    "world_size": 1,
    "ranks_participated": 1,
    "num_nodes": 1,
    "threads_per_rank": 1,
    "hosts": [
      { "hostname": "fedora", "ranks": [0] }
    ]
  }
}
```

Field-by-field explanation of the key fields:

| Field                            | Meaning                                                                                                                                                                                      |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cobre_version`                  | The cobre binary version that produced this output. Useful for auditing results from different releases.                                                                                     |
| `solver`                         | LP backend used: `"highs"` or `"clp"`.                                                                                                                                                       |
| `status`                         | `"complete"` when the training run finished normally.                                                                                                                                        |
| `iterations.completed`           | Number of training iterations that were executed.                                                                                                                                            |
| `iterations.converged_at`        | If training stopped early due to a convergence criterion, the iteration number where it stopped. `null` for an iteration-limit stop.                                                         |
| `convergence.achieved`           | `true` if a convergence stopping rule was satisfied, `false` if the iteration limit was reached first.                                                                                       |
| `convergence.final_gap_percent`  | The gap between lower and upper bounds at the end of training, as a percentage. A large or negative value (as seen in the 1dtoy case) indicates the bounds have not tightened sufficiently. |
| `convergence.termination_reason` | Machine-readable reason for stopping. Common values: `"iteration_limit"`, `"bound_stalling"`.                                                                                                |
| `row_pool.total_generated`       | Total optimality cut rows created across all stages over the entire training run.                                                                                                            |
| `row_pool.total_active`          | Cut rows still active in the pool at the end of training.                                                                                                                                    |
| `row_pool.cuts_active`           | Cut rows currently active in the LP at termination.                                                                                                                                          |
| `bounds.final_lower_bound`       | Final proven lower bound on the minimum expected cost at termination.                                                                                                                        |
| `bounds.final_upper_bound`       | Final upper bound estimate at termination. `null` when upper-bound evaluation is disabled.                                                                                                   |
| `distribution.backend`           | Communication backend: `"local"` for single-process, `"mpi"` for distributed runs.                                                                                                          |
| `distribution.world_size`        | Number of processes involved in the run. `1` for single-process runs.                                                                                                                       |
| `distribution.threads_per_rank`  | Number of rayon worker threads per process.                                                                                                                                                  |

**What "converged" means in practice.** A converged run (`convergence.achieved:
true`) means a stopping rule determined that continuing would not meaningfully
improve the policy. The 1dtoy case hits its 128-iteration budget before a
convergence rule fires, so `achieved` is `false`. For larger studies, configure
a `bound_stalling` or `gap_threshold` stopping rule in `config.json` to stop
automatically when the gap stabilizes.

---

## Simulation Results

### Hive-Partitioned Layout

The simulation output uses Hive partitioning: results are split into one
`data.parquet` file per scenario, stored in a directory named
`scenario_id=NNNN/`. This layout is natively understood by Polars, Pandas
(via PyArrow), R's arrow package, and DuckDB — they can read the entire
`simulation/costs/` directory as a single table and filter by `scenario_id`
at the storage layer without loading all data into memory.

The four entity categories are:

| Directory   | Contents                                                                                                                  |
| ----------- | ------------------------------------------------------------------------------------------------------------------------- |
| `buses/`    | Power balance results: load, generation injections, deficit, and excess at each bus per stage and block.                  |
| `hydros/`   | Hydro dispatch: turbined flow, spillage, reservoir storage levels, inflows, and generation per plant per stage and block. |
| `thermals/` | Thermal dispatch: generation output per unit per cost segment per stage and block.                                        |
| `costs/`    | Objective cost breakdown: total cost, thermal cost, hydro cost, penalty cost, and discount factor per stage.              |

Results are in Parquet format. To read them, use any columnar data tool:

```python
# Polars — reads all 100 scenarios at once
import polars as pl
df = pl.read_parquet("my_first_study/results/simulation/costs/")
print(df.head())
```

```python
# Pandas + PyArrow
import pandas as pd
df = pd.read_parquet("my_first_study/results/simulation/costs/")
print(df.head())
```

```sql
-- DuckDB — filter to a single scenario
SELECT * FROM read_parquet('my_first_study/results/simulation/costs/**/*.parquet')
WHERE scenario_id = 0;
```

```r
# R with arrow
library(arrow)
ds <- open_dataset("my_first_study/results/simulation/costs/")
dplyr::collect(dplyr::filter(ds, scenario_id == 0))
```

---

## Querying Results with `cobre report`

`cobre report` reads the JSON metadata files and prints a structured JSON summary to
stdout. Use it with `jq` to extract specific metrics in scripts or CI pipelines.

```bash
# Print the full report
cobre report my_first_study/results
```

The output has this top-level shape:

```json
{
  "output_directory": "/abs/path/to/results",
  "status": "complete",
  "bounds": { "final_lower_bound": ..., "final_upper_bound": ... },
  "training": { "iterations": {}, "convergence": {}, "row_pool": {}, "bounds": {}, "configuration": {}, "problem_dimensions": {} },
  "cost": { "mean_cost": ..., "std_cost": ... } | null,
  "simulation": { "scenarios": {}, "cost": {} } | null
}
```

### Practical `jq` queries

```bash
# Extract the final convergence gap
cobre report my_first_study/results | jq '.training.convergence.final_gap_percent'

# Check how many iterations ran
cobre report my_first_study/results | jq '.training.iterations.completed'

# Check simulation scenario counts
cobre report my_first_study/results | jq '.simulation.scenarios'

# Use the status in a CI script: exit non-zero if training failed
status=$(cobre report my_first_study/results | jq -r '.status')
if [ "$status" != "complete" ]; then
  echo "Run did not complete successfully: $status" >&2
  exit 1
fi

# Check convergence was achieved (returns true or false)
cobre report my_first_study/results | jq '.training.convergence.achieved'
```

For the complete `cobre report` documentation and all available JSON fields,
see [CLI Reference](../guide/cli-reference.md#cobre-report).

For a detailed description of every field in every output file,
see [Output Format Reference](../reference/output-format.md).

---

## See Also

- [Convergence & Diagnostics](../guide/interpreting-results.md) — advanced analysis patterns and convergence assessment
- [CLI Reference](../guide/cli-reference.md) — all flags, subcommands, and exit codes
- [Configuration](../guide/configuration.md) — every `config.json` field documented
