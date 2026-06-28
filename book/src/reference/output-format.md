# Output Format Reference

This page is the complete schema reference for every file produced by
`cobre run`. It documents column names, Arrow data types, nullability, JSON
field structures, and binary format layouts for the Parquet schemas, the
metadata files, the dictionary files, and the policy checkpoint format.

If you are new to Cobre output, start with
[Understanding Results](../tutorial/understanding-results.md) first. That page
explains what each file means conceptually and shows how to read results
programmatically. This page is for readers who need the precise schema
definition — for writing parsers, building dashboards, or implementing
compatibility checks.

---

## Output Directory Tree

A complete `cobre run` produces the following directory structure. Not every
entity directory appears in every run: `cobre run` only writes directories for
entity types present in the case. For example, a case with no pumping stations
will not produce `simulation/pumping_stations/`.

```
<output_dir>/
  training/
    metadata.json
    convergence.parquet
    dictionaries/
      codes.json
      entities.csv
      variables.csv
      bounds.parquet
    timing/
      iterations.parquet
      mpi_ranks.parquet
    solver/
      iterations.parquet
      retry_histogram.parquet
    scaling_report.json
    cut_selection/
      iterations.parquet         (when cut_selection is enabled)
  policy/
    cuts/
      stage_000.bin
      stage_001.bin
      ...
      stage_NNN.bin
    basis/
      stage_000.bin
      stage_001.bin
      ...
      stage_NNN.bin
    metadata.json
    states/                         # when exports.states = true
      stage_000.bin
      stage_001.bin
      ...
      stage_NNN.bin
  simulation/
    metadata.json
    costs/
      scenario_id=0000/
        data.parquet
      scenario_id=0001/
        data.parquet
      ...
    hydros/
      scenario_id=0000/data.parquet
      ...
    thermals/
      scenario_id=0000/data.parquet
      ...
    exchanges/
      scenario_id=0000/data.parquet
      ...
    buses/
      scenario_id=0000/data.parquet
      ...
    pumping_stations/
      scenario_id=0000/data.parquet
      ...
    contracts/
      scenario_id=0000/data.parquet
      ...
    non_controllables/
      scenario_id=0000/data.parquet
      ...
    inflow_lags/
      scenario_id=0000/data.parquet
      ...
    violations/
      generic/
        scenario_id=0000/data.parquet
        ...
    solver/
      iterations.parquet
      retry_histogram.parquet
  hydro_models/
    fpha_hyperplanes.parquet         (when any hydro uses source: "computed")
    evaporation_models.parquet       (when any hydro has evaporation)
    fpha_deviation_points.parquet    (when exports.fpha_deviation_points = true)
  stochastic/
    inflow_seasonal_stats.parquet    (when estimation was performed)
    inflow_ar_coefficients.parquet   (when estimation was performed)
    correlation.json                 (always)
    fitting_report.json              (when estimation was performed)
    noise_openings.parquet           (always)
    load_seasonal_stats.parquet      (when load buses exist)
```

---

## Training Output

### `training/metadata.json`

The training metadata file is written atomically at the end of the training run.
It merges run context, configuration, convergence outcome, row-pool statistics,
objective bounds, LP solver statistics, and distribution information into a
single file. Consumers should check `status` before interpreting other fields.

**Example (from `output/training/metadata.json` after a run):**

```json
{
  "cobre_version": "0.9.1",
  "hostname": "<hostname>",
  "solver": "highs",
  "solver_version": "<solver version>",
  "started_at": "<timestamp>",
  "completed_at": "<timestamp>",
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
    "final_gap_percent": -2590.77,
    "termination_reason": "iteration_limit"
  },
  "row_pool": {
    "total_generated": 384,
    "total_active": 384,
    "peak_active": 384,
    "cuts_active": 384,
    "rows_in_lp_total": 0,
    "rows_in_lp_solve_count": 0,
    "rows_in_lp_max": 0
  },
  "bounds": {
    "final_lower_bound": 15595518.38,
    "final_upper_bound": 579592.2,
    "final_upper_bound_std": 0.0
  },
  "solve_stats": {
    "total_lp_solves": 5632,
    "first_try": 5632,
    "retried": 0,
    "failed": 0,
    "forward_solve_seconds": 0.016,
    "backward_solve_seconds": 0.079,
    "parallelism": 1
  },
  "distribution": {
    "backend": "local",
    "world_size": 1,
    "ranks_participated": 1,
    "num_nodes": 1,
    "threads_per_rank": 1,
    "hosts": [{ "hostname": "<hostname>", "ranks": [0] }]
  }
}
```

**Top-level fields:**

| Field              | Type   | Nullable | Description                                                                       |
| ------------------ | ------ | -------- | --------------------------------------------------------------------------------- |
| `cobre_version`    | string | No       | Version of the cobre binary that produced this output (from `CARGO_PKG_VERSION`). |
| `hostname`         | string | No       | Hostname of the machine that ran training.                                        |
| `solver`           | string | No       | LP solver backend: `"highs"` or `"clp"`.                                          |
| `solver_version`   | string | Yes      | Version string of the linked LP solver library. Omitted when not available.       |
| `started_at`       | string | No       | ISO 8601 timestamp when training started.                                         |
| `completed_at`     | string | No       | ISO 8601 timestamp when training completed.                                       |
| `duration_seconds` | number | No       | Total training wall-clock duration in seconds.                                    |
| `status`           | string | No       | Run status: `"complete"` or `"partial"`.                                          |

**`configuration` fields:**

| Field            | Type    | Nullable | Description                                                                              |
| ---------------- | ------- | -------- | ---------------------------------------------------------------------------------------- |
| `seed`           | integer | Yes      | Random seed used for scenario generation. `null` when not set.                           |
| `max_iterations` | integer | Yes      | Maximum iterations from the iteration-limit stopping rule. `null` when no limit was set. |
| `forward_passes` | integer | Yes      | Number of forward-pass scenario trajectories per iteration.                              |
| `stopping_mode`  | string  | No       | How multiple stopping rules combine: `"any"` or `"all"`.                                 |
| `policy_mode`    | string  | No       | Policy warm-start mode: `"fresh"` or `"resume"`.                                         |

**`problem_dimensions` fields:**

| Field          | Type    | Nullable | Description                               |
| -------------- | ------- | -------- | ----------------------------------------- |
| `num_stages`   | integer | No       | Number of stages in the planning horizon. |
| `num_hydros`   | integer | No       | Total number of hydro plants.             |
| `num_thermals` | integer | No       | Total number of thermal plants.           |
| `num_buses`    | integer | No       | Total number of buses.                    |
| `num_lines`    | integer | No       | Total number of transmission lines.       |

**`iterations` fields:**

| Field          | Type    | Nullable | Description                                                                                             |
| -------------- | ------- | -------- | ------------------------------------------------------------------------------------------------------- |
| `completed`    | integer | No       | Number of training iterations that finished.                                                            |
| `converged_at` | integer | Yes      | Iteration at which a convergence stopping rule triggered termination. `null` for iteration-limit stops. |

**`convergence` fields:**

| Field                | Type    | Nullable | Description                                                                                                                   |
| -------------------- | ------- | -------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `achieved`           | boolean | No       | `true` if a convergence-oriented stopping rule terminated the run.                                                            |
| `final_gap_percent`  | number  | Yes      | Optimality gap between lower and upper bounds at termination as a percentage. `null` when upper bound evaluation is disabled. |
| `termination_reason` | string  | No       | Machine-readable termination label. Common values: `"iteration_limit"`, `"bound_stalling"`.                                   |

**`row_pool` fields:**

| Field                    | Type    | Nullable | Description                                                                                             |
| ------------------------ | ------- | -------- | ------------------------------------------------------------------------------------------------------- |
| `total_generated`        | integer | No       | Total cut rows generated over the entire run.                                                           |
| `total_active`           | integer | No       | Cut rows still active in the pool at termination.                                                       |
| `peak_active`            | integer | No       | Highest number of simultaneously active cut rows observed.                                              |
| `cuts_active`            | integer | No       | Cut rows currently active in the LP at termination.                                                     |
| `rows_in_lp_total`       | integer | No       | Sum of resident rows-in-LP over every lazy-selection solve in the run. Zero when no lazy selection ran. |
| `rows_in_lp_solve_count` | integer | No       | Number of lazy-selection solves in the run. Zero when no lazy selection ran.                            |
| `rows_in_lp_max`         | integer | No       | Largest resident rows-in-LP over any single lazy-selection solve. Zero when no lazy selection ran.      |

**`bounds` fields:**

| Field                   | Type   | Nullable | Description                                                                    |
| ----------------------- | ------ | -------- | ------------------------------------------------------------------------------ |
| `final_lower_bound`     | number | No       | Final lower bound on the objective at termination.                             |
| `final_upper_bound`     | number | Yes      | Final upper bound estimate. `null` when upper-bound evaluation is disabled.    |
| `final_upper_bound_std` | number | Yes      | Standard deviation of the final upper-bound estimate. `null` when unavailable. |

**`solve_stats` fields:**

| Field                    | Type    | Nullable | Description                                                   |
| ------------------------ | ------- | -------- | ------------------------------------------------------------- |
| `total_lp_solves`        | integer | Yes      | Total number of LP solves performed during training.          |
| `first_try`              | integer | Yes      | Number of LP solves that succeeded on the first attempt.      |
| `retried`                | integer | Yes      | Number of LP solves that succeeded after one or more retries. |
| `failed`                 | integer | Yes      | Number of LP solves that failed terminally.                   |
| `forward_solve_seconds`  | number  | Yes      | Cumulative wall-clock seconds in forward-phase LP solves.     |
| `backward_solve_seconds` | number  | Yes      | Cumulative wall-clock seconds in backward-phase LP solves.    |
| `parallelism`            | integer | Yes      | Degree of parallelism (worker count) used during training.    |

**`distribution` fields:**

| Field                | Type          | Nullable | Description                                                                                                                      |
| -------------------- | ------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `backend`            | string        | No       | Communication backend: `"mpi"` or `"local"`.                                                                                     |
| `world_size`         | integer       | No       | Total number of processes in the communicator. `1` for single-process runs.                                                      |
| `ranks_participated` | integer       | No       | Number of processes that participated in computation.                                                                            |
| `num_nodes`          | integer       | No       | Number of distinct physical hosts.                                                                                               |
| `threads_per_rank`   | integer       | No       | Rayon worker threads per process.                                                                                                |
| `mpi_library`        | string        | Yes      | MPI implementation version (e.g. `"Open MPI v4.1.6"`). Omitted for the local backend.                                            |
| `mpi_standard`       | string        | Yes      | MPI standard version (e.g. `"MPI 4.0"`). Omitted for the local backend.                                                          |
| `thread_level`       | string        | Yes      | Negotiated MPI thread safety level. Omitted for the local backend.                                                               |
| `slurm_job_id`       | string        | Yes      | SLURM job ID when running under SLURM. Omitted otherwise.                                                                        |
| `hosts`              | array         | No       | Per-host rank assignment. One entry per physical host. For local single-process runs, contains a single entry with `ranks: [0]`. |
| `hosts[].hostname`   | string        | No       | Hostname for this entry.                                                                                                         |
| `hosts[].ranks`      | integer array | No       | Sorted global ranks assigned to this host.                                                                                       |

**`setup` fields** (absent from legacy metadata produced before setup timing was collected):

| Field                     | Type   | Nullable | Description                                                               |
| ------------------------- | ------ | -------- | ------------------------------------------------------------------------- |
| `load_seconds`            | number | No       | Wall-clock seconds spent loading the input case.                          |
| `stochastic_fit_seconds`  | number | No       | Wall-clock seconds spent fitting the stochastic process.                  |
| `production_fit_seconds`  | number | No       | Wall-clock seconds spent fitting the production model (FPHA hyperplanes). |
| `evaporation_fit_seconds` | number | No       | Wall-clock seconds spent fitting the evaporation model.                   |
| `broadcast_seconds`       | number | No       | Wall-clock seconds spent broadcasting setup data across MPI ranks.        |

These values are non-deterministic (informational only): they vary run-to-run with
machine load and are excluded from any parity computation. The entire `setup` key
is omitted from metadata produced before setup timing was introduced, and any field
absent in such legacy metadata deserialises as `0.0`.

---

### `training/convergence.parquet`

Per-iteration convergence log. One row per training iteration. 14 columns.

| Column             | Type    | Nullable | Description                                                                                                   |
| ------------------ | ------- | -------- | ------------------------------------------------------------------------------------------------------------- |
| `iteration`        | Int32   | No       | Training iteration number (1-based).                                                                          |
| `lower_bound`      | Float64 | No       | Best proven lower bound on the minimum expected cost after this iteration.                                    |
| `upper_bound_mean` | Float64 | No       | Mean upper bound estimate from the forward-pass scenarios in this iteration.                                  |
| `upper_bound_std`  | Float64 | No       | Standard deviation of the upper bound estimate across forward-pass scenarios.                                 |
| `gap_percent`      | Float64 | Yes      | Relative gap between lower and upper bounds as a percentage. `null` when the lower bound is zero or negative. |
| `cuts_added`       | Int32   | No       | Number of new cuts added to the pool during this iteration's backward pass.                                   |
| `cuts_removed`     | Int32   | No       | Number of cuts deactivated by the cut selection strategy in this iteration.                                   |
| `cuts_active`      | Int64   | No       | Total number of active cuts across all stages at the end of this iteration.                                   |
| `time_forward_ms`  | Int64   | No       | Wall-clock time spent in the forward pass, in milliseconds.                                                   |
| `time_backward_ms` | Int64   | No       | Wall-clock time spent in the backward pass, in milliseconds.                                                  |
| `time_total_ms`    | Int64   | No       | Total wall-clock time for this iteration, in milliseconds.                                                    |
| `forward_passes`   | Int32   | No       | Number of forward-pass scenario trajectories evaluated in this iteration.                                     |
| `lp_solves`        | Int64   | No       | Total number of LP solves across all stages and forward passes in this iteration.                             |
| `mean_rows_in_lp`  | Float64 | No       | Mean number of active LP rows across all stage solves in this iteration.                                      |

---

### `training/timing/iterations.parquet`

Per-iteration wall-clock timing breakdown by phase. 19 columns. Emitted as one
row per `(iteration, rank)` for rank-only sequential values (`worker_id` is
NULL) and one row per `(iteration, rank, worker_id)` for per-worker
parallel-region values; `SUM(col) GROUP BY iteration` recovers the
per-iteration total for each timing column. `rank` and `worker_id` are nullable
Int32; the 16 timing columns are non-nullable.

The top-level non-overlapping phases are: `forward_wall_ms`,
`backward_wall_ms`, `cut_selection_ms`, `mpi_allreduce_ms`, and
`lower_bound_ms`. The backward parallel overhead is decomposed into three
components: `bwd_setup_ms` (aggregate non-solve work summed across
workers), `bwd_load_imbalance_ms` (max-worker minus average-worker),
and `bwd_scheduling_overhead_ms` (parallel wall minus max-worker). The
forward pass carries the same three sub-components with `fwd_` prefix.
The backward phase also has the sub-components `cut_sync_ms`,
`state_exchange_ms`, and `cut_batch_build_ms`. The residual not
attributed to any phase is `overhead_ms`.

| Column                       | Type  | Nullable | Description                                                                                                                                                                                 |
| ---------------------------- | ----- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `iteration`                  | Int32 | No       | Training iteration number (1-based).                                                                                                                                                        |
| `rank`                       | Int32 | Yes      | MPI rank that produced this row. NULL for rank-aggregated rows.                                                                                                                             |
| `worker_id`                  | Int32 | Yes      | Rayon worker index within the rank's pool. NULL for rank-only sequential rows.                                                                                                              |
| `forward_wall_ms`            | Int64 | No       | Wall-clock time for the forward pass (all stages and scenarios).                                                                                                                            |
| `backward_wall_ms`           | Int64 | No       | Wall-clock time for the backward pass (all stages and trial points).                                                                                                                        |
| `cut_selection_ms`           | Int64 | No       | Time spent running the cut selection pipeline (all three stages).                                                                                                                           |
| `mpi_allreduce_ms`           | Int64 | No       | Time spent in MPI allreduce (forward-pass bound synchronization).                                                                                                                           |
| `cut_sync_ms`                | Int64 | No       | Time spent in per-stage cut sync allgatherv (sub-component of backward).                                                                                                                    |
| `lower_bound_ms`             | Int64 | No       | Time spent evaluating the lower bound (stage-0 LP solves for all openings).                                                                                                                 |
| `state_exchange_ms`          | Int64 | No       | Time spent in state exchange allgatherv (sub-component of backward).                                                                                                                        |
| `cut_batch_build_ms`         | Int64 | No       | Time spent assembling cut row batches (sub-component of backward).                                                                                                                          |
| `bwd_setup_ms`               | Int64 | No       | Aggregate non-solve work (load_model + add_rows + set_bounds + basis_set) summed across backward workers, in ms. May exceed `backward_wall_ms`; it is a cost metric, not a wall-time slice. |
| `bwd_load_imbalance_ms`      | Int64 | No       | Backward load imbalance: `max_worker_total - avg_worker_total`, clamped to zero.                                                                                                            |
| `bwd_scheduling_overhead_ms` | Int64 | No       | Backward scheduling overhead: `parallel_wall - max_worker_total`, clamped to zero.                                                                                                          |
| `fwd_setup_ms`               | Int64 | No       | Aggregate non-solve work summed across forward workers, in ms. Same aggregate semantics as `bwd_setup_ms`.                                                                                  |
| `fwd_load_imbalance_ms`      | Int64 | No       | Forward load imbalance: `max_worker_total - avg_worker_total`, clamped to zero.                                                                                                             |
| `fwd_scheduling_overhead_ms` | Int64 | No       | Forward scheduling overhead: `parallel_wall - max_worker_total`, clamped to zero.                                                                                                           |
| `overhead_ms`                | Int64 | No       | Residual wall-clock time not attributed to any of the above phases.                                                                                                                         |
| `lazy_scoring_ms`            | Int64 | No       | Per-worker time spent in lazy candidate scoring inside the lazy-selection solve. A sub-component of the forward/backward phases (not a top-level addend); `0` when the lazy path is unused. |

> **Schema migration note (v0.4.x):** The single columns `bwd_rayon_overhead_ms` and `fwd_rayon_overhead_ms` from earlier releases were replaced with three columns each (`_setup_ms`, `_load_imbalance_ms`, `_scheduling_overhead_ms`). Downstream scripts that read the parquet by column name must be updated. The invariant `load_imbalance + scheduling <= parallel_wall` holds; `setup_ms` is a separate aggregate-across-workers cost and is not bounded by wall time.

### `training/timing/mpi_ranks.parquet`

Per-iteration, per-rank timing statistics for distributed runs. One row per
(iteration, rank) pair. 8 columns. All columns are non-nullable.

| Column                  | Type  | Nullable | Description                                                   |
| ----------------------- | ----- | -------- | ------------------------------------------------------------- |
| `iteration`             | Int32 | No       | Training iteration number (1-based).                          |
| `rank`                  | Int32 | No       | MPI rank index (0-based).                                     |
| `forward_time_ms`       | Int64 | No       | Wall-clock time this rank spent in the forward pass.          |
| `backward_time_ms`      | Int64 | No       | Wall-clock time this rank spent in the backward pass.         |
| `communication_time_ms` | Int64 | No       | Wall-clock time this rank spent in MPI communication.         |
| `idle_time_ms`          | Int64 | No       | Wall-clock time this rank was idle (waiting for other ranks). |
| `lp_solves`             | Int64 | No       | Number of LP solves performed by this rank in this iteration. |
| `scenarios_processed`   | Int32 | No       | Number of scenario trajectories processed by this rank.       |

---

### `training/solver/iterations.parquet`

Per-iteration, per-phase, per-stage, per-opening, per-worker LP solver
statistics for diagnosing conditioning issues and retry behavior. One row per
`(iteration, phase, stage, opening, rank, worker_id)` tuple on the backward
phase (per-opening, per-worker); one row per `(iteration, phase, stage)` tuple
on the forward, `lower_bound`, and `simulation` phases. 18 columns. Columns
`opening`, `rank`, and `worker_id` are nullable Int32; all other columns are
non-nullable.

| Column                       | Type    | Nullable | Description                                                                                                          |
| ---------------------------- | ------- | -------- | -------------------------------------------------------------------------------------------------------------------- |
| `iteration`                  | UInt32  | No       | Training iteration (1-based) or simulation scenario id (0-based).                                                    |
| `phase`                      | Utf8    | No       | `"forward"`, `"backward"`, `"lower_bound"`, or `"simulation"`.                                                       |
| `stage`                      | Int32   | No       | Stage index (0-based).                                                                                               |
| `opening`                    | Int32   | Yes      | Opening (noise realization) index within the stage for backward rows. NULL for forward, `lower_bound`, `simulation`. |
| `rank`                       | Int32   | Yes      | MPI rank that produced this row. NULL for rank-aggregated rows.                                                      |
| `worker_id`                  | Int32   | Yes      | Rayon worker index within the rank's pool. NULL for rows without a per-worker dimension.                             |
| `lp_solves`                  | UInt32  | No       | Number of LP solves in this row's bucket.                                                                            |
| `lp_successes`               | UInt32  | No       | Number of solves that returned optimal.                                                                              |
| `lp_retries`                 | UInt32  | No       | Number of solves that required at least one retry.                                                                   |
| `lp_failures`                | UInt32  | No       | Number of solves that failed after exhausting all retry levels.                                                      |
| `retry_attempts`             | UInt32  | No       | Total retry attempts across all LP solves in this bucket.                                                            |
| `basis_offered`              | UInt32  | No       | Number of `solve(Some(&basis))` calls (warm-start attempts).                                                         |
| `basis_consistency_failures` | UInt32  | No       | Number of warm-start calls in which the basis was rejected because `isBasisConsistent` returned false.               |
| `simplex_iterations`         | UInt64  | No       | Total simplex iterations (or IPM iterations) across all solves.                                                      |
| `solve_time_ms`              | Float64 | No       | Cumulative LP solve wall-clock time in milliseconds.                                                                 |
| `load_model_time_ms`         | Float64 | No       | Cumulative time spent in `load_model` calls, in milliseconds.                                                        |
| `set_bounds_time_ms`         | Float64 | No       | Cumulative time spent in `set_row_bounds` / `set_col_bounds` calls, in milliseconds.                                 |
| `basis_set_time_ms`          | Float64 | No       | Cumulative time spent installing bases for warm-start, in milliseconds.                                              |

### `simulation/solver/iterations.parquet`

Identical schema to [`training/solver/iterations.parquet`](#trainingsolveriterationsparquet).
One row per `(scenario, phase, stage)` triple where `phase == "simulation"`.

### `training/solver/retry_histogram.parquet`

Per-level retry success counts, normalized from the solver iterations
table. One row per `(iteration, phase, stage, retry_level)` tuple where
the count is positive (sparse encoding). 5 columns. All non-nullable.

| Column        | Type   | Nullable | Description                                                                                                      |
| ------------- | ------ | -------- | ---------------------------------------------------------------------------------------------------------------- |
| `iteration`   | UInt32 | No       | Training iteration number (1-based).                                                                             |
| `phase`       | Utf8   | No       | Algorithm phase: `"forward"`, `"backward"`, or `"lower_bound"`.                                                  |
| `stage`       | Int32  | No       | Stage index (0-based).                                                                                           |
| `retry_level` | UInt32 | No       | Retry escalation level (0--11). See [Solver Safeguards](../guide/performance-accelerators.md#solver-safeguards). |
| `count`       | UInt64 | No       | Number of LP solves recovered at this retry level.                                                               |

### `training/scaling_report.json`

LP prescaling diagnostics written once after stage template construction.
Documents the coefficient range before and after column/row scaling for
each stage. Useful for diagnosing numerical conditioning issues.

The JSON is an array of per-stage objects, each containing:

| Field                    | Type    | Description                                                   |
| ------------------------ | ------- | ------------------------------------------------------------- |
| `stage`                  | integer | Stage index (0-based).                                        |
| `before.coefficient_min` | number  | Smallest absolute non-zero matrix coefficient before scaling. |
| `before.coefficient_max` | number  | Largest absolute matrix coefficient before scaling.           |
| `before.rhs_min`         | number  | Smallest absolute non-zero RHS value before scaling.          |
| `before.rhs_max`         | number  | Largest absolute RHS value before scaling.                    |
| `after.coefficient_min`  | number  | Smallest absolute non-zero coefficient after scaling.         |
| `after.coefficient_max`  | number  | Largest absolute coefficient after scaling.                   |
| `after.rhs_min`          | number  | Smallest absolute non-zero RHS value after scaling.           |
| `after.rhs_max`          | number  | Largest absolute RHS value after scaling.                     |

### `training/cut_selection/iterations.parquet`

Per-stage cut selection statistics. One row per `(iteration, stage)` pair,
written only at iterations where selection ran. 10 columns.

| Column                | Type    | Nullable | Description                                                                 |
| --------------------- | ------- | -------- | --------------------------------------------------------------------------- |
| `iteration`           | Int32   | No       | Training iteration number (1-based).                                        |
| `stage`               | Int32   | No       | Stage index (0-based).                                                      |
| `cuts_populated`      | Int32   | No       | Total cut slots containing cuts (active + inactive).                        |
| `cuts_active_before`  | Int32   | No       | Active cuts before this iteration's selection pipeline.                     |
| `cuts_deactivated`    | Int32   | No       | Cuts deactivated by the strategy-based selection (Stage 1).                 |
| `cuts_reactivated`    | Int32   | No       | Cuts reactivated by the strategy-based selection (Stage 1).                 |
| `cuts_active_after`   | Int32   | No       | Active cuts after Stage 1 selection.                                        |
| `selection_time_ms`   | Float64 | No       | Wall-clock time for the full selection pipeline.                            |
| `budget_evicted`      | Int32   | Yes      | Cuts evicted by budget enforcement (Stage 2). `null` when S2 is disabled.   |
| `active_after_budget` | Int32   | Yes      | Active cuts after budget enforcement (Stage 2). `null` when S2 is disabled. |

---

### `training/dictionaries/`

Five self-documenting files that allow output Parquet files to be interpreted
without reference to the original input case. All files are written atomically.

#### `codes.json`

Static mapping from integer codes to human-readable labels for all categorical
fields used in Parquet output. The same mapping applies for the lifetime of a
release (the version field tracks breaking changes).

```json
{
  "version": "1.0",
  "generated_at": "<timestamp>",
  "operative_state": {
    "0": "deactivated",
    "1": "maintenance",
    "2": "operating",
    "3": "saturated"
  },
  "storage_binding": {
    "0": "none",
    "1": "below_minimum",
    "2": "above_maximum",
    "3": "both"
  },
  "contract_type": {
    "0": "import",
    "1": "export"
  },
  "entity_type": {
    "0": "hydro",
    "1": "thermal",
    "2": "bus",
    "3": "line",
    "4": "pumping_station",
    "5": "contract",
    "7": "non_controllable"
  },
  "bound_type": {
    "0": "storage_min",
    "1": "storage_max",
    "2": "turbined_min",
    "3": "turbined_max",
    "4": "outflow_min",
    "5": "outflow_max",
    "6": "generation_min",
    "7": "generation_max",
    "8": "flow_min",
    "9": "flow_max"
  }
}
```

#### `entities.csv`

One row per entity across all entity types. Columns:

| Column             | Description                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------ |
| `entity_type_code` | Integer entity type code (see `codes.json` `entity_type` mapping).                         |
| `entity_id`        | Integer entity ID matching the `*_id` column in the corresponding simulation Parquet file. |
| `name`             | Human-readable entity name from the case input files.                                      |
| `bus_id`           | Integer bus ID to which this entity is connected. For buses, equals `entity_id`.           |
| `system_id`        | System partition index. Always `0` in the current release (single-system cases).           |

Rows are ordered by `entity_type_code` ascending, then by `entity_id`
ascending within each type.

#### `variables.csv`

One row per output column across all Parquet schemas. Documents every column
name, its parent schema, and its unit of measure. Useful for building generic
result readers that do not hard-code column names.

| Column        | Description                                                                                                                               |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `schema`      | Name of the Parquet schema this column belongs to (e.g. `"hydros"`, `"costs"`).                                                           |
| `column_name` | Exact column name as it appears in the Parquet file.                                                                                      |
| `arrow_type`  | Arrow data type string (e.g. `"Int32"`, `"Float64"`, `"Boolean"`).                                                                        |
| `nullable`    | `"true"` or `"false"`.                                                                                                                    |
| `unit`        | Physical unit or `"code"` for categorical fields, `"boolean"` for flag fields, `"id"` for identifiers, `"dimensionless"` for pure ratios. |
| `description` | Short description of the column's meaning.                                                                                                |

#### `bounds.parquet`

Per-entity, per-stage resolved LP variable bounds. Documents the actual
numerical bounds used in each LP solve, after applying the three-tier penalty
resolution (global / entity / stage overrides).

| Column             | Type    | Nullable | Description                                              |
| ------------------ | ------- | -------- | -------------------------------------------------------- |
| `entity_type_code` | Int8    | No       | Entity type code (see `codes.json`).                     |
| `entity_id`        | Int32   | No       | Entity ID.                                               |
| `stage_id`         | Int32   | No       | Stage index (0-based).                                   |
| `bound_type_code`  | Int8    | No       | Bound type code (see `codes.json` `bound_type` mapping). |
| `lower_bound`      | Float64 | No       | Resolved lower bound value in the bound's natural unit.  |
| `upper_bound`      | Float64 | No       | Resolved upper bound value in the bound's natural unit.  |

> The state-space structure — which entity owns each state-vector dimension —
> is recorded directly in the policy checkpoint. Every `policy/cuts/stage_NNN.bin`
> and `policy/states/stage_NNN.bin` embeds a per-slot entity manifest, described
> under [Policy Checkpoint](#policy-checkpoint) below. There is no separate
> state-dictionary sidecar; read the manifest from the policy file to interpret a
> cut's coefficient vector.

---

## Policy Checkpoint

> The wire format of the binary files below is described by the
> canonical schema at `crates/cobre-io/schemas/policy.fbs`. See
> [FlatBuffers Schema (`policy/*.bin`)](./flatbuffers-schema.md) for
> recipes on dumping a `.bin` to JSON and on generating typed readers
> in Python, C++, TypeScript, and other languages with `flatc`.

### `policy/cuts/stage_NNN.bin`

FlatBuffers binary file encoding all cuts for a single stage. One file per
stage; file names are zero-padded to three digits (e.g. `stage_000.bin`,
`stage_012.bin`).

The binary is not human-readable. The logical record structure for each cut
contained in the file is:

| Field                | Type      | Description                                                                                                                          |
| -------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `cut_id`             | uint64    | Unique identifier for this cut across all iterations. Assigned monotonically by the training loop.                                   |
| `slot_index`         | uint32    | LP row position. Required for checkpoint reproducibility and basis warm-starting.                                                    |
| `iteration`          | uint32    | Training iteration that generated this cut.                                                                                          |
| `forward_pass_index` | uint32    | Forward pass index within the generating iteration.                                                                                  |
| `intercept`          | float64   | Pre-computed cut intercept: `alpha - beta' * x_hat`, where `x_hat` is the state at the generating forward pass node.                 |
| `coefficients`       | float64[] | Gradient coefficient vector. Length equals the file's `state_dimension` and the length of its `entity_manifest`.                     |
| `is_active`          | bool      | Whether this cut is currently active in the LP. Inactive cuts are retained for potential reactivation by the cut selection strategy. |

Alongside the cuts, the file carries an `entity_manifest`: a per-slot record of
which entity owns each state-vector dimension, so coefficient position `i` can
be attributed without an external lookup. The manifest length equals the file's
`state_dimension`, and slot `i` describes coefficient position `i` on every cut.

| Field         | Type   | Description                                                                                                                                                           |
| ------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `entity_type` | enum   | State-variable class: hydro storage, hydro inflow lag, or anticipated thermal commitment state.                                                                       |
| `entity_id`   | int32  | Owning entity's id, matching the `entity_id` columns in `entities.csv` and `bounds.parquet`.                                                                          |
| `subindex`    | uint32 | Secondary index within the owning entity: the 1-based autoregressive lag order for an inflow lag, the ring-buffer slot for an anticipated state, and `0` for storage. |
| `was_active`  | bool   | Whether the owning entity was operationally active at this slot's stage. A dormant entity's slot is present but carries zero influence.                               |

The encoding uses the FlatBuffers runtime builder API (little-endian, no
reflection, no generated code). Field order in the binary matches the
declaration order above.

Legacy policy files that still contain the `CUT_FIELD_DOMINATION_COUNT`
FlatBuffer slot deserialise via the `field_pos` graceful-absence pattern
and the value is discarded; the field is not present in policy files written
by the current release.

### `policy/basis/stage_NNN.bin`

FlatBuffers binary file encoding the LP simplex basis checkpoint for a single
stage. One file per stage. Used to warm-start LP solves when resuming a study.

The logical record structure is:

| Field           | Type    | Description                                                                                                 |
| --------------- | ------- | ----------------------------------------------------------------------------------------------------------- |
| `stage_id`      | uint32  | Stage index (0-based).                                                                                      |
| `iteration`     | uint32  | Training iteration that produced this basis.                                                                |
| `column_status` | uint8[] | One status code per LP column (variable). Encoding is HiGHS-specific.                                       |
| `row_status`    | uint8[] | One status code per LP row (constraint). Encoding is HiGHS-specific.                                        |
| `num_cut_rows`  | uint32  | Number of trailing rows in `row_status` that correspond to cut rows (as opposed to structural constraints). |

### `policy/states/stage_NNN.bin`

FlatBuffers binary file encoding the visited forward-pass trial points for a
single stage. One file per stage. Present only when `exports.states` is `true`
(default is `false`). The `states/` directory is omitted entirely when disabled.

Trial points are the state vectors observed at each forward-pass scenario
during training. They are always collected in memory regardless of the cut
selection method, but persisted to disk only when this export flag is set.
Dominated cut selection uses these states at pruning time; for other methods
they serve as a diagnostic and analysis artifact.

| Field             | Type      | Description                                                                      |
| ----------------- | --------- | -------------------------------------------------------------------------------- |
| `stage_id`        | uint32    | Stage index (0-based).                                                           |
| `state_dimension` | uint32    | Length of each state vector. Equals the length of the file's `entity_manifest`.  |
| `count`           | uint32    | Number of state vectors stored for this stage.                                   |
| `data`            | float64[] | Flat array of `count * state_dimension` elements, row-major (one state per row). |

This file carries the same per-slot `entity_manifest` as
`policy/cuts/stage_NNN.bin` (one entry per state-vector dimension), so each
column of `data` can be attributed to its owning entity.

### `policy/metadata.json`

Small JSON file describing the checkpoint at a high level. Human-readable and
machine-readable by tooling that inspects policy files.

| Field                  | Type      | Nullable | Description                                                                                                                      |
| ---------------------- | --------- | -------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `cobre_version`        | string    | No       | Version of the cobre binary that wrote this checkpoint.                                                                          |
| `created_at`           | string    | No       | ISO 8601 timestamp when the checkpoint was written.                                                                              |
| `completed_iterations` | integer   | No       | Number of training iterations completed at checkpoint time.                                                                      |
| `final_lower_bound`    | number    | No       | Lower bound value after the final completed iteration.                                                                           |
| `best_upper_bound`     | number    | Yes      | Best upper bound observed during training. `null` when upper bound evaluation was disabled.                                      |
| `state_dimension`      | integer   | No       | Length of each cut's coefficient vector. Equals the `state_dimension` of every per-stage cut and state file.                     |
| `num_stages`           | integer   | No       | Number of stages. Must match the case configuration on resume.                                                                   |
| `max_iterations`       | integer   | No       | Maximum iterations configured for the run.                                                                                       |
| `forward_passes`       | integer   | No       | Number of forward passes per iteration configured for the run.                                                                   |
| `warm_start_cuts`      | integer   | No       | Number of cuts loaded from a previous policy at run start. `0` for fresh runs.                                                   |
| `warm_start_counts`    | integer[] | No       | Per-stage warm-start cut counts (one per stage, 0-based). Empty in old checkpoints; supersedes `warm_start_cuts` when non-empty. |
| `rng_seed`             | integer   | No       | RNG seed used by the scenario sampler. Required for reproducibility.                                                             |
| `total_visited_states` | integer   | No       | Total number of visited state vectors across all stages. `0` when `exports.states` is off.                                       |

---

## Simulation Output

All simulation results use Hive partitioning: one `data.parquet` file per
scenario stored in a `scenario_id=NNNN/` subdirectory. See
[Hive Partitioning](#hive-partitioning) below for how to read these files.

### `simulation/metadata.json`

The simulation metadata file is written atomically when simulation completes.
It captures run context, scenario completion counts, aggregate cost statistics,
LP solver statistics, and distribution information.

**Example (from `output/simulation/metadata.json` after a run):**

```json
{
  "cobre_version": "0.9.1",
  "hostname": "<hostname>",
  "solver": "highs",
  "started_at": "<timestamp>",
  "completed_at": "<timestamp>",
  "duration_seconds": 0.103,
  "status": "complete",
  "scenarios": {
    "total": 100,
    "completed": 100,
    "failed": 0
  },
  "cost": {
    "mean_cost": 14532064.35,
    "std_cost": 35658862.19,
    "cvar": 143086183.17,
    "cvar_alpha": 0.95
  },
  "solve_stats": {
    "total_lp_solves": 400,
    "first_try": 400,
    "retried": 0,
    "failed": 0,
    "solve_seconds": 0.017,
    "parallelism": 1
  },
  "distribution": {
    "backend": "local",
    "world_size": 1,
    "ranks_participated": 1,
    "num_nodes": 1,
    "threads_per_rank": 1,
    "hosts": [{ "hostname": "<hostname>", "ranks": [0] }]
  }
}
```

**Top-level fields:**

| Field              | Type   | Nullable | Description                                                   |
| ------------------ | ------ | -------- | ------------------------------------------------------------- |
| `cobre_version`    | string | No       | Version of the cobre binary that produced this output.        |
| `hostname`         | string | No       | Hostname of the machine that ran simulation.                  |
| `solver`           | string | No       | LP solver backend: `"highs"` or `"clp"`.                      |
| `solver_version`   | string | Yes      | LP solver library version string. Omitted when not available. |
| `started_at`       | string | No       | ISO 8601 timestamp when simulation started.                   |
| `completed_at`     | string | No       | ISO 8601 timestamp when simulation completed.                 |
| `duration_seconds` | number | No       | Total simulation wall-clock duration in seconds.              |
| `status`           | string | No       | Run status: `"complete"` or `"partial"`.                      |

**`scenarios` fields:**

| Field       | Type    | Nullable | Description                                            |
| ----------- | ------- | -------- | ------------------------------------------------------ |
| `total`     | integer | No       | Total number of scenarios dispatched for simulation.   |
| `completed` | integer | No       | Number of scenarios that completed without error.      |
| `failed`    | integer | No       | Number of scenarios that encountered a terminal error. |

**`cost` fields** (omitted when cost was not persisted):

| Field        | Type   | Nullable | Description                                                      |
| ------------ | ------ | -------- | ---------------------------------------------------------------- |
| `mean_cost`  | number | No       | Mean total cost across simulated scenarios.                      |
| `std_cost`   | number | No       | Standard deviation of the total cost across simulated scenarios. |
| `cvar`       | number | No       | Conditional Value-at-Risk at `cvar_alpha`.                       |
| `cvar_alpha` | number | No       | Confidence level for the CVaR computation, in `(0, 1)`.          |

**`solve_stats` fields:**

| Field             | Type    | Nullable | Description                                                   |
| ----------------- | ------- | -------- | ------------------------------------------------------------- |
| `total_lp_solves` | integer | Yes      | Total number of LP solves performed during simulation.        |
| `first_try`       | integer | Yes      | Number of LP solves that succeeded on the first attempt.      |
| `retried`         | integer | Yes      | Number of LP solves that succeeded after one or more retries. |
| `failed`          | integer | Yes      | Number of LP solves that failed terminally.                   |
| `solve_seconds`   | number  | Yes      | Cumulative wall-clock seconds spent in simulation LP solves.  |
| `parallelism`     | integer | Yes      | Degree of parallelism (worker count) used during simulation.  |

The `distribution` object has the same field structure as in `training/metadata.json`.
See the **`distribution` fields** table above.

---

### `simulation/costs/`

Stage and block-level cost breakdown. One row per (stage, block) pair. 27 columns.

| Column                         | Type    | Nullable | Description                                                                                                                  |
| ------------------------------ | ------- | -------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `stage_id`                     | Int32   | No       | Stage index (0-based).                                                                                                       |
| `block_id`                     | Int32   | Yes      | Load block index within the stage. `null` for stage-level (non-block) records.                                               |
| `total_cost`                   | Float64 | No       | Total discounted cost for this stage/block (monetary units).                                                                 |
| `immediate_cost`               | Float64 | No       | Immediate (undiscounted) cost for this stage/block.                                                                          |
| `future_cost`                  | Float64 | No       | Future cost estimate (Benders cut value) at the end of this stage.                                                           |
| `discount_factor`              | Float64 | No       | Discount factor applied to this stage's costs.                                                                               |
| `thermal_cost`                 | Float64 | No       | Thermal generation cost component.                                                                                           |
| `anticipated_thermal_cost`     | Float64 | No       | Anticipated (forward-committed) thermal generation cost, booked at the decision stage. Zero when no anticipated units exist. |
| `contract_cost`                | Float64 | No       | Energy contract cost component (positive for imports, negative for exports).                                                 |
| `deficit_cost`                 | Float64 | No       | Cost of unserved load (deficit penalty).                                                                                     |
| `excess_cost`                  | Float64 | No       | Cost of excess generation (excess penalty).                                                                                  |
| `storage_violation_cost`       | Float64 | No       | Cost of reservoir storage bound violations.                                                                                  |
| `filling_target_cost`          | Float64 | No       | Cost of missing reservoir filling targets.                                                                                   |
| `hydro_violation_cost`         | Float64 | No       | Cost of hydro operational bound violations.                                                                                  |
| `outflow_violation_below_cost` | Float64 | No       | Cost of total outflow below-minimum violations.                                                                              |
| `outflow_violation_above_cost` | Float64 | No       | Cost of total outflow above-maximum violations.                                                                              |
| `turbined_violation_cost`      | Float64 | No       | Cost of turbined flow bound violations.                                                                                      |
| `generation_violation_cost`    | Float64 | No       | Cost of generation bound violations.                                                                                         |
| `evaporation_violation_cost`   | Float64 | No       | Cost of evaporation violations.                                                                                              |
| `withdrawal_violation_cost`    | Float64 | No       | Cost of water withdrawal violations.                                                                                         |
| `inflow_penalty_cost`          | Float64 | No       | Cost of inflow non-negativity slack (numerical penalty).                                                                     |
| `generic_violation_cost`       | Float64 | No       | Cost of generic constraint violations.                                                                                       |
| `spillage_cost`                | Float64 | No       | Cost of reservoir spillage.                                                                                                  |
| `turbined_cost`                | Float64 | No       | Turbined flow penalty from the future-production hydro approximation.                                                        |
| `curtailment_cost`             | Float64 | No       | Cost of non-controllable source curtailment.                                                                                 |
| `exchange_cost`                | Float64 | No       | Transmission exchange cost component.                                                                                        |
| `pumping_cost`                 | Float64 | No       | Pumping station energy cost component.                                                                                       |

---

### `simulation/hydros/`

Hydro plant dispatch results. One row per (stage, block, hydro) triplet. 35 columns.

See [Energy Variables](../guide/energy-variables.md) for an explanation of the
five energy columns (`equivalent_productivity_mw_per_m3s` through
`stored_energy_final_mwh`).

| Column                                | Type    | Nullable | Description                                                                                                                                                                                          |
| ------------------------------------- | ------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `stage_id`                            | Int32   | No       | Stage index (0-based).                                                                                                                                                                               |
| `block_id`                            | Int32   | Yes      | Load block index. `null` for stage-level records.                                                                                                                                                    |
| `hydro_id`                            | Int32   | No       | Hydro plant ID.                                                                                                                                                                                      |
| `turbined_m3s`                        | Float64 | No       | Turbined flow in cubic metres per second (m³/s).                                                                                                                                                     |
| `spillage_m3s`                        | Float64 | No       | Spilled flow in m³/s.                                                                                                                                                                                |
| `outflow_m3s`                         | Float64 | No       | Total outflow (turbined + spilled) in m³/s.                                                                                                                                                          |
| `evaporation_m3s`                     | Float64 | Yes      | Net evaporation flow in m³/s; signed. Positive values are net evaporative loss; negative values are net rainfall input on the lake surface. `null` if evaporation is not modelled for this plant.    |
| `diverted_inflow_m3s`                 | Float64 | Yes      | Diverted inflow to this reservoir in m³/s. `null` if no diversion is configured.                                                                                                                     |
| `diverted_outflow_m3s`                | Float64 | Yes      | Diverted outflow from this reservoir in m³/s. `null` if no diversion is configured.                                                                                                                  |
| `incremental_inflow_m3s`              | Float64 | No       | Natural incremental inflow to this reservoir in m³/s (excluding upstream contributions).                                                                                                             |
| `inflow_m3s`                          | Float64 | No       | Total inflow to this reservoir in m³/s (including upstream contributions).                                                                                                                           |
| `storage_initial_hm3`                 | Float64 | No       | Reservoir storage at the start of the stage in hectare-metres cubed (hm³).                                                                                                                           |
| `storage_final_hm3`                   | Float64 | No       | Reservoir storage at the end of the stage in hm³.                                                                                                                                                    |
| `generation_mw`                       | Float64 | No       | Average power generation over the block in megawatts (MW).                                                                                                                                           |
| `generation_mwh`                      | Float64 | No       | Total energy generated over the block in megawatt-hours (MWh).                                                                                                                                       |
| `equivalent_productivity_mw_per_m3s`  | Float64 | No       | Equivalent productivity ρ_eq [MW/(m³/s)] at the reference operating point for this stage.                                                                                                            |
| `accumulated_productivity_mw_per_m3s` | Float64 | No       | Accumulated cascade productivity ρ_acum [MW/(m³/s)]: sum of ρ_eq for this plant and all downstream plants.                                                                                           |
| `incremental_inflow_energy_mw`        | Float64 | No       | Power equivalent of incremental inflow: ρ_acum × incremental_inflow_m3s [MW].                                                                                                                        |
| `stored_energy_initial_mwh`           | Float64 | No       | Energy content of usable storage at stage start: (storage_initial_hm3 − V_min) × ρ_acum × 1e6/3600 [MWh].                                                                                            |
| `stored_energy_final_mwh`             | Float64 | No       | Energy content of usable storage at stage end: (storage_final_hm3 − V_min) × ρ_acum × 1e6/3600 [MWh].                                                                                                |
| `spillage_cost`                       | Float64 | No       | Monetary cost attributed to spillage.                                                                                                                                                                |
| `water_value_per_hm3`                 | Float64 | No       | Shadow price of the reservoir water balance constraint (monetary units per hm³).                                                                                                                     |
| `storage_binding_code`                | Int8    | No       | Whether the storage bounds were binding (see `codes.json` `storage_binding` mapping).                                                                                                                |
| `operative_state_code`                | Int8    | No       | Operative state code (see `codes.json` `operative_state` mapping).                                                                                                                                   |
| `turbined_slack_m3s`                  | Float64 | No       | Turbined flow slack variable (non-negativity enforcement). Zero under normal operation.                                                                                                              |
| `outflow_slack_below_m3s`             | Float64 | No       | Outflow lower-bound slack in m³/s.                                                                                                                                                                   |
| `outflow_slack_above_m3s`             | Float64 | No       | Outflow upper-bound slack in m³/s.                                                                                                                                                                   |
| `generation_slack_mw`                 | Float64 | No       | Generation bound slack in MW.                                                                                                                                                                        |
| `storage_violation_below_hm3`         | Float64 | No       | Reservoir storage below-minimum violation in hm³. Zero under feasible operation.                                                                                                                     |
| `filling_target_violation_hm3`        | Float64 | No       | Filling target miss in hm³. Zero when the target is met.                                                                                                                                             |
| `evaporation_violation_pos_m3s`       | Float64 | No       | Slack absorbing a positive deviation of the signed evaporation flow from the linearised target in m³/s (solver chose a less-negative net flux than the model predicts). Zero under normal operation. |
| `evaporation_violation_neg_m3s`       | Float64 | No       | Slack absorbing a negative deviation of the signed evaporation flow from the linearised target in m³/s (solver chose a less-positive net flux than the model predicts). Zero under normal operation. |
| `inflow_nonnegativity_slack_m3s`      | Float64 | No       | Inflow non-negativity slack in m³/s. Zero under normal operation.                                                                                                                                    |
| `water_withdrawal_violation_pos_m3s`  | Float64 | No       | Water withdrawal over-target violation in m³/s. Zero when withdrawal is at or below target.                                                                                                          |
| `water_withdrawal_violation_neg_m3s`  | Float64 | No       | Water withdrawal under-target violation in m³/s. Zero when withdrawal is at or above target.                                                                                                         |

---

### `simulation/thermals/`

Thermal unit dispatch results. One row per (stage, block, thermal) triplet. 10 columns.

| Column                     | Type    | Nullable | Description                                                                            |
| -------------------------- | ------- | -------- | -------------------------------------------------------------------------------------- |
| `stage_id`                 | Int32   | No       | Stage index (0-based).                                                                 |
| `block_id`                 | Int32   | Yes      | Load block index. `null` for stage-level records.                                      |
| `thermal_id`               | Int32   | No       | Thermal unit ID.                                                                       |
| `generation_mw`            | Float64 | No       | Average power generation over the block in MW.                                         |
| `generation_mwh`           | Float64 | No       | Total energy generated over the block in MWh.                                          |
| `generation_cost`          | Float64 | No       | Monetary generation cost for this block.                                               |
| `is_anticipated`           | Boolean | No       | `true` if this unit is configured for anticipated dispatch.                            |
| `anticipated_committed_mw` | Float64 | Yes      | Committed capacity under anticipated dispatch in MW. `null` for non-anticipated units. |
| `anticipated_decision_mw`  | Float64 | Yes      | Dispatch decision under anticipated dispatch in MW. `null` for non-anticipated units.  |
| `operative_state_code`     | Int8    | No       | Operative state code (see `codes.json` `operative_state` mapping).                     |

---

### `simulation/exchanges/`

Transmission line flow results. One row per (stage, block, line) triplet. 11 columns.

| Column                 | Type    | Nullable | Description                                                        |
| ---------------------- | ------- | -------- | ------------------------------------------------------------------ |
| `stage_id`             | Int32   | No       | Stage index (0-based).                                             |
| `block_id`             | Int32   | Yes      | Load block index. `null` for stage-level records.                  |
| `line_id`              | Int32   | No       | Transmission line ID.                                              |
| `direct_flow_mw`       | Float64 | No       | Flow in the forward (direct) direction in MW.                      |
| `reverse_flow_mw`      | Float64 | No       | Flow in the reverse direction in MW.                               |
| `net_flow_mw`          | Float64 | No       | Net flow (direct minus reverse) in MW.                             |
| `net_flow_mwh`         | Float64 | No       | Net energy flow over the block in MWh.                             |
| `losses_mw`            | Float64 | No       | Transmission losses in MW.                                         |
| `losses_mwh`           | Float64 | No       | Transmission losses in MWh over the block.                         |
| `exchange_cost`        | Float64 | No       | Monetary cost attributed to this line's exchange.                  |
| `operative_state_code` | Int8    | No       | Operative state code (see `codes.json` `operative_state` mapping). |

---

### `simulation/buses/`

Bus load balance results. One row per (stage, block, bus) triplet. 10 columns.

| Column        | Type    | Nullable | Description                                                                                         |
| ------------- | ------- | -------- | --------------------------------------------------------------------------------------------------- |
| `stage_id`    | Int32   | No       | Stage index (0-based).                                                                              |
| `block_id`    | Int32   | Yes      | Load block index. `null` for stage-level records.                                                   |
| `bus_id`      | Int32   | No       | Bus ID.                                                                                             |
| `load_mw`     | Float64 | No       | Total load demand at this bus in MW.                                                                |
| `load_mwh`    | Float64 | No       | Total load energy demand over the block in MWh.                                                     |
| `deficit_mw`  | Float64 | No       | Unserved load (deficit) at this bus in MW. Zero under feasible dispatch.                            |
| `deficit_mwh` | Float64 | No       | Unserved load energy over the block in MWh.                                                         |
| `excess_mw`   | Float64 | No       | Excess generation at this bus in MW. Zero under feasible dispatch.                                  |
| `excess_mwh`  | Float64 | No       | Excess generation energy over the block in MWh.                                                     |
| `spot_price`  | Float64 | No       | Locational marginal price (shadow price of the power balance constraint) in monetary units per MWh. |

---

### `simulation/pumping_stations/`

Pumping station results. One row per (stage, block, pumping station) triplet. 9 columns.

| Column                   | Type    | Nullable | Description                                                        |
| ------------------------ | ------- | -------- | ------------------------------------------------------------------ |
| `stage_id`               | Int32   | No       | Stage index (0-based).                                             |
| `block_id`               | Int32   | Yes      | Load block index. `null` for stage-level records.                  |
| `pumping_station_id`     | Int32   | No       | Pumping station ID.                                                |
| `pumped_flow_m3s`        | Float64 | No       | Pumped flow rate in m³/s.                                          |
| `pumped_volume_hm3`      | Float64 | No       | Total pumped volume over the stage in hm³.                         |
| `power_consumption_mw`   | Float64 | No       | Power consumed by the pumping station in MW.                       |
| `energy_consumption_mwh` | Float64 | No       | Energy consumed over the block in MWh.                             |
| `pumping_cost`           | Float64 | No       | Monetary cost of pumping energy.                                   |
| `operative_state_code`   | Int8    | No       | Operative state code (see `codes.json` `operative_state` mapping). |

---

### `simulation/contracts/`

Energy contract results. One row per (stage, block, contract) triplet. 8 columns.

| Column                 | Type    | Nullable | Description                                                                                                                                                             |
| ---------------------- | ------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `stage_id`             | Int32   | No       | Stage index (0-based).                                                                                                                                                  |
| `block_id`             | Int32   | Yes      | Load block index. `null` for stage-level records.                                                                                                                       |
| `contract_id`          | Int32   | No       | Contract ID.                                                                                                                                                            |
| `power_mw`             | Float64 | No       | Contracted power in MW, non-negative for both import and export contracts. Direction is carried by the contract type and the price sign, not by the sign of this value. |
| `energy_mwh`           | Float64 | No       | Contracted energy over the block in MWh.                                                                                                                                |
| `price_per_mwh`        | Float64 | No       | Contract price in monetary units per MWh.                                                                                                                               |
| `total_cost`           | Float64 | No       | Total contract cost for this block: positive for imports (cost), negative for exports (revenue).                                                                        |
| `operative_state_code` | Int8    | No       | Operative state code (see `codes.json` `operative_state` mapping); always `1` for contracts (a dormant stage emits a zero-`power_mw` row, not a distinct code).         |

---

### `simulation/non_controllables/`

Non-controllable source results (wind, solar, run-of-river hydro without
storage, etc.). One row per (stage, block, non-controllable) triplet. 10 columns.

| Column                 | Type    | Nullable | Description                                                                   |
| ---------------------- | ------- | -------- | ----------------------------------------------------------------------------- |
| `stage_id`             | Int32   | No       | Stage index (0-based).                                                        |
| `block_id`             | Int32   | Yes      | Load block index. `null` for stage-level records.                             |
| `non_controllable_id`  | Int32   | No       | Non-controllable source ID.                                                   |
| `generation_mw`        | Float64 | No       | Actual generation dispatched in MW.                                           |
| `generation_mwh`       | Float64 | No       | Actual energy generated over the block in MWh.                                |
| `available_mw`         | Float64 | No       | Maximum available generation in MW (before curtailment).                      |
| `curtailment_mw`       | Float64 | No       | Generation curtailed in MW. Zero when all available generation is dispatched. |
| `curtailment_mwh`      | Float64 | No       | Curtailed energy over the block in MWh.                                       |
| `curtailment_cost`     | Float64 | No       | Monetary cost attributed to curtailment.                                      |
| `operative_state_code` | Int8    | No       | Operative state code (see `codes.json` `operative_state` mapping).            |

---

### `simulation/inflow_lags/`

Autoregressive inflow lag state variables. One row per (stage, hydro, lag)
triplet. No block dimension — inflow lags are stage-level state variables.
4 columns. All columns are non-nullable.

| Column       | Type    | Nullable | Description                                                               |
| ------------ | ------- | -------- | ------------------------------------------------------------------------- |
| `stage_id`   | Int32   | No       | Stage index (0-based).                                                    |
| `hydro_id`   | Int32   | No       | Hydro plant ID.                                                           |
| `lag_index`  | Int32   | No       | Autoregressive lag order (1-based). Lag 1 is the previous stage's inflow. |
| `inflow_m3s` | Float64 | No       | Inflow value for this lag in m³/s.                                        |

---

### `simulation/violations/generic/`

Generic user-defined constraint violations. One row per (stage, block,
constraint) triplet where a violation occurred. 5 columns.

| Column          | Type    | Nullable | Description                                                                    |
| --------------- | ------- | -------- | ------------------------------------------------------------------------------ |
| `stage_id`      | Int32   | No       | Stage index (0-based).                                                         |
| `block_id`      | Int32   | Yes      | Load block index. `null` for stage-level constraints.                          |
| `constraint_id` | Int32   | No       | Constraint ID as defined in the case input files.                              |
| `slack_value`   | Float64 | No       | Violation magnitude in the constraint's natural unit. Zero means no violation. |
| `slack_cost`    | Float64 | No       | Monetary cost attributed to this violation.                                    |

---

## Hive Partitioning

All simulation Parquet output uses Hive partitioning: results for each scenario
are stored in a directory named `scenario_id=NNNN/` containing a single
`data.parquet` file. The `scenario_id` column is encoded in the directory name,
not as a column inside the Parquet file.

All major columnar data tools understand this layout and can read an entire
`simulation/<entity>/` directory as a single table with an automatically
inferred `scenario_id` column:

```python
# Polars — reads all scenarios at once, infers scenario_id from directory names
import polars as pl

df = pl.read_parquet("results/simulation/costs/")
print(df.head())
```

```python
# Pandas with PyArrow backend
import pandas as pd

df = pd.read_parquet("results/simulation/costs/")
```

```sql
-- DuckDB — filter to a specific scenario at the storage layer
SELECT * FROM read_parquet('results/simulation/costs/**/*.parquet')
WHERE scenario_id = 0;
```

```r
# R with the arrow package
library(arrow)
ds <- open_dataset("results/simulation/costs/")
dplyr::collect(dplyr::filter(ds, scenario_id == 0))
```

Scenario IDs are zero-based integers. The total number of scenarios is
documented in `simulation/metadata.json` under `scenarios.total`.

---

## Metadata Files

Both `training/metadata.json` and `simulation/metadata.json` use an atomic
write protocol:

1. Serialize JSON to a temporary `.json.tmp` sibling file.
2. Atomically rename the `.tmp` file to the target path.

This ensures consumers never observe a partial file. If a metadata file exists,
it contains a complete, valid JSON document. If a run is interrupted before the
final write, the `.tmp` sibling may remain, but the target file reflects the
last successfully completed write.

The `status` field is always the first indicator to check:

| Status       | Meaning                                                                |
| ------------ | ---------------------------------------------------------------------- |
| `"complete"` | The run finished normally. All output files are present.               |
| `"partial"`  | Not all scenarios completed without error. (Simulation metadata only.) |

`cobre report` reads both metadata files and prints a combined JSON summary to
stdout. Use it in CI pipelines or shell scripts to inspect outcomes without
parsing JSON directly:

```bash
# Extract the termination reason
cobre report results/ | jq '.training.convergence.termination_reason'

# Fail a CI job if the run did not complete
status=$(cobre report results/ | jq -r '.status')
[ "$status" = "complete" ] || exit 1
```

---

## Hydro Model Artifacts

The `hydro_models/` directory is written when at least one of the following
conditions holds: any hydro plant uses `fpha_config.source: "computed"` in
`system/hydro_production_models.json`, any hydro plant has an evaporation model,
or `exports.fpha_deviation_points` is `true`. The directory is omitted when none
of these conditions are met.

### `hydro_models/fpha_hyperplanes.parquet`

Fitted FPHA hyperplane coefficients for all hydros that used `source: "computed"`
in the current run. The schema is identical to the input file
`system/fpha_hyperplanes.parquet`: 11 columns, all with the same names, types,
and nullability.

| Column            | Type   | Nullable | Description                                               |
| ----------------- | ------ | -------- | --------------------------------------------------------- |
| `hydro_id`        | INT32  | No       | Hydro plant ID                                            |
| `stage_id`        | INT32  | Yes      | Stage the plane applies to. `null` = valid for all stages |
| `plane_id`        | INT32  | No       | Plane index within this hydro (and stage)                 |
| `gamma_0`         | DOUBLE | No       | Intercept coefficient (MW), unscaled                      |
| `gamma_v`         | DOUBLE | No       | Volume coefficient (MW/hm³)                               |
| `gamma_q`         | DOUBLE | No       | Turbined flow coefficient (MW per m³/s)                   |
| `gamma_s`         | DOUBLE | No       | Spillage coefficient (MW per m³/s)                        |
| `kappa`           | DOUBLE | Yes      | Correction factor. Defaults to `1.0` when absent or null. |
| `valid_v_min_hm3` | DOUBLE | Yes      | Volume range minimum where this plane is valid (hm³)      |
| `valid_v_max_hm3` | DOUBLE | Yes      | Volume range maximum where this plane is valid (hm³)      |
| `valid_q_max_m3s` | DOUBLE | Yes      | Maximum turbined flow where this plane is valid (m³/s)    |

The file is written atomically (via a `.tmp` rename) and uses the same
`(hydro_id, stage_id, plane_id)`-sorted row order as the input schema. It can
be used directly as a future `source: "precomputed"` input by copying it to
`system/fpha_hyperplanes.parquet`.

See [Case Format Reference — `system/fpha_hyperplanes.parquet`](./case-format.md#systemfpha_hyperplanesparquet)
for the full column definitions and validity constraints.

### `hydro_models/evaporation_models.parquet`

Written when any hydro plant has an evaporation model. Contains the fitted
evaporation coefficients for all plants that have evaporation, keyed by
`(hydro_id, stage_id)`. Rows with `stage_id = null` are per-hydro defaults.

Six columns:

| Column                     | Type   | Nullable | Description                                                       |
| -------------------------- | ------ | -------- | ----------------------------------------------------------------- |
| `hydro_id`                 | INT32  | No       | Hydro plant identifier                                            |
| `stage_id`                 | INT32  | Yes      | Stage; `null` = per-hydro default applicable to all stages        |
| `intercept_m3s`            | DOUBLE | No       | Evaporation intercept coefficient (m³/s)                          |
| `volume_slope_m3s_per_hm3` | DOUBLE | No       | Volume-dependent slope coefficient (m³/s per hm³)                 |
| `reference_volume_hm3`     | DOUBLE | No       | Reference volume used for linearisation (hm³)                     |
| `source`                   | STRING | No       | Derivation label (e.g. `"default_midpoint"` or `"user_supplied"`) |

### `hydro_models/fpha_deviation_points.parquet`

Written only when `exports.fpha_deviation_points: true` is set in `config.json`.
Contains one row per `(hydro, stage, V, Q)` grid point at spillage = 0, recording
how closely the fitted FPHA plane set approximates the exact production function at
each sample point. Opt-in because it can be large (one row per grid-point combination
for each computed-FPHA plant and stage).

Eight columns:

| Column        | Type   | Nullable | Description                                                                                                   |
| ------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------- |
| `hydro_id`    | INT32  | No       | Hydro plant identifier                                                                                        |
| `stage_id`    | INT32  | Yes      | Stage; `null` when the fit applies to all stages                                                              |
| `v`           | DOUBLE | No       | Volume sample point (hm³)                                                                                     |
| `q`           | DOUBLE | No       | Turbined-flow sample point (m³/s)                                                                             |
| `fph_exact`   | DOUBLE | No       | Exact production function value at this (V, Q) point (MW)                                                     |
| `fpha_fitted` | DOUBLE | No       | Fitted FPHA approximation at this (V, Q) point (MW)                                                           |
| `deviation`   | DOUBLE | No       | Signed residual `fpha_fitted − fph_exact` (MW); positive = fitted cap above the exact surface                 |
| `relative`    | DOUBLE | No       | `\|deviation\|` relative to the grid's peak exact generation (dimensionless, ≥ 0); `0` when the grid peak ≤ 0 |

The values are a pure function of geometry and config — the file is reproducible
when emitted and never enters the parity hash.

---

## Stochastic Artifacts

When `exports.stochastic: true` is set in `config.json`, Cobre writes the
stochastic preprocessing artifacts to `output/stochastic/` before training
begins.

The directory is not written when the config field is not set. Export is off
by default.

### Exported files

| File path                                   | Export condition         | Schema source                                            |
| ------------------------------------------- | ------------------------ | -------------------------------------------------------- |
| `stochastic/inflow_seasonal_stats.parquet`  | Estimation was performed | Same as input `scenarios/inflow_seasonal_stats.parquet`  |
| `stochastic/inflow_ar_coefficients.parquet` | Estimation was performed | Same as input `scenarios/inflow_ar_coefficients.parquet` |
| `stochastic/correlation.json`               | Always                   | Same as input `scenarios/correlation.json`               |
| `stochastic/fitting_report.json`            | Estimation was performed | JSON diagnostic report (see below)                       |
| `stochastic/noise_openings.parquet`         | Always                   | Same schema as `scenarios/noise_openings.parquet`        |
| `stochastic/load_seasonal_stats.parquet`    | Load buses exist         | Same as input `scenarios/load_seasonal_stats.parquet`    |

"Estimation was performed" means the user did not supply the corresponding
scenario file directly; Cobre derived it from `inflow_history.parquet`.

### `stochastic/noise_openings.parquet`

The opening tree used during the training run, written in the same schema as
the input file `scenarios/noise_openings.parquet`. See the
[Case Format Reference](./case-format.md#scenariosnoise_openingsparquet) for
the 4-column schema (`stage_id`, `opening_index`, `entity_index`, `value`).

### `stochastic/fitting_report.json`

A JSON diagnostic report for the PAR model fitting. This file is written only
when Cobre performed estimation from `inflow_history.parquet`.

**Structure:**

```json
{
  "hydros": {
    "<hydro_id>": {
      "selected_order": 3,
      "aic_scores": [12.4, 11.1, 10.8, 11.3],
      "coefficients": [[0.42, -0.11, 0.07]]
    }
  }
}
```

| Field            | Type         | Description                                                                      |
| ---------------- | ------------ | -------------------------------------------------------------------------------- |
| `selected_order` | integer      | AIC-selected AR order for this hydro plant                                       |
| `aic_scores`     | number array | AIC score for each candidate order; `aic_scores[i]` is the score for order `i+1` |
| `coefficients`   | nested array | One row per season; each row contains the AR coefficients for that season        |

This file is diagnostic only. It is not consumed as input on subsequent runs.

### Round-trip workflow

Every exported Parquet and JSON file uses the exact same column names, types,
and layout as the corresponding input file. To replay a run with identical
stochastic context:

```bash
# Run with exports.stochastic: true in config.json
cobre run my_case

# Copy exported artifacts to scenarios/
cp -r my_case/output/stochastic/* my_case/scenarios/

# Re-run: the loader finds the files already present and skips estimation
cobre run my_case
```

The re-run produces bit-for-bit identical stochastic artifacts because the
round-trip eliminates the estimation step. The opening tree is loaded directly
from `scenarios/noise_openings.parquet` instead of being regenerated.

See [Exporting Stochastic Artifacts](../guide/running-studies.md#exporting-stochastic-artifacts)
in the Running Studies guide for the end-to-end workflow.
