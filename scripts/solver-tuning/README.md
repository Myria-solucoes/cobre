# Solver / accelerator tuning runner

A SLURM-array harness to sweep LP-solver parameters and performance accelerators
(warm-start mode, cut selection) on a production-scale benchmark case, **one full
`cobre run` per cell**, single node, 96 threads, no MPI.

Design rationale lives in `docs/design/solver-parameter-tuning.md` and
`docs/design/accelerator-effectiveness-research.md`.

## How a cell is parameterized

- **Solver params + warm-start** → `COBRE_TUNE_*` environment variables (no
  config edit). See the env schema in `docs/design/solver-parameter-tuning.md`.
- **Cut selection** → `training.cut_selection` in `config.json`; since
  `cobre run` always reads `<case>/config.json` (no `--config` flag), each
  selection method gets its own full case copy (`prep_cases.sh`).

The campaign is **staged, one backend at a time** (build that backend's binary
first):

- **Stage 1** — OFAT over solver params (warm-start `full`, cut selection
  `none`). The `baseline` cell carries no overrides and is the correctness
  reference.
- **Stage 2** — accelerator matrix (warm-start `{full,core,off}` × cut selection
  `{none,level1,dominated}`) with Stage-1's winning solver env fixed.

A **manual gate** sits between the stages: you inspect Stage-1 results and choose
the winning solver env.

## Files

| File              | Role                                                                                  |
| ----------------- | ------------------------------------------------------------------------------------- |
| `grid.py`         | Emit the manifest (one JSONL line per `cell × repeat`) for a backend/stage            |
| `patch_config.py` | Set `training.cut_selection` for one method in a case's `config.json`                 |
| `prep_cases.sh`   | Copy the base case once per cut-selection method and patch each                       |
| `run_cell.py`     | Run one manifest line: set env, `cobre run`, write `tune_params.json` + `result.json` |
| `sweep.sbatch`    | SLURM array wrapper (1 node, 96 cpus, `--exclusive`) calling `run_cell.py`            |
| `aggregate.py`    | Collect cells → `results.csv` + ranked summary + correctness gate + suggested winner  |

## Runbook (example: HiGHS)

```bash
cd scripts/solver-tuning
BASE=/path/to/production-case          # your 10–20 iter, 96-fwd, 64-stage case
BIN=/path/to/cobre                     # built with the HiGHS backend (default features)

# 1. Per-method case copies (none/level1/dominated)
./prep_cases.sh "$BASE" cases/highs

# 2. Stage 1 manifest (OFAT solver params)
mkdir -p manifests
python3 grid.py --backend highs --stage 1 --reps 1 > manifests/highs.s1.jsonl
N=$(wc -l < manifests/highs.s1.jsonl)

# 3. Submit the array (one node per cell, 96 threads)
TUNE_MANIFEST=manifests/highs.s1.jsonl TUNE_CASES=cases/highs \
TUNE_RUNS=runs TUNE_BIN="$BIN" \
  sbatch --array=0-$((N-1)) sweep.sbatch

# 4. Aggregate + manual gate
python3 aggregate.py --runs runs --backend highs --stage 1
#  -> runs/highs/s1/results.csv, a ranked table, and runs/highs/s1/suggested_winner.json
#  Review; copy/edit the chosen env into winner.json.

# 5. Stage 2 manifest (accelerator matrix on the winning solver env)
python3 grid.py --backend highs --stage 2 --winner winner.json --reps 1 \
  > manifests/highs.s2.jsonl
N=$(wc -l < manifests/highs.s2.jsonl)
TUNE_MANIFEST=manifests/highs.s2.jsonl TUNE_CASES=cases/highs \
TUNE_RUNS=runs TUNE_BIN="$BIN" \
  sbatch --array=0-$((N-1)) sweep.sbatch

# 6. Final comparison
python3 aggregate.py --runs runs --backend highs --stage 2
```

For CLP: rebuild the binary `--no-default-features --features clp`, then repeat
with `--backend clp` and `TUNE_CASES=cases/clp`.

## Outputs per cell

```
runs/<backend>/s<stage>/<cell>__rep<r>/
  ├── tune_params.json   # exact params used: env, cut_sel, threads, binary, git commit, host, timestamps, SLURM ids
  ├── result.json        # parsed: backward/forward_solve_seconds, duration_seconds, final_lower_bound, retried/failed, exit
  └── output/            # cobre's training/ + simulation/ outputs
```

Cells are **resumable**: a cell with an existing `result.json` is skipped, so a
re-submitted array only fills gaps.

## Notes / knobs

- `--reps N` (default 1) emits N repeats per cell; run finalists with `--reps 3`
  and take the **min** (least-contended) — full runs are deterministic in
  _result_, so repeats only quantify 96-thread timing noise.
- `aggregate.py --ref-tol` is the relative final-LB tolerance for the correctness
  gate vs the `baseline` cell (different solver/accelerator configs may reach
  alternate optima → small LB drift; tune the tolerance from observed drift).
- `TUNE_THREADS` overrides 96; `OMP_NUM_THREADS` is pinned to match.
- **Confirm before trusting:** the HiGHS `simplex_scale_strategy` /
  `simplex_price_strategy` enum values used in `grid.py` (the ideas doc and a
  code comment disagreed on the equilibration index); and `domination_epsilon`
  in `patch_config.py` (placeholder).
- The `off`/`core` warm-start modes still _capture_ unused bases (small fixed
  overhead that slightly flatters `full`).
