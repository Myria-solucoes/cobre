# Myria runtime based on upstream develop

This runtime starts from upstream commit
`f8ae8f88599ab900c124b46ec0d652dc72620cd3`, without replacing the upstream
cut-selection, simulation, checkpoint or solver implementation. It retains
regularization of automatically estimated PAR models, reported as
`StationarityRegularized`, and recurring-season historical coverage.
User-supplied AR coefficients retain the upstream rejection rules.

Enumerated training reuses the forward basis captured for the same node in the
preceding iteration. An explicit `training.solver.forward.use_warm_start`
setting takes precedence. Sampled training and simulation retain the upstream
defaults. Reuse can change valid optimal vertices, cuts and convergence paths;
performance comparisons must use the same precision target. HiGHS statistics
include the work of every retry, even when the solve ultimately fails.

Outputs and policies identify `cobre-myria` / `0.18.0-myria.3`. The Python ABI
remains 0.18.0, but policies use native format 4. Reconversion is required
when switching from the released 0.18 or earlier Myria runtimes. The matching
CLI and wheel are built and checked on native x86_64 and ARM64 runners.

The earlier fork's approximate point selection, progressive forward population,
custom scheduling and CPU-affinity options are not ported. The current upstream
owns cut selection and exact dynamic separation; do not copy those old options
into new cases. No performance advantage is claimed without measurement.

```sh
cobre validate case --output output
cobre run case --output output --threads 4
python scripts/benchmarks/verify_runtime_develop.py --cli target/release/cobre --case examples/1dtoy
```

A source build alone is not a numerical validation. Real NEWAVE and DECOMP
integration evidence belongs to the consuming Myria revision.
