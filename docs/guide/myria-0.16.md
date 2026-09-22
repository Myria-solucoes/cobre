# Myria runtime based on Cobre 0.16

The `v0.16.0-myria.1` runtime combines upstream `v0.16.0` with the Myria
optimized-training branch and the worker-affinity implementation. It is an
independent runtime: use version-specific installation and output directories.
Publishing this build does not replace a Myria installation or its recommended
runtime. CLI and Python packages report the upstream version `0.16.0`; identify
the Myria build by its immutable release tag, source commit and artifact SHA-256.

## Included changes

- The expectation/CVaR mixture preserves each scenario's expectation floor in
  both scalar risk evaluation and backward cuts. For equally likely costs 0 and
  100, alpha 0.15 and lambda 0.4, the value is 70, not 100. Retrain policies made
  with the incorrect mixture; existing cuts cannot be repaired by loading them.
- Dynamic cut selection uses partial sorting and avoids unnecessary coefficient
  copies. Its final fallback loads every eligible missing row. Adaptive separation
  batches remain optional.
- Backward workers reuse frozen LP matrices after a solver cold reset. Independent
  work units retain independent solver history.
- Optional complete-state deduplication, audited backward-point selection,
  progressive forward populations and fixed cross-point basis groups are retained.
- Periodic atomic checkpoints work in the CLI and both Python training paths
  (including iteration callbacks). Completed-training CLI resume and the separate
  `lb_stability_v1` diagnostic remain available. The diagnostic is not a stopping rule.
  Resuming without saved solver bases can change rounding relative to uninterrupted
  training; identical resumed runs retain the reproducibility contract.
- CLI `--cpu-bind none|core|numa` and Python `cpu_bind` expose the same optional
  Linux worker-affinity policy, inherited-CPU-set restrictions and placement metadata.
  `none` is the default. On unsupported platforms explicit binding is rejected.
- The automatic PAR stationarity repair used by Myria remains restricted to
  history-estimated models and reports every reduction.

See [optimized training](optimized-training.md) for configuration examples and
historical measurements. Those measurements used the 0.15 preview and are not
performance claims for this 0.16 build. Approximate controls remain experimental;
fixed population, all backward points and independent chains are the defaults.

## Upstream migration

This build includes the 0.16 productivity and stored-energy semantics. Review the
[productivity contract](../design/hydro-productivity-and-stored-energy.md) before
comparing stored-energy outputs with earlier versions or NEWAVE.

Remove the retired `policy.boundary.source_stage` key from input configuration;
boundary selection now uses calendar dates. The upstream format-version-2
checkpoint requirement remains in force. Keep old runtime/policy pairs together;
a new build does not migrate a checkpoint or make an old policy CVaR-correct.

Python callers control simulation through `simulation.enabled`; the retired
`skip_simulation` argument is not restored. `threads=0` remains invalid.

## Artifact verification

The release workflow runs the native affinity and CLI checkpoint tests, then
exercises the built CLI and installed Python wheel on copied 4REE cases with
mixed CVaR, progressive population, audited point selection, grouped bases,
and both frozen and dynamic cut selection. Linux x86_64 and ARM64 artifacts are
validated independently. All outputs are temporary and separate from an installed
Myria runtime. Release provenance and test results accompany the build report.

See the [integration validation report](myria-0.16-validation.md) for the local
results, reproduced upstream failures and remaining operational checks.
