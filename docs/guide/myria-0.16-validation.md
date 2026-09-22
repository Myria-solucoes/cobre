# Myria 0.16 integration validation

Validation date: 2026-09-22. Candidate: `codex/myria-0.16-optimized`.
This report records integration correctness, not a new speedup measurement.

## Provenance

- Upstream `v0.16.0`: `3d4765a2b048e6efef8d4641632f44e1d1ca2ffd`.
- Myria optimized training: `5a52fbbc` (includes the CVaR correction,
  checkpoint/resume support and the optimization series through `3e3bdbc4`).
- Worker affinity: public PR #64, `3d3cef3be72a2112a9152ccffc5db846db361763`.
- Merge commits: `88b4cef4` and `2740d9c1`. Subsequent integration fixes share
  periodic checkpoint writing between CLI and both Python event paths.

The release tag identifies the final source commit. The release includes SHA-256
checksums for both CLI archives and both Python wheels. Existing Myria runtimes
and their pins are not replaced by publishing this candidate.

## Local checks (macOS ARM64)

- Workspace/all-target checks, Rust formatting and workspace/Python Clippy with
  warnings denied passed.
- CLI/Python output parity: 20 shared writer functions, no mismatch.
- Unit suites: 5,480 passed across comm, core, IO, SDDP, solver and stochastic;
  one SDDP failure reproduced unchanged on upstream (see below), one ignored.
- CLI tests: 244 passed, including checkpoint/resume and output parity.
- Python study/run/IO/affinity suites: 44 passed, two Linux-only tests skipped.
  Both new periodic-checkpoint tests resumed an iteration-2 generation to
  iteration 4 and reproduced the uninterrupted final lower bound, with and
  without an iteration callback.
- Integration: anticipated core 29 passed/one ignored; deterministic 114 passed
  and one upstream-reproduced failure; boundary format three passed; MPI wire
  40 passed; right-boundary output six passed. Wire tests exercise communication
  shape and invariance; they are not a deployment on a multi-node MPI cluster.
- The built CLI and installed Python module passed `verify_runtime.py` on
  copied 4REE cases: mixed CVaR, progressive trajectories, audited point selection,
  deduplication, grouped bases, frozen and dynamic cut selection, and periodic
  checkpoints. CLI (two threads) and Python (one/two threads) produced the same
  final lower bound: 81525499777.04082 (frozen), 81525499777.04083 (dynamic).

## Upstream failures reproduced in an isolated checkout

Both failures below also occur on **unmodified upstream v0.16.0**, using a separate
build directory on the same Mac. No assertion or numerical tolerance was relaxed.

1. `computed_source_end_to_end_produces_valid_fpha_planes`: a generated volume
   coefficient is `-7.538531311133364e-19`, failing a strict nonnegative assertion.
2. `test_fpha_variable_head_case_bit_exact`: final lower-bound bits are
   `0x4148da6e907f6e5b`, while the fixed golden expects `0x4148da6e907f6e5c` (one ULP).
   The behavioral FPHA test passes. This is a mismatch against the fixed golden,
   not observed run-to-run nondeterminism.

The full workspace is therefore not reported as entirely green on this platform.

## Linux release gate and operational boundary

The release workflow must pass native affinity, risk-measure, progressive-population,
CLI/checkpoint and installed-wheel integration checks independently on x86_64 and
ARM64 before publishing any release. This report alone is not evidence that a
particular workflow run passed; consult the release's linked Actions run.

Testing on `myria-manager` awaits SSH authentication. This build has not been
deployed there, and prior 0.15 benchmark speedups have not been remeasured for 0.16.
The matching wheel must accompany the CLI when integrating a worker; selecting a
0.16 executable while retaining a 0.15 Python validator is not a supported setup.
