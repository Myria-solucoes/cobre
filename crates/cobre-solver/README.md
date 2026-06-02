# cobre-solver

LP/MIP solver abstraction for the [Cobre](https://github.com/cobre-rs/cobre)
power systems ecosystem.

Defines a backend-agnostic `SolverInterface` trait for LP and MIP problem
construction, solving, dual extraction, and basis warm-starting. The default
backend is [HiGHS](https://highs.dev), a production-grade open-source solver
well-suited to the iterative LP workloads of power system optimization. The
crate includes a 12-level retry escalation strategy for numerically difficult
LPs: when HiGHS returns an infeasible or numerically unstable status, the
solver retries with progressively more aggressive scaling, presolve, and
simplex strategy options before propagating failure. An optional CLP backend
exists behind the `clp` feature (off by default); it implements the same
`SolverInterface` and is conformance-validated as a drop-in for HiGHS.

## When to Use

Depend on `cobre-solver` directly when you are writing an optimization
algorithm that needs to build and solve LP subproblems and you want
backend-portability without coupling to HiGHS internals. If you only need to
run the full SDDP pipeline, depend on `cobre-sddp` instead, which manages the
solver lifecycle for you.

## Features

| Feature        | Default | Description                                                                                                                             |
| -------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `highs`        | on      | Gates `HighsSolver` and `HighsProfile`. On by default.                                                                                  |
| `clp`          | off     | Gates `ClpSolver`, `ClpProfile`, and `clp_version`. Additive — `highs` and `clp` may be enabled together. See build requirements below. |
| `test-support` | off     | Exposes FFI option-setting helpers for integration tests. Must not be enabled in production builds.                                     |

### `clp` build requirements

The `clp` feature builds CLP and CoinUtils from vendored source. Before
enabling it, initialize the submodules:

```
git submodule update --init --recursive
```

This fetches the Clp (`releases/1.17.11`) and CoinUtils (`releases/2.11.13`)
sources into `crates/cobre-solver/vendor/`.

The first build with `--features clp` runs a CoinUtils + Clp cmake superbuild
(approximately 150 C++ translation units), which takes several minutes. The
cmake output directory is cached across rebuilds, so subsequent builds are
fast.

### CI behavior

CI jobs (`check`, `test`, `clippy`, `docs`, `coverage`) all run with
`--all-features` and check out submodules recursively, so the CLP backend is
built and its test suite is exercised in every CI run. Expect the first CI
build after a cache miss to be slower than a HiGHS-only build due to the C++
superbuild.

### Per-solver test invocations

Each backend's test suite can be run in isolation, building only that solver:

```
# HiGHS backend (default features)
cargo test -p cobre-solver --features highs

# CLP backend only (HiGHS excluded)
cargo test -p cobre-solver --no-default-features --features clp
```

Both invocations compile, run, and lint clean on their own: the test suite is
backend-agnostic where possible, and each backend's tests are gated on its
feature so neither build pulls in the other solver. The CLP-only build always
includes at least one runnable end-to-end integration test.

## Key Types

- **`SolverInterface`** — the core trait every backend must implement; defines
  problem construction, solve, and dual/basis extraction methods
- **`HighsSolver`** — the default HiGHS backend; feature-gated behind `highs`
  (enabled by default)
- **`ClpSolver`** — the optional CLP/CoinUtils backend; feature-gated behind
  `clp` (off by default)
- **`Basis`** — LP basis snapshot for warm-starting a subsequent solve on a
  structurally related problem
- **`LpSolution`** — solved LP result carrying primal values, duals, and
  objective value
- **`SolverStatistics`** — per-solve diagnostics including iteration count,
  wall-clock time, basis rejections, and retry escalation level reached
- **`StageTemplate`** — pre-built LP structure for a single time stage,
  cloned and patched each iteration to avoid repeated matrix assembly

## Links

| Resource   | URL                                                        |
| ---------- | ---------------------------------------------------------- |
| Book       | <https://cobre-rs.github.io/cobre/crates/solver.html>      |
| API Docs   | <https://docs.rs/cobre-solver/latest/cobre_solver/>        |
| Repository | <https://github.com/cobre-rs/cobre>                        |
| CHANGELOG  | <https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md> |

## Status

**Alpha** — API is functional but not yet stable.

## License

Apache-2.0
