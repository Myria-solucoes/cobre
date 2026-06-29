# Environment variables are not a Cobre input channel — policy & migration

> **Status**: Policy decision + migration plan. No code change applied yet.
> Cobre takes input through exactly two channels — declarative config/data files
> and `cobre-cli` arguments. Environment variables are **not** an accepted
> configuration or input mechanism. This document fixes that as a hard rule,
> adjudicates every environment read in the workspace against it, and lists what
> must be migrated or removed. Scope: the whole Cargo workspace, current branch
> state.

## 1. The rule (inflexible)

Cobre is configured in two ways and no others:

1. **Config / data files** parsed by `cobre-io`.
2. **CLI arguments** to `cobre-cli`.

**Environment variables are not a third channel.** They are ambient, invisible at
the call site, unvalidated by the schema layer, and undiscoverable from `--help` —
the opposite of the explicit, file- and flag-driven configuration Cobre commits
to. No `COBRE_*` variable may exist. OS/terminal conventions (`NO_COLOR`,
`COLUMNS`, `HOSTNAME`) and third-party tool overrides (`FLATC`) are likewise not
accepted: the same capability is delivered through a CLI flag or a proper API.

The **only** permitted environment reads are the ones Rust and Cargo themselves
mandate, and only at build or compile time:

- Cargo build-script inputs in `build.rs` (`CARGO_MANIFEST_DIR`, `OUT_DIR`,
  `CARGO_CFG_TARGET_*`, `CARGO_FEATURE_*`).
- The compile-time `env!("CARGO_PKG_VERSION")` / `env!("CARGO_MANIFEST_DIR")`
  macros.

These are not "Cobre inputs" — they are how Cargo communicates with a build
script and how a constant is baked at compile time. There is no alternative
mechanism for them, and a user cannot use them to reconfigure a run.

Everything else is rejected. The test harness is **not** exempt: a custom
environment variable read by a test (e.g. a golden-file regen switch) violates the
rule just as a runtime one does and must use an explicit, non-ambient mechanism.

## 2. Decision table

Adjudication of every environment read in the workspace against §1.

| Variable                                                                       | Where                                                        | Verdict              | Replacement                                                                      |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------ | -------------------- | -------------------------------------------------------------------------------- |
| `CARGO_MANIFEST_DIR`, `OUT_DIR`, `CARGO_CFG_TARGET_OS`, `CARGO_CFG_TARGET_ENV` | `*/build.rs`                                                 | **Accept**           | — (Cargo build-script interface)                                                 |
| `CARGO_FEATURE_HIGHS`, `CARGO_FEATURE_CLP`                                     | `cobre-solver/build.rs`                                      | **Accept**           | — (Cargo's documented per-feature build-script detection; see §3)                |
| `env!("CARGO_PKG_VERSION")`                                                    | `cobre-io`, `cobre-sddp` writers                             | **Accept**           | — (compile-time version stamping)                                                |
| `env!("CARGO_MANIFEST_DIR")`                                                   | `*/tests`, a few `#[cfg(test)]`                              | **Accept**           | — (compile-time fixture-path constant)                                           |
| `COBRE_W1_DIAG`                                                                | `cobre-sddp/src/stochastic/noise_key_diag.rs`                | **Reject — remove**  | delete (vestigial; §4.1)                                                         |
| `COBRE_THREADS`                                                                | `cobre-cli/src/commands/run/setup.rs`                        | **Reject — migrate** | `--threads` (already exists); drop the env fallback                              |
| `COBRE_COLOR`, `FORCE_COLOR`, `NO_COLOR`                                       | `cobre-cli/src/main.rs`, `banner.rs`                         | **Reject — migrate** | `--color {auto,always,never}` flag                                               |
| `COLUMNS`                                                                      | `cobre-cli/src/progress.rs`                                  | **Reject — migrate** | query the terminal directly (e.g. `terminal_size`); no env read                  |
| `COBRE_COMM_BACKEND`                                                           | `cobre-comm/src/factory.rs`                                  | **Reject — migrate** | `--comm-backend` CLI arg / config field                                          |
| MPI launch vars (`OMPI_*`, `PMI_*`, …)                                         | `cobre-comm/src/factory.rs`                                  | **Reject — remove**  | drop auto-detection; require explicit `--comm-backend mpi` (§5)                  |
| `HOSTNAME`                                                                     | `cobre-comm/src/local.rs`, `cobre-io/src/output/manifest.rs` | **Reject — migrate** | a hostname syscall/crate (e.g. `gethostname`), not the env var                   |
| `FLATC`                                                                        | `cobre-io/tests/flatbuffers_schema_conformance.rs`           | **Reject — migrate** | resolve `flatc` from `PATH`; drop the custom override                            |
| `COBRE_PARITY_REGEN`                                                           | `cobre-sddp/tests/parity_hash_*.rs`                          | **Reject — migrate** | an explicit regen entry point (§4.8), not an env-gated branch in the assert test |

## 3. Accepted reads — why they are on the right side of the line

Each accepted read is Cargo/Rust machinery, not a Cobre configuration surface:

- **`build.rs` Cargo vars** (`CARGO_MANIFEST_DIR`, `OUT_DIR`, `CARGO_CFG_TARGET_*`).
  Cargo sets these for every build script; they are the script's only way to learn
  its own location, output directory, and target triple.
- **`CARGO_FEATURE_HIGHS` / `CARGO_FEATURE_CLP`.** Listed for scrutiny, and kept:
  reading `CARGO_FEATURE_<NAME>` is the **documented and only** way for a build
  script to branch on which Cargo feature is active (a build script is compiled
  without the crate's own features, so `cfg!(feature = …)` is unavailable to it).
  It is Cargo's interface, evaluated once at build time, and a user cannot set it
  to reconfigure a run — feature selection happens through `--features` / the
  manifest, which are themselves explicit. This is "standard from Cargo" under §1.
- **`env!("CARGO_PKG_VERSION")` / `env!("CARGO_MANIFEST_DIR")`.** Compile-time
  macros that bake a constant into the binary; not a runtime environment read.

If a future change can replace any of these with a non-environment mechanism
without fighting Cargo, it should — but none has a practical alternative today.

## 4. Rejected reads — migration detail

### 4.1 `COBRE_W1_DIAG` → remove

`cobre-sddp/src/stochastic/noise_key_diag.rs` is a backward-pass opening-ordering
diagnostic gated on this variable; its own module doc calls it "Throwaway,
env-gated." It is **vestigial**: per `docs/design/backward-opening-ordering.md` the
question it was built to measure is settled — the reorder ships default-on
(descending noise-key order) and its configuration options were removed after the
A/B measurement confirmed the gain. Remove the module, the `NoiseKeyDiag` type,
the `noise_key_diag` field on `TrainingContext`, the `pub use` re-export in
`cobre-sddp/src/lib.rs`, and the `None` initializers threaded through
`simulation/pipeline.rs` and `training/forward_pass_state.rs`. Behavior-neutral
(the default path already builds nothing).

### 4.2 `COBRE_THREADS` → `--threads`

`cobre-cli` already exposes `--threads`. Delete the env fallback in
`commands/run/setup.rs`; thread count comes from the flag (or a config field) only.

### 4.3 `COBRE_COLOR` / `FORCE_COLOR` / `NO_COLOR` → `--color`

Replace all three reads with a single `--color {auto,always,never}` argument
(default `auto`, which still honors a non-tty by emitting no color). **Tradeoff to
note:** `NO_COLOR` is a near-universal cross-tool convention; dropping it has a
real UX cost for users who set it globally. The rule in §1 is inflexible, so the
recommendation is to drop it — but this is the one rejection a maintainer might
reasonably want to grant a documented exception, and the decision is recorded here
rather than made silently.

### 4.4 `COLUMNS` → terminal query

Progress rendering should obtain width by querying the terminal (e.g. the
`terminal_size` crate or an `ioctl`), not by reading `COLUMNS`. Falls back to a
fixed default when not a tty.

### 4.5 `COBRE_COMM_BACKEND` → `--comm-backend` / config

Backend selection (`auto` / `local` / `mpi` / …) becomes an explicit CLI argument
or config field resolved by `cobre-cli`, threaded into `cobre-comm` as a typed
parameter. `cobre-comm` stops reading the environment for selection.

### 4.6 `HOSTNAME` → hostname API

The two provenance reads should call a hostname syscall (via a small crate such as
`gethostname`, or `rustix`) rather than the `HOSTNAME` env var, which is not
reliably exported to child processes anyway. Provenance is recorded from the OS,
not the environment.

### 4.7 `FLATC` → `PATH`

The conformance test should invoke `flatc` resolved from `PATH` and skip (or fail
with a clear message) when absent, without a custom env override.

### 4.8 `COBRE_PARITY_REGEN` → explicit regen entry point

The golden-baseline regeneration must not be an `if env == "1" { write } else {
assert }` branch inside the asserting test. Move regeneration to an explicit,
non-ambient mechanism — options, in preference order:

1. A dedicated regen command (a small `xtask` or a `cobre`-adjacent dev binary)
   that recomputes and rewrites the `parity_baselines/*.sha256` files.
2. A separate `#[ignore]`d `regenerate_parity_baselines` test, run on demand with
   `cargo test regenerate_parity_baselines -- --ignored` — explicit and visible,
   no custom variable.

The default test then only ever asserts; regeneration is a deliberate, named
action.

## 5. MPI runtime detection — decided: remove auto-detection (strict)

`cobre-comm` currently probes MPI launcher variables (`OMPI_*`, `PMI_*`, …) to
auto-detect whether the process was started under `mpirun`/`srun`. This is
distinct from `COBRE_COMM_BACKEND`: it does not read a _Cobre_ variable, it reads
the _MPI launcher's_ standard environment. It was the one read that could have
earned a documented exception — an external system's interface, like Cargo's build
vars.

**Decision: the exception is not granted.** Auto-detection is removed. The
communication backend is selected solely by the explicit `--comm-backend` argument
(§4.5); to run under MPI the user passes `--comm-backend mpi`. No launcher-env
probing remains anywhere in `cobre-comm`. This keeps §1 absolute — the workspace
reads _no_ environment for configuration, the sole carve-out being Cargo/Rust
build-and-compile machinery — at the cost of one explicit flag when launching
under `mpirun`/`srun`, which is acceptable and in fact more legible than ambient
detection.

## 6. Reproducibility is unaffected either way

No current environment read changes a numerical result, so this migration is about
configuration hygiene, not correctness. `COBRE_THREADS` and `COBRE_COMM_BACKEND`
affect performance and process topology only — the determinism contract
(reproducibility + declaration-order invariance + fixed reduction order)
guarantees bit-identical output regardless. Color/columns/hostname/version are
cosmetic, provenance, or metadata. `COBRE_W1_DIAG` only observes. Migrating these
to flags/config/APIs removes an invisible input surface; it does not change any
answer, and `parity_baselines_unchanged` plus the `parity_hash_*` suite remain the
backstop.

## 7. Migration impact

| Item                      | Crates touched                                | Risk                                                                |
| ------------------------- | --------------------------------------------- | ------------------------------------------------------------------- |
| Remove `COBRE_W1_DIAG`    | `cobre-sddp`                                  | behavior-neutral; deletes a `TrainingContext` field + threading     |
| `--threads` only          | `cobre-cli`                                   | trivial                                                             |
| `--color` flag            | `cobre-cli`                                   | small; collapses 3 reads into 1 flag                                |
| terminal-width query      | `cobre-cli` (+ a width crate)                 | small; new dev-dependency                                           |
| `--comm-backend`          | `cobre-cli`, `cobre-comm`                     | moderate; new typed param across the comm boundary                  |
| Remove MPI auto-detection | `cobre-comm`                                  | small; deletes the launcher-env probe (folds into `--comm-backend`) |
| hostname API              | `cobre-comm`, `cobre-io` (+ a hostname crate) | small; new dependency                                               |
| `flatc` via `PATH`        | `cobre-io` tests                              | trivial                                                             |
| parity regen entry point  | `cobre-sddp` tests (+ maybe an `xtask`)       | small–moderate                                                      |

Recommended ordering: do the behavior-neutral removal (`COBRE_W1_DIAG`) and the
test-only migrations (`FLATC`, `COBRE_PARITY_REGEN`) first, then the `cobre-cli`
flag migrations, then the `cobre-comm` backend/detection work (which carries the
§5 decision). Each step is independently shippable and parity-neutral.
