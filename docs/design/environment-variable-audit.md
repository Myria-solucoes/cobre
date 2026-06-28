# Environment-variable usage audit & recommendation

> **Status**: Assessment / recommendation. No change applied yet. Scope: every
> environment variable read or stamped anywhere in the Cargo workspace, as of the
> current branch state (post comment-cleanup). The actionable recommendation is a
> single removal (§5); everything else is judged acceptable and documented here so
> the next reader does not re-open the question.

## 1. Why this exists

A reviewer asked what `COBRE_PARITY_REGEN` is for and, more broadly, whether
env-var-driven behavior belongs in Cobre's source at all. This document answers
both: it inventories every environment variable the workspace touches, classifies
each by where it lives and what it controls, states the one place the practice is
a genuine smell, and records the guideline that keeps the surface from growing.

The load-bearing question for a scientific solver is **reproducibility**: can the
ambient environment change the numerical answer? The short answer is no (§4). The
remaining concerns are about hygiene, not correctness.

## 2. Inventory

Grouped by category. Locations are crate-relative; symbols, not line numbers, so
this does not rot.

### 2a. Test-only

| Variable                     | Location                                                        | Purpose                                                                                                                                                                                                                                                                                              |
| ---------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `COBRE_PARITY_REGEN`         | `cobre-sddp/tests/parity_hash_d01_d15.rs`, `parity_hash_clp.rs` | Golden-file regen switch. Unset: the test hashes the active cut set and asserts equality against the committed `parity_baselines/DNN.sha256`; mismatch fails. `=1`: the test **writes** the baseline instead — the deliberate "I changed an output on purpose, regenerate the goldens" escape hatch. |
| `FLATC`                      | `cobre-io/tests/flatbuffers_schema_conformance.rs`              | Locate the `flatc` binary for the FlatBuffers conformance round-trip.                                                                                                                                                                                                                                |
| `env!("CARGO_MANIFEST_DIR")` | many `*/tests/*.rs`, a few `src/**` `#[cfg(test)]` blocks       | Resolve fixture paths at compile time.                                                                                                                                                                                                                                                               |

### 2b. Build-time (Cargo-injected / version stamping)

| Variable                                                                                                                   | Location                                                        | Purpose                                                                                                                                   |
| -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `CARGO_MANIFEST_DIR`, `OUT_DIR`, `CARGO_CFG_TARGET_OS`, `CARGO_CFG_TARGET_ENV`, `CARGO_FEATURE_HIGHS`, `CARGO_FEATURE_CLP` | `cobre-{cli,sddp,solver}/build.rs`                              | Standard Cargo build-script inputs (output dir, target triple, which solver feature is active). Required by Cargo; not user-facing knobs. |
| `env!("CARGO_PKG_VERSION")`                                                                                                | `cobre-io` output writers, `cobre-sddp/policy/orchestration.rs` | Stamp the producing Cobre version into output/policy provenance.                                                                          |

### 2c. CLI binary (`cobre-cli`)

| Variable                                 | Symbol                  | Purpose                                                                                            |
| ---------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------- |
| `COBRE_THREADS`                          | `commands/run/setup.rs` | Worker-thread count override (the `--threads` flag is the primary path; this is the env fallback). |
| `COBRE_COLOR`, `NO_COLOR`, `FORCE_COLOR` | `main.rs`, `banner.rs`  | Color control. `NO_COLOR` is a documented cross-tool convention.                                   |
| `COLUMNS`                                | `progress.rs`           | Terminal width for progress rendering.                                                             |

Reading environment in the _binary_ (front-end UX and run configuration) is
conventional and appropriate.

### 2d. Library — infrastructure / provenance

| Variable                                                      | Symbol                                                       | Purpose                                                                                                                                                                                                                    |
| ------------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `COBRE_COMM_BACKEND` + MPI launch vars (`OMPI_*`, `PMI_*`, …) | `cobre-comm/src/factory.rs`                                  | Select the communication backend (`auto` default) and auto-detect whether the process was launched under an MPI runtime. Detecting the launcher via its environment is the idiomatic mechanism for an MPI-capable library. |
| `HOSTNAME`                                                    | `cobre-comm/src/local.rs`, `cobre-io/src/output/manifest.rs` | Machine identity for output provenance / diagnostics. Read-only, fallback-chained.                                                                                                                                         |

These read environment inside library crates, but each selects _infrastructure_
or records _provenance_ — never the numerical result — and the comm-backend case
is the standard MPI-detection pattern.

### 2e. Library — diagnostic (the finding)

| Variable        | Symbol                                                                               | Purpose                                                                                                                                                                                                                                                                        |
| --------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `COBRE_W1_DIAG` | `cobre-sddp/src/stochastic/noise_key_diag.rs` (`NoiseKeyDiag::from_keys_if_enabled`) | Gate for a backward-pass opening-ordering diagnostic: when set, the backward pass emits, per opening, a σ-weighted noise key paired with that opening's warm-resolve `simplex_iterations`, to analyze whether reordering openings by noise similarity shrinks warm-start work. |

This is the only env-gated _behavior_ in pure-library source, and its own module
doc opens with "Throwaway, env-gated backward-pass diagnostic." See §5.

## 3. Classification summary

- **Test-only** (§2a): idiomatic. A golden-file regen switch and a `flatc` locator
  are exactly what test harnesses use; they ship in no binary.
- **Build-time** (§2b): required Cargo machinery and version stamping. Not optional.
- **CLI binary** (§2c): conventional. A CLI reading `NO_COLOR`/`COLUMNS`/a thread
  override is expected; the binary is the right layer for ambient UX/run config.
- **Library infra/provenance** (§2d): acceptable. Selects backend or records the
  host; the MPI launch-environment probe is the standard mechanism.
- **Library diagnostic** (§2e): a smell — see §5.

## 4. Reproducibility is not at risk

No environment variable changes a numerical result:

- `COBRE_THREADS` and `COBRE_COMM_BACKEND` change _performance and process
  topology_ only. Cobre's determinism contract — reproducibility plus
  declaration-order invariance, with fixed reduction order — guarantees
  bit-identical results regardless of thread count or backend.
- Color, columns, hostname, and version are cosmetic, provenance, or metadata.
- `COBRE_W1_DIAG` only _observes_; on the default (unset) path it builds and
  computes nothing.

So ambient environment is orthogonal to the answer. The concern with §2e is
hygiene and shipped scaffolding, not correctness.

## 5. Recommendation: remove `COBRE_W1_DIAG`

`COBRE_W1_DIAG` and its module `cobre-sddp/src/stochastic/noise_key_diag.rs` are
**vestigial** and should be removed:

1. **Its question is already answered.** The diagnostic was built to measure
   whether reordering backward openings by noise-key similarity reduces warm-start
   pivots. `docs/design/backward-opening-ordering.md` records that this is settled:
   the reorder is implemented default-on (descending noise-key order) and its
   configuration options were _removed_ after the A/B measurement confirmed the
   gain. The measurement scaffold that fed that decision has outlived its purpose.
2. **It is library-level env-gated behavior.** It lives in shipped source (not
   `tests/`), gated on an undocumented magic string `"COBRE_W1_DIAG"`, and is
   threaded through the hot-path context: a `noise_key_diag` field on
   `TrainingContext`, a `pub use` re-export in `cobre-sddp/src/lib.rs`, and a
   `None` initializer repeated across the simulation and forward-pass construction
   sites. That is surface area and reader-attention cost for a probe nobody runs.
3. **Removal is behavior-neutral.** On the default path it does nothing; deleting
   the module, the `TrainingContext` field, the re-export, and the `None`
   threading cannot move any result or parity hash.

Removal is the recommended action. If the opening-ordering analysis is expected to
recur, the alternative is to **promote** the probe to a first-class, documented
mechanism — a `--diagnostics` CLI flag or a Cargo feature — rather than an ambient
magic env var in the library. Keeping it as-is is the option this document argues
against.

The other variables (§2a–§2d) should be left as they are.

## 6. Guideline for future env-var use

To keep this surface from regrowing:

- **Numerical results must never depend on the environment.** An env var may
  select infrastructure (threads, backend) or toggle diagnostics/provenance; it
  must not change the answer. This follows from the determinism contract.
- **The binary is the place for run/UX config.** New ambient configuration belongs
  in `cobre-cli`, ideally behind a flag with an env fallback — not in a library
  crate's hot path.
- **Test-only knobs belong in `tests/`.** Golden-file regen switches and tool
  locators are fine there and ship in nothing.
- **A library that must read the environment should do so at a single, named,
  documented seam** (as `cobre-comm`'s backend factory does), not via an
  undocumented string buried beside the algorithm.
- **No throwaway scaffolding in shipped source.** A measurement probe lives until
  its measurement concludes; once the decision it informed is locked, the probe is
  removed (or promoted to a real feature), not left env-gated in the library.
