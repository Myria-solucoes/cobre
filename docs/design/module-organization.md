# Cobre module organization — design proposal

**Status:** Proposal for review. Not yet scheduled for execution.
**Scope:** Crate-root file clustering **and** god-file internal splits across
`cobre-sddp`, `cobre-solver`, `cobre-core` (the three crates with flat-root
sprawl). `cobre-io`, `cobre-stochastic`, `cobre-cli` are already adequately
organized and out of scope except where they consume moved symbols.

## 1. Why

A newcomer opening `cobre-sddp/src` sees **46 flat files / ~72k lines** with no
map. The crate already uses directory modules (`cut/`, `lp_builder/`, `setup/`,
`simulation/`, `training_session/`), so finishing the job is consistent with
established intent, not a new style. Two compounding problems:

1. **Flat-root sprawl** — navigation/onboarding cost.
2. **God-files** — `backward.rs` 6888, `indexer.rs` 5499, `hydro_models.rs` 4981,
   `forward.rs` 4734, `fpha_fitting.rs` 4673; `highs.rs` 3149, `clp.rs` 2589;
   `resolved.rs` 2562, `system.rs` 2405.

Clustering fixes (1); it does **not** fix (2). The full quality bar needs both.

### Current state (measured)

| Crate            | Flat root `.rs` | Subdirs | Root lines |
| ---------------- | --------------- | ------- | ---------- |
| cobre-sddp       | 46              | 5       | 71,875     |
| cobre-solver     | 10              | 0       | 9,772      |
| cobre-core       | 13              | 2       | 11,162     |
| cobre-comm       | 7               | 0       | 3,802      |
| cobre-io         | 10              | 8       | 6,449      |
| cobre-stochastic | 4               | 6       | 2,627      |
| cobre-cli        | 6               | 1       | 4,155      |

## 2. Invariants every step must hold

These are non-negotiable and are what make a reorg of this size safe:

- **Determinism oracle on every commit.** `examples/1dtoy` bit-identical
  before/after (release build → run → `diff -r` excluding the 8 timestamp/timing
  files). A reorg that changes one output byte is a bug. This is the primary
  guard for the relocation phase.
- **Public re-export surface preserved verbatim.** `cobre_sddp::X`,
  `cobre_solver::X`, `cobre_core::X` must keep resolving for `cobre-cli`,
  `cobre-python`, and the in-crate `tests/` (which compile as separate crates).
  Internals move; the curated `lib.rs` `pub use` block is the contract.
- **Genericity (infra crates).** `cobre-core` and `cobre-solver` carry zero
  `sddp`/`SDDP`/`Benders` vocabulary — proposed dir/file names below are
  algorithm-neutral (`backends/`, `ffi/`, `model/`, `constraints/`, `stats/`).
  Re-verified per step with `scripts/check-infra-genericity.sh`.
- **Contract pointers move with their files.** `CLAUDE.md` and
  `.claude/rules/sddp.md` name specific files (`basis_reconstruct.rs`,
  `workspace.rs` byte layout, `indexer.rs`, `lp_builder/mod.rs`, `setup/mod.rs`,
  `cut::row::push_scaled_coefficient`). Each move updates those pointers in the
  same commit. §7 lists them.
- **Determinism-critical regions preserved verbatim.** Benders cut-sign
  negation, Welford canonical-order statistics, basis reconstruction, per-stage
  backward exchange, fixing-row `0..0` sentinels, FPHA average-storage
  coefficient, NCS availability factors. The god-file splits below mark these as
  **SEALED** submodules — relocated as whole units, never edited mid-split.
- **The PAR-annual path is oracle-blind.** `1dtoy` uses non-annual selection, so
  splits touching `estimation`/`noise`/`par`-adjacent or annual code rely on the
  `cobre-stochastic`/`cobre-sddp` unit tests as the complementary guard (same
  lesson as the deferred partitioned-covariance underflow).
- **No plan-leakage** in any shipped comment touched.

## 3. Two phases (risk-separated)

**Phase A — relocation only (low risk).** Move flat files into cluster
directories; fix `mod`/`use` paths; preserve re-exports. No file internals
change. Each cluster is one commit, determinism-verified. This is the exact
pattern `020`/`020b` proved, applied breadth-first.

**Phase B — god-file internal splits (higher risk).** Split the large files into
directory submodules. This changes file _internals_, so it is per-file, each its
own commit, with SEALED determinism-critical submodules moved verbatim. Some
files are explicitly **not** split (see §5).

Phase A first means the tree is already navigable before the riskier Phase B,
and Phase B operates on already-relocated files.

## 4. Target structure — Phase A clustering

### cobre-sddp/src (46 flat → ~8 flat + domain dirs)

| Target dir         | Absorbs (flat files today)                                                                                                                                                                                                                                                                                                                                          |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `training/`        | train (loop, from `training.rs`) + the `training_session/` contents (session, results) + training_output, **and** the training-phase passes: forward, backward, forward_pass_state, backward_pass_state, lower_bound, state_exchange, visited_states, trajectory. Sibling to `simulation/`. **Eliminates the root `training.rs` and the `training_session/` name.** |
| `solve/`           | stage_solve, solver_phase — the LP-solve seam **shared** by training and simulation (so it lives outside `training/`)                                                                                                                                                                                                                                               |
| `cut/` _(extend)_  | + cut_selection, cut_sync, dcs, basis_reconstruct                                                                                                                                                                                                                                                                                                                   |
| `stochastic/`      | estimation, noise, noise_key_diag, lag_transition, inflow_method, stochastic_summary                                                                                                                                                                                                                                                                                |
| `production/`      | hydro_models, fpha_fitting, energy_conversion, conversion                                                                                                                                                                                                                                                                                                           |
| `lp/`              | indexer, generic_constraints, **and the existing `lp_builder/` nested as `lp/builder/`**                                                                                                                                                                                                                                                                            |
| `convergence/`     | convergence, stopping_rule, risk_measure                                                                                                                                                                                                                                                                                                                            |
| `policy/`          | policy_load, policy_export, provenance, resolved_parameters, scaling_report, orchestration                                                                                                                                                                                                                                                                          |
| `workspace/`       | workspace, context                                                                                                                                                                                                                                                                                                                                                  |
| **root keeps**     | lib, error, config, horizon_mode, solver_stats, gemm, validate_phases                                                                                                                                                                                                                                                                                               |
| _(unchanged dirs)_ | setup/, simulation/                                                                                                                                                                                                                                                                                                                                                 |

**`training/` ⇄ `simulation/` symmetry (owner decision):** the two solver phases
become sibling directory modules. `training.rs` and `training_session/` both
dissolve into `training/`. The exact training-vs-shared boundary (which pass code
is training-only vs reused by simulation — e.g. is any of `forward` shared?) is
confirmed during plan refinement by reading the cross-module call graph;
`solve/` holds what is provably shared.

### cobre-solver/src (10 flat → 4 flat + 3 dirs)

| Target      | Files                                                                                                     |
| ----------- | --------------------------------------------------------------------------------------------------------- |
| `backends/` | highs, clp, profiled                                                                                      |
| `ffi/`      | highs (from `ffi.rs`), clp (from `clp_ffi.rs`) — **the unsafe boundary, isolated**                        |
| root keeps  | types, trait_def, profile (the crate's public vocabulary — **kept at root, owner decision**), baking, lib |

(`cobre-solver` → 2 dirs + root: `backends/`, `ffi/`; everything else stays at root.)

### cobre-core/src (13 flat → ~3 flat + dirs)

| Target             | Files                                                         |
| ------------------ | ------------------------------------------------------------- |
| `model/`           | resolved, scenario, temporal, parameters, penalty             |
| `constraints/`     | generic_constraint, initial_conditions, training_event        |
| `stats/`           | welford                                                       |
| root keeps         | lib, error, entity*id, system *(or `system/` if split — §5)\_ |
| _(unchanged dirs)_ | entities/, topology/                                          |

### cobre-comm/src — SKIPPED (owner decision)

Small enough (7 files / 3.8k lines) that clustering adds little. Out of scope.

## 5. Target structure — Phase B god-file splits

Verdict per god-file, from the seam analysis. **SEALED** = move as one verbatim
unit (determinism-critical), do not edit during the split.

### Split (high value, feasible)

**`training/backward/`** (6888) → `mod` (docs, `BackwardResult`, `StagedCut`,
`SuccessorSpec`) · `lp_setup` (load_backward_lp, patch_opening_bounds,
resolve_backward_basis) · **`duals_extraction` [SEALED]** (extract_duals_from_view,
extract_state_duals_only — Benders sign + `col_scale` division) ·
`outcome_aggregation` · **`trial_point` [SEALED]** (process_trial_point_backward —
hot-path kernel, deterministic ordering) · `tests`.

**`training/forward/`** (4734) → `mod` (docs, `ForwardResult`, `SyncResult`,
`sync_forward`) · `scenario_partition` · **`stats_aggregation` [SEALED]** (Welford
canonical-order) · `stage_loop` (hot-path scratch reuse — careful) ·
`delta_cut_batch` · `basis_capture` · `tests`.

**`lp/indexer/`** (5499) → `mod` · `layout` (offset arithmetic) ·
**`state_mapping` [SEALED, MEDIUM]** (state_to_lp_column /
state_to_lp_incoming_column + anticipated ring-buffer) · `anticipated` (iterators)
· `sparse_state` (set_nonzero_mask) · `constructors` (MEDIUM — many dependent
ranges). Preserve the `0..0` fixing-row sentinels and the generic-constraint
sentinel contract.

**`production/hydro_models/`** (4981, LOW) → `mod` · `types` · `production` ·
`evaporation` · `summary`.

**`production/fpha_fitting/`** (4673, MEDIUM-HIGH) → `mod` · `error` · `geometry`
(ForebayTable, bounds) · `production` (ProductionFunction) · `tangent` · `grid`
(the single authoritative grid formula — keep one owner) · `selection` (greedy +
kappa + validate; γᵥ≥0 check) · `tests`. The FPHA average-storage coefficient is
_used_ in `lp_builder/matrix.rs` (not moved); only its static validation lives
here.

**`production/energy_conversion/`** (1921, LOW-MED) → `mod` · `types` · `override`
· `builder`.

**`backends/highs/`** (3149) and **`backends/clp/`** (2589) → split config /
solver / retry-escalation / `SolverInterface` impl / tests; FFI already isolated
in `ffi/`. Keep `// SAFETY:` blocks intact.

**`model/resolved/`** (2562, LOW) → `penalties` · `bounds` · `factors`
(load/exchange/ncs) · `generic` · `mod`. Pure data containers, clean seams.

**`system/`** (2405) → `mod` (System + accessors) · `builder` (SystemBuilder) ·
`validate`.

### Do NOT split (cohesive or orchestration-tangled)

`backward_pass_state.rs`, `forward_pass_state.rs` (orchestrators — splitting
fragments tightly-coupled iteration logic), `lower_bound.rs`, `stage_solve.rs`
(the **shared** 3-driver invariant-enforcement seam — fragmenting risks a missed
`enforce_basic_count_invariant`), `solver_phase.rs`, `state_exchange.rs`. These
relocate in Phase A but keep their single-file form.

## 6. Re-export plan (how the public API survives)

`lib.rs` is the contract. After a file moves from `foo.rs` to `cluster/foo.rs`:
the `pub use foo::{...}` line becomes `pub use cluster::foo::{...}` (or the
`cluster/mod.rs` re-exports `foo`'s public items and `lib.rs` points at the
cluster). External call sites (`cobre_sddp::X`) and the integration tests in
`tests/` (which the analysis confirmed import via the re-export surface, not raw
module paths) are unaffected. Where an in-crate test _does_ use a raw `pub mod`
path (audit per cluster), either update the test path or add a thin re-export
shim — decided per case, preferring the re-export shim to keep test churn low.

The grouped banner style already in `lib.rs` is preserved.

## 7. Contract-pointer updates (must accompany the moves)

When the named file moves, update the pointer in the **same commit**:

- `CLAUDE.md` — `basis_reconstruct.rs` entry-point note; `workspace.rs`
  `CapturedBasis` byte-layout owner; `lp_builder/mod.rs` + `indexer.rs` "adding LP
  vars/constraints"; `setup/mod.rs` StudySetup note; hot-path file list
  (`forward.rs`, `backward.rs`, `training.rs`, `simulation/pipeline.rs`,
  `lower_bound.rs`).
- `.claude/rules/sddp.md` — `backward.rs (extract_duals_from_view)`,
  `cut::row::push_scaled_coefficient`, `indexer.rs`, `lp_builder/matrix.rs` +
  `template.rs`, `noise.rs`, `lower_bound.rs`, `backward_pass_state.rs`,
  `convergence.rs`, `cut/pool.rs`, `basis_reconstruct.rs`.
- `.claude/architecture-rules.md` — context-struct file column, hot-path driver
  list, StudySetup sub-struct map.

These are load-bearing references; a stale pointer is a silent lie.

## 8. Risk register

| Item                                 | Risk     | Mitigation                                                                                   |
| ------------------------------------ | -------- | -------------------------------------------------------------------------------------------- |
| Phase A relocation                   | LOW      | Pure motion; 1dtoy oracle per cluster; re-export preserved                                   |
| `backward` duals/trial-point split   | HIGH     | SEALED submodules moved verbatim; oracle + anticipated-cut unit tests                        |
| `forward` Welford split              | HIGH     | SEALED stats submodule verbatim; oracle                                                      |
| `indexer` state_mapping/constructors | MED      | SEALED state_mapping; preserve `0..0` sentinels; conformance/determinism tests               |
| `fpha_fitting` grid                  | MED-HIGH | Single grid-formula owner; oracle-blind → rely on in-file fpha tests (some slow-tests-gated) |
| Solver backend splits                | MED      | FFI/unsafe already isolated; SAFETY blocks intact; genericity gate                           |
| `git blame` churn                    | —        | Accepted cost; justified by onboarding goal; `--follow` survives moves                       |
| Integration-test raw module paths    | LOW      | Audited per cluster; re-export shim or test-path update                                      |
| MEMORY pointers to moved files       | LOW      | Memories are snapshots; update the load-bearing ones post-reorg                              |

## 9. Sequencing (proposed)

Phase A, lowest-coupling clusters first so the pattern is proven before the
busy ones:

1. cobre-solver `ffi/` + `backends/` (isolated, genericity-checked).
2. cobre-core `model/` + `constraints/` + `stats/`.
3. cobre-sddp `convergence/`, `policy/`, `workspace/`, `production/` (relocate
   only), `stochastic/`, `solve/`, `lp/` (+ nest `lp_builder/`→`lp/builder/`),
   extend `cut/`, then `training/` (fold in `training.rs` + `training_session/`).

Phase B, one god-file per commit, easiest→hardest: `model/resolved`, `system`,
`hydro_models`, `energy_conversion`, `indexer`, `fpha_fitting`, solver backends,
`forward`, `backward`.

Each commit: relocation/split → `cargo build --release --workspace` +
`cargo build --manifest-path crates/cobre-python/Cargo.toml` → targeted nextest →
clippy → genericity (infra) → 1dtoy oracle → `cargo fmt`.

## 10. Resolved decisions (owner)

1. **`lp_builder/` nesting** — ✅ NEST as `lp/builder/`.
2. **`training.rs` home** — ✅ Create a `training/` directory module, sibling to
   `simulation/`, that absorbs `training.rs` **and** `training_session/`. No root
   `training.rs`; no `training_session/` name.
3. **cobre-solver `core/`** — ✅ KEEP `types`/`trait_def`/`profile` at root (the
   crate's public vocabulary). Only `backends/` + `ffi/` are introduced.
4. **cobre-comm** — ✅ SKIP (small enough).
5. **Phase B depth** — ✅ Split ALL listed god-files (cobre-sddp + solver + core).
6. **Vehicle** — ✅ Formal progressive plan via the `/plan` skill, executed with
   `/implement-plan` (same guardian/score/boundary rigor as the prior plan).

## 11. Effort (rough)

Phase A: ~12–15 relocation commits. Phase B: ~9 split commits (the SEALED-heavy
`backward`/`forward`/`indexer`/`fpha_fitting` are the bulk). Each is small,
mechanical, and independently verified — the program is long but low-variance
because the oracle catches any behavior drift immediately.
