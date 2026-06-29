---
paths:
  - "crates/**/tests/**/*.rs"
  - "examples/deterministic/**"
---

# Cobre Testing Rules

How Cobre is tested, to balance coverage, precision, and maintainability. The full
analysis, smell inventory, and migration plan live in
`docs/design/testing-strategy.md`; this file is the standing contract. Generic
cross-language testing pillars (test pyramid, few binaries, one builder, property
tests for ordering, benches≠tests, uniform slow-gating) live in the global Rust
rule; the rules below are the Cobre-specific ones.

## Tiers — use the cheapest tier that catches the regression

1. **Golden bit-exact (parity hash).** SHA-256 over **final** output (cut pool,
   simulation primal / dual / equipment / cost). Reserved for a small, deliberately
   feature-combined golden set whose union spans the
   cross-feature interactions that hide dormant bugs (cascade, anticipated, NCS,
   multi-resolution, energy contracts / pumping) and which exercise the shared
   LP-build / scaling / cut / basis machinery. The authoritative membership is the
   set of `parity_hash_<case>` test names, not a list frozen here; every other
   hashed case is demoted to tier 2 (behavioral). NOT the default; promoting a case
   into the golden set needs justification. The hash MUST NOT include the
   convergence trajectory or any per-iteration state — a faster warm-start to the
   same optimum must not break it.
2. **Behavioral (the default for deterministic cases).** `LB == UB` to tolerance,
   known optimum cost, water-balance closure, feature dispatch values. Backend-
   AGNOSTIC: one assertion covers HiGHS and CLP and survives benign refactors with no
   re-baseline. `crates/cobre-sddp/tests/deterministic.rs` is the model.
3. **LP structural.** Column / row counts, CSC validity, objective-coefficient wiring,
   without running a solver (`template_integration.rs`). First-class, not optional: a
   wrong column count produces numerically-smooth wrong bounds a tolerance assertion
   can miss.
4. **Analytical derivation.** Closed-form expected cut / coefficient from the model
   (`anticipated_backward_cut`-style). The gold standard for a NEW correctness claim —
   write one before adding a deterministic case.

## Contracts

- **Declaration-order invariance is a sort contract, tested as a unit test.** Build a
  `System` from permuted input and assert identical canonical order. A full training
  run over already-sorted input is a tautology that cannot detect an ordering bug — do
  not pass it off as a DOI probe.
- **Feature interactions are tested explicitly.** Each feature ships at least one test
  exercising it alongside its nearest neighbours in the LP column layout. Dormant bugs
  come from untested combinations (discount × anticipated; nonuniform-block extraction).
- **Parity baselines have ONE source of truth — the committed `.sha256`.** Do not add a
  second pinned copy (an `EXPECTED_HASHES`-style mirror): git diff records changes and
  the end-to-end test verifies the computation. A moved hash means the numbers changed —
  investigate, do not reflexively re-baseline.
- **Backend parity lives at the behavioral tier.** HiGHS and CLP differ at the
  FP-trajectory level; assert costs / invariants to tolerance (covers both), reserve
  bit-exact for the golden HiGHS path only.
- **Test output goes to `TempDir`.** No test writes into
  `examples/deterministic/*/output/`; generated artifacts self-delete.

## Cost discipline — Cobre links a solver into every test binary

- `cobre-solver` statically links HiGHS/CLP into every dependent `tests/*.rs`, so each
  new integration-test **binary** pays a full C++ link. Add a test function to an
  existing domain binary rather than a new file; group related tests with `mod`.
- The shared harness helpers (`StubComm`, `build_setup_in_code`, `build_setup_for_case`,
  `run_simulation`) and entity construction (`Stage`/`Hydro`/`Bus`/`Thermal` through the
  `make_*` builders) live once in `tests/common/` — never re-defined per file. Because
  every `tests/*.rs` construction routes through the shared builder, adding a required
  field to one of those `cobre-core` entities is a one-place change in
  `tests/common/builders.rs` (the `<Entity>Spec` default + the `make_<entity>` mapping);
  no `tests/*.rs` consumer is touched.
- That one-place property is scoped to the integration-test layer and does NOT hold
  workspace-wide: production parquet readers and inline `#[cfg(test)]` `--lib` unit-test
  literals construct these entities directly and must each wire a new field themselves.
