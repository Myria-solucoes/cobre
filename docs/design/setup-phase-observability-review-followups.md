# Setup-phase observability — deferred review follow-ups

**Status:** Backlog. Findings from the plan-completion code review (full diff of
the five setup-phase observability epics) that were **deliberately deferred**, not
fixed in the plan. None is a regression in the new code; they are robustness gaps
and accumulated docstring mirror-drift. Recorded here so they are not lost.

**Already fixed in the plan (not deferred):** the 23 simplifier cleanups
(comment/plan-token removal, two drifted `generation.start` numbers), and the one
important finding that corrupted a committed artifact — the `ExportsConfig`
schemars rustdoc said "two active fields" with three present, propagated into
`book/src/schemas/config.schema.json`; fixed to the count-free invariant form and
the schema regenerated.

## Robustness (worth doing first)

1. **`evaporation_models.rs` accepts NaN / ±Inf coefficients silently**
   (`crates/cobre-io/src/extensions/evaporation_models.rs`, ~108–110). The sibling
   `hydro_geometry.rs` parser enforces `is_finite()` on every `f64` field; the
   evaporation parser does not, so a NaN/Inf in `intercept_m3s`,
   `volume_slope_m3s_per_hm3`, or `reference_volume_hm3` flows straight into LP
   matrix coefficients. Mirror the `hydro_geometry.rs` `is_finite()` check + typed
   error, and add a rejecting test.
2. **`extract_required_string` does not null-check the source column**
   (same file, ~125). Arrow `StringArray::value(i)` returns `""` for a null slot
   rather than panicking, so a Parquet file with null `source` passes validation
   and yields `source = ""`, violating the non-nullable contract. Add a null check
   before `value(i)`.

## Docstring mirror-drift (accumulated across epics)

Each is a drifted mirror — present-tense doc no longer matches the code it
describes. Fix by restating the current shape (no frozen counts).

- **`manifest.rs` `SetupTimings` doc** (~253–254) says legacy metadata "reads back
  as zeros", but `Option<SetupTimings>` deserializes to `None` when the `setup`
  key is absent (the per-field `#[serde(default)]` only fires when the object is
  present). The test `training_metadata_without_setup_reads_as_none` asserts
  `None`. Correct the doc to match.
- **`cobre-io/src/output/hydro_models.rs` module doc** (~1–25) and
  **`cobre-sddp/src/production/hydro_models/export.rs` module doc** (~1–7) and its
  **`mod.rs` submodule line** (~23–24): none mention the new
  `build_deviation_summary` / `build_fpha_deviation_point_rows` writers.
- **`cobre-python/src/run.rs` `run_via_study` docstring** (~1203–1212) omits the
  two newly-added write helpers.
- **`cobre-sddp/src/production/fpha_fitting/mod.rs` module doc** (~40–47) omits the
  re-exported `FphaDeviationPoint`.
- **`cobre-cli/src/commands/summary.rs` module doc** (~9–11) claims full section
  parity with the live run but omits the Setup timings section.
- **Evaporation formula mirror** — `hydro_models/evaporation.rs` module doc shows
  the raw-volume form `intercept + slope·v`; `types.rs` shows the centered form
  `intercept + slope·(V − reference_volume)`. Mathematically equal (the intercept
  absorbs `−slope·reference_volume`), but the shapes differ; align them or note
  the equivalence in one place.
- **Narrator comments touched-but-not-cleaned** — `cobre-cli/src/commands/run/mod.rs`
  (~167–172) and `cobre-cli/src/summary.rs` (~153–158): N1 what-narration left in
  place where the diff already touched the surrounding code.
- **`cobre-io/src/config/mod.rs`** (~1022) test doc has a temporal anchor
  ("byte-identical to today") banned by N2 — restate as a durable invariant.

## Behavior nuance (verify, low priority)

- **`broadcast_seconds` timer scope** (`cobre-cli/src/commands/run/setup.rs`,
  ~310–380): the timer spans `build_study_setup` in addition to the MPI
  broadcasts, so the metadata field measures setup-construction work too. Decide
  whether the field should bracket only the broadcast calls (rename it, or tighten
  the timer).

## Missing-test observations (nice-to-have)

- `fpha_deviation_points.rs` parser error paths (~99–168) have no in-file tests
  (only the round-trip happy path is covered elsewhere).
- `collect_fit_deviation_points` zero-production guard
  (`fpha_fitting/deviation.rs` ~242–245) is not exercised by a test for that
  function.
