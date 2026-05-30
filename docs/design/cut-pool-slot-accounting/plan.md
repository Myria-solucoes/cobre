# Cut-pool slot accounting: reporting fix (1) + dense slot packing (2)

Status: Phase 1 done · Phase 2 done (2026-05-30)

Implementation note (Phase 2): rather than threading an `iteration_base`
parameter through all ~240 `CutPool::new`/`FutureCostFunction::new` call sites,
the field defaults to 0 (reproducing the legacy formula exactly) and is set once
centrally via `fcf.set_iteration_base(start_iteration + 1)` in
`TrainingSession::new` — the single path all training (fresh/warm/resume) flows
through. Base 0 stays a correct fallback (just gappy), so the change degrades
gracefully and touches zero existing call sites. Three tests needed fixes: the
two anticipated-cut tests hard-coded the iteration-1 cut at slot 1 (now slot 0);
the basis-churn test's multi-phase artifice tripped an over-strict guard on
`set_iteration_base`, which was removed (the real hazard — overwriting an active
slot — is already guarded by `add_cut`). All `cobre-sddp` (1714), `cobre-cli` +
`cobre-io` (1500) tests pass; clippy clean.

## Symptom

The training console always shows fewer active than generated policy rows, the
difference being exactly the number of cut-receiving stages, even with cut
selection disabled. Example: `Policy rows: 3150 active / 3213 generated`.

## Root cause (confirmed)

Cut slot index (`crates/cobre-sddp/src/cut/pool.rs:211`):

```
slot = warm_start_count + iteration * forward_passes + forward_pass_index
```

The training loop numbers iterations **1-based**
(`training_session/mod.rs:343` → `(start_iteration + 1)..=max_iterations`) and
passes that value straight to `add_cut` (`backward_pass_state.rs:850`). So the
slot block `[warm_start_count, warm_start_count + forward_passes)` is never
written in any pool. `populated_count` is a high-water mark (`pool.rs:280`), so
it counts that empty block. `total_generated = Σ populated_count`
(`training_output.rs:342`), giving an over-count of
`forward_passes × (cut-receiving stages)`. The reported example had
`forward_passes = 1`.

Pool capacity is sized `(max_iterations + 1) * forward_passes`
(`setup/mod.rs:462`) precisely to fit the 1-based offset.

## Guiding principle

The slot formula conflates two concepts: **where** a cut is stored vs. **which
iteration** produced it (`metadata.iteration_generated`). Both fixes keep
`iteration_generated` as the *true* iteration so the five filters that compare
against `current_iteration` stay byte-for-byte unchanged:

1. delta-cut batching — `pool.rs:363` (`active_delta_cuts`)
2. MPI sync packing — `cut_sync.rs:412` (`pack_local_records`)
3. cut-selection eligibility — `cut_selection.rs:496`
4. budget protection — `pool.rs:942`
5. warm-start sentinel — `pool.rs` (`WARM_START_ITERATION = u64::MAX`)

Crucial verified property: **the LP only ever includes active cuts**
(`forward.rs:319, 455/474, 614, 791`; `stage_solve.rs:185` uses
`active_cuts()`/`active_delta_cuts()`). The empty reserved slots are inactive,
so they are never constraints — removing them cannot change any bound or result.

---

## Phase 1 — Reporting fix (independent, low-risk)

Decision D1 (resolved): "generated" **includes** warm-start/loaded cuts, matching
current intent.

Changes:

1. `cut/pool.rs`: add `generated_count: usize` to `CutPool`. Initialize to
   `warm_start_count` in every constructor (`new`, `new_with_warm_start`,
   `from_deserialized`). Increment once in `add_cut` (`pool.rs:243`, the sole
   insertion site — also covers MPI remote cuts, which arrive via
   `fcf.add_cut`). Generation is cumulative: `set_active`/`deactivate` do not
   touch it.
2. `cut/fcf.rs`: add `total_generated_cuts(&self) -> usize` = `Σ
   pool.generated_count` (mirrors `total_active_cuts`).
3. `training_output.rs:342`: `total_generated: fcf.total_generated_cuts() as u64`.

Parity: automatic. CLI and Python share the single write path
(`results_writer.rs:98-103`); `cobre-python` has no independent assignment. No
second edit.

Tests: update the value-asserting tests (`training_output.rs:623`,
`cobre-io mod.rs:546/636`, `manifest.rs:513` roundtrip) and add a
`forward_passes > 1` test asserting `total_generated < Σ populated_count`
(gap excluded) and `total_generated == actual add_cut count`.

Risk: none to optimization / wire format / determinism. Pure accounting. Fixes
the console line on its own.

---

## Phase 2 — Dense slot packing (internal optimization, careful) — DONE

Design — `iteration_base`:

- Add `iteration_base: u64` to `CutPool`. Formula:
  `slot = warm_start_count + (iteration − iteration_base) * forward_passes + fpi`
- Thread `iteration_base` through `CutPool::new` / `FCF::new` /
  `new_with_warm_start` as an explicit parameter.
  - Tests pass `0` ⇒ formula reduces to exactly today's ⇒ most slot-assertion
    tests need only a mechanical `, 0` (no expectation rewrites).
  - Production passes `start_iteration + 1` ⇒ first training iteration maps to
    slot `warm_start_count` (dense). Fresh: base 1. Resume: base `completed+1`.
- `metadata.iteration_generated` keeps the true `iteration` ⇒ all five filters
  untouched.
- `add_cut`: `debug_assert!(iteration >= iteration_base)`.

Confirmed safe:

- MPI determinism — receiver recomputes slot from transmitted (true `iteration`,
  global `forward_pass_index`) with identical `iteration_base` per rank ⇒
  cross-rank agreement and bit-determinism across rank counts preserved
  (`cut_sync.rs:372-378, 543-549`).
- Policy load / old checkpoints — wire format stores `slot_index` but load
  ignores it (`new_with_warm_start`/`from_deserialized` pack by array order);
  old gappy policies still load.
- Resume — base `start_iteration+1` lands resumed cuts densely after the warm
  region; also fixes today's larger resume gap. Needs an explicit resume test.
- Basis cache — keys on slot identity; dense slots only shrink lookup buffers
  (`stage_solve.rs:161`), headroom only.
- Capacity — current `(max_iterations+1)·forward_passes` stays safe (now
  over-allocates one block). Tightening deferred.

Tests: mechanical `iteration_base` arg (value `0`) at all
`CutPool::new`/`FCF::new` call sites; update the ~3 load-bearing dense-slot
assertions to base `1` (`backward.rs:1633` slot 7→4, `:2485` slot 17→11,
`:4012` `populated_count` 6→3 / slot range `3..6`→`0..3`); add a resume-placement
test and a `generated_count == populated_count − warm_start_count` (fresh)
invariant test. Production sites: `setup/mod.rs:463`, CLI `run.rs:399/439`,
Python `run.rs` warm/resume paths.

---

## Sequencing & verification gates

1. Phase 1 → `cobre-sddp` + `cobre-io` + `cobre-cli` tests; confirm console reads
   true. Commit.
2. Phase 2 → full `cobre-sddp` suite + determinism tests. Gate: A/B run (before
   vs. after Phase 2) shows bit-identical lower bound and results;
   `total_generated` unchanged; `Σ populated_count` drops to match it. Commit.

## Optional follow-ups (out of scope)

- Filter inactive/empty rows from `policy_export` (`policy_export.rs:32`) to slim
  policy files.
- Tighten pool capacity to drop the now-unneeded `+1` block.
