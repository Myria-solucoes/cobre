# Generalization program — refinement tracker

**Status:** working tracker for the pre-plan refinement stage (owner directive
2026-07-23: refine before invoking the implementation-planning workflow).
Architecture facts live in `beyond-sddp-generalization.md`; this file tracks
only what is open or pending. Strike items as they resolve; this tracker is
superseded by the implementation plan when planning starts.

## Now — active track (owner sequencing, 2026-07-23: DECOMP support first)

The DECOMP program takes implementation priority over the generalization
program; Phase-0a work queues behind it (cheap to reorder while no 0a work
has started). Cobre-side execution order:

1. **bridge-D1** — cobre-python checkpoint-writer binding (small; unblocks
   the FCF importer the moment cut files arrive).
2. **Windowed inflow-history & seeding epic** — ratified 2026-07-24
   (local plan `plans/inflow-windows-epic.md`, untracked; reconciliation in
   `decomp-program-reconciliation.md` §7). Subsumes and supersedes W1;
   breaking input change → v0.13.0; lands **before** Rung 1 (shares the
   external-ingest/replay surfaces and hands Rung 1 the shared
   `DerivedInflowSeeds` helper). **← ACTIVE: in implementation planning.**
3. **Rung-1 bundle + bridge-D8** — the faithful-DECOMP-training gate (full
   item list in the commissioned section below); bridge-D8's depth validation
   lands in the epic's `inflow_seeding.rs` module.
4. **bridge-D9 / W2, Phase 0–1** — unit-groups schema + capability parity,
   staged alongside; needed before fidelity validation, not for the zero-FCF
   smoke.
5. **Rung-2 joint design doc** — after Rung 1; the 0a seam is NOT a
   prerequisite (engine-internal + checkpoint format).

Known accepted cost of this ordering: DECOMP output-schema additions
(simulation probabilities, later `node_id`/`unit_group_id`) land through the
current hand-mirrored CLI + Python writer lists — each pays the double-mirror
tax until 0a's shared orchestration entry point exists.

**Paper-cut inbox:** `plans/conversion-found-improvements.md` tracks cobre
limitations surfaced by real-deck conversion (zero block factors,
availability-fraction float bound, snappy-less parquet build, mandatory
empty `lines.json`) — each backed by an explicitly-interim bridge
workaround with a removal condition; sweep it when touching the affected
surfaces.

## Queued — generalization refinement (behind the DECOMP track)

1. **Schema design (headline when resumed)** — the `study` block, the D15
   boundary-condition kinds, per-engine solver-profile scoping, ED input
   semantics. Working doc: `study-schema-design.md` (this directory); 4 open
   questions parked there (Q1 = ED deterministic inflow source).
2. **Output-orchestration migration scope** — ED-only through the new
   `cobre-io` entry point at Phase 0a, or migrate SDDP's mirrored writer
   call sites (~13 per side, CLI `outputs.rs` + Python `run.rs`) in the same
   move.
3. **ED horizon shape** — single-period vs chronological over the case's
   stages (D15 resolved the boundary condition; horizon length and coupling
   are still open — tracked with the schema doc's open questions).
4. **Phase-0a baseline pinning mechanics** — which cases and hashes pin the
   "SDDP bit-for-bit unchanged" gate; pin the baseline commit explicitly
   (V.1 amendment 2026-07-23).

## Open forks (Part VI of the roadmap)

- **D4** AC-OPF defer vs commit — recommendation (a) defer; unstamped.
- **D5** capacity-expansion method — per-study choice; decided at Phase 4.
- **D6** uncertainty-layer scope — recommendation (a) generalize; bears on
  Phase 1.
- **D3 (partial)** — UC determinism tier sign-off belongs to the Phase-3 gate.

## Commissioned DECOMP-pulled cobre work (owner, 2026-07-23 — cobre-bridge plans)

All default-path byte-neutral per the refined V.0 scoping; details and
namespace disambiguation (`bridge-D<n>` ≠ roadmap `D<n>`) in
`decomp-program-reconciliation.md`.

1. **bridge-D1** — cobre-python checkpoint-writer binding
   (`cobre_io::write_policy_checkpoint`); small, unblocks the FCF importer.
2. **Rung-1 bundle** — external openings in the backward opening tree;
   per-(stage, scenario) probability input (`scenario_probabilities.parquet`)
   threaded into `SuccessorSpec` + the lower bound; enumerated trajectory
   selection; exact weighted upper bound under full enumeration; weighted
   simulation statistics + probabilities exposed in simulation outputs;
   `StoppingRuleConfig::Gap { tolerance }` (absolute, canonical units,
   admission-rejected under sampled forwards).
3. **bridge-D8** — `state_space.inflow_lag_depth` first-class declaration;
   `L_state = max(AR order, declared depth)`; activeness from declaration;
   crisp boundary-cut-depth validation error.
4. **bridge-D9 / W2 — unit groups, staged** — **Phase 0 is now a written
   epic: `plans/blocks-and-units-epic.md`** (local, untracked; target
   v0.14.0, 9 tickets across the block axis and the group axis, five
   decisions listed for ratification). Phase 1 — cobre-computed per-group
   capability from unit nominals, gated on parity with the bridge's
   head-corrected bounds — stays a separate epic and is explicitly out of
   that document's scope, with its schema and precedence slots reserved.
   Phase 0 covers the nullable `block_id` column on bounds parquets (also
   settling per-block thermal capacity) and activates the column that is
   parsed and silently dropped today.
   _Additions 2026-07-24 (from the decoded Itaipu `RI` records + the
   contract decision, bridge `decomp-converter-core.md` §1.4/§1.5):_
   (a) hydro generation bounds need a **group axis** —
   `(hydro_id, unit_group_id, stage_id, block_id?)` min/max MW — because
   the 50/60 Hz floors genuinely vary per stage _and_ per block (the 50 Hz
   floor = ANDE load + committed HVDC flow); (b) `contract_bounds.parquet`
   joins the same nullable `block_id` convention (per-block contract
   limits are supported, no aggregation fallback).
5. ~~**W1** — `recent_observations` seeding under Weekly/Custom cycles~~ —
   **superseded 2026-07-24** by the windowed inflow epic (which found the
   underlying seed path critically defective, not merely cycle-limited; see
   `decomp-program-reconciliation.md` §7).
6. **Rung-2 joint design doc** (this directory) — node-axis policy graph +
   per-node cut pools + path traversal + node-aware checkpoint redesign
   (+ the `ValueFunctionArtifact` decision folded in; `policy.boundary`
   source selector stage → node); design-coordinated with Phase 1;
   implementation after Rung 1.

## Phase-gated obligations (recorded so they are not rediscovered)

- **Phase 0a**: verify `std_mw = 0` noise annihilation in exact arithmetic and
  add the `LoadModel` doc comment (V.1); admission gate generalizes the live
  CLP profile-rejection pattern; the gate run includes `mpirun -n > 1`
  exercising D14; seam covers `validate`/`report`/`summary`, not only `run`.
- **Phase 0b**: re-run `cargo-bloat` / `cargo-llvm-lines` on the real kernel
  (D10 regression check, not a gate).
- **Phase 1**: descriptor-codegen spike (III.1); bulk time-series binary
  trade-off decided as a named choice (III.7); v2 numerically frozen (V.2);
  extend the infrastructure-genericity rule text + CI crate list to
  `cobre-model` / `cobre-network` (IV.1); unify the `policy.boundary`
  spelling with the D15 `study.boundary` axis in the field-relocation map
  (see `study-schema-design.md`).
- **Phase 3**: extend the determinism harness to MIP (`spikes/mipdet/` is the
  prototype); re-run it on every vendored HiGHS upgrade — also a correctness
  gate; sign off UC's tier (D3).
- **Phase 4**: answer the composition-vs-MPI axis question before building
  the study DAG (III.4); promote the cut pool to `ValueFunctionArtifact`
  only then.

## Standing conventions

- Roadmap amendments are dated in place; regenerate
  `beyond-sddp-generalization.html` via `build_beyond_sddp_html.py` after any
  edit to the `.md`.
- Engines are behavior-frozen for the duration of the generalization program
  (V.0 owner scoping, 2026-07-23): raw functionality is kept and prepared for
  the future, not extended algorithmically.
