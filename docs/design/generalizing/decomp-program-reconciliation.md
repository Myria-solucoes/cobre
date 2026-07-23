# DECOMP program ↔ generalization roadmap — reconciliation

**Date:** 2026-07-23 · **Status:** reconciliation record (binding inputs
absorbed; one scoping amendment applied to the roadmap; sequencing resolved —
DECOMP support first, §5).
**Inputs:** `~/git/cobre-bridge/plans/decomp-conversion-analysis.md` and
`decomp-round-2-revision.md` (verified against cobre `develop` @ `a017facc` =
v0.12.0 — the same tree this corpus's 2026-07-23 re-audit used). The bridge
documents' decisions D1–D9, W1/W2, and the Rung-1/Rung-2 program are **owner
decisions (2026-07-23)** and are treated here as binding, not re-litigated.

**Namespace warning:** the bridge plans and this corpus both use `D<n>` labels
and they DO NOT refer to the same decisions. Here, bare `D<n>` means this
corpus's Part-VI fork; bridge decisions are written `bridge-D<n>` (e.g.
roadmap D9 = engine-selection; bridge-D9 = unit-based hydro capability).

---

## 1. The one real conflict, and its resolution

The roadmap's V.0 owner scoping (earlier on 2026-07-23) froze engine
internals ("no big changes in any engine"); the bridge program (same day)
commissions substantial engine work — Rung 1 (external openings in the
backward pass, per-scenario probabilities, enumeration, exact bound, an
absolute-gap stopping rule), bridge-D8 (`state_space.inflow_lag_depth`),
bridge-D9/W2 (unit groups), W1, and eventually Rung 2 (per-node value
functions).

**Resolution (applied to V.0): the freeze means default-path byte-frozen,
not surface-frozen.** What any existing study computes does not change; every
commissioned extension is opt-in config, byte-neutral at defaults — exactly
the discipline the per-phase solver profiles and the cost-scale factor
already shipped under. Speculative engine work (SDDiP — roadmap D13) stays
deferred; the DECOMP-pulled work is commissioned by a real consumer, which is
precisely the roadmap's own pull-don't-push rule operating.

## 2. Where the bridge program validates the roadmap (no action needed)

- **The FCF is the inter-model boundary** (II.2's central lesson) is now
  practice, not thesis: NEWAVE-converted case → policy checkpoint →
  `policy.boundary` → DECOMP-converted case. The chain the roadmap described
  as the domain's canonical hand-off is the bridge program's product path.
- **D15 (boundary-condition axis)**: the `ValueFunction` kind acquires a real
  external _producer_ — the FCF importer authors synthetic Cobre checkpoints
  (from NEWAVE `cortes`/`cortesh`, individualized cut files) against the
  target case's own manifest. The kind is no longer only SDDP-internal.
- **Closed-world + admission-gate philosophy**: the bridge program
  independently arrived at gate-shaped requirements — the `gap` stopping rule
  must be _rejected_ under sampled forwards (soundness gate), phase-keyed
  profile config is rejected per engine, `inflow_lag_depth` gets a crisp
  validation error. The 0a admission-gate machinery has three more customers.
- **The manifest-bootstrap idiom** (author external artifacts against a
  manifest read back from a 1-iteration run, never re-implement slot layout
  outside cobre) is the sanctioned external-coupling pattern — it is what
  makes bridge-D6's "breaking checkpoint changes acceptable, version marker +
  clean rejection" cheap to live with.

## 3. Where it accelerates or reshapes roadmap elements

| Roadmap element                                                    | Bridge input                                                                                                                                                                                                                     | Consequence                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| III.4 `PolicyGraph → HorizonGraph` (Phase 1)                       | Rung 2: `nodes[] = {id, stage_id}`, transitions re-pointed at nodes, per-node cut pools, path traversal; `Transition.probability` today parsed but consumed nowhere (latent discount bug)                                        | The temporal generalization is a **joint design**: the rename is subsumed by the node-graph redesign; do not do a cosmetic rename first and a node axis second. Rung-2 design doc lives in this directory.                                                                                                                                                      |
| I.3 #6 checkpoint coupling + V.5 `ValueFunctionArtifact` (Phase 4) | bridge-D6: breaking checkpoint/schema changes acceptable now (version marker, clean rejection); Rung 2 adds a node axis; the importer is an external producer                                                                    | The artifact-promotion **precondition ("a second value-function participant") is met**. Recommendation: still no isolated promotion — fold the artifact question into the single node-axis checkpoint redesign (one format break, not two).                                                                                                                     |
| III.2 `Unit`-under-`Plant` (Phase 3)                               | bridge-D9 (subsumes W2): `unit_groups[]` under `Hydro` — per-group `bus_id`, `n_units`, nominal ratings; cobre computes per-group capability; same-bus groups collapse to today's LP; **mandatory-canonical long-term**          | The unit hierarchy arrives **early and continuous**, pulled by Itaipu's 50/60 Hz split. Phase-3 commitment attaches to this substrate as cluster commitment over `n_units`-identical units per group — not to a freshly invented flat `Unit` list. Case-format impact is additive-optional (the V.9 idiom), so it does not violate roadmap D2's deferred break. |
| III.7 / Phase-1 uncertainty store                                  | Rung 1 inputs: `scenario_probabilities.parquet {stage_id, scenario_id, probability}` (bridge-D5a), `scenario_source.*.openings = generated\|external`, `selection = hash\|enumerate`, `state_space.inflow_lag_depth` (bridge-D8) | Concrete pull shaping the store's design: per-scenario **probability is first-class stochastic data**, external realizations must reach the **backward** pass (today external is forward-only; the opening tree is generated noise), and lag depth decouples from fitted AR order. These are exactly the "generalize, don't dump" inputs roadmap D6 wanted.     |
| V.7 determinism                                                    | Rung 1: full enumeration + exact weighted bound + absolute gap (canonical R$ units; `NI` iteration-limit backstop mandatory)                                                                                                     | Strengthens Tier 1: DECOMP-shaped studies get a _deterministic_ gap (no statistical CI), node/iteration-limited termination, no wall-clock. The gap rule's enumeration-only validity is enforced at admission.                                                                                                                                                  |
| III.7 output boundary / Phase-0a shared orchestration entry point  | bridge-D5 (probabilities exposed in simulation outputs), bridge-D7 (`node_id` column on every entity table), bridge-D9 (`unit_group_id` dimension)                                                                               | Three output-schema growth events are coming. Landing the **single shared output-orchestration entry point first** turns each into a one-list change mirrored in Python by construction — an argument for early 0a sequencing of that deliverable.                                                                                                              |
| I.3 #7 config shape / BroadcastConfig                              | New engine-owned config: openings/selection/probabilities/lag-depth/gap-rule                                                                                                                                                     | All new fields are SDDP-engine-owned — consistent with the relocation direction. `BroadcastConfig` grows again; the 0a engine-header-first broadcast design stands.                                                                                                                                                                                             |

## 4. New cobre work items absorbed (commissioned; tracked in `refinement-todo.md`)

1. **bridge-D1 — cobre-python checkpoint-writer binding** exposing the
   already-public `cobre_io::write_policy_checkpoint`. Small; unblocks the
   FCF importer; keeps the byte format single-sourced in cobre.
2. **Rung-1 bundle** (one work package, all default-path byte-neutral):
   external openings in the backward opening tree; per-(stage, scenario)
   probability input + threading into `SuccessorSpec` and the lower bound
   (replacing the two uniform fills); enumerated trajectory selection;
   exact probability-weighted upper bound under full enumeration; weighted
   simulation statistics + probabilities exposed in simulation outputs;
   `StoppingRuleConfig::Gap { tolerance }` (absolute, canonical units,
   admission-rejected under sampled forwards).
3. **bridge-D8 — `state_space.inflow_lag_depth`**: first-class lag-depth
   declaration, `L_state = max(AR order, declared depth)`, slot activeness
   from declaration, crisp boundary-cut-vs-depth validation error.
4. **bridge-D9 / W2 — unit groups, staged**: Phase 0 optional schema +
   bit-parity; Phase 1 cobre-computed per-group capability gated on parity
   with the bridge's head-corrected bounds; nullable `block_id` column on
   bounds parquets (also settles per-block thermal capacity — old bridge Q7).
5. **W1 — `recent_observations` seeding under Weekly/Custom cycles**:
   robustness item, explicitly off the DECOMP critical path
   (calendar-monthly seasons sidestep it); schedule opportunistically.
6. **Rung-2 joint design doc** (this directory): node-axis policy graph +
   per-node pools + path traversal + node-aware checkpoint format (+ the
   `ValueFunctionArtifact` decision folded in) + `policy.boundary` source
   selector generalized stage → node. Design-coordinated with Phase 1's
   temporal/uncertainty purification; implementation after Rung 1.

## 5. Sequencing (recommendation + the one open call)

Two tracks, parallel-safe by subsystem:

- **Track A — generalization 0a** (CLI/io seam, `study` schema, admission
  gate, output-orchestration entry point, then ED): unchanged.
- **Track B — DECOMP-pulled engine work** (cobre-sddp/stochastic internals +
  config): bridge-D1 first (hours, unblocks importer development the moment
  cut files arrive); then the Rung-1 bundle + bridge-D8; bridge-D9 Phase 0/1
  staged alongside; Rung-2 joint design after Rung 1 (the 0a seam is not a
  prerequisite — it is engine-internal plus the checkpoint format).

They touch different subsystems almost everywhere; the shared surfaces
(config schema export, `BroadcastConfig`, output writers) are
merge-manageable and all flow through gates both tracks already respect.

**Resolved (owner, 2026-07-23, follow-up): DECOMP support is implemented
before the generalization program.** Track B leads; Phase-0a work queues
behind it — cheap to reorder while no 0a work has started. The grounds are
the prior recommendation: Rung 1 gates any faithful DECOMP training run
(today the backward pass would fabricate stage-6 inflows, uniformly
weighted), while 0a's slices are internally ordered but not calendar-urgent.
The zero-FCF converter smoke milestone (bridge §11.5) needs _neither_ — it
proceeds immediately on v0.12.0. Known accepted cost: DECOMP output-schema
additions land through the hand-mirrored CLI + Python writer lists until 0a's
shared orchestration entry point exists (each addition pays the
double-mirror tax).

## 6. Amendments applied to the roadmap (2026-07-23, this reconciliation)

- **V.0 owner scoping** — freeze re-stated as _default-path byte-frozen_ with
  the commissioned DECOMP carve-out (§1).
- **III.4** — value-function participant status (external producer exists;
  first live composition instance runs manually via the bridge) + the
  fold-the-artifact-into-Rung-2 note.
- **III.2** — unit-groups substrate note (Phase-3 commitment = cluster
  commitment over the bridge-D9 hierarchy).
- **D15** — external producer + Rung-2 node-selector facts recorded for the
  Phase-1 `policy.boundary` → `study.boundary` unification.

`study-schema-design.md` gains the corresponding §9 (config-surface impact);
`refinement-todo.md` gains the commissioned work-item section.
