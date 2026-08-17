# Design docs

Design specifications, decision records, and proposals for the Cobre workspace.
Each doc carries a **status** at its top; this index is the map. Status vocabulary:

- **Live spec** — documents shipped behavior. The cited symbols exist in the tree;
  verify them before acting, but the described behavior is real.
- **Decision record** — a settled evaluation: what was measured, what was decided,
  and what not to re-attempt. Includes pre-registered experiments where noted.
- **Living register** — tracks the workspace's current reserved/deferred state; each
  entry is self-guarding and re-derived against the live tree.
- **Proposal** — a target design that is **not yet implemented**. Snapshot figures in
  a proposal are calibration-time context, not standing claims.

| Doc                                                                                              | Status          | Summary                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------ | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`policy-graph-limitations.md`](policy-graph-limitations.md)                                     | Live spec       | Where the node-native engine is a deliberate subset of a general Markovian policy graph, plus the probability/discount semantics and SDDP.jl convention mapping                      |
| [`enumerated-traversal-distribution.md`](enumerated-traversal-distribution.md)                   | Live spec       | The current "replicate" MPI work-distribution for enumerated traversal, and the deferred intent to migrate to "broadcast"                                                            |
| [`reserved-seams-and-deferred-debt.md`](reserved-seams-and-deferred-debt.md)                     | Living register | Inert-but-intentional config/fields/functions (owner + consuming milestone) and the deferred-architectural-debt ledger                                                               |
| [`backward-warm-start-channels.md`](backward-warm-start-channels.md)                             | Decision record | Backward-pass warm-start channels: one measured-and-closed (H3), two pre-registered, cluster-gated (H1, H2)                                                                          |
| [`testing-architecture.md`](testing-architecture.md)                                             | Proposal        | Workspace-wide target testing standard — layering, per-crate structure, the `test-support` convention, and the migration path                                                        |
| [`anticipated-thermals-and-water-travel-time.md`](anticipated-thermals-and-water-travel-time.md) | Live spec       | How the anticipated-thermal commitment ring and the water-travel-time bucket ring share one lagged-delivery-ring substrate — input parsing, slot-count sizing, and LP entry for both |

## Maintenance convention

A proposal that ships is **deleted from this directory** once its content lands in
its authoritative home (a `.claude/rules/*` rule, a code contract, an
`ARCHITECTURE.md` section) — git history preserves the original. This keeps
`docs/design/` a set of live references, not an archive of superseded plans. When a
proposal is implemented, fold any durable rationale into the home it describes, then
remove the doc.
