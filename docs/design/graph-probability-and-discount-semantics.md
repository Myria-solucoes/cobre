# Graph Probability and Discount Semantics

The node-native engine carries randomness on two independent axes and treats
discounting as a quantity separate from probability. This note reconciles the
two probability axes with the discount treatment, states how both map onto
[SDDP.jl](https://github.com/odow/SDDP.jl), and records the standing behaviors
that replaced the validation rules retired when the graph became node-native.
It is the sibling of `docs/design/policy-graph-limitations.md`, which records
where the engine is a subset of, or differs in convention from, a general
Markovian policy graph.

## Two probability axes

A declared `nodes[]` policy graph carries probability on two axes that compose
but never mix.

### Between-node axis — graph transition edges

Graph transitions are **node-edge weights**: each out-edge of a node carries the
conditional probability of moving to that successor. They are first-class inputs
of the node-native dialect. At load, a source node's out-edge weights are
**normalized to sum to 1** by `normalize_out_edge_probabilities`, subject to a
per-source `Σ ≈ 1` agreement check within `PROB_TOLERANCE`. The path probability
of a root→leaf trajectory is the product of the normalized out-edge weights along
its edges (times one opening weight per node; see below), and the exact upper
bound `Σ_ℓ P(ℓ)·C(ℓ)` is the probability-weighted walk over every root→leaf path
(`enumerate_forward_paths`).

### Within-node axis — per-node openings

A node also owns an opening set `Ω_n` — its inflow realizations. Today a node's
openings are bound one of two ways: a single external-library column
(`realization_id`/`scenario_id` present ⇒ `|Ω_n| = 1`), or the **stage's**
generated opening set shared by every generated node at that stage. A node
carrying its **own** distinct multi-opening weight distribution — different from
its siblings at the same stage — is not user-file data today. That per-node
opening-weight axis is a **deferred future feature** (returning with a per-node
Markov configuration), not an input a study supplies now. The within-node
enumeration restrictions and the two binding paths are detailed in
`docs/design/policy-graph-limitations.md`.

## Discount is a per-stage rate, not edge mass

The discount override is a **per-stage** quantity, declared on
`stages[].annual_discount_rate_override` and compounded into the per-stage
`cumulative_discount_factors` the stage templates consume.

A **per-edge** — hence potentially per-node — rate is **rejected under
`nodes[]`** by `check_edge_discount_override_under_nodes` (rule 42). The
cumulative discount factor is baked into each stage's template objective
coefficients, so a per-edge rate would force per-node templates. The edge
spelling stays legal in the chain dialect, where every edge from a stage shares
one factor.

## Mapping to SDDP.jl: explicit rate vs. discount-in-edge-mass

SDDP.jl and cobre make the same modeling distinction differently, and the
difference is exactly where discounting lives.

- **SDDP.jl folds discounting into edge mass.** A node's out-arc weights are
  **not required to sum to 1**; the deficit `1 − Σ_j w_{ij}` is the probability
  of _terminating_ (leaving the graph). Discounting is expressed through that
  same deficit: in a cyclic (infinite-horizon) graph, the **probability on the
  arc that completes a cycle equals the discount factor** — the cycle-closing arc
  carries weight `ρ`, so continuation costs enter the Bellman update multiplied by
  `ρ` per cycle. Probability and discount share one channel.

- **cobre keeps the two channels separate.** Out-edge weights are normalized to
  `Σ ≈ 1` per source (`normalize_out_edge_probabilities`), discounting is carried
  as an **explicit per-stage rate** (`cumulative_discount_factors`), and a
  trajectory _terminates at a leaf_ — a node with no out-edges at the final stage,
  with a mid-horizon leaf rejected by rule 38 — never through a probability
  deficit. Probability mass and the discount factor are validated and applied
  independently.

The two coincide for a finite horizon. An SDDP.jl graph that encodes a discount
as an out-arc probability deficit must therefore be **translated** — normalize
the edges to sum to 1 and set the per-stage discount rate — not transcribed
edge-for-edge.

## Standing behaviors that replaced retired validation rules

The move to a node-native graph removed a family of collapse-era structural
rules. Their intent survives as standing engine behavior, not as re-added
validation:

- **A node's realization is defined by its own pointer.** A node's `scenario_id`
  fixes its realization directly; there is no descendant-agreement or trunk
  constancy to enforce across a shared trunk. The only surviving pointer rule is
  the per-class bound (rule 37).
- **Output identity re-keys to the leaf node id.** Per-node outputs and the
  simulation summary key on the node id, so injectivity across realizations is
  automatic rather than validated.
- **Probability totals come from the normalized transitions.** The census
  enumerates the graph's leaves and `scenario_summary` keys by leaf node id, so
  library exhaustion is no longer a soundness condition — the totals are the
  normalized out-edge weights. Coverage is not a validation rule.
- **Genuine branching is admissible.** A node with multiple successors — including
  sibling successors that share a realization pointer — loads without error. The
  admissibility predicate and the duplicate-pointer / degenerate-branching
  rejections are gone; a structurally recombinable signature draws a warning
  (rule 40), not a rejection. `check_node_graph`'s surviving structural rules
  (well-formedness, every edge `t → t+1`, the pointer bound) are the whole gate.

Retired rule numbers in the Layer-5b catalog
(`crates/cobre-io/src/validation/semantic/mod.rs`) are **never reused** — a
retired number is marked as retired and left unoccupied, so a rule number always
denotes the same behavior across the project's life.

## References

- Policy-graph subset and convention notes — `docs/design/policy-graph-limitations.md`
- SDDP.jl documentation — <https://sddp.dev/stable/>
- SDDP.jl general policy graphs (arc weights, termination, discounting) —
  <https://sddp.dev/stable/guides/create_a_general_policy_graph/>
- SDDP.jl source — <https://github.com/odow/SDDP.jl>
