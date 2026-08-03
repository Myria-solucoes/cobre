# Policy Graph — Known Limitations and SDDP.jl Compatibility

The node-native engine models a policy graph — nodes, transition-weighted edges,
and a per-node value function — in the same family as
[SDDP.jl](https://github.com/odow/SDDP.jl). This document records where the current
implementation is a deliberate **subset of**, or **differs in convention from**, a
general Markovian policy graph, so the boundaries are explicit for anyone porting a
model from SDDP.jl or extending the engine.

## Compatible core

For a **finite, acyclic, leaf-terminated** graph the engine maps directly onto the
SDDP.jl policy-graph model:

- A node is a `(stage, state)` subproblem, and there may be more than one node per
  stage (the Markovian case: several discrete states at the same stage).
- Each non-leaf node owns its own value function — one future-cost pool per node,
  carrying that node's own Benders cuts.
- Edges carry conditional transition probabilities; the backward pass builds a
  node's cut by aggregating its successors' value functions weighted by those
  probabilities.
- The forward pass samples a root-to-leaf trajectory by following the transition
  probabilities.

The scenario-tree case — a single known initial state, interior branching, and
terminal leaves — is fully supported and is the primary target of the node engine.

## Known limitations (recorded for a future release)

### 1. Single initial node — no multi-state first stage

The engine requires exactly one node at the first stage. `find_root_position`
(`crates/cobre-sddp/src/training/lower_bound.rs`) rejects a first stage that holds
more than one node. A general Markovian graph can begin from an initial probability
distribution over several first-stage states (an uncertain initial regime); the
engine cannot represent that today.

Closing it needs a dedicated root source that carries the initial distribution
`P(root → n)` across the first-stage siblings, wired through the lower-bound,
forward, and backward passes. It is not required when the initial state is known,
which is the usual case (current storage and a known starting condition).

### 2. Single terminal value across terminal states

All terminal (leaf) nodes share one value-function pool — see `NodeRuntime` in
`crates/cobre-sddp/src/setup/node_graph.rs`: leaves share one pool id, while a node
with successors owns its own. Terminal nodes accumulate no cost-to-go, so the shared
pool is harmless for ordinary training.

It becomes a limitation only under terminal boundary-cut injection that needs a
**per-terminal-state** continuation value (for example a wetter versus drier ending
regime with a different water value): the shared pool holds one boundary-cut set for
all leaves. Closing it needs a per-leaf terminal pool on the boundary-injection path.

The shared stage template across leaves is correct while a state affects only the
inflow realization, not the LP structure — which holds for the hydrothermal model.

## Deliberate differences from SDDP.jl (by design, not limitations)

### Transition-probability convention

Out-edge probabilities are normalized to sum to 1 at load time
(`normalize_out_edge_probabilities`), and discounting is carried as a **separate
per-stage factor** (`cumulative_discount_factors`), never folded into the edge
weights. SDDP.jl instead permits a node's out-edges to sum to less than 1, with the
deficit encoding a discount factor or a termination probability. The two coincide
for a finite horizon; an SDDP.jl graph that uses the probability deficit as a
discount must be **translated** (normalize the edges, set the per-stage discount),
not transcribed edge-for-edge.

### Finite horizon only

The engine supports finite, acyclic, leaf-terminated graphs. It has no cyclic,
infinite-horizon (discounted-cycle) construction.

### Autocorrelation model

Inter-stage inflow dependence is modeled with a continuous periodic-autoregressive
lag state (the `state_space.inflow_lag_depth` configuration), not by discretizing
the process into a Markov lattice as SDDP.jl's "Markov chain approach" does. The
node graph is for **explicit discrete branching** (scenario trees, regime states);
the two compose — a branching graph whose nodes each carry the continuous lag state.

## Enumerated forward selection requires a structural tree of single-realization nodes

`training.selection = enumerated` computes the exact upper bound `Σ_ℓ P(ℓ)·C(ℓ)` by
walking every root→leaf path. It enumerates only the **structural** axis of the graph
— the node/edge branching — solving each node once per distinct incoming prefix, and
multiplying a single opening weight per node into the path probability
(`enumerate_forward_paths`, `crates/cobre-sddp/src/setup/node_graph.rs`). It does
**not** enumerate the other two ways a graph can carry randomness, and rejects them at
setup rather than compute a biased bound.

### Within-node openings must be singleton (`|Ω_n| = 1`)

A node also owns an opening set `Ω_n` (its inflow realizations). Because the engine
solves each node once and its probability walk multiplies one opening weight per node,
a node with `|Ω_n| > 1` would cover only a subset of scenarios, with weights that no
longer sum to one and a single-sample cost — an invalid bound. `setup` rejects
`|Ω_n| > 1` under enumerated training with a named error
(`reject_within_node_opening_enumeration`, `crates/cobre-sddp/src/setup/mod.rs`).
Express the stochasticity structurally (one realization per node) or use sampled
selection.

### The graph must be a tree (single predecessor per node)

A node reached from two different parents — a recombination join, i.e. the
dense-transition-matrix Markov case — has a different incoming continuous state per
incoming prefix. The engine reconstructs that state from a single-predecessor map
(`build_parent_map`), so a recombining node would be solved once under one arbitrarily
chosen parent's state while paths arriving via the other parent silently read the
wrong state — an invalid bound in a release build. `setup` rejects a node with
in-degree ≥ 2 under enumerated training with a named error
(`reject_recombining_node_enumeration`). Full recombination support — per-prefix state
reconstruction for a multi-parent node — is a reserved seam, and is what a genuine
Markovian graph with a dense transition matrix would need under enumerated selection.

### Sampled selection has neither restriction

`training.selection = sampled` Monte-Carlo draws one opening per visited node and
carries per-trajectory state, so it handles both `|Ω_n| > 1` and recombining graphs
natively. It is the general mode; enumerated is the exact-bound specialization for
small structural trees (scenario trees, the extensive-form oracle fixtures). A
multi-opening chain has `Π|Ω|` enumerated scenarios and is intractable to enumerate
regardless.

### How a node's openings are bound (both modes)

A declared node's opening set is either a single external-library column
(`realization_id` present → `|Ω_n| = 1`) or the **stage's** generated opening set
(`realization_id` absent → `|Ω_n| = n_openings(stage)`), the latter shared by every
generated node at that stage. So a node with its own distinct multi-opening
distribution — different from its siblings at the same stage — is not yet expressible:
siblings either share the stage's generated set or are pinned to single external
columns (`build_declared_node_graph`, `crates/cobre-sddp/src/setup/node_graph.rs`).

## References

- SDDP.jl policy graphs — <https://sddp.dev/stable/tutorial/first_steps/>
- SDDP.jl Markov chain approach — <https://sddp.dev/stable/tutorial/arma/>
- SDDP.jl source — <https://github.com/odow/SDDP.jl>
