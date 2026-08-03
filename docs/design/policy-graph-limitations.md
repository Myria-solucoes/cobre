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

## References

- SDDP.jl policy graphs — <https://sddp.dev/stable/tutorial/first_steps/>
- SDDP.jl Markov chain approach — <https://sddp.dev/stable/tutorial/arma/>
- SDDP.jl source — <https://github.com/odow/SDDP.jl>
