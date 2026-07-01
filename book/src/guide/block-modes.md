# Block Modes

Every stage in a Cobre study is divided into **blocks** — load levels, or
_patamares_, that partition the stage's duration. A monthly stage might carry
three blocks (peak, medium, off-peak load), each with its own duration in
hours; a stage modeling daily granularity inside a month might carry thirty
blocks, one per day. Every entity that dispatches power or water — thermals,
hydros, contracts, non-controllable sources — contributes one decision per
block, so a three-block stage already gives the LP room to vary generation
and flow within the stage.

What blocks do **not** automatically give you is a reservoir that evolves
within the stage. That behavior is controlled by the stage's **block mode**,
and it is the subject of this page.

---

## Parallel mode (the default)

In parallel mode, all of a stage's blocks share **one** water balance per
hydro plant. The reservoir has a single incoming storage `S⁰` at the start of
the stage and a single outgoing storage `Sᴷ` at the end of the stage; nothing
is modeled in between. Turbined flow, spillage, and generation are still
per-block — a hydro plant can turbine more water in the peak block than in
the off-peak block — but every block's flow feeds the same stage-total water
balance row. The stage's inflow is a single rate applied across the whole
stage.

This is the right model when within-stage storage dynamics are not the
question you are asking: the reservoir's trajectory across stages (wet season
into dry season, year over year) is what the optimization is solving for, and
the hours within a stage are a load-shape detail that does not need its own
storage path.

## Chronological mode

In chronological mode, a stage's blocks happen **sequentially**, and storage
evolves block-by-block: `S⁰ → S¹ → S² → … → Sᴷ`. Block `b`'s water balance
starts from `Sᵇ`, applies that block's turbined flow, spillage, evaporation,
and share of the stage's inflow, and produces `Sᵇ⁺¹`. The stage's incoming
storage is still `S⁰` (the first block's start) and its outgoing storage is
still `Sᴷ` (the last block's end) — chronological mode does not change what
the stage boundary means, only what happens between the boundaries.

Choose chronological mode when within-stage cycling is the behavior you need
to see: a reservoir that draws down during the day's peak hours and refills
overnight, for example, inside a stage whose boundary is a full month. Under
parallel mode that daily cycle is invisible — the LP only ever sees the
month's starting and ending storage. Under chronological mode, each block's
own storage boundaries appear in the LP and in the simulation output.

Chronological mode raises the per-stage LP cost relative to parallel mode.
The mechanism: chronological mode adds an interior storage column for every
block boundary beyond the two stage endpoints, and it evaluates the
production function (FPHA) and evaporation independently for each block
rather than once per stage. A stage with more blocks or more hydro plants
pays a correspondingly larger LP per stage.

### Worked example

Suppose a stage models one month and is divided into 30 daily blocks (one per
calendar day) to capture daily storage cycling at a hydro plant with
significant intra-month drawdown and refill.

- Under **parallel** mode, the LP has one water-balance row per hydro for the
  whole stage. All 30 blocks' turbined flow and spillage feed that single
  row; the reservoir's storage before and after the month (`S⁰` and `S³⁰`,
  using the block count as the boundary index) is all the LP sees of storage.
- Under **chronological** mode, the LP has 30 water-balance rows per hydro,
  one per day, chained `S⁰ → S¹ → … → S³⁰`. Day 5's water balance consumes
  day 5's inflow share and day 4's ending storage, and produces day 5's ending
  storage `S⁵`, which day 6 then consumes. The LP can now represent a
  reservoir that draws down through the work week and refills over the
  weekend, entirely within one stage.

## Selecting the mode

The block mode is a **per-stage** setting: `block_mode: "chronological"` (or
the default, `"parallel"`) on a stage entry in `stages.json`. See
[Case Directory Format](../reference/case-format.md) (`### stages.json`) for
the field's exact position and validation rules. Different stages in the same
study may use different modes — a coarse annual horizon in parallel mode
followed by a fine-grained near-term horizon in chronological mode is a valid
configuration.

## What changes in simulation output

The `hydros` simulation output reports `storage_initial_hm3` and
`storage_final_hm3` on every `(stage, block, hydro)` row, along with the
per-block evaporation columns. In parallel mode, every block row within a
stage repeats the same pair, `(S⁰, Sᴷ)` — the stage-level boundary, unchanged
across blocks. In chronological mode, each block row reports that block's own
boundary pair, `(Sᵇ, Sᵇ⁺¹)`, so the columns become genuinely block-resolved:
reading down the rows of one stage traces the reservoir's storage path
through the stage. The evaporation columns follow the same pattern — a block's
own evaporation reading in chronological mode, the stage-level reading
repeated in parallel mode. No columns are added or removed; only the values
they carry change. See [Output Format](../reference/output-format.md)
(`### simulation/hydros/`) for the full column reference.

## Coarse-train, fine-simulate

A trained policy is a set of Benders cuts over the state vector — storage,
inflow lags, and anticipated thermal state — evaluated at each stage's
incoming boundary, `S⁰`. That state vector, and therefore the cut's byte
representation, does not depend on how many blocks a stage has or which block
mode produced it. A consequence follows directly: a policy trained under one
block mode loads and applies under the other. You can train a policy with
stages in parallel mode — cheaper per stage, since there is no interior
storage path to solve for — and then simulate it with the same stages
switched to chronological mode, to see finer within-stage dynamics play out
under the trained policy. You can also train directly in chronological mode
if you want the training itself to account for within-stage cycling, at the
higher per-stage cost described above.

Read this workflow for what it is: a supported way to combine a cheaper
training run with a finer simulation, not a claim that the two modes solve
the same problem. For two or more blocks per stage, parallel and
chronological are genuinely different subproblems — chronological adds
interior storage-path constraints and per-block production-function
evaluation that parallel does not have. A cut trained under parallel dynamics
is a valid lower approximation of the cost-to-go **under those training
dynamics**; simulating it under chronological dynamics evaluates that same
cut against a finer model, not an identical one. Nothing about cross-mode
loading tells you that the parallel-trained policy is a better or worse fit
for chronological dynamics than a policy trained directly in chronological
mode — that is a question you answer by comparing simulation results, not one
the loading mechanism answers for you.

The block mode used during training is recorded in `policy/metadata.json`
(`training_block_mode`, and `training_block_mode_per_stage` when a study mixes
modes across stages), so a policy checkpoint carries its own training
provenance. See [Output Format](../reference/output-format.md)
(`### policy/metadata.json`) for the field reference.

---

## Related Pages

- [Case Directory Format](../reference/case-format.md) — `stages.json` schema,
  including the `block_mode` field
- [Output Format](../reference/output-format.md) — `simulation/hydros/` and
  `policy/metadata.json` schemas
- [System Modeling](./system-modeling.md) — how blocks fit into the broader
  case structure
- [Hydro Plants](./hydro-plants.md) — reservoir, production function, and
  evaporation modeling
