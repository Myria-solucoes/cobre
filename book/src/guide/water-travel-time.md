# Water Travel Time

When two hydro plants in a cascade are far enough apart that water released
upstream takes real time to reach the downstream reservoir, treating that
release as arriving instantly is a modeling simplification — one that is fine
when the delay is small relative to a stage's length, and increasingly wrong
as the delay grows toward a stage or more. `travel_time_hours` lets you
declare that delay explicitly, so Cobre carries the released-but-not-yet-
arrived volume forward as state instead of crediting it to the downstream
reservoir the moment it leaves the upstream plant.

`travel_time_hours` is a scalar, in hours, set on the **upstream** hydro
plant: it describes how long it takes for that plant's release to reach the
plant named in its own `downstream_id`. It applies only to that one cascade
arc. A diversion channel's flow and a pumping station's flow are not delayed
by this field — they still move within the same stage regardless of what
`travel_time_hours` is set to. Declaring a delay on a diversion or pumping
arc is not supported; if you need delayed transport on those paths, model it
outside Cobre for now.

## Declaring a cascade arc's travel time

Set `travel_time_hours` on the upstream plant's entry in `system/hydros.json`,
alongside the `downstream_id` it already carries:

```json
{
  "id": 2,
  "name": "Upper Plant",
  "downstream_id": 1,
  "travel_time_hours": 360.0
}
```

Leaving the field absent, or `null`, keeps the arc instantaneous — the
behavior every case had before this field existed. Setting it to `0.0` also
keeps the arc instantaneous; Cobre treats an explicit zero as "no delay
declared" rather than as a one-instant delay, and records an advisory when it
sees this so a `0.0` left over from an edit doesn't go unnoticed. Only a
finite value greater than zero declares a delayed arc. A `downstream_id` of
`null` (a tailwater plant, whose outflow leaves the system) has nothing
downstream to deliver a delayed release into, so `travel_time_hours` has no
effect there.

## The k-factor intuition: spreading a release over its arrival window

A stage is not an instant — it is a span of hours, and the plant's release
happens throughout that whole span, not at one moment. Cobre treats a stage's
release as spread uniformly across the stage's hours: think of it as a
continuous trickle rather than a single dump of water at the stage boundary.

Delay every drop of that trickle by `travel_time_hours` and its arrival times
shift forward by exactly that much, without changing shape: a release spread
uniformly over the stage's `[0, h)` window arrives spread uniformly over the
shifted window `[t_v, t_v + h)`, where `t_v` is `travel_time_hours` and `h` is
the stage's length in hours. That arrival window can span more than one
future stage, depending on how those stages are actually laid out on the
calendar — so Cobre finds out how much of the arrival window falls into each
future stage by intersecting it against the real, possibly non-uniform,
sequence of stage lengths that follow.

The result is a set of fractions, one per stage the arrival window touches:
`k_0` is the share that lands back in the release's own stage (part of the
arrival window can still fall before that stage ends), and `k_1, k_2, …` are
the shares landing one, two, or more stages later. They always sum to `1.0` —
every unit of released water is accounted for somewhere, never lost or
duplicated in the accounting. The deepest stage index with a nonzero share is
how many future stages this arc's release can still be arriving into; Cobre
keeps that much in-transit state per downstream plant, so a future stage can
receive its share when the time comes.

### Worked example: a monthly stage followed by weekly stages

Suppose an upstream plant declares `travel_time_hours: 360.0` (15 days), its
release happens during a stage spanning one 30-day month (720 hours), and the
three stages that follow are 7-day weeks (168 hours each) rather than more
months.

A guess based only on the release stage's own length might reason: 15 days is
half of a 30-day month, so the water arrives, at most, partway into the very
next stage. That guess is wrong, and wrong in a way that matters: it silently
drops water that really does arrive several stages later.

Intersecting the true arrival window, `[360, 1080)` hours, against the actual
calendar that follows gives:

| Stage                 | Length (h) | Overlap with `[360, 1080)` | Share (`k`)   |
| --------------------- | ---------- | -------------------------- | ------------- |
| Release stage (month) | 720        | `[360, 720)` = 360 h       | `k_0 = 0.50`  |
| +1 stage (week 1)     | 168        | `[720, 888)` = 168 h       | `k_1 = 0.233` |
| +2 stages (week 2)    | 168        | `[888, 1056)` = 168 h      | `k_2 = 0.233` |
| +3 stages (week 3)    | 168        | `[1056, 1080)` = 24 h      | `k_3 = 0.033` |

Half of the month's release (water released in the month's first half)
arrives before the month ends — that is `k_0`. The other half, released
later in the month, arrives after the month boundary, and because the
calendar turns from 30-day resolution to 7-day resolution right at that
boundary, the same fixed 15-day delay now spans three weekly stages instead
of one: full week 1, full week 2, and a 24-hour sliver of week 3. The true
depth is **three** stages beyond the release stage — deeper than the "half a
month, so about one stage" guess, and deeper than a fixed formula like
`round(travel_time_hours / stage_length)` would give applied to the release
stage's own 720-hour length. Cobre reaches this depth by walking the actual
calendar, so it is exactly as deep as the real stage lengths require, whether
the calendar coarsens, stays uniform, or refines after the release stage.

## Seeding water already in transit at study start

A positive `travel_time_hours` means water released _before_ the study began
can still be in transit when the study starts — the study's first stage
needs to know how much, and when it was released, to seed the in-transit
state correctly. That history is supplied through `past_defluences` in
`initial_conditions.json`, one entry per upstream hydro that declares an arc:

```json
{
  "past_defluences": [
    { "hydro_id": 2, "values_m3s": [600.0, 500.0], "season_ids": [3, 2] }
  ]
}
```

`values_m3s` is ordered from most recent (index `0`) to oldest, in m³/s — the
same pre-study-period convention `past_inflows` already uses. `season_ids` is
optional, one entry per value, for temporal validation against declared
seasons.

Cobre works out how many of the most-recent pre-study periods are needed to
fully cover the arc's travel-time window before the study starts, and
requires at least that many entries in `values_m3s` for that hydro. If
`past_defluences` for a hydro is absent or shorter than required, Cobre falls
back to that hydro's `past_inflows` history instead — treating past inflow as
a stand-in for past release — provided `past_inflows` is long enough, and
logs an advisory naming the hydro and asking for `past_defluences` to be
supplied directly for better first-stage accuracy. If neither history is long
enough, the case is rejected outright, naming how many periods are missing:
Cobre never silently seeds an incomplete in-transit state.

## Parallel vs. chronological attribution

How deep the in-transit state needs to be — the depth worked out above — is a
property of the stage calendar alone; it does not depend on
[block mode](./block-modes.md). What block mode changes is _which_ release
column feeds that state and in what proportion, because the two modes differ
in how much they know about when, within a stage, a plant actually released.

In **parallel** mode, a stage has a single water balance and a single release
value per plant, so every block in the stage is treated identically: each
block contributes the same `k_0`/`k_d` shares toward the same-stage balance
and the shared in-transit state, matching the uniform-release assumption
above exactly, because parallel mode has nothing finer to go on.

In **chronological** mode, each block has its own release columns and its own
position on the stage's internal calendar, so Cobre computes each block's own
arrival window and attributes that block's release only to the block(s) — in
this stage or a future one — its delay actually reaches. A block whose
release arrives entirely within the same stage contributes nothing to the
shared in-transit state; a block near the end of the stage can contribute all
of its release to it. For a stage split into three 240-hour blocks with a
250-hour travel time, for instance: block 1's release arrives almost entirely
in block 2 (`230/240`) with a small remainder (`10/240`) reaching into the
next stage; block 3's release arrives entirely in the next stage. Averaged
across the three blocks, that reproduces the same `250/720 ≈ 34.7%`
stage-level share that parallel mode deposits uniformly from every block.

Either way, the deposit lands in the **same** in-transit state — the same
downstream plant, the same lag slots. Chronological mode only refines which
release column the deposit is drawn from and in what fraction; it never adds
state that parallel mode does not have. When a stage has a single block
spanning its whole duration, chronological mode has nothing left to refine:
its computation reduces to exactly the parallel one.

## The terminal-drop caveat

A study has a finite horizon, and Cobre assigns no value to anything still
"in the pipe" past the very last stage. If a release's arrival window reaches
past the study's last stage, the portion of it that would have arrived after
the horizon ends is dropped — not carried forward, and not credited to any
terminal value.

In practice, this means a release made close to the end of the horizon is
undervalued relative to what it would be worth if the study continued: the
plant gets no downstream credit for water that would only arrive after the
study ends. This is a documented imprecision in how the finite horizon is
handled, not a defect to work around case by case — extend the study horizon
past the travel times you care about if end-of-horizon accuracy matters for a
particular study, and read results from the last few stages of any study with
a declared arc with this in mind.

---

## Related Pages

- [Hydro Plants](./hydro-plants.md) — cascade topology (`downstream_id`) and
  reservoir modeling
- [Block Modes](./block-modes.md) — parallel vs. chronological block
  resolution, the distinction this page's attribution section builds on
- [Case Directory Format](../reference/case-format.md) — `system/hydros.json`
  and `initial_conditions.json` field reference
