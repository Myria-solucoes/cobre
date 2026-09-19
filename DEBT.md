# DEBT register

## Entry Format

```
### DEBT-<nnn>
what:      one-line description of the shortcut/smell and where (file:symbol)
why-now:   the deliberate-prudent reason it was taken (or "inadvertent — found")
quadrant:  deliberate|inadvertent x prudent|reckless
interest:  what it slows / risks, and whether it compounds
payoff:    the concrete remediation
trigger:   the event that forces payoff
owner:     who closes this entry
blast:     files & consumers affected; tests that must pass
origin:    the plan and ticket that minted this entry
paid:      the ticket that paid it off (written only on retirement)
```

## Open

_None._

## Paid

### DEBT-001
what:      `fill_commitment_hold_box` (crates/cobre-sddp/src/lp/builder/state_box.rs) resolves the delivery-stage bound only for the anticipated-thermal ring slot freshly decided at the current stage; every other reachable slot for a plant whose own lead exceeds one stage (k_max > 1) — an already-decided, not-yet-matured in-flight commitment held via the ring's same-slot carry — defaults to [0, 0] instead of its own held delivery target's resolved bound.
why-now:   mirrors `fill_anticipated_columns`'s own per-stage resolution exactly, per the ticket's explicit "reuse the delivery-stage lookup, do not re-derive it" constraint; resolving every in-flight slot's currently-held target requires the ring's own interior/carry-row row-position machinery (`build_anticipated_slot_row_pos`, private to `lp/builder/layout.rs`, outside this ticket's declared file scope) and was out of scope for the box's first build.
quadrant:  deliberate x prudent
interest:  a future outgoing-state canonicalization seam that projects onto this box would clamp a legitimate, non-zero in-flight commitment down to [0, 0] for any k_max > 1 anticipated-thermal study — silently destroying committed generation state on every stage the ring is carrying rather than depositing or fishing. Does not compound today (no consumer reads `state_boxes` yet); becomes load-bearing the moment the read-back seam lands.
payoff:    extend `fill_commitment_hold_box` to resolve, for every `(slot, plant)` pair in `layout.commit_out`, the physical delivery target currently occupying that slot (mirroring `build_anticipated_slot_row_pos`'s ring-axis sweep: `r = stage_idx + depth + 1`, `m = point.physical_target(r)`, `is_deposit`/`is_ready_at` gating) rather than only the plant's `genuine_decisions_at(stage_idx)` slot, then read that target's `thermal_block_base`/post-study bound the same way.
trigger:   the outgoing-state canonicalization seam (read-back of `state_boxes`) is implemented against a k_max > 1 anticipated-thermal fixture, or a state-family audit exercises a k_max > 1 case.
owner:     cobre-sddp state-canonicalization plan owner
blast:     crates/cobre-sddp/src/lp/builder/state_box.rs (`fill_commitment_hold_box`); any test asserting a k_max > 1 study's in-flight (non-freshly-decided) commitment slot box.
origin:    plans/state-canonicalization, epic-01-admissible-state-box, ticket-002-statebox-build
paid:      ticket-002-statebox-build (`fill_commitment_hold_box` reworked to mirror `build_anticipated_slot_row_pos`'s ring-axis sweep, resolving every reachable slot — deposit and interior alike; regression test in `crates/cobre-sddp/tests/state_box_commitment.rs`)
