# Cobre Commenting & Documentation Philosophy

> **Status:** Design proposal. Not yet a coding standard and not yet enforced.
> This document is written to be promoted, intact, into a path-scoped rule
> (`.claude/rules/comments.md`) plus a short CLAUDE.md pointer and a set of CI
> gates. See [§9 Promotion path](#9-promotion-path).
>
> **Scope of this document:** the philosophy and directive set only. It does
> **not** perform any cleanup of existing comments. The concrete debt it
> identifies (§7) is catalogued for a _future_, separately-approved pass.
>
> **Provenance note:** the file/line citations below were harvested from the
> tree and verified at the time of writing. They are _illustrative anchors_ for
> the rules, not a maintained index — by this document's own Durability
> principle (§4) they may drift. Trust the rule, re-locate the example.

---

## 0. The reframe — why this is not a "minimize comments" policy

The intuition behind this work was "clean code explains itself; our codebase is
over-commented and verbose; let's establish a minimalist philosophy." A full
audit of the codebase's comments contradicts that premise in a way that changes
the whole direction:

- **Signal-to-noise is uniformly high across every crate.** The audited risk is
  **over-deletion, not bloat.**
- **Classic AI/boilerplate bloat is nearly absent**: zero commented-out code
  anywhere; five `TODO`s in the whole tree — three in `crates/*/src` plus two
  plan-token `TODO(Epic 01 ticket-003)`s in a test and a bench (themselves §5
  violations, catalogued in §7); almost no "narrate the next line"
  comments; `SAFETY` comments exist only where `unsafe` is actually allowed.
- **The dominant comment species is load-bearing**: correctness contracts,
  `SAFETY` justifications, units/sign conventions, determinism invariants, and
  wire-format layouts. These are comments whose _plausible_ deletion reintroduces
  a **silent wrong-result bug that still compiles and passes most tests**.

A blanket "the code self-documents, cut the comments" pass would therefore do
**net harm** here. The correct philosophy is **contract-first**, and "clean"
means **truth-density**, not line count.

The original instinct is still right — but it applies to a **narrow, concentrated
debt** (§7), not to a general over-commenting problem. The remedy is _surgical_,
not a sweep.

---

## 1. The thesis and the one test

> **A comment must carry information the code cannot.** In Cobre, the dominant
> such information is a **contract**: an invariant whose obvious alternative
> compiles and is wrong.

### The Earned-Comment Test

Before writing **or keeping** any comment, ask: _does this tell the reader
something the code cannot, that they would get wrong without it?_

| If the comment…                                                | Verdict                               |
| -------------------------------------------------------------- | ------------------------------------- |
| states an invariant whose obvious alternative is wrong         | **KEEP** — Contract (Voice 1)         |
| explains a non-obvious _why_                                   | **KEEP** — Rationale (Voice 2)        |
| explains why an unused item exists / what will activate it     | **KEEP** — Intent/Seam (Voice 4)      |
| would be carried by a better name, type, or extracted function | **CONVERT** — refactor; don't comment |
| restates what the code already says                            | **DELETE**                            |
| narrates how the code got here                                 | **MOVE to git**                       |
| points at something that can drift (a line number, a hash)     | reference by **symbol**, or delete    |

---

## 2. The Four Voices

Every comment in shipped code must speak in one of the first three voices (Voice
4 is a tightly-bounded fourth). The Narrator voice is **not allowed** in shipped
code.

### Voice 1 — Contract (ALLOWED, the house style)

A present-tense invariant **+ the wrong-but-compiling alternative it forbids +**
a citation of the owning symbol. This is exactly the style of
`.claude/rules/sddp.md`; make it the house style everywhere, not just in SDDP.

> _Example (`crates/cobre-sddp/src/backward.rs`):_ the Benders subgradient is
> `rc_scaled / col_scale[col]` — **divided, not multiplied** — because the pin
> sets `v_scaled = v_orig / col_scale`. Stating the forbidden alternative
> (`* col_scale`) is what makes the comment load-bearing.

### Voice 2 — Rationale (ALLOWED)

_Why_ a non-obvious choice was made; what regresses if a maintainer "simplifies"
it. The `X instead of Y` form **is rationale** and is **kept whenever Y is a
still-plausible wrong simplification** a future editor might reach for.

> _Example (`crates/cobre-sddp/src/forward.rs`):_ "Welford's online algorithm is
> used instead of the two-pass naive formula to avoid catastrophic cancellation
> when sum_sq ≈ n \* mean^2." Two-pass is a plausible simplification, so the
> rationale earns its place.

### Voice 3 — ~~Narrator~~ (NOT ALLOWED in shipped code)

What the next line does; how we _used to_ do it; which ticket/phase/workstream
changed it; project-event dates ("discovered 2026-06"); byte-count deltas. This
information belongs in **names, git history, commit messages, or test names** —
never in a shipped comment.

### Voice 4 — Intent/Seam (ALLOWED — the only licensed future-tense exception)

A present-tense statement of **why a currently-unused item exists and what will
activate it** — the future caller, the structural symmetry, the migration tool
it serves — **pinned to the `#[allow(dead_code)]` / `#[allow(unused_*)]`** it
justifies. This is the sole exception to the "no future tense" reading of N1/N2,
**because the suppression attribute is itself the durable anchor**: when the
future caller lands, the lint fires and forces the comment current.

> _Exemplars:_ "Retained for the planned checkpoint-migration tool and exercised
> by this module's own tests" (`policy_load.rs`); "Pre-allocated here; read sites
> are not yet wired" (`workspace.rs`).

---

## 3. The highest-leverage rule: Intra-Comment Surgery

The audited risk is over-deletion, and one comment block frequently **welds
several voices together** — a contract sentence, a rationale clause, and a rot
tail, all in one paragraph. The rule:

> **Adjudicate clause-by-clause, never block-by-block. Amputate the rot tail;
> never delete the invariant line.**

Operational procedure:

1. Strip any parenthetical/clause matching a **plan token** or **drift-ref**
   (delete `(F1-007 fix)`, `(see MEMORY.md D15 note)`, `:1555`).
2. A determinism/correctness sentence is a **Contract clause** → always KEEP.
3. An `X instead of Y` clause is **Rationale** → KEEP when Y is a still-plausible
   wrong simplification.
4. The amputation target is the **rot token/tail**, never the invariant.

### Canonical worked example

Engineers split three ways on this real block (quoted verbatim from
`crates/cobre-sddp/src/forward.rs`) until the rule existed:

```rust
// BEFORE
// Canonical-order single-pass statistics. All ranks iterate global_costs in
// the same order, producing bit-identical statistics regardless of rank count.
// Welford's online algorithm is used instead of the two-pass naive formula to
// avoid catastrophic cancellation when sum_sq ≈ n * mean^2 (F1-007 fix).
// MPI Welford merge is not used here because the full gathered array is
// already available — a single sequential pass suffices.
```

- "All ranks iterate … bit-identical … regardless of rank count" =
  **Contract (determinism)** → keep.
- "Welford's online algorithm … instead of the two-pass naive formula …" =
  **Rationale** (two-pass is plausible) → keep; likewise the
  MPI-merge-not-needed sentence.
- `(F1-007 fix)` = **plan token**, a trailing tag on the _rationale_ clause →
  amputate the tag only.

```rust
// AFTER
// Canonical-order single-pass statistics. All ranks iterate global_costs in
// the same order, producing bit-identical statistics regardless of rank count.
// Welford's online algorithm is used instead of the two-pass naive formula to
// avoid catastrophic cancellation when sum_sq ≈ n * mean^2.
// MPI Welford merge is not used here because the full gathered array is
// already available — a single sequential pass suffices.
```

Other live cases of the same shape: `lower_bound.rs` (keep the NCS-patch
correctness contract, strip "see MEMORY.md D15 note", optionally name the
regression test); `convergence.rs` (keep the gap formula, strip the trailing
plan tag).

---

## 4. Principles (with the carve-outs that survived adversarial review)

### Durability — a shipped comment must stay true with zero maintenance

Reference things that **cannot rot**:

- a symbol name (`Mod::symbol`) or an intra-doc link (`[Symbol]`),
- a named regression **test**,
- a **stable external spec anchor** (e.g. `output-schemas.md §5.1`) **when the
  cited document lives in a declared source-of-truth root** (the `cobre-docs`
  methodology repo).

**Never** reference by `file.rs:NNN`, commit hash, dead in-repo path, or private
memory (`MEMORY.md`, `.claude/`). When a basename is ambiguous across crates,
**qualify it** (`clp.rs` lives in `cobre-solver`, not `cobre-sddp`).
**Symbol+line hybrids** (`Symbol at file.rs:1555`) are the common local form —
**keep the symbol, strip the `:NNN`.** If a fact can rot and cannot be made
un-rottable, delete it.

### Provenance — history lives in git, not in comments

Comments are present-tense; convert story-tails to the durable fact. If a
discovery points at a **still-living regression test or a deterministic
regression case** (e.g. `D06`/`D15`) that pins a contract, **name the
test/case** instead of narrating the discovery — this is exactly what the
gold-standard `sddp.md` does.

**Explicit carve-outs that are NOT history (KEEP):**

- bibliographic citation years — `Moro (1995)` is provenance-of-algorithm;
- calendar years/dates denoting **data coverage** or domain time windows
  (e.g. an inflow series spanning `1991–2019`) — these are data contracts.

### Contract-mirroring beats DRY

Deliberately duplicated contracts (producer vs consumer, rule-vs-code, book vs
rustdoc, rustdoc vs methodology spec) are **redundancy-with-purpose**; do not
DRY them away. But:

- A mirror restates the **shape** of a contract, **never a magic number**.
  Numbers live only in the single authoritative owner; mirrors reference the
  owner by symbol.
- **A drifted mirror is a lie, not a mirror** — fix it to match, or reduce it to
  the shape-only form. (This is the `cut/mod.rs` "24-byte header" vs the
  authoritative 25-byte `wire.rs` layout class.)

### Single-owner

Comments naming the **sole owner** of a byte layout, or the **single hot-path
entry** ("never bypass"), are load-bearing — keep and reinforce. A single-owner
byte table (`cut/wire.rs`) is explicitly out of reach of the
"don't duplicate a source of truth" rule even though it restates what the
serializer enforces.

### Length is not the metric

A 200-line module header duplicating a schema that is the source of truth is
noise; a 50-line passage that is the **only** place an invariant is explained is
signal. **Exception:** prose **compiled into** the source of truth — the
`schemars`-derived config/output rustdoc that becomes `config.schema.json` — is
not a duplicate of the schema, it **is** the schema; never delete it as
redundant, regardless of length.

---

## 5. The directive set

### DO

- **D1 — Contracts.** When the obvious alternative is wrong, say so: invariant +
  forbidden alternative + owning symbol.
- **D2 — SAFETY.** Every `unsafe` block carries a multi-clause `// SAFETY:`
  mapping each Rust-side invariant to the C precondition it satisfies.
  (Reinforces the workspace `unsafe_code = "forbid"` policy.)
- **D3 — Units.** Annotate fields/consts carrying physical units, dimensionless
  factors, a sign convention, or a coefficient-evaluation order — **and the
  inverse-direction trap** (e.g. "divided by `col_scale`, not multiplied").
- **D4 — Rationale above suppression.** Every `#[allow(...)]` for a
  _refactor-decision_ lint (`clippy::too_many_arguments`,
  `clippy::too_many_lines`, `clippy::type_complexity`, `dead_code`, `unused_*`)
  and every borrow-checker workaround carries a rationale explaining why the
  refactor that would remove the lint is inappropriate. The established
  `// Rationale:`, `// RATIONALE:`, and inline-trailing forms all count. For
  `dead_code`/`unused_*`, that rationale **is** the Intent/Seam voice (Voice 4).
  (Reinforces `architecture-rules.md`.)
- **D5 — Determinism.** Where solve/thread order is deliberately decoupled from
  aggregation order (stable sort after a parallel region, canonical iteration,
  Welford), say why — it upholds the declaration-order bit-determinism hard rule.

### DON'T

- **N1 — No what-narration.** Rename instead.
- **N2 — No history narration.** Ban `replaces`/`formerly`, byte-count deltas,
  commit hashes, **project-event dates** ("discovered 2026-06", "as of last
  sprint"), and plan tokens. KEEP the present-tense fact. **Carve-outs (not
  history):** bibliographic citation years `Author (YYYY)`; calendar/data-coverage
  years; a deterministic regression-case id (`D06`/`D15`) naming a still-existing
  reproducible case that pins a contract (same status as naming a living test).
- **N3 — No drift-prone refs.** Ban `file.rs:NNN` / `file.rs:NNN-MMM` (including
  en-dash and comma-chained `:N, :M` forms), commit hashes, dead in-repo paths,
  and `MEMORY.md` / `.claude/` paths. Reference by symbol, intra-doc link, or
  named test. **Stable external spec anchors** (`§x.y`) into a declared
  source-of-truth root are **not** drift-prone and are allowed (see Durability).
  For a symbol+line hybrid, strip the line and keep the symbol.
- **N4 — No plan/workstream leakage** in shipped code, `book/`, `CHANGELOG`, or
  **inline test/bench comments**. Ban `Epic`/`ticket`/`T0NN`/`sprint` and the
  workstream forms `F-NNN`, `FN-NNN` (e.g. `F1-007`, `F2-002`, `F3-004`), and
  `W-N` _in workstream context_. Plan
  tokens are never durable symbols: when a banned token is a **trailing tag** on
  a contract/rationale line, **amputate only the tag** and preserve the
  invariant/formula — never delete the line. Plan refs in **test names** remain
  allowed (per N2's name-the-test rule).
- **N5 — No banners fencing groups inside one long production function** —
  extract a function instead. Decorative dividers **between top-level items**, in
  `extern "C"` blocks (mirroring the C header), and in `#[cfg(test)]` modules are
  fine.
- **N6 — Don't duplicate a source-of-truth in prose** — but **preserve
  contract-mirroring** (§4). **Exceptions:** `schemars`-derived config/output
  rustdoc (it _is_ the artifact) and single-owner byte tables (`cut/wire.rs`)
  are out of reach.

### Special-case clauses

- **TODO/FIXME.** A shipped `TODO` MUST carry a durable behavioural parenthetical
  tag (`TODO(historical-replay-non-monthly)`) and SHOULD reference the guard/test
  that enforces the current limitation (exemplar: a `TODO` pinned to the
  `debug_assert!` it explains in `sampling/historical.rs`). A `TODO` MUST NOT
  carry a plan token (`TODO(Epic..)`, `TODO(ticket..)`). A bare ownerless `TODO`
  is discouraged: add a behavioural tag or convert to a tracked issue.

---

## 6. Enforcement layer (thin and mechanical; the rest is judgment)

**Most of this philosophy is judgment-only.** No grep distinguishes
`rc / col_scale, divided not multiplied` (keep) from `has moved to column bounds
(Phase 1)` (convert) — both mention behaviour; only a human/LLM reviewer can tell
the durable invariant from the migration tail. The mechanical layer is a narrow
backstop, governed by one overriding rule:

> **Over-deletion is the risk. Every gate reports the matched _token span_, never
> flags the whole line, and prints a "strip the rot, keep the invariant"
> instruction** so a hurried fix never amputates a contract.

| Gate   | Detects                                                                           | Enforcement                 | Mechanism & false-positive guard                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------ | --------------------------------------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **E1** | Plan/workstream leaks: `F-NNN`/`FN-NNN` (`F1`/`F2`/`F3`…), context-anchored `W-N` | **hard CI gate**            | Extend `scripts/check-no-plan-leaks.sh`. Add patterns `\bF[0-9]?-[0-9]{2,}\b` (the live debt is dominated by `F2-NNN`/`F3-NNN` tokens, not just `F1-NNN`) and the context-anchored `\bW[0-9]+ (reset\|rebake\|workstream\|phase)\b`. Add `crates/*/benches` and `crates/*/tests` to scan paths (and the unscanned `cobre-flow`/`uc`/`emt` + umbrella `cobre` src dirs). **Guard:** drop any bare `W-N` pattern (most `W[0-9]` hits are domain _week_ labels or vendored JS); require ≥2 trailing digits on `F[0-9]?-NNN` so unit tokens don't collide; exclude `*.min.js`. Report the matched span only; print the amputation instruction. Pilot-run first to allowlist legitimate ticket-encoding **test names**. |
| **E2** | Drift-prone source line refs (`file.rs:NNN`)                                      | **warning**                 | Advisory grep `[a-z_]+\.rs:[0-9]+([–—-][0-9]+)?` and comma-chained `:NNN` over `crates/*/src` `//` and `///` lines. **Guard:** warning, not hard — the form is trivially evaded and a hard line-flag risks deleting the paired symbol prose. Do the one-time cleanup (strip `:NNN`, keep the symbol) before enabling. Never match a bare `Symbol` ref or a path without a trailing `:NNN`; report only the `:NNN` span.                                                                                                                                                                                                                                                                                            |
| **E3** | Un-rottable-or-delete refs                                                        | **hard (a) + advisory (b)** | **(a) HARD:** forbid the literal token `MEMORY.md` and any `.claude/` path in shipped `//`/`///` comments. **(b) ADVISORY:** flag repo-relative dead paths under `docs/`, `artifacts/`, or `plans/` (gitignored — dead for any cloner) that don't resolve. **Guard — critical:** do **NOT** require "every `.md` resolves on disk" — the `cobre-docs` sibling repo is absent in CI, and the external spec references (`output-schemas.md`, etc.) are deliberate contract-mirroring into a declared methodology root, not rot. Advisory (b) fires only on repo-relative prefixes, never on external spec filenames.                                                                                                 |
| **E4** | Mandated rationale on refactor-decision suppressions                              | **hard on git-diff**        | grep/clippy: for each `#[allow(...)]` of a shape/dead-code lint in production code, require a justifying comment within a ≤4-line upward window (skipping doc/attribute-continuation lines) **or** a trailing inline comment. **Guard:** case-insensitive matcher (accepts `// RATIONALE:`); handle multi-line `#[allow(\n ... )]` blocks; `#[cfg(test)]` allows are out of scope. **Enforce on NEW/CHANGED allows only** with a tracked allowlist of pre-existing sites — a tree-wide hard gate would red-flag the legacy `too_many_lines` allows lacking a rationale (18–32 of the 37 production-scope sites, strictness-dependent; see §7).                                                                     |
| **E5** | Placeholder text in user-facing docs                                              | **hard CI gate**            | grep `\bTBD\b` (word-boundary) and "to be inserted" in `book/src` and rustdoc. **Guard:** exclude vendored `book/*.min.js` (the mermaid engine contains literal `Error("TBD")`). **Prerequisite:** fill the one real live placeholder (a pending benchmark value in the SDDP book chapter) _before_ wiring, so the gate's first action isn't to red-flag protected prose.                                                                                                                                                                                                                                                                                                                                          |
| **E6** | Missing round-trip / version-byte test on a dual-owned wire format                | **judgment-only checklist** | Not a bespoke script. comments.md lists each dual-owned format (`cut/wire.rs`, `policy/codec.rs`, `workspace.rs` `CapturedBasis`, `cut_sync`, `resolved_parameters`); each must have **both** a round-trip and a reject-old-version `#[test]`. Most already exist; add only the missing ones. Enforced by `cargo test` + review.                                                                                                                                                                                                                                                                                                                                                                                   |
| **E7** | Intra-function box-drawing banners                                                | **warning**                 | grep for box-drawing glyphs `U+2500..U+257F` in `//`/`///` comments in `crates/*/src`, excluding `#[cfg(test)]` tail blocks and `extern "C"` blocks (mirroring N5). **Guard:** comment lines only (not TUI/output string literals); the exact glyph range leaves ASCII `---` and prose en-dashes untouched. Pilot result: ~500 production hits remain even after both exclusions — almost all legitimate between-top-level-item dividers N5 allows, which no grep can separate from intra-function banners; stays advisory pending a sharper heuristic (e.g. indented `//` lines only).                                                                                                                            |

---

## 7. The actual debt (what a future cleanup would target)

For the record, since the philosophy is also meant to _characterise_ the
codebase: the debt is **not** "too many comments." It is three concentrated,
low-severity patterns plus a thin tail.

1. **Drift-prone references** — 8 `file.rs:NNN` comment refs, at least three
   already drifted (an `indexer.rs:1555` fact fanned into three files, a stale
   `training.rs` range in `workspace.rs`, stale `matrix.rs` anchors in
   `extraction.rs`), one dead-file pointer (`artifacts/layout-decision.md`),
   a dead `plans/lp-consistency-gap/` pointer in a committed test, and two
   private-memory citations (`MEMORY.md`).
2. **Plan-structure leakage** — 13 comment lines across 10 production files in
   `crates/*/src`, plus one in `book/src/crates/sddp.md`, carry `F-NNN` /
   `F1-NNN` / `F2-NNN` / `F3-NNN` / `W2 reset` tokens the current plan-leak
   gate's pattern misses; ~36 more sit in `#[cfg(test)]` modules,
   `crates/*/tests`, and `benches/` (including two `TODO(Epic 01 ticket-003)`s
   the existing pattern would already catch if those paths were scanned).
3. **Story-tails welded onto live contracts** — the Intra-Comment Surgery class
   (§3): keep the contract, amputate the tail.

Plus a thin tail of genuinely verbose rustdoc (a module-header schema catalogue
that duplicates per-writer schema tables).

**Healthy — leave alone:** `cobre-cli` (~90%+ load-bearing), the `.claude/rules`
and `architecture-rules` meta-docs (load-bearing by construction), and the
`cobre-io` `config/` rustdoc (it _is_ the public JSON schema via `schemars`).

**Dense-but-earned:** the SDDP hot path and algorithm core (the live home of the
cut-sign/scaling/determinism contracts) and the stochastic PAR/quantile math.

### Residual risks to accept knowingly

- **E4 backlog:** 18–32 of the 37 production-scope `too_many_lines` suppression
  sites carry no rationale (strictness-dependent; the other 111 of the 148 raw
  attribute sites in `crates/*/src` sit in `#[cfg(test)]` modules E4 excludes —
  the inventory's oft-quoted 79 is a single-line-only grep over a mixed scope).
  The git-diff + allowlist approach defers them; the allowlist must be actively
  burned down or it becomes a permanent escape hatch.
- **E1 `W-N` anchor is heuristic:** a future comment using a different verb
  ("after the W2 step") evades it. The durable fix is to **rename** the three
  "W2 reset" comments to behavioural language ("per-opening solver-state reset");
  the gate is a backstop, not a substitute.
- **E2 as a warning** relies on reviewer diligence for new bare line refs —
  accepted, because a hard line-flag gate's over-deletion hazard is worse.
- **Symbol+line hybrid stripping** assumes the symbol is still correct; if both
  symbol and line drifted, stripping the line leaves a stale symbol. Only a
  compiler-checked intra-doc link would catch that — a future hardening.

---

## 8. Relationship to existing rules (must not contradict)

This philosophy **builds on**, and must not contradict, the conventions already
in force:

- **`.claude/rules/sddp.md`** — the codebase's existing load-bearing-comment
  doctrine ("a _contract_, not a style preference … verify against the cited code
  before changing"). It is the **gold standard for the Contract voice**; this
  document generalises it to all crates, it does not re-explain it differently.
- **`.claude/architecture-rules.md`** — mandates the `// Rationale:` comment above
  any unavoidable `#[allow(clippy::too_many_arguments)]`. D4 reinforces this;
  treat suppression rationales as **required**, not bloat.
- **`CLAUDE.md` hard rules** — infra-crate genericity (no `sddp`/`SDDP`/`Benders`
  vocab in `cobre-core`/`io`/`solver`/`stochastic`/`comm` per the rule text; the
  enforcing `check-infra-genericity.sh` gate additionally bans standalone `cut`
  vocabulary), declaration-order bit-determinism, and the existing
  plan-structure ban
  (git commit messages are the allowed home for plan history). N4 **extends** the
  plan-structure ban's pattern coverage (`F-NNN`/`W-N`); it does not relax it.
- **The three CI gate scripts** (`check-infra-genericity.sh`,
  `check-cut-selection-determinism.sh`, `check-no-plan-leaks.sh`) — the new gates
  (§6) sit beside them and reuse the conventions established by
  `check-infra-genericity.sh` (its `EXCLUDED_FILES` array and `#[cfg(test)]`
  tail-block exclusion; the other two gates are plain pattern scans without
  those mechanisms, and `check-no-plan-leaks.sh` scans only 10 crates' `src/`
  trees — `cobre-flow`/`uc`/`emt`, the umbrella `cobre` crate, `tests/`, and
  `benches/` are unscanned).

---

## 9. Promotion path

To turn this design into an enforced standard:

1. **Promote the rule.** Lift §1–§5 (philosophy, voices, principles, directives)
   into a path-scoped **`.claude/rules/comments.md`** that auto-loads on
   `**/*.rs`, sibling to `sddp.md`. Add a one-line pointer under **CLAUDE.md →
   Hard Rules**. Keep `sddp.md` as the canonical _example_ of the Contract voice;
   `comments.md` is the _general_ rule it instantiates.
2. **Land the gates incrementally**, cheapest-and-safest first:
   E3(a) and E5 (after filling the one live placeholder), then E1 (after a
   pilot/allowlist run over `crates/` + `book/` + `benches/` + `tests/`), then E4
   as a git-diff gate. Ship E2 and E7 as **warnings**; treat E6 as a review
   checklist.
3. **Do the one-time surgical cleanup** (§7) as a _separate, approved_ change —
   not as part of adopting the philosophy.
4. **Burn down the E4 allowlist** over subsequent PRs so the backlog escape hatch
   closes.
