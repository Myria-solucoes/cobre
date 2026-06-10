---
paths:
  - "**/*.rs"
---

# Cobre Comment & Documentation Rules

Governs every comment written in any `.rs` file in this workspace. The rule
auto-loads on the `**/*.rs` glob — including infra crates (`cobre-core`,
`cobre-io`, `cobre-solver`, `cobre-stochastic`, `cobre-comm`) that are bound by
the genericity hard rule. Keep all directive statements free of algorithm names.

For the canonical worked example of the Contract voice applied to numerical
algorithms, see `.claude/rules/sddp.md`.

---

## 1. The Earned-Comment Test

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

Every comment in shipped code speaks in one of the first three voices. Voice 4
is a tightly-bounded fourth. The Narrator voice is **not allowed** in shipped code.

### Voice 1 — Contract (ALLOWED, the house style)

A present-tense invariant **+ the wrong-but-compiling alternative it forbids +**
a citation of the owning symbol. `.claude/rules/sddp.md` is the gold-standard
exemplar — apply the same discipline to all crates, not just algorithmic ones.

> _Worked exemplar (algorithmic hot path in `crates/cobre-sddp/src/backward.rs`):_
> the subgradient coefficient is `rc_scaled / col_scale[col]` — **divided, not
> multiplied** — because the pin sets `v_scaled = v_orig / col_scale`. Stating the
> forbidden alternative (`* col_scale`) is what makes the comment load-bearing.

### Voice 2 — Rationale (ALLOWED)

_Why_ a non-obvious choice was made; what regresses if a maintainer "simplifies"
it. The `X instead of Y` form **is rationale** and is **kept whenever Y is a
still-plausible wrong simplification** a future editor might reach for.

> _Worked exemplar (`crates/cobre-sddp/src/forward.rs`):_
> "Welford's online algorithm is used instead of the two-pass naive formula to
> avoid catastrophic cancellation when sum_sq ≈ n \* mean^2."
> Two-pass is a plausible simplification, so the rationale earns its place.

### Voice 3 — ~~Narrator~~ (NOT ALLOWED in shipped code)

What the next line does; how it _used to_ work; which ticket/phase/workstream
changed it; project-event dates; byte-count deltas. This information belongs in
**names, git history, commit messages, or test names** — never in a shipped comment.

### Voice 4 — Intent/Seam (ALLOWED — the only licensed future-tense exception)

A present-tense statement of **why a currently-unused item exists and what will
activate it** — the future caller, the structural symmetry, the migration tool it
serves — **pinned to the `#[allow(dead_code)]` / `#[allow(unused_*)]`** it
justifies. This is the sole exception to the "no future tense" reading of N1/N2,
because the suppression attribute is itself the durable anchor: when the future
caller lands, the lint fires and forces the comment current.

> _Exemplar:_ "Pre-allocated here; read sites are not yet wired" (`workspace.rs`).

---

## 3. Intra-Comment Surgery

One comment block often **welds several voices together** — a contract sentence,
a rationale clause, and a rot tail, all in one paragraph.

> **Adjudicate clause-by-clause, never block-by-block. Amputate the rot tail;
> never delete the invariant line.**

Operational procedure:

1. Strip any parenthetical/clause matching a **plan token** or **drift-ref**
   (e.g. delete trailing `(F1-007 fix)`, `(see MEMORY.md note)`, `:1555`).
2. A determinism/correctness sentence is a **Contract clause** → always KEEP.
3. An `X instead of Y` clause is **Rationale** → KEEP when Y is a still-plausible
   wrong simplification.
4. The amputation target is the **rot token/tail**, never the invariant.

### Canonical worked example

Engineers split three ways on this real block (from
`crates/cobre-sddp/src/forward.rs`) until this rule existed:

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
  **Rationale** (two-pass is plausible) → keep; likewise the MPI-merge sentence.
- `(F1-007 fix)` = **plan token**, a trailing tag on the rationale clause →
  amputate the tag only.

```rust
// AFTER (matches live forward.rs — WelfordAccumulator symbol is the durable anchor)
// Canonical-order single-pass statistics. All ranks iterate global_costs in
// the same order, producing bit-identical statistics regardless of rank count.
// Welford's online algorithm is used instead of the two-pass naive formula to
// avoid catastrophic cancellation when sum_sq ≈ n * mean^2.
// MPI Welford merge is not used here because the full gathered array is
// already available — a single sequential pass suffices.
```

---

## 4. Principles

### Durability — a shipped comment must stay true with zero maintenance

Reference things that **cannot rot**:

- a symbol name (`Mod::symbol`) or an intra-doc link (`[Symbol]`),
- a named regression **test**,
- a **stable external spec anchor** (e.g. `output-schemas.md §5.1`) when the
  cited document lives in a declared source-of-truth root (the `cobre-docs`
  methodology repo).

**Never** reference by `file.rs:NNN`, commit hash, dead in-repo path, or private
memory (`MEMORY.md`, `.claude/`). When a basename is ambiguous across crates,
qualify it. **Symbol+line hybrids** (`Symbol at file.rs:NNN`) are common —
**keep the symbol, strip the `:NNN`.** If a fact can rot and cannot be made
un-rottable, delete it.

### Provenance — history lives in git, not in comments

Comments are present-tense; convert story-tails to the durable fact. If a
discovery points at a **still-living regression test or a deterministic
regression case** that pins a contract, **name the test/case** — this is exactly
what `.claude/rules/sddp.md` does.

**Explicit carve-outs that are NOT history (KEEP):**

- bibliographic citation years — `Author (YYYY)` is provenance-of-algorithm;
- calendar years/dates denoting **data coverage** or domain time windows —
  these are data contracts.

### Contract-mirroring beats DRY

Deliberately duplicated contracts (producer vs consumer, rule vs code, rustdoc
vs methodology spec) are **redundancy-with-purpose**; do not DRY them away. But:

- A mirror restates the **shape** of a contract, **never a magic number**.
  Numbers live only in the single authoritative owner; mirrors reference the
  owner by symbol.
- **A drifted mirror is a lie, not a mirror** — fix it to match, or reduce it to
  the shape-only form.

### Single-owner

Comments naming the **sole owner** of a byte layout, or the **single hot-path
entry** ("never bypass"), are load-bearing — keep and reinforce. A single-owner
byte table is explicitly out of reach of the "don't duplicate a source of truth"
rule even though it restates what the serializer enforces.

### Length is not the metric

A 200-line module header duplicating a schema that is the source of truth is
noise; a 50-line passage that is the **only** place an invariant is explained is
signal. **Exception:** prose **compiled into** a source of truth — `schemars`-
derived config/output rustdoc that becomes a `.schema.json` artifact — is not a
duplicate of the schema, it **is** the schema; never delete it as redundant,
regardless of length.

---

## 5. The Directive Set

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
  (Reinforces `.claude/architecture-rules.md`.)
- **D5 — Determinism.** Where solve/thread order is deliberately decoupled from
  aggregation order (stable sort after a parallel region, canonical iteration,
  online-algorithm accumulator), say why — it upholds the declaration-order
  bit-determinism hard rule.

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
  source-of-truth root are not drift-prone and are allowed (see Durability). For
  a symbol+line hybrid, strip the line and keep the symbol.
- **N4 — No plan/workstream leakage** in shipped code, `book/`, `CHANGELOG`, or
  **inline test/bench comments**. Ban `Epic`/`ticket`/`T0NN`/`sprint` and the
  workstream forms `F-NNN`, `FN-NNN` (e.g. `F1-007`, `F2-002`, `F3-004`), and
  `W-N` _in workstream context_. Plan tokens are never durable symbols: when a
  banned token is a **trailing tag** on a contract/rationale line, **amputate
  only the tag** and preserve the invariant/formula — never delete the line. Plan
  refs in **test names** remain allowed (per N2's name-the-test rule).
- **N5 — No banners fencing groups inside one long production function** —
  extract a function instead. Decorative dividers **between top-level items**, in
  `extern "C"` blocks (mirroring the C header), and in `#[cfg(test)]` modules are
  fine.
- **N6 — Don't duplicate a source-of-truth in prose** — but **preserve
  contract-mirroring** (§4 above). **Exceptions:** `schemars`-derived
  config/output rustdoc (it _is_ the artifact) and single-owner byte tables are
  out of reach.

### Special-case clauses

- **TODO/FIXME.** A shipped `TODO` MUST carry a durable behavioural parenthetical
  tag (`TODO(historical-replay-non-monthly)`) and SHOULD reference the guard/test
  that enforces the current limitation. A `TODO` MUST NOT carry a plan token
  (`TODO(Epic..)`, `TODO(ticket..)`). A bare ownerless `TODO` is discouraged:
  add a behavioural tag or convert to a tracked issue.

---

## E6 — Dual-owned wire formats checklist

The following formats are **dual-owned**: a serializer owns the byte layout and
one or more callers depend on exact round-trip fidelity and forward/backward
compatibility. Each must carry **both** tests:

1. A **round-trip `#[test]`** — serialise → deserialise → assert equality.
2. A **reject-old-version `#[test]`** — feed a byte payload with a stale version
   byte and assert the decoder returns an error (not silent corruption).

| Format                         | Authoritative owner symbol                                                          |
| ------------------------------ | ----------------------------------------------------------------------------------- |
| `cut/wire.rs`                  | `CutRecord::to_bytes` / `CutRecord::try_from_bytes`                                 |
| `policy/codec.rs`              | policy encode/decode entry points in `policy/codec.rs`                              |
| `workspace.rs` `CapturedBasis` | `CapturedBasis::to_broadcast_payload` / `CapturedBasis::try_from_broadcast_payload` |
| `cut_sync`                     | `cut_sync` serialisation entry points                                               |
| `resolved_parameters`          | `resolved_parameters` encode/decode entry points                                    |

Enforced by `cargo test` + review. Not a bespoke CI script.
