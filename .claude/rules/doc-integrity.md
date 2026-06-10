---
paths:
  - "book/**/*.md"
  - "*.md"
  - "CONTRIBUTING.md"
  - "CHANGELOG.md"
  - ".claude/*.md"
  - ".claude/rules/*.md"
---

# Cobre Prose Documentation Integrity Rules

Governs every Markdown file that serves as a user-facing or agent-facing artifact:
`book/`, `CLAUDE.md`, `.claude/rules/*`, `CONTRIBUTING.md`, `CHANGELOG.md`,
and root-level `*.md` files. The rule auto-loads on the matching globs.

For the code-comment counterpart (Four Voices, Earned-Comment Test, directives
D1–D5 / N1–N6), see `.claude/rules/comments.md`. That file governs `.rs` files;
this one governs prose.

---

## 1. Scope matrix — how far each concern transfers, per doc

Transfer strength tracks **(machine reader) × (low teaching mandate)**.

| Doc                                          | Reader                                                 | Transfer profile                                                                                                                                                 |
| -------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CLAUDE.md**                                | an agent that acts on assertions, no human in the loop | **near-total**: Durability/anti-drift dominates; Plan-leakage (it defines the rule); Provenance (clean). Voice machinery N/A.                                    |
| **`.claude/rules/*`, architecture-rules.md** | agent + maintainer                                     | **near-total**: machine-consumed contracts; same as code-adjacent docs.                                                                                          |
| **CHANGELOG / THIRD_PARTY_NOTICES / CoC**    | release / license / community                          | **strong** for Plan-leakage; Provenance **inverted** (CHANGELOG _is_ history — the carve-out); Durability adapted (NOTICES is a maintained single-owner mirror). |
| **CONTRIBUTING.md**                          | external contributors                                  | **adapted**: every path/command/flag must resolve; Self-executable-instruction applies; teaching repetition is signal, not bloat.                                |
| **README.md**                                | newcomers / evaluators                                 | **adapted**: Plan-leakage full; Durability in spirit; narrative/promotional voice is the job (Length N/A).                                                       |
| **book/**                                    | end users (modelers)                                   | **adapted**: executable-claim integrity dominates the real debt; teaching mandate supersedes Length/Voice.                                                       |

### Which of the six concerns transfer

- **Transfer with force (and bite harder than in code):**
  - **Durability / anti-drift** — re-expressed for prose (see §2, the key adaptation).
  - **Plan-leakage** — already a CLAUDE.md hard rule for user-facing artifacts;
    `comments.md` N4 extends the pattern.
  - **Contract-mirroring** — "a drifted mirror is a lie"; prose mirrors many
    code-owned facts and is the densest drift source.
- **Transfers adapted:**
  - **Provenance** — present-tense rule holds, **except** the CHANGELOG, which is
    history by design (the standing carve-out). Forward-looking "planned" status
    prose is a legitimate register but a future stale-snapshot vector.
- **Does NOT transfer (out of scope for prose):**
  - **Voice machinery** (Four Voices, Intra-Comment Surgery, banned Narrator) —
    category error; the prose reader cannot see a contract/rationale seam, and
    the banned Narrator voice is the _correct_ register for README/book/CoC/CONTRIBUTING.
    These concerns are governed by `.claude/rules/comments.md` for `.rs` files and
    do not apply to Markdown prose.
  - **Strict Truth-density / Length** — superseded by the teaching mandate. Prose
    docs legitimately repeat, explain, and elaborate; the Length heuristic from
    `comments.md` does not transfer here.

---

## 2. The one adaptation that does the heavy lifting

> The code rule "**reference by stable SYMBOL, never `file.rs:NNN`**" relies on
> the compiler validating the symbol. Prose cannot link a symbol. So it becomes:
>
> **Name the external contract (filename / flag / config field / path), but
> never freeze a COUNT, VERSION, ENUMERATION, or run-snapshot NUMBER. Pin any
> literal that must appear with a _guard_, not by hand — otherwise state the
> invariant instead of the number.**

Corollaries:

- `env!(CARGO_PKG_VERSION)` is **not** the prose answer — rendered Markdown is not
  compiled. The answer is a guard script (e.g. `check_book_version.py`).
- "State the invariant" beats "pin a source" when no generator exists: write
  "every CLI output is mirrored in Python" (a rule a gate enforces), not
  "currently none are missing" (a snapshot that rots); write "one analytic case
  per modeled feature; indices are sparse where a planned case was retired", not
  "27 cases; d12, d17, d18 unoccupied" (false — 29 ship, only d18 is missing).

---

## 3. Six prose-only failure modes

A code comment is co-located with one code site, so these cannot arise there:

1. **Single-source fan-out (cache coherency).** One authoritative fact cached
   into N audience docs that drift apart. _Rule:_ one owner per fact; every other
   doc is a shape-only pointer or a guard-pinned literal.
   _Example:_ MSRV 1.86 (book) vs 1.88 (Cargo.toml) — this specific recurrence
   is now prevented by a guard, but the failure mode remains live for any
   multi-doc fact without a comparable guard.

2. **Stale current-state snapshots presented as standing truth.** A census or
   completeness claim that rots silently (no adjacent diff forces an edit).
   _Rule:_ state the durable invariant, not the transient count.
   _Example:_ crate-partition prose in `CLAUDE.md`; deterministic-suite case count.

3. **Audience-bleed.** Detail aimed at the wrong reader (internal struct names on
   an end-user page; finding-IDs in a release ledger). _Rule:_ match content to
   the doc's declared reader; relocate or signpost.

4. **Executable/resolvable claim with no compiler backstop (scoped).** Paths,
   commands, flags, and copy-paste code blocks a reader runs. _Rule:_ every cited
   **repo-relative** path/command/flag must resolve against the live tree or
   binary. **Scope bound (critical):** repo-relative prefixes only — **never**
   "every cited filename resolves", because the book legitimately cites the external
   `cobre-docs` spec/theory pages absent from this repo. The path/link checker
   (when wired) enforces this bound; it is the spec the checker must honor.

5. **Self-executable agent instruction.** A command an agent-facing doc tells the
   reader to run must itself _succeed_ — verified by running it, not by
   cross-checking sibling prose. Distinct from cross-doc contradiction: it can be
   false even if every other doc were deleted.
   _Example:_ a non-compiling `--all-features` invocation in CLAUDE.md.

6. **Misplaced artifact + false-confidence guards.** _Where_ a Markdown file sits
   is itself a truth claim — a dated working artifact committed at the repo root
   reads as canonical. _Corollary:_ a partial guard lies; a passing guard is
   evidence only for the fact-class it parses. Audit guard **coverage**, not just
   pass/fail.
   _Examples:_ `allow-attribute-inventory.md` tracked at root;
   `check_book_version.py` green while MSRV drift survives; a schema-count assert
   that passes while the actual count has grown.

---

## 4. Applying the rules

### When writing or reviewing prose docs

- For every **number, version, or enumeration** in a doc: ask whether a guard pins
  it. If not, replace the count/snapshot with the invariant form. Do not hand-edit
  a number that a guard script could own.
- For every **path/command/flag** cited in a repo-relative context: confirm it
  resolves in the live tree. Do not cite paths to external repositories (e.g.,
  `cobre-docs` spec pages) as if they were repo-relative — they are legitimately
  absent.
- For every **fact that appears in more than one doc**: identify the single owner.
  Secondary docs may carry a shape-only pointer ("see CONTRIBUTING.md for the full
  list") or a guard-pinned literal; they must not silently diverge.
- For **CHANGELOG entries**: Provenance is inverted — historical narrative is
  correct here. Plan-leakage still applies: no `Epic`/`ticket`/`workstream` tokens
  in released CHANGELOG entries.

### Directives

- **Do** name the external contract (filename, flag, config field, path).
- **Do** state the invariant when no machine can pin the count.
- **Do** flag audience-bleed: if a reader of this doc would not recognize a term,
  it belongs in a different doc or needs a signpost.
- **Do not** freeze a COUNT, VERSION, ENUMERATION, or run-snapshot NUMBER without
  a guard.
- **Do not** add prose that only makes sense to a reader familiar with internal
  plan structure (`Epic N`, `ticket-NNN`, `workstream F-NNN`).
- **Do not** conflate the scope bound: the path/link checker (when wired) checks
  repo-relative prefixes only — treat external `cobre-docs` citations as
  intentionally unresolvable from this repo.
