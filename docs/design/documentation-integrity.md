# Cobre Documentation Integrity — Prose-Doc Scope of the Commenting Philosophy

> **Status:** Design proposal / companion report. Not yet a coding standard and
> not yet enforced. Read together with
> [`commenting-philosophy.md`](./commenting-philosophy.md): that doc governs
> `**/*.rs` comments + rustdoc; **this** doc determines how far its concerns
> extend to developer-facing _prose_ docs (CLAUDE.md, README, CONTRIBUTING, the
> mdBook, governance files) and records the empirical debt that motivates it.
>
> **Self-pinned provenance (this report eats its own dog food):** all findings
> were verified against the working tree at commit **`06333b73`** on
> **2026-06-09**. Every count and `file:line` below is a point-in-time snapshot
> by the very definition this report warns about — **re-verify before acting**;
> several will have drifted by the time anyone reads this.
>
> **Method:** two adversarially-structured agent workflows (an 8-area empirical
> verification + an 8-group scope characterization with a 4-lens stress pass),
> reconciled against direct `grep`/binary/`cargo` inspection by the author. Where
> the two workflows disagreed, the live source tree (excluding `target/`) was
> treated as ground truth — see [§7 Verification notes](#7-verification-notes-and-corrections).

---

## 1. Purpose

Answer one question precisely, with evidence: **to what extent must the
commenting philosophy's concerns contemplate the repo's prose documentation,
and what should we change when we next touch these files?**

The short answer: **the truth-integrity half of the philosophy must govern every
developer-facing doc — and it matters _more_ in prose than in code — while the
stylistic/voice half must not.** The docs are already **17.2% adrift**, so this is
remediation, not prevention.

---

## 2. The reasoning — why prose is different, and where it is the same

The commenting philosophy is **contract-first**: a comment must carry what the
code cannot, and the dominant risk in this codebase is _over-deletion_ of
load-bearing contracts. Prose docs invert part of that and amplify the rest:

- **They legitimately narrate.** A README, the book, CONTRIBUTING, and the CoC
  exist to _teach and persuade_. The "Narrator voice" the code rules ban
  (`commenting-philosophy.md` Voice 3) is the **correct** register here. The
  strict truth-density / "length is not the metric" bar does **not** transfer —
  teaching repetition is signal, not noise.
- **But their facts rot harder than code comments**, for three structural
  reasons the code philosophy never had to handle:
  1. **A consuming agent acts on them with no human in the loop.** `CLAUDE.md` is
     read by Claude Code, which _executes_ its assertions. A wrong fact there is
     not a stale comment — it is a wrong action.
  2. **No compiler backstop.** A renamed symbol breaks `cargo build`. A retired
     filename, a dead test path, or a removed config key in prose fails
     **silently** — nothing recompiles a Markdown sentence.
  3. **Facts are cached across many audience-specific files.** The same fact
     (MSRV, crate count, build command, feature list) is restated in CLAUDE.md,
     CONTRIBUTING, README, and the book. One source change must be hand-propagated
     to N docs, and rarely is — so the copies drift apart and openly contradict.

So the **truth-integrity concerns** (Durability/anti-drift, Plan-leakage,
Contract-mirroring, and the present-tense rule of Provenance) transfer with
_more_ force in prose; the **voice/length concerns** do not transfer at all.

---

## 3. The proof — a measured 17.2% documentation drift rate

250 checkable doc claims were verified against the binary (`cobre --help`,
`cobre schema`, `cobre report` on a real run), the source structs, the generated
Arrow schemas, and `cargo` itself. **43 drifted** — 41 verified + 2 likely,
**0 uncertain**, **11 high-severity**.

### Drift density by surface

| Doc surface                                   | Drift density        |
| --------------------------------------------- | -------------------- |
| Cross-doc contradictions                      | **47.4%** (9 / 19)   |
| book — contracts & numbers                    | **33.3%** (7 / 21)   |
| Output files & parquet columns                | 21.9% (7 / 32)       |
| CONTRIBUTING — commands & features            | 21.7% (5 / 23)       |
| Config docs vs schema                         | 16.1% (5 / 31)       |
| README + governance                           | 9.7% (3 / 31)        |
| meta-rules (CLAUDE.md, architecture-rules.md) | 9.7% (6 / 62)        |
| CLI reference vs binary                       | 3.2% (1 / 31)        |
| **Overall**                                   | **17.2% (43 / 250)** |

### High-severity findings (the load-bearing kind)

Each was re-confirmed by the author against the live tree:

1. **CLAUDE.md tells the agent to run a command that does not compile.**
   `CLAUDE.md:12` makes `cargo test --workspace --all-features` canonical;
   `--all-features` enables both LP backends, which hit the `compile_error!` at
   `crates/cobre-solver/src/lib.rs:45` ("enable exactly one LP backend: `highs`
   OR `clp`"). The agent obeying its own instruction file gets a compile error,
   not tests. _Fix:_ `cargo test --workspace` (`default = ["highs"]`, confirmed
   in `cobre-cli/Cargo.toml:40` and `cobre-solver/Cargo.toml:16`).
2. **CLAUDE.md mis-models its own workspace.** `CLAUDE.md:10` says
   "14 crates (8 workspace + 6 excluded)". Ground truth (`Cargo.toml`):
   **13 workspace members + 1 excluded** (`cobre-python` only). The five "stub"
   crates it lists as _excluded_ (`cobre-mcp/tui/flow/uc/emt`) are workspace
   **members**. Drifted since 2026-04-29 (commit `ec9d5240`).
3. **A `_manifest.json` artifact documented across 5 book files is never
   written.** Confirmed in `1dtoy.md`, `cli-reference.md`,
   `interpreting-results.md`, `reference/output-format.md`,
   `tutorial/understanding-results.md`. The binary writes `training/metadata.json`
   - `_SUCCESS` (`results_writer.rs:115-118`); `manifest.rs:10` records that the
     merged `metadata.json` _replaced_ `_manifest.json`. A reader's `cat` /
     `cobre summary` action on the documented path fails.
4. **Copy-paste-poison config table.** `configuration.md:535-545` documents 9
   `exports` fields; `ExportsConfig` (`config/exports.rs:15-21`) accepts **only**
   `states` + `stochastic`. Under `deny_unknown_fields`, a config copied from the
   docs fails to parse.
5. **Wrong manifest JSON shape on every key a reader greps.**
   `understanding-results.md:139-181` shows top-level `version`/`checksum` and a
   `cuts` object; reality (`manifest.rs:167-296`, confirmed by a live `cobre
report`) is `cobre_version`, **no** `checksum`, and `row_pool` (not `cuts`).
6. **The `dynamic` cut-selection method is undocumented.** `configuration.md:288`
   lists only `level1`/`lml1`/`domination`; `dynamic` is in active use
   (`training.rs`, drives the lazy-solve loop) with 5 undocumented knobs, and the
   documented `check_frequency` default (1) is wrong — effective default is 5
   (`cut_selection.rs:626`, `unwrap_or(5)`).
7. **CONTRIBUTING installs the wrong mdBook preprocessor.** `CONTRIBUTING.md:17`
   says `cargo install mdbook mdbook-katex`; the repo uses **`mdbook-mermaid`**
   (`book/book.toml:32`, `.github/workflows/docs.yml:31-32`). `mdbook serve` fails
   on the missing mermaid command; katex is never used anywhere.
8. **Dead CLAUDE.md navigation pointers.** `CLAUDE.md:83` cites `write_outputs`
   (gone; real: `write_training_outputs` at `run.rs:1841`, `write_simulation_outputs`
   at `run.rs:1947`); `CLAUDE.md:84` cites `run_inner` (gone — see the
   stale-cache note in §7); `CLAUDE.md:74` cites `lp_builder.rs` (now the
   directory module `lp_builder/mod.rs`).
9. **MSRV contradiction.** `installation.md:65,78` say "Rust 1.86"; everything
   else (Cargo.toml:28, README.md:33, CLAUDE.md:8) says 1.88. The book is the
   sole outlier; the existing `check_book_version.py` passes green because it
   only checks the package banner, not the toolchain string (see §6).
10. **Stale parquet column counts (×4).** `output-format.md` / `interpreting-results.md`
    / `performance-accelerators.md` undercount the real Arrow schemas:
    convergence 13 → **14** (missing `mean_rows_in_lp`), iteration_timing 18 → **19**
    (missing `lazy_scoring_ms`), cut_selection 9 → **10** (missing
    `cuts_reactivated`). Each contradicts a `schemas.rs` field-count test.
11. **README over-claims a stub feature.** `README.md:28` advertises an
    "MCP server for AI agents"; `cobre-mcp` is a stub that prints "not yet
    implemented" and exits 1, and there is no `mcp` subcommand.

Lower-severity (medium/low) items — README book-host inconsistency
(`cobre-rs.github.io/cobre` vs `docs.cobre-rs.dev`), `cobre-comm` advertising a
non-existent TCP backend, the deterministic-suite case census (27 vs 29 cases),
the misplaced `allow-attribute-inventory.md`, and the `architecture-rules.md`
"Current gaps: None" snapshot — are catalogued in [§9 Appendix](#9-appendix--full-drift-ledger).

---

## 4. Scope matrix — how far each concern transfers, per doc

Transfer strength tracks **(machine reader) × (low teaching mandate)**.

| Doc                                          | Reader                                                     | Transfer profile                                                                                                                                                 |
| -------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CLAUDE.md**                                | an agent that **acts** on assertions, no human in the loop | **near-total**: Durability/anti-drift dominates; Plan-leakage (it defines the rule); Provenance (clean). Voice machinery N/A.                                    |
| **`.claude/rules/*`, architecture-rules.md** | agent + maintainer                                         | **near-total**: machine-consumed contracts; same as code-adjacent docs.                                                                                          |
| **CHANGELOG / THIRD_PARTY_NOTICES / CoC**    | release / license / community                              | **strong** for Plan-leakage; Provenance **inverted** (CHANGELOG _is_ history — the carve-out); Durability adapted (NOTICES is a maintained single-owner mirror). |
| **CONTRIBUTING.md**                          | external contributors                                      | **adapted**: every path/command/flag must resolve; Self-executable-instruction applies; teaching repetition is signal, not bloat.                                |
| **README.md**                                | newcomers / evaluators                                     | **adapted**: Plan-leakage full; Durability in spirit; narrative/promotional voice is the job (Length N/A).                                                       |
| **book/**                                    | end users (modelers)                                       | **adapted**: executable-claim integrity dominates the real debt; teaching mandate supersedes Length/Voice.                                                       |

### Which of the six concerns transfer

- **Transfer with force (and bite harder than in code):**
  - **Durability / anti-drift** — but _re-expressed_ for prose (see §5, the key
    adaptation).
  - **Plan-leakage** — already a CLAUDE.md hard rule for user-facing artifacts;
    `commenting-philosophy.md` N4 extends the pattern (`F-NNN`/`W-N`).
  - **Contract-mirroring** — "a drifted mirror is a lie"; prose mirrors many
    code-owned facts and is the densest drift source.
- **Transfers adapted:**
  - **Provenance** — present-tense rule holds, **except** the CHANGELOG, which is
    history by design (the standing carve-out). Forward-looking "planned" status
    prose is a legitimate register but a future stale-snapshot vector.
- **Does NOT transfer:**
  - **Voice machinery** (Four Voices, Intra-Comment Surgery, banned Narrator) —
    category error; the prose reader cannot see a contract/rationale seam and the
    banned Narrator voice is the _correct_ register for README/book/CoC/CONTRIBUTING.
  - **Strict Truth-density / Length** — superseded by the teaching mandate.

---

## 5. The one adaptation that does the heavy lifting

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

## 6. Six prose-only failure modes the code philosophy does not cover

A code comment is co-located with one code site, so these cannot arise there:

1. **Single-source fan-out (cache coherency).** One authoritative fact cached
   into N audience docs that drift apart. _Rule:_ one owner per fact; every other
   doc is a shape-only pointer or a guard-pinned literal.
   _Example:_ MSRV 1.86 (book) vs 1.88 (Cargo.toml).
2. **Stale current-state snapshots presented as standing truth.** A census or
   completeness claim that rots silently (no adjacent diff forces an edit).
   _Rule:_ state the durable invariant, not the transient count.
   _Example:_ `CLAUDE.md:10` crate partition; deterministic-suite case count.
3. **Audience-bleed.** Detail aimed at the wrong reader (internal struct names on
   an end-user page; finding-IDs in a release ledger). _Rule:_ match content to
   the doc's declared reader; relocate or signpost.
4. **Executable/resolvable claim with no compiler backstop (scoped).** Paths,
   commands, flags, and copy-paste code blocks a reader runs. _Rule:_ every cited
   **repo-relative** path/command/flag must resolve against the live tree or
   binary. **Scope bound (inherited from `commenting-philosophy.md` E3, "Guard —
   critical"):** repo-relative prefixes only — **never** "every cited filename
   resolves", because the book legitimately cites the external `cobre-docs`
   spec/theory pages absent from this repo.
5. **Self-executable agent instruction.** A command an agent-facing doc tells the
   reader to run must itself _succeed_ — verified by running it, not by
   cross-checking sibling prose. Distinct from cross-doc contradiction: it can be
   false even if every other doc were deleted. _Example:_ the non-compiling
   `--all-features` command (§3.1).
6. **Misplaced artifact + false-confidence guards.** _Where_ a Markdown file sits
   is itself a truth claim — a dated working artifact committed at the repo root
   reads as canonical. _Corollary:_ a partial guard lies; a passing guard is
   evidence only for the fact-class it parses. Audit guard **coverage**, not just
   pass/fail. _Examples:_ `allow-attribute-inventory.md` tracked at root;
   `check_book_version.py` green while MSRV drift survives; `cli_schema.rs`
   asserts `len() >= 8` while 18 schemas ship.

---

## 7. Verification notes and corrections

This report's claims were reconciled against the live tree; the two workflows
disagreed in three places, resolved here by direct inspection (a useful
demonstration of the thesis):

- **`run_inner` is DEAD — and the verification was nearly fooled.** One workflow
  "verified" `fn run_inner` live at `cobre-python/src/run.rs:470`. Direct check:
  no `fn run_inner` exists in `crates/cobre-python/src/`. The only occurrences of
  `run_inner` are in **`crates/cobre-python/target/debug/.fingerprint/.../output-*`**
  — stale clippy-cache files from a prior build when the function existed at line 470. The current line 470 is unrelated signal-handling code. **A careful
  verification agent was drift-fooled by a stale cached artifact** — which is
  exactly why prose facts must be checked against live source, `target/` excluded.
- **`assessment-report.md` is NOT committed debt.** It is gitignored
  (`.gitignore:38`, `assessment*.md`) and untracked. _However:_ a committed test
  references it — `crates/cobre-sddp/tests/anticipated_5stage_k2_smoke.rs:486`
  ("See F3-003 in assessment-report.md") — a dead reference + plan token in
  shipped code for anyone who clones. The only misplaced **tracked** root artifact
  is `allow-attribute-inventory.md`.
- **`allow-attribute-inventory.md` counts have themselves drifted.** It states
  "79 too_many_lines / 28 dead_code"; the live tree has ~61 `too_many_lines`
  `#[allow]`/`#[expect]` sites and ~27 `dead_code` sites. (A naive bare-token grep
  returns ~148, including doc-comments and the inventory's own prose — not
  attribute sites.) The fix is **relocation** into `plans/`, not editing the
  numbers; a dated snapshot is not maintainable standing documentation.

---

## 8. Recommendations (ROI-ordered)

### Capture, not stretch

Add a **sibling rule**, not a section bolted onto `commenting-philosophy.md`.
That doc is scoped to `**/*.rs` and is on a promotion path to a `**/*.rs`-glob
auto-loading rule; folding prose rules in would break its scope and mis-fire its
glob. Promote **this report's** §4–§6 into a `## Prose-doc truth integrity` rule
(or `.claude/rules/doc-integrity.md`) that reuses the five truth-integrity
principles verbatim, states the §5 adaptation, declares Voice-machinery and
strict-Length out of scope, and lists the six §6 failure modes. Add a CLAUDE.md
cross-pointer.

### Phase 1 — editorial, do now (hours, no CI). Highest ROI by far.

The debt is concentrated; ~12 one-line fixes outweigh any gate machinery. Each
is a concrete edit (re-verify line numbers first — they drift):

| #   | Location                                                                                             | Change                                                                                                                          |
| --- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `CLAUDE.md:12`                                                                                       | `cargo test --workspace --all-features` → `cargo test --workspace` (the `--all-features` form does not compile)                 |
| 2   | `CLAUDE.md:10`                                                                                       | "8 workspace + 6 excluded" → "13 workspace members + the maturin-built `cobre-python`" (state the shape, drop the split number) |
| 3   | `CLAUDE.md:74`                                                                                       | `lp_builder.rs` → `lp_builder/mod.rs`                                                                                           |
| 4   | `CLAUDE.md:83`                                                                                       | `write_outputs` → `write_training_outputs` / `write_simulation_outputs`                                                         |
| 5   | `CLAUDE.md:84`                                                                                       | `run_inner` → the live Python write path (`run_via_study` / `run_training_phase_py`) — `run_inner` is gone                      |
| 6   | `installation.md:65,78`                                                                              | Rust 1.86 → 1.88                                                                                                                |
| 7   | book ×5 (`1dtoy`, `cli-reference`, `interpreting-results`, `output-format`, `understanding-results`) | `_manifest.json` → `metadata.json` (+ note the `_SUCCESS` marker)                                                               |
| 8   | `understanding-results.md:139-181`                                                                   | manifest shape: `version`→`cobre_version`, drop `checksum`, `cuts`→`row_pool`                                                   |
| 9   | `configuration.md:535-545`                                                                           | trim `exports` table to `states` + `stochastic` (only accepted fields)                                                          |
| 10  | `configuration.md:288-311`                                                                           | add the `dynamic` method + its knobs; fix `check_frequency` default 1 → 5                                                       |
| 11  | `CONTRIBUTING.md:17`                                                                                 | `mdbook-katex` → `mdbook-mermaid`; `CONTRIBUTING.md:181-182` `tests/cli_version.rs` → `cli_smoke.rs`                            |
| 12  | `README.md:28`                                                                                       | qualify the MCP claim ("reserved/experimental"); reconcile the book host to one canonical URL                                   |
| 13  | `git mv allow-attribute-inventory.md plans/`                                                         | relocate the dated snapshot out of the repo root                                                                                |
| 14  | book parquet column counts                                                                           | convergence 13→14, iteration_timing 18→19, cut_selection 9→10                                                                   |

### Phase 2 — worth building (only the guards that pay for themselves)

- A **repo-relative path/link checker** over README/CONTRIBUTING/book/CLAUDE.md,
  scoped to code-fence tokens that look like repo-relative paths or
  `.json`/`.rs`/`.toml` filenames + intra-book relative links. **Must** exclude
  absolute URLs and the external `cobre-docs` refs (per `commenting-philosophy.md`
  E3). Catches `_manifest.json`×5, `tests/cli_version.rs`, `lp_builder.rs`.
- **Strengthen the two lying guards:** make `cli_schema.rs` exact-diff the
  committed `book/src/schemas/*.json` set (18 today) instead of `len() >= 8`;
  extend `check_book_version.py` to also pin MSRV (`Rust 1.\d+` near
  "MSRV"/"rust-version" vs `Cargo.toml`).

### Phase 3 — only if recurrence proves it

- Extend `check-no-plan-leaks.sh` scan paths to the repo root and `docs/`
  (excluding `plans/` and the gitignored `assessment*.md`). Low priority — its
  present offender is removed by Phase-1 #13.

### Drop entirely (over-engineering / category errors)

- `env!(CARGO_PKG_VERSION)` steering for rendered Markdown.
- An open-ended snapshot-fact grep (high churn; flags legitimate teaching numbers).
- Any CI diff of **run-snapshot numbers** against a live solve (couples docs to
  solver determinism; every algorithm change would fail the docs). Delete or
  invariant-state those numbers instead.
- A standing root-placement gate (one-file problem; a pre-commit warning on a new
  root `*.md` carrying a `Date:`/finding-ID header is the most that's justified).

### Judgment-only (cannot mechanize)

Whether a feature _claim_ is true (the MCP stub), whether "planned" status prose
is still accurate, paragraph-level audience fit, and whether a given count is
load-bearing enough to guard-pin vs simply delete.

---

## 9. Appendix — full drift ledger

Totals: **250 claims checked, 43 drifted** (41 verified, 2 likely, 0 uncertain),
**11 high / 22 medium / 10 low**. Category counts: stale-count 10, renamed-key 7,
wrong-number 6, missing-subcommand-or-flag 6, cross-doc-contradiction 6,
dead-link-or-path 3, other 3, misplaced-artifact 1, wrong-command 1.

| Category                              | Representative items (doc `file:line` — claim vs ground truth)                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **stale-count**                       | `configuration.md:535` 9 exports fields vs 2 accepted; `output-format.md:273` convergence 13 vs 14 cols; `output-format.md:295` iteration_timing 18 vs 19; `performance-accelerators.md:226` cut_selection 9 vs 10; `CLAUDE.md:10` "8+6" vs 13+1; `understanding-results.md:210` "four entity categories" vs up to 10 emitted; `BRAND-GUIDELINES.md:18` PyPI "cobre/pycobre" vs shipped `cobre-python` |
| **renamed-key**                       | manifest `version`→`cobre_version`, no `checksum`, `cuts`→`row_pool` (`understanding-results.md:139-181`); FPHA γᵥ "must be positive" vs `≥0` valid (`hydro-plants.md:333`); `architecture-rules.md:36,71` `OpeningTreeInputs`/`SimulationConfig` field drift; `CLAUDE.md:84` `run_inner` dead                                                                                                         |
| **wrong-number**                      | `sddp.md:628` CUT_WIRE_VERSION 2 vs 1; `sddp.md:633` 24-byte header vs 25; `cli-reference.md:355` "solver: HiGHS" vs "HiGHS 1.13.1"; `configuration.md:293` check_frequency 1 vs 5; `understanding-results.md:268` report `metadata` key absent                                                                                                                                                        |
| **cross-doc-contradiction**           | `CLAUDE.md:12` vs `CONTRIBUTING.md:40-46` on `--all-features` (CLAUDE side doesn't compile); MSRV 1.86 vs 1.88; `interpreting-results.md:86` lp_solves cumulative vs per-iteration; `overview.md:12` cobre-comm TCP backend (none); `CONTRIBUTING.md:189` 13-dir tree omits the `cobre` umbrella crate                                                                                                 |
| **missing-subcommand-or-flag**        | `configuration.md:288` omits `dynamic` method + 5 knobs; no `energy` section documented; `CLAUDE.md:83` `write_outputs` dead; `CONTRIBUTING.md:86` `--all-features` omits the flatc-panic note                                                                                                                                                                                                         |
| **dead-link-or-path**                 | `interpreting-results.md:18,61` `_manifest.json` (→ `metadata.json`); `CONTRIBUTING.md:181` `tests/cli_version.rs` (absent); `README.md:51` `docs.cobre-rs.dev` host inconsistency; book `stochastic-modeling.md` → 3 non-existent `docs/design/` files                                                                                                                                                |
| **other / misplaced / wrong-command** | `understanding-results.md` thermals cost-segment dimension (none); `README.md:28` MCP stub claim; `CONTRIBUTING.md:17` `mdbook-katex` (→ `mdbook-mermaid`); `allow-attribute-inventory.md` misplaced at root                                                                                                                                                                                           |

**2 "likely" (not full-confidence) items:** the phantom "Activity-update" wire
record in `sddp.md:636` (source-grep negative, but a future-planned variant can't
be excluded from grep alone); the `docs.cobre-rs.dev` host liveness (couldn't be
tested offline).

**Accurate today (risk-of-future-drift only, not present lies):**
`THIRD_PARTY_NOTICES`, `CODE_OF_CONDUCT`, `BRAND-GUIDELINES` palette/namespace
facts, the CHANGELOG commit-hash/source-line anchors (banned by _no_ existing
rule — only the proposed prose-Durability extension), and the meta-rule
contract-mirrors.

---

## 10. Relationship to existing rules

- **`commenting-philosophy.md`** — the parent. This report reuses its five
  truth-integrity principles and its E3 scope bound; it deliberately does _not_
  reuse its Voice machinery.
- **CLAUDE.md hard rules** — the plan-structure ban already names CHANGELOG,
  README, book/, and public rustdoc; `check-no-plan-leaks.sh` enforces it over
  `crates/`, `book/`, `CHANGELOG`, `README` (note: **CLAUDE.md and CONTRIBUTING
  are not in that scan set**, so their compliance is currently unguarded).
- **Existing doc guards** — `check_book_version.py` (package banner only) and
  `cli_schema.rs` (`len() >= 8`) are the false-confidence guards of §6.6;
  strengthen, don't trust.
