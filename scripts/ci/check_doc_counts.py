#!/usr/bin/env python3
"""Guard hand-pinned "N columns/fields" counts against their adjacent table.

Several book pages introduce a schema table with a sentence like
"... 27 columns." or "Eight columns ...", immediately followed by a markdown
table whose data rows enumerate exactly those columns/fields. The number and the
table can drift apart silently (the Rust schema tests assert the *code*, not the
prose), so this gate cross-checks every such count against the row count of the
table that follows it.

It is deliberately conservative: a count is only checked when a markdown table
begins within a few lines after it. Counts with no following table (back-
references such as "all eleven columns must be present", or struct-field counts
on a crate page) are skipped — those are handled by stating the invariant in
prose instead of pinning a guarded number.

Usage:
    python3 scripts/ci/check_doc_counts.py
    python3 scripts/ci/check_doc_counts.py --root /path/to/repo
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Book files that use the "N columns/fields" + adjacent table convention.
TARGET_FILES: tuple[str, ...] = (
    "book/src/reference/output-format.md",
    "book/src/reference/case-format.md",
    "book/src/crates/core.md",
    "book/src/guide/performance-accelerators.md",
)

# Spelled-out cardinals the book uses for small counts.
WORD_TO_INT: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

# A count token: digits or a spelled cardinal, then "columns"/"fields".
# `(?<![\w-])` rejects "4-column" and mid-word matches.
_COUNT_RE = re.compile(
    r"(?<![\w-])(\d+|" + "|".join(WORD_TO_INT) + r")\s+(?:columns|fields)\b",
    re.IGNORECASE,
)


def _parse_count(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    return WORD_TO_INT.get(token.lower())


def _is_table_separator(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and set(s) <= set("|:- ") and "-" in s


def _count_table_rows(lines: list[str], header_idx: int) -> int:
    """Count data rows of the table whose header is at `header_idx`.

    `header_idx` is the `| ... |` header line; `header_idx + 1` is its
    `|---|` separator. Data rows are the contiguous `|`-prefixed lines after it.
    """
    rows = 0
    for line in lines[header_idx + 2 :]:
        if line.lstrip().startswith("|"):
            rows += 1
        else:
            break
    return rows


def _table_rows_after(lines: list[str], start: int) -> int | None:
    """Return the data-row count of the table that immediately follows the count
    line, or None if the next non-blank content is not a table.

    Only blank lines may separate the count from its table. This is what
    distinguishes a table-intro count ("... 27 columns." directly above the
    schema table) from a prose mention ("replaced with three columns each ...",
    "`DeficitSegment` has two fields: ...") that happens to precede an unrelated
    table further down — the latter has intervening prose and is skipped.
    """
    idx = start + 1
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if (
        idx + 1 < len(lines)
        and lines[idx].lstrip().startswith("|")
        and _is_table_separator(lines[idx + 1])
    ):
        return _count_table_rows(lines, idx)
    return None


def check_file(path: Path) -> list[str]:
    """Return a list of drift messages for one file (empty = clean)."""
    problems: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    in_fence = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # Back-reference forms ("all N columns must be present") point at a table
        # above, not below — skip them.
        if "must be present" in line.lower():
            continue
        match = _COUNT_RE.search(line)
        if match is None:
            continue
        stated = _parse_count(match.group(1))
        if stated is None:
            continue
        rows = _table_rows_after(lines, i)
        if rows is None:
            continue
        if stated != rows:
            problems.append(
                f"{path}:{i + 1}: states {match.group(0)!r} but the adjacent "
                f"table has {rows} data rows"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (defaults to two levels above this script)",
    )
    args = parser.parse_args()
    root: Path = args.root

    problems: list[str] = []
    checked = 0
    for rel in TARGET_FILES:
        path = root / rel
        if not path.is_file():
            print(f"ERROR: target file not found: {path}", file=sys.stderr)
            return 2
        problems.extend(check_file(path))
        checked += 1

    if problems:
        print("ERROR: doc count drift (a pinned column/field count disagrees")
        print("with its adjacent table). Fix the number or the table:")
        for p in problems:
            print(f"  {p}")
        return 1

    print(
        f"OK: column/field counts in {checked} doc files match their adjacent tables."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
