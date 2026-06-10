#!/usr/bin/env python3
"""Check that all version references in book/src/**/*.md match the workspace Cargo.toml.

Scans all Markdown files under book/src/ for two patterns:
  - COBRE v<VERSION>      (uppercase banner, e.g. quickstart.md)
  - cobre   v<VERSION>    (lowercase CLI output with three spaces, e.g. installation.md)

Parses the workspace version from `Cargo.toml` (the first `version = "..."` line)
and reports any version reference that does not match.

It also pins the book's MSRV: every Rust toolchain reference in MSRV context
(see `scan_book_msrv`) must equal the workspace `rust-version` from `Cargo.toml`,
so the toolchain string cannot drift from the declared minimum.

Usage:
    python3 scripts/check_book_version.py
    python3 scripts/check_book_version.py --root /path/to/repo

Exit code 0 only if every matched version equals the Cargo.toml `version` AND
every MSRV reference equals the Cargo.toml `rust-version`; 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_cargo_version(cargo_path: Path) -> str | None:
    """Extract the workspace version from Cargo.toml."""
    if not cargo_path.exists():
        return None
    for line in cargo_path.read_text().splitlines():
        m = re.match(r'^version\s*=\s*"([^"]+)"', line)
        if m:
            return m.group(1)
    return None


def parse_cargo_rust_version(cargo_path: Path) -> str | None:
    """Extract the workspace MSRV (`rust-version`) from Cargo.toml.

    Returns the first `rust-version = "..."` value (e.g. "1.88"), or None if the
    field is absent.  This is a sibling of :func:`parse_cargo_version`; it does
    NOT reuse that regex because `^version\\s*=` also matches the package
    `version` line and would wrongly return the package version.
    """
    if not cargo_path.exists():
        return None
    for line in cargo_path.read_text().splitlines():
        m = re.match(r'^rust-version\s*=\s*"([^"]+)"', line)
        if m:
            return m.group(1)
    return None


_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"COBRE v(\d+\.\d+\.\d+)"),
    re.compile(r"cobre\s{3}v(\d+\.\d+\.\d+)"),
]

# A minor-version token (e.g. "1.88"), optionally with a trailing "+" and/or a
# " (stable)" suffix, as the book writes it in MSRV references.
_MSRV_TOKEN: re.Pattern[str] = re.compile(r"\b(1\.\d+)(\+)?(?:\s*\(stable\))?")

# Substrings that mark a line as an MSRV (toolchain) statement.  Anchoring on
# these prevents false positives on unrelated prose that mentions a Rust version
# for some other reason (e.g. "the Rust 1.50 release introduced const generics").
_MSRV_CONTEXT_MARKERS: tuple[str, ...] = ("MSRV", "rust-version", "Rust toolchain")

# The "requires" form, e.g. "Requires Rust 1.88+"; also serves as MSRV context.
_MSRV_REQUIRES_FORM: re.Pattern[str] = re.compile(r"Rust 1\.\d+\+")


def _is_msrv_context(line: str) -> bool:
    """Return True if a line is an MSRV (toolchain) statement worth checking."""
    return (
        any(marker in line for marker in _MSRV_CONTEXT_MARKERS)
        or _MSRV_REQUIRES_FORM.search(line) is not None
    )


def scan_book_versions(book_src: Path, expected: str) -> list[tuple[Path, int, str]]:
    """Scan all .md files under book_src for version mismatches.

    Returns a list of (file_path, line_number, found_version) tuples where the
    found version does not equal expected.  File read errors are printed to
    stderr and counted as a single mismatch entry with found_version="<unreadable>".
    """
    mismatches: list[tuple[Path, int, str]] = []

    md_files = sorted(book_src.rglob("*.md"))
    if not md_files:
        print(f"WARNING: No .md files found under {book_src}", file=sys.stderr)
        return mismatches

    for md_path in md_files:
        try:
            lines = md_path.read_text(encoding="utf-8").splitlines()
        except OSError as err:
            print(f"WARNING: Could not read {md_path}: {err}", file=sys.stderr)
            mismatches.append((md_path, 0, "<unreadable>"))
            continue

        for lineno, line in enumerate(lines, start=1):
            for pattern in _PATTERNS:
                for m in pattern.finditer(line):
                    found = m.group(1)
                    if found != expected:
                        mismatches.append((md_path, lineno, found))

    return mismatches


def scan_book_msrv(book_src: Path, expected_msrv: str) -> list[tuple[Path, int, str]]:
    """Scan all .md files under book_src for MSRV (Rust toolchain) mismatches.

    Only lines in MSRV context (see :func:`_is_msrv_context`) are inspected, so
    incidental prose that names a Rust version is not flagged.  Within such a
    line, every `1.\\d+` token is compared (after stripping a trailing `+` and a
    ` (stable)` suffix) against expected_msrv.

    Returns the same (file_path, line_number, found_token) shape as
    :func:`scan_book_versions`.  File read errors are printed to stderr and
    counted as a single mismatch entry with found_token="<unreadable>".
    """
    mismatches: list[tuple[Path, int, str]] = []

    md_files = sorted(book_src.rglob("*.md"))
    if not md_files:
        print(f"WARNING: No .md files found under {book_src}", file=sys.stderr)
        return mismatches

    for md_path in md_files:
        try:
            lines = md_path.read_text(encoding="utf-8").splitlines()
        except OSError as err:
            print(f"WARNING: Could not read {md_path}: {err}", file=sys.stderr)
            mismatches.append((md_path, 0, "<unreadable>"))
            continue

        for lineno, line in enumerate(lines, start=1):
            if not _is_msrv_context(line):
                continue
            for m in _MSRV_TOKEN.finditer(line):
                minor_pair = m.group(1)
                if minor_pair != expected_msrv:
                    # Preserve the trailing "+" (if any) in the reported token so
                    # the message reflects what the book actually wrote.
                    found = minor_pair + (m.group(2) or "")
                    mismatches.append((md_path, lineno, found))

    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check book/src/**/*.md version references match Cargo.toml."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root (default: current directory).",
    )
    args = parser.parse_args()

    cargo_path = args.root / "Cargo.toml"
    book_src = args.root / "book" / "src"

    cargo_version = parse_cargo_version(cargo_path)
    if cargo_version is None:
        print(f"ERROR: Could not parse version from {cargo_path}", file=sys.stderr)
        sys.exit(1)

    cargo_msrv = parse_cargo_rust_version(cargo_path)
    if cargo_msrv is None:
        print(f"ERROR: Could not parse rust-version from {cargo_path}", file=sys.stderr)
        sys.exit(1)

    version_mismatches = scan_book_versions(book_src, cargo_version)
    msrv_mismatches = scan_book_msrv(book_src, cargo_msrv)

    if not version_mismatches and not msrv_mismatches:
        # Count total matched references to include in the OK message.
        total = 0
        if book_src.exists():
            for md_path in sorted(book_src.rglob("*.md")):
                try:
                    text = md_path.read_text(encoding="utf-8")
                    for pattern in _PATTERNS:
                        total += len(pattern.findall(text))
                except OSError:
                    pass
        print(
            f"OK: All {total} crate-version references in book/ match Cargo.toml "
            f"(v{cargo_version}); all MSRV references match rust-version {cargo_msrv}."
        )
        sys.exit(0)

    for file_path, lineno, found in version_mismatches:
        print(
            f"MISMATCH: {file_path}:{lineno}: found v{found}, expected v{cargo_version}"
        )
    for file_path, lineno, found in msrv_mismatches:
        print(
            f"MISMATCH: {file_path}:{lineno}: found MSRV {found}, expected {cargo_msrv}"
        )
    total_mismatches = len(version_mismatches) + len(msrv_mismatches)
    print(f"FAIL: {total_mismatches} mismatches found.")
    sys.exit(1)


if __name__ == "__main__":
    main()
