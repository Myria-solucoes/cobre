#!/usr/bin/env python3
"""Check that CLI and Python bindings write the same output files.

Parses both the CLI `run` module (`crates/cobre-cli/src/`, a directory) and
`crates/cobre-python/src/` for calls to writers from `cobre_io` and
`cobre_sddp::orchestration`. Resolves bare imported calls back to their
canonical names through `use` statements, then compares the two sets.

Usage:
    python3 scripts/ci/check_python_parity.py              # default: --max 0, --min-shared 18
    python3 scripts/ci/check_python_parity.py --max 0 --min-shared 18
    python3 scripts/ci/check_python_parity.py --min-shared 19  # floor breach check

Exit code 0 if parity holds (mismatches <= --max, shared >= --min-shared), 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Patterns for imports and calls.
# Match `use cobre_io::` and `use cobre_sddp::orchestration::` imports.
IMPORT_RE = re.compile(
    r"""
    ^\s*(?:pub\s+)?use\s+
    (?:cobre_io|cobre_sddp::orchestration)
    (?:::[\w:]+)*\s*
    (?:\{[^}]+\}|::[\w]+)
    \s*(?:as\s+\w+)?
    \s*;
    """,
    re.VERBOSE | re.MULTILINE,
)

# Normalisation map: different names that map to the same logical write.
NORMALISE: dict[str, str] = {
    "write_checkpoint": "write_policy_checkpoint",
    "io_write_policy_checkpoint": "write_policy_checkpoint",
}


def parse_imports(text: str) -> dict[str, str]:
    """Parse `use` statements from cobre_io/cobre_sddp::orchestration into a local_name -> canonical_name map.

    Handles single-line, multi-line brace groups, nested {}, and `as` aliases.
    """
    import_map: dict[str, str] = {}

    # Join continuation lines into whole statements by accumulating until `;`.
    lines_iter = iter(text.splitlines())
    current = ""

    for line in lines_iter:
        stripped = line.strip()
        # Skip comments and attributes.
        if stripped.startswith("//") or stripped.startswith("#["):
            continue

        current += " " + line
        if ";" in current:
            # Process complete statement.
            statement = current[: current.index(";") + 1]
            current = ""

            # Check if it's a relevant import.
            if not (
                "use cobre_io::" in statement
                or "use cobre_sddp::orchestration::" in statement
            ):
                continue

            # Parse the import.
            _parse_import_statement(statement, import_map)

    return import_map


def _parse_import_statement(statement: str, import_map: dict[str, str]) -> None:
    """Parse a single import statement and update import_map."""
    # Extract the path and items.
    statement = statement.strip()

    # Handle `as` aliases.
    if " as " in statement:
        # Extract the original name and the alias.
        match = re.search(r"use\s+([\w:]+)\s+as\s+(\w+)\s*;", statement)
        if match:
            full_path = match.group(1)
            alias = match.group(2)
            canonical = full_path.split("::")[-1]
            import_map[alias] = canonical
            return

    # Handle brace groups.
    if "{" in statement:
        # Extract base path and items.
        match = re.match(r".*use\s+([\w:]+)::\{([^}]+)\}", statement)
        if match:
            base = match.group(1)
            items_str = match.group(2)

            for item in items_str.split(","):
                item = item.strip()
                if not item:
                    continue

                # Handle nested paths.
                if "::" in item:
                    # Nested item like `output::write_foo`.
                    parts = item.split("::")
                    canonical = parts[-1]
                    import_map[canonical] = canonical
                else:
                    # Simple item.
                    import_map[item] = item
    else:
        # Single-item import.
        match = re.match(r".*use\s+([\w:]+)::(\w+)\s*;", statement)
        if match:
            canonical = match.group(2)
            import_map[canonical] = canonical


def _extract_from_text(text: str, names: set[str], import_map: dict[str, str]) -> None:
    """Collect write function names from one Rust source body into ``names``."""
    for line in text.splitlines():
        stripped = line.strip()
        # Skip comments, use/import lines, and attributes.
        if (
            stripped.startswith("//")
            or stripped.startswith("use ")
            or stripped.startswith("#[")
        ):
            continue

        # Match fully-qualified calls.
        for match in re.finditer(
            r"cobre_io::([\w:]+::)*(write_\w+|export_\w+)\s*\(", line
        ):
            name = match.group(2)
            if name:
                name = NORMALISE.get(name, name)
                names.add(name)

        for match in re.finditer(
            r"cobre_sddp::orchestration::([\w:]+::)*(write_\w+|export_\w+)\s*\(", line
        ):
            name = match.group(2)
            if name:
                name = NORMALISE.get(name, name)
                names.add(name)

        # Match bare calls and resolve through import map.
        # Matches write_foo(, export_foo(, FooWriter(, and FooWriter::
        for match in re.finditer(
            r"\b(write_\w+|export_\w+|\w+Writer)(?:\s*\(|::)", line
        ):
            name = match.group(1)
            if name in import_map:
                canonical = import_map[name]
                canonical = NORMALISE.get(canonical, canonical)
                names.add(canonical)


def extract_write_functions(path: Path) -> set[str]:
    """Extract the set of write function names from a Rust source location.

    Accepts either a single ``.rs`` file or a directory module: when ``path``
    is a directory, every ``.rs`` file beneath it is scanned recursively and the
    extracted names are unioned. This keeps the full CLI write surface in scope
    even when the writes are spread across submodules of a directory module.
    """
    names: set[str] = set()

    if path.is_dir():
        for rs_file in sorted(path.rglob("*.rs")):
            text = rs_file.read_text(errors="replace")
            import_map = parse_imports(text)
            _extract_from_text(text, names, import_map)
        return names

    if not path.exists():
        print(f"WARNING: {path} does not exist", file=sys.stderr)
        return set()

    text = path.read_text(errors="replace")
    import_map = parse_imports(text)
    _extract_from_text(text, names, import_map)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Python parity for output writes."
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Maximum allowed mismatches (default: 0). Exit 1 if exceeded.",
    )
    parser.add_argument(
        "--min-shared",
        type=int,
        default=18,
        help="Minimum shared write functions (default: 18). Exit 1 if below this floor. The floor exists to catch a gate that stopped seeing call sites; raise it when writers are added, never lower it.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root (default: current directory).",
    )
    args = parser.parse_args()

    cli_path = args.root / "crates" / "cobre-cli" / "src"
    python_path = args.root / "crates" / "cobre-python" / "src"

    cli_writes = extract_write_functions(cli_path)
    python_writes = extract_write_functions(python_path)

    cli_only = sorted(cli_writes - python_writes)
    python_only = sorted(python_writes - cli_writes)
    mismatches = len(cli_only) + len(python_only)
    shared = sorted(cli_writes & python_writes)

    # Check floor before mismatch.
    if len(shared) < args.min_shared:
        print(
            f"FAIL: {len(shared)} write functions in both paths (floor: {args.min_shared}). "
            f"The gate stopped seeing call sites. Check that imports are resolved correctly."
        )
        sys.exit(1)

    if mismatches > args.max:
        print(f"FAIL: {mismatches} parity mismatch(es) (max allowed: {args.max})")
        if cli_only:
            print()
            print("  In CLI but missing from Python:")
            for name in cli_only:
                print(f"    - {name}")
        if python_only:
            print()
            print("  In Python but missing from CLI:")
            for name in python_only:
                print(f"    - {name}")
        print()
        print("Fix: add the missing write call(s) to the other path.")
        print("CLI path:    crates/cobre-cli/src/")
        print("Python path: crates/cobre-python/src/")
        sys.exit(1)
    else:
        print(
            f"OK: {mismatches} parity mismatch(es) (max allowed: {args.max}). "
            f"{len(shared)} write functions in both paths."
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
