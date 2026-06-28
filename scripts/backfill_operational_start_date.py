#!/usr/bin/env python3
"""One-shot fixture migration: add operational_start_date to entity objects.

For every non-empty entity array in a system/<entity>.json fixture, this assigns
operational_start_date so that, within each file, the date strictly increases
with the entity's id: rank entities by id ascending and set
date = base_date + rank (day offset). The smallest id gets base_date.

Because dates strictly increase with id and SystemBuilder sorts by
(operational_start_date, name), the built order equals the id-ascending order for
every file, including the name-divergent cases (D31, D39, D40, D41, D42). The
name tiebreak never fires because every entity in a collection gets a distinct
date.

The field is inserted as a text edit immediately after each entity's "name" line
(matching its indentation), so existing object order and formatting are
preserved; the file is not reserialized.

Idempotent: a file whose entities already carry the field is skipped.
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import re
import sys

BASE_DATE = datetime.date(2020, 1, 1)

# Entity files in scope and the JSON key that holds their entity array.
ENTITY_FILES: dict[str, str] = {
    "buses.json": "buses",
    "hydros.json": "hydros",
    "thermals.json": "thermals",
    "lines.json": "lines",
    "non_controllable_sources.json": "non_controllable_sources",
    "pumping_stations.json": "pumping_stations",
    "energy_contracts.json": "contracts",
}

NAME_LINE = re.compile(r'^(?P<indent>\s*)"name"\s*:\s*.*?(?P<comma>,?)\s*$')


def array_key(doc: dict, fname: str) -> str | None:
    declared = ENTITY_FILES[fname]
    if declared in doc and isinstance(doc[declared], list):
        return declared
    stem = fname[:-5]
    if stem in doc and isinstance(doc[stem], list):
        return stem
    for k, v in doc.items():
        if isinstance(v, list):
            return k
    return None


def backfill_file(path: str) -> bool:
    fname = os.path.basename(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    doc = json.loads(text)
    key = array_key(doc, fname)
    if key is None:
        return False
    entities = doc[key]
    if not entities:
        return False
    if any("operational_start_date" in e for e in entities):
        return False

    ids = [e["id"] for e in entities]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: duplicate ids {ids}; non-monotonic scheme")

    # rank by id ascending -> date offset; map document-order index -> date.
    rank_by_id = {eid: r for r, eid in enumerate(sorted(ids))}
    dates_in_doc_order = [
        (BASE_DATE + datetime.timedelta(days=rank_by_id[e["id"]])).isoformat()
        for e in entities
    ]

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    seen = 0
    for line in lines:
        m = NAME_LINE.match(line.rstrip("\n").rstrip("\r"))
        if m is None:
            out.append(line)
            continue
        indent = m.group("indent")
        date = dates_in_doc_order[seen]
        newline = "\n" if line.endswith("\n") else ""
        if m.group("comma"):
            # name is not the last field: keep its line, insert a trailing-comma date line.
            out.append(line)
            out.append(f'{indent}"operational_start_date": "{date}",{newline}')
        else:
            # name was the last field (no comma): add a comma to it, insert a comma-less date line.
            stripped = line.rstrip()
            trailing = line[len(stripped) :]
            out.append(f"{stripped},{trailing}")
            out.append(f'{indent}"operational_start_date": "{date}"{newline}')
        seen += 1
    if seen != len(entities):
        raise SystemExit(
            f"{path}: matched {seen} name lines but {len(entities)} entities"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(out))
    return True


def main() -> int:
    targets: list[str] = []
    for sysdir in sorted(glob.glob("examples/deterministic/d*/system")):
        for fname in ENTITY_FILES:
            p = os.path.join(sysdir, fname)
            if os.path.exists(p):
                targets.append(p)
    for fixture in (
        "crates/cobre-sddp/tests/fixtures/b6a_hydro_inflow_cascade/system",
        "crates/cobre-sddp/tests/fixtures/pumping_transfer/system",
    ):
        for fname in ENTITY_FILES:
            p = os.path.join(fixture, fname)
            if os.path.exists(p):
                targets.append(p)

    changed = 0
    for p in targets:
        if backfill_file(p):
            changed += 1
    print(f"backfilled {changed} files (of {len(targets)} candidate entity files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
