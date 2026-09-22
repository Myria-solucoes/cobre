"""Linux CLI/Python parity for the shared NUMA-aware worker mapping."""

from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import sys
from datetime import datetime

import pytest


D01_CASE = "examples/deterministic/d01-thermal-dispatch"


def _cli_binary() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).parents[3]
    for profile in ("release", "debug"):
        candidate = repo_root / "target" / profile / "cobre"
        if candidate.is_file():
            return candidate
    pytest.skip("No compiled `cobre` CLI binary found")
    raise RuntimeError("unreachable: pytest.skip raises Skipped")


def _policy_bytes(path: pathlib.Path) -> bytes:
    data = path.read_bytes()
    if path.name != "manifest.bin":
        return data
    assert data[4:8] == b"CBVF"
    table = struct.unpack_from("<I", data)[0]
    vtable = table - struct.unpack_from("<i", data, table)[0]
    # CheckpointManifest.created_at is id 2 (vtable slot 8) in policy.fbs.
    offset = struct.unpack_from("<H", data, vtable + 8)[0]
    assert offset > 0
    field = table + offset
    string = field + struct.unpack_from("<I", data, field)[0]
    size = struct.unpack_from("<I", data, string)[0]
    begin = string + 4
    assert size == 20
    datetime.strptime(data[begin:begin + size].decode("ascii"), "%Y-%m-%dT%H:%M:%SZ")
    return data[:begin] + b"0" * size + data[begin + size:]


def _policy_files(root: pathlib.Path) -> dict[pathlib.Path, bytes]:
    policy = root / "policy"
    return {
        path.relative_to(policy): _policy_bytes(path)
        for path in sorted(policy.rglob("*"))
        if path.is_file()
    }


@pytest.mark.skipif(sys.platform != "linux", reason="native affinity is Linux-only")
def test_cli_python_core_affinity_and_policy_parity(tmp_path: pathlib.Path) -> None:
    """Both front ends resolve the same mapping and numerical policy."""
    cobre_run = pytest.importorskip("cobre.run")
    repo_root = pathlib.Path(__file__).parents[3]
    case_dir = repo_root / D01_CASE
    cli_out = tmp_path / "cli"
    py_out = tmp_path / "python"

    result = subprocess.run(
        [
            str(_cli_binary()),
            "run",
            str(case_dir),
            "--output",
            str(cli_out),
            "--threads",
            "2",
            "--cpu-bind",
            "core",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(f"cobre CLI failed: {result.stderr}")

    cobre_run.run(
        str(case_dir),
        output_dir=str(py_out),
        threads=2,
        cpu_bind="core",
        config_overrides={"simulation.enabled": False},
    )

    cli_metadata = json.loads((cli_out / "training/metadata.json").read_text())
    py_metadata = json.loads((py_out / "training/metadata.json").read_text())
    cli_affinity = cli_metadata["distribution"]["rank_affinity"]
    py_affinity = py_metadata["distribution"]["rank_affinity"]
    assert cli_affinity == py_affinity
    assert cli_metadata["solve_stats"].keys() == py_metadata["solve_stats"].keys()
    assert _policy_files(cli_out) == _policy_files(py_out)
