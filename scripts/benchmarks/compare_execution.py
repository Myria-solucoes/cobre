#!/usr/bin/env python3
"""Run serial matched-epoch comparisons from an explicit JSON arm matrix.

Each arm supplies name, binary, case, threads, and optionally cpu_bind.
Requires pyarrow to compare numerical Parquet contents independently of timing.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

import pyarrow as pa
import pyarrow.parquet as pq


TIMING_COLUMNS = {
    "time_forward_ms", "time_backward_ms", "time_total_ms", "selection_time_ms",
    "solve_time_ms", "load_model_time_ms", "set_bounds_time_ms", "basis_set_time_ms",
}


def digest_files(root, files):
    digest = hashlib.sha256()
    for path in sorted(files):
        name = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(name).to_bytes(8, "little") + name)
        digest.update(len(data).to_bytes(8, "little") + data)
    return digest.hexdigest()


def numerical_signature(output):
    policy = output / "policy"
    cuts = sorted((policy / "cuts").rglob("*.bin"))
    if not cuts:
        raise RuntimeError(f"no policy cuts in {output}")
    parquet = {}
    for path in sorted(output.rglob("*.parquet")):
        relative = path.relative_to(output).as_posix()
        if relative.startswith(("training/timing/", "training/solver/")):
            continue
        table = pq.read_table(path)
        columns = [name for name in table.column_names if name not in TIMING_COLUMNS]
        table = table.select(columns).combine_chunks().replace_schema_metadata(None)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        parquet[relative] = hashlib.sha256(sink.getvalue()).hexdigest()
    if "training/convergence.parquet" not in parquet:
        raise RuntimeError("missing convergence history")
    metadata = json.loads((output / "training/metadata.json").read_text())
    solver_rows = pq.read_table(output / "training/solver/iterations.parquet").to_pylist()
    solver_work = {}
    for row in solver_rows:
        phase = solver_work.setdefault(str(row["phase"]), {})
        for field in ("lp_solves", "simplex_iterations", "retry_attempts", "basis_consistency_failures"):
            phase[field] = phase.get(field, 0) + row[field]
    return {
        "cuts": digest_files(policy, cuts),
        "bases": digest_files(policy, (policy / "basis").rglob("*.bin")),
        "parquet": parquet,
        "bounds": metadata["bounds"],
        "row_pool": metadata["row_pool"],
        "iterations": metadata["iterations"],
        "work": {key: metadata["solve_stats"][key]
                 for key in ("total_lp_solves", "first_try", "retried", "failed")},
        "solver_work": solver_work,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    arms = json.loads(args.matrix.read_text())
    if args.repeats < 1 or not arms or len({a["name"] for a in arms}) != len(arms):
        parser.error("positive repetitions and distinct nonempty arms required")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"platform": platform.platform(), "affinity": sorted(os.sched_getaffinity(0)),
              "arms": arms, "runs": []}
    rng = random.Random(42)
    report_path = args.output / "report.json"
    for epoch in range(args.repeats):
        order = list(arms)
        rng.shuffle(order)
        for arm in order:
            run = args.output / f"{epoch}-{arm['name']}"
            run.mkdir()
            binary, case = Path(arm["binary"]), Path(arm["case"])
            command = [str(binary), "run", str(case), "--output", str(run / "output"),
                       "--threads", str(arm["threads"]), "--cpu-bind",
                       arm.get("cpu_bind", "none"), "--comm-backend", "local",
                       "--color", "never", "--quiet"]
            record = {"arm": arm["name"], "epoch": epoch, "command": command,
                      "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                      "case_sha256": digest_files(case, (p for p in case.rglob("*") if p.is_file()))}
            started = time.monotonic()
            with (run / "stdout.log").open("w") as stdout, (run / "stderr.log").open("w") as stderr:
                child = subprocess.Popen(command, stdout=stdout, stderr=stderr)
                peak_rss = 0
                while child.poll() is None:
                    try:
                        status = Path(f"/proc/{child.pid}/status").read_text()
                        for line in status.splitlines():
                            if line.startswith(("VmHWM:", "VmRSS:")):
                                peak_rss = max(peak_rss, int(line.split()[1]))
                    except FileNotFoundError:
                        pass
                    if time.monotonic() - started > args.timeout:
                        child.kill()
                        child.wait()
                        break
                    time.sleep(0.1)
            record.update(wall_seconds=time.monotonic() - started,
                          exit_code=child.returncode, peak_rss_kib=peak_rss)
            report["runs"].append(record)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            if child.returncode:
                raise SystemExit(f"failed arm {arm['name']}; see {run}")
            metadata = json.loads((run / "output/training/metadata.json").read_text())
            record.update(training_seconds=metadata["duration_seconds"],
                          solve_stats=metadata["solve_stats"],
                          signature=numerical_signature(run / "output"))
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: record[k] for k in ("arm", "epoch", "training_seconds", "peak_rss_kib")}), flush=True)
    report["summary"] = {}
    for arm in arms:
        values = [r["training_seconds"] for r in report["runs"] if r["arm"] == arm["name"]]
        report["summary"][arm["name"]] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    signatures = [json.dumps(r["signature"], sort_keys=True) for r in report["runs"]]
    report["exact_numerical_parity"] = len(set(signatures)) == 1
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": report["summary"], "exact_numerical_parity": report["exact_numerical_parity"]}), flush=True)


if __name__ == "__main__":
    main()
