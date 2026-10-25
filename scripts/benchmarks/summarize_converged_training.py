#!/usr/bin/env python3
"""Compare completed exact-bound runs at a common requested relative gap."""
import argparse
import json
import statistics
from pathlib import Path


def read_run(path, relative_gap):
    metadata = json.loads((path / 'training/metadata.json').read_text())
    bounds = metadata['bounds']
    lower, upper = bounds['final_lower_bound'], bounds['final_upper_bound']
    assert metadata['status'] == 'complete', path
    assert metadata['convergence']['achieved'], path
    assert bounds['final_upper_bound_kind'] == 'exact', path
    assert metadata['solve_stats']['failed'] == 0, path
    assert lower <= upper + 1e-6 * max(1, abs(lower)), path
    assert (upper - lower) <= relative_gap * max(1, abs(lower)), path
    return {
        'run': path.name,
        'seconds': metadata['duration_seconds'],
        'iterations': metadata['iterations']['completed'],
        'lower_bound': lower,
        'upper_bound': upper,
        'lp_solves': metadata['solve_stats']['total_lp_solves'],
        'retried': metadata['solve_stats']['retried'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, nargs='+', required=True)
    parser.add_argument('--after', type=Path, nargs='+', required=True)
    parser.add_argument('--relative-gap', type=float, required=True)
    args = parser.parse_args()
    before = [read_run(path, args.relative_gap) for path in args.before]
    after = [read_run(path, args.relative_gap) for path in args.after]
    before_time = statistics.median(run['seconds'] for run in before)
    after_time = statistics.median(run['seconds'] for run in after)
    best_lower = max(run['lower_bound'] for run in before + after)
    for run in before + after:
        assert run['upper_bound'] >= best_lower - 1e-6 * max(1, abs(best_lower)), run
    print(json.dumps({
        'relative_gap': args.relative_gap,
        'before': before,
        'after': after,
        'median_before_seconds': before_time,
        'median_after_seconds': after_time,
        'reduction_percent': 100 * (1 - after_time / before_time),
    }, indent=2))


if __name__ == '__main__':
    main()
