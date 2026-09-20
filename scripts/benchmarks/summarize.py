#!/usr/bin/env python3
"""Summarize training.py outputs. Requires pyarrow; does not certify policy quality."""
import argparse
import json
from pathlib import Path
import statistics

import pyarrow.parquet as pq


def rows(path):
    return pq.ParquetFile(path).read().to_pylist()


def summarize(root):
    records = json.loads((root / 'results.json').read_text())
    summary = {}
    for arm in ('baseline', 'candidate', 'selected'):
        runs = [r for r in records if r['arm'] == arm and r['exit_code'] == 0]
        if not runs:
            continue
        timings = []
        for run in runs:
            output = root / f"{run['repeat']}-{arm}" / 'output'
            meta = json.loads((output / 'training/metadata.json').read_text())
            timings.append(meta['duration_seconds'])
        # Timing repeats use the same seed; they are not independent quality samples.
        output = root / f"{runs[0]['repeat']}-{arm}" / 'output'
        costs = {r['scenario_id']: r['discounted_immediate_cost']
                 for r in rows(output / 'simulation/scenario_summary.parquet')}
        reference = root / f"{runs[0]['repeat']}-baseline" / 'output'
        baseline = {r['scenario_id']: r['discounted_immediate_cost']
                    for r in rows(reference / 'simulation/scenario_summary.parquet')}
        assert costs.keys() == baseline.keys(), 'Scenario identity mismatch'
        assert rows(output / 'simulation/paths.parquet') == rows(reference / 'simulation/paths.parquet')
        differences = [costs[i] - baseline[i] for i in sorted(costs)]
        mean_base = statistics.mean(baseline.values())
        # Normal approximation, conditional on this one trained policy pair.
        se = statistics.stdev(differences) / len(differences) ** .5 if len(differences) > 1 else 0
        meta = json.loads((output / 'training/metadata.json').read_text())
        bus_rows = [r for p in (output / 'simulation/buses').glob('*/data.parquet') for r in rows(p)]
        hydro_rows = [r for p in (output / 'simulation/hydros').glob('*/data.parquet') for r in rows(p)]
        final_stage = max((r['stage_id'] for r in hydro_rows), default=None)
        # Storage is repeated per load block; take each plant/stage once.
        terminal_storage = {(r['scenario_id'], r['hydro_id']): r['storage_final_hm3']
                            for r in hydro_rows if r['stage_id'] == final_stage}
        summary[arm] = {
            'completed_runs': len(runs),
            'median_wall_seconds': statistics.median(r['wall_seconds'] for r in runs),
            'median_training_seconds': statistics.median(timings),
            'training_seconds_range': [min(timings), max(timings)],
            'lp_solves': meta['solve_stats']['total_lp_solves'],
            'cuts_generated': meta['row_pool']['total_generated'],
            'final_lower_bound': meta['bounds']['final_lower_bound'],
            'mean_simulated_cost': statistics.mean(costs.values()),
            'paired_cost_difference_percent': 100 * statistics.mean(differences) / mean_base,
            'paired_cost_difference_95ci_percent': [
                100 * (statistics.mean(differences) + sign * 1.96 * se) / mean_base for sign in (-1, 1)],
            'mean_deficit_mwh': sum(r['deficit_mwh'] for r in bus_rows) / len(costs) if bus_rows else None,
            'mean_terminal_storage_hm3': sum(terminal_storage.values()) / len(costs) if hydro_rows else None,
            'simulation_scenarios': len(costs),
        }
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2))
