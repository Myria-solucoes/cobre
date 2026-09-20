#!/usr/bin/env python3
"""Summarize training.py outputs. Requires pyarrow; does not certify policy quality."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import pyarrow.parquet as pq


def rows(path):
    return pq.ParquetFile(path).read().to_pylist()


def solver_profile(output, expected_solves):
    samples = rows(output / 'training/solver/iterations.parquet')
    assert sum(row['lp_solves'] for row in samples) == expected_solves, 'Incomplete solver telemetry'
    fields = ('lp_solves', 'simplex_iterations', 'retry_attempts', 'basis_consistency_failures',
              'solve_time_ms', 'load_model_time_ms', 'set_bounds_time_ms', 'basis_set_time_ms')
    phases = defaultdict(lambda: dict.fromkeys(fields, 0))
    stages = defaultdict(float)
    for row in samples:
        for field in fields:
            phases[row['phase']][field] += row[field]
        if row['stage_id'] is not None:
            stages[(row['phase'], row['stage_id'])] += row['solve_time_ms']
    return {
        'phases': dict(phases),
        'slowest_stages_by_cumulative_solver_time': [
            {'phase': phase, 'stage_id': stage, 'solve_time_ms': milliseconds}
            for (phase, stage), milliseconds in sorted(stages.items(), key=lambda item: -item[1])[:10]],
    }


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
        meta = json.loads((output / 'training/metadata.json').read_text())
        summary[arm] = {
            'completed_runs': len(runs),
            'median_wall_seconds': statistics.median(r['wall_seconds'] for r in runs),
            'median_training_seconds': statistics.median(timings),
            'training_seconds_range': [min(timings), max(timings)],
            'lp_solves': meta['solve_stats']['total_lp_solves'],
            'cuts_generated': meta['row_pool']['total_generated'],
            'final_lower_bound': meta['bounds']['final_lower_bound'],
            'solver_profile_first_repeat': solver_profile(output, meta['solve_stats']['total_lp_solves']),
        }
        if not (output / 'simulation/scenario_summary.parquet').exists():
            summary[arm]['quality_evaluated'] = False
            continue
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
        summary[arm].update({
            'quality_evaluated': True,
            'mean_simulated_cost': statistics.mean(costs.values()),
            'paired_cost_difference_percent': 100 * statistics.mean(differences) / mean_base,
            'paired_cost_difference_95ci_percent': [
                100 * (statistics.mean(differences) + sign * 1.96 * se) / mean_base for sign in (-1, 1)],
            'mean_deficit_mwh': sum(r['deficit_mwh'] for r in bus_rows) / len(costs) if bus_rows else None,
            'mean_terminal_storage_hm3': sum(terminal_storage.values()) / len(costs) if hydro_rows else None,
            'simulation_scenarios': len(costs),
        })
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2))
