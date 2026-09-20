#!/usr/bin/env python3
"""Exercise installed optimized CLI/wheel artifacts on a copied compact case."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile


def configure(case, dynamic):
    config_path = case / 'config.json'
    config = json.loads(config_path.read_text())
    training = config['training']
    training['selection'] = {'method': 'sampled', 'forward_passes': 8}
    training['stopping_rules'] = [{'type': 'iteration_limit', 'limit': 4}]
    training['forward_schedule'] = {
        'initial_passes': 1, 'growth_interval': 1, 'full_from_iteration': 4}
    training['parallelism'] = {'backward_scheduler': {
        'method': 'by_node', 'block_size': 5, 'point_block_size': 2}}
    training['backward_selection'] = {
        'initial_points': 1, 'exploration_points': 1, 'full_every': 2,
        'full_from_iteration': 3, 'deduplicate': True, 'audit_relative_tolerance': .01}
    training['cut_selection'] = {'selection': (
        {'method': 'dynamic', 'seed_window': 1, 'candidate_recency': None,
         'max_added_per_round': 2, 'adaptive_max_added_per_round': 8}
        if dynamic else {'method': 'lml1', 'check_frequency': 1})}
    config['simulation'] = {'enabled': False}
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    stages_path = case / 'stages.json'
    stages = json.loads(stages_path.read_text())
    for stage in stages['stages']:
        stage['risk_measure'] = {'cvar': {'alpha': .15, 'lambda': .4}}
    stages_path.write_text(json.dumps(stages, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', required=True, type=Path)
    parser.add_argument('--case', required=True, type=Path)
    args = parser.parse_args()
    import cobre

    results = []
    with tempfile.TemporaryDirectory(prefix='cobre-artifact-check-') as directory:
        root = Path(directory)
        for dynamic in (False, True):
            case = root / ('dynamic' if dynamic else 'frozen')
            shutil.copytree(args.case, case, ignore=shutil.ignore_patterns('output'))
            configure(case, dynamic)
            validation = cobre.io.validate(str(case))
            assert validation['valid'], validation
            cli_output = root / (case.name + '-cli')
            subprocess.run([str(args.cli.resolve()), 'run', str(case), '--output',
                            str(cli_output), '--threads', '2', '--comm-backend',
                            'local', '--color', 'never'], check=True, timeout=120)
            meta = json.loads((cli_output / 'training/metadata.json').read_text())
            assert meta['iterations']['completed'] == 4, meta
            cli_lb = meta['bounds']['final_lower_bound']
            assert math.isfinite(cli_lb)
            py_results = [cobre.run.run(str(case), output_dir=str(root / f'{case.name}-py-{threads}'),
                                       threads=threads, skip_simulation=True) for threads in (1, 2)]
            for result in py_results:
                assert result['iterations'] == 4, result
                assert math.isclose(result['lower_bound'], cli_lb, rel_tol=1e-8), (result, cli_lb)
            assert py_results[0]['lower_bound'] == py_results[1]['lower_bound'], py_results
            invalid_training = json.loads((case / 'config.json').read_text())['training']
            invalid_training['forward_schedule']['initial_passes'] = 9
            invalid = cobre.io.validate(str(case), config_overrides={'training': invalid_training})
            assert not invalid['valid'], invalid
            assert any('forward_schedule' in error.get('message', '') for error in invalid['errors']), invalid
            results.append({'dynamic': dynamic, 'cli_lower_bound': cli_lb,
                            'python_lower_bound': py_results[0]['lower_bound'],
                            'invalid_schedule_rejected': True})
    print(json.dumps({'artifact_checks': results}))


if __name__ == '__main__':
    main()
