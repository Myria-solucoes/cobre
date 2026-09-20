#!/usr/bin/env python3
"""Serial, isolated training comparisons; keeps inputs, logs and commands."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=12)
    parser.add_argument('--forwards', type=int, default=24)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--arms', nargs='+', choices=['baseline', 'candidate', 'selected'], default=['baseline', 'candidate', 'selected'])
    parser.add_argument('--lml1-frequency', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--cut-method', choices=['lml1', 'dynamic'], default='lml1')
    parser.add_argument('--cvar', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--simulation-scheme', choices=['in_sample', 'out_of_sample'], default='in_sample')
    parser.add_argument('--simulation-seed', type=int, default=8675309)
    parser.add_argument('--scheduler', choices=['by_scenario', 'by_node'], default='by_scenario')
    parser.add_argument('--block-size', type=int, default=10)
    parser.add_argument('--deduplicate', action='store_true')
    parser.add_argument('--deduplicate-only', action='store_true', help='Process every distinct state in every iteration')
    parser.add_argument('--audit-relative-tolerance', type=float)
    parser.add_argument('--adaptive-max-added-per-round', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for repeat in range(args.repeats):
        arms = args.arms.copy()
        if repeat % 2:
            arms.reverse()
        for arm in arms:
            run = args.output / f'{repeat}-{arm}'
            case = run / 'case'
            shutil.copytree(args.case, case, ignore=shutil.ignore_patterns('output'))
            config = json.loads((case / 'config.json').read_text())
            config['training']['tree_seed'] = args.seed
            config['training'].setdefault('scenario_source', {})['seed'] = args.seed
            config['training']['parallelism'] = {'backward_scheduler': (
                {'method': 'by_node', 'block_size': args.block_size} if args.scheduler == 'by_node' else
                {'method': 'by_scenario'})}
            config['training']['selection'] = {'method': 'sampled', 'forward_passes': args.forwards}
            config['training']['stopping_rules'] = [{'type': 'iteration_limit', 'limit': args.iterations}]
            config['training']['cut_selection'] = {'selection': (
                {'method': 'lml1', 'check_frequency': args.lml1_frequency} if args.cut_method == 'lml1' else
                {'method': 'dynamic', 'candidate_recency': None, 'seed_window': 2, 'max_added_per_round': 20})}
            if args.cut_method == 'dynamic' and arm != 'baseline' and args.adaptive_max_added_per_round:
                config['training']['cut_selection']['selection']['adaptive_max_added_per_round'] = args.adaptive_max_added_per_round
            if args.cvar:
                stages = json.loads((case / 'stages.json').read_text())
                for stage in stages['stages']:
                    stage['risk_measure'] = {'cvar': {'alpha': 0.15, 'lambda': 0.4}}
                (case / 'stages.json').write_text(json.dumps(stages, indent=2) + '\n')
            config['training'].pop('backward_selection', None)
            config['simulation'] = {'enabled': True, 'selection': {'method': 'sampled', 'num_scenarios': 256},
                                    'scenario_source': {'seed': args.simulation_seed, 'inflow': {'scheme': args.simulation_scheme}}}
            if arm == 'selected':
                config['training']['backward_selection'] = {
                    'deduplicate': args.deduplicate, 'audit_relative_tolerance': args.audit_relative_tolerance,
                    'initial_points': max(1, args.forwards // 4), 'exploration_points': 2,
                    'full_every': 4, 'full_from_iteration': max(2, args.iterations * 2 // 3)}
            if arm == 'selected' and args.deduplicate_only:
                config['training']['backward_selection'].update(deduplicate=True, full_from_iteration=1)
            (case / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
            binary = (args.baseline if arm == 'baseline' else args.candidate).resolve()
            command = [str(binary), 'run', str(case.resolve()), '--output', str((run / 'output').resolve()),
                       '--threads', str(args.threads), '--comm-backend', 'local', '--color', 'never']
            start = time.monotonic()
            with (run / 'stdout.log').open('w') as stdout, (run / 'stderr.log').open('w') as stderr:
                try:
                    result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=args.timeout, check=False)
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    code = 124
            results.append({'arm': arm, 'repeat': repeat, 'wall_seconds': time.monotonic()-start,
                            'exit_code': code, 'command': command,
                            'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()})
            (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(results[-1]), flush=True)
            if code:
                raise SystemExit(code)


if __name__ == '__main__':
    main()
