#!/usr/bin/env python3
"""Exercise the shipped CLI/wheel pair, sparse policy resume and producer identity."""
import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import cobre
import cobre.results
import cobre.run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    args = parser.parse_args()
    cli = args.cli.resolve()
    with tempfile.TemporaryDirectory(prefix='cobre-018-pair-') as temporary:
        root = Path(temporary)
        case = root / 'case'
        shutil.copytree(args.case, case, ignore=shutil.ignore_patterns('output'))
        config = json.loads((case / 'config.json').read_text())
        config['training'].update({
            'selection': {'method': 'sampled', 'forward_passes': 8},
            'stopping_rules': [{'type': 'iteration_limit', 'limit': 4}],
            'forward_schedule': {'initial_passes': 2, 'growth_interval': 2, 'full_from_iteration': 4},
            'backward_selection': {'initial_points': 1, 'exploration_points': 1,
                                   'full_every': 4, 'full_from_iteration': 4},
        })
        config['simulation']['enabled'] = False
        config['policy'] = {'mode': 'fresh', 'checkpointing': {
            'enabled': True, 'interval_iterations': 1, 'store_basis': True,
        }}
        (case / 'config.json').write_text(json.dumps(config))
        outputs = [root / 'cli', root / 'python']
        subprocess.run([str(cli), 'run', str(case), '--output', str(outputs[0]),
                        '--threads', '2', '--quiet'], check=True, timeout=120)
        result = cobre.run.run(str(case), output_dir=str(outputs[1]), threads=2)
        assert result['iterations'] == 4, result
        producers = []
        for output in outputs:
            metadata = cobre.results.load_policy(output)['metadata']
            assert metadata['software'] == 'cobre-myria', metadata
            assert metadata['software_version'] == '0.18.0-myria.1', metadata
            assert metadata['producer']['completed_iterations'] == 4, metadata
            producers.append(metadata['producer'])
        assert math.isclose(producers[0]['final_lower_bound'], producers[1]['final_lower_bound'],
                            rel_tol=1e-9, abs_tol=1e-6), producers
        # Export compacts sparse cut slots, so incompatible cached bases must be omitted.
        config['policy']['mode'] = 'resume'
        config['training']['stopping_rules'][0]['limit'] = 6
        (case / 'config.json').write_text(json.dumps(config))
        subprocess.run([str(cli), 'validate', str(case), '--output', str(outputs[0])],
                       check=True, timeout=120)
        subprocess.run([str(cli), 'run', str(case), '--output', str(outputs[0]),
                        '--threads', '2', '--quiet'], check=True, timeout=120)
        resumed = cobre.run.run(str(case), output_dir=str(outputs[1]), threads=2)
        assert resumed['iterations'] == 6, resumed
        for output in outputs:
            assert cobre.results.load_policy(output)['metadata']['producer']['completed_iterations'] == 6
        print(json.dumps({'abi_version': cobre.__version__, 'software': 'cobre-myria',
                          'software_version': '0.18.0-myria.1', 'fresh_iterations': 4,
                          'resumed_iterations': 6, 'cli_python_parity': True}))


if __name__ == '__main__':
    main()
