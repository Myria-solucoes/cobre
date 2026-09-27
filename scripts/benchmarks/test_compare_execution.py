"""Protect the benchmark's numerical parity gate against timing-only changes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location('compare_execution', Path(__file__).with_name('compare_execution.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NumericalSignatureTest(unittest.TestCase):
    def test_ignores_only_timing_and_observes_cuts_values_and_solver_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'policy/cuts').mkdir(parents=True)
            (root / 'training/solver').mkdir(parents=True)
            (root / 'simulation/solver').mkdir(parents=True)
            cut = root / 'policy/cuts/000.bin'
            cut.write_bytes(b'cut-coefficients')
            (root / 'training/metadata.json').write_text(json.dumps({
                'bounds': {'final_lower_bound': 5.0}, 'row_pool': {'total_generated': 2},
                'iterations': {'completed': 2}, 'solve_stats': {
                    'total_lp_solves': 10, 'first_try': 10, 'retried': 0, 'failed': 0,
                },
            }))
            solver = {'phase': ['backward'], 'lp_solves': [10], 'simplex_iterations': [50],
                      'retry_attempts': [0], 'basis_consistency_failures': [0]}
            pq.write_table(pa.table(solver), root / 'training/solver/iterations.parquet')
            path = root / 'training/convergence.parquet'
            history = {'iteration': [1, 2], 'lower_bound': [3., 5.], 'time_total_ms': [10, 20]}
            pq.write_table(pa.table(history), path)
            simulation_path = root / 'simulation/solver/iterations.parquet'
            simulation = {'simplex_iterations': [50], 'solve_time_ms': [10.],
                          'load_model_time_ms': [2.], 'set_bounds_time_ms': [1.],
                          'basis_set_time_ms': [1.]}
            pq.write_table(pa.table(simulation), simulation_path)
            reference = module.numerical_signature(root)
            simulation['solve_time_ms'] = [20.]
            pq.write_table(pa.table(simulation), simulation_path)
            self.assertEqual(reference, module.numerical_signature(root))
            simulation['simplex_iterations'] = [51]
            pq.write_table(pa.table(simulation), simulation_path)
            self.assertNotEqual(reference, module.numerical_signature(root))
            simulation['simplex_iterations'] = [50]
            pq.write_table(pa.table(simulation), simulation_path)
            history['time_total_ms'] = [100, 200]
            pq.write_table(pa.table(history), path)
            self.assertEqual(reference, module.numerical_signature(root))
            history['lower_bound'] = [3., 5.001]
            pq.write_table(pa.table(history), path)
            self.assertNotEqual(reference, module.numerical_signature(root))
            history['lower_bound'] = [3., 5.]
            pq.write_table(pa.table(history), path)
            solver['simplex_iterations'] = [51]
            pq.write_table(pa.table(solver), root / 'training/solver/iterations.parquet')
            self.assertNotEqual(reference, module.numerical_signature(root))
            solver['simplex_iterations'] = [50]
            pq.write_table(pa.table(solver), root / 'training/solver/iterations.parquet')
            cut.write_bytes(b'changed-cut-coefficients')
            self.assertNotEqual(reference, module.numerical_signature(root))


if __name__ == '__main__':
    unittest.main()
