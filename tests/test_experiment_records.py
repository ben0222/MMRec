import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('experiment_runner', REPO_ROOT / 'experiments' / 'run.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ExperimentRecordTests(unittest.TestCase):
    def test_append_csv_preserves_existing_history_when_columns_expand(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'results.csv'
            path.write_text('run_id,legacy_metric\nold,0.1\n', encoding='utf-8')
            runner.append_csv(path, {'run_id': 'new', 'new_metric': 0.2})
            with path.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]['run_id'], 'old')
            self.assertEqual(rows[0]['legacy_metric'], '0.1')
            self.assertEqual(rows[1]['new_metric'], '0.2')

    def test_validation_selection_is_independent_of_test_scores(self):
        result = {
            'model': 'MGCN',
            'dataset': 'sports',
            'valid_metric': 'recall@20',
            'best_by_validation_index': 1,
            'combinations': [
                {'index': 0, 'hyperparameters': {'cl_loss': 0.001}, 'best_valid_score': 0.1,
                 'best_valid_epoch': 4, 'valid': {'recall@20': 0.1}, 'test_upon_valid': {'recall@20': 0.9}},
                {'index': 1, 'hyperparameters': {'cl_loss': 0.01}, 'best_valid_score': 0.2,
                 'best_valid_epoch': 5, 'valid': {'recall@20': 0.2}, 'test_upon_valid': {'recall@20': 0.1}},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            previous_root = runner.REPO_ROOT
            runner.REPO_ROOT = Path(temporary_directory)
            try:
                runner.record_result_indexes('run-1', 'test', result)
                with (runner.REPO_ROOT / 'experiments' / 'results.csv').open(newline='', encoding='utf-8') as handle:
                    row = next(csv.DictReader(handle))
            finally:
                runner.REPO_ROOT = previous_root
        self.assertEqual(row['selected_combination_index'], '1')
        self.assertIn('0.01', row['hyperparameters_json'])
        self.assertEqual(row['legacy_console_hyperparameter_selection'], 'test_metric')


if __name__ == '__main__':
    unittest.main()
