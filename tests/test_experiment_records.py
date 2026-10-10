import argparse
import csv
import importlib.util
import logging
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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

    def test_append_csv_preserves_original_when_replace_fails(self):
        from experiments.run import append_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            original = "run_id,model\nold_run,VBPR\n"
            csv_path.write_text(original, encoding="utf-8")

            with patch(
                "experiments.run.os.replace",
                side_effect=OSError("simulated replacement failure"),
            ):
                with self.assertRaises(OSError):
                    append_csv(
                        csv_path,
                        {"run_id": "new_run", "model": "MGCN"},
                    )

            self.assertEqual(
                csv_path.read_text(encoding="utf-8"),
                original,
            )

            # Failed writes must not leave temporary CSV files behind.
            self.assertEqual(
                list(Path(tmpdir).iterdir()),
                [csv_path],
            )

    def test_run_experiment_marks_failed_when_recording_fails(self):
        import argparse
        import json

        from experiments.run import run_experiment

        fake_result = {
            "model": "VBPR",
            "dataset": "sports",
            "valid_metric": "recall@20",
            "resolved_config": {},
            "combinations": [],
            "best_by_validation_index": None,
        }

        args = argparse.Namespace(
            model="VBPR",
            dataset="sports",
            tag="failure-test",
            config=None,
            set_values=[],
            no_save_model=True,
            mg=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "test_run"

            with (
                patch(
                    "experiments.run.create_run_directory",
                    return_value=("test_run", run_dir),
                ),
                patch("experiments.run.snapshot_git"),
                patch(
                    "experiments.run.record_result_indexes",
                    side_effect=OSError("simulated CSV failure"),
                ),
                patch(
                    "utils.quick_start.quick_start",
                    return_value=fake_result,
                ),
            ):
                run_dir.mkdir()

                with self.assertRaises(OSError):
                    run_experiment(args)

            metadata = json.loads(
                (run_dir / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(metadata["status"], "failed")
            self.assertIn("simulated CSV failure", metadata["error"])

    def test_run_experiment_captures_logging_and_restores_handlers(self):
        from experiments.run import run_experiment

        fake_result = {
            "model": "VBPR",
            "dataset": "sports",
            "valid_metric": "recall@20",
            "resolved_config": {},
            "combinations": [],
            "best_by_validation_index": None,
        }

        args = argparse.Namespace(
            model="VBPR",
            dataset="sports",
            tag="logging-test",
            config=None,
            set_values=[],
            no_save_model=True,
            mg=False,
        )

        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level

        def fake_quick_start(*args, **kwargs):
            # Simulate MMRec's init_logger() creating a StreamHandler.
            logging.basicConfig(level=logging.INFO)
            logging.getLogger().info("MMREC_LOG_CAPTURE_TEST")
            print("MMREC_STDOUT_CAPTURE_TEST")
            return fake_result

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "test_run"
            run_dir.mkdir()

            with (
                patch(
                    "experiments.run.create_run_directory",
                    return_value=("test_run", run_dir),
                ),
                patch("experiments.run.snapshot_git"),
                patch(
                    "experiments.run.record_result_indexes",
                ),
                patch(
                    "utils.quick_start.quick_start",
                    side_effect=fake_quick_start,
                ),
            ):
                run_experiment(args)

            full_log = (run_dir / "full.log").read_text(
                encoding="utf-8"
            )

            self.assertIn("MMREC_LOG_CAPTURE_TEST", full_log)
            self.assertIn("MMREC_STDOUT_CAPTURE_TEST", full_log)

            metadata = json.loads(
                (run_dir / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["status"], "completed")

        self.assertEqual(root_logger.handlers, original_handlers)
        self.assertEqual(root_logger.level, original_level)
        
    def test_working_directory_restores_after_exception(self):
        from experiments.run import working_directory

        original = Path.cwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(RuntimeError):
                with working_directory(tmpdir):
                    self.assertEqual(Path.cwd().resolve(), Path(tmpdir).resolve())
                    raise RuntimeError("simulated failure")

        self.assertEqual(Path.cwd(), original)
        
    def test_structured_result_uses_model_name(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "utils"
            / "quick_start.py"
        ).read_text(encoding="utf-8")

        self.assertIn("'model': config['model']", source)

if __name__ == '__main__':
    unittest.main()
