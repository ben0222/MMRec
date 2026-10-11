"""CPU-only integration tests: real configs/records, mocked training and children."""
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from experiments import run as runner, sweep, state


IDENTITY = {'git_commit': 'abc123', 'source_fingerprint': 'source123'}


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        for target, value in [('experiments.sweep.ROOT', self.root),
                              ('experiments.run.REPO_ROOT', self.root)]:
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # Config still reads the actual src tree; all generated outputs use the temp root.
        for target in ('experiments.sweep.code_identity', 'experiments.run.code_identity'):
            patcher = patch(target, return_value=IDENTITY)
            patcher.start()
            self.addCleanup(patcher.stop)
        (self.root / 'src').mkdir()
        self.calls = []

    def args(self, *extra):
        return sweep.parse_args(['--model', 'VBPR', '--dataset', 'sports', '--sweep-id', 'test',
                                 '--set', 'use_gpu=false', *extra])

    @property
    def directory(self):
        return self.root / 'experiments' / 'sweeps' / 'test'

    def fake_child(self, command, lock_fd):
        self.calls.append(command)
        self.assertEqual(command[0], sys.executable)
        args = runner.parse_args(command[3:])
        context = sweep.read_json(Path(args.sweep_context))
        config = context['resolved_config']
        parameters = context['hyperparameters']
        score = 0.2 if parameters['reg_weight'] == 2.0 else 0.1
        result = {'model': 'VBPR', 'dataset': 'sports', 'valid_metric': 'recall@20',
                  'valid_metric_bigger': True, 'hyperparameter_names': config['hyper_parameters'],
                  'resolved_config': dict(config, device='cpu'), 'best_by_validation_index': 0,
                  'combinations': [{'index': 0, 'hyperparameters': parameters,
                                    'resolved_config': dict(config, **parameters),
                                    'best_valid_epoch': 5, 'best_valid_score': score,
                                    'valid': {name.lower() + '@' + str(k): score for name in config['metrics'] for k in config['topk']},
                                    'test_upon_valid': {name.lower() + '@' + str(k): 1 - score for name in config['metrics'] for k in config['topk']}}]}

        def snapshot(path):
            (path / 'git_commit.txt').write_text(IDENTITY['git_commit'])

        def train(*args, **kwargs):
            print('MOCK TRAINING OUTPUT')
            return result

        with patch('experiments.run.snapshot_git', side_effect=snapshot), patch('utils.quick_start.quick_start', side_effect=train):
            runner.run_experiment(args)
        return 0

    def execute(self, *extra, child=None):
        with patch('experiments.sweep.launch_child', side_effect=child or self.fake_child), patch('sys.stdout', io.StringIO()):
            return sweep.execute_sweep(self.args(*extra))

    def test_real_vbpr_config_expands_seven_single_combinations(self):
        plan = sweep.make_plan(self.args())
        self.assertEqual([item['reg_weight'] for item in plan['combinations']], [2., 1., .1, .01, .001, .0001, .00001])
        for index in range(7):
            context = sweep.combination_context(plan, index)
            self.assertEqual(len(state.expand(context['resolved_config'])), 1)
            self.assertEqual(context['resolved_config']['seed'], [999])

    def test_yaml_set_and_mg_merge_use_real_config(self):
        config_path = self.root / 'override.yaml'
        config_path.write_text('reg_weight: [3, 4]\nepochs: 2\n')
        plan = sweep.make_plan(self.args('--config', str(config_path), '--set', 'epochs=9', '--mg'))
        self.assertEqual(plan['resolved_config']['epochs'], 9)
        self.assertEqual(len(plan['combinations']), 6)
        self.assertEqual(plan['resolved_config']['alpha2'], [.1, .2, .3])
        self.assertTrue(plan['options']['mg'])

    def test_dry_run_has_no_outputs_or_training(self):
        self.assertEqual(self.execute('--dry-run'), 0)
        self.assertFalse(self.calls)
        self.assertFalse((self.root / 'experiments').exists())

    def test_maximum_resume_csv_and_validation_selection(self):
        self.assertEqual(self.execute('--max-combinations', '1'), 0)
        first = sweep.read_json(sweep.record_path(self.directory, 0))
        self.assertEqual(first['status'], 'completed')
        self.assertEqual(sweep.read_json(self.directory / 'manifest.json')['status'], 'paused')
        self.assertEqual(self.execute('--resume'), 0)
        self.assertEqual(len(self.calls), 7)
        manifest = sweep.read_json(self.directory / 'manifest.json')
        self.assertEqual(manifest['completed_combinations'], 7)
        self.assertEqual(manifest['best_by_validation_index'], 0)
        self.assertEqual(self.execute('--resume'), 0)
        self.assertEqual(len(self.calls), 7)
        for name in ('runs.csv', 'results.csv'):
            with (self.root / 'experiments' / name).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 7)
            self.assertEqual(len({row['run_id'] for row in rows}), 7)
        self.assertIn('MOCK TRAINING OUTPUT', (self.root / 'experiments' / 'runs' / first['run_id'] / 'full.log').read_text())

    def test_failure_preserves_completed_and_propagates_exit_code(self):
        def child(command, fd):
            if len(self.calls) == 1:
                self.calls.append(command)
                return 17
            return self.fake_child(command, fd)
        self.assertEqual(self.execute(child=child), 17)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(sweep.read_json(sweep.record_path(self.directory, 0))['status'], 'completed')
        self.assertEqual(sweep.read_json(sweep.record_path(self.directory, 1))['status'], 'failed')
        self.assertEqual(self.execute('--resume', '--max-combinations', '1'), 0)
        record = sweep.read_json(sweep.record_path(self.directory, 1))
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(len(record['attempts']), 2)

    def test_continue_on_error_finishes_other_combinations(self):
        def child(command, fd):
            if not self.calls:
                self.calls.append(command)
                return 5
            return self.fake_child(command, fd)
        self.assertEqual(self.execute('--continue-on-error', child=child), 5)
        self.assertEqual(len(self.calls), 7)
        self.assertEqual(sweep.read_json(self.directory / 'manifest.json')['completed_combinations'], 6)

    def test_interrupted_child_retried(self):
        with self.assertRaises(KeyboardInterrupt):
            self.execute(child=lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
        self.assertEqual(sweep.read_json(self.directory / 'manifest.json')['status'], 'interrupted')
        self.assertEqual(self.execute('--resume', '--max-combinations', '1'), 0)
        self.assertEqual(len(self.calls), 1)

    def test_completed_child_after_parent_crash_is_adopted(self):
        self.execute('--max-combinations', '1')
        path = sweep.record_path(self.directory, 0)
        record = sweep.read_json(path)
        record['status'] = 'running'
        state.atomic_json(path, record)
        self.execute('--resume', '--max-combinations', '1')
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(sweep.read_json(path)['status'], 'completed')

    def test_changed_config_commit_and_dirty_source_rejected(self):
        self.execute('--max-combinations', '1')
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.execute('--resume', '--set', 'epochs=3')
        for identity in (dict(IDENTITY, git_commit='changed'), dict(IDENTITY, source_fingerprint='dirty')):
            with patch('experiments.sweep.code_identity', return_value=identity):
                with self.assertRaisesRegex(ValueError, 'differs'):
                    self.execute('--resume')

    def test_missing_or_corrupt_completed_results_are_never_skipped(self):
        self.execute('--max-combinations', '1')
        record = sweep.read_json(sweep.record_path(self.directory, 0))
        result_path = self.root / 'experiments' / 'runs' / record['run_id'] / 'results.json'
        for invalid in ('{', '{}'):
            result_path.write_text(invalid)
            with self.assertRaises(ValueError):
                self.execute('--resume')
        result_path.unlink()
        with self.assertRaises(ValueError):
            self.execute('--resume')
        self.assertEqual(len(self.calls), 1)

    def test_corrupt_record_and_metric_tampering_rejected(self):
        self.execute('--max-combinations', '1')
        path = sweep.record_path(self.directory, 0)
        record = sweep.read_json(path)
        record['best_valid_score'] = 999
        state.atomic_json(path, record)
        with self.assertRaisesRegex(ValueError, 'metrics changed'):
            self.execute('--resume')
        path.write_text('{')
        with self.assertRaises(ValueError):
            self.execute('--resume')

    def test_runtime_budget_does_not_start_next_child(self):
        with patch('experiments.sweep.time.monotonic', side_effect=[0, 0, 61]):
            self.execute('--max-runtime-minutes', '1')
        self.assertEqual(len(self.calls), 1)

    def test_existing_directory_is_not_overwritten(self):
        self.execute('--max-combinations', '1')
        before = (self.directory / 'sweep_config.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.execute()
        self.assertEqual((self.directory / 'sweep_config.json').read_bytes(), before)

    def test_import_preview_never_creates_completed_records(self):
        path = self.root / 'recovered.json'
        path.write_text(json.dumps({'results': [{'reg_weight': 2, 'valid': {'recall@20': .0828}}]}))
        output = io.StringIO()
        with patch('sys.stdout', output):
            sweep.execute_sweep(self.args('--import-results', str(path)))
        preview = json.loads(output.getvalue())
        self.assertFalse(preview['verified'])
        self.assertIn('git_commit', preview['missing_provenance'])
        self.assertFalse(self.directory.exists())

    def test_runner_single_guard_rejects_before_training(self):
        with patch('utils.quick_start.quick_start') as train:
            with self.assertRaisesRegex(ValueError, 'multiple combinations'):
                runner.run_experiment(runner.parse_args(['--model', 'VBPR', '--dataset', 'sports', '--single-combination']))
            train.assert_not_called()
        self.assertFalse((self.root / 'experiments').exists())

    def test_invalid_search_and_ids_rejected(self):
        for options in (['--sweep-id', '../bad'], ['--set', 'seed=999'], ['--set', 'reg_weight=[1,1]']):
            with self.assertRaises(ValueError):
                self.execute('--dry-run', *options)

    def test_smaller_validation_metric_selection(self):
        plan = sweep.make_plan(self.args())
        plan['resolved_config']['valid_metric_bigger'] = False
        records = [{'status': 'completed', 'best_valid_score': .4, 'combination_index': 0},
                   {'status': 'completed', 'best_valid_score': .1, 'combination_index': 1}]
        self.assertEqual(sweep.summarize(plan, records, 'completed')['best_by_validation_index'], 1)


class PersistenceAndLauncherTests(unittest.TestCase):
    def test_json_overrides_preserve_float_and_pad_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'config.json'
            path.write_text('{"layer_norm_eps": 1e-12, "pad_token_id": 0}')
            values = runner.load_overrides(str(path), [])
            self.assertEqual(values['layer_norm_eps'], 1e-12)
            self.assertEqual(runner.json_safe(values)['pad_token_id'], 0)
            self.assertEqual(runner.json_safe({'api_token': 'secret'})['api_token'], '<redacted>')

    def test_reindexing_same_run_is_idempotent_and_conflict_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'runs.csv'
            row = {'run_id': 'test', 'combination_index': 0, 'score': .2}
            runner.append_csv(path, row)
            runner.append_csv(path, row)
            with path.open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                runner.append_csv(path, dict(row, score=.3))

    def test_atomic_json_failure_preserves_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'record.json'
            state.atomic_json(path, {'original': True})
            original = path.read_bytes()
            with patch('experiments.state.os.replace', side_effect=OSError('disk error')):
                with self.assertRaises(OSError):
                    state.atomic_json(path, {'original': False})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_lock_prevents_concurrent_scheduler(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'lock'
            with state.exclusive_lock(path):
                with self.assertRaisesRegex(ValueError, 'active'):
                    with state.exclusive_lock(path):
                        pass

    def test_child_inherits_lock_and_streams_without_shell(self):
        child = Mock()
        child.wait.return_value = 9
        with patch('experiments.sweep.subprocess.Popen', return_value=child) as popen:
            self.assertEqual(sweep.launch_child([sys.executable, 'fake.py'], 42), 9)
        kwargs = popen.call_args.kwargs
        self.assertEqual(kwargs['pass_fds'], (42,))
        self.assertTrue(kwargs['start_new_session'])
        self.assertNotIn('shell', kwargs)
        self.assertNotIn('stdout', kwargs)

    def test_child_signal_cleanup(self):
        child = Mock(pid=123)
        child.wait.side_effect = [KeyboardInterrupt(), 0]
        child.poll.return_value = None
        with patch('experiments.sweep.subprocess.Popen', return_value=child), patch('experiments.sweep.os_signal_group') as stop:
            with self.assertRaises(KeyboardInterrupt):
                sweep.launch_child(['fake'], 42)
        stop.assert_called_once()

    def test_autodl_check_only_is_safe_on_mac(self):
        script = Path(__file__).resolve().parents[1] / 'experiments' / 'run_autodl.sh'
        subprocess.run(['bash', '-n', str(script)], check=True)
        result = subprocess.run(['bash', str(script), '--check-only', '--model', 'VBPR'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('Preview only', result.stdout)

    def test_autodl_worker_success_failure_and_backup_with_mock_commands(self):
        original = (Path(__file__).resolve().parents[1] / 'experiments' / 'run_autodl.sh').read_text()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'experiments').mkdir()
            binary = root / 'bin'
            binary.mkdir()
            events = root / 'events'
            # Only this temporary script bypasses platform checks; its shutdown
            # path is replaced before execution. No production shutdown is reachable.
            shutdown = binary / 'mock_shutdown'
            shutdown.write_text('#!/bin/bash\necho shutdown >> "$MOCK_EVENTS"\nexit "${MOCK_SHUTDOWN_CODE:-0}"\n')
            shutdown.chmod(0o755)
            source = original.replace('/usr/bin/shutdown', str(shutdown))
            guard = next(line for line in source.splitlines() if line.startswith('if [[ "$(uname -s)"'))
            source = source.replace(guard, 'if false; then')
            script = root / 'experiments' / 'run_autodl.sh'
            script.write_text(source)
            for name, command in [('setsid', 'exec "$@"'), ('flock', 'exit 0'),
                                  ('sync', 'echo sync >> "$MOCK_EVENTS"'),
                                  ('tar', 'echo backup >> "$MOCK_EVENTS"')]:
                path = binary / name
                path.write_text('#!/bin/bash\n' + command + '\n')
                path.chmod(0o755)
            (root / 'experiments' / 'sweep.py').write_text('import os\nprint("MOCK WORKER")\nraise SystemExit(int(os.environ["MOCK_TRAINING_CODE"]))\n')
            (root / 'experiments' / 'runs').mkdir()
            for code, shutdown_code in [(0, 0), (7, 0), (9, 3)]:
                launch = root / ('launch-' + str(code))
                launch.mkdir()
                events.write_text('')
                env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
                           AUTODL_PYTHON=sys.executable, AUTODL_BACKUP_DIR=str(root / 'backup'),
                           MOCK_EVENTS=str(events), MOCK_TRAINING_CODE=str(code),
                           MOCK_SHUTDOWN_CODE=str(shutdown_code))
                result = subprocess.run(['bash', str(script), '--worker', str(launch)],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertIn('training_exit_code=' + str(code), (launch / 'exit_status.txt').read_text())
                self.assertEqual(events.read_text().splitlines()[:3], ['backup', 'sync', 'shutdown'])
                self.assertIn('shutdown_exit_code=' + str(shutdown_code), (launch / 'exit_status.txt').read_text())


if __name__ == '__main__':
    unittest.main()
