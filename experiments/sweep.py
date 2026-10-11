#!/usr/bin/env python3
"""Schedule isolated MMRec combinations with strict, combination-level recovery."""
import argparse
import json
import math
import re
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import run as runner
from experiments.state import (atomic_json, code_identity, digest, exclusive_lock,
                               expand, portable_config, resolve_config)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError('Missing or damaged JSON: {}'.format(path)) from error
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: {}'.format(path))
    return value


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_result(result: dict, context: dict) -> dict:
    """Validate the actual child output, including both base and scalar configs."""
    config = context['resolved_config']
    metric = config['valid_metric'].lower()
    if (result.get('model') != config['model'] or result.get('dataset') != config['dataset']
            or result.get('valid_metric') != metric
            or result.get('valid_metric_bigger') != config['valid_metric_bigger']
            or result.get('hyperparameter_names') != config['hyper_parameters']
            or result.get('best_by_validation_index') != 0
            or portable_config(result.get('resolved_config', {})) != config):
        raise ValueError('Child result configuration/selection mismatch')
    combinations = result.get('combinations')
    if not isinstance(combinations, list) or len(combinations) != 1:
        raise ValueError('Child result must contain exactly one combination')
    item = combinations[0]
    expected_scalar = dict(config, **context['hyperparameters'])
    if (item.get('index') != 0 or item.get('hyperparameters') != context['hyperparameters']
            or portable_config(item.get('resolved_config', {})) != expected_scalar):
        raise ValueError('Child hyperparameters/resolved configuration mismatch')
    epoch = item.get('best_valid_epoch')
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError('Missing/invalid best validation epoch')
    for field in ('valid', 'test_upon_valid'):
        metrics = item.get(field)
        if (not isinstance(metrics, dict) or metric not in metrics or not metrics
                or not all(finite_number(value) for value in metrics.values())):
            raise ValueError('Missing or invalid {} metrics'.format(field))
        required = {str(name).lower() + '@' + str(k) for name in config['metrics'] for k in config['topk']}
        if not required.issubset(metrics):
            raise ValueError('Incomplete {} metrics'.format(field))
    if (not finite_number(item.get('best_valid_score'))
            or not math.isclose(item['best_valid_score'], item['valid'][metric], rel_tol=1e-6, abs_tol=1e-8)):
        raise ValueError('Validation score does not match selection metric')
    return item


def make_plan(args: argparse.Namespace) -> dict:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,119}', args.sweep_id):
        raise ValueError('--sweep-id must use 1–120 letters, digits, underscores or hyphens')
    overrides = runner.load_overrides(args.config, args.set_values)
    config = resolve_config(args.model, args.dataset, overrides, args.mg)
    options = {'mg': args.mg, 'save_model': not args.no_save_model}
    return {'schema_version': 1, 'sweep_id': args.sweep_id, 'model': args.model,
            'dataset': args.dataset, 'code_identity': code_identity(), 'resolved_config': config,
            'options': options, 'config_fingerprint': digest({'config': config, 'options': options}),
            'combinations': expand(config)}


def combination_context(plan: dict, index: int) -> dict:
    parameters = plan['combinations'][index]
    config = dict(plan['resolved_config'])
    config.update({key: [value] for key, value in parameters.items()})
    if len(expand(config)) != 1:
        raise ValueError('Single-combination configuration expanded unexpectedly')
    return {'sweep_id': plan['sweep_id'], 'combination_index': index,
            'config_fingerprint': plan['config_fingerprint'], 'code_identity': plan['code_identity'],
            'options': plan['options'], 'hyperparameters': parameters, 'resolved_config': config}


def record_path(directory: Path, index: int) -> Path:
    return directory / 'combinations' / '{:03d}'.format(index) / 'record.json'


def completed_record(record: dict, context: dict) -> dict:
    run_id = record.get('run_id', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid run ID in combination record')
    run_dir = ROOT / 'experiments' / 'runs' / run_id
    metadata = read_json(run_dir / 'metadata.json')
    if (metadata.get('status') != 'completed' or metadata.get('run_id') != run_id
            or metadata.get('model') != context['resolved_config']['model']
            or metadata.get('dataset') != context['resolved_config']['dataset']
            or metadata.get('sweep_context') != context):
        raise ValueError('Run metadata is incomplete or incompatible: ' + run_id)
    commit = (run_dir / 'git_commit.txt').read_text(encoding='utf-8').strip()
    if commit != context['code_identity']['git_commit']:
        raise ValueError('Run Git commit mismatch: ' + run_id)
    result = read_json(run_dir / 'results.json')
    item = validate_result(result, context)
    if portable_config(read_json(run_dir / 'resolved_config.json')) != context['resolved_config']:
        raise ValueError('Run configuration snapshot mismatch: ' + run_id)
    result_hash = digest(result)
    if record['status'] == 'completed':
        if record.get('result_fingerprint') != result_hash:
            raise ValueError('Completed results changed: ' + run_id)
        for key in ('best_valid_score', 'best_valid_epoch', 'valid', 'test_upon_valid'):
            if record.get(key) != item[key]:
                raise ValueError('Completed record metrics changed: ' + run_id)
    return dict(record, status='completed', result_fingerprint=result_hash,
                **{key: item[key] for key in ('best_valid_score', 'best_valid_epoch', 'valid', 'test_upon_valid')})


def scan_records(directory: Path, plan: dict) -> list:
    records = []
    for index, parameters in enumerate(plan['combinations']):
        context = combination_context(plan, index)
        path = record_path(directory, index)
        if not path.exists():
            records.append({'schema_version': 1, 'combination_index': index,
                            'hyperparameters': parameters, 'context': context, 'status': 'pending', 'attempts': []})
            continue
        record = read_json(path)
        if record.get('run_id') and not re.fullmatch(r'[A-Za-z0-9_-]+', record['run_id']):
            raise ValueError('Invalid run ID in record')
        if (record.get('schema_version') != 1 or record.get('combination_index') != index
                or record.get('hyperparameters') != parameters or record.get('context') != context):
            raise ValueError('Combination configuration/code mismatch: {}'.format(path))
        status = record.get('status')
        if status == 'completed':
            record = completed_record(record, context)
        elif status in ('running', 'interrupted', 'failed'):
            # Recover a child that committed all artifacts before its parent died.
            metadata_path = ROOT / 'experiments' / 'runs' / record.get('run_id', '') / 'metadata.json'
            if metadata_path.exists() and read_json(metadata_path).get('status') == 'completed':
                record = completed_record(record, context)
            elif status == 'running':
                record = dict(record, status='interrupted')
        elif status != 'pending':
            raise ValueError('Unknown combination status: {}'.format(status))
        records.append(record)
    return records


def summarize(plan: dict, records: list, status: str) -> dict:
    completed = [record for record in records if record['status'] == 'completed']
    best = None
    if completed:
        choose = max if plan['resolved_config']['valid_metric_bigger'] else min
        best = choose(completed, key=lambda record: record['best_valid_score'])['combination_index']
    return {'schema_version': 1, 'sweep_id': plan['sweep_id'], 'model': plan['model'], 'dataset': plan['dataset'],
            **plan['code_identity'], 'config_fingerprint': plan['config_fingerprint'],
            'selection_metric': plan['resolved_config']['valid_metric'].lower(),
            'total_combinations': len(records), 'completed_combinations': len(completed),
            'failed_combinations': sum(record['status'] == 'failed' for record in records),
            'best_by_validation_index': best, 'status': status}


def preview_import(path: str, plan: dict) -> None:
    """Read-only preview. Even complete provenance is not automatically trusted."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(payload, list):
        payload = {'results': payload}
    if not isinstance(payload, dict):
        raise ValueError('Recovered results must be a JSON object or list')
    required = {'git_commit', 'model', 'dataset', 'resolved_config', 'seed', 'search_space',
                'valid_metric', 'source'}
    missing = sorted(key for key in required if key not in payload or payload[key] is None)
    mismatches = []
    expected = {'git_commit': plan['code_identity']['git_commit'], 'model': plan['model'],
                'dataset': plan['dataset'], 'resolved_config': plan['resolved_config'],
                'seed': plan['resolved_config']['seed'], 'search_space': plan['combinations'],
                'valid_metric': plan['resolved_config']['valid_metric'].lower()}
    for key, value in expected.items():
        if key in payload and payload[key] != value:
            mismatches.append(key)
    print(json.dumps({'source': payload.get('source', 'recovered_from_full_log'), 'verified': False,
                      'missing_provenance': missing, 'mismatched_provenance': mismatches,
                      'message': 'Preview only; no completed records imported or skipped.',
                      'results': payload.get('combinations', payload.get('results', []))}, indent=2))


@contextmanager
def interruption_handlers():
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Received signal {}'.format(signum))
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def execute_sweep(args: argparse.Namespace) -> int:
    plan = make_plan(args)
    directory = ROOT / 'experiments' / 'sweeps' / args.sweep_id
    if args.import_results:
        preview_import(args.import_results, plan)
        return 0
    if directory.exists():
        if not args.resume:
            raise ValueError('Sweep already exists; use --resume with identical configuration')
        if read_json(directory / 'sweep_config.json') != plan:
            raise ValueError('Sweep code/configuration differs; restore original inputs or use a new sweep ID')
    elif args.resume:
        raise ValueError('--resume requires an existing sweep ID')
    records = scan_records(directory, plan)
    pending = [record for record in records if record['status'] != 'completed']
    planned = len(pending) if args.max_combinations is None else min(len(pending), args.max_combinations)
    print('Sweep: {}\nModel: {}\nDataset: {}\nTotal combinations: {}\nCompleted: {}\nPending: {}\nPlanned this invocation: {}\nConfig fingerprint: {}\nGit commit: {}\nSource fingerprint: {}'.format(
        args.sweep_id, args.model, args.dataset, len(records), len(records) - len(pending), len(pending),
        planned, plan['config_fingerprint'], plan['code_identity']['git_commit'], plan['code_identity']['source_fingerprint']))
    for record in records:
        print('[{}] {} {}'.format(record['combination_index'], record['hyperparameters'], record['status']))
    if args.max_runtime_minutes is not None:
        print('Time budget only stops NEW combinations. A running combination can exceed the budget.')
    if args.dry_run:
        return 0
    sweeps_root = directory.parent
    sweeps_root.mkdir(parents=True, exist_ok=True)
    # One scheduler across this checkout also protects the shared CSV indexes.
    with exclusive_lock(sweeps_root / '.scheduler.lock') as lock_fd, interruption_handlers():
        if directory.exists():
            if not args.resume or read_json(directory / 'sweep_config.json') != plan:
                raise ValueError('Sweep changed while acquiring lock')
            records = scan_records(directory, plan)
        else:
            directory.mkdir(exist_ok=False)
            atomic_json(directory / 'sweep_config.json', plan)
        return dispatch(args, directory, plan, records, lock_fd)


def launch_child(command: list, lock_fd: int) -> int:
    # Keep the scheduler lock alive in the child even if the parent is SIGKILLed.
    # A new session lets us stop the entire training process group on handled signals.
    child = subprocess.Popen(command, cwd=ROOT, start_new_session=True, pass_fds=(lock_fd,))
    try:
        return child.wait()
    except BaseException:
        if child.poll() is None:
            os_signal_group(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os_signal_group(child.pid, signal.SIGKILL)
                child.wait()
        raise


def os_signal_group(pid: int, sig: int) -> None:
    import os
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def dispatch(args: argparse.Namespace, directory: Path, plan: dict, records: list, lock_fd: int) -> int:
    start = time.monotonic()
    count = 0
    failure_code = 0
    active = None
    manifest = directory / 'manifest.json'
    try:
        for record in records:
            path = record_path(directory, record['combination_index'])
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(path, record)
        atomic_json(manifest, summarize(plan, records, 'running'))
        for record in records:
            if record['status'] == 'completed':
                continue
            if args.max_combinations is not None and count >= args.max_combinations:
                break
            if args.max_runtime_minutes is not None and time.monotonic() - start >= args.max_runtime_minutes * 60:
                break
            if code_identity() != plan['code_identity']:
                raise ValueError('Code changed during sweep')
            active = record
            index = record['combination_index']
            path = record_path(directory, index)
            context = record['context']
            child_config = path.parent / 'config.json'
            context_path = path.parent / 'context.json'
            atomic_json(child_config, context['resolved_config'])
            atomic_json(context_path, context)
            run_id = 'sweep_{}_{}_{:s}'.format(args.sweep_id, index, uuid.uuid4().hex[:12])
            record.setdefault('attempts', []).append(run_id)
            record.update(status='running', run_id=run_id)
            record.pop('error', None)
            atomic_json(path, record)
            command = [sys.executable, '-u', str(ROOT / 'experiments' / 'run.py'),
                       '--model', args.model, '--dataset', args.dataset, '--tag', args.sweep_id,
                       '--config', str(child_config), '--single-combination', '--sweep-context', str(context_path),
                       '--run-dir', str(ROOT / 'experiments' / 'runs' / run_id)]
            if args.no_save_model:
                command.append('--no-save-model')
            if args.mg:
                command.append('--mg')
            count += 1
            try:
                # Inherited streams keep terminal output live; run.py archives full.log.
                returncode = launch_child(command, lock_fd)
                if returncode:
                    failure_code = returncode if returncode > 0 else 128 - returncode
                    raise RuntimeError('Child exited with code {}'.format(returncode))
                complete = completed_record(record, context)
                if code_identity() != plan['code_identity']:
                    raise ValueError('Source changed during child execution')
                record.update(complete)
            except Exception as error:
                failure_code = failure_code or 1
                record.update(status='failed', error='{}: {}'.format(type(error).__name__, error))
            atomic_json(path, record)
            atomic_json(manifest, summarize(plan, records, 'running'))
            active = None
            if failure_code and not args.continue_on_error:
                break
        status = ('completed' if all(record['status'] == 'completed' for record in records)
                  else 'failed' if failure_code else 'paused')
        atomic_json(manifest, summarize(plan, records, status))
        return failure_code
    except BaseException as error:
        if active is not None and active['status'] == 'running':
            active.update(status='interrupted', error='{}: {}'.format(type(error).__name__, error))
            atomic_json(record_path(directory, active['combination_index']), active)
        atomic_json(manifest, summarize(plan, records, 'interrupted'))
        raise


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', '-m', required=True)
    parser.add_argument('--dataset', '-d', required=True)
    parser.add_argument('--sweep-id', required=True)
    parser.add_argument('--config')
    parser.add_argument('--set', dest='set_values', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-combinations', type=int)
    parser.add_argument('--max-runtime-minutes', type=float)
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument('--no-save-model', action='store_true')
    parser.add_argument('--mg', action='store_true')
    parser.add_argument('--import-results', help='Read-only provenance/metrics preview; does not import')
    args = parser.parse_args(argv)
    if args.max_combinations is not None and args.max_combinations < 1:
        parser.error('--max-combinations must be positive')
    if args.max_runtime_minutes is not None and (not math.isfinite(args.max_runtime_minutes) or args.max_runtime_minutes <= 0):
        parser.error('--max-runtime-minutes must be finite and positive')
    return args


if __name__ == '__main__':
    try:
        sys.exit(execute_sweep(parse_args()))
    except KeyboardInterrupt:
        print('Sweep interrupted; resume retries incomplete combinations.', file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print('Sweep failed: {}'.format(error), file=sys.stderr)
        sys.exit(1)
