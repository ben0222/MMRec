#!/usr/bin/env python3
"""Run MMRec and persist phase-one, reproducible experiment records.

This wrapper intentionally delegates training to ``quick_start`` unchanged.
It records both the legacy test-selected result and the validation-selected
result so that historical MMRec behaviour is never confused with reporting.
"""
import argparse
import csv
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
import tempfile
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
@contextmanager
def working_directory(path):
    """Temporarily change the working directory."""
    previous = Path.cwd()

    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)
        
class TeeStream:
    """Write output to both the terminal and an experiment log file."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text):
        self.terminal.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()

    @property
    def encoding(self):
        return self.terminal.encoding

SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


SENSITIVE_KEY_PARTS = ('api_key', 'password', 'passwd', 'secret', 'token')
SOURCE_SUFFIXES = {'.py', '.yaml', '.yml', '.json', '.sh'}


def json_safe(value, key=None):
    """Make values serializable and redact likely credentials in snapshots."""
    if key and any(part in key.lower() for part in SENSITIVE_KEY_PARTS):
        return '<redacted>'
    if isinstance(value, dict):
        return {str(k): json_safe(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, 'item'):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path, payload):
    with path.open('w', encoding='utf-8') as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write('\n')


def command_output(args):
    completed = subprocess.run(args, cwd=REPO_ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               check=False)
    return completed.stdout if completed.returncode == 0 else completed.stderr


def snapshot_git(run_dir):
    """Save revision state without adding any artifact back to Git."""
    (run_dir / 'git_commit.txt').write_text(command_output(['git', 'rev-parse', 'HEAD']).strip() + '\n', encoding='utf-8')
    (run_dir / 'git_status.txt').write_text(command_output(['git', 'status', '--short']), encoding='utf-8')
    (run_dir / 'code_diff.patch').write_text(command_output(['git', 'diff', '--binary', 'HEAD']), encoding='utf-8')

    untracked_root = run_dir / 'untracked_source'
    for relative_name in command_output(['git', 'ls-files', '--others', '--exclude-standard']).splitlines():
        relative_path = Path(relative_name)
        source = REPO_ROOT / relative_path
        if source.suffix.lower() not in SOURCE_SUFFIXES or not source.is_file():
            continue
        destination = untracked_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def load_overrides(config_path, set_values):
    import yaml

    overrides = {}
    if config_path:
        with Path(config_path).open(encoding='utf-8') as handle:
            overrides = yaml.safe_load(handle) or {}
        if not isinstance(overrides, dict):
            raise ValueError('--config must contain a YAML mapping')
    for assignment in set_values:
        key, separator, raw_value = assignment.partition('=')
        if not separator or not key:
            raise ValueError('--set must use KEY=VALUE')
        overrides[key] = yaml.safe_load(raw_value)
    return overrides


def append_csv(path, row):
    """
    Append one experiment record while preserving existing data.

    Supports schema evolution by adding new columns.
    Uses atomic replacement to avoid corrupting existing CSV files.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = []
    fieldnames = []

    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)

    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(row)

            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, path)

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def compact(value):
    return json.dumps(json_safe(value), sort_keys=True, separators=(',', ':'))


def metric_fields(prefix, result):
    return {prefix + key.replace('@', '_at_'): value for key, value in result.items()}


def create_run_directory(model, dataset, tag, runs_root):
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_tag = ''.join(char if char.isalnum() or char in '-_' else '_' for char in tag)
    run_id = '{}_{}_{}_{}_{}'.format(timestamp, model, dataset, safe_tag, uuid.uuid4().hex[:8])
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def record_result_indexes(run_id, tag, result):
    combinations = result['combinations']
    for combination in combinations:
        row = {
            'run_id': run_id,
            'tag': tag,
            'model': result['model'],
            'dataset': result['dataset'],
            'selection_protocol': 'validation_metric',
            'valid_metric': result['valid_metric'],
            'combination_index': combination['index'],
            'hyperparameters_json': compact(combination['hyperparameters']),
            'best_valid_score': combination['best_valid_score'],
            'best_valid_epoch': combination['best_valid_epoch'],
            'valid_json': compact(combination['valid']),
            'test_upon_valid_json': compact(combination['test_upon_valid']),
        }
        row.update(metric_fields('valid_', combination['valid']))
        row.update(metric_fields('test_', combination['test_upon_valid']))
        append_csv(REPO_ROOT / 'experiments' / 'runs.csv', row)

    selected = combinations[result['best_by_validation_index']]
    summary = {
        'run_id': run_id,
        'tag': tag,
        'model': result['model'],
        'dataset': result['dataset'],
        'selection_protocol': 'validation_metric',
        'historical_training_protocol': 'test_evaluated_each_validation_step',
        'legacy_console_hyperparameter_selection': 'test_metric',
        'valid_metric': result['valid_metric'],
        'selected_combination_index': selected['index'],
        'hyperparameters_json': compact(selected['hyperparameters']),
        'best_valid_score': selected['best_valid_score'],
        'best_valid_epoch': selected['best_valid_epoch'],
        'valid_json': compact(selected['valid']),
        'test_upon_valid_json': compact(selected['test_upon_valid']),
    }
    summary.update(metric_fields('valid_', selected['valid']))
    summary.update(metric_fields('test_', selected['test_upon_valid']))
    append_csv(REPO_ROOT / 'experiments' / 'results.csv', summary)


def run_experiment(args):
    from utils.quick_start import quick_start

    overrides = load_overrides(args.config, args.set_values)
    runs_root = REPO_ROOT / 'experiments' / 'runs'
    run_id, run_dir = create_run_directory(args.model, args.dataset, args.tag, runs_root)
    snapshot_git(run_dir)
    metadata = {
        'schema_version': 1,
        'run_id': run_id,
        'tag': args.tag,
        'model': args.model,
        'dataset': args.dataset,
        'started_at': dt.datetime.now().astimezone().isoformat(),
        'hostname': platform.node(),
        'python': sys.version,
        'overrides': overrides,
        'phase': 1,
        'evaluation_note': 'MMRec training and epoch evaluation are unchanged; records select hyperparameters by validation metric.',
    }
    write_json(run_dir / 'metadata.json', metadata)
    root_logger = logging.getLogger()

    # Preserve any handlers installed before this experiment.
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    try:
        with (run_dir / "full.log").open("w", encoding="utf-8") as log_file:

            tee_stdout = TeeStream(sys.stdout, log_file)
            tee_stderr = TeeStream(sys.stderr, log_file)

            with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
                # Force MMRec to initialize its own handlers
                # for this experiment.
                for handler in root_logger.handlers[:]:
                    root_logger.removeHandler(handler)

                try:
                    with working_directory(REPO_ROOT / "src"):
                        result = quick_start(
                            args.model,
                            args.dataset,
                            overrides,
                            save_model=not args.no_save_model,
                            mg=args.mg,
                        )

                finally:
                    # Flush and close experiment-specific handlers.
                    for handler in root_logger.handlers[:]:
                        root_logger.removeHandler(handler)
                        handler.flush()
                        handler.close()

    except Exception as error:
        metadata["finished_at"] = (
            dt.datetime.now().astimezone().isoformat()
        )
        metadata["status"] = "failed"
        metadata["error"] = (
            f"{type(error).__name__}: {error}"
        )
        write_json(run_dir / "metadata.json", metadata)
        raise

    finally:
        # Restore the previous logging environment.
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        for handler in original_handlers:
            root_logger.addHandler(handler)

        root_logger.setLevel(original_level)
    try:
        write_json(
            run_dir / "resolved_config.json",
            result["resolved_config"],
        )

        write_json(
            run_dir / "results.json",
            result,
        )

        combination_dir = run_dir / "combinations"
        combination_dir.mkdir()

        for combination in result["combinations"]:
            write_json(
                combination_dir / "{:03d}.json".format(
                    combination["index"]
                ),
                combination,
            )

        record_result_indexes(run_id, args.tag, result)

    except Exception as error:
        metadata["finished_at"] = (
            dt.datetime.now().astimezone().isoformat()
        )
        metadata["status"] = "failed"
        metadata["error"] = (
            f"{type(error).__name__}: {error}"
        )

        write_json(
            run_dir / "metadata.json",
            metadata,
        )
        raise

    metadata["finished_at"] = (
        dt.datetime.now().astimezone().isoformat()
    )
    metadata["status"] = "completed"

    write_json(
        run_dir / "metadata.json",
        metadata,
    )
    return run_id, run_dir, result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', '-m', required=True)
    parser.add_argument('--dataset', '-d', required=True)
    parser.add_argument('--tag', default='default', help='Short label used only for record organization.')
    parser.add_argument('--config', help='Optional YAML mapping of configuration overrides.')
    parser.add_argument('--set', dest='set_values', action='append', default=[], metavar='KEY=VALUE',
                        help='YAML-parsed override; may be supplied more than once.')
    parser.add_argument('--no-save-model', action='store_true', help='Pass save_model=False to MMRec.')
    parser.add_argument('--mg', action='store_true', help='Enable the existing MMRec mg configuration path.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    parsed_args = parse_args()
    try:
        run_id, run_dir, _ = run_experiment(parsed_args)
    except Exception as exc:
        print('Experiment failed: {}'.format(exc), file=sys.stderr)
        raise
    print('Experiment recorded: {} ({})'.format(run_id, run_dir))
