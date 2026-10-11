"""Shared persistence and configuration checks; never imports training code."""
import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name, suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def portable_config(config: dict) -> dict:
    # Device is derived from host CUDA availability; use_gpu/gpu_id remain checked.
    return {key: value for key, value in config.items() if key != 'device'}


def resolve_config(model: str, dataset: str, overrides: dict, mg: bool = False) -> dict:
    from utils.configurator import Config
    from experiments.run import json_safe, working_directory
    with working_directory(ROOT / 'src'):
        return portable_config(json_safe(Config(model, dataset, dict(overrides), mg).final_config_dict))


def expand(config: dict) -> list:
    names = config['hyper_parameters']
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError('hyper_parameters must be a list of names')
    if len(names) != len(set(names)) or 'seed' not in names:
        raise ValueError('hyper_parameters must be unique and include seed')
    axes = []
    for name in names:
        values = config.get(name) or [None]
        if not isinstance(values, list):
            raise ValueError('{} must be a list (use --set {}=[VALUE])'.format(name, name))
        axes.append(values)
    combinations = [dict(zip(names, values)) for values in product(*axes)]
    if len({digest(item) for item in combinations}) != len(combinations):
        raise ValueError('Duplicate hyperparameter combinations would repeat training')
    return combinations


def code_identity() -> dict:
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    # Hash actual source contents, including dirty/untracked source. Generated artifacts
    # are excluded so completing a combination cannot change the code identity.
    paths = list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'src').rglob('*.yaml'))
    paths += list((ROOT / 'src').rglob('*.yml'))
    paths += list((ROOT / 'experiments').glob('*.py')) + list((ROOT / 'experiments').glob('*.sh'))
    paths += [ROOT / 'requirements.txt']
    source = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(set(paths)) if path.is_file()}
    return {'git_commit': commit, 'source_fingerprint': digest(source)}


@contextmanager
def exclusive_lock(path: Path):
    """Kernel releases the lock after a crash; no stale PID file recovery needed."""
    import fcntl
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another experiment scheduler is active') from error
        try:
            yield handle.fileno()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
