# MMRec experiment records (phase one)

Run experiments through the wrapper from the repository root:

```bash
python experiments/run.py --model MGCN --dataset sports --tag baseline
```

Use a YAML override file or repeat `--set` for values that should override the
normal MMRec configuration.  Values are parsed as YAML, so lists are supported.

```bash
python experiments/run.py --model MGCN --dataset sports --tag cl-search \
  --set 'seed=[999]' --set 'cl_loss=[0.001, 0.01, 0.1]'
```

Every invocation creates `experiments/runs/<run_id>/` containing the effective
base configuration, a per-combination resolved configuration and metrics
snapshot, Git revision/status/diff, eligible untracked source, metadata, and
`full.log`. This directory is intentionally ignored by Git: it may contain
large logs and local environment details. Store it in persistent artifact
storage if the run must be retained.

`experiments/runs.csv` receives one row per hyperparameter combination.
`experiments/results.csv` is only appended to: existing historical rows are
never rewritten. Its new rows are selected with the validation metric, while
the records explicitly label MMRec's unchanged legacy behaviour (test is still
evaluated at each validation step, and its console "BEST" uses a test metric).

Phase one does not change model code, trainer stopping, checkpoint behaviour,
or evaluation timing. Do not put credentials in an override file; likely
credential-shaped keys are redacted in JSON snapshots, but command-line and
environment secrets can still be exposed elsewhere.
