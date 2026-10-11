#!/usr/bin/env bash
# Explicit AutoDL-only, detached sweep launcher. No shutdown in --check-only.
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--check-only" ]]; then
    shift
    printf 'Preview only: detached AutoDL sweep; sync, optional backup, /usr/bin/shutdown on exit.\n'
    printf 'Sweep arguments: '; printf '%q ' "$@"; printf '\n'
    exit 0
fi
if [[ "${1:-}" != "--autodl" && "${1:-}" != "--worker" ]]; then
    echo 'Usage: bash experiments/run_autodl.sh --autodl --model VBPR --dataset sports --sweep-id ID [sweep options]' >&2
    exit 2
fi
mode="$1"
shift
if [[ "$(uname -s)" != Linux || "$EUID" != 0 || ! -d /root/autodl-tmp || ! -x /usr/bin/shutdown ]]; then
    echo 'Refusing execution: requires Linux root, /root/autodl-tmp and AutoDL /usr/bin/shutdown.' >&2
    exit 2
fi
for dependency in setsid nohup flock sync tar; do
    command -v "$dependency" >/dev/null || { echo "Missing command: $dependency" >&2; exit 2; }
done
python_bin="${AUTODL_PYTHON:-$(command -v python)}"
[[ -x "$python_bin" ]] || { echo 'AUTODL_PYTHON must name an executable interpreter.' >&2; exit 2; }
cd "$repo_root"
mkdir -p experiments/sweeps

if [[ "$mode" == --autodl ]]; then
    launch_dir="$(mktemp -d "$repo_root/experiments/sweeps/autodl-launch-XXXXXXXX")"
    nohup setsid bash "$repo_root/experiments/run_autodl.sh" --worker "$launch_dir" "$@" \
        >"$launch_dir/launcher.log" 2>&1 < /dev/null &
    echo "Detached launcher PID: $!"
    echo "Log: $launch_dir/launcher.log"
    echo "Exit status: $launch_dir/exit_status.txt (written before shutdown)"
    exit 0
fi

launch_dir="$1"
shift
# A second launcher must never shut down an instance used by the first launcher.
exec 9>experiments/sweeps/.autodl.lock
flock -n 9 || { echo 'Another AutoDL launcher is active; refusing.' >&2; exit 2; }
child_pid=''
finish() {
    training_code=$?
    trap - EXIT INT TERM
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" || true
    fi
    printf 'training_exit_code=%s\n' "$training_code" >"$launch_dir/exit_status.txt"
    if [[ -n "${AUTODL_BACKUP_DIR:-}" ]]; then
        # Local/mounted-storage backup; no Git operations or automatic network transfer.
        if mkdir -p "$AUTODL_BACKUP_DIR"; then
            backup_file="$(mktemp "$AUTODL_BACKUP_DIR/mmrec-XXXXXXXX.tar.gz")"
            tar_args=(experiments/runs experiments/sweeps)
            [[ ! -f experiments/runs.csv ]] || tar_args+=(experiments/runs.csv)
            [[ ! -f experiments/results.csv ]] || tar_args+=(experiments/results.csv)
            tar -czf "$backup_file" "${tar_args[@]}" && echo "Backup: $backup_file" || echo 'Backup failed' >&2
        else
            echo 'Backup directory unavailable' >&2
        fi
    fi
    sync || echo 'sync failed' >&2
    echo "Training exited $training_code; requesting AutoDL shutdown."
    if /usr/bin/shutdown; then
        printf 'shutdown_exit_code=0\n' >>"$launch_dir/exit_status.txt"
    else
        shutdown_code=$?
        printf 'shutdown_exit_code=%s\n' "$shutdown_code" >>"$launch_dir/exit_status.txt"
        echo 'Shutdown failed: use the AutoDL console immediately.' >&2
    fi
    sync || true
    exit "$training_code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set +e
"$python_bin" -u experiments/sweep.py "$@" &
child_pid=$!
wait "$child_pid"
training_code=$?
child_pid=''
exit "$training_code"
