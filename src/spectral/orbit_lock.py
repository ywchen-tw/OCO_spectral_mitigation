"""Advisory lock files so several jobs can share one date's orbits.

Several Slurm jobs may run ``fitting.py`` on the same date.  Before a job
starts an expensive per-orbit step (optical-depth precompute, spectral fit)
it claims the step by creating ``<output>.lock`` with ``O_CREAT | O_EXCL``,
which is atomic on POSIX file systems including Lustre.  A job that finds the
lock already present skips the orbit; the owner removes the lock when the
output has been written.  A lock older than ``stale_seconds`` is assumed to
belong to a dead job and is taken over.

A lock is also taken over at once when its owner is provably gone
(``owner_dead``, 2026-11-02): same host and the pid no longer exists, or the
owner Slurm job is finished / pending / running on another node -- the usual
result of a preempted-and-requeued task, which keeps its job id but restarts
elsewhere and would otherwise block the orbit (and itself) for 4 h.  When the
owner cannot be checked (no Slurm id, squeue error) the lock is treated as live.
"""

import logging
import os
import socket
import subprocess
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

LOCK_SUFFIX = ".lock"
DEFAULT_STALE_SECONDS = 4 * 3600


def lock_path(target):
    """Path of the lock file guarding ``target``."""
    return f"{target}{LOCK_SUFFIX}"


def _owner_tag():
    return (
        f"host={socket.gethostname()} pid={os.getpid()} "
        f"slurm={os.environ.get('SLURM_JOB_ID', '-')} "
        f"since={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
    )


_DEAD_CACHE = {}          # (lock path, lock text) -> (checked_at, dead)
_DEAD_CACHE_SECONDS = 300


def _short(host):
    return (host or "").split(".")[0]


def _parse_owner(text):
    out = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def _slurm_state(job_id):
    """(state, node, start) of a Slurm job; ('GONE', '', '') when squeue says the id is
    unknown (finished); None when squeue cannot be asked."""
    try:
        r = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T %N %S"],
                           capture_output=True, text=True, timeout=30)
    except Exception:            # noqa: BLE001 -- no squeue / timeout: cannot tell
        return None
    if r.returncode != 0:
        return ("GONE", "", "") if "Invalid job id" in (r.stderr or "") else None
    line = (r.stdout or "").strip().splitlines()
    if not line:
        return ("GONE", "", "")
    parts = line[0].split() + ["", ""]
    return (parts[0], parts[1], parts[2])


def owner_dead(target):
    """True when the process that holds the lock on ``target`` is provably gone."""
    path = lock_path(target)
    try:
        with open(path) as handle:
            text = handle.read()
    except OSError:
        return False
    key = (path, text)
    hit = _DEAD_CACHE.get(key)
    if hit and time.time() - hit[0] < _DEAD_CACHE_SECONDS:
        return hit[1]
    own = _parse_owner(text)
    host, pid, job = _short(own.get("host")), own.get("pid"), own.get("slurm", "-")
    dead = False
    if host and host == _short(socket.gethostname()) and pid and pid.isdigit():
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            dead = True
        except PermissionError:
            pass
    if not dead and job not in ("", "-") and job.isdigit():
        st = _slurm_state(job)
        if st is not None:
            state, node, start = st
            if state == "GONE" or state in ("PENDING", "COMPLETING", "COMPLETED", "FAILED",
                                            "CANCELLED", "PREEMPTED", "TIMEOUT", "NODE_FAIL",
                                            "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE"):
                dead = True
            elif state == "RUNNING" and host and _short(node) and _short(node) != host:
                dead = True      # requeued onto another node: the lock's process is gone
            elif state == "RUNNING" and own.get("since") and len(start) == 19 and own["since"] < start:
                dead = True      # lock taken before the job's current run started (requeued)
    _DEAD_CACHE[key] = (time.time(), dead)
    if dead:
        logger.warning("Lock %s belongs to a dead owner (%s)", path, text.strip())
    return dead


def lock_age_seconds(target):
    """Age of the lock on ``target`` in seconds, or None when unlocked."""
    try:
        return time.time() - os.path.getmtime(lock_path(target))
    except OSError:
        return None


def is_locked(target, stale_seconds=DEFAULT_STALE_SECONDS):
    """True when a live (non-stale) lock exists for ``target``."""
    age = lock_age_seconds(target)
    return age is not None and age < stale_seconds and not owner_dead(target)


def try_acquire(target, stale_seconds=DEFAULT_STALE_SECONDS):
    """Claim ``target``; return True when this process now owns the lock."""
    path = lock_path(target)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    for _attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            age = lock_age_seconds(target)
            if age is not None and (age >= stale_seconds or owner_dead(target)):
                logger.warning(
                    "Taking over stale lock %s (age %.1f h)", path, age / 3600.0
                )
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                continue
            return False
        with os.fdopen(fd, "w") as handle:
            handle.write(_owner_tag())
        return True
    return False


def release(target):
    """Remove the lock on ``target`` (no error when already gone)."""
    try:
        os.remove(lock_path(target))
    except FileNotFoundError:
        pass


@contextmanager
def held(target, stale_seconds=DEFAULT_STALE_SECONDS):
    """Context manager yielding True when the lock was acquired.

    The lock is released on exit only when this process acquired it, so a
    job that merely observed another job's lock never removes it.
    """
    acquired = try_acquire(target, stale_seconds)
    try:
        yield acquired
    finally:
        if acquired:
            release(target)


def wait_for(target, is_ready, timeout_seconds, poll_seconds=30,
             stale_seconds=DEFAULT_STALE_SECONDS):
    """Wait while another job holds ``target``.

    Returns as soon as ``is_ready()`` is True, or when no live lock remains
    (the caller then decides whether to do the work itself), or after
    ``timeout_seconds``.  The return value is ``is_ready()`` at that moment.
    """
    deadline = time.time() + timeout_seconds
    while True:
        if is_ready():
            return True
        if not is_locked(target, stale_seconds):
            return is_ready()
        if time.time() >= deadline:
            return is_ready()
        time.sleep(min(poll_seconds, max(1.0, deadline - time.time())))
