"""Advisory lock files so several jobs can share one date's orbits.

Several Slurm jobs may run ``fitting.py`` on the same date.  Before a job
starts an expensive per-orbit step (optical-depth precompute, spectral fit)
it claims the step by creating ``<output>.lock`` with ``O_CREAT | O_EXCL``,
which is atomic on POSIX file systems including Lustre.  A job that finds the
lock already present skips the orbit; the owner removes the lock when the
output has been written.  A lock older than ``stale_seconds`` is assumed to
belong to a dead job and is taken over.
"""

import logging
import os
import socket
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


def lock_age_seconds(target):
    """Age of the lock on ``target`` in seconds, or None when unlocked."""
    try:
        return time.time() - os.path.getmtime(lock_path(target))
    except OSError:
        return None


def is_locked(target, stale_seconds=DEFAULT_STALE_SECONDS):
    """True when a live (non-stale) lock exists for ``target``."""
    age = lock_age_seconds(target)
    return age is not None and age < stale_seconds


def try_acquire(target, stale_seconds=DEFAULT_STALE_SECONDS):
    """Claim ``target``; return True when this process now owns the lock."""
    path = lock_path(target)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    for _attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            age = lock_age_seconds(target)
            if age is not None and age >= stale_seconds:
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
