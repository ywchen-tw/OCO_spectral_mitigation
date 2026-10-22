"""Unit tests for spectral.orbit_lock (per-orbit advisory lock files)."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectral import orbit_lock  # noqa: E402


def test_acquire_is_exclusive_and_released(tmp_path):
    target = str(tmp_path / "orbit" / "fp_tau_combined.h5")
    assert orbit_lock.try_acquire(target)
    assert orbit_lock.is_locked(target)
    assert not orbit_lock.try_acquire(target)          # second claimant loses
    orbit_lock.release(target)
    assert not orbit_lock.is_locked(target)
    assert orbit_lock.try_acquire(target)              # free again
    orbit_lock.release(target)
    orbit_lock.release(target)                         # idempotent


def test_held_releases_only_when_acquired(tmp_path):
    target = str(tmp_path / "fitting_details_x.h5")
    with orbit_lock.held(target) as got:
        assert got
        with orbit_lock.held(target) as got_inner:
            assert not got_inner
        assert orbit_lock.is_locked(target)            # inner exit must not release
    assert not orbit_lock.is_locked(target)


def test_held_releases_on_exception(tmp_path):
    target = str(tmp_path / "f.h5")
    try:
        with orbit_lock.held(target) as got:
            assert got
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not orbit_lock.is_locked(target)


def test_stale_lock_is_taken_over(tmp_path):
    target = str(tmp_path / "f.h5")
    assert orbit_lock.try_acquire(target)
    old = time.time() - 10 * 3600
    os.utime(orbit_lock.lock_path(target), (old, old))
    assert not orbit_lock.is_locked(target, stale_seconds=3600)
    assert orbit_lock.try_acquire(target, stale_seconds=3600)
    assert orbit_lock.is_locked(target, stale_seconds=3600)
    orbit_lock.release(target)


def test_wait_for_returns_when_ready_or_unlocked(tmp_path):
    target = str(tmp_path / "f.h5")
    # Unlocked and not ready: returns immediately with False.
    t0 = time.time()
    assert orbit_lock.wait_for(target, lambda: False, timeout_seconds=30, poll_seconds=1) is False
    assert time.time() - t0 < 5
    # Locked and never ready: respects the timeout.
    assert orbit_lock.try_acquire(target)
    t0 = time.time()
    assert orbit_lock.wait_for(target, lambda: False, timeout_seconds=2, poll_seconds=1) is False
    assert 1.5 <= time.time() - t0 < 10
    # Locked but ready: returns True immediately.
    assert orbit_lock.wait_for(target, lambda: True, timeout_seconds=30, poll_seconds=1) is True
    orbit_lock.release(target)
