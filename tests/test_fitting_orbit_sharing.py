"""Two jobs on one date must split orbits instead of repeating each other.

Heavy stages are monkeypatched; only the lock / defer / second-pass logic of
spectral.fitting.run_simulation and _ensure_fp_tau is exercised.
"""
import os
import sys
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from spectral import fitting, orbit_lock  # noqa: E402


def _fake_sat0(result_dir, orbits):
    date = datetime.datetime(2020, 7, 3)
    sat0 = {"date": date, "data_dir": "x", "result_dir": str(result_dir),
            "orbit_list": list(orbits), "oco_lite": "lite.nc4"}
    for o in orbits:
        sat0[o] = {"fp_tau_file": fitting._fp_tau_path(str(result_dir), date, o)}
    return sat0


def _patch(monkeypatch, sat0, fitted, computed, current_tau):
    monkeypatch.setattr(fitting, "validate_cloud_distance_file", lambda *a, **k: None)
    monkeypatch.setattr(fitting, "preprocess", lambda *a, **k: sat0)
    monkeypatch.setattr(fitting, "load_shared_data", lambda s: {})
    monkeypatch.setattr(fitting, "fp_tau_file_is_current", lambda p: p in current_tau)

    def compute(s, orbit_id, fp_tau_file):
        computed.append(orbit_id)
        current_tau.add(fp_tau_file)
    monkeypatch.setattr(fitting, "_compute_fp_tau", compute)

    def process(s, orbit_id, shared, **k):
        # the fit must run while this job holds the orbit's output lock
        out = f"{s['result_dir']}/fitting_details/fitting_details_2020-07-03_{orbit_id}.h5"
        assert orbit_lock.is_locked(out)
        fitted.append(orbit_id)
    monkeypatch.setattr(fitting, "process_orbit", process)


def test_locked_orbit_is_deferred_then_fitted_when_lock_clears(tmp_path, monkeypatch):
    orbits = ["o1", "o2", "o3"]
    sat0 = _fake_sat0(tmp_path, orbits)
    fitted, computed, current = [], [], set()
    _patch(monkeypatch, sat0, fitted, computed, current)

    # Another live job holds o2's fit during the first pass and finishes
    # (releases the lock) shortly afterwards.
    import threading
    out_o2 = f"{tmp_path}/fitting_details/fitting_details_2020-07-03_o2.h5"
    assert orbit_lock.try_acquire(out_o2)
    threading.Timer(1.0, lambda: orbit_lock.release(out_o2)).start()

    fitting.run_simulation(sat0["date"], "x", str(tmp_path),
                           lock_stale_hours=4.0, tau_wait_minutes=0.1)

    assert computed == orbits                  # tau computed once per orbit
    assert fitted == ["o1", "o3", "o2"]        # o2 deferred to the second pass
    for o in orbits:                           # all locks released afterwards
        assert not os.path.exists(orbit_lock.lock_path(sat0[o]["fp_tau_file"]))
    assert not os.path.exists(orbit_lock.lock_path(out_o2))


def test_stale_lock_is_taken_over_in_first_pass(tmp_path, monkeypatch):
    orbits = ["o1", "o2", "o3"]
    sat0 = _fake_sat0(tmp_path, orbits)
    fitted, computed, current = [], [], set()
    _patch(monkeypatch, sat0, fitted, computed, current)
    out_o2 = f"{tmp_path}/fitting_details/fitting_details_2020-07-03_o2.h5"
    assert orbit_lock.try_acquire(out_o2)
    old = os.path.getmtime(orbit_lock.lock_path(out_o2)) - 10 * 3600
    os.utime(orbit_lock.lock_path(out_o2), (old, old))   # left by a dead job

    fitting.run_simulation(sat0["date"], "x", str(tmp_path),
                           lock_stale_hours=1.0, tau_wait_minutes=0.01)

    assert fitted == ["o1", "o2", "o3"]        # taken over in place
    assert not os.path.exists(orbit_lock.lock_path(out_o2))


def test_live_lock_held_by_other_job_is_left_alone(tmp_path, monkeypatch):
    orbits = ["o1", "o2"]
    sat0 = _fake_sat0(tmp_path, orbits)
    fitted, computed, current = [], [], set()
    _patch(monkeypatch, sat0, fitted, computed, current)
    out_o1 = f"{tmp_path}/fitting_details/fitting_details_2020-07-03_o1.h5"
    assert orbit_lock.try_acquire(out_o1)      # fresh lock: another live job

    fitting.run_simulation(sat0["date"], "x", str(tmp_path),
                           lock_stale_hours=4.0, tau_wait_minutes=0.02)

    assert fitted == ["o2"]                    # o1 never fitted here
    assert orbit_lock.is_locked(out_o1)        # and its lock untouched
    orbit_lock.release(out_o1)


def test_ensure_fp_tau_waits_for_other_job(tmp_path, monkeypatch):
    sat0 = _fake_sat0(tmp_path, ["o1"])
    current, computed = set(), []
    monkeypatch.setattr(fitting, "fp_tau_file_is_current", lambda p: p in current)
    monkeypatch.setattr(fitting, "_compute_fp_tau",
                        lambda s, o, f: computed.append(o) or current.add(f))
    tau = sat0["o1"]["fp_tau_file"]
    assert orbit_lock.try_acquire(tau)         # other job computing it

    # Simulate the other job finishing while we wait: release lock + mark current.
    import threading
    def finish():
        current.add(tau); orbit_lock.release(tau)
    threading.Timer(1.0, finish).start()
    real_sleep = orbit_lock.time.sleep
    monkeypatch.setattr(orbit_lock.time, "sleep", lambda s: real_sleep(min(s, 0.5)))
    assert fitting._ensure_fp_tau(sat0, "o1", 3600, 30) is True
    assert computed == []                      # we did not recompute it


def test_ensure_fp_tau_computes_when_free(tmp_path, monkeypatch):
    sat0 = _fake_sat0(tmp_path, ["o1"])
    current, computed = set(), []
    monkeypatch.setattr(fitting, "fp_tau_file_is_current", lambda p: p in current)
    monkeypatch.setattr(fitting, "_compute_fp_tau",
                        lambda s, o, f: computed.append(o) or current.add(f))
    assert fitting._ensure_fp_tau(sat0, "o1", 3600, 30) is True
    assert computed == ["o1"]
    assert not orbit_lock.is_locked(sat0["o1"]["fp_tau_file"])


def test_select_orbits_filters_and_validates():
    orbits = ["33111a_GL", "33124a_GL", "33124a_TG", "33125a_GL"]
    assert fitting._select_orbits(orbits) == orbits
    assert fitting._select_orbits(orbits, ["33124a_TG"]) == ["33124a_TG"]
    assert fitting._select_orbits(orbits, ["33125a_GL", "33111a_GL"]) == ["33111a_GL", "33125a_GL"]
    assert fitting._select_orbits(orbits, None, 2) == orbits[:2]
    try:
        fitting._select_orbits(orbits, ["99999a_GL"])
    except FileNotFoundError as e:
        assert "99999a_GL" in str(e)
    else:
        raise AssertionError("missing orbit must raise")
