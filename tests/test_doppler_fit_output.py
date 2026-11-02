"""Doppler-variant fit set (cumulant_fit.FIT_FIELDS) and the fitting_details
file process_orbit writes since 2026-10-31: no smoothing, no production /
_nosg fields, every fitted coefficient with a sig_* error, reference statistics
of the _am / _t2 coefficients, cache keyed on fit_schema + tau_convolution."""
import os
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from abs_util.ils_tau import tau_convolution_version  # noqa: E402
from spectral import fitting  # noqa: E402
from spectral.cumulant_fit import (  # noqa: E402
    EXTRA_FIT_FIELDS, FIT_FIELDS, FIT_SCHEMA, _doppler_fit_chunk, _extra_fit_chunk)

IDX = {n: i for i, n in enumerate(FIT_FIELDS)}
NCH = 1016
X = (np.arange(1, NCH - 1) - 507.5) / 507.5          # slab channels j = 1..1014


def _band_truth(rng):
    """tau [3, n], frac [3, n] and the noiseless ln T of each band's agreed model."""
    n = NCH - 2
    tau = rng.uniform(0.0, 2.5, size=(3, n))
    frac = np.zeros((3, n))
    frac[1:] = rng.uniform(0.0, 0.6, size=(2, n))
    tw, td = frac * tau, tau - frac * tau
    ln = np.empty((3, n))
    # O2A: c + a1 x + ln[(1-f) e^{-k1 τ} + f]
    ln[0] = -0.10 + 0.02 * X + np.log(0.99 * np.exp(-1.00 * tau[0]) + 0.01)
    # WCO2: c + a1 x - k1 τ_dry - k1w τ_h2o
    ln[1] = -0.20 + 0.03 * X - 1.00 * td[1] - 1.20 * tw[1]
    # SCO2 (water): c + a1 x - k1 τ_dry - k1w τ_h2o + ½ k2 τ²
    ln[2] = -0.15 - 0.01 * X - 0.95 * td[2] - 1.10 * tw[2] + 0.5 * 0.05 * tau[2] ** 2
    return tau, frac, ln


def test_fit_fields_have_errors_and_no_legacy_names():
    coefs = [n for n in FIT_FIELDS
             if not n.startswith(("sig_", "rms_", "ok_")) and n != "nchan"]
    for name in coefs:
        assert f"sig_{name}" in FIT_FIELDS, name
    assert not any(n.endswith("_nosg") or "fitting" in n for n in FIT_FIELDS)


def test_doppler_chunk_recovers_agreed_model_and_matches_older_fit():
    rng = np.random.default_rng(1)
    tau, frac, ln = _band_truth(rng)
    m = 4
    tau_s = np.repeat(tau[:, None], m, axis=1)
    fr_s = np.repeat(frac[:, None], m, axis=1)
    ln_s = np.repeat(ln[:, None], m, axis=1) + rng.normal(0, 0.002, size=(3, m, NCH - 2))
    _, fits = _doppler_fit_chunk(np.arange(m), tau_s, ln_s, fr_s)
    assert fits.shape == (3, m, len(FIT_FIELDS))
    k1 = fits[:, :, IDX["k1_am"]]
    assert np.allclose(k1[0], 1.00, atol=0.02)
    assert np.allclose(k1[1], 1.00, atol=0.01) and np.allclose(fits[1, :, IDX["k1w_am"]], 1.20, atol=0.01)
    assert np.allclose(k1[2], 0.95, atol=0.02)
    assert np.allclose(fits[0, :, IDX["f_am"]], 0.01, atol=0.005)
    # exp_int_am and its error follow c_am
    c = fits[:, :, IDX["c_am"]]
    assert np.allclose(fits[:, :, IDX["exp_int_am"]], np.exp(c))
    assert np.allclose(fits[:, :, IDX["sig_exp_int_am"]], np.exp(c) * fits[:, :, IDX["sig_c_am"]])
    # fixed / absent parameters: value 0 or NaN, error NaN
    assert np.all(fits[0, :, IDX["k2_am"]] == 0) and np.all(np.isnan(fits[0, :, IDX["sig_k2_am"]]))
    assert np.all(np.isnan(fits[0, :, IDX["k1w_am"]])) and np.all(np.isnan(fits[1, :, IDX["f_am"]]))
    assert np.all(np.isnan(fits[2, :, IDX["a2_am"]]))      # water: no x² term
    # the shared fields agree with the older _extra_fit_chunk on the same input
    _, ex = _extra_fit_chunk(np.arange(m), tau_s, ln_s, (7, 3, 7), frac_slab=fr_s)
    eidx = {n: i for i, n in enumerate(EXTRA_FIT_FIELDS)}
    for name in ("k1_o2", "sig_k1_o2", "k1_o3", "sig_k1_o3", "sig_k2_o3", "k1_dry_t2",
                 "sig_k1w_t2", "k1_am", "sig_k1_am", "k2diag_am", "rms_am"):
        assert np.allclose(fits[:, :, IDX[name]], ex[:, :, eidx[name]], equal_nan=True), name


def _exact_truth(kind, tau, frac, ln_am):
    """Noiseless ln T [3, n] that the fit `kind` reproduces exactly."""
    tw, td = frac * tau, tau - frac * tau
    if kind == "cum2":
        return -0.2 - 1.0 * tau + 0.5 * 0.05 * tau ** 2
    if kind == "cum3":
        return -0.2 - 1.0 * tau + 0.5 * 0.05 * tau ** 2 - 0.01 / 3 * tau ** 3
    if kind == "t2":
        return -0.2 - 1.0 * td - 1.2 * tw + 0.5 * 0.05 * tau ** 2
    return ln_am


@pytest.mark.parametrize("field,band,kind", [
    ("k1_o2", 1, "cum2"), ("c_o2", 0, "cum2"), ("c_o3", 2, "cum3"), ("k3_o3", 1, "cum3"),
    ("k1w_t2", 1, "t2"), ("c_t2", 2, "t2"), ("k1_dry_t2", 2, "t2"),
    ("k1_am", 1, "am"), ("k1w_am", 1, "am"), ("k2_am", 2, "am"), ("a1_am", 2, "am"),
    ("k1_am", 0, "am"), ("f_am", 0, "am"), ("exp_int_am", 1, "am")])
def test_formal_errors_match_monte_carlo_scatter(field, band, kind):
    """sig_* equals the scatter over noise realisations when the fitted model is
    exact and the noise iid (the setting the formal error assumes; with misfit
    the residual-scaled sig_* is larger than the noise scatter by design)."""
    rng = np.random.default_rng(7)
    tau, frac, ln_am = _band_truth(rng)
    ln = _exact_truth(kind, tau, frac, ln_am)
    m = 300
    noise = 0.01
    tau_s = np.repeat(tau[:, None], m, axis=1)
    fr_s = np.repeat(frac[:, None], m, axis=1)
    ln_s = np.repeat(ln[:, None], m, axis=1) + rng.normal(0, noise, size=(3, m, NCH - 2))
    _, fits = _doppler_fit_chunk(np.arange(m), tau_s, ln_s, fr_s)
    vals = fits[band, :, IDX[field]]
    sig = fits[band, :, IDX[f"sig_{field}"]]
    ok = np.isfinite(vals) & np.isfinite(sig)
    assert ok.sum() > 0.95 * m
    ratio = np.std(vals[ok]) / np.median(sig[ok])
    assert 0.85 < ratio < 1.15, (field, band, ratio)


# ─── process_orbit end to end on a synthetic orbit ──────────────────────────

class _Lite(dict):
    def __init__(self, n, rng):
        super().__init__()
        self.n, self.rng = n, rng
        self["xco2_corr"] = 410.0 + rng.normal(0, 0.3, n)
        self["xco2_raw"] = 411.0 + rng.normal(0, 0.3, n)
        self["vertex_longitude"] = np.tile([0.0, 0.01, 0.01, 0.0], (n, 1))
        self["vertex_latitude"] = np.tile([0.0, 0.0, 0.02, 0.02], (n, 1))
        for b in ("o2a", "wco2", "sco2"):
            self[f"albedo_{b}"] = rng.uniform(0.1, 0.3, n)

    def __missing__(self, key):                # every other Lite variable
        return self.rng.uniform(0.0, 1.0, self.n)

    def __contains__(self, key):               # AK arrays absent -> skipped
        return dict.__contains__(self, key)


def _synthetic_orbit(n=60, seed=3):
    rng = np.random.default_rng(seed)
    tau1, frac1, ln1 = _band_truth(rng)
    pad = lambda a: np.concatenate([a[:, :1], a, a[:, -1:]], axis=1)   # back to 1016 channels
    tau = np.repeat(pad(tau1)[:, None], n, axis=1)
    frac = np.repeat(pad(frac1)[:, None], n, axis=1)
    ln = np.repeat(pad(ln1)[:, None], n, axis=1) + rng.normal(0, 0.003, size=(3, n, NCH))
    sid = np.arange(n, dtype=np.int64) + 2020070300000000
    od = {
        "sounding_id": sid, "fp_number": np.arange(n) % 8,
        "lon": np.zeros(n), "lat": np.linspace(10.0, 10.4, n),
        "sza": np.full(n, 30.0), "vza": np.full(n, 5.0), "wvl": np.ones((3, 8, NCH)),
        "radiances": np.exp(ln) / np.pi, "toa_sol": np.ones((3, n, NCH)), "tau": tau,
        "valid_l1b": np.ones(n, bool), "frac_h2o": frac, "land": np.arange(n) % 2 == 0,
        "tau_convolution": tau_convolution_version(True),
    }
    lite = _Lite(n, rng)
    shared = {"lite": lite, "lite_index": {int(s): i for i, s in enumerate(sid)},
              "cld_dist_index": {int(s): 50.0 for s in sid}, "weighted_cld_dist_index": {}}
    prof = {k: np.zeros((n, 72)) for k in ("t_profile", "q_profile", "co2prior_profile", "p_profile")}
    prof.update({k: np.zeros(n) for k in ("psurf_met", "tropopause_pressure", "tropopause_temp")})
    return od, shared, prof


def _run(tmp_path, monkeypatch, od, shared, prof, **kw):
    from datetime import datetime
    monkeypatch.setattr(fitting, "load_orbit_data", lambda s, o: od)
    monkeypatch.setattr(fitting, "load_profile_data", lambda *a: prof)
    sat = {"date": datetime(2020, 7, 3), "result_dir": str(tmp_path),
           "orb1": {"oco_met": None, "oco_co2prior": None}}
    fitting.process_orbit(sat, "orb1", shared, **kw)
    return f"{tmp_path}/{fitting.FIT_SUBDIR}/fitting_details_2020-07-03_orb1.h5"


def test_process_orbit_writes_new_schema(tmp_path, monkeypatch):
    od, shared, prof = _synthetic_orbit()
    out = _run(tmp_path, monkeypatch, od, shared, prof)
    with h5py.File(out, "r") as f:
        assert f.attrs["fit_schema"] == FIT_SCHEMA
        assert f.attrs["tau_convolution"] == tau_convolution_version(True)
        keys = set(f.keys())
        for band in ("o2a", "wco2", "sco2"):
            for name in FIT_FIELDS:
                assert f"{band}_{name}" in keys
        assert not [k for k in keys if k.endswith("_nosg") or "_fitting" in k
                    or k.endswith("_kappa") or k.startswith("exp_intercept")]
        for prefix in ("ref", "r15", "r05"):
            assert f"{prefix}_n" in keys
            for band, names in fitting.REF_FIT_VARS.items():
                for name in names:
                    assert f"{prefix}_{band}_{name}_mean" in keys and f"{prefix}_{band}_{name}_std" in keys
            for band, name in fitting.REF_FIT_VARS_SOLO:
                assert f"{prefix}_{band}_{name}_mean" in keys
            assert f"{prefix}_alb_o2a_mean" in keys
        assert "xco2_bc_anomaly" in keys and "xco2_raw_anomaly_r15" in keys
        assert np.isfinite(f["xco2_bc_anomaly"][:]).mean() > 0.9
        assert np.isfinite(f["ref_wco2_k1_am_mean"][:]).mean() > 0.9
        assert np.allclose(np.nanmedian(f["ref_wco2_k1_am_mean"][:]), 1.0, atol=0.01)
        # SCO2 a2_am: land only, so its reference stats come from land soundings
        assert np.isfinite(f["ref_sco2_a2_am_mean"][:]).mean() > 0.9
        assert np.isfinite(f["sco2_a2_am"][:])[od["land"]].all()
        assert np.isnan(f["sco2_a2_am"][:])[~od["land"]].all()
    pngs = sorted(os.listdir(f"{tmp_path}/2020-07-03/orb1"))
    assert [p.split("_fit_example_")[0] for p in pngs] == ["o2a", "sco2", "wco2"]   # one per band


def test_cache_reused_only_for_matching_schema(tmp_path, monkeypatch):
    od, shared, prof = _synthetic_orbit(n=20)
    out = _run(tmp_path, monkeypatch, od, shared, prof)
    t0 = os.path.getmtime(out)
    os.utime(out, (t0 - 100, t0 - 100))
    _run(tmp_path, monkeypatch, od, shared, prof)
    assert os.path.getmtime(out) == t0 - 100           # current: skipped
    with h5py.File(out, "a") as f:
        f.attrs["fit_schema"] = "older"
    _run(tmp_path, monkeypatch, od, shared, prof)
    assert os.path.getmtime(out) > t0 - 100            # stale schema: refitted
    with h5py.File(out, "r") as f:
        assert f.attrs["fit_schema"] == FIT_SCHEMA


def test_non_doppler_tau_is_rejected(tmp_path, monkeypatch):
    od, shared, prof = _synthetic_orbit(n=10)
    od["tau_convolution"] = tau_convolution_version(False)
    with pytest.raises(RuntimeError, match="telluric-Doppler"):
        _run(tmp_path, monkeypatch, od, shared, prof)
    od["tau_convolution"] = tau_convolution_version(True)
    od["frac_h2o"] = None
    with pytest.raises(RuntimeError, match="frac_h2o missing"):
        _run(tmp_path, monkeypatch, od, shared, prof)
