"""Unit tests for the telluric-Doppler test variant of the tau / fitting pipeline.

Covers the pure pieces that need no ABSCO tables or orbit files:
tau version stamping, the per-channel tau attribution, and the extra fit
statistics worker.
"""

import os
import sys

import h5py
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from abs_util import fp_atm  # noqa: E402
from abs_util.ils_tau import (  # noqa: E402
    TAU_CONVOLUTION_VERSION, TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER,
    attribute_optical_depth, convolve_optical_depth_in_transmission, tau_convolution_version,
)
from spectral.cumulant_fit import (  # noqa: E402
    EXTRA_FIT_FIELDS, _extra_fit_chunk, get_design_matrix,
)


def test_version_stamp_distinguishes_variants():
    assert tau_convolution_version(False) == TAU_CONVOLUTION_VERSION
    assert tau_convolution_version(True) == TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER
    assert TAU_CONVOLUTION_VERSION != TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER


def _synthetic_window(n_hr=400, n_lay=10, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-3, 3, n_hr)
    ils = np.exp(-0.5 * x**2)
    p_lay = np.linspace(950.0, 50.0, n_lay)          # hPa, surface first
    dz = np.full(n_lay, 1.0)
    line = np.exp(-0.5 * (x / 0.4) ** 2)              # one absorption line
    profile_dry = np.linspace(1.0, 0.2, n_lay)         # absorption weighted to low layers
    profile_h2o = np.linspace(0.3, 0.0, n_lay)         # water only near the surface
    ext_dry = 0.5 * line[:, None] * profile_dry[None, :]
    ext_h2o = 0.2 * line[:, None] * profile_h2o[None, :] + 1e-6 * rng.random((n_hr, n_lay))
    ray = np.full(n_hr, 0.02)
    return ils, p_lay, dz, ext_dry, ext_h2o, ray


def test_attribution_fractions_are_consistent():
    ils, p_lay, dz, ext_dry, ext_h2o, ray = _synthetic_window()
    airmass = 2.3
    frac_h2o, frac_ray, p_abs = attribute_optical_depth(
        ext_dry + ext_h2o, ext_h2o, dz, ils, airmass, ray, p_lay)
    assert 0.0 < frac_h2o < 1.0
    assert 0.0 < frac_ray < 1.0
    assert frac_h2o + frac_ray < 1.0
    assert p_lay.min() < p_abs < p_lay.max()
    # water sits near the surface, so the H2O-only centroid is deeper than the total
    _, _, p_abs_h2o = attribute_optical_depth(ext_h2o, ext_h2o, dz, ils, airmass, 0 * ray, p_lay)
    assert p_abs_h2o > p_abs


def test_attribution_limits():
    ils, p_lay, dz, ext_dry, ext_h2o, ray = _synthetic_window()
    airmass = 1.5
    # no water, no Rayleigh -> both fractions zero
    f_h2o, f_ray, _ = attribute_optical_depth(ext_dry, 0 * ext_h2o, dz, ils, airmass, 0 * ray, p_lay)
    assert abs(f_h2o) < 1e-12 and abs(f_ray) < 1e-12
    # Rayleigh only -> frac_ray = 1, pressure centroid undefined
    f_h2o, f_ray, p_abs = attribute_optical_depth(0 * ext_dry, 0 * ext_h2o, dz, ils, airmass, ray, p_lay)
    assert abs(f_ray - 1.0) < 1e-12 and np.isnan(p_abs)
    # no absorption at all -> NaN
    out = attribute_optical_depth(0 * ext_dry, 0 * ext_h2o, dz, ils, airmass, 0 * ray, p_lay)
    assert all(np.isnan(v) for v in out)


def test_attribution_matches_linear_split_at_small_tau():
    """For optically thin channels the attribution must reduce to the plain
    ratio of the ILS-weighted component optical depths."""
    ils, p_lay, dz, ext_dry, ext_h2o, ray = _synthetic_window()
    scale = 1e-3
    airmass = 1.0
    f_h2o, f_ray, _ = attribute_optical_depth(
        scale * (ext_dry + ext_h2o), scale * ext_h2o, dz, ils, airmass, scale * ray, p_lay)
    tau_h2o = scale * (ext_h2o @ dz)
    tau_tot = scale * ((ext_dry + ext_h2o) @ dz + ray)
    expected = float(np.dot(ils, tau_h2o) / np.dot(ils, tau_tot))
    assert abs(f_h2o - expected) < 1e-3 * expected
    # and the effective tau itself is what the production convolution gives
    tau_eff = convolve_optical_depth_in_transmission(
        scale * (ext_dry + ext_h2o), dz, ils, airmass, rayleigh_tau_window=scale * ray)
    assert abs(tau_eff - float(np.dot(ils, tau_tot) / ils.sum())) < 1e-6


def test_tau_file_currency_is_variant_specific(tmp_path):
    n, nchan = 3, 1016
    datasets = {name: np.zeros((n, nchan)) for name in fp_atm._TAU_DATASETS}
    for name in ("sza", "vza", "sounding_id", "fp_number"):
        datasets[name] = np.zeros(n)
    prod = tmp_path / "prod.h5"
    fp_atm._write_tau_file_atomic(str(prod), datasets, use_ring=False, telluric_doppler=False)
    assert fp_atm._tau_file_is_current(str(prod))
    assert not fp_atm._tau_file_is_current(str(prod), telluric_doppler=True)

    dop = tmp_path / "dop.h5"
    comp = {name: np.zeros((n, nchan), dtype=np.float32) for name in fp_atm._COMPONENT_DATASETS}
    fp_atm._write_tau_file_atomic(str(dop), {**datasets, **comp}, use_ring=False, telluric_doppler=True)
    assert fp_atm._tau_file_is_current(str(dop), telluric_doppler=True)
    assert not fp_atm._tau_file_is_current(str(dop))
    with h5py.File(dop, "r") as f:
        assert f.attrs["tau_convolution"] == TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER
        assert bool(f.attrs["telluric_doppler"]) is True

    # a Doppler-stamped file without the component datasets is not current
    bare = tmp_path / "bare.h5"
    fp_atm._write_tau_file_atomic(str(bare), datasets, use_ring=False, telluric_doppler=True)
    assert not fp_atm._tau_file_is_current(str(bare), telluric_doppler=True)


def test_extra_fit_chunk_recovers_exact_quadratic():
    rng = np.random.default_rng(1)
    m, nchan = 4, 300
    k1, k2, c = 0.97, 0.08, -2.5
    tau = np.sort(rng.uniform(0.01, 5.0, size=(3, m, nchan)), axis=-1)
    ln_t = c - k1 * tau + 0.5 * k2 * tau**2
    ln_t[0, 0, :5] = np.nan                        # a few invalid channels
    j_idx, extras = _extra_fit_chunk(np.arange(m), tau, ln_t, (7, 3, 7))
    assert extras.shape == (3, m, len(EXTRA_FIT_FIELDS))
    idx = {n: i for i, n in enumerate(EXTRA_FIT_FIELDS)}
    assert np.allclose(extras[:, :, idx["k1_o2"]], k1, atol=1e-8)
    assert np.allclose(extras[:, :, idx["k2_o2"]], k2, atol=1e-8)
    assert np.allclose(extras[:, :, idx["k1_o3"]], k1, atol=1e-6)
    assert np.allclose(extras[:, :, idx["k3_o3"]], 0.0, atol=1e-6)
    assert np.all(extras[:, :, idx["rms_o2"]] < 1e-8)
    assert np.all(extras[:, :, idx["rms_prod"]] < 1e-6)
    assert extras[0, 0, idx["nchan"]] == nchan - 5
    assert np.all(extras[1:, :, idx["nchan"]] == nchan)
    # standard errors scale with the residual: add noise and they become positive
    ln_noisy = ln_t + rng.normal(0, 0.01, ln_t.shape)
    _, ex2 = _extra_fit_chunk(np.arange(m), tau, ln_noisy, (7, 3, 7))
    assert np.all(ex2[:, :, idx["sig_k1_o2"]] > 0)
    assert np.all(ex2[:, :, idx["sig_k2_o3"]] > ex2[:, :, idx["sig_k2_o2"]])


def test_design_matrix_sign_convention():
    tau = np.array([0.5, 1.0])
    A = get_design_matrix(tau, 3)
    assert np.allclose(A[:, 1], -tau)
    assert np.allclose(A[:, 2], 0.5 * tau**2)
    assert np.allclose(A[:, 3], -tau**3 / 3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
