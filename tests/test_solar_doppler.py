"""Solar Doppler shift of the TOA irradiance in the telluric-Doppler variant.

Pure tests (synthetic ILS and solar spectrum, no data files) of
fp_abs_coeff.toa_sol_track / solar_doppler_beta and the v4 variant stamp.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from abs_util.abs.oco_convolve_absco import oco_conv_absco  # noqa: E402
from abs_util.fp_abs_coeff import C_LIGHT, solar_doppler_beta, toa_sol_track  # noqa: E402
from abs_util.ils_tau import (  # noqa: E402
    TAU_CONVOLUTION_VERSION, TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER,
    TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER_V3, tau_convolution_version)

LAM0 = 1.6                      # um, line centre in the solar rest frame
DLAM_CH = 1.0e-6                # um, channel spacing (1 pm) = grid resolution of the test
SIGMA_ILS = 1.0e-5              # um (10 pm)
SIGMA_LINE = 5.0e-6             # um (5 pm)
V_SOLAR, V_INST = 500.0, 5000.0   # m/s


def _synthetic(sigma_ils=SIGMA_ILS, sigma_line=SIGMA_LINE, depth=0.5, dlam_ch=DLAM_CH,
               n_ch=201, nu_step=1.0e-3):
    """float32 ILS like the L1B one, float64 solar model on an increasing nu grid."""
    wloco = (LAM0 + dlam_ch * (np.arange(n_ch) - n_ch // 2)).astype(np.float32)
    x = np.linspace(-4 * sigma_ils, 4 * sigma_ils, 200)
    xx = np.tile(x, (n_ch, 1)).astype(np.float32)
    yy = np.tile(np.exp(-0.5 * (x / sigma_ils) ** 2), (n_ch, 1)).astype(np.float32)
    span = dlam_ch * n_ch + 20 * sigma_ils + 0.05 * LAM0 * 1e-3
    nu = np.arange(1e4 / (LAM0 + span), 1e4 / (LAM0 - span), nu_step)
    lam = 1e4 / nu
    sol_abs = 1.0 - depth * np.exp(-0.5 * ((lam - LAM0) / sigma_line) ** 2)
    sol_cont = np.full_like(nu, 2.0)
    return wloco, xx, yy, nu, sol_abs, sol_cont


def _toa(v_solar, v_inst, telluric_doppler, syn=None, dist_au=1.01, musolzen=1.2):
    wloco, xx, yy, nu, sol_abs, sol_cont = syn if syn is not None else _synthetic()
    return toa_sol_track(wloco, xx, yy, nu, sol_abs, sol_cont,
                         v_solar, v_inst, dist_au, musolzen, telluric_doppler)


def _toa_pre_change(wloco, xx, yy, sol_abs_nu, sol_abs_val, sol_cont_val,
                    v_solar, dist_au, musolzen):
    """Verbatim copy of the inline TOA code of _process_track_all_bands at e8ebba0."""
    fsol = (sol_abs_val * sol_cont_val * (1.0 / dist_au) ** 2)
    beta_total   = (v_solar) / C_LIGHT
    wvlsol_ori = 1e4/sol_abs_nu
    wvlsol_obs = wvlsol_ori * (1.0 + beta_total)
    fsol_obs = fsol / (1.0 + beta_total)
    indlr_sol  = oco_conv_absco(wloco, xx, yy, wvlsol_obs, len(wvlsol_obs))
    xx_wloco    = xx + wloco[:, np.newaxis]
    nx = len(wloco)
    toa_sol     = np.empty(nx)
    for ind in range(nx):
        start_sol, end_sol = indlr_sol[ind, 1], indlr_sol[ind, 0] + 1
        ilg0_sol     = np.interp(wvlsol_obs[start_sol:end_sol], xx_wloco[ind], yy[ind])
        ilg0_sol_sum = np.sum(ilg0_sol)
        toa_sol[ind] = np.dot(fsol_obs[start_sol:end_sol], ilg0_sol) / ilg0_sol_sum * musolzen
    return toa_sol


def _line_position(toa, wloco):
    return float(wloco[int(np.argmin(toa))])


@pytest.mark.parametrize("telluric_doppler, beta", [
    (True, (V_SOLAR - V_INST) / C_LIGHT),
    (False, V_SOLAR / C_LIGHT),
])
def test_convolved_line_lands_at_doppler_shifted_position(telluric_doppler, beta):
    syn = _synthetic()
    toa = _toa(V_SOLAR, V_INST, telluric_doppler, syn=syn)
    expected = LAM0 * (1.0 + beta)
    # float32 channel grid: allow half a channel plus float32 rounding at 1.6 um
    tol = 0.5 * DLAM_CH + 2 * np.spacing(np.float32(LAM0))
    assert abs(_line_position(toa, syn[0]) - expected) <= tol
    # and the two variants are clearly distinguishable (-24 pm vs +2.7 pm)
    other = _toa(V_SOLAR, V_INST, not telluric_doppler, syn=syn)
    assert abs(_line_position(toa, syn[0]) - _line_position(other, syn[0])) > 20 * DLAM_CH


def test_beta_sign_matches_telluric_term():
    # telluric grid: lambda_obs = lambda_rest * (1 - v_inst/c); the solar grid
    # gets the same -v_inst/c on top of +v_solar/c
    assert solar_doppler_beta(V_SOLAR, V_INST, True) == (V_SOLAR - V_INST) / C_LIGHT
    assert solar_doppler_beta(V_SOLAR, V_INST, False) == V_SOLAR / C_LIGHT
    assert solar_doppler_beta(0.0, V_INST, True) == pytest.approx(-V_INST / C_LIGHT)


def test_production_path_reproduces_pre_change_formula_exactly():
    wloco, xx, yy, nu, sol_abs, sol_cont = syn = _synthetic()
    for v_solar, v_inst, dist_au, musolzen in [(V_SOLAR, V_INST, 1.01, 1.2),
                                               (-312.25, -4870.5, 0.987, 2.7),
                                               (float(np.float32(431.7)), 0.0, 1.0, 1.0)]:
        ref = _toa_pre_change(wloco, xx, yy, nu, sol_abs, sol_cont, v_solar, dist_au, musolzen)
        new = _toa(v_solar, v_inst, False, syn=syn, dist_au=dist_au, musolzen=musolzen)
        np.testing.assert_array_equal(new, ref)
        # v_inst must not enter the production path at all
        np.testing.assert_array_equal(
            _toa(v_solar, 0.0, False, syn=syn, dist_au=dist_au, musolzen=musolzen), ref)
    # telluric_doppler=True with v_inst=0 is the same computation as production
    np.testing.assert_array_equal(_toa(V_SOLAR, 0.0, True, syn=syn),
                                  _toa(V_SOLAR, 0.0, False, syn=syn))


def test_nadir_like_velocity_effect_is_small_and_linear():
    # The handoff expected a nadir-like v_inst = 50 m/s to change ln(TOA) by
    # < 1e-4.  That holds only for shallow solar lines: on real orbits the
    # change is linear in v_inst, ~0.042 / 0.028 / 0.016 per km/s (O2A / WCO2 /
    # SCO2, row max), i.e. up to 2.8e-3 on nadir orbit 31316a_ND
    # (|v_inst| <= 65 m/s, 2026-10-30).  So test (i) the < 1e-4 bound for a
    # shallow (2 %) line, (ii) linearity in v_inst for a deeper OCO-like line,
    # (iii) the continuum change = v/c (1/(1+beta) flux factor).
    shallow = _synthetic(sigma_ils=34e-6, sigma_line=10e-6, depth=0.02, dlam_ch=31e-6, n_ch=101)
    d_shallow = np.abs(np.log(_toa(V_SOLAR, 50.0, True, syn=shallow))
                       - np.log(_toa(V_SOLAR, 0.0, True, syn=shallow)))
    assert d_shallow.max() < 1e-4

    deep = _synthetic(sigma_ils=34e-6, sigma_line=10e-6, depth=0.2, dlam_ch=31e-6, n_ch=101)
    base = np.log(_toa(V_SOLAR, 0.0, True, syn=deep))
    d50 = np.log(_toa(V_SOLAR, 50.0, True, syn=deep)) - base
    d500 = np.log(_toa(V_SOLAR, 500.0, True, syn=deep)) - base
    assert np.abs(d50).max() < 1e-3
    # first order: the largest change scales with v (channels at the line
    # centre are second order, so compare the row maxima as on real orbits)
    assert np.abs(d500).max() == pytest.approx(10.0 * np.abs(d50).max(), rel=0.05)
    # far continuum channels: only the bin-stretch factor, ln(1+beta) ~ -beta
    beta_diff = (V_SOLAR - 50.0) / C_LIGHT - V_SOLAR / C_LIGHT
    np.testing.assert_allclose(d50[[0, -1]], -beta_diff, rtol=1e-3)


def test_variant_stamps():
    assert TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER == "transmission_ils_rayleigh_hr_v4_telluric_solar_doppler"
    assert TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER_V3 == "transmission_ils_rayleigh_hr_v3_telluric_doppler"
    assert len({TAU_CONVOLUTION_VERSION, TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER,
                TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER_V3}) == 3
    assert tau_convolution_version(True) == TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER
    assert tau_convolution_version(False) == TAU_CONVOLUTION_VERSION == "transmission_ils_rayleigh_hr_v2"
