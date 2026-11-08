"""t2s test on the production fit example 2020-01-01 / 29252a_GL (2026-11-02).

t2s ("two-tau order-2 with slope", spectral.cumulant_fit.fit_t2s / t2s_fit_chunk):
    ln T = c + a1·x (+ a2·x² over land, SCO2) − k1·τ_dry − k1w·τ_h2o + ½k2·τ²
one linear design per band, OLS, no bounds, no floor, formal 1σ (_lin_se).

Part A, sounding 2020010100214501: the inputs are staged by the Appendix A
figure loader (make_appendix_a_fit_figure._load_fit_data, imported, not
copied), which also re-verifies the agreed-model (am) refit against the stored
v2 fitting_details.  The staged slabs are captured by wrapping the loader's
_doppler_fit_chunk and passed unchanged to t2s_fit_chunk.

Part B, whole granule (fp_tau_combined.h5, every sounding with an L1B match):
staged with the production loader spectral.orbit_data.load_orbit_data and the
transmittance / ln T / [1:-1] steps of fitting.process_orbit; the am values
are the stored fitting_details ones (a 1-in-25 subset is refitted with
_doppler_fit_chunk to check the staging).

Usage:
    PYTHONPATH=src:workspace python idea_test/fit_order_study/t2s_fit_example.py
Output: idea_test/fit_order_study/results/t2s_fit_example.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "workspace", REPO / "manuscript" / "JQSRT_draft" / "scripts"):
    sys.path.insert(0, str(p))

import make_appendix_a_fit_figure as figa  # noqa: E402
from spectral.cumulant_fit import (  # noqa: E402
    FIT_FIELDS, T2S_FIELDS, _doppler_fit_chunk, compute_transmittance, t2s_fit_chunk)
from spectral.orbit_data import load_orbit_data  # noqa: E402

BANDS = ("o2a", "wco2", "sco2")
LITE_FILE = REPO / "data" / "OCO2" / "2020" / "001" / "oco2_LtCO2_200101_B11210Ar_240913160328s.nc4"
GRANULE_TAU = figa.FIT_EXAMPLE_DIR / "fp_tau_combined.h5"
OUT_JSON = Path(__file__).resolve().parent / "results" / "t2s_fit_example.json"
STUBS = ("k1", "k1w", "k2", "a1", "a2", "c", "rms")
AM_STUBS = ("k1", "k1w", "k2", "k2diag", "f", "a1", "a2", "c", "exp_int", "rms")


def part_a() -> dict:
    captured = {}

    def capture(j, tau, lnT, frac, land):
        captured.update(j=j, tau=tau, lnT=lnT, frac=frac, land=land)
        return _doppler_fit_chunk(j, tau, lnT, frac, land)

    figa._doppler_fit_chunk = capture          # loader stages and verifies am
    bands, meta = figa._load_fit_data(figa.DEFAULT_SOUNDING_ID, figa.TAU_FILE,
                                      figa.FIT_FILE, figa.L1B_FILE)
    _, fits = t2s_fit_chunk(captured["j"], captured["tau"], captured["lnT"],
                            captured["frac"], captured["land"])
    out = {"sounding_id": figa.DEFAULT_SOUNDING_ID, "land": meta["land"],
           "land_fraction": meta["land_fraction"]}
    print(f"\n== Part A: sounding {figa.DEFAULT_SOUNDING_ID} "
          f"(land fraction {meta['land_fraction']:g} %, {'land' if meta['land'] else 'water'})")
    for i, tag in enumerate(BANDS):
        t2s = {n: float(fits[i, 0, k]) for k, n in enumerate(T2S_FIELDS)}
        am = {s: float(bands[i]["parameters"][f"{s}_am"]) for s in AM_STUBS}
        am.update({f"sig_{s}": float(bands[i]["parameters"][f"sig_{s}_am"]) for s in AM_STUBS
                   if f"sig_{s}_am" in bands[i]["parameters"]})
        out[tag] = {"t2s": t2s, "am": am}
        print(f"  {tag} (nchan {t2s['nchan']:.0f})")
        for s in STUBS:
            v, e = t2s[f"{s}_t2s"], t2s.get(f"sig_{s}_t2s", np.nan)
            print(f"    {s:4s} t2s {v:+.5f} ± {e:.1e}   am {am.get(s, np.nan):+.5f}")
        print(f"    am extras: k2diag {am['k2diag']:+.5f}  f {am['f']:.4f}  "
              f"gamma {am['exp_int']:.4f}")
    return out


def stage_granule():
    """Production staging (orbit_data.load_orbit_data + fitting.process_orbit step 2)."""
    l1b = Path(figa.L1B_FILE)
    sat = {"oco_lite": str(LITE_FILE),
           figa.GRANULE: {"fp_tau_file": str(GRANULE_TAU), "oco_l1b": str(l1b)}}
    od = load_orbit_data(sat, figa.GRANULE)
    T_all = compute_transmittance(od["radiances"], od["toa_sol"])
    ln_T_all = np.where(T_all > 0, np.log(T_all), np.nan)
    return od, ln_T_all


def part_b() -> dict:
    od, ln_T_all = stage_granule()
    valid = np.flatnonzero(od["valid_l1b"])
    args = lambda c: (c, od["tau"][:, c, 1:-1], ln_T_all[:, c, 1:-1],   # noqa: E731
                      od["frac_h2o"][:, c, 1:-1], od["land"][c])
    N = len(od["sounding_id"])
    t2s = np.full((3, N, len(T2S_FIELDS)), np.nan)
    for c in np.array_split(valid, max(1, len(valid) // 256)):
        jj, fc = t2s_fit_chunk(*args(c))
        t2s[:, jj] = fc

    t2s_nosplit = np.full((N, len(T2S_FIELDS)), np.nan)
    for c in np.array_split(valid, max(1, len(valid) // 256)):
        a = args(c)
        jj, fc = t2s_fit_chunk(a[0], a[1], a[2], np.zeros_like(a[3]), a[4])
        t2s_nosplit[jj] = fc[0]

    with h5py.File(figa.FIT_FILE, "r") as f:
        fp_id = f["fp_id"][:].astype(np.int64)
        assert np.array_equal(fp_id, od["sounding_id"]), "fitting_details row order"
        am = {(b, s): f[f"{b}_{s}_am"][:].astype(np.float64)
              for b in BANDS for s in ("k1", "k1w", "k2", "k2diag", "rms", "ok")}
        cld = f["cld_dist_km"][:].astype(np.float64)

    # staging check: refit am on every 25th sounding, compare with stored
    sub = valid[::25]
    _, fsub = _doppler_fit_chunk(*args(sub))
    worst = 0.0
    for i, b in enumerate(BANDS):
        for s in ("k1", "k2diag", "rms"):
            new = fsub[i, :, FIT_FIELDS.index(f"{s}_am")]
            old = am[(b, s)][sub]
            ok = np.isfinite(new) & np.isfinite(old)
            assert np.allclose(new[ok], old[ok], rtol=figa.VERIFY_RTOL, atol=figa.VERIFY_ATOL), (b, s)
            worst = max(worst, float(np.max(np.abs(new[ok] - old[ok]) / np.maximum(np.abs(old[ok]), 1e-30))))
    print(f"\n== Part B: granule {figa.GRANULE}: {N} soundings, {len(valid)} with L1B; "
          f"am refit of {len(sub)} soundings matches stored (max rel diff {worst:.1e})")

    land = od["land"]
    out = {"n_soundings": int(N), "n_l1b": int(len(valid)),
           "land_fraction_ge80": float(land[valid].mean()), "n_land": int(land[valid].sum()),
           "am_refit_check": {"n": int(len(sub)), "max_rel_diff": worst}, "bands": {}}
    print(f"  land (land_fraction >= 80): {land[valid].sum()} of {len(valid)} "
          f"= {100 * land[valid].mean():.1f} %")
    q = lambda v: [float(np.nanpercentile(v, p)) for p in (25, 50, 75)]   # noqa: E731
    F = T2S_FIELDS.index
    for i, b in enumerate(BANDS):
        k1, k1w, k2, rms = (t2s[i, valid, F(f"{s}_t2s")] for s in ("k1", "k1w", "k2", "rms"))
        ok = np.isfinite(k1) & (am[(b, "ok")][valid] == 1) & np.isfinite(am[(b, "k1")][valid])
        r = float(np.corrcoef(k1[ok], am[(b, "k1")][valid][ok])[0, 1])
        rms_am = am[(b, "rms")][valid]
        lv, far = land[valid], cld[valid] > 20
        d = {"n": int(ok.sum()),
             "k1_q": q(k1[ok]), "k1w_q": q(k1w[ok]), "k2_q": q(k2[ok]),
             "frac_k2_neg": float(np.mean(k2[ok] < 0)),
             "frac_k2_neg_land": float(np.mean(k2[ok & lv] < 0)) if (ok & lv).any() else None,
             "frac_k2_neg_water": float(np.mean(k2[ok & ~lv] < 0)) if (ok & ~lv).any() else None,
             "frac_k2_neg_far20": float(np.mean(k2[ok & far] < 0)) if (ok & far).any() else None,
             "rms_t2s_med": float(np.median(rms[ok])), "rms_am_med": float(np.median(rms_am[ok])),
             "rms_ratio_med": float(np.median(rms[ok] / rms_am[ok])),
             "r_k1_t2s_am": r,
             "am_k1_q": q(am[(b, "k1")][valid][ok]), "am_k2diag_q": q(am[(b, "k2diag")][valid][ok]),
             "frac_am_k2diag_neg": float(np.mean(am[(b, "k2diag")][valid][ok] < 0)),
             "median_k1_diff_t2s_minus_am": float(np.median(k1[ok] - am[(b, "k1")][valid][ok]))}
        if b == "o2a":
            # side variant: O2A with tau_h2o = 0 (single tau, as fit_am_o2a), to see
            # whether the ill-constrained k1w distorts k1 / k2
            k1s, k2s, rmss = (t2s_nosplit[valid, F(f"{s}_t2s")] for s in ("k1", "k2", "rms"))
            d["no_h2o_split"] = {
                "k1_q": q(k1s[ok]), "k2_q": q(k2s[ok]), "frac_k2_neg": float(np.mean(k2s[ok] < 0)),
                "rms_t2s_med": float(np.median(rmss[ok])),
                "r_k1_t2s_am": float(np.corrcoef(k1s[ok], am[(b, "k1")][valid][ok])[0, 1])}
        out["bands"][b] = d
        fmt = lambda v: f"{v[1]:+.4f} [{v[0]:+.4f}, {v[2]:+.4f}]"   # noqa: E731
        print(f"  {b}: n {d['n']}  k1 {fmt(d['k1_q'])}  k1w {fmt(d['k1w_q'])}  k2 {fmt(d['k2_q'])}")
        print(f"        k2<0 {100 * d['frac_k2_neg']:.1f} % (land {d['frac_k2_neg_land']}, "
              f"water {d['frac_k2_neg_water']}, cld>20 km {d['frac_k2_neg_far20']})")
        print(f"        rms t2s {d['rms_t2s_med']:.5f} vs am {d['rms_am_med']:.5f} "
              f"(median ratio {d['rms_ratio_med']:.3f});  r(k1 t2s, am) {r:.4f};  "
              f"am k1 {fmt(d['am_k1_q'])}, am k2diag {fmt(d['am_k2diag_q'])} "
              f"(k2diag<0 {100 * d['frac_am_k2diag_neg']:.1f} %)")
        if "no_h2o_split" in d:
            e = d["no_h2o_split"]
            print(f"        O2A without H2O split: k1 {fmt(e['k1_q'])}  k2 {fmt(e['k2_q'])}  "
                  f"k2<0 {100 * e['frac_k2_neg']:.1f} %  rms {e['rms_t2s_med']:.5f}  "
                  f"r(k1, am) {e['r_k1_t2s_am']:.4f}")
    return out


def main() -> None:
    res = {"model": "t2s", "part_a": part_a(), "part_b": part_b()}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
